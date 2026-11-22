"""Hard three-layer contrast constraint for the normal branch (from the
tools/probe_hard_depth.py, experiment docs/硬分层约束试验_20261107.md), with test-time depth.

Regions whose depth lies within 2.5 cm of an LCD plane (and that carry target texture) must
show more target-aligned contrast on that plane's pattern than on each other pattern by a
margin. After the usual refinement, 36 extra steps alternate Adam on the image loss with
projections onto those half-spaces and the [0,1] box; the RGB8 checkpoint that satisfies the
margin with the best image PSNR is kept only if it loses at most `floor_drop` dB. At test time
depth comes from the two input images (stereo SGBM, depth_constraints.stereo_depth); any
failure or absence of near-plane content leaves the refined result unchanged.
"""
from pathlib import Path
import sys

import numpy as np
import torch

from .core import image_loss, scores
from .depth_constraints import PhaseAllocation
from .depth_eval import depth_regions
from .exam_geometry import LAYER_X_M

NAMES = ('near', 'middle', 'far')


def ensure_cv2():
    try:
        import cv2  # noqa: F401
    except ImportError:
        root = Path(__file__).resolve().parents[1]
        for candidate in (root/'python_packages_cv', root.parents[1]/'python_packages_cv'):
            if candidate.is_dir() and str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
        import cv2  # noqa: F401


def estimate_depth(left_rgb, right_rgb, poses):
    """Per-eye axial depth in metres (0 where stereo matching is not trusted)."""
    ensure_cv2()
    from .depth_constraints import stereo_depth
    return stereo_depth(np.stack([left_rgb, right_rgb]), np.asarray(poses, dtype=np.float64))


class LayerInequalities:
    """Each row maps three phase patterns to a matched-layer contrast gap."""

    def __init__(self, allocation, world_x, reference, stride=3):
        grid = allocation.grid
        target = allocation.principal[..., ::stride, ::stride]
        screen_layers = reference.new_zeros(reference.shape, requires_grad=True)
        sampled = allocation.sample(screen_layers, grid)
        rows, labels, support = [], [], {}
        for layer, position in enumerate(LAYER_X_M):
            roi = allocation.trusted & ((world_x-position).abs() <= .025)
            region = roi[:, None, ::stride, ::stride]
            edges = []
            for axis in (-1, -2):
                a, b = allocation.endpoints(region, axis)
                edges.append((axis, torch.diff(target, dim=axis)*(a & b)))
            energy = sum(edge.square().sum() for _, edge in edges)
            support[NAMES[layer]] = {'roi_fraction': float(roi.float().mean()), 'edge_energy': float(energy)}
            if int(region.sum()) < 64 or float(energy) < .01:
                continue
            score = sum((torch.diff(sampled, dim=axis)*edge[None]).sum() for axis, edge in edges)/energy
            gain, = torch.autograd.grad(score, screen_layers, retain_graph=True)
            for other in range(3):
                if other == layer:
                    continue
                row = torch.zeros_like(gain)
                row[layer], row[other] = gain[layer], -gain[other]
                rows.append(row.flatten())
                labels.append(f'{NAMES[layer]}:L{layer+1}-L{other+1}')
        self.labels, self.support = labels, support
        if rows:
            self.rows = torch.stack(rows)
            self.norm2 = self.rows.square().sum(1).clamp_min(1e-12)

    def gaps(self, patterns):
        return self.rows @ patterns.float().flatten()

    @torch.no_grad()
    def project(self, patterns, margin, rounds=3):
        """Cyclic projections onto each half-space and the physical box."""
        flat = patterns.flatten()
        for _ in range(rounds):
            for row, norm in zip(self.rows, self.norm2):
                correction = (margin-torch.dot(row, flat)).clamp_min(0)/norm
                flat.addcmul_(row, correction)
            flat.clamp_(0, 1)


def constrained_tail(renderer, target, initial, constraints, margin, floor):
    parameter = initial.float().contiguous().clone().requires_grad_(True)
    iteration = renderer.iteration_copy(torch.float32)
    optimizer = torch.optim.Adam([parameter], lr=.003)
    best, best_psnr, best_step = None, float('-inf'), None
    history = []
    slack = .003
    for step in range(37):
        if step:
            optimizer.param_groups[0]['lr'] = .003 if step <= 18 else .0008
            optimizer.zero_grad(set_to_none=True)
            loss = image_loss(iteration(parameter), target, iteration.geometry_mask)
            loss.backward()
            optimizer.step()
        constraints.project(parameter, margin+slack, rounds=6 if step == 0 else 2)
        if step % 6 == 0:
            with torch.no_grad():
                rgb8 = renderer.quantize_patterns(parameter)
                gap = float(constraints.gaps(rgb8).min())
                psnr = scores(renderer, rgb8, target)['psnr_mean_eyes']
            history.append({'step': step, 'gap': gap, 'psnr_dB': psnr})
            if gap < margin:
                slack = min(.03, slack+margin-gap+.001)
            if gap >= margin and psnr > best_psnr:
                best, best_psnr, best_step = rgb8, psnr, step
    return best if best_psnr >= floor else None, {
        'best_margin_feasible_psnr_dB': best_psnr if best is not None else None,
        'selected_step': best_step if best_psnr >= floor else None, 'history': history}


def apply(renderer, target, result, depths, poses, margin=.10, floor_drop=.25):
    """Returns (patterns, report); patterns is `result` unless a feasible constrained one exists."""
    depths = [np.asarray(d, dtype=np.float32) for d in depths]
    alphas = [((d > 0)*255).astype(np.uint8) for d in depths]
    data = [depth_regions(d, a, p) for d, a, p in zip(depths, alphas, poses)]
    device = result.device
    world_x = torch.as_tensor(np.stack([x for _, x in data]), device=device).float()
    trusted = torch.as_tensor(np.stack([(a >= 128)&np.isfinite(d)&(d > 0) for d, a in zip(depths, alphas)]), device=device)
    allocation = PhaseAllocation(renderer, target.float(), world_x, trusted)
    constraints = LayerInequalities(allocation, world_x, result.float())  # grids are float32
    report = {'support': constraints.support, 'labels': constraints.labels, 'margin': margin, 'applied': False}
    if not constraints.labels:
        report['reason'] = 'no textured content within 2.5 cm of an LCD plane'
        return result, report
    base = scores(renderer, result, target)['psnr_mean_eyes']
    report['base_psnr_dB'] = base
    report['base_min_gap'] = float(constraints.gaps(result).min())
    candidate, tail = constrained_tail(renderer, target, result, constraints, margin, base-floor_drop)
    report.update(tail)
    if candidate is None:
        report['reason'] = f'no RGB8 checkpoint met the margin within {floor_drop} dB'
        return result, report
    report['applied'] = True
    report['min_gap'] = float(constraints.gaps(candidate).min())
    return candidate, report
