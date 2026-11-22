"""Project onto hard depth-contrast inequalities and verify the exported RGB8 result."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lightfield_lab.core import image_loss, refine, scores
from lightfield_lab.depth_constraints import PhaseAllocation
from lightfield_lab.depth_eval import depth_regions
from lightfield_lab.exam_data import read_manifest
from lightfield_lab.exam_geometry import ExamRenderer, HEIGHT, WIDTH, LAYER_X_M
from tools.compare_depth_constraints import contact_sheet
from tools.overnight_eval import Context, append, save_patterns, sync, to_images
from tools.refine_eye_neighborhood import centres

MARGINS = (.05, .10)
NAMES = ('near', 'middle', 'far')
IDS = {'lego': ('eyebox_train_000034', 'eyebox_train_000051', 'eyebox_val_000000', 'eyebox_test_000018'),
       'bookshelf': ('bookshelf_000000', 'bookshelf_000333', 'bookshelf_000666', 'bookshelf_000999')}


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
        'selected_step': best_step if best_psnr >= floor else None,
        'history': history}


def run_case(context, folder, row, source, cache, out):
    start = time.perf_counter()
    rgb, poses = centres(folder, row)
    target = to_images(rgb, context.device)
    renderer = ExamRenderer(poses, HEIGHT, WIDTH, context.device, geometry_backend='torch')
    initial = context.initial(target, poses)
    baseline, _ = refine(renderer, target, initial, steps=100, lr=.08, iteration_dtype=torch.float32)
    base_score = scores(renderer, baseline, target)['psnr_mean_eyes']
    if source == 'lego':
        arrays = [np.load(folder/row[key]) for key in ('depth_l', 'depth_r')]
        depths = [a['depth_m'].astype(np.float32) for a in arrays]
        alphas = [a['alpha_u8'] for a in arrays]
        for a in arrays:
            a.close()
    else:
        with np.load(cache/f"{row['id']}.npz") as arrays:
            depths = arrays['depth']
        alphas = [((d > 0)*255).astype(np.uint8) for d in depths]
    data = [depth_regions(d, a, p) for d, a, p in zip(depths, alphas, poses)]
    to_device = lambda a: torch.as_tensor(np.stack(a), device=context.device)
    world_x = to_device([x for _, x in data]).float()
    trusted = to_device([(a >= 128)&np.isfinite(d)&(d > 0) for d, a in zip(depths, alphas)])
    allocation = PhaseAllocation(renderer, target, world_x, trusted)
    constraints = LayerInequalities(allocation, world_x, initial)
    paths = {'adam100': save_patterns(out/'patterns'/row['id'], 'adam100', baseline, out)}
    record = {'id': row['id'], 'source': source, 'split': row.get('split', 'bookshelf'),
              'base_psnr_dB': base_score, 'support': constraints.support, 'labels': constraints.labels,
              'base_gaps': constraints.gaps(baseline).tolist() if constraints.labels else [],
              'experiments': {}, 'patterns': paths}
    if constraints.labels:
        control, trace = refine(renderer, target, baseline, steps=36, lr=.003, eval_every=6,
                                lr_schedule=((19, .0008),), iteration_dtype=torch.float32)
        paths['center_tail'] = save_patterns(out/'patterns'/row['id'], 'center_tail', control, out)
        record['center_tail'] = {'psnr_dB': scores(renderer, control, target)['psnr_mean_eyes'],
                                 'gaps': constraints.gaps(control).tolist(), 'step': trace['selected_step']}
    for margin in MARGINS if constraints.labels else ():
        name = f'margin_{margin:.2f}'
        candidate, report = constrained_tail(renderer, target, baseline, constraints, margin, base_score-.25)
        report['passes_both'] = candidate is not None
        report['baseline_passes'] = min(record['base_gaps']) >= margin
        if candidate is not None:
            paths[name] = save_patterns(out/'patterns'/row['id'], name, candidate, out)
            report['quantized_gaps'] = constraints.gaps(candidate).tolist()
        record['experiments'][name] = report
    contact_sheet(out, row['id'], paths)
    sync()
    record['seconds'] = time.perf_counter()-start
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', choices=IDS, required=True)
    parser.add_argument('--gt', type=Path, required=True)
    parser.add_argument('--depth-cache', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--weights', type=Path, default=ROOT/'weights/pretrained_model.pth')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest, rows = read_manifest(args.gt/'eyebox.json')
    rows = {r['id']: r for r in rows}
    context = Context(args)
    for identifier in IDS[args.source]:
        row = run_case(context, manifest.parent, rows[identifier], args.source, args.depth_cache, args.out)
        append(args.out/f'{args.source}_records.jsonl', row)
        print(identifier, row['base_psnr_dB'], row['base_gaps'],
              {k: (v['passes_both'], v['best_margin_feasible_psnr_dB'])
               for k, v in row['experiments'].items()}, flush=True)


if __name__ == '__main__':
    main()
