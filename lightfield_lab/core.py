"""Cached EyeReal renderer, bounded pattern refinement, and PNG-aware metrics.

The physical model is inherited from WeijieMax/EyeReal (AGPL-3.0). Geometry is
constant during pattern optimization, so its sampling grids can be cached.
This implementation uses torchvision's private homography helpers deliberately;
the verification suite checks forward and gradient parity with the upstream code.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torchvision.transforms.functional import _get_perspective_coeffs
from torchvision.transforms._functional_tensor import _perspective_grid

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / 'EyeReal'
if str(UPSTREAM) not in sys.path:
    sys.path.insert(0, str(UPSTREAM))

from model.network import EyeRealNet
from data.dataset import EyeRealDataset, eye2world_pytroch
from config.scene_dict import scene_dict


def synchronize(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def make_model(height, width, device, checkpoint=None):
    args = SimpleNamespace(N_screen=3, image_height=height, image_width=width,
        aux_loss=False, l1_loss=False, l1_mutex=False, mutex=False, embed_dim=32)
    model = EyeRealNet(args, FOV=math.radians(40))
    if checkpoint is not None:
        # Published checkpoint contains its argparse configuration as metadata.
        # Permit that data-only class while keeping the weights-only loader.
        with torch.serialization.safe_globals([argparse.Namespace]):
            state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        model.load_state_dict(state['model'])
    return model.eval().requires_grad_(False).to(device)


def load_demo(height, width, device):
    """Reproduce inference_figures.py's view order and scene geometry."""
    paths = sorted((UPSTREAM / 'dataset/demo/lego_bulldozer').glob('*.jpg'),
                   key=lambda p: int(p.stem.rsplit('_', 1)[1]))
    if len(paths) != 2:
        raise ValueError('Expected the two upstream demo images.')
    images, views = [], []
    for path in paths:
        image = Image.open(path).convert('RGB').resize((width, height), Image.Resampling.BILINEAR)
        images.append(torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float()/255)
        parts = path.stem.split('_')
        position = torch.tensor([float(v[1:]) for v in parts[1:4]])
        views.append(eye2world_pytroch('z', position))
    config = scene_dict['lego_bulldozer']
    holder = SimpleNamespace(H=height, W=width, N_screen=3)
    screens = EyeRealDataset.get_screen_coords_world(holder,
        thickness=config['thickness'], scale_physical2world=config['scale_physical2world'],
        physical_width=config.get('physical_width'), ground=config.get('ground'),
        ground_coefficient=config.get('ground_coefficient'), orientation=config['orientation'],
        delta=torch.tensor([config.get('delta_x', 0.), config.get('delta_y', 0.), config.get('delta_z', 0.)]))
    return torch.stack(images).to(device), torch.stack(views).to(device), screens.to(device), {
        'source': 'upstream demo; scene geometry, not the camp 27-inch configuration',
        'view_files': [p.name for p in paths], 'scene_config': config}


def synthetic_geometry(height, width, device='cpu', oblique=False, profile='legacy'):
    """Centimetre geometry; legacy default preserves old experiments.

    New experiment CLIs explicitly default to profile='ta'.
    """
    if profile not in ('legacy', 'ta'):
        raise ValueError('Geometry profile must be legacy or ta.')
    from .exam_geometry import SCREEN_WIDTH_M
    width_cm = SCREEN_WIDTH_M*100 if profile == 'ta' else 27 * 2.54 * 16 / math.sqrt(16**2 + 9**2)
    height_cm = width_cm * 9 / 16
    screens = torch.tensor([[[x, -width_cm/2, height_cm/2], [x, width_cm/2, height_cm/2],
                              [x, -width_cm/2, -height_cm/2], [x, width_cm/2, -height_cm/2]]
                            for x in [6., 0., -6.]])
    positions = [[90., 22.-3., 12.], [90., 22.+3., 12.]] if oblique else [[90., -3., 0.], [90., 3., 0.]]
    views = torch.stack([eye2world_pytroch('z', torch.tensor(p)) for p in positions])
    return views.to(device), screens.to(device)


class CachedRenderer:
    """Single-sample, three-layer renderer with a batch of eye views.

    soft_mask reproduces the upstream interpolated masks. geometry_mask uses
    the common geometric footprint [0,W] x [0,H]. The official boundary rule
    differs numerically from the TA ray reference at native resolution.
    Use ExamRenderer for TA-facing prediction and scoring.
    """
    def __init__(self, views, screens, height, width, fov=math.radians(40)):
        self.device, self.height, self.width = views.device, height, width
        self.layers, self.eyes = screens.shape[0], views.shape[0]
        if self.layers != 3:
            raise ValueError('This prototype assumes exactly three LCD layers.')
        count = self.layers * self.eyes
        screen_points = screens[:, None].expand(-1, self.eyes, -1, -1).reshape(count, 4, 3)
        view_mats = views[None].expand(self.layers, -1, -1, -1).reshape(count, 4, 4)
        homogeneous = torch.cat([screen_points, torch.ones(count, 4, 1, device=self.device)], -1)
        camera = (torch.linalg.inv(view_mats)[:, None] @ homogeneous[..., None])[..., :3, 0]
        if not torch.isfinite(camera).all() or (camera[..., 2] >= -1e-6).any():
            raise ValueError('All screen corners must be in front of each eye camera; near-plane clipping is not implemented.')
        focal = width / (2 * math.tan(fov / 2))
        projected = torch.stack([-focal * camera[..., 0] / camera[..., 2] + width/2,
                                  focal * camera[..., 1] / camera[..., 2] + height/2], -1)
        corners = [[0., 0.], [float(width), 0.], [0., float(height)], [float(width), float(height)]]
        grids = []
        for destination in projected.cpu().tolist():
            coefficients = _get_perspective_coeffs(corners, destination)
            grids.append(_perspective_grid(coefficients, width, height, torch.float32, self.device)[0])
        self.grid = torch.stack(grids)
        ones = torch.ones(count, 1, height, width, device=self.device)
        footprints = F.grid_sample(ones, self.grid, align_corners=False)
        self.soft_mask = footprints.view(self.layers, self.eyes, 1, height, width).prod(0)
        geometric = (self.grid.abs() <= 1).all(-1)
        self.geometry_mask = geometric.view(self.layers, self.eyes, height, width).all(0)[:, None].float()
        if (self.geometry_mask.sum((1, 2, 3)) == 0).any():
            raise ValueError('Each eye must have a nonempty common LCD footprint.')

    def accumulate(self, patterns):
        if patterns.shape != (self.layers, 3, self.height, self.width):
            raise ValueError('Expected layer patterns with shape [3,3,H,W].')
        sources = patterns[:, None].expand(-1, self.eyes, -1, -1, -1).reshape(-1, 3, self.height, self.width)
        samples = F.grid_sample(sources, self.grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        return samples.view(self.layers, self.eyes, 3, self.height, self.width).sum(0)

    def __call__(self, patterns):
        return torch.sin(self.accumulate(patterns) * (math.pi/2)).square()


def png_values(patterns):
    """Exactly the upstream positive 8-bit round/clamp convention."""
    return torch.floor(patterns.detach().clamp(0, 1) * 255 + .5) / 255


def image_loss(prediction, target, mask):
    per_eye = ((prediction-target).square()*mask).sum((1, 2, 3))
    per_eye = per_eye / (mask.sum((1, 2, 3))*target.shape[1])
    return per_eye.mean()


@torch.no_grad()
def scores(renderer, patterns, target):
    prediction = renderer(patterns)
    mask = renderer.geometry_mask
    mse = ((prediction-target).square()*mask).sum((1, 2, 3))/(mask.sum((1, 2, 3))*3)
    psnr = -10*torch.log10(mse.clamp_min(1e-12))
    # Legacy EyeReal: soft masks multiply the error before squaring, then
    # get_PSNR normalizes by the (unsquared) mask area; views pooled together.
    legacy_mse = ((prediction-target).square()*renderer.soft_mask.square()).sum()/(renderer.soft_mask.sum()*3)
    return {'psnr_mean_eyes': float(psnr.mean()), 'psnr_each_eye': psnr.cpu().tolist(),
            'mse_each_eye': mse.cpu().tolist(), 'upstream_pooled_psnr': float(-10*legacy_mse.clamp_min(1e-12).log10())}


def refine(renderer, target, initial, steps=50, lr=.02, phase_steps=0, anchor_weight=0., eval_every=10,
           consistency_renderer=None, consistency_weight=0., deadline=None, iteration_dtype=None, lr_schedule=(), checkpoint_callback=None, highpass_weight=0., quantize_schedule=(),
           pattern_loss=None, checkpoint_rank=None):
    """Projected Adam; optional nearest periodic phase-branch warm start.

    For normalized phases s=sum(g_i), sin^2(pi*s/2)=I has branch candidates
    b, 2-b, 2+b in [0,3], b=2/pi*asin(sqrt(I)). Branch loss avoids zero
    intensity gradients at black/white extrema. It is a proposal to test,
    not a claim of convexity or a globally optimal solver.
    iteration_dtype selects a separate ExamRenderer sampling/parameter precision;
    None preserves the existing path. Quantized checkpoint scores use renderer.
    Optional pattern_loss adds a differentiable term. checkpoint_rank ranks RGB8
    candidates from their quantized patterns and mean-eye PSNR (higher is better).
    """
    if steps < 0 or not 0 <= phase_steps <= steps or eval_every < 1:
        raise ValueError('Invalid iteration settings.')
    if consistency_weight < 0 or (consistency_weight and consistency_renderer is None):
        raise ValueError('Positive consistency weight requires a renderer.')
    if iteration_dtype not in (None, torch.float32, torch.float64):
        raise ValueError('iteration_dtype must be float32 or float64.')
    iteration_renderer = renderer if iteration_dtype is None else renderer.iteration_copy(iteration_dtype)
    iteration_target = target if iteration_dtype is None else target.to(iteration_dtype)
    if highpass_weight < 0:
        raise ValueError('highpass_weight must be nonnegative.')
    freeze_at = dict(quantize_schedule)
    if any(not isinstance(step, int) or step < 1 or step > steps or layer not in (0, 1, 2) for step, layer in freeze_at.items()):
        raise ValueError('Quantization schedule requires valid steps and layer indices.')
    schedule = dict(lr_schedule)
    if any(not isinstance(k, int) or k < 1 or k > steps or not math.isfinite(v) or v <= 0 for k, v in schedule.items()):
        raise ValueError('Schedule entries require step in [1,steps] and positive finite lr.')
    parameter = initial.detach().to(iteration_dtype or initial.dtype).clamp(0, 1).clone().requires_grad_(True)
    anchor = parameter.detach().clone()
    # No ground-truth neighbouring images enter this regularizer. The teacher
    # is the frozen network initialization rendered at extra positions.
    with torch.no_grad():
        teacher = consistency_renderer(anchor) if consistency_weight else None
    frozen = {}
    optimizer = torch.optim.Adam([parameter], lr=lr)
    principal = 2/math.pi * torch.asin(iteration_target.clamp(0, 1).sqrt())
    branches = torch.stack([principal, 2-principal, 2+principal])
    quantize = getattr(renderer, 'quantize_patterns', png_values)
    best = quantize(parameter)
    best_score = scores(renderer, best, target)['psnr_mean_eyes']
    best_rank = checkpoint_rank(best, best_score) if checkpoint_rank else best_score
    best_step = 0
    history = [{'step': 0, 'png_psnr': best_score}]
    synchronize(renderer.device)
    started = time.perf_counter()
    steps_run = 0
    for step in range(1, steps+1):
        if deadline is not None and time.perf_counter() >= deadline:
            break
        if step in freeze_at:
            layer = freeze_at[step]
            frozen[layer] = quantize(parameter)[layer].to(parameter.dtype)
            with torch.no_grad():
                parameter[layer].copy_(frozen[layer])
        if step in schedule:
            for group in optimizer.param_groups:
                group['lr'] = schedule[step]
        optimizer.zero_grad(set_to_none=True)
        accumulated = iteration_renderer.accumulate(parameter)
        if step <= phase_steps:
            distances = (accumulated.detach()[None]-branches).abs()
            chosen = branches.gather(0, distances.argmin(0, keepdim=True))[0]
            loss = image_loss(accumulated, chosen, iteration_renderer.geometry_mask)
        else:
            loss = image_loss(torch.sin(accumulated*math.pi/2).square(), iteration_target, iteration_renderer.geometry_mask)
        if highpass_weight:
            loss = loss+highpass_weight*(parameter-F.avg_pool2d(parameter, 5, stride=1, padding=2)).square().mean()
        if anchor_weight:
            loss = loss + anchor_weight*(parameter-anchor).square().mean()
        if consistency_weight:
            loss = loss + consistency_weight*image_loss(consistency_renderer(parameter), teacher,
                                                        consistency_renderer.geometry_mask)
        if pattern_loss is not None:
            loss = loss + pattern_loss(parameter)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            parameter.clamp_(0, 1)
            for layer, value in frozen.items():
                parameter[layer].copy_(value)
        steps_run = step
        if step % eval_every == 0 or step == steps:
            candidate = quantize(parameter)
            candidate_score = scores(renderer, candidate, target)['psnr_mean_eyes']
            history.append({'step': step, 'png_psnr': candidate_score,
                            'objective': 'phase' if step <= phase_steps else 'intensity'})
            rank = checkpoint_rank(candidate, candidate_score) if checkpoint_rank else candidate_score
            if rank > best_rank:
                best, best_score, best_step = candidate.clone(), candidate_score, step
                best_rank = rank
            if checkpoint_callback is not None:
                checkpoint_callback(step, best, best_score)
    synchronize(renderer.device)
    elapsed = time.perf_counter()-started
    return best, {'steps_run': steps_run, 'steps_requested': steps, 'selected_step': best_step,
                  'stopped_for_time': steps_run < steps,
                  'iteration_dtype': ('legacy: parameters '+str(parameter.dtype)+', sampling '+str(iteration_renderer.grid.dtype)
                                      if iteration_dtype is None else str(iteration_dtype)),
                  'seconds_refine_and_checkpoint_scoring': elapsed, 'history': history}


def save_images(tensor, folder, stem):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for index, image in enumerate(tensor.detach().cpu()):
        array = torch.floor(image.clamp(0, 1)*255+.5).byte().permute(1, 2, 0).numpy()
        Image.fromarray(array).save(folder/f'{stem}-{index+1}.png')


def read_patterns(folder, dtype=torch.float32):
    paths = [Path(folder)/f'layer-{i+1}.png' for i in range(3)]
    arrays = [torch.from_numpy(np.asarray(Image.open(p).convert('RGB')).copy()).permute(2, 0, 1).to(dtype)/255 for p in paths]
    return torch.stack(arrays)
