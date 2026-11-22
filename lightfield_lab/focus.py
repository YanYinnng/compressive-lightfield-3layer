"""Fixed-pattern geometric-optics pupil integration, in physical centimetres.

Focal planes are perpendicular to the reference camera axis. At every focal
distance the same reference pixels and the same pupil samples are used. The
LCD evaluator samples the three fixed patterns, then applies sin², then pupil
integration; it neither re-solves patterns nor averages phases across rays.
"""
import math

import torch
from torch.nn import functional as F

from .scenes import camera_rays, render_scene_rays


def pupil_offsets(diameter_mm, rings, device='cpu'):
    """Equal-area annular quadrature: ring i has 6*(2*i+1) angular samples."""
    if diameter_mm < 0 or rings < 1:
        raise ValueError('Require nonnegative diameter and positive ring count.')
    if diameter_mm == 0:
        return torch.zeros(1, 2, device=device)
    points = []
    radius_cm = diameter_mm/20
    for ring in range(rings):
        radius = radius_cm*math.sqrt(((ring+.5)/rings)**2 + 1/(4*rings*rings))
        count = 6*(2*ring+1)
        for index in range(count):
            angle = 2*math.pi*(index+.5*(ring % 2))/count
            points.append((radius*math.cos(angle), radius*math.sin(angle)))
    return torch.tensor(points, device=device)


class FocusGeometry:
    def __init__(self, view, height, width, diameter_mm=4., rings=3, fov=math.radians(40)):
        self.view, self.height, self.width = view, height, width
        self.origin, self.direction = camera_rays(view[None], height, width, fov)
        self.offsets = pupil_offsets(diameter_mm, rings, view.device)
        self.displacements = self.offsets @ view[:3, :2].T
        self.origins = view[:3, 3]+self.displacements

    def rays(self, distance_cm, start=0, end=None):
        if distance_cm <= 0:
            raise ValueError('Axial eye-to-focus distance must be positive.')
        # direction has optical-axis component 1, not Euclidean unit length.
        displacement = self.displacements[start:end]
        origins = self.origins[start:end, None, None].expand(-1, self.height, self.width, -1)
        directions = self.direction-displacement[:, None, None]/distance_cm
        return origins, directions


def _lcd_grid(screens, origins, directions):
    right = screens[:, 1]-screens[:, 0]
    down = screens[:, 2]-screens[:, 0]
    normal = torch.linalg.cross(right, down, dim=-1)
    numerator = ((screens[None, :, None, None, 0]-origins[:, None])*normal[None, :, None, None]).sum(-1)
    denominator = (directions[:, None]*normal[None, :, None, None]).sum(-1)
    t = numerator/denominator
    relative = origins[:, None]+t[..., None]*directions[:, None]-screens[None, :, None, None, 0]
    u = (relative*right[None, :, None, None]).sum(-1)/right.square().sum(-1)[None, :, None, None]
    v = (relative*down[None, :, None, None]).sum(-1)/down.square().sum(-1)[None, :, None, None]
    grid = torch.stack([2*u-1, 2*v-1], -1)
    valid = ((u >= 0)&(u <= 1)&(v >= 0)&(v <= 1)&(t > 0)&torch.isfinite(t)).all(1)
    return grid, valid[:, None]


@torch.no_grad()
def lcd_rays(patterns, screens, origins, directions):
    """Independent ray/rectangle intersection; patterns [layers,3,H,W]."""
    grid, valid = _lcd_grid(screens, origins, directions)
    grid = grid.to(patterns.dtype)
    pupils, layers, height, width, _ = grid.shape
    samples = F.grid_sample(patterns[None].expand(pupils, -1, -1, -1, -1).reshape(-1, 3, *patterns.shape[-2:]),
                            grid.reshape(-1, height, width, 2), align_corners=False)
    intensity = torch.sin(samples.reshape(pupils, layers, 3, height, width).sum(1)*math.pi/2).square()
    return intensity, valid


@torch.no_grad()
def focal_stack_many(geometry, distances_cm, screens, patterns, chunk=3):
    """Evaluate methods [B,layers,3,H,W] with the exact same ray grids."""
    methods, layers, channels, source_h, source_w = patterns.shape
    height, width = geometry.height, geometry.width
    images, masks = [], []
    for distance in distances_cm:
        total = torch.zeros(methods, 3, height, width, device=patterns.device)
        valid = torch.ones(1, height, width, dtype=torch.bool, device=patterns.device)
        for start in range(0, len(geometry.origins), chunk):
            origins, directions = geometry.rays(float(distance), start, start+chunk)
            grid, ray_valid = _lcd_grid(screens, origins, directions)
            grid = grid.to(patterns.dtype)
            pupils = len(origins)
            # Pack pupil rows into the output image axis; no source replication.
            grid = grid.permute(1, 0, 2, 3, 4).reshape(layers, pupils*height, width, 2)
            grid = grid[None].expand(methods, -1, -1, -1, -1).reshape(methods*layers, pupils*height, width, 2)
            values = F.grid_sample(patterns.reshape(methods*layers, channels, source_h, source_w),
                                   grid, align_corners=False)
            phases = values.reshape(methods, layers, channels, pupils, height, width).sum(1)
            total += torch.sin(phases*math.pi/2).square().sum(2)
            valid &= ray_valid.all(0)
        images.append(total/len(geometry.origins))
        masks.append(valid)
    return torch.stack(images, 1), torch.stack(masks)


@torch.no_grad()
def focal_stack(geometry, distances_cm, screens=None, patterns=None, seed=None, chunk=6, texture_scale=1.):
    """Return [focus,3,H,W] images and [focus,1,H,W] all-pupil validity.

    With seed set, ray trace the independent 3D scene instead of LCD patterns.
    All quadrature samples have equal area weights by construction.
    """
    if (patterns is None) == (seed is None):
        raise ValueError('Supply exactly one of fixed patterns or a scene seed.')
    images, masks = [], []
    for distance in distances_cm:
        total = torch.zeros(3, geometry.height, geometry.width, device=geometry.view.device)
        valid = torch.ones(1, geometry.height, geometry.width, dtype=torch.bool, device=geometry.view.device)
        for start in range(0, len(geometry.origins), chunk):
            origins, directions = geometry.rays(float(distance), start, start+chunk)
            if patterns is None:
                values = render_scene_rays(origins, directions, seed, texture_scale=texture_scale)
            else:
                values, ray_valid = lcd_rays(patterns, screens, origins, directions)
                valid &= ray_valid.all(0)
            total += values.sum(0)
        images.append(total/len(geometry.origins))
        masks.append(valid)
    return torch.stack(images), torch.stack(masks)


def roi_masks(ids, valid, erosion=2):
    """Keep card interiors for peak diagnostics; background is reported too."""
    masks = {}
    for number, label in [(0, 'background'), (1, 'far'), (2, 'middle'), (3, 'near')]:
        mask = (ids == number).float()[None, None]
        if erosion:
            mask = 1-F.max_pool2d(1-mask, 2*erosion+1, stride=1, padding=erosion)
        mask = mask[0]*valid
        if mask.sum() >= 16:
            masks[label] = mask
    return masks


def focus_metrics(prediction, truth, sharp, valid, rois, distances_cm, signal_threshold=1/255):
    """Describe full-stack fidelity and per-ROI focus, never a hidden grade."""
    def mse_curve(a, b, mask):
        return ((a-b).square()*mask).sum((1, 2, 3))/(mask.sum()*3)
    def psnr(mse):
        return -10*mse.clamp_min(1e-12).log10()
    errors = mse_curve(prediction, truth, valid)
    result = {'stack_psnr_mean_dB': float(psnr(errors).mean()),
              'stack_mse_mean': float(errors.mean()), 'stack_psnr_by_focus_dB': psnr(errors).tolist(),
              'valid_fraction': float(valid.float().mean()), 'rois': {}}
    diopters = 100/torch.as_tensor(distances_cm, device=prediction.device)
    for label, mask in rois.items():
        curve = psnr(mse_curve(prediction, sharp, mask))
        reference = psnr(mse_curve(truth, sharp, mask))
        signal = (((truth-truth.mean(0)).square()*mask).sum()/(len(truth)*mask.sum()*3)).sqrt()
        truth_response = truth-truth.mean(0)
        output_response = prediction-prediction.mean(0)
        output_energy = (output_response.square()*mask).sum()
        truth_energy = (truth_response.square()*mask).sum()
        cross = (output_response*truth_response*mask).sum()
        pred_index, gt_index = int(curve.argmax()), int(reference.argmax())
        observable = bool(signal >= signal_threshold)
        result['rois'][label] = {
            'pixels': int(mask.sum()), 'curve_psnr_dB': curve.tolist(), 'gt_curve_psnr_dB': reference.tolist(),
            'gt_focus_signal_rms': float(signal), 'signal_threshold': signal_threshold,
            'peak_observable': observable,
            'peak_distance_cm': float(distances_cm[pred_index]) if observable else None,
            'gt_peak_distance_cm': float(distances_cm[gt_index]) if observable else None,
            'peak_error_diopters': float((diopters[pred_index]-diopters[gt_index]).abs()) if observable else None,
            'gt_peak_at_scan_boundary': gt_index in (0, len(truth)-1),
            'prediction_peak_at_scan_boundary': pred_index in (0, len(truth)-1),
            'stack_psnr_mean_dB': float(psnr(mse_curve(prediction, truth, mask)).mean()),
            'focus_response_nrmse': float((((output_response-truth_response).square()*mask).sum()/truth_energy).sqrt()) if observable else None,
            'focus_response_correlation': float(cross/(output_energy*truth_energy).sqrt()) if observable and output_energy > 1e-15 else None,
            'focus_response_gain': float(cross/truth_energy) if observable else None,
        }
    return result
