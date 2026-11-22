"""Depth proxies for fixed RGB phase patterns. These are not official scores."""
import numpy as np
import torch
from torch.nn import functional as F


def depth_regions(depth_m, alpha, pose, threshold_m=.03):
    """Expected axial depth -> world X relative to layer2, with foreground validity."""
    height, width = depth_m.shape
    focal = width/(2*np.tan(np.deg2rad(40.)/2))
    y, x = np.meshgrid(np.arange(height)+.5, np.arange(width)+.5, indexing='ij')
    p = np.asarray(pose)
    direction_x = p[0, 0]*(x-width/2)/focal-p[0, 1]*(y-height/2)/focal-p[0, 2]
    world_x = p[0, 3]+depth_m*direction_x
    valid = (alpha >= 128)&np.isfinite(depth_m)&(depth_m > 0)
    return {'near': valid&(world_x > threshold_m),
            'middle': valid&(np.abs(world_x) <= threshold_m),
            'far': valid&(world_x < -threshold_m)}, world_x


@torch.no_grad()
def layer_depth_matrix(patterns, renderer, rois, target):
    """Sample each LCD layer's high-pass energy along rays in each depth region.

    No diagonal-dominance score: additive phases have periodic ambiguities, so
    near content need not be localized to the nearest layer in an optimal solution.
    """
    patterns = patterns.double()
    high = patterns-F.avg_pool2d(patterns, 9, stride=1, padding=4)
    energies, correlations, pixels = {}, {}, {}
    target = target.double()
    for label, roi in rois.items():
        mask = roi[:, None]*renderer.geometry_mask
        count = int(mask.sum())
        pixels[label] = count
        if count < 16:
            energies[label] = correlations[label] = [None]*3
            continue
        weights = mask.expand(-1, 3, -1, -1)
        denominator = weights.sum()
        energy, correlation = [], []
        for layer in range(3):
            grids = renderer.grid.reshape(3, renderer.eyes, renderer.height, renderer.width, 2)[layer]
            value = F.grid_sample(patterns[layer:layer+1].expand(renderer.eyes, -1, -1, -1), grids, align_corners=False)
            residual = F.grid_sample(high[layer:layer+1].expand(renderer.eyes, -1, -1, -1), grids, align_corners=False)
            energy.append(float((residual.square()*weights).sum()/denominator))
            a = value-(value*weights).sum()/denominator
            b = target-(target*weights).sum()/denominator
            norm = ((a.square()*weights).sum()*(b.square()*weights).sum()).sqrt()
            correlation.append(float((a*b*weights).sum()/norm) if norm > 1e-12 else None)
        energies[label], correlations[label] = energy, correlation
    return {'columns': ['L1', 'L2', 'L3'], 'highpass_energy': energies,
            'phase_target_correlation': correlations, 'region_pixels': pixels,
            'interpretation': 'diagnostic only; no physical requirement of diagonal dominance'}
