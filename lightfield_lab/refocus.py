"""Shift-and-add refocusing for parallel sub-aperture cameras, in metres.

Offsets are (right, up) in the reference camera frame. All cameras must share
rotation, intrinsics and resolution. Ring look-at views cannot use this model.
"""
import math

import numpy as np
import torch
from torch.nn import functional as F


def aperture_subset(offsets_mm, spacing_mm, radius_mm=None, shape='square', tolerance=1e-6):
    """Indices of offsets forming one regular aperture grid.

    The Lego eye-box set mixes an 11x11 2 mm grid and a 5x5 1 mm grid; averaging all of
    them would over-weight the aperture centre. Select points on a single lattice of
    the given spacing, within |offset| <= radius ('square', per axis) or a circular pupil.
    """
    offsets = np.asarray(offsets_mm, dtype=np.float64)
    if offsets.ndim != 2 or offsets.shape[1] != 2 or spacing_mm <= 0 or shape not in ('square', 'circle'):
        raise ValueError('Require [V,2] offsets, positive spacing and shape square|circle.')
    on_lattice = (np.abs(offsets/spacing_mm-np.round(offsets/spacing_mm)) < tolerance).all(1)
    if radius_mm is None:
        inside = np.ones(len(offsets), dtype=bool)
    elif shape == 'square':
        inside = (np.abs(offsets) <= radius_mm+tolerance).all(1)
    else:
        inside = np.hypot(offsets[:, 0], offsets[:, 1]) <= radius_mm+tolerance
    return np.flatnonzero(on_lattice & inside)


@torch.no_grad()
def refocus_views(images, offsets_m, distances_m, masks=None, hfov_deg=40., weights=None):
    """[V,C,H,W] -> focal stack [D,C,H,W], all-view validity [D,1,H,W].

    Images and masks must be in the same order as offsets. Distances are positive
    axial eye-to-focus distances. Invalid samples are excluded from evaluation,
    rather than renormalized with a varying pupil size.
    weights: optional nonnegative per-view aperture weights (normalized here); the
    default equal weighting is only correct for a regular, uniformly sampled aperture
    (see aperture_subset).
    """
    if images.ndim != 4 or not images.is_floating_point():
        raise ValueError('images must be floating [V,C,H,W].')
    offsets = torch.as_tensor(offsets_m, device=images.device, dtype=images.dtype)
    distances = torch.as_tensor(distances_m, device=images.device, dtype=images.dtype)
    if offsets.shape != (len(images), 2) or distances.ndim != 1 or not len(distances):
        raise ValueError('Require [V,2] offsets and nonempty [D] distances.')
    if not torch.isfinite(offsets).all() or not torch.isfinite(distances).all() or (distances <= 0).any():
        raise ValueError('Require finite offsets and positive finite distances.')
    _, _, height, width = images.shape
    y, x = torch.meshgrid(torch.arange(height, device=images.device, dtype=images.dtype)+.5,
                          torch.arange(width, device=images.device, dtype=images.dtype)+.5, indexing='ij')
    focal = width/(2*math.tan(math.radians(hfov_deg)/2))
    if masks is None:
        masks = torch.ones(len(images), 1, height, width, device=images.device, dtype=images.dtype)
    if masks.shape != (len(images), 1, height, width):
        raise ValueError('masks must have shape [V,1,H,W].')
    if weights is None:
        weights = torch.full((len(images),), 1/len(images), device=images.device, dtype=images.dtype)
    else:
        weights = torch.as_tensor(weights, device=images.device, dtype=images.dtype)
        if weights.shape != (len(images),) or not torch.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
            raise ValueError('weights must be [V], finite, nonnegative, with a positive sum.')
        weights = weights/weights.sum()
    stack, valid = [], []
    for distance in distances:
        u = x[None]-focal*offsets[:, 0, None, None]/distance
        v = y[None]+focal*offsets[:, 1, None, None]/distance
        grid = torch.stack([2*u/width-1, 2*v/height-1], -1)
        stack.append((F.grid_sample(images, grid, align_corners=False)*weights[:, None, None, None]).sum(0))
        support = F.grid_sample(masks.to(images.dtype), grid, align_corners=False)
        valid.append((support >= 1-1e-6).all(0))
    return torch.stack(stack), torch.stack(valid)
