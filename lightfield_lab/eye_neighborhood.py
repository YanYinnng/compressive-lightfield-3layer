"""Depth-guided supervision for the angular variation of the emitted light field."""
import math

import numpy as np
import torch
from torch.nn import functional as F

from .intrinsics import standard_intrinsics
from .ray_eval import ta_grids


@torch.no_grad()
def warp_parallel(image, depth, trusted, offset_m, stride):
    """Inverse-warp to a camera translated in its image plane, with no rotation."""
    height, width = depth.shape
    fx, fy, _, _ = standard_intrinsics(height, width)
    a, b = offset_m
    rows = torch.arange(0, height, stride, device=depth.device, dtype=depth.dtype)+.5
    cols = torch.arange(0, width, stride, device=depth.device, dtype=depth.dtype)+.5
    v, u = torch.meshgrid(rows, cols, indexing='ij')
    # Require a valid neighbourhood and discard depth discontinuities. This
    # keeps one-view resampling away from occlusions and disocclusions.
    z = torch.where(trusted, depth, 1.)[None, None]
    bad = F.max_pool2d((~trusted).float()[None, None], 9, stride=1, padding=4)
    span = F.max_pool2d(z, 9, stride=1, padding=4)+F.max_pool2d(-z, 9, stride=1, padding=4)
    safe = (bad == 0) & (span < .025)

    def grid(px, py):
        return torch.stack((2*px/width-1, 2*py/height-1), dim=-1)[None]

    def sample(array, xy):
        return F.grid_sample(array, xy, align_corners=False)[0]

    x, y = u, v
    for _ in range(4):
        distance = sample(z, grid(x, y))[0].clamp_min(.2)
        x, y = u+fx*a/distance, v-fy*b/distance
    xy = grid(x, y)
    distance = sample(z, xy)[0].clamp_min(.2)
    residual = (x-u-fx*a/distance).abs()+(y-v+fy*b/distance).abs()
    valid = (sample(safe.float(), xy)[0] > .999) & (residual < .5)
    valid &= (x >= .5)&(x <= width-.5)&(y >= .5)&(y <= height-.5)
    return sample(image[None], xy), valid


class AngularTargets:
    """Fit changes between same-pixel rays before and after moving the pupil."""

    def __init__(self, renderer, views, eyes, target, centre, trust, stride=3):
        self.eyes = torch.as_tensor(eyes, device=target.device)
        device = target.device
        grids, valid = [], []
        # Construct one view at a time to bound temporary CUDA memory.
        for pose in views:
            grid, support = ta_grids([pose], renderer.height, renderer.width, device, torch.float32)
            grids.append(grid[:, 0, ::stride, ::stride].contiguous())
            valid.append(support[0, ::stride, ::stride].contiguous())
        central = renderer.grid.reshape(3, 2, renderer.height, renderer.width, 2)
        self.grid = torch.cat((central[:, :, ::stride, ::stride].float(),
                               torch.stack(grids, dim=1)), dim=1)
        central_valid = renderer.geometry_mask[:, 0, ::stride, ::stride].bool()[self.eyes]
        self.mask = (trust & torch.stack(valid) & central_valid)[:, None]
        self.delta = target-centre[self.eyes, :, ::stride, ::stride]
        self.count = (self.mask.sum()*3).clamp_min(1)
        self.signal = self.error(self.delta)

    def error(self, residual):
        return (residual.square()*self.mask).sum()/self.count

    def loss(self, patterns):
        _, views, h, w, _ = self.grid.shape
        phases = []
        for layer, grid in zip(patterns.float(), self.grid):
            sampled = F.grid_sample(layer[None], grid.reshape(1, views*h, w, 2), align_corners=False)
            phases.append(sampled[0].unflatten(1, (views, h)).transpose(0, 1))
        images = torch.sin(((phases[0]+phases[1])+phases[2])*(math.pi/2)).square()
        return self.error(images[2:]-images[:2][self.eyes]-self.delta)


def shifted_pose(pose, offset_mm):
    a, b = np.asarray(offset_mm)*.001
    result = pose.copy()
    result[:3, 3] += a*pose[:3, 0]+b*pose[:3, 1]
    return result
