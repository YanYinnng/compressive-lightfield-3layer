"""GPU TA-ray sampling grids for many poses, for evaluation at extra viewpoints.

Same expressions as the TA forward_render / ExamRenderer (pixel centres, horizontal
FOV 40 deg, three planes at x=+.06/0/-.06 m, hard mask), evaluated in torch so that
hundreds of neighbouring poses can be rendered without the CPU NumPy cache.
GPU fused multiply-add may move a boundary pixel of the mask; the unit test bounds it.
"""
import math

import numpy as np
import torch
from torch.nn import functional as F

from .exam_geometry import HFOV_DEG, LAYER_X_M, SCREEN_WIDTH_M, validate_pose


def ta_grids(poses, height, width, device, dtype=torch.float32, intrinsics=None):
    """c2w poses [N,4,4] metres -> grids [3,N,H,W,2] (front-to-back) and mask [N,H,W]."""
    poses = torch.as_tensor(np.stack([validate_pose(p) for p in poses]), dtype=torch.float64, device=device)
    from .intrinsics import normalize_intrinsics
    intrinsics = normalize_intrinsics(intrinsics, len(poses), height, width)
    focal = width/(2*math.tan(math.radians(HFOV_DEG)/2))
    local_x = (torch.arange(width, dtype=torch.float64, device=device)+.5-width/2)/focal
    local_y = -(torch.arange(height, dtype=torch.float64, device=device)+.5-height/2)/focal
    rotation, origin = poses[:, :3, :3], poses[:, :3, 3]
    directions = (rotation[:, :, 0, None, None]*local_x[None, None, None, :]
                  + rotation[:, :, 1, None, None]*local_y[None, None, :, None] - rotation[:, :, 2, None, None])
    if intrinsics is not None:
        k = torch.as_tensor(intrinsics, dtype=torch.float64, device=device)
        local_x = (torch.arange(width, dtype=torch.float64, device=device)[None]+.5-k[:, 2, None])/k[:, 0, None]
        local_y = -(torch.arange(height, dtype=torch.float64, device=device)[None]+.5-k[:, 3, None])/k[:, 1, None]
        directions = (rotation[:, :, 0, None, None]*local_x[:, None, None, :]
                      +rotation[:, :, 1, None, None]*local_y[:, None, :, None]-rotation[:, :, 2, None, None])
    dx, dy, dz = directions[:, 0], directions[:, 1], directions[:, 2]
    parallel = dx.abs() < 1e-9
    safe = torch.where(parallel, torch.ones_like(dx), dx)
    screen_height = SCREEN_WIDTH_M*height/width
    grids = torch.empty(3, len(poses), height, width, 2, dtype=dtype, device=device)
    mask = ~parallel
    for layer, x in enumerate(LAYER_X_M):
        t = torch.where(parallel, torch.zeros_like(dx), (x-origin[:, 0, None, None])/safe)
        u = (origin[:, 1, None, None]+t*dy+SCREEN_WIDTH_M/2)/SCREEN_WIDTH_M*width
        v = (screen_height/2-(origin[:, 2, None, None]+t*dz))/screen_height*height
        mask &= (t > 0)&(u >= 0)&(u <= width)&(v >= 0)&(v <= height)
        support = torch.isfinite(u)&torch.isfinite(v)&(u > -.5)&(u < width+.5)&(v > -.5)&(v < height+.5)
        grids[layer, ..., 0] = torch.where(support, 2*u/width-1, torch.full_like(u, 3.)).to(dtype)
        grids[layer, ..., 1] = torch.where(support, 2*v/height-1, torch.full_like(v, 3.)).to(dtype)
    return grids, mask


def render_views(patterns, grids):
    """patterns [3,3,H,W] in [0,1] (L1/L2/L3), grids from ta_grids -> eyes [N,3,H,W].

    Differentiable with respect to patterns (for eye-box losses); wrap calls in
    torch.no_grad() when only evaluating.
    """
    layers, count, height, width, _ = grids.shape
    sources = patterns.to(grids.dtype)[:, None].expand(-1, count, -1, -1, -1).reshape(-1, 3, *patterns.shape[-2:])
    sampled = F.grid_sample(sources, grids.reshape(-1, height, width, 2), align_corners=False)
    sampled = sampled.reshape(layers, count, 3, height, width)
    return torch.sin(((sampled[0]+sampled[1])+sampled[2])*(math.pi/2)).square()


@torch.no_grad()
def masked_psnr_views(prediction, target, mask):
    """Per-view PSNR on the TA hard mask; prediction/target [N,3,H,W] in [0,1], mask [N,H,W]."""
    weights = mask[:, None].to(prediction.dtype)
    area = weights.sum((1, 2, 3))*3
    mse = ((prediction-target.to(prediction.dtype)).square()*weights).sum((1, 2, 3))/area.clamp_min(1)
    psnr = -10*torch.log10(mse.clamp_min(1e-12))
    return psnr.where(area > 0, torch.full_like(psnr, float('nan'))), mask.flatten(1).float().mean(1)
