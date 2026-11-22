"""TA interface geometry. Public API uses metres and unchanged c2w matrices."""
import math

import numpy as np
import torch
from torch.nn import functional as F

HEIGHT, WIDTH = 1080, 1920
SCREEN_WIDTH_M, SCREEN_HEIGHT_M = .596736, .335664
LAYER_X_M = (.06, 0., -.06)  # L1 nearest, L3 farthest
IPD_M, HFOV_DEG = .06, 40.
# Published EyeReal lego_bulldozer data: screen 0 is farthest, image 0 is the
# right eye (scene_dict "1 left; 0 right"). Index TA L1/L2/L3 and [left,right].
NETWORK_LAYER_ORDER, NETWORK_EYE_ORDER = (2, 1, 0), (1, 0)


def screen_corners(device='cpu', dtype=torch.float64, height=HEIGHT, width=WIDTH):
    """Same aspect-ratio rule as TA forward_render, front-to-back order."""
    w, h = SCREEN_WIDTH_M, SCREEN_WIDTH_M*height/width
    return torch.tensor([[[x, -w/2, h/2], [x, w/2, h/2], [x, -w/2, -h/2], [x, w/2, -h/2]]
                         for x in LAYER_X_M], device=device, dtype=dtype)


def validate_pose(value):
    pose = np.asarray(value, dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all() or not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-8):
        raise ValueError('Expected finite 4x4 c2w with last row [0,0,0,1].')
    r = pose[:3, :3]
    if not np.allclose(r.T@r, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(r), 1., atol=1e-5):
        raise ValueError('c2w rotation must be orthonormal and right-handed; values are not repaired.')
    return pose


class ExamRenderer:
    """Cached TA ray geometry with differentiable double-precision sampling.

    Compute rays and the hard mask in float64 using NumPy (default) or
    Torch (opt-in), then cache float64 scoring grids on CPU/GPU. Unlike the historical
    homography adapter, individual rays remain defined when a panel crosses
    the camera near plane. No pose reconstruction or cm/m conversion occurs.
    """
    def __init__(self, views_m, height, width, device=None, geometry_backend="numpy", intrinsics=None):
        if isinstance(views_m, torch.Tensor):
            if device is None:
                device = views_m.device
            views_m = views_m.detach().cpu().numpy()
        views = np.stack([validate_pose(view) for view in views_m])
        self.device = torch.device(device or 'cpu')
        self.height, self.width, self.eyes, self.layers = height, width, len(views), 3
        if geometry_backend not in ('numpy', 'torch'):
            raise ValueError('geometry_backend must be numpy or torch.')
        from .intrinsics import normalize_intrinsics
        intrinsics = normalize_intrinsics(intrinsics, len(views), height, width)
        self.intrinsics = intrinsics
        self.geometry_backend = geometry_backend
        if geometry_backend == 'torch':
            from .ray_eval import ta_grids
            grids, masks = ta_grids(views, height, width, self.device, torch.float64, intrinsics=intrinsics)
            self.grid = grids.reshape(-1, height, width, 2)
            self.geometry_mask = masks[:, None].double()
        else:
            focal = width/(2*np.tan(np.deg2rad(HFOV_DEG)/2))
            local_x = (np.arange(width, dtype=np.float64)+.5-width/2)/focal
            screen_height = SCREEN_WIDTH_M*height/width
            grids = np.empty((3, len(views), height, width, 2), dtype=np.float64)
            masks = np.ones((len(views), height, width), dtype=bool)
            # Match the TA tiled expression exactly, including non-exact input rotations.
            for eye, pose in enumerate(views):
                rotation, origin = pose[:3, :3], pose[:3, 3]
                if intrinsics is not None:
                    fx, fy, cx, cy = intrinsics[eye]
                    local_x = (np.arange(width, dtype=np.float64)+.5-cx)/fx
                for top in range(0, height, 64):
                    bottom = min(top+64, height)
                    local_y = -(np.arange(top, bottom, dtype=np.float64)+.5-height/2)/focal
                    if intrinsics is not None:
                        local_y = -(np.arange(top, bottom, dtype=np.float64)+.5-cy)/fy
                    dx, dy, dz = [rotation[j, 0]*local_x[None, :]+rotation[j, 1]*local_y[:, None]-rotation[j, 2]
                                  for j in range(3)]
                    parallel = np.abs(dx) < 1e-9
                    for layer, x in enumerate(LAYER_X_M):
                        t = np.divide(x-origin[0], dx, out=np.zeros_like(dx), where=~parallel)
                        y, z = origin[1]+t*dy, origin[2]+t*dz
                        u = (y+SCREEN_WIDTH_M/2)/SCREEN_WIDTH_M*width
                        v = (screen_height/2-z)/screen_height*height
                        masks[eye, top:bottom] &= (~parallel)&(t > 0)&(u >= 0)&(u <= width)&(v >= 0)&(v <= height)
                        support = np.isfinite(u)&np.isfinite(v)&(u > -.5)&(u < width+.5)&(v > -.5)&(v < height+.5)
                        grids[layer, eye, top:bottom, :, 0] = np.where(support, 2*u/width-1, 3.)
                        grids[layer, eye, top:bottom, :, 1] = np.where(support, 2*v/height-1, 3.)
            self.grid = torch.from_numpy(grids.reshape(-1, height, width, 2)).to(self.device)
            self.geometry_mask = torch.from_numpy(masks[:, None]).to(self.device).double()
        if (self.geometry_mask.sum((1, 2, 3)) == 0).any():
            raise ValueError('At least one eye has no valid three-panel intersection.')
        ones = torch.ones(3*self.eyes, 1, height, width, dtype=torch.float64, device=self.device)
        footprints = F.grid_sample(ones, self.grid, align_corners=False)
        self.soft_mask = footprints.reshape(3, self.eyes, 1, height, width).prod(0)

    def accumulate(self, patterns):
        if patterns.shape != (3, 3, self.height, self.width):
            raise ValueError('Expected front-to-back [3,3,H,W] patterns.')
        values = patterns.to(self.grid.dtype)/255 if patterns.dtype == torch.uint8 else patterns.to(self.grid.dtype)
        sources = values[:, None].expand(-1, self.eyes, -1, -1, -1).reshape(-1, 3, self.height, self.width)
        sampled = F.grid_sample(sources, self.grid, align_corners=False)
        # Sequential addition follows reference, rather than an unspecified reduction tree.
        sampled = sampled.reshape(3, self.eyes, 3, self.height, self.width)
        return (sampled[0]+sampled[1])+sampled[2]

    def iteration_copy(self, dtype):
        """Share geometry semantics while using a separate sampling precision for optimization."""
        import copy
        renderer = copy.copy(self)
        renderer.grid = self.grid.to(dtype)
        renderer.geometry_mask = self.geometry_mask.to(dtype)
        renderer.soft_mask = self.soft_mask.to(dtype)
        return renderer

    @classmethod
    def from_centimetres(cls, views_cm, height, width, device=None):
        views_m = views_cm.detach().double().clone()
        views_m[:, :3, 3] /= 100
        return cls(views_m, height, width, device)

    def __call__(self, patterns):
        return torch.sin(self.accumulate(patterns)*(np.pi/2)).square().float()

    @staticmethod
    def quantize_patterns(patterns):
        return torch.floor(patterns.detach().clamp(0, 1)*255+.5).double()/255


def masked_psnr(eyes, target, masks):
    """TA float eyes, HWC arrays: each eye separately, then equal-eye mean."""
    errors = (np.asarray(eyes, dtype=np.float64)-np.asarray(target, dtype=np.float64))**2
    masks = np.asarray(masks, dtype=bool)
    areas = masks.sum((1, 2))
    if (areas == 0).any():
        raise ValueError('Cannot score an eye with zero valid pixels.')
    mse = (errors*masks[..., None]).sum((1, 2, 3))/(areas*3)
    psnr = -10*np.log10(np.maximum(mse, 1e-12))
    return {'psnr_mean_eyes_dB': float(psnr.mean()), 'psnr_each_eye_dB': psnr.tolist(),
            'mse_each_eye': mse.tolist(), 'valid_pixels_each_eye': areas.tolist()}
