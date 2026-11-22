"""Soft phase allocation with a depth prior for the three fixed LCD planes."""
import numpy as np
import torch
from torch.nn import functional as F

from .exam_geometry import LAYER_X_M
from .intrinsics import standard_intrinsics


def stereo_depth(images, poses):
    """Metric axial depth from rectified RGB only, with left/right consistency."""
    import cv2

    height, width = images.shape[1:3]
    size = (width//2, height//2)
    fx, fy, cx, cy = standard_intrinsics(height, width)
    k = np.array([[fx/2, 0, cx/2-.5], [0, fy/2, cy/2-.5], [0, 0, 1.]])
    rotations = poses[:, :3, :3] * np.array([1, -1, -1])
    rotation = rotations[1].T @ rotations[0]
    translation = rotations[1].T @ (poses[0, :3, 3] - poses[1, :3, 3])
    distortion = np.zeros(5)
    r0, r1, p0, p1, _, _, _ = cv2.stereoRectify(
        k, distortion, k, distortion, size, rotation, translation,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    baseline_focal = -p1[0, 3]
    if baseline_focal <= 0 or abs(p1[1, 3]) > 1e-6:
        raise ValueError('Expected left-to-right, horizontally rectifiable poses.')
    rectified = []
    for rgb, r, p in zip(images, (r0, r1), (p0, p1)):
        gray = cv2.cvtColor(cv2.resize(rgb, size, interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
        rectified.append(cv2.warpPerspective(gray, p[:, :3] @ r @ np.linalg.inv(k), size))
    # Cover ranges down to 25 cm with a rectified focal length in pixels.
    num = min((size[0]//16-1)*16, 16*(int(baseline_focal/.25)//16+1))
    disparities = []
    for eye, minimum in enumerate((0, 1-num)):
        matcher = cv2.StereoSGBM_create(minDisparity=minimum, numDisparities=num, blockSize=5,
                                        P1=8*25, P2=32*25, uniquenessRatio=10,
                                        speckleWindowSize=80, speckleRange=2)
        disparities.append(matcher.compute(rectified[eye], rectified[1-eye]).astype(np.float32)/16)
    y, x = np.indices(size[::-1])
    depths = []
    for eye, (r, p) in enumerate(zip((r0, r1), (p0, p1))):
        d = disparities[eye]
        matched_x = np.rint(x-d).astype(int)
        other = disparities[1-eye][y, np.clip(matched_x, 0, size[0]-1)]
        valid = (d*((-1)**eye) > 0) & (other*((-1)**eye) < 0)
        valid &= (np.abs(d) < num) & (np.abs(other) < num)
        valid &= (matched_x >= 0) & (matched_x < size[0]) & (np.abs(d+other) <= 1.)
        z = np.where(valid, baseline_focal/np.maximum(np.abs(d), 1.), 0.)
        native_k = np.array([[fx, 0, cx-.5], [0, fy, cy-.5], [0, 0, 1.]])
        native_to_rect = p[:, :3] @ r @ np.linalg.inv(native_k)
        z = cv2.warpPerspective(z, np.linalg.inv(native_to_rect), (width, height), flags=cv2.INTER_NEAREST)
        ray_z = (r[2, 0]*(np.arange(width)+.5-cx)/fx
                 + r[2, 1]*(np.arange(height)[:, None]+.5-cy)/fy + r[2, 2])
        depth = (z/np.maximum(ray_z, .1)).astype(np.float32)
        depths.append(depth)
    return depths


class PhaseAllocation:
    """Match target phase *differences* by depth, with confidence on both endpoints."""

    def __init__(self, renderer, target, world_x, trusted, stride=3):
        self.full_grid = renderer.grid.float()
        self.grid = self.full_grid[:, ::stride, ::stride].contiguous()
        self.principal = (2/np.pi)*torch.asin(target.clamp(0, 1).sqrt())
        safe_x = torch.where(trusted, world_x, 0.)
        layers = target.new_tensor(LAYER_X_M)[:, None, None, None]
        self.weights = (1-(safe_x[None]-layers).abs()/.06).clamp_min(0)
        # Clip beyond the physical planes to the nearest layer.
        self.weights[0] = torch.where(safe_x >= LAYER_X_M[0], 1., self.weights[0])
        self.weights[2] = torch.where(safe_x <= LAYER_X_M[2], 1., self.weights[2])
        self.trusted = trusted & renderer.geometry_mask[:, 0].bool()
        self.support = renderer.geometry_mask.float()
        ref = self.principal[..., ::stride, ::stride][None]
        weights = self.weights[..., ::stride, ::stride][:, :, None]
        good = self.trusted[:, None, ::stride, ::stride][None]
        self.edges = []
        for axis in (-1, -2):
            a, b = self.endpoints(weights, axis)
            ca, cb = self.endpoints(good, axis)
            mask = ca & cb
            desired = torch.diff(ref, dim=axis)*(a+b)*.5
            self.edges.append((axis, mask, desired, (mask.sum()*9).clamp_min(1)))

    @staticmethod
    def endpoints(value, axis):
        count = value.shape[axis]-1
        return value.narrow(axis, 0, count), value.narrow(axis, 1, count)

    @staticmethod
    def sample(patterns, grid):
        sources = patterns[:, None].expand(-1, 2, -1, -1, -1).flatten(0, 1)
        return F.grid_sample(sources, grid, align_corners=False).unflatten(0, (3, 2))

    def loss(self, patterns):
        phases = self.sample(patterns.float(), self.grid)
        return sum(((torch.diff(phases, dim=axis)-desired).square()*mask).sum()/count
                   for axis, mask, desired, count in self.edges)

    @torch.no_grad()
    def contrast_matrix(self, patterns, regions, stride=3):
        """Rows are true depth ROIs, columns are layer gains along target edges."""
        phases = self.sample(patterns.float(), self.grid)
        reference = self.principal[..., ::stride, ::stride]
        result = {}
        for name, roi in regions.items():
            region = (roi & self.trusted)[:, None, ::stride, ::stride]
            numer = phases.new_zeros(3)
            denom = 0
            for axis in (-1, -2):
                a, b = self.endpoints(region, axis)
                edge = torch.diff(reference, dim=axis)*(a & b)
                numer += (edge[None]*torch.diff(phases, dim=axis)).sum((1, 2, 3, 4))
                denom += edge.square().sum()
            result[name] = (numer/denom).tolist() if denom > 1e-5 else None
        return result

    def geometric_initial(self):
        """Splat soft-split target phase from both eyes into the physical screens."""
        weights = torch.where(self.trusted[None], self.weights, 1/3)
        target = self.principal[None]*weights[:, :, None]
        # grid_sample's adjoint deposits bilinear values and normalizes hit mass.
        with torch.enable_grad():
            grid = self.full_grid
            values = self.principal.new_zeros(3, 3, *self.principal.shape[-2:], requires_grad=True)
            samples = self.sample(values, grid)
            numerator, = torch.autograd.grad(samples, values, target*self.support, retain_graph=True)
            mass, = torch.autograd.grad(samples, values, self.support.expand_as(samples).contiguous())
        return (numerator/mass.clamp_min(1e-6)).clamp(0, 1)
