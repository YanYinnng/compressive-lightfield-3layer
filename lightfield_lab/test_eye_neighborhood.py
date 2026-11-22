import unittest

import torch

from .exam_geometry import ExamRenderer
from .eye_neighborhood import AngularTargets, shifted_pose, warp_parallel
from .gaussian_scene import stereo_poses
from .intrinsics import standard_intrinsics
from .ray_eval import render_views, ta_grids


class EyeNeighborhoodTests(unittest.TestCase):
    def test_inverse_warp_parallel_constant_depth(self):
        h, w = 32, 64
        fx, fy, _, _ = standard_intrinsics(h, w)
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        source = torch.stack((x, y, x+y)).float()
        depth = torch.full((h, w), .75)
        trusted = torch.ones_like(depth, dtype=torch.bool)
        trusted[14, 25] = False
        warped, valid = warp_parallel(source, depth, trusted, (2*.75/fx, 3*.75/fy), stride=1)
        torch.testing.assert_close(warped[:, 10, 12], source[:, 7, 14], atol=1e-4, rtol=0.)
        self.assertFalse(valid[17, 23])
        self.assertFalse(valid[0, 10])

    def test_angular_targets_match_native_rays_and_have_gradients(self):
        height, width, stride = 36, 64, 3
        poses = stereo_poses(.75, 12., 3.)
        renderer = ExamRenderer(poses, height, width, geometry_backend='torch')
        eyes = [0, 0, 1, 1]
        views = [shifted_pose(poses[i], (a, b)) for i, (a, b) in zip(eyes, [(2, 2), (-2, 2), (2, -2), (-2, -2)])]
        generator = torch.Generator().manual_seed(4)
        patterns = torch.rand((3, 3, height, width), generator=generator)*.25
        central = renderer(patterns)
        grids, mask = ta_grids(views, height, width, 'cpu', torch.float32)
        nearby = render_views(patterns, grids)[..., ::stride, ::stride]
        trust = torch.ones_like(mask[:, ::stride, ::stride])
        objective = AngularTargets(renderer, views, eyes, nearby, central, trust, stride)
        self.assertGreater(int(objective.mask.sum()), 0)
        self.assertGreater(float(objective.signal), 1e-6)
        self.assertLess(float(objective.loss(patterns)), 1e-12)
        other = (patterns*.8).requires_grad_(True)
        objective.loss(other).backward()
        self.assertTrue(bool(torch.isfinite(other.grad).all()))
        self.assertGreater(float(other.grad.abs().sum()), 0.)


if __name__ == '__main__':
    unittest.main()
