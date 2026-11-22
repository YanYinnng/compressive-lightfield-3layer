"""Meaningful verification of cached optics and the proposed optimization path."""
import math
from pathlib import Path
import tempfile
import unittest

import torch

from .benchmark import upstream_render
from .core import CachedRenderer, png_values, read_patterns, refine, save_images, scores, synthetic_geometry


class RendererTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_forward_and_gradient_match_upstream(self):
        for oblique in (False, True):
            with self.subTest(oblique=oblique):
                views, screens = synthetic_geometry(36, 64, oblique=oblique)
                renderer = CachedRenderer(views, screens, 36, 64)
                torch.manual_seed(9)
                patterns = torch.rand(3, 3, 36, 64, requires_grad=True)*.5
                expected, mask = upstream_render(patterns, views, screens)
                actual = renderer(patterns)
                torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
                torch.testing.assert_close(renderer.soft_mask.expand_as(mask), mask, atol=2e-5, rtol=2e-5)
                weights = torch.rand_like(actual)
                expected_grad = torch.autograd.grad((expected*weights).sum(), patterns, retain_graph=True)[0]
                actual_grad = torch.autograd.grad((actual*weights).sum(), patterns)[0]
                torch.testing.assert_close(actual_grad, expected_grad, atol=5e-5, rtol=5e-5)

    def test_geometry_mask_against_independent_ray_intersection(self):
        height, width = 36, 64
        views, screens = synthetic_geometry(height, width, oblique=True)
        renderer = CachedRenderer(views, screens, height, width)
        y, x = torch.meshgrid(torch.arange(height)+.5, torch.arange(width)+.5, indexing='ij')
        focal = width/(2*math.tan(math.radians(20)))
        camera_directions = torch.stack([(x-width/2)/focal, -(y-height/2)/focal, -torch.ones_like(x)], -1)
        eye_masks = []
        for view in views:
            directions = camera_directions@view[:3, :3].T
            eye = view[:3, 3]
            valid = torch.ones(height, width, dtype=torch.bool)
            for panel in screens:
                horizontal, vertical = panel[1]-panel[0], panel[2]-panel[0]
                normal = torch.linalg.cross(horizontal, vertical)
                denominator = directions@normal
                distance = ((panel[0]-eye)@normal)/denominator
                point = eye+distance[..., None]*directions-panel[0]
                u = (point@horizontal)/horizontal.square().sum()
                v = (point@vertical)/vertical.square().sum()
                valid &= (distance > 0)&(u >= 0)&(u <= 1)&(v >= 0)&(v <= 1)
            eye_masks.append(valid)
        self.assertTrue(torch.equal(torch.stack(eye_masks)[:, None], renderer.geometry_mask.bool()))

    def test_png_roundtrip_and_negative_clipping_are_real(self):
        patterns = torch.tensor([-.1, .4, .2])[:, None, None, None].expand(3, 3, 4, 4)
        with tempfile.TemporaryDirectory() as folder:
            save_images(patterns, folder, 'layer')
            actual = read_patterns(folder)
        torch.testing.assert_close(actual, png_values(patterns))
        raw = torch.sin(patterns[:, 0, 0, 0].sum()*math.pi/2).square()
        exported = torch.sin(actual[:, 0, 0, 0].sum()*math.pi/2).square()
        self.assertGreater(float((raw-exported).abs()), .15)

    def test_phase_warm_start_escapes_all_black_stationary_point(self):
        views, screens = synthetic_geometry(36, 64)
        renderer = CachedRenderer(views, screens, 36, 64)
        initial = torch.zeros(3, 3, 36, 64)
        target = renderer(torch.full_like(initial, .12)).detach()
        vanilla, _ = refine(renderer, target, initial, steps=15)
        phase, _ = refine(renderer, target, initial, steps=15, phase_steps=5)
        self.assertEqual(float(vanilla.abs().max()), 0.)
        self.assertGreater(scores(renderer, phase, target)['psnr_mean_eyes'],
                           scores(renderer, vanilla, target)['psnr_mean_eyes']+5)

    def test_screen_behind_eye_is_rejected(self):
        views, screens = synthetic_geometry(36, 64)
        views[:, :3, 2] *= -1
        with self.assertRaises(ValueError):
            CachedRenderer(views, screens, 36, 64)

    def test_prediction_consistency_preserves_the_initial_rendering(self):
        views, screens = synthetic_geometry(36, 64)
        renderer = CachedRenderer(views, screens, 36, 64)
        initial = torch.full((3, 3, 36, 64), .12)
        target = renderer(torch.full_like(initial, .2)).detach()
        plain, _ = refine(renderer, target, initial, steps=30)
        protected, _ = refine(renderer, target, initial, steps=30,
                              consistency_renderer=renderer, consistency_weight=10.)
        teacher = renderer(initial)
        plain_change = (renderer(plain)-teacher).square().mean()
        protected_change = (renderer(protected)-teacher).square().mean()
        self.assertLess(float(protected_change), float(plain_change))


if __name__ == '__main__':
    unittest.main(verbosity=2)
