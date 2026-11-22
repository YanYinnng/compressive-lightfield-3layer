import unittest
from unittest.mock import patch
import numpy as np
import torch

from .core import refine
from .depth_eval import depth_regions
from .exam_geometry import ExamRenderer
from .test_ray_eval import pose


class DepthProxyTests(unittest.TestCase):
    def test_depth_bins_are_relative_to_layer2_world_x(self):
        p = pose(1., 0., 0.)
        depth = np.array([[.9, 1., 1.1, .9]])
        regions, x = depth_regions(depth, np.array([[255, 255, 255, 0]]), p)
        np.testing.assert_allclose(x, [[.1, 0., -.1, .1]], atol=1e-12)
        self.assertEqual(regions['near'].tolist(), [[True, False, False, False]])
        self.assertEqual(regions['middle'].tolist(), [[False, True, False, False]])
        self.assertEqual(regions['far'].tolist(), [[False, False, True, False]])

    def test_quantized_layer_stays_fixed_during_tail_optimization(self):
        renderer = ExamRenderer(np.stack([pose(.9, -.03, 0.), pose(.9, .03, 0.)]), 9, 16)
        initial = torch.full((3, 3, 9, 16), .15)
        target = renderer(torch.full_like(initial, .2)).detach()
        observed = []
        original = ExamRenderer.accumulate
        def observe(self, patterns):
            if patterns.requires_grad:
                observed.append(patterns.detach().clone())
            return original(self, patterns)
        with patch.object(ExamRenderer, 'accumulate', observe):
            output, _ = refine(renderer, target, initial, steps=5, eval_every=1,
                               iteration_dtype=torch.float32, quantize_schedule=[(3, 0)], highpass_weight=.001)
        self.assertEqual(len(observed), 5)
        torch.testing.assert_close(observed[2][0], observed[4][0], rtol=0, atol=0)
        torch.testing.assert_close(observed[2][0].double(), renderer.quantize_patterns(observed[2])[0], atol=1e-7, rtol=0)
        self.assertTrue(torch.isfinite(output).all())


if __name__ == '__main__':
    unittest.main(verbosity=2)
