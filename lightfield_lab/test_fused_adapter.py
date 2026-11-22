import importlib.util
import unittest
from unittest.mock import patch

import numpy as np
import torch

from .exam_geometry import HEIGHT, WIDTH
from .gaussian_scene import stereo_poses
from .v3_submission import V3Predictor


@unittest.skipUnless(torch.cuda.is_available() and importlib.util.find_spec('cupy'),
                     'CUDA and CuPy required for the fused backend')
class FusedAdapterTest(unittest.TestCase):
    def test_two_v3_branches_match_torch_at_native_resolution(self):
        import cupy

        try:
            if cupy.cuda.runtime.getDeviceCount() == 0:
                self.skipTest('CuPy cannot access a CUDA device')
        except cupy.cuda.runtime.CUDARuntimeError as error:
            self.skipTest(f'CuPy cannot access CUDA: {error}')

        poses = stereo_poses(.85, 8., -3.)
        ramp = np.linspace(45, 184, WIDTH, dtype=np.uint8)
        left = np.broadcast_to(ramp[None, :, None], (HEIGHT, WIDTH, 3))
        right = np.broadcast_to(ramp[::-1][None, :, None], left.shape)

        def initial(images, _views, _screens):
            return torch.full((1, 3, 3, HEIGHT, WIDTH), .2, device=images.device)

        with (patch('lightfield_lab.exam_adapter.make_model', return_value=initial),
              patch.dict('os.environ', {'LF_FUSED_BACKEND': 'cupy'})):
            reference = V3Predictor(weights=__file__, steps=2, device='cuda')
            expected = reference(left, right, *poses)
            candidate = V3Predictor(weights=__file__, steps=2, device='cuda', solver='fused')
            with patch('lightfield_lab.exam_adapter.ExamRenderer') as renderer:
                actual = candidate(left, right, *poses)
                renderer.assert_not_called()

        for branch in ('normal', 'offaxis'):
            self.assertEqual(candidate.last_scores[f'{branch}_valid_pixels'],
                             reference.last_scores[f'{branch}_valid_pixels'])
            self.assertAlmostEqual(candidate.last_scores[branch]['psnr_mean_eyes'],
                                   reference.last_scores[branch]['psnr_mean_eyes'], delta=.01)
            trace = candidate.last_scores[f'solver_{branch}']['refinement']
            self.assertEqual(trace['steps_run'], 2)
            self.assertEqual([x['step'] for x in trace['history']], [0, 2])

        for side in ('K_l', 'K_r', 'pose_l', 'pose_r'):
            np.testing.assert_array_equal(actual['offaxis'][side], expected['offaxis'][side])
        for layer in [*actual['patterns'], *actual['offaxis']['patterns']]:
            self.assertEqual((layer.shape, layer.dtype), (left.shape, np.uint8))
