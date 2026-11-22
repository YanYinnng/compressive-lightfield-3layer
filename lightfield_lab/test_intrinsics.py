import unittest
import numpy as np
import torch

from .exam_geometry import ExamRenderer
from .intrinsics import standard_intrinsics, forward_render_intrinsics, warp_to_standard, normalize_intrinsics
from .ta_reference import load_reference
from .test_ray_eval import pose


class IntrinsicsTests(unittest.TestCase):
    def test_explicit_standard_is_exactly_original_for_both_backends(self):
        views = np.stack([pose(.8, -.03, .05), pose(.8, .03, .05)])
        k = standard_intrinsics(36, 64)
        for backend in ('numpy', 'torch'):
            before = ExamRenderer(views, 36, 64, geometry_backend=backend)
            after = ExamRenderer(views, 36, 64, geometry_backend=backend, intrinsics=k)
            self.assertTrue(torch.equal(before.grid, after.grid))
            self.assertTrue(torch.equal(before.geometry_mask, after.geometry_mask))
        patterns = np.random.default_rng(3).integers(0, 256, (3, 36, 64, 3), dtype=np.uint8)
        actual, masks = forward_render_intrinsics(patterns, views, k)
        expected, expected_masks = load_reference().forward_render(patterns, *views)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(masks, expected_masks)

    def test_shift_and_anisotropic_focal_match_independent_reference(self):
        views = np.stack([pose(.8, -.03, .05, .2), pose(.8, .03, .05, -.1)])
        k = np.tile(standard_intrinsics(36, 64), (2, 1))
        k[:, 0] *= [1.1, .85]; k[:, 1] *= [.9, 1.15]; k[:, 2:] += [[4., -2.], [-3., 1.]]
        patterns = np.random.default_rng(4).integers(0, 256, (3, 36, 64, 3), dtype=np.uint8)
        expected, expected_masks = forward_render_intrinsics(patterns, views, k)
        tensor = torch.from_numpy(patterns).permute(0, 3, 1, 2)
        for backend in ('numpy', 'torch'):
            renderer = ExamRenderer(views, 36, 64, geometry_backend=backend, intrinsics=k)
            np.testing.assert_array_equal(renderer.geometry_mask[:, 0].bool().numpy(), expected_masks)
            np.testing.assert_allclose(renderer(tensor).permute(0, 2, 3, 1).numpy(), expected, atol=1e-7, rtol=0)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_native_custom_intrinsics_cuda_match_numpy_reference(self):
        views = np.stack([pose(.8, -.17, .05, .2), pose(.8, .23, -.08, -.1)])
        k = np.tile(standard_intrinsics(1080, 1920), (2, 1))
        k[:, 0] *= [.97, 1.03]; k[:, 1] *= [1.02, .95]
        k[:, 2:] += [[44., -27.], [-63., 15.]]
        patterns = np.random.default_rng(9).integers(0, 256, (3, 1080, 1920, 3), dtype=np.uint8)
        expected, expected_masks = forward_render_intrinsics(patterns, views, k)
        renderer = ExamRenderer(views, 1080, 1920, 'cuda', geometry_backend='torch', intrinsics=k)
        with torch.no_grad():
            actual = renderer(torch.from_numpy(patterns).permute(0, 3, 1, 2).cuda()).permute(0, 2, 3, 1).cpu().numpy()
        np.testing.assert_array_equal(renderer.geometry_mask[:, 0].bool().cpu().numpy(), expected_masks)
        np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=0)

    def test_direct_focus_intrinsics_make_pupil_rays_meet_layer2(self):
        from .ray_eval import ta_grids
        centre = pose(1., 0., 0.)
        offsets = np.array([[-.002, -.001], [0., 0.], [.002, .001]])
        poses = np.repeat(centre[None], 3, axis=0)
        poses[:, :3, 3] += offsets@centre[:3, :2].T
        k = np.tile(standard_intrinsics(36, 64), (3, 1))
        k[:, 2] += k[:, 0]*offsets[:, 0]
        k[:, 3] -= k[:, 1]*offsets[:, 1]
        grids, _ = ta_grids(poses, 36, 64, 'cpu', torch.float64, intrinsics=k)
        torch.testing.assert_close(grids[1], grids[1, 1:2].expand_as(grids[1]), atol=1e-12, rtol=0)

    def test_warp_inverse_mapping_has_correct_shift_and_support(self):
        image = torch.arange(16.).view(1, 1, 1, 16).expand(1, 3, 9, 16)
        k = standard_intrinsics(9, 16); k[2] += 2
        warped, valid = warp_to_standard(image, k)
        torch.testing.assert_close(warped[..., :-2], image[..., 2:], atol=1e-5, rtol=0)
        self.assertFalse(valid[..., -2:].any())
        original, valid = warp_to_standard(image, standard_intrinsics(9, 16))
        self.assertIs(original, image)
        self.assertTrue(valid.all())
        with self.assertRaises(ValueError):
            normalize_intrinsics([-1, 1, 0, 0], 1, 9, 16)




class V3SubmissionTests(unittest.TestCase):
    def test_template_differs_from_ta_only_in_create_predictor(self):
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        ta = (root/'testing_v3/inference.py').read_text(encoding='utf-8').splitlines()
        ours = (root/'solution/ta_v3_inference.py').read_text(encoding='utf-8').splitlines()
        start = ta.index('def create_predictor(weights):')
        end = next(i for i in range(start+1, len(ta)) if ta[i].startswith('def '))
        self.assertEqual(ta[:start], ours[:start])
        self.assertEqual(ta[end:], ours[len(ours)-(len(ta)-end):])

    def test_offaxis_k_is_valid_ta_matrix_and_standard_when_frontal(self):
        from .v3_submission import k_matrix, offaxis_intrinsics
        k = offaxis_intrinsics(pose(.9, -.03, 0.), pose(.9, .03, 0.))
        matrix = k_matrix(*k[0])
        np.testing.assert_array_equal(matrix[2], [0, 0, 1])
        self.assertEqual(matrix[0, 1], 0)
        np.testing.assert_allclose(k[:, :2], standard_intrinsics(1080, 1920)[0])
        self.assertLess(np.abs(k[:, 3]-540).max(), 1e-6)  # level eyes: no vertical shift


if __name__ == '__main__':
    unittest.main(verbosity=2)
