import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from .core import refine, scores
from .exam_geometry import ExamRenderer
from .refocus import refocus_views
from .test_ray_eval import pose
from tools.overnight_eval import evaluation_subset, summarize_eyebox, summarize_sweep


class LocalWorkTests(unittest.TestCase):
    def test_float32_refinement_selects_by_double_quantized_score(self):
        renderer = ExamRenderer(np.stack([pose(.9, -.03, 0.), pose(.9, .03, 0.)]), 18, 32)
        torch.manual_seed(12)
        truth = torch.rand(3, 3, 18, 32)*.4
        initial = torch.full_like(truth, .18)
        target = renderer(truth).detach()
        result, trace = refine(renderer, target, initial, steps=12, eval_every=3, phase_steps=2,
                               anchor_weight=.001, iteration_dtype=torch.float32)
        self.assertEqual(renderer.grid.dtype, torch.float64)
        self.assertEqual(result.dtype, torch.float64)
        score = scores(renderer, result, target)['psnr_mean_eyes']
        self.assertAlmostEqual(score, max(r['png_psnr'] for r in trace['history']))
        torch.testing.assert_close(result, renderer.quantize_patterns(result), rtol=0, atol=0)
        self.assertGreaterEqual(score, trace['history'][0]['png_psnr'])

    def test_torch_geometry_on_cpu_and_differentiable_iteration_copy(self):
        views = np.stack([pose(.8, .2, -.1, .3), pose(.8, .26, -.1, .3)])
        original = ExamRenderer(views, 36, 64)
        actual = ExamRenderer(views, 36, 64, geometry_backend='torch')
        torch.testing.assert_close(actual.grid, original.grid, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(actual.geometry_mask, original.geometry_mask, rtol=0, atol=0)
        patterns = torch.full((3, 3, 36, 64), .2, requires_grad=True)
        actual.iteration_copy(torch.float32).accumulate(patterns).sum().backward()
        self.assertTrue(torch.isfinite(patterns.grad).all())
        self.assertGreater(float(patterns.grad.abs().sum()), 0)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_native_torch_geometry_oblique_roll_matches_numpy(self):
        views = np.stack([pose(.6, .3, -.1, .3), pose(.9, -.35, .1, -.2)])
        original = ExamRenderer(views, 1080, 1920, 'cuda')
        actual = ExamRenderer(views, 1080, 1920, 'cuda', geometry_backend='torch')
        torch.testing.assert_close(actual.geometry_mask, original.geometry_mask, rtol=0, atol=0)
        patterns = torch.rand(3, 3, 1080, 1920, device='cuda')
        with torch.no_grad():
            difference = (actual(patterns)-original(patterns)).abs()
        self.assertLessEqual(float(difference[actual.geometry_mask.bool().expand(-1, 3, -1, -1)].max()), 1e-7)

    def test_refocus_zero_offset_and_invalid_input(self):
        images = torch.rand(1, 3, 9, 16, dtype=torch.float64)
        stack, mask = refocus_views(images, [[0., 0.]], [.5, 1.])
        torch.testing.assert_close(stack, images.expand(2, -1, -1, -1))
        self.assertTrue(mask.all())
        with self.assertRaises(ValueError):
            refocus_views(images, [[0., 0.]], [0.])

    def test_refocus_known_depths_match_independent_pupil_rays(self):
        from tools.check_refocus import known_depth_views, direct_focus_truth
        images, offsets, rois, depths = known_depth_views()
        distances = [.4, .5, .8, 1.2, 1.4]
        stack, valid = refocus_views(images, offsets, distances)
        direct = direct_focus_truth(offsets, distances, *images.shape[-2:])
        sharp = images[len(images)//2]
        for label, roi in rois.items():
            mask = roi*valid.all(0)
            errors = ((stack-sharp).square()*mask).sum((1, 2, 3))
            direct_errors = ((direct-sharp).square()*mask).sum((1, 2, 3))
            self.assertEqual(distances[int(errors.argmin())], depths[label])
            self.assertEqual(distances[int(direct_errors.argmin())], depths[label])
            rmse = (((stack-direct).square()*mask).sum()/(len(stack)*3*mask.sum())).sqrt()
            self.assertLess(float(rmse), .01)

    def test_refocus_weights_and_regular_aperture_subset(self):
        from .refocus import aperture_subset
        from tools.render_lego_gt import subaperture_offsets_mm
        offsets = np.array(subaperture_offsets_mm())  # mixed 11x11@2mm + 5x5@1mm, 137 per eye
        self.assertEqual(len(offsets), 137)
        self.assertEqual(len(aperture_subset(offsets, 2.)), 121)
        self.assertEqual(len(aperture_subset(offsets, 1., 2.)), 25)
        self.assertEqual(len(aperture_subset(offsets, 2., 4., 'circle')), 13)
        inner = aperture_subset(offsets, 1., 2.)
        np.testing.assert_allclose(np.unique(offsets[inner, 0]), [-2, -1, 0, 1, 2])
        with self.assertRaises(ValueError):
            aperture_subset(offsets, 0.)
        images = torch.rand(3, 3, 9, 16, dtype=torch.float64)
        weighted, _ = refocus_views(images, np.zeros((3, 2)), [1.], weights=[2., 1., 1.])
        torch.testing.assert_close(weighted[0], (2*images[0]+images[1]+images[2])/4)
        equal, _ = refocus_views(images, np.zeros((3, 2)), [1.])
        torch.testing.assert_close(equal[0], images.mean(0))
        for bad in ([1., -1., 1.], [0., 0., 0.], [1., 1.]):
            with self.assertRaises(ValueError):
                refocus_views(images, np.zeros((3, 2)), [1.], weights=bad)

    def test_schedule_callback_and_invalid_schedule(self):
        renderer = ExamRenderer(np.stack([pose(.9, -.03, 0.), pose(.9, .03, 0.)]), 9, 16)
        initial = torch.full((3, 3, 9, 16), .15)
        target = renderer(torch.full_like(initial, .2)).detach()
        captured = []
        result, trace = refine(renderer, target, initial, steps=4, eval_every=2, iteration_dtype=torch.float32,
                               lr_schedule=[(3, .005)], checkpoint_callback=lambda step, p, score: captured.append((step, score)))
        self.assertEqual([step for step, score in captured], [2, 4])
        self.assertAlmostEqual(scores(renderer, result, target)['psnr_mean_eyes'], max(score for step, score in captured))
        with self.assertRaises(ValueError):
            refine(renderer, target, initial, steps=4, lr_schedule=[(0, .01)])

    def test_predictor_defaults_use_h100_verified_mode(self):
        import inspect
        from .exam_adapter import ExamPredictor
        parameters = inspect.signature(ExamPredictor).parameters
        self.assertEqual(parameters['iteration_dtype'].default, 'float32')
        self.assertEqual(parameters['geometry_backend'].default, 'torch')
        self.assertEqual(parameters['steps'].default, 30)
        self.assertEqual(parameters['lr'].default, .02)
        self.assertFalse(parameters['quantized_tail'].default)
        with self.assertRaisesRegex(ValueError, 'at least 30'):
            ExamPredictor(steps=20, quantized_tail=True)

    def test_iteration_modes_are_named_by_what_they_do(self):
        from .exam_adapter import ExamPredictor
        self.assertEqual(ExamPredictor.ITERATION_MODES, {'legacy': None, 'float32': torch.float32, 'float64': torch.float64})
        with self.assertRaises(ValueError):
            ExamPredictor(iteration_dtype='float16')
        renderer = ExamRenderer(np.stack([pose(.9, -.03, 0.), pose(.9, .03, 0.)]), 9, 16)
        target = renderer(torch.full((3, 3, 9, 16), .2)).detach()
        initial = torch.full((3, 3, 9, 16), .15)  # float32, like the network output
        _, legacy = refine(renderer, target, initial, steps=2)
        _, double = refine(renderer, target, initial, steps=2, iteration_dtype=torch.float64)
        self.assertEqual(legacy['iteration_dtype'], 'legacy: parameters torch.float32, sampling torch.float64')
        self.assertEqual(double['iteration_dtype'], 'torch.float64')

    def test_our_shell_scripts_use_lf_line_endings(self):
        # CRLF breaks `set -Eeuo pipefail` on Linux; testing/*.sh are the unchanged TA originals.
        root = Path(__file__).resolve().parents[1]
        for path in sorted((root/'solution').glob('*.sh')):
            self.assertNotIn(b'\r\n', path.read_bytes(), path.name)

    def test_summary_does_not_pool_selection_and_verification(self):
        rows = []
        for split, value in [('train', 10.), ('val', 20.), ('test', 30.)]:
            rows.append({'id': split, 'split': split, 'centre_psnr_dB': {'a': value},
                         'views': [{'kind': 'subaperture', 'offset_mm': [0., 0.], 'psnr_dB': {'a': value}}],
                         'runs': [{'init': 'network', 'lr': .02, 'trace': [{'step': 10, 'best': value, 'seconds': 1.}]}],
                         'network_s': .1, 'geometry_cache_s': .2})
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder/'records').write_text('\n'.join(json.dumps(r) for r in rows))
            summarize_eyebox(folder/'records', folder/'eyebox')
            summarize_sweep(folder/'records', folder/'sweep')
            eyebox, sweep = [json.loads((folder/name).read_text()) for name in ('eyebox', 'sweep')]
        self.assertEqual(eyebox['by_split']['val']['centre_psnr_dB_mean']['a'], 20.)
        subset = evaluation_subset(rows[0])
        self.assertEqual(eyebox['by_evaluation_subset'][subset]['samples'], 1)
        self.assertEqual(sweep['curves_by_evaluation_subset'][subset]['network_lr0.02']['10']['mean_best_psnr_dB'], 10.)


if __name__ == '__main__':
    unittest.main(verbosity=2)
