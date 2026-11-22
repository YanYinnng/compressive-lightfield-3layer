import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from .exam_adapter import network_initial
from .exam_data import ExamDataset, read_manifest
from .exam_geometry import ExamRenderer, masked_psnr, screen_corners, validate_pose
from .ta_reference import load_reference


class ExamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.reference = load_reference()

    def test_ray_mask_and_rgb_match_ta_for_normal_oblique_roll_and_near_views(self):
        generator = np.random.default_rng(21)
        patterns = generator.integers(0, 256, (3, 36, 64, 3), dtype=np.uint8)
        for x, y, roll in [(1., 0., 0.), (.6, .3, .3), (.075, 0., 0.)]:
            back = np.array([x, y, 0.]);back /= np.linalg.norm(back)
            right = np.cross([0., 0., 1.], back); right /= np.linalg.norm(right)
            up = np.cross(back, right)
            rotation = np.stack([right*np.cos(roll)+up*np.sin(roll), -right*np.sin(roll)+up*np.cos(roll), back], axis=1)
            view = np.eye(4); view[:3, :3] = rotation;view[:3, 3] = [x, y, .02]
            views = np.stack([view, view.copy()]); views[1, :3, 3] += .06*rotation[:, 0]
            expected, mask = self.reference.forward_render(patterns, *views)
            renderer = ExamRenderer(views, 36, 64)
            actual = renderer(torch.from_numpy(patterns).permute(0, 3, 1, 2)).permute(0, 2, 3, 1).numpy()
            np.testing.assert_array_equal(renderer.geometry_mask[:, 0].numpy().astype(bool), mask)
            np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=1e-7)

    def test_gradient_matches_finite_difference_and_is_finite(self):
        view = np.eye(4); view[:3, :3] = [[0, 0, 1], [1, 0, 0], [0, 1, 0]];view[:3, 3] = [.8, -.03, 0]
        renderer = ExamRenderer(np.stack([view, view]), 9, 16)
        pattern = torch.full((3, 3, 9, 16), .2, dtype=torch.float64, requires_grad=True)
        # Use the double accumulated phase to avoid float32 output finite-difference noise.
        objective = lambda p: torch.sin(renderer.accumulate(p)*np.pi/2).square().mean()
        gradient = torch.autograd.grad(objective(pattern), pattern)[0]
        index = (1, 0, 4, 8)
        plus, minus = pattern.detach().clone(), pattern.detach().clone()
        plus[index] += 1e-5; minus[index] -= 1e-5
        numerical = (objective(plus)-objective(minus))/2e-5
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertAlmostEqual(float(gradient[index]), float(numerical), places=8)

    def test_network_input_uses_checkpoint_eye_and_layer_order_with_paired_poses(self):
        class Recorder:
            def __call__(self, images, views, screens):
                self.images, self.views, self.screens = images, views, screens
                return torch.tensor([.1, .2, .3])[None, :, None, None, None].expand(1, 3, 3, 2, 2)
        recorder = Recorder()
        images = torch.arange(2.)[:, None, None, None].expand(2, 3, 2, 2)
        views = torch.eye(4).repeat(2, 1, 1)
        views[:, 1, 3] = torch.tensor([-.03, .03])  # left, right
        output = network_initial(recorder, images, views, screen_corners())
        torch.testing.assert_close(recorder.screens[0, :, 0, 0], torch.tensor([-.06, 0, .06], dtype=torch.float64))
        torch.testing.assert_close(output[:, 0, 0, 0], torch.tensor([.3, .2, .1]))
        # Checkpoint convention: right eye first, each image still with its own pose.
        torch.testing.assert_close(recorder.images[0], images.flip(0))
        torch.testing.assert_close(recorder.views[0, :, 1, 3], torch.tensor([.03, -.03]))

    def test_pretrained_initialization_prefers_right_eye_first(self):
        weights = Path(__file__).resolve().parents[1]/'weights/pretrained_model.pth'
        if not weights.is_file():
            self.skipTest('pretrained weights missing')
        from .core import make_model
        from .exam_data import read_rgb8
        path, rows = read_manifest(Path(__file__).resolve().parents[1]/'testing/demo_input/test.json')
        row, (height, width) = rows[0], (270, 480)
        poses = np.stack([row['pose_l'], row['pose_r']])
        images = torch.from_numpy(np.stack([read_rgb8(path.parent/row[k]) for k in ['left', 'right']]))
        images = torch.nn.functional.interpolate(images.permute(0, 3, 1, 2).float()/255, (height, width),
                                                 mode='bilinear', antialias=True, align_corners=False)
        model = make_model(height, width, 'cpu', weights)
        renderer = ExamRenderer(poses, height, width)
        views = torch.from_numpy(poses)
        def score(patterns):
            return masked_psnr(renderer(renderer.quantize_patterns(patterns)).permute(0, 2, 3, 1).numpy(),
                               images.permute(0, 2, 3, 1).numpy(), renderer.geometry_mask[:, 0].bool().numpy())
        chosen = score(network_initial(model, images, views, screen_corners()))['psnr_mean_eyes_dB']
        layers = torch.tensor([2, 1, 0])
        with torch.no_grad():
            unswapped = model(images[None], views[None], screen_corners()[layers][None])[0][layers].clamp(0, 1)
        self.assertGreater(chosen, score(unswapped)['psnr_mean_eyes_dB']+2)

    def test_manifest_rejects_duplicate_ids_and_centimetres_as_metres(self):
        import json
        ref = Path(__file__).resolve().parents[1]/'testing/demo_input/test.json'
        data = json.loads(ref.read_text())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'test.json'
            data['samples'][1]['id'] = data['samples'][0]['id']
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                read_manifest(path)
            data['samples'] = data['samples'][:1]
            for key in ['pose_l', 'pose_r']:
                for axis in range(3):data['samples'][0][key][axis][3] *= 100
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'baseline'):
                read_manifest(path)

    def test_dataset_uses_manifest_index_and_native_training_layer_order(self):
        path = Path(__file__).resolve().parents[1]/'testing/demo_input/test.json'
        dataset = ExamDataset(path)
        images, first, screens = dataset[0]
        _, second, _ = dataset[1]
        self.assertEqual(tuple(images.shape), (2, 3, 1080, 1920))
        self.assertFalse(torch.equal(first, second))
        torch.testing.assert_close(screens[:, 0, 0], torch.tensor([-.06, 0, .06], dtype=torch.float64))
        import json
        row = json.loads(path.read_text())['samples'][0]
        torch.testing.assert_close(first, torch.tensor([row['pose_r'], row['pose_l']], dtype=torch.float64))

    def test_non_rotation_pose_is_rejected_and_eye_psnr_aggregation_is_explicit(self):
        bad = np.eye(4);bad[0, 0] = 2
        with self.assertRaisesRegex(ValueError, 'rotation'):
            validate_pose(bad)
        truth = np.zeros((2, 2, 2, 3)); prediction = truth.copy(); prediction[0] = .1;prediction[1] = .01
        result = masked_psnr(prediction, truth, np.ones((2, 2, 2), dtype=bool))
        self.assertAlmostEqual(result['psnr_mean_eyes_dB'], 30.)

    def test_case_budget_reserves_measured_export_and_setup_time(self):
        import time
        from .exam_adapter import ExamPredictor
        predictor = ExamPredictor.__new__(ExamPredictor)
        predictor.solver_backend = 'torch'  # budget logic only; no model load
        predictor.setup_seconds = 4.
        predictor.configure_run(time.perf_counter()+200, 4, export_seconds=5., final_seconds=15.)
        # (200-15-1.25*4*5-1.25*3*4)/4
        self.assertAlmostEqual(predictor._case_deadline()-time.perf_counter(), 36.25, delta=.05)
        predictor.note_export(10.)
        self.assertAlmostEqual(predictor._case_deadline()-time.perf_counter(), 30., delta=.05)
        predictor.configure_run(time.perf_counter()+10, 4, export_seconds=5.)
        self.assertLessEqual(predictor._case_deadline()-time.perf_counter(), 0.)

    def test_expired_refinement_budget_keeps_a_valid_quantized_initialization(self):
        import time
        from .core import refine
        view = np.eye(4);view[:3, :3] = [[0, 0, 1], [1, 0, 0], [0, 1, 0]];view[:3, 3] = [.8, 0, 0]
        renderer = ExamRenderer(np.stack([view, view]), 9, 16)
        initial = torch.full((3, 3, 9, 16), .2)
        target = renderer(initial)
        output, trace = refine(renderer, target, initial, steps=30, deadline=time.perf_counter()-1)
        self.assertEqual(trace['steps_run'], 0)
        self.assertTrue(trace['stopped_for_time'])
        torch.testing.assert_close(output, renderer.quantize_patterns(initial))


if __name__ == '__main__':
    unittest.main(verbosity=2)
