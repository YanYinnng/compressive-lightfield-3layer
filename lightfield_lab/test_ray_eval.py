import unittest

import numpy as np
import torch

from .exam_geometry import ExamRenderer
from .ray_eval import masked_psnr_views, render_views, ta_grids
from .ta_reference import load_reference


def pose(x, y, z, roll=0.):
    back = np.array([x, y, z]); back /= np.linalg.norm(back)
    right = np.cross([0., 0., 1.], back); right /= np.linalg.norm(right)
    up = np.cross(back, right)
    c, s = np.cos(roll), np.sin(roll)
    view = np.eye(4); view[:3, :3] = np.stack([c*right+s*up, -s*right+c*up, back], axis=1); view[:3, 3] = [x, y, z]
    return view


class RayEvalTests(unittest.TestCase):
    def test_grids_and_render_match_exam_renderer_and_ta_reference(self):
        generator = np.random.default_rng(5)
        patterns = generator.integers(0, 256, (3, 36, 64, 3), dtype=np.uint8)
        views = np.stack([pose(1., -.03, .02), pose(.6, .3, -.1, .3), pose(.075, 0., 0.), pose(.9, .2, .1, -.2)])
        grids, mask = ta_grids(views, 36, 64, 'cpu', torch.float64)
        source = torch.from_numpy(patterns).permute(0, 3, 1, 2).double()/255
        eyes = render_views(source, grids)
        reference = load_reference()
        for index in range(0, len(views), 2):
            pair = views[index:index+2]
            expected, expected_mask = reference.forward_render(patterns, *pair)
            renderer = ExamRenderer(pair, 36, 64)
            self.assertLessEqual(int((mask[index:index+2].numpy() != expected_mask).sum()), 2)
            agree = expected_mask & mask[index:index+2].numpy()
            np.testing.assert_allclose(eyes[index:index+2].permute(0, 2, 3, 1).numpy()[agree], expected[agree], atol=1e-7)
            cached = renderer(source).double().permute(0, 2, 3, 1).numpy()
            np.testing.assert_allclose(eyes[index:index+2].permute(0, 2, 3, 1).numpy()[agree], cached[agree], atol=1e-6)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required for the native-resolution GPU check')
    def test_native_resolution_gpu_float64_has_zero_mask_difference(self):
        """A3 acceptance: 1920x1080, float64 GPU grids versus the TA reference (TA demo poses when present)."""
        import json
        from pathlib import Path
        demo = Path(__file__).resolve().parents[1]/'testing/demo_input/test.json'
        if demo.is_file():
            rows = json.loads(demo.read_text())['samples']
        else:  # same geometry as the two demo pairs: eyes 6 cm apart, about 0.96 m and 0.90 m from O2, looking at O2
            rows = []
            for x, y, z in ((.957, -.02, .01), (.899, .05, -.02)):
                centre = np.array([x, y, z]); right = np.cross([0., 0., 1.], centre); right /= np.linalg.norm(right)
                rows.append({'pose_l': pose(*(centre-.03*right)), 'pose_r': pose(*(centre+.03*right))})
        patterns = np.random.default_rng(11).integers(0, 256, (3, 1080, 1920, 3), dtype=np.uint8)
        source = torch.from_numpy(patterns).permute(0, 3, 1, 2).cuda().double()/255
        reference = load_reference()
        for row in rows:
            poses = np.stack([row['pose_l'], row['pose_r']])
            expected, expected_mask = reference.forward_render(patterns, *poses)
            grids, mask = ta_grids(poses, 1080, 1920, 'cuda', torch.float64)
            with torch.no_grad():
                eyes = render_views(source, grids).permute(0, 2, 3, 1).cpu().numpy()
            np.testing.assert_array_equal(mask.cpu().numpy(), expected_mask)
            self.assertLessEqual(float(np.abs(eyes-expected)[expected_mask].max()), 1e-7)
            del grids, mask

    def test_render_views_is_differentiable(self):
        views = np.stack([pose(1., -.03, .02), pose(1., .03, .02)])
        grids, mask = ta_grids(views, 9, 16, 'cpu', torch.float64)
        patterns = torch.full((3, 3, 9, 16), .2, dtype=torch.float64, requires_grad=True)
        render_views(patterns, grids)[mask[:, None].expand(-1, 3, -1, -1)].sum().backward()
        self.assertTrue(torch.isfinite(patterns.grad).all())
        self.assertGreater(float(patterns.grad.abs().sum()), 0)

    def test_masked_psnr_views(self):
        target = torch.zeros(2, 3, 4, 4)
        prediction = target.clone(); prediction[0] = .1; prediction[1] = .01
        psnr, fraction = masked_psnr_views(prediction, target, torch.ones(2, 4, 4, dtype=torch.bool))
        torch.testing.assert_close(psnr, torch.tensor([20., 40.]))
        torch.testing.assert_close(fraction, torch.ones(2))


if __name__ == '__main__':
    unittest.main(verbosity=2)
