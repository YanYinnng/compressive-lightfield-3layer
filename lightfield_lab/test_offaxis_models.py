import json
from pathlib import Path
import unittest

import numpy as np

from .exam_geometry import HEIGHT, WIDTH, screen_corners
from .offaxis_models import camera, k_matrix, look_at_o2

ROOT = Path(__file__).resolve().parents[1]
PRACTICE = ROOT/'_ta_releases/lego100_practice_v1/testing/test_input/lego/test.json'


def project(k, c2w, points):
    camera_points = (points-c2w[:3, 3])@c2w[:3, :3]
    depth = -camera_points[:, 2]
    return np.stack([k[0]*camera_points[:, 0]/depth+k[2], k[3]-k[1]*camera_points[:, 1]/depth], -1)


class OffaxisModelTests(unittest.TestCase):
    def test_look_at_is_the_spec_construction(self):
        c2w = look_at_o2([1.1, -.3, .2])
        r = c2w[:3, :3]
        np.testing.assert_allclose(r.T@r, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(r), 1., places=12)
        np.testing.assert_allclose(r[:, 2], c2w[:3, 3]/np.linalg.norm(c2w[:3, 3]), atol=1e-12)  # back axis to the eye
        self.assertAlmostEqual(r[2, 0], 0., places=12)  # right axis horizontal: no roll
        self.assertGreater(r[2, 1], 0)  # up axis points up
        np.testing.assert_allclose(look_at_o2([1., 0., 0.])[:3, :3], [[0, 0, 1], [1, 0, 0], [0, 1, 0]], atol=1e-12)

    @unittest.skipUnless(PRACTICE.is_file(), 'practice manifest not present')
    def test_supplied_poses_are_look_at_up_to_float32(self):
        rows = json.loads(PRACTICE.read_text(encoding='utf-8'))['samples']
        worst = 0.
        for row in rows:
            for side in ('pose_l', 'pose_r'):
                pose = np.array(row[side])
                rebuilt = look_at_o2(pose[:3, 3])
                cosine = np.clip((np.trace(rebuilt[:3, :3].T@pose[:3, :3])-1)/2, -1, 1)
                worst = max(worst, np.degrees(np.arccos(cosine)))
        self.assertLess(worst, .05)

    def test_bbox_frustum_puts_l2_corners_on_the_image_borders(self):
        corners = screen_corners(height=HEIGHT, width=WIDTH)[1].numpy()
        for position in ([.9, 0., 0.], [.75, -.4, .12], [1.3, .55, -.08]):
            left, right = look_at_o2(np.array(position)+[0, -.03, 0]), look_at_o2(np.array(position)+[0, .03, 0])
            k, c2w = camera('bbox_eye_exact', left, right)
            for eye in range(2):
                uv = project(k[eye], c2w[eye], corners)
                np.testing.assert_allclose([uv[:, 0].min(), uv[:, 0].max(), uv[:, 1].min(), uv[:, 1].max()],
                                           [0, WIDTH, 0, HEIGHT], atol=1e-6)
                matrix = k_matrix(k[eye])
                np.testing.assert_array_equal(matrix[2], [0, 0, 1])
                self.assertEqual(matrix[0, 1], 0)

    def test_submission_defaults_are_l2_camera_with_sart(self):
        from .v3_submission import DEFAULTS
        self.assertEqual(DEFAULTS['offaxis_model'], 'l2_camera')
        self.assertEqual(DEFAULTS['offaxis_initial'], 'sart')


if __name__ == '__main__':
    unittest.main(verbosity=2)
