import unittest

import numpy as np
from PIL import Image

from tools.build_eyereal_eval import HEIGHT, TA_K, WIDTH, source_intrinsics, source_pose, to_ta_image


class EyeRealEvalConversionTest(unittest.TestCase):
    def test_pose_uses_scene_scale_and_proper_world_rotation(self):
        left = source_pose(np.array([-.21, 0., 10.5]), .07)
        right = source_pose(np.array([.21, 0., 10.5]), .07)
        np.testing.assert_allclose((left[:3, 3]+right[:3, 3])/2, [1.5, 0., 0.])
        np.testing.assert_allclose(right[:3, 3]-left[:3, 3], [0., .06, 0.])
        for pose in (left, right):
            np.testing.assert_allclose(pose[:3, 2], pose[:3, 3]/np.linalg.norm(pose[:3, 3]))
            self.assertGreater(pose[1, 0], .99)

    def test_pixels_follow_the_source_and_target_intrinsics(self):
        k = source_intrinsics((WIDTH, HEIGHT), 40)
        self.assertLess(k[0, 0], k[1, 1])
        y, x = np.indices((HEIGHT, WIDTH))
        image = Image.fromarray(np.stack((x//8, y//5, 0*x), axis=-1).astype(np.uint8))
        converted = to_ta_image(image, k)
        for u, v in ((100, 100), (960, 540), (1800, 920)):
            src_x, src_y, _ = k@np.linalg.solve(TA_K, [u+.5, v+.5, 1.])
            red, green, _ = converted.getpixel((u, v))
            self.assertLessEqual(abs(red-src_x//8), 1)
            self.assertLessEqual(abs(green-src_y//5), 1)
        with self.assertRaisesRegex(ValueError, 'beyond the source image'):
            to_ta_image(image, TA_K*[[2, 1, 1], [1, 2, 1], [1, 1, 1]])


if __name__ == '__main__':
    unittest.main()
