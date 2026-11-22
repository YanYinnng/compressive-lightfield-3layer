import io
import struct
import unittest

import numpy as np
from PIL import Image

from lightfield_lab.png_fast import png_bytes


class PngFastTest(unittest.TestCase):
    def test_lossless_rgb8_round_trip(self):
        rng = np.random.default_rng(0)
        array = rng.integers(0, 256, (1080, 1920, 3), dtype=np.uint8)
        array[:40] = 255
        data = png_bytes(array, 1)
        self.assertEqual(struct.unpack('>IIBB', data[16:26]), (1920, 1080, 8, 2))
        self.assertEqual(data[-8:-4], b'IEND')
        with Image.open(io.BytesIO(data)) as image:
            self.assertEqual(image.mode, 'RGB')
            np.testing.assert_array_equal(np.asarray(image), array)


if __name__ == '__main__':
    unittest.main()
