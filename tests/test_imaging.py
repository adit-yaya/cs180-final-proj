import base64
import io
import unittest

import numpy as np
from PIL import Image

from imaging import decode_zoom_crop, feather_patch


class ImagingTests(unittest.TestCase):
    def test_crop_uses_half_of_frame_width_and_stays_inside_edges(self):
        rng = np.random.default_rng(42)
        frame = Image.fromarray(rng.integers(0, 256, (1024, 1024, 3), dtype='uint8'))
        buffer = io.BytesIO()
        frame.save(buffer, format='PNG')
        encoded = base64.b64encode(buffer.getvalue()).decode()
        for center, expected in [
            ((0.5, 0.5), (256, 256, 768, 768)),
            ((0, 1), (0, 512, 512, 1024)),
            ((1, 0), (512, 0, 1024, 512)),
        ]:
            with self.subTest(center=center):
                crop, bounds = decode_zoom_crop(encoded, center)
                self.assertEqual(bounds, expected)
                np.testing.assert_array_equal(crop, frame.crop(expected))

    def test_only_border_blends(self):
        rng = np.random.default_rng(42)
        background = Image.fromarray(rng.integers(0, 256, (1024, 1024, 3), dtype='uint8'))
        patch = Image.fromarray(rng.integers(0, 256, (1024, 1024, 3), dtype='uint8'))
        for width in (1, 32, 64, 256):
            with self.subTest(width=width):
                output = np.asarray(feather_patch(background, patch, width))
                np.testing.assert_array_equal(output[width:-width, width:-width],
                                              np.asarray(patch)[width:-width, width:-width])
                np.testing.assert_array_equal(output[0], np.asarray(background)[0])

    def test_zero_width_uses_generated_patch(self):
        background = Image.new('RGB', (1024, 1024), (20, 40, 60))
        patch = Image.new('RGB', (1024, 1024), (200, 180, 160))
        np.testing.assert_array_equal(feather_patch(background, patch, 0), patch)


if __name__ == '__main__':
    unittest.main()
