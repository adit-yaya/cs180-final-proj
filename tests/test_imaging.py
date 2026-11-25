import base64
import io
import unittest

import numpy as np
from PIL import Image

from imaging import decode_zoom_crop, feather_patch, preserve_source_detail


class ImagingTests(unittest.TestCase):
    def test_crop_uses_selected_zoom_and_stays_inside_edges(self):
        rng = np.random.default_rng(42)
        frame = Image.fromarray(rng.integers(0, 256, (1024, 1024, 3), dtype='uint8'))
        buffer = io.BytesIO()
        frame.save(buffer, format='PNG')
        encoded = base64.b64encode(buffer.getvalue()).decode()
        for factor, center, expected in [
            (2, (0.5, 0.5), (256, 256, 768, 768)),
            (2, (0, 1), (0, 512, 512, 1024)),
            (2, (1, 0), (512, 0, 1024, 512)),
            (4, (0.5, 0.5), (384, 384, 640, 640)),
            (4, (0, 1), (0, 768, 256, 1024)),
            (4, (1, 0), (768, 0, 1024, 256)),
        ]:
            with self.subTest(factor=factor, center=center):
                crop, bounds = decode_zoom_crop(encoded, center, factor)
                self.assertEqual(bounds, expected)
                np.testing.assert_array_equal(crop, frame.crop(expected))

    def test_high_resolution_original_preserves_pixels_and_keeps_display_bounds(self):
        frame = Image.new('RGB', (2048, 2048), 'black')
        frame.putpixel((1536, 17), (255, 0, 0))
        buffer = io.BytesIO()
        frame.save(buffer, format='PNG')
        crop, bounds = decode_zoom_crop(base64.b64encode(buffer.getvalue()).decode(), (1, 0), 4)
        self.assertEqual(bounds, (768, 0, 1024, 256))
        self.assertEqual(crop.size, (512, 512))
        self.assertEqual(crop.getpixel((0, 17)), (255, 0, 0))
        self.assertEqual(crop.getpixel((1, 17)), (0, 0, 0))

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

    def test_structure_preservation_restrains_tone_and_keeps_more_texture_near_click(self):
        source = Image.new('RGB', (64, 64), (100, 100, 100))
        pixels = np.full((64, 64, 3), 160, dtype=np.uint8)
        pixels[:, ::2, :] = 200
        generated = Image.fromarray(pixels)
        output = np.asarray(preserve_source_detail(source, generated, (0.25, 0.5)), dtype=int)
        near = abs(output[32, 16, 0] - output[32, 17, 0])
        far = abs(output[32, 56, 0] - output[32, 57, 0])
        self.assertGreater(near, far)
        self.assertGreater(far, 0)
        self.assertLess(output.mean(), 114)
        np.testing.assert_array_equal(preserve_source_detail(source, source, (0, 1)), source)

    def test_structure_preservation_retains_fine_texture_when_render_smooths_it(self):
        striped = np.full((128, 128, 3), 100, dtype=np.uint8)
        striped[:, ::2, :] = 160
        source = Image.fromarray(striped)
        generated = Image.new('RGB', source.size, (130, 130, 130))
        result = np.asarray(preserve_source_detail(source, generated, (0.5, 0.5)), dtype=int)
        difference = result[64, 64, 0] - result[64, 65, 0]
        self.assertGreater(difference, 45)


if __name__ == '__main__':
    unittest.main()
