"""Relative zoom geometry and conservative detail compositing."""
import base64
import io

import numpy as np
from PIL import Image, ImageChops, ImageFilter


FRAME_SIZE = 1024


def decode_zoom_crop(encoded_image, center, zoom_factor=2):
    """Return crop pixels and the corresponding bounds in 1024px display coordinates."""
    if len(center) != 2 or not all(0 <= value <= 1 for value in center):
        raise ValueError('Center must lie inside the image')
    try:
        data = base64.b64decode(encoded_image.split(',')[-1], validate=True)
        with Image.open(io.BytesIO(data)) as source:
            if source.size not in ((FRAME_SIZE, FRAME_SIZE), (2 * FRAME_SIZE, 2 * FRAME_SIZE)):
                raise ValueError('Unsupported source dimensions')
            size = FRAME_SIZE // zoom_factor
            left, top = (
                max(0, min(FRAME_SIZE - size, int(value * FRAME_SIZE + 0.5) - size // 2))
                for value in center
            )
            bounds = (left, top, left + size, top + size)
            scale = source.width // FRAME_SIZE
            return source.crop(tuple(value * scale for value in bounds)).convert('RGB'), bounds
    except (ValueError, OSError) as exc:
        raise ValueError('Invalid input image; expected a 1024px or 2048px square frame') from exc


def crop_focus(center, bounds):
    """Express the original click in the coordinate system of the clamped crop."""
    left, top, right, bottom = bounds
    return ((center[0] * FRAME_SIZE - left) / (right - left),
            (center[1] * FRAME_SIZE - top) / (bottom - top))


def local_fine_contrast(image):
    lightness = image.convert('L')
    fine = ImageChops.difference(lightness, lightness.filter(ImageFilter.GaussianBlur(2)))
    return np.asarray(fine.filter(ImageFilter.GaussianBlur(5)), dtype=np.float32)


def preserve_source_detail(source, generated, focus):
    """Keep the source's broad structure and favor bounded new texture near the click."""
    source_rgb = np.asarray(source, dtype=np.float32)
    delta = np.asarray(generated, dtype=np.float32) - source_rgb
    blur = ImageFilter.GaussianBlur(source.width / 128)
    broad_delta = (np.asarray(generated.filter(blur), dtype=np.float32)
                   - np.asarray(source.filter(blur), dtype=np.float32))
    delta -= broad_delta
    # A smooth falloff works even when the click is at the very edge of the crop.
    x = np.linspace(0, 1, source.width, dtype=np.float32)
    y = np.linspace(0, 1, source.height, dtype=np.float32)[:, None]
    distance = (x - focus[0]) ** 2 + (y - focus[1]) ** 2
    weight = np.exp(-distance / (2 * 0.28 ** 2))[..., None]
    # Limit large changes, which are often a shifted edge or an invented object.
    np.clip(broad_delta, -42, 42, out=broad_delta)
    np.clip(delta, -42, 42, out=delta)
    # Resist replacing real fine detail in areas where the render is smoother.
    source_contrast = local_fine_contrast(source)
    generated_contrast = local_fine_contrast(generated)
    texture_weight = np.clip((generated_contrast + 1) / (source_contrast + 1), 0.25, 1)[..., None]
    adjustment = (0.10 + 0.22 * weight) * broad_delta + (0.28 + 0.54 * weight) * texture_weight * delta
    output = np.clip(source_rgb + adjustment, 0, 255)
    return Image.fromarray(output.round().astype(np.uint8))


def feather_patch(background, patch, feather):
    """Blend an aligned square patch at its output resolution."""
    if feather == 0:
        return patch.copy()
    size = patch.width
    axis = np.minimum(np.arange(size), np.arange(size)[::-1])
    distance = np.minimum(axis[:, None], axis[None, :]).astype(np.float32)
    mask = (0.5 - 0.5 * np.cos(np.pi * np.clip(distance / feather, 0, 1)))[..., None]
    blended = mask * np.asarray(patch) + (1 - mask) * np.asarray(background)
    return Image.fromarray(blended.round().astype(np.uint8))


def sharpen_input(image, amount=0):
    """Unsharp mask before VAE encoding/noising; amount=0 disables it."""
    if amount == 0:
        return image.copy()
    return image.filter(ImageFilter.UnsharpMask(radius=1.5, percent=round(100*amount), threshold=3))
