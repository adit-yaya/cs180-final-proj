"""Relative zoom geometry and border-only feathering."""
import base64
import io

import numpy as np
from PIL import Image, ImageFilter


FRAME_SIZE = 1024


def decode_zoom_crop(encoded_image, center, zoom_factor=2):
    """Return the original crop pixels and the source bounds used to select them."""
    if len(center) != 2 or not all(0 <= value <= 1 for value in center):
        raise ValueError('Center must lie inside the image')
    try:
        data = base64.b64decode(encoded_image.split(',')[-1], validate=True)
        with Image.open(io.BytesIO(data)) as source:
            if source.size != (FRAME_SIZE, FRAME_SIZE):
                raise ValueError('Expected a 1024 × 1024 input image')
            size = source.width // zoom_factor
            left, top = (
                max(0, min(source.width - size, int(value * source.width + 0.5) - size // 2))
                for value in center
            )
            bounds = (left, top, left + size, top + size)
            return source.crop(bounds).convert('RGB'), bounds
    except (ValueError, OSError) as exc:
        raise ValueError('Invalid input image; expected a 1024 × 1024 frame') from exc


def feather_patch(background, patch, feather):
    """Blend an aligned square patch at its output resolution."""
    if feather == 0:
        return patch.copy()
    size = patch.width
    axis = np.minimum(np.arange(size), np.arange(size)[::-1])
    distance = np.minimum(axis[:, None], axis[None, :])
    mask = (0.5 - 0.5 * np.cos(np.pi * np.clip(distance / feather, 0, 1)))[..., None]
    blended = mask * np.asarray(patch) + (1 - mask) * np.asarray(background)
    return Image.fromarray(blended.round().astype(np.uint8))


def sharpen_input(image, amount=1.5):
    """Unsharp mask before VAE encoding/noising; amount=0 disables it."""
    if amount == 0:
        return image.copy()
    return image.filter(ImageFilter.UnsharpMask(radius=1.5, percent=round(100*amount), threshold=3))
