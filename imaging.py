"""Fixed zoom geometry and border-only feathering."""
import base64
import io

import numpy as np
from PIL import Image, ImageFilter


def decode_zoom_context(encoded_image, center):
    """Decode one working frame and extract the shared click-centered context."""
    if len(center) != 2 or not all(0 <= value <= 1 for value in center):
        raise ValueError('Center must lie inside the image')
    try:
        data = base64.b64decode(encoded_image.split(',')[-1], validate=True)
        with Image.open(io.BytesIO(data)) as source:
            if source.size != (1024, 1024):
                raise ValueError('Expected a 1024 × 1024 input image')
            return context_at(source.convert('RGB'), center)
    except (ValueError, OSError) as exc:
        raise ValueError('Invalid input image; expected a 1024 × 1024 frame') from exc


def context_at(image, center):
    """Extract a 1024-square context without resampling source pixels."""
    x, y = [min(1023, round(v * 1024)) for v in center]
    padded = np.pad(np.asarray(image), ((512,512),(512,512),(0,0)), mode='edge')
    return Image.fromarray(padded[y:y+1024, x:x+1024])


def zoom_center(context):
    """Render the center of a completed composite at 2x."""
    return context.crop((256,256,768,768)).resize((1024,1024), Image.Resampling.LANCZOS)


def insert_center(context, patch, feather=32):
    """Blend only inside the patch border; preserve interior pixels exactly."""
    if not 0 <= feather <= 128:
        raise ValueError('Border width must be between 0 and 128 pixels')
    if context.size != (1024,1024) or patch.size != (512,512):
        raise ValueError('Expected a 1024 context and 512 patch')
    out = context.copy()
    out.paste(feather_patch(context.crop((256, 256, 768, 768)), patch, feather), (256, 256))
    return out


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


def raw_center(context, patch):
    """Unblended paste for an honest raw-generation comparison."""
    out = context.copy()
    out.paste(patch, (256,256))
    return out


def sharpen_input(image, amount=1.5):
    """Unsharp mask before VAE encoding/noising; amount=0 disables it."""
    if amount == 0:
        return image.copy()
    return image.filter(ImageFilter.UnsharpMask(radius=1.5, percent=round(100*amount), threshold=3))
