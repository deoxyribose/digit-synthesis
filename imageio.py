"""Tiny helpers for turning renderer output into PNG bytes / files."""

import base64
import io
import numpy as np
from PIL import Image


def to_u8(img):
    return (np.clip(np.asarray(img), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def ink(img, rgb=(17, 17, 17), bg=(255, 255, 255)):
    """Render a scalar coverage map as coloured ink on a background."""
    a = np.clip(np.asarray(img), 0.0, 1.0)[..., None]
    return (np.array(bg) * (1 - a) + np.array(rgb) * a + 0.5).astype(np.uint8)


def overlay(current, target, cur_rgb=(214, 69, 45), tgt_rgb=(40, 90, 200)):
    """Target in blue, current render in red, overlapping ink goes dark."""
    c = np.clip(np.asarray(current), 0, 1)[..., None]
    t = np.clip(np.asarray(target), 0, 1)[..., None]
    out = np.ones(c.shape[:2] + (3,))
    out = out * (1 - t) + (np.array(tgt_rgb) / 255.0) * t
    out = out * (1 - c) + (np.array(cur_rgb) / 255.0) * c
    return (out * 255 + 0.5).astype(np.uint8)


def png_bytes(arr, scale=1):
    im = Image.fromarray(arr)
    if scale != 1:
        im = im.resize((im.width * scale, im.height * scale), Image.NEAREST)
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def data_uri(arr, scale=1):
    return "data:image/png;base64," + base64.b64encode(png_bytes(arr, scale)).decode()


def save(path, arr, scale=1):
    with open(path, "wb") as f:
        f.write(png_bytes(arr, scale))


def grid(tiles, cols, pad=2, bg=245):
    """Tile equal-sized RGB images into a sheet.

    Tile size is read from the tiles themselves rather than a module global,
    which is what lets a sheet of 64x64 glyphs and a sheet of 32x96 triptychs
    share one implementation.
    """
    th, tw = tiles[0].shape[:2]
    rows = (len(tiles) + cols - 1) // cols
    out = np.full((rows * (th + pad) + pad, cols * (tw + pad) + pad, 3), bg, np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        y, x = pad + r * (th + pad), pad + c * (tw + pad)
        out[y:y + th, x:x + tw] = t
    return out
