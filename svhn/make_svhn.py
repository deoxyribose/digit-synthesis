"""Convert the SVHN .mat files to .npz arrays that JAX can consume directly.

The .mat files need scipy.io.loadmat and hold `(32, 32, 3, N)` -- channels
third, batch last -- with labels 1..10. Neither is what you want in a training
loop, so this writes one .npz of plain arrays instead:

    train_x, test_x            (N, 32, 32, 3) uint8   NHWC, the Flax conv layout
    train_y, test_y            (N,)           int32   0..9
    train_gray28, test_gray28  (N, 784)       uint8   28x28 grayscale, flattened

`jnp.asarray` takes any of these as-is. They stay uint8 because float32 at
native size would be 900 MB for the training split; divide by 255 at use, which
is one fused op on device.

The gray28 arrays are there to match this repo's own convention -- ood.py and
the RMNIST sets are all `(n, 784)` float32 in [0, 1] at 28x28 -- so SVHN can
drop into the OOD comparison without a second resampling step. Two notes:

  * Luma is ITU-R BT.601 (0.299, 0.587, 0.114), and the 32->28 resize is
    Pillow's antialiased BILINEAR, i.e. support-scaled, so it averages rather
    than drops rows. MNIST itself was built by antialiased downsampling.

  * Polarity is left exactly as the photographs have it. RMNIST is 0.0
    background / 1.0 ink by construction; SVHN house numbers are roughly half
    light-on-dark and half dark-on-light, so a good fraction of these are the
    inverse of the convention the fitter expects. That is a property of the
    data, and normalising it is a modelling decision, so it is not made here.
    `border_minus_centre` is saved to support whichever choice you make: it is
    mean(border ring) - mean(centre 14x14) per image, in [0, 1] units, so it is
    positive for a dark digit on a light ground.

Labels: upstream uses 10 for the digit 0, so `y % 10` is the whole remap.

Usage: python3 -I make_svhn.py <data_dir>
"""

import os
import sys

import numpy as np
from PIL import Image
from scipy.io import loadmat

H = W = 28
LUMA = np.array([0.299, 0.587, 0.114], np.float32)   # ITU-R BT.601


def load_split(data, name):
    d = loadmat(os.path.join(data, "%s_32x32.mat" % name))
    x = np.ascontiguousarray(d["X"].transpose(3, 0, 1, 2))   # (32,32,3,N) -> NHWC
    y = (d["y"].ravel().astype(np.int32)) % 10               # 10 is the digit 0
    return x, y


def to_gray28(x):
    """NHWC uint8 -> (N, 784) uint8, BT.601 luma, antialiased 32->28."""
    luma = x.astype(np.float32) @ LUMA
    out = np.empty((len(x), H * W), np.uint8)
    for i, im in enumerate(luma):
        small = Image.fromarray(im, mode="F").resize((W, H), Image.BILINEAR)
        out[i] = np.asarray(small, np.float32).round().clip(0, 255).astype(np.uint8).ravel()
    return out


def border_minus_centre(gray):
    """mean(border ring) - mean(centre 14x14), in [0,1]. Positive = dark digit."""
    g = gray.reshape(-1, H, W).astype(np.float32) / 255.0
    ring = np.concatenate([g[:, 0, :], g[:, -1, :], g[:, 1:-1, 0], g[:, 1:-1, -1]], 1)
    return ring.mean(1) - g[:, 7:21, 7:21].mean((1, 2))


def preview(path, x, y, gray, per_class=10):
    """One row per digit 0..9: native RGB on the left, the 28x28 grayscale right."""
    rgb_rows, gry_rows = [], []
    for d in range(10):
        idx = np.flatnonzero(y == d)[:per_class]
        rgb_rows.append(np.concatenate(list(x[idx]), 1))
        g = gray[idx].reshape(-1, H, W)
        g = np.pad(g, ((0, 0), (2, 2), (2, 2)))                  # 28 -> 32, align rows
        gry_rows.append(np.repeat(np.concatenate(list(g), 1)[..., None], 3, 2))
    gutter = np.full((32, 8, 3), 255, np.uint8)
    rows = [np.concatenate([r, gutter, g], 1) for r, g in zip(rgb_rows, gry_rows)]
    tile = np.concatenate(rows, 0)
    Image.fromarray(tile).resize((tile.shape[1] * 2, tile.shape[0] * 2),
                                 Image.NEAREST).save(path)
    return tile.shape


def main():
    data = sys.argv[1]
    out, arrays = os.path.join(data, "svhn.npz"), {}

    for name in ("train", "test"):
        x, y = load_split(data, name)
        gray = to_gray28(x)
        pol = border_minus_centre(gray)
        arrays["%s_x" % name] = x
        arrays["%s_y" % name] = y
        arrays["%s_gray28" % name] = gray
        arrays["%s_border_minus_centre" % name] = pol.astype(np.float32)
        print("%-5s  x %s %s  y %s in %d..%d  per-digit %s"
              % (name, x.shape, x.dtype, y.shape, y.min(), y.max(),
                 list(np.bincount(y, minlength=10))))
        print("       gray28 %s %s   dark-digit share %.1f%%  (border-centre > 0)"
              % (gray.shape, gray.dtype, 100.0 * (pol > 0).mean()))
        if name == "test":
            shape = preview(os.path.join(data, "svhn_preview.png"), x, y, gray)
            print("       preview svhn_preview.png %dx%d" % (shape[1] * 2, shape[0] * 2))

    np.savez(out, **arrays)       # uncompressed: photographs barely deflate,
    print("wrote %s (%.0f MB)"   # and this loads in a fraction of the time
          % (os.path.basename(out), os.path.getsize(out) / 1e6))


if __name__ == "__main__":
    main()
