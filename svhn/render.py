"""Tile a random sample of SVHN: native RGB on the left, its gray28 twin on the right.

Side by side because the two caveats in README.md are only visible by comparison --
mixed polarity, and the neighbouring digits format 2 leaves in the crop.

Usage: python3 -I render.py <data_dir> <split> <side> <seed> <out.png>
   e.g. python3 -I render.py data test 8 7 data/svhn_8x8.png
"""
import os
import sys

import numpy as np
from PIL import Image

data, split, side, seed, out = (sys.argv[1], sys.argv[2], int(sys.argv[3]),
                                int(sys.argv[4]), sys.argv[5])
n = side * side

z = np.load(os.path.join(data, "svhn.npz"))
idx = np.random.default_rng(seed).choice(len(z["%s_y" % split]), n, replace=False)
x, y = z["%s_x" % split][idx], z["%s_y" % split][idx]
gray = z["%s_gray28" % split][idx]
pol = z["%s_border_minus_centre" % split][idx]


def grid(tiles, pad=1):
    t = np.pad(np.stack(tiles), ((0, 0), (pad, pad), (pad, pad)) + ((0, 0),) * 1,
               constant_values=255)
    return np.concatenate([np.concatenate(list(t[side * r:side * (r + 1)]), 1)
                           for r in range(side)], 0)


rgb = grid(list(x))
g = np.pad(gray.reshape(-1, 28, 28), ((0, 0), (2, 2), (2, 2)))   # 28 -> 32, keep alignment
gry = grid(list(np.repeat(g[..., None], 3, 3)))
tile = np.concatenate([rgb, np.full((rgb.shape[0], 12, 3), 255, np.uint8), gry], 1)
Image.fromarray(tile).resize((tile.shape[1] * 3, tile.shape[0] * 3),
                             Image.NEAREST).save(out)

print("SVHN %s, %d random images (seed %d), labels %d per row:" % (split, n, seed, side))
for r in range(side):
    print("   " + " ".join(str(int(v)) for v in y[side * r:side * (r + 1)]))
print("dark digit on light ground: %d/%d" % ((pol > 0).sum(), n))
print("wrote %s (%dx%d)" % (out, tile.shape[1] * 3, tile.shape[0] * 3))
