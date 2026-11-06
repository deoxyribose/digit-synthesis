"""SVHN crops, and the ink estimate a single-glyph program is fitted to.

Split out of the old `svhn_fit.py`, whose other half fitted the spline scene
model that `pen/` replaced.  Everything here is numpy and the data: no
renderer, no optimiser, so `pen/` depends on this without pulling a second
renderer in behind it.

The ink estimate is the point of the module.  SVHN is photographs -- a digit
is dark-on-light or light-on-dark, under a lighting gradient, at whatever
exposure the camera chose -- and a pen program draws coverage in [0, 1].
`plane_fit` reads the illumination off the border ring, `ink_hat` divides it
out and signs the result by the crop's own polarity, and what comes back is
comparable to an alpha map.  Fitting against raw RGB would make the program
explain the lighting, which it cannot do and should not try to.
"""

import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

NPZ = os.path.join(HERE, "svhn", "data", "svhn.npz")

# The twelve polarity-matched crops of the plan: light-on-dark, in the sharper
# and more contrasty half, drawn with seed 11 so the set is fixed before any
# result is seen.
INDICES = [595, 3006, 3158, 3598, 10456, 12423, 12673, 15303, 15508, 18275,
           20429, 24143]

EXTRA_SEED = 11

def study_indices(n_total=24, seed=EXTRA_SEED, split="test"):
    """The fixed twelve, extended to `n_total` with the classes balanced.

    The original twelve are kept verbatim and first, so every number already
    reported against them stays comparable.  The additions are drawn from the
    same pool on the same two criteria the twelve were drawn on -- polarity
    matched, and in the contrastier half -- with the only new criterion being
    the digit's *label*, because the twelve happened to contain four 1s and no
    7 or 9 at all, and a gallery that never shows a 7 is not evidence about 7s.

    Label and contrast are both properties of the photograph.  Nothing here
    looks at how well anything fitted, which is the line that matters: select
    on the input and the sample is honest, select on the outcome and it is
    not.
    """
    z = np.load(NPZ)
    y = z["%s_y" % split]
    pol = z["%s_border_minus_centre" % split]
    x = z["%s_x" % split]
    have = list(INDICES)
    need = max(0, n_total - len(have))
    if need == 0:
        return have

    pool = np.where(pol < 0)[0]
    pool = np.setdiff1d(pool, np.asarray(have))
    g = x[pool].astype(np.float32).mean(3) / 255.0
    con = g.reshape(len(pool), -1).std(1)
    pool = pool[con >= np.median(con)]            # the contrastier half, as before

    counts = {c: int((y[np.asarray(have)] == c).sum()) for c in range(10)}
    rng = np.random.default_rng(seed)
    by_class = {c: list(rng.permutation(pool[y[pool] == c])) for c in range(10)}

    add = []
    while len(add) < need:
        c = min(range(10), key=lambda k: (counts[k], k))
        if not by_class[c]:
            counts[c] += 1                        # exhausted; stop favouring it
            continue
        add.append(int(by_class[c].pop()))
        counts[c] += 1
    return have + sorted(add)

def ring_mask(n=32):
    m = np.zeros((n, n), bool)
    m[0, :] = m[-1, :] = m[:, 0] = m[:, -1] = True
    return m

def plane_fit(img32):
    """c0, gx, gy per channel from the border ring.  img32 is (n,32,32,3)."""
    n = img32.shape[1]
    ax = (np.arange(n, dtype=np.float32) - (n - 1) / 2.0) / ((n - 1) / 2.0)
    X, Y = np.meshgrid(ax, ax)            # X varies along the last axis
    m = ring_mask(n)
    A = np.stack([np.ones(m.sum(), np.float32), X[m], Y[m]], 1)
    out = np.empty((len(img32), 3, 3), np.float32)        # (n, [c0,gx,gy], chan)
    for i, im in enumerate(img32):
        out[i] = np.linalg.lstsq(A, im[m], rcond=None)[0]
    return out

def plane_image(coef, n=32):
    ax = (np.arange(n, dtype=np.float32) - (n - 1) / 2.0) / ((n - 1) / 2.0)
    X, Y = np.meshgrid(ax, ax)
    return (coef[:, 0][:, None, None, :]
            + coef[:, 1][:, None, None, :] * X[None, ..., None]
            + coef[:, 2][:, None, None, :] * Y[None, ..., None])

LUMA = np.array([0.299, 0.587, 0.114], np.float32)

def ink_hat(img32, coef):
    """The digit's alpha, as far as anything model-free can tell."""
    g = img32 @ LUMA
    p = plane_image(coef, img32.shape[1]) @ LUMA
    ex = g - p
    sc = np.percentile(ex.reshape(len(ex), -1), 98, axis=1)[:, None, None]
    return np.clip(ex / np.maximum(sc, 1e-3), 0.0, 1.0)

def load(indices=INDICES, split="test"):
    z = np.load(NPZ)
    idx = np.asarray(indices)
    x = z["%s_x" % split][idx].astype(np.float32) / 255.0
    y = z["%s_y" % split][idx]
    pol = z["%s_border_minus_centre" % split][idx]
    coef = plane_fit(x)
    return {"idx": idx, "y": y, "pol": pol, "tgt32": x,
            "tgt64": np.repeat(np.repeat(x, 2, 1), 2, 2).astype(np.float32),
            "plane": coef, "ink": ink_hat(x, coef)}

def iou(a32, ink, thr=0.5, core=None):
    """Overlap, optionally restricted to a window of the frame."""
    A, B = a32 > thr, ink > thr
    if core is not None:
        A, B = A & core[None], B & core[None]
    inter = (A & B).sum((1, 2)).astype(np.float32)
    union = (A | B).sum((1, 2)).astype(np.float32)
    return inter / np.maximum(union, 1.0)

def core_window(n=32, half=9):
    """The central 2*half square: where the labelled digit of a format-2 crop
    is, and where asking 'did the glyph slot find it' is a fair question."""
    m = np.zeros((n, n), bool)
    c = n // 2
    m[c - half:c + half, c - half:c + half] = True
    return m
