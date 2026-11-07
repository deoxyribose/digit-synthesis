"""Fit the continuous parameters of a pen program to an image.

The division of labour this whole design rests on: the STRUCTURE -- which
moves, in which order, at which discrete angles -- is searched, and the
NUMBERS -- lengths, deviations, glyph pose, pen width -- are fitted by
gradient descent.  This module is the second half.  It does no search at all:
you hand it programs and it finds their best numbers.

One compiled kernel covers every structure, because every discrete choice
rides as data in `enc` and every parameter a structure does not use is
multiplied by exactly zero (`check_pen.py` asserts the gradient is 0.0).  So a
batch may mix a one-move program with a twelve-move one.

Restarts spread over `phi0`, deliberately.  Of the seven glyph parameters it
is the only one the loss is badly multimodal in: translation and scale
descend to the right answer from almost anywhere, but a drawing rotated by 90
degrees is a different picture, and no gradient walks from one to the other.
So the restarts are a sweep of the initial heading rather than a cloud of
noise.  Everything else is jittered at the prior's own relative scales, from
`pen.SD_POSE` -- one statement of how much each parameter varies, not two.

The Adam scan is plain and deliberately so.  The studies this repo grew out
of ran their descent through a scan that also carried a blur anneal, a
gradient-noise ramp and a staged parameter freeze; none of that is here.  If
the landscape turns out to need annealing -- and against a photograph it may
-- that is the first thing to add, and the gate below is how to tell.
"""

import jax
import jax.numpy as jnp
import numpy as np

import svhn_data as F

from . import dsl as PN

CORE = F.core_window(32, 10)       # the centre 20 px: where the labelled digit is
JIT_MOVE = 0.25                    # restart jitter, in units of the prior's sd


def init(progs, rng, cfg=PN.PEN, n_restart=8, page=0.72, jitter=JIT_MOVE):
    """(encoding, p0) for every program x restart.

    Restart 0 of each program is the prior's mean at phi0 = 0 -- the clean
    one, so there is always a candidate nobody jittered.
    """
    reps = [pr for pr in progs for _ in range(n_restart)]
    enc = PN.encode(reps, cfg)
    sd = np.ones(PN.nparam(cfg), np.float32) * PN.SD_LEN
    sd[cfg.NMOVE * PN.NP:] = PN.SD_POSE          # phi0's entry is 0: swept, not jittered
    p = np.repeat(np.stack([PN.mean_params(pr, cfg) for pr in progs]),
                  n_restart, axis=0)
    for j in range(len(reps)):
        k = j % n_restart
        if k:
            p[j] += rng.normal(0, jitter, PN.nparam(cfg)) * sd
        # the sweep: equally spaced headings, then jitter inside the gap
        p[j, cfg.NMOVE * PN.NP + PN.GI["phi0"]] = (
            2 * np.pi * k / n_restart
            + (rng.normal(0, 0.4 / n_restart) if k else 0.0))
    return enc, np.asarray(PN.scale_to_page(p, enc, cfg, page), np.float32)


def parts(p, enc, tgt, ink, cfg=PN.PEN):
    """(weighted mse, flat mse) from ONE render.

    `w = 1 + ink*target` is the weighting from `[[ink-weighted-loss]]`.  Flat
    MSE counts a background pixel as dearly as an ink pixel, and a digit is
    mostly background, so the cheapest way to improve a misaligned stroke can
    be to draw LESS of it.  Both numbers come back because the two
    disagreeing is the entire reason `ink` exists; `ink = 0` makes them equal.
    """
    e = (PN.alpha(p, enc, cfg, down=cfg.H // tgt.shape[-1]) - tgt) ** 2
    w = 1.0 + ink * tgt
    return jnp.mean(w * e, (1, 2)) / jnp.mean(w, (1, 2)), jnp.mean(e, (1, 2))


def mse(p, enc, tgt, ink=0.0, cfg=PN.PEN):
    return parts(p, enc, tgt, ink, cfg)[0]


def fit(p0, enc, tgt, ink, cfg=PN.PEN, n_iter=300, lr=0.05):
    """Plain Adam in one scan.  A masked-off parameter has gradient exactly
    0.0, so Adam's update for it is 0/(0 + eps) = 0 and it never moves.

    `lr` is NOT a static argument: it only enters arithmetic, and making it
    static would buy a fresh ~10 s XLA compile for every learning rate.
    """
    def step(carry, _):
        p, m, v, t = carry
        gr = jax.grad(lambda q: jnp.sum(mse(q, enc, tgt, ink, cfg)))(p)
        t = t + 1
        m = 0.9 * m + 0.1 * gr
        v = 0.999 * v + 0.001 * gr ** 2
        mh, vh = m / (1 - 0.9 ** t), v / (1 - 0.999 ** t)
        return (p - lr * mh / (jnp.sqrt(vh) + 1e-8), m, v, t), None

    z = jnp.zeros_like(p0)
    (p, *_), _ = jax.lax.scan(step, (p0, z, z, 0), None, length=n_iter)
    return (p,) + parts(p, enc, tgt, ink, cfg)


_fit = jax.jit(fit, static_argnums=(4, 5))


def run(progs, tgt, cfg=PN.PEN, n_restart=8, n_iter=300, lr=0.05, seed=0,
        chunk=512, ink=0.0):
    """Fit every program to its own target, keeping the best restart.

    `tgt` is (len(progs), n, n).  Selection is by the loss actually optimised;
    the flat MSE of the winner comes back too.

    Chunks are padded by tiling, the idiom `search.chunked` documents: a
    ragged final chunk is a second compiled shape, and a compile costs more
    than the chunk it serves.  But padding a 40-item tail up to 512 buys that
    one compile with 472 wasted fits, so the chunk count is fixed first and
    the size balanced across it -- 552 items at chunk 512 become 2 x 276, one
    shape and nothing wasted, instead of 512 + 512.
    """
    rng = np.random.default_rng(seed)
    enc, p0 = init(progs, rng, cfg, n_restart)
    t = np.repeat(np.asarray(tgt, np.float32), n_restart, axis=0)
    n = len(p0)
    chunk = -(-n // max(1, -(-n // chunk)))
    out_p = np.empty_like(p0)
    out_m = np.empty(n, np.float32)
    out_f = np.empty(n, np.float32)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        pad = chunk - (e - s)
        take = lambda a: (a[s:e] if pad == 0 else
                          np.concatenate([a[s:e], np.repeat(a[s:s + 1], pad, 0)]))
        sl = {k: jnp.asarray(take(np.asarray(v))) for k, v in enc.items()}
        p, m, f = _fit(jnp.asarray(take(p0)), sl, jnp.asarray(take(t)),
                       ink, cfg, n_iter, lr)
        out_p[s:e] = np.asarray(p)[:e - s]
        out_m[s:e] = np.asarray(m)[:e - s]
        out_f[s:e] = np.asarray(f)[:e - s]
    shape = (len(progs), n_restart)
    p = out_p.reshape(shape + (-1,))
    m, f = out_m.reshape(shape), out_f.reshape(shape)
    k = m.argmin(1)
    i = np.arange(len(progs))
    return p[i, k], m[i, k], f[i, k]


# ------------------------------------------------------------------- scoring

def iou(a, t, thr=0.5, core=None):
    """Overlap, via the repo's own `svhn_data.iou`.

    `core=CORE` restricts both to the middle of the crop.  An SVHN crop is
    centred on its focal digit and usually carries a slice of its neighbours
    at the edges, and a pen program draws ONE glyph -- so the neighbours are
    in the residual by construction and the full-frame number measures the
    crop as much as the fit.
    """
    return F.iou(np.asarray(a), np.asarray(t), thr, core)


def edge_ink(t, core=CORE):
    """Fraction of the target's ink outside the core window -- i.e. how much
    of this crop is somebody else's digit."""
    t = np.asarray(t, np.float64)
    tot = t.reshape(len(t), -1).sum(1)
    return 1.0 - (t * core[None]).reshape(len(t), -1).sum(1) / np.maximum(tot, 1e-6)


def r2(a, t):
    """Against the variance of the INK ESTIMATE, not of the 3-channel photo.
    The studies this grew out of reported R2 the second way, so the numbers
    carry the same name and a different denominator and are not comparable
    across the two."""
    v = t.reshape(len(t), -1).var(1)
    e = ((a - t) ** 2).reshape(len(t), -1).mean(1)
    return 1.0 - e / np.maximum(v, 1e-9)


def mdl(progs, p, m, cfg=PN.PEN, n=32 * 32):
    """Description length in nats: the residual, the structure, the numbers.

    (n/2) log mse is the Gaussian likelihood term, so a comparison WITHIN an
    image is (n/2) log(mse_A / mse_B) -- a ratio, which is what makes it
    usable against SVHN's irreducible texture floor.  Never compare across
    images.  `n` is the ink map's pixel count, where `search.N_RES` counts
    three colour channels.
    """
    res = 0.5 * n * np.log(np.maximum(m, 1e-12))
    nat = np.array([PN.cost(pr, p[i], cfg) for i, pr in enumerate(progs)])
    return np.column_stack([res, nat])


POLISH_K, POLISH_RESTART, POLISH_ITER = 8, 64, 1200


def polish(progs, p, m, tgt, cfg=PN.PEN, seed=0, k=POLISH_K):
    """Refit the best `k` structures hard, then rank.

    Measured: at the search's own budget (8 restarts x 400 iters) the
    GENERATING structure of a noiseless target sits at median mse 2.2e-07
    when it can reach 3.7e-10.  Through (n/2) log mse that slack is a median
    3429 nats -- against a prior that prices a whole extra move at 3.2.  So
    without this the ranking between two structures is decided by which one
    Adam happened to optimise better, by a factor of a thousand, and the
    symptom was two identical runs disagreeing about a target because the
    fitter found a better optimum for the mirror than for the truth.

    Only the finalists are refit, so the cost is a handful of structures
    rather than the whole hypothesis space.
    """
    L = mdl(progs, p, m, cfg).sum(1)
    top = np.argsort(L)[:k]
    sub = [progs[i] for i in top]
    t = np.repeat(np.asarray(tgt, np.float32)[None], len(sub), 0)
    p2, _, m2 = run(sub, t, cfg, POLISH_RESTART, POLISH_ITER, 0.05, seed=seed)
    p, m = np.array(p, copy=True), np.array(m, copy=True)
    for j, i in enumerate(top):           # keep the better of the two fits
        if m2[j] < m[i]:
            p[i], m[i] = p2[j], m2[j]
    return p, m


def blank_mse(tgt):
    """What drawing NOTHING costs.  Study 1 found 21% of runs converge to this
    against a single stroke on an empty field; `[[ink-weighted-loss]]` found
    flat MSE's own optimum on a faint target IS the blank image.  So this is
    not a curiosity, it is the number a fit has to beat to mean anything."""
    return (np.asarray(tgt, np.float64) ** 2).reshape(len(tgt), -1).mean(1)
