"""Step 4: the proposer's training examples, manufactured by the sampler.

One sampled program is not one example.  It is one example per PREFIX: the
proposer is asked, at every point of a drawing, which move comes next, so a
program of four moves is five decisions (four moves and a stop).  Every one of
them is free, because the suffix is known -- that is the whole reason this
stage needs no RL and no rollouts.

                        what an example consists of

    x       5 x 64 x 64   target, canvas, pen dot, cos phi, sin phi
    tip     the pen's position in pixels, for the trunk's bilinear readout
    act     (kind, angle, length), the move the generator actually made
    mask    which of those are legal here, from `dsl.legal` evaluated
            incrementally -- masked, not merely trained against, because an
            illegal action and a legal one that draws the same picture split
            the particle set for nothing
    val     IoU(the generator's completed drawing, the target)

Three things are worth stating plainly, because each is a decision and not a
detail.

THE CANVAS IS PLACED BY THE WHOLE PROGRAM, NOT BY ITS PREFIX.  `place`
centres a glyph on its own inked centroid, so rendering a 2-move prefix on its
own would put those two moves in the middle of the frame, which is not where
they will be when the drawing is finished.  `alpha(..., sel=)` exists exactly
for this: it masks the ink AFTER placement, so each prefix shows up where it
really sits.  Getting this wrong would teach the policy that the drawing
slides around as it is drawn.

THE POSE IS THE GENERATOR'S.  At training time tx, ty, log s, phi0 come from
the program that made the target.  At inference nothing knows them until the
one-shot scene head of step 8 regresses them from the target.  So this module
assumes that head: until it exists, SMC has to be given a pose.  That is a
real dependency and it is the reason step 8 is not optional.

A TARGET IS NOT ALWAYS THE PREFIX'S OWN.  If it were, the value would be 1.0
on every example and the head would learn the constant function.  Each example
draws its target from one of three modes:

    matched      the prefix's own program.  value 1.0 by construction
    divergent    a program that AGREES with this prefix and parts after it,
                 sampled by continuing the prefix from the prior.  This is
                 the hard case and the one SMC actually faces: the canvas is
                 consistent with the target so far, and the question is
                 whether the completion still fits
    unrelated    another program's target.  The easy negative

Risk 2 -- "is a Markov value enough?" -- is answered by whether a head trained
on this separates the three.  If it cannot tell divergent from matched, the
canvas does not carry enough of the past and the policy will propose
incoherent continuations.  That test is step 6, and this module is what makes
it possible to run it before any policy exists.

Usage: .venv/bin/python -m pen.data [n_prog] [seed]
"""

import sys

import numpy as np

import imageio as IO

from . import dsl as PN
from . import fit as PF

CFG = PN.PEN
OUT = "out_shots"

KINDS = ("d", "t", "h", "s")              # draw, turn, lift, stop
KI = {k: i for i, k in enumerate(KINDS)}
AI = {a: i for i, a in enumerate(PN.ANGLE)}
NK, NA, NL = len(KINDS), len(PN.ANGLE), 8
NCH = 5
MODES = ("matched", "divergent", "unrelated")

# Buckets are uniform in log SPAN, not in log arc length: span is what the
# same number means for a line and for a loop (see `dsl.ext_frac`), and the
# head's job is to land in the right basin, not to be exact.
SPAN_LO, SPAN_HI = 0.30, 2.00             # x LREF, for draws
HOP_LO, HOP_HI = 0.20, 1.00               # x HOPMAX, for lifts
DOT_SD = 2.0                              # px, the pen-position blob

# One float per angle, as `dsl.MU_LEN_OF` is: ext_frac is a jnp function and
# costs ~90 us a call on a 0-d array.
EXT_OF = {a: float(PN.ext_frac(PN.ANGLE[a] * np.pi)) for a in PN.ANGLE}


def edges(cfg=CFG):
    """(draw bin edges, lift bin edges) in pixels."""
    ln = lambda lo, hi: np.exp(np.linspace(np.log(lo), np.log(hi), NL + 1))
    return (ln(SPAN_LO * cfg.LREF, SPAN_HI * cfg.LREF),
            ln(HOP_LO * cfg.HOPMAX, HOP_HI * cfg.HOPMAX))


def bucket(v, e):
    return int(np.clip(np.searchsorted(e, v) - 1, 0, NL - 1))


def centre(k, e):
    """The bin's representative length, geometric because the bins are."""
    return float(np.sqrt(e[k] * e[k + 1]))


def q_of(k, a, cfg=CFG):
    """Bucket -> the `log len` parameter Adam starts from.

    The inverse of the bucketing, and the only place the proposer touches a
    continuous number: it picks a bin, this turns the bin into an init, and
    gradient descent does the rest.
    """
    ed, eh = edges(cfg)
    if a is None:                                     # a lift
        d = min(max(centre(k, eh), 1e-3), cfg.HOPMAX - 1e-3)
        return float(np.log(d / (cfg.HOPMAX - d)))    # logit, `path` sigmoids
    return float(np.log(centre(k, ed) / (cfg.LREF * EXT_OF[a])))


def actions(prog, p, cfg=CFG):
    """The generator's own moves as (kind, angle, length), plus the stop.

    -1 marks a factor the kind does not have: a turn has no length (it does
    not travel), a lift no angle (it does not bend).  The head never scores
    those, and `-1` rather than 0 is so a bug shows up as an index error
    instead of as a plausible wrong label.
    """
    ed, eh = edges(cfg)
    out = []
    for i, m in enumerate(prog):
        k, q = PN.kind(m), float(p[i * PN.NP])
        if k == "d":
            out.append((KI["d"], AI[m[1]], bucket(cfg.LREF * np.exp(q) * EXT_OF[m[1]], ed)))
        elif k == "t":
            out.append((KI["t"], AI[m[1]], -1))
        else:
            out.append((KI["h"], -1, bucket(cfg.HOPMAX / (1.0 + np.exp(-q)), eh)))
    out.append((KI["s"], -1, -1))
    return np.array(out, np.int32)


def _completable(prefix, m, cfg):
    """Can `prefix + m` still be finished?

    `legal` demands a program end on a draw, so a turn or a lift is only
    allowed if there is still room for the draw its group owes.  Asking the
    question this way means the mask is `legal` itself rather than a second,
    drifting copy of its rules.
    """
    q = prefix + (m,)
    if PN.kind(m) == "d":
        return PN.legal(q, cfg)
    return any(PN.legal(q + (PN.draw(a),), cfg) for a in ("0", "+90"))


def masks(prefix, cfg=CFG):
    """(kind, angle, length) legality at this point, as float 0/1 arrays.

    The angle mask is conditional on the kind, so it comes back as (NK, NA) --
    the head is autoregressive and reads the row for the kind it chose.
    """
    km, am = np.zeros(NK, np.float32), np.zeros((NK, NA), np.float32)
    for a in PN.DRAWS:
        am[KI["d"], AI[a]] = _completable(prefix, PN.draw(a), cfg)
    for a in PN.SPINS:
        am[KI["t"], AI[a]] = _completable(prefix, PN.spin(a), cfg)
    km[KI["d"]] = am[KI["d"]].any()
    km[KI["t"]] = am[KI["t"]].any()
    km[KI["h"]] = _completable(prefix, PN.HOP, cfg)
    km[KI["s"]] = PN.legal(prefix, cfg)
    lm = np.zeros((NK, NL), np.float32)
    lm[KI["d"]] = lm[KI["h"]] = 1.0                   # every bin is reachable
    return km, am, lm


def corpus(rng, n, cfg=CFG, frac=0.72):
    """`n` programs from the prior, with parameters, scaled to the page."""
    progs = [PN.sample_struct(rng, cfg) for _ in range(n)]
    p = np.stack([PN.sample_params(rng, q, cfg) for q in progs])
    return progs, np.asarray(PN.scale_to_page(p, PN.encode(progs, cfg), cfg, frac))


def _sel(progs, ts, cfg):
    """(E, NMOVE) ink switch: the first `t` moves on, the rest off."""
    s = np.zeros((len(ts), cfg.NMOVE), np.float32)
    for j, t in enumerate(ts):
        s[j, :t] = 1.0
    return s


def render(progs, p, cfg=CFG, sel=None, down=1):
    return np.asarray(PN.alpha(p, PN.encode(progs, cfg), cfg, down=down, sel=sel))


def state(progs, p, cfg=CFG):
    """Pen tip (E, 2) in pixels and heading (E,) before EACH move, plus the
    state after the last one, which is where `stop` is decided.

    The tip at t = 0 is the first sample of the first move rather than the
    pen's exact start: `path` samples u = 1/SPS .. 1, so the start point is
    not in the array.  That is at most LREF/SPS ~ 1.3 px, against a position
    blob of sd 2 px, and it costs a copy of `place`'s algebra to do better.
    """
    enc = PN.encode(progs, cfg)
    P = PN.path(p, enc, cfg)
    w = PN.weights(P, enc)
    Pp, _ = PN.place(P, w, p, cfg)
    Pp = np.asarray(Pp)
    _, phi = PN.pose(p, enc, cfg)
    phi = np.asarray(phi)
    tip = np.concatenate([Pp[:, :1, 0], Pp[:, :, -1]], 1)          # (n, NMOVE+1, 2)
    hd = np.concatenate([p[:, cfg.NMOVE * PN.NP + PN.GI['phi0']][:, None], phi], 1)
    return tip, hd


def channels(tgt, canvas, tip, hd, cfg=CFG):
    """(E, 5, H, H): target, canvas, a blob at the pen, cos phi, sin phi."""
    e, h = len(tgt), cfg.H
    ax = np.arange(h, dtype=np.float32) + 0.5
    dx = ax[None, None, :] - tip[:, 0, None, None]
    dy = ax[None, :, None] - tip[:, 1, None, None]
    dot = np.exp(-(dx ** 2 + dy ** 2) / (2.0 * DOT_SD ** 2)).astype(np.float32)
    one = np.ones((e, h, h), np.float32)
    return np.stack([tgt, canvas, dot,
                     np.cos(hd)[:, None, None] * one,
                     np.sin(hd)[:, None, None] * one], 1)


def batch(rng, n_prog=64, cfg=CFG, mix=(0.4, 0.4, 0.2)):
    """One training batch: every prefix of `n_prog` sampled programs.

    Rendering is batched three times -- the prefixes, the generators' finished
    drawings, and the divergent targets -- rather than once per example, for
    the reason `fit.run` gives: a new shape is a new XLA compile.
    """
    progs, p = corpus(rng, n_prog, cfg)
    full = render(progs, p, cfg)                        # the generators, finished
    full32 = render(progs, p, cfg, down=2)

    ip = np.concatenate([[i] * (len(q) + 1) for i, q in enumerate(progs)])
    ts = np.concatenate([np.arange(len(q) + 1) for q in progs])
    e = len(ts)
    pe, ge = p[ip], [progs[i] for i in ip]
    canvas = render(ge, pe, cfg, sel=_sel(ge, ts, cfg))
    tipa, hda = state(progs, p, cfg)
    tip, hd = tipa[ip, ts], hda[ip, ts]

    mode = rng.choice(len(MODES), e, p=np.asarray(mix) / np.sum(mix))
    tgt = full[ip].copy()
    tgt32 = full32[ip].copy()

    # divergent: continue THIS prefix from the prior, keeping the glyph pose
    # and the shared moves' numbers, so the two drawings differ only where
    # the programs do.
    dv = np.flatnonzero(mode == 1)
    if len(dv):
        dp, dq = [], []
        for j in dv:
            pre = ge[j][:ts[j]]
            alt = PN.sample_struct(rng, cfg, pre)
            q = pe[j].copy()
            fresh = PN.sample_params(rng, alt, cfg)
            q[ts[j] * PN.NP:cfg.NMOVE * PN.NP] = fresh[ts[j] * PN.NP:cfg.NMOVE * PN.NP]
            dp.append(alt)
            dq.append(q)
        dq = np.stack(dq)
        dq = np.asarray(PN.scale_to_page(dq, PN.encode(dp, cfg), cfg))
        tgt[dv] = render(dp, dq, cfg)
        tgt32[dv] = render(dp, dq, cfg, down=2)

    un = np.flatnonzero(mode == 2)
    if len(un):
        other = (ip[un] + rng.integers(1, n_prog, len(un))) % n_prog
        tgt[un], tgt32[un] = full[other], full32[other]

    act = np.stack([actions(ge[j], pe[j], cfg)[ts[j]] for j in range(e)])
    km, am, lm = zip(*[masks(ge[j][:ts[j]], cfg) for j in range(e)])
    val = PF.iou(full32[ip], tgt32).astype(np.float32)
    return dict(x=channels(tgt, canvas, tip, hd, cfg), tip=tip.astype(np.float32),
                act=act, kmask=np.stack(km), amask=np.stack(am),
                lmask=np.stack(lm), val=val, mode=mode.astype(np.int32),
                prog=ge, par=pe, t=ts)


def _lastpar(b):
    """The parameter vectors of the examples whose prefix is the whole program.
    `batch` does not return them, so main() re-derives them from the corpus it
    was given -- only the check needs them."""
    return b["par"][b["t"] == np.array([len(q) for q in b["prog"]])]


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    rng = np.random.default_rng(seed)
    b = batch(rng, n)
    e = len(b["val"])
    print(f"{n} programs -> {e} examples, {e / n:.1f} decisions per program")
    print(f"x {b['x'].shape} {b['x'].dtype}, "
          f"{b['x'].nbytes / 2 ** 20:.0f} MiB per batch")

    k = b["act"][:, 0]
    print("\n  kind      share   legal share   mean value")
    for i, nm in enumerate(("draw", "turn", "lift", "stop")):
        s = k == i
        print(f"  {nm:8s}  {s.mean():.2f}       {b['kmask'][:, i].mean():.2f}"
              f"          {b['val'][s].mean() if s.any() else float('nan'):.2f}")

    print("\n  target mode   share   value: mean   sd    min")
    for i, nm in enumerate(MODES):
        s = b["mode"] == i
        v = b["val"][s]
        print(f"  {nm:12s}  {s.mean():.2f}      {v.mean():.3f}  {v.std():.3f}  "
              f"{v.min():.3f}")
    m, d = b["val"][b["mode"] == 0], b["val"][b["mode"] == 1]
    print(f"\n  matched vs divergent: {m.mean():.3f} vs {d.mean():.3f} "
          f"-- gap {m.mean() - d.mean():+.3f}, and {(d > 0.9).mean():.0%} of "
          f"divergent targets are still >= 0.9")
    print("  (that overlap is the ceiling on what ANY Markov value can score,"
          "\n   and the number risk 2 is argued from)")

    # invariants, here rather than in check.py because they are about the
    # BATCH and not about the DSL
    last = np.array([b["t"][i] == len(b["prog"][i]) for i in range(e)])
    assert (b["act"][last, 0] == KI["s"]).all(), "last decision must be stop"
    assert (b["act"][~last, 0] != KI["s"]).all(), "stop only at the end"
    assert b["kmask"][np.arange(e), b["act"][:, 0]].all(), "took an illegal kind"
    an = b["act"][:, 1] >= 0
    assert b["amask"][np.arange(e)[an], b["act"][an, 0], b["act"][an, 1]].all(), \
        "took an illegal angle"
    assert (b["val"][b["mode"] == 0] > 0.999).all(), "matched value must be 1"
    assert not b["kmask"][b["t"] == 0, KI["s"]].any(), "stopped before drawing"
    # the one that proves `sel` placement: a full-length prefix IS the drawing,
    # and an empty one is blank.  Render the prefix on its own instead and this
    # fails by ~10 px of translation.
    whole = render([b["prog"][j] for j in np.flatnonzero(last)],
                   _lastpar(b), CFG)
    assert np.abs(b["x"][last, 1] - whole).max() < 1e-5, "prefix placement drifts"
    assert b["x"][b["t"] == 0, 1].max() < 1e-6, "empty prefix is not blank"
    print("\n  invariants: stop only at the end, every taken action legal,"
          " matched value 1.0 -- all hold")

    sh = np.argsort(-b["val"] * (b["mode"] == 1))[:4]
    pick = list(sh) + list(np.flatnonzero(b["mode"] == 0)[:4])
    tiles = [np.concatenate([IO.ink(b["x"][j, 0]), IO.ink(b["x"][j, 1]),
                             IO.ink(b["x"][j, 2]),
                             IO.overlay(b["x"][j, 1], b["x"][j, 0])], axis=1)
             for j in pick]
    IO.save(f"{OUT}/data_prefixes.png", IO.grid(tiles, 2), 2)
    print(f"\nwrote {OUT}/data_prefixes.png  (target | canvas | pen | overlay;"
          " top row divergent, bottom matched)")


if __name__ == "__main__":
    main()
