"""Step 7: particles that draw, guided by the trained proposer.

The enumeration of step 3 is the thing this replaces.  It fitted every legal
structure, which is honest and does not scale: 3705 programs of <= 3 moves,
and 21^k beyond that.  Here N particles draw one move at a time, the policy
says what to draw and the value head says which particles to keep, and the
whole search costs N rollouts instead of 21^k fits.

                        what a particle is

A partial program, the page point its pen started at, and its initial
heading.  Nothing else: lengths come from the policy's buckets, which are in
PAGE PIXELS, so the glyph scale can be held at 1 and the drawing is exactly
what the buckets say.  That is why the pose reduces to (origin, phi0) -- and
both of those are genuinely unknown, so particles carry them and the weights
sort them out.  tx, ty never appear: under start-anchored placement they do
not enter, and the final fit re-derives the whole pose anyway.

THE ROLLOUT IS START-ANCHORED and the training canvases were centroid-placed.
Those are the same placement -- `check.py` asserts it to 0.0 alpha -- which
is the only reason a half-drawn program can be rendered at all.  Centroid
placement needs the whole drawing to know where any of it goes, so a rollout
using it would see its own canvas slide across the page at every move.

                          what SMC does here

Weights are the value head, w ∝ v^(1/T), with systematic resampling when the
effective sample size falls below half.  THIS IS A HEURISTIC TWIST, not an
exact posterior: v is a regression on IoU, not an incremental likelihood
ratio, so the particle set is not a consistent estimate of p(program|image).
It does not have to be.  What it has to do is end up with structures worth
fitting, and that is measured here against two controls:

    prior      the same rollout with the PRIOR in place of the policy, same
               particle count, same fitter, same scoring.  If the net does
               not beat this it has learned nothing useful, however good its
               held-out accuracy looked
    oracle     the generator's true (origin, phi0) instead of sampled ones.
               The gap between oracle and sampled is the cost of not having
               the step-8 scene head, isolated from the cost of searching

Structure comes from the search, continuous parameters from the fitter, and
scoring is step 3's: `fit.run` then `fit.mdl`, so the numbers are comparable
with the enumeration's.

Usage: .venv/bin/python -m pen.smc [n_targets] [n_particles] [mode]
       mode: net (default) | prior | both
"""

import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

import imageio as IO

from . import data as DA
from . import dsl as PN
from . import fit as PF
from . import net as NT

CFG = PN.PEN
OUT = "out_shots"
TEMP = 1.0                     # policy sampling temperature
TEMP_W = 1.0                   # weight sharpness, w ∝ v^(1/TEMP_W)
ESS_FRAC = 0.5
N_RESTART, N_ITER = 8, 400     # the fitter, as step 3 ran it


def pose_proposal(rng, tgt, n):
    """`n` guesses at (origin, phi0) for one target.

    The pen starts somewhere on the ink -- that is all we know before the
    scene head exists -- so origins are drawn from the target's own ink as a
    density, and headings uniformly.  Deliberately weak: the point of
    measuring against the oracle pose is to find out how much this costs.
    """
    t = np.asarray(tgt, np.float64)
    h = t.shape[0]
    flat = t.reshape(-1)
    q = flat / max(flat.sum(), 1e-9)
    k = rng.choice(len(q), n, p=q)
    y, x = np.divmod(k, h)
    jit = rng.normal(0, 1.0, (n, 2))
    org = np.stack([x + 0.5, y + 0.5], 1) + jit
    phi = rng.uniform(-np.pi, np.pi, n)
    return org.astype(np.float32), phi.astype(np.float32)


def params(progs, phi, cfg=CFG):
    """Parameter vectors for partial programs: bucket centres and nothing else.

    Lengths are already fixed, in page pixels, by the buckets the policy
    chose -- `q_of` converts at scale 1 -- so this is not an initial guess
    being refined, it is the drawing the particle has actually made.  bend
    and wig stay at zero: the proposer does not propose them and the fitter
    is what finds them.
    """
    p = np.zeros((len(progs), PN.nparam(cfg)), np.float32)
    p[:, cfg.NMOVE * PN.NP + PN.GI['phi0']] = phi
    return p


def set_len(p, i, t, m, k, cfg=CFG):
    """Write move `t`'s length parameter for the bucket the policy picked."""
    a = m[1] if PN.kind(m) == "d" else None
    p[i, t * PN.NP] = DA.q_of(k, a, 1.0, cfg)


def canvases(progs, p, org, cfg=CFG):
    return np.asarray(PN.alpha(jnp.asarray(p), PN.encode(progs, cfg), cfg,
                               anchor=jnp.asarray(org)))


def tips(progs, p, org, cfg=CFG):
    """Pen position and heading after the moves made so far."""
    enc = PN.encode(progs, cfg)
    P = PN.path(jnp.asarray(p), enc, cfg)
    w = PN.weights(P, enc)
    Pp, _ = PN.place(P, w, jnp.asarray(p), cfg, anchor=jnp.asarray(org))
    Pp = np.asarray(Pp)
    _, phi = PN.pose(jnp.asarray(p), enc, cfg)
    phi = np.asarray(phi)
    n = np.array([len(q) for q in progs])
    tip = np.where((n == 0)[:, None], org, Pp[np.arange(len(progs)),
                                              np.maximum(n - 1, 0), -1])
    hd = np.where(n == 0, p[:, cfg.NMOVE * PN.NP + PN.GI['phi0']],
                  phi[np.arange(len(progs)), np.maximum(n - 1, 0)])
    return tip.astype(np.float32), hd.astype(np.float32)


def _pick(rng, logits, mask, temp):
    """Sample one index per row from masked logits at temperature `temp`."""
    z = np.where(mask > 0, logits / temp, -np.inf)
    dead = ~np.isfinite(z).any(1)            # a factor this kind does not have
    z = np.where(dead[:, None], 0.0, z)
    z = z - z.max(1, keepdims=True)
    q = np.exp(z)
    q /= q.sum(1, keepdims=True)
    u = rng.random(len(q))[:, None]
    return (q.cumsum(1) < u).sum(1).clip(0, q.shape[1] - 1)


def _prior_logits(progs, cfg=CFG):
    """The prior's own next-move distribution, as logits, for the control.

    Not a uniform baseline: the sampler's actual conditional, which knows
    P_STOP, P_HOP, P_SPIN and the angle weights.  A policy that cannot beat
    the distribution it was trained to imitate has not learned to look.
    """
    lk = np.zeros((len(progs), NT.NK), np.float32)
    la = np.zeros((len(progs), NT.NA), np.float32)
    ll = np.zeros((len(progs), NT.NL), np.float32)
    for i, q in enumerate(progs):
        gap = [m for m in PN._gap(q, len(q))]
        kinds = [PN.kind(m) for m in gap]
        lk[i, DA.KI["s"]] = np.log(PN.P_STOP)
        lk[i, DA.KI["h"]] = np.log(PN.P_HOP if "h" not in kinds else 1e-9)
        lk[i, DA.KI["t"]] = np.log(PN.P_SPIN)
        lk[i, DA.KI["d"]] = np.log(1.0 - PN.P_STOP)
        for a, v in PN.DP.items():
            la[i, DA.AI[a]] = np.log(v)
        for a, v in PN.SP.items():
            la[i, DA.AI[a]] = max(la[i, DA.AI[a]], np.log(v))
    return lk, la, ll


def rollout(rng, tgt, n, w=None, oracle=None, cfg=CFG, temp=TEMP):
    """Run `n` particles against one target.  Returns the structures seen.

    `w = None` is the prior control.  `oracle` is a (origin, phi0) pair to use
    instead of the proposal.
    """
    if oracle is None:
        org, phi = pose_proposal(rng, tgt, n)
    else:
        # Jittered, not identical.  A pose the particles all share leaves
        # policy sampling as the only diversity and the set collapses -- the
        # first version of this control scored WORSE than the sampled pose it
        # was meant to be an upper bound on.  The spread matches the
        # proposal's, so what is being compared is where the pose is, not how
        # many of them there are.
        org = (np.asarray(oracle[0])[None] + rng.normal(0, 1.0, (n, 2))
               ).astype(np.float32)
        phi = (float(oracle[1]) + rng.normal(0, 0.25, n)).astype(np.float32)
    progs = [()] * n
    p = params(progs, phi, cfg)
    done = np.zeros(n, bool)
    T = np.repeat(np.asarray(tgt, np.float32)[None], n, 0)
    seen, val = {}, np.ones(n, np.float32)

    for _ in range(cfg.NMOVE):
        if done.all():
            break
        cv = canvases(progs, p, org, cfg)
        tip, hd = tips(progs, p, org, cfg)
        km, am, lm = zip(*[DA.masks(q, cfg) for q in progs])
        km, am, lm = np.stack(km), np.stack(am), np.stack(lm)
        x = DA.channels(T, cv, tip, hd, cfg)

        if w is None:
            lk, la, ll = _prior_logits(progs, cfg)
            v = val
        else:
            b = dict(x=jnp.asarray(x), tip=jnp.asarray(tip),
                     act=jnp.zeros((n, 3), jnp.int32), kmask=jnp.asarray(km),
                     amask=jnp.asarray(am), lmask=jnp.asarray(lm))
            e = NT.trunk(w, b["x"], b["tip"])
            lk = np.asarray(NT._mask(e @ w["kind"] + w["kb"], b["kmask"]))
            v = np.asarray(jax.nn.sigmoid(e @ w["val"] + w["vb"])[:, 0])

        kind = _pick(rng, lk, km, temp)
        kind = np.where(done, DA.KI["s"], kind)

        if w is not None:
            ek = np.asarray(w["ek"])[kind]
            la = np.asarray(NT._mask(
                jnp.concatenate([e, jnp.asarray(ek)], 1) @ w["ang"] + w["ab"],
                jnp.asarray(am[np.arange(n), kind])))
        ang = _pick(rng, la, am[np.arange(n), kind], temp)

        if w is not None:
            ea = np.asarray(w["ea"])[ang]
            ll = np.asarray(NT._mask(
                jnp.concatenate([e, jnp.asarray(ek), jnp.asarray(ea)], 1)
                @ w["len"] + w["lb"], jnp.asarray(lm[np.arange(n), kind])))
        lb = _pick(rng, ll, lm[np.arange(n), kind], temp)

        nxt = list(progs)
        for i in range(n):
            if done[i] or kind[i] == DA.KI["s"]:
                done[i] = True
                continue
            k = DA.KINDS[kind[i]]
            m = (PN.HOP if k == "h" else
                 (PN.draw if k == "d" else PN.spin)(list(PN.ANGLE)[ang[i]]))
            t = len(progs[i])
            nxt[i] = progs[i] + (m,)
            if k != "t":
                set_len(p, i, t, m, int(lb[i]), cfg)
        progs = nxt
        val = v

        for i in np.flatnonzero(done):
            if PN.legal(progs[i], cfg):
                seen[progs[i]] = max(seen.get(progs[i], 0.0), float(val[i]))

        # resample on the value, systematically, when the set has collapsed
        q = np.maximum(val, 1e-6) ** (1.0 / TEMP_W)
        q = q / q.sum()
        if 1.0 / np.sum(q ** 2) < ESS_FRAC * n:
            u = (rng.random() + np.arange(n)) / n
            idx = np.searchsorted(q.cumsum(), u).clip(0, n - 1)
            progs = [progs[i] for i in idx]
            p, org, done, val = p[idx], org[idx], done[idx], val[idx]

    for i in range(n):
        if PN.legal(progs[i], cfg):
            seen[progs[i]] = max(seen.get(progs[i], 0.0), float(val[i]))
    return seen


def best(cands, tgt, cfg=CFG, seed=0):
    """Fit every candidate structure and rank by description length.

    Step 3's scoring exactly, so the two searches are comparable.
    """
    cands = list(cands)
    if not cands:
        return None, np.inf, np.inf, None
    t = np.repeat(np.asarray(tgt, np.float32)[None], len(cands), 0)
    p, _, f = PF.run(cands, t, cfg, N_RESTART, N_ITER, 0.05, seed=seed)
    L = PF.mdl(cands, p, f, cfg).sum(1)
    i = int(np.argmin(L))
    return cands[i], float(L[i]), float(f[i]), p[i]


def short_sample(rng, cfg=CFG, nmax=3):
    """A generator from the prior restricted to <= nmax moves.

    The only regime where step 3 has a ground truth to compare against: a
    rejection sample, not a different prior, so the comparison is against
    exactly the 3705 structures `enumerate` searches.
    """
    while True:
        q = PN.sample_struct(rng, cfg)
        if len(q) <= nmax:
            return q, PN.sample_params(rng, q, cfg)


def main():
    n_tgt = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    n_par = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    mode = sys.argv[3] if len(sys.argv) > 3 else "both"
    short = len(sys.argv) > 4 and sys.argv[4] == "short"
    rng = np.random.default_rng(0)

    wts = np.load(f"{OUT}/net_all.npy", allow_pickle=True).item()
    w = {k: jnp.asarray(v) for k, v in wts.items()}

    draw1 = (lambda: short_sample(rng, CFG)) if short else (lambda: PN.sample(rng, CFG))
    gen, gp = zip(*[draw1() for _ in range(n_tgt)])
    gp = np.asarray(PN.scale_to_page(np.stack(gp), PN.encode(gen, CFG), CFG))
    tgt = np.asarray(PN.alpha(jnp.asarray(gp), PN.encode(gen, CFG), CFG))
    org = np.asarray(PN.origin(jnp.asarray(gp), PN.encode(gen, CFG), CFG))
    phi0 = gp[:, CFG.NMOVE * PN.NP + PN.GI['phi0']]

    runs = ([("prior", None, False), ("net", w, False), ("net+pose", w, True)]
            if mode == "both" else
            [("net", w, False)] if mode == "net" else [("prior", None, False)])
    print(f"{n_tgt} targets from the prior, {n_par} particles, "
          f"{PN.nparam(CFG)} parameters fitted per candidate")
    print(f"generators: {np.mean([len(g) for g in gen]):.1f} moves on average\n")

    out = {}
    for name, ww, orac in runs:
        t0 = time.time()
        rows = []
        for i in range(n_tgt):
            o = (org[i], phi0[i]) if orac else None
            seen = rollout(np.random.default_rng(100 + i), tgt[i], n_par,
                           ww, o, CFG)
            b, L, mse, _ = best(seen, tgt[i], CFG, seed=i)
            hit = b == gen[i]
            rows.append((len(seen), hit, L, mse, b))
            print(f"  [{name}] {i + 1}/{n_tgt} {PN.show(gen[i]):22s} -> "
                  f"{PN.show(b) if b else '-':22s} {len(seen):3d} distinct"
                  f"{'  RECOVERED' if hit else ''}", flush=True)
        wall = time.time() - t0
        out[name] = (rows, wall)
        hits = sum(r[1] for r in rows)
        print(f"  {name}: recovered {hits}/{n_tgt}, "
              f"{np.mean([r[0] for r in rows]):.0f} distinct structures per "
              f"target, mse {np.median([r[3] for r in rows]):.2e}, "
              f"{wall / n_tgt:.1f}s per target\n", flush=True)

    print("  run        recovered   distinct   median mse     s/target")
    for name, (rows, wall) in out.items():
        print(f"  {name:10s}   {sum(r[1] for r in rows)}/{n_tgt}      "
              f"{np.mean([r[0] for r in rows]):6.0f}   "
              f"{np.median([r[3] for r in rows]):.3e}    {wall / n_tgt:6.1f}")
    if short:
        from .enumerate import rank_one
        hyps = PN.structures(CFG, nmax=3)
        print(f"\n  against the enumeration: all {len(hyps)} structures of "
              f"<= 3 moves, fitted and ranked by the same MDL")
        t0 = time.time()
        hit = 0
        for i in range(n_tgt):
            order, *_ = rank_one(hyps, tgt[i], seed=i, nmax=3)
            hit += order[0] == gen[i]
            print(f"  [enum] {i + 1}/{n_tgt} {PN.show(gen[i]):18s} -> "
                  f"{PN.show(order[0]):18s}"
                  f"{'  RECOVERED' if order[0] == gen[i] else ''}", flush=True)
        print(f"  enumerate: recovered {hit}/{n_tgt}, "
              f"{(time.time() - t0) / n_tgt:.0f}s per target, against SMC's "
              f"{min(v[1] / n_tgt for v in out.values()):.1f}s")

    print("\n  'prior' is the same search with the sampler in place of the")
    print("  policy -- the control that says whether the net learned to look.")
    print("  'net+pose' gets the generator's true start point and heading,")
    print("  so the gap to 'net' is the price of having no scene head yet.")

    name = "net+pose" if "net+pose" in out else list(out)[0]
    rows = out[name][0]
    tiles = []
    for i in range(min(n_tgt, 8)):
        b = rows[i][4]
        if b is None:
            continue
        _, _, _, pb = best([b], tgt[i], CFG, seed=i)
        a = np.asarray(PN.alpha(pb[None], PN.encode([b], CFG), CFG))[0]
        tiles.append(np.concatenate([IO.ink(tgt[i]), IO.ink(a),
                                     IO.overlay(a, tgt[i])], axis=1))
    IO.save(f"{OUT}/smc.png", IO.grid(tiles, 2), 2)
    print(f"\nwrote {OUT}/smc.png  (target | best found | overlay)")


if __name__ == "__main__":
    main()
