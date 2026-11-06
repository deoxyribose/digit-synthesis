"""Draw from the pen prior and look at it.

Three sheets:
  pen_alphabet.png   every draw primitive and every turn, from one pen state
  pen_random.png     programs sampled from the prior, nothing fitted
  pen_strokes.png    the same samples with each stroke in its own colour, so
                     the chaining and the pen-up hops are visible

The samples are not digits and are not meant to be.  The question this
answers is the one before that: does the prior put its mass on things that
look like something a pen did?

Usage: .venv/bin/python -m pen.sample [seed]
"""

import sys

import numpy as np

import imageio as IO

from . import dsl as PN

CFG = PN.PEN
H = CFG.H
OUT = "out_shots"
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 3
WANT = 140
BATCH = 64
COLS = [(214, 69, 45), (40, 90, 200), (28, 140, 70), (170, 80, 190),
        (200, 150, 20), (90, 90, 90), (20, 160, 170), (190, 60, 120)]


def placed(progs, ps):
    """(points, weights) in canvas pixels, for a whole batch at once."""
    enc = PN.encode(progs, CFG)
    p = np.asarray(np.stack(ps), np.float32)
    P = PN.path(p, enc, CFG)
    return PN.place(P, PN.weights(P, enc), p, CFG) + (enc, p)


def render(progs, ps, sel=None):
    return np.asarray(PN.alpha(np.asarray(np.stack(ps), np.float32),
                               PN.encode(progs, CFG), CFG, sel=sel))


def dot(img, xy, rgb, r=2):
    x, y = int(round(float(xy[0]))), int(round(float(xy[1])))
    y0, y1 = max(0, y - r), min(H, y + r + 1)
    x0, x1 = max(0, x - r), min(H, x + r + 1)
    img[y0:y1, x0:x1] = rgb


# ---------------------------------------------------------------- alphabet
# Row 1: every draw.  Row 2: every turn, shown between two straights, plus
# the hop -- a turn draws nothing by itself, which is the point of it.
progs = [(PN.draw(a),) for a in PN.DRAWS] \
    + [(PN.draw("0"), PN.spin(a), PN.draw("0")) for a in PN.SPINS] \
    + [(PN.draw("0"), PN.HOP, PN.draw("0"))]
ps = [PN.mean_params(pr, CFG, hop=0.5) for pr in progs]
a = render(progs, ps)
P, _, _, _ = placed(progs, ps)
P = np.asarray(P)
tiles = []
for i, pr in enumerate(progs):
    t = IO.ink(a[i])
    dot(t, P[i, 0, 0], (30, 120, 220))             # where the pen went down
    dot(t, P[i, len(pr) - 1, -1], (220, 60, 40))   # where it came up
    tiles.append(t)
IO.save(f"{OUT}/pen_alphabet.png", IO.grid(tiles, len(PN.DRAWS)), 3)
print(f"alphabet    {len(PN.DRAWS)} draws + {len(PN.SPINS)} turns + 1 hop = "
      f"{PN.nsymbol()} symbols")
print("  draws:   ", " ".join(PN.DRAWS))
print("  turns:   ", " ".join(PN.SPINS))

# ---------------------------------------------------------------- samples
# Batched: one render per BATCH programs rather than one per program, since
# every structure goes through the same compiled kernel anyway.
rng = np.random.default_rng(SEED)
keep, kp, bad = [], [], 0
while len(keep) < WANT:
    batch = [PN.sample(rng, CFG) for _ in range(BATCH)]
    # `sample_struct` cannot emit an illegal program -- it samples from the
    # same restricted sets `legal` checks -- so this asserts the contract
    # rather than filtering on it.
    assert all(PN.legal(pr, CFG) for pr, _ in batch)
    enc = PN.encode([pr for pr, _ in batch], CFG)
    p0 = np.stack([x for _, x in batch]).astype(np.float32)
    # `path` does not read `log s` -- only `place` does -- so the arclength
    # weights are the same before and after rescaling, and computing them
    # here saves `scale_to_page`'s work being redone.
    w = np.asarray(PN.weights(PN.path(p0, enc, CFG), enc)).reshape(len(batch), -1)
    q = np.asarray(PN.scale_to_page(p0, enc, CFG), np.float32)
    for j, (pr, _) in enumerate(batch):
        if len(keep) >= WANT:
            break
        if np.isfinite(q[j]).all() and w[j].sum() >= 8.0:
            keep.append(pr)
            kp.append(q[j])
        else:
            bad += 1       # degenerate: too little ink to be a drawing
print(f"\nsampled    {WANT} kept, {bad} rejected as degenerate "
      f"({100.0 * WANT / (WANT + bad):.0f}% usable)")
nmv = np.array([len(x) for x in keep])
ndr = np.array([PN.ndraw(x) for x in keep])
nst = np.array([PN.nstroke(x) for x in keep])
print(f"moves/prog {nmv.mean():.2f} mean, {nmv.max()} max   "
      f"draws/prog {ndr.mean():.2f}   strokes/prog {nst.mean():.2f} mean, "
      f"{nst.max()} max")
print(f"hops       {100.0 * (nst > 1).mean():.0f}% of programs have one   "
      f"turns      {100.0 * (nmv - ndr - (nst - 1) > 0).mean():.0f}%")
cs = np.array([PN.cost(x, y, CFG) for x, y in zip(keep, kp)])
print(f"cost       structure {cs[:, 0].mean():.1f} nats, "
      f"parameters {cs[:, 1].mean():.1f} nats")
print("\nfirst twelve programs (- = straight, (a) = turn in place, ^ = hop):")
for i, x in enumerate(keep[:12]):
    print(f"  {cs[i, 0]:5.1f} nats   {PN.show(x)}")

a = render(keep, kp)
IO.save(f"{OUT}/pen_random.png", IO.grid([IO.ink(x) for x in a], 14), 2)

# ------------------------------------------------------- strokes in colour
sub, sp = keep[:24], kp[:24]
enc = PN.encode(sub, CFG)
dn, lv = np.asarray(enc["down"]), np.asarray(enc["live"])
sid = np.cumsum(np.asarray(enc["hop"]), 1) * lv   # a hop ends a stroke
layers = [render(sub, sp, sel=(sid == s).astype(np.float32) * dn)
          for s in range(int(sid.max()) + 1)]
tiles = []
for j in range(len(sub)):
    t = np.full((H, H, 3), 255, np.float32)
    for s, lay in enumerate(layers):
        c = np.array(COLS[s % len(COLS)], np.float32)
        al = lay[j][..., None]
        t = t * (1 - al) + c * al
    tiles.append(t.astype(np.uint8))
IO.save(f"{OUT}/pen_strokes.png", IO.grid(tiles, 8), 3)
print(f"\nwrote {OUT}/pen_alphabet.png  {OUT}/pen_random.png  {OUT}/pen_strokes.png")
