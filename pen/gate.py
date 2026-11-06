"""The gate: can a hand-picked pen program fit a real SVHN digit at all?

No search.  For each crop we take the digit's label, write down one or two
ways a person might draw it, and fit only the numbers.  If that cannot reach
a decent overlap, the DSL is too rigid and no proposer, no SMC and no library
fixes it -- so this runs before any of them get built.

Each hand program is fitted together with its MIRROR (every angle negated).
Not to widen the search: to keep my own sign conventions out of the result.
A mirrored `2` is not a `2`, so if mirrors keep winning, the programs are
written wrong and the gate is measuring my handwriting, not the DSL.

Usage: .venv/bin/python -m pen.gate [ink_weight]
"""

import sys
import time

import numpy as np

import imageio as IO
import svhn_data as F

from . import dsl as PN
from . import fit as PF

CFG = PN.PEN
OUT = "out_shots"
N_RESTART = 12
N_ITER = 400
LR = 0.05
INK = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0

# Written in `pen.show`'s own notation, which `pen.read` parses back: "-" or
# "0" a straight, "(a)" a turn in place, "^" a pen-up hop, anything else an
# arc.  Positive angles are clockwise on screen (the canvas y axis runs down).
HAND = {
    0: [("+360",)],
    1: [("0",), ("+45",)],
    2: [("+180", "(+45)", "0", "(-135)", "0"), ("+270", "(-135)", "0")],
    # A 3 is NOT two half-arcs chained: with no turn between them the
    # curvature continues and 180+180 is a circle.  The pen ends the top bowl
    # heading back the way it came, so it must turn before the second.
    3: [("+180", "(+180)", "+180"), ("+270", "(+135)", "+270")],
    4: [("0", "(+90)", "0", "^", "0"), ("+45", "(+90)", "0", "^", "0")],
    5: [("0", "(+90)", "0", "(-45)", "+270"), ("0", "(+90)", "+270")],
    6: [("+180", "+360"), ("+90", "+360")],
    7: [("0", "(+135)", "0"), ("0", "(+135)", "+45")],
    8: [("+360", "+360"), ("+360", "-360")],
    9: [("+360", "(+45)", "0"), ("-360", "+180")],
}


HANDP = {k: [PN.read(t) for t in v] for k, v in HAND.items()}


def main():
    d = F.load()
    tgt = np.asarray(d["ink"], np.float32)
    lab = np.asarray(d["y"]).reshape(-1).astype(int)
    n = len(lab)

    # one flat batch: every candidate for every crop, each with its mirror.
    # Whether a candidate IS the mirror is known here, so it is carried rather
    # than re-derived by membership tests later.
    cand, is_mir, counts = [], [], []
    for i in range(n):
        for pr in HANDP[lab[i]]:
            for q, mm in ((pr, False), (PN.mirror(pr), True)):
                assert PN.legal(q, CFG), PN.show(q)
                cand.append(q)
                is_mir.append(mm)
        counts.append(2 * len(HANDP[lab[i]]))
    owner = np.repeat(np.arange(n), counts)
    is_mir = np.array(is_mir)
    edges = np.cumsum([0] + counts)
    print(f"{n} crops, {len(cand)} candidate fits "
          f"x {N_RESTART} restarts = {len(cand) * N_RESTART} fits, ink {INK}")

    t0 = time.time()
    p, m, fm = PF.run(cand, tgt[owner], CFG, N_RESTART, N_ITER, LR,
                      seed=0, ink=INK)
    wall = time.time() - t0
    print(f"wall {wall:.1f}s, {wall / (len(cand) * N_RESTART) * 1000:.0f} ms/fit "
          f"(includes one XLA compile)")

    best = np.array([s + m[s:e].argmin() for s, e in zip(edges, edges[1:])])
    bp, bf, mir = p[best], fm[best], is_mir[best]
    bprog = [cand[j] for j in best]

    a = np.asarray(PN.alpha(bp, PN.encode(bprog, CFG), CFG, down=2))
    bl = PF.blank_mse(tgt)
    R2, IU = PF.r2(a, tgt), PF.iou(a, tgt)
    IC, ED = PF.iou(a, tgt, core=PF.CORE), PF.edge_ink(tgt)
    ML = PF.mdl(bprog, bp, bf, CFG)

    print()
    print("  idx  lab   mse    blank    R2    IoU  IoUc  edge  mir  program")
    for i in range(n):
        print(f" {d['idx'][i]:6d}  {lab[i]}  {bf[i]:.4f}  {bl[i]:.4f}  "
              f"{R2[i]:+.2f}  {IU[i]:.2f}  {IC[i]:.2f}  {ED[i]:.2f}  "
              f"{'yes' if mir[i] else ' - '}  {PN.show(bprog[i])}")

    print()
    print(f"blank beats the fit on {int((bf >= bl).sum())}/{n} crops")
    print(f"IoU   median {np.median(IU):.2f} full frame, "
          f"{np.median(IC):.2f} centred;  >= 0.5 on {int((IU >= 0.5).sum())}/{n} "
          f"and {int((IC >= 0.5).sum())}/{n}")
    print(f"edge  {100 * ED.mean():.0f}% of target ink is outside the centre, "
          f"i.e. a neighbouring digit the program cannot draw")
    print(f"R2    median {np.median(R2):+.2f}")
    print(f"mirror won on {int(mir.sum())}/{n}")
    print(f"MDL   residual {ML[:, 0].mean():7.1f}  structure {ML[:, 1].mean():5.1f}  "
          f"params {ML[:, 2].mean():5.1f} nats")

    tiles = [np.concatenate([IO.ink(tgt[i]), IO.ink(a[i]),
                             IO.overlay(a[i], tgt[i])], axis=1) for i in range(n)]
    IO.save(f"{OUT}/gate_pen.png", IO.grid(tiles, 4), 3)
    print(f"\nwrote {OUT}/gate_pen.png  (ink | render | overlay, per crop)")


if __name__ == "__main__":
    main()
