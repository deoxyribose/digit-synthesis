"""Step 6: train on synthetic data only, and the VALUE HEAD FIRST, ALONE.

Risk 2 is that the state is not Markov.  The pen's geometry is -- it cannot
reach back -- but a person draws a 4's crossbar knowing they drew a stem, and
if the canvas does not carry that, the policy will propose continuations that
do not cohere with what is already on the page.  The cheapest possible test is
to train only the value head: it sees exactly what the policy would see, and
if it cannot tell a prefix that leads to THIS target from one that leads
somewhere else, no policy on the same input will do better.

The measurement that settles it is not the value head's MSE, which is a
number without a scale.  It is:

    AUC(predicted)   how often the head ranks a matched example above a
                     divergent one -- same prefix, different completion
    AUC(true IoU)    the same statistic computed on the TRUE values.  This is
                     the ceiling, and it is below 1.0 because two different
                     completions of the same prefix sometimes draw nearly the
                     same picture.  Reporting the first without the second
                     would make an ambiguity in the DATA look like a failure
                     of the MODEL

Everything here is synthetic.  No SVHN crop is touched at this step, by
design: the domain gap is measured at step 9 rather than hidden inside a
training curve.

Usage: .venv/bin/python -m pen.train [value|policy|all] [steps] [seed]
"""

import sys
import time

import numpy as np

import jax

from . import data as DA
from . import net as NT

POOL, REUSE = 8, 20            # batches held at once, steps per refresh
N_EX = 512
HOLD = 4                       # held-out batches, a different seed forever
USE = {"value": (0.0, 0.0, 0.0, 1.0),
       "policy": (1.0, 1.0, 1.0, 0.0),
       "all": (1.0, 1.0, 1.0, 1.0)}


def auc(score, pos, neg):
    """P(score of a positive > score of a negative), ties at half.

    Rank-based rather than threshold-based because the two heads being
    compared -- the network and the oracle -- are on different scales.
    """
    a, b = np.asarray(score)[pos], np.asarray(score)[neg]
    if not len(a) or not len(b):
        return float("nan")
    d = a[:, None] - b[None, :]
    return float((d > 0).mean() + 0.5 * (d == 0).mean())


def baseline(hold):
    """What the PRIOR alone scores: the most frequent action, restricted to
    what is legal here, without looking at the image.

    The control the action accuracies have to be read against.  A head that
    beats nothing but chance has learned the alphabet's frequencies; a head
    worth putting in SMC has to beat this, which already knows the grammar
    and the prior's skew.
    """
    out = []
    for f, n in ((0, NT.NK), (1, NT.NA), (2, NT.NL)):
        md = np.concatenate([np.asarray(b["mode"]) for b in hold])
        y = np.concatenate([np.asarray(b["act"][:, f]) for b in hold])
        mk = np.concatenate([np.asarray(b[("kmask", "amask", "lmask")[f]])
                             for b in hold])
        if f:                                  # angle/length masks are per kind
            k = np.concatenate([np.asarray(b["act"][:, 0]) for b in hold])
            mk = mk[np.arange(len(k)), k]
        ok = (y >= 0) & (md == 0)         # scored where the net is scored
        cnt = np.bincount(y[ok], minlength=n).astype(np.float64)
        pred = (mk[ok] * cnt).argmax(1)
        out.append(float((pred == y[ok]).mean()))
    return out


def report(w, hold, tag=""):
    ks, as_, ls, mse, r = [], [], [], [], []
    pv, tv, md = [], [], []
    for b in hold:
        k, a, l, m, c = NT.accuracy(w, b)
        ks.append(k), as_.append(a), ls.append(l), mse.append(m), r.append(c)
        *_, v = NT.forward(w, b)
        pv.append(np.asarray(v))
        tv.append(np.asarray(b["val"]))
        md.append(np.asarray(b["mode"]))
    pv, tv, md = np.concatenate(pv), np.concatenate(tv), np.concatenate(md)
    m0, m1, m2 = md == 0, md == 1, md == 2
    out = dict(kind=np.mean(ks), angle=np.mean(as_), length=np.mean(ls),
               mse=np.mean(mse), r=np.mean(r),
               auc_div=auc(pv, m0, m1), ceil_div=auc(tv, m0, m1),
               auc_un=auc(pv, m0, m2), ceil_un=auc(tv, m0, m2),
               mean=(pv[m0].mean(), pv[m1].mean(), pv[m2].mean()))
    if tag:
        print(f"  {tag}  kind {out['kind']:.2f} angle {out['angle']:.2f} "
              f"length {out['length']:.2f} | value mse {out['mse']:.4f} "
              f"r {out['r']:+.2f} auc {out['auc_div']:.3f}", flush=True)
    return out


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "value"
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    use = USE[mode]

    rng = np.random.default_rng(seed)
    hrng = np.random.default_rng(999)
    t0 = time.time()
    hold = []
    for _ in range(HOLD):
        hb = DA.batch(hrng, n_ex=N_EX)
        hold.append(NT.jnpify(hb))
    pool = [NT.jnpify(DA.batch(rng, n_ex=N_EX)) for _ in range(POOL)]
    print(f"{mode}: {steps} steps, batch {N_EX}, pool {POOL} x reuse {REUSE}, "
          f"held out {HOLD * N_EX} examples  ({time.time() - t0:.0f}s of data)")

    w = NT.init(np.random.default_rng(seed + 1))
    st = NT.adam_init(w)
    print(f"{NT.nparam(w)} parameters on {jax.default_backend()}")
    report(w, hold, "step     0")

    t0, gen = time.time(), 0.0
    for i in range(1, steps + 1):
        b = pool[i % POOL]
        w, st, tot, parts = NT.step(w, st, b, use, lr=1e-3)
        if i % REUSE == 0:                       # one batch out, one in
            tg = time.time()
            pool[(i // REUSE) % POOL] = NT.jnpify(DA.batch(rng, n_ex=N_EX))
            gen += time.time() - tg
        if i % max(1, steps // 10) == 0:
            report(w, hold, f"step {i:5d}")
    wall = time.time() - t0
    print(f"\n{wall:.0f}s, {wall / steps * 1e3:.1f} ms/step "
          f"({100 * gen / wall:.0f}% of it generating data)")

    o = report(w, hold)
    print("\n  held out, {} examples".format(HOLD * N_EX))
    print(f"  value mse {o['mse']:.4f}, correlation {o['r']:+.3f}")
    print(f"  mean predicted value: matched {o['mean'][0]:.3f}, "
          f"divergent {o['mean'][1]:.3f}, unrelated {o['mean'][2]:.3f}")
    print()
    print("  ranking a matched prefix above a wrong one:")
    print(f"    vs divergent   {o['auc_div']:.3f}   ceiling {o['ceil_div']:.3f}"
          f"   ({100 * o['auc_div'] / o['ceil_div']:.0f}% of it)")
    print(f"    vs unrelated   {o['auc_un']:.3f}   ceiling {o['ceil_un']:.3f}"
          f"   ({100 * o['auc_un'] / o['ceil_un']:.0f}% of it)")
    print("  the ceiling is below 1.0 because two completions of one prefix")
    print("  can draw nearly the same picture -- that is the DATA's ambiguity,")
    print("  not the model's, and risk 2 is the gap to it, not to 1.0.")
    if mode != "value":
        bk, ba, bl = baseline(hold)
        print(f"\n  action accuracy      net     prior alone")
        for nm, v, c in (("kind  ", o['kind'], bk), ("angle ", o['angle'], ba),
                         ("length", o['length'], bl)):
            print(f"    {nm}            {v:.3f}      {c:.3f}"
                  f"      {v - c:+.3f}")
        print("  the control is the most frequent action among the LEGAL ones,")
        print("  with no image: it already knows the grammar and the prior.")
    np.save(f"out_shots/net_{mode}.npy",
            np.array({k: np.asarray(v) for k, v in w.items()}, dtype=object))
    print(f"\nwrote out_shots/net_{mode}.npy")


if __name__ == "__main__":
    main()
