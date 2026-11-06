"""Step 3: exhaustive search over program STRUCTURE, ranked by description
length -- and first, on targets we generated ourselves.

This is the gate one level up.  `gate.py` asked whether the DSL can express
an SVHN digit, by handing the fitter a structure and letting it fit numbers.
This asks whether the OBJECTIVE can pick the right structure out of all of
them, which is a different question and the one the proposer depends on: a
proposer is trained to predict a score, so if the score does not rank correct
programs first, the proposer is being trained toward the wrong thing.

Self-generated targets come first because they are the only place the answer
is known.  Render a program, search every legal structure against the render,
and the true structure can drive the residual to zero -- so it must come out
at or near the top, and if it does not, the objective is wrong before any
network exists.

One honest complication, and the reason this module reports two numbers
rather than one.  "The generating structure must rank first" is false as
stated, because distinct structures can render the SAME PICTURE.  Two 45
degree arcs chained with no turn between them continue through 90 degrees in
total, which is what one 90 degree arc draws; `check.py` asserts the same
thing for turn+turn and for straight+straight.  When that happens the shorter
program is the better answer and MDL should prefer it.  So we measure:

    recovery    the generator itself ranks first
    explained   the winner fits AT LEAST AS WELL as the generator, i.e.
                whatever the score returned does explain the image.  One
                sided on purpose: a winner that fits much better than the
                generator means the generator was under-fitted, which is a
                search failure and shows up in `recovery`, not here

A second complication, learned the hard way on the first run.  A structure
can be mis-ranked for three different reasons, and only one of them is the
objective's fault:

    recall      stage A never shortlisted the true structure
    search      it was shortlisted but fitted worse than a rival that is
                genuinely a worse explanation
    objective   both were fitted to their own optimum and the score still
                preferred the wrong one

The first run reported 8/12 recovery and it read like an objective failure.
It was not: refitting the generators at 64 x 1200 put the true structure
first on 11 of 12, and the 12th is a real equivalence -- for a SINGLE arc,
`+a` and `-a` differ by a 180 degree rotation, which `phi0` can produce.  So
this module now forces the generator into stage B whatever stage A thought,
reports recall separately, and compares residuals RELATIVELY.  Those three
changes are what make the headline number mean the objective.

Usage: .venv/bin/python -m pen.enumerate [n_targets] [nmax]
"""

import sys
import time
from dataclasses import dataclass

import numpy as np

import imageio as IO

from . import dsl as PN
from . import fit as PF

CFG = PN.PEN
OUT = "out_shots"
N_TGT = int(sys.argv[1]) if len(sys.argv) > 1 else 12
NMAX = int(sys.argv[2]) if len(sys.argv) > 2 else 2
# Stage A shortlists cheaply over everything, stage B refits the survivors
# properly.  Measured on CPU: 50 ms per fit at 150 iterations, 130 at 400, so
# spending the long runs only on the top few is a 7x saving overall.
# Budgets are MEASURED, not guessed, and the first guess was far too small.
# At 4 restarts x 150 iterations stage A dropped the true structure out of the
# shortlist on 1 of 12 self-generated targets, and at 12 x 400 stage B left it
# under-fitted against a rival on 3 more -- `+90 -45` only beat `+45 -` at
# 64 x 1200 (1.2e-7 against 3.1e-7).  Reading those as objective failures
# would have been wrong; they were search failures, and the cost of finding
# that out was one hard refit.
A_RESTART, A_ITER = 4, 150
B_KEEP, B_RESTART, B_ITER = 4, 8, 400
B_ROUNDS = 2
# Stage A is back to 4 restarts, which 3a showed was too weak when stage B
# merely refitted the shortlist.  It is adequate now BECAUSE stage B climbs:
# a climb recovers from a shortlist that lands near the answer, where a refit
# can only polish whatever it was handed.  Measured budget at <= 3 moves:
# stage A 12 min a target, stage B 12.


def make_targets(rng, hyps, n):
    """n noiseless targets, each rendered from a structure in `hyps`.

    Noiseless on purpose: the residual of the true structure can reach zero,
    which is what makes the ranking question well posed.  A real crop has a
    texture floor and no program reaches it, so there "best" is relative and
    only comparable within one image.
    """
    gen = [hyps[i] for i in rng.choice(len(hyps), n, replace=False)]
    p = np.stack([PN.sample_params(rng, pr, CFG) for pr in gen]).astype(np.float32)
    enc = PN.encode(gen, CFG)
    p = np.asarray(PN.scale_to_page(p, enc, CFG), np.float32)
    tgt = np.asarray(PN.alpha(p, enc, CFG, down=2))
    return gen, p, tgt


def neighbours(prog, cfg=CFG, nmax=None):
    """Every program one edit away: retune a move's angle, insert a move,
    delete a move.  `legal` does the pruning.

    `nmax` caps the length the climb may reach.  It matters for the
    self-generated experiment: insertion otherwise walks straight out of the
    space the generator was drawn from, and since a longer program fits
    better and the complexity penalty is weak (+3.2 nats a move), the climb
    would "win" by leaving the comparison rather than by solving it.

    This is the move set hill-climbing and SMC refinement both walk, and it
    is small -- 84 on average for a <= 3-move program, against 3705 to
    enumerate, so a climb step is 44x cheaper than a sweep.  It costs no new
    machinery either: discrete choices ride as data, so "this program and all
    its neighbours" is the same compiled kernel call as "all 3705
    structures".
    """
    lim = cfg.NMOVE if nmax is None else min(nmax, cfg.NMOVE)
    out, n = set(), len(prog)
    for i in range(n):
        k = PN.kind(prog[i])
        for a in (PN.DRAWS if k == "d" else PN.SPINS if k == "t" else ()):
            q = prog[:i] + ((prog[i][0], a),) + prog[i + 1:]
            if PN.legal(q, cfg):
                out.add(q)
        q = prog[:i] + prog[i + 1:]
        if q and PN.legal(q, cfg):
            out.add(q)
    moves = ([PN.draw(a) for a in PN.DRAWS]
             + [PN.spin(a) for a in PN.SPINS] + [PN.HOP])
    for i in range(n + 1):
        for m in moves:
            q = prog[:i] + (m,) + prog[i:]
            if len(q) <= lim and PN.legal(q, cfg):
                out.add(q)
    out.discard(prog)
    return sorted(out)


def climb(seeds, tgt, seed=0, rounds=3, keep=8, n_restart=16, n_iter=400,
          nmax=None):
    """Hill-climb from `seeds` over the edit neighbourhood, by description
    length.  Returns every program evaluated, with its score.

    This is stage B, and it replaces refitting the shortlist harder.  The
    difference matters because stage A over 3705 structures keeps the top 8 --
    0.2%, twenty times harsher than the 4.4% that gave 3a its 8/8 recall.  A
    harder refit of a near-miss shortlist cannot reach the answer; a climb
    only needs the shortlist to land NEAR it.
    """
    seen = {}
    front = list(dict.fromkeys(seeds))
    for _ in range(rounds):
        cand = [q for pr in front for q in neighbours(pr, CFG, nmax)] + front
        cand = [q for q in dict.fromkeys(cand) if q not in seen]
        if not cand:
            break
        t = np.repeat(tgt[None], len(cand), 0)
        p, _, f = PF.run(cand, t, CFG, n_restart, n_iter, 0.05, seed=seed,
                         chunk=512)
        tot = PF.mdl(cand, p, f, CFG).sum(1)
        for j, q in enumerate(cand):
            seen[q] = (tot[j], f[j], p[j])
        front = [q for q in sorted(seen, key=lambda z: seen[z][0])[:keep]]
    return seen


@dataclass
class Ranked:
    """One target's ranking.  A record, not a tuple: this started as a tuple
    and grew a field twice, and both times a positional index somewhere else
    silently read the wrong column -- once crashing the run after all the
    useful output had already printed."""
    order: list              # programs, best first
    total: np.ndarray        # description length, prior parameter term
    bic: np.ndarray          # description length, BIC parameter term
    parts: np.ndarray        # (k, 3) residual / structure / parameters
    params: np.ndarray       # fitted parameter vectors
    res: np.ndarray          # flat mse
    rank: int                # where the generator landed
    recalled: bool           # did stage A shortlist the generator


def bic_par(progs, cfg=CFG, n=32 * 32):
    """(k/2) log n over LIVE parameters -- a code length for the numbers.

    `dsl.cost`'s parameter term is the prior's negative log DENSITY, which is
    a differential entropy and not a code length: at SD_LEN = 0.35 a
    parameter sitting at its prior mean contributes log(0.35*sqrt(2pi)) =
    -0.13 nats, so the term REWARDS extra parameters.  Measured, the prior
    prices an extra move at +3.2 nats net, which an extra move repays by
    improving mse 0.62%; under BIC it is +12.6 nats and 2.4%.  On a noiseless
    target this does not matter -- the true structure reaches zero -- but
    against a texture floor almost any extra freedom buys 0.62%, so both are
    reported and the gap is the thing to watch.
    """
    k = np.array([int(np.asarray(PN.encode([p], cfg)["pmask"]).sum()) + PN.NG
                  for p in progs], np.float64)
    return 0.5 * k * np.log(n)


def rank_one(hyps, tgt, seed=0, must=None):
    """Fit every hypothesis to one target and rank by description length.

    `must` is a hypothesis index that goes into stage B whether or not stage A
    shortlisted it.  That separates two failures the first version of this
    conflated: stage A losing the right answer (recall) and the score ranking
    a properly-fitted right answer below a wrong one (the objective).  Only
    the second is the objective's fault, and only the second is fatal.
    """
    t = np.repeat(tgt[None], len(hyps), 0)
    _, _, fa = PF.run(hyps, t, CFG, A_RESTART, A_ITER, 0.05, seed=seed, chunk=512)
    short = list(np.argsort(fa)[:B_KEEP])
    recall = must is None or must in short
    seeds = [hyps[i] for i in short]
    if must is not None and not recall:
        seeds.append(hyps[must])          # measure ranking, not stage A's luck
    seen = climb(seeds, tgt, seed=seed, rounds=B_ROUNDS, keep=B_KEEP,
                 n_restart=B_RESTART, n_iter=B_ITER, nmax=NMAX)
    progs = list(seen)
    pb = np.stack([seen[q][2] for q in progs])
    fb = np.array([seen[q][1] for q in progs])
    parts = PF.mdl(progs, pb, fb, CFG)        # [residual, structure, params]
    tot = parts.sum(1)
    tot_bic = parts[:, 0] + parts[:, 1] + bic_par(progs)
    o = np.argsort(tot)
    return ([progs[i] for i in o], tot[o], tot_bic[o], parts[o], pb[o], fb[o],
            recall)


def main():
    rng = np.random.default_rng(0)
    hyps = PN.structures(CFG, nmax=NMAX)
    gen, gp, tgt = make_targets(rng, hyps, N_TGT)
    nfit = len(hyps) * A_RESTART + B_ROUNDS * B_KEEP * 84 * B_RESTART // 8
    print(f"{len(hyps)} legal structures of <= {NMAX} moves, {N_TGT} "
          f"self-generated targets")
    print(f"{nfit} fits per target, {nfit * N_TGT} total "
          f"(stage A {A_ITER} iter x {A_RESTART}, stage B = "
          f"{B_ROUNDS} climb rounds at {B_ITER} x {B_RESTART})")

    t0 = time.time()
    rows, wins = [], []
    gidx = [hyps.index(g) for g in gen]
    for i in range(N_TGT):
        order, total, tbic, parts, pb, fb, rec = rank_one(
            hyps, tgt[i], seed=i, must=gidx[i])
        rank = order.index(gen[i])
        rows.append(Ranked(order, total, tbic, parts, pb, fb, rank, rec))
        wins.append(order[0])
    wall = time.time() - t0
    print(f"wall {wall:.1f}s, {wall / (nfit * N_TGT) * 1000:.0f} ms/fit\n")

    print("  generator            rank  A?  winner               "
          "  res_gen    res_win   dL     verdict")
    rec = exp = recall = bicrec = 0
    for i in range(N_TGT):
        r = rows[i]
        gi = r.rank
        rg, rw = r.res[gi], r.res[0]
        dL = r.total[gi] - r.total[0]
        is_rec = gi == 0
        # RELATIVE, not absolute.  These residuals run to 1e-12; the absolute
        # 1e-4 the first version used called a 3x worse fit "equivalent" and
        # hid three search failures.
        is_exp = is_rec or rw <= rg * 1.05
        rec += is_rec
        exp += is_exp
        recall += r.recalled
        bicrec += int(np.argmin(r.bic) == gi)
        v = "recovered" if is_rec else ("equivalent" if is_exp else "WRONG")
        print(f"  {PN.show(gen[i]):20s} {gi:4d}  {'y' if r.recalled else 'n'}   "
              f"{PN.show(wins[i]):20s} {rg:.3e}  {rw:.3e}  {dL:+7.1f}  {v}")

    print()
    print(f"recall     stage A shortlisted the generator on {recall}/{N_TGT}"
          f"  (a SEARCH number, not the objective's)")
    print(f"recovery   generator ranks first on {rec}/{N_TGT}")
    print(f"explained  winner fits at least as well as the generator "
          f"(5% slack) on {exp}/{N_TGT}")
    print(f"under BIC  generator ranks first on {bicrec}/{N_TGT}")

    # Which MDL term decides?  If the residual swamps the other two, the rank
    # is really just "best fit" and the structure prior is decorative.
    allp = np.concatenate([r.parts for r in rows])
    print(f"MDL spread over the top {B_KEEP}: residual "
          f"{allp[:, 0].std():.1f}, structure {allp[:, 1].std():.1f}, "
          f"parameters {allp[:, 2].std():.1f} nats (sd)")

    # one batched render of every winner at its own fitted parameters
    wp = np.stack([r.params[0] for r in rows])
    aw = np.asarray(PN.alpha(wp, PN.encode(wins, CFG), CFG, down=2))
    # --------------------------------------------------- the basin map
    # Hill-climbing assumes the structure landscape is locally searchable:
    # that single edits lead from a good-but-wrong program to the best one
    # without first passing through worse ones.  Discrete program spaces are
    # full of plateaus where that is false.  This measures it on the one data
    # we have a full ranking for -- start a climb from the k-th best
    # structure and see whether it arrives at the best.  The largest k that
    # still arrives is how many structurally distinct particles SMC needs to
    # carry; if only k = 1 works, hill-climbing is not enough and the
    # proposer has to be right first time.
    print()
    print("basin map: climb from the k-th best structure, does it reach the best?")
    KS, NB = (1, 4, 16), min(N_TGT, 3)
    print("  target   " + "   ".join(f"k={k:<3d}" for k in KS))
    hit = {k: 0 for k in KS}
    for i in range(NB):
        r = rows[i]
        best = r.order[0]
        line = []
        for k in KS:
            if k > len(r.order):
                line.append("  -   ")
                continue
            seen = climb([r.order[k - 1]], tgt[i], seed=100 + i, rounds=2,
                         keep=2, n_restart=8, n_iter=B_ITER, nmax=NMAX)
            got = min(seen, key=lambda z: seen[z][0])
            ok = got == best
            hit[k] += ok
            line.append(" yes  " if ok else " no   ")
        print(f"  {i:5d}   " + "  ".join(line))
    print("  reached the best from " + ", ".join(
        f"k={k} on {hit[k]}/{NB}" for k in KS))
    print("  k=1 is the CONTROL: a climb started from the best should return")
    print("  it, so a 'no' there is refit noise, not landscape structure, and")
    print("  the other columns mean little until it is clean.")
    print("  -> the largest clean k is how many structurally distinct")
    print("     particles SMC has to carry.")

    tiles = [np.concatenate([IO.ink(tgt[i]), IO.ink(aw[i]),
                             IO.overlay(aw[i], tgt[i])], axis=1)
             for i in range(N_TGT)]
    IO.save(f"{OUT}/enum_self.png", IO.grid(tiles, 4), 3)
    print(f"\nwrote {OUT}/enum_self.png  (target | winner | overlay)")


if __name__ == "__main__":
    main()
