"""Recover the 3b-real table for the 10 crops that finished, from the winners
the log printed.  The ranking over candidates is gone -- only the winner's own
numbers can be rebuilt, by refitting it.  The logged mse is the check."""
import numpy as np
import imageio as IO
import svhn_data as D
from pen import dsl as PN, fit as PF
from pen.enumerate import bic_par, B_ITER, B_RESTART

LOG = [  # crop, label, winner, mse as the killed run printed it
    (  595, 2, "+90 +180 -270", 0.0402),
    ( 3006, 1, "-135 ^ -90",    0.0380),
    ( 3158, 6, "-270 +360 -45", 0.0448),
    ( 3598, 3, "-360 ^ -180",   0.1133),
    (10456, 2, "+45 +360 +135", 0.0266),
    (12423, 1, "- +180 +45",    0.0396),
    (12673, 4, "+90 +270 +90",  0.0184),
    (15303, 8, "-360 -180 +270",0.0370),
    (15508, 0, "-90 ^ -360",    0.0568),
    (18275, 5, "-270 +180 -",   0.0589),
]
n = len(LOG)
d = D.load()
tgt = np.asarray(d["ink"], np.float32)[:n]
idx = np.asarray(d["idx"])[:n]
assert list(idx) == [r[0] for r in LOG], (list(idx), [r[0] for r in LOG])

wins = [PN.read(r[2].split()) for r in LOG]
for w, r in zip(wins, LOG):
    assert PN.show(w) == r[2], (PN.show(w), r[2])

p, m, f = PF.run(wins, tgt, PN.PEN, B_RESTART, B_ITER, 0.05, seed=0)
parts = PF.mdl(wins, p, f, PN.PEN)
prior = parts.sum(1)
bic = parts[:, 0] + parts[:, 1] + bic_par(wins)

a = np.asarray(PN.alpha(p, PN.encode(wins, PN.PEN), PN.PEN, down=2))
IU, IC = PF.iou(a, tgt), PF.iou(a, tgt, core=PF.CORE)
bl, R2 = PF.blank_mse(tgt), PF.r2(a, tgt)
logged = np.array([r[3] for r in LOG])

print("  crop  lab    mse  logged  blank    R2   IoU  IoUc   L_prior    L_bic   structure")
for i, (c, lb, s, _) in enumerate(LOG):
    print(f" {c:6d}   {lb}  {f[i]:.4f}  {logged[i]:.4f} {bl[i]:.4f} {R2[i]:+.2f} "
          f" {IU[i]:.2f}  {IC[i]:.2f} {prior[i]:9.1f} {bic[i]:8.1f}   {s}")

print()
print(f"refit vs logged mse: max relative drift {np.abs(f/logged-1).max()*100:.1f}%"
      f", in nats {np.abs(512*np.log(f/logged)).max():.1f}")
print(f"IoU    median {np.median(IU):.2f} full frame, {np.median(IC):.2f} centred"
      f"   (gate, hand-written: 0.60 / 0.71)")
print(f"       >= 0.5 on {int((IU>=.5).sum())}/{n} and {int((IC>=.5).sum())}/{n}"
      f"   (gate: 10/12 and 9/12)")
print(f"R2     median {np.median(R2):+.2f}   (gate: +0.41)")
print(f"blank beats the fit on {int((f>=bl).sum())}/{n}")
nm = np.array([len(w) for w in wins])
print(f"moves in the winner: mean {nm.mean():.1f}, max {nm.max()} (cap 3)")
print(f"L_bic - L_prior: mean {np.mean(bic-prior):+.1f} nats "
      f"(the parameter term's whole effect)")
print(f"MDL parts (sd over crops): residual {parts[:,0].std():.1f}, "
      f"structure {parts[:,1].std():.1f}, params {parts[:,2].std():.1f}")

tiles = [np.concatenate([IO.ink(tgt[i]), IO.ink(a[i]),
                         IO.overlay(a[i], tgt[i])], axis=1) for i in range(n)]
IO.save("out_shots/enum_real.png", IO.grid(tiles, 4), 3)
print("\nwrote out_shots/enum_real.png  (ink | best program | overlay)")

print("\nper-crop refit drift vs the killed run (nats of L_res):")
for i, (c, *_rest) in enumerate(LOG):
    print(f"   crop {c:6d}  {512*np.log(f[i]/logged[i]):+7.1f}")
