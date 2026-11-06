"""Invariants of pen.py.  Each one is a design claim the header makes.

Run with float64: the chain, the deviation basis and the canonicalisation
rules are supposed to hold exactly, and in float32 you cannot tell exact from
nearly.
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from . import dsl as PN

CFG = PN.PEN
RNG = np.random.default_rng(7)
FAIL = []


def chk(name, got, tol, unit=""):
    ok = got <= tol
    if not ok:
        FAIL.append(name)
    print(f"{'ok  ' if ok else 'FAIL'} {name:46s} {got:10.3e} {unit} (tol {tol:.0e})")


def progs_random(n):
    """`sample_struct` samples from the same restricted sets `legal` checks,
    so this asserts the contract instead of filtering on it."""
    out = [PN.sample_struct(RNG, CFG) for _ in range(n)]
    assert all(PN.legal(pr, CFG) for pr in out)
    return out


def pack(progs):
    enc = PN.encode(progs, CFG)
    p = np.stack([PN.sample_params(RNG, pr, CFG) for pr in progs]).astype(np.float64)
    return enc, jnp.asarray(p)


# 1 -- the chain: an independent turtle walked move by move in numpy must
#      reproduce every move-end pose.  This is the test that the cumulative
#      sums in `path` are assembled in the right order.
progs = progs_random(64)
enc, p = pack(progs)
end, phi = PN.pose(p, enc, CFG)
th = np.asarray(enc["theta"])
rt = np.asarray(enc["rot"])
dn = np.asarray(enc["down"])
hp = np.asarray(enc["hop"])
lv = np.asarray(enc["live"])
q = (np.asarray(p)[:, :CFG.NMOVE * PN.NP].reshape(-1, CFG.NMOVE, PN.NP)
     * np.asarray(enc["pmask"]))
g = np.asarray(p)[:, CFG.NMOVE * PN.NP:]
ref_e = np.zeros((len(progs), CFG.NMOVE, 2))
ref_p = np.zeros((len(progs), CFG.NMOVE))
for j in range(len(progs)):
    x = y = 0.0
    a = g[j, 4]
    for i in range(CFG.NMOVE):
        a += rt[j, i]
        L = (dn[j, i] * CFG.LREF * np.exp(q[j, i, 0])
             + hp[j, i] * CFG.HOPMAX / (1.0 + np.exp(-q[j, i, 0]))) * lv[j, i]
        t = th[j, i]
        s = 1.0 if t == 0 else np.sin(t / 2) / (t / 2)
        x += L * s * np.cos(a + t / 2)
        y += L * s * np.sin(a + t / 2)
        a += t
        ref_e[j, i], ref_p[j, i] = (x, y), a
chk("chain: move-end position vs numpy turtle",
    float(np.abs(np.asarray(end) - ref_e).max()), 1e-9, "px")
chk("chain: move-end heading vs numpy turtle",
    float(np.abs(np.asarray(phi) - ref_p).max()), 1e-12, "rad")

# 2 -- bending a move cannot move the pen.  Not approximately: the deviation
#      basis is zero at u = 1 bit-exactly, so this must be 0.0.
p0 = np.asarray(p).copy()
for i in range(CFG.NMOVE):
    p0[:, i * PN.NP + 1:i * PN.NP + 3] = 0.0
e0, f0 = PN.pose(jnp.asarray(p0), enc, CFG)
chk("bend: move-end position unchanged",
    float(np.abs(np.asarray(end) - np.asarray(e0)).max()), 0.0, "px")
chk("bend: move-end heading unchanged",
    float(np.abs(np.asarray(phi) - np.asarray(f0)).max()), 0.0, "rad")

# 3 -- a turn is pure structure: it travels nowhere and has no live
#      parameters at all.
tm = (np.asarray(enc["rot"]) != 0)
P = np.asarray(PN.path(p, enc, CFG))
mv = np.abs(P[:, :, -1, :] - P[:, :, 0, :]).max(-1)
chk("turn: travels zero distance", float(mv[tm].max() if tm.any() else 0), 0.0, "px")
chk("turn: has no live parameters",
    float(np.abs(np.asarray(enc["pmask"])[tm]).max() if tm.any() else 0), 0.0)

# 4 -- a parameter the structure does not use takes EXACTLY zero gradient.
#      The whole one-kernel-for-every-structure design rests on this.
def loss(pp):
    return jnp.sum(PN.alpha(pp, enc, CFG) ** 2)


gr = np.asarray(jax.grad(loss)(p))
pm = np.asarray(enc["pmask"]).reshape(len(progs), -1)
dead = np.abs(gr[:, :CFG.NMOVE * PN.NP][pm == 0])
chk("mask: dead parameters, max |grad|", float(dead.max() if dead.size else 0), 0.0)
live = np.abs(gr[:, :CFG.NMOVE * PN.NP][pm == 1])
print(f"     (live parameters: {int((pm == 1).sum())}, "
      f"max |grad| {live.max():.2e}, zero-grad live {int((live == 0).sum())})")
chk("grad: all finite", float(0.0 if np.isfinite(gr).all() else 1.0), 0.0)

# 5 -- a line is straight and an arc is an arc.  The angle is data, so this is
#      a statement about the sinc form, including at 0 where a radius
#      parameterisation would divide by zero.
for name in ("0", "+90", "-180", "+360"):
    pr = (PN.draw(name),)
    e1 = PN.encode([pr], CFG)
    p1 = np.array([PN.sample_params(RNG, pr, CFG)], np.float64)
    p1[0, 1:3] = 0.0                      # no deviation: test the template
    P = np.asarray(PN.path(jnp.asarray(p1), e1, CFG))[0, 0]
    d = np.diff(P, axis=0)
    hdg = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    # chord j (samples j, j+1) has heading phi_in + theta*(j+0.5)/SPS, and the
    # samples run u = 1/SPS .. 1, so the first and last chords are 1.5/SPS and
    # (SPS-0.5)/SPS of the way round: SPS-2 steps apart.
    want = PN.ANGLE[name] * np.pi * (CFG.SPS - 2) / CFG.SPS
    chk(f"{name:4s}: total turn along the samples",
        float(abs((hdg[-1] - hdg[0]) - want)), 1e-9, "rad")
    if name == "0":
        v = P - P[0]
        n = np.array([-v[-1, 1], v[-1, 0]]) / np.linalg.norm(v[-1])
        chk("0   : off-line deviation", float(np.abs(v @ n).max()), 1e-12, "px")
    else:
        ds = np.linalg.norm(d, axis=1)
        k = np.diff(hdg) / ds[1:]
        chk(f"{name:4s}: curvature spread (rel)",
            float(np.ptp(k) / abs(k.mean())), 1e-9)

# 6 -- the pen lays no ink while it is up, and none at all on a dead slot.
pl = [x for x in progs_random(256) if PN.nstroke(x) > 1][:32]
encl, ppl = pack(pl)
w = np.asarray(PN.weights(PN.path(ppl, encl, CFG), encl))
up = (np.asarray(encl["hop"]) == 1)
chk("pen-up: arclength weight", float(np.abs(w[up]).max()), 0.0, "px")
chk("dead slots: arclength weight",
    float(np.abs(w[np.asarray(encl["live"]) == 0]).max()), 0.0, "px")

# 7 -- the two canonicalisation rules in `legal` are claims about geometry,
#      so test the geometry rather than asserting them.
#      (a) two turns in a row draw what one turn through the sum draws
a2 = (PN.draw("+90"), PN.spin("+45"), PN.spin("+45"), PN.draw("-90"))
a1 = (PN.draw("+90"), PN.spin("+90"), PN.draw("-90"))
v2 = np.zeros((1, PN.nparam(CFG)), np.float64)
v1 = np.zeros((1, PN.nparam(CFG)), np.float64)
v2[0, 3 * PN.NP] = v1[0, 2 * PN.NP] = 0.3          # the second draw's length
A = PN.alpha(jnp.asarray(v2), PN.encode([a2], CFG), CFG)
B = PN.alpha(jnp.asarray(v1), PN.encode([a1], CFG), CFG)
chk("canon: turn+turn == one folded turn",
    float(jnp.abs(A - B).max()), 1e-12, "alpha")
assert not PN.legal(a2, CFG) and PN.legal(a1, CFG)

#      (b) two straight draws in a row are one longer straight.  The claim is
#      about the DRAWING, not the samples: splitting a line in two samples it
#      at twice the density, and the first sample of a move sits at u = 1/SPS,
#      so the rasters differ slightly at the end cap.  Test the geometry.
b2 = (PN.draw("0"), PN.draw("+90"), PN.draw("0"), PN.draw("0"))
b1 = (PN.draw("0"), PN.draw("+90"), PN.draw("0"))
v2 = np.zeros((1, PN.nparam(CFG)), np.float64)
v1 = np.zeros((1, PN.nparam(CFG)), np.float64)
v2[0, 2 * PN.NP] = v2[0, 3 * PN.NP] = np.log(0.5)   # two halves
v1[0, 2 * PN.NP] = 0.0                              # one whole
e2, e1 = PN.encode([b2], CFG), PN.encode([b1], CFG)
end2, phi2 = PN.pose(jnp.asarray(v2), e2, CFG)
end1, phi1 = PN.pose(jnp.asarray(v1), e1, CFG)
chk("canon: straight+straight ends where one straight does",
    float(jnp.abs(end2[0, 3] - end1[0, 2]).max()), 1e-12, "px")
P2 = np.asarray(PN.path(jnp.asarray(v2), e2, CFG))[0, 2:4].reshape(-1, 2)
P1 = np.asarray(PN.path(jnp.asarray(v1), e1, CFG))[0, 2]
d = P1[-1] - P1[0]
n = np.array([-d[1], d[0]]) / np.linalg.norm(d)
chk("canon: both halves lie on that straight",
    float(np.abs((P2 - P1[0]) @ n).max()), 1e-12, "px")
A = PN.alpha(jnp.asarray(v2), e2, CFG)
B = PN.alpha(jnp.asarray(v1), e1, CFG)
print(f"     (raster differs by {float(jnp.abs(A - B).max()):.3f} alpha at the "
      f"end cap, from the 2x sampling density)")
assert not PN.legal(b2, CFG) and PN.legal(b1, CFG)

# 8 -- place() does what it says: tx moves the drawing by TRANS px, log s
#      scales it about its own centre of ink.
def centroid(pp, ee):
    P = PN.path(pp, ee, CFG)
    w = PN.weights(P, ee)
    P, w = PN.place(P, w, pp, CFG)
    W = w / jnp.sum(w, (1, 2), keepdims=True)
    return np.asarray(jnp.stack([jnp.sum(W * P[..., 0], (1, 2)),
                                 jnp.sum(W * P[..., 1], (1, 2))], -1))


gi = CFG.NMOVE * PN.NP
c0 = centroid(p, enc)
c1 = centroid(jnp.asarray(p).at[:, gi].add(0.25), enc)
chk("place: tx moves ink centroid by TRANS*dtx",
    float(np.abs((c1 - c0) - np.array([CFG.TRANS * 0.25, 0.0])).max()), 1e-6, "px")
c2 = centroid(jnp.asarray(p).at[:, gi + 2].add(0.3), enc)
chk("place: log s leaves the ink centroid put",
    float(np.abs(c2 - c0).max()), 1e-6, "px")

# 9 -- phi0 rotates the drawing rigidly.  This is why a rotated glyph is the
#      SAME program, and why the proposer's rotation invariance can be left to
#      augmentation.  Slant is a page-frame shear, so it is set to 0 here.
pr = np.asarray(p).copy()
pr[:, gi + 3] = 0.0
pr[:, gi + 6] = 0.0          # aspect is a page shear too
worst, ref = 0.0, None
for ang in np.linspace(-np.pi, np.pi, 7):
    pr[:, gi + 4] = ang
    P = PN.path(jnp.asarray(pr), enc, CFG)
    w = PN.weights(P, enc)
    P, w = PN.place(P, w, jnp.asarray(pr), CFG)
    P, w = np.asarray(P).reshape(len(progs), -1, 2), np.asarray(w).reshape(len(progs), -1)
    c = (w[..., None] * P).sum(1) / w.sum(1)[:, None]
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    V = (P - c[:, None]) @ R
    if ref is None:
        ref = V
    else:
        worst = max(worst, float(np.abs(V - ref)[w > 0].max()))
chk("phi0: rotates the drawing rigidly", worst, 1e-9, "px")

# 10 -- a prefix encoding renders exactly the prefix program.  This is what
#       makes one sampled program into T training examples for a proposer.
jobs = [(pr, t) for pr in progs[:24] for t in range(1, len(pr) + 1)]
par = {pr: PN.sample_params(RNG, pr, CFG) for pr in progs[:24]}
bad = 0.0
B = 64                           # the shape tests 1-4 already compiled
for s0 in range(0, len(jobs), B):
    blk = jobs[s0:s0 + B]
    pad = [blk[0]] * (B - len(blk))
    full, cut = zip(*(blk + pad))
    pp = jnp.asarray(np.array([par[pr] for pr in full], np.float64))
    a = PN.alpha(pp, PN.prefix(PN.encode(full, CFG), list(cut)), CFG)
    b = PN.alpha(pp, PN.encode([pr[:t] for pr, t in zip(full, cut)], CFG), CFG)
    bad = max(bad, float(jnp.abs(a - b)[:len(blk)].max()))
chk("prefix: render matches the prefix program", bad, 0.0, "alpha")

# 11 -- mirror() really is a reflection.  It carries the trickiest rule in
#       the DSL (+180 and -180 are the same in-place turn, so +180 is its own
#       mirror) and it feeds proposer augmentation, so it is asserted here
#       rather than trusted.
# Negating every angle negates every heading, so d is unchanged (sinc is
# even) and the drawing comes out with y flipped -- a reflection about the
# HORIZONTAL centre line.  phi0 is a heading too, so it negates with them;
# bend/wig and slant are signed in the same way, and are zeroed here so the
# claim under test is the one about the discrete angles.
gi = CFG.NMOVE * PN.NP
worst = 0.0
for pr in progs[:32]:
    q = PN.mirror(pr)
    assert PN.legal(q, CFG) and PN.mirror(q) == pr, PN.show(pr)
    p = np.array([PN.sample_params(RNG, pr, CFG)] * 2, np.float64)
    p[:, [j * PN.NP + k for j in range(CFG.NMOVE) for k in (1, 2)]] = 0.0
    p[:, gi + PN.GI["slant"]] = 0.0
    p[:, gi + PN.GI["tx"]] = p[:, gi + PN.GI["ty"]] = 0.0
    p[1, gi + PN.GI["phi0"]] *= -1.0
    e = PN.encode([pr, q], CFG)
    P = PN.path(jnp.asarray(p), e, CFG)
    V = np.asarray(PN.place(P, PN.weights(P, e), jnp.asarray(p), CFG)[0])
    wv = np.asarray(PN.weights(P, e))
    ref = V[1].copy()
    ref[..., 1] = CFG.H - ref[..., 1]
    worst = max(worst, float(np.abs(V[0] - ref)[wv[0] > 0].max()))
chk("mirror: renders as the y-reflection", worst, 1e-9, "px")

# 12 -- show and read are inverses, so there is one program notation and
#       `HAND` can be written the way printouts read.
bad = sum(PN.read(PN.show(pr)) != pr for pr in progs)
chk("show/read: round-trips", float(bad), 0.0, "progs")

# 13 -- the prior is a distribution: cost() must price exactly what
#       sample_struct() samples.  They are separate code paths over the same
#       rule set, and when they drifted `- +90` came out 1.24x off.
z = np.zeros(PN.nparam(CFG), np.float32)
st3 = PN.structures(CFG, nmax=3)
mass = sum(float(np.exp(-PN.cost(pr, z, CFG)[0])) for pr in st3)
emp = np.mean([len(PN.sample_struct(RNG, CFG)) <= 3 for _ in range(40000)])
se = np.sqrt(emp * (1 - emp) / 40000)
chk("prior: cost integrates to the sampler's mass", abs(mass - emp) / se, 4.0,
    "sigma")

print()
print("FAILED: " + ", ".join(FAIL) if FAIL else "all invariants hold")
