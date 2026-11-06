"""A pen, not a set of stamps.  Directional strokes with shared pen state.

An earlier attempt fixed the right problem the wrong way round.  Shape identity became
one number (the total turn), which was the point, but each piece still carried
its own absolute affine -- tx, ty, phi, scale, aspect, shear -- so a glyph was
a *set of independent stamps* that happened to overlap.  Nothing said the
second piece began where the first one ended, so the fit was free to scatter
four arcs over the digit and the structure said nothing about how it was
drawn.  Two fits of the same digit still shared no subtree.

This is the Bayesian-Program-Learning factorisation instead (Lake et al.
2015).  There is one pen.  It has a position and a heading.  There are exactly
three things it can do, and every one of them is relative to that state:

    turn(a)    rotate in place.  No travel, no ink, no parameters at all.
    draw(a)    travel an arc of total turn `a`, pen down, leaving the pen at
               the far end with its heading rotated by `a`.
    hop        pen up, straight, a bounded short distance.

So "the next stroke starts where the last one stopped" is not a constraint to
enforce, it is the only thing the parameterisation can express.  A stroke is a
maximal run of draws; a hop is the stroke boundary.  There are no stroke
positions anywhere in the parameter vector -- a glyph has ONE position, the
pen's starting point.

                            the alphabet

ONE angle set, used two ways.  `draw` takes any of the 13; `turn` takes the 7
that are not redundant in place (0 is a no-op, and a turn past 180 is the same
rotation the other way).  Plus the hop: 21 symbols.

Rotation is its own move rather than a corner glued to the front of a draw.
Gluing them meant 11 x 8 = 88 draw symbols, most of which a skewed prior never
produces, so a proposer would have no examples of them; and it meant a learned
abstraction could only ever capture one specific (corner, arc) pair instead of
reusing the corner.  The cost is about 15% longer programs.

Both the kind and the angle ride as *data* (`encode`), never as parameters, so
the whole alphabet is one compiled kernel and an arc cannot drift into a line
under gradient descent.  a = 0 is not special-cased in the code: the sinc form
below is exact and smooth through it.

                        what is left continuous

Per move, three numbers, and a turn uses none of them:

    log len     how far the pen travels.  ARC length, which is the pen's own
                measure -- a circle of path length L has radius L/2pi, and
                that is correct here precisely because the pen really does
                travel L.  (that attempt had to normalise by extent instead,
                because comparing independent stamps' scales is a different
                question from asking how far a pen moved.)  Live on draw and
                hop.
    bend, wig   deviation from the template, in a basis that is EXACTLY zero
                at the move's far end, so the knots can never move the pen.
                Bounded by tanh at BEND of the move's span and priced by
                `cost` on top of that: the template is cheap, leaving it is
                expensive.  This is `strokes.warp`'s bend/wiggle pair with a
                hard prior and an anchor.  Live on draw only -- a hop lays no
                ink, so bending one would be unobservable.

Per glyph, seven: tx, ty, log s, slant, aspect, phi0, log width.  Those are the cheap
ones -- where the drawing sits, how big it is, how it leans, which way the pen
set off, how broad the nib.  A glyph of three draws, a turn and a hop is
3*3 + 0 + 1 + 6 = 16 numbers, against 116 for four splines.

Deliberately not here yet: brush profile, serifs, per-stroke width, and the
scene around the glyph (ink colour, background, blur, neighbours), which is a
layer wrapping this one and not something a pen does.  A broad-nib width
model is the natural first addition, and it grafts onto `weights` without
touching the geometry.
"""

import itertools
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

# ------------------------------------------------------------------ alphabet
# values are multiples of pi.

# Named by SIGNED angle in degrees, not by "left" and "right".  The canvas y
# axis runs down, so a positive turn -- from +x toward +y -- is clockwise on
# screen, and any l/r naming is wrong for half the readers and half the time.
# A signed number is wrong for nobody.
ANGLE = {"0": 0.0,
         "+45": 0.25, "-45": -0.25,
         "+90": 0.5, "-90": -0.5,
         "+135": 0.75, "-135": -0.75,
         "+180": 1.0, "-180": -1.0,
         "+270": 1.5, "-270": -1.5,
         "+360": 2.0, "-360": -2.0}        # closed loop
DRAWS = tuple(ANGLE)                       # 13: every angle draws something
# In place, "0" does nothing and a turn past 180 is the same rotation taken
# the other way round, so the canonical in-place set is these seven.
SPINS = ("+45", "-45", "+90", "-90", "+135", "-135", "+180")

NEG = {a: n for a in ANGLE for n in ANGLE if ANGLE[n] == -ANGLE[a]}


def canon_spin(a):
    """The canonical in-place representative of an angle: +180 and -180 are
    the same rotation, and only +180 is in SPINS."""
    return a if a in SPINS else NEG[a]


def mirror(prog):
    """The x-reflection of a program: negate every angle.

    A DSL operation, not a gate one -- a proposer augmenting its training set
    and a library counting symmetric fragments both want it, and
    `check_pen.py` asserts it really is a reflection.
    """
    return tuple(m if m[0] == "h" else
                 (m[0], canon_spin(NEG[m[1]]) if m[0] == "t" else NEG[m[1]])
                 for m in prog)


NP = 3                                     # per move
PNAMES = ["log len", "bend", "wig"]
GNAMES = ["tx", "ty", "log s", "slant", "phi0", "log w", "log a"]
GI = {n: i for i, n in enumerate(GNAMES)}  # the glyph block, by name
NG = len(GNAMES)

HOP = ("h",)                               # the one pen-up move


@dataclass(frozen=True)
class Pen:
    H: int = 64
    SPS: int = 24                # samples per move
    NMOVE: int = 12              # turns take a slot, so this is ~8 draws
    SIGMA: float = 2.2
    GAIN: float = 6.5
    LREF: float = 30.0           # reference draw EXTENT, px at H = 64;
                                 # the parameter is arc length (see ext_frac)
    HOPMAX: float = 20.0         # hard ceiling on a pen-up hop: it is short
    BEND: float = 0.10           # max template deviation, fraction of span
    TRANS: float = 12.0
    SLANT: float = 0.4           # hard bound on the glyph shear
    ASPECT: float = 0.45         # hard bound on the glyph aspect, [0.64, 1.57]


PEN = Pen()


def nparam(cfg=PEN):
    return cfg.NMOVE * NP + NG


def nsymbol():
    return len(DRAWS) + len(SPINS) + 1


# ------------------------------------------------------------------ programs
# A program is a tuple of moves: ("d", angle), ("t", angle) or ("h",).

def draw(a):
    return ("d", a)


def spin(a):
    return ("t", a)


def kind(m):
    return m[0]


def legal(prog, cfg=PEN):
    """Canonical form.  Each rule deletes a redundancy, never a drawing.

    * at most NMOVE moves, and the first and last are draws -- a leading turn
      is absorbed by `phi0`, a leading hop by `tx, ty`, and a trailing turn or
      hop moves a pen that never comes down again
    * no two turns in a row: one turn through the sum draws the same thing
    * no two hops in a row: one longer hop does
    * no two straight draws in a row with nothing between them: that is one
      longer straight.  (Two *arcs* in a row are not redundant -- different
      lengths give different radii.)
    """
    if not prog or len(prog) > cfg.NMOVE:
        return False
    if kind(prog[0]) != "d" or kind(prog[-1]) != "d":
        return False
    for i, m in enumerate(prog):
        if kind(m) == "h":
            if kind(prog[i - 1]) == "h":
                return False
        elif kind(m) == "t":
            if m[1] not in SPINS or kind(prog[i - 1]) == "t":
                return False
        elif kind(m) == "d":
            if m[1] not in ANGLE:
                return False
            if i and prog[i - 1] == ("d", "0") and m[1] == "0":
                return False
        else:
            return False
    return True


def nstroke(prog):
    return 1 + sum(kind(m) == "h" for m in prog)


def ndraw(prog):
    return sum(kind(m) == "d" for m in prog)


def read(toks):
    """The inverse of `show`.  One syntax for a program, not two: `show`
    prints `-` for a straight and `^` for a hop, and this accepts exactly
    that, so hand-written programs can be written the way printouts read.
    """
    out = []
    for t in toks.split() if isinstance(toks, str) else toks:
        if t == "^":
            out.append(HOP)
        elif t.startswith("("):
            out.append(spin(t[1:-1]))
        else:
            out.append(draw("0" if t == "-" else t))
    return tuple(out)


def show(prog):
    return " ".join("^" if kind(m) == "h" else
                    f"({m[1]})" if kind(m) == "t" else
                    ("-" if m[1] == "0" else m[1]) for m in prog)


def encode(progs, cfg=PEN):
    """Programs -> the arrays that carry the discrete choice.

    `pmask` is the invariant the no-recompile design rests on: a parameter the
    structure does not use is multiplied by exactly 0.0, so it takes exactly
    0.0 gradient and one kernel covers every structure.  A turn move's three
    parameters are all masked off -- it is pure structure.
    """
    b, n = len(progs), cfg.NMOVE
    th = np.zeros((b, n), np.float64)      # arc turn, draws only
    rt = np.zeros((b, n), np.float64)      # in-place rotation, turns only
    dn = np.zeros((b, n), np.float64)      # pen down
    hp = np.zeros((b, n), np.float64)      # pen-up travel
    lv = np.zeros((b, n), np.float64)
    for j, prog in enumerate(progs):
        for i, m in enumerate(prog):
            lv[j, i] = 1.0
            if kind(m) == "h":
                hp[j, i] = 1.0
            elif kind(m) == "t":
                rt[j, i] = ANGLE[m[1]] * np.pi
            else:
                dn[j, i] = 1.0
                th[j, i] = ANGLE[m[1]] * np.pi
    pm = np.stack([(dn + hp) * lv, dn * lv, dn * lv], -1)
    return dict(theta=jnp.asarray(th), rot=jnp.asarray(rt),
                down=jnp.asarray(dn), hop=jnp.asarray(hp),
                live=jnp.asarray(lv), pmask=jnp.asarray(pm))


def prefix(enc, t):
    """The first `t` moves of each program, as an encoding.

    A proposer is trained on partial programs, so every prefix of every
    sampled program is a training example.  Masking the six arrays is enough
    to make one -- no re-encoding, no recompile -- because every discrete
    choice already rides as data and every unused parameter is already
    multiplied by zero.  `t` may be an array, one length per batch element.
    """
    k = (jnp.arange(enc["live"].shape[1])[None, :] < jnp.asarray(t)[..., None])
    k = k.astype(enc["live"].dtype)
    return {n: enc[n] * (k[..., None] if n == "pmask" else k) for n in enc}


# ------------------------------------------------------------------ geometry

def sinc(z):
    """sin(z)/z, exact and smooth at 0; jnp.sinc is sin(pi z)/(pi z)."""
    return jnp.sinc(z / jnp.pi)


def ext_frac(theta):
    """extent / path length, for a constant-curvature move.

    A pen that walks L round a closed loop spans only L/pi, and one that walks
    L in a straight line spans L.  The parameter is and stays ARC length --
    that is the pen's own measure, and an extent normalisation is
    exactly what made two pieces incomparable once they were chained.  But a
    PRIOR that puts the same mass on "walk 30 px" for a line and for a loop
    says a loop should be a third the size of a line, which is not what anyone
    draws.  So the prior means "span 30 px" and divides by this, and the
    deviation bound is a fraction of the span too.

    chord/L is sinc(theta/2); past theta = 2 the diameter 2/theta takes over.
    """
    t = jnp.abs(jnp.asarray(theta))
    return jnp.maximum(jnp.abs(sinc(0.5 * t)), 2.0 / jnp.maximum(t, 2.0))


def bumps(u):
    """Two deviation modes, EXACTLY zero at u = 0 and u = 1.

    Sine modes would be zero only to float32 -- sin(pi) is -8.7e-8 -- and the
    whole point of the basis is that bending a move cannot move the pen that
    follows it.  Polynomials in u(1-u) are zero at the ends bit-exactly.
    Peaks normalised to 1 so BEND is readable as a fraction of the span.
    """
    v = u * (1.0 - u)
    return 4.0 * v, 10.3923 * v * (1.0 - 2.0 * u)


def path(p, enc, cfg=PEN):
    """The pen's whole trajectory, (b, NMOVE, SPS, 2) in pen units.

    One scan-free closed form: headings are an exclusive cumulative sum of the
    turns, and each move's starting point is an exclusive cumulative sum of
    the previous moves' end displacements.  A turn move has length exactly 0,
    so its SPS samples all sit on the pen and it displaces nothing.
    """
    b = p.shape[0]
    n, sps = cfg.NMOVE, cfg.SPS
    q = p[:, :n * NP].reshape(b, n, NP) * enc["pmask"]
    g = p[:, n * NP:]
    th, rt = enc["theta"], enc["rot"]
    dn, hp, lv = enc["down"], enc["hop"], enc["live"]

    # length: an arc length for a draw, a hard-bounded hop for a pen-up move,
    # exactly zero for a turn
    L = (dn * cfg.LREF * jnp.exp(q[..., 0])
         + hp * cfg.HOPMAX * jax.nn.sigmoid(q[..., 0])) * lv

    # heading entering move i = phi0 + every turn taken before it.  A turn
    # move rotates AT its slot; an arc's own turn accumulates across it.
    phi_in = g[:, GI['phi0']:GI['phi0'] + 1] + jnp.cumsum(rt, 1) \
        + jnp.cumsum(th, 1) - th

    # dtype left to jax: under jax_enable_x64 the geometry is then genuinely
    # float64, which is what check_pen.py asserts against.
    u = (jnp.arange(1, sps + 1) / sps)[None, None, :]
    ht = 0.5 * th[..., None] * u
    d = L[..., None] * u * sinc(ht)
    a = phi_in[..., None] + ht
    dx, dy = d * jnp.cos(a), d * jnp.sin(a)

    # template deviation, normal to the local heading
    f1, f2 = bumps(u)
    amp = cfg.BEND * (L * ext_frac(th))[..., None] * (
        jnp.tanh(q[..., 1:2]) * f1 + jnp.tanh(q[..., 2:3]) * f2)
    au = phi_in[..., None] + th[..., None] * u
    dx = dx - amp * jnp.sin(au)
    dy = dy + amp * jnp.cos(au)

    # move i starts where move i-1 ended.  The u = 1 column is the end, and
    # the deviation is exactly 0 there, so bending never shifts the chain.
    ex, ey = dx[:, :, -1], dy[:, :, -1]
    x = (jnp.cumsum(ex, 1) - ex)[..., None] + dx
    y = (jnp.cumsum(ey, 1) - ey)[..., None] + dy
    return jnp.stack([x, y], -1)


def weights(P, enc):
    """Arclength weight per sample, zero wherever the pen is up.

    Computed PER MOVE and only then flattened, so no ink is laid down along a
    pen-up hop and no phantom segment joins the end of one move to the start
    of the next.
    """
    seg = jnp.sqrt(jnp.sum(jnp.diff(P, axis=2) ** 2, -1) + 1e-12)
    w = 0.5 * (jnp.concatenate([seg[:, :, :1], seg], 2)
               + jnp.concatenate([seg, seg[:, :, -1:]], 2))
    return w * (enc["down"] * enc["live"])[..., None]


def place(P, w, p, cfg=PEN):
    """Glyph similarity: slant, scale, then put the drawing's centre of ink at
    tx, ty.

    Anchoring on the inked centroid rather than the pen's start point is what
    makes tx, ty mean "where the drawing is".  Anchored at the start, every
    length change would drag the whole glyph across the canvas and the fit
    would fight itself.  The glyph scale and the per-move lengths are
    deliberately redundant -- one overall size, and relative lengths priced
    around 1 -- which is the same split BPL uses.

    The slant is applied in the PAGE frame, so it does not rotate with the
    glyph.  That is what slant means in type, and what plate perspective does
    to a photographed number; it is also the one reason `phi0` is not an
    exactly rigid rotation of the rendered glyph.
    """
    g = p[:, cfg.NMOVE * NP:]
    x, y = P[..., 0], P[..., 1]
    # Aspect and slant are PAGE properties, like condensed or italic type: a
    # narrow digit is narrow on the page whichever way the pen was moving.
    # Bounded hard by tanh rather than priced, for the reason A_MAX exists in
    # the earlier A_MAX -- unbounded anisotropy is a shape change in disguise, and at
    # log a = 2 a loop is a line and the alphabet has quietly collapsed.
    asp = jnp.exp(cfg.ASPECT * jnp.tanh(g[:, GI['log a']]))[:, None, None]
    x, y = x * asp, y / asp
    x = x + cfg.SLANT * jnp.tanh(g[:, GI['slant']])[:, None, None] * y
    s = jnp.exp(g[:, GI['log s']])[:, None, None]
    x, y = s * x, s * y
    W = w / jnp.sum(w, (1, 2), keepdims=True).clip(1e-6)
    cx = jnp.sum(W * x, (1, 2), keepdims=True)
    cy = jnp.sum(W * y, (1, 2), keepdims=True)
    mid = 0.5 * cfg.H
    x = x - cx + mid + cfg.TRANS * g[:, GI['tx']][:, None, None]
    y = y - cy + mid + cfg.TRANS * g[:, GI['ty']][:, None, None]
    return jnp.stack([x, y], -1), w * s


def tau(P, w, sig, cfg=PEN):
    b = P.shape[0]
    pts, W = P.reshape(b, -1, 2), w.reshape(b, -1)
    ax = (jnp.arange(cfg.H) + 0.5)[None, None].astype(pts.dtype)
    two = (2.0 * sig ** 2)[:, None, None]
    ey = jnp.exp(-(ax - pts[:, :, 1, None]) ** 2 / two) * (W / sig[:, None])[..., None]
    ex = jnp.exp(-(ax - pts[:, :, 0, None]) ** 2 / two)
    return jnp.einsum("bsh,bsw->bhw", ey, ex, precision=jax.lax.Precision.HIGHEST)


def alpha(p, enc, cfg=PEN, down=1, sel=None):
    """Coverage in [0, 1] at H/down.  One pen for the whole glyph.

    `sel` is an optional (b, NMOVE) multiplier on the ink, applied AFTER
    place() so the glyph is still centred on all of its ink: that is what
    makes a per-move on/off toggle show each piece where it really sits
    rather than re-centring it.
    """
    P = path(p, enc, cfg)
    w = weights(P, enc)
    P, w = place(P, w, p, cfg)
    if sel is not None:
        w = w * sel[..., None]
    sig = cfg.SIGMA * jnp.exp(p[:, cfg.NMOVE * NP + GI['log w']])
    a = 1.0 - jnp.exp(-cfg.GAIN * tau(P, w, sig, cfg))
    if down == 1:
        return a
    m = cfg.H // down
    return a.reshape(-1, m, down, m, down).mean((2, 4))


def scale_to_page(p, enc, cfg=PEN, frac=0.72):
    """Set each glyph's `log s` so its ink spans `frac` of the canvas.

    Sampling without this draws programs the prior likes and then throws away
    the ones that walk off the page, which is a rejection step that quietly
    rewrites the length and move-count priors -- fewer moves survive, so the
    samples come out shorter than the prior says.  Fixing the size instead
    leaves the structure prior alone.  It is also what Omniglot does to its
    characters before anyone looks at them: size is not part of the identity.

    So sampled programs are drawn from the prior CONDITIONED on filling the
    page.  `cost` still prices `log s` under its own Gaussian, which is the
    right thing for a fit, where the glyph's size is evidence.
    """
    p = jnp.asarray(p)
    P = path(p, enc, cfg)
    w = weights(P, enc)
    P, w = place(P, w, p, cfg)
    on = (w > 0)[..., None]
    lo = jnp.min(jnp.where(on, P, jnp.inf), (1, 2))
    hi = jnp.max(jnp.where(on, P, -jnp.inf), (1, 2))
    cur = jnp.max(hi - lo, -1).clip(1e-3)
    return p.at[:, cfg.NMOVE * NP + GI['log s']].add(jnp.log(frac * cfg.H / cur))


def pose(p, enc, cfg=PEN):
    """Pen position and heading after each move, (b, NMOVE, 2) and (b, NMOVE).

    Not used by the renderer -- this is what a proposer network is told, and
    what `check_pen.py` asserts the chain against."""
    P = path(p, enc, cfg)
    phi = p[:, cfg.NMOVE * NP + GI['phi0']][:, None] \
        + jnp.cumsum(enc["rot"] + enc["theta"], 1)
    return P[:, :, -1, :], phi


# --------------------------------------------------------------------- prior
# The generative model.  `sample` draws from it; `cost` is its negative log
# density in nats, which is the structure + parameter term of the MDL score.

P_STOP = 0.30                    # stop after this draw
P_HOP = 0.35                     # else: lift and start a new stroke
P_SPIN = 0.45                    # and turn in place before the next draw
W_DRAW = {"0": 3.0, "+45": 1.0, "-45": 1.0, "+90": 2.0, "-90": 2.0,
          "+135": 0.8, "-135": 0.8, "+180": 1.5, "-180": 1.5,
          "+270": 0.6, "-270": 0.6, "+360": 0.35, "-360": 0.35}
W_SPIN = {"+45": 1.0, "-45": 1.0, "+90": 1.3, "-90": 1.3,
          "+135": 0.5, "-135": 0.5, "+180": 0.3}
MU_LEN, SD_LEN = -0.10, 0.35     # log len, draws, measured as SPAN
MU_HOP, SD_HOP = -0.10, 0.80     # pre-sigmoid hop: HOPMAX*sig -> ~9 px.
                                 # Shorter than this and a new stroke lands
                                 # on top of the one before it.
SD_BEND = 0.30                   # the heavy part of "heavily penalised"
# log s is wide because `scale_to_page` sets it from the drawing's own extent:
# a four-move glyph and a one-move glyph need very different s to fill the
# same page, and that is not a strained parameter, it is the size prior doing
# its job.
SD_POSE = np.array([0.15, 0.15, 0.60, 0.30, 0.0, 0.15, 0.30], np.float32)


def mu_len(a):
    """Prior mean of `log len` for a draw angle: the same expected SPAN for
    every primitive, which for a loop means walking pi times as far.

    Thirteen constants, so they are computed once at import.  Through
    `ext_frac` (a jnp function) each call dispatched eager kernels on a 0-d
    array at ~90 us, and `cost` and `mean_params` call it per draw move.
    """
    return MU_LEN_OF[a]


MU_LEN_OF = {a: MU_LEN - float(np.log(ext_frac(ANGLE[a] * np.pi)))
             for a in ANGLE}


def _norm(w):
    t = sum(w.values())
    return {k: v / t for k, v in w.items()}


DP, SP = _norm(W_DRAW), _norm(W_SPIN)


def groups(prog):
    """Split a program into the sampler's own unit: an optional hop, an
    optional turn, then exactly one draw.

    `sample_struct` emits programs one group at a time and `cost` prices them
    the same way, so the generative model and its density cannot drift apart.
    They had: pricing move-by-move missed that the sampler renormalises the
    draw distribution after a straight, and `- +90` came out 1.24x too likely
    under `cost` than under the sampler that produced it.
    """
    out, cur = [], []
    for m in prog:
        cur.append(m)
        if kind(m) == "d":
            out.append(tuple(cur))
            cur = []
    return out


def mean_params(prog, cfg=PEN, hop=None):
    """The prior's mean parameter vector for a program.

    One home for "what does an unfitted program look like": the fitter starts
    restarts here and the sampler draws its reference sheets here, and before
    this they disagreed about a hop by half a log unit.
    """
    p = np.zeros(nparam(cfg), np.float32)
    for i, m in enumerate(prog):
        if kind(m) == "d":
            p[i * NP] = mu_len(m[1])
        elif kind(m) == "h":
            p[i * NP] = MU_HOP if hop is None else hop
    return p


def structures(cfg=PEN, nmax=3):
    """Every legal program up to `nmax` moves.  Only useful for small nmax --
    the point of the write/execute/assess proposer is that this blows up."""
    moves = [HOP] + [draw(a) for a in DRAWS] + [spin(a) for a in SPINS]
    out = []
    for k in range(1, nmax + 1):
        out += [s for s in itertools.product(moves, repeat=k) if legal(s, cfg)]
    return out


def _pick(rng, dist, drop=()):
    """Draw a key from a weight dict, optionally with some keys removed and
    the rest renormalised.  `cost` prices the same restriction."""
    ks = [k for k in dist if k not in drop]
    v = np.array([dist[k] for k in ks])
    return ks[rng.choice(len(ks), p=v / v.sum())]


def _bare_drop(prog, nxt):
    """Which draw angles this group may not use.  A straight immediately
    after a straight is one longer straight, so `legal` forbids it and the
    prior must not propose it."""
    return ("0",) if (not nxt and prog and prog[-1] == ("d", "0")) else ()


def sample_struct(rng, cfg=PEN):
    prog = [draw(_pick(rng, DP))]
    while True:
        if rng.random() < P_STOP:
            break
        nxt = []
        if rng.random() < P_HOP:
            nxt.append(HOP)
        if rng.random() < P_SPIN:
            nxt.append(spin(_pick(rng, SP)))
        nxt.append(draw(_pick(rng, DP, _bare_drop(prog, nxt))))
        if len(prog) + len(nxt) > cfg.NMOVE:
            break
        prog += nxt
    return tuple(prog)


def sample_params(rng, prog, cfg=PEN):
    p = np.zeros(nparam(cfg), np.float32)
    for i, m in enumerate(prog):
        o = i * NP
        if kind(m) == "h":
            p[o] = rng.normal(MU_HOP, SD_HOP)
        elif kind(m) == "d":
            p[o] = rng.normal(mu_len(m[1]), SD_LEN)
            p[o + 1:o + 3] = rng.normal(0.0, SD_BEND, 2)
    g = cfg.NMOVE * NP
    p[g:g + NG] = rng.normal(0.0, 1.0, NG) * SD_POSE
    p[g + 4] = rng.uniform(-np.pi, np.pi)
    return p


def sample(rng, cfg=PEN):
    prog = sample_struct(rng, cfg)
    return prog, sample_params(rng, prog, cfg)


def cost(prog, p, cfg=PEN):
    """(structure nats, parameter nats).  Reported apart because they answer
    different questions: how surprising is this way of drawing, and how
    strained are its numbers."""
    gs = groups(prog)
    ls = -np.log(DP[gs[0][-1][1]])
    done = list(gs[0])
    for g in gs[1:]:
        kinds = [kind(m) for m in g]
        nxt = [m for m in g if kind(m) != "d"]
        ls -= np.log(1.0 - P_STOP)
        ls -= np.log(P_HOP if "h" in kinds else 1.0 - P_HOP)
        if "t" in kinds:
            ls -= np.log(P_SPIN) + np.log(SP[[m for m in g if kind(m) == "t"][0][1]])
        else:
            ls -= np.log(1.0 - P_SPIN)
        drop = _bare_drop(done, nxt)
        tot = sum(v for k, v in DP.items() if k not in drop)
        ls -= np.log(DP[g[-1][1]] / tot)
        done += list(g)
    ls -= np.log(P_STOP)

    def gauss(x, mu, sd):
        return 0.5 * ((x - mu) / sd) ** 2 + np.log(sd * np.sqrt(2 * np.pi))

    lp = 0.0
    for i, m in enumerate(prog):
        o = i * NP
        if kind(m) == "h":
            lp += gauss(p[o], MU_HOP, SD_HOP)
        elif kind(m) == "d":
            lp += gauss(p[o], mu_len(m[1]), SD_LEN)
            lp += sum(gauss(p[o + j], 0.0, SD_BEND) for j in (1, 2))
    g = cfg.NMOVE * NP
    for j, sd in enumerate(SD_POSE):
        lp += gauss(p[g + j], 0.0, sd) if sd > 0 else np.log(2 * np.pi)
    return float(ls), float(lp)
