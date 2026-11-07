"""Step 5: the proposer's trunk and its four heads, hand-rolled.

No Flax, no Haiku.  The model is ~40k parameters in a dict of arrays, Adam is
nine lines, and the whole thing is one `jax.jit` -- which is worth more here
than a framework would be, because the point of this stage is to be able to
say exactly what the network sees and exactly what it is trained on.

                              the trunk

    in    5 x 64 x 64   target, canvas, pen dot, cos phi, sin phi
          conv 3x3 x32  relu  maxpool 2      -> 32 x 32 x 32
          conv 3x3 x32  relu  maxpool 2      -> 32 x 16 x 16
          conv 3x3 x32  relu                 -> 32 x 16 x 16
          conv 3x3 x16  relu                 -> 16 x 16 x 16
    emb   global average pool                   16
          ++ BILINEAR READOUT AT THE PEN TIP    16
          -> dense 32 -> 256, relu            =  e

The pen-tip readout is the one piece that is not standard.  Pooling alone
destroys locality, and the next move starts *at the pen*: what is near the pen
is what decides whether to turn, how far to go, or to stop.  It is a readout
and not a reframing -- no rotation, no resampling -- so the convolutions stay
translation-equivariant and the glyph pose stays the scene head's problem.

                        the head, autoregressive

    kind    softmax(W_k e)                        4    draw turn lift stop
    angle   softmax(W_a [e ; emb kind])          13    masked per kind
    length  softmax(W_l [e ; emb kind ; emb a])   8    draw and lift only
    value   sigmoid(W_v e)                        1

    log pi(a) = log p(kind) + log p(angle|kind) + log p(length|kind, angle)

Three conditionals rather than three independent softmaxes, because the
factors are correlated -- a 90 degree turn is usually followed by a long
straight -- and independent heads can only ever propose the product of the
marginals.  The extra cost is two small matmuls on a 256-vector.

ILLEGAL ACTIONS ARE MASKED TO -inf, from `data.masks`, which is `dsl.legal`
evaluated incrementally rather than a second copy of its rules.  This matters
for SMC and not merely for tidiness: without it particles spend their mass on
distinct action sequences that draw the same picture, and the particle set
silently shrinks.

The value head is a REGRESSION on IoU, not an RL critic (see the README): on
synthetic data the suffix is known, so the number it predicts is computed, not
bootstrapped.

Usage: .venv/bin/python -m pen.net        (shapes, a gradient check, timing)
"""

import time

import jax
import jax.numpy as jnp
import numpy as np

from . import data as DA

WIDTH, EMB, KEMB = 32, 256, 16
NK, NA, NL = DA.NK, DA.NA, DA.NL
CHAN = (DA.NCH, WIDTH, WIDTH, WIDTH, WIDTH // 2)      # 5 -> 32 32 32 16
POOL = (True, True, False, False)
FEAT = CHAN[-1]
STRIDE = 4                                            # two maxpools: 64 -> 16


def init(rng, cfg=DA.CFG):
    """He-scaled normals, one dict, no module objects."""
    def n(*s, fan):
        return np.asarray(rng.normal(0, np.sqrt(2.0 / fan), s), np.float32)
    w = {}
    for i in range(len(CHAN) - 1):
        ci, co = CHAN[i], CHAN[i + 1]
        w[f"c{i}"] = n(co, ci, 3, 3, fan=ci * 9)
        w[f"b{i}"] = np.zeros(co, np.float32)
    w["e"] = n(2 * FEAT, EMB, fan=2 * FEAT)
    w["eb"] = np.zeros(EMB, np.float32)
    w["kind"] = n(EMB, NK, fan=EMB)
    w["kb"] = np.zeros(NK, np.float32)
    w["ek"] = n(NK, KEMB, fan=NK)
    w["ea"] = n(NA, KEMB, fan=NA)
    w["ang"] = n(EMB + KEMB, NA, fan=EMB + KEMB)
    w["ab"] = np.zeros(NA, np.float32)
    w["len"] = n(EMB + 2 * KEMB, NL, fan=EMB + 2 * KEMB)
    w["lb"] = np.zeros(NL, np.float32)
    w["val"] = n(EMB, 1, fan=EMB)
    w["vb"] = np.zeros(1, np.float32)
    return {k: jnp.asarray(v) for k, v in w.items()}


def nparam(w):
    return int(sum(np.prod(v.shape) for v in w.values()))


def _conv(x, k, b):
    y = jax.lax.conv_general_dilated(x, k, (1, 1), "SAME",
                                     dimension_numbers=("NCHW", "OIHW", "NCHW"))
    return jax.nn.relu(y + b[None, :, None, None])


def _pool(x):
    return jax.lax.reduce_window(x, -jnp.inf, jax.lax.max,
                                 (1, 1, 2, 2), (1, 1, 2, 2), "SAME")


def readout(f, tip):
    """One bilinear sample of the feature map at the pen, (B, FEAT).

    `tip` is in target pixels; the map is at stride 4 with its first cell
    centred on pixel 1.5, so the feature coordinate is tip/4 - 0.5.  Clamped
    rather than zero-padded: a pen just off the canvas should read the edge
    feature, not a hole that looks like empty paper.
    """
    n, _, h, _ = f.shape
    u = jnp.clip(tip / STRIDE - 0.5, 0.0, h - 1.0)
    x0 = jnp.floor(u[:, 0]).astype(jnp.int32)
    y0 = jnp.floor(u[:, 1]).astype(jnp.int32)
    x1, y1 = jnp.minimum(x0 + 1, h - 1), jnp.minimum(y0 + 1, h - 1)
    fx, fy = u[:, 0] - x0, u[:, 1] - y0
    g = lambda yy, xx: f[jnp.arange(n), :, yy, xx]
    top = g(y0, x0) * (1 - fx)[:, None] + g(y0, x1) * fx[:, None]
    bot = g(y1, x0) * (1 - fx)[:, None] + g(y1, x1) * fx[:, None]
    return top * (1 - fy)[:, None] + bot * fy[:, None]


def trunk(w, x, tip):
    for i in range(len(CHAN) - 1):
        x = _conv(x, w[f"c{i}"], w[f"b{i}"])
        if POOL[i]:
            x = _pool(x)
    e = jnp.concatenate([x.mean((2, 3)), readout(x, tip)], 1)
    return jax.nn.relu(e @ w["e"] + w["eb"])


def _mask(logits, m):
    """-1e30, not -inf: an all-masked row would otherwise make the softmax
    NaN and poison the gradient of every other row in the batch."""
    return jnp.where(m > 0, logits, -1e30)


def heads(w, e, kind, ang, kmask, amask, lmask):
    """Logits for the three factors, CONDITIONED on the given kind and angle.

    At training that is teacher forcing on the generator's own move; when
    sampling, `kind` and `ang` are what the previous factor just produced.
    A factor the kind does not have is passed as -1 and embeds to zero.
    """
    ek = jnp.where(kind[:, None] >= 0, w["ek"][jnp.maximum(kind, 0)], 0.0)
    ea = jnp.where(ang[:, None] >= 0, w["ea"][jnp.maximum(ang, 0)], 0.0)
    lk = _mask(e @ w["kind"] + w["kb"], kmask)
    la = _mask(jnp.concatenate([e, ek], 1) @ w["ang"] + w["ab"],
               amask[jnp.arange(len(e)), jnp.maximum(kind, 0)])
    ll = _mask(jnp.concatenate([e, ek, ea], 1) @ w["len"] + w["lb"],
               lmask[jnp.arange(len(e)), jnp.maximum(kind, 0)])
    # v is a function of the STATE, not of the action being scored: it is
    # what SMC weights a particle by, before any continuation is chosen.
    v = jax.nn.sigmoid(e @ w["val"] + w["vb"])
    return lk, la, ll, v[:, 0]


def forward(w, b):
    e = trunk(w, b["x"], b["tip"])
    return heads(w, e, b["act"][:, 0], b["act"][:, 1],
                 b["kmask"], b["amask"], b["lmask"])


def _xent(logits, y, w=None):
    """Cross entropy on the rows where the label exists (y >= 0), weighted.

    `w` is how the policy and the value part company.  The value head needs
    mismatched targets -- with matched ones alone every label is 1.0.  The
    POLICY must not see them: the action label is the move the generator
    made, and against a target that generator did not draw, that move is not
    the right answer to the question the policy is being asked.  Training on
    them is training on noise, and it showed -- the length head came out
    BELOW the prior-alone control (0.226 against 0.268) until this mask
    existed, because length is the factor that depends most on the target
    actually being the one the move draws.
    """
    ok = (y >= 0).astype(logits.dtype)
    if w is not None:
        ok = ok * w
    lp = jax.nn.log_softmax(logits)
    pick = lp[jnp.arange(len(y)), jnp.maximum(y, 0)]
    return -(pick * ok).sum() / jnp.maximum(ok.sum(), 1.0)


def loss(w, b, use=(1.0, 1.0, 1.0, 1.0)):
    """(total, parts).  `use` weights kind, angle, length, value.

    `use = (0, 0, 0, 1)` is the value head alone, which is step 6's first
    experiment and the cheapest way to settle risk 2: if a Markov value
    cannot separate good prefixes from bad, a Markov policy will not propose
    coherent continuations either.
    """
    lk, la, ll, v = forward(w, b)
    a = b["act"]
    m = (b["mode"] != 2).astype(lk.dtype)        # matched and divergent
    pk = _xent(lk, a[:, 0], m)
    pa = _xent(la, a[:, 1], m)
    pl = _xent(ll, a[:, 2], m)
    pv = jnp.mean((v - b["val"]) ** 2)
    parts = jnp.stack([pk, pa, pl, pv])
    return jnp.sum(parts * jnp.asarray(use)), parts


def accuracy(w, b):
    """(kind, angle, length) top-1 on the MATCHED rows each factor applies
    to, and the value head's MSE and correlation over all of them.

    Matched only, for the same reason the loss is: on a divergent target
    there is no right answer to score against.
    """
    lk, la, ll, v = forward(w, b)
    a = b["act"]
    out = []
    for lg, y in ((lk, a[:, 0]), (la, a[:, 1]), (ll, a[:, 2])):
        ok = (y >= 0) & (b["mode"] != 2)
        out.append(float((lg.argmax(1)[ok] == y[ok]).mean()))
    vv, tv = np.asarray(v), np.asarray(b["val"])
    out += [float(np.mean((vv - tv) ** 2)),
            float(np.corrcoef(vv, tv)[0, 1]) if vv.std() > 1e-6 else 0.0]
    return out


# --------------------------------------------------------------------- Adam

def adam_init(w):
    z = lambda t: jax.tree.map(jnp.zeros_like, t)
    return (z(w), z(w), 0)


def adam(w, g, st, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
    m, v, t = st
    t = t + 1
    m = jax.tree.map(lambda a, b: b1 * a + (1 - b1) * b, m, g)
    v = jax.tree.map(lambda a, b: b2 * a + (1 - b2) * b * b, v, g)
    mh = jax.tree.map(lambda a: a / (1 - b1 ** t), m)
    vh = jax.tree.map(lambda a: a / (1 - b2 ** t), v)
    w = jax.tree.map(lambda p, a, b: p - lr * a / (jnp.sqrt(b) + eps), w, mh, vh)
    return w, (m, v, t)


@jax.jit
def _step(w, st, b, use, lr):
    (tot, parts), g = jax.value_and_grad(loss, has_aux=True)(w, b, use)
    w, st = adam(w, g, st, lr)
    return w, st, tot, parts


def step(w, st, b, use=(1.0, 1.0, 1.0, 1.0), lr=1e-3):
    return _step(w, st, b, jnp.asarray(use, jnp.float32), lr)


def jnpify(b):
    """The numpy batch `data.batch` returns, minus the python-side fields."""
    return {k: jnp.asarray(v) for k, v in b.items()
            if k in ("x", "tip", "act", "kmask", "amask", "lmask", "val",
                     "mode")}


def main():
    rng = np.random.default_rng(0)
    w = init(rng)
    print(f"{nparam(w)} parameters, {jax.default_backend()}")
    b = jnpify(DA.batch(rng, 32))
    lk, la, ll, v = forward(w, b)
    print(f"logits {lk.shape} {la.shape} {ll.shape}, value {v.shape}")
    assert jnp.isfinite(lk).all() and jnp.isfinite(v).all()

    # a masked action must be unreachable, not merely unlikely
    p = jax.nn.softmax(lk)
    assert float(jnp.max(p * (1 - b["kmask"]))) < 1e-12, "mask leaks"
    print(f"masked kinds carry {float(jnp.max(p * (1 - b['kmask']))):.1e} "
          f"probability")

    st = adam_init(w)
    w, st, tot, parts = step(w, st, b)       # compile
    t0 = time.time()
    for _ in range(20):
        w, st, tot, parts = step(w, st, b)
    jax.block_until_ready(tot)
    dt = (time.time() - t0) / 20
    print(f"loss {float(tot):.3f}  parts {np.asarray(parts).round(3)}")
    print(f"{dt * 1e3:.1f} ms per step at batch {len(b['val'])}, "
          f"{len(b['val']) / dt:.0f} examples/s")

    # it must be able to overfit one batch -- if it cannot, nothing else
    # measured later means anything
    for i in range(300):
        w, st, tot, parts = step(w, st, b, lr=3e-3)
    k, a, l, mse, r = accuracy(w, b)
    print(f"after 300 steps on ONE batch: kind {k:.2f} angle {a:.2f} "
          f"length {l:.2f}, value mse {mse:.4f} r {r:.2f}")


if __name__ == "__main__":
    main()
