# The `pen` package

A BPL-style pen DSL for digits, its gradient fitter, and the measured gate
that says whether the DSL can express an SVHN digit at all.  `dsl.py` holds
the alphabet, the geometry and the prior; `fit.py` fits the numbers; `check.py`
asserts the invariants the design rests on; `sample.py` draws from the prior;
`gate.py` is the measurement.  Run them from the repo root as `python -m
pen.check`, `python -m pen.sample [seed]`, `python -m pen.gate [ink]`.

The rest of this file is the design note for the next piece: the network that
proposes program STRUCTURE, which gradient descent then refines.

## Proposing pen programs: write/execute/assess, minus most of it

Ellis, Nye, Pu, Sosa, Tenenbaum, Solar-Lezama, *Write, Execute, Assess:
Program Synthesis with a REPL*, NeurIPS 2019 (arXiv:1906.04604).

### What the paper does

Search in **semantic** space, not syntactic space. A partial program is
executed as it is written, so the searcher sees what the code *did*, not what
it says. Three parts:

1. **REPL.** Every partial program is rendered to its own canvas. State is
   `(set of partial programs, spec)`; an action is one line of code.
2. **Policy** `pi(a | [[pp]], spec)` over actions, conditioned on the rendered
   canvases and the target. **Value** `v([[pp]], spec)`, the probability a
   rollout from here ends up satisfying the spec.
3. **SMC at test time.** K particles; propose with `pi`, reweight by `v`,
   resample. Doubling K until a 2-minute timeout.

For 2D CSG: 64x64 inputs; 4 conv layers of 3x3 (32, 32, 32, 16 channels),
ReLU, 2x2 maxpool after the first two; each canvas encoded separately and the
activations **summed** (Deep Sets), concatenated with the spec encoding; a
one-layer GRU **pointer network** with 512 hidden units for the policy, so the
decoder can point at objects already in scope; a 512-unit MLP for the value,
with a negated-softplus output. Trained on **random programs sampled from the
CFG and executed** -- the spec is free and the ground-truth action sequence is
known, so pretraining is plain supervised likelihood. Then REINFORCE
fine-tuning, because the reward (did the rollout match?) is binary and not
differentiable. Branching factor ~1.3M per line; about three days of
pretraining plus five of RL.

### What we keep

- **Render during rollout.** This is the whole idea and it is free for us:
  `pen.prefix(enc, t)` is the first `t` moves of a program, and
  `pen/check.py` asserts it renders *bit-identically* to the prefix program.
  One batched call gives every prefix of every program in the batch, with no
  re-encoding and no recompile, because every discrete choice already rides as
  data and every unused parameter is already multiplied by zero.
- **Train on synthetic data.** `pen.sample` draws a program from the prior and
  `pen.alpha` executes it. A program of T moves is T training examples. We
  never need a labelled digit.
- **SMC with a learned value as the weight**, then gradient descent on the
  continuous parameters of the surviving structures.

### What we drop, and why our DSL lets us

**1. No pointer network. No Deep Sets. One canvas.**
CSG needs pointers because `union` and `difference` take *arguments* that are
earlier objects, so the decoder must name one. A pen move takes no arguments.
It is appended to a chain, and everything it needs to know about the past is
the pen's position and heading. The action space collapses from ~1.3M per step
to:

    kind    draw | turn | hop | stop                     4
    angle   13 for a draw, 7 in place for a turn        13
    length  8 log-spaced buckets                         8

21 move symbols, ~200 actions as a factorised head. That is four orders of
magnitude, and it comes from the DSL, not from the network.

**2. No recurrence. The policy is Markov.**
The paper keeps a GRU over the action history even though it has a REPL. But
if the REPL's claim is true -- the rendered state exposes the semantics --
then for a *sequential* DSL the canvas plus the pen state is a sufficient
statistic for what remains to be drawn. Nothing earlier can matter: the pen
cannot reach back. So drop the decoder RNN and make the policy feed-forward.

The pen state goes in as image channels rather than as a side vector, so the
whole thing stays convolutional and translation-equivariant:

    ch 0   target
    ch 1   canvas so far (alpha of the prefix)
    ch 2   a small Gaussian blob at the pen tip
    ch 3   cos(heading) * that blob
    ch 4   sin(heading) * that blob

**Invariance is learned, not built in.** `phi0` rotates the drawing rigidly
(measured: 2e-13 px over 200 programs x 7 angles), so the DSL is already
rotation-invariant -- but the network sees pixels, and a rotated 5 is a
different image. The tempting fix is to canonicalise the input into the pen
frame. We are not doing that: it throws away uprightness, which is real
evidence (a 5 is not a rotated 2), and it buys an invariance the generator can
simply teach. The augmentation is not a pipeline, it is the prior: `phi0` is
already uniform on (-pi, pi], with `log s`, `slant`, `tx`, `ty` drawn
alongside, so every batch is already pose-augmented. Widening the prior is the
whole implementation. The cost is honest -- learned invariance needs more
data and capacity than built-in invariance -- but data is free here and the
binding constraint is CPU time, i.e. risk 4.

**3. No RL stage. The value head is a regression.**
The paper needs REINFORCE because its reward is binary success, which it
cannot differentiate or observe partially. Ours need not be binary. On
synthetic data we know the suffix, so we can compute, for free, the IoU the
*generating* completion reaches from this prefix, and regress it:

    v(target, canvas, pen) -> IoU(alpha(full program), target)

Same supervised pass as the policy, same batch, no rollouts, no credit
assignment. **The honest caveat:** this is the value of the completion the
generator happened to choose, not of the best reachable one, so it is a biased
(pessimistic where the prior is unlucky, optimistic where the policy is worse
than the prior) estimate. It is still monotone in the right direction, which
is all SMC needs from a weight.

**4. The length head is coarse, on purpose.**
8 buckets, and gradient descent fixes the rest. The proposer's job is to land
in the right basin; the fitter's job is to be exact. The paper quantises its
coordinates to 16x16 for the same reason, and has no fitter to fall back on.

### Architecture, concretely

#### Trunk

    input    C x 64 x 64
             C = target (1 or 3) + canvas alpha (1) + pen state (3)
             conv 3x3 x32, relu, maxpool 2      -> 32 x 32 x 32
             conv 3x3 x32, relu, maxpool 2      -> 32 x 16 x 16
             conv 3x3 x32, relu                 -> 32 x 16 x 16
             conv 3x3 x16, relu                 -> 16 x 16 x 16

    embed    global average pool                   16
             ++ bilinear readout at the pen tip    16
             -> MLP 32 -> 256, relu              =  e

The pen-tip readout is one bilinear sample of the feature map at the pen's
current position. Global pooling alone would destroy locality, and the next
move starts *at the pen*, so what is near the pen is what matters. It is a
readout, not a reframing: no rotation, no resampling of the image.

#### The proposal head

The action space is factorised, and the factors are **autoregressive within
one step**. Three tiny conditional softmaxes off the same embedding:

    kind    softmax(W_k e)                              4   draw turn lift stop
    angle   softmax(W_a [e ; emb kind])                 13  draw and turn only,
                                                            masked to the 7
                                                            in-place angles
                                                            when kind = turn
    length  softmax(W_l [e ; emb kind ; emb angle])     8   draw and lift only

    value   sigmoid(W_v e)                              1

    log pi(a) = log p(kind) + log p(angle | kind) + log p(length | kind, angle)

**Why autoregressive and not three independent heads.** The factors are
correlated: a 90-degree turn is usually followed by a long straight, no turn
by a short arc. Independent softmaxes cannot represent that and would propose
the product of the marginals. Three conditionals can, and the extra cost is
two small matmuls on a 256-vector -- the whole head is ~6k parameters against
~150k in the trunk. Sampling stays exact and one-pass.

**Illegal actions are masked to -inf**, not merely trained against: no turn
after a turn, no lift after a lift, no second lift without a draw between, no
stop before the first draw, nothing past NMOVE. This is `legal()` evaluated
incrementally. It matters for SMC, not just for tidiness -- without it,
particles spend their mass on distinct action sequences that are the same
drawing, and the particle set silently shrinks.

**What the head does NOT propose:** bend/wig, the glyph pose, and the exact
lengths. Those start at their prior means and at the bucket centre, and Adam
finds the rest. Discrete structure is what search is for; continuous
parameters are what gradients are for, and that split is the only reason this
is cheaper than the paper.

**Length buckets** are 8 bins uniform in log *span* over [0.3, 2.0] x LREF,
which is span and not arc length so the bins mean the same thing for a line
and for a loop (see `pen.ext_frac`). The bucket centre is the Adam init.

**Where library entries will attach.** `kind` is a separate head precisely so
it can grow a fifth value, `macro`, with a second-level softmax over the
learned library. An abstraction round then adds columns to two small matrices
and leaves the trunk alone -- but the head must be retrained, because the
prior it is amortising has changed.

#### Scene parameters are not moves

SVHN needs shear, ink and background colour, blur, and neighbouring-glyph
clutter. None of those are things a pen *does*, and none belong in the move
alphabet:

  * **shear** is already there, as the glyph's `slant` (page frame, so it does
    not rotate with the glyph -- which is what slant means in type, and what
    plate perspective does).
  * **colour, tone, blur, background gradient** are a scene layer *wrapping*
    the pen layer. That is the scene model, still to be written: background,
    per-channel tone, a separable blur, and a crop that composes into the
    affine instead of resampling.
  * **clutter** is the neighbouring glyphs of the string, each its own pen
    program on the shared baseline. Causally that is what the edge
    distractors are.

So they get a **one-shot scene head** off the same trunk at step 0,
regressing the ~15 scene numbers from the target alone -- they are global, not
sequential, and nothing about them is a search problem. Then the background is
explained away and the pen rollout works on the residual, which is close to
the clean ink map the policy was trained on. That shrinks the domain gap as a
side effect rather than by hoping.

### The gate, measured

`pen/gate.py`, 12 polarity-matched SVHN crops, the digit's label used to pick
one or two hand-written programs (plus each one's mirror, to keep my own sign
conventions out of the result), fitting the numbers only. 552 fits, ~82 s on
CPU, ~148 ms/fit including one XLA compile, at 400 Adam iterations and 12
restarts swept over `phi0`.

    blank beats the fit    0 / 12
    IoU median             0.60 full frame, 0.71 centred
    IoU >= 0.5             10 / 12 full frame, 9 / 12 centred
    R2 median              +0.41
    mirror won             7 / 12

**The blank-render failure did not happen.** Study 1 measured 21% of runs
converging to drawing nothing against a single stroke on an empty field, and
`[[ink-weighted-loss]]` found flat MSE's own optimum on a faint target IS the
blank image. Here no fit was beaten by the blank, on any crop. Worth saying
plainly because it was risk 1 and it was expected to bite.

**45% of the target ink is outside the central window** -- it is a
neighbouring digit. SVHN crops carry slices of the rest of the house number,
and a pen program draws ONE glyph, so that ink is in the residual by
construction and no fit can remove it. The full-frame IoU is therefore a
measure of the crop as much as of the model, which is why both numbers are
reported. It is also the argument for the string sketch (three glyphs on a
shared baseline) being a modelling decision rather than a refinement.

**The mirror won on 7 of 12**, which is the uncomfortable number here. It
means my hand-written programs are not reliably the better-oriented of the
two, so the gate is measuring the DSL's reach and not my handwriting -- which
is what the mirrors were for -- but it also says the `HAND` table should not
be read as a claim about how people draw digits. A proposer trained on the
prior has no such bias, and step 3's enumeration will replace the table
outright.

### Three kinds of failure, and how to tell them apart

Worth separating, because only the first is the proposer's problem:

* **search** -- wrong structure, or the right one started in the wrong basin.
  Expected now. Test: refit with 5x the restarts. If the answer moves, it was
  search.
* **model** -- the right structure cannot express the target at any
  parameters. Test: the answer does NOT move with more restarts, and some
  hand-constructed better picture scores better too.
* **objective** -- the model can express it, the loss ranks it worse. Test:
  the answer does not move, and a better-looking picture scores WORSE.

The distinction earns its keep immediately. Crop 15508, a `0` drawn as one
`+360` loop, sat at centred IoU 0.38 and would not move: 12 restarts, 64
restarts and a dedicated `log s` sweep all returned 0.38-0.39 at mse 0.104.
I read that as an objective failure and reached for the ink weighting. It was
a **model** failure: a pen walking a constant-curvature loop draws a CIRCLE,
the glyph transform had `slant` but no **aspect**, and a shear does not make
a symmetric shape narrow. The target `0` is a narrow italic ellipse, so the
loop could match its width or its height and compromised by shrinking.

One bounded parameter fixed it -- aspect in the page frame, hard-bounded by
tanh at [0.64, 1.57] for the reason the earlier A_MAX existed, since
unbounded anisotropy turns a loop into a line and collapses the alphabet:

    15508 centred IoU   0.38 -> 0.67        mse 0.1043 -> 0.0766
    all 12, centred     0.69 -> 0.71        R2 +0.34 -> +0.41

**The ink weighting, tried on the wrong diagnosis, did not help.**
`w = 1 + a*target` at `a = 4`:

    centred IoU median      0.69 -> 0.49       4 crops better, 8 worse
    corr(change, edge ink)  +0.01              not the neighbours
    corr(change, flat IoUc) -0.68              helps bad fits, hurts good ones

`a = 4` is simply too strong: it drags every fit toward a middling fattened
stroke. Choosing `a` properly means a sweep on *held-out* crops -- never on
the ones that failed, which is the discipline the note itself asks for -- so
`ink_alpha` stays 0 and is not tuned here.

**A hand-written program was also wrong**, which is what the mirror control
is for: a `3` is not `+180 +180`. With no turn between them the curvature
continues and two half-turns are a circle. The pen finishes the top bowl
heading back the way it came, so a `3` is `+180 (+180) +180`.

### Risks

1. **Domain gap, the big one.** The paper trains and tests on the same
   generator. We would train on prior samples and run on SVHN crops. The
   samples look like pen drawings (`out_shots/pen_random.png`) but they are
   not digits, and a policy that has never seen a 3 may not propose one.
   Mitigation, in order: widen the prior; then a wake-sleep loop -- fit real
   crops with the search, keep what scores well, and retrain the proposer on
   those as pseudo-labels. That loop is DreamCoder's sleep phase, so it is
   where this was always going.
2. **The Markov assumption is a claim, not a theorem.** It is true of the
   *geometry* -- the pen cannot reach back -- but a human draws a 4's crossbar
   knowing they drew a stem. If the canvas does not carry that, the policy
   will propose incoherent continuations. Cheap test before building the
   policy: train only the value head and check it separates good prefixes from
   bad ones. If a Markov value cannot, a Markov policy will not either.
3. **The value target is biased** (above). Measure it: on held-out synthetic
   data, compare regressed value against the IoU the *search* actually
   reaches, not against the generator's.
4. **No GPU on this box right now.** `nvidia-smi` cannot reach the driver and
   jaxlib falls back to CPU -- which is also the likely reason the rigid
   search cost ~10 min/image. Training a CNN on CPU is the difference between
   an afternoon and a week. Fix the driver before starting step 2 of anything
   here.

### A string of digits, and why there is no union operator

The DSL has no operator for "draw this glyph and that one". It does not need
one: **union is already the sum inside `tau`.**

`tau` flattens the sample axis and runs a single einsum, and coverage is
`1 - exp(-GAIN * tau)` -- an exponential of a SUM. So k glyphs composite by
concatenating along that axis: one longer einsum, not k einsums, and no
compositing step. Measured: concatenating two glyphs' samples into one einsum
equals summing their separate `tau`s to 2.2e-16, float64 epsilon, and the
gradient reaches each glyph's own `tx` independently (-94.9 and +94.9 on a
two-glyph string). Soft union is free and already differentiable.

What a string actually needs is three changes, none of them to the alphabet.
All three are the same kind of change -- one glyph-id array in `enc`, exactly
the shape `sel` already is:

1. **`place` centres on the inked centroid of everything.** `cx = sum(W*x,
   (1,2))` reduces over all moves, so two glyphs in one program would each
   drag the other's centring. It becomes a per-glyph segment sum.
2. **The heading carries across the whole program.** `phi_in = phi0 +
   cumsum(rot) + cumsum(theta)` would make glyph 2's orientation a function
   of how much glyph 1 happened to turn. It becomes a segment-cumsum that
   resets per glyph.
3. **The glyph block becomes per-glyph** -- `NG = 7` once becomes 7 each,
   plus SHARED baseline parameters (pitch, slant).

Point 3 is the one that matters. A shared baseline is what makes neighbours
**explanatory** rather than free ink: they cannot sit wherever the loss wants
them, they have to sit at the focal digit's pitch and slant. Free-floating
neighbour glyphs are just extra ink to absorb the residual -- including
absorbing the focal digit, which is risk 3.

**The hop is not the mechanism.** `HOPMAX = 20 px`, prior mean ~9.5 px at
H = 64: it is a within-glyph pen lift, the gap in a 4 or a 5. Building strings
out of hops would conflate "the pen lifted" with "this is a different digit",
and that destroys the property the whole design rests on -- a glyph is one
program whose structure is comparable across instances. A string is k glyph
programs sharing a baseline, not one long program.

One limit on the free-union claim: it is union in ONE layer, with one colour
and one pen. Occlusion, and digits of different colours, need genuine
over-compositing with layers. That is a different mechanism and is the scene
model's job, not this one's.

Why it is worth doing: 45% of the gate's target ink is neighbouring digits,
in the residual by construction, and it is the largest single term left. Why
it is not next: searching k glyphs jointly multiplies the structure space --
three glyphs over the 3705 programs of <= 3 moves is 3705^3 -- so a string is
tractable only WITH a proposer, or with a staged fit (focal glyph first,
neighbours second). It belongs after the proposer, not before it.

### Order of work

The fitter comes before the network, which is not the paper's order. The
reason is that it is the cheapest thing that can prove the DSL wrong, and the
network is the most expensive thing to build on top of a DSL that is wrong.
It is also a dependency of steps 4 and 5: the value target is an IoU, which
needs fits, and "did the search work" has no answer without enumeration to
compare against.

    1  dsl.py, check.py, sample.py                    done, and corrected
                   once more at step 6: `cost` was not a density.  Shrink
                   NMOVE until the whole space enumerates and the prior's
                   total mass must be exactly 1 -- it was 0.4957 at NMOVE 3,
                   1.1103 at 4, 1.4410 at 5, three separate places where the
                   sampler and its density disagreed:

                     * a FORCED stop priced as a chosen one.  `sample_struct`
                       breaks when the next group will not fit; `cost` still
                       charged -log(P_STOP).  The right term is P_STOP +
                       (1 - P_STOP) P(the group does not fit), and a 3-move
                       program at NMOVE 3 came out 3.34x too unlikely, which
                       is 1 / P_STOP exactly
                     * `legal` accepted hop/turn ORDERINGS the sampler never
                       emitted.  Turning before a hop aims the hop, turning
                       after it aims the next stroke, and they start the next
                       stroke in different places -- so these are different
                       drawings, not a canonical form
                     * and two turns in one group (`- (-90) ^ (-90) -`),
                       which needs five moves to appear at all, which is why
                       only NMOVE 5 showed it

                   Fixed at the root rather than in `cost` a third time: a
                   stroke boundary is (turn?, hop, turn?) in the sampler AND
                   in `legal`, both turn slots independent, one pen-up per
                   gap.  Mass is now 1.000000 at NMOVE 3, 4 and 5 with zero
                   illegal samples, and `check.py` carries that by
                   enumeration -- the old Monte-Carlo version could not see
                   any of it, because at NMOVE 12 only programs of 10+ moves
                   are affected and no frequency test on common structures
                   reaches them.

                   `structures(<= 3)` is still exactly 3705 and step 3's
                   rankings stand; a program containing `^` costs +0.6
                   structure nats more under the corrected prior, against 78
                   nats between rival structures.

                   What the prior actually draws, 40k samples: median 4
                   moves, mean 4.83, and bimodal -- 30% are ONE stroke and
                   13% are stopped by the NMOVE cap rather than by the stop
                   rule.  Per program 2.98 draws, 1.17 turns, 0.67 hops,
                   1.67 strokes.  Only 48% fall inside the <= 3-move space
                   step 3 enumerated, which is worth remembering when the
                   proposer is measured against that table.
    2  fit.py      batched Adam on the continuous parameters.  Then the
                   GATE, gate.py: hand-write one
                   program per digit for 12 polarity-matched SVHN crops and
                   fit only the numbers.
                   DONE, and it passes.  Blank beats the fit on 0/12 (risk 1
                   does not bite yet).  IoU median 0.60 full frame, 0.71 over
                   the centre 20 px, >= 0.5 on 10/12 and 9/12; R^2 median
                   +0.41.  45% of the target ink is a neighbouring digit the
                   program cannot draw, which is why the centred number is
                   the fair one.  The one real failure, crop 15508, was a
                   MODEL failure and is fixed: a pen loop is always a circle,
                   so the glyph needed a bounded aspect (0.38 -> 0.67).
                   Cost on this box, CPU, warm (compile excluded): 130 ms
                   per fit at 400 Adam iterations, 50 ms at 150.
    3  enumerate   all 3705 programs of <= 3 moves (`dsl.structures`), fit,
                   rank by MDL.  This is the ground truth steps 4-5 are
                   measured against.  Do it in two stages, and do SYNTHETIC
                   targets before real ones.

                   3a self-generated targets, <= 2 moves (181 legal) as both
                      the generator and the hypothesis space.  On a target
                      rendered from a known program the residual goes to zero
                      for the true structure, so this is the only place "did
                      the ranking work" has a ground truth.  DONE, and the
                      objective passes: recall 8/8, recovery 7/8, explained
                      8/8, and the 8th is a genuine equivalence rather than a
                      wrong answer -- for a SINGLE arc, `+a` and `-a` differ
                      by a 180 degree rotation, which `phi0` can produce, so
                      they are the same picture and no score can separate
                      them.  15,680 fits, 33 minutes.

                      Getting there took one correction worth recording.  The
                      first run said recovery 8/12 and looked like an
                      objective failure.  It was a SEARCH failure: at 4
                      restarts x 150 iterations stage A dropped the true
                      structure from the shortlist once in 12, and at 12 x
                      400 stage B left it under-fitted against a rival three
                      more times -- `+90 -45` only beat `+45 -` at 64 x 1200.
                      Hence the budgets below, the generator being forced
                      into stage B regardless of stage A, and residuals
                      compared RELATIVELY: an absolute 1e-4 tolerance on
                      residuals that run to 1e-12 had called a 3x worse fit
                      "equivalent" and hidden exactly those failures.
                   3a-wide DONE at <= 3 moves, 3705 structures, 6 targets,
                      1h55m: recall 4/6, recovery 5/6, explained 6/6, BIC
                      agrees 5/6.  The objective survives a 20x bigger
                      hypothesis space.  Recall fell from 8/8 because stage A
                      keeps the top 8 of 3705 (0.2%) rather than of 181
                      (4.4%) -- which is what the climb is for, though this
                      run cannot prove the climb recovers it, see below.
    3b the same at <= 3 moves, then the 12 real crops.  Stage
                      A 150 iterations x 8 restarts over all 3705 to
                      shortlist, stage B 1200 x 64 on the top 8.  At the
                      measured 127 ms/fit that is ~7 CPU-hours for 12
                      targets, not the 2.6 an earlier draft of this note
                      claimed: the two-stage filter saves less than it looks
                      once stage B is given a budget that actually converges.
                      Still an overnight job, still no GPU.

                   One thing to watch in 3b, measured in 3a.  The MDL
                   parameter term is the prior's negative log DENSITY, which
                   is a differential entropy and not a code length: at
                   SD_LEN = 0.35 a parameter at its prior mean contributes
                   log(0.35*sqrt(2pi)) = -0.13 nats, so the term REWARDS
                   extra parameters.  Net, an extra move costs +3.2 nats and
                   repays that by improving mse 0.62%; under a BIC term,
                   (k/2) log N, it is +12.6 nats and 2.4%.  On noiseless
                   targets this does not bite -- BIC and the prior both gave
                   7/8 -- because the true structure reaches zero.  Against
                   SVHN's texture floor almost any extra freedom buys 0.62%,
                   so expect the prior term to favour whichever structure has
                   the most slots.  `enumerate.py` reports both rankings.

                   REFIT NOISE, and why it is not the problem it looked
                   like.  On synthetic targets, changing only the random seed
                   moved the score by up to 15x in mse -- 1398 nats, against
                   a structure-term spread of 1.7 and a per-move cost of 3.2.
                   That would make the ranking a lottery.  It does not
                   transfer: the residual term is 512*log(mse), so log
                   amplifies relative differences at mse ~1e-9, and a
                   noiseless target lets the true structure get there.  On a
                   REAL crop, where every structure sits near the texture
                   floor at mse ~0.08, the same measurement gives 0 to 2
                   nats against 78 nats between competing structures.  So the
                   real-crop ranking is not decided by optimiser luck, and
                   the synthetic one is noisy exactly where it does not
                   matter (the true structure wins by orders of magnitude
                   anyway).

                   A GAP this run does not close.  `rank_one` forces the
                   generator into stage B whatever stage A thought, which is
                   what makes the objective testable -- but it also means
                   recovery 5/6 does not say the SEARCH would have found the
                   generator on the 2 crops where recall failed.  To measure
                   that, the climb has to run from the shortlist alone and be
                   checked for whether it reaches the generator.  Two
                   numbers, not one; only the first is measured so far.

                   Also measured in 3a: the residual term dwarfs the other
                   two (sd 1379 nats against 1.2 for structure and 40.3 for
                   parameters over the top candidates), so within one image
                   the ranking is very nearly "best fit" and the structure
                   prior is close to decorative.  That is fine where the true
                   structure can reach zero and is the thing most likely to
                   mislead where it cannot.
                   3b real crops, <= 3 moves (3705 legal).  Killed by
                      accident at 10 of 12 crops, after ~2h25m; the winners
                      below are what it found, refitted from their structures
                      to recover the numbers the lost summary would have
                      printed.  9 of the 10 refits land within 1.3 nats of
                      the killed run.

    crop  lab    mse   blank    R2   IoU  IoUc   L_prior    L_bic   structure
     595   2  0.0402  0.1882 +0.67  0.75  0.80   -1507.4  -1578.3   +90 +180 -270
    3006   1  0.0380  0.2005 +0.63  0.76  0.86   -1577.4  -1617.2   -135 ^ -90
    3158   6  0.0448  0.1876 +0.58  0.71  0.80   -1509.2  -1520.8   -270 +360 -45
    3598   3  0.1135  0.2227 -0.09  0.54  0.55   -1072.4  -1056.2   -360 ^ -180
   10456   2  0.0267  0.1303 +0.71  0.75  0.69   -1753.7  -1786.8   +45 +360 +135
   12423   1  0.0395  0.2023 +0.63  0.79  0.80   -1593.7  -1588.5   - +180 +45
   12673   4  0.0184  0.0842 +0.74  0.72  0.78   -1956.0  -1977.7   +90 +270 +90
   15303   8  0.0370  0.1379 +0.61  0.69  0.87   -1640.5  -1618.5   -360 -180 +270
   15508   0  0.0568  0.1265 +0.36  0.51  0.65   -1262.8  -1410.8   -90 ^ -360
   18275   5  0.0511  0.1332 +0.43  0.58  0.73   -1367.3  -1455.8   -270 +180 -

                      Search beats the hand-written gate on every number,
                      compared LIKE FOR LIKE on these same 10 crops (the
                      gate's headline 0.60 / 0.71 is over 12, and the two
                      missing crops include its worst): IoU median 0.71
                      against 0.60, centred 0.79 against 0.74, R^2 +0.62
                      against +0.41, >= 0.5 on 10/10 and 10/10 against 9/10
                      and 8/10.  Per crop it is 10 out of 10 on mse, not a
                      median effect.  Blank still beats the fit on 0/10.  Crop 3598 is the one failure, R^2 -0.09: a
                      low-contrast "33" where the program drew a circle and
                      an arc over the wrong digit.

                      EVERY winner uses all 3 moves, so the cap is binding
                      and this table is a lower bound -- <= 4 moves would
                      win again.  That is the data term dominating, as 3a
                      measured: a move costs +3.2 nats under the prior and
                      repays it with 0.62% of mse.

                      What was lost with the run and is NOT recoverable from
                      the winners: whether BIC and the prior term rank the
                      same structure first.  The two differ by a mean 37
                      nats on the winners alone -- the same order as the 78
                      nats between competing structures -- so they can
                      disagree, and the question stands.  Re-measuring it
                      costs ~15 min per crop.

                      Refit noise, re-measured here on 10 crops rather than
                      1: nine agree to within 1.3 nats, and crop 18275 came
                      back 13% BETTER (72 nats) from a different seed.  So
                      the earlier "0 to 2 nats" was optimistic; the drift is
                      usually negligible but has a tail big enough to flip a
                      ranking, and it is an UNDER-fit, not noise in both
                      directions.

    4  data.py     prefix batches + IoU-to-go targets, from the sampler.
                   DONE.  One example per PREFIX, so a 5-move program is 6
                   decisions; 64 programs give ~400 examples and 32 MiB.
                   Each example carries the 5-channel input, the pen tip for
                   the bilinear readout, the generator's own (kind, angle,
                   length-bucket), the legality masks (`legal` itself,
                   evaluated incrementally, not a second copy of its rules)
                   and the value.

                   The decision that makes it work: a prefix is rendered with
                   `alpha(..., sel=)`, so the ink is masked AFTER `place` and
                   each partial drawing sits where it will sit when finished.
                   Rendering the prefix on its own re-centres it on its own
                   centroid and teaches the policy that the drawing slides
                   around as it is drawn.  The batch asserts it: a
                   full-length prefix equals the finished render to 1e-5.

                   Targets come three ways, because if every target were the
                   prefix's own the value would be 1.0 everywhere and the
                   head would learn a constant.  Measured over 400 examples:

                       matched     value 1.000            (own program)
                       divergent   value 0.45 +- 0.24     (same prefix,
                                   prior-sampled completion, same pose)
                       unrelated   value 0.27 +- 0.11     (another target)

                   5-8% of DIVERGENT targets still score >= 0.9 -- two
                   different completions of the same prefix that draw nearly
                   the same picture.  That is a ceiling on what any value
                   head can score, Markov or not, and it is the number risk 2
                   has to be argued against rather than the raw gap.

                   This module ASSUMES THE STEP-8 SCENE HEAD.  Training
                   canvases are placed by the generator's own pose; at
                   inference nothing knows the pose until a one-shot head
                   regresses it.  Until then SMC has to be handed one.
    5  net.py      the trunk and heads, hand-rolled Adam.  DONE: 40,458
                   parameters, no framework, one jit, 3.1 ms per step at
                   batch 512 on the 3090 Ti.  Masked actions carry exactly
                   0.0 probability, not merely a small one, and the model
                   overfits a single batch to 1.00 on all three factors --
                   the precondition for anything measured after it.

                   The device changes the project's economics more than it
                   changes this file: the FITTER went 130 ms to 20 ms per
                   fit (gate wall 77 s -> 11 s, same numbers to the
                   decimal), so the 3705-structure enumeration that took
                   2 h 25 m is now about 22 minutes.

                   A bigger bottleneck than the device was in `data.batch`:
                   programs have different lengths, so every batch had a
                   different example count, which is a new XLA shape and a
                   fresh compile of the renderer on EVERY call -- 23 s a
                   batch against 3 ms of training.  Fixing the example count
                   and the per-mode counts took it to 0.38 s warm, 1332
                   examples/s.  Same trap `fit.run` balances chunks for.
    6  train.py    synthetic only; held-out action accuracy and value MSE.
                   The VALUE HEAD FIRST and alone, because if a Markov value
                   cannot separate good prefixes from bad, a Markov policy
                   will not propose coherent continuations either -- risk 2
                   settled for the price of one head.

                   6a value alone, 4000 steps, 166 s.  RISK 2 DOES NOT BITE.
                      On 2048 held-out examples: mse 0.0082, correlation
                      +0.966, mean predicted value 0.988 matched / 0.452
                      divergent / 0.304 unrelated.

                          ranks matched above divergent   0.933
                          the same statistic on TRUE IoU  0.963   (97% of it)
                          ranks matched above unrelated   0.998  (ceiling 1.000)

                      The ceiling is the point.  It is below 1.0 because two
                      completions of one prefix sometimes draw nearly the
                      same picture, so an AUC reported against 1.0 would
                      charge the model for an ambiguity in the DATA.  The
                      Markov state carries 97% of what is there to carry.

                   6b all four heads, 4000 steps.  Value mse 0.0212, r
                      +0.910, AUC 0.903 against the same 0.963 ceiling --
                      the policy costs the value about 3 points of shared
                      capacity.  Action accuracy, against the control that
                      matters:

                                    net    prior alone    gain
                          kind     0.860      0.513      +0.347
                          angle    0.428      0.169      +0.259
                          length   0.349      0.195      +0.154

                      The control is the most frequent action among the
                      LEGAL ones, with no image at all.  It already knows
                      the grammar and the prior's skew, so beating chance
                      means nothing and beating this means something.  It is
                      also what found the two bugs below, both of which left
                      a head looking merely weak.

                      BUG 1, the policy was trained on noise.  60% of
                      examples carry a divergent or unrelated target, which
                      the VALUE head needs -- with matched targets alone
                      every label is 1.0 -- but the POLICY must never see:
                      the action label is the move the generator made, and
                      against a target that generator did not draw it is not
                      the right answer to the question being asked.  Masking
                      the policy loss to matched targets: kind 0.767 ->
                      0.84, angle 0.315 -> 0.42.

                      BUG 2, the length label was not a function of anything
                      the network could see.  `log len` is in GLYPH units and
                      `scale_to_page` gives each drawing its own scale so it
                      fills the page -- 20x spread, sd 0.64 nats, against
                      bins 1.27x wide, so the scale alone smeared one picture
                      over ~13 bins of an 8-bin head.  Length scored BELOW
                      the control (0.226 against 0.268): not a weak head, an
                      unlearnable target.  Bucketing PAGE PIXELS instead (6
                      to 66 px for a draw's span, 1 to 20 px for a hop, the
                      1st to 99th percentile of what the prior draws) and
                      multiplying the scale back in inside `q_of` -- which
                      the proposer may do, because the scene head regresses
                      the pose before the rollout starts.  0.226 -> 0.349
                      against a 0.195 control.
    7  smc.py      particles + resampling, measured against step 3
    8  scene.py    the colour/blur/background layer and its one-shot head (new)
    9  SVHN, and the domain gap measured rather than assumed
    10 abstraction: mine fragments from the fitted corpus, add them to the
       grammar, retrain the head -- the library changes the PRIOR, which the
       proposer cannot do
