# digit-synthesis

Fitting **programs**, not parameters, to images of digits.

A digit is drawn by a pen: it goes down somewhere, moves in some direction
along a straight line or an arc, turns, moves again, maybe lifts and hops a
short way, and comes up. That is a short program, and the factorisation is
Lake, Salakhutdinov and Tenenbaum's Bayesian Program Learning model of
Omniglot. Here the target is SVHN -- photographs of house numbers, where no
renderer parameter covers the variation and the digit is one glyph in a
string, surrounded by its neighbours.

The split that makes it work: the **structure** (which moves, in which order,
at which discrete angles) is searched, and the **numbers** (lengths,
deviations, glyph pose, pen width) are fitted by gradient descent. Every
discrete choice rides as data rather than as code, so one compiled kernel
covers every program structure and a parameter the structure does not use
takes exactly 0.0 gradient. A batch may mix a one-move program with a
twelve-move one.

```
pen/            the DSL, the fitter, the invariants, the gate   (see pen/README.md)
svhn_data.py    SVHN crops and the ink estimate programs are fitted to
imageio.py      render output -> PNG
svhn/           fetch and pack the dataset
```

```
python -m pen.check            28 invariants
python -m pen.sample [seed]    draw from the prior and look at it
python -m pen.gate [ink]       the measurement
```

## Where it stands

The gate passes. Twelve polarity-matched SVHN crops, one or two hand-written
programs per digit plus their mirrors, numbers fitted, no search:

    blank beats the fit    0 / 12
    IoU median             0.60 full frame, 0.71 over the centre 20 px
    IoU >= 0.5             10 / 12 and 9 / 12
    R2 median              +0.41
    cost                   149 ms/fit, CPU

Two numbers worth reading twice. **Blank beats the fit on none of them** --
flat MSE's own optimum on a faint target is the empty image, and that
collapse was the predicted first failure; it did not happen. And **45% of the
target ink is a neighbouring digit**, which a single-glyph program cannot
draw and no fit can remove, so the full-frame IoU measures the crop as much
as the model. Both numbers are reported for that reason.

Next is the proposer: a network that proposes structure from the image, which
gradient descent then refines. `pen/README.md` is its design note, including
what it takes from *Write, Execute, Assess* (arXiv:1906.04604) and what it
drops.

## History

This grew out of a set of studies on a differentiable spline renderer and its
inversions, fitted to rotated MNIST. Those made the case that a renderer with
29 free parameters per stroke has no shape identity left after fitting -- a
"line" can end up an S -- which is what motivated a DSL whose primitives
cannot deform. They are kept on disk under `mnist/`, out of this repo.
