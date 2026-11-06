"""A pen that draws digits: a BPL-style DSL, its fitter, and its gate.

The factorisation is Lake et al.'s: ONE pen with a position and a heading,
and every move stated relative to it.  Strokes chain by construction -- the
next one starts where the last ended, or the pen lifts, hops a short way and
starts there -- so there are no per-stroke positions to fit, only one glyph
pose.  That is what makes two instances of a digit comparable: the program,
not the coordinates, carries the shape.

    dsl.py      the alphabet, the geometry, the renderer, the prior
    fit.py      batched Adam over the continuous parameters only
    check.py    the invariants the design rests on       python -m pen.check
    sample.py   draws from the prior and looks at them   python -m pen.sample [seed]
    gate.py     can a hand-written program fit SVHN?     python -m pen.gate [ink]
    README.md   the proposer design note (write/execute/assess)

Run the scripts as modules from the repo root, which is where their imports of
`imageio` and `svhn_data` resolve.  Nothing is imported here: `fit`
pulls in jax and the SVHN loader, and `dsl` alone is cheap.
"""
