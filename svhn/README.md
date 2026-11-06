# SVHN (cropped digits)

Street View House Numbers, format 2 — 32x32 colour crops centred on one digit.
From [Netzer et al., NIPS 2011](http://ufldl.stanford.edu/housenumbers/), here as a
far-OOD companion to the RMNIST sets: real photographs rather than corruptions of
the in-distribution data, so no renderer parameter covers any of it.

## Provenance

The official host, `ufldl.stanford.edu`, refused connections on both :80 and :443 when
this was fetched (2026-11-05), so the `.mat` files came from the `MarMaster/svhn`
mirror on Hugging Face. They are byte-identical to the originals — MD5s match the
values torchvision pins:

| file | MD5 | matches upstream |
|---|---|---|
| `data/train_32x32.mat` | `e26dedcc434d2e4c54c9b2d4a06d8373` | yes |
| `data/test_32x32.mat` | `eb5a983be6a315427106f1b164d9cef3` | yes |

The `extra` split (531,131 images, ~1.3 GB) is not here; that mirror does not carry it.
It is on `ufldl-stanford/svhn` as parquet, which would need pyarrow in `.venv`.

## Files

| file | what it is |
|---|---|
| `data/{train,test}_32x32.mat` | upstream, untouched (182 MB / 64 MB) |
| `data/svhn.npz` | the same data as plain arrays (384 MB) |
| `data/svhn_preview.png` | one row per digit 0-9, native RGB left, `gray28` right |
| `data/svhn_8x8.png` | 64 random test crops, same layout (`render.py data test 8 7`) |
| `make_svhn.py` | the converter |
| `render.py` | tiles a random sample, RGB beside its `gray28` twin |

## Format

`.mat` holds `(32, 32, 3, N)` — channels third, batch last — with labels 1..10, and
needs `scipy.io.loadmat`. `svhn.npz` is plain arrays, so `jnp.asarray` takes any of
them directly:

| key | shape | dtype | |
|---|---|---|---|
| `{train,test}_x` | `(N, 32, 32, 3)` | uint8 | NHWC, the Flax conv layout |
| `{train,test}_y` | `(N,)` | int32 | 0..9 |
| `{train,test}_gray28` | `(N, 784)` | uint8 | 28x28 grayscale, flattened |
| `{train,test}_border_minus_centre` | `(N,)` | float32 | polarity estimate, see below |

N is 73,257 train and 26,032 test. Arrays stay uint8 because float32 at native size
would be 900 MB for the training split; divide by 255 at use, which fuses on device.
The file is uncompressed on purpose — photographs barely deflate, and `np.load` is
lazy per array, so this costs ~8% disk to load several times faster.

Labels: upstream uses **10 for the digit 0**, so the remap is `y % 10`. The converted
per-digit counts are `[4948, 13861, 10585, 8497, 7458, 6882, 5727, 5595, 5045, 4659]`
for train, i.e. the class that became 0 is exactly the old class 10.

`gray28` matches this repo's own convention — `ood.py` and the RMNIST sets are all
`(n, 784)` at 28x28 — so `ood.corrupt(svhn_gray/255.0, name)` works as-is. Luma is
ITU-R BT.601 and the 32->28 resize is Pillow's antialiased BILINEAR, which averages
rather than drops rows, as MNIST's own downsampling did.

## Two things that are not corrected

Both are properties of the data, and fixing either is a modelling decision, so
`make_svhn.py` leaves them alone.

* **Polarity is mixed.** RMNIST is 0.0 background / 1.0 ink by construction. House
  numbers are photographed both ways: by the border-minus-centre heuristic, **53.2%**
  of train and **51.6%** of test are dark digits on a light ground, i.e. inverted
  relative to what the fitter expects. `border_minus_centre` is
  `mean(border ring) - mean(centre 14x14)` in [0, 1] units, positive for a dark digit,
  so selecting or flipping on `> 0` is a one-liner if you want a polarity-matched set.

* **Distractor digits.** Format 2 crops are centred on one labelled digit but keep
  whatever neighbours fall in the 32x32 window — visible in the preview as partial
  digits at the left and right edges. A single-digit fitter sees them as ink.

## Regenerating

    .venv/bin/python -I make_svhn.py data

Verified after conversion: `{train,test}_x` is pixel-identical to the `.mat` arrays
under the transpose, every old label 10 maps to 0 and nothing else moves, and
`ood.corrupt` accepts `gray28 / 255.0` on all 14 corruptions.
