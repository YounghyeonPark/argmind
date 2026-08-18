# SPDX-License-Identifier: Apache-2.0
"""Can a stored descriptor recognise the same object again? Yes — and the crop, not learning, is why.

**The proposal** (project owner, 2026-08-19): raise the level of perception with a long-term memory
that holds each label's characteristic features, so that when the same features appear again the
same label is assigned. It is well aimed: `demo/test_wean.py` measured that self-training on the
head's own output degrades it 0.938 → 0.804, and a descriptor matched against a *stored exemplar* is
an anchor rather than a moving target — a training signal that is not the head's own output, which
is what the 2026-08-14 decision requires and what nothing here provided.

**What was already recorded, and what was missing.** `AppearanceMatcher` re-identifies
differently-coloured objects on an untrained backbone (within-object 0.048 against between-object
0.232). But for objects that share a colour and differ in *shape*, `tests/test_learned_features.py`
measured that training the hidden layers on a **position** objective leaves single-unit selectivity
where random initialisation put it — 0.511 against 0.532 — and recorded the next step as "an
objective that *requires* identity".

**That was tried on 2026-08-19 and it does not work either.** Paired, same initialisation before and
after, four seeds, at a learning rate where **both** objectives train (position residual 0.010–0.042;
identity 0.272–0.397 against a chance 0.8165):

| objective | paired change in mean d-prime | every seed positive |
|---|---|---|
| position | +0.031 | no (worst −0.025) |
| identity | **−0.061** | no (worst −0.170) |

**And then the interesting part.** Single-unit selectivity is the wrong instrument: the question is
whether the *population* carries shape. Decoded with a linear head, held out by position:

| descriptor | frozen random backbone | features learned too |
|---|---|---|
| whole frame, hold-out sharing positions | 0.822–1.000 | 0.989–1.000 |
| whole frame, **positions disjoint** | **0.367** | 0.342 |
| **24×24 crop centred on the object, positions disjoint** | **0.996** | — |

chance is 0.333. **The first row is position memorisation and was nearly recorded as a finding.**
Its hold-out shared phases with training — the orbit is periodic mod 61, so it differed only in
noise, and a linear head with 109 coefficients over 600 samples can memorise that. Splitting the
positions properly takes it straight to chance.

**So the whole-frame embedding does not carry position-invariant shape at all, and no objective
makes it.** Grid pooling entangles what with where by construction, which is also the honest reading
of the d-prime results: units are not object-selective because the representation is not
object-factored.

**The crop removes position by construction, and then a completely untrained backbone is enough** —
0.996 across positions it never saw. That is why the recorded colour result worked: it was measured
on crops.

**The consequence for the proposal.** Recognition memory does not need representation learning. It
needs *where to look*. The missing faculty is **attention** — one of the eight this repository
records as absent in `tests/test_what_this_is_not.py` — and not a better descriptor.
"""

from __future__ import annotations

import numpy as np
import pytest

from argmind.adapt import RecursiveLeastSquares
from argmind.backbone import Backbone
from tests.test_learned_features import H, POOL, SAME, WALL

SHAPES = ("disc", "square", "bar")
PLAN = [("conv", 8, 5, 2), ("dw", 3, 2), ("pw", 12)]
W = 48
CROP = 24


def render(shape_i, cx, cy, seed, r=0.13):
    yy, xx = np.mgrid[0:H, 0:W]
    img = np.full((H, W, 3), WALL)
    if SHAPES[shape_i] == "disc":
        m = np.hypot(xx - cx * W, yy - cy * H) <= r * W
    elif SHAPES[shape_i] == "square":
        m = (np.abs(xx - cx * W) <= r * W * 0.8) & (np.abs(yy - cy * H) <= r * W * 0.8)
    else:
        m = (np.abs(xx - cx * W) <= r * W * 1.6) & (np.abs(yy - cy * H) <= r * W * 0.35)
    img[m] = SAME
    g = np.random.default_rng(seed)
    return np.clip(img + g.standard_normal(img.shape) * 0.01, 0, 1).astype("f8")


def place(rng, *, test):
    """Training positions in the left band, test positions in the right band — **disjoint**, which
    is the whole point. A hold-out that shares positions measures memorisation, and this file's
    docstring records that being nearly published as a finding."""
    return (rng.uniform(0.62, 0.74) if test else rng.uniform(0.26, 0.38)), rng.uniform(0.34, 0.66)


def crop_at(img, cx, cy):
    out = np.full((CROP, CROP, 3), WALL)
    y0, x0 = int(cy * H) - CROP // 2, int(cx * W) - CROP // 2
    ys, xs = slice(max(y0, 0), min(y0 + CROP, H)), slice(max(x0, 0), min(x0 + CROP, W))
    out[ys.start - y0:ys.stop - y0, xs.start - x0:xs.stop - x0] = img[ys, xs]
    return out.astype("f8")


def decode(*, cropped, seed, steps=600, n_test=120):
    """Held-out shape-decoding accuracy from a **frozen, untrained** backbone."""
    shape = (CROP, CROP, 3) if cropped else (H, W, 3)
    bb = Backbone(PLAN, shape, dtype="f8", seed=seed, pool=POOL)
    head = RecursiveLeastSquares(bb.n_outputs + 1, 3, forgetting=0.999, p0=100.0, dtype="f8")

    def embed(i, cx, cy, s):
        img = render(i, cx, cy, seed=s)
        return np.concatenate([np.asarray(bb.features(crop_at(img, cx, cy) if cropped else img)),
                               [1.0]])

    rng = np.random.default_rng(5)
    for t in range(steps):
        cx, cy = place(rng, test=False)
        head.update(embed(t % 3, cx, cy, t), np.eye(3)[t % 3])
    tr, hits = np.random.default_rng(77), 0
    for t in range(n_test):
        cx, cy = place(tr, test=True)
        hits += int(int(np.argmax(np.asarray(head.predict(embed(t % 3, cx, cy, 8000 + t))))) == t % 3)
    return hits / n_test


@pytest.mark.parametrize("seed", (7, 23, 41))
def test_a_whole_frame_embedding_carries_NO_position_invariant_shape(seed):
    """The negative half. Trained at one set of positions and tested at another, a whole-frame
    descriptor decodes shape at chance — so "the same features appear again" is not something it can
    express once the object has moved."""
    acc = decode(cropped=False, seed=seed)
    assert acc < 0.55, (
        f"whole-frame decoding reached {acc:.3f} across disjoint positions; if that became true the "
        "representation has gained position-invariant shape and this file's conclusion changes")


@pytest.mark.parametrize("seed", (7, 23, 41))
def test_a_CROP_makes_it_recognisable_on_an_UNTRAINED_backbone(seed):
    """The positive half, and the answer to the proposal: the crop is load-bearing, not the
    learning. No gradient has touched this backbone."""
    acc = decode(cropped=True, seed=seed)
    assert acc > 0.9, (
        f"cropped decoding is only {acc:.3f}; the recognition memory rests on this and it has "
        "stopped working")


def test_the_contrast_is_the_finding_and_the_two_are_compared_on_the_same_seed():
    """Stated as one assertion because either number alone invites the wrong reading: the frame
    result is not "the backbone is bad" and the crop result is not "recognition is solved"."""
    for seed in (7, 23):
        frame, cropped = decode(cropped=False, seed=seed), decode(cropped=True, seed=seed)
        assert cropped - frame > 0.4, (
            f"seed {seed}: crop {cropped:.3f} against frame {frame:.3f} — the gap that says "
            "*where to look* is the missing faculty has closed, and the conclusion should move")
