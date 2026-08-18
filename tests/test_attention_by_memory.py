# SPDX-License-Identifier: Apache-2.0
"""Can a stored exemplar say WHERE TO LOOK? Measured — yes, on an untrained backbone.

`tests/test_recognition_memory.py` established that a crop makes shape recognisable and a whole
frame does not, so recognition memory needs no representation learning — it needs *where to look*.
This measures whether the memory itself can supply that: score candidate crop centres by similarity
to the nearest stored exemplar, and take the best.

**Every number here is 1.000, which is exactly why the controls matter.** A tidy result in this
repository is evidence the measurement is broken until three confounds are ruled out, and the first
version of this experiment fell to two of them:

| confound | how it was ruled out |
|---|---|
| **saliency** — one object on a plain wall makes "not wall" the whole answer | two objects in every frame, both non-wall |
| **memory content irrelevant** — the score may ignore what is stored | swapping which colour is remembered **flips** the answer, 1.000 / 0.000 both ways |
| **position** — exemplars came from the left band and the target sat there too | the remembered object alternates sides; still 1.000 |

Measured, untrained backbone, seed 23, 40 trials each:

| memory | distractor | lands on the remembered object |
|---|---|---|
| red disc | blue disc | 1.000 (other: 0.000) |
| blue disc | red disc | 1.000 (other: 0.000) |
| disc | bar, **same colour** | 1.000 |
| bar | disc, **same colour** | 1.000 |
| square | bar, **same colour** | 1.000 |

**What this is not.** It is a brute-force scan of ~117 candidate crops per frame, so it is *search*,
not attention — a cheap proposal mechanism is not built. And the exemplars were seeded from
ground-truth-centred crops, so the bootstrap question — where the first exemplar comes from without
the classical segmenter — is untouched. What is established is only that the matching **signal**
exists and discriminates, which is the precondition for learning attention from it.

**Two mutants survive and are recorded rather than contrived around.** Fixing the sides instead of
alternating them still scores 1.000, so the alternation is not what carries the parametrised tests —
the position confound is covered by its own test below, which places the remembered object on the
right only. And building the exemplars at the *test* positions also survives, which makes sense: it
strengthens the position cue rather than removing the shape one, so a shape-driven result is
unaffected. A third mutant, dropping the distractor, **was** a real gap and is closed by an assertion
that both objects were actually rendered.
"""

from __future__ import annotations

import numpy as np
import pytest

from argmind.backbone import Backbone
from tests.test_recognition_memory import (CROP, H, PLAN, POOL, SAME, W, WALL, crop_at, place)

STEP = 3
RED, BLUE = (0.85, 0.25, 0.20), (0.20, 0.35, 0.85)


def _blob(img, xx, yy, shape, cx, cy, colour, r=0.13):
    if shape == "disc":
        m = np.hypot(xx - cx * W, yy - cy * H) <= r * W
    elif shape == "square":
        m = (np.abs(xx - cx * W) <= r * W * 0.8) & (np.abs(yy - cy * H) <= r * W * 0.8)
    else:
        m = (np.abs(xx - cx * W) <= r * W * 1.6) & (np.abs(yy - cy * H) <= r * W * 0.35)
    img[m] = colour


def render(a, b, seed):
    """`a` and `b` are `(shape, colour, cx, cy)` or None."""
    yy, xx = np.mgrid[0:H, 0:W]
    img = np.full((H, W, 3), WALL)
    for spec in (a, b):
        if spec is not None:
            shape, colour, cx, cy = spec
            _blob(img, xx, yy, shape, cx, cy, colour)
    g = np.random.default_rng(seed)
    return np.clip(img + g.standard_normal(img.shape) * 0.01, 0, 1).astype("f8")


def embed(bb, img, cx, cy):
    v = np.asarray(bb.features(crop_at(img, cx, cy)), dtype=np.float64)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def memory_of(bb, shape, colour, n=40, seed=5):
    """Exemplars from crops centred on the object, at LEFT-band positions only."""
    rng, mem = np.random.default_rng(seed), []
    for t in range(n):
        cx, cy = place(rng, test=False)
        mem.append(embed(bb, render((shape, colour, cx, cy), None, seed=t), cx, cy))
    return np.asarray(mem)


def look(bb, mem, img):
    """The candidate centre whose crop best matches something remembered."""
    best, xy = -2.0, None
    for py in range(CROP // 2, H - CROP // 2 + 1, STEP):
        for px in range(CROP // 2, W - CROP // 2 + 1, STEP):
            cx, cy = px / W, py / H
            s = float(np.max(mem @ embed(bb, img, cx, cy)))
            if s > best:
                best, xy = s, (cx, cy)
    return xy


def trial(bb, mem, remembered, other, *, n=16):
    """Fraction of frames where the chosen centre is nearer the remembered object.

    **The sides alternate**, because the exemplars come from the left band and a version of this
    test with the target always on the left scored 1.000 by matching *position*.
    """
    hits = 0
    for t in range(n):
        swap = (t % 2 == 0)
        rx, ry = (0.70 if swap else 0.30), (0.60 if swap else 0.35)
        ox, oy = (0.30 if swap else 0.70), (0.35 if swap else 0.60)
        img = render((remembered[0], remembered[1], rx, ry),
                     (other[0], other[1], ox, oy), seed=9000 + t)
        # **Both objects must really be there.** A mutant that dropped the distractor survived: with
        # one object, "nearer the remembered one" is vacuously true and the saliency confound this
        # fixture exists to rule out comes straight back.
        for (px, py) in ((rx, ry), (ox, oy)):
            patch = img[int(py * H) - 2:int(py * H) + 3, int(px * W) - 2:int(px * W) + 3]
            assert float(np.abs(patch - np.asarray(WALL)).max()) > 0.1, (
                f"nothing was rendered at ({px}, {py}); with one object this trial cannot "
                "distinguish recognition from 'find the patch that is not wall'")
        cx, cy = look(bb, mem, img)
        hits += int(np.hypot(cx - rx, cy - ry) < np.hypot(cx - ox, cy - oy))
    return hits / n


@pytest.fixture(scope="module")
def bb():
    return Backbone(PLAN, (CROP, CROP, 3), dtype="f8", seed=23, pool=POOL)


def test_memory_content_decides_where_it_looks_and_swapping_it_FLIPS_the_answer(bb):
    """The control that rules out saliency: both objects are non-wall, and only what is remembered
    differs between the two runs."""
    red = trial(bb, memory_of(bb, "disc", RED), ("disc", RED), ("disc", BLUE))
    blue = trial(bb, memory_of(bb, "disc", BLUE), ("disc", BLUE), ("disc", RED))
    assert red > 0.8 and blue > 0.8, (
        f"remembering red lands on red {red:.3f} and remembering blue lands on blue {blue:.3f}; "
        "if either fell to chance the score is not reading the memory")


@pytest.mark.parametrize("remembered,other", [("disc", "bar"), ("bar", "disc"), ("square", "bar")])
def test_it_works_on_SHAPE_at_the_same_colour_which_is_where_everything_else_here_failed(
        bb, remembered, other):
    """Same colour, so colour cannot answer it — the condition under which this repository measured
    single-unit selectivity at random-init level under both a position and an identity objective."""
    got = trial(bb, memory_of(bb, remembered, SAME), (remembered, SAME), (other, SAME))
    assert got > 0.8, f"remembering a {remembered} against a {other} landed on it only {got:.3f}"


def test_the_sides_alternate_so_it_cannot_be_matching_POSITION(bb):
    """Stated as its own test because it is the confound that broke the first version, and the same
    one that turned an earlier 'the descriptor already exists' result into position memorisation.

    The exemplars are built only in the left band. If the score were matching where rather than
    what, putting the remembered object on the right would send attention to the left."""
    mem = memory_of(bb, "disc", SAME)
    right_only = 0
    for t in range(8):
        img = render(("disc", SAME, 0.70, 0.60), ("bar", SAME, 0.30, 0.35), seed=7000 + t)
        cx, _cy = look(bb, mem, img)
        right_only += int(cx > 0.5)
    assert right_only >= 7, (
        f"with the remembered object always on the RIGHT, attention went there {right_only}/8 — "
        "the exemplars are all from the left band, so anything less says it is matching position")
