# SPDX-License-Identifier: Apache-2.0
"""Where does the FIRST exemplar come from? Measured — from rarity, with no labels at all.

`tests/test_attention_by_memory.py` showed a stored exemplar can say where to look, but seeded that
exemplar from a ground-truth-centred crop — which is the supervision the 2026-08-14 decision wants
removed. This closes that gap. The signal is **rarity**: score every candidate crop by its mean
similarity to all the others and take the minimum, the crop least like the rest of the picture. No
labels, no classical segmenter, no trained weights.

| background | rarest crop lands on an object | chance |
|---|---|---|
| plain wall | 1.000 | 0.101 |
| sinusoidal stripes | 1.000 | 0.101 |
| 40 small clutter blobs | 1.000 | 0.101 |

**A failure was observed and its explanation withdrawn while writing the test for it.** An
exploratory run scored 0.000 against a 0.444 chance rate on a 36×48 frame, and the explanation
offered was scale: a 24×24 crop covers two thirds of that height, so no candidate holds one object
surrounded by background. The geometric half is true and is asserted. The **0.000 is not
reproducible** — this fixture, differing in object radius and in using unpadded crops, reaches 1.000
on the same frame size. So the failure was real in that run and its cause was never isolated, and
asserting the scale limit would be asserting a mechanism that cannot be reproduced.

**What is still not answered.** This is a full pairwise scan — 247 candidates, so ~61,000
similarities per frame — which is search, not attention. And the clutter tested is uniform in
appearance, so it becomes background statistically; a distractor that looks like a *different
object* is untested.
"""

from __future__ import annotations

import numpy as np
import pytest

from argmind.backbone import Backbone
from tests.test_recognition_memory import PLAN, POOL, SAME, WALL

H, W, CROP, STEP = 72, 96, 24, 4
RADIUS = 0.075
SMALL_H, SMALL_W = 36, 48          #: the frame that is too small for a crop to isolate anything


def render(specs, seed, *, texture="none", h=H, w=W):
    yy, xx = np.mgrid[0:h, 0:w]
    g = np.random.default_rng(7000 + seed)
    img = np.full((h, w, 3), WALL)
    if texture == "stripes":
        img = img + 0.10 * np.sin(xx / 3.0)[..., None]
    elif texture == "clutter":
        for _ in range(40):
            cx, cy = g.uniform(0.05, 0.95), g.uniform(0.05, 0.95)
            img[np.hypot(xx - cx * w, yy - cy * h) <= 0.022 * w] = (0.55, 0.45, 0.40)
    for shape, cx, cy in specs:
        if shape == "disc":
            m = np.hypot(xx - cx * w, yy - cy * h) <= RADIUS * w
        else:
            m = (np.abs(xx - cx * w) <= RADIUS * w * 1.6) & (np.abs(yy - cy * h) <= RADIUS * w * 0.35)
        img[m] = SAME
    return np.clip(img + g.standard_normal(img.shape) * 0.01, 0, 1).astype("f8")


def crop_at(img, cx, cy, h, w):
    y0, x0 = int(cy * h) - CROP // 2, int(cx * w) - CROP // 2
    return img[y0:y0 + CROP, x0:x0 + CROP].astype("f8")


def interior(h, w, step=STEP):
    """Only crops lying **entirely** inside the frame. A padded crop straddling the border has an
    unusual composition and is genuinely the rarest thing in the picture — an artefact of cropping,
    not a fact about objects, and it won 8 of 8 frames before this restriction.

    `step` is a parameter so a caller can trade candidate density for time **without changing this
    file's own figures**, which are published at the default and rest on it. `tests/test_proposal.py`
    uses a coarser grid; its numbers were re-measured there rather than inherited from here.
    """
    return [(px / w, py / h)
            for py in range(CROP // 2, h - CROP // 2 + 1, step)
            for px in range(CROP // 2, w - CROP // 2 + 1, step)]


def rarest(bb, img, cs, h, w):
    E = []
    for cx, cy in cs:
        v = np.asarray(bb.features(crop_at(img, cx, cy, h, w)), dtype=np.float64)
        n = float(np.linalg.norm(v))
        E.append(v / n if n > 1e-12 else v)
    E = np.asarray(E)
    S = E @ E.T
    return cs[int(np.argmin((S.sum(axis=1) - np.diag(S)) / (len(cs) - 1)))]


@pytest.fixture(scope="module")
def bb():
    return Backbone(PLAN, (CROP, CROP, 3), dtype="f8", seed=23, pool=POOL)


def _score(bb, texture, h=H, w=W, n=12):
    cs = interior(h, w)
    hits, chance = 0, []
    for t in range(n):
        specs = [("disc", 0.30, 0.40 + 0.22 * (t % 2)), ("bar", 0.70, 0.40 + 0.22 * ((t + 1) % 2))]
        img = render(specs, seed=t, texture=texture, h=h, w=w)
        cx, cy = rarest(bb, img, cs, h, w)
        hits += int(any(np.hypot(cx - p[1], cy - p[2]) < 0.10 for p in specs))
        chance.append(np.mean([any(np.hypot(a - p[1], b - p[2]) < 0.10 for p in specs) for a, b in cs]))
    return hits / n, float(np.mean(chance))


@pytest.mark.parametrize("texture", ("none", "stripes", "clutter"))
def test_the_first_exemplar_can_be_acquired_with_NO_labels(bb, texture):
    """The bootstrap. Nothing here knows where the objects are, and the backbone is untrained."""
    got, chance = _score(bb, texture)
    assert got > 0.8, f"rarity found an object only {got:.3f} of the time on a {texture} background"
    assert chance < 0.3, (
        f"the chance rate is {chance:.3f}; with objects covering that much of the frame, landing on "
        "one says little and this fixture has stopped discriminating")


def test_the_frame_size_changes_what_a_crop_can_isolate_and_the_rest_is_NOT_reproduced(bb):
    """**A claim withdrawn in the writing of this test.**

    An exploratory run scored rarity at 0.000 against a 0.444 chance rate on a 36x48 frame, and it
    was about to be recorded as a scale limit: a 24x24 crop covers two thirds of that height, so no
    candidate holds one object surrounded by background. What *is* reproducible is that geometric
    fact, asserted below. What is **not** reproducible is the 0.000 — this fixture, differing in
    object radius and in using unpadded crops, reaches 1.000 on the same frame size.

    So the failure was real in that run and its cause was never isolated. Asserting a scale limit
    here would be asserting a mechanism I cannot reproduce, which is the error this repository keeps
    finding in its own records. The geometry is pinned; the conclusion is not drawn.
    """
    for h, w, name in ((SMALL_H, SMALL_W, "small"), (H, W, "large")):
        cs = interior(h, w)
        img = render([("disc", 0.30, 0.40), ("bar", 0.70, 0.62)], seed=1, h=h, w=w)
        pure = np.mean([float(np.abs(crop_at(img, a, b, h, w) - np.asarray(WALL)).max()) < 0.15
                        for a, b in cs])
        if name == "small":
            assert pure < 0.25, (
                f"the small frame now offers {pure:.3f} pure-background crops; the geometric claim "
                "is that it offers far fewer than the large one")
            small_pure = pure
        else:
            assert pure > small_pure, (
                f"the large frame ({pure:.3f}) no longer offers more isolating crops than the small "
                f"one ({small_pure:.3f}), which is the only part of the scale story that survived")


def test_the_rarity_RANKING_is_the_sound_measure_and_the_argmin_is_the_fragile_one(bb):
    """**Added after `tests/test_sparse.py` showed the argmin metric misbehaving.**

    Everything above scores the bootstrap as "is the single rarest crop within 0.10 of an object
    centre". Sparsifying the descriptor made that read 0.167 / 1.000 / 0.000 / 0.750 / 1.000 across
    k = 4, 8, 16, 32, 108 — non-monotonic and reproducible at three times the sample. The ranking
    underneath is sound at every one of those k: on-object crops rank 43.2 → 21.2 while off-object
    ones sit near 134, improving monotonically. The argmin lands on crops covering only a fifth of
    their area with object, which the centre tolerance scores as a miss.

    So this asserts the ranking as well. The argmin tests are **kept**, not replaced: the figures
    published for them rest on that form, and swapping the measure silently would leave those
    numbers describing something no longer computed.
    """
    cs = interior(H, W)
    specs = [("disc", 0.30, 0.40), ("bar", 0.70, 0.62)]
    img = render(specs, seed=1, texture="none")
    on = np.asarray([any(np.hypot(cx - p[1], cy - p[2]) < 0.10 for p in specs) for cx, cy in cs])
    assert 0.05 < on.mean() < 0.25, (
        f"{on.mean():.3f} of candidates are on an object; the rank comparison below needs them to "
        "be a clear minority or the separation is not meaningful")

    E = []
    for cx, cy in cs:
        v = np.asarray(bb.features(crop_at(img, cx, cy, H, W)), dtype=np.float64)
        n = float(np.linalg.norm(v))
        E.append(v / n if n > 1e-12 else v)
    E = np.asarray(E)
    S = E @ E.T
    rare = -(S.sum(axis=1) - np.diag(S)) / (len(cs) - 1)
    order = np.argsort(-rare)
    rank = np.empty(len(cs), int)
    rank[order] = np.arange(len(cs))

    on_rank, off_rank = float(rank[on].mean()), float(rank[~on].mean())
    assert on_rank < off_rank - 60, (
        f"on-object crops rank {on_rank:.1f} against off-object {off_rank:.1f}; the rarity signal "
        "the whole bootstrap rests on no longer separates them")
