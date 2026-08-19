# SPDX-License-Identifier: Apache-2.0
"""Can the rarity SCAN be distilled into a cheap proposal? Partly — and a point cannot do it at all.

`tests/test_bootstrap.py` acquires a first exemplar with no labels, but at ~247 crop embeddings and
~61,000 similarities per frame. That is search, not attention. This asks whether a head can predict
where the scan would have looked, from one whole-frame pass.

**Why this is not the trap the wean experiment found.** Rarity is a fixed function of the frame, not
the head's own output, so the teacher does not drift as the student learns. Self-training degraded
0.938 → 0.804 because the target moved; this target cannot.

| proposal | lands on an object |
|---|---|
| full scan (the teacher) | 0.933 |
| learned **map**, one frame pass | **0.617** |
| learned **point**, one frame pass | 0.267 |
| chance (a grid cell) | 0.062 |

Those are at 160 training frames. **This file trains on 80** so it costs 55 s rather than two
minutes, and the gap narrows there — map 0.500 against point 0.300. The assertions below are set to
what is reproducible at the size they actually run, and the larger figures are quoted as
measurements rather than being asserted at a size nothing here reaches.

**A point fails for a measured reason, not a mysterious one.** With two objects the teacher picks one
or the other, so a least-squares point regression learns their average: **8 of 10** proposals land
between them, and every predicted `y` sits at 0.42–0.49, the mean of the two bands. That is this
repository's recorded "a difference map with two movers puts its centroid between them — a point
cannot stand in for a field", and the recorded remedy is the one that works here: a map output has
nothing to permute, nothing to count and no assignment to make.

**The remaining shortfall is not grid quantisation.** An oracle picking the best cell of the 6×8 grid
scores **1.000** at this tolerance, so the head reaches 62% of what the grid can express and the gap
is the head's.

**A hold-out was corrected mid-experiment.** The first split trained on objects in the left band and
tested in the right, which is the right hold-out for *recognition* and the wrong one for
*localisation*, where the position **is** the output: it demands extrapolation beyond any value ever
seen. Measured, the head scores 0.000 under it. The split was changed because the demand was
incoherent, not because the number was bad, and both numbers are recorded.
"""

from __future__ import annotations

import numpy as np
import pytest

from argmind.adapt import RecursiveLeastSquares
from argmind.backbone import Backbone
from tests.test_bootstrap import H, POOL, W, crop_at, interior, rarest, render
from tests.test_bootstrap import PLAN as CROP_PLAN
from tests.test_recognition_memory import CROP

FRAME_PLAN = [("conv", 12, 5, 4), ("dw", 3, 2), ("pw", 16)]
GH, GW = 6, 8
TOL = 0.10


def specs_at(rng):
    """Positions from the same range for train and test — see the docstring on the corrected split."""
    return [("disc", rng.uniform(0.20, 0.80), rng.uniform(0.28, 0.46)),
            ("bar", rng.uniform(0.20, 0.80), rng.uniform(0.58, 0.76))]


def _embed_all(crop_bb, img, cs):
    E = []
    for cx, cy in cs:
        v = np.asarray(crop_bb.features(crop_at(img, cx, cy, H, W)), dtype=np.float64)
        n = float(np.linalg.norm(v))
        E.append(v / n if n > 1e-12 else v)
    return np.asarray(E)


def rarity_map(crop_bb, img, cs):
    """Per-candidate rarity, block-averaged onto the grid. Label-free: nothing knows where the
    objects are."""
    E = _embed_all(crop_bb, img, cs)
    S = E @ E.T
    rare = -(S.sum(axis=1) - np.diag(S)) / (len(cs) - 1)
    grid, count = np.zeros((GH, GW)), np.zeros((GH, GW))
    for (cx, cy), r in zip(cs, rare):
        gy, gx = min(int(cy * GH), GH - 1), min(int(cx * GW), GW - 1)
        grid[gy, gx] += r
        count[gy, gx] += 1
    grid = np.where(count > 0, grid / np.maximum(count, 1), rare.min())
    lo, hi = grid.min(), grid.max()
    return ((grid - lo) / (hi - lo + 1e-12)).ravel()


def train_and_score(kind, *, seed=23, n_train=80, n_test=30):
    crop_bb = Backbone(CROP_PLAN, (CROP, CROP, 3), dtype="f8", seed=seed, pool=POOL)
    frame_bb = Backbone(FRAME_PLAN, (H, W, 3), dtype="f8", seed=seed, pool=3)
    cs = interior(H, W)
    m = GH * GW if kind == "map" else 2
    head = RecursiveLeastSquares(frame_bb.n_outputs + 1, m, forgetting=1.0, p0=100.0, dtype="f8")

    rng = np.random.default_rng(5)
    for t in range(n_train):
        img = render(specs_at(rng), seed=t, texture="none")
        x = np.concatenate([np.asarray(frame_bb.features(img)), [1.0]])
        head.update(x, rarity_map(crop_bb, img, cs) if kind == "map"
                    else list(rarest(crop_bb, img, cs, H, W)))

    tr = np.random.default_rng(77)
    hits, between, chance = 0, 0, []
    for t in range(n_test):
        specs = specs_at(tr)
        img = render(specs, seed=9000 + t, texture="none")
        near = lambda cx, cy: any(np.hypot(cx - p[1], cy - p[2]) < TOL for p in specs)
        out = np.asarray(head.predict(
            np.concatenate([np.asarray(frame_bb.features(img)), [1.0]])))
        if kind == "map":
            gy, gx = np.unravel_index(int(np.argmax(out.reshape(GH, GW))), (GH, GW))
            px, py = (gx + 0.5) / GW, (gy + 0.5) / GH
        else:
            px, py = float(out[0]), float(out[1])
        hits += int(near(px, py))
        between += int(not near(px, py)
                       and abs(py - 0.5 * (specs[0][2] + specs[1][2])) < TOL)
        chance.append(np.mean([near((x + 0.5) / GW, (y + 0.5) / GH)
                               for y in range(GH) for x in range(GW)]))
    return hits / n_test, between / n_test, float(np.mean(chance))


@pytest.fixture(scope="module")
def scores():
    return {"map": train_and_score("map"), "point": train_and_score("point")}


def test_a_MAP_proposal_beats_a_POINT_proposal_by_a_wide_margin(scores):
    """The recorded remedy, working. Either number alone would be uninformative — the contrast is
    the finding, and both are trained on the same teacher from the same frames."""
    (m_hit, _mb, chance), (p_hit, _pb, _c) = scores["map"], scores["point"]
    assert chance < 0.15, f"a grid cell hits an object {chance:.3f} of the time by luck"
    assert m_hit > 3 * chance, f"the map proposal reaches only {m_hit:.3f} against chance {chance:.3f}"
    assert m_hit > p_hit + 0.15, (
        f"map {m_hit:.3f} against point {p_hit:.3f} — the gap that says a point cannot stand in for "
        "a field has closed, and the conclusion should move with it")


def test_the_POINT_proposal_fails_by_averaging_the_two_objects(scores):
    """Named mechanism, not a mysterious shortfall: the teacher picks one object or the other and
    least squares learns their mean."""
    _hit, between, _chance = scores["point"]
    assert between > 0.3, (
        f"only {between:.3f} of the point proposals land between the two objects; the averaging "
        "explanation for its failure is no longer supported and should be re-derived")


def test_the_shortfall_is_NOT_grid_quantisation():
    """Asked before concluding: can this metric reach 1.0 at all? An oracle choosing the best cell
    scores 1.000, so the gap to the teacher belongs to the head."""
    tr = np.random.default_rng(77)
    oracle = []
    for _t in range(30):
        specs = specs_at(tr)
        oracle.append(any(any(np.hypot((x + 0.5) / GW - p[1], (y + 0.5) / GH - p[2]) < TOL
                              for p in specs)
                          for y in range(GH) for x in range(GW)))
    assert np.mean(oracle) > 0.95, (
        f"the grid can only express {np.mean(oracle):.3f}; the map head's shortfall would then be "
        "quantisation and not its own")
