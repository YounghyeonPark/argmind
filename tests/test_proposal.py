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

Those are at 160 training frames on a 247-candidate grid. **This file uses a 70-candidate grid**
(`STEP = 8`), which leaves the teacher unchanged at 0.938 and runs 3.4× faster, and trains on 200
frames. Measured at that setting the map reads 0.583 against the point's 0.300.

**80 frames was too few and the first version of this file used it.** The gap there is only +0.100,
because the *point* head scores 0.467 on noise it loses with more data (0.300 at 200 frames and at
400). Sharpening the contrast cost nothing here, since the coarser grid had already paid for the
extra frames: 12 s at 80/30, 39 s at 200/60, 109 s at 400/60.

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
#: A coarser candidate grid than `test_bootstrap`'s default — 70 crops per frame instead of 247.
#: The teacher is **unchanged** by it (0.938 either way, measured) and the file runs 3.4x faster,
#: which matters because the pre-commit hook runs the whole suite and this was its slowest file.
#: Every figure in this module is re-measured at this density rather than inherited.
STEP = 8
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


def train_and_score(kind, *, seed=23, n_train=200, n_test=60):
    crop_bb = Backbone(CROP_PLAN, (CROP, CROP, 3), dtype="f8", seed=seed, pool=POOL)
    frame_bb = Backbone(FRAME_PLAN, (H, W, 3), dtype="f8", seed=seed, pool=3)
    cs = interior(H, W, STEP)
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
    assert m_hit > p_hit + 0.2, (
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


_TEACHER: dict = {}


def _teacher(seed, n_train):
    """The scan's targets, computed once and shared.

    **The rarity map does not depend on `pool`** — it is built from crop embeddings, and only the
    *student's* frame embedding changes across the sweep. Recomputing it per setting doubled the
    runtime of this file for nothing.
    """
    key = (seed, n_train)
    if key not in _TEACHER:
        crop_bb = Backbone(CROP_PLAN, (CROP, CROP, 3), dtype="f8", seed=seed, pool=POOL)
        cs, rng, out = interior(H, W), np.random.default_rng(5), []
        for t in range(n_train):
            img = render(specs_at(rng), seed=t, texture="none")
            out.append((img, rarity_map(crop_bb, img, cs)))
        _TEACHER[key] = out
    return _TEACHER[key]


def _proposal_at(pool, *, seed=23, n_train=120, n_test=48):
    """Score the map proposal with the frame embedding pooled to `pool` x `pool`."""
    frame_bb = Backbone(FRAME_PLAN, (H, W, 3), dtype="f8", seed=seed, pool=pool)
    head = RecursiveLeastSquares(frame_bb.n_outputs + 1, GH * GW,
                                 forgetting=1.0, p0=100.0, dtype="f8")
    for img, target in _teacher(seed, n_train):
        head.update(np.concatenate([np.asarray(frame_bb.features(img)), [1.0]]), target)
    tr, hits = np.random.default_rng(77), 0
    for t in range(n_test):
        specs = specs_at(tr)
        img = render(specs, seed=9000 + t, texture="none")
        out = np.asarray(head.predict(
            np.concatenate([np.asarray(frame_bb.features(img)), [1.0]]))).reshape(GH, GW)
        gy, gx = np.unravel_index(int(np.argmax(out)), (GH, GW))
        hits += int(any(np.hypot((gx + 0.5) / GW - p[1], (gy + 0.5) / GH - p[2]) < TOL
                        for p in specs))
    return hits / n_test, frame_bb.n_outputs


def test_the_gap_to_the_teacher_is_SPATIAL_RESOLUTION_and_not_training_data():
    """**Where the missing third went.** The target is a 6×8 map and the frame embedding pooled to
    3×3, so the head was asked to upsample. Sweeping only that axis, at 160 training frames:

    | pool | embedding | proposal |
    |---|---|---|
    | 2 | 64 | 0.517 |
    | 3 | 144 | 0.617 |
    | 4 | 256 | 0.717 |
    | 6 | 576 | 0.817 |
    | 9 | 1,296 | 0.850 |

    against a teacher at 0.933. **Tripling the data instead buys almost nothing** — 0.617 → 0.667 at
    pool 3, 0.817 → 0.833 at pool 6 — so the limit is structural.

    Asserted on two points at a size the suite can afford, since the monotone claim is what matters
    and the full sweep is minutes. **The margin is set from measurement, not from the headline
    figures**: at 120 training frames and 48 test frames the gap is +0.292, and at 60/24 it is only
    +0.125 — small samples move it by more than a fifth, so the threshold is placed where the
    smaller of the two measured settings still clears it.
    """
    coarse, n_coarse = _proposal_at(2)
    fine, n_fine = _proposal_at(6)
    assert n_fine > n_coarse, "the pool no longer changes the embedding width"
    assert fine > coarse + 0.15, (
        f"pool 6 reaches {fine:.3f} against pool 2's {coarse:.3f}; the resolution explanation for "
        "the shortfall is no longer supported and should be re-derived")


def test_the_resolution_ceiling_is_the_feature_map_and_it_is_REFUSED_not_truncated():
    """A hard structural limit, and the library says so rather than silently coarsening: the plan's
    last feature map is 9×12, so a 12×12 grid cannot be taken. That is why the sweep stops at 9."""
    frame_bb = Backbone(FRAME_PLAN, (H, W, 3), dtype="f8", seed=23, pool=9)
    assert frame_bb.n_outputs == 9 * 9 * 16

    # **The refusal is at CALL time, not construction** -- `Backbone(pool=12)` builds happily and
    # `grid_avg_pool` raises when it sees the 9x12 map. Asserted as it behaves rather than as it was
    # first written, which expected the constructor to catch it.
    too_fine = Backbone(FRAME_PLAN, (H, W, 3), dtype="f8", seed=23, pool=12)
    with pytest.raises(ValueError, match="at least that big"):
        too_fine.features(np.zeros((H, W, 3), dtype="f8"))


def test_the_last_gap_belongs_to_the_TARGET_and_not_to_the_head():
    """**Asked before varying the head's form, and it made that unnecessary.**

    The student reaches 0.850 against a teacher at 0.933, and the sweep above showed the shortfall
    is neither resolution nor data — which pointed at the head being linear. It is not. The teacher's
    own targets are 247 candidate scores block-averaged onto a 6×8 grid, and *taking the argmax of
    that averaged map* is what the student is being trained to imitate. Measured over four test
    seeds:

    | test seed | teacher, raw argmax over 247 | teacher, argmax of its own 6×8 map |
    |---|---|---|
    | 77 | 0.933 | 0.833 |
    | 101 | 0.900 | 0.800 |
    | 202 | 0.900 | 0.850 |
    | 303 | 0.900 | 0.850 |

    Block-averaging costs about **0.075**, and the student at pool 9 measured 0.850 — at the ceiling
    within seed noise. **So the distillation is closed at the student's level**, and going further
    means changing the target representation — a finer grid, or regressing an offset within the cell
    — not the head. Varying the head form would have been chasing a gap that does not exist.

    Two seeds here rather than four, for runtime; the four-seed table is quoted as a measurement.
    """
    crop_bb = Backbone(CROP_PLAN, (CROP, CROP, 3), dtype="f8", seed=23, pool=POOL)
    cs = interior(H, W, STEP)
    raws, mappeds = [], []
    for ts in (77, 202):
        tr, raw, mapped, n = np.random.default_rng(ts), 0, 0, 30
        for t in range(n):
            specs = specs_at(tr)
            img = render(specs, seed=9000 + t + ts, texture="none")
            near = lambda cx, cy: any(np.hypot(cx - p[1], cy - p[2]) < TOL for p in specs)
            sx, sy = rarest(crop_bb, img, cs, H, W)
            raw += int(near(sx, sy))
            m = rarity_map(crop_bb, img, cs).reshape(GH, GW)
            gy, gx = np.unravel_index(int(np.argmax(m)), (GH, GW))
            mapped += int(near((gx + 0.5) / GW, (gy + 0.5) / GH))
        raws.append(raw / n)
        mappeds.append(mapped / n)

    assert np.mean(raws) > 0.85, f"the raw teacher fell to {np.mean(raws):.3f}"
    assert np.mean(raws) - np.mean(mappeds) > 0.03, (
        f"block-averaging the target now costs only {np.mean(raws) - np.mean(mappeds):.3f}; if that "
        "became free, the student's shortfall would be the head's after all and the conclusion here "
        "should be re-derived")
    assert np.mean(mappeds) < 0.90, (
        f"the averaged-map ceiling is {np.mean(mappeds):.3f}, at or above what the student reached — "
        "there would then be room the head is not using")
