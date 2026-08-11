# SPDX-License-Identifier: Apache-2.0
"""Telling objects apart by what they look like, and every step of the chain that makes it work.

Sorting labels by x gave *roles*: "leftmost" is a function of the current frame, so a head can
learn it. Identity is not — which object this is depends on which it *was* — and the sort rule
duly failed at it, 0.234 against a centre guess of 0.222.

**Appearance removes the history.** If the objects look different, the current frame says which is
which, so the head's target is a function of the frame again and the association moves to the
caller's side of the label. Measured end to end: **0.041, or 5.36x a centre guess**, better than
the sort rule ever managed even on roles.

Four things had to hold, and three of them were measured before anything was built:

  1. a random backbone's crop embeddings separate the objects        0.206 within/between
  2. nearest-template matching assigns them correctly               100.0%
  3. the head can express "where is the red one"                    0.072 with perfect labels
  4. the LOCATOR puts one detection on each object                  20.3% -> 96.6%

(4) was the whole failure and it took the longest to find: a frame difference is a temporal
derivative, so a mover leaves a dipole and two peaks usually land on one object.
"""

from __future__ import annotations

import numpy as np
import pytest

from argmind.adapt import NormalisedLeastMeanSquares
from argmind.backbone import Backbone
from demo.label_sources import AppearanceMatcher, BackgroundLocator, find_peaks, _grey
from demo.pointer import CROP, IN_SHAPE, PLAN, POOL, crop_at

H, W = IN_SHAPE[0], IN_SHAPE[1]
#: A red-ish and a blue-ish object, and the control that must fail: two the same.
DISTINCT = ((0.95, 0.25, 0.20), (0.20, 0.35, 0.95))
IDENTICAL = ((0.6, 0.6, 0.6), (0.6, 0.6, 0.6))


def scene(pts, colours, seed=0, bright=0.45, sigma=0.09):
    yy, xx = np.mgrid[0:H, 0:W]
    img = np.full((H, W, 3), bright)
    for (px, py), c in zip(pts, colours):
        d = ((xx - px * W) ** 2 + (yy - py * H) ** 2) / (2 * (sigma * W) ** 2)
        img += np.exp(-d)[:, :, None] * np.asarray(c, dtype=float)[None, None, :]
    return (img + np.random.default_rng(seed).standard_normal(img.shape) * 0.02).astype("f4")


def crossing(k, period=60.0, same_y=False):
    """Two objects sweeping through each other, repeatedly, staying in frame."""
    u = abs(1.0 - 2.0 * ((k / period) % 1.0))
    y0, y1 = (0.50, 0.50) if same_y else (0.35, 0.65)
    return (0.20 + 0.60 * u, y0), (0.80 - 0.60 * u, y1)


@pytest.fixture(scope="module")
def crop_backbone():
    return Backbone(PLAN, (CROP, CROP, 3), dtype="f4", seed=1, pool=2)


def describer(crop_backbone):
    def describe(frame, point):
        v = crop_backbone.features(crop_at(frame, point)).astype(np.float64)
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else v
    return describe


def test_a_random_backbone_separates_the_objects_by_appearance(crop_backbone):
    """Step 1, and the one that had to be true first. Within-object descriptor distance against
    between-object: 0.048 against 0.232, a ratio of 0.206. The identical pair reads 0.915."""
    describe = describer(crop_backbone)
    ratios = {}
    for name, colours in (("distinct", DISTINCT), ("identical", IDENTICAL)):
        a, b = [], []
        for k in range(60):
            pts = crossing(k)
            img = scene(pts, colours, seed=k)
            a.append(describe(img, pts[0]))
            b.append(describe(img, pts[1]))
        a, b = np.array(a), np.array(b)

        def pdist(x, y):
            return np.linalg.norm(x[:, None, :] - y[None, :, :], axis=2)

        n = len(a)
        within = (pdist(a, a)[np.triu_indices(n, 1)].mean()
                  + pdist(b, b)[np.triu_indices(n, 1)].mean()) / 2
        ratios[name] = float(within / pdist(a, b).mean())

    assert ratios["distinct"] < 0.4, ratios
    assert ratios["identical"] > 0.8, (
        f"the identical pair separates at {ratios['identical']:.3f}, so the descriptor is "
        "picking up something other than appearance and every result below is suspect")
    assert ratios["distinct"] < ratios["identical"] * 0.5


def test_a_temporal_derivative_is_the_wrong_locator_and_a_background_is_the_right_one():
    """Step 4, which was the actual failure. A frame difference leaves a dipole -- a lobe where
    the object was and one where it is -- so two peaks usually land on one object.

    Measured: 20.3% of detections put one peak on each object, against 96.6% for a running
    background. And no suppression radius fixes the derivative here: the dipole spacing is ~17 px
    for a blob of sigma 8.6 px, while the two objects come within ~21.6 px of each other.
    """
    def pairing_rate(kind, n=600, every=5):
        prev = bg = None
        det = paired = 0
        for k in range(n):
            pts = crossing(k)
            img = scene(pts, DISTINCT, seed=k)
            g = _grey(img)
            bg = g.astype(np.float64) if bg is None else bg + 0.05 * (g - bg)
            if prev is not None and k % every == 0:
                m = np.abs(g - prev) if kind == "motion" else np.abs(g - bg)
                found = find_peaks(m, 2, suppress=0.18 if kind == "motion" else 0.10)
                if len(found) == 2:
                    det += 1
                    d = np.array([[np.hypot(f[0] - p[0], f[1] - p[1]) for p in pts]
                                  for f in found])
                    paired += (d.min(axis=0) < 0.06).all() and \
                        len(set(np.argmin(d, axis=1))) == 2
            prev = g
        return paired / max(det, 1)

    motion, background = pairing_rate("motion"), pairing_rate("background")
    assert background > 0.85, f"background pairing fell to {background:.1%}"
    assert motion < 0.5, (
        f"frame differencing now pairs at {motion:.1%}; if the dipole has stopped appearing, the "
        "reason for BackgroundLocator is no longer demonstrated")
    assert background > motion * 3


def test_appearance_matching_is_exact_and_falls_to_chance_when_it_must(crop_backbone):
    """Steps 2 and 3 of the control chain, and the most important assertion in this file is the
    second one.

    Given detections that land on the objects, nearest-template matching was 100% correct. With
    *identical* objects it must fall to chance -- and it only does so fully when they are at the
    same height: at different heights the crop carries POSITION as well as appearance and matching
    reads 74.3%, falling to 50.6% when that leak is removed. That residual is why the crop size is
    documented as carrying position, and why the honest control puts them level.
    """
    describe = describer(crop_backbone)

    def accuracy(colours, same_y, n=900, every=5):
        loc = BackgroundLocator(2, rate=0.05)
        match = AppearanceMatcher(2, describe)
        seeded, ok, total = None, 0, 0
        for k in range(n):
            pts = crossing(k, same_y=same_y)
            img = scene(pts, colours, seed=k)
            found = loc(img) if k % every == 0 else None
            if found is None:
                continue
            d = np.array([[np.hypot(f[0] - p[0], f[1] - p[1]) for p in pts] for f in found])
            if not ((d.min(axis=0) < 0.06).all() and len(set(np.argmin(d, axis=1))) == 2):
                continue
            out = match(img, found)
            first = int(np.argmin([np.hypot(out[0][0] - p[0], out[0][1] - p[1]) for p in pts]))
            if seeded is None:
                seeded = first
                continue
            total += 1
            ok += first == seeded
        assert total > 40, f"only {total} usable detections; the fixture is not exercising this"
        return ok / total

    assert accuracy(DISTINCT, same_y=False) > 0.95
    assert accuracy(DISTINCT, same_y=True) > 0.95
    level = accuracy(IDENTICAL, same_y=True)
    assert 0.35 < level < 0.65, (
        f"identical objects at the same height are matched {level:.1%} of the time. Chance is the "
        "only honest answer here -- anything else means the descriptor is reading position or "
        "noise, and the appearance claim is not what it says it is")
    # And the leak itself, so the crop's position sensitivity is recorded rather than implied.
    assert accuracy(IDENTICAL, same_y=False) > level


def end_to_end(assign, colours, score="identity", n=1200, every=5, crop_backbone=None):
    """Drive the real composition and score against the mapping the matcher seeded.

    Scoring against "object A is slot 0" gave 0.430 -- the slot names are arbitrary, the matcher
    cannot know which object the fixture calls A, and at k=0 this crossing has A on the right. That
    was a constant swap read as a tracking failure.
    """
    backbone = Backbone(PLAN, IN_SHAPE, dtype="f4", seed=1, pool=POOL)
    cal = np.array([backbone.features(scene([(0.5, 0.5)], [colours[0]], seed=i))
                    for i in range(8)])
    backbone.output_scale = 1.0 / max(float(cal.std()), 1e-9)
    head = NormalisedLeastMeanSquares(backbone.n_outputs + 1, 4, mu=0.3, dtype="f4")
    loc = BackgroundLocator(2, rate=0.05)
    match = AppearanceMatcher(2, describer(crop_backbone)) if assign == "appearance" else None

    slot0, errs, ctrl = None, [], []
    for k in range(n):
        pts = crossing(k)
        img = scene(pts, colours, seed=k)
        x = np.concatenate([backbone.features(img), [1.0]])
        pred = head.predict(x).copy()
        found = loc(img) if k % every == 0 else None
        if found is not None:
            if match is not None:
                found = match(img, found)
            if slot0 is None:
                slot0 = int(np.argmin([np.hypot(found[0][0] - p[0], found[0][1] - p[1])
                                       for p in pts]))
            head.update(x, [c for xy in found for c in xy])
        if k > n // 2 and slot0 is not None:
            truth = ([pts[slot0], pts[1 - slot0]] if score == "identity"
                     else sorted(pts, key=lambda p: p[0]))
            errs.append((np.hypot(pred[0] - truth[0][0], pred[1] - truth[0][1]),
                         np.hypot(pred[2] - truth[1][0], pred[3] - truth[1][1])))
            ctrl.append((np.hypot(0.5 - truth[0][0], 0.5 - truth[0][1]),
                         np.hypot(0.5 - truth[1][0], 0.5 - truth[1][1])))
    return float(np.mean(errs)), float(np.mean(ctrl))


def test_appearance_tracks_identity_where_the_sort_rule_cannot(crop_backbone):
    """The claim, end to end through the real composition.

    | assignment | scored as | head | centre | |
    |---|---|---|---|---|
    | sorted by x | identity | 0.234 | 0.222 | 0.95x, fails |
    | appearance | identity | **0.041** | 0.222 | **5.36x** |
    """
    ident_app, centre = end_to_end("appearance", DISTINCT, crop_backbone=crop_backbone)
    ident_sort, _ = end_to_end("sorted", DISTINCT, crop_backbone=crop_backbone)
    assert centre > ident_app * 3.0, (
        f"appearance-assigned identity reads {ident_app:.3f} against a centre guess of "
        f"{centre:.3f}; the claim is that this is several times better, not marginally")
    assert ident_sort > centre * 0.8, (
        f"the sort rule now tracks identity at {ident_sort:.3f} against a centre guess of "
        f"{centre:.3f}. It should not be able to: which object is which depends on which it was")
    assert ident_app < ident_sort * 0.4


def test_each_assignment_rule_answers_only_its_own_question(crop_backbone):
    """The symmetry that says neither is simply better: sorting tracks roles and fails identity,
    appearance tracks identity and fails roles. Measured 0.063 against 0.242 on roles.

    Without this, "appearance is better" would be the reading, and a caller wanting *leftmost*
    would be given the wrong tool.
    """
    role_sort, centre = end_to_end("sorted", DISTINCT, score="role",
                                   crop_backbone=crop_backbone)
    role_app, _ = end_to_end("appearance", DISTINCT, score="role",
                             crop_backbone=crop_backbone)
    assert centre > role_sort * 2.0, f"sorting no longer tracks roles: {role_sort:.3f}"
    assert role_app > role_sort * 2.0, (
        f"appearance assignment now tracks roles too ({role_app:.3f} against {role_sort:.3f}); "
        "if that is real then one rule serves both questions and this file is wrong")


def test_identical_objects_degrade_the_end_to_end_result(crop_backbone):
    """The control on the whole chain. With nothing to tell apart, identity tracking has to
    degrade -- measured 0.162 against 0.041, and the residual is the crop's position sensitivity
    rather than appearance."""
    good, _ = end_to_end("appearance", DISTINCT, crop_backbone=crop_backbone)
    poor, centre = end_to_end("appearance", IDENTICAL, crop_backbone=crop_backbone)
    assert poor > good * 2.5, (
        f"identical objects give {poor:.3f} against {good:.3f} for distinct ones. If these are "
        "close, the result is not coming from appearance")
    assert poor < centre * 1.2


def test_the_matcher_holds_its_templates_and_refuses_a_size_it_cannot_solve(crop_backbone):
    """Two deliberate choices.

    Templates are seeded once and **held**: updating them per frame would let a template drift
    onto the other object during a crossing, which is the same shape as the guard's baseline
    tracking the very fault it was holding out against.

    And above four objects it **refuses** rather than falling back to a greedy match, which would
    be silently wrong exactly when the objects are hard to tell apart.
    """
    describe = describer(crop_backbone)
    m = AppearanceMatcher(2, describe)
    assert not m.seeded
    img = scene(crossing(0), DISTINCT, seed=1)
    first = m(img, [(0.8, 0.35), (0.2, 0.65)])
    assert m.seeded
    assert first == [(0.2, 0.65), (0.8, 0.35)], "seeding is left-to-right"
    before = [t.copy() for t in m._templates]                      # noqa: SLF001
    m(scene(crossing(30), DISTINCT, seed=2), [(0.5, 0.35), (0.5, 0.65)])
    for a, b in zip(before, m._templates):                          # noqa: SLF001
        assert np.array_equal(a, b), "the templates moved; they are supposed to be held"

    m.reset()
    assert not m.seeded
    with pytest.raises(ValueError, match="expected 2 points"):
        AppearanceMatcher(2, describe)(img, [(0.5, 0.5)])
    with pytest.raises(ValueError, match="assignment algorithm"):
        AppearanceMatcher(5, describe)(img, [(0.1 * i, 0.5) for i in range(5)])
    with pytest.raises(ValueError, match="at least 1"):
        AppearanceMatcher(0, describe)


def test_the_background_locator_warms_up_and_refuses_a_short_list():
    """It cannot report anything until the background has settled, and a scene with fewer objects
    than slots gets None rather than a partial list that would silently reassign them."""
    loc = BackgroundLocator(2, warmup=10)
    early = [loc(scene(crossing(k), DISTINCT, seed=k)) for k in range(9)]
    assert all(v is None for v in early), "reported before the background settled"
    assert not loc.ready
    later = [loc(scene(crossing(k), DISTINCT, seed=k)) for k in range(10, 40)]
    assert loc.ready
    assert any(v is not None for v in later)
    for v in later:
        if v is not None:
            assert len(v) == 2 and v[0][0] < v[1][0]

    loc.reset()
    assert not loc.ready
    with pytest.raises(ValueError, match="at least 1"):
        BackgroundLocator(0)
    with pytest.raises(ValueError, match="rate must be"):
        BackgroundLocator(1, rate=0.0)


def test_the_locator_does_not_count_and_loses_a_stationary_object():
    """Two limits of background subtraction, both measured, and the second one surprised me.

    **It cannot count.** `k` is a declaration of how many slots the caller wants, not a detection.
    Asked for two peaks in a one-object scene it returns two, one of them noise. That is the same
    limitation already recorded for the motion detector, arriving by another route -- so a caller
    setting `--objects 2` on a scene with one thing in it gets a second slot chasing nothing.

    **And a stationary object is absorbed into the background and lost, at every rate.** Measured,
    the fraction of frames where the found peak lands on the object:

    | px/frame | rate=0.02 | 0.05 | 0.15 | 0.4 |
    |---|---|---|---|---|
    | 0.0 | **0%** | **0%** | **0%** | **0%** |
    | 0.5 | 91% | 93% | 0% | 0% |
    | 1.0 | **100%** | 98% | 13% | 0% |
    | 1.9 | 27% | 27% | 24% | 4% |
    | 4.8 | 0% | 0% | 0% | 0% |

    So `rate` has a narrow usable window and it is not a free parameter. An earlier version of this
    measurement reported 100% at every rate, because it asked whether *a* peak existed rather than
    whether the peak was on the object -- a check that could not produce a negative result, which
    is the thing this project's standards forbid. Reading the peak's coordinates is what caught it:
    (0.89, 0.82) for an object at (0.35, 0.50).
    """
    def on_object_rate(speed, rate, n=80, warm=25):
        loc = BackgroundLocator(1, rate=rate, warmup=warm, suppress=0.10)
        hits = total = 0
        for k in range(n):
            p = (0.25 + speed * k, 0.5)
            got = loc(scene([p], [DISTINCT[0]], seed=k))
            if got is None:
                continue
            total += 1
            hits += np.hypot(got[0][0] - p[0], got[0][1] - p[1]) < 0.06
        assert total > 20, total
        return hits / total

    assert on_object_rate(0.0, 0.05) < 0.05, (
        "a stationary object is now found against the background; if that is real the rate knob "
        "behaves differently and the table above needs re-measuring")
    assert on_object_rate(0.01, 0.02) > 0.8, "the usable window has moved"
    assert on_object_rate(0.05, 0.05) < 0.2, "a fast mover is now tracked; re-measure"

    # And the counting limitation, asserted rather than implied.
    lone = BackgroundLocator(2, warmup=5, rate=0.02)
    got = [v for v in (lone(scene([(0.25 + 0.01 * k, 0.5)], [DISTINCT[0]], seed=k))
                       for k in range(40)) if v is not None]
    assert got, "the one-object scene produced nothing at all"
    assert all(len(v) == 2 for v in got), (
        "the locator now returns fewer than k on a one-object scene. That would be an "
        "improvement, and the docstring claiming it cannot count would need to change")
