# SPDX-License-Identifier: Apache-2.0
"""Tracking more than one object: what works, and the boundary that is structural.

The demo tracked one point, and pointed it at the *midpoint* of two movers — accurately, which is
why it looked like a recognition failure rather than an aggregation one. Two things had to be
separated before anything was built:

  (a) the LABEL aggregates. A difference map with two movers has one centroid and it sits between
      them. Fixable: find peaks instead.
  (b) the HEAD is permutation-symmetric. With 2k outputs nothing distinguishes output pair 1 from
      pair 2, so unless the labels break that symmetry the same way every frame, the least-squares
      answer for both pairs is the same thing.

(b) is settled by an assignment rule on the caller's side — peaks sorted by x — and what that buys
is **roles, not identities**. Both halves are asserted here, and the identity failure is the more
important one to keep: it is worse than guessing the centre of the frame.
"""

from __future__ import annotations

import numpy as np
import pytest

from demo.label_sources import _grey, find_peaks
from demo.pointer import IN_SHAPE, build, fingerprint

H, W = IN_SHAPE[0], IN_SHAPE[1]


def two_movers(k, cross=False):
    """Two blobs. `cross` sends them back and forth through each other, staying in frame.

    The first version of this used `u = k / 300` over 900 frames, which sent them to x = 2.0 —
    off the picture entirely — and the errors read 0.89 on a frame 1.0 wide. An absurd number is
    what caught it, which is the argument for reading them.
    """
    if cross:
        u = abs(1.0 - 2.0 * ((k / 300.0) % 1.0))
        return (0.20 + 0.60 * u, 0.35), (0.80 - 0.60 * u, 0.65)
    a = k * 0.09
    return ((0.25 + 0.10 * np.sin(a), 0.35 + 0.15 * np.cos(a)),
            (0.75 + 0.10 * np.cos(a * 1.1), 0.65 + 0.15 * np.sin(a * 0.9)))


def scene(pts, seed=0, bright=0.45):
    yy, xx = np.mgrid[0:H, 0:W]
    img = np.full((H, W, 3), bright)
    for px, py in pts:
        d = ((xx - px * W) ** 2 + (yy - py * H) ** 2) / (2 * (0.09 * W) ** 2)
        img += np.exp(-d)[:, :, None] * 0.9
    return (img + np.random.default_rng(seed).standard_normal(img.shape) * 0.02).astype("f4")


def run(order, cross=False, score="role", n=900, every=10, seed=1):
    """Drive a two-object head. `order` is 'sorted' or 'arbitrary'; `score` is 'role' (against
    leftmost/rightmost) or 'identity' (against the persistent trajectories)."""
    backbone, _head, guard, features, taught, _fmap, pipe = build(2)
    cal = np.array([backbone.features(scene([(0.5, 0.5)], seed=i)) for i in range(8)])
    backbone.output_scale = 1.0 / max(float(cal.std()), 1e-9)
    base_std = float(np.std(scene([(0.5, 0.5)], seed=0)))
    g = np.random.default_rng(seed)

    prev, t, errs, ctrl = None, 0.0, [], []
    for k in range(n):
        t += 1.0 / 30.0
        pts = two_movers(k, cross)
        img = scene(pts, seed=k)
        grey = _grey(img)
        guard.set_external(0, float(np.std(img)) / max(base_std, 1e-6))
        features.push(t, backbone.features(img))
        if prev is not None and k % every == 0:
            found = find_peaks(np.abs(grey - prev), 2)
            if len(found) == 2:
                if order == "sorted":
                    found.sort(key=lambda p: p[0])
                elif g.random() < 0.5:
                    found = found[::-1]        # the order a detector happens to return
                taught.push(t, [c for xy in found for c in xy])
        prev = grey
        report = pipe.step()
        if report.prediction is None or k <= n // 2:
            continue
        pred = tuple(float(v) for v in report.prediction)     # read now, not later
        truth = sorted(pts, key=lambda p: p[0]) if score == "role" else list(pts)
        errs.append((np.hypot(pred[0] - truth[0][0], pred[1] - truth[0][1]),
                     np.hypot(pred[2] - truth[1][0], pred[3] - truth[1][1])))
        ctrl.append((np.hypot(0.5 - truth[0][0], 0.5 - truth[0][1]),
                     np.hypot(0.5 - truth[1][0], 0.5 - truth[1][1])))
    return float(np.mean(errs)), float(np.mean(ctrl))


def test_two_objects_are_tracked_when_the_labels_are_ordered():
    """The claim: 2k outputs plus a consistent assignment rule tracks both.

    Measured 0.129 against a centre-guess 0.312, so 2.43x. The assertion is the direction and a
    generous margin, because the ratio depends on how far apart the objects are in the fixture.
    """
    head, centre = run("sorted")
    assert centre > head * 1.8, (
        f"the two-object head ({head:.3f}) no longer clearly beats a centre guess "
        f"({centre:.3f}); with 2k outputs the failure mode is predicting the mean of the "
        "objects, which is very close to the centre in this fixture")


def test_labels_in_an_arbitrary_order_learn_the_mean_instead():
    """The mechanism behind the rule, and the reason it is not optional.

    A head with 2k outputs is permutation-symmetric: nothing distinguishes output pair 1 from pair
    2. Feed the pairs in whatever order the detector found them and the least-squares answer for
    both is the same thing. Measured 0.256 against a centre guess of 0.312 — 1.22x, which is the
    centre guess with extra steps.
    """
    sorted_err, centre = run("sorted")
    arb_err, _ = run("arbitrary")
    assert arb_err > sorted_err * 1.6, (
        f"arbitrary order ({arb_err:.3f}) is no longer much worse than sorted "
        f"({sorted_err:.3f}), so the assignment rule is not doing what this file says")
    assert arb_err > centre * 0.7, (
        f"arbitrary order ({arb_err:.3f}) is doing better than expected against the centre "
        f"guess ({centre:.3f}); if it genuinely learns, the permutation argument is wrong")


def test_a_crossing_does_not_break_ROLE_tracking():
    """A prediction of mine that was wrong, kept because the correction is the useful part.

    I expected crossings to break a sort rule. They do not: sorting makes "leftmost" a
    well-defined function of the current frame, and a function is exactly what a head can learn.
    Measured 0.137 with crossings against 0.129 without.
    """
    plain, _ = run("sorted", cross=False)
    crossed, centre = run("sorted", cross=True)
    assert crossed < plain * 1.5, (
        f"crossings now cost {crossed / plain:.2f}x ({plain:.3f} -> {crossed:.3f}); if that is "
        "real, role tracking is not the frame-wise function this file claims it is")
    assert centre > crossed * 1.3, (crossed, centre)


def test_it_cannot_track_IDENTITY_through_a_crossing_and_that_is_structural():
    """The boundary, and it is the negative result worth protecting.

    Scored against the persistent trajectories instead of against leftmost/rightmost, the same run
    reads 0.251 against a centre guess of 0.222 — **worse than guessing the centre of the frame**.

    Not a tuning failure. Association is a question about the input's *history*: which blob this
    is depends on which blob it was. CLAUDE.md's opening states the constraint — nothing with
    memory of its own input can be fitted by a map from instantaneous features — and this is that
    statement arriving in a concrete task. Anything that fixes it is a tracker with state, which
    is a different component and not this model class.
    """
    role_err, _ = run("sorted", cross=True, score="role")
    ident_err, centre = run("sorted", cross=True, score="identity")
    assert ident_err > role_err * 1.5, (
        f"identity scoring ({ident_err:.3f}) is now close to role scoring ({role_err:.3f}); "
        "either the fixture stopped crossing or something is carrying state")
    assert ident_err > centre, (
        f"identity tracking ({ident_err:.3f}) now beats a centre guess ({centre:.3f}). If that "
        "is real it is a significant finding and this test should say so -- but check first "
        "that the objects actually cross, because without a crossing identity and role are the "
        "same thing")


def test_the_composition_is_shaped_by_the_object_count():
    """The shapes, and the fingerprint that stops a saved block being restored into the wrong one.

    Feature order is parameter order, so a two-object block loaded into a one-object head would
    put the second object's coordinates nowhere and the CRC would be perfectly happy.
    """
    shapes = {}
    for k in (1, 2, 3):
        backbone, head, _guard, _features, taught, fmap, _pipe = build(k)
        assert head.n_outputs == 2 * k
        assert taught.dim == 2 * k
        assert len(fmap.target) == 2 * k
        shapes[k] = (np.asarray(head.state).nbytes, fingerprint(backbone, k))
    assert shapes[2][0] == 2 * shapes[1][0]
    assert len({fp for _n, fp in shapes.values()}) == 3, (
        "two object counts share a fingerprint, so a block from one would load into the other")
    with pytest.raises(ValueError, match="at least 1"):
        build(0)


def test_find_peaks_reports_fewer_rather_than_inventing_them():
    """A partial list would silently reassign slots, which is the permutation problem the sort
    rule exists to prevent -- so the detector declares the shortfall and the caller pushes
    nothing."""
    prev = _grey(scene([(0.3, 0.4), (0.7, 0.6)], seed=1))
    both = np.abs(_grey(scene([(0.34, 0.44), (0.74, 0.64)], seed=2)) - prev)
    assert len(find_peaks(both, 2)) == 2
    assert len(find_peaks(both, 5)) < 5, "five peaks found in a two-blob difference map"

    flat = np.zeros((H, W))
    assert find_peaks(flat, 2) == []
    with pytest.raises(ValueError, match="at least 1"):
        find_peaks(both, 0)

    # Suppression is what stops one object being reported as two, and it bounds how close two
    # objects may be before they merge -- a limit of the detector, not of the head.
    one_prev = _grey(scene([(0.5, 0.5)], seed=3))
    one = np.abs(_grey(scene([(0.54, 0.54)], seed=4)) - one_prev)
    assert len(find_peaks(one, 2, suppress=0.30)) == 1
