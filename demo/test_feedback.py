# SPDX-License-Identifier: Apache-2.0
"""Feeding the head's own past output back into its input: what it buys, and what it costs.

DESIGN.md says no choice of *instantaneous* features represents a history-dependent quantity, and
which object this is depends on which it was. That is a statement about the **basis**, not the
model class: `y_t = f(x_t, y_{t-1}, ...)` is still linear in the parameters, and "the caller
supplies the basis" is the licence to put lagged outputs in it.

Every claim here is measured on a fixture where nothing else can possibly work: two identical
objects orbiting 180 degrees apart, whose image is **bit-identical** under swapping their labels.
The fixture asserts that invariance itself, because three earlier attempts at it leaked -- objects
at different heights, different colours, and a trajectory that never actually swapped.
"""

from __future__ import annotations

import numpy as np
import pytest

from argmind.adapt import RecursiveLeastSquares
from demo.pointer import (FEEDBACK_RATES, OutputMemory, basis, build, feedback_width,
                          frame_width, raw_features, raw_width)
from demo.test_pointer import frame_at
from demo.test_object_memory import scene

N = 2
#: Identical on purpose: appearance must be unable to help, or the fixture answers a different
#: question. Measured earlier: two distinct colours make the frame-only head reach 118x, so a
#: coloured fixture cannot test feedback at all.
SAME = ((0.60, 0.60, 0.60), (0.60, 0.60, 0.60))


def orbit_pair(k, period=140.0):
    """Two identical objects 180 degrees apart on a circle."""
    a = 2 * np.pi * k / period
    return ((0.5 + 0.26 * np.cos(a), 0.5 + 0.24 * np.sin(a)),
            (0.5 - 0.26 * np.cos(a), 0.5 - 0.24 * np.sin(a)))


def test_the_fixture_is_genuinely_ambiguous():
    """The guard on everything below, and it earned its place: three earlier versions leaked.
    Objects at fixed different heights let height identify them; different colours let appearance
    do it; a trajectory crossing only in x never actually swapped. Here the image must be
    bit-identical under exchanging the two labels."""
    for k in (0, 37, 91):
        a = raw_features(scene(list(orbit_pair(k)), SAME, seed=5))
        b = raw_features(scene(list(orbit_pair(k))[::-1], SAME, seed=5))
        assert np.array_equal(a, b), (
            f"frame {k} differs under the swap by {np.abs(a - b).max():.2e}; the fixture leaks "
            "and every number in this file would be measuring something else")


def drive(form, *, free_running, lags=1, rates=FEEDBACK_RATES, n=3000, p0=100.0, blind=None):
    """Returns (error, centre-guess error, persistence error) over the held-out half."""
    bank = OutputMemory(2 * N, rates) if form == "bank" else None
    hist: list[np.ndarray] = []
    probe = raw_features(scene(list(orbit_pair(0)), SAME, seed=0))
    extra = 0 if form == "frame" else (lags * 2 * N if form == "lag" else bank.width)
    head = RecursiveLeastSquares(len(probe) + extra + 1, 2 * N,
                                 forgetting=1.0, p0=p0, dtype="f8")
    errs, ctrl, pers, prev = [], [], [], None
    for k in range(n):
        pts = list(orbit_pair(k))
        img = scene(pts, SAME, seed=k)
        if blind and blind[0] <= k < blind[1]:
            img = np.zeros_like(img)
        feat = raw_features(img)
        truth = np.array([c for xy in pts for c in xy])

        if form == "frame":
            x = np.concatenate([feat, [1.0]])
        elif form == "lag":
            mem = [hist[i] if i < len(hist) else np.zeros(2 * N) for i in range(lags)]
            x = np.concatenate([feat, *mem, [1.0]])
        else:
            x = np.concatenate([feat, bank.features(), [1.0]])

        pred = head.predict(x).copy()
        train = k <= n // 2
        if train:
            head.update(x, truth)
        fed = truth if (train or not free_running) else pred
        if form == "bank":
            bank.observe(fed)
        elif form == "lag":
            hist.insert(0, np.asarray(fed, dtype=float).copy())
            del hist[lags:]

        if not train:
            errs.append(np.mean([np.hypot(pred[2 * i] - truth[2 * i],
                                          pred[2 * i + 1] - truth[2 * i + 1])
                                 for i in range(N)]))
            ctrl.append(np.mean([np.hypot(0.5 - truth[2 * i], 0.5 - truth[2 * i + 1])
                                 for i in range(N)]))
            if prev is not None:
                pers.append(np.mean([np.hypot(prev[2 * i] - truth[2 * i],
                                              prev[2 * i + 1] - truth[2 * i + 1])
                                     for i in range(N)]))
        prev = truth
    return float(np.mean(errs)), float(np.mean(ctrl)), float(np.mean(pers))


def test_a_frame_only_head_cannot_do_this_at_all():
    """The control that says the question is real: exactly a centre guess, because the frame is
    genuinely ambiguous and the best a memoryless map can do is split the difference."""
    err, centre, _pers = drive("frame", free_running=True)
    assert 0.85 < centre / err < 1.2, (
        f"a frame-only head reads {centre / err:.2f}x a centre guess. Above 1.2 the fixture has "
        "stopped being ambiguous; below 0.85 something else is wrong")


def test_the_output_bank_is_the_only_thing_that_works_here():
    """Measured, free-running, against a centre guess and against copying the previous TRUE
    position -- the second control is the one that matters, because persistence tracks identity
    through a smooth crossing almost perfectly:

    | basis | x centre | x persistence |
    |---|---|---|
    | frame only | 1.00x | 0.04x |
    | + one lag | 12.8x | **0.57x** |
    | + three lags | 71.0x | 3.18x |
    | + the bank | **141.7x** | **6.36x** |

    One lag free-running is **worse than persistence**: its own errors compound. The
    multi-timescale part is what makes this pay, which is the half of the proposal that could
    have been dropped as decoration.
    """
    one = drive("lag", free_running=True, lags=1)
    three = drive("lag", free_running=True, lags=3)
    bank = drive("bank", free_running=True)

    assert one[2] / one[0] < 1.0, (
        f"one lag now beats persistence ({one[2] / one[0]:.2f}x); if that is real the "
        "multi-timescale argument is weaker than this file claims")
    assert three[2] / three[0] > 1.5, f"three lags: {three[2] / three[0]:.2f}x persistence"
    assert bank[2] / bank[0] > 3.0, f"the bank: {bank[2] / bank[0]:.2f}x persistence"
    assert bank[1] / bank[0] > 50, f"the bank: {bank[1] / bank[0]:.2f}x a centre guess"


def test_teacher_forcing_is_the_ceiling_and_not_the_shippable_number():
    """Training on the true previous output and running on the predicted one is a different
    distribution. Teacher forcing reads 1572x a centre guess against free-running's 141.7x -- an
    order of magnitude apart, and only one of them describes a system anyone can deploy."""
    forced = drive("lag", free_running=False, lags=1)
    free = drive("lag", free_running=True, lags=1)
    assert forced[1] / forced[0] > 10 * (free[1] / free[0]), (
        f"forced {forced[1] / forced[0]:.0f}x against free {free[1] / free[0]:.0f}x; if the gap "
        "has closed, exposure bias has stopped mattering and that is worth knowing")


def test_it_cannot_recover_from_a_dropout_and_that_is_the_cost():
    """The failure that decides how this must be used.

    Sixty blind frames take the error from 0.0009 to 0.70, and nine hundred frames later it is
    still **0.46**. It never comes back. This is not divergence -- the prediction stays bounded --
    it is a **stable wrong fixed point**, and it follows from the same property that made feedback
    necessary: a frame that cannot disambiguate cannot correct either.

    **The guard cannot catch it**, and that is the point rather than an omission: the head is
    healthy and its residual is low, because it fits the wrong assignment well. Same shape as the
    two-rate finding already recorded -- being asked the wrong question is invisible from inside.
    What re-anchors it is appearance, when the objects differ.
    """
    clean, _c, _p = drive("bank", free_running=True, n=3000)
    broken, _c2, _p2 = drive("bank", free_running=True, n=3000, blind=(2000, 2060))
    assert broken > clean * 20, (
        f"a 60-frame blackout cost only {broken / clean:.1f}x ({clean:.4f} -> {broken:.4f}); if "
        "it now recovers, the stable-wrong-fixed-point claim is wrong and must be rewritten")

    # And it is not divergence: the head stays bounded while being confidently wrong.
    assert broken < 1.0, f"error {broken:.3f} is larger than the frame; that would be divergence"


def test_more_time_constants_do_not_help():
    """Four rates behave identically to three, so the recovery failure is not a tuning problem.
    Recorded because adding another band is the obvious thing to try next."""
    three = drive("bank", free_running=True, rates=(0.5, 0.15, 0.03), blind=(2000, 2060))[0]
    four = drive("bank", free_running=True, rates=(0.9, 0.5, 0.15, 0.03), blind=(2000, 2060))[0]
    assert abs(four - three) < 0.05 * max(three, 1e-9), (
        f"four rates now differ from three ({four:.4f} vs {three:.4f}); if a band helps, the "
        "recovery failure may be tunable after all")


def test_the_bank_is_sized_and_validated():
    """`build` sizes the basis and `main` constructs the memory, so the two have to agree without
    one owning the other -- `feedback_width` is that agreement."""
    m = OutputMemory(4)
    assert m.width == feedback_width(2) == len(FEEDBACK_RATES) * 4
    assert np.array_equal(m.features(), np.zeros(m.width))

    m.observe([1.0, 1.0, 1.0, 1.0])
    got = m.features()
    assert np.allclose(got[:4], 0.5) and np.allclose(got[8:], 0.03), got
    m.reset()
    assert np.array_equal(m.features(), np.zeros(m.width))

    with pytest.raises(ValueError, match="expected 4 outputs"):
        m.observe([1.0, 2.0])
    with pytest.raises(ValueError, match="at least 1"):
        OutputMemory(0)
    with pytest.raises(ValueError, match="rate"):
        OutputMemory(2, rates=(0.0,))
    with pytest.raises(ValueError, match="rate"):
        OutputMemory(2, rates=())


def test_the_composition_grows_by_exactly_the_bank():
    """And the persistence fingerprint has to change with it: a block trained with feedback means
    something different from one trained without, at the same length."""
    from demo.pointer import fingerprint

    plain = build(2, "raw")
    fed = build(2, "raw", feedback=True)
    assert fed[1].n_features == plain[1].n_features + feedback_width(2)
    assert fed[3].dim == plain[3].dim + feedback_width(2)
    assert len(fed[5].features) == len(plain[5].features) + feedback_width(2)
    assert fingerprint(plain[0], 2, "raw") != fingerprint(fed[0], 2, "raw-fb")

def test_the_basis_keeps_its_width_across_calibration():
    """The bug this file did not catch, and the reason it did not.

    `main()` calibrates on frame 16 and **recomputes the basis** afterwards, so anything
    concatenated outside the extraction path is silently dropped on that one frame. With
    `--feedback` that produced `expected shape (162,), got (144,)` on the first calibrated frame --
    and it is the same defect the extraction path was introduced to fix, arriving a second time by
    the same door.

    Every test here drove `build()`, which sizes the streams, and none drove the per-frame
    assembly, which is where the widths actually have to agree. This drives the assembly the way
    the loop does, calibration included.
    """
    objects = 3
    _bb, _head, _guard, features, _taught, fmap, _pipe = build(objects, "raw", feedback=True)
    mem = OutputMemory(2 * objects)
    want = raw_width() + feedback_width(objects)
    assert features.dim == want == len(fmap.features)

    scale = 1.0
    backbone = _bb

    def extract(img):
        """The APP's own function, not a local copy of it. Two deliberate bugs in the real one
        survived the first version of this test, which reimplemented the concatenation here."""
        return basis(img, backbone=backbone, source="raw", scale=scale, output_memory=mem)

    t = 0.0
    for k in range(40):
        t += 1.0 / 30.0
        img = frame_at(0.3 + 0.005 * k, 0.5, seed=k)
        emb = extract(img)
        assert emb.shape == (want,), f"frame {k}: basis is {emb.shape}, stream wants {want}"
        if k == 16:
            # Calibrate from the FRAME features only -- the bank starts at zero and would drag
            # the standard deviation down on this one frame.
            scale = 1.0 / max(float(np.std(emb[:frame_width(backbone, "raw")])), 1e-6)
            emb = extract(img)
            assert emb.shape == (want,), "the basis lost the bank across calibration"
        features.push(t, emb)          # the push is what raised in the running app
        mem.observe(np.full(2 * objects, 0.5))

    # And the VALUE has to be checked, not merely its existence. Two mutants survived a version
    # of this test that only asserted `scale != 1.0`: one that scaled from the zero-padded whole,
    # and one that never applied the scale at all. What calibration is *for* is that the frame
    # features come out at unit scale, so that is the assertion.
    assert scale != 1.0, "calibration never ran"
    n_frame = frame_width(backbone, "raw")
    assert n_frame == raw_width(), (n_frame, raw_width())
    final = extract(frame_at(0.3, 0.5, seed=99))
    assert float(np.std(final[:n_frame])) == pytest.approx(1.0, rel=0.35), (
        f"after calibration the frame features have std {np.std(final[:n_frame]):.3f}, not ~1. "
        "Either the scale was derived from the zero-padded whole or it is not being applied")
    # The bank is NOT scaled -- it is already O(1) by construction, and scaling it would make the
    # head's memory terms depend on the frame's contrast.
    assert float(np.max(np.abs(final[n_frame:]))) <= 1.0 + 1e-6


def test_feedback_runs_through_the_whole_composition_headlessly():
    """End to end with the pipeline and guard, which is what `main()` actually does. Without this
    the only check on the feedback wiring was a shape assertion in `build()`."""
    objects = 2
    backbone, head, guard, features, taught, _fmap, pipe = build(objects, "raw", feedback=True)
    mem = OutputMemory(2 * objects)
    base_std = float(np.std(frame_at(0.5, 0.5, seed=0)))
    scale = 1.0 / max(float(np.std([raw_features(frame_at(0.5, 0.5, seed=i))
                                    for i in range(8)])), 1e-9)
    g = np.random.default_rng(3)
    t, errs, updated = 0.0, [], 0
    for k in range(600):
        t += 1.0 / 30.0
        xs = [(float(g.uniform(0.2, 0.8)), float(g.uniform(0.2, 0.8))) for _ in range(objects)]
        xs.sort(key=lambda p: p[0])
        img = frame_at(xs[0][0], xs[0][1], seed=k)
        guard.set_external(0, float(np.std(img)) / max(base_std, 1e-6))
        features.push(t, np.concatenate([raw_features(img) * scale,
                                         mem.features().astype(np.float32)]))
        taught.push(t, [c for p in xs for c in p])
        report = pipe.step()
        if report.updated:
            updated += 1
        if report.prediction is not None:
            pred = tuple(float(v) for v in report.prediction)
            mem.observe(pred)                       # free-running, as the app does
            if k > 300:
                errs.append(np.hypot(pred[0] - xs[0][0], pred[1] - xs[0][1]))
    assert updated > 400, f"only {updated} updates; the pipeline is not learning"
    assert errs and np.all(np.isfinite(errs)), "the feedback loop produced non-finite output"
    assert float(np.mean(errs)) < 0.35, (
        f"the feedback composition reached {np.mean(errs):.3f}; it should at least stay bounded "
        "and roughly track the leftmost object")
    assert np.all(np.isfinite(np.asarray(head.state))), "the head went non-finite under feedback"
