# SPDX-License-Identifier: Apache-2.0
"""Labels out of the input rather than out of the mouse, and the controls that scope the claim.

The claim is narrow, and the controls are what make it worth anything: a learned head covering an
intermittent label source beats holding that source's last output — but only when the source's
blindness is independent of the target's motion, and never when the source can be afforded on
every frame. All three are asserted here, including the two that are unflattering.
"""

from __future__ import annotations

import numpy as np
import pytest

from demo.label_sources import (
    CadenceDetector,
    MotionLabeller,
    _centroid,
    _grey,
    brightest_window,
)
from demo.pointer import build
from demo.test_pointer import frame_at


def wander(k):
    """Always moving, so a cadence-blind detector is blind while the estimand changes."""
    a = k * 0.09
    return 0.5 + 0.26 * np.sin(a), 0.5 + 0.22 * np.cos(a * 1.3)


@pytest.fixture
def ready():
    parts = build()
    backbone = parts[0]
    sample = np.array([backbone.features(frame_at(0.5, 0.5, seed=i)) for i in range(8)])
    backbone.output_scale = 1.0 / max(float(sample.std()), 1e-9)
    return parts, float(np.std(frame_at(0.5, 0.5, seed=0)))


def drive(ready, source, steps=660):
    """Drive the composition with labels from `source` instead of from a mouse."""
    parts, frame_std = ready
    backbone, _head, guard, features, taught, _fmap, pipe = parts
    t, last, rows = 0.0, None, []
    for k in range(steps):
        t += 1.0 / 30.0
        tx, ty = wander(k)
        img = frame_at(tx, ty, seed=k)
        guard.set_external(0, float(np.std(img)) / max(frame_std, 1e-6))
        features.push(t, backbone.features(img))
        got = as_points(source(img))
        if got is not None:
            last = got[0]
            taught.push(t, [c for xy in got for c in xy])
        r = pipe.step()
        # Read now: `StepReport.prediction` is a preallocated buffer the next step overwrites.
        pred = None if r.prediction is None else (float(r.prediction[0]),
                                                 float(r.prediction[1]))
        rows.append((pred, last, (tx, ty), got is not None))
    return rows


def as_points(got):
    """`CadenceDetector` returns a list of points; `MotionLabeller` returns one point. Normalise
    here rather than letting either grow a return type that varies with its arguments."""
    if got is None:
        return None
    return list(got) if isinstance(got[0], tuple) else [tuple(got)]


def radial(rows, pick):
    d = [np.hypot(v[0] - tr[0], v[1] - tr[1])
         for p, l, tr, _had in rows for v in (pick(p, l),) if v is not None]
    assert d, "nothing to score"
    return float(np.mean(d))


def test_a_cadence_blind_label_source_earns_the_learned_head(ready):
    """The claim, scored only on the frames where the source is blind — the ones a caller with no
    learned model would have to guess.

    Measured 1.26x at every=5, 1.31x at 10, 1.41x at 15. The bound is loose because the ratio is
    a property of the cadence; the assertion that matters is the direction.
    """
    rows = drive(ready, CadenceDetector(10))
    warm = rows[len(rows) // 2:]
    blind = [r for r in warm if not r[3]]
    assert len(blind) > 0.8 * len(warm), "the source was not mostly blind, so nothing is covered"

    head = radial(blind, lambda p, _l: p)
    hold = radial(blind, lambda _p, l: l)
    centre = radial(blind, lambda _p, _l: (0.5, 0.5))
    assert hold > head * 1.15, (
        f"the head ({head:.3f}) no longer beats holding the last detection ({hold:.3f}); "
        "without that it is covering nothing and the arrangement is decoration")
    assert head < centre * 0.6, f"head {head:.3f} against a centre-guess {centre:.3f}"


def test_a_motion_blind_source_does_not_earn_it_and_that_is_the_finding(ready):
    """The negative control, and it is structural rather than a tuning failure.

    Motion energy is blind exactly when the target is not moving, so it cannot be blind while the
    estimand changes, and holding its last output is right whenever it is unavailable. Measured
    0.87x — the head loses. **An aiding sensor earns a learned model only if it can be blind
    while the quantity changes**, and this is the case that establishes the rule.

    The first fixture for this was rigged, and the label count gave it away: the blob teleported
    into its parked position, a teleport is a motion event, so the detector got one free reading
    at exactly the right place.
    """
    lab = MotionLabeller()
    lab.calibrate([frame_at(0.5, 0.5, seed=100 + i) for i in range(12)])
    rows = drive(ready, lab)
    warm = rows[len(rows) // 2:]
    blind = [r for r in warm if not r[3]]
    assert blind, "the motion source was never blind, so the control is vacuous"

    head = radial(blind, lambda p, _l: p)
    hold = radial(blind, lambda _p, l: l)
    assert hold < head, (
        f"holding ({hold:.3f}) no longer beats the head ({head:.3f}) on a motion-blind source. "
        "If that has genuinely changed, re-measure and rewrite demo/label_sources.py — but check "
        "the fixture first: a target that stops moving where it already was makes holding "
        "correct by construction")


def test_a_source_you_can_run_every_frame_beats_the_head(ready):
    """The unflattering control, and it is what scopes the whole approach.

    The head is not more accurate than its own teacher — it is an interpolator between the
    teacher's runs. Measured, the detector on every frame scores 0.010 against the head's ~0.08.
    So automatic labelling buys "no hand on the mouse", and the learned head buys accuracy only
    where the teacher cannot run at the output rate.
    """
    direct = []
    for k in range(200):
        tx, ty = wander(k)
        x, y = brightest_window(frame_at(tx, ty, seed=k))
        direct.append(np.hypot(x - tx, y - ty))
    every_frame = float(np.mean(direct))
    assert every_frame < 0.05, every_frame

    rows = drive(ready, CadenceDetector(10, locate=lambda f: [brightest_window(f)]))
    warm = rows[len(rows) // 2:]
    head = radial([r for r in warm if not r[3]], lambda p, _l: p)
    assert every_frame < head, (
        f"an every-frame detector ({every_frame:.3f}) no longer beats the head ({head:.3f}); if "
        "that is real then the head is doing more than interpolating and the scope stated in "
        "demo/label_sources.py is understated")


def test_an_accumulated_motion_map_labels_the_interval_not_the_instant():
    """The design decision inside `CadenceDetector`, measured with no learning involved.

    The first version accumulated the frame difference between runs. That map is about the
    *interval*, so its centroid is the middle of the arc the target travelled and its error grows
    with the window: 0.048 / 0.074 / 0.121 / 0.160 at every = 2 / 5 / 10 / 15, against a flat
    ~0.036 for a single-step locator. Stamping the accumulated label at the interval's midpoint
    does **not** fix it — measured identical to three decimals, because with a fast reference
    `hold_last` pairs it with the current features either way. It is the label that is wrong
    about every instant, not the alignment, and that distinction is worth keeping: this project's
    recorded two-rate lesson is about stamping, and this is the case where stamping is not it.
    """
    def label_error(accumulate, every, n=400):
        prev = acc = None
        out = []
        for k in range(n):
            tx, ty = wander(k)
            g = _grey(frame_at(tx, ty, seed=k))
            if prev is None:
                prev, acc = g, np.zeros_like(g)
                continue
            step = np.abs(g - prev)
            acc = acc + step
            prev = g
            if k % every:
                continue
            got = _centroid(acc if accumulate else step, 0.35)
            acc = np.zeros_like(g)
            if got is not None:
                out.append(np.hypot(got[0] - tx, got[1] - ty))
        return float(np.mean(out))

    acc_small, acc_big = label_error(True, 2), label_error(True, 15)
    step_small, step_big = label_error(False, 2), label_error(False, 15)
    assert acc_big > acc_small * 2.0, (
        f"the accumulated label no longer smears with the window ({acc_small:.3f} -> "
        f"{acc_big:.3f}), so the reason for the single-step default is not demonstrated")
    assert step_big < step_small * 1.5, (
        f"the single-step label is no longer flat in the window ({step_small:.3f} -> "
        f"{step_big:.3f})")
    assert step_big < acc_big * 0.5


def test_the_detector_is_blind_by_cadence_and_nothing_else():
    """Availability is a declared interval, not a property of the scene — which is the whole
    reason it earns a learned head. Asserted directly, because two tests above depend on it."""
    det = CadenceDetector(4)
    ran = [det(frame_at(*wander(k), seed=k)) is not None for k in range(21)]
    assert ran[0] is False, "the first frame has no predecessor and cannot difference"
    assert sum(ran) == 5, ran
    assert det.runs == 5
    assert det.last is not None
    det.reset()
    assert det.runs == 0 and det.last is None
    with pytest.raises(ValueError, match="at least 1"):
        CadenceDetector(0)


def test_a_still_frame_makes_the_motion_source_declare_itself_blind():
    """`None`, not an invented position. A caller pushes nothing on a None, which is what makes
    it an intermittent sensor rather than one quietly reporting the middle of the frame."""
    lab = MotionLabeller()
    still = [frame_at(0.5, 0.5, seed=200) for _ in range(6)]     # identical frames
    lab.calibrate(still)
    assert lab(still[0]) is None                                  # no predecessor
    assert all(lab(f) is None for f in still[1:])
    assert lab.energy == 0.0
    assert lab(frame_at(0.8, 0.2, seed=201)) is not None
    assert lab.energy > lab.min_energy


def test_the_centroid_floor_keeps_noise_from_dragging_it_to_the_middle():
    """Without the floor, sensor noise spread over the whole frame biases the centroid towards
    the centre — which is a bias towards exactly the answer the control predictor gives, i.e.
    towards the one thing this has to be able to beat."""
    h, w = 40, 60
    m = np.full((h, w), 0.02, dtype=np.float32)      # a noise floor everywhere
    m[8:12, 48:52] = 1.0                            # a real signal, well off centre
    floored = _centroid(m, 0.35)
    flat = _centroid(m, 0.0)
    truth = (50 / (w - 1), 10 / (h - 1))
    d_floored = np.hypot(floored[0] - truth[0], floored[1] - truth[1])
    d_flat = np.hypot(flat[0] - truth[0], flat[1] - truth[1])
    assert d_floored < 0.02, d_floored
    assert d_flat > d_floored * 5, (d_flat, d_floored)
    assert _centroid(np.zeros((4, 4)), 0.35) is None


def frame_with_a_lamp(x, y, seed=0):
    """A moving blob plus a BRIGHTER static one — a lamp or a window, which is what a real
    camera has in it. This is the scene that separates a motion locator from a brightness one.
    """
    img = frame_at(x, y, seed=seed)
    h, w = img.shape[0], img.shape[1]
    yy, xx = np.mgrid[0:h, 0:w]
    lx, ly = 0.15 * w, 0.85 * h
    return img + np.exp(-(((xx - lx) ** 2 + (yy - ly) ** 2) / (2 * (0.10 * w) ** 2)))[:, :, None] * 1.4


def test_the_default_locator_follows_the_mover_and_not_the_brightest_thing():
    """Why the default reads the frame DIFFERENCE and not the frame.

    A mutant locating on the raw frame instead of the difference survived every other test here,
    and that was a fixture limitation rather than a missing assertion: on a synthetic frame the
    moving blob *is* the brightest thing, so the two agree. A real camera has a window in it. With
    a brighter static distractor the two separate completely, which is also the reason
    `brightest_window` is offered but is not the default.
    """
    det_motion = CadenceDetector(3)
    det_bright = CadenceDetector(3, locate=lambda f: [brightest_window(f)])
    err_motion, err_bright = [], []
    for k in range(30):
        tx, ty = wander(k)
        img = frame_with_a_lamp(tx, ty, seed=k)
        got_m, got_b = as_points(det_motion(img)), as_points(det_bright(img))
        if got_m is not None:
            err_motion.append(np.hypot(got_m[0][0] - tx, got_m[0][1] - ty))
        if got_b is not None:
            err_bright.append(np.hypot(got_b[0][0] - tx, got_b[0][1] - ty))
    m, b = float(np.mean(err_motion)), float(np.mean(err_bright))
    # Measured 0.114 against 0.679 with the shipped default. The bound moved from 0.10 when the
    # default became `find_peaks`, which refines inside a suppression window rather than taking
    # a floored global centroid -- re-measured rather than loosened by feel.
    assert m < 0.15, f"the motion locator lost the mover: {m:.3f}"
    assert b > m * 4.0, f"only {b / m:.1f}x apart, so the fixture barely separates them"
    assert b > 0.4, (
        f"the brightness locator found the mover anyway ({b:.3f}), so this fixture no longer "
        "separates the two and the default's justification is untested")

    # And the raw frame is not a usable weight map here, which is the surviving mutant made
    # explicit. The bound is against a centre guess rather than a magnitude: the frame has a
    # bright background, so above a 0.35 floor much of it passes and the centroid lands between
    # the mover, the lamp and the wall -- measured 0.24, which is a centre guess with extra
    # steps, not a detector.
    on_frame, centre = [], []
    for k in range(30):
        tx, ty = wander(k)
        got = _centroid(_grey(frame_with_a_lamp(tx, ty, seed=k)), 0.35)
        on_frame.append(np.hypot(got[0] - tx, got[1] - ty))
        centre.append(np.hypot(0.5 - tx, 0.5 - ty))
    assert float(np.mean(on_frame)) > float(np.mean(centre)) * 0.8, (
        f"locating on the raw frame ({np.mean(on_frame):.3f}) is now meaningfully better than "
        f"a centre guess ({np.mean(centre):.3f}); the mutant it stands for would survive")
    assert float(np.mean(on_frame)) > m * 2.0


def test_the_detector_sorts_its_peaks_and_refuses_a_short_list():
    """Both of these live in `CadenceDetector`, and both survived a mutation run because
    `test_multi_object.py` calls `find_peaks` directly and does its own sorting -- so the
    component that actually ships was bypassed.

    The sort is what breaks the head's permutation symmetry (measured: sorted labels 0.129
    against arbitrary 0.256, on a centre guess of 0.312), and refusing a short list is what
    stops slots being silently reassigned when the scene loses a mover.
    """
    from demo.test_multi_object import scene, two_movers

    det = CadenceDetector(2, k=2)
    # The non-None returns, not the last one: most calls are blind by cadence, and taking the
    # final return read None for that reason alone.
    seen = [v for v in (det(scene(two_movers(k), seed=k)) for k in range(8)) if v is not None]
    assert seen, "no cadence instant produced a pair"
    for got in seen:
        assert len(got) == 2
        assert got[0][0] < got[1][0], f"slot 0 is not the leftmost: {got}"

    # Reversed input to `locate` must still come out leftmost-first: the rule is the detector's,
    # not the locator's.
    right, left = (0.8, 0.3), (0.2, 0.7)
    det2 = CadenceDetector(1, k=2, locate=lambda _f: [right, left])
    det2(scene([(0.5, 0.5)], seed=1))                 # first frame: no predecessor
    out = det2(scene([(0.5, 0.5)], seed=2))
    assert out == [left, right], out

    # One mover, two slots asked for: None rather than a list of one.
    #
    # `suppress=0.45` is needed and the reason is a measured property of frame differencing, not
    # a tuning preference: a moving object leaves a DIPOLE -- |current - previous| has a lobe
    # where it was and one where it is -- so at the default 0.18 a single mover legitimately
    # yields two peaks. See `test_one_mover_leaves_two_difference_lobes`.
    lone = CadenceDetector(2, k=2, suppress=0.45)
    alone = [lone(scene([(0.3 + 0.05 * k, 0.5)], seed=k)) for k in range(8)]
    assert all(v is None for v in alone), alone
    # ...and the control: the same detector asking for one slot does find it, so the None above
    # is the shortfall and not a broken fixture.
    solo = CadenceDetector(2, k=1, suppress=0.45)
    assert any(solo(scene([(0.3 + 0.05 * k, 0.5)], seed=k)) for k in range(8))


def test_one_mover_leaves_two_difference_lobes():
    """Why `k > 1` cannot be used to count, measured.

    `|current - previous|` has a lobe where the object was and one where it is, separated by
    roughly the object's own size. Asking for three peaks on a single blob of sigma 8.6 px:

    | displacement | 0.5 px | 1.9 px | 4.8 px | 9.6 px | 19.2 px |
    |---|---|---|---|---|---|
    | `suppress=0.18` | 3 | 3 | 2 | 2 | 2 |
    | `suppress=0.45` | 3 | 1 | 1 | 1 | 1 |

    The left column is a different failure: at 0.5 px the difference is noise, and a floor
    relative to the *peak* cannot catch that because the peak is noise too. `min_energy` is for
    that half, and it is off by default because the separation is only 1.35x.

    So the shipped claim is scoped: `k > 1` separates objects that are far apart, and it cannot
    count. The two-object accuracy result is for movers half a frame apart.
    """
    from demo.label_sources import find_peaks
    from demo.test_multi_object import scene

    def peaks_for(dx, suppress):
        a = _grey(scene([(0.40, 0.50)], seed=1))
        b = _grey(scene([(0.40 + dx, 0.50)], seed=2))
        return len(find_peaks(np.abs(b - a), 3, suppress=suppress))

    assert peaks_for(0.05, 0.18) == 2, "the dipole no longer appears at the default radius"
    assert peaks_for(0.05, 0.45) == 1, "a wide radius no longer merges the dipole"
    # And the noise-dominated end, which suppression cannot fix at any radius.
    assert peaks_for(0.005, 0.45) == 3
    assert peaks_for(0.005, 0.18) == 3

def test_the_energy_floor_refuses_a_still_scene_and_its_margin_is_tight():
    """`min_energy` is the half of the noise problem that suppression cannot fix, and its default
    is **off** because the separation is small enough to be dangerous.

    Measured mean absolute frame difference: a still scene reads 0.0129-0.0131 and a fixture whose
    blobs move ~2 px per frame reads 0.0177-0.0190 -- **1.35x**. So a margin of 2.0 over the still
    quantile refuses real motion outright, which is how this was found rather than reasoned about.

    A peak-to-median ratio was measured as the alternative, on the theory that noise is uniform
    while a mover is localised. It is **worse** -- 0.97x, the distributions overlap, because a
    second mover raises the map's own median. Asserted here so it is not tried again.
    """
    from demo.test_multi_object import scene, two_movers

    still = [scene([(0.4, 0.5)], seed=100 + i) for i in range(12)]
    moving = [scene(two_movers(k), seed=k) for k in range(12)]

    def energies(frames):
        prev, out = None, []
        for f in frames:
            g = _grey(f)
            if prev is not None:
                out.append(float(np.abs(g - prev).mean()))
            prev = g
        return np.array(out)

    e_still, e_move = energies(still), energies(moving)
    ratio = float(e_move.min() / e_still.max())
    assert 1.1 < ratio < 1.8, (
        f"still-to-moving separation is now {ratio:.2f}x, not the ~1.35x this file is written "
        "around; the margin defaults below were chosen against that number")

    # A calibrated floor refuses the still scene...
    det = CadenceDetector(1, k=1)
    thr = det.calibrate(still)
    assert e_still.max() < thr < e_move.min(), (thr, e_still.max(), e_move.min())
    assert all(det(f) is None for f in still), "a still scene produced a label"
    # ...and does not refuse the moving one, which is the half a bigger margin would break.
    det.reset()
    assert any(det(f) is not None for f in moving), "the calibrated floor refused real motion"

    # The finding: the margin that looks safe is not. 2.0 refuses everything.
    greedy = CadenceDetector(1, k=1)
    greedy.calibrate(still, margin=2.0)
    greedy.reset()
    assert all(greedy(f) is None for f in moving), (
        "a margin of 2.0 no longer refuses real motion, so the reason the default is 1.2 -- and "
        "the reason min_energy ships disabled -- is no longer demonstrated")

    # And the peak-to-median alternative, refuted rather than described.
    def peak_over_median(frames):
        prev, out = None, []
        for f in frames:
            g = _grey(f)
            if prev is not None:
                d = np.abs(g - prev)
                out.append(float(d.max() / max(float(np.median(d)), 1e-9)))
            prev = g
        return np.array(out)

    p_still, p_move = peak_over_median(still), peak_over_median(moving)
    assert p_move.min() < p_still.max(), (
        f"peak/median now separates ({p_move.min():.2f} vs {p_still.max():.2f}); if that is real "
        "it is a better gate than mean energy and label_sources.py should say so")


def test_the_detector_leaves_the_energy_floor_off_by_default():
    """A gate whose safe margin is 1.35x must not be on without the caller measuring it."""
    assert CadenceDetector(1).min_energy == 0.0
    assert CadenceDetector(1, min_energy=0.05).min_energy == 0.05
