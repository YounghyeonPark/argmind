# SPDX-License-Identifier: Apache-2.0
"""Tests for the pipeline: streams to frames to an adapted model.

The scenario throughout is the design driver reduced to its smallest honest form:
a sensor whose reading needs calibrating, a temperature that its bias depends on,
and a slow truth source. Frames happen when truth arrives — which is the point of
having an aligner, since the other two streams run faster and are held.

As elsewhere here, the tests that matter most are the ones that can fail:
staleness that must block adaptation, a sensor that flags itself invalid, a
non-causal policy that must not be usable by accident, and a frame map that must
refuse to build a regressor that does not match the adapter.
"""

import sys
import numpy as np
import pytest

from argmind import (Aligner, Bound, FrameMap, Guard, Pipeline,
                     RecursiveLeastSquares, Stream)

TRUE = (2.0, -0.5, 0.25)          # ref = 2.0*raw - 0.5*temp + 0.25


def build(k=200, dtype="f8", seed=61, offset=TRUE[2], invalid_from=None,
          temp_stops_at=None, capacity=None):
    """Streams for a calibration problem, wired so the answer is exact.

    Each frame is built by pushing `raw` and `temp` just *before* the truth
    sample, then computing truth from exactly those values. Without that the
    held-last values lag the target by a fraction of a period and the test would
    be measuring alignment error rather than the pipeline.

    Capacity defaults to more than `k` so that these tests measure the pipeline
    rather than ring overflow — overflow gets its own test below, and it found a
    real gap when it was still accidental.
    """
    cap = capacity or (k + 8)
    raw = Stream("raw", dim=1, capacity=cap, rate_hz=100, dtype=dtype)
    temp = Stream("temp", dim=1, capacity=cap, rate_hz=50, dtype=dtype)
    ref = Stream("ref", dim=1, capacity=cap, rate_hz=5, dtype=dtype)
    rng = np.random.default_rng(seed)

    for i in range(k):
        t = 0.2 * (i + 1)
        r = float(rng.standard_normal())
        c = float(20.0 + 5.0 * rng.standard_normal())
        raw.push(t - 0.02, [r])
        if temp_stops_at is None or i < temp_stops_at:
            temp.push(t - 0.01, [c])
        else:
            c = float(temp.latest().values[0])       # what hold_last will supply
        y = TRUE[0] * r + TRUE[1] * c + offset
        ref.push(t, [y], valid=(invalid_from is None or i < invalid_from))
    return [raw, temp, ref]


def make_pipeline(streams, constant=True, n_features=3, adapter=None,
                  **kwargs):
    ad = adapter or RecursiveLeastSquares(n_features=n_features, forgetting=1.0,
                                          p0=1e6, dtype="f8")
    return Pipeline(
        streams=streams,
        aligner=Aligner.hold_last(reference="ref"),
        adapter=ad,
        frame_map=FrameMap(features=["raw", "temp"], target="ref",
                           constant=constant),
        **kwargs)


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------

def test_it_learns_the_calibration_from_multi_rate_streams():
    pipe = make_pipeline(build())
    report = pipe.step()

    assert report.frames == 200
    assert report.updated == 200, f"frames were dropped: {report.skipped}"
    assert report.healthy
    assert not report.idle
    theta = pipe.adapter.theta.ravel()
    assert np.allclose(theta, TRUE, atol=1e-6), (
        f"learned {theta} instead of {TRUE}")


def design_matrix(streams, constant):
    """Rebuild ``(X, Y)`` from the streams with an independent aligner.

    An external oracle for the regressor the pipeline constructs. Checking the
    learned parameters against `TRUE` only says the fit is good; checking them
    against the least-squares solution of *this* design matrix says the pipeline
    built the regressor it claimed to — which is a different question, and the
    one a mutation that quietly overwrote the last feature slipped through.
    """
    frames = Aligner.hold_last("ref").drain({s.name: s for s in streams})
    X = np.array([[f.values["raw"][0], f.values["temp"][0]]
                  + ([1.0] if constant else []) for f in frames], dtype=float)
    Y = np.array([[f.values["ref"][0]] for f in frames], dtype=float)
    return X, Y


@pytest.mark.parametrize("constant,n", [(True, 3), (False, 2)])
def test_the_pipeline_builds_the_regressor_it_says_it_does(constant, n):
    streams = build()
    pipe = make_pipeline(streams, constant=constant, n_features=n)
    pipe.step()

    X, Y = design_matrix(streams, constant)
    expect = np.linalg.lstsq(X, Y, rcond=None)[0].ravel()
    assert np.allclose(pipe.adapter.theta.ravel(), expect, atol=1e-6), (
        f"learned {pipe.adapter.theta.ravel()}, least squares on the declared "
        f"design matrix gives {expect}")


def test_the_constant_term_is_what_learns_the_offset():
    """Without it the model cannot express a bias, which is the entire problem
    this library's driving example is about. The negative half is the evidence."""
    with_const = make_pipeline(build())
    with_const.step()
    assert np.allclose(with_const.adapter.theta.ravel(), TRUE, atol=1e-6)

    without = make_pipeline(build(), constant=False, n_features=2)
    without.step()
    learned = without.adapter.theta.ravel()
    assert not np.allclose(learned[:2], TRUE[:2], atol=1e-3), (
        f"a model with no constant term fitted an offset anyway ({learned}); "
        "either the offset is not in the data or the constant is being added "
        "when it was not asked for")


def test_step_is_idempotent_when_no_new_samples_have_arrived():
    streams = build(k=20)
    pipe = make_pipeline(streams)
    first = pipe.step()
    assert first.frames == 20

    second = pipe.step()
    assert second.idle
    assert second.frames == 0
    assert second.updated == 0
    assert pipe.stats["frames"] == 20, "frames were re-consumed"
    assert pipe.stats["steps"] == 2


def test_frames_lost_to_a_full_ring_are_reported_not_silently_learned_around():
    """Found by a test that was accidentally configured this way, which is why
    `StepReport.lost_frames` exists at all.

    Before it did, a 200-frame workload against a 64-deep ring adapted 64 times,
    returned a report with nothing amiss in it, and produced a perfectly
    plausible fit from a third of the data. Bounded buffers are the design;
    losing data without saying so is not.
    """
    pipe = make_pipeline(build(k=200, capacity=64))
    r = pipe.step()

    assert r.frames == 64, "the ring did not overflow, so this proves nothing"
    assert r.lost_frames == 136, f"lost samples were not reported: {r}"
    assert r.healthy is False, "a pipeline that lost 136 frames reported healthy"
    assert pipe.stats["lost_frames"] == 136

    # Counted once, not again on every subsequent step.
    assert pipe.step().lost_frames == 0


def test_a_full_ring_in_steady_state_is_not_a_loss():
    """The counterpart, and the reason `lost_frames` is not simply
    `Stream.dropped`.

    A correctly sized ring overwrites on every push once it is full; that is what
    a ring is for. Reading those overwrites as lost data raises an alarm on every
    healthy system, which is the fastest way to get an alarm ignored. Here the
    ring is deliberately far smaller than the workload and nothing is lost,
    because the pipeline keeps up.
    """
    streams = build(k=0, capacity=16)
    pipe = make_pipeline(streams)
    raw, temp, ref = streams
    rng = np.random.default_rng(5)
    total = 0
    for i in range(200):
        t = 0.2 * (i + 1)
        r, c = float(rng.standard_normal()), float(20.0 + rng.standard_normal())
        raw.push(t - 0.02, [r])
        temp.push(t - 0.01, [c])
        ref.push(t, [TRUE[0] * r + TRUE[1] * c + TRUE[2]])
        rep = pipe.step()
        assert rep.lost_frames == 0, (
            f"reported a loss at i={i} with capacity 16 while keeping up")
        total += rep.updated
    assert total == 200
    assert ref.dropped > 150, (
        "the ring never overflowed, so this does not demonstrate the distinction")
    assert np.allclose(pipe.adapter.theta.ravel(), TRUE, atol=1e-6)


def test_frames_arriving_and_none_being_used_is_not_the_same_as_idle():
    """`idle` means nothing was available. A step that produced frames and
    adapted to none of them is a fault, and conflating the two hides it."""
    streams = build(k=30, invalid_from=0)
    pipe = make_pipeline(streams)
    r = pipe.step()
    assert r.frames == 30
    assert r.updated == 0
    assert r.idle is False
    assert r.skipped == {"invalid": 30}


# ---------------------------------------------------------------------------
# The input gate: staleness and validity, before the learner sees anything
# ---------------------------------------------------------------------------

def test_a_stream_that_dies_stops_adaptation_instead_of_poisoning_it():
    """hold_last will contribute a dead sensor's final value forever. The gate
    that stops that is here, because the aligner reports staleness and the
    adapter cannot see it."""
    streams = build(k=100, temp_stops_at=40)
    ungated = make_pipeline(streams)
    r_ungated = ungated.step()
    assert r_ungated.updated == 100, "expected the ungated pipeline to swallow it"
    assert r_ungated.worst_staleness_s > 10.0

    gated = make_pipeline(build(k=100, temp_stops_at=40), max_stale_s=0.5)
    r = gated.step()
    assert r.updated < 100
    assert r.skipped.get("stale", 0) > 0, (
        "a stream stopped for sixty frames and the staleness gate said nothing")
    assert r.staleness_s["temp"] < -10.0, "staleness was not reported per stream"
    assert r.staleness_s["ref"] == 0.0, "the reference stream cannot be stale"


def test_an_invalid_sensor_blocks_adaptation_unless_told_otherwise():
    streams = build(k=60, invalid_from=30)
    strict = make_pipeline(streams)
    r = strict.step()
    assert r.updated == 30
    assert r.skipped == {"invalid": 30}

    lax = make_pipeline(build(k=60, invalid_from=30), require_valid=False)
    r2 = lax.step()
    assert r2.updated == 60
    assert "invalid" not in r2.skipped


def test_the_input_gate_and_the_guard_are_different_gates():
    """Input health is checked before the update; model health after it. A test
    that could not tell them apart would let either one be quietly removed."""
    streams = build(k=60, invalid_from=30)
    guarded = Guard(RecursiveLeastSquares(3, forgetting=1.0, p0=1e6, dtype="f8"),
                    [Bound("p_min_diag", lo=0.0)])
    pipe = make_pipeline(streams, adapter=guarded)
    r = pipe.step()
    assert r.skipped == {"invalid": 30}, "the invalid frames reached the guard"
    assert guarded.stats["applied"] == 30
    assert guarded.stats["trips"] == 0
    assert r.frozen is False


# ---------------------------------------------------------------------------
# Guard integration
# ---------------------------------------------------------------------------

def test_a_guard_trip_shows_up_in_the_step_report():
    streams = build(k=50)
    guarded = Guard(RecursiveLeastSquares(3, forgetting=1.0, p0=1e6, dtype="f8"),
                    [Bound("p_trace", hi=1.0, name="trace")])
    pipe = make_pipeline(streams, adapter=guarded)
    r = pipe.step()

    assert r.frozen is True
    assert r.frozen_by == ("trace",), f"guard trip not surfaced: {r}"
    assert r.updated < 50
    assert r.skipped.get("guard:trace") == 1, (
        "the tripping frame was not attributed to the check that caught it")
    assert r.skipped.get("frozen", 0) > 0, "later frames were not blocked"
    assert pipe.frozen and pipe.guarded


def test_an_unguarded_pipeline_says_so():
    """Permitted, because the reference has to be runnable unguarded to be the
    oracle the C++ core is checked against. It should not be silent about it."""
    pipe = make_pipeline(build(k=5))
    assert pipe.guarded is False
    assert pipe.frozen is False


# ---------------------------------------------------------------------------
# The frame map
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("features,expected", [
    (["raw", "temp"], (("raw", 0), ("temp", 0))),
    ([("raw", 0), ("temp", 0)], (("raw", 0), ("temp", 0))),
    ([("raw", [0]), "temp"], (("raw", 0), ("temp", 0))),
    ([("raw", slice(0, 1)), ("temp", 0)], (("raw", 0), ("temp", 0))),
    ("raw", (("raw", 0),)),
    (("raw", 0), (("raw", 0),)),
])
def test_frame_map_accepts_the_convenient_forms_and_resolves_them_once(
        features, expected):
    streams = {s.name: s for s in build(k=1)}
    fm = FrameMap(features=features, target="ref")
    fm.resolve(streams)
    assert fm.slots[0] == expected
    assert fm.slots[1] == (("ref", 0),)


def test_a_list_of_names_is_not_mistaken_for_one_name_and_a_selector():
    """`["raw", "temp"]` and `("raw", 0)` are both two-element sequences. Reading
    the first as a name-plus-selector would silently build the wrong regressor."""
    streams = {s.name: s for s in build(k=1)}
    fm = FrameMap(features=["raw", "temp"], target="ref")
    fm.resolve(streams)
    assert fm.slots[0] == (("raw", 0), ("temp", 0))
    assert fm.n_features == 2


def test_feature_order_is_the_parameter_order():
    """It has to be, and it matters: a persisted state block restored against a
    different order is wrong in a way nothing downstream can detect."""
    forward = make_pipeline(build())
    forward.step()
    reverse = Pipeline(
        streams=build(),
        aligner=Aligner.hold_last(reference="ref"),
        adapter=RecursiveLeastSquares(3, forgetting=1.0, p0=1e6, dtype="f8"),
        frame_map=FrameMap(features=["temp", "raw"], target="ref", constant=True))
    reverse.step()

    a = forward.adapter.theta.ravel()
    b = reverse.adapter.theta.ravel()
    assert np.allclose([b[1], b[0], b[2]], a, atol=1e-6), (
        "swapping the feature order did not swap the learned parameters")


def test_a_map_that_does_not_fit_the_adapter_is_refused_at_construction():
    streams = build(k=1)
    with pytest.raises(ValueError, match="adapter expects 3"):
        make_pipeline(streams, constant=False)          # 2 features, adapter wants 3
    with pytest.raises(ValueError, match="outputs"):
        Pipeline(streams=streams, aligner=Aligner.hold_last("ref"),
                 adapter=RecursiveLeastSquares(3, n_outputs=2, dtype="f8"),
                 frame_map=FrameMap(features=["raw", "temp"], target="ref",
                                    constant=True))


def test_a_map_naming_something_that_does_not_exist_is_refused():
    streams = build(k=1)
    with pytest.raises(KeyError, match="no stream named"):
        Pipeline(streams=streams, aligner=Aligner.hold_last("ref"),
                 adapter=RecursiveLeastSquares(2, dtype="f8"),
                 frame_map=FrameMap(features=["raw", "humidity"], target="ref"))
    with pytest.raises(IndexError, match="outside stream"):
        Pipeline(streams=streams, aligner=Aligner.hold_last("ref"),
                 adapter=RecursiveLeastSquares(2, dtype="f8"),
                 frame_map=FrameMap(features=[("raw", 0), ("raw", 4)],
                                    target="ref"))


def test_predict_frame_uses_the_same_layout_as_step():
    """So a caller cannot get the regressor right while adapting and wrong while
    predicting — a mistake that produces plausible numbers and no error."""
    streams = build()
    pipe = make_pipeline(streams)
    pipe.step()

    frame = Aligner.hold_last("ref").align({s.name: s for s in streams},
                                           float(streams[2].latest().t))
    assert frame is not None
    got = pipe.predict_frame(frame)
    want = float(frame.values["ref"][0])
    assert got[0] == pytest.approx(want, abs=1e-5), (
        f"predicting through the map gave {got[0]}, truth was {want}")


# ---------------------------------------------------------------------------
# Configuration that cannot work, refused at construction
# ---------------------------------------------------------------------------

def test_a_non_causal_policy_must_be_asked_for_explicitly():
    """`nearest` and `interpolate` consult samples from after the reference
    instant. Offline that is legitimate; in a causal loop it is a correctness
    bug, and a pipeline is a causal loop."""
    streams = build(k=20)
    for policy in ("nearest", "interpolate"):
        with pytest.raises(ValueError, match="after the reference instant"):
            Pipeline(streams=streams, aligner=Aligner(reference="ref",
                                                      policy=policy),
                     adapter=RecursiveLeastSquares(3, dtype="f8"),
                     frame_map=FrameMap(["raw", "temp"], "ref", constant=True))

    ok = Pipeline(streams=build(k=20), aligner=Aligner.nearest("ref"),
                  adapter=RecursiveLeastSquares(3, forgetting=1.0, p0=1e6,
                                                dtype="f8"),
                  frame_map=FrameMap(["raw", "temp"], "ref", constant=True),
                  allow_non_causal=True)
    assert ok.step().updated == 20


def test_duplicate_stream_names_and_a_missing_reference_are_refused():
    a = Stream("raw", dim=1, dtype="f8")
    b = Stream("raw", dim=1, dtype="f8")
    with pytest.raises(ValueError, match="share a name"):
        Pipeline(streams=[a, b], aligner=Aligner.hold_last("raw"),
                 adapter=RecursiveLeastSquares(1, dtype="f8"),
                 frame_map=FrameMap("raw", "raw"))
    with pytest.raises(KeyError, match="reference stream"):
        Pipeline(streams=build(k=1), aligner=Aligner.hold_last("absent"),
                 adapter=RecursiveLeastSquares(3, dtype="f8"),
                 frame_map=FrameMap(["raw", "temp"], "ref", constant=True))


def test_a_non_positive_staleness_limit_is_refused():
    with pytest.raises(ValueError, match="max_stale_s"):
        make_pipeline(build(k=1), max_stale_s=0.0)


# ---------------------------------------------------------------------------
# Precision travels through the pipeline, and is visible
# ---------------------------------------------------------------------------

def test_stream_storage_precision_costs_a_factor_of_a_few_not_orders():
    """Samples stored as f4 arrive as f4 whatever the adapter computes in. How
    much that costs was measured rather than guessed, and the guess was wrong.

    Measured across four seeds on this fixture: f4 storage gives a parameter
    error 2.2x to 6.7x the f8 one, not the orders of magnitude an initial
    assertion here claimed. The reason is that least squares averages the
    quantisation error down over 200 samples. Worth knowing before anyone trades
    stream precision for memory on the strength of intuition.

    The f8 figure is not fp64's floor either — at ~2e-8 it is the ``p0`` prior
    term from `RecursiveLeastSquares`, the same 1/p0 bias measured in
    `test_adapt.py`. Both numbers here are about the problem, not the format.
    """
    exact = make_pipeline(build(dtype="f8"))
    exact.step()
    coarse = make_pipeline(build(dtype="f4"))
    coarse.step()

    err_f8 = float(np.max(np.abs(exact.adapter.theta.ravel() - TRUE)))
    err_f4 = float(np.max(np.abs(coarse.adapter.theta.ravel() - TRUE)))
    assert err_f8 < 1e-6
    assert err_f4 > 2.0 * err_f8, (
        f"f4 storage cost nothing measurable ({err_f4:.2e} vs {err_f8:.2e}); "
        "either the streams are not storing at f4 or the fixture can no longer "
        "distinguish them")
    assert err_f4 < 20.0 * err_f8, (
        f"f4 storage cost {err_f4 / err_f8:.1f}x, far outside the 2.2-6.7x "
        "measured range; something other than storage precision has changed")


# ---------------------------------------------------------------------------
# Low latency against honest weighting, which pull in opposite directions
# ---------------------------------------------------------------------------
#
# Latency wants the fastest stream as the reference, so a frame -- and an output -- exists at
# every fast sample. But when the target rides a slower stream, hold-last then hands the same
# measurement to the learner over and over. These tests pin both halves: that the reuse is
# real and costs something measurable, and that the freshness gate removes it without giving
# the latency back.


def _rates(fresh, jitter, gate=None, policy="hold_last", ref="fast", seed=1, n=200):
    """A 10 Hz reference and a 1 Hz target: twenty measurements, ~190 frames."""
    rng = np.random.default_rng(seed)
    fast = Stream("fast", dim=1, capacity=4096)
    slow = Stream("slow", dim=1, capacity=4096)
    adapter = RecursiveLeastSquares(2, 1, forgetting=1.0, p0=100.0, dtype="f8")
    pipe = Pipeline([fast, slow], Aligner(ref, policy, None), adapter,
                    FrameMap([("fast", 0)], ("slow", 0), constant=True),
                    max_stale_s=gate, require_fresh_target=fresh,
                    allow_non_causal=policy in ("nearest", "interpolate"))
    for k in range(1, n + 1):
        fast.push(t=k / 10.0, values=np.array([float(k % 7)]))
        if k % 10 == 0:
            t = k / 10.0 + (float(rng.uniform(-jitter, jitter)) if jitter else 0.0)
            slow.push(t=t, values=np.array([2.0 + 0.01 * k]))
    return pipe.step(), adapter


def test_a_slow_target_is_relearned_once_per_fast_frame_without_the_gate():
    """The problem the gate exists for, pinned first so the fix has something to fix.

    Twenty measurements become 191 adaptations. That is not merely wasteful: RLS weights each
    measurement ten times, so P shrinks faster than the information justifies.
    """
    report, adapter = _rates(fresh=False, jitter=0.0)
    assert report.frames == 191
    assert report.updated == 191, "the fixture no longer reuses the target"
    assert report.skipped == {}
    p_trace_reused, _ = adapter.health()

    fresh_report, fresh_adapter = _rates(fresh=True, jitter=0.0)
    assert fresh_report.updated == 20, "one adaptation per distinct measurement"
    p_trace_once, _ = fresh_adapter.health()

    # The cost of the reuse, as a number rather than as an adjective: the covariance ends up
    # an order of magnitude smaller for the same twenty measurements.
    assert p_trace_once > 5 * p_trace_reused, (
        f"reuse left tr(P)={p_trace_reused:.4g} against {p_trace_once:.4g} for the same "
        "information; if these are close the fixture is no longer demonstrating anything")


@pytest.mark.parametrize("jitter", [0.0, 0.02])
def test_the_freshness_gate_is_unaffected_by_jitter(jitter):
    """Twenty measurements, twenty adaptations, whatever the timestamps do.

    This is the whole reason the gate is an identity check on the sample's timestamp rather
    than a threshold on staleness.
    """
    report, _ = _rates(fresh=True, jitter=jitter)
    assert report.updated == 20
    assert report.skipped["stale_target"] == report.frames - 20


@pytest.mark.parametrize("gate,jittered,clean", [(0.05, 12, 20), (0.15, 31, 39)])
def test_max_stale_s_cannot_do_the_gates_job(gate, jittered, clean):
    """The negative control, and it is what justifies adding an option at all.

    `max_stale_s` looks like it can stand in for freshness: with perfect timestamps a 0.05 s
    gate adapts exactly twenty times, once per measurement. Add ±20 ms of jitter -- nothing
    for a real sensor -- and it fails in both directions. Too tight and measurements are lost
    outright; too loose and some are used twice.

    The numbers are pinned rather than described, because "it is fragile" is not a finding
    and "12 of 20" is. They were also written into this test before being measured in *this*
    fixture, and were wrong: the figures came from a scratch script whose random stream was
    consumed differently. Measured here, they are 12 and 31.
    """
    report, _ = _rates(fresh=False, jitter=0.02, gate=gate)
    assert report.updated == jittered, (
        f"a {gate}s staleness gate adapted {report.updated} times against 20 distinct "
        "measurements")
    steady, _ = _rates(fresh=False, jitter=0.0, gate=gate)
    assert steady.updated == clean


def test_no_fixed_staleness_threshold_is_right_for_both_cases():
    """The sharper version of the point above, and the one that actually rules the option in.

    It is not that every threshold is bad -- 0.10 s happens to land on exactly twenty with
    ±20 ms of jitter in this fixture. It is that the *same* threshold gives 32 without the
    jitter. A value tuned on a bench recording is wrong in the field and a value tuned in the
    field is wrong on the bench, so there is no number a caller could have chosen. The
    freshness gate needs no number at all.
    """
    lucky_jittered, _ = _rates(fresh=False, jitter=0.02, gate=0.10)
    lucky_clean, _ = _rates(fresh=False, jitter=0.0, gate=0.10)
    assert lucky_jittered.updated == 20            # looks perfect...
    assert lucky_clean.updated == 32               # ...and is not

    for jitter in (0.0, 0.02):
        fresh, _ = _rates(fresh=True, jitter=jitter)
        assert fresh.updated == 20


def test_the_first_target_is_learned_even_at_time_zero():
    """A first target sample at exactly t = 0 must not be gated as stale.

    Zero is not exotic -- it is the first tick of any device whose timebase is milliseconds
    since boot -- and "no target yet" has to be distinguishable from "the target is 0.0".
    The two implementations do that differently: here `_last_target_t` starts as `None`, so
    no timestamp can equal it; the C++ core has no optional and carries a `have_last_target_`
    flag instead. A mutation check on the C++ side is what prompted testing it on both.
    """
    fast = Stream("fast", dim=1, capacity=16)
    slow = Stream("slow", dim=1, capacity=16)
    adapter = RecursiveLeastSquares(2, 1, dtype="f8")
    pipe = Pipeline([fast, slow], Aligner("fast", "hold_last"), adapter,
                    FrameMap([("fast", 0)], ("slow", 0), constant=True),
                    require_fresh_target=True)
    slow.push(t=0.0, values=np.array([3.0]))
    fast.push(t=0.0, values=np.array([1.5]))
    first = pipe.step()
    assert (first.frames, first.updated) == (1, 1)
    assert "stale_target" not in first.skipped

    fast.push(t=0.1, values=np.array([1.5]))
    second = pipe.step()
    assert (second.frames, second.updated) == (1, 0)
    assert second.skipped["stale_target"] == 1


def test_the_gate_costs_nothing_when_the_target_rides_the_reference():
    """Every frame then carries a new target by construction, so the gate can never fire and
    turning it on is free. Worth pinning because a gate that fired here would silently halve
    the learning rate of the most ordinary configuration there is."""
    off, _ = _rates(fresh=False, jitter=0.0, ref="slow")
    on, _ = _rates(fresh=True, jitter=0.0, ref="slow")
    assert off.updated == on.updated == off.frames
    assert "stale_target" not in on.skipped


def test_interpolation_produces_a_new_target_every_frame():
    """Under interpolation the target is synthesised at the reference instant, so it really is
    new each time and the gate correctly never fires. Stated as a test because the opposite
    would be a defensible-looking bug: the *measurements* behind it have not changed."""
    off, _ = _rates(fresh=False, jitter=0.0, policy="interpolate")
    on, _ = _rates(fresh=True, jitter=0.0, policy="interpolate")
    assert off.frames > 0, "the fixture built no frames, so this compares nothing"
    assert on.updated == off.updated
    assert "stale_target" not in on.skipped


def test_a_gated_frame_still_produces_an_output():
    """The decoupling the whole option exists to allow: refusing to *learn* from a sample is
    not a reason to stop *predicting* on it. Without this a low-latency consumer would lose
    nine outputs in ten."""
    report, _ = _rates(fresh=True, jitter=0.0)
    assert report.prediction is not None
    assert report.skipped["stale_target"] > 0, "no frame was gated, so nothing is shown"


def test_the_prediction_is_the_newest_frames_output():
    fast = Stream("a", dim=1, capacity=16)
    adapter = RecursiveLeastSquares(2, 1, dtype="f8")
    pipe = Pipeline([fast], Aligner("a", "hold_last"), adapter,
                    FrameMap([("a", 0)], ("a", 0), constant=True))
    for k in range(1, 6):
        fast.push(t=float(k), values=np.array([float(k)]))
    report = pipe.step()
    assert report.frames == 5
    # The last frame's regressor against the parameters the step finished with -- not the
    # first frame's, and not a value computed before the last update.
    expect = adapter.predict(np.array([5.0, 1.0], dtype="f8"))
    assert report.prediction == pytest.approx(expect)

    idle = pipe.step()
    assert idle.frames == 0
    assert idle.prediction is None, (
        "a step with no frames has no newest frame to predict on, and a stale answer "
        "presented as a current one is worse than none")


def test_the_prediction_buffer_is_shared_and_says_so():
    """It is preallocated on purpose -- a fresh array per step is an allocation per step, and
    `bench.memory_growth` exists to catch that. The consequence is that it is overwritten by
    the next step, and it is read-only so nobody can mistake it for their own copy."""
    fast = Stream("a", dim=1, capacity=16)
    pipe = Pipeline([fast], Aligner("a", "hold_last"),
                    RecursiveLeastSquares(2, 1, dtype="f8"),
                    FrameMap([("a", 0)], ("a", 0), constant=True))
    fast.push(t=1.0, values=np.array([1.0]))
    first = pipe.step().prediction
    assert not first.flags.writeable
    with pytest.raises(ValueError):
        first[0] = 0.0
    fast.push(t=2.0, values=np.array([9.0]))
    second = pipe.step().prediction
    assert second is first, "the buffer is meant to be reused; see the docstring"


def test_the_prediction_does_not_reintroduce_a_per_step_allocation():
    """The property the preallocation exists for, measured rather than asserted. `Adapter.state`
    was already fixed for exactly this once -- it built a fresh view per call on a path the
    guard took every checkpoint -- so the same mistake in `step()` would be a repeat."""
    fast = Stream("a", dim=1, capacity=64)
    pipe = Pipeline([fast], Aligner("a", "hold_last"),
                    RecursiveLeastSquares(2, 1, dtype="f8"),
                    FrameMap([("a", 0)], ("a", 0), constant=True))
    t = 0.0
    for _ in range(50):                      # settle the interpreter's own bookkeeping
        t += 1.0
        fast.push(t=t, values=np.array([1.0]))
        pipe.step()
    before = sys.getallocatedblocks()
    for _ in range(500):
        t += 1.0
        fast.push(t=t, values=np.array([1.0]))
        pipe.step()
    after = sys.getallocatedblocks()
    # A fresh ndarray per step would be hundreds of blocks; the arrays pushed above are the
    # caller's and are freed each iteration. A handful of blocks of slack absorbs the
    # interpreter's own arenas without absorbing a real per-step leak.
    assert after - before < 50, f"{after - before} blocks grew over 500 steps"


# -- a target that arrives after the features it grades -------------------------------

def _delayed_target_run(feature_stamp_shift, p0=100.0, steps=400,
                        n=4, delay=0.30, feat_hz=50.0, out_hz=10.0):
    """Self-supervision with a physically delayed label, which is the shape an online
    head has on a robot: something is predicted at t and a slower, more trustworthy
    sensor confirms it at t + delay -- foot contact against a predicted step, measured
    torque against a predicted load.

    The reference is the OUTCOME stream, so instants are label arrivals and `hold_last`
    picks the newest features at or before each one. `feature_stamp_shift` is the whole
    experiment.
    """
    true = np.array([2.0, -1.0, 0.5, 0.25])[:n]
    feats = Stream("feat", dim=n, capacity=256, dtype="f8")
    outcome = Stream("outcome", dim=1, capacity=64, dtype="f8")
    pipe = Pipeline(
        streams=[feats, outcome],
        aligner=Aligner(reference="outcome", policy="hold_last"),
        adapter=RecursiveLeastSquares(n, 1, forgetting=0.999, p0=p0, dtype="f8"),
        frame_map=FrameMap([("feat", i) for i in range(n)], ("outcome", 0)),
    )
    t, next_out, pending = 0.0, delay, []
    for k in range(steps):
        e = np.random.default_rng(1000 + k).standard_normal(n)
        feats.push(t + feature_stamp_shift, e)
        pending.append((t + delay, e))
        while next_out <= t:
            ready = [p for p in pending if p[0] <= next_out]
            if ready:
                outcome.push(next_out, [float(ready[-1][1] @ true)])
            next_out += 1.0 / out_hz
        pipe.step()
        t += 1.0 / feat_hz
    theta = np.asarray(pipe.adapter.state[:n], dtype=float)
    return float(np.linalg.norm(theta - true) / np.linalg.norm(true))


def test_a_feature_must_be_stamped_with_the_instant_it_is_about():
    """The trap, and it is silent.

    An online head is graded by a label that arrives later than the features that earned
    it. Stamping each feature vector with the time it was COMPUTED aligns every label
    against features from the wrong moment, and the adapter converges happily on the
    wrong map -- `residual_norm` is *low*, because it fits that wrong map well. **No
    guard can catch this.** Freezing, rollback and the divergence detector all watch the
    adapter's own health, and the adapter is healthy; it is being asked the wrong
    question.

    What works needs no library feature: stamp the feature with the instant it is ABOUT
    rather than the instant it was produced, and `hold_last` on the outcome's timebase
    pairs them correctly.
    """
    wrong = _delayed_target_run(feature_stamp_shift=0.0)
    right = _delayed_target_run(feature_stamp_shift=0.30)
    assert wrong > 0.5, (
        f"the naive stamping no longer learns the wrong map (error {wrong:.3e}); this "
        "fixture no longer demonstrates the trap, so the comparison proves nothing")
    assert right < 1e-3, f"stamping by the instant it is about did not converge: {right:.3e}"
    assert wrong / right > 1000.0


def test_the_delayed_target_residual_is_the_prior_term_and_not_an_error():
    """The correct arrangement leaves ~1.6e-4 at p0=100, which could be read as "nearly
    right" and left alone. It is not error at all: this project measured elsewhere that
    RLS's residual on noiseless data is the `P0^-1` prior term and scales as exactly
    1/p0, and the same law holds here across four decades. Asserting the LAW rather than
    a tolerance is what distinguishes a converged head from an almost-converged one."""
    errs = {p0: _delayed_target_run(0.30, p0=p0) for p0 in (10.0, 100.0, 1000.0, 10000.0)}
    ratios = [errs[a] / errs[b] for a, b in ((10.0, 100.0), (100.0, 1000.0),
                                             (1000.0, 10000.0))]
    for r in ratios:
        assert 9.5 < r < 10.5, f"the 1/p0 law does not hold here: ratios {ratios}"
