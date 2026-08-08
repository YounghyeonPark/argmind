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
