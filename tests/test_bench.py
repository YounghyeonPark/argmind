# SPDX-License-Identifier: Apache-2.0
"""Tests for the benchmark harness.

The harness exists to make a claim falsifiable, so the tests here are mostly
about whether it can actually report something bad. A benchmark that has never
failed is in the same position as an optimisation with no way to fail.

Nothing here asserts an absolute duration. Timings are a property of the machine
and gating on them in a repository produces a suite that fails on everyone else's
laptop and gets switched off. What is gated: the deterministic counters, the
state digest, the memory leak rate, and ratios measured within one run.
"""

import json

import numpy as np
import pytest

from argmind.adapt import Adapter, RecursiveLeastSquares
from argmind.align import Aligner
from argmind.bench import (Regression, Result, Timing, Workload, compare,
                           fit_leak, fit_step, gate_saving, main, measure,
                           memory_growth, scaling, step_cost)
from argmind.pipeline import FrameMap, Pipeline
from argmind.stream import Stream

SMALL = Workload(frames=60)


# ---------------------------------------------------------------------------
# The workload is the workload
# ---------------------------------------------------------------------------

def test_the_counters_are_exact():
    r = measure(SMALL, warmup_frames=5)
    assert r.counters["frames"] == 60
    assert r.counters["updated"] == 60
    assert r.counters["pushes"] == 60 * SMALL.pushes_per_frame
    assert r.counters["lost_frames"] == 0, (
        "the benchmark lost input, so it is not measuring the declared workload")
    assert r.counters["state_size"] == 6 + 36


def test_per_sample_and_per_frame_are_genuinely_separate_axes():
    """The requirement they exist for: they scale with different things. Doubling
    the sensor rate must double the pushes and leave the frame count alone. If
    one number moved both, an average over them would hide which one changed."""
    base = measure(Workload(frames=40, imu_hz=200.0), warmup_frames=5)
    faster = measure(Workload(frames=40, imu_hz=400.0), warmup_frames=5)

    assert faster.counters["pushes"] == 2 * base.counters["pushes"]
    assert faster.counters["frames"] == base.counters["frames"] == 40
    assert faster.push_ns.n > base.push_ns.n
    assert faster.frame_ns.n == base.frame_ns.n


def test_timings_are_recorded_per_operation_so_a_p99_exists():
    """`p99` has to be the tail and not another name for the median. The standard
    it exists for is 'measure worst case, not just typical', and a p99 that
    quietly equals p50 would satisfy every assertion about ordering while hiding
    exactly what it was added to show."""
    r = measure(SMALL, warmup_frames=5)
    for t in (r.push_ns, r.frame_ns, r.update_ns):
        assert t.n > 0
        assert t.p50 > 0
        assert t.p99 > t.p50, (
            f"p99 ({t.p99}) is not above p50 ({t.p50}); on real timings the tail "
            "is always heavier, so this is a reporting bug rather than a very "
            "even machine")
        assert t.max >= t.p99


def test_a_frame_costs_more_than_a_push_because_it_does_more():
    """A ratio inside one run, so the machine cancels out. It also pins that the
    two are separate measurements: a step aligns, extracts, adapts and guards,
    while a push appends to a ring."""
    r = measure(SMALL, warmup_frames=5)
    assert r.frame_ns.p50 > 3.0 * r.push_ns.p50, (
        f"a frame ({r.frame_ns.p50:.0f} ns) costs no more than a push "
        f"({r.push_ns.p50:.0f} ns); the two timers are probably measuring the "
        "same thing")


def test_pushes_per_frame_follows_the_declared_rates():
    assert Workload(imu_hz=200, aid_hz=5).pushes_per_frame == 40
    assert Workload(imu_hz=1000, aid_hz=10).pushes_per_frame == 100
    assert Workload(imu_hz=5, aid_hz=5).pushes_per_frame == 1


# ---------------------------------------------------------------------------
# The correctness oracle
# ---------------------------------------------------------------------------

def test_the_state_digest_is_deterministic_for_the_same_workload():
    """Otherwise it cannot be used to tell a speed-up from a changed answer."""
    a = measure(SMALL, warmup_frames=5)
    b = measure(SMALL, warmup_frames=5)
    assert a.state_digest == b.state_digest


def test_the_state_digest_changes_when_the_computation_changes():
    """The other half. A digest that never changes would pass every comparison
    and prove nothing."""
    a = measure(Workload(frames=60, seed=1), warmup_frames=5)
    b = measure(Workload(frames=60, seed=2), warmup_frames=5)
    c = measure(Workload(frames=60, seed=1, dtype="f8"), warmup_frames=5)
    assert a.state_digest != b.state_digest
    assert a.state_digest != c.state_digest


# ---------------------------------------------------------------------------
# Can it report a regression? This is the point.
# ---------------------------------------------------------------------------

def synth(p50=1000.0, p99=2000.0, digest="abc", **counters) -> Result:
    t = Timing(n=100, p50=p50, p99=p99, max=p99 * 2, mean=p50)
    return Result(workload={}, push_ns=t, frame_ns=t, update_ns=t,
                  counters={"frames": 100, **counters}, state_digest=digest)


def test_comparing_a_result_with_itself_reports_nothing():
    r = synth()
    assert compare(r, r) == []


def test_a_slowdown_beyond_the_tolerance_is_reported():
    slower = synth(p50=1400.0, p99=2800.0)
    found = compare(synth(), slower, tolerance=1.25)
    assert found, "a 40% slowdown was not reported"
    assert all(f.kind == "timing" for f in found)
    assert {f.what for f in found} >= {"push_ns.p50", "frame_ns.p99"}


def test_a_slowdown_inside_the_tolerance_is_not_reported():
    """The other direction matters as much. A gate that fires on ordinary
    variation is a gate someone turns off."""
    assert compare(synth(), synth(p50=1100.0, p99=2200.0), tolerance=1.25) == []


def test_measured_noise_widens_the_gate_rather_than_a_guess():
    """A machine that varies 60% between identical runs cannot support a 25%
    gate. The spread recorded by --record is what raises the threshold."""
    noisy = synth(p50=1500.0, p99=3000.0)
    assert compare(synth(), noisy, tolerance=1.25) != []
    quiet = compare(synth(), noisy, tolerance=1.25,
                    noise={"push_ns.p50": 0.4, "push_ns.p99": 0.4,
                           "frame_ns.p50": 0.4, "frame_ns.p99": 0.4,
                           "update_ns.p50": 0.4, "update_ns.p99": 0.4})
    assert quiet == [], f"measured noise did not widen the gate: {quiet}"


def test_a_metric_that_recorded_no_spread_inherits_the_machines_noise():
    """Agreeing twice is a small sample, not a noise-free metric.

    Taking a recorded spread of exactly zero at face value hands that metric the
    *tightest* gate of any, which is backwards. Measured on the machine this was
    written on, one metric in six records exactly zero while the others range
    from 38% to 566% — and that is what made the record/compare round trip flake
    before the floor existed.
    """
    fast = Timing(n=100, p50=1000.0, p99=2000.0, max=4000.0, mean=1000.0)
    slow = Timing(n=100, p50=1800.0, p99=2000.0, max=4000.0, mean=1800.0)
    base = Result({}, fast, fast, fast, {"frames": 1}, "d")
    now = Result({}, slow, fast, fast, {"frames": 1}, "d")
    noise = {"push_ns.p50": 0.0, "push_ns.p99": 0.6, "frame_ns.p50": 0.6,
             "frame_ns.p99": 0.6, "update_ns.p50": 0.6, "update_ns.p99": 0.6}

    assert compare(base, now, tolerance=1.25) != [], "the 80% jump is real"
    assert compare(base, now, tolerance=1.25, noise=noise) == [], (
        "a metric that recorded zero spread was gated at 1.25 while every other "
        "metric on the same machine varied by 60%")


def test_a_changed_answer_is_reported_first_and_as_its_own_kind():
    """A change that goes faster by computing something else is not a speed-up,
    and folding it into a timing delta would let it pass as one."""
    found = compare(synth(), synth(p50=900.0, digest="different"))
    assert found[0].kind == "correctness"
    assert found[0].what == "state_digest"
    assert "different computations" in found[0].detail


def test_a_changed_workload_is_reported_rather_than_compared_through():
    found = compare(synth(), synth(frames=50))
    assert [f.kind for f in found] == ["counter"]
    assert "no longer the same workload" in found[0].detail


def test_regressions_are_ordered_most_serious_first():
    found = compare(synth(), synth(p50=9999.0, digest="x", frames=1))
    assert [f.kind for f in found[:2]] == ["correctness", "counter"]
    assert isinstance(found[0], Regression)
    assert str(found[0]).startswith("[correctness]")


# ---------------------------------------------------------------------------
# Memory: a function of configuration, not of uptime
# ---------------------------------------------------------------------------

def test_cost_per_frame_is_not_the_leak_rate():
    """The idea the memory measurement rests on, checked on known inputs.

    At a small batch, fixed overhead divided by a few frames is indistinguishable
    from accumulation. Here the cost is identical at both batch sizes, so the
    true leak is zero — but cost-per-frame at the small batch reads as 15.6
    blocks per frame, which is the wrong answer by any margin.

    This is checked synthetically rather than on the live pipeline because the
    live pipeline's fixed overhead is now near zero, so the two methods agree
    there and nothing would be demonstrated. That was not true before
    `Adapter.state` stopped allocating a view per call.
    """
    flat = [{"frames": 64, "blocks_per_batch": 1000.0},
            {"frames": 256, "blocks_per_batch": 1000.0}]
    leak, fixed = fit_leak(flat)
    assert leak == pytest.approx(0.0, abs=1e-9)
    assert fixed == pytest.approx(1000.0, rel=1e-9)
    assert flat[0]["blocks_per_batch"] / flat[0]["frames"] > 15.0

    real = [{"frames": 64, "blocks_per_batch": 100.0 + 2.0 * 64},
            {"frames": 256, "blocks_per_batch": 100.0 + 2.0 * 256}]
    leak, fixed = fit_leak(real)
    assert leak == pytest.approx(2.0, rel=1e-9)
    assert fixed == pytest.approx(100.0, rel=1e-9)


def test_the_pipeline_does_not_accumulate_memory_with_uptime():
    """DESIGN.md asserts this in several places and nothing checked it until now.

    The number that means "leak" is the slope of cost against frames, not the
    cost itself — see `memory_growth`, where the first attempt fitted a trend
    line to a converging series and reported growth that was a settling
    transient.
    """
    m = memory_growth(Workload(frames=200), batch_frames=(64, 256), repeats=3)
    assert m["blocks_per_frame"] < 0.1, (
        f"retained memory grows with uptime at {m['blocks_per_frame']:.4f} "
        f"blocks/frame: {m['rows']}")


def test_the_leak_detector_detects_a_leak():
    """The positive control, and the reason to believe the test above.

    A leak detector that has never detected a leak is not evidence. This adapter
    keeps every report it has ever produced, which is the most ordinary way an
    online system grows without anyone noticing.
    """
    class Hoarder(Adapter):
        """Deliberately leaky: keeps every report forever."""

        def __init__(self, inner):
            self.inner = inner
            self.kept = []

        n_features = property(lambda self: self.inner.n_features)
        n_outputs = property(lambda self: self.inner.n_outputs)
        dtype = property(lambda self: self.inner.dtype)
        state = property(lambda self: self.inner.state)

        def update(self, x, y):
            r = self.inner.update(x, y)
            self.kept.append(r)
            return r

        def predict(self, x):
            return self.inner.predict(x)

        def health(self):
            return self.inner.health()

        def load_state(self, flat):
            self.inner.load_state(flat)

        def reset(self):
            self.inner.reset()

    def leaky_build(w):
        imu = Stream("imu", dim=w.n_features - 1,
                     capacity=w.pushes_per_frame + 8, dtype=w.dtype)
        aid = Stream("aid", dim=1, capacity=16, dtype=w.dtype)
        adapter = Hoarder(RecursiveLeastSquares(w.n_features, dtype=w.dtype))
        pipe = Pipeline(streams=[imu, aid], aligner=Aligner.hold_last("aid"),
                        adapter=adapter,
                        frame_map=FrameMap("imu", "aid", constant=True))
        return imu, aid, pipe

    m = memory_growth(Workload(frames=200), batch_frames=(64, 256), repeats=3,
                      build=leaky_build)
    assert m["blocks_per_frame"] > 1.0, (
        f"an adapter that keeps every report was not detected as leaking "
        f"({m['blocks_per_frame']:.4f} blocks/frame): {m['rows']}")


# ---------------------------------------------------------------------------
# Scaling: measured, and it says something the project needs to know
# ---------------------------------------------------------------------------

def test_update_cost_is_flat_at_small_dimensions_because_overhead_dominates():
    """A measured result rather than an assumed one, and it was not what a first
    guess would have been.

    RLS is O(n^2), so a naive scaling test would assert cost(32)/cost(4) near 64.
    Measured, it is about 1.1: cost is flat from n=4 to n=32 and only starts
    climbing at n=64. The arithmetic is not the cost at these sizes; the Python
    and numpy call overhead is. That is direct support for the C++ decision in
    DESIGN.md, which argued it from the parked project rather than from here.
    """
    s = scaling(dims=(4, 32, 128), updates=800, warmup=100)
    flat = s[32] / s[4]
    assert flat < 2.0, (
        f"cost from n=4 to n=32 rose x{flat:.2f}; the small-dimension region is "
        "no longer overhead-bound, which changes what the C++ port is for")


def test_a_cubic_regression_in_the_update_would_still_be_caught():
    """The flat region above is only safe to assert because the large end is
    still checked. O(n^2) predicts x16 from n=32 to n=128 and measurement gives
    about x2.9 with overhead included; O(n^3) would give x64."""
    s = scaling(dims=(32, 128), updates=800, warmup=100)
    ratio = s[128] / s[32]
    assert ratio > 1.3, (
        f"n=128 costs only x{ratio:.2f} of n=32, so the large end is still "
        "overhead-bound and this test cannot detect a complexity regression")
    assert ratio < 10.0, (
        f"n=128 costs x{ratio:.2f} of n=32, far above the x16 an O(n^2) "
        "recursion allows even before overhead flattens it — the update may "
        "have become cubic")


# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------

def test_record_then_compare_round_trips(tmp_path, capsys):
    """The plumbing, with a deliberately generous tolerance.

    This machine's run-to-run spread on a 40-frame workload was measured between
    38% and 566%, so a tight gate here would be testing the laptop rather than
    the harness. The gate's ability to *fail* is covered deterministically by the
    digest test below, which owes nothing to timing.
    """
    base = tmp_path / "baseline.json"
    assert main(["--frames", "40", "--repeats", "3", "--record", str(base)]) == 0
    blob = json.loads(base.read_text("utf-8"))
    assert "result" in blob and "noise" in blob
    assert blob["result"]["counters"]["frames"] == 40
    assert any(v > 0 for v in blob["noise"].values()), (
        f"every metric recorded zero run-to-run spread ({blob['noise']}); that "
        "is not what a real machine does, so the noise was not measured and the "
        "gate has nothing to widen")

    assert main(["--frames", "40", "--tolerance", "50",
                 "--compare", str(base)]) == 0
    assert "no regressions" in capsys.readouterr().out


def test_compare_exits_non_zero_when_something_got_worse(tmp_path, capsys):
    """The harness has to be usable as a gate, which means failing loudly enough
    for a shell to notice."""
    base = tmp_path / "baseline.json"
    main(["--frames", "40", "--repeats", "2", "--record", str(base)])

    blob = json.loads(base.read_text("utf-8"))
    blob["result"]["state_digest"] = "deliberately-different"
    blob["noise"] = {}
    base.write_text(json.dumps(blob), "utf-8")

    assert main(["--frames", "40", "--compare", str(base)]) == 1
    out = capsys.readouterr().out
    assert "REGRESSIONS" in out
    assert "[correctness]" in out


def test_the_bare_workload_runs_without_a_guard():
    r = measure(Workload(frames=40, guarded=False), warmup_frames=5)
    assert r.counters["updated"] == 40
    assert Workload(guarded=False).name().endswith("bare-200/5Hz")


@pytest.mark.parametrize("dtype", ["f4", "f8"])
def test_both_precisions_are_benchmarkable(dtype):
    r = measure(Workload(frames=40, dtype=dtype), warmup_frames=5)
    assert r.counters["updated"] == 40
    assert np.isfinite(r.update_ns.p50)


# ---------------------------------------------------------------------------
# step() cost against frames drained, which is the latency question
# ---------------------------------------------------------------------------


def test_fit_step_recovers_a_known_line():
    """The part that can be checked without a stopwatch, and the reason it is a separate
    function. Same argument as `fit_leak`: cost divided by frames at one batch size cannot
    tell fixed overhead from per-frame work."""
    rows = [{"frames": k, "ns_per_step": 5000.0 + 800.0 * k}
            for k in (1, 2, 4, 8, 16)]
    per_frame, fixed = fit_step(rows)
    assert per_frame == pytest.approx(800.0, rel=1e-9)
    assert fixed == pytest.approx(5000.0, rel=1e-9)


def test_cost_per_frame_at_one_batch_size_cannot_separate_the_two():
    """The negative control for the function above, and the whole reason the sweep exists.

    A pipeline with heavy fixed overhead and a cheap frame, and one with no overhead and a
    dearer frame, cost exactly the same at one frame per step. Only the slope tells them
    apart, and a benchmark that reported the one-frame figure alone would call them equal.
    """
    heavy_fixed = [{"frames": k, "ns_per_step": 9000.0 + 1000.0 * k} for k in (1, 2, 4, 8)]
    no_fixed = [{"frames": k, "ns_per_step": 0.0 + 10000.0 * k} for k in (1, 2, 4, 8)]
    assert heavy_fixed[0]["ns_per_step"] == no_fixed[0]["ns_per_step"]
    assert fit_step(heavy_fixed)[0] == pytest.approx(1000.0)
    assert fit_step(no_fixed)[0] == pytest.approx(10000.0)


def test_a_slope_through_one_point_is_refused_rather_than_invented():
    """numpy will fit a line to a single point and warn about the conditioning in a message
    nobody reads. A fabricated slope in a benchmark is worse than no slope: it would be
    quoted. So one point, or several at the same frame count, reports NaN."""
    import math
    assert all(math.isnan(v) for v in fit_step([{"frames": 4, "ns_per_step": 1000.0}]))
    assert all(math.isnan(v) for v in fit_step(
        [{"frames": 4, "ns_per_step": 1000.0}, {"frames": 4, "ns_per_step": 1200.0}]))
    got = step_cost(frames_per_step=(8,), steps=10, warmup=2, guarded=False)
    assert math.isnan(got["ns_per_frame"])
    # ...and the directly measured figure is still there, because it needed no fit.
    assert got["measured_one_frame_ns"] is None      # no one-frame row in this sweep
    assert got["rows"][0]["ns_per_step"] > 0


def test_step_cost_sweeps_and_the_counters_say_what_ran():
    """The timings are machine-specific and are not asserted. The counters are not: they say
    the sweep actually drained the frames it claims to, which is what makes the slope a
    measurement of `step()` rather than of an empty loop."""
    r = step_cost(frames_per_step=(1, 2, 4), steps=40, warmup=5, guarded=False)
    assert [row["frames"] for row in r["rows"]] == [1, 2, 4]
    for row in r["rows"]:
        # Every frame adapts with the gate off, so the count is steps x frames plus the
        # warmup. Asserted as a lower bound because the very first frames of a run can be
        # refused while the slower stream has nothing to offer.
        assert row["updated"] >= 40 * row["frames"], row
    assert r["ns_per_frame"] > 0.0
    assert r["measured_one_frame_ns"] == r["rows"][0]["ns_per_step"]


def test_the_freshness_gate_adapts_once_per_step_however_many_frames_are_drained():
    """The counter half of `gate_saving`, which is deterministic where the timings are not.

    This is the option's whole behaviour in one number: with the gate on, a sixteen-frame
    step carrying one target adapts once, not sixteen times.
    """
    off = step_cost(frames_per_step=(8,), steps=40, warmup=5, guarded=False,
                          fresh_target=False)
    on = step_cost(frames_per_step=(8,), steps=40, warmup=5, guarded=False,
                         fresh_target=True)
    assert off["rows"][0]["updated"] >= 8 * 40
    assert on["rows"][0]["updated"] <= 45, (
        f"the gate let {on['rows'][0]['updated']} adaptations through where one per step "
        "was expected")


def test_the_gate_is_cheaper_per_frame_than_adapting():
    """A ratio measured inside one run, so the machine cancels out — the only kind of timing
    this suite is willing to assert. The margin is wide (measured about 0.23) because a
    loaded machine widens everything; what would fail is the gate becoming no cheaper than
    the update it skips, which would mean it had stopped skipping."""
    gs = gate_saving(frames_per_step=(1, 4, 16), steps=60, guarded=False)
    assert gs["per_frame_ratio"] < 0.7, gs["per_frame_ratio"]
    assert gs["adaptations_on"] * 4 < gs["adaptations_off"]


# -- update cost by adapter, and the flop model that makes it portable -------------------

def test_the_flop_model_is_tied_to_the_implementation_not_to_a_comment():
    """`update_flops` is the transferable half of `adapter_scaling`: host microseconds are
    dispatch, a flop count is architectural. A formula in a docstring rots, so it is checked
    against the adapters' own declared shapes and against the scaling laws it encodes.
    """
    from argmind.adapt import MultilayerPerceptron, NormalisedLeastMeanSquares
    from argmind.bench import update_flops

    # RLS is O(n^2): doubling n must quadruple the count, to within the linear terms.
    r = [update_flops("rls", n) for n in (64, 128, 256)]
    assert 3.9 < r[1] / r[0] < 4.1 and 3.9 < r[2] / r[1] < 4.1

    # NLMS has no n^2 term at all, which is the entire reason it exists.
    q = [update_flops("nlms", n) for n in (64, 128, 256)]
    assert 1.9 < q[1] / q[0] < 2.1 and 1.9 < q[2] / q[1] < 2.1
    assert update_flops("nlms", 256) < update_flops("rls", 256) / 100

    # A network's count must follow its OWN weight block, so a layout change breaks this
    # rather than leaving the formula quietly describing something else.
    for hidden in ((), (16,), (32, 32), (7, 5, 3)):
        net = MultilayerPerceptron(4, 2, hidden=hidden, dtype="f8")
        biases = sum((*hidden, 2))
        weights = int(np.asarray(net.state).size) - biases
        assert update_flops("mlp", 4, 2, hidden) == 8 * weights, (
            f"the flop model for {hidden} does not match the adapter's own weight count")

    # RLS's state and its flops both grow as n^2, so at EQUAL STATE a network costs about
    # twice an RLS. That is the claim the docstring makes and it is arithmetic, not timing.
    rls = RecursiveLeastSquares(n_features=64, dtype="f8")
    net = MultilayerPerceptron(3, 1, hidden=(64, 64), dtype="f8")
    assert 0.9 < np.asarray(net.state).size / np.asarray(rls.state).size < 1.2
    ratio = (update_flops("mlp", 3, 1, (64, 64))
             / update_flops("rls", 64, 1))
    assert 1.6 < ratio < 2.6, f"at equal state the flop ratio is {ratio:.2f}, not ~2"
    assert isinstance(NormalisedLeastMeanSquares(4, 1, dtype="f8"), object)

    with pytest.raises(ValueError, match="unknown adapter kind"):
        update_flops("kalman", 6)


def test_adapter_scaling_shows_the_small_adapters_are_dispatch_bound():
    """The one timing claim here that is safe to assert: `ns/flop` spans two orders of
    magnitude. A constant would mean arithmetic-bound; it is not, and that span is the
    evidence for DESIGN.md's rule that these figures are not a baseline for the C++ core.

    Nothing tighter is asserted. Run-to-run spread is 1.07-1.18x interleaved, and the
    equal-accuracy comparison this function's docstring records as **not reported** failed
    exactly that test -- 1.26x against spreads of 1.33x and 1.62x.
    """
    from argmind.bench import adapter_scaling

    rows = adapter_scaling(updates=400, warmup=100)
    assert {r["kind"] for r in rows} == {"rls", "nlms", "mlp"}
    per = [r["ns_per_flop"] for r in rows]
    assert max(per) / min(per) > 20.0, (
        f"ns/flop spans only {max(per) / min(per):.1f}x; if the cost has become "
        "arithmetic-bound across the whole range, the claim that these timings are "
        "dispatch must be re-derived rather than this bound loosened")

    # The smallest adapter must be the most dispatch-bound one, or "small is overhead" is
    # not what the span is showing.
    worst = max(rows, key=lambda r: r["ns_per_flop"])
    assert worst["flops"] < 2000, (
        f"the worst ns/flop is {worst['label']} at {worst['flops']} flops, which is not a "
        "small adapter -- the span is not the dispatch effect it is described as")


def test_the_adapters_report_runs_from_the_command_line(capsys):
    """Two entry points in this repository have shipped with no caller. The reporting path
    has its own formatting and its own division, so compiling is not enough."""
    from argmind.bench import main

    assert main(["--frames", "40", "--adapters"]) == 0
    out = capsys.readouterr().out
    assert "ns/flop spans" in out and "mlp (32, 32)" in out
