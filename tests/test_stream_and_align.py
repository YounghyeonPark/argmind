# SPDX-License-Identifier: Apache-2.0
"""Tests for bounded streams and multi-rate alignment.

Following the standard carried over from the parked project: a suite that can
only demonstrate success is not evidence. Several tests here exist specifically
to prove that failures are *visible* — that dropped samples are counted, that a
dead stream is reported stale rather than silently held forever, and that a
non-causal policy is identifiable as such.
"""

import numpy as np
import pytest

from argmind import Aligner, Stream


# ---------------------------------------------------------------------------
# Stream: bounded memory and honest accounting
# ---------------------------------------------------------------------------

def test_capacity_is_fixed_and_overflow_is_counted():
    """Memory must not grow with uptime, and loss must not be silent."""
    s = Stream("s", dim=1, capacity=4)
    for i in range(10):
        s.push(t=0.1 * (i + 1), values=[i])

    assert len(s) == 4, "buffer grew beyond its declared capacity"
    assert s.dropped == 6, "overwritten samples were not counted"
    # The four newest survived, oldest-first.
    assert [s.at(i).values[0] for i in range(4)] == [6, 7, 8, 9]


def test_ring_wraps_without_reordering():
    s = Stream("s", dim=1, capacity=3)
    for i in range(7):
        s.push(t=float(i + 1), values=[i])
    times = s.times()
    assert list(times) == [5.0, 6.0, 7.0]
    assert np.all(np.diff(times) > 0), "wrapped buffer returned times out of order"


def test_backward_time_is_rejected_at_the_source():
    """Out-of-order time breaks every alignment policy downstream, so it fails
    here where the bug is localisable rather than in a fused estimate later."""
    s = Stream("s", dim=1)
    s.push(t=1.0, values=[0])
    with pytest.raises(ValueError, match="must increase"):
        s.push(t=0.9, values=[1])
    with pytest.raises(ValueError, match="must increase"):
        s.push(t=1.0, values=[1])          # equal is also rejected


def test_returned_samples_do_not_alias_the_ring():
    """A caller holding a sample must not have it mutated by a later push."""
    s = Stream("s", dim=1, capacity=2)
    s.push(t=1.0, values=[42])
    held = s.at(0)
    for i in range(5):
        s.push(t=2.0 + i, values=[i])
    assert held.values[0] == 42, "sample aliased the ring buffer and was mutated"


def test_dimension_and_construction_are_checked():
    with pytest.raises(ValueError):
        Stream("s", dim=0)
    with pytest.raises(ValueError):
        Stream("s", dim=1, capacity=0)
    with pytest.raises(ValueError):
        Stream("s", dim=1, rate_hz=-5)
    s = Stream("s", dim=3)
    with pytest.raises(ValueError, match="expected shape"):
        s.push(t=1.0, values=[1, 2])


def test_invalid_samples_are_retained_not_discarded():
    """"the sensor said it was bad" and "nothing arrived" are different, and a
    consumer usually needs to distinguish them."""
    s = Stream("s", dim=1)
    s.push(t=1.0, values=[0.0], valid=False)
    assert len(s) == 1
    assert s.latest().valid is False


# ---------------------------------------------------------------------------
# Alignment: the policy changes the answer, and that must be visible
# ---------------------------------------------------------------------------

@pytest.fixture
def pair():
    """Fast reference at 100 Hz, slow aiding source at 10 Hz, deliberately offset.

    The slow stream must *straddle* the reference instants used below, or there is
    nothing to interpolate between. An earlier version of this fixture started the
    slow stream at t=0.097 and then asked for alignment at t=0.05 — the code
    correctly refused to extrapolate and returned no frame, which looked like a
    bug in the aligner and was a bug in the fixture.
    """
    fast = Stream("fast", dim=1, capacity=64, rate_hz=100)
    slow = Stream("slow", dim=1, capacity=16, rate_hz=10)
    for i in range(1, 21):
        fast.push(t=i * 0.01, values=[float(i)])
    slow.push(t=0.007, values=[100.0])
    slow.push(t=0.107, values=[200.0])
    return {"fast": fast, "slow": slow}


def test_hold_last_never_looks_ahead(pair):
    al = Aligner.hold_last("fast")
    for f in al.drain(pair):
        assert f.staleness_s["slow"] <= 0.0, (
            "hold_last used a sample from after the reference instant")
    assert al.is_causal


def test_nearest_does_look_ahead_and_says_so(pair):
    """`nearest` is legitimate offline and a bug in a causal loop. The framework
    permits it and identifies it rather than choosing for the user."""
    al = Aligner.nearest("fast")
    frames = al.drain(pair)
    assert not al.is_causal
    assert any(f.staleness_s["slow"] > 0.0 for f in frames), (
        "nearest never picked a future sample on a fixture built to force it — "
        "if this holds, the policies are no longer distinguishable")


def test_policies_disagree_on_the_same_data(pair):
    """If every policy returned the same answer the abstraction would be pointless."""
    ref_t = 0.05
    got = {}
    for name in ("hold_last", "nearest", "interpolate"):
        al = Aligner("fast", name)  # type: ignore[arg-type]
        frame = al.align(pair, ref_t)
        assert frame is not None
        got[name] = float(frame.values["slow"][0])
    assert len({round(v, 9) for v in got.values()}) > 1, (
        f"all policies agreed ({got}); the alignment choice is not being applied")


def test_interpolation_lands_between_its_neighbours(pair):
    al = Aligner.interpolate("fast")
    frame = al.align(pair, 0.05)
    assert frame is not None
    v = float(frame.values["slow"][0])
    assert 100.0 < v < 200.0, f"interpolated value {v} is outside its bracket"


def test_interpolation_refuses_to_extrapolate(pair):
    """Past the end of a stream, interpolation is a guess. It yields nothing
    rather than inventing a value."""
    al = Aligner.interpolate("fast")
    assert al.align(pair, 0.30) is None
    assert al.stats["no_data"] == 1


def test_a_dead_stream_is_reported_stale_not_silently_held():
    """The failure this guards: a sensor stops, hold_last keeps contributing its
    final value forever, and the fused estimate drifts with nothing to show why."""
    fast = Stream("fast", dim=1, capacity=64, rate_hz=100)
    dead = Stream("dead", dim=1, capacity=8, rate_hz=10)
    dead.push(t=0.001, values=[7.0])          # then never again
    for i in range(1, 51):
        fast.push(t=i * 0.01, values=[float(i)])

    held = Aligner.hold_last("fast").drain({"fast": fast, "dead": dead})
    assert held, "expected frames"
    assert held[-1].staleness_s["dead"] < -0.4, "staleness did not accumulate"
    assert held[-1].worst_staleness_s > 0.4

    dropped = Aligner.drop_if_stale("fast", max_stale_s=0.05)
    frames = dropped.drain({"fast": fast, "dead": dead})
    assert dropped.stats["dropped_stale"] > 0, (
        "drop_if_stale emitted every frame despite a stream that died — the "
        "staleness limit is not being enforced")
    assert len(frames) < 50


def test_drain_does_not_re_emit_consumed_frames(pair):
    al = Aligner.hold_last("fast")
    first = al.drain(pair)
    assert first
    assert al.drain(pair) == [], "drain re-emitted frames already consumed"

    pair["fast"].push(t=0.25, values=[99.0])
    assert len(al.drain(pair)) == 1


def test_frames_carry_validity_and_the_policy_that_made_them(pair):
    pair["slow"].push(t=0.25, values=[300.0], valid=False)
    al = Aligner.hold_last("fast")
    pair["fast"].push(t=0.26, values=[26.0])
    frame = al.align(pair, 0.26)
    assert frame is not None
    assert frame.policy == "hold_last"
    assert frame.valid["slow"] is False
    assert frame.all_valid() is False


def test_unknown_policy_and_missing_reference_fail_loudly():
    with pytest.raises(ValueError, match="unknown policy"):
        Aligner("a", "telepathy")  # type: ignore[arg-type]
    with pytest.raises(KeyError, match="reference stream"):
        Aligner.hold_last("absent").drain({"present": Stream("present", dim=1)})


def test_no_frame_before_the_slow_stream_has_any_data():
    """Alignment cannot invent a sample that has not arrived."""
    fast = Stream("fast", dim=1, capacity=8)
    slow = Stream("slow", dim=1, capacity=8)
    fast.push(t=0.01, values=[1.0])
    al = Aligner.hold_last("fast")
    assert al.drain({"fast": fast, "slow": slow}) == []
    assert al.stats["no_data"] == 1


def test_a_non_finite_timestamp_is_refused_before_it_destroys_the_ordering():
    """A defect the C++ port exposed by disagreeing about a single push.

    The ordering check cannot catch a NaN: `NaN <= last` is False, so the sample was
    accepted, `_last_t` became NaN, and then *every* later comparison was False too — the
    monotonicity guarantee was gone for the life of the stream. Observed before the fix:
    pushing 1.0, 1.5, NaN, then 0.1 left the ring holding [1.0, 1.5, nan, 0.1].

    That matters beyond tidiness. Every alignment policy reads the ring through a scan that
    assumes the timestamps are ordered, so one NaN makes `newest_at_or_before` — and with
    it hold_last, nearest and interpolate — capable of returning the wrong sample.
    """
    s = Stream("s", dim=1, capacity=8, dtype="f8")
    s.push(t=1.0, values=[1.0])
    s.push(t=1.5, values=[2.0])

    for bad in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError, match="must be finite"):
            s.push(t=bad, values=[3.0])

    # The ordering check still works afterwards, which is the property the NaN destroyed.
    with pytest.raises(ValueError, match="must increase"):
        s.push(t=0.1, values=[4.0])
    s.push(t=2.0, values=[5.0])

    times = list(s.times())
    assert times == [1.0, 1.5, 2.0]
    assert all(a < b for a, b in zip(times, times[1:])), (
        "the retained timestamps are no longer increasing, which every alignment policy "
        "assumes when it scans the ring")


def test_the_ordering_scan_would_have_been_wrong_with_a_poisoned_ring():
    """Why the check above is worth having rather than merely tidy.

    Constructed directly on the ring's storage, since `push` now refuses to build it: a
    ring whose timestamps are out of order makes `newest_at_or_before` return a sample that
    is not the newest one at or before the query. This is what a single accepted NaN used
    to leave behind.
    """
    s = Stream("s", dim=1, capacity=4, dtype="f8")
    for t, v in ((1.0, 10.0), (1.5, 15.0), (2.0, 20.0)):
        s.push(t=t, values=[v])
    assert s.newest_at_or_before(1.7).values[0] == 15.0

    s._t[2] = 0.1                      # what the NaN path used to allow
    got = s.newest_at_or_before(1.7)
    # The fixture fixes this exactly: the scan runs from the newest end and the newest slot
    # now claims t=0.1, which is <= 1.7, so it returns 20.0 — a sample that is neither the
    # newest at or before 1.7 nor anywhere near it. `!= 15.0` would have passed on any
    # wrong answer at all, including a crash-adjacent one.
    assert got.values[0] == 20.0, (
        f"expected the scan to be fooled into returning 20.0, got {got.values[0]}")


# ---------------------------------------------------------------------------
# What each policy costs in latency, with the fixture stated
# ---------------------------------------------------------------------------
#
# DESIGN.md carried this table for a day with no test behind it, and an audit could not
# reproduce it: the frame count disagreed with another number in the same section, the
# `drop_if_stale` row did not say what staleness limit produced it, and the `interpolate`
# row depended on a phase relationship nobody had written down. All three are pinned here.


def _drain_per_arrival(policy, max_stale=None, n_fast=200, fast_hz=10.0, slow_hz=1.0,
                       slow_phase=0.0):
    """Push in timestamp order and drain after EVERY arrival, which is what a low-latency
    caller does. Returns (frames, worst added delay in seconds, aligner stats).

    The drain cadence is the whole point and it is why this fixture is not the one in
    `test_pipeline.py`. There, 200 fast samples are pushed and `step()` runs once at the
    end, so the frame at t=1.0 is built after the slow stream's first sample has arrived and
    the count is 191. Here the drain at t=1.0 happens before it, so that instant is declined
    and the count is 190. Both are right for their own cadence; quoting them side by side
    without saying which is which is what made the documentation look self-contradictory.
    """
    fast = Stream("fast", dim=1, capacity=4096)
    slow = Stream("slow", dim=1, capacity=4096)
    aligner = Aligner("fast", policy, max_stale)
    events = sorted([(k / fast_hz, "f", k) for k in range(1, n_fast + 1)] +
                    [(j / slow_hz + slow_phase, "s", j)
                     for j in range(1, int(n_fast / (fast_hz / slow_hz)) + 1)])
    pending, delays, frames = {}, [], 0
    for now, kind, idx in events:
        if kind == "f":
            fast.push(t=now, values=np.array([float(idx % 7)]))
            pending[now] = now
        else:
            slow.push(t=now, values=np.array([2.0 + 0.01 * idx]))
        for fr in aligner.drain({"fast": fast, "slow": slow}):
            if fr.t in pending:
                delays.append(now - pending.pop(fr.t))
                frames += 1
    return frames, (max(delays) if delays else 0.0), aligner.stats


def test_the_causal_policies_add_no_latency():
    """A causal policy can answer for instant t the moment t arrives. The measurement is
    close to tautological -- that is the point of choosing a causal policy -- but it is
    measured rather than argued, and it would catch a change that made `drain` wait."""
    for policy in ("hold_last", "nearest"):
        frames, worst, _ = _drain_per_arrival(policy)
        assert frames == 190, f"{policy}: {frames}"
        assert worst == 0.0, f"{policy} added {worst * 1000:.1f} ms"


def test_ten_instants_cannot_be_framed_and_that_is_all_of_them():
    """200 fast samples, 190 frames. The ten are the instants at and before the slow
    stream's first sample: nine strictly before t=1.0, plus t=1.0 itself, whose drain runs
    before the slow sample of the same timestamp arrives."""
    frames, _worst, stats = _drain_per_arrival("hold_last")
    assert frames == 190
    assert stats["no_data"] == 10
    assert stats["dropped_stale"] == 0


def test_nearest_returns_exactly_what_hold_last_returns_here():
    """With the fastest stream as the reference and a drain after every arrival, the future
    sample `nearest` exists to consult has not arrived yet. It buys nothing and costs a
    non-causal flag -- pinned because the aligner's own docstring recommends it for accuracy
    without saying that recommendation is about offline use."""
    def picks(policy):
        fast = Stream("fast", dim=1, capacity=4096)
        slow = Stream("slow", dim=1, capacity=4096)
        aligner = Aligner("fast", policy)
        out = []
        events = sorted([(k / 10.0, "f", k) for k in range(1, 201)] +
                        [(float(j), "s", j) for j in range(1, 21)])
        for now, kind, idx in events:
            (fast if kind == "f" else slow).push(
                t=now,
                values=np.array([2.0 + 0.01 * idx if kind == "s" else float(idx % 7)]))
            for fr in aligner.drain({"fast": fast, "slow": slow}):
                out.append((fr.t, float(fr.values["slow"][0]), fr.staleness_s["slow"]))
        return out

    h, n = picks("hold_last"), picks("nearest")
    assert len(h) == len(n) == 190
    assert h == n, "nearest chose differently somewhere; the degeneracy claim is wrong"


def test_interpolation_builds_nothing_in_a_streaming_loop():
    """Not "adds latency" -- nothing. Bracketing needs a sample later than the reference
    instant, there never is one when the reference is the fastest stream, and `drain`
    consumes the instant anyway so it is never retried.

    Checked at two phases of the slow stream, because the first version of this claim was
    measured at one and an audit reasonably asked whether the phase was load-bearing.
    """
    for phase in (0.0, 0.05):
        frames, _worst, stats = _drain_per_arrival("interpolate", slow_phase=phase)
        assert frames == 0, f"phase {phase}: {frames} frames"
        assert stats["no_data"] == 200


@pytest.mark.parametrize("limit,frames,dropped", [(0.05, 0, 190), (0.5, 95, 95),
                                                  (2.0, 190, 0)])
def test_drop_if_stale_costs_what_its_limit_says(limit, frames, dropped):
    """The row DESIGN.md printed without its limit. At 2 s the policy abandons nothing and
    looks identical to hold-last, which is a row that looks free because the axis that
    costs was switched off. At 0.05 s it abandons everything. The middle is the only
    interesting one, and none of the three is the policy's cost -- the limit is."""
    got_frames, worst, stats = _drain_per_arrival("drop_if_stale", max_stale=limit)
    assert (got_frames, stats["dropped_stale"]) == (frames, dropped)
    assert worst == 0.0
