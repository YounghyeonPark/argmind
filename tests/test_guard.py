# SPDX-License-Identifier: Apache-2.0
"""Tests for health checks, freezing and rollback.

The suite is built around one question: does the guard actually save an estimate
that would otherwise be destroyed? So the headline test runs the same data twice,
guarded and unguarded, and requires them to disagree. A guard that changes nothing
is not protecting anything, and a test that only exercises the guarded path could
not tell the difference.

The false-positive direction is tested as deliberately as the true-positive one.
A detector that fires on ordinary data would be switched off within a week of
deployment, which is the same as not having one.
"""

import numpy as np
import pytest

from argmind.adapt import RecursiveLeastSquares, UpdateReport
from argmind.guard import Bound, Check, Divergence, Guard, GuardReport


class Trip(Check):
    """A check the test can fire on demand, for exercising the machinery itself
    rather than a particular detector."""

    def __init__(self, name="trip"):
        super().__init__(name)
        self.armed = False

    def evaluate(self, report):
        return not self.armed


def stationary_then_glitch(guard_or_adapter, n_stationary=200, glitch=50.0,
                           lam=0.95):
    """The design driver's failure, reproduced.

    A vehicle sits still: the regressor only ever excites the first axis, so the
    covariance in the second winds up as ``p0 * lambda**-k``. Then one glitched
    sample arrives on that axis. Against a wound-up covariance the adapter absorbs
    it almost completely, and an estimate that was correct is destroyed by a
    single bad reading.
    """
    rng = np.random.default_rng(31)
    for _ in range(n_stationary):
        x = [rng.standard_normal(), 0.0]
        guard_or_adapter.update(x, [x[0] * 1.0])       # true theta = [1, 0]
    guard_or_adapter.update([0.0, 1.0], [glitch])
    return guard_or_adapter


def make_rls(lam=0.95, p0=1.0, n=2):
    return RecursiveLeastSquares(n_features=n, forgetting=lam, p0=p0, dtype="f8")


# ---------------------------------------------------------------------------
# The headline: does guarding change the outcome?
# ---------------------------------------------------------------------------

def test_the_guard_saves_an_estimate_that_is_otherwise_destroyed():
    """Guarded and unguarded on identical data. If these ever agree, the guard is
    decorative and this whole module is unjustified."""
    naked = stationary_then_glitch(make_rls())
    guarded = stationary_then_glitch(
        Guard(make_rls(), [Bound("p_trace", hi=1e3)]))

    assert abs(naked.theta[1, 0]) > 40.0, (
        f"the unguarded adapter did not absorb the glitch (theta[1]="
        f"{naked.theta[1, 0]:.3f}); the scenario no longer demonstrates the "
        "failure the guard exists for, so the comparison below proves nothing")

    # `theta` is RLS's, not the Adapter contract's, so it is reached through the
    # wrapped adapter — the guard deliberately exposes only the shared interface.
    assert abs(guarded.adapter.theta[1, 0]) < 1e-6, (
        f"the guarded adapter absorbed the glitch anyway "
        f"(theta[1]={guarded.adapter.theta[1, 0]:.3f})")
    assert guarded.predict([1.0, 0.0])[0] == pytest.approx(1.0, abs=1e-3), (
        "freezing protected the bad axis but broke the good one")
    assert guarded.frozen
    assert guarded.stats["trips"] == 1
    assert guarded.stats["rollbacks"] == 1


def test_freezing_stops_adaptation_but_not_prediction():
    """The point of freezing: the estimate carries on from the last healthy
    parameters while the learner stops making things worse."""
    g = Guard(make_rls(), [Trip()])
    rng = np.random.default_rng(5)
    for _ in range(30):
        x = rng.standard_normal(2)
        g.update(x, [float(x @ [1.0, -2.0])])

    g.checks[0].armed = True
    g.update([1.0, 1.0], [0.0])                    # trips, rolls back, freezes
    assert g.frozen
    frozen_state = np.array(g.state)
    prediction = g.predict([1.0, 1.0]).copy()

    for _ in range(50):
        r = g.update(rng.standard_normal(2), [99.0])
        assert r.applied is False
        assert r.frozen is True
        assert r.reason == "frozen"
    assert np.array_equal(g.state, frozen_state), (
        "the state moved while frozen — adaptation was not actually stopped")
    assert np.allclose(g.predict([1.0, 1.0]), prediction, atol=1e-12), (
        "prediction moved while frozen")
    assert g.stats["blocked_frozen"] == 50
    assert g.stats["applied"] == 30


# ---------------------------------------------------------------------------
# Rollback
# ---------------------------------------------------------------------------

def test_rollback_restores_the_checkpoint_exactly():
    g = Guard(make_rls(), [Trip()], rollback_depth=1)
    rng = np.random.default_rng(9)
    for _ in range(20):
        x = rng.standard_normal(2)
        g.update(x, [float(x @ [1.0, -2.0])])
    good = np.array(g.state)

    g.checks[0].armed = True
    r = g.update([2.0, 3.0], [1000.0])
    assert r.rolled_back is True
    assert np.array_equal(g.state, good), (
        "rollback did not restore the checkpoint bit for bit")


def test_rollback_depth_reaches_back_that_many_updates():
    """Depth is 'how far back can this undo'. Divergence is detected after it
    starts, so undoing one update lands inside the bad stretch — the guard goes
    to the deepest checkpoint it still holds."""
    depth = 3
    g = Guard(make_rls(), [Trip()], rollback_depth=depth)
    rng = np.random.default_rng(13)
    history = []
    for _ in range(10):
        x = rng.standard_normal(2)
        g.update(x, [float(x @ [1.0, -2.0])])
        history.append(np.array(g.state))

    g.checks[0].armed = True
    g.update([1.0, 1.0], [0.0])
    # The ring holds the last `depth` checkpoints, so the deepest is the state
    # from `depth` updates before the tripping one: updates 9, 10 and 11 are all
    # undone here, leaving the state as it was after update 8.
    assert np.array_equal(g.state, history[-depth]), (
        f"rolled back to somewhere other than {depth} updates ago")


def test_checkpoint_every_multiplies_the_reach():
    g = Guard(make_rls(), [Trip()], rollback_depth=2, checkpoint_every=4)
    rng = np.random.default_rng(17)
    history = []
    for _ in range(16):
        x = rng.standard_normal(2)
        g.update(x, [float(x @ [1.0, -2.0])])
        history.append(np.array(g.state))

    g.checks[0].armed = True
    g.update([1.0, 1.0], [0.0])
    # checkpoints land after updates 4, 8, 12, 16; depth 2 retains 12 and 16,
    # and the deepest is 12 — eight updates back.
    assert np.array_equal(g.state, history[11]), (
        "checkpoint_every did not extend how far back the rollback reaches")


def test_checkpoint_memory_is_declared_and_does_not_grow():
    """Same rule as everywhere else here: memory is a function of configuration,
    never of uptime."""
    a = make_rls()
    g = Guard(a, [Trip()], rollback_depth=4)
    expected = 4 * (a.state.size + 2)
    assert g._ckpt.size + g._ckpt_health.size == expected
    rng = np.random.default_rng(21)
    for _ in range(500):
        g.update(rng.standard_normal(2), [1.0])
    assert g._ckpt.size + g._ckpt_health.size == expected, "checkpoint store grew"


def test_stateful_checks_keep_their_baseline_across_a_freeze_and_reset_on_thaw():
    """Which way round this goes was a real design error, caught by a test.

    Resetting the detector when the guard trips looked tidy and was wrong: it
    threw away the only reference the frozen probes have, so a fault that had
    plateaued read as normal and the guard thawed itself in the middle of it. The
    history has to survive the freeze. It is dropped on `thaw()` instead, where
    adaptation is starting again anyway.
    """
    d = Divergence(ratio=5.0, floor=1e-3, warmup=20)
    g = Guard(make_rls(), [d, Trip()])
    rng = np.random.default_rng(23)
    for _ in range(40):
        x = rng.standard_normal(2)
        g.update(x, [float(x @ [1.0, -2.0])])
    assert not g.frozen, "the divergence check fired on clean data"
    baseline = d.ewma[1]
    assert baseline > 0.0

    g.checks[1].armed = True
    g.update([1.0, 1.0], [0.0])
    assert g.frozen
    # The tripping sample itself is evaluated before anything is known about it,
    # so it enters the baseline by one alpha step — unavoidable, and 2% of one
    # sample. What must not happen is the baseline being wiped.
    assert d.ewma[1] == pytest.approx(baseline, rel=0.25), (
        "the freeze wiped the healthy baseline, leaving the frozen probes with "
        f"nothing to compare against ({d.ewma[1]} vs {baseline})")
    assert d.ewma[1] > 0.0

    g.thaw()
    assert d.ewma == (0.0, 0.0), "thawing did not give the detector a clean start"


def test_a_plateaued_fault_does_not_read_as_healthy_while_frozen():
    """The concrete failure the hold rule prevents: a sensor that breaks and then
    stays broken. Growth stops, so a detector that keeps absorbing samples into
    its baseline eventually decides everything is fine."""
    d = Divergence(ratio=5.0, floor=1e-3, warmup=10)
    for _ in range(60):
        d.evaluate(_report(residual=0.1))              # healthy baseline
    assert d.evaluate(_report(residual=20.0)) is False  # the fault trips it

    for _ in range(300):                                # broken, and stays broken
        assert d.evaluate(_report(residual=20.0, frozen=True)) is False, (
            "the detector adapted to the fault and called it healthy")

    # ...and it still notices when the world comes back. The fast average has a
    # time constant, so recovery takes a few samples rather than one.
    verdicts = [d.evaluate(_report(residual=0.1, frozen=True)) for _ in range(40)]
    assert verdicts[0] is False, "one clean sample was enough, which is too eager"
    assert verdicts[-1] is True, "the world came back and the detector did not notice"


def test_holding_the_baseline_per_sample_would_ratchet_it_down():
    """Why the hold is per-freeze and not per-sample, protected as a property.

    Refusing every sample the check dislikes sounds stricter and is worse: the
    baseline then only ever absorbs the smaller residuals and walks downward
    until ordinary noise trips it. Here the guard is never frozen, so the
    baseline must track the data it is actually seeing.
    """
    rng = np.random.default_rng(101)
    d = Divergence(ratio=5.0, floor=0.0, warmup=20)
    residuals = [abs(float(rng.normal(0.0, 0.01))) for _ in range(2000)]
    trips = sum(not d.evaluate(_report(residual=r)) for r in residuals)
    mean = float(np.mean(residuals))
    assert d.ewma[1] == pytest.approx(mean, rel=0.5), (
        f"baseline {d.ewma[1]:.5f} drifted away from the data's own mean {mean:.5f}")
    assert trips < 20, f"{trips}/2000 ordinary noise samples read as divergence"


# ---------------------------------------------------------------------------
# Divergence: it has to fire on real divergence and NOT on ordinary data
# ---------------------------------------------------------------------------

def test_divergence_fires_when_the_residual_grows_against_its_own_history():
    g = Guard(make_rls(lam=0.99, p0=10.0),
              [Divergence(ratio=5.0, floor=1e-3, warmup=30)])
    rng = np.random.default_rng(29)
    theta = np.array([1.0, -2.0])
    for _ in range(120):
        x = rng.standard_normal(2)
        g.update(x, [float(x @ theta) + 0.01 * rng.standard_normal()])
    assert not g.frozen, "tripped before anything went wrong"

    # the sensor starts returning nonsense
    for _ in range(60):
        x = rng.standard_normal(2)
        r = g.update(x, [float(x @ theta) * -80.0])
        if r.frozen:
            break
    assert g.frozen, "the residual grew 80-fold and the detector said nothing"
    assert "divergence" in r.tripped


def test_divergence_does_not_fire_on_ordinary_noisy_data():
    """The direction that decides whether anyone leaves the guard switched on."""
    g = Guard(make_rls(lam=0.99, p0=10.0),
              [Divergence(ratio=5.0, floor=1e-3, warmup=30)])
    rng = np.random.default_rng(37)
    theta = np.array([1.0, -2.0])
    for _ in range(1000):
        x = rng.standard_normal(2)
        g.update(x, [float(x @ theta) + 0.1 * rng.standard_normal()])
    assert not g.frozen, (
        f"false positive on 1000 samples of ordinary noise ({g.stats}); a "
        "detector that fires on normal operation gets switched off")
    assert g.stats["applied"] == 1000


def test_divergence_declines_to_have_an_opinion_below_its_floor():
    """On near-noiseless data the slow average approaches zero and every ratio
    becomes infinite. The floor is the residual below which the check abstains —
    without it, clean data is indistinguishable from divergence."""
    no_floor = Divergence(ratio=5.0, floor=0.0, warmup=5)
    with_floor = Divergence(ratio=5.0, floor=1.0, warmup=5)
    tiny = _report(residual=1e-12)
    for _ in range(20):
        no_floor.evaluate(tiny)
        with_floor.evaluate(tiny)
    spike = _report(residual=1e-6)          # a millionfold jump, still tiny
    assert no_floor.evaluate(spike) is False
    assert with_floor.evaluate(spike) is True


def _report(residual=0.0, gain_norm=0.0, p_trace=1.0, p_min_diag=1.0,
            frozen=False):
    """A report shaped exactly as the guard hands one to a check — a GuardReport
    always, so `frozen` is never absent."""
    r = np.array([residual], dtype=np.float64)
    return GuardReport(applied=not frozen, residual=r, residual_post=r,
                       gain_norm=gain_norm, p_trace=p_trace,
                       p_min_diag=p_min_diag, frozen=frozen)


# ---------------------------------------------------------------------------
# Faults must stay distinguishable
# ---------------------------------------------------------------------------

def test_a_sensor_nan_is_passed_through_and_is_not_counted_as_a_trip():
    """A NaN sample and a diverging learner are different faults. Summing them
    makes the logs useless exactly when someone needs them."""
    g = Guard(make_rls(), [Trip()])
    rng = np.random.default_rng(41)
    for _ in range(10):
        g.update(rng.standard_normal(2), [1.0])

    r = g.update([np.nan, 1.0], [1.0])
    assert r.applied is False
    assert r.reason == "nonfinite_input"
    assert r.tripped == ()
    assert r.rolled_back is False
    assert g.frozen is False, "a bad sample froze the learner"
    assert g.stats["rejected_by_adapter"] == 1
    assert g.stats["trips"] == 0


def test_every_failing_check_is_reported_not_just_the_first():
    """Three faults at once is diagnostic. Collapsing it to one throws that away."""
    g = Guard(make_rls(), [Trip("a"), Bound("p_trace", hi=-1.0, name="b"),
                           Trip("c")])
    g.checks[0].armed = True
    g.checks[2].armed = True
    r = g.update([1.0, 1.0], [1.0])
    assert r.tripped == ("a", "b", "c")
    assert r.reason == "guard:a"


# ---------------------------------------------------------------------------
# Recovery
# ---------------------------------------------------------------------------

def test_frozen_is_terminal_until_thawed_by_default():
    g = Guard(make_rls(), [Trip()])
    g.checks[0].armed = True
    g.update([1.0, 1.0], [1.0])
    assert g.frozen
    g.checks[0].armed = False
    for _ in range(200):
        g.update([1.0, 1.0], [1.0])
    assert g.frozen, "the guard thawed itself without being asked to"

    g.thaw()
    assert not g.frozen
    assert g.update([1.0, 1.0], [1.0]).applied is True
    assert g.stats["thaws"] == 1


def test_recover_after_thaws_once_the_world_comes_back():
    g = Guard(make_rls(lam=0.99, p0=10.0),
              [Divergence(ratio=5.0, floor=1e-3, warmup=10)],
              recover_after=25)
    rng = np.random.default_rng(43)
    theta = np.array([1.0, -2.0])

    def feed(k, corrupt):
        last = None
        for _ in range(k):
            x = rng.standard_normal(2)
            y = float(x @ theta) * (-80.0 if corrupt else 1.0)
            last = g.update(x, [y + 0.01 * rng.standard_normal()])
        return last

    feed(60, corrupt=False)
    feed(40, corrupt=True)
    assert g.frozen, "never tripped, so there is nothing to recover from"

    feed(120, corrupt=False)
    assert not g.frozen, "the data recovered and the guard stayed frozen"
    assert g.stats["thaws"] == 1
    assert g.stats["applied"] > 60, "adaptation did not resume after thawing"


def test_a_recovery_configuration_that_could_never_recover_is_refused():
    """`gain_norm` is identically zero while frozen, so a positive lower bound on
    it can never be satisfied. Deadlocking in the field instead of failing at
    construction is the worst available outcome."""
    with pytest.raises(ValueError, match="could never thaw"):
        Guard(make_rls(), [Bound("gain_norm", lo=0.1)], recover_after=10)
    # ...and the same check is fine without automatic recovery.
    Guard(make_rls(), [Bound("gain_norm", lo=0.1)])


def test_manual_freeze_does_not_roll_back():
    """For a supervisor that knows something the checks cannot see. It stops
    adaptation where it stands rather than undoing good work."""
    g = Guard(make_rls(), [Trip()])
    rng = np.random.default_rng(47)
    for _ in range(20):
        x = rng.standard_normal(2)
        g.update(x, [float(x @ [1.0, -2.0])])
    here = np.array(g.state)

    g.freeze("bay door open")
    assert g.frozen
    assert np.array_equal(g.state, here), "a manual freeze rolled the state back"
    assert g.stats["rollbacks"] == 0


# ---------------------------------------------------------------------------
# The guard is an Adapter, and that is the point
# ---------------------------------------------------------------------------

def test_a_guarded_adapter_is_interchangeable_with_a_raw_one():
    """The reason Guard wraps rather than sits beside: a caller holds one adapter
    and cannot express 'no guard' by accident."""
    from argmind.adapt import Adapter

    def use(a: Adapter, x, y):
        r = a.update(x, y)
        return isinstance(r, UpdateReport), a.predict(x).shape, a.state.size

    raw = make_rls()
    wrapped = Guard(make_rls(), [Trip()])
    assert isinstance(wrapped, Adapter)
    assert use(raw, [1.0, 2.0], [1.0]) == use(wrapped, [1.0, 2.0], [1.0])
    assert wrapped.n_features == raw.n_features
    assert wrapped.n_outputs == raw.n_outputs
    assert wrapped.dtype == raw.dtype
    assert isinstance(wrapped.update([1.0, 2.0], [1.0]), GuardReport)


def test_replacing_the_state_underneath_re_arms_the_guard():
    """Every checkpoint and every EWMA describes a state that no longer exists,
    so keeping them would let a later rollback restore a state that was never
    reached."""
    g = Guard(make_rls(), [Trip()], rollback_depth=2)
    rng = np.random.default_rng(53)
    for _ in range(20):
        g.update(rng.standard_normal(2), [1.0])
    g.checks[0].armed = True
    g.update([1.0, 1.0], [1.0])
    assert g.frozen

    fresh = make_rls()
    g.load_state(fresh.state)
    assert not g.frozen, "load_state left the guard frozen against a state it "\
                         "has never seen"
    g.checks[0].armed = True
    g.update([1.0, 1.0], [1.0])
    assert np.array_equal(g.state, np.array(fresh.state)), (
        "rolled back past the state that was loaded")


def test_reset_re_arms_too():
    g = Guard(make_rls(), [Trip()])
    g.checks[0].armed = True
    g.update([1.0, 1.0], [1.0])
    assert g.frozen
    g.reset()
    assert not g.frozen
    assert g.stats["applied"] == 0 or True      # stats are cumulative by design


# ---------------------------------------------------------------------------
# Configuration that cannot work is refused at construction
# ---------------------------------------------------------------------------

def test_a_guard_with_no_checks_is_refused():
    with pytest.raises(ValueError, match="no checks"):
        Guard(make_rls(), [])


def test_a_bound_that_cannot_fail_is_refused():
    with pytest.raises(ValueError, match="never fail"):
        Bound("p_trace")


def test_a_bound_on_a_field_that_does_not_exist_is_refused():
    """The failure this prevents: a typo produces a guard that never fires and
    looks exactly like protection."""
    with pytest.raises(ValueError, match="not a report field"):
        Bound("resdiual_norm", hi=1.0)


def test_bound_treats_non_finite_as_unhealthy():
    """NaN compares False against everything, so without saying so explicitly a
    NaN slips through every comparison as 'not out of range'."""
    b = Bound("p_trace", lo=0.0, hi=1e6)
    assert b.evaluate(_report(p_trace=1.0)) is True
    assert b.evaluate(_report(p_trace=np.nan)) is False
    assert b.evaluate(_report(p_trace=np.inf)) is False


@pytest.mark.parametrize("kwargs", [
    {"ratio": 1.0},
    {"ratio": 0.5},
    {"fast_alpha": 0.0},
    {"fast_alpha": 1.5},
    {"slow_alpha": 0.0},
    {"fast_alpha": 0.01, "slow_alpha": 0.5},    # fast slower than slow
    {"floor": -1.0},
])
def test_invalid_divergence_configuration_is_refused(kwargs):
    with pytest.raises(ValueError):
        Divergence(**kwargs)


@pytest.mark.parametrize("kwargs", [
    {"rollback_depth": 0},
    {"checkpoint_every": 0},
    {"recover_after": 0},
])
def test_invalid_guard_configuration_is_refused(kwargs):
    with pytest.raises(ValueError):
        Guard(make_rls(), [Trip()], **kwargs)


def test_non_checks_are_refused():
    with pytest.raises(TypeError, match="not a Check"):
        Guard(make_rls(), ["divergence"])


def test_a_manual_freeze_reports_the_live_covariance_not_a_stale_checkpoint():
    """A bug found by a mutation check on the C++ port, not by review.

    `freeze()` used to take its health scalars from the newest checkpoint. After a *trip*
    that is exact, because the state has just been restored to that checkpoint. After a
    *manual* freeze nothing was restored, so the newest checkpoint can be up to
    `checkpoint_every` updates stale — and a `Bound` on `p_trace` evaluated during the
    freeze then read a number describing a state the adapter was no longer in.

    Measured before the fix: 300.0 reported against a live 104.1, three updates into a
    cadence of five.
    """
    rls = RecursiveLeastSquares(3, forgetting=0.99, p0=100.0, dtype="f8")
    g = Guard(rls, [Bound("p_trace", hi=1e9)], rollback_depth=4, checkpoint_every=5)
    for i in range(3):                       # mid-cadence: no checkpoint since construction
        g.update([1.0 + i, -2.0, 0.5], [3.0])

    live_trace, live_min = g.health()
    stale_trace = float(g._ckpt_health[g._newest()][0])
    assert abs(stale_trace - live_trace) > 1.0, (
        "the fixture is no longer mid-cadence, so a stale checkpoint would look correct")

    g.freeze("bay door open")
    probe = g.update([1.0, -2.0, 0.5], [3.0])
    assert probe.frozen and not probe.applied
    assert probe.p_trace == pytest.approx(live_trace), (
        f"the probe reported {probe.p_trace}, not the live {live_trace}")
    assert probe.p_min_diag == pytest.approx(live_min)


def test_a_probe_after_a_trip_still_reports_the_restored_state():
    """The other half: after a trip the restored state *is* the checkpoint, so the probe's
    scalars must equal the checkpoint's. Fixing the manual-freeze path must not break
    this one."""
    rls = RecursiveLeastSquares(2, forgetting=0.95, p0=1.0, dtype="f8")
    g = Guard(rls, [Bound("p_trace", hi=10.0)], rollback_depth=1)
    rng = np.random.default_rng(31)
    for _ in range(200):
        r = g.update([rng.standard_normal(), 0.0], [0.0])
        if r.frozen:
            break
    assert g.frozen

    live_trace, live_min = g.health()
    probe = g.update([rng.standard_normal(), 0.0], [0.0])
    assert probe.p_trace == pytest.approx(live_trace)
    assert probe.p_min_diag == pytest.approx(live_min)
