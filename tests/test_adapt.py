# SPDX-License-Identifier: Apache-2.0
"""Tests for the Adapter contract and recursive least squares.

Same standard as the stream/align suite: a test that can only pass is not
evidence. Several here exist to prove that the *failures* are visible —
covariance windup under poor excitation, a NaN sample that must not be absorbed
silently, a corrupt persisted state, and the precision floor that was measured
rather than inherited.

The correctness anchor is deliberately external. `test_matches_the_weighted_batch
_solution` compares against the closed-form exponentially weighted least squares
solution built independently from the same data, so the recursion is checked
against mathematics rather than against itself.
"""

import numpy as np
import pytest

from argmind.adapt import NormalisedLeastMeanSquares, RecursiveLeastSquares, UpdateReport


def batch_ewls(xs, ys, lam, p0, n, m):
    """Closed-form exponentially weighted least squares, the oracle.

    Solves ``(sum_i lam^(k-i) x_i x_i^T + lam^k P0^-1) theta = sum_i lam^(k-i) x_i y_i^T``
    directly. This is what the recursion is supposed to be computing, derived
    independently of it.
    """
    k = len(xs)
    R = np.eye(n) / p0 * lam ** k
    b = np.zeros((n, m))
    for i, (x, y) in enumerate(zip(xs, ys), start=1):
        w = lam ** (k - i)
        R += w * np.outer(x, x)
        b += w * np.outer(x, y)
    return np.linalg.solve(R, b)


@pytest.fixture
def clean_problem():
    """A well-posed 4-feature, 1-output problem with a known answer."""
    rng = np.random.default_rng(11)
    n, m, k = 4, 1, 120
    theta = np.array([[1.5], [-0.5], [2.0], [0.25]])
    xs = rng.standard_normal((k, n))
    ys = xs @ theta
    return n, m, xs, ys, theta


# ---------------------------------------------------------------------------
# Correctness, against an external oracle
# ---------------------------------------------------------------------------

def test_recovers_known_parameters_up_to_the_prior_it_was_given(clean_problem):
    """On noiseless data the estimate does *not* reach the true parameters, and
    the gap is not numerical slop — it is the prior.

    ``p0`` is a prior covariance, so the recursion solves a ridge problem:
    ``(sum x x^T + I/p0) theta = sum x y``. The bias that leaves behind is
    ``O(1 / (p0 * lambda_min(sum x x^T)))``. Asserting a tight tolerance and
    calling it convergence would hide that; instead the bias is shown to scale
    as ``1/p0`` across four decades, which is what proves it is the prior and
    not an error in the recursion.
    """
    n, m, xs, ys, theta = clean_problem
    errs = {}
    for p0 in (1e1, 1e3, 1e5, 1e8):
        rls = RecursiveLeastSquares(n_features=n, forgetting=1.0, p0=p0, dtype="f8")
        for x, y in zip(xs, ys):
            assert rls.update(x, y).applied
        errs[p0] = float(np.linalg.norm(rls.theta - theta))
        assert rls.stats["updates"] == len(xs)

    assert errs[1e8] < 1e-9, f"never approached the true parameters: {errs}"
    # The 1/p0 law is first order; the departure from it is the next term,
    # O(1/(p0*lambda_min)) ~ 1e-3 at the smallest p0 here, which is what the
    # tolerance allows for. Tighter than this would be asserting an exactness
    # the expansion does not have.
    for a, b in ((1e1, 1e3), (1e3, 1e5), (1e5, 1e8)):
        assert errs[a] / errs[b] == pytest.approx(b / a, rel=5e-3), (
            f"residual error does not scale as 1/p0 ({errs}); it is therefore "
            "not the prior term, and something else is biasing the estimate")


@pytest.mark.parametrize("lam", [1.0, 0.99, 0.9])
def test_matches_the_weighted_batch_solution(clean_problem, lam):
    """The recursion must equal the closed form it claims to compute.

    This is the test that makes the C++ port safe later: the Python reference is
    only a useful oracle if it is itself anchored to something external.
    """
    n, m, xs, ys, _ = clean_problem
    p0 = 1e3
    rls = RecursiveLeastSquares(n_features=n, forgetting=lam, p0=p0, dtype="f8")
    for x, y in zip(xs, ys):
        rls.update(x, y)
    expect = batch_ewls(xs, ys, lam, p0, n, m)
    assert np.allclose(rls.theta, expect, rtol=1e-7, atol=1e-9), (
        f"recursion and closed form disagree at lambda={lam}: "
        f"{rls.theta.ravel()} vs {expect.ravel()}")


def test_a_posteriori_error_never_exceeds_a_priori():
    """An exact property of the update: ``e_post = e * lam / (lam + x'Px)``.

    It holds for every sample, so a violation means the gain, the residual, or
    the ordering inside `update` is wrong — the kind of bug that still converges
    on friendly data and fails on real data.
    """
    rng = np.random.default_rng(4)
    rls = RecursiveLeastSquares(n_features=5, forgetting=0.97, dtype="f8")
    for _ in range(200):
        r = rls.update(rng.standard_normal(5), rng.standard_normal(1))
        assert r.applied
        assert r.residual_post_norm <= r.residual_norm + 1e-12, (
            "the update increased the error on the sample it just consumed")


def test_multi_output_equals_independent_single_output_adapters():
    """Sharing one P across outputs must be an implementation detail, not a
    behaviour change. If this ever fails, the shared covariance is wrong."""
    rng = np.random.default_rng(19)
    n, k = 3, 80
    xs = rng.standard_normal((k, n))
    ys = rng.standard_normal((k, 2))

    both = RecursiveLeastSquares(n_features=n, n_outputs=2, forgetting=0.98,
                                 dtype="f8")
    singles = [RecursiveLeastSquares(n_features=n, forgetting=0.98, dtype="f8")
               for _ in range(2)]
    for x, y in zip(xs, ys):
        both.update(x, y)
        for j, s in enumerate(singles):
            s.update(x, y[j])

    for j, s in enumerate(singles):
        assert np.allclose(both.theta[:, j], s.theta[:, 0], rtol=1e-10, atol=1e-12)


# ---------------------------------------------------------------------------
# Forgetting: it has to actually forget, and the cost has to be visible
# ---------------------------------------------------------------------------

def test_forgetting_tracks_a_parameter_jump_and_infinite_memory_does_not():
    """The negative half matters as much as the positive one: lambda=1 is
    *supposed* to fail here, and a suite that only showed lambda=0.9 succeeding
    would not be evidence that forgetting is doing anything."""
    rng = np.random.default_rng(23)
    n = 3
    before = np.array([[1.0], [0.0], [-1.0]])
    after = np.array([[-2.0], [3.0], [0.5]])

    fast = RecursiveLeastSquares(n_features=n, forgetting=0.9, p0=1e2, dtype="f8")
    slow = RecursiveLeastSquares(n_features=n, forgetting=1.0, p0=1e2, dtype="f8")
    for i in range(400):
        x = rng.standard_normal(n)
        theta = before if i < 300 else after
        y = x @ theta
        fast.update(x, y)
        slow.update(x, y)

    err_fast = np.linalg.norm(fast.theta - after)
    err_slow = np.linalg.norm(slow.theta - after)

    # 100 samples at lambda=0.9 leaves the pre-jump data weighted by 0.9**100
    # ~ 2.7e-5, so a residual error near 1e-4 is the expected tail rather than a
    # failure to converge. Asserting 0 here would be asserting the wrong thing.
    assert err_fast < 1e-3, f"forgetting adapter did not re-converge: {err_fast:.3e}"
    assert err_slow > 1.0, (
        f"lambda=1.0 tracked a parameter jump it has no mechanism to track "
        f"(error {err_slow:.3e}); if infinite memory now adapts, the forgetting "
        "factor is not what is driving adaptation and this comparison is void")
    assert err_slow > 100 * err_fast, (
        f"lambda=1.0 tracked the jump about as well as lambda=0.9 "
        f"(errors {err_slow:.3e} vs {err_fast:.3e})")


def test_memory_horizon_is_reported_and_infinite_at_lambda_one():
    assert RecursiveLeastSquares(2, forgetting=0.99).memory_horizon_samples == \
        pytest.approx(100.0)
    assert RecursiveLeastSquares(2, forgetting=1.0).memory_horizon_samples == \
        float("inf")


# ---------------------------------------------------------------------------
# The failures this library exists because of
# ---------------------------------------------------------------------------

def test_covariance_windup_under_poor_excitation_is_visible_and_then_bursts():
    """The stationary-vehicle failure, reproduced.

    With one direction never excited, P grows like lambda**-k there, and the
    first sample that finally arrives off-axis produces a burst. This class does
    not clamp it — that is the Guard's job — so what is asserted here is that the
    growth is *observable* in the report and that the burst is real rather than
    hypothetical.
    """
    lam = 0.95
    rng = np.random.default_rng(31)
    rls = RecursiveLeastSquares(n_features=2, forgetting=lam, p0=1.0, dtype="f8")

    first = rls.update([1.0, 0.0], [1.0])
    traces = [first.p_trace]
    for _ in range(200):
        # excitation confined to the first axis; the second learns nothing
        r = rls.update([rng.standard_normal(), 0.0], [0.0])
        traces.append(r.p_trace)

    assert traces[-1] > 1e4 * traces[0], (
        f"covariance did not wind up ({traces[0]:.3e} -> {traces[-1]:.3e}); the "
        "unexcited direction is being bounded somewhere it should not be")

    # The growth law, not just the direction of it. The unexcited diagonal gets
    # divided by lambda every update and touched by nothing else, so after k
    # updates it is exactly p0 * lambda**-k. Pinning the law rather than a
    # threshold means a later change that merely slows the windup still fails.
    p_unexcited = float(np.array(rls.state)[2:].reshape(2, 2)[1, 1])
    assert p_unexcited == pytest.approx(1.0 * lam ** -201, rel=1e-9), (
        f"unexcited covariance is {p_unexcited:.6e}, not p0*lambda**-k = "
        f"{lam ** -201:.6e}")
    assert np.all(np.diff(traces[100:]) > 0), (
        "windup stopped being monotonic once it dominated the trace")

    # ...and the burst. One off-axis sample against a covariance that huge moves
    # the parameter by orders of magnitude more than the data justifies.
    before = rls.theta[1, 0]
    burst = rls.update([0.0, 1.0], [1.0])
    after = rls.theta[1, 0]
    assert abs(after - before) > 0.99, (
        "the wound-up direction absorbed a single sample almost completely, "
        f"moving only {abs(after - before):.3e} — expected a near-total jump")
    assert burst.gain_norm > 0.99


def test_nonfinite_input_is_rejected_counted_and_leaves_the_state_untouched():
    """A NaN from a sensor is a runtime event, not a programming error. It must
    neither crash the loop nor disappear into the estimate."""
    rls = RecursiveLeastSquares(n_features=3, dtype="f8")
    rng = np.random.default_rng(5)
    for _ in range(10):
        rls.update(rng.standard_normal(3), rng.standard_normal(1))

    snapshot = np.array(rls.state)
    for bad_x, bad_y in (([np.nan, 1.0, 2.0], [1.0]),
                         ([1.0, 2.0, 3.0], [np.inf])):
        r = rls.update(bad_x, bad_y)
        assert r.applied is False
        assert r.reason == "nonfinite_input"
        assert np.array_equal(rls.state, snapshot), (
            "a rejected update still modified the adaptation state")
    assert rls.stats["rejected_nonfinite"] == 2
    assert rls.stats["updates"] == 10, "a rejected update was counted as applied"


def test_a_covariance_that_lost_definiteness_is_refused_not_propagated():
    """Reachable in the field via a corrupt persisted block or precision loss.
    The gain would be meaningless, so the update is refused and named."""
    rls = RecursiveLeastSquares(n_features=2, forgetting=0.99, dtype="f8")
    broken = np.array(rls.state)
    broken[2:] = np.array([-1.0, 0.0, 0.0, -1.0])   # P = -I
    rls.load_state(broken)

    r = rls.update([3.0, 0.0], [1.0])
    assert r.applied is False
    assert r.reason == "nonpositive_denominator"
    assert r.p_min_diag < 0, "p_min_diag did not expose the indefinite covariance"
    assert rls.stats["rejected_denominator"] == 1


def test_shape_errors_raise_but_value_errors_do_not():
    """The distinction is deliberate: a wrong shape is a bug at the call site and
    should fail loudly; a wrong value arrived from a sensor and should not."""
    rls = RecursiveLeastSquares(n_features=3, dtype="f8")
    with pytest.raises(ValueError, match="3 features"):
        rls.update([1.0, 2.0], [1.0])
    with pytest.raises(ValueError, match="1 outputs"):
        rls.update([1.0, 2.0, 3.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="3 features"):
        rls.predict([1.0])
    assert rls.update([np.nan, 0.0, 0.0], [0.0]).applied is False


# ---------------------------------------------------------------------------
# Precision: measured here, not inherited from the parked project
# ---------------------------------------------------------------------------

def _recursion_in(dtype, kappa, n=6, steps=300, lam=0.98, p0=1e2, seed=7):
    """The same recursion as `RecursiveLeastSquares`, run at an arbitrary dtype
    so precisions the class refuses can still be measured.

    Every regressor is held inside fp16's range (max |x| = 30) and only the
    *conditioning* of the regressor correlation varies, so a failure here is the
    recursion's and not the format's dynamic range. Returns the first step at
    which P lost positive-definiteness or the state went non-finite, or None.
    """
    rng = np.random.default_rng(seed)
    scale = np.logspace(0, np.log10(kappa), n)
    scale = scale / scale.max() * 30.0
    theta_true = np.linspace(1.0, 2.0, n)
    dt = np.dtype(dtype)
    P = np.eye(n, dtype=dt) * dt.type(p0)
    th = np.zeros(n, dtype=dt)

    for k in range(steps):
        x = (rng.standard_normal(n) * scale).astype(dt)
        y = dt.type(float(x.astype(np.float64) @ theta_true))
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            Px = P @ x
            den = dt.type(lam) + x @ Px
            g = Px / den
            th = th + g * (y - x @ th)
            P = (P - np.outer(g, Px)) / dt.type(lam)
            P = (P + P.T) / dt.type(2)
        if not (np.all(np.isfinite(P)) and np.all(np.isfinite(th))):
            return k
        if float(np.linalg.eigvalsh(P.astype(np.float64)).min()) < 0.0:
            return k
    return None


def test_fp16_refusal_is_backed_by_measurement():
    """fp32 is the declared floor. This is why, measured for RLS specifically.

    DESIGN.md's fp32 finding was measured on the EKF covariance recursion, which
    is evidence about that recursion and not this one. Carrying the conclusion
    across would be the borrowed-assumption error the project's standards name,
    so it was re-measured. The result: fp16 loses positive-definiteness even when
    the regressor correlation is perfectly conditioned and nothing is near the
    format's range limit — so the failure is arithmetic, not dynamic range.
    """
    assert _recursion_in("f2", kappa=1.0) is not None, (
        "fp16 survived a workload it was measured failing; if this now holds, "
        "the constructor's refusal of fp16 needs re-deriving, not relaxing")
    assert _recursion_in("f2", kappa=10.0) is not None

    # The refusal must be specific, not a blanket suspicion of low precision.
    assert _recursion_in("f4", kappa=1.0) is None
    assert _recursion_in("f4", kappa=1e3) is None
    assert _recursion_in("f8", kappa=1e3) is None


def test_fp32_is_the_floor_and_fp16_is_refused_at_construction():
    with pytest.raises(ValueError, match="not supported"):
        RecursiveLeastSquares(n_features=3, dtype="f2")
    for dt in ("f4", "f8"):
        assert RecursiveLeastSquares(n_features=3, dtype=dt).dtype == np.dtype(dt)


def test_fp32_does_not_maintain_positive_definiteness_at_large_p0_times_x():
    """The uncomfortable half of the precision finding, protected by a test.

    fp32 is the floor but it is not a comfortable one: with a large prior
    covariance and large regressors, the conventional form shows a negative
    eigenvalue at roundoff scale on the very first update. It does not blow up,
    and the parameters stay sound — but 'P stays PSD' is not a property this form
    has in fp32, and a later change must not be able to quietly claim otherwise.
    """
    rls = RecursiveLeastSquares(n_features=6, forgetting=0.98, p0=1e2, dtype="f4")
    rng = np.random.default_rng(3)
    worst = np.inf
    for _ in range(50):
        rls.update(rng.standard_normal(6) * 1e3, [1.0])
        P = rls.state[6:].reshape(6, 6).astype(np.float64)
        worst = min(worst, float(np.linalg.eigvalsh(P).min()))
    assert worst < 0.0, (
        "fp32 held positive-definiteness where it was measured not to; the "
        "precision note in argmind/adapt.py is now wrong and must be re-measured")
    assert worst > -1e-3, (
        f"the indefiniteness is no longer roundoff-scale ({worst:.3e}); that is a "
        "different and much worse failure than the one documented")


# ---------------------------------------------------------------------------
# The state block: persistence and the C ABI both depend on this shape
# ---------------------------------------------------------------------------

def test_state_is_one_flat_contiguous_block_of_the_declared_size():
    rls = RecursiveLeastSquares(n_features=4, n_outputs=3, dtype="f4")
    s = rls.state
    assert s.ndim == 1
    assert s.size == 4 * 3 + 4 * 4 == rls.state_size
    assert s.flags["C_CONTIGUOUS"]
    assert s.dtype == np.dtype("f4")


def test_state_view_is_read_only_so_it_cannot_be_edited_behind_validation():
    rls = RecursiveLeastSquares(n_features=2, dtype="f8")
    with pytest.raises(ValueError):
        rls.state[0] = 1.0


def test_state_round_trips_and_restores_behaviour_exactly():
    """Adaptation state that does not survive a power cycle is useless in the
    field, so 'restored' has to mean bit-identical behaviour, not approximately."""
    rng = np.random.default_rng(7)
    xs = rng.standard_normal((100, 3))
    ys = rng.standard_normal((100, 1))

    rls = RecursiveLeastSquares(n_features=3, forgetting=0.98, dtype="f8")
    for x, y in zip(xs[:50], ys[:50]):
        rls.update(x, y)
    saved = np.array(rls.state)          # what would go to flash

    for x, y in zip(xs[50:], ys[50:]):
        rls.update(x, y)
    moved = rls.predict(xs[0])

    restored = RecursiveLeastSquares(n_features=3, forgetting=0.98, dtype="f8")
    restored.load_state(saved)
    for x, y in zip(xs[50:], ys[50:]):
        restored.update(x, y)
    assert np.array_equal(restored.predict(xs[0]), moved), (
        "a restored adapter did not reproduce the un-interrupted one")


def test_corrupt_persisted_state_is_refused_rather_than_loaded():
    """Flash comes back truncated or bit-flipped; that is the whole reason to
    validate. A stored NaN loaded silently poisons every later update."""
    rls = RecursiveLeastSquares(n_features=3, dtype="f8")
    good = np.array(rls.state)

    with pytest.raises(ValueError, match="elements, expected"):
        rls.load_state(good[:-1])
    with pytest.raises(ValueError, match="elements, expected"):
        rls.load_state(np.zeros(good.size + 4))

    bad = good.copy()
    bad[0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        rls.load_state(bad)
    assert np.array_equal(rls.state, good), "a refused load still mutated the state"


def test_reset_returns_to_construction_state():
    rls = RecursiveLeastSquares(n_features=3, p0=5.0, dtype="f8")
    fresh = np.array(rls.state)
    rng = np.random.default_rng(2)
    for _ in range(20):
        rls.update(rng.standard_normal(3), rng.standard_normal(1))
    assert not np.array_equal(rls.state, fresh)
    rls.reset()
    assert np.array_equal(rls.state, fresh)


def test_theta_is_a_copy_and_does_not_alias_the_state():
    rls = RecursiveLeastSquares(n_features=2, dtype="f8")
    rls.update([1.0, 1.0], [1.0])
    held = rls.theta
    before = held.copy()
    for _ in range(10):
        rls.update([1.0, -1.0], [5.0])
    assert np.array_equal(held, before), "theta aliased the live state"


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs", [
    {"n_features": 0},
    {"n_features": 3, "n_outputs": 0},
    {"n_features": 3, "forgetting": 0.0},
    {"n_features": 3, "forgetting": 1.5},
    {"n_features": 3, "forgetting": -0.5},
    {"n_features": 3, "p0": 0.0},
    {"n_features": 3, "p0": -1.0},
])
def test_invalid_construction_fails_loudly(kwargs):
    with pytest.raises(ValueError):
        RecursiveLeastSquares(**kwargs)


def test_report_carries_what_a_guard_will_need():
    """The Guard is next and consumes exactly these fields. If the report shrinks,
    the guard cannot be written without reaching into the adapter's internals."""
    rls = RecursiveLeastSquares(n_features=2, dtype="f8")
    r = rls.update([1.0, 2.0], [3.0])
    assert isinstance(r, UpdateReport)
    for field in ("applied", "residual", "residual_post", "gain_norm",
                  "p_trace", "p_min_diag", "reason"):
        assert hasattr(r, field)
    assert r.residual.shape == (1,)
    assert r.p_trace > 0
    assert r.reason is None


def test_a_non_finite_p0_is_refused_rather_than_producing_a_dead_adapter():
    """`not p0 > 0.0` lets `+inf` through, and for a while it did.

    An independent parity check found the disagreement: `Rls` in C++ has always tested
    `is_finite(p0)` and this reference never did, so the two differed on exactly one cell
    of a 108-point sweep over lambda and p0.

    What got through is worse than a wrong number. `eye(n) * inf` is `inf` on the diagonal
    and `0 * inf = NaN` off it, so the adapter is born with a non-finite covariance, every
    denominator is NaN, and **every update for the rest of its life is rejected**. It never
    complains at construction. numpy emits a RuntimeWarning, which is the kind nobody reads.
    """
    with pytest.raises(ValueError, match="positive and finite"):
        RecursiveLeastSquares(3, p0=float("inf"), dtype="f8")

    # ...and the neighbours still behave, so this is one cell and not a blanket tightening.
    for bad in (float("-inf"), float("nan"), 0.0, -1.0):
        with pytest.raises(ValueError, match="p0 must be positive"):
            RecursiveLeastSquares(3, p0=bad, dtype="f8")
    RecursiveLeastSquares(3, p0=1e-300, dtype="f8")     # denormal, but finite and positive
    RecursiveLeastSquares(3, p0=1e300, dtype="f8")

    # `forgetting` needs no separate finiteness clause: the range test already covers it.
    for bad in (float("inf"), float("nan")):
        with pytest.raises(ValueError, match="forgetting must be"):
            RecursiveLeastSquares(3, forgetting=bad, dtype="f8")


def test_the_dead_adapter_a_non_finite_p0_used_to_produce_is_what_makes_it_worth_refusing():
    """The refusal above is only justified if what it prevents is actually bad. This builds
    that state deliberately -- past the constructor, the way the old code let it happen --
    and shows the adapter rejects every sample forever."""
    a = RecursiveLeastSquares(3, p0=100.0, dtype="f8")
    with np.errstate(invalid="ignore"):              # the 0*inf IS the demonstration
        a._p[:] = np.eye(3) * np.inf                 # what `p0=inf` used to construct
    assert not np.isfinite(np.asarray(a.state)).all()
    for _ in range(5):
        r = a.update([1.0, 2.0, 3.0], [1.0])
        assert r.applied is False
        assert r.reason == "nonpositive_denominator"


# -- the second adapter -----------------------------------------------------------

def test_nlms_projects_a_sample_out_exactly_at_mu_one():
    """The exact law, which is what makes this a verification rather than a comparison.

    After an update the a posteriori error is exactly `(1 - mu*x.x/(eps + x.x))` times
    the a priori one, for any data whatsoever. At mu=1 with negligible eps that is zero.
    Asserting the law rather than a tolerance is the same reason RLS is checked against
    the closed-form weighted solution instead of against itself.
    """
    rng = np.random.default_rng(3)
    true = np.array([2.0, -1.0, 0.5, 0.25])
    for mu in (0.25, 0.5, 1.0, 1.5, 2.0):
        a = NormalisedLeastMeanSquares(4, 1, mu=mu, eps=1e-12, dtype="f8")
        worst = 0.0
        for _ in range(200):
            x = rng.standard_normal(4)
            r = a.update(x, [float(x @ true)])
            xx = float(x @ x)
            want = (1.0 - mu * xx / (1e-12 + xx)) * float(r.residual[0])
            worst = max(worst, abs(float(r.residual_post[0]) - want))
        assert worst < 1e-12, f"the law does not hold at mu={mu}: {worst:.3e}"


def test_nlms_converges_and_is_a_quarter_of_the_state_rls_needs():
    a = NormalisedLeastMeanSquares(4, 1, mu=1.0, eps=1e-12, dtype="f8")
    rng = np.random.default_rng(7)
    true = np.array([2.0, -1.0, 0.5, 0.25])
    for _ in range(200):
        x = rng.standard_normal(4)
        a.update(x, [float(x @ true)])
    x = rng.standard_normal(4)
    assert a.predict(x)[0] == pytest.approx(float(x @ true), abs=1e-9)
    # The reason this adapter exists: n*m against RLS's n*m + n*n.
    assert a.state.size == 4
    assert RecursiveLeastSquares(4, 1, dtype="f8").state.size == 20


def test_nlms_rejects_bad_values_rather_than_raising():
    """The same contract RLS has: shape errors raise, bad values are rejected, counted
    and reported. One vocabulary, so a guard's counters mean the same thing whichever
    adapter is underneath."""
    a = NormalisedLeastMeanSquares(3, 1, mu=0.5, eps=1e-6, dtype="f4")
    before = np.array(a.state)

    r = a.update([1.0, np.nan, 3.0], [1.0])
    assert r.applied is False and r.reason == "nonfinite_input"
    assert np.array_equal(a.state, before)

    # Finite input reaching the denominator path: 1e30 squares to infinity in f32.
    with np.errstate(over="ignore"):
        r = a.update([1e30, 1e30, 1e30], [1.0])
    assert r.applied is False and r.reason == "nonpositive_denominator"
    assert np.array_equal(a.state, before)

    with pytest.raises(ValueError, match="must have 3 features"):
        a.update([1.0, 2.0], [1.0])


@pytest.mark.parametrize("mu", [0.0, -0.1, 2.1, float("inf"), float("nan")])
def test_nlms_refuses_a_step_size_outside_its_stability_bound(mu):
    with pytest.raises(ValueError, match="mu must be in"):
        NormalisedLeastMeanSquares(3, mu=mu, dtype="f8")
    NormalisedLeastMeanSquares(3, mu=2.0, dtype="f8")     # the boundary is inclusive


@pytest.mark.parametrize("eps", [0.0, -1.0, float("inf"), float("nan")])
def test_nlms_refuses_an_eps_that_would_not_guard_the_zero_regressor(eps):
    with pytest.raises(ValueError, match="eps must be positive"):
        NormalisedLeastMeanSquares(3, eps=eps, dtype="f8")


def test_nlms_does_nothing_on_an_all_zero_regressor_rather_than_dividing_by_zero():
    """What `eps` is for. A silent stretch of input should leave the model alone."""
    a = NormalisedLeastMeanSquares(3, mu=1.0, eps=1e-6, dtype="f8")
    a.update([1.0, 2.0, 3.0], [1.0])
    before = np.array(a.state)
    r = a.update([0.0, 0.0, 0.0], [5.0])
    assert r.applied is True                    # accepted...
    assert np.array_equal(a.state, before)      # ...and changed nothing


def test_nlms_declares_no_covariance_so_a_guard_refuses_to_watch_one():
    from argmind.guard import Bound, Guard
    a = NormalisedLeastMeanSquares(4, dtype="f8")
    assert a.health_fields == frozenset(
        {"residual_norm", "residual_post_norm", "gain_norm"})
    with pytest.raises(ValueError, match="does not report"):
        Guard(a, [Bound("p_trace", hi=1e6)])
    Guard(NormalisedLeastMeanSquares(4, dtype="f8"), [Bound("residual_norm", hi=1e6)])


# -- keeping what was learned while adding capacity ----------------------------------
#
# There are two reasons to widen and they want OPPOSITE answers, which is why
# `keep_confidence` is keyword-only with no default. Both are tested here, and each
# asserts the direction rather than a tolerance, because a bound that both cases pass
# would be measuring nothing.


def _trained_linear(n=6, count=800, seed=1, p0=100.0):
    """New INPUTS case: the truth is linear in twelve features and this sees the first
    six, so its coefficients on those six are CORRECT -- nothing is being absorbed."""
    rng = np.random.default_rng(7)
    true = rng.standard_normal(12)
    g = np.random.default_rng(seed)
    a = RecursiveLeastSquares(n, 1, forgetting=0.999, p0=p0, dtype="f8")
    for _ in range(count):
        x = g.standard_normal(n)
        a.update(x, [float(x @ true[:n]) + 0.01 * g.standard_normal()])
    return a, true


def _truth_nonlinear(s):
    return (2.0 * s[0] - 1.5 * s[1] + 0.5 * s[2]
            + 0.8 * s[0] * s[0] - 1.2 * s[0] * s[1] + 0.3)


def _linear_basis(s):
    return np.array([s[0], s[1], s[2], 1.0])


def _rich_basis(s):
    return np.array([s[0], s[1], s[2], 1.0, s[0] ** 2, s[1] ** 2, s[0] * s[1]])


def test_widening_carries_theta_and_optionally_the_covariance():
    old, _ = _trained_linear()
    ot = np.asarray(old.state)

    for keep in (True, False):
        wide = old.widened(12, keep_confidence=keep)
        wt = np.asarray(wide.state)
        assert wide.n_features == 12 and wt.size == 12 + 144
        assert np.array_equal(wt[:6], ot[:6]), "theta is carried either way"
        assert np.all(wt[6:12] == 0.0), "a new coefficient starts at zero"
        P = wt[12:].reshape(12, 12)
        assert np.allclose(np.diag(P)[6:], 100.0), "new features must get the prior"
        assert np.all(P[:6, 6:] == 0.0), "cross terms claim evidence that does not exist"
        if keep:
            assert np.array_equal(P[:6, :6], ot[6:].reshape(6, 6))
        else:
            assert np.allclose(P[:6, :6], np.eye(6) * 100.0), (
                "keep_confidence=False must leave the prior everywhere")


def test_widening_is_keyword_only_because_a_default_would_be_a_trap():
    old, _ = _trained_linear()
    with pytest.raises(TypeError):
        old.widened(12)                       # no default: the caller has to decide
    with pytest.raises(ValueError, match="only appends"):
        old.widened(4, keep_confidence=False)
    with pytest.raises(ValueError, match="cannot change the number of outputs"):
        old.widened(12, keep_confidence=False, n_outputs=3)


def test_new_inputs_want_the_confidence_carried():
    """The old coefficients are still correct, so the earned certainty is worth keeping
    -- and it is the certainty that buys the head start, not the coefficients. Measured:
    fresh 0.410, keep_confidence=True 0.0010, False 0.327 at ten updates."""
    old, true = _trained_linear()
    fresh = RecursiveLeastSquares(12, 1, forgetting=0.999, p0=100.0, dtype="f8")
    kept = old.widened(12, keep_confidence=True)
    dropped = old.widened(12, keep_confidence=False)

    g = np.random.default_rng(3)
    for _ in range(10):
        x = g.standard_normal(12)
        y = float(x @ true) + 0.01 * g.standard_normal()
        for m in (fresh, kept, dropped):
            m.update(x, [y])

    mask = np.zeros(12)
    mask[:6] = 1.0
    probe = np.random.default_rng(50).standard_normal((200, 12)) * mask
    want = probe @ true

    def err(m):
        return float(np.linalg.norm(probe @ np.asarray(m.state)[:12] - want) / np.sqrt(200))

    assert err(kept) < err(fresh) / 100.0, (
        f"carrying the confidence bought little: kept {err(kept):.5f} against fresh "
        f"{err(fresh):.5f}")
    assert err(dropped) > err(kept) * 50.0, (
        "dropping the confidence performed as well as keeping it, so this case does not "
        "distinguish the two and the API's choice is unmotivated")


def test_a_richer_basis_wants_the_confidence_dropped():
    """The opposite case, and the one that makes a default impossible.

    A misspecified fit is **biased**, not merely incomplete: fitting a truth with an
    `0.8 s0^2` term using a linear basis lands the constant at 1.136 rather than 0.3,
    having absorbed `E[0.8 s0^2] = 0.8`, and `tr(P)` falls to 0.0032. Carrying that
    certainty makes the adapter sure of wrong values and it resists correction.
    """
    g = np.random.default_rng(4)
    lin = RecursiveLeastSquares(4, 1, forgetting=0.9995, p0=100.0, dtype="f8")
    for _ in range(2000):
        s = g.standard_normal(3)
        lin.update(_linear_basis(s), [_truth_nonlinear(s)])

    # the bias is the whole mechanism, so it is asserted rather than described
    assert abs(float(np.asarray(lin.state)[3]) - 0.3) > 0.5, (
        "the linear fit did not absorb the quadratic term, so this fixture does not "
        "demonstrate a misspecified fit at all")
    assert float(np.trace(np.asarray(lin.state)[4:].reshape(4, 4))) < 0.1

    fresh = RecursiveLeastSquares(7, 1, forgetting=0.9995, p0=100.0, dtype="f8")
    kept = lin.widened(7, keep_confidence=True)
    dropped = lin.widened(7, keep_confidence=False)

    g2 = np.random.default_rng(4)
    for _ in range(2000):
        s = g2.standard_normal(3)
        b, y = _rich_basis(s), [_truth_nonlinear(s)]
        for m in (fresh, kept, dropped):
            m.update(b, y)

    h = np.random.default_rng(99)
    pts = h.standard_normal((500, 3))

    def rmse(m):
        return float(np.sqrt(np.mean([(m.predict(_rich_basis(x))[0] - _truth_nonlinear(x)) ** 2
                                      for x in pts])))

    assert rmse(dropped) < 1e-3, f"dropping the confidence did not converge: {rmse(dropped)}"
    assert rmse(kept) > rmse(dropped) * 1000.0, (
        f"carrying the confidence was not harmful here ({rmse(kept):.5f} against "
        f"{rmse(dropped):.5f}), so the two cases do not disagree and the API's "
        "keyword-only choice is unmotivated")
    # ...and the richer basis does break the linear ceiling, or none of this matters.
    assert rmse(dropped) < 0.01 < float(np.sqrt(np.mean(
        [(lin.predict(_linear_basis(x))[0] - _truth_nonlinear(x)) ** 2 for x in pts])))
