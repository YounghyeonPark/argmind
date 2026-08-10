# SPDX-License-Identifier: Apache-2.0
"""The runtime-shaped adapters, against the same Python reference.

`argmind/host/runtime_shape.hpp` exists because the core takes its dimensions as template
parameters and a platform above an OS wants them from a configuration file. The whole
value of it is **the shapes the instantiation list does not contain**, so that is what
this compares: 5x1, 7x1, 7x2, 50x1, 6x4. Comparing at 6x1 would mostly re-test the
reference against itself.

`shapecheck` reads and writes `xcheck`'s file format unchanged -- the case header already
carries `n` and `m` as `uint16_t` -- so this module reuses that harness's encoder and
reader rather than growing a second pair that could drift.

Measured, and it corrects a claim the header made first: matching the reference's
division instead of the core's reciprocal multiply removes divergences 1 and 2 but not
3. Reduction order still differs, so this is a tolerance rather than an equality, and it
grows with the sum length exactly as reduction order should.
"""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from test_cpp_parity import excited, run_both, scale_deviation, ulps

REPO = Path(__file__).resolve().parent.parent
CPP = REPO / "cpp"

#: Shapes chosen for one property: **none of them is in the ABI's instantiation list.**
#: A configuration file produces shapes like these and a template parameter cannot.
SHAPES = [(5, 1), (7, 1), (7, 2), (50, 1), (6, 4)]


@pytest.fixture(scope="session")
def shapecheck(tmp_path_factory):
    cxx = shutil.which("g++")
    if cxx is None:
        pytest.skip("no host g++")
    exe = tmp_path_factory.mktemp("shape") / "shapecheck.exe"
    done = subprocess.run(
        [cxx, "-std=c++17", "-O2", "-Wall", "-Wextra", "-I", str(CPP / "include"),
         str(CPP / "tools" / "shapecheck.cpp"), "-o", str(exe)],
        capture_output=True, text=True)
    assert done.returncode == 0, f"shapecheck does not build:\n{done.stderr}"
    return exe


@pytest.mark.parametrize("dtype", ["f4", "f8"])
@pytest.mark.parametrize("n,m", SHAPES)
@pytest.mark.parametrize("adapter,params", [(0, (0.99, 100.0)), (1, (0.5, 1e-6))],
                         ids=["rls", "nlms"])
def test_a_runtime_shape_agrees_with_the_reference(shapecheck, tmp_path, dtype, n, m,
                                                   adapter, params):
    lam, p0 = params
    xs, ys = excited(n, 300, seed=41, outputs=m)
    py, py_state, cpp, cpp_state = run_both(shapecheck, tmp_path, xs, ys, lam=lam, p0=p0,
                                            dtype=dtype, adapter=adapter)

    assert len(py) == len(cpp) == 300
    for i, (a, b) in enumerate(zip(py, cpp)):
        assert (a["applied"], a["reason"]) == (b["applied"], b["reason"]), i

    # 50 ulp against a measured worst of 4.0, which is headroom rather than a number
    # chosen to pass. `scale_deviation` returns a FRACTION -- comparing it against an ulp
    # count directly is an assertion that cannot fail, and that mistake was made once in
    # `test_cpp_parity.py` and caught by perturbing a state block by 1%.
    dev = scale_deviation(py_state, cpp_state)
    assert dev < ulps(dtype, 50.0), (
        f"{n}x{m} deviates by "
        f"{dev / float(np.finfo(np.dtype(dtype)).eps):.1f} ulp")


def test_the_deviation_is_reduction_order_and_grows_with_the_sum_length(shapecheck,
                                                                        tmp_path):
    """Classifying it, rather than accepting a small number.

    Divergences 1 and 2 are gone here by construction -- this divides where the core
    multiplies by a reciprocal -- so what is left should be 3, reduction order. The
    signature of that is a deviation which grows with `n` and is the same order at f32 as
    at f64. A logic difference would not do either.
    """
    got = {}
    for n in (2, 6, 50):
        for dtype in ("f4", "f8"):
            xs, ys = excited(n, 300, seed=41, outputs=1)
            _py, ps, _cpp, cs = run_both(shapecheck, tmp_path, xs, ys, lam=0.99, p0=100.0,
                                         dtype=dtype, adapter=0)
            got[(n, dtype)] = (scale_deviation(ps, cs)
                               / float(np.finfo(np.dtype(dtype)).eps))

    assert got[(50, "f8")] > got[(2, "f8")], (
        f"deviation does not grow with the sum length: {got}; that is not the signature "
        "of a reduction-order difference and the classification above is wrong")
    for n in (2, 6, 50):
        lo, hi = sorted((got[(n, "f4")], got[(n, "f8")]))
        assert hi <= max(lo * 8.0, 2.0), (
            f"n={n}: f32 {got[(n, 'f4')]:.1f} against f64 {got[(n, 'f8')]:.1f} ulp -- "
            "the same order is what says algorithmic rounding rather than a "
            "precision-dependent bug")


def test_a_runtime_shape_and_the_compile_time_core_hold_the_same_state_layout(shapecheck,
                                                                              tmp_path):
    """`RuntimeShapedRls` claims theta-then-P row-major, byte-identical to
    `Rls<T,N,M>`, so a block
    written by one is readable by the other and `persist` needs no special case. Checked
    at 6x1, the one shape both can express."""
    from test_cpp_parity import _write_case  # noqa: PLC0415
    xs, ys = excited(6, 200, seed=41, outputs=1)
    _py, dyn_state, _cpp, _ = run_both(shapecheck, tmp_path, xs, ys, lam=0.99, p0=100.0,
                                       dtype="f8", adapter=0)
    assert dyn_state.size == 6 * 1 + 6 * 6, "not theta-then-P at this shape"
    # theta first: the reference's own theta must match the leading block.
    from argmind.adapt import RecursiveLeastSquares  # noqa: PLC0415
    ref = RecursiveLeastSquares(6, 1, forgetting=0.99, p0=100.0, dtype="f8")
    for x, y in zip(np.asarray(xs, dtype="f8"), np.asarray(ys, dtype="f8")):
        ref.update(x, y)
    assert np.allclose(dyn_state[:6], np.asarray(ref.state)[:6], rtol=0, atol=1e-12)
    assert _write_case is not None


# -- the guard, at shapes no instantiation list contains ------------------------------

def _windup(n, m, count=400, seed=5):
    """Only axis 0 excited, so the covariance in every other direction grows as
    `p0 * lambda**-k` and `tr(P)` crosses a bound at a step that can be predicted rather
    than observed. The same scenario the core's guard parity uses, for the same reason: a
    fixture where the trip step is whatever happens to occur tests nothing about *when*."""
    xs = np.zeros((count, n))
    ys = np.zeros((count, m))
    rng = np.random.default_rng(seed)
    for k in range(count):
        xs[k, 0] = rng.standard_normal()
        ys[k] = xs[k, 0]
    return xs, ys


#: **Every axis here earns its place, and that was learned the hard way.** The first
#: version of this fixture used one bound at depth 1 with no divergence and no rejected
#: samples, and FOUR deliberate bugs in the guard survived it: rolling back to the newest
#: checkpoint instead of the deepest (identical at depth 1), stopping at the first failing
#: check (identical with one check), resetting the divergence history on a trip (nothing
#: to reset), and counting an adapter rejection as a trip (no rejections occurred). Depth
#: is 4, there are two bounds, divergence is on, and the data contains a NaN.
GUARD_CASES = [
    (7, 1, "f8"), (7, 2, "f8"), (5, 1, "f4"), (50, 1, "f8"),
]


@pytest.mark.parametrize("n,m,dtype", GUARD_CASES)
def test_a_runtime_shaped_guard_makes_the_same_decisions(shapecheck, tmp_path, n, m,
                                                         dtype):
    """A guard is decisions, so a decision that differs is not a rounding difference.
    Compared: the step it freezes on, whether each step rolled back, whether each step
    tripped anything, and WHICH checks tripped."""
    xs, ys = _windup(n, m)
    ys[17] = np.nan          # the adapter's own rejection path, which is not a trip
    guard = {"bounds": [("p_trace", None, 5e3), ("gain_norm", None, 1e6)],
             "divergence": {"ratio": 8.0, "fast_alpha": 0.25, "slow_alpha": 0.02,
                            "floor": 0.0, "warmup": 20},
             "depth": 4, "checkpoint_every": 3}
    py, py_state, cpp, cpp_state = run_both(shapecheck, tmp_path, xs, ys, lam=0.95,
                                            p0=1.0, dtype=dtype, guard=guard)

    # The rejection has to actually happen, or one of the four axes is decorative again.
    assert any(r["reason"] == "nonfinite_input" for r in cpp), (
        "no sample was rejected, so the rejection-is-not-a-trip axis tests nothing")
    assert not cpp[17]["rolled_back"], "an adapter rejection was treated as a trip"

    py_freeze = next((i for i, r in enumerate(py) if r["frozen"]), None)
    cpp_freeze = next((i for i, r in enumerate(cpp) if r["frozen"]), None)
    assert py_freeze is not None, "nothing froze, so this fixture tests no decision"
    assert py_freeze == cpp_freeze, f"froze at {py_freeze} against {cpp_freeze}"

    # Which checks, not merely whether any: with two bounds and a divergence detector,
    # "something tripped" would pass while the two disagreed about what.
    names = ("p_trace", "gain_norm")
    for i, (a, b) in enumerate(zip(py, cpp)):
        assert a["frozen"] == b["frozen"], f"step {i}: frozen"
        assert a["rolled_back"] == b["rolled_back"], f"step {i}: rolled_back"
        got = {names[k] for k in range(2) if b["tripped"] & (1 << k)}
        if b["tripped"] & (1 << 31):
            got.add("divergence")
        assert set(a["tripped_names"]) == got, (
            f"step {i}: reference tripped {sorted(a['tripped_names'])}, core {sorted(got)}")

    # And the state after the rollback, which is what says both chose the same checkpoint
    # out of four rather than merely both freezing.
    assert np.array_equal(py_state, cpp_state) or (
        scale_deviation(py_state, cpp_state) < ulps(dtype, 400.0)), (
        "the two rolled back to different checkpoints")


def test_the_guarded_comparison_is_not_two_untouched_adapters(shapecheck, tmp_path):
    """The trap this project already fell into once: after a trip both sides looked
    bit-identical, which proved nothing because both had rolled back to the *construction*
    state. Three things say this fixture is not that."""
    n, m, p0 = 7, 1, 1.0
    xs, ys = _windup(n, m)
    guard = {"bounds": [("p_trace", None, 5e3)]}
    _py, state, _cpp, _cs = run_both(shapecheck, tmp_path, xs, ys, lam=0.95, p0=p0,
                                     dtype="f8", guard=guard)

    construction = np.zeros(n * m + n * n)
    construction[n * m:] = (np.eye(n) * p0).ravel()
    assert not np.array_equal(state, construction), (
        "the run ended at the construction state, so a bit-identical comparison would "
        "prove nothing")
    assert abs(state[0] - 1.0) < 1e-3, "the excited axis did not learn"
    assert np.trace(state[n * m:].reshape(n, n)) > 100 * n * p0, (
        "the covariance did not wind up, so the p_trace bound was not what tripped")

    _py2, unguarded, _c2, _cs2 = run_both(shapecheck, tmp_path, xs, ys, lam=0.95, p0=p0,
                                          dtype="f8")
    assert not np.allclose(state, unguarded), (
        "guarded and unguarded end in the same place, so the guard changed nothing and "
        "the comparison above is decorative")


def test_a_runtime_shaped_guard_refuses_what_the_core_refuses(shapecheck, tmp_path):
    """Same refusals, so a configuration that is rejected in firmware is rejected here.
    Checked through the tool, because that is the path a caller takes."""
    from test_cpp_parity import _write_case  # noqa: PLC0415
    xs, ys = _windup(7, 1, count=20)

    # NLMS has no covariance, so a `p_trace` bound is refused rather than compared
    # against a fabricated zero -- the same rule `Nlms` follows in the core.
    case = tmp_path / "bad.bin"
    _write_case(case, np.asarray(xs, "f8"), np.asarray(ys, "f8"), 0.5, 1e-6, "f8",
                guard={"bounds": [("p_trace", None, 1e6)]}, adapter=1)
    done = subprocess.run([str(shapecheck), str(case), str(tmp_path / "o.bin")],
                          capture_output=True, text=True)
    assert done.returncode != 0, "a covariance bound was accepted on an adapter with none"
    assert "refused by the guard" in done.stderr, done.stderr


def test_the_divergence_history_is_held_through_a_freeze(shapecheck, tmp_path):
    """The case that found a real bug in this port, and it is the one this project has
    made twice before.

    `Divergence` holds its slow average while the guard is frozen -- `if (!r.frozen)
    slow += ...`. Without that, the baseline adapts to the very fault it is holding out
    against and the guard **thaws itself in the middle of one**. The runtime-shaped guard
    projects its report on to a one-output shell before evaluating checks, and the first
    version of that projection copied the scalars and not `frozen`. Measured: the
    reference thawed at 218 and this thawed at **166, while the fault ran to 200**.

    So the fixture is a fault that ENDS: noise, a 40-unit offset over steps 150 to 200,
    then normal data. A fault that never ends would let a premature thaw look correct,
    because the guard would simply freeze again.
    """
    n = 7
    rng = np.random.default_rng(3)
    true = rng.standard_normal(n)
    xs = rng.standard_normal((300, n))
    ys = (xs @ true).reshape(-1, 1) + 0.01 * rng.standard_normal((300, 1))
    ys[150:200] += 40.0

    guard = {"divergence": {"ratio": 8.0, "fast_alpha": 0.25, "slow_alpha": 0.02,
                            "floor": 1e-3, "warmup": 20},
             "depth": 4, "checkpoint_every": 3, "recover_after": 12}
    py, py_state, cpp, cpp_state = run_both(shapecheck, tmp_path, xs, ys, lam=0.98,
                                            p0=100.0, dtype="f8", guard=guard)

    def frozen_steps(rows):
        return [i for i, r in enumerate(rows) if r["frozen"]]

    def thaw_steps(rows):
        return [i for i in range(1, len(rows))
                if rows[i - 1]["frozen"] and not rows[i]["frozen"]]

    assert frozen_steps(py) and frozen_steps(cpp), "nothing froze"
    assert frozen_steps(py)[0] == frozen_steps(cpp)[0] == 150
    assert thaw_steps(py) == thaw_steps(cpp), (
        f"thaw disagreement: reference {thaw_steps(py)}, core {thaw_steps(cpp)}")

    # The discriminator: a thaw before the fault ends is the bug this exists for.
    assert thaw_steps(cpp) and thaw_steps(cpp)[0] > 200, (
        f"thawed at {thaw_steps(cpp)[0]}, while the fault runs to 200 -- the divergence "
        "baseline is tracking the fault instead of holding")
    assert len(frozen_steps(py)) == len(frozen_steps(cpp))
    assert scale_deviation(py_state, cpp_state) < ulps("f8", 400.0)


def test_every_check_runs_and_the_guard_does_not_stop_at_the_first_failure(shapecheck,
                                                                           tmp_path):
    """A documented property with no natural fixture, so this one is deliberate.

    Two checks failing on the *same* step is what distinguishes "every check runs" from
    "stop at the first", and in the windup scenario nothing else crosses its bound where
    `p_trace` does -- `p_min_diag` is not monotone and dips at step 40, and `gain_norm`
    peaked at 0.50 long before. Both were measured before resorting to this. So the two
    bounds here watch the same quantity at the same threshold, which is artificial and is
    the honest way to reach the property: the mask must have two bits, not one.
    """
    from test_cpp_parity import _write_case  # noqa: PLC0415
    xs, ys = _windup(7, 1)
    guard = {"bounds": [("p_trace", None, 5e3), ("p_trace", None, 5e3)],
             "depth": 4, "checkpoint_every": 3}
    py, _ps, cpp, _cs = run_both(shapecheck, tmp_path, xs, ys, lam=0.95, p0=1.0,
                                 dtype="f8", guard=guard)
    trip = next(i for i, r in enumerate(cpp) if r["tripped"])
    assert bin(cpp[trip]["tripped"]).count("1") == 2, (
        f"only {bin(cpp[trip]['tripped'])} tripped; the guard stopped at the first check")
    assert len(py[trip]["tripped_names"]) == 2, py[trip]["tripped_names"]
    assert _write_case is not None
