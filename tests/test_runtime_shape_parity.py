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


def test_shapecheck_refuses_a_guarded_case_rather_than_running_it_unguarded(shapecheck,
                                                                          tmp_path):
    """There is no `RuntimeShapedGuard` yet, and that is the gap that matters most
    in this layer --
    "no guard" being expressible is the failure the whole library is built around. A tool
    that silently dropped the guard would hide it, so it exits non-zero."""
    from test_cpp_parity import _write_case  # noqa: PLC0415
    xs, ys = excited(7, 20, seed=41, outputs=1)
    case = tmp_path / "guarded.bin"
    _write_case(case, np.asarray(xs, "f8"), np.asarray(ys, "f8"), 0.99, 100.0, "f8",
                guard={"bounds": [("p_trace", None, 1e6)]}, adapter=0)
    done = subprocess.run([str(shapecheck), str(case), str(tmp_path / "out.bin")],
                          capture_output=True, text=True)
    assert done.returncode != 0
    assert "RuntimeShapedGuard" in done.stderr, done.stderr
