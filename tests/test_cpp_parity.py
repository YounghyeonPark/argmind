# SPDX-License-Identifier: Apache-2.0
"""The C++ core against the Python reference.

DESIGN.md's plan for the port is that the Python becomes the oracle and the C++ is
verified against it — "same inputs, same outputs, to the tolerance the precision
allows". This is where that stops being a plan.

**Bit-identical agreement is not the target, and claiming it would be false.** Three
differences are deliberate and are documented in `cpp/include/argmind/rls.hpp`:

1. `P /= lambda` is a multiply by a precomputed reciprocal in C++. On an M4F
   `VDIV.F32` costs 14 cycles and that statement runs n² times, so at n=6 dividing
   would be ~500 cycles for one line against 14 plus 36 multiplies. It costs one
   extra rounding.
2. `k = Px / den` likewise.
3. numpy dispatches its reductions to BLAS, which may block or vectorise them and
   therefore sum in a different order than an ordered C++ loop. Neither is more
   correct.

So what is asserted is agreement to a tolerance derived from measurement, and — more
usefully — that the *trajectory* agrees rather than only the endpoint. An endpoint
comparison can pass while the two disagree in compensating ways, and a divergence
that appears on step one is a different fault from one that grows over a run.

**The metric took two corrections, and both are worth keeping.**

Element-wise relative error was the obvious first choice and it is the wrong one: a
state block contains elements that are near zero through cancellation, and a relative
error there is dominated by the cancellation rather than by anything either
implementation did. It read 1e-13 at f64 and 1e-4 at f32 — alarming, and meaningless.
Deviation relative to the *scale of the block* is the honest measure.

The second correction was that one tolerance does not fit every quantity, and that a
*relative* tolerance does not fit one of them at all. Measured maxima, in ulp of each
run's own precision, across n=2/6/12, one and three outputs, runs of 300 to 1600
steps, and lambda 0.97 and 0.99:

| quantity | f64 | f32 |
|---|---|---|
| whole state block, and theta | 1.5 | 1.5 |
| p_trace | 1.4 | 4.9 |
| p_min_diag | 2.4 | 1.9 |
| residual_norm | 8.4 | 14.2 |
| P alone | 26.7 | 37.6 |
| gain_norm | 58.0 | 64.6 |
| residual_post_norm | 597 | 597 |

The first thing to read out of that table is that the two implementations differ by
**about the same number of ulp whichever precision they run at**. That is what says
the difference is algorithmic rounding rather than a precision-dependent bug, and it
is the strongest single piece of evidence here that the port is right.

The second is that `residual_post_norm` does not belong in a relative comparison at
all, and the 597 above is not the whole story. It is ``y - x'theta`` *after* fitting
to that very sample, so it is a small difference of large nearly-equal numbers.
Diagnosed at the worst step of one run: the two agree on a residual of 6.72e-5 and a
post-residual of 5.91e-5, and disagree by 1.5e-15 in absolute terms — a few ulp of the
~3-magnitude predictor, amplified by a cancellation factor of about 5e4. A first
version of this file tried to escape that by comparing the *ratio* e_post/e instead,
on the reasoning that the ratio is well conditioned. It is not: the ratio inherits the
numerator's cancellation exactly, and measured 1.06e5 ulp for saying so.

So `residual_post_norm` is compared against an **absolute** bound scaled by the target
values — a few hundred ulp of ``max|y|``, which is the scale the cancellation happens
at — and its *property*, that it never exceeds the a priori error, is asserted directly.
Everything else gets 200 ulp relative, three to eight times the worst measured.

If no C++ compiler is available these tests skip. That is deliberate: the Python
reference must stay usable without a toolchain, and a suite that cannot run at all
on a fresh clone teaches people to ignore it.
"""

import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from argmind.adapt import RecursiveLeastSquares

REPO = Path(__file__).resolve().parent.parent
CPP = REPO / "cpp"

_CASE = struct.Struct("<4sHBBHHIdd")
_RESULT = struct.Struct("<4sHBBHHI")
_ROW = struct.Struct("<iidddd d".replace(" ", ""))   # applied, reason, 5 doubles

REASONS = {0: None, 1: "nonfinite_input", 2: "nonpositive_denominator"}


def _compiler() -> str | None:
    for name in ("g++", "clang++"):
        if shutil.which(name):
            return name
    return None


@pytest.fixture(scope="session")
def xcheck(tmp_path_factory) -> Path:
    """Build the parity tool once per session, or skip the module."""
    cxx = _compiler()
    if cxx is None:
        pytest.skip("no C++ compiler on PATH; the Python reference stands alone")
    out = tmp_path_factory.mktemp("cpp") / ("xcheck.exe" if sys.platform == "win32"
                                            else "xcheck")
    cmd = [cxx, "-std=c++17", "-O2", "-fno-exceptions", "-fno-rtti",
           "-I", str(CPP / "include"), str(CPP / "tools" / "xcheck.cpp"),
           "-o", str(out)]
    done = subprocess.run(cmd, capture_output=True, text=True)
    if done.returncode != 0:
        pytest.fail(f"xcheck did not build:\n{done.stdout}\n{done.stderr}")
    return out


def _write_case(path: Path, xs, ys, lam, p0, dtype) -> None:
    dt = np.dtype(dtype)
    n, m = xs.shape[1], ys.shape[1]
    with open(path, "wb") as fh:
        fh.write(_CASE.pack(b"AMXC", 1, dt.itemsize, 0, n, m, len(xs),
                            float(lam), float(p0)))
        for x, y in zip(xs, ys):
            fh.write(np.ascontiguousarray(x, dtype=dt).tobytes())
            fh.write(np.ascontiguousarray(y, dtype=dt).tobytes())


def _read_result(path: Path):
    blob = path.read_bytes()
    magic, version, code, _pad, n, m, count = _RESULT.unpack_from(blob, 0)
    assert magic == b"AMXR", "the tool did not write a result file"
    assert version == 1
    dt = np.dtype("f4" if code == 4 else "f8")
    off = _RESULT.size
    rows = []
    for _ in range(count):
        applied, reason, rn, rpn, gn, tr, md = _ROW.unpack_from(blob, off)
        off += _ROW.size
        rows.append({"applied": bool(applied), "reason": REASONS[reason],
                     "residual_norm": rn, "residual_post_norm": rpn,
                     "gain_norm": gn, "p_trace": tr, "p_min_diag": md})
    state = np.frombuffer(blob, dtype=dt, count=n * m + n * n, offset=off)
    return rows, np.array(state)


def run_both(xcheck: Path, tmp_path: Path, xs, ys, lam=0.99, p0=100.0,
             dtype="f8"):
    """Run the same case through both implementations. Returns (py_rows, py_state,
    cpp_rows, cpp_state)."""
    xs = np.ascontiguousarray(xs, dtype=dtype)
    ys = np.ascontiguousarray(ys, dtype=dtype)
    case, result = tmp_path / "case.bin", tmp_path / "result.bin"
    _write_case(case, xs, ys, lam, p0, dtype)
    done = subprocess.run([str(xcheck), str(case), str(result)],
                          capture_output=True, text=True)
    assert done.returncode == 0, f"xcheck failed: {done.stdout}\n{done.stderr}"

    ref = RecursiveLeastSquares(n_features=xs.shape[1], n_outputs=ys.shape[1],
                                forgetting=lam, p0=p0, dtype=dtype)
    py_rows = []
    for x, y in zip(xs, ys):
        r = ref.update(x, y)
        py_rows.append({"applied": r.applied, "reason": r.reason,
                        "residual_norm": r.residual_norm,
                        "residual_post_norm": r.residual_post_norm,
                        "gain_norm": r.gain_norm, "p_trace": r.p_trace,
                        "p_min_diag": r.p_min_diag})
    cpp_rows, cpp_state = _read_result(result)
    return py_rows, np.array(ref.state), cpp_rows, cpp_state


def scale_deviation(a, b) -> float:
    """Largest absolute difference, relative to the scale of the values compared.

    The primary metric, for the reason in the module docstring: element-wise
    relative error on a block containing near-zero entries measures the
    cancellation that produced them, not the two implementations.
    """
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    scale = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))), 1e-300)
    return float(np.max(np.abs(a - b))) / scale


def ulps(dtype, n=200.0) -> float:
    """A tolerance in units of last place, which is portable in a way an absolute
    number is not. The default is set from the measured table in the module
    docstring, with headroom."""
    return n * float(np.finfo(np.dtype(dtype)).eps)


#: Per-quantity relative tolerances, in ulp. `residual_post_norm` is absent
#: deliberately: it is a cancelled quantity and a relative comparison of one measures
#: the cancellation. It is checked absolutely instead, against `absolute_ulps`.
FIELD_ULP = {
    "residual_norm": 200.0,
    "gain_norm": 200.0,
    "p_trace": 200.0,
    "p_min_diag": 200.0,
}


def absolute_ulps(dtype, scale, n=400.0) -> float:
    """An absolute bound: `n` ulp of `scale`.

    For quantities whose own magnitude is the result of cancellation, so that
    relative error describes the cancellation rather than the implementations. The
    scale to use is the one the cancellation happens at — for a residual, the size of
    the values being subtracted.
    """
    return n * float(np.finfo(np.dtype(dtype)).eps) * float(scale)


def excited(n, count, seed=99, outputs=1):
    """A well-excited regressor. Excitation matters here: under poor excitation the
    covariance winds up as p0*lambda**-k and any difference between the two
    implementations is amplified along with it, which would make this a test of
    RLS's known instability rather than of the port."""
    rng = np.random.default_rng(seed)
    xs = rng.standard_normal((count, n))
    truth = np.linspace(1.0, 2.0, n)
    ys = (xs @ truth).reshape(count, 1) + 0.01 * rng.standard_normal((count, 1))
    if outputs > 1:
        ys = np.hstack([ys * (j + 1) for j in range(outputs)])
    return xs, ys


# ---------------------------------------------------------------------------
# Trajectory agreement
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n,dtype", [(2, "f8"), (6, "f8"), (12, "f8"),
                                    (6, "f4"), (12, "f4")])
def test_the_trajectories_agree_to_the_tolerance_the_precision_allows(
        xcheck, tmp_path, n, dtype):
    tol = ulps(dtype)
    xs, ys = excited(n, 300)
    py, py_state, cpp, cpp_state = run_both(xcheck, tmp_path, xs, ys,
                                            dtype=dtype)
    assert len(py) == len(cpp) == 300

    for i, (a, b) in enumerate(zip(py, cpp)):
        assert a["applied"] == b["applied"], f"step {i}: applied disagrees"
        assert a["reason"] == b["reason"], f"step {i}: reason disagrees"

    for field, allow in FIELD_ULP.items():
        limit = ulps(dtype, allow)
        dev = scale_deviation([r[field] for r in py], [r[field] for r in cpp])
        assert dev < limit, (
            f"{field} deviates by {dev:.3e} over the trajectory, "
            f"{dev / float(np.finfo(np.dtype(dtype)).eps):.0f} ulp, past the "
            f"{allow:.0f} ulp allowed")

    # The cancelled one, bounded absolutely against the scale it cancels at.
    limit = absolute_ulps(dtype, np.max(np.abs(ys)))
    gap = float(np.max(np.abs(np.array([r["residual_post_norm"] for r in py]) -
                              np.array([r["residual_post_norm"] for r in cpp]))))
    assert gap < limit, (
        f"residual_post_norm differs absolutely by {gap:.3e}, past the "
        f"{limit:.3e} that a few hundred ulp of the target scale allows")

    dev = scale_deviation(py_state, cpp_state)
    assert dev < tol, f"final state deviates by {dev:.3e}, past {tol:.3e}"


@pytest.mark.parametrize("dtype", ["f8", "f4"])
def test_three_outputs_agree(xcheck, tmp_path, dtype):
    """Sharing one covariance across outputs is where a port most easily gets the
    layout wrong, and a wrong layout still produces plausible numbers."""
    xs, ys = excited(6, 200, outputs=3)
    py, py_state, cpp, cpp_state = run_both(xcheck, tmp_path, xs, ys,
                                            dtype=dtype)
    dev = scale_deviation(py_state, cpp_state)
    assert dev < ulps(dtype), f"deviation {dev:.3e} past {ulps(dtype):.3e}"
    assert all(a["applied"] == b["applied"] for a, b in zip(py, cpp))


def test_the_state_block_layout_is_shared_byte_for_byte(xcheck, tmp_path):
    """theta (N*M) then P (N*N), row-major, in both. This is the block
    argmind/persist.py writes to flash, so a layout disagreement would mean state
    saved by one and restored by the other is silently wrong."""
    xs, ys = excited(6, 50, outputs=3)
    _, py_state, _, cpp_state = run_both(xcheck, tmp_path, xs, ys, dtype="f8")
    assert py_state.shape == cpp_state.shape == (6 * 3 + 6 * 6,)

    py_theta, cpp_theta = py_state[:18].reshape(6, 3), cpp_state[:18].reshape(6, 3)
    py_p, cpp_p = py_state[18:].reshape(6, 6), cpp_state[18:].reshape(6, 6)
    assert scale_deviation(py_theta, cpp_theta) < ulps("f8")
    assert scale_deviation(py_p, cpp_p) < ulps("f8")
    # P must be symmetric in both, since both symmetrise every step.
    for label, p in (("python", py_p), ("c++", cpp_p)):
        assert np.allclose(p, p.T, atol=0.0), f"{label} P is not symmetric"


# ---------------------------------------------------------------------------
# The rejection paths, which are the ones a port silently gets wrong
# ---------------------------------------------------------------------------

def test_a_nan_sample_is_rejected_identically(xcheck, tmp_path):
    xs, ys = excited(6, 60)
    xs[20, 3] = np.nan
    ys[41, 0] = np.inf
    py, py_state, cpp, cpp_state = run_both(xcheck, tmp_path, xs, ys, dtype="f8")

    assert [i for i, r in enumerate(py) if not r["applied"]] == [20, 41]
    assert [i for i, r in enumerate(cpp) if not r["applied"]] == [20, 41]
    assert py[20]["reason"] == cpp[20]["reason"] == "nonfinite_input"
    assert py[41]["reason"] == cpp[41]["reason"] == "nonfinite_input"
    assert scale_deviation(py_state, cpp_state) < ulps("f8")


def test_a_rejected_update_leaves_both_states_untouched(xcheck, tmp_path):
    """The property that matters in the field: a NaN must not take the estimator
    with it, in either implementation."""
    xs, ys = excited(6, 40)
    clean_py, clean_state, clean_cpp, clean_cpp_state = run_both(
        xcheck, tmp_path, xs, ys, dtype="f8")

    poisoned_x = np.vstack([xs, np.full((1, 6), np.nan)])
    poisoned_y = np.vstack([ys, np.zeros((1, 1))])
    _, py_state, _, cpp_state = run_both(xcheck, tmp_path, poisoned_x,
                                         poisoned_y, dtype="f8")
    assert np.array_equal(py_state, clean_state), "python absorbed the NaN"
    assert np.array_equal(cpp_state, clean_cpp_state), "c++ absorbed the NaN"


@pytest.mark.filterwarnings("ignore:overflow encountered:RuntimeWarning")
def test_an_indefinite_covariance_is_refused_by_both(xcheck, tmp_path):
    """Driven into the nonpositive-denominator path by a huge p0 against a large
    regressor, which is the fp32 cancellation regime the precision note describes.

    numpy warns about the overflow on the way in. That is the fixture working, not a
    problem: the whole point is to reach a denominator the recursion has to refuse.
    """
    n = 6
    xs = np.full((3, n), 1e18, dtype=np.float64)
    ys = np.ones((3, 1), dtype=np.float64)
    py, py_state, cpp, cpp_state = run_both(xcheck, tmp_path, xs, ys,
                                            lam=0.99, p0=1e300, dtype="f8")
    assert [r["reason"] for r in py] == [r["reason"] for r in cpp], (
        f"python said {[r['reason'] for r in py]}, c++ said "
        f"{[r['reason'] for r in cpp]}")
    assert any(r["reason"] == "nonpositive_denominator" for r in py), (
        "the fixture no longer reaches the path it exists to test")


# ---------------------------------------------------------------------------
# How the deviation behaves, rather than only whether it is small now
# ---------------------------------------------------------------------------

def test_the_a_posteriori_error_property_holds_in_both(xcheck, tmp_path):
    """`residual_post_norm` is too cancelled for a useful relative comparison, so
    what is checked is the property it must satisfy rather than its value.

    ``e_post = e * lambda / (lambda + x'Px)`` exactly, so the a posteriori error can
    never exceed the a priori one. That is the part worth asserting tightly, and both
    implementations satisfy it on every step.

    A first version of this test also compared the *ratio* e_post/e, on the reasoning
    that the ratio is well conditioned even though the numerator is not. That was
    wrong and the measurement said so: the ratio disagreed by 1.06e5 ulp. Dividing a
    cancelled quantity by a clean one rescales its error, it does not remove it. So
    the ratio is not compared here at all — the property is what this test is for.
    """
    xs, ys = excited(6, 400)
    py, _, cpp, _ = run_both(xcheck, tmp_path, xs, ys, dtype="f8")

    for i, (a, b) in enumerate(zip(py, cpp)):
        assert a["residual_post_norm"] <= a["residual_norm"] * (1 + 1e-12), (
            f"python step {i}: the update increased the error on its own sample")
        assert b["residual_post_norm"] <= b["residual_norm"] * (1 + 1e-12), (
            f"c++ step {i}: the update increased the error on its own sample")

    # The shrinkage must be real in both: with lambda < 1 and any excitation at all,
    # a step that left the error untouched would mean the gain did nothing.
    shrank_py = sum(1 for r in py if r["residual_post_norm"] < r["residual_norm"])
    shrank_cpp = sum(1 for r in cpp if r["residual_post_norm"] < r["residual_norm"])
    assert shrank_py > 390 and shrank_cpp > 390, (
        f"the update reduced its own sample's error on {shrank_py} of 400 steps in "
        f"python and {shrank_cpp} in c++; both should be nearly all of them")


def test_the_deviation_does_not_grow_with_run_length(xcheck, tmp_path):
    """A port can agree for fifty steps and drift apart over five thousand. The
    deliberate divergences are one extra rounding per step, so under good
    excitation the deviation should stay at roundoff rather than accumulate."""
    devs = {}
    for count in (100, 400, 1600):
        xs, ys = excited(6, count)
        _, py_state, _, cpp_state = run_both(xcheck, tmp_path, xs, ys,
                                             dtype="f8")
        devs[count] = scale_deviation(py_state, cpp_state)
    assert devs[1600] < ulps("f8"), f"deviation at 1600 steps: {devs}"
    # Measured at 300 against 1600 steps: 1.11e-16 and 2.22e-16, flat to within a
    # factor of two. A tenfold band allows that without admitting growth.
    assert devs[1600] < 10 * max(devs[100], float(np.finfo(np.float64).eps)), (
        f"the deviation grows with run length rather than staying at roundoff: "
        f"{devs}")


def test_f4_deviates_more_than_f8_and_the_gap_is_the_precision(xcheck, tmp_path):
    """A sanity check on the tolerances above: if f4 and f8 agreed equally well,
    the tolerances would not be measuring what they claim to."""
    xs, ys = excited(6, 300)
    _, py8, _, cpp8 = run_both(xcheck, tmp_path, xs, ys, dtype="f8")
    _, py4, _, cpp4 = run_both(xcheck, tmp_path, xs, ys, dtype="f4")
    d8, d4 = scale_deviation(py8, cpp8), scale_deviation(py4, cpp4)
    assert d4 > 1000 * d8, (
        f"f4 agreed nearly as well as f8 ({d4:.3e} vs {d8:.3e}); the tolerances "
        "are then not measuring the precision they are named for")
    assert d8 < ulps("f8")
    assert d4 < ulps("f4")


def test_the_reciprocal_divergence_is_the_size_it_should_be(xcheck, tmp_path):
    """The declared divergence, checked rather than asserted.

    C++ multiplies by 1/lambda where the Python divides by lambda. That is one
    extra rounding on a value of order one, so it should show up at a few ulp and
    not more. A far larger deviation would mean something other than the
    reciprocal is different.
    """
    xs, ys = excited(6, 500)
    _, py_state, _, cpp_state = run_both(xcheck, tmp_path, xs, ys, lam=0.97,
                                         dtype="f8")
    dev = scale_deviation(py_state, cpp_state)
    assert dev < ulps("f8"), (
        f"deviation {dev:.3e} is beyond what 20 ulp allows; a per-step rounding "
        "difference does not explain it, so look for a real disagreement")
    assert dev > 0.0, (
        "the two implementations agree bit-for-bit, which the declared "
        "divergences say they should not — either the reciprocal optimisation "
        "was reverted or this test is not exercising it")


# ---------------------------------------------------------------------------
# The C++ side's own suite, run from here so one command covers everything
# ---------------------------------------------------------------------------

def test_the_cpp_unit_tests_pass(tmp_path):
    """`python -m pytest` should be the single command that checks this repository,
    including the parts of the C++ core that only the C++ side can check."""
    cxx = _compiler()
    if cxx is None:
        pytest.skip("no C++ compiler on PATH")
    exe = tmp_path / ("test_core.exe" if sys.platform == "win32" else "test_core")
    build = subprocess.run(
        [cxx, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Wpedantic",
         "-fno-exceptions", "-fno-rtti", "-I", str(CPP / "include"),
         str(CPP / "tests" / "test_core.cpp"), str(CPP / "src" / "abi.cpp"),
         "-o", str(exe)], capture_output=True, text=True)
    assert build.returncode == 0, f"core tests did not build:\n{build.stderr}"
    run = subprocess.run([str(exe)], capture_output=True, text=True)
    assert run.returncode == 0, f"{run.stdout}\n{run.stderr}"
    assert "0 failures" in run.stdout, run.stdout


def test_the_core_headers_build_freestanding(tmp_path):
    """The target is a Cortex-M4F, so the core must not need a hosted runtime. This
    catches an accidental <cmath> or <vector> the moment it is added, on the host,
    without a cross-compiler."""
    cxx = _compiler()
    if cxx is None:
        pytest.skip("no C++ compiler on PATH")
    obj = tmp_path / "rls.o"
    done = subprocess.run(
        [cxx, "-std=c++17", "-O2", "-ffreestanding", "-fno-exceptions",
         "-fno-rtti", "-Wall", "-Wextra", "-I", str(CPP / "include"),
         "-c", "-x", "c++", str(CPP / "include" / "argmind" / "rls.hpp"),
         "-o", str(obj)], capture_output=True, text=True)
    assert done.returncode == 0, (
        f"the core no longer builds freestanding:\n{done.stderr}")


def test_the_c_header_is_valid_c99(tmp_path):
    """It is a C ABI. A header that only compiles as C++ would not be one."""
    cc = shutil.which("gcc") or shutil.which("clang")
    if cc is None:
        pytest.skip("no C compiler on PATH")
    obj = tmp_path / "hdr.o"
    done = subprocess.run(
        [cc, "-std=c99", "-Wall", "-Wextra", "-Wpedantic",
         "-I", str(CPP / "include"), "-c", "-x", "c",
         str(CPP / "include" / "argmind" / "argmind.h"), "-o", str(obj)],
        capture_output=True, text=True)
    assert done.returncode == 0, f"argmind.h is not valid C99:\n{done.stderr}"
