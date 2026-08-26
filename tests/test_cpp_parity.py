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
steps, **at lambda 0.99 on the host**:

| quantity | f64 | f32 |
|---|---|---|
| whole state block, and theta | 1.5 | 1.5 |
| p_trace | 1.4 | 4.9 |
| p_min_diag | 2.4 | 1.9 |
| residual_norm | 8.4 | 14.2 |
| P alone | 26.7 | 37.6 |
| gain_norm | 58.0 | 64.6 |
| residual_post_norm | 597 | 597 |

**That caption said "lambda 0.97 and 0.99" until 2026-08-25 and the table is lambda
0.99 only** -- corrected after an independent parity verification, and reproduced here.
The file already knew: the `FIELD_ULP` comment below records `p_trace` peaking at 10
where this table says 1.4, and says outright that the figure it once compared against
"had been taken at one lambda". Two numbers in one file disagreed and the stale one was
the one in the docstring, which is the one that gets quoted. Measured across lambda at
n=12, 300 steps, `gain_norm` in ulp:

| lambda | 0.99 | 0.97 | 0.95 | 0.90 |
|---|---|---|---|---|
| host f64 | 58.0 | 110.0 | 188.6 | 77.5 |
| target f32 | 98.0 | 78.8 | 164.9 | **203.6** |

So the table below is a *best case*, not a maximum, and the widest figures live with
`FIELD_ULP`. The first thing to read out of that table is that the two implementations
differ by **about the same number of ulp whichever precision they run at**. That is what says
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

import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import _toolbuild

from argmind.adapt import NormalisedLeastMeanSquares, RecursiveLeastSquares
from argmind.guard import Bound, Divergence, Guard

REPO = Path(__file__).resolve().parent.parent
CPP = REPO / "cpp"

#: The C++ self-check count, quoted in CLAUDE.md and under docs/. Pinned rather than
#: floored -- see `test_the_cpp_unit_tests_pass` -- and gated in the documents by
#: `test_every_DOCUMENT_stating_the_check_count_states_the_CURRENT_one` below.
DOCUMENTED_CHECKS = 5653

# Format v2. Every field sits at its natural alignment and every block is a multiple of
# eight bytes, so both sides read these with one call and no packing pragmas. The C++ side
# static-asserts the sizes; if these formats drift apart the build breaks rather than the
# comparison quietly comparing the wrong bytes.
_CASE_FIXED = struct.Struct("<4sHBBHHIddIIII")   # 48 bytes
_BOUND = struct.Struct("<ddII")                  # 24 bytes, four slots always written
_DIVERGENCE = struct.Struct("<ddddII")           # 40 bytes
_RESULT = struct.Struct("<4sHBBHHI")             # 16 bytes
_ROW = struct.Struct("<iiIIddddd")               # 56 bytes
CASE_HEADER_BYTES = _CASE_FIXED.size + 4 * _BOUND.size + _DIVERGENCE.size
assert CASE_HEADER_BYTES == 184, CASE_HEADER_BYTES

FORMAT_VERSION = 2

REASONS = {0: None, 1: "nonfinite_input", 2: "nonpositive_denominator"}

#: Field ids, matching argmind::Field in cpp/include/argmind/guard.hpp. The C++ uses an
#: enum where the Python uses a string, so this table is the only place the two spellings
#: meet — a mismatch here would silently watch the wrong quantity.
FIELD = {"residual_norm": 0, "residual_post_norm": 1, "gain_norm": 2,
         "p_trace": 3, "p_min_diag": 4}


def _compiler() -> str | None:
    for name in ("g++", "clang++"):
        if shutil.which(name):
            return name
    return None


@pytest.fixture(scope="session", params=_toolbuild.ARCHES)
def xcheck(request, tmp_path_factory) -> Path:
    """The parity tool, built for the host and for the declared target.

    Both, on every case: a review measured a QEMU invocation at 157 ms and showed
    that the hand-picked subset which used to stand in for the target passed with
    the host binaries substituted, and passed again with the target FPU set to
    round-toward-zero. See tests/_toolbuild.py.
    """
    work = tmp_path_factory.mktemp("cpp_" + request.param)
    return _toolbuild.build_tool("xcheck", request.param, work)


def _write_case(path: Path, xs, ys, lam, p0, dtype, guard=None, adapter=0) -> None:
    """Write one case. `guard` is None for a bare run.

    A guard with no bounds and no divergence is simply not applied on the other side,
    which keeps the tool to one code path instead of two that can drift apart.
    """
    dt = np.dtype(dtype)
    n, m = xs.shape[1], ys.shape[1]
    g = guard or {}
    bounds = g.get("bounds", [])
    assert len(bounds) <= 4, "the case format carries four bound slots"
    div = g.get("divergence")

    with open(path, "wb") as fh:
        # The byte after `dtype` was padding and is now the adapter selector, the
        # same trick `pipecheck`'s n_outputs used. The header stays 48 bytes.
        fh.write(_CASE_FIXED.pack(b"AMXC", FORMAT_VERSION, dt.itemsize, adapter,
                                  n, m,
                                  len(xs), float(lam), float(p0),
                                  int(g.get("depth", 1)),
                                  int(g.get("checkpoint_every", 1)),
                                  int(g.get("recover_after", 0)), len(bounds)))
        for i in range(4):
            if i < len(bounds):
                field, lo, hi = bounds[i]
                flags = (1 if lo is not None else 0) | (2 if hi is not None else 0)
                fh.write(_BOUND.pack(float(lo or 0.0), float(hi or 0.0),
                                     FIELD[field], flags))
            else:
                fh.write(_BOUND.pack(0.0, 0.0, 0, 0))
        if div:
            fh.write(_DIVERGENCE.pack(float(div["ratio"]), float(div["fast_alpha"]),
                                      float(div["slow_alpha"]), float(div["floor"]),
                                      int(div["warmup"]), 1))
        else:
            fh.write(_DIVERGENCE.pack(0.0, 0.0, 0.0, 0.0, 0, 0))
        for x, y in zip(xs, ys):
            fh.write(np.ascontiguousarray(x, dtype=dt).tobytes())
            fh.write(np.ascontiguousarray(y, dtype=dt).tobytes())


def _read_result(path: Path, state_size=None):
    blob = path.read_bytes()
    magic, version, code, _pad, n, m, count = _RESULT.unpack_from(blob, 0)
    assert magic == b"AMXR", "the tool did not write a result file"
    assert version == FORMAT_VERSION, f"result format v{version}, expected "\
                                      f"v{FORMAT_VERSION}"
    dt = np.dtype("f4" if code == 4 else "f8")
    off = _RESULT.size
    rows = []
    for _ in range(count):
        applied, reason, tripped, flags, rn, rpn, gn, tr, md = _ROW.unpack_from(blob, off)
        off += _ROW.size
        rows.append({"applied": bool(applied), "reason": REASONS[reason],
                     "tripped": tripped, "frozen": bool(flags & 1),
                     "rolled_back": bool(flags & 2),
                     "residual_norm": rn, "residual_post_norm": rpn,
                     "gain_norm": gn, "p_trace": tr, "p_min_diag": md})
    # The result header carries n and m but not the STATE size, and those stopped
    # being the same thing when a second adapter arrived: RLS writes `n*m + n*n` and
    # NLMS writes `n*m`. Reading the RLS size off an NLMS block is a short buffer, which
    # is at least loud -- the reverse would have silently read past the end.
    if state_size is None:
        state_size = n * m + n * n
    state = np.frombuffer(blob, dtype=dt, count=state_size, offset=off)
    return rows, np.array(state)


def _row_of(report, guarded: bool) -> dict:
    row = {"applied": report.applied, "reason": report.reason,
           "residual_norm": report.residual_norm,
           "residual_post_norm": report.residual_post_norm,
           "gain_norm": report.gain_norm, "p_trace": report.p_trace,
           "p_min_diag": report.p_min_diag}
    if guarded:
        row["frozen"] = bool(getattr(report, "frozen", False))
        row["rolled_back"] = bool(getattr(report, "rolled_back", False))
        row["tripped_names"] = tuple(getattr(report, "tripped", ()))
    return row


def build_reference(xs, ys, lam, p0, dtype, guard=None, adapter=0):
    """The Python side of a case, guarded or not.

    The guard is built here rather than in each test so that the two sides are configured
    from one description. A mismatch between how the two are configured would look exactly
    like a parity failure, which is the most expensive kind of test bug.
    """
    if adapter == 1:
        # `lam`/`p0` carry `mu`/`eps` for this adapter; the case format has one pair of
        # doubles and giving it a second that is null half the time would be worse.
        ref = NormalisedLeastMeanSquares(
            n_features=xs.shape[1], n_outputs=ys.shape[1], mu=lam, eps=p0, dtype=dtype)
    else:
        ref = RecursiveLeastSquares(n_features=xs.shape[1], n_outputs=ys.shape[1],
                                    forgetting=lam, p0=p0, dtype=dtype)
    if not guard:
        return ref, []
    checks, names = [], []
    for field, lo, hi in guard.get("bounds", []):
        checks.append(Bound(field, lo=lo, hi=hi, name=field))
        names.append(field)
    div = guard.get("divergence")
    if div:
        checks.append(Divergence(ratio=div["ratio"], fast_alpha=div["fast_alpha"],
                                 slow_alpha=div["slow_alpha"], floor=div["floor"],
                                 warmup=div["warmup"], name="divergence"))
        names.append("divergence")
    wrapped = Guard(ref, checks,
                    rollback_depth=guard.get("depth", 1),
                    checkpoint_every=guard.get("checkpoint_every", 1),
                    recover_after=guard.get("recover_after") or None)
    return wrapped, names


def run_both(xcheck: Path, tmp_path: Path, xs, ys, lam=0.99, p0=100.0,
             dtype="f8", guard=None, adapter=0):
    """Run the same case through both implementations. Returns (py_rows, py_state,
    cpp_rows, cpp_state)."""
    xs = np.ascontiguousarray(xs, dtype=dtype)
    ys = np.ascontiguousarray(ys, dtype=dtype)
    case, result = tmp_path / "case.bin", tmp_path / "result.bin"
    _write_case(case, xs, ys, lam, p0, dtype, guard=guard, adapter=adapter)
    done = subprocess.run([str(xcheck), str(case), str(result)],
                          capture_output=True, text=True)
    assert done.returncode == 0, f"xcheck failed: {done.stdout}\n{done.stderr}"

    ref, _names = build_reference(xs, ys, lam, p0, dtype, guard, adapter=adapter)
    py_rows = [_row_of(ref.update(x, y), bool(guard)) for x, y in zip(xs, ys)]
    cpp_rows, cpp_state = _read_result(result, ref.state.size)
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


#: Per-quantity relative tolerances, in ulp. `residual_post_norm` is absent deliberately: it
#: is a cancelled quantity and a relative comparison of one measures the cancellation. It is
#: checked absolutely instead, against `absolute_ulps`.
#:
#: The headroom is uneven and worth knowing rather than assuming. Measured over every shape
#: this tool instantiates, at 300/800/1600 steps and lambda 0.97 and 0.99, on both
#: architectures: `p_trace` peaks at 10 and `p_min_diag` at 10.1, so 200 is twenty times what
#: they need -- while `gain_norm` peaks at 134 on the target at f32, which is 1.5x. An earlier
#: version of DESIGN.md described this as "three to eight times the worst measured"; for
#: `gain_norm` it never was, and the figure it compared against had been taken at one lambda.
#:
#: **The 200 is scoped to the lambdas this suite instantiates, and outside them it is
#: breached (measured 2026-08-25, independently and then reproduced).** At n=12, 300 steps,
#: f32 on the target, `gain_norm` reads **203.6 ulp at lambda 0.90** and 164.9 at 0.95; on
#: the host at f64 it reads 188.6 at 0.95. No fixture uses lambda below 0.97, so the suite
#: passes and nothing here is loosened -- not-tested is not the same as passing. If anyone
#: widens a fixture's lambda this is the assertion that fails first, and the honest record is
#: that 200 holds for lambda in [0.97, 0.99] rather than for the whole legal range.
#:
#: **And no recorded sweep covers the library's own default.** `RecursiveLeastSquares`
#: defaults to `forgetting=0.999` and the legal range is `0.0 < forgetting <= 1.0`; the
#: sweeps here use 0.99, 0.97 and 0.95. An independent verification measured "P alone" at
#: 376.8 ulp at lambda 0.999 against this docstring's 26.7, with the state block unaffected
#: (0.1-1.0 ulp) because theta carries the block's scale while P shrinks. No assertion is
#: threatened -- but "P alone 26.7" is not a figure that holds at the default. Not
#: re-derived here; recorded as that verification's measurement.
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
    # ...and that the checks did not quietly disappear. "0 failures" alone is equally true of
    # a suite that stopped running.
    #
    # **This was a `>= 4580` floor until 2026-08-25, and the floor is why the documented figure
    # was wrong for a thousand checks.** A floor catches a count falling and is blind to it
    # rising, so CLAUDE.md and docs/findings/runtime-library-and-cpp.md both went on saying
    # 4,652 while the suite grew to 5,653. Pinned exactly instead: this number is quoted in
    # prose, and prose is protected by nothing. Adding C++ checks is *supposed* to fail here
    # once — update the two documents, then this constant.
    m = re.search(r"(\d+) checks", run.stdout)
    assert m, f"the self-check binary printed no count at all: {run.stdout!r}"
    got = int(m.group(1))
    assert got == DOCUMENTED_CHECKS, (
        f"the C++ self-check count is {got}, and CLAUDE.md and "
        f"docs/findings/runtime-library-and-cpp.md say {DOCUMENTED_CHECKS}. "
        + ("Checks were removed, which is the case this test was written for."
           if got < DOCUMENTED_CHECKS else
           "Checks were added, which is good news: update the claim in both documents and "
           "then this constant, rather than loosening this assertion back into a floor."))


def test_the_core_headers_build_freestanding(tmp_path):
    """The target is a Cortex-M4F, so the core must not need a hosted runtime. This
    catches an accidental <cmath> or <vector> the moment it is added, on the host,
    without a cross-compiler."""
    cxx = _compiler()
    if cxx is None:
        pytest.skip("no C++ compiler on PATH")
    # Every core header, not just rls.hpp. An audit found persist.hpp was in neither this
    # gate nor the C99 one, so an accidental <cstring> in it would have gone unnoticed.
    #
    # **The list was hand-maintained until 2026-08-25, and it had drifted: nlms.hpp was
    # missing from it while the comment above said "every core header".** The target gate in
    # tests/test_target_build.py did list it, but that gate needs the ARM cross toolchain and
    # skips without it -- so on a clone with only a host compiler, a hosted #include in
    # nlms.hpp compiled clean. Same shape as the persist.hpp incident this comment records,
    # one header later. Derived from the directory now, so a NEW core header is gated the day
    # it appears rather than the day someone remembers. Cross-checked against the target
    # gate's own tuple so the two cannot drift apart again in either direction.
    core = sorted(h.name for h in (CPP / "include" / "argmind").glob("*.hpp"))
    from tests.test_target_build import HEADERS as TARGET_HEADERS
    assert core, "no core headers found to gate; the glob has stopped matching"
    assert set(core) == set(TARGET_HEADERS), (
        f"the host freestanding gate would check {core} and the target gate checks "
        f"{sorted(TARGET_HEADERS)}; they must cover the same headers or one of them is a "
        "gate with a hole in it")
    for header in core:
        obj = tmp_path / (header.replace(".hpp", ".o"))
        done = subprocess.run(
            [cxx, "-std=c++17", "-O2", "-ffreestanding", "-fno-exceptions",
             "-fno-rtti", "-Wall", "-Wextra", "-Wpedantic", "-I", str(CPP / "include"),
             "-c", "-x", "c++", str(CPP / "include" / "argmind" / header),
             "-o", str(obj)], capture_output=True, text=True)
        assert done.returncode == 0, (
            f"{header} no longer builds freestanding:\n{done.stderr}")


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



#: A plain-C caller, compiled by a C compiler and linked against the C++ core. It uses the
#: guard rather than the bare adapter deliberately: DESIGN.md's rule is that a safety
#: mechanism which can be left out gets left out, and until the guard had ABI entry points
#: the C boundary made "no guard" the only option there was.
_C_CALLER = r"""
#include <stdio.h>
#include <string.h>
#include "argmind/argmind.h"

int drive_pipeline(void);

int main(void) {
    static am_rls_storage store[AM_GUARD_F32_6_1_MAX_BYTES / sizeof(am_rls_storage)];
    float x[6];
    float y[1];
    float restored[AM_RLS_F32_6_1_STATE];
    static unsigned char slot[512];
    am_guard_report r;
    am_guard_counters c;
    am_record rec;
    unsigned int seed = 7u;
    size_t n;
    int k, j;

    if (am_guard_f32_6_1_init(store, sizeof(store), 0.98f, 10.0f, 1, 0) != AM_OK)
        return 1;
    if (am_guard_f32_6_1_add_bound(store, AM_FIELD_RESIDUAL_NORM, 0, 0.0, 1, 8.0)
        != AM_OK) return 2;
    if (am_guard_f32_6_1_valid(store) != AM_OK) return 3;

    for (k = 0; k < 80; ++k) {
        for (j = 0; j < 6; ++j) {
            seed = seed * 1664525u + 1013904223u;
            x[j] = (float)((int)(seed >> 8) % 1000) / 500.0f;
        }
        y[0] = x[0] + 2.0f * x[1] - 0.5f + (k == 50 ? 60.0f : 0.0f);
        if (am_guard_f32_6_1_update(store, x, y, &r, NULL) != AM_OK) return 4;
    }
    if (!am_guard_f32_6_1_frozen(store)) return 5;
    if (am_guard_f32_6_1_frozen_by(store) == 0u) return 6;
    if (am_guard_f32_6_1_counters(store, &c) != AM_OK) return 7;
    if (c.trips != 1u) return 8;

    n = am_persist_pack_f32(am_guard_f32_6_1_state(store), AM_RLS_F32_6_1_STATE, 6, 1,
                            1, 0xFEEDu, 1, "residual", 8, slot, sizeof(slot));
    if (n == 0u) return 9;
    if (am_persist_unpack_f32(slot, n, &rec) != AM_PERSIST_OK) return 10;
    if (rec.sequence != 1u || rec.frozen != 1) return 11;
    if (am_persist_read_state_f32(slot, n, restored, AM_RLS_F32_6_1_STATE)
        != AM_PERSIST_OK) return 12;
    if (memcmp(restored, am_guard_f32_6_1_state(store),
               sizeof(restored)) != 0) return 13;
    if (am_persist_next_slot(0) != 1) return 14;

    if (drive_pipeline() != 0) return 15;

    printf("ok trips=%u frozen_by=%u bytes=%u\n", c.trips,
           am_guard_f32_6_1_frozen_by(store), (unsigned)n);
    return 0;
}
"""

#: The composed handle, driven from C: three caller-owned rings of different widths, a
#: guarded six-feature adapter, an index frame map built one entry at a time, and `step()`.
#: This is the shape an integrator actually wires up, and until the pipeline had ABI entry
#: points none of it was reachable from C at all.
_C_PIPELINE = r"""
#include "argmind/argmind.h"

static float  pv0[16 * 1], pv1[16 * 3], pv2[16 * 2];
static double pt0[16], pt1[16], pt2[16];
static unsigned char pk0[16], pk1[16], pk2[16];
static am_rls_storage pipe[AM_PIPE_G_F32_6_1_MAX_BYTES / sizeof(am_rls_storage)];

int drive_pipeline(void) {
    am_ring_f32 rings[3];
    am_step_report rep;
    am_guard_counters c;
    unsigned int seed = 4242u;
    unsigned int frames = 0, updated = 0;
    float a[3], b[2], y[1];
    int k, j;

    if (am_ring_f32_init(&rings[0], pv0, pt0, pk0, 1, 16) != AM_OK) return 1;
    if (am_ring_f32_init(&rings[1], pv1, pt1, pk1, 3, 16) != AM_OK) return 1;
    if (am_ring_f32_init(&rings[2], pv2, pt2, pk2, 2, 16) != AM_OK) return 1;

    if (am_pipe_g_f32_6_1_init(pipe, sizeof(pipe), 0.98f, 10.0f, 2, 0,
                               AM_POLICY_HOLD_LAST, 0.0, 0) != AM_OK) return 2;
    if (am_pipe_g_f32_6_1_add_feature(pipe, 1, 0) != AM_OK) return 3;
    if (am_pipe_g_f32_6_1_add_feature(pipe, 2, 1) != AM_OK) return 3;
    if (am_pipe_g_f32_6_1_add_feature(pipe, 1, 2) != AM_OK) return 3;
    if (am_pipe_g_f32_6_1_add_feature(pipe, 2, 0) != AM_OK) return 3;
    if (am_pipe_g_f32_6_1_add_feature(pipe, 1, 1) != AM_OK) return 3;
    if (am_pipe_g_f32_6_1_add_target(pipe, 0, 0) != AM_OK) return 3;
    if (am_pipe_g_f32_6_1_set_constant(pipe, 1) != AM_OK) return 3;
    if (am_pipe_g_f32_6_1_add_bound(pipe, AM_FIELD_RESIDUAL_NORM, 0, 0.0, 1, 8.0)
        != AM_OK) return 4;
    /* The gate that lets a fast output run on a slow target. Here the target rides the
       reference, so it can never fire -- which is worth the C caller proving, since a gate
       that fired here would silently halve the learning rate of the plainest wiring. */
    if (am_pipe_g_f32_6_1_set_require_fresh_target(pipe, 1) != AM_OK) return 4;
    if (am_pipe_g_f32_6_1_configure(pipe, rings, 3) != AM_OK) return 5;

    for (k = 1; k <= 60; ++k) {
        for (j = 0; j < 3; ++j) {
            seed = seed * 1664525u + 1013904223u;
            a[j] = (float)((int)(seed >> 9) % 1000) / 500.0f;
        }
        for (j = 0; j < 2; ++j) {
            seed = seed * 1664525u + 1013904223u;
            b[j] = (float)((int)(seed >> 9) % 1000) / 500.0f;
        }
        y[0] = 2.0f * a[0] - b[1] + 0.5f * a[2] + 0.25f * b[0] - 1.5f * a[1] + 0.3f
             + (k == 35 ? 90.0f : 0.0f);
        if (am_ring_f32_push(&rings[1], k - 0.85, a, 1) != AM_OK) return 6;
        if (am_ring_f32_push(&rings[2], k - 0.45, b, 1) != AM_OK) return 6;
        if (am_ring_f32_push(&rings[0], (double)k, y, 1) != AM_OK) return 6;
        if (k % 3) continue;
        if (am_pipe_g_f32_6_1_step(pipe, rings, 3, &rep) != AM_OK) return 7;
        if (!rep.ok) return 8;
        frames += rep.frames;
        updated += rep.updated;
        if (rep.frames && !rep.has_prediction) return 15;
        if (rep.skipped[AM_SKIP_STALE_TARGET]) return 16;
    }
    /* The fault has to have tripped the guard, or this exercised only the healthy path. */
    if (!am_pipe_g_f32_6_1_frozen(pipe)) return 9;
    if (am_pipe_g_f32_6_1_counters(pipe, &c) != AM_OK) return 10;
    if (c.trips != 1u) return 11;
    /* Every reference sample became a frame: the ring is 16 deep and step() runs every
       third one, so nothing was overwritten unseen. */
    if (frames != 60u) return 12;
    /* And roughly the pre-fault half of them adapted. The guard freezes at k=35 and blocks
       everything after, so an `updated` near 60 would mean the freeze did not take -- the
       first version of this check demanded 50 and failed for exactly that reason. */
    if (updated < 28u || updated > 40u) return 13;
    if (c.blocked_frozen < 15u) return 14;
    return 0;
}
"""


def test_a_plain_c_program_can_drive_the_guard_and_the_record_format(tmp_path):
    """The C ABI's whole claim, exercised rather than asserted.

    `test_the_c_header_is_valid_c99` compiles the header and stops there, which says
    nothing about whether the symbols are callable and linkable from C — a header can
    parse perfectly while every entry point is name-mangled out of reach. This compiles a
    C99 translation unit with a *C* compiler, links it against the C++ core, and runs it.

    It also goes end to end on purpose: adapt, trip the guard, pack the state, read it
    back. Each half working says nothing about the two meeting, and the reason the guard
    and the record format got ABI entry points at all is that `am_rls_*_state()` used to
    hand out a block a C caller had no way to frame.
    """
    cc = shutil.which("gcc")
    cxx = _compiler()
    if cc is None or cxx is None:
        pytest.skip("need both a C and a C++ compiler on PATH")
    objs = []
    for name, text in (("pipeline.c", _C_PIPELINE), ("caller.c", _C_CALLER)):
        src = tmp_path / name
        src.write_text(text, encoding="utf-8")
        obj = tmp_path / (name[:-2] + ".o")
        done = subprocess.run(
            [cc, "-std=c99", "-Wall", "-Wextra", "-Wpedantic",
             "-I", str(CPP / "include"), "-c", str(src), "-o", str(obj)],
            capture_output=True, text=True)
        assert done.returncode == 0, (
            f"{name} does not compile as C99:\n{done.stderr}")
        objs.append(str(obj))

    exe = tmp_path / ("caller.exe" if sys.platform == "win32" else "caller")
    done = subprocess.run(
        [cxx, "-std=c++17", "-O2", "-fno-exceptions", "-fno-rtti",
         "-I", str(CPP / "include"), *objs, str(CPP / "src" / "abi.cpp"),
         "-o", str(exe)], capture_output=True, text=True)
    assert done.returncode == 0, f"the C caller does not link:\n{done.stderr}"

    run = subprocess.run([str(exe)], capture_output=True, text=True)
    assert run.returncode == 0, (
        f"the C caller returned {run.returncode}; the number is the failing step in "
        f"caller.c\n{run.stdout}\n{run.stderr}")
    assert run.stdout.startswith("ok trips=1"), run.stdout


# ---------------------------------------------------------------------------
# The guard, which is where the two implementations differ structurally
# ---------------------------------------------------------------------------
#
# The C++ guard cannot hold a list of polymorphic checks: no virtuals, no heap. So the
# checks are concrete members with declared capacity and their identities are an enum
# rather than a string. The *behaviour* has to be identical anyway, and these tests are
# what says it is — which matters more here than for RLS, because the guard's job is to
# decide, and a decision that differs between reference and implementation is not a
# rounding difference.

GUARD_TRACE_BOUND = {
    "bounds": [("p_min_diag", 0.0, None), ("p_trace", None, 1e9)],
    "depth": 4, "checkpoint_every": 5,
}
GUARD_DIVERGENCE = {
    "bounds": [("p_min_diag", 0.0, None)],
    "divergence": {"ratio": 8.0, "fast_alpha": 0.25, "slow_alpha": 0.02,
                   "floor": 1e-3, "warmup": 20},
    "depth": 4, "checkpoint_every": 5,
}


def tripped_names(row, order):
    """Turn the C++ bitmask into the names the Python reports.

    Bit i is bound i in declaration order; bit 31 is the divergence check. This mapping is
    the only place the two spellings meet, so it is worth being explicit about.
    """
    if "tripped_names" in row:
        return tuple(row["tripped_names"])
    mask = row["tripped"]
    out = [order[i] for i in range(len(order))
           if order[i] != "divergence" and mask & (1 << i)]
    if mask & (1 << 31):
        out.append("divergence")
    return tuple(out)


@pytest.mark.parametrize("dtype", ["f8", "f4"])
def test_a_guarded_run_agrees_step_for_step(xcheck, tmp_path, dtype):
    """Nothing should trip here. The point is that a guard which never fires still
    produces the same trajectory as the reference, including the checkpointing."""
    xs, ys = excited(6, 300)
    py, py_state, cpp, cpp_state = run_both(xcheck, tmp_path, xs, ys, dtype=dtype,
                                            guard=GUARD_TRACE_BOUND)
    assert all(r["applied"] for r in py), "the fixture tripped; it is meant not to"
    assert all(r["applied"] for r in cpp)
    assert not any(r["frozen"] for r in py + cpp)
    assert scale_deviation(py_state, cpp_state) < ulps(dtype)


def windup(count, seed=31):
    """Excitation confined to the first axis, so the covariance in the second winds up as
    `p0 * lambda**-k` and `tr(P)` crosses a threshold at a *predictable* step.

    This fixture exists because the obvious one does not test what it looks like it tests.
    With `p0=100` and six features, `tr(P)` starts at 600, so a `p_trace` bound anywhere
    below that trips on the very first update — and a test asserting the two
    implementations froze on the same step then compares 0 against 0 for every bound it
    sweeps. A swept axis that does not move the outcome is not evidence.
    """
    rng = np.random.default_rng(seed)
    xs = np.column_stack([rng.standard_normal(count), np.zeros(count)])
    ys = np.zeros((count, 1))
    return xs, ys


@pytest.mark.parametrize("hi,expected", [(10.0, 44), (100.0, 89), (1000.0, 134)])
def test_the_guard_freezes_on_the_same_step_in_both(xcheck, tmp_path, hi, expected):
    """The step a guard trips on must match exactly: freezing one sample later means one
    more bad update was absorbed before anyone noticed.

    The expected step is not a recorded observation, it is the growth law. Under no
    excitation the unexcited covariance is exactly `p0 * lambda**-k`, so a `tr(P)` bound
    of `hi` is crossed at `k = ln(hi) / ln(1/lambda)` — 45, 90 and 135 here, against 44,
    89 and 134 measured. Pinning that rather than only "the two agree" is what stops this
    test degrading into 0 == 0 if a fixture changes.
    """
    guard = {"bounds": [("p_trace", None, hi)], "depth": 1}
    xs, ys = windup(260)
    py, py_state, cpp, cpp_state = run_both(xcheck, tmp_path, xs, ys, lam=0.95, p0=1.0,
                                            guard=guard)

    py_first = next(i for i, r in enumerate(py) if r["frozen"])
    cpp_first = next(i for i, r in enumerate(cpp) if r["frozen"])
    assert py_first == cpp_first, (
        f"python froze at step {py_first}, c++ at {cpp_first}")
    assert py_first == expected, (
        f"froze at step {py_first}, not the {expected} the p0*lambda**-k growth law "
        "predicts; the fixture is no longer exercising what this test claims")
    assert py[py_first]["rolled_back"] and cpp[cpp_first]["rolled_back"]
    assert [r["frozen"] for r in py] == [r["frozen"] for r in cpp]
    assert [r["applied"] for r in py] == [r["applied"] for r in cpp]
    assert scale_deviation(py_state, cpp_state) < ulps("f8"), (
        "the rollback restored different states")


@pytest.mark.parametrize("depth,every", [(1, 1), (2, 1), (4, 5), (10, 1)])
def test_the_rollback_picks_the_same_checkpoint_in_both(xcheck, tmp_path, depth, every):
    """Which checkpoint a trip rolls back to is a *decision*, and decisions must match
    exactly even where arithmetic cannot.

    Comparing the two restored states to each other cannot answer this on its own: they
    would agree to a few ulp whether or not the same checkpoint was chosen, because
    neighbouring checkpoints are themselves close. So each side is compared to **its own**
    unguarded trajectory to find which step its restored state came from, and the two step
    numbers must be identical. That comparison is bit-exact within one implementation, so
    it identifies the checkpoint rather than merely being consistent with it.

    An earlier version of this test asserted the two restored states were bit-identical.
    They were — but only because the fixture tripped on step 0, so both rolled back to the
    *construction* state, which is identical by definition. With a fixture that trips at a
    real step they differ by about 41 ulp, which is the recursion's own disagreement
    carried into the checkpoint. Bit-identity was never available here, and the tidy
    result came from a broken measurement.
    """
    guard = {"bounds": [("p_trace", None, 100.0)], "depth": depth,
             "checkpoint_every": every}
    xs, ys = windup(200)
    py, py_state, cpp, cpp_state = run_both(xcheck, tmp_path, xs, ys, lam=0.95, p0=1.0,
                                            guard=guard)
    trip = next((i for i, r in enumerate(py) if r["frozen"]), None)
    assert trip is not None and trip > 10, (
        f"depth={depth} tripped at step {trip}; nothing to identify")

    # Which step of the unguarded Python run does the restored state come from?
    ref = RecursiveLeastSquares(n_features=2, forgetting=0.95, p0=1.0, dtype="f8")
    step_of = None
    for k, (x, y) in enumerate(zip(xs, ys)):
        ref.update(x, y)
        if np.array_equal(np.array(ref.state), py_state):
            step_of = k
            break
    assert step_of is not None, (
        "the restored state is not a state the unguarded run passed through, so the "
        "rollback did not restore a checkpoint")

    # The C++ must have restored the same one. Bit-exact within C++, so this identifies it.
    _, _, _, cpp_prefix = run_both(xcheck, tmp_path, xs[:step_of + 1], ys[:step_of + 1],
                                   lam=0.95, p0=1.0)
    assert np.array_equal(cpp_state, cpp_prefix), (
        f"depth={depth} every={every}: python rolled back to the state after update "
        f"{step_of + 1}, and the C++ did not")
    assert scale_deviation(py_state, cpp_state) < ulps("f8"), (
        "the same checkpoint, but the states disagree beyond the recursion's own level")


def test_every_failing_check_is_reported_in_both(xcheck, tmp_path):
    """Three faults at once is diagnostic information. The C++ reports a bitmask and the
    Python a tuple of names; both must name the same set."""
    guard = {"bounds": [("p_trace", None, 1.0), ("gain_norm", None, 1e-9)], "depth": 1}
    order = ["p_trace", "gain_norm"]
    xs, ys = excited(6, 20)
    py, _, cpp, _ = run_both(xcheck, tmp_path, xs, ys, guard=guard)

    i = next(k for k, r in enumerate(py) if r["frozen"])
    assert set(tripped_names(py[i], order)) == set(tripped_names(cpp[i], order))
    assert len(tripped_names(cpp[i], order)) == 2, (
        f"only {tripped_names(cpp[i], order)} reported; both bounds fail on this step")


def test_the_divergence_check_agrees_including_the_held_baseline(xcheck, tmp_path):
    """The check whose semantics were a design error twice over in the Python.

    A clean stretch, then a sensor returning nonsense, then clean again. What has to match
    is the step it trips on and the fact that it stays frozen while the fault persists,
    which is what the held baseline is for. Getting the hold backwards in the port would
    show up here as the C++ thawing early.
    """
    rng = np.random.default_rng(7)
    theta = np.linspace(1.0, 2.0, 6)
    xs = rng.standard_normal((220, 6))
    ys = (xs @ theta).reshape(220, 1) + 0.01 * rng.standard_normal((220, 1))
    ys[120:180] *= -80.0                      # the sensor breaks, then recovers

    guard = dict(GUARD_DIVERGENCE, recover_after=25)
    py, py_state, cpp, cpp_state = run_both(xcheck, tmp_path, xs, ys, lam=0.99,
                                            p0=10.0, guard=guard)

    py_frozen = [r["frozen"] for r in py]
    cpp_frozen = [r["frozen"] for r in cpp]
    assert any(py_frozen), "the fixture no longer trips the divergence check"
    assert py_frozen.index(True) == cpp_frozen.index(True), (
        f"python tripped at {py_frozen.index(True)}, c++ at {cpp_frozen.index(True)}")
    first_gap = next((i for i, (a, b) in enumerate(zip(py_frozen, cpp_frozen))
                      if a != b), None)
    assert first_gap is None, (
        f"the two disagree about when adaptation was frozen, first at step {first_gap}")
    assert scale_deviation(py_state, cpp_state) < ulps("f8")


def test_recovery_happens_on_the_same_step_in_both(xcheck, tmp_path):
    """`recover_after` consecutive healthy probes thaw the guard. Thawing at different
    times means the two resume learning from different data."""
    rng = np.random.default_rng(11)
    theta = np.linspace(1.0, 2.0, 6)
    xs = rng.standard_normal((300, 6))
    ys = (xs @ theta).reshape(300, 1) + 0.01 * rng.standard_normal((300, 1))
    ys[100:140] *= -80.0

    guard = dict(GUARD_DIVERGENCE, recover_after=25)
    py, _, cpp, _ = run_both(xcheck, tmp_path, xs, ys, lam=0.99, p0=10.0, guard=guard)

    py_frozen = [r["frozen"] for r in py]
    cpp_frozen = [r["frozen"] for r in cpp]
    assert True in py_frozen, "the fixture never tripped"
    start = py_frozen.index(True)
    assert False in py_frozen[start:], (
        "the fixture never recovered, so there is nothing to compare")
    thaw_py = py_frozen.index(False, start)
    thaw_cpp = cpp_frozen.index(False, cpp_frozen.index(True))
    assert thaw_py == thaw_cpp, f"python thawed at {thaw_py}, c++ at {thaw_cpp}"


def test_an_adapter_rejection_is_not_a_guard_trip_in_either(xcheck, tmp_path):
    """A NaN sample and a diverging learner are different faults. Blurring them makes the
    logs useless exactly when someone needs them, and a port could easily blur them by
    routing the rejection through the check path."""
    xs, ys = excited(6, 60)
    xs[25, 2] = np.nan
    py, py_state, cpp, cpp_state = run_both(xcheck, tmp_path, xs, ys,
                                            guard=GUARD_TRACE_BOUND)

    for rows, label in ((py, "python"), (cpp, "c++")):
        assert rows[25]["applied"] is False, f"{label} applied a NaN sample"
        assert rows[25]["reason"] == "nonfinite_input", f"{label}: {rows[25]['reason']}"
        assert rows[25]["frozen"] is False, f"{label} froze on a bad sample"
        assert rows[25]["rolled_back"] is False
        assert not any(r["frozen"] for r in rows), f"{label} froze somewhere"
    assert scale_deviation(py_state, cpp_state) < ulps("f8")


def test_a_frozen_probe_reports_the_same_numbers_in_both(xcheck, tmp_path):
    """While frozen the gain is genuinely zero and the covariance scalars are those of the
    restored checkpoint. Both are easy for a port to fake with a stale value."""
    guard = {"bounds": [("p_trace", None, 1.0)], "depth": 1}
    xs, ys = excited(6, 60)
    py, _, cpp, _ = run_both(xcheck, tmp_path, xs, ys, guard=guard)

    # The tripping step reports frozen=True with a real gain: the update was computed and
    # then rolled back. Only the steps after it are probes, and `rolled_back` is exactly
    # the flag that distinguishes them — which is what those two fields are for.
    frozen = [i for i, r in enumerate(py) if r["frozen"] and not r["rolled_back"]]
    assert len(frozen) > 10, "not enough probe steps to compare"
    assert [i for i, r in enumerate(py) if r["rolled_back"]] ==            [i for i, r in enumerate(cpp) if r["rolled_back"]],         "the two disagree about which step was rolled back"
    for i in frozen:
        assert py[i]["gain_norm"] == 0.0 and cpp[i]["gain_norm"] == 0.0, (
            f"step {i}: a frozen step reported a non-zero gain")
    for field in ("residual_norm", "p_trace", "p_min_diag"):
        dev = scale_deviation([py[i][field] for i in frozen],
                              [cpp[i][field] for i in frozen])
        assert dev < ulps("f8", FIELD_ULP.get(field, 200.0)), (
            f"frozen {field} deviates by {dev:.3e}")
    # p_trace must be constant while frozen: the state is not moving.
    assert len({round(cpp[i]["p_trace"], 15) for i in frozen}) == 1, (
        "c++ p_trace changed while frozen, so it is not the checkpoint's value")


def test_a_configuration_that_cannot_work_is_refused_by_the_core(xcheck, tmp_path):
    """`recover_after` with a positive lower bound on `gain_norm` can never thaw, because
    the gain is zero while frozen. The Python raises; the C++ has no exceptions, so its
    `valid()` returns false and the tool exits non-zero. Different mechanisms, the same
    refusal, and the refusal is the part that matters.
    """
    xs, ys = excited(6, 10)
    guard = {"bounds": [("gain_norm", 0.1, None)], "depth": 1, "recover_after": 10}

    with pytest.raises(ValueError, match="could never thaw"):
        build_reference(np.ascontiguousarray(xs), np.ascontiguousarray(ys),
                        0.99, 100.0, "f8", guard)

    case = tmp_path / "bad.bin"
    _write_case(case, np.ascontiguousarray(xs), np.ascontiguousarray(ys),
                0.99, 100.0, "f8", guard=guard)
    done = subprocess.run([str(xcheck), str(case), str(tmp_path / "out.bin")],
                          capture_output=True, text=True)
    assert done.returncode != 0, "the C++ core accepted a guard that could never thaw"
    assert "rejected" in done.stderr


def test_a_guard_with_no_checks_is_refused_by_the_core():
    """A guard that can never freeze is a wrapper that looks like a safety mechanism. Note
    that the *case format* treats no-checks as "run bare", so the C++ side of this is
    covered by cpp/tests/test_core.cpp rather than through the tool."""
    with pytest.raises(ValueError, match="no checks"):
        Guard(RecursiveLeastSquares(6, dtype="f8"), [])


# The three tests below were added because a mutation check found survivors: the parity
# suite had no run that continued past a thaw, none with two trips, and none where a
# rejected sample falling through to the check path would have changed anything.


def test_behaviour_after_a_thaw_agrees_too(xcheck, tmp_path):
    """A run that trips, recovers, and keeps going.

    Checks are reset on `thaw()` and not on the trip, which is the way round the Python
    arrived at only after getting it backwards twice. A port that reset them on the trip,
    or forgot to reset them on the thaw, produces a detector with the wrong history after
    recovery — and that only shows up in a run that continues long enough afterwards to
    trip a second time or decline to. Neither existed in this suite until a mutant
    survived.
    """
    rng = np.random.default_rng(19)
    theta = np.linspace(1.0, 2.0, 6)
    xs = rng.standard_normal((600, 6))
    ys = (xs @ theta).reshape(600, 1) + 0.01 * rng.standard_normal((600, 1))
    ys[100:140] *= -80.0            # first fault
    ys[380:420] *= -80.0            # and a second, well after recovery

    guard = dict(GUARD_DIVERGENCE, recover_after=25)
    py, py_state, cpp, cpp_state = run_both(xcheck, tmp_path, xs, ys, lam=0.99,
                                            p0=10.0, guard=guard)
    fz_py = [r["frozen"] for r in py]
    fz_cpp = [r["frozen"] for r in cpp]

    first = fz_py.index(True)
    thawed = fz_py.index(False, first)
    assert any(fz_py[thawed:]), (
        "the fixture never froze a second time, so nothing after the thaw is exercised")
    gap = next((i for i, (a, b) in enumerate(zip(fz_py, fz_cpp)) if a != b), None)
    assert gap is None, (
        f"the two disagree about freezing from step {gap} onward; the first fault ended "
        f"at 140 and the thaw was at {thawed}, so this is post-thaw behaviour")
    assert scale_deviation(py_state, cpp_state) < ulps("f8")


def test_two_trips_in_one_run_agree(xcheck, tmp_path):
    """After a rollback the retained checkpoints are all at or after the restored state, so
    keeping them would let the next trip restore a moment that came afterwards. That needs
    two trips and a depth above one to show, and this suite had neither."""
    guard = {"bounds": [("p_trace", None, 100.0)], "depth": 4, "checkpoint_every": 1,
             "recover_after": 5}
    xs, ys = windup(400)
    py, py_state, cpp, cpp_state = run_both(xcheck, tmp_path, xs, ys, lam=0.95, p0=1.0,
                                            guard=guard)

    rolled_py = [i for i, r in enumerate(py) if r["rolled_back"]]
    rolled_cpp = [i for i, r in enumerate(cpp) if r["rolled_back"]]
    assert len(rolled_py) >= 2, (
        f"only {len(rolled_py)} rollback(s); the fixture no longer produces two")
    assert rolled_py == rolled_cpp, (
        f"python rolled back at {rolled_py[:6]}, c++ at {rolled_cpp[:6]}")
    assert scale_deviation(py_state, cpp_state) < ulps("f8")


def test_a_rejection_does_not_reach_the_checks_in_either(xcheck, tmp_path):
    """The stronger form of the rejection test.

    The earlier one used bounds a rejected sample satisfies anyway, so a port that routed
    the rejection through the check path behaved identically and the mutant survived. Here
    the divergence check is enabled, and its exponential averages absorb every report they
    are shown — so if a rejection reaches it, its baseline moves and the trip step changes.
    """
    xs, ys = excited(6, 300)
    for i in (40, 41, 42, 90, 150):
        xs[i, 2] = np.nan

    guard = dict(GUARD_DIVERGENCE)
    py, py_state, cpp, cpp_state = run_both(xcheck, tmp_path, xs, ys, guard=guard)

    rejected = [i for i, r in enumerate(py) if r["reason"] == "nonfinite_input"]
    assert rejected == [40, 41, 42, 90, 150], f"rejections landed at {rejected}"
    for rows, label in ((py, "python"), (cpp, "c++")):
        for i in rejected:
            assert rows[i]["frozen"] is False, f"{label} froze on a rejected sample"
            assert rows[i]["rolled_back"] is False, f"{label} rolled back a rejection"
        assert [k for k, r in enumerate(rows) if r["reason"] == "nonfinite_input"] == \
            rejected, f"{label} rejected different samples"

    # The comparison that has teeth: with the divergence check enabled, a rejection that
    # reached it would move its baseline and change everything downstream.
    assert [r["frozen"] for r in py] == [r["frozen"] for r in cpp]
    dev = scale_deviation([r["residual_norm"] for r in py],
                          [r["residual_norm"] for r in cpp])
    assert dev < ulps("f8", FIELD_ULP["residual_norm"]), (
        f"residual trajectory deviates by {dev:.3e}; a rejection is being folded into "
        "something it should not reach")
    assert scale_deviation(py_state, cpp_state) < ulps("f8")


# ---------------------------------------------------------------------------
# Tests written against a specific mutant
# ---------------------------------------------------------------------------
#
# A mutation check on the C++ guard left eight survivors on its first run. Several were
# unreachable from this harness by construction and became cpp/tests/test_core.cpp tests.
# The four below are the ones that belong here, and each was *designed against its mutant*
# — the first attempt at all four failed to discriminate, for four different and instructive
# reasons recorded in the docstrings. A test written without checking that the bug it is
# meant to catch would actually fail it is a guess.


def linear(n, count, seed, noise=0.0):
    rng = np.random.default_rng(seed)
    xs = rng.standard_normal((count, n))
    ys = (xs @ np.linspace(1.0, 2.0, n)).reshape(count, 1)
    if noise:
        ys = ys + noise * rng.standard_normal((count, 1))
    return xs, ys


def test_two_trips_close_together_restore_the_same_checkpoints(xcheck, tmp_path):
    """After a rollback the retained checkpoints are all at or after the restored state, so
    keeping them would let the next trip restore a moment that came *afterwards*.

    The two trips have to be close together. The first attempt put the second spike one
    sample after the first, which lands while the guard is still frozen — so it never
    became a second trip at all. And with four or more healthy updates between trips the
    ring is fully refreshed and a stale-checkpoint bug becomes invisible. A residual bound
    is used rather than a covariance one because it can be tripped on demand.
    """
    xs, ys = linear(6, 60, seed=3)
    ys[30] += 500.0
    ys[34] += 500.0            # after the thaw at ~32, with two healthy updates between
    guard = {"bounds": [("residual_norm", None, 5.0)], "depth": 4,
             "checkpoint_every": 1, "recover_after": 1}
    py, py_state, cpp, cpp_state = run_both(xcheck, tmp_path, xs, ys, guard=guard)

    rolled_py = [i for i, r in enumerate(py) if r["rolled_back"]]
    rolled_cpp = [i for i, r in enumerate(cpp) if r["rolled_back"]]
    assert rolled_py == [30, 34], (
        f"the fixture produced rollbacks at {rolled_py}, not two close together")
    assert rolled_py == rolled_cpp, f"python {rolled_py}, c++ {rolled_cpp}"
    assert [r["frozen"] for r in py] == [r["frozen"] for r in cpp]
    assert scale_deviation(py_state, cpp_state) < ulps("f8")


def test_a_fault_soon_after_a_thaw_agrees(xcheck, tmp_path):
    """Checks are reset on `thaw()` and not on the trip, which is the way round the Python
    reached only after getting it backwards twice.

    The second fault has to arrive *within the warmup window* after the thaw. Reset checks
    pass their first `warmup` samples unconditionally, so a fault inside that window cannot
    trip; a detector that kept its history can. The first attempt put the second fault 240
    samples later, by which time the exponential averages have re-converged and the two
    behave identically — a gap that made the mutant survive.
    """
    xs, ys = linear(6, 400, seed=5, noise=0.01)
    ys[60:90] *= -80.0
    ys[150:200] *= -80.0
    guard = dict(GUARD_DIVERGENCE, recover_after=10)
    py, py_state, cpp, cpp_state = run_both(xcheck, tmp_path, xs, ys, lam=0.99, p0=10.0,
                                            guard=guard)

    fz_py = [r["frozen"] for r in py]
    fz_cpp = [r["frozen"] for r in cpp]
    assert fz_py.count(True) > 50, "the fixture barely froze"
    thawed = fz_py.index(False, fz_py.index(True))
    assert any(fz_py[thawed:]), "nothing after the thaw is exercised"
    gap = next((i for i, (a, b) in enumerate(zip(fz_py, fz_cpp)) if a != b), None)
    assert gap is None, f"the two disagree about freezing from step {gap} onward"
    assert scale_deviation(py_state, cpp_state) < ulps("f8")


def test_a_rejection_does_not_reach_the_checkpoints_either(xcheck, tmp_path):
    """A rejected sample must not take the applied path, and the observable that proves it
    is the *checkpoint* rather than the trip step.

    A rejection routed through the check path stores a checkpoint, so a trip immediately
    afterwards rolls back to a state recorded during the dropout instead of the last
    healthy one. Two earlier attempts failed: bounds that a rejected report satisfies
    anyway made the mutant behave identically, and a version relying on the rejection
    polluting the divergence baseline could not discriminate because the fault tripped on
    its first sample and there was no room to trip earlier.
    """
    xs, ys = linear(6, 60, seed=9)
    xs[30:41, 2] = np.nan      # an eleven-sample dropout
    ys[41] += 500.0            # and a trip on the very next sample
    guard = {"bounds": [("residual_norm", None, 50.0)], "depth": 4,
             "checkpoint_every": 1}
    py, py_state, cpp, cpp_state = run_both(xcheck, tmp_path, xs, ys, guard=guard)

    rejected = [i for i, r in enumerate(py) if r["reason"] == "nonfinite_input"]
    assert rejected == list(range(30, 41)), f"rejections landed at {rejected}"
    rolled = [i for i, r in enumerate(py) if r["rolled_back"]]
    assert rolled == [41], f"expected one rollback right after the dropout, got {rolled}"
    for rows, label in ((py, "python"), (cpp, "c++")):
        for i in rejected:
            assert rows[i]["frozen"] is False, f"{label} froze on a rejected sample"
            assert rows[i]["rolled_back"] is False
    assert rolled == [i for i, r in enumerate(cpp) if r["rolled_back"]]
    assert np.array_equal(py_state, cpp_state) or \
        scale_deviation(py_state, cpp_state) < ulps("f8"), (
        "the rollback after a dropout restored different states, so a rejection is "
        "reaching the checkpoint path in one of them")


def test_intermittent_recovery_during_a_fault_does_not_thaw_either(xcheck, tmp_path):
    """`recover_after` counts *consecutive* healthy probes, so an unhealthy one resets the
    run. Without that reset the count accumulates across a fault that is intermittently
    quiet, and the guard thaws in the middle of it.

    The clean stretches have to be long enough for the fast average to fall below the
    threshold, or there are no healthy probes at all and nothing accumulates — which is why
    a first attempt with every third sample clean did not discriminate. Thirty faulty then
    fifteen clean, repeatedly, does.
    """
    xs, ys = linear(6, 400, seed=13, noise=0.01)
    bad = np.zeros(400, dtype=bool)
    for start in range(100, 380, 45):
        bad[start:start + 30] = True
    ys[bad] *= -80.0

    guard = dict(GUARD_DIVERGENCE, recover_after=25)
    py, py_state, cpp, cpp_state = run_both(xcheck, tmp_path, xs, ys, lam=0.99, p0=10.0,
                                            guard=guard)
    fz_py = [r["frozen"] for r in py]
    fz_cpp = [r["frozen"] for r in cpp]
    assert sum(fz_py) > 250, (
        f"only {sum(fz_py)} frozen steps; a guard that thaws through an intermittent "
        "fault is exactly what this test exists to detect, so the fixture must keep it "
        "frozen")
    assert fz_py == fz_cpp, (
        "the two disagree about staying frozen through an intermittently quiet fault; "
        f"python froze {sum(fz_py)} steps, c++ {sum(fz_cpp)}")
    assert scale_deviation(py_state, cpp_state) < ulps("f8")


# -- the second adapter, compared the same way the first one is -----------------------

@pytest.mark.parametrize("dtype", ["f4", "f8"])
@pytest.mark.parametrize("n,m", [(2, 1), (6, 1), (12, 1), (6, 3)])
@pytest.mark.parametrize("mu", [0.1, 0.5, 1.0, 2.0])
def test_nlms_agrees_with_the_reference(xcheck, tmp_path, dtype, n, m, mu):
    """NLMS is short enough to look obviously right, which is exactly why it gets the
    same treatment RLS got rather than a reading.

    `mu = 2.0` is the stability boundary and is included on purpose: it is where the two
    implementations are most likely to part company, not a value anyone should use.
    """
    xs, ys = excited(n, 400, seed=17, outputs=m)
    py, py_state, cpp, cpp_state = run_both(
        xcheck, tmp_path, xs, ys, lam=mu, p0=1e-6, dtype=dtype, adapter=1)

    assert len(py) == len(cpp)
    for i, (a, b) in enumerate(zip(py, cpp)):
        assert a["applied"] == b["applied"], i
        assert a["reason"] == b["reason"], i

    # `scale_deviation` returns a FRACTION; `ulps()` converts an ulp count to one. The
    # first version of this assertion compared the fraction against 400 directly and
    # could never fail -- caught by perturbing a state block by 1% and watching it pass.
    dev = scale_deviation(py_state, cpp_state)
    assert dev < ulps(dtype, 50.0), (
        f"state block deviates by {dev / float(np.finfo(np.dtype(dtype)).eps):.1f} ulp "
        f"at mu={mu}, n={n}, m={m}")

    if dtype == "f8" and m == 1:
        # **Bit-identical at f64 with a single output**, at every mu including the
        # stability boundary. Worth asserting separately rather than folding into the
        # tolerance: RLS cannot manage this -- it has two declared reciprocal-multiply
        # divergences -- and NLMS adds none of its own, so the only thing that could
        # separate the two sides is reduction order.
        #
        # **Scoped to m == 1 because that is where it was measured.** The first version
        # of this assertion said "at f64" and was generalised from n=6, m=1; at (6,3) the
        # two differ by 1.3 ulp at mu=0.5 and 7.6 at mu=2, because numpy takes a matvec
        # path for a (6,3) theta that sums in a different order from `vec_mat`'s loop.
        # That is declared divergence 3 in `rls.hpp`, not a new one -- but the claim had
        # to shrink to what was actually measured.
        #
        # Measured f32, 400 steps: 0.0 at n=2, 1.0-1.5 at mu=0.5, and 7.5-13.1 at mu=2,
        # where the recursion is marginally stable so a rounding difference persists
        # instead of decaying.
        assert np.array_equal(py_state, cpp_state), (
            f"f64/m=1 is no longer bit-identical at mu={mu}, n={n}: max |diff| "
            f"{np.max(np.abs(py_state - cpp_state)):.3e}")


@pytest.mark.parametrize("dtype", ["f4", "f8"])
def test_nlms_projects_a_sample_out_exactly_at_mu_one(xcheck, tmp_path, dtype):
    """The law, on both sides. After an update the a posteriori error is exactly
    `(1 - mu*x.x/(eps + x.x))` times the a priori one; at `mu = 1` with negligible `eps`
    that is zero for any data whatsoever.

    Asserting the LAW rather than a tolerance is what makes this a verification instead
    of a comparison of two things that could be wrong together -- the same reason RLS is
    checked against the closed-form weighted solution and not against itself.
    """
    xs, ys = excited(6, 200, seed=23, outputs=1)
    py, _py_state, cpp, _cpp_state = run_both(
        xcheck, tmp_path, xs, ys, lam=1.0, p0=1e-12, dtype=dtype, adapter=1)

    tol = 1e-5 if dtype == "f4" else 1e-9
    for i, row in enumerate(cpp):
        if not row["applied"]:
            continue
        # a priori is O(1) here, so an a posteriori norm at the noise floor is the law
        # holding rather than a vacuous comparison of two zeros.
        assert row["residual_post_norm"] < tol, (
            f"step {i}: mu=1 left {row['residual_post_norm']:.3e} of the residual")
    assert any(r["residual_norm"] > 0.1 for r in cpp), (
        "every a priori residual was tiny, so the assertion above proves nothing")
    for i, (a, b) in enumerate(zip(py, cpp)):
        assert a["applied"] == b["applied"], i


def test_nlms_rejects_the_same_things_rls_does(xcheck, tmp_path):
    """One vocabulary across adapters, so a guard's counters mean the same thing whichever
    is underneath. A regressor of 1e30 squares to infinity in f32 -- finite input reaching
    the denominator path, which is the case worth pinning."""
    xs, ys = excited(6, 20, seed=5, outputs=1)
    xs = np.array(xs, dtype="f4")
    ys = np.array(ys, dtype="f4")
    xs[5, :] = 1e30
    ys[9, :] = np.nan
    with np.errstate(over="ignore"):     # the 1e30 squaring IS the demonstration
        py, _ps, cpp, _cs = run_both(xcheck, tmp_path, xs, ys, lam=0.5, p0=1e-6,
                                     dtype="f4", adapter=1)
    assert cpp[5]["applied"] is False, cpp[5]
    assert cpp[5]["reason"] == "nonpositive_denominator", cpp[5]
    assert cpp[9]["applied"] is False, cpp[9]
    assert cpp[9]["reason"] == "nonfinite_input", cpp[9]
    for i, (a, b) in enumerate(zip(py, cpp)):
        assert (a["applied"], a["reason"]) == (b["applied"], b["reason"]), i


def test_the_parity_tool_refuses_a_guarded_nlms_rather_than_substituting_rls(
        xcheck, tmp_path):
    """The honest gap. Guarded NLMS is exercised by the C++ unit suite and the Python
    guard tests, but the two together have no parity harness. A tool that silently ran
    RLS instead would make that gap invisible, so it exits non-zero."""
    xs, ys = excited(6, 10, seed=1, outputs=1)
    case = tmp_path / "guarded_nlms.bin"
    _write_case(case, xs, ys, 0.5, 1e-6, "f8",
                guard={"bounds": [("residual_norm", None, 1e6)]}, adapter=1)
    done = subprocess.run([str(xcheck), str(case), str(tmp_path / "out.bin")],
                          capture_output=True, text=True)
    assert done.returncode != 0
    assert "guarded cases are Rls only" in done.stderr


#: Phrases that mark a check count as a deliberate historical record rather than a live claim.
#: A dated ledger entry is allowed to keep its own number; an unmarked one is a stale claim.
_HISTORICAL = ("when this was written", "when this entry was written", "as of this entry",
               "Re-measured", "re-measured", "passed **when", "of them when")


def test_every_DOCUMENT_stating_the_check_count_states_the_CURRENT_one():
    """**The coupling commit 5e48f9c claimed and did not have.**

    That commit pinned the C++ count in two assertions and then said "adding C++ checks now
    fails the suite until the documents are updated". It did not: both assertions compare the
    binary against a literal in a test, and nothing read a document. Update the literal alone
    and the suite goes green with every document stale -- which is the exact failure the commit
    set out to close. An independent audit caught it, and also found **three more stale sites at
    a fourth value (4,650)** that the commit's own sweep had missed, because that sweep grepped
    for "4,652" and a different stale number was sitting in three other files.

    So the mechanism is here instead of in the prose. Every check count stated in a document
    must be the current one, unless the sentence marks itself as a historical record -- which is
    a deliberate act, and the ledger files under `docs/` use it on purpose.

    `tests/test_architecture_doc.py` and `tests/test_design_doc.py` already read documents and
    check them against the code; this is the same idea applied to the one number that has now
    gone stale in six places across two values.
    """
    files = sorted(set(REPO.glob("*.md")) | set(REPO.glob("docs/**/*.md")))
    assert len(files) > 5, (
        f"only {len(files)} documents found to check; the glob has stopped matching")

    pattern = re.compile(r"([\d,]{3,6})\s+(?:C\+\+\s+)?(?:self-)?checks")
    BLANK, NEWLINE = chr(10) * 2, chr(10)
    stale = []
    for path in files:
        # Paragraph-scoped, not line-scoped: prose wraps, and a marker belongs to the sentence
        # it is in rather than to the line the wrap happened to put it on. A table's caption is
        # in the same paragraph as its rows only if no blank line separates them, so a marker
        # for a historical table goes in the row or the line above it.
        lineno = 1
        for para in path.read_text(encoding="utf-8").split(BLANK):
            for m in pattern.finditer(para):
                if int(m.group(1).replace(",", "")) == DOCUMENTED_CHECKS:
                    continue
                # A WINDOW around this number, not the whole paragraph. Scoping the marker to
                # the paragraph was the first version and it was useless: CLAUDE.md's live
                # claim sits in a paragraph that also says "re-measured", so one marker
                # exempted every count near it -- including the one being claimed.
                near = para[max(0, m.start() - 120):m.end() + 160]
                if any(k in near for k in _HISTORICAL):
                    continue
                stale.append(f"{path.relative_to(REPO)}:~{lineno}: {m.group(1)} -- "
                             f"{near.strip()[:100]}")
            lineno += para.count(NEWLINE) + 2

    assert not stale, (
        f"{len(stale)} document(s) state a C++ check count that is not the current "
        f"{DOCUMENTED_CHECKS:,} and do not mark themselves historical:\n  "
        + "\n  ".join(stale)
        + "\n\nEither update the figure, or -- if it is a dated ledger entry that should keep "
          f"its own number -- say so in the same sentence using one of {_HISTORICAL}.")
