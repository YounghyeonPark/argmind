"""`RuntimeShapedMlp` against `MultilayerPerceptron`.

The C++ side's initialisation deliberately does not match the reference's, so every case
hands over the reference's own starting weights. That is not a workaround: it is what a
device restoring a trained model does, and it makes the comparison about the *recursion*
rather than about two random number generators.

What is compared: applied/reason per step, the residual scalars per step, and the whole
state block at the end. What is expected to differ: reduction order, since numpy dispatches
its matvecs to BLAS and these are ordered loops — the same declared divergence 3 the other
adapters carry. **And one this adapter adds: `tanh`.** numpy's and libm's need not agree to
the last bit, and unlike a reduction difference that one compounds through the recursion.
Whether it does is measured rather than assumed, by comparing against `relu`, which is exact
in both.
"""
import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest

from argmind.adapt import MultilayerPerceptron

REPO = Path(__file__).resolve().parents[1]
CPP = REPO / "cpp"

CASE = struct.Struct("<4sHBBHHHHIId")      # 32 bytes; mlpcheck asserts the same
RESULT = struct.Struct("<4sHBBHHI")        # 16 bytes; xcheck's, reused
ROW = struct.Struct("<iiIIddddd")          # 56 bytes; xcheck's, reused
ACT = {"tanh": 0, "relu": 1}

#: Architectures no instantiation list could contain, which is the point of the hosted
#: tier. `()` is included because it is the one shape with an exact law.
ARCHES = [((), 3, 1), ((5,), 3, 1), ((8, 6), 4, 2), ((4, 4, 4), 3, 1), ((16,), 7, 3)]


@pytest.fixture(scope="session")
def mlpcheck(tmp_path_factory):
    cxx = shutil.which("g++")
    if cxx is None:
        pytest.skip("no host g++")
    exe = tmp_path_factory.mktemp("mlp") / "mlpcheck.exe"
    done = subprocess.run(
        [cxx, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Wpedantic",
         "-I", str(CPP / "include"), str(CPP / "tools" / "mlpcheck.cpp"), "-o", str(exe)],
        capture_output=True, text=True)
    assert done.returncode == 0, f"mlpcheck does not build:\n{done.stderr}"
    return exe


def run_both(tool, tmp_path, hidden, n, m, dtype, activation, lr, xs, ys, seed=1):
    ref = MultilayerPerceptron(n, m, hidden=hidden, activation=activation,
                               learning_rate=lr, dtype=dtype, seed=seed)
    init = np.array(ref.state, dtype=dtype)

    job = tmp_path / "case.bin"
    with job.open("wb") as f:
        f.write(CASE.pack(b"AMMP", 1, np.dtype(dtype).itemsize, ACT[activation],
                          n, m, len(hidden), 0, len(xs), init.size, lr))
        f.write(np.asarray(hidden, dtype="<u4").tobytes())
        f.write(init.astype(dtype).tobytes())
        f.write(np.concatenate([np.asarray(xs, dtype=dtype),
                                np.asarray(ys, dtype=dtype)], axis=1).tobytes())

    out = tmp_path / "res.bin"
    done = subprocess.run([str(tool), str(job), str(out)], capture_output=True, text=True)
    assert done.returncode == 0, f"mlpcheck failed: {done.stderr}"

    blob = out.read_bytes()
    magic, ver, dt, _pad, rn, rm, count = RESULT.unpack_from(blob, 0)
    assert magic == b"AMRS" and ver == 1 and (rn, rm) == (n, m) and count == len(xs)
    cpp_rows, off = [], RESULT.size
    for _ in range(count):
        a, reason, _t, _f, rnorm, rpost, gain, pt, pmd = ROW.unpack_from(blob, off)
        off += ROW.size
        cpp_rows.append({"applied": bool(a), "reason": reason, "residual_norm": rnorm,
                         "residual_post_norm": rpost, "gain_norm": gain})
    cpp_state = np.frombuffer(blob, dtype=dtype, count=init.size, offset=off)
    assert off + init.size * np.dtype(dtype).itemsize == len(blob), "trailing bytes"

    py_rows = []
    for x, y in zip(np.asarray(xs, dtype=dtype), np.asarray(ys, dtype=dtype)):
        r = ref.update(x, y)
        py_rows.append({"applied": r.applied,
                        "reason": 0 if r.reason is None else 1,
                        "residual_norm": float(np.linalg.norm(r.residual)),
                        "residual_post_norm": float(np.linalg.norm(r.residual_post)),
                        "gain_norm": r.gain_norm})
    return py_rows, np.asarray(ref.state), cpp_rows, cpp_state


def data(n, m, count, seed=7, scale=1.0):
    g = np.random.default_rng(seed)
    xs = g.standard_normal((count, n)) * scale
    ys = np.stack([np.sin(1.7 * xs[:, 0]) + 0.4 * xs[:, min(1, n - 1)] + j
                   for j in range(m)], axis=1)
    return xs, ys


def ulp_dev(a, b, dtype):
    scale = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))), 1e-30)
    return float(np.max(np.abs(a - b))) / scale / float(np.finfo(np.dtype(dtype)).eps)


@pytest.mark.parametrize("dtype", ["f4", "f8"])
@pytest.mark.parametrize("activation", ["tanh", "relu"])
@pytest.mark.parametrize("hidden,n,m", ARCHES)
def test_the_port_agrees_with_the_reference(mlpcheck, tmp_path, dtype, activation,
                                            hidden, n, m):
    xs, ys = data(n, m, 300)
    py, py_state, cpp, cpp_state = run_both(mlpcheck, tmp_path, hidden, n, m, dtype,
                                            activation, 0.01, xs, ys)

    assert len(py) == len(cpp) == 300
    eps = float(np.finfo(np.dtype(dtype)).eps)
    for i, (a, b) in enumerate(zip(py, cpp)):
        assert (a["applied"], a["reason"]) == (b["applied"], b["reason"]), i

    # Every reported scalar, not only the state block. Collecting a field and never
    # comparing it is the quietest way for it to go untested, and `gain_norm` reporting
    # the raw gradient instead of `lr * ||grad||` survived a whole run that way -- twice,
    # once in the reference and once here.
    for name, tol in (("residual_norm", 400.0), ("residual_post_norm", 400.0),
                      ("gain_norm", 400.0)):
        scale = max(max(abs(r[name]) for r in py), 1e-30)
        worst = max(abs(a[name] - b[name]) for a, b in zip(py, cpp)) / scale / eps
        assert worst < tol, f"{name} deviates by {worst:.1f} ulp"
        assert scale > 1e-9, (
            f"{name} is ~zero for the whole run, so comparing it proves nothing")

    dev = ulp_dev(py_state, cpp_state, dtype)
    assert dev < 400.0, f"{hidden} {activation} {dtype}: state deviates by {dev:.1f} ulp"
    # A control: without it a comparison of two zero blocks would pass everything.
    assert float(np.max(np.abs(py_state))) > 1e-3, "the weights never moved"


def test_the_deviation_is_reduction_order_and_tanh_is_not_the_problem(mlpcheck, tmp_path):
    """Classifying the difference rather than accepting a small number.

    Two things could produce it. Reduction order is the declared divergence every adapter
    here carries. `tanh` differing between numpy and libm would be a **new** one, and worse
    in kind, because a transcendental disagreement enters the recursion at every hidden
    unit rather than at a summation. `relu` is exact in both implementations, so comparing
    the two activations at the same architecture separates them.
    """
    got = {}
    for act in ("tanh", "relu"):
        for hidden in ((), (5,), (32,)):
            xs, ys = data(3, 1, 300)
            _p, ps, _c, cs = run_both(mlpcheck, tmp_path, hidden, 3, 1, "f8", act,
                                      0.01, xs, ys)
            got[(act, hidden)] = ulp_dev(ps, cs, "f8")

    assert got[("relu", ())] < 8.0, (
        f"a linear relu network — no transcendental, no hidden sum — already deviates by "
        f"{got[('relu', ())]:.1f} ulp, so something other than rounding differs")
    assert got[("tanh", (32,))] < max(got[("relu", (32,))] * 20.0, 40.0), (
        f"tanh at width 32 deviates {got[('tanh', (32,))]:.1f} ulp against relu's "
        f"{got[('relu', (32,))]:.1f} — that gap says numpy's tanh and libm's disagree "
        "enough to matter, which would be a fourth divergence and needs declaring")
    assert got[("relu", (32,))] >= got[("relu", ())], (
        "the deviation did not grow with the network, which is not the signature of a "
        "reduction-order difference and means this classification is wrong")


def test_the_exact_law_holds_on_both_sides(mlpcheck, tmp_path):
    """With no hidden layer the update is closed-form, so both sides must satisfy
    `e_post == (1 - 2*lr*(x.x + 1)) * e` — an equality rather than a tolerance, and the
    strongest statement available about a port."""
    xs, ys = data(4, 1, 120, seed=3)
    lr = 0.01
    py, _ps, cpp, _cs = run_both(mlpcheck, tmp_path, (), 4, 1, "f8", "tanh", lr, xs, ys)
    worst = 0.0
    for k, (a, b) in enumerate(zip(py, cpp)):
        factor = abs(1.0 - 2.0 * lr * (float(xs[k] @ xs[k]) + 1.0))
        want = factor * a["residual_norm"]
        for side in (a, b):
            worst = max(worst, abs(side["residual_post_norm"] - want) / max(want, 1e-30))
    assert worst < 1e-9, f"the law does not hold on both sides: {worst:.3e}"


def test_a_nonfinite_sample_is_rejected_identically(mlpcheck, tmp_path):
    xs, ys = data(3, 1, 200, seed=11)
    xs[50, 1] = np.nan
    ys[120, 0] = np.inf
    py, ps, cpp, cs = run_both(mlpcheck, tmp_path, (6,), 3, 1, "f8", "tanh", 0.01, xs, ys)
    assert [i for i, r in enumerate(py) if not r["applied"]] == [50, 120]
    assert [i for i, r in enumerate(cpp) if not r["applied"]] == [50, 120]
    assert ulp_dev(ps, cs, "f8") < 400.0, "rejection left the two states apart"


def test_the_state_layout_matches_so_a_block_crosses(mlpcheck, tmp_path):
    """One flat block in layer order, `W0 b0 W1 b1 ...`, W row-major `(in, out)`. If the
    two disagreed the port would still *run*, and would train to a different model — which
    is why this is asserted rather than assumed from a matching size."""
    xs, ys = data(4, 2, 40, seed=5)
    _py, ps, _cpp, cs = run_both(mlpcheck, tmp_path, (3,), 4, 2, "f8", "tanh", 0.01,
                                 xs, ys)
    assert ps.size == cs.size == (4 * 3 + 3) + (3 * 2 + 2)
    # Element-wise, not just in norm: a permuted layout can have the same norm.
    scale = max(float(np.max(np.abs(ps))), 1e-30)
    assert float(np.max(np.abs(ps - cs))) / scale < 1e-9, (
        "the blocks differ element by element, so the layouts are not the same")


COMPOSE = r"""
#include <stdio.h>
#include <cmath>
#include "argmind/host/runtime_shape_guard.hpp"
#include "argmind/host/runtime_shape_mlp.hpp"
using namespace argmind;
using namespace argmind::host;
int main() {
    RuntimeShapedMlp<double> net(3, 1, {16}, Activation::Tanh, 0.02, 7);
    const int sz = net.state_size();
    RuntimeShapedGuard<RuntimeShapedMlp<double>> g(std::move(net), 4, 4, 0, 10);

    Bound b; b.field = Field::ResidualNorm; b.hi = 8.0; b.has_hi = true;
    Bound p; p.field = Field::PTrace; p.hi = 1e6; p.has_hi = true;
    printf("state %d\n", sz);
    printf("residual %d\n", (int)g.add_bound(b));
    printf("ptrace %d\n", (int)!g.add_bound(p));

    unsigned s = 1; double x[3], y[1];
    int froze = -1;
    for (int k = 0; k < 4000; ++k) {
        for (int i = 0; i < 3; ++i) {
            s = s * 1664525u + 1013904223u;
            x[i] = (double)(s >> 8) / 8388608.0 - 1.0;
        }
        y[0] = std::sin(3 * x[0]) * std::exp(-x[1] * x[1]) + std::fabs(x[2]) - 0.4;
        if (k >= 3000 && k < 3100) y[0] = 40.0;      // sensor latched high
        if (g.update(x, y).frozen && froze < 0) froze = k;
    }
    printf("froze %d\n", froze);
    int finite = 1;
    for (int i = 0; i < sz; ++i) if (!(g.state()[i] == g.state()[i])) finite = 0;
    printf("finite %d\n", finite);
    return 0;
}
"""


def test_the_guard_composes_over_the_network_in_cpp(tmp_path):
    """The load-bearing claim of the scope decision, at the C++ level.

    The reference side is covered in `test_mlp.py`; this is the port. `Guard` and
    `Pipeline` were shown model-agnostic with NLMS -- a different algorithm with no
    covariance -- and a network is the harder case: layers, weights instead of a
    covariance, a backward pass. If `RuntimeShapedGuard` needs a single change for it, the
    contract is not one.
    """
    cxx = shutil.which("g++")
    if cxx is None:
        pytest.skip("no host g++")
    src = tmp_path / "compose.cpp"
    src.write_text(COMPOSE, encoding="utf-8")
    exe = tmp_path / "compose.exe"
    build = subprocess.run(
        [cxx, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Wpedantic",
         "-I", str(CPP / "include"), str(src), "-o", str(exe)],
        capture_output=True, text=True)
    assert build.returncode == 0, (
        "a network does not go through RuntimeShapedGuard unchanged:\n" + build.stderr)

    run = subprocess.run([str(exe)], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    got = dict(line.split() for line in run.stdout.splitlines() if line.strip())

    assert got["state"] == str(3 * 16 + 16 + 16 * 1 + 1)
    assert got["residual"] == "1", "the guard refused a field the network does report"
    assert got["ptrace"] == "1", (
        "the guard accepted a p_trace bound on an adapter with no covariance -- the same "
        "bound that was measured breaking in both directions at once")
    assert int(got["froze"]) == 3000, (
        f"the guard froze at {got['froze']}, not at the fault step 3000")
    assert got["finite"] == "1", "the guard froze but left a non-finite state"
