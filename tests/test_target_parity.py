# SPDX-License-Identifier: Apache-2.0
"""Parity between the Python reference and the C++ core **built for the target**.

Every parity claim this project has made was measured x86-64 against x86-64. That is a real
gap and the record format makes it obvious: `persist.hpp` declares that "the writer and the
reader may be different machines" and had only ever been checked on one machine. The others
are quieter but the same shape — AAPCS lays structs out differently, `size_t` is half the
width, and a single-precision FPU is not an SSE unit.

So the four parity tools are built for Cortex-M4 and run under QEMU against the *same* Python
reference the host build is compared to. Each is invoked through a small wrapper that looks
like an executable taking `(job, out)`, which is exactly what the existing harnesses already
call — so this file reuses their encoders and their comparison functions rather than growing
a second set that could drift.

**What runs here is a curated subset, not the whole suite**, and the reason is cost: each
QEMU invocation is about a second, and the host suites sweep hundreds of configurations. The
subset is chosen to cover each tool's distinguishing claim — byte identity for the record
format, ulp-level agreement for the estimator, decision agreement for the aligner, and
counter agreement for the pipeline. Saying "parity holds on ARM" without saying which cases
ran would be the kind of unbounded claim this project keeps having to withdraw.

Skipped, not failed, without the cross toolchain and `qemu-system-arm` on PATH.
"""

import shutil
import stat
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import test_cpp_parity as X
import test_cpp_persist_parity as PS
import test_cpp_pipeline_parity as PL
import test_cpp_stream_parity as ST

from argmind import persist
from argmind.adapt import RecursiveLeastSquares

REPO = Path(__file__).resolve().parent.parent
CPP = REPO / "cpp"
QEMU_DIR = CPP / "target" / "qemu"

TARGET_CFLAGS = ["-mcpu=cortex-m4", "-mfpu=fpv4-sp-d16", "-mfloat-abi=hard",
                 "-fno-math-errno", "-O2", "-ffunction-sections", "-fdata-sections"]
QEMU_ARGS = ["-M", "mps2-an386", "-cpu", "cortex-m4", "-nographic"]


def _tool(name):
    return shutil.which(f"arm-none-eabi-{name}")


def _run(cmd, **kw):
    return subprocess.run([str(c) for c in cmd], capture_output=True, text=True, **kw)


@pytest.fixture(scope="session")
def arm(tmp_path_factory):
    """Build all four parity tools for the target and hand back callable wrappers."""
    qemu = shutil.which("qemu-system-arm")
    if qemu is None or _tool("g++") is None:
        pytest.skip("need arm-none-eabi and qemu-system-arm on PATH")

    work = tmp_path_factory.mktemp("armtools")
    cc, cxx = _tool("gcc"), _tool("g++")
    flags = [*TARGET_CFLAGS, "-I", str(CPP / "include")]

    done = _run([cc, "-std=c99", *flags, "-c", QEMU_DIR / "startup.c",
                 "-o", work / "startup.o"])
    assert done.returncode == 0, done.stderr

    wrappers = {}
    for name in ("xcheck", "pcheck", "scheck", "pipecheck"):
        obj = work / f"{name}.o"
        done = _run([cxx, "-std=c++17", *flags, "-fno-exceptions", "-fno-rtti",
                     "-c", CPP / "tools" / f"{name}.cpp", "-o", obj])
        assert done.returncode == 0, f"{name} does not build for the target:\n{done.stderr}"
        elf = work / f"{name}.elf"
        done = _run([cxx, "-mcpu=cortex-m4", "-mfpu=fpv4-sp-d16", "-mfloat-abi=hard",
                     "--specs=rdimon.specs", "--specs=nano.specs", "-Wl,--gc-sections",
                     "-T", QEMU_DIR / "mps2.ld", work / "startup.o", obj, "-o", elf])
        stderr = "\n".join(l for l in done.stderr.splitlines()
                           if "RWX permissions" not in l)
        assert done.returncode == 0, f"{name} does not link:\n{stderr}"

        # The harnesses call `subprocess.run([tool, job, out])`, so the wrapper has to look
        # like a program taking two paths. Semihosting carries argv in and file I/O out, so
        # the tools themselves need no change at all -- which is the point: what runs on ARM
        # is the same source the host runs.
        sem = ("enable=on,target=native,arg={name},arg=%~1,arg=%~2" if sys.platform == "win32"
               else "enable=on,target=native,arg={name},arg=$1,arg=$2").format(name=name)
        if sys.platform == "win32":
            w = work / f"{name}.cmd"
            w.write_text(f'@echo off\r\n"{qemu}" ' + " ".join(QEMU_ARGS) +
                         f' -semihosting-config "{sem}" -kernel "{elf}"\r\n', encoding="ascii")
        else:
            w = work / f"{name}.sh"
            w.write_text("#!/bin/sh\nexec " + qemu + " " + " ".join(QEMU_ARGS) +
                         f" -semihosting-config '{sem}' -kernel '{elf}'\n", encoding="ascii")
            w.chmod(w.stat().st_mode | stat.S_IEXEC)
        wrappers[name] = w
    return wrappers


# -- the record format: the portability claim, finally tested across machines -------------


@pytest.mark.parametrize("dtype,n,m", [("f4", 6, 1), ("f8", 6, 1), ("f4", 12, 3)])
def test_the_record_bytes_are_identical_on_the_target(arm, tmp_path, dtype, n, m):
    """`persist.hpp` says the writer and the reader may be different machines. Until now
    both were this one. The comparison is byte identity, as it is on the host, because
    serialisation has no rounding to hide behind."""
    a = RecursiveLeastSquares(n, m, dtype=dtype)
    rng = np.random.default_rng(11)
    for _ in range(30):
        a.update(rng.standard_normal(n).astype(dtype),
                 rng.standard_normal(m).astype(dtype))
    reference = persist.pack(a, sequence=9, fingerprint=0xFEEDFACE)
    got = PS.cpp_pack(arm["pcheck"], tmp_path, np.array(a.state), n_features=n,
                      n_outputs=m, sequence=9, fingerprint=0xFEEDFACE, frozen=False,
                      cause=b"", dtype=dtype)
    assert bytes(got) == bytes(reference), (
        f"{dtype} n={n} m={m}: the target packs different bytes")


def test_the_target_reads_a_record_the_host_wrote(arm, tmp_path):
    """The direction that matters for a device: a block written elsewhere, restored here.
    Everything the header carries is compared, not just the state."""
    a = RecursiveLeastSquares(6, 1, dtype="f4")
    rng = np.random.default_rng(3)
    for _ in range(25):
        a.update(rng.standard_normal(6).astype("f4"), rng.standard_normal(1).astype("f4"))
    blob = persist.pack(a, sequence=4, fingerprint=0x1234ABCD)

    status, fields, cause, state = PS.cpp_unpack(arm["pcheck"], tmp_path, blob, dtype="f4")
    assert status == "ok", status
    assert fields["sequence"] == 4
    assert fields["fingerprint"] == 0x1234ABCD
    assert fields["n_features"] == 6 and fields["n_outputs"] == 1
    assert np.array_equal(state, np.array(a.state))

    # ...and a corrupt one is still refused, so the agreement above is not the CRC being
    # ignored on a machine where the arithmetic differs.
    torn = bytearray(blob)
    torn[len(torn) // 2] ^= 0x01
    status, _f, _c, _s = PS.cpp_unpack(arm["pcheck"], tmp_path, bytes(torn), dtype="f4")
    assert status == "crc_mismatch", status


# -- the estimator: the same numbers, on a different FPU -----------------------------------


@pytest.mark.parametrize("dtype", ["f4", "f8"])
def test_rls_agrees_with_the_reference_on_the_target(arm, tmp_path, dtype):
    """x86-64 does this arithmetic on SSE and the target on a VFPv4 single-precision unit.
    Both are IEEE-754, so they should agree to the same handful of ulp the host build does
    -- but "should" is what this project measures instead of assuming."""
    xs, ys = X.excited(6, 300, seed=41)
    py_rows, py_state, cpp_rows, cpp_state = X.run_both(
        arm["xcheck"], tmp_path, xs, ys, lam=0.99, p0=100.0, dtype=dtype)
    dev = X.scale_deviation(py_state, cpp_state)
    assert dev <= X.ulps(dtype), (
        f"{dtype}: the target's state block deviates by "
        f"{dev / float(np.finfo(np.dtype(dtype)).eps):.1f} ulp")
    assert len(cpp_rows) == len(py_rows)
    for field, tol in X.FIELD_ULP.items():
        a = np.array([r[field] for r in py_rows])
        b = np.array([r[field] for r in cpp_rows])
        assert X.scale_deviation(a, b) <= X.ulps(dtype, tol), field


def test_the_guard_makes_the_same_decisions_on_the_target(arm, tmp_path):
    """A guard decides rather than computes, so a decision that differs is not a rounding
    difference. Freeze step, thaw step and the tripped set have to match exactly."""
    xs, ys = X.excited(6, 400, seed=17)
    ys = ys.copy()
    ys[250] += 500.0
    py_rows, _py_state, cpp_rows, _cpp_state = X.run_both(
        arm["xcheck"], tmp_path, xs, ys, lam=0.995, p0=10.0, dtype="f4",
        guard=X.GUARD_DIVERGENCE)
    py_frozen = [r["frozen"] for r in py_rows]
    cpp_frozen = [r["frozen"] for r in cpp_rows]
    assert py_frozen == cpp_frozen, "the two froze at different steps on the target"
    assert any(py_frozen), "the fixture never froze, so this compares nothing"
    assert [r["applied"] for r in py_rows] == [r["applied"] for r in cpp_rows]


# -- alignment and the pipeline ------------------------------------------------------------


@pytest.mark.parametrize("policy", ["hold_last", "nearest", "interpolate"])
def test_alignment_picks_the_same_samples_on_the_target(arm, tmp_path, policy):
    """Reuses the host suite's own comparison, which comes down to bit patterns for
    staleness and raw bytes for values."""
    rng = np.random.default_rng(8)
    specs = [(1, 16), (3, 16)]
    pushes = []
    for k in range(1, 25):
        pushes.append((0, k * 0.5, [float(k)], True))
        pushes.append((1, k * 0.5 - 0.17, list(rng.normal(size=3)), k % 7 != 0))
    queries = [0.3 + 0.25 * i for i in range(40)]
    ST.compare(arm["scheck"], tmp_path, specs, pushes, queries, policy=policy)


def test_the_pipeline_agrees_step_for_step_on_the_target(arm, tmp_path):
    """The whole bookkeeping comparison the host suite runs, against a target build: frames,
    updated, lost_frames, every skip counter, the tripped mask, staleness by bit pattern and
    the output."""
    case = PL.simple(dtype="f4", capacity=64)
    PL.both(arm["pipecheck"], tmp_path, case)


def test_the_freshness_gate_agrees_on_the_target(arm, tmp_path):
    case = PL.Case([1, 1], 64, [(0, 0)], (1, 0), constant=True,
                   require_fresh_target=True)
    for k in range(1, 41):
        case.push(0, k / 10.0, [float(k % 7)])
        if k % 10 == 0:
            case.push(1, k / 10.0, [2.0 + 0.01 * k])
        case.step()
    py, cpp = PL.both(arm["pipecheck"], tmp_path, case)
    gated = sum(s["skipped"][PL.SKIP.index("stale_target")] for s in cpp["steps"])
    assert gated > 0, "no frame was gated, so this compares nothing"
    assert py["totals"]["updated"] == 4
