# SPDX-License-Identifier: Apache-2.0
"""The runtime-shaped pipeline, against the same Python reference.

`shapepipecheck` reads and writes `pipecheck`'s job and result formats byte for byte, so
this module reuses that harness's `Case`, `both` and `compare` rather than growing a
second encoder that could drift. Every field in the job header was already a runtime
value; the only thing that made `pipecheck` compile-time was its dispatch table.

**A pipeline has to agree about bookkeeping, not arithmetic.** It is driven by two
independent clocks -- samples arriving and `step()` being called -- so the event log
interleaves pushes and steps, which is the only way to reach a ring that overflowed
between two steps or a guard that froze half-way through one.

The first version of this file called `both()` on cases with **no events at all** and
reported four agreements out of four. `steps=0` on every one of them: a vacuous pass,
and the third time this session that a comparison ran against nothing. The driver below
exists so that cannot happen quietly, and every case asserts it moved data.
"""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from test_cpp_pipeline_parity import Case, both, compare

REPO = Path(__file__).resolve().parent.parent
CPP = REPO / "cpp"


@pytest.fixture(scope="session")
def shapepipecheck(tmp_path_factory):
    cxx = shutil.which("g++")
    if cxx is None:
        pytest.skip("no host g++")
    exe = tmp_path_factory.mktemp("shapepipe") / "shapepipecheck.exe"
    done = subprocess.run(
        [cxx, "-std=c++17", "-O2", "-Wall", "-Wextra", "-I", str(CPP / "include"),
         str(CPP / "tools" / "shapepipecheck.cpp"), "-o", str(exe)],
        capture_output=True, text=True)
    assert done.returncode == 0, f"shapepipecheck does not build:\n{done.stderr}"
    return exe


def drive(case, n=200, ref_every=1, seed=4):
    """Push one sample into every stream per tick, and step every `ref_every` ticks.

    `ref_every > 1` is what makes the reference ring overflow, which is the only way to
    reach the lost-frame path -- and that path is the pipeline's least obvious
    bookkeeping, so a suite without it would be testing the easy half.
    """
    rng = np.random.default_rng(seed)
    t = 0.0
    for k in range(n):
        for s, d in enumerate(case.dims):
            case.push(s, t, [float(rng.standard_normal()) for _ in range(d)])
        if k % ref_every == 0:
            case.step()
        t += 0.01
    return case


def _totals(py):
    st = py["steps"]
    return (len(st), sum(s["frames"] for s in st), sum(s["updated"] for s in st),
            sum(s["lost_frames"] for s in st))


def _agree(shapepipecheck, tmp_path, case, *, want_lost=False):
    py, cpp = both(shapepipecheck, tmp_path, case)
    compare(case, py, cpp)
    steps, frames, updated, lost = _totals(py)
    # The guard against the failure that produced this file: a case with no events
    # compares two empty lists and passes.
    assert steps > 0 and frames > 0 and updated > 0, (
        f"nothing happened: steps={steps} frames={frames} updated={updated}")
    if want_lost:
        assert lost > 0, "the ring never overflowed, so the lost-frame path was not reached"
    return steps, frames, updated, lost


@pytest.mark.parametrize("dtype", ["f4", "f8"])
def test_a_bare_runtime_pipeline_agrees(shapepipecheck, tmp_path, dtype):
    case = drive(Case([1, 2], 64, [(1, 0), (1, 1)], (0, 0), constant=True, dtype=dtype))
    _agree(shapepipecheck, tmp_path, case)


def test_a_guarded_runtime_pipeline_agrees(shapepipecheck, tmp_path):
    case = drive(Case([1, 1], 64, [(1, 0)], (0, 0), constant=True,
                      guard={"depth": 4, "checkpoint_every": 3,
                             "bounds": [("p_trace", None, 1e4)]}))
    _agree(shapepipecheck, tmp_path, case)


def test_an_overflowing_reference_ring_loses_the_same_frames(shapepipecheck, tmp_path):
    """`lost_frames` is the pipeline's least obvious figure: reference samples that were
    overwritten before they could become frames. A capacity-4 ring stepped once every
    five pushes reaches it -- measured at 39 lost over 40 steps when this was written."""
    case = drive(Case([1, 1], 4, [(1, 0)], (0, 0), constant=True), ref_every=5)
    _agree(shapepipecheck, tmp_path, case, want_lost=True)


def test_three_streams_of_different_widths_agree(shapepipecheck, tmp_path):
    case = drive(Case([1, 3, 2], 64, [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1)], (0, 0)))
    _agree(shapepipecheck, tmp_path, case)


def test_the_input_gates_agree(shapepipecheck, tmp_path):
    """Two gates at two different times, and this is the earlier one: the pipeline
    refuses to LEARN from a stale or invalid frame while still reporting it."""
    case = Case([1, 1], 64, [(1, 0)], (0, 0), constant=True, max_stale_pipeline=0.015)
    rng = np.random.default_rng(9)
    t = 0.0
    for k in range(120):
        case.push(0, t, [float(rng.standard_normal())])
        # the aiding stream arrives at a third of the rate, so staleness grows and the
        # gate fires on the frames between its samples
        if k % 3 == 0:
            case.push(1, t, [float(rng.standard_normal())], valid=(k % 21 != 0))
        case.step()
        t += 0.01
    py, cpp = both(shapepipecheck, tmp_path, case)
    compare(case, py, cpp)
    skipped = [sum(s["skipped"][i] for s in py["steps"]) for i in range(8)]
    assert sum(skipped) > 0, (
        f"no frame was gated, so this case tests nothing it claims: {skipped}")


def test_an_instant_that_cannot_build_a_frame_agrees(shapepipecheck, tmp_path):
    """The axis every other case here was missing, and it hid three bugs.

    Nothing in the cases above ever *declines* an instant: every stream has a sample at
    every reference instant, so `pick` never fails. Three deliberate bugs survived
    because of it -- not advancing the aligner past a declined instant, and predicting
    only `if (frames && frame_ready_)`, which is the exact bug a parity review found in
    the core (118 disagreeing steps in 250).

    An aiding stream that starts late reaches it: the early reference instants have no
    aiding sample, `pick` fails, and with several instants per step a step can contain
    both a failure and a success. Widening this way also found a real port bug --
    `n_staleness` was derived from `ok` rather than from whether a frame was built, and
    disagreed on the very first step.
    """
    case = Case([1, 1], 64, [(1, 0)], (0, 0), constant=True)
    rng = np.random.default_rng(11)
    t = 0.0
    for k in range(150):
        case.push(0, t, [float(rng.standard_normal())])
        if k >= 20:                       # the aiding stream is late
            case.push(1, t, [float(rng.standard_normal())])
        if k % 3 == 0:                    # several instants per step
            case.step()
        t += 0.01

    py, cpp = both(shapepipecheck, tmp_path, case)
    compare(case, py, cpp)
    steps = py["steps"]
    empty = sum(1 for s in steps if s["frames"] == 0)
    assert empty > 0, (
        "every step produced a frame, so no instant was declined and this case is the "
        "same as the others")
    assert sum(s["frames"] for s in steps) > 0, "and some had to succeed"
    # A step that reports no frame must report no staleness either -- not `ok`-derived.
    for i, s in enumerate(steps):
        if s["frames"] == 0:
            assert s["n_staleness"] == 0, f"step {i} reported staleness with no frame"
