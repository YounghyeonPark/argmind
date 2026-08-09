# SPDX-License-Identifier: Apache-2.0
"""The checked-in examples have to build and run.

An example that no longer compiles is worse than none: it is documentation asserting
something false. `examples/hosted_inference.c` is the answer to "what does hosted
inference actually mean", and its value is entirely in being real, so it is built with
the same warnings-as-errors discipline as the library and its output is checked.
"""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
CPP = REPO / "cpp"
EXAMPLES = REPO / "examples"


def _run(cmd, **kw):
    return subprocess.run([str(c) for c in cmd], capture_output=True, text=True, **kw)


@pytest.fixture(scope="session")
def host_toolchain():
    cc, cxx = shutil.which("gcc"), shutil.which("g++")
    if cc is None or cxx is None:
        pytest.skip("no host gcc/g++")
    return cc, cxx


def test_the_hosted_inference_example_builds_and_converges(host_toolchain, tmp_path):
    """The whole hosted story in one program: a stand-in vendor runtime writes int8 into
    its own tensor, the caller dequantises to fp32 and pushes the embedding with the
    timestamp of the instant it is ABOUT, a slow sensor supplies the label later, and the
    guarded head learns. argmind never calls the runtime and never sees a graph.

    Built as C99 with -Wall -Wextra -Werror, because an example that only compiles under
    lenient flags is not the example an integrator will copy.
    """
    cc, cxx = host_toolchain
    obj = tmp_path / "hosted.o"
    done = _run([cc, "-std=c99", "-Wall", "-Wextra", "-Werror", "-O2",
                 "-I", CPP / "include", "-c", EXAMPLES / "hosted_inference.c", "-o", obj])
    assert done.returncode == 0, f"the example does not build:\n{done.stderr}"

    abi = tmp_path / "abi.o"
    done = _run([cxx, "-std=c++17", "-O2", "-fno-exceptions", "-fno-rtti",
                 "-I", CPP / "include", "-c", CPP / "src" / "abi.cpp", "-o", abi])
    assert done.returncode == 0, done.stderr

    exe = tmp_path / ("hosted.exe")
    done = _run([cxx, obj, abi, "-o", exe])
    assert done.returncode == 0, done.stderr

    out = _run([exe]).stdout
    assert "HOSTED OK" in out, out

    # ...and it has to have actually learned, not merely printed. The head is 128 wide,
    # so a run that produced no updates would sail past a loose check.
    m = re.search(r"updates (\d+)\s+predictions (\d+)", out)
    assert m, out
    updates, predictions = int(m.group(1)), int(m.group(2))
    assert updates > 1000, f"only {updates} updates; the example is not exercising the head"
    assert predictions == updates, (
        f"{predictions} predictions against {updates} updates -- the report is supposed to "
        "carry an output for every frame, including ones the gate refused to learn from")

    m = re.search(r"error\s+(-?[\d.e+-]+)", out)
    assert m, out
    assert abs(float(m.group(1))) < 1e-3, f"the head did not converge: {m.group(1)}"
