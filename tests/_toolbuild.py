# SPDX-License-Identifier: Apache-2.0
"""Building a parity tool for the host or for the declared target, behind one call.

The four parity suites each built their tool for x86-64 and compared it against the Python
reference. A separate file then ran a hand-picked dozen cases against a Cortex-M4 build,
justified by "each QEMU invocation is about a second".

**That justification was wrong by six times.** A parity review measured 157 ms per invocation
and ran all four suites on the target in about 38 seconds including the cross build. With no
cost argument for sampling, there is no reason for the target to see a subset: every parity
case now runs on both architectures, and the suites say `[host]` and `[target]` beside each
test rather than one of them standing in for the other.

The review also showed why sampling was worse than slow. The old subset passed unchanged when
the host binaries were substituted for the target ones, and passed again with the target's FPU
set to round-toward-zero — a change that costs nineteen times in state deviation. A dozen
easy cases discriminate nothing; the full suite at least runs the cases that were designed to
fail.

Nothing here changes the tools. Semihosting carries `argv` in and file I/O out, so a target
build is invoked through a wrapper that looks like a program taking `(job, out)` — which is
exactly what the harnesses already call.
"""

import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
CPP = REPO / "cpp"
QEMU_DIR = CPP / "target" / "qemu"

#: The declared floor. `mps2-an386` is QEMU's Cortex-M4 board; naming the CPU explicitly means
#: a QEMU that defaulted to something else fails rather than quietly testing an M3.
QEMU_ARGS = ["-M", "mps2-an386", "-cpu", "cortex-m4", "-nographic"]

TARGET_CFLAGS = ["-mcpu=cortex-m4", "-mfpu=fpv4-sp-d16", "-mfloat-abi=hard",
                 "-fno-math-errno", "-O2", "-ffunction-sections", "-fdata-sections"]

#: Both architectures, as a fixture `params` list. Import and use directly so the four suites
#: cannot drift apart on which ones they cover.
ARCHES = ["host", "target"]


def host_compiler():
    for name in ("g++", "clang++"):
        if shutil.which(name):
            return name
    return None


def _cross(name):
    return shutil.which(f"arm-none-eabi-{name}")


def _run(cmd):
    return subprocess.run([str(c) for c in cmd], capture_output=True, text=True)


def build_tool(name: str, arch: str, work: Path) -> Path:
    """A parity tool built for `arch`, or a skip. `work` should be session-scoped.

    Returns something the harnesses can call as `[tool, job, out]`: the executable itself for
    the host, and for the target a wrapper that hands the two paths to QEMU as semihosting
    arguments.
    """
    if arch == "host":
        cxx = host_compiler()
        if cxx is None:
            pytest.skip("no C++ compiler on PATH; the Python reference stands alone")
        out = work / (f"{name}.exe" if sys.platform == "win32" else name)
        done = _run([cxx, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Wpedantic",
                     "-fno-exceptions", "-fno-rtti", "-I", CPP / "include",
                     CPP / "tools" / f"{name}.cpp", "-o", out])
        if done.returncode != 0:
            pytest.fail(f"{name} did not build for the host:\n{done.stdout}\n{done.stderr}")
        return out

    qemu = shutil.which("qemu-system-arm")
    if qemu is None or _cross("g++") is None:
        pytest.skip("need arm-none-eabi and qemu-system-arm on PATH for the target build")

    cc, cxx = _cross("gcc"), _cross("g++")
    flags = [*TARGET_CFLAGS, "-I", str(CPP / "include")]
    startup = work / "startup.o"
    if not startup.exists():
        done = _run([cc, "-std=c99", *flags, "-c", QEMU_DIR / "startup.c", "-o", startup])
        assert done.returncode == 0, done.stderr

    obj = work / f"{name}.o"
    done = _run([cxx, "-std=c++17", *flags, "-fno-exceptions", "-fno-rtti",
                 "-c", CPP / "tools" / f"{name}.cpp", "-o", obj])
    if done.returncode != 0:
        pytest.fail(f"{name} did not build for the target:\n{done.stderr}")

    elf = work / f"{name}.elf"
    done = _run([cxx, "-mcpu=cortex-m4", "-mfpu=fpv4-sp-d16", "-mfloat-abi=hard",
                 "--specs=rdimon.specs", "--specs=nano.specs", "-Wl,--gc-sections",
                 "-T", QEMU_DIR / "mps2.ld", startup, obj, "-o", elf])
    stderr = "\n".join(l for l in done.stderr.splitlines() if "RWX permissions" not in l)
    if done.returncode != 0:
        pytest.fail(f"{name} did not link for the target:\n{stderr}")

    # QEMU's semihosting command line truncates past roughly 255 characters and the tool then
    # reports a usage error, which reads to the harness as "the tool failed". Nothing here can
    # shorten the paths pytest hands out, so it is written down rather than worked around.
    sem = ("enable=on,target=native,arg={n},arg=%~1,arg=%~2" if sys.platform == "win32"
           else "enable=on,target=native,arg={n},arg=$1,arg=$2").format(n=name)
    if sys.platform == "win32":
        w = work / f"{name}_arm.cmd"
        w.write_text(f'@echo off\r\n"{qemu}" ' + " ".join(QEMU_ARGS) +
                     f' -semihosting-config "{sem}" -kernel "{elf}"\r\n', encoding="ascii")
    else:
        w = work / f"{name}_arm.sh"
        w.write_text("#!/bin/sh\nexec " + qemu + " " + " ".join(QEMU_ARGS) +
                     f" -semihosting-config '{sem}' -kernel '{elf}'\n", encoding="ascii")
        w.chmod(w.stat().st_mode | stat.S_IEXEC)
    return w
