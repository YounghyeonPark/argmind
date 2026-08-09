# SPDX-License-Identifier: Apache-2.0
"""The core, actually running on a Cortex-M4.

"Execution is host-only" was this project's standing verification limit from the day the
target range was settled. It is not any more: `cpp/tests/test_core.cpp` now runs under
QEMU's `mps2-an386` — the declared floor, ARMv7E-M with a single-precision FPU — and the
same checks that pass on x86-64 pass there.

**What this is worth, concretely.** `persist.hpp` guards `count * sizeof(T)` against
wrapping, and its own comment records that the guard is an *equivalent mutant* on a 64-bit
host: the threshold is 2.3e18 for double and `count` is a `uint32_t`, so nothing can reach
it. On a 32-bit target the same expression is 536,870,876. Measured here by removing the
guard and running both ways: **the mutant fails on ARM with two failures and passes on
x86-64 with none.** A guard that could only be argued for is now one that is tested.

**What this is not.** QEMU models the instruction set, not the pipeline, the flash wait
states or the bus. A cycle count taken here would be fiction dressed as measurement, so
none is taken, and the on-target timing claim stays absent exactly as it was. The Windows
build also ships no TCG plugins, so even an instruction count — which would be honest,
being architectural rather than temporal — needs a plugin built first and is not attempted.

Skipped, not failed, without `qemu-system-arm` or the cross toolchain on PATH.
"""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
CPP = REPO / "cpp"
QEMU_DIR = CPP / "target" / "qemu"

#: The declared floor. `mps2-an386` is QEMU's Cortex-M4 board; the CPU is named explicitly so
#: a QEMU that defaulted to something else would fail rather than quietly test an M3.
QEMU_ARGS = ["-M", "mps2-an386", "-cpu", "cortex-m4", "-nographic",
             "-semihosting-config", "enable=on,target=native"]

TARGET_CFLAGS = ["-mcpu=cortex-m4", "-mfpu=fpv4-sp-d16", "-mfloat-abi=hard",
                 "-fno-math-errno", "-O2", "-ffunction-sections", "-fdata-sections"]


def _tool(name):
    return shutil.which(f"arm-none-eabi-{name}")


@pytest.fixture(scope="session")
def emulator():
    q = shutil.which("qemu-system-arm")
    if q is None:
        pytest.skip("no qemu-system-arm on PATH")
    if _tool("g++") is None:
        pytest.skip("no arm-none-eabi toolchain on PATH")
    return q


def _run(cmd, **kw):
    return subprocess.run([str(c) for c in cmd], capture_output=True, text=True, **kw)


def build_image(tmp_path, extra_cflags=()):
    """Compile the self-check suite for the target and link it against the QEMU harness."""
    cxx, cc = _tool("g++"), _tool("gcc")
    flags = [*TARGET_CFLAGS, *extra_cflags, "-I", str(CPP / "include")]
    objs = []
    done = _run([cc, "-std=c99", *flags, "-c", QEMU_DIR / "startup.c",
                 "-o", tmp_path / "startup.o"])
    assert done.returncode == 0, done.stderr
    objs.append(tmp_path / "startup.o")
    for src in (CPP / "tests" / "test_core.cpp", CPP / "src" / "abi.cpp"):
        obj = tmp_path / (src.stem + ".o")
        done = _run([cxx, "-std=c++17", *flags, "-fno-exceptions", "-fno-rtti",
                     "-c", src, "-o", obj])
        assert done.returncode == 0, f"{src.name} does not build:\n{done.stderr}"
        objs.append(obj)

    elf = tmp_path / "selfcheck.elf"
    done = _run([cxx, "-mcpu=cortex-m4", "-mfpu=fpv4-sp-d16", "-mfloat-abi=hard",
                 "--specs=rdimon.specs", "--specs=nano.specs", "-Wl,--gc-sections",
                 "-T", QEMU_DIR / "mps2.ld", *objs, "-o", elf])
    # The RWX warning is the linker noticing that this harness's script does not separate
    # code from data. It is a test image, not a product one.
    stderr = "\n".join(l for l in done.stderr.splitlines() if "RWX permissions" not in l)
    assert done.returncode == 0, f"the target image does not link:\n{stderr}"
    return elf


def run_image(emulator, elf, timeout=600):
    done = _run([emulator, *QEMU_ARGS, "-kernel", elf], timeout=timeout)
    return done.stdout + done.stderr


def test_the_self_checks_pass_on_a_cortex_m4(emulator, tmp_path):
    """The whole suite, on the declared floor. Not a subset and not a smoke test: if the
    two architectures disagreed anywhere -- AAPCS layout, single-precision FPU behaviour,
    32-bit `size_t`, anything -- this is where it would show."""
    out = run_image(emulator, build_image(tmp_path))
    m = re.search(r"(\d+) checks, (\d+) failures", out)
    assert m, f"the image produced no result line:\n{out[-3000:]}"
    checks, failures = int(m.group(1)), int(m.group(2))
    assert failures == 0, f"{failures} of {checks} checks failed on ARM:\n{out[-3000:]}"
    # ...and it has to be the whole suite. A harness that silently ran a fraction of it
    # would report "0 failures" just as happily.
    assert checks >= 4600, (
        f"only {checks} checks ran on the target; the host runs the same file")
    assert "FAULT" not in out, out[-2000:]


def test_the_target_is_a_32_bit_environment(emulator, tmp_path):
    """The property that makes the next test possible, asserted separately so that a
    toolchain or board change which quietly widened `size_t` is a distinct failure rather
    than a confusing one."""
    src = tmp_path / "widths.c"
    src.write_text(
        '#include <stdio.h>\n'
        'int main(void) {\n'
        '  printf("widths %d %d %d %d\\n", (int)sizeof(int), (int)sizeof(long),\n'
        '         (int)sizeof(void*), (int)sizeof(size_t));\n'
        '  return 0;\n'
        '}\n', encoding="utf-8")
    cc, cxx = _tool("gcc"), _tool("g++")
    _run([cc, "-std=c99", *TARGET_CFLAGS, "-c", src, "-o", tmp_path / "widths.o"])
    _run([cc, "-std=c99", *TARGET_CFLAGS, "-c", QEMU_DIR / "startup.c",
          "-o", tmp_path / "startup.o"])
    elf = tmp_path / "widths.elf"
    done = _run([cxx, "-mcpu=cortex-m4", "-mfpu=fpv4-sp-d16", "-mfloat-abi=hard",
                 "--specs=rdimon.specs", "--specs=nano.specs", "-Wl,--gc-sections",
                 "-T", QEMU_DIR / "mps2.ld", tmp_path / "startup.o",
                 tmp_path / "widths.o", "-o", elf])
    assert done.returncode == 0, done.stderr
    out = run_image(emulator, elf, timeout=120)
    assert "widths 4 4 4 4" in out, out


def test_the_32_bit_overflow_guard_is_no_longer_an_equivalent_mutant(emulator, tmp_path):
    """The finding that justified running on the target at all.

    `persist.hpp`'s wraparound guard is documented as untestable on a 64-bit host, and it
    was. Here the guard is removed and the suite is run on ARM: it must fail. The same
    mutation on the host is asserted to pass, because that is the claim the header makes
    and an unverified claim about untestability is still an unverified claim.

    This test edits a copy of the header in a temporary tree. It does not touch the working
    copy -- an agent once ran `git checkout --` in this repository and discarded uncommitted
    work, and the lesson generalises to tests that mutate sources.
    """
    guard = ("    if (count > (SIZE_MAX - HEADER_BYTES - CRC_BYTES - cause_len) / sizeof(T))"
             "\n        return Status::LengthMismatch;")
    header = CPP / "include" / "argmind" / "persist.hpp"
    original = header.read_text(encoding="utf-8")
    assert original.count(guard) == 1, "the guard has moved; this test names it literally"

    shadow = tmp_path / "include"
    shutil.copytree(CPP / "include", shadow)
    (shadow / "argmind" / "persist.hpp").write_text(
        original.replace(guard, "    /* guard removed by the test */"), encoding="utf-8")

    cxx, cc = _tool("g++"), _tool("gcc")
    flags = ["-I", str(shadow)]
    objs = [tmp_path / "startup.o"]
    _run([cc, "-std=c99", *TARGET_CFLAGS, *flags, "-c", QEMU_DIR / "startup.c",
          "-o", objs[0]])
    for src in (CPP / "tests" / "test_core.cpp", CPP / "src" / "abi.cpp"):
        obj = tmp_path / ("mut_" + src.stem + ".o")
        done = _run([cxx, "-std=c++17", *TARGET_CFLAGS, *flags, "-fno-exceptions",
                     "-fno-rtti", "-c", src, "-o", obj])
        assert done.returncode == 0, done.stderr
        objs.append(obj)
    elf = tmp_path / "mutant.elf"
    done = _run([cxx, "-mcpu=cortex-m4", "-mfpu=fpv4-sp-d16", "-mfloat-abi=hard",
                 "--specs=rdimon.specs", "--specs=nano.specs", "-Wl,--gc-sections",
                 "-T", QEMU_DIR / "mps2.ld", *objs, "-o", elf])
    assert done.returncode == 0, done.stderr

    out = run_image(emulator, elf)
    m = re.search(r"(\d+) checks, (\d+) failures", out)
    assert m, out[-2000:]
    assert int(m.group(2)) > 0, (
        "removing the wraparound guard still passed on ARM, so either the guard is dead "
        "there too or the case that reaches it has stopped reaching it")

    # ...and the other half of the claim: on this 64-bit host the same mutation is invisible.
    host = shutil.which("g++") or shutil.which("clang++")
    if host is None:
        pytest.skip("no host C++ compiler to complete the comparison")
    host_exe = tmp_path / ("mutant_host.exe")
    done = _run([host, "-std=c++17", "-O2", "-fno-exceptions", "-fno-rtti", *flags,
                 CPP / "tests" / "test_core.cpp", CPP / "src" / "abi.cpp",
                 "-o", host_exe])
    assert done.returncode == 0, done.stderr
    host_out = _run([host_exe]).stdout
    hm = re.search(r"(\d+) checks, (\d+) failures", host_out)
    assert hm and int(hm.group(2)) == 0, (
        "the guard turned out to be reachable on the host after all, which would make "
        f"persist.hpp's 'equivalent mutant' note wrong: {host_out}")
