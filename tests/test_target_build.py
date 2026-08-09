# SPDX-License-Identifier: Apache-2.0
"""The declared target, compiled and linked for real.

Every size figure this project quoted before 2026-08-09 was an x86-64 shape, because there
was no ARM toolchain on the machine. DESIGN.md said so each time and said flash and RAM would
have to come from a **link map**. This is that link, and it found three things a host build
cannot:

1. `norm()` compiles to a call into libm unless `-fno-math-errno` is given. It is not
   `-ffreestanding` that does it — GCC assumes `sqrtf` must set `errno` on a negative
   argument, so it will not use the FPU's `VSQRT.F32`. `Rls::update` calls `norm()` three
   times.
2. The core needs `memcpy` and `memset` from its environment. GCC is permitted to emit calls
   to those even in a freestanding build, and it does — 48 and 69 call sites. A bare-metal
   integrator has to provide them; nothing in this project said so before.
3. The checked-in CMake cross recipe did not work, having been written before any toolchain
   existed. `project()`'s compiler check links a full executable, which fails bare-metal.

Skipped, not failed, without `arm-none-eabi-gcc` on PATH: a clone with no cross toolchain
still tests everything else, the same arrangement `test_cpp_parity.py` uses for the host
compiler.

**No timing figure appears here or anywhere else.** There is no hardware, and QEMU -- which
`tests/test_target_run.py` now drives -- models the instruction set rather than the pipeline,
so a cycle count taken either way would be interpolated rather than measured. DESIGN.md is
explicit that an on-target timing claim stays absent until it can be taken on silicon.
"""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
CPP = REPO / "cpp"
TARGET = CPP / "target"

#: The reference flags from DESIGN.md, plus the one this file added. `-fno-math-errno` is
#: not a fast-math option and does not change any result: it drops the promise to set
#: `errno`, which is the only thing standing between `sqrtf` and one FPU instruction. The
#: values `norm()` takes a root of are sums of squares, so the promise was never meaningful.
TARGET_CFLAGS = [
    "-mcpu=cortex-m4", "-mfpu=fpv4-sp-d16", "-mfloat-abi=hard",
    "-ffreestanding", "-fno-math-errno", "-Os",
    "-ffunction-sections", "-fdata-sections",
]
LINKFLAGS = [
    "-mcpu=cortex-m4", "-mfpu=fpv4-sp-d16", "-mfloat-abi=hard",
    "-nostartfiles", "--specs=nano.specs", "--specs=nosys.specs",
    "-Wl,--gc-sections",
]

#: DESIGN.md's declared budget. The linker script carries the same numbers so that
#: overflowing is a link error rather than something to notice in a report.
FLASH_BUDGET = 32 * 1024
RAM_BUDGET = 8 * 1024

HEADERS = ("mat.hpp", "rls.hpp", "guard.hpp", "persist.hpp", "stream.hpp",
           "align.hpp", "pipeline.hpp")

#: Everything the core needs from outside, and the whole of it. Pinned as a set rather than
#: a count so that a *new* dependency fails this test and has to be argued for. `memcpy` and
#: `memset` are the freestanding-permitted ones; `sqrt` is the double root that only the f64
#: instantiations reach; the rest are libgcc's soft-float double helpers, which are the cost
#: `align.hpp` already documents for a single-precision FPU.
EXPECTED_EXTERNALS = {
    "memcpy", "memset", "sqrt",
    "__aeabi_d2f", "__aeabi_f2d", "__aeabi_dadd", "__aeabi_dsub", "__aeabi_dmul",
    "__aeabi_ddiv", "__aeabi_dcmpeq", "__aeabi_dcmpge", "__aeabi_dcmpgt",
    "__aeabi_dcmple", "__aeabi_dcmplt",
}


def _tool(name):
    return shutil.which(f"arm-none-eabi-{name}")


@pytest.fixture(scope="session")
def cross():
    """The toolchain, or a skip. Installed per-user from Arm's own zip on 2026-08-09;
    `winget`'s package is a machine-scope NSIS installer and needs elevation."""
    missing = [n for n in ("gcc", "g++", "size", "nm", "objdump") if _tool(n) is None]
    if missing:
        pytest.skip(f"no arm-none-eabi toolchain on PATH (missing {missing})")
    return {n: _tool(n) for n in ("gcc", "g++", "size", "nm", "objdump")}


def _run(cmd, **kw):
    return subprocess.run([str(c) for c in cmd], capture_output=True, text=True, **kw)


@pytest.fixture(scope="session")
def abi_object(cross, tmp_path_factory):
    out = tmp_path_factory.mktemp("arm") / "abi.o"
    done = _run([cross["g++"], "-std=c++17", *TARGET_CFLAGS, "-fno-exceptions",
                 "-fno-rtti", "-Wall", "-Wextra", "-I", CPP / "include",
                 "-c", CPP / "src" / "abi.cpp", "-o", out])
    assert done.returncode == 0, f"the ABI does not build for the target:\n{done.stderr}"
    return out


@pytest.mark.parametrize("header", HEADERS)
def test_every_core_header_compiles_for_the_target(cross, tmp_path, header):
    """The host freestanding gate checks that no hosted header sneaks in. This checks the
    thing it cannot: that the code is valid for ARMv7E-M with a single-precision FPU."""
    done = _run([cross["g++"], "-std=c++17", *TARGET_CFLAGS, "-fno-exceptions", "-fno-rtti",
                 "-Wall", "-Wextra", "-Wpedantic", "-I", CPP / "include",
                 "-c", "-x", "c++", CPP / "include" / "argmind" / header,
                 "-o", tmp_path / f"{header}.o"])
    assert done.returncode == 0, f"{header} does not build for the target:\n{done.stderr}"


def test_the_c_header_is_valid_c99_for_the_target(cross, tmp_path):
    done = _run([cross["gcc"], "-std=c99", "-mcpu=cortex-m4", "-mfpu=fpv4-sp-d16",
                 "-mfloat-abi=hard", "-Wall", "-Wextra", "-Wpedantic",
                 "-I", CPP / "include", "-fsyntax-only", "-x", "c",
                 CPP / "include" / "argmind" / "argmind.h"])
    assert done.returncode == 0, done.stderr


def test_the_core_needs_nothing_from_outside_but_these(cross, abi_object):
    """A freestanding library's external dependencies are part of its contract, so they are
    pinned as a set. A new one appearing means an integrator's environment has to grow, which
    is a decision rather than an accident."""
    done = _run([cross["nm"], "-u", abi_object])
    assert done.returncode == 0, done.stderr
    got = {ln.split()[-1] for ln in done.stdout.splitlines() if ln.strip()}
    assert got == EXPECTED_EXTERNALS, (
        f"the core's external dependencies changed.\n"
        f"  new:  {sorted(got - EXPECTED_EXTERNALS)}\n"
        f"  gone: {sorted(EXPECTED_EXTERNALS - got)}")


def test_the_library_defines_no_writable_static_memory(cross, abi_object):
    """The claim the whole storage-is-the-caller's design rests on, checked rather than
    asserted. Measured: 0 bytes of writable static memory and 72 of `.rodata`.

    The first version of this test read the sections named exactly `.data` and `.bss` and
    asserted they were zero. They are **always** zero under `-fdata-sections`, which puts
    every object in `.data.<name>` or `.bss.<name>` -- an audit compiled a probe declaring
    288 bytes of globals and this test passed. It now sums every section whose name starts
    with either prefix, and pins `.rodata` to the figure the documentation quotes rather
    than to a guessed ceiling.
    """
    done = _run([cross["size"], "-A", abi_object])
    assert done.returncode == 0, done.stderr
    sizes = {}
    for ln in done.stdout.splitlines():
        parts = ln.split()
        if len(parts) >= 2 and parts[0].startswith("."):
            try:
                sizes[parts[0]] = int(parts[1])
            except ValueError:
                pass

    def total(prefix):
        return sum(v for k, v in sizes.items() if k == prefix or k.startswith(prefix + "."))

    assert total(".data") == 0, {k: v for k, v in sizes.items() if k.startswith(".data")}
    assert total(".bss") == 0, {k: v for k, v in sizes.items() if k.startswith(".bss")}
    assert total(".rodata") == 72, (
        f"the library's read-only data is {total('.rodata')} bytes; DESIGN.md quotes 72")


def test_math_errno_is_what_stands_between_norm_and_the_fpu(cross, tmp_path):
    """The finding this file exists for, protected so it cannot quietly come back.

    Without `-fno-math-errno` the core calls `sqrtf`; with it, the compiler emits
    `VSQRT.F32` inline. That is not a host-versus-target difference and not a
    `-ffreestanding` one — it is the promise to set `errno`, which for a sum of squares can
    never be kept meaningfully anyway.
    """
    counts = {}
    for label, extra in (("with_errno", []), ("without_errno", ["-fno-math-errno"])):
        flags = [f for f in TARGET_CFLAGS if f != "-fno-math-errno"] + extra
        obj = tmp_path / f"{label}.o"
        done = _run([cross["g++"], "-std=c++17", *flags, "-fno-exceptions", "-fno-rtti",
                     "-I", CPP / "include", "-c", CPP / "src" / "abi.cpp", "-o", obj])
        assert done.returncode == 0, done.stderr
        dis = _run([cross["objdump"], "-d", obj]).stdout
        counts[label] = (len(re.findall(r"bl?\s+[0-9a-f]+ <sqrtf>", dis)),
                         dis.count("vsqrt.f32"))
    assert counts["with_errno"][0] > 0 and counts["with_errno"][1] == 0, counts
    assert counts["without_errno"][0] == 0 and counts["without_errno"][1] > 0, counts


@pytest.mark.parametrize("cfg,label", [(1, "bare RLS"), (2, "guarded"),
                                       (3, "guarded + records"), (4, "full pipeline")])
def test_a_firmware_shaped_caller_links_inside_the_declared_budget(
        cross, abi_object, tmp_path, cfg, label):
    """Four configurations rather than one, because `--gc-sections` means there is no single
    "size of the library": what reaches flash depends on which entry points a caller touches.
    Reporting one number would be reporting whichever configuration happened to be measured.

    The bound is the declared budget with room to spare, not the measured figure. Pinning the
    measurement would turn every compiler upgrade into a failing test; what this protects
    against is the library growing past what DESIGN.md promised.
    """
    obj = tmp_path / f"u{cfg}.o"
    done = _run([cross["gcc"], "-std=c99", *TARGET_CFLAGS, f"-DAM_CFG={cfg}",
                 "-Wall", "-Wextra", "-I", CPP / "include",
                 "-c", TARGET / "usage.c", "-o", obj])
    assert done.returncode == 0, f"{label} does not compile:\n{done.stderr}"

    elf = tmp_path / f"cfg{cfg}.elf"
    done = _run([cross["g++"], *LINKFLAGS, f"-Wl,-Map={tmp_path / f'cfg{cfg}.map'}",
                 "-T", TARGET / "cortex-m4.ld", obj, abi_object, "-o", elf])
    assert done.returncode == 0, f"{label} does not link:\n{done.stderr}"

    out = _run([cross["size"], elf]).stdout.splitlines()[1].split()
    flash, data, bss = int(out[0]), int(out[1]), int(out[2])
    assert flash + data <= FLASH_BUDGET, f"{label}: {flash + data} bytes of flash"
    assert bss + data <= RAM_BUDGET, f"{label}: {bss + data} bytes of RAM"
    # ...and it has to be doing something. A configuration that linked to nothing would sail
    # under any budget.
    assert flash > 1000, f"{label} linked to {flash} bytes; the calls were optimised away"


def test_an_f32_only_image_does_not_carry_the_double_root(cross, abi_object, tmp_path):
    """`sqrt` is in the dependency list because the f64 instantiations exist for the parity
    harness. A firmware that touches only f32 entry points must not pay for it, which is what
    `--gc-sections` is for -- verified rather than assumed."""
    obj = tmp_path / "u4.o"
    _run([cross["gcc"], "-std=c99", *TARGET_CFLAGS, "-DAM_CFG=4", "-I", CPP / "include",
          "-c", TARGET / "usage.c", "-o", obj])
    elf = tmp_path / "f32only.elf"
    done = _run([cross["g++"], *LINKFLAGS, "-T", TARGET / "cortex-m4.ld", obj, abi_object,
                 "-o", elf])
    assert done.returncode == 0, done.stderr
    syms = _run([cross["nm"], elf]).stdout
    assert not re.search(r"^\S+ T sqrt$", syms, re.M), (
        "an f32-only image linked the soft-float double root")
