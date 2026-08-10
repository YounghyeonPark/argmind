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

HEADERS = ("mat.hpp", "rls.hpp", "nlms.hpp", "guard.hpp", "persist.hpp",
           "stream.hpp", "align.hpp", "pipeline.hpp")

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


#: Measured with -fstack-usage for Cortex-M4 at -Os, and bounded rather than pinned exactly so
#: a compiler upgrade does not fail the suite. The hot path is what runs forever; `init` is a
#: transient spike that still has to fit inside the declared RAM.
MAX_STEP_STACK = 512
#: Was 5 KB when `init` copied an adapter into place. The measured worst over ALL `_init`
#: frames -- which is what this bounds -- is now 504 B at the 128-feature head shape.
#: An audit caught the first version of this note quoting 504 while the true worst was
#: 664, because `am_rls_*_init` had been left out of the in-place fix.
MAX_INIT_STACK = 1024


def _declared_object_bounds():
    """The composed-pipeline storage bounds, read from the header an integrator reads.

    These are upper bounds held to being ones by a `static_assert`, which is the right side
    to err on here: the sum they feed is what someone sizing storage from the header would
    compute, not what a particular compiler happened to lay out.
    """
    text = (CPP / "include" / "argmind" / "argmind.h").read_text(encoding="utf-8")
    found = dict(re.findall(r"#define AM_PIPE_G_(F32_\d+_\d+)_MAX_BYTES\s+(\d+)", text))
    assert found, "no composed-pipeline storage bounds in the header"
    return {k.lower(): int(v) for k, v in found.items()}


def test_the_stack_a_caller_must_provide_is_bounded(cross, tmp_path):
    """`am_*_bytes()` reports the object; these functions also use the caller's stack, and
    nobody had measured how much until a hardware question asked for it.

    What it found: `init` built the whole object as a temporary and copied it, so
    `am_pipe_g_f32_12_1_init` wanted 8,296 bytes -- more than the entire RAM budget this
    library declares as its floor. On the target it names, that entry point could not be
    called. Constructing in the caller's bytes brought it to 4,288.

    Bounded rather than pinned to the figure, because the exact number is a compiler's
    business and the budget is not.
    """
    done = _run([cross["g++"], "-std=c++17", *TARGET_CFLAGS, "-fno-exceptions", "-fno-rtti",
                 "-fstack-usage", "-I", CPP / "include", "-c", CPP / "src" / "abi.cpp",
                 "-o", tmp_path / "abi_su.o"], cwd=tmp_path)
    assert done.returncode == 0, done.stderr
    su = tmp_path / "abi_su.su"
    assert su.exists(), f"no stack-usage output; looked for {su}"

    frames = {}
    for line in su.read_text(encoding="utf-8").splitlines():
        parts = line.split("	")
        if len(parts) < 2:
            continue
        name, size = parts[0], int(parts[1])
        if "am_" not in name:
            continue
        sym = name.split(":")[-1].split("(")[0].split()[-1]
        frames[sym] = max(frames.get(sym, 0), size)

    steps = {k: v for k, v in frames.items()
             if k.endswith(("_step", "_update", "_push"))}
    inits = {k: v for k, v in frames.items() if k.endswith("_init")}
    assert steps and inits, sorted(frames)

    worst_step = max(steps.items(), key=lambda kv: kv[1])
    worst_init = max(inits.items(), key=lambda kv: kv[1])
    assert worst_step[1] <= MAX_STEP_STACK, (
        f"the hot path now wants {worst_step[1]} bytes of stack ({worst_step[0]})")
    assert worst_init[1] <= MAX_INIT_STACK, (
        f"{worst_init[0]} wants {worst_init[1]} bytes of stack; the declared RAM floor is "
        f"{RAM_BUDGET}, so this has to stay well inside it")
    # ...and the measurement has to have found something. An empty parse would pass both.
    assert worst_step[1] > 32, f"suspiciously small: {worst_step}"

    # The two halves have to be added up, and until 2026-08-09 nothing did. `MAX_INIT_STACK`
    # bounds the frame alone; an integrator's actual question is whether the object AND the
    # frame that constructs it fit together, because both live in the same RAM. That sum is
    # what the header lets someone compute: `AM_*_MAX_BYTES` plus the `init` frame.
    bounds = _declared_object_bounds()
    fits = {}
    for shape, obj_bytes in bounds.items():
        frame = frames.get(f"am_pipe_g_{shape}_init")
        assert frame, f"no init frame measured for {shape}; have {sorted(inits)}"
        fits[shape] = obj_bytes + frame

    # **Every first-tier shape fits, and it did not always.** This assertion first read "the
    # 8 KB floor is a six-feature floor", because `f32_12_1` came to 4,096 + 4,288 = 8,384
    # against 8,192 and the twelve-feature pipeline could not be constructed on the declared
    # minimum. That was the by-value adapter, not the shape: `Pipeline` and `Guard` each
    # copied an adapter the caller had already built, so `init` needed a whole one of each
    # alive on the stack. Constructing in place instead took `f32_12_1` from 4,288 bytes of
    # frame to 168 and the sum to 4,264, and the claim was withdrawn rather than the test
    # loosened -- which is what its own failure message asked for.
    for shape in ("f32_6_1", "f32_6_3", "f32_12_1"):
        assert fits[shape] <= RAM_BUDGET, (
            f"{shape} no longer fits the declared floor: {fits[shape]} > {RAM_BUDGET}")

    # The tier boundary. **This one cannot fail** -- a 128x128 fp32 covariance is 330 KB
    # whatever any code does -- so it documents rather than protects, and an audit was
    # right to say calling it "the negative result pinned" overstated it. The assertion
    # below it on the frame is the one with teeth.
    assert fits["f32_128_1"] > RAM_BUDGET, (
        f"f32_128_1 now fits the {RAM_BUDGET}-byte floor at {fits['f32_128_1']} bytes; "
        "update the two-tier claim rather than this test.")

    # ...and the reason has to be the object rather than the frame, or the sentence above is
    # wrong even though the assertion passes.
    assert bounds["f32_128_1"] > RAM_BUDGET
    assert frames["am_pipe_g_f32_128_1_init"] < 1024, (
        f"the wide shape's init frame grew back to {frames['am_pipe_g_f32_128_1_init']}; "
        "it was 397,192 before the adapter stopped being copied")


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


# -- the recorded sizes, pinned because bounding them was not enough ------------------

SIZES_MD = TARGET / "SIZES.md"

#: Configurations linked for the recorded table, as (label, AM_CFG, core flags).
SIZE_CONFIGS = [
    ("bare RLS f32 6x1", 1, None),
    ("+ guard", 2, None),
    ("+ record format", 3, None),
    ("full composed pipeline", 4, None),
]


def _link_for_sizes(cross, tmp_path, cfg, core, tag):
    """Link one configuration and return (flash, argmind, caller, libc, ram)."""
    rest = [f for f in TARGET_CFLAGS if not f.startswith(("-mcpu", "-mfpu", "-mfloat-abi"))]
    flags = [*core, *rest]
    abi = tmp_path / f"abi_{tag}.o"
    if not abi.exists():
        done = _run([cross["g++"], "-std=c++17", *flags, "-fno-exceptions", "-fno-rtti",
                     "-I", CPP / "include", "-c", CPP / "src" / "abi.cpp", "-o", abi])
        assert done.returncode == 0, done.stderr
    obj = tmp_path / f"u_{tag}_{cfg}.o"
    done = _run([cross["gcc"], "-std=c99", *flags, f"-DAM_CFG={cfg}",
                 "-I", CPP / "include", "-c", TARGET / "usage.c", "-o", obj])
    assert done.returncode == 0, done.stderr
    elf = tmp_path / f"e_{tag}_{cfg}.elf"
    mp = tmp_path / f"m_{tag}_{cfg}.map"
    done = _run([cross["g++"], *core, "-nostartfiles", "--specs=nano.specs",
                 "--specs=nosys.specs", "-Wl,--gc-sections", f"-Wl,-Map={mp}",
                 "-T", TARGET / "cortex-m4.ld", obj, abi, "-o", elf])
    assert done.returncode == 0, done.stderr
    text, data, bss = (int(x) for x in
                       _run([cross["size"], elf]).stdout.splitlines()[1].split()[:3])
    # Only what follows this line: above it is the DISCARDED input-section list, and
    # summing the whole map reads 41,072 bytes for a 2,340-byte image.
    whole = mp.read_text(errors="replace")
    body = whole[whole.find("Linker script and memory map"):]
    # The caller's code lives in `.am_entry`, not `.text`; a section list naming only the
    # usual four attributes 24 bytes to a caller that contributes 160.
    pat = re.compile(
        r"^\s\.(text|rodata|am_entry)[.\S]*\s*(?:\n)?\s*0x[0-9a-f]+\s+0x([0-9a-f]+)\s+(\S+)$",
        re.M)
    am = caller = 0
    for m in pat.finditer(body):
        n, owner = int(m.group(2), 16), m.group(3)
        if owner.endswith(f"abi_{tag}.o"):
            am += n
        elif re.search(r"u_\w+_\d+\.o$", owner):
            caller += n
    flash = text + data
    return flash, am, caller, flash - am - caller, data + bss


def _measure_sizes(cross, tmp_path):
    m4 = ["-mcpu=cortex-m4", "-mfpu=fpv4-sp-d16", "-mfloat-abi=hard"]
    m55 = ["-mcpu=cortex-m55", "-mfloat-abi=hard"]
    rows = {label: _link_for_sizes(cross, tmp_path, cfg, m4, "m4")
            for label, cfg, _ in SIZE_CONFIGS}
    rows["cortex-m55 composed"] = _link_for_sizes(cross, tmp_path, 4, m55, "m55")
    return rows


def _recorded_sizes():
    """Parse the tables out of SIZES.md, so the file is the single source."""
    out = {}
    for line in SIZES_MD.read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 6 and cells[1].isdigit():
            out[cells[0]] = tuple(int(c) for c in cells[1:])
        elif len(cells) == 3 and cells[1].isdigit():
            out["cortex-m55 composed"] = (int(cells[1]), int(cells[2]))
    return out


def test_the_recorded_target_sizes_are_current(cross, tmp_path):
    """`cpp/target/SIZES.md` must match a fresh link.

    The four size figures used to live in DESIGN.md and CLAUDE.md as hand-copied prose,
    bounded against the flash/RAM *budget* rather than pinned -- deliberately, so that a
    compiler upgrade would not be a failing test. **Three staleness events say the cost of
    that choice was higher than the benefit**, the third being the commit that documented
    the first two. So the numbers live in one generated file now and this pins them.

    A compiler upgrade will fail this. That is the intended behaviour and the failure says
    exactly what to do: the new table is printed, so re-measure and commit the diff. It is
    a changelog gate, not a brittle assertion.
    """
    measured = _measure_sizes(cross, tmp_path)
    recorded = _recorded_sizes()
    assert recorded, f"no tables parsed out of {SIZES_MD}"

    bad = []
    for label, got in measured.items():
        want = recorded.get(label)
        if want is None:
            bad.append(f"  {label}: not in SIZES.md; measured {got}")
        elif label == "cortex-m55 composed":
            if (got[0], got[4]) != want:
                bad.append(f"  {label}: recorded {want}, measured "
                           f"{(got[0], got[4])}")
        elif got != want:
            bad.append(f"  {label}: recorded {want}, measured {got}")
    assert not bad, (
        "cpp/target/SIZES.md no longer matches a fresh link:\n" + "\n".join(bad) +
        "\n(columns: flash, of which argmind, the caller, libc/libgcc, static RAM)")

    # ...and the rows have to sum, which is the property that would have made all three
    # earlier drifts visible in the file itself.
    for label, (flash, am, caller, lib, _ram) in measured.items():
        if label == "cortex-m55 composed":
            continue
        assert am + caller + lib == flash, f"{label}: columns do not sum to {flash}"


#: Headers that are deliberately NOT freestanding. The hosted tier uses the heap and
#: `std::vector`; the path is the boundary and this is where it is checked.
HOSTED_HEADERS = ("host/dyn.hpp",)


def test_no_core_header_reaches_into_the_hosted_tier(cross):
    """The boundary is a directory name, and a directory name is a convention until
    something checks it.

    A core header including `argmind/host/...` would drag the heap into a freestanding
    build, and the failure would surface as a confusing link error in somebody's firmware
    rather than here. `cross` is required so this runs in the same place as the rest of
    the target gate.
    """
    for name in HEADERS:
        text = (CPP / "include" / "argmind" / name).read_text(encoding="utf-8")
        assert "argmind/host/" not in text, (
            f"{name} includes a hosted header; the freestanding core cannot")


@pytest.mark.parametrize("header", HOSTED_HEADERS)
def test_a_hosted_header_is_hosted_and_says_so_by_failing_freestanding(cross, tmp_path,
                                                                       header):
    """The other half, and the one that would otherwise be assumed.

    If `host/dyn.hpp` happened to compile freestanding, the separation would be
    decoration. It must not: `<vector>` is a hosted header. This asserts the split is
    real rather than aspirational -- and if it ever starts passing, that is a signal the
    header stopped needing the heap, which is a decision to make rather than a change to
    absorb.
    """
    src = CPP / "include" / "argmind" / header

    # It has to build in a HOSTED build first. Without this the assertion below is
    # satisfied by a syntax error, and a broken header would read as a healthy boundary.
    # A mutation attempt found exactly that: deleting `<vector>` "passed".
    host_cxx = shutil.which("g++")
    if host_cxx is None:
        pytest.skip("no host g++")
    done = _run([host_cxx, "-std=c++17", "-Wall", "-Wextra", "-I", CPP / "include",
                 "-c", "-x", "c++", src, "-o", tmp_path / "hosted.o"])
    assert done.returncode == 0, (
        f"{header} does not build hosted either, so the assertion below would be "
        f"satisfied by a syntax error:\n{done.stderr}")

    done = _run([cross["g++"], "-std=c++17", *TARGET_CFLAGS, "-fno-exceptions",
                 "-fno-rtti", "-I", CPP / "include", "-c", "-x", "c++",
                 src, "-o", tmp_path / "h.o"])
    assert done.returncode != 0, (
        f"{header} compiles for the freestanding target, so the hosted/core split is "
        "decoration; either it no longer needs the heap or it is in the wrong directory")
    # ...and for the right reason. Any compile error satisfies the line above.
    assert "vector" in done.stderr or "memory" in done.stderr, (
        f"{header} failed the freestanding build for some reason other than a hosted "
        f"header, so this proves nothing about the boundary:\n{done.stderr[:600]}")
