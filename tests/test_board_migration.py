# SPDX-License-Identifier: Apache-2.0
"""Swapping one board for a higher-specified one: does anything change?

Every parity claim in this project compares an implementation against the Python
reference. That answers "is the port right". It does not answer the question an
integrator actually asks before a hardware change: **if I move this firmware from a
Cortex-M4F to a Cortex-M55, do I get the same numbers?**

Nothing measured that until 2026-08-10, and the pieces that would have suggested an
answer point in different directions. `rls.hpp` declares a divergence that is
*target-only* -- VFPv4 contracts `a*b + c` into `VFMA.F32` and the x86-64 SSE2 baseline
does not -- so two ARM cores might be expected to agree with each other more closely
than either does with the host. But the M55 also has **hardware double precision**,
which the M4F does not, so the f64 path is soft-float on one and not the other.

Measured here rather than reasoned about.

Getting it to run needed a harness fix that is recorded in `cpp/target/qemu/startup.c`:
`malloc` returns NULL on QEMU's `mps3-an547`, so `fopen` fails and a parity tool reports
"cannot open <file>" -- an error that names the file rather than the heap.
"""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from test_cpp_parity import (_read_result, _write_case, build_reference, excited,
                             scale_deviation)

REPO = Path(__file__).resolve().parent.parent
CPP = REPO / "cpp"
QEMU_DIR = CPP / "target" / "qemu"

#: (label, compile flags, QEMU machine flags). Both are above the declared floor: the
#: M4F *is* the floor, and the M55 is ARMv8.1-M, which is strictly above it.
BOARDS = {
    "cortex-m4": (["-mcpu=cortex-m4", "-mfpu=fpv4-sp-d16", "-mfloat-abi=hard"],
                  ["-M", "mps2-an386", "-cpu", "cortex-m4"]),
    "cortex-m55": (["-mcpu=cortex-m55", "-mfloat-abi=hard"],
                   ["-M", "mps3-an547", "-cpu", "cortex-m55"]),
}
REST = ["-ffreestanding", "-fno-math-errno", "-O2",
        "-ffunction-sections", "-fdata-sections"]

#: Measured worst over the sweep below: 3.5 ulp at f32 and 3.3 at f64. Bounded at 50,
#: which is headroom rather than a number chosen to pass -- and the point of the bound is
#: that a board swap must stay in the same band as the port itself, not that it be exact.
MIGRATION_ULP = 50.0


def _cross(name):
    return shutil.which(f"arm-none-eabi-{name}")


def _run(cmd, **kw):
    return subprocess.run([str(c) for c in cmd], capture_output=True, text=True, **kw)


@pytest.fixture(scope="session")
def images(tmp_path_factory):
    """`xcheck` linked for each board."""
    if _cross("g++") is None or shutil.which("qemu-system-arm") is None:
        pytest.skip("need arm-none-eabi and qemu-system-arm for the migration check")
    work = tmp_path_factory.mktemp("migrate")
    out = {}
    for board, (cflags, machine) in BOARDS.items():
        listing = _run([shutil.which("qemu-system-arm"), "-machine", "help"]).stdout
        if not any(l.split()[:1] == [machine[1]] for l in listing.splitlines() if l.strip()):
            pytest.skip(f"this qemu-system-arm has no {machine[1]} machine")
        tag = board.replace("-", "")
        flags = [*cflags, *REST, "-I", str(CPP / "include")]
        su = work / f"startup_{tag}.o"
        d = _run([_cross("gcc"), "-std=c99", *flags, "-c", QEMU_DIR / "startup.c",
                  "-o", su])
        assert d.returncode == 0, d.stderr
        obj = work / f"xcheck_{tag}.o"
        d = _run([_cross("g++"), "-std=c++17", *flags, "-fno-exceptions", "-fno-rtti",
                  "-c", CPP / "tools" / "xcheck.cpp", "-o", obj])
        assert d.returncode == 0, d.stderr
        elf = work / f"xcheck_{tag}.elf"
        d = _run([_cross("g++"), *cflags, "--specs=rdimon.specs", "--specs=nano.specs",
                  "-Wl,--gc-sections", "-T", QEMU_DIR / "mps2.ld", su, obj, "-o", elf])
        assert d.returncode == 0, "\n".join(
            l for l in d.stderr.splitlines() if "RWX permissions" not in l)
        out[board] = (elf, machine)
    return work, out


def _state_from(work, board, elf, machine, xs, ys, lam, p0, dtype, size):
    """Run one case on one board and return the state block it ends with.

    Relative filenames with `cwd` set, because QEMU's semihosting command line truncates
    past roughly 255 characters and pytest's temporary paths are long enough to hit it --
    the tool then reports a usage error, which reads as "the tool failed".
    """
    job, res = work / "job.bin", work / f"res_{board}.bin"
    _write_case(job, xs, ys, lam, p0, dtype)
    sem = f"enable=on,target=native,arg=xcheck,arg={job.name},arg={res.name}"
    d = _run([shutil.which("qemu-system-arm"), *machine, "-nographic",
              "-semihosting-config", sem, "-kernel", elf.name], cwd=work, timeout=300)
    assert res.exists(), f"{board} produced no result:\n{d.stdout[-800:]}{d.stderr[-800:]}"
    return _read_result(res, size)[1]


@pytest.mark.parametrize("dtype", ["f4", "f8"])
@pytest.mark.parametrize("n,m", [(2, 1), (6, 1), (12, 1), (6, 3)])
def test_two_boards_agree_about_as_closely_as_either_agrees_with_the_reference(
        images, dtype, n, m):
    """The migration claim: a board swap does not move the numbers further than the port
    already does.

    Bit identity is NOT asserted and would be the wrong bar -- `rls.hpp` declares three
    divergences from the reference and a fourth that is target-only, so two different
    cores rounding differently is expected. What matters is that the swap stays inside
    the same band, because a caller's tolerances are set against the port's figures.
    """
    work, built = images
    size = n * m + n * n
    xs, ys = excited(n, 400, seed=17, outputs=m)
    xs = np.ascontiguousarray(xs, dtype=dtype)
    ys = np.ascontiguousarray(ys, dtype=dtype)

    states = {b: _state_from(work, b, elf, machine, xs, ys, 0.99, 100.0, dtype, size)
              for b, (elf, machine) in built.items()}

    ref = build_reference(xs, ys, 0.99, 100.0, dtype)[0]
    for x, y in zip(xs, ys):
        ref.update(x, y)
    rs = np.asarray(ref.state)

    eps = float(np.finfo(np.dtype(dtype)).eps)
    a, b = states["cortex-m4"], states["cortex-m55"]
    swap = scale_deviation(a, b) / eps
    port = max(scale_deviation(rs, a), scale_deviation(rs, b)) / eps

    assert swap < MIGRATION_ULP, (
        f"{n}x{m} {dtype}: swapping boards moves the state by {swap:.1f} ulp")
    # ...and it has to be comparable to the port's own deviation, not merely small. A
    # swap ten times worse than the port would still pass a bare 50-ulp bound.
    assert swap <= max(port * 4.0, 4.0), (
        f"{n}x{m} {dtype}: the board swap costs {swap:.1f} ulp against the port's own "
        f"{port:.1f} -- a hardware change should not dominate the port")


def test_the_state_block_one_board_writes_is_the_one_the_other_reads(images):
    """The other half of migration, and the half that decides whether a fielded device
    survives the change: the persisted state.

    `persist`'s record is byte-identical across architectures -- already verified host
    against Cortex-M4 -- and carries a configuration fingerprint that REFUSES a block
    whose shape or `FrameMap` order differs rather than misreading it. This checks the
    state block itself: same length, same layout, and finite on both, so a block copied
    from the old board loads on the new one.
    """
    work, built = images
    n, m, dtype = 6, 1, "f8"
    size = n * m + n * n
    xs, ys = excited(n, 200, seed=5, outputs=m)
    xs = np.ascontiguousarray(xs, dtype=dtype)
    ys = np.ascontiguousarray(ys, dtype=dtype)
    states = {b: _state_from(work, b, elf, machine, xs, ys, 0.99, 100.0, dtype, size)
              for b, (elf, machine) in built.items()}
    for board, st in states.items():
        assert st.size == size, f"{board}: state block is {st.size}, expected {size}"
        assert np.all(np.isfinite(st)), f"{board}: state block is not finite"
    # theta first, then P row-major: the leading block is the model and must match to the
    # same band as the whole.
    a, b = states["cortex-m4"], states["cortex-m55"]
    assert scale_deviation(a[:n * m], b[:n * m]) < MIGRATION_ULP * float(
        np.finfo(np.dtype(dtype)).eps)
