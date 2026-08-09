# Target sizes — generated, do not edit by hand

Regenerated and checked by `tests/test_target_build.py::test_the_recorded_target_sizes_are_current`.
If that test fails it prints the new table; re-run it with `--regen-sizes` and commit the diff.

**This file exists because the same table went stale three times.** It lived in DESIGN.md and
CLAUDE.md as hand-copied prose, was bounded against the flash/RAM *budget* rather than pinned —
deliberately, so a compiler upgrade would not be a failing test — and the cost of that choice
turned out to be higher than the benefit: it drifted at `47a4dd2`, again at `17cff2b`, and a
third time at `0ada177`, which was **the commit documenting the first two**. Two independent
audits found the third. Pinning here makes a size change a failing test that says exactly what
to do, which is a changelog gate rather than a brittle assertion, and the prose now points here
instead of restating figures that nobody re-runs.

`--gc-sections` means there is no single "size of the library": what reaches flash depends on
which entry points a caller touches. So `cpp/target/usage.c` is linked in four
firmware-shaped configurations against `cpp/target/cortex-m4.ld`.

Toolchain: Arm GNU Toolchain 14.2.Rel1, `-Os`, newlib-nano, flags as in
`tests/test_target_build.py`. Columns are text+rodata attributed by the link map, and they sum
to the total — a `caller` column exists because without it the drift was invisible.

## Cortex-M4F — the declared floor

| configuration | flash | of which argmind | the caller | libc/libgcc | static RAM |
|---|---|---|---|---|---|
| bare RLS f32 6x1 | 2308 | 1256 | 160 | 892 | 272 |
| + guard | 6168 | 3552 | 344 | 2272 | 1296 |
| + record format | 6404 | 3830 | 304 | 2270 | 1972 |
| full composed pipeline | 10452 | 7440 | 564 | 2448 | 3264 |

Against the declared 32 KB flash and 8 KB RAM, the full composed pipeline is **31.9%** of the
flash budget and **39.8%** of the RAM.

The +0/+4/+8 against the first recorded version is the guard's caller-supplied health
scalars (`Guard`'s `NExternal`), and **this file's gate is what reported it** — the first
time it fired in anger, on the change immediately after it was written.

## Cortex-M55 — the second tier's core

| configuration | flash | static RAM |
|---|---|---|
| full composed pipeline | 7924 | 3264 |

Moving up to ARMv8.1-M makes the image **smaller**, at identical static RAM.

## Stack, `-fstack-usage`, Cortex-M4F

Storage is the caller's and so is the stack; `am_*_bytes()` reports only the object.

| entry point | bytes |
|---|---|
| `am_pipe_*_step` | 312 to 328 |
| `am_guard_*_update` | 88 to 104 |
| `am_rls_*_update` | 72 to 88 |
| `am_rls_*_init` | 24 to 32 |
| `am_guard_*_init` | 32 to 40 |
| `am_pipe_*_init` (first tier) | 136 to 168 |
| `am_pipe_g_f32_128_1_init` (head shape) | 504 |
