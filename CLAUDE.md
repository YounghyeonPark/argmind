# Argmind — project context

**Multi-sensor AI software: fusion, on-device learning, and inference.**
Started 2026-08-08. This is the active project.

Author: Younghyeon Park <ypark.dev@gmail.com>. Apache-2.0 (`LICENSE`, `NOTICE`).
Side project — open source, not a product, no support commitment.

## Working scope

The whole repository. There is one project here.

**A previous, unrelated project — a hardware accelerator IP study — was stopped and removed from
the working tree on 2026-08-08.** It is not deleted: it lives in history under the tag
`accelerator-final`, which is where an archive belongs. It is out of the tree because carrying
forty files and ten documents of a different project taxes every search and every exploration
from here on, and makes the repository look like two projects to anyone reading it.

    git show accelerator-final:accelerator/README.md      # browse
    git checkout accelerator-final -- accelerator/        # restore, if ever needed

**Do not restore it to continue it.** Everything worth carrying forward was extracted into this
file before removal.

Two things from it are still worth knowing, because they were measured rather than assumed and
they bear on sensor/estimation work:

- **The EKF covariance recursion needs fp32 minimum. There is no fp16/bf16 path in any filter
  form** — conventional, Joseph and square-root all lose positive-definiteness on the *first*
  step, and fp16 additionally overflows at κ(P)=1e6. Measured numerically.
- **Precision floors are a property of the algorithm, not of the platform.** Three distinct
  floors appeared across one workload set. Do not assume a single precision policy works
  everywhere.

Measured *here*, on 2026-08-08, for RLS — deliberately re-derived rather than inherited from the
above, because carrying a conclusion across algorithms is the borrowed-assumption error:

- **fp16 fails the RLS covariance recursion even at κ=1**, losing positive-definiteness by step
  115 with every regressor inside fp16's range, so it is arithmetic and not dynamic range. It is
  refused at construction.
- **fp32 survives but does not hold positive-definiteness exactly** — a negative eigenvalue at
  roundoff scale (≈-5e-7 against tr(P)~1e2) on the first update when `p0·‖x‖² >> 1`. The fix if
  it ever bites is a square-root/UD form, not more precision. Both halves have tests.

Full detail lives in the tag, not in this tree: `git show accelerator-final:accelerator/PHASE0_FINDINGS.md`.

## Standards of evidence — carried over deliberately

These came from the previous project, where seventeen modelling errors were found and most of
them flattered the design. They cost real time to learn and they apply to anything measured.

- **Distinguish measured from estimated, every time.** Say which is which in the same sentence
  as the number.
- **A model must be able to fail.** If an analysis cannot produce a negative result, it is not
  evidence. Where a result is negative, protect it with a test so a later change cannot quietly
  optimise it away.
- **A flattering result means find the responsible assumption first.** The most common shapes:
  a model that cannot express a cost, so an axis looks free; a comment claiming a term the code
  does not contain; a swept axis that moves only one side of a comparison.
- **A uniformly tidy result is evidence the measurement is broken**, not that the world is
  simple. This recurred often enough to be a rule.
- **Scrutinise corrections that favour you as hard as ones that do not.**
- Accuracy figures carry the dataset, the split, and the conditions. "95%" without those is not
  a number.
- Cite URLs for external claims. "Not published" is a valid finding; never present an estimate
  as someone else's specification.

## Conventions

**Location: `C:\dev\argmind`.** Moved off Google Drive on 2026-08-08 — Drive sync can corrupt a
git repository by uploading partial writes or creating conflict copies of `.git/index` and loose
objects. Backup belongs on a remote, not on a syncing filesystem.

Windows, PowerShell primary; Python 3.14; git repo. `python -m pytest` before quoting any
number. A tracked pre-commit hook in `.githooks/` runs the suite and blocks on failure —
install once per clone with `git config core.hooksPath .githooks`.

## Status — settled 2026-08-08

**Requirements are settled. Building has started.** The four scope decisions, made explicitly
and not to be re-litigated without a reason:

| axis | decision |
|---|---|
| platform | **embedded / edge** |
| learning | **on-device online adaptation** — not offline training |
| sensors | **sensor-agnostic**, general framework |
| deliverable | **a library**, not an application |
| language | **C++ core behind a C ABI**, optimised for speed (see DESIGN.md) |
| target floor | **Cortex-M4F class** — ARMv7E-M with an fp32 FPU; own matrix types, no Eigen |

The design driver, chosen so the API is shaped against something real: an inertial + aiding-sensor
estimator with online bias calibration. Sensor-agnostic in the API, specific in what it must
handle.

**Built:** `argmind/stream.py` (fixed-capacity ring buffers, counted drops, monotonic time
enforced at the source), `argmind/align.py` (four explicit alignment policies, signed staleness,
causal/non-causal declared), `argmind/adapt.py` (the `Adapter` contract and
`RecursiveLeastSquares`), `argmind/guard.py` (`Check`/`Bound`/`Divergence`, freeze, bounded
rollback, explicit recovery), `argmind/pipeline.py` (`FrameMap`, input gate, bounded `step()`),
`argmind/persist.py` (two-slot atomic store, CRC, configuration fingerprint, declared cadence),
`argmind/bench.py` (fixed workload, correctness digest, leak detector, regression gate), and
`cpp/` (the C++ core: `mat.hpp`, `rls.hpp`, `guard.hpp`, the C ABI, the parity tool, 631
self-checks).
**The Python reference is complete and RLS is across to C++.** 185 Python tests pass; several
exist to prove failures are visible rather than silent, and every suite was mutation-checked —
seventy-five deliberate bugs, all caught.

**The adapter contract, since everything after it fits into it** (full reasoning in DESIGN.md):
state is one flat contiguous block so persistence is a copy and the C ABI needs no serialisation;
`update()` always returns an `UpdateReport` because the `Guard` is written against those fields;
shape errors raise but bad *values* are rejected-and-counted, since a sensor NaN is a runtime
event and not a programming error; and covariance windup is reported rather than clamped, because
bounding it inside the adapter would make the guard optional.

RLS was verified against the closed-form exponentially weighted least squares solution, not
against itself. Two apparent failures turned out to be real properties worth pinning: the residual
error on noiseless data is the `P0⁻¹` prior term and scales as exactly `1/p0` across four decades,
and the unexcited covariance direction grows as exactly `p0·λ⁻ᵏ` (12 digits). The tests assert
those laws rather than loose tolerances.

**The guard wraps the adapter and *is* an `Adapter`** — a deliberate change from the API sketch,
which had them side by side in the pipeline. Side by side leaves "no guard" expressible, and a
safety mechanism that can be left out gets left out. Checks run after the update (the quantities
worth checking do not exist until then), which is why rollback exists rather than prevention.

Two design errors were made here and caught by tests, both about what a stateful detector does
while frozen: resetting checks on a trip made the guard **thaw itself in the middle of a fault**,
and the first fix — holding the baseline per sample — ratcheted it down until ordinary noise
tripped it. The rule that works is per-*freeze*. The tidy-looking version was the broken one,
twice.

**The pipeline maps a `Frame` to `(x, y)` through a declared index table, not a callable** — a
callable is the obvious Python answer and does not cross a C ABI. `FrameMap` resolves convenient
forms once at construction. Feature order is parameter order, so a persisted state block restored
against a different order is wrong undetectably.

**Two gates at two different times**, and they must not merge: the pipeline checks *input* health
(staleness, sensor validity) before the update; the guard checks *model* health after it, because
those quantities do not exist until the update is computed.

A test found a real observability gap here: 200 frames against a 64-deep ring adapted 64 times and
reported nothing wrong. The first fix was wrong the other way — `Stream.dropped` counts
*overwrites*, which a healthy ring does on every push, so it false-alarms on every working system.
`StepReport.lost_frames` counts what is actually knowable: reference samples overwritten before
they could become frames. `Stream.dropped`'s docstring was overclaiming and is corrected.

**Persistence is two slots written alternately**, each with a sequence number and CRC, so a write
interrupted by power loss destroys at most the slot not in use. The reference writes files but
models the raw-flash scheme faithfully, because the C side has no `os.replace`. A configuration
fingerprint (`crc32`, never Python's salted `hash()`) refuses a block whose `FrameMap` order
differs — the one corruption nothing downstream can detect. Cadence is declared, because a sector
rated for 100k erase cycles written at 100 Hz is gone in seventeen minutes.

**The guard-state question is answered**: checkpoints and check EWMAs describe a run and are not
persisted; the frozen flag and its cause are, because they are a conclusion. What a reboot does
with a freeze is the caller's call (`on_frozen="thaw"|"keep"|"refuse"`) — thawing risks re-learning
into the same trouble, keeping risks stranding a device a power cycle would have fixed.

**The benchmark refuses to commit absolute timings as a gate.** A baseline is a statement about
one machine; as a repository gate it fails on everyone else's laptop and gets switched off. Record
locally (`python -m argmind.bench --record`), compare with `--compare`, which exits non-zero. The
suite gates only what survives moving machines: exact counters, the state digest, the memory leak
rate, and ratios measured inside one run. The tolerance is widened by *measured* run-to-run
spread, not a guessed constant.

**Two measured results from building it**, both of which contradicted a plausible first guess:

- **RLS update cost is flat from n=4 to n=32** (~22 µs) and only climbs at n=64. RLS is O(n²), so
  a naive scaling test asserting ×64 would have failed. At the design driver's n=6 the time is
  *all* call overhead — the regime the C++ decision was made for, now measured here rather than
  inherited.
- **Memory does not grow with uptime**, but proving it needed two corrections. Fitting a trend
  line to successive batches reports a settling transient as growth; the right tool is a
  discriminator across batch *sizes*, fitting `cost = fixed + leak × frames`. And `tracemalloc`
  allocates while measuring, so its own bookkeeping swamped the residual — `sys.getallocatedblocks`
  does not. The measurement found one real thing: `Adapter.state` built a fresh view per call, on
  a path the guard takes every checkpoint. Fixed. A positive control (an adapter that hoards every
  report) keeps the detector honest.

**The C++ core — started 2026-08-08, RLS across.** `cpp/include/argmind/{mat,rls}.hpp`,
`argmind.h` + `cpp/src/abi.cpp`, `cpp/tools/xcheck.cpp`, `cpp/tests/test_core.cpp`, `CMakeLists.txt`.
Build and run it through `python -m pytest` — `tests/test_cpp_parity.py` invokes the compiler
directly and skips if there is none, so a clone without a toolchain still tests the Python.

**The C ABI's real tension**: compile-time dimensions are why C++ was chosen, and a C ABI wants
runtime arguments. Runtime `n` would discard the reason, so the ABI exports an **explicit
instantiation list** — adding a dimension is one line plus a rebuild. Storage is the caller's;
`AM_*_MAX_BYTES` is an upper bound held to being one by a `static_assert`.

**Parity: ~1.5 ulp on the state block, and the same ulp count at f32 as at f64** — which is what
says the difference is algorithmic rounding, not a precision-dependent bug. Worst quantity is the a
posteriori residual at 597 ulp, and that is cancellation, not error: it is `y - x'θ` after fitting
to that sample. Three divergences are deliberate (two reciprocal multiplies from the port notes,
plus BLAS reduction order).

**The parity metric took two corrections**, both non-obvious and both recorded in
`tests/test_cpp_parity.py`: element-wise relative error is meaningless on a block with
cancellation-produced near-zero entries (reads 1e-13/1e-4), and the a posteriori residual does not
belong in a *relative* comparison at all — nor does its ratio to the a priori one, which inherits
the same cancellation (1.06e5 ulp). It is bounded absolutely against the target scale instead.

**Verification limits, unchanged and important**: execution is host-only. `rls.hpp` is checked to
compile `-ffreestanding` and `argmind.h` to be valid C99, on the host. Host `-Os` code size is 576
bytes for `Rls<float,6,1>::update` — **x86-64, a shape, not a target figure**. Flash/RAM for the
M4F floor must come from a link map, and on-target timing claims stay absent.

**The Guard is across too** (`cpp/include/argmind/guard.hpp`). `Guard<Adapter, Depth,
MaxBounds>` — the checkpoint ring and check capacity live in the type, so the memory is
declared: `Guard<Rls<float,6,1>, 4, 4>` is 1208 bytes of which 736 is the ring.

**One structural difference, forced not chosen**: the Python's list of polymorphic `Check`
objects needs virtuals and a heap, and the core has neither. Checks become concrete members
with declared capacity and their identities become an enum, which is what a `tripped` bitmask
is built from. The cost is that a user cannot supply their own check type — the same trade the
C ABI makes for compile-time dimensions.

A guard *decides* rather than computes, so parity here is about decisions and a decision that
differs is not a rounding difference. Matching exactly at both precisions, across depths
1/2/4/10 and cadences 1 and 5: freeze step, thaw step, the whole frozen vector, which steps
rolled back, the set of failing checks, and which checkpoint a trip restores.

**A mutation check on the port found a bug in the reference.** `Guard.freeze()` read its health
scalars from the newest checkpoint. Exact after a *trip* — the state has just been restored to
that checkpoint, and `_probe`'s docstring justified it on that ground — but after a *manual*
freeze nothing was restored, so the value could be `checkpoint_every` updates stale: 300.0
reported against a live 104.1, three updates into a cadence of five. A `Bound` on `p_trace`
during a manual freeze read a state the adapter was no longer in. Both implementations now take
the live values. That leaves the ring's stored health scalars redundant; removing them would
drop the declared cost to `Depth * state_size`, and it is **recorded in DESIGN.md rather than
half-done.**

**Two flattering results had to be withdrawn, both from one broken fixture.** With `p0=100` and
six features `tr(P)` starts at 600, so a `p_trace` bound below that trips on the *first* update
— a sweep over four bounds compared step 0 against step 0 four times. The fixture is now a
windup scenario where `tr(P)` grows as exactly `p0·λ^-k`, and the tests pin the *predicted* step
(`ln(hi)/ln(1/λ)`: 45/90/135 predicted, 44/89/134 measured). With that broken fixture the
restored states also came out bit-identical, which looked like proof that a rollback copies
rather than recomputes — it proved nothing, since both rolled back to the *construction* state.
At a real trip step they differ by ~41 ulp. What is verifiable is *which* checkpoint each side
chose, tested by comparing each side against **its own** unguarded trajectory.

**How to write these tests, learned the expensive way.** Eight of eighteen mutants survived the
first run because there were no Guard tests in the C++ unit suite at all — a parity harness can
only test what the tool exercises. The four that belonged in the parity suite were then designed
*against their mutants*, and the first attempt at all four failed to discriminate: a second
spike one sample later lands while still frozen; a second fault 240 samples after a thaw arrives
once the EWMAs have re-converged; bounds a rejected report satisfies anyway make a leaked
rejection identical (the observable that works is the *checkpoint* it stores); and one clean
sample in three never lets the fast average fall below the threshold. **A test written without
checking that its bug would fail it is a guess.**

**Next:** `Pipeline` and persistence in C++, then the on-target build once a cross-compiler and
hardware exist.

**Before any optimisation lands, a benchmark that can report a regression must exist.** An
optimisation with no way to fail is not an optimisation, it is a hope.

**Target range — settled 2026-08-08. The floor is Cortex-M4F class**: ARMv7E-M with a
single-precision hardware FPU (M4F/M33/M7, or RISC-V with F). fp32 FPU required, 8 KB RAM for the
library, 32 KB flash, C++17 freestanding with no exceptions/RTTI/heap, reference flags
`-mcpu=cortex-m4 -mfpu=fpv4-sp-d16 -mfloat-abi=hard`. That closes the matrix-type question: **own
minimal fixed-size types, no Eigen.**

**What decided it was the precision floor, not size or speed, which was not the expected answer.**
Computed from the reference's declared shapes: the driver configuration needs 3.3 KB of RAM, and an
analytic flop count puts n=6 at 5 Hz at 1.3 kflop/s — even n=30 at 200 Hz is ~2% of a 48 MHz M4F
(that ratio an estimate from published cycle counts, not measured). Both comfortable. The binding
constraint is that every covariance operation is fp32 with no fp16 path, so without an FPU it is
soft float at 20–100 cycles per op on data-dependent paths, and this project bounds the *worst*
case. An FPU is what makes the timing analysable.

**M0+/M3 are not supported and not claimed** — they would probably work (n=6 computes to 0.27 ms
per soft-float update, 0.14% duty at 5 Hz) but nobody has run it on one, and writing untested
support down is how a claim becomes a fiction. No ARM cross-compiler or target hardware exists on
this machine: flash/RAM will come from a link map, execution is verified on the host, and **any
on-target timing claim stays absent rather than interpolated from host numbers.**

**Two port notes** from that review, deliberately not applied to Python where call overhead hides
them: `P /= lambda` is n² divisions (36 at n=6; `VDIV.F32` is 14 cycles on an M4F, so ~500 cycles
for one line) and must become one reciprocal plus n² multiplies, as must `k = Px / den`. And the
Python timings are **not** a baseline for the C++ — they are interpreter and numpy dispatch cost,
which the port does not pay, so comparing them would flatter it for reasons unrelated to the port.

## Sub-agent team

Five project-scoped agents in `.claude/agents/`, checked in for the same reason `.githooks/`
is — the discipline belongs to the project, not to one machine. Each covers work that was
done by hand at least twice, with this project's actual mistakes encoded in it. See
`.claude/agents/README.md` for why these five and what was deliberately left out.

| agent | when to reach for it |
|---|---|
| `evidence-auditor` | before a commit stating a number; after editing DESIGN.md or CLAUDE.md; when a result looks good |
| `numerics-reviewer` | before declaring a precision floor or a tolerance; when an estimator misbehaves |
| `mutation-tester` | after writing a module and its tests; before claiming a suite protects something |
| `parity-verifier` | after any change to `cpp/` or the Python adapters; when a tolerance is under pressure |
| `embedded-reviewer` | after any change under `cpp/`; before adding an ABI entry point |

After a change under `cpp/`, run `embedded-reviewer` and `parity-verifier` together: does it
still fit the target, and does it still compute the right thing. Independent questions.

A sub-agent's conclusion is a claim about the repository, and claims get verified.

## Working state

No git remote is configured — deliberately, so a stray `push` cannot write to the retired Google
Drive location. Add one when publishing.
