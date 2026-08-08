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

The design driver, chosen so the API is shaped against something real: an inertial + aiding-sensor
estimator with online bias calibration. Sensor-agnostic in the API, specific in what it must
handle.

**Built:** `argmind/stream.py` (fixed-capacity ring buffers, counted drops, monotonic time
enforced at the source), `argmind/align.py` (four explicit alignment policies, signed staleness,
causal/non-causal declared), `argmind/adapt.py` (the `Adapter` contract and
`RecursiveLeastSquares`), `argmind/guard.py` (`Check`/`Bound`/`Divergence`, freeze, bounded
rollback, explicit recovery), `argmind/pipeline.py` (`FrameMap`, input gate, bounded `step()`),
`argmind/persist.py` (two-slot atomic store, CRC, configuration fingerprint, declared cadence),
and `argmind/bench.py` (fixed workload, correctness digest, leak detector, regression gate).
**The Python reference is complete and measurable.** 166 tests pass; several exist to prove
failures are visible rather than silent, and every new suite was mutation-checked — fifty-nine
deliberate bugs, all caught.

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

**Next:** the C++ core, verified against the Python reference. Everything it needs is in place.
Still gated on the one remaining open question: whether the target range reaches down to MCUs,
which decides own-minimal-matrix-types versus Eigen and the `-mcpu` set.

**Before any optimisation lands, a benchmark that can report a regression must exist.** An
optimisation with no way to fail is not an optimisation, it is a hope.

**One question still open:** whether the target range reaches down to MCUs. It decides the matrix
types (own minimal types vs Eigen) and the `-mcpu` set. It does not block the Python reference.

## Working state

No git remote is configured — deliberately, so a stray `push` cannot write to the retired Google
Drive location. Add one when publishing.
