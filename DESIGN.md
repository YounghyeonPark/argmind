# Argmind — design

**A sensor-agnostic library for on-device online adaptation and inference, for embedded and
edge targets.** Started 2026-08-08. Status: design under review, nothing built yet.

## What it is

Multi-sensor systems in the field drift. Bias moves with temperature, a mount shifts, a surface
changes, a battery ages, a user is not the average user. The usual answer is to train offline and
hope the deployment matches the training set. This library takes the other route: **the model
keeps updating from the sensor stream after deployment**, inside bounded memory and bounded time,
with explicit guardrails so that adaptation cannot quietly destroy the system it is adapting.

It is a **library**, not an application and not a pipeline tool. The API is the product.

## What it is not

Naming these now because a sensor-agnostic framework drifts into "everything" otherwise.

- **Not a training framework.** No backpropagation, no autograd, no GPU. Offline training is a
  solved problem with excellent tools; use them and bring the result here.
- **Not a deep-learning runtime.** If you need to run a CNN, use ONNX Runtime or TFLite. This
  library adapts *parameters and estimates*, and can sit alongside such a runtime.
- **Not a driver layer or a robotics middleware.** It does not talk to hardware, does not own
  threads, and does not depend on ROS. You feed it samples.
- **Not a research playground for online deep learning.** The adaptation methods here are ones
  whose stability can be reasoned about.

## The design driver

The framework is sensor-agnostic, but designing against nothing produces an API that fits
nothing. So one concrete case drives the design, and it is stated rather than assumed:

> **An inertial + aiding-sensor estimator whose sensor biases are calibrated online.** IMU at
> 100–1000 Hz, an aiding source at 1–10 Hz, clocks that disagree, samples that arrive late or not
> at all, and biases that drift with temperature. Adaptation must not diverge while the vehicle
> is stationary, and must survive an aiding-source dropout.

Every abstraction below must serve this cleanly *and* be expressible without mentioning inertial
sensors. Where those two conflict, generality loses — a general API nobody can use for a real
problem is worth less than a specific one someone can.

## The four hard parts

A framework earns its existence by solving problems its users would otherwise solve badly.

**1. Multi-rate, asynchronous streams.** Real sensors do not arrive on a shared grid. They have
different rates, clock offsets, jitter, and dropouts. Fusing them requires an explicit alignment
policy, and the choice of policy changes the answer. Most code gets this wrong silently by
zipping arrays.

**2. Adaptation that cannot be trusted blindly.** An online learner in a deployed system can
diverge, chase noise, or overfit a transient. This is the part that makes online learning
frightening in production, and a framework that leaves it to the user is incomplete.

**3. Bounded resources.** Embedded means no unbounded buffers and no allocation in the steady
state. Anything whose memory grows with time is a bug, not a tuning parameter.

**4. Observability.** An adaptive system you cannot inspect is an adaptive system you cannot
debug. Why did the estimate move? Which stream was stale? When did the guard trip?

## Core abstractions

```
Sample      one observation: timestamp, values, validity
Stream      a declared source: name, dtype, dimension, nominal rate
Aligner     N streams -> synchronised frames, under an explicit policy
Transform   per-stream preprocessing, stateless or stateful
Adapter     the online learner: update(frame) / predict(x)
Guard       health and stability: divergence, bounds, freeze, rollback
Pipeline    wires the above; step() performs bounded work
```

### Design decisions, and why

**Pull-based and step-driven.** `pipeline.step()` does a bounded amount of work and returns. No
threads, no callbacks, no event loop inside the library. The caller owns timing — which is what
an embedded integrator needs, and what makes the whole thing deterministically testable.

**Preallocated, bounded buffers.** Every buffer declares its capacity at construction. No
allocation in the steady state. Memory is a function of configuration, not of uptime.

**The Guard is first-class, not an afterthought.** Adaptation is gated: an update that fails a
health check does not silently apply. The framework's opinion is that *shipping online learning
without a freeze path is negligent*, so the API makes the freeze path the default shape rather
than something the user remembers to add.

**Built 2026-08-08, and it changed the API sketch below: a `Guard` wraps an `Adapter` and *is*
an `Adapter`,** rather than sitting beside one in the pipeline. Side by side leaves "no guard"
expressible, and a safety mechanism that can be left out is one that gets left out. Wrapping
makes the guarded object interchangeable with the raw one, so a caller holds exactly one adapter
and cannot tell whether it is guarded except by the extra fields on the report.

Checks run *after* the update, because the quantities worth checking only exist once the update
has been computed. That is why rollback exists rather than pure prevention — the state has already
moved by the time anything is known about it, so the guard restores a checkpoint instead of
vetoing in advance. Checkpoints cost `rollback_depth * (state_size + 2)` elements, fixed at
construction, and the guard rolls back to the *deepest* retained one: divergence is detected after
it starts, so undoing a single update lands inside the bad stretch.

**Precision is declared and checked, not assumed.** Carried forward from measurement in the
parked accelerator project: **the EKF covariance recursion needs fp32 minimum, and there is no
fp16/bf16 path in any filter form** — conventional, Joseph and square-root all lose
positive-definiteness on the first step, and fp16 additionally overflows at κ(P)=1e6. More
generally, **precision floors are a property of the algorithm, not of the platform.** So an
Adapter declares the precision it requires and the framework refuses an unsafe configuration at
construction time rather than producing quiet garbage at hour six.

**That floor was re-measured for RLS rather than inherited** (2026-08-08). Carrying the EKF
conclusion across would have been the borrowed-assumption error the standards below exist to
catch, so the same recursion was run at f2/f4/f8 with every regressor held inside fp16's range,
varying only the conditioning — so a failure is the arithmetic's, not the format's dynamic range.
Measured, at n=6, λ=0.98, p0=1e2:

- **fp16 fails outright even at κ=1**, losing positive-definiteness by step 115; at κ=10 by step
  3, with the parameter error reaching NaN. `RecursiveLeastSquares` refuses it at construction.
- **fp32 survives but does not maintain positive-definiteness exactly.** With well-conditioned
  regressors of magnitude ≥1e2 against p0=1e2, it showed a negative eigenvalue on the *first*
  update — at roundoff scale (≈-5e-7 against a trace of order 1e2), not a blowup, and the
  parameters stayed sound. fp64 never lost it up to magnitude 1e6.

So fp32 is the floor and it is not a comfortable one. If the roundoff-scale indefiniteness ever
matters on a target, **the fix is a factored form (square-root/UD), not more precision** —
the conventional recursion is the weak one. Both halves are protected by tests, including the
unflattering half. `p0` is a prior covariance and interacts with the regressor scale: it is
`p0 * ||x||²  >> 1` that puts the first update in the cancellation regime, so it should be
scaled to the problem rather than left large by default.

**Alignment policy is explicit and never defaulted silently.** `nearest`, `hold-last`,
`interpolate`, `drop-if-stale` give different answers. The user chooses; the framework records
the choice and reports staleness.

## API sketch — for review, not final

Updated 2026-08-08 for the guard as built — it wraps rather than sits beside, for the reason
given above.

```python
from argmind import (Stream, Pipeline, Aligner, Guard, Bound, Divergence,
                     RecursiveLeastSquares)

imu   = Stream("imu",  dim=6, rate_hz=200, dtype="f4")
aid   = Stream("aid",  dim=3, rate_hz=5,   dtype="f4")

pipe = Pipeline(
    streams=[imu, aid],
    aligner=Aligner.hold_last(reference="imu", max_stale_s=0.5),
    adapter=Guard(
        RecursiveLeastSquares(n_features=6, forgetting=0.999, dtype="f4"),
        checks=[Divergence(ratio=5.0, floor=1e-3),
                Bound("p_trace", hi=1e4),
                Bound("p_min_diag", lo=0.0)],
        rollback_depth=10,
        checkpoint_every=5,       # reaches 50 updates back, for 10 * (state+2)
        recover_after=None,       # thaw() is deliberate; nothing thaws itself
    ),
)

imu.push(t=0.005, values=[...])       # bounded ring buffer, no allocation
aid.push(t=0.004, values=[...])

report = pipe.step()                  # bounded work; never blocks
report.updated                        # did adaptation apply this step?
report.frozen_by                      # which guard, if any
report.staleness_s                    # per stream
y = pipe.predict(x)
```

The shape to notice: **`step()` returns a report rather than nothing.** Observability is not a
logging afterthought; it is the return value.

## Adaptation methods, in the order they should exist

Chosen for stability that can be argued about, not for novelty.

1. **Recursive least squares** with a forgetting factor — the workhorse. Bounded state, well
   understood, and directly useful for online calibration. **Built** (Python reference,
   `argmind/adapt.py`), checked against the closed-form exponentially weighted solution rather
   than against itself.
2. **Kalman-family parameter estimation** — an EKF whose state is the parameters. Carries the
   covariance conditioning problem, and therefore the precision floor above.
3. **Online gradient methods** for the cases the first two cannot express, with explicit step-size
   control.
4. **Drift/change detection** to decide *when* to adapt, which is often more valuable than how.

## Open questions

Genuinely open, listed rather than guessed:

- ~~Language~~ — **DECIDED 2026-08-08: a C++ core with a C ABI, optimised for speed.** See
  "Language and structure" below.
- ~~Target range~~ — **DECIDED 2026-08-08: the floor is Cortex-M4F class**, ARMv7E-M with a
  single-precision hardware FPU. Set by the measured fp32 precision floor rather than by memory or
  speed, both of which are comfortable. See "Target range" below. This closes the matrix-type
  question with own minimal fixed-size types and no Eigen.
- **How much of a timebase to own.** Does the library assume a monotonic clock per stream, or
  handle clock skew estimation between sources?
- **Where inference sits.** The library adapts parameters and estimates. Whether it also wraps a
  third-party inference runtime, or simply provides adapted inputs to one, is a scope boundary
  not yet drawn.
- ~~**Persistence**~~ — **BUILT 2026-08-08** (`argmind/persist.py`). See below.
- ~~**Whether a guard's own state should persist**~~ — **ANSWERED 2026-08-08.** The checkpoints
  and the EWMAs inside checks are not persisted: they describe a run, and a run ends at power-off,
  so restoring them would assert a continuity that does not exist. The *frozen flag* is carried,
  with what tripped it, because it is a conclusion rather than working state. What to do with it
  on load is a real decision with no correct default, so it is the caller's — `on_frozen` is
  `"thaw"` (default), `"keep"` or `"refuse"`. Both arguments stand and neither wins in general:
  thawing risks re-learning into the same trouble; keeping risks stranding a device that a power
  cycle would have fixed, with no way back except a site visit. The record says which happened.

## Language and structure — decided 2026-08-08

**A C++ core behind a C ABI. Python becomes bindings and the reference oracle.**

### C++ rather than C

Not for the usual reasons. This library lives at small fixed matrix sizes — sensor-fusion state
dimensions are typically 3 to 30 — and at those sizes runtime is dominated by call overhead,
memory layout, and whether the compiler can unroll and vectorise. **Compile-time dimensions and
precision are what make that possible**, and in C the same thing is macro duplication or a
runtime `n` the optimiser cannot exploit. Templates give it for free:

```cpp
Rls<float, 6> rls;        // dimensions known at compile time: unrolled, no bounds
                          // checks, no allocation, SIMD-friendly
```

C++ costs are real and are paid deliberately: **no exceptions, no RTTI, no dynamic allocation**
(`-fno-exceptions -fno-rtti`), so the core stays usable in a freestanding environment.

### A C ABI at the boundary

Everything crossing the library boundary is `extern "C"` with plain structs and pointers. Two
reasons: it lets a plain-C firmware, Python, Rust or MATLAB call in without knowing C++ is
inside; and it insulates callers from C++ ABI instability across compilers and versions, which
is a genuine hazard for a library shipped as a binary.

### Target range — decided 2026-08-08: Cortex-M4F class

**The floor is ARMv7E-M with a single-precision hardware FPU** — Cortex-M4F, M33, M7 and up, or
a RISC-V core with the F extension. Concretely, as a testable contract:

| | |
|---|---|
| FPU | fp32 in hardware, required |
| RAM for the library | 8 KB (the design driver's configuration computes to 3.3 KB) |
| flash | 32 KB, code plus the two persistence slots |
| language | C++17, freestanding, `-fno-exceptions -fno-rtti`, no heap |
| reference flags | `-mcpu=cortex-m4 -mfpu=fpv4-sp-d16 -mfloat-abi=hard` |

**What decided it was the precision floor, not size or speed** — which was not the expected
answer. RAM and compute are both comfortable by a wide margin. Computed from the reference's own
declared shapes, and marked as computed because nothing here has been measured on a device:

| configuration | RAM | note |
|---|---|---|
| driver, n=6 f4, rollback depth 4, capacity 64 | 3.3 KB | the workload the API was designed against |
| minimal, n=6 f4, depth 1, capacity 16 | 1.4 KB | |
| n=30 f8, depth 10, capacity 128 | 125 KB | only the top end needs real memory |

Compute is smaller still: an analytic flop count gives ~262 flops per RLS update at n=6, so 1.3
kflop/s at the driver's 5 Hz aiding rate. Even n=30 driven at 200 Hz is 1.1 Mflop/s, roughly 2% of
a 48 MHz M4F — that last figure being an *estimate* from the core's published cycle counts rather
than a measurement.

So the binding constraint is the measured fp32 floor (see the precision section above). Every
covariance operation is fp32 and there is no fp16 path in any filter form. Without an FPU that
becomes software floating point at 20–100 cycles per operation, on data-dependent paths — and this
project's standard is to bound the *worst* case, which soft-float division makes hard. An FPU is
what makes the timing analysable, and that is the boundary.

**M0+ and M3 are not supported and are not claimed.** They would probably work: n=6 in soft float
computes to about 0.27 ms per update, a 0.14% duty cycle at 5 Hz. But "probably works, by
calculation, on hardware nobody has run it on" is not support, and writing it down as support is
how a claim becomes a fiction.

**Verification, and its honest limit.** There is no ARM cross-compiler on the development machine
at the time of writing, and no target hardware. What can be verified without either is that the
core cross-compiles freestanding under the flags above and what it costs — flash and RAM come out
of the link map, which is real evidence. Execution will be verified on the host against the Python
reference. **Any on-target timing claim must be labelled as absent until a device exists**, not
interpolated from host numbers.

### Own the matrix types — settled by the decision above

Eigen is excellent and is the default in robotics state estimation, but it is heavy for a
Cortex-M4F: header bloat, compile times, and a C++14+ requirement. With MCUs in range the question
is closed — **a minimal fixed-size type covering exactly what the adapters need, and no Eigen.** At
n=6 Eigen would buy nothing anyway: the measurement above says the cost at these sizes is call
overhead, not arithmetic.

### Port notes, collected while deciding the target

Things that are invisible in the Python reference because call overhead swamps them, and that
matter on an M4F. Recorded rather than acted on, since changing them in Python would be optimising
a number nobody should be reading:

- **`P /= lambda` is n² divisions.** At n=6 that is 36, and `VDIV.F32` on an M4F is 14 cycles —
  about 500 cycles for one line. The C++ must compute one reciprocal and do n² multiplies. The
  same applies to `k = Px / den`: one reciprocal, n multiplies.
- **The Python timings are not a baseline for the C++.** They are dominated by interpreter and
  numpy dispatch overhead, which the C++ does not pay. Comparing the two would flatter the port by
  a factor that has nothing to do with the port.

### Do not call a BLAS at these sizes — which is not the same as forgoing hardware acceleration

This is evidence, not preference. Measured in the parked project on the same class of problem:
at n=80, **numpy/OpenBLAS ran 12.8× SLOWER than a plain C loop**, because the per-call dispatch
overhead of a tuned BLAS exceeded the arithmetic it was asked to perform. Sensor-fusion matrices
are smaller still, and the ratio gets worse as they shrink.

**BLAS is a library, not silicon.** What is being avoided is the call cost — dispatch, runtime
CPU feature detection, threading decisions, argument validation — not the SIMD units. Those are
still used, reached a different way:

| approach | suits | here |
|---|---|---|
| compiler auto-vectorisation of inlined fixed-size loops | small, fixed sizes | **this** — portable, zero dependency |
| hand-written SIMD intrinsics | where auto-vectorisation measurably failed | later, and only where measured |
| a BLAS | large matrices | no |
| offload to an NPU/DSP/accelerator | large batched work | **no — see below** |

With dimensions fixed at compile time the compiler can fully unroll, keep operands in registers,
and emit NEON/AVX/SVE without runtime checks. Same silicon, no call overhead.

**Offloading to a dedicated accelerator would lose here, and that is also measured.** The parked
project found an invocation cost of ~0.8 µs, and any workload smaller than that is net-negative
to offload — it recorded a case where the accelerator computed a tick 1.24× faster than the host
core and then spent 0.8 µs being *asked*, ending below parity. A 6×6 recursive update sits well
inside that regime. Run it on the CPU and use the CPU's vector units.

**Auto-vectorisation is not guaranteed, so it must be verified rather than assumed.** Compilers
give up on aliasing uncertainty, non-contiguous access, and reductions under strict
floating-point semantics. Therefore: mark pointers `__restrict`, keep data contiguous and
aligned, and **check `-fopt-info-vec` or the emitted assembly**. Assuming acceleration one does
not have is exactly the class of error this project's standards exist to catch.

One expectation to set: the tuned C measured in the parked project reached **9.3% of the
machine's SIMD peak**, not 80%. At these sizes the work is operand-fetch-bound rather than
compute-bound, so peak is unreachable — by any method, BLAS included. That is normal and not a
sign of a bad implementation.

### What happens to the Python already written

`Stream` and `Aligner` become the **reference implementation and test oracle**, not throwaway
prototype. The C++ is verified against them — same inputs, same outputs, to the tolerance the
precision allows. This pattern worked well in the parked project, where a functional simulator
matched a reference solver bit-for-bit and that agreement is what made later optimisation safe.

### Before "optimised for speed" means anything

**A benchmark that can report a regression has to exist first**, or the criterion is
unfalsifiable and every change looks like an improvement. Required before the first optimisation
lands, not after:

1. A fixed workload derived from the design driver — declared rates, declared dimensions.
2. The Python reference as the correctness oracle, so speed work cannot quietly change answers.
3. Reporting that separates **per-sample cost** from **per-frame cost**, because they scale with
   different things and an average hides which one moved.

Carried from the parked project: measure worst case, not just typical. A p50 that looks good and
a p99 that misses is the shape that fails in the field.

**BUILT 2026-08-08** — `argmind/bench.py`, all three requirements met, plus two things the writing
of it turned up.

**Absolute timings are not committed as a gate**, and that is a decision rather than an omission.
A recorded baseline is a statement about one machine, one CPU governor and one Python build; as a
repository gate it fails on everyone else's laptop and gets switched off within a week. Record
locally with `--record`, compare with `--compare`, which exits non-zero. What the *test suite*
gates on is what survives moving between machines: the exact counters, the state digest, the
memory leak rate, and ratios measured inside one run. The comparison tolerance is widened by the
run-to-run spread `--record` measures on that machine, rather than by a guessed constant.

**Measured on this codebase, and it confirms the C++ argument above rather than echoing it.**
`argmind.bench.scaling` times one RLS update against the state dimension:

| n | ns/update | ratio |
|---|---|---|
| 4 | 22525 | — |
| 8 | 22416 | ×1.00 |
| 16 | 22356 | ×1.00 |
| 32 | 24410 | ×1.09 |
| 64 | 32127 | ×1.32 |
| 128 | 70231 | ×2.19 |

RLS is O(n²), so cost should rise 64-fold from n=4 to n=32. It rises 1.09-fold. **At the design
driver's dimension of six, essentially all of the time is call overhead and none of it is
arithmetic** — the exact regime the language decision was made for, now measured here rather than
inferred from the parked project. It also sets an expectation for the port: the C++ core must not
be judged against these figures, because it is not paying the cost they are made of. A naive
scaling test asserting O(n²) would have failed, which is why it was measured before being
asserted.

**"Memory is a function of configuration, not of uptime" is now checked**, having been asserted
throughout this document with nothing behind it. Getting a trustworthy answer took two corrections
and both are worth keeping:

- *Fitting a trend line to successive batches reports growth that is a settling transient.* The
  series converges — buffers fill, caches settle — and a straight line through it gave +414
  bytes/batch on deltas that ran 1126, 375, 375, 179, 211 and then repeatedly exactly 0. What
  works is a discriminator: measure at several batch *sizes* and fit `cost = fixed + leak ×
  frames`. Cost per frame is not the leak rate; only the slope is.
- *`tracemalloc` allocates while it measures*, and its own bookkeeping dominated the residual it
  was being used to find. `sys.getallocatedblocks` does not.

The measurement then found one real thing: `Adapter.state` built a fresh read-only view on every
access, on a path the guard takes at every checkpoint. It is now built once at construction. After
that the leak rate is −0.004 blocks/frame, i.e. zero within noise. The detector is kept honest by
a positive control — an adapter that keeps every report it has produced is detected at over 1
block/frame — because a leak detector that has never detected a leak is not evidence.

## Where this stands, and what comes next

**Built:** `Stream` (fixed-capacity ring buffers, counted overflow, monotonic-time
enforcement), `Aligner` (four explicit policies, staleness reporting, causal/non-causal
declared), `Adapter` + `RecursiveLeastSquares` (flat state block, forgetting factor, per-update
report, measured precision floor), `Guard` + `Bound` + `Divergence` (checks, freeze, bounded
rollback, explicit recovery), `Pipeline` + `FrameMap` (frame to regressor by index table,
input gate, bounded `step()`), and `Store` (two-slot atomic persistence, CRC, configuration
fingerprint, declared cadence), `bench` (fixed workload, correctness digest, leak detector,
regression gate), and **the C++ core — RLS behind a C ABI, plus the guard, the record format,
streams, alignment and the pipeline as headers — verified against the Python to about 1.5 ulp,
byte for byte for the record format, and exactly for every decision and counter above that**.
The runtime library is fully across; `argmind/bench.py` has no C++ counterpart and is not
getting one, being a host measurement harness rather than part of what ships. (Everything the library does is reachable from C: RLS, the
guard, the record format, caller-owned rings and the composed pipeline.)
496 Python tests plus 4632 C++ checks, several of which exist to prove failures are visible
rather than silent.

**What the pipeline settled:**

- **A `Frame` becomes `(x, y)` through a declared index table, not a callable.** A callable is the
  obvious Python answer and does not cross a C ABI; a pair of `(stream, component)` lists is two
  integer arrays on the other side. `FrameMap` takes convenient forms — a bare name, `(name, i)`,
  a list, a slice — and resolves them *once* at construction against the streams that exist, so
  the convenience costs nothing at run time and a typo fails at construction rather than at 3am.
  Order is the parameter order, and a persisted state block restored against a different order is
  wrong in a way nothing downstream can detect.
- **Two gates, at two different times.** The pipeline gates on *input* health — staleness, sensor
  validity — before the update, because bad input should not reach the learner. The `Guard` gates
  on *model* health after it, because those quantities do not exist until the update has been
  computed. Collapsing them would mean either checking the model before it moved or letting
  known-bad data into it first.
- **A non-causal aligner policy has to be asked for.** The `Aligner` alone permits `nearest` and
  `interpolate` and records that they are non-causal, which is right for offline use. A pipeline
  *is* the causal loop, so it refuses them unless `allow_non_causal=True`.
- **`StepReport.lost_frames` exists because a test found the gap.** A 200-frame workload against a
  64-deep ring adapted 64 times, reported nothing amiss, and produced a plausible fit from a third
  of the data. The first fix was wrong in the other direction: `Stream.dropped` counts
  *overwrites*, which a healthy ring does on every push, so reporting those raises an alarm on
  every working system. What is actually countable is reference samples that were overwritten
  before they could become frames — each one a frame that will never exist. `Stream.dropped`'s
  docstring was overclaiming and has been corrected.

**What the adapter contract settled**, since everything after it has to fit:

- State is **one flat contiguous block** (`adapter.state`, `theta` then `P`). Persistence is a
  copy of that block and nothing else, and it is what crosses the C ABI later without a
  serialisation format in the middle. Loading one validates it, because flash comes back
  truncated and bit-flipped and a stored NaN poisons every subsequent update.
- **`update()` returns an `UpdateReport`, never nothing** — a priori and a posteriori residual,
  gain norm, `tr(P)`, and the smallest diagonal of `P`. The `Guard` is written against exactly
  these fields; an adapter that only mutated state silently would make the guard impossible.
  `p_min_diag` is an O(n) proxy for the O(n³) eigenvalue test, chosen because it has to be
  affordable on every update on an MCU.
- **Shape errors raise; bad values do not.** A wrong shape is a bug at the call site. A NaN from
  a sensor is a runtime event: it is rejected, counted, named in the report, and leaves the state
  untouched. Taking the estimator down with it is the wrong response in a deployed system.
- **Covariance windup is left visible, not clamped.** Under poor excitation `P` grows exactly as
  `p0·λ^-k` in the unexcited direction (verified to 12 digits) and the next off-axis sample
  bursts. Bounding it inside the adapter would make the `Guard` optional — which is the mistake
  this design explicitly refuses. It is reported; deciding what to do about it is the guard's job.

**What the guard settled, and what it cost to get right.** Two design errors were made and caught
by tests, both in the same place — what a stateful detector should do while the guard is frozen:

- Resetting the checks on a trip looked tidy and threw away the only reference the frozen probes
  have. A fault that had plateaued then read as normal and **the guard thawed itself in the middle
  of the fault.** Checks are now reset on `thaw()` instead.
- Holding the detector's baseline *per sample* — refusing any sample the check disliked — was the
  first fix and was worse: the baseline only ever absorbs the smaller residuals, so it ratchets
  downward until ordinary noise trips it. The hold is per-*freeze*, so during normal operation the
  baseline stays unbiased and during a fault it stays a reference.

Both are protected by tests, and both survive as named mutants in the mutation check. The general
lesson is the one already in the standards: the tidy-looking version was the broken one.

**What persistence settled**, each piece present because of a specific way stored state goes wrong
on a device:

- **Two slots, written alternately, each with a sequence number and CRC.** A write interrupted by
  power loss can destroy at most the slot being written, which is never the one in use. The
  reference writes files but models the raw-flash scheme faithfully — the C side will not have
  `os.replace`, so a design leaning on rename would not survive the port.
- **A configuration fingerprint, refused on mismatch.** Shape and precision catch the crude cases.
  The dangerous case is `FrameMap` order: a block restored after two features were swapped is
  numerically valid, silently wrong, and undetectable downstream. `fingerprint_of` is `crc32` over
  a canonical rendering rather than Python's `hash()`, which is salted per process and would
  reject every valid block on the second run — a failure that only appears in the field.
- **Cadence is a declared policy, not caller discipline.** A sector rated for 100k erase cycles,
  written on every update at 100 Hz, is gone in about seventeen minutes.
- **Little-endian with explicit widths.** The writer and the reader may be different machines, and
  "whatever this compiler does" is not a format. Worth noting that this one cannot be verified by
  comparing bytes on an x86 host, where `<` and `=` are identical — the declaration itself is what
  the test pins, and the test says so.
- **No state at all is a return value; a state that does not fit is an exception.** First boot is
  normal and should not need a try/except around every start-up. A mismatched block is a
  configuration error, and continuing from one is worse than not starting.

**Next:**

**The C++ core**, verified against the Python reference — same inputs, same outputs, to the
tolerance the declared precision allows. Everything it needs is now in place: a contract shaped
for a C ABI, a reference anchored to closed-form solutions rather than to itself, a state block
that is already the persistence format, and a benchmark that can report a regression.

**STARTED 2026-08-08** — `cpp/`, and RLS is across. `mat.hpp` (fixed-size primitives),
`rls.hpp` (`Rls<T, N, M>`), `argmind.h` + `src/abi.cpp` (the C ABI), `tools/xcheck.cpp`, `tools/pcheck.cpp`, `tools/scheck.cpp` and
`tools/pipecheck.cpp` (the four parity tools), `tests/test_core.cpp` (4603 checks the C++ side can make on its own), and a
`CMakeLists.txt` that knows the cross-compilation recipe.

**The C ABI had one real tension to resolve.** Compile-time dimensions are the reason C++ was
chosen; a C ABI wants runtime arguments. Taking `n` at runtime resolves it by discarding the
reason, so it is resolved the other way: **the ABI exports an explicit list of instantiations**,
and adding a dimension is one line plus a rebuild. That is a real cost and it is the honest one.
Storage is the caller's — nothing allocates — with `am_rls_*_bytes()` for the true size and an
`AM_*_MAX_BYTES` compile-time *upper bound* for statically declared arrays, held to being a bound
by a `static_assert` per instantiation.

**Parity: the two agree to a handful of ulp on the state block, at whichever precision they
run at.** That last clause is the finding. Measured maxima in ulp over every shape `xcheck`
instantiates -- (n, m) of (2,1), (3,1), (6,1), (12,1) and (6,3) -- at 300, 800 and 1600 steps,
λ = 0.97 and 0.99, p0 = 100, on both architectures:

| quantity | host f64 | host f32 | target f64 | target f32 |
|---|---|---|---|---|
| whole state block, and θ | 2.0 | 2.0 | 2.0 | 5.0 |
| tr(P) | 10.0 | 4.9 | 10.0 | 2.2 |
| min diag P | 8.4 | 10.1 | 8.4 | 8.4 |
| residual norm | 30.1 | 14.2 | 30.1 | 38.0 |
| P alone | 29.6 | 43.2 | 29.6 | 40.8 |
| gain norm | 110.0 | 87.4 | 110.0 | 134.0 |
| a posteriori residual norm, against max\|y\| | 1.8 | 1.4 | 1.8 | 3.6 |

**This table replaces one that was understated on almost every row**, and how it got that way
is the useful part. It said its maxima were taken over "n=2/6/12, one and three outputs,
λ=0.97 and 0.99", and an audit found every figure reproducing only at λ=0.99: `gain_norm` was
58 where 0.97 gives 110, `p_trace` 1.4 where it gives 10.0. The axis was named in the prose and
not swept in the measurement. The same description also implied three outputs at every n, and
`xcheck` instantiates (6,3) alone.

Two columns are new because they have to be: `VFMA.F32` contraction makes the target's f32
figures its own (`rls.hpp`'s fourth divergence), and f64 is identical between the two, having
nothing to contract on a single-precision FPU.

The last row is a different metric from the rest and is not comparable to the 597 the old table
carried: that was element-wise relative error on a quantity produced by cancellation, which is
what the paragraphs below explain the project stopped doing. It is bounded absolutely against
the scale of the values being subtracted.

**Differing by the same number of ulp at both precisions is what says the difference is
algorithmic rounding rather than a precision-dependent bug.** Three divergences are deliberate and
declared in `rls.hpp`: `P /= λ` and `k = Px/den` become reciprocal multiplies (the port notes
above), and numpy's BLAS reductions sum in a different order than an ordered loop.

**The parity metric took two corrections, and neither was obvious.** Element-wise relative error —
the obvious choice — reads 1e-13 at f64 and 1e-4 at f32 on a state block, because such a block
contains entries that are near zero *through cancellation* and a relative error there measures the
cancellation. Deviation relative to the block's scale is the honest measure. Then the a posteriori
residual turned out not to belong in a relative comparison at all: it is `y - x'θ` after fitting to
that very sample, so at the worst step the two agree on 6.72e-5 and disagree by 1.5e-15 — a few ulp
of the ~3-magnitude predictor amplified by a cancellation factor of ~5e4. An attempt to escape that
by comparing the *ratio* e_post/e failed too, at 1.06e5 ulp: a ratio inherits its numerator's
cancellation. It is now bounded absolutely against the target scale, and its *property* — never
exceeding the a priori error — is asserted directly.

**Sixteen mutants, all caught**: transposed row-major access, a flipped θ layout, symmetrisation
removed, the forgetting factor dropped, the rank-1 term dropped, the trace accumulated in `T`
instead of double, `is_finite` accepting NaN, both rejection paths silently applying, the state
layout swapped to P-then-θ, and λ used where the reciprocal belongs.

### The Guard, across 2026-08-08

`cpp/include/argmind/guard.hpp`. `Guard<Adapter, Depth, MaxBounds>`, with the checkpoint ring
and the check capacity in the type so the memory is declared: `Guard<Rls<float,6,1>, 4, 4>` is
1208 bytes, of which 736 is the ring, reported by `checkpoint_bytes`.

**The structural difference is forced rather than chosen.** The Python holds a list of `Check`
subclasses and calls `evaluate` on each; that needs virtual dispatch and heap objects, and the
core has neither. So checks become concrete members with declared capacity — an array of `Bound`
plus one optional `Divergence` — and their identities become an enum rather than a string, which
is what a `tripped` bitmask can be built from. **The cost is that a user cannot supply their own
check type;** adding a kind of check means adding it here. That is the same trade the C ABI makes
for compile-time dimensions, made for the same reason: a function pointer per check would restore
the flexibility and give up the inlining the core exists for.

Because a guard *decides* rather than computes, parity here is about decisions, and a decision
that differs is not a rounding difference. Measured to match exactly, at both precisions, across
depths 1/2/4/10 and cadences 1 and 5: the freeze step, the thaw step, the whole frozen vector,
which steps were rolled back, the set of failing checks, and which checkpoint a trip restores.
Report scalars stay at the levels in the table above.

**Two tests were vacuous before a measurement said so, and both had one cause.** With `p0=100`
and six features `tr(P)` starts at 600, so a `p_trace` bound anywhere below that trips on the
first update — and a sweep over four bounds then compared step 0 against step 0 four times. A
swept axis that does not move the outcome is not evidence. The fixture is now a windup scenario
with excitation confined to one axis, where `tr(P)` grows as exactly `p0·λ^-k` and a bound of
`hi` is crossed at `k = ln(hi)/ln(1/λ)`: 45, 90 and 135 predicted against 44, 89 and 134
measured. The tests pin the predicted step, not merely that the two agree, so the fixture cannot
degrade back into `0 == 0` unnoticed.

**The same broken fixture produced a flattering result that had to be withdrawn.** With the trip
at step 0 the two implementations' restored states were bit-identical, which looked like proof
that a rollback copies rather than recomputes. It proved nothing: both had rolled back to the
*construction* state, identical by definition. At a real trip step they differ by about 41 ulp —
the recursion's own disagreement, carried into the checkpoint. Bit-identity was never available
here. What *is* verifiable is which checkpoint each side chose, and that is now tested by
comparing each implementation's restored state against **its own** unguarded trajectory, which is
bit-exact within one implementation and therefore identifies the step rather than merely being
consistent with it.

**A mutation check on the port found a bug in the reference.** `Guard.freeze()` took its health
scalars from the newest checkpoint. After a *trip* that is exact, because the state has just been
restored to that checkpoint — and `_probe`'s docstring justified it on exactly that ground. After a
*manual* freeze nothing was restored, so the value could be up to `checkpoint_every` updates stale:
measured at 300.0 reported against a live 104.1, three updates into a cadence of five. A `Bound` on
`p_trace` evaluated during a manual freeze therefore read a number describing a state the adapter
was no longer in. Both implementations now take the live values, which are exact in both paths, and
the docstring's over-broad justification is corrected.

That leaves the checkpoint ring's stored health scalars redundant — they are now only ever equal to
what `health()` returns for the state being restored. Removing them would drop the declared
checkpoint cost from `Depth * (state_size + 2)` to `Depth * state_size`, which is a change to a
figure quoted in several places. **Not done yet**, recorded here rather than half-done.

**Eighteen mutants against the C++ guard.** The first run left eight survivors, and the cause was
one thing: a parity harness can only test what the tool exercises, and there were no Guard tests in
the C++ unit suite at all. Several survivors were unreachable from the tool by construction — it
never builds an invalid guard, never calls `load_state` on one, never produces a non-finite report
scalar, and cannot trigger a manual freeze — and those became `cpp/tests/test_core.cpp` tests.

The four that belonged in the parity suite were then **designed against their mutants**, and the
first attempt at all four failed to discriminate, for four different reasons worth keeping:

- a second spike one sample after the first lands while the guard is still frozen, so it never
  becomes a second trip;
- a second fault 240 samples after a thaw arrives once the exponential averages have re-converged,
  making a missing reset invisible;
- bounds that a rejected report satisfies anyway make a rejection reaching the check path
  behaviourally identical — the observable that works is the *checkpoint* it stores, not the trip
  step;
- one clean sample in three is not enough for the fast average to fall below the threshold, so
  there are no healthy probes at all and nothing accumulates.

A test written without checking that the bug it is meant to catch would actually fail it is a guess.

**What is verified and what is not.** Execution was host-only when this was written; a cross-compiler and an emulator arrived on 2026-08-09 and the section below records what they found. There is still no target hardware, so: `rls.hpp` is checked to compile `-ffreestanding
-fno-exceptions -fno-rtti` and `argmind.h` to be valid C99, both on the host, which catches an
accidental `<cmath>` or a C++-only header the moment it appears. Code size on the host at `-Os` is
576 bytes of text for `Rls<float,6,1>::update` and 8468 for all four instantiations plus the ABI —
**a rough shape and explicitly not a target figure**, since it is x86-64. The flash and RAM numbers
for the declared floor must come from a link map, and **any on-target timing claim stays absent
rather than interpolated.**

**Now decided (see "Language and structure" above), so the `Adapter` contract must be shaped for
a C ABI from the start:** fixed dimensions known at construction, no allocation after it, state
addressable as a flat block so it can be persisted or crossed over a C boundary without
serialisation, and no Python-only conveniences in the core interface.

**Before any of it is called fast:** the benchmark harness. It is listed after the adapters only
because there must be something to measure — not because optimisation may start without it.

### Persistence, across 2026-08-08

`cpp/include/argmind/persist.hpp` and `cpp/tools/pcheck.cpp`. This is the piece the format
was designed for: the Python writes files but models the raw-flash scheme faithfully because
*this* side has no `os.replace`.

**Scope is deliberately narrow.** The header does the format — pack, validate, unpack,
arbitrate between two slots — and does **not** write to flash. Erase granularity, write
width, wear levelling and the driver API are all platform-specific, and a library that
guessed at them would be wrong everywhere.

**Parity here is byte identity, not a tolerance**, because serialisation has no rounding.
The suite runs the sweeps rather than sampling: **840 pack configurations** (two precisions
× four shapes × seven sequence values including both ends of the range × five fingerprints ×
three cause lengths), **every single-bit flip in a 78-byte record** (624 of them), **every
truncation length** (78), and **400 seeded differential fuzz blobs**. Zero byte differences,
zero status disagreements.

Those numbers were quoted here once from an ephemeral sub-agent run — "1008 configurations,
80,000 fuzz blobs" — against a harness that no longer existed, while the suite itself did
eight configurations, one bit flip and two truncations. An audit caught it. **The sweeps are
now checked in, so the figures above are reproducible by running the tests**, which was the
only honest way to keep making the claim. Batching them through the parity tool rather than
one process per case took the file from 53 seconds to under 4. The 4 is reproducible by
running the file; the 53 was measured here against the unbatched version, which is not in the
tree — a before-figure rather than something anyone can re-check.

The bit-flip sweep pins *check ordering* as well as agreement: version, dtype and length are
all checked before the CRC on both sides, so a flip in the version field reports
`future_version` even though it has broken the CRC too. That ordering needed its own test —
a mutant moving the C++ version check after the CRC passed the entire suite, because every
test that touched versioning repaired the CRC first, which is exactly what removes the
ordering from what they test.

**Three things byte identity depends on.** The header is 26 unpadded bytes with
`fingerprint` at offset 18 and `count` at 22, neither four-byte aligned, so nothing may be
pointer-cast onto the buffer. The CRC is zlib's *reflected* IEEE 802.3, not the same
polynomial unreflected — pinned by the standard check vector for `"123456789"`, 0xCBF43926.
And element bytes are little-endian by declaration rather than by inheriting the host's;
that last one **cannot be distinguished by comparing output on an x86 host**, so what the
test pins is that a known value appears in the known order, and the limitation is stated in
the test rather than implied away.

**Reviewing this port found two bugs that neither the tests nor I had caught**, and both
were outside the new header:

- **The reference destroyed the record it had just loaded.** `Store.__init__` starts at slot
  0 and `load()` updated only the sequence number, so after any *odd* number of writes in the
  previous session the newest record sat in slot 0 and the first write after a reboot landed
  on top of it. Measured: three saves leave `{slot 0: 3, slot 1: 2}` and the next save
  destroys 3. That is exactly the failure two slots exist to prevent — the promise is that an
  interrupted write destroys at most the copy *not* in use. `load()` now targets the other
  slot and `live_slot` reports which one is live.

  Fixing `load()` turned out to be half a fix, and an audit found the other half: a `Store`
  that *writes before it reads* knows nothing about what is on disk either. Measured — three
  saves leave `{0: 3, 1: 2}`, and a fresh `Store` calling `save()` without `load()` leaves
  `{0: 1, 1: 2}`. The newest record is destroyed **and the sequence counter regresses**, so
  from then on the older record wins arbitration and the device silently reverts a
  generation. `save()` now arbitrates once if nothing has been loaded.
- **A 32-bit overflow that cannot be reproduced on the host.** `count` is read straight out
  of the record and `size_t` is 32 bits on the M4F floor, so `count * sizeof(T)` can wrap and
  make a forged length look correct. The CRC still catches it in every case anyone could
  construct — but that made the safety depend on an unstated invariant between two
  independent checks. **This is the project's rule that host figures are not target figures,
  applied to behaviour rather than to timing.** The guard added for `unpack` left the
  identical multiplication in `pack` unguarded, which a later audit found: on a 32-bit target
  `pack(count = 0x20000000)` at `T=double` wraps the total to 30 bytes, passes the
  buffer-size check and then writes half a gigabyte past the caller's buffer. Both
  directions are guarded now. Guarding one and not the other was an omission, not a
  decision.

**And two ways a suite can look complete while testing nothing.** A mutation check found
`select_newest` was compiled but never called by any binary the suite built — the
arbitration test decoded each slot separately and compared against the Python's own,
separate, arbitration logic. And a bounds check added to the parity *tool* had made the
library's own clamp unreachable, so a mutant removing it survived. Both are closed; the tool
gained a mode that calls `select_newest` directly, and it now hands `pack` the true length
and lets the library clamp.

### Streams and alignment, across 2026-08-08

`cpp/include/argmind/stream.hpp` and `align.hpp`. `Stream<T, Dim, Capacity>` is a ring whose
memory is declared by its type; `Aligner` carries the four policies.

**The structural decision, challenged in review and upheld.** An aligner has to see streams
of *different* widths at once. A base class with a virtual `sample_at` is the obvious answer
and this core has no virtuals; a variadic `Aligner<Streams...>` keeps every dimension at
compile time and cannot be named from a C ABI, which is the same wall the ABI's explicit
instantiation list already exists to avoid. So `Stream` keeps `Dim` and `Capacity` in the
type — its storage and its push loop stay compile-time sized — and hands out a **`StreamView`
with a runtime `dim`** to anything that must treat several streams uniformly. Review
confirmed `dim` only ever drives an O(dim) copy and pointer arithmetic, never an O(n²)
accumulation: the compile-time shape stays where it buys inlining.

**A third reference bug, found by one disagreeing push.** `Stream.push`'s ordering check
cannot catch a NaN: `NaN <= last` is False, so the sample was accepted, `_last_t` became NaN,
and every later comparison was False too. Monotonicity was gone for the life of the stream —
observed leaving a ring holding `[1.0, 1.5, nan, 0.1]`. Every alignment policy scans that ring
assuming order, so one NaN can make hold_last, nearest and interpolate all return the wrong
sample. Non-finite timestamps now have their own check on both sides — though for different reasons,
which is worth not blurring. The C++ ordering test is `!(t > last_t_)`, which already refuses
NaN and -inf; its finiteness check exists for **+inf** alone, which passes the ordering test
and then makes every later push impossible. The Python's `t <= last` was the comparison that
let a NaN through. A test demonstrates the hazard by corrupting a ring directly rather than
asserting it in the abstract.

**Parity here is decisions rather than arithmetic**, and it holds. The suite's own sweeps are
checked in, so the numbers below come from running the tests: four streams of differing widths
and rates over 300 instants per policy, compared **bit-exactly** — staleness by bit pattern,
values by raw bytes — so a signed zero or a NaN payload shows. A `pytest.approx` comparator was
there first and would not have.

The negative control needs stating carefully. The three *distinguishable* policies return
different samples at **292 of 300** instants, so a port that collapsed two of them cannot pass.
`drop_if_stale` is excluded from that count deliberately: it shares a branch with `hold_last`
in `pick()` and never consults the staleness limit, so counting it as a fourth distinct policy
inflated the figure. An earlier version of this paragraph said "four policies … 872 of 942" and
was wrong on both halves. What actually distinguishes `drop_if_stale` is the stale-drop
decision, and until an audit pointed it out that decision was compared nowhere. It is now
checked on every pick.

A one-off run against scratchpad scripts covered far more — 30,144 picks, 96,160 blends, and a
soak after roughly 915 overwrites per stream, which is about 14 wraps of a 64-deep ring rather
than 915 wraps — and found zero disagreements. Those scripts are not in the tree, so that is a
reported result rather than one anyone can re-run here, and it is labelled rather than quoted
as if the suite produced it.

**The one arithmetic detail, and its precondition.** Interpolation blends in float64 and
narrows back even for an f32 stream, because the reference does. A checked-in test counts how
many of the weights it sweeps would land on a different float32 under an f32 blend and asserts
that count is large enough for the comparison to have grip, so the agreement is a measurement
rather than a formality. (A one-off run over a much wider sweep put that at 18,342; the
checked-in version is smaller and reproducible.) It also depends on the blend *not* being contracted into
an FMA: with `-march=x86-64-v3 -ffp-contract=fast`, 1506 of 9995 f64 blends differ by one ulp
of the bracket's scale — one ulp of `max(|a|,|b|)` exactly, confirmed separately. The M4F's
double path is soft float and so cannot contract; that is reasoning from the declared flags
rather than a measurement, since no cross-compiler exists here. An M7 or a host `-march=native`
build can, and the answer there is `-ffp-contract=off` rather than a tolerance.

**And the cost that paragraph was quietly not mentioning.** The declared floor has a *single*-
precision FPU, so widening to double means every interpolated pick runs a double divide plus
two double multiplies and an add per element **in software** on an M4F. It is the most
expensive thing in that header on the target, it is the price of matching the reference's
arithmetic, and it sits on the precision axis the target decision was made on — exactly the
kind of cost that looks free because nothing in the model expresses it.

**Two review findings worth keeping.** The dimension cap on interpolation's blend buffer used
to sit ahead of the policy dispatch, refusing wide streams under `HoldLast` and `Nearest` as
well — for a buffer those policies never touch — and its refusal was indistinguishable from
"no sample", so a wiring mistake read as a dead sensor. And `dropped` was a `uint32_t`, which
wraps in 49.7 days at 1 kHz — arithmetic, not a measurement. `dropped + size()` is the identity
the Python pipeline's lost-frame count is built on, so an identity that stops holding is not one
a C++ pipeline should be built on either; the Python's own counter is an unbounded int and
cannot wrap, so the hazard is for the port rather than for today. The widening turned out to
cost **nothing** — `sizeof(Stream)` is unchanged, because padding before the 8-aligned timestamp
absorbs it. An earlier version of this sentence claimed four bytes.

### The pipeline, across 2026-08-09 — the last piece of the core

`cpp/include/argmind/pipeline.hpp`. `Pipeline<Adapter, MaxStreams>` plus `FrameMap<N, M>`,
with a fourth parity tool (`cpp/tools/pipecheck.cpp`) and 23 tests in
`tests/test_cpp_pipeline_parity.py`.

**What has to agree here is bookkeeping, not arithmetic.** `xcheck` settled that RLS and the
guard compute the same numbers and `scheck` that the aligner picks the same samples. What is
new at this level is that a pipeline is driven by two independent clocks — samples arriving,
and `step()` being called — and every quantity it reports is a difference between counters
taken at different moments. So the job file is an **event log**: `push` and `step` interleave
in whatever order a case wants, which is the only way to reach a ring that overflowed between
two steps, a step with nothing new, or a guard that froze half-way through one.

Everything except the model state is compared **exactly** — integer counters, and doubles by
bit pattern so a signed zero shows. Across the checked-in suite plus a 120-case randomised
sweep (policies, both precisions, capacities 4/8/64, guard depths, dims 1–3, adversarial
interleavings) there were **zero disagreements** in any counter, flag, staleness figure or
reference time. The state block is the one place a tolerance belongs, since it is RLS
arithmetic: worst measured deviation **149.8 ulp** against the 400 the suite allows, which is
the headroom `rls.hpp`'s three declared divergences need and not much more.

**Three structural differences, all forced rather than chosen.**

- **Frames are never materialised.** The Python builds a list of `Frame` objects in
  `Aligner.drain()` and loops over it; a list needs a heap and a `Frame` needs a dict of
  arrays. So the two are fused: one reference instant is aligned straight into the
  preallocated `x`/`y` and updated, then the next. Observationally identical as long as
  nothing pushes mid-step — alignment does not mutate a stream and neither does an update —
  and it is the difference between one `Pick` alive at a time and `MaxStreams` of them: for
  `Pick<double>`, 280 bytes against 1,120 at four streams, on a target budgeting 8 KB.
- **Views are passed to `step()`, never stored.** A `StreamView` snapshots `head` and `count`.
  A pipeline that captured one at construction would align against a picture of the ring
  frozen at that moment forever — a bug that looks like a sensor that stopped.
- **Every stream in one pipeline shares a scalar type.** The Python allows an f8 stream and an
  f4 stream in the same frame; here the aligner is `pick<T>` and the array is
  `StreamView<T>*`. A mixture needs a tagged union with a switch on every element copy, or a
  variadic pipeline type a C ABI cannot name. This is a real reduction in what the port
  accepts, and the parity cases therefore only ever use matched dtypes — said plainly so
  "parity holds" is not read as covering a case nobody ran.

**The cost of writing into `x` before the frame is known to exist.** A frame that fails
half-way leaves `x` holding some of this instant's streams and some of the last one's.
Predicting from that produces a plausible wrong answer and no error, which is the failure
shape this project keeps finding. `frame_ready()` is false after a partial write and
`predict_last()` refuses; a unit test drives the case where the reference stream answers and
a later one does not, because that is the only ordering in which the mixture actually occurs.

**One difference from `align.hpp` deliberately not repeated.** `Aligner` folds "no limit" and
"a limit of zero" into one non-positive sentinel, so a caller passing 0 meaning zero-tolerance
silently gets no check — recorded there as a real divergence. The pipeline's own input gate
keeps a separate flag, so `None` and `0.0` stay distinguishable and a zero limit is *refused*
by `configure()` exactly as the reference's constructor raises on it.

**`StreamView` gained a `dropped` field**, because `dropped + count` is the total ever pushed
and that identity is the only honest basis for a lost-frame count. Without it the pipeline
would need the concrete `Stream<T, Dim, Capacity>` — the type `StreamView` exists to avoid
naming. Measured cost: `sizeof(StreamView<float>)` 40 to 48 bytes; `sizeof(Stream<float,6,64>)`
unchanged at 2136.

**Declared memory, host figures on x86-64 and not target figures.**
`Pipeline<Guard<Rls<float,6,1>,4,4>, 4>` is 1616 bytes, of which the guard is 1208 and the
pipeline's own parts 408 — the `Pick<float>` scratch buffer is 152 of those. `pick_bytes` is
`sizeof(Pick<T>)` rather than a hand-summed payload, and a unit check pins it against `sizeof`:
`stream.hpp` shipped a constant that added up the payload, told callers to budget from it, and
was 24 bytes short of the object.

**Seventeen mutants, sixteen caught, and the survivor is the finding.** Caught: the drain
watermark re-offering the boundary instant; the watermark advancing only on frames that were
built; the staleness gate turned inclusive; worst staleness taking the smallest; the constant
term zeroed; a sensor's invalid flag ignored; the arrival watermark frozen; `frozen` never
reported; the validity gate dropped; a declined instant uncounted; the feature scatter
transposed; the target scatter ignoring the map; a view whose width changed accepted;
`configure` accepting a component past the end of its sensor; a partial frame still looking
ready; and `StreamView::dropped` reporting zero.

The survivor is the **`lost_frames` clamp**, and it survives because it is dead. Each visited
instant increments exactly one of frames/no_data/dropped_stale, so `considered` is the number
of instants visited; every unconsumed reference sample in the ring arrived after the last
successful step; therefore `considered <= owed` by construction. The reference's `max(0, ...)`
is dead for the same reason. It stays on both sides because the C++ subtraction is unsigned and
a wrap would report about 1.8e19 lost frames rather than a small wrong number — but it is
recorded as unreachable rather than described as tested.

**Two tests written to kill that mutant failed to, and that is the more useful lesson.** The
first advanced the aligner from outside between steps, on the theory that external counter
movement would reach the clamp. It does not: the deltas are sampled *inside* `step()`, so
anything that happened between calls is invisible to them. The second — asserting the figure
survives such interference — turned out to assert nothing the clamp was needed for. Both were
guesses, and the rule that has now failed twice in this project is the same one: **a test
written without checking that its bug would fail it is a guess.**

Chasing that survivor found a real coverage gap next door, and this one was an eighteenth
mutant built afterwards rather than part of the scripted seventeen. Baselining `before_no_data`
at zero — using the aligner's *absolute* counters instead of the step's delta — survived the
entire suite, because no case had both declined instants and an overflowing ring in one run.
With one added (four instants declined while s1 is silent, then four framed while six are
overwritten) it is caught: a cumulative `considered` would report two lost in the second phase
instead of six.

**`SkipReason::Rejected` is unreachable** with the adapters this core ships, and so is the
reference's `or "rejected"` fallback: `Rls::update` refuses only for non-finite input or a
non-positive denominator, and `Guard` either passes one of those through or sets
`guard_reason`. It exists for a future adapter. There is no test asserting it stays at zero,
because that assertion could not fail.

**One reduction in what the suite covers, stated rather than left implied.** Two sub-agent runs
— an embedded review of the header and an independent parity verification — were launched and
both terminated on an API spend limit before reporting. Everything above was measured directly
instead. What that costs is the independence: the mutation list, the sweep and the memory
figures were all designed by the same hand that wrote the code, and this project's own history
says that is exactly the arrangement under which a flattering result survives. The two reviews
are worth re-running before the next component lands.

### The C ABI grows a guard and the record format, 2026-08-09

`cpp/include/argmind/argmind.h` and `cpp/src/abi.cpp`.

**This started as an inconsistency, not a feature request.** The rule this project designed
the guard around is that a `Guard` *is* an `Adapter`, deliberately, so that "no guard" is not
expressible — a safety mechanism that can be left out gets left out. The C ABI exported bare
`Rls` and nothing else, so from C the guard was not merely optional, it was **unreachable**.
The boundary contradicted the principle it was supposed to carry. The same argument, weaker
but real, applies to persistence: `am_rls_*_state()` handed out the block a device has to
survive a power cut with, and there was no way to frame it from C, so the state block
dead-ended at the boundary.

**What was added.** `am_guard_*` over four instantiations matching the RLS list, and
`am_persist_*` over f32 and f64. `Depth` and `MaxBounds` are fixed at 4 in the ABI rather
than exported per value: they are the memory, and taking them per call would put them in the
type and back into the instantiation list. `checkpoint_every` stays a runtime argument, and
it is the tunable that matters — rollback reaches `depth * checkpoint_every` updates back.
The record format needs no instantiation list at all; everything but the scalar type is
already a runtime argument there.

**Declared memory, measured on x86-64 — a shape, not a target figure, and AAPCS will differ.**

| instantiation | sizeof | of which the checkpoint ring |
|---|---|---|
| `am_guard_f32_6_1` | 1208 | 736 |
| `am_guard_f32_6_3` | 1448 | 928 |
| `am_guard_f32_12_1` | 3488 | 2560 |
| `am_guard_f64_6_1` | 2064 | 1408 |

The ring is the majority at every shape, which is the thing a caller weighing rollback depth
needs to see and the reason `am_guard_*_checkpoint_bytes()` reports it separately rather than
leaving it to be inferred. **`f32_12_1` is worth looking at twice: 3488 bytes is 43% of the
declared 8 KB RAM budget for one guarded adapter.** That is arithmetic on the declared budget,
not a measurement of a real firmware, and it is the kind of number that should be seen before
a target build rather than after.

**The `MAX_BYTES` static assertion earned its keep immediately.** The four bounds were written
into the header before anything was measured, and `AM_GUARD_F32_12_1_MAX_BYTES` was 3328
against a real 3488. The build failed with the macro named. That is exactly the failure the
assertion exists for — a compile-time promise a C header makes without being able to compute
the C++ layout — and it is worth recording that the first draft was wrong, because the
temptation with a bound is to write a comfortable number and move on. The bounds now carry a
small margin over measured sizes, and a test asserts each is within 2x of the real size:
`sizeof <= 99999` is true forever while telling a caller to reserve 100 KB.

**The C99 gate was weaker than it read.** `test_the_c_header_is_valid_c99` compiles the header
as C and stops, which says nothing about whether the entry points are callable and linkable —
a header can parse perfectly while every symbol is mangled out of reach. There is now a test
that compiles a C99 translation unit with **gcc**, links it against the C++ core, and runs it:
it configures a guard, drives 80 updates through a fault, checks the trip, packs the state,
unpacks it and compares. The claim "a plain-C firmware can call in" is exercised rather than
asserted, and the test goes end to end because each half working says nothing about the two
meeting.

**Seventeen mutants on the shim, sixteen caught, one equivalent.** Caught: a field code
outside the enum cast in anyway; an unusable guard reported as usable; `add_bound` accepting a
check that can never fail; a divergence whose "fast" average is the slower of the two; the
report dropping `rolled_back` or the `tripped` mask; the a posteriori residual reported as the
a priori one; `init` accepting storage smaller than the object; a dropped counter;
`load_state` accepting a non-finite block; `unpack` reporting `Ok` whatever the status;
`read_state` ignoring a length mismatch; `select_newest` always naming slot 0; the record's
frozen flag or sequence number dropped; and `next_slot` writing over the newest record.

The survivor is **equivalent, not untested**, and the distinction is the point.
`am_guard_*_state(nullptr)` tests for null before dereferencing; removing that test survives
everything, because — measured — `state()` sits at offset 0 of the guard: `state_` is the
first member of `Rls` and `Rls` is the first member of `Guard`, so the mutant returns null for
a null argument too. No test can separate them. The check stays because dereferencing a null
pointer is undefined whatever the address arithmetic happens to produce, and because that
offset is a layout accident rather than a guarantee.

**One API constraint found by a test that failed, and it bites on real flash.**
`persist::unpack` compares the length it is given against the one the header describes
*exactly*, so a caller must pass the **record's** length, not the capacity of the buffer it
was read into. That refusal is the right one — a torn write looks exactly like a short record
— but it means a slot rounded up to an erase page cannot be handed in whole. The header now
says so and says what to do instead (keep the length beside the slot, or re-derive it with
`am_persist_record_bytes`).

**What a C caller could not do at this point**, and the reason, kept because the next
section is the answer to it: there were no `am_stream_*`, `am_align_*` or `am_pipeline_*`
entry points. `Pipeline::step()` takes an array of `StreamView<T>`, a C++ template, so
exposing it needs a C mirror of that struct plus a stream ABI — and `Stream<T, Dim, Capacity>`
puts two more axes into an instantiation list that would then multiply against the adapter's.
The shape that avoids the explosion is one composed handle per wiring, and that was a design
decision worth making deliberately rather than tacking onto a list.

**A note on the C++ check count.** It moved from 868 to 3351 across the two ABI changes, and
most of that growth is two loops rather than two thousand distinct properties: the
ABI-versus-header comparisons run their fixtures for 120 and 20 iterations and make ten to
fifteen checks on each. The number is a floor to catch checks silently disappearing, not a
measure of coverage, and reading it as the latter would be reading a loop bound.

### The composed handle: streams and the pipeline reach C, 2026-08-09

The previous ABI section ended by saying a C caller could not reach the pipeline, and why:
`Pipeline::step()` takes an array of `StreamView<T>`, and `Stream<T, Dim, Capacity>` would put
two more axes into an instantiation list that then multiplies against the adapter's.

**The resolution is to move those two axes out of the type entirely: the caller owns the
sample storage.** `am_ring_f32` is a plain C descriptor over buffers the caller declares, so
`dim` and `capacity` are runtime values on that side and nothing about a stream's shape
reaches the instantiation list. What stays in the type is the adapter — which is where the
compile-time shape was earning its keep all along, in the arithmetic — plus `AM_MAX_STREAMS`.
That is the same trade `StreamView` already makes inside the core, made once more at the
boundary.

**One push, not two, and getting there took two attempts and a measurement.** A C ring and a
`Stream<T, Dim, Capacity>` would otherwise hold two copies of the monotonicity and finiteness
refusals, and `persist.hpp` already argues that a refusal implemented in two places is one
that will eventually disagree with itself. So `push` moved into a shared `ring_push` that both
call.

The first version took a `Ring<T>*`, which meant `Stream::push` had to materialise one and
write four fields back: **+32 bytes of code** per instantiation at `-Os` on x86-64, 160 to 192
for `Stream<float,6,64>::push`. Passing the members as separate pointers brought it to 176 —
still +16, because the rewritten wrap used a compare-and-branch where the original used
`% Capacity`, which a power-of-two capacity folds to a single AND. Restoring the modulo lands
back on exactly 160. The comment claiming the extraction was free was written before any of
this was measured and was wrong twice before it was right.

That modulo is a division where `slot()` twelve lines away deliberately avoids one, and the
two are not in conflict: this runs **once per push** and that ran twice per element of every
scan. The C ring, whose capacity is a runtime value, pays one integer division per push — a
single `SDIV` at the sensor's rate on the declared floor, which is not worth making the
compile-time path larger to avoid.

**Declared memory, measured on x86-64 — a shape, not a target figure.** Bare `f32_6_1` 616
bytes, `f64_6_1` 952. Guarded: `g_f32_6_1` 1632, `g_f32_6_3` 1896, `g_f32_12_1` 3952,
`g_f64_6_1` 2640. A guarded pipeline is its guard plus **424 to 576 bytes** — 424 / 448 / 464
/ 576 across the four — and the guard is mostly its checkpoint ring.

These figures were 16 to 32 bytes smaller when first written and an audit caught the drift:
adding `require_fresh_target` put three members into `Pipeline`, and nothing pinned the
numbers. `cpp/tests/test_core.cpp` only checked them against their `MAX_BYTES` bounds, which a
16-byte growth passes comfortably. "About 400 bytes" was also wrong at f64 by 44%. Six instantiations is a lot of code for a caller who wants one;
`-ffunction-sections -fdata-sections` with `--gc-sections`, which the embedded CMake path
already sets, keeps the unused ones out of flash.

**`map_mutable()` clears the configuration, and that is the point rather than a side effect.**
A C caller cannot construct a `FrameMap<N, M>` and hand it to a constructor, so it builds one
entry at a time after `init`. The map is exactly what `configure()` bounds-checked against the
streams; leaving the pipeline configured through a mutation would let a caller widen a
component past a sensor's width after the check passed, and the next `step()` would read off
the end of a ring. `step()` returns `AM_ERR_UNCONFIGURED` until `configure()` runs again.

**Sixteen mutants on the new surface, and two survived long enough to be findings.**

The first is a **real defect the C path had and the C++ path did not**. `am_ring_*_init`
setting `last_t` to zero rather than `-inf` survived the whole suite — and it would silently
refuse the first sample of any device whose timebase is milliseconds since boot, which is most
of them. `Stream<>::reset` had this right; nothing compared the two on a *fresh* ring, only on
one that had already accepted a sample. A test now pushes `t = 0` and a negative `t` into both
and demands they agree.

The second is quieter and worth naming as a category: **zeroing `lost_frames` in the report
survived**, because no case on this side of the boundary ever overflowed a reference ring.
Every other field had a non-zero value somewhere and that one did not, so it was present,
compared, and compared against zero. It is also the field that says input never reached the
learner, which makes it a poor one to lose. Four-deep rings and ten arrivals between steps now
lose thirty frames across five steps, and the mutant fails.

The rest were caught: an equal or `+inf` timestamp accepted; overwrites uncounted; the ring
wrapping a slot early; a mutated map leaving the pipeline configured; a policy code outside
the enum cast in anyway; `step()` running unconfigured; a ring with null buffers accepted;
staleness, the skip counters or `frozen` dropped; `updated` reported as `frames`; and
`configure()`'s failure reported as success. One of them — accepting more streams than
`AM_MAX_STREAMS` — was caught by the test binary dying rather than by an assertion, which is
what a stack array overrun looks like and is the reason the bound is there.

**The C caller test grew a second translation unit** that drives the whole composed handle:
three caller-owned rings of different widths, a guarded six-feature adapter, an index frame
map built one entry at a time, sixty pushes with a fault at the thirty-fifth, and a check that
the guard froze and blocked the rest. That is the shape an integrator actually wires up, and
none of it was reachable from C before this change.

**A fixture that agreed while testing nothing, caught by three failing assertions at the
end.** The first version of the pipeline ABI comparison mapped `(0, 0)` — the reference
stream's own value, which is the *target* — into the regressor. The model read the answer off
its own input, the residual sat at zero, and the fault injected to trip the guard did nothing.
Everything else matched perfectly. The fixture is now three streams with the target outside
the regressor, and the features interleaved across the two sensors so a transposed scatter
would show.

### Low latency against honest weighting, 2026-08-09

A requirement arrived — inputs at whatever rate each source runs, taken as fast and with as
little latency as possible, and an output that is also fast — and it turned out to pull
against something the framework already did. Both halves are measured here rather than
argued.

**What each alignment policy costs in latency.** Reference = the fastest stream, 10 Hz and
1 Hz, 200 samples, drained on every arrival:

**The fixture, because it decides the numbers**: reference is the fast stream, 200 fast
samples at 10 Hz, 20 slow ones at 1 Hz, and `drain` runs **after every arrival** — which is
what a low-latency caller does.

| policy | limit | frames | max added latency | abandoned |
|---|---|---|---|---|
| `hold_last` | — | 190/200 | 0.0 ms | 0 |
| `nearest` | — | 190/200 | 0.0 ms | 0 |
| `interpolate` | — | **0/200** | — | 0 |
| `drop_if_stale` | 0.05 s | 0/200 | — | 190 |
| `drop_if_stale` | 0.5 s | 95/200 | 0.0 ms | 95 |
| `drop_if_stale` | 2.0 s | 190/200 | 0.0 ms | 0 |

Three things an audit had to correct here. The `drop_if_stale` row originally appeared without
its staleness limit, and the only limit that reproduces 190/200 is one large enough that the
policy never abandons anything — **a row that looked free because the axis that costs was
switched off**. The `interpolate` row was measured at one phase of the slow stream and the
phase was not stated; checked at two, it is 0 either way, but that was luck rather than
knowledge. And "0 ms" sat in the same column as the counts as though it were the same kind of
measurement: it is measured, but for a causal policy it is close to tautological — the harness
times how long after a reference sample its frame becomes buildable, and a policy that never
looks ahead answers immediately by construction. It confirms rather than discovers.

The ten instants that cannot be framed are the nine before the slow stream's first sample plus
`t = 1.0` itself, whose drain runs before the slow sample of the same timestamp arrives.

**190 here and 191 elsewhere in this document are different fixtures, not a contradiction.**
`test_pipeline.py`'s `_rates` pushes everything and calls `step()` once at the end, so the
frame at `t = 1.0` is built after the slow sample exists. The drain cadence is the difference,
and printing the two counts in the same section without saying so is what made it look like
one of them was wrong. Every row above is now pinned by
`tests/test_stream_and_align.py`.

Two rows were not what the docstrings imply. **`nearest` degenerates to `hold_last`** in a
streaming loop with the fastest stream as the reference — same sample, same staleness, on
every one of the 190 frames — because the future sample it exists to consult has not arrived
yet. It buys nothing and costs a non-causal flag. And **`interpolate` does not merely add
latency, it produces nothing**: bracketing needs a sample later than the reference instant,
there is never one, and `drain` consumes the instant anyway so it is not retried. That is
consistent with what the design already says — non-causal means offline — but the magnitude
is worth writing down.

**So the answer to the requirement is `hold_last` with the fastest stream as the reference,
and there is no latency to trade away.** `step()` is pull-based and waits for nothing; output
latency is the caller's call cadence, which is the caller's to choose.

**The cost that arrangement hides.** When the target rides a slower stream, hold-last hands
the same measurement to the learner once per fast frame: measured, 191 frames from 20
measurements, each used up to ten times. RLS weights it ten times, so `tr(P)` ends an order
of magnitude below where the same information should leave it: measured **0.0185 against
0.196** in the reference. The C++ side
shows **12.1×** on a fully excited five-wide regressor — a figure taken from a scratch
programme rather than from the suite, which asserts only the weaker `> 5×` it can hold across
compilers. Said that way because an audit pointed out the 12.1 had no fixture behind it in the
tree.

**`max_stale_s` looks like the lever and is not.** It is a threshold on time where the
question is one of identity:

| jitter | gate | adaptations (20 measurements) |
|---|---|---|
| none | 0.05 s | 20 ✓ |
| ±20 ms | 0.05 s | **12** — eight measurements lost |
| ±20 ms | 0.15 s | **31** — some used twice |
| ±20 ms | 0.10 s | 20 ✓ |
| none | 0.10 s | **32** |

The last two rows are the argument. It is not that every threshold is bad — 0.10 s is exactly
right at ±20 ms — it is that the *same* threshold is wrong without the jitter. A value tuned
on a bench recording is wrong in the field and the reverse, so there is no number a caller
could have picked.

**`require_fresh_target` needs no number.** It compares the target sample's own timestamp,
recovered from the signed staleness the aligner already reports (`t_sample = t_reference +
staleness`), against the last one adapted on. Twenty measurements, twenty adaptations, with
or without jitter. Off by default, so nothing that existed before changes.

Three edges, all tested. Under `interpolate` the target is synthesised at the reference
instant and really is new every frame, so the gate never fires. If the target rides the
reference stream every frame carries a new target, so the gate costs nothing and does
nothing. And a first target sample at **exactly t = 0** must not read as stale — zero is the
first tick of any device counting milliseconds since boot. The two implementations
distinguish "no target yet" from "the target is 0.0" differently: the reference uses a `None`
sentinel, the core a `have_last_target_` flag, because it has no optional.

**`StepReport` now carries the output**, and that is the other half of the requirement.
Present whenever a frame existed, **including one the gate refused to learn from** — refusing
to *learn* from a sample is not a reason to stop *predicting* on it, and separating the two
is exactly what lets a fast output run on a slow target. It costs one `predict` per step,
`O(n·m)`, six multiply-adds at the design driver's shape. Extraction moved ahead of the input
gates in the reference so a gated frame still has a regressor to predict from; the port
already did it in that order, and the two were unobservably different until `prediction` gave
`x` a reader.

The reference's array is preallocated and handed out read-only. A fresh array per step is an
allocation per step, which `bench.memory_growth` exists to notice and which `Adapter.state`
was already fixed for once; a test measures `sys.getallocatedblocks` across 500 steps rather
than trusting the reasoning.

**Comparing the output across the boundary took three attempts at the metric**, and each
failure was the same lesson in a new place. Bits fail: a prediction is `x'θ` and θ already
carries `rls.hpp`'s three declared divergences, so the f32 case differs on the tenth frame.
Element-wise relative error fails too, at **217 ulp at f64**, on a step where the prediction
happens to be near zero — a prediction is a sum of signed terms and can cancel. Against the
run's own scale that same step is **14.0 ulp**. This is now the third quantity in the project
to need that treatment, after the state block and the a posteriori residual, and the rule is
worth stating plainly: **whenever a quantity can pass through zero, a relative comparison
measures the cancellation rather than the implementations.**

One more property fell out of it: a frame carrying a NaN feature produces a **NaN output on
both sides**. The adapter refuses to learn from it, but handing back a number would be worse
— a stale value presented as this frame's answer is a wrong answer, and a NaN is a thing a
caller can test for.

**Fifteen mutants, thirteen caught, and both survivors were worth the run.** One was
*equivalent* rather than untested: `target_is_new()` initialised its flag to
`!have_last_target_` when the loop's own condition already covered the first frame. That was
removed rather than recorded — the shorter form has no dead branch to explain — and the guard
that does the work, inside the loop, is caught by the `t = 0` test. The other was a real hole:
zeroing the prediction in the ABI shim survived, because every test compared
`has_prediction` and nothing compared what it pointed at.

### What `step()` costs, and where the cost sits, 2026-08-09

The benchmark already timed `step()` — `frame_ns` in the fixed workload — but only in the
*slow-reference* wiring, where a step drains one frame because the reference is the aiding
sensor. The low-latency wiring inverts that: the fastest stream is the reference, `step()` is
called after every one of its samples, and each call pays the call's fixed overhead in full.
Whether that overhead matters is a property of the implementation and it is not visible in a
single per-step figure.

**So `step_cost` sweeps the cost against frames drained and fits `fixed + per_frame × frames`.**
The technique is `fit_leak`'s, reused on purpose: a cost per frame at one batch size cannot
tell fixed overhead from real per-frame work, and only the slope separates them. That
distinction already earned a correction once in this file.

Measured on this host — **absolute nanoseconds are machine-specific and are not a gate**;
what travels is the decomposition:

| frames/step | ns/step | ns/frame |
|---|---|---|
| 1 | 50,600 | 50,600 |
| 2 | 95,500 | 47,750 |
| 4 | 184,300 | 46,075 |
| 8 | 334,000 | 41,750 |
| 16 | 676,900 | 42,306 |

Fit: **≈11.4 µs fixed + ≈41.4 µs per frame** on that run, so about a quarter of a one-frame
step is overhead rather than work.

**The stability this paragraph originally claimed was not real, and the way it was wrong is
worth keeping.** It said the slope held to 0.5% across runs, citing three consecutive
measurements of 41065 / 40874 / 41028 ns. Over ten interleaved runs the slope moves **10.9%
about its mean** (32364 to 36043 ns) and the overhead fraction ranges from **19.9% to 37.2%**.
Three consecutive runs of those ten agree to 2.6% — a drifting mean impersonating a stable
one, and exactly what sampling three times in a row measures. An audit ran thirteen and found
a 22% range.

So the honest statement is qualitative: overhead is somewhere between a fifth and a third of a
one-frame step on this machine, and any figure taken from `step_cost` needs its spread beside
it. The one number that does travel is a ratio measured inside a single run, which is what
`gate_saving` reports.

**Which is why `fit_step` returns the intercept but nothing here compares two of them.** An
early version of `gate_saving` reported the ratio of two intercepts as a finding; it was
mostly the conditioning of a five-point fit. It also fitted a line through a **single** point
when the sweep had one entry, which numpy does with a warning nobody reads. A fabricated
slope in a benchmark is worse than no slope, because it gets quoted — `fit_step` now returns
NaN for a sweep that cannot determine a line, and a test pins that.

**What `require_fresh_target` costs, measured in one run so the machine cancels out.**

- Where it can help — a multi-frame step whose frames share one target — a gated frame costs
  **×0.22** of an adapting one (0.22 / 0.24 / 0.21 over three runs). It skips the update,
  which is nearly all of the work.
- Where it cannot — one frame per step, a new target every time — it costs **×1.07**, which
  is 1.00 inside the harness's own spread (0.94 / 1.08 / 0.91). The comparison is free when
  the gate never fires.

Those two answer different questions and an earlier version of this conflated them into one
number. The counters behind them are deterministic and are what the tests actually gate on: a
sixteen-frame step carrying one target adapts once with the gate on and sixteen times without
it — 330 against 5,265 over the sweep.

**The honest limit of all of this: it is Python.** At ~41 µs per frame the arithmetic is not
the cost — `scaling` already showed RLS is flat from n=4 to n=32, which means interpreter and
numpy dispatch overhead dominate at the design driver's dimension. These figures are not a
baseline the C++ core should be judged against, for the same reason DESIGN.md already gives
about the port. What they are good for is exactly what a benchmark is for here: noticing when
a change makes one of them worse.

### The target build, 2026-08-09 — a cross-compiler exists now, and it found things

Every size figure this project quoted before today was an x86-64 shape, said so each time,
and DESIGN.md promised that flash and RAM would come from a **link map**. An
`arm-none-eabi` toolchain is now installed (Arm GNU Toolchain 14.2.Rel1, per-user from Arm's
own zip with the published SHA-256 checked — `winget`'s package is a machine-scope NSIS
installer and needs elevation this session did not have). So the promise is kept, and it
was worth keeping: the link found three things a host build cannot.

**1. `norm()` was calling into libm, and `-ffreestanding` was not the reason.** GCC will not
emit `VSQRT.F32` for `sqrtf` unless told it need not set `errno` on a negative argument.
Measured: without `-fno-math-errno`, three `sqrtf` call sites and zero `VSQRT.F32`; with it,
zero calls and five inlined instructions. `Rls::update` calls `norm()` three times, so this
is on the hot path. The flag is now in the CMake embedded profile and in the test's flag
list, and it is **not** a fast-math option — it changes no result, only the promise to set
`errno`, which for a sum of squares was never meaningful. No cycle figure is quoted for the
difference, because there is still no hardware.

**2. The core needs `memcpy` and `memset` from its environment** — 48 and 69 call sites. GCC
is permitted to emit calls to those even in a freestanding build and it does, from struct
assignment and from `__builtin_memcpy` on a runtime length. Nothing in this project said so
before, and a bare-metal integrator has to supply them. Beyond those it needs only libgcc's
soft-float double helpers (`__aeabi_dadd`, `__aeabi_ddiv`, the comparisons and the
conversions). It would be convenient to call those an f64-only cost and
they are not: a review checked, and they appear in the **f32-only** image too, because
`Report`'s health scalars — `p_trace`, `p_min_diag`, the residual norms — are `double`
regardless of the adapter's `T`, deliberately, so that a guard's thresholds mean the same thing
in both languages. `sqrt` is the one genuinely f64-only symbol, and `--gc-sections` does keep it
out of an f32 image. The whole set is **pinned as a set**
in `tests/test_target_build.py`, so a new external dependency fails a test and has to be
argued for.

**3. The checked-in CMake cross recipe did not work.** It was written before any toolchain
existed and had therefore never been run. `project()` runs a compiler check that links a
full executable, which fails bare-metal because newlib's stdio wants `_read`, `_write` and
`_sbrk`; the fix is `CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY` before `project()`. Same
category as `select_newest` being compiled but never called: a thing that looks maintained
because nothing had asked it to run.

**Flash and RAM, from a link map at last.** `--gc-sections` means there is no single "size of
the library" — what reaches flash depends on which entry points a caller touches. So
`cpp/target/usage.c` is a firmware-shaped caller in four configurations, linked separately
against `cpp/target/cortex-m4.ld`. Cortex-M4F, `-Os`, newlib-nano:

| configuration | flash | of which argmind | libc/libgcc | static RAM |
|---|---|---|---|---|
| bare RLS f32 6×1 | 2,340 | 1,258 | 916 | 272 |
| + guard | 6,144 | 3,526 | 2,264 | 1,296 |
| + record format | 6,376 | 3,804 | 2,264 | 1,972 |
| full composed pipeline | 10,380 | 7,374 | 2,424 | 3,264 |

Two corrections from an audit are folded into that table. The guarded row was 5,936 because
`usage.c` never called `freeze`, `thaw`, `reset`, `counters` or `frozen_by` — so `--gc-sections`
stripped six entry points out of the configuration commented as "what the field should run",
while DESIGN.md's own `on_frozen` policy and its "a silent guard is an undebuggable one" both
assume a device that reads its counters and can thaw. Calling them costs 208 bytes. And the
`argmind` column was `.text` only, which understated the one row where the library's 72 bytes
of `.rodata` survive `--gc-sections`; it is text plus rodata in every row now.

Against the declared 32 KB flash and 8 KB RAM, the whole composed pipeline is **32% of the
flash budget and 40% of the RAM**. The linker script carries those budgets so overflowing
them is a link error rather than a number somebody has to notice.

**Every byte of that RAM is the caller's.** The library defines **zero** bytes of `.data` and
`.bss` and 72 bytes of `.rodata`. The `.bss` figures above are the storage `usage.c` declares —
the handle, the rings, the two flash slots, the restore buffer — plus 16 bytes of the
measurement harness's own `volatile` sinks, which exist to stop the optimiser deleting the
calls. 3,248 declared and 3,264 measured for the full pipeline; the earlier "exactly" was 16
bytes out.

That test needed fixing too. It read the sections named exactly `.data` and `.bss`, which are
**always** zero under `-fdata-sections` because every object lands in `.data.<name>` or
`.bss.<name>` — an audit compiled a probe with 288 bytes of globals and the test passed. It now
sums both families and pins `.rodata` at 72 rather than under a guessed 512.

**What is still absent, deliberately: any on-target timing.** There is no hardware and no
emulator on this machine, so a cycle count would be interpolated from a host measurement,
which DESIGN.md has refused to do since the target range was settled. The figures above are
sizes and dependencies, which a link can establish; speed is not one of them.

Two smaller notes. The f64 instantiations pull a soft-float `sqrt`; `--gc-sections` keeps it
out of an image that only touches f32 entry points, verified rather than assumed. And
`cpp/target/cortex-m4.ld` is **not a board support file** — no vector table, no startup code,
no peripheral regions. Its only job is to give the linker somewhere to put sections so a map
exists. A real firmware replaces it and adds its own drivers, so these are the library's
contribution to a budget rather than an image anyone could flash.

### Execution on the target, 2026-08-09 — the last verification limit closes

"Execution is host-only" had been this project's standing limit since the target range was
settled. It is not any more. `cpp/tests/test_core.cpp` runs under QEMU's `mps2-an386` — the
declared floor, ARMv7E-M with a single-precision FPU — and all of it passes there.

QEMU came the same way the toolchain did: extracted rather than installed, because its
installer is NSIS with `requireAdministrator` and this session has no elevation. Getting an
extractor was itself a small chain — 7-Zip's own installer is a 7-Zip SFX, so the standalone
`7zr.exe` unpacks it, and the full `7z.exe` that falls out unpacks NSIS.

**What running it bought, and it is the thing that justified doing it.** `persist.hpp` guards
`count * sizeof(T)` against wrapping, and its own comment records the guard as an *equivalent
mutant* on a 64-bit host: the threshold is 2.3e18 for `double` and `count` is a `uint32_t`, so
no record can express a value that reaches it. On a 32-bit target the same expression is
536,870,876. There is now a case that forges a `count` of `1 + 2^29`, chosen so the wrapped
product makes the length check *agree* with the real buffer, and repairs the CRC so the record
is rejected for its length rather than for being corrupt. Measured by removing the guard and
running both ways:

| | guard present | guard removed |
|---|---|---|
| ARM, 32-bit `size_t` | 4,603 checks, 0 failures | **2 failures — caught** |
| x86-64, 64-bit `size_t` | 4,603 checks, 0 failures | 0 failures — equivalent |

A guard that could only be argued for is now one that is tested, and the header's claim of
untestability on the host is itself verified rather than taken on trust.

**That covered one of two identical guards, which an audit had to point out.** `pack` carries
the same wraparound check with the same "equivalent mutant" comment, and removing *it* still
passed on ARM — the on-target run had made the read path testable and left the write path
exactly where it was. Guarding one direction and not the other is already recorded in
CLAUDE.md as an earlier omission here; testing one direction and not the other is the same
mistake one level up. There is now a case for the write path too, and it matters more than the
read one: `count` reaches `pack` from the caller across the C ABI rather than out of a record
that a CRC also protects.

**Three bugs in the harness, all of which the host build would have hidden.** They are worth
recording because each is a way a cross build that compiles perfectly still does not run.

- **The vector table had two entries.** A UsageFault then has nowhere to go: the core reads a
  null handler, faults again, and locks up with `PC=0` and the fault cause already gone. A
  lockup says something went wrong and nothing about what.
- **The fault reporter called `printf`.** The original fault was *inside* printf, so
  re-entering stdio from the handler faulted again — a second lockup, subtler than the first.
  A fault reporter may not depend on anything that can be the thing that broke; it uses raw
  `bkpt 0xAB` semihosting now.
- **Nothing copied `.data`.** newlib's crt0 zeroes `.bss` and assumes a loader put `.data` at
  its run address, which is true for a hosted program and false for a ROM image whose LMA and
  VMA differ. The first casualty is newlib's own `_impure_ptr`, so the symptom was
  `__sflush_r` doing `blx r6` on a garbage write pointer — stdout's vtable read out of
  uninitialised RAM.

Diagnosing the third took the first two being fixed, which is the argument for a startup file
that reports rather than one that is merely short.

**Two smaller confirmations.** The FPU works and is enabled correctly: `VSQRT.F32` of 2.0f
returns `0x3fb504f3` exactly. And `sizeof(int)/(long)/(void*)/(size_t)` are all 4 on the
target — asserted as its own test, so a board or toolchain change that widened `size_t` fails
distinctly rather than confusingly.

**And two things a review established about the harness itself.** Its 64 KB stack and 256 KB
heap are not what bounds the running image: `rdimon-crt0` opens with a `SYS_HEAPINFO`
semihosting call and overwrites the stack pointer with whatever the host reports — measured,
the linked `_estack` is `0x20050398` and the live SP at `main()` is `0x21fffff0`. The linker
script's numbers are a floor and the symbols the link needs, nothing more. More importantly,
**this image could never be firmware**, and for a stronger reason than the usual absent clock
setup: semihosting means real silicon with no debug host attached faults on the first
`bkpt 0xAB`. `cortex-m4.ld` and `usage.c` are the product-shaped pair; the QEMU harness runs
tests and nothing else. Both are now said in the files themselves.

That review also answered a question the budget measurement had left open: a real vector table
plus a reset handler that enables the FPU, copies `.data` and zeroes `.bss` measures **154
bytes**, so omitting them from `cortex-m4.ld` understates a firmware's flash by under half a
percent of the budget.

**What is still absent, and now for a narrower reason.** QEMU models the instruction set, not
the pipeline, the flash wait states or the bus, so a cycle count taken here would be fiction
dressed as measurement. None is taken. An *instruction* count would be honest — it is
architectural rather than temporal — but the Windows QEMU build ships no TCG plugins, so it
needs one built first and was not attempted. The on-target timing claim therefore stays
absent, as it has since the target range was settled; what changed is that the reason is now
"nobody has measured it on hardware" rather than "nothing can run the code at all".

### Parity across architectures, 2026-08-09

Every parity claim this project had made was x86-64 C++ against x86-64 Python. That is a real
gap, and the record format makes it obvious: `persist.hpp` declares that "the writer and the
reader may be different machines" and had only ever been checked on one. The others are
quieter but the same shape — AAPCS lays structs out differently, `size_t` is half the width,
and a VFPv4 single-precision unit is not an SSE unit.

All four parity tools now build for Cortex-M4 and run under QEMU against the **same** Python
reference the host build is compared to. No tool changed: semihosting carries `argv` in and
file I/O out, so each is invoked through a wrapper that looks like a program taking
`(job, out)` — which is exactly what the existing harnesses already call. `tests/
test_target_parity.py` therefore reuses their encoders and their comparison functions rather
than growing a second set that could drift out of agreement with the first.

What holds, on the target:

- **The record format is byte-identical** to the reference at f4 and f8, at 6×1 and 12×3. A
  record the host wrote unpacks on the target with every header field intact, and a
  single-bit corruption in it is still refused with `crc_mismatch` — so the agreement is not
  the CRC being skipped on a machine where the arithmetic differs.
- **RLS agrees to the same handful of ulp** as the host build does, at both precisions, over
  300 excited steps.
- **The guard makes the same decisions**: the freeze step and the applied/blocked sequence
  match exactly across 400 steps with a divergence check and an injected fault. A guard
  decides rather than computes, so a decision that differed would not be a rounding
  difference.
- **Alignment picks the same samples** under hold-last, nearest and interpolate — compared by
  bit pattern for staleness and raw bytes for values, as on the host.
- **The pipeline agrees step for step**, including the freshness gate: frames, updated,
  lost_frames, every skip counter, the tripped mask and the output.

**What runs there is a curated subset and not the whole suite**, and the reason is cost:
each QEMU invocation is about a second and the host suites sweep hundreds of configurations —
840 pack configurations for the record format alone. The subset covers each tool's
distinguishing claim rather than sampling evenly. Saying "parity holds on ARM" without saying
which cases ran would be the kind of unbounded claim this project has had to withdraw before.

### What the parity audit found, 2026-08-09

The third of three independent reviews, and the sharpest. It found a **real bug**, an
**undeclared divergence**, and four places where a tolerance or a fixture set is narrower
than the claim resting on it. The first two are fixed; the rest are recorded here as open,
because writing them down is cheaper than pretending and much cheaper than forgetting.

**Fixed: the port withheld an output the reference produced.** `Pipeline::step` emitted its
prediction only `if (frames && frame_ready_)`, and `build_frame` cleared `frame_ready_` on
entry. So a step in which an earlier reference instant built a frame and a later one failed
reported frames, a reference time and staleness — and no output. The reference loops over the
frames `drain()` returned, all of which exist, so it never sees a failed instant at all.
Measured: **118 disagreeing steps in 250 randomised cases**, identically on the host and the
target, which is what says logic rather than rounding.

The fix is the one that removes the hazard rather than guarding it: extraction builds into
scratch and commits on success, so `x` always holds the last *complete* frame and the mixture
of two instants that `frame_ready_` existed to mark cannot arise. 28 bytes at the design
driver's shape. The test that pinned the old behaviour asserted the opposite of what it now
asserts, which is what a test pinning a divergence looks like.

Worth one more note, because it is the session's recurring mistake in miniature: the parity
case added for this **does not catch the gate alone**. Restoring both halves of the old
behaviour fails it; restoring only the `&& frame_ready_` does not, because the same fix
stopped a failed build from touching the flag. The gate on its own is now an equivalent
mutant. Checking which mutant a test kills is a different act from writing the test, and
skipping it is how this project keeps producing tests that pass for the wrong reason.

**Declared: a fourth divergence, on the target only.** `rls.hpp` listed three deliberate
differences from the reference. There is a fourth and nobody chose it: GCC contracts `a*b + c`
into `VFMA.F32`, which VFPv4 on the Cortex-M4F has and the x86-64 SSE2 baseline does not. Over
sixty configurations the target's f32 figures are not the host's — state 2.0 ulp against 5.0,
`residual_norm` 14.2 against 38.0, `gain_norm` 87.4 against 134.0 — and `-ffp-contract=off` on
the target build reproduces the host numbers *exactly*, which is what identifies the cause
rather than merely correlating with it. f64 is untouched, having nothing to contract on a
single-precision FPU. `align.hpp` had said the M4F "cannot" contract; true of the double blend
it was describing, false of everything in `rls.hpp`, and the two sentences sat next to each
other reading as one.

**Open, with the measurement that shows it.**

- **The state tolerance of 400 ulp is generous for the fixtures and is exceeded by what the
  harness can express.** Over a 486-configuration grid the worst is 1008 ulp at f32 and 472 at
  f64. It is not the pipeline: replaying the same `(x, y)` through plain RLS gives the
  identical figure to the last digit. It is the **declared** RLS divergence at conditions the
  baseline never swept — the "about 1.5 ulp on theta" headline is a property of `excited()`,
  whose SNR is about 200. At SNR ≈ 1 the same code disagrees by 30–100 ulp; at `p0 = 1000`
  over a short run, by several hundred, peaking in the transient and decaying. The honest fix
  is a noisy fixture and a re-derived tolerance, not a looser number.
- **The prediction metric normalises by the wrong scale.** It bounds against the run's largest
  *prediction*, where the project's own rule for a cancelled quantity is the scale of the
  values being subtracted — here `max_j |x_j θ_j|`. The two coincide when θ and x are O(1),
  which is why the fixtures pass, and diverge under covariance windup. Worst over the grid:
  472.6 ulp at f32 against the 200 allowed.
- ~~The cross-architecture subset has almost no discriminating power, and the cost argument behind it was wrong by six times.~~ **Closed** — see below.
- Two smaller ones. The baseline ulp table says its maxima are over λ ∈ {0.97, 0.99} and every
  figure reproduces only at 0.99; at 0.97, `gain_norm` is 110 against the tabulated 58 and
  `p_trace` is 10.0 against 1.4. And `pipecheck` instantiates only `Rls<T, N, 1>`, so no
  multi-output case has ever run through the freshness gate — which the reviewer checked by
  construction and could not check by execution.

The gate itself came through clean: sixteen targeted edge cases and sixty randomised seeds, on
both the host and the target, with no disagreement — including timestamps at 1e15 where
`t_ref + staleness` no longer round-trips, staleness overflowing to −∞, and a NaN target
recorded on the attempt rather than on success.

### The subset became the whole thing, 2026-08-09

The parity review left two findings about the cross-architecture suite that were really one:
its dozen hand-picked cases discriminated nothing, and the reason given for picking a dozen
was wrong by six times. A QEMU invocation is **157 ms**, not "about a second".

So there is no subset. `tests/_toolbuild.py` builds a parity tool for either architecture and
the four suites take their tool fixture from it, parametrised `[host]` / `[target]`. Every
parity case runs on both: **479 tests in 105 seconds** with the toolchains present, and the
target half skips cleanly without them. The tools are unchanged — semihosting carries `argv`
in and file I/O out, so a target build is invoked through a wrapper that looks like a program
taking `(job, out)`, which is what the harnesses already called.

**What is left in `test_target_parity.py` is the one question the parametrised suites cannot
ask.** Each of them compares one build against the reference; none compares the two builds
against each other. A target whose arithmetic is subtly not IEEE would agree with the
reference to within its own tolerance and be an order of magnitude worse than the host, and
every other test in this repository would pass. So that file now measures both builds'
deviation on the same case and bounds the ratio.

The bound is 4×, and it cannot be 1× for a stated reason rather than for slack: `VFMA.F32`
contraction is `rls.hpp`'s fourth divergence and legitimately costs about 2.5× on the worst
quantity at f32. **Verified rather than assumed**: the review's round-toward-zero fault was
rebuilt here and pointed at the new check, which reports 37.5× on the state block and fails.
f64 is asserted *identical* between the two builds as the control — soft float has nothing to
contract and nothing for a rounding mode to change, so if it ever drifts the 4× above is
masking something rather than tolerating a known difference.

### Tolerances derived from the problem, not from the fixtures, 2026-08-09

The parity review's last two open findings were the same finding twice: a tolerance that
looked comfortable because every fixture behind it was quiet.

**The prediction metric was normalising by the cancelled quantity.** It bounded against the
largest *prediction* in a run, where this project's rule for a cancelled quantity is the scale
of the values being subtracted — here `max_j |x_j theta_j|`, the largest term summed to make
it. The two coincide when theta and x are O(1), which is why every fixture passed, and they
diverge exactly where the cancellation is worst. Switching to the correct scale made the
comparison **stricter**: a case that read 134 ulp under the old metric reads 401 under the new
one. Third quantity in this project to need that treatment, and the third time the first
attempt normalised by the wrong thing.

**And the fixture set was two hundred times quieter than the problem.** `excited()` puts
`0.01*randn` on a signal of about 3. There is now a case where the noise is the size of the
signal, and both bounds were re-derived from a sweep across it — SNR 300 down to 1, `p0` from
1 to 1000, on both architectures, under the corrected metric:

| | SNR 300 | SNR 3 | SNR 1 | SNR 1, `p0`=1000 |
|---|---|---|---|---|
| state | 1.5 | 10.4 | 28.5 | **249** |
| prediction | 4.8 | 47.7 | 152 | **947** |

**One number cannot cover that, so there is not one.** The defaults — 400 on the state, 200 on
the prediction — hold everywhere with a sane prior. A prior chosen far larger than the data
warrants gets its own declared pair, 500 and 1500, with the measurement beside it. Loosening
the default to cover the worst case would have made it meaningless for the other twelve, which
is how a tolerance stops being evidence.

The deviation is a property of the conditioning rather than of the port, and that is checkable:
replaying the same `(x, y)` through plain RLS gives the identical figure to the last digit. So
the "about 1.5 ulp on theta" headline in the RLS section is true of `excited()` and is not a
property of the port at large.

**The host-to-target ratio bound moved too, for the same reason in miniature.** Four was
derived from one fixture, where the FMA divergence costs 2.5×. The SNR sweep found 4.7 on a
noisier one — so the bound would have failed the moment the fixture set grew, which it just
did. Eight covers the measured range and still fails a round-toward-zero FPU, at 37.5, by a
factor of five.

### The last two audit items, 2026-08-09

**The baseline ulp table was understated on almost every row, and the reason is instructive.**
It said its maxima came from "n=2/6/12, one and three outputs, λ=0.97 and 0.99". Every figure
reproduced at λ=0.99 and none at 0.97 — `gain_norm` was 58 where 0.97 gives 110, `p_trace` 1.4
where it gives 10.0. The axis was named in the prose and not swept in the measurement. The
same sentence implied three outputs at every n, and `xcheck` instantiates (6,3) alone. The
table is re-measured over every shape that exists, at three run lengths, both λ, and now both
architectures — that last pair of columns being new because `VFMA.F32` contraction makes the
target's f32 figures its own.

It also changed what the tolerances look like. `FIELD_ULP` is 200 for four quantities and the
headroom is not uniform: twenty times for `p_trace` and `p_min_diag`, **1.5 times** for
`gain_norm` at f32 on the target. DESIGN.md had called it "three to eight times the worst
measured"; for `gain_norm` it never was, and the worst it compared against had been taken at
one λ.

**And no multi-output case had ever run through the freshness gate.** The gate decides from
the target sample's timestamp, and the two implementations get there differently: the
reference builds a tuple over the *unique target streams* and compares it whole, the core
iterates *outputs* and ORs "changed". Equivalent by construction — and `pipecheck` instantiated
only `Rls<T, N, 1>`, so the review could check that by reading and not by running. It now has
a three-output axis (at six features, where the ABI's `6_3` shape already lives), and a case
where two targets share a stream and the third has its own, which is where "unique streams"
and "per output" are genuinely different loops.

The first version of that fixture put the targets on the reference stream, so every frame
carried a new target and the gate could not fire. It failed its own "no frame was gated"
guard, which is the third time this session a fixture has been caught testing nothing by an
assertion written for exactly that.

### The stack nobody had measured, 2026-08-09

A question about which hardware to buy asked how much RAM the library needs, and the answer
in this document was 3,264 bytes of static storage. That is the *object*. These functions also
use the caller's stack, and nobody had ever measured it.

Measured with `-fstack-usage` for Cortex-M4 at `-Os`:

| | bytes |
|---|---|
| `am_pipe_*_step` — the hot path | 312 – 328 |
| `am_guard_*_update` | 88 – 104 |
| `am_rls_*_update` | 72 – 144 |
| `am_pipe_g_f32_6_1_init` | 1,536 |
| `am_pipe_g_f32_12_1_init` | 4,288 |

**Those are the fixed figures.** `init` built the whole object as a temporary and copied it
into the caller's storage, so before the fix `am_pipe_g_f32_12_1_init` wanted **8,296 bytes of
stack** — more than the 8 KB of RAM this library declares as its floor. On the target it names,
that entry point could not be called at all, and the disassembly agreed: `subw sp, sp, #3148`
for the six-feature one. The ABI's contract is that storage is the caller's and nothing
allocates, `am_*_bytes()` reports the object, and a caller who budgeted from that and gave the
stack a kilobyte would have been destroyed on the first call.

The fix is placement new — constructing in the caller's bytes instead of building a temporary
and assigning. The header comment that ruled it out said `<new>` was unavailable freestanding;
it is a freestanding-conforming header in C++17, which a two-line compile for the target
settles. What remains is one adapter-sized temporary, because `Pipeline`'s constructor takes
its adapter by value; removing it needs a default-constructible `Guard` and an in-place
configure, and that is recorded rather than half-done.

**The shape of the answer matters more than the numbers.** The steady state is a third of a
kilobyte — nothing. Initialisation is a transient spike of up to 4.3 KB, so a device sizes its
stack for `init` or calls `init` before anything else has grown it. `tests/test_target_build.py`
bounds both, so neither can grow back quietly.

## Standards

Carried from the parked project, where seventeen modelling errors were found and most flattered
the design. See [CLAUDE.md](CLAUDE.md). The one that matters most here: **anything that can
report only success is not evidence.** Tests must be able to show the adapter diverging, the
guard tripping, and the aligner dropping data.
