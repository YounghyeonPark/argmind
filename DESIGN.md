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
regression gate), and **the C++ core's RLS behind a C ABI, verified against the Python to about
1.5 ulp**. 185 Python tests plus 521 C++ checks, several of which exist to prove failures are
visible rather than silent.

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
`rls.hpp` (`Rls<T, N, M>`), `argmind.h` + `src/abi.cpp` (the C ABI), `tools/xcheck.cpp` (the
parity tool), `tests/test_core.cpp` (521 checks the C++ side can make on its own), and a
`CMakeLists.txt` that knows the cross-compilation recipe.

**The C ABI had one real tension to resolve.** Compile-time dimensions are the reason C++ was
chosen; a C ABI wants runtime arguments. Taking `n` at runtime resolves it by discarding the
reason, so it is resolved the other way: **the ABI exports an explicit list of instantiations**,
and adding a dimension is one line plus a rebuild. That is a real cost and it is the honest one.
Storage is the caller's — nothing allocates — with `am_rls_*_bytes()` for the true size and an
`AM_*_MAX_BYTES` compile-time *upper bound* for statically declared arrays, held to being a bound
by a `static_assert` per instantiation.

**Parity: the two agree to about 1.5 ulp on the state block, at whichever precision they run at.**
That last clause is the finding. Measured maxima in ulp, across n=2/6/12, one and three outputs,
300 to 1600 steps, λ=0.97 and 0.99:

| quantity | f64 | f32 |
|---|---|---|
| whole state block, and θ | 1.5 | 1.5 |
| tr(P) | 1.4 | 4.9 |
| min diag P | 2.4 | 1.9 |
| residual norm | 8.4 | 14.2 |
| P alone | 26.7 | 37.6 |
| gain norm | 58.0 | 64.6 |
| a posteriori residual norm | 597 | 597 |

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

**What is verified and what is not.** Execution is host-only. There is still no ARM cross-compiler
or target hardware on this machine, so: `rls.hpp` is checked to compile `-ffreestanding
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

## Standards

Carried from the parked project, where seventeen modelling errors were found and most flattered
the design. See [CLAUDE.md](CLAUDE.md). The one that matters most here: **anything that can
report only success is not evidence.** Tests must be able to show the adapter diverging, the
guard tripping, and the aligner dropping data.
