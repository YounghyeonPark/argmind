# Argmind — design

**A sensor-agnostic library for on-device online adaptation and inference, for embedded and
edge targets.** Started 2026-08-08. Status: built and verified on the declared floor — the
runtime is across to C++, and what is absent is timing on real silicon.

**The adaptation is one algorithm: recursive least squares, linear in the parameters.** That
sentence belongs at the top rather than three sections down, because it is the single fact
that decides whether this library can do a given job. See *What it is not* below for what
that rules out.

## What it is

Multi-sensor systems in the field drift. Bias moves with temperature, a mount shifts, a surface
changes, a battery ages, a user is not the average user. The usual answer is to train offline and
hope the deployment matches the training set. This library takes the other route: **the model
keeps updating from the sensor stream after deployment**, inside bounded memory and bounded time,
with explicit guardrails so that adaptation cannot quietly destroy the system it is adapting.

It is a **library**, not an application and not a pipeline tool. The API is the product.

## What it is not

Naming these now because a sensor-agnostic framework drifts into "everything" otherwise.

**Two of these were reversed on 2026-08-10** and are left standing with their reversal
attached, because what a project stopped believing is evidence about how it decides. See *The
platform generates the basis* below for the decision and the measurement that priced it.

- ~~**Not a training framework.** No backpropagation, no autograd, no GPU.~~ **Reversed
  2026-08-10 for the second tier.** Generating the basis is the platform's job, and a network
  small enough to be a head trains inside the `Adapter` contract with the framework unchanged —
  measured. *Offline* training of a deep backbone is still out: backprop retains activations,
  not weights. The core tier keeps its model class.
- ~~**Not a deep-learning runtime.** If you need to run a CNN, use ONNX Runtime or TFLite.~~
  **Reversed 2026-08-10 for the second tier**, which must run its own inference network rather
  than receive a feature vector from someone else's. Sitting alongside an external runtime
  stays supported and stays the recommendation on an NPU part, where the vendor's runtime is
  the only path to the accelerator.
- **Not a driver layer or a robotics middleware.** It does not talk to hardware, does not own
  threads, and does not depend on ROS. You feed it samples.
- **Not a research playground for online deep learning.** Still true, and the reversal above
  gives it a sharper edge than "stability can be reasoned about": the guard has no `p_trace`
  for a backpropagated weight, and **`residual_norm` was measured not to substitute for one**
  (below). Methods land here when their failure is *detectable*. For this model class nothing
  in the library detects a slow loss of quality, and that is stated rather than designed
  around.
- **Not a model that can learn a representation, and this is the load-bearing limit.** Every
  bullet above says what the *implementation* is not. None of them said what the *model class*
  cannot express, and until 2026-08-09 the phrase "linear in the parameters" appeared nowhere
  in this repository — the most important property of the only algorithm here was undocumented
  while four sentences explained that it is not a CNN.

  `Rls` fits `y = xᵀθ`. `predict` is one `vec_mat` — at n=6, M=1 that is six multiply-adds
  over a 24-byte θ, and **those 24 bytes are the whole of what is learned**. So:

  - **You supply the basis.** The library will not discover that `sign(v)` matters, that a
    product term is needed, or that a feature should be logged. Feature choice is the
    modelling work and it stays with the caller — which is also why `FrameMap` makes feature
    order a declared, fingerprinted thing rather than a convenience.
  - **Nothing with memory of its own input can be fitted memorylessly.** Hysteresis, stiction
    state, backlash and rate-dependent friction all depend on the input's *history*, and no
    choice of instantaneous features represents them. Fitting one needs a state-carrying basis
    (LuGre, Bouc-Wen, a lagged term) built by the caller and fed in as features.
  - **What it is genuinely good at** is tracking a *drift* in something whose form is already
    known: a bias that moves with temperature, a gain that ages, a mount that shifts. The
    forgetting factor is exactly a statement that the coefficients are expected to move.

  This bullet exists because a hardware-scoping question produced an over-optimistic
  application for this library, and the source cited to justify it said in the same paragraph
  that the phenomenon "requir[es] dedicated compensation models, not a scalar correction".
  The favourable half of a sentence was quoted and the unfavourable half was not.

## Where the running log went

This file is a **design document**: what the thing is, what it is not, the abstractions and the
decisions behind them. The dated measurement narrative that used to live here — 1,983 lines of it —
is verbatim under `docs/design/`, because a design document that grows a log stops being readable as
either.

| file | what is in it |
|---|---|
| `docs/design/basis-and-scope.md` | the 2026-08-09 widening and the 2026-08-10 basis decision, with the measurements each rests on |
| `docs/design/status-log.md` | the running "where this stands" log — every component as it landed |
| `docs/findings/` | the same treatment applied to CLAUDE.md; seven files by area |
| `docs/ARCHITECTURE.md` | the block diagrams, checked against the running code |

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

## What `step()` returns, and why that is the design

**`step()` returns a report rather than nothing.** Observability is not a logging afterthought here;
it is the return value. `StepReport` carries `updated`, `frozen_by`, `staleness_s` and the
prediction, so a caller can answer "is it learning right now" without reaching inside anything —
which is exactly the question the demo could not answer until its overlay was fixed.

**A worked example used to sit here and it did not run.** It was written before `FrameMap` existed
and omitted it, so it was missing a required argument — and `FrameMap` is not a detail: mapping a
frame to `(x, y)` through a *declared index table* rather than a callable is one of the load-bearing
decisions in this library, because a callable does not cross a C ABI. A sample that predates the
decision it should be demonstrating is worse than no sample.

Examples that are executed rather than written: `README.md`'s C program is compiled and run against
an installed prefix by `tests/test_packaging.py`; `tests/test_pipeline.py` and `demo/test_pointer.py`
drive the whole composition. Read those.

## Adaptation methods — the plan, and where it did not hold

Chosen for stability that can be argued about, not for novelty. **Two of the four are built, and
they are not the first two** — recording that is the point of keeping the list, since a plan that is
quietly re-ordered stops being evidence of anything.

1. **Recursive least squares** with a forgetting factor — the workhorse. Bounded state, well
   understood, directly useful for online calibration. **BUILT**, and checked against the
   closed-form exponentially weighted solution rather than against itself.
2. **Kalman-family parameter estimation** — an EKF whose state is the parameters. Carries the
   covariance conditioning problem, and therefore the precision floor above. **NOT BUILT.**
3. **Online gradient methods** for the cases the first two cannot express, with explicit step-size
   control. **BUILT, out of order** — `NormalisedLeastMeanSquares` came first because a network
   head's width made RLS's n² the binding constraint (330,544 B against 2,872 at 128 features), and
   `MultilayerPerceptron` followed the 2026-08-10 scope decision. Neither was reached by working
   down this list; both were forced by a measurement.
4. **Drift/change detection to decide *when* to adapt**, which is often more valuable than how.
   **NOT BUILT.** `Divergence` is adjacent and is not this: it is a guard *check* that freezes an
   adapter that has already gone wrong, not a detector that decides adaptation is warranted.

## Open questions

Genuinely open, listed rather than guessed. Questions this file used to carry with their answers
struck through — language, target range, persistence, whether a guard's own state persists — are
**decided**, and their reasoning lives in `docs/design/status-log.md` and CLAUDE.md rather than
being restated here as crossed-out text.

- **How much of a timebase to own.** `Stream` enforces monotonic time at the source, so ordering
  within one stream is settled. Clock *skew* between sources is not: nothing here estimates it, and
  whether that belongs in the library or in the caller has never been decided.
- **Where inference sits.** The library adapts parameters and estimates. Whether it also wraps a
  third-party inference runtime, or simply provides adapted inputs to one, is a scope boundary
  still not drawn — and the 2026-08-14 decision that argmind must do the learning and inferencing
  itself sharpens the question rather than answering it.

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

## Standards

The evidence rules are in CLAUDE.md and are not restated here. The one that bites hardest at this
level: **anything that can report only success is not evidence.** Tests must be able to show the
adapter diverging, the guard tripping, and the aligner dropping data — and this repository has
caught fixtures failing that three times, each of which passed until a control was added.
