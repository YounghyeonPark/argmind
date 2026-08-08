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

**Precision is declared and checked, not assumed.** Carried forward from measurement in the
parked accelerator project: **the EKF covariance recursion needs fp32 minimum, and there is no
fp16/bf16 path in any filter form** — conventional, Joseph and square-root all lose
positive-definiteness on the first step, and fp16 additionally overflows at κ(P)=1e6. More
generally, **precision floors are a property of the algorithm, not of the platform.** So an
Adapter declares the precision it requires and the framework refuses an unsafe configuration at
construction time rather than producing quiet garbage at hour six.

**Alignment policy is explicit and never defaulted silently.** `nearest`, `hold-last`,
`interpolate`, `drop-if-stale` give different answers. The user chooses; the framework records
the choice and reports staleness.

## API sketch — for review, not final

```python
from argmind import Stream, Pipeline, Aligner, Guard

imu   = Stream("imu",  dim=6, rate_hz=200, dtype="f4")
aid   = Stream("aid",  dim=3, rate_hz=5,   dtype="f4")

pipe = Pipeline(
    streams=[imu, aid],
    aligner=Aligner.hold_last(reference="imu", max_stale_s=0.5),
    adapter=RecursiveLeastSquares(n_features=6, forgetting=0.999, dtype="f4"),
    guard=Guard(
        freeze_on=["divergence", "stale_reference"],
        bounds={"residual_rms": (0.0, 10.0)},
        rollback_window=50,
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
   understood, and directly useful for online calibration.
2. **Kalman-family parameter estimation** — an EKF whose state is the parameters. Carries the
   covariance conditioning problem, and therefore the precision floor above.
3. **Online gradient methods** for the cases the first two cannot express, with explicit step-size
   control.
4. **Drift/change detection** to decide *when* to adapt, which is often more valuable than how.

## Open questions

Genuinely open, listed rather than guessed:

- ~~Language~~ — **DECIDED 2026-08-08: a C++ core with a C ABI, optimised for speed.** See
  "Language and structure" below. What remains open is whether the target range includes MCUs,
  which decides the matrix-type question.
- **How much of a timebase to own.** Does the library assume a monotonic clock per stream, or
  handle clock skew estimation between sources?
- **Where inference sits.** The library adapts parameters and estimates. Whether it also wraps a
  third-party inference runtime, or simply provides adapted inputs to one, is a scope boundary
  not yet drawn.
- **Persistence.** Adaptation state must survive a power cycle for any of this to be useful in
  the field. Format and cadence are undecided.

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

### Own the matrix types, at least initially

Eigen is excellent and is the default in robotics state estimation, but it is heavy for a
library that may need to run on a Cortex-M — header bloat, compile times, and a C++14+
requirement. Start with a minimal fixed-size type covering exactly what the adapters need.
**Revisit if the target range turns out to exclude MCUs**, which is the one open question left.

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

## Where this stands, and what comes next

**Built:** `Stream` (fixed-capacity ring buffers, counted overflow, monotonic-time
enforcement) and `Aligner` (four explicit policies, staleness reporting, causal/non-causal
declared). 16 tests, several of which exist to prove failures are visible rather than silent.

**Next, in order:**

1. **`Adapter`, starting with recursive least squares** with a forgetting factor. Bounded state,
   stability that can be argued about, and immediately useful for the driving example. Its
   `update`/`predict` shape sets the contract every later method must fit.
   **Write it twice, deliberately**: Python first as the executable specification and oracle,
   then C++ verified against it. Doing Python first is not wasted work — it is how the C++ gets a
   correctness anchor before anyone starts making it fast.
2. **`Guard`, immediately after — not later.** Half the reason this library exists is that online
   adaptation can diverge in the field, so the freeze path has to be the default shape of the API
   rather than something a user remembers to add. Building it second keeps that honest; building
   it fifth would not.
3. **`Pipeline`** to wire stream → align → adapt → predict behind one bounded `step()`.
4. **Persistence**, because adaptation state that does not survive a power cycle is useless in the
   field.

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
