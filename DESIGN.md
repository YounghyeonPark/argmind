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

- **Language.** Python is right for the reference implementation and the API design. But
  "embedded/edge" spans a Jetson running Python comfortably and an MCU that cannot. Whether this
  ends as Python-only, Python with a C core, or a C library with Python bindings is unresolved
  and depends on how far down the target range it must go.
- **How much of a timebase to own.** Does the library assume a monotonic clock per stream, or
  handle clock skew estimation between sources?
- **Where inference sits.** The library adapts parameters and estimates. Whether it also wraps a
  third-party inference runtime, or simply provides adapted inputs to one, is a scope boundary
  not yet drawn.
- **Persistence.** Adaptation state must survive a power cycle for any of this to be useful in
  the field. Format and cadence are undecided.

## Where this stands, and what comes next

**Built:** `Stream` (fixed-capacity ring buffers, counted overflow, monotonic-time
enforcement) and `Aligner` (four explicit policies, staleness reporting, causal/non-causal
declared). 16 tests, several of which exist to prove failures are visible rather than silent.

**Next, in order:**

1. **`Adapter`, starting with recursive least squares** with a forgetting factor. Bounded state,
   stability that can be argued about, and immediately useful for the driving example. Its
   `update`/`predict` shape sets the contract every later method must fit.
2. **`Guard`, immediately after — not later.** Half the reason this library exists is that online
   adaptation can diverge in the field, so the freeze path has to be the default shape of the API
   rather than something a user remembers to add. Building it second keeps that honest; building
   it fifth would not.
3. **`Pipeline`** to wire stream → align → adapt → predict behind one bounded `step()`.
4. **Persistence**, because adaptation state that does not survive a power cycle is useless in the
   field.

**Decide before (1) is finalised:** the language question in the open list below. It does not
block the Python reference implementation, but it decides whether the `Adapter` contract should
be shaped for a future C core.

## Standards

Carried from the parked project, where seventeen modelling errors were found and most flattered
the design. See [CLAUDE.md](CLAUDE.md). The one that matters most here: **anything that can
report only success is not evidence.** Tests must be able to show the adapter diverging, the
guard tripping, and the aligner dropping data.
