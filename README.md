# Argmind — a Computational Module for iterative solvers

A **side project**: an architecture study, a functional simulator, and a working code generator
for accelerating the algorithms embedded control actually runs — ADMM/OSQP, interior-point, SQP,
MPC, Kalman filters — plus small-network inference on the same engine.

Not a product, not a company, not for sale. Software and analysis, published because the results
are worth having in the open and because most published architecture models never admit what they
got wrong.

---

## What actually works

**ADMM runs end to end as a descriptor sequence on the functional simulator and matches reference
OSQP.** Not approximately:

| comparison | result |
|---|---|
| vs OSQP, tight bounds, **10 active constraints**, eps=1e-6 / 1e-8 | **22/22 and 30/30 iterations, max\|Δx\| = 9.2e-15** |
| control action vs a polished eps=1e-11 solve | 4.2e-6 |
| closed loop, 200 warm-started ticks | p50 7 / p95 8 / max 9 iterations |
| generated program vs hand-written | **identical, descriptor for descriptor** |

**The code generator takes a CVXPY problem and needs zero hints.** A `cp.Parameter` *is* the
statement "this changes between solves", so canonicalising at two parameter values and
differencing recovers which blocks vary. Predicted residency equals measured residency: 267 KB on
the first call, then **5.8 KB in / 0.6 KB out per tick — 46×** — with the Cholesky running on 1 of
200 ticks.

```
pip install numpy scipy osqp        # cvxpy optional, for the front end
python -m pytest                    # 429 tests
python examples/codegen_quadrotor.py
python perf_model/argmind_model.py
```

## What is measured, and what is only modelled

This distinction is enforced, not decorative. **Every performance figure from the model is a ±2×
estimate and is not quotable as anything else.**

**Measured** (real numerics or real timing, on the stated host):
- Real OSQP iteration counts on a coupled quadrotor MPC: p95 = 7 at 1e-3, 21 at 1e-6, **60–73
  cold** — the cold figure being the one a hard deadline must survive.
- Precision floors. **The EKF covariance recursion needs fp32**: bf16 and fp16 lose
  positive-definiteness on the *first* step in all three filter forms, square-root included, and
  fp16 overflows at κ(P)=1e6. For solver operands **fp16 beats bf16 by 10×** — mantissa matters
  more than exponent range, the opposite of the ML tradeoff.
- CPU software headroom: a tuned C implementation of the same ADMM beats OSQP by **1.8–3.0×**, and
  nothing reaches more than **9.3%** of the machine's own SIMD peak, because the workload is
  operand-fetch-bound rather than compute-bound.

**Modelled, ±2×, never quotable**: all area, energy and perf/W figures. See
[PHASE0_FINDINGS.md](PHASE0_FINDINGS.md).

## The most useful result

**Batch-1 solver inner loops are matrix-*vector*, so they are bound by operand fetch, not by
arithmetic.** A 16× larger MAC array bought exactly zero on the flagship benchmark; scratchpad
bandwidth bought ~9×. The general form is stronger: *a workload is operand-fetch-bound whenever
its reuse depth is set by a problem dimension rather than by the tile.*

This was then confirmed from the opposite direction on a CPU — nobody reaches SIMD peak on this
workload for the same reason, and **a GPU or NPU fails here for the same reason too.**

## Seventeen modelling errors, and why they are documented

Most of them flattered the design. They are listed in [PHASE0_FINDINGS.md](PHASE0_FINDINGS.md)
and each one is encoded as a regression test, including:

- A **phantom transcendental term** — ADMM's box projection charged as divides when it is min/max.
  That single error accounted for the *entire* apparent latency advantage.
- **GEMV modelled as GEMM**, which hid the finding above.
- Scratchpad energy **2.3× low**, MAC count conflated with operand-reuse depth, a capacity model
  assuming a packed triangular layout the address generators cannot express, working vectors
  charged **zero** bytes.
- Two found only by *building* the thing, and two more by auditing a change that favoured us.

[tests/test_guards_actually_bite.py](tests/test_guards_actually_bite.py) re-introduces each bug at
runtime and asserts the guard fails — because a regression suite that has never failed has not
been shown to catch anything. Several tests exist specifically to **protect negative results**, so
a later change cannot quietly optimise away a row that loses.

## Honest limits

- **No RTL.** This is an architecture model, a simulator and a compiler. Nothing has been
  synthesised, so every area and timing figure is an estimate.
- **A CPU meets every solver deadline in the frozen benchmark set.** The modelled advantage is
  energy (~12–22×) and a bounded worst case, not raw speed — latency is 1.0–2.1×, with one row
  *below* parity.
- **The ML efficiency claim is unsupported**, not merely narrow: every ML perf/W figure divides a
  modelled accelerator energy by a modelled *CPU* energy, and nobody runs edge inference on a
  Cortex-A when the SoC has an NPU. We have no measured NPU data and will not compare a modelled
  figure against a vendor claim.
- Large-model on-device LLM decode is outside the differentiated envelope. It is the memory wall,
  and no compute architecture wins it.

## Layout

| | |
|---|---|
| [`argmind/`](argmind/) | Runtime API, functional simulator, descriptor format v0.1, MACIF, codegen |
| [`perf_model/`](perf_model/) | Architecture model, SRAM/fp64/sequencer costing, measurement harnesses |
| [`cpu_backend/`](cpu_backend/) | Tuned C ADMM and the CPU software-headroom measurement |
| [`tests/`](tests/) | 429 invariant, regression and mutation tests |
| [`examples/`](examples/) | End-to-end quadrotor MPC, hand-written and generated |
| [PHASE0_FINDINGS.md](PHASE0_FINDINGS.md) | All findings, corrections and evidence |
| [BENCHMARKS.md](BENCHMARKS.md) | The frozen benchmark set and why each row is in it |
| [TESTING.md](TESTING.md) | Test strategy, and why the tests pin relationships rather than numbers |
| [`business/`](business/) | Written when this was framed as a company. Parked, kept for the record |

## Status

Side project. **Nothing here is published yet** — two decisions are outstanding: a trademark
register check on the name, and whether to preserve patentability (publication permanently closes
European patentability, since the EPC has no grace period). See
[PLAN.md](PLAN.md) §7.1.

`business/` contains a customer-discovery program, a go/no-go memo and an IP strategy from when
this was scoped as a startup. Parked rather than deleted: the analysis is not reconstructable and
the framing could change.
