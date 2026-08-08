# Argmind CM — Frozen Phase-0 Benchmark Set (v1.2)

> **v1.2, 2026-07-30 — #3 PWR-MPC re-specified.** The v1.1 definition (continuous-control-set
> condensed QP solved by ADMM at 20 kHz) described a formulation that appears on
> rapid-prototyping platforms but is not what ships in the motor-drive / power-electronics
> socket. Deployed practice is **finite-control-set (direct) MPC**: enumerate |U|^Np switching
> sequences by exhaustive search, no QP at all — or explicit MPC, which is a table lookup and not
> a compute workload. Rate also **raised 20 → 40 kHz** (Ts = 25 µs), because direct MPC needs
> fs ≈ 100× the switching frequency, so the old rate was *below* deployed practice. Sources and
> the full consequence analysis: [PHASE0_FINDINGS §0.3](PHASE0_FINDINGS.md),
> [perf_model/results_pwr_mpc_respec.txt](perf_model/results_pwr_mpc_respec.txt).
>
> **Consequences: the run-ahead hardware requirement and the ≥8-wide chain-blocking RTL
> requirement both lose their only deadline justification (downgraded, not removed); the
> 64 B/cycle bandwidth floor now rests on QUAD-MPC alone; the thinnest-codegen-margin row is now
> ARM-MPC. The row is still below parity (0.62×) — for a new and more defensible reason.**

> **v1.1, 2026-07-30 — #2 ARM-MPC re-specified.** The v1.0 definition (60 SQP iterations of a
> fully-condensed QP forming `AᵀA`) overstated the work ~23× and matched no production stack:
> OSQP explicitly does not form `AᵀA`, and deployed nonlinear MPC uses the real-time-iteration
> scheme — *one* SQP step per tick. Rate also lowered 500 → 250 Hz, since published 7-DoF
> manipulator NMPC runs at 100–250 Hz above a 1 kHz joint loop. **Consequence: ARM-MPC now has
> 24× deadline margin instead of being the binding constraint, so no solver benchmark is
> latency-binding, and the CM-100 sizing that was justified by it must be re-run.**

**Frozen 2026-07-29.** These seven workloads define what "winning" means. The go/no-go
decision, all architecture trade-offs, and all external claims are evaluated against this
list and nothing else. Adding a benchmark after freeze requires re-running the whole set.

Implemented in [perf_model/argmind_model.py](perf_model/argmind_model.py); results in
[PHASE0_FINDINGS.md](PHASE0_FINDINGS.md).

## Why these seven

Coverage requirements: both workload classes (sequential-iterative and matmul-dominated);
a spread of problem sizes spanning below and above the expected crossover point; at least
one workload where the CPU is expected to win (to keep the model honest); at least one on
incumbent-NPU home turf (to know what we concede).

| # | Name | Class | Definition | Rate / deadline | Why it is in the set |
|---|---|---|---|---|---|
| 1 | **QUAD-MPC** | solver | Quadrotor MPC, 12 states, 4 inputs, horizon N=20, condensed QP via ADMM (OSQP-style), 40 iterations, cached factorization | 1 kHz / 1000 µs | The flagship beachhead case. LTI problem → matrices fixed → pure matrix-vector inner loop |
| 2 | **ARM-MPC** (re-spec v1.1) | solver | 7-DoF manipulator NMPC as deployed: **acados SQP-RTI — one SQP step per tick** (preparation + feedback phase), partial condensing + **HPIPM** block Riccati, 15 interior-point iterations, **fp64**. 14 states, 7 inputs, N=30 | **250 Hz** / 4000 µs | Genuine matrix-matrix work, but on small (n_x+n_u)=21 blocks at ~47% tile utilization |
| 3 | **PWR-MPC** (re-spec v1.2) | solver | Motor-drive / power-electronics MPC **as deployed: finite-control-set (direct) MPC**. Three-level NPC inverter → U = {−1,0,1}³, **\|U\| = 27** switching states; horizon **N<sub>p</sub> = 2** by exhaustive enumeration (**729 sequences**); n_x = 4, n_y = 2 controlled outputs; fp32. **No iteration — fixed op count, no convergence tail** | **40 kHz** / 25 µs (Ts = 25 µs) | Latency extreme — tests whether per-call overhead is survivable at all. **Answer: it is not, at this size.** Also the only row whose binding resource is the vector unit |
| 4 | **EKF-24** | solver | INS/GNSS 24-state EKF, 12 measurements, Joseph-form covariance update | 1 kHz / 1000 µs | Deliberately the smallest problem — expected to be a CPU win or a tie. Defines the floor |
| 5 | **EKF-96** | solver | Large-state estimator (SLAM-class), 96 states, 24 measurements | 200 Hz / 5000 µs | Covariance recursion is matrix-matrix → tests the tile array on solver work |
| 6 | **SLM-DECODE** | ML | 1.1B-parameter SLM (TinyLlama-class), INT8, batch 1, autoregressive decode, 22 layers, hidden 2048, context 512 | 20 tok/s / 50 ms | The edge-LLM claim. Batch-1 decode is matrix-vector and weight-bound |
| 7 | **VIT-TINY** | ML | ViT-Tiny, 224×224, INT8, 1.26 GMAC/inference, 5.7 MB weights | 30 fps / 33.3 ms | Incumbent NPU home turf. We need to know exactly how much we concede here |

### Off-set rows (modelled, **not** part of the frozen seven)

Nothing may be provisioned from these without saying so out loud. They exist so a re-specification
leaves an audit trail and so the model retains live examples of effects the frozen set no longer
exhibits. Defined in `perf_model/argmind_model.py:OFF_SET`.

| Row | What it is | Why kept |
|---|---|---|
| **PWR-CCS** | The retired v1.1 PWR-MPC: continuous-control-set ADMM, 20 kHz | CCS MPC is a real if smaller socket (indirect MPC + modulator, solved with qpOASES/ODYS). It is also the only live example of the run-ahead and chain-blocking deadline misses |
| **PWR-FCS1** | One-step FCS-MPC, 27 candidates, 40 kHz | What MCUs ship today, and **a case we lose** (0.12×, and below the ≥8-cycles/descriptor codegen rule). Keeps the model able to report a minimum viable problem size |
| **PWR-FAST** | Two-level, \|U\| = 8, N<sub>p</sub> = 2 (64 sequences), Ts = 10 µs / 100 kHz | The fastest deployed direct-MPC sampling interval. Also below the codegen rule |
| **PWR-FUSE** | Frozen PWR-MPC with accumulator broadcast-init + square-accumulate | Prices two RTL features against the vector-unit bottleneck |

## Baselines every benchmark is measured against

| Baseline | Configuration | Role |
|---|---|---|
| **Cortex-A55-class** | 1.5 GHz, in-order, NEON (8 fp32 FLOP/cyc), 16 B/cyc L1, ~0.15 W | The honest **perf/W** adversary — what cost-sensitive robotics actually ships |
| **Cortex-A78-class** | 2.0 GHz, OoO, NEON (16 fp32 FLOP/cyc), 32 B/cyc L1, ~0.9 W | Mainstream application core |
| **Cortex-A720-class + SME2** | 2.2 GHz, 256-bit streaming (64 fp32 FLOP/cyc **matrix**, **32 FLOP/cyc plain SIMD**), 64 B/cyc L1, ~1.3 W | The **future** competitor — CPU matrix extensions are our highest-probability real rival |

> **Correction, 2026-07-30, and it moves ratios in our favour so it is flagged rather than
> banked.** The model charged *elementwise and reduction* work at each baseline's matrix rate,
> i.e. it ran min/max and horizontal reductions on SME2's ZA tile, which cannot execute them.
> Measured M4: FMOPA FP32 2008 GFLOP/s versus ≈250 GFLOP/s for streaming-SVE non-outer-product
> — an 8× gap. The SME2 baseline now carries a separate `simd_flops_per_cycle = 32` [EST]: **half**
> the matrix rate rather than the measured eighth, i.e. the value least favourable to us. NEON
> baselines are unchanged (one set of units). Effect: vector-heavy rows move, GEMM-dominated rows
> do not. **The retired PWR-CCS row's 0.974× was partly this artefact — it is 1.08× corrected,
> 1.72× at the measured M4 ratio. Neither number is quotable; the row sits on the assumption.**

Comparison rule, conservative on both axes: **speedup** is measured against the *fastest*
baseline, **perf/W** against the *most energy-efficient* baseline. Solver working sets are
assumed cache-resident on the CPU (favourable to the CPU, and true at these sizes).

## ⚠ QUAD-MPC has an interior optimum and cannot validate ADMM's distinctive parts

Found 2026-07-30 while implementing the solver on the functional simulator. At the nominal ±2 N
per-rotor bounds, **QUAD-MPC's optimum is interior** — OSQP reports no active set on all 200
closed-loop ticks. Since y* = 0 at an interior optimum, both the box-projection kernel and the
dual update are exercised only trivially: a genuinely wrong dual update (`y += ρ(t−z)` rather than
`y = ρ(t−z)`) matched OSQP to 2.9e-15 across 60 solves, and diverges immediately once bounds are
tightened. The same blind spot hid the ρ-reciprocal defect (CLAUDE.md).

**Requirement: every validation of an ADMM row must also run a tight-bound variant with a real
active set** (10–19 active constraints is what the harness now uses). This is a defect in the
benchmark as frozen, not in the solver — a constrained-optimization benchmark whose constraints
never bind is not testing constrained optimization.

Not a re-spec: the nominal row stays, because an interior optimum is a legitimate operating point
for a hovering quadrotor. The tight-bound variant is an addition, not a replacement.

## Kill criterion

> Solver workloads (#1–#5) must show **≥5× perf/W** versus the best CPU baseline.
> If they do not, the premise is wrong and the project stops.

Secondary gates, not kill criteria but material to the product definition:
- Any benchmark where a CPU **already meets the deadline** must be sold on energy or
  consolidation, never on feasibility. Flagged automatically in the model output.
- Any benchmark that is **bandwidth-bound** rather than compute-bound reveals a provisioning
  requirement, not an architecture win. Also flagged automatically.

## Measurement protocol (Phase 0 completion criteria)

1. **Model** each benchmark analytically — done, v0.
2. **Measure the CPU baselines for real** on hardware (OSQP / acados / CasADi for #1–#3,
   hand-written EKF for #4–#5, llama.cpp for #6, ONNX Runtime for #7). Replace the modelled
   CPU numbers; the model's CPU-side efficiency assumption (25–35% of SIMD peak on small
   matrices) is the single largest source of error in the whole comparison.
3. **Re-run** the go/no-go with measured baselines.
4. Publish the harness so results are reproducible (`argmind-benchmarks`, per §7.1 of the plan).

## Explicitly out of scope for Phase 0

Training, batched inference, >4B-parameter models, sparse/irregular graph workloads,
multi-CM scaling, fixed-point-only microcontroller-class targets. Each may return later;
none may influence Phase-0 architecture decisions.

**Added v1.2 — explicit MPC is conceded, not modelled.** Offline piecewise-affine MPC with online
point location and table lookup is a real deployed answer in this socket and it is a *memory*
workload, not a compute one: there is no arithmetic to accelerate. We should say so when asked
rather than pretend the CM competes there.
[Reference](https://www.mathworks.com/help/mpc/ref/explicitmpccontroller.html).
