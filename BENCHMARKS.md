# Argmind CM — Frozen Phase-0 Benchmark Set (v1.0)

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
| 2 | **ARM-MPC** | solver | 7-DoF manipulator nonlinear SQP-MPC, 14 states, 7 inputs, N=30, 60 iterations, **re-linearizes and re-factorizes every call** | 500 Hz / 2000 µs | Hardest solver case; the re-factorization is genuine matrix-matrix work, unlike #1 |
| 3 | **PWR-MPC** | solver | Motor-drive / power-electronics MPC, 4 states, 3 inputs, N=10, 25 iterations | **20 kHz** / 50 µs | Latency extreme — tests whether per-call overhead is survivable at all |
| 4 | **EKF-24** | solver | INS/GNSS 24-state EKF, 12 measurements, Joseph-form covariance update | 1 kHz / 1000 µs | Deliberately the smallest problem — expected to be a CPU win or a tie. Defines the floor |
| 5 | **EKF-96** | solver | Large-state estimator (SLAM-class), 96 states, 24 measurements | 200 Hz / 5000 µs | Covariance recursion is matrix-matrix → tests the tile array on solver work |
| 6 | **SLM-DECODE** | ML | 1.1B-parameter SLM (TinyLlama-class), INT8, batch 1, autoregressive decode, 22 layers, hidden 2048, context 512 | 20 tok/s / 50 ms | The edge-LLM claim. Batch-1 decode is matrix-vector and weight-bound |
| 7 | **VIT-TINY** | ML | ViT-Tiny, 224×224, INT8, 1.26 GMAC/inference, 5.7 MB weights | 30 fps / 33.3 ms | Incumbent NPU home turf. We need to know exactly how much we concede here |

## Baselines every benchmark is measured against

| Baseline | Configuration | Role |
|---|---|---|
| **Cortex-A55-class** | 1.5 GHz, in-order, NEON (8 fp32 FLOP/cyc), 16 B/cyc L1, ~0.15 W | The honest **perf/W** adversary — what cost-sensitive robotics actually ships |
| **Cortex-A78-class** | 2.0 GHz, OoO, NEON (16 fp32 FLOP/cyc), 32 B/cyc L1, ~0.9 W | Mainstream application core |
| **Cortex-A720-class + SME2** | 2.2 GHz, 256-bit streaming (64 fp32 FLOP/cyc), 64 B/cyc L1, ~1.3 W | The **future** competitor — CPU matrix extensions are our highest-probability real rival |

Comparison rule, conservative on both axes: **speedup** is measured against the *fastest*
baseline, **perf/W** against the *most energy-efficient* baseline. Solver working sets are
assumed cache-resident on the CPU (favourable to the CPU, and true at these sizes).

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
