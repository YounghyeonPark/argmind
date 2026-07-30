# Argmind CM — Phase 0 Go/No-Go Decision Memo

**Version 1.0 · 2026-07-30 · Author: Yoo Park (founder)**

**Audience.** The founder, deciding whether to commit the next twelve months; and secondarily a
technical advisor or angel investor the founder hands it to. It assumes a competent, skeptical
reader who has not followed the work.

**Status of this document.** This is the Phase-0 output that PLAN.md §6 names and that did not
exist: the go/no-go. It consolidates PLAN.md, PHASE0_FINDINGS.md (whose §0 supersedes §1 and whose
§2/§3 tables are stale), BENCHMARKS.md v1.1, CUSTOMER_DISCOVERY.md, TESTING.md and the git
correction history. Where those documents disagree with each other, §3.4 below says so and names
which figure is authoritative.

**No number in this memo may be quoted externally.** Almost every quantity here is a model output
with a stated ±2× uncertainty. §2 separates what we have measured from what we have modelled from
what we have read, and that separation is the single most important thing in the document.

---

## Evidence tiers — used as labels throughout

| Tag | Meaning | How much of this memo |
|---|---|---|
| **[M]** | **Measured by us.** Real solver runs on real hardware, with the harness in the repo. | Two things only: OSQP iteration counts and problem structure for QUAD-MPC. |
| **[E]** | **Modelled estimate, ±2×.** Analytical Python model, first-order energy and area constants, some interpolated from published 45 nm data and scaled. Not simulated RTL, not synthesised, not silicon. | Every performance, energy, area, latency and perf/W figure. |
| **[P]** | **Published third-party data or vendor material**, read off the public record. Includes some third-party *measured-silicon* figures (e.g. Apple M4 SME2 throughput) — measured by someone else, on their workload, not ours. Also includes the many cells that are **not published at all**. | The entire competitive picture. |

Three rules that follow, and that this memo obeys:

1. An **[E]** figure never appears without its configuration, process-node class and precision.
2. An **[E]** figure is never compared to a vendor marketing figure. Where a competitor number
   exists only as marketing (the AURIX PPU's "up to 78× vs a single TriCore," Ceva's
   "350 TOPS/W," Arm's Lumex "5× faster AI"), it is recorded as uncomparable and not used.
3. Where we have not measured something, the sentence that would otherwise imply we had says so
   instead.

---

## 1. What Phase 0 set out to test

### 1.1 The product, in plain language, and the category

Argmind CM is **licensable synthesizable SystemVerilog IP** — a block a chip company instantiates
inside its own SoC or MCU — that runs a complete iterative numerical algorithm, including the
algorithm's own control flow, without the host CPU being involved once per iteration. The target
algorithms are convex and nonlinear optimization solvers (ADMM/OSQP, interior point, SQP), model
predictive control, and Kalman/extended-Kalman state estimation, plus small-matrix machine-learning
inference on the same datapath. The intended buyer is an automotive or industrial semiconductor
vendor, or a Tier-1 with a captive ASIC group, shipping control silicon.

We are coining a category name for this, **Computational Module (CM)**, so it needs defining and
bounding, because a coined category that assumes its own name explains itself dies:

- A CM **is** a sequencer with its own instruction stream, tightly coupled to a matrix/vector
  datapath and a software-managed banked scratchpad, delivered with (a) an explicit contract for
  what data stays resident between calls, (b) a coherent zero-copy host data path, and (c) a code
  generator that takes a problem written in CVXPY/CasADi/acados and emits kernels. The software and
  the residency contract are part of the definition, not accessories to it.
- A CM is **not an NPU.** An NPU executes a command stream compiled offline from a static dataflow
  graph. None of the four commercial NPU IP families we tore down publishes any mechanism for the
  accelerator to decide "iterate again" from a value it just computed [P]; Cadence's answer is to
  require a second licensed control core [P].
- A CM is **not a compute module or system-on-module.** Jetson, Auterion Skynode and the like are
  purchased boards. A CM is RTL that goes inside a chip. The naming collision is real and worth
  pre-empting in every external document.
- A CM is **not a DSP.** A DSP is a general programmable core that happens to have a vector unit.
  The distinction is the residency/host contract and the solver codegen, not the presence of a
  scalar core next to a vector unit.

That last boundary is the fragile one, and §4 deals with it honestly: the Infineon AURIX TC4x PPU
is a scalar core plus a SIMD vector unit plus a closely-coupled memory running its own instruction
stream [P]. Something very close to our category definition already ships in an automotive MCU.

### 1.2 The mechanisms Phase 0 was testing, and why each was supposed to matter

Stating mechanism before benefit, because a reader who does not believe the mechanism should not
believe the benefit:

**Sequencer coupling.** A solve is 15–20 ADMM iterations in the warm-started steady state and
60–74 on a cold solve [M], each iteration a few microseconds of work. If the host must be
interrupted, scheduled, and must re-issue work once per iteration, that per-iteration round trip is
comparable to the iteration itself. Running the convergence test in-fabric removes it entirely.
This is why the claim was ever about latency per iteration rather than throughput.

**Zero-copy plus call-to-call residency.** An MPC controller re-solves a structurally identical
problem every tick. For QUAD-MPC the persistent operand set is 100 KB [E], while the per-tick
change is the new state and reference — order a few hundred bytes. If the operands are resident and
the host writes only the delta through a mapped scratchpad window, the per-call transfer is ~1 kB
rather than ~100 kB of re-upload plus cache maintenance [E, byte accounting from the benchmark
definitions]. The mechanism is a generation tag per buffer plus a dirty mask in the call descriptor;
re-fetch only what changed.

**Bandwidth-first provisioning.** This one is arithmetic, not modelling, and it is the most robust
technical conclusion of the phase. At batch 1 the ADMM inner loop (`P@x` folded into a cached
triangular solve, `Γ@x`, `Γᵀ@y`) is matrix-**vector**: every multiply-accumulate consumes a fresh
matrix element that is never reused. So one operand element must be fetched per MAC. At fp32, 64
bytes/cycle of scratchpad read bandwidth therefore caps the useful rate at 16 MAC/cycle, and 128
bytes/cycle at 32 MAC/cycle, *regardless of how many multipliers exist*. A 144-MAC array at 800 MHz
is oversubscribed roughly 4.5× against what a 128 B/cycle scratchpad can feed on this workload. The
correct part is therefore a wide, heavily banked scratchpad with a modest array — the opposite of
NPU provisioning, whose entire area budget sits in an array that batch-1 solver code leaves idle.

### 1.3 The pre-registered kill criterion, quoted exactly

PLAN.md §6:

> **Kill criterion:** if modeled advantage on the solver workloads is <5× perf/W vs a Cortex-A +
> NEON at same node, rethink.

BENCHMARKS.md restates it as: solver workloads #1–#5 must show **≥5× perf/W versus the best CPU
baseline**, with the comparison rule that speedup is measured against the *fastest* baseline and
perf/W against the *most energy-efficient* baseline, and with solver working sets assumed
cache-resident on the CPU (deliberately favourable to the CPU).

**Two honest caveats about our own criterion, which a skeptical reader will find:**

1. **"At same node" is not satisfied.** The three CPU baselines are specified by microarchitecture
   class, clock, SIMD width, L1 bandwidth and power (Cortex-A55-class 1.5 GHz / NEON / 16 B/cyc /
   ~0.15 W; A78-class 2.0 GHz / 32 B/cyc / ~0.9 W; A720-class + SME2 2.2 GHz / 64 B/cyc / ~1.3 W).
   **No process node is attached to any of them**, while the CM side is costed at 16 nm-class. The
   criterion as written cannot be evaluated at same node until the CPU side is measured on named
   parts. This is a defect in the criterion, not a result.
2. The criterion is a **modelled** threshold. It was always meant to be re-run against measured CPU
   baselines (BENCHMARKS.md §"Measurement protocol", step 3). That step has not happened.

Alongside the kill criterion, Phase 0 was to produce: a frozen benchmark set (done, BENCHMARKS.md
v1.1), a performance model (done, `perf_model/`), a competitive teardown (done, desk research), and
**15–20 customer-discovery conversations (not done — zero)**.

---

## 2. What is now established, and with what confidence

### 2.1 Tier [M] — measured. This is all of it.

Harness `perf_model/measure_osqp_baseline.py`, OSQP 1.1.3, run on an x86_64 host. **The timings do
not transfer to Arm and are not used as a baseline anywhere.** What transfers is iteration counts
and problem structure, and both had previously been assumed.

| Measured quantity | Result | Why it matters |
|---|---|---|
| Constraint rows, real OSQP form, QUAD-MPC | **320** (80 input box + 240 predicted-state box) | The model had assumed 640. OSQP expresses `l ≤ Ax ≤ u` as one row. Inflated `A@x`/`Aᵀ@y` in every solver row. |
| Condensed Hessian structure, physically coupled plant | `nnz(P)` = 6400, **density 1.00** | A decoupled toy plant gives a 4-block-diagonal 25%-dense Hessian nobody would solve as one 80-variable QP. The dense case is the real one. |
| ADMM iterations, warm-started, aggressive scenario, tol 1e-3 | p50 **6**, p95 **7**, p99 9 | The model had assumed 40. Steady-state work is far smaller than assumed. |
| Same, tol 1e-6 | p50 **20**, p95 **21**, p99 22 | Bounds the accuracy/work trade; `iters=15` is a defensible design point between the two. |
| **Cold solve (no warm start)** | **60** iterations at 1e-3, **73–74** at 1e-6 | **This is the real-time requirement.** Any mode change, fault recovery or reference discontinuity invalidates the warm start, and the deadline must survive the next tick. |
| Iteration-count spread on one unchanged problem | p50 12 → max 60 at 1e-6 (settle scenario) | **A 5× swing driven by disturbances and active-set changes.** This is the measured form of the determinism argument. |

Two conclusions rest directly on this measurement and are therefore the best-supported conclusions
we have:

- **Size against the worst-case iteration tail, never the typical one.** At p95 iterations every
  solver benchmark meets its deadline at 16 B/cycle, the narrowest point in the entire sizing grid
  [E]. A sweep run on typical iterations cannot produce a binding constraint and therefore cannot
  justify a configuration — it can only ratify one. The measured cold tail is what produces a floor
  at all.
- **The determinism pitch is "our worst case is bounded and close to our average, and a CPU's is
  not"** — not "we are faster." That is measured, and it is the only pillar of the value
  proposition with any measured support behind it.

**What is [M] and is *not*:** the harness took three passes to get right, and the first two were
wrong in ways that flattered us (OSQP's default `check_termination=25` quantises reported iteration
counts to multiples of 25, which produced a suspiciously flat "25 iterations" across four tolerance
decades; the decoupled plant above; and an "aggressive" scenario that was physically infeasible and
drove OSQP primal-infeasible, reporting p95 ≈ 900 iterations that measured nothing but struggle).
All three are fixed and the harness now discards any run with non-`solved` ticks. Recording this
because the pattern is the point, and it recurs in §3.

### 2.2 Tier [E] — modelled at ±2×. Everything quantitative below is in this tier.

**Configuration for every figure in this subsection — CM-100 revised baseline, costed 2026-07-30:**
128 KB scratchpad in 16 single-port banks × 8 B/cycle = 128 B/cycle; 144 MACs in a 12×12 array,
bf16 multiply with fp32 accumulate; 16 fp32 vector lanes; 800 MHz; **16 nm-class**; area
0.492 mm² + 15% top-level margin = **0.565 mm²**. Per-benchmark precision is fp32 for QUAD-MPC,
PWR-MPC and both EKF rows, **fp64** for ARM-MPC (interior point demands it), INT8 for the two ML
rows. **Modelled estimate, ±2×, from an analytical Python model — not simulated RTL, not
synthesised, not silicon. The fp64 datapath area is zero in this budget because it has not been
costed.**

| Benchmark (BENCHMARKS.md v1.1) | Deadline | Latency, p95 iters | Latency, worst-case iters | Margin at worst case | Energy/call | perf/W vs most efficient modelled CPU baseline | Binding resource |
|---|---|---|---|---|---|---|---|
| QUAD-MPC (quadrotor MPC, fp32, 1 kHz) | 1000 µs | 34.1 µs | 161.5 µs | 6.19× | 5.31 µJ | 14.5× | scratchpad bandwidth |
| ARM-MPC (7-DoF NMPC, acados SQP-RTI, fp64, 250 Hz) | 4000 µs | 177.9 µs | 338.5 µs | 11.82× | 19.58 µJ | 14.1× | MAC array (Riccati) |
| PWR-MPC (motor drive MPC, fp32, 20 kHz) | 50 µs | 4.2 µs | 14.3 µs | 3.50× | 0.35 µJ | 15.6× | scratchpad bandwidth |
| EKF-24 (INS/GNSS, fp32, 1 kHz) | 1000 µs | 2.6 µs | 2.6 µs | 388× | 0.19 µJ | 20.9× | MAC array |
| EKF-96 (SLAM-class, fp32, 200 Hz) | 5000 µs | 65.4 µs | 65.4 µs | 76× | 8.22 µJ | 22.2× | MAC array |
| SLM-DECODE (1.1B SLM, INT8, batch 1) | 50 ms | 174 ms | 174 ms | **0.29× — MISSES** | 17.6 mJ | 2.4× | external DMA bandwidth |
| VIT-TINY (ViT-Tiny, INT8, 30 fps) | 33.3 ms | 7.6 ms | 7.6 ms | 4.39× | 660 µJ | 26.8× | MAC array |

**Solver perf/W band: 14.1–22.2× [E, ±2×], against the Cortex-A55-class modelled baseline
(1.5 GHz, NEON, in-order, ~0.15 W, node unspecified) for the MPC rows and the A720-class + SME2
modelled baseline for the EKF rows.** Across the scratchpad-energy sensitivity band the figure
moves to roughly 9–26×.

**So the kill criterion passes — with an important qualification about how much margin it really
has.** On the central estimate, 14–22× against a 5× threshold is a 2.8–4.4× margin. Applying the
model's own ±2× symmetrically, the pessimistic edge is ~7×, still above 5×: the criterion survives
its stated uncertainty. But the pessimistic edge of the scratchpad-energy sensitivity band (9×)
combined with the pessimistic edge of ±2× lands at **~4.5×, below the criterion.** And the largest
single error source in the ratio — the CPU side's assumed 25–35% of SIMD peak on small matrices —
has never been measured, and an error there does not cancel. The honest statement is: **the kill
criterion passes on modelled data with real margin, and has not yet been evaluated on measured
data at all.**

Other [E] conclusions, in descending order of how much I would bet on them:

| Conclusion | Confidence and why |
|---|---|
| **Batch-1 solver inner loops are bandwidth-bound, not MAC-bound.** 16× more MACs bought exactly zero on QUAD-MPC; 16→256 B/cycle bought ~9×. | **High.** It follows from operand reuse being zero, which is a property of the algorithm, not of our energy constants. Encoded as a regression test. |
| **64 B/cycle is the latency floor**, set by the measured ADMM cold tail inside a 50%-of-period engineering budget: at 16 B/cycle QUAD-MPC cold misses outright (1092 µs > 1000 µs); at 32 B/cycle it blows the 50% budget (560 µs > 500 µs) as does PWR-MPC (33 µs > 25 µs); at 64 B/cycle both fit (295 µs, 21 µs). | **Medium-high**, but note the load-bearing assumption stated in the open: `PERIOD_BUDGET = 0.50`. At 100% of period, 32 B/cycle suffices. |
| **Nothing in the frozen set needs more than 36 MACs to meet a deadline** (VIT-TINY 36; every solver row ≤ 16). The array is sized for *energy*, not latency: tile dimension is operand-reuse depth ((1/Tm + 1/Tn) operand elements per GEMM MAC), so 64→256 MACs cuts EKF-96 energy 1.59× and EKF-24 1.54× while doing exactly nothing (1.00×) for the pure-GEMV ADMM rows. | **Medium.** The reuse-depth mechanism is sound. The adversarial objection is unresolved: growing the array buys reuse at O(T²) area, whereas an L0 operand register file buys the same traffic cut at O(T) area. The model cannot yet evaluate a tiered scratchpad. **If an L0 tier lands, the honest array size drops to 36–64 MACs.** |
| **Power gating is an architectural requirement, not an optimization.** Solver duty cycles are 0.3–11.7%. Charged ungated, QUAD-MPC perf/W falls 11.6× → 3.7×, and EKF-24 19.3× → **0.3×** — below the kill criterion. | **High as a conclusion, medium as a number.** The 15 mW static figure is itself an estimate, and it is now the most load-bearing single estimate in the model. |
| **Banking beats multi-porting**: an 8T dual-port cell is ~2.1× the 6T single-port area, so two read ports on 256 KB costs ~0.582 mm² versus 0.319 mm² for 16 single-port banks. Design rule: **bank word ≥ 64 bits**, below which no published energy data exists and the model becomes unphysical. | **High.** Rests on a cited measured area ratio, not on an interpolation. |
| **128 KB suffices, and dominates 256 KB.** Corrected footprints: QUAD-MPC 100 KB, EKF-96 117 KB, ARM-MPC 172 KB, PWR-MPC and EKF-24 8 KB each. The determinism claim is "no DRAM in the control loop," so it must be tested per co-resident workload: drone node (QUAD-MPC + EKF-24, 1 kHz) 108 KB fits; drive node (PWR-MPC + EKF-24, 20 kHz) 16 KB fits; manipulator (ARM-MPC + EKF-96, 250 Hz) 289 KB fits neither 128 KB nor 256 KB. | **Medium-high.** 256 KB does not extend the claim to the case that fails; it only costs 0.116 mm² and doubles leakage (0.52 → 1.04 mW at 85 °C). The area and leakage halves of the argument do not depend on the energy model at all. |
| **The wider 128 B/cycle configuration is an energy bet, not a feasibility requirement**: −11.6% solver energy for +0.043 mm², of which 3.49 µJ of the 4.42 µJ delta is pJ/byte falling 0.884 → 0.718 — i.e. **79% of it is bank geometry (16 banks of 8 KB being cheaper per access than 8 of 16 KB), not bandwidth.** The byte count fetched is identical at both widths. | **Low-medium, and explicitly a bet.** It is an interpolation of a single published 45 nm energy table scaled to 16 nm by one 0.30 factor. The 128 KB energy claim rests on the *same* interpolation, so the two are correlated, not independent evidence. If a foundry compiler datasheet shows flat pJ/byte versus access width, drop to the 64 B/cycle floor. |
| **Zero-copy and residency pay on small problems at high rates, and are nearly irrelevant on large ones** — order 10× on EKF-24, order 2–5× on PWR-MPC where a copy-based path misses the 50 µs deadline, ~1.1× on the large and ML rows. | **Medium on the shape, low on the specific ratios** — see §3.4, the ablation has not been regenerated at the revised configuration and two documents quote different numbers. Sell it as a real-time-control feature, not a general one. |
| **SLM-DECODE misses by 3.4× on external bandwidth** (~22 GB/s needed, 6.4 GB/s provisioned). | **High**, and it is a provisioning requirement rather than a design flaw. **Consequence: do not claim general edge-LLM capability.** Scope the ML claim to resident-weight models until a wide external port exists. |

### 2.3 Tier [P] — desk research on published specifications

Fully cited in PHASE0_FINDINGS §5/§5.1. The load-bearing content:

| Finding | Class |
|---|---|
| **All four commercial NPU IP families compile data-dependent control flow away offline.** None publishes a mechanism for the accelerator to decide "iterate again" from a computed value. Cadence's Neo NPU requires a second licensed core (Tensilica NeuroEdge 130) as control processor. | [P] vendor documentation, four independent families |
| **None has usable FP matrix throughput.** Ethos-U85 is INT8-weights-only; Ceva NeuPro-M's FP32 path is 32 MACs against 4K INT8, ≈1%. **So NPU IP cannot add solver support in a software release — it lacks the arithmetic.** | [P] vendor documentation |
| **Batch-1 / small-tensor latency is universally unpublished**, across NPU IP *and* the control-domain incumbents. No FLOP/s of any kind is published for the AURIX PPU or the TI C2000 CLA, and no measured FP32 GFLOP/s exists publicly for any RVV silicon. | [P] absence of publication |
| **Arm SME2, measured by third parties on Apple M4:** 2008 GFLOP/s FP32 FMOPA, SVL 512-bit, P-core cluster, **3 nm class; per-unit power not published.** The SME block is **shared per CPU cluster**, not per core. | [P] third-party measured silicon, two independent sources |
| **SME2 is not buyable in an automotive or industrial part today.** Cortex-A320 is published with SVE2 and Arm does not state SME2; the A720AE and A78AE pages do not state SME2. Record as *not published*, not as absent, since SME is optional in Armv9.2-A. | [P] vendor pages / not published |
| **RISC-V matrix extension is not ratified as of 2026** and is fragmented across two task groups (IME, VME), with RISC-V's own TSC minutes recording ecosystem-fragmentation concern and that matrix extensions are behind schedule. | [P] standards-body minutes |
| **The AURIX TC4x PPU is a licensed Synopsys ARC EV71/EV7x**: 32-bit RISC scalar core + SIMD vector unit + vector closely-coupled memory, running its own instruction stream and its own ISA. Automotive-qualified, supported by MathWorks Embedded Coder, EV7xFS variants marketed ASIL-B/C/D-Ready. **Whether the optional IEEE-754 vector FPU is instantiated in TC4x is not published. VCCM size, inter-invocation persistence, host coherency, throughput, clock and node are all not published.** | [P] secondary (emmtrix) corroborated by the existence of a Synopsys MetaWare toolkit for AURIX; plus vendor family documentation |
| **The only measured-silicon efficiency reference in the whole competitive picture** is MediaTek's NVE at ISSCC 2024: 23.2 TOPS/W at 12-bit in 3 nm. Vendor claims run far above it (Ceva 350 TOPS/W "at 3 nm," Synopsys 30 TOPS/W at 5 nm) but are pre-silicon estimates at low or unstated precision, usually with sparsity. | [P] peer-reviewed vs. vendor claims |
| **In-memory computing wins on stationary-weight low-precision inference and we concede it.** Published macros: 3 nm 12b digital SRAM-CIM 23.2 TOPS/W; 28 nm hybrid-domain FP SRAM-CIM 72 TFLOPS/W; 65 nm analog CIM 33.6–136.2 TOPS/W. **Argmind must never claim to beat CIM on inference TOPS/W.** Our answer is the MACIF swappable-macro boundary, so a CIM macro is adoptable rather than fatal. | [P] ISSCC / arXiv 2024–25 |

The corresponding IMC concession has a real counterpart in our favour, and it is device physics
rather than an engineering gap: analog IMC amortises a slow, energy-expensive weight write over
many matrix-vector products, and state-of-the-art "resilient" ReRAM tolerates ~10⁵ update pulses
[P] — a 1 kHz controller re-linearising its model every tick exhausts that in about 100 seconds.
Add ~4-bit practical device resolution against interior point's ill-conditioning near convergence,
and data-dependent latency governed by the problem matrix's minimum eigenvalue in analog matrix
computing. That is why the solver half of the pitch is durable against IMC and the inference half
is not.

---

## 3. What was corrected during Phase 0, and why it cuts both ways

### 3.1 The ten modelling errors

Each is now encoded as a named regression test in `tests/test_model_invariants.py`, and
`tests/test_guards_actually_bite.py` mutates the model back to each buggy formulation and asserts
the guard trips.

| # | Error | Direction of bias | Effect when fixed |
|---|---|---|---|
| 1 | Energy counted only the multiplier-accumulator; clock distribution, register files, operand staging and control omitted | **Flattered us** | Removing overhead energy inflates perf/W by >30% (measured by the mutation test) |
| 2 | Matrix-vector work modelled as matrix-matrix | **Flattered us** | Made a 16× larger array look 5%+ faster on GEMV-bound work; this is the error whose fix produced the bandwidth-first conclusion |
| 3 | **Phantom transcendental term** — ADMM's box projection `Π_[l,u]` charged as 2·n_con divide/sqrt operations when it is min/max, zero divides. The CPU model charges 12 cycles per transcendental against the CM's 4 per vector lane, so ~640 fake divides per iteration penalised the CPU ~3× harder than the CM | **Flattered us, decisively** | **This single error accounted for the entire apparent latency advantage.** QUAD-MPC went from 1.47× to 0.92× — slower than the fastest CPU baseline |
| 4 | Two-sided bounds `l ≤ Ax ≤ u` counted as two constraint rows | **Flattered us** (inflated both sides' work, but inflated absolute latencies and the deadline story) | 320 rows, not 640; cross-checked against a measured OSQP instance |
| 5 | `P@x` double-counted — it lives inside the cached factor and is not a separate matvec | **Flattered us** | Per-iteration GEMV term corrected to a hand-derivable closed form |
| 6 | Constraint matrix A treated as dense when its input-box rows are the identity and cost zero MACs | **Flattered us** | Same |
| 7 | Scratchpad energy a flat 0.40 pJ/byte, omitting the bank→datapath distribution network | **Flattered us by 2.3×** | This is the correction that moved solver perf/W from 17–25× to the current band |
| 8 | The SQP re-factorisation term formed `AᵀA` — which OSQP explicitly does not do and no production MPC stack does | **Flattered us, then hurt us** | ARM-MPC was overstated ~23–31×. Removing it destroyed the *justification* for the previous CM-100 sizing |
| 9 | Once-per-call assembly of `q` and the state-dependent bounds omitted — it depends on the new `x0`, cannot be warm-started, and sits on the critical path before iteration 1 | **Flattered us** | Now counted |
| 10 | MAC count and operand-reuse depth were the same number (the model used `2/tile`, correct only for a square array) | **Flattered us, and misattributed the cause** | The energy benefit of a bigger array had been attributed to the MACs rather than to reuse depth; correcting it is what exposed the L0-register-file objection |

Separately, three **measurement-harness** errors are recorded in §2.1, all three of which would also
have flattered us.

### 3.2 The pattern that matters more than any individual error

Twice, a sizing recommendation rested on a single benchmark row that nobody had audited.

The first CM-100 recommendation (128 B/cycle, 256 MACs, 256 KB, 0.820 mm²) was justified on two
grounds: that ARM-MPC was the only latency-binding benchmark, and that its working set was 861 KB.
Error #8 destroyed the first ground and made the second wrong by 5× (172 KB, not 861 KB). Neither
agent working on it could see the problem, because one owned the benchmark and the other owned the
sizing. Re-run against the corrected set, the revised part is **0.565 mm², 31% smaller, at slightly
lower solver energy — it dominates the old recommendation on both axes.** The old point was simply
off the Pareto front; there was no trade being made.

Then the re-run's own sweep turned out to have the same structural defect in a different place: run
at typical (p95) iterations, every cell of the 5×4 grid met every solver deadline, tightest margin
3.2×. A sweep that cannot produce a binding constraint cannot justify a configuration. The fix was
to implement `iters_worst` (referenced in the model's comments for weeks but never implemented) and
to add two tests: worst-case iteration tails must exist **and must bind somewhere**, and GEMM
energy must track reuse depth rather than MAC count.

### 3.3 Both readings, stated together, because only stating one would be dishonest

**The favourable reading.** The review process works. Ten errors were found and fixed inside one
phase; seven of them were found by adversarial audit rather than by the person who wrote the code;
each is now a regression test, and the meta-tests prove those guards actually bite rather than
merely existing. The model has been made *able to fail*: there is a test asserting some benchmark
misses its deadline, and another asserting the CPU wins somewhere. Two conclusions survived every
correction unchanged — bandwidth-first provisioning, and the kill criterion — which is meaningful
precisely because so much around them moved.

**The unfavourable reading, which is the one to act on.** Every single one of the ten errors, plus
all three harness errors, biased in our favour. That is not a coincidence and it is not bad luck;
it is what happens when the person building the model wants a particular answer. The base rate for
"a number in this corpus that has not yet survived an adversarial read is wrong in our favour" is,
on the evidence of this phase, high. **Therefore: no figure in this memo should be treated as
settled merely because it currently appears in a document.** The specific numbers most likely to
move next are the ones §2.2 marks low-confidence — the scratchpad pJ/byte interpolation, the 15 mW
static power estimate, the uncosted fp64 datapath, and the entire CPU side of every ratio.

### 3.4 Stale and mutually inconsistent figures currently in the corpus

Found while writing this memo. These are flagged rather than quietly fixed, because an
unsupportable claim someone else has already believed is a problem that needs to be known about.

| Claim, and where it appears | Problem | Authoritative version |
|---|---|---|
| Solver perf/W "17–25×" — PHASE0_FINDINGS §1, §7, and **CUSTOMER_DISCOVERY §4.2 Q-B4** | Superseded by corrections #3, #5, #6, #7 | **14.1–22.2× [E, ±2×] at the revised CM-100 config.** §1 of PHASE0_FINDINGS is marked superseded; **CUSTOMER_DISCOVERY is not, and it is the document that will be read aloud in live customer calls** |
| Latency advantage "1.2–2.9×" (PHASE0_FINDINGS §1), "0.6–2.9×, below parity on 3 of 5" (§0), "~0.9–3.2×" (CLAUDE.md) | Three different ranges in three places. The current model output at the **superseded CM-16 configuration** (256 MACs, 256 KB) gives 0.9 / 1.4 / 0.6 / 1.3 / 3.2 — a range of **0.6–3.2×, below parity on two of five** (QUAD-MPC 0.9×, PWR-MPC 0.6×), not three of five. **No latency ratio has been regenerated at the revised CM-100 configuration at all.** | Until regenerated, the only defensible statement is: **at or below CPU latency on the flagship benchmark, and roughly parity to ~3× elsewhere, at a configuration we no longer recommend.** I have softened "below parity on three of five" because the model output does not support the count |
| Zero-copy ablation: "EKF-24 13.3×, PWR-MPC 1.9×" (PHASE0_FINDINGS §4) vs "EKF-24 12.0×, PWR-MPC 4.8×" (`results_v0.txt`) | Two different runs, neither at the revised configuration | Use the shape, not the ratios: order 10× on the smallest/highest-rate rows, ~1.1× on large rows. Regenerate before quoting |
| "A CPU already meets 3 of 5 solver deadlines" — PHASE0_FINDINGS §1 and **CUSTOMER_DISCOVERY §1** | Superseded | **A CPU meets 5 of 5 solver deadlines** in the frozen set [E]. This is a bigger concession than the documents currently make, and it needs to be the one we say out loud |
| PHASE0_FINDINGS §2/§3 latency tables (QUAD-MPC 431.8 µs, ARM-MPC 3745 µs, etc.) | Pre-correction absolute numbers | Superseded by §0 and by `results_cm100_sizing.txt`. The *relative* conclusion in §2 (MACs buy nothing, bandwidth buys everything) is unaffected |
| CUSTOMER_DISCOVERY §8.2 cold-email template offers "my full benchmark set and modelled results ... with process node and precision attached" | **That artefact does not exist in a publishable form.** Everything we have is ±2× and non-quotable | Either build the publishable artefact before sending the email, or change the email. Sending it as-is creates an obligation we cannot meet without violating our own evidence standard |
| TESTING.md "nine modelling errors" | Ten are now encoded (`ERROR 10` = reuse depth vs MAC count) | Ten |
| TESTING.md Gap 1, "there is no git repository" | Fixed 2026-07-30 | Stale |

---

## 4. What the moat actually is, after narrowing

The teardown narrowed the moat twice, and the second narrowing is the one that matters. **Know
which differentiator applies to which rival** — asserting the wrong one to the wrong audience is
how a technical credibility is spent in a single meeting.

| Rival class | Sequencer / in-fabric control flow | Coherent zero-copy | Call-to-call residency contract | What actually survives |
|---|---|---|---|---|
| **Commercial NPU IP** (Arm Ethos-U85/N78, Synopsys ARC NPX6, Ceva NeuPro-M, Cadence Neo) | **Real structural gap** — all compile control flow away offline; Cadence needs a second licensed core | **Real gap** — only Ethos-N78 has ACE5-Lite + SMMU | **Real gap** — not published, or fixed-function buffering | All four differentiators, **plus** the decisive one: they have no usable FP matrix arithmetic, so they cannot close this in a software release |
| **CPUs with matrix/vector extensions** (Arm SME2, RISC-V RVV) | **Not a differentiator. It is the CPU.** | **Not a differentiator** — by construction | **Not a differentiator** — the CPU's caches, implicitly | **Energy per solve, deterministic worst-case latency, core offload, and the codegen/software ecosystem. Nothing else.** |
| **MCU-class control coprocessors** (TI C2000 CLA) | They already have it (`MBCNDD` conditional delayed branch) | **Void** — no DRAM, no cache, nothing to eliminate | **Void** — an all-SRAM MCU has nothing to evict | Arithmetic throughput (their CLA is a *scalar* FP datapath at 200 MHz), and energy |
| **Automotive MCU vector DSP** (Infineon AURIX TC4x PPU = licensed Synopsys ARC EV7x) | They have it — own instruction stream, own ISA | **Not published** | **Not published** | **Possibly nothing.** See below |

Three consequences, stated plainly.

**First, against CPU matrix extensions three of our four differentiators do not exist at all.**
Because SME2 and RVV are *inside* the CPU, they get data-dependent control flow, coherent zero-copy
access to application memory, and call-to-call cache residency for free, by construction. The
sequencer, the T1 coherent port and the generation-tag residency model differentiate against
**accelerators, not against CPUs.** Worse, our own §2 finding cuts against us here: the solver inner
loop is GEMV and bandwidth-bound, and a plain vector unit is the *right* shape for GEMV. So RVV's
missing matrix extension protects the ML half of the pitch and **not the solver half.** What
survives is energy per solve, deterministic worst-case latency, core offload, and the software
ecosystem — and of those, only determinism currently has measured support (§2.1), and only against
CPU *iteration-count* variance, not yet against an Arm part.

One genuinely strong determinism leg does exist and it is architectural rather than incidental: the
M4's SME block is **shared per CPU cluster** (one for four P-cores) [P, third-party measured], so a
thread's worst-case latency depends on what sibling cores are doing. A part that cannot bound its
own WCET independently of co-runners is a poor fit for safety-relevant control, and that is a
structural property, not a tuning problem.

There is also a timing observation that must not be mistaken for a moat: **a robotics, industrial
or automotive designer cannot buy an SME2-bearing automotive-grade or industrial part today** [P,
with the caveat that Arm's automotive-core pages say *nothing* about SME2, which is "not published,"
not "absent"]. Bringing SME2 to automotive cores is a routine Arm roadmap item, not a research
problem. **That is a window, and windows close.**

**Second, the AURIX PPU is the biggest unhedged risk in the entire technical case.** It is a
licensed Synopsys ARC EV71/EV7x: a scalar RISC core plus a SIMD vector unit plus a closely-coupled
memory, running its own instruction stream — a small version of our own architecture, already
automotive-qualified, already in a shipping MCU, already supported by MathWorks Embedded Coder [P].
If Infineon instantiated the EV7x's optional IEEE-754 vector FPU, then the PPU can in principle run
an ADMM inner loop today with the convergence test executing in-fabric, and **what it is missing is
a solver library, not new silicon.** Whether that FP option is instantiated **is not published**,
and neither are the VCCM's size, its inter-invocation persistence contract, its coherency with
TriCore memory, its throughput, its clock, or its node. Which cuts both ways for us: we cannot
assert those as gaps either. A vision VPU is moreover provisioned to stream pixels, so it is
plausibly *also* bandwidth-first, which makes it a worse comparison for us than an NPU rather than a
better one.

The one clearly favourable reading: **Infineon licensed third-party vector-compute IP for exactly
this socket rather than building it.** That is the best evidence available anywhere that an
automotive MCU vendor will license outside compute IP for control acceleration — and it also makes
Synopsys/MIPS ARC a competitor for the *license*, not merely for the socket.

**Third, there is an unresolved internal contradiction that must be settled before the motor-drive
pitch is made to anyone.** PHASE0_FINDINGS §4 shows zero-copy is what makes the 20 kHz PWR-MPC
deadline feasible at all, and motor drive is the socket most likely to be MCU-class — where
zero-copy and residency have nothing to bite on, because there is no DRAM and no cache. Both
statements cannot be load-bearing at once. Resolve it explicitly (most likely by re-deriving
PWR-MPC's per-call overhead against a tightly-coupled-SRAM host rather than a DRAM host) rather
than letting each document keep its own version.

**What I would therefore write on one line, and defend:** *the moat is energy per solve,
deterministic worst-case latency, core offload, and the codegen/software ecosystem — not
arithmetic, not control flow, not coherency* — with the further honesty that against NPU IP the
gaps are structural and against a CPU they are not, and that the category name has an
already-shipping near-occupant.

---

## 5. What has NOT been established

This section is the one that decides the memo.

**The market case has zero evidence and it is the binding constraint on the decision — not the
technical case.**

| Not established | Current state | Consequence |
|---|---|---|
| **Customer demand, at any level** | **Zero conversations held.** CUSTOMER_DISCOVERY.md is entirely plan: no customer opinion, quote, price, requirement or budget number appears anywhere in the corpus, because none exists. The pre-registered minimum before *any* falsification criterion may be evaluated is **17 substantive 30-minute conversations** (8 demand witnesses, 6 licensees, 3 specifiers) | Nothing about willingness to pay, license structure, safety-package requirements, or whether anyone owns a control-compute power budget is known. **A go decision taken today is a decision taken with no market evidence whatsoever** |
| **Whether energy is a purchase driver** | Untested. Our headline advantage is energy, and we do not know whether anyone buys energy. The strongest disconfirming public fact we already hold: ODYS claims production embedded MPC on 3M+ vehicles on ordinary automotive MCUs with no accelerator [P, vendor claim, volume not independently verified], and a secondary summary reports acados HIL runs holding computation under 10 ms against a 50 ms sampling time [P, **not verified against the primary paper**] | This is pre-registered falsifier **F2**, identified in advance as the most likely and most damaging |
| **Measured Arm baselines** | Not measured: Arm latency, Arm energy per solve, achieved SIMD efficiency on Arm. The model assumes 25–35% of SIMD peak on small matrices, described in its own caveats as "a literature-informed guess" | **This is the largest single error source in every ratio in this memo**, and it does not cancel |
| **Any foundry PPA data** | None. Area from a bitcell-area model calibrated against one published measured 16 nm macro density (8.1 Mb/mm²); energy from one published 45 nm table scaled by a single 0.30 factor; **no foundry memory-compiler datasheet has been consulted**; static power is a 15 mW estimate | The two −10%-class energy claims (128 B/cycle, 128 KB) are the *same bet made twice*, hence correlated rather than additive |
| **The fp64 datapath** | **Area is zero in the CM-100 budget because it has not been costed.** The model obtains fp64 via a 0.125 rate factor on a bf16 array, which is a modelling convenience, not a design. A real 1/8-rate fp64 path needs either multi-pass decomposition (unmodelled sequencer and accumulator work) or a separate narrow fp64 FMA cluster | Largest un-costed item in the area estimate, and it sits exactly on the MACIF boundary. ARM-MPC requires fp64 because interior-point KKT conditioning reaches ~1e12 near convergence |
| **Any RTL** | None written. No cycle-approximate model, no ISA freeze, no MACIF definition | Everything architectural is currently a spreadsheet-grade assertion |
| **The RTL toolchain even runs on this machine** | Verilator, cocotb, Yosys, iverilog, make, Docker and WSL are all absent, and `wsl` reports "not installed." Verilator has no supported native Windows build | The cash plan depends explicitly on open-source EDA. Discovering a Windows-specific obstacle at month 6 with RTL waiting is the expensive version of this |
| **The four decisive competitor cells** | PPU FP option; any FLOP/s for PPU or CLA; measured RVV FP32; energy per solve for any of them. All **not published** | Cannot be resolved by more desk research. Needs boards |
| **Trademark and domain clearance for "Argmind"** | USPTO/EUIPO Nice class 9/42 search and domain acquisition still pending | Cheap, and it gates every external artefact |
| **The open-core strategy** | Untested by any customer. Q-C9 exists to test whether open RTL helps (evaluation without procurement) or hurts (license-compliance review, provenance concerns) | The entire adoption funnel in PLAN.md §7.1 is a hypothesis |
| **Iteration tails for PWR-MPC and ARM-MPC** | Estimated by scaling the measured QUAD-MPC 74/15 ratio. Only QUAD-MPC's tail is measured | The 64 B/cycle floor depends on PWR-MPC's estimated tail as well as QUAD-MPC's measured one |

Three concessions to state in every document, in our own voice, before a reader finds them:

1. **A CPU meets every solver deadline in the frozen benchmark set** [E]. We must never pitch "the
   CPU can't do this," because on this set it can.
2. **We are at or below CPU latency on the flagship benchmark** (QUAD-MPC 0.9× at the superseded
   configuration; not yet regenerated). There is no latency story to tell on QUAD-MPC at all.
3. **In-memory computing wins on stationary-weight inference efficiency** and we will not claim
   otherwise.

---

## 6. The decisions the founder must make

Three, in priority order. Each is stated with what it costs, what it buys, and what it decides.
Note that the first two are cheap and the third is only expensive in founder-time — which is
exactly why none of them justifies deferring.

### D1 — Buy the evaluation boards. Gate: the Synopsys MetaWare toolchain licence price, not the board.

| Platform | Part | Gate | Cash |
|---|---|---|---|
| AURIX PPU (threat #2, F4) | An Infineon AURIX TC4x application kit | **The Synopsys ARC MetaWare Toolkit for AURIX is the gate.** Establish access and price *before* buying hardware | board low; **toolchain unknown, possibly prohibitive** |
| TI C2000 CLA | LAUNCHXL-F28379D (dual 200 MHz C28x + dual 200 MHz CLA) | none — Code Composer Studio is free | very low |
| Arm SME2 | Any Apple M-series Mac. **No automotive or industrial SME2 part exists to buy**, so this is the only measurable SME2 silicon; results are an *architectural upper bound* and Apple remains a non-integration-target | none | low if already owned |
| RISC-V RVV | Banana Pi BPI-F3 (SpacemiT K1 / X60, RVV 1.0, VLEN 256) | none | ~$100 class |
| **Arm CPU baselines** | An A55-class and an A78-class board | none | low |

**What it costs.** Excluding the AURIX toolchain, this is a few hundred dollars of hardware and
roughly two to three weeks of engineering to port BENCHMARKS.md #1–#5 and instrument rail-level
energy. The AURIX toolchain price is unknown and could be prohibitive; **find the number before
buying the board**, and if it is prohibitive, say so in the memo rather than substituting an
opinion about the PPU.

**What it buys.** Two Phase-0 critical-path items with one purchase. (a) It settles falsifier F4 —
does the PPU have an FP path at all, and can any of these parts run our benchmarks within 2× of our
modelled energy per solve. (b) It produces **the measured Arm baselines that every ratio in this
memo depends on.** It also converts our strongest uncontested marketing position — credible
batch-1/small-matrix latency and energy data, which *no* incumbent publishes — from an assertion
into a publishable artefact.

**What must be measured**, specified in advance so the purchase decision is about cost and not
scope: wall-clock per solve and per iteration (median *and* worst case over ≥10⁵ invocations);
**energy per solve measured at the rail**, not inferred from datasheet TDP, with the measurement
point and method reported; achieved FP32 GFLOP/s on the GEMV inner kernel against that platform's
peak; sustained load bandwidth in B/cycle into the compute unit's local memory; whether the
convergence test can run without host involvement, and if not, the per-iteration host round-trip
cost in microseconds — that number *is* the size of our sequencer advantage on real silicon;
worst-case versus median latency with an adversarial co-runner pinned to the same cluster (for SME2
this is the *only* claim we have); and for the PPU, decisively, whether an FP32 kernel compiled with
MetaWare vectorises to an FPU or falls back to scalar emulation.

**Two caveats to carry into anything published from this.** Process node is **not published** for
TC4x, F28379D or the SpacemiT K1, and is only "3 nm class" for M4, so cross-platform energy
comparisons **cannot be node-normalised** and must be reported with nodes attached and that
limitation stated in the same sentence. And F4 triggers if any of these parts, in the same power
class, lands within **2×** of our modelled energy per solve — which, given the model's ±2×, must be
read as *no measurable advantage*, not as a narrow win.

### D2 — Commission one foundry memory-compiler datasheet run.

**What it costs.** A foundry or EDA startup-program relationship plus an NDA; the run itself is a
compiler invocation, not a tape-out. The real cost is the relationship and the time to establish it.
**We do not currently know the price, and establishing it is part of this decision.**

**What it buys.** It settles the single most correlated assumption in the architecture. We need
energy tables at 8 / 16 / 32 B access width and 8 / 16 / 32 KB macro size, at a 16 nm-class node,
plus static power. Specifically:

- If pJ/byte turns out **flat versus access width**, the −11.6% energy case for 128 B/cycle
  collapses toward −2% and **the CM-100 drops to the 64 B/cycle feasibility floor** (0.430 mm² at
  64 MACs, or 0.516 mm² keeping 144 MACs). That is a 24% area saving and it also lets the segmented-
  reduction RTL feature be deferred, since at 64 B/cycle its penalty is only 1.00–1.05×. A real
  schedule saving.
- It also replaces the 15 mW static-power estimate, which now sets ungated perf/W on the low-duty
  rows and is the most load-bearing single estimate in the model.

Both of the model's −10%-class energy claims rest on the same 45 nm interpolation scaled by one
factor. One datasheet run converts a correlated bet into a measurement. **Note that the 128 KB
capacity decision survives either way**, since its area (−0.134 mm²) and leakage (halved) halves do
not depend on the energy model at all.

### D3 — Start the 17-call discovery minimum now, in parallel with D1 and D2.

**What it costs.** Founder time, essentially no cash. The funnel arithmetic, stated honestly:
cold outreach from an unknown pre-revenue founder gets roughly 5–15% substantive reply and lower
into large semiconductor companies, so holding 24 conversations implies **~200 approaches across
~120 organisations over 8–12 weeks.** Two calls per day maximum, notes written within two hours.

**What it buys.** The only evidence that can move the actual decision. Below 17 substantive
conversations (8 demand witnesses / 6 licensees / 3 specifiers) **no go/no-go conclusion may cite
discovery evidence in either direction** — a clause that protects equally against premature kill
and premature confidence.

**Sequence it as written**: weeks 1–3 the R1s (comma.ai, imperix, Embotech, ODYS, Auterion,
ANYbotics) to calibrate vocabulary cheaply before spending an R3 introduction; weeks 3–6 the R2
technical depth (ABB, Andes, Quintauris, Microchip, Telechips); weeks 6–12 the R3 licensees
(Infineon, NXP, Renesas, Bosch, ZF) armed with anonymised demand-witness evidence. Ask every
interviewee for one referral — it is the only mechanism that converts an R3 into an R2.

**Two corrections to make to the discovery material before the first call**, both from §3.4: the
interview guide's rationale for Q-B4 quotes the superseded "17–25% perf/W and 1.2–2.9× latency," and
§1 still says a CPU meets 3 of 5 solver deadlines when it now meets 5 of 5. Volunteering our weak
results is a deliberate and correct tactic — but volunteering *stale* weak results wastes the
credibility it is supposed to buy. Also decide what the cold email's promised "benchmark set with
node and precision attached" actually is, given that everything we have is ±2× and non-quotable.

### D4 — Minor, cheap, and worth doing before it is on the critical path

Stand up WSL2 + Ubuntu and smoke-test Verilator (`apt install verilator`, elaborate a trivial
module) purely to prove the toolchain works on this host. Needs administrator rights and a reboot.
Also close out the USPTO/EUIPO Nice class 9/42 search and the domains, since they gate every
external artefact.

---

## 7. Recommendation

### 7.1 The recommendation

**GO — but do not commit the twelve months yet. Commit ten weeks and under about $1,500 of
hardware to D1, D2 and D3 in parallel, and take the twelve-month decision at the 17-conversation
checkpoint, no later than 2026-10-15.**

If forced to answer the binary question *today, with no further evidence*, the answer is still GO.
But that framing should be resisted, because committing twelve months today buys nothing that
committing ten weeks today does not, while the ten-week package is precisely what makes the
twelve-month decision informed rather than hopeful.

### 7.2 Why GO

**The pre-registered kill criterion passes, and it passes with margin against its own stated
uncertainty.** 14.1–22.2× solver perf/W at the CM-100 configuration [E, 16 nm-class, mixed
fp32/fp64, ±2×] against a 5× threshold. Applying ±2× symmetrically, the pessimistic edge is ~7×,
still above the line. This is the criterion the founder wrote down *before* doing the work
specifically so that it could not be re-negotiated afterwards, and re-negotiating it now — in
either direction — would destroy its only value.

**The technical conclusions that survived correction are the ones a design would actually be built
on, and they are mechanism-level rather than constant-level.** Bandwidth-first provisioning follows
from operand reuse being zero in batch-1 GEMV, which is a property of the algorithm and not of our
energy constants. Sizing against the worst-case iteration tail follows from a **measured** cold-solve
tail of 60–74 iterations. Power gating being a requirement rather than an optimization follows from
measured-shape duty cycles of 0.3–11.7%. None of these three depends on the numbers most likely to
move.

**Nearly every architectural decision is a synthesis parameter, so being wrong is cheap.** Bandwidth,
capacity, array size and lane count are all instantiation-time choices behind the same MACIF
boundary. The solver-only SKU at 64 MACs / 0.430 mm² is a parameter, not a second design. The one
decision that is genuinely expensive to get wrong and genuinely cheap to get right now is **freezing
the MACIF boundary and the runtime API early** — which is Phase 1 work costing paper and thought,
not capital.

**The cheapest disconfirming experiments have not been run yet.** It would be indefensible to
declare no-go while the two experiments most likely to kill the project — a $100-class board plus a
weekend, and 17 phone calls — remain undone. It would be equally indefensible to spend a year of
runway before running them.

**And one asymmetry worth naming.** The failure mode that actually kills small IP companies is not a
wrong architecture; it is building the right architecture for a socket where nobody licenses. That
failure is detectable in D3 for the cost of founder time, and it is detectable *now*.

### 7.3 Why NOT commit twelve months today

Because the two things that would justify a twelve-month commitment are both absent. There is **no
market evidence of any kind** — zero conversations, no named power budget, no license-structure
information, no safety-package cost. And **the largest error source in every number above has never
been measured**: the CPU side of every ratio is a modelled guess at small-matrix SIMD efficiency,
and the previous ten corrections all moved in our favour, which is exactly the base rate that should
make a founder decline to bet a year on the eleventh.

A twelve-month commitment made in October with D1–D3 in hand is a substantially different and much
better decision than the same commitment made today, and it costs ten weeks to get there.

### 7.4 The conditions under which the answer flips to no-go

Using the falsifiers already pre-registered and frozen in CUSTOMER_DISCOVERY.md §7.2 — deliberately
not inventing new ones, and noting that **falsification there is easier to trigger than
confirmation, which is intentional.**

| Trigger | Where from | Consequence if it fires |
|---|---|---|
| **F2 — energy is not a purchase driver.** ≥6 of 8 demand witnesses cannot name a control-compute power budget, **and** rank compute >10× below actuators/radios/thermal, **and** ≥7 of 8 have never spent money on control-compute efficiency | §7.2 | **The most likely and most damaging.** Our headline advantage would have no buyer while latency is only ~1–3×. Then test determinism and core offload directly as the primary pitch; **if determinism also fails to be a purchased feature, that is a no-go on this beachhead** — and the remaining defensible market (battery-limited mobile robotics) is precisely the segment that cannot license IP, i.e. a no-go on the *business model* |
| **F3 — no licensee exists for control IP.** ≥5 of 6 licensees say it is not on the roadmap or is already covered in-house, **and** ≥4 of 6 categorically rule out non-silicon-proven IP from a sub-5-person vendor | §7.2 | Product *form* is wrong, not the technology. Ranked alternatives already written down: FPGA IP for industrial drives; an accelerated backend sold *through* Embotech/ODYS/MathWorks; a chiplet; or a software product (a better CPU solver on SME2/RVV) |
| **F4 — incumbent silicon already covers the socket.** ≥3 of 6 licensees state unprompted that an existing vector DSP runs their customers' MPC adequately and we cannot obtain measured counter-evidence — **or**, from D1, a PPU / same-power-class SME2 / RVV part runs benchmarks #1–#3 within **2×** of our modelled energy per solve | §7.2 + §5.1 experiment | Given ±2×, "inside 2×" reads as **no measurable advantage.** The gap becomes a feature gap closable in a software release. This is the most likely *technical* no-go and D1 is what resolves it |
| **F1 — the pain is not felt.** ≥6 of 8 witnesses report the solve consumes <25% of the control period with margin nobody tracks, **and** ≥6 of 8 have no war story of a feature cut for compute | §7.2 | Beachhead wrong at the *workload* level; re-scope to the two places the solve genuinely does not fit (20 kHz power-electronics MPC, and whole-body/long-horizon NMPC) and drop the general "robotics and automotive MPC" claim |
| **F5 — safety documentation cost is disqualifying.** ≥3 of 4 automotive respondents require a full SEooC package at license signature rather than before tape-out | §7.2 | Automotive moves to Phase 4+; beachhead becomes industrial drives under IEC 61508 — where per §2 of CUSTOMER_DISCOVERY the deadlines are harder anyway |
| **F7 — the workload is being replaced.** ≥5 of 8 witnesses are actively replacing online optimization with learned policies, explicit/offline MPC, or lookup tables | §7.2 | TAM shrinking underneath us. Note the trap: pivoting to the ML path lands us in a *worse* competitive position, since we concede inference efficiency to CIM and NPUs. A GO in that world needs a different differentiator — most plausibly the hybrid: learned policy plus constraint-enforcing solver on one engine |
| **The kill criterion fails on measured data.** Measured Arm energy per solve puts solver perf/W below 5× on benchmarks #1–#5 | PLAN.md §6, BENCHMARKS.md | This is the original pre-registered kill and it stands unmodified. It is also the criterion the ~9× low end of the sensitivity band could breach once ±2× is applied |
| **F6 — right for the wrong reason** (not a kill; a mandatory repositioning). ≥5 of 17 respondents volunteer determinism/WCET/timing-certification as the primary value and ≥4 explicitly deprioritise energy | §7.2 | Reposition to lead with provable WCET for constrained optimization. Changes benchmark *reporting* to worst-case rather than average, and possibly the architecture priority order. Cheap to act on, expensive to miss |

**Two technical results that are *not* no-go triggers, recorded so they are not mistaken for them.**
If the foundry datasheet shows flat pJ/byte versus access width, the CM-100 drops to 64 B/cycle and
the architecture is *better* (smaller, and segmented reduction deferrable) — that is a configuration
change, not a kill. If an L0 operand register file proves to buy the same operand-traffic reduction
at O(T) area, the honest array size drops to 36–64 MACs — again smaller and better.

**And what does not count as falsification**, per §7.3 of the pre-registration: any single "no,"
however senior; non-response (that is evidence about our reachability, not about the market);
"interesting, but come back with silicon," which is the *expected* answer from a licensee to a
pre-silicon vendor and becomes F3 evidence only in the categorical Q-C3/Q-C6 form; and enthusiasm
from academics, which is weighted at zero.

### 7.5 What Phase 1 should and should not contain, if GO is confirmed in October

Phase 1 remains deliberately paper-, model- and software-heavy, and therefore solo-feasible. It
should contain: the cycle-approximate model; **the frozen MACIF boundary and the frozen runtime API**
(cheap now, near-impossible to retrofit, and the API is the constitution — identical across
simulator, open core and commercial core); the solver codegen prototype against the functional
model; the L0-operand-tier model extension, which is the highest-value outstanding modelling work;
the fp64 datapath area estimate, which is the largest un-costed item; and a whitepaper that **leads
with architecture and mechanism rather than benchmark ratios**, because until measured data exists
we have no externally quotable ratios and should not pretend otherwise.

It should not contain RTL beyond what the FPGA reference needs, and it should not contain any
external document that presents a ±2× model output as anything else.

---

## Provenance

| Claim class | Source of record |
|---|---|
| Benchmark definitions, baselines, kill criterion | [BENCHMARKS.md](BENCHMARKS.md) v1.1 |
| Modelled results, corrections, teardown, citations | [PHASE0_FINDINGS.md](PHASE0_FINDINGS.md) — **§0 supersedes §1; §2/§3 tables are pre-correction** |
| Sizing sweep, area/energy decomposition, adversarial checks | [perf_model/results_cm100_sizing.txt](perf_model/results_cm100_sizing.txt), [perf_model/results_sram_cost.txt](perf_model/results_sram_cost.txt) |
| The only measured data we own | [perf_model/results_osqp_measured.txt](perf_model/results_osqp_measured.txt), harness [perf_model/measure_osqp_baseline.py](perf_model/measure_osqp_baseline.py) |
| Discovery program, pre-registered falsifiers §7 | [CUSTOMER_DISCOVERY.md](CUSTOMER_DISCOVERY.md) v1.0, frozen 2026-07-29 |
| Test suite, encoded error regressions, environment gaps | [TESTING.md](TESTING.md), `tests/` |
| Roadmap, business model, IMC durability analysis | [PLAN.md](PLAN.md) |
| Correction chronology | `git log` — four commits, 2026-07-30 |

Run `python -m pytest` before quoting any modelled number from this memo.
