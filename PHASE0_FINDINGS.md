# Argmind CM — Phase 0 Findings (interim, 2026-07-29)

Inputs: [BENCHMARKS.md](BENCHMARKS.md) v1.0 run through
[perf_model/argmind_model.py](perf_model/argmind_model.py) v0, plus a competitive teardown of
four commercial NPU IP families and one measured CIM macro.

**All model numbers are first-order estimates (±2×) and must not be quoted externally.**
Raw output: [perf_model/results_v0.txt](perf_model/results_v0.txt).

---

> ## ⚠ §0 — CORRECTED HEADLINE (2026-07-29, after three audits). Supersedes §1.
>
> All corrections applied and regenerated from one run of the current model:
>
> **All figures below regenerated 2026-07-30 at the RECOMMENDED CM-100 configuration**
> (144 MACs, 128 KB, 16 banks × 8 B = 128 B/cycle, 16 lanes, 800 MHz, T1). Every ratio quoted
> before this date was computed at the legacy 256-MAC / 256 KB / 64 B/cycle configuration — a
> part we no longer recommend — which is why the latency numbers moved.
>
> | | Corrected (CM-100) | Previously claimed |
> |---|---|---|
> | Solver perf/W | **14.1–22.2×** (modelled, ±2×) | 17–25× |
> | Solver latency | **1.0–2.1×** — parity at worst, nothing below | 1.2–2.9× |
> | Deadlines met | 6 of 7 (only SLM-DECODE misses) | 5 of 7 |
> | CPU also meets deadline | **5 of 5 solver benchmarks** | 3 of 5 |
>
> Note the "below parity on 2–3 of 5" finding recorded earlier was an artefact of reporting at
> 64 B/cycle. At the recommended 128 B/cycle it disappears — QUAD-MPC goes 0.9× → 1.8×. The
> concession that stands, and must be said out loud, is that **a CPU meets every solver deadline
> in the frozen set**; we are 1.0–2.1× faster and ~14–22× more efficient, not enabling.
>
> **The kill criterion (≥5×) still passes on every solver benchmark, and bandwidth-first is
> confirmed and now costed. Everything else has moved.** Three corrections drove it: the phantom
> transcendental term, `P@x` double-counted plus A treated as dense, and the scratchpad energy
> constant proving 2.3× too low. The honest one-line summary is now: *the CPU meets every solver
> deadline we benchmark, we are at or below its latency on three of five, and our case is
> ~10× energy and a bounded worst case.*
>
> ### The cross-agent contradiction: the CM-100 sizing is justified by a broken benchmark
>
> The scratchpad costing chose **128 B/cycle** and **256 MACs** on the grounds that "ARM-MPC is
> the *only* latency-binding benchmark" (margin 0.96× at 64 B/cycle → 1.60× at 128; and 64 MACs
> misses it while 144 is the true minimum). Independently, the op-count audit found ARM-MPC
> **overstated ~31×**: it charged `n_con·n_var²` to form `AᵀA`, which OSQP explicitly does not do
> and no production MPC stack does either, and it assumed 60 SQP iterations when deployed practice
> (acados `SQP_RTI`) is **one SQP step per tick** with partial condensing and HPIPM.
>
> With just the `AᵀA` term removed, **ARM-MPC now meets its deadline at 64 B/cycle** (1179 µs vs
> 2000 µs). So:
>
> - **No solver benchmark is latency-binding any more.** The justification for 128 B/cycle over
>   64 collapses, and with it the justification for 256 MACs over 144.
> - The remaining honest case for 128 B/cycle is **energy** (−15% solver energy for +0.042 mm²,
>   the area of 45 MACs), not deadline feasibility. That is a real but much weaker argument.
> - Re-specifying ARM-MPC as acados SQP-RTI would cut it a further ~31×, making the array
>   **entirely ML-justified**. Say so internally rather than implying solvers need it.
>
> **Action required before CM-100 is fixed: re-specify ARM-MPC (BENCHMARKS.md #2) and re-run the
> sizing.** Neither agent could see this — one owned the benchmark, the other owned the sizing.
>
> ### RESOLVED (2026-07-30) — sizing re-run: **the previous recommendation was over-provisioned**
>
> ARM-MPC re-specified (`sqp_rti_mpc`, fp64, 250 Hz) and the sizing re-run against the corrected
> set. Raw output: [perf_model/results_cm100_sizing.txt](perf_model/results_cm100_sizing.txt).
> **All figures modelled-estimate, 16nm-class, ±2×, not quotable.**
>
> | | Previous | **REVISED** | Min viable |
> |---|---|---|---|
> | Scratchpad bandwidth | 128 B/cyc | **128 B/cyc** | 64 B/cyc |
> | Scratchpad capacity | 256 KB | **128 KB** | 128 KB |
> | MAC array | 256 | **144 (12×12)** | 64 |
> | Core area | 0.820 mm² | **0.565 mm² (−31%)** | 0.430 mm² (−48%) |
> | Solver energy (Σ 5 rows) | 34.21 µJ | **33.64 µJ** | 45.28 µJ |
> | Solver perf/W | 11.9–23.6× | **14.1–22.2×** | 10.7–15.1× |
>
> The revised part **dominates** the old one — 31% less silicon at slightly lower energy. The old
> point was simply off the Pareto front. Four specific answers:
>
> **1. The typical case sizes nothing; the cold-solve tail does.** At p95 iterations every solver
> row meets its deadline at **16 B/cycle** (tightest margin in the whole 5×4 grid: 3.2×). A sweep
> run that way can only ratify a configuration, never justify one — which is what the previous run
> did. `iters_worst` (referenced in v0 comments but never implemented) is now in the model. Against
> the **measured** cold tail (QUAD-MPC 74 iters, `results_osqp_measured.txt`) inside a 50%-of-period
> engineering budget: 16 B/cyc misses outright (1092 µs > 1000 µs), 32 B/cyc blows the budget on
> QUAD-MPC (560 µs > 500) and PWR-MPC (33 µs > 25), 64 B/cyc clears both (295 µs, 21 µs).
> **64 B/cycle is the new latency floor — set by ADMM cold solves at 1 kHz/20 kHz, not by ARM-MPC.**
> At 100% of period, 32 B/cycle would suffice; that assumption is load-bearing and is stated.
> Binding resource: `gemv_bw` (scratchpad bandwidth) on the ADMM rows, `gemm_mac` on ARM-MPC's
> Riccati factorisation, `dma` on SLM-DECODE.
>
> **2. The array is *latency*-ML-justified but *energy*-solver-justified — the repositioning in the
> §0 note above is half wrong.** Nothing in the set needs >36 MACs to meet a deadline (VIT-TINY 36;
> every solver row ≤16). But tile dimension **is operand-reuse depth**: a Tm×Tn array fetches
> (1/Tm + 1/Tn) operand elements per GEMM MAC. 64→256 MACs cuts EKF-96 energy 1.59×, EKF-24 1.54×,
> ARM-MPC 1.29×, and QUAD-MPC/PWR-MPC by exactly 1.00× (pure GEMV). **EKF is a beachhead workload,
> not an ML one**, so "solver customers pay area for an ML feature" is not the honest story. The
> array is an *energy* feature for matrix-matrix solver work and a *throughput* feature for ML —
> consistent with the settled finding that our advantage is energy, not speed. Marginal return
> collapses past 144 MACs (84 µJ/mm² for 8→12, 27 for 12→16, 8 for 16→24), hence 144.
> *Adversarial:* growing the array buys reuse at O(T²) area for O(1/T) traffic; an **L0 operand
> register file** buys the same traffic cut at O(T) area. The model cannot yet evaluate a tiered
> scratchpad — highest-value model extension outstanding. If an L0 tier lands, the honest array
> size drops back to the ML-only 36–64 MACs.
> A **solver-only SKU is real**: 64 MACs at 0.430 mm² meets every solver deadline at worst-case
> iterations, 48% less silicon, at 35% higher solver energy (all of it on the EKF rows). Right SKU
> for an MPC-only socket, wrong one for state estimation. Synthesis parameter, not a second design —
> which is what the MACIF boundary is for.
>
> **3. The energy case for 128 B/cycle survives at −12%, but ~80% of it is a bank-geometry bet.**
> At 128 KB / 144 MACs: 38.05 → 33.64 µJ (−11.6%) for +0.043 mm² (+0.049 with margin). Decomposed:
> **3.49 µJ of the 4.42 µJ is pJ/byte falling 0.884 → 0.718**, only 0.93 µJ is the shorter active
> window. Byte *count* is identical at both widths — this is 16 banks of 8 KB being cheaper per
> access than 8 banks of 16 KB, not a bandwidth effect. It is the same [A2] 45nm-interpolation
> mechanism already flagged as at-risk from column gating, and **the same bet as the 128 KB energy
> claim, so the two are correlated rather than independent evidence.** If the Phase-1 compiler
> datasheet shows flat pJ/byte vs access width, drop to the 64 B/cycle floor.
> ARM-MPC's move to fp64 halves the MACs a given width can feed (8 B/element), but its 4000 µs
> deadline is 16× QUAD-MPC's against ~2× the op count, so it still does not bind — even 16 B/cycle
> lands at 1059 µs. **The fp64 consequence that does matter is the datapath, not the bandwidth:**
> the model gets fp64 from a 0.125 rate factor on a bf16 array, which is a modelling convenience.
> A real 1/8-rate fp64 path needs multi-pass decomposition (unmodelled sequencer/accumulator work)
> or a separate narrow fp64 FMA cluster (**area currently zero in the budget — largest un-costed
> item, and it sits exactly on the MACIF boundary**).
>
> **4. 128 KB suffices — and is better on area, energy and leakage.** Corrected footprints:
> QUAD-MPC 100 KB (was 150), EKF-96 117 KB, ARM-MPC 172 KB (was 861), PWR-MPC 8 KB, EKF-24 8 KB.
> The determinism claim is *no DRAM in the control loop*, so it must be tested per co-resident
> workload, not per benchmark: drone node (QUAD-MPC + EKF-24, 1 kHz) = 108 KB **fits**; drive node
> (PWR-MPC + EKF-24, 20 kHz — now **40 kHz and 8.6 KB** after the §0.3 re-spec) = 16 KB **fits**,
> and fits more easily; manipulator (ARM-MPC + EKF-96, 250 Hz) = 289 KB
> **fits neither 128 KB nor 256 KB**. So 256 KB does not extend the claim to the case that fails —
> it only reduces the streamed fraction of a workload with 24× deadline margin and 0.04 GB/s of
> streaming demand. 256→128 KB: **−0.116 mm² SRAM (−0.134 with margin), pJ/byte 0.930 → 0.718,
> solver energy −10%, leakage halved** (1.04 → 0.52 mW @85 °C, 4.18 → 2.09 @125 °C). The leakage
> halving matters most: ungated perf/W on the low-duty rows is set almost entirely by static power.
> 64 KB is rejected — QUAD-MPC's 100 KB set would stream every 1 ms tick, which is the claim we sell.
>
> **Unchanged:** bandwidth-first over MAC-first (ADMM rows are still GEMV-bound and still bind on
> scratchpad bandwidth, just at a lower absolute width); ≥5× kill criterion (11–22×); zero-copy +
> residency; **power gating is a requirement, not an optimisation** — and more load-bearing now,
> since a smaller core makes the 15 mW static [EST] a larger fraction of the total. SLM-DECODE
> still misses by 3.4× on external bandwidth (~22 GB/s needed, 6.4 provisioned): a provisioning
> requirement to state, not a design flaw.
>
> **Process note:** the error that caused this re-run was structurally the same as the one that
> caused the last one — a sizing decision resting on a single benchmark row nobody had audited.
> Two tests now encode the fix: worst-case iteration tails must exist *and must bind somewhere*,
> and GEMM energy must track reuse depth rather than MAC count (32 tests).

> ### §0.1 — CLOSED (2026-07-30): the fp64 datapath is costed. Core area restated 0.565 → 0.698 mm²
>
> Raw output: [perf_model/results_fp64_datapath.txt](perf_model/results_fp64_datapath.txt);
> model in [perf_model/fp64_datapath.py](perf_model/fp64_datapath.py). **All figures
> modelled-estimate at a 16nm-class node, ±2×, not quotable** — and the fp64 gate count carries a
> further **2.1× band** of its own, making these the least certain numbers in the budget.
>
> **Latency figures restated 2026-07-30** when the sequencer was modelled properly (§0.2):
> ARM-MPC's worst case moves 742 → **822 µs** and the refinement rows rise similarly, because the
> flat sequencer term was replaced by descriptor issue + dependency drain. **Every conclusion in
> this subsection survives unchanged** — the margins tighten by ~10% and no floor, crossover or
> verdict moves. The stated numbers below are the current ones.
>
> **Decision: 8 merged multi-format FMA units (1 × fp64 *or* 2 × fp32 each), which are also the
> 16 fp32 vector lanes. fp64 increment +0.067 mm² (+0.077 with top margin).** The `0.125` rate
> factor is deleted from the model; fp64 throughput is now an absolute FMA count that carries area
> and can be under-provisioned.
>
> | Option | fp64 area | ARM-MPC worst case | Verdict |
> |---|---|---|---|
> | (a) multi-pass on the bf16 array | +0.040 mm² (wide accumulators) | 3423–24705 µs | **rejected — energy** |
> | (b) separate fp64-only cluster, 8 units | +0.090 mm² | 822 µs | viable |
> | (c) native fp64 throughout (144 cells) | +1.417 mm² | 152 µs | **rejected — 3.9× the part** |
> | **(d) merge the cluster with the vector unit** | **+0.067 mm²** | **822 µs** | **RECOMMENDED** |
>
> **Gate-count basis.** A parametric IEEE-754 FMA model, `gates = 4.5·n² + 240·n + 1200` for
> significand width *n* (`sram_cost.fma_gates_ieee`), at 0.26 µm²/NAND2: **fp32 9.55 kgates, fp64
> 26.6 kgates (2.8×)**. Calibrated so fp32 lands in the published 8–12 kgate band; cross-checks at
> n=8 against the independently-guessed 3.0 kgate bf16 array cell to within 14%. Pessimistic edge
> is a **measured** 28 nm dual-precision FMA at
> [56.2 kgates](https://www.sciencedirect.com/science/article/abs/pii/S0141933117305665) (2.1×),
> which takes core area to **0.845 mm²**. Existence proof for the merged unit is
> [FPnew/CVFPU](https://arxiv.org/abs/2007.01530) (GF 22FDX, FP64/FP32/FP16/bfloat16 in one slice).
>
> **1. The requirement is small, and only interior point creates it.** ARM-MPC is 1 of 7 benchmarks
> and 79% matrix-**matrix** (block Riccati), not GEMV. At worst-case 30 IP iterations it is
> 3.77 M fp64 MACs, i.e. **2.36 fp64 MAC/cycle** at 800 MHz inside a 50%-of-period budget before
> any overhead. With utilisation, GEMV/GEMM split, vector and sequencer terms charged:
> **1 FMA misses the 4000 µs deadline (5967 µs), 2 misses the 50% budget, 4 is the floor (1.24×),
> 8 gives 2.43× on the budget and 4.87× on the deadline.** 8 is chosen over 4 because it is *free* —
> 8 merged units is exactly the 16 fp32 lanes the part already carries — and because above 8, fp64
> GEMV is bandwidth-capped at 16 MAC/cycle and further units buy 64 µs of a 2000 µs budget. The old
> `0.125` factor was **over-providing fp64 by 2.25×** (18 vs 8 MAC/cycle) at zero area.
>
> **What the sizing is and is not sensitive to.** The softest modelling assumption — softening the
> cluster's quantisation penalty by a square root, worth 1.26× — does **not** move the answer:
> charged the full 2-D penalty, 8 FMAs gives 2.06× instead of 2.43× and the floor is still 4.
> What *does* move it is `iters_worst = 30`, which is **[EST]** (2× the design point), not measured:
> at a 45-iteration tail the floor is 6 FMAs and the 8-FMA margin is 1.65×; at 60 iterations it is
> **1.25×** — still feasible, no longer comfortable. **Measuring ARM-MPC's actual interior-point
> iteration tail is the highest-value input to this decision, ahead of the gate count and ahead of
> the energy constant.**
>
> **2. Multi-pass is rejected on energy and interface risk, not on the deadline** — the deadline
> argument turned out weaker than expected and is reported as such. Error-free (Ozaki-scheme) fp64
> emulation costs **25–196 GEMM passes with fp16 inputs and 121 with fp8**
> ([arXiv:2508.00441](https://arxiv.org/abs/2508.00441), Table 1); bfloat16 has fewer mantissa bits
> so needs more, extrapolating to **49–361 passes [EST from that data]**. The optimistic edge of
> that band *does* meet the raw 4000 µs deadline. What kills it: **3.5–23× worse energy than native**
> (152–987 µJ vs 43 µJ), because P passes multiply both MAC energy and sliced-operand fetch traffic —
> unaffordable on the axis we sell, to save **0.028 mm², ~4% of the core**. **The asymmetry is the
> real finding: bf16 multi-pass reaches fp32 accuracy in 6–9 passes and fp64 accuracy in 49–361.
> fp32 is exactly the precision interior point cannot use.** (Consistent, not contradictory: the
> model still shows 9-pass bf16 *cheaper* than native fp64, which is why the fp32 path already uses
> operand splitting.)
>
> **3. The sequencer cost was checked, and the expected hidden expense is not there — but a
> different one surfaced.** The multi-pass issue:array ratio is **pass-count-independent** (both
> terms scale with P), so multi-pass carries no *relative* sequencer penalty. However, forcing the
> descriptor count to be consistent with the benchmark's own GEMM count shows an HPIPM block op at
> nb=21 is only **~5.6 array cycles**. At 8 sequencer cycles/descriptor the ratio is 1.42× — i.e.
> **ARM-MPC would be sequencer-issue-bound rather than array-bound, at any precision.** The model
> does not represent this (flat 150 scalar ops/iteration). It does not change the fp64 decision (an
> issue bound makes a *larger* fp64 unit even less useful) but it is a **new open item**: descriptor
> issue must be ≤4 cycles, or the sequencer needs a hardware loop emitting a stage's descriptors
> from one instruction. **Logged, not resolved.**
> **→ RESOLVED in §0.2, and the claim above is wrong: the "~5.6 array cycles" divided an fp64
> workload by the bf16 *array* rate. The correct figure is ~30 cycles and "at any precision" is
> false. The real exposure is elsewhere — see §0.2.**
>
> **4. fp32 + iterative refinement does not remove the fp64 requirement — it relocates it.** Fixed-
> precision refinement converges only while cond·ε < 1 (Wilkinson/Skeel/Higham); at the barrier
> endgame cond ≈ 1e12 and ε_fp32 ≈ 6e-8, so the product is ~6e4 ≫ 1 and it does not converge.
> Mixed-precision (GMRES-IR, Carson & Higham 2017/18) works but needs the **residual in fp64**. So
> fp64 silicon is required either way — refinement moves it from the O(n³) factorisation to the
> O(n²) residual. Modelled honestly, refinement is **faster at every k (1.2–2.7×)** and there is a
> genuine **energy crossover at k = 3**: below it refinement wins on both axes, above it native
> fp64 wins on energy. The numerics force us above it. Two further points decide it: **the
> asymptotic case for refinement fails at our block sizes** (residual:factorisation = 3k/nb, which
> at nb=21, k=3 is **43%**, not the ~5% HPC intuition — refinement is only "cheap" above nb≈90);
> and the **certifiable worst case with an all-fp64 fallback is the sum of both paths, 1328 µs vs
> 822 µs for all-fp64 alone**. A scheme whose worst case exceeds the thing it replaces cannot be
> the baseline for a hard-deadline part. **Verdict: native fp64 is the baseline; refinement stays
> an optional codegen mode.** This rests on a citation, not our own measurement — see below.
> **Answer for the solver-only SKU: unchanged, and structurally so** — the cluster is not part of
> the array, so all-fp64 latency is *identical* (822 µs) on both parts. What changes is the area
> **fraction**: 9.6% of the CM-100 vs **11.9% of the floor SKU**. fp64 is proportionally more
> expensive on the small part, and the small part is the MPC-only SKU that most needs it. **There
> is no SKU from which fp64 can be removed while keeping interior-point MPC.**
>
> **5. Restated area budget.** Two corrections, reported separately so neither hides in the other:
>
> | Item | Was | Now | Δ |
> |---|---|---|---|
> | Scratchpad (128 KB, 16 banks) | 0.2026 | 0.2026 | — |
> | MAC array (144 bf16) | 0.1348 | 0.1348 | — |
> | Vector unit (16 fp32 lanes) | 0.0092 | **0.0577** | +0.0485 ← *pre-existing underestimate* |
> | fp64 increment (8 merged FMAs) | **0.0000** | **0.0670** | +0.0670 ← *the item closed here* |
> | Sequencer + DMA + misc | 0.1450 | 0.1450 | — |
> | Subtotal / +15% margin | 0.4916 / **0.565** | 0.6071 / **0.698** | +24% |
>
> The old `VEC_LANE_GATES = 2200` was **unphysical**: an fp32 multiplier alone is ~2.6 kgates, so
> 2200 could not hold one. It is now derived from the same FMA gate model. **Honest range to state
> for the CM-100 core: 0.70–0.85 mm² at 16nm-class, ±2× on top.** Floor SKU: 0.430 → **0.563 mm²**.
>
> **6. fp64 energy per MAC is load-bearing, and the kill criterion survives it.** At 2.20 pJ [EST]
> it is **39% of ARM-MPC's 22.6 µJ/call** — the single largest term. Sensitivity: **19.6× / 16.5× /
> 14.2× / 11.1× perf/W at 1× / 1.5× / 2× / 3× the constant — passing 5× throughout.** So the
> go/no-go does not depend on it; the *quoted ratio* does, by ~2×. An independent n²-scaling
> derivation from Horowitz gives 2.45 pJ, so the constant is ~13% optimistic on its own logic.
>
> **7. A CPU-baseline bug was fixed at the same time, and it moves the ratio in our favour — so it
> is flagged, not banked.** `model_cpu` charged fp64 work at every baseline's **fp32** rate. NEON/SVE
> fp64 is half-rate and SME fp64 outer products are quarter-rate. ARM-MPC perf/W therefore moves
> **14.1× → 19.6×**, and *all* of the increase comes from the CPU side (A55: 276 → 445 µJ, 1.6×)
> against our own side getting slightly worse (19.6 → 22.6 µJ, from the honest fp64 throughput).
> Two errors pointing in opposite directions that had partly cancelled. Sanity check the fix is not
> overdone: **the A55 baseline still meets ARM-MPC's raw 4000 µs deadline (2955 µs)** — we are not
> claiming the CPU cannot do this. Note also `FEAT_SME_F64F64` is **architecturally optional**; an
> SME2 part without it has no fp64 matrix path at all. Falsifier F4 — settle on an eval board.
>
> **8. MACIF consequence: precision must be a declared contract, not an implementation detail.**
> - fp64 is an **algorithm requirement**, not a performance knob — a boundary that lets a macro
>   silently substitute lower precision turns a swap decision into a field correctness failure.
>   Precision must be queryable and the runtime must **refuse** an fp64 kernel on a macro that does
>   not declare fp64 — a hard codegen-time error.
> - **The boundary runs between the bf16/int8 array (swappable) and the fp32/fp64 FMA cluster (not
>   swappable).** Neither analog nor digital CIM has a path to 53-bit significands, so the fp64 unit
>   must sit in the fixed part of the CM, merged with the vector unit. This is the architectural
>   consequence of the whole analysis.
> - Choosing (d) over (a) **keeps MACIF's contract weak enough for a CIM macro to honour it.**
>   Multi-pass would have pushed error-free limb semantics and an exact 128-bit accumulator *into*
>   the swappable macro — a contract analog CIM cannot meet at all.
> - fp64 is a **feature bit, not a scaling parameter** (same 8 units on the 144-MAC and 64-MAC
>   parts), so MACIF v0 must declare, per format: `supported`, `macs_per_cycle[format]` (absolute,
>   not a rate factor), `accumulator_width[format]`, rounding mode and subnormal behaviour. **The
>   deleted `0.125` factor is exactly the shape of interface that would have been wrong.**
>
> **What would firm this up, in order of value per day:** (1) synthesise one CVFPU merged
> fp64|2×fp32 FMA at 800 MHz with Yosys + an open PDK — closes both the 2.1× gate band and the
> 2.20 pJ constant, one day, open source; (2) run acados+HPIPM in fp64 and in fp32-with-refinement
> and **measure** k and whether it converges at the endgame — item 4 rests on a citation;
> (3) measure ARM-MPC's true worst-case IP iteration count (`iters_worst=30` is [EST] and it sizes
> the cluster); (4) count HPIPM's actual block ops per IP iteration, which sets the newly-surfaced
> descriptor-granularity risk. **28 new invariant tests (70 total).**

> ### §0.2 — CLOSED (2026-07-30): the sequencer is modelled. The logged issue-bound risk was a unit error; two real risks replaced it
>
> Raw output: [perf_model/results_sequencer.txt](perf_model/results_sequencer.txt); model in
> [perf_model/sequencer.py](perf_model/sequencer.py), with the descriptor cost itself in
> `argmind_model.descriptor_cost`. **All figures modelled-estimate at a 16nm-class node, ±2×, not
> quotable externally.** Evidence classes below: [EST] estimate, [DER] derived from another model
> figure, [MEAS] measured, [C*] cited.
>
> **Decision: no provisioning change. The recommended sequencer is a small scalar core plus a
> hardware descriptor generator with a depth-8 run-ahead queue (S2), 4 sequencer cycles per
> descriptor, 1 descriptor/cycle issue width — 0.028 mm² all in, inside the existing 0.045 mm²
> allocation, so core area stays 0.698 mm². Run-ahead and wide dependency-chain blocking are both
> *requirements*, not optimisations.** The binding resource is unchanged on every benchmark:
> scratchpad bandwidth on the ADMM rows, the fp64 FMA cluster on ARM-MPC, the tile array on
> EKF/ViT, DMA on SLM decode.
>
> **1. The logged claim was a unit error, and it is worth being precise about which one.** The
> "~5.6 array cycles" figure divided an HPIPM block operation by the **144-MAC bf16 array** rate.
> ARM-MPC is fp64, and the decision made two sections earlier in the *same file* is that fp64 does
> **not** come from the array — it comes from the 8-unit FMA cluster, at 1/16th the rate. Correcting
> the denominator gives ~68 cycles; correcting the descriptor count in the other direction (the
> structural count is **540/IP iteration, 3× the guessed 180**, because the Cholesky and the gain
> solve are dependency chains that must be blocked) gives **30.3 datapath cycles per descriptor
> against a crossover at 34.3 setup cycles [DER]** — array-bound with 8.6× margin at the
> recommended 4-cycle setup cost. **"At any precision" is false, and false in the instructive
> direction: exposure scales with arithmetic *rate*, so the fastest precision is the most exposed,
> not the slowest.** Crossover setup cycles for ARM-MPC by precision: **fp64 34.3, fp32 10.1,
> bf16 7.5 [DER]**.
>
> **2. What replaced the fudge.** `scalar_ops_per_iter` (flat, 150 for ARM-MPC) and the fixed 50%
> overlap are gone from the CM path — `scalar_ops_per_iter` is now a **CPU-only** parameter, and a
> test asserts that changing it cannot move CM latency. In their place, three terms with physical
> meanings, all derived from a per-benchmark kernel decomposition (`KernelOp`) whose MAC counts are
> **asserted to sum to the benchmark's declared totals** — the discipline whose absence produced the
> original error:
>
> | Term | Definition | Relieved by |
> |---|---|---|
> | `issue_cycles` | descriptors × setup cycles ÷ issue width | faster descriptor emission |
> | `dep_drain` | one datapath drain per dependency-chain boundary | wider chain blocking, shallower pipeline |
> | `serial_scalar` | loop-carried scalar chain (step size α, convergence test) | nothing — it is a true dependency |
>
> Per iteration: `max(datapath + dep_drain, issue) + serial`, with a run-ahead queue; the **sum** if
> queue depth is 0. Descriptors per iteration and datapath cycles each [DER]:
>
> | Benchmark | desc/iter | of which chained | datapath cyc/desc | crossover (setup cyc) | binding resource |
> |---|---|---|---|---|---|
> | PWR-MPC (ADMM, n_var=30) | 16 | 8 | **10.2** | **14.1** | scratchpad bandwidth |
> | ARM-MPC (HPIPM, nb=21, fp64) | 540 | 270 | 30.3 | 34.3 | fp64 FMA cluster |
> | QUAD-MPC (ADMM, n_var=80) | 28 | 20 | 59.3 | 65.0 | scratchpad bandwidth |
> | EKF-24 | 17 | 4 | 81.7 | 83.6 | tile array |
> | EKF-96 | 21 | 8 | 2461 | 2464 | tile array |
> | SLM-DECODE | 330 | 0 | 27279 | 27279 | DMA |
> | VIT-TINY | 144 | 0 | 42209 | 42209 | tile array |
>
> **3. The design rule, because that is what this becomes in codegen.**
>
> > **A code-generated iteration must average at least 2 × `seq_setup_cycles` datapath cycles per
> > descriptor (≥ 8 at the recommended S2 sequencer), and no burst of sub-threshold descriptors may
> > exceed the descriptor-queue depth (8). Below either bar, fuse the operations or unroll the
> > chain.**
>
> The rule is an *average* plus a *burst length* rather than a per-descriptor minimum because
> run-ahead lets a short descriptor borrow slack from a long neighbour. The bursts are the
> dependency chains: an HPIPM stage at nb=21 on 8 fp64 units emits **9 consecutive sub-threshold
> descriptors** (potrf blocked 3 ways × 2, plus trsm). Margin against the bar across the frozen
> set: QUAD-MPC 7.4×, ARM-MPC 3.8×, EKF-24 10.2×, EKF-96 308×, **PWR-MPC 1.27× — the thinnest row
> in the set.**
>
> **4. The dominant new cost is dependency drain, not issue throughput — and that changes the
> design response.** At 4 cycles/descriptor, **97–100% of the sequencer's latency contribution is
> drain**, not issue: 8.7% of QUAD-MPC's latency, 11.4% of ARM-MPC's, **26.7% of PWR-MPC's [DER]**.
> This is not a provisioning problem. No amount of issue bandwidth, MAC count or scratchpad width
> touches it (a test asserts the drain term is invariant to scratchpad bandwidth). What *does*:
> **blocking the chain as wide as the datapath, and keeping the datapath pipeline short.** The
> latter gives PLAN.md §3.1's short-fill/drain requirement a second, independent justification.
>
> **5. Two real risks replaced the logged one, and both can miss a deadline.**
>
> - **Run-ahead is mandatory.** A scalar-core sequencer that writes a descriptor, starts it and
>   polls for completion (setup ≈ 20 cycles, queue 0) **misses PWR-MPC's 50%-of-period budget:
>   38.1 µs against 25 µs [DER]**, i.e. 0.66× margin. Software double-buffering alone (queue 1)
>   recovers it to 1.23×; the S2 hardware generator gives 1.71×. This is the only deadline miss the
>   analysis produces, and it is on the 20 kHz motor-drive row — the MCU-class socket.
> - **Narrow dependency-chain blocking breaks the same row.** If the RTL's blocked Cholesky can
>   only advance 4 elements per dependent step instead of a full datapath width, PWR-MPC's margin
>   falls to **0.96×** and at 1 element to **0.37× [DER]**. **RTL requirement: a blocked
>   Cholesky/triangular solve must advance ≥ 8 (a full datapath width) per dependent step.** This
>   is a bigger lever than the descriptor setup cost itself and is the highest-value Phase-1 check.
>
> **6. ADMM vs block-structured interior point, checked separately — and the tidy answer is wrong.**
> The hypothesis was that ADMM's few large GEMVs make it insensitive and the exposure is specific to
> block-structured IP. **Half confirmed.** ARM-MPC does issue **19× more descriptors per iteration**
> than QUAD-MPC, and QUAD-MPC is genuinely insensitive (crossover 65 setup cycles). **But the most
> exposed row in the set is PWR-MPC, which is ADMM** — exposed because n_var=30 is tiny and the rate
> is 20 kHz, not because of its algorithm. So the correct statement is: **exposure is set by datapath
> cycles per descriptor, which block structure and small problems both reduce independently.** Do not
> pitch this as "interior point is the risky one."
>
> **7. A new, third argument for native fp64 that §0.1 did not charge.** The fp32 +
> iterative-refinement option puts the O(n³) factorisation on the array in fp32, where the crossover
> is **10.1 setup cycles [DER]**. That option is **sequencer-issue-bound with anything but a hardware
> descriptor generator** — a cost §0.1 did not price. Native fp64 wins on energy, on numerics, and
> now on issue pressure.
>
> **8. When sequencer work genuinely overlaps compute — stated as a condition, which is what the 50%
> fudge lacked.** Overlap is legitimate **because the CM's allocation and schedule are statically
> code-generated**: every descriptor's addresses, dims and strides are known before the loop runs, so
> setup never waits on data, and the loop-continue branch is correct on every iteration but the last.
> **Determinism and overlap-legality are the same architectural property** — a point worth making in
> the pitch. Overlap does *not* hold for (a) the chain boundary itself, charged as `dep_drain`, or
> (b) genuinely data-dependent descriptor *parameters* — a fraction-to-boundary α scaling the next
> update, an active-set index, a pivot — charged as `serial_scalar_per_iter` and small (60
> cycles/IP iteration, 25/`check_term` for ADMM).
>
> **9. Was the fudge actually wrong? Plainly: for ADMM it was a good approximation by accident.**
> 75 modelled cycles/iteration against 161, but both are 5–10% of the iteration, so QUAD-MPC latency
> moves **+5.6%** and PWR-MPC **+1.8%**. ARM-MPC moves **+11.8%** (the constant was ~30× low in
> cycles). EKF and ML rows move <0.2%. **Headline ratios are unchanged: solver perf/W stays 14–22×;
> solver latency 0.97–2.1× vs 0.99–2.11×.** The error was *qualitative*, not quantitative — the
> constant stood in for descriptor issue, while the term that actually costs time is dependency
> drain, and the two have opposite design responses.
>
> **10. Area: costed bottom-up, and it fits — contrary to expectation.** `REST_MM2` allocated
> 0.045 mm² to "sequencer" as a flat [EST] with no memory line item, which looked like a candidate
> omission. Costed properly: **0.0137 mm² of logic** (RV32IMC ~30 kgates + 3 descriptor AGUs +
> 3-level hardware loop + 4 templates + 8-deep queue + issue FSM, at 0.26 µm²/NAND2 [EST]) **plus
> 0.0143 mm² for an 8 KB I-mem and 4 KB D-mem [DER from `sram_cost.bank_geometry`] = 0.0280 mm², 62%
> of the allocation.** The S0→S2 descriptor-generator delta is **0.0048 mm², 0.69% of the core**. It
> would take a 26 KB program store to break the budget. **No area change; what was an assertion is
> now a derivation.**
>
> **11. Consequence for the D2/D3 integration depths (PLAN.md §4.2).** D2/D3 shift descriptor issue
> onto the host core, which raises setup cost (~20 cycles, no dedicated AGUs) and removes reliable
> run-ahead (the host is also running the application). Worst solver margin against the 50% budget:
> **D1 with its own sequencer 1.71×; D2 with the generator moved to the CM side 1.71×; D2/D3 with the
> host issuing and software double-buffering 1.23×; D2/D3 with the host issuing and no run-ahead
> 0.66× — misses.** The binding row is PWR-MPC every time.
>
> > **Decision: the descriptor generator belongs on the CM side of the interface at every integration
> > depth. D2/D3 may move the control flow to the host; they must not move descriptor emission
> > there.** This is also a MACIF question — the descriptor format is part of the boundary a CIM
> > macro must honour, so the generator sits *above* MACIF and cannot be a host-side artefact.
>
> It bites precisely where §4.2 nominates D2: the MCU-class socket where D1's zero-copy has nothing
> to bite on (the unresolved §4.1 tension) *is* the motor-drive row, and that is the most
> issue-sensitive benchmark in the set. D2 stays "a wrapper rather than a redesign" **provided the
> wrapper keeps the generator.**
>
> **What would firm this up, in order of value per day:** (1) write the descriptor-emit sequence for
> one HPIPM stage in RV32 assembly and count instructions — settles `seq_setup_cycles`, the single
> most load-bearing [EST] here, in an afternoon; (2) determine whether the RTL's blocked Cholesky can
> advance a full datapath width per dependent step — it is a bigger lever than (1); (3) instrument
> HPIPM's actual BLASFEO call counts on the real problem, which settles both the descriptor count and
> the 30/45/15/10 split of nb³/3 across trmm/syrk/potrf/trsm [EST]. Note the benchmark's `N·nb³/3`
> GEMM count is **~2.2× below** a true HPIPM stage count at nb=21 [EST], so this analysis is
> *conservative* on that axis — fixing it raises cycles/descriptor and makes every conclusion here
> more comfortable. **52 new invariant tests (122 total).**

> ### §0.3 — CLOSED (2026-07-30): PWR-MPC re-specified. Three conclusions downgraded, the row is still below parity, and the MCU-class contradiction is settled
>
> Raw output: [perf_model/results_pwr_mpc_respec.txt](perf_model/results_pwr_mpc_respec.txt);
> model in [perf_model/pwr_mpc_respec.py](perf_model/pwr_mpc_respec.py) plus the FCS-MPC
> benchmark, mixed-precision operands, the GEMM bandwidth ceiling and the MCU-class host tiers in
> `argmind_model.py`. **All latency/energy/area figures are modelled estimates at a 16nm-class
> node, ±2×, not quotable externally.** Numerical results are [MEAS] *numerics*, not silicon.
> **29 new invariant tests (151 total).**
>
> Three threads converged on this one row and none had been settled: it was the only solver row
> below the fastest CPU baseline (0.974×), it had the thinnest codegen margin (1.27×), it forced
> run-ahead to be a hardware requirement, and it is the socket where CLAUDE.md records an
> unresolved zero-copy tension. **The first question dissolved most of the rest.**
>
> #### 1. The benchmark was wrong for its socket. Re-specified as finite-control-set MPC
>
> Deployed practice in motor drives and power electronics is **direct / finite-control-set MPC**,
> not a continuous-control-set QP. From Karamanakos, Liegmann, Geyer & Kennel, *Model Predictive
> Control of Power Electronic Systems* (IEEE Open Journal of Industry Applications, vol. 1, 2020,
> [PDF](http://www.tobiasgeyer.org/KLGK20_MPCsurvey.pdf)), quoting directly:
>
> - *"direct MPC for power electronics is most often formulated as an IQP. The common practice for
>   solving such optimization problems in embedded control systems is with exhaustive search."*
> - *"Since the problem is NP-hard, such an approach is impractical for horizons longer than one
>   step (N_p > 1)."* Candidate count for a three-phase two-level converter is 2^(3·N_p) — 8 at
>   N_p=1, 64 at N_p=2; a three-level converter gives {−1,0,1}³ = **27** and **729**.
> - *"most direct MPC algorithms require a sampling frequency of about two orders of magnitude
>   higher than the desired switching frequency"*; published implementations use **Ts = 5, 10, 25,
>   100 and 125 µs**. **So 20 kHz was below deployed practice — the re-spec makes the deadline
>   harder, not easier.**
> - Continuous-control-set MPC *does* appear here (indirect MPC + carrier PWM, solved with
>   qpOASES/ODYS; miOSQP/ADMM for the integer problem) but on dSpace-class rapid-prototyping
>   hardware, not in a shipping drive.
>
> Commercial reference point: MPDTC with a tree search over candidate switching sequences at
> **Ts = 25 µs**, embedded in ABB medium-voltage drive control software (Geyer & Papafotiou,
> [PDF](https://tobiasgeyer.org/GePa09_MPDTC_ACS5000.pdf); Geyer is R&D platform manager for the
> ACS6000/6080). Explicit MPC — offline piecewise-affine law + point location + table lookup — is
> the other deployed answer and is a **memory** workload; we concede it rather than model it.
>
> **PWR-MPC v1.2: three-level NPC, |U| = 27, N_p = 2 exhaustive (729 sequences), n_x = 4, n_y = 2,
> fp32, Ts = 25 µs / 40 kHz.** The retired definition survives off-set as **PWR-CCS**, alongside
> **PWR-FCS1** (one-step, 27 candidates — the case we *lose*) and **PWR-FAST** (two-level, 100 kHz).
>
> #### 2. The result is NOT favourable, and the hoped-for reading is dead
>
> The hope was that enumeration — embarrassingly parallel, no loop-carried dependency, not
> bandwidth-bound — would move the row above parity. **Both halves of that are wrong.**
>
> | | PWR-CCS (retired) | **PWR-MPC (v1.2)** | PWR-FCS1 (N_p=1) |
> |---|---|---|---|
> | Speedup vs fastest CPU | 1.08× | **0.62×** | 0.12× |
> | perf/W vs most efficient CPU | 15.2× | **23.5×** | 7.1× |
> | Worst-case margin, 50% of period | 1.71× | **7.76×** | 14.73× |
> | Cycles/descriptor vs ≥8 rule | 1.27× | **6.76×** | **0.61× — fails** |
> | Binding resource | scratchpad bandwidth | **per-call overhead, then vector unit** | per-call overhead |
>
> - **Latency gets worse: 1.08× → 0.62×.** The whole tick is 0.81 µs of compute against a 0.80 µs
>   invocation overhead, while the host core pays no invocation cost at all. Decomposed:
>   **datapath-only 1.24× → with the MCU-class 0.30 µs tier 0.90× → with T1's 0.80 µs 0.62×.**
>   The CM *computes* this tick 1.24× faster than an SME2 core and then pays 0.8 µs to be asked.
>   **For a sub-microsecond workload no invocation protocol involving a descriptor and a completion
>   notification puts an accelerator ahead of the host core on latency.** That is an
>   integration-depth statement: this socket wants D3, or it is sold on energy and offload alone.
> - **It is still bandwidth-bound, for a new and more general reason.** The candidate set *is* a
>   real batch dimension, so the work is genuinely matrix-**matrix** — but the reuse depth is set
>   by **n_y = 2, the number of controlled outputs**, not by the tile. Arithmetic intensity is
>   **2.33 B/MAC against 0.67 B/MAC** for a tile-limited GEMM: the batch buys a 1.7× traffic cut
>   over pure GEMV, not the ~12× the array size suggests. **A GEMM bandwidth ceiling had to be
>   added to the model to express this — it previously bounded only GEMV, so it was structurally
>   incapable of reporting a bandwidth-starved GEMM.** A test asserts the new term binds on
>   PWR-MPC and on nothing else.
> - **A crossover, not a win ratio:** FCS-MPC is *array*-bound below 12×12 (tile=8 costs it 1.14×)
>   and intensity-bound above. **144 MACs is the knee for this row too** — the same number the EKF
>   energy argument produced, arrived at independently.
>
> What the re-spec *does* buy is worth more than the ratio: **the op count is fixed** (no
> convergence tail, worst case == typical case — determinism as strong as the EKF's, on the row
> where determinism is most saleable), **zero dependency chains** (drain 64 → 0 cycles/iter), and
> the codegen margin goes 1.27× → 6.76×.
>
> #### 3. Three conclusions lose their justification — listed, not silently changed
>
> Worst-case margin against 50% of the control period:
>
> | Configuration | QUAD-MPC | PWR-MPC (v1.2) | PWR-CCS |
> |---|---|---|---|
> | Recommended | 2.93× | 7.76× | 1.71× |
> | No run-ahead | 2.21× | 6.54× | **0.66× misses** |
> | Scalar chain blocking | 1.27× | 7.76× (invariant) | **0.37× misses** |
> | Both absent | **0.37× misses** | 6.54× | **0.08× misses** |
>
> 1. **Run-ahead is no longer a deadline requirement.** It was recorded as "the only deadline miss
>    the analysis produces." **No row in the frozen set now misses from either pathology alone.**
>    Downgraded statement that survives: *you may skip at most one of the two* — QUAD-MPC, at its
>    **measured** worst-case tail, misses at 0.37× when both are absent. Run-ahead moves from
>    *requirement* to *required unless chain blocking is full-width*.
> 2. **The ≥8-wide chain-blocking RTL requirement** loses its deadline miss and keeps its lever: it
>    is worth **2.3× of QUAD-MPC's worst-case margin**, the largest single lever in the analysis,
>    but the justification is now a thin margin on one measured row rather than a miss on two.
>    Still the first thing to check in Phase 1.
> 3. **The 64 B/cycle bandwidth floor now rests on QUAD-MPC alone.** It does not move — the
>    surviving row is the one whose tail is *measured* (74 iterations) rather than estimated, so
>    the floor rests on the better of the two pieces of evidence. But a single-row justification is
>    the exact structural weakness that caused the two previous sizing re-runs. Said out loud.
> 4. **Not lost, and now sharper:** "a CPU already meets this deadline" gets *stronger*, and "the
>    thinnest codegen margin in the set" transfers to ARM-MPC (3.78×).
>
> **The thinnest-margin problem moves rather than vanishing**, and this is the part to be careful
> about: **PWR-FCS1 (one-step, 27 candidates — what MCUs ship today) is at 0.61× of the codegen
> rule, and PWR-FAST (two-level, 100 kHz) at 0.69×.** Two adjacent, real configurations of the
> same socket sit *under* the bar. Honest reading: **for one-step FCS-MPC, tell the customer to
> use the MCU.** The minimum viable version of this socket is a multi-step horizon.
>
> #### 4. Would 16-bit operands fix the latency? Partly — and the array's format is wrong
>
> `precision_study.py` was extended with the configuration we would actually build. Previously
> every 16-bit-operand row was paired with an **fp64** solve, which no CM SKU offers on the ADMM
> path, so the proposal had never been tested. Relative error in the commanded control action
> against the fp64 solution of the same problem [MEAS numerics]:
>
> | Configuration | Rel. error in x |
> |---|---|
> | fp32 operands, fp32 solve — **CM-100 today** | 8.2e-07 |
> | **fp16 operands, fp32 solve — the proposal** | **1.9e-03** |
> | bf16 operands, fp32 solve — with a bf16 array | 1.9e-02 |
> | fp64 operands, **bf16 solve** | 6.1e-02 |
>
> - **Keeping the solve at fp32 costs nothing:** fp16/fp32-solve and fp16/fp64-solve score
>   *identically* (1.946e-03). The operand precision, not the solve, limits the mixed scheme. So
>   the split datapath — narrow operands, `prec` solve, `prec` accumulate — is numerically sound.
> - **fp16 beats bf16 by 10× at the same width.** Mantissa bits dominate for these matrices, the
>   opposite of the ML tradeoff. **The array's bf16 specification is wrong for the solver path.**
> - **New (C4): FCS-MPC has a completely different precision floor**, because its output is a
>   discrete argmin rather than a real number. Over 729 sequences × 200 trials, fp16 changes the
>   selected switching sequence on **3.0%** of ticks at a mean excess cost of **1.5e-06** of the
>   candidate cost spread; bf16 changes **11.0%** at 9.2e-05. A disagreement between two near-equal
>   candidates is not an error. **Different algorithm, different floor — that is a MACIF fact.**
>
> Modelled effect of fp16 operands + fp32 solve + fp32 accumulate [DER]:
>
> | Row | Latency | Energy | Speedup | perf/W |
> |---|---|---|---|---|
> | QUAD-MPC | 35.96 → 24.07 µs (**1.49×**) | 5.46 → 3.05 µJ (**1.79×**) | 1.74× → 2.60× | 14.1× → **25.3×** |
> | PWR-CCS | 4.29 → 3.69 µs (**1.16×**) | 0.359 → 0.237 µJ (**1.51×**) | 1.08× → 1.26× | 15.2× → 22.9× |
>
> **Why it is not 2×, which is the adversarial point:** halving operand width doubles the GEMV
> rate, but once it halves the **vector unit becomes co-binding** — ADMM's projection, dual update
> and over-relaxation are a fixed element-op count that 16 fp32 lanes cannot narrow, and the
> dependency drain does not narrow either. *A bandwidth-bound row stops being bandwidth-bound
> before the bandwidth doubling is spent.* Same shape as the 128 B/cycle finding.
>
> **Verdict: adopt fp16 as the 16-bit solver operand format and declare it in MACIF; do not rely
> on it for latency.** ~2400× more operand error (8e-7 → 1.9e-3, i.e. ~0.2% in a commanded control
> action) is plausibly fine for a drive and *not* obviously fine when the constraint is a hard
> current limit — a per-problem codegen option a customer opts into, never a default. It does
> nothing for the re-specified PWR-MPC row, which is not bandwidth-bound in the first place.
>
> #### 5. The MCU-class contradiction, settled — and CLAUDE.md's framing needs correcting
>
> The 25 µs T0 constant was never an MCU-class number. Decomposed [EST], each term naming a
> mechanism that can be measured on its own:
>
> | Term | T0 (rich OS) | T1 (coherent) | T0M (MCU) | T1M (MCU) |
> |---|---|---|---|---|
> | driver / RTOS API | 8.0 | — | 1.5 | — |
> | buffer map + pin | 6.0 | — | — | — |
> | cache clean + invalidate | 6.0 | — | — | — |
> | IRQ + scheduler wakeup | 5.0 | — | 0.5 | — |
> | descriptor write | — | 0.1 | — | 0.10 |
> | doorbell store | — | 0.1 | — | 0.05 |
> | completion poll / latency | — | 0.6 | — | 0.15 |
> | **TOTAL µs** | **25.0** | **0.8** | **2.0** | **0.30** |
>
> **Two of T0's four terms — buffer map/pin and cache maintenance — are artefacts of an MMU and a
> cache.** An all-SRAM MCU has neither. T1M is *cheaper* than T1 because there is no coherence
> round trip either. Ablation [DER]:
>
> | Row | T1/T0 latency | T1M/T0M latency | T1/T0 energy | T1M/T0M energy |
> |---|---|---|---|---|
> | PWR-MPC (v1.2) | 16.1× | **2.95×** | 7.7× | **1.70×** |
> | PWR-CCS | 7.0× | 2.90× | 2.4× | 1.34× |
> | EKF-24 | 10.9× | 4.29× | 3.6× | 1.63× |
> | QUAD-MPC | 2.1× | 2.87× | 1.3× | 1.21× |
>
> **Answers, in order:**
>
> - **What does T1 zero-copy actually buy on an MCU-class host? Almost nothing that is zero-copy.**
>   Of the 2.17 µs T1M advantage on the re-specified row, **1.70 µs (78%) is invocation overhead
>   and 0.47 µs (22%) is an avoided on-chip copy** — and the copy costs anything at all only
>   because MCU buses are narrow, not because the bytes are far away. The data-movement half
>   collapses for two independent reasons: the bytes are cheap (~1.2 pJ/B on-chip [EST] versus
>   ~12 pJ/B LPDDR) and there are barely any (**716 B** of persistent state for FCS-MPC, versus
>   8.4 KB for the retired ADMM row and 100 KB for QUAD-MPC).
> - **Is the per-call overhead really descriptor/doorbell/syscall cost rather than data movement?
>   Yes, structurally** — `call_overhead_us` is charged independently of `bytes_moved`, and the
>   decomposition shows which terms survive the removal of DRAM. **But the honest magnitude is far
>   smaller than the headline: the advantage falls from 31× (25/0.8) to 6.7× (2.0/0.30), and in
>   absolute terms from 24.2 µs to 1.7 µs.** Against a 25 µs period that is 6.8% — real, not
>   decisive. Against the 5 µs period of the fastest deployed direct MPC it is 34%, and there it
>   *is* decisive. **The invocation-overhead pitch for MCU class is rate-dependent and needs
>   100 kHz+ to carry a deadline argument.**
> - **What buys PWR-MPC its deadline there, and does it still meet it?** It meets it in **every**
>   tier including T0M (3.28 µs against 25 µs). Nothing "buys" it — the re-specified workload is
>   ~640 cycles of arithmetic and the deadline was never the issue.
>
> > **Correction to CLAUDE.md's framing.** "Zero-copy and residency are *void* in MCU class" is
> > right about **residency** and about **coherency**, and wrong to bundle **invocation overhead**
> > in with them: invocation cost is descriptor/doorbell/RTOS cost, it survives the removal of
> > DRAM, and it is ~78% of what remains. **Withdraw the zero-copy argument from the MCU-class
> > pitch entirely** and replace it with *low invocation overhead + determinism + energy*.
> > Zero-copy stays a real differentiator on rich hosts (QUAD-MPC, ARM-MPC, the ML rows), where
> > DRAM and cache maintenance exist. This also resolves the §5.1 / §4 tension below.
>
> **Adversarial note that cuts against us and was never charged before:** if the CM addresses host
> SRAM directly (the D2/D3 tightly-coupled story) it inherits the host's bus width — 8 B/cycle at
> 200 MHz is **1.6 GB/s [EST], 4× *less* than the LPDDR path it replaces**. The data is closer and
> the pipe is narrower. Fine for a 716-byte working set, not fine for anything that streams; note
> QUAD-MPC is *slower* on T0M (102 µs) than on rich-host T0 (76 µs) for exactly this reason. **Any
> MCU-class SKU must keep its own scratchpad**, which means the copy does not disappear — it just
> becomes cheap.
>
> #### 6. Microarchitecture consequences
>
> 1. **Invocation overhead is the binding resource on FCS-MPC; inside the datapath it is the
>    VECTOR UNIT** (525 of 649 datapath cycles, 81% — cost evaluation and argmin). No other row in
>    the set has that shape. *An earlier draft of this analysis claimed we were 5× under-provisioned
>    on elementwise throughput versus SME2; that was a unit error (lanes/cycle compared against a
>    FLOP figure) and it changed the recommendation.* In the same units: **CM-100 12.80 vs SME2
>    12.32 G element-ops/s — parity, not a 5× deficit.**
> 2. **Recommended response is two accumulator features, not more lanes.** 16 → 32 lanes costs
>    **+0.143 mm² (+20% of the core)** and buys 1.26× — rejected on that ratio, since the
>    merged-FMA construction doubles the fp64 cluster with it and half the latency is not datapath.
>    Instead: **broadcast/bias initialisation of the accumulator** (folds the B@U broadcast-add) and
>    **square-accumulate** (folds the ‖r‖² cost reduction). Modelled as PWR-FUSE: vector cycles
>    **525 → 154**, datapath 0.81 → 0.35 µs, latency 1.61 → 1.15 µs, **0.62× → 0.88×** vs the
>    fastest CPU, at near-zero area. Note the descriptor margin *falls* 6.76× → 3.86× (fewer, fatter
>    descriptors is the point, but it is a trade). **This is the same L0-staging argument already
>    flagged as the highest-value model extension: fold work into the accumulator at O(1) area
>    instead of buying O(T²) of array.**
> 3. **MACIF v0 gains two entries.** A broadcast-initialised, square-accumulating accumulator is a
>    property of the swappable macro, so it belongs alongside `accumulator_width[format]`. And
>    **fp16 must be a declared format distinct from bf16** — a macro that silently substituted bf16
>    would cost 10× the control-action error, and the runtime must refuse it exactly as it refuses
>    a missing fp64 path.
> 4. **Bandwidth-first survives with a third, more general mechanism.** It was argued from batch-1
>    GEMV (no reuse) and from bank-geometry energy. FCS-MPC adds: *even a workload with a real batch
>    dimension is operand-fetch-bound if its reuse depth is set by a problem dimension rather than
>    by the tile.* Stronger than "batch-1 solvers have no reuse."
>
> #### 7. What would firm this up, in order of value per day
>
> 1. **Confirm the FCS-MPC parameter set against a real deployed controller** (n_x, n_y, N_p, Ts,
>    |U|, terminal cost / move blocking). Everything in §1 hangs on the shape, and the shape is
>    cited literature, not a measured deployment.
> 2. **Measure a C2000 CLA or AURIX PPU task-dispatch latency.** The entire MCU-class answer rests
>    on T0M = 2.0 µs and T1M = 0.30 µs, both [EST]; it is a half-day on a LaunchPad board.
> 3. **Measure FCS-MPC on an F28379D and on an A78/SME2 board.** The CPU side of this row is now
>    the dominant uncertainty — the elementwise-rate correction alone moved a verdict across parity.
> 4. **Re-run the C4 precision experiment across plant families and switching-set sizes.** One
>    plant family with a fixed seed establishes a mechanism, not a distribution.

> ## ⚠ SUPERSEDED (kept for the audit trail) — correction notice, solver-numerics audit
> **The latency figures in §1 below are wrong and are being regenerated.** An independent audit
> of the operation counts found seven errors, the largest being a **phantom transcendental
> term**: the model charged ADMM's box projection `Π_[l,u]` as 2·n_con divide/sqrt operations,
> when it is min/max — zero divides. Because the CPU model charges 12 cycles per transcendental
> against the CM's 4/vector-lane, ~640 fake divides per iteration penalized the CPU ~3× harder
> than the CM. **That single error accounted for the entire apparent latency advantage.**
> Corrected, QUAD-MPC goes from 1.47× to **0.92× — the CM is slower than the fastest CPU
> baseline.** So the true range is roughly 0.9–3.2×, not 1.2–2.9×, and on the flagship benchmark
> we are at parity or behind.
>
> **perf/W is robust** (17.0 → 15.7 on QUAD-MPC): it is a ratio of two quantities that both
> scale linearly with the op count, so op-count errors cancel there and do *not* cancel in
> latency or deadline feasibility. **The ≥5× kill criterion and the bandwidth-first conclusion
> both survive; every latency-derived claim must be re-derived.** This is the third
> flattering-assumption catch this phase (after undercounted overhead energy and GEMV-modelled-
> as-GEMM), which is evidence the review discipline works — and a reason to trust no number here
> that has not survived an adversarial read.

## 1. Headline: the premise survives, but for a different reason than assumed

| | Result |
|---|---|
| **Kill criterion (≥5× perf/W on solvers)** | **PASS, 5/5** — modelled 17–25× perf/W |
| **Latency advantage** | **Much weaker than assumed: 1.2–2.9×**, not the 5–10× the thesis implied |
| **Deadlines met at baseline config** | 5 of 7 (ARM-MPC and SLM-DECODE miss) |

The original thesis said Argmind wins because iterative solvers are *latency*-bound and a
tightly-coupled accelerator makes iterations faster. The model says the durable advantage is
**energy, not speed**: ~20× perf/W but often under 2× wall-clock. On three of five solver
benchmarks a CPU already meets the deadline unaided.

**Repositioning this implies:** lead with *energy per solve*, *determinism*, and *CPU
offload/consolidation* (freeing the application core), not with "the CPU can't do this." For
most beachhead problems the CPU can — it just costs 20× the energy and gives up determinism.

---

## 2. The finding that changes the microarchitecture

Splitting matrix-vector from matrix-matrix work exposed an error in the v0 assumptions. ADMM's
inner loop (`P@x`, cached triangular solves, `A@x`, `Aᵀ@y`) is **all matrix-vector at batch 1**:
every MAC consumes a fresh matrix element with no reuse, so throughput is capped by scratchpad
bandwidth, not by MAC count.

**Tile size is nearly irrelevant to solver latency:**

| Benchmark | 256 MACs | 1024 MACs | 4096 MACs |
|---|---|---|---|
| QUAD-MPC | 431.8 µs | 431.8 µs | 431.8 µs |
| PWR-MPC | 32.3 µs | 32.3 µs | 32.3 µs |
| EKF-96 (matrix-matrix) | 36.2 µs | 16.5 µs | 11.7 µs |

A **16× increase in MACs buys literally zero** on the flagship benchmark.

**Scratchpad bandwidth is the knob that works:**

| Benchmark | 16 B/cyc | 64 B/cyc | 256 B/cyc |
|---|---|---|---|
| QUAD-MPC | 1511.8 µs | 431.8 µs | **161.8 µs** |
| ARM-MPC | 12427.3 µs | 3745.1 µs | **1574.5 µs** (meets deadline) |
| PWR-MPC | 92.0 µs | 32.3 µs | **17.3 µs** |

> **Architecture decision for Phase 1: a wide, heavily-banked scratchpad with a modest MAC
> array — the opposite of an NPU's provisioning.** Sizing the array like an NPU would spend
> most of the area budget on MACs that solver workloads cannot use.

This also explains structurally why NPU IP cannot be retargeted at this workload: their whole
area budget is in an array that batch-1 solver code leaves idle.

## 3. Deadline misses — both are provisioning problems, not architecture failures

- ~~**ARM-MPC** (nonlinear SQP, 500 Hz): 3745 µs vs a 2000 µs deadline. Fixed by bandwidth
  (1574 µs at 256 B/cyc). This is the benchmark that justifies the wide scratchpad.~~
  **SUPERSEDED — see §0.** Re-specified as acados SQP-RTI + HPIPM at 250 Hz: 178 µs vs 4000 µs,
  22× margin, meets its deadline at every bandwidth in the sweep including 16 B/cycle. It justifies
  nothing. The wide scratchpad is now argued from the *measured ADMM cold-solve tail* (64 B/cycle
  floor) plus a bank-geometry energy bet, not from this row.
- **SLM-DECODE**: 176 ms/token vs 50 ms, and **DRAM-bandwidth-bound** — needs ~22 GB/s
  sustained but the baseline config has 6.4 GB/s. Batch-1 LLM decode is a *memory-bandwidth*
  product, not a MAC product. Either provision a wide external port (multi-channel LPDDR5) or
  scope the ML claim to models whose weights fit resident in on-chip SRAM. **Do not claim
  general edge-LLM capability until this is resolved.**

## 4. Zero-copy / residency ablation (§3.2.1/3.2.2 of the plan)

> **⚠ SUPERSEDED 2026-07-30 by §0.3 — the two load-bearing claims in this section are both wrong.**
> (a) The "1.9× on PWR-MPC, and T0 misses the 50 µs deadline" cell was computed on a benchmark
> definition that has since been re-specified *and* at a configuration we no longer recommend. The
> re-specified row meets its deadline in **every** tier, including copy-based offload.
> (b) The whole table assumes a **rich host with DRAM, an MMU and a cache**. On an MCU-class host —
> which is the socket this section was used to justify — the advantage is **2.9× not 16×**, and
> **78% of what remains is invocation overhead, not data movement.** Use §0.3's tier table.
> Kept for the audit trail only. **Do not quote any figure below.**

T1 (zero-copy + residency) vs T0 (copy-based offload), latency ratio:

| Benchmark | T1 advantage | Note |
|---|---|---|
| EKF-24 | **13.3×** | Tiny problem: overhead *is* the workload. 2.1 µs vs 27.6 µs |
| PWR-MPC | **1.9×** | T0 **misses** the 50 µs deadline (60.3 µs); T1 meets it at 32.3 µs |
| EKF-96 | 2.2× | |
| QUAD-MPC, ARM-MPC, ML rows | 1.1× | Larger/slower workloads amortize the overhead |

**Verdict: the zero-copy/residency work is justified, but specifically for small problems at
high rates** — exactly the embedded-control beachhead. It is what makes the 20 kHz motor-drive
socket possible at all, and it is worth ~13× on the smallest workloads. On large problems it is
nearly irrelevant, so it should be sold as a *real-time-control* feature, not a general one.

> **Corrected verdict (2026-07-30):** zero-copy and residency remain justified **on rich hosts**,
> where DRAM and cache maintenance actually exist — that is QUAD-MPC, ARM-MPC and the ML rows, plus
> any Linux/AUTOSAR-Adaptive integration. **On MCU-class hosts the differentiator is low invocation
> overhead and determinism, not zero-copy**, and the invocation argument only carries a *deadline*
> above ~100 kHz. Sell it as a real-time-control feature, and say which host class you mean.

## 5. Competitive teardown — the gaps are wider than expected

Sourced from vendor product briefs and technical overviews (full citations at bottom).

| | Data-dependent control flow | Call-to-call residency | Zero-copy / coherency | FP matrix throughput |
|---|---|---|---|---|
| **Arm Ethos-U85** | **No** — statically compiled command stream (Vela); no scalar unit, no branching; host MCU runs anything else | **No** — 29–267 KB is fixed-function buffering; all registers/memories cleared at reset | **No** — AXI manager + APB only; no ACE-Lite, no CHI, no SMMU | **None — INT8 weights only** |
| **Arm Ethos-N78** | No | Unclear (384 KB–4 MB SRAM, no published contract) | **Yes, partially** — AXI4 + ACE-5-Lite + Arm SMMU (the only one) | INT8/INT16 only |
| **Synopsys/MIPS ARC NPX6** | **No** — fixed conv + tensor accelerator; arbitrary control flow needs a paired ARC VPX DSP | Not published | Partial — MMUs in L1/L2 controllers; coherency protocol not published | Optional BF16/FP16 unit |
| **Ceva NeuPro-M** | **Closest** — programmable VPU per engine, hardware task-flow control, "no redispatch per layer" — but graph flow control, **not** convergence-loop control | Not published | Not published | **FP32: 32 MACs vs 4K INT8 (~1%)** |
| **Cadence Neo NPU** | **No** — requires a *second licensed core* (Tensilica NeuroEdge 130) as control processor | Not published (no SRAM size disclosed at all) | Not published | INT4–INT16 + FP16 |

**Three structural gaps confirmed:**

1. **Every commercial NPU IP compiles control flow away offline.** None publishes a mechanism
   for the accelerator itself to decide "iterate again" from a computed value. Cadence's answer
   costs a second IP license. This is the moat, and it is real.
2. **Nobody has usable FP matrix throughput.** Ethos-U85 is INT8-weights-only; Ceva's FP32 path
   is ~1% of its INT8 rate. **Incumbents cannot add solver support in software** — they lack the
   arithmetic. This is stronger than the plan previously assumed.
3. **Batch-1 / small-tensor latency is universally unpublished**, and Ethos-U85 is architecturally
   N=1-only in its feature-map formats while still quoting only large-tensor throughput. There is
   no public number any incumbent can be held to on small matrices — the regime solvers live in.
   Publishing credible small-matrix latency data is therefore an uncontested marketing position.

**Calibration warning for our own claims:** the only *measured-silicon* efficiency reference is
MediaTek's NVE (ISSCC 2024): **23.2 TOPS/W at 12-bit in 3 nm**. Vendor claims run far above it
(Ceva 350 TOPS/W "at 3nm", Synopsys 30 TOPS/W at 5nm) but are pre-silicon estimates at low or
unstated precision, usually with sparsity. **Benchmark against measured numbers, publish our own
with node and precision attached, and never compare against marketing figures.**

### 5.1 Addendum (2026-07-29): control-domain incumbents — the row the teardown owed

The teardown above covered **NPU IP**. It did not cover the products actually occupying the
socket the beachhead targets. Desk research only; no measurement. Ranked by threat.

**Headline, and it is a downgrade to the moat claim.** Against NPU IP the three gaps in §5 are
structural. Against the *control-domain* incumbents they are not the same gaps, and against CPU
matrix extensions **three of our four differentiators do not exist at all** — because SME2 and
RVV are *inside the CPU*, they get data-dependent control flow, coherent zero-copy access to
application memory, and call-to-call cache residency **for free, by construction.** Against
SME2/RVV our only defensible claims are **energy per solve, deterministic worst-case latency,
and core offload.** The sequencer, the zero-copy path and the generation-tag residency model are
differentiators against accelerators, not against CPUs.

#### The capability matrix

| | In-fabric data-dependent control flow (decide "iterate again" from a computed residual, no host) | Call-to-call residency with an explicit contract | Zero-copy / coherent host memory | FP32 matrix or vector throughput | Threat |
|---|---|---|---|---|---|
| **Four NPU IP families** (§5) | **No** — all compile control flow away offline; Cadence needs a second licensed core | Not published, or fixed-function buffering | Ethos-N78 only (ACE5-Lite + SMMU) | **None usable** (INT8-weights-only; Ceva FP32 ≈1% of INT8) | 5 |
| **Infineon AURIX TC4x PPU** (Synopsys ARC EV7x/EV71) | **Yes** — executes its own instruction stream in its own ISA; "does not execute TriCore instructions and vice versa" | VCCM exists; **size and inter-invocation persistence contract not published** | **Not published**; DMA mentioned, coherency with TriCore memory not stated | Vector FPU is an **option** in the EV7x family (IEEE-754, single or half precision); **whether Infineon enabled it in TC4x is not published**. No GFLOP/s, MAC/cycle, or PPU clock published anywhere | **2** |
| **TI C2000 CLA** | **Yes** — has its own conditional delayed branch (`MBCNDD`) | Inherent, not contractual — an all-SRAM MCU has nothing to evict | Message RAM only, and **unidirectional** (CPU→CLA: CLA read-only; CLA→CPU: CPU read-only). 16-bit CLA address bus | **FP32 scalar**, 200 MHz on TMS320F28379D. **No FLOP/s figure published by TI** | 4 |
| **Arm SME2** | **Yes — it is the CPU** | **Yes — the CPU's caches, implicitly** | **Yes — by construction** | **Measured: 2008 GFLOP/s FP32 FMOPA**, SVL 512-bit, Apple M4 P-core cluster, 3 nm class | **1** |
| **RISC-V RVV 1.0** | **Yes — it is the CPU** | Yes — the CPU's caches | Yes — by construction | FP32 vector supported; **no matrix/outer-product primitive ratified**; **no measured FP32 GFLOP/s found in public sources** | 3 |

#### 1. Infineon AURIX TC4x PPU — the biggest unhedged risk in the Phase-0 case

| Item | Published | Class |
|---|---|---|
| Identity | Implemented as a vector processor **based on the Synopsys DesignWare ARC EV71 / EV7x architecture**; 32-bit RISC scalar core + SIMD vector unit; runs code compiled for the ARC EV71 ISA | secondary ([emmtrix](https://www.emmtrix.com/wiki/Infineon_AURIX_TC4x_Parallel_Processing_Unit_(PPU))), corroborated by the existence of a [Synopsys MetaWare Toolkit for AURIX TC4x](https://mips.com/processor-solutions/sw-metaware/) |
| Vector width | **128-bit or 256-bit** in production TC4x; 512-bit noted for certain pre-production models. ARC EV7x family is published as a "512-bit wide vector DSP" per VPU, configurable for 8-, 16- or 32-bit operations | secondary + [Synopsys/MIPS EV7x page](https://mips.com/processor-solutions/ev-processors/ev7x-vision-processors/) (vendor) |
| **Precision — the decisive cell** | EV7x offers an **optional** "IEEE 754-compliant vector floating point unit… single precision or half precision operations." **Whether the FP option is instantiated in the TC4x PPU is NOT PUBLISHED.** Infineon-facing material describes 8/16/32-bit *element widths* without naming float support | vendor (family) + **not published** (this part) |
| VCCM size | **Not published** — neither Infineon nor Synopsys states a capacity | **not published** |
| Persistence between invocations | **Not published.** No mechanism comparable to generation tags / dirty masks is described | **not published** |
| Host memory model | **Not published** whether the PPU has coherent access to TriCore memory | **not published** |
| Throughput | **No GFLOP/s, no MAC/cycle, no PPU clock, no process node published** | **not published** |
| "up to 78× vs a single TriCore 1.8 core" | Benchmark **not identified**; the secondary source attributes it to a master's thesis and an Infineon presentation, neither accessible. No node, no precision, no workload. **Recorded as UNCOMPARABLE. Do not use this figure in any document, including as a competitor number.** | vendor claim via secondary |
| Programming | Synopsys ARC MetaWare Toolkit; an AUTOSAR Complex Device Driver plus a "PPU dispatcher" runtime; [MathWorks Embedded Coder generates PPU code](https://www.mathworks.com/help/ecoder/infineonaurix/ug/soc-ifx-tc4x-ppu.html). EV7x family toolchain supports OpenVX, OpenCL C, ONNX/TF/Caffe import | vendor |
| Safety | The EV7xFS variants are marketed as "ASIL B, ASIL C, or ASIL D Ready" with lockstep options | vendor |

**Two consequences, pulling in opposite directions.**

*Against us:* the PPU is a scalar core + SIMD vector unit + closely-coupled memory running its own
instruction stream. That is not an absence of arithmetic and not an absence of a control plane —
it is a small version of our own architecture, already qualified, already in an automotive MCU,
already supported by Embedded Coder. **If Infineon enabled the FP option, the PPU can in
principle run an ADMM inner loop with the convergence test executing in-fabric today, and the
only thing missing is a solver library — i.e. a software release.** Note also that a vision VPU
is provisioned to stream pixels, so it is plausibly *also* bandwidth-first, which makes it a
worse comparison for us than an NPU rather than a better one.

*For us:* Infineon **licensed third-party vector-compute IP** (Synopsys ARC) for exactly this
socket rather than building it. That is the strongest evidence available that an automotive MCU
vendor will license outside compute IP for control acceleration — directly relevant to
CUSTOMER_DISCOVERY.md falsifier F3, and it is also a reason to treat Synopsys/MIPS ARC as an
incumbent competitor for the *license*, not just for the socket.

#### 2. TI C2000 CLA — small, but it voids two of our differentiators in MCU class

| Item | Published | Class |
|---|---|---|
| Arithmetic | "fully-programmable independent **32-bit floating-point** hardware accelerator," "designed to execute real-time control algorithms in parallel with the C28x CPU" | [vendor](https://software-dl.ti.com/C2000/docs/cla_software_dev_guide/intro.html) |
| Clock | **200 MHz** — TMS320F28379D is dual 200 MHz C28x + **dual 200 MHz CLA**, 800 MIPS device total, with FPU and TMU | [vendor](https://www.ti.com/product/TMS320F28379D) |
| Throughput | **No FLOP/s figure published.** No vector unit is published; the CLA is a scalar FP datapath | **not published** |
| Control flow | **Supported** — own conditional delayed branch `MBCNDD`; Type 2 adds a background task, software/peripheral triggering and uninterruptible sections | [vendor](https://software-dl.ti.com/C2000/docs/cla_software_dev_guide/features.html) |
| Memory model | Memory blocks are device-family dependent and allocated to the CLA by the CPU; two dedicated **unidirectional** message RAMs; **16-bit CLA address bus** vs 32-bit pointers on C28x; shared globals must be declared in the C28x source; C support is restricted | [vendor FAQ](https://software-dl.ti.com/C2000/docs/cla_software_dev_guide/faq.html) |
| MPC / solver benchmark from TI | **None found.** Third-party peer-reviewed work runs FCS-MPC on an F28379D ([MDPI 2023](https://www.mdpi.com/2227-7390/11/21/4553)), but finite-control-set MPC is exhaustive enumeration over a switching set, **not a QP solve**, so it is not comparable to BENCHMARKS.md #1–#5 | **not published** (TI); peer-reviewed (third party, non-comparable workload) |

*Bound, clearly labelled as our inference and not TI's spec:* at the published 200 MHz, a scalar
FP datapath cannot reach the GFLOP/s range at any plausible issue width — three orders of
magnitude below measured M4 SME2 FP32. So the CLA is not a throughput threat.

**But it is a strategic warning.** In MCU class there is no DRAM and no cache, so **our zero-copy
and residency differentiators have nothing to bite on** — everything is already SRAM-resident and
there is no cache maintenance to eliminate. PLAN.md §3.2.1 already concedes T2 needs an SMMU
absent in MCU-class parts; this extends the concession to T1 and to §3.2.2. ~~**The 20 kHz
PWR-MPC socket, where §4 showed zero-copy is worth 1.9× and decides deadline feasibility, is
exactly the socket most likely to be MCU-class.** That tension needs resolving before the
zero-copy work is used to justify the motor-drive pitch.~~

> **RESOLVED 2026-07-30 — see §0.3(5).** The tension was created by charging an MCU-class host a
> rich-host launch cost. Decomposed, two of T0's four terms (buffer map/pin, cache maintenance) are
> MMU/cache artefacts that an all-SRAM part does not have, so the honest MCU-class copy tier is
> ~2.0 µs [EST], not 25 µs. **The concession above is right about residency and coherency and wrong
> to bundle invocation overhead in with them:** invocation cost is descriptor/doorbell/RTOS cost, it
> survives the removal of DRAM, and it is ~78% of the remaining 2.9× advantage. And the re-specified
> PWR-MPC meets its deadline in every tier, so nothing "decides feasibility" there any more.
> **Correct MCU-class pitch: low invocation overhead, determinism, energy per solve — not
> zero-copy.** Note also the TI comparison in the row above was already the right instinct: the
> peer-reviewed C2000 work runs FCS-MPC, which is now what our own benchmark #3 models, so the
> workload is comparable for the first time.

#### 3. Arm SME2 — the top threat, and the only measured matrix-extension silicon we have

**Measured (third-party, independent of Arm and Apple):** Apple M4, SVL **512-bit**, ZA tile
4096 B, **3 nm class** (Apple's own product statements; per-unit power **not published**):

| Instruction | Precision | P-core cluster block | E-core cluster block |
|---|---|---|---|
| FMOPA (outer product) | **FP32** | **2008 GFLOP/s** | 357 GFLOP/s |
| FMOPA | **FP64** | 503 GFLOP/s | 89 GFLOP/s |
| FMOPA | FP16 → FP32 accum | 2010 GOPS | 358 GOPS |
| BFMOPA | BF16 → FP32 accum | 2011 GOPS | 357 GOPS |
| Streaming-SVE (non-outer-product) | FP32 | ≈250 GFLOP/s (and only 31 GFLOP/s via a naive vector formulation) | — |

Sources: [Hello SME microbenchmarks, Univ. Jena](https://scalable.uni-jena.de/opt/sme/micro.html)
and [independent M4 SME exploration](https://github.com/tzakharko/m4-sme-exploration/blob/main/reports/01-sme-overview.md).
Both are **measured silicon**, precision and vector length stated, **node stated only as "3 nm
class," power per unit not published.**

Three details that matter more than the headline number:

1. **The SME block is shared per CPU cluster, not per core** — one large block for the four
   M4 P-cores, a smaller one for the six E-cores. So 2 TFLOP/s is a *cluster* figure, and
   worst-case latency for one thread depends on what the other cores in the cluster are doing.
   **This is the strongest determinism argument we have against SME2, and it is architectural
   rather than incidental.**
2. The gap between the outer-product path (2008 GFLOP/s) and the streaming-vector path
   (≈250 GFLOP/s), and the further collapse to 31 GFLOP/s on a naive formulation, means SME2
   performance is **highly dependent on hand-tuned kernels**. Our codegen-versus-their-library
   argument is real here — but it is a software-ecosystem argument, not a hardware one.
3. Reduced precision buys nothing: the source notes "the performance of instructions with
   reduced precision is underwhelming" because halved instruction throughput offsets the wider
   dot products.

**Shipping Arm-designed silicon.** The Lumex CSS platform's **C1-Ultra, C1-Premium, C1-Pro and
C1-Nano are SME2-enabled, optimized for 3 nm**, with SoCs expected in mobile devices during 2026
([CNX](https://www.cnx-software.com/2025/09/10/arm-introduces-lumex-platform-with-sme2-enabled-c1-cpu-for-edge-ai-mali-g1-gpu/),
[Arm C1-Ultra](https://www.arm.com/products/silicon-ip-cpu/c1-ultra)). Arm's uplift figures —
"5x faster AI performance," "4.7x lower latency for speech-based workloads," "2.8x faster audio
generation," C1-Ultra "25% single-thread vs Cortex-X925" — are **vendor claims with workloads
only loosely named and no precision attached to the multipliers.** Not usable as comparison
points.

**Availability where our customers live is the real finding.** Cortex-A320 (the small Armv9.2-A
edge/IoT core) is published with **SVE2**, and Arm does not state SME2 for it
([Arm](https://www.arm.com/products/silicon-ip-cpu/cortex-a/cortex-a320)). For the automotive
line, the reviewed Cortex-A720AE and Cortex-A78AE product pages **do not state SME2 support**
([A720AE](https://www.arm.com/products/silicon-ip-cpu/cortex-a/cortex-a720ae)) — record that as
**not published, not as absent**, since SME is optional in Armv9.2-A. Practically:
**a robotics, industrial or automotive designer cannot buy an SME2 automotive-grade or
industrial part today.** The SME2 threat currently exists in Apple silicon and 2026 mobile SoCs.
**That is a timing window, not a moat**, and Arm closing it is a routine roadmap item rather than
a research problem.

#### 4. RISC-V RVV — weaker near-term, but it does not protect the solver claim

- RVV 1.0 is a **vector** extension with FP32 support and **no matrix/outer-product primitive**.
- Shipping, buyable today: **SpacemiT K1 / X60**, 8-core, RVV 1.0 with **256-bit** vector
  registers, on a ~$100-class Banana Pi BPI-F3 ([BananaPi docs](https://docs.banana-pi.org/en/BPI-F3/SpacemiT_K1)).
  The "2.0 TOPS" AI figure is a **vendor claim with precision and node not stated** — not
  comparable. SiFive X280 is published at VLEN=512.
- **Measured FP32 GFLOP/s for any RVV silicon: not found in public sources.** Phoronix's RVV
  review of a 16-core SpacemiT K3 publishes no FLOP/s and notes the SoC "doesn't expose any power
  metrics under Linux at this time"
  ([Phoronix](https://www.phoronix.com/review/risc-v-rvv-vector-benchmarks)). This is a hole in
  our own competitive picture, not just theirs.
- **Matrix extension status: not ratified as of 2026**, and fragmented. Two task groups —
  **IME** (reuses the existing V registers, adds no new architected matrix state, HPC-focused)
  and **VME** — with RISC-V's own TSC minutes recording concern about "ecosystem fragmentation"
  and that "RISC-V is already behind schedule for matrix extensions"
  ([TSC 2026-02-04](https://riscv.atlassian.net/wiki/spaces/TSCX/pages/1100742657/2026-02-04+TSC+Meeting),
  [IME TG](https://github.com/riscv-admin/integrated-matrix-extension)). Andes implements an
  `amm` instruction per IME "Option E."

**The uncomfortable nuance.** Per §2, our solver inner loop is GEMV and **bandwidth**-bound; a
plain vector unit is the correct shape for GEMV and no matrix extension is required. So RVV's
missing matrix extension protects the **ML/GEMM** half of our pitch and **does not protect the
solver half at all.** An RVV core with sufficient load bandwidth is a legitimate solver
competitor today, and like SME2 it inherits control flow, coherency and cache residency for free.

#### Structural gap or feature gap — the verdict, per competitor

| Competitor | Gap we exploit | Kind | Closable in a software release? |
|---|---|---|---|
| Four NPU IP families | No FP matrix arithmetic; no control plane | **Structural** | **No — needs new silicon.** §5 conclusion stands unchanged |
| **AURIX TC4x PPU** | Possibly none. Contingent entirely on whether the EV7x FP option is instantiated | **Feature gap at most — possibly no gap** | **Yes, if FP is present** — it needs a solver library, not silicon. Our residual claims (coherent host path, explicit residency contract, bandwidth provisioning) are all *not published* for the PPU, so we cannot currently claim them as gaps either |
| TI C2000 CLA | FP32 throughput (scalar, 200 MHz) | **Structural in our favour on arithmetic** | No — but irrelevant, because in MCU class **our zero-copy and residency differentiators are void**, and the CLA already has in-fabric control flow |
| **Arm SME2** | Perf/W; deterministic WCET under cluster contention; core offload. **Nothing else** | **Not a moat — a timing window** | **N/A** — SME2 needs no retargeting, only libraries. Bringing SME2 to automotive-grade cores needs new silicon, but that is Arm's routine roadmap |
| RISC-V RVV | ML/GEMM throughput (no matrix primitive, spec late and fragmented). **On solvers, effectively nothing** | **Feature/timing on ML; no structural gap on solvers** | Matrix ops need new silicon *and* an unratified spec; GEMV needs neither |

#### What this does to the three "settled" claims

1. **"Incumbents can't retarget in software."** True for NPU IP. **Not established for the
   PPU** (contingent on the FP option), and **false for SME2 and RVV** — they need no
   retargeting at all, only a tuned solver library. The claim must always carry the scope
   qualifier "commercial NPU IP."
2. **"Batch-1 / small-matrix latency is universally unpublished."** Survives and *widens*: no
   FLOP/s of any kind is published for the PPU or the CLA, and no measured FP32 figure exists
   publicly for RVV silicon. Publishing credible small-matrix latency and energy data across
   PPU / CLA / SME2 / RVV is a stronger uncontested position than §5 claimed.
3. **The moat narrows to three things:** energy per solve, deterministic worst-case latency
   (strengthened by the cluster-shared SME block), and the codegen/software ecosystem. **Not
   arithmetic, not control flow, not coherency.** Positioning, patents and the whitepaper should
   all be re-aimed accordingly.

#### The eval-board experiment that settles F4 — specified in advance

Desk research cannot resolve the decisive cells: the PPU's FP option, any FLOP/s figure for PPU
or CLA, measured RVV FP32, and every energy number. This specifies the experiment so the
purchase decision is about cost, not scope.

| Platform | Part / board | Gating item | Cost class |
|---|---|---|---|
| PPU (threat #2) | An Infineon AURIX TC4x application kit | **The Synopsys ARC MetaWare Toolkit for AURIX is the gate, not the board** — a licensed toolchain. Establish access and price *before* buying hardware | board low; toolchain **unknown, possibly prohibitive** |
| CLA | LAUNCHXL-F28379D (dual 200 MHz C28x + dual 200 MHz CLA) | none — free toolchain (Code Composer Studio) | very low |
| SME2 | Any Apple M-series Mac. **No automotive/industrial SME2 part exists to buy**, so this is the only measurable SME2 silicon; results are an *architectural* upper bound, and Apple remains a non-target for integration | none | low if already owned |
| RVV | Banana Pi BPI-F3 (SpacemiT K1 / X60, RVV 1.0, VLEN 256) | none | very low |

Measurements each platform must produce, using the *same* OSQP/acados problem formulations as
BENCHMARKS.md #1–#5:

1. **Wall-clock per solve and per iteration** — median and, separately, worst case over ≥10⁵
   invocations.
2. **Energy per solve measured at the rail**, not estimated from datasheet TDP. Report the
   measurement point and method.
3. **Achieved FP32 GFLOP/s on the GEMV inner kernel**, against that platform's published or
   measured peak — this is the number that tells us whether their provisioning is
   bandwidth-limited the way ours is.
4. **Sustained load bandwidth in B/cycle** into the compute unit's local memory (VCCM, CLA data
   RAM, L1). §2 says this is the only knob that matters; we need theirs.
5. **Can the convergence test run without host involvement?** If yes, measure it. If no, measure
   the per-iteration host round-trip cost — that is the size of our sequencer advantage, in
   microseconds, on real silicon.
6. **Worst-case vs median latency with an adversarial co-runner** pinned to the same
   cluster/core complex. This is the determinism claim, and for SME2 it is the *only* claim.
7. **PPU only, and decisive: does an FP path exist at all?** Compile an FP32 kernel with
   MetaWare and check whether it vectorizes to the FPU or falls back to scalar/emulation.

**F4 triggers if** the PPU — or an SME2/RVV part in the same power class — runs BENCHMARKS.md
#1–#3 within **2×** of our modelled Argmind energy per solve. Given the model's stated ±2×
uncertainty, a result inside 2× must be read as "no measurable advantage," not as a narrow win.

**Two honest caveats to state in whatever we publish from this.** (a) Node is **not published**
for TC4x, F28379D or the SpacemiT K1, and is only "3 nm class" for M4 — so cross-platform
energy comparisons cannot be node-normalised and must be reported with nodes attached and the
limitation stated in the same sentence. (b) These same boards **also produce the measured CPU
baselines that PLAN.md §10 lists as Phase-0 critical-path item 1**, where the modelled CPU
efficiency assumption is the largest error source in every ratio in this document. **One
purchase closes two critical-path items.** That is the argument to hand the founder.

---

## 6. Revised go/no-go recommendation: **GO, with three amendments**

1. **Re-provision the architecture**: bandwidth-first (wide banked scratchpad, modest array).
   Model the area/power cost of 256 B/cycle banking before committing.
2. **Re-position the pitch**: energy per solve + determinism + core offload. Drop implied
   large latency wins; on many problems the CPU meets the deadline and we must say so.
3. **Scope the ML claim**: resident-weight models only until an external-bandwidth story exists.
   Keep VIT-TINY-class claims (32× perf/W, meets deadline); defer general edge-LLM claims.

**Highest-value next work, in order:**
1. Measure the real CPU baselines (OSQP/acados on A55- and A78-class hardware). The modelled
   CPU efficiency assumption is the largest error source in every ratio above.
2. Cost the wide-scratchpad banking in area and power — the whole architecture now rests on it.
3. Verify the four "not published" competitor cells via evaluation NDAs or customer interviews
   before relying on them in collateral.

---

## 7. First MEASURED data — real OSQP on QUAD-MPC (2026-07-29)

Harness: [perf_model/measure_osqp_baseline.py](perf_model/measure_osqp_baseline.py); output in
[perf_model/results_osqp_measured.txt](perf_model/results_osqp_measured.txt). Run on x86, so
**timings do not transfer to Arm — but iteration counts and problem structure do**, and both
were previously assumed rather than measured.

### Two model inputs were wrong

**1. Constraint rows overcounted 2×.** OSQP expresses a two-sided bound `l ≤ Ax ≤ u` as *one*
row; the model counted two (`2*n_var + 2*N*n_x`). Real problem: **320 rows, not 640.** This
inflated the `A@x` and `Aᵀ@y` terms in every solver benchmark. Fixed.

**2. Iteration count ~3× too high.** The model assumed 40 ADMM iterations. Measured, with
warm-starting from the previous tick:

Final measured numbers, on the **physically coupled** plant (rotor-thrust inputs through the
mixing matrix, plus drag and gyroscopic coupling — dense condensed Hessian, `nnz(P)`=6400, 100%):

| Scenario | tol | cold | p50 | p95 | p99 | max |
|---|---|---|---|---|---|---|
| settle (hover, then hold) | 1e-3 | 60 | 1 | 4 | 7 | 60 |
| settle | 1e-6 | 73 | 12 | 17 | 20 | 73 |
| **aggressive** (feasible setpoint steps + gusts) | **1e-3** | 60 | **6** | **7** | 9 | 60 |
| aggressive | 1e-6 | 73 | 20 | 21 | 22 | 74 |

So `iters=15` is defensible as a design point between 1e-3 (p95=7) and 1e-6 (p95=21) accuracy —
but the **cold-solve figure of 60–73 is the real-time requirement**, because any mode change,
fault recovery, or reference discontinuity invalidates the warm start and the deadline must
survive the next tick.

**This measurement took three passes to get right, and the first two were wrong in ways that
would have flattered us.** Recording all three, because the pattern is the point:

1. *Quantized counts.* OSQP's default `check_termination=25` reports iterations in multiples of
   25. The first run showed a suspiciously flat "25 iterations" across four tolerance decades.
   Fixed by setting `check_termination=1` when the count is the measurand.
2. *Decoupled plant.* The first quadrotor model applied inputs one-per-state with diagonal
   inertia and no drag, which **decouples into four independent SISO channels** — the condensed
   Hessian came out 4-block-diagonal at 25% density (`nnz(P)`=1600), and nobody would solve that
   as one 80-variable QP. Caught by the solver-numerics audit, not by us. Fixed by using the
   four rotor thrusts through the mixing matrix.
3. *Infeasible scenario.* The first "aggressive" scenario used ±3 m setpoint steps against a
   100 ms horizon and ±2 N per-rotor authority — physically unreachable, so OSQP was driven
   primal-infeasible and reported p95 ≈ 900 iterations that measured nothing but struggle.
   Caught by checking solver status rather than just reading the numbers. Fixed by making the
   scenario reachable; the harness now flags and discards any run with non-`solved` ticks.

### What this does to the conclusions

Per-solve work drops ~4–6× (2× constraints × ~3× iterations). Both sides of the comparison
scale together, so **perf/W is essentially unchanged at 17–25×** — that conclusion is robust.
But absolute latencies fall, and the consequence is uncomfortable:

> **QUAD-MPC: modelled CM 90 µs vs CPU 133 µs against a 1000 µs deadline.** The CPU meets the
> flagship benchmark with 7.5× margin. On this benchmark the product case is *entirely* energy
> and determinism — there is no latency story to tell at all.

### But the determinism argument got stronger

The measurement also quantified CPU **variance**: iteration count swings from p50=12 to max=60
on the same problem, a 5× spread driven by disturbances and active-set changes. A hard
real-time system must be provisioned for the worst case, not the median. That is the honest,
measured form of the determinism pitch — not "we are faster," but "our worst case is bounded and
close to our average, and a CPU's is not." Worth building the Phase-1 story around, and worth
measuring properly on Arm.

Also: with only ~15 iterations per call rather than 40, fixed per-call overhead is a larger
fraction of each solve, which **strengthens** the zero-copy/residency case (§4) rather than
weakening it.

### Still not measured

Arm latency, Arm energy, achieved SIMD efficiency on Arm, and iteration counts for the
nonlinear SQP case (ARM-MPC) — where real deployments often use a real-time-iteration scheme of
*one* QP solve per tick, which if true would make the model's 60 badly wrong. Flagged for the
solver-numerics audit.

---

## 8. Scaling analysis (2026-07-31) — the band is narrow, and that is the product

Inputs: [perf_model/scaling_study.py](perf_model/scaling_study.py) → raw output
[perf_model/results_scaling_study.txt](perf_model/results_scaling_study.txt).
**All figures modelled-estimate, 16nm-class, ±2×, NOT QUOTABLE EXTERNALLY.** Precision is stated
per row (solvers fp32 except ARM-MPC fp64; ML int8). 29 new regression tests
(`tests/test_scaling_study.py`); suite is now 336.

PLAN.md §5 markets a "parameterized" core and Phase 4 lists "multi-core scaling"; §9
simultaneously warns "few validated configs, not full matrix". Neither the axes nor the
multi-core question had been analysed. Both are now priced.

### 8.0 Headline

**Exactly two axes scale by more than 3× over their defensible range: scratchpad bandwidth
(64 → 256 B/cycle, 3.87× on QUAD-MPC worst case) and the fp64 FMA cluster (4 → 16 units, 3.76× on
ARM-MPC).** Measured spans over the same ranges: frequency 1.64×, capacity 1.29× *as a cost*,
MAC array **1.00×**. Everything else is a fit, a cost, or a distraction. That is a ~4×
configuration band, and it is a legitimate product position for control-domain IP where problem
sizes are set by actuator counts and physics rather than by a scaling roadmap.

**Recommended family: two validated SKUs, not three.** CM-100 (144 MACs / 192 KB / 128 B/cycle /
16 lanes / 8 fp64, 0.786 mm²) and CM-50 (64 MACs / 64 KB / 64 B/cycle / 8 lanes / 4 fp64,
**0.424 mm²**, 54% of the flagship). The floating 16-MAC solver-only SKU and a 256 B/cycle
CM-200 are both **rejected with reasons** (§8.9).

> ### Two flattering model omissions found and fixed (errors 13 and 14)
>
> Neither axis had ever produced a wall, and in both cases the reason was that the model could
> not express one:
>
> 13. **`freq_mhz` scaled latency as 1/f and left energy per operation untouched**, so the model
>     reported 2 GHz as 2.5× the speed at *lower* energy per call — leakage integrated over a
>     shorter window fell and nothing offset it. There was no supply-voltage term. Added
>     `argmind_model.vdd_for_freq` (f–V table, 16nm-class [EST]) with dynamic energy × (V/V_nom)²
>     and leakage × (V/V_nom)³.
> 14. **The longest single-cycle-read bitline was a hard constant** (`ROWS_MAX_800MHZ`), so
>     raising the clock cost nothing in scratchpad area either. Added
>     `sram_cost.rows_max_for_freq` (rows_max ∝ 1/f, charge-limited bitline; honest band 1/f to
>     1/√f).
>
> Both are **exactly 1.0 / 256 rows at the nominal 800 MHz**, verified by regenerating every
> `results_*.txt` byte-identical. No established figure moved. The point is that the frequency
> axis previously looked free because the model was structurally unable to charge for it.

### 8.1 MAC array — the array is 2.25× oversized for its own bandwidth, and that is correct

A **balance rule** derived rather than swept, and it is the most useful new result in this
section. A GEMV consumes one matrix element per MAC, so its ceiling is `bw/elem` MAC/cycle while
the array supplies `tile²·scale`. Equating:

> **min tile for a given GEMV bandwidth = √(bw / (elem·scale))**, which is **√(bw/2) at fp32.**

The sweep and the closed form agree to within one step of the tile grid (pinned by a test).
Consequences:

- **`tile 12` (144 MACs) absorbs GEMV up to 288 B/cycle — and the same 288 at fp32, bf16 *and*
  int8.** Not a coincidence worth hiding: this datapath's rate multiplier is ~1/element-size
  (fp32 half-rate 4 B, bf16 full-rate 2 B, int8 double-rate 1 B), so `tile²·scale·elem` is
  invariant. **The GEMV balance point is a property of the array, not of the precision** — one
  number sizes it for every format.
- **This is where "16× more MACs bought zero" gets its scope.** That finding is a statement about
  **64–128 B/cycle**, not about arrays in general. At 128 B/cycle tile 8→32 is flat to within 2%
  (re-verified at the corrected 192 KB). At 256 B/cycle tile 8 is *already binding* (218.2 µs vs
  120.6 for tile 12). At 384 B/cycle tile 12 binds. **More MACs do eventually pay for GEMV work,
  and the crossover is computable.**
- CM-100's 12×12 therefore has **2.25× MAC headroom over the bandwidth it is fed**, which is the
  right amount: it is the headroom that makes 256 B/cycle a drop-in upgrade rather than a redesign.

**Energy return dies at 144 MACs**, confirming the earlier sizing at the corrected capacity:
marginal solver energy per mm² is 618 (tile 6), 191 (8), **56 (12)**, 19 (16), 6 (24), 2 (32).

**ML latency is the only thing that wants more, and it stops at ~1024 MACs** because the vector
unit takes over as the ML bottleneck at tile 32. The 30 fps int8 CNN envelope is **~4 GMAC/frame
at CM-100, ~15 at tile 32**; a 17 GMAC/frame model misses at every array size in the sweep.

### 8.2 Scratchpad bandwidth — the knob that works, and it works to 256 B/cycle

Balanced packages (lanes = bw/8, fp64 = lanes/2, ≤16 banks with ≥64-bit words), QUAD-MPC at its
**measured** worst-case tail:

| B/cycle | banks × word | lanes | core mm² | pJ/B | QUAD wc µs | Σ solver µJ | ×/step | **mm² per ×** |
|---|---|---|---|---|---|---|---|---|
| 16 | 2 × 8 B | 4 | 0.611 | 1.761 | 1712.0 | 88.45 | — | — |
| 32 | 4 × 8 B | 4 | 0.627 | 1.319 | 931.6 | 74.24 | 1.84× | 0.020 |
| 64 | 8 × 8 B | 8 | 0.668 | 1.033 | 467.3 | 54.24 | 1.99× | 0.041 |
| **128** | **16 × 8 B** | **16** | **0.786** | **0.839** | **235.2** | **43.15** | 1.99× | 0.120 |
| 256 | 16 × 16 B | 32 | 0.947 | 0.747 | 120.6 | 37.82 | 1.95× | 0.169 |
| 512 | 16 × 32 B | 64 | 1.268 | 0.647 | 101.3 | 33.99 | **1.19×** | **1.682** |

Read the last column — marginal area per unit of speedup, so smaller is better and it is
comparable step to step. **16 → 256 B/cycle returns a near-perfect 1.84–1.99× per doubling.
256 → 512 returns 1.19× at ten times the marginal cost**, and the reason is precisely §8.1: the
fp32 GEMV ceiling (bw/4) crosses the 144-MAC array rate (72 MAC/cycle) at 288 B/cycle. Past
there, bandwidth alone buys nothing — you must buy array *and* lanes *and* banks together, which
is a different part rather than a wider one.

**512 B/cycle is rejected on integrability, not on model output**: 16 banks × 32 B is a
**5530-bit read bus** including address/control/ECC. No soft-IP licensee will floorplan that;
it is a hard macro in disguise.

**Lanes must scale in step, but less than expected — and for one row only.** Minimum lanes at
which the vector unit stops binding: QUAD-MPC, ARM-MPC, EKF-24 and EKF-96 need only **4** lanes up
to 256 B/cycle (they are that GEMV-dominated). **FCS-MPC needs 16 at every bandwidth in the
sweep** — it is the one solver row where the vector unit binds first. So **lanes are provisioned
for FCS-MPC and for ML, not for the solver rows**, which is the same fact the accumulator-fusion
finding rests on. Raising bandwidth while holding 16 lanes costs only 1.14–1.25×.

### 8.3 Capacity — NOT a scaling axis. A step function with a monotone cost

**Solver energy is monotone increasing in capacity across the whole 32 KB → 1024 KB range**
(37.90 → 60.47 µJ). There is no capacity at which the part gets cheaper to run: bigger banks
raise the cost of every access and the byte count a workload fetches does not change. Pinned by
a test.

The decisive case, and it confirms the earlier ARM-MPC observation by a second route:

- ARM-MPC at 192 KB: 323.2 KB footprint **overflows**, streams 135 KB, **25.23 µJ**
- ARM-MPC at 384 KB: **fits**, streams 3 KB, **26.34 µJ**
- → making it fit costs **+1.11 µJ (1.044×) and +0.228 mm²**, for **zero** latency benefit
  (250 Hz, 5.5× worst-case margin). **The pJ/byte penalty on every access exceeds the DRAM
  traffic it saves.**

**ARM-MPC overflowing CM-100 is deliberate, not a gap.** Determinism is load-bearing only on the
fast loops; a 250 Hz interior-point solve streaming 0.034 GB/s is allowed to touch DRAM, and we
should say so rather than size a scratchpad to hide it.

> **RULE: size capacity to the fastest determinism-critical resident working set and to nothing
> else. A slow loop that overflows should overflow.** Capacity belongs in the SKU table as a
> discrete choice, never in a scaling roadmap.

### 8.4 Frequency — three walls, and the arrow points DOWN

| MHz | Vdd | rows_max | folds | spad mm² | core mm² | dynE | leakE | drain | QUAD wc µs | Σ solver µJ |
|---|---|---|---|---|---|---|---|---|---|---|
| 400 | 0.65 | 512 | 1 | 0.246 | 0.748 | 0.660 | 0.536 | 6 | 462.2 | **31.73** |
| 600 | 0.72 | 341 | 1 | 0.279 | 0.786 | 0.810 | 0.729 | 7 | 310.9 | 36.36 |
| **800** | **0.80** | **256** | **1** | **0.279** | **0.786** | **1.000** | **1.000** | **8** | **235.2** | **43.15** |
| 1200 | 0.92 | 170 | **2** | 0.407 | 0.933 | 1.322 | 1.521 | 11 | 160.8 | 55.65 |
| 1600 | 1.02 | 128 | 2 | 0.407 | 0.933 | 1.626 | 2.073 | 14 | 123.6 | 66.73 |

1. **Bitline length.** A 12 KiB × 64b bank needs a second vertical fold at ~1100 MHz: scratchpad
   **0.279 → 0.407 mm² (+46%)**.
2. **Supply.** 800 → 1600 MHz gives QUAD-MPC **1.90×** (not 2.00×) at **1.547× the solver
   energy** and 1.19× the core area. **The frequency axis buys latency we do not need at the cost
   of the energy advantage that IS the product.**
3. **Pipeline depth**, and this one is a coupling nobody had priced. A deeper datapath raises
   `drain_cycles`, and 97–100% of the sequencer's latency cost is dependency drain. QUAD-MPC worst
   case over a drain 4 → 32 sweep: **1.23× with chain blocking at the datapath width, 3.45×
   without it** (338.8 → 1167.6 µs). **Raising the clock and skipping chain blocking are not
   independent decisions** — chain blocking is what makes a deeper pipeline survivable, and it was
   already an RTL requirement on a thinner margin.

> #### **The uncomfortable finding: the recommended clock may be too HIGH**
>
> Every solver deadline has 4–15× worst-case margin, so the interesting question is not how fast
> the part can go but how slowly it can go and still make the deadline — because **energy falls
> with frequency**. Against a 50%-of-period budget at worst-case iterations:
>
> | MHz | Σ solver µJ | vs 800 | verdict @50% period |
> |---|---|---|---|
> | 300 | 34.16 | 0.79× | **MISSES** (QUAD-MPC cold 611 µs > 500 µs) |
> | 400 | 31.73 | **0.74×** | all OK (QUAD-MPC 462 µs, 1.08× margin) |
> | 600 | 36.36 | 0.84× | all OK (QUAD-MPC 311 µs, 1.61× margin) |
> | 800 | 43.15 | 1.00× | all OK |
>
> **600 MHz meets every frozen worst case inside a 50% budget at 0.84× the solver energy of
> 800 MHz, at identical area.** 800 MHz was chosen before the frequency axis had an energy cost in
> the model, i.e. when lowering it appeared to buy nothing.
>
> **DO NOT re-spec on this alone.** The 400 MHz point has only 1.08× margin on the one row whose
> tail is *measured*, the f–V table is [EST] with no PDK behind it, and the non-monotonicity below
> 400 MHz is a **model artefact** (`vdd_for_freq` clamps at 0.65 V and refuses to extrapolate into
> near-threshold). What it *does* establish: **the frequency parameter must be presented to a
> licensee as a downward energy knob with a deadline floor, never as an upward performance knob.**
> Firm it up with a real library f–V curve in Phase 1 — one of the cheapest measurements
> outstanding, and it moves the headline energy number.

**Process node is not a competitive axis at all.** Every perf/W figure in this document is a
**ratio** against a CPU baseline. Moving 16nm → 5nm moves the CPU too. Node buys absolute power
and area for a licensee's SoC budget; it does not move a single claim here. **Node is a cost axis.
Do not put it on a roadmap slide as though it improved the argument.**

### 8.5 Sequencer — fixed-size by nature; the only lever points down

- **Issue width buys literally nothing**, from 0.5× to 4×. Even *halving* it is free. Do not build
  a multi-ported descriptor queue. (Pinned by a test.)
- **Setup cycles per descriptor are free up to 16**; the knee is ~17–20, and even the naive
  **23-cycle RV32 emit costs 4% on QUAD-MPC and 5% on ARM-MPC**. **Consequence for the roadmap:
  the hardware descriptor generator is an area-cheap insurance policy, not a deadline requirement
  for the frozen set.** It stays in (0.028 mm², already budgeted) but must not be sold as
  load-bearing — the *arm-templates-once-per-solve* codegen fix is what removes the exposure, and
  that is free.
- **Queue depth: only 0 → 1 moves anything** in this aggregate model (246.7 → 235.2 µs, 4.9%), and
  **that DISAGREES with the settled finding that depth 8 is a measured floor.**
  **The measurement wins and the reason is structural:** the aggregate model averages descriptor
  cost over a block, while the measured DAG contains a **run of six consecutive descriptors
  averaging 5.8 cycles** that a queue must absorb *instantaneously*. An average cannot see a
  burst. **Depth 8 stays an RTL requirement; this sweep is not evidence against it, it is evidence
  that this sweep cannot resolve it.**
- **Drain cycles are the one sequencer number with leverage, and it scales the wrong way**:
  ARM-MPC 652 → 1037 µs over drain 2 → 32. You cannot buy performance with it, only lose it.

**The sequencer is therefore identical in every SKU.** Nothing about it is a per-SKU parameter;
chain-block width follows the datapath width automatically.

### 8.6 Scaling down — the customer rule, derived

With `k = cpu_time / cm_compute_time` the datapath ratio:

> **speedup ≥ 1 ⟺ cpu_time ≥ overhead · k/(k−1)**, and it is **unreachable when k ≤ 1**.

The floor is **a ratio between the launch protocol and the datapath advantage** — not a MAC
count, not a problem dimension. Measured k band across the frozen solver set: **1.14–2.17**. So
at T1 (0.80 µs) the floor is **1.5 µs of host time at k = 2.17 and 4.1 µs at k = 1.24**; at T1M
(0.30 µs), **0.56 µs and 1.6 µs**.

Verified against a sweep that straddles the crossover (EKF n=6 → 96, ADMM n_u = 1 → 8, FCS
2-level Np=1 → 3-level Np=2): EKF n=6 is **0.10×** with k = 0.92 — the model correctly reports
the floor as **infinite**, i.e. no launch protocol rescues it. EKF n=18 is MCU-class-only;
EKF n=24 wins at 1.21×.

**The energy claim has its own, lower floor, and its mechanism is different.** The CM pays static
power across the whole latency window including the launch, during which it computes nothing:
15 mW × 0.80 µs = **12.0 nJ** (T1), 4.5 nJ (T1M). An A55-class core at 0.15 W burns 12 nJ in
0.08 µs of compute, so a 5× advantage needs **≳0.4 µs of A55 compute at T1, ≳0.15 µs at T1M**.
Consistency check against the sweep: FCS 2-level Np=1 has 0.063 µs of host work and scores
**3.2× perf/W — below the 5× kill criterion**; FCS 3-level Np=1 has 0.102 µs and scores 7.0×.
**"We are always more efficient" is false**, and this is where it stops being true. Rounded for a
customer: **0.4 µs / 0.15 µs.**

### 8.7 External bandwidth — large-model ML is outside the *differentiated* envelope

| DMA GB/s | lat ms | DMA-bound % | perf/W | verdict |
|---|---|---|---|---|
| 6.4 | 174.1 | 93.5% | 2.35× | MISSES (deadline 50 ms) |
| 12.8 | 88.2 | 87.2% | 2.53× | MISSES |
| **25.6** | **45.2** | **75.1%** | **2.63×** | **MEETS** |
| 102.4 | 13.0 | 13.4% | 2.71× | MEETS |

"Misses at 6.4, meets at 25.6 GB/s" is a **provisioning requirement to state**. But read the
perf/W column, not the verdict column: at the bandwidth where it becomes feasible it is **75%
DRAM-bound**, and a CPU with the same 25.6 GB/s hits the same wall. **We are not differentiated on
a workload whose binding resource is off-chip.** A test pins that SLM-DECODE is invariant to
array, bandwidth, capacity *and* lanes — if it ever responds, this statement is wrong.

**The envelope, at a 50%-of-period transfer budget:** weight bytes/token = parameters (int8), so
at CM-100's assumed **6.4 GB/s and 20 tok/s the envelope is ~0.16 G int8 parameters** — a
sub-Phi-class model. 1.28 G needs 25.6 GB/s at 10 tok/s. **Above ~0.16 G, sell the CNN/ViT row,
not the LLM row.** VIT-TINY needs 0.17 GB/s and scores 28.2× perf/W; that is the ML story.

### 8.8 Multi-CM — a distraction, and the prior was right for better reasons

**Q1 — intra-solve splitting is dead in principle *and* in numbers.** The iteration is
loop-carried, but there *is* work-level parallelism inside one iteration: row-split every GEMV.
The cost is a **barrier at every chain step**, because the next step's operand is the
concatenation of this step's partials. **Measured** chain depth of one checked ADMM iteration is
**17 steps** under wait-mask semantics (32 under v0 one-token semantics). Charged generously
(perfect load balance, zero broadcast bytes, no row-quantisation waste), 2 CMs on QUAD-MPC:

| barrier model | µs/iter of sync | QUAD wc µs | vs 1 CM |
|---|---|---|---|
| 1 CM | — | 235.2 | 1.00× |
| shared scratchpad (16 cyc) | 0.340 | 142.8 | 1.65× |
| tightly-coupled mailbox (120 cyc) | 2.550 | 306.3 | **0.77×** |
| host-coherent fabric (400 cyc) | 8.500 | 746.6 | **0.32×** |

**And the one row that wins is unphysical, which is the real answer.** The `shared scratchpad`
row charges each CM 1/n of the datapath time *while sharing one scratchpad*. QUAD-MPC's binding
resource is scratchpad read bandwidth. **Two datapaths behind one 128 B/cycle scratchpad on a
bandwidth-bound row deliver 1.00×, not 1.65× — the second datapath starves.** The row is only
reachable if bandwidth also doubles, at which point bandwidth bought the speedup and the
replication is incidental. **On the workload class that defines this architecture there is no
mechanism by which intra-solve multi-CM helps.**

**Q2 — "it is one wider CM", priced.** 2 × CM-100 = **1.572 mm²** for 2 × 128 B/cycle over two
192 KB scratchpads; 1 CM at 256 B/cycle = **0.947 mm²**, QUAD-MPC 120.6 µs vs 142.8 µs
best-case-barrier. **The single wide CM is 1.66× cheaper and 1.18× faster.** Marginal cost of
bandwidth: **+0.161 mm² for +128 B/cycle inside one CM vs +0.786 mm² for a second CM — 4.9× more
area per byte/cycle via replication.**

**Q3 — FCS-MPC's 729 independent candidates: this is the one case where the answer is not an
immediate no, and it should be said out loud.** Zero dependency chains, so a split needs exactly
*one* barrier (the cross-CM argmin). Two CMs give **1.95× datapath and reach ~1.22× vs the
fastest CPU** — it genuinely converts the 0.62× row into parity. Rejected anyway, on three
grounds that must be given together:

1. **No deadline needs it.** FCS-MPC already meets its 25 µs period at **15.5×** margin; the
   socket is sold on 22.0× perf/W and core offload.
2. **It is the wrong bottleneck.** FCS-MPC is vector-bound, not bandwidth-bound. Ranked by area
   per unit of speedup: **MACIF `broadcast_init` + `square_accumulate` (525 → 154 vector cycles,
   0.62× → 0.88×, near-zero area) < 16 → 32 lanes (+0.143 mm², 1.26×) < a second CM (+0.786 mm²,
   ~1.95×)**. The second CM is last of three.
3. **It doubles static power on the row whose entire pitch is energy.**

Note the `shared scratchpad` row is admissible here and not in Q1 — because FCS-MPC is
vector-bound, so the replicated resource really is the scarce one. **Stating why the same row is
valid on one benchmark and invalid on another is the whole point of tracking the binding
constraint per row.**

**Q4 — many independent loops. Consolidation is capacity-bound, not throughput-bound.**

| row | duty % (wc) | footprint KB | loops by duty | loops by capacity | binding |
|---|---|---|---|---|---|
| QUAD-MPC | 23.52 | 138.8 | 4.3 | **1.38** | capacity |
| ARM-MPC | 18.26 | 323.2 | 5.5 | **0.59** | capacity |
| PWR-MPC | 6.45 | 0.7 | 15.5 | 274.3 | duty |
| EKF-24 | 0.26 | 7.9 | 388.3 | **24.3** | capacity |
| EKF-96 | 1.31 | 117.0 | 76.4 | **1.64** | capacity |

**Capacity binds 4 of 5 rows, by 3.1–47×.** One CM-100 hosts **one** flagship MPC loop, not four.
**Provisioning a shared accelerator from duty cycle alone would oversell consolidation by up to
47×** — worth remembering, because duty cycle is the number an integrator will reach for.

**And the stronger argument is jitter.** A drive node running FCS-MPC at 40 kHz (25 µs period)
alongside QUAD-MPC at 1 kHz: a QUAD-MPC worst-case solve is 235.2 µs = **9.4 periods** of the fast
loop. **Non-preemptible at solve granularity, the fast loop misses 10 deadlines — fatal.
Preemptible at descriptor granularity, worst-case blocking is the longest single descriptor
(~1600 cycles = 2.0 µs = 8.0% of the period) — tolerable.**
→ **Descriptor-granularity preemption, not a second CM, is what makes mixed-rate consolidation
safe.** It is a sequencer/scoreboard feature (save wait-mask and token state, admit the
higher-priority template) and it is nearly free next to a 0.786 mm² replica. **New architectural
requirement, and the only one this section adds.**

> **VERDICT: what survives is INDEPENDENT INSTANCES, NO INTERCONNECT** — for licensees wanting
> hard physical isolation between safety domains (ASIL) or control rates. That is **already
> supported**; it is instantiating the IP twice. It needs exactly one thing from us: **the runtime
> API must address multiple devices.** Nothing in RTL, no shared scratchpad, no CM-to-CM
> coherency, no one-sequencer-many-datapaths.
>
> **"Multi-core scaling" should come OFF the Phase 4 list**, because keeping it there implies a
> verification obligation we have no reason to accept.
>
> **What would change this:** a row that (a) needs >288 B/cycle of fp32 GEMV to make a deadline,
> or (b) needs >384 KB resident at >1 kHz. Neither exists today.

### 8.9 The verification tension, and the SKU family

§5 and §9 are both right; the resolution is that **parameters are not all the same kind of
parameter.**

**REPLICATIVE** — changes a width or depth, reuses the same control FSMs. Verification is a re-run
of an existing regression at a new elaboration. *Sell these as parameters.*
`tile`/`tile_n` (validated 8–16, square only) · `vector_lanes` (8–32) · `scratchpad_kb` (bank
depth; re-run the memory compiler) · bank word bytes (≥64 bits by design rule, 8–16 B) ·
frequency target (constraint-level, not RTL).

**STRUCTURAL** — changes control logic, protocol behaviour, or a declared contract. Each value is
a separate verification target with its own coverage closure. *Ration these; sell them as a SKU
list.*
`n_banks` (arbiter, address interleave, conflict resolution — **validate 8 and 16 only**) ·
precision set (MACIF declaration **plus the runtime's refusal paths**; each removal is a new
negative-test suite, not a smaller design) · fp64 present/absent (changes the *product*) ·
`seq_queue_depth` (scoreboard + token-reuse hazard rule — **fix at 8**) · chain-block width
(follows datapath width; not a knob) · pipeline depth (changes scoreboard hazard distances — fix
per SKU, do not expose).

> **`bus type` is the most expensive parameter in §5 and the most casually marketed.** AXI4,
> ACE-Lite and TileLink each need their own protocol-checker verification and their own
> coherency-behaviour proof. **It is not a parameter, it is three deliverables.** Never print it
> in a parameter list again.

#### The family: two SKUs

| SKU | MACs | B/cyc | banks | KB | lanes | fp64 | core mm² | pJ/B | Σ solver µJ | perf/W | residency violations |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **CM-100** | 144 | 128 | 16 | 192 | 16 | 8 | **0.786** | 0.839 | 43.15 | 12.3–22.0× | ARM-MPC *(deliberate)* |
| **CM-50** | 64 | 64 | 8 | 64 | 8 | 4 | **0.424** | 0.683 | 53.23 | 11.9–19.6× | QUAD-MPC, ARM-MPC, EKF-96 |
| ~~CM-100S~~ | 16 | 128 | 16 | 192 | 16 | 8 | 0.648 | 0.839 | 67.05 | 7.9–17.9× | ARM-MPC |
| ~~CM-200~~ | 256 | 256 | 16 | 256 | 32 | 16 | 1.113 | 0.816 | 36.48 | 13.1–25.3× | ARM-MPC |

**Read the residency column, not a deadline column.** All four "meet all deadlines" — latency
margin is 4–15× everywhere, so that column separates nothing. **What separates the SKUs is which
working sets stay resident**, because a residency violation puts DRAM in the control loop and
forfeits the determinism claim, which is the thing we sell. An overflowing row still *meets* its
deadline here (streaming is small and double-buffered); it just stops being deterministic.

- **CM-100** — robotics/drone node: condensed-QP MPC + state estimation + CNN/ViT-class ML +
  interior-point NMPC. Holds QUAD-MPC, PWR-MPC, EKF-24, EKF-96 resident. Tightest worst case
  QUAD-MPC at 4.25× (2.13× against a 50%-of-period budget).
- **CM-50** — MCU-class / motor-drive node. **0.424 mm², 54% of the flagship.**
  **Datasheet scoping line: CM-50 does not support the QUAD-MPC class.** Its 64 KB holds 0.46 of
  that 138.8 KB working set, so a 12-state N=20 condensed QP would stream ~75 KB every
  millisecond tick — still meets the deadline, no longer deterministic. Honest envelope:
  **n_var ≤ ~40 (ADMM), n ≤ ~48 (EKF), and finite-control-set MPC of any size** (FCS-MPC's whole
  working set is 0.7 KB). Keeps fp64 — a solver SKU cannot drop it.

**Rejected, with reasons:**
- **CM-100S (16-MAC solver-only)** saves only **0.138 mm² (18%)** — the array is 17% of the core,
  the cheapest thing in the part — and costs **2.68× the EKF-96 energy**. EKF is a *beachhead*
  workload, so a SKU that is bad at state estimation is a SKU for a socket we are not selling
  into. **Tile stays a replicative synthesis parameter over 8–16, not a SKU.**
- **CM-200** is a real Pareto point (1.95× QUAD-MPC worst case, 1.18× solver energy for 1.42× the
  core) with **no workload in the frozen set** against deadlines that already have 4–14× margin.
  Keep it as a documented synthesis configuration; do not verify it, do not price it, do not put
  it on a roadmap.

#### fp64 up to `lanes/2` is FREE — and this has been treated as a cost all along

The merged multi-format cluster is sized by `max(fp64_fmas, lanes/2)`, so at 16 lanes an fp64
count of **1, 2, 4 or 8 all cost the same 0.786 mm²**. What you pay is the multi-format **merge
premium — +0.067 mm² at 16 lanes over an fp32-only unit of the same width — once, not per fp64
unit.** Interior-point capability is a ~9%-of-core decision taken once, not a scaling axis with a
running cost. This is also why CM-50 can keep fp64 at 4 units for +0.034 mm².

#### The one axis to grow on demand — and it is not the one Phase 4 lists

**The fp64 FMA count.** ARM-MPC is the only row bound by *arithmetic* rather than bandwidth, and
it scales near-linearly: 8 → 16 FMAs is **1.85×** (730.3 → 380.2 µs) for **+0.144 mm²**; 32 FMAs
reaches 235.0 µs. But ARM-MPC already has **5.5× deadline margin** and the energy return is
**~19 µJ/mm² against tile 8→12's 84** — right axis, **no customer yet**. Grow it if and only if a
licensee brings a longer horizon, a faster rate, or an NMPC needing more interior-point
iterations.

### 8.10 Customer-facing envelope rule — "is my workload in the band?"

Five tests, all must pass; each uses a number the customer already has.

1. **Residency.** Sum every matrix as a **dense rectangle** at your working precision, plus every
   working vector and temporary — **not** packed triangular, the AGUs cannot address it.
   **≤192 KB (CM-100) or ≤64 KB (CM-50).** Over that, DRAM enters your control loop and the
   determinism claim is void. A hard edge, not a gradient.
2. **Latency floor.** Time your solve on your host core. **≥5 µs** → the latency claim holds on a
   rich host (T1). **≥1.7 µs** → holds on an MCU-class doorbell (T1M). **<1.5 µs → use the CPU**;
   no datapath size fixes this, the launch protocol is the cost.
3. **Energy floor** (lower than the latency floor, so most workloads failing test 2 still pass
   this): **≥0.4 µs of an efficient in-order core → ≥5× perf/W** (≥0.15 µs at T1M). Below that the
   CM's static power during launch exceeds your entire compute energy. Requires power gating —
   mandatory anyway.
4. **Descriptor density.** Average **≥8 datapath cycles per descriptor**, no burst of more than 8
   consecutive short descriptors. In practice your inner-loop matrix/vector operations must
   average **≥~64 elements** at 128 B/cycle fp32. Frozen-set rows run 238 cycles/descriptor (30×
   the bar); **one-step FCS-MPC and 100 kHz two-level FCS-MPC fail it — tell those customers to
   use the MCU.**
5. **Off-chip traffic.** (bytes crossing the boundary per call) / deadline **≤6.4 GB/s** at
   baseline provisioning. If your answer is a weight-streaming model, this is the **only** test
   that matters and no on-chip parameter changes it.

> **And the reuse-depth diagnostic, which tells a customer which knob is theirs.** Take your
> dominant product: how many times is each fetched operand element consumed before it is
> discarded? Call it **R**.
> - **R = 1** — pure GEMV. Bandwidth-bound. **MAC count is irrelevant to you.**
> - **R < 8** — still bandwidth-bound *even with a batch dimension*. FCS-MPC has R = n_y = 2
>   across 729 candidates and is bandwidth-bound; the batch buys 1.7×, not 12×.
> - **R ≥ tile** — tile-limited GEMM. The only regime where a bigger array pays, and even then it
>   pays in **energy** first.
>
> **If R is set by a *problem* dimension rather than by the tile, you are operand-fetch-bound.**
> That is the general form of the batch-1 finding and the single most useful thing to tell a
> customer architect.

### 8.11 What would falsify this section

**An L0 operand register file.** It buys the array's reuse at O(T) area instead of O(T²), which
(a) drops the honest array size to 36–64 MACs and (b) **raises the bandwidth at which the array
starts to bind**, moving the 288 B/cycle wall in §8.1–8.2 outward. It is the highest-value model
extension outstanding and it moves two of the walls above. Secondary: a real library f–V curve
(§8.4) could move the recommended clock down and the headline energy number with it.

## Sources

Model: this repo. Teardown:
[Ethos-U85 product brief](https://armkeil.blob.core.windows.net/developer/Files/pdf/datasheets/arm-ethos-u85-product-brief.pdf),
[Ethos-U85 technical overview](https://developer.arm.com/documentation/102684/latest/),
[Ethos-N78 product brief](https://armkeil.blob.core.windows.net/developer/Files/pdf/ML%20on%20Arm/arm-ethos-n78-product-brief.pdf),
[ARC NPX6](https://mips.com/processor-solutions/arc-npx6/),
[Synopsys NPX6 announcement](https://news.synopsys.com/2022-04-19-Synopsys-Introduces-Industrys-Highest-Performance-Neural-Processor-IP),
[Ceva NeuPro-M](https://www.ceva-ip.com/product/ceva-neupro-m/),
[NeuPro-M precision table (Embedded Vision Summit 2022)](https://www.edge-ai-vision.com/wp-content/uploads/2022/05/E2W07_Siegel_CEVA_2022.pdf),
[Cadence Neo NPU](https://www.cadence.com/en_US/home/tools/silicon-solutions/ai-ip-platform/neo-npu.html),
[Tensilica NeuroEdge 130](https://www.cadence.com/en_US/home/tools/silicon-solutions/ai-ip-platform/tensilica-neuroedge-ai-co-processor.html),
[MediaTek NVE, ISSCC 2024](https://ieeexplore.ieee.org/document/10454482/).

§5.1 control-domain addendum — **AURIX TC4x PPU:**
[emmtrix wiki, PPU (secondary)](https://www.emmtrix.com/wiki/Infineon_AURIX_TC4x_Parallel_Processing_Unit_(PPU)) ·
[Infineon PPU product page (vendor)](https://www.infineon.com/product-information/new-ppu-simd-vector-dsp) ·
[AURIX TC4x product page (vendor)](https://www.infineon.com/products/microcontroller/32-bit-tricore/aurix-tc4x) ·
[MetaWare Toolkit for AURIX TC4x (Synopsys/MIPS)](https://mips.com/processor-solutions/sw-metaware/) ·
[ARC EV7x vision processors — vector FPU option, 512-bit vector DSP (vendor)](https://mips.com/processor-solutions/ev-processors/ev7x-vision-processors/) ·
[EV7xFS functional-safety variants (vendor)](https://www.synopsys.com/designware-ip/processor-solutions/arc-evfs.html) ·
[Embedded Coder PPU support (MathWorks)](https://www.mathworks.com/help/ecoder/infineonaurix/ug/soc-ifx-tc4x-ppu.html).
**TI C2000 CLA:**
[CLA software guide — overview (vendor)](https://software-dl.ti.com/C2000/docs/cla_software_dev_guide/intro.html) ·
[device-specific CLA features / Type 0-1-2 (vendor)](https://software-dl.ti.com/C2000/docs/cla_software_dev_guide/features.html) ·
[CLA FAQ — memory model, message RAM, C restrictions (vendor)](https://software-dl.ti.com/C2000/docs/cla_software_dev_guide/faq.html) ·
[TMS320F28379D — 200 MHz dual C28x + dual CLA (vendor)](https://www.ti.com/product/TMS320F28379D) ·
[LAUNCHXL-F28379D (vendor)](https://www.ti.com/tool/LAUNCHXL-F28379D) ·
[FCS-MPC on F28379D (peer-reviewed third party; non-comparable workload)](https://www.mdpi.com/2227-7390/11/21/4553).
**Arm SME2:**
[Hello SME microbenchmarks, Univ. Jena — measured M4 (third party)](https://scalable.uni-jena.de/opt/sme/micro.html) ·
[Hello SME paper, arXiv 2409.18779](https://arxiv.org/pdf/2409.18779) ·
[M4 SME exploration — cluster-shared SME block, measured (third party)](https://github.com/tzakharko/m4-sme-exploration/blob/main/reports/01-sme-overview.md) ·
[Arm SME introduction, part 1 (vendor)](https://developer.arm.com/community/arm-community-blogs/b/architectures-and-processors-blog/posts/arm-scalable-matrix-extension-introduction) ·
[Lumex platform / SME2-enabled C1 cores (secondary, reporting Arm claims)](https://www.cnx-software.com/2025/09/10/arm-introduces-lumex-platform-with-sme2-enabled-c1-cpu-for-edge-ai-mali-g1-gpu/) ·
[Arm C1-Ultra (vendor)](https://www.arm.com/products/silicon-ip-cpu/c1-ultra) ·
[Cortex-A320 — SVE2, SME2 not stated (vendor)](https://www.arm.com/products/silicon-ip-cpu/cortex-a/cortex-a320) ·
[Cortex-A720AE — SME2 not stated (vendor)](https://www.arm.com/products/silicon-ip-cpu/cortex-a/cortex-a720ae).
**RISC-V RVV:**
[SpacemiT K1 / X60 brief, RVV 1.0 VLEN 256 (BananaPi)](https://docs.banana-pi.org/en/BPI-F3/SpacemiT_K1) ·
[Phoronix RVV review — no FLOP/s, no power metrics exposed](https://www.phoronix.com/review/risc-v-rvv-vector-benchmarks) ·
[RISC-V TSC minutes 2026-02-04 — IME/VME fragmentation, matrix extensions behind schedule](https://riscv.atlassian.net/wiki/spaces/TSCX/pages/1100742657/2026-02-04+TSC+Meeting) ·
[Integrated Matrix Extension TG (RISC-V)](https://github.com/riscv-admin/integrated-matrix-extension) ·
[IME ratification plan](https://riscv.atlassian.net/wiki/spaces/IMEX/pages/598867969/IME+Ratification+Plan).
