# Argmind — project context

Fabless semiconductor IP startup, pre-revenue, Phase 0, solo founder.
Detail: [PLAN.md](PLAN.md) · [PHASE0_FINDINGS.md](PHASE0_FINDINGS.md) ·
[BENCHMARKS.md](BENCHMARKS.md) · [agent team](.claude/agents/README.md).

**Argmind CM**: licensable SystemVerilog IP in a category we coin, the **Computational Module
(CM)**; SKUs `Argmind CM-<n>`. One engine for iterative solvers (ADMM/OSQP, interior-point, SQP,
MPC, EKF) and edge ML matmul. Mechanisms: a sequencer coupled to the datapath so convergence
loops run in-fabric with no host round-trip, plus zero-copy host memory and call-to-call
residency. Open-core licensing; beachhead is embedded MPC/state estimation in robotics and
automotive.

**Know which differentiator applies to which rival** (PHASE0_FINDINGS §5/§5.1):
- vs **NPU IP** — sequencer, coherency and residency are all real, structural gaps.
- vs **CPUs with vector/matrix extensions** (Arm SME2, RISC-V RVV) — those three are *not*
  differentiators at all: being inside the CPU, they get control flow, coherent zero-copy and
  cache residency for free. Only three claims survive: **energy per solve, deterministic
  worst-case latency, and core offload** — plus the codegen/software ecosystem.
- vs **MCU-class parts** (TI C2000 CLA) — **residency and coherency are void** (no DRAM, no cache,
  nothing to eliminate) but **invocation overhead is not**, and that distinction was the whole
  tension. Settled 2026-07-30 (§0.3): the honest MCU-class launch costs are ~2.0 µs copy-based vs
  ~0.30 µs doorbell [EST] — a 6.7× / 1.7 µs advantage, not the rich-host 31× / 24 µs — because
  buffer-map and cache-maintenance are MMU/cache artefacts an all-SRAM part does not have.
  **Pitch for MCU class = low invocation overhead + determinism + energy per solve. Never
  zero-copy.** And the invocation argument only carries a *deadline* above ~100 kHz. Zero-copy
  remains real on rich hosts (DRAM + cache present).

## Settled — don't re-litigate silently (evidence in PHASE0_FINDINGS.md)

- Batch-1 solver loops (`P@x`, cached triangular solves, `A@x`, `Aᵀ@y`) are matrix-**vector**
  (GEMV) → no operand reuse → bandwidth-bound. 16× more MACs bought zero; 16→256 B/cycle bought ~9×.
- **The general form of that finding, which is stronger** (2026-07-30, §0.3): *a workload is
  operand-fetch-bound whenever its reuse depth is set by a **problem** dimension rather than by the
  tile* — batch-1 having reuse depth 1 is only the extreme case. FCS-MPC has a real batch dimension
  (729 candidate sequences) and is *still* bandwidth-bound, because its dominant product reuses each
  predicted state exactly **n_y = 2** times: 2.33 B/MAC against 0.67 for a tile-limited GEMM. The
  batch buys 1.7×, not the ~12× the array size suggests. **The model now applies a bandwidth ceiling
  to GEMM as well as GEMV; before this it was structurally unable to report a bandwidth-starved
  GEMM.** Corollary worth having: **144 MACs is also the knee for FCS-MPC** (array-bound below
  12×12, intensity-bound above) — the same number the EKF energy argument produced, independently.
- **Bandwidth-first provisioning**: wide banked scratchpad, modest MAC array — opposite of an
  NPU. **CM-100 baseline (costed 2026-07-30): 128 KB, 16 banks × 8 B = 128 B/cycle, 144 MACs
  (12×12), 8 merged FMA units (= 16 fp32 lanes or 8 fp64), 800 MHz → 0.698 mm² at 16nm**
  (modelled, ±2×, and ±2.1× again on the FMA gate count → 0.845 mm² at the pessimistic anchor).
  Minimum viable 64 B/cycle / 64 MACs / 0.563 mm²; a solver-only SKU at 64 MACs is real
  (+35% energy, all on EKF) but **cannot drop fp64**.
- **fp64 is a requirement and does not come from the array.** Interior-point KKT conditioning
  reaches ~1e12, so ARM-MPC (acados SQP-RTI + HPIPM) needs it. Multi-pass emulation from bf16
  needs **49–361 passes** — 3.5–23× worse energy to save ~4% of the core — and fp32 + iterative
  refinement only *relocates* the fp64 requirement to the residual. Answer: **merge fp64 into the
  vector unit** (8 multi-format FMAs, 1×fp64 or 2×fp32 each), +0.067 mm².
- **Precision floors, now numerically demonstrated** (`perf_model/precision_study.py`, measured
  numerics not silicon): **EKF needs fp32 — confirmed hard.** bf16 and fp16 lose
  positive-definiteness on the *first* step in all three filter forms, square-root included;
  fp16 overflows at κ(P)=1e6, so range binds as well as precision. **Interior-point fp64 is NOT
  measured** — κ growth (3e2→2e13 as μ→1e-12) reproduces, but fp32 error stays ~5e-6, so the
  requirement still rests on literature and the audit; do not cite it as measured. **ADMM
  "tolerates bf16" is too strong**: bf16 operands cost ~2% control-action error vs 4e-7 for fp32,
  though the directional claim holds — the triangular solve is ~3× more sensitive than operands.
  **For solver operands fp16 beats bf16 by 10×** (mantissa matters more than range — the opposite
  of the ML tradeoff), so if 16-bit operands are ever used to double GEMV bandwidth, use fp16.
- **fp16 operands are ADOPTED as an opt-in codegen mode, not as a default** (settled 2026-07-30,
  §0.3). Keeping the triangular **solve at fp32 costs nothing** — fp16/fp32-solve and fp16/fp64-solve
  score *identically* (1.9e-3), so the operand width, not the solve, limits the mixed scheme. Worth
  **1.16–1.49× latency and 1.51–1.79× energy** on the ADMM rows (QUAD-MPC perf/W 14.1× → 25.3×) at
  ~2400× the operand error (8e-7 → 1.9e-3, ~0.2% of a commanded control action). **Not 2×, because
  the vector unit becomes co-binding once GEMV halves** — a bandwidth-bound row stops being
  bandwidth-bound before the doubling is spent. **The array's bf16 spec is wrong for the solver
  path; bf16 stays correct for ML.**
- **FCS-MPC has a different precision floor from every other row, because its output is a discrete
  argmin.** fp16 changes the selected switching sequence on 3.0% of ticks at 1.5e-6 of the candidate
  cost spread; bf16 on 11.0% at 9.2e-5 (measured numerics, 729 sequences × 200 trials). A
  disagreement between two near-equal candidates is not an error. Precision is a property of the
  **algorithm**, and this is the third distinct floor in the set.
- **MACIF carries precision as a declared contract, not an implementation detail.** The boundary
  runs between the **bf16/int8 array (swappable, CIM-eligible)** and the **fp32/fp64 FMA cluster
  (not swappable)** — no in-memory-computing macro has a path to 53-bit significands. The runtime
  must *refuse* an fp64 kernel on a macro that does not declare fp64, at codegen time.
  **Two entries added 2026-07-30 (§0.3):** (1) **fp16 must be declared distinct from bf16** — a macro
  silently substituting bf16 costs 10× the control-action error, and the runtime must refuse it
  exactly as it refuses a missing fp64 path; (2) **accumulator capabilities are part of the
  contract** — broadcast/bias initialisation and square-accumulate. Those two features cut FCS-MPC's
  vector work 525 → 154 cycles at near-zero area (0.62× → 0.88× vs the fastest CPU), where the
  alternative — 16 → 32 vector lanes — costs **+0.143 mm², +20% of the core, for only 1.26×**. Same
  L0-staging logic already flagged as the top model extension: fold work into the accumulator at
  O(1) area instead of buying O(T²) of array.
- **Size against the worst-case iteration tail, never the typical one.** At p95 iterations every
  solver row meets its deadline at 16 B/cycle — a sweep run that way can only ratify a
  configuration, never justify one. The measured cold-solve tail (QUAD-MPC 74 iterations) inside
  a 50%-of-period budget is what sets the 64 B/cycle floor. **Since the PWR-MPC re-spec
  (2026-07-30) that floor rests on QUAD-MPC ALONE** — it does not move, because the surviving row is
  the one whose tail is *measured*, but a single-row justification is the exact weakness that caused
  two previous sizing re-runs.
- **The MAC array is *latency*-ML-justified but *energy*-solver-justified.** Nothing needs >36
  MACs to meet a deadline, but tile dimension **is operand-reuse depth** ((1/Tm + 1/Tn) operand
  elements per GEMM MAC), so 64→256 MACs cuts EKF-96 energy 1.59× and EKF-24 1.54× while doing
  exactly nothing for pure-GEMV ADMM. EKF is a beachhead workload, so "solver customers subsidise
  an ML feature" is *not* the honest story. Caveat: an L0 operand register file would buy the same
  traffic cut at O(T) area instead of O(T²) — if that lands, honest array size drops to 36–64.
- Advantage is **energy + determinism, not speed**: **14–24× perf/W at the CM-100 config
  (modelled, ±2×; 9–26× across the scratchpad-energy sensitivity band)**, latency **0.62–2.15× on
  solvers** — QUAD 1.74×, EKF-96 2.15×, ARM 1.14×, EKF-24 1.21×, and **PWR-MPC 0.62×, genuinely
  below parity**. **A CPU meets 5 of 5 solver deadlines in the frozen set** — never pitch "the CPU
  can't do this"; we are more efficient, not enabling.
  (History: 17–25×/1.2–2.9×, then 9.5–21.3×/"below parity on 2–3 of 5", then 14–22×/0.97–2.11×.
  Always quote at the CM-100 configuration.)
- **PWR-MPC is below parity and the reason is structural, not fixable by provisioning**
  (settled 2026-07-30, §0.3). Re-specified as finite-control-set MPC it is 0.62×, and the
  decomposition is **datapath-only 1.24× → 0.90× with a 0.30 µs MCU-class launch → 0.62× with T1's
  0.80 µs**. The CM *computes* the tick 1.24× faster than an SME2 core and then pays 0.8 µs to be
  asked. **For a sub-microsecond workload no invocation protocol involving a descriptor and a
  completion notification beats the host core on latency** — an integration-depth fact (wants D3),
  not a datapath one. Sell this socket on energy (23.5× perf/W) and offload. **Related: the old
  0.974× figure was partly a CPU-model artefact** — elementwise work was charged at SME2's
  outer-product rate; the ZA tile cannot execute min/max or horizontal reductions. Corrected, the
  retired ADMM row is 1.08× (1.72× at the measured M4 8× ratio). **Neither side of that is
  quotable; the row sits on the assumption.**
- **The sequencer is not the bottleneck, but two of its properties are requirements**
  (settled 2026-07-30, PHASE0_FINDINGS §0.2). The earlier "ARM-MPC is sequencer-issue-bound at any
  precision" note was a **unit error** — it divided an fp64 workload by the bf16 *array* rate.
  Corrected: 30 datapath cycles/descriptor vs a 34-cycle crossover, and exposure scales with
  arithmetic **rate**, so fp64 is the *safest* case and bf16/fp32 the most exposed. Sequencer =
  scalar core **+ hardware descriptor generator, ≤4 cycles/descriptor, 8-deep run-ahead queue**;
  0.028 mm², inside the existing budget, **no area change**. Issue *width* >1/cycle buys nothing.
  **DOWNGRADED 2026-07-30 by the PWR-MPC re-spec (§0.3): (a) run-ahead and (b) chain blocking ≥ one
  datapath width were each justified by a deadline miss on PWR-MPC; re-specified as FCS-MPC that row
  has ZERO dependency chains and is invariant to both. NO frozen row now misses from either alone.**
  What survives: **you may skip at most one of the two** — QUAD-MPC at its *measured* worst-case tail
  is 2.21× without run-ahead, 1.27× with scalar chain blocking, and **0.37× (misses) with both
  absent**. Chain blocking is worth 2.3× of that margin, the largest single lever in the analysis, so
  it stays an RTL requirement — on a thin measured margin rather than a miss. **97–100% of the
  sequencer's latency cost is dependency drain, not issue throughput** — relieved only by wider chain
  blocks and a shorter pipeline, never by bandwidth or MACs. Codegen rule: **≥8 datapath cycles per
  descriptor on average, bursts ≤ queue depth**; thinnest frozen row is now ARM-MPC at 3.78×, but
  **one-step FCS-MPC (0.61×) and the 100 kHz two-level variant (0.69×) FAIL it** — for those, tell the
  customer to use the MCU. Exposure is set by cycles/descriptor, **not** algorithm class (three
  algorithm classes now demonstrate it), so do not pitch this as "interior point is the risky one."
  Consequence for §4.2: **descriptor emission stays on the CM side at every integration depth.**
- **Descriptor format v0 frozen and the emit cost derived, not guessed**
  (`perf_model/seq_emit_cost.py`): **8 words / 32 B**, three base addresses, packed dims, a
  chain/wait token pair. RV32 emit counted at **23 cycles naive, 10 delta-encoded, ~0 with the
  hardware generator** — so `seq_setup_cycles=4` was never reachable in software and silently
  presumed the generator. Drop delta encoding. **New requirement the flat assumption hid: arming
  costs ~22 instructions *per template*, so a template must cover ≥6 descriptors.** An HPIPM stage
  sweep gets 0.04 cycles/desc; an **ADMM iteration gets 8.25 — above the bar** — because it issues
  ~6 distinct shapes over only 16–28 descriptors. Fix is free and codegen-side: **arm templates
  once per solve, not per iteration** (legal because allocation is static — the same property that
  makes overlap legal), which means **the generator must hold ≥6 armed templates.**
- **Power gating is an architectural requirement, not an optimization.** Solver duty cycles are
  0.2–12%; charged ungated, QUAD-MPC perf/W falls 12.6× → 4.8× — *below the kill criterion* —
  and EKF-24 to 0.2×. Coarse-grain gating, or at minimum full clock gating plus retention-mode
  scratchpad, is mandatory.
- Scratchpad: **bandwidth is cheap, bank count is what costs area.** Design rule: **bank word
  ≥ 64 bits** (bank count ≤ bytes-per-cycle ÷ 8) — below that the energy model has no published
  data and becomes unphysical. Never multi-port; bank instead (8T dual-port is 2.1× the area).
- Determinism is a feature: statically code-generated allocation, no runtime cache behavior in
  the solver path.
- **NPU IP** can't retarget in software — no arithmetic (Ethos-U85 is INT8-weights-only; Ceva
  FP32 ≈1% of INT8) and no control plane (all four majors compile control flow away offline).
  **Scope limit — control-domain incumbents are a different story** (PHASE0_FINDINGS §5.1).
  The AURIX TC4x **PPU** is a licensed Synopsys ARC EV7x: scalar core + SIMD vector unit +
  closely-coupled memory running its own instruction stream — a small version of our
  architecture, automotive-qualified today. If its optional FP path is instantiated it needs
  **a solver library, not new silicon**. Arm **SME2** is measured at 2008 GFLOP/s FP32 (Apple M4
  P-cluster, 3nm class) — no retargeting needed at all, only tuned libraries. Open falsifier F4;
  settle with an eval board, not opinion. Also: Infineon *licensing* the PPU rather than building
  it is our best evidence that MCU vendors will license outside compute IP — and makes Synopsys a
  competitor for the license, not just the socket.
- Never claim to beat in-memory computing on inference TOPS/W; our answer is the **MACIF**
  swappable-macro boundary.
- **Runtime API is the constitution** — identical across simulator, open core, and commercial
  core. Changing it is a business decision.

## Evidence standards (non-negotiable — one bad number costs the company)

- Efficiency figures always carry **process node + precision**.
- Label **measured silicon vs. vendor estimate**; benchmark only against measured.
- Model outputs are **±2× and never quotable externally** — say so in any doc containing them.
- **"Not published"** is a valid, preferred finding; never present an estimate as a
  competitor's spec.
- **Models must be able to fail**: a flattering result means find the assumption first. **Twelve
  modelling errors so far**, most of them flattering — undercounted overhead energy, GEMV modelled
  as GEMM, a phantom transcendental term that alone produced the entire apparent latency win,
  scratchpad energy 2.3× low, MAC count conflated with operand-reuse depth, an unphysical vector
  gate count. All are encoded as regression tests; run them.
- **Scrutinise corrections that favour us as hard as ones that don't.** The fp64 pass found a CPU
  baseline charging fp64 at fp32 rates — fixing it moved ARM-MPC perf/W 14.1× → 20.2×, entirely
  on the CPU side. Two errors had been partly cancelling.
- Cite URLs for external claims.

## Conventions

Windows/PowerShell; python 3.14; **git repo since 2026-07-30** — commit before and after any
change to `perf_model/`, and run `python -m pytest` (151 invariant + mutation tests) before
quoting any modelled number. Markdown docs at root, model in `perf_model/`.
Prefer open-source EDA (Verilator, cocotb, Yosys) — the cash plan depends on it. Interfaces:
AXI4/AXI4-Lite baseline; ACE-Lite (Arm) and coherent TileLink (RISC-V) for zero-copy; CHI +
SMMU/SVM deferred to Phase 4. Apple is not an integration target — they don't license
third-party accelerator IP; AMBA compliance is the only "Apple story."
