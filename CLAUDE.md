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
- vs **MCU-class parts** (TI C2000 CLA) — zero-copy and residency are *void*: no DRAM, no cache,
  nothing to eliminate. Unresolved tension: §4 says zero-copy decides PWR-MPC deadline
  feasibility, and motor drive is the socket most likely to be MCU-class. Settle before pitching
  zero-copy there.

## Settled — don't re-litigate silently (evidence in PHASE0_FINDINGS.md)

- Batch-1 solver loops (`P@x`, cached triangular solves, `A@x`, `Aᵀ@y`) are matrix-**vector**
  (GEMV) → no operand reuse → bandwidth-bound. 16× more MACs bought zero; 16→256 B/cycle bought ~9×.
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
- **MACIF carries precision as a declared contract, not an implementation detail.** The boundary
  runs between the **bf16/int8 array (swappable, CIM-eligible)** and the **fp32/fp64 FMA cluster
  (not swappable)** — no in-memory-computing macro has a path to 53-bit significands. The runtime
  must *refuse* an fp64 kernel on a macro that does not declare fp64, at codegen time.
- **Size against the worst-case iteration tail, never the typical one.** At p95 iterations every
  solver row meets its deadline at 16 B/cycle — a sweep run that way can only ratify a
  configuration, never justify one. The measured cold-solve tail (QUAD-MPC 74 iterations) inside
  a 50%-of-period budget is what sets the 64 B/cycle floor.
- **The MAC array is *latency*-ML-justified but *energy*-solver-justified.** Nothing needs >36
  MACs to meet a deadline, but tile dimension **is operand-reuse depth** ((1/Tm + 1/Tn) operand
  elements per GEMM MAC), so 64→256 MACs cuts EKF-96 energy 1.59× and EKF-24 1.54× while doing
  exactly nothing for pure-GEMV ADMM. EKF is a beachhead workload, so "solver customers subsidise
  an ML feature" is *not* the honest story. Caveat: an L0 operand register file would buy the same
  traffic cut at O(T) area instead of O(T²) — if that lands, honest array size drops to 36–64.
- Advantage is **energy + determinism, not speed**: **15–22× perf/W at the CM-100 config
  (modelled, ±2×; 9–26× across the scratchpad-energy sensitivity band)**, latency **1.0–2.1× on
  solvers — parity at worst, nothing below it**. **A CPU meets 5 of 5 solver deadlines in the
  frozen set** — never pitch "the CPU can't do this"; we are more efficient, not enabling.
  (History: quoted as 17–25× and 1.2–2.9×, then 9.5–21.3× and "below parity on 2–3 of 5". The
  latter was an artefact of reporting at 64 B/cycle, a config we do not recommend. Always quote
  at the CM-100 configuration.)
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
change to `perf_model/`, and run `python -m pytest` (70 invariant + mutation tests) before
quoting any modelled number. Markdown docs at root, model in `perf_model/`.
Prefer open-source EDA (Verilator, cocotb, Yosys) — the cash plan depends on it. Interfaces:
AXI4/AXI4-Lite baseline; ACE-Lite (Arm) and coherent TileLink (RISC-V) for zero-copy; CHI +
SMMU/SVM deferred to Phase 4. Apple is not an integration target — they don't license
third-party accelerator IP; AMBA compliance is the only "Apple story."
