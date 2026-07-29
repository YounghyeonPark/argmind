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
  NPU. Array sized for ML plus real matrix-matrix work (EKF covariance, SQP re-factorization).
- Advantage is **energy + determinism, not speed**: **~10–20× perf/W (modelled, ±2×; band
  9.0–26.2× over the scratchpad-energy sensitivity)**, and latency is **~0.9–3.2×, i.e. at or
  below parity on the flagship benchmark**. A CPU already meets several deadlines — never pitch
  "the CPU can't do this." (Was quoted as 17–25× and 1.2–2.9×; both were corrected 2026-07-29
  after the scratchpad energy constant proved 2.3× too low and a phantom transcendental term was
  found in the CPU comparison.)
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
- **Models must be able to fail**: a flattering result means find the assumption first — needed
  twice already (undercounted overhead energy; GEMV modelled as GEMM).
- Cite URLs for external claims.

## Conventions

Windows/PowerShell; python 3.14; not a git repo. Markdown docs at root, model in `perf_model/`.
Prefer open-source EDA (Verilator, cocotb, Yosys) — the cash plan depends on it. Interfaces:
AXI4/AXI4-Lite baseline; ACE-Lite (Arm) and coherent TileLink (RISC-V) for zero-copy; CHI +
SMMU/SVM deferred to Phase 4. Apple is not an integration target — they don't license
third-party accelerator IP; AMBA compliance is the only "Apple story."
