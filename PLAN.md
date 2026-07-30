# Argmind — Fabless Computational Module (CM) IP — Project Plan

**Product category (coined):** Computational Module (CM) — a licensable IP class that runs
both iterative optimization algorithms and matrix-multiplication workloads on one engine.
Category positioning: "the industry's first Computational Module."
**Brand name:** **Argmind** (decided 2026-07-29) — argmin + *mind*: "the mind that
minimizes." Keeps the math meaning for the optimization audience, adds the AI connotation.
Pending: USPTO/EUIPO Nice class 9/42 clearance + domains (argmind.io, argmind.ai).
**Product naming structure:** `Argmind CM-<n>` (e.g., "Argmind CM-100"), so the category
label is embedded in every SKU.
**Date:** 2026-07-28 (naming decided 2026-07-29)
**Status:** Draft v0.3

---

## 0. Brand name — decision record

**Decision (2026-07-29): Argmind.** Shortlist and twists below kept for the record.

| Candidate | Rationale | Collision check (2026-07) |
|---|---|---|
| **Argmin** (recommended) | The math operator for "the argument that minimizes" — literally what the core computes. Instantly resonates with the control/optimization audience; short; distinctive. | No semiconductor company found. argmin.net is a personal ML blog (not a trademark conflict in IC class). Verify USPTO/EUIPO Class 9/42 before filing. |
| **Krylov** | Evokes Krylov-subspace iterative solvers; scientist-name precedent (Tesla, Ampere). | No tech-company collision found, but search footprint skews toward Russian chip ventures — mild geopolitical branding baggage. |
| **Minima** | Minimization; clean sound. | Collides with Minima (blockchain company); crowded. |
| **Iterion** | Iteration + semiconductor-sounding suffix. | Taken: Iterion Therapeutics (US), Iterion Limited (UK IT consultancy). Different classes, but crowded — weak pick. |

Twists on "Argmin" (all clean of company collisions as of 2026-07):

| Twist | Rationale |
|---|---|
| **Argmind** (leading twist) | argmin + *mind* — "the mind that minimizes"; keeps the math pun (near-identical pronunciation) and adds an AI/intelligence connotation for the ML side. |
| **Argmini** | argmin + *mini* — edge/small-footprint wink; playful ("Argmini CM-100"); may read too cute for automotive. |
| **Argminium** | *-ium* element suffix (silicon, lithium) — "material of computation" vibe; longer but very ownable. |
| **Minarg** | Syllable reversal — maximally distinctive proper-name sound, but loses instant math recognition. |
| Stylizations | Arqmin / ARGMN / logo rendered as the expression "arg min ƒ" — visual-identity twists composable with any name above. |

Action before committing: USPTO TESS + EUIPO search in Nice classes 9 (ICs/IP cores) and 42
(design services); domain check (argmin.io / argmin-semi.com etc.).

---

## 1. Vision

Design and license a **soft IP core** (synthesizable RTL) that accelerates two workload classes
that today fall between the cracks of CPUs, GPUs, and NPUs:

1. **Sequential / iterative algorithms** — convex optimization loops (ADMM, interior-point,
   proximal gradient), Kalman filters, model-predictive control (MPC), iterative linear solvers
   (CG, GMRES), dynamic programming. These are dominated by **loop-carried dependencies and
   small-to-medium matrix ops**, where GPU/NPU launch latency and batch-oriented design waste
   most of their throughput.
2. **Matrix-multiplication-dominated workloads** — ML inference, transformer/LLM blocks
   (GEMM, attention), classical DSP. These need raw MAC throughput with flexible precision.

**Core thesis:** one architecture — a dense matmul engine **tightly coupled** to a programmable
scalar sequencer — can serve both, because iterative solvers are mostly small GEMMs plus
scalar control (convergence checks, step-size updates) that must run with near-zero latency
between iterations. Incumbent NPUs optimize batched throughput; Argmind optimizes
**latency-per-iteration** while still delivering competitive GEMM throughput.

**Delivery model:** fabless, IP-licensing only (no chip productization initially). Customers
integrate Argmind into their own SoCs. Optional test chip later purely for PPA credibility.

---

## 2. Target markets & use cases

| Segment | Workload | Why Argmind wins |
|---|---|---|
| Robotics / drones | MPC @ 1–10 kHz control loops, state estimation | Iteration latency, determinism, low power |
| Automotive | Trajectory optimization, sensor fusion, embedded ML | Latency + functional-safety-friendly determinism |
| Industrial / power electronics | Real-time optimal control, grid optimization | Hard real-time solve deadlines |
| Telecom / SDR | Beamforming, equalization, iterative decoding | Small-matrix ops at line rate |
| Edge AI | On-device LLM/SLM inference, vision transformers | Shared GEMM engine, quantized inference |
| Finance / embedded quant | Portfolio/risk optimization on the edge | Niche but high value |

Beachhead recommendation: **embedded MPC + state estimation (robotics/automotive)** — it is the
segment where the sequential-algorithm advantage is largest and where incumbent NPU IP
(Arm Ethos, Ceva, Cadence, Synopsys) has no story. Edge-LLM capability rides along as the
second selling point on the same engine.

---

## 3. Product definition

### 3.1 Hardware architecture (concept)

```
            AXI4/AXI5 (or CHI/ACE-Lite, TileLink)          
   ┌─────────────────┬──────────────────────────┬─────────┐
   │  DMA engine     │   Control/CSR (APB/AXI-L)│  IRQ    │
   ├─────────────────┴──────────────────────────┴─────────┤
   │            Multi-bank local SRAM scratchpad          │
   ├───────────────┬───────────────────┬───────────────── ┤
   │  GEMM tile    │  Vector unit      │  Sequencer core  │
   │  engine       │  (elementwise,    │  (RISC-V RV32/64 │
   │  (systolic /  │  reductions,      │  + custom ISA    │
   │  output-      │  activations,     │  extensions for  │
   │  stationary)  │  norms, sqrt/div) │  loop control)   │
   └───────────────┴───────────────────┴──────────────────┘
```

> **REVISED 2026-07-29 by Phase-0 modelling — see [PHASE0_FINDINGS.md](PHASE0_FINDINGS.md) §2.**
> Solver inner loops are matrix-**vector** at batch 1, so they are bounded by scratchpad
> bandwidth, not MAC count: a 16× larger array bought *zero* latency improvement on QUAD-MPC,
> while 16→256 B/cycle scratchpad bandwidth bought 9×. **Provision bandwidth-first: a wide,
> heavily-banked scratchpad with a modest MAC array — the opposite of NPU provisioning.**
> The MAC array is sized for the ML path and for matrix-matrix solver work (EKF covariance,
> SQP re-factorization), not for the ADMM inner loop.

- **GEMM engine:** configurable tile array (CM-100 baseline 12×12 = 144 MACs), **bf16/fp16
  multiply with fp32 accumulate**; INT8 double-pumped, **fp32 by 2-pass operand split** (products
  exact in the fp32 accumulator). Must be efficient at **small matrices** (n = 4–128), not just
  large tiles — short pipeline fill/drain, no fixed large-tile penalty. **The array does NOT do
  fp64** — see below.
- **Merged FMA cluster (vector unit + fp64 datapath, one block):** 8 multi-format IEEE-754 FMA
  units, each **1 × fp64 *or* 2 × fp32**, giving 16 fp32 lanes or 8 fp64 FMAs. Handles elementwise
  ops, reductions (dot, norm) and the scalar-nonlinear ops solvers need (div, sqrt, exp) — the
  things that usually stall solver loops on pure-GEMM NPUs — *and* the interior-point fp64 path.
  > **fp64 is a requirement, not an option, and it does not come from the array**
  > (costed 2026-07-30; PHASE0_FINDINGS §0.1, `perf_model/fp64_datapath.py`).
  > Interior-point KKT conditioning reaches ~1e12, so ARM-MPC (acados SQP-RTI + HPIPM) needs
  > fp64. Getting it by multi-pass decomposition onto the bf16 multipliers costs **49–361 passes**
  > for error-free accuracy and **3.5–23× the energy** — rejected. Native fp64 across all 144
  > cells costs +1.42 mm² — rejected. **A merged 8-unit cluster costs +0.067 mm²** and meets the
  > deadline at 2.4× margin on worst-case iteration counts. Consequence for the RTL: the vector
  > unit and the fp64 unit are **one scheduling resource**, which the sequencer and codegen must
  > model. Consequence for sizing: 8 units on *every* SKU — fp64 is a **feature bit, not a
  > scaling parameter**, and there is no SKU from which it can be removed while keeping
  > interior-point MPC.
- **Sequencer:** a small RISC-V core (or licensed equivalent) running the solver loop control
  in-fabric: convergence tests, branching, step-size logic — no host round-trip per iteration.
  This is what makes 1–10 kHz MPC feasible. **Plus a hardware descriptor generator: 3 AGUs walking
  a descriptor template under a 3-level hardware loop, and an 8-deep run-ahead descriptor queue.**
  > **Specified 2026-07-30 (PHASE0_FINDINGS §0.2, `perf_model/sequencer.py`).** Sizing target:
  > **≤4 sequencer cycles per descriptor, 1 descriptor/cycle issue width, queue depth 8** —
  > 0.028 mm² all in (0.0137 logic + 0.0143 for an 8 KB I-mem / 4 KB D-mem), inside the existing
  > 0.045 mm² allocation. Issue *width* above 1/cycle buys nothing on any benchmark; **setup cost
  > per descriptor is the axis that matters.** Two things are requirements rather than
  > optimisations: **(a) run-ahead** — a write-descriptor-and-poll sequencer misses PWR-MPC's
  > 50%-of-period budget (38.1 µs vs 25 µs); **(b) dependency-chain blocking ≥ one full datapath
  > width** — a blocked Cholesky that advances only 4 elements per dependent step also misses it.
  > 97–100% of the sequencer's latency contribution is **dependency drain**, not issue throughput,
  > and drain cannot be provisioned away by bandwidth or MACs.
- **Codegen contract:** a code-generated iteration must average **≥ 8 datapath cycles per
  descriptor** (2 × the setup cost) with no burst of sub-threshold descriptors longer than the
  descriptor queue. HPIPM at nb=21 sits at 30 cycles/descriptor, ADMM at n_var=80 at 59, and the
  20 kHz motor-drive row at **10.2 — the thinnest margin in the frozen set.**
- **Memory:** software-managed multi-bank scratchpad + descriptor-based DMA; double-buffering
  so data movement overlaps compute.
- Deterministic execution mode (no cache-miss variance) for real-time / safety markets.

### 3.2 SoC integration ("easy connection" requirement)

| Ecosystem | Interface strategy |
|---|---|
| **ARM SoCs** | AXI4 primary data port + AXI4-Lite/APB control. Optional ACE-Lite or CHI wrapper for coherent variants. AMBA specs are freely licensable from Arm for interoperability. |
| **RISC-V SoCs** | TileLink wrapper (Chipyard/Rocket ecosystem) + standard AXI (most commercial RISC-V SoCs use AMBA anyway). Optionally expose sequencer as a RoCC-style coprocessor for tight coupling demos. |
| **Apple** | Reality check: Apple does not license third-party accelerator IP into its SoCs. "Apple connection" should be interpreted as (a) AMBA-compliant so any ARM-architecture licensee can drop it in, and (b) a software story that mirrors Apple's Accelerate/ANE-style APIs so code ports easily. Do not plan revenue on Apple as an integration target. |

Integration deliverables: parameterizable Verilog/SystemVerilog top with clean clock/reset
domains, single synchronous clock option, standard interrupt lines, optional power-gating
islands, DFT hooks (scan-ready, MBIST-ready SRAM wrappers), and an integration guide with
example SoC glue.

#### 3.2.1 Zero-copy CPU↔CM data path (requirement, added 2026-07-29)

**Goal:** the CPU passes a pointer; the CM computes on the data in place. No memcpy into DMA
buffers, no manual cache clean/invalidate in the per-iteration loop. This directly serves the
latency thesis — at kHz solver rates with small matrices, copy + cache-maintenance overhead
otherwise dominates the solve time.

Offered as three integration tiers (instantiation-time choice, same programmer's model):

| Tier | Mechanism | What licensee gets |
|---|---|---|
| **T0 — Baseline** | Plain AXI4 master + software-managed buffers | Works in any SoC; driver does map/flush; not zero-copy (fallback only) |
| **T1 — IO-coherent** (default) | ACE-Lite / ACE5-Lite (ARM) or coherent TileLink (RISC-V) master port: CM snoops CPU caches | Zero-copy on *physically contiguous / pinned* shared buffers (Linux dma-buf); no cache maintenance; CPU writes descriptors + operands, rings doorbell |
| **T2 — Shared virtual memory** | T1 + SMMU/RISC-V-IOMMU ATS with page-fault (PRI) support: CM uses the process's virtual addresses | True pointer-passing from user space (malloc'd data, no pinning) — unified-memory programming model |

Complementary low-latency mechanisms (all tiers):
- **CM scratchpad mapped into CPU address space** (AXI slave window): for tiny operands
  (gains, references, warm-starts) the CPU stores directly into CM SRAM — cheaper than any
  DMA descriptor round-trip.
- **Descriptor rings + doorbells in shared coherent memory**; completion via interrupt or
  polled flag word (no syscall in the hot loop).
- **Cache stashing** (CHI variant, Phase 4): CM pushes results into the CPU's cache so the
  consumer-side read is a hit.

Software commitments this creates: dma-buf/DMA-heaps support in the Linux driver from day
one, a user-space runtime API where buffers are pointers (not opaque handles), and the codegen
emitting in-place kernels. Verification commitment: coherent-port compliance suite (ACE-Lite
protocol checkers) and an SMMU/IOMMU integration test for T2.

Reality note: T2 depends on the host SoC providing an SMMU/IOMMU with ATS — common in
application-class ARM/RISC-V SoCs, absent in MCU-class parts; T1 is therefore the default
product configuration and T2 a premium option.

#### 3.2.2 Data residency across iterative calls (on-CM caching, added 2026-07-29)

Zero-copy (§3.2.1) makes data *transfer* free of copies; residency makes repeated transfer
unnecessary. Iterative workloads reuse most of their data — the CM must exploit this at two
timescales:

1. **Within one solve (iteration-resident data):** problem matrices, KKT/Cholesky factors,
   and constants stay pinned in scratchpad for the whole solve; only iterates (x, λ) update
   in place. The codegen computes a static residency plan — for embedded MPC, problem
   dimensions are known at code-generation time, so scratchpad allocation is fully static
   and deterministic (no runtime cache misses → preserves the real-time execution model).

2. **Across successive calls (call-to-call persistence):** an MPC/estimator running at
   1–10 kHz re-solves a *structurally identical* problem every tick. The CM keeps problem
   data, factorization structure, and the previous solution (warm-start) resident in SRAM
   between calls; each new call transfers only the delta — new sensor state and references,
   typically a few hundred bytes written directly through the CPU→scratchpad window.
   Steady-state invocation cost ≈ delta write + doorbell, not a full problem upload.

   Coherence/invalidation contract: CM scratchpad is deliberately *not* hardware-coherent
   (determinism, area). Instead, buffers carry **generation tags** — the runtime bumps a
   buffer's generation when the CPU rewrites it; each call descriptor carries a dirty mask,
   and the CM re-fetches (via the coherent T1 port) only buffers whose generation changed.
   Same-generation buffers are trusted as resident. LLM decode maps onto the identical
   mechanism: weights are generation-stable across tokens (resident tiles), KV-cache is
   append-only on-CM state, per-token delta is one embedding vector.

3. **Optional read cache (instantiation parameter):** a small read-only cache on the
   coherent master port for streaming read-mostly data larger than scratchpad (LLM weight
   tiles). Off by default in real-time configs (determinism first), on for ML-heavy configs.

Runtime API implication: buffer allocation takes a lifetime hint —
`ARGMIND_STREAMING` (per-call) vs `ARGMIND_PERSISTENT` (survives calls; generation-tagged) —
and the profiler reports scratchpad residency so users can see when they've oversubscribed it.

### 3.3 Software stack (this is half the product)

IP without software does not get licensed. Plan the stack from day one:

1. **Low level:** C driver/HAL (bare-metal + Linux kernel driver + userspace lib), register
   map auto-generated from a single SSOT (SystemRDL or similar).
2. **Solver path:** code generator that takes problem descriptions (CVXPY / CasADi / OSQP-style
   problem forms) and emits Argmind kernels — analogous to CVXPYgen/acados for embedded MPC.
   This is the moat: nobody hand-writes accelerator kernels for their MPC controller.
3. **ML path:** MLIR-based compiler backend; import from ONNX / PyTorch (torch.export).
   Start with a fixed op library (GEMM, conv-as-GEMM, attention block, layernorm, softmax)
   rather than a general compiler — cover 90% of edge transformer/CNN graphs first.
4. **Simulation targets:** cycle-approximate C++ performance model + Verilator model, so
   customers can evaluate before touching RTL.

### 3.4 Implementation blueprint — zero-copy + residency (§3.2.1/3.2.2)

Build order chosen so every layer is testable before the layer below it exists:

1. **API + functional simulator first (no RTL required).** Define the user-facing contract:
   `am_alloc(size, AM_PERSISTENT|AM_STREAMING)` returning plain pointers, `am_call(desc)`,
   generation tags, dirty masks. Implement it against a C++ functional model of the CM.
   Everything downstream (codegen, benchmarks, user code) targets this API and never changes.
2. **Codegen + residency planner (Python).** CVXPY/CasADi problem in → static scratchpad
   allocation map + kernel schedule + descriptor templates out. Validated entirely on the
   functional model; cycle model added later gives perf numbers.
3. **RTL blocks, in dependency order:**
   a. scratchpad controller + CPU-visible SRAM window (AXI slave) — enables direct-store path;
   b. descriptor ring + doorbell + generation-tag table (small CAM/SRAM) — the residency
      mechanism is ~a lookup table plus a fetch-on-mismatch state machine, deliberately simple;
   c. DMA engine with dirty-mask-driven selective fetch;
   d. AXI4 master (T0), then ACE-Lite wrapper (T1) with open protocol checkers in cocotb;
   e. sequencer + GEMM/vector datapath (developed in parallel — orthogonal to the data path).
4. **Linux driver:** dma-buf importer + UIO/char device for doorbell & rings; generation
   tagging lives in the user-space runtime (mmap'd), so the kernel driver stays thin (~2 kLOC)
   — important for upstreamability and for users porting to their own kernels/RTOSes.
5. **FPGA reference:** Zynq MPSoC (Kria KV260-class, ~$250 board) — its ACP/ACE-Lite-capable
   ports make the T1 zero-copy path demonstrable on hardware anyone can buy.

---

## 4. Competitive landscape

Teardown completed 2026-07-29 — findings and citations in [PHASE0_FINDINGS.md](PHASE0_FINDINGS.md)
§5. Two results strengthen the thesis beyond what this table originally assumed:
**(a)** no commercial NPU IP supports data-dependent control flow — all compile it away offline,
and Cadence requires a *second licensed core* for it; **(b)** none has usable FP matrix
throughput (Ethos-U85 is INT8-weights-only; Ceva's FP32 rate is ~1% of its INT8 rate), so
**NPU IP cannot add solver support in software — it lacks the arithmetic.**

> **Scope limit, added 2026-07-29 — the moat is narrower than this table implies.** Both findings
> hold for *NPU IP only*. Against the products actually in our beachhead socket
> (PHASE0_FINDINGS §5.1):
>
> - **CPUs with matrix/vector extensions (Arm SME2, RISC-V RVV): three of our four
>   differentiators do not exist.** Being *inside* the CPU, they get in-fabric data-dependent
>   control flow, coherent zero-copy access to application memory, and call-to-call cache
>   residency for free, by construction. The sequencer, the T1 coherent port and the
>   generation-tag residency model differentiate against **accelerators, not against CPUs.**
>   SME2 is measured at 2008 GFLOP/s FP32 (Apple M4 P-cluster, 3nm class) and needs no
>   retargeting — only tuned libraries. Uncomfortably, our inner loop is GEMV and
>   bandwidth-bound, and a plain vector unit is the *right* shape for GEMV: RVV's missing matrix
>   extension protects the ML half of the pitch and **not the solver half.**
> - **AURIX TC4x PPU:** a licensed Synopsys ARC EV7x — scalar core + SIMD vector unit +
>   closely-coupled memory with its own instruction stream, i.e. a small version of our own
>   architecture, already automotive-qualified and supported by Embedded Coder. If the optional
>   FP path is instantiated it needs **a solver library, not silicon.** Our residual claims
>   against it (coherent host path, residency contract, bandwidth) are all *not published*, so
>   they cannot be asserted as gaps either.
> - **TI C2000 CLA:** our arithmetic advantage is structural, but in MCU class **zero-copy and
>   residency are void** — no DRAM, no cache, nothing to eliminate. Unresolved tension with §4,
>   where zero-copy decides PWR-MPC deadline feasibility and motor drive is the socket most
>   likely to be MCU-class.
>
> **Net: the moat is energy per solve, deterministic worst-case latency, and the codegen/software
> ecosystem — not arithmetic, not control flow, not coherency.** One strong new determinism leg:
> the M4's SME block is *shared per CPU cluster* (one per four P-cores), so a thread's worst case
> depends on what sibling cores are doing — architectural, not incidental.
>
> Falsifier F4 in [CUSTOMER_DISCOVERY.md](CUSTOMER_DISCOVERY.md) triggers if a PPU or
> same-power-class SME2/RVV part runs benchmarks #1–#3 within **2×** of our modelled energy per
> solve — and given the model's ±2×, "inside 2×" must read as *no measurable advantage*, not a
> narrow win. Gate on the PPU is the Synopsys ARC MetaWare toolchain licence, not the board.

| Player | Offering | Gap Argmind exploits |
|---|---|---|
| Arm Ethos-U/N series | Edge NPU IP | Batch-throughput oriented; no solver/sequential story |
| Cadence Tensilica, Synopsys ARC NPX | DSP + NPU IP | Same; also heavyweight licensing |
| Ceva NeuPro, VeriSilicon | Edge NPU IP | Same |
| Quadric, Expedera, EdgeCortix | Startup NPU IP | ML-only positioning |
| FPGA solver cores (academic) | MPC/QP accelerators | Not productized IP, no ML capability |
| CPU libs (Eigen, OSQP, acados) | Software solvers | 10–100× power/latency gap at kHz control rates |

Positioning: **"the first Computational Module (CM)"** — the only licensable core that runs
the control/optimization loop *and* the neural network on one engine. The CM category label
is ours to define, the way "NPU" was coined and evangelized a decade ago.

### 4.1 Comparison vs in-memory computing (IMC/CIM) and other future architectures

Added 2026-07-29. This is the durability question: does a digital, GEMM-plus-sequencer core
still make sense when compute is moving *into* memory? Summary answer: **yes for the solver
half (structurally, on device physics), no for the raw-inference half (honestly, IMC wins
there) — so the architecture must be designed so an IMC macro can be dropped inside it.**

#### Where IMC genuinely beats us — state this honestly

For **stationary-weight, low-precision matrix-vector work**, an IMC macro beats any digital
datapath on energy per MAC, because it eliminates weight movement entirely (Ohm's/Kirchhoff's
laws do the multiply-accumulate in place). Published macros: 3 nm 12b digital SRAM-CIM at
23.2 TOPS/W, 28 nm hybrid-domain FP SRAM-CIM at 72 TFLOPS/W, 65 nm analog CIM at
33.6–136.2 TOPS/W (ISSCC/arXiv 2024–25). **Argmind must never market itself as beating CIM on
inference TOPS/W** — that claim is false and technically literate customers will catch it.

#### Where IMC structurally fails on our beachhead — the real moat

| Requirement of iterative solvers | Why IMC cannot meet it |
|---|---|
| **Matrices change** (KKT matrix per interior-point step; re-linearized A,B every control tick; active-set changes) | IMC amortizes a *slow, energy-expensive write* over many MVMs. Analog NVM endurance is the wall: state-of-the-art "resilient" ReRAM tolerates ~10⁵ update pulses — a 1 kHz controller rewriting weights each tick exhausts that in **~100 seconds**. This is device physics, not an engineering gap. |
| **Precision & conditioning** (interior-point is ill-conditioned near convergence) | Practical memristive devices give ~4-bit resolution; ≥8-bit conductance states are needed merely to match digital baselines in training. Analog noise, drift, process variation and temperature all inject error where solvers amplify it. Iterative refinement can recover accuracy — but only by adding *more sequential steps*, i.e. more of exactly what we accelerate. |
| **Deterministic, guaranteed latency** (hard real-time deadline every tick) | In analog matrix computing (AMC), solve time is governed by the *minimum eigenvalue* of the problem matrix — latency is data-dependent. Worse, the matrix-inversion circuit is only stable for positive-definite matrices, while MPC KKT systems are routinely indefinite. Non-deterministic, conditionally-stable compute is disqualifying for safety-relevant control. |
| **Loop-carried scalar control** (convergence test, step size, active-set update, branching) | An IMC array is an MVM primitive with no control plane. Every CIM system still needs a sequencer and a low-latency host data path — *which is precisely what Argmind is.* |
| **Scaling to real problem sizes** | CIM demonstrations remain largely confined to sub-128 Kb arrays because analog non-idealities worsen with array size, and there is *no established architecture for interconnecting CIM tiles*. Tile orchestration is an open problem — again our layer. |

Commercial reality check (SemiEngineering, 2026): in-memory compute "hasn't achieved
commercial success," though analog IMC is getting a second chance. What is actually shipping
is **digital SRAM-CIM arriving as a macro inside conventional NPUs** — an evolution of the
multiplier array, not a replacement for accelerator architecture.

#### Strategic consequence: make the compute array swappable (architecture requirement)

Our durable value is the **system layer** — sequencer + ISA, residency/generation-tag model
(§3.2.2), zero-copy coherent host path (§3.2.1), and codegen — none of which cares how the
multiply-accumulate is physically realized. Therefore:

> **Requirement (Phase 1 spec): define a Macro-Agnostic Compute InterFace (MACIF)** — a
> narrow, versioned internal boundary (operand tiles in / accumulators out, with declared
> precision, latency and throughput contracts) between the sequencer/scratchpad layer and the
> matmul array. The digital systolic tile is the first implementation behind it; a digital
> SRAM-CIM macro or a licensed analog macro must be substitutable **without touching codegen,
> runtime API, or host integration.**

**Where exactly the boundary runs (settled 2026-07-30 by the fp64 costing; PHASE0_FINDINGS §0.1):**
MACIF separates the **bf16/int8 tile array (swappable)** from the **fp32/fp64 merged FMA cluster
(not swappable)**. No CIM macro, analog or digital, has a path to 53-bit significands, so the fp64
datapath must live in the fixed part of the CM. Two consequences for the interface we intend to
freeze in Phase 1:

1. **Precision is a declared contract, not an implementation detail.** fp64 is an *algorithm*
   requirement — interior point at cond ≈ 1e12 either has it or returns a wrong answer. So MACIF
   must make precision **queryable** and the runtime must **refuse** an fp64 kernel on a macro that
   does not declare fp64: a hard codegen-time error, never a silently degraded result. Per format,
   declare `supported`, `macs_per_cycle[format]` (**absolute, not a rate factor on some other
   format**), `accumulator_width[format]`, rounding mode and subnormal behaviour. The performance
   model previously derived fp64 from a `0.125` multiplier on the bf16 rate; that is precisely the
   shape of interface that would have been wrong, and it has been deleted.
2. **Keeping fp64 outside MACIF keeps MACIF's contract weak enough for a CIM macro to honour.**
   The rejected multi-pass option would have required error-free limb splitting and an exact
   128-bit accumulator *inside* the macro — a contract analog CIM cannot meet at all. Choosing the
   separate cluster preserves the interface's whole value.

This converts the IMC roadmap from an existential threat into a product line:
`Argmind CM-100` (digital) → `CM-100C` (CIM inference macro behind MACIF, partnered or
licensed) for ML-heavy sockets, while the solver path stays digital and deterministic. The
likely 2030 edge architecture — analog/CIM macro running a learned policy or dynamics model,
digital solver enforcing constraints, one sequencer orchestrating both under a real-time
deadline — is a system *we* are positioned to supply and a bare CIM macro is not.

### 4.2 Can this expand into a CPU or a GPU? (asked 2026-07-30)

**GPU: no, and the reason is structural, not a matter of resources.** A GPU is a
throughput machine that hides latency with massive multithreading. Our measured finding is that
the solver inner loop is matrix-**vector**, bandwidth-bound, and full of loop-carried
dependencies — the single worst fit for that architecture, because there is no parallelism to
hide latency with and per-kernel launch cost dominates a 15-iteration solve. Add the graphics
software burden (Vulkan/OpenCL conformance, drivers, compilers) and a licensable-GPU market
already held by Arm Mali, Imagination and VeriSilicon, and it is not a near-miss. **Where GPUs
matter to us is as neighbours, not targets:** in the robotics socket a Jetson-class GPU does
perception while a deterministic block does the control loop. That is an integration story the
existing AXI/ACE-Lite plan already serves.

**CPU: do not build one — but *get inside* one.** This is the interesting answer, and it is
forced by our own competitive finding (§4.1). Against Arm SME2 and RISC-V RVV, three of our four
differentiators evaporate *because those extensions live inside the CPU* and therefore get
control flow, coherent zero-copy and cache residency for free. The logical response is not to
compete with that position but to occupy it.

Note also what we already are: a sequencer plus a vector/matrix datapath plus a
closely-coupled scratchpad. That is structurally *already* a small CPU-with-matrix-extension —
which is exactly what the AURIX PPU (a licensed Synopsys ARC EV7x) is. We are nearer to that
category than to an NPU.

Three integration depths, as a roadmap ladder rather than alternatives:

| Depth | What it is | Status |
|---|---|---|
| **D1 — bus-attached** | AXI4 / ACE-Lite peer, own sequencer, zero-copy + residency | The current plan (§3.2) |
| **D2 — tightly-coupled coprocessor** | RoCC-style attachment to a RISC-V core; the host issues to us over a private interface, no bus round-trip | Cheap follow-on; strongest fit for MCU-class sockets where D1's zero-copy has nothing to bite on (the unresolved §4.1 tension) |
| **D3 — in-pipeline ISA extension** | Our datapath instantiated *inside* a licensee's core, driven by custom or standard RISC-V instructions | The structural answer to SME2/RVV — if the differentiator is "being inside the CPU," then be inside the CPU |

**Why not just build the CPU.** CPU IP is the most contested market in the industry (Arm,
SiFive, Andes, MIPS, Codasip, plus free cores), it demands a full toolchain, OS, ABI, debug and
memory-model story, and ISA-compliance verification alone would exceed this company's entire
Phase-2 budget. Worse, it discards the differentiator: we would be competing on
general-purpose IPC, where energy-per-solve on iterative workloads is irrelevant. Our advantage
lives in a datapath and a residency model, not in a core.

**The real risks on the D3 path**, which should be tested in discovery rather than assumed:
- The customer *is* the competitor. CPU and MCU vendors are who we would sell an extension to,
  and Synopsys already sells them one (the PPU). Infineon licensing rather than building it is
  encouraging evidence they buy — and evidence about who we bid against.
- **Nonstandard RISC-V extensions fragment the ecosystem**, which RISC-V International actively
  discourages via the profile/RVA process. A custom-opcode block is easy to ship and hard to get
  adopted. The credible variants are a tightly-coupled coprocessor with a stable ABI (D2), or
  proposing solver primitives as a *standard* extension — a multi-year committee exercise that a
  pre-revenue company cannot lead alone, but could co-sponsor.
- D3 gives up the "one engine, two workloads" story: inside someone's CPU pipeline we are an
  arithmetic unit, not a Computational Module with its own sequencer. The category we are coining
  presumes D1/D2.
- **D2/D3 must not move descriptor emission to the host** (quantified 2026-07-30, PHASE0_FINDINGS
  §0.2). "The host issues to us" means ~20 sequencer cycles per descriptor and no reliable
  run-ahead, because the host is also executing the application. Worst solver margin against the
  50%-of-period budget: **D1 1.71× · D2 with the generator moved to the CM side 1.71× · host
  issuing with software double-buffering 1.23× · host issuing with no run-ahead 0.66× (misses)**.
  The binding row is PWR-MPC every time — and PWR-MPC *is* the MCU-class socket D2 is nominated
  for. So D2 stays a wrapper rather than a redesign **provided the wrapper keeps the descriptor
  generator on the CM side**. This is also a MACIF constraint: the descriptor format is part of
  the boundary a CIM macro must honour, so the generator sits *above* MACIF.

**Recommendation:** stay on D1 for Phase 2, design the MACIF and the runtime API so D2 is a
wrapper rather than a redesign, and treat D3 as a licensing conversation to open only once a
D1 design win exists. Do not spend Phase-0 or Phase-1 effort on it.

#### Other future architectures

| Trend | Threat to Argmind | Read |
|---|---|---|
| **Digital SRAM-CIM in edge NPUs** (shipping) | Erodes our *inference* selling point | Real but bounded — absorb via MACIF; never compete on TOPS/W alone |
| **DRAM-PIM / HBM-PIM, near-memory** | Low reuse, irregular-access workloads (graph, DB) — different target | Not our socket; embedded control is latency- and power-bound, not HBM-bandwidth-bound |
| **Chiplets / UCIe, HBM4** (UCIe now de facto standard) | Changes *packaging*, not the compute layer | Opportunity: publish Argmind as a UCIe-attachable chiplet variant later; extends TAM beyond soft-IP licensees |
| **CPU matrix extensions** (Arm SME, RVV) | Narrows the low end — the "just use the CPU" answer gets stronger | Highest-probability real competitor. Already covered by the Phase-0 kill criterion (must show ≥5× perf/W vs Cortex-A+NEON/SME) |
| **Analog matrix computing for optimization** (one-step LP/QP in RRAM, PoliMi/Ielmini et al.) | Direct hit on the solver claim *if* it ever gets deterministic and accurate | Genuine research watch item — precision-, stability- and maturity-limited today, but the one line of work that could invalidate the core thesis. Track publications; re-evaluate annually. |

Framing to use externally: the whole industry agrees the enemy is **data movement**. IMC
attacks it with new device physics (high reward, high risk, immature, foundry-restricted).
Argmind attacks the same problem with **residency and zero-copy in standard CMOS** — lower
ceiling, but available now, portable to any foundry, deterministic, and verifiable today.

---

## 5. IP product deliverables (what a licensee receives)

- Synthesizable SystemVerilog RTL, parameterized (tile size, precisions, SRAM depth, bus type)
- Testbench & verification suite (UVM or cocotb-based), coverage reports
- Synthesis constraints + reference PPA reports (per node, via foundry PDK partners or
  open PDKs initially)
- FPGA prototype/eval platform (e.g., AMD Zynq UltraScale+/Versal board image)
- Software stack (driver, codegen, compiler, examples), benchmark suite
- Integration guide, register map docs, safety/quality documentation (later: ISO 26262
  ASIL-ready package for automotive)

---

## 6. Roadmap & phases

### Phase 0 — Validation & spec (months 0–3)
- Define 5–8 concrete benchmark problems (e.g., quadrotor MPC N=20 horizon, EKF 24-state,
  OSQP problem set, TinyLlama/Phi-class LLM block, MobileNet/ViT block).
- Build an analytical + Python performance model; quantify speedup vs Cortex-A/M, vs a
  representative NPU, vs Jetson-class GPU. **Kill criterion:** if modeled advantage on the
  solver workloads is <5× perf/W vs a Cortex-A + NEON at same node, rethink.
- 15–20 customer-discovery conversations (robotics/automotive Tier-2s, drone makers,
  industrial control vendors). Validate licensing appetite and must-have integration specs.
- Output: architecture spec v1, market memo, go/no-go.

### Phase 1 — Architecture model & ISA (months 3–6)
- Cycle-approximate C++/SystemC model; freeze sequencer ISA + descriptor formats.
- **Freeze MACIF** (§4.1): the macro-agnostic boundary between sequencer/scratchpad and the
  matmul array, so a future digital-CIM or analog macro is substitutable without disturbing
  codegen, runtime API, or host integration. Cheap now; near-impossible to retrofit later.
- Solver codegen prototype targeting the model (CVXPY/CasADi front door).
- Publish a technical whitepaper (credibility + inbound interest).

### Phase 2 — RTL & FPGA prototype (months 6–15)
- RTL: GEMM tile + vector unit + sequencer + DMA + AXI wrapper; ACE-Lite (T1 zero-copy)
  port with protocol-checker verification. (FPGA caveat: Zynq/Versal expose IO-coherent
  ports — ACP/ACE-Lite on MPSoC — so the zero-copy path is demoable on the prototype.)
- Open-source-first flow to control cost: Verilator + cocotb regression, Yosys for early
  synthesis sanity; commercial EDA (rented/cloud: Synopsys/Cadence via startup programs)
  for real timing/power numbers.
- FPGA bring-up on Zynq/Versal; run the Phase-0 benchmark suite end-to-end on silicon-ish
  hardware. Demo video: kHz-rate MPC + on-device SLM on the same core.

### Phase 3 — Productization & first deals (months 15–24)
- Verification hardening (coverage closure, formal on control paths), documentation,
  packaging, licensing collateral.
- PPA characterization on at least one commercial node (via foundry/EDA startup programs,
  e.g., 22/16/12 nm class) — paper tape-out only.
- 2–3 evaluation licenses → target 1 paid design win.
- Optional Phase 3b: shuttle test chip (TSMC/Europractice/Efabless-class shuttle) if
  customers demand silicon proof; budget separately.

### Phase 4 — Scale (months 24+)
- CHI variant with cache stashing, T2 shared-virtual-memory option (SMMU/IOMMU ATS+PRI),
  multi-core scaling, ASIL-ready package, bigger LLM support (KV-cache management, FP8),
  design-services revenue alongside licensing.

---

## 7. Business model

- **Revenue:** per-design license fee + per-unit royalty (standard IP model); evaluation
  licenses (paid or free-with-NDA); NRE/design services for customization early on
  (services fund the roadmap before royalties arrive).
- **Pricing anchor:** edge-NPU IP licenses typically run low-hundreds-of-k$ to low-M$ per
  design + royalties; validate in Phase 0 conversations.
- **Ecosystem moves:** RISC-V International membership; Arm AMBA spec license (free);
  university collaboration for benchmark credibility; open-core strategy per §7.1.

### 7.1 Open-source distribution & adoption strategy (GitHub)

**Model: open core.** Everything a developer needs to *evaluate, integrate, and build
software for* Argmind is open; the full-performance parameterized RTL + verification suite +
PPA collateral (what a chip company pays for) stays proprietary.

| Repo | Contents | License |
|---|---|---|
| `argmind-sdk` | Runtime lib (pointer API, generation tags), Linux driver, RTOS/bare-metal HAL | Apache-2.0 |
| `argmind-codegen` | CVXPY/CasADi → CM kernels; ONNX importer; residency planner | Apache-2.0 |
| `argmind-sim` | Functional + cycle-approximate simulator, `pip install argmind` backend | Apache-2.0 |
| `argmind-lite` | Reference RTL: small fixed config (e.g., 8×8 tile, T0/T1, RV32 sequencer) | Solderpad-2.1 |
| `argmind-examples` | Kria/PYNQ prebuilt bitstream + notebooks, Chipyard & LiteX integration, MPC/EKF/SLM demos | Apache-2.0 |
| `argmind-benchmarks` | Reproducible benchmark harness + published results | Apache-2.0 |

Design principles behind the split:
- **Apache-2.0 for software** (explicit patent grant — required for corporate adoption);
  **Solderpad-2.1 for open RTL** (the hardware-standard Apache variant). **Trademark stays
  owned:** anyone can use the code, only we ship "Argmind®" — the Arm/RISC-V-era moat is
  brand + verification + support, not source secrecy.
- **`argmind-lite` is genuinely useful but deliberately small** — big enough for real MPC/EKF
  workloads and student SoC projects (Chipyard, LiteX, Efabless shuttles), small enough that
  serious ML/production users need the licensed parameterized core. NVDLA's failure mode
  (open RTL, no maintained software, community stalled) is avoided by making *software* the
  center of gravity, as Arm did with Ethos-U/Vela.
- **The API is the product's constitution:** `argmind-sdk`'s interface is identical across
  simulator, argmind-lite, and the commercial core — code written against the free stack
  runs unmodified on the licensed silicon. That is the adoption→revenue bridge.

**Adoption on-ramps, lowest friction first:**
1. `pip install argmind` → run your existing CVXPY/CasADi problem on the simulator, get
   cycle/latency estimates in 10 minutes, no hardware, no signup.
2. $250 Kria board + prebuilt bitstream + PYNQ notebooks → real hardware numbers in an
   afternoon, no EDA tools.
3. `argmind-lite` in Chipyard/LiteX → RISC-V SoC builders and universities instantiate the
   CM in their own designs; Efabless-class shuttle projects give free silicon proof points.
4. Framework plugins: CVXPY solver interface, acados/CasADi backend, ONNX Runtime execution
   provider — meet users inside tools they already use.
5. Commercial license → same API, full-size core, verification collateral, support.

**Community mechanics:** public CI (Verilator + cocotb regressions on every PR), versioned
ISA/descriptor spec as a repo (spec-first culture, like RISC-V), CONTRIBUTING + DCO,
benchmark PRs welcome (reproducibility builds credibility), Discord/discussions for support.
Cadence: quarterly tagged releases; simulator/codegen accept community targets.

**Funnel math to watch:** GitHub stars are vanity; track (a) `pip` installs → (b) Kria
bitstream downloads → (c) argmind-lite instantiations in public repos → (d) inbound
commercial-license conversations. The open stack exists to manufacture (d).
- **IP protection:** patent 2–3 core claims early (sequencer↔GEMM coupling mechanism,
  small-matrix scheduling, deterministic-latency execution model); trade secret for the rest.

## 8. Team, tools, budget (lean scenario)

- **Minimum viable team:** 1 architect/RTL (you/founder), +1 RTL/verification, +1 compiler/SW
  by Phase 2. Fractional: DFT/synthesis consultant, IP lawyer.
- **Tools:** open-source core flow (Verilator, cocotb, Yosys); EDA startup programs
  (Synopsys/Cadence/Siemens all have them) for signoff-grade runs; 1–2 FPGA boards ($5–15k);
  cloud compute for regressions.
- **Cash burn drivers:** salaries ≫ tools if open-source-first; test chip is the big
  discretionary item ($50–300k depending on shuttle/node).

## 9. Risks & mitigations

| Risk | Mitigation |
|---|---|
| NPU incumbents add solver features | Move fast on the codegen/software moat; patents on coupling mechanism |
| "IP + software" scope too big for small team | Fixed op library + codegen for one solver family first (QP/MPC), not a general compiler |
| Customers demand silicon proof | FPGA demo first; shuttle chip only after eval-license interest |
| Apple/first-party SoCs unreachable | Treat as compatibility target only; revenue plan built on ARM licensees + RISC-V vendors |
| Verification cost explosion | Parameterization discipline (few validated configs, not full matrix); formal on control logic |
| One-person start | Phase 0/1 are deliberately model-and-paper heavy — solo-feasible; hire at RTL phase |
| **IMC/CIM erodes the inference half** of the pitch (§4.1) | MACIF makes a CIM macro adoptable rather than fatal; never market on inference TOPS/W; lead with solver latency + determinism |
| **Analog matrix computing matures for QP/LP** — would hit the solver thesis directly | Annual re-evaluation; today blocked by precision, PD-only stability, data-dependent latency. Watch PoliMi/Ielmini-line publications |
| System features (zero-copy, residency) are copyable by incumbents — not a physics moat | Patent the specific mechanisms (generation-tag residency, sequencer↔GEMM coupling); compound the software/codegen ecosystem where copying is slowest |

## 10. Next actions (30/60/90 days)

> **Status 2026-07-29:** the 30-day items are substantially done — benchmark set frozen
> ([BENCHMARKS.md](BENCHMARKS.md) v1.0), performance model v0 built and run
> ([perf_model/](perf_model/)), competitor teardown complete. Interim verdict: **GO with three
> amendments** (bandwidth-first provisioning, energy/determinism positioning rather than raw
> latency, ML claim scoped to resident-weight models). Full results and the revised
> recommendation are in [PHASE0_FINDINGS.md](PHASE0_FINDINGS.md).
>
> Remaining Phase-0 critical path, in priority order:
> 1. **Measure the real CPU baselines** (OSQP/acados on A55- and A78-class hardware). The
>    modelled CPU efficiency assumption is the largest error source in every ratio we have.
> 2. **Cost the wide-scratchpad banking** in area and power — the architecture now rests on it.
> 3. **Customer discovery** (15–20 conversations) — not started, and now the binding constraint
>    on the go/no-go, since the technical case is provisionally made.

- **30 days:** benchmark problem list frozen; Python perf model v0; competitive teardown of
  Ethos-U85, ARC NPX, Ceva NeuPro-M specs; start customer-discovery outreach list.
  Add to the teardown: one CIM data point (a published digital SRAM-CIM macro) as the
  energy-per-MAC *upper bound* we concede, so the perf model reports honest headroom (§4.1).
- **60 days:** perf model results vs CPU/NPU baselines; draft architecture spec v0.5;
  10 discovery calls done.
- **90 days:** go/no-go review against kill criterion; if go — spec v1, whitepaper outline,
  begin C++ cycle model.

---

*Decisions still open (flag for review): beachhead segment (MPC/robotics recommended),
FP32-capable vs inference-precision-only GEMM engine, RISC-V sequencer core choice
(in-house tiny core vs. licensed e.g. Ibex/CV32E40P), test-chip yes/no timing.*
