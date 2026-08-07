# Argmind — Computational Module (CM) — Plan

> ## STATUS: reframed as a side project (2026-08-07). Read this first.
>
> This document was written as a **startup plan** — fabless IP licensing, design wins, royalties.
> That framing is **parked**. The project is now a side project: open source, donation-funded at
> most, realistically stopping at the software and simulator layer rather than RTL and silicon.
>
> What that changes:
>
> - **Still current and load-bearing:** everything architectural. §3 (hardware architecture,
>   zero-copy, residency), §3.3 (scaling), §3.4 (implementation blueprint), §4.1 (in-memory
>   computing), §4.2 (CPU/GPU expansion), and the RISC-V ruling in §3.2. These are technical
>   conclusions and they do not depend on the business model.
> - **Parked:** §2 (markets), §5 (deliverables), §6 (roadmap phases 2–4), §7 (business model),
>   §8 (team/budget), §9 (risks), §10 (30/60/90 actions), and §4.15's market-route rulings.
>   Kept rather than deleted because the analysis is not reconstructable and the framing could
>   change back. Moved company-specific documents to [`business/`](business/).
> - **Wrong-sized for a side project and should not be pursued:** the 17-interview customer
>   discovery program, ISO 26262 / SEooC certification packages, PPA characterisation, foundry
>   engagement, eval-board purchases, and the Phase-2 RTL path with its verification cost.
> - **One decision is now conscious rather than deferred:** the §7.1 publication hold exists to
>   preserve patentability. For a side project a provisional filing is probably wasted money and
>   free publication is the point — but publication permanently closes European patentability
>   (EPC Art. 54, no grace period). That door does not reopen. Decide it explicitly.
>
> Entry point for the project itself is now [README.md](README.md).

**Product category (coined):** Computational Module (CM) — a licensable IP class that runs
both iterative optimization algorithms and matrix-multiplication workloads on one engine.
Category positioning: "the industry's first Computational Module."
**Brand name:** **Argmind** (decided 2026-07-29) — argmin + *mind*: "the mind that
minimizes." Keeps the math meaning for the optimization audience, adds the AI connotation.
**NOT CLEARED** — and note what that means ([IP_STRATEGY.md](business/IP_STRATEGY.md) F-4): no trademark
register was actually searched, because all ten access routes failed (USPTO search is a JS app
with a POST-only API; EUIPO eSearch likewise; TMview has a TLS certificate mismatch;
Justia/Trademarkia return 403). "No blocker found" therefore means *nothing was looked at*, not
that the name is clear. One live neighbour does exist: **Argmin AI Inc.** (argminai.com), an AI
software company plausibly in classes 9/42. `github.com/argmind` and PyPI `argmind` are
unclaimed. A 30-minute manual register check is specified in IP_STRATEGY §1.
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

### 0.1 Naming decision rule (added 2026-07-30) — do not re-litigate on web searches

**What actually blocks a name:** a registered mark in **Nice class 9 or 42 in the US/EU**, or any
**semiconductor, EDA or IP** company with a confusingly similar name. Trademark protects against
confusion *in the relevant market*.

**What does not block it:** a same-named software consultancy in an unrelated field. Testing
showed essentially every pronounceable, semantically-rooted 2–3 syllable tech word is already
held by some small IT firm somewhere — Convexa (Norwegian VC + an IT services firm), Proxa (Hong
Kong software), Iteron (Swiss/German AI consultancy), Solvara (Indian dev shop), Karush (Nebraska
.NET consultancy, owns karush.com), Kairon (three companies). **That space is exhausted
worldwide, so "someone shares the name" is the normal condition, not a disqualifier.**

**Current status of Argmind:** the only concerning neighbour is **Argmin AI Inc.**
(argminai.com) — concerning specifically because it is plausibly class 9/42 *and* AI-adjacent,
unlike the consultancies. Whether it is actually registered in those classes is **unknown**: no
register was reachable ([IP_STRATEGY.md](business/IP_STRATEGY.md) F-4).

**Rule: do not rename on the strength of a web search.** Renaming costs a rewrite of every
document, the SKU scheme, the GitHub org and the PyPI name. The 30-minute manual register check
in IP_STRATEGY §1 is the only thing that decides it. Do that first.

**If Argmind is blocked, two things change the approach:**

1. **Go fully coined.** Only invented strings came back empty — `Detenta` and `Isochra` returned
   *zero* results, against 3+ companies for every rooted candidate. Industry precedent is
   overwhelmingly coined (Nvidia, Ceva, Altera, Rambus, Codasip, Quadric, Groq, Cerebras,
   Tenstorrent). Coined marks are also the strongest trademark category, whereas "argmin" is
   arguably *descriptive* of the function to our own audience — the weak end of the spectrum.
2. **Pick for determinism, not minimization.** Our own findings say the moat is *energy per solve
   and bounded worst-case latency* — and that on latency we are at or near parity with a CPU. A
   name advertising `argmin` therefore markets the weakest claim we have. `Detenta` (from *detent*
   — the mechanical feature holding a mechanism at a definite, repeatable position) and `Isochra`
   (from *isochron* — a line of equal time) both point at determinism instead, which is what we
   actually sell.

Screening method for any further candidate: web search must return **<5 unrelated results**,
then `.com` / GitHub org / PyPI must be free, then the class 9/42 register check. In that order —
the cheap filters first.

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
  > per descriptor is the axis that matters.**
  >
  > **DOWNGRADED 2026-07-30 (PHASE0_FINDINGS §0.3, PWR-MPC re-spec).** Both of the following were
  > justified by a deadline miss on PWR-MPC; re-specified as finite-control-set MPC that row has
  > **zero dependency chains** and is invariant to both, so **no frozen-set row now misses from
  > either alone**: **(a) run-ahead** and **(b) dependency-chain blocking ≥ one full datapath
  > width**. What survives, on QUAD-MPC's **measured** worst-case tail: **you may implement at most
  > one of the two omissions** — without run-ahead 2.21×, with scalar chain blocking 1.27×, with
  > both absent **0.37× (misses)**. Chain blocking is worth **2.3×** of that margin, the largest
  > single lever in the analysis, so it remains an **RTL requirement** — now on a thin measured
  > margin rather than on a deadline miss. Run-ahead becomes *required unless chain blocking is
  > full-width*. 97–100% of the sequencer's latency contribution is still **dependency drain**, not
  > issue throughput, and drain cannot be provisioned away by bandwidth or MACs.
- **Codegen contract:** a code-generated iteration must average **≥ 8 datapath cycles per
  descriptor** (2 × the setup cost) with no burst of sub-threshold descriptors longer than the
  descriptor queue. HPIPM at nb=21 sits at 30 cycles/descriptor, ADMM at n_var=80 at 59, and the
  20 kHz motor-drive row at **10.2 — the thinnest margin in the frozen set.**
  > **RE-CHECKED against the built implementation 2026-07-30** (`argmind.descriptor`
  > `SEQUENCER_RECHECK`). The measured count is **31 descriptors per checked ADMM iteration, not
  > the 28 assumed**, and 48% of them are the convergence test. **The average is not threatened:**
  > at fp32 / 128 B per cycle the block runs **238 cycles/descriptor, 29.8× the bar** (fp64: 474,
  > 59.3×), because the residual block is 48% of descriptors *and 49% of cycles* — its mix is two
  > full A-products, not the seven cheap reductions.
  > **The queue-depth clause is where it is thin, and that is now a measurement rather than an
  > assumption:** the residual block contains a run of **six consecutive descriptors averaging 5.8
  > cycles**, below the bar. Six fits an 8-deep queue so run-ahead absorbs it — which **confirms
  > queue depth 8 as a floor** (4 would not absorb it) and makes this the thinnest point in the
  > sequencer analysis. The fix is a MACIF capability rather than silicon: **`maxabs_accumulate`**
  > folds the ∞-norm into the producing pass, taking the block 15 → 10 descriptors and the run
  > 6 → 4.
- **Memory:** software-managed multi-bank scratchpad + descriptor-based DMA; double-buffering
  so data movement overlaps compute.
- **Scratchpad capacity: 192 KB, 16 banks × 12 KiB (= 128 B/cycle).**
  > **CORRECTED 128 → 192 KB, 2026-07-30, by MEASUREMENT** (descriptor v0.1 ruling R8;
  > `argmind.descriptor.SCRATCHPAD_RULING`). The functional simulator's measured static allocation
  > plan for QUAD-MPC is **163.8 KiB at fp32** (138.8 KiB with `A = [I;Γ]` exploited) against the
  > 128 KB costed. Two model errors, both flattering: P and L were charged n²/2 each — **a packed
  > triangular layout that base+leading-dimension AGUs cannot express**, so the capacity model
  > assumed a storage layout the descriptor format cannot address; and working vectors were charged
  > **zero** (14.1 KiB). **128 KB was arithmetically impossible, not tight: the flagship's three
  > matrices alone are 125.0 KiB of 128.0 KiB.** Cost: core 0.698 → 0.786 mm² (+12.6%), pJ/byte
  > 0.718 → 0.839, solver energy +5.5%, QUAD-MPC perf/W 14.13× → 12.55×. **Latency does not move,
  > and neither bandwidth nor bank count changes — this is a capacity correction, not a
  > rebalancing, and the bandwidth-first finding is untouched.** 192 rather than 160 KB because
  > 12 KiB/bank (1536×64b, array efficiency 0.394) dominates 10 KiB/bank (0.354). It also holds the
  > *naive* 163.8 KiB layout at 85.3%, so the flagship fits even from a code generator that has not
  > done the `[I;Γ]` optimisation. **ARM-MPC (323 KiB honest) overflows at 128, 192 and 256 KB and
  > always did** — 250 Hz, ~10× deadline margin, 0.034 GB/s of streaming demand; it does not drive
  > capacity. Capacity is set by the **fastest determinism-critical loop**, not the biggest
  > footprint.
- **Descriptor format v0.1** (`argmind/descriptor.py`): **8 words / 32 B**, three base addresses,
  packed 16-bit dims, `chain_token` 3b + `wait_mask` 8b, two precision fields (`prec_m` for the
  bulk/matrix operand, `prec_v` for vectors and results), 3-bit `structure`, 2 bits reserved.
  Ten opcodes: GEMM GEMV TRSV AXPY PROJ DOT SCAL CHOL **NRMI VMUL**.
  > **v0.1 rules on all seven gaps that implementing v0 exposed, and it is paid for entirely by
  > reclaiming v0 waste** — v0 had zero spare bits, and narrowing `chain_token` 8→3 and `prec_m`
  > 4→3 funds `prec_v` and the wider `structure`. **No extra word, so descriptor bandwidth, queue
  > SRAM and AGU count are all unchanged.** Load-bearing consequences for the RTL:
  > **(a)** a token is a **scoreboard slot, not a name** (measured peak liveness 7), and reuse is
  > safe **only if a descriptor that reclaims a token also waits on it** — otherwise a waiter that
  > resolves a token to a later owner gets no guarantee about the earlier one, because in-order
  > issue does not imply in-order completion;
  > **(b)** a **zero AGU stride is legal on reads (broadcast) and illegal on writes**, and a
  > zero-stride read must be served from the **L0 operand staging register, not re-fetched**, or n
  > reads of one bank word serialise and the broadcast is slower than the vector it replaces;
  > **(c)** elementwise ops walk **M×N**, so the AGU needs a 2-level walk (the descriptor generator
  > already has a 3-level loop) and the n-cap goes 255 → 65535;
  > **(d)** PROJ takes **interleaved (lo,hi) bound pairs**, which is why no fourth AGU is needed —
  > and at fp32 a pair is 8 B = exactly one bank word.
- Deterministic execution mode (no cache-miss variance) for real-time / safety markets.

### 3.2 SoC integration ("easy connection" requirement)

| Ecosystem | Interface strategy |
|---|---|
| **ARM SoCs** | AXI4 primary data port + AXI4-Lite/APB control. Optional ACE-Lite or CHI wrapper for coherent variants. AMBA specs are freely licensable from Arm for interoperability. |
| **RISC-V SoCs** | TileLink wrapper (Chipyard/Rocket ecosystem) + standard AXI (most commercial RISC-V SoCs use AMBA anyway). Optionally expose sequencer as a RoCC-style coprocessor for tight coupling demos. |

> **RISC-V exposure is an implementation detail, not an ecosystem commitment** (recorded
> 2026-08-07; two independent analyses reached this and it should stop being reopened).
> §4.2 found that **nonstandard extensions fragment the ecosystem and RISC-V International
> actively discourages them** via the profile process, so a custom-opcode block is easy to ship
> and hard to get adopted. Independently, the sequencer analysis found that **descriptor emission
> must stay on the CM side at every integration depth**, because host-side issuing without
> run-ahead lands at 0.66× on the binding row. Together: **we need no custom instructions in the
> licensee's CPU.** The sequencer is ours and internal; any custom instructions live inside our
> block and are never exposed to licensee software. No fragmentation debate, no ratification
> dependency, no toolchain fight.
>
> Consequences: **core choice is decided by inherited verification, not features** — Ibex
> (Apache-2.0, verified via OpenTitan) or CV32E40P (Solderpad, ships verification collateral) both
> beat building our own, because verification is the largest Phase-2 cost and our sequencer's job
> (~22 instructions to arm a template) does not justify a bespoke ISA. **Membership is not
> required to implement the ISA** — it matters for influencing extensions or using the trademark,
> neither of which is a priority at zero customer conversations.
>
> Two things worth borrowing from how RVI governs itself: **open the specification, own the
> trademark** (exactly our open-core plan, validated at scale), and **profiles as an
> anti-fragmentation mechanism** — whose analogue is our own §9 discipline of a few validated SKUs
> rather than a full parameter matrix.
>
> And the honest note: **RVV does not protect the solver half of our story.** Our inner loop is
> GEMV and bandwidth-bound, and a plain vector unit is the *right* shape for GEMV, so RISC-V's
> missing matrix extension protects only the ML half. In the RISC-V ecosystem our position is an
> **adoption path** (Chipyard/LiteX, university and shuttle projects — §7.1's third on-ramp), not
> a competitive front.
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

### 3.3 Scaling band and SKU family

> **Added 2026-07-31 from the scaling analysis** — full derivation, walls and numbers in
> [PHASE0_FINDINGS.md](PHASE0_FINDINGS.md) §8, model in
> [perf_model/scaling_study.py](perf_model/scaling_study.py). **All figures modelled-estimate,
> 16nm-class, ±2×, not quotable externally.**

**Exactly two axes scale by more than 3×**: scratchpad bandwidth (64 → 256 B/cycle, 3.87× on
QUAD-MPC worst case) and the fp64 FMA cluster (4 → 16 units, 3.76× on ARM-MPC). Over the same
defensible ranges: frequency 1.64×, capacity 1.29× *as a cost*, MAC array **1.00×**. **This is a
~4× configuration band and it should be described as one.** It is a legitimate position for
control-domain IP, where problem sizes are set by actuator counts and physics rather than by a
scaling roadmap — but it must not be marketed as an open-ended parameterization.

**The balance rule that ties the two main knobs together** (derived, and agreeing with the sweep):
a GEMV consumes one matrix element per MAC, so **min tile = √(bw / (elem·scale)) = √(bw/2) at
fp32**. A 12×12 array absorbs GEMV up to **288 B/cycle** — and the same 288 at fp32, bf16 and int8,
because this datapath's rate multiplier is ~1/element-size. So CM-100 has 2.25× MAC headroom over
its 128 B/cycle, which is why 256 B/cycle is a drop-in upgrade rather than a redesign, and why
"16× more MACs bought zero" is a statement about **64–128 B/cycle** rather than about arrays in
general.

**The family is TWO validated SKUs.** Everything else is either a replicative synthesis parameter
or a documented-but-unverified configuration.

| SKU | MACs | B/cyc | banks | KB | lanes | fp64 | core mm² | target |
|---|---|---|---|---|---|---|---|---|
| **CM-100** | 144 (12×12) | 128 | 16 × 8 B | 192 | 16 | 8 | **0.786** | robotics/drone node: condensed-QP MPC + state estimation + CNN/ViT ML + interior-point NMPC |
| **CM-50** | 64 (8×8) | 64 | 8 × 8 B | 64 | 8 | 4 | **0.424** | MCU-class / motor-drive node: FCS-MPC, small EKF, small condensed QP |

**Datasheet scoping line, non-negotiable: CM-50 does not support the QUAD-MPC class.** Its 64 KB
holds 0.46 of that 138.8 KB working set. It still *meets* the deadline; it stops being
deterministic, which is the thing being sold. CM-50's honest envelope is **n_var ≤ ~40 (ADMM),
n ≤ ~48 (EKF), and finite-control-set MPC of any size.** It keeps fp64 — no solver SKU can drop it.

Rejected with reasons (§8.9): a **16-MAC solver-only SKU** (saves only 18% of core — the array is
17% of the part — and costs 2.68× the EKF-96 energy, on a beachhead workload) and a **256 B/cycle
CM-200** (a real Pareto point with no workload in the frozen set).

**fp64 up to `lanes/2` is free.** The merged cluster is sized by `max(fp64_fmas, lanes/2)`, so 1,
2, 4 or 8 fp64 units at 16 lanes all cost the same. The cost is the multi-format merge premium
(+0.067 mm² at 16 lanes), paid **once**, not per unit. Interior-point capability is a
~9%-of-core decision, not a scaling axis with a running cost.

**One new architectural requirement falls out of the multi-CM analysis: descriptor-granularity
preemption.** A QUAD-MPC worst-case solve is 9.4 periods of a 40 kHz FCS-MPC loop, so a CM that is
non-preemptible at *solve* granularity cannot host mixed-rate loops at all. Preemptible at
*descriptor* granularity, worst-case blocking is the longest single descriptor (~2.0 µs, 8% of a
25 µs period). This is a sequencer/scoreboard feature — save wait-mask and token state, admit the
higher-priority template — and it is what makes consolidation safe, not a second CM.

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
> - **TI C2000 CLA:** our arithmetic advantage is structural, but in MCU class **residency and
>   coherency are void** — no DRAM, no cache, nothing to eliminate. **Invocation overhead is NOT
>   void, and that distinction resolves the tension with §4** (settled 2026-07-30, §0.3): the honest
>   MCU-class launch costs are ~2.0 µs copy-based vs ~0.30 µs doorbell [EST], a 6.7× / 1.7 µs
>   advantage rather than the rich-host 31× / 24 µs, because buffer-map and cache-maintenance are
>   MMU/cache artefacts an all-SRAM part lacks. ~78% of the surviving advantage is invocation
>   protocol, ~22% an avoided on-chip copy. **Pitch MCU class on low invocation overhead,
>   determinism and energy — never on zero-copy.** And the re-specified PWR-MPC meets its deadline
>   in every tier, so nothing "decides feasibility" there.
>
> **Net: the moat is energy per solve, deterministic worst-case latency, and the codegen/software
> ecosystem — not arithmetic, not control flow, not coherency.** One strong new determinism leg:
> the M4's SME block is *shared per CPU cluster* (one per four P-cores), so a thread's worst case
> depends on what sibling cores are doing — architectural, not incidental.
>
> Falsifier F4 in [CUSTOMER_DISCOVERY.md](business/CUSTOMER_DISCOVERY.md) triggers if a PPU or
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
3. **Accumulator capabilities are part of the contract, and `maxabs_accumulate` joins them
   (MACIF 0.2, 2026-07-30).** The declared set is now `broadcast_init`, `square_accumulate` and
   **`maxabs_accumulate`** — a running max-absolute carried alongside the sum, folding an
   **infinity-norm** reduction into the pass that produces the vector. It is not a nicety: OSQP's
   stopping rule is stated entirely in ∞-norms, so `square_accumulate` folds the *wrong* norm for
   the beachhead solver, and without the fold each residual norm costs a separate descriptor plus a
   re-read. That is 5 of the ADMM residual block's 15 descriptors, and they are the *short* ones —
   the one place the ≥8-cycles-per-descriptor contract is measurably thin (a run of six descriptors
   averaging 5.8 cycles). Folding takes the block to 10 descriptors and the run to 4. Cost is a
   compare-select and one register per accumulator lane — **cheaper than `square_accumulate`, which
   needs a multiplier.** Same principle that decided the array-size question: fold work into the
   accumulator at O(1) area instead of buying array at O(T²).
4. **Both precision fields are checked, not just the operand one.** Descriptor v0.1 lets the store
   format differ from the bulk-operand format (`prec_v` vs `prec_m`), which makes "fp16 matrix,
   fp32 result" expressible — and makes it a *reachable mistake* to build a kernel whose store
   format the macro does not declare. The runtime refuses on either field, at build time. The
   existing refusals are unchanged and still hard errors: **no fp64 path**, and **fp16 ↔ bf16
   substitution in either direction** (same width, ~10× the control-action error on the solver
   path).

This converts the IMC roadmap from an existential threat into a product line:
`Argmind CM-100` (digital) → `CM-100C` (CIM inference macro behind MACIF, partnered or
licensed) for ML-heavy sockets, while the solver path stays digital and deterministic. The
likely 2030 edge architecture — analog/CIM macro running a learned policy or dynamics model,
digital solver enforcing constraints, one sequencer orchestrating both under a real-time
deadline — is a system *we* are positioned to supply and a bare CIM macro is not.

### 4.15 Market-route rulings (2026-08-07) — recorded so they stop reopening

See [WHERE_WE_COMPETE.md](business/WHERE_WE_COMPETE.md) for the socket-by-socket evidence and the
seven-point envelope test. These are the *route* decisions behind it.

**There is no B2C route, and the question that matters is different.** We sell synthesizable
RTL; no form of it can be sold to a consumer. Reaching one needs silicon — mask set, packaging,
test, certification, distribution, warranty — which is a different company with a different
capital structure. And consumers do not buy determinism; they buy range, battery life and
"it doesn't get stuck."

So the route is **consumer pull, not consumer sales**: a metric the consumer pays for → an OEM
competes on it → a chip vendor accommodates the OEM → our licence. Arm sells to nobody's
consumer and is in every consumer product. Under that lens the two best sockets are **consumer
robotics** (battery life *and* reliability are the purchase criteria, workload is EKF+planning,
rich host so all four differentiators survive) and **personal mobility** (range *is* energy, and
FCS-MPC is our highest perf/W row — but latency is 0.62×, so energy only).

A developer/maker module (FPGA board + runtime + codegen) is the only consumer-*adjacent* thing
that works, and it should be accounted as **demand-validation expense, not revenue.**

**Live learning: the strong half is online estimation, not backprop.** Backprop on-device needs
gradients, optimiser state and stored activations — roughly 3× inference memory — which puts the
trainable-resident envelope at ~0.07–0.2 M parameters, and it is batched GEMM, i.e. NPU/GPU/CIM
territory. What robots actually need to learn in the field is *themselves and the world*: a worn
joint, a changed payload, a different floor. That is **recursive estimation** — an EKF estimating
parameters is the same machinery as one estimating states, which is our **best row** (EKF-96,
2.1× / 21.1×) at a precision NPUs cannot provide (fp32 minimum, **no fp16/bf16 path in any
filter form**, measured).

The differentiated shape is **estimate → update model → re-solve, in fabric, in one deterministic
tick.** Note that the G5 bug fixed on 2026-07-31 — a dirty mask refreshing an operand while
leaving the factorisation derived from it stale — is a defect that *only occurs when the model
changes*. The plumbing for the adaptive loop is already in place.

> **OPEN, and measurable with what exists:** if the model changes every tick, the Cholesky
> re-runs every tick. Measured in the static case it ran on **1 of 200 ticks**, which is what the
> 46× residency figure rests on. Re-factorisation is O(n³) against O(n²) per iteration, so at
> 200-of-200 the claim may not survive. The simulator already reports `setup_ran` per tick, so
> this is a day's work and it decides whether "live learning" is a spec or a slogan. Physics
> suggests it survives — joint wear does not change on millisecond timescales — but that is a
> hypothesis, not a result.

**Certification: the specific standards, since "certification" was used loosely.** For an IP
supplier the relevant instrument is **ISO 26262:2018 as a Safety Element out of Context (SEooC)**
— we are not certified; we ship a safety package (safety manual documenting Assumptions of Use,
FMEDA, DFA, verification evidence) that lets a licensee achieve *their* certification. ASIL-D
hardware metrics are SPFM ≥99%, LFM ≥90%, PMHF <10 FIT; assessment is by a third party such as
TÜV SÜD or exida. Competitors already hold ground here: Ceva claims ASIL-B and Synopsys ships
ARC NPX6FS for functional safety. Industrial sockets use **IEC 61508 (SIL)**, **ISO 13849 (PL)**,
and for our motor-drive socket specifically **IEC 61800-5-2**. Aviation (DO-254) is out of
near-term scope.

Where determinism actually enters, stated precisely because it was overstated before: ISO 26262
requires timing evidence, and the mechanism is **WCET analysis**. Our contribution is that static
code-generated allocation with no cache makes the **per-iteration cost computable rather than
measured**. But the *iteration count* remains data-dependent — measured p50 12 → max 60 — so a
whole-solve WCET additionally requires **fixed-iteration or anytime operation, with the applied
input taken from the projected (feasible) iterate.** Half hardware, half codegen contract.

For the learned part the standard is **ISO 21448 (SOTIF)**, which addresses *functional
insufficiency* rather than malfunction — the correct name for what was earlier called
"certification" in the context of learned policies. **UL 4600** covers autonomous products
goal-based. SOTIF's standard remedy is wrapping an unverifiable function in a verifiable
constraint layer, which is precisely what our solver is.

> **HYPOTHESIS, not a finding — "the safety filter for learned control".** As learned policies
> reach physical systems, the network cannot be certified, so the industry answer is a runtime
> constraint layer that provably keeps it inside a safe set — an optimisation solve at control
> rate with a bounded worst case, i.e. our workload at our one claim that survived every audit.
> Attraction: regulation creates non-optional demand. Against it: it is a 5–10 year thesis,
> certification is expensive, and the incumbent position is held by **lockstep MCUs** (AURIX
> safety islands), so it means fighting on their home ground. **Decided by one discovery question,
> not by analysis:** "when you put a learned policy on a real machine, how do you argue safety?
> Do you use a runtime constraint layer? Must you prove its worst-case execution time — to whom?"

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
  ~~The binding row is PWR-MPC every time~~ — **restated 2026-07-30 (§0.3): those margins were
  computed on the retired continuous-control-set PWR-MPC definition, which is now an off-set row.
  The re-specified FCS-MPC row survives host-issued, no-run-ahead at 6.5×.** The conclusion stands
  but on weaker evidence: the live examples are now the off-set PWR-CCS row and **one-step FCS-MPC
  at 4.8 datapath cycles per descriptor (0.61× of the ≥8 codegen rule)**, which fails the rule
  regardless of who issues. So D2 stays a wrapper rather than a redesign **provided the wrapper
  keeps the descriptor generator on the CM side**. This is also a MACIF constraint: the descriptor format is part of
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

> **⚠ THE PARAMETER LIST BELOW IS WRONG AND MUST BE REWRITTEN** (flagged 2026-07-31,
> [PHASE0_FINDINGS.md](PHASE0_FINDINGS.md) §8.9). It resolves the §9 "parameterization
> discipline" risk in the wrong direction by advertising four knobs of three different kinds as
> though they cost the same. **Parameters split into two classes:**
>
> - **REPLICATIVE** (a width or depth; reuses the same control FSMs; verification is a re-run at a
>   new elaboration): tile 8–16 square, vector lanes 8–32, scratchpad depth, bank word ≥64 bits,
>   frequency target. **Sell these as parameters.**
> - **STRUCTURAL** (changes control logic, protocol behaviour or a declared contract; each value
>   is its own coverage-closure target): bank count (**validate 8 and 16 only**), precision set
>   (MACIF declaration *plus* the runtime's refusal paths — each removal is a new negative-test
>   suite, not a smaller design), fp64 present/absent, descriptor queue depth (**fix at 8**),
>   pipeline depth. **Sell these as a SKU list** (§3.3), not as parameters.
>
> **`bus type` is the most expensive item in the list and the most casually marketed.** AXI4,
> ACE-Lite and TileLink each need their own protocol-checker verification and coherency proof.
> **It is not a parameter, it is three deliverables.** Do not print it in a parameter list again.

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
  > **SEQUENCE THIS AGAINST THE FILING.** This whitepaper is the *earliest* public disclosure of
  > all four candidate mechanisms and is currently unsequenced against any patent filing
  > ([IP_STRATEGY.md](business/IP_STRATEGY.md)). Europe has **no grace period** — EPC Art. 54 is absolute
  > novelty and Art. 55's six-month window covers only evident abuse and recognised
  > exhibitions, neither of which a publication is. Since the licensees we want are European,
  > losing EP means losing the patent where the design wins are. Reordering a document costs
  > nothing; the loss is permanent.

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
  ~~multi-core scaling~~, ASIL-ready package, bigger LLM support (KV-cache management, FP8),
  design-services revenue alongside licensing.

> **"Multi-core scaling" is STRUCK, 2026-07-31** ([PHASE0_FINDINGS.md](PHASE0_FINDINGS.md) §8.8).
> A single ADMM or interior-point solve cannot be usefully split: the iteration is loop-carried,
> and splitting *within* an iteration pays a barrier at each of the **17 measured chain steps**, so
> 2 CMs give **0.77×** with a tightly-coupled mailbox and **0.32×** through a coherent fabric. The
> one organisation that wins (shared scratchpad, 16-cycle barrier) is **unphysical on a
> bandwidth-bound row** — two datapaths behind one 128 B/cycle scratchpad starve — and where it is
> physical it *is* one wider CM, which is **4.9× cheaper per byte/cycle** than replication.
> FCS-MPC's 729 independent candidates *do* reach ~1.22× CPU parity on 2 CMs, and are still
> rejected: no deadline needs it (15.5× margin), the scarce resource is vector lanes (MACIF
> accumulator fusion buys them at ~zero area, lanes at +0.143 mm², a second CM at +0.786 mm²), and
> it doubles static power on the row whose whole pitch is energy.
>
> **What survives is independent instances with no interconnect** — for licensees wanting hard
> isolation between ASIL domains or control rates. That is already supported (instantiate the IP
> twice) and needs exactly one thing: **the runtime API must address multiple devices.** No shared
> scratchpad, no CM-to-CM coherency, no one-sequencer-many-datapaths. Keeping the line on the
> roadmap implies a verification obligation with no workload behind it.
>
> **Re-open if** a row appears that needs >288 B/cycle of fp32 GEMV to make a deadline, or >384 KB
> resident at >1 kHz. Neither exists today. Note the real Phase-4 growth axis is the **fp64 FMA
> count** (8 → 16 units = 1.85× on interior-point MPC for +0.144 mm²), which nothing in the frozen
> set needs yet.

> **Also for Phase 4 / §9 — bigger LLM support needs a bandwidth commitment, not a compute one.**
> SLM-DECODE (1.1B int8, 20 tok/s) misses at 6.4 GB/s and meets at **25.6 GB/s**, at which point it
> is **75% DRAM-bound and scores only 2.6× perf/W**. A CPU with the same bandwidth hits the same
> wall. The differentiated decode envelope at 6.4 GB/s / 20 tok/s is **~0.16 G int8 parameters**;
> above that, sell the CNN/ViT row (VIT-TINY: 0.17 GB/s, 28.2× perf/W), not the LLM row.

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

> ## ⛔ HOLD — do not make any repository public until the filing question is settled
>
> Added 2026-07-30. **Publication is irreversible and is prior art against our own later
> patent filings.** Apache-2.0 additionally grants patent rights in the contributed code. The
> mechanisms this plan intends to publish now exist in concrete form and are exactly the ones
> worth claiming: generation-tag call-to-call residency, the descriptor format plus
> multi-template hardware generator, the MACIF precision-declaration contract, and
> sequencer↔datapath convergence-loop control.
>
> Getting the order wrong cannot be undone — there is no way to un-publish prior art. Getting it
> right costs a provisional filing fee and some weeks of delay.
>
> **Gate:** every repo in the table below stays private until [IP_STRATEGY.md](business/IP_STRATEGY.md)
> records, per mechanism, either (a) a filing made or deliberately forgone, or (b) a finding that
> it is not novel and therefore free to publish. Note the live prior-art risk on the sequencer
> claim: the AURIX TC4x PPU is a licensed Synopsys ARC EV7x — scalar core + SIMD vector unit +
> closely-coupled memory with its own instruction stream — which may already read on it
> (PHASE0_FINDINGS §5.1).
>
> This hold does **not** block writing the code, the docs or the CI. It blocks flipping visibility.
>
> ### Publication checklist, added 2026-08-07 — ordered by irreversibility
>
> **1. Do NOT publish *this* repository.** Its 30+ commits contain the entire internal strategy:
> [CUSTOMER_DISCOVERY.md](business/CUSTOMER_DISCOVERY.md) (25 named targets, pre-registered falsifiers),
> [PHASE0_DECISION.md](business/PHASE0_DECISION.md) (kill criteria, conceded weaknesses),
> [IP_STRATEGY.md](business/IP_STRATEGY.md) (patent strategy and the PPU prior-art exposure), and every one
> of the 17 modelling errors. Publishing it publishes our competitive analysis, customer list and
> patent strategy. **Create a separate repo containing only `argmind/`, `tests/` and `examples/`,
> with fresh history — do not fork this one.** History cannot be retracted later: forks, mirrors
> and archives keep it.
>
> **2. `perf_model/results_*.txt` must not go out unmarked.** They are ±2× and non-quotable, and
> if published someone will quote them. Exclude them or carry an explicit notice per file.
>
> **3. Clear the trademark before creating the GitHub org.** Brand equity accumulates in the org
> name, PyPI name, docs, stars and citations; abandoning the name later forfeits all of it. The
> register check is 30 minutes ([IP_STRATEGY.md](business/IP_STRATEGY.md) §1) and has never been done.
>
> **4. File the provisional before `argmind-sim` publishes** — see the hold above. **MACIF is the
> exception and should publish early**: it is not novel (OpenCL's `cl_khr_fp64` has been optional,
> queryable and a compile-time error since 2008), and publishing pre-empts a CIM vendor patenting
> the boundary and charging us to cross it.
>
> **5. Contribution mechanics differ by repo.** DCO for the software repos; **a CLA for
> `argmind-lite` specifically**, because it is the only repo whose code could migrate into the paid
> proprietary core and Apache §5 gives inbound-equals-outbound but **no relicensing right**. If a
> CLA is too much friction for the university contributors that repo targets, the honest
> alternative is a public commitment in CONTRIBUTING.md never to upstream community RTL. One or
> the other must be chosen.
>
> **6. Avoid the NVDLA failure mode.** Nvidia published full RTL, the software languished, the
> community died. Keep **software as the centre of gravity** — which our build order already does,
> since the simulator and codegen work and no RTL exists yet.
>
> **7. Track the funnel, not stars:** pip installs → bitstream downloads → public `argmind-lite`
> instantiations → inbound licence conversations. And budget the support burden honestly: an open
> repo with users generates issues, which for a solo founder is real time competing with outreach.
>
> **Open question for counsel:** export control. Publicly available open-source software normally
> has exemptions, but semiconductor design IP can touch EAR. Notable evidence it is not
> theoretical: **RISC-V International relocated its incorporation to Switzerland** explicitly to
> reduce single-jurisdiction export-control exposure.

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
  **Solderpad-2.1 for open RTL** — but **not for the reason previously assumed**
  ([IP_STRATEGY.md](business/IP_STRATEGY.md) F-2): SHL-2.1 excludes Patents from its definition of
  "Rights", which reads protectively, yet it replaces **only Apache §2 and the Appendix**.
  **Apache §3 — the patent grant — survives, and a licensee may elect to treat the work as plain
  Apache-2.0.** So publishing `argmind-lite` grants a royalty-free patent licence to make and
  sell silicon containing it. Solderpad is still the right licence; it simply does not do the job
  we thought. Attorney question L-1. **Trademark stays
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
- **IP protection — REVISED 2026-07-30, all three claims named here were wrong**
  ([IP_STRATEGY.md](business/IP_STRATEGY.md) F-3). "Sequencer↔GEMM coupling / accelerator decides to
  iterate" is **anticipated three ways**: TI's C2000 CLA (independent programmable FP coprocessor
  with its own conditional branch `MBCNDD`, shipping since Piccolo), the AURIX PPU / Synopsys ARC
  EV7x, and an FPGA-MPC literature deep enough to have survey papers. Prior art need not perform
  well, only disclose the structure — so the PPU's unresolved FP/VCCM/coherency questions do not
  save it. "Small-matrix scheduling" and "deterministic-latency execution model" are both
  unpatentable as phrased. **What survives: one omnibus provisional on generation-tag
  call-to-call residency plus the sequencer *quantitative* contracts** (chain blocking ≥ one
  datapath width, run-ahead queue sized to the measured worst-case tail, ≥8 datapath cycles per
  descriptor), which appear nowhere in the searched literature. Trade secret for the rest.

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
| Verification cost explosion | Parameterization discipline (few validated configs, not full matrix); formal on control logic. **RESOLVED 2026-07-31 into a concrete rule** (§8.9, and the flag on §5): classify every parameter as **replicative** (a width/depth reusing the same control FSMs — sell as a parameter) or **structural** (changes control logic or a declared contract — sell as a SKU). **Two validated SKUs: CM-100 and CM-50** (§3.3). Bank count validated at 8 and 16 only; queue depth fixed at 8; `bus type` is three deliverables, not a parameter |
| One-person start | Phase 0/1 are deliberately model-and-paper heavy — solo-feasible; hire at RTL phase |
| **IMC/CIM erodes the inference half** of the pitch (§4.1) | MACIF makes a CIM macro adoptable rather than fatal; never market on inference TOPS/W; lead with solver latency + determinism |
| **Analog matrix computing matures for QP/LP** — would hit the solver thesis directly | Annual re-evaluation; today blocked by precision, PD-only stability, data-dependent latency. Watch PoliMi/Ielmini-line publications |
| System features (zero-copy, residency) are copyable by incumbents — not a physics moat | Patent **generation-tag residency and the sequencer quantitative contracts** — the sequencer *coupling* claim is dead prior art (IP_STRATEGY F-3). Compound the software/codegen ecosystem, where copying is slowest and where the real moat now sits |

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
