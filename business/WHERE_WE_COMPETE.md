# Where Argmind CM competes — one page

Consolidated 2026-08-07 from the Phase-0 evidence. **All ratios are modelled at the CM-100
configuration, ±2×, and are NOT quotable externally.** Two of the numbers below are measured and
labelled as such.

The point of this page is to be usable in a live conversation, including the part where we say
what we are *not* good at. Conceding credibly is what makes the rest believable.

---

## Competitive — the evidence supports these

### 1. Application-processor-class robotics: state estimation + MPC on a rich host

| benchmark | latency | perf/W |
|---|---|---|
| EKF-96 (SLAM-class estimation) | **2.1×** | 21.1× |
| QUAD-MPC (quadrotor, 1 kHz) | 1.8× | 12.3× |
| EKF-24 | 1.2× | 20.6× |

The only socket where **all four differentiators survive**: zero-copy and call-to-call residency
both apply (a rich host has DRAM and cache to eliminate — **measured 46× steady-state transfer
reduction**), determinism applies, and NPU IP cannot reach the workload at all.

That last part is structural, not a feature gap: **Ethos-U85 is INT8-weights-only and Ceva
NeuPro-M's FP32 rate is ~1% of its INT8 rate**, so incumbents cannot add solver support in
software — they lack the arithmetic.

Concretely: robot vacuums and lawn mowers (SLAM + planning, judged on battery life), legged and
mobile robots, drones with companion computers, AGV/AMR.

### 2. Consolidating several control loops onto one engine

A bill-of-materials and complexity argument, not a speed argument: one engine instead of several
MCUs/DSPs, one address space, one deterministic tick. **Capacity-bound, not throughput-bound**
— sizing it by duty cycle would oversell by 47× — and it requires **descriptor-granularity
preemption**, since one QUAD-MPC solve spans 9.4 periods of a 40 kHz loop.

---

## Competitive with a stated caveat

### 3. Interior-point nonlinear MPC (manipulators, trajectory optimization)

ARM-MPC is only **1.1× on latency** but **17.2× perf/W**. The real argument here is not
performance: **it needs fp64, and that is exactly where NPU IP is structurally excluded.** The
defensible claim is "the only licensable accelerator that can run interior point at all."

Caveat to state first: a CPU meets the deadline. Sell energy and core offload.

### 4. Vision inference with resident weights

VIT-TINY is **2.0× latency and 28.2× perf/W** — the highest perf/W in the whole set.

**But do not sell the efficiency.** Every ML perf/W figure we have divides a modelled CM energy
by a modelled *CPU* energy, and nobody runs edge CNN inference on a Cortex-A when the SoC has an
NPU. We have no measured NPU data and our own standard forbids comparing modelled figures to
vendor claims, so **the ML efficiency claim is unsupported, not merely narrow.**

What survives, because none of it needs an inference comparison: **arithmetic retargetability,
co-residency with the solver in one deterministic joint tick, and MACIF.** The pitch is "you do
not need a second engine," not "we are more efficient than an NPU."

The hybrid loop (learned policy → reference → MPC, one call) lives here and **needs 256 KB**:
policy + QUAD-MPC fits 192 KB only for the smaller of two real cited policies, at 1.3% margin.

---

## Not competitive — say so before a customer finds out

| | why |
|---|---|
| **High-rate motor drive / power electronics** (FCS-MPC, 40 kHz) | **0.62× — below parity.** We compute the tick 1.24× faster than an SME2 core, then pay 0.8 µs to be *asked*. Energy is still 22×, but latency loses. One-step FCS (0.61×) and 100 kHz two-level (0.69×) also **fail the codegen contract** — for those, tell the customer to use the MCU. |
| **Large-model on-device generative AI** | Misses the deadline at 6.4 GB/s, needs 22–25 GB/s, and even when feasible is 75% DRAM-bound at 2.4×. The **differentiated envelope tops out at 0.2–0.7 M parameters — an MLP, not an SLM.** It is the memory wall; no compute architecture wins it. |
| **Anything under ~0.4 µs of work per call** | The energy floor. Below it we do not clear the 5× kill criterion. |
| **Batch/throughput ML** | GPU/NPU territory, and in-memory computing beats everyone on stationary-weight low-precision TOPS/W. |

---

## The unresolved competitive risk

**Infineon AURIX TC4x PPU** is a licensed Synopsys ARC EV7x — a scalar core plus SIMD vector unit
plus closely-coupled memory running its own instruction stream. That is **a small version of our
own architecture, already automotive-qualified**, sitting in the socket we are targeting. Its FP
option, VCCM size and coherency are all *not published*, so we cannot assert those as gaps
either. This is discovery falsifier **F4** and a ~$300 eval board settles it — though the real
gate is the Synopsys MetaWare toolchain licence cost, not the board.

---

## The customer-facing test: "is my workload in the envelope?"

All seven must hold.

1. Is it an **iterative solve or matrix-matrix estimation with a hard deadline**?
2. Does one call do **≥ ~0.4 µs** of in-order-core work? *(else the energy claim fails)*
3. Is the problem dimension **≥ ~24**? *(else it is sequencer-issue-bound)*
4. Is the call rate **below ~100 kHz**? *(else invocation overhead dominates)*
5. Does the host have **DRAM and cache**? *(else zero-copy is void; invocation overhead still
   helps, at ~6.7× / 1.7 µs rather than 31× / 24 µs)*
6. Does the working set fit **~192–256 KB**? *(else DRAM enters the loop and determinism dies)*
7. Do they value **energy or a bounded worst case** above peak speed? *(if they need raw speed, a
   CPU is already sufficient — it meets 5 of 5 solver deadlines in our frozen set)*

---

## Two measured facts worth carrying

**A CPU meets every solver deadline we benchmark.** We are 1.0–2.1× faster and ~12–22× more
efficient. We are not enabling; we are efficient. Saying this first is what makes the efficiency
claim land.

**Software alone recovers only 1.8–3.0×** [MEASURED, x86 fp64]: a tuned C implementation of the
same ADMM beats OSQP by that much, and *nobody* reaches more than 9.3% of the machine's own SIMD
peak — because the workload is operand-fetch-bound, not compute-bound. **A GPU or NPU fails here
for the same reason.** So "just optimise the software" and "just use a GPU" are both bounded
answers, which is the space our energy and determinism claims occupy.
