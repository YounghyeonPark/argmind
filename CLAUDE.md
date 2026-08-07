# Argmind — project context

**A side project** (reframed 2026-08-07): open source, donation-funded at most, realistically
ending at the software and simulator layer rather than RTL and silicon. It was previously scoped
as a fabless IP startup; that framing is parked in [`business/`](business/) and flagged at the
top of [PLAN.md](PLAN.md). Entry point: [README.md](README.md).
Detail: [PHASE0_FINDINGS.md](PHASE0_FINDINGS.md) · [BENCHMARKS.md](BENCHMARKS.md) ·
[TESTING.md](TESTING.md) · [agent team](.claude/agents/README.md).

**What it is technically** — and this part is unchanged by the reframe. A **Computational Module
(CM)**: one engine for iterative solvers (ADMM/OSQP, interior-point, SQP, MPC, EKF) and small
resident-network inference. Mechanisms: a sequencer coupled to the datapath so convergence loops
run in-fabric with no host round-trip, plus zero-copy host memory and call-to-call residency.

**Working scope now**: `argmind/` (runtime API, simulator, descriptor v0.1, codegen),
`perf_model/` (architecture model and measurement harnesses), `cpu_backend/`, `tests/`.
**Out of scope**: RTL, verification, PPA, certification, customer discovery, licensing.
Do not propose those as next steps; if a finding implies one, record it and move on.

**Everything below is still load-bearing** — the technical conclusions do not depend on the
business model, and re-deriving them is expensive.

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
  NPU. **CM-100 baseline (re-costed 2026-07-30): 192 KB, 16 banks × 12 KiB = 128 B/cycle, 144
  MACs (12×12), 8 merged FMA units (= 16 fp32 lanes or 8 fp64), 800 MHz → 0.786 mm² at 16nm**
  (modelled, ±2×, and ±2.1× again on the FMA gate count → ~0.94 mm² at the pessimistic anchor).
  Minimum viable 64 B/cycle / 64 MACs; a solver-only SKU at 64 MACs is real
  (+35% energy, all on EKF) but **cannot drop fp64**.
- **Capacity was 128 KB and that was WRONG — corrected by measurement, not by modelling**
  (2026-07-30, descriptor v0.1 ruling R8; `argmind.descriptor.SCRATCHPAD_RULING`). The functional
  simulator's **measured** allocation plan for QUAD-MPC is **163.8 KiB at fp32** (138.8 KiB once
  `A = [I;Γ]` is exploited) against the 128 KB costed. Two model errors, **both flattering**:
  (1) P and L were charged n²/2 each — a **packed triangular layout that base+leading-dimension
  AGUs cannot express**, i.e. the capacity model assumed a storage layout the descriptor format
  cannot address; (2) working vectors and temporaries were charged **zero** (14.1 KiB at fp32),
  structurally invisible because the field is named `persistent_bytes` and a temporary is not
  persistent. **128 KB was arithmetically impossible, not merely tight: the flagship's three
  matrices alone are 125.0 KiB of 128.0 KiB.** Cost of the fix: core 0.698 → 0.786 mm² (+12.6%),
  pJ/byte 0.718 → 0.839, solver energy +5.5%, QUAD-MPC perf/W 14.13× → 12.55×. **Latency does not
  move** — capacity was never the binding constraint, bandwidth is, and **bandwidth and bank count
  are unchanged, so the bandwidth-first finding is untouched.** 192 not 160 KB because 12 KiB/bank
  (1536×64b, array efficiency 0.394) dominates 10 KiB/bank (0.354): the last 32 KiB costs
  0.020 mm². **Never size against QUAD-MPC at fp64 (327.6 KiB) — that is a validation artefact**
  (bit-exact OSQP match); fp64 is an interior-point requirement, not an ADMM one.
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
  **1.16–1.49× latency and 1.51–1.79× energy** on the ADMM rows — **those are OUR OWN numbers and
  they stand** — at ~2400× the operand error (8e-7 → 1.9e-3, ~0.2% of a commanded control action).
  **Not 2×, because the vector unit becomes co-binding once GEMV halves** — a bandwidth-bound row
  stops being bandwidth-bound before the doubling is spent. **The array's bf16 spec is wrong for the
  solver path; bf16 stays correct for ML.**
  **The "perf/W 14.1× → 25.3×" pairing is RETRACTED (2026-08-05, modelling error 16, §8.12):**
  `model_cpu` gave the CPU no credit for narrow operands, so we narrowed our fetch and not the
  adversary's. Corrected, the fp16-operand perf/W uplift is **1.18×, not 1.79×** (22.4× → 14.7× on
  that module's row) and fp16 operands make us *relatively slower* on latency vs a vector CPU
  (2.60× → 1.60×; PWR-CCS 1.26× → **0.93×, below parity**), because the CPU gains a full 2× on 90%
  of its GEMV while ours is vector-capped. **Adopt on the energy argument, never on a perf/W ratio.**
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
- Advantage is **energy + determinism, not speed**: **12.3–22.0× perf/W at the CM-100 config
  (modelled, ±2×)**, latency **0.62–2.15× on solvers** — QUAD 1.81×, EKF-96 2.15×, ARM 1.14×,
  EKF-24 1.21×, and **PWR-MPC 0.62×, genuinely below parity**. **A CPU meets 5 of 5 solver
  deadlines in the frozen set** — never pitch "the CPU can't do this"; we are more efficient, not
  enabling. All five stay above the 5× kill criterion.
  (History: 17–25×/1.2–2.9×, then 9.5–21.3×, then 14–22×, then 14–24×. The 14–24× → 12.3–22.0×
  move is the 192 KB capacity correction plus charging ARM-MPC's real footprint.
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
- **Descriptor format v0.1 is the current spec** (`argmind/descriptor.py`; `V0_1_RULINGS` is
  authoritative, `V0_LIMITATIONS` is the paired v0 record). Still **8 words / 32 B**, three base
  addresses, packed dims — so **descriptor traffic, queue SRAM and AGU count are unchanged, which
  is what made the change affordable.** Layout: opcode 6b, prec_m 3b, prec_v 3b, transpose 2b,
  structure 3b, accumulate 1b, unit_diag 1b, chain_token 3b, wait_mask 8b, reserved 2b.
  **The whole change is paid for by reclaiming waste in v0**, which had *zero* spare bits:
  chain_token 8b→3b (−5) and prec_m 4b→3b (−1) fund prec_v (+3) and structure (+1), leaving 2 bits
  reserved. Rulings: **NRMI and VMUL added** (opcode space was never scarce — 10 of 64 used);
  **wait token → wait mask**; **prec_v** makes "fp16 operands, fp32 solve" expressible;
  **PROJ becomes out-of-place** via interleaved (lo,hi) bound pairs instead of a fourth AGU;
  **elementwise ops go 2-D** (n-cap 255 → 65535, zero opcodes, zero bits); **zero AGU stride is
  legal on reads, illegal on writes.**
- **Two of the seven v0 findings were argued on grounds that do not hold. Do not repeat either.**
  (1) "Without NRMI the convergence test leaves the fabric" is **false** — the 2-norm path is DOT
  plus two sqrt on the FMA cluster plus compares on the sequencer, no host round-trip. NRMI is
  justified by **equivalence with the reference solver** (3.7× better control action, an iteration
  count a customer can check against OSQP) at essentially zero cost, which is enough. (2) "Scalar-only
  ρ rules out the sparse MPC form acados and HPIPM use" **conflates two solvers**: ρ is ADMM's
  penalty parameter and **HPIPM is interior-point, with no ρ at all.** v0 never blocked ARM-MPC —
  its per-stage n_b×n_b blocks are exactly what three base addresses express, and it is the case
  that gets 0.04 cycles/descriptor from the generator. VMUL enables **sparse-form ADMM**; claim that
  and nothing more.
- **The wait mask is worth −5 bits, NOT 1.42×. Adopt it for the bits.** The v0 depth penalties
  (iteration 17 vs 12, residual 15 vs 5) are real but they count **chain steps, and a chain step
  here is ~228 datapath cycles.** Measured on the emitted DAG the mask removes **120 of 7881 cycles
  per checked QUAD-MPC iteration = 1.5%** — the depth-ratio metric overstates the value ~100×. It
  pays where cycles/descriptor approaches `drain_cycles`=8: total drain is 10.6% of ARM-MPC latency
  at 30.3 cycles/descriptor, and **the reducible share of that is NOT measured** — no HPIPM
  descriptor program exists yet. Related metric bug found and fixed: `chain_depth` was computing
  1 + *ancestor count*, not longest path; those coincide only for a fully serialised block.
- **A token is a scoreboard slot, not a name — but reuse is only safe under an extra rule.**
  Measured peak simultaneous liveness on the real ADMM DAG is **7 (iteration), 3 (residual),
  3 (setup)**, so 8 tokens with one spare, and v0's 255 was waste. Reuse is safe **only if a
  descriptor that reclaims a token also waits on it**; without that, a waiter resolving a token to a
  later owner gets no guarantee about the earlier one, because in-order *issue* does not imply
  in-order *completion*. That is an **RTL scoreboard and codegen requirement**, it was found by the
  concatenation hazard test rather than by reasoning, and with it token exhaustion costs overlap and
  cannot produce a wrong answer. v0's 255-descriptor `concat` limit is gone.
- **The emit cost derived, not guessed** (`perf_model/seq_emit_cost.py`): RV32 emit counted at
  **23 cycles naive, 10 delta-encoded, ~0 with the hardware generator** — so `seq_setup_cycles=4`
  was never reachable in software and silently presumed the generator. Drop delta encoding.
  **Arming costs ~22 instructions *per template*, so a template must cover ≥6 descriptors.** An
  HPIPM stage sweep gets 0.04 cycles/desc; an ADMM iteration gets 8.25. Fix is free and
  codegen-side: **arm templates once per solve, not per iteration** (legal because allocation is
  static — the same property that makes overlap legal), so **the generator must hold ≥6 armed
  templates.**
- **The measured descriptor count is 31 per checked ADMM iteration, not 28, and the ≥8-cycle
  contract is NOT threatened** (`argmind.descriptor.SEQUENCER_RECHECK`). At fp32 / 128 B per cycle
  the block averages **238 cycles/descriptor, 29.8× the bar** (fp64: 474, 59.3×). The convergence
  test is 48% of descriptors *and 49% of cycles* — its mix is dominated by two full A-products, not
  by the seven cheap reductions, so the scary fraction is not a problem on average.
  **Where it IS thin is the queue-depth clause:** the residual block holds a run of **six
  consecutive descriptors averaging 5.8 cycles**, below the bar. Six fits an 8-deep queue, so
  run-ahead absorbs it — this **confirms queue depth 8 as a floor by measurement** (a 4-deep queue
  would not) and it is the thinnest point in the whole sequencer analysis. Fix is a MACIF
  capability, not silicon: **`maxabs_accumulate` (MACIF 0.2)** folds the ∞-norm into the pass that
  produces the vector, taking the residual block 15 → 10 descriptors and the run 6 → 4. Same
  principle as `broadcast_init`/`square_accumulate` — fold work into the accumulator at O(1) area.
- **`check_term=25` was wrong and flattering.** OSQP's library default never fires on a solve that
  converges in 6–14 iterations, so residual work was being amortised over checks that would not
  happen. Corrected to 1, matching the validated simulator. Cost: QUAD-MPC latency **36.0 → 49.0 µs
  (+36%)**, energy 6.15 → 8.90 µJ; perf/W barely moves (the CPU pays the same extra work). The
  payoff is a **cross-validation**: at `check_term=1` the model's kernel decomposition emits
  **31.0 descriptors against the simulator's measured 31**, from two independently written code
  paths. It did not agree at 25.
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

## Scalability — the band is narrow, and that is the honest position (2026-07-31, §8)

- **Only two axes move a benchmark by >3×:** scratchpad bandwidth (64→256 B/cycle, **3.87×** on
  QUAD-MPC worst case) and the fp64 FMA cluster (4→16 units, **3.76×** on ARM-MPC). Over
  defensible ranges: frequency 1.64×, capacity **1.29× as a *cost***, MAC array **1.00×**.
  A tripwire test fails if a third axis ever clears 3×.
- **The balance rule, which refines the settled MAC finding:** min tile = **√(bw/2)** at fp32. A
  12×12 array absorbs GEMV up to **288 B/cycle** — the same 288 at fp32, bf16 *and* int8, because
  this datapath's rate multiplier is ~1/element-size. So *"16× more MACs bought zero" is a
  statement about 64–128 B/cycle, not about arrays in general*: at 256 B/cycle a tile of 8 already
  binds. CM-100's 2.25× MAC headroom is exactly what makes 256 B/cycle a drop-in upgrade.
- **Capacity is not a scaling axis.** Solver energy is *monotone increasing* 32 KB → 1024 KB.
  Making ARM-MPC fit at 384 KB costs +1.11 µJ and +0.228 mm² for **zero** latency benefit.
- **Frequency's arrow points DOWN.** 600 MHz meets every frozen worst case inside the
  50%-of-period budget at **0.84× the solver energy and identical area**. Not re-spec'd (400 MHz
  leaves only 1.08× margin on the one *measured* tail, and the f–V table is [EST]) — but sell
  frequency as a downward energy knob with a deadline floor, not as headroom.
- **Multi-CM is struck from Phase 4.** A loop-carried solve cannot be split: 2 CMs give **0.77×**
  (mailbox) or **0.32×** (coherent) on ADMM's 17 measured chain steps. FCS-MPC's 729 candidates
  *do* reach ~1.22×, but that row has 15.5× deadline margin, its scarce resource is vector lanes
  (MACIF accumulator fusion ≈ free < +0.143 mm² lanes < +0.786 mm² CM), and a second CM doubles
  static power on the row sold on energy. **Buying bandwidth inside one CM is 4.9× cheaper per
  byte/cycle.** Consolidation across loops is **capacity-bound, not throughput-bound** — sizing it
  by duty cycle would oversell 47× — and needs **descriptor-granularity preemption** (a QUAD-MPC
  solve is 9.4 periods of a 40 kHz loop; preemption drops blocking to 2.0 µs).
- **SKU family is TWO, not three:** **CM-100** (144 MACs / 192 KB / 128 B/cycle, 0.786 mm²) and
  **CM-50** (64 / 64 KB / 64 B/cycle, 0.424 mm²), where CM-50 explicitly **does not support the
  QUAD-MPC class**. Rejected: a 16-MAC solver-only part (18% saving for 2.68× the EKF-96 energy —
  and EKF is a beachhead workload) and CM-200 (a real Pareto point with no workload asking for it).
  **fp64 up to lanes/2 is free** — the merged cluster is sized `max(fp64, lanes/2)`, so interior
  point is a one-off +0.067 mm² merge premium, not a running cost.
- §5-vs-§9 resolved by classifying parameters **replicative** (sell as parameters) vs **structural**
  (sell as a short SKU list). **`bus type` is not a parameter — it is three deliverables.**
- **The energy claim has a floor: ~0.4 µs of in-order-core work to reach 5× perf/W.** Two-level
  FCS-MPC at Np=1 scores **3.2×, below the kill criterion**. "We are always more efficient" is
  false — say so before a customer finds it.
- **Large-model ML is outside the *differentiated* envelope even where it is feasible.**
  SLM-DECODE meets its deadline at 25.6 GB/s but is 76% DRAM-bound at 2.6× perf/W. At 6.4 GB/s and
  20 tok/s the envelope is ~**0.16 G int8 parameters** (50%-of-period budget).
- **The hardware descriptor generator is NOT deadline-load-bearing** — setup is free up to 16
  cycles/descriptor and the naive 23-cycle RV32 emit costs 4%. Keep it as cheap insurance; **stop
  describing it as required** (this corrects the earlier "precondition for the modelled numbers").
- Two more modelling errors, **13 and 14**, both of which made an axis look free because the model
  could not express a cost: frequency had **no supply-voltage term** (2 GHz read as 2.5× speed at
  *lower* energy per call; corrected, 800→1600 MHz is 1.90× latency at **1.547×** energy), and the
  longest single-cycle bitline was a **hard constant** (a 12 KiB×64b bank needs a second fold above
  ~1100 MHz: **+46% scratchpad area**). Both are exactly neutral at the nominal 800 MHz, verified
  by regenerating every pre-existing `results_*.txt` byte-identical.

## On-device AI — two boundaries, two resources (2026-08-05, §8.12, `perf_model/ml_envelope.py`)

- **THE ML INFERENCE-EFFICIENCY CLAIM IS UNSUPPORTED, NOT MERELY NARROW.** Every ML perf/W figure
  we have divides a modelled CM energy by a modelled **CPU** energy, and nobody runs edge CNN
  inference on a Cortex-A when the SoC has an NPU. VIT-TINY's **28.2×** (not 30.4× — use the
  recorded value) is correct against a baseline no engineer would choose. We cannot fix it: no
  measured NPU energy at a stated node/precision exists, and our own standard forbids comparing a
  modelled figure to a vendor claim. **Never quote an ML perf/W number.** Three ML-adjacent claims
  survive because none needs an inference comparison: **arithmetic retargetability** (from the
  rival's datasheet), **co-residency + one joint deterministic tick**, and **MACIF**.
- **FEASIBILITY is set by external bandwidth; DIFFERENTIATION is set by residency.** Over a
  6.4→51.2 GB/s sweep the feasible model size moves **8.0×** (0.32 → 2.56 G int8 at 20 tok/s,
  `params_max = BW_ext/(tok_s·bytes_per_weight)`) while the ≥5× perf/W boundary moves **1.21×**
  (0.00021 → 0.00025 G) and is **flat at int4** (0.00066 G). So buying bandwidth buys feasibility
  and buys ~nothing of differentiation. **The feasible-but-undifferentiated zone is ~1000–1500×
  wide in model size and contains every model anyone would call a language model**; the
  differentiated zone tops out at **0.2–0.7 M parameters, which is an MLP, not an SLM.**
- **The invariant is external arithmetic intensity: ≥~2.6 MAC per byte crossing the CM boundary.**
  One number orders the whole set — batch-1 int8 decode **1.00** (by definition), int4 decode 1.93,
  ViT-Tiny 223, resident MLP policy 258–989, QUAD-MPC 1022. Derivation: at batch 1,
  `compute/transfer = BW_ext/BW_spad`, **independent of size and precision**, and BW_spad is
  102.4 GB/s. **int4 does not change which resource binds** — it halves both sides.
  **Honest weakness: int4 decode is only 1.35× under the threshold and intensity is LINEAR in batch,
  so batch 2 crosses it.** The model cannot express a batched decode. Always say **batch-1 decode**.
- **int4 is in the model, weights-only (GPTQ/AWQ-class), and buys BANDWIDTH ONLY.**
  `macs_per_cycle(int4) == macs_per_cycle(int8)`: a digital multiplier is sized by its wider
  operand and the product is 4b×8b. Full 12 pJ/B still charged on streamed weights (83% of the
  row). **Accuracy is not modelled — an int4 result is never an endorsement of int4 accuracy.**
  **int4 is NOT in descriptor v0.1 or `Format`, deliberately:** two elements share a byte, so row
  length is even and leading dimension counts nibbles, which byte-addressing AGUs cannot express.
  MACIF: substituting int8 for int4 is a **deadline miss**; int4 for int8 is a **wrong answer**;
  refuse both directions.
  **int4 AMENDS THE BALANCE RULE — the one architectural consequence of adding it.** The array knee
  is `macs_per_cycle × bytes_per_elem` = 288 B/cycle at fp32/bf16/int8 but **144 B/cycle at int4**,
  because int4's rate multiplier is int8's while its element halves. 128 B/cycle is still under it,
  but **256 B/cycle is NOT a drop-in upgrade at int4: it wants a 16×16 tile, not 12×12.**
- **There is NO resident-weight ML SKU, killed twice over.** 4 MB holds 8.39 M int4 params — ~80×
  short of what a 6.4 GB/s stream already delivers — at **6.31× core area and 2.17× solver energy**.
  And the network population is **bimodal**: MLP policies at 25–200 KB or backbones at 3–6 MB+.
  Nothing wants 384 KB–1 MB *except* co-residency with the solver.
- **The hybrid loop (learned policy → reference → MPC, one call) SURVIVES, but not as an ML claim
  and not at 192 KB.** Policy + QUAD-MPC (138.8 KiB [MEAS]) fits 192 KB only for the smaller cited
  policy (Hwangbo 256-128-32, 49,964 params → 189.5 KiB, **1.3% margin, and only because codegen
  exploits A=[I;Γ]**; at the dense-A plan it overflows). The widely used `legged_gym` shape (Rudin
  512-256-128, 190,860 params) needs **256 KB at int4 or 384 KB at int8** — **state 256 KB**.
  End-to-end vs an NPU+CPU split with the **NPU charged ZERO**: 2.2–2.3× latency, **9.6–12.2×
  energy**. But **the network is 1.1–4.0% of the tick, so the hybrid loop's advantage IS the
  solver's advantage** — it removes a reason to buy a second engine, not a new efficiency claim.
  Both sides meet 1 kHz. Do not sell it on launch overhead (2.5% of the period). **Sell it on one
  engine / one address space / one deterministic joint tick.**
  **It only closes when the network runs at the CONTROL rate, which forces an MLP:** ViT+QUAD in a
  1 kHz tick misses **8.5× on COMPUTE** (array-bound GEMM), not on capacity — perception at 30 Hz
  beside control at 1 kHz is the consolidation-across-loops question, needing preemption.
- **This is the first workload asking for capacity for a reason other than making one solver fit.**
  It does not overturn "capacity is a cost, not an axis" — it supplies the demand side and pays
  §8.3's price (256 KB: +6% area, +3.2% solver energy, zero solver latency benefit).

## Built and validated (2026-07-30) — `argmind/` runtime API + functional simulator

ADMM runs end-to-end as a descriptor sequence and **matches reference OSQP exactly** (22/22 and
30/30 iterations, max|Δx| = 9.2e-15, with an active set present); control action within 4.2e-6 of
a polished eps=1e-11 solve; closed-loop p50 7 / p95 8 / max 9 iterations, inside the measured
6–14 band. **Residency demonstrated, not modelled:** 268 KB on the first call, then
**5.8 KB in / 0.6 KB out per tick (46×)**, with the Cholesky running on 1 of 200 ticks.

Implementation against the *frozen* spec found seven gaps. **All seven are ruled on in descriptor
v0.1** (above; `argmind.descriptor.V0_1_RULINGS`). The two that change how we work:

- **ρ must be a power of two — but the root cause is NOT ρ.** Descriptor word 7 is fp32 and ADMM
  multiplies by both ρ and 1/ρ; the fixed point requires them to be exact reciprocals, since with an
  active constraint `y = ρ·(1/ρ)·y` forces `y = 0` otherwise and the dual never settles.
  **ρ=0.1 (OSQP's default) stalls at 1500 iterations; ρ=0.125 converges in 34.** Re-diagnosed
  (ruling R5): the failure is **an fp32 immediate multiplying fp64 data**, not ρ. OSQP runs happily
  at ρ=0.1 because it keeps ρ and ρ⁻¹ as fp64 **vectors**, where the mismatch is ~1e-16 rather than
  1.5e-8. So the fix is not a wider w7 — that would cost a word and still not be exact — it is:
  **any scalar that must be exact relative to another scalar lives in scratchpad, at the buffer's
  own precision, and is read by a zero-stride VMUL.** w7 keeps standalone tuning constants.
  Power-of-two ρ stays *recommended* (free, and makes the mismatch exactly zero) but stops being
  load-bearing. `ADMMSolver` still refuses non-power-of-two ρ on the w7 path.
- **QUAD-MPC cannot validate the projection kernel or the dual update.** At its nominal ±2 N
  bounds the optimum is *interior* — OSQP reports no active set on all 200 closed-loop ticks — so
  y* = 0 and a broken dual fixed point is satisfied trivially. A genuinely wrong dual update
  (`y += ρ(t−z)` instead of `y = ρ(t−z)`) matched OSQP to 2.9e-15 across 60 solves. **Always run a
  tight-bound variant with a real active set alongside it.**

**Both open items are now closed.** The seven descriptor gaps are ruled on in v0.1, and the
scratchpad contradiction (QUAD-MPC 163.8 KiB at fp32 against a costed 128 KiB) is resolved by
growing to 192 KB — see the two capacity bullets above. The measurement was authoritative over the
model, and it exposed two flattering model errors rather than one implementation mistake.
**Note the format change is not a capacity remedy:** v0.1 moves the measured footprint by 0.2%
(167736 → 167440 B, from deleting the σ·ones(n) vector). The capacity ruling stands on its own.

## MEASURED 2026-08-07 — CPU software headroom (`cpu_backend/`)

Three implementations of the *same* reduced-form ADMM, same problem (coupled QUAD-MPC,
tight bounds, active set present), same 22 iterations, x86 i9-13900HX, single core, fp64:

| | best µs | GFLOP/s | % of machine peak | vs OSQP |
|---|---|---|---|---|
| OSQP (solve only, sparse LDLᵀ) | 283.0 | n/a | n/a | 1.00× |
| numpy / OpenBLAS (dense) | 1201.3 | 1.64 | 0.7% | **0.24×** |
| tuned C, strict IEEE | 154.1 | 12.77 | 5.7% | 1.84× |
| tuned C, `-ffast-math` | 93.6 | 21.02 | 9.3% | **3.02×** |

- **Software headroom is real but bounded: 1.8–3.0×.** Worth having, not transformative.
- **numpy/OpenBLAS is 12.8× SLOWER than a plain C loop** — at n=80 a BLAS call's dispatch
  overhead exceeds the arithmetic it performs. Library ≠ fast at solver scale.
- **The model's CPU baseline is ~3× OPTIMISTIC.** It assumes 25–35% of SIMD peak; the best
  tuned implementation reaches **9.3%**. Direction matters: correcting this makes our
  accelerator look *better*, so it must be re-measured on **Arm at fp32** before being banked
  — x86/fp64 does not transfer.
- **Why nobody reaches peak confirms the central finding, now measured on a CPU rather than
  modelled on our datapath**: `Γᵀv` and `Γx` read every element exactly once with no reuse,
  and the two triangular solves are loop-carried and cannot vectorise across rows. **A GPU or
  NPU fails here for the same reason** — so "SIMD is weak next to a GPU/NPU" does not hold on
  this workload; all three sit in the same operand-fetch regime and only the CPU gets the
  control plane for free.
- Limits: cache-resident working set (~250 KB, favours the CPU), single core, fp64, and it
  measures **time, not energy** — so the perf/W and determinism claims are untouched. Same
  silicon at the same voltage cannot yield a structural energy win.

## Evidence standards (non-negotiable — one bad number costs the company)

- Efficiency figures always carry **process node + precision**.
- Label **measured silicon vs. vendor estimate**; benchmark only against measured.
- Model outputs are **±2× and never quotable externally** — say so in any doc containing them.
- **"Not published"** is a valid, preferred finding; never present an estimate as a
  competitor's spec.
- **Models must be able to fail**: a flattering result means find the assumption first. **Seventeen
  modelling errors so far**, most of them flattering — undercounted overhead energy, GEMV modelled
  as GEMM, a phantom transcendental term that alone produced the entire apparent latency win,
  scratchpad energy 2.3× low, MAC count conflated with operand-reuse depth, an unphysical vector
  gate count, the three found 2026-07-30 by building the simulator (**a working set charged at a
  packed triangular layout the AGU cannot address, working vectors charged zero bytes, and
  convergence checked every 25th iteration on a solve that takes 6–14**), and the two found
  2026-08-05 by the on-device-AI audit: **16, the CPU baseline got no credit for narrow matrix
  operands** (we narrowed our fetch and not the adversary's — retracts the fp16 perf/W pairing) and
  **17, the SLM row's KV cache was never counted although the comment said it was**. All are encoded
  as regression tests; run them.
- **The strongest error-finding tool so far was building the thing.** Three of the seventeen came
  from one functional simulator, and the two capacity errors had survived every review because they
  were *inside a plausible formula* rather than in a stated assumption. Corollary: **an
  implementation measurement outranks the model**, and when they disagree the model is where to look
  first. **Errors 16 and 17 add a second pattern: a COMMENT that claims a term the expression does
  not contain, and a SWEPT AXIS that moves only one side of a comparison.** Both are cheap to check
  and neither shows up in a review of the conclusion.
- **Scrutinise corrections that favour us as hard as ones that don't.** The fp64 pass found a CPU
  baseline charging fp64 at fp32 rates — fixing it moved ARM-MPC perf/W 14.1× → 20.2×, entirely
  on the CPU side. Two errors had been partly cancelling.
- Cite URLs for external claims.

## Conventions

Windows/PowerShell; python 3.14; **git repo since 2026-07-30** — commit before and after any
change to `perf_model/`, and run `python -m pytest` (429 invariant + mutation tests) before
quoting any modelled number. Markdown docs at root, model in `perf_model/`.
Prefer open-source EDA (Verilator, cocotb, Yosys) — the cash plan depends on it. Interfaces:
AXI4/AXI4-Lite baseline; ACE-Lite (Arm) and coherent TileLink (RISC-V) for zero-copy; CHI +
SMMU/SVM deferred to Phase 4. Apple is not an integration target — they don't license
third-party accelerator IP; AMBA compliance is the only "Apple story."
