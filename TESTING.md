# Test environment — state, strategy, and gaps

Surveyed and set up 2026-07-29. Run everything with `python -m pytest` from the repo root.

## What exists now

| | Version | Role |
|---|---|---|
| pytest | 9.1.1 | test runner |
| numpy / scipy | 2.4.6 / 1.18.0 | model + harness |
| osqp | 1.1.3 | reference solver for measured baselines |
| python | 3.14.5 | — |
| gcc | MinGW-w64 | present, unused so far |

**429 tests, all passing**, in seventeen files. (The per-file counts below are maintained by hand
and drift; `python -m pytest -q` is authoritative. The six `test_argmind_*.py` files cover the
runtime API, descriptor format, kernels, memory, MACIF and program builder and are documented in
[argmind/](argmind/) rather than here.)

- [tests/test_model_invariants.py](tests/test_model_invariants.py) — 41 tests. One regression
  test per historical modelling error, plus physical invariants.
- [tests/test_scaling_study.py](tests/test_scaling_study.py) — 29 tests covering the scaling
  analysis (2026-07-31). Two are regression tests for **modelling errors 13 and 14**, both of
  which made an axis look free: the frequency axis had no supply-voltage term (so 2 GHz was 2.5×
  the speed at *lower* energy per call) and the longest single-cycle bitline was a hard constant
  (so a faster clock cost no scratchpad area). Both new terms are asserted **exactly neutral at
  the nominal 800 MHz**, which is what let every pre-existing `results_*.txt` regenerate
  byte-identical. The rest pin the scaling conclusions as directions, and **five of them exist to
  protect the ability to report a negative**: faster must cost energy, more capacity must cost
  energy, two CMs must be *slower* than one on a loop-carried solve, the scale-down sweep must
  contain losses as well as wins (and must report an *infinite* offload floor when the datapath
  ratio k ≤ 1), and CM-50 must **not** hold the flagship working set. One test is a tripwire
  rather than an invariant: `test_only_two_axes_scale_by_more_than_3x` fails if a third axis ever
  clears 3×, which would mean the SKU family should be revisited.
- [tests/test_fp64_datapath.py](tests/test_fp64_datapath.py) — 28 tests covering the fp64 datapath
  costing: FMA gate model vs its published anchors, fp64 throughput decoupled from the MAC array,
  fp64 area nonzero and monotone, multi-pass scaling, and the refinement crossover.
- [tests/test_sequencer.py](tests/test_sequencer.py) — 50 tests covering the descriptor-issue
  model: every benchmark's kernel decomposition must sum to its declared MAC/vec/trans counts and
  must reproduce the main model's datapath cycle count; the retired flat `scalar_ops_per_iter`
  must no longer reach the CM path; monotonicity in setup cost, run-ahead and chain-block width;
  dependency drain invariant to scratchpad bandwidth; and the findings themselves pinned as
  relationships (block-structured IP issues >10× the descriptors of ADMM; exposure ordered by
  cycles/descriptor; faster precision is *more* exposed).
- [tests/test_sram_and_harness.py](tests/test_sram_and_harness.py) — 10 tests covering the
  SRAM cost model and the OSQP harness (see "Not yet covered" below, now largely closed).
- [tests/test_pwr_mpc_respec.py](tests/test_pwr_mpc_respec.py) — 29 tests covering the PWR-MPC
  re-specification (2026-07-30): the FCS-MPC benchmark and its arithmetic intensity (which must stay
  set by n_y, not by the tile, or the row's binding resource is reported wrongly); the GEMM
  bandwidth ceiling (must bind on PWR-MPC and on nothing else); mixed-precision operands
  (`solve_prec_frac` must equal the chained ops' share, and fp16 must help by <2×); the MCU-class
  host tiers (T1M cheaper than T1, the tier advantage must *collapse* on MCU class, and the
  surviving advantage must be majority invocation overhead); and the CPU elementwise-rate
  correction. **Two tests exist to protect NEGATIVE results**: the re-specified row must stay below
  parity, and one-step FCS-MPC must stay a CPU win — a model that quietly optimises those away has
  lost the ability to report a minimum viable problem size.
- [tests/test_ml_envelope.py](tests/test_ml_envelope.py) — 31 tests covering the on-device-AI
  envelope (2026-08-05, PHASE0_FINDINGS §8.12). Two are regression tests for **modelling errors 16
  and 17**, both flattering: the CPU baseline got no credit for narrow matrix operands (so a
  mixed-precision row narrowed *our* fetch and not the adversary's) and the SLM row's KV cache was
  never counted although the code comment claimed it was. Three guard the new precision from
  becoming a free win — int4 must **not** raise the array rate, must still pay 12 pJ/B on every
  streamed byte, and must not become MAC-energy-sensitive. One guards baseline fairness on the new
  swept axis: a DRAM-bound CPU row must get slower when external bandwidth is cut, or `model_cpu`
  has re-acquired a private bandwidth. **Four protect NEGATIVE results**:
  `test_large_model_llm_decode_is_never_differentiated` (parametrised over int8/int4 × 8 sizes ×
  4 bandwidths, and its failure message names the recorded finding it would contradict),
  `test_a_perception_network_cannot_be_in_a_1khz_tick`, `test_no_plausible_scratchpad_makes_an_llm_resident`,
  and `test_the_network_is_a_small_fraction_of_the_hybrid_tick` (the deflation guard: if the policy
  ever exceeds 25% of the tick, "the hybrid loop's advantage is the solver's advantage" stops
  holding). Two are honesty guards rather than physics: the zone label must keep its `DIFF*`
  asterisk, and no NPU stand-in may appear among the baselines.
- [tests/test_guards_actually_bite.py](tests/test_guards_actually_bite.py) — 10 meta-tests that
  re-introduce real past bugs at runtime and assert the matching guard *fails*.

## Test strategy: pin relationships, not numbers

Every number in the model is a ±2× estimate that we *expect* to refine. Pinning outputs would
freeze today's estimates and make each legitimate improvement look like a regression. So the
suite pins **relationships and directions** instead:

- More bandwidth never increases latency; more MACs never increases latency.
- GEMV-bound latency is insensitive to MAC count (the settled architecture finding, encoded so
  drift becomes visible).
- Latency decomposes exactly into compute + transfer + overhead.
- Zero-copy is never slower than copy-based.
- perf/W stays inside a believable band (1–100×) — the guard that would have caught the original
  absurd 128× result.
- The model **can still report failure**: there is a test asserting some benchmark misses its
  deadline, and another asserting the CPU wins somewhere. Two more were added 2026-07-30 to protect
  specific negative results from being optimised away — **the re-specified PWR-MPC must stay below
  parity (0.62×), and one-step FCS-MPC must stay a CPU win (0.12×)** — because those two are what
  keep the model able to report a minimum viable problem size on the motor-drive socket. A model that cannot produce a negative
  result is not evidence, and CLAUDE.md's rule to that effect is now mechanically enforced
  rather than merely stated.

### Why the meta-tests matter

Phase 0 has produced **seventeen** modelling errors, each caught by manual or agent review —
expensive and non-repeatable. The suite encodes all seventeen. But a regression suite that has
never failed has not been shown to catch anything, so `test_guards_actually_bite.py` mutates the
model back to each buggy formulation and asserts the guard trips. Confirmed live:

- Misclassifying GEMV as GEMM makes a 16× larger array look 5%+ faster → caught.
- Restoring the phantom transcendental term inflates apparent speedup by >25% → caught.
- Removing overhead energy inflates perf/W by >30% → caught.
- Zeroing descriptor setup cost and dependency drain makes **QUAD-MPC's** 50%-budget miss disappear,
  i.e. the sequencer terms are load-bearing → caught. *(Re-pointed 2026-07-30: this used PWR-MPC
  until that row was re-specified as FCS-MPC, which has no dependency chain and so cannot
  demonstrate the drain term at all. QUAD-MPC can, but only with both run-ahead and full-width
  chain blocking absent — which is itself the downgraded finding, recorded in §0.3.)*
- Inflating one kernel op's MAC count breaks the descriptor/MAC consistency assertion → caught.
- Dropping the SLM row's KV cache (error 17) changes its external traffic *and* its latency → caught.
- Pinning the CPU baseline to a private DRAM bandwidth makes it insensitive to the swept axis
  (error 16's sibling) → caught.
- Giving int4 a 4× array rate is **NOT** caught by any benchmark row at 128 B/cycle, and the
  meta-test asserts that: an int4 GEMV is bandwidth-bound there, so the array rate is
  unobservable and only the direct `macs_per_cycle` invariant protects it. The same mutation *is*
  observable at 256 B/cycle, which is where the int4 array knee (144 B/cycle) has been crossed.
  **Written expecting a row-level catch and it failed; the failure was the finding.**

If a test in that file ever starts passing *trivially*, the corresponding guard has gone slack.

## Gap 1 — no version control — **CLOSED 2026-07-30**

`git init` done, baseline and subsequent work committed. Recording why it mattered, because the
failure mode is the one the repo exists to prevent: three agents edited `perf_model/` and the
Markdown docs concurrently on 2026-07-29 with no way to diff, attribute, revert or bisect any of
it — so when a benchmark number moved, nobody could say what moved it. It also blocked
[PLAN.md](PLAN.md) §7.1, which commits to public repositories with CI on every PR.

`results_*.txt` are tracked on purpose: they are the evidence behind claims in
PHASE0_FINDINGS.md, so versioning them is what makes a changed number visible and attributable.

Caveat: the repo lives inside a Google Drive folder, and Drive sync can corrupt `.git` if it
touches objects mid-write. Unexplained index errors would be that; the robust fix is a local
working copy synced through a remote.

## Gap 2 — the Phase-2 RTL flow has no environment, and Windows is the blocker

Absent: **Verilator, cocotb, Yosys, SymbiYosys, iverilog, GTKWave, make, Docker, and WSL.**
`winget` is available; `wsl` reports "not installed."

This matters more than a missing-package list suggests. Verilator has no supported native
Windows build — it runs under WSL2 or MSYS2. The plan's cash case depends explicitly on an
open-source EDA flow ("Prefer open-source EDA (Verilator, cocotb, Yosys) — the cash plan depends
on it"), and Phase 2 begins around month 6. So the verification environment needs one of:

1. **WSL2 + Ubuntu** — cheapest path; needs administrator rights and a reboot, so worth doing
   before it is on the critical path rather than during bring-up.
2. **A dedicated Linux machine or cloud VM** — better for long regressions and closer to what CI
   will run.
3. **Docker** — reproducible and matches CI images, but also needs WSL2 on Windows.

Recommendation: stand up option 1 now as a smoke test (install WSL2, `apt install verilator`,
elaborate a trivial module), purely to prove the toolchain works on this host. Discovering a
Windows-specific obstacle at month 6, with RTL waiting, is the expensive version of this.

## Gap 3 — nothing runs the tests automatically — **CLOSED 2026-07-30**

`.githooks/pre-commit` is tracked in the repo and does two things:

1. **Runs the suite and blocks the commit on failure.** The suite encodes twelve historical
   modelling errors as regression tests, so a commit that breaks them has usually re-introduced a
   flattering assumption.
2. **Warns when `perf_model/` code is staged without any `results_*.txt`.** Those outputs are
   tracked deliberately — they are the evidence behind PHASE0_FINDINGS claims — so code moving
   without outputs moving means the documents now disagree with the model.

Install once per clone: `git config core.hooksPath .githooks`. Bypass with `--no-verify`, rarely
justified.

**Verified to actually block**, because a hook that has never refused anything is unproven: a
deliberately failing test makes `git commit` exit 1 and the commit does not land. Worth recording
one trap found while verifying it — running `git commit … | head -8` makes the commit appear to
succeed, because `head` exiting early kills the hook with SIGPIPE before it reaches its `exit 1`.
Test hooks with git's output unpiped, or you will certify an unenforced hook.

CI still does not exist; the hook is a local gate only.

## Not yet covered by tests

Closed 2026-07-30: `sram_cost.py` and `measure_osqp_baseline.py` now have invariants — area
monotonic in bank count, the ≥64-bit bank-word validity floor flagged rather than silently
extrapolated, density bracketing the measured 8.1 Mb/mm² anchor, and the coupled-plant Hessian
density check that would have caught the decoupled-plant error automatically.

Closed 2026-07-30: **the fp64 datapath is costed** (`perf_model/fp64_datapath.py`,
PHASE0_FINDINGS §0.1, 28 new tests in `tests/test_fp64_datapath.py`). The `0.125` rate factor is
gone — `ArgmindCfg.fp64_fmas` is an absolute FMA count that carries area, so fp64 can now be
under-provisioned and the sizing can fail. Invariants pin: fp64 throughput independent of array
size, fp64 area nonzero and monotone, the FMA gate model bracketing its published anchors,
worst-case iterations requiring more fp64 units than typical, and the refinement trade modelled as
a real crossover rather than one-sided. Writing those tests caught three errors in the analysis
itself (array quantisation charged to the fp64 cluster; descriptor count and GEMM op count assumed
independently, double-counting GEMM 5.7×; option (b) quoted at option (d)'s merged area).

Still open:

- **`cm100_sizing.py` has no direct tests**, though `test_fp64_datapath.py` now exercises its
  `area_breakdown` and the recommended configuration. The invariants still worth pinning are that
  the recommendation meets every solver deadline at *worst-case* iteration counts (not typical),
  and that it is not dominated by any other point in the sweep.
- **No numerical-accuracy tests.** Nothing checks that a reduced-precision variant still
  converges, which is where `solver-numerics` set hard floors: fp32 minimum for the EKF covariance
  recursion with no fp16/bf16 path in any filter form, and fp64 for interior point. These are
  claims about arithmetic the silicon must provide, currently enforced only as a metadata check on
  the benchmark definitions — not demonstrated numerically.
- **Descriptor-issue granularity is not modelled, and may be the real ARM-MPC bottleneck.**
  Surfaced by the fp64 run: an HPIPM block op at nb=21 is only ~5.6 array cycles, so at >4
  sequencer cycles per descriptor the benchmark is issue-bound rather than array-bound, at any
  precision. The model charges a flat 150 scalar ops/iteration and cannot express this. Needs a
  real count of HPIPM's block operations per interior-point iteration, then a sequencer issue-rate
  term in the model.
- **No numerical demonstration that fp32 + refinement fails where we claim it does.** §0.1 item 4
  rests on the Wilkinson/Skeel/Higham convergence condition and on Carson & Higham, not on our own
  run of acados/HPIPM at the barrier endgame. This is the same gap as the reduced-precision item
  above, now narrowed to one specific, cheap experiment.
