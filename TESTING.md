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

**42 tests, all passing**, in three files:

- [tests/test_model_invariants.py](tests/test_model_invariants.py) — 24 tests. One regression
  test per historical modelling error, plus physical invariants.
- [tests/test_sram_and_harness.py](tests/test_sram_and_harness.py) — 10 tests covering the
  SRAM cost model and the OSQP harness (see "Not yet covered" below, now largely closed).
- [tests/test_guards_actually_bite.py](tests/test_guards_actually_bite.py) — 5 meta-tests that
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
  deadline, and another asserting the CPU wins somewhere. A model that cannot produce a negative
  result is not evidence, and CLAUDE.md's rule to that effect is now mechanically enforced
  rather than merely stated.

### Why the meta-tests matter

Phase 0 produced **ten** modelling errors, each caught by manual or agent review — expensive
and non-repeatable. The suite encodes all ten (the tenth: MAC count and operand-reuse depth were the same number). But a regression suite that has never failed has
not been shown to catch anything, so `test_guards_actually_bite.py` mutates the model back to
each buggy formulation and asserts the guard trips. Confirmed live:

- Misclassifying GEMV as GEMM makes a 16× larger array look 5%+ faster → caught.
- Restoring the phantom transcendental term inflates apparent speedup by >25% → caught.
- Removing overhead energy inflates perf/W by >30% → caught.

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

## Gap 3 — nothing runs the tests automatically

There are no git hooks and no CI, so the suite runs only when invoked. Now that the repo exists,
a pre-commit hook is cheap and worth adding. Until then the working rule (also in CLAUDE.md) is:
run `python -m pytest` after any change to `perf_model/`, and before quoting any modelled number.

## Not yet covered by tests

Closed 2026-07-30: `sram_cost.py` and `measure_osqp_baseline.py` now have invariants — area
monotonic in bank count, the ≥64-bit bank-word validity floor flagged rather than silently
extrapolated, density bracketing the measured 8.1 Mb/mm² anchor, and the coupled-plant Hessian
density check that would have caught the decoupled-plant error automatically.

Still open:

- **`cm100_sizing.py` has no tests.** Its output is the recommended part, so the invariants worth
  pinning are that the recommendation meets every solver deadline at *worst-case* iteration
  counts (not typical), and that it is not dominated by any other point in the sweep.
- **No numerical-accuracy tests.** Nothing checks that a reduced-precision variant still
  converges, which is exactly where `solver-numerics` set hard floors: fp32 minimum for the EKF
  covariance recursion with no fp16/bf16 path in any filter form, and fp64 (or fp32 plus
  mandatory iterative refinement, which the model does not represent) for interior-point. These
  are claims about arithmetic the silicon must provide, and they are currently asserted in prose
  and enforced only as a metadata check on the benchmark definitions — not demonstrated
  numerically.
- **The fp64 datapath is un-costed.** The model obtains fp64 from a 0.125 rate factor on a bf16
  array, which is a modelling convenience; a real 1/8-rate path needs either multi-pass
  decomposition or a separate narrow FMA cluster, and its area is currently zero in the budget.
  It sits exactly on the MACIF boundary and interior-point MPC requires it, so it is not optional.
