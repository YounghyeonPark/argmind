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

**29 tests, all passing**, in two files:

- [tests/test_model_invariants.py](tests/test_model_invariants.py) — 24 tests. One regression
  test per historical modelling error, plus physical invariants.
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

Phase 0 produced **nine** modelling errors, each caught by manual or agent review — expensive
and non-repeatable. The suite encodes all nine. But a regression suite that has never failed has
not been shown to catch anything, so `test_guards_actually_bite.py` mutates the model back to
each buggy formulation and asserts the guard trips. Confirmed live:

- Misclassifying GEMV as GEMM makes a 16× larger array look 5%+ faster → caught.
- Restoring the phantom transcendental term inflates apparent speedup by >25% → caught.
- Removing overhead energy inflates perf/W by >30% → caught.

If a test in that file ever starts passing *trivially*, the corresponding guard has gone slack.

## Gap 1 — no version control (most urgent)

There is no git repository. Consequences already felt: three agents edited `perf_model/` and the
Markdown docs concurrently on 2026-07-29, and there is **no way to diff, attribute, revert, or
bisect** any of it. When a benchmark number moves, we cannot tell what moved it.

This also blocks the plan directly: [PLAN.md](PLAN.md) §7.1 commits to public repositories with
CI running on every PR. That is unreachable without a repo.

Recommended: `git init`, a `.gitignore` for `results_*.txt` and `__pycache__`, and commit the
current state as a baseline. Left as a founder decision because CLAUDE.md currently records
"not a git repo" as a deliberate convention.

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

No git hooks, no CI, so the suite only runs when invoked by hand. Until there is a repo, the
practical mitigation is to run `python -m pytest` after any change to `perf_model/`, and
before quoting any number from it.

## Not yet covered by tests

- `sram_cost.py` and `cm100_sizing.py` have **no tests**. The scratchpad energy figure is now
  ~80% of modelled solver call energy and the single most important constant in the model, so it
  deserves invariants of its own: monotonicity of area in bank count, the bank-word ≥64-bit
  validity floor (below which the model is known to become unphysical), and agreement with the
  measured 8.1 Mb/mm² anchor.
- `measure_osqp_baseline.py` is unvalidated by tests. Worth pinning: the condensed Hessian must
  be dense for a coupled plant (this is the invariant that would have caught the decoupled-plant
  error automatically), and every reported iteration statistic must come from an all-`solved`
  run.
- No numerical-accuracy tests: nothing checks that a reduced-precision variant still converges,
  which is the requirement `solver-numerics` flagged as fp32-minimum for EKF covariance and
  fp64-or-refinement for interior-point.
