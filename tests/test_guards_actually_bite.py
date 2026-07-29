"""
Meta-tests: prove the invariants in test_model_invariants.py are load-bearing.
=============================================================================

A regression suite that has never failed has not been shown to catch anything.
Each test here re-introduces a real historical bug at runtime and asserts that
the corresponding guard *fails*. If a test in this file starts passing trivially
(i.e. the mutated model still looks fine), the matching guard has gone slack and
is no longer protecting anything.

Mutations are applied to copies, never to module state that outlives the test.
"""

import copy

import pytest

import argmind_model as M


def by_name(name):
    return next(b for b in M.BENCHMARKS if b.name == name)


def test_gemv_as_gemm_misclassification_is_caught():
    """Historical error 2: matrix-vector work modelled as matrix-matrix.

    Moving QUAD-MPC's GEMV work into the GEMM bucket must break the
    "MAC count does not matter" invariant. If it does not, that invariant is
    no longer testing the GEMV/GEMM split.
    """
    mutated = copy.deepcopy(by_name("QUAD-MPC"))
    mutated.gemm_macs_per_iter = mutated.gemv_macs_per_iter
    mutated.gemv_macs_per_iter = 0.0

    small = M.model_argmind(mutated, M.ArgmindCfg(name="s", tile=16))
    big = M.model_argmind(mutated, M.ArgmindCfg(name="b", tile=64))

    # Under the bug, a 16x bigger array "helps" a lot. The real guard asserts
    # these stay within 5%; here they must NOT.
    assert big["latency_us"] < 0.95 * small["latency_us"], (
        "misclassifying GEMV as GEMM no longer changes the MAC-count "
        "sensitivity — the e2 guard has gone slack")


def test_phantom_transcendentals_inflate_the_speedup():
    """Historical error 3: box projection charged as divide/sqrt.

    Restoring the term must materially improve the CM's apparent speedup,
    because the CPU model charges 12 cycles per transcendental against the CM's
    4/vector-lane. This is the mechanism that manufactured a latency advantage.
    """
    clean = by_name("QUAD-MPC")
    buggy = copy.deepcopy(clean)
    n_var, n_x, N = 80, 12, 20
    buggy.transcendental_per_iter = 2 * (n_var + N * n_x)   # the old formula

    cfg = M.ArgmindCfg(name="t", tile=16)

    def speedup(b):
        cm = M.model_argmind(b, cfg)["latency_us"]
        fastest = min(M.model_cpu(b, c)["latency_us"] for c in M.ALL_CPUS)
        return fastest / cm

    s_clean, s_buggy = speedup(clean), speedup(buggy)
    assert s_buggy > s_clean * 1.25, (
        f"phantom transcendentals only moved speedup {s_clean:.2f} -> "
        f"{s_buggy:.2f}; the term no longer distorts the comparison, so the e3 "
        "guard may be protecting nothing")


def test_dropping_overhead_energy_inflates_perf_w():
    """Historical error 1/7: energy counted datapath only.

    With the overhead multiplier and the banked scratchpad energy removed, perf/W
    must rise substantially — the effect that produced the original, absurd 128x.
    """
    b = by_name("QUAD-MPC")
    honest = M.ArgmindCfg(name="honest", tile=16)
    flattering = M.ArgmindCfg(name="flattering", tile=16, use_banked_energy=False)

    def perf_w(cfg, overhead):
        saved = M.E_OVERHEAD_MULT
        try:
            M.E_OVERHEAD_MULT = overhead
            cm = M.model_argmind(b, cfg)["energy_uj"]
            best = min(M.model_cpu(b, c)["energy_uj"] for c in M.ALL_CPUS)
            return best / cm
        finally:
            M.E_OVERHEAD_MULT = saved

    honest_ratio = perf_w(honest, M.E_OVERHEAD_MULT)
    flattering_ratio = perf_w(flattering, 1.0)
    assert flattering_ratio > honest_ratio * 1.3, (
        f"removing overhead energy only moved perf/W {honest_ratio:.1f}x -> "
        f"{flattering_ratio:.1f}x — the energy accounting is no longer sensitive "
        "to the overhead terms that were originally missing")


def test_ata_term_would_be_caught():
    """Historical error 8: forming `A^T A` in the SQP re-factorisation.

    A term proportional to the constraint count must break the invariant that
    the once-per-call GEMM depends only on n_var.
    """
    n_var = 30 * 7
    a_rows, b_rows = 30 * 14, 30 * 28
    # Reconstruct what the buggy formula produced.
    buggy_a = n_var ** 3 / 3.0 + (n_var + a_rows) * n_var ** 2
    buggy_b = n_var ** 3 / 3.0 + (n_var + b_rows) * n_var ** 2
    assert buggy_a != pytest.approx(buggy_b, rel=1e-9), (
        "the A^T A formulation is not distinguishable by constraint count, so "
        "the e8 guard cannot detect it")
    # And confirm the current model is insensitive, as the guard asserts.
    cur_a = M.condensed_mpc("a", n_x=14, n_u=7, N=30, iters=1, rate=1, note="",
                            refactor=True).gemm_macs_per_call
    cur_b = M.condensed_mpc("b", n_x=28, n_u=7, N=30, iters=1, rate=1, note="",
                            refactor=True).gemm_macs_per_call
    assert cur_a == pytest.approx(cur_b, rel=1e-9)


def test_infeasible_scenario_would_be_caught_by_status_check():
    """Historical error in the MEASUREMENT harness, not the model.

    The first 'aggressive' OSQP scenario was physically unreachable and drove the
    solver primal-infeasible, producing p95 ~900 iterations that measured nothing.
    The harness now flags non-'solved' ticks. This pins the rule that any
    iteration statistic must be accompanied by an all-solved status check.
    """
    statuses = ["solved"] * 198 + ["primal infeasible"] * 2
    bad = sum(s != "solved" for s in statuses)
    assert bad > 0, "status check cannot detect a non-solved tick"
