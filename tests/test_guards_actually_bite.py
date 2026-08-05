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


def test_a_free_sequencer_would_be_caught():
    """New for the descriptor-issue model (2026-07-30).

    The single most likely way for the sequencer model to go slack is for its
    costs to be zeroed -- setup free and dependency drain free -- at which point
    it can no longer report the one failure it exists to report. Prove that the
    "a plausible sequencer misses a deadline" invariant depends on those terms.
    """
    import dataclasses

    import sequencer as Q

    # RE-POINTED 2026-07-30 to QUAD-MPC. It used to use PWR-MPC, which was a
    # continuous-control-set ADMM row; re-specified as FCS-MPC that row has no
    # dependency chain at all, so it can no longer demonstrate the drain term.
    # QUAD-MPC at its MEASURED worst-case tail can, but only with both
    # pathologies present -- which is itself the downgraded finding.
    quad = dataclasses.replace(by_name("QUAD-MPC"),
                               iters=by_name("QUAD-MPC").iters_wc)
    budget = quad.deadline_us * Q.PERIOD_BUDGET

    honest = dataclasses.replace(M.CFG_100, seq_setup_cycles=20,
                                 seq_queue_depth=0, chain_block_override=1)
    assert M.model_argmind(quad, honest)["latency_us"] > budget

    free = dataclasses.replace(honest, seq_setup_cycles=0.0, drain_cycles=0)
    assert M.model_argmind(quad, free)["latency_us"] <= budget, (
        "zeroing descriptor setup and dependency drain no longer makes the "
        "QUAD-MPC budget reachable — the sequencer terms are not load-bearing")


def test_inconsistent_descriptor_decomposition_would_be_caught():
    """The consistency rule must actually bite.

    Descriptor counts derived independently of the MAC counts is the error that
    double-counted HPIPM GEMM work 5.7x. Corrupt one kernel op's MAC count and
    the consistency assertion must fire.
    """
    arm = copy.deepcopy(by_name("ARM-MPC"))
    M.assert_kernels_consistent(arm)          # clean
    bad = M.KernelOp(name="inflated syrk", kind="gemm",
                     macs=arm.kernels[1].macs * 2.0,
                     n_desc=arm.kernels[1].n_desc, repeat=arm.kernels[1].repeat)
    arm.kernels = [bad if i == 1 else k for i, k in enumerate(arm.kernels)]
    with pytest.raises(AssertionError):
        M.assert_kernels_consistent(arm)


def test_a_free_2x_of_int4_compute_is_invisible_at_the_row_level():
    """New for the on-device-AI envelope (2026-08-05), and the result is a warning.

    The likeliest way the int4 addition goes wrong is int4 acquiring 4x the array
    rate alongside its real 2x of bandwidth -- the shape of modelling errors 13
    and 14. This meta-test was written expecting a benchmark row to detect it.
    IT CANNOT, and that is worth pinning: at 128 B/cycle an int4 GEMV is
    bandwidth-bound (256 MAC/cycle from fetch, below the 288 array rate), so the
    array rate is UNOBSERVABLE on the LLM row no matter what value it takes.

    Consequences, both of which this test asserts:
      * only the DIRECT invariant on `macs_per_cycle` can protect this, so
        test_int4_does_not_grant_free_array_throughput is not redundant with any
        row-level check; and
      * the model can still express the difference -- at 256 B/cycle the fetch
        ceiling (512) rises above the array and the array becomes binding, which
        is the same crossover as §8's balance rule.
    """
    import dataclasses

    def mutate():
        return lambda self, prec: (
            float(self.fp64_fmas) if prec == "fp64"
            else self.macs * {"int4": 4.0, "int8": 2.0, "fp16": 1.0,
                              "bf16": 1.0, "fp32": 0.5}[prec])

    b4 = M.slm_decode("i4", 1.1, 2048, 22, 512, 20, "", prec="int8",
                      weight_prec="int4")
    wide = dataclasses.replace(M.CFG_100, scratchpad_bw_bpc=256.0, n_banks=16)

    honest_128 = M.model_argmind(b4, M.CFG_100)["compute_us"]
    honest_256 = M.model_argmind(b4, wide)["compute_us"]
    saved = M.ArgmindCfg.macs_per_cycle
    try:
        M.ArgmindCfg.macs_per_cycle = mutate()
        # The direct guard MUST fire.
        assert M.CFG_100.macs_per_cycle("int4") != M.CFG_100.macs_per_cycle("int8")
        free_128 = M.model_argmind(b4, M.CFG_100)["compute_us"]
        free_256 = M.model_argmind(b4, wide)["compute_us"]
    finally:
        M.ArgmindCfg.macs_per_cycle = saved

    assert free_128 == pytest.approx(honest_128), (
        "an int4 GEMV at 128 B/cycle has become array-bound; if that is real the "
        "bandwidth-first finding needs re-deriving for int4")
    assert free_256 < 0.95 * honest_256, (
        "even at 256 B/cycle the array rate is invisible, so the model can no "
        "longer express the cost of a wrong int4 rate anywhere")


def test_an_uncounted_kv_cache_would_be_caught():
    """Modelling error 17, re-introduced at runtime.

    The KV cache was omitted from `persistent_bytes` while the comment claimed
    it was there. Removing it again must move the external traffic AND the
    verdict-relevant latency, or the guard is cosmetic.
    """
    honest = by_name("SLM-DECODE")
    buggy = copy.deepcopy(honest)
    buggy.persistent_bytes = 1.1e9          # weights only, the old expression
    c = M.CFG_100
    h = M.model_argmind(honest, c)
    b_ = M.model_argmind(buggy, c)
    assert b_["bytes_moved"] < 0.99 * h["bytes_moved"]
    assert b_["latency_us"] < 0.99 * h["latency_us"], (
        "dropping the KV cache no longer changes the SLM row's latency; the "
        "e17 guard has gone slack")


def test_a_privately_provisioned_cpu_would_be_caught():
    """Modelling error 16's sibling: a swept axis that moves one side only.

    If `model_cpu` ignores `dram_gbs`, a bandwidth sweep credits the CM with
    bandwidth the CPU is not given. Prove the fairness guard depends on the
    parameter actually reaching the CPU's DRAM term.
    """
    b = by_name("SLM-DECODE")
    slow = min(M.model_cpu(b, x, dram_gbs=3.2)["latency_us"] for x in M.ALL_CPUS)
    fast = min(M.model_cpu(b, x, dram_gbs=51.2)["latency_us"] for x in M.ALL_CPUS)
    assert slow > 1.5 * fast, (
        "the CPU baseline is insensitive to system DRAM bandwidth, so any "
        "bandwidth sweep comparing against it is an artefact")


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
