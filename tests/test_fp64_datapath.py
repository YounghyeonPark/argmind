# SPDX-License-Identifier: Apache-2.0
"""
Invariants for the fp64 datapath costing (perf_model/fp64_datapath.py).
=======================================================================

WHY THIS FILE EXISTS

The fp64 datapath was the largest un-costed item in the CM-100 area budget: the
model got fp64 from a `0.125` rate factor on the bf16 MAC array, which implied
18 fp64 MAC/cycle for ZERO AREA. Two failure modes have to be prevented from
returning, and both are mechanically checkable:

  F1  fp64 throughput reappearing as a free multiple of the MAC array
  F2  fp64 area silently returning to zero in the budget

Plus the errors actually made while writing this analysis, each of which now has
a test:

  F3  the fp64 GEMM charged the ARRAY's tile-quantisation penalty, so a BIGGER
      array made fp64 SLOWER (caught by test_more_macs_never_increases_latency)
  F4  the multi-pass sequencer model assumed a descriptor count and a GEMM op
      count independently, double-counting the GEMM work 5.7x
  F5  the option comparison quoting a merged-multiformat area for the
      "separate fp64-only cluster" option, flattering option (b)

As elsewhere, these pin RELATIONSHIPS and DIRECTIONS, not today's numbers.
"""

import dataclasses
import math

import pytest

import argmind_model as M
import sram_cost as S
import cm100_sizing as C
import fp64_datapath as F


# ---------------------------------------------------------------------------
# F1 — fp64 throughput is a real, sized resource, not a rate factor
# ---------------------------------------------------------------------------

def test_f1_fp64_throughput_is_independent_of_mac_array_size():
    """fp64 does not fall out of the bf16 array for free.

    Error-free splitting of fp64 onto bf16 multipliers costs 49-361 passes
    (arXiv:2508.00441 extrapolated by mantissa width), not 8. So fp64 MAC/cycle
    must be the fp64 FMA count and must NOT scale with `tile`.
    """
    a = M.ArgmindCfg(name="small", tile=8, fp64_fmas=8)
    b = M.ArgmindCfg(name="big", tile=64, fp64_fmas=8)
    assert a.macs_per_cycle("fp64") == b.macs_per_cycle("fp64") == 8.0, (
        "fp64 throughput scales with the MAC array again — the 0.125 rate "
        "factor has returned and fp64 is being modelled as free")
    # And it must still scale with the thing that DOES buy it.
    assert M.ArgmindCfg(name="w", tile=8, fp64_fmas=16).macs_per_cycle("fp64") == 16.0


def test_f1_narrower_precisions_still_use_the_array():
    """The converse error: decoupling everything from the array."""
    c = M.ArgmindCfg(name="c", tile=12, fp64_fmas=8)
    assert c.macs_per_cycle("bf16") == 144
    assert c.macs_per_cycle("fp32") == 72
    assert c.macs_per_cycle("int8") == 288


# ---------------------------------------------------------------------------
# F2 — fp64 area is nonzero and monotone
# ---------------------------------------------------------------------------

def test_f2_fp64_area_is_nonzero_in_the_recommended_budget():
    a = C.area_breakdown(F.REC)
    assert a["fp64"] > 0.0, (
        "the fp64 datapath is back to zero area in the CM-100 budget — the "
        "exact defect this analysis was run to close")
    # It must be a real fraction of the part, not a rounding token.
    assert 0.02 < a["fp64"] / a["total"] < 0.40, (
        f"fp64 is {100*a['fp64']/a['total']:.1f}% of core area, outside the "
        "believable band for 8 FMAs on a ~0.7 mm2 part")


def test_f2_area_is_monotone_in_fp64_width_and_pessimism():
    prev = -1.0
    for f in (0, 1, 2, 4, 8, 16, 32):
        a = C.area_breakdown(dataclasses.replace(F.REC, fp64_fmas=f))["total"]
        assert a >= prev, "core area is not monotone in fp64 FMA count"
        prev = a
    central = C.area_breakdown(F.REC)["total"]
    pess = C.area_breakdown(F.REC, pessimistic=True)["total"]
    assert pess > central, "the pessimistic gate-count edge is not pessimistic"
    # The band must be wide enough to be honest about (2.1x on the FMA term).
    assert pess / central > 1.10


def test_f2_gate_model_brackets_the_published_anchors():
    """The calibration is the load-bearing part of every fp64 area number.

    Anchors: fp32 FMA in the published 8-12 kgate band; fp64 FMA below the
    measured 56.2 kgate 28nm dual-precision unit (which is a high-performance
    merged design and is used as the pessimistic edge, not the central value).
    """
    g32 = S.fma_gates_ieee(S.SIG_BITS["fp32"])
    g64 = S.fma_gates_ieee(S.SIG_BITS["fp64"])
    assert 8e3 <= g32 <= 12e3, f"fp32 FMA {g32:.0f} gates is outside 8-12 kgates"
    assert 20e3 <= g64 <= 40e3, f"fp64 FMA {g64:.0f} gates is outside 20-40 kgates"
    assert S.fma_gates_ieee(S.SIG_BITS["fp64"], pessimistic=True) \
        == pytest.approx(56.2e3, rel=0.10), (
        "the pessimistic edge has drifted away from the 56.2 kgate measured "
        "anchor it is supposed to represent")
    # Monotone in significand width, and superlinear (multiplier is O(n^2)).
    gs = [S.fma_gates_ieee(n) for n in (8, 11, 24, 53)]
    assert gs == sorted(gs)
    assert g64 / g32 > (53 / 24), "fp64/fp32 area ratio is sublinear in n — the " \
        "O(n^2) multiplier term has been lost"


def test_f2_vector_lane_estimate_can_hold_an_fp32_multiplier():
    """The correction made alongside the fp64 costing.

    The old 2200 gates/lane was below the gate count of an fp32 MULTIPLIER
    alone, so it could not describe an fp32 vector lane at all.
    """
    mult_only = S.K_FMA_MUL_GATES_PER_BIT2 * S.SIG_BITS["fp32"] ** 2
    assert C.VEC_LANE_GATES > mult_only, (
        f"VEC_LANE_GATES={C.VEC_LANE_GATES} is below the {mult_only:.0f}-gate "
        "fp32 multiplier it must contain")
    assert C.VEC_LANE_GATES >= S.fma_gates_ieee(S.SIG_BITS["fp32"])


# ---------------------------------------------------------------------------
# F3 — the quantisation penalty is charged against the unit doing the work
# ---------------------------------------------------------------------------

def test_f3_bigger_array_never_slows_fp64_work():
    arm = next(b for b in M.BENCHMARKS if b.prec == "fp64")
    lat = [M.model_argmind(arm, M.ArgmindCfg(name=f"t{t}", tile=t, fp64_fmas=8)
                           )["latency_us"] for t in (8, 12, 16, 32, 64)]
    assert lat == sorted(lat, reverse=True) or max(lat) / min(lat) < 1.0001, (
        f"fp64 latency is not monotone non-increasing in array size: {lat}. The "
        "fp64 GEMM is being charged the ARRAY's tile-quantisation penalty again")


def test_f3_fp64_util_uses_the_cluster_width():
    dim = 21.0
    a = M.ArgmindCfg(name="a", tile=12, fp64_fmas=4)
    b = M.ArgmindCfg(name="b", tile=12, fp64_fmas=8)
    c = M.ArgmindCfg(name="c", tile=64, fp64_fmas=8)
    # Same cluster, different array -> IDENTICAL fp64 utilisation. THIS is the
    # invariant that matters; it is what test_more_macs_never_increases_latency
    # was catching.
    for d in (10.0, 21.0, 40.0, 79.0):
        assert b.gemm_util("fp64", d) == pytest.approx(c.gemm_util("fp64", d)), (
            f"fp64 utilisation depends on the array size at dim={d}")
    # Same array, different cluster -> must differ at SOME dimension. (Not at
    # every dimension: at dim=21 both 4 and 8 quantise to 24, coincidentally.)
    assert any(a.gemm_util("fp64", d) != b.gemm_util("fp64", d)
               for d in (10.0, 21.0, 40.0, 79.0)), (
        "fp64 utilisation is insensitive to the cluster width at every "
        "dimension tried — it is not being charged against the cluster at all")
    # Converse for fp32, which DOES run on the array.
    assert b.gemm_util("fp32", dim) != c.gemm_util("fp32", dim)
    # Utilisation is a fraction.
    for cfg in (a, b, c):
        for p in ("bf16", "fp32", "fp64"):
            assert 0.0 < cfg.gemm_util(p, dim) <= 1.0


# ---------------------------------------------------------------------------
# F4 — the multi-pass model is internally consistent
# ---------------------------------------------------------------------------

def test_f4_block_op_count_is_consistent_with_the_gemm_op_count():
    """Descriptor count and GEMM MAC count must not be assumed independently.

    An earlier version assumed 180 block ops/iteration AND nb^3/3 MACs each,
    which double-counted ARM-MPC's per-iteration GEMM work 5.7x and produced a
    fictitious sequencer bottleneck.
    """
    total = F.MACS_PER_BLOCK_OP * F.BLOCK_OPS_PER_IP_ITER
    assert total == pytest.approx(F.ARM.gemm_macs_per_iter, rel=1e-9), (
        "MACS_PER_BLOCK_OP x BLOCK_OPS_PER_IP_ITER no longer equals the "
        "benchmark's GEMM MACs per iteration")


def test_f4_multipass_scales_the_way_the_physics_says():
    r = {p: F.multipass_analysis(F.REC, p, F.ARM.iters_wc) for p in (9, 36, 144)}
    # Array time is linear in the pass count.
    assert r[36]["array_us"] / r[9]["array_us"] == pytest.approx(4.0, rel=1e-6)
    # GEMV (bandwidth-bound on sliced operands) scales with the SLICE count,
    # i.e. sqrt of the pass count -- not the pass count.
    assert r[36]["gemv_us"] / r[9]["gemv_us"] == pytest.approx(2.0, rel=1e-6)
    # The issue:array ratio is pass-count-INDEPENDENT. This is the finding that
    # replaced the expected "multi-pass is sequencer-bound" story.
    ratios = [r[p]["issue_frac_of_array"] for p in (9, 36, 144)]
    assert max(ratios) / min(ratios) == pytest.approx(1.0, rel=1e-6), (
        "the multi-pass sequencer tax has become pass-count-dependent; both "
        "terms scale with P so the ratio must not")


def test_f4_multipass_is_worse_on_energy_than_native_at_fp64_pass_counts():
    """The ground the rejection of option (a) actually rests on.

    At 9 passes (fp32-equivalent accuracy only) multi-pass is CHEAPER, and the
    model must keep saying so -- that is why the fp32 path uses operand
    splitting. At the pass counts fp64 accuracy requires, it must be worse.
    """
    native = F.all_fp64_energy(F.REC, F.ARM.iters_wc)["energy_uj"]
    cheap = F.multipass_analysis(F.REC, 9, F.ARM.iters_wc)["energy_uj"]
    assert cheap < native, (
        "9-pass bf16 is no longer cheaper than native fp64 — the model has lost "
        "the reason the fp32 path uses operand splitting at all")
    for p in F.OZAKI_PASSES_BF16:
        e = F.multipass_analysis(F.REC, p, F.ARM.iters_wc)["energy_uj"]
        assert e > 2.0 * native, (
            f"multi-pass at {p} passes is within 2x of native fp64 energy; the "
            "rejection of option (a) rests on this margin")


def test_f4_pass_counts_stay_anchored_to_the_citation():
    """arXiv:2508.00441 Table 1: FP16 25-196 passes, FP8(E4M3) 121."""
    assert F.OZAKI_PASSES_FP16 == (25, 196)
    assert F.OZAKI_PASSES_FP8 == 121
    # bf16 has FEWER mantissa bits than fp16, so it needs MORE slices/passes.
    assert F.OZAKI_PASSES_BF16[0] > F.OZAKI_PASSES_FP16[0]
    assert F.OZAKI_PASSES_BF16[1] > F.OZAKI_PASSES_FP16[1]
    # And fp32-equivalent accuracy must be far cheaper than fp64-equivalent:
    # that asymmetry IS the finding.
    assert F.BF16_PASSES_FOR_FP32[1] < F.OZAKI_PASSES_BF16[0] / 4


# ---------------------------------------------------------------------------
# F5 — option (b) is not quoted at option (d)'s merged area
# ---------------------------------------------------------------------------

def test_f5_separate_cluster_costs_more_than_the_merged_increment():
    """A dedicated fp64-only cluster is NOT the same cost as widening the
    existing vector unit. Conflating them flatters option (b)."""
    standalone = S.fma_cluster_mm2(8, "fp64", multiformat=False)
    merged_block = (S.fma_cluster_mm2(8, "fp64", multiformat=True)
                    + 16 * C.VEC_TRANSCENDENTAL_GATES * S.A_GATE_UM2 / 1e6)
    vec_only = (16 * C.VEC_LANE_GATES * S.A_GATE_UM2
                * S.MAC_STAGING_OVERHEAD / 1e6)
    increment = merged_block - vec_only
    assert increment > 0, "merging made fp64 free — check the attribution"
    assert increment < standalone, (
        "the merged fp64 INCREMENT is not cheaper than a standalone fp64-only "
        "cluster, so the recommendation to merge has no basis")
    assert S.fma_gates_ieee(53, multiformat=True) > S.fma_gates_ieee(53)


def test_f5_native_throughout_is_the_most_expensive_option():
    """Option (c) must dominate the others on cost, or the sweep is broken."""
    c = S.native_fp64_array_mm2(F.REC.macs)
    b = S.fma_cluster_mm2(8, "fp64")
    a = S.wide_accum_delta_mm2(F.REC.macs)
    assert c > 5 * b > 0, "native-throughout is not decisively more expensive"
    assert a < b, "the multi-pass accumulator now costs more than 8 native FMAs"


# ---------------------------------------------------------------------------
# The requirement itself: sizing must be able to FAIL
# ---------------------------------------------------------------------------

def test_fp64_sizing_can_report_failure():
    """A cluster small enough must MISS the deadline, or the sizing sweep is
    ratifying rather than deciding -- the same defect that made the p95-iteration
    bandwidth sweep worthless."""
    fails = [f for f in (1, 2, 4, 8, 16)
             if F.arm_mpc_latency(dataclasses.replace(F.REC, fp64_fmas=f),
                                  F.ARM.iters_wc)["_total"] > F.ARM.deadline_us]
    assert fails, ("no fp64 cluster width misses ARM-MPC's deadline; the "
                   "requirement is not being sized against anything")
    budget_fails = [f for f in (1, 2, 4, 8, 16)
                    if F.arm_mpc_latency(dataclasses.replace(F.REC, fp64_fmas=f),
                                         F.ARM.iters_wc)["_total"]
                    > F.ARM.deadline_us * F.PERIOD_BUDGET]
    assert len(budget_fails) > len(fails), (
        "the 50%-of-period budget excludes no additional configuration, so it "
        "is doing no work")


def test_worst_case_iterations_bind_the_fp64_sizing():
    """The typical case must not be able to size the cluster.

    At 15 IP iterations a smaller cluster suffices than at 30 — if not, the
    worst-case tail is decoration here as it was in the bandwidth sweep.
    """
    typ = F.min_fp64_fmas(iters=F.ARM.iters)
    wc = F.min_fp64_fmas(iters=F.ARM.iters_wc)
    assert 0 < typ < wc, (
        f"worst-case iterations ({F.ARM.iters_wc}) require no more fp64 units "
        f"than typical ({F.ARM.iters}): typ={typ}, wc={wc}")


def test_recommended_cluster_has_real_margin_but_is_not_absurd():
    d = F.arm_mpc_latency(F.REC, F.ARM.iters_wc)
    margin = F.ARM.deadline_us * F.PERIOD_BUDGET / d["_total"]
    assert 2.0 <= margin <= 5.0, (
        f"recommended fp64 width gives {margin:.2f}x margin on the 50%-period "
        "budget — either under-provisioned or grossly over-provisioned")
    assert d["_bind"] == "fp64_gemm", (
        f"ARM-MPC binds on {d['_bind']}, not the fp64 GEMM. If that is real it "
        "changes the sizing conclusion and needs a human")


def test_only_interior_point_needs_fp64():
    """The requirement's scope. If a second benchmark acquires fp64, the cluster
    must be re-sized, so make that visible rather than silent."""
    fp64_benches = [b.name for b in M.BENCHMARKS if b.prec == "fp64"]
    assert fp64_benches == ["ARM-MPC"], (
        f"fp64 benchmarks are now {fp64_benches}; the 8-FMA cluster was sized "
        "against ARM-MPC alone and must be re-derived")
    for b in M.BENCHMARKS:
        if b.algo == "ip":
            assert b.prec == "fp64"


# ---------------------------------------------------------------------------
# Q3: refinement must be modelled as a real trade, not dismissed
# ---------------------------------------------------------------------------

def test_refinement_is_modelled_as_a_genuine_crossover():
    """If refinement always loses, the model is not representing it honestly.

    It is FASTER at every k (it replaces fp64 O(n^3) with fp32 O(n^3)), and it
    must win on energy at small k too -- the argument against it is numerical
    and determinism-based, not "our model says it is worse".
    """
    s1 = F.all_fp64_energy(F.REC, F.ARM.iters_wc)
    lat = [F.refinement_analysis(F.REC, k, F.ARM.iters_wc)["latency_us"]
           for k in (1, 2, 3, 5)]
    assert all(x < s1["latency_us"] for x in lat), (
        "iterative refinement is never faster than all-fp64, which contradicts "
        "replacing an fp64 O(n^3) factorisation with an fp32 one")
    assert lat == sorted(lat), "refinement latency is not monotone in k"
    e = [F.refinement_analysis(F.REC, k, F.ARM.iters_wc)["energy_uj"]
         for k in (1, 2, 3, 5)]
    assert e[0] < s1["energy_uj"] < e[-1], (
        "there is no energy crossover in k — the refinement trade is being "
        "modelled as one-sided")


def test_refinement_residual_cost_ratio_matches_the_hand_derivation():
    """residual : factorisation = 3k/nb. This is the claim that the asymptotic
    HPC case for refinement does not transfer to embedded MPC block sizes."""
    for k in (1, 2, 3, 5):
        r = F.refinement_analysis(F.REC, k, F.ARM.iters_wc)
        assert r["resid_vs_factor"] == pytest.approx(3.0 * k / F.NB, rel=1e-9)
    # At our block size the residual is NOT cheap; at HPC sizes it is.
    assert 3.0 * 3 / F.NB > 0.25
    assert 3.0 * 3 / 200.0 < 0.05


def test_refinement_still_requires_fp64_hardware():
    """The point that settles Q3: refinement RELOCATES the fp64 requirement.

    Whatever k is, the residual is fp64 work, so the fp64 MAC count must stay
    strictly positive and grow with k.
    """
    prev = 0.0
    for k in (1, 2, 3, 5):
        r = F.refinement_analysis(F.REC, k, F.ARM.iters_wc)
        assert r["fp64_macs"] > prev > -1
        prev = r["fp64_macs"]
        assert 0.0 < r["fp64_frac_of_work"] < 1.0


# ---------------------------------------------------------------------------
# CPU baseline fairness on fp64
# ---------------------------------------------------------------------------

def test_cpu_baselines_are_charged_a_real_fp64_rate():
    """The model used to charge fp64 at the CPU's fp32 rate, which understated
    our advantage. Fixing it INCREASES our ratio, so it needs pinning."""
    for c in M.ALL_CPUS:
        f32 = c.peak_flops_per_cycle("fp32")
        f64 = c.peak_flops_per_cycle("fp64")
        assert f64 < f32, (
            f"{c.name}: fp64 peak is not below fp32 peak. NEON/SVE fp64 is "
            "half-rate and SME fp64 outer products are quarter-rate")
        assert f64 >= f32 / 4.0, f"{c.name}: fp64 rate below quarter of fp32"
        assert c.peak_flops_per_cycle("int8") == c.int8_ops_per_cycle


def test_the_cheap_cpu_still_meets_arm_mpc_deadline():
    """CLAUDE.md: never pitch 'the CPU can't do this'. After the fp64 rate fix
    the A55 baseline got 1.6x slower; check it did not get so slow that the
    model starts claiming infeasibility for the CPU."""
    r = M.model_cpu(F.ARM, M.CPU_A55)
    assert r["latency_us"] <= F.ARM.deadline_us, (
        "the most efficient CPU baseline now MISSES ARM-MPC's deadline. That "
        "would be a new and much stronger claim than energy — verify the fp64 "
        "rate assumption before relying on it")


def test_arm_mpc_perf_w_is_load_bearing_on_the_fp64_energy_constant():
    """Documented sensitivity: the go/no-go must survive a 3x error in
    E_MAC_PJ['fp64'], and the constant must be a material share of the total."""
    orig = M.E_MAC_PJ["fp64"]
    best_cpu = min(M.model_cpu(F.ARM, c)["energy_uj"] for c in M.ALL_CPUS)
    try:
        base = M.model_argmind(F.ARM, F.REC)["energy_uj"]
        M.E_MAC_PJ["fp64"] = orig * 3.0
        hi = M.model_argmind(F.ARM, F.REC)["energy_uj"]
    finally:
        M.E_MAC_PJ["fp64"] = orig
    assert hi > 1.3 * base, (
        "tripling fp64 MAC energy moves ARM-MPC's total by <30%; the constant "
        "is being reported as load-bearing when it is not")
    assert best_cpu / hi >= 5.0, (
        "the 5x kill criterion does NOT survive a 3x error in the fp64 energy "
        "constant. That must be stated as a dependency, not buried")


# ---------------------------------------------------------------------------
# The report must run and stay self-consistent
# ---------------------------------------------------------------------------

def test_report_runs(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    F.main()
    txt = capsys.readouterr().out
    for token in ("Q1", "Q2", "Q3", "Q4", "Q5", "MACIF", "NOT QUOTABLE"):
        assert token in txt
    assert "0.125" in txt or "rate-factor" in txt, (
        "the report no longer records what it replaced")


def test_no_stale_zero_fp64_area_in_the_sizing_report():
    a = C.area_breakdown(F.REC)
    assert "fp64" in a and a["fp64"] > 0
    assert a["sub"] == pytest.approx(
        a["spad"] + a["mac"] + a["vec"] + a["fp64"] + a["rest"], rel=1e-9)
    assert a["total"] == pytest.approx(a["sub"] * C.TOP_MARGIN, rel=1e-9)


def test_wide_accumulator_model_is_sane():
    assert S.wide_accum_delta_mm2(0) == 0.0
    assert S.wide_accum_delta_mm2(144, bits=S.BASE_ACCUM_BITS) == 0.0
    assert S.wide_accum_delta_mm2(144) > S.wide_accum_delta_mm2(72) > 0
    # Exact fp64 product accumulation needs at least 2*53 bits.
    assert S.WIDE_ACCUM_BITS >= 2 * S.SIG_BITS["fp64"]


def test_fp64_element_size_is_eight_bytes_everywhere():
    assert M.BYTES_PER_ELEM["fp64"] == 8
    c = F.REC
    assert c.gemv_macs_per_cycle("fp64") == min(c.scratchpad_bw_bpc / 8.0,
                                                c.fp64_fmas)
    assert c.gemm_operand_bytes_per_mac("fp64") == pytest.approx(
        8.0 * (1 / c.tile + 1 / c.tile_cols))
