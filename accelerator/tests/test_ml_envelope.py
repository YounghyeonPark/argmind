# SPDX-License-Identifier: Apache-2.0
"""
Invariant tests for the on-device AI envelope (perf_model/ml_envelope.py).
=========================================================================

WHY THIS FILE EXISTS

Two model changes landed with this study, and BOTH were of the shape that has
produced eleven of our fifteen flattering modelling errors -- a new precision and
a new swept axis:

  * int4 was added. The failure mode to guard is int4 acquiring a free 2x of
    COMPUTE alongside its real 2x of bandwidth, or shedding DRAM energy on the
    weights it streams.
  * external bandwidth became a swept parameter. The failure mode to guard is a
    sweep that moves only OUR side of the comparison, which is precisely what
    `model_cpu`'s hard-coded 12.8 GB/s would have done.

Plus two corrections found by the same audit, both of which move AGAINST us:

  * the SLM row's KV cache was never counted, although the code comment claimed
    it was;
  * `model_cpu` gave the CPU no credit for narrow matrix operands, so the fp16
    operand study compared a narrowed CM against an un-narrowed CPU.

AND ONE NEGATIVE RESULT WHICH MUST STAY LOCKED. Large-model batch-1 LLM decode is
NOT in the differentiated zone at any (size, bandwidth, precision) point in the
frozen sweep. A later model change must not quietly move it there, so the
guard here is stated as a property of the whole plane rather than of one row.

As elsewhere in this suite: relationships, directions and named mechanisms, not
pinned numbers.
"""

import dataclasses as dc

import pytest

import argmind_model as M
import ml_envelope as E


# ---------------------------------------------------------------------------
# int4: bandwidth yes, compute no
# ---------------------------------------------------------------------------

def test_int4_is_half_a_byte_and_nothing_else_in_the_tables():
    assert M.BYTES_PER_ELEM["int4"] == 0.5
    assert M.BYTES_PER_ELEM["int4"] * 2 == M.BYTES_PER_ELEM["int8"]
    # int4 MAC energy must be BELOW int8 (fewer partial products on one operand)
    # but NOWHERE NEAR half: the accumulator, staging and clock do not narrow.
    assert M.E_MAC_PJ["int4"] < M.E_MAC_PJ["int8"]
    assert M.E_MAC_PJ["int4"] > 0.6 * M.E_MAC_PJ["int8"], (
        "an int4 MAC cheaper than 0.6x int8 implies the accumulator and operand "
        "staging narrowed too, which they do not for a 4b x 8b product")


def test_int4_does_not_grant_free_array_throughput():
    """THE headline guard on this change.

    Weight-only 4-bit quantisation multiplies a 4-bit weight by an 8-bit
    activation. A digital multiplier is sized by its wider operand, so the array
    rate is the int8 rate. If a future edit gives int4 a 4.0 scale factor this
    test fails, and it should: that would be a MACRO capability declared through
    MACIF, not a property of this datapath.
    """
    for c in (M.CFG_100, M.CFG_16, M.CFG_64):
        assert c.macs_per_cycle("int4") == c.macs_per_cycle("int8")


def test_int4_doubles_the_gemv_fetch_ceiling_and_only_that():
    c = M.CFG_100
    assert c.gemv_macs_per_cycle("int4") == pytest.approx(
        2.0 * c.gemv_macs_per_cycle("int8"))
    # ...and the doubled ceiling must still be BELOW the array rate, i.e. int4
    # GEMV stays bandwidth-bound on chip. If this ever inverts, the balance rule
    # (min tile = sqrt(bw/2)) needs re-deriving for int4.
    assert c.gemv_macs_per_cycle("int4") < c.macs_per_cycle("int4")


def test_int4_breaks_the_288_bytes_per_cycle_balance_coincidence():
    """§8's balance rule needs an int4 amendment, and it must stay visible.

    §8: "a 12x12 array absorbs GEMV up to 288 B/cycle -- the same 288 at fp32,
    bf16 AND int8, because this datapath's rate multiplier is ~1/element-size."
    int4's multiplier is int8's while its element is half the size, so the knee
    halves to 144 B/cycle. CM-100's 128 B/cycle is under it, but a 256 B/cycle
    upgrade would be ARRAY-bound at int4 with a 12x12 tile.
    """
    c = M.CFG_100
    knee = {p: c.macs_per_cycle(p) * M.BYTES_PER_ELEM[p]
            for p in ("fp32", "bf16", "int8", "int4")}
    assert knee["fp32"] == knee["bf16"] == knee["int8"] == 288.0
    assert knee["int4"] == 144.0
    # Today's 128 B/cycle is still under the int4 knee, so nothing moves...
    assert c.scratchpad_bw_bpc < knee["int4"]
    # ...but the advertised 256 B/cycle upgrade is not drop-in at int4.
    assert 256.0 > knee["int4"]


def test_int4_still_pays_full_dram_energy_per_streamed_byte():
    """int4 halves the BYTES. It must not make a byte cheaper."""
    c = E.cfg_bw(25.6)
    b8 = E.slm(1.1, 20, weight_prec="int8")
    b4 = E.slm(1.1, 20, weight_prec="int4")
    r8, r4 = M.model_argmind(b8, c), M.model_argmind(b4, c)
    e8 = r8["bytes_moved"] * c.offchip_pj_per_byte
    e4 = r4["bytes_moved"] * c.offchip_pj_per_byte
    # Bytes roughly halve (not exactly: the KV cache stays int8).
    assert 0.45 < r4["bytes_moved"] / r8["bytes_moved"] < 0.6
    # Per-byte cost is IDENTICAL.
    assert (e4 / r4["bytes_moved"]) == pytest.approx(e8 / r8["bytes_moved"])
    # And DRAM must still dominate the row at int4, or the study's central
    # claim (this workload's binding resource is off chip) has been broken.
    assert e4 / 1e6 > 0.5 * r4["energy_uj"]


def test_int4_energy_estimate_is_not_load_bearing():
    """Forcing the int4 MAC to int8's energy must not move the row materially.

    This is the stated reason the 0.06 pJ estimate is acceptable. If a future
    change makes the row MAC-energy-sensitive, the estimate needs real evidence.
    """
    c = E.cfg_bw(25.6)
    b4 = E.slm(1.1, 20, weight_prec="int4")
    base = M.model_argmind(b4, c)["energy_uj"]
    saved = M.E_MAC_PJ["int4"]
    try:
        M.E_MAC_PJ["int4"] = M.E_MAC_PJ["int8"]
        pess = M.model_argmind(b4, c)["energy_uj"]
    finally:
        M.E_MAC_PJ["int4"] = saved
    assert pess / base < 1.02


def test_int4_is_sub_byte_and_that_is_recorded():
    """The addressing consequence must stay visible in the model, not in prose.

    Two int4 elements share a byte, so a row length is even and a leading
    dimension counts nibbles, which the v0.1 byte-addressing AGUs cannot express.
    The last two capacity errors were both a layout the descriptor cannot
    address; this one is declared.
    """
    assert "int4" in M.SUB_BYTE_FORMATS
    for f in M.SUB_BYTE_FORMATS:
        assert M.BYTES_PER_ELEM[f] < 1.0
    # And it is NOT in the runtime's wire encoding, deliberately: adding it is a
    # descriptor-format decision with a codepoint cost, not a model detail.
    from argmind.precision import Format
    assert not any(f.name == "INT4" for f in Format)


# ---------------------------------------------------------------------------
# Baseline fairness on the swept axis
# ---------------------------------------------------------------------------

def test_external_bandwidth_override_is_neutral_when_unset():
    """Default None must reproduce the tier defaults exactly."""
    assert M.CFG_100.dma_bw_gbs_override is None
    assert M.CFG_100.dma_bw_gbs == 6.4
    mcu = dc.replace(M.CFG_100, tier="T1M")
    assert mcu.dma_bw_gbs == 1.6
    assert dc.replace(M.CFG_100, dma_bw_gbs_override=25.6).dma_bw_gbs == 25.6


def test_cpu_dram_bandwidth_default_reproduces_the_old_hard_constant():
    """`model_cpu`'s default must be the 12.8 GB/s it used to hard-code."""
    for b in M.BENCHMARKS:
        for cpu in M.ALL_CPUS:
            a = M.model_cpu(b, cpu)
            c = M.model_cpu(b, cpu, dram_gbs=12.8)
            assert a["latency_us"] == pytest.approx(c["latency_us"])
            assert a["energy_uj"] == pytest.approx(c["energy_uj"])


def test_cpu_feels_the_swept_bandwidth_on_a_weight_streaming_row():
    """The sweep must move BOTH sides. This is the baseline-fairness guard.

    A DRAM-bound ML row on the CPU must get slower when external bandwidth is
    cut. If it does not, `model_cpu` has re-acquired a private bandwidth and any
    perf/W figure from the sweep is an artefact.
    """
    b = E.slm(1.1, 20, weight_prec="int8")
    slow = min(M.model_cpu(b, x, dram_gbs=3.2)["latency_us"] for x in M.ALL_CPUS)
    fast = min(M.model_cpu(b, x, dram_gbs=51.2)["latency_us"] for x in M.ALL_CPUS)
    assert slow > 1.5 * fast


def test_cpu_gets_credit_for_narrow_matrix_operands():
    """The correction that moves AGAINST us, locked so it cannot regress.

    Before 2026-08-05 `model_cpu` charged every GEMV at `b.prec`, so an
    fp16-operand or int4-weight row narrowed OUR fetch and not the CPU's. On a
    baseline whose GEMV is L1-bandwidth-bound (SME2 here) that is a real 2x we
    were withholding from the adversary.
    """
    kw = dict(n_x=12, n_u=4, N=20, iters=15, rate=1000, iters_worst=74,
              check_term=1)
    base = M.condensed_mpc("Q", note="", **kw)
    narrow = M.condensed_mpc("Q/fp16", note="", operand_prec="fp16", **kw)
    assert 0.0 < narrow.solve_prec_frac < 1.0
    b_lat = min(M.model_cpu(base, x)["latency_us"] for x in M.ALL_CPUS)
    n_lat = min(M.model_cpu(narrow, x)["latency_us"] for x in M.ALL_CPUS)
    assert n_lat < b_lat, (
        "narrowing the bulk matrix operands must make the CPU faster too; if it "
        "does not, the mixed-precision study is comparing a narrowed CM against "
        "an un-narrowed CPU")


def test_cpu_has_no_int4_compute_path():
    """No shipping CPU has a 4-bit dot product; int4 runs at the int8 rate.

    Deliberately generous to the baseline: a real int4 CPU kernel also pays
    nibble unpack, which is not charged.
    """
    for cpu in M.ALL_CPUS:
        assert (cpu.peak_flops_per_cycle("int4")
                == cpu.peak_flops_per_cycle("int8"))


# ---------------------------------------------------------------------------
# The SLM row's KV cache
# ---------------------------------------------------------------------------

def test_slm_persistent_bytes_include_the_kv_cache():
    """The comment used to claim a term the expression did not contain."""
    b = M.slm_decode("t", params_b=1.1, hidden=2048, layers=22, seq=512,
                     rate=20, note="", prec="int8")
    weights = 1.1e9
    kv = 2 * 22 * 512 * 2048          # int8 K and V
    assert b.persistent_bytes == pytest.approx(weights + kv)
    assert b.persistent_bytes > weights, "KV cache is missing again"


def test_weight_only_int4_leaves_the_kv_cache_at_int8():
    """4-bit KV is a separate and much more accuracy-sensitive technique."""
    b8 = M.slm_decode("t8", 1.1, 2048, 22, 512, 20, "", prec="int8")
    b4 = M.slm_decode("t4", 1.1, 2048, 22, 512, 20, "", prec="int8",
                      weight_prec="int4")
    kv = 2 * 22 * 512 * 2048
    assert b4.persistent_bytes == pytest.approx(1.1e9 * 0.5 + kv)
    # Strictly MORE than half of the int8 footprint, because KV did not shrink.
    assert b4.persistent_bytes > 0.5 * b8.persistent_bytes
    # The attention GEMVs are the fraction held at int8.
    assert 0.0 < b4.solve_prec_frac < 0.1
    assert b4.operand_prec == "int4"


def test_uniform_precision_slm_row_is_unchanged_by_the_weight_prec_knob():
    a = M.slm_decode("a", 1.1, 2048, 22, 512, 20, "", prec="int8")
    b = M.slm_decode("b", 1.1, 2048, 22, 512, 20, "", prec="int8",
                     weight_prec="int8")
    assert a.operand_prec is None and b.operand_prec is None
    assert a.solve_prec_frac == 0.0 and b.solve_prec_frac == 0.0
    assert a.persistent_bytes == b.persistent_bytes


# ---------------------------------------------------------------------------
# THE NEGATIVE RESULT. This is the one a later change must not quietly move.
# ---------------------------------------------------------------------------

LLM_SIZES = (0.1, 0.16, 0.3, 0.5, 1.0, 1.1, 2.0, 3.0)


@pytest.mark.parametrize("wprec", ["int8", "int4"])
def test_large_model_llm_decode_is_never_differentiated(wprec):
    """NO point in the frozen plane puts a real LLM in the DIFFERENTIATED zone.

    "Real" means >=0.1 G parameters, which is already an order of magnitude below
    the smallest shipping SLM. The mechanism is structural, not incidental: at
    batch 1 every weight is fetched from off chip and used once, so external
    arithmetic intensity is 1/bytes_per_weight and DRAM energy dominates BOTH
    sides of the comparison.

    IF THIS TEST FAILS, DO NOT RELAX IT. Find out which term changed, and check
    it against the two things that could legitimately move the result: a batch
    dimension (which the model cannot currently express) or a measured NPU
    baseline (which would replace the metric entirely).
    """
    for params in LLM_SIZES:
        b = E.slm(params, 20, weight_prec=wprec)
        for gbs in E.BW_POINTS:
            c = E.cfg_bw(gbs)
            z, r, pw = E.zone(b, c)
            assert z != E.Z_DIFF, (
                f"{params} G {wprec} at {gbs} GB/s now scores {pw:.2f}x and is "
                f"in the DIFFERENTIATED zone. This contradicts a recorded "
                f"finding (PHASE0_FINDINGS 8.6). Find the term that moved.")
            assert pw < M.KILL_CRITERION


def test_the_three_zones_are_all_reachable():
    """A classifier that cannot produce every label is not a classifier.

    DIFF* is reachable only by a resident workload; INFEASIBLE only by a large
    model at low bandwidth; the middle zone by the rest. If any becomes
    unreachable the plane has stopped being informative.
    """
    seen = set()
    for params in (1e-5, 0.1, 1.1, 3.0):
        for gbs in E.BW_POINTS:
            seen.add(E.zone(E.slm(params, 20, weight_prec="int8"),
                            E.cfg_bw(gbs))[0])
    assert seen == {E.Z_DIFF, E.Z_FEAS, E.Z_INF}


def test_int4_moves_the_feasibility_boundary_by_exactly_two():
    """Not more. int4 halves the bytes and that is the whole mechanism."""
    for gbs in E.BW_POINTS:
        c = E.cfg_bw(gbs)
        for params in LLM_SIZES:
            b8 = E.slm(params, 20, weight_prec="int8")
            b4 = E.slm(params * 2.0, 20, weight_prec="int4")
            # Twice the parameters at half the width is the same weight bytes;
            # the KV cache differs (it scales with hidden/layers), so allow a
            # band rather than equality.
            r8 = M.model_argmind(b8, c)["bytes_moved"]
            r4 = M.model_argmind(b4, c)["bytes_moved"]
            assert 0.8 < r4 / r8 < 1.5


def test_external_bandwidth_barely_moves_the_differentiation_boundary():
    """FEASIBILITY is a bandwidth question; DIFFERENTIATION is a residency one.

    The perf/W at a fixed model size must be nearly insensitive to external
    bandwidth, while the deadline verdict must be strongly sensitive to it. That
    asymmetry IS the two-boundary finding, and it is what makes "provision more
    bandwidth" the wrong answer to the differentiation question.
    """
    b = E.slm(1.1, 20, weight_prec="int8")
    pws = [E.perf_w(b, E.cfg_bw(g)) for g in E.BW_POINTS]
    assert max(pws) / min(pws) < 1.5, "perf/W became bandwidth-sensitive"
    lats = [M.model_argmind(b, E.cfg_bw(g))["latency_us"] for g in E.BW_POINTS]
    assert lats[0] / lats[-1] > 4.0, "latency stopped being bandwidth-bound"


def test_batch_one_int8_decode_has_external_intensity_of_exactly_one():
    """The definitional anchor of the whole analysis.

    One MAC per weight byte, because batch 1 gives no reuse. If this drifts, the
    kernel decomposition and the byte accounting have come apart.
    """
    b = E.slm(1.1, 20, weight_prec="int8")
    ii = E.external_intensity(b, E.cfg_bw(6.4))
    assert 0.98 < ii < 1.05


# ---------------------------------------------------------------------------
# The resident regime and the ML-SKU question
# ---------------------------------------------------------------------------

def test_bigger_scratchpad_still_costs_solver_energy():
    """§8's capacity finding must keep applying to the ML-SKU proposal.

    If capacity ever becomes free, the "no resident-weight ML SKU" verdict loses
    half its argument and must be re-run.
    """
    solvers = [b for b in M.BENCHMARKS if b.kind == "solver"]

    def e(kb):
        c = dc.replace(M.CFG_100, scratchpad_kb=kb)
        return sum(M.model_argmind(b, c)["energy_uj"] for b in solvers)

    seq = [e(kb) for kb in (192, 256, 384, 512, 1024, 2048, 4096)]
    assert all(a < b for a, b in zip(seq, seq[1:])), (
        "solver energy is no longer monotone increasing in capacity")


def test_no_plausible_scratchpad_makes_an_llm_resident():
    """The reach argument, independent of the solver-tax argument."""
    for kb in (192, 4096):
        fits_int4 = kb * 1024 / M.BYTES_PER_ELEM["int4"]
        # Feasible streaming envelope at the CM-100's own 6.4 GB/s, 20 tok/s.
        streamable = 6.4e9 / 20.0 / M.BYTES_PER_ELEM["int4"]
        assert fits_int4 < streamable / 10.0, (
            f"{kb} KB now reaches within 10x of the streaming envelope; the "
            f"resident-weight ML SKU verdict needs re-running")


# ---------------------------------------------------------------------------
# The hybrid loop
# ---------------------------------------------------------------------------

def test_cited_policy_shapes_are_the_ones_claimed():
    """Parameter counts must match the cited architectures, not drift."""
    assert E.POLICY_STD.hidden == (512, 256, 128)      # Rudin et al. CoRL 2021
    assert E.POLICY_SMALL.hidden == (256, 128, 32)     # Hwangbo et al. 2019
    assert 185_000 < E.POLICY_STD.params < 195_000
    assert 48_000 < E.POLICY_SMALL.params < 52_000
    for p in (E.POLICY_STD, E.POLICY_SMALL):
        assert p.obs == 48 and p.act == 12
        assert "arXiv" in p.cite


def test_a_batch_one_policy_network_is_gemv_not_gemm():
    """The settled batch-1 finding applies to a policy net exactly as to a solver.

    If a policy row ever acquires GEMM MACs, its bandwidth-bound conclusion (and
    the claim that the MAC array is idle) no longer follows.
    """
    b = E.policy_bench(E.POLICY_STD, 1000)
    assert b.gemm_macs_per_iter == 0.0
    assert b.gemv_macs_per_iter == float(E.POLICY_STD.macs)
    M.assert_kernels_consistent(b)


def test_hybrid_footprint_fits_192kb_only_for_the_smaller_policy():
    """The capacity requirement, locked as a requirement.

    Policy A + QUAD-MPC fits 192 KB with ~1% margin. Policy B does not fit at
    int8 OR int4. If a change makes policy B fit 192 KB, the 256 KB requirement
    in PHASE0_FINDINGS 8.6 is no longer justified and must be withdrawn.
    """
    small8 = E.POLICY_SMALL.bytes_at("int8") / 1024 + E.QUAD_FOOTPRINT_KB
    std8 = E.POLICY_STD.bytes_at("int8") / 1024 + E.QUAD_FOOTPRINT_KB
    std4 = E.POLICY_STD.bytes_at("int4") / 1024 + E.QUAD_FOOTPRINT_KB
    assert small8 <= 192
    assert (192 - small8) / 192 < 0.05, "the margin is supposed to be thin"
    assert std8 > 192 and std8 <= 384
    assert std4 > 192 and std4 <= 256


def test_hybrid_fit_depends_on_the_sparse_A_optimisation():
    """A fragility that must stay visible.

    138.8 KiB is the QUAD-MPC plan WITH A = [I;Gamma] exploited. At the
    dense-A measured plan (163.8 KiB) even the smaller policy overflows 192 KB.
    The product claim therefore rests on a codegen optimisation, which is why
    the stated requirement is 256 KB.
    """
    dense = 163.8 + E.POLICY_SMALL.bytes_at("int8") / 1024
    assert dense > 192


def test_the_network_is_a_small_fraction_of_the_hybrid_tick():
    """Deflation guard: the hybrid loop's advantage is the SOLVER's advantage.

    If the policy's share of the tick ever exceeds ~25%, the argument that fusing
    it adds no new efficiency claim stops holding and section 4's conclusion must
    be re-derived.
    """
    quad = next(b for b in M.BENCHMARKS if b.name == "QUAD-MPC")
    for p in (E.POLICY_SMALL, E.POLICY_STD):
        pb = E.policy_bench(p, 1000)
        rp = M.model_argmind(pb, M.CFG_100)
        rq = M.model_argmind(quad, M.CFG_100)
        share = rp["compute_us"] / (rp["compute_us"] + rq["compute_us"])
        assert share < 0.25


def test_a_perception_network_cannot_be_in_a_1khz_tick():
    """And it fails on the ARRAY, not on capacity or bandwidth.

    This is what bounds the hybrid-loop claim to MLP-scale policies. If a change
    makes a ViT frame fit 1 ms, the scope statement in section 4d is wrong.
    """
    vit = next(b for b in M.BENCHMARKS if b.name == "VIT-TINY")
    r = M.model_argmind(vit, M.CFG_100)
    assert r["compute_us"] > 1000.0
    # Compute, not transfer, is what breaks it.
    assert r["compute_us"] > 5.0 * max(r["transfer_us"], 1.0)


# ---------------------------------------------------------------------------
# The baseline caveat has to stay attached to the numbers
# ---------------------------------------------------------------------------

def test_the_zone_label_carries_its_caveat_marker():
    """DIFF* is spelled with the asterisk on purpose: it is vs a CPU baseline.

    A label that reads as an unqualified "differentiated" is how an ML perf/W
    figure escapes its caveat, and this study's central honesty constraint is
    that it must not.
    """
    assert E.Z_DIFF.endswith("*")


def test_no_measured_npu_baseline_exists_in_the_model():
    """There is no NPU CpuCfg, and there must not be a modelled stand-in.

    Inventing one would be comparing our model against our own guess about a
    competitor, which is the failure mode "never present an estimate as a
    competitor's spec" exists to prevent.
    """
    names = " ".join(c.name.lower() for c in M.ALL_CPUS)
    assert "npu" not in names
    assert "ethos" not in names and "ceva" not in names
