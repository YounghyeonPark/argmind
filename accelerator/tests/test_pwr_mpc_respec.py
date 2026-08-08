# SPDX-License-Identifier: Apache-2.0
"""
Invariants for the PWR-MPC resolution (2026-07-30).
===================================================

Three model extensions landed together and each can go slack in a specific way:

  * The FCS-MPC benchmark and the GEMM bandwidth ceiling. The failure mode is a
    flattering one: FCS-MPC's candidate set is a real batch dimension, and it is
    very easy to let the model treat that batch as tile-depth reuse, at which
    point the row becomes array-bound, looks fast, and reports the wrong binding
    resource. Tests pin the arithmetic intensity and the binding resource.

  * Mixed-precision operands. The failure mode is that `solve_prec_frac` drifts
    away from the actual chained-op share, so the model narrows the numerically
    sensitive triangular solve while claiming it did not.

  * MCU-class host tiers. The failure mode is the one being corrected here: an
    MCU-class T0 charged at rich-host prices, which is how zero-copy came to be
    credited with 1.9x in a socket that has no DRAM.

And one negative result to protect: the re-specified PWR-MPC row is SLOWER than
the fastest CPU baseline. That is the honest answer and it must not be quietly
optimised away by a later model change.
"""

import dataclasses as dc

import pytest

import argmind_model as M


REC = M.CFG_100
ALL = M.BENCHMARKS + M.OFF_SET


def by_name(name):
    return next(b for b in ALL if b.name == name)


# ---------------------------------------------------------------------------
# FCS-MPC: the re-specified row
# ---------------------------------------------------------------------------

def test_the_respecified_row_is_allowed_to_lose():
    """THE NEGATIVE RESULT, pinned.

    The re-specified PWR-MPC is ~1.6x SLOWER end-to-end than the fastest CPU
    baseline, because a 0.81 us tick cannot absorb a 0.80 us invocation cost. If
    this ever flips to a win, either the invocation overhead was reduced (say so
    explicitly, it is a [EST] and load-bearing) or the workload grew. Both need a
    human, because the surviving pitch for this socket is energy and offload, and
    it would be replaced by a latency claim the analysis does not support.
    """
    b = by_name("PWR-MPC")
    cm = M.model_argmind(b, REC)["latency_us"]
    cpu = min(M.model_cpu(b, c)["latency_us"] for c in M.ALL_CPUS)
    assert cpu / cm < 1.0, (
        f"the re-specified PWR-MPC is now {cpu / cm:.2f}x FASTER than the "
        f"fastest CPU. The reported conclusion was 0.62x — a latency win here "
        f"would be a new claim, not a refinement")
    # And the reason must still be the invocation overhead, not the datapath.
    compute = M.model_argmind(b, REC)["compute_us"]
    assert cpu / compute > 1.0, (
        "the CM datapath is no longer faster than the CPU on this row — the "
        "'we compute it faster and then pay to be asked' framing is wrong")


def test_one_step_fcs_is_a_cpu_win():
    """The crossover / minimum-viable-problem-size result.

    One-step FCS-MPC (27 candidates) is what MCUs ship today, and the honest
    answer there is 'use the MCU'. A model in which even this is an accelerator
    win has lost the ability to report a minimum viable problem size.
    """
    b = by_name("PWR-FCS1")
    cm = M.model_argmind(b, REC)["latency_us"]
    cpu = min(M.model_cpu(b, c)["latency_us"] for c in M.ALL_CPUS)
    assert cpu / cm < 0.5, (
        f"one-step FCS-MPC now scores {cpu / cm:.2f}x — the model no longer "
        f"reports a minimum viable problem size on this socket")
    # It must also fail the codegen design rule, which is the other half of why
    # it is the wrong socket.
    d = M.descriptor_cost(b, REC)
    assert d["cycles_per_desc"] < 2 * REC.seq_setup_cycles, (
        "one-step FCS-MPC now satisfies the >=8-cycles-per-descriptor rule; the "
        "'too small to code-generate for' finding has gone")


def test_fcs_arithmetic_intensity_is_set_by_n_y_not_by_the_tile():
    """The single most flattering mistake available on this row.

    The dominant product is (Q^1/2 C) @ X, so each predicted state element is
    consumed n_y = 2 times. If the model ever charges tile-depth reuse instead,
    the row becomes array-bound and every conclusion about it inverts. Pinned as
    a relationship: intensity must be INDEPENDENT of the tile dimension.
    """
    b = by_name("PWR-MPC")
    bpm = b.gemm_bytes_per_mac
    assert bpm is not None
    # A tile-limited GEMM would fetch elem*(1/Tm + 1/Tn); this row must be well
    # above that, and INDEPENDENT of the tile, because n_y is what limits it.
    for tile in (8, 12, 24, 48):
        c = dc.replace(REC, tile=tile, name=f"t{tile}")
        assert bpm > 2.0 * c.gemm_operand_bytes_per_mac(b.prec), (
            f"at tile={tile} FCS intensity {bpm:.2f} is within 2x of the "
            f"tile-limited figure — the n_y-limited reuse depth has been lost")
    # At and above the recommended array size the bandwidth ceiling must bind.
    for tile in (12, 24, 48):
        c = dc.replace(REC, tile=tile, name=f"t{tile}")
        array = c.macs_per_cycle(b.prec) * c.gemm_util(b.prec,
                                                      M.representative_dim(b))
        assert c.scratchpad_bw_bpc / bpm < array, (
            f"at tile={tile} the FCS GEMM became array-bound; arithmetic "
            f"intensity is no longer the binding term")


def test_the_fcs_row_becomes_bandwidth_bound_at_the_recommended_array_size():
    """A crossover, reported as a crossover rather than as a win ratio.

    An earlier version of this test asserted "more MACs never helps FCS-MPC" and
    FAILED, correctly: at tile=8 the fp32 array rate is BELOW the intensity-
    limited bandwidth rate, so the row really is array-bound there and 8->12 buys
    1.14x. Above 12x12 the array is irrelevant. **144 MACs turns out to be the
    knee for FCS-MPC as well as the energy-optimal point for the EKF rows**, which
    is a useful coincidence and the reason to state the crossover rather than
    round it into the settled "MACs never help solvers" claim.
    """
    b = by_name("PWR-MPC")
    lat = {t: M.model_argmind(b, dc.replace(REC, tile=t, name=f"t{t}"))
           ["compute_us"] for t in (8, 12, 16, 24, 48)}
    assert lat[8] > 1.05 * lat[12], (
        "tile=8 is no longer array-bound on FCS-MPC — the crossover has moved "
        "and the '144 MACs is the knee' statement needs re-deriving")
    for t in (16, 24, 48):
        assert lat[t] >= 0.995 * lat[12], (
            f"tile={t} beats tile=12 on FCS-MPC ({lat[t]:.3f} vs "
            f"{lat[12]:.3f} us) — the row is no longer intensity-limited above "
            f"the recommended array size")


def test_fcs_has_no_iteration_tail():
    for b in (by_name("PWR-MPC"), by_name("PWR-FCS1"), by_name("PWR-FAST")):
        assert b.iters == b.iters_worst == 1
        assert b.serial_scalar_per_iter == 0.0, (
            f"{b.name}: FCS-MPC has no convergence test and no step size, so "
            f"there is no loop-carried scalar chain")


def test_accumulator_fusion_only_helps_the_cm_side():
    """The fused variant must be scored CM-vs-unfused-CPU.

    Guards the mistake documented in the study: PWR-FUSE has fewer vector ops, so
    a naive comparison hands the CPU an accumulator it does not have.
    """
    fcs, fuse = by_name("PWR-MPC"), by_name("PWR-FUSE")
    assert fuse.vector_ops_per_iter < fcs.vector_ops_per_iter
    cm_fused = M.model_argmind(fuse, REC)["latency_us"]
    cm_plain = M.model_argmind(fcs, REC)["latency_us"]
    assert cm_fused < cm_plain, "accumulator fusion must reduce CM latency"
    cpu_plain = min(M.model_cpu(fcs, c)["latency_us"] for c in M.ALL_CPUS)
    cpu_fused = min(M.model_cpu(fuse, c)["latency_us"] for c in M.ALL_CPUS)
    assert cpu_fused < cpu_plain, (
        "the CPU does not get faster on PWR-FUSE, so the 'score it against the "
        "unfused CPU' warning in the study is no longer needed — check why")
    # The honest score improves; it must not reach a win, since half the latency
    # is still invocation overhead.
    assert cpu_plain / cm_fused < 1.0


# ---------------------------------------------------------------------------
# Mixed-precision operands
# ---------------------------------------------------------------------------

def test_solve_prec_frac_matches_the_chained_kernel_share():
    """The consistency rule, applied to precision.

    `solve_prec_frac` says how much of the GEMV work is the numerically sensitive
    cached triangular solve. It must equal the chained ops' share of the kernel
    decomposition, or the model narrows the solve while claiming it did not --
    which would silently buy the 1.9e-3 error figure at a 6.1e-2 error cost.
    """
    for b in ALL:
        if b.algo != "admm" or not b.kernels:
            continue
        chained = sum(o.total_macs for o in b.kernels
                      if o.kind == "gemv" and o.chain_dim > 0)
        assert b.solve_prec_frac == pytest.approx(
            chained / b.gemv_macs_per_iter, rel=1e-9), (
            f"{b.name}: solve_prec_frac={b.solve_prec_frac:.4f} but the chained "
            f"GEMV ops are {chained / b.gemv_macs_per_iter:.4f} of the total")
        assert 0.0 < b.solve_prec_frac < 1.0


def test_fp16_operands_cut_latency_and_energy_but_not_by_2x():
    """Both directions matter.

    Narrowing operands must help (or the mixed-precision path is not wired up),
    and must help by LESS than 2x (or something other than the bandwidth term has
    been narrowed -- the vector unit and the dependency drain do not narrow).
    """
    kw = dict(n_x=12, n_u=4, N=20, iters=15, rate=1000, iters_worst=74)
    base = M.condensed_mpc("q32", note="", **kw)
    narrow = M.condensed_mpc("q16", note="", operand_prec="fp16", **kw)
    rb = M.model_argmind(base, REC)
    rn = M.model_argmind(narrow, REC)
    for key in ("latency_us", "energy_uj"):
        gain = rb[key] / rn[key]
        assert 1.05 < gain < 2.0, (
            f"fp16 operands change {key} by {gain:.2f}x — outside the "
            f"physically possible band for halving operand width")
    # Energy must gain MORE than latency: bytes halve and the multiplier shrinks,
    # and neither is capped by the vector unit.
    assert (rb["energy_uj"] / rn["energy_uj"]
            > rb["latency_us"] / rn["latency_us"]), (
        "fp16 now buys more latency than energy, which is backwards: the vector "
        "unit caps the latency gain and does not cap the energy gain")


def test_narrowing_the_solve_too_is_strictly_worse_numerically():
    """Encodes the measured C3 ordering as a model-side guard.

    precision_study measures fp16-operands/fp32-solve at 1.9e-3 and a low-
    precision SOLVE at 6.1e-2. The model has no error term, so what is asserted
    here is that the two configurations are DISTINGUISHABLE in the model -- if
    solve_prec_frac stops mattering, the model can no longer represent the
    difference between a safe and an unsafe narrowing.
    """
    kw = dict(n_x=4, n_u=3, N=10, iters=12, rate=20000, iters_worst=48)
    safe = M.condensed_mpc("safe", note="", operand_prec="fp16", **kw)
    unsafe = dc.replace(safe, solve_prec_frac=0.0)
    assert (M.model_argmind(unsafe, REC)["latency_us"]
            < M.model_argmind(safe, REC)["latency_us"]), (
        "narrowing the triangular solve no longer shows up in the model at all, "
        "so a codegen mode that does it cannot be distinguished from one that "
        "does not")


def test_operand_prec_none_is_bit_identical_to_before():
    """The mixed-precision path must not perturb any single-precision row."""
    for b in M.BENCHMARKS:
        assert b.operand_prec is None, (
            f"{b.name} in the FROZEN set now uses mixed-precision operands — "
            f"that is a benchmark re-specification, not a model change")
        r = M.model_argmind(b, REC)
        direct = b.gemv_macs_per_iter / REC.gemv_macs_per_cycle(
            b.prec, b.gemv_row_len)
        assert M.split_gemv_cycles(b, REC, b.gemv_macs_per_iter) == \
            pytest.approx(direct, rel=1e-12), b.name
        assert r["latency_us"] > 0


# ---------------------------------------------------------------------------
# MCU-class host tiers
# ---------------------------------------------------------------------------

def test_mcu_tiers_do_not_charge_dram_prices():
    """The error being corrected. An all-SRAM part has no LPDDR, no MMU and no
    cache to maintain, so neither the 25 us launch cost nor the 12 pJ/B transfer
    cost applies."""
    rich0, rich1 = dc.replace(REC, tier="T0"), dc.replace(REC, tier="T1")
    mcu0, mcu1 = dc.replace(REC, tier="T0M"), dc.replace(REC, tier="T1M")
    assert mcu0.call_overhead_us < 0.2 * rich0.call_overhead_us, (
        "the MCU-class copy tier still costs rich-host prices; buffer map and "
        "cache maintenance are being charged on a part with neither")
    assert mcu1.call_overhead_us < rich1.call_overhead_us, (
        "T1M should be CHEAPER than T1 — there is no coherence round trip")
    assert mcu0.offchip_pj_per_byte < 0.2 * rich0.offchip_pj_per_byte
    for c in (mcu0, mcu1):
        assert c.is_mcu_class
        # The counter-intuitive half: closer data, narrower pipe.
        assert c.dma_bw_gbs < rich0.dma_bw_gbs, (
            "an MCU-class on-chip bus is being modelled as WIDER than an LPDDR "
            "channel — that removes the adversarial finding")
    for c in (rich0, rich1):
        assert not c.is_mcu_class


def test_zero_copy_advantage_collapses_on_mcu_class():
    """The settled contradiction, encoded.

    On a rich host T1-over-T0 is worth order 10x on small high-rate rows. On an
    MCU-class host it must be far smaller, or CLAUDE.md's 'zero-copy is void in
    MCU class' concession is being contradicted by the model.
    """
    for name in ("PWR-MPC", "EKF-24"):
        b = by_name(name)
        def lat(t):
            return M.model_argmind(b, dc.replace(REC, tier=t))["latency_us"]
        def en(t):
            return M.model_argmind(b, dc.replace(REC, tier=t))["energy_uj"]
        rich = lat("T0") / lat("T1")
        mcu = lat("T0M") / lat("T1M")
        assert mcu < 0.5 * rich, (
            f"{name}: the tier advantage is {mcu:.2f}x on MCU class against "
            f"{rich:.2f}x on a rich host — it has not collapsed, so either the "
            f"MCU tiers are mis-specified or the concession is wrong")
        assert en("T0M") / en("T1M") < 0.5 * en("T0") / en("T1")


def test_the_surviving_mcu_advantage_is_invocation_not_data_movement():
    """The corrected pitch, encoded so it cannot silently revert.

    If the MCU-class T1M advantage ever becomes majority data movement again, the
    'low invocation overhead and determinism, not zero-copy' framing is wrong and
    CLAUDE.md needs changing back.
    """
    b = by_name("PWR-MPC")
    lat = {t: M.model_argmind(b, dc.replace(REC, tier=t))["latency_us"]
           for t in ("T0M", "T1M")}
    total = lat["T0M"] - lat["T1M"]
    overhead = (M.ArgmindCfg.CALL_OVERHEAD_US["T0M"]
                - M.ArgmindCfg.CALL_OVERHEAD_US["T1M"])
    assert total > 0
    assert overhead / total > 0.6, (
        f"only {100 * overhead / total:.0f}% of the MCU-class T1M advantage is "
        f"invocation overhead — the honest MCU pitch would have to go back to "
        f"data movement")


def test_every_row_still_meets_its_deadline_on_an_mcu_class_host():
    """A provisioning statement, not an architecture claim. If a row misses on
    T0M, say which and why -- do not let it pass silently."""
    misses = []
    for b in M.BENCHMARKS:
        if b.kind != "solver":
            continue
        lat = M.model_argmind(b, dc.replace(REC, tier="T0M"))["latency_us"]
        if lat > b.deadline_us:
            misses.append((b.name, lat, b.deadline_us))
    assert not misses, f"MCU-class copy tier misses: {misses}"


# ---------------------------------------------------------------------------
# The CPU-baseline SIMD correction
# ---------------------------------------------------------------------------

def test_simd_rate_correction_leaves_neon_baselines_untouched():
    """NEON has one set of units, so the elementwise/matrix split must be a
    no-op there. If it is not, the correction is doing something unintended."""
    for c in (M.CPU_A55, M.CPU_A78):
        assert c.simd_flops_per_cycle is None
        assert c.simd_rate == c.fp32_flops_per_cycle
    assert M.CPU_SME2.simd_rate < M.CPU_SME2.fp32_flops_per_cycle, (
        "SME2's elementwise rate is not below its outer-product rate — the ZA "
        "tile is being credited with min/max and horizontal reductions again")


def test_simd_rate_only_moves_vector_heavy_rows():
    """Bounds what the correction can and cannot explain.

    It must move the vector-dominated rows and leave the GEMM-dominated ones
    alone, or it is not the correction it claims to be.
    """
    def sme(simd):
        return dc.replace(M.CPU_SME2, simd_flops_per_cycle=simd)

    def ratio(b):
        return (M.model_cpu(b, sme(8.0))["latency_us"]
                / M.model_cpu(b, sme(64.0))["latency_us"])

    # EKF-96 is 99.9% matrix-matrix: the correction must barely touch it.
    assert ratio(by_name("EKF-96")) < 1.3, (
        "the elementwise-rate correction moves a GEMM-dominated row a lot — it "
        "is not isolated to elementwise work")
    # The FCS row is majority elementwise: it must move a lot.
    assert ratio(by_name("PWR-MPC")) > 2.0, (
        "the correction barely moves the most elementwise row in the set")


def test_the_parity_verdict_is_still_not_quotable_but_the_reason_changed():
    """The retired ADMM row's parity position, and an audit of why it moved.

    History. This test used to assert that the verdict STRADDLES 1.0 across the
    elementwise-rate band -- 0.97x at 64 flops/cycle, above parity at 32 -- so
    that neither side of it could be quoted. The check_term 25 -> 1 correction
    (2026-07-30) moved it: the row is now above parity at every rate in the band.

    That is a shift in our favour, so it gets audited rather than accepted. The
    mechanism is legitimate and is asserted below: checking convergence every
    iteration adds GEMV work (residual norms: Ax, Px, A'y) and no elementwise
    work, so the elementwise share of the row falls 21.8% -> 16.0% and the
    elementwise-rate assumption simply has less leverage than it did. Both sides
    of the comparison do the same arithmetic; nothing was removed from the CPU.

    The verdict is STILL NOT QUOTABLE, for a different and more durable reason:
    PWR-CCS is a RETIRED row. Continuous-control-set MPC is research-platform
    practice, not the deployed motor-drive formulation, which is finite control
    set (PWR-MPC, and that row is 0.62x -- genuinely below parity). And the
    pessimistic end of the band is only 1.06x, which is parity in all but name.
    """
    ccs = by_name("PWR-CCS")
    cm = M.model_argmind(ccs, REC)["latency_us"]
    speeds = {}
    for simd in (8.0, 32.0, 64.0):
        cpus = [dc.replace(c, simd_flops_per_cycle=simd) if c is M.CPU_SME2
                else c for c in M.ALL_CPUS]
        speeds[simd] = min(M.model_cpu(ccs, c)["latency_us"]
                           for c in cpus) / cm

    # The advantage is thin enough at the pessimistic end that it is parity in
    # all but name. If this ever exceeds ~1.3x at 64 flops/cycle, something has
    # changed materially and the row's status needs re-deciding deliberately.
    assert 1.0 < speeds[64.0] < 1.3, speeds
    assert speeds[8.0] > speeds[32.0] > speeds[64.0], speeds

    # The audit: the shift is explained by a falling elementwise share, not by
    # anything having been taken off the CPU side.
    at25 = M.condensed_mpc("c25", n_x=4, n_u=3, N=10, iters=12, rate=20000,
                           note="", check_term=25)
    share25 = at25.vector_ops_per_iter / (at25.vector_ops_per_iter
                                          + at25.gemv_macs_per_iter)
    share1 = ccs.vector_ops_per_iter / (ccs.vector_ops_per_iter
                                        + ccs.gemv_macs_per_iter)
    assert ccs.vector_ops_per_iter == at25.vector_ops_per_iter   # nothing removed
    assert ccs.gemv_macs_per_iter > at25.gemv_macs_per_iter      # only GEMV added
    assert share1 < share25
    assert 0.15 < share1 < 0.17 and 0.21 < share25 < 0.23

    # And the row that is actually in the frozen set is still below parity, so
    # the model has not lost the ability to report that we lose.
    fcs = by_name("PWR-MPC")
    fcs_speed = min(M.model_cpu(fcs, c)["latency_us"] for c in M.ALL_CPUS) \
        / M.model_argmind(fcs, REC)["latency_us"]
    assert fcs_speed < 1.0, fcs_speed
