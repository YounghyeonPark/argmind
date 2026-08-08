# SPDX-License-Identifier: Apache-2.0
"""
Invariant tests for the scaling analysis (perf_model/scaling_study.py).
======================================================================

WHY THIS FILE EXISTS

The scaling analysis added two axes the model previously could not express, and
BOTH omissions were flattering:

 13. `freq_mhz` scaled latency as 1/f but left energy per operation untouched, so
     the model reported 2 GHz as 2.5x the speed at LOWER energy per call. There
     was no supply-voltage term, hence no wall on the frequency axis.
 14. The longest single-cycle-read bitline was a hard constant (ROWS_MAX_800MHZ),
     so raising the clock cost nothing in scratchpad area either.

Both are now charged, and the tests below fail if either becomes free again.

The rest are the "model must be able to fail" invariants for the scaling
conclusions themselves. Every one of them can report a NEGATIVE result about the
architecture, and several do:

  * frequency must be able to say "faster is less efficient"
  * capacity must be able to say "more capacity costs more energy"
  * multi-CM must be able to say "two CMs are slower than one"
  * the scale-down sweep must be able to say "use the CPU"
  * a SKU must be able to say "this working set does not fit"

As elsewhere in this suite: relationships and directions, not pinned numbers.
"""

import dataclasses

import pytest

import argmind_model as M
import scaling_study as SS
import sram_cost as S


# ---------------------------------------------------------------------------
# Neutrality at the nominal operating point
# ---------------------------------------------------------------------------

def test_new_axes_are_exactly_neutral_at_800mhz():
    """The two additions must not silently move any established figure.

    Every headline number in PHASE0_FINDINGS is quoted at the CM-100
    configuration, which runs at 800 MHz. If either new term is off by even a
    percent there, dozens of published-internally figures shift for a reason
    unrelated to the change.
    """
    assert S.rows_max_for_freq(800.0) == S.ROWS_MAX_800MHZ
    assert M.vdd_for_freq(800.0) == pytest.approx(M.V_NOM_16NM)
    assert M.CFG_100.dyn_energy_scale == pytest.approx(1.0)
    assert M.CFG_100.leak_scale == pytest.approx(1.0)


def test_bank_geometry_default_rows_max_unchanged():
    """`bank_geometry` gained a rows_max parameter; the default must be the old
    hard constant, or every pre-existing area figure moves."""
    for kb, wb in ((12.0, 64), (8.0, 64), (16.0, 128)):
        a = S.bank_geometry(kb, wb)
        b = S.bank_geometry(kb, wb, rows_max=S.ROWS_MAX_800MHZ)
        assert a.area_um2 == pytest.approx(b.area_um2)


# ---------------------------------------------------------------------------
# ERROR 13 — frequency must not be a free energy win
# ---------------------------------------------------------------------------

def test_vdd_rises_with_frequency():
    vs = [M.vdd_for_freq(f) for f in (400, 600, 800, 1000, 1200, 1600, 2000)]
    assert all(b >= a for a, b in zip(vs, vs[1:])), vs
    assert vs[-1] > vs[0]


def test_faster_clock_costs_energy_per_call():
    """THE anti-flattery test for the frequency axis.

    Before the supply-voltage term, pushing the clock up reduced reported energy
    per call (shorter window -> less integrated leakage) with no offsetting cost.
    A model in which going faster is unambiguously better cannot be used to
    choose an operating point.
    """
    slow = SS.solver_energy(SS.cfg(freq=800, drain=8))
    fast = SS.solver_energy(SS.cfg(freq=1600, drain=14))
    assert fast > slow, (
        f"solver energy at 1600 MHz ({fast:.2f} uJ) must exceed 800 MHz "
        f"({slow:.2f} uJ): dynamic energy goes as V^2")


def test_frequency_still_buys_latency_but_sublinearly():
    """It must remain a real latency axis -- the fix must not overshoot into
    making frequency useless -- but the fixed per-call terms mean a 2x clock
    cannot give a 2x latency win."""
    q800 = SS.wc(SS.QUAD, SS.cfg(freq=800, drain=8))["latency_us"]
    q1600 = SS.wc(SS.QUAD, SS.cfg(freq=1600, drain=14))["latency_us"]
    assert q1600 < q800
    assert q800 / q1600 < 2.0


# ---------------------------------------------------------------------------
# ERROR 14 — a faster clock must not make the scratchpad cheaper
# ---------------------------------------------------------------------------

def test_rows_max_falls_with_frequency():
    rs = [S.rows_max_for_freq(f) for f in (400, 800, 1200, 1600, 2000)]
    assert all(b <= a for a, b in zip(rs, rs[1:])), rs


def test_faster_clock_never_shrinks_the_scratchpad():
    areas = [SS.cfg(freq=f).spad.area_mm2()["total_mm2"]
             for f in (400, 800, 1200, 1600)]
    assert all(b >= a - 1e-9 for a, b in zip(areas, areas[1:])), areas
    # And it must actually bite somewhere in the range, or the axis is still free.
    assert areas[-1] > areas[1] * 1.05


# ---------------------------------------------------------------------------
# The array/bandwidth balance rule
# ---------------------------------------------------------------------------

def test_balance_rule_matches_the_model():
    """`min_tile_for_bw` is a closed form; the swept model must agree with it.

    This is the result that makes "16x more MACs bought zero" a statement about
    64-128 B/cycle rather than about arrays in general. If the closed form and
    the sweep disagree, one of them is wrong and the sizing argument is unsafe.
    """
    for bw in (64, 128, 256, 512):
        need = SS.min_tile_for_bw(bw)
        lanes = max(4, int(bw // 8))
        below = SS.wc(SS.QUAD, SS.cfg(bw=bw, tile=max(4, int(need // 2)),
                                      lanes=lanes))["latency_us"]
        at = SS.wc(SS.QUAD, SS.cfg(bw=bw, tile=int(need) + 4,
                                   lanes=lanes))["latency_us"]
        assert at <= below + 1e-6, (
            f"at {bw} B/cycle a tile above sqrt(bw/2)={need:.1f} must not be "
            f"slower than one below it")


def test_balance_rule_is_precision_invariant():
    """The GEMV balance point is a property of the array, not of the format,
    because this datapath's rate multiplier is ~1/element-size."""
    ceilings = [SS.bw_ceiling_for_tile(12, p) for p in ("fp32", "bf16", "int8")]
    assert max(ceilings) == pytest.approx(min(ceilings))


def test_more_macs_does_nothing_at_cm100_bandwidth():
    """The settled finding, re-asserted at the CORRECTED 192 KB configuration.

    A 16x MAC increase at 128 B/cycle must buy essentially nothing on the
    GEMV-bound flagship row.
    """
    lat = [SS.wc(SS.QUAD, SS.cfg(bw=128, tile=t, lanes=16))["latency_us"]
           for t in (8, 12, 16, 24, 32)]
    assert max(lat) / min(lat) < 1.02, lat


# ---------------------------------------------------------------------------
# Capacity is not a scaling axis
# ---------------------------------------------------------------------------

def test_capacity_energy_is_monotone_increasing():
    """Capacity must never look like a free win.

    Bigger banks raise the cost of every access and the byte count a workload
    fetches does not change. If this ever reports energy FALLING with capacity,
    the bank energy model has been broken -- and the 128 -> 192 KB correction was
    only defensible because its cost was visible.
    """
    es = [SS.solver_energy(SS.cfg(kb=kb))
          for kb in (32, 64, 128, 192, 256, 384, 512)]
    assert all(b > a for a, b in zip(es, es[1:])), es


def test_making_arm_mpc_fit_costs_more_energy_than_it_saves():
    """The counter-intuitive result, pinned as a direction.

    ARM-MPC overflows 192 KB and fits 384 KB, yet grows more expensive when it
    fits: the pJ/byte penalty on every access exceeds the DRAM traffic saved.
    This is why capacity is sized to the FASTEST resident loop, not the biggest.
    """
    small = M.model_argmind(SS.ARM, SS.cfg(kb=192))
    big = M.model_argmind(SS.ARM, SS.cfg(kb=384))
    assert small["bytes_moved"] > big["bytes_moved"]      # it really does fit
    assert big["energy_uj"] > small["energy_uj"]


# ---------------------------------------------------------------------------
# Sequencer: nothing scales up
# ---------------------------------------------------------------------------

def test_issue_width_buys_nothing():
    base = SS.wc(SS.ARM, SS.cfg())["latency_us"]
    for w in (2.0, 4.0, 8.0):
        c = dataclasses.replace(SS.cfg(), seq_issue_width=w)
        assert SS.wc(SS.ARM, c)["latency_us"] == pytest.approx(base)


def test_deeper_pipeline_only_ever_costs():
    lat = [SS.wc(SS.ARM, SS.cfg(drain=d))["latency_us"] for d in (2, 4, 8, 16, 32)]
    assert all(b > a for a, b in zip(lat, lat[1:])), lat


def test_chain_blocking_amplifies_the_pipeline_penalty():
    """Raising the clock and skipping chain blocking are NOT independent
    decisions: without chain blocking a deeper pipeline is catastrophic."""
    with_cb = [SS.wc(SS.QUAD, SS.cfg(drain=d))["latency_us"] for d in (4, 32)]
    without = [SS.wc(SS.QUAD, SS.cfg(drain=d, chain=1))["latency_us"]
               for d in (4, 32)]
    assert (without[1] / without[0]) > 2.0 * (with_cb[1] / with_cb[0])


# ---------------------------------------------------------------------------
# Scaling down: the model must be able to say "use the CPU"
# ---------------------------------------------------------------------------

def test_offload_floor_reports_unreachable_when_datapath_loses():
    """If the CM's datapath is not faster than the host at all (k <= 1), no
    launch protocol can rescue it and the floor must be reported as infinite --
    not as some large finite number a salesperson could argue with."""
    tiny = M.ekf("EKF n6", n=6, m=3, rate=1000, note="")
    f = SS.offload_floor_us(tiny, M.CFG_100)
    assert f["k"] <= 1.0
    assert f["floor_cpu_us"] == float("inf")
    assert f["speedup"] < 1.0


def test_scale_down_sweep_contains_losses_and_wins():
    """The sweep must straddle the crossover. A sweep that only wins is not
    evidence, and one that only loses would mean the part has no socket."""
    wins, losses = 0, 0
    for n in (6, 12, 24, 48, 96):
        b = M.ekf(f"EKF n{n}", n=n, m=max(2, n // 2), rate=1000, note="")
        if SS.offload_floor_us(b, M.CFG_100)["speedup"] >= 1.0:
            wins += 1
        else:
            losses += 1
    assert wins > 0 and losses > 0


def test_mcu_class_launch_lowers_the_floor_but_does_not_remove_it():
    rich = SS.offload_floor_us(SS.FCS, SS.cfg(tier="T1"))
    mcu = SS.offload_floor_us(SS.FCS, SS.cfg(tier="T1M"))
    assert mcu["floor_cpu_us"] < rich["floor_cpu_us"]
    assert mcu["speedup"] > rich["speedup"]
    # FCS-MPC must remain below parity even on an MCU-class doorbell: that is
    # the settled 0.62x finding, and its cause is invocation, not the datapath.
    assert mcu["speedup"] < 1.0
    assert rich["k"] > 1.0


# ---------------------------------------------------------------------------
# External bandwidth
# ---------------------------------------------------------------------------

def test_slm_decode_is_untouched_by_every_on_chip_axis():
    """A row whose binding resource is off-chip must not respond to array,
    bandwidth, capacity or lane changes. If it ever does, the ML envelope
    statement is wrong."""
    slm = next(b for b in M.BENCHMARKS if b.name == "SLM-DECODE")
    base = M.model_argmind(slm, SS.cfg())["latency_us"]
    for c in (SS.cfg(tile=32), SS.cfg(bw=512), SS.cfg(kb=512), SS.cfg(bw=256)):
        assert M.model_argmind(slm, c)["latency_us"] == pytest.approx(base, rel=0.02)


# ---------------------------------------------------------------------------
# Multi-CM: the model must be able to say replication loses
# ---------------------------------------------------------------------------

def test_two_cms_lose_on_a_loop_carried_solve():
    """Splitting one ADMM solve across two CMs must be reported as a LOSS at
    realistic barrier costs. The iteration is loop-carried and the split pays a
    barrier per chain step; a model that reported a win here would be ignoring
    the premise of the architecture."""
    for sync in ("tightly_cpl", "coherent"):
        r = SS.split_solve(SS.QUAD, M.CFG_100, 2, sync, SS.ADMM_CHAIN_STEPS)
        assert r["speedup"] < 1.0, (sync, r["speedup"])


def test_one_wide_cm_beats_two_narrow_ones_on_area_and_latency():
    import cm100_sizing as C
    a1 = C.area_breakdown(M.CFG_100)["total"]
    wide = SS.cfg(bw=256, lanes=32)
    a2 = C.area_breakdown(wide)["total"]
    best_multi = SS.split_solve(SS.QUAD, M.CFG_100, 2, "shared_spad",
                                SS.ADMM_CHAIN_STEPS)["split_us"]
    assert a2 < 2 * a1
    assert SS.wc(SS.QUAD, wide)["latency_us"] < best_multi


def test_consolidation_is_capacity_bound_not_throughput_bound():
    """Provisioning a shared CM from duty cycle alone would oversell
    consolidation. The capacity limit must be the tighter one on the flagship."""
    by_duty = 1.0 / SS.duty_cycle(SS.QUAD, M.CFG_100)
    by_cap = M.CFG_100.scratchpad_kb / SS.FOOTPRINT_KB["QUAD-MPC"]
    assert by_cap < by_duty
    assert by_cap < 2.0      # one flagship loop per CM-100, not several


# ---------------------------------------------------------------------------
# SKU family
# ---------------------------------------------------------------------------

def test_cm50_does_not_hold_the_flagship_working_set():
    """The scoping line that must stay true, or CM-50 is being oversold.

    If someone grows CM-50's scratchpad to make this pass, the SKU has silently
    become a second flagship and the two-SKU rationale is gone.
    """
    cm50 = SS.cfg(bw=64, tile=8, lanes=8, kb=64, fp64=4)
    assert SS.FOOTPRINT_KB["QUAD-MPC"] > cm50.scratchpad_kb


def test_cm50_is_substantially_cheaper_than_cm100():
    import cm100_sizing as C
    cm50 = SS.cfg(bw=64, tile=8, lanes=8, kb=64, fp64=4)
    assert (C.area_breakdown(cm50)["total"]
            < 0.65 * C.area_breakdown(M.CFG_100)["total"])


def test_both_skus_keep_fp64():
    """A solver SKU cannot drop fp64: interior-point KKT conditioning reaches
    ~1e12 and multi-pass emulation costs 49-361 passes."""
    for c in (M.CFG_100, SS.cfg(bw=64, tile=8, lanes=8, kb=64, fp64=4)):
        assert c.fp64_fmas >= 1
        assert c.macs_per_cycle("fp64") >= 1


def test_fp64_throughput_is_free_below_half_the_lane_count():
    """The merged multi-format cluster is sized by max(fp64_fmas, lanes/2), so
    interior-point capability is a one-off merge premium rather than a per-unit
    cost. Pinned because it is the argument for keeping fp64 in the small SKU."""
    import cm100_sizing as C
    areas = [C.area_breakdown(SS.cfg(fp64=n, lanes=16))["total"]
             for n in (1, 2, 4, 8)]
    assert max(areas) == pytest.approx(min(areas))
    assert C.area_breakdown(SS.cfg(fp64=16, lanes=16))["total"] > max(areas)


def test_solver_only_tiny_array_is_worse_where_it_matters():
    """The rejected third SKU must be shown to be rejected for a REASON that the
    model can still produce: it is much worse on EKF, a beachhead workload."""
    ekf96 = next(b for b in M.BENCHMARKS if b.name == "EKF-96")
    big = M.model_argmind(ekf96, SS.cfg(tile=12))["energy_uj"]
    small = M.model_argmind(ekf96, SS.cfg(tile=4))["energy_uj"]
    assert small > 2.0 * big


# ---------------------------------------------------------------------------
# The band itself
# ---------------------------------------------------------------------------

def test_bandwidth_return_collapses_past_the_array_balance_point():
    """The bandwidth wall must be reported as a wall.

    Per-doubling return should stay near-linear up to the point where the fp32
    GEMV ceiling crosses the array rate (288 B/cycle for 144 MACs), then die.
    """
    def q(bw):
        return SS.wc(SS.QUAD, SS.cfg(bw=bw))["latency_us"]
    below = q(128) / q(256)
    above = q(256) / q(512)
    assert below > 1.7
    assert above < 1.3
    assert below > above


def test_only_two_axes_scale_by_more_than_3x():
    """The narrow-band conclusion, mechanised.

    Over each axis's defensible range, only scratchpad bandwidth and the fp64
    FMA cluster move a benchmark by more than 3x. If a third axis ever clears
    this bar the scaling story has genuinely changed and the SKU family should
    be revisited -- so this test is a tripwire, not a ceiling.
    """
    spans = {}
    spans["bandwidth"] = (SS.wc(SS.QUAD, SS.cfg(bw=64))["latency_us"]
                          / SS.wc(SS.QUAD, SS.cfg(bw=256))["latency_us"])
    spans["fp64"] = (SS.wc(SS.ARM, SS.cfg(fp64=4, lanes=16))["latency_us"]
                     / SS.wc(SS.ARM, SS.cfg(fp64=16, lanes=32))["latency_us"])
    spans["array"] = (SS.wc(SS.QUAD, SS.cfg(tile=8))["latency_us"]
                      / SS.wc(SS.QUAD, SS.cfg(tile=16))["latency_us"])
    spans["capacity"] = (SS.solver_energy(SS.cfg(kb=512))
                         / SS.solver_energy(SS.cfg(kb=64)))
    spans["frequency"] = (SS.wc(SS.QUAD, SS.cfg(freq=600, drain=7))["latency_us"]
                          / SS.wc(SS.QUAD, SS.cfg(freq=1000, drain=9))["latency_us"])
    big = {k for k, v in spans.items() if v > 3.0}
    assert big == {"bandwidth", "fp64"}, spans
