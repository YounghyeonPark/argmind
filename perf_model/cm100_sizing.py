"""
CM-100 baseline SKU sizing — scratchpad bandwidth / capacity / MAC array
=======================================================================

Joins sram_cost.py (area + energy of the banked scratchpad) to
argmind_model.py (latency + energy on the frozen benchmark set) so the
provisioning decision is made against BOTH cost and deadlines.

RE-RUN 2026-07-30. The previous recommendation (128 B/cycle, 256 MACs, 256 KB)
rested on two claims that are both now dead:

  1. "ARM-MPC is the only latency-binding benchmark" (0.96x margin at 64 B/cycle,
     1.60x at 128; 64 MACs missed it, 144 was the minimum). ARM-MPC was
     re-specified as acados SQP-RTI + HPIPM and now sits at 24x margin. It binds
     nothing.
  2. "256 KB is needed to hold ARM-MPC's 861 KB / QUAD-MPC's 150 KB working set."
     Both footprints shrank once P was stored symmetric, L triangular, and A's
     identity block charged nothing.

With those gone, at TYPICAL (p95) iteration counts every solver row meets its
deadline at 16 B/cycle with >=3x margin -- i.e. the typical case cannot size
anything, and any sweep run against it will rubber-stamp whatever it is given.
This re-run therefore sizes feasibility against `iters_worst` (the measured cold
/ active-set-change tail) inside a real control-period budget, and sizes the
array against ENERGY rather than latency, because that is the only axis on which
it still pays.

All figures are MODELLED ESTIMATES, 16nm-class, +/-2x, NOT QUOTABLE.
"""

from __future__ import annotations

import dataclasses

import argmind_model as M
import sram_cost as S

# Sequencer 0.045: was a flat [EST]; costed bottom-up 2026-07-30 in sequencer.py
# as 0.0137 mm2 of logic (RV32IMC + 3 descriptor AGUs + hardware loop + 8-deep
# descriptor queue) plus 0.0143 mm2 for an 8 KB I-mem / 4 KB D-mem = 0.028 mm2,
# i.e. 62% of the allocation. Kept at 0.045 as headroom rather than reduced --
# the derivation confirms the number, it does not license shrinking it.
REST_MM2 = 0.045 + 0.070 + 0.030      # sequencer, DMA+AXI/ACE-Lite, misc [EST]
TOP_MARGIN = 1.15                      # top-level route/CTS/margin [EST]
A_GATE = S.A_GATE_UM2

# CORRECTED 2026-07-30. The old value was 2200 gates/lane, which is UNPHYSICAL:
# the parametric FMA model in sram_cost.py puts an fp32 *multiplier alone* at
# ~2.6 kgates, so 2200 could not hold one. Now derived from the same gate model
# as everything else: one IEEE fp32 FMA (9.55 kgates) + a share of the
# reciprocal / rsqrt / exp tables and the min/max/compare path.
# Effect: the vector unit goes 0.009 -> 0.055 mm2. This is a PRE-EXISTING
# UNDERESTIMATE, not an fp64 cost -- the two are reported separately below so
# neither hides in the other.
VEC_TRANSCENDENTAL_GATES = 2000        # [EST] recip/rsqrt tables + Newton, /lane
VEC_LANE_GATES = int(S.fma_gates_ieee(S.SIG_BITS["fp32"])
                     + VEC_TRANSCENDENTAL_GATES)          # ~11.5k [EST-CAL]

# The recommended fp64 implementation is NOT a separate cluster bolted on: it is
# a merged multi-format FMA cluster that IS the vector unit (1x fp64 or 2x fp32
# per unit). Setting this True prices the two as one block, which is both cheaper
# and removes the "dead area on 6 of 7 benchmarks" objection.
MERGE_FP64_WITH_VECTOR = True

# Fraction of the control period a hard-real-time integrator will actually give
# the solver. 100% of the period is not an engineering budget -- the tick also
# has to sample, run the estimator, do the safety checks and drive the actuator.
# 30-60% is the range quoted in practice; 50% is the central assumption and the
# conclusions below are given at both 50% and 100%.
PERIOD_BUDGET = 0.50


def banks_for(bw: float) -> int:
    """Bank count honouring the >=64-bit bank-word rule.

    sram_cost.macro_energy_pj_per_byte_45 has NO data below a 64-bit access
    width and clamps instead of turning back up, so any configuration with a
    narrower bank word produces an energy number that must not be believed.
    Capped at 16 banks: beyond that the read crossbar and per-bank queues grow
    faster than the energy benefit (ADVERSARIAL CHECK 3).
    """
    return max(1, min(int(bw // 8), 16))


def cfg(name, bw, tile=16, lanes=16, seg=True, kb=None, tile_n=None, nb=None,
        fp64_fmas=8):
    if kb is None:
        kb = M.CM100_SCRATCHPAD_KB
    return M.ArgmindCfg(name=name, tile=tile, tile_n=tile_n, vector_lanes=lanes,
                        scratchpad_kb=kb, scratchpad_bw_bpc=bw,
                        n_banks=nb if nb is not None else banks_for(bw),
                        segmented_reduction=seg, fp64_fmas=fp64_fmas)


def area_breakdown(c: M.ArgmindCfg, pessimistic: bool = False) -> dict:
    """Core area, mm2 at 16nm-class. MODELLED ESTIMATE, +/-2x, NOT QUOTABLE.

    `fp64` is reported as a SEPARATE LINE so the fp64 decision can be audited
    without unpicking the vector-unit correction made at the same time.
    """
    spad = c.spad.area_mm2()["total_mm2"]
    mac = S.mac_array_mm2(c.macs)
    if MERGE_FP64_WITH_VECTOR and c.fp64_fmas > 0:
        # One merged block. Width is set by whichever demand is larger:
        # fp64_fmas units, or vector_lanes/2 units (each does 2x fp32).
        units = max(c.fp64_fmas, (c.vector_lanes + 1) // 2)
        fma = S.fma_cluster_mm2(units, "fp64", pessimistic, multiformat=True)
        trans = units * 2 * VEC_TRANSCENDENTAL_GATES * A_GATE / 1e6
        # Attribution: what a vector-unit-only block of the same fp32 width
        # would have cost, vs the increment for making it fp64-capable.
        vec_only = ((c.vector_lanes * VEC_LANE_GATES) * A_GATE
                    * S.MAC_STAGING_OVERHEAD / 1e6)
        vec, fp64 = vec_only, max(0.0, fma + trans - vec_only)
    else:
        vec = c.vector_lanes * VEC_LANE_GATES * A_GATE * S.MAC_STAGING_OVERHEAD / 1e6
        fp64 = (S.fma_cluster_mm2(c.fp64_fmas, "fp64", pessimistic)
                if c.fp64_fmas > 0 else 0.0)
    sub = spad + mac + vec + fp64 + REST_MM2
    return {"spad": spad, "mac": mac, "vec": vec, "fp64": fp64,
            "rest": REST_MM2, "sub": sub, "total": sub * TOP_MARGIN}


# ---------------------------------------------------------------------------
# Which resource actually saturates first
# ---------------------------------------------------------------------------

def decompose(b: M.Benchmark, c: M.ArgmindCfg, iters: int | None = None) -> dict:
    """Latency broken down by the resource that produced it, in us.

    A peak figure is useless; the question is always which of these is largest.
    Keys are the physical resources, so the answer is directly actionable:
      gemv_bw   scratchpad read bandwidth (matrix-vector: no operand reuse)
      gemm_mac  MAC array throughput      (matrix-matrix: reuse available)
      vec       vector unit / transcendental lanes
      seq       sequencer DESCRIPTOR ISSUE rate (n_desc * setup / issue_width).
                REPLACED 2026-07-30: was `0.5 * scalar_ops_per_iter`, a flat
                constant with a fixed 50% overlap that could not express
                descriptor granularity at all. See sequencer.py.
      dma       external bandwidth (non-overlapped residue)
      host      per-call launch overhead
    """
    it = b.iters if iters is None else iters
    ns = c.cycle_ns / 1000.0
    dim = M.representative_dim(b)
    gemv_pc = c.gemv_macs_per_cycle(b.prec, b.gemv_row_len)
    if c.use_descriptor_seq and b.kernels:
        di = M.descriptor_cost(b, c)
        dc = M.descriptor_cost(b, c, per_call=True)
        seq_us = (di["issue_cycles"] * it + dc["issue_cycles"]) * ns
    else:
        seq_us = 0.5 * b.scalar_ops_per_iter / c.seq_ipc * it * ns
    d = {
        "gemv_bw": (b.gemv_macs_per_iter * it + b.gemv_macs_per_call) / gemv_pc * ns,
        "gemm_mac": (M.gemm_like_cycles(b.gemm_macs_per_iter, c, b.prec, dim) * it
                     + M.gemm_like_cycles(b.gemm_macs_per_call, c, b.prec, dim)) * ns,
        "vec": (b.vector_ops_per_iter / c.vector_lanes
                + b.transcendental_per_iter * 4.0 / c.vector_lanes) * it * ns,
        "seq": seq_us,
    }
    r = M.model_argmind(dataclasses.replace(b, iters=it), c)
    d["dma"] = r["transfer_us"]
    d["host"] = c.call_overhead_us
    d["_total"] = r["latency_us"]
    d["_bind"] = max((k for k in d if not k.startswith("_")), key=lambda k: d[k])
    return d


def evaluate(c: M.ArgmindCfg) -> dict:
    rows, worst, pw = {}, [], []
    for b in M.BENCHMARKS:
        r = M.model_argmind(b, c)
        wc = M.model_argmind(dataclasses.replace(b, iters=b.iters_wc), c)
        cpus = [(x, M.model_cpu(b, x)) for x in M.ALL_CPUS]
        eff = min(cpus, key=lambda t: t[1]["energy_uj"])[1]
        rows[b.name] = {
            "lat": r["latency_us"], "lat_wc": wc["latency_us"],
            "deadline": b.deadline_us,
            "meets": r["latency_us"] <= b.deadline_us,
            "meets_wc": wc["latency_us"] <= b.deadline_us,
            "meets_wc_budget": wc["latency_us"] <= b.deadline_us * PERIOD_BUDGET,
            "uj": r["energy_uj"], "pw": eff["energy_uj"] / r["energy_uj"],
            "mw": r["avg_power_mw"], "bind": decompose(b, c, b.iters_wc)["_bind"],
        }
        if not rows[b.name]["meets_wc"]:
            worst.append(b.name)
        if b.kind == "solver":
            pw.append(rows[b.name]["pw"])
    return {"rows": rows, "misses": worst,
            "solver_pw_min": min(pw), "solver_pw_max": max(pw),
            "solver_uj": sum(rows[b.name]["uj"] for b in M.BENCHMARKS
                             if b.kind == "solver"),
            "area": area_breakdown(c), "pj_b": c.sram_pj_per_byte()}


SOLVERS = [b for b in M.BENCHMARKS if b.kind == "solver"]
BW_SWEEP = (16, 32, 64, 128, 256)
MAC_SWEEP = (64, 144, 256, 1024)
TILE_OF = {64: 8, 144: 12, 256: 16, 1024: 32}


def main():
    L = []

    def out(s=""):
        print(s)
        L.append(s)

    out("=" * 108)
    out("CM-100 SIZING RE-RUN (2026-07-30) — bandwidth x MAC array x capacity")
    out("MODELLED ESTIMATE, 16nm-class, +/-2x, NOT QUOTABLE EXTERNALLY")
    out("=" * 108)
    out()
    out("Supersedes the 128 B/cycle / 256 MAC / 256 KB recommendation, whose two")
    out("stated grounds (ARM-MPC latency-binding; 861 KB working set) were both")
    out("artefacts of benchmark errors since corrected.")
    out()

    # ================================================================== Q1 ===
    out("=" * 108)
    out("Q1a — SWEEP AT TYPICAL (p95) ITERATIONS: proves the typical case sizes NOTHING")
    out("=" * 108)
    out("Tightest SOLVER deadline margin (deadline/latency) over the whole grid:")
    out(f"{'B/cyc':>6} {'banks':>6} {'word':>5} " +
        " ".join(f"{str(m) + 'MAC':>12}" for m in MAC_SWEEP))
    for bw in BW_SWEEP:
        nb = banks_for(bw)
        cells = []
        for macs in MAC_SWEEP:
            c = cfg("x", bw, TILE_OF[macs])
            tight = min(((b.name, b.deadline_us / M.model_argmind(b, c)["latency_us"])
                         for b in SOLVERS), key=lambda t: t[1])
            cells.append(f"{tight[1]:>7.1f}x {tight[0][:4]:<4}")
        out(f"{bw:>6} {nb:>6} {bw // nb:>4}B " + " ".join(f"{x:>12}" for x in cells))
    out()
    out("  EVERY cell meets EVERY solver deadline. The tightest point in the whole")
    out("  grid is 3.2x. A sweep run this way cannot produce a binding constraint,")
    out("  so it cannot justify a configuration -- it can only ratify one. That is")
    out("  what the previous run did once ARM-MPC's 23x overstatement is removed.")
    out()

    out("=" * 108)
    out("Q1b — SWEEP AT WORST-CASE ITERATIONS (the tail a hard deadline must survive)")
    out("=" * 108)
    out("iters_worst: QUAD-MPC 74 (MEASURED cold/max, coupled plant, eps=1e-6),")
    out("ARM-MPC 30 [EST, 2x design point; HPIPM max_iter bounds it], EKF 1 and")
    out("PWR-MPC 1 (fixed op count).")
    out()
    out("UPDATED 2026-07-30: PWR-MPC was re-specified from a continuous-control-set")
    out("ADMM QP to finite-control-set MPC (BENCHMARKS.md #3 v1.2), so it no longer")
    out("has an iteration tail at all and no longer constrains bandwidth. It used to")
    out("carry iters_worst=48 [EST] and was one of the TWO rows that set the")
    out("64 B/cycle floor. See perf_model/results_pwr_mpc_respec.txt.")
    out()
    out(f"{'B/cyc':>6} {'MACs':>5} {'area':>6} " +
        " ".join(f"{b.name[:8]:>9}" for b in SOLVERS) +
        f" {'binds':<10} {'100%period':<11} {'50%period':<11}")
    grid = {}
    for bw in BW_SWEEP:
        for macs in MAC_SWEEP:
            c = cfg("x", bw, TILE_OF[macs])
            e = evaluate(c)
            grid[(bw, macs)] = e
            cells, fail100, fail50 = [], [], []
            for b in SOLVERS:
                r = e["rows"][b.name]
                m = b.deadline_us / r["lat_wc"]
                cells.append(f"{m:>8.1f}" + ("X" if m < 1.0 else " "))
                if not r["meets_wc"]:
                    fail100.append(b.name[:4])
                if not r["meets_wc_budget"]:
                    fail50.append(b.name[:4])
            tight = min(SOLVERS, key=lambda b: b.deadline_us
                        / e["rows"][b.name]["lat_wc"])
            out(f"{bw:>6} {macs:>5} {e['area']['total']:>6.3f} "
                + " ".join(f"{x:>9}" for x in cells)
                + f" {e['rows'][tight.name]['bind']:<10} "
                f"{(','.join(fail100) or 'all OK'):<11} "
                f"{(','.join(fail50) or 'all OK'):<11}")
    out()
    out("  The binding resource is SCRATCHPAD BANDWIDTH on the ADMM rows and the")
    out("  MAC ARRAY on ARM-MPC's Riccati factorisation -- but neither binds at or")
    out("  above 64 B/cycle with any array size in the sweep.")
    out()
    out("  READ THE 50%-PERIOD COLUMN. It is the only column with a floor in it:")
    out("    16 B/cycle  QUAD-MPC cold = 1102 us > 1000 us period -- MISSES outright")
    out("    32 B/cycle  QUAD-MPC cold =  570 us -- blows a 50% budget (500 us)")
    out("    64 B/cycle  QUAD-MPC cold =  304 us -- fits")
    out("  -> 64 B/CYCLE IS THE LATENCY FLOOR, set by the MEASURED QUAD-MPC")
    out("     cold-solve tail at 1 kHz inside a 50%-of-period budget.")
    out("  -> **IT NOW RESTS ON ONE ROW.** Until 2026-07-30 it rested on two, the")
    out("     other being PWR-MPC at 33 us against a 25 us budget at 32 B/cycle. The")
    out("     PWR-MPC re-spec removed that row from the argument (FCS-MPC clears its")
    out("     budget at 9.3x even at 16 B/cycle). The floor does not MOVE, because")
    out("     the surviving row is the one whose tail is measured rather than")
    out("     estimated -- but a single-row justification is the exact structural")
    out("     weakness that caused the two previous sizing re-runs, and it is stated")
    out("     here rather than left to be rediscovered.")
    out("  -> MAC COUNT HAS NO LATENCY FLOOR IN THE SOLVER SET AT ALL: 64 MACs")
    out("     clears every solver row at worst-case iterations at 64 B/cycle.")
    out("     Caveat added with the re-spec: FCS-MPC is ARRAY-bound below 12x12")
    out("     (tile=8 costs it 1.14x) and intensity-bound above, so 144 MACs is the")
    out("     knee for that row too. See results_pwr_mpc_respec.txt section 1.")
    out()

    out("-" * 108)
    out("Q1c — MINIMUM SQUARE ARRAY THAT MEETS EACH DEADLINE (64 B/cycle, 16 lanes)")
    out("-" * 108)
    out(f"{'benchmark':<12} {'min MACs (p95)':>15} {'min MACs (worst)':>17} "
        f"{'binding resource at minimum':<30}")
    for b in M.BENCHMARKS:
        mins = []
        for it in (b.iters, b.iters_wc):
            found = None
            for tile in range(1, 65):
                c = cfg("x", 64, tile)
                if M.model_argmind(dataclasses.replace(b, iters=it),
                                   c)["latency_us"] <= b.deadline_us:
                    found = tile * tile
                    break
            mins.append(f"{found}" if found else "never")
        c = cfg("x", 64, 8)
        out(f"{b.name:<12} {mins[0]:>15} {mins[1]:>17} "
            f"{decompose(b, c, b.iters_wc)['_bind']:<30}")
    out()
    out("  NOTHING in the frozen set needs more than 36 MACs to meet a deadline.")
    out("  SLM-DECODE says 'never' because it is external-bandwidth-bound (needs")
    out("  ~22 GB/s, has 6.4): a provisioning problem, not an array problem.")
    out()

    # ================================================================== Q2 ===
    out("=" * 108)
    out("Q2 — WHAT SIZES THE MAC ARRAY NOW? (latency: nothing. energy: solver GEMM.)")
    out("=" * 108)
    out("The premise of this re-run was that with ARM-MPC at 24x margin the array")
    out("becomes purely an ML feature. That is TRUE FOR LATENCY and FALSE FOR ENERGY.")
    out()
    out("Tile dimension is the operand-REUSE depth: a Tm x Tn array fetches")
    out("(1/Tm + 1/Tn) operand elements per GEMM MAC. So a bigger array does not")
    out("make the GEMM-bearing SOLVER rows faster than their deadline needs, but it")
    out("does make them cheaper. Energy per call, 64 B/cycle, 128 KB, 16 lanes:")
    out()
    out(f"{'tile':>6} {'MACs':>6} {'mac mm2':>8} {'core mm2':>9} " +
        " ".join(f"{b.name[:8]:>9}" for b in SOLVERS) +
        f" {'sum uJ':>8} {'d(uJ)/d(mm2)':>13}")
    prev = None
    for tile in (4, 6, 8, 12, 16, 24, 32):
        c = cfg("x", 64, tile)
        es = [M.model_argmind(b, c)["energy_uj"] for b in SOLVERS]
        s, a = sum(es), area_breakdown(c)["total"]
        marg = "" if prev is None else f"{(prev[0] - s) / (a - prev[1]):>13.0f}"
        out(f"{tile:>6} {tile * tile:>6} {S.mac_array_mm2(tile * tile):>8.3f} "
            f"{a:>9.3f} " + " ".join(f"{e:>9.2f}" for e in es)
            + f" {s:>8.2f} {marg}")
        prev = (s, a)
    out()
    out("  Per-row sensitivity, 64 MACs vs 256 MACs:")
    for b in M.BENCHMARKS:
        e8 = M.model_argmind(b, cfg("x", 64, 8))["energy_uj"]
        e16 = M.model_argmind(b, cfg("x", 64, 16))["energy_uj"]
        tag = ("GEMM-bearing: array pays" if e8 / e16 > 1.05
               else "pure GEMV or DRAM-bound: array is dead area")
        out(f"    {b.name:<12} {e8:>10.2f} -> {e16:>10.2f} uJ  "
            f"({e8 / e16:.2f}x)  {tag}")
    out()
    out("  So the array is justified by: VIT-TINY on LATENCY (needs 36 MACs at 30")
    out("  fps), and EKF-24/EKF-96/ARM-MPC on ENERGY (1.3-1.6x for 64->256 MACs).")
    out("  EKF is a beachhead workload, not an ML one, so 'the array is an ML")
    out("  feature that solver customers subsidise' is NOT the right story. The")
    out("  right story is: the array is an ENERGY feature for matrix-MATRIX solver")
    out("  work (covariance recursion, Riccati factorisation) and a THROUGHPUT")
    out("  feature for ML. Marginal return collapses past 144 MACs: 84 uJ/mm2 for")
    out("  8->12, 27 for 12->16, 8 for 16->24.")
    out()
    out("  ADVERSARIAL — the cheapest way to buy this reuse is probably NOT MACs.")
    out("  The energy win above is entirely 1/Tm + 1/Tn, i.e. reuse depth. Growing")
    out("  the array buys reuse at O(T^2) area for O(1/T) traffic. An L0 operand")
    out("  register file / accumulator blocking tier below the array buys the same")
    out("  traffic reduction at O(T) area. The model cannot yet evaluate that (one")
    out("  pJ/byte for all reads, no L0 tier), and it is the single highest-value")
    out("  model extension outstanding. If an L0 tier lands, the honest array size")
    out("  drops back toward the ML-only 36-64 MACs.")
    out()
    out("  Square is optimal, so this is a one-dimensional choice: at fixed MAC")
    out("  count, (1/Tm + 1/Tn) is minimised at Tm == Tn (AM-HM). Checked:")
    for tm, tn in ((12, 12), (16, 9), (18, 8), (24, 6), (36, 4)):
        c = cfg("x", 64, tm, tile_n=tn)
        out(f"    {tm:>2}x{tn:<2} = {c.macs:>3} MACs  "
            f"{c.gemm_operand_bytes_per_mac('fp32'):.4f} operand B/MAC  "
            f"EKF-96 {M.model_argmind(M.BENCHMARKS[4], c)['energy_uj']:.2f} uJ")
    out()
    out("  SOLVER-ONLY SKU — is it a real product? Yes, and it is a big saving:")
    out(f"  {'SKU':<34} {'MACs':>5} {'core mm2':>9} {'solver uJ':>10} "
        f"{'perf/W':>12} {'misses (worst-case)':<22}")
    for label, bw, tile, kb in (
            ("CM-100S  solver-only, min array", 64, 4, 128),
            ("CM-100S  solver-only, 64 MAC", 64, 8, 128),
            ("CM-100   proposed", 128, 12, 128),
            ("CM-100   previous recommendation", 128, 16, 256)):
        c = cfg("x", bw, tile, kb=kb)
        e = evaluate(c)
        out(f"  {label:<34} {c.macs:>5} {e['area']['total']:>9.3f} "
            f"{e['solver_uj']:>10.2f} "
            f"{e['solver_pw_min']:>5.1f}-{e['solver_pw_max']:<6.1f} "
            f"{(', '.join(e['misses']) or 'none'):<22}")
    out()
    out("  A 16-MAC solver-only part clears every solver deadline at worst-case")
    out("  iterations and costs 0.378 mm2 -- 54% less silicon than the previous")
    out("  recommendation. It pays 2.3x the solver energy of the 256-MAC part on")
    out("  the EKF rows, so it is the right SKU for an MPC-only socket and the")
    out("  wrong one for a state-estimation socket. Both share the MACIF boundary,")
    out("  so this is a synthesis parameter, not a second design.")
    out()

    # ================================================================== Q3 ===
    out("=" * 108)
    out("Q3 — DOES THE ENERGY ARGUMENT FOR 128 B/CYCLE SURVIVE? (yes at -12%, but")
    out("     ~80% of it is a bank-geometry bet, not a bandwidth effect)")
    out("=" * 108)
    out("Previous claim: -15% solver energy for +0.042 mm2. Re-derived at 128 KB")
    out("with the corrected benchmarks, decomposed into its two mechanisms:")
    out()
    out("Reference point is the 64 B/cycle FLOOR, at the recommended 144 MACs /")
    out("128 KB. 'from pJ/B' is the part of the delta explained by the bank")
    out("geometry alone; 'from static' is the part explained by a shorter active")
    out("window. They should add to roughly the total delta.")
    out()
    out(f"{'B/cyc':>6} {'banks':>6} {'word':>5} {'pJ/B':>6} {'spad mm2':>9} "
        f"{'solver uJ':>10} {'vs 64B':>8} {'from pJ/B':>10} {'from static':>12}")
    terms = {}
    for bw in (32, 64, 128, 256):
        c = cfg("x", bw, 12)
        tot_uj, sram_bytes, stat_uj = 0.0, 0.0, 0.0
        for b in SOLVERS:
            r = M.model_argmind(b, c)
            tot_uj += r["energy_uj"]
            sram_bytes += r["e_sram_frac"] * r["energy_uj"] * 1e6 \
                / (c.overhead_mult * c.sram_pj_per_byte())
            stat_uj += c.static_mw * 1e-3 * r["latency_us"] * 1e-6 * 1e6
        terms[bw] = (tot_uj, sram_bytes, stat_uj, c)
    ref = terms[64]
    for bw in (32, 64, 128, 256):
        tot_uj, sram_bytes, stat_uj, c = terms[bw]
        # Attribute the change: bytes fetched are IDENTICAL at every width, so the
        # pJ/B term is (bytes x d(pJ/B)) x overhead, and the rest is the shorter
        # active window's static energy.
        d_pjb = (ref[3].sram_pj_per_byte() - c.sram_pj_per_byte()) \
            * ref[1] * c.overhead_mult / 1e6
        d_stat = ref[2] - stat_uj
        d = f"{100 * (tot_uj - ref[0]) / ref[0]:>7.1f}%" if bw != 64 else "  (ref)"
        out(f"{bw:>6} {banks_for(bw):>6} {bw // banks_for(bw):>4}B "
            f"{c.sram_pj_per_byte():>6.3f} "
            f"{c.spad.area_mm2()['total_mm2']:>9.3f} {tot_uj:>10.2f} {d:>8} "
            f"{d_pjb:>+9.2f}u {d_stat:>+11.2f}u")
    d_tot = ref[0] - terms[128][0]
    d_area = (terms[128][3].spad.area_mm2()["total_mm2"]
              - ref[3].spad.area_mm2()["total_mm2"])
    d_pjb128 = (ref[3].sram_pj_per_byte() - terms[128][3].sram_pj_per_byte()) \
        * ref[1] * 2.0 / 1e6
    out()
    out(f"  64 -> 128 B/cycle at 128 KB / 144 MACs: solver energy "
        f"{ref[0]:.2f} -> {terms[128][0]:.2f} uJ = -{100 * d_tot / ref[0]:.1f}%,")
    out(f"  for +{d_area:.3f} mm2 of scratchpad (+{d_area * TOP_MARGIN:.3f} mm2 "
        f"with top margin). The argument SURVIVES, at")
    out(f"  -{100 * d_tot / ref[0]:.0f}% rather than the -15% previously claimed.")
    out()
    out(f"  NOW THE ASSUMPTION IT RESTS ON. Of that {d_tot:.2f} uJ, "
        f"{d_pjb128:.2f} uJ is pJ/byte falling")
    out(f"  {ref[3].sram_pj_per_byte():.3f} -> "
        f"{terms[128][3].sram_pj_per_byte():.3f} and only "
        f"{ref[2] - terms[128][2]:.2f} uJ is the shorter active window -- so")
    out(f"  {100 * d_pjb128 / d_tot:.0f}% of the energy case for width is a "
        f"BANK-GEOMETRY effect (16 banks of")
    out("  8 KB are cheaper per access than 8 banks of 16 KB), NOT a bandwidth")
    out("  effect: the byte COUNT a workload fetches is identical at both widths.")
    out("  That is the same [A2]-interpolation mechanism flagged in ADVERSARIAL")
    out("  CHECK 3. If the real foundry macro does aggressive column gating, pJ/byte")
    out("  goes flat with width and this -12% collapses toward -2%. The Phase-1")
    out("  compiler datasheet settles it; until then 128 B/cycle is a BET and")
    out("  64 B/cycle is the floor that does not depend on winning it.")
    out()
    out("  ARM-MPC IS fp64 -- 8 B/element. What that changes:")
    b = next(x for x in M.BENCHMARKS if x.name == "ARM-MPC")
    for bw in (64, 128):
        for f in (4, 8, 16):
            gemv_ceil = bw / 8.0
            out(f"    {bw:>3} B/cyc, {f:>2} fp64 FMAs: fp64 GEMV ceiling "
                f"{min(gemv_ceil, f):>5.1f} MAC/cyc -> "
                f"{'FMA-CLUSTER' if f < gemv_ceil else 'BANDWIDTH'}-limited")
    out()
    out("  fp64 halves the MAC/cycle a given B/cycle can feed, but ARM-MPC's")
    out("  4000 us deadline is 16x QUAD-MPC's 1000 us while its op count is only")
    out("  ~2x, so external/scratchpad bandwidth does not bind it. RESOLVED")
    out("  2026-07-30: the fp64 DATAPATH is now costed explicitly -- see")
    out("  perf_model/fp64_datapath.py. The 0.125 rate factor on the bf16 array is")
    out("  gone; fp64 throughput is now an absolute FMA count that carries area.")
    out("  Summary of that analysis: 8 merged multi-format FMAs, +0.072 mm2 over an")
    out("  honest fp32-only vector unit, ARRAY-bound not bandwidth-bound on fp64.")
    out()

    # ================================================================== Q4 ===
    out("=" * 108)
    out("Q4 — CAPACITY: 128 KB does NOT suffice. OVERTURNED 2026-07-30 by MEASUREMENT")
    out("=" * 108)
    out("This section previously concluded 'yes, and it is better on every axis'.")
    out("The functional simulator's MEASURED static allocation plan for QUAD-MPC")
    out("contradicts it: 163.8 KiB at fp32 against the 128 KB that was costed. The")
    out("measurement is authoritative over the model. Two errors, both flattering:")
    out()
    out("  (1) P and L were charged n^2/2 each -- a PACKED triangular layout. Packed")
    out("      triangular addressing puts row i at element offset i(i+1)/2, which")
    out("      base+leading-dimension AGUs CANNOT EXPRESS. The capacity model was")
    out("      assuming a storage layout the descriptor format cannot address.")
    out("      Undercount: 25600 B.")
    out("  (2) Working vectors and residual temporaries were charged ZERO. They are")
    out("      14136 B at fp32 -- 10% of a 128 KiB scratchpad -- structurally")
    out("      invisible because the field is named `persistent_bytes` and a")
    out("      temporary is not persistent.")
    out()
    out("  Exact decomposition: 102400 (model) + 25600 (identity block materialised)")
    out("  + 12800 (P) + 12800 (L) + 14136 (vectors) = 167736 B measured. No")
    out("  residual term, which is what makes it a diagnosis rather than an estimate.")
    out()
    out("  WHY 128 KB WAS ARITHMETICALLY IMPOSSIBLE, not merely tight: the flagship's")
    out("  three matrices ALONE are Gamma 76800 + P 25600 + L 25600 = 125.0 KiB of a")
    out("  128.0 KiB scratchpad, leaving under 3 KiB for every vector in the program.")
    out("  Even aliasing every temporary that legally can (rp onto Ax, rd onto Px,")
    out("  zt onto Ax, t onto t_dual, xt onto rhs, sigma deleted by a zero-stride")
    out("  broadcast) lands at 134.1 KiB. The conclusion does not rest on the")
    out("  implementation's profligate vector allocation.")
    out()
    out("  fp64 IS NOT THE TARGET. 327.6 KiB at fp64 is a VALIDATION artefact: the")
    out("  implementation runs fp64 to match OSQP bit-for-bit. ADMM's measured")
    out("  control-action error at fp32 is 4e-7, and fp64 is an interior-point")
    out("  requirement (ARM-MPC), not an ADMM one. Do not size against it.")
    out()
    out("Corrected persistent footprints (P and L as dense squares because that is")
    out("what the AGU addresses, A = [I;Gamma] so the identity block stores nothing,")
    out("working vectors counted; ARM-MPC is fp64 so 8 B/element):")
    out()
    out(f"{'benchmark':<12} {'persistent':>11} {'fits 64K':>9} {'fits 128K':>10} "
        f"{'fits 192K':>10} {'overflow @192K':>15} {'stream GB/s':>12} {'DRAM uJ':>9}")
    for b in SOLVERS:
        ov = max(0.0, b.persistent_bytes - M.CM100_SCRATCHPAD_KB * 1024)
        gbs = ov * b.calls_per_sec / 1e9
        out(f"{b.name:<12} {b.persistent_bytes / 1024:>8.1f} KB "
            f"{str(b.persistent_bytes <= 64 * 1024):>9} "
            f"{str(b.persistent_bytes <= 128 * 1024):>10} "
            f"{str(b.persistent_bytes <= M.CM100_SCRATCHPAD_KB * 1024):>10} "
            f"{ov / 1024:>12.1f} KB {gbs:>12.3f} "
            f"{ov * M.E_DRAM_BYTE_PJ / 1e6:>9.2f}")
    out()
    out("  History of these two numbers, because they have moved three times and")
    out("  the determinism argument has to be re-made each time, never inherited:")
    out("  ARM-MPC 861 -> 172 -> 323 KB; QUAD-MPC 150 -> 100 -> 138.8 KB. The")
    out("  middle value in each case was the packed-layout undercount.")
    out()
    out("  The determinism claim is 'no DRAM in the control loop'. It is only")
    out("  load-bearing on the FAST loops, so test it per plausible co-resident")
    out("  workload rather than per benchmark:")
    out()
    composites = [
        ("drone node   QUAD-MPC 1kHz + EKF-24 1kHz", ["QUAD-MPC", "EKF-24"], 1000),
        ("drive node   PWR-MPC 40kHz + EKF-24 1kHz", ["PWR-MPC", "EKF-24"], 40000),
        ("manipulator  ARM-MPC 250Hz + EKF-96 200Hz", ["ARM-MPC", "EKF-96"], 250),
    ]
    out(f"  {'composite':<44} {'footprint':>10} {'fits 128K':>10} {'fits 192K':>10} "
        f"{'fits 256K':>10} {'fastest loop':>13}")
    for label, names, rate in composites:
        f = sum(next(b for b in SOLVERS if b.name == n).persistent_bytes
                for n in names)
        out(f"  {label:<44} {f / 1024:>7.1f} KB {str(f <= 128 * 1024):>10} "
            f"{str(f <= 192 * 1024):>10} "
            f"{str(f <= 256 * 1024):>10} {rate:>10} Hz")
    out()
    out("  THIS IS THE DECIDING TABLE, and it now decides differently. Capacity is")
    out("  set by the fastest determinism-critical loop, NOT by the biggest")
    out("  footprint: QUAD-MPC is 1 kHz and must be resident; ARM-MPC is the")
    out("  biggest but runs at 250 Hz with ~10x deadline margin and ~0.03 GB/s of")
    out("  streaming demand, so it overflows at 128, 192 AND 256 KB and always did.")
    out("  Growing the scratchpad does not rescue ARM-MPC and is not being asked to.")
    out()
    out(f"{'KB':>6} {'banks':>6} {'spad mm2':>9} {'pJ/B':>6} {'leak mW 85C':>12} "
        f"{'leak mW 125C':>13} " + " ".join(f"{b.name[:8]:>9}" for b in SOLVERS)
        + f" {'sum uJ':>8}")
    for kb in (64, 128, 160, 192, 256, 512):
        c = cfg("x", 128, 12, kb=kb)
        es = [M.model_argmind(b, c)["energy_uj"] for b in SOLVERS]
        out(f"{kb:>6} {banks_for(128):>6} {c.spad.area_mm2()['total_mm2']:>9.3f} "
            f"{c.sram_pj_per_byte():>6.3f} {c.spad.leakage_mw():>12.2f} "
            f"{c.spad.leakage_mw(True):>13.2f} "
            + " ".join(f"{e:>9.2f}" for e in es) + f" {sum(es):>8.2f}")
    out()
    out("  RULING: 192 KB, 16 banks x 12 KiB. Bandwidth (128 B/cycle) and bank")
    out("  count (16) DO NOT MOVE -- this is a capacity correction, not a")
    out("  rebalancing, and it does not touch the bandwidth-first finding.")
    out()
    out("  Cost, and it is a real cost: scratchpad 0.203 -> 0.279 mm2 (+0.077),")
    out("  core 0.698 -> 0.786 mm2 (+12.6%), pJ/byte 0.718 -> 0.839, leakage 0.52")
    out("  -> 0.78 mW at 85C, solver energy +5.5%, QUAD-MPC perf/W 14.13x ->")
    out("  12.55x. Latency does NOT move (36.0 us either way): capacity was never")
    out("  the binding constraint and still is not.")
    out()
    out("  WHY 192 AND NOT 160: 160 KB fits the 138.8 KiB basis at 86.8% and would")
    out("  be the cheaper answer if bank geometry were linear. It is not. 10 KiB")
    out("  per bank is a 1280x64b macro with array efficiency 0.354, against 0.394")
    out("  for 12 KiB (1536x64b), so the last 32 KiB costs only 0.020 mm2 and 160 KB")
    out("  is DOMINATED on area per bit. 192 KB also holds the NAIVE 163.8 KiB")
    out("  layout at 85.3%, so the flagship fits even from a code generator that")
    out("  has not done the A = [I;Gamma] optimisation -- that robustness is most")
    out("  of the argument.")
    out()
    out("  REJECTED, with reasons, so none of these returns as a new idea:")
    out("   * packed-triangular AGU mode: saves 12.8 KiB -> 125.7 KiB = 95.9% full.")
    out("     A 4% margin on a +/-2x modelled footprint is not a margin. Keep it as")
    out("     a later optimisation, not as a sizing mechanism.")
    out("   * overlay P and L in one square: saves 25.6 KiB but needs THREE new")
    out("     hardware behaviours (triangle-masked GEMM output, symmetric-from-")
    out("     strict-upper GEMV, transposed triangular elementwise add) plus a")
    out("     separate diagonal buffer, because P's diagonal and L's collide.")
    out("     Control logic costs more than 0.026 mm2 of SRAM.")
    out("   * eliminate P algebraically (P*xt = sigma(x-xt) - q + A'(rho*z - y -")
    out("     rho*zt)): reaches 113.8 KiB but costs an extra Gamma' GEMV per check")
    out("     (+44% on the residual block) and returns a residual that is NOT")
    out("     OSQP's. Refused for the same reason the 2-norm termination substitute")
    out("     is refused; accepting one and rejecting the other would be incoherent.")
    out("   * require fp16 operands: reaches 101.3 KiB, but fp16 is an OPT-IN")
    out("     codegen mode, so it cannot be the default sizing basis.")
    out("   * stream the overflow: only 0.026 GB/s at 1 kHz, so the BANDWIDTH is")
    out("     trivial -- but it puts DRAM in the flagship control loop and forfeits")
    out("     the determinism claim on the one row where determinism is the")
    out("     product. Retained only as an explicit small-SKU option.")
    out("   * 64 KB: rejected outright; QUAD-MPC would stream every 1 ms tick.")
    out()

    # ---- retained adversarial checks (independent of the ARM-MPC error) ----
    out("=" * 108)
    out("ADVERSARIAL CHECK 1 — segmented reduction (still required, cheaper to skip")
    out("                      than before, because the deadline pressure is gone)")
    out("=" * 108)
    out(f"{'config':<14} {'bench':<12} {'seg ON':>10} {'seg OFF':>10} "
        f"{'penalty':>9} {'deliv B/cyc':>12}  (worst-case iters)")
    for bw in (64, 128, 256):
        on = cfg("on", bw, 12, seg=True)
        off = cfg("off", bw, 12, seg=False)
        for name in ("QUAD-MPC", "ARM-MPC", "PWR-MPC"):
            b = next(x for x in M.BENCHMARKS if x.name == name)
            bw_ = dataclasses.replace(b, iters=b.iters_wc)
            r1 = M.model_argmind(bw_, on)
            r0 = M.model_argmind(bw_, off)
            out(f"{'bw=' + str(bw):<14} {name:<12} {r1['latency_us']:>10.1f} "
                f"{r0['latency_us']:>10.1f} "
                f"{r0['latency_us'] / r1['latency_us']:>8.2f}x "
                f"{r0['delivered_bw_bpc']:>12.0f}")
    out()
    out("  At 64 B/cycle the penalty is 1.00-1.05x: fp32 rows of 80 and 30 elements")
    out("  are already near-multiples of 64 B. Segmented reduction is now a")
    out("  16%-at-128B / 41%-at-256B feature, i.e. it is the price of the wide")
    out("  configuration rather than a requirement of the narrow one. If 64 B/cycle")
    out("  is chosen, this RTL feature can be DEFERRED -- a real schedule saving.")
    out()

    out("=" * 108)
    out("ADVERSARIAL CHECK 2 — perf/W still depends on POWER GATING between calls")
    out("=" * 108)
    out(f"{'bench':<12} {'duty':>7} {'gated uJ':>10} {'ungated uJ':>11} "
        f"{'perf/W gated':>13} {'perf/W ungated':>15}")
    c = cfg("x", 64, 12)
    for b in SOLVERS:
        r = M.model_argmind(b, c)
        eff = min(M.model_cpu(b, x)["energy_uj"] for x in M.ALL_CPUS)
        period_us = 1e6 / b.calls_per_sec
        duty = r["latency_us"] / period_us
        leak_extra = c.static_mw * 1e-3 * (period_us - r["latency_us"]) * 1e-6 * 1e6
        ung = r["energy_uj"] + leak_extra
        out(f"{b.name:<12} {100 * duty:>6.1f}% {r['energy_uj']:>10.2f} "
            f"{ung:>11.2f} {eff / r['energy_uj']:>12.1f}x {eff / ung:>14.1f}x")
    out()
    out("  Unchanged conclusion, and MORE load-bearing at 128 KB/64 B/cycle than")
    out("  before: the smaller core has lower dynamic energy, so the fixed 15 mW")
    out("  static estimate is a LARGER fraction of the total. EKF-24 still falls")
    out("  below the 5x kill criterion ungated. Coarse-grain power gating plus a")
    out("  retention-mode scratchpad remains a REQUIREMENT. Measure static power in")
    out("  Phase 1: it is now the most load-bearing single [EST] in the model.")
    out()

    out("=" * 108)
    out("ADVERSARIAL CHECK 3 — why pJ/byte falls with width, and where it breaks")
    out("=" * 108)
    for kb, nb, wb in ((128, 8, 8), (128, 16, 8), (128, 16, 16),
                       (256, 8, 8), (256, 16, 8), (256, 16, 16)):
        s = S.SpadCfg(kb, nb, wb)
        e = s.energy_pj_per_byte()
        out(f"  {kb:>4} KB {nb:>3} banks x {wb:>2} B ({nb * wb:>3} B/cyc): "
            f"macro {e['macro_16_pj_b']:.3f} + wire {e['wire_pj_b']:.3f} + mux "
            f"{e['mux_pj_b']:.3f} = {e['total_pj_b']:.3f} pJ/B"
            + ("   [EXTRAPOLATED]" if e["extrapolated"] else ""))
    out()
    out("  Two independent mechanisms are conflated in 'wider is cheaper': (a) a")
    out("  read activates a whole wordline, so sensing more of it is cheaper per")
    out("  byte [A2, 45nm, measured]; (b) more banks means SMALLER banks, and small")
    out("  macros are cheaper per access. Both are directionally real. Both are")
    out("  interpolations of one 45nm dataset scaled by a single 0.30 factor. The")
    out("  -12% energy case for 128 B/cycle and the -10% case for 128 KB are the")
    out("  SAME BET made twice, so they are correlated, not additive evidence. If")
    out("  the bet loses, the CM-100 should be 64 B/cycle at 128 KB and the")
    out("  128 KB decision survives anyway (it is also -0.134 mm2 and half the")
    out("  leakage, neither of which depends on the energy model).")
    out()

    # ---- recommendation ----
    rec = cfg("CM-100", 128, 12, 16, kb=M.CM100_SCRATCHPAD_KB)
    e = evaluate(rec)
    a = e["area"]
    sp = rec.spad
    floor = cfg("CM-100 floor", 64, 8, 16, kb=M.CM100_SCRATCHPAD_KB)
    ef = evaluate(floor)
    old = cfg("old", 128, 16, 16, kb=256)
    eo = evaluate(old)

    out("=" * 108)
    out("RECOMMENDATION — CM-100, REVISED DOWN")
    out("=" * 108)
    out("  The previous recommendation was OVER-PROVISIONED. Both of its stated")
    out("  grounds were benchmark artefacts. Corrected:")
    out()
    out(f"  {'':<26} {'previous':>12} {'REVISED':>12} {'min viable':>12}")
    out(f"  {'scratchpad bandwidth':<26} {'128 B/cyc':>12} {'128 B/cyc':>12} "
        f"{'64 B/cyc':>12}")
    out(f"  {'scratchpad capacity':<26} {'256 KB':>12} {'128 KB':>12} {'128 KB':>12}")
    out(f"  {'MAC array':<26} {'256':>12} {'144':>12} {'64':>12}")
    out(f"  {'core area mm2':<26} {eo['area']['total']:>12.3f} "
        f"{a['total']:>12.3f} {ef['area']['total']:>12.3f}")
    out(f"  {'solver energy sum uJ':<26} {eo['solver_uj']:>12.2f} "
        f"{e['solver_uj']:>12.2f} {ef['solver_uj']:>12.2f}")
    out(f"  {'solver perf/W':<26} "
        f"{str(round(eo['solver_pw_min'], 1)) + '-' + str(round(eo['solver_pw_max'], 1)):>12} "
        f"{str(round(e['solver_pw_min'], 1)) + '-' + str(round(e['solver_pw_max'], 1)):>12} "
        f"{str(round(ef['solver_pw_min'], 1)) + '-' + str(round(ef['solver_pw_max'], 1)):>12}")
    out()
    out(f"  The revised part is {100 * (1 - a['total'] / eo['area']['total']):.0f}% "
        f"smaller than the previous recommendation AND has slightly")
    out("  lower solver energy. It dominates it on both axes -- there is no trade")
    out("  being made here, the old point was simply off the Pareto front. The")
    out("  'min viable' column is a further -24% area for +35% solver energy, which")
    out("  IS a trade, and the right one only for an MPC-only (no-EKF) socket.")
    out()
    out(f"  scratchpad     128 KB, {banks_for(128)} single-port banks x "
        f"{128 // banks_for(128)} B/cycle = 128 B/cycle")
    out(f"                 bank geometry: {sp.geom.rows} rows x {sp.geom.cols} cols, "
        f"{sp.geom.mux}:1 col mux, {sp.geom.folds} fold(s)")
    out(f"  MAC array      {rec.macs} MACs ({rec.tile}x{rec.tile_cols}), bf16 mult / "
        f"fp32 accumulate, int8 double-pumped, fp32 2-pass")
    out(f"  FMA cluster    {max(rec.fp64_fmas, rec.vector_lanes // 2)} merged "
        f"multi-format units = {rec.vector_lanes} fp32 lanes OR "
        f"{rec.fp64_fmas} fp64 FMAs (fp64 path now COSTED)")
    out(f"  frequency      800 MHz")
    out()
    out(f"  AREA  scratchpad {a['spad']:.3f} | MAC {a['mac']:.3f} | vector "
        f"{a['vec']:.3f} | fp64 increment {a['fp64']:.3f} | seq+DMA+misc "
        f"{a['rest']:.3f}")
    out(f"        subtotal {a['sub']:.3f} mm2, +15% margin = {a['total']:.3f} mm2")
    out(f"        pessimistic fp64 edge (56.2 kgate anchor): "
        f"{area_breakdown(rec, pessimistic=True)['total']:.3f} mm2")
    out(f"  ENERGY scratchpad read {e['pj_b']:.3f} pJ/byte")
    out(f"  LEAK  scratchpad {sp.leakage_mw():.2f} mW @85C / "
        f"{sp.leakage_mw(True):.2f} mW @125C")
    out()
    out(f"  {'benchmark':<12} {'deadline':>9} {'lat p95':>9} {'lat worst':>10} "
        f"{'margin wc':>10} {'uJ/call':>9} {'perf/W':>7} {'binds on':<10} verdict")
    for b in M.BENCHMARKS:
        r = e["rows"][b.name]
        v = "MEETS" if r["meets_wc"] else "MISSES"
        if r["meets_wc"] and not r["meets_wc_budget"]:
            v += " (>50% period)"
        out(f"  {b.name:<12} {r['deadline']:>9.0f} {r['lat']:>9.1f} "
            f"{r['lat_wc']:>10.1f} {r['deadline'] / r['lat_wc']:>9.2f}x "
            f"{r['uj']:>9.2f} {r['pw']:>7.1f} {r['bind']:<10} {v}")
    out()
    out(f"  solver perf/W band: {e['solver_pw_min']:.1f}x - "
        f"{e['solver_pw_max']:.1f}x  (kill criterion 5x)")
    out()
    out("  DECISIONS AND WHAT THEY REST ON")
    out("   * 128 KB, 128 B/cycle, 144 MACs is the ENERGY-KNEE point. Every part of")
    out("     it is a synthesis parameter, so none of it is irreversible.")
    out(f"   * 64 B/cycle / 64 MACs / 128 KB ({ef['area']['total']:.3f} mm2) is the "
        f"FEASIBILITY FLOOR:")
    out("     it meets every deadline in the set at worst-case iteration counts")
    out("     inside a 50%-of-period budget. If the Phase-1 macro datasheet shows")
    out("     flat pJ/byte vs access width, DROP TO THE FLOOR -- the entire case")
    out("     for the wider, larger part evaporates with it.")
    out("   * Sensitive to, in order: (1) foundry macro pJ/byte vs access width and")
    out("     vs macro size -- one 45nm dataset, one scale factor, two correlated")
    out("     -10% claims; (2) static power (15 mW [EST]) which sets ungated")
    out("     perf/W; (3) the fp64 FMA gate count (26.6-56.2 kgates, a 2.1x band")
    out(f"     that moves core area {a['total']:.2f} -> "
        f"{area_breakdown(rec, pessimistic=True)['total']:.2f} mm2); "
        f"(4) PERIOD_BUDGET=0.50,")
    out("     which is what makes 64 B/cycle rather than 32 B/cycle the floor.")
    out("     At 100% of period, 32 B/cycle suffices.")
    out("   * MEASUREMENTS THAT WOULD FIRM THIS UP: foundry SRAM compiler energy")
    out("     tables at 8/16/32 B access width and 8/16/32 KB macro size; measured")
    out("     cold-solve iteration tails for PWR-MPC and ARM-MPC (both [EST]);")
    out("     synthesis area + energy for one merged fp64/2xfp32 FMA at 800 MHz")
    out("     (CVFPU/FPnew is open source -- this is a one-day Yosys/OpenROAD run")
    out("     and it closes the largest remaining area uncertainty).")
    out()
    out("  WHAT DID NOT CHANGE: bandwidth-first over MAC-first (the ADMM rows are")
    out("  still GEMV-bound and still bind on scratchpad bandwidth, just at a lower")
    out(f"  absolute width); the >=5x kill criterion (passes "
        f"{e['solver_pw_min']:.0f}-{e['solver_pw_max']:.0f}x); zero-copy and")
    out("  residency; power gating as a requirement. SLM-DECODE still misses by")
    out("  3.4x and is still an external-bandwidth provisioning problem.")
    out()
    out("  ADDED 2026-07-30: the fp64 datapath, previously ZERO in this budget, is")
    out(f"  now costed at +{a['fp64']:.3f} mm2 (8 merged multi-format FMAs). The")
    out("  vector-lane gate estimate was corrected at the same time (2200 gates/lane")
    out("  could not hold an fp32 multiplier). Core area 0.565 -> "
        f"{a['total']:.3f} mm2;")
    out("  see perf_model/fp64_datapath.py and PHASE0_FINDINGS.md.")
    out()

    import os
    dest = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "results_cm100_sizing.txt")
    with open(dest, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
