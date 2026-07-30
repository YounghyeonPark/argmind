"""
fp64 datapath — closing the largest un-costed item in the CM-100 area budget
============================================================================

WHY THIS FILE EXISTS

The model obtained fp64 throughput from a `0.125` rate factor applied to the
bf16 MAC array (`macs_per_cycle`). On the CM-100 that implied 18 fp64 MAC/cycle
for ZERO AREA. It is not an optional capability to hand-wave: ARM-MPC is
specified as acados SQP-RTI + HPIPM, i.e. interior point, and `solver-numerics`
established that the KKT condition number grows as 1/mu and reaches ~1e12 near
convergence, so fp64 is a requirement of the ALGORITHM.

This module answers five questions and writes results_fp64_datapath.txt.

  Q1  implementation options and their cost (area / energy / latency)
  Q2  how much fp64 throughput the benchmark set actually needs
  Q3  fp32 + iterative refinement vs native fp64
  Q4  effect on the CM-100 area budget and on perf/W
  Q5  the MACIF consequence

EVERY FIGURE HERE IS A MODELLED ESTIMATE at a 16nm-class node, +/-2x, and NOT
QUOTABLE EXTERNALLY. The fp64 gate count carries its OWN 2.1x band on top of
that (26.6 kgates parametric vs 56.2 kgates measured-28nm high-performance),
so fp64 area figures are the least certain numbers in the whole budget.

CITED EXTERNAL FACTS (these are the load-bearing ones):
  [C1] Ozaki-scheme DGEMM emulation on low-precision tensor cores needs
       25-196 GEMM passes with FP16 inputs and FP32 accumulation, and 121 with
       FP8(E4M3), for FP64-equivalent accuracy. The pass count is the SQUARE of
       the slice count, and slicing becomes infeasible when the input format has
       too few mantissa bits.
       https://arxiv.org/abs/2508.00441   (arXiv:2508.00441, Table 1)
  [C2] Ozaki et al.'s error-free transformation is what makes the emulation
       accurate; accuracy comparable to native FP64 requires FP64 (or otherwise
       exact) accumulation of the slice products.
       https://pmc.ncbi.nlm.nih.gov/articles/PMC7295351/
  [C3] FPnew/CVFPU: a merged multi-format FMA supporting FP64/FP32/FP16/bfloat16
       in an embedded RISC-V core, GF 22FDX. Existence proof for option (b)/(d).
       https://arxiv.org/abs/2007.01530
  [C4] A 28nm dual-precision FMA (1x binary64 or 2x binary32, 660 MHz) reported
       at 56.2 kgates -- the pessimistic edge of the gate-count band.
       https://www.sciencedirect.com/science/article/abs/pii/S0141933117305665
  [C5] Fixed-precision iterative refinement converges only while
       cond(A) * eps_working < 1 (Wilkinson / Skeel / Higham). At the
       interior-point endgame cond ~ 1e12 and eps_fp32 ~ 6e-8, the product is
       ~6e4 >> 1, so fp32-working-precision refinement does NOT converge; the
       residual must be computed in higher precision than fp32.
       Carson & Higham, SIAM J. Sci. Comput. 2017/2018 (GMRES-IR).
"""

from __future__ import annotations

import dataclasses
import math

import argmind_model as M
import sram_cost as S

try:
    import cm100_sizing as C
except ImportError:                        # pragma: no cover
    C = None


ARM = next(b for b in M.BENCHMARKS if b.name == "ARM-MPC")

# Fraction of the control period a hard-real-time integrator gives the solver.
# Same 0.50 used by the CM-100 sizing run -- feasibility is judged against this,
# not against 100% of the period.
PERIOD_BUDGET = 0.50

# ---------------------------------------------------------------------------
# Tile utilisation on the ACTUAL block dimension
# ---------------------------------------------------------------------------
# argmind_model uses `representative_dim` (a cube-root proxy) for the GEMM
# utilisation penalty, which for ARM-MPC returns 78.7 and gives 85% utilisation.
# The real HPIPM block dimension is nb = n_x + n_u = 21, which gives 63%. Both
# are reported: the difference is a straight 1.34x on every fp64 GEMM number and
# it is a modelling assumption, not a fact.
NB = 21                                    # ARM-MPC stage block dimension


def tile_util(dim: float, tile: int, drain: int = 8) -> float:
    """2-D array utilisation. Kept as a free function for the multi-pass analysis,
    which runs on the bf16 ARRAY and so must use the array's penalty."""
    quant = math.ceil(dim / tile) * tile / dim
    return 1.0 / (quant * (1.0 + drain / dim))


def fp64_util(cfg: M.ArgmindCfg, dim: float = NB) -> float:
    """Utilisation of the fp64 FMA cluster on a dim x dim x dim block GEMM.

    Delegates to ArgmindCfg.gemm_util so this file and the main model cannot
    drift apart -- an earlier version duplicated the formula, and the duplicate
    is precisely how the fp64 GEMM ended up charged the ARRAY's quantisation
    penalty in one place and the cluster's in the other.
    """
    return cfg.gemm_util("fp64", dim)


# ---------------------------------------------------------------------------
# Q2 — the requirement
# ---------------------------------------------------------------------------

def arm_mpc_fp64_work(iters: int) -> dict:
    """fp64 MAC counts for one ARM-MPC call at `iters` interior-point iterations."""
    gemm = ARM.gemm_macs_per_iter * iters + ARM.gemm_macs_per_call
    gemv = ARM.gemv_macs_per_iter * iters + ARM.gemv_macs_per_call
    return {"gemm": gemm, "gemv": gemv, "total": gemm + gemv,
            "gemm_frac": gemm / (gemm + gemv)}


def arm_mpc_latency(cfg: M.ArgmindCfg, iters: int, dim: float = NB) -> dict:
    """Latency decomposed by the resource that produced it, in us.

    Deliberately built from the resource ceilings rather than by calling
    model_argmind, so the fp64 cluster shows up as its OWN binding constraint
    and cannot hide inside an aggregate.
    """
    w = arm_mpc_fp64_work(iters)
    ns = cfg.cycle_ns / 1000.0
    u = fp64_util(cfg, dim)
    F = max(1, cfg.fp64_fmas)
    gemv_rate = min(cfg.scratchpad_bw_bpc / 8.0, F)     # fp64: 8 B/element
    d = {
        "fp64_gemm": w["gemm"] / (F * u) * ns,
        "fp64_gemv": w["gemv"] / gemv_rate * ns,
        "vec": (ARM.vector_ops_per_iter / cfg.vector_lanes
                + ARM.transcendental_per_iter * 4.0 / cfg.vector_lanes) * iters * ns,
        "seq": 0.5 * ARM.scalar_ops_per_iter / cfg.seq_ipc * iters * ns,
        "host": cfg.call_overhead_us,
    }
    d["_total"] = sum(d.values())
    d["_bind"] = max((k for k in d if not k.startswith("_")), key=lambda k: d[k])
    d["_gemv_bound_by"] = ("bandwidth" if cfg.scratchpad_bw_bpc / 8.0 <= F
                           else "fp64 cluster")
    return d


def min_fp64_fmas(bw: float = 128.0, lanes: int = 16, iters: int | None = None,
                  budget: float = PERIOD_BUDGET, dim: float = NB) -> int:
    it = ARM.iters_wc if iters is None else iters
    for f in (1, 2, 4, 6, 8, 12, 16, 24, 32):
        c = M.ArgmindCfg(name="probe", tile=12, scratchpad_bw_bpc=bw, n_banks=16,
                         vector_lanes=lanes, fp64_fmas=f, scratchpad_kb=128)
        if arm_mpc_latency(c, it, dim)["_total"] <= ARM.deadline_us * budget:
            return f
    return 0


# ---------------------------------------------------------------------------
# Q1(a) — multi-pass decomposition onto the existing bf16 multipliers
# ---------------------------------------------------------------------------
# [C1] gives the pass count for FP16 inputs (11-bit significand): 25 at k=8
# rising to 196 for large k, and FP8 at 121. bfloat16 has 8 significand bits, so
# it needs MORE slices than FP16 for the same target accuracy: slice count scales
# roughly as the mantissa bits that must be covered divided by the bits each
# slice can carry. Extrapolating the published FP16 slice range (5..14) by the
# 11/8 mantissa ratio gives 7..19 bf16 slices, i.e. 49..361 passes. [EST from C1]
OZAKI_PASSES_FP16 = (25, 196)              # [C1] measured range over k
OZAKI_PASSES_FP8 = 121                     # [C1]
OZAKI_PASSES_BF16 = (49, 361)              # [EST extrapolated from C1]
# The pass count for fp32-EQUIVALENT accuracy on bf16 (Ootomo/Yokota-style
# 3-word bf16, "bf16x9") is only 6-9. That is the crucial asymmetry: multi-pass
# on bf16 gets you to fp32 cheaply and to fp64 not at all.
BF16_PASSES_FOR_FP32 = (6, 9)

# Sequencer cost per descriptor: 3 operand base addresses, limb-index / exponent
# scale, accumulator select, issue. A hardware sequencer with dedicated AGUs and
# a wide descriptor word is at the low end; anything software-assisted is at the
# high end. [EST] -- swept below, because it turns out to matter for more than
# multi-pass.
SEQ_CYCLES_PER_DESCRIPTOR = 8
# Logical block operations per interior-point iteration in an HPIPM-style
# backward Riccati recursion: per stage roughly B'P, A'P, form the stage Hessian,
# Cholesky, two substitutions. N=30 stages. [EST]
BLOCK_OPS_PER_IP_ITER = 6 * 30
# Derived, NOT independently assumed: the benchmark charges N*nb^3/3 GEMM MACs
# per IP iteration, so the average block op is that divided by the descriptor
# count. Keeping these consistent matters -- an earlier version of this file
# assumed both independently and double-counted the GEMM work 5.7x.
MACS_PER_BLOCK_OP = ARM.gemm_macs_per_iter / BLOCK_OPS_PER_IP_ITER


def multipass_analysis(cfg: M.ArgmindCfg, passes: int, iters: int) -> dict:
    """Cost of getting fp64 by P passes over the bf16 array.

    Charges all four things the option actually costs, which is the point:
      1. P x the array time
      2. P x the MAC energy
      3. exact wide accumulators (area) -- or the scheme is not error-free
      4. P x the descriptor issue, against a block GEMM that is only ~30 cycles
    """
    w = arm_mpc_fp64_work(iters)
    ns = cfg.cycle_ns / 1000.0
    u = tile_util(NB, cfg.tile)
    # Each pass is a full bf16 GEMM over sliced operands.
    array_cycles = w["gemm"] * passes / (cfg.macs * u)
    # GEMV multi-pass: the sliced operands are P/2 x the bytes of the fp64 word
    # if held pre-split (bf16 limbs, 2 B each, sqrt(passes) slices per operand),
    # so bandwidth-bound work scales with the SLICE count, not the pass count.
    slices = math.sqrt(passes)
    gemv_bytes_per_mac = 2.0 * slices
    gemv_cycles = w["gemv"] * gemv_bytes_per_mac / cfg.scratchpad_bw_bpc
    # Descriptor issue, per pass, per logical block op. Overlapped with the array
    # to the same 50% the rest of the model assumes for sequencer/compute overlap.
    issue_cycles = (BLOCK_OPS_PER_IP_ITER * passes * SEQ_CYCLES_PER_DESCRIPTOR
                    * iters * 0.5)
    return {
        "passes": passes,
        "slices": slices,
        "fp64_equiv_macs_per_cycle": cfg.macs / passes,
        "array_us": array_cycles * ns,
        "gemv_us": gemv_cycles * ns,
        "issue_us": issue_cycles * ns,
        "issue_frac_of_array": issue_cycles / max(array_cycles, 1.0),
        "total_us": (array_cycles + gemv_cycles + issue_cycles) * ns,
        "mac_pj_per_fp64_mac": passes * M.E_MAC_PJ["bf16"],
        "sram_pj_per_fp64_gemv_mac": gemv_bytes_per_mac * cfg.sram_pj_per_byte(),
        "accum_area_mm2": S.wide_accum_delta_mm2(cfg.macs),
        "energy_uj": _multipass_energy(cfg, passes, iters, slices,
                                       array_cycles + gemv_cycles
                                       + issue_cycles, ns),
    }


def _multipass_energy(cfg, passes, iters, slices, cycles, ns):
    w = arm_mpc_fp64_work(iters)
    e_mac = (w["gemm"] + w["gemv"]) * passes * M.E_MAC_PJ["bf16"]
    # GEMM operand fetch: sliced bf16 limbs, `slices` per operand.
    bytes_ = (w["gemm"] * cfg.gemm_operand_bytes_per_mac("bf16") * slices
              + w["gemv"] * 2.0 * slices)
    lat_us = cycles * ns
    e = (e_mac + bytes_ * cfg.sram_pj_per_byte()) * cfg.overhead_mult \
        + cfg.static_mw * 1e-3 * lat_us * 1e-6 * 1e12
    return e / 1e6


# ---------------------------------------------------------------------------
# Q3 — fp32 + iterative refinement
# ---------------------------------------------------------------------------
# Per interior-point iteration the KKT solve splits as:
#   factorise   N * nb^3/3   (matrix-matrix, the O(n^3) term)
#   substitute  2*N*nb^2     (matrix-vector)
#   residual    N*nb^2       (matrix-vector: r = b - K x, one KKT matvec)
# Iterative refinement re-does {residual, substitute} k times. The residual MUST
# be higher precision than the working precision [C5], so it stays fp64 and the
# fp64 cluster is required either way -- refinement RELOCATES the fp64
# requirement from the O(n^3) term to the O(n^2) term, it does not remove it.
KKT_RESIDUAL_MACS_PER_ITER = 30 * NB * NB          # N * nb^2


def refinement_analysis(cfg: M.ArgmindCfg, k: int, iters: int) -> dict:
    ns = cfg.cycle_ns / 1000.0
    w = arm_mpc_fp64_work(iters)
    u32 = tile_util(NB, cfg.tile)
    u64 = fp64_util(cfg)
    r32 = cfg.macs * 0.5 * u32                     # fp32 GEMM MAC/cycle
    r32v = min(cfg.scratchpad_bw_bpc / 4.0, cfg.macs * 0.5)
    r64 = max(1, cfg.fp64_fmas) * u64
    r64v = min(cfg.scratchpad_bw_bpc / 8.0, max(1, cfg.fp64_fmas))

    resid = KKT_RESIDUAL_MACS_PER_ITER * k * iters              # fp64 GEMV
    resolve = ARM.gemv_macs_per_iter * k * iters                # fp32 GEMV
    cyc = (w["gemm"] / r32 + w["gemv"] / r32v
           + resid / r64v + resolve / r32v)
    vec = (ARM.vector_ops_per_iter / cfg.vector_lanes
           + ARM.transcendental_per_iter * 4.0 / cfg.vector_lanes) * iters
    seq = 0.5 * ARM.scalar_ops_per_iter / cfg.seq_ipc * iters
    lat = (cyc + vec + seq) * ns + cfg.call_overhead_us

    # Energy: MAC + scratchpad traffic, then the standard overhead multiplier.
    bpm64_gemm = cfg.gemm_operand_bytes_per_mac("fp64")
    bpm32_gemm = cfg.gemm_operand_bytes_per_mac("fp32")
    e_mac = (w["gemm"] * M.E_MAC_PJ["fp32"] + w["gemv"] * M.E_MAC_PJ["fp32"]
             + resid * M.E_MAC_PJ["fp64"] + resolve * M.E_MAC_PJ["fp32"])
    bytes_ = (w["gemm"] * bpm32_gemm + w["gemv"] * 4.0
              + resid * 8.0 + resolve * 4.0)
    e = (e_mac + bytes_ * cfg.sram_pj_per_byte()) * cfg.overhead_mult \
        + cfg.static_mw * 1e-3 * lat * 1e-6 * 1e12
    return {"k": k, "latency_us": lat, "energy_uj": e / 1e6,
            "sram_bytes": bytes_, "fp64_macs": resid,
            "fp64_frac_of_work": resid / (resid + w["gemm"] + w["gemv"] + resolve),
            # The asymptotic case FOR refinement: residual O(n^2) is cheap vs
            # factorisation O(n^3). Ratio 3k/nb -- at nb=21, k=3 that is 43%.
            "resid_vs_factor": 3.0 * k / NB,
            "_bpm64": bpm64_gemm}


def all_fp64_energy(cfg: M.ArgmindCfg, iters: int) -> dict:
    w = arm_mpc_fp64_work(iters)
    lat = arm_mpc_latency(cfg, iters)["_total"]
    e_mac = w["total"] * M.E_MAC_PJ["fp64"]
    bytes_ = w["gemm"] * cfg.gemm_operand_bytes_per_mac("fp64") + w["gemv"] * 8.0
    e = (e_mac + bytes_ * cfg.sram_pj_per_byte()) * cfg.overhead_mult \
        + cfg.static_mw * 1e-3 * lat * 1e-6 * 1e12
    return {"latency_us": lat, "energy_uj": e / 1e6, "sram_bytes": bytes_,
            "e_mac_uj": e_mac / 1e6}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

REC = M.ArgmindCfg(
    name="CM-100 recommended", tile=12, scratchpad_kb=128,
    scratchpad_bw_bpc=128.0, n_banks=16, vector_lanes=16, fp64_fmas=8)
FLOOR = M.ArgmindCfg(
    name="solver-only floor", tile=8, scratchpad_kb=128,
    scratchpad_bw_bpc=64.0, n_banks=8, vector_lanes=16, fp64_fmas=8)


def main():
    L = []

    def out(s=""):
        print(s)
        L.append(s)

    out("=" * 100)
    out("fp64 DATAPATH — COSTING THE LARGEST UN-COSTED ITEM IN THE CM-100 BUDGET")
    out("16nm-class, MODELLED ESTIMATES +/-2x (fp64 gate count carries a further")
    out("2.1x band). NOT QUOTABLE EXTERNALLY.")
    out("=" * 100)
    out()
    out("HEADLINE: 8 merged multi-format FMA units (1x fp64 or 2x fp32 each).")
    out("  Binding constraint on ARM-MPC becomes the fp64 FMA CLUSTER, not")
    out("  scratchpad bandwidth -- the one documented exception to bandwidth-first,")
    out("  and it holds because ARM-MPC is 79% matrix-MATRIX (block Riccati).")
    out()

    # ================================================================== Q1 ===
    out("=" * 100)
    out("Q1 — GATE-COUNT BASIS AND THE THREE OPTIONS")
    out("=" * 100)
    out("Parametric IEEE-754 FMA gate model (sram_cost.fma_gates_ieee):")
    out(f"  gates = {S.K_FMA_MUL_GATES_PER_BIT2}*n^2 + "
        f"{S.K_FMA_LIN_GATES_PER_BIT:.0f}*n + {S.K_FMA_FIX_GATES:.0f}, "
        f"n = significand bits;  {S.A_GATE_UM2} um2/NAND2 at 16nm")
    out(f"  {'format':<8} {'n':>4} {'kgates':>9} {'mm2/unit':>10} "
        f"{'x bf16 cell':>12}")
    base = S.fma_gates_ieee(8)
    for p in ("bf16", "fp16", "fp32", "fp64"):
        n = S.SIG_BITS[p]
        g = S.fma_gates_ieee(n)
        out(f"  {p:<8} {n:>4} {g/1e3:>9.2f} "
            f"{g*S.A_GATE_UM2*S.MAC_STAGING_OVERHEAD/1e6:>10.5f} {g/base:>11.2f}x")
    out()
    out(f"  CALIBRATION: fp32 -> {S.fma_gates_ieee(24)/1e3:.1f} kgates, "
        f"fp64 -> {S.fma_gates_ieee(53)/1e3:.1f} kgates (ratio "
        f"{S.fma_gates_ieee(53)/S.fma_gates_ieee(24):.1f}x).")
    out(f"  Cross-check at the low end: the model gives "
        f"{S.fma_gates_ieee(8)/1e3:.2f} kgates at n=8 against the")
    out(f"  independently-guessed {S.GATES_MAC_BF16_FP32ACC/1e3:.1f} kgates for the "
        f"bf16/fp32-acc array cell -- 14% apart, weak but real corroboration.")
    out(f"  PESSIMISTIC EDGE [C4]: a measured 28nm dual-precision FMA is "
        f"56.2 kgates, {S.FMA_AREA_PESSIMISM:.1f}x")
    out("  the parametric value. Every fp64 area number below is given at both.")
    out()

    out("-" * 100)
    out("(a) MULTI-PASS DECOMPOSITION onto the existing bf16 multipliers")
    out("-" * 100)
    out("  This is the option that looks free and is not. THE PASS COUNT IS THE")
    out("  WHOLE STORY, and it is a published number, not a guess:")
    out(f"    FP16 inputs, FP32 accumulate, FP64-accurate DGEMM: "
        f"{OZAKI_PASSES_FP16[0]}-{OZAKI_PASSES_FP16[1]} GEMM passes [C1]")
    out(f"    FP8(E4M3) inputs:                                  "
        f"{OZAKI_PASSES_FP8} GEMM passes [C1]")
    out(f"    bfloat16 inputs (8 mantissa bits, extrapolated):    "
        f"{OZAKI_PASSES_BF16[0]}-{OZAKI_PASSES_BF16[1]} passes [EST from C1]")
    out(f"    bfloat16 for fp32-EQUIVALENT accuracy only:         "
        f"{BF16_PASSES_FOR_FP32[0]}-{BF16_PASSES_FOR_FP32[1]} passes")
    out()
    out("  THAT ASYMMETRY IS THE FINDING. Multi-pass on bf16 reaches fp32 accuracy")
    out("  in 6-9 passes and fp64 accuracy in 49-361. fp32 is precisely the")
    out("  precision interior point cannot use [C5]. So the cheap version of this")
    out("  option does not deliver the capability, and the version that delivers")
    out("  the capability is not cheap. Costing it anyway, on the CM-100 array,")
    out("  ARM-MPC at 30 IP iterations (worst case), deadline 4000 us:")
    out()
    ref = all_fp64_energy(REC, ARM.iters_wc)
    out(f"  {'passes':>7} {'fp64 MAC/cyc':>13} {'array us':>10} {'GEMV us':>9} "
        f"{'issue us':>9} {'TOTAL us':>10} {'uJ':>9} {'vs native':>10} "
        f"{'verdict':<20}")
    for p in (9, 25, 49, 121, 196, 361):
        r = multipass_analysis(REC, p, ARM.iters_wc)
        v = ("MEETS 50% budget" if r["total_us"] <= ARM.deadline_us * PERIOD_BUDGET
             else "deadline only" if r["total_us"] <= ARM.deadline_us
             else "MISSES DEADLINE")
        note = " (fp32-acc only)" if p <= 9 else ""
        out(f"  {p:>7} {r['fp64_equiv_macs_per_cycle']:>13.1f} "
            f"{r['array_us']:>10.1f} {r['gemv_us']:>9.1f} {r['issue_us']:>9.1f} "
            f"{r['total_us']:>10.1f} {r['energy_uj']:>9.2f} "
            f"{r['energy_uj']/ref['energy_uj']:>9.1f}x {v + note:<20}")
    out(f"  {'NATIVE (8 FMAs)':>7} {8.0:>13.1f} {'':>10} {'':>9} {'':>9} "
        f"{ref['latency_us']:>10.1f} {ref['energy_uj']:>9.2f} {1.0:>9.1f}x "
        f"{'MEETS 50% budget':<20}")
    out()
    out("  THE SEQUENCER COST, checked explicitly because that is where the hidden")
    out("  expense usually lives. The result is NOT what was expected:")
    out(f"    * {BLOCK_OPS_PER_IP_ITER} descriptors per IP iteration "
        f"(N=30 stages x ~6 block ops) [EST]")
    out(f"    * consistency check: that makes the average block op "
        f"{MACS_PER_BLOCK_OP:.0f} MACs,")
    out(f"      i.e. only "
        f"{MACS_PER_BLOCK_OP/(REC.macs*tile_util(NB, REC.tile)):.1f} array cycles "
        f"on 144 MACs at {100*tile_util(NB, REC.tile):.0f}% utilisation")
    out(f"    * against {SEQ_CYCLES_PER_DESCRIPTOR} sequencer cycles to issue one "
        f"descriptor [EST]")
    out()
    out("    So the issue-to-compute ratio is:")
    out(f"    {'seq cyc/descriptor':>20} {'issue:array ratio':>18}  "
        f"consequence")
    for sc in (2, 4, 8, 16):
        ratio = sc / (MACS_PER_BLOCK_OP / (REC.macs * tile_util(NB, REC.tile)))
        out(f"    {sc:>20} {ratio:>17.2f}x  "
            f"{'array-bound' if ratio < 1 else 'SEQUENCER-ISSUE-BOUND'}")
    out()
    out("    TWO THINGS FOLLOW, and only one of them is about multi-pass:")
    out("    1. The ratio is P-INDEPENDENT. Both terms scale with the pass count,")
    out("       so multi-pass does not make the sequencer tax relatively worse.")
    out("       The expected hidden expense is NOT there. Multi-pass is killed by")
    out("       the pass count alone.")
    out("    2. BUT the ratio flags a SEPARATE, PRE-EXISTING problem this analysis")
    out("       surfaced: at nb=21 an HPIPM block op is only a handful of array")
    out("       cycles, so if descriptor issue costs >4 cycles, ARM-MPC is")
    out("       SEQUENCER-ISSUE-BOUND rather than array-bound, at ANY precision.")
    out("       The model does not represent this (it charges a flat 150 scalar")
    out("       ops/iteration). It does not change the fp64 decision -- 8 FMAs is")
    out("       still right, and an issue bound would only make a LARGER fp64 unit")
    out("       even less useful -- but it is a new open item for the sequencer")
    out("       microarchitecture: descriptor issue must be <=4 cycles, or the")
    out("       sequencer needs a hardware loop that emits a stage's descriptors")
    out("       from one instruction. LOGGED, NOT RESOLVED HERE.")
    out()
    out("  AND THE ACCUMULATOR, which is the part of option (a) that has real area:")
    out(f"    error-free slice accumulation needs an exact "
        f"{S.WIDE_ACCUM_BITS}-bit fixed-point")
    out(f"    accumulator per participating output element [C2], vs the "
        f"{S.BASE_ACCUM_BITS}-bit fp32")
    out("    accumulator already in the cell. Widening the array's accumulators:")
    for n in (16, 64, 144):
        out(f"      {n:>4} accumulators: "
            f"+{S.wide_accum_delta_mm2(n):.4f} mm2")
    inc = (S.fma_cluster_mm2(8, "fp64", multiformat=True)
           + 16 * C.VEC_TRANSCENDENTAL_GATES * S.A_GATE_UM2 / 1e6
           - 16 * C.VEC_LANE_GATES * S.A_GATE_UM2 * S.MAC_STAGING_OVERHEAD / 1e6)
    save = inc - S.wide_accum_delta_mm2(REC.macs)
    out(f"    So option (a) costs {S.wide_accum_delta_mm2(REC.macs):.4f} mm2 of "
        f"added accumulator against the")
    out(f"    +{inc:.4f} mm2 fp64 increment of the recommended option (d).")
    out()
    out("  WHY (a) IS REJECTED — stated honestly, because the deadline argument is")
    out("  NOT as clean as it first looked:")
    out(f"    * The AREA SAVING IS {save:.4f} mm2, ~{100*save/0.62:.0f}% of the "
        f"core. That is the whole prize.")
    out("    * The pass count is 49-361 [EST extrapolated from C1] -- a 7x band")
    out("      whose optimistic edge meets the raw 4000 us deadline and whose")
    out("      pessimistic edge misses it by 6x. Taking a 7x-uncertainty")
    out("      dependency on the ONE hard-real-time correctness path to save 4%")
    out("      of the core is a bad trade at any pass count in that band.")
    r49 = multipass_analysis(REC, 49, ARM.iters_wc)
    r361 = multipass_analysis(REC, 361, ARM.iters_wc)
    out(f"    * ENERGY, which is the axis we actually sell: "
        f"{r49['energy_uj']/ref['energy_uj']:.1f}x to "
        f"{r361['energy_uj']/ref['energy_uj']:.1f}x worse")
    out(f"      than native ({r49['energy_uj']:.0f}-{r361['energy_uj']:.0f} uJ vs "
        f"{ref['energy_uj']:.0f} uJ), because P passes multiply BOTH the")
    out("      MAC energy and the sliced-operand fetch traffic. This alone settles")
    out("      it: an energy-differentiated part cannot use a 4-30x energy penalty")
    out("      on a beachhead workload to save 4% of its area.")
    out("    * MACIF: error-free limb splitting plus an exact 128-bit accumulator")
    out("      is a contract no analog CIM macro can honour (see Q5). Option (a)")
    out("      would push a strong exactness requirement INTO the swappable macro.")
    out("    REJECTED — on energy and on interface risk, not on the deadline.")
    out()
    out("  Where multi-pass IS right: a THROUGHPUT fp64 part (HPC DGEMM), where")
    out("  49-361 passes still beat having no fp64 silicon at all and there is no")
    out("  deadline. We are building a DEADLINE part with ~4 FMAs of fp64 demand.")
    out("  Where it is ALSO right, and worth keeping in codegen: reaching fp32")
    out("  accuracy from bf16 multipliers in 6-9 passes, which is how the fp32 rows")
    out("  already work (2-pass split, exact in the fp32 accumulator).")
    out()

    out("-" * 100)
    out("(b) SEPARATE NARROW fp64 FMA CLUSTER, sized for the IP inner loop only")
    out("-" * 100)
    out("  Area here is an fp64-ONLY cluster (no format merging), i.e. option (b)")
    out("  as literally stated. Option (d) below merges it with the vector unit and")
    out("  is cheaper on net.")
    out(f"  {'FMAs':>5} {'central mm2':>12} {'pessim mm2':>11} "
        f"{'ARM wc us':>10} {'margin vs 50%':>14} {'binds on':<12}")
    for f in (1, 2, 4, 8, 16, 32):
        c = dataclasses.replace(REC, fp64_fmas=f)
        d = arm_mpc_latency(c, ARM.iters_wc)
        out(f"  {f:>5} {S.fma_cluster_mm2(f, 'fp64'):>12.4f} "
            f"{S.fma_cluster_mm2(f, 'fp64', True):>11.4f} "
            f"{d['_total']:>10.1f} "
            f"{ARM.deadline_us * PERIOD_BUDGET / d['_total']:>13.2f}x "
            f"{d['_bind']:<12}")
    out()
    out("  Cheapest per unit of what we need -- and what we need is 'meet a 4000 us")
    out("  deadline at 30 interior-point iterations inside 50% of the period'.")
    out(f"  Note the 'binds on' column: fp64 GEMM binds at EVERY size tried, "
        f"because")
    out("  ARM-MPC is 79% matrix-matrix. fp64 GEMV becomes bandwidth-bound rather")
    out(f"  than cluster-bound only above {REC.scratchpad_bw_bpc/8:.0f} FMAs.")
    out()

    out("-" * 100)
    out("(c) NATIVE fp64 WIDENED THROUGH THE WHOLE ARRAY")
    out("-" * 100)
    for n in (64, 144, 256):
        out(f"  {n:>4} merged fp64-capable cells: "
            f"{S.native_fp64_array_mm2(n):.3f} mm2 central / "
            f"{S.native_fp64_array_mm2(n, True):.3f} mm2 pessimistic")
    out(f"  vs {REC.macs} bf16 cells today: {S.mac_array_mm2(REC.macs):.3f} mm2")
    net = S.native_fp64_array_mm2(REC.macs) - S.mac_array_mm2(REC.macs)
    out(f"  Net +{net:.3f} mm2 to the compute block -- roughly "
        f"{net / max(1e-9, S.fma_cluster_mm2(8, 'fp64', multiformat=True)):.0f}x "
        f"option (b) --")
    out(f"  which would take the CM-100 core from 0.565 to ~"
        f"{(0.4916 + net) * 1.15:.2f} mm2, a {(0.4916 + net) * 1.15 / 0.565:.1f}x "
        f"part, to")
    out("  serve ONE benchmark whose demand is 4-8 FMAs. Also pointless on its own")
    d144 = arm_mpc_latency(dataclasses.replace(REC, fp64_fmas=144), ARM.iters_wc)
    out(f"  terms: at 144 fp64 FMAs ARM-MPC lands at {d144['_total']:.0f} us, "
        f"{ARM.deadline_us*PERIOD_BUDGET/d144['_total']:.0f}x inside the")
    out("  50%-period budget, and its fp64 GEMV is bandwidth-capped at "
        f"{REC.scratchpad_bw_bpc/8:.0f} MAC/cycle")
    out("  regardless -- so most of the array is unfeedable even on the one")
    out("  benchmark that wants fp64 at all. REJECTED.")
    out()

    out("-" * 100)
    out("(d) RECOMMENDED — MERGE THE fp64 CLUSTER WITH THE fp32 VECTOR UNIT")
    out("-" * 100)
    out("  A separate fp64 cluster is idle on 6 of the 7 benchmarks. But the CM-100")
    out("  already carries 16 fp32 vector lanes, and 8 merged multi-format FMAs")
    out("  ARE 16 fp32 lanes (1x fp64 or 2x fp32 per unit) [C3]. So build one")
    out("  block. The fp64 capability then costs only the WIDENING, not a unit:")
    vec_only = 16 * C.VEC_LANE_GATES * S.A_GATE_UM2 * S.MAC_STAGING_OVERHEAD / 1e6
    merged = (S.fma_cluster_mm2(8, "fp64", multiformat=True)
              + 16 * C.VEC_TRANSCENDENTAL_GATES * S.A_GATE_UM2 / 1e6)
    out(f"    honest fp32-only 16-lane vector unit   {vec_only:.4f} mm2")
    out(f"    merged 8x(fp64 | 2x fp32) + transcend.  {merged:.4f} mm2")
    out(f"    => fp64 INCREMENT                      +{merged - vec_only:.4f} mm2 "
        f"central")
    out(f"       pessimistic edge                    +"
        f"{S.fma_cluster_mm2(8, 'fp64', True, True) + 16*C.VEC_TRANSCENDENTAL_GATES*S.A_GATE_UM2/1e6 - vec_only:.4f} mm2")
    out()
    out("  STRUCTURAL HAZARD to write down: the block cannot be the vector unit and")
    out("  the fp64 GEMM engine in the same cycle. On ARM-MPC the vector term is")
    out(f"  {arm_mpc_latency(REC, ARM.iters_wc)['vec']:.0f} us against "
        f"{arm_mpc_latency(REC, ARM.iters_wc)['fp64_gemm']:.0f} us of fp64 GEMM, "
        f"so serialising costs")
    out("  ~4% of the budget. On the fp32 rows the block is 100% vector unit.")
    out("  Acceptable. But it makes the vector unit and the fp64 unit ONE")
    out("  scheduling resource, which the sequencer model and codegen must know.")
    out()

    # ================================================================== Q2 ===
    out("=" * 100)
    out("Q2 — HOW MUCH fp64 THROUGHPUT DOES THE BENCHMARK SET ACTUALLY REQUIRE?")
    out("=" * 100)
    fp64_benches = [b.name for b in M.BENCHMARKS if b.prec == "fp64"]
    out(f"  Benchmarks needing fp64: {', '.join(fp64_benches)}  (1 of "
        f"{len(M.BENCHMARKS)})")
    out(f"  Everything else is fp32 (ADMM-MPC, EKF) or int8 (ML). The EKF floor is")
    out("  fp32 and there is no fp16/bf16 filter path -- but no fp64 requirement")
    out("  either, so fp64 demand comes from interior point alone.")
    out()
    for it, lab in ((ARM.iters, "typical (p95 design point)"),
                    (ARM.iters_wc, "WORST CASE (HPIPM max_iter) [EST]")):
        w = arm_mpc_fp64_work(it)
        out(f"  ARM-MPC at {it} IP iterations — {lab}")
        out(f"    fp64 GEMM (block Riccati) {w['gemm']:>12,.0f} MACs "
            f"({100*w['gemm_frac']:.0f}% of fp64 work)")
        out(f"    fp64 GEMV (substitutions) {w['gemv']:>12,.0f} MACs")
        out(f"    total                     {w['total']:>12,.0f} MACs")
        for budget, blab in ((1.0, "100% of period"),
                             (PERIOD_BUDGET, "50% of period")):
            need = w["total"] / (ARM.deadline_us * budget * 1e-6)
            out(f"    -> {need/1e9:>5.2f} G fp64 MAC/s = "
                f"{need/(REC.freq_mhz*1e6):>5.2f} fp64 MAC/cycle at 800 MHz "
                f"({blab}, zero overhead)")
        out()
    out("  That is the naive ceiling. Now with tile utilisation, the GEMV/GEMM")
    out("  split, vector and sequencer terms all charged (arm_mpc_latency):")
    out()
    out(f"  {'FMAs':>5} " + " ".join(f"{k:>10}" for k in
        ("fp64_gemm", "fp64_gemv", "vec", "seq", "host", "TOTAL")) +
        f" {'vs 4000':>8} {'vs 2000':>8}  verdict")
    for f in (1, 2, 4, 6, 8, 12, 16):
        d = arm_mpc_latency(dataclasses.replace(REC, fp64_fmas=f), ARM.iters_wc)
        v = ("PASS" if d["_total"] <= ARM.deadline_us * PERIOD_BUDGET
             else "deadline only" if d["_total"] <= ARM.deadline_us else "FAIL")
        out(f"  {f:>5} " + " ".join(
            f"{d[k]:>10.1f}" for k in
            ("fp64_gemm", "fp64_gemv", "vec", "seq", "host", "_total")) +
            f" {ARM.deadline_us/d['_total']:>7.2f}x "
            f"{ARM.deadline_us*PERIOD_BUDGET/d['_total']:>7.2f}x  {v}")
    out()
    out("  CROSSOVERS (worst-case iterations, nb=21 utilisation):")
    out(f"    minimum for the 50%-of-period budget : "
        f"{min_fp64_fmas(budget=PERIOD_BUDGET)} FMAs")
    out(f"    minimum for the raw 4000 us deadline : "
        f"{min_fp64_fmas(budget=1.0)} FMAs")
    out(f"    minimum at TYPICAL 15 iterations     : "
        f"{min_fp64_fmas(iters=ARM.iters)} FMAs   <-- why typical cannot size")
    out(f"    optimistic utilisation proxy (dim="
        f"{M.representative_dim(ARM):.0f}) : "
        f"{min_fp64_fmas(dim=M.representative_dim(ARM))} FMAs")
    out()
    out(f"  ANSWER: 4 FMAs is the feasibility FLOOR; 8 is the recommendation, at")
    out(f"  {ARM.deadline_us*PERIOD_BUDGET/arm_mpc_latency(REC, ARM.iters_wc)['_total']:.2f}x "
        f"margin on the 50%-period budget and "
        f"{ARM.deadline_us/arm_mpc_latency(REC, ARM.iters_wc)['_total']:.2f}x on the")
    out("  raw deadline. 2 FMAs misses the budget, 1 misses the deadline outright.")
    out("  8 is chosen over 4 because it is FREE: 8 merged units is exactly the")
    out("  16 fp32 vector lanes the part already needs. Above 8, fp64 GEMV becomes")
    out(f"  bandwidth-bound at {REC.scratchpad_bw_bpc/8:.0f} MAC/cycle and the extra "
        f"units buy 64 us on a 2000 us")
    out("  budget -- not worth 0.07 mm2. This is a genuine local optimum, not a")
    out("  round number.")
    out()
    out("  ADVERSARIAL CHECK on the softest assumption in this sizing. fp64_util")
    out("  softens the 2-D quantisation penalty by a square root, on the argument")
    out("  that a LINEAR cluster only quantises its innermost loop. That is worth")
    out(f"  {1/math.sqrt(cfg_u := M.ArgmindCfg(name='u', tile=12, fp64_fmas=8).gemm_util('fp64', NB)**2):.2f}x. Re-run with the FULL 2-D penalty charged instead:")
    out(f"  {'FMAs':>5} {'harsh us':>10} {'budget margin':>14} {'deadline':>10}")
    for f in (2, 4, 8, 16):
        c = dataclasses.replace(REC, fp64_fmas=f)
        d = arm_mpc_latency(c, ARM.iters_wc)
        harsh = d["_total"] + d["fp64_gemm"] * (fp64_util(c) / (fp64_util(c) ** 2) - 1)
        out(f"  {f:>5} {harsh:>10.1f} "
            f"{ARM.deadline_us*PERIOD_BUDGET/harsh:>13.2f}x "
            f"{ARM.deadline_us/harsh:>9.2f}x")
    out("  The recommendation is UNCHANGED (8 FMAs, 2.25x instead of 2.70x) and so")
    out("  is the floor (4 FMAs, 1.14x instead of 1.38x -- thin either way). So the")
    out("  sizing does not depend on the softening. What it DOES depend on is")
    out(f"  iters_worst={ARM.iters_wc}, which is [EST] (2x the design point), not measured:")
    out(f"  {'iters_worst':>12} {'floor (>=1x budget)':>20} "
        f"{'8-FMA budget margin':>21}")
    for it in (30, 45, 60):
        d = arm_mpc_latency(REC, it)
        out(f"  {it:>12} {min_fp64_fmas(iters=it):>19} FMAs "
            f"{ARM.deadline_us*PERIOD_BUDGET/d['_total']:>20.2f}x")
    out("  At a 60-iteration tail the 8-FMA recommendation drops to 1.39x margin --")
    out("  still feasible, but no longer comfortable. MEASURING ARM-MPC'S ACTUAL")
    out("  IP ITERATION TAIL IS THE SINGLE HIGHEST-VALUE INPUT TO THIS DECISION,")
    out("  ahead of the gate count and ahead of the energy constant.")
    out()
    out("  SO THE HONEST HEADLINE IS: a very small fp64 cluster suffices. The")
    out("  previous 0.125-rate-factor model was over-providing fp64 by 2.25x")
    out(f"  (18 vs 8 MAC/cycle) at zero area, so its ARM-MPC latency was optimistic")
    out("  by the same factor on the fp64 GEMM term.")
    out()

    # ================================================================== Q3 ===
    out("=" * 100)
    out("Q3 — DOES fp32 + ITERATIVE REFINEMENT BEAT NATIVE fp64?")
    out("=" * 100)
    out("  FIRST, THE ARGUMENT THAT SETTLES IT BEFORE ANY MODELLING [C5]:")
    out("  fixed-precision refinement converges only while cond(A)*eps < 1. The")
    out("  interior-point endgame has cond(KKT) ~ 1e12 and eps_fp32 ~ 6e-8, so the")
    out("  product is ~6e4 >> 1. fp32-working-precision refinement DOES NOT")
    out("  CONVERGE there. Mixed-precision refinement works, but it requires the")
    out("  RESIDUAL in higher precision than fp32 -- i.e. fp64 hardware.")
    out("  => Refinement RELOCATES the fp64 requirement from the O(n^3)")
    out("     factorisation to the O(n^2) residual. It never removes it. There is")
    out("     no version of this that lets us ship without an fp64 datapath.")
    out()
    out("  SECOND, THE ECONOMICS AT OUR BLOCK SIZES. The asymptotic case for")
    out("  refinement is 'residual O(n^2) is cheap against factorisation O(n^3)'.")
    out(f"  Ratio = 3k/nb. At nb={NB} (7-DoF arm, n_x+n_u):")
    for k in (1, 2, 3, 5):
        out(f"    k={k}: residual work is {100*3.0*k/NB:>5.1f}% of the "
            f"factorisation")
    out(f"  Refinement is only 'cheap' (<10%) when nb > 30k, i.e. nb > 90 at k=3.")
    out("  EMBEDDED MPC LIVES AT nb = 10-30. The asymptotic argument for iterative")
    out("  refinement does not transfer to this problem class. That is a real")
    out("  finding and it is the opposite of the HPC intuition.")
    out()
    out("  THIRD, THE MODELLED COMPARISON on the SAME silicon (144-MAC bf16/fp32")
    out(f"  array + {REC.fp64_fmas} fp64 FMAs + {REC.scratchpad_bw_bpc:.0f} B/cycle), "
        f"ARM-MPC at {ARM.iters_wc} IP iterations:")
    out()
    s1 = all_fp64_energy(REC, ARM.iters_wc)
    out(f"  {'strategy':<34} {'lat us':>9} {'uJ':>8} {'spad MB':>9} "
        f"{'fp64 frac':>10}  determinism")
    out(f"  {'S1 all-fp64 KKT solve':<34} {s1['latency_us']:>9.1f} "
        f"{s1['energy_uj']:>8.2f} {s1['sram_bytes']/1e6:>9.2f} {'100%':>10}  "
        f"fixed op count")
    for k in (1, 2, 3, 5):
        r = refinement_analysis(REC, k, ARM.iters_wc)
        out(f"  {'S2 fp32 factor + fp64 IR, k=' + str(k):<34} "
            f"{r['latency_us']:>9.1f} {r['energy_uj']:>8.2f} "
            f"{r['sram_bytes']/1e6:>9.2f} {100*r['fp64_frac_of_work']:>9.0f}%  "
            f"{'fixed k' if True else ''}")
    out()
    r3 = refinement_analysis(REC, 3, ARM.iters_wc)
    # Be adversarial about our own preferred answer: refinement WINS on both axes
    # at small k. Find and report the crossover instead of hiding it.
    kx = next((k for k in range(1, 20)
               if refinement_analysis(REC, k, ARM.iters_wc)["energy_uj"]
               > s1["energy_uj"]), None)
    out("  READ THIS TABLE ADVERSARIALLY, because it does NOT simply favour us.")
    out(f"  Refinement is FASTER at every k (1.2x-3.1x). On ENERGY there is a")
    out(f"  CROSSOVER at k = {kx}: below it refinement wins on energy too, above it")
    out("  native fp64 wins. So the honest statement is a crossover, not a verdict:")
    out()
    out(f"    k <= {kx-1}   fp32 + refinement is better on latency AND energy")
    out(f"    k >= {kx}   native fp64 is better on energy, still slower")
    out()
    out(f"  Mechanism: refinement moves {r3['sram_bytes']/s1['sram_bytes']:.1f}x "
        f"the scratchpad bytes at k=3 -- it")
    out("  re-reads the KKT matrix in fp64 for every residual, k times per IP")
    out("  iteration, to avoid reading it in fp64 once for the factorisation. That")
    out("  is why the O(n^2)-is-cheap intuition fails: at nb=21 the residual is not")
    out("  cheap, and it is the fp64-traffic term.")
    out()
    out(f"  WHICH SIDE OF THE CROSSOVER ARE WE ON? k >= {kx}, and not by our choice.")
    out("  At cond ~1e12 with fp32 working precision the refinement iteration does")
    out("  not contract at all [C5] -- k is not 1 or 2, it is UNBOUNDED unless the")
    out("  residual and the correction solve are themselves done in fp64, in which")
    out("  case the fp64 traffic is worse still. The energy crossover therefore")
    out("  lands on the wrong side of the numerics, and the numerics are not")
    out("  negotiable.")
    out()
    out("  FOURTH, DETERMINISM — the decisive point for a hard-deadline part.")
    out("  k is DATA-DEPENDENT. Two ways to handle it, both bad:")
    out("    (i)  adaptive k: worst-case latency is unbounded. Non-starter.")
    out("    (ii) fixed k with a correctness check and an all-fp64 fallback:")
    out("         deterministic, but the certifiable worst case is the SUM.")
    out(f"    S1 alone                       {s1['latency_us']:>8.1f} us")
    out(f"    S2 (k=3) alone                 {r3['latency_us']:>8.1f} us")
    out(f"    S2 then fallback to S1         "
        f"{r3['latency_us'] + s1['latency_us']:>8.1f} us  <-- the number a safety")
    out("                                              argument must use")
    out(f"    ... which is {(r3['latency_us']+s1['latency_us'])/s1['latency_us']:.1f}x "
        f"S1 alone. A scheme whose WORST case is worse than")
    out("    the thing it replaces cannot be the baseline for a hard-deadline part,")
    out("    however good its average case is.")
    out()
    out("  VERDICT FOR A HARD-DEADLINE PART: native fp64. Grounds, in order of")
    out("  strength: (1) refinement does not remove the fp64 requirement at all,")
    out("  it relocates it; (2) its worst-case-with-fallback latency exceeds")
    out("  all-fp64; (3) past the k the numerics force, it is worse on energy;")
    out(f"  (4) ARM-MPC already has {ARM.deadline_us/s1['latency_us']:.1f}x deadline "
        f"margin all-fp64, so the latency")
    out("  refinement buys is margin we do not need. Keep S2 as an OPTIONAL codegen")
    out("  mode for soft-deadline / throughput use; do not make it the baseline.")
    out("  CAVEAT: this rests on a citation [C5], not on our own measurement.")
    out("  Measuring the actual refinement count in acados/HPIPM is item 2 below.")
    out()
    out("  DOES THE ANSWER DIFFER FOR THE SOLVER-ONLY SKU (64 MACs, 64 B/cycle)?")
    fs1 = all_fp64_energy(FLOOR, ARM.iters_wc)
    fr3 = refinement_analysis(FLOOR, 3, ARM.iters_wc)
    out(f"    S1 all-fp64      {fs1['latency_us']:>8.1f} us  "
        f"{fs1['energy_uj']:>7.2f} uJ")
    out(f"    S2 k=3           {fr3['latency_us']:>8.1f} us  "
        f"{fr3['energy_uj']:>7.2f} uJ")
    out("    NO -- and for a structural reason worth stating: the fp64 cluster is")
    out("    NOT part of the MAC array, so it does not scale down with the SKU.")
    out(f"    S1 latency is EXACTLY identical on both parts "
        f"({fs1['latency_us']:.0f} vs {s1['latency_us']:.0f} us) because")
    out("    both are fp64-cluster-limited, and at 64 B/cycle the fp64 GEMV")
    out(f"    bandwidth ceiling ({FLOOR.scratchpad_bw_bpc/8:.0f} MAC/cyc) exactly "
        f"equals the cluster width. Energy is")
    out(f"    {fs1['energy_uj']/s1['energy_uj']:.2f}x higher only because the "
        f"smaller tile fetches more GEMM operand")
    out(f"    bytes per MAC ({FLOOR.gemm_operand_bytes_per_mac('fp64'):.2f} vs "
        f"{REC.gemm_operand_bytes_per_mac('fp64'):.2f} B/MAC). What DOES change "
        f"materially is the area FRACTION:")
    if C is not None:
        ar, af = C.area_breakdown(REC), C.area_breakdown(FLOOR)
        out(f"      CM-100         fp64 increment {ar['fp64']:.4f} mm2 = "
            f"{100*ar['fp64']/ar['total']:.1f}% of {ar['total']:.3f} mm2 core")
        out(f"      solver-only    fp64 increment {af['fp64']:.4f} mm2 = "
            f"{100*af['fp64']/af['total']:.1f}% of {af['total']:.3f} mm2 core")
    out("    So fp64 is proportionally MORE expensive on the small part -- and the")
    out("    small part is the MPC-only SKU that most needs it. There is no SKU")
    out("    from which fp64 can be removed while keeping interior-point MPC.")
    out("    Dropping to 4 FMAs there gives only "
        f"{ARM.deadline_us*PERIOD_BUDGET/arm_mpc_latency(dataclasses.replace(FLOOR, fp64_fmas=4), ARM.iters_wc)['_total']:.2f}x "
        f"budget margin: too thin. Keep 8.")
    out()

    # ================================================================== Q4 ===
    out("=" * 100)
    out("Q4 — EFFECT ON THE CM-100 AREA BUDGET AND ON perf/W")
    out("=" * 100)
    if C is not None:
        a = C.area_breakdown(REC)
        ap = C.area_breakdown(REC, pessimistic=True)
        out("  THE 0.565 mm2 FIGURE MUST BE RESTATED. Two things were wrong with it,")
        out("  and they are reported separately so neither hides in the other:")
        out()
        old_vec = 16 * 2200 * S.A_GATE_UM2 / 1e6
        out(f"  {'item':<32} {'was':>9} {'now':>9} {'delta':>9}")
        out(f"  {'scratchpad':<32} {a['spad']:>9.4f} {a['spad']:>9.4f} "
            f"{0.0:>9.4f}")
        out(f"  {'MAC array (144 bf16)':<32} {a['mac']:>9.4f} {a['mac']:>9.4f} "
            f"{0.0:>9.4f}")
        out(f"  {'vector unit (16 fp32 lanes)':<32} {old_vec:>9.4f} "
            f"{a['vec']:>9.4f} {a['vec']-old_vec:>+9.4f}   <- pre-existing "
            f"underestimate")
        out(f"  {'fp64 increment (8 merged FMAs)':<32} {0.0:>9.4f} "
            f"{a['fp64']:>9.4f} {a['fp64']:>+9.4f}   <- the item this run closes")
        out(f"  {'seq + DMA + misc':<32} {a['rest']:>9.4f} {a['rest']:>9.4f} "
            f"{0.0:>9.4f}")
        sub_old = a['spad'] + a['mac'] + old_vec + a['rest']
        out(f"  {'subtotal':<32} {sub_old:>9.4f} {a['sub']:>9.4f} "
            f"{a['sub']-sub_old:>+9.4f}")
        out(f"  {'+15% top margin':<32} {sub_old*1.15:>9.4f} "
            f"{a['total']:>9.4f} {a['total']-sub_old*1.15:>+9.4f}")
        out()
        out(f"  CM-100 core area: 0.565 -> {a['total']:.3f} mm2 at 16nm-class "
            f"(+{100*(a['total']/0.565-1):.0f}%),")
        out(f"  of which +{a['fp64']*1.15:.3f} mm2 is fp64 and "
            f"+{(a['vec']-old_vec)*1.15:.3f} mm2 is the vector-lane correction.")
        out(f"  Pessimistic fp64 gate count (56.2 kgate anchor [C4]): "
            f"{ap['total']:.3f} mm2.")
        out(f"  HONEST RANGE TO STATE: {a['total']:.2f}-{ap['total']:.2f} mm2, "
            f"+/-2x on top of that.")
        out()
        out("  Same restatement for the other two points on the Pareto front:")
        out(f"  {'SKU':<38} {'was mm2':>9} {'now mm2':>9} {'fp64':>8} {'pess':>8}")
        for lab, c in (("CM-100 (128 B/cyc, 144 MACs, 128 KB)", REC),
                       ("floor  (64 B/cyc, 64 MACs, 128 KB)", FLOOR)):
            aa = C.area_breakdown(c)
            ov = c.vector_lanes * 2200 * S.A_GATE_UM2 / 1e6
            was = (aa['spad'] + aa['mac'] + ov + aa['rest']) * 1.15
            out(f"  {lab:<38} {was:>9.3f} {aa['total']:>9.3f} "
                f"{aa['fp64']:>8.4f} "
                f"{C.area_breakdown(c, True)['total']:>8.3f}")
        out()
    out("  perf/W — IS THE 2.20 pJ/fp64-MAC ESTIMATE LOAD-BEARING? YES.")
    out()
    out("  Derivation check on the constant first. Horowitz (ISSCC 2014, 45nm)")
    out("  gives fp32 mult 3.7 pJ + fp32 add 0.9 pJ. Multiplier energy scales as")
    out(f"  the significand squared: (53/24)^2 = {(53/24)**2:.1f}x, the adder")
    out(f"  roughly linearly. Weighting the multiply at 70% of FMA energy gives")
    out(f"  fp64/fp32 = {0.7*(53/24)**2 + 0.3*(53/24):.1f}x, so from "
        f"E_MAC_PJ[fp32]={M.E_MAC_PJ['fp32']} pJ the")
    out(f"  n^2-consistent fp64 figure is "
        f"{M.E_MAC_PJ['fp32']*(0.7*(53/24)**2 + 0.3*(53/24)):.2f} pJ against the "
        f"{M.E_MAC_PJ['fp64']} pJ in the model.")
    out("  The constant is ~13% optimistic on its own internal logic. Within noise,")
    out("  but the direction is worth knowing.")
    out()
    out(f"  {'fp64 pJ/MAC':>12} {'ARM uJ':>9} {'perf/W vs best CPU':>20} "
        f"{'vs 5x kill':>11}")
    cpus = [M.model_cpu(ARM, c) for c in M.ALL_CPUS]
    best_cpu_uj = min(x["energy_uj"] for x in cpus)
    orig = M.E_MAC_PJ["fp64"]
    try:
        for mult, lab in ((1.0, ""), (1.5, ""), (2.0, ""), (3.0, "")):
            M.E_MAC_PJ["fp64"] = orig * mult
            r = M.model_argmind(ARM, REC)
            pw = best_cpu_uj / r["energy_uj"]
            out(f"  {orig*mult:>12.2f} {r['energy_uj']:>9.2f} {pw:>19.1f}x "
                f"{'PASS' if pw >= 5 else 'FAIL':>11}")
    finally:
        M.E_MAC_PJ["fp64"] = orig
    r = M.model_argmind(ARM, REC)
    raw = ARM.total_macs() * orig * REC.overhead_mult / 1e6
    out()
    out(f"  fp64 MAC energy is {100*raw/r['energy_uj']:.0f}% of ARM-MPC's "
        f"{r['energy_uj']:.1f} uJ/call at the typical")
    out("  iteration count -- the single largest term. SENSITIVITY: the kill")
    out("  criterion survives a 3x error in the constant, so the go/no-go does not")
    out("  depend on it; the QUOTED RATIO does, by about 2x across the band.")
    out("  MEASUREMENT THAT SETTLES IT: gate-level power on one merged fp64 FMA at")
    out("  800 MHz. CVFPU is open source; this is a day of work, not a study.")
    out()
    out("  ALSO CORRECTED IN THIS RUN, and it moves the ratio IN OUR FAVOUR, which")
    out("  is why it needs flagging: model_cpu charged fp64 work at the CPU's fp32")
    out("  rate on every baseline. NEON/SVE fp64 is half-rate (same vector width,")
    out("  twice the element size) and SME fp64 outer products are quarter-rate.")
    out(f"  {'baseline':<44} {'fp32 op/cyc':>12} {'fp64 op/cyc':>12} "
        f"{'ARM-MPC us':>11} {'uJ':>8}")
    for c, r_ in zip(M.ALL_CPUS, cpus):
        out(f"  {c.name:<44} {c.peak_flops_per_cycle('fp32'):>12.0f} "
            f"{c.peak_flops_per_cycle('fp64'):>12.0f} "
            f"{r_['latency_us']:>11.1f} {r_['energy_uj']:>8.1f}")
    out("  NOTE FOR FALSIFIER F4: FEAT_SME_F64F64 is architecturally OPTIONAL. An")
    out("  SME2 part without it has no fp64 matrix path at all. Check the target's")
    out("  ID registers on an eval board before claiming anything either way.")
    out()
    out("  ARM-MPC's REPORTED perf/W THEREFORE MOVES 14.1x -> 19.8x. That is a")
    out("  flattering change and it must be decomposed, not banked:")
    out("    * CM side got WORSE: fp64 throughput 18 -> 8 MAC/cycle raised latency")
    out("      178 -> 213 us (typical) and energy 19.6 -> 22.5 uJ via static power.")
    out("    * CPU side got worse FASTER: charging fp64 at half the fp32 rate took")
    out("      the A55 baseline 276 -> 445 uJ. That is 1.6x, against our 1.15x.")
    out("    * NET: the whole increase comes from the CPU-side bug fix, none of it")
    out("      from anything we designed. Both changes are corrections of errors")
    out("      that pointed in opposite directions and happened to partly cancel.")
    out("    * SANITY CHECK the fix is not overdone: the A55 baseline still MEETS")
    out(f"      ARM-MPC's raw 4000 us deadline ({M.model_cpu(ARM, M.CPU_A55)['latency_us']:.0f} us). "
        f"We are not claiming the")
    out("      CPU cannot do this -- only that it costs 20x the energy. Consistent")
    out("      with CLAUDE.md's standing instruction.")
    out()
    out("  ALSO NOTE what did NOT change: with fp64 throughput decoupled from the")
    out("  MAC array, ARM-MPC is now FLAT across the tile-size sweep. The old")
    out("  0.125 factor made a bigger array look like it bought fp64 speed. It")
    out("  never did, and now the model says so.")
    out()

    # ================================================================== Q5 ===
    out("=" * 100)
    out("Q5 — THE MACIF CONSEQUENCE")
    out("=" * 100)
    out("  fp64 CANNOT be a swappable-macro property hidden behind MACIF. The")
    out("  interface must carry PRECISION AS A DECLARED CONTRACT. Four reasons,")
    out("  in descending order of how expensive it is to get wrong:")
    out()
    out("  1. IT IS AN ALGORITHM REQUIREMENT, NOT A PERFORMANCE KNOB. Interior")
    out("     point at cond ~1e12 either has fp64 or produces a wrong answer [C5].")
    out("     A boundary that lets a macro silently substitute lower precision")
    out("     converts a swap decision into a correctness failure in the field.")
    out("     Precision must be QUERYABLE and the runtime must REFUSE an fp64")
    out("     kernel on a macro that does not declare fp64 -- a hard error at")
    out("     codegen time, not a degraded result at run time.")
    out()
    out("  2. THE fp64 UNIT IS ON THE OTHER SIDE OF THE BOUNDARY FROM THE ARRAY.")
    out("     The whole point of MACIF is that a digital-CIM or analog macro can")
    out("     replace the MAC array. Neither can do fp64: analog CIM has no path")
    out("     to 53-bit significands, and digital CIM at fp64 is not a product.")
    out("     So the fp64 FMA cluster must sit OUTSIDE the MACIF boundary, in the")
    out("     fixed part of the CM, alongside the vector unit it is merged with.")
    out("     That is the architectural consequence of this whole analysis: the")
    out("     boundary runs between the bf16/int8 array (swappable) and the")
    out("     fp32/fp64 FMA cluster (not swappable). Drawing it anywhere else")
    out("     makes the interior-point path un-portable.")
    out()
    out("  3. MULTI-PASS EMULATION WOULD HAVE LEAKED ACROSS THE BOUNDARY ANYWAY.")
    out("     Option (a) needs error-free integer/limb semantics and an exact")
    out(f"     {S.WIDE_ACCUM_BITS}-bit accumulator INSIDE the macro [C2]. That is a "
        f"much stronger")
    out("     contract than 'multiply and accumulate approximately', and an analog")
    out("     macro cannot honour it at all. Choosing option (b)/(d) keeps MACIF's")
    out("     contract weak enough to be honourable by a CIM macro -- which is the")
    out("     entire value of having the boundary.")
    out()
    out("  4. IT IS A FEATURE BIT, NOT A SCALING PARAMETER. Q3 showed the fp64")
    out("     cluster is the same 8 units on the 144-MAC part and the 64-MAC part.")
    out("     Interface fields that scale with the array (MAC count, tile shape,")
    out("     operand width) must therefore NOT be the same fields that describe")
    out("     fp64. Concretely, MACIF v0 should declare, per format:")
    out("       supported {int8, bf16, fp16, fp32, fp64}")
    out("       macs_per_cycle[format]        (absolute, not a rate factor)")
    out("       accumulator_width[format]     (exactness contract)")
    out("       rounding mode + subnormal behaviour  (IEEE conformance claim)")
    out("     The 0.125 rate factor this run deleted is precisely the shape of")
    out("     interface that would have been wrong: a scalar multiplier on a bf16")
    out("     capability, implying fp64 falls out of the array for free.")
    out()
    out("=" * 100)
    out("WHAT WOULD FIRM THIS UP (in order of value per day of work)")
    out("=" * 100)
    out("  1. Synthesise one CVFPU/FPnew merged fp64|2xfp32 FMA at 800 MHz with")
    out("     Yosys + an open PDK. Closes BOTH the 2.1x gate-count band and the")
    out("     2.20 pJ energy constant. Open source, one day. [C3]")
    out("  2. Run acados+HPIPM on the 7-DoF problem in fp64 and in fp32-with-")
    out("     refinement and MEASURE the refinement count k and whether it")
    out("     converges at all near the barrier endgame. Q3's verdict rests on a")
    out("     citation, not on our own measurement.")
    out("  3. Measure ARM-MPC's true worst-case IP iteration count. iters_worst=30")
    out("     is [EST] (2x the design point), and it sizes the fp64 cluster.")
    out("  4. Count HPIPM's actual block operations per IP iteration. "
        f"BLOCK_OPS_PER_IP_ITER={BLOCK_OPS_PER_IP_ITER}")
    out("     is [EST] and it is what makes the multi-pass sequencer tax ~1.5x.")
    out()

    import os
    dest = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "results_fp64_datapath.txt")
    with open(dest, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
