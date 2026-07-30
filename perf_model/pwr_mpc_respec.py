"""
PWR-MPC resolution: re-specification, 16-bit operands, and the MCU-class tier.
=============================================================================

Three threads converged on one benchmark row and none had been settled:

  (1) PWR-MPC was specified as a continuous-control-set condensed QP solved by
      ADMM at 20 kHz. The deployed-practice literature says the socket runs
      FINITE-CONTROL-SET MPC -- enumerate |U|^Np switching sequences, no QP at
      all -- or explicit (offline piecewise-affine) MPC. If that is right, the
      row misrepresents its socket, and it was driving four separate
      architecture conclusions.

  (2) The row sat at 0.974x -- the only solver row below the fastest CPU
      baseline. precision_study.py found fp16 operands 10x better than bf16 for
      solver matrices. The ADMM rows are bandwidth-bound, so halving operand
      width should roughly double effective bandwidth. Does that fix it?

  (3) The socket is the one most likely to be MCU-class, where CLAUDE.md records
      that zero-copy and residency have nothing to bite on -- yet
      PHASE0_FINDINGS credited zero-copy with 1.9x and with deciding this row's
      deadline feasibility.

Order matters: (1) can dissolve (2) and (3), so it is answered first.

ALL LATENCY/ENERGY/AREA FIGURES ARE MODELLED ESTIMATES at a 16nm-class node,
+/-2x, and are NOT QUOTABLE EXTERNALLY. The numerical results imported from
precision_study.py are MEASURED numerics (not silicon). Deployed-practice claims
are CITED; the citations are listed at the end of the report.
"""

from __future__ import annotations

import dataclasses as dc

try:
    from . import argmind_model as M
    from . import cm100_sizing as Z
    from . import sequencer as Q
except ImportError:                       # run as a script
    import argmind_model as M
    import cm100_sizing as Z
    import sequencer as Q


REC = M.CFG_100
BUDGET = Q.PERIOD_BUDGET                  # 0.50 of the control period
DESC_RULE = 8.0                           # >= 2 * seq_setup_cycles


def by_name(name):
    return next(b for b in M.BENCHMARKS + M.OFF_SET if b.name == name)


# ---------------------------------------------------------------------------
# Shared analysis helpers
# ---------------------------------------------------------------------------

def binding_resource(b: M.Benchmark, cfg: M.ArgmindCfg) -> tuple[str, dict]:
    """Which resource saturates first on THIS workload.

    Reported per row rather than as a peak figure, because the peak figure is the
    thing this project has repeatedly caught itself being misled by.
    """
    dim = M.representative_dim(b)
    bpm = (cfg.gemm_operand_bytes_per_mac(b.prec)
           if b.gemm_bytes_per_mac is None else b.gemm_bytes_per_mac)
    array_rate = cfg.macs_per_cycle(b.prec) * cfg.gemm_util(b.prec, dim)
    gemm_bw_rate = cfg.scratchpad_bw_bpc / bpm if bpm > 0 else float("inf")
    d = M.descriptor_cost(b, cfg)
    cyc = {
        "gemm_array": (b.gemm_macs_per_iter / array_rate
                       if b.gemm_macs_per_iter and array_rate < gemm_bw_rate
                       else 0.0),
        "gemm_bw": (b.gemm_macs_per_iter / gemm_bw_rate
                    if b.gemm_macs_per_iter and gemm_bw_rate <= array_rate
                    else 0.0),
        "gemv_bw": M.split_gemv_cycles(b, cfg, b.gemv_macs_per_iter),
        "vector": (b.vector_ops_per_iter
                   + 4.0 * b.transcendental_per_iter) / cfg.vector_lanes,
        "seq_overhead": d["seq_overhead_cycles"],
    }
    per_call = M.model_argmind(b, cfg)
    cyc_total = sum(cyc.values())
    # Per-call overhead expressed in cycles so it competes on the same axis.
    cyc["call_overhead"] = (cfg.call_overhead_us * 1000.0 / cfg.cycle_ns
                            / max(1, b.iters))
    top = max(cyc, key=cyc.get)
    return top, {"cycles": cyc, "iter_cycles": cyc_total, "res": per_call}


def desc_margin(b: M.Benchmark, cfg: M.ArgmindCfg) -> float:
    return M.descriptor_cost(b, cfg)["cycles_per_desc"] / DESC_RULE


def row(b: M.Benchmark, cfg: M.ArgmindCfg = REC) -> dict:
    cm = M.model_argmind(b, cfg)
    cpus = [(c, M.model_cpu(b, c)) for c in M.ALL_CPUS]
    fast = min(cpus, key=lambda t: t[1]["latency_us"])
    effi = min(cpus, key=lambda t: t[1]["energy_uj"])
    wc = dc.replace(b, iters=b.iters_wc)
    top, det = binding_resource(b, cfg)
    return {
        "cm_us": cm["latency_us"],
        "compute_us": cm["compute_us"],
        "overhead_us": cm["overhead_us"],
        "cpu_us": fast[1]["latency_us"],
        "cpu_name": fast[0].name,
        "speedup": fast[1]["latency_us"] / cm["latency_us"],
        "cm_uj": cm["energy_uj"],
        "cpu_uj": effi[1]["energy_uj"],
        "perf_w": effi[1]["energy_uj"] / cm["energy_uj"],
        "wc_margin": (b.deadline_us * BUDGET
                      / M.model_argmind(wc, cfg)["latency_us"]),
        "desc_margin": desc_margin(b, cfg),
        "binding": top,
        "detail": det,
    }


# ---------------------------------------------------------------------------
# (2) 16-bit operands
# ---------------------------------------------------------------------------

def fp16_variants():
    """ADMM rows with fp16 operands and an fp32 triangular solve.

    THE SPLIT IS THE POINT. precision_study C3 measures the cached triangular
    solve as ~3x more error-sensitive than the matrix operands, so the solve
    stays at fp32 and only the bulk GEMVs narrow. `solve_prec_frac` carries that
    split and is derived inside `condensed_mpc`, not hand-entered.
    """
    out = {}
    for name, kw in (
        ("QUAD-MPC", dict(n_x=12, n_u=4, N=20, iters=15, rate=1000,
                          iters_worst=74)),
        ("PWR-CCS", dict(n_x=4, n_u=3, N=10, iters=12, rate=20000,
                         iters_worst=48)),
    ):
        base = M.condensed_mpc(name, note="", **kw)
        narrow = M.condensed_mpc(name + "/fp16", note="", operand_prec="fp16",
                                 **kw)
        # The wrong version, kept so the difference is visible: narrowing the
        # solve TOO. precision_study says this is where the error lives.
        allnarrow = dc.replace(narrow, solve_prec_frac=0.0)
        out[name] = (base, narrow, allnarrow)
    return out


# ---------------------------------------------------------------------------
# (3) MCU-class host tiers
# ---------------------------------------------------------------------------

TIERS = [
    ("T1", "rich host, coherent zero-copy + residency (DRAM present)"),
    ("T0", "rich host, copy-based offload (DRAM, MMU, cache maintenance)"),
    ("T1M", "MCU-class, memory-mapped doorbell + residency (SRAM only)"),
    ("T0M", "MCU-class, RTOS-mediated copy every call (SRAM only)"),
]


def tier_cfg(tier: str) -> M.ArgmindCfg:
    return dc.replace(REC, tier=tier, name=f"CM-100/{tier}")


def main():
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    out("=" * 100)
    out("PWR-MPC RESOLUTION — re-specification, 16-bit operands, MCU-class tier")
    out("=" * 100)
    out("Evidence classes: [MEAS] measured, [CITE] cited literature, [EST]")
    out("estimate, [DER] derived from another model figure. All latency/energy/area")
    out("figures are MODELLED ESTIMATES at a 16nm-class node, +/-2x, NOT QUOTABLE.")
    out()

    # =====================================================================
    out("=" * 100)
    out("(1) IS PWR-MPC THE RIGHT BENCHMARK FOR ITS SOCKET?  Answer: NO — re-specified")
    out("=" * 100)
    out()
    out("The deployed-practice case, all [CITE] (URLs at the end):")
    out()
    out("  a. Direct / finite-control-set MPC is the dominant formulation in power")
    out("     electronics. Karamanakos-Liegmann-Geyer-Kennel (IEEE OJIA 2020):")
    out("     \"direct MPC for power electronics is most often formulated as an\"")
    out("     \"IQP. The common practice for solving such optimization problems in\"")
    out("     \"embedded control systems is with exhaustive search.\"")
    out("  b. The horizon is SHORT because enumeration is exponential: \"Since the")
    out("     problem is NP-hard, such an approach is impractical for horizons")
    out("     longer than one step (Np > 1).\" For a three-phase two-level converter")
    out("     with U = {-1,1}^3 the candidate count is 2^(3*Np) — 8 at Np=1, 64 at")
    out("     Np=2; a three-level converter gives {-1,0,1}^3 = 27 and 729.")
    out("  c. The SAMPLING RATE IS HIGHER THAN THE ROW ASSUMED, not lower. Direct")
    out("     MPC needs fs about two orders of magnitude above the switching")
    out("     frequency; published implementations use Ts = 5, 10, 25, 100 and")
    out("     125 us. The commercial reference point — MPDTC embedded in ABB's")
    out("     medium-voltage drive control software — is Ts = 25 us, i.e. 40 kHz.")
    out("     **The v1.1 row's 20 kHz was BELOW deployed practice.**")
    out("  d. Continuous-control-set MPC does exist here (indirect MPC with a")
    out("     carrier-based modulator, solved with qpOASES/ODYS, and miOSQP for the")
    out("     integer problem) but the published implementations run on dSpace-class")
    out("     rapid-prototyping hardware. It is a real but smaller socket.")
    out("  e. Explicit MPC (offline piecewise-affine + point location + table")
    out("     lookup) is the other deployed answer and is NOT a compute workload at")
    out("     all — it is a memory workload. We should concede it, not model it.")
    out()
    out("RE-SPECIFICATION (BENCHMARKS.md #3, v1.2): three-level NPC, |U| = 27,")
    out("Np = 2 exhaustive (729 sequences), n_x = 4, n_y = 2, Ts = 25 us / 40 kHz,")
    out("fp32. Retired definition retained off-set as PWR-CCS.")
    out()
    out("Np = 2 is chosen deliberately over the Np = 1 that ships today: Np = 1 is")
    out("modelled too (PWR-FCS1) and is a case we LOSE. Np = 2 is the capability the")
    out("literature says needs an FPGA or a branch-and-bound sphere decoder, so it is")
    out("the only version of this socket where an accelerator has an argument at all.")
    out()

    fcs, ccs = by_name("PWR-MPC"), by_name("PWR-CCS")
    out("What the re-spec does to the row, at the recommended CM-100 [DER]:")
    out()
    hdr = (f"{'row':<10} {'algo':<5} {'deadline':>9} {'CM us':>8} {'ofwhich':>8} "
           f"{'CPU us':>8} {'speedup':>8} {'perf/W':>8} {'wc margin':>10} "
           f"{'desc mgn':>9}  binding resource")
    out(hdr)
    out("-" * len(hdr))
    for b in (ccs, fcs, by_name("PWR-FCS1"), by_name("PWR-FAST"),
              by_name("PWR-FUSE")):
        r = row(b)
        out(f"{b.name:<10} {b.algo:<5} {b.deadline_us:>9.1f} {r['cm_us']:>8.2f} "
            f"{r['overhead_us']:>7.2f}o {r['cpu_us']:>8.2f} "
            f"{r['speedup']:>7.2f}x {r['perf_w']:>7.1f}x {r['wc_margin']:>9.2f}x "
            f"{r['desc_margin']:>8.2f}x  {r['binding']}")
    out()
    out("  'ofwhich ...o' = the per-call invocation overhead component of CM latency.")
    out("  'wc margin' = worst-case-iteration latency against 50% of the control")
    out("  period. 'desc mgn' = datapath cycles per descriptor against the >= 8 rule.")
    out()
    out("THE HEADLINE IS NOT FAVOURABLE, and the flattering reading has to be killed")
    out("first. The hoped-for result was that enumeration, being embarrassingly")
    out("parallel and not bandwidth-bound, would move this row above parity. Both")
    out("halves of that are wrong:")
    out()
    r_fcs, r_ccs = row(fcs), row(ccs)
    out(f"  * LATENCY GETS WORSE, not better: {r_ccs['speedup']:.2f}x -> "
        f"{r_fcs['speedup']:.2f}x. The re-specified row is ~"
        f"{1/r_fcs['speedup']:.1f}x SLOWER than the fastest CPU baseline, because")
    out(f"    the whole tick is {r_fcs['compute_us']:.2f} us of compute against a")
    out(f"    {r_fcs['overhead_us']:.2f} us invocation overhead, while the CPU pays no")
    out("    invocation cost at all. **Binding resource: per-call overhead, and")
    out("    below it the vector unit.** No amount of MAC array or bandwidth moves it.")
    out("  * IT IS STILL BANDWIDTH-BOUND, for a new reason. The candidate set IS a")
    out("    batch dimension, so the work is genuinely matrix-MATRIX — but the reuse")
    out("    depth is set by n_y = 2, THE NUMBER OF CONTROLLED OUTPUTS, not by the")
    out(f"    tile. Arithmetic intensity is {fcs.gemm_bytes_per_mac:.2f} B/MAC against")
    out(f"    {REC.gemm_operand_bytes_per_mac('fp32'):.2f} B/MAC for a tile-limited")
    out("    GEMM, so the batch buys a 1.7x traffic cut over pure GEMV, not the ~12x")
    out("    the array size suggests. The GEMM bandwidth ceiling had to be ADDED to")
    out("    the model for this to be expressible: it previously bounded only GEMV.")
    out()
    out("What the re-spec DOES buy, and it is worth more than the ratio:")
    out()
    d_fcs = M.descriptor_cost(fcs, REC)
    d_ccs = M.descriptor_cost(ccs, REC)
    out(f"  * The deadline stops being interesting. {r_fcs['wc_margin']:.1f}x margin "
        f"against 50% of a 25 us period, and the op count is FIXED — no")
    out("    convergence tail, worst case == typical case. Determinism as strong as")
    out("    the EKF's, on the row where determinism is most saleable.")
    out(f"  * The codegen margin problem dissolves: {r_ccs['desc_margin']:.2f}x -> "
        f"{r_fcs['desc_margin']:.2f}x against the >= 8-cycles-per-descriptor rule,")
    out(f"    because {d_ccs['n_desc']:.0f} descriptors of which "
        f"{d_ccs['n_dep']:.0f} chained becomes {d_fcs['n_desc']:.0f} of which "
        f"{d_fcs['n_dep']:.0f}.")
    out("  * There is NO dependency chain. No Cholesky, no triangular solve, no")
    out("    loop-carried scalar. Dependency drain goes "
        f"{d_ccs['dep_drain_cycles']:.0f} -> {d_fcs['dep_drain_cycles']:.0f} cycles.")
    out()
    out("AND THE THREE CONCLUSIONS THAT LOSE THEIR JUSTIFICATION — listed rather than")
    out("silently changed:")
    out()
    quad = by_name("QUAD-MPC")
    combos = {
        "recommended": REC,
        "no run-ahead": dc.replace(REC, seq_setup_cycles=20, seq_queue_depth=0),
        "scalar chain block": dc.replace(REC, chain_block_override=1),
        "both absent": dc.replace(REC, seq_setup_cycles=20, seq_queue_depth=0,
                                  chain_block_override=1),
    }
    out(f"  {'configuration':<20} " + " ".join(f"{n:>11}" for n in
                                               ("QUAD-MPC", "PWR-MPC", "PWR-CCS")))
    for label, c in combos.items():
        cells = []
        for b in (quad, fcs, ccs):
            wc = dc.replace(b, iters=b.iters_wc)
            m = b.deadline_us * BUDGET / M.model_argmind(wc, c)["latency_us"]
            cells.append(f"{m:>10.2f}x" + ("!" if m < 1.0 else " "))
        out(f"  {label:<20} " + " ".join(cells))
    out("  ('!' = misses the 50%-of-control-period budget at worst-case iterations)")
    out()
    nra = M.model_argmind(dc.replace(fcs, iters=fcs.iters_wc),
                          combos["no run-ahead"])["latency_us"]
    out("  (i)  RUN-AHEAD is no longer a deadline requirement. It was recorded as")
    out("       'the only deadline miss the analysis produces' at 0.66x on this row.")
    out(f"       The re-specified row survives no-run-ahead at {nra:.1f} us against a")
    out(f"       {fcs.deadline_us * BUDGET:.1f} us budget "
        f"({fcs.deadline_us * BUDGET / nra:.1f}x).")
    out("  (ii) The >= 8-WIDE CHAIN BLOCKING RTL requirement loses the same row, and")
    out("       the re-specified row is EXACTLY insensitive to it (zero chains).")
    out("       DOWNGRADED STATEMENT that survives: at QUAD-MPC's MEASURED worst-case")
    out("       tail you may skip AT MOST ONE of the two — either alone leaves")
    out("       2.21x / 1.27x, both absent gives 0.37x and misses. Run-ahead moves")
    out("       from 'requirement' to 'required unless chain blocking is full-width'.")
    out("  (iii) The 64 B/cycle BANDWIDTH FLOOR loses one of its two rows. It was set")
    out("       by 'QUAD-MPC cold 560 us > 500 at 32 B/cyc AND PWR-MPC 33 > 25'. The")
    out("       re-specified row no longer constrains it:")
    out()
    out(f"  {'B/cyc':>6} " + " ".join(f"{n:>12}" for n in
                                      ("QUAD-MPC", "PWR-MPC", "PWR-CCS")))
    for bw in (16, 32, 64, 128, 256):
        c = Z.cfg(f"bw{bw}", bw, tile=12, lanes=16)
        cells = []
        for b in (quad, fcs, ccs):
            wc = dc.replace(b, iters=b.iters_wc)
            lat = M.model_argmind(wc, c)["latency_us"]
            m = b.deadline_us * BUDGET / lat
            cells.append(f"{lat:>10.1f}us" + ("!" if m < 1.0 else " "))
        out(f"  {bw:>6} " + " ".join(cells))
    out()
    out("       **The 64 B/cycle floor now rests on QUAD-MPC alone** — but QUAD-MPC's")
    out("       tail is the MEASURED one (74 iterations, results_osqp_measured.txt),")
    out("       so the floor rests on the better of the two pieces of evidence. It")
    out("       does not move. What changes is that it is now a single-row")
    out("       justification, which is the exact structural weakness that caused two")
    out("       previous re-runs. Say so.")
    out()
    out("  (iv) NOT LOST: 'a CPU already meets this deadline' gets STRONGER, and")
    out("       'the only solver row below parity' survives with a different cause.")
    out()

    # =====================================================================
    out("=" * 100)
    out("(1b) THE CASE WE LOSE, reported because a model that cannot lose is not evidence")
    out("=" * 100)
    out()
    f1 = row(by_name("PWR-FCS1"))
    out(f"One-step FCS-MPC (27 candidates, 40 kHz) — what an MCU ships TODAY:")
    out(f"  CM {f1['cm_us']:.2f} us ({f1['compute_us']:.2f} compute + "
        f"{f1['overhead_us']:.2f} overhead) vs CPU {f1['cpu_us']:.2f} us "
        f"= {f1['speedup']:.2f}x")
    out(f"  perf/W {f1['perf_w']:.1f}x — above the 5x kill criterion but the lowest")
    out(f"  in the whole solver set, because the workload is smaller than the fixed")
    out(f"  overheads it has to amortise.")
    out(f"  Descriptor margin {f1['desc_margin']:.2f}x — **BELOW the >= 8 codegen rule**.")
    out(f"  The two-level 100 kHz variant is {row(by_name('PWR-FAST'))['desc_margin']:.2f}x, also below.")
    out()
    out("  So the thinnest-codegen-margin problem does not vanish, it MOVES: the")
    out("  frozen row improves to 6.7x, and two adjacent real configurations of the")
    out("  same socket sit under the bar. The rule is not violated by the benchmark")
    out("  we chose; it is violated by the smaller ones next to it. Honest reading:")
    out("  **for one-step FCS-MPC, tell the customer to use the MCU.** The minimum")
    out("  viable version of this socket is a multi-step horizon.")
    out()

    # =====================================================================
    out("=" * 100)
    out("(2) WOULD 16-BIT OPERANDS FIX THE LATENCY?  Partly — and it changes the format")
    out("=" * 100)
    out()
    out("Numerical basis, all [MEAS] (perf_model/results_precision_study.txt, C3/C4;")
    out("measured numerics, NOT silicon). Relative error in the commanded control")
    out("action against the fp64 solution of the SAME problem:")
    out()
    out("    fp32 operands, fp32 solve  (CM-100 today)   8.2e-07")
    out("    fp16 operands, fp32 solve  (the proposal)   1.9e-03")
    out("    bf16 operands, fp32 solve  (bf16 array)     1.9e-02")
    out("    fp64 operands, bf16 SOLVE                   6.1e-02")
    out()
    out("  Three things follow, and the second is the load-bearing new one:")
    out("  * fp16 beats bf16 by 10x at the same width. Mantissa bits dominate for")
    out("    these matrices — the OPPOSITE of the ML tradeoff that makes bf16 the")
    out("    default. **The array's bf16 specification is wrong for the solver path.**")
    out("  * KEEPING THE SOLVE AT fp32 COSTS NOTHING: fp16-operands/fp32-solve and")
    out("    fp16-operands/fp64-solve score IDENTICALLY (1.946e-03 both). The earlier")
    out("    study only ever paired 16-bit operands with an fp64 solve, which no CM")
    out("    SKU offers on the ADMM path, so the proposal was untested until now.")
    out("    The operand precision, not the solve, is what limits the mixed scheme.")
    out("  * FCS-MPC is far more tolerant (C4, 729 sequences, 200 trials): fp16")
    out("    changes the selected switching sequence on 3.0% of ticks at a mean")
    out("    excess cost of 1.5e-06 of the candidate cost spread; bf16 changes 11.0%")
    out("    at 9.2e-05. Because the output is a discrete argmin, a disagreement")
    out("    between two near-equal candidates is not an error. **Different algorithm,")
    out("    different precision floor — which is a MACIF fact, not a nicety.**")
    out()
    out("Modelled effect [DER]. fp16 operands, fp32 accumulate, triangular solve at")
    out("fp32 (its share of GEMV work is carried as `solve_prec_frac`):")
    out()
    hdr2 = (f"{'row':<24} {'CM us':>9} {'speedup':>8} {'CM uJ':>8} {'perf/W':>8} "
            f"{'wc margin':>10}  binding")
    out(hdr2)
    out("-" * len(hdr2))
    for name, (base, narrow, allnarrow) in fp16_variants().items():
        for tag, b in (("fp32 (today)", base), ("fp16 operands", narrow),
                       ("fp16 incl. SOLVE (bad)", allnarrow)):
            r = row(b)
            label = f"{name.split('-')[0][:4]}/{tag}"
            out(f"{label:<24} {r['cm_us']:>9.2f} "
                f"{r['speedup']:>7.2f}x {r['cm_uj']:>8.3f} {r['perf_w']:>7.1f}x "
                f"{r['wc_margin']:>9.2f}x  {r['binding']}")
    out()
    out("  The third variant is listed so the trade is visible, NOT recommended:")
    out("  narrowing the triangular solve too costs 6.1e-2 control-action error "
        "(C3),")
    out("  ~30x worse than fp16 operands alone, for a further 1.06-1.09x of latency.")
    out()
    v = fp16_variants()
    for name in v:
        base, narrow, _ = v[name]
        rb, rn = row(base), row(narrow)
        out(f"  {name}: latency {rb['cm_us']:.2f} -> {rn['cm_us']:.2f} us "
            f"({rb['cm_us']/rn['cm_us']:.2f}x), energy {rb['cm_uj']:.3f} -> "
            f"{rn['cm_uj']:.3f} uJ ({rb['cm_uj']/rn['cm_uj']:.2f}x), "
            f"speedup {rb['speedup']:.2f}x -> {rn['speedup']:.2f}x")
    out()
    out("  WHY IT IS ONLY ~1.3x AND NOT 2x, which is the adversarial point: halving")
    out("  operand width doubles the GEMV rate, but the GEMV term is not the whole")
    out("  iteration. Once it halves, the VECTOR UNIT becomes co-binding — ADMM's")
    out("  projection, dual update and over-relaxation are a fixed element-op count")
    out("  that 16 fp32 lanes cannot narrow. The dependency drain does not narrow")
    out("  either. **A bandwidth-bound row stops being bandwidth-bound before the")
    out("  bandwidth doubling is spent.** Same shape as the 128 B/cycle finding.")
    out()
    out("  Energy is the better half of the answer: operand bytes halve AND the")
    out("  multiplier drops 0.60 -> 0.20 pJ [EST], and neither is capped by the")
    out("  vector unit.")
    out()
    out("  DOES IT MOVE THE ROW ABOVE PARITY? Reported precisely, because this is")
    out("  where a flattering answer would hide:")
    rb, rn, _ = v["PWR-CCS"][0], v["PWR-CCS"][1], None
    out(f"    PWR-CCS: {row(rb)['speedup']:.2f}x -> {row(rn)['speedup']:.2f}x")
    out(f"    QUAD-MPC: {row(v['QUAD-MPC'][0])['speedup']:.2f}x -> "
        f"{row(v['QUAD-MPC'][1])['speedup']:.2f}x")
    out("    But note what the CPU-side correction below did to the SAME question,")
    out("    and read the two together before quoting either.")
    out()
    out("  A CPU-BASELINE CORRECTION LANDED IN THE MIDDLE OF THIS AND MOVES THE RATIO")
    out("  IN OUR FAVOUR, so it is flagged, not banked. `model_cpu` charged")
    out("  ELEMENTWISE work at the outer-product rate, i.e. it ran min/max and")
    out("  horizontal reductions on SME2's ZA tile, which cannot execute them. The")
    out("  measured M4 gap is 2008 GFLOP/s FMOPA against ~250 GFLOP/s for streaming")
    out("  SVE — 8x. The model now carries `simd_flops_per_cycle = 32` for the SME2")
    out("  baseline, HALF the matrix rate rather than the measured eighth, i.e. the")
    out("  value least favourable to us. Effect on the retired ADMM row:")
    out()
    for simd, tag in ((64.0, "uncorrected (elementwise on the ZA tile)"),
                      (32.0, "corrected, generous  <-- adopted"),
                      (8.0, "corrected, measured M4 ratio")):
        cpus = [dc.replace(c, simd_flops_per_cycle=simd)
                if c is M.CPU_SME2 else c for c in M.ALL_CPUS]
        lat = min(M.model_cpu(ccs, c)["latency_us"] for c in cpus)
        cm = M.model_argmind(ccs, REC)["latency_us"]
        out(f"    simd={simd:>5.0f} FLOP/cyc  PWR-CCS speedup {lat/cm:>5.2f}x   "
            f"({tag})")
    out()
    out("    **So the famous 0.974x was partly an artefact of a CPU-side modelling")
    out("    error.** Corrected at the generous setting the retired row is at or just")
    out("    above parity. That is not a win — it is a row sitting exactly on a")
    out("    modelling assumption, and the honest instruction is that NEITHER 0.97x")
    out("    NOR 1.08x is quotable. What is quotable in shape: on this workload class")
    out("    we are at parity with a modern vector CPU and win on energy.")
    out()
    out("  VERDICT ON (2): adopt fp16 as the 16-bit solver operand format and make it")
    out("  a declared MACIF format. It is worth **1.16-1.49x latency and 1.51-1.79x")
    out("  energy** on the ADMM rows (PWR-CCS / QUAD-MPC respectively) at ~2400x the")
    out("  operand error — 8e-7 -> 1.9e-3 relative in the commanded control action.")
    out("  0.2% error in a commanded current is plausibly fine for a drive and is NOT")
    out("  obviously fine for a system whose constraint is a hard current limit, so")
    out("  this is a per-problem codegen OPTION a customer opts into, not a default.")
    out("  It does not rescue any row's parity position on its own: the re-specified")
    out("  PWR-MPC row is not bandwidth-bound in the first place, so fp16 does")
    out("  essentially nothing for it, and PWR-CCS moves only 1.08x -> 1.26x.")
    out("  The energy number is the one worth having: QUAD-MPC perf/W 14.1x -> 25.3x.")
    out()

    # =====================================================================
    out("=" * 100)
    out("(3) THE MCU-CLASS CONTRADICTION — settled, and CLAUDE.md's framing is wrong")
    out("=" * 100)
    out()
    out("The contradiction: PHASE0_FINDINGS §4 credits zero-copy with 1.9x on this")
    out("row and with deciding its deadline feasibility; §5.1 records that in MCU")
    out("class there is no DRAM and no cache, so zero-copy and residency have")
    out("nothing to bite on. Both cannot be true.")
    out()
    out("The resolution is that the 25 us T0 constant was never an MCU-class number.")
    out("Decomposed [EST], each term naming a mechanism that can be measured alone:")
    out()
    out(f"  {'term':<28} {'T0':>8} {'T1':>8} {'T0M':>8} {'T1M':>8}")
    for term, vals in (("driver / RTOS API", (8.0, None, 1.5, None)),
                       ("buffer map + pin", (6.0, None, None, None)),
                       ("cache clean + invalidate", (6.0, None, None, None)),
                       ("IRQ + scheduler wakeup", (5.0, None, 0.5, None)),
                       ("descriptor write", (None, 0.1, None, 0.10)),
                       ("doorbell store", (None, 0.1, None, 0.05)),
                       ("completion poll / latency", (None, 0.6, None, 0.15))):
        cells = " ".join(f"{v:>8.2f}" if v is not None else f"{'--':>8}"
                         for v in vals)
        out(f"  {term:<28} {cells}")
    out(f"  {'TOTAL us':<28} " + " ".join(
        f"{M.ArgmindCfg.CALL_OVERHEAD_US[t]:>8.2f}" for t in
        ("T0", "T1", "T0M", "T1M")))
    out()
    out("  TWO of T0's four terms — buffer map/pin and cache maintenance — are")
    out("  artefacts of an MMU and a cache. An all-SRAM MCU has neither. What is left")
    out("  is an RTOS API call and an interrupt, and T1M is CHEAPER than T1 because")
    out("  there is no coherence round trip either.")
    out()
    out("  So the honest ablation on an MCU-class host [DER]:")
    out()
    hdr3 = (f"{'row':<10} " + " ".join(f"{t:>9}" for t, _ in TIERS)
            + f" {'T1/T0':>7} {'T1M/T0M':>8}  T0M meets?")
    out(hdr3)
    out("-" * len(hdr3))
    for b in (fcs, ccs, by_name("QUAD-MPC"), by_name("EKF-24")):
        lats = {t: M.model_argmind(b, tier_cfg(t))["latency_us"]
                for t, _ in TIERS}
        ok = "MEETS" if lats["T0M"] <= b.deadline_us else "MISSES"
        out(f"{b.name:<10} " + " ".join(f"{lats[t]:>9.2f}" for t, _ in TIERS)
            + f" {lats['T0']/lats['T1']:>6.2f}x {lats['T0M']/lats['T1M']:>7.2f}x"
              f"  {ok}")
    out()
    out("  And the energy side, where the answer is starker [DER]:")
    out()
    out(f"  {'row':<10} " + " ".join(f"{t:>9}" for t, _ in TIERS)
        + f" {'T1/T0 uJ':>9} {'T1M/T0M':>8}")
    for b in (fcs, ccs, by_name("QUAD-MPC"), by_name("EKF-24")):
        es = {t: M.model_argmind(b, tier_cfg(t))["energy_uj"] for t, _ in TIERS}
        out(f"  {b.name:<10} " + " ".join(f"{es[t]:>9.3f}" for t, _ in TIERS)
            + f" {es['T0']/es['T1']:>8.2f}x {es['T0M']/es['T1M']:>7.2f}x")
    out()
    out("ANSWERS, in the order asked:")
    out()
    fl = {t: M.model_argmind(fcs, tier_cfg(t))["latency_us"] for t, _ in TIERS}
    fe = {t: M.model_argmind(fcs, tier_cfg(t))["energy_uj"] for t, _ in TIERS}
    out("  Q: For an MCU-class host with on-chip SRAM only, what does the T1")
    out("     zero-copy tier actually buy?")
    out(f"  A: On the re-specified row, {fl['T0M']/fl['T1M']:.2f}x latency and "
        f"{fe['T0M']/fe['T1M']:.2f}x energy — against {fl['T0']/fl['T1']:.2f}x and "
        f"{fe['T0']/fe['T1']:.2f}x")
    out("     on a rich host. **Nearly all of what remains is invocation overhead,")
    out("     not data movement.** The data-movement half really does collapse, and")
    out("     for two independent reasons: the bytes are cheap (an on-chip copy at")
    out(f"     ~1.2 pJ/B [EST] instead of ~12 pJ/B of LPDDR) and there are barely any")
    out(f"     ({fcs.persistent_bytes:.0f} B of persistent state for FCS-MPC, versus")
    out(f"     {ccs.persistent_bytes:.0f} B for the retired ADMM row and "
        f"{by_name('QUAD-MPC').persistent_bytes:.0f} B for QUAD-MPC).")
    out("     CLAUDE.md's 'zero-copy and residency are VOID in MCU class' is right")
    out("     about residency and about coherency, and WRONG to bundle invocation")
    out("     overhead in with them.")
    out()
    out("     Decomposing the T1M-over-T0M advantage on the re-specified row, which")
    out("     is the crux of the whole question [DER]:")
    ovh_d = (M.ArgmindCfg.CALL_OVERHEAD_US["T0M"]
             - M.ArgmindCfg.CALL_OVERHEAD_US["T1M"])
    tot_d = fl["T0M"] - fl["T1M"]
    out(f"       total T1M advantage                {tot_d:>6.2f} us")
    out(f"       of which invocation overhead       {ovh_d:>6.2f} us  "
        f"({100 * ovh_d / tot_d:.0f}%)")
    out(f"       of which avoided on-chip copy      {tot_d - ovh_d:>6.2f} us  "
        f"({100 * (tot_d - ovh_d) / tot_d:.0f}%)")
    out("     So on MCU class the advantage is ~4/5 invocation protocol and ~1/5 data")
    out("     movement, and the data-movement fifth exists only because an MCU bus is")
    out("     narrow, not because the bytes are far away.")
    out()
    out("  Q: Is the per-call overhead difference really descriptor/doorbell/syscall")
    out("     cost rather than data movement?")
    out("  A: YES, structurally — `call_overhead_us` is a fixed launch cost that the")
    out("     model charges independently of `bytes_moved`, and the decomposition")
    out("     above shows which terms survive the removal of DRAM. But the honest")
    out("     magnitude is much smaller than the headline: the advantage falls from")
    out("     31x (25/0.8) to 6.7x (2.0/0.30), and in ABSOLUTE terms from 24.2 us to")
    out("     1.7 us. Against a 25 us period that is 6.8% — real, not decisive.")
    out("     Against the 5 us period of the fastest deployed direct MPC (Ts = 5 us)")
    out("     it is 34%, and there it IS decisive. **The invocation-overhead pitch")
    out("     for MCU class is rate-dependent, and it needs 100 kHz+ to carry a")
    out("     deadline argument.**")
    out()
    out("  Q: So what buys PWR-MPC its deadline on an MCU-class host, and does the")
    out("     row still meet it?")
    out(f"  A: It meets it in EVERY tier, including T0M ({fl['T0M']:.2f} us against")
    out(f"     25 us). Nothing 'buys' the deadline — the re-specified workload is")
    out("     ~640 cycles of arithmetic and the deadline was never the issue. What")
    out("     the CM buys on an MCU-class host is (a) energy per solve, (b) a bounded")
    out("     worst case with no iteration tail, and (c) offloading the control core.")
    out("     **The zero-copy argument should be withdrawn from the MCU-class pitch")
    out("     entirely** and replaced with low invocation overhead + determinism +")
    out("     energy. Zero-copy stays a real differentiator on rich hosts (QUAD-MPC,")
    out("     ARM-MPC, the ML rows), where DRAM and cache maintenance exist.")
    out()
    out("  ADVERSARIAL NOTE, because this cut the other way and should not be buried:")
    out("  an MCU-class integration is WORSE for us on one axis the previous analysis")
    out("  never charged. If the CM addresses host SRAM directly (the D2/D3 tightly-")
    out("  coupled story) it inherits the host's bus width — 8 B/cycle at 200 MHz is")
    out("  1.6 GB/s [EST], 4x LESS than the LPDDR path it replaces. The data is")
    out("  closer and the pipe is narrower. That is fine for a 716-byte FCS-MPC")
    out("  working set and it is NOT fine for anything that streams. Any MCU-class")
    out("  SKU must keep its own scratchpad, which means the copy does not disappear")
    out("  — it just becomes cheap.")
    out()

    # =====================================================================
    out("=" * 100)
    out("MICROARCHITECTURE CONSEQUENCES — what actually changes in the design")
    out("=" * 100)
    out()
    cyc = binding_resource(fcs, REC)[1]["cycles"]
    dp = sum(v for k, v in cyc.items() if k != "call_overhead")
    cpu_best = min(M.model_cpu(fcs, c)["latency_us"] for c in M.ALL_CPUS)
    cm_r = M.model_argmind(fcs, REC)
    cm_t1m = M.model_argmind(fcs, dc.replace(REC, tier="T1M"))["latency_us"]
    out("1. INVOCATION OVERHEAD IS THE BINDING RESOURCE, and inside the datapath it")
    out("   is the VECTOR UNIT. Per-tick cycles at CM-100 [DER]:")
    for k in ("gemm_array", "gemm_bw", "gemv_bw", "vector", "seq_overhead",
              "call_overhead"):
        if cyc.get(k, 0) > 0:
            out(f"     {k:<14} {cyc[k]:>8.1f} cycles")
    out(f"   Cost evaluation and argmin are {cyc['vector']:.0f} of {dp:.0f} datapath")
    out(f"   cycles (81%); the invocation overhead is another {cyc['call_overhead']:.0f}"
        f" cycles on top,")
    out("   i.e. HALF the end-to-end latency.")
    out()
    out("   AN EARLIER DRAFT OF THIS SECTION HAD A UNIT ERROR AND THE CORRECTION")
    out("   CHANGES THE RECOMMENDATION, so it is recorded. It claimed the CM was 5x")
    out("   under-provisioned on elementwise throughput versus SME2, by comparing 16")
    out("   lanes/cycle against an SME2 figure counted in FLOPs. In the same units:")
    out()
    out(f"     CM-100, {REC.vector_lanes} lanes @ {REC.freq_mhz:.0f} MHz"
        f"        {REC.vector_lanes * REC.freq_mhz / 1000:>6.2f} G element-ops/s")
    for c in M.ALL_CPUS:
        r = c.simd_rate * c.small_gemm_efficiency / 2.0
        out(f"     {c.name.split('-class')[0]:<28} "
            f"{r * c.freq_mhz / 1000:>6.2f} G element-ops/s "
            f"({r:.1f}/cyc @ {c.freq_mhz:.0f} MHz)")
    out()
    out("   **We are at PARITY with the SME2 baseline on elementwise throughput, not")
    out("   5x behind.** The correct decomposition of why the row loses:")
    out()
    out(f"     CM datapath only          {cm_r['compute_us']:.3f} us   -> "
        f"{cpu_best / cm_r['compute_us']:.2f}x vs the fastest CPU")
    out(f"     CM + T1M overhead (0.30)  {cm_t1m:.3f} us   -> "
        f"{cpu_best / cm_t1m:.2f}x")
    out(f"     CM + T1  overhead (0.80)  {cm_r['latency_us']:.3f} us   -> "
        f"{cpu_best / cm_r['latency_us']:.2f}x")
    out()
    out("   The CM COMPUTES this tick 1.24x faster than an SME2 core and then pays")
    out("   0.8 us to be asked. **For a sub-microsecond workload, no invocation")
    out("   protocol that involves a descriptor and a completion notification puts an")
    out("   accelerator ahead of the host core on latency** — even the MCU-class")
    out("   0.30 us tier only reaches 0.90x. That is an integration-depth statement,")
    out("   not a datapath one: this socket wants D3 (in-pipeline coupling) or it")
    out("   wants to be sold on energy and offload alone. Recommend the latter, and")
    out("   note it agrees with the settled finding that the advantage is energy.")
    out()
    out("   Two datapath responses remain worth having for their energy effect and")
    out("   for the descriptor margin; the second is far cheaper than the first:")
    out()
    out(f"   {'lanes':>6} {'fp64 FMAs':>10} {'PWR-MPC us':>11} {'speedup':>8} "
        f"{'core mm2':>9} {'d mm2':>7}")
    base_area = Z.area_breakdown(REC)["total"]
    for lanes in (8, 16, 32, 64):
        c = dc.replace(REC, vector_lanes=lanes, fp64_fmas=lanes // 2,
                       name=f"lanes{lanes}")
        r = row(fcs, c)
        a = Z.area_breakdown(c)["total"]
        out(f"   {lanes:>6} {lanes//2:>10} {r['cm_us']:>11.2f} "
            f"{r['speedup']:>7.2f}x {a:>9.3f} {a-base_area:>+7.3f}")
    out()
    fuse = by_name("PWR-FUSE")
    fu_cm = M.model_argmind(fuse, REC)
    fu_e = fu_cm["energy_uj"]
    out("   (a) More lanes. 16 -> 32 costs +0.143 mm2 (+20% of the core) and buys")
    out("       1.26x on this row, because the merged-FMA construction doubles the")
    out("       fp64 cluster with it and because half the latency is not datapath.")
    out("       REJECTED on that ratio.")
    out("   (b) TWO ACCUMULATOR FEATURES that fold the two largest vector terms into")
    out("       the array at near-zero area: BROADCAST/BIAS INITIALISATION of the")
    out("       accumulator (folds the B@U broadcast-add) and SQUARE-ACCUMULATE")
    out("       (folds the ||r||^2 cost reduction). Modelled as PWR-FUSE:")
    out(f"         vector cycles   {cyc['vector']:.0f} -> "
        f"{fuse.vector_ops_per_iter / REC.vector_lanes:.0f}")
    out(f"         CM latency      {cm_r['latency_us']:.2f} -> "
        f"{fu_cm['latency_us']:.2f} us  "
        f"(datapath {cm_r['compute_us']:.2f} -> {fu_cm['compute_us']:.2f})")
    out(f"         vs fastest CPU  {cpu_best / cm_r['latency_us']:.2f}x -> "
        f"{cpu_best / fu_cm['latency_us']:.2f}x")
    out(f"         CM energy       {cm_r['energy_uj']:.4f} -> {fu_e:.4f} uJ "
        f"({cm_r['energy_uj'] / fu_e:.2f}x)")
    out(f"         descriptors     {d_fcs['n_desc']:.0f} -> "
        f"{M.descriptor_cost(fuse, REC)['n_desc']:.0f}, descriptor margin "
        f"{r_fcs['desc_margin']:.2f}x -> {row(fuse)['desc_margin']:.2f}x")
    out("       NOTE ON HOW THIS IS SCORED, because the obvious way is wrong: the")
    out("       fusion is a CM DATAPATH capability, not a change to the algorithm, so")
    out("       the CPU still executes those vector ops. The comparison above is")
    out("       therefore CM(fused) against CPU(unfused). Scoring PWR-FUSE against a")
    out("       CPU running PWR-FUSE — which the table further up does — credits the")
    out("       CPU with an accumulator it does not have and makes the feature look")
    out("       like a regression. Read the table row for the CM column only.")
    out("       **RECOMMENDATION: (b), and it is a MACIF-boundary question** — a")
    out("       broadcast-initialised, square-accumulating accumulator is a property")
    out("       of the swappable macro, so it belongs in MACIF v0 alongside")
    out("       `accumulator_width[format]`. It is also the same L0-staging argument")
    out("       already flagged as the highest-value model extension: fold work into")
    out("       the accumulator at O(1) area instead of buying O(T^2) of array.")
    out()
    out("2. THE ARRAY BANDWIDTH CEILING IS NOW MODELLED, and it binds on exactly one")
    out("   row. Before this change the model applied a bandwidth bound only to")
    out("   GEMV, so it was structurally incapable of reporting a bandwidth-starved")
    out("   GEMM. A test asserts the new term binds on PWR-MPC and on nothing else,")
    out("   so it is neither decoration nor a silent perturbation of the other rows.")
    out()
    out("3. fp16, NOT bf16, IS THE 16-BIT SOLVER FORMAT. bf16 remains correct for")
    out("   ML. This makes the array multi-format on the 16-bit path and it is a")
    out("   declared MACIF capability, not an implementation choice — a macro that")
    out("   silently substituted bf16 for fp16 would cost 10x the control-action")
    out("   error and the runtime must refuse it, exactly as it refuses a missing")
    out("   fp64 path.")
    out()
    out("4. BANDWIDTH-FIRST SURVIVES, with a third mechanism. It was argued from")
    out("   batch-1 GEMV (no reuse) and from bank-geometry energy. FCS-MPC adds:")
    out("   even a workload with a real batch dimension can be operand-fetch-bound")
    out("   if its REUSE DEPTH is set by a problem dimension (n_y = 2) rather than")
    out("   by the tile. That is a stronger and more general statement than 'batch-1")
    out("   solvers have no reuse'.")
    out()

    # =====================================================================
    out("=" * 100)
    out("WHAT WOULD FIRM THIS UP, in order of value per day")
    out("=" * 100)
    out("  1. Confirm the FCS-MPC parameter set against a real deployed controller")
    out("     (n_x, n_y, Np, Ts, |U| and whether a terminal cost or move-blocking is")
    out("     used). Everything in (1) hangs on the shape being right, and the shape")
    out("     is CITED literature, not a measured deployment.")
    out("  2. Measure a C2000 CLA or AURIX PPU task dispatch latency. The whole")
    out("     MCU-class answer rests on T0M = 2.0 us and T1M = 0.30 us, both [EST],")
    out("     and it is a half-day measurement on a launchpad board.")
    out("  3. Measure FCS-MPC on an F28379D and on an A78/SME2 board. The CPU side")
    out("     of this row is now the dominant uncertainty, and the elementwise-rate")
    out("     correction above shows the CPU model can move a verdict by 10%.")
    out("  4. Re-run C4 across plant families and switching-set sizes. One plant")
    out("     family with a fixed seed established the mechanism, not a distribution.")
    out()
    out("=" * 100)
    out("SOURCES for the deployed-practice claims in (1)")
    out("=" * 100)
    out("  * P. Karamanakos, E. Liegmann, T. Geyer, R. Kennel, \"Model Predictive")
    out("    Control of Power Electronic Systems: Methods, Results, and Challenges,\"")
    out("    IEEE Open Journal of Industry Applications, vol. 1, 2020.")
    out("    http://www.tobiasgeyer.org/KLGK20_MPCsurvey.pdf")
    out("    Quoted above: exhaustive search is common practice; impractical for")
    out("    Np > 1; 2^(3Np) candidates for a two-level converter; fs ~ 100x fsw;")
    out("    Ts = 5/10/25/100/125 us across published implementations; FPGA is the")
    out("    preferred platform; solver table (qpOASES AS, ODYS IP, sphere decoder")
    out("    BnB, miOSQP ADMM).")
    out("  * T. Geyer, G. Papafotiou, \"Model Predictive Direct Torque Control of a")
    out("    Variable Speed Drive with a Five-Level Inverter.\"")
    out("    https://tobiasgeyer.org/GePa09_MPDTC_ACS5000.pdf")
    out("    Ts = 25 us \"typical for DTC\"; tree of candidate switching sequences;")
    out("    three-level NPC in ABB ACS6000 and five-level in ACS5000.")
    out("  * MPDTC embedded in the ABB ACS6000 medium-voltage drive control software:")
    out("    T. Geyer, G. Papafotiou, M. Morari, \"Model Predictive Direct Torque")
    out("    Control — Part II: Implementation and Experimental Evaluation,\"")
    out("    IEEE Trans. Ind. Electron. 2009. Geyer is R&D platform manager for the")
    out("    ACS6000/6080 at ABB (author biography in the OJIA survey above).")
    out("  * FCS-MPC on a TI TMS320F28379D (peer-reviewed third party):")
    out("    https://www.mdpi.com/2227-7390/11/21/4553")
    out("  * Explicit MPC as a lookup of a precomputed piecewise-affine law:")
    out("    https://www.mathworks.com/help/mpc/ref/explicitmpccontroller.html")
    out()
    out("  CAVEAT ON THE CITATIONS: they establish the FORMULATION and the SAMPLING")
    out("  RATES that are published. They do NOT establish n_x, n_y or the exact")
    out("  candidate-set size of any specific shipping product, and no vendor")
    out("  publishes a per-tick op count. Those remain [EST] and item 1 above is the")
    out("  measurement that would close them.")

    with open("perf_model/results_pwr_mpc_respec.txt", "w",
              encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
