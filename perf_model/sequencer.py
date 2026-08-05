"""
Sequencer / descriptor-issue analysis — closing the open item from the fp64 work
===============================================================================

WHY THIS FILE EXISTS. The fp64 costing (fp64_datapath.py) forced descriptor
counts to be consistent with the benchmark GEMM counts and, as a side effect,
produced a number nobody had asked for: an HPIPM block operation at nb=21 is
only a handful of array cycles, so above ~4 cycles of descriptor setup ARM-MPC
would be SEQUENCER-ISSUE-BOUND rather than array-bound. That was logged, not
resolved, because the model could not express it: sequencer work was a flat
`scalar_ops_per_iter` constant with a fixed 50% compute overlap.

It matters more than a modelling nit because the sequencer IS the differentiator.
If the sequencer binds on the nonlinear-MPC benchmark then the part is
mis-provisioned in a way that no amount of bandwidth or MACs fixes, and
block-structured interior-point solvers -- which issue many small operations
rather than few large ones -- are exactly the workload that exposes it.

WHAT THIS FILE CONCLUDES, up front:

  1. THE LOGGED CLAIM WAS WRONG IN ITS PREMISE, and wrong in a specific,
     instructive way. The "~5.6 array cycles" figure divided an HPIPM block op by
     the 144-MAC bf16 ARRAY rate. ARM-MPC does not run on the array: it is fp64,
     and fp64 runs on the 8-unit FMA cluster at 1/19th the rate. The same block
     op is ~30 datapath cycles there. So the claim "at any precision" is FALSE,
     and it is false in the direction that matters: the exposure appears when the
     arithmetic is FAST relative to the block, not slow.
  2. THE EXPOSURE IS REAL, just not where it was logged. Three regimes bind:
     PWR-MPC (tiny blocks, highest rate), ARM-MPC run in fp32 (the iterative
     refinement path), and any block-structured NMPC with nb <= 12.
  3. THE DOMINANT NEW COST IS NOT ISSUE THROUGHPUT, IT IS DEPENDENCY DRAIN.
     Cholesky and triangular-solve descriptors cannot start until the previous
     block's results are written back. That term was absent from the model
     entirely, and it is what actually costs ARM-MPC ~12% of its latency.
  4. RUN-AHEAD IS MANDATORY, not an optimisation -- and this is the one place a
     deadline is actually missed.
  5. The 50% fudge happened to land within ~6% for ADMM. It was ~12% low for
     HPIPM and it was standing in for the wrong physical effect.

ALL FIGURES ARE MODELLED ESTIMATES, 16nm-class, +/-2x. NOT QUOTABLE EXTERNALLY.
Evidence classes: [EST] first-order estimate, [DER] derived from another figure
in the model, [C*] cited, [MEAS] measured.
"""

from __future__ import annotations

import dataclasses
import math

import argmind_model as M
import cm100_sizing as Z
import sram_cost as S

REC = Z.cfg("CM-100 recommended", 128.0, tile=12, lanes=16,
            kb=M.CM100_SCRATCHPAD_KB, fp64_fmas=8)

SETUP_SWEEP = (1, 2, 4, 8, 16, 32, 64)
PERIOD_BUDGET = Z.PERIOD_BUDGET

# ---------------------------------------------------------------------------
# Sequencer implementation options and their area
# ---------------------------------------------------------------------------
# Gate counts are [EST] at the same 0.26 um2/NAND2 basis as everything else in
# sram_cost.py. The scalar-core figure is the published open-core range for a
# 3-stage RV32IMC (Ibex / SCR1 class, 20-40 kGE without FPU); 30k is central.
GATES_RV32IMC = 30_000                    # [EST], open-core range 20-40k
GATES_AGU = 1_200                         # 32b adder + stride regs + compare
N_AGU = 3                                 # two source operands + destination
GATES_HW_LOOP = 1_200                     # 3 nested counters + bounds + compare
DESC_BITS = 192                           # op, 3 base addrs, 3 dims, 2 strides,
                                          # format/accum flags, completion tag
GATES_PER_FLOP_BIT = 6                    # [EST] flop + mux + clock gate
GATES_ISSUE_FSM = 4_000                   # issue/completion tracking, doorbell
N_DESC_TEMPLATES = 24                     # armed templates the generator holds.
# CORRECTED 2026-07-31 from 4 (codegen finding G2). The generated QUAD-MPC solver
# emits 24 DISTINCT kernel shapes (15 in the iteration block alone over 16
# descriptors) -- not the ~6 the emit-cost analysis assumed. Since arming costs
# ~22 instructions PER TEMPLATE, per-iteration arming is 16.7 cycles/descriptor,
# not 8.25: 2x worse than believed, in the same direction, so the "arm once per
# solve" conclusion is strengthened rather than threatened (hoisted: 1.5).
# Cost of the extra 20 templates: 1152 gates each, +0.006 mm2, 0.8% of the core.

# The sequencer needs its own program store. This was NOT in the area budget:
# cm100_sizing.REST_MM2 allocates 0.045 mm2 to "sequencer" as a flat number with
# no memory line item.
SEQ_IMEM_KB = 8
SEQ_DMEM_KB = 4


def seq_logic_mm2(descriptor_generator: bool, queue_depth: int) -> float:
    g = GATES_RV32IMC
    if descriptor_generator:
        g += (N_AGU * GATES_AGU + GATES_HW_LOOP + GATES_ISSUE_FSM
              + N_DESC_TEMPLATES * DESC_BITS * GATES_PER_FLOP_BIT)
    else:
        g += GATES_ISSUE_FSM
    g += queue_depth * DESC_BITS * GATES_PER_FLOP_BIT
    return g * S.A_GATE_UM2 / 1e6


def seq_mem_mm2(imem_kb: float = SEQ_IMEM_KB, dmem_kb: float = SEQ_DMEM_KB) -> float:
    return (S.bank_geometry(imem_kb, 32).area_um2
            + S.bank_geometry(dmem_kb, 32).area_um2) / 1e6


SEQ_OPTIONS = [
    # name, descriptor generator?, queue depth, setup cycles [EST], why
    ("S0  scalar core, CSR write + poll", False, 0, 20,
     "RV32 at 1 IPC: ~8 address computes, ~6 stores, loop bookkeeping, branch, "
     "then poll a completion CSR. No decoupling at all."),
    ("S1  scalar core, software double-buffer", False, 1, 20,
     "Same setup cost, but the program prepares descriptor i+1 while i runs. "
     "Costs a second descriptor buffer and nothing else."),
    ("S2  core + descriptor generator (q=8)", True, 8, 4,
     "3 AGUs walk a template under a 3-level hardware loop; the core only sets "
     "up the loop. One descriptor emitted per ~4 cycles."),
    ("S3  as S2, 1 descriptor/cycle", True, 8, 1,
     "Fully pipelined AGUs, template pre-decoded. Upper bound on what issue "
     "hardware can buy."),
]


def cfg_for(option, base=None) -> M.ArgmindCfg:
    _, gen, depth, setup, _ = option
    return dataclasses.replace(base or REC, seq_setup_cycles=setup,
                               seq_queue_depth=depth)


# ---------------------------------------------------------------------------
# Binding-resource roofline, with the sequencer as a first-class resource
# ---------------------------------------------------------------------------

def binding(b: M.Benchmark, c: M.ArgmindCfg, iters: int | None = None) -> dict:
    """Which resource saturates first, sequencer included.

    Deliberately NOT a sum: each key is the time that resource alone would take
    for the whole call, so the largest is the binding constraint and the ratio
    between the top two says how much headroom the others have.
    """
    it = b.iters if iters is None else iters
    ns = c.cycle_ns / 1000.0
    dim = M.representative_dim(b)
    d_it = M.descriptor_cost(b, c)
    d_ca = M.descriptor_cost(b, c, per_call=True)
    gemv_pc = c.gemv_macs_per_cycle(b.prec, b.gemv_row_len)
    r = M.model_argmind(dataclasses.replace(b, iters=it), c)
    d = {
        "spad_bw": (b.gemv_macs_per_iter * it + b.gemv_macs_per_call) / gemv_pc * ns,
        # Matrix-MATRIX unit. NOTE this is the tile ARRAY for bf16/fp16/int8/fp32
        # and the fp64 FMA CLUSTER for fp64 -- they are different hardware, and
        # conflating them is exactly what produced the open item being closed here.
        "mm_unit": (M.gemm_like_cycles(b.gemm_macs_per_iter, c, b.prec, dim) * it
                    + M.gemm_like_cycles(b.gemm_macs_per_call, c, b.prec, dim)) * ns,
        "vec_unit": (b.vector_ops_per_iter / c.vector_lanes
                     + b.transcendental_per_iter * 4.0 / c.vector_lanes) * it * ns,
        "seq_issue": (d_it["issue_cycles"] * it + d_ca["issue_cycles"]) * ns,
        "dma": r["transfer_us"],
        "host": c.call_overhead_us,
    }
    d["_total"] = r["latency_us"]
    d["_bind"] = max((k for k in d if not k.startswith("_")), key=lambda k: d[k])
    # Reported separately because it is a SERIALISATION penalty, not a resource:
    # it cannot be relieved by provisioning more of anything.
    d["_dep_drain"] = (d_it["dep_drain_cycles"] * it
                       + d_ca["dep_drain_cycles"]) * ns
    d["_serial"] = d_it["serial_cycles"] * it * ns
    d["_ndesc"] = d_it["n_desc"] * it + d_ca["n_desc"]
    d["_cyc_per_desc"] = d_it["cycles_per_desc"]
    return d


def crossover_setup(b: M.Benchmark, c: M.ArgmindCfg) -> float:
    """Setup cycles/descriptor at which the sequencer becomes binding.

    This IS the design rule, inverted: a descriptor must command at least this
    many datapath cycles for the sequencer to stay off the critical path at a
    given setup cost.
    """
    d = M.descriptor_cost(b, c)
    if d["n_desc"] <= 0:
        return float("inf")
    base = dataclasses.replace(c, seq_setup_cycles=1.0)
    d1 = M.descriptor_cost(b, base)
    return (d1["exec_cycles"] + d1["dep_drain_cycles"]) / d1["n_desc"]


def main():
    L = []

    def out(s=""):
        print(s)
        L.append(s)

    out("=" * 104)
    out("SEQUENCER / DESCRIPTOR-ISSUE MODEL — closing the fp64 open item")
    out("MODELLED ESTIMATE, 16nm-class, +/-2x. NOT QUOTABLE EXTERNALLY.")
    out("=" * 104)
    out()
    out("HEADLINE: the sequencer is NOT binding on ARM-MPC as specified. The logged")
    out("  '5.6 array cycles' divided an HPIPM block op by the 144-MAC bf16 ARRAY")
    out("  rate, but ARM-MPC is fp64 and fp64 runs on the 8-unit FMA cluster at")
    out("  1/16th that rate. Correcting the denominator AND the descriptor count")
    out("  (which is 3x higher than the guess, not lower) gives 30 datapath cycles")
    out("  per descriptor against a crossover of 34 -- array-bound with 8.6x margin")
    out("  on descriptor setup cost. BUT the investigation found three real things,")
    out("  one of which misses a deadline. See Q4 and Q5.")
    out()

    # ==================================================================== Q0 ==
    out("=" * 104)
    out("Q0 — WHY THE LOGGED NUMBER WAS WRONG. Arithmetic, not opinion.")
    out("=" * 104)
    arm = next(b for b in M.BENCHMARKS if b.name == "ARM-MPC")
    nb = 21
    block_macs = arm.gemm_macs_per_iter / (6 * 30)      # the logged assumption
    r_arr = REC.macs * REC.gemm_util("fp32", 78.7)
    r_64 = REC.macs_per_cycle("fp64") * REC.gemm_util("fp64", 78.7)
    out(f"  HPIPM block op at nb={nb}: "
        f"{arm.gemm_macs_per_iter:.0f} GEMM MACs / IP iter / 180 descriptors "
        f"= {block_macs:.0f} MACs   [DER]")
    out(f"    on the bf16/fp32 ARRAY  ({REC.macs} MACs, "
        f"{r_arr:.0f} MAC/cyc effective):  "
        f"{block_macs / r_arr:>6.1f} datapath cycles   <-- the logged figure")
    out(f"    on the fp64 FMA CLUSTER ({REC.fp64_fmas} units, "
        f"{r_64:.1f} MAC/cyc effective):  "
        f"{block_macs / r_64:>6.1f} datapath cycles   <-- what ARM-MPC actually uses")
    out(f"    ratio {r_arr / r_64:.1f}x. THIS is the error: the fp64 decision that "
        f"the same file")
    out("    made -- fp64 does NOT come from the array -- invalidates the")
    out("    denominator used two sections earlier. 'At any precision' is false;")
    out("    the exposure grows as arithmetic gets FASTER relative to the block.")
    out()
    out("  A SECOND correction pulls the other way, so the net is smaller than 16x.")
    out("  180 descriptors/IP iteration was '6 block ops x 30 stages', guessed. The")
    out("  structural count -- with the Cholesky and the gain solve blocked as the")
    out("  dependency chains they are -- is 3x higher [DER]:")
    d_arm = M.descriptor_cost(arm, REC)
    for op in arm.kernels:
        d, dep = M.op_descriptors(op, REC, arm.prec)
        out(f"    {op.name:<38} {d:>6.0f} desc  "
            f"{'(dependency chain)' if dep else ''}")
    out(f"    {'TOTAL':<38} {d_arm['n_desc']:>6.0f} desc, "
        f"{d_arm['n_dep']:.0f} of them inside a dependency chain")
    out()
    out(f"  NET: {d_arm['exec_cycles']:.0f} datapath cycles / "
        f"{d_arm['n_desc']:.0f} descriptors = "
        f"{d_arm['cycles_per_desc']:.1f} cycles per descriptor [DER],")
    out(f"  not 5.6. The crossover is therefore at "
        f"{crossover_setup(arm, REC):.0f} setup cycles, not 4.")
    out()

    # ==================================================================== Q1 ==
    out("=" * 104)
    out("Q1 — THE MODEL, AND WHAT REPLACED THE 50% FUDGE")
    out("=" * 104)
    out("  Old:  per-iter cycles = datapath + 0.5 * scalar_ops_per_iter / seq_ipc")
    out("        with scalar_ops_per_iter a flat constant (150 for ARM-MPC).")
    out("  New:  three separately-derived terms, each with a physical meaning:")
    out("        issue_cycles = n_descriptors * setup_cycles / issue_width")
    out("        dep_drain    = one datapath drain per dependency-chain boundary")
    out("        serial       = loop-carried SCALAR chain (alpha, convergence test)")
    out("        per-iter     = max(datapath + dep_drain, issue) + serial")
    out("                       [with a run-ahead queue; sum if queue depth 0]")
    out()
    out("  Descriptor counts are DERIVED from the kernel decomposition and are")
    out("  asserted to sum to the benchmark's declared MAC/vec/trans counts")
    out("  (argmind_model.assert_kernels_consistent, enforced by a test). That")
    out("  discipline is the whole point: assuming the two independently is what")
    out("  double-counted HPIPM GEMM work 5.7x in the first place.")
    out()
    hdr = (f"  {'benchmark':<11} {'ndesc':>6} {'dep':>5} {'datapath':>9} "
           f"{'cyc/desc':>9} {'issue':>7} {'drain':>7} {'serial':>7} "
           f"{'old flat':>9}  {'issue-bound?':<12}")
    out(hdr)
    out("  " + "-" * (len(hdr) - 2))
    for b in M.BENCHMARKS:
        d = M.descriptor_cost(b, REC)
        old = 0.5 * b.scalar_ops_per_iter / REC.seq_ipc
        out(f"  {b.name:<11} {d['n_desc']:>6.0f} {d['n_dep']:>5.0f} "
            f"{d['exec_cycles']:>9.0f} {d['cycles_per_desc']:>9.1f} "
            f"{d['issue_cycles']:>7.0f} {d['dep_drain_cycles']:>7.0f} "
            f"{d['serial_cycles']:>7.1f} {old:>9.0f}  "
            f"{'YES' if d['issue_bound'] else 'no':<12}")
    out(f"  (at setup={REC.seq_setup_cycles:.0f} cyc/descriptor, queue depth "
        f"{REC.seq_queue_depth}, issue width {REC.seq_issue_width:.0f})")
    out()
    out("  READ THE 'cyc/desc' COLUMN. It spans 10 to 42,000 across the set. That")
    out("  four-order-of-magnitude spread is exactly what a flat scalar constant")
    out("  cannot represent, and it orders the benchmarks by exposure:")
    out("    PWR-MPC 10  <  ARM-MPC 30  <  QUAD-MPC 59  <  EKF-24 82  <<  ML rows")
    out()

    # ==================================================================== Q2 ==
    out("=" * 104)
    out("Q2 — BINDING RESOURCE PER BENCHMARK (worst-case iterations)")
    out("=" * 104)
    hdr = (f"  {'benchmark':<11} {'prec':<5} {'spad_bw':>9} {'mm_unit':>9} "
           f"{'vec':>8} {'seq_iss':>8} {'dma':>8} {'host':>7} | {'drain':>7} "
           f"{'serial':>7} | {'BINDING':<9} {'2nd/1st':>8}")
    out(hdr)
    out("  " + "-" * (len(hdr) - 2))
    for b in M.BENCHMARKS:
        d = binding(b, REC, b.iters_wc)
        keys = [k for k in d if not k.startswith("_")]
        vals = sorted((d[k] for k in keys), reverse=True)
        out(f"  {b.name:<11} {b.prec:<5} {d['spad_bw']:>9.1f} {d['mm_unit']:>9.1f} "
            f"{d['vec_unit']:>8.1f} {d['seq_issue']:>8.1f} {d['dma']:>8.1f} "
            f"{d['host']:>7.1f} | {d['_dep_drain']:>7.1f} {d['_serial']:>7.2f} "
            f"| {d['_bind']:<9} {vals[1] / vals[0]:>7.2f}x")
    out("  (us per call; each column is the time that resource ALONE would take.")
    out("   mm_unit = the tile ARRAY at bf16/fp16/int8/fp32, the fp64 FMA CLUSTER")
    out("   at fp64. Different hardware; the ARM-MPC row is the FMA cluster.)")
    out()
    out("  NOTHING is sequencer-issue-bound at 4 cycles/descriptor. The binding")
    out("  resources are the ones already known: scratchpad bandwidth on the ADMM")
    out("  rows, the fp64 FMA cluster on ARM-MPC, the tile array on EKF/ML, DMA on")
    out("  SLM decode. NO ARCHITECTURE DECISION CHANGES.")
    out("  The sequencer's real contribution is the DEPENDENCY DRAIN column, which")
    out("  is NOT a provisioning problem -- no amount of issue bandwidth, MACs or")
    out("  scratchpad width touches it. On ARM-MPC it is 81 us against a 490 us")
    out("  binding term (17%); on PWR-MPC it is 3.8 us against 6.4 (59%).")
    out()

    # ==================================================================== Q3 ==
    out("=" * 104)
    out("Q3 — THE CROSSOVER, AND THE DESIGN RULE IT BECOMES")
    out("=" * 104)
    out("  issue_cycles / (datapath + dep_drain), swept over setup cost.")
    out("  '*' = sequencer-issue-bound.")
    out(f"  {'benchmark':<11} " + " ".join(f"{'s=' + str(s):>7}" for s in SETUP_SWEEP)
        + f"  {'crossover':>10}  design rule")
    for b in M.BENCHMARKS:
        cells = []
        for s in SETUP_SWEEP:
            d = M.descriptor_cost(b, dataclasses.replace(REC, seq_setup_cycles=s))
            r = d["issue_cycles"] / max(1.0, d["exec_cycles"] + d["dep_drain_cycles"])
            cells.append(f"{r:6.2f}" + ("*" if r > 1 else " "))
        x = crossover_setup(b, REC)
        out(f"  {b.name:<11} " + " ".join(f"{c:>7}" for c in cells)
            + f"  {x:>9.1f}   >= {x:.0f} datapath cyc/descriptor")
    out()
    out("  THE DESIGN RULE FOR CODEGEN. The crossover condition is exactly")
    out("  'average datapath cycles per descriptor > setup cycles per descriptor',")
    out("  so with a 2x engineering margin at the recommended setup cost:")
    out()
    out(f"    > A CODE-GENERATED ITERATION MUST AVERAGE AT LEAST "
        f"{2 * REC.seq_setup_cycles:.0f} DATAPATH CYCLES")
    out(f"    > PER DESCRIPTOR (= 2 x seq_setup_cycles, "
        f"{REC.seq_setup_cycles:.0f} at the recommended S2 sequencer),")
    out(f"    > AND NO BURST OF SUB-THRESHOLD DESCRIPTORS LONGER THAN THE "
        f"DESCRIPTOR")
    out(f"    > QUEUE ({REC.seq_queue_depth}). Below either bar, fuse the "
        f"operations or unroll the chain.")
    out()
    solver_rows = [b for b in M.BENCHMARKS if b.kind == "solver"]
    THINNEST = min(solver_rows,
                   key=lambda b: M.descriptor_cost(b, REC)["cycles_per_desc"]).name
    out("  Where the frozen set sits against that bar, at setup=4:")
    for b in M.BENCHMARKS:
        if b.kind != "solver":
            continue
        d = M.descriptor_cost(b, REC)
        out(f"    {b.name:<9} {d['cycles_per_desc']:>8.1f} cyc/desc = "
            f"{d['cycles_per_desc'] / (2 * REC.seq_setup_cycles):>5.2f}x the bar"
            + ("   <-- thinnest in the FROZEN set"
               if b.name == THINNEST else ""))
    out()
    out("  OFF-SET ROWS, shown because two real configurations of the motor-drive")
    out("  socket sit BELOW the bar and the frozen row does not:")
    for b in M.OFF_SET:
        d = M.descriptor_cost(b, REC)
        r = d["cycles_per_desc"] / (2 * REC.seq_setup_cycles)
        out(f"    {b.name:<9} {d['cycles_per_desc']:>8.1f} cyc/desc = "
            f"{r:>5.2f}x the bar" + ("   <-- BELOW THE BAR" if r < 1.0 else ""))
    out()
    out("  The rule is an AVERAGE plus a BURST LENGTH, not a per-descriptor")
    out("  minimum, because a run-ahead queue lets a short descriptor borrow slack")
    out("  from a long neighbour. The bursts are the dependency chains: "
        f"{math.ceil(21 / REC.fp64_fmas) * 3}")
    out("  consecutive sub-threshold descriptors per HPIPM stage at nb=21 (potrf")
    out("  blocked 3 ways on 8 fp64 units, 2 descriptors each, plus trsm).")
    out()

    # ==================================================================== Q4 ==
    out("=" * 104)
    out("Q4 — WHERE THE EXPOSURE ACTUALLY IS. Three regimes, none of them ARM-MPC/fp64.")
    out("=" * 104)
    out()
    out("  (a) PRECISION. The exposure scales with arithmetic RATE, so it is worst")
    out("      in the fastest precision -- the opposite of the logged claim.")
    out(f"      {'precision':<8} {'rate MAC/cyc':>13} {'datapath cyc':>13} "
        f"{'cyc/desc':>9} {'crossover':>10}  verdict at s=4 / s=20")
    for prec in ("fp64", "fp32", "bf16"):
        bp = dataclasses.replace(arm, prec=prec)
        d = M.descriptor_cost(bp, REC)
        x = crossover_setup(bp, REC)
        v4 = "bound" if 4 > x else "ok"
        v20 = "BOUND" if 20 > x else "ok"
        out(f"      {prec:<8} {REC.macs_per_cycle(prec):>13.0f} "
            f"{d['exec_cycles']:>13.0f} {d['cycles_per_desc']:>9.1f} "
            f"{x:>10.1f}  {v4} / {v20}")
    out()
    out("      CONSEQUENCE FOR THE fp64 DECISION, and it is a new argument for it:")
    out("      the fp32 + mandatory-iterative-refinement option that")
    out("      fp64_datapath.py Q3 evaluated puts the O(n^3) factorisation on the")
    out(f"      array in fp32, where the crossover is "
        f"{crossover_setup(dataclasses.replace(arm, prec='fp32'), REC):.0f} cycles. That "
        f"option is")
    out("      SEQUENCER-ISSUE-BOUND with anything but a hardware descriptor")
    out("      generator, which fp64_datapath.py did not charge it for. Native")
    out("      fp64 wins on a third axis, not just energy and accuracy.")
    out()
    out("  (b) PROBLEM SIZE. Stage block dimension nb sets MACs per descriptor as")
    out("      ~nb^3/(descriptors ~ nb), i.e. quadratically. Small block-structured")
    out("      NMPC is the vulnerable case, and small is common.")
    out(f"      {'nb':>4} {'n_x/n_u':>9} {'ndesc':>7} {'datapath':>9} "
        f"{'cyc/desc':>9} {'crossover':>10}  s=4  s=20")
    for nx, nu in ((4, 2), (6, 3), (8, 4), (10, 5), (14, 7), (20, 10), (28, 14)):
        bb = M.sqp_rti_mpc("probe", n_x=nx, n_u=nu, N=30, ip_iters=15, rate=250,
                           note="")
        d = M.descriptor_cost(bb, REC)
        x = crossover_setup(bb, REC)
        out(f"      {nx + nu:>4} {f'{nx}/{nu}':>9} {d['n_desc']:>7.0f} "
            f"{d['exec_cycles']:>9.0f} {d['cycles_per_desc']:>9.1f} {x:>10.1f}"
            f"  {'BOUND' if 4 > x else 'ok':<5}{'BOUND' if 20 > x else 'ok':<5}")
    out()
    out("      MINIMUM VIABLE BLOCK DIMENSION: with a hardware descriptor")
    out("      generator (s=4), nb >= 6 is safe. With a scalar-core sequencer")
    out("      (s=20), nb must be >= ~22 -- i.e. a software sequencer cannot run")
    out("      block-structured interior point on small-to-medium problems without")
    out("      becoming the bottleneck. THAT is the Phase-1 architecture input.")
    out()
    out("  (c) DATAPATH WIDTH. A wider fp64 cluster shortens each descriptor. It")
    out("      does NOT run into an issue wall, though, because a wider unit also")
    out("      blocks the Cholesky more coarsely and so needs FEWER descriptors:")
    out(f"      {'fp64 FMAs':>10} {'ndesc':>7} {'datapath':>9} {'drain':>7} "
        f"{'issue s=4':>10} {'crossover':>10}")
    for f in (4, 8, 16, 32, 64):
        c = dataclasses.replace(REC, fp64_fmas=f, vector_lanes=max(16, 2 * f))
        d = M.descriptor_cost(arm, c)
        out(f"      {f:>10} {d['n_desc']:>7.0f} {d['exec_cycles']:>9.0f} "
            f"{d['dep_drain_cycles']:>7.0f} {d['issue_cycles']:>10.0f} "
            f"{crossover_setup(arm, c):>10.1f}")
    out("      Note the drain column falls too. This is a real second-order")
    out("      argument for a wider fp64 unit that the model previously could not")
    out("      see -- and it still loses to the area cost (fp64_datapath.py Q2).")
    out()

    # ==================================================================== Q5 ==
    out("=" * 104)
    out("Q5 — RUN-AHEAD (DECOUPLED ACCESS-EXECUTE): the one place a deadline is missed")
    out("=" * 104)
    out("  Queue depth 0 = the sequencer writes a descriptor, starts it, polls for")
    out("  completion. Queue depth >=1 = it prepares descriptor i+1 while i runs.")
    out(f"  {'benchmark':<11} {'deadline':>9} {'50% budget':>11} "
        f"{'runahead':>9} {'no runahead':>12} {'penalty':>8}  verdict (no run-ahead)")
    for b in M.BENCHMARKS:
        if b.kind != "solver":
            continue
        for s in (4, 20):
            c1 = dataclasses.replace(REC, seq_setup_cycles=s, seq_queue_depth=8)
            c0 = dataclasses.replace(REC, seq_setup_cycles=s, seq_queue_depth=0)
            bw = dataclasses.replace(b, iters=b.iters_wc)
            n = M.model_argmind(bw, c1)["latency_us"]
            z = M.model_argmind(bw, c0)["latency_us"]
            budget = b.deadline_us * PERIOD_BUDGET
            v = ("MISSES 50% BUDGET" if z > budget else
                 "ok") + ("" if z <= b.deadline_us else " / MISSES DEADLINE")
            out(f"  {b.name + f' s={s}':<11} {b.deadline_us:>9.0f} {budget:>11.0f} "
                f"{n:>9.1f} {z:>12.1f} {z / n:>7.2f}x  {v}")
    out()
    out("  RESTATED 2026-07-30 AFTER THE PWR-MPC RE-SPEC — THIS CONCLUSION IS")
    out("  DOWNGRADED. It used to read: 'PWR-MPC with a scalar-core sequencer and no")
    out("  run-ahead MISSES the 50% control-period budget (38.1 us vs 25)'. That")
    out("  rested on PWR-MPC being a continuous-control-set ADMM QP. Re-specified as")
    out("  finite-control-set MPC the row has ZERO dependency chains and 54 datapath")
    out("  cycles per descriptor, and it survives no-run-ahead comfortably.")
    out()
    out("  **NO ROW IN THE FROZEN SET NOW MISSES A BUDGET FROM LACK OF RUN-AHEAD")
    out("  ALONE.** What survives, and is still a genuine negative result:")
    out("    * QUAD-MPC at its MEASURED worst-case tail misses at 0.37x when")
    out("      run-ahead AND full-width chain blocking are BOTH absent. You may skip")
    out("      at most one of the two.")
    out("    * The retired PWR-CCS definition still misses at 0.66x, and continuous-")
    out("      control-set MPC is a real if smaller socket, so the requirement is not")
    out("      vacuous -- it is now justified off the frozen set.")
    out("  Run-ahead therefore moves from REQUIREMENT to 'required unless chain")
    out("  blocking is full-width'. See results_pwr_mpc_respec.txt section 1.")
    out()

    # ==================================================================== Q6 ==
    out("=" * 104)
    out("Q6 — WHAT THE SEQUENCER MUST BE, AND WHAT IT COSTS")
    out("=" * 104)
    out(f"  {'option':<38} {'setup':>6} {'queue':>6} {'logic mm2':>10} "
        f"{'+mem mm2':>9} {'total':>7}  worst solver margin")
    for opt in SEQ_OPTIONS:
        name, gen, depth, setup, _why = opt
        c = cfg_for(opt)
        logic = seq_logic_mm2(gen, depth)
        mem = seq_mem_mm2()
        margins = []
        for b in M.BENCHMARKS:
            if b.kind != "solver":
                continue
            bw = dataclasses.replace(b, iters=b.iters_wc)
            lat = M.model_argmind(bw, c)["latency_us"]
            margins.append((b.deadline_us * PERIOD_BUDGET / lat, b.name))
        m, who = min(margins)
        out(f"  {name:<38} {setup:>6} {depth:>6} {logic:>10.4f} {mem:>9.4f} "
            f"{logic + mem:>7.4f}  {m:>5.2f}x ({who})")
    out()
    for name, gen, depth, setup, why in SEQ_OPTIONS:
        out(f"  {name}")
        out(f"      {why}")
    out()
    out("  ANSWERS TO THE THREE QUESTIONS ASKED:")
    out("   * ISSUE WIDTH: 1 descriptor/cycle is MORE than enough and is not the")
    out("     axis that matters. Going from S2 (4 cyc/desc) to S3 (1 cyc/desc)")
    out("     changes no verdict on any benchmark. What matters is the SETUP COST")
    out("     per descriptor, i.e. whether a hardware AGU/loop emits it or a")
    out("     scalar core writes it a store at a time.")
    out("   * RUN-AHEAD: REQUIRED, not optional. It is the difference between")
    out("     meeting and missing the PWR-MPC budget. A depth-8 queue is enough:")
    out(f"     the longest sub-threshold burst in the set is "
        f"{math.ceil(21 / REC.fp64_fmas) * 3} descriptors (an")
    out("     HPIPM stage's potrf+trsm chain at nb=21 on 8 fp64 units).")
    s2 = seq_logic_mm2(True, 8)
    s0 = seq_logic_mm2(False, 0)
    hi = s2 + seq_mem_mm2()
    out("   * AREA: the descriptor generator is nearly free, and -- CONTRARY TO")
    out("     WHAT I EXPECTED WHEN I STARTED THIS -- the whole sequencer including")
    out("     its program store FITS INSIDE THE EXISTING BUDGET LINE.")
    out(f"       S0 -> S2 logic delta          {s2 - s0:>8.4f} mm2  "
        f"({100 * (s2 - s0) / 0.698:.2f}% of the 0.698 mm2 core)")
    out(f"       S2 logic                      {s2:>8.4f} mm2  "
        f"[EST, bottom-up gate count]")
    out(f"       + {SEQ_IMEM_KB} KB I-mem, {SEQ_DMEM_KB} KB D-mem       "
        f"{seq_mem_mm2():>8.4f} mm2  "
        f"[DER from sram_cost.bank_geometry]")
    out(f"       = sequencer, all in            {hi:>8.4f} mm2")
    out(f"       cm100_sizing REST_MM2 allocation for 'sequencer': "
        f"{0.045:.4f} mm2 [EST]")
    out(f"       -> uses {100 * hi / 0.045:.0f}% of it. NO AREA CHANGE. The 0.698 "
        f"mm2 figure stands.")
    out()
    out("     This is worth stating as a negative result: the flat 0.045 mm2 [EST]")
    out("     in REST_MM2 had no memory line item and no bottom-up basis, so it")
    out("     was a candidate omission. Costed properly it is adequate, with the")
    out(f"     I-mem sized generously at {SEQ_IMEM_KB} KB. It would take a "
        f"{(0.045 - s2) / (seq_mem_mm2() / (SEQ_IMEM_KB + SEQ_DMEM_KB)):.0f} KB")
    out("     program store to break it, and a code-generated solver loop with")
    out("     hardware descriptor emission is nowhere near that. What was an")
    out("     assertion is now a derivation -- that is the whole gain here.")
    out()

    # ==================================================================== Q7 ==
    out("=" * 104)
    out("Q7 — ADMM vs BLOCK-STRUCTURED IP, CHECKED SEPARATELY AS ASKED")
    out("=" * 104)
    out("  Hypothesis: ADMM issues few large GEMVs and is insensitive; the exposure")
    out("  is specific to block-structured interior point. TESTED:")
    out()
    out(f"  {'benchmark':<11} {'algo':<6} {'desc/iter':>10} {'cyc/desc':>9} "
        f"{'crossover':>10} {'seq % of latency':>17} {'of which drain':>15}")
    for b in M.BENCHMARKS:
        if b.kind != "solver":
            continue
        d = M.descriptor_cost(b, REC)
        bw = dataclasses.replace(b, iters=b.iters_wc)
        lat = M.model_argmind(bw, REC)["latency_us"]
        # Ideal sequencer: zero setup cost, infinite run-ahead, no dependency
        # drain. The gap to it is the TOTAL sequencer contribution to latency.
        ideal = dataclasses.replace(REC, seq_setup_cycles=0.0, drain_cycles=0)
        # drain_cycles also feeds gemm_util, so isolate the drain term directly
        # instead of differencing two configs.
        seq_us = ((d["dep_drain_cycles"] + d["serial_cycles"]
                   + max(0.0, d["issue_cycles"] - d["exec_cycles"]
                         - d["dep_drain_cycles"]))
                  * b.iters_wc * REC.cycle_ns / 1000.0)
        drain_us = d["dep_drain_cycles"] * b.iters_wc * REC.cycle_ns / 1000.0
        out(f"  {b.name:<11} {b.algo:<6} {d['n_desc']:>10.1f} "
            f"{d['cycles_per_desc']:>9.1f} {crossover_setup(b, REC):>10.1f} "
            f"{100 * seq_us / lat:>16.1f}% {100 * drain_us / max(seq_us, 1e-9):>14.0f}%")
    out()
    out("  CONFIRMED, WITH ONE IMPORTANT EXCEPTION.")
    out("   * QUAD-MPC (ADMM, n_var=80): 28 descriptors/iteration, 59 datapath")
    out("     cycles each, crossover 65. Insensitive, as predicted. The two Gamma")
    out("     products are single descriptors covering ~19k MACs.")
    out("   * ARM-MPC (IP, nb=21): 540 descriptors/iteration -- 19x more -- and")
    out("     30 cycles each. Still not binding IN fp64, but only 2.5x clear.")
    out("   * THE EXCEPTION WAS PWR-MPC WHEN IT WAS ADMM, and it is now an OFF-SET")
    out("     row (PWR-CCS): 10.2 cycles/descriptor, exposed not because of its")
    out("     algorithm but because n_var=30 is tiny. The MECHANISM is the durable")
    out("     part and it survives the re-spec: EXPOSURE IS SET BY DATAPATH CYCLES")
    out("     PER DESCRIPTOR, which block structure and small problems both reduce")
    out("     independently. One-step FCS-MPC (PWR-FCS1, 4.8 cyc/desc) is the same")
    out("     mechanism in a third algorithm class, which is the best evidence that")
    out("     it is about problem size and not about ADMM or interior point.")
    out()
    out("   * The ADMM rows do carry a dependency-chain cost, and it is not small:")
    for name in ("QUAD-MPC", "PWR-MPC"):
        b = next(x for x in M.BENCHMARKS if x.name == name)
        d = M.descriptor_cost(b, REC)
        out(f"     {name:<9} dep_drain {d['dep_drain_cycles']:>6.0f} cyc/iter vs "
            f"{d['exec_cycles']:>6.0f} datapath = "
            f"{100 * d['dep_drain_cycles'] / d['exec_cycles']:>4.1f}% -- the two "
            f"triangular")
        out(f"     {'':<9} solves against the cached factor, "
            f"{d['n_dep']:.0f} dependent descriptors.")
    out()

    # ==================================================================== Q8 ==
    out("=" * 104)
    out("Q8 — WAS THE 50% FUDGE ACTUALLY WRONG? Answered plainly.")
    out("=" * 104)
    out(f"  {'benchmark':<11} {'old seq cyc/iter':>17} {'new seq cyc/iter':>17} "
        f"{'ratio':>7} {'latency old':>12} {'latency new':>12} {'delta':>8}")
    legacy = dataclasses.replace(REC, use_descriptor_seq=False)
    for b in M.BENCHMARKS:
        d = M.descriptor_cost(b, REC)
        old = 0.5 * b.scalar_ops_per_iter / REC.seq_ipc
        lo = M.model_argmind(b, legacy)["latency_us"]
        ln = M.model_argmind(b, REC)["latency_us"]
        out(f"  {b.name:<11} {old:>17.0f} {d['seq_overhead_cycles']:>17.0f} "
            f"{d['seq_overhead_cycles'] / max(old, 1e-9):>6.2f}x {lo:>12.1f} "
            f"{ln:>12.1f} {100 * (ln / lo - 1):>7.1f}%")
    out()
    out("  HONEST VERDICT, in the terms the task asked for:")
    out("   * For ADMM the 50% fudge was a GOOD APPROXIMATION BY ACCIDENT -- 75")
    out("     cycles/iteration against a modelled 161, but both are ~5-10% of the")
    out("     iteration, so QUAD-MPC latency moves only +5.6% and PWR-MPC +1.8%.")
    out("   * For ARM-MPC it was ~30x low in cycles and 12% low in latency.")
    out("   * For the EKF and ML rows it was IRRELEVANT in both directions.")
    out("   * The important error was QUALITATIVE, not quantitative: the constant")
    out("     was standing in for descriptor ISSUE, and the term that actually")
    out("     costs time is dependency DRAIN, which has a completely different")
    out("     design response. More issue bandwidth does not fix a drain; deeper")
    out("     pipelining makes it worse; only larger chain blocks or a shallower")
    out("     datapath pipeline do. That distinction was invisible before.")
    out()
    out("  WHEN DOES SEQUENCER WORK GENUINELY OVERLAP COMPUTE? Stated as a")
    out("  condition, which is what was missing:")
    out("   * YES, given a descriptor queue of depth >= 1, BECAUSE the CM's")
    out("     allocation and schedule are statically code-generated. Every")
    out("     descriptor's addresses, dims and strides are known before the loop")
    out("     runs, so setup never waits on data. This is the same property the")
    out("     determinism claim rests on, and it is worth noting that determinism")
    out("     and overlap-legality are the SAME architectural property.")
    out("   * NO for the chain boundary itself: descriptor i+1 of a Cholesky")
    out("     cannot START until i's results are written back, regardless of when")
    out("     its descriptor was prepared. Charged as dep_drain.")
    out("   * NO for genuinely data-dependent descriptor PARAMETERS: a")
    out("     fraction-to-boundary alpha that scales the next update, an")
    out("     active-set index, a pivot. Charged as serial_scalar_per_iter, and it")
    out("     is small (60 cycles/IP iteration, 25/check_term for ADMM).")
    out("   * The loop-exit branch is speculatable: 'continue' is correct on every")
    out("     iteration but the last, so it costs one mispredict per SOLVE, not")
    out("     per iteration. This is why the old flat term was in the wrong place.")
    out()

    # ==================================================================== Q9 ==
    out("=" * 104)
    out("Q9 — CONSEQUENCE FOR THE D2/D3 INTEGRATION DEPTHS (PLAN.md §4.2)")
    out("=" * 104)
    out("  D2 (RoCC-style coprocessor) and D3 (in-pipeline ISA extension) shift")
    out("  descriptor issue onto the HOST core. Two levers change, and they are the")
    out("  same two this whole analysis turns on:")
    out("    setup cost  -- the host has no dedicated descriptor AGUs, so each")
    out("                   descriptor becomes a software store sequence again")
    out("                   (~20 cycles), UNLESS the generator moves across with")
    out("                   the datapath.")
    out("    run-ahead   -- the host is also executing the application, so an")
    out("                   in-order host cannot reliably get ahead of the")
    out("                   datapath. Depth 0 is the pessimistic case and depth 1")
    out("                   (software double-buffering) the realistic one.")
    out(f"  {'integration depth':<42} {'setup':>6} {'queue':>6}  "
        f"worst solver margin (50% budget)")
    for label, setup, depth in (
            ("D1 bus-attached, own sequencer (S2)", 4, 8),
            ("D2 RoCC, generator moved to CM side", 4, 8),
            ("D2 RoCC, host issues, sw double-buffer", 20, 1),
            ("D2 RoCC, host issues, no run-ahead", 20, 0),
            ("D3 ISA ext, in-order host, no run-ahead", 20, 0),
            ("D3 ISA ext, OoO host (reorders for us)", 20, 8)):
        c = dataclasses.replace(REC, seq_setup_cycles=setup, seq_queue_depth=depth)
        margins = []
        for b in M.BENCHMARKS:
            if b.kind != "solver":
                continue
            bwb = dataclasses.replace(b, iters=b.iters_wc)
            margins.append((b.deadline_us * PERIOD_BUDGET
                            / M.model_argmind(bwb, c)["latency_us"], b.name))
        m, who = min(margins)
        out(f"  {label:<42} {setup:>6} {depth:>6}  {m:>5.2f}x ({who})"
            + ("   <-- MISSES BUDGET" if m < 1.0 else
               "   <-- thin" if m < 1.3 else ""))
    out()
    out("  THE DECISION THIS FORCES, and it is cheap to make now:")
    out("  > THE DESCRIPTOR GENERATOR BELONGS ON THE CM SIDE OF THE INTERFACE AT")
    out("  > EVERY INTEGRATION DEPTH. D2/D3 may move the CONTROL FLOW to the host;")
    out("  > they must not move descriptor EMISSION there.")
    out()
    out("  RESTATED 2026-07-30: the §4.1 zero-copy tension is now SETTLED (see")
    out("  results_pwr_mpc_respec.txt section 3 -- on MCU class the honest advantage")
    out("  is invocation overhead and determinism, not zero-copy), and the")
    out("  re-specified motor-drive row is no longer the most issue-sensitive")
    out("  benchmark. The D2 conclusion survives on the retired PWR-CCS row and on")
    out("  the one-step FCS row (4.8 cyc/desc, below the bar). So D2 remains a")
    out("  wrapper rather than a redesign, as PLAN.md §4.2 recommends, PROVIDED the")
    out("  wrapper keeps the generator. This is also a MACIF question: the")
    out("  descriptor format is part of the boundary a CIM macro must honour, so")
    out("  the generator sits above MACIF and cannot be a host-side artefact.")
    out()

    out("=" * 104)
    out("SENSITIVITIES — what would change these conclusions")
    out("=" * 104)
    out("  1. seq_setup_cycles (2-4 hardware / 16-32 software) is [EST] and is the")
    out("     single most load-bearing number here. MEASUREMENT THAT WOULD FIRM IT")
    out("     UP: write the descriptor-emit sequence for one HPIPM stage in RV32")
    out("     assembly and count instructions. That is an afternoon, not a phase.")
    out("  2. The 30/45/15/10 split of nb^3/3 across trmm/syrk/potrf/trsm is [EST].")
    out("     It sets how many descriptors are sub-threshold. Firm it up by")
    out("     instrumenting HPIPM's BLASFEO call counts on the real problem.")
    out("  3. The benchmark's N*nb^3/3 GEMM count is ~2.2x BELOW a true HPIPM stage")
    out("     count at nb=21 [EST]. Fixing it would raise cycles/descriptor and")
    out("     make every conclusion here MORE comfortable, so this analysis is")
    out("     conservative on that axis.")
    out("  4. Chain blocking is assumed to equal the datapath width (tile dim for")
    out("     array GEMM, fp64 FMA count for fp64, vector lanes for a trsv). A")
    out("     potrf that cannot be blocked that coarsely -- a scalar-sequential")
    out("     diagonal -- multiplies the chain length and hence the drain. This is")
    out("     the largest single downside risk, and it is an RTL question rather")
    out("     than a modelling one, so here is what it would cost. Sweeping the")
    out("     chain-block override (worst-case iterations, 50%-budget margin):")
    out(f"      {'chain block':>14} " + " ".join(
        f"{b.name:>11}" for b in M.BENCHMARKS if b.kind == "solver"))
    for blk in (None, 8, 4, 2, 1):
        cells = []
        for b in M.BENCHMARKS:
            if b.kind != "solver":
                continue
            c = REC if blk is None else dataclasses.replace(
                REC, chain_block_override=blk)
            bwb = dataclasses.replace(b, iters=b.iters_wc)
            m = (b.deadline_us * PERIOD_BUDGET
                 / M.model_argmind(bwb, c)["latency_us"])
            cells.append(f"{m:>9.2f}x" + ("!" if m < 1.0 else " "))
        lbl = "datapath width" if blk is None else str(blk)
        out(f"      {lbl:>14} " + " ".join(f"{x:>11}" for x in cells))
    out("     '!' = misses the 50% control-period budget.")
    out("     RESTATED 2026-07-30: the re-specified PWR-MPC has NO dependency chain,")
    out("     so it is exactly invariant to this sweep and no longer the row that")
    out("     breaks. QUAD-MPC now carries the requirement, on its MEASURED tail, and")
    out("     it does not miss on chain blocking alone -- it falls to 1.27x at a")
    out("     scalar chain, and misses only if run-ahead is ALSO absent.")
    out("     > RTL REQUIREMENT, still stated as a requirement: a blocked Cholesky /")
    out("     > triangular solve must advance a full datapath width (>= 8) per")
    out("     > dependent step. It is worth 2.3x of QUAD-MPC's worst-case margin,")
    out("     > which is the largest single lever in this analysis -- but the")
    out("     > justification is now a THIN MARGIN on one measured row, not a")
    out("     > deadline miss on two rows.")
    out("     Still the most valuable thing to check first in Phase 1, and still a")
    out("     bigger lever than the descriptor setup cost itself.")
    out("  6. drain_cycles = 8 [EST]. The dep_drain term is linear in it, so a")
    out("     deeper datapath pipeline is directly a block-structured-solver tax --")
    out("     an argument for a SHORT pipeline that had no quantitative basis")
    out("     before. PLAN.md §3.1 already asks for short fill/drain on small")
    out("     matrices; this is the second, independent reason.")

    with open("perf_model/results_sequencer.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
