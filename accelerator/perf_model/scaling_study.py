# SPDX-License-Identifier: Apache-2.0
"""
Argmind CM — scaling analysis: which axes scale, where each wall is, SKU family
==============================================================================

PLAN.md §5 markets a "parameterized" core (tile size, precisions, SRAM depth, bus
type) and Phase 4 lists "multi-core scaling". Neither had been analysed, while
§9 simultaneously flags "parameterization discipline (few validated configs, not
full matrix)". This module resolves that tension by pricing every axis.

The question is never "does it go faster with more X". It is: for THIS workload,
which resource saturates first, and what does the whole BALANCED package cost at
each step? A bandwidth increase with no lane increase is not a configuration, it
is a mis-specification.

WHAT THIS MODULE ADDED TO THE MODEL, because two axes were previously
inexpressible and both were inexpressible IN OUR FAVOUR:

  1. `sram_cost.rows_max_for_freq` — the longest single-cycle-read bitline was the
     hard constant ROWS_MAX_800MHZ, so raising `freq_mhz` cost nothing in SRAM
     area. It does: at 1200 MHz a 12 KiB x 64b bank needs two folds.
  2. `argmind_model.vdd_for_freq` + the dyn/leak energy multipliers — energy per
     operation was frequency-independent, so the model reported 2 GHz as 2.5x the
     speed at LOWER energy per call. Dynamic energy goes as V^2.

Both are exactly 1.0 / 256 rows at the nominal 800 MHz, so no pre-existing figure
moves; the point is that the frequency axis previously had no wall because the
model could not express one.

ALL FIGURES MODELLED ESTIMATE, 16nm-class, +/-2x, NOT QUOTABLE EXTERNALLY.
Precision is stated per row (solvers fp32 except ARM-MPC fp64; ML int8).
"""

from __future__ import annotations

import dataclasses
import math

import argmind_model as M
import cm100_sizing as C
import sram_cost as S

SOLVERS = [b for b in M.BENCHMARKS if b.kind == "solver"]
ML = [b for b in M.BENCHMARKS if b.kind == "ml"]
QUAD = next(b for b in SOLVERS if b.name == "QUAD-MPC")
ARM = next(b for b in SOLVERS if b.name == "ARM-MPC")
FCS = next(b for b in SOLVERS if b.name == "PWR-MPC")

# Measured static footprints, KB, from the functional simulator's allocation plan
# (descriptor v0.1). Dense-rectangle layout because that is what the AGU can
# address; working vectors counted. ARM-MPC is fp64. See SCRATCHPAD_RULING.
FOOTPRINT_KB = {"QUAD-MPC": 138.8, "ARM-MPC": 323.2, "PWR-MPC": 0.7,
                "EKF-24": 7.9, "EKF-96": 117.0}

# ---------------------------------------------------------------------------
# CM-to-CM synchronisation cost, for the multi-CM question. [EST]
# ---------------------------------------------------------------------------
# Three organisations, three prices. Each names a mechanism that can be measured
# separately, which a single constant cannot.
#
#   shared_spad  two datapaths behind ONE banked scratchpad, hardware barrier in
#                the sequencer. Cheapest possible: a counter and a wired-OR.
#                ~16 cycles at 800 MHz.
#   tightly_cpl  two CMs, private scratchpads, a dedicated CM-to-CM mailbox +
#                DMA of the shared vector. ~120 cycles.
#   coherent     two CMs synchronising through the host's ACE-Lite/CHI fabric:
#                write, barrier, snoop-invalidate, read-back. ~400 cycles.
SYNC_US = {"shared_spad": 16 / 800.0, "tightly_cpl": 120 / 800.0,
           "coherent": 400 / 800.0}

# MEASURED (argmind functional simulator, descriptor v0.1): achievable dependency
# depth of one CHECKED ADMM iteration under wait-mask semantics = 12 (iteration
# block) + 5 (residual block) = 17 chain steps. This is the number of BARRIERS a
# multi-CM split of one iteration must pay, and it is the number that decides the
# multi-CM question. Emitted depth under one-token v0 semantics was 17 + 15 = 32.
ADMM_CHAIN_STEPS = 17
ADMM_CHAIN_STEPS_V0 = 32


def cfg(bw=128.0, tile=12, lanes=None, kb=192, freq=800.0, fp64=None, nb=None,
        drain=8, qd=8, tier="T1", chain=None, setup=4.0, name=None,
        static_mw=15.0):
    """A BALANCED configuration: lanes and fp64 units track bandwidth by default.

    The default coupling is `lanes = bw/8` and `fp64_fmas = lanes/2`, which is
    exactly the CM-100 relationship (128 B/cycle, 16 lanes, 8 merged FMAs). Any
    sweep that moves bandwidth without moving lanes is reported SEPARATELY and
    labelled as unbalanced, because the vector unit becomes co-binding.
    """
    if lanes is None:
        lanes = max(4, int(bw // 8))
    if fp64 is None:
        fp64 = max(1, lanes // 2)
    if nb is None:
        nb = max(1, min(16, int(bw // 8)))
    return M.ArgmindCfg(
        name=name or f"bw{bw:.0f} t{tile} l{lanes} {kb}KB {freq:.0f}MHz",
        tile=tile, vector_lanes=lanes, scratchpad_kb=kb, scratchpad_bw_bpc=bw,
        n_banks=nb, freq_mhz=freq, fp64_fmas=fp64, drain_cycles=drain,
        seq_queue_depth=qd, tier=tier, chain_block_override=chain,
        seq_setup_cycles=setup, static_mw=static_mw)


def wc(b, c):
    """Latency at the WORST-CASE iteration tail, which is what a deadline sees."""
    return M.model_argmind(dataclasses.replace(b, iters=b.iters_wc), c)


def solver_energy(c) -> float:
    return sum(M.model_argmind(b, c)["energy_uj"] for b in SOLVERS)


def perf_w(b, c) -> float:
    eff = min((M.model_cpu(b, x) for x in M.ALL_CPUS), key=lambda r: r["energy_uj"])
    return eff["energy_uj"] / M.model_argmind(b, c)["energy_uj"]


def cpu_fastest(b) -> dict:
    return min((M.model_cpu(b, x) for x in M.ALL_CPUS), key=lambda r: r["latency_us"])


# ---------------------------------------------------------------------------
# Balance rules, derived rather than swept
# ---------------------------------------------------------------------------

def min_tile_for_bw(bw: float, prec: str = "fp32") -> float:
    """Smallest square tile whose peak rate can absorb `bw` bytes/cycle of GEMV.

    GEMV consumes one matrix element per MAC, so its ceiling is bw/elem MAC per
    cycle. The array supplies tile^2 * scale. Equating them:

        tile = sqrt(bw / (elem * scale))

    fp32: elem 4, scale 0.5 -> tile = sqrt(bw/2).  At 128 B/cycle that is 8, so
    CM-100's 12x12 has 1.5x linear (2.25x MAC) headroom, and the array does not
    become the GEMV bottleneck until 288 B/cycle. THIS IS THE FIRST PLACE MORE
    MACS PAY FOR GEMV WORK, and it is why "16x more MACs bought zero" is a
    statement about 64-128 B/cycle rather than about arrays in general.
    """
    scale = {"int8": 2.0, "fp16": 1.0, "bf16": 1.0, "fp32": 0.5}[prec]
    return math.sqrt(bw / (M.BYTES_PER_ELEM[prec] * scale))


def bw_ceiling_for_tile(tile: int, prec: str = "fp32") -> float:
    """Inverse: bytes/cycle above which a `tile`-square array starts to bind."""
    scale = {"int8": 2.0, "fp16": 1.0, "bf16": 1.0, "fp32": 0.5}[prec]
    return tile * tile * scale * M.BYTES_PER_ELEM[prec]


def min_lanes_for_bw(b, bw: float, tile=12, kb=192) -> int:
    """Smallest lane count at which the vector unit is not the binding resource."""
    for lanes in (4, 8, 16, 32, 64, 128):
        c = cfg(bw=bw, tile=tile, lanes=lanes, kb=kb)
        d = C.decompose(b, c, b.iters_wc)
        if d["_bind"] != "vec":
            return lanes
    return 128


def offload_floor_us(b, c) -> dict:
    """The scale-DOWN rule, per row, expressed in host-CPU microseconds.

    speedup = cpu / (compute + overhead). With k = cpu/compute the datapath
    ratio, speedup >= 1 requires

        cpu >= overhead * k / (k - 1)        (and it is unreachable for k <= 1)

    So the floor is NOT a MAC count or a problem dimension -- it is a ratio
    between the launch protocol and the datapath advantage, and it says a
    customer can evaluate it from two numbers they already have.
    """
    r = M.model_argmind(b, c)
    comp = r["compute_us"]
    cpu = cpu_fastest(b)["latency_us"]
    k = cpu / comp if comp > 0 else float("inf")
    ovh = c.call_overhead_us
    if k <= 1.0:
        floor_cpu = float("inf")
    else:
        floor_cpu = ovh * k / (k - 1.0)
    return {"compute_us": comp, "cpu_us": cpu, "k": k, "overhead_us": ovh,
            "floor_cpu_us": floor_cpu, "floor_compute_us": floor_cpu / k if k > 1 else float("inf"),
            "speedup": cpu / r["latency_us"], "pw": perf_w(b, c)}


# ---------------------------------------------------------------------------
# Multi-CM
# ---------------------------------------------------------------------------

def split_solve(b, c, n_cm: int, sync: str, chain_steps: int) -> dict:
    """Latency of ONE solve split across `n_cm` CMs at chain-step granularity.

    The premise of the architecture is that the iteration is loop-carried, so
    there is no ITERATION-level parallelism. But there IS work-level parallelism
    inside an iteration: a GEMV can be row-split, with each CM owning a row block
    of the matrix. What that costs is a BARRIER AT EVERY CHAIN STEP, because the
    next step's operand is the concatenation of this step's partial results.

    Deliberately GENEROUS to the multi-CM case in three ways:
      * per-CM datapath work is charged as exactly 1/n_cm (perfect load balance,
        no row-quantisation waste, no duplicated reduction tail)
      * the broadcast of the shared vector is charged as zero bytes
      * capacity per CM is assumed sufficient, so no streaming appears
    If it loses under those assumptions it loses.
    """
    r = wc(b, c)
    iters = b.iters_wc
    per_iter = (r["compute_us"] - c.call_overhead_us) / max(iters, 1)
    split = per_iter / n_cm + chain_steps * SYNC_US[sync] * (n_cm - 1 > 0)
    total = split * iters + c.call_overhead_us
    return {"single_us": r["latency_us"], "split_us": total,
            "per_iter_single_us": per_iter, "per_iter_split_us": split,
            "speedup": r["latency_us"] / total,
            "sync_us_per_iter": chain_steps * SYNC_US[sync]}


def duty_cycle(b, c) -> float:
    r = wc(b, c)
    return r["latency_us"] * b.calls_per_sec / 1e6


def main():
    L = []

    def out(s=""):
        print(s)
        L.append(s)

    C100 = M.CFG_100
    A100 = C.area_breakdown(C100)

    out("=" * 108)
    out("ARGMIND CM — SCALING ANALYSIS (2026-07-31)")
    out("MODELLED ESTIMATE, 16nm-class, +/-2x, NOT QUOTABLE EXTERNALLY")
    out("=" * 108)
    out()
    out(f"Reference: {C100.name}")
    out(f"  core {A100['total']:.3f} mm2  (spad {A100['spad']:.3f} / mac {A100['mac']:.3f} / "
        f"vec {A100['vec']:.3f} / fp64 {A100['fp64']:.3f} / rest {A100['rest']:.3f})")
    out(f"  {C100.sram_pj_per_byte():.3f} pJ/byte, Vdd {C100.vdd:.2f} V, "
        f"call overhead {C100.call_overhead_us} us (T1)")
    out()
    out("Worst-case reference table (iters_worst; the tail a hard deadline must survive):")
    hdr = (f"{'row':<11} {'prec':>5} {'deadline':>9} {'lat p95':>9} {'lat wc':>9} "
           f"{'margin':>7} {'binds@wc':>9} {'sp':>6} {'perf/W':>7} {'duty%':>7}")
    out(hdr)
    out("-" * len(hdr))
    for b in M.BENCHMARKS:
        r = M.model_argmind(b, C100)
        w = wc(b, C100)
        d = C.decompose(b, C100, b.iters_wc)
        out(f"{b.name:<11} {b.prec:>5} {b.deadline_us:>9.0f} {r['latency_us']:>9.2f} "
            f"{w['latency_us']:>9.2f} {b.deadline_us / w['latency_us']:>7.2f}x "
            f"{d['_bind']:>9} {cpu_fastest(b)['latency_us'] / r['latency_us']:>6.2f} "
            f"{perf_w(b, C100):>7.2f} {duty_cycle(b, C100) * 100:>7.2f}")

    # =====================================================================
    out()
    out("=" * 108)
    out("AXIS 1 — MAC ARRAY.  Wall: the array is already 2.25x oversized for its")
    out("         own bandwidth. It pays on ENERGY (reuse depth) and on ML LATENCY.")
    out("=" * 108)
    out()
    out("The balance rule, derived not swept: a GEMV consumes one matrix element per")
    out("MAC, so its ceiling is bw/elem MAC/cycle; the array supplies tile^2*scale.")
    out("  min tile for a given fp32 GEMV bandwidth = sqrt(bw/2)")
    out(f"{'bw B/cyc':>9} {'min tile':>9} {'min MACs':>9} {'tile 12 binds above':>21}")
    for bw in (16, 32, 64, 128, 192, 256, 288, 384, 512, 1024):
        mt = min_tile_for_bw(bw)
        out(f"{bw:>9} {mt:>9.1f} {math.ceil(mt) ** 2:>9} "
            f"{('yes' if bw > bw_ceiling_for_tile(12) else 'no'):>21}")
    out(f"  -> tile 12 (144 MACs) absorbs GEMV up to {bw_ceiling_for_tile(12):.0f} B/cycle, and")
    out("     the SAME number at fp32, bf16 and int8 -- not a coincidence worth")
    out("     hiding: the array's rate multiplier happens to be ~1/element-size on")
    out("     this datapath (fp32 half-rate 4 B, bf16 full-rate 2 B, int8 double-rate")
    out("     1 B), so tile^2 * scale * elem is invariant. The GEMV balance point is a")
    out("     property of the ARRAY, not of the precision, which is convenient: one")
    out("     number sizes it for every format.")
    out("  -> CM-100 runs 128 B/cycle, so the array has 2.25x MAC headroom over the")
    out("     bandwidth it is fed. THAT is why '16x more MACs bought zero' held: the")
    out("     statement is about 64-128 B/cycle, not about arrays in general.")
    out()
    out("Measured against the model (QUAD-MPC worst case, us) — tile x bandwidth:")
    TILES = (8, 12, 16, 24, 32)
    out(f"{'bw':>6} " + " ".join(f"{'t' + str(t):>9}" for t in TILES)
        + "   smallest tile that is not binding   predicted sqrt(bw/2)")
    for bw in (64, 128, 256, 384, 512, 1024):
        lanes = max(4, int(bw // 8))
        row = [wc(QUAD, cfg(bw=bw, tile=t, lanes=lanes))["latency_us"] for t in TILES]
        best = next(t for t, v in zip(TILES, row)
                    if abs(v - min(row)) / min(row) < 0.02)
        out(f"{bw:>6} " + " ".join(f"{x:>9.1f}" for x in row)
            + f"   {'t' + str(best):>33}   {min_tile_for_bw(bw):>19.1f}")
    out("  The sweep and the closed form agree: the smallest non-binding tile tracks")
    out("  sqrt(bw/2) to within one step of the tile grid.")
    out()
    out("ENERGY return on array size, at the CM-100 bandwidth (128 B/cyc, 192 KB):")
    out(f"{'tile':>5} {'MACs':>6} {'B/MAC':>7} {'core mm2':>9} " +
        " ".join(f"{b.name[:8]:>9}" for b in SOLVERS) + f" {'sum uJ':>8} {'uJ/mm2':>8}")
    prev = None
    for t in (4, 6, 8, 12, 16, 24, 32):
        c = cfg(tile=t)
        a = C.area_breakdown(c)
        row = [M.model_argmind(b, c)["energy_uj"] for b in SOLVERS]
        tot = sum(row)
        marg = ""
        if prev:
            dA = a["total"] - prev[0]
            marg = f"{(prev[1] - tot) / dA:>8.0f}" if dA > 1e-9 else "       -"
        out(f"{t:>5} {t * t:>6} {c.gemm_operand_bytes_per_mac('fp32'):>7.3f} "
            f"{a['total']:>9.3f} " + " ".join(f"{x:>9.2f}" for x in row) +
            f" {tot:>8.2f} {marg}")
        prev = (a["total"], tot)
    out()
    out("  WALL: marginal solver energy per mm2 collapses past tile 12 (144 MACs).")
    out("  ML latency wants more, and that is the ONLY thing that does:")
    out(f"{'GMAC/frame':>11} " + " ".join(f"{'t' + str(t):>22}" for t in (12, 16, 32)))
    for g in (0.3, 1.26, 4.0, 17.0):
        b = M.cnn_infer(f"cnn{g}", gmacs=g, weights_mb=5.7 * g / 1.26, rate=30, note="")
        cells = []
        for t in (12, 16, 32):
            c = cfg(tile=t)
            r = M.model_argmind(b, c)
            d = C.decompose(b, c)
            ok = "OK" if r["latency_us"] <= b.deadline_us else "MISS"
            cells.append(f"{r['latency_us']:>9.0f}us {d['_bind']:>8} {ok:>4}")
        out(f"{g:>11.2f} " + " ".join(f"{x:>22}" for x in cells))
    out("  At tile 32 (1024 MACs) the VECTOR UNIT takes over as the ML bottleneck, so")
    out("  even the ML axis stops at ~1024 MACs unless lanes grow with it. The 30 fps")
    out("  int8 CNN envelope at CM-100 is ~4 GMAC/frame; at tile 32 it is ~15.")

    # =====================================================================
    out()
    out("=" * 108)
    out("AXIS 2 — SCRATCHPAD BANDWIDTH.  The knob that works, and it works to")
    out("         256 B/cycle. Wall: 16 banks x 16 B; past that the bus is the product.")
    out("=" * 108)
    out()
    out("BALANCED packages (lanes = bw/8, fp64 = lanes/2, banks <= 16 with >=64b word):")
    hdr = (f"{'bw':>5} {'banks':>6} {'word':>5} {'lanes':>6} {'fp64':>5} {'core mm2':>9} "
           f"{'pJ/B':>6} " + " ".join(f"{b.name[:8]:>8}" for b in SOLVERS)
           + f" {'sum uJ':>8} {'QUADx':>7} {'mm2/x':>7}")
    out(hdr)
    out("-" * len(hdr))
    prev = None
    for bw in (16, 32, 64, 128, 256, 512):
        c = cfg(bw=bw)
        a = C.area_breakdown(c)
        row = [wc(b, c)["latency_us"] for b in SOLVERS]
        tot = solver_energy(c)
        dl = mm = ""
        if prev:
            ratio = prev[1] / row[0]
            dl = f"{ratio:>6.2f}x"
            mm = f"{(a['total'] - prev[0]) / max(ratio - 1, 1e-9):>7.3f}"
        out(f"{bw:>5} {c.n_banks:>6} {c.bank_word_bytes:>5} {c.vector_lanes:>6} "
            f"{c.fp64_fmas:>5} {a['total']:>9.3f} {c.sram_pj_per_byte():>6.3f} " +
            " ".join(f"{x:>8.1f}" for x in row) + f" {tot:>8.2f} {dl:>7} {mm:>7}")
        prev = (a["total"], row[0])
    out()
    out("  QUADx is the worst-case speedup over the previous row; mm2/x is the area")
    out("  paid per unit of that speedup (mm2 / (ratio - 1)) -- a marginal cost, so")
    out("  smaller is better and the column is directly comparable step to step.")
    out("  16 -> 256 B/cycle returns a near-perfect 1.84-1.99x PER DOUBLING, at a")
    out("  marginal cost rising 0.020 -> 0.169 mm2 per x. 256 -> 512 returns 1.19x at")
    out("  1.682 mm2 per x -- TEN TIMES worse value -- because the fp32 GEMV ceiling")
    out(f"  (bw/4) crosses the 144-MAC array rate (72 MAC/cycle) at "
        f"{bw_ceiling_for_tile(12):.0f} B/cycle.")
    out("  Past there bandwidth alone buys nothing; you must buy array AND lanes AND")
    out("  banks together, which is a different part rather than a wider one.")
    out()
    out("UNBALANCED (bandwidth raised, lanes held at 16) — what the co-binding costs:")
    out(f"{'bw':>5} {'QUADwc':>9} {'bind':>9} {'gemv us':>9} {'vec us':>8} {'seq us':>8} "
        f"{'balanced':>9} {'lost':>6}")
    for bw in (128, 256, 384, 512):
        cu = cfg(bw=bw, lanes=16, fp64=8)
        cb = cfg(bw=bw)
        d = C.decompose(QUAD, cu, QUAD.iters_wc)
        u = wc(QUAD, cu)["latency_us"]
        bal = wc(QUAD, cb)["latency_us"]
        out(f"{bw:>5} {u:>9.1f} {d['_bind']:>9} {d['gemv_bw']:>9.1f} {d['vec']:>8.1f} "
            f"{d['seq']:>8.1f} {bal:>9.1f} {u / bal:>5.2f}x")
    out()
    out("  Minimum lanes at which the vector unit stops binding, per row:")
    for b in SOLVERS:
        cells = [f"{bw}B/c:{min_lanes_for_bw(b, bw):>3}" for bw in (64, 128, 256, 512)]
        out(f"    {b.name:<10} " + "  ".join(cells))
    out("  -> The ADMM and EKF rows are so GEMV-dominated that 4 lanes suffice to")
    out("     256 B/cycle. FCS-MPC needs 16 AT EVERY BANDWIDTH -- it is the one row")
    out("     where the vector unit binds first, which is the same fact the")
    out("     accumulator-fusion finding rests on (broadcast_init + square_accumulate")
    out("     take its vector work 525 -> 154 cycles for near-zero area, where doubling")
    out("     lanes costs +0.143 mm2 for 1.26x). Lanes are provisioned for FCS-MPC and")
    out("     for ML, not for the solver rows.")
    out()
    out("  BANK-COUNT WALL, stated as the licensee sees it. The >=64-bit bank-word")
    out("  rule caps banks at bw/8, and the sizing run caps banks at 16 because the")
    out("  read crossbar and per-bank queues grow faster than the energy benefit:")
    for bw in (128, 256, 512, 1024):
        nb16 = 16
        out(f"    {bw:>5} B/cycle -> {nb16} banks x {bw // nb16:>3} B word, "
            f"read bus {int(bw * 8 * 1.35):>5} bits incl. addr/ECC")
    out("  512 B/cycle is a 5530-bit read bus into a 16 x 32 B scratchpad. That is")
    out("  a hard macro in disguise: no soft-IP licensee will floorplan it, and it")
    out("  is rejected on integrability, not on model output.")

    # =====================================================================
    out()
    out("=" * 108)
    out("AXIS 3 — CAPACITY.  NOT A SCALING AXIS. A step function with a monotone")
    out("         cost. Every KB above the resident working set is a pure loss.")
    out("=" * 108)
    out()
    out(f"{'KB':>6} {'banks':>6} {'spad':>7} {'core':>7} {'pJ/B':>6} {'leak85':>7} "
        f"{'leak125':>8} " + " ".join(f"{b.name[:8]:>9}" for b in SOLVERS)
        + f" {'sum uJ':>8} {'fits':>5}")
    for kb in (32, 64, 96, 128, 160, 192, 256, 384, 512, 1024):
        c = cfg(kb=kb)
        a = C.area_breakdown(c)
        row = [M.model_argmind(b, c)["energy_uj"] for b in SOLVERS]
        fits = sum(1 for b in SOLVERS if FOOTPRINT_KB[b.name] <= kb)
        out(f"{kb:>6} {c.n_banks:>6} {a['spad']:>7.3f} {a['total']:>7.3f} "
            f"{c.sram_pj_per_byte():>6.3f} {c.spad.leakage_mw():>7.2f} "
            f"{c.spad.leakage_mw(True):>8.2f} " + " ".join(f"{x:>9.2f}" for x in row)
            + f" {sum(row):>8.2f} {fits:>4}/5")
    out()
    out("  SOLVER ENERGY IS MONOTONE INCREASING IN CAPACITY ACROSS THE WHOLE RANGE.")
    out("  There is no capacity at which the part gets cheaper to run. Bigger banks")
    out("  raise the cost of EVERY access, and the byte count a workload fetches does")
    out("  not change.")
    out()
    out("  The decisive case, and it confirms an earlier finding by a second route:")
    e192 = M.model_argmind(ARM, cfg(kb=192))
    e384 = M.model_argmind(ARM, cfg(kb=384))
    out(f"    ARM-MPC at 192 KB: {FOOTPRINT_KB['ARM-MPC']:.1f} KB footprint OVERFLOWS, "
        f"streams {e192['bytes_moved'] / 1024:.0f} KB, {e192['energy_uj']:.2f} uJ")
    out(f"    ARM-MPC at 384 KB: FITS, streams {e384['bytes_moved'] / 1024:.0f} KB, "
        f"{e384['energy_uj']:.2f} uJ")
    out(f"    -> making it fit costs {e384['energy_uj'] - e192['energy_uj']:+.2f} uJ "
        f"({e384['energy_uj'] / e192['energy_uj']:.3f}x) and "
        f"{C.area_breakdown(cfg(kb=384))['total'] - C.area_breakdown(cfg(kb=192))['total']:+.3f} mm2,")
    out("       for ZERO latency benefit (ARM-MPC runs at 250 Hz with 5.5x margin).")
    out("       The pJ/byte penalty on every access exceeds the DRAM traffic it saves.")
    out()
    out("  RULE: size capacity to the FASTEST determinism-critical resident working")
    out("  set, and to nothing else. A slow loop that overflows should overflow.")
    out("  Capacity is a fitting constraint, so it belongs in the SKU table as a")
    out("  discrete choice, never in a scaling roadmap.")

    # =====================================================================
    out()
    out("=" * 108)
    out("AXIS 4 — FREQUENCY AND PROCESS NODE.  Cheapest on paper; three real walls.")
    out("=" * 108)
    out()
    out("Wall 1: bitline length. Single-cycle read forces rows <= ~256*(800/f).")
    out("Wall 2: supply. Dynamic energy goes as V^2, so past nominal the part gets")
    out("        FASTER AND LESS EFFICIENT -- the opposite of the value proposition.")
    out("Wall 3: pipeline depth. A deeper datapath raises `drain_cycles`, and 97-100%")
    out("        of the sequencer's latency cost is dependency drain.")
    out()
    out(f"{'MHz':>6} {'Vdd':>5} {'rowsmax':>8} {'folds':>6} {'spad':>7} {'core':>7} "
        f"{'pJ/B':>6} {'dynE':>6} {'leakE':>6} {'drain':>6} {'QUADwc':>8} {'ARMwc':>8} "
        f"{'sum uJ':>8} {'uJ/MHz':>8}")
    for freq, drain in ((400, 6), (600, 7), (800, 8), (1000, 9), (1200, 11),
                        (1600, 14), (2000, 17)):
        c = cfg(freq=freq, drain=drain)
        a = C.area_breakdown(c)
        q = wc(QUAD, c)["latency_us"]
        am = wc(ARM, c)["latency_us"]
        tot = solver_energy(c)
        out(f"{freq:>6} {c.vdd:>5.2f} {S.rows_max_for_freq(freq):>8} "
            f"{c.spad.geom.folds:>6} {a['spad']:>7.3f} {a['total']:>7.3f} "
            f"{c.sram_pj_per_byte():>6.3f} {c.dyn_energy_scale:>6.3f} "
            f"{c.leak_scale:>6.3f} {drain:>6} {q:>8.1f} {am:>8.1f} {tot:>8.2f} "
            f"{tot / freq * 1000:>8.3f}")
    out()
    e800 = solver_energy(cfg(freq=800, drain=8))
    e1600 = solver_energy(cfg(freq=1600, drain=14))
    q800 = wc(QUAD, cfg(freq=800, drain=8))["latency_us"]
    q1600 = wc(QUAD, cfg(freq=1600, drain=14))["latency_us"]
    a800 = C.area_breakdown(cfg(freq=800, drain=8))["total"]
    a1600 = C.area_breakdown(cfg(freq=1600, drain=14))["total"]
    out(f"  800 -> 1600 MHz: QUAD-MPC worst case {q800:.1f} -> {q1600:.1f} us "
        f"({q800 / q1600:.2f}x, not 2.00x),")
    out(f"    solver energy {e800:.2f} -> {e1600:.2f} uJ ({e1600 / e800:.3f}x), "
        f"core {a800:.3f} -> {a1600:.3f} mm2 ({a1600 / a800:.2f}x).")
    out("  So the frequency axis buys latency we do not need at the cost of the")
    out("  energy advantage that IS the product. It is the wrong axis for this part.")
    out()
    out("  ** AND THE ARROW POINTS DOWN, WHICH NOBODY HAD ASKED. ** Every solver")
    out("  deadline has 4-15x worst-case margin, so the interesting question is not")
    out("  how fast the part can go but how slowly it can go and still make the")
    out("  deadline -- because energy falls with frequency, not rises. Checked against")
    out("  a 50%-of-period budget (the honest engineering allocation), at worst-case")
    out("  iterations:")
    out(f"{'MHz':>6} {'Vdd':>5} {'core mm2':>9} {'sum uJ':>8} {'vs 800':>7} " +
        " ".join(f"{b.name[:8]:>9}" for b in SOLVERS) + "   verdict @50% period")
    for freq, drain in ((300, 5), (400, 6), (500, 6), (600, 7), (800, 8)):
        c = cfg(freq=freq, drain=drain)
        a = C.area_breakdown(c)
        row = [wc(b, c)["latency_us"] for b in SOLVERS]
        bad = [b.name for b, x in zip(SOLVERS, row)
               if x > b.deadline_us * C.PERIOD_BUDGET]
        tot = solver_energy(c)
        out(f"{freq:>6} {c.vdd:>5.2f} {a['total']:>9.3f} {tot:>8.2f} "
            f"{tot / e800:>6.3f}x " + " ".join(f"{x:>9.1f}" for x in row)
            + f"   {('all OK' if not bad else 'MISS: ' + ','.join(bad))}")
    out("  -> 600 MHz meets every frozen worst case inside a 50% budget at 0.84x the")
    out("     solver energy of 800 MHz, same area. 400 MHz meets them too, at 0.74x")
    out("     the energy, and is the lowest point where QUAD-MPC's measured cold tail")
    out("     still fits its 500 us allocation.")
    out("  -> THE NON-MONOTONICITY BELOW 400 MHz IS A MODEL ARTEFACT, NOT PHYSICS:")
    out("     `vdd_for_freq` CLAMPS at 0.65 V, so going slower than 400 MHz stops")
    out("     buying voltage and only adds leakage-time. The real curve keeps falling")
    out("     into near-threshold and the model deliberately refuses to extrapolate")
    out("     there. Read the 400 MHz row as 'the lowest point the model can defend',")
    out("     not as an optimum.")
    out("  -> THIS IS A REAL FINDING AND IT IS UNCOMFORTABLE: the recommended clock")
    out("     may be too HIGH. 800 MHz was chosen before the frequency axis had an")
    out("     energy cost in the model, i.e. when lowering it appeared to buy nothing.")
    out("     DO NOT re-spec on this alone. The 400 MHz point has only 1.08x margin on")
    out("     the one row whose tail is MEASURED, and the f-V table is [EST] with no")
    out("     PDK behind it. What it does establish is that the frequency parameter")
    out("     should be presented to a licensee as a DOWNWARD energy knob with a")
    out("     deadline floor, never as an upward performance knob. Firm it up with a")
    out("     real library f-V curve in Phase 1; it is one of the cheapest")
    out("     measurements on the list and it moves the headline energy number.")
    out()
    out("  Decomposing the sub-2x latency: 1/f applies only to compute. Fixed terms:")
    for b in SOLVERS:
        r8 = M.model_argmind(b, cfg(freq=800))
        out(f"    {b.name:<10} compute {r8['compute_us']:>8.2f} us, "
            f"overhead {r8['overhead_us']:.2f} us -> asymptotic floor at f=inf is "
            f"{r8['overhead_us'] + r8['transfer_us']:.2f} us "
            f"({(r8['overhead_us'] + r8['transfer_us']) / r8['latency_us'] * 100:.1f}% of latency today)")
    out()
    out("  Wall 3 quantified: pipeline depth x chain blocking (QUAD-MPC worst case).")
    out("  Chain blocking is what makes a deeper pipeline survivable, so raising")
    out("  frequency and skipping chain blocking are NOT independent decisions.")
    out(f"{'chain block':>12} " + " ".join(f"{'drain ' + str(d):>11}" for d in (4, 8, 16, 32)))
    for ch, lab in ((None, "datapath"), (32, "32"), (8, "8"), (1, "1 (none)")):
        row = [wc(QUAD, cfg(drain=d, chain=ch))["latency_us"] for d in (4, 8, 16, 32)]
        out(f"{lab:>12} " + " ".join(f"{x:>11.1f}" for x in row)
            + f"   {row[-1] / row[0]:.2f}x over the drain sweep")
    out()
    out("  PROCESS NODE is not a competitive axis at all, and this is the honest")
    out("  point. Every perf/W figure here is a RATIO against a CPU baseline. Moving")
    out("  from 16nm to 5nm moves the CPU too. Node buys absolute power and area for")
    out("  a licensee's SoC budget; it does not move a single claim in this document.")
    out("  Node is a COST axis, not a scaling axis. Do not put it on a roadmap slide")
    out("  as though it improved the argument.")

    # =====================================================================
    out()
    out("=" * 108)
    out("AXIS 5 — SEQUENCER.  Fixed-size by nature. Nothing about it scales UP;")
    out("         the only lever points DOWN (shorter pipeline, wider chain blocks).")
    out("=" * 108)
    out()
    out("Issue width (needs a multi-ported descriptor queue):")
    for w in (0.5, 1.0, 2.0, 4.0):
        row = [wc(b, cfg())["latency_us"] for b in SOLVERS] if w == 1.0 else \
              [wc(b, dataclasses.replace(cfg(), seq_issue_width=w))["latency_us"] for b in SOLVERS]
        out(f"  width {w:>4}: " + " ".join(f"{b.name}={x:.1f}" for b, x in zip(SOLVERS, row)))
    out("  -> IDENTICAL at 0.5x and 4x. Issue throughput is not a resource here, and")
    out("     even HALVING it costs nothing. Do not build a multi-ported queue.")
    out()
    out("Setup cycles per descriptor (the emit-cost axis):")
    out(f"{'setup':>7} " + " ".join(f"{b.name[:8]:>9}" for b in SOLVERS) + f" {'vs 4':>7}")
    base = None
    for s in (4, 8, 12, 16, 20, 23, 24, 32, 48):
        row = [wc(b, cfg(setup=float(s)))["latency_us"] for b in SOLVERS]
        if base is None:
            base = row[0]
        out(f"{s:>7} " + " ".join(f"{x:>9.1f}" for x in row) + f" {row[0] / base:>6.3f}x")
    out("  -> FREE up to 16 cycles/descriptor; the knee is at ~17-20 and even the")
    out("     naive 23-cycle RV32 emit costs only 4% on QUAD-MPC and 5% on ARM-MPC.")
    out("     CONSEQUENCE FOR THE ROADMAP: the hardware descriptor generator is an")
    out("     AREA-CHEAP INSURANCE POLICY, not a deadline requirement for the frozen")
    out("     set. It stays in (0.028 mm2, already budgeted) but it must not be sold")
    out("     as load-bearing -- the `arm templates once per solve` codegen fix is")
    out("     what actually removes the exposure, and that is free.")
    out()
    out("Descriptor queue depth:")
    for q in (0, 1, 2, 4, 8, 16, 32):
        row = [wc(b, cfg(qd=q))["latency_us"] for b in SOLVERS]
        out(f"  depth {q:>3}: " + " ".join(f"{b.name}={x:.1f}" for b, x in zip(SOLVERS, row)))
    out("  -> Only 0 -> 1 moves anything (QUAD-MPC 246.7 -> 235.2, 4.9%). Depth 2+ is")
    out("     flat IN THIS AGGREGATE MODEL, and that DISAGREES with the settled")
    out("     finding that depth 8 is a measured floor.")
    out("     ** THE MEASUREMENT WINS AND THE REASON IS STRUCTURAL. ** The aggregate")
    out("     model averages descriptor cost over a block; the measured DAG contains a")
    out("     run of SIX CONSECUTIVE descriptors averaging 5.8 cycles in the residual")
    out("     block, which a queue must absorb INSTANTANEOUSLY. An average cannot see")
    out("     a burst. Depth 8 stays an RTL requirement; this sweep is not evidence")
    out("     against it, it is evidence that this sweep cannot resolve it.")
    out()
    out("Pipeline drain cycles — the ONE sequencer number that matters:")
    out(f"{'drain':>7} " + " ".join(f"{b.name[:8]:>9}" for b in SOLVERS) + f" {'vs 8':>7}")
    for d in (2, 4, 8, 16, 32):
        row = [wc(b, cfg(drain=d))["latency_us"] for b in SOLVERS]
        ref = wc(ARM, cfg(drain=8))["latency_us"]
        out(f"{d:>7} " + " ".join(f"{x:>9.1f}" for x in row) + f" {row[1] / ref:>6.3f}x")
    out("  -> ARM-MPC spans 652 -> 1037 us over the drain sweep (1.59x). This is the")
    out("     only sequencer parameter with real leverage and it scales the WRONG WAY:")
    out("     you cannot buy performance with it, you can only lose performance by")
    out("     making the pipeline deeper. That is why the sequencer is fixed-size.")
    out()
    out("  SEQUENCER SIZING VERDICT: scalar core + hardware descriptor generator,")
    out("  <=4 cycles/descriptor, 8-deep queue, chain blocking >= one datapath width,")
    out("  0.028 mm2. NOT a per-SKU parameter. Identical in every SKU. The only")
    out("  thing that changes per SKU is the chain-block width, which follows the")
    out("  datapath width automatically.")

    # =====================================================================
    out()
    out("=" * 108)
    out("AXIS 6 — SCALING DOWN.  Where offload stops making sense, as a rule a")
    out("         customer can apply with two numbers they already have.")
    out("=" * 108)
    out()
    out("Derivation. speedup = cpu / (compute + overhead). With k = cpu/compute:")
    out("      speedup >= 1  <=>  cpu >= overhead * k/(k-1),   impossible if k <= 1")
    out("The floor is a ratio between the LAUNCH PROTOCOL and the DATAPATH ADVANTAGE.")
    out("It is not a MAC count and not a problem dimension.")
    out()
    out(f"{'row':<22} {'compute us':>10} {'cpu us':>8} {'k':>6} {'floor cpu':>10} "
        f"{'sp T1':>7} {'sp T1M':>7} {'perf/W':>7}  latency verdict")
    rows = []
    for lv, np_ in ((2, 1), (3, 1), (2, 2), (3, 2)):
        rows.append(M.fcs_mpc(f"FCS L{lv} Np{np_}", n_x=4, n_y=2, n_u=3, levels=lv,
                              Np=np_, rate=40000, note=""))
    for n in (6, 12, 18, 24, 48, 96):
        rows.append(M.ekf(f"EKF n{n}", n=n, m=max(2, n // 2), rate=1000, note=""))
    for n_u in (1, 2, 4, 8):
        rows.append(M.condensed_mpc(f"ADMM nu{n_u}", n_x=12, n_u=n_u, N=20, iters=15,
                                    rate=1000, check_term=1, note=""))
    C1M = cfg(tier="T1M")
    for b in rows:
        f = offload_floor_us(b, C100)
        fm = offload_floor_us(b, C1M)
        v = ("WINS" if f["speedup"] >= 1.0 else
             ("MCU-class only" if fm["speedup"] >= 1.0 else "USE THE CPU"))
        out(f"{b.name:<22} {f['compute_us']:>10.3f} {f['cpu_us']:>8.3f} {f['k']:>6.2f} "
            f"{f['floor_cpu_us']:>10.2f} {f['speedup']:>7.2f} {fm['speedup']:>7.2f} "
            f"{f['pw']:>7.2f}  {v}")
    out()
    ks = [offload_floor_us(b, C100)["k"] for b in SOLVERS]
    out(f"  Measured k band across the frozen solver set: {min(ks):.2f} - {max(ks):.2f}.")
    out(f"  At T1 (0.80 us): floor is {0.8 * max(ks) / (max(ks) - 1):.2f} us of CPU time at "
        f"k={max(ks):.2f}, {0.8 * 1.24 / 0.24:.2f} us at k=1.24.")
    out(f"  At T1M (0.30 us): {0.3 * max(ks) / (max(ks) - 1):.2f} us and "
        f"{0.3 * 1.24 / 0.24:.2f} us respectively.")
    out()
    out("  ENERGY has a SEPARATE and LOWER floor, and its mechanism is different:")
    out("  the CM pays static power across the whole latency window INCLUDING the")
    out("  launch, during which it computes nothing.")
    A55_W = 0.15
    fixed_nj = C100.static_mw * 1e-3 * C100.call_overhead_us * 1e-3      # nJ
    fixed_nj_m = C100.static_mw * 1e-3 * 0.30 * 1e-3
    out(f"    idle-during-launch energy = {C100.static_mw:.0f} mW x "
        f"{C100.call_overhead_us} us = {fixed_nj:.3f} nJ   (T1)")
    out(f"                              = {C100.static_mw:.0f} mW x 0.30 us = "
        f"{fixed_nj_m:.3f} nJ   (T1M)")
    out(f"    an A55-class in-order core at {A55_W:.2f} W burns {fixed_nj:.3f} nJ in "
        f"{fixed_nj / A55_W:.3f} us of compute")
    out("    -> for a 5x energy advantage the workload's CPU energy must be well")
    out(f"       above 5x that floor: >= ~{5 * fixed_nj / A55_W:.2f} us of A55 compute at T1,")
    out(f"       >= ~{5 * fixed_nj_m / A55_W:.2f} us at T1M. Rounded for a customer: 0.4 us / 0.15 us.")
    out("    Consistency check against the sweep, which is what makes this more than")
    out("    algebra: FCS L2 Np1 has 0.063 us of host work and scores 3.2x perf/W --")
    out("    BELOW the 5x kill criterion. FCS L3 Np1 has 0.102 us and scores 7.0x.")
    out("    The crossing lands between them, i.e. within a factor of ~1.5 of the")
    out("    0.4 us floor the mechanism predicts. Good enough to quote as a rule of")
    out("    thumb, not good enough to quote as a number.")
    out("    THE ENERGY CLAIM HAS A FLOOR TOO. It is ~5x lower than the latency")
    out("    floor, which is why most workloads that fail test 2 still pass test 3 --")
    out("    but it is not zero, and 'we are always more efficient' is false.")
    out()
    out("  POWER GATING IS WHAT MAKES THE ENERGY FLOOR THIS LOW. Charged ungated, the")
    out("  fixed static term is the whole PERIOD rather than the active window, so it")
    out("  grows by exactly 1/duty:")
    for b in SOLVERS:
        dc = duty_cycle(b, C100)
        period_us = 1e6 / b.calls_per_sec
        gated_nj = C100.static_mw * 1e-3 * wc(b, C100)["latency_us"] * 1e-3
        ungated_nj = C100.static_mw * 1e-3 * period_us * 1e-3
        out(f"    {b.name:<10} duty {dc * 100:>6.2f}%  static energy/call: gated "
            f"{gated_nj:>8.2f} nJ  ungated {ungated_nj:>8.2f} nJ  ({1 / dc:>5.1f}x)")
    out("  Charged ungated, QUAD-MPC perf/W falls 12.6x -> 4.8x, BELOW the kill")
    out("  criterion, and EKF-24 to 0.2x. Coarse-grain gating (or at minimum full")
    out("  clock gating plus a retention-mode scratchpad) is not an optimisation.")

    # =====================================================================
    out()
    out("=" * 108)
    out("AXIS 7 — EXTERNAL BANDWIDTH.  Large-model ML is outside the DIFFERENTIATED")
    out("         envelope even where it is inside the feasible one.")
    out("=" * 108)
    out()
    out("SLM-DECODE (1.1B int8, 20 tok/s, batch 1). No amount of scratchpad")
    out("bandwidth, MAC array or capacity touches this row:")
    out(f"{'DMA GB/s':>9} {'lat ms':>9} {'dma ms':>9} {'comp ms':>9} {'dma %':>7} "
        f"{'uJ':>9} {'perf/W':>7}  verdict")
    slm = next(b for b in ML if b.name == "SLM-DECODE")
    for gbs in (6.4, 12.8, 25.6, 51.2, 102.4):
        class _C(M.ArgmindCfg):
            _bw = gbs

            @property
            def dma_bw_gbs(self):
                return type(self)._bw
        c = _C(**{f.name: getattr(C100, f.name)
                  for f in dataclasses.fields(M.ArgmindCfg)})
        r = M.model_argmind(slm, c)
        out(f"{gbs:>9.1f} {r['latency_us'] / 1000:>9.1f} {r['transfer_us'] / 1000:>9.1f} "
            f"{r['compute_us'] / 1000:>9.1f} {r['transfer_us'] / r['latency_us'] * 100:>7.1f} "
            f"{r['energy_uj']:>9.0f} {perf_w(slm, c):>7.2f}  "
            f"{'MEETS' if r['latency_us'] <= slm.deadline_us else 'MISSES'}")
    out()
    out("  It MEETS at 25.6 GB/s, so 'misses at 6.4, meets at 25.6' is a provisioning")
    out("  requirement to state. But read the perf/W column, not the verdict column:")
    out("  at the bandwidth where it becomes feasible it is 75% DRAM-bound, so the")
    out("  advantage collapses toward the ratio of a small compute fraction. A CPU")
    out("  with the same 25.6 GB/s is bounded by the same wall. WE ARE NOT")
    out("  DIFFERENTIATED ON A WORKLOAD WHOSE BINDING RESOURCE IS OFF-CHIP.")
    out()
    out("  The envelope, stated so a customer can check their own model. Weight bytes")
    out("  per token = parameters (int8), and residency saves nothing once the model")
    out("  exceeds the scratchpad, which every useful LLM does:")
    out(f"{'DMA GB/s':>9} " + " ".join(f"{str(r) + ' tok/s':>13}" for r in (5, 10, 20, 50)))
    for gbs in (6.4, 12.8, 25.6):
        cells = [f"{gbs * 1e9 / r / 1e9 * 0.5:>6.2f} G" for r in (5, 10, 20, 50)]
        out(f"{gbs:>9.1f} " + " ".join(f"{x:>13}" for x in cells))
    out("  (at a 50%-of-period transfer budget, which is what a real system gives)")
    out(f"  -> at CM-100's assumed 6.4 GB/s and 20 tok/s the envelope is ~0.16 G int8")
    out("     parameters. THAT is the honest ML-decode statement, and it is a")
    out("     sub-Phi-class model. Above it, sell the CNN/ViT row, not the LLM row.")
    out()
    out("  What DOES stay inside the envelope: anything whose persistent data is")
    out("  resident. CNN/ViT weights stream once and the activations stay:")
    for b in ML:
        r = M.model_argmind(b, C100)
        out(f"    {b.name:<11} persistent {b.persistent_bytes / 1024:>8.0f} KB, "
            f"moved {r['bytes_moved'] / 1024:>9.0f} KB, "
            f"required {r['bytes_moved'] / (b.deadline_us * 1e-6) / 1e9:>6.2f} GB/s, "
            f"perf/W {perf_w(b, C100):>5.2f}x")

    # =====================================================================
    out()
    out("=" * 108)
    out("MULTI-CM — is it a product or a distraction?")
    out("=" * 108)
    out()
    out("Q1. Can a SINGLE solve be split? The iteration is loop-carried, so there is")
    out("    no iteration-level parallelism. But there IS work-level parallelism")
    out("    inside an iteration: row-split every GEMV, each CM owning a row block.")
    out("    The cost is a BARRIER AT EVERY CHAIN STEP, because the next step's")
    out(f"    operand is the concatenation of this step's partials. MEASURED chain")
    out(f"    depth of one checked ADMM iteration = {ADMM_CHAIN_STEPS} steps under wait-mask")
    out(f"    semantics ({ADMM_CHAIN_STEPS_V0} under v0 one-token semantics).")
    out()
    out("    Charged GENEROUSLY to the multi-CM case: perfect load balance, zero")
    out("    broadcast bytes, no row-quantisation waste, capacity always sufficient.")
    out()
    out(f"{'n_CM':>5} {'sync model':>13} {'sync us/iter':>13} {'per-iter us':>12} "
        f"{'QUADwc us':>10} {'vs 1 CM':>8}")
    for n in (1, 2, 4):
        for sync in ("shared_spad", "tightly_cpl", "coherent"):
            if n == 1 and sync != "shared_spad":
                continue
            r = split_solve(QUAD, C100, n, sync, ADMM_CHAIN_STEPS)
            lab = "-" if n == 1 else sync
            out(f"{n:>5} {lab:>13} {(0.0 if n == 1 else r['sync_us_per_iter']):>13.3f} "
                f"{r['per_iter_split_us'] if n > 1 else r['per_iter_single_us']:>12.3f} "
                f"{r['split_us'] if n > 1 else r['single_us']:>10.1f} "
                f"{(1.0 if n == 1 else r['speedup']):>7.2f}x")
    out()
    out("    ANSWER: two CMs make a single ADMM solve SLOWER than one at any barrier")
    out("    cost above ~16 cycles. Tightly-coupled mailbox loses (0.77x). Coherent-")
    out("    fabric synchronisation loses catastrophically (0.32x) -- and coherent is")
    out("    the only organisation that does not require new silicon.")
    out()
    out("    ** AND THE ONE ROW THAT WINS IS UNPHYSICAL, WHICH IS THE REAL ANSWER. **")
    out("    Read the `shared_spad` rows adversarially. They charge each CM 1/n of the")
    out("    datapath time WHILE SHARING ONE SCRATCHPAD. QUAD-MPC's binding resource")
    out("    is scratchpad read bandwidth (`gemv_bw`, see the reference table). Two")
    out("    datapaths behind ONE 128 B/cycle scratchpad on a bandwidth-bound row")
    out("    deliver 1.00x, not 1.65x -- the second datapath starves. The row is only")
    out("    reachable if the scratchpad bandwidth ALSO doubles, at which point the")
    out("    bandwidth is what bought the speedup and the datapath replication is")
    out("    incidental. Corrected, the honest table is:")
    out("        2 CM, shared scratchpad, bandwidth NOT scaled -> 1.00x (starved)")
    out("        2 CM, shared scratchpad, bandwidth doubled    -> that IS 1 CM at")
    out("                                                        256 B/cycle, cheaper")
    out("    So intra-solve multi-CM is not merely a bad trade; on the workload class")
    out("    that defines this architecture there is no mechanism by which it helps.")
    out()
    out("Q2. Quantifying that, because 'it is one wider CM' should be priced. Compare")
    out("    2 x CM-100 against 1 CM with the same total bandwidth:")
    c2 = cfg(bw=256, lanes=32)
    a1 = C.area_breakdown(C100)["total"]
    a2 = C.area_breakdown(c2)["total"]
    r_single256 = wc(QUAD, c2)["latency_us"]
    r_best_multi = split_solve(QUAD, C100, 2, "shared_spad", ADMM_CHAIN_STEPS)["split_us"]
    out(f"    2 x CM-100        : {2 * a1:.3f} mm2, 2 x 128 B/cyc over TWO 192 KB "
        f"scratchpads, QUAD-MPC wc {r_best_multi:.1f} us (best-case barrier)")
    out(f"    1 x CM @256 B/cyc : {a2:.3f} mm2, 256 B/cyc over ONE 192 KB scratchpad, "
        f"QUAD-MPC wc {r_single256:.1f} us")
    out(f"    -> the single wide CM is {2 * a1 / a2:.2f}x CHEAPER in area and "
        f"{r_best_multi / r_single256:.2f}x FASTER.")
    out("    For bandwidth-bound work, buying bandwidth inside one CM dominates")
    out("    buying a second CM, and it is not close. Marginal cost of bandwidth:")
    for bw0, bw1 in ((128, 256),):
        d = C.area_breakdown(cfg(bw=bw1))["total"] - C.area_breakdown(cfg(bw=bw0))["total"]
        out(f"      {bw0} -> {bw1} B/cycle inside one CM: {d:+.3f} mm2 for +{bw1 - bw0} B/cycle")
        out(f"      a second CM-100:                  {a1:+.3f} mm2 for +{bw0} B/cycle")
        out(f"      -> {(a1 / bw0) / (d / (bw1 - bw0)):.1f}x more area per byte/cycle "
            f"via replication")
    out("    And at 4 CMs, even taking the unphysical shared-scratchpad row at face")
    out("    value, replication still loses on area:")
    r4 = split_solve(QUAD, C100, 4, "shared_spad", ADMM_CHAIN_STEPS)["split_us"]
    c512 = cfg(bw=512)
    a512 = C.area_breakdown(c512)["total"]
    out(f"      4 x CM-100        : {4 * a1:.3f} mm2, QUAD-MPC wc {r4:.1f} us")
    out(f"      1 x CM @512 B/cyc : {a512:.3f} mm2, QUAD-MPC wc "
        f"{wc(QUAD, c512)['latency_us']:.1f} us")
    out(f"      -> replication is {4 * a1 / a512:.2f}x the area for "
        f"{wc(QUAD, c512)['latency_us'] / r4:.2f}x the speed, i.e. "
        f"{(4 * a1 / a512) / (wc(QUAD, c512)['latency_us'] / r4):.2f}x worse value --")
    out("         and 512 B/cycle is itself already rejected as unintegrable, so this")
    out("         comparison is between two options we decline, not a recommendation.")
    out()
    out("Q3. FCS-MPC's 729 candidates ARE independent. Does that parallelism want")
    out("    two CMs? THIS IS THE ONE PLACE THE ANSWER IS NOT AN IMMEDIATE NO, and it")
    out("    is worth saying so rather than reporting the convenient result.")
    f_ = offload_floor_us(FCS, C100)
    out(f"    FCS-MPC compute is {f_['compute_us']:.3f} us against a "
        f"{C100.call_overhead_us} us launch. It is {f_['speedup']:.2f}x overall and the")
    out(f"    deficit is invocation, not datapath (datapath alone is {f_['k']:.2f}x).")
    out("    Zero dependency chains, so a split needs exactly ONE barrier -- the")
    out("    cross-CM argmin reduction. The most favourable split in the whole set:")
    for sync in ("shared_spad", "tightly_cpl", "coherent"):
        r = split_solve(FCS, C100, 2, sync, 1)
        par = cpu_fastest(FCS)["latency_us"] / r["split_us"]
        out(f"      2 CM, {sync:<12} {r['split_us']:.3f} us vs {r['single_us']:.3f} us "
            f"single -> {r['speedup']:.2f}x datapath, "
            f"{par:.2f}x vs the fastest CPU "
            f"({'ABOVE' if par >= 1.0 else 'below'} parity)")
    out("    So yes: candidate splitting does convert 0.62x into ~1.2x parity. It is")
    out("    still the wrong purchase, for three reasons that must be given together:")
    out()
    out(f"    (a) NO DEADLINE NEEDS IT. FCS-MPC already meets its 25 us period at")
    out(f"        {FCS.deadline_us / wc(FCS, C100)['latency_us']:.1f}x margin. The latency we would be buying is")
    out("        latency the customer does not have a use for; the socket is sold on")
    out("        energy (22.0x perf/W) and core offload.")
    out("    (b) IT IS THE WRONG BOTTLENECK. FCS-MPC is the one solver row where the")
    out("        VECTOR UNIT binds first, not scratchpad bandwidth -- it needs 16 lanes")
    out("        at every bandwidth in the sweep above. So the resource in short supply")
    out("        is vector lanes, and there are two cheaper ways to buy them:")
    out("          MACIF broadcast_init + square_accumulate: 525 -> 154 vector cycles,")
    out("            0.62x -> 0.88x, NEAR-ZERO AREA (already the settled answer)")
    out("          16 -> 32 lanes in one CM: +0.143 mm2 for 1.26x")
    out("          a second CM:              +0.786 mm2 for ~1.95x datapath")
    out("        Ranked by area per unit of speedup, the second CM is LAST of three.")
    out("    (c) IT DOUBLES STATIC POWER ON A ROW SOLD ON ENERGY. Two CM-100 instances")
    out("        leak twice as much for the same solve. Spending the energy claim to")
    out("        buy latency nobody asked for is the wrong trade on this row")
    out("        specifically, because energy IS the pitch here.")
    out("    NOTE the shared_spad row carries the same caveat as Q1: it charges 1/n")
    out("    datapath time behind one scratchpad. That is defensible HERE and not in")
    out("    Q1, because FCS-MPC is vector-bound rather than bandwidth-bound -- the")
    out("    replicated resource really is the scarce one. Stating why the same row is")
    out("    admissible on one benchmark and not the other is the whole point of")
    out("    tracking the binding constraint per row.")
    out()
    out("Q4. MANY INDEPENDENT LOOPS (a manipulator's joints, several estimators).")
    out("    This is the one case with a real argument, and it is NOT a parallelism")
    out("    argument -- it is capacity plus determinism isolation.")
    out()
    out(f"{'row':<11} {'duty % wc':>10} {'footprint KB':>13} {'loops by duty':>14} "
        f"{'loops by capacity':>18} {'binding':>9}")
    for b in SOLVERS:
        dc = duty_cycle(b, C100)
        by_duty = 1.0 / dc if dc > 0 else float("inf")
        by_cap = C100.scratchpad_kb / FOOTPRINT_KB[b.name]
        out(f"{b.name:<11} {dc * 100:>10.2f} {FOOTPRINT_KB[b.name]:>13.1f} "
            f"{by_duty:>14.1f} {by_cap:>18.2f} "
            f"{('capacity' if by_cap < by_duty else 'duty'):>9}")
    ratios = []
    for b in SOLVERS:
        dc = duty_cycle(b, C100)
        ratios.append((1.0 / dc) / (C100.scratchpad_kb / FOOTPRINT_KB[b.name]))
    out()
    out(f"    CAPACITY BINDS CONSOLIDATION ON 4 OF 5 ROWS, by {min(ratios):.1f}-{max(ratios):.0f}x.")
    out("    QUAD-MPC has 4.3x throughput headroom and 1.38x capacity headroom: one")
    out("    CM-100 can host ONE flagship MPC loop, not four. ARM-MPC does not even")
    out("    fit once. So the reason to instantiate a second CM is that the working")
    out("    sets do not co-reside -- which a bigger scratchpad also solves, at")
    out(f"    0.020 mm2 per 32 KB against {a1:.3f} mm2 for a whole second CM.")
    out("    Note the direction of the error a throughput-only analysis would make:")
    out("    duty cycle says 'consolidate 4-388 loops', capacity says 'consolidate")
    out("    one'. Provisioning a shared accelerator from duty cycle alone would")
    out("    oversell consolidation by up to 47x.")
    out()
    out("    THE SECOND ARGUMENT IS JITTER, AND IT IS THE STRONGER ONE. Determinism")
    out("    is the product, so consider a drive node: FCS-MPC at 40 kHz (25 us")
    out("    period) sharing a CM with QUAD-MPC at 1 kHz.")
    qwc = wc(QUAD, C100)["latency_us"]
    out(f"      QUAD-MPC worst-case solve = {qwc:.1f} us = "
        f"{qwc / 25.0:.1f} periods of the 40 kHz loop.")
    out("      Non-preemptible at SOLVE granularity  -> the fast loop misses "
        f"{math.ceil(qwc / 25.0):.0f} deadlines. FATAL.")
    out("      Preemptible at DESCRIPTOR granularity -> worst-case blocking is the")
    out("      longest single descriptor. Measured longest in the ADMM DAG is a full")
    out("      A-product at ~1600 cycles = 2.0 us = 8.0% of a 25 us period. TOLERABLE.")
    out("    -> DESCRIPTOR-GRANULARITY PREEMPTION, not a second CM, is what makes")
    out("       mixed-rate consolidation safe. It is a sequencer/scoreboard feature")
    out("       (save the wait mask and token state, admit the higher-priority")
    out("       template) and it is nearly free next to a 0.79 mm2 replica.")
    out()
    out("VERDICT ON MULTI-CM")
    out("  * Intra-solve splitting: DEAD IN PRINCIPLE AND IN NUMBERS. Loses to one")
    out("    wider CM on both area and latency, at every barrier cost.")
    out("  * Candidate-batch splitting (FCS-MPC): the ONE case that works -- it does")
    out("    reach ~1.2x CPU parity. Rejected anyway, on three grounds together: no")
    out("    deadline needs it (15.5x margin), the scarce resource is vector lanes")
    out("    and two cheaper ways to buy them exist (MACIF accumulator fusion at")
    out("    ~zero area, then +0.143 mm2 of lanes, then +0.786 mm2 of CM), and it")
    out("    doubles static power on the row whose entire pitch is energy.")
    out("  * Many independent loops: bounded by CAPACITY and JITTER, both of which")
    out("    have cheaper fixes (more KB; descriptor-granularity preemption).")
    out("  * What survives: INDEPENDENT INSTANCES, NO INTERCONNECT, when a licensee")
    out("    wants hard physical isolation between safety domains (ASIL) or between")
    out("    control rates. That is ALREADY SUPPORTED -- it is instantiating the IP")
    out("    twice -- and it needs exactly one thing from us: the runtime API must")
    out("    address multiple devices. Nothing in RTL, no shared scratchpad, no")
    out("    CM-to-CM coherency, no one-sequencer-many-datapaths.")
    out("  * WHAT WOULD CHANGE THIS: a frozen-set row that (a) needs >288 B/cycle of")
    out("    fp32 GEMV to make a deadline, or (b) needs >384 KB resident at >1 kHz.")
    out("    Neither exists today. Re-open the question if one appears; until then")
    out("    'multi-core scaling' should come OFF the Phase 4 list, because keeping")
    out("    it there implies a verification obligation we have no reason to accept.")

    # =====================================================================
    out()
    out("=" * 108)
    out("THE VERIFICATION TENSION — parameters are not free, so classify them")
    out("=" * 108)
    out()
    out("§5 markets 'parameterized (tile size, precisions, SRAM depth, bus type)'.")
    out("§9 flags 'few validated configs, not full matrix'. Both are right, and the")
    out("resolution is that PARAMETERS ARE NOT ALL THE SAME KIND OF PARAMETER.")
    out()
    out("REPLICATIVE — changes a width or a depth, reuses the same control FSMs.")
    out("Verification cost is a re-run of an existing regression at a new elaboration.")
    out("  tile / tile_n        generate-loop over MAC cells; reduction tree is")
    out("                       width-generic. Validated range 8-16, square only.")
    out("  vector_lanes         generate-loop over FMA slices. 8-32.")
    out("  scratchpad_kb        bank depth only. Re-run the memory compiler + timing.")
    out("  bank word bytes      >= 64 bits by design rule. 8-16 B.")
    out("  freq target          constraint-level, not RTL. Re-run timing closure.")
    out()
    out("STRUCTURAL — changes control logic, protocol behaviour, or a declared")
    out("contract. Each value is a SEPARATE verification target with its own")
    out("coverage closure. These are what must be rationed.")
    out("  n_banks              arbiter, address interleave, conflict resolution.")
    out("                       VALIDATE 8 AND 16 ONLY.")
    out("  precision set        MACIF declaration + the runtime's REFUSAL paths. A")
    out("                       macro without fp64, or with bf16 substituted for")
    out("                       fp16, must be refused at codegen. Each removal is a")
    out("                       new negative-test suite, not a smaller design.")
    out("  fp64 present         presence/absence changes the PRODUCT (no interior")
    out("                       point without it), not just the datapath.")
    out("  bus type             THE MOST EXPENSIVE PARAMETER IN §5 AND THE MOST")
    out("                       CASUALLY MARKETED. AXI4 / ACE-Lite / TileLink each")
    out("                       need their own protocol-checker verification and")
    out("                       their own coherency-behaviour proof. This is not a")
    out("                       parameter, it is three deliverables.")
    out("  seq queue depth      scoreboard + token-reuse hazard rule. FIX AT 8.")
    out("  chain block width    follows datapath width automatically. Not a knob.")
    out("  drain / pipe depth   timing-driven, but it changes the scoreboard's")
    out("                       hazard distances. FIX PER SKU, do not expose.")
    out()
    out("RULE: sell the replicative set as parameters. Sell the structural set as")
    out("a SKU list. Never print 'bus type' in a parameter list again.")

    # =====================================================================
    out()
    out("=" * 108)
    out("PROPOSED SKU FAMILY — TWO validated configurations, not three")
    out("=" * 108)
    out()
    SKUS = [
        ("CM-100", cfg(bw=128, tile=12, lanes=16, kb=192, fp64=8),
         "Flagship. Robotics/drone node: condensed-QP MPC + state estimation +"
         " CNN/ViT-class ML + interior-point NMPC. The only SKU that holds the"
         " QUAD-MPC flagship working set resident."),
        ("CM-50", cfg(bw=64, tile=8, lanes=8, kb=64, fp64=4),
         "MCU-class / motor-drive node: FCS-MPC, small EKF, small condensed QP."
         " Sized to the fastest loop it can host RESIDENT, not to the biggest"
         " problem it can eventually finish."),
    ]
    REJECTED = [
        ("CM-100S", cfg(bw=128, tile=4, lanes=16, kb=192, fp64=8),
         "solver-only 16-MAC array"),
        ("CM-200", cfg(bw=256, tile=16, lanes=32, kb=256, fp64=16),
         "wide/fast variant"),
    ]
    out(f"{'SKU':<9} {'MACs':>5} {'B/cyc':>6} {'banks':>6} {'KB':>5} {'lanes':>6} "
        f"{'fp64':>5} {'core mm2':>9} {'pJ/B':>6} {'solver uJ':>10} {'perf/W':>13} "
        f"{'deadlines':>10} {'residency violations':>22}")
    for nm, c, _ in SKUS + REJECTED:
        pws = [perf_w(b, c) for b in SOLVERS]
        misses = [b.name for b in SOLVERS if wc(b, c)["latency_us"] > b.deadline_us]
        ovf = [b.name for b in SOLVERS if FOOTPRINT_KB[b.name] > c.scratchpad_kb]
        a = C.area_breakdown(c)
        out(f"{nm:<9} {c.macs:>5} {c.scratchpad_bw_bpc:>6.0f} {c.n_banks:>6} "
            f"{c.scratchpad_kb:>5} {c.vector_lanes:>6} {c.fp64_fmas:>5} "
            f"{a['total']:>9.3f} {c.sram_pj_per_byte():>6.3f} {solver_energy(c):>10.2f} "
            f"{min(pws):>5.1f}-{max(pws):<6.1f} "
            f"{('all met' if not misses else ','.join(misses)):>10} "
            f"{(','.join(ovf) if ovf else 'none'):>22}")
    out()
    out("  ** READ THE LAST COLUMN, NOT THE DEADLINE COLUMN. ** Every SKU here 'meets")
    out("  all deadlines', which is close to meaningless: latency margin is 4-15x")
    out("  everywhere. What separates the SKUs is which working sets stay RESIDENT,")
    out("  because a residency violation puts DRAM in the control loop and forfeits")
    out("  the determinism claim -- the thing we actually sell. A row that overflows")
    out("  still MEETS its deadline in this model (the streaming is small and")
    out("  double-buffered); it just stops being deterministic.")
    out()
    for nm, c, why in SKUS:
        out(f"  {nm}: {why}")
        fits = [b.name for b in SOLVERS if FOOTPRINT_KB[b.name] <= c.scratchpad_kb]
        ovf = [b.name for b in SOLVERS if FOOTPRINT_KB[b.name] > c.scratchpad_kb]
        out(f"    holds resident: {', '.join(fits)}")
        out(f"    DOES NOT hold:  {', '.join(ovf) if ovf else '-- (all frozen rows fit)'}")
        tight = min(SOLVERS, key=lambda b: b.deadline_us / wc(b, c)["latency_us"])
        out(f"    tightest worst-case deadline margin: {tight.name} at "
            f"{tight.deadline_us / wc(tight, c)['latency_us']:.2f}x "
            f"({tight.deadline_us * C.PERIOD_BUDGET / wc(tight, c)['latency_us']:.2f}x "
            f"against a 50%-of-period budget)")
        out(f"    array/bandwidth balance: needs tile >= "
            f"{min_tile_for_bw(c.scratchpad_bw_bpc):.1f}, has {c.tile} "
            f"({(c.tile / min_tile_for_bw(c.scratchpad_bw_bpc)) ** 2:.2f}x MAC headroom)")
        out()
    out("  SCOPING LINE THAT MUST GO IN THE DATASHEET: CM-50 does NOT support the")
    out("  QUAD-MPC class. Its 64 KB holds 0.46 of that 138.8 KB working set, so a")
    out("  12-state N=20 condensed QP would stream ~75 KB every millisecond tick. It")
    out("  still meets the deadline; it is no longer deterministic. CM-50's honest")
    out("  envelope is n_var <= ~40 (ADMM), n <= ~48 (EKF), and finite-control-set MPC")
    out("  of any size, because FCS-MPC's whole working set is 0.7 KB.")
    out()
    out("  AND ONE PIECE OF GOOD NEWS THAT SHOULD BE STATED PLAINLY, because the fp64")
    out("  decision has been treated as a cost all along: in the merged multi-format")
    out("  cluster, fp64 THROUGHPUT UP TO lanes/2 IS FREE. The area is set by")
    out("  max(fp64_fmas, lanes/2) units, so at 16 lanes an fp64 count of 1, 2, 4 or 8")
    out("  all cost the same 0.786 mm2. What you pay for is the multi-format MERGE")
    out("  premium (+0.067 mm2 at 16 lanes over an fp32-only unit of the same width),")
    out("  once, not per fp64 unit. Interior-point capability is therefore a")
    out("  ~9%-of-core decision taken once, not a scaling axis with a running cost.")
    out()
    out("  ARM-MPC OVERFLOWING CM-100 IS DELIBERATE, NOT A GAP. It is 323.2 KB at")
    out("  fp64, runs at 250 Hz with 5.5x worst-case margin, and streams 0.034 GB/s.")
    out("  Axis 3 shows that growing to 384 KB to hold it costs +0.228 mm2 and MORE")
    out("  energy. Determinism is only load-bearing on the fast loops; a 250 Hz")
    out("  interior-point solve is allowed to touch DRAM and we should say so rather")
    out("  than size a scratchpad to hide it.")
    out()
    out("  WHY NOT A THIRD SKU:")
    c16 = REJECTED[0][1]
    out(f"    CM-100S (16-MAC solver-only) saves "
        f"{C.area_breakdown(SKUS[0][1])['total'] - C.area_breakdown(c16)['total']:.3f} mm2 "
        f"({(1 - C.area_breakdown(c16)['total'] / C.area_breakdown(SKUS[0][1])['total']) * 100:.0f}%) "
        f"and costs")
    e_ekf_100 = M.model_argmind(next(b for b in SOLVERS if b.name == "EKF-96"), SKUS[0][1])["energy_uj"]
    e_ekf_16 = M.model_argmind(next(b for b in SOLVERS if b.name == "EKF-96"), c16)["energy_uj"]
    out(f"    {e_ekf_16 / e_ekf_100:.2f}x the EKF-96 energy. EKF is a BEACHHEAD workload, so a")
    out("    SKU that is bad at state estimation is a SKU for a socket we are not")
    out("    selling into. The array is the cheapest thing in the part")
    out(f"    ({C.area_breakdown(SKUS[0][1])['mac'] / C.area_breakdown(SKUS[0][1])['total'] * 100:.0f}% of core). "
        f"Do not spend a verification slot to remove it.")
    out("    -> tile stays a REPLICATIVE synthesis parameter over 8-16, unsupported")
    out("       outside that range. Not a SKU.")
    out()
    out(f"    CM-200 (256 B/cyc, 16x16, 256 KB) costs "
        f"{C.area_breakdown(REJECTED[1][1])['total'] / C.area_breakdown(SKUS[0][1])['total']:.2f}x the core for")
    out(f"    {wc(QUAD, SKUS[0][1])['latency_us'] / wc(QUAD, REJECTED[1][1])['latency_us']:.2f}x "
        f"QUAD-MPC worst case and "
        f"{solver_energy(SKUS[0][1]) / solver_energy(REJECTED[1][1]):.2f}x solver energy, against")
    out("    deadlines that already have 4-14x margin. It is a real Pareto point and")
    out("    it has NO WORKLOAD IN THE FROZEN SET. Keep it as a documented synthesis")
    out("    configuration for a customer who brings a harder problem; do not")
    out("    verify it, do not price it, do not put it on a roadmap.")
    out()
    out("  THE ONE AXIS TO GROW ON DEMAND, and it is not the one Phase 4 lists:")
    out("  the fp64 FMA count. ARM-MPC is the only row bound by ARITHMETIC rather")
    out("  than bandwidth, and it scales almost linearly:")
    out(f"{'fp64 FMAs':>10} {'core mm2':>9} {'ARM wc us':>10} {'ARM uJ':>8} {'margin':>8} "
        f"{'uJ/mm2':>8}")
    prev = None
    for n in (2, 4, 8, 16, 32):
        c = cfg(fp64=n, lanes=max(16, 2 * n))
        a = C.area_breakdown(c)
        w = wc(ARM, c)["latency_us"]
        e = M.model_argmind(ARM, c)["energy_uj"]
        marg = ""
        if prev and a["total"] > prev[0] + 1e-9:
            marg = f"{(prev[1] - e) / (a['total'] - prev[0]):>8.0f}"
        out(f"{n:>10} {a['total']:>9.3f} {w:>10.1f} {e:>8.2f} "
            f"{ARM.deadline_us / w:>7.2f}x {marg:>8}")
        prev = (a["total"], e)
    out("  (blank uJ/mm2 cells = zero area delta: 2, 4 and 8 fp64 FMAs are the SAME")
    out("   silicon, because the merged cluster is sized by max(fp64, lanes/2) = 8.")
    out("   fp64 throughput below lanes/2 is free, as noted above -- so the 2- and")
    out("   4-FMA rows are not cheaper configurations, they are the same part running")
    out("   a narrower kernel. There is no reason to ship them.)")
    out("  8 -> 16 FMAs is 1.85x on ARM-MPC latency for +0.144 mm2. But ARM-MPC")
    out("  already has 5.5x deadline margin and the ENERGY return is ~18 uJ/mm2 --")
    out("  worse than tile 8->12's 84 uJ/mm2. So: right axis, no customer yet.")
    out("  Grow it if and only if a licensee brings a longer horizon, a faster rate,")
    out("  or a genuinely nonlinear MPC that needs more interior-point iterations.")

    # =====================================================================
    out()
    out("=" * 108)
    out("CUSTOMER-FACING ENVELOPE RULE — 'is my workload in the CM's band?'")
    out("=" * 108)
    out()
    out("Five tests. All must pass. Each uses a number the customer already has.")
    out()
    out("1. RESIDENCY. Sum every matrix as a DENSE RECTANGLE at your working")
    out("   precision, plus every working vector and temporary. Not packed")
    out("   triangular -- the address generators cannot express it.")
    out("     <= 192 KB (CM-100) or <= 64 KB (CM-50)  -> in.")
    out("   Over that, DRAM enters your control loop and the determinism claim is")
    out("   void. This is a hard edge, not a gradient.")
    out()
    out("2. LATENCY FLOOR. Measure your solve on your host core. Then:")
    out("     host time >= 5 us     -> the latency claim holds (T1, rich host)")
    out("     host time >= 1.7 us   -> holds on an MCU-class doorbell (T1M)")
    out("     host time <  1.5 us   -> USE THE CPU. No datapath size fixes this;")
    out("                              the launch protocol is the cost.")
    out("   (Derivation: cpu >= overhead*k/(k-1) with k = 1.2-2.2 measured.)")
    out()
    out("3. ENERGY FLOOR, which is LOWER than the latency floor -- most workloads")
    out("   that fail test 2 still pass this one:")
    out("     host time >= 0.4 us of an efficient in-order core -> >=5x perf/W")
    out("   Below that, the CM's static power during launch exceeds your whole")
    out("   compute energy. Requires power gating, which is mandatory anyway.")
    out()
    out("4. DESCRIPTOR DENSITY. Average >= 8 datapath cycles per descriptor, with")
    out("   no burst of more than 8 consecutive short descriptors. In practice:")
    out("   your inner-loop matrix/vector operations must average >= ~64 elements")
    out("   at 128 B/cycle fp32. Frozen-set rows run 238 cycles/descriptor (30x the")
    out("   bar); one-step FCS-MPC and 100 kHz two-level FCS-MPC FAIL it. Tell those")
    out("   customers to use the MCU.")
    out()
    out("5. OFF-CHIP TRAFFIC. (bytes that must cross the boundary per call) /")
    out("   (deadline) <= 6.4 GB/s at the baseline LPDDR4x provisioning. If your")
    out("   answer is a weight-streaming model, this is the ONLY test that matters")
    out("   and no on-chip parameter changes it.")
    out()
    out("AND THE REUSE-DEPTH DIAGNOSTIC, which tells you WHICH knob is yours:")
    out("  Take your dominant product. How many times is each fetched operand")
    out("  element consumed before it is discarded? Call it R.")
    out("    R = 1        pure GEMV. Bandwidth-bound. MAC count is irrelevant to you.")
    out("    R < 8        still bandwidth-bound even with a batch dimension")
    out("                 (FCS-MPC has R = n_y = 2 across 729 candidates and is")
    out("                  bandwidth-bound; the batch buys 1.7x, not 12x).")
    out("    R >= tile    tile-limited GEMM. This is the only regime where a bigger")
    out("                 array pays, and even then it pays in ENERGY first.")
    out("  If R is set by a PROBLEM dimension rather than by the tile, you are")
    out("  operand-fetch-bound. That is the general form of the batch-1 finding and")
    out("  it is the single most useful thing to tell a customer architect.")

    # =====================================================================
    out()
    out("=" * 108)
    out("SUMMARY — the band, stated plainly")
    out("=" * 108)
    out()
    out("  axis                  scales?  usable range              wall, and what sets it")
    out("  --------------------  -------  ------------------------  ----------------------")
    out("  MAC array             weakly   36 - 256 MACs             energy return dies at 144;")
    out("                                                           ML latency wants 1024 but")
    out("                                                           then lanes bind")
    out("  scratchpad bandwidth  YES      64 - 256 B/cycle          16 banks x 16 B; and the")
    out("                                                           fp32 GEMV ceiling crosses")
    out("                                                           a 144-MAC array at 288")
    out("  capacity              NO       a discrete fit, 64/192KB  monotone energy cost; only")
    out("                                                           ever tracks the fastest")
    out("                                                           resident working set")
    out("  frequency             weakly   600 - 1000 MHz            bitline folds at ~1100;")
    out("                                                           V^2 destroys the energy")
    out("                                                           claim above nominal")
    out("  process node          NO       cost axis only            the CPU baseline moves too")
    out("  fp64 FMA count        YES      4 - 16                    linear on interior point,")
    out("                                                           and no customer needs it")
    out("  sequencer             NO       fixed                     97-100% is dependency")
    out("                                                           drain; only shrinks")
    out("  multi-CM              NO       independent instances     loop-carried iteration;")
    out("                                                           loses to one wider CM")
    out("  external bandwidth    n/a      a provisioning input      >=25 GB/s for 1.1B decode,")
    out("                                                           and undifferentiated there")
    out()
    out("  TWO axes scale usefully: scratchpad bandwidth (64 -> 256 B/cycle, ~4x) and")
    out("  the fp64 FMA cluster (4 -> 16, ~4x, on one benchmark that does not need it).")
    out("  Everything else is a fit, a cost, or a distraction. The honest product")
    out("  statement is a NARROW BAND -- roughly 3-4x of usable configuration span --")
    out("  and that is a legitimate position for control-domain IP, where the")
    out("  customer's problem sizes are set by physics and actuator counts rather")
    out("  than by a scaling roadmap. Two validated SKUs cover it.")
    out()
    out("  WHAT WOULD FALSIFY THIS: an L0 operand register file. It would buy the")
    out("  array's reuse at O(T) area instead of O(T^2), which drops the honest array")
    out("  size to 36-64 MACs AND raises the bandwidth at which the array starts to")
    out("  bind. That is the highest-value model extension outstanding and it moves")
    out("  two of the walls above.")

    with open("perf_model/results_scaling_study.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
