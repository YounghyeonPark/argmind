"""
Argmind CM — scratchpad banking cost model (area + energy vs bandwidth)
=======================================================================

Purpose: cost the bandwidth-first decision. `argmind_model.py` shows solver
latency scaling with `scratchpad_bw_bpc` while MAC count barely matters. That
decision was never costed. This module costs it.

Everything here is a MODELLED ESTIMATE at a 16nm-class node, +/-2x, NOT
QUOTABLE EXTERNALLY. Anchors that are published are marked [CITED]; anything
else is marked [EST] and is a judgement call that Phase 1 synthesis + a real
foundry memory compiler must replace.

PUBLISHED ANCHORS
-----------------
[A1] 16nm HD 6T SRAM bitcell = 0.070 um^2.
     Chang et al., "A 16nm 128Mb SRAM in high-k metal-gate FinFET technology
     with write-assist circuitry for low-VMIN applications", ISSCC 2015.
     https://www.researchgate.net/publication/280217709
     Cross-checked: https://en.wikichip.org/wiki/16_nm_lithography_process

[A2] SRAM access energy vs MACRO SIZE and ACCESS WIDTH, 45nm, pJ per 16b word.
     TABLE II of Schneider et al., "A Construction Kit for Efficient Low Power
     Neural Network Accelerator Designs", arXiv:2106.12810.
     https://arxiv.org/pdf/2106.12810
     This is the key table: it shows per-bit energy FALLING with smaller macros
     and WIDER access, which is the opposite of the naive "more banks = more
     periphery = more energy" intuition.

[A3] On-chip wire energy ~0.1 pJ/bit/mm at 45nm (50% transition probability).
     Derived from Keckler, Dally et al., "GPUs and the Future of Parallel
     Computing", IEEE Micro 2011: 256 bits of operand from an 8KB SRAM is
     ~56 pJ, and 310 pJ if that memory is 10 mm away
     -> (310-56)/(256 bits * 10 mm) = 0.099 pJ/bit/mm.
     https://www.cs.toronto.edu/~pekhimenko/courses/csc2224-f19/docs/GPU.pdf
     Same source independently corroborates [A2]: 56 pJ / 4 accesses / 8 B
     = 1.75 pJ/byte for an 8KB 64b-word macro at 45nm, vs [A2]'s 1.60. Two
     independent sources within 10% — the 45nm baseline is solid; the 45->16nm
     scaling is where the uncertainty lives.

[A4] MEASURED 16nm finFET 6T single-port macro density = 8.1 Mb/mm^2, and an 8T
     dual-port macro of the same generation is ~2.1x the area ("53% area
     savings", ~60% leakage savings for the 6T-SP-based design).
     "An ultra high density pseudo dual-port SRAM in 16nm FINFET process for
     graphics processors", IEEE 2017. https://ieeexplore.ieee.org/document/8225996/
     THIS IS THE CALIBRATION ANCHOR for the area model below: 8.1 / 14.3
     (bitcell-limited) = 57% array efficiency for a real, fast 16nm macro.
     The peripheral geometry estimates are tuned to reproduce it. It also
     prices multi-porting against banking directly.

[A5] Large hero macros reach ~80% of bitcell-limited density (TSMC N2:
     38.1 Mb/mm^2 macro vs 0.021 um^2 cell = 47.6 Mb/mm^2 bitcell-limited),
     partly by extending bitlines to 512 rows.
     https://spectrum.ieee.org/sram-intel-tsmc
     We cannot use 512-row bitlines: see ROWS_MAX_800MHZ.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

A_CELL_HD_UM2 = 0.070        # [A1] 16nm high-density 6T
A_CELL_HC_UM2 = 0.105        # [EST] high-current/perf cell, ~1.5x HD
CELL_ASPECT = 2.5            # [EST] thin-cell width:height along WL:BL

# Peripheral geometry. These four numbers carry most of the area uncertainty.
# They are CALIBRATED so that a mid-size fast macro reproduces [A4]'s measured
# 8.1 Mb/mm^2 (57% array efficiency) at 16nm. See calibration check in main().
W_ROWDEC_UM = 16.0           # [EST-CAL] row decoder + WL drivers + control column
H_COL_BASE_UM = 22.0         # [EST-CAL] precharge + col mux + IO + redundancy strip
H_COL_PER_SA_UM = 32.0       # [EST-CAL] sense-amp/write-driver height, /mux ratio
H_FOLD_MUX_UM = 8.0          # [EST] fold select mux + local data bus per macro
A_CTRL_UM2 = 400.0           # [EST] per-macro timing, redundancy, MBIST wrapper
HALO_UM = 2.0                # [EST] per-side macro keep-out a licensee must honour

# 800 MHz single-cycle read at 16nm/0.8V with an HD cell: bitline must stay
# short. Foundry compilers will generate 512-row arrays but not at this speed
# with margin for a licensee's own timing closure. [EST]
ROWS_MAX_800MHZ = 256
ASPECT_MAX = 8.0             # [EST] placeable macro aspect ratio limit


def rows_max_for_freq(freq_mhz: float) -> int:
    """Longest single-cycle-read bitline at `freq_mhz`. [EST]

    ADDED 2026-07-31 for the scaling analysis, which needed the FREQUENCY axis
    to have a cost and previously found it free. ROWS_MAX_800MHZ was a single
    hard constant, so the model reported frequency as a pure 1/f latency win.

    Physical basis: a 6T read discharges the bitline through one access
    transistor at roughly constant cell current, so the discharge time is set by
    the bitline capacitance, which is linear in the number of attached cells.
    Hence rows_max ~ 1/f. This is the CENTRAL assumption; the honest band is
    1/f (charge-limited, used here) to 1/sqrt(f) (if the compiler also trades
    cell current or adds hierarchical bitlines). The sensitivity is reported.

    At 800 MHz this returns exactly ROWS_MAX_800MHZ, so nothing pre-existing
    moves. Clamped to [32, 512]: below 32 rows the macro is all periphery, and
    no 16nm compiler in the survey offers a single-cycle 1024-row array.
    """
    return int(max(32, min(512, ROWS_MAX_800MHZ * 800.0 / freq_mhz)))

# Energy scaling 45nm -> 16nm. C scales with physical length
# (sqrt(0.070/0.25) = 0.53); V^2 scales 1.1V -> 0.8V (0.53).
# Product 0.28; use 0.30 central. Range 0.22-0.45 is the honest band. [EST]
E_SCALE_45_TO_16 = 0.30
WIRE_PJ_PER_BIT_MM_45 = 0.099            # [A3]
WIRE_PJ_PER_BIT_MM_16 = 0.099 * 0.50     # [EST] wire cap/um ~flat, V^2 gives 0.53
E_MUX_FLOP_PJ_PER_BYTE = 0.06            # [EST] read crossbar + pipeline staging

SRAM_LEAK_UW_PER_KB_85C = 2.4            # [EST] 16nm HD array, 0.8V, 85C
PERIPH_LEAK_FRACTION = 0.35              # [EST] periphery leak as frac of array

# Logic density for the MAC array, for the area comparison.
A_GATE_UM2 = 0.26            # [EST] NAND2-equiv incl. 70% placement utilisation
GATES_MAC_BF16_FP32ACC = 3000            # [EST] bf16 mult + fp32 accumulate cell
GATES_MAC_INT8 = 700                     # [EST]
MAC_STAGING_OVERHEAD = 1.20              # [EST] operand skew regs, accum RF

# ---------------------------------------------------------------------------
# IEEE-754 FMA gate model, parametric in significand width
# ---------------------------------------------------------------------------
# Added 2026-07-30 to cost the fp64 datapath, which was ZERO in the CM-100 area
# budget while ARM-MPC (acados SQP-RTI + HPIPM, interior point) requires fp64.
#
# Structure of the estimate, so the reader can attack each term separately:
#   * multiplier   radix-4 Booth + Wallace/Dadda tree + final CPA. Partial-product
#                  count is n/2, each ~n bits wide, reduced by full adders
#                  (~6 NAND2-equiv each): ~3n^2, plus Booth encode/select ~1.25n^2,
#                  plus a 2n-bit fast CPA. => K_MUL ~ 4.5 gates per bit^2.
#   * linear terms alignment barrel shifter over ~3n, LZA + normalise shifter,
#                  3n-wide adder, rounding. All scale ~n log n; folded into a
#                  linear coefficient over the n range of interest (8..53).
#   * fixed terms  exponent datapath, sign, flags, IEEE exception logic.
#
# CALIBRATION (this is the load-bearing part, so state it):
#   n=24 (fp32) -> 9.55 kgates. n=53 (fp64) -> 26.6 kgates. Ratio 2.8x.
#   Anchors:
#     - FPnew / CVFPU (Mach, Schuiki, Zaruba, Benini, IEEE TVLSI 2021, GF 22FDX)
#       is the existence proof that a merged FP64/FP32/FP16/bfloat16 FMA slice
#       fits in an embedded RISC-V core's area budget.
#       https://arxiv.org/abs/2007.01530
#     - A 28nm dual-precision FMA (1x binary64 or 2x binary32, 3-stage, 660 MHz)
#       is reported at 56.2 kgates -- Microprocess. Microsyst. 2018,
#       https://www.sciencedirect.com/science/article/abs/pii/S0141933117305665
#       That is a HIGH-PERFORMANCE merged unit and is used here as the pessimistic
#       edge, not the central value.
#   Sanity check at the low end: the model returns 3.4 kgates for n=8, against the
#   independently-guessed GATES_MAC_BF16_FP32ACC = 3000 for the array cell. Two
#   estimates made different ways agreeing to 14% is weak but real corroboration.
#
# The honest band on an fp64 FMA is therefore 26.6 - 56.2 kgates, i.e. the fp64
# area figures carry their own +/-2x on top of everything else. [EST-CAL]
K_FMA_MUL_GATES_PER_BIT2 = 4.5
K_FMA_LIN_GATES_PER_BIT = 240.0
K_FMA_FIX_GATES = 1200.0
# 56.2 / 26.6: the measured high-performance merged unit over the parametric
# central value. Applied as the pessimistic edge of the area band.
FMA_AREA_PESSIMISM = 2.1
# A slice that can do 1x fp64 OR 2x fp32 OR 4x bf16 needs segmentation muxes in
# the reduction tree and per-segment rounding. Published multi-format units land
# ~1.2-1.4x a single-format unit of the widest format, and well below the sum of
# separate units. [EST]
FMA_MULTIFORMAT_MERGE = 1.30
# Cluster-level: 8-byte operand staging (3 operands), accumulator register file,
# scratchpad read-port share, descriptor decode. On top of MAC_STAGING_OVERHEAD.
FP64_CLUSTER_OVERHEAD = 1.35              # [EST]

SIG_BITS = {"int8": 8, "bf16": 8, "fp16": 11, "fp32": 24, "fp64": 53}


def fma_gates_ieee(sig_bits: int, pessimistic: bool = False,
                   multiformat: bool = False) -> float:
    """NAND2-equivalent gate count for one IEEE-754 FMA of the given significand.

    `multiformat` = the slice can also be segmented into narrower formats.
    `pessimistic` = the measured-28nm-high-performance edge of the band.
    """
    g = (K_FMA_MUL_GATES_PER_BIT2 * sig_bits ** 2
         + K_FMA_LIN_GATES_PER_BIT * sig_bits
         + K_FMA_FIX_GATES)
    if multiformat:
        g *= FMA_MULTIFORMAT_MERGE
    if pessimistic:
        g *= FMA_AREA_PESSIMISM
    return g


def fma_cluster_mm2(n_fma: int, prec: str = "fp64", pessimistic: bool = False,
                    multiformat: bool = False) -> float:
    """Area of a standalone FMA cluster of `n_fma` units at 16nm-class.

    This is the option-(b) cost: a narrow cluster sized for the interior-point
    inner loop only, NOT native fp64 widened through the whole MAC array.
    """
    return (n_fma * fma_gates_ieee(SIG_BITS[prec], pessimistic, multiformat)
            * A_GATE_UM2 * MAC_STAGING_OVERHEAD * FP64_CLUSTER_OVERHEAD / 1e6)


def native_fp64_array_mm2(n_macs: int, pessimistic: bool = False) -> float:
    """Option (c): every array cell natively fp64 (merged multi-format).

    No FP64_CLUSTER_OVERHEAD here -- the array already has its operand staging
    counted in MAC_STAGING_OVERHEAD, same basis as mac_array_mm2.
    """
    return (n_macs * fma_gates_ieee(SIG_BITS["fp64"], pessimistic, multiformat=True)
            * A_GATE_UM2 * MAC_STAGING_OVERHEAD / 1e6)


# Multi-pass (option a) needs an EXACT wide accumulator per output element, or
# the error-free-splitting property that makes the scheme work is lost. 2*53 bits
# of product plus guard bits for the summation length.
WIDE_ACCUM_BITS = 128                     # [EST] 106 product + 22 guard
GATES_PER_ACCUM_BIT = 11.0                # [EST] flop (~5) + adder slice (~6)
BASE_ACCUM_BITS = 32                      # fp32 accumulator already in the cell


def wide_accum_delta_mm2(n_accum: int, bits: int = WIDE_ACCUM_BITS) -> float:
    """Extra area to widen `n_accum` array accumulators for exact multi-pass fp64.

    This is the hidden cost of option (a): it buys no multipliers, but every
    participating output element needs an exact fixed-point accumulator.
    """
    extra = max(0, bits - BASE_ACCUM_BITS)
    return n_accum * extra * GATES_PER_ACCUM_BIT * A_GATE_UM2 / 1e6


# ---------------------------------------------------------------------------
# [A2] 45nm SRAM access energy, pJ per 16-bit word
# ---------------------------------------------------------------------------
_A2_SIZES_KB = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
_A2_WIDTHS_B = [64, 128, 256, 512]
_A2 = {
    1:    [1.20, 0.93, 0.69, 0.57],
    2:    [1.54, 1.37, 0.91, 0.68],
    4:    [2.11, 1.68, 1.34, 0.90],
    8:    [3.19, 2.71, 2.21, 1.33],
    16:   [4.36, 3.57, 2.66, 2.19],
    32:   [5.82, 4.80, 3.52, 2.64],
    64:   [8.10, 7.51, 5.79, 4.67],
    128:  [11.66, 11.50, 8.46, 6.15],
    256:  [15.60, 15.51, 13.09, 8.99],
    512:  [23.37, 23.24, 17.93, 15.76],
    1024: [36.32, 32.81, 28.88, 25.22],
}


def _loglin(x, x0, x1, y0, y1):
    if x1 == x0:
        return y0
    t = (math.log(x) - math.log(x0)) / (math.log(x1) - math.log(x0))
    return math.exp(math.log(y0) + t * (math.log(y1) - math.log(y0)))


MIN_SUPPORTED_BANK_WORD_BITS = _A2_WIDTHS_B[0]     # 64


def macro_energy_pj_per_byte_45(size_kb: float, width_bits: float) -> float:
    """Bilinear (log-log) interpolation of [A2]; returns pJ/byte at 45nm.

    HARD VALIDITY LIMIT: [A2] has no data below a 64-bit access width. Below
    that, per-access fixed costs (row decode, wordline drive, timing, precharge
    of the whole accessed row) stop being amortised and pJ/byte must turn back
    UP -- but this function CLAMPS instead, so it keeps predicting cheaper
    bytes. Any configuration with a bank word narrower than 64 bits is
    EXTRAPOLATION and its energy number must not be believed. This is the
    reason the recommended bank count is capped at bw_bpc/8.
    """
    size_kb = min(max(size_kb, _A2_SIZES_KB[0]), _A2_SIZES_KB[-1])
    width_bits = min(max(width_bits, _A2_WIDTHS_B[0]), _A2_WIDTHS_B[-1])

    def at_size(sz_lo, sz_hi):
        row_lo, row_hi = _A2[sz_lo], _A2[sz_hi]
        # interpolate over width first
        def w(row):
            for i in range(len(_A2_WIDTHS_B) - 1):
                a, b = _A2_WIDTHS_B[i], _A2_WIDTHS_B[i + 1]
                if a <= width_bits <= b:
                    return _loglin(width_bits, a, b, row[i], row[i + 1])
            return row[-1]
        return w(row_lo), w(row_hi)

    for i in range(len(_A2_SIZES_KB) - 1):
        lo, hi = _A2_SIZES_KB[i], _A2_SIZES_KB[i + 1]
        if lo <= size_kb <= hi:
            y0, y1 = at_size(lo, hi)
            e_per_16b = _loglin(size_kb, lo, hi, y0, y1)
            return e_per_16b / 2.0      # 16b word -> per byte
    raise ValueError(size_kb)


# ---------------------------------------------------------------------------
# Bank / macro geometry
# ---------------------------------------------------------------------------

@dataclass
class BankGeom:
    size_kb: float
    word_bits: int
    mux: int
    folds: int
    rows: int          # rows per fold (= bitline length in cells)
    cols: int          # columns per fold
    w_um: float
    h_um: float
    area_um2: float
    cell_um2: float = A_CELL_HD_UM2

    @property
    def bitcell_um2(self) -> float:
        return self.size_kb * 1024 * 8 * self.cell_um2

    @property
    def array_eff(self) -> float:
        return self.bitcell_um2 / self.area_um2


def bank_geometry(size_kb: float, word_bits: int, cell_um2: float = A_CELL_HD_UM2,
                  mux: int | None = None, rows_max: int | None = None) -> BankGeom:
    """Minimum-area legal macro geometry.

    Two internal degrees of freedom, both of which real compilers expose:
      * column mux ratio M  — trades sense-amp count against column count
      * vertical folds F    — splits the array into F stacked sub-arrays so
                              that bitlines stay short and the aspect ratio
                              stays placeable. Each fold pays its own column
                              periphery; the row decoder is shared.
    Constraints that bite: rows-per-fold <= ROWS_MAX_800MHZ (timing) and macro
    aspect ratio <= ASPECT_MAX (placeability in a licensee's floorplan).
    """
    h_cell = math.sqrt(cell_um2 / CELL_ASPECT)
    w_cell = cell_um2 / h_cell
    bits = size_kb * 1024 * 8
    r_max = ROWS_MAX_800MHZ if rows_max is None else rows_max

    best = None
    muxes = [mux] if mux else [1, 2, 4, 8, 16, 32]
    for m in muxes:
        cols = word_bits * m
        for f in (1, 2, 4, 8):
            rows_f = bits / (cols * f)
            if rows_f < 16 or abs(rows_f - round(rows_f)) > 1e-9:
                continue
            rows_f = int(round(rows_f))
            if rows_f > r_max:
                continue
            h_col = H_COL_BASE_UM + H_COL_PER_SA_UM / m
            w = cols * w_cell + W_ROWDEC_UM
            h = f * (rows_f * h_cell + h_col) + (H_FOLD_MUX_UM if f > 1 else 0.0)
            if max(w / h, h / w) > ASPECT_MAX:
                continue
            area = w * h + A_CTRL_UM2
            cand = BankGeom(size_kb, word_bits, m, f, rows_f, cols,
                            w, h, area, cell_um2)
            if best is None or area < best.area_um2:
                best = cand
    if best is None:
        raise ValueError(f"no legal geometry for {size_kb}KB x {word_bits}b "
                         f"under {r_max}-row / AR{ASPECT_MAX} limits")
    return best


# ---------------------------------------------------------------------------
# Whole-scratchpad configuration
# ---------------------------------------------------------------------------

@dataclass
class SpadCfg:
    total_kb: float
    n_banks: int
    word_bytes: int            # bytes delivered per bank per cycle
    cell_um2: float = A_CELL_HD_UM2
    route_over_macro: bool = True   # can the licensee route the read bus over
                                    # the SRAM macros (M6+ free)?
    # Target clock. Sets the longest legal bitline via rows_max_for_freq. Default
    # 800 MHz reproduces every pre-existing number exactly.
    freq_mhz: float = 800.0

    @property
    def bw_bpc(self) -> float:
        return self.n_banks * self.word_bytes

    @property
    def bank_kb(self) -> float:
        return self.total_kb / self.n_banks

    @property
    def geom(self) -> BankGeom:
        return bank_geometry(self.bank_kb, self.word_bytes * 8, self.cell_um2,
                             rows_max=rows_max_for_freq(self.freq_mhz))

    # ---- area ----------------------------------------------------------
    def area_mm2(self) -> dict:
        g = self.geom
        placed = (g.w_um + 2 * HALO_UM) * (g.h_um + 2 * HALO_UM) + A_CTRL_UM2
        macros = self.n_banks * placed

        # Read-bus routing channel. SRAM macros at 16nm typically block up to
        # ~M4; if the licensee's flow allows routing the read bus on M6+ over
        # the macros, the channel cost mostly vanishes. If not, you pay for it.
        side = math.sqrt(macros)
        bus_bits = self.bw_bpc * 8 * 1.35   # + address/control/ECC overhead [EST]
        track_pitch_um = 0.16               # [EST] M5-M7 pitch at 16nm
        usable_layers = 2
        channel_w = bus_bits * track_pitch_um / usable_layers
        chan = 0.0 if self.route_over_macro else channel_w * side

        # Bank arbiter + address decode + per-bank request queues [EST]
        ctrl = self.n_banks * 2500.0

        total = macros + chan + ctrl
        return {
            "macro_mm2": macros / 1e6,
            "channel_mm2": chan / 1e6,
            "ctrl_mm2": ctrl / 1e6,
            "total_mm2": total / 1e6,
            "bitcell_mm2": self.total_kb * 1024 * 8 * self.cell_um2 / 1e6,
            "array_eff": (self.total_kb * 1024 * 8 * self.cell_um2) / total,
            "mb_per_mm2": (self.total_kb * 1024 * 8 / 1e6) / (total / 1e6),
            "extent_mm": side / 1000.0,
        }

    # ---- energy --------------------------------------------------------
    def energy_pj_per_byte(self) -> dict:
        e_macro_45 = macro_energy_pj_per_byte_45(self.bank_kb, self.word_bytes * 8)
        e_macro_16 = e_macro_45 * E_SCALE_45_TO_16

        a = self.area_mm2()
        # Mean bank->datapath Manhattan distance: ~0.75 * region side for a
        # compact placement, + 0.15 mm to cross the datapath itself. [EST]
        dist_mm = 0.75 * a["extent_mm"] + 0.15
        e_wire = WIRE_PJ_PER_BIT_MM_16 * 8 * dist_mm
        e_mux = E_MUX_FLOP_PJ_PER_BYTE
        return {
            "macro_45_pj_b": e_macro_45,
            "macro_16_pj_b": e_macro_16,
            "wire_pj_b": e_wire,
            "mux_pj_b": e_mux,
            "total_pj_b": e_macro_16 + e_wire + e_mux,
            "dist_mm": dist_mm,
            # True when the bank word is narrower than any published data point:
            # the energy figure is then an unbelievable extrapolation.
            "extrapolated": self.word_bytes * 8 < MIN_SUPPORTED_BANK_WORD_BITS,
        }

    def leakage_mw(self, temp_hot: bool = False) -> float:
        base = self.total_kb * SRAM_LEAK_UW_PER_KB_85C / 1000.0
        base *= (1.0 + PERIPH_LEAK_FRACTION * (self.n_banks / 8.0))
        return base * (4.0 if temp_hot else 1.0)   # ~4x at 125C [EST]


# ---------------------------------------------------------------------------
# MAC array area, for the comparison the decision actually turns on
# ---------------------------------------------------------------------------

def mac_array_mm2(n_macs: int, native: str = "bf16") -> float:
    g = GATES_MAC_BF16_FP32ACC if native == "bf16" else GATES_MAC_INT8
    return n_macs * g * A_GATE_UM2 * MAC_STAGING_OVERHEAD / 1e6


# ---------------------------------------------------------------------------
# Effective vs peak bandwidth: short rows and bank conflicts
# ---------------------------------------------------------------------------

def effective_bw_short_rows(row_len_elems: int, elem_bytes: int, bw_bpc: float,
                            segmented_reduction: bool) -> float:
    """A GEMV reads one matrix row per dot product. If the row is not a
    multiple of the fetch width, the tail beat is partly wasted UNLESS the
    reduction tree can split one beat across two output accumulators
    ("segmented reduction"). This is the single biggest reason peak B/cycle
    overstates delivered B/cycle on small solver problems."""
    row_bytes = row_len_elems * elem_bytes
    if segmented_reduction:
        return bw_bpc          # rows are contiguous in a row-major matrix
    cycles = math.ceil(row_bytes / bw_bpc)
    return row_bytes / cycles


def transpose_conflict_factor(row_len_elems: int, elem_bytes: int,
                              n_banks: int, word_bytes: int) -> float:
    """A.T @ y done as a COLUMN walk of a row-major matrix is a strided access.
    With addr-interleaved banks the stride can collapse onto few banks.
    Returns the achieved fraction of peak bandwidth for that walk.

    The fix is a loop-order change, not hardware: A.T@y = sum_i A[i,:].T * y_i
    is row-sequential and conflict-free. This function exists to show what the
    naive column walk would cost, i.e. the size of the codegen obligation."""
    stride_words = (row_len_elems * elem_bytes) / word_bytes
    if abs(stride_words - round(stride_words)) > 1e-9:
        return 1.0                     # unaligned stride spreads naturally
    s = int(round(stride_words)) % n_banks
    if s == 0:
        distinct = 1
    else:
        distinct = n_banks // math.gcd(s, n_banks)
    distinct = min(distinct, n_banks)
    return distinct / n_banks


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

CANDIDATES = [
    # (label, total_kb, n_banks, word_bytes)
    ("64 B/cyc,  4 banks x 16B", 256, 4, 16),
    ("64 B/cyc,  8 banks x  8B", 256, 8, 8),
    ("64 B/cyc, 16 banks x  4B", 256, 16, 4),
    ("128 B/cyc,  8 banks x 16B", 256, 8, 16),
    ("128 B/cyc, 16 banks x  8B", 256, 16, 8),
    ("128 B/cyc, 32 banks x  4B", 256, 32, 4),
    ("256 B/cyc,  8 banks x 32B", 256, 8, 32),
    ("256 B/cyc, 16 banks x 16B", 256, 16, 16),
    ("256 B/cyc, 32 banks x  8B", 256, 32, 8),
    ("256 B/cyc, 64 banks x  4B", 256, 64, 4),
    ("128 B/cyc, 128KB 8x16B", 128, 8, 16),
    ("128 B/cyc, 512KB 8x16B", 512, 8, 16),
]


def main():
    L = []

    def out(s=""):
        print(s)
        L.append(s)

    out("=" * 102)
    out("ARGMIND CM — SCRATCHPAD BANKING COST MODEL")
    out("16nm-class, MODELLED ESTIMATE +/-2x, NOT QUOTABLE EXTERNALLY")
    out("=" * 102)
    out()
    out("Bitcell 0.070 um^2 [A1 cited]. Energy vs macro size/width from a 45nm")
    out("published table [A2 cited], scaled to 16nm by 0.30x [ESTIMATED].")
    out("Wire energy from 0.099 pJ/bit/mm at 45nm [A3 cited], halved for 16nm [EST].")
    out()

    # ---- 1. AREA -------------------------------------------------------
    out("-" * 102)
    out("1. AREA vs BANKING (256 KB unless noted; single-port 6T HD cell)")
    out("-" * 102)
    out("CALIBRATION CHECK against [A4] measured 16nm 6T-SP macro = 8.1 Mb/mm^2:")
    for kb, wb in ((16, 8), (32, 8), (32, 16), (64, 16)):
        g = bank_geometry(kb, wb * 8)
        d = (kb * 1024 * 8 / 1e6) / (g.area_um2 / 1e6)
        out(f"  {kb:>3} KB x {wb*8:>3}b macro alone: {d:>5.2f} Mb/mm^2 "
            f"(array eff {100*g.array_eff:>4.1f}%)  target ~8.1 / 57%")
    out()
    hdr = (f"{'config':<26} {'bank':>6} {'row':>4} {'col':>5} {'mux':>4} {'fld':>4} "
           f"{'macro':>7} {'ctrl':>6} {'TOTAL':>7} {'eff':>6} {'Mb/mm2':>7} "
           f"{'extent':>7}")
    out(hdr)
    out(f"{'':<26} {'KB':>6} {'/fld':>4} {'/fld':>5} {'':>4} {'':>4} {'mm2':>7} "
        f"{'mm2':>6} {'mm2':>7} {'%':>6} {'':>7} {'mm':>7}")
    out("-" * len(hdr))
    for label, kb, nb, wb in CANDIDATES:
        c = SpadCfg(kb, nb, wb)
        g, a = c.geom, c.area_mm2()
        out(f"{label:<26} {c.bank_kb:>6.0f} {g.rows:>4} {g.cols:>5} {g.mux:>4} "
            f"{g.folds:>4} "
            f"{a['macro_mm2']:>7.3f} {a['ctrl_mm2']:>6.3f} {a['total_mm2']:>7.3f} "
            f"{100*a['array_eff']:>6.1f} {a['mb_per_mm2']:>7.2f} "
            f"{a['extent_mm']:>7.2f}")
    out()
    out("  Bitcell-limited density at 16nm = 14.3 Mb/mm^2. A large hero macro")
    out("  reaches ~80% of that [A5]; nothing here does, because the 800 MHz")
    out("  256-row bitline limit forbids the long bitlines that buy that density.")
    out()
    out("  Small-macro penalty, measured on this model (the thing the task asked")
    out("  us not to assume away):")
    for kb, nb, wb in [(256, 2, 32), (256, 4, 16), (256, 8, 16), (256, 16, 16),
                       (256, 32, 8), (256, 64, 4)]:
        try:
            c = SpadCfg(kb, nb, wb)
            a = c.area_mm2()
            out(f"    {nb:>3} banks x {c.bank_kb:>5.0f} KB : "
                f"{a['total_mm2']:.3f} mm2, array eff {100*a['array_eff']:.1f}%, "
                f"{a['mb_per_mm2']:.2f} Mb/mm2")
        except ValueError as e:
            out(f"    {nb:>3} banks x {256/nb:>5.0f} KB : ILLEGAL — {e}")
    out()

    # ---- routing channel sensitivity ----
    out("  If the licensee CANNOT route the read bus over the macros (macro")
    out("  blockage to top metal, or a flow that forbids it):")
    for label, kb, nb, wb in [("64 B/cyc,  8 banks x  8B", 256, 8, 8),
                              ("128 B/cyc, 16 banks x  8B", 256, 16, 8),
                              ("256 B/cyc, 32 banks x  8B", 256, 32, 8)]:
        ov = SpadCfg(kb, nb, wb, route_over_macro=True).area_mm2()
        no = SpadCfg(kb, nb, wb, route_over_macro=False).area_mm2()
        out(f"    {label:<26} {ov['total_mm2']:.3f} -> {no['total_mm2']:.3f} mm2 "
            f"(+{100*(no['total_mm2']/ov['total_mm2']-1):.0f}%, channel "
            f"{no['channel_mm2']:.3f} mm2)")
    out()

    # ---- cell choice sensitivity ----
    out("  If 800 MHz + licensee timing margin forces the high-current cell")
    out("  (0.105 um^2 [EST]) instead of the HD cell:")
    for label, kb, nb, wb in [("128 B/cyc, 16 banks x  8B", 256, 16, 8),
                              ("256 B/cyc, 32 banks x  8B", 256, 32, 8)]:
        hd = SpadCfg(kb, nb, wb).area_mm2()
        hc = SpadCfg(kb, nb, wb, cell_um2=A_CELL_HC_UM2).area_mm2()
        out(f"    {label:<26} {hd['total_mm2']:.3f} -> {hc['total_mm2']:.3f} mm2 "
            f"(+{100*(hc['total_mm2']/hd['total_mm2']-1):.0f}%)")
    out()

    # ---- 2. MAC ARRAY COMPARISON ---------------------------------------
    out("-" * 102)
    out("2. WHAT THE SAME AREA BUYS IN MACs  (the decision this all turns on)")
    out("-" * 102)
    for n in (256, 512, 1024, 4096):
        out(f"  {n:>5} MACs (bf16 mult / fp32 accum) : "
            f"{mac_array_mm2(n):.3f} mm2   |   int8-only: "
            f"{mac_array_mm2(n, 'int8'):.3f} mm2")
    out()
    a64 = SpadCfg(256, 8, 8).area_mm2()["total_mm2"]
    a128 = SpadCfg(256, 16, 8).area_mm2()["total_mm2"]
    a256 = SpadCfg(256, 32, 8).area_mm2()["total_mm2"]
    out(f"  Going 64 -> 128 B/cycle costs {a128 - a64:+.3f} mm2 "
        f"({100*(a128/a64-1):+.0f}%)  = {(a128-a64)/mac_array_mm2(1):.0f} bf16 MACs")
    out(f"  Going 64 -> 256 B/cycle costs {a256 - a64:+.3f} mm2 "
        f"({100*(a256/a64-1):+.0f}%)  = {(a256-a64)/mac_array_mm2(1):.0f} bf16 MACs")
    out(f"  For reference 256 -> 1024 MACs costs "
        f"{mac_array_mm2(1024)-mac_array_mm2(256):+.3f} mm2 and buys ZERO on")
    out("  batch-1 GEMV (settled finding, PHASE0_FINDINGS.md).")
    out()
    out(f"  Multi-porting instead of banking: 8T dual-port cell is ~2.1x the 6T")
    out(f"  single-port area [A4 cited]. Two read ports on 256 KB via a DP cell:")
    out(f"    ~{SpadCfg(256, 8, 8).area_mm2()['total_mm2']*2.1:.3f} mm2 for 2x8B/bank "
        f"= 128 B/cyc, vs {a128:.3f} mm2 for 16 single-port banks.")
    out("    -> BANKING WINS ON AREA BY ~2x. Multi-porting is also worse on")
    out("       energy (longer bitlines, more per-access periphery) and on")
    out("       timing. Decision: bank, never multi-port. Confidence: high.")
    out()

    # ---- 3. ENERGY -----------------------------------------------------
    out("-" * 102)
    out("3. ENERGY PER BYTE vs BANKING — does E_SRAM_BYTE_PJ = 0.40 survive?")
    out("-" * 102)
    hdr = (f"{'config':<26} {'macro45':>8} {'macro16':>8} {'wire':>7} {'mux':>6} "
           f"{'TOTAL':>7} {'vs 0.40':>8} {'leak mW':>8}")
    out(hdr)
    out(f"{'':<26} {'pJ/B':>8} {'pJ/B':>8} {'pJ/B':>7} {'pJ/B':>6} {'pJ/B':>7} "
        f"{'':>8} {'85C':>8}")
    out("-" * len(hdr))
    for label, kb, nb, wb in CANDIDATES:
        c = SpadCfg(kb, nb, wb)
        e = c.energy_pj_per_byte()
        out(f"{label:<26} {e['macro_45_pj_b']:>8.2f} {e['macro_16_pj_b']:>8.3f} "
            f"{e['wire_pj_b']:>7.3f} {e['mux_pj_b']:>6.3f} {e['total_pj_b']:>7.3f} "
            f"{e['total_pj_b']/0.40:>7.2f}x {c.leakage_mw():>8.2f}"
            + ("  <-- EXTRAPOLATED, do not believe" if e['extrapolated'] else ""))
    out()
    out("  VALIDITY LIMIT: rows marked EXTRAPOLATED have a bank word narrower than")
    out("  64 bits, below which no published energy data exists. The model clamps")
    out("  and therefore keeps predicting ever-cheaper bytes; physically pJ/byte")
    out("  must turn back UP as per-access fixed costs stop amortising. DESIGN RULE:")
    out("  bank word >= 64 bits, i.e. bank count <= (bytes/cycle)/8.")
    out()
    out("  READ THIS CAREFULLY — the naive intuition in the task prompt is WRONG:")
    out("  macro-internal energy per byte FALLS with more banks (shorter bitlines,")
    out("  fewer rows precharged) and FALLS with wider per-bank words (fixed decode")
    out("  amortised over more bits). [A2] is unambiguous on both.")
    out("  What RISES with bandwidth is the DISTRIBUTION network: a physically")
    out("  larger scratchpad region and a wider bus mean more wire energy per byte.")
    out("  That term is absent from the current model, and at 16nm it is comparable")
    out("  to the macro read itself.")
    out()

    # ---- 4. SHORT ROWS -------------------------------------------------
    out("-" * 102)
    out("4. DELIVERED vs PEAK BANDWIDTH — short GEMV rows (fp32, 4 B/elem)")
    out("-" * 102)
    probs = [("PWR-MPC   n_var=30", 30), ("QUAD-MPC  n_var=80", 80),
             ("ARM-MPC   n_var=210", 210), ("EKF-24    n=24", 24),
             ("EKF-96    n=96", 96), ("SLM row   n=2048", 2048)]
    hdr = (f"{'workload':<22} " + " ".join(f"{'bw'+str(b):>16}" for b in (64, 128, 256)))
    out(hdr)
    out(f"{'':<22} " + " ".join(f"{'naive   seg':>16}" for _ in range(3)))
    out("-" * len(hdr))
    for name, n in probs:
        cells = []
        for bw in (64, 128, 256):
            naive = effective_bw_short_rows(n, 4, bw, False)
            seg = effective_bw_short_rows(n, 4, bw, True)
            cells.append(f"{naive:>7.0f} {seg:>8.0f}")
        out(f"{name:<22} " + " ".join(f"{c:>16}" for c in cells))
    out()
    out("  'naive' = one dot product per fetch beat; the tail beat is wasted.")
    out("  'seg'   = segmented reduction: one beat may straddle two output rows.")
    out("  Without segmented reduction, 256 B/cycle delivers as little as 47% of")
    out("  peak on PWR-MPC and 62% on QUAD-MPC. The model's bandwidth sweep")
    out("  assumes 100%. SEGMENTED REDUCTION IS A REQUIRED FEATURE, not an")
    out("  optimisation — without it, roughly a third of the bandwidth you paid")
    out("  area and power for is unreachable on the beachhead workload.")
    out()

    # ---- 5. BANK CONFLICTS ---------------------------------------------
    out("-" * 102)
    out("5. BANK CONFLICTS — naive A.T@y column walk (fraction of peak achieved)")
    out("-" * 102)
    hdr = (f"{'workload':<22} " + " ".join(f"{str(nb)+'bx'+str(wb)+'B':>10}"
           for nb, wb in ((8, 8), (16, 8), (32, 8), (16, 16), (32, 16))))
    out(hdr)
    out("-" * len(hdr))
    for name, n in probs:
        cells = [f"{100*transpose_conflict_factor(n, 4, nb, wb):>9.0f}%"
                 for nb, wb in ((8, 8), (16, 8), (32, 8), (16, 16), (32, 16))]
        out(f"{name:<22} " + " ".join(f"{c:>10}" for c in cells))
    out()
    out("  Sequential row reads (P@x, A@x, cached triangular solve) are")
    out("  CONFLICT-FREE BY CONSTRUCTION with address-interleaved banks — the")
    out("  dominant solver pattern is fine, which is the good news.")
    out("  A.T@y walked as columns is NOT: on QUAD-MPC (n_var=80, 320 B stride)")
    out("  a 32-bank x 8B organisation lands on 4 of 32 banks -> 12.5% of peak.")
    out("  FIX IS FREE AND IN SOFTWARE: A.T@y = sum_i A[i,:].T * y_i is")
    out("  row-sequential. Cost is one n_var-length fp32 accumulator (320 B for")
    out("  QUAD-MPC, 840 B for ARM-MPC) held in the vector RF, not a transposed")
    out("  copy of A. This is a CODEGEN OBLIGATION to write down in the MACIF /")
    out("  codegen contract, and a reason the number of banks should not be a")
    out("  power of two that divides common matrix strides. Prefer XOR-swizzled")
    out("  bank indexing as belt-and-braces (deterministic, statically known).")
    out()

    # ---- 6. RECOMMENDATION ---------------------------------------------
    out("=" * 102)
    out("6. RECOMMENDED CM-100 BASELINE")
    out("=" * 102)
    rec = SpadCfg(256, 16, 8)
    a, e = rec.area_mm2(), rec.energy_pj_per_byte()
    out("  Scratchpad : 256 KB, 16 single-port banks x 8 B/cycle = 128 B/cycle")
    out(f"               {rec.geom.rows} rows x {rec.geom.cols} cols, "
        f"{rec.geom.mux}:1 col mux, {rec.bank_kb:.0f} KB/bank")
    out("  MAC array  : 256 MACs (16x16), bf16/fp16 mult + fp32 accumulate,")
    out("               int8 double-pumped, fp32 by 2-pass decomposition")
    out("  Vector unit: 16 fp32 lanes (up from 8: past ~128 B/cycle the ADMM")
    out("               projection/transcendental work becomes the co-bottleneck)")
    out("  Frequency  : 800 MHz")
    out()
    out(f"  Scratchpad area      {a['total_mm2']:.3f} mm2  ({a['mb_per_mm2']:.1f} Mb/mm2, "
        f"{100*a['array_eff']:.0f}% array efficiency)")
    out(f"  MAC array area       {mac_array_mm2(256):.3f} mm2")
    out(f"  Vector unit          {16*2200*A_GATE_UM2/1e6:.3f} mm2  [EST 2.2k gates/lane]")
    out(f"  Sequencer + descr.   {0.045:.3f} mm2  [EST]")
    out(f"  DMA + AXI/ACE-Lite   {0.070:.3f} mm2  [EST]")
    out(f"  misc (CSR, MBIST..)  {0.030:.3f} mm2  [EST]")
    sub = (a['total_mm2'] + mac_array_mm2(256) + 16*2200*A_GATE_UM2/1e6
           + 0.045 + 0.070 + 0.030)
    out(f"  ------------------------------------")
    out(f"  Subtotal             {sub:.3f} mm2")
    out(f"  +15% top-level route/CTS/margin      -> {sub*1.15:.3f} mm2 TOTAL [EST]")
    out()
    out(f"  Scratchpad read energy {e['total_pj_b']:.2f} pJ/byte  "
        f"(vs 0.40 assumed = {e['total_pj_b']/0.40:.1f}x)")
    out(f"  Scratchpad leakage     {rec.leakage_mw():.2f} mW @85C, "
        f"{rec.leakage_mw(True):.2f} mW @125C")
    out()

    import os
    dest = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "results_sram_cost.txt")
    with open(dest, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
