"""
Argmind CM — Phase 0 first-order performance model (v0)
=======================================================

Purpose: decide the Phase-0 go/no-go. Estimates latency and energy for the frozen
benchmark set (see ../BENCHMARKS.md) on:

  * Argmind CM   — GEMM tile + vector unit + sequencer, with residency tiers
  * CPU baseline — Cortex-A78-class + NEON, and an SME2-class variant
  * Naive accel  — an Argmind-class datapath WITHOUT zero-copy/residency (§3.2.1/3.2.2
                   ablation), i.e. what a conventional NPU-style offload would achieve

Design stance: this model is deliberately built so it CAN report failure. The kill
criterion (>=5x perf/W vs CPU on solver workloads) is evaluated mechanically, and the
crossover sweep reports where the CPU wins. No dependencies beyond the stdlib.

ALL ENERGY CONSTANTS ARE FIRST-ORDER ESTIMATES at a 16nm-class node, scaled from
Horowitz (ISSCC 2014) and vendor datasheets. They are good to maybe +/-2x, which is
adequate for a >=5x go/no-go screen and NOT adequate for customer-facing claims.
Phase 1 replaces them with synthesis+PDK numbers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

try:
    from . import sram_cost
except ImportError:          # run as a script
    import sram_cost

# ---------------------------------------------------------------------------
# Energy constants (ESTIMATES, 16nm-class)
# ---------------------------------------------------------------------------

# Per-MAC dynamic energy (pJ), datapath only.
#
# int4 ADDED 2026-08-05 for the ML-envelope study. READ THE THREE RULES BELOW
# BEFORE USING IT; two of them exist to stop int4 from becoming a free win.
#
#  (1) int4 IS WEIGHTS-ONLY, WHICH IS WHAT SHIPPING LLM QUANTISATION DOES.
#      GPTQ (Frantar et al., ICLR 2023, arXiv:2210.17323) and AWQ (Lin et al.,
#      MLSys 2024, arXiv:2306.00978) quantise WEIGHTS to 3-4 bits and keep
#      activations at 8 or 16 bits. So the product a 4-bit kernel actually
#      executes is 4b x 8b, not 4b x 4b, and it is expressed in this model as
#      `prec="int8", operand_prec="int4"` using the EXISTING mixed-precision
#      machinery -- not as a uniform int4 row. A uniform int4 row would be a
#      workload nobody ships.
#  (2) int4 DOES NOT DOUBLE THE ARRAY RATE. A digital multiplier is sized by its
#      WIDER operand, so a 4b x 8b product costs the same array slot as
#      8b x 8b. `macs_per_cycle` therefore gives int4 the int8 rate (see
#      ArgmindCfg.macs_per_cycle). int4 buys BANDWIDTH -- half the bytes per
#      weight element -- and nothing else. Giving it 4x the array rate is the
#      exact shape of the two most recent modelling errors (13, 14) inverted:
#      an axis made to look free because the model could not express its cost.
#  (3) ENERGY FALLS, BUT NOWHERE NEAR 2x. Narrowing one operand from 8b to 4b
#      halves that operand's partial-product rows, but the accumulator, the
#      operand registers, the clock and the adder tree do not narrow at all.
#      0.06 pJ is ~0.86x int8 [EST]. The study reports the sensitivity: setting
#      it equal to int8's 0.07 moves no conclusion, because every row where int4
#      matters is DRAM-dominated.
#
# ACCURACY IS NOT MODELLED. Nothing here says a network still works at int4.
# GPTQ/AWQ report single-digit-percent perplexity cost on billion-scale models
# and much worse on small ones; that is a workload property, and this model
# speaks only about what the datapath can move and compute.
E_MAC_PJ = {"int4": 0.06, "int8": 0.07, "fp16": 0.20, "bf16": 0.20,
            "fp32": 0.60, "fp64": 2.20}
BYTES_PER_ELEM = {"int4": 0.5, "int8": 1, "fp16": 2, "bf16": 2,
                  "fp32": 4, "fp64": 8}

# int4 is SUB-BYTE, and that is an interface fact, not just a table entry: two
# elements share a byte, so an int4 matrix row must be an even number of elements
# and a leading dimension is a count of NIBBLES. The v0.1 descriptor's AGUs
# address bytes. Recording it here because the last two capacity errors were
# both "the model assumed a layout the descriptor format cannot address"
# (packed triangular, PHASE0_FINDINGS R8) -- so this one is stated out loud
# rather than discovered later. See PHASE0_FINDINGS 8.6 for the cost.
SUB_BYTE_FORMATS = ("int4",)

# LEGACY flat scratchpad energy. Retained only so the v0 numbers stay
# reproducible. sram_cost.py shows this is ~2x low once the bank->datapath
# distribution network is counted; set ArgmindCfg.use_banked_energy=True
# (the default) to use the banking-derived figure instead.
E_SRAM_BYTE_PJ = 0.40
E_DRAM_BYTE_PJ = 12.0    # LPDDR4/5-class, incl. PHY+controller
E_L2_BYTE_PJ = 1.5       # CPU-side L2 access

# Everything a raw MAC-energy count ignores: clock distribution, register files,
# operand staging, control/sequencing logic, on-chip interconnect. Published
# accelerator designs land 2-4x above their datapath-only energy; ignoring this
# is the single most common way an architecture model flatters itself.
E_OVERHEAD_MULT = 2.5
# When the scratchpad's own wire/mux energy is modelled EXPLICITLY (banked
# mode), part of what E_OVERHEAD_MULT stood in for is now counted directly.
# Keeping 2.5 would double-count it. 2.0 is the non-double-counting value.
E_OVERHEAD_MULT_BANKED = 2.0

# ---------------------------------------------------------------------------
# Frequency <-> supply voltage, 16nm-class. ADDED 2026-07-31.
# ---------------------------------------------------------------------------
# WHY THIS EXISTS. Before this, `freq_mhz` scaled latency as 1/f and left energy
# per operation untouched, so the model reported FREQUENCY AS A FREE LATENCY
# AXIS -- push to 2 GHz, get 2.5x the speed at LOWER energy per call (because
# static power integrated over a shorter window fell). That is unphysical above
# the nominal operating point: a 16nm standard-cell path only closes at 2 GHz
# with a supply boost, and dynamic energy goes as V^2.
#
# It was a FLATTERING omission and it is the reason the frequency axis had never
# produced a wall. Two costs are now charged:
#   * dynamic (MAC + scratchpad + overhead multiplier)  x (V/V_NOM)^2
#   * static / leakage                                  x (V/V_NOM)^3
# Cubic on leakage is a crude proxy for V*exp(V/nVt) over this narrow range; the
# honest band is quadratic to quartic. [EST]
#
# At 800 MHz both multipliers are EXACTLY 1.0, so no pre-existing number moves.
FV_POINTS_16NM = ((400.0, 0.65), (600.0, 0.72), (800.0, 0.80), (1000.0, 0.86),
                  (1200.0, 0.92), (1600.0, 1.02), (2000.0, 1.10))   # [EST]
V_NOM_16NM = 0.80


def vdd_for_freq(freq_mhz: float) -> float:
    """Supply needed to close timing at `freq_mhz`, 16nm-class. [EST]

    Piecewise-linear over FV_POINTS_16NM, clamped at both ends. Clamping at the
    low end is deliberate: below ~400 MHz you are near-threshold and the model
    has no basis to extrapolate, so it stops giving credit for going slower.
    """
    pts = FV_POINTS_16NM
    if freq_mhz <= pts[0][0]:
        return pts[0][1]
    if freq_mhz >= pts[-1][0]:
        return pts[-1][1]
    for (f0, v0), (f1, v1) in zip(pts, pts[1:]):
        if f0 <= freq_mhz <= f1:
            return v0 + (v1 - v0) * (freq_mhz - f0) / (f1 - f0)
    return pts[-1][1]


# ---------------------------------------------------------------------------
# Hardware configurations
# ---------------------------------------------------------------------------

@dataclass
class ArgmindCfg:
    """One instantiation of the CM."""
    name: str
    tile: int = 16               # systolic tile dimension (tile x tile_cols MACs)
    # Second array dimension. None = square. Exists because MAC COUNT and
    # OPERAND-REUSE DEPTH are separate knobs and the model previously conflated
    # them: a Tm x Tn array reuses each A element Tn times and each B element Tm
    # times, so GEMM operand traffic is (1/Tm + 1/Tn) elements per MAC, not
    # 2/sqrt(macs). This matters because the surviving argument for a larger
    # array is GEMM operand-fetch ENERGY, not latency.
    tile_n: int | None = None
    freq_mhz: float = 800.0
    vector_lanes: int = 8        # fp32 lanes in the vector unit
    drain_cycles: int = 8        # pipeline fill/drain per tile pass
    scratchpad_kb: int = 256
    # Scratchpad read bandwidth (bytes/cycle) across all banks. THIS, not the MAC
    # count, is the binding constraint on matrix-VECTOR work, where every MAC needs
    # a fresh matrix element and there is no operand reuse to exploit.
    scratchpad_bw_bpc: float = 64.0
    static_mw: float = 15.0
    # Sequencer runs the solver loop in-fabric at ~1 IPC.
    seq_ipc: float = 1.0
    # Host interaction cost per CM *call* (not per iteration).
    #   tier "T0" = copy-based offload, tier "T1" = zero-copy + residency
    tier: str = "T1"

    # ---- banking / physical realism (added when the wide scratchpad was
    #      finally costed; see sram_cost.py) -------------------------------
    # n_banks * bank_word_bytes must equal scratchpad_bw_bpc. If n_banks is
    # None the model falls back to the flat legacy behaviour so v0 numbers
    # stay reproducible.
    n_banks: int | None = None
    use_banked_energy: bool = True
    # Can the reduction tree split one fetch beat across two output rows? If
    # not, the tail beat of every short GEMV row is wasted and DELIVERED
    # bandwidth is well below peak. This is a real RTL feature with a cost.
    segmented_reduction: bool = True

    # ---- sequencer / descriptor issue (modelled 2026-07-30; sequencer.py) ---
    # Sequencer cycles to emit ONE descriptor: 3 operand base addresses, the
    # dims/strides, the accumulate/format flags, the queue push. A hardware
    # descriptor generator with AGUs and a hardware loop that walks a template
    # is at the low end (2-4); a scalar core writing CSRs at 1 IPC is at the
    # high end (16-32). THIS REPLACES the old flat `scalar_ops_per_iter` term
    # for the CM (the CPU model still charges scalar ops, correctly, because a
    # CPU really does execute the loop in software).
    seq_setup_cycles: float = 4.0
    # Descriptors the sequencer can emit per cycle. >1 needs a multi-ported
    # descriptor queue; the analysis in sequencer.py shows it is never needed.
    seq_issue_width: float = 1.0
    # Descriptor queue depth between sequencer and datapath. 0 = NO decoupling:
    # the sequencer writes a descriptor, starts it, waits for completion. Any
    # value >=1 is decoupled access-execute and lets setup overlap execution.
    # This is the knob that replaces the old fixed "50% overlap" fudge.
    seq_queue_depth: int = 8
    # Use the descriptor-derived sequencer model when the benchmark carries a
    # kernel decomposition. False reproduces the legacy flat-scalar behaviour.
    use_descriptor_seq: bool = True
    # Override the blocking factor of a dependency chain (Cholesky, triangular
    # solve). None = the datapath width, which is the design intent. Set it
    # smaller to price the RTL risk that a potrf cannot be blocked that coarsely;
    # sequencer.py sweeps it because that is the largest downside risk here.
    chain_block_override: int | None = None

    # ---- fp64 datapath (costed 2026-07-30; see fp64_datapath.py) -----------
    # Number of IEEE-754 fp64 FMA units. THIS IS AN ABSOLUTE COUNT, NOT A RATE
    # FACTOR ON THE bf16 ARRAY. The model previously obtained fp64 from a 0.125
    # multiplier on `macs`, which implied 18 fp64 MAC/cycle on the CM-100 for
    # zero area -- a modelling convenience with no corresponding hardware.
    # Interior-point MPC (ARM-MPC / HPIPM) needs fp64, so the capability is not
    # optional and must appear in both the area budget and the throughput ceiling.
    # The recommended implementation is a merged multi-format cluster that is
    # ALSO the fp32 vector unit (1x fp64 or 2x fp32 per unit), so
    # `vector_lanes == 2 * fp64_fmas` in the recommended configurations.
    fp64_fmas: int = 8

    @property
    def bank_word_bytes(self) -> int:
        if self.n_banks is None:
            return 8
        return int(self.scratchpad_bw_bpc / self.n_banks)

    @property
    def spad(self):
        nb = self.n_banks if self.n_banks is not None else \
            max(1, int(self.scratchpad_bw_bpc // 8))
        wb = max(1, int(round(self.scratchpad_bw_bpc / nb)))
        return sram_cost.SpadCfg(self.scratchpad_kb, nb, wb,
                                 freq_mhz=self.freq_mhz)

    def sram_pj_per_byte(self) -> float:
        if not self.use_banked_energy:
            return E_SRAM_BYTE_PJ
        try:
            return self.spad.energy_pj_per_byte()["total_pj_b"]
        except ValueError:
            return E_SRAM_BYTE_PJ

    @property
    def overhead_mult(self) -> float:
        return E_OVERHEAD_MULT_BANKED if self.use_banked_energy else E_OVERHEAD_MULT

    # ---- frequency / voltage coupling (added 2026-07-31) -------------------
    @property
    def vdd(self) -> float:
        return vdd_for_freq(self.freq_mhz)

    @property
    def dyn_energy_scale(self) -> float:
        """Dynamic energy multiplier vs the 800 MHz nominal point. == 1.0 at 800."""
        return (self.vdd / V_NOM_16NM) ** 2

    @property
    def leak_scale(self) -> float:
        """Leakage multiplier vs the 800 MHz nominal point. == 1.0 at 800."""
        return (self.vdd / V_NOM_16NM) ** 3

    @property
    def tile_cols(self) -> int:
        return self.tile if self.tile_n is None else self.tile_n

    @property
    def macs(self) -> int:
        return self.tile * self.tile_cols

    def gemm_operand_bytes_per_mac(self, prec: str) -> float:
        """Scratchpad bytes fetched per GEMM MAC.

        Output-stationary Tm x Tn tiling reads A once per column-block and B
        once per row-block: M*K*(N/Tn) + K*N*(M/Tm) = MACs*(1/Tm + 1/Tn).
        At FIXED MAC count Tm*Tn, (1/Tm + 1/Tn) is minimised when Tm == Tn
        (AM-HM), so a SQUARE array is provably optimal for operand-fetch energy
        and a rectangular one cannot buy reuse depth cheaply. This is why the
        array-size question reduces to choosing one number.

        CONSERVATIVE: assumes no operand register file below the tile. An L0
        staging tier would cut this without adding MACs -- see the sizing note,
        because it is the cheapest way to buy the reuse this term charges for.
        """
        return BYTES_PER_ELEM[prec] * (1.0 / self.tile + 1.0 / self.tile_cols)

    def macs_per_cycle(self, prec: str) -> float:
        """Peak MAC/cycle at `prec`.

        int8 double-pumps and fp32 is half-rate through the bf16 array (2-pass
        operand split, products exact in the fp32 accumulator).

        fp64 DOES NOT come from the array. Splitting fp64 onto bf16 multipliers
        with the error-free (Ozaki) property that interior point actually needs
        costs 49-361 passes, not 8 -- see fp64_datapath.py for the derivation and
        the citation. fp64 throughput is therefore the size of the dedicated FMA
        cluster and is INDEPENDENT of the MAC array. Consequence worth stating:
        on fp64 work the part is ARRAY-bound, not bandwidth-bound -- the one
        documented exception to the bandwidth-first finding, and it holds because
        ARM-MPC is 79% matrix-MATRIX (block Riccati), not GEMV.

        int4 GETS THE int8 RATE, NOT DOUBLE IT. A digital multiplier array is
        sized by its wider operand and weight-only 4-bit quantisation
        (GPTQ/AWQ-class) multiplies a 4-bit weight by an 8-bit activation, so the
        product occupies the same array slot as int8 x int8. int4's entire
        benefit in this model is that a weight element costs 0.5 B of fetch
        instead of 1 B. If a future macro really does implement 4b x 4b at 2x the
        int8 rate it declares that through MACIF, and it is a MACRO capability,
        not a property of this datapath.
        """
        if prec == "fp64":
            return float(self.fp64_fmas)
        scale = {"int4": 2.0, "int8": 2.0, "fp16": 1.0, "bf16": 1.0,
                 "fp32": 0.5}[prec]
        return self.macs * scale

    def gemm_util(self, prec: str, effective_dim: float) -> float:
        """Fraction of peak achieved on a GEMM whose operands are `effective_dim`.

        The quantisation penalty must be charged against THE UNIT THAT DOES THE
        WORK. fp64 runs on the FMA cluster, not the tile array, so charging it
        `cfg.tile` made a BIGGER ARRAY look SLOWER on fp64 -- caught by
        test_more_macs_never_increases_latency, which is exactly what that
        invariant is for.

        A linear cluster of F units accumulates along the reduction axis, so only
        the innermost loop quantises: modelled as the geometric mean of perfect
        utilisation and the 2-D penalty. Deliberately not claimed as 1.0.
        """
        if effective_dim <= 0:
            return 1.0
        if prec == "fp64":
            T, soften = max(1, self.fp64_fmas), True
        else:
            T, soften = self.tile, False
        quant = math.ceil(effective_dim / T) * T / effective_dim       # >= 1.0
        drain = 1.0 + self.drain_cycles / max(effective_dim, 1.0)
        u = 1.0 / (quant * drain)
        return math.sqrt(u) if soften else u

    def delivered_bw_bpc(self, prec: str, row_len: float | None) -> float:
        """Bandwidth actually DELIVERED to a GEMV, not the peak figure.

        A GEMV reads one matrix row per dot product. If the row is not a
        multiple of the fetch width, the tail beat is partly idle -- unless
        segmented reduction lets a beat straddle two rows. Peak B/cycle
        overstates delivered B/cycle by up to 2x on small solver problems,
        which is exactly where the beachhead lives.
        """
        if row_len is None or row_len <= 0 or self.segmented_reduction:
            return self.scratchpad_bw_bpc
        return sram_cost.effective_bw_short_rows(
            int(row_len), BYTES_PER_ELEM[prec], self.scratchpad_bw_bpc, False)

    def gemv_macs_per_cycle(self, prec: str,
                            row_len: float | None = None) -> float:
        """Matrix-vector throughput: bounded by operand fetch, not by MAC count.

        One matrix element consumed per MAC (no reuse), so the ceiling is
        DELIVERED scratchpad bandwidth / element size. A bigger tile array does
        NOT help.
        """
        bw_limit = self.delivered_bw_bpc(prec, row_len) / BYTES_PER_ELEM[prec]
        return min(bw_limit, self.macs_per_cycle(prec))

    @property
    def cycle_ns(self) -> float:
        return 1000.0 / self.freq_mhz

    # ---- host-interface characteristics -----------------------------------
    #
    # DECOMPOSED 2026-07-30 to settle the MCU-class contradiction. The two flat
    # constants (0.8 us and 25.0 us) were fine for a rich-OS/DRAM host and wrong
    # for an MCU-class one, because TWO of T0's four cost terms -- buffer
    # map/pin and cache clean+invalidate -- are artefacts of an MMU and a cache.
    # An all-SRAM MCU has neither, so quoting 25 us there overstates what
    # zero-copy buys by roughly 12x. Every number here is [EST]; the point of
    # writing them out is that each one names a mechanism that can be measured
    # separately, which a single constant cannot.
    #
    #   term                    T0 (rich OS)  T1 (coherent)  T0M (MCU)  T1M (MCU)
    #   driver / RTOS API             8.0          --            1.5       --
    #   buffer map + pin              6.0          --            --        --
    #   cache clean + invalidate      6.0          --            --        --
    #   IRQ + scheduler wakeup        5.0          --            0.5       --
    #   descriptor write              --           0.1           --        0.10
    #   doorbell store                --           0.1           --        0.05
    #   completion poll / latency     --           0.6           --        0.15
    #   TOTAL                        25.0          0.8           2.0       0.30
    #
    # T1M is CHEAPER than T1 because there is no coherence-protocol round trip:
    # the descriptor lands in tightly-coupled memory the CM can already see.
    # Note what the table says about the pitch: the T1-vs-T0 overhead advantage
    # is 31x on a rich host and 6.7x on an MCU-class host, and in ABSOLUTE terms
    # 24.2 us versus 1.7 us. It is still a real saving, and it is still
    # descriptor/doorbell cost rather than data movement -- but it is 14x smaller
    # than the headline figure and only decides a deadline above ~100 kHz.
    CALL_OVERHEAD_US = {"T0": 25.0, "T1": 0.8, "T0M": 2.0, "T1M": 0.30}

    @property
    def call_overhead_us(self) -> float:
        """Fixed cost to launch one CM call from the host."""
        return self.CALL_OVERHEAD_US[self.tier]

    @property
    def is_mcu_class(self) -> bool:
        """MCU-class host: on-chip SRAM only. No DRAM, no cache, no MMU."""
        return self.tier.endswith("M")

    # External (host<->CM) bandwidth in GB/s, overriding the tier default.
    # ADDED 2026-08-05. It exists because the ML-envelope study sweeps external
    # bandwidth, and the previous way to do that was to SUBCLASS ArgmindCfg and
    # shadow the property (see scaling_study.py AXIS 7) -- which works but cannot
    # be handed to `dataclasses.replace`, and which silently hid the fact that
    # `model_cpu` was pinning the CPU's DRAM bandwidth to a hard 12.8 GB/s
    # regardless. A swept axis that moves only one side of a comparison is a
    # baseline-fairness bug; both sides now take the same number. None = the
    # tier default, so no pre-existing figure moves.
    dma_bw_gbs_override: float | None = None

    @property
    def dma_bw_gbs(self) -> float:
        if self.dma_bw_gbs_override is not None:
            return self.dma_bw_gbs_override
        if self.is_mcu_class:
            # There is no LPDDR. What a copy actually crosses on an MCU-class
            # part is a narrow on-chip bus: 8 B/cycle at a 200 MHz system clock
            # = 1.6 GB/s [EST]. This is 4x WORSE than the DRAM path it replaces,
            # which is the counter-intuitive half of the MCU-class answer -- the
            # data is closer but the pipe is narrower.
            return 1.6
        return 6.4   # one LPDDR4x-2400 x32 channel share

    @property
    def offchip_pj_per_byte(self) -> float:
        """Energy of one byte crossing the host<->CM boundary.

        On an MCU-class part this is an SRAM-to-SRAM copy over an on-chip bus,
        not a DRAM access, so the 12 pJ/B LPDDR figure is wrong by ~10x. This is
        the term that makes residency nearly worthless in MCU class: it is not
        that the bytes disappear, it is that they were never expensive.
        """
        if self.is_mcu_class:
            return 1.2       # 2x the scratchpad's own pJ/B, for the bus [EST]
        return E_DRAM_BYTE_PJ

    def transfers_persistent_every_call(self) -> bool:
        """Without residency, problem data must be re-uploaded every call."""
        return self.tier in ("T0", "T0M")


@dataclass
class CpuCfg:
    name: str
    freq_mhz: float
    fp32_flops_per_cycle: float     # peak
    int8_ops_per_cycle: float       # peak
    small_gemm_efficiency: float    # achieved fraction of peak on small matrices
    active_w: float                 # sustained core power incl. L1/L2
    l1_bw_bpc: float = 32.0         # L1 load bandwidth, bytes/cycle
    scalar_ipc: float = 2.0
    # fp64 peak. MODEL BUG FIXED 2026-07-30: the model charged fp64 work at the
    # fp32 rate on every CPU baseline, i.e. it gave the CPU a free doubling on
    # ARM-MPC. That error was CONSERVATIVE for us (a faster CPU burns less energy
    # per solve, so our perf/W ratio was understated) -- fixing it INCREASES our
    # reported advantage, which is exactly the direction that demands scrutiny.
    # None = derive as half the fp32 rate, which is the NEON/SVE datapath reality
    # (same vector width, twice the element size).
    fp64_flops_per_cycle: float | None = None
    # Plain-SIMD fp32 rate for ELEMENTWISE and REDUCTION work, as distinct from
    # the matrix/outer-product rate.
    #
    # ADDED 2026-07-30, and it is a correction that moves the ratio IN OUR FAVOUR,
    # so it is flagged rather than banked. `model_cpu` charged `vector_ops` at
    # `peak_flops_per_cycle * small_gemm_efficiency`, i.e. it ran elementwise work
    # on the outer-product unit. For NEON that is harmless -- there is one set of
    # units -- but SME2's ZA-tile outer product cannot execute an elementwise
    # min/max or a horizontal reduction at all. The measured M4 numbers make the
    # gap explicit: FMOPA FP32 2008 GFLOP/s against ~250 GFLOP/s for streaming
    # SVE non-outer-product on the same core (an 8x ratio).
    #
    # None = same as fp32_flops_per_cycle, which is correct for NEON parts and
    # keeps A55/A78 bit-identical. The SME2 value below is deliberately set at the
    # GENEROUS end for the CPU (half the matrix rate, not the measured eighth).
    simd_flops_per_cycle: float | None = None

    @property
    def simd_rate(self) -> float:
        return (self.fp32_flops_per_cycle if self.simd_flops_per_cycle is None
                else self.simd_flops_per_cycle)

    def peak_flops_per_cycle(self, prec: str) -> float:
        # int4 runs at the int8 rate on every CPU baseline, because there IS no
        # 4-bit SIMD dot product: Armv9 gives SDOT/SMMLA at 8 bits and RVV gives
        # widening 8-bit multiplies. A real int4 CPU kernel unpacks nibbles to
        # bytes first, which costs shift/and/interleave work this model does NOT
        # charge -- so the CPU is being handed a free unpack. That is deliberate:
        # the baseline gets its fair advantages, and an unmodelled CPU cost is
        # the direction that hurts our own ratio.
        if prec in ("int8", "int4"):
            return self.int8_ops_per_cycle
        if prec == "fp64":
            return (self.fp32_flops_per_cycle / 2.0
                    if self.fp64_flops_per_cycle is None
                    else self.fp64_flops_per_cycle)
        return self.fp32_flops_per_cycle

    def gemv_macs_per_cycle(self, prec: str) -> float:
        """The CPU is bandwidth-bound on GEMV too -- model it the same way."""
        peak = self.peak_flops_per_cycle(prec) / 2.0             # ops -> MACs
        bw_limit = self.l1_bw_bpc / BYTES_PER_ELEM[prec]
        return min(peak, bw_limit) * self.small_gemm_efficiency


CPU_A78 = CpuCfg(
    name="Cortex-A78-class @2.0GHz (NEON)",
    freq_mhz=2000.0,
    fp32_flops_per_cycle=16.0,      # 2x FMA, 4 lanes fp32
    int8_ops_per_cycle=64.0,        # SDOT
    small_gemm_efficiency=0.30,     # realistic for n<128 GEMM in Eigen/BLIS
    active_w=0.90,
)

CPU_SME2 = CpuCfg(
    name="Cortex-A720-class @2.2GHz (SME2, 256b)",
    freq_mhz=2200.0,
    fp32_flops_per_cycle=64.0,      # streaming outer-product
    int8_ops_per_cycle=256.0,
    small_gemm_efficiency=0.35,
    active_w=1.30,
    l1_bw_bpc=64.0,
    # An SME outer product covers SVL/esize elements per dimension, so fp64 is
    # QUARTER rate, not half: 4x4 fp64 vs 8x8 fp32 at SVL=256b. [EST]
    # NOTE FOR FALSIFIER F4: FEAT_SME_F64F64 is ARCHITECTURALLY OPTIONAL. An
    # SME2 part without it has NO fp64 matrix path at all and falls back to SVE
    # fp64 FMA, which would be a genuine structural gap rather than a ratio.
    # Do not claim either way without checking the target part's ID registers.
    fp64_flops_per_cycle=16.0,
    # Elementwise / reduction work runs on streaming SVE, not on the ZA tile.
    # 32 = HALF the outer-product rate, i.e. ~8 fp32 lanes x 2 FLOP x 2 pipes at
    # SVL=256b [EST]. The measured M4 FMOPA:streaming-SVE ratio is 8x, which would
    # put this at 8; 32 is chosen because it is the value least favourable to us
    # and the conclusion should not need the generous reading.
    simd_flops_per_cycle=32.0,
)

# The honest perf/W adversary: cost-sensitive robotics/industrial designs use an
# efficient in-order core, not a big OoO one. Comparing only against A78/SME2
# would inflate our energy advantage several-fold.
CPU_A55 = CpuCfg(
    name="Cortex-A55-class @1.5GHz (NEON, in-order)",
    freq_mhz=1500.0,
    fp32_flops_per_cycle=8.0,       # 1x FMA pipe, 4 lanes
    int8_ops_per_cycle=32.0,
    small_gemm_efficiency=0.25,
    active_w=0.15,                  # ~6x more efficient per watt than A78
    l1_bw_bpc=16.0,
    scalar_ipc=1.3,
)

ALL_CPUS = [CPU_A55, CPU_A78, CPU_SME2]


# ---------------------------------------------------------------------------
# Kernel decomposition — what the sequencer actually has to issue
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KernelOp:
    """One logically distinct operation in a benchmark's inner loop.

    WHY THIS EXISTS. The model used to charge sequencer work as a flat
    `scalar_ops_per_iter` constant with a fixed 50% compute overlap. That cannot
    express the thing that actually matters for a block-structured solver: how
    many descriptors the work decomposes into, and therefore how much datapath
    work each descriptor commands. HPIPM issues hundreds of small block
    operations per interior-point iteration; ADMM issues a handful of large
    GEMVs. Those are opposite ends of a spectrum the flat constant collapses.

    THE CONSISTENCY RULE, which is the whole discipline here: the `macs` /
    `vec_ops` / `trans_ops` of a benchmark's kernel list must SUM TO the
    benchmark's declared per-iteration totals. Descriptor counts and operation
    counts derived independently is exactly the error that let an earlier
    analysis double-count HPIPM GEMM work 5.7x. `assert_kernels_consistent()`
    enforces it and a test calls it for every benchmark.

    Fields
      kind        "gemm" | "gemv" | "vec" | "trans" — selects the datapath
                  resource, and hence the rate, so the cycles-per-descriptor
                  figure is per-resource rather than an average.
      n_desc      descriptors per `repeat`, for an op with no dependency chain.
      chain_dim   >0 marks a DEPENDENCY CHAIN of this dimension (Cholesky,
                  triangular solve, backward Riccati within a stage). The op is
                  re-blocked along it, so descriptor count becomes
                  ceil(chain_dim / block) * n_desc, and each boundary costs a
                  datapath drain because descriptor i+1 cannot START until i's
                  results are written back. Setup still overlaps (the addresses
                  are statically known) — it is the DRAIN that does not.
      repeat      structural repetition (N MPC stages, transformer layers).
                  `macs` is per repeat.
    """
    name: str
    kind: str
    macs: float = 0.0
    vec_ops: float = 0.0
    trans_ops: float = 0.0
    n_desc: float = 1.0
    chain_dim: float = 0.0
    repeat: float = 1.0
    # Scratchpad bytes fetched per MAC for THIS op, overriding the tile-derived
    # figure. Added 2026-07-30 for FCS-MPC, where the reuse depth of the dominant
    # GEMM is set by the number of controlled OUTPUTS (n_y = 2), not by the tile
    # dimension -- so `elem*(1/Tm + 1/Tn)` massively overstates the reuse. None =
    # use cfg.gemm_operand_bytes_per_mac, i.e. no change to any existing row.
    bytes_per_mac: float | None = None

    @property
    def total_macs(self) -> float:
        return self.macs * self.repeat

    @property
    def total_vec(self) -> float:
        return self.vec_ops * self.repeat

    @property
    def total_trans(self) -> float:
        return self.trans_ops * self.repeat


def chain_block(cfg: ArgmindCfg, prec: str, kind: str) -> int:
    """Blocking factor of a dependency chain on this datapath.

    A blocked Cholesky or triangular solve advances one block of the diagonal at
    a time, and the block is as wide as the unit that does the work: the tile
    dimension for array GEMM, the fp64 FMA count for fp64 (fp64 does NOT run on
    the array), the vector-lane count for a vector-unit triangular solve.
    Smaller block => more descriptors => more drain boundaries and less work per
    descriptor, which is precisely the mechanism under investigation.

    `cfg.chain_block_override` forces a smaller value, to price the RTL risk that
    a blocked Cholesky cannot in fact advance a full datapath width per step.
    """
    if cfg.chain_block_override is not None:
        return max(1, cfg.chain_block_override)
    if kind == "gemm":
        return max(1, cfg.fp64_fmas) if prec == "fp64" else max(1, cfg.tile)
    return max(1, cfg.vector_lanes)


def op_descriptors(op: KernelOp, cfg: ArgmindCfg, prec: str) -> tuple[float, float]:
    """(descriptor count, dependent-boundary count) for one kernel op."""
    if op.chain_dim > 0:
        steps = math.ceil(op.chain_dim / chain_block(cfg, prec, op.kind))
        n = steps * op.n_desc * op.repeat
        # Every descriptor inside a chain is gated on the previous one.
        return n, n
    return op.n_desc * op.repeat, 0.0


def op_cycles(op: KernelOp, cfg: ArgmindCfg, prec: str, dim: float,
              row_len: float | None = None) -> float:
    """Datapath cycles for one kernel op, on the resource its `kind` names."""
    if op.kind == "gemm":
        return gemm_like_cycles(op.total_macs, cfg, prec, dim,
                                bytes_per_mac=op.bytes_per_mac)
    if op.kind == "gemv":
        return op.total_macs / cfg.gemv_macs_per_cycle(prec, row_len)
    if op.kind == "vec":
        return op.total_vec / cfg.vector_lanes
    if op.kind == "trans":
        return op.total_trans * 4.0 / cfg.vector_lanes
    raise ValueError(op.kind)


# ---------------------------------------------------------------------------
# Benchmark definition
# ---------------------------------------------------------------------------

@dataclass
class Benchmark:
    name: str
    kind: str                  # "solver" | "ml"
    note: str
    gemv_macs_per_iter: float  # matrix-VECTOR MACs: no operand reuse, bandwidth-bound
    gemm_macs_per_iter: float  # matrix-MATRIX MACs: reusable, tile-array-bound
    gemm_macs_per_call: float  # once-per-call GEMM (e.g. SQP re-factorisation)
    vector_ops_per_iter: float # elementwise/reduction element-ops
    transcendental_per_iter: float  # div/sqrt/exp (4x cost in vector unit)
    scalar_ops_per_iter: float # sequencer control work per iteration
    iters: int                 # iterations per call
    calls_per_sec: float       # invocation rate (control rate / token rate)
    deadline_us: float         # hard deadline per call (inf if throughput-only)
    persistent_bytes: float    # data reusable across calls (matrices, weights)
    delta_bytes: float         # genuinely new data per call
    prec: str
    # Typical GEMV inner-dot-product length (matrix row length, in elements).
    # Governs how much of the peak fetch width is actually usable. None = long
    # enough not to matter.
    gemv_row_len: float | None = None
    # Which algorithm this benchmark runs. Precision floors and transcendental
    # counts are properties of the ALGORITHM, not of the benchmark's name --
    # keying tests or requirements off the name silently breaks when a benchmark
    # is re-specified (which is exactly what happened to ARM-MPC).
    #   "admm"  first-order splitting: box projection is min/max, tolerant of
    #           bf16 operands with fp32 accumulate
    #   "ip"    interior-point: real barrier transcendentals, KKT conditioning
    #           ~1/mu, needs fp64 or fp32 + mandatory iterative refinement
    #   "ekf"   covariance recursion: fp32 minimum, no fp16/bf16 path in any form
    #   "nn"    neural network inference
    algo: str = ""
    # Once-per-call matrix-VECTOR work that cannot be warm-started: for MPC this is
    # rebuilding q and the state-dependent bounds from the new x0. It sits on the
    # critical path before iteration 1. Previously omitted entirely.
    gemv_macs_per_call: float = 0.0
    # WORST-CASE iteration count that a hard deadline must survive: the cold /
    # active-set-change tail, not the p95 design point. `iters` sizes typical
    # energy; `iters_worst` sizes FEASIBILITY. Referenced in comments since v0 but
    # never actually implemented, which let the sizing sweep be run against the
    # typical case only -- and the typical case meets every deadline at 16 B/cycle,
    # so it cannot size anything. 0 = not characterised.
    iters_worst: int = 0

    # ---- mixed-precision operands (added 2026-07-30) -----------------------
    # Precision of the BULK matrix operands, when it differs from `prec`.
    # `prec` remains the precision of the accumulator and of the numerically
    # sensitive fraction below. None = single-precision-everywhere, i.e. no
    # change to any pre-existing row.
    #
    # WHY THIS IS A SEPARATE KNOB. perf_model/precision_study.py (MEASURED
    # numerics) found ADMM's cached triangular solve ~3x more error-sensitive
    # than its matrix operands, and found fp16 operands 10x better than bf16 at
    # the same width because MANTISSA bits dominate for these matrices. So the
    # honest datapath is: narrow operands on the bulk GEMVs, `prec` on the
    # solve, `prec` accumulate everywhere. One number per benchmark cannot say
    # that, and using one number is how a bandwidth-bound row gets quoted at the
    # wrong bandwidth.
    operand_prec: str | None = None
    # Fraction of `gemv_macs_per_iter` that must stay at `prec` (the cached
    # triangular solve). Derived by the benchmark builder, not hand-entered, and
    # asserted against the kernel decomposition's chained ops.
    solve_prec_frac: float = 0.0

    # ---- explicit GEMM shape (added 2026-07-30) ----------------------------
    # Override the crude cube-root proxy for the GEMM utilisation penalty.
    # None = representative_dim(), i.e. no change to any pre-existing row.
    gemm_dim_override: float | None = None
    # Scratchpad bytes per GEMM MAC, when reuse depth is set by a PROBLEM
    # dimension rather than by the tile. See KernelOp.bytes_per_mac.
    gemm_bytes_per_mac: float | None = None

    # ---- kernel decomposition (added 2026-07-30; see KernelOp) --------------
    # Per-iteration and once-per-call operation lists. Empty = not decomposed,
    # in which case the CM model falls back to the legacy flat scalar term.
    kernels: list = field(default_factory=list)
    kernels_call: list = field(default_factory=list)
    # Loop-carried SCALAR chain per iteration, in sequencer cycles: work whose
    # result a subsequent descriptor's PARAMETERS depend on, so it can be
    # overlapped by neither run-ahead nor speculation. For ADMM this is the
    # convergence test (two reduction results -> compare -> branch), evaluated
    # only every `check_term` iterations. For interior point it is the
    # fraction-to-boundary step size alpha, which scales the very next update.
    # For the EKF it is ZERO: a fixed schedule with no data-dependent control --
    # the determinism story, and it costs nothing to claim.
    # DELIBERATELY SMALL. The old flat 150 ops/iteration was standing in for
    # descriptor issue, not for a dependent scalar chain; conflating the two is
    # what hid the block-structured-solver exposure.
    serial_scalar_per_iter: float = 0.0

    @property
    def iters_wc(self) -> int:
        return max(self.iters, self.iters_worst)

    @property
    def macs_per_iter(self) -> float:
        return self.gemv_macs_per_iter + self.gemm_macs_per_iter

    def total_macs(self) -> float:
        return (self.macs_per_iter * self.iters
                + self.gemm_macs_per_call + self.gemv_macs_per_call)


def condensed_mpc(name, n_x, n_u, N, iters, rate, note, prec="fp32", refactor=False,
                  check_term=25, iters_worst=0, operand_prec=None):
    """Condensed QP solved by ADMM (OSQP-style), cached factorization.

    CRITICAL: every dominant operation here is matrix-VECTOR (P@x, triangular
    solves with a cached L, A@x, A.T@y). Batch size is 1 -- there is no second
    column to give the tile array reuse. So this workload is bounded by operand
    bandwidth, NOT by MAC count. `refactor=True` adds the once-per-call
    factorisation of a nonlinear/SQP problem, which IS real matrix-matrix work.
    """
    n_var = N * n_u
    # MEASURED CORRECTION (2026-07-29, perf_model/measure_osqp_baseline.py):
    # OSQP expresses a two-sided bound `l <= Ax <= u` as ONE constraint row, not two.
    # The previous `2*n_var + 2*N*n_x` overcounted rows by exactly 2x, inflating the
    # A@x and A.T@y terms in every solver benchmark. Verified against a real OSQP
    # problem instance: 320 rows actual vs 640 assumed.
    n_con = n_var + N * n_x               # input box + predicted-state box
    # AUDIT CORRECTIONS (2026-07-29, solver-numerics):
    #  (a) `P@x` was DOUBLE-COUNTED: P lives inside the cached factorisation. It
    #      appears only in the dual residual, at n^2/2 (P symmetric), amortised
    #      over `check_term` iterations.
    #  (b) A is NOT dense. In the condensed form A = [I ; Gamma]; the input-box
    #      rows are the identity, so A@x copies and A.T@y accumulates at ZERO
    #      MACs. Only the Gamma block costs anything.
    Annz = N * n_x * n_var                 # the Gamma block only
    gemv = (
        n_var * n_var                      # forward+back solve w/ cached L (~n^2)
        + 2 * Annz                         # Gamma@x and Gamma.T@y
        + (n_var * n_var / 2 + Annz) / check_term   # residuals, amortised
    )
    # Once-per-call, not warm-startable: rebuild q = Gamma^T Qbar (Phi x0 - xref)
    # and the state-dependent bounds. Previously omitted entirely.
    gemv_call = n_var * (N * n_x) + N * n_x * n_x
    # SQP: re-linearise and re-factorise once per call -> genuine GEMM/Cholesky.
    # The former `n_con*n_var**2` term was the cost of forming A^T A, which OSQP
    # explicitly does not do (it destroys sparsity) and no production MPC stack
    # does either. Removed.
    gemm_call = (n_var ** 3) / 3.0 if refactor else 0.0
    vec = 8 * (n_var + n_con) + 2 * n_con  # + box projection as min/max element ops
    # ---- CAPACITY. CORRECTED 2026-07-30 (descriptor v0.1 ruling R8) after the
    # functional simulator's MEASURED allocation plan contradicted this line.
    # It said 4*(n^2/2 + n^2/2 + Annz) = 100.0 KiB for QUAD-MPC; the plan measures
    # 163.8 KiB at fp32. Two errors, BOTH FLATTERING:
    #
    #  (1) P and L were charged at n^2/2 -- a PACKED triangular layout. Packed
    #      triangular addressing (row i at offset i(i+1)/2) is NOT EXPRESSIBLE with
    #      the base+leading-dimension AGU the descriptor format has. The capacity
    #      model was assuming a storage layout the hardware cannot address. They
    #      cost n^2 each. Undercount: 25600 B.
    #  (2) Working vectors and residual temporaries were charged ZERO. They are
    #      14136 B at fp32 for QUAD-MPC -- 10% of a 128 KiB scratchpad -- and were
    #      structurally invisible because the field is named `persistent_bytes`
    #      and a temporary is not persistent.
    #
    # What the line got RIGHT and keeps: A = [I;Gamma], so the identity block
    # stores nothing. That one is a strict win, since the identity rows were also
    # being multiplied by for no MACs.
    e = BYTES_PER_ELEM[prec]
    matrices = e * (n_var * n_var          # P, dense square (symmetric-lower)
                    + n_var * n_var        # L, dense square (factor in place)
                    + Annz)                # Gamma only
    # Iterates x,y,z; per-tick q and interleaved (lo,hi) bounds; and the
    # temporaries the descriptor program actually names. Counted from the
    # simulator's allocation plan, not estimated.
    vectors = e * (2 * n_con               # interleaved bounds
                   + 5 * n_con             # y, z, t, t_dual, and one of zt/Ax
                   + 2 * n_con             # rp, and the second of zt/Ax
                   + 6 * n_var             # x, q, rhs, xt, Px, Aty
                   + 2 * n_var             # rd, and one spare n-vector
                   + 8)                    # convergence scalars, bank-aligned
    persistent = matrices + vectors
    delta = 4 * (n_x + 3 * n_var)                          # state + reference
    # ---- kernel decomposition: what codegen has to ISSUE per ADMM iteration.
    # This is the FEW-LARGE-OPERATIONS end of the spectrum, which is the point:
    # the two Gamma products are single descriptors covering ~19k MACs each. The
    # only fine-grained part is the pair of triangular solves against the cached
    # factor, which ARE a dependency chain: one diagonal block at a time, and
    # each block's descriptor cannot start until the previous block's results are
    # written back. n_desc=2 per chain step = diagonal-block solve + trailing
    # panel update. MACs sum exactly to the `gemv` expression above.
    resid_macs = (n_var * n_var / 2 + Annz) / check_term
    kern = [
        KernelOp("Gamma^T y (rhs assembly)", "gemv", macs=Annz, n_desc=1),
        KernelOp("L forward solve (cached factor)", "gemv", macs=n_var * n_var / 2,
                 n_desc=2, chain_dim=n_var),
        KernelOp("L^T back solve", "gemv", macs=n_var * n_var / 2,
                 n_desc=2, chain_dim=n_var),
        KernelOp("Gamma x", "gemv", macs=Annz, n_desc=1),
        KernelOp("residual re-evaluation (amortised)", "gemv", macs=resid_macs,
                 n_desc=3.0 / check_term),
        KernelOp("z projection (min/max)", "vec", vec_ops=2 * n_con, n_desc=1),
        KernelOp("y update (axpy)", "vec", vec_ops=2 * n_con, n_desc=1),
        KernelOp("x over-relaxation", "vec", vec_ops=2 * n_var, n_desc=1),
        KernelOp("norms and misc elementwise", "vec",
                 vec_ops=(8 * (n_var + n_con) + 2 * n_con) - (4 * n_con + 2 * n_var),
                 n_desc=3),
    ]
    kern_call = [
        KernelOp("q = Gamma^T Qbar (Phi x0 - xref)", "gemv", macs=n_var * (N * n_x),
                 n_desc=2),
        KernelOp("state-dependent bounds", "gemv", macs=N * n_x * n_x, n_desc=1),
    ]
    return Benchmark(
        name=name, kind="solver", note=note,
        gemv_macs_per_iter=gemv, gemm_macs_per_iter=0.0,
        gemm_macs_per_call=gemm_call,
        vector_ops_per_iter=vec,
        # AUDIT CORRECTION: was `2*n_con`. The ADMM box projection is min/max --
        # ZERO divides, sqrt or exp. Because the CPU model charges 12 cycles per
        # transcendental against the CM's 4/vector-lane, those phantom operations
        # penalised the CPU ~3x harder than the CM and were single-handedly
        # responsible for the apparent latency advantage on this benchmark.
        transcendental_per_iter=0.0,
        scalar_ops_per_iter=120,             # convergence test + loop control
        iters=iters, calls_per_sec=rate,
        deadline_us=1e6 / rate,
        persistent_bytes=persistent, delta_bytes=delta, prec=prec,
        # Every dominant GEMV here walks a row of length n_var: P@x rows, A rows,
        # and triangular-solve rows (n_var/2 average, rounded up to n_var here --
        # slightly optimistic, and flagged as such).
        gemv_row_len=n_var,
        gemv_macs_per_call=gemv_call, algo="admm",
        iters_worst=iters_worst,
        # Mixed-precision operands. `solve_prec_frac` is the cached triangular
        # solve's share of the per-iteration GEMV work -- DERIVED from the same
        # expression that produced `gemv`, never hand-entered, because the whole
        # point is that the sensitive fraction and the total must stay consistent.
        # precision_study.py measures the solve as ~3x more error-sensitive than
        # the operands, so it is the fraction that must NOT be narrowed.
        operand_prec=operand_prec,
        solve_prec_frac=(n_var * n_var) / gemv,
        kernels=kern, kernels_call=kern_call,
        # OSQP evaluates termination every `check_term` iterations. The chain is
        # reduction-result -> scalar register -> two compares -> branch; call it
        # ~25 cycles of true latency, amortised. [EST]
        serial_scalar_per_iter=25.0 / check_term,
    )


def fcs_mpc(name, n_x, n_y, n_u, levels, Np, rate, note, prec="fp32",
            fuse_accum_init=False, tile_hint=12):
    """Finite-control-set (direct) MPC as actually deployed in power electronics.

    ADDED 2026-07-30. This is the re-specification of BENCHMARKS.md #3. The
    previous definition -- a continuous-control-set condensed QP solved by ADMM
    at 20 kHz -- described a formulation that is used on research/rapid-
    prototyping platforms but is not what ships in the socket, and it drove four
    separate architecture conclusions. See the citations in PHASE0_FINDINGS.

    THE COMPUTATIONAL SHAPE, and every part of it differs from ADMM:

      * NO ITERATION. The op count is fixed and known at codegen time. There is
        no convergence tail, so `iters_worst == iters == 1`. This is the same
        determinism story as the EKF and it is stronger than any ADMM row's.
      * NO LOOP-CARRIED DEPENDENCY. Candidates are independent; there is no
        Cholesky, no triangular solve, no dependency chain at all. Every
        `chain_dim` below is zero, which is the whole reason the run-ahead and
        chain-blocking exposures vanish on this row.
      * THE CANDIDATE SET IS A BATCH DIMENSION. |U|^Np sequences share the same
        A, B, C, so what was matrix-VECTOR work at batch 1 becomes matrix-MATRIX
        work at batch |U|^Np. This is the operand reuse the settled finding says
        batch-1 solver loops do not have.

    AND THE ADVERSARIAL PART, which is where the flattering reading dies: the
    reuse depth is NOT the tile dimension. The dominant product is
    (Q^(1/2) C) @ X2, with C an n_y x n_x matrix -- so each predicted state
    element is consumed exactly n_y times, and n_y = 2. Arithmetic intensity is
    therefore `elem * (1/n_y + 1/tile)` bytes per MAC, ~2.33 B/MAC at fp32
    against 0.67 B/MAC for a tile-limited GEMM. The batch buys a 1.7x traffic cut
    over pure GEMV, not the 12x the array size suggests. **FCS-MPC is still
    bandwidth-bound.** Modelled via `gemm_bytes_per_mac`, and the GEMM bandwidth
    ceiling had to be added to the model for it to be expressible at all.

    Deployed-practice parameters (see PHASE0_FINDINGS for citations):
      levels=3  three-level NPC -> U = {-1,0,1}^3, |U| = 27 switching states
      levels=2  two-level       -> U = {-1,1}^3,   |U| =  8
      Np=1      the horizon exhaustive search is actually practical at
      Np=2      the horizon that needs an FPGA or a sphere decoder today

    `fuse_accum_init=True` prices two accumulator features -- broadcast/bias
    initialisation of the accumulator, and square-accumulate -- which move the
    two largest vector terms into the array. It is off by default because it is
    an RTL feature we have not committed to, and the difference between the two
    settings is the size of the finding.
    """
    nU = levels ** n_u                     # switching states per step
    n_seq = nU ** Np                       # candidate sequences
    e = BYTES_PER_ELEM[prec]
    # ONE arithmetic-intensity figure for the whole GEMM group, set by the
    # DOMINANT product (Q^(1/2)C @ X, reuse depth n_y) and applied uniformly.
    # The two cheaper products (B @ U and A @ X, reuse depth n_x) are 11% of the
    # MACs; charging them at the dominant rate is ~3% pessimistic on cycles and
    # ~7% on GEMM traffic. That is deliberate: a per-op figure would make the
    # aggregate and kernel-decomposition cycle counts disagree by 4% (min() over
    # array-vs-bandwidth does not commute with a MAC-weighted mean), and the
    # invariant that those two paths agree exactly is worth more than 3%.
    bpm = e * (1.0 / n_y + 1.0 / tile_hint)

    # ---- GEMM work, per stage. Stage s enumerates nU^s candidates. ----------
    # free response A x(k): a single GEMV, shared by every candidate.
    gemv = n_x * n_x
    gemm = 0.0
    vec = 0.0
    kern = []
    kern.append(KernelOp("free response A x(k)", "gemv", macs=n_x * n_x,
                         n_desc=1))
    # B @ U for the nU DISTINCT switch positions -- computed once and reused by
    # every parent, which is why this term does not scale with nU^Np.
    bu = n_x * n_u * nU
    gemm += bu
    kern.append(KernelOp("forced response B @ U (nU distinct columns)", "gemm",
                         macs=bu, n_desc=1, bytes_per_mac=bpm))
    for s in range(1, Np + 1):
        cand = nU ** s
        if s > 1:
            # propagate every stage-(s-1) state: A @ X_{s-1}. Reuse depth n_x.
            prop = n_x * n_x * (nU ** (s - 1))
            gemm += prop
            kern.append(KernelOp(f"stage {s}: propagate A @ X{s-1}", "gemm",
                                 macs=prop, n_desc=1,
                                 bytes_per_mac=bpm))
            # broadcast-add the nU precomputed Bu columns to each parent state.
            # FOLDABLE into the accumulator as a bias/broadcast initialisation.
            if not fuse_accum_init:
                vec += cand * n_x
                kern.append(KernelOp(f"stage {s}: broadcast-add B u", "vec",
                                     vec_ops=cand * n_x, n_desc=1))
        # outputs and weighted residual: (Q^(1/2) C) @ X_s. THE DOMINANT TERM,
        # and its reuse depth is n_y, not the tile dimension.
        outp = n_y * n_x * cand
        gemm += outp
        kern.append(KernelOp(f"stage {s}: (Q^1/2 C) @ X{s}", "gemm", macs=outp,
                             n_desc=1,
                             bytes_per_mac=bpm))
        # square-and-accumulate the residual into a per-candidate cost.
        # FOLDABLE into the array as a square-accumulate pass.
        if not fuse_accum_init:
            vec += cand * 2 * n_y
            kern.append(KernelOp(f"stage {s}: ||r||^2 accumulate", "vec",
                                 vec_ops=cand * 2 * n_y, n_desc=1))
    # switching-effort penalty table over the nU distinct transitions, then two
    # table adds per sequence to assemble the sequence cost.
    vec += nU * 3 * n_u
    kern.append(KernelOp("switching-effort penalty table", "vec",
                         vec_ops=nU * 3 * n_u, n_desc=1))
    vec += n_seq * Np
    kern.append(KernelOp("assemble sequence costs", "vec",
                         vec_ops=n_seq * Np, n_desc=1))
    # argmin over every candidate at every stage.
    n_cand_total = sum(nU ** s for s in range(1, Np + 1))
    vec += n_cand_total
    kern.append(KernelOp("argmin over candidates", "vec",
                         vec_ops=n_cand_total, n_desc=2))

    # Persistent: A, B, Q^(1/2)C, the nU precomputed B u columns, the penalty
    # table, weights. Tiny -- and that is itself a finding about residency.
    persistent = e * (n_x * n_x + n_x * n_u + n_y * n_x + n_x * nU + nU + 8)
    delta = e * (n_x + n_u + n_y)          # x(k), u(k-1), y_ref

    return Benchmark(
        name=name, kind="solver", note=note,
        gemv_macs_per_iter=gemv, gemm_macs_per_iter=gemm,
        gemm_macs_per_call=0.0,
        vector_ops_per_iter=vec,
        # No barrier, no projection, no normalisation: cost evaluation is
        # multiply-add and selection is compare. ZERO transcendentals, and this
        # is the same trap that produced a 3x phantom CPU penalty once already.
        transcendental_per_iter=0.0,
        # Enumeration bookkeeping on a CPU. Deliberately modest (200, in line
        # with the other rows) rather than one branch per candidate, because a
        # competently vectorised CPU implementation amortises the loop -- this is
        # the CONSERVATIVE choice and it is conservative against us.
        scalar_ops_per_iter=200,
        iters=1, calls_per_sec=rate, deadline_us=1e6 / rate,
        persistent_bytes=persistent, delta_bytes=delta, prec=prec,
        gemv_row_len=n_x,
        gemv_macs_per_call=0.0, algo="fcs",
        # Fixed op count. Worst case IS the typical case.
        iters_worst=1,
        kernels=kern,
        # No convergence test, no step size, no active set. Nothing is
        # loop-carried because there is no loop.
        serial_scalar_per_iter=0.0,
        # Utilisation is set by the candidate batch, which is long. The array is
        # NOT what limits this row; arithmetic intensity is.
        gemm_dim_override=float(n_seq),
        # Weighted mean of the per-op figures, so the aggregate path and the
        # kernel path agree. Asserted by a test.
        gemm_bytes_per_mac=bpm,
    )


def sqp_rti_mpc(name, n_x, n_u, N, ip_iters, rate, note, prec="fp64",
                iters_worst=0):
    """Nonlinear MPC as actually deployed: acados SQP-RTI + HPIPM.

    RE-SPECIFIED 2026-07-30 (solver-numerics audit). The previous specification --
    60 SQP iterations of a fully-condensed QP that forms A^T A -- overstated the
    work ~31x and matched no production stack. Deployed practice is the real-time
    iteration scheme (Diehl et al.): ONE SQP step per sampling instant, split into
    a preparation phase (re-linearise, form the QP) and a feedback phase (solve
    one QP). acados defaults to partial condensing with HPIPM's block-structured
    Riccati recursion, whose cost is O(N*(n_x+n_u)^3) per interior-point iteration
    rather than O((N*n_u)^3) for full condensing (Frison & Diehl, IFAC 2020).

    Precision: fp64 by default, because an interior-point method's KKT condition
    number grows as 1/mu and reaches ~1e12 near convergence. fp32 requires
    mandatory iterative refinement, which costs extra SEQUENTIAL steps. Modelling
    this in fp32 without refinement would be a flattering assumption.
    """
    nb = n_x + n_u                     # stage block dimension -- the key quantity
    # Per interior-point iteration: backward Riccati factorisation per stage is
    # genuine matrix-matrix work, but on SMALL blocks (nb ~ 21), so tile
    # utilisation is poor. Forward/backward substitution is matrix-vector.
    gemm_iter = N * (nb ** 3) / 3.0
    gemv_iter = 2 * N * nb * nb
    # Preparation phase, once per call: dynamics Jacobians and sensitivity
    # propagation. Cannot be warm-started -- the linearisation point moved.
    gemm_call = N * n_x * nb * nb
    gemv_call = 2 * N * nb * n_x
    e = BYTES_PER_ELEM[prec]
    # ---- CAPACITY. CORRECTED 2026-07-30, same error class as the ADMM rows
    # (descriptor v0.1 ruling R8): this line was `e * (N*nb*nb + N*n_x*nb)`, which
    # counted the stage Hessians and the [A B] blocks and NOTHING ELSE -- no
    # Riccati factor, no feedback gains, no slack/dual vectors, no trajectory.
    # Fixed here rather than merely flagged, because having just corrected the
    # identical undercount on QUAD-MPC it would be incoherent to leave a known
    # 1.9x one in place -- and because uncorrected it made ARM-MPC appear to FIT
    # the new 192 KB scratchpad, which is a flattering conclusion resting on a
    # known error.
    #
    # The conclusion does NOT change: ARM-MPC overflows at 128, 192 and 256 KB and
    # always did. That was already accepted explicitly -- it runs at 250 Hz with
    # ~10x deadline margin and its streaming demand is ~0.03 GB/s -- so it does
    # not drive capacity. QUAD-MPC does. What changes is only that the number is
    # now honest about it.
    persistent = e * (
        N * nb * nb                 # stage Hessians
        + N * nb * nb               # Riccati factors (NOT in place: the Hessian
                                    # is needed again on the forward recursion)
        + N * n_x * nb              # [A B] sensitivity blocks
        + N * n_u * n_x             # feedback gains K
        + 4 * N * nb                # slacks and their duals, both bounds
        + N * nb                    # state/input trajectory
    )
    delta = e * (n_x + N * n_x)
    # ---- kernel decomposition: the MANY-SMALL-OPERATIONS end of the spectrum,
    # and the reason this analysis exists. HPIPM's backward Riccati recursion is
    # per-stage, and each stage is a sequence of BLASFEO-level block operations
    # on nb x nb blocks. Following Frison & Diehl (IFAC 2020) the per-stage
    # factorisation is: trmm (Lxx^T [A B]), syrk (form the stage Hessian), potrf
    # (Cholesky of it), trsm (extract the feedback gain), then vector backward
    # substitution and forward recursion.
    #
    # THE SPLIT OF nb^3/3 ACROSS THOSE FOUR IS [EST], derived from their true
    # relative op counts (trmm n_x^2*nb/2, syrk nb^2*n_x/2, potrf nb^3/6, trsm
    # the gain block) normalised so the group SUMS TO the benchmark's declared
    # N*nb^3/3. CONSERVATIVE NOTE, stated because it runs against us: the true
    # HPIPM per-stage count at nb=21 is ~2.2x nb^3/3, so this decomposition puts
    # FEWER MACs behind each descriptor than reality and therefore OVERSTATES
    # sequencer pressure. Fixing the benchmark's GEMM count would make the
    # sequencer picture better, not worse.
    #
    # potrf and trsm carry chain_dim=nb: a blocked Cholesky advances one diagonal
    # block at a time and cannot be issued as one dataflow descriptor.
    g = nb ** 3 / 3.0
    kern = [
        KernelOp("Riccati: Lxx^T [A B] (trmm)", "gemm", macs=0.30 * g,
                 n_desc=1, repeat=N),
        KernelOp("Riccati: stage Hessian (syrk)", "gemm", macs=0.45 * g,
                 n_desc=1, repeat=N),
        KernelOp("Riccati: Cholesky (potrf)", "gemm", macs=0.15 * g,
                 n_desc=2, chain_dim=nb, repeat=N),
        KernelOp("Riccati: feedback gain (trsm)", "gemm", macs=0.10 * g,
                 n_desc=1, chain_dim=nb, repeat=N),
        KernelOp("backward substitution", "gemv", macs=nb * nb, n_desc=2,
                 repeat=N),
        KernelOp("forward state recursion", "gemv", macs=nb * nb, n_desc=2,
                 repeat=N),
        KernelOp("slack / complementarity update", "vec", vec_ops=12 * nb,
                 n_desc=2, repeat=N),
        KernelOp("barrier divides", "trans", trans_ops=2 * nb, n_desc=1,
                 repeat=N),
    ]
    kern_call = [
        KernelOp("preparation: dynamics Jacobians", "gemm", macs=n_x * nb * nb,
                 n_desc=2, repeat=N),
        KernelOp("preparation: gradient and bounds", "gemv", macs=2 * nb * n_x,
                 n_desc=2, repeat=N),
    ]
    return Benchmark(
        name=name, kind="solver", note=note,
        gemv_macs_per_iter=gemv_iter, gemm_macs_per_iter=gemm_iter,
        gemm_macs_per_call=gemm_call,
        vector_ops_per_iter=12 * N * nb,
        transcendental_per_iter=2 * N * nb,   # IP barrier: real divides, unlike ADMM
        scalar_ops_per_iter=150,
        iters=ip_iters, calls_per_sec=rate, deadline_us=1e6 / rate,
        persistent_bytes=persistent, delta_bytes=delta, prec=prec,
        gemv_row_len=nb,
        gemv_macs_per_call=gemv_call, algo="ip",
        iters_worst=iters_worst,
        kernels=kern, kernels_call=kern_call,
        # Interior point: the fraction-to-boundary step size alpha is a reduction
        # over all complementarity pairs, then a divide, then it SCALES the very
        # next update -- a genuine loop-carried dependency that run-ahead cannot
        # hide. Reduction drain + divide + compares ~ 60 cycles, once per IP
        # iteration. Plus the barrier parameter mu update. [EST]
        serial_scalar_per_iter=60.0,
    )


def ekf(name, n, m, rate, note, prec="fp32"):
    """EKF update: propagation + Joseph-form covariance + gain solve.

    Unlike ADMM-MPC, the EKF covariance recursion is genuinely matrix-MATRIX
    (A@P@A.T, Joseph form), so the tile array is usable here. Only the Cholesky
    has limited parallelism and is counted as bandwidth-bound.
    """
    # AUDIT CORRECTIONS (2026-07-30): the Joseph form was counted as 2*n^2*m,
    # omitting its two DOMINANT n^3 products entirely -- (I-KH) is n^2*m to form,
    # then (I-KH)P is n^3 and [.](I-KH)^T another n^3/2. Undercounted 2.5-3.9x.
    # Also: F@P@F.T is 1.5n^3 not 2n^3 (the second product is symmetric),
    # H@P@H.T is n*m^2/2 (S symmetric), and Cholesky is m^3/6 not m^3/3 (that is LU).
    gemm = (
        1.5 * n ** 3           # F@P@F.T  (FP is n^3, (FP)F.T symmetric -> n^3/2)
        + n * n * m            # P@H.T
        + n * m * m / 2.0      # H@P@H.T, S symmetric
        + n * m * m            # gain solve
        + (n * n * m + 1.5 * n ** 3 + n * m * m + n * n * m / 2.0)   # Joseph form
    )
    gemv = m ** 3 / 6.0        # Cholesky: sequential, poor tile utilisation
    persistent = 4 * (3 * n * n + n * m)
    delta = 4 * (m + n)
    # ---- kernel decomposition. A FIXED schedule: ~10 large block products plus
    # two short dependency chains (the m x m Cholesky of the innovation
    # covariance, and the triangular solve for the gain). No data-dependent
    # control at all, hence serial_scalar_per_iter = 0. MACs sum exactly to the
    # `gemm` and `gemv` expressions above.
    kern = [
        KernelOp("F@P", "gemm", macs=n ** 3, n_desc=1),
        KernelOp("(F P)@F^T (symmetric)", "gemm", macs=0.5 * n ** 3, n_desc=1),
        KernelOp("P@H^T", "gemm", macs=n * n * m, n_desc=1),
        KernelOp("H@(P H^T), S symmetric", "gemm", macs=n * m * m / 2.0, n_desc=1),
        KernelOp("gain solve K S = P H^T (trsm)", "gemm", macs=n * m * m,
                 n_desc=2, chain_dim=m),
        KernelOp("Joseph: form (I-KH)", "gemm", macs=n * n * m, n_desc=1),
        KernelOp("Joseph: (I-KH)@P", "gemm", macs=n ** 3, n_desc=1),
        KernelOp("Joseph: [.]@(I-KH)^T", "gemm", macs=0.5 * n ** 3, n_desc=1),
        KernelOp("Joseph: K R K^T", "gemm", macs=n * m * m, n_desc=1),
        KernelOp("Joseph: symmetrise", "gemm", macs=n * n * m / 2.0, n_desc=1),
        KernelOp("Cholesky of S (potrf)", "gemv", macs=m ** 3 / 6.0,
                 n_desc=2, chain_dim=m),
        KernelOp("covariance symmetrise / state update", "vec",
                 vec_ops=6 * n * n, n_desc=3),
        KernelOp("innovation normalisation", "trans", trans_ops=m, n_desc=1),
    ]
    return Benchmark(
        name=name, kind="solver", note=note,
        gemv_macs_per_iter=gemv, gemm_macs_per_iter=gemm,
        gemm_macs_per_call=0.0,
        vector_ops_per_iter=6 * n * n,
        transcendental_per_iter=m, scalar_ops_per_iter=60, algo="ekf",
        # The covariance recursion is a FIXED op count -- no iteration tail at
        # all. Worst case == typical case. This is the strongest determinism
        # story in the set and it costs nothing to claim.
        iters_worst=1,
        iters=1, calls_per_sec=rate, deadline_us=1e6 / rate,
        persistent_bytes=persistent, delta_bytes=delta, prec=prec,
        gemv_row_len=m,       # Cholesky of the m x m innovation covariance
        kernels=kern, serial_scalar_per_iter=0.0,
    )


def slm_decode(name, params_b, hidden, layers, seq, rate, note, prec="int8",
               weight_prec=None):
    """Autoregressive decode, batch 1: weight-bound GEMV + attention over KV cache.

    `prec` is the ACTIVATION / KV format; `weight_prec` is the weight format and
    defaults to `prec`. Setting `weight_prec="int4"` with `prec="int8"` is
    weight-only 4-bit quantisation, i.e. what GPTQ/AWQ actually ship, and it maps
    onto the existing `operand_prec` / `solve_prec_frac` mixed-precision
    machinery with no new field: the weight GEMVs are the "bulk operand" and the
    attention GEMVs over the KV cache are the fraction that stays at `prec`.

    TWO CORRECTIONS LANDED HERE 2026-08-05, BOTH FLATTERING BEFORE THE FIX:

      * THE KV CACHE WAS NOT COUNTED. `persistent` was
        `params * 1e9 * (1 if int8 else 2)`, with a comment reading "weights +
        KV" -- the comment claimed a term the expression did not contain. For the
        frozen 1.1B / 22L / 512-token row the KV cache is 46.1 MB against 1.10 GB
        of weights (+4.2% of external traffic), and at the ~0.16 G-parameter
        envelope boundary it is 29% -- i.e. it matters precisely where the
        envelope is drawn. Omitting it understated the external bytes on BOTH
        sides of the comparison, which pushes perf/W AWAY from its true asymptote
        of 1.0 and makes the deadline look easier than it is.
      * WEIGHT WIDTH WAS `1 if int8 else 2`, so any format wider than int8 became
        2 B and int4 would have become 2 B. It now goes through BYTES_PER_ELEM.

    KV IS NOT QUANTISED TO 4 BITS HERE. Weight-only 4-bit quantisation leaves the
    KV cache at 8 or 16 bits, which is what GPTQ/AWQ do; 4-bit KV is a separate
    (and much more accuracy-sensitive) technique and claiming it here would halve
    a term we have no basis to halve.
    """
    weight_prec = weight_prec or prec
    # Batch-1 decode is pure GEMV: each weight is used exactly once per token.
    attn_macs = 2 * layers * seq * hidden
    gemv = params_b * 1e9 + attn_macs
    weight_bytes = params_b * 1e9 * BYTES_PER_ELEM[weight_prec]
    kv_bytes = attn_macs * BYTES_PER_ELEM[prec]      # 2 (K and V) already in attn_macs
    persistent = weight_bytes + kv_bytes
    delta = 4 * hidden
    # Kernel decomposition: 7 weight GEMVs per transformer layer (q,k,v,o,gate,
    # up,down) plus 2 attention GEMVs. The FAR end of the granularity spectrum --
    # each descriptor commands millions of MACs -- which is the control case
    # showing the exposure is specific to block-structured solvers.
    kern = [
        KernelOp("layer weight GEMVs (q,k,v,o,gate,up,down)", "gemv",
                 macs=params_b * 1e9 / layers, n_desc=7, repeat=layers),
        KernelOp("attention over KV cache", "gemv", macs=2 * seq * hidden,
                 n_desc=2, repeat=layers),
        KernelOp("norms / residual adds / SwiGLU", "vec", vec_ops=hidden * 12,
                 n_desc=4, repeat=layers),
        KernelOp("softmax / rsqrt", "trans", trans_ops=hidden + seq, n_desc=2,
                 repeat=layers),
    ]
    return Benchmark(
        name=name, kind="ml", note=note,
        gemv_macs_per_iter=gemv, gemm_macs_per_iter=0.0, gemm_macs_per_call=0.0,
        vector_ops_per_iter=layers * hidden * 12,
        transcendental_per_iter=layers * (hidden + seq), scalar_ops_per_iter=200,
        algo="nn",
        iters=1, calls_per_sec=rate, deadline_us=1e6 / rate,
        persistent_bytes=persistent, delta_bytes=delta, prec=prec,
        gemv_row_len=hidden,   # weight-matrix rows are long: fetch width is fine
        kernels=kern, serial_scalar_per_iter=0.0,
        # Weight-only quantisation. None when weight_prec == prec, so the
        # single-precision row is bit-identical to before this change.
        operand_prec=None if weight_prec == prec else weight_prec,
        solve_prec_frac=0.0 if weight_prec == prec else attn_macs / gemv,
    )


def cnn_infer(name, gmacs, weights_mb, rate, note, prec="int8", blocks=12):
    # Conv/attention as GEMM with real operand reuse -- the tile array's home turf.
    kern = [
        KernelOp("block GEMMs (qkv, attn, proj, mlp x2)", "gemm",
                 macs=gmacs * 1e9 / blocks, n_desc=6, repeat=blocks),
        KernelOp("norms / residual / GELU", "vec",
                 vec_ops=gmacs * 1e9 * 0.02 / blocks, n_desc=4, repeat=blocks),
        KernelOp("softmax", "trans", trans_ops=1e5 / blocks, n_desc=2,
                 repeat=blocks),
    ]
    return Benchmark(
        name=name, kind="ml", note=note,
        gemv_macs_per_iter=0.0, gemm_macs_per_iter=gmacs * 1e9,
        gemm_macs_per_call=0.0,
        vector_ops_per_iter=gmacs * 1e9 * 0.02,
        transcendental_per_iter=1e5, scalar_ops_per_iter=500, algo="nn",
        iters=1, calls_per_sec=rate, deadline_us=1e6 / rate,
        persistent_bytes=weights_mb * 1e6, delta_bytes=224 * 224 * 3, prec=prec,
        kernels=kern, serial_scalar_per_iter=0.0,
    )


# --- The frozen Phase-0 benchmark set -------------------------------------
BENCHMARKS = [
    # MEASURED (perf_model/measure_osqp_baseline.py): real OSQP on this exact problem,
    # warm-started closed-loop with reference steps and disturbances, needs p50=12 /
    # p95=14 iterations at eps=1e-3, rising to p95=27 at eps=1e-6, with a 56-70
    # iteration COLD solve. The previous assumption of 40 was ~3x too high for typical
    # warm-started operation. 15 is the p95 design point at control-loop tolerance;
    # `iters_worst` carries the cold-solve tail that a hard deadline must survive.
    # iters_worst=74 is MEASURED: the cold / worst-tick solve on the coupled plant
    # at eps=1e-6 (results_osqp_measured.txt: cold 73, max 74). This -- not the p95
    # design point -- is what a hard deadline must be sized against, and it is the
    # only thing in the current set that constrains scratchpad bandwidth at all.
    # check_term=1 CORRECTED 2026-07-30 from OSQP's library default of 25.
    # The default is wrong for this workload and the error was flattering: this
    # problem converges in 6-14 iterations (measured), so checking every 25th
    # means never checking -- the solver would always run to max_iterations.
    # The validated functional simulator checks every iteration, and at
    # check_term=1 this decomposition emits 31.0 descriptors per checked
    # iteration against the simulator's MEASURED 31. That agreement is the
    # cross-validation; it did not hold at 25 (28.1 descriptors).
    # Cost of the correction: latency 36.0 -> 49.0 us, energy 6.15 -> 8.90 uJ.
    # perf/W barely moves (12.55x -> 12.33x) because the CPU baseline pays the
    # same extra residual work -- but the LATENCY is 36% worse and that is real.
    condensed_mpc("QUAD-MPC",  n_x=12, n_u=4, N=20, iters=15, rate=1000,
                  iters_worst=74, check_term=1,
                  note="Quadrotor MPC, 12 states, N=20, ADMM p95=15 iters (measured), 1 kHz"),
    # RE-SPECIFIED 2026-07-30: acados SQP-RTI (one SQP step/tick) + HPIPM partial
    # condensing, 15 interior-point iterations, fp64. Rate lowered 500 -> 250 Hz:
    # published 7-DoF manipulator NMPC runs at 100-250 Hz above a 1 kHz joint loop,
    # so 500 Hz was above deployed practice.
    # iters_worst=30: HPIPM is run with a hard max_iter, so the tail is BOUNDED BY
    # CONFIGURATION rather than by the problem -- 2x the design point is generous.
    # [EST], not measured.
    sqp_rti_mpc("ARM-MPC", n_x=14, n_u=7, N=30, ip_iters=15, rate=250,
                iters_worst=30,
                note="7-DoF manipulator NMPC: acados SQP-RTI + HPIPM, 15 IP iters, "
                     "fp64, 250 Hz"),
    # RE-SPECIFIED 2026-07-30 (v1.2): finite-control-set / direct MPC, which is
    # what the socket actually runs. The v1.1 definition was a continuous-
    # control-set condensed QP solved by ADMM at 20 kHz. Rate raised 20 -> 40 kHz
    # (Ts = 25 us) because deployed direct MPC needs fs ~ 100x fsw, so 20 kHz was
    # BELOW deployed practice -- i.e. the re-spec makes the deadline HARDER, not
    # easier. The retired ADMM definition is kept as PWR_MPC_CCS below.
    fcs_mpc("PWR-MPC", n_x=4, n_y=2, n_u=3, levels=3, Np=2, rate=40000,
            note="Motor-drive FCS-MPC: 3-level NPC, |U|=27, Np=2 exhaustive "
                 "(729 sequences), Ts=25 us / 40 kHz"),
    ekf("EKF-24", n=24, m=12, rate=1000,
        note="INS/GNSS 24-state EKF, 12 measurements, 1 kHz -- smallest problem"),
    ekf("EKF-96", n=96, m=24, rate=200,
        note="Large-state estimator (SLAM-ish), 96 states, 200 Hz"),
    slm_decode("SLM-DECODE", params_b=1.1, hidden=2048, layers=22, seq=512, rate=20,
               note="1.1B SLM INT8 decode, batch 1, 20 tok/s target"),
    cnn_infer("VIT-TINY", gmacs=1.26, weights_mb=5.7, rate=30,
              note="ViT-Tiny 224x224 INT8, 30 fps -- incumbent NPU home turf"),
]


# ---------------------------------------------------------------------------
# Off-set rows kept for the audit trail. NOT part of the frozen seven; nothing
# may be provisioned from them without saying so out loud.
# ---------------------------------------------------------------------------

# The RETIRED PWR-MPC definition. Continuous-control-set MPC does appear in
# power electronics -- indirect MPC with a carrier-based modulator is solved as a
# QP with qpOASES or ODYS -- but the published implementations run on dSpace-class
# rapid-prototyping hardware, not in a shipping drive, and the deployed-majority
# formulation is finite-control-set. Retained because (a) the CCS socket is real
# even if smaller, and (b) two architecture requirements were justified from this
# row and the audit trail has to show what happens to them.
PWR_MPC_CCS = condensed_mpc(
    "PWR-CCS", n_x=4, n_u=3, N=10, iters=12, rate=20000, iters_worst=48,
    check_term=1,
    note="RETIRED v1.1 row: motor-drive continuous-control-set MPC via ADMM, "
         "20 kHz. Research-platform practice, not shipping product")

# One-step FCS-MPC: what an MCU ships today. Deliberately in the set of things we
# model because it is the case we should LOSE, and a model that cannot report
# that is not evidence.
PWR_MPC_FCS1 = fcs_mpc(
    "PWR-FCS1", n_x=4, n_y=2, n_u=3, levels=3, Np=1, rate=40000,
    note="One-step FCS-MPC, 3-level, 27 candidates, 40 kHz -- the deployed "
         "baseline, and the case where the host MCU should win")

# Two-level converter at 100 kHz (Ts = 10 us), Np = 2: the fastest deployed
# direct-MPC sampling interval in the survey, 64 sequences.
PWR_MPC_FCS_FAST = fcs_mpc(
    "PWR-FAST", n_x=4, n_y=2, n_u=3, levels=2, Np=2, rate=100000,
    note="Two-level FCS-MPC, |U|=8, Np=2 (64 sequences), Ts=10 us / 100 kHz")

# Same as the frozen row but with the two accumulator features (broadcast/bias
# init and square-accumulate) that fold the dominant vector terms into the array.
PWR_MPC_FUSED = fcs_mpc(
    "PWR-FUSE", n_x=4, n_y=2, n_u=3, levels=3, Np=2, rate=40000,
    fuse_accum_init=True,
    note="Frozen PWR-MPC with accumulator broadcast-init + square-accumulate")

OFF_SET = [PWR_MPC_CCS, PWR_MPC_FCS1, PWR_MPC_FCS_FAST, PWR_MPC_FUSED]


# ---------------------------------------------------------------------------
# Argmind model
# ---------------------------------------------------------------------------

def gemm_macs_per_cycle_bounded(cfg: ArgmindCfg, prec: str, effective_dim: float,
                                bytes_per_mac: float | None = None) -> float:
    """Achieved GEMM MAC/cycle, bounded by BOTH the array and operand fetch.

    ADDED 2026-07-30. The model previously applied a bandwidth ceiling only to
    GEMV, on the reasoning that tile reuse always keeps GEMM under it. That
    reasoning is sound at `elem*(1/Tm + 1/Tn)` bytes per MAC on the CM-100
    (0.67 B/MAC fp32 -> 192 MAC/cycle at 128 B/cycle, well above the 72 fp32
    array rate), and a test asserts the new bound is INACTIVE on every row of the
    frozen set that existed before this change. But it is not sound in general:
    the moment a GEMM's reuse depth is set by a problem dimension rather than by
    the tile -- FCS-MPC's dominant product reuses each predicted state exactly
    n_y = 2 times, because there are two controlled outputs -- arithmetic
    intensity collapses and the array cannot be fed.

    Without this term the model was structurally incapable of reporting a
    bandwidth-bound GEMM, which is a class of failure it must be able to report.
    """
    array = cfg.macs_per_cycle(prec) * cfg.gemm_util(prec, effective_dim)
    bpm = (cfg.gemm_operand_bytes_per_mac(prec) if bytes_per_mac is None
           else bytes_per_mac)
    if bpm <= 0:
        return array
    return min(array, cfg.scratchpad_bw_bpc / bpm)


def gemm_like_cycles(macs: float, cfg: ArgmindCfg, prec: str,
                     effective_dim: float,
                     bytes_per_mac: float | None = None) -> float:
    """Cycles for `macs` MACs, penalised for small-matrix tile underuse.

    A tile of dimension T is only fully used when the operand dimensions are
    multiples of T. `effective_dim` is a representative matrix dimension; the
    utilisation term captures quantisation waste plus fill/drain. See
    ArgmindCfg.gemm_util for why the penalty is per-PRECISION: fp64 runs on the
    FMA cluster, not on the array.
    """
    return macs / gemm_macs_per_cycle_bounded(cfg, prec, effective_dim,
                                              bytes_per_mac)


def representative_dim(b: Benchmark) -> float:
    """Crude inner-dimension proxy used for the GEMM utilisation penalty."""
    if b.gemm_dim_override is not None:
        return b.gemm_dim_override
    if b.kind == "ml":
        return 512.0            # large tensors -> tile stays busy
    return max(4.0, b.macs_per_iter ** (1.0 / 3.0) * 1.6)


def split_gemv_cycles(b: Benchmark, cfg: ArgmindCfg, macs: float) -> float:
    """GEMV cycles for `macs`, split across the two operand precisions.

    `solve_prec_frac` of the work runs at `b.prec` (the numerically sensitive
    cached triangular solve); the rest runs at `b.operand_prec`. Returns the
    single-precision answer unchanged when `operand_prec` is None, so this cannot
    move a pre-existing row.
    """
    if b.operand_prec is None:
        return macs / cfg.gemv_macs_per_cycle(b.prec, b.gemv_row_len)
    hi = macs * b.solve_prec_frac
    lo = macs - hi
    return (hi / cfg.gemv_macs_per_cycle(b.prec, b.gemv_row_len)
            + lo / cfg.gemv_macs_per_cycle(b.operand_prec, b.gemv_row_len))


def descriptor_cost(b: Benchmark, cfg: ArgmindCfg, per_call: bool = False) -> dict:
    """Sequencer cost of one iteration (or of the once-per-call work).

    REPLACES the flat `0.5 * scalar_ops_per_iter` term. Three separate physical
    effects, none of which the old fudge could express:

      issue_cycles   n_descriptors * seq_setup_cycles / seq_issue_width.
                     The sequencer's own throughput.
      dep_drain      one datapath drain per dependency-chain boundary. Descriptor
                     i+1 of a Cholesky cannot START until i's results are written
                     back. This is the term the flat constant missed entirely and
                     it is the one that actually costs HPIPM time.
      serial_scalar  the loop-carried scalar chain (convergence test, step size).

    OVERLAP, stated as a condition instead of a fudge factor. With a descriptor
    queue of depth >=1 the sequencer runs AHEAD: it computes descriptor i+1 while
    the datapath executes i, so setup and execute overlap and the pair costs
    max(execute, issue), not their sum. This is legitimate here for a specific
    reason -- the CM's allocation and schedule are STATICALLY CODE-GENERATED, so
    every descriptor's addresses, dims and strides are known before the loop
    runs. Nothing in the descriptor stream is data-dependent except the trip
    count, and continuing the loop is the correct speculation on every iteration
    but the last. If a descriptor's PARAMETERS were data-dependent (pivoting, an
    active-set index, alpha scaling the next operand) the overlap would not hold;
    that work is charged to `serial_scalar_per_iter` instead.

    Set cfg.seq_queue_depth = 0 to get the non-decoupled bound (issue serialised
    with execute), which is what a naive CSR-write-and-poll sequencer gives. The
    model must be able to produce that answer.

    LOCAL issue binding: a burst of short descriptors longer than the queue
    cannot borrow slack from a long neighbour, so a chain group is checked
    against its own issue cost when its descriptor count exceeds the queue depth.
    """
    ops = b.kernels_call if per_call else b.kernels
    dim = representative_dim(b)
    n_desc = n_dep = 0.0
    exec_cycles = 0.0
    group_cycles = 0.0
    for op in ops:
        d, dep = op_descriptors(op, cfg, b.prec)
        c = op_cycles(op, cfg, b.prec, dim, b.gemv_row_len)
        n_desc += d
        n_dep += dep
        exec_cycles += c
        # Drain between dependent descriptors: the chain's own serialisation.
        g = c + dep * cfg.drain_cycles
        issue_g = d * cfg.seq_setup_cycles / cfg.seq_issue_width
        if dep > 0 and d > cfg.seq_queue_depth:
            g = max(g, issue_g)
        group_cycles += g
    issue_cycles = n_desc * cfg.seq_setup_cycles / cfg.seq_issue_width
    serial = 0.0 if per_call else b.serial_scalar_per_iter
    if cfg.seq_queue_depth >= 1:
        total = max(group_cycles, issue_cycles) + serial
    else:
        total = group_cycles + issue_cycles + serial
    return {
        "n_desc": n_desc,
        "n_dep": n_dep,
        "exec_cycles": exec_cycles,
        "dep_drain_cycles": group_cycles - exec_cycles,
        "issue_cycles": issue_cycles,
        "serial_cycles": serial,
        "total_cycles": total,
        # The design-rule quantity: datapath cycles commanded per descriptor.
        "cycles_per_desc": exec_cycles / n_desc if n_desc else float("inf"),
        "issue_bound": issue_cycles > group_cycles,
        # Everything above and beyond the raw datapath cycles.
        "seq_overhead_cycles": total - exec_cycles,
    }


def assert_kernels_consistent(b: Benchmark, tol: float = 1e-6) -> None:
    """The kernel list must account for the declared operation counts exactly.

    Descriptor counts derived from a decomposition that does not add up to the
    benchmark's MAC counts is the error that double-counted HPIPM GEMM 5.7x.
    """
    if not b.kernels:
        return
    for label, got, want in (
        ("gemm", sum(o.total_macs for o in b.kernels if o.kind == "gemm"),
         b.gemm_macs_per_iter),
        ("gemv", sum(o.total_macs for o in b.kernels if o.kind == "gemv"),
         b.gemv_macs_per_iter),
        ("vec", sum(o.total_vec for o in b.kernels), b.vector_ops_per_iter),
        ("trans", sum(o.total_trans for o in b.kernels),
         b.transcendental_per_iter),
        ("gemm_call", sum(o.total_macs for o in b.kernels_call
                          if o.kind == "gemm"), b.gemm_macs_per_call),
        ("gemv_call", sum(o.total_macs for o in b.kernels_call
                          if o.kind == "gemv"), b.gemv_macs_per_call),
    ):
        if abs(got - want) > tol * max(1.0, abs(want)):
            raise AssertionError(
                f"{b.name}: kernel decomposition {label} sums to {got:.6g} but "
                f"the benchmark declares {want:.6g}")


def model_argmind(b: Benchmark, cfg: ArgmindCfg) -> dict:
    ns = cfg.cycle_ns
    dim = representative_dim(b)

    # Matrix-vector work: bandwidth-bound, indifferent to tile size. Uses
    # DELIVERED bandwidth (short-row derated), not peak.
    gemv_cycles = split_gemv_cycles(b, cfg, b.gemv_macs_per_iter)
    # Matrix-matrix work: tile-array-bound, penalised for small dimensions --
    # and, since 2026-07-30, also bounded by operand fetch when reuse depth is
    # set by a problem dimension rather than by the tile.
    gemm_cycles = gemm_like_cycles(b.gemm_macs_per_iter, cfg, b.prec, dim,
                                   bytes_per_mac=b.gemm_bytes_per_mac)
    vec_cycles = b.vector_ops_per_iter / cfg.vector_lanes
    trans_cycles = b.transcendental_per_iter * 4.0 / cfg.vector_lanes
    scal_cycles = b.scalar_ops_per_iter / cfg.seq_ipc

    datapath_iter = gemv_cycles + gemm_cycles + vec_cycles + trans_cycles
    once_datapath = (gemm_like_cycles(b.gemm_macs_per_call, cfg, b.prec, dim,
                                      bytes_per_mac=b.gemm_bytes_per_mac)
                     + split_gemv_cycles(b, cfg, b.gemv_macs_per_call))
    if cfg.use_descriptor_seq and b.kernels:
        # Descriptor-derived sequencer cost. `seq_overhead_cycles` is everything
        # the datapath cycles do not already contain: dependency drains, issue
        # cycles when the sequencer is the bottleneck, and the loop-carried
        # scalar chain. Adding the OVERHEAD (rather than substituting the
        # descriptor total) keeps this term additive on top of the existing
        # gemv/gemm/vec decomposition, so no other model output shifts for a
        # reason unrelated to the sequencer.
        d_iter = descriptor_cost(b, cfg)
        d_call = descriptor_cost(b, cfg, per_call=True)
        per_iter_cycles = datapath_iter + d_iter["seq_overhead_cycles"]
        once_cycles = once_datapath + d_call["seq_overhead_cycles"]
        seq_cycles_iter = d_iter["seq_overhead_cycles"]
    else:
        # LEGACY: flat scalar count, fixed 50% compute overlap. Retained so v0
        # numbers stay reproducible and so the delta is auditable.
        per_iter_cycles = datapath_iter + 0.5 * scal_cycles
        once_cycles = once_datapath
        seq_cycles_iter = 0.5 * scal_cycles
    compute_us = (per_iter_cycles * b.iters + once_cycles) * ns / 1000.0

    # ---- data movement --------------------------------------------------
    bytes_moved = b.delta_bytes
    if cfg.transfers_persistent_every_call():
        bytes_moved += b.persistent_bytes
    else:
        # Residency: persistent data stays in scratchpad. If it does not fit,
        # the excess must stream every call (LLM weights, large factorisations).
        cap = cfg.scratchpad_kb * 1024
        overflow = max(0.0, b.persistent_bytes - cap)
        bytes_moved += overflow

    dma_us = bytes_moved / (cfg.dma_bw_gbs * 1e9) * 1e6
    # Double-buffered: DMA overlaps compute except for the unavoidable head.
    transfer_us = (dma_us if cfg.transfers_persistent_every_call()
                   else max(0.0, dma_us - compute_us * 0.8))

    latency_us = compute_us + transfer_us + cfg.call_overhead_us

    # ---- energy ----------------------------------------------------------
    # Total GEMV MACs, split by operand precision. Narrow operands cut BOTH the
    # multiplier energy and the scratchpad traffic, and both must move together
    # or the model would report a bandwidth win with no energy win.
    gemv_all = b.gemv_macs_per_iter * b.iters + b.gemv_macs_per_call
    gemm_all = b.gemm_macs_per_iter * b.iters + b.gemm_macs_per_call
    op_p = b.operand_prec or b.prec
    gemv_hi = gemv_all * (b.solve_prec_frac if b.operand_prec else 0.0)
    gemv_lo = gemv_all - gemv_hi
    e_mac = (gemv_hi * E_MAC_PJ[b.prec] + gemv_lo * E_MAC_PJ[op_p]
             + gemm_all * E_MAC_PJ[b.prec])
    # Scratchpad traffic: GEMV touches a fresh element per MAC (no reuse);
    # GEMM amortises operand fetch over the tile dimension (or over an explicit
    # per-benchmark reuse depth when the tile is not what limits it).
    #
    # SEPARATE CORRECTION, 2026-07-30, flagged rather than bundled: the old
    # expression omitted `gemv_macs_per_call` from the scratchpad byte count
    # while `e_mac` (via total_macs) charged it and the latency model charged it.
    # Once-per-call GEMV traffic is real traffic. Including it raises reported
    # solver energy a few percent -- i.e. it moves AGAINST our own ratio, which
    # is why it is being fixed now rather than deferred.
    gemm_bpm = (cfg.gemm_operand_bytes_per_mac(b.prec)
                if b.gemm_bytes_per_mac is None else b.gemm_bytes_per_mac)
    sram_bytes = (gemv_hi * BYTES_PER_ELEM[b.prec]
                  + gemv_lo * BYTES_PER_ELEM[op_p]
                  + gemm_all * gemm_bpm)
    e_sram_pj_b = cfg.sram_pj_per_byte()
    e_sram = sram_bytes * e_sram_pj_b
    e_dram = bytes_moved * cfg.offchip_pj_per_byte
    # V^2 on everything the CM's own supply pays for; leakage worse than V^2.
    # Both multipliers are exactly 1.0 at the nominal 800 MHz, so this term
    # cannot move a configuration that does not touch the frequency axis. DRAM /
    # off-chip energy is NOT scaled: it is on the host's supply, not ours.
    e_static = (cfg.static_mw * cfg.leak_scale) * 1e-3 * (latency_us * 1e-6) * 1e12
    energy_pj = ((e_mac + e_sram) * cfg.overhead_mult * cfg.dyn_energy_scale
                 + e_dram + e_static)

    return {
        "latency_us": latency_us,
        "compute_us": compute_us,
        "transfer_us": transfer_us,
        "overhead_us": cfg.call_overhead_us,
        "energy_uj": energy_pj / 1e6,
        "bytes_moved": bytes_moved,
        "avg_power_mw": (energy_pj / 1e12) * b.calls_per_sec * 1e3,
        "sram_pj_per_byte": e_sram_pj_b,
        "e_sram_frac": e_sram * cfg.overhead_mult / energy_pj,
        "delivered_bw_bpc": cfg.delivered_bw_bpc(b.prec, b.gemv_row_len),
        "seq_cycles_per_iter": seq_cycles_iter,
        "seq_frac_of_compute": (seq_cycles_iter * b.iters * ns / 1000.0
                                / compute_us if compute_us > 0 else 0.0),
    }


def model_cpu(b: Benchmark, cpu: CpuCfg, dram_gbs: float = 12.8) -> dict:
    """CPU baseline. `dram_gbs` is the SYSTEM's external bandwidth.

    IT MUST BE THE SAME NUMBER THE CM IS GIVEN. It was a hard-coded 12.8 GB/s,
    which was harmless while nothing swept the axis and became a
    baseline-fairness bug the moment the ML-envelope study did: at 51.2 GB/s the
    CM would have been credited with 4x the CPU's weight-streaming rate on the
    same SoC. Default 12.8 reproduces every pre-existing figure byte-identically.
    """
    peak = cpu.peak_flops_per_cycle(b.prec)
    eff = peak * cpu.small_gemm_efficiency          # ops/cycle achieved
    gemm_cycles = (b.gemm_macs_per_iter * 2.0) / eff
    # Mixed-precision operands: the bulk matrix at `operand_prec`, the sensitive
    # fraction at `prec`. On the CPU this changes only the L1 bandwidth term
    # (there is no narrower compute path), which is the honest effect.
    if b.operand_prec is None:
        gemv_cycles = b.gemv_macs_per_iter / cpu.gemv_macs_per_cycle(b.prec)
    else:
        hi = b.gemv_macs_per_iter * b.solve_prec_frac
        gemv_cycles = (hi / cpu.gemv_macs_per_cycle(b.prec)
                       + (b.gemv_macs_per_iter - hi)
                       / cpu.gemv_macs_per_cycle(b.operand_prec))
    # Elementwise/reduction work runs on the plain SIMD path, NOT the outer-
    # product unit. See CpuCfg.simd_flops_per_cycle. `*2.0` is kept because the
    # model counts vector "element-ops" in the same FLOP-pair units as MACs.
    vec_eff = cpu.simd_rate * cpu.small_gemm_efficiency
    vec_cycles = (b.vector_ops_per_iter * 2.0) / vec_eff
    trans_cycles = b.transcendental_per_iter * 12.0      # div/sqrt latency-bound
    scal_cycles = b.scalar_ops_per_iter / cpu.scalar_ipc

    cycles = ((gemm_cycles + gemv_cycles + vec_cycles + trans_cycles
               + scal_cycles) * b.iters
              + (b.gemm_macs_per_call * 2.0) / eff
              + b.gemv_macs_per_call / cpu.gemv_macs_per_cycle(b.prec))
    latency_us = cycles / cpu.freq_mhz

    # Memory energy: assume working set is cache-resident for solver problems
    # (favourable to the CPU, and true for small MPC), DRAM-streamed for ML.
    if b.kind == "ml":
        mem_bytes = b.persistent_bytes + b.delta_bytes
        e_mem = mem_bytes * E_DRAM_BYTE_PJ
        # DRAM bandwidth bound check for weight-streaming workloads.
        dram_us = mem_bytes / (dram_gbs * 1e9) * 1e6
        latency_us = max(latency_us, dram_us)
    else:
        mem_bytes = b.total_macs() * 0.5
        e_mem = mem_bytes * E_L2_BYTE_PJ

    energy_pj = cpu.active_w * (latency_us * 1e-6) * 1e12 + e_mem
    return {
        "latency_us": latency_us,
        "energy_uj": energy_pj / 1e6,
        "avg_power_mw": (energy_pj / 1e12) * b.calls_per_sec * 1e3,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

# THE RECOMMENDED PART, as re-sized 2026-07-30 (see PHASE0_FINDINGS.md §0
# resolution block). Every headline ratio must be reported at THIS configuration.
# Until now they were all reported at CFG_16_LEGACY below -- 256 MACs and 256 KB,
# a configuration we no longer recommend -- so the quoted latency and perf/W
# figures described a part that was never the product.
# CAPACITY CORRECTED 128 -> 192 KB, 2026-07-30, by descriptor v0.1 ruling R8.
# The functional simulator's MEASURED allocation plan for QUAD-MPC is 163.8 KiB
# at fp32 (138.8 KiB once A = [I;Gamma] is exploited) against the 128 KB that was
# costed. The flagship's three matrices ALONE are 125.0 KiB of a 128.0 KiB
# scratchpad, so 128 KB was arithmetically impossible, not merely tight.
# Bandwidth (128 B/cycle) and bank count (16) DO NOT MOVE: capacity was never the
# binding constraint on latency and still is not -- this is a capacity correction,
# not a rebalancing, and it does not touch the bandwidth-first finding.
# Cost: core 0.698 -> 0.786 mm2 (+12.6%), pJ/byte 0.718 -> 0.839, QUAD-MPC
# perf/W 14.13x -> 12.55x. 192 KB rather than 160 KB because 12 KiB per bank
# (1536 x 64b) has array efficiency 0.394 against 0.354 for 10 KiB, so the last
# 32 KiB costs 0.020 mm2 -- 160 KB is dominated.
CM100_SCRATCHPAD_KB = 192
CM100_SCRATCHPAD_KB_PRE_R8 = 128        # kept so the failure stays reproducible

CFG_100 = ArgmindCfg(
    name="CM-100 (12x12=144 MACs, 192 KB, 16 banks x 8 B = 128 B/cyc, "
         "8 merged FMAs = 16 fp32 lanes / 8 fp64, 800 MHz, T1)",
    tile=12, scratchpad_kb=CM100_SCRATCHPAD_KB, scratchpad_bw_bpc=128.0,
    n_banks=16, vector_lanes=16, fp64_fmas=8)

# Retained for comparison and for the tile/bandwidth sweeps. NOT the product.
# NOTE: all of these carry the SAME 8-unit fp64 cluster, because fp64 throughput
# is not a function of the array size. The tile sweep therefore shows ARM-MPC
# FLAT across tile sizes -- correct, and previously hidden by the 0.125 factor.
CFG_16 = ArgmindCfg(name="CM-16 legacy (16x16 tile, 256 KB, 64 B/cyc, T1)",
                    tile=16)
CFG_32 = ArgmindCfg(name="CM-32 (32x32 tile, 800MHz, T1)", tile=32)
CFG_64 = ArgmindCfg(name="CM-64 (64x64 tile, 800MHz, T1)", tile=64)
CFG_100_T0 = ArgmindCfg(
    name="CM-100, T0 copy-based (ablation)", tile=12,
    scratchpad_kb=CM100_SCRATCHPAD_KB,
    scratchpad_bw_bpc=128.0, n_banks=16, vector_lanes=16, fp64_fmas=8, tier="T0")

KILL_CRITERION = 5.0   # perf/W multiple vs best CPU baseline, solver workloads


def fmt(x, w=9, p=1):
    if x >= 1e5:
        return f"{x:>{w}.3g}"
    return f"{x:>{w}.{p}f}"


def main():
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    out("=" * 100)
    out("ARGMIND CM — PHASE 0 PERFORMANCE MODEL v0   (first-order estimates, +/-2x)")
    out("=" * 100)

    # ---- main table ----
    out()
    out("Configuration under test: " + CFG_100.name)
    out(f"  {CFG_100.macs} MACs, {CFG_100.scratchpad_kb} KB scratchpad, "
        f"{CFG_100.scratchpad_bw_bpc:.0f} B/cycle, {CFG_100.sram_pj_per_byte():.3f} pJ/byte, "
        f"call overhead {CFG_100.call_overhead_us} us")
    out()
    out("Comparison rule (conservative on BOTH axes):")
    out("  speedup  = vs the FASTEST CPU baseline")
    out("  perf/W   = vs the MOST ENERGY-EFFICIENT CPU baseline (usually A55-class)")
    out()
    hdr = (f"{'benchmark':<12} {'deadline':>9} {'CM lat':>9} {'fastCPU':>9} "
           f"{'speedup':>8} {'CM uJ':>9} {'bestCPU uJ':>11} "
           f"{'perf/W':>8}  verdict")
    out(hdr)
    out("-" * len(hdr))

    solver_ratios = []
    for b in BENCHMARKS:
        cm = model_argmind(b, CFG_100)
        cpus = [(c, model_cpu(b, c)) for c in ALL_CPUS]

        fastest_cfg, fastest = min(cpus, key=lambda t: t[1]["latency_us"])
        efficient_cfg, efficient = min(cpus, key=lambda t: t[1]["energy_uj"])

        speedup = fastest["latency_us"] / cm["latency_us"]
        # perf/W = (work/time)/power -> ratio reduces to energy ratio per call
        perf_w = efficient["energy_uj"] / cm["energy_uj"]

        meets = cm["latency_us"] <= b.deadline_us
        verdict = "MEETS deadline" if meets else "MISSES deadline"
        if b.kind == "solver":
            solver_ratios.append((b.name, speedup, perf_w, efficient_cfg.name))
            if perf_w < KILL_CRITERION:
                verdict += " / BELOW 5x perf-W"
        # Does the cheap CPU already hit the deadline unaided?
        if fastest["latency_us"] <= b.deadline_us:
            verdict += " (CPU also meets it)"

        out(f"{b.name:<12} {fmt(b.deadline_us)} {fmt(cm['latency_us'])} "
            f"{fmt(fastest['latency_us'])} "
            f"{fmt(speedup,8)} {fmt(cm['energy_uj'])} {fmt(efficient['energy_uj'],11)} "
            f"{fmt(perf_w,8)}  {verdict}")

    out()
    out("Notes on the table:")
    out("  * 'CPU lat' assumes the solver working set is CACHE-RESIDENT — deliberately")
    out("    favourable to the CPU. ML rows are DRAM-bandwidth-bound for the CPU.")
    out("  * perf/W ratio == energy-per-call ratio (same work, same rate).")
    out("  * '(CPU also meets it)' is a WARNING, not a win: if a CPU already makes the")
    out("    deadline, we must sell on energy/consolidation, not on feasibility.")
    out()
    out("CPU baselines modelled: " + "; ".join(c.name for c in ALL_CPUS))

    # ---- bandwidth requirement check ----
    out()
    out("BANDWIDTH CHECK — required sustained DMA bandwidth to hit each deadline")
    out(f"{'benchmark':<12} {'bytes/call':>12} {'deadline us':>12} "
        f"{'req GB/s':>9} {'have GB/s':>10}  status")
    for b in BENCHMARKS:
        cm = model_argmind(b, CFG_100)
        req = cm["bytes_moved"] / (b.deadline_us * 1e-6) / 1e9
        ok = "OK" if req <= CFG_16.dma_bw_gbs else "BANDWIDTH-BOUND"
        out(f"{b.name:<12} {fmt(cm['bytes_moved'],12,0)} {fmt(b.deadline_us,12)} "
            f"{fmt(req,9,2)} {fmt(CFG_16.dma_bw_gbs,10,1)}  {ok}")

    # ---- kill criterion ----
    out()
    out("=" * 100)
    out(f"KILL CRITERION: solver workloads must show >= {KILL_CRITERION}x perf/W "
        f"vs best CPU baseline")
    out("=" * 100)
    passed = [n for n, _, p, _ in solver_ratios if p >= KILL_CRITERION]
    failed = [n for n, _, p, _ in solver_ratios if p < KILL_CRITERION]
    for n, s, p, ref in solver_ratios:
        flag = "PASS" if p >= KILL_CRITERION else "FAIL"
        out(f"  {n:<12} speedup {s:>7.1f}x   perf/W {p:>7.1f}x  vs {ref:<38} {flag}")
    out()
    out(f"  -> {len(passed)}/{len(solver_ratios)} solver benchmarks pass "
        f"({', '.join(passed) if passed else 'none'})")
    if failed:
        out(f"  -> below criterion: {', '.join(failed)}  "
            f"(these define the lower bound of the addressable socket)")

    # ---- tile-size sensitivity ----
    out()
    out("=" * 100)
    out("TILE SIZE SENSITIVITY — does a bigger array help solver work?")
    out("=" * 100)
    out(f"{'benchmark':<12} " + " ".join(f"{c.name.split()[0]:>10}" for c in
                                         (CFG_16, CFG_32, CFG_64)) + "   (latency us)")
    for b in BENCHMARKS:
        row = [f"{model_argmind(b, c)['latency_us']:>10.1f}"
               for c in (CFG_16, CFG_32, CFG_64)]
        out(f"{b.name:<12} " + " ".join(row))
    out()
    out("  Read: if a solver row barely improves (or worsens) from CM-16 -> CM-64,")
    out("  the array is already underutilised — buying MACs does not buy solver speed.")
    out(f"  (CM-16={CFG_16.macs} MACs, CM-32={CFG_32.macs}, CM-64={CFG_64.macs} — "
        f"a 16x MAC increase.)")

    # ---- scratchpad bandwidth sensitivity ----
    out()
    out("=" * 100)
    out("SCRATCHPAD BANDWIDTH SENSITIVITY — the knob that actually moves solver work")
    out("=" * 100)
    bw_cfgs = [ArgmindCfg(name=f"bw{int(bw)}", tile=16, scratchpad_bw_bpc=bw)
               for bw in (16, 32, 64, 128, 256)]
    out(f"{'benchmark':<12} " + " ".join(f"{'B/cyc=' + str(int(c.scratchpad_bw_bpc)):>11}"
                                         for c in bw_cfgs) + "   (latency us)")
    for b in BENCHMARKS:
        row = [f"{model_argmind(b, c)['latency_us']:>11.1f}" for c in bw_cfgs]
        out(f"{b.name:<12} " + " ".join(row))
    out()
    out("  Compare against the tile-size table above. If bandwidth moves the solver")
    out("  rows more than MAC count does, then the correct architecture is a WIDE,")
    out("  heavily-banked scratchpad with a MODEST array — not a big systolic tile.")
    out("  That is a Phase-1 microarchitecture decision worth getting right.")
    out()
    out("  CAUTION — the table above is the OPTIMISTIC reading. It assumes (a) the")
    out("  full fetch width is usable on every GEMV row, which needs SEGMENTED")
    out("  REDUCTION in the RTL, and (b) the vector unit is not the bottleneck.")
    out("  Same sweep with segmented reduction ABSENT (tail beat of each matrix")
    out("  row wasted) — this is what you get if that RTL feature is skipped:")
    bw_ns = [ArgmindCfg(name=f"bw{int(bw)}", tile=16, scratchpad_bw_bpc=bw,
                        segmented_reduction=False) for bw in (16, 32, 64, 128, 256)]
    out(f"{'benchmark':<12} " + " ".join(
        f"{'B/cyc=' + str(int(c.scratchpad_bw_bpc)):>11}" for c in bw_ns))
    for b in BENCHMARKS:
        row = [f"{model_argmind(b, c)['latency_us']:>11.1f}" for c in bw_ns]
        out(f"{b.name:<12} " + " ".join(row))
    out()
    out("  And the vector unit becomes the co-bottleneck past ~128 B/cycle: on")
    out("  QUAD-MPC with 8 fp32 lanes, vector+transcendental work is a fixed 720")
    out("  cycles/iteration, so it caps the achievable speedup regardless of")
    out("  scratchpad width. See perf_model/cm100_sizing.py for the full costing")
    out("  (area, energy, bank conflicts) behind the bandwidth-first decision.")

    # ---- zero-copy / residency ablation ----
    out()
    out("=" * 100)
    out("ABLATION — value of zero-copy + residency (§3.2.1/3.2.2)")
    out("=" * 100)
    hdr2 = (f"{'benchmark':<12} {'T1 lat':>9} {'T0 lat':>9} {'T1 adv':>8} "
            f"{'T1 bytes':>11} {'T0 bytes':>11}  deadline w/ T0")
    out(hdr2)
    out("-" * len(hdr2))
    for b in BENCHMARKS:
        t1 = model_argmind(b, CFG_100)
        t0 = model_argmind(b, CFG_100_T0)
        adv = t0["latency_us"] / t1["latency_us"]
        ok = "MEETS" if t0["latency_us"] <= b.deadline_us else "MISSES"
        out(f"{b.name:<12} {fmt(t1['latency_us'])} {fmt(t0['latency_us'])} "
            f"{fmt(adv,8)} {fmt(t1['bytes_moved'],11,0)} "
            f"{fmt(t0['bytes_moved'],11,0)}  {ok}")
    out()
    out("  This quantifies the architectural bet: if T1-vs-T0 advantage is large on")
    out("  the high-rate rows, the zero-copy/residency work IS the product.")

    # ---- crossover sweep ----
    out()
    out("=" * 100)
    out("CROSSOVER FRONTIER — smallest problem worth offloading")
    out("=" * 100)
    out("Condensed MPC, N=20, 40 ADMM iters, 1 kHz; sweeping control dimension n_u")
    out(f"{'n_u':>5} {'n_var':>6} {'CM lat us':>10} {'CPU lat us':>11} "
        f"{'speedup':>8} {'perf/W':>8}  winner")
    for n_u in (1, 2, 3, 4, 6, 8, 12, 16, 24):
        b = condensed_mpc(f"sweep{n_u}", n_x=12, n_u=n_u, N=20, iters=40,
                          rate=1000, note="")
        cm = model_argmind(b, CFG_100)
        cp = model_cpu(b, CPU_A78)
        sp = cp["latency_us"] / cm["latency_us"]
        pw = cp["energy_uj"] / cm["energy_uj"]
        win = "Argmind" if pw >= KILL_CRITERION else ("marginal" if pw >= 1.0
                                                      else "CPU")
        out(f"{n_u:>5} {n_u*20:>6} {fmt(cm['latency_us'],10)} "
            f"{fmt(cp['latency_us'],11)} {fmt(sp,8)} {fmt(pw,8)}  {win}")
    out()
    out("  The first row where 'winner' becomes Argmind is the minimum viable socket.")
    out("  Below it, tell the customer to use the CPU — credibility is worth more")
    out("  than a bad design win.")

    out()
    out("=" * 100)
    out("CAVEATS (do not quote these numbers externally)")
    out("=" * 100)
    out("  1. Energy constants are scaled estimates, not PDK numbers (+/-2x).")
    out("  2. CPU small-GEMM efficiency (30%) is a literature-informed guess; measure")
    out("     it on real hardware with OSQP/acados before trusting any ratio.")
    out("  3. Sequencer cost is now DESCRIPTOR-DERIVED (issue rate + dependency")
    out("     drain + loop-carried scalar chain), not the old flat 50%-overlap")
    out("     fudge -- see perf_model/sequencer.py. The load-bearing [EST] is")
    out(f"     seq_setup_cycles = {CFG_100.seq_setup_cycles:.0f} cycles/descriptor; "
        f"count the instructions in a")
    out("     real RV32 descriptor-emit sequence to firm it up.")
    out("  4. No thermal, DVFS, or contention modelling.")
    out("  5. Next fidelity step: measure the CPU baselines for real (Phase 0), then")
    out("     replace the CM side with the cycle-approximate model (Phase 1).")

    with open("perf_model/results_v0.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
