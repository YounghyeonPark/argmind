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
E_MAC_PJ = {"int8": 0.07, "fp16": 0.20, "bf16": 0.20, "fp32": 0.60, "fp64": 2.20}
BYTES_PER_ELEM = {"int8": 1, "fp16": 2, "bf16": 2, "fp32": 4, "fp64": 8}

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
        return sram_cost.SpadCfg(self.scratchpad_kb, nb, wb)

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
        """
        if prec == "fp64":
            return float(self.fp64_fmas)
        scale = {"int8": 2.0, "fp16": 1.0, "bf16": 1.0, "fp32": 0.5}[prec]
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
    @property
    def call_overhead_us(self) -> float:
        """Fixed cost to launch one CM call from the CPU."""
        if self.tier == "T1":
            # Descriptor write + doorbell store + completion poll, all in
            # coherent shared memory. No syscall, no cache maintenance.
            return 0.8
        # T0: driver call, buffer map, cache clean/invalidate, IRQ wakeup.
        return 25.0

    @property
    def dma_bw_gbs(self) -> float:
        return 6.4   # one LPDDR4x-2400 x32 channel share

    def transfers_persistent_every_call(self) -> bool:
        """Without residency, problem data must be re-uploaded every call."""
        return self.tier == "T0"


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

    def peak_flops_per_cycle(self, prec: str) -> float:
        if prec == "int8":
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
                  check_term=25, iters_worst=0):
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
    # P symmetric (n^2/2), L triangular (n^2/2), A's identity block stores nothing.
    persistent = 4 * (n_var * n_var / 2 + n_var * n_var / 2 + Annz)
    delta = 4 * (n_x + 3 * n_var)                          # state + reference
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
    persistent = e * (N * nb * nb + N * n_x * nb)
    delta = e * (n_x + N * n_x)
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
    )


def slm_decode(name, params_b, hidden, layers, seq, rate, note, prec="int8"):
    """Autoregressive decode, batch 1: weight-bound GEMV + attention over KV cache."""
    # Batch-1 decode is pure GEMV: each weight is used exactly once per token.
    gemv = params_b * 1e9 + 2 * layers * seq * hidden
    persistent = params_b * 1e9 * (1 if prec == "int8" else 2)   # weights + KV
    delta = 4 * hidden
    return Benchmark(
        name=name, kind="ml", note=note,
        gemv_macs_per_iter=gemv, gemm_macs_per_iter=0.0, gemm_macs_per_call=0.0,
        vector_ops_per_iter=layers * hidden * 12,
        transcendental_per_iter=layers * (hidden + seq), scalar_ops_per_iter=200,
        algo="nn",
        iters=1, calls_per_sec=rate, deadline_us=1e6 / rate,
        persistent_bytes=persistent, delta_bytes=delta, prec=prec,
        gemv_row_len=hidden,   # weight-matrix rows are long: fetch width is fine
    )


def cnn_infer(name, gmacs, weights_mb, rate, note, prec="int8"):
    # Conv/attention as GEMM with real operand reuse -- the tile array's home turf.
    return Benchmark(
        name=name, kind="ml", note=note,
        gemv_macs_per_iter=0.0, gemm_macs_per_iter=gmacs * 1e9,
        gemm_macs_per_call=0.0,
        vector_ops_per_iter=gmacs * 1e9 * 0.02,
        transcendental_per_iter=1e5, scalar_ops_per_iter=500, algo="nn",
        iters=1, calls_per_sec=rate, deadline_us=1e6 / rate,
        persistent_bytes=weights_mb * 1e6, delta_bytes=224 * 224 * 3, prec=prec,
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
    condensed_mpc("QUAD-MPC",  n_x=12, n_u=4, N=20, iters=15, rate=1000,
                  iters_worst=74,
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
    # Iteration count scaled from the QUAD-MPC measurement (not independently measured).
    # iters_worst scaled by the same 74/15 measured cold:p95 ratio -> 48. [EST]
    condensed_mpc("PWR-MPC",   n_x=4,  n_u=3, N=10, iters=12, rate=20000,
                  iters_worst=48,
                  note="Motor-drive MPC, tiny problem, 20 kHz -- latency extreme"),
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
# Argmind model
# ---------------------------------------------------------------------------

def gemm_like_cycles(macs: float, cfg: ArgmindCfg, prec: str,
                     effective_dim: float) -> float:
    """Cycles for `macs` MACs, penalised for small-matrix tile underuse.

    A tile of dimension T is only fully used when the operand dimensions are
    multiples of T. `effective_dim` is a representative matrix dimension; the
    utilisation term captures quantisation waste plus fill/drain. See
    ArgmindCfg.gemm_util for why the penalty is per-PRECISION: fp64 runs on the
    FMA cluster, not on the array.
    """
    return macs / (cfg.macs_per_cycle(prec) * cfg.gemm_util(prec, effective_dim))


def representative_dim(b: Benchmark) -> float:
    """Crude inner-dimension proxy used for the GEMM utilisation penalty."""
    if b.kind == "ml":
        return 512.0            # large tensors -> tile stays busy
    return max(4.0, b.macs_per_iter ** (1.0 / 3.0) * 1.6)


def model_argmind(b: Benchmark, cfg: ArgmindCfg) -> dict:
    ns = cfg.cycle_ns
    dim = representative_dim(b)

    # Matrix-vector work: bandwidth-bound, indifferent to tile size. Uses
    # DELIVERED bandwidth (short-row derated), not peak.
    gemv_cycles = b.gemv_macs_per_iter / cfg.gemv_macs_per_cycle(
        b.prec, b.gemv_row_len)
    # Matrix-matrix work: tile-array-bound, penalised for small dimensions.
    gemm_cycles = gemm_like_cycles(b.gemm_macs_per_iter, cfg, b.prec, dim)
    vec_cycles = b.vector_ops_per_iter / cfg.vector_lanes
    trans_cycles = b.transcendental_per_iter * 4.0 / cfg.vector_lanes
    scal_cycles = b.scalar_ops_per_iter / cfg.seq_ipc

    # Sequencer control overlaps compute only partially; assume 50% overlap.
    per_iter_cycles = (gemv_cycles + gemm_cycles + vec_cycles + trans_cycles
                       + 0.5 * scal_cycles)
    once_cycles = (gemm_like_cycles(b.gemm_macs_per_call, cfg, b.prec, dim)
                   + b.gemv_macs_per_call / cfg.gemv_macs_per_cycle(
                       b.prec, b.gemv_row_len))
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
    transfer_us = dma_us if cfg.tier == "T0" else max(0.0, dma_us - compute_us * 0.8)

    latency_us = compute_us + transfer_us + cfg.call_overhead_us

    # ---- energy ----------------------------------------------------------
    e_mac = b.total_macs() * E_MAC_PJ[b.prec]
    # Scratchpad traffic: GEMV touches a fresh element per MAC (no reuse);
    # GEMM amortises operand fetch over the tile dimension.
    sram_bytes = (b.gemv_macs_per_iter * b.iters * BYTES_PER_ELEM[b.prec]
                  + (b.gemm_macs_per_iter * b.iters + b.gemm_macs_per_call)
                  * cfg.gemm_operand_bytes_per_mac(b.prec))
    e_sram_pj_b = cfg.sram_pj_per_byte()
    e_sram = sram_bytes * e_sram_pj_b
    e_dram = bytes_moved * E_DRAM_BYTE_PJ
    e_static = cfg.static_mw * 1e-3 * (latency_us * 1e-6) * 1e12
    energy_pj = (e_mac + e_sram) * cfg.overhead_mult + e_dram + e_static

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
    }


def model_cpu(b: Benchmark, cpu: CpuCfg) -> dict:
    peak = cpu.peak_flops_per_cycle(b.prec)
    eff = peak * cpu.small_gemm_efficiency          # ops/cycle achieved
    gemm_cycles = (b.gemm_macs_per_iter * 2.0) / eff
    gemv_cycles = b.gemv_macs_per_iter / cpu.gemv_macs_per_cycle(b.prec)
    vec_cycles = (b.vector_ops_per_iter * 2.0) / eff
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
        # DRAM bandwidth bound check for weight-streaming workloads
        dram_us = mem_bytes / (12.8e9) * 1e6
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
CFG_100 = ArgmindCfg(
    name="CM-100 (12x12=144 MACs, 128 KB, 16 banks x 8 B = 128 B/cyc, "
         "8 merged FMAs = 16 fp32 lanes / 8 fp64, 800 MHz, T1)",
    tile=12, scratchpad_kb=128, scratchpad_bw_bpc=128.0, n_banks=16,
    vector_lanes=16, fp64_fmas=8)

# Retained for comparison and for the tile/bandwidth sweeps. NOT the product.
# NOTE: all of these carry the SAME 8-unit fp64 cluster, because fp64 throughput
# is not a function of the array size. The tile sweep therefore shows ARM-MPC
# FLAT across tile sizes -- correct, and previously hidden by the 0.125 factor.
CFG_16 = ArgmindCfg(name="CM-16 legacy (16x16 tile, 256 KB, 64 B/cyc, T1)",
                    tile=16)
CFG_32 = ArgmindCfg(name="CM-32 (32x32 tile, 800MHz, T1)", tile=32)
CFG_64 = ArgmindCfg(name="CM-64 (64x64 tile, 800MHz, T1)", tile=64)
CFG_100_T0 = ArgmindCfg(
    name="CM-100, T0 copy-based (ablation)", tile=12, scratchpad_kb=128,
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
    out("  3. Sequencer/compute overlap fixed at 50% — a modelling assumption.")
    out("  4. No thermal, DVFS, or contention modelling.")
    out("  5. Next fidelity step: measure the CPU baselines for real (Phase 0), then")
    out("     replace the CM side with the cycle-approximate model (Phase 1).")

    with open("perf_model/results_v0.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
