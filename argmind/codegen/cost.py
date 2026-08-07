# SPDX-License-Identifier: Apache-2.0
"""A per-descriptor datapath cost model, and the bank arithmetic behind rule C2.

The sequencer contract is stated in **datapath cycles per descriptor**, so a code
generator that cannot count them cannot check its own output. This module counts
them from the emitted descriptors, using the same rate table as
``perf_model/argmind_model.py`` -- a test asserts the two agree, because a codegen
compliance report computed against different rates than the performance model
would be worse than no report.

MODELLED, +/-2x, NOT QUOTABLE EXTERNALLY. What is *measured* here is the
descriptor stream: opcodes, dimensions, formats and their order. The cycles are a
model applied to that measurement.

The rate table, and why each entry is what it is
------------------------------------------------
=============== ======================================================
resource        rate
=============== ======================================================
array, fp32     ``macs * 0.5`` -- fp32 is a two-pass operand split
                through a bf16 array, products exact in the accumulator
array, bf16/fp16 ``macs``
array, int8     ``macs * 2`` -- double-pumped
fp64            ``fp64_fmas`` -- fp64 does **not** come from the array;
                it is the size of the merged FMA cluster
GEMV            ``min(array, bytes_per_cycle / itemsize)`` -- one matrix
                element per MAC, no reuse, so delivered bandwidth is
                the ceiling
vector lanes    ``vector_lanes`` at fp32 and narrower, ``fp64_fmas`` at
                fp64 (the cluster is one fp64 or two fp32 per unit)
=============== ======================================================

Every kernel is charged the **max of its unit rate and its bandwidth**, and both
are reported. The bandwidth term matters and is not in the aggregate performance
model: a 16-lane fp32 AXPY wants two read streams and one write stream, i.e.
192 B/cycle against a 128 B/cycle scratchpad, so elementwise work at fp32 is
bandwidth-bound before it is lane-bound. Leaving it out understates elementwise
cycles, which for the >=8-cycles-per-descriptor contract is the *flattering*
direction -- it makes short descriptors look shorter than they are.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..descriptor import Descriptor, Opcode
from ..kernels import accesses
from ..precision import Format

#: fp32-relative arithmetic rate of the MAC array, by format. fp64 is absent on
#: purpose: it does not come from the array at all.
ARRAY_RATE_SCALE: dict[Format, float] = {
    Format.INT8: 2.0,
    Format.FP16: 1.0,
    Format.BF16: 1.0,
    Format.FP32: 0.5,
}


@dataclass(frozen=True)
class DatapathModel:
    """CM-100 provisioning, as the sequencer contract is stated against.

    Defaults are the CM-100 baseline: 192 KiB in 16 banks x 12 KiB giving
    128 B/cycle, 144 MACs (12x12), 16 fp32 vector lanes on 8 merged FMA units
    (1 x fp64 or 2 x fp32 each), 800 MHz.
    """

    name: str = "CM-100"
    bytes_per_cycle: float = 128.0
    macs: int = 144
    tile: int = 12
    vector_lanes: int = 16
    fp64_fmas: int = 8
    banks: int = 16
    freq_mhz: float = 800.0

    @property
    def bank_word_bytes(self) -> int:
        return int(self.bytes_per_cycle / self.banks)

    @property
    def cycle_ns(self) -> float:
        return 1000.0 / self.freq_mhz

    # -- rates ------------------------------------------------------------

    def array_macs_per_cycle(self, fmt: Format) -> float:
        if fmt is Format.FP64:
            return float(self.fp64_fmas)
        return self.macs * ARRAY_RATE_SCALE[fmt]

    def gemv_macs_per_cycle(self, fmt: Format) -> float:
        """Matrix-vector: one matrix element per MAC, so fetch width is the ceiling.

        Note what this does at fp64 on the CM-100: the bandwidth limit is
        128/8 = 16 MAC/cycle but the cluster is 8, so **fp64 GEMV is array-bound,
        not bandwidth-bound**. That inversion is the one documented exception to
        the bandwidth-first finding and it is why an fp64 cycle count cannot be
        halved to get an fp32 one.
        """
        bw = self.bytes_per_cycle / fmt.itemsize
        return min(bw, self.array_macs_per_cycle(fmt))

    def lanes(self, fmt: Format) -> float:
        return float(self.fp64_fmas if fmt is Format.FP64 else self.vector_lanes)

    def gemm_bytes_per_mac(self, fmt: Format) -> float:
        """Operand bytes per GEMM MAC for a square output-stationary tile."""
        return fmt.itemsize * (2.0 / self.tile)

    def gemm_macs_per_cycle(self, fmt: Format) -> float:
        bw = self.bytes_per_cycle / self.gemm_bytes_per_mac(fmt)
        return min(bw, self.array_macs_per_cycle(fmt))

    # -- per-descriptor cost ----------------------------------------------

    def bytes_touched(self, d: Descriptor) -> int:
        """Scratchpad bytes a descriptor's operands span.

        Taken from :func:`argmind.kernels.accesses`, so it is derived from the
        same access model the hazard checker and the residency planner use. A
        zero-stride broadcast read spans one element, which is correct.
        """
        reads, writes = accesses(d)
        return sum(max(a.stop - a.start, 0) for a in list(reads) + list(writes))

    def mac_count(self, d: Descriptor) -> float:
        op = d.opcode
        if op is Opcode.GEMM:
            return float(d.m) * float(d.k) * float(max(d.n, 1))
        if op is Opcode.GEMV:
            return float(d.m) * float(d.k)
        if op is Opcode.TRSV:
            return float(d.m) * (float(d.m) + 1.0) / 2.0
        if op is Opcode.CHOL:
            return float(d.m) ** 3 / 6.0
        if op is Opcode.DOT:
            return float(d.m)
        return 0.0

    def unit_cycles(self, d: Descriptor) -> float:
        """Cycles on the unit that does the work, ignoring operand fetch."""
        op, fm, fv = d.opcode, d.prec_m, d.prec_v
        if op is Opcode.GEMM:
            return self.mac_count(d) / self.gemm_macs_per_cycle(fm)
        if op in (Opcode.GEMV, Opcode.TRSV):
            return self.mac_count(d) / self.gemv_macs_per_cycle(fm)
        if op is Opcode.CHOL:
            # Blocked Cholesky is dominated by its trailing GEMM updates.
            return self.mac_count(d) / self.gemm_macs_per_cycle(fm)
        if op is Opcode.DOT:
            return float(d.m) / self.lanes(fm)
        if op is Opcode.NRMI:
            return float(d.m) / self.lanes(fm)
        # Elementwise: AXPY, SCAL, VMUL, PROJ walk M x N.
        return float(d.m) * float(max(d.n, 1)) / self.lanes(fv)

    def bandwidth_cycles(self, d: Descriptor) -> float:
        return self.bytes_touched(d) / self.bytes_per_cycle

    def cycles(self, d: Descriptor) -> float:
        """Datapath cycles for one descriptor: whichever resource binds."""
        return max(1.0, self.unit_cycles(d), self.bandwidth_cycles(d))

    def breakdown(self, d: Descriptor) -> dict[str, float]:
        u, b = self.unit_cycles(d), self.bandwidth_cycles(d)
        return {
            "unit_cycles": u,
            "bandwidth_cycles": b,
            "cycles": max(1.0, u, b),
            "bound": "bandwidth" if b > u else "unit",
            "macs": self.mac_count(d),
            "bytes": float(self.bytes_touched(d)),
        }

    def block_cycles(self, descriptors) -> float:
        return sum(self.cycles(d) for d in descriptors)


#: The baseline the contract is stated against.
CM100 = DatapathModel()

#: CM-50: 64 MACs (8x8), 64 B/cycle, 8 banks, 8 lanes, 4 fp64 FMAs, 64 KiB.
#: Present so a compliance report can be produced for the SKU that explicitly
#: does *not* support the QUAD-MPC class -- a generator that cannot say which
#: SKU its output targets is not much of a generator.
CM50 = DatapathModel(name="CM-50", bytes_per_cycle=64.0, macs=64, tile=8,
                     vector_lanes=8, fp64_fmas=4, banks=8)


# ---------------------------------------------------------------------------
# Bank arithmetic -- the quantitative basis for contract rule C2
# ---------------------------------------------------------------------------

def bank_spread(stride_bytes: int, banks: int, bank_word_bytes: int) -> int:
    """How many distinct banks a walk of constant `stride_bytes` ever touches.

    Bank = ``(address / bank_word) mod banks``, so a walk with element stride
    ``s = stride/bank_word`` visits the subgroup generated by ``s`` in
    ``Z_banks``, whose size is ``banks / gcd(s, banks)``. A stride that is not a
    whole number of bank words touches every bank, because successive accesses
    land at every residue.

    This is what makes rule C2 a number rather than an opinion. For QUAD-MPC's
    ``A`` at fp32 (``lda = 80``, so a column walk strides 320 B = 40 bank words
    of 8 B), 40 mod 16 = 8 and gcd(8, 16) = 8, so a column walk reaches
    **2 of 16 banks = 12.5% of peak bandwidth**. The 12.5% in the contract
    reproduces exactly; the "4 distinct banks" it is usually quoted with does
    not, and 2 is the number this configuration gives.
    """
    if banks <= 1:
        return 1
    if stride_bytes <= 0:
        return 1
    if stride_bytes % bank_word_bytes:
        return int(banks)
    s = (stride_bytes // bank_word_bytes) % banks
    if s == 0:
        return 1
    return int(banks // math.gcd(int(s), int(banks)))


@dataclass(frozen=True)
class ColumnWalkPenalty:
    """What a transposed GEMV would cost if it walked columns instead of rows."""

    stride_bytes: int
    banks_reached: int
    banks: int
    #: Fraction of peak scratchpad bandwidth a column walk delivers.
    delivered_fraction: float
    #: Accumulator elements a row-wise walk needs live in the vector register
    #: file. This is the price of doing it correctly, and it is one vector.
    accumulator_elements: int

    def summary(self) -> str:
        return (
            f"column walk strides {self.stride_bytes} B -> "
            f"{self.banks_reached}/{self.banks} banks = "
            f"{100 * self.delivered_fraction:.1f}% of peak; row-wise accumulation "
            f"needs a {self.accumulator_elements}-element accumulator"
        )


def column_walk_penalty(d: Descriptor, model: DatapathModel) -> ColumnWalkPenalty:
    """Quantify rule C2 for one transposed GEMV descriptor."""
    stride = d.lda * d.prec_m.itemsize
    reached = bank_spread(stride, model.banks, model.bank_word_bytes)
    return ColumnWalkPenalty(
        stride_bytes=stride,
        banks_reached=reached,
        banks=model.banks,
        delivered_fraction=reached / model.banks,
        accumulator_elements=d.k if d.transpose.a else d.m,
    )
