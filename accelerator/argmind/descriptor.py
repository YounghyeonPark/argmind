# SPDX-License-Identifier: Apache-2.0
"""Descriptor format **v0.1** -- the 8-word command word the sequencer hands the datapath.

v0.1 is the ruling on the seven gaps that building the functional simulator found
against frozen v0. It is still **8 words / 32 bytes**: one two-beat burst on a
128-bit bus, a power of two so queue indexing is a shift, three independent base
addresses. Descriptor traffic, queue SRAM and AGU count are all unchanged, which
is what made the change affordable -- see :data:`V0_1_BIT_BUDGET`.

Bit layout
----------
::

     w0  [31:26] opcode       6b  GEMM GEMV TRSV AXPY PROJ DOT SCAL CHOL NRMI VMUL
         [25:23] prec_m       3b  bulk / matrix operand format          (Format)
         [22:20] prec_v       3b  vector / result format             (VecFormat)
         [19:18] transpose    2b  none / A / B / both
         [17:15] structure    3b  dense / tril / triu / sym-lower / sym-upper
         [14]    accumulate   1b  C += (vs C =)
         [13]    unit_diag    1b
         [12:10] chain_token  3b  token this descriptor sets on completion (0-7)
         [9:2]   wait_mask    8b  bit t set = wait on token t
         [1:0]   reserved     2b  MUST be zero
     w1  A base address (scratchpad byte offset)
     w2  B / x base address
     w3  C / y base address
     w4  [31:16] M            [15:0] K
     w5  [31:16] N            [15:0] lda
     w6  [31:16] ldb          [15:0] ldc
     w7  alpha immediate (fp32 bit pattern)

What changed from v0, and what paid for it
------------------------------------------
============================== ======== ========= =======================================
field                          v0       v0.1      why
============================== ======== ========= =======================================
opcode                          6b       6b       unchanged; 10 of 64 values defined
precision -> prec_m             4b       3b       **-1b reclaimed.** 5 formats need 3.
prec_v                          --       3b       +3b. Fixes L4 (mixed precision).
transpose                       2b       2b       unchanged. B-transpose is real: EKF
                                                  needs ``A P A'``.
structure                       2b       3b       +1b for symmetric-upper.
accumulate                      1b       1b
unit_diag                       1b       1b
chain_token                     8b       3b       **-5b reclaimed.** A token is a
                                                  scoreboard slot, not a name.
wait_token -> wait_mask         8b       8b       same width, different meaning. Fixes L3
                                                  for merges up to 8-wide.
reserved                        --       2b       forward compatibility, must be zero.
============================== ======== ========= =======================================

Net: 32 bits in, 30 bits used, 2 spare. **The format change is paid for entirely
by reclaiming waste in v0's precision and token fields** -- no extra word, no
fourth AGU, no change to descriptor bandwidth.

Operand-address conventions per opcode
--------------------------------------
Three base addresses, and each opcode fits its operands into ``a`` / ``b`` / ``c``.
These assignments are part of the interface.

===== ================================= ====================== ==========================
op    a                                 b                      c
===== ================================= ====================== ==========================
GEMM  matrix A (prec_m)                 matrix B (prec_m)      matrix C, out (prec_v)
GEMV  matrix A (prec_m)                 vector x (prec_v)      vector y, out (prec_v)
TRSV  triangular matrix A (prec_m)      rhs b (prec_v)         solution x (prec_v; may
                                                               alias b)
AXPY  (unused)                          vector x               vector y, in/out
SCAL  (unused)                          vector x               vector y, out (may alias)
VMUL  vector u                          vector v               vector w, out
DOT   scalar result, out (prec_v)       vector x (prec_m)      vector y (prec_m)
NRMI  scalar result, out (prec_v)       vector x (prec_m)      (unused)
PROJ  **input** vector (prec_v)         **interleaved bounds** output vector (prec_v)
                                        (lo,hi) pairs (prec_v)
CHOL  matrix, in/out, factor in place   (unused)               (unused)
===== ================================= ====================== ==========================

``PROJ`` is the v0.1 ruling on L2 and is worth reading twice. v0 gave the bounds
two separate addresses and had none left for the input, making projection
in-place-only and forcing a staging copy. v0.1 instead requires the lower and
upper bounds to be **interleaved as (lo, hi) pairs in one array**, freeing an
address for the input. Projection becomes out-of-place with three addresses and
no fourth AGU. ``ldb`` is the element stride between successive pairs, normally 2.
At fp32 a pair is 8 bytes -- exactly one bank word -- so one bank access delivers
both bounds where v0 needed two accesses in two banks.

Elementwise ops are two-dimensional
-----------------------------------
``AXPY``, ``SCAL``, ``VMUL`` and ``PROJ`` walk **M rows x N columns** with leading
dimensions ``ldb`` / ``ldc`` (and ``lda`` where ``a`` is read). ``N = 0`` is read as
``N = 1``, so every v0 flat-vector descriptor decodes identically under v0.1.
This is the v0.1 ruling on L6: adding a matrix-elementwise capability to the
existing opcodes costs zero opcodes and zero bits, and lifts the cap on an
``n x n`` elementwise op from **n = 255 to n = 65535** without tiling.

Zero AGU stride
---------------
**Legal on reads, illegal on writes.** See :data:`V0_1_RULINGS` entry R6. A
zero-stride read is a broadcast and is what lets a scalar held in scratchpad --
at storage precision rather than w7's fp32 -- multiply a vector.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable

from .errors import FieldOverflow, UnsupportedOpcode
from .precision import Format, VecFormat

#: Version of the descriptor format this module implements.
DESCRIPTOR_VERSION = "0.1"

DESC_WORDS = 8
DESC_BYTES = DESC_WORDS * 4


class Opcode(IntEnum):
    """Descriptor opcodes. All ten are part of **v0.1**.

    ``NRMI`` and ``VMUL`` were proposals discovered by implementing v0 and are
    ruled in (:data:`V0_1_RULINGS` R1 and R7). The opcode field is 6 bits, so
    54 values remain: opcode space was never the scarce resource and adding
    these two cost nothing that any other ruling wanted.
    """

    GEMM = 0
    GEMV = 1
    TRSV = 2
    AXPY = 3
    PROJ = 4
    DOT = 5
    SCAL = 6
    CHOL = 7
    #: Max-absolute (infinity-norm) reduction, ``a[0] = alpha * max|x_i|``.
    #: OSQP's termination criterion is stated entirely in infinity norms.
    NRMI = 8
    #: Elementwise multiply, ``c = alpha * a * b``. Carries a per-constraint rho
    #: vector, and -- with a zero stride on ``a`` -- a scratchpad-resident scalar
    #: at storage precision instead of w7's fp32.
    VMUL = 9


#: Opcodes in v0.1. Retained as a named set because the device still has a
#: version gate; it is just that in v0.1 nothing is outside it.
V0_1_OPCODES: frozenset[Opcode] = frozenset(Opcode)

#: Opcodes that were in frozen v0. Kept so a strict-v0 device can still be
#: simulated, which is how the cost of each addition stays measurable.
FROZEN_V0_OPCODES: frozenset[Opcode] = frozenset(
    (Opcode.GEMM, Opcode.GEMV, Opcode.TRSV, Opcode.AXPY,
     Opcode.PROJ, Opcode.DOT, Opcode.SCAL, Opcode.CHOL)
)

#: Opcodes v0.1 adds.
V0_1_ADDED_OPCODES: frozenset[Opcode] = V0_1_OPCODES - FROZEN_V0_OPCODES


class Transpose(IntEnum):
    NONE = 0
    A = 1
    B = 2
    BOTH = 3

    @property
    def a(self) -> bool:
        return self in (Transpose.A, Transpose.BOTH)

    @property
    def b(self) -> bool:
        return self in (Transpose.B, Transpose.BOTH)


class Structure(IntEnum):
    """Operand structure. Widened to 3 bits in v0.1 to add symmetric-upper.

    ``SYMMETRIC`` is retained as an alias of :attr:`SYM_LOWER` because that is
    what v0's value 3 meant, but the wire encoding is renumbered: v0's 2-bit
    field had no room for a fifth value, so v0.1 assigns them afresh. A v0
    descriptor's structure bits therefore do **not** decode as the same
    structure under v0.1, and that is the one place the format is not
    bit-compatible. Recorded here rather than glossed.
    """

    DENSE = 0
    TRIL = 1              # triangular, lower
    TRIU = 2              # triangular, upper
    SYM_LOWER = 3         # symmetric, data in the lower triangle
    SYMMETRIC = 3         # v0 spelling; same meaning, alias of SYM_LOWER
    SYM_UPPER = 4         # symmetric, data in the upper triangle

    @property
    def is_symmetric(self) -> bool:
        return self in (Structure.SYM_LOWER, Structure.SYM_UPPER)

    @property
    def reads_lower(self) -> bool:
        """True when the meaningful data lives in the lower triangle."""
        return self in (Structure.TRIL, Structure.SYM_LOWER)


# --------------------------------------------------------------------------
# Bit layout
# --------------------------------------------------------------------------

_W0_FIELDS = (
    # (name, shift, width)
    ("opcode", 26, 6),
    ("prec_m", 23, 3),
    ("prec_v", 20, 3),
    ("transpose", 18, 2),
    ("structure", 15, 3),
    ("accumulate", 14, 1),
    ("unit_diag", 13, 1),
    ("chain_token", 10, 3),
    ("wait_mask", 2, 8),
    ("reserved", 0, 2),
)

#: Total w0 bits each field claims, and the spare. Asserted by a test, because
#: "w0 has zero spare bits" was the fact that made L3 and L4 look unaffordable,
#: and the whole v0.1 change rests on it no longer being true.
V0_1_BIT_BUDGET = {
    "used": sum(w for name, _, w in _W0_FIELDS if name != "reserved"),
    "reserved": sum(w for name, _, w in _W0_FIELDS if name == "reserved"),
    "total": 32,
    #: What v0 spent, for the comparison that justifies the change.
    "v0_used": 32,
    "v0_spare": 0,
}

_MASK32 = 0xFFFFFFFF
_MASK16 = 0xFFFF

#: Widths of the packed dimension/stride fields, in bits.
DIM_BITS = 16
DIM_MAX = (1 << DIM_BITS) - 1

#: Number of dependency tokens. A token is a **scoreboard slot, not a name**:
#: see :data:`V0_1_RULINGS` R3. Measured peak simultaneous liveness on the ADMM
#: blocks is 7 (iteration), 3 (residual), 3 (setup).
N_TOKENS = 8
TOKEN_MAX = N_TOKENS - 1
WAIT_MASK_MAX = (1 << N_TOKENS) - 1

#: Opcodes whose destination at ``c`` is a vector or matrix, so ``ldc = 0`` would
#: mean many writers and one address -- no defined order, therefore illegal.
_VECTOR_DEST_OPS = frozenset(
    (Opcode.GEMM, Opcode.GEMV, Opcode.TRSV, Opcode.AXPY,
     Opcode.SCAL, Opcode.PROJ, Opcode.VMUL)
)

#: Opcodes that are elementwise and therefore walk M x N under v0.1.
ELEMENTWISE_OPS = frozenset(
    (Opcode.AXPY, Opcode.SCAL, Opcode.VMUL, Opcode.PROJ)
)

#: Opcodes that reduce to a single element written at ``a``.
REDUCTION_OPS = frozenset((Opcode.DOT, Opcode.NRMI))


def _check(name: str, value: int, width: int) -> int:
    if not isinstance(value, (int,)) or isinstance(value, bool):
        raise FieldOverflow(f"{name} must be a plain int, got {value!r}")
    if value < 0 or value > (1 << width) - 1:
        raise FieldOverflow(
            f"descriptor v0.1 field {name!r} is {width} bits "
            f"(max {(1 << width) - 1}); got {value}"
        )
    return value


def _f32_bits(x: float) -> int:
    return int(struct.unpack("<I", struct.pack("<f", float(x)))[0])


def _bits_f32(bits: int) -> float:
    return float(struct.unpack("<f", struct.pack("<I", bits & _MASK32))[0])


@dataclass(frozen=True)
class Descriptor:
    """One descriptor in format v0.1.

    All addresses are scratchpad byte offsets. All dimensions are element counts;
    leading dimensions are in elements, not bytes, as is conventional for BLAS.

    ``precision`` is the **bulk / matrix operand** format (wire field ``prec_m``).
    ``vec_precision`` is the vector / result format (wire field ``prec_v``), and
    defaults to :attr:`VecFormat.SAME`, which is what every v0 descriptor meant.

    ``alpha`` round-trips through an **fp32** bit pattern because that is what w7
    is. v0.1 does not widen it, and the reason is worth stating: the failure that
    made w7 look too narrow (L5, the stalled solve) was diagnosed as *fp32 alpha
    against fp64 data*, and the fix is not a wider immediate but to keep any
    scalar that must be exact **relative to another scalar** in scratchpad, where
    it is stored at the buffer's own precision. A zero-stride ``VMUL`` reads it
    from there. w7 stays what it is good for: a standalone tuning constant.
    """

    opcode: Opcode
    precision: Format = Format.FP64
    vec_precision: VecFormat = VecFormat.SAME
    a: int = 0
    b: int = 0
    c: int = 0
    m: int = 0
    k: int = 0
    n: int = 0
    #: Leading dimension / stride of the operand at ``a``, in elements.
    #: **Defaults to 1, not 0.** In v0 the default was 0 and meant "unused",
    #: which was harmless because no v0 opcode read ``a`` as a strided operand.
    #: v0.1's VMUL and PROJ do, and under ruling R6b a stride of 0 means
    #: *broadcast* -- so a caller who forgot to set ``lda`` would have got a
    #: silent one-element broadcast instead of a vector. Defaulting to 1 makes
    #: the broadcast something you have to ask for.
    lda: int = 1
    ldb: int = 1
    ldc: int = 1
    alpha: float = 1.0
    transpose: Transpose = Transpose.NONE
    structure: Structure = Structure.DENSE
    accumulate: bool = False
    unit_diag: bool = False
    chain_token: int = 0
    wait_mask: int = 0
    #: Not encoded. Purely a label carried alongside for readable traces and
    #: error messages; the hardware descriptor has no room for it.
    label: str = ""

    def __post_init__(self) -> None:
        # Validate everything the bit layout and the semantic rules constrain, at
        # construction time, so a code generator cannot emit an illegal descriptor.
        self.encode()
        self._check_strides()

    # -- semantic rules ---------------------------------------------------

    def _check_strides(self) -> None:
        """Enforce the v0.1 zero-stride ruling (R6): reads may, writes may not."""
        if self.opcode in _VECTOR_DEST_OPS and self.ldc == 0:
            raise FieldOverflow(
                f"{self.opcode.name}: ldc = 0 writes every element of the "
                f"destination to one address. There is no defined order, so it is "
                f"ILLEGAL in descriptor v0.1. A zero stride is legal on READ "
                f"operands only (it means broadcast); the legitimate many-to-one "
                f"write is DOT or NRMI, which have defined reduction semantics."
            )
        if self.opcode is Opcode.CHOL and self.lda == 0:
            raise FieldOverflow(
                "CHOL: lda = 0 aliases every row of an in-place factorisation "
                "onto row 0. Illegal for the same reason as ldc = 0."
            )

    # -- resolved formats -------------------------------------------------

    @property
    def prec_m(self) -> Format:
        """Format of the bulk / matrix operand."""
        return self.precision

    @property
    def prec_v(self) -> Format:
        """Concrete format of vectors and results, with ``SAME`` resolved."""
        return self.vec_precision.resolve(self.precision)

    @property
    def is_mixed_precision(self) -> bool:
        return self.prec_v is not self.prec_m

    # -- shape ------------------------------------------------------------

    @property
    def rows(self) -> int:
        """Row count for an elementwise op."""
        return self.m

    @property
    def cols(self) -> int:
        """Column count for an elementwise op. ``N = 0`` reads as ``N = 1``, so
        every v0 flat-vector descriptor keeps its v0 meaning."""
        return max(self.n, 1)

    @property
    def elements(self) -> int:
        """Element count an elementwise op touches."""
        return self.m * self.cols

    # -- encoding ---------------------------------------------------------

    def encode(self) -> tuple[int, ...]:
        """Encode to 8 unsigned 32-bit words."""
        w0 = 0
        values = {
            "opcode": int(self.opcode),
            "prec_m": int(self.precision),
            "prec_v": int(self.vec_precision),
            "transpose": int(self.transpose),
            "structure": int(self.structure),
            "accumulate": int(bool(self.accumulate)),
            "unit_diag": int(bool(self.unit_diag)),
            "chain_token": int(self.chain_token),
            "wait_mask": int(self.wait_mask),
            "reserved": 0,
        }
        for name, shift, width in _W0_FIELDS:
            w0 |= _check(name, values[name], width) << shift

        for name in ("a", "b", "c"):
            _check(name, int(getattr(self, name)), 32)
        for name in ("m", "k", "n", "lda", "ldb", "ldc"):
            _check(name, int(getattr(self, name)), DIM_BITS)

        w4 = (self.m << 16) | self.k
        w5 = (self.n << 16) | self.lda
        w6 = (self.ldb << 16) | self.ldc
        w7 = _f32_bits(self.alpha)
        return (w0, int(self.a), int(self.b), int(self.c), w4, w5, w6, w7)

    def to_bytes(self) -> bytes:
        return struct.pack("<8I", *self.encode())

    @classmethod
    def decode(cls, words: Iterable[int]) -> "Descriptor":
        """Decode 8 words back into a Descriptor. Inverse of :meth:`encode`."""
        w = tuple(int(x) & _MASK32 for x in words)
        if len(w) != DESC_WORDS:
            raise FieldOverflow(f"expected {DESC_WORDS} words, got {len(w)}")
        get = {
            name: (w[0] >> shift) & ((1 << width) - 1)
            for name, shift, width in _W0_FIELDS
        }
        if get["reserved"]:
            raise FieldOverflow(
                f"descriptor v0.1 reserved bits w0[1:0] are {get['reserved']}, "
                f"must be zero. A future version assigns them; refusing rather "
                f"than ignoring is what makes that possible."
            )
        return cls(
            opcode=Opcode(get["opcode"]),
            precision=Format(get["prec_m"]),
            vec_precision=VecFormat(get["prec_v"]),
            transpose=Transpose(get["transpose"]),
            structure=Structure(get["structure"]),
            accumulate=bool(get["accumulate"]),
            unit_diag=bool(get["unit_diag"]),
            chain_token=get["chain_token"],
            wait_mask=get["wait_mask"],
            a=w[1], b=w[2], c=w[3],
            m=w[4] >> 16, k=w[4] & _MASK16,
            n=w[5] >> 16, lda=w[5] & _MASK16,
            ldb=w[6] >> 16, ldc=w[6] & _MASK16,
            alpha=_bits_f32(w[7]),
        )

    @classmethod
    def from_bytes(cls, raw: bytes) -> "Descriptor":
        return cls.decode(struct.unpack("<8I", raw))

    # -- tokens -----------------------------------------------------------

    @property
    def waits_on(self) -> tuple[int, ...]:
        """The tokens this descriptor waits on, from its mask."""
        return tuple(t for t in range(N_TOKENS) if self.wait_mask & (1 << t))

    @property
    def merge_width(self) -> int:
        """How many independent producers this descriptor's mask orders."""
        return len(self.waits_on)

    # -- helpers ----------------------------------------------------------

    @property
    def is_frozen_v0(self) -> bool:
        """True if the opcode existed in frozen v0. Kept so the cost of each
        v0.1 addition stays measurable against a strict-v0 device."""
        return self.opcode in FROZEN_V0_OPCODES

    def require_frozen_v0(self) -> None:
        if not self.is_frozen_v0:
            raise UnsupportedOpcode(
                f"opcode {self.opcode.name} is a descriptor v0.1 addition, not "
                f"part of frozen v0. Set descriptor_version='0.1' on the device "
                f"to run it. This gate exists so that the cost of each addition "
                f"can still be measured; it is not a deprecation."
            )

    @property
    def alpha_f32(self) -> float:
        """The value of `alpha` as it survives the fp32 w7 field."""
        return _bits_f32(_f32_bits(self.alpha))

    def __str__(self) -> str:
        bits = [self.opcode.name, str(self.precision)]
        if self.vec_precision is not VecFormat.SAME:
            bits.append(f"->{self.vec_precision}")
        if self.transpose is not Transpose.NONE:
            bits.append(f"T={self.transpose.name}")
        if self.structure is not Structure.DENSE:
            bits.append(self.structure.name)
        if self.accumulate:
            bits.append("acc")
        dims = ",".join(
            f"{n}={getattr(self, n)}" for n in ("m", "k", "n") if getattr(self, n)
        )
        tok = f"->t{self.chain_token}"
        wait = f" wait{{{','.join(str(t) for t in self.waits_on)}}}" if self.wait_mask else ""
        name = f" {self.label}" if self.label else ""
        return f"{' '.join(bits)} [{dims}]{tok}{wait}{name}"


# ---------------------------------------------------------------------------
# What implementing v0 revealed, and the v0.1 ruling on each. Kept in code, next
# to the format, so it cannot drift away from it.
# ---------------------------------------------------------------------------

V0_LIMITATIONS: tuple[dict[str, str], ...] = (
    {
        "id": "L1",
        "severity": "blocking for reference-solver equivalence",
        "what": "No max-absolute (infinity-norm) reduction opcode.",
        "why": "OSQP's termination criterion is stated entirely in infinity "
               "norms. v0 offers DOT, i.e. a 2-norm. Measured cost of the "
               "v0-legal substitute on QUAD-MPC: same 15 descriptors, one "
               "opcode changed, but a different iteration count (8 vs 9) and "
               "3.7x worse control action at the same tolerance.",
        "fix": "One opcode (NRMI). No new fields, no new base addresses.",
        "ruling": "R1 -- ADDED.",
    },
    {
        "id": "L2",
        "severity": "restricts kernels to in-place form",
        "what": "Only three base addresses, so out-of-place box projection "
                "(input, lower, upper, output) cannot be expressed.",
        "why": "PROJ must clip in place, with the value pre-staged in the "
               "destination by a preceding SCAL: one extra descriptor and one "
               "extra full-vector copy per projection.",
        "fix": "A fourth AGU, or a packing convention that frees an address.",
        "ruling": "R2 -- FIXED WITHOUT A FOURTH AGU. Bounds interleave as "
                  "(lo,hi) pairs at `b`; `a` becomes the input.",
    },
    {
        "id": "L3",
        "severity": "removes achievable overlap",
        "what": "Exactly one wait token per descriptor.",
        "why": "A descriptor with several independent producers cannot express "
               "them, so codegen must serialise. Emitted vs achievable "
               "dependency depth: iteration 17 vs 12, residual 15 vs 5.",
        "fix": "A second wait token, or wait-mask semantics, or an explicit "
               "statement that codegen owns serialisation.",
        "ruling": "R3 -- WAIT MASK. Note the measured latency value is 1.5%, "
                  "not 1.42x; adopted because it costs -5 bits, not for speed.",
    },
    {
        "id": "L4",
        "severity": "blocks the recommended numerics",
        "what": "One precision field, so operand and result formats are "
                "necessarily the same.",
        "why": "'fp16 operands, fp32 triangular solve' is inexpressible -- not "
               "for an arithmetic reason but a STORAGE one: a buffer written by "
               "an fp16 GEMV is read by an fp32 TRSV. w0 had zero spare bits.",
        "fix": "A second precision field, a second header word, or declaring "
               "the accumulator implied by MACIF (which does not fix the store).",
        "ruling": "R4 -- SECOND FIELD (prec_v, 3b), paid for out of reclaimed "
                  "token and prec_m bits. Accumulator stays a MACIF declaration.",
    },
    {
        "id": "L5",
        "severity": "SILENT WRONG ANSWER unless codegen constrains rho",
        "what": "w7 is an fp32 bit pattern, so a kernel cannot receive a matched "
                "pair of reciprocal scalars unless they are exact in fp32.",
        "why": "ADMM multiplies by 1/rho and by rho. At a fixed point with an "
               "active constraint y = rho*(1/rho)*y, so if fl32(rho)*fl32(1/rho) "
               "!= 1 the only solution is y = 0 and the solve stalls at a "
               "residual floor. rho=0.1 stalls at 1500 iterations; rho=0.125 "
               "converges in 34. Invisible when the optimum is interior.",
        "fix": "'rho must be a power of two' in the codegen contract.",
        "ruling": "R5 -- ROOT CAUSE RE-DIAGNOSED. It is not rho, it is an fp32 "
                  "immediate against fp64 data. Scalars in a fixed point move to "
                  "scratchpad (zero-stride VMUL); w7 keeps standalone constants. "
                  "Power-of-two rho stays recommended, stops being load-bearing.",
    },
    {
        "id": "L6",
        "severity": "bounded, but reachable",
        "what": "16-bit dimension and leading-dimension fields.",
        "why": "Treating an n x n matrix as a flat vector for an elementwise op "
               "(which v0 forces, having no matrix-add) caps n at 255. "
               "QUAD-MPC at n=80 fits; a 300-variable condensed QP would not.",
        "fix": "Tile flat elementwise ops in codegen, or widen the dims.",
        "ruling": "R6a -- ELEMENTWISE OPS BECOME 2-D. Zero opcodes, zero bits, "
                  "cap goes 255 -> 65535 per dimension.",
    },
    {
        "id": "L7",
        "severity": "affects ADMM on problems with equality constraints",
        "what": "No elementwise vector multiply, and only a scalar rho field.",
        "why": "OSQP uses a per-constraint rho vector with equality rows scaled "
               "by 1e3, so sparse-form ADMM cannot be expressed.",
        "fix": "One opcode (VMUL), three addresses, no new fields.",
        "ruling": "R7 -- ADDED, but the stated scope was WRONG: ARM-MPC is "
                  "HPIPM, an interior-point method, which has no rho at all. v0 "
                  "never ruled out the sparse form for ARM-MPC.",
    },
)


V0_1_RULINGS: tuple[dict[str, str], ...] = (
    {
        "id": "R1",
        "closes": "L1",
        "decision": "ADD the NRMI opcode. Required, not optional.",
        "cost": "One of 54 unused opcode values. No new fields, no new base "
                "address, no new AGU mode. The compare-select reduction tree is "
                "CHEAPER than DOT's, which needs a multiplier; NRMI reuses the "
                "same reduction network with one extra mux per node.",
        "why": "The benefit is equivalence with the reference solver at the same "
               "tolerance -- 3.7x better control action, and an iteration count "
               "a customer can check against OSQP. At this cost it does not need "
               "a stronger argument than that.",
        "correction": "The argument that WITHOUT NRMI 'the convergence test "
                      "leaves the fabric' is NOT TRUE and must not be pitched. "
                      "The 2-norm test also runs in fabric: DOT plus two sqrt on "
                      "the FMA cluster plus compares on the sequencer, no host "
                      "round trip. What the substitute costs is a DIFFERENT "
                      "ANSWER, not a host round trip. Keep the claim that "
                      "survives.",
    },
    {
        "id": "R2",
        "closes": "L2",
        "decision": "REFUSE a fourth base address. PROJ takes interleaved "
                    "(lo,hi) bound pairs at `b`, input at `a`, output at `c`.",
        "cost": "Codegen interleaves the two bound vectors, which is free: "
                "allocation is static and the host writes both every tick "
                "anyway.",
        "why": "A fourth AGU and a fourth address word would cost a 50% wider "
               "descriptor (32 -> 48 B, and 48 is not a power of two, which "
               "breaks queue-index-by-shift) for one opcode. Interleaving is "
               "strictly better than break-even: PROJ becomes out-of-place, the "
               "staging SCAL disappears, and at fp32 a (lo,hi) pair is 8 bytes = "
               "exactly one bank word, so one bank access delivers both bounds "
               "where v0 needed two accesses in two banks.",
        "general_rule": "v0 was never short of ADDRESSES -- `a` is unused in "
                        "AXPY and SCAL. It was short of an address IN ONE "
                        "OPCODE. Fix that with a per-opcode operand convention, "
                        "not with format-wide width.",
    },
    {
        "id": "R3",
        "closes": "L3",
        "decision": "REPLACE {chain_token 8b, wait_token 8b} with "
                    "{chain_token 3b, wait_mask 8b}. Net -5 bits, and merges up "
                    "to 8-wide become expressible.",
        "cost": "Negative. The five reclaimed bits are what pay for prec_v and "
                "the widened structure field.",
        "why": "A token is a SCOREBOARD SLOT, NOT A NAME. It must only be unique "
               "among simultaneously live dependencies, and 255 of them was "
               "waste. Measured peak simultaneous liveness on the real ADMM DAG: "
               "7 (iteration), 3 (residual), 3 (setup) -- so 8 tokens with one "
               "spare at block granularity. Reuse is SAFE: a waiter takes the "
               "most recent producer of a token, which can only over-serialise, "
               "never under-serialise. So exhausting the token space degrades "
               "overlap and CANNOT produce a wrong answer.",
        "adversarial": "The 1.42x / 3.00x depth penalties are real but they are "
                       "CHAIN STEPS, not cycles, and on this workload a chain "
                       "step is ~228 datapath cycles. Measured on the emitted "
                       "DAG, the mask removes 120 of 7881 cycles per checked "
                       "QUAD-MPC iteration = 1.5%. The depth-ratio metric "
                       "overstates the value by ~100x. The mask is adopted "
                       "because it costs NEGATIVE BITS, not because it is fast. "
                       "It pays where cycles/descriptor approaches drain_cycles "
                       "(8): total drain is 10.6% of ARM-MPC latency at 30.3 "
                       "cycles/descriptor, of which the reducible share is NOT "
                       "MEASURED, because no HPIPM descriptor program exists yet.",
    },
    {
        "id": "R4",
        "closes": "L4",
        "decision": "ADD prec_v (3b): the format of vectors and results, with "
                    "prec_m (3b) giving the bulk/matrix operand format.",
        "cost": "+3b for prec_v, -1b by narrowing prec_m from 4b to 3b (five "
                "formats never needed four bits, and the VALUES do not change). "
                "Net +2b, paid from R3's reclaim.",
        "why": "The accumulator half of the cheap answer is CORRECT and is kept: "
               "accumulator width stays a MACIF declaration and is not a "
               "descriptor field. What it cannot fix is the STORE format, and "
               "that is what the numerics need -- the mixed scheme fails because "
               "a buffer written by an fp16 GEMV is read by an fp32 TRSV, which "
               "is a storage-type conflict, not an arithmetic one.",
        "limit": "Exactly two formats per descriptor. Three (int8 matrix, bf16 "
                 "vector, fp32 result) needs two descriptors. Nothing in the "
                 "frozen set wants three.",
    },
    {
        "id": "R5",
        "closes": "L5",
        "decision": "w7 stays fp32. Scalars that must be EXACT RELATIVE TO "
                    "ANOTHER SCALAR move to scratchpad and are read by a "
                    "zero-stride VMUL at the buffer's own precision.",
        "why": "Re-diagnosis: the failure is not rho and not the descriptor "
               "field width per se -- it is an fp32 immediate multiplying fp64 "
               "data in a fixed point that requires two constants to be exact "
               "reciprocals. OSQP runs happily at rho=0.1 because it keeps rho "
               "and rho^-1 as fp64 VECTORS, where the mismatch is ~1e-16 rather "
               "than 1.5e-8. Widening w7 to fp64 would cost a word and still not "
               "give exactness; moving the scalar to storage precision reduces "
               "the mismatch to below any realistic tolerance.",
        "contract": "'rho must be a power of two' stays RECOMMENDED (it is free "
                    "and it makes the mismatch exactly zero) but stops being "
                    "load-bearing once rho is a scratchpad vector.",
    },
    {
        "id": "R6a",
        "closes": "L6",
        "decision": "AXPY, SCAL, VMUL and PROJ walk M rows x N columns with "
                    "leading dimensions. N = 0 reads as N = 1.",
        "cost": "Zero opcodes, zero bits. The AGU needs a 2-level walk, which "
                "the hardware descriptor generator already has (3-level loop).",
        "why": "Flattening an n x n matrix into a vector was the thing that "
               "capped n at 255. Making the existing elementwise opcodes 2-D "
               "removes the flattening instead of raising the cap, and lifts the "
               "limit to 65535 per dimension. `N = 0 reads as N = 1` means every "
               "v0 flat-vector descriptor decodes identically under v0.1.",
    },
    {
        "id": "R6b",
        "closes": "the open semantic question",
        "decision": "ZERO AGU STRIDE IS LEGAL ON READS, ILLEGAL ON WRITES. "
                    "Enforced by the descriptor constructor.",
        "why": "A zero-stride read is a broadcast and buys three things: the "
               "diagonal add drops its sigma vector and n-1 redundant fetches "
               "(the case that prompted the question); a scratchpad-resident "
               "scalar at STORAGE precision can multiply a vector, which is the "
               "mechanism R5 depends on; and a GEMM bias vector can broadcast "
               "across columns, complementing MACIF broadcast_init. A "
               "zero-stride WRITE is n writers on one address with no defined "
               "order -- the legitimate many-to-one is DOT/NRMI, which have "
               "defined reduction semantics.",
        "rtl_requirement": "A zero-stride read must be satisfied from the L0 "
                           "operand staging register, NOT re-fetched. n reads of "
                           "one bank word otherwise serialise at one word per "
                           "cycle per bank and the broadcast becomes slower than "
                           "the materialised vector it replaces. This ties the "
                           "ruling to the L0 operand register file already "
                           "flagged as the top model extension.",
        "model_note": "The bandwidth model charges n*itemsize for an n-element "
                      "read and so OVERCHARGES a broadcast. Left as is: the "
                      "error is conservative and the traffic is negligible.",
    },
    {
        "id": "R7",
        "closes": "L7",
        "decision": "ADD the VMUL opcode, and carry rho as a scratchpad vector "
                    "rather than a w7 scalar.",
        "cost": "One opcode value, three existing addresses, no new fields.",
        "correction": "The stated justification was WRONG and must not be "
                      "repeated. 'Scalar-only rho rules out the sparse MPC form "
                      "acados and HPIPM use' conflates two solvers: rho is "
                      "ADMM's penalty parameter, and HPIPM is an interior-point "
                      "method with no rho. ARM-MPC's sparse form needs "
                      "BLOCK-STRUCTURED descriptors on per-stage n_b x n_b "
                      "blocks, which v0 already expressed with three addresses "
                      "-- it is the case that gets 0.04 cycles/descriptor from "
                      "the hardware generator. What VMUL actually enables is "
                      "SPARSE-FORM ADMM (OSQP on the non-condensed QP, equality "
                      "rows at 1e3*rho), which is a legitimate target and worth "
                      "an opcode at this price. Do not claim it unblocks ARM-MPC.",
    },
    {
        "id": "R8",
        "closes": "the scratchpad contradiction",
        "decision": "GROW THE SCRATCHPAD TO 192 KiB (16 banks x 12 KiB). "
                    "Bandwidth stays 128 B/cycle and bank count stays 16, so "
                    "this is a CAPACITY correction, not a rebalancing.",
        "why": "See :data:`SCRATCHPAD_RULING`.",
    },
)


#: The scratchpad ruling, kept next to the format because the measurement that
#: forced it came from the allocation plan the format addresses.
SCRATCHPAD_RULING: dict[str, object] = {
    "verdict": "128 KiB was wrong. CM-100 becomes 192 KiB, 16 banks x 12 KiB.",
    "authority": "The MEASURED allocation plan is authoritative over the model, "
                 "per the instruction and per the evidence standard that a "
                 "flattering model result means find the assumption first.",
    #: Measured from the allocation plan under **descriptor v0**, i.e. the numbers
    #: the build report quoted. These are the figures the ruling is made against.
    "measured": {
        "QUAD-MPC fp16": 83896,
        "QUAD-MPC fp32": 167736,
        "QUAD-MPC fp64": 335416,
    },
    #: The same plan under v0.1, which is 296 B smaller only because ruling R6b
    #: deletes the materialised sigma*ones(n) vector. Recorded so nobody mistakes
    #: the format change for a capacity remedy: it moved the footprint by 0.2%.
    "measured_v0_1": {
        "QUAD-MPC fp16": 83760,
        "QUAD-MPC fp32": 167440,
        "QUAD-MPC fp64": 334800,
    },
    "model_said": 102400,
    "decomposition_fp32": (
        "102400 B model + 25600 (A's identity block materialised) "
        "+ 12800 (P stored as a dense square, not packed) "
        "+ 12800 (L stored as a dense square, not packed) "
        "+ 14136 (working vectors and residual temporaries, which the model "
        "counted as ZERO) = 167736 B. The decomposition is exact -- no residual "
        "term, which is what makes it a diagnosis rather than an estimate."
    ),
    "model_errors_found": (
        "TWO, both flattering. (1) `persistent_bytes` charged P and L at n^2/2 "
        "each -- a PACKED triangular layout that base+leading-dimension "
        "addressing CANNOT EXPRESS. The capacity model assumed a storage layout "
        "the descriptor format cannot address. (2) It counted zero bytes for "
        "working vectors and temporaries, which are 13.8 KiB at fp32 -- 10% of "
        "the scratchpad, structurally invisible because the field was named "
        "`persistent_bytes` and vectors are not persistent."
    ),
    "fp64_is_not_the_target": (
        "327.6 KiB at fp64 is a VALIDATION ARTEFACT, not a product "
        "configuration: the implementation runs fp64 to match OSQP bit-for-bit. "
        "ADMM's measured control-action error at fp32 is 4e-7. fp64 is required "
        "by interior-point conditioning (ARM-MPC), not by ADMM, so QUAD-MPC at "
        "fp64 does not size the scratchpad and must not be quoted as if it did."
    ),
    "sizing_basis_fp32": 142136,
    "sizing_basis_note": (
        "142136 B = 138.8 KiB: the honest fp32 footprint with A = [I;Gamma] "
        "exploited (the one remedy that is a strict win, since the identity "
        "block was also being multiplied by for nothing) but with P and L as "
        "dense squares, because that is what the AGU can address."
    ),
    "why_128_is_arithmetically_impossible": (
        "The flagship's THREE MATRICES ALONE are Gamma 76800 + P 25600 + L 25600 "
        "= 128000 B = 125.0 KiB of a 128.0 KiB scratchpad, leaving 3 KiB for "
        "every vector. Even after aliasing every temporary that can legally "
        "alias (rp onto Ax, rd onto Px, zt onto Ax, t onto t_dual, xt onto rhs, "
        "and sigma_diag deleted by R6b) the total is 134.1 KiB. The conclusion "
        "does not rest on the implementation's profligate vector allocation."
    ),
    "rejected": {
        "packed triangular AGU": "Saves 12.8 KiB -> 125.7 KiB = 95.9% full. A "
                                 "4% margin on a +/-2x modelled footprint is not "
                                 "a margin. Keep as a later optimisation, not as "
                                 "a sizing mechanism.",
        "overlay P and L in one square": "Saves 25.6 KiB but needs THREE new "
                                         "hardware behaviours (triangle-masked "
                                         "GEMM output, symmetric-from-strict-"
                                         "upper GEMV, transposed triangular "
                                         "elementwise add) plus a separate "
                                         "diagonal buffer, because P's diagonal "
                                         "and L's diagonal collide. Control "
                                         "logic exceeds 0.026 mm2 of SRAM.",
        "eliminate P algebraically": "P*xt = sigma(x-xt) - q + A'(rho*z - y - "
                                     "rho*zt) removes P entirely (-> 113.8 KiB) "
                                     "but costs an extra Gamma' GEMV per check "
                                     "(+44% on the residual block) and returns a "
                                     "residual that is NOT OSQP's. Rejected for "
                                     "the same reason R1 rejects the 2-norm "
                                     "substitute; accepting one and refusing the "
                                     "other would be incoherent.",
        "require fp16": "Lands at 101.3 KiB and fits, but fp16 operands are an "
                        "OPT-IN codegen mode (PHASE0_FINDINGS 0.3), so they "
                        "cannot be the default sizing basis.",
        "stream the overflow from DRAM": "Only 0.026 GB/s at 1 kHz, so the "
                                         "BANDWIDTH is trivial -- but it puts "
                                         "DRAM in the flagship control loop and "
                                         "forfeits the determinism claim on the "
                                         "one row where determinism is the "
                                         "product. Retained only as an explicit "
                                         "small-SKU option, never as the default.",
        "160 KiB": "Fits at 86.8% but is DOMINATED by 192 KiB: 10 KiB per bank "
                   "is a 1280x64b macro with array efficiency 0.354, against "
                   "0.394 for 12 KiB (1536x64b). +0.020 mm2 buys +32 KiB.",
    },
    "cost_of_192": {
        "scratchpad_mm2": "0.2026 -> 0.2792 (+0.0765)",
        "core_mm2": "0.698 -> 0.786 (+12.6%)",
        "sram_pj_per_byte": "0.718 -> 0.839",
        "leakage_mw_85C": "0.52 -> 0.78",
        "solver_energy": "+5.5%",
        "quad_mpc_perf_per_w": "14.13x -> 12.55x",
        "quad_mpc_latency": "36.0 us, UNCHANGED -- capacity was never the "
                            "binding constraint, bandwidth is, and bandwidth "
                            "does not move",
    },
    "headroom": "138.8 KiB of 192 KiB = 72.3%. Also holds the NAIVE 163.8 KiB "
                "layout at 85.3%, so the flagship fits even from a code "
                "generator that has not done the A = [I;Gamma] optimisation. "
                "That robustness is most of the argument for 192 over 160.",
    "not_affected": "ARM-MPC overflows at any of these capacities and always "
                    "did; that was already accepted on the ground that it runs "
                    "at 250 Hz with 24x deadline margin and 0.04 GB/s of "
                    "streaming demand. It does not drive capacity. QUAD-MPC "
                    "does, because it is the 1 kHz flagship where determinism is "
                    "the product.",
    "label": "MODELLED ESTIMATE, 16nm-class, +/-2x, NOT QUOTABLE. The working-set "
             "BYTES are measured (from the allocation plan); the area, energy "
             "and perf/W consequences are modelled.",
}


#: The sequencer contract, re-checked against the measured descriptor count.
SEQUENCER_RECHECK: dict[str, object] = {
    "measured_count": "32 descriptors per CHECKED ADMM iteration (17 iteration + "
                      "15 residual), against the 28 the sequencer analysis "
                      "assumed. 47% of them are the convergence test.",
    "why_the_model_said_28": (
        "The model amortises the residual over `check_term=25`, giving 17.6 "
        "descriptors per average iteration; 28 came from its own kernel "
        "decomposition, which sits between. But check_term=25 is WRONG FOR THIS "
        "WORKLOAD: QUAD-MPC converges in 6-14 iterations, so checking every 25th "
        "means never checking, i.e. always running to max_iterations. The "
        "implementation's check_every=1 is right and the model's 25 is not."
    ),
    "verdict_on_the_8_cycle_bar": (
        "NOT THREATENED, but the fp32 figure recorded here was FLATTERING and is "
        "corrected (codegen finding G3, 2026-07-31). It said 238.3 "
        "cycles/descriptor; recounted from the generator's emitted descriptors "
        "it is 137.5, i.e. 1.7x too high. The error: 238.3 was obtained by "
        "HALVING the fp64 figure, but the two precisions are bound by different "
        "resources -- fp32 GEMV is bandwidth-bound at 32 MAC/cycle while fp64 "
        "GEMV is ARRAY-bound at the 8-unit FMA cluster, a 4x rate change, not "
        "2x. Verdict is unchanged: 17.2x the >=8 bar rather than 29.8x. Note "
        "also that this entry paired an fp64 AVERAGE with an fp32 WORST CASE "
        "(the run of six averaging 5.83 cycles, which reproduces exactly and is "
        "genuinely fp32). At fp64 there is no sub-threshold descriptor at all, "
        "so the thinnest point in the sequencer analysis is a property of the "
        "PRECISION, not of the algorithm."
    ),
    "the_47_percent_is_not_the_problem_it_looks_like": (
        "The residual is 47% of descriptors and 49% of CYCLES, so on average it "
        "is exactly as heavy per descriptor as the iteration -- its descriptor "
        "mix is dominated by two full A-products (Ax and A'y at 1600 cycles "
        "each), not by the seven cheap reductions."
    ),
    "where_it_IS_thin": (
        "The queue-depth clause, not the average. The residual block contains a "
        "run of SIX CONSECUTIVE descriptors averaging 5.8 cycles (the n=80 "
        "reductions and the dual-residual assembly), which is below the 8-cycle "
        "bar. Six is inside the 8-deep queue, so run-ahead absorbs it, but this "
        "is the thinnest point in the whole sequencer analysis and it is inside "
        "the convergence test. Queue depth 8 is CONFIRMED as a floor by "
        "measurement rather than by assumption -- a 4-deep queue would not "
        "absorb this burst."
    ),
    "the_fix_is_a_MACIF_capability": (
        "Fold the reduction into the pass that produces the vector. MACIF "
        "already declares `square_accumulate` for the 2-norm; v0.1 adds "
        "`maxabs_accumulate` for the infinity norm. That folds |Ax|, |Px|, "
        "|A'y|, |rp| and |rd| into their producers, taking the residual block "
        "from 15 descriptors to 10 and the short-descriptor run from 6 to 4. "
        "Same principle as broadcast_init: fold work into the accumulator at "
        "O(1) area instead of buying array."
    ),
    "queue_sizing": "8 descriptors x 32 B = 256 B of queue SRAM. Unchanged: the "
                    "descriptor stayed 8 words.",
}
