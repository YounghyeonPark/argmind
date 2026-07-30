"""Descriptor format v0 -- the 8-word command word the sequencer hands the datapath.

This module implements the layout **exactly as frozen**, including the field
widths, so that anything the format cannot express fails here rather than being
quietly worked around::

     w0  [31:26] opcode      GEMM, GEMV, TRSV, AXPY, PROJ, DOT, SCAL, CHOL
         [25:22] precision   int8/fp16/bf16/fp32/fp64
         [21:20] transpose   none / A / B / both
         [19:18] structure   dense / triangular-lower / triangular-upper / symmetric
         [17]    accumulate  C += (vs C =)
         [16]    unit_diag
         [15:8]  chain_token dependency token this descriptor produces
         [7:0]   wait_token  token it waits on (0 = no wait)
     w1  A base address (scratchpad byte offset)
     w2  B / x base address
     w3  C / y base address
     w4  [31:16] M           [15:0] K
     w5  [31:16] N           [15:0] lda
     w6  [31:16] ldb         [15:0] ldc
     w7  alpha / rho scalar (fp32 bit pattern), or immediate for PROJ bounds index

32 bytes: one two-beat burst on a 128-bit bus, a power of two so queue indexing
is a shift, and three independent base addresses.

Operand-address conventions per opcode
--------------------------------------
v0 gives three base addresses and no more, so each opcode has to fit its operands
into ``a`` / ``b`` / ``c``. The assignments below are part of the interface:

===== ============================ ============== ==============
op    a                            b              c
===== ============================ ============== ==============
GEMM  matrix A                     matrix B       matrix C (out)
GEMV  matrix A                     vector x       vector y (out)
TRSV  triangular matrix A          rhs b          solution x (out; may alias b)
AXPY  (unused)                     vector x       vector y (in/out)
SCAL  (unused)                     vector x       vector y (out; may alias x)
DOT   scalar result (out)          vector x       vector y
PROJ  lower-bound vector           upper-bound v  vector clipped **in place**
CHOL  matrix (in/out, factor in place)  (unused)  (unused)
===== ============================ ============== ==============

Note ``PROJ``: with only three addresses, a box projection whose input and output
differ cannot be expressed, because it needs four (input, lower, upper, output).
v0 therefore makes projection intrinsically in-place. That happens to agree with
the in-place-kernel requirement, but it is a constraint, not a choice --
see :data:`V0_LIMITATIONS`.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Iterable

from .errors import FieldOverflow, UnsupportedOpcode
from .precision import Format

DESC_WORDS = 8
DESC_BYTES = DESC_WORDS * 4


class Opcode(IntEnum):
    """Descriptor opcodes.

    Values 0-7 are **frozen format v0**. Values from
    :data:`PROPOSED_OPCODE_BASE` upward are *proposals discovered by implementing
    v0*; they are not part of the frozen interface, a strict-v0 device refuses
    them, and each one needs an architectural ruling before it may be relied on.
    """

    # ---- frozen v0 ----
    GEMM = 0
    GEMV = 1
    TRSV = 2
    AXPY = 3
    PROJ = 4
    DOT = 5
    SCAL = 6
    CHOL = 7

    # ---- proposed, NOT in v0 ----
    #: Max-absolute (infinity-norm) reduction. Needed because OSQP's termination
    #: criterion is stated in infinity norms, and v0 offers only DOT (a 2-norm).
    #: Without it the convergence test cannot be evaluated in fabric with the
    #: reference solver's own criterion.
    NRMI = 32
    #: Elementwise vector multiply, c = a * b. Needed for a per-constraint rho
    #: vector, which OSQP uses whenever the problem contains equality rows
    #: (rho_i scaled by 1e3 there). v0's single fp32 scalar can only express a
    #: uniform rho.
    VMUL = 33


#: First opcode value outside frozen v0.
PROPOSED_OPCODE_BASE = 32

FROZEN_V0_OPCODES: frozenset[Opcode] = frozenset(
    op for op in Opcode if op < PROPOSED_OPCODE_BASE
)

PROPOSED_OPCODES: frozenset[Opcode] = frozenset(
    op for op in Opcode if op >= PROPOSED_OPCODE_BASE
)


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
    DENSE = 0
    TRIL = 1          # triangular, lower
    TRIU = 2          # triangular, upper
    SYMMETRIC = 3


# --------------------------------------------------------------------------
# Bit layout
# --------------------------------------------------------------------------

_W0_FIELDS = (
    # (name, shift, width)
    ("opcode", 26, 6),
    ("precision", 22, 4),
    ("transpose", 20, 2),
    ("structure", 18, 2),
    ("accumulate", 17, 1),
    ("unit_diag", 16, 1),
    ("chain_token", 8, 8),
    ("wait_token", 0, 8),
)

_MASK32 = 0xFFFFFFFF
_MASK16 = 0xFFFF

#: Widths of the packed dimension/stride fields, in bits. Every dimension and
#: leading dimension in v0 is 16 bits, which caps any single descriptor at
#: 65535 in each of M, K, N, lda, ldb, ldc.
DIM_BITS = 16
DIM_MAX = (1 << DIM_BITS) - 1


def _check(name: str, value: int, width: int) -> int:
    if not isinstance(value, (int,)) or isinstance(value, bool):
        raise FieldOverflow(f"{name} must be a plain int, got {value!r}")
    if value < 0 or value > (1 << width) - 1:
        raise FieldOverflow(
            f"descriptor v0 field {name!r} is {width} bits "
            f"(max {(1 << width) - 1}); got {value}"
        )
    return value


def _f32_bits(x: float) -> int:
    return int(struct.unpack("<I", struct.pack("<f", float(x)))[0])


def _bits_f32(bits: int) -> float:
    return float(struct.unpack("<f", struct.pack("<I", bits & _MASK32))[0])


@dataclass(frozen=True)
class Descriptor:
    """One descriptor in format v0.

    All addresses are scratchpad byte offsets. All dimensions are element counts;
    leading dimensions are in elements, not bytes, as is conventional for BLAS.

    ``alpha`` round-trips through an **fp32** bit pattern because that is what w7
    is: a descriptor cannot carry an fp64 scalar. For ADMM that is harmless
    (rho, sigma and alpha are algorithm parameters and do not change the
    optimum, only the path to it), but it is a real constraint on any fp64 kernel
    whose scalar is *data* rather than a tuning parameter.
    """

    opcode: Opcode
    precision: Format = Format.FP64
    a: int = 0
    b: int = 0
    c: int = 0
    m: int = 0
    k: int = 0
    n: int = 0
    lda: int = 0
    ldb: int = 1
    ldc: int = 1
    alpha: float = 1.0
    transpose: Transpose = Transpose.NONE
    structure: Structure = Structure.DENSE
    accumulate: bool = False
    unit_diag: bool = False
    chain_token: int = 0
    wait_token: int = 0
    #: Not encoded. Purely a label carried alongside for readable traces and
    #: error messages; the hardware descriptor has no room for it.
    label: str = ""

    def __post_init__(self) -> None:
        # Validate everything that the bit layout constrains, at construction
        # time, so a code generator cannot emit an unencodable descriptor.
        self.encode()

    # -- encoding ---------------------------------------------------------

    def encode(self) -> tuple[int, ...]:
        """Encode to 8 unsigned 32-bit words."""
        w0 = 0
        values = {
            "opcode": int(self.opcode),
            "precision": int(self.precision),
            "transpose": int(self.transpose),
            "structure": int(self.structure),
            "accumulate": int(bool(self.accumulate)),
            "unit_diag": int(bool(self.unit_diag)),
            "chain_token": int(self.chain_token),
            "wait_token": int(self.wait_token),
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
        return cls(
            opcode=Opcode(get["opcode"]),
            precision=Format(get["precision"]),
            transpose=Transpose(get["transpose"]),
            structure=Structure(get["structure"]),
            accumulate=bool(get["accumulate"]),
            unit_diag=bool(get["unit_diag"]),
            chain_token=get["chain_token"],
            wait_token=get["wait_token"],
            a=w[1], b=w[2], c=w[3],
            m=w[4] >> 16, k=w[4] & _MASK16,
            n=w[5] >> 16, lda=w[5] & _MASK16,
            ldb=w[6] >> 16, ldc=w[6] & _MASK16,
            alpha=_bits_f32(w[7]),
        )

    @classmethod
    def from_bytes(cls, raw: bytes) -> "Descriptor":
        return cls.decode(struct.unpack("<8I", raw))

    # -- helpers ----------------------------------------------------------

    @property
    def is_frozen_v0(self) -> bool:
        return self.opcode in FROZEN_V0_OPCODES

    def require_frozen_v0(self) -> None:
        if not self.is_frozen_v0:
            raise UnsupportedOpcode(
                f"opcode {self.opcode.name} is a PROPOSED addition, not part of "
                f"descriptor format v0. Set strict_v0=False on the device to run "
                f"it anyway, and expect an architectural ruling to be required "
                f"before depending on it."
            )

    @property
    def alpha_f32(self) -> float:
        """The value of `alpha` as it survives the fp32 w7 field."""
        return _bits_f32(_f32_bits(self.alpha))

    def __str__(self) -> str:
        bits = [self.opcode.name, str(self.precision)]
        if self.transpose is not Transpose.NONE:
            bits.append(f"T={self.transpose.name}")
        if self.structure is not Structure.DENSE:
            bits.append(self.structure.name)
        if self.accumulate:
            bits.append("acc")
        dims = ",".join(
            f"{n}={getattr(self, n)}" for n in ("m", "k", "n") if getattr(self, n)
        )
        tok = f"->{self.chain_token}" if self.chain_token else ""
        wait = f" wait{self.wait_token}" if self.wait_token else ""
        name = f" {self.label}" if self.label else ""
        return f"{' '.join(bits)} [{dims}]{tok}{wait}{name}"


# ---------------------------------------------------------------------------
# What implementing v0 revealed. Kept in code, next to the format, so it cannot
# drift away from it.
# ---------------------------------------------------------------------------

V0_LIMITATIONS: tuple[dict[str, str], ...] = (
    {
        "id": "L1",
        "severity": "blocking for in-fabric convergence test",
        "what": "No max-absolute (infinity-norm) reduction opcode.",
        "why": "OSQP's termination criterion is stated entirely in infinity "
               "norms. v0 offers DOT, i.e. a 2-norm. A 2-norm test is a valid "
               "stopping rule but is not the reference solver's rule, so a "
               "customer comparing iteration counts against OSQP sees a "
               "different number. Worse, if the criterion cannot be evaluated "
               "in fabric the convergence loop returns to the host, which is "
               "the mechanism the architecture exists to avoid.",
        "fix": "One opcode (NRMI). No new fields, no new base addresses.",
    },
    {
        "id": "L2",
        "severity": "restricts kernels to in-place form",
        "what": "Only three base addresses, so out-of-place box projection "
                "(input, lower, upper, output) cannot be expressed.",
        "why": "PROJ must clip in place, with the value pre-staged in the "
               "destination by a preceding SCAL. Costs one extra descriptor per "
               "projection and forces a copy the residency model would rather "
               "not do.",
        "fix": "Either accept in-place-only projection as the contract (it "
               "agrees with the in-place kernel requirement) or add a fourth "
               "AGU/base address. The emit-cost analysis already flagged a "
               "fourth AGU as the open question.",
    },
    {
        "id": "L3",
        "severity": "removes achievable overlap",
        "what": "Exactly one wait token per descriptor.",
        "why": "A descriptor with two independent producers cannot express both "
               "dependencies, so the code generator must serialise the "
               "producers. Since 97-100% of the sequencer's latency "
               "contribution is dependency drain, artificially deepening the "
               "chain is not free.",
        "fix": "Either a second wait token (8 bits are not available in w0 -- "
               "it is exactly full) or a wait *mask* semantics change, or an "
               "explicit statement that codegen owns serialisation.",
    },
    {
        "id": "L4",
        "severity": "blocks the recommended numerics",
        "what": "One precision field, so operand and result formats are "
                "necessarily the same.",
        "why": "The configuration the precision study recommends -- 16-bit "
               "matrix operands with an fp32 solve and fp32 accumulate -- is a "
               "mixed-format kernel and therefore inexpressible. w0 has zero "
               "spare bits (6+4+2+2+1+1+8+8 = 32), so this is not a free "
               "change.",
        "fix": "Architect ruling: widen w0's precision field by taking bits "
               "from elsewhere, add a second header word, or declare that the "
               "accumulator format is implied by the MACIF contract for the "
               "operand format (which is nearly true already, and would be the "
               "cheap answer, but it fixes the *store* format too).",
    },
    {
        "id": "L5",
        "severity": "SILENT WRONG ANSWER unless codegen constrains rho",
        "what": "w7 is an fp32 bit pattern, so a kernel cannot receive an exact "
                "fp64 scalar -- and it cannot receive a matched pair of "
                "reciprocal scalars unless they happen to be exact in fp32.",
        "why": "This looked benign ('rho only changes the path, not the "
               "optimum') and it is not. ADMM multiplies by 1/rho when forming "
               "the projection argument and by rho in the dual update. At a "
               "fixed point with an active constraint, y = rho*(1/rho)*y, so if "
               "fl32(rho)*fl32(1/rho) != 1 the only solution is y = 0: the dual "
               "variable cannot settle and the solve stalls at a residual floor "
               "of roughly that mismatch times the dual magnitude. Measured on "
               "QUAD-MPC with a 10-constraint active set, requesting 1e-9: "
               "rho=0.1 and rho=0.2 (mismatch 1.5e-08) both stall; every power "
               "of two converges. Note the failure is invisible on a problem "
               "whose optimum is interior, because y* = 0 there anyway.",
        "fix": "Cheapest: put 'rho must be a power of two' in the codegen "
               "contract and have the runtime refuse otherwise -- which it now "
               "does. Alternatives: a reciprocal-capable scalar field, or a "
               "kernel that divides rather than multiplying by a stored "
               "reciprocal.",
    },
    {
        "id": "L6",
        "severity": "bounded, but reachable",
        "what": "16-bit dimension and leading-dimension fields.",
        "why": "Any single descriptor is capped at 65535 elements per dimension. "
               "Treating an n x n matrix as a flat vector for an elementwise op "
               "(which v0 forces, since there is no matrix-add opcode) caps n at "
               "255. QUAD-MPC at n=80 fits (6400); a 300-variable condensed QP "
               "would not.",
        "fix": "Codegen must tile flat elementwise ops. Cheap, but it must be "
               "in the codegen contract or it will be discovered by a customer.",
    },
    {
        "id": "L7",
        "severity": "affects problems with equality constraints",
        "what": "No elementwise vector multiply, and only a scalar rho field.",
        "why": "OSQP uses a per-constraint rho vector, with equality rows scaled "
               "by 1e3. Any problem with equality constraints therefore cannot "
               "be expressed. QUAD-MPC condensed to the control sequence has "
               "only box constraints, so it is unaffected -- but the *sparse* "
               "(non-condensed) MPC form, which is what acados and HPIPM use, "
               "is all equality constraints.",
        "fix": "One opcode (VMUL), three addresses, no new fields.",
    },
)
