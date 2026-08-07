# SPDX-License-Identifier: Apache-2.0
"""Numeric formats, and their encoding in the descriptor precision fields.

The integer values here ARE the wire encoding of descriptor v0.1 field
``prec_m`` (``w0[25:23]``). They are part of the published interface: changing one
changes the meaning of every previously generated descriptor, so treat them as a
published constant. **The values did not change between v0 and v0.1** -- only the
field narrowed from 4 bits to 3, which every defined value already fits.

bfloat16 has no numpy dtype. It is therefore represented in the simulated
scratchpad as raw 16-bit containers (`numpy.uint16`), with explicit
encode/decode helpers, and a buffer of format ``BF16`` exposes a ``uint16``
view. This mirrors the C API, where bf16 is likewise a 16-bit storage type with
no native language support -- the simulator does not get to be more convenient
than the hardware.

fp16 and bf16 are **distinct formats and never interchangeable**. Same width,
same MAC cost, 10x different error in a commanded control action on the solver
path (PHASE0_FINDINGS 0.3, measured: 1.9e-03 for fp16 operands vs 1.9e-02 for
bf16 operands, both with an fp32 solve). Substitution is refused by
:mod:`argmind.macif`.
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np


class Format(IntEnum):
    """Numeric format. The value is the descriptor ``prec_m`` wire encoding.

    Five values, so the field is **3 bits** in v0.1 (it was 4 in v0, which was
    one bit of waste; that bit paid for part of the second precision field).
    Values 5-7 are unassigned and reserved for the two fp8 variants.
    """

    INT8 = 0
    FP16 = 1
    BF16 = 2
    FP32 = 3
    FP64 = 4

    @property
    def itemsize(self) -> int:
        """Bytes per element in scratchpad."""
        return _ITEMSIZE[self]

    @property
    def storage_dtype(self) -> np.dtype:
        """The dtype a buffer of this format exposes to the host.

        For every format except :attr:`BF16` this is the arithmetic dtype too.
        BF16 exposes ``uint16`` raw containers; use :func:`bf16_decode` /
        :func:`bf16_encode` to convert.
        """
        return _STORAGE[self]

    @property
    def is_float(self) -> bool:
        return self is not Format.INT8

    def __str__(self) -> str:  # nicer error messages
        return self.name.lower()


class VecFormat(IntEnum):
    """Descriptor v0.1 ``prec_v`` field: the format of vectors and results.

    Added in v0.1 to fix limitation L4. ``prec_m`` gives the format of the
    **bulk operand** -- the one whose size scales with the largest product of
    dimensions, i.e. the matrix -- and ``prec_v`` gives the format of everything
    else the descriptor touches. :attr:`SAME` means "as ``prec_m``", which is
    what every v0 descriptor meant implicitly.

    Why a second field rather than the cheap answer
    -----------------------------------------------
    The cheap answer was to let the MACIF operand contract imply the accumulator
    width, and for the *accumulator* that answer is correct and is still what
    happens -- accumulator width is a MACIF declaration, not a descriptor field.
    What it cannot fix is the **store** format, and the store format is the one
    the numerics need. PHASE0_FINDINGS 0.3 adopts fp16 operands with the
    triangular solve at fp32; the operative constraint there is not that TRSV
    carries a different opcode-level precision (v0 could already express that)
    but that a **buffer written by an fp16 GEMV is read by an fp32 TRSV**. One
    precision field forces the GEMV to store fp16 and the TRSV then reads a
    different type at the same address. So the mixed scheme was inexpressible for
    a storage reason, not an arithmetic one.

    The honest limit of this encoding: exactly two formats per descriptor. A
    kernel wanting three (int8 matrix, bf16 vector, fp32 result) must split into
    two descriptors. Nothing in the frozen benchmark set wants three.
    """

    SAME = 0
    INT8 = 1
    FP16 = 2
    BF16 = 3
    FP32 = 4
    FP64 = 5

    def resolve(self, prec_m: "Format") -> "Format":
        """The concrete format this field denotes, given the bulk-operand format."""
        if self is VecFormat.SAME:
            return Format(prec_m)
        return _VEC_TO_FORMAT[self]

    @classmethod
    def of(cls, fmt: "Format") -> "VecFormat":
        """The explicit :class:`VecFormat` naming `fmt` (never :attr:`SAME`)."""
        return _FORMAT_TO_VEC[Format(fmt)]

    def __str__(self) -> str:
        return "same" if self is VecFormat.SAME else self.name.lower()


_VEC_TO_FORMAT = {
    VecFormat.INT8: Format.INT8,
    VecFormat.FP16: Format.FP16,
    VecFormat.BF16: Format.BF16,
    VecFormat.FP32: Format.FP32,
    VecFormat.FP64: Format.FP64,
}

_FORMAT_TO_VEC = {v: k for k, v in _VEC_TO_FORMAT.items()}


_ITEMSIZE = {
    Format.INT8: 1,
    Format.FP16: 2,
    Format.BF16: 2,
    Format.FP32: 4,
    Format.FP64: 8,
}

_STORAGE = {
    Format.INT8: np.dtype(np.int8),
    Format.FP16: np.dtype(np.float16),
    Format.BF16: np.dtype(np.uint16),   # raw container: see module docstring
    Format.FP32: np.dtype(np.float32),
    Format.FP64: np.dtype(np.float64),
}

#: Formats that occupy 16 bits. Listed so that a "same width, therefore
#: interchangeable" assumption can be tested for and refused explicitly.
SIXTEEN_BIT_FORMATS = (Format.FP16, Format.BF16)


def bf16_encode(values: np.ndarray) -> np.ndarray:
    """fp32/fp64 -> raw bf16 bit containers, round-to-nearest-even."""
    f32 = np.ascontiguousarray(values, dtype=np.float32)
    bits = f32.view(np.uint32)
    # round-to-nearest-even on the truncated 16 low mantissa bits
    lsb = (bits >> np.uint32(16)) & np.uint32(1)
    rounded = bits + np.uint32(0x7FFF) + lsb
    return (rounded >> np.uint32(16)).astype(np.uint16)


def bf16_decode(containers: np.ndarray) -> np.ndarray:
    """Raw bf16 bit containers -> fp32."""
    bits = np.asarray(containers, dtype=np.uint16).astype(np.uint32) << np.uint32(16)
    return bits.view(np.float32)


def to_arithmetic(values: np.ndarray, fmt: Format) -> np.ndarray:
    """Read a scratchpad view as numbers, whatever the storage container is."""
    if fmt is Format.BF16:
        return bf16_decode(values)
    return values


def from_arithmetic(values: np.ndarray, fmt: Format) -> np.ndarray:
    """Convert numbers to the storage container for `fmt`, rounding as hardware would."""
    if fmt is Format.BF16:
        return bf16_encode(values)
    if fmt is Format.INT8:
        return np.clip(np.rint(values), -128, 127).astype(np.int8)
    return np.asarray(values, dtype=fmt.storage_dtype)
