"""Descriptor format v0: encoding, field widths, and the frozen/proposed split."""

from __future__ import annotations

import pytest

from argmind import (
    DESC_BYTES,
    DESC_WORDS,
    FROZEN_V0_OPCODES,
    PROPOSED_OPCODES,
    V0_LIMITATIONS,
    Descriptor,
    FieldOverflow,
    Format,
    Opcode,
    Structure,
    Transpose,
    UnsupportedOpcode,
)
from argmind.descriptor import DIM_MAX, PROPOSED_OPCODE_BASE


def test_descriptor_is_eight_words_thirty_two_bytes():
    assert DESC_WORDS == 8
    assert DESC_BYTES == 32
    d = Descriptor(Opcode.GEMV, Format.FP32, a=64, b=128, c=192, m=4, k=4, lda=4)
    assert len(d.encode()) == 8
    assert len(d.to_bytes()) == 32


def test_encode_decode_round_trip_preserves_every_field():
    d = Descriptor(
        opcode=Opcode.GEMM,
        precision=Format.FP16,
        a=0x1000, b=0x2000, c=0x3000,
        m=12, k=34, n=56, lda=78, ldb=90, ldc=123,
        alpha=0.5,                      # exactly representable in fp32
        transpose=Transpose.BOTH,
        structure=Structure.SYMMETRIC,
        accumulate=True,
        unit_diag=True,
        chain_token=200,
        wait_token=199,
    )
    back = Descriptor.decode(d.encode())
    for field in ("opcode", "precision", "a", "b", "c", "m", "k", "n",
                  "lda", "ldb", "ldc", "alpha", "transpose", "structure",
                  "accumulate", "unit_diag", "chain_token", "wait_token"):
        assert getattr(back, field) == getattr(d, field), field
    assert Descriptor.from_bytes(d.to_bytes()) == back


def test_w0_bit_positions_are_exactly_as_frozen():
    d = Descriptor(
        Opcode.CHOL, Format.FP64, transpose=Transpose.B, structure=Structure.TRIU,
        accumulate=True, unit_diag=True, chain_token=0xAB, wait_token=0xCD,
        m=1, lda=1,
    )
    w0 = d.encode()[0]
    assert (w0 >> 26) & 0x3F == int(Opcode.CHOL)
    assert (w0 >> 22) & 0x0F == int(Format.FP64)
    assert (w0 >> 20) & 0x03 == int(Transpose.B)
    assert (w0 >> 18) & 0x03 == int(Structure.TRIU)
    assert (w0 >> 17) & 0x01 == 1
    assert (w0 >> 16) & 0x01 == 1
    assert (w0 >> 8) & 0xFF == 0xAB
    assert w0 & 0xFF == 0xCD


def test_w0_has_no_spare_bits():
    """The reason limitation L4 is not a free change."""
    from argmind.descriptor import _W0_FIELDS

    assert sum(width for _, _, width in _W0_FIELDS) == 32


def test_dimension_fields_are_sixteen_bits_and_refuse_overflow():
    Descriptor(Opcode.AXPY, Format.FP32, b=0, c=8, m=DIM_MAX)
    with pytest.raises(FieldOverflow, match="16 bits"):
        Descriptor(Opcode.AXPY, Format.FP32, b=0, c=8, m=DIM_MAX + 1)
    with pytest.raises(FieldOverflow):
        Descriptor(Opcode.GEMM, Format.FP32, m=1, k=1, n=1, lda=DIM_MAX + 1)


def test_flat_matrix_axpy_caps_matrix_side_at_255():
    """Limitation L6, as an executable statement rather than prose."""
    n_ok = 255
    Descriptor(Opcode.AXPY, Format.FP32, b=0, c=8, m=n_ok * n_ok)
    with pytest.raises(FieldOverflow):
        Descriptor(Opcode.AXPY, Format.FP32, b=0, c=8, m=256 * 256)


def test_alpha_is_an_fp32_field():
    """Limitation L5: w7 cannot carry an fp64 scalar."""
    d = Descriptor(Opcode.SCAL, Format.FP64, b=0, c=8, m=1, alpha=0.1)
    assert d.alpha_f32 != 0.1
    assert abs(d.alpha_f32 - 0.1) < 1e-8
    exact = Descriptor(Opcode.SCAL, Format.FP64, b=0, c=8, m=1, alpha=0.125)
    assert exact.alpha_f32 == 0.125


def test_frozen_and_proposed_opcodes_are_disjoint_and_labelled():
    assert FROZEN_V0_OPCODES == {
        Opcode.GEMM, Opcode.GEMV, Opcode.TRSV, Opcode.AXPY,
        Opcode.PROJ, Opcode.DOT, Opcode.SCAL, Opcode.CHOL,
    }
    assert PROPOSED_OPCODES == {Opcode.NRMI, Opcode.VMUL}
    assert not (FROZEN_V0_OPCODES & PROPOSED_OPCODES)
    assert all(op < PROPOSED_OPCODE_BASE for op in FROZEN_V0_OPCODES)
    assert all(op >= PROPOSED_OPCODE_BASE for op in PROPOSED_OPCODES)


def test_proposed_opcode_is_refused_under_strict_v0():
    frozen = Descriptor(Opcode.DOT, Format.FP32, a=0, b=8, c=16, m=2)
    frozen.require_frozen_v0()          # no raise
    proposed = Descriptor(Opcode.NRMI, Format.FP32, a=0, b=8, m=2)
    assert not proposed.is_frozen_v0
    with pytest.raises(UnsupportedOpcode, match="PROPOSED"):
        proposed.require_frozen_v0()


def test_v0_limitations_are_documented_in_code():
    """The findings must live next to the format so they cannot drift from it."""
    ids = [entry["id"] for entry in V0_LIMITATIONS]
    assert ids == ["L1", "L2", "L3", "L4", "L5", "L6", "L7"]
    for entry in V0_LIMITATIONS:
        assert set(entry) == {"id", "severity", "what", "why", "fix"}
        assert all(entry[k].strip() for k in entry)
