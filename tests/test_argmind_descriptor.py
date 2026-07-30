"""Descriptor format v0.1: encoding, field widths, and every ruling as a test.

This file IS the executable half of the v0.1 specification. Where a ruling made a
quantitative claim -- "the change is paid for by reclaimed bits", "peak token
liveness is 7", "the n=255 cap is gone" -- the claim is asserted here, because a
spec whose numbers are only in prose drifts from the code within one revision.

It also keeps asserting the **v0** behaviour that motivated each change (via the
strict-v0 gate and :func:`argmind.auto_chain_v0`). A model that can no longer
reproduce the failure it was corrected for is not evidence.
"""

from __future__ import annotations

import pytest

from argmind import (
    DESC_BYTES,
    DESC_WORDS,
    DESCRIPTOR_VERSION,
    FROZEN_V0_OPCODES,
    N_TOKENS,
    SCRATCHPAD_RULING,
    SEQUENCER_RECHECK,
    V0_1_ADDED_OPCODES,
    V0_1_BIT_BUDGET,
    V0_1_OPCODES,
    V0_1_RULINGS,
    V0_LIMITATIONS,
    Descriptor,
    FieldOverflow,
    Format,
    Opcode,
    Structure,
    Transpose,
    UnsupportedOpcode,
    VecFormat,
)
from argmind.descriptor import DIM_MAX, WAIT_MASK_MAX, _W0_FIELDS


# ---------------------------------------------------------------------------
# Size and encoding
# ---------------------------------------------------------------------------

def test_descriptor_is_still_eight_words_thirty_two_bytes():
    """The affordability argument for v0.1: no extra word, so no extra traffic."""
    assert DESC_WORDS == 8
    assert DESC_BYTES == 32
    d = Descriptor(Opcode.GEMV, Format.FP32, a=64, b=128, c=192, m=4, k=4, lda=4)
    assert len(d.encode()) == 8
    assert len(d.to_bytes()) == 32


def test_encode_decode_round_trip_preserves_every_field():
    d = Descriptor(
        opcode=Opcode.GEMM,
        precision=Format.FP16,
        vec_precision=VecFormat.FP32,
        a=0x1000, b=0x2000, c=0x3000,
        m=12, k=34, n=56, lda=78, ldb=90, ldc=123,
        alpha=0.5,                      # exactly representable in fp32
        transpose=Transpose.BOTH,
        structure=Structure.SYM_UPPER,
        accumulate=True,
        unit_diag=True,
        chain_token=5,
        wait_mask=0b1010_1010,
    )
    back = Descriptor.decode(d.encode())
    for field in ("opcode", "precision", "vec_precision", "a", "b", "c",
                  "m", "k", "n", "lda", "ldb", "ldc", "alpha", "transpose",
                  "structure", "accumulate", "unit_diag", "chain_token",
                  "wait_mask"):
        assert getattr(back, field) == getattr(d, field), field
    assert Descriptor.from_bytes(d.to_bytes()) == back


def test_w0_bit_positions_are_exactly_as_specified():
    d = Descriptor(
        Opcode.CHOL, Format.FP64, vec_precision=VecFormat.FP32,
        transpose=Transpose.B, structure=Structure.TRIU,
        accumulate=True, unit_diag=True, chain_token=0b101, wait_mask=0b1100_0011,
        m=1, lda=1,
    )
    w0 = d.encode()[0]
    assert (w0 >> 26) & 0x3F == int(Opcode.CHOL)
    assert (w0 >> 23) & 0x07 == int(Format.FP64)
    assert (w0 >> 20) & 0x07 == int(VecFormat.FP32)
    assert (w0 >> 18) & 0x03 == int(Transpose.B)
    assert (w0 >> 15) & 0x07 == int(Structure.TRIU)
    assert (w0 >> 14) & 0x01 == 1
    assert (w0 >> 13) & 0x01 == 1
    assert (w0 >> 10) & 0x07 == 0b101
    assert (w0 >> 2) & 0xFF == 0b1100_0011
    assert w0 & 0x03 == 0                       # reserved


def test_reserved_bits_are_refused_rather_than_ignored():
    """Forward compatibility only works if a future field cannot be silently lost."""
    d = Descriptor(Opcode.DOT, Format.FP32, a=0, b=8, c=16, m=2)
    words = list(d.encode())
    words[0] |= 0b01
    with pytest.raises(FieldOverflow, match="reserved"):
        Descriptor.decode(words)


# ---------------------------------------------------------------------------
# The bit budget -- the fact the whole change rests on
# ---------------------------------------------------------------------------

def test_the_format_change_is_paid_for_by_reclaimed_waste():
    """v0 had zero spare bits; v0.1 adds two fields and still has two spare.

    This is the load-bearing arithmetic of the entire ruling. If it stops being
    true, prec_v or the wait mask has to come back out.
    """
    assert sum(w for _, _, w in _W0_FIELDS) == 32
    assert V0_1_BIT_BUDGET["used"] == 30
    assert V0_1_BIT_BUDGET["reserved"] == 2
    assert V0_1_BIT_BUDGET["v0_spare"] == 0


def test_where_the_reclaimed_bits_came_from():
    widths = {name: w for name, _, w in _W0_FIELDS}
    # -5 from the chain token: 8 bits named 255 tokens, and a token is a
    # scoreboard slot, not a name.
    assert widths["chain_token"] == 3
    # -1 from the operand precision field: five formats never needed four bits.
    assert widths["prec_m"] == 3
    assert max(int(f) for f in Format) < 2 ** widths["prec_m"]
    # ...which paid for +3 prec_v and +1 structure.
    assert widths["prec_v"] == 3
    assert max(int(f) for f in VecFormat) < 2 ** widths["prec_v"]
    assert widths["structure"] == 3
    assert max(int(s) for s in Structure) < 2 ** widths["structure"]
    # The wait field is the same WIDTH as v0's, with different meaning.
    assert widths["wait_mask"] == 8


# ---------------------------------------------------------------------------
# R1 -- NRMI
# ---------------------------------------------------------------------------

def test_nrmi_and_vmul_are_v0_1_opcodes_and_cost_nothing_scarce():
    assert V0_1_ADDED_OPCODES == {Opcode.NRMI, Opcode.VMUL}
    assert not (FROZEN_V0_OPCODES & V0_1_ADDED_OPCODES)
    opcode_bits = next(w for name, _, w in _W0_FIELDS if name == "opcode")
    # 10 opcodes defined out of 64: opcode space was never the scarce resource,
    # which is why R1 and R7 did not have to compete with R3 and R4 for bits.
    assert len(V0_1_OPCODES) == 10
    assert 2 ** opcode_bits - len(V0_1_OPCODES) == 54


def test_a_strict_v0_device_still_refuses_the_additions():
    """The refusal must stay reproducible or the cost stops being measurable."""
    frozen = Descriptor(Opcode.DOT, Format.FP32, a=0, b=8, c=16, m=2)
    frozen.require_frozen_v0()          # no raise
    added = Descriptor(Opcode.NRMI, Format.FP32, a=0, b=8, m=2)
    assert not added.is_frozen_v0
    with pytest.raises(UnsupportedOpcode, match="NRMI"):
        added.require_frozen_v0()


# ---------------------------------------------------------------------------
# R3 -- tokens and the wait mask
# ---------------------------------------------------------------------------

def test_wait_mask_expresses_a_merge_wider_than_two():
    """The measured merge width on the ADMM iteration is 5, so two wait tokens
    would not have been enough -- a mask was the only encoding that works."""
    d = Descriptor(Opcode.GEMV, Format.FP32, a=0, b=8, c=16, m=2, k=2, lda=2,
                   chain_token=1, wait_mask=0b0011_1110)
    assert d.waits_on == (1, 2, 3, 4, 5)
    assert d.merge_width == 5


def test_token_space_is_eight_and_the_field_refuses_more():
    assert N_TOKENS == 8
    assert WAIT_MASK_MAX == 255
    Descriptor(Opcode.SCAL, Format.FP32, b=0, c=8, m=1, chain_token=N_TOKENS - 1)
    with pytest.raises(FieldOverflow, match="chain_token"):
        Descriptor(Opcode.SCAL, Format.FP32, b=0, c=8, m=1, chain_token=N_TOKENS)


# ---------------------------------------------------------------------------
# R4 -- mixed precision
# ---------------------------------------------------------------------------

def test_the_recommended_mixed_configuration_is_now_expressible():
    """L4 closed: fp16 matrix operand, fp32 store, in one descriptor."""
    d = Descriptor(Opcode.GEMV, Format.FP16, vec_precision=VecFormat.FP32,
                   a=0, b=8, c=16, m=4, k=4, lda=4)
    assert d.prec_m is Format.FP16
    assert d.prec_v is Format.FP32
    assert d.is_mixed_precision


def test_vec_precision_same_means_what_every_v0_descriptor_meant():
    d = Descriptor(Opcode.GEMV, Format.FP64, a=0, b=8, c=16, m=4, k=4, lda=4)
    assert d.vec_precision is VecFormat.SAME
    assert d.prec_v is Format.FP64
    assert not d.is_mixed_precision


def test_fp16_and_bf16_stay_distinct_in_both_precision_fields():
    """The 10x control-action difference is a property of the format, not the width."""
    assert VecFormat.of(Format.FP16) is not VecFormat.of(Format.BF16)
    assert VecFormat.FP16.resolve(Format.BF16) is Format.FP16


# ---------------------------------------------------------------------------
# R6a -- 2-D elementwise, and the death of the n=255 cap
# ---------------------------------------------------------------------------

def test_matrix_elementwise_lifts_the_255_cap():
    """L6 closed. v0 forced a flat n*n AXPY, which overflowed 16 bits at n=256."""
    with pytest.raises(FieldOverflow):
        Descriptor(Opcode.AXPY, Format.FP32, b=0, c=8, m=256 * 256)  # v0 form
    # v0.1 form: 2-D, so each dimension is separately 16 bits.
    d = Descriptor(Opcode.AXPY, Format.FP32, b=0, c=8, m=300, n=300,
                   ldb=300, ldc=300)
    assert d.elements == 90_000
    big = Descriptor(Opcode.AXPY, Format.FP32, b=0, c=8, m=DIM_MAX, n=DIM_MAX,
                     ldb=DIM_MAX, ldc=DIM_MAX)
    assert big.elements == DIM_MAX * DIM_MAX


def test_n_zero_reads_as_n_one_so_v0_vector_descriptors_are_unchanged():
    d = Descriptor(Opcode.AXPY, Format.FP32, b=0, c=8, m=17)
    assert d.n == 0 and d.cols == 1 and d.elements == 17


def test_dimension_fields_are_sixteen_bits_and_refuse_overflow():
    Descriptor(Opcode.AXPY, Format.FP32, b=0, c=8, m=DIM_MAX)
    with pytest.raises(FieldOverflow, match="16 bits"):
        Descriptor(Opcode.AXPY, Format.FP32, b=0, c=8, m=DIM_MAX + 1)
    with pytest.raises(FieldOverflow):
        Descriptor(Opcode.GEMM, Format.FP32, m=1, k=1, n=1, lda=DIM_MAX + 1)


# ---------------------------------------------------------------------------
# R6b -- zero AGU stride
# ---------------------------------------------------------------------------

def test_zero_stride_is_legal_on_reads():
    """A broadcast: one element read m times. The diagonal-add case."""
    d = Descriptor(Opcode.AXPY, Format.FP64, b=0, c=64, m=80, ldb=0, ldc=81)
    assert d.ldb == 0
    Descriptor(Opcode.VMUL, Format.FP64, a=0, b=8, c=64, m=32, lda=0)
    Descriptor(Opcode.GEMV, Format.FP32, a=0, b=8, c=16, m=4, k=4, lda=4, ldb=0)


@pytest.mark.parametrize("op", [Opcode.AXPY, Opcode.SCAL, Opcode.GEMV,
                                Opcode.TRSV, Opcode.PROJ, Opcode.VMUL,
                                Opcode.GEMM])
def test_zero_stride_is_illegal_on_writes(op):
    """m writers, one address, no defined order."""
    with pytest.raises(FieldOverflow, match="ldc = 0"):
        Descriptor(op, Format.FP32, a=0, b=8, c=16, m=4, k=4, n=4, lda=4, ldc=0)


def test_zero_stride_write_is_still_illegal_for_chol_via_lda():
    with pytest.raises(FieldOverflow, match="lda = 0"):
        Descriptor(Opcode.CHOL, Format.FP32, a=0, m=4, lda=0)


def test_reductions_are_the_defined_many_to_one_write():
    """DOT and NRMI are exempt because their reduction semantics ARE defined."""
    Descriptor(Opcode.DOT, Format.FP32, a=0, b=8, c=16, m=64, ldc=0)
    Descriptor(Opcode.NRMI, Format.FP32, a=0, b=8, m=64, ldc=0)


def test_lda_defaults_to_one_so_a_forgotten_stride_is_not_a_silent_broadcast():
    d = Descriptor(Opcode.VMUL, Format.FP32, a=0, b=8, c=16, m=4)
    assert d.lda == 1


# ---------------------------------------------------------------------------
# R5 -- w7 stays fp32, and why that is now acceptable
# ---------------------------------------------------------------------------

def test_alpha_is_still_an_fp32_field_and_that_is_the_ruling_not_an_oversight():
    d = Descriptor(Opcode.SCAL, Format.FP64, b=0, c=8, m=1, alpha=0.1)
    assert d.alpha_f32 != 0.1
    assert abs(d.alpha_f32 - 0.1) < 1e-8
    exact = Descriptor(Opcode.SCAL, Format.FP64, b=0, c=8, m=1, alpha=0.125)
    assert exact.alpha_f32 == 0.125


def test_the_reciprocal_mismatch_that_w7_causes_and_scratchpad_removes():
    """The quantitative basis of R5, at four significant figures."""
    import numpy as np

    rho = 0.1
    w7_mismatch = abs(float(np.float32(rho)) * float(np.float32(1.0 / rho)) - 1.0)
    r32 = float(np.float32(rho))
    spad_mismatch = abs(r32 * (1.0 / r32) - 1.0)
    assert w7_mismatch > 1e-8                 # measured 1.49e-08: stalls the solve
    assert spad_mismatch < 1e-15              # fp64 storage: below any tolerance
    assert w7_mismatch / max(spad_mismatch, 5e-17) > 1e6


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------

def test_symmetric_upper_exists_and_symmetric_is_still_spellable():
    assert Structure.SYMMETRIC is Structure.SYM_LOWER
    assert Structure.SYM_UPPER not in (Structure.SYM_LOWER, Structure.TRIU)
    assert Structure.SYM_LOWER.is_symmetric and Structure.SYM_UPPER.is_symmetric
    assert Structure.SYM_LOWER.reads_lower
    assert not Structure.SYM_UPPER.reads_lower
    assert Structure.TRIL.reads_lower and not Structure.TRIU.reads_lower


# ---------------------------------------------------------------------------
# The rulings themselves must stay in code, next to the format
# ---------------------------------------------------------------------------

def test_every_v0_limitation_has_a_ruling():
    ids = [e["id"] for e in V0_LIMITATIONS]
    assert ids == ["L1", "L2", "L3", "L4", "L5", "L6", "L7"]
    for e in V0_LIMITATIONS:
        assert {"id", "severity", "what", "why", "fix", "ruling"} <= set(e)
        assert all(str(v).strip() for v in e.values())
    closed = {e.get("closes") for e in V0_1_RULINGS}
    for lid in ids:
        assert lid in closed, f"{lid} has no ruling"


def test_rulings_are_labelled_and_complete():
    ids = [r["id"] for r in V0_1_RULINGS]
    assert ids == ["R1", "R2", "R3", "R4", "R5", "R6a", "R6b", "R7", "R8"]
    for r in V0_1_RULINGS:
        assert r["decision"].strip()
        assert len(r["decision"]) > 20


def test_the_two_corrections_to_the_build_report_are_recorded():
    """Two of the seven findings were argued on grounds that do not hold. Both
    corrections are part of the ruling and must not be quietly dropped."""
    r1 = next(r for r in V0_1_RULINGS if r["id"] == "R1")
    assert "NOT TRUE" in r1["correction"]
    assert "leaves the fabric" in r1["correction"]
    r7 = next(r for r in V0_1_RULINGS if r["id"] == "R7")
    assert "WRONG" in r7["correction"]
    assert "interior-point" in r7["correction"]


def test_scratchpad_ruling_carries_its_measurement_and_its_rejects():
    assert SCRATCHPAD_RULING["measured"]["QUAD-MPC fp32"] == 167736
    assert SCRATCHPAD_RULING["model_said"] == 102400
    # The decomposition must be exact -- that is what makes it a diagnosis rather
    # than an estimate.
    assert (102400 + 25600 + 12800 + 12800 + 14136
            == SCRATCHPAD_RULING["measured"]["QUAD-MPC fp32"])
    assert SCRATCHPAD_RULING["sizing_basis_fp32"] == 167736 - 25600
    # The format change is not a capacity remedy and must not be sold as one.
    v0 = SCRATCHPAD_RULING["measured"]["QUAD-MPC fp32"]
    v01 = SCRATCHPAD_RULING["measured_v0_1"]["QUAD-MPC fp32"]
    assert 0 < v0 - v01 < 0.005 * v0
    assert "NOT QUOTABLE" in SCRATCHPAD_RULING["label"]
    # Every rejected option must say why, so none of them gets re-proposed as new.
    assert len(SCRATCHPAD_RULING["rejected"]) >= 5
    for option, reason in SCRATCHPAD_RULING["rejected"].items():
        assert len(reason) > 40, option


def test_sequencer_recheck_reports_a_negative_result():
    """The descriptor-count correction did NOT move the contract, and the model
    has to be able to say so."""
    assert "NOT THREATENED" in SEQUENCER_RECHECK["verdict_on_the_8_cycle_bar"]
    assert "29.8" in SEQUENCER_RECHECK["verdict_on_the_8_cycle_bar"]
    assert "SIX CONSECUTIVE" in SEQUENCER_RECHECK["where_it_IS_thin"]


def test_descriptor_version_is_declared():
    assert DESCRIPTOR_VERSION == "0.1"
