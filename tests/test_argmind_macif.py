"""The MACIF precision contract: it must refuse, never substitute."""

from __future__ import annotations

import numpy as np
import pytest

from argmind import (
    Descriptor,
    Device,
    DeviceConfig,
    Format,
    FormatContract,
    Lifetime,
    MacroCapabilities,
    Opcode,
    PrecisionNotDeclared,
    Program,
    macif,
)


def _tiny_device(macro: MacroCapabilities) -> Device:
    return Device(DeviceConfig(scratchpad_bytes=8192, banks=8, macro=macro,
                               strict_v0=False, host_bytes=1 << 16))


def _one_descriptor_program(dev: Device, fmt: Format) -> Program:
    x = dev.alloc("x", 4, fmt, Lifetime.STREAMING)
    y = dev.alloc("y", 4, fmt, Lifetime.STREAMING)
    return Program("p", [Descriptor(Opcode.AXPY, fmt, b=x.spad_offset,
                                    c=y.spad_offset, m=4, alpha=1.0)])


# ---------------------------------------------------------------------------
# fp64
# ---------------------------------------------------------------------------

def test_fp64_kernel_is_refused_on_a_macro_without_fp64_at_build_time():
    dev = _tiny_device(macif.CIM_INFERENCE_MACRO)
    prog = _one_descriptor_program(dev, Format.FP64)
    with pytest.raises(PrecisionNotDeclared) as exc:
        dev.build(prog)
    msg = str(exc.value)
    assert "does not declare fp64" in msg
    assert "algorithm requirement" in msg          # explains *why* it is fatal


def test_fp64_refusal_happens_at_build_not_at_execution():
    """The error must surface before anything runs, as it would in codegen."""
    dev = _tiny_device(macif.CIM_INFERENCE_MACRO)
    prog = _one_descriptor_program(dev, Format.FP64)
    before = dev.total_descriptors_issued
    with pytest.raises(PrecisionNotDeclared):
        dev.build(prog)
    assert dev.total_descriptors_issued == before


def test_fp64_is_accepted_on_the_cm100_configuration():
    dev = _tiny_device(macif.CM100)
    built = dev.build(_one_descriptor_program(dev, Format.FP64))
    assert len(built) == 1


# ---------------------------------------------------------------------------
# fp16 vs bf16 -- same width, not interchangeable
# ---------------------------------------------------------------------------

def test_fp16_and_bf16_are_distinct_declarations():
    assert Format.FP16 is not Format.BF16
    assert int(Format.FP16) != int(Format.BF16)
    assert Format.FP16.itemsize == Format.BF16.itemsize == 2
    assert macif.CM100.supports(Format.BF16)
    assert not macif.CM100.supports(Format.FP16)


def test_bf16_is_not_substituted_for_a_requested_fp16_kernel():
    dev = _tiny_device(macif.CM100)          # declares bf16, not fp16
    prog = _one_descriptor_program(dev, Format.FP16)
    with pytest.raises(PrecisionNotDeclared) as exc:
        dev.build(prog)
    msg = str(exc.value)
    assert "does not declare fp16" in msg
    assert "REFUSED" in msg
    assert "10x" in msg                       # names the cost of substituting


def test_fp16_is_not_substituted_for_a_requested_bf16_kernel_either():
    only_fp16 = MacroCapabilities(
        name="fp16-only",
        formats={Format.FP16: FormatContract(macs_per_cycle=144,
                                             accumulator_width=32)},
    )
    dev = _tiny_device(only_fp16)
    with pytest.raises(PrecisionNotDeclared) as exc:
        dev.build(_one_descriptor_program(dev, Format.BF16))
    assert "does not declare bf16" in str(exc.value)
    assert "REFUSED" in str(exc.value)


def test_fp16_declaring_variant_accepts_the_fp16_kernel():
    dev = _tiny_device(macif.CM100_FP16)
    assert len(dev.build(_one_descriptor_program(dev, Format.FP16))) == 1


# ---------------------------------------------------------------------------
# contract shape
# ---------------------------------------------------------------------------

def test_rates_are_absolute_not_derived_from_another_format():
    """A rate expressed as a factor on another format is the shape of interface
    that hides a missing datapath, so each format states its own absolute rate."""
    for fmt in macif.CM100.declared_formats:
        contract = macif.CM100.require(fmt)
        assert isinstance(contract.macs_per_cycle, int)
        assert contract.macs_per_cycle > 0
    # fp64 is not 1/8 of bf16 by construction; it is its own declaration.
    assert macif.CM100.macs_per_cycle(Format.FP64) == 8
    assert macif.CM100.macs_per_cycle(Format.BF16) == 144


def test_accumulator_width_is_declared_per_format():
    assert macif.CM100.accumulator_width(Format.BF16) == 32
    assert macif.CM100.accumulator_width(Format.FP64) == 64


def test_swappable_flag_separates_the_tile_array_from_the_fma_cluster():
    assert macif.CM100_TILE_ARRAY.swappable
    assert not macif.CM100_FMA_CLUSTER.swappable
    # No CIM macro has a path to a 53-bit significand, so fp64 must not live
    # behind the swappable boundary.
    assert not macif.CM100_TILE_ARRAY.supports(Format.FP64)
    assert macif.CM100_FMA_CLUSTER.supports(Format.FP64)


def test_accumulator_features_are_declared():
    """broadcast-init and square-accumulate fold work into the accumulator at
    O(1) area instead of buying O(T^2) of array, so they are macro properties."""
    c = macif.UNIVERSAL.require(Format.FP32)
    assert c.broadcast_init and c.square_accumulate
    c2 = macif.CM100.require(Format.FP32)
    assert isinstance(c2.broadcast_init, bool)


def test_a_zero_rate_cannot_be_used_to_fake_an_unsupported_format():
    with pytest.raises(ValueError, match="omit the format"):
        FormatContract(macs_per_cycle=0, accumulator_width=32)


def test_merge_does_not_sum_rates_across_independently_scheduled_units():
    merged = macif.merge("m", macif.CM100_TILE_ARRAY, macif.CM100_FMA_CLUSTER)
    assert merged.macs_per_cycle(Format.BF16) == 144       # not 144 + something
    assert merged.macs_per_cycle(Format.FP32) == 16


# ---------------------------------------------------------------------------
# bf16 numerics
# ---------------------------------------------------------------------------

def test_bf16_round_trip_loses_mantissa_but_keeps_exponent_range():
    from argmind.precision import bf16_decode, bf16_encode

    values = np.array([1.0, 1.5, 3.1415927, 1e30, -2.5], dtype=np.float32)
    back = bf16_decode(bf16_encode(values))
    assert np.all(np.isfinite(back))
    assert back[0] == 1.0 and back[1] == 1.5
    assert abs(back[3] / 1e30 - 1.0) < 0.01       # exponent range survives
    assert abs(back[2] - 3.1415927) > 1e-4        # mantissa does not


def test_bf16_buffer_exposes_raw_containers_not_floats():
    dev = _tiny_device(macif.CM100)
    b = dev.alloc("b", 4, Format.BF16, Lifetime.PERSISTENT)
    assert b.array.dtype == np.uint16
    b.write(np.array([1.0, 2.0, 4.0, 8.0]))
    assert np.allclose(b.numbers(), [1.0, 2.0, 4.0, 8.0])
