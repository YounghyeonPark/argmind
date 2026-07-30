"""The pointer contract, the static allocation plan, and generation tags.

The generation-tag tests are the important ones. A stale read in a control loop
is silent and dangerous: the CM computes on last tick's sensor data and the
controller commands a plausible-looking wrong action. So it is not enough that
the mechanism exists -- it must be demonstrated that omitting it *does* produce
a stale read, otherwise the tests are passing for the wrong reason.
"""

from __future__ import annotations

import numpy as np
import pytest

from argmind import (
    Descriptor,
    Device,
    DeviceConfig,
    Format,
    Lifetime,
    Opcode,
    Program,
    ScratchpadExhausted,
    StaleHostCopy,
    macif,
)
from argmind.memory import BANK_WORD_BYTES


def _device(**kw) -> Device:
    kw.setdefault("scratchpad_bytes", 64 * 1024)
    kw.setdefault("banks", 8)
    kw.setdefault("macro", macif.UNIVERSAL)
    kw.setdefault("strict_v0", False)
    kw.setdefault("host_bytes", 1 << 20)
    return Device(DeviceConfig(**kw))


def _copy_program(dev: Device, src, dst, n: int, fmt=Format.FP64) -> Program:
    """dst := 1.0 * src, as a single SCAL descriptor."""
    return Program("copy", [Descriptor(Opcode.SCAL, fmt, b=src.spad_offset,
                                       c=dst.spad_offset, m=n, alpha=1.0)])


# ---------------------------------------------------------------------------
# The pointer contract
# ---------------------------------------------------------------------------

def test_alloc_returns_a_view_not_a_copy():
    dev = _device()
    buf = dev.alloc("v", 8, Format.FP64, Lifetime.PERSISTENT)
    a = buf.array
    assert isinstance(a, np.ndarray)
    a[3] = 7.0
    assert buf.array[3] == 7.0            # same memory, seen through a new view
    assert np.shares_memory(buf.array, dev.host_memory.raw)


def test_the_cm_sees_a_mutation_made_through_the_raw_view():
    """Provided the write is declared -- see the generation-tag tests."""
    dev = _device()
    src = dev.alloc("src", 4, Format.FP64, Lifetime.PERSISTENT)
    dst = dev.alloc("dst", 4, Format.FP64, Lifetime.PERSISTENT)
    built = dev.build(_copy_program(dev, src, dst, 4))
    src.array[:] = [1.0, 2.0, 3.0, 4.0]
    src.mark_dirty()
    dev.call(built, outputs=[dst])
    assert np.allclose(dst.numbers(), [1.0, 2.0, 3.0, 4.0])


def test_write_is_in_place_and_never_reallocates():
    dev = _device()
    buf = dev.alloc("v", 8, Format.FP64, Lifetime.PERSISTENT)
    offset, spad = buf.host_offset, buf.spad_offset
    ptr = buf.array.__array_interface__["data"][0]
    for _ in range(5):
        buf.write(np.arange(8.0))
    assert (buf.host_offset, buf.spad_offset) == (offset, spad)
    assert buf.array.__array_interface__["data"][0] == ptr


def test_writable_context_manager_bumps_the_generation_on_exit():
    dev = _device()
    buf = dev.alloc("v", 4, Format.FP64, Lifetime.PERSISTENT)
    gen = buf.generation
    with buf.writable() as a:
        a[:] = 1.0
        assert buf.generation == gen        # not yet
    assert buf.generation == gen + 1


# ---------------------------------------------------------------------------
# Generation tags: the mechanism, and the failure it prevents
# ---------------------------------------------------------------------------

def test_unchanged_persistent_buffer_is_not_refetched():
    dev = _device()
    src = dev.alloc("src", 16, Format.FP64, Lifetime.PERSISTENT)
    dst = dev.alloc("dst", 16, Format.FP64, Lifetime.PERSISTENT)
    built = dev.build(_copy_program(dev, src, dst, 16))
    src.write(np.arange(16.0))

    first = dev.call(built, outputs=[dst])
    assert src.name in first.buffers_fetched
    assert first.bytes_fetched == src.nbytes

    second = dev.call(built, outputs=[dst])
    assert second.bytes_fetched == 0
    assert src.name in second.buffers_trusted_resident


def test_streaming_buffer_is_always_refetched():
    dev = _device()
    src = dev.alloc("src", 16, Format.FP64, Lifetime.STREAMING)
    dst = dev.alloc("dst", 16, Format.FP64, Lifetime.PERSISTENT)
    built = dev.build(_copy_program(dev, src, dst, 16))
    src.write(np.arange(16.0))
    for _ in range(3):
        stats = dev.call(built, outputs=[dst])
        assert stats.bytes_fetched == src.nbytes


def test_generation_bump_causes_a_refetch_and_a_fresh_result():
    dev = _device()
    src = dev.alloc("src", 4, Format.FP64, Lifetime.PERSISTENT)
    dst = dev.alloc("dst", 4, Format.FP64, Lifetime.PERSISTENT)
    built = dev.build(_copy_program(dev, src, dst, 4))

    src.write(np.array([1.0, 1.0, 1.0, 1.0]))
    dev.call(built, outputs=[dst])
    assert np.allclose(dst.numbers(), 1.0)

    src.write(np.array([9.0, 9.0, 9.0, 9.0]))       # bumps the generation
    stats = dev.call(built, outputs=[dst])
    assert stats.bytes_fetched == src.nbytes
    assert np.allclose(dst.numbers(), 9.0)


def test_UNDECLARED_write_produces_a_stale_read_and_the_tag_is_what_prevents_it():
    """The load-bearing test.

    Writing through the raw pointer without declaring the write must leave the
    resident copy stale, and the CM must then compute on the OLD data. If this
    test ever starts passing trivially -- i.e. the fresh value appears anyway --
    then the simulator has stopped modelling a non-coherent scratchpad and the
    generation-tag mechanism is no longer being exercised by anything.
    """
    dev = _device()
    src = dev.alloc("src", 4, Format.FP64, Lifetime.PERSISTENT)
    dst = dev.alloc("dst", 4, Format.FP64, Lifetime.PERSISTENT)
    built = dev.build(_copy_program(dev, src, dst, 4))

    src.write(np.full(4, 1.0))
    dev.call(built, outputs=[dst])
    assert np.allclose(dst.numbers(), 1.0)

    # Undeclared write: legal (it is a pointer) but invisible to the runtime.
    src.array[:] = 42.0
    assert np.allclose(src.numbers(), 42.0)          # host memory really changed
    assert src.is_resident                            # ...but the CM still trusts

    stats = dev.call(built, outputs=[dst])
    assert stats.bytes_fetched == 0
    assert np.allclose(dst.numbers(), 1.0), "expected the STALE value"

    # Declaring the write after the fact repairs it. This is the entire mechanism.
    src.mark_dirty()
    assert not src.is_resident
    stats = dev.call(built, outputs=[dst])
    assert stats.bytes_fetched == src.nbytes
    assert np.allclose(dst.numbers(), 42.0)


def test_dirty_mask_forces_a_refetch_even_without_a_generation_change():
    """Belt and braces: the mask is a second, independent path to a refetch."""
    dev = _device()
    src = dev.alloc("src", 4, Format.FP64, Lifetime.PERSISTENT)
    dst = dev.alloc("dst", 4, Format.FP64, Lifetime.PERSISTENT)
    built = dev.build(_copy_program(dev, src, dst, 4))
    src.write(np.full(4, 1.0))
    dev.call(built, outputs=[dst])

    src.array[:] = 5.0                     # undeclared
    stats = dev.call(built, dirty=[src], outputs=[dst])
    assert stats.bytes_fetched == src.nbytes
    assert np.allclose(dst.numbers(), 5.0)


def test_dirty_mask_accepts_names_and_rejects_unknown_ones():
    dev = _device()
    src = dev.alloc("src", 4, Format.FP64, Lifetime.PERSISTENT)
    dst = dev.alloc("dst", 4, Format.FP64, Lifetime.PERSISTENT)
    built = dev.build(_copy_program(dev, src, dst, 4))
    src.write(np.full(4, 1.0))
    dev.call(built, dirty=["src"], outputs=[dst])
    with pytest.raises(ValueError, match="unknown buffers"):
        dev.call(built, dirty=["nope"], outputs=[dst])


def test_cm_written_buffer_reports_a_stale_host_copy_until_written_back():
    dev = _device()
    src = dev.alloc("src", 4, Format.FP64, Lifetime.PERSISTENT)
    dst = dev.alloc("dst", 4, Format.FP64, Lifetime.PERSISTENT)
    built = dev.build(_copy_program(dev, src, dst, 4))
    src.write(np.full(4, 3.0))

    dev.call(built)                       # no outputs requested: result stays on CM
    assert dst.host_is_stale
    with pytest.raises(StaleHostCopy, match="outputs"):
        dst.read()
    assert np.allclose(dst.numbers(), 0.0)    # the raw pointer does not lie or raise

    dev.call(built, outputs=[dst])
    assert not dst.host_is_stale
    assert np.allclose(dst.read(), 3.0)


def test_cm_written_result_is_not_clobbered_by_a_stale_fetch_next_call():
    """A buffer the CM wrote must not be overwritten from host memory."""
    dev = _device()
    acc = dev.alloc("acc", 4, Format.FP64, Lifetime.PERSISTENT)
    one = dev.alloc("one", 4, Format.FP64, Lifetime.PERSISTENT)
    one.write(np.ones(4))
    acc.write(np.zeros(4))
    built = dev.build(Program("accumulate", [
        Descriptor(Opcode.AXPY, Format.FP64, b=one.spad_offset,
                   c=acc.spad_offset, m=4, alpha=1.0)]))
    for expected in (1.0, 2.0, 3.0):
        dev.call(built, outputs=[acc])
        assert np.allclose(acc.read(), expected)


# ---------------------------------------------------------------------------
# Static allocation plan
# ---------------------------------------------------------------------------

def test_allocation_plan_is_deterministic_and_bank_aligned():
    def layout():
        dev = _device()
        for name, n in (("a", 3), ("b", 17), ("c", 1), ("d", 40)):
            dev.alloc(name, n, Format.FP64, Lifetime.PERSISTENT)
        return [(a.name, a.offset, a.nbytes) for a in dev.plan.allocations]

    first = layout()
    assert first == layout()                      # pure function of the requests
    for _, offset, _ in first:
        assert offset % BANK_WORD_BYTES == 0


def test_plan_refuses_to_overflow_rather_than_growing():
    dev = _device(scratchpad_bytes=1024)
    dev.alloc("big", 120, Format.FP64, Lifetime.PERSISTENT)     # 960 B
    with pytest.raises(ScratchpadExhausted, match="does not fit"):
        dev.alloc("more", 64, Format.FP64, Lifetime.PERSISTENT)


def test_bank_word_below_64_bits_is_refused():
    with pytest.raises(ValueError, match="64 bits"):
        DeviceConfig(scratchpad_bytes=64, banks=16)


def test_lifetime_values_match_the_c_api_enum():
    assert int(Lifetime.STREAMING) == 0
    assert int(Lifetime.PERSISTENT) == 1


def test_residency_report_names_every_buffer():
    dev = _device()
    for name in ("p", "q"):
        dev.alloc(name, 8, Format.FP32, Lifetime.PERSISTENT)
    text = dev.residency_report()
    assert "p" in text and "q" in text and "scratchpad" in text
