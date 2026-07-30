"""Dependency tokens: assignment, hazard detection, and what v0's one token costs.

The hazard checker matters because a missing dependency is invisible on an
in-order functional model and becomes a race the moment the hardware runs
descriptors ahead of each other -- which it must, since run-ahead is required
unless chain blocking is full-width.
"""

from __future__ import annotations

import pytest

from argmind import (
    Descriptor,
    Device,
    DeviceConfig,
    Format,
    Lifetime,
    MissingDependency,
    Opcode,
    Program,
    auto_chain,
    check_dependencies,
    concat,
    macif,
)
from argmind.errors import DependencyViolation, FieldOverflow
from argmind.program import critical_path, dependency_graph


def _device() -> Device:
    return Device(DeviceConfig(scratchpad_bytes=1 << 16, banks=8,
                               macro=macif.UNIVERSAL, strict_v0=False,
                               host_bytes=1 << 18))


def _chain_of(dev, length: int):
    """v_{i+1} := v_i, a strictly serial chain of `length` descriptors."""
    bufs = [dev.alloc(f"v{i}", 4, Format.FP64, Lifetime.PERSISTENT)
            for i in range(length + 1)]
    return [Descriptor(Opcode.SCAL, Format.FP64, b=bufs[i].spad_offset,
                       c=bufs[i + 1].spad_offset, m=4, alpha=1.0)
            for i in range(length)]


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def test_missing_token_between_dependent_descriptors_is_detected():
    dev = _device()
    descs = _chain_of(dev, 3)                     # no tokens at all
    with pytest.raises(MissingDependency, match="RAW"):
        check_dependencies(Program("p", descs), strict=True)


def test_wait_on_an_unproduced_token_is_detected():
    dev = _device()
    a = dev.alloc("a", 4, Format.FP64, Lifetime.PERSISTENT)
    b = dev.alloc("b", 4, Format.FP64, Lifetime.PERSISTENT)
    prog = Program("p", [Descriptor(Opcode.SCAL, Format.FP64, b=a.spad_offset,
                                    c=b.spad_offset, m=4, alpha=1.0,
                                    wait_token=42)])
    with pytest.raises(DependencyViolation, match="token 42"):
        check_dependencies(prog, strict=True)


def test_write_after_read_hazard_is_detected():
    dev = _device()
    a = dev.alloc("a", 4, Format.FP64, Lifetime.PERSISTENT)
    b = dev.alloc("b", 4, Format.FP64, Lifetime.PERSISTENT)
    c = dev.alloc("c", 4, Format.FP64, Lifetime.PERSISTENT)
    descs = [
        Descriptor(Opcode.SCAL, Format.FP64, b=a.spad_offset, c=b.spad_offset,
                   m=4, alpha=1.0, chain_token=1),
        # overwrites `a`, which descriptor 0 reads -- WAR, and not ordered
        Descriptor(Opcode.SCAL, Format.FP64, b=c.spad_offset, c=a.spad_offset,
                   m=4, alpha=1.0, chain_token=2),
    ]
    report = check_dependencies(Program("p", descs), strict=False)
    assert not report.ok
    assert report.unordered[0][2] == "WAR"


def test_the_device_refuses_to_build_a_block_with_an_unordered_hazard():
    dev = _device()
    descs = _chain_of(dev, 3)
    with pytest.raises(MissingDependency):
        dev.build(Program("p", descs))


def test_hazard_checking_can_be_switched_off():
    dev = Device(DeviceConfig(scratchpad_bytes=1 << 16, banks=8,
                              macro=macif.UNIVERSAL, strict_v0=False,
                              check_hazards=False, host_bytes=1 << 18))
    dev.build(Program("p", _chain_of(dev, 3)))


# ---------------------------------------------------------------------------
# Assignment
# ---------------------------------------------------------------------------

def test_auto_chain_output_passes_the_checker():
    dev = _device()
    result = auto_chain(_chain_of(dev, 5))
    report = check_dependencies(Program("p", result.descriptors), strict=True)
    assert report.ok


def test_auto_chain_gives_a_serial_chain_its_true_depth():
    dev = _device()
    result = auto_chain(_chain_of(dev, 5))
    assert result.policy == "minimal"
    assert result.true_depth == 5
    assert result.chain_depth == 5
    assert result.depth_penalty == 1.0
    assert result.forced_serialisations == 0


def test_auto_chain_leaves_independent_descriptors_unordered():
    """Two descriptors with no shared bytes must not be chained to each other."""
    dev = _device()
    bufs = [dev.alloc(f"v{i}", 4, Format.FP64, Lifetime.PERSISTENT)
            for i in range(4)]
    descs = [
        Descriptor(Opcode.SCAL, Format.FP64, b=bufs[0].spad_offset,
                   c=bufs[1].spad_offset, m=4, alpha=1.0),
        Descriptor(Opcode.SCAL, Format.FP64, b=bufs[2].spad_offset,
                   c=bufs[3].spad_offset, m=4, alpha=1.0),
    ]
    result = auto_chain(descs)
    assert result.true_depth == 1
    assert all(d.wait_token == 0 for d in result.descriptors)


def test_a_forking_dependency_graph_forces_full_serialisation():
    """Limitation L3, as an executable statement.

    Two independent chains merging into one consumer cannot be ordered by a
    single wait token, so the whole block has to be linearised and the emitted
    dependency depth exceeds the achievable one.
    """
    dev = _device()
    z = dev.alloc("z", 4, Format.FP64, Lifetime.PERSISTENT)
    y = dev.alloc("y", 4, Format.FP64, Lifetime.PERSISTENT)
    t = dev.alloc("t", 4, Format.FP64, Lifetime.PERSISTENT)
    x = dev.alloc("x", 4, Format.FP64, Lifetime.PERSISTENT)
    q = dev.alloc("q", 4, Format.FP64, Lifetime.PERSISTENT)
    r = dev.alloc("r", 4, Format.FP64, Lifetime.PERSISTENT)
    A = dev.alloc("A", (4, 4), Format.FP64, Lifetime.PERSISTENT)
    descs = [
        # chain 1: build t from z and y
        Descriptor(Opcode.SCAL, Format.FP64, b=z.spad_offset, c=t.spad_offset,
                   m=4, alpha=1.0),
        Descriptor(Opcode.AXPY, Format.FP64, b=y.spad_offset, c=t.spad_offset,
                   m=4, alpha=-1.0),
        # chain 2: build r from x and q
        Descriptor(Opcode.SCAL, Format.FP64, b=x.spad_offset, c=r.spad_offset,
                   m=4, alpha=1.0),
        Descriptor(Opcode.AXPY, Format.FP64, b=q.spad_offset, c=r.spad_offset,
                   m=4, alpha=-1.0),
        # merge: r += A' t  -- depends on both chains
        Descriptor(Opcode.GEMV, Format.FP64, a=A.spad_offset, b=t.spad_offset,
                   c=r.spad_offset, m=4, k=4, lda=4, alpha=1.0,
                   accumulate=True),
    ]
    graph = dependency_graph(descs)
    assert graph[4] == (0, 1, 2, 3)              # a genuine four-way merge
    assert critical_path(graph) == 3
    result = auto_chain(descs)
    assert result.policy == "serial"
    assert result.chain_depth == 5
    assert result.chain_depth > result.true_depth
    assert result.depth_penalty > 1.0
    check_dependencies(Program("p", result.descriptors), strict=True)


def test_multi_producer_descriptors_are_counted():
    dev = _device()
    result = auto_chain(_chain_of(dev, 4))
    assert result.multi_producer == ()


def test_concat_renumbers_tokens_and_stays_valid():
    dev = _device()
    a = Program("a", auto_chain(_chain_of(dev, 3)).descriptors)
    joined = concat("aa", a, a, a)
    assert len(joined) == 9
    check_dependencies(joined, strict=True)


def test_concat_refuses_more_descriptors_than_the_token_field_holds():
    dev = _device()
    block = Program("b", auto_chain(_chain_of(dev, 30)).descriptors)
    with pytest.raises(FieldOverflow, match="255 chain tokens"):
        concat("too_long", *([block] * 9))


def test_token_values_fit_the_eight_bit_field():
    dev = _device()
    result = auto_chain(_chain_of(dev, 200))
    for d in result.descriptors:
        assert 0 <= d.chain_token <= 255
        assert 0 <= d.wait_token <= 255
        d.encode()


# ---------------------------------------------------------------------------
# Frozen-v0 gating
# ---------------------------------------------------------------------------

def test_program_reports_which_proposed_opcodes_it_uses():
    dev = _device()
    v = dev.alloc("v", 4, Format.FP64, Lifetime.PERSISTENT)
    s = dev.alloc("s", 1, Format.FP64, Lifetime.PERSISTENT)
    prog = Program("p", [Descriptor(Opcode.NRMI, Format.FP64, a=s.spad_offset,
                                    b=v.spad_offset, m=4)])
    assert not prog.uses_only_frozen_v0
    assert prog.proposed_opcodes == (Opcode.NRMI,)


def test_strict_v0_device_refuses_a_proposed_opcode():
    from argmind import UnsupportedOpcode

    dev = Device(DeviceConfig(scratchpad_bytes=1 << 16, banks=8,
                              macro=macif.UNIVERSAL, strict_v0=True,
                              host_bytes=1 << 18))
    v = dev.alloc("v", 4, Format.FP64, Lifetime.PERSISTENT)
    s = dev.alloc("s", 1, Format.FP64, Lifetime.PERSISTENT)
    prog = Program("p", [Descriptor(Opcode.NRMI, Format.FP64, a=s.spad_offset,
                                    b=v.spad_offset, m=4)])
    with pytest.raises(UnsupportedOpcode, match="NRMI"):
        dev.build(prog)


def test_descriptor_referencing_an_unallocated_address_is_refused():
    from argmind.errors import UnmappedAddress

    dev = _device()
    v = dev.alloc("v", 4, Format.FP64, Lifetime.PERSISTENT)
    prog = Program("p", [Descriptor(Opcode.SCAL, Format.FP64,
                                    b=v.spad_offset, c=0xF000, m=4, alpha=1.0)])
    with pytest.raises(UnmappedAddress):
        dev.build(prog)
