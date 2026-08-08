# SPDX-License-Identifier: Apache-2.0
"""Dependency tokens: assignment, hazard detection, and what v0's one token costs.

The hazard checker matters because a missing dependency is invisible on an
in-order functional model and becomes a race the moment the hardware runs
descriptors ahead of each other -- which it must, since run-ahead is required
unless chain blocking is full-width.
"""

from __future__ import annotations

import pytest

from argmind import (
    N_TOKENS,
    Descriptor,
    Device,
    DeviceConfig,
    Format,
    Lifetime,
    MissingDependency,
    Opcode,
    Program,
    auto_chain,
    auto_chain_v0,
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
                                    wait_mask=1 << 5)])
    with pytest.raises(DependencyViolation, match="token 5"):
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
    assert result.policy == "mask"
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
    assert all(d.wait_mask == 0 for d in result.descriptors)


def test_a_forking_graph_forced_v0_to_serialise_and_v0_1_does_not():
    """Limitation L3 and ruling R3, as one executable statement.

    Two independent chains merging into one consumer cannot be ordered by a
    single wait token, so under v0 the whole block had to be linearised and the
    emitted dependency depth exceeded the achievable one. Both encodings are
    exercised here: the v0 baseline has to stay reproducible, because the entire
    justification for changing the token fields is the comparison between them.
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
    # v0: one wait token, so the only encodable schedule is fully serial.
    v0 = auto_chain_v0(descs)
    assert v0.policy == "serial"
    assert v0.chain_depth == 5
    assert v0.chain_depth > v0.true_depth
    assert v0.depth_penalty > 1.0
    check_dependencies(Program("v0", v0.descriptors), strict=True)

    # v0.1: the wait mask names the exact producer set, so emitted depth == true
    # depth and nothing is serialised that did not have to be.
    result = auto_chain(descs)
    assert result.policy == "mask"
    assert result.chain_depth == result.true_depth == 3
    assert result.depth_penalty == 1.0
    assert result.forced_serialisations == 0
    assert result.max_merge_width == 4
    assert result.descriptors[4].merge_width == 4
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


def test_concat_has_no_descriptor_limit_because_tokens_are_recycled():
    """v0 raised above 255 descriptors. That limit was based on the wrong model
    of a token -- a scoreboard slot, not a name -- and ruling R3 removes it."""
    dev = _device()
    block = Program("b", auto_chain(_chain_of(dev, 30)).descriptors)
    joined = concat("long", *([block] * 9))
    assert len(joined) == 270
    check_dependencies(joined, strict=True)


def test_token_values_fit_the_three_bit_field_at_any_block_length():
    dev = _device()
    result = auto_chain(_chain_of(dev, 200))
    assert result.tokens_used <= N_TOKENS
    for d in result.descriptors:
        assert 0 <= d.chain_token <= N_TOKENS - 1
        assert 0 <= d.wait_mask <= 255
        d.encode()
    check_dependencies(Program("p", result.descriptors), strict=True)


def test_token_exhaustion_costs_overlap_and_never_correctness():
    """The safety property the ruling rests on, and it needed a rule to be true.

    Reuse is only safe because a descriptor that reclaims a token also waits on
    it, so the current owner transitively covers every past owner. Without that
    rule the concatenated block races -- which is how the bug was found.
    """
    dev = _device()
    # A wide fan-out then fan-in, deliberately exceeding the 8-token space.
    src = dev.alloc("src", 4, Format.FP64, Lifetime.PERSISTENT)
    mids = [dev.alloc(f"m{i}", 4, Format.FP64, Lifetime.PERSISTENT)
            for i in range(N_TOKENS + 6)]
    sink = dev.alloc("sink", 4, Format.FP64, Lifetime.PERSISTENT)
    descs = [Descriptor(Opcode.SCAL, Format.FP64, b=src.spad_offset,
                        c=mm.spad_offset, m=4, alpha=1.0) for mm in mids]
    descs.append(Descriptor(Opcode.SCAL, Format.FP64, b=mids[0].spad_offset,
                            c=sink.spad_offset, m=4, alpha=1.0))
    for mm in mids[1:]:
        descs.append(Descriptor(Opcode.AXPY, Format.FP64, b=mm.spad_offset,
                                c=sink.spad_offset, m=4, alpha=1.0))
    result = auto_chain(descs)
    assert result.peak_live_tokens > N_TOKENS      # pressure really is exceeded
    assert result.forced_serialisations > 0        # ...and it cost overlap
    assert result.chain_depth >= result.true_depth  # never less
    # The only thing that must not happen: an unordered hazard.
    check_dependencies(Program("p", result.descriptors), strict=True)


def test_measured_token_pressure_on_the_admm_blocks_fits_the_field():
    """The measurement that sizes the token field at 3 bits."""
    from argmind.solvers.admm import ADMMConfig, ADMMSolver

    dev = Device(DeviceConfig(scratchpad_bytes=1 << 24, banks=16,
                              macro=macif.UNIVERSAL, host_bytes=1 << 25))
    s = ADMMSolver(dev, 80, 320, ADMMConfig())
    peaks = {k: s.chain_info[k].peak_live_tokens
             for k in ("setup", "iteration", "residual")}
    assert peaks["setup"] <= 3
    assert peaks["iteration"] <= N_TOKENS - 1      # measured 7 of 8: one spare
    assert peaks["residual"] <= 3
    assert all(v <= N_TOKENS for v in peaks.values())
    for k in ("setup", "iteration", "residual"):
        c = s.chain_info[k]
        assert c.forced_serialisations == 0
        assert c.depth_penalty == 1.0, k


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
