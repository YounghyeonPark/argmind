# SPDX-License-Identifier: Apache-2.0
"""The runtime API and the functional simulator behind it.

This is the interface that must be byte-for-byte identical across the simulator,
the open reference core and the commercial core: code written against the free
stack has to run unmodified on licensed silicon. Treat every public name here as
published. The C API it mirrors is::

    am_device_t *am_open(const am_config_t *cfg);
    void        *am_alloc(am_device_t *, size_t, am_format_t, am_lifetime_t);
    void         am_mark_dirty(am_device_t *, void *);
    uint32_t     am_generation(am_device_t *, void *);
    am_program_t *am_build(am_device_t *, const am_desc_t *, size_t);
    int          am_call(am_device_t *, am_program_t *, const am_dirty_mask_t *,
                         void *const *outputs, size_t n_outputs);

``am_alloc`` returns a plain pointer. In Python that means a real numpy view into
the simulated host memory, so a mutation the user makes is a mutation the CM
sees. It is not an opaque handle and there is no accessor ceremony, because the C
API has none either.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .descriptor import DESC_BYTES, DESCRIPTOR_VERSION
from .errors import ConvergenceFailure, UnmappedAddress
from .kernels import KERNELS, accesses
from .macif import CM100, MacroCapabilities
from .memory import (
    BANK_WORD_BYTES,
    Buffer,
    Lifetime,
    ScratchpadPlan,
    _FlatHostMemory,
)
from .precision import Format
from .program import HazardReport, Program, SolverProgram, check_dependencies

#: CM-100 baseline scratchpad: **192 KiB**, 16 banks x 12 KiB = 128 B/cycle.
#:
#: Was 128 KiB. Corrected 2026-07-30 by descriptor-v0.1 ruling R8 -- see
#: :data:`argmind.descriptor.SCRATCHPAD_RULING`. The measured QUAD-MPC allocation
#: plan is 163.8 KiB at fp32 (138.8 KiB with A = [I;Gamma] exploited) against a
#: costed 128 KiB, and the flagship's three matrices ALONE are 125.0 KiB, so
#: 128 KiB was arithmetically impossible rather than merely tight. Bandwidth and
#: bank count do not move: this is a capacity correction, not a rebalancing.
CM100_SCRATCHPAD_BYTES = 192 * 1024
CM100_BANKS = 16

#: The capacity that was costed before ruling R8. Kept named so that a
#: regression test can assert the flagship does NOT fit in it -- a model that
#: cannot reproduce the failure it was corrected for is not evidence.
CM100_SCRATCHPAD_BYTES_V0 = 128 * 1024


@dataclass
class DeviceConfig:
    """Instantiation-time configuration. Fixed for the life of the device."""

    scratchpad_bytes: int = CM100_SCRATCHPAD_BYTES
    banks: int = CM100_BANKS
    macro: MacroCapabilities = CM100
    #: Descriptor format this device accepts: ``"0.1"`` or ``"0"``. A ``"0"``
    #: device refuses the opcodes v0.1 adds, which is how the cost of each
    #: addition stays measurable instead of becoming an assumption.
    descriptor_version: str = DESCRIPTOR_VERSION
    #: Legacy spelling of the same gate. ``True`` forces
    #: ``descriptor_version = "0"``; the two are reconciled at construction so
    #: there is one source of truth afterwards.
    strict_v0: bool = False
    #: Check that chain tokens / wait masks order every real dependency.
    check_hazards: bool = True
    host_bytes: int = 16 * 1024 * 1024

    def __post_init__(self) -> None:
        bank_word = self.scratchpad_bytes // max(self.banks, 1)
        if self.banks and (self.scratchpad_bytes // self.banks) < BANK_WORD_BYTES:
            raise ValueError(
                f"bank word would be {bank_word} B; the design rule is "
                f">= {BANK_WORD_BYTES} B (64 bits), below which the energy model "
                f"has no published data"
            )
        if self.strict_v0:
            self.descriptor_version = "0"
        if self.descriptor_version not in ("0", DESCRIPTOR_VERSION):
            raise ValueError(
                f"descriptor_version must be '0' or {DESCRIPTOR_VERSION!r}, "
                f"got {self.descriptor_version!r}"
            )
        self.strict_v0 = self.descriptor_version == "0"


@dataclass
class CallStats:
    """What one call cost, in the terms the residency model is stated in."""

    program: str = ""
    iterations: int = 0
    converged: bool = False
    descriptors_issued: int = 0
    bytes_fetched: int = 0
    bytes_written_back: int = 0
    buffers_fetched: tuple[str, ...] = ()
    buffers_trusted_resident: tuple[str, ...] = ()
    setup_ran: bool = False
    descriptor_bytes_issued: int = 0

    def summary(self) -> str:
        return (
            f"{self.program}: {self.iterations} iter"
            f"{' (converged)' if self.converged else ' (LIMIT)'}, "
            f"{self.descriptors_issued} desc, "
            f"fetched {self.bytes_fetched} B "
            f"({len(self.buffers_fetched)} buf), "
            f"resident {len(self.buffers_trusted_resident)} buf, "
            f"wrote back {self.bytes_written_back} B"
            f"{', setup ran' if self.setup_ran else ''}"
        )


class BuiltProgram:
    """A device-bound, validated descriptor block.

    Building is where every static check happens: precision against the MACIF
    declaration, opcodes against frozen v0, addresses against the allocation
    plan, and chain/wait tokens against the real dependencies. Nothing is
    checked for the first time at call time, because in the hardware nothing
    *can* be.
    """

    def __init__(self, device: "Device", program: Program) -> None:
        self.device = device
        self.program = program
        self.hazards: HazardReport | None = None
        self._validate()

    def _validate(self) -> None:
        dev, prog = self.device, self.program
        for i, d in enumerate(prog.descriptors):
            if dev.config.strict_v0:
                d.require_frozen_v0()
            # MACIF precision refusal -- at build time, not run time. BOTH
            # formats are checked: a mixed-precision descriptor whose store
            # format the macro does not declare is exactly as unrunnable as one
            # whose operand format it does not declare, and v0.1 made that a
            # reachable mistake by allowing the two to differ.
            ctx = (f"{prog.name}[{i}] {d.opcode.name}"
                   + (f" ({d.label})" if d.label else ""))
            dev.config.macro.require(d.prec_m, context=ctx)
            if d.is_mixed_precision:
                dev.config.macro.require(d.prec_v, context=ctx + " [result format]")
            reads, writes = accesses(d)
            for acc in list(reads) + list(writes):
                if acc.stop > acc.start:
                    dev.plan.lookup(acc.start)
                    dev.plan.lookup(acc.stop - 1)
        if dev.config.check_hazards:
            self.hazards = check_dependencies(prog, strict=True)

        # Static residency analysis: which buffers this block reads before
        # writing (true inputs, must be fetched) and which it writes.
        first_write: dict[str, int] = {}
        inputs: list[Buffer] = []
        outputs: list[Buffer] = []
        for i, d in enumerate(prog.descriptors):
            reads, writes = accesses(d)
            for acc in reads:
                if acc.stop <= acc.start:
                    continue
                buf = dev.buffer_at(acc.start)
                if buf.name not in first_write and buf not in inputs:
                    inputs.append(buf)
            for acc in writes:
                if acc.stop <= acc.start:
                    continue
                buf = dev.buffer_at(acc.start)
                first_write.setdefault(buf.name, i)
                if buf not in outputs:
                    outputs.append(buf)
        self.inputs: tuple[Buffer, ...] = tuple(inputs)
        self.outputs: tuple[Buffer, ...] = tuple(outputs)

    def __len__(self) -> int:
        return len(self.program)

    def __repr__(self) -> str:
        return (
            f"<BuiltProgram {self.program.name!r} {len(self.program)} desc, "
            f"{len(self.inputs)} in, {len(self.outputs)} out>"
        )


class BuiltSolver:
    """A device-bound :class:`SolverProgram`."""

    def __init__(self, device: "Device", solver: SolverProgram) -> None:
        self.device = device
        self.solver = solver
        self.setup = BuiltProgram(device, solver.setup)
        self.iteration = BuiltProgram(device, solver.iteration)
        self.residual = BuiltProgram(device, solver.residual)
        #: Generations of the setup block's inputs when setup last ran, so that a
        #: change to problem data re-triggers factorisation and nothing else does.
        self._setup_generations: dict[str, int] | None = None

    @property
    def blocks(self) -> tuple[BuiltProgram, ...]:
        return (self.setup, self.iteration, self.residual)

    def setup_needed(self, dirty_names: frozenset[str] = frozenset()) -> bool:
        """Must the setup block (factorisation) re-run before this call?

        RULING 2026-07-31 on codegen finding G5, which was a silent wrong answer
        reachable through the supported API. Two mechanisms decided two different
        things and could disagree: the **dirty mask** decides what is re-fetched,
        the **generation tag** decides whether setup re-runs. Naming `P` dirty
        without bumping its generation refreshed the operand and left the Cholesky
        factor *derived from it* resident and stale — the solve then converged, with
        no error, to the wrong answer.

        The rule adopted: **the dirty mask is conservative with respect to derived
        data.** Naming a setup input dirty invalidates everything derived from it.
        That can only cause more factorisations, never fewer, and re-factorising
        unnecessarily costs microseconds while a stale factor costs correctness in
        a control loop. Determinism is unaffected: the decision is made from the
        call's declared mask, not from data values, so it stays statically
        analysable.
        """
        if self._setup_generations is None:
            return True
        if any(b.name in dirty_names for b in self.setup.inputs):
            return True
        return any(
            self._setup_generations.get(b.name) != b.generation
            for b in self.setup.inputs
        )

    def _note_setup_ran(self) -> None:
        self._setup_generations = {b.name: b.generation for b in self.setup.inputs}


class Device:
    """A simulated Computational Module.

    Functional only: it computes the right numbers and accounts for data
    movement, and says nothing about cycles. The cycle-approximate model is a
    separate target that consumes the same descriptors.
    """

    def __init__(self, config: DeviceConfig | None = None, **kwargs) -> None:
        if config is None:
            config = DeviceConfig(**kwargs)
        elif kwargs:
            raise TypeError("pass either a DeviceConfig or keyword arguments")
        self.config = config
        self.scratchpad = np.zeros(config.scratchpad_bytes, dtype=np.uint8)
        self.host_memory = _FlatHostMemory(config.host_bytes)
        self.plan = ScratchpadPlan(config.scratchpad_bytes)
        self.buffers: dict[str, Buffer] = {}
        self._by_offset: list[tuple[int, int, Buffer]] = []
        # Lifetime counters, for the profiler.
        self.total_bytes_fetched = 0
        self.total_bytes_written_back = 0
        self.total_descriptors_issued = 0
        self.calls: list[CallStats] = []

    # -- allocation -------------------------------------------------------

    def alloc(
        self,
        name: str,
        shape: int | Sequence[int],
        fmt: Format = Format.FP64,
        lifetime: Lifetime = Lifetime.STREAMING,
    ) -> Buffer:
        """Allocate a buffer. Returns a plain pointer (a numpy view).

        Mirrors ``am_alloc(size, format, AM_PERSISTENT|AM_STREAMING)``.

        The scratchpad address is assigned here, in call order, and never moves.
        That is the static allocation plan: no runtime allocator sits in the
        solver path, so there is no allocation-dependent latency.
        """
        if name in self.buffers:
            raise ValueError(f"buffer {name!r} already allocated")
        shape = (int(shape),) if isinstance(shape, int) else tuple(int(s) for s in shape)
        fmt = Format(fmt)
        nbytes = int(np.prod(shape, dtype=np.int64)) * fmt.itemsize
        spad_offset = self.plan.place(name, nbytes, fmt, shape, Lifetime(lifetime))
        host_offset = self.host_memory.alloc(nbytes)
        buf = Buffer(self, name, fmt, shape, Lifetime(lifetime), host_offset,
                     spad_offset)
        self.buffers[name] = buf
        self._by_offset.append((spad_offset, spad_offset + max(nbytes, 1), buf))
        self._by_offset.sort()
        return buf

    def buffer_at(self, address: int) -> Buffer:
        """Which buffer contains a scratchpad byte address."""
        for start, stop, buf in self._by_offset:
            if start <= address < stop:
                return buf
        raise UnmappedAddress(f"scratchpad address 0x{address:x} is in no buffer")

    # -- build ------------------------------------------------------------

    def build(self, program: Program) -> BuiltProgram:
        """Validate and bind a descriptor block. Mirrors ``am_build``."""
        return BuiltProgram(self, program)

    def build_solver(self, solver: SolverProgram) -> BuiltSolver:
        return BuiltSolver(self, solver)

    # -- data movement ----------------------------------------------------

    def _fetch(self, buf: Buffer) -> int:
        src = self.host_memory[buf.host_offset:buf.host_offset + buf.nbytes]
        self.scratchpad[buf.spad_offset:buf.spad_offset + buf.nbytes] = src
        buf._note_fetched()
        self.total_bytes_fetched += buf.nbytes
        return buf.nbytes

    def _writeback(self, buf: Buffer) -> int:
        src = self.scratchpad[buf.spad_offset:buf.spad_offset + buf.nbytes]
        self.host_memory[buf.host_offset:buf.host_offset + buf.nbytes] = src
        buf._note_written_back()
        self.total_bytes_written_back += buf.nbytes
        return buf.nbytes

    def _resolve_dirty(self, dirty: Iterable[Buffer | str] | None) -> set[str]:
        names: set[str] = set()
        for item in dirty or ():
            names.add(item if isinstance(item, str) else item.name)
        unknown = names - set(self.buffers)
        if unknown:
            raise ValueError(f"dirty mask names unknown buffers: {sorted(unknown)}")
        return names

    def _stage_inputs(
        self, inputs: Sequence[Buffer], dirty: set[str], stats: CallStats
    ) -> None:
        fetched, resident = [], []
        for buf in inputs:
            must = (
                buf.lifetime is Lifetime.STREAMING
                or buf.name in dirty
                or not buf.is_resident
            )
            if must:
                stats.bytes_fetched += self._fetch(buf)
                fetched.append(buf.name)
            else:
                resident.append(buf.name)
        stats.buffers_fetched = tuple(fetched)
        stats.buffers_trusted_resident = tuple(resident)

    # -- execution --------------------------------------------------------

    def _run_block(self, built: BuiltProgram, stats: CallStats) -> None:
        macro = self.config.macro
        for d in built.program.descriptors:
            KERNELS[d.opcode](self.scratchpad, d, macro)
            stats.descriptors_issued += 1
            stats.descriptor_bytes_issued += DESC_BYTES
            self.total_descriptors_issued += 1
        for buf in built.outputs:
            buf._note_written_by_cm()

    def call(
        self,
        built: BuiltProgram,
        *,
        dirty: Iterable[Buffer | str] | None = None,
        outputs: Iterable[Buffer | str] = (),
    ) -> CallStats:
        """Run one descriptor block. Mirrors ``am_call``.

        ``dirty`` is the dirty mask: buffers the host rewrote and whose resident
        copies must therefore be refreshed. A buffer whose generation already
        differs from its resident copy's is refreshed whether or not it is named,
        so the mask is a belt-and-braces mechanism, not the only one.

        ``outputs`` names the buffers to write back to host memory. Anything not
        named stays resident on the CM -- which is the point.
        """
        stats = CallStats(program=built.program.name)
        self._stage_inputs(built.inputs, self._resolve_dirty(dirty), stats)
        self._run_block(built, stats)
        for buf in self._resolve_outputs(outputs):
            stats.bytes_written_back += self._writeback(buf)
        self.calls.append(stats)
        return stats

    def _resolve_outputs(self, outputs: Iterable[Buffer | str]) -> list[Buffer]:
        out = []
        for item in outputs:
            out.append(self.buffers[item] if isinstance(item, str) else item)
        return out

    def solve(
        self,
        built: BuiltSolver,
        *,
        dirty: Iterable[Buffer | str] | None = None,
        outputs: Iterable[Buffer | str] = (),
        raise_on_limit: bool = False,
    ) -> CallStats:
        """Run an iterative solver to convergence **inside one call**.

        The iteration loop, the residual evaluation and the convergence branch
        all happen here, between one doorbell and one completion -- there is no
        host round trip per iteration. That is what the sequencer buys, and
        modelling it any other way would quietly assume it away.

        Setup runs only if a buffer it reads has changed generation since the last
        time it ran. For an MPC re-solving a structurally identical problem every
        tick, that means the Cholesky factor is computed once and reused for the
        life of the plant model.
        """
        solver = built.solver
        stats = CallStats(program=solver.name)
        dirty_names = self._resolve_dirty(dirty)

        # Stage every input the solver will read, once, up front -- except
        # anything the setup block produces. A factorisation lives only on the
        # CM: host memory never holds it, so fetching it would overwrite the
        # resident factor with zeros.
        produced_by_setup = set(built.setup.outputs)
        seen: list[Buffer] = []
        for block in built.blocks:
            for buf in block.inputs:
                if buf not in seen and buf not in produced_by_setup:
                    seen.append(buf)
        self._stage_inputs(seen, dirty_names, stats)

        if built.setup_needed(dirty_names):
            self._run_block(built.setup, stats)
            built._note_setup_ran()
            stats.setup_ran = True

        for it in range(1, solver.max_iterations + 1):
            self._run_block(built.iteration, stats)
            stats.iterations = it
            if it % solver.check_every == 0:
                self._run_block(built.residual, stats)
                if solver.converged():
                    stats.converged = True
                    break

        for buf in self._resolve_outputs(outputs):
            stats.bytes_written_back += self._writeback(buf)
        self.calls.append(stats)
        if raise_on_limit and not stats.converged:
            raise ConvergenceFailure(
                f"{solver.name} did not converge in {solver.max_iterations} "
                f"iterations"
            )
        return stats

    # -- profiling --------------------------------------------------------

    def residency_report(self) -> str:
        """Static plan plus the current residency state, as the profiler emits."""
        lines = [self.plan.report(), "", "residency:"]
        for buf in self.buffers.values():
            state = "resident" if buf.is_resident else "stale/absent"
            lines.append(
                f"  {buf.name:<18} {buf.lifetime.name.lower():<10} "
                f"gen={buf.generation:<4} resident_gen={buf.resident_generation:<4} "
                f"{state}"
            )
        lines += [
            "",
            f"lifetime totals: fetched {self.total_bytes_fetched} B, "
            f"written back {self.total_bytes_written_back} B, "
            f"{self.total_descriptors_issued} descriptors",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"<Device macro={self.config.macro.name!r} "
            f"scratchpad={self.config.scratchpad_bytes // 1024}KiB "
            f"strict_v0={self.config.strict_v0}>"
        )
