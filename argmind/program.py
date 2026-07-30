"""Programs: descriptor blocks, their static analysis, and the convergence loop.

A :class:`Program` is an ordered descriptor block. A :class:`SolverProgram` is
what an iterative solver actually is on this architecture: a setup block that
runs only when the problem data changes, an iteration block, a residual block,
and a **convergence predicate**.

Where the predicate lives matters, so it is stated here rather than buried:
descriptor format v0 has no branch and no compare, and it should not have one.
Data-dependent control flow is the sequencer's job -- it is a small core running
its own instruction stream, and running the convergence test there without a host
round-trip is the mechanism the whole architecture is built around. So the
predicate in this model stands for a handful of sequencer instructions reading
scalars out of scratchpad, not for a descriptor. The sequencer ISA is a separate
Phase-1 freeze item and is **not** frozen yet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

from .descriptor import Descriptor, FROZEN_V0_OPCODES, Opcode
from .errors import DependencyViolation, FieldOverflow, MissingDependency
from .kernels import accesses


@dataclass(frozen=True)
class Program:
    """An ordered block of descriptors."""

    name: str
    descriptors: tuple[Descriptor, ...]

    def __init__(self, name: str, descriptors: Iterable[Descriptor]) -> None:
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "descriptors", tuple(descriptors))

    def __len__(self) -> int:
        return len(self.descriptors)

    def __iter__(self):
        return iter(self.descriptors)

    @property
    def uses_only_frozen_v0(self) -> bool:
        return all(d.opcode in FROZEN_V0_OPCODES for d in self.descriptors)

    @property
    def proposed_opcodes(self) -> tuple[Opcode, ...]:
        seen: list[Opcode] = []
        for d in self.descriptors:
            if d.opcode not in FROZEN_V0_OPCODES and d.opcode not in seen:
                seen.append(d.opcode)
        return tuple(seen)

    def listing(self) -> str:
        return "\n".join(f"{i:>3}  {d}" for i, d in enumerate(self.descriptors))


@dataclass
class HazardReport:
    """Result of checking whether chain/wait tokens order the real dependencies.

    ``unordered`` lists read-after-write and write-after-write pairs that the
    tokens do not order. On an in-order functional model these are harmless; with
    hardware run-ahead they are races. ``multi_producer`` lists descriptors that
    depend on more than one earlier descriptor -- interesting because descriptor
    v0 has exactly one wait token, so codegen has to serialise the producers and
    give up overlap it could otherwise have had.
    """

    unordered: list[tuple[int, int, str]] = field(default_factory=list)
    multi_producer: list[tuple[int, tuple[int, ...]]] = field(default_factory=list)
    chain_length: int = 0

    @property
    def ok(self) -> bool:
        return not self.unordered

    def summary(self) -> str:
        return (
            f"{len(self.unordered)} unordered hazard(s); "
            f"{len(self.multi_producer)} descriptor(s) with >1 producer; "
            f"critical chain {self.chain_length}"
        )


def check_dependencies(program: Program, *, strict: bool = True) -> HazardReport:
    """Verify the token annotation against the actual byte-range dependencies.

    Two things are checked:

    1. Every ``wait_token`` was produced by an earlier descriptor. A wait on a
       token nobody produces would hang the hardware.
    2. Every true dependency (read-after-write, write-after-write, write-after-read)
       between two descriptors is ordered by a token path. This is the check that
       matters, because it is exactly the class of bug that an in-order model
       cannot expose and that run-ahead turns into a wrong answer.
    """
    descs = program.descriptors
    produced: dict[int, int] = {}          # token -> producing index
    waits_on: list[int | None] = []
    for i, d in enumerate(descs):
        if d.wait_token:
            if d.wait_token not in produced:
                raise DependencyViolation(
                    f"{program.name}[{i}] ({d}) waits on token {d.wait_token}, "
                    f"which no earlier descriptor produces"
                )
            waits_on.append(produced[d.wait_token])
        else:
            waits_on.append(None)
        if d.chain_token:
            produced[d.chain_token] = i

    # Transitive closure of "must complete before", over token edges.
    before: list[set[int]] = []
    for i in range(len(descs)):
        s: set[int] = set()
        p = waits_on[i]
        if p is not None:
            s.add(p)
            s |= before[p]
        before.append(s)

    report = HazardReport(chain_length=max((len(s) for s in before), default=0) + 1)
    acc = [accesses(d) for d in descs]
    for j in range(len(descs)):
        producers: list[int] = []
        rj, wj = acc[j]
        for i in range(j):
            ri, wi = acc[i]
            kind = None
            if any(a.overlaps(b) for a in rj for b in wi):
                kind = "RAW"
            elif any(a.overlaps(b) for a in wj for b in wi):
                kind = "WAW"
            elif any(a.overlaps(b) for a in wj for b in ri):
                kind = "WAR"
            if kind is None:
                continue
            producers.append(i)
            if i not in before[j]:
                report.unordered.append((i, j, kind))
        if len(producers) > 1:
            report.multi_producer.append((j, tuple(producers)))

    if strict and report.unordered:
        i, j, kind = report.unordered[0]
        raise MissingDependency(
            f"{program.name}: {kind} hazard between descriptor {i} "
            f"({descs[i]}) and {j} ({descs[j]}) is not ordered by any "
            f"chain/wait token. There are {len(report.unordered)} such pair(s). "
            f"This is silent on an in-order model and a race with run-ahead."
        )
    return report


_COPY_FIELDS = (
    "opcode", "precision", "a", "b", "c", "m", "k", "n", "lda", "ldb", "ldc",
    "alpha", "transpose", "structure", "accumulate", "unit_diag", "label",
)


def _retag(d: Descriptor, chain_token: int, wait_token: int) -> Descriptor:
    return Descriptor(
        **{f: getattr(d, f) for f in _COPY_FIELDS},
        chain_token=chain_token,
        wait_token=wait_token,
    )


def dependency_graph(descriptors: Sequence[Descriptor]) -> list[tuple[int, ...]]:
    """For each descriptor, the indices of every earlier descriptor it depends on.

    Derived from the descriptors alone -- byte-range overlap on reads and writes
    -- so this is exactly the information the hardware has, and exactly what a
    code generator must reason about.
    """
    acc = [accesses(d) for d in descriptors]
    graph: list[tuple[int, ...]] = []
    for j in range(len(descriptors)):
        rj, wj = acc[j]
        producers = []
        for i in range(j):
            ri, wi = acc[i]
            if (any(a.overlaps(b) for a in rj for b in wi)
                    or any(a.overlaps(b) for a in wj for b in wi)
                    or any(a.overlaps(b) for a in wj for b in ri)):
                producers.append(i)
        graph.append(tuple(producers))
    return graph


def critical_path(graph: Sequence[tuple[int, ...]]) -> int:
    """Longest chain in the true dependency DAG, in descriptors.

    This is the dependency depth the hardware would have to drain through *if*
    every real dependency could be expressed. Compare it with the depth actually
    emitted to see what the encoding costs.
    """
    depth = [1] * len(graph)
    for j, producers in enumerate(graph):
        for i in producers:
            depth[j] = max(depth[j], depth[i] + 1)
    return max(depth, default=0)


@dataclass
class ChainResult:
    """Outcome of :func:`auto_chain`."""

    descriptors: tuple[Descriptor, ...]
    #: Descriptors that genuinely depend on more than one earlier descriptor.
    multi_producer: tuple[tuple[int, tuple[int, ...]], ...] = ()
    #: Descriptors given a deeper wait than their data dependency required,
    #: because a single wait token could not cover their producer set.
    forced_serialisations: int = 0
    #: Depth of the emitted token chain.
    chain_depth: int = 0
    #: Depth of the true dependency DAG, ignoring the one-token limit.
    true_depth: int = 0
    #: ``"minimal"`` if every wait matched a real dependency, ``"serial"`` if the
    #: block had to be fully linearised to be encodable at all.
    policy: str = "minimal"

    @property
    def depth_penalty(self) -> float:
        """Emitted chain depth divided by the achievable depth."""
        return self.chain_depth / self.true_depth if self.true_depth else 1.0


def auto_chain(descriptors: Sequence[Descriptor]) -> ChainResult:
    """Assign chain/wait tokens from the descriptors' actual data dependencies.

    This is the part of code generation that decides what the hardware may
    overlap, and it is where descriptor v0's **single wait token** costs
    something measurable.

    First attempt: each descriptor waits on the latest producer that
    transitively covers all its other producers. Where no single producer covers
    the set -- which happens as soon as the dependency graph *forks*, e.g. a
    GEMV accumulating into a vector that a different chain of descriptors built
    -- fall back to waiting on the immediately preceding descriptor. If even that
    is not sufficient (it is not, whenever two independent chains merge), the
    whole block is emitted **fully serialised**, which is the only encodable
    schedule, and ``policy`` reports ``"serial"``.

    ``chain_depth`` versus ``true_depth`` is then the price of the encoding.
    """
    descs = list(descriptors)
    graph = dependency_graph(descs)
    multi = tuple((j, p) for j, p in enumerate(graph) if len(p) > 1)
    true_depth = critical_path(graph)

    def emit(waits: Sequence[int | None]) -> tuple[tuple[Descriptor, ...], int]:
        before: list[set[int]] = []
        out = []
        for j, d in enumerate(descs):
            w = waits[j]
            s: set[int] = set()
            if w is not None:
                s.add(w)
                s |= before[w]
            before.append(s)
            out.append(_retag(d, j + 1, 0 if w is None else w + 1))
        return tuple(out), max((len(s) for s in before), default=0) + 1

    # -- attempt 1: minimal waits ------------------------------------------
    before: list[set[int]] = []
    waits: list[int | None] = []
    forced = 0
    encodable = True
    for j, producers in enumerate(graph):
        if not producers:
            w: int | None = None
        else:
            cand = producers[-1]
            if all(p == cand or p in before[cand] for p in producers):
                w = cand
            else:
                w = j - 1
                forced += 1
                if not all(p == w or p in before[w] for p in producers):
                    encodable = False
                    break
        s: set[int] = set()
        if w is not None:
            s.add(w)
            s |= before[w]
        before.append(s)
        waits.append(w)

    if encodable:
        out, depth = emit(waits)
        return ChainResult(
            descriptors=out, multi_producer=multi, forced_serialisations=forced,
            chain_depth=depth, true_depth=true_depth, policy="minimal",
        )

    # -- attempt 2: full serialisation, the only encodable fallback --------
    serial: list[int | None] = [None] + list(range(len(descs) - 1))
    out, depth = emit(serial)
    forced = sum(
        1 for j, producers in enumerate(graph)
        if j > 0 and (not producers or max(producers) != j - 1)
    )
    return ChainResult(
        descriptors=out, multi_producer=multi, forced_serialisations=forced,
        chain_depth=depth, true_depth=true_depth, policy="serial",
    )


def concat(name: str, *programs: Program) -> Program:
    """Concatenate blocks, renumbering tokens so the result is one valid stream.

    Used to hazard-check a whole solve, including loop-carried dependencies
    between one iteration and the next, which a per-block check cannot see.
    Raises if the concatenation needs more than the 255 tokens v0 encodes.
    """
    descs: list[Descriptor] = []
    for p in programs:
        descs.extend(p.descriptors)
    if len(descs) > 255:
        raise FieldOverflow(
            f"{len(descs)} descriptors need more than the 255 chain tokens "
            f"descriptor v0 encodes in 8 bits; tokens must be recycled, which "
            f"means codegen owns a token-liveness analysis"
        )
    return Program(name, auto_chain(descs).descriptors)


@dataclass
class SolverProgram:
    """An iterative solver as the sequencer sees it.

    Parameters
    ----------
    setup:
        Runs when the problem data changes -- factorisations, precomputations.
        Skipped entirely when every buffer it reads is still resident at the same
        generation, which is what makes call-to-call residency worth having.
    iteration:
        The loop body.
    residual:
        Computes the scalars the convergence test reads. Separated from
        ``iteration`` so that a configuration which checks convergence every k-th
        iteration is expressible without editing the body.
    converged:
        The sequencer's convergence test. Takes no arguments and returns bool;
        close it over the scalar buffers it reads.
    max_iterations:
        Hard bound. A real-time system needs one; there is no "run until done".
    check_every:
        Evaluate ``residual`` and ``converged`` every k-th iteration.
    """

    name: str
    setup: Program
    iteration: Program
    residual: Program
    converged: Callable[[], bool]
    max_iterations: int = 200
    check_every: int = 1

    @property
    def programs(self) -> tuple[Program, ...]:
        return (self.setup, self.iteration, self.residual)

    @property
    def uses_only_frozen_v0(self) -> bool:
        return all(p.uses_only_frozen_v0 for p in self.programs)

    @property
    def proposed_opcodes(self) -> tuple[Opcode, ...]:
        seen: list[Opcode] = []
        for p in self.programs:
            for op in p.proposed_opcodes:
                if op not in seen:
                    seen.append(op)
        return tuple(seen)
