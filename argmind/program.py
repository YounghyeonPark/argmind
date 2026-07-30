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

from .descriptor import (
    Descriptor, FROZEN_V0_OPCODES, N_TOKENS, Opcode, TOKEN_MAX,
)
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
    depend on more than one earlier descriptor -- under descriptor v0 these were
    the descriptors codegen had to over-serialise, because v0 had exactly one
    wait token. **v0.1's wait mask expresses them directly** up to
    :data:`argmind.descriptor.N_TOKENS` producers, so a non-empty
    ``multi_producer`` list is now informational rather than a cost.

    ``peak_live_tokens`` is the number of tokens that must be simultaneously live
    for this block, i.e. the token pressure. It is the measurement that sizes the
    3-bit token field: if it exceeds :data:`N_TOKENS` the block still runs
    correctly (token reuse can only over-serialise) but loses overlap, so it is
    reported rather than raised on.
    """

    unordered: list[tuple[int, int, str]] = field(default_factory=list)
    multi_producer: list[tuple[int, tuple[int, ...]]] = field(default_factory=list)
    chain_length: int = 0
    peak_live_tokens: int = 0
    max_merge_width: int = 0

    @property
    def ok(self) -> bool:
        return not self.unordered

    @property
    def token_pressure_ok(self) -> bool:
        return self.peak_live_tokens <= N_TOKENS

    def summary(self) -> str:
        return (
            f"{len(self.unordered)} unordered hazard(s); "
            f"{len(self.multi_producer)} descriptor(s) with >1 producer "
            f"(max merge width {self.max_merge_width}); "
            f"critical chain {self.chain_length}; "
            f"peak live tokens {self.peak_live_tokens}/{N_TOKENS}"
        )


def peak_live_tokens(descriptors: Sequence[Descriptor]) -> int:
    """Maximum number of tokens that must be live at once, under mask semantics.

    A token is live from its producer's issue until the issue of the last
    descriptor that waits on it. This is the quantity the token field has to
    cover, and it is what makes 3 bits defensible where 8 looked necessary: a
    token is a **scoreboard slot, not a name**.
    """
    graph = dependency_graph(descriptors)
    last_use: dict[int, int] = {}
    for j, producers in enumerate(graph):
        for i in producers:
            last_use[i] = j
    peak = 0
    for pos in range(len(descriptors)):
        live = sum(1 for i, lu in last_use.items() if i <= pos < lu)
        peak = max(peak, live)
    return peak


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
    produced: dict[int, int] = {}          # token -> most recent producing index
    waits_on: list[tuple[int, ...]] = []
    for i, d in enumerate(descs):
        deps: list[int] = []
        for t in d.waits_on:
            if t not in produced:
                raise DependencyViolation(
                    f"{program.name}[{i}] ({d}) waits on token {t}, which no "
                    f"earlier descriptor has set. In hardware this hangs."
                )
            deps.append(produced[t])
        waits_on.append(tuple(deps))
        # Every descriptor sets its token on completion. A reused token resolves
        # to its MOST RECENT producer, which can only over-serialise -- the
        # property that makes a 3-bit token field safe.
        produced[d.chain_token] = i

    # Transitive closure of "must complete before", over token edges.
    before: list[set[int]] = []
    for i in range(len(descs)):
        s: set[int] = set()
        for p in waits_on[i]:
            s.add(p)
            s |= before[p]
        before.append(s)

    # Longest path over the emitted token edges, NOT the size of the ancestor
    # set. Those coincide only for a fully serialised block, so conflating them
    # overstates the emitted depth of anything with parallelism -- which is
    # precisely the quantity the v0-vs-v0.1 comparison turns on.
    edge_depth = [1] * len(descs)
    for j in range(len(descs)):
        for p in waits_on[j]:
            edge_depth[j] = max(edge_depth[j], edge_depth[p] + 1)
    report = HazardReport(chain_length=max(edge_depth, default=0))
    acc = [accesses(d) for d in descs]
    widths: list[int] = []
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
        widths.append(len(producers))
        if len(producers) > 1:
            report.multi_producer.append((j, tuple(producers)))
    report.max_merge_width = max(widths, default=0)
    report.peak_live_tokens = peak_live_tokens(descs)

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
    "opcode", "precision", "vec_precision", "a", "b", "c", "m", "k", "n",
    "lda", "ldb", "ldc", "alpha", "transpose", "structure", "accumulate",
    "unit_diag", "label",
)


def _retag(d: Descriptor, chain_token: int, wait_mask: int) -> Descriptor:
    return Descriptor(
        **{f: getattr(d, f) for f in _COPY_FIELDS},
        chain_token=chain_token,
        wait_mask=wait_mask,
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
    #: Descriptors given a deeper wait than their data dependency required.
    #: Zero under v0.1 mask semantics unless token pressure forced reuse.
    forced_serialisations: int = 0
    #: Depth of the emitted token chain.
    chain_depth: int = 0
    #: Depth of the true dependency DAG.
    true_depth: int = 0
    #: ``"mask"`` for v0.1 exact-producer-set encoding, ``"minimal"`` /
    #: ``"serial"`` for the v0 single-token fallbacks.
    policy: str = "mask"
    #: Peak simultaneous token liveness -- the token-field sizing measurement.
    peak_live_tokens: int = 0
    #: Widest producer set the mask had to express.
    max_merge_width: int = 0
    #: Number of distinct tokens actually used.
    tokens_used: int = 0

    @property
    def depth_penalty(self) -> float:
        """Emitted chain depth divided by the achievable depth.

        1.00x under v0.1. Kept because the v0 values (1.42x on the ADMM
        iteration, 3.00x on the residual) are the measurement that motivated the
        change, and because :func:`auto_chain_v0` still reports them.
        """
        return self.chain_depth / self.true_depth if self.true_depth else 1.0


def _token_assignment(
    graph: Sequence[tuple[int, ...]],
) -> tuple[list[int], list[int], int]:
    """Give each descriptor a token, recycling tokens whose last use has passed.

    Returns ``(token_of, extra_wait_tokens, n_forced)``.

    Recycling is what makes a 3-bit token field enough. But the safety argument
    needs one more rule than it first appears, and getting it wrong is a race
    rather than a slowdown -- the concatenation hazard test found exactly that:

        A waiter resolves a token to its **most recent** producer. If token *t*
        was claimed by descriptor 5, reclaimed by descriptor 40, and descriptor
        50 depends on 5, then waiting on *t* waits for 40 -- which says nothing
        about 5, because in-order ISSUE does not imply in-order COMPLETION.

    So reuse is **not** unconditionally safe. It is safe under this rule, which
    is a requirement on both codegen and the RTL scoreboard:

        **A descriptor that reclaims a token must itself wait on that token.**

    Then the reclaiming descriptor cannot start until the previous owner has
    completed, so the previous owner's completion is implied transitively by the
    new owner's, and every later waiter is correctly ordered. The cost is the
    over-serialisation the fallback was always supposed to cost -- and now it is
    the *only* cost, which is what makes "token exhaustion loses overlap, it does
    not produce a wrong answer" a true statement rather than a hopeful one.
    """
    last_use: dict[int, int] = {}
    for j, producers in enumerate(graph):
        for i in producers:
            last_use[i] = j
    token_of: list[int] = [0] * len(graph)
    extra_wait: list[int] = [0] * len(graph)
    free: list[int] = list(range(N_TOKENS))
    busy: dict[int, int] = {}                 # token -> descriptor still needed
    reuse_forced = 0
    for j in range(len(graph)):
        # Retire tokens whose last consumer has already been issued.
        for tok, owner in list(busy.items()):
            if last_use.get(owner, -1) < j:
                del busy[tok]
                free.append(tok)
        if free:
            tok = free.pop(0)
        else:
            # Pressure exceeds the token space. Reclaim the token whose owner is
            # oldest, and make THIS descriptor wait on it so the reclaim is
            # ordered. Costs overlap; preserves correctness.
            tok = min(busy, key=lambda t: busy[t])
            del busy[tok]
            extra_wait[j] = 1 << tok
            reuse_forced += 1
        token_of[j] = tok
        if j in last_use:
            busy[tok] = j
        else:
            # No consumer, so nothing can ever wait on this token: return it
            # immediately. Forgetting this leaks a slot per consumer-less
            # descriptor and manufactures spurious token pressure -- the residual
            # block reported 8 tokens used at a measured peak liveness of 3.
            free.append(tok)
    return token_of, extra_wait, reuse_forced


def auto_chain(descriptors: Sequence[Descriptor]) -> ChainResult:
    """Assign chain tokens and wait masks from the actual data dependencies.

    Under descriptor **v0.1** this is straightforward, which is the point: each
    descriptor's wait mask names exactly its producer set, so the emitted
    dependency depth equals the true DAG depth and no serialisation is forced.
    Compare :func:`auto_chain_v0`, which is kept so the cost of v0's single wait
    token stays measurable rather than becoming folklore.
    """
    descs = list(descriptors)
    graph = dependency_graph(descs)
    multi = tuple((j, p) for j, p in enumerate(graph) if len(p) > 1)
    true_depth = critical_path(graph)
    token_of, extra_wait, reuse_forced = _token_assignment(graph)

    out: list[Descriptor] = []
    depth_of: list[int] = []
    for j, d in enumerate(descs):
        producers = graph[j]
        # Always assert the producer's token bit, even if the token has since
        # been reclaimed. The reclaim rule makes this correct by induction: a
        # reclaiming descriptor waits on the token it takes, so the CURRENT owner
        # of a token transitively covers every past owner of it. A waiter that
        # resolves the token to a later owner is therefore ordered after its real
        # producer -- more conservatively than it asked, which is the whole
        # bargain. (Dropping the bit instead is a race, and the concatenation
        # hazard test is what caught that.)
        mask = extra_wait[j]
        depth = 1
        for i in producers:
            mask |= 1 << token_of[i]
            depth = max(depth, depth_of[i] + 1)
        if extra_wait[j]:
            # A reclaim edge is a real ordering edge and must be counted.
            for i in range(j):
                if (1 << token_of[i]) & extra_wait[j]:
                    depth = max(depth, depth_of[i] + 1)
        depth_of.append(depth)
        out.append(_retag(d, token_of[j], mask))
    depth = max(depth_of, default=0)

    return ChainResult(
        descriptors=tuple(out), multi_producer=multi,
        forced_serialisations=reuse_forced,
        chain_depth=depth, true_depth=true_depth, policy="mask",
        peak_live_tokens=peak_live_tokens(descs),
        max_merge_width=max((len(p) for p in graph), default=0),
        tokens_used=len(set(token_of)),
    )


def auto_chain_v0(descriptors: Sequence[Descriptor]) -> ChainResult:
    """The v0 single-wait-token encoding, retained as the measurement baseline.

    Kept in the shipping code on purpose. The whole justification for the v0.1
    token change is a comparison -- emitted chain depth 17 vs an achievable 12 on
    the ADMM iteration, 15 vs 5 on the residual -- and a comparison whose
    baseline has been deleted stops being checkable. This function is what those
    numbers come from, and a regression test pins them.

    First attempt: each descriptor waits on the latest producer that transitively
    covers all its other producers. Where no single producer covers the set --
    which happens as soon as the dependency graph forks -- fall back to waiting on
    the immediately preceding descriptor; and if that is still insufficient (it is,
    whenever two independent chains merge) emit the block **fully serialised**,
    which is the only encodable schedule.
    """
    descs = list(descriptors)
    graph = dependency_graph(descs)
    multi = tuple((j, p) for j, p in enumerate(graph) if len(p) > 1)
    true_depth = critical_path(graph)

    def emit(waits: Sequence[int | None]) -> tuple[tuple[Descriptor, ...], int]:
        depth_of: list[int] = []
        out = []
        for j, d in enumerate(descs):
            w = waits[j]
            depth_of.append(1 if w is None else depth_of[w] + 1)
            # One token, expressed in the v0.1 field as a single-bit mask, with
            # tokens recycled modulo N_TOKENS so the encoding stays legal.
            tok = (j % TOKEN_MAX) + 1
            wtok = 0 if w is None else (1 << ((w % TOKEN_MAX) + 1))
            out.append(_retag(d, tok, wtok))
        return tuple(out), max(depth_of, default=0)

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
        policy = "minimal"
    else:
        out, depth = emit([None] + list(range(len(descs) - 1)))
        forced = sum(
            1 for j, producers in enumerate(graph)
            if j > 0 and (not producers or max(producers) != j - 1)
        )
        policy = "serial"
    return ChainResult(
        descriptors=out, multi_producer=multi, forced_serialisations=forced,
        chain_depth=depth, true_depth=true_depth, policy=policy,
        peak_live_tokens=peak_live_tokens(descs),
        max_merge_width=max((len(p) for p in graph), default=0),
    )


def concat(name: str, *programs: Program) -> Program:
    """Concatenate blocks, retagging tokens so the result is one valid stream.

    Used to hazard-check a whole solve, including loop-carried dependencies
    between one iteration and the next, which a per-block check cannot see.

    v0 raised here above 255 descriptors, on the theory that a block needed one
    token per descriptor. That was the wrong model of a token and the limit is
    gone: :func:`auto_chain` recycles tokens by liveness, so a block of any length
    is encodable. What a long block can run out of is *overlap*, which
    :attr:`ChainResult.forced_serialisations` reports.
    """
    descs: list[Descriptor] = []
    for p in programs:
        descs.extend(p.descriptors)
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
