"""The residency planner: what stays on the CM, and what crosses the bus per call.

This is derived, not configured. Given the kernel graph and the one declaration
the arrays cannot carry -- which QP data blocks change from call to call -- every
other residency decision follows from the emitted dataflow:

* a tensor whose **first access in execution order is a write** is CM-internal:
  never fetched, never written back, and host memory need not even hold it;
* a tensor read before it is written must be fetched, and is fetched **once** if
  it is structural and **every call** if it is per-instance;
* the setup block is skipped whenever every buffer it reads is still resident at
  the same generation, which for a controller means the Cholesky runs once ever;
* the write-back set is the host-visible tensors, and nothing else -- anything not
  named stays on the CM, which is the point.

The planner's output is then **cross-checked two ways**: against the device's own
independently-written static analysis (:class:`argmind.device.BuiltProgram`
computes its input set from descriptor byte ranges), and against the bytes a real
call actually moves. Predicted-equals-measured is the only version of this claim
worth having; the 46x figure it reproduces was previously hand-arranged.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..memory import Buffer
from .ir import KernelGraph, Source


@dataclass
class ResidencyPlan:
    """What the runtime must fetch, when, and what it may trust as resident."""

    #: Fetched on the first call and trusted resident thereafter.
    fetch_once: tuple[str, ...] = ()
    #: Fetched on every call: the host rewrote them.
    fetch_per_call: tuple[str, ...] = ()
    #: Never fetched. Written before read, or produced on the CM.
    never_fetched: tuple[str, ...] = ()
    #: Written back to host memory after the call.
    written_back: tuple[str, ...] = ()
    first_call_bytes_in: int = 0
    steady_bytes_in: int = 0
    steady_bytes_out: int = 0
    #: Bytes that would cross the bus per call with no residency at all, i.e. if
    #: every buffer the program reads were re-fetched. The honest denominator.
    no_residency_bytes_in: int = 0
    setup_runs_when: str = ""
    #: Tensors whose declared source contradicts the emitted dataflow. Non-empty
    #: means the generator is wrong about its own program; it is reported rather
    #: than silently trusted.
    contradictions: tuple[str, ...] = ()
    #: The declarations that could not be derived and had to be supplied.
    hints_required: tuple[str, ...] = ()
    by_tensor: dict[str, str] = field(default_factory=dict)

    @property
    def reduction(self) -> float:
        """Steady-state transfer reduction: no-residency bytes / steady bytes."""
        if not self.steady_bytes_in:
            return float("inf")
        return self.no_residency_bytes_in / self.steady_bytes_in

    @property
    def ok(self) -> bool:
        return not self.contradictions

    def report(self) -> str:
        lines = [
            "residency plan (derived from the emitted dataflow):",
            f"  fetch once    : {', '.join(self.fetch_once) or '-'}",
            f"  fetch per call: {', '.join(self.fetch_per_call) or '-'}",
            f"  never fetched : {len(self.never_fetched)} tensors "
            f"({', '.join(self.never_fetched) or '-'})",
            f"  written back  : {', '.join(self.written_back) or '-'}",
            f"  first call in {self.first_call_bytes_in} B; steady state "
            f"{self.steady_bytes_in} B in / {self.steady_bytes_out} B out",
            f"  against {self.no_residency_bytes_in} B per call with no "
            f"residency -> {self.reduction:.1f}x",
            f"  setup block runs when: {self.setup_runs_when}",
        ]
        if self.hints_required:
            lines.append(f"  declarations that are NOT derivable: "
                         f"{', '.join(self.hints_required)}")
        if self.contradictions:
            lines.append(f"  CONTRADICTIONS: {'; '.join(self.contradictions)}")
        return "\n".join(lines)


def first_access(graph: KernelGraph) -> dict[str, str]:
    """For each tensor, whether it is first read or first written.

    Execution order is setup (possibly skipped), then iteration, then residual.
    A tensor first *written* by the setup block is still never fetched -- that is
    the factor -- so the setup block is included, and the case where setup is
    skipped is exactly the case where its outputs are already resident.
    """
    seen: dict[str, str] = {}
    for k in graph.all_kernels:
        for t in k.reads:
            seen.setdefault(t.name, "read")
        for t in k.writes:
            seen.setdefault(t.name, "write")
    return seen


def plan_residency(
    graph: KernelGraph,
    buffers: dict[str, Buffer],
    *,
    warm_start: bool = True,
) -> ResidencyPlan:
    """Derive the residency plan from the graph. No hints beyond ``Source``."""
    access = first_access(graph)
    by_name = {t.name: t for t in graph.tensors}

    fetch_once: list[str] = []
    fetch_per_call: list[str] = []
    never: list[str] = []
    written_back: list[str] = []
    contradictions: list[str] = []
    by_tensor: dict[str, str] = {}

    for t in graph.tensors:
        first = access.get(t.name)
        if first is None:
            by_tensor[t.name] = "unused"
            never.append(t.name)
            continue
        if first == "write":
            # Written before read: nothing in host memory can be needed.
            never.append(t.name)
            by_tensor[t.name] = "cm-internal (written before read)"
            if t.source in (Source.STRUCTURAL, Source.INSTANCE):
                contradictions.append(
                    f"{t.name} is declared {t.source.name} -- host-supplied -- but "
                    f"the emitted program writes it before reading it, so the "
                    f"host's value is dead. Either the declaration or the program "
                    f"is wrong.")
            continue
        # Read before written: must be fetched at least once.
        if t.source is Source.DERIVED:
            contradictions.append(
                f"{t.name} is declared DERIVED (produced on the CM) but is read "
                f"before any kernel writes it. Fetching it would overwrite the "
                f"resident copy with whatever host memory holds -- zeros.")
            never.append(t.name)
            by_tensor[t.name] = "DERIVED but read first: contradiction"
            continue
        if t.source is Source.INSTANCE:
            fetch_per_call.append(t.name)
            by_tensor[t.name] = "fetched every call (host rewrites it)"
        elif t.source is Source.ITERATE and not warm_start:
            fetch_per_call.append(t.name)
            by_tensor[t.name] = "fetched every call (cold start)"
        else:
            fetch_once.append(t.name)
            by_tensor[t.name] = (
                "fetched once, then trusted resident by generation tag")
        if t.host_visible:
            written_back.append(t.name)

    def nbytes(names) -> int:
        return sum(buffers[nm].nbytes for nm in names)

    setup_block = graph.blocks.get("setup")
    setup_reads = sorted(
        {t.name for k in (setup_block.kernels if setup_block else []) for t in k.reads}
    )
    setup_inputs = [nm for nm in setup_reads if by_name[nm].source.host_writes]
    varying_setup_inputs = [nm for nm in setup_inputs
                            if by_name[nm].source is Source.INSTANCE]
    if varying_setup_inputs:
        setup_when = (
            f"EVERY CALL -- {', '.join(varying_setup_inputs)} change per call, so "
            f"the factorisation cannot be hoisted out of the control loop")
    else:
        setup_when = (
            f"only when {', '.join(setup_inputs) or 'nothing'} change generation "
            f"-- once for the life of the plant model")

    # The honest no-residency denominator: every buffer the program reads before
    # writing, re-fetched every call. It is not the whole scratchpad, because the
    # temporaries would never be fetched even by a runtime with no residency.
    all_fetchable = list(fetch_once) + list(fetch_per_call)

    return ResidencyPlan(
        fetch_once=tuple(fetch_once),
        fetch_per_call=tuple(fetch_per_call),
        never_fetched=tuple(never),
        written_back=tuple(written_back),
        first_call_bytes_in=nbytes(all_fetchable),
        steady_bytes_in=nbytes(fetch_per_call),
        steady_bytes_out=nbytes(written_back),
        no_residency_bytes_in=nbytes(all_fetchable),
        setup_runs_when=setup_when,
        contradictions=tuple(contradictions),
        hints_required=(
            "which QP data blocks change from call to call (QPProblem.varies); "
            "everything else in this plan is derived from the emitted dataflow",
        ),
        by_tensor=by_tensor,
    )


def cross_check_against_device(
    plan: ResidencyPlan, built, prefix: str = "",
) -> tuple[str, ...]:
    """Compare the planner's fetch set with the device's own static analysis.

    :class:`argmind.device.BuiltProgram` computes, from descriptor byte ranges
    alone, which buffers a block reads before writing. That is the same question
    the planner answers from the tensor graph, by a different route, so the two
    must agree. Returns a tuple of disagreements -- empty is the passing result.
    """
    # Mirror Device.solve exactly: a buffer the SETUP block produces is never
    # fetched (host memory does not hold a factorisation), but a buffer the
    # iteration block writes is still an input if it is read before being written
    # -- that is what warm starting is.
    produced_by_setup = {buf.name for buf in built.setup.outputs}
    device_reads: set[str] = set()
    for block in built.blocks:
        for buf in block.inputs:
            if buf.name not in produced_by_setup:
                device_reads.add(buf.name)
    planned = {f"{prefix}{nm}" for nm in
               tuple(plan.fetch_once) + tuple(plan.fetch_per_call)}
    problems = []
    for nm in sorted(device_reads - planned):
        problems.append(
            f"{nm}: the device's descriptor-range analysis says it is read before "
            f"it is written, but the residency plan does not fetch it")
    for nm in sorted(planned - device_reads):
        problems.append(
            f"{nm}: the residency plan fetches it, but no descriptor reads it "
            f"before writing it -- a fetch with no consumer")
    return tuple(problems)
