# SPDX-License-Identifier: Apache-2.0
"""The codegen contract, and the report that checks it.

Phase 0 accumulated eight obligations on generated code. Every one of them exists
because breaking it produced a *silent* failure -- a stalled solve, a wrong
answer, a missed deadline, or a residency architecture quietly defeated by a
memcpy. None of them is visible in a descriptor listing, and a contract nothing
checks is decoration. So they are enforced here, on the emitted program, and the
report is part of the generator's output rather than a document beside it.

Two of the eight are enforced by **refusal at generation time** (C1 and C7): the
generator will not emit the program at all. The other six are measured on the
emitted program and reported, because their answer is a number rather than a
yes/no -- and in one case (C3) the number is worse than the prose suggested,
which is exactly why it is measured.

+----+-------------------------------------------+------------------------------+
| id | rule                                      | how it is enforced           |
+====+===========================================+==============================+
| C1 | rho exact against its own reciprocal      | refuse at generation         |
| C2 | A'y accumulated row-wise, never columns   | emitted shape + bank count   |
| C3 | >= 8 datapath cycles/descriptor, and no   | measured distribution        |
|    | sub-threshold run longer than the queue   |                              |
| C4 | templates armed once per solve            | template count + emit cost   |
| C5 | static allocation, no runtime reallocation | plan determinism             |
| C6 | in-place kernels, no staging copies        | copy count after elision     |
| C7 | narrow operands opt-in, solve kept wide   | refuse at generation         |
| C8 | a descriptor reclaiming a token waits on it| scoreboard replay            |
+----+-------------------------------------------+------------------------------+
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from ..descriptor import N_TOKENS, Descriptor, Opcode
from ..precision import Format
from ..program import Program, dependency_graph
from .cost import ColumnWalkPenalty, DatapathModel, column_walk_penalty
from .ir import KernelGraph
from .problem import CodegenOptions, ContractViolation

#: Datapath cycles per descriptor the sequencer contract requires **on average**.
#: It is ``2 * seq_setup_cycles`` at the recommended 4-cycle descriptor
#: generator: the sequencer must be able to prepare the next descriptor in half
#: the time the datapath spends on the current one.
CYCLES_PER_DESCRIPTOR_BAR = 8.0

#: Descriptor queue depth. A run of sub-threshold descriptors shorter than this
#: is absorbed by run-ahead; a longer run is not, and that is the clause the
#: measured ADMM residual block sits closest to. Confirmed as a floor by
#: measurement, not assumed: a 4-deep queue would not absorb the measured run.
QUEUE_DEPTH = 8

#: Instructions to arm one hardware-descriptor-generator template
#: (``perf_model/seq_emit_cost.py``, counted from RV32 rather than estimated).
ARMING_INSTRUCTIONS = 22

#: Templates the CM-100 area model instantiates
#: (``perf_model/sequencer.py::N_DESC_TEMPLATES``). Reported against what the
#: generated program needs, because "arm once per solve" is only available if the
#: generator can hold every template the solve uses.
TEMPLATES_IN_AREA_MODEL = 4

#: Gate cost of one more armed template: the descriptor's own width in flops.
#: ``DESC_BITS * GATES_PER_FLOP_BIT`` at ``perf_model/sequencer.py``'s basis.
GATES_PER_TEMPLATE = 192 * 6
GATE_UM2 = 0.26


@dataclass
class Finding:
    rule: str
    name: str
    status: str            # "pass" | "fail" | "info"
    detail: str
    numbers: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status != "fail"

    def __str__(self) -> str:
        mark = {"pass": "PASS", "fail": "FAIL", "info": "INFO"}[self.status]
        return f"[{self.rule} {mark}] {self.name}\n      {self.detail}"


@dataclass
class ComplianceReport:
    """The generator's own audit of its output."""

    target: str = ""
    findings: list[Finding] = field(default_factory=list)

    def add(self, *f: Finding) -> None:
        self.findings.extend(f)

    @property
    def ok(self) -> bool:
        return all(f.ok for f in self.findings)

    @property
    def failures(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.status == "fail")

    def by_rule(self, rule: str) -> Finding:
        for f in self.findings:
            if f.rule == rule:
                return f
        raise KeyError(rule)

    def text(self) -> str:
        head = (f"codegen contract compliance -- target {self.target}: "
                f"{'ALL RULES PASS' if self.ok else f'{len(self.failures)} FAILURE(S)'}")
        return "\n".join([head] + [f"  {f}" for f in self.findings])


@dataclass
class ContractContext:
    """Everything the checks need, and nothing they do not."""

    options: CodegenOptions
    graph: KernelGraph
    blocks: dict[str, Program]
    block_order: tuple[str, ...]
    model: DatapathModel
    plan_signature: str
    plan_used_bytes: int
    plan_capacity_bytes: int
    #: The whole solve as one hazard-checked stream, including the loop-carried
    #: edges a per-block check cannot see.
    unrolled: Program


# ---------------------------------------------------------------------------
# C1 -- rho must be exact against its own reciprocal
# ---------------------------------------------------------------------------

def check_rho(opt: CodegenOptions) -> Finding:
    """Refuse a rho whose fp32 reciprocal is inexact **on the w7 path**.

    The failure this prevents is not an inaccuracy, it is a stall. At an ADMM
    fixed point with an active constraint ``y = rho*(1/rho)*y``, so if
    ``fl32(rho)*fl32(1/rho) != 1`` the only solution is ``y = 0`` and the dual
    variable cannot settle: rho = 0.1 runs to 1500 iterations, rho = 0.125
    converges in 34. It is invisible when the optimum is interior, which is
    exactly the configuration a nominal MPC benchmark has.

    Descriptor v0.1 ruling R5 re-diagnosed the root cause as *an fp32 immediate
    multiplying fp64 data*, not as rho, so there are two compliant answers:
    a power-of-two rho in w7, or rho in scratchpad at the buffer's own precision
    read by a zero-stride VMUL. The generator's default is the second.
    """
    err = opt.rho_reciprocal_error
    if opt.rho_in_scratchpad:
        return Finding(
            "C1", "rho is exact against its reciprocal", "pass",
            f"rho = {opt.rho!r} travels as a scratchpad scalar at the buffer's "
            f"own precision, read by a zero-stride VMUL (ruling R5). The w7 fp32 "
            f"immediate would have carried a reciprocal mismatch of "
            f"{err:.3e}; at fp64 storage it is ~1e-16. Power-of-two rho remains "
            f"recommended (it makes the mismatch exactly zero) but is not "
            f"load-bearing on this path.",
            {"rho": opt.rho, "w7_reciprocal_error": err,
             "encoding": "scratchpad-vmul",
             "power_of_two": opt.rho_is_power_of_two})
    if err != 0.0:
        raise ContractViolation(
            f"codegen contract C1: rho = {opt.rho!r} is not exactly invertible in "
            f"the fp32 descriptor immediate (w7): fl32(rho)*fl32(1/rho) - 1 = "
            f"{err:.3e}. The ADMM fixed point requires those two constants to be "
            f"exact reciprocals -- with an active constraint y = rho*(1/rho)*y, so "
            f"a mismatch forces y = 0 and the solve STALLS at a primal-residual "
            f"floor of about {abs(err):.1e} times the dual magnitude instead of "
            f"converging (measured: rho=0.1 runs to 1500 iterations, rho=0.125 "
            f"converges in 34). This is refused, not warned about, because the "
            f"failure mode is a quietly unconverged answer in a control loop. Fix "
            f"it either way: choose a power of two (0.0625, 0.125, 0.25, 0.5, ...) "
            f"or set rho_in_scratchpad=True, which is ruling R5's answer and the "
            f"generator's default.")
    return Finding(
        "C1", "rho is exact against its reciprocal", "pass",
        f"rho = {opt.rho!r} is a power of two, so the fp32 w7 immediates for rho "
        f"and 1/rho are exact reciprocals (error exactly 0) and the dual fixed "
        f"point is attainable.",
        {"rho": opt.rho, "w7_reciprocal_error": 0.0, "encoding": "w7-immediate",
         "power_of_two": True})


# ---------------------------------------------------------------------------
# C2 -- A'y accumulated row-wise
# ---------------------------------------------------------------------------

def check_transposed_gemv(ctx: ContractContext) -> Finding:
    """Every transposed matrix-vector product must be one row-major descriptor.

    A transposed GEMV can be lowered three ways and only one is acceptable:

    * **as a column walk** over the row-major matrix -- one dot product per
      output element, striding ``lda`` between successive reads. The stride is a
      whole number of bank words, so the walk visits a *subgroup* of the banks and
      delivers a fraction of peak bandwidth. Quantified below.
    * **as a materialised transpose** -- correct, full bandwidth, and it costs a
      second copy of the largest matrix in the program plus a transpose pass. On
      QUAD-MPC that is 100 KiB at fp32, which does not fit anywhere.
    * **row-wise accumulation** -- walk the matrix in its natural row order,
      scaling each row by one element of the input and accumulating into a
      full-length accumulator held in the vector register file. Contiguous reads,
      full bandwidth, and the price is one accumulator vector.

    The generator emits the third, so what the check verifies is that no
    descriptor stream is expressing one of the first two: exactly one descriptor
    per transposed product, reading the same row-major buffer the untransposed
    product reads, writing a unit-stride destination. It then reports the
    accumulator depth that requires, because that is a hardware requirement the
    architect has to rule on.
    """
    penalties: list[tuple[str, ColumnWalkPenalty]] = []
    problems: list[str] = []
    max_acc = 0
    matrices_read: dict[int, set[bool]] = {}
    for name in ctx.block_order:
        for i, d in enumerate(ctx.blocks[name].descriptors):
            if d.opcode is not Opcode.GEMV:
                continue
            matrices_read.setdefault(d.a, set()).add(bool(d.transpose.a))
            if not d.transpose.a:
                continue
            p = column_walk_penalty(d, ctx.model)
            penalties.append((f"{name}[{i}] {d.label}", p))
            max_acc = max(max_acc, p.accumulator_elements)
            if d.ldc != 1:
                problems.append(
                    f"{name}[{i}]: transposed GEMV writes with ldc = {d.ldc}; a "
                    f"row-wise accumulation writes one contiguous accumulator")
            if d.lda < d.k:
                problems.append(
                    f"{name}[{i}]: lda = {d.lda} < k = {d.k}, so the operand is "
                    f"not row-major -- this descriptor is a column walk")
    # A materialised transpose would show up as a second matrix buffer holding the
    # same data. The generator never allocates one; check that no two distinct
    # matrix buffers of transposed shape exist.
    shapes: dict[tuple[int, int, Format], list[str]] = {}
    for t in ctx.graph.tensors:
        if len(t.shape) == 2:
            shapes.setdefault((t.shape[0], t.shape[1], t.fmt), []).append(t.name)
    for (r, c, f), names in shapes.items():
        mirror = shapes.get((c, r, f))
        if mirror and r != c:
            problems.append(
                f"tensors {names} and {mirror} are transposes of one another in "
                f"shape -- a materialised transpose defeats the point of the "
                f"transpose bit")

    if problems:
        return Finding("C2", "A'y is accumulated row-wise", "fail",
                       "; ".join(problems), {"violations": len(problems)})
    if not penalties:
        return Finding("C2", "A'y is accumulated row-wise", "info",
                       "the program contains no transposed matrix-vector product",
                       {"transposed_gemvs": 0})
    worst = min(penalties, key=lambda kv: kv[1].delivered_fraction)[1]
    shared = sum(1 for v in matrices_read.values() if v == {True, False})
    return Finding(
        "C2", "A'y is accumulated row-wise", "pass",
        f"{len(penalties)} transposed GEMV(s), each one descriptor over the "
        f"row-major operand ({shared} matrix address read both transposed and "
        f"untransposed, i.e. one stored copy). The column-walk alternative would "
        f"stride {worst.stride_bytes} B = "
        f"{worst.stride_bytes // ctx.model.bank_word_bytes} bank words, reaching "
        f"{worst.banks_reached} of {ctx.model.banks} banks = "
        f"{100 * worst.delivered_fraction:.1f}% of peak scratchpad bandwidth. "
        f"HARDWARE REQUIREMENT: a {max_acc}-element accumulator in the vector "
        f"register file, live for the duration of one descriptor.",
        {"transposed_gemvs": len(penalties),
         "accumulator_elements": max_acc,
         "column_walk_banks": worst.banks_reached,
         "banks": ctx.model.banks,
         "column_walk_delivered_fraction": worst.delivered_fraction,
         "shared_matrix_addresses": shared})


# ---------------------------------------------------------------------------
# C3 -- cycles per descriptor
# ---------------------------------------------------------------------------

@dataclass
class RunOfShortDescriptors:
    block: str
    start: int
    length: int
    mean_cycles: float
    labels: tuple[str, ...]


def _short_runs(block: str, descriptors: Sequence[Descriptor],
                model: DatapathModel) -> list[RunOfShortDescriptors]:
    runs: list[RunOfShortDescriptors] = []
    cur: list[int] = []
    cycles = [model.cycles(d) for d in descriptors]
    for i, c in enumerate(cycles + [float("inf")]):
        if c < CYCLES_PER_DESCRIPTOR_BAR:
            cur.append(i)
            continue
        if cur:
            runs.append(RunOfShortDescriptors(
                block=block, start=cur[0], length=len(cur),
                mean_cycles=sum(cycles[j] for j in cur) / len(cur),
                labels=tuple(descriptors[j].label or descriptors[j].opcode.name
                             for j in cur)))
            cur = []
    return runs


def check_cycles_per_descriptor(ctx: ContractContext) -> Finding:
    """Measure the emitted program against both clauses of the sequencer rule.

    The average clause is not where this workload is thin and never was. The
    queue-depth clause is: a run of sub-threshold descriptors longer than the
    queue cannot borrow slack from a long neighbour, and the ADMM convergence
    test contains exactly such a run -- the n-length reductions and the
    dual-residual assembly. It is reported per block, with the run listed, because
    "the average is fine" is the answer that hides it.
    """
    per_block: dict[str, dict] = {}
    all_runs: list[RunOfShortDescriptors] = []
    total_cycles = total_desc = 0.0
    for name in ctx.block_order:
        descs = ctx.blocks[name].descriptors
        cyc = [ctx.model.cycles(d) for d in descs]
        runs = _short_runs(name, descs, ctx.model)
        all_runs.extend(runs)
        per_block[name] = {
            "descriptors": len(descs),
            "cycles": sum(cyc),
            "mean": sum(cyc) / len(cyc) if cyc else 0.0,
            "min": min(cyc) if cyc else 0.0,
            "below_bar": sum(1 for c in cyc if c < CYCLES_PER_DESCRIPTOR_BAR),
            "longest_run": max((r.length for r in runs), default=0),
        }
        if name != "setup":
            total_cycles += sum(cyc)
            total_desc += len(descs)

    mean = total_cycles / total_desc if total_desc else float("inf")
    worst_run = max(all_runs, key=lambda r: r.length, default=None)
    worst_len = worst_run.length if worst_run else 0
    status = "pass"
    notes = []
    if mean < CYCLES_PER_DESCRIPTOR_BAR:
        status = "fail"
        notes.append(
            f"the checked-iteration average is {mean:.1f} cycles/descriptor, "
            f"below the {CYCLES_PER_DESCRIPTOR_BAR:.0f}-cycle bar: this program is "
            f"sequencer-issue-bound and codegen must fuse descriptors")
    if worst_len > QUEUE_DEPTH:
        status = "fail"
        notes.append(
            f"a run of {worst_len} consecutive sub-threshold descriptors in "
            f"{worst_run.block} exceeds the {QUEUE_DEPTH}-deep queue, so run-ahead "
            f"cannot absorb it")
    detail = (
        f"checked iteration = {int(total_desc)} descriptors, "
        f"{total_cycles:.0f} datapath cycles = {mean:.1f} cycles/descriptor "
        f"({mean / CYCLES_PER_DESCRIPTOR_BAR:.1f}x the bar). "
        + " | ".join(
            f"{nm}: {v['descriptors']} desc, mean {v['mean']:.1f}, min "
            f"{v['min']:.1f}, {v['below_bar']} below bar, longest sub-threshold "
            f"run {v['longest_run']}"
            for nm, v in per_block.items())
    )
    if worst_run is not None:
        detail += (
            f". Thinnest point: {worst_run.block}[{worst_run.start}..]"
            f"{worst_run.start + worst_run.length - 1}, {worst_run.length} "
            f"descriptors averaging {worst_run.mean_cycles:.1f} cycles "
            f"({', '.join(worst_run.labels)}); "
            f"{'ABSORBED' if worst_run.length <= QUEUE_DEPTH else 'NOT ABSORBED'} "
            f"by a {QUEUE_DEPTH}-deep queue with "
            f"{QUEUE_DEPTH - worst_run.length} slot(s) of margin.")
    if notes:
        detail += " " + " ".join(notes)
    return Finding(
        "C3", ">= 8 datapath cycles per descriptor, bursts <= queue depth",
        status, detail,
        {"mean_cycles_per_descriptor": mean,
         "bar": CYCLES_PER_DESCRIPTOR_BAR,
         "margin_x": mean / CYCLES_PER_DESCRIPTOR_BAR,
         "longest_short_run": worst_len,
         "queue_depth": QUEUE_DEPTH,
         "per_block": per_block})


# ---------------------------------------------------------------------------
# C4 -- templates armed once per solve
# ---------------------------------------------------------------------------

def check_templates(ctx: ContractContext, iterations: int) -> Finding:
    """Count generator templates and price both arming policies.

    Arming costs ~22 RV32 instructions **per template**, so the emit cost depends
    on descriptors-per-template rather than on descriptor count. Hoisting the
    arming out of the iteration loop is legal precisely because allocation is
    static: every iteration issues descriptors with identical opcodes, dimensions,
    strides, formats *and addresses*, so the only thing that changes between
    iterations is the data at those addresses.
    """
    shapes: dict[tuple, int] = {}
    no_alpha: set[tuple] = set()
    per_block: dict[str, int] = {}
    for name in ctx.block_order:
        block = ctx.graph.blocks[name]
        keys = {k.shape_key for k in block.kernels}
        per_block[name] = len(keys)
        for k in block.kernels:
            shapes[k.shape_key] = shapes.get(k.shape_key, 0) + 1
            no_alpha.add(k.shape_key_without_alpha)
    templates = len(shapes)
    loop_desc = (len(ctx.blocks["iteration"].descriptors)
                 + len(ctx.blocks["residual"].descriptors))
    loop_templates = per_block["iteration"] + per_block["residual"]

    per_iter_cost = ARMING_INSTRUCTIONS * loop_templates / max(loop_desc, 1)
    solve_desc = loop_desc * max(iterations, 1) + len(
        ctx.blocks["setup"].descriptors)
    per_solve_cost = ARMING_INSTRUCTIONS * templates / max(solve_desc, 1)

    extra = max(0, templates - TEMPLATES_IN_AREA_MODEL)
    extra_mm2 = extra * GATES_PER_TEMPLATE * GATE_UM2 / 1e6

    status = "pass"
    note = ""
    if per_iter_cost < CYCLES_PER_DESCRIPTOR_BAR and per_solve_cost < 1.0:
        note = ("per-iteration arming would already be inside the bar here, so "
                "hoisting is an optimisation rather than a requirement on this "
                "program")
    if per_solve_cost > CYCLES_PER_DESCRIPTOR_BAR:
        status = "fail"
        note = ("even armed once per solve the emit cost exceeds the bar: the "
                "program has more distinct shapes than descriptors to spread them "
                "over")
    detail = (
        f"{templates} distinct templates over the whole solve "
        f"({', '.join(f'{k}: {v}' for k, v in per_block.items())}); the loop "
        f"(iteration + residual) issues {loop_desc} descriptors from "
        f"{loop_templates} templates. Arming at {ARMING_INSTRUCTIONS} "
        f"instructions/template costs {per_iter_cost:.2f} cycles/descriptor if "
        f"done PER ITERATION and {per_solve_cost:.3f} if done ONCE PER SOLVE "
        f"({iterations} iterations, {solve_desc} descriptors) -- a "
        f"{per_iter_cost / max(per_solve_cost, 1e-9):.0f}x difference, and the "
        f"hoist is legal because every iteration's descriptors are bit-identical. "
        f"HARDWARE REQUIREMENT: the descriptor generator must hold {templates} "
        f"armed templates; perf_model/sequencer.py instantiates "
        f"{TEMPLATES_IN_AREA_MODEL}. {extra} more costs "
        f"{extra_mm2 * 1e6:.0f} um2 = {extra_mm2:.4f} mm2 at "
        f"{GATES_PER_TEMPLATE} gates/template. Templates that differ ONLY in the "
        f"w7 alpha immediate could share a slot and re-arm one word: that would "
        f"take {templates} -> {len(no_alpha)}, which is not obviously worth the "
        f"control logic at this area price.")
    if note:
        detail += " " + note
    return Finding(
        "C4", "templates armed once per solve, not per iteration", status, detail,
        {"templates": templates, "loop_templates": loop_templates,
         "templates_ignoring_alpha": len(no_alpha),
         "loop_descriptors": loop_desc,
         "cycles_per_desc_per_iteration_arming": per_iter_cost,
         "cycles_per_desc_per_solve_arming": per_solve_cost,
         "templates_in_area_model": TEMPLATES_IN_AREA_MODEL,
         "extra_template_mm2": extra_mm2,
         "per_block": per_block})


# ---------------------------------------------------------------------------
# C5 -- static allocation
# ---------------------------------------------------------------------------

def check_static_allocation(ctx: ContractContext, second_signature: str) -> Finding:
    """The address map must be a pure function of the program.

    Checked by generating twice and comparing a signature over every allocation's
    (name, offset, size, format). Anything that made allocation depend on call
    order, on data, or on a hash seed would show up here, and there would be no
    determinism claim left to make.
    """
    if ctx.plan_signature != second_signature:
        return Finding(
            "C5", "static allocation, no runtime reallocation", "fail",
            "two builds of the same program produced different address maps, so "
            "the allocation is not a pure function of the program and the "
            "determinism claim does not hold",
            {"signature_a": ctx.plan_signature, "signature_b": second_signature})
    pct = 100.0 * ctx.plan_used_bytes / max(ctx.plan_capacity_bytes, 1)
    return Finding(
        "C5", "static allocation, no runtime reallocation", "pass",
        f"the address map is fixed at generation time and identical across two "
        f"independent builds (signature {ctx.plan_signature[:16]}...). Working "
        f"set {ctx.plan_used_bytes} B = {ctx.plan_used_bytes / 1024:.1f} KiB of "
        f"{ctx.plan_capacity_bytes / 1024:.0f} KiB ({pct:.1f}%). No allocator runs "
        f"in the solver path, so there is no allocation-dependent latency.",
        {"signature": ctx.plan_signature, "used_bytes": ctx.plan_used_bytes,
         "capacity_bytes": ctx.plan_capacity_bytes, "occupancy_pct": pct})


# ---------------------------------------------------------------------------
# C6 -- in-place kernels
# ---------------------------------------------------------------------------

def check_in_place(ctx: ContractContext, elided: Sequence[tuple[str, str, int]],
                   elision_enabled: bool) -> Finding:
    """No staging copies, and persistent buffers updated at their own addresses.

    The rule bites in two places. Obviously, no descriptor may be a pure copy of
    one buffer into another. Less obviously, the *iterates* must be updated in
    place: ``x = alpha*xt + (1-alpha)*x`` is emitted as a scale of x followed by
    an accumulate into x, both writing x's own address, so warm starting costs
    nothing and the generation tag is what tells the runtime the CM's copy is
    newer than the host's.
    """
    copies = [(nm, i, k) for nm in ctx.block_order
              for i, k in enumerate(ctx.graph.blocks[nm].kernels)
              if k.is_pure_copy]
    in_place_writes = 0
    for nm in ctx.block_order:
        for k in ctx.graph.blocks[nm].kernels:
            if k.writes and any(t in k.reads for t in k.writes):
                in_place_writes += 1
    saved = sum(b for _, _, b in elided)
    if copies and elision_enabled:
        return Finding(
            "C6", "in-place kernels, no staging copies", "fail",
            f"{len(copies)} pure copy descriptor(s) survived elision: "
            + "; ".join(f"{nm}[{i}] {k.label}" for nm, i, k in copies),
            {"copies": len(copies)})
    status = "pass" if elision_enabled else "info"
    detail = (
        f"{in_place_writes} descriptors update a buffer at its own address "
        f"(read-modify-write), and the generation tag is bumped by the runtime on "
        f"every CM write so the resident copy stays authoritative. ")
    if elision_enabled:
        detail += (
            f"{len(elided)} staging copy/copies eliminated by the elision pass, "
            f"reclaiming {saved} B and {len(elided)} descriptors per checked "
            f"iteration: "
            + "; ".join(f"{d} onto {s}" for d, s, _ in elided) + ". "
            "Both were present in the hand-written program, which is the argument "
            "for enforcing this mechanically rather than by review.")
    else:
        detail += (
            f"elision is DISABLED (reference-equivalent mode), so {len(copies)} "
            f"staging copy/copies remain: "
            + "; ".join(f"{nm}[{i}] {k.label}" for nm, i, k in copies)
            + ". This mode exists to reproduce the hand-written program "
              "descriptor for descriptor and is not a shipping configuration.")
    return Finding(
        "C6", "in-place kernels, no staging copies", status, detail,
        {"surviving_copies": len(copies), "elided": len(elided),
         "bytes_reclaimed": saved, "in_place_writes": in_place_writes})


# ---------------------------------------------------------------------------
# C7 -- narrow operands are opt-in, and the solve stays wide
# ---------------------------------------------------------------------------

def check_precision_policy(opt: CodegenOptions) -> Finding:
    """Refuse narrow bulk operands unless asked for, and never narrow the solve.

    fp16 operands are worth 1.16-1.49x latency and 1.51-1.79x energy on the ADMM
    rows at ~2400x the operand error -- about 0.2% of a commanded control action.
    That is a trade a customer makes, not one a compiler makes for them. And the
    measured result that makes the mode cheap is that keeping the triangular solve
    at fp32 costs *nothing*: fp16/fp32-solve and fp16/fp64-solve score
    identically, so the operand width is what limits the scheme.

    fp16 and bf16 are **not** interchangeable here: same width, 10x the error on
    this path, because mantissa bits dominate for these matrices -- the opposite
    of the ML tradeoff. The MACIF contract refuses a macro that declares one and
    is handed the other, and this check refuses a policy that assumes they are the
    same thing.
    """
    pol = opt.precision
    if pol.uses_narrow_operands and not opt.allow_narrow_operands:
        raise ContractViolation(
            f"codegen contract C7: precision policy {pol.name!r} asks for "
            f"{pol.bulk} bulk operands, which is an OPT-IN mode and not a default. "
            f"It is worth 1.16-1.49x latency and 1.51-1.79x energy on ADMM at "
            f"about 0.2% of a commanded control action (operand error 8e-7 -> "
            f"1.9e-3, measured). Pass allow_narrow_operands=True to accept that "
            f"trade explicitly. Note also that fp16 and bf16 are NOT "
            f"interchangeable at this width: bf16 is 10x worse on the solver path "
            f"because mantissa bits dominate, which is the opposite of the ML "
            f"tradeoff.")
    if pol.uses_narrow_operands and pol.solve.itemsize <= 2:
        raise ContractViolation(
            f"codegen contract C7: bulk operands at {pol.bulk} with the triangular "
            f"solve also at {pol.solve}. The measured numerics put the solve ~3x "
            f"more error-sensitive than the operands, and keeping it at fp32 costs "
            f"NOTHING -- fp16/fp32-solve and fp16/fp64-solve score identically. "
            f"Narrowing the solve gives up accuracy for no gain.")
    if not pol.uses_narrow_operands:
        return Finding(
            "C7", "narrow operands opt-in, solve kept wide", "pass",
            f"policy {pol.name!r}: bulk {pol.bulk}, solve {pol.solve}, vectors "
            f"{pol.vector}. No narrow operands, so nothing to opt in to.",
            {"bulk": str(pol.bulk), "solve": str(pol.solve),
             "vector": str(pol.vector), "opt_in": False})
    return Finding(
        "C7", "narrow operands opt-in, solve kept wide", "pass",
        f"policy {pol.name!r} accepted with an explicit opt-in: bulk operands at "
        f"{pol.bulk}, triangular solve at {pol.solve}, vectors and results at "
        f"{pol.vector}. Expressible only because descriptor v0.1 carries prec_v "
        f"separately from prec_m -- the conflict v0 could not encode is a STORAGE "
        f"conflict (a buffer written by an fp16 GEMV read by an fp32 TRSV), not an "
        f"arithmetic one. The macro must declare {pol.bulk} distinct from bf16.",
        {"bulk": str(pol.bulk), "solve": str(pol.solve),
         "vector": str(pol.vector), "opt_in": True})


# ---------------------------------------------------------------------------
# C8 -- token reclaim safety
# ---------------------------------------------------------------------------

def check_token_reclaim(program: Program) -> Finding:
    """A descriptor that reclaims a live token must itself wait on that token.

    A token is a scoreboard slot, not a name, so 8 of them suffice for a measured
    peak liveness of 7. But reuse is only safe under this extra rule: a waiter
    resolves a token to its *most recent* producer, so if token t was claimed by
    descriptor 5, reclaimed by 40, and 50 depends on 5, then waiting on t waits
    for 40 -- which says nothing about 5, because in-order **issue** does not imply
    in-order **completion**. Making the reclaiming descriptor wait on the token it
    takes restores the ordering by induction.

    Checked on the unrolled stream, which is where it matters: the loop-carried
    edges between one iteration and the next are exactly the long-range
    dependencies a per-block check cannot see.
    """
    descs = program.descriptors
    graph = dependency_graph(descs)
    last_use: dict[int, int] = {}
    for j, producers in enumerate(graph):
        for i in producers:
            last_use[i] = max(last_use.get(i, -1), j)
    owner: dict[int, int] = {}
    violations: list[str] = []
    reclaims = 0
    for j, d in enumerate(descs):
        tok = d.chain_token
        prev = owner.get(tok)
        if prev is not None and last_use.get(prev, -1) >= j:
            # The previous owner is still needed by a later descriptor, so this is
            # a reclaim of a LIVE token.
            reclaims += 1
            if not (d.wait_mask & (1 << tok)):
                violations.append(
                    f"descriptor {j} ({d.label or d.opcode.name}) reclaims token "
                    f"{tok} from descriptor {prev}, which is still depended on by "
                    f"descriptor {last_use[prev]}, without waiting on it")
        owner[tok] = j
    if violations:
        return Finding(
            "C8", "a descriptor reclaiming a live token waits on it", "fail",
            f"{len(violations)} unsafe reclaim(s): " + "; ".join(violations[:5]),
            {"violations": len(violations), "live_reclaims": reclaims})
    return Finding(
        "C8", "a descriptor reclaiming a live token waits on it", "pass",
        f"{len(descs)} descriptors in the unrolled stream, {reclaims} reclaim(s) "
        f"of a still-live token, all of them ordered by a wait on the token they "
        f"take. {N_TOKENS} tokens available.",
        {"descriptors": len(descs), "live_reclaims": reclaims,
         "tokens": N_TOKENS})


# ---------------------------------------------------------------------------

def build_report(
    ctx: ContractContext, *, elided: Sequence[tuple[str, str, int]],
    elision_enabled: bool, second_signature: str, iterations: int,
) -> ComplianceReport:
    rep = ComplianceReport(target=ctx.model.name)
    rep.add(
        check_rho(ctx.options),
        check_transposed_gemv(ctx),
        check_cycles_per_descriptor(ctx),
        check_templates(ctx, iterations),
        check_static_allocation(ctx, second_signature),
        check_in_place(ctx, elided, elision_enabled),
        check_precision_policy(ctx.options),
        check_token_reclaim(ctx.unrolled),
    )
    return rep


__all__ = [
    "ARMING_INSTRUCTIONS", "CYCLES_PER_DESCRIPTOR_BAR", "ComplianceReport",
    "ContractContext", "Finding", "QUEUE_DEPTH", "TEMPLATES_IN_AREA_MODEL",
    "build_report", "check_cycles_per_descriptor", "check_in_place",
    "check_precision_policy", "check_rho", "check_static_allocation",
    "check_templates", "check_token_reclaim", "check_transposed_gemv",
]
