# SPDX-License-Identifier: Apache-2.0
"""The generator driver: problem description in, runnable descriptor program out.

``generate(problem, options)`` runs the passes in order and returns a
:class:`GeneratedSolver` that is immediately runnable on the functional
simulator -- and, because the runtime API is identical across simulator, open core
and commercial core, on silicon.

Pass order, and why it is this order
------------------------------------
1. **Lower** the QP to a kernel graph (:mod:`argmind.codegen.admm`). Tensors, not
   addresses.
2. **Elide staging copies** (contract rule C6). Before allocation, because the
   whole point is that the eliminated buffer is never allocated.
3. **Allocate** in declaration order. Deterministic by construction, and it is
   here that a working set which does not fit becomes a build error rather than a
   runtime surprise.
4. **Lower to descriptors**, deriving ``prec_m`` / ``prec_v`` from the operands'
   formats and refusing what the format cannot encode.
5. **Assign tokens and wait masks** from the real data dependencies
   (:func:`argmind.program.auto_chain`).
6. **Build** on the device, which re-validates everything independently: MACIF
   precision refusals, opcode version gate, addresses against the allocation plan,
   and every dependency against the emitted tokens.
7. **Plan residency** and cross-check it against the device's own analysis.
8. **Check the contract** and attach the report.

Steps 6 and 7 are deliberately not the generator's own code. A generator that
validated its output with its own analysis would agree with itself.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from ..device import CM100_SCRATCHPAD_BYTES, Device, DeviceConfig
from ..macif import CM100 as MACIF_CM100
from ..macif import CM100_FP16, MacroCapabilities
from ..memory import Buffer
from ..precision import Format
from ..program import Program, SolverProgram, auto_chain, concat
from . import contract as C
from .admm import ADMMLowering, build_admm_graph
from .cost import CM100 as MODEL_CM100
from .cost import DatapathModel
from .ir import (
    ElisionResult, KernelGraph, allocate, elide_pure_copies, lower_block,
)
from .problem import (
    CodegenOptions, PrecisionPolicy, QPProblem, TerminationNorm,
)
from .residency import ResidencyPlan, cross_check_against_device, plan_residency

#: Iteration count the template-arming cost is amortised over in the report.
#: The measured QUAD-MPC closed loop is p50 7 / p95 8 / max 9 warm and 74 on the
#: cold worst-case tail; 12 is the eps=1e-8 cold figure and is used because
#: quoting the best case would flatter the arming argument.
REPORT_ITERATIONS = 12


def _plan_signature(plan) -> str:
    h = hashlib.sha256()
    for a in plan.allocations:
        h.update(f"{a.name}|{a.offset}|{a.nbytes}|{int(a.fmt)}|{a.shape}\n"
                 .encode("utf-8"))
    return h.hexdigest()


def _macro_for(policy: PrecisionPolicy) -> MacroCapabilities:
    """The smallest CM-100 macro variant that declares what the policy needs.

    fp16 is not part of the CM-100 baseline declaration, so a policy that asks
    for fp16 operands needs the variant that declares it -- and it must declare
    fp16 *distinct from* bf16, which is why substituting is refused rather than
    silently accepted.
    """
    needed = {policy.bulk, policy.solve, policy.vector}
    if Format.FP16 in needed:
        return CM100_FP16
    return MACIF_CM100


@dataclass
class GeneratedSolver:
    """A generated, allocated, validated, runnable ADMM solver.

    The host-side API is deliberately the same shape as the hand-written
    reference solver's, because that is the API a user writes against and it is
    the constitution: ``set_problem`` / ``set_instance`` / ``solve`` /
    ``solution``.
    """

    problem: QPProblem
    options: CodegenOptions
    device: Device
    graph: KernelGraph
    lowering: ADMMLowering
    buffers: dict[str, Buffer]
    solver: SolverProgram
    built: object
    residency: ResidencyPlan
    compliance: C.ComplianceReport
    model: DatapathModel
    elision: ElisionResult
    chain_info: dict = field(default_factory=dict)
    residency_cross_check: tuple[str, ...] = ()
    prefix: str = ""

    # -- host-side data -------------------------------------------------

    def buf(self, name: str) -> Buffer:
        return self.buffers[name]

    def load_problem_data(self) -> None:
        """Write the whole problem, plus the codegen constants, to host memory."""
        p, o = self.problem, self.options
        self.buf("P").write(p.P)
        self.buf("A").write(p.A)
        self.set_instance(p.q, p.l, p.u)
        self.buf("sigma_s").write(np.array([o.sigma], dtype=np.float64))
        # rho at STORAGE precision, and its reciprocal taken from the value the
        # setup GEMM's fp32 w7 alpha will actually carry, so the two agree. The
        # residual floor is set by rho*(1/rho) - 1, and mixing an fp32 rho into K
        # with an fp64 reciprocal of the unrounded value would reintroduce exactly
        # the 1e-8 mismatch that ruling R5 exists to remove.
        r = o.rho_as_encoded
        self.buf("rho_s").write(np.array([r], dtype=np.float64))
        self.buf("rho_inv_s").write(np.array([1.0 / r], dtype=np.float64))

    def set_problem(self, P=None, A=None) -> None:
        """Reload the structural data. Bumps generations, so setup re-runs."""
        if P is not None:
            self.buf("P").write(np.asarray(P, dtype=np.float64))
        if A is not None:
            self.buf("A").write(np.asarray(A, dtype=np.float64))

    def set_instance(self, q=None, l=None, u=None) -> None:
        """Load the per-tick data. Bounds are interleaved into (lo, hi) pairs.

        Interleaving is a codegen decision, not an API change: the signature is
        still ``(q, l, u)`` and the host pays the same number of stores.
        """
        if q is not None:
            self.buf("q").write(np.asarray(q, dtype=np.float64).ravel())
        if l is not None or u is not None:
            m = self.problem.n_con
            lu = np.array(self.buf("lu").numbers(), dtype=np.float64).reshape(m, 2)
            if l is not None:
                lu[:, 0] = np.asarray(l, dtype=np.float64).ravel()
            if u is not None:
                lu[:, 1] = np.asarray(u, dtype=np.float64).ravel()
            self.buf("lu").write(lu)

    def reset_iterates(self) -> None:
        """Cold start. Omit to warm start from the previous solve."""
        for nm in ("x", "y", "z"):
            b = self.buf(nm)
            b.write(np.zeros(b.shape, dtype=np.float64))

    # -- running --------------------------------------------------------

    @property
    def dirty(self) -> tuple[Buffer, ...]:
        """The dirty mask, derived: exactly the per-call tensors."""
        return tuple(self.buf(nm) for nm in self.residency.fetch_per_call)

    def solve(self, *, warm: bool = True, outputs: Iterable[Buffer] | None = None):
        if not warm:
            self.reset_iterates()
        outs = (list(outputs) if outputs is not None
                else [self.buf(nm) for nm in self.residency.written_back])
        return self.device.solve(self.built, dirty=self.dirty, outputs=outs)

    @property
    def solution(self) -> np.ndarray:
        return np.asarray(self.buf("x").read(), dtype=np.float64)

    # -- reporting ------------------------------------------------------

    def descriptor_counts(self) -> dict[str, int]:
        return {nm: len(getattr(self.solver, nm)) for nm in
                ("setup", "iteration", "residual")}

    @property
    def descriptors_per_checked_iteration(self) -> int:
        c = self.descriptor_counts()
        return c["iteration"] + c["residual"]

    def listing(self) -> str:
        out = []
        for nm in ("setup", "iteration", "residual"):
            out.append(f"{nm}:")
            out.append(getattr(self.solver, nm).listing())
        return "\n".join(out)

    def chain_report(self) -> str:
        lines = []
        for key in ("setup", "iteration", "residual"):
            c = self.chain_info[key]
            lines.append(
                f"  {key:<10} {len(c.descriptors):>3} desc  "
                f"emitted chain depth {c.chain_depth:>3} vs true DAG depth "
                f"{c.true_depth:>3} ({c.depth_penalty:.2f}x)  "
                f"policy={c.policy:<7} {len(c.multi_producer):>3} multi-producer  "
                f"{c.forced_serialisations:>3} forced  "
                f"peak live tokens {c.peak_live_tokens}/{8}")
        return "\n".join(lines)

    def report(self) -> str:
        cnt = self.descriptor_counts()
        parts = [
            f"generated solver {self.solver.name!r} from {self.problem.summary()}",
            f"  precision policy : {self.options.precision.name} "
            f"(bulk {self.options.precision.bulk}, solve "
            f"{self.options.precision.solve}, vectors "
            f"{self.options.precision.vector})",
            f"  descriptors      : setup {cnt['setup']}, iteration "
            f"{cnt['iteration']}, residual {cnt['residual']} -> "
            f"{self.descriptors_per_checked_iteration} per checked iteration",
            f"  scratchpad       : {self.device.plan.used} B "
            f"({self.device.plan.used / 1024:.1f} KiB) of "
            f"{self.device.config.scratchpad_bytes / 1024:.0f} KiB",
            "  dependency structure (emitted vs achievable):",
            self.chain_report(),
        ]
        if self.lowering.format_notes:
            parts.append("  format constraints the generator worked around:")
            parts.extend(f"    - {n}" for n in self.lowering.format_notes)
        parts.append(self.residency.report())
        if self.residency_cross_check:
            parts.append("  residency cross-check DISAGREEMENTS:")
            parts.extend(f"    - {p}" for p in self.residency_cross_check)
        else:
            parts.append("  residency plan agrees with the device's independent "
                         "descriptor-range analysis")
        parts.append(self.compliance.text())
        return "\n".join(parts)


def _build_graph_and_plan(
    problem: QPProblem, options: CodegenOptions, device: Device, prefix: str,
) -> tuple[ADMMLowering, ElisionResult, dict[str, Buffer]]:
    lowering = build_admm_graph(problem, options)
    elision = (elide_pure_copies(lowering.graph) if options.elide_copies
               else ElisionResult(graph=lowering.graph))
    buffers = allocate(lowering.graph, device, prefix=prefix)
    return lowering, elision, buffers


def generate(
    problem: QPProblem,
    options: CodegenOptions | None = None,
    *,
    device: Device | None = None,
    scratchpad_bytes: int = CM100_SCRATCHPAD_BYTES,
    macro: MacroCapabilities | None = None,
    model: DatapathModel = MODEL_CM100,
    prefix: str = "",
    host_bytes: int = 16 * 1024 * 1024,
    report_iterations: int = REPORT_ITERATIONS,
) -> GeneratedSolver:
    """Generate a descriptor program and a static allocation plan for `problem`.

    Raises
    ------
    ContractViolation
        A codegen contract rule the generator refuses to break (C1, C7).
    UnexpressiblePrecision
        Operand formats descriptor v0.1 cannot encode in one descriptor.
    ScratchpadExhausted
        The static working set does not fit. Not a warning: spilling a
        determinism-critical solver path to host memory forfeits the product.
    """
    options = options or CodegenOptions()

    # C1 and C7 are refusals and must happen before anything is emitted.
    rho_finding = C.check_rho(options)
    precision_finding = C.check_precision_policy(options)

    if device is None:
        device = Device(DeviceConfig(
            scratchpad_bytes=scratchpad_bytes,
            banks=model.banks,
            macro=macro or _macro_for(options.precision),
            host_bytes=host_bytes,
            check_hazards=True,
        ))
    lowering, elision, buffers = _build_graph_and_plan(
        problem, options, device, prefix)
    graph = lowering.graph
    signature = _plan_signature(device.plan)

    # Determinism check (C5): build the same program again on a throwaway device
    # and compare address maps. A second *independent* build, not a re-hash.
    shadow = Device(DeviceConfig(
        scratchpad_bytes=device.config.scratchpad_bytes,
        banks=device.config.banks, macro=device.config.macro,
        # The working set cannot exceed the scratchpad (allocation raises if it
        # does), so this much host memory always suffices and the shadow build
        # does not cost a second full host heap.
        host_bytes=device.config.scratchpad_bytes + len(graph.tensors) * 8,
        check_hazards=False))
    _build_graph_and_plan(problem, options, shadow, prefix)
    second_signature = _plan_signature(shadow.plan)

    # Lower to descriptors and assign tokens.
    chains = {}
    programs: dict[str, Program] = {}
    for name in graph.block_order:
        descs = lower_block(graph.blocks[name], buffers)
        chains[name] = auto_chain(descs)
        programs[name] = Program(f"{options.name}.{name}",
                                 chains[name].descriptors)

    solver_program = SolverProgram(
        name=options.name,
        setup=programs["setup"],
        iteration=programs["iteration"],
        residual=programs["residual"],
        converged=_make_convergence_test(options, problem, buffers),
        max_iterations=options.max_iterations,
        check_every=options.check_every,
    )
    built = device.build_solver(solver_program)

    residency = plan_residency(graph, buffers)
    cross = cross_check_against_device(residency, built, prefix=prefix)

    # The whole solve as one stream: two iterations and two residual blocks, so
    # the loop-carried dependencies are inside the checked window.
    unrolled = concat(f"{options.name}.unrolled", programs["setup"],
                      programs["iteration"], programs["residual"],
                      programs["iteration"], programs["residual"])

    ctx = C.ContractContext(
        options=options, graph=graph, blocks=programs,
        block_order=graph.block_order, model=model,
        plan_signature=signature, plan_used_bytes=device.plan.used,
        plan_capacity_bytes=device.config.scratchpad_bytes,
        unrolled=unrolled,
    )
    # C1 and C7 were already evaluated above, before anything was emitted --
    # they are refusals, so they cannot wait until reporting time. Re-evaluating
    # them inside build_report is harmless (both are pure functions of the
    # options) and keeps one place that composes the report.
    assert rho_finding.ok and precision_finding.ok
    report = C.build_report(
        ctx, elided=elision.eliminated, elision_enabled=options.elide_copies,
        second_signature=second_signature, iterations=report_iterations)

    gen = GeneratedSolver(
        problem=problem, options=options, device=device, graph=graph,
        lowering=lowering, buffers=buffers, solver=solver_program, built=built,
        residency=residency, compliance=report, model=model, elision=elision,
        chain_info=chains, residency_cross_check=cross, prefix=prefix,
    )
    gen.load_problem_data()
    return gen


def _make_convergence_test(options: CodegenOptions, problem: QPProblem,
                           buffers: dict[str, Buffer]):
    """The sequencer's convergence branch, as a closure over scratchpad scalars.

    Not a descriptor: descriptor v0.1 has no compare and no branch, and it should
    not -- data-dependent control flow is what the sequencer is for. This stands
    for a handful of scalar instructions reading scratchpad, and it is generated
    from the same options the descriptors are, so the stopping rule and the
    residual descriptors cannot disagree.
    """
    n, m = problem.n_var, problem.n_con
    eps_a, eps_r = options.eps_abs, options.eps_rel
    linf = options.termination is TerminationNorm.LINF

    def scalar(name: str) -> float:
        return float(buffers[name].spad_array.reshape(-1)[0])

    def converged() -> bool:
        if linf:
            r_prim = scalar("s_rp")
            r_dual = scalar("s_rd")
            eps_p = eps_a + eps_r * max(scalar("s_Ax"), scalar("s_z"))
            eps_d = eps_a + eps_r * max(scalar("s_Px"), scalar("s_Aty"),
                                        scalar("s_q"))
        else:
            # DOT returns squared 2-norms; one sqrt each on the FMA cluster.
            r_prim = math.sqrt(max(scalar("s_rp"), 0.0))
            r_dual = math.sqrt(max(scalar("s_rd"), 0.0))
            eps_p = eps_a * math.sqrt(m) + eps_r * max(
                math.sqrt(max(scalar("s_Ax"), 0.0)),
                math.sqrt(max(scalar("s_z"), 0.0)))
            eps_d = eps_a * math.sqrt(n) + eps_r * max(
                math.sqrt(max(scalar("s_Px"), 0.0)),
                math.sqrt(max(scalar("s_Aty"), 0.0)),
                math.sqrt(max(scalar("s_q"), 0.0)))
        return r_prim <= eps_p and r_dual <= eps_d

    return converged


def canonical_form(program: Program, device: Device) -> tuple[tuple, ...]:
    """A descriptor stream with addresses replaced by (buffer, element offset).

    This is what makes "generated == hand-written" a checkable claim rather than a
    hopeful one. Two programs built by different code paths will not agree on
    absolute addresses unless their allocation orders coincide, and requiring that
    would be testing the wrong thing -- what matters is that they issue the same
    opcodes with the same shapes on the same *operands*.
    """
    out = []
    for d in program.descriptors:
        ops = []
        for slot in ("a", "b", "c"):
            addr = getattr(d, slot)
            try:
                alloc = device.plan.lookup(addr)
            except Exception:
                ops.append((slot, None, addr))
                continue
            name = alloc.name.split(".")[-1]
            ops.append((slot, name,
                        (addr - alloc.offset) // alloc.fmt.itemsize))
        out.append((
            d.opcode, int(d.prec_m), int(d.prec_v), d.m, d.k, d.n,
            d.lda, d.ldb, d.ldc, float(np.float32(d.alpha)),
            int(d.transpose), int(d.structure), bool(d.accumulate),
            bool(d.unit_diag), tuple(ops),
        ))
    return tuple(out)


def compare_programs(
    a: Program, dev_a: Device, b: Program, dev_b: Device,
) -> tuple[str, ...]:
    """Differences between two descriptor streams, in canonical form."""
    ca, cb = canonical_form(a, dev_a), canonical_form(b, dev_b)
    diffs: list[str] = []
    if len(ca) != len(cb):
        diffs.append(f"length {len(ca)} vs {len(cb)}")
    for i, (x, y) in enumerate(zip(ca, cb)):
        if x != y:
            diffs.append(f"[{i}] {x} != {y}")
    return tuple(diffs)


__all__ = [
    "GeneratedSolver", "REPORT_ITERATIONS", "canonical_form", "compare_programs",
    "generate",
]
