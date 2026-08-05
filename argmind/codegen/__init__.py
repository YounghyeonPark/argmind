"""Solver code generation: a problem description in, a descriptor program out.

This is the moat, stated plainly: nobody is going to hand-write accelerator
kernels for their MPC controller, so the generator is what makes the IP usable and
what makes the socket sticky. It targets the same runtime API the simulator and
the silicon expose, so a program generated here runs unmodified on both.

Usage
-----
::

    from argmind.codegen import CodegenOptions, QPProblem, generate

    problem = QPProblem.from_arrays(P, q, A, l, u, varies=("q", "l", "u"))
    gen = generate(problem, CodegenOptions(rho=0.125))
    stats = gen.solve()
    print(gen.report())          # program, allocation, residency, compliance

or from CVXPY, which is the on-ramp -- a user's existing problem, with the
call-to-call-varying data as ``cp.Parameter`` objects::

    from argmind.codegen.frontend import from_cvxpy

    problem, extraction = from_cvxpy(prob)
    gen = generate(problem)

What comes out
--------------
* a **descriptor program** in format v0.1: setup, iteration and residual blocks,
  with chain tokens and wait masks assigned from the real data dependencies;
* a **static scratchpad allocation plan** -- every address decided at generation
  time, identical across builds, no allocator in the solver path;
* a **residency plan**, derived from the emitted dataflow: what is fetched once,
  what is fetched per call, what never crosses the bus at all;
* a **compliance report** against the eight-rule codegen contract
  (:mod:`argmind.codegen.contract`), two of whose rules are enforced by refusing
  to generate at all.

Scope, deliberately
-------------------
One solver family, done properly: the OSQP problem form solved by reduced-form
ADMM. Not a general compiler, not a second solver family, not sparse. The
interior-point path (acados / HPIPM, and with it the CasADi front end) is a
separate target with a separate validation reference, and claiming it works
before it does would be a false claim about the on-ramp.
"""

from .admm import build_admm_graph
from .contract import (
    ARMING_INSTRUCTIONS,
    CYCLES_PER_DESCRIPTOR_BAR,
    QUEUE_DEPTH,
    TEMPLATES_IN_AREA_MODEL,
    ComplianceReport,
    Finding,
)
from .cost import CM50, CM100, DatapathModel, bank_spread, column_walk_penalty
from .generate import (
    GeneratedSolver,
    canonical_form,
    compare_programs,
    generate,
)
from .ir import (
    Kernel,
    KernelGraph,
    Ref,
    Source,
    Tensor,
    derive_precisions,
    elide_pure_copies,
)
from .problem import (
    QP_BLOCKS,
    CodegenError,
    CodegenOptions,
    ContractViolation,
    PrecisionPolicy,
    QPProblem,
    TerminationNorm,
    UnexpressiblePrecision,
)
from .residency import ResidencyPlan, plan_residency

__all__ = [
    "ARMING_INSTRUCTIONS",
    "CM50",
    "CM100",
    "CYCLES_PER_DESCRIPTOR_BAR",
    "CodegenError",
    "CodegenOptions",
    "ComplianceReport",
    "ContractViolation",
    "DatapathModel",
    "Finding",
    "GeneratedSolver",
    "Kernel",
    "KernelGraph",
    "PrecisionPolicy",
    "QP_BLOCKS",
    "QPProblem",
    "QUEUE_DEPTH",
    "Ref",
    "ResidencyPlan",
    "Source",
    "TEMPLATES_IN_AREA_MODEL",
    "Tensor",
    "TerminationNorm",
    "UnexpressiblePrecision",
    "bank_spread",
    "build_admm_graph",
    "canonical_form",
    "column_walk_penalty",
    "compare_programs",
    "derive_precisions",
    "elide_pure_copies",
    "generate",
    "plan_residency",
]
