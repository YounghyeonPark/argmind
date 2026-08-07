# SPDX-License-Identifier: Apache-2.0
"""Lowering a QP to an ADMM kernel graph.

This is the solver-specific half of the generator: everything that knows what
ADMM *is*. It produces a :class:`~argmind.codegen.ir.KernelGraph` and nothing
else -- no addresses, no descriptors, no device. The passes that follow
(:mod:`argmind.codegen.generate`) are solver-agnostic.

The algorithm is the reduced (Cholesky) form of the OSQP iteration, so the
generated program can be compared against OSQP directly rather than against a
paraphrase of it::

    K    = P + sigma*I + rho*A'A          (once, while P, A and rho hold)
    K   := chol(K)

    rhs  = sigma*x - q + A'(rho*z - y)
    xt   = K^-1 rhs                       (two triangular solves on the factor)
    zt   = A xt
    x    = alpha*xt + (1-alpha)*x
    t    = alpha*zt + (1-alpha)*z + y/rho
    z    = clip(t, l, u)
    y    = rho*(t - z)

Two things OSQP does at runtime are deliberately not generated: **adaptive rho**
(data-dependent refactorisation mid-solve, i.e. data-dependent latency) and
**Ruiz equilibration** (fine in itself, computable at code-generation time, but
measured to *increase* iteration count at fixed rho on this problem). Both
decisions move to codegen rather than disappearing.

The dual update is emitted in the compact form ``y = rho*(t - z)``. The expanded
form ``y += rho*(t - z)`` is a different algorithm and is wrong -- but only when
the active set at the optimum is non-empty, which is why every validation of this
generator runs a tight-bound variant alongside the nominal one.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..descriptor import Opcode, Structure, Transpose
from .ir import Kernel, KernelGraph, Ref, Source, Tensor
from .problem import CodegenOptions, QPProblem, TerminationNorm

#: Which QP data block maps to which tensor name, for the residency report.
BLOCK_TENSORS: dict[str, tuple[str, ...]] = {
    "P": ("P",),
    "q": ("q",),
    "A": ("A",),
    "l": ("lu",),
    "u": ("lu",),
}


@dataclass
class ADMMLowering:
    """The graph plus the handful of tensors the host-side API needs by name."""

    graph: KernelGraph
    P: Tensor
    A: Tensor
    q: Tensor
    lu: Tensor
    L: Tensor
    x: Tensor
    y: Tensor
    z: Tensor
    sigma_s: Tensor
    rho_s: Tensor
    rho_inv_s: Tensor
    scalars: dict[str, Tensor]
    #: Notes about format decisions forced by the descriptor format, for the
    #: compliance report. Each one is a place v0.1 constrained the generator.
    format_notes: tuple[str, ...] = ()


def _source_for(block: str, varies: frozenset[str]) -> Source:
    return Source.INSTANCE if block in varies else Source.STRUCTURAL


def build_admm_graph(problem: QPProblem, opt: CodegenOptions) -> ADMMLowering:
    """Lower a QP to the ADMM kernel graph. No addresses are assigned here."""
    n, m = problem.n_var, problem.n_con
    pol = opt.precision
    g = KernelGraph(name=opt.name)

    notes: list[str] = []

    # ---------------- tensors ------------------------------------------
    #
    # Declaration order IS allocation order, so it is chosen deliberately:
    # the three large matrices first (they dominate the footprint and a reader of
    # the address map should see them immediately), then per-call data, then
    # constants, then the factor, iterates, temporaries, scalars.

    # P's format is NOT a free choice, and this is the first place descriptor
    # v0.1 constrains the generator. `K += P` is an elementwise matrix add, and
    # the elementwise opcodes have no bulk operand, so they carry exactly one
    # format: P must be stored at the factor's format even when the policy asks
    # for narrow bulk operands. See ir.derive_precisions.
    p_fmt = pol.solve
    if p_fmt is not pol.bulk:
        notes.append(
            f"P is stored at {p_fmt} rather than the requested bulk format "
            f"{pol.bulk} because the setup block adds it into K with an "
            f"elementwise AXPY, and descriptor v0.1's elementwise opcodes carry "
            f"exactly one format (prec_m is unused by them). Cost: "
            f"{n * n * (p_fmt.itemsize - pol.bulk.itemsize)} B of extra "
            f"scratchpad and a P*x GEMV at {p_fmt} instead of {pol.bulk}.")

    P = g.declare("P", (n, n), _source_for("P", problem.varies), p_fmt, role="P")
    A = g.declare("A", (m, n), _source_for("A", problem.varies), pol.bulk, role="A")
    q = g.declare("q", n, _source_for("q", problem.varies), pol.vector, role="q")
    # Bounds INTERLEAVED as (lo, hi) pairs -- descriptor v0.1 ruling R2. That is
    # what frees a base address so PROJ can be out-of-place with three AGUs, and
    # at fp32 a pair is 8 B = exactly one bank word.
    lu_src = (Source.INSTANCE
              if ({"l", "u"} & problem.varies) else Source.STRUCTURAL)
    lu = g.declare("lu", (m, 2), lu_src, pol.vector, role="l,u")

    sigma_s = g.declare("sigma_s", 1, Source.CONSTANT, pol.solve)
    rho_s = g.declare("rho_s", 1, Source.CONSTANT, pol.vector)
    rho_inv_s = g.declare("rho_inv_s", 1, Source.CONSTANT, pol.vector)

    L = g.declare("L", (n, n), Source.DERIVED, pol.solve)

    x = g.declare("x", n, Source.ITERATE, pol.vector, host_visible=True)
    y = g.declare("y", m, Source.ITERATE, pol.vector)
    z = g.declare("z", m, Source.ITERATE, pol.vector)

    def tmp(name: str, length: int) -> Tensor:
        return g.declare(name, length, Source.INTERNAL, pol.vector)

    t_dual = tmp("t_dual", m)
    rhs = tmp("rhs", n)
    xt = tmp("xt", n)
    zt = tmp("zt", m)
    t = tmp("t", m)
    Ax = tmp("Ax", m)
    rp = tmp("rp", m)
    Px = tmp("Px", n)
    Aty = tmp("Aty", n)
    rd = tmp("rd", n)

    scalar_names = ("s_rp", "s_Ax", "s_z", "s_rd", "s_Px", "s_Aty", "s_q")
    scalars = {nm: g.declare(nm, 1, Source.INTERNAL, pol.vector)
               for nm in scalar_names}

    R = Ref  # brevity below; every operand is a Ref

    # ---------------- setup: K = P + sigma*I + rho*A'A, then chol -------
    #
    # Runs only when P, A, rho or sigma change generation. For a controller
    # re-solving a structurally identical problem every tick that is once, ever.

    g.emit("setup", Kernel(
        Opcode.GEMM, a=R(A), b=R(A), c=R(L), m=n, k=m, n=n,
        lda=n, ldb=n, ldc=n, alpha=opt.rho, transpose=Transpose.A,
        label="K = rho*A'A"))
    # A genuine MATRIX add: v0.1 ruling R6a makes the elementwise opcodes walk
    # M rows x N columns. v0 had no matrix add and forced the flat form m = n*n,
    # which is what capped n at 255.
    g.emit("setup", Kernel(
        Opcode.AXPY, b=R(P), c=R(L), m=n, n=n, ldb=n, ldc=n, alpha=1.0,
        label="K += P"))
    # ldb = 0 is a zero-stride BROADCAST read of a one-element buffer (ruling
    # R6b); v0 needed a materialised sigma*ones(n) vector here.
    g.emit("setup", Kernel(
        Opcode.AXPY, b=R(sigma_s), c=R(L), m=n, ldb=0, ldc=n + 1, alpha=1.0,
        label="K += sigma*I  (broadcast)"))
    g.emit("setup", Kernel(
        Opcode.CHOL, a=R(L), m=n, lda=n, structure=Structure.TRIL,
        label="K := chol(K)"))

    # ---------------- the rho-scaling helper ---------------------------

    def scale_by_rho(src: Tensor, dest: Tensor, length: int, label: str, *,
                     recip: bool = False, negate: bool = False,
                     accumulate: bool = False) -> Kernel:
        """Multiply a vector by rho or 1/rho, in one of two encodings.

        Descriptor v0.1 ruling R5. Either the constant travels as an **fp32
        immediate in w7** (v0's only option) or it is read from a **scratchpad
        scalar at the buffer's own precision by a zero-stride VMUL**. The second
        takes the reciprocal mismatch that stalls the solve from 1.5e-8 to
        ~1e-16, which is why it is the generator's default.
        """
        sign = -1.0 if negate else 1.0
        if opt.rho_in_scratchpad:
            scalar = rho_inv_s if recip else rho_s
            return Kernel(
                Opcode.VMUL, a=R(scalar), b=R(src), c=R(dest), m=length,
                lda=0, ldb=1, ldc=1, alpha=sign, accumulate=accumulate,
                label=label)
        k = (1.0 / opt.rho) if recip else opt.rho
        return Kernel(
            Opcode.AXPY if accumulate else Opcode.SCAL,
            b=R(src), c=R(dest), m=length, alpha=sign * k, label=label)

    # ---------------- iteration ----------------------------------------

    al, sigma = opt.alpha, opt.sigma
    it = [
        # t_dual = rho*z - y
        scale_by_rho(z, t_dual, m, "t_dual = rho*z"),
        Kernel(Opcode.AXPY, b=R(y), c=R(t_dual), m=m, alpha=-1.0,
               label="t_dual -= y"),
        # rhs = sigma*x - q + A' t_dual
        Kernel(Opcode.SCAL, b=R(x), c=R(rhs), m=n, alpha=sigma,
               label="rhs = sigma*x"),
        Kernel(Opcode.AXPY, b=R(q), c=R(rhs), m=n, alpha=-1.0, label="rhs -= q"),
        # ONE descriptor, transpose=A, reading the SAME row-major A the
        # untransposed product reads. Contract rule C2: the hardware accumulates
        # row-wise into an n_var accumulator; a column walk would reach a
        # fraction of the banks. Never a materialised transpose, never a
        # per-column DOT loop.
        Kernel(Opcode.GEMV, a=R(A), b=R(t_dual), c=R(rhs), m=m, k=n, lda=n,
               alpha=1.0, transpose=Transpose.A, accumulate=True,
               label="rhs += A'*t_dual"),
        # xt = K^-1 rhs, on the resident factor
        Kernel(Opcode.TRSV, a=R(L), b=R(rhs), c=R(xt), m=n, lda=n,
               structure=Structure.TRIL, label="xt = L^-1 rhs"),
        Kernel(Opcode.TRSV, a=R(L), b=R(xt), c=R(xt), m=n, lda=n,
               structure=Structure.TRIL, transpose=Transpose.A,
               label="xt = L^-T xt"),
        Kernel(Opcode.GEMV, a=R(A), b=R(xt), c=R(zt), m=m, k=n, lda=n,
               alpha=1.0, label="zt = A*xt"),
        # x = alpha*xt + (1-alpha)*x, in place on the persistent iterate
        Kernel(Opcode.SCAL, b=R(x), c=R(x), m=n, alpha=1.0 - al,
               label="x *= (1-alpha)"),
        Kernel(Opcode.AXPY, b=R(xt), c=R(x), m=n, alpha=al,
               label="x += alpha*xt"),
        # t = alpha*zt + (1-alpha)*z + y/rho
        Kernel(Opcode.SCAL, b=R(zt), c=R(t), m=m, alpha=al, label="t = alpha*zt"),
        Kernel(Opcode.AXPY, b=R(z), c=R(t), m=m, alpha=1.0 - al,
               label="t += (1-alpha)*z"),
        scale_by_rho(y, t, m, "t += y/rho", recip=True, accumulate=True),
        # z = clip(t, lu). ONE descriptor in v0.1: interleaved (lo,hi) pairs at
        # `b` free `a` for the input, so projection is out of place. v0 needed a
        # staging SCAL of the whole constraint vector first (limitation L2).
        Kernel(Opcode.PROJ, a=R(t), b=R(lu), c=R(z), m=m, lda=1, ldb=2, ldc=1,
               label="z = clip(t, l, u)"),
        # y = rho*(t - z)
        scale_by_rho(t, y, m, "y = rho*t"),
        scale_by_rho(z, y, m, "y -= rho*z", negate=True, accumulate=True),
    ]
    for k in it:
        g.emit("iteration", k)

    # ---------------- residual ------------------------------------------
    #
    # OSQP's stopping rule, in the norms OSQP states it in. The scaling terms
    # (|Ax|, |z|, |Px|, |A'y|, |q|) are not optional decoration: without them the
    # test is a different test and the iteration count stops being comparable.

    red = Opcode.DOT if opt.termination is TerminationNorm.L2 else Opcode.NRMI

    def reduce_(dest: Tensor, src: Tensor, length: int, label: str) -> Kernel:
        if red is Opcode.DOT:
            return Kernel(Opcode.DOT, a=R(dest), b=R(src), c=R(src), m=length,
                          label=label)
        return Kernel(Opcode.NRMI, a=R(dest), b=R(src), m=length, label=label)

    res = [
        Kernel(Opcode.GEMV, a=R(A), b=R(x), c=R(Ax), m=m, k=n, lda=n, alpha=1.0,
               label="Ax = A*x"),
        reduce_(scalars["s_Ax"], Ax, m, "|Ax|"),
        reduce_(scalars["s_z"], z, m, "|z|"),
        # rp = Ax - z. The copy here is what contract rule C6 removes: |Ax| is
        # already taken, so Ax is dead afterwards and the subtraction can happen
        # in place. Emitted anyway so that the elision pass has something to
        # find and so the reference-equivalent program is still generatable.
        Kernel(Opcode.SCAL, b=R(Ax), c=R(rp), m=m, alpha=1.0, label="rp = Ax"),
        Kernel(Opcode.AXPY, b=R(z), c=R(rp), m=m, alpha=-1.0, label="rp -= z"),
        reduce_(scalars["s_rp"], rp, m, "|Ax - z|  primal residual"),
        Kernel(Opcode.GEMV, a=R(P), b=R(x), c=R(Px), m=n, k=n, lda=n, alpha=1.0,
               structure=Structure.SYMMETRIC, label="Px = P*x"),
        reduce_(scalars["s_Px"], Px, n, "|Px|"),
        Kernel(Opcode.GEMV, a=R(A), b=R(y), c=R(Aty), m=m, k=n, lda=n, alpha=1.0,
               transpose=Transpose.A, label="Aty = A'*y"),
        reduce_(scalars["s_Aty"], Aty, n, "|A'y|"),
        Kernel(Opcode.SCAL, b=R(Px), c=R(rd), m=n, alpha=1.0, label="rd = Px"),
        Kernel(Opcode.AXPY, b=R(q), c=R(rd), m=n, alpha=1.0, label="rd += q"),
        Kernel(Opcode.AXPY, b=R(Aty), c=R(rd), m=n, alpha=1.0, label="rd += A'y"),
        reduce_(scalars["s_rd"], rd, n, "|Px + q + A'y|  dual residual"),
        reduce_(scalars["s_q"], q, n, "|q|"),
    ]
    for k in res:
        g.emit("residual", k)

    return ADMMLowering(
        graph=g, P=P, A=A, q=q, lu=lu, L=L, x=x, y=y, z=z,
        sigma_s=sigma_s, rho_s=rho_s, rho_inv_s=rho_inv_s, scalars=scalars,
        format_notes=tuple(notes),
    )
