# SPDX-License-Identifier: Apache-2.0
"""ADMM for the OSQP problem form, as a descriptor sequence.

Problem::

    minimize    (1/2) x' P x + q' x
    subject to  l <= A x <= u

with ``P`` symmetric positive semidefinite, ``n`` variables and ``m`` constraint
rows. This is OSQP's native form, so the generated program can be compared
against OSQP directly rather than against a paraphrase of it.

Algorithm
---------
The reduced (Cholesky) form of the OSQP iteration. Eliminating the dual variable
from the KKT system leaves

.. math::

    K = P + \\sigma I + \\rho A^\\top A

which is symmetric positive definite and, crucially, **constant for as long as
the problem structure and rho are constant**. So it is factorised once and the
factor stays resident; a controller re-solving a structurally identical problem
every tick never refactorises. One further simplification falls out of the
elimination: :math:`\\tilde z^{k+1} = A \\tilde x^{k+1}` exactly, so no separate
update for it is needed.

Per iteration::

    rhs  = sigma*x - q + A'(rho*z - y)
    xt   = K^-1 rhs                       (two triangular solves on the factor)
    zt   = A xt
    x    = alpha*xt + (1-alpha)*x
    t    = alpha*zt + (1-alpha)*z + y/rho
    z    = clip(t, l, u)
    y    = rho*(t - z)

The dual update is written in that compact form deliberately. The expanded form
``y <- y + rho*(t - z)`` is a *different* algorithm and is wrong, but it is wrong
only when the active set at the optimum is non-empty -- at an unconstrained
optimum both forms give y = 0 and agree to machine precision. A validation
problem whose optimum is interior therefore cannot detect the error. See
:mod:`tests.test_admm_vs_osqp`, which pins a case with an active set for exactly
this reason.

Determinism
-----------
Two things OSQP does at runtime are deliberately **not** done here:

* **Adaptive rho.** OSQP recomputes rho from the residual ratio and refactorises
  the KKT matrix mid-solve. That is data-dependent work of data-dependent
  duration -- exactly the latency variance a real-time guarantee cannot absorb.
  rho is fixed at code-generation time here.
* **Ruiz equilibration.** Scaling is fine in itself -- it is a similarity
  transform of the problem data and can be computed once, at code-generation
  time, costing nothing at runtime. It is simply not applied here, because on this
  problem it is not beneficial (measured: it *increases* iteration count at fixed
  rho). Whether to scale is a per-problem codegen decision, and it is not free to
  assume either way.

The consequence is that rho becomes a code-generation tuning parameter whose value
the generator must choose, and choose per problem. That responsibility does not
disappear by being static; it moves.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np

from ..descriptor import Descriptor, Opcode, Structure, Transpose
from ..device import Device
from ..errors import ArgmindError
from ..memory import Buffer, Lifetime
from ..precision import Format
from ..program import Program, SolverProgram, auto_chain


class TerminationNorm(Enum):
    """Which norm the convergence test uses.

    ``L2`` is expressible in **frozen descriptor format v0** (via ``DOT``).
    ``LINF`` reproduces OSQP's own criterion and uses ``NRMI``, which descriptor
    **v0.1 adds** (ruling R1). The two programs are otherwise identical -- the
    same fifteen descriptors with one opcode substituted -- which is the cheapest
    possible demonstration of what the missing reduction cost: a different
    iteration count and 3.7x worse control action at the same tolerance.

    Both run entirely in fabric. The claim that without ``NRMI`` the convergence
    test returns to the host is **not true** and is not the argument for R1 --
    the L2 path is DOT plus two square roots on the FMA cluster plus compares on
    the sequencer, with no host round trip. What the substitute costs is a
    different answer from the reference solver.
    """

    L2 = "l2"
    LINF = "linf"

    @property
    def opcode(self) -> Opcode:
        return Opcode.DOT if self is TerminationNorm.L2 else Opcode.NRMI


@dataclass(frozen=True)
class ADMMConfig:
    """Code-generation-time parameters. All of these are baked into descriptors.

    ``rho`` defaults to **0.125, not OSQP's 0.1, and the reason is a property of
    the descriptor format.** The iteration multiplies by ``1/rho`` (building
    ``t``) and by ``rho`` (the dual update), and both constants travel in
    descriptor word 7, which is an fp32 field. The ADMM fixed point requires
    those two constants to be exact reciprocals: at a fixed point with an active
    constraint, ``y = rho * (1/rho) * y``, so if ``fl32(rho) * fl32(1/rho) != 1``
    the only solution is ``y = 0`` and the dual variable cannot settle. The solve
    then stalls at a residual floor rather than converging.

    Measured on a 6-variable / 8-constraint dense QP, requesting 1e-9::

        rho     fl32(rho)*fl32(1/rho)-1   iters   final primal residual
        0.1                   1.490e-08    4000+  1.02e-07   (stalled)
        0.2                   1.490e-08    4000+  5.08e-08   (stalled)
        0.3                   1.589e-08    4000+  3.61e-08   (stalled)
        0.125                 0            56     1.05e-09   converged
        0.25                  0            46     1.10e-09   converged
        0.5                   0            52     6.36e-10   converged

    So: **rho must be a power of two**, and that belongs in the codegen contract.
    It costs nothing -- rho is a tuning parameter and ADMM convergence is smooth
    in log rho -- but it is invisible in the paper specification and it fails
    silently, as a stalled solve rather than an error.
    """

    rho: float = 0.125
    sigma: float = 1e-6
    alpha: float = 1.6
    eps_abs: float = 1e-6
    eps_rel: float = 1e-6
    max_iterations: int = 200
    check_every: int = 1
    fmt: Format = Format.FP64
    termination: TerminationNorm = TerminationNorm.LINF
    #: Carry rho and 1/rho as scratchpad scalars read by a zero-stride ``VMUL``,
    #: instead of as fp32 immediates in descriptor word 7. This is descriptor
    #: v0.1 ruling R5, and it is what makes the power-of-two rho contract
    #: non-load-bearing: at fp64 storage the reciprocal mismatch is ~1e-16
    #: instead of w7's 1.5e-8, which is below any realistic tolerance. Off by
    #: default so the w7 path -- and the stall it causes -- stays testable.
    rho_in_scratchpad: bool = False

    @property
    def rho_as_encoded(self) -> float:
        """rho as it survives descriptor word 7, which is an **fp32** field."""
        return float(np.float32(self.rho))

    @property
    def sigma_as_encoded(self) -> float:
        return float(np.float32(self.sigma))

    @property
    def alpha_as_encoded(self) -> float:
        return float(np.float32(self.alpha))

    @property
    def rho_reciprocal_error(self) -> float:
        """``fl32(rho) * fl32(1/rho) - 1``. Must be exactly 0 for convergence.

        Also bounds the residual floor if it is not: the dual update is off by
        this relative factor every iteration.
        """
        r = np.float32(self.rho)
        c = np.float32(1.0 / float(r))
        return float(np.float64(r) * np.float64(c) - 1.0)


class ADMMSolver:
    """Builds and drives the ADMM descriptor program for one problem shape.

    Dimensions are fixed at construction: every address, dimension and stride in
    every descriptor is a literal, and the scratchpad map is decided once. That
    is the whole point -- embedded MPC dimensions are known when the code is
    generated, so nothing about the schedule needs to be discovered at run time.
    """

    def __init__(
        self,
        device: Device,
        n: int,
        m: int,
        config: ADMMConfig | None = None,
        name: str = "admm",
        allow_inexact_rho: bool = False,
    ) -> None:
        self.device = device
        self.n = int(n)
        self.m = int(m)
        self.cfg = config or ADMMConfig()
        self.name = name
        self.fmt = self.cfg.fmt
        err = self.cfg.rho_reciprocal_error
        if err != 0.0 and not allow_inexact_rho and not self.cfg.rho_in_scratchpad:
            raise ArgmindError(
                f"rho={self.cfg.rho!r} is not exactly invertible in the fp32 "
                f"descriptor scalar field: fl32(rho)*fl32(1/rho) - 1 = {err:.3e}. "
                f"The ADMM fixed point requires these to be exact reciprocals, so "
                f"the solve will stall at a primal-residual floor of roughly "
                f"{abs(err):.1e} times the dual magnitude instead of converging. "
                f"Choose a power of two (0.0625, 0.125, 0.25, 0.5, 1.0, ...), or "
                f"pass allow_inexact_rho=True if you are deliberately measuring "
                f"the stall."
            )
        self._alloc()
        self.solver = self._build_solver()
        self.built = device.build_solver(self.solver)

    # ------------------------------------------------------------------
    # Static allocation. Order here IS the scratchpad layout.
    # ------------------------------------------------------------------

    def _alloc(self) -> None:
        dev, n, m, f = self.device, self.n, self.m, self.fmt
        p = Lifetime.PERSISTENT

        def b(nm: str, shape, life=p) -> Buffer:
            return dev.alloc(f"{self.name}.{nm}", shape, f, life)

        # Problem data. Generation-stable across control ticks except q and lu.
        self.P = b("P", (n, n))
        self.A = b("A", (m, n))
        # ONE element, not n. Descriptor v0.1 ruling R6b makes a zero AGU stride
        # legal on reads, so the diagonal add broadcasts a single scalar instead
        # of walking a materialised sigma*ones(n) vector: n-1 fewer element
        # fetches and n-1 fewer stored elements, for zero descriptors.
        self.sigma_s = b("sigma_s", 1)
        self.q = b("q", n)
        # Bounds INTERLEAVED as (lo, hi) pairs -- descriptor v0.1 ruling R2. This
        # is what frees a base address so PROJ can be out-of-place with three
        # AGUs. At fp32 a pair is 8 B = exactly one bank word, so one bank access
        # delivers both bounds where two separate vectors needed two accesses in
        # two banks.
        self.lu = b("lu", (m, 2))
        # rho and 1/rho at storage precision, for ruling R5. Allocated
        # unconditionally (16-32 B) so the allocation plan does not depend on a
        # codegen flag; only the descriptors change.
        self.rho_s = b("rho_s", 1)
        self.rho_inv_s = b("rho_inv_s", 1)

        # The factor. Built by the setup block, overwriting K in place.
        self.L = b("L", (n, n))

        # Iterates. Persistent because that is what warm starting *is*.
        self.x = b("x", n)
        self.y = b("y", m)
        self.z = b("z", m)

        # Working vectors. Persistent (they live on the CM) but never host-fetched,
        # because every one of them is written before it is read.
        self.t_dual = b("t_dual", m)
        self.rhs = b("rhs", n)
        self.xt = b("xt", n)
        self.zt = b("zt", m)
        self.t = b("t", m)
        self.Ax = b("Ax", m)
        self.rp = b("rp", m)
        self.Px = b("Px", n)
        self.Aty = b("Aty", n)
        self.rd = b("rd", n)

        # Convergence scalars, read by the sequencer's branch.
        self.s_rp = b("s_rp", 1)
        self.s_Ax = b("s_Ax", 1)
        self.s_z = b("s_z", 1)
        self.s_rd = b("s_rd", 1)
        self.s_Px = b("s_Px", 1)
        self.s_Aty = b("s_Aty", 1)
        self.s_q = b("s_q", 1)

    # ------------------------------------------------------------------
    # Descriptor blocks
    # ------------------------------------------------------------------

    def _setup_descriptors(self) -> list[Descriptor]:
        """K = P + sigma*I + rho*A'A, then K := chol(K) in place."""
        n, m, f = self.n, self.m, self.fmt
        cfg = self.cfg
        return [
            # K := rho * A' A. Both operands are the same buffer.
            Descriptor(
                Opcode.GEMM, f, a=self.A.spad_offset, b=self.A.spad_offset,
                c=self.L.spad_offset, m=n, k=m, n=n,
                lda=n, ldb=n, ldc=n, alpha=cfg.rho,
                transpose=Transpose.A, label="K = rho*A'A",
            ),
            # K += P as a genuine MATRIX add: v0.1 ruling R6a makes the
            # elementwise opcodes walk M rows x N columns. v0 had no matrix-add
            # and forced the flat form m = n*n, which is what capped n at 255.
            # Here n = 80 needs 6400 in a 16-bit field either way, but a
            # 300-variable condensed QP would have overflowed and now does not.
            Descriptor(
                Opcode.AXPY, f, b=self.P.spad_offset, c=self.L.spad_offset,
                m=n, n=n, ldb=n, ldc=n, alpha=1.0, label="K += P",
            ),
            # K += sigma*I. The diagonal walk is ldc = n+1 as in v0; what is new
            # is ldb = 0, a zero-stride BROADCAST read of a one-element buffer
            # (ruling R6b). v0 needed a materialised sigma*ones(n) vector here.
            Descriptor(
                Opcode.AXPY, f, b=self.sigma_s.spad_offset,
                c=self.L.spad_offset, m=n, ldb=0, ldc=n + 1, alpha=1.0,
                label="K += sigma*I  (broadcast)",
            ),
            Descriptor(
                Opcode.CHOL, f, a=self.L.spad_offset, m=n, lda=n,
                structure=Structure.TRIL, label="K := chol(K)",
            ),
        ]

    def _scale_by_rho(self, src, dest, length: int, label: str,
                      recip: bool = False, negate: bool = False,
                      accumulate: bool = False) -> Descriptor:
        """Multiply a vector by rho or 1/rho.

        Two encodings of the same arithmetic, and the choice is descriptor v0.1
        ruling R5. Either the constant travels as an **fp32 immediate in w7**
        (v0's only option), or it is read from a **scratchpad scalar at the
        buffer's own precision by a zero-stride VMUL**. The second is what takes
        the reciprocal mismatch that stalls the solve from 1.5e-8 to ~1e-16.
        """
        f, cfg = self.fmt, self.cfg
        sign = -1.0 if negate else 1.0
        if cfg.rho_in_scratchpad:
            scalar = self.rho_inv_s if recip else self.rho_s
            return Descriptor(
                Opcode.VMUL, f, a=scalar.spad_offset, b=src.spad_offset,
                c=dest.spad_offset, m=length, lda=0, ldb=1, ldc=1,
                alpha=sign, accumulate=accumulate, label=label,
            )
        k = (1.0 / cfg.rho) if recip else cfg.rho
        op = Opcode.AXPY if accumulate else Opcode.SCAL
        return Descriptor(
            op, f, b=src.spad_offset, c=dest.spad_offset, m=length,
            alpha=sign * k, label=label,
        )

    def _iteration_descriptors(self) -> list[Descriptor]:
        n, m, f = self.n, self.m, self.fmt
        cfg = self.cfg
        rho, sigma, al = cfg.rho, cfg.sigma, cfg.alpha
        D = Descriptor
        return [
            # t_dual = rho*z - y
            self._scale_by_rho(self.z, self.t_dual, m, "t_dual = rho*z"),
            D(Opcode.AXPY, f, b=self.y.spad_offset, c=self.t_dual.spad_offset,
              m=m, alpha=-1.0, label="t_dual -= y"),
            # rhs = sigma*x - q + A' t_dual
            D(Opcode.SCAL, f, b=self.x.spad_offset, c=self.rhs.spad_offset,
              m=n, alpha=sigma, label="rhs = sigma*x"),
            D(Opcode.AXPY, f, b=self.q.spad_offset, c=self.rhs.spad_offset,
              m=n, alpha=-1.0, label="rhs -= q"),
            D(Opcode.GEMV, f, a=self.A.spad_offset, b=self.t_dual.spad_offset,
              c=self.rhs.spad_offset, m=m, k=n, lda=n, alpha=1.0,
              transpose=Transpose.A, accumulate=True, label="rhs += A'*t_dual"),
            # xt = K^-1 rhs, via the resident Cholesky factor
            D(Opcode.TRSV, f, a=self.L.spad_offset, b=self.rhs.spad_offset,
              c=self.xt.spad_offset, m=n, lda=n, structure=Structure.TRIL,
              label="xt = L^-1 rhs"),
            D(Opcode.TRSV, f, a=self.L.spad_offset, b=self.xt.spad_offset,
              c=self.xt.spad_offset, m=n, lda=n, structure=Structure.TRIL,
              transpose=Transpose.A, label="xt = L^-T xt"),
            # zt = A xt
            D(Opcode.GEMV, f, a=self.A.spad_offset, b=self.xt.spad_offset,
              c=self.zt.spad_offset, m=m, k=n, lda=n, alpha=1.0, label="zt = A*xt"),
            # x = alpha*xt + (1-alpha)*x
            D(Opcode.SCAL, f, b=self.x.spad_offset, c=self.x.spad_offset,
              m=n, alpha=1.0 - al, label="x *= (1-alpha)"),
            D(Opcode.AXPY, f, b=self.xt.spad_offset, c=self.x.spad_offset,
              m=n, alpha=al, label="x += alpha*xt"),
            # t = alpha*zt + (1-alpha)*z + y/rho
            D(Opcode.SCAL, f, b=self.zt.spad_offset, c=self.t.spad_offset,
              m=m, alpha=al, label="t = alpha*zt"),
            D(Opcode.AXPY, f, b=self.z.spad_offset, c=self.t.spad_offset,
              m=m, alpha=1.0 - al, label="t += (1-alpha)*z"),
            self._scale_by_rho(self.y, self.t, m, "t += y/rho",
                               recip=True, accumulate=True),
            # z = clip(t, lu). ONE descriptor in v0.1: the bounds are interleaved
            # (lo,hi) pairs at `b`, which frees `a` for the input, so projection is
            # out of place. v0 needed a staging SCAL of the whole vector first,
            # purely because it had three addresses and a box projection needs
            # four (limitation L2, ruling R2).
            D(Opcode.PROJ, f, a=self.t.spad_offset, b=self.lu.spad_offset,
              c=self.z.spad_offset, m=m, lda=1, ldb=2, ldc=1,
              label="z = clip(t, l, u)"),
            # y = rho*(t - z)
            self._scale_by_rho(self.t, self.y, m, "y = rho*t"),
            self._scale_by_rho(self.z, self.y, m, "y -= rho*z",
                               negate=True, accumulate=True),
        ]

    def _residual_descriptors(self) -> list[Descriptor]:
        """Residuals and the scaling terms OSQP's stopping rule needs."""
        n, m, f = self.n, self.m, self.fmt
        red = self.cfg.termination.opcode
        D = Descriptor

        def reduce_(dest: Buffer, src: Buffer, length: int, label: str) -> Descriptor:
            # DOT wants two vectors and squares the result; NRMI wants one.
            if red is Opcode.DOT:
                return D(Opcode.DOT, f, a=dest.spad_offset, b=src.spad_offset,
                         c=src.spad_offset, m=length, label=label)
            return D(Opcode.NRMI, f, a=dest.spad_offset, b=src.spad_offset,
                     m=length, label=label)

        return [
            D(Opcode.GEMV, f, a=self.A.spad_offset, b=self.x.spad_offset,
              c=self.Ax.spad_offset, m=m, k=n, lda=n, alpha=1.0, label="Ax = A*x"),
            reduce_(self.s_Ax, self.Ax, m, "|Ax|"),
            reduce_(self.s_z, self.z, m, "|z|"),
            D(Opcode.SCAL, f, b=self.Ax.spad_offset, c=self.rp.spad_offset,
              m=m, alpha=1.0, label="rp = Ax"),
            D(Opcode.AXPY, f, b=self.z.spad_offset, c=self.rp.spad_offset,
              m=m, alpha=-1.0, label="rp -= z"),
            reduce_(self.s_rp, self.rp, m, "|Ax - z|  primal residual"),
            D(Opcode.GEMV, f, a=self.P.spad_offset, b=self.x.spad_offset,
              c=self.Px.spad_offset, m=n, k=n, lda=n, alpha=1.0,
              structure=Structure.SYMMETRIC, label="Px = P*x"),
            reduce_(self.s_Px, self.Px, n, "|Px|"),
            D(Opcode.GEMV, f, a=self.A.spad_offset, b=self.y.spad_offset,
              c=self.Aty.spad_offset, m=m, k=n, lda=n, alpha=1.0,
              transpose=Transpose.A, label="Aty = A'*y"),
            reduce_(self.s_Aty, self.Aty, n, "|A'y|"),
            D(Opcode.SCAL, f, b=self.Px.spad_offset, c=self.rd.spad_offset,
              m=n, alpha=1.0, label="rd = Px"),
            D(Opcode.AXPY, f, b=self.q.spad_offset, c=self.rd.spad_offset,
              m=n, alpha=1.0, label="rd += q"),
            D(Opcode.AXPY, f, b=self.Aty.spad_offset, c=self.rd.spad_offset,
              m=n, alpha=1.0, label="rd += A'y"),
            reduce_(self.s_rd, self.rd, n, "|Px + q + A'y|  dual residual"),
            reduce_(self.s_q, self.q, n, "|q|"),
        ]

    # ------------------------------------------------------------------

    def _build_solver(self) -> SolverProgram:
        chains = {}
        for key, descs in (
            ("setup", self._setup_descriptors()),
            ("iteration", self._iteration_descriptors()),
            ("residual", self._residual_descriptors()),
        ):
            chains[key] = auto_chain(descs)
        self.chain_info = chains
        return SolverProgram(
            name=self.name,
            setup=Program(f"{self.name}.setup", chains["setup"].descriptors),
            iteration=Program(f"{self.name}.iteration",
                              chains["iteration"].descriptors),
            residual=Program(f"{self.name}.residual", chains["residual"].descriptors),
            converged=self._converged,
            max_iterations=self.cfg.max_iterations,
            check_every=self.cfg.check_every,
        )

    # -- the sequencer's convergence branch ----------------------------
    #
    # A handful of scalar instructions on the sequencer, reading scratchpad. Not
    # a descriptor: descriptor v0 has no compare and no branch, and it should not
    # -- data-dependent control flow is what the sequencer is for.

    def _scalar(self, buf: Buffer) -> float:
        return float(buf.spad_array.reshape(-1)[0])

    def _converged(self) -> bool:
        cfg = self.cfg
        if cfg.termination is TerminationNorm.LINF:
            r_prim = self._scalar(self.s_rp)
            r_dual = self._scalar(self.s_rd)
            eps_p = cfg.eps_abs + cfg.eps_rel * max(
                self._scalar(self.s_Ax), self._scalar(self.s_z))
            eps_d = cfg.eps_abs + cfg.eps_rel * max(
                self._scalar(self.s_Px), self._scalar(self.s_Aty),
                self._scalar(self.s_q))
        else:
            # DOT returns squared 2-norms; one sqrt each on the FMA cluster.
            r_prim = math.sqrt(max(self._scalar(self.s_rp), 0.0))
            r_dual = math.sqrt(max(self._scalar(self.s_rd), 0.0))
            eps_p = cfg.eps_abs * math.sqrt(self.m) + cfg.eps_rel * max(
                math.sqrt(max(self._scalar(self.s_Ax), 0.0)),
                math.sqrt(max(self._scalar(self.s_z), 0.0)))
            eps_d = cfg.eps_abs * math.sqrt(self.n) + cfg.eps_rel * max(
                math.sqrt(max(self._scalar(self.s_Px), 0.0)),
                math.sqrt(max(self._scalar(self.s_Aty), 0.0)),
                math.sqrt(max(self._scalar(self.s_q), 0.0)))
        return r_prim <= eps_p and r_dual <= eps_d

    # ------------------------------------------------------------------
    # Host-side interface
    # ------------------------------------------------------------------

    def set_problem(self, P, A) -> None:
        """Load the structurally stable problem data. Bumps their generations."""
        self.P.write(np.asarray(P, dtype=np.float64))
        self.A.write(np.asarray(A, dtype=np.float64))
        self.sigma_s.write(np.array([self.cfg.sigma], dtype=np.float64))
        # rho at STORAGE precision, and its reciprocal taken from the value that
        # descriptor word 7 would have carried, so the setup GEMM's fp32 alpha and
        # the iteration's scratchpad scalar are the same number. That consistency
        # is the whole point of ruling R5: the residual floor is set by
        # rho * (1/rho) - 1, and mixing an fp32 rho with an fp64 reciprocal of the
        # unrounded value would reintroduce the 1e-8 mismatch it exists to remove.
        r = float(np.float32(self.cfg.rho))
        self.rho_s.write(np.array([r], dtype=np.float64))
        self.rho_inv_s.write(np.array([1.0 / r], dtype=np.float64))

    def set_instance(self, q, l, u) -> None:
        """Load the per-tick data: cost vector and bounds. Bumps generations.

        The bounds are interleaved into one (m, 2) buffer here. That is a codegen
        detail, not an API change: the signature is still ``(q, l, u)``, and the
        host pays two strided stores per constraint instead of two contiguous
        ones, which is the same store count.
        """
        self.q.write(np.asarray(q, dtype=np.float64))
        lu = np.empty((self.m, 2), dtype=np.float64)
        lu[:, 0] = np.asarray(l, dtype=np.float64)
        lu[:, 1] = np.asarray(u, dtype=np.float64)
        self.lu.write(lu)

    def reset_iterates(self) -> None:
        """Cold start: zero x, y, z. Omit this to warm start from the last solve."""
        for buf in (self.x, self.y, self.z):
            buf.write(np.zeros(buf.shape, dtype=np.float64))

    def solve(self, *, warm: bool = True, outputs: Sequence[Buffer] | None = None):
        """Run one solve. Returns :class:`argmind.device.CallStats`."""
        if not warm:
            self.reset_iterates()
        return self.device.solve(
            self.built,
            dirty=(self.q, self.lu),
            outputs=list(outputs) if outputs is not None else [self.x],
        )

    @property
    def solution(self) -> np.ndarray:
        """The primal solution, from host memory. Requires x to be written back."""
        return np.asarray(self.x.read(), dtype=np.float64)

    # -- reporting ------------------------------------------------------

    def descriptor_counts(self) -> dict[str, int]:
        return {
            "setup": len(self.solver.setup),
            "iteration": len(self.solver.iteration),
            "residual": len(self.solver.residual),
        }

    def chain_report(self) -> str:
        lines = []
        for key in ("setup", "iteration", "residual"):
            c = self.chain_info[key]
            lines.append(
                f"  {key:<10} {len(c.descriptors):>3} desc  "
                f"emitted chain depth {c.chain_depth:>3} vs true DAG depth "
                f"{c.true_depth:>3} ({c.depth_penalty:.2f}x)  "
                f"policy={c.policy:<7} "
                f"{len(c.multi_producer):>3} multi-producer  "
                f"{c.forced_serialisations:>3} forced"
            )
        return "\n".join(lines)
