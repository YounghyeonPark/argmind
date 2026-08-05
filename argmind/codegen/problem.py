"""The problem description a code generator consumes, and the options it takes.

A :class:`QPProblem` is the *whole* input to codegen: the numerical data, plus one
piece of structural information the numbers cannot carry -- **which blocks of that
data change from one call to the next**. Everything else the generator needs
(dependency structure, buffer lifetimes, what may stay resident, what must be
re-fetched, how many descriptors, where the tokens go) is derived.

Why ``varies`` is a required part of the description, not a hint
---------------------------------------------------------------
For an embedded MPC the condensed QP splits cleanly: ``P`` and ``A`` are functions
of the plant model and the horizon, so they are constant for the life of the
controller, while ``q``, ``l`` and ``u`` are functions of the current state and
reference and change every tick. That split is the entire basis of the residency
argument -- it is what makes the Cholesky factor a once-ever cost and takes the
per-tick transfer from the whole working set down to two vectors.

**It is not recoverable from the arrays.** Two problems with identical ``P`` and
``A`` can differ in whether ``P`` is about to change. So the description carries
it, and the default is the *conservative* one: everything varies, no residency
win, the factorisation re-runs every call. A generator that guessed "P is
probably constant" would silently produce a wrong answer the first time a user
retuned a weight, so it does not guess.

The CVXPY front end (:mod:`argmind.codegen.frontend`) derives ``varies``
automatically, because a CVXPY ``Parameter`` *is* the declaration -- see that
module.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Iterable, Sequence

import numpy as np

from ..errors import ArgmindError
from ..precision import Format

#: The data blocks of a QP in OSQP's form. Order is the order they are described
#: in, and is part of the report.
QP_BLOCKS: tuple[str, ...] = ("P", "q", "A", "l", "u")


class CodegenError(ArgmindError):
    """A problem or option set the generator refuses to compile."""


class UnexpressiblePrecision(CodegenError):
    """The requested per-buffer formats cannot be encoded in one descriptor.

    Descriptor v0.1 carries exactly two formats (``prec_m`` for the bulk/matrix
    operand, ``prec_v`` for vectors and results), and for the elementwise opcodes
    it effectively carries **one**, because they have no bulk operand. A code
    generator that assigns formats per buffer therefore finds the boundary of the
    format automatically, which is the point of assigning them per buffer.
    """


class ContractViolation(CodegenError):
    """The generator refused to emit a program that breaks the codegen contract.

    Refuse, not warn: every rule in the contract exists because breaking it
    produced either a silently wrong answer or a silently missed deadline.
    """


@dataclass(frozen=True)
class QPProblem:
    """A convex QP in OSQP's form.

    ::

        minimize    (1/2) x' P x + q' x
        subject to  l <= A x <= u

    Dense arrays: the CM's scratchpad is a flat address space with base +
    leading-dimension addressing and no sparse index machinery, so a sparse
    problem is densified at code-generation time or not accepted at all. That is
    a real scope limit and it is stated rather than worked around -- condensed
    MPC, which is the beachhead, is genuinely dense (measured density(P) = 1.00 on
    QUAD-MPC).

    Parameters
    ----------
    varies:
        Names from :data:`QP_BLOCKS` that change from call to call. Defaults to
        **all of them**, i.e. no residency assumption whatsoever.
    origin:
        Provenance string for the report: ``"arrays"``, ``"cvxpy"``, ...
    """

    P: np.ndarray
    q: np.ndarray
    A: np.ndarray
    l: np.ndarray
    u: np.ndarray
    varies: frozenset[str] = frozenset(QP_BLOCKS)
    name: str = "qp"
    origin: str = "arrays"
    #: Free-form notes the front end wants carried into the report.
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "P", np.ascontiguousarray(self.P, dtype=np.float64))
        object.__setattr__(self, "A", np.ascontiguousarray(self.A, dtype=np.float64))
        for nm in ("q", "l", "u"):
            object.__setattr__(
                self, nm,
                np.ascontiguousarray(getattr(self, nm), dtype=np.float64).ravel())
        object.__setattr__(self, "varies", frozenset(self.varies))
        self._validate()

    def _validate(self) -> None:
        if self.P.ndim != 2 or self.P.shape[0] != self.P.shape[1]:
            raise CodegenError(f"P must be square, got shape {self.P.shape}")
        n = self.P.shape[0]
        if self.A.ndim != 2 or self.A.shape[1] != n:
            raise CodegenError(
                f"A must have {n} columns to match P, got shape {self.A.shape}")
        m = self.A.shape[0]
        for nm, want in (("q", n), ("l", m), ("u", m)):
            v = getattr(self, nm)
            if v.shape != (want,):
                raise CodegenError(
                    f"{nm} must have length {want}, got {v.shape}")
        asym = float(np.max(np.abs(self.P - self.P.T))) if n else 0.0
        pmax = float(np.max(np.abs(self.P))) if n else 0.0
        scale = max(pmax, 1.0)
        if asym > 1e-10 * scale:
            raise CodegenError(
                f"P is not symmetric (max|P - P'| = {asym:.3e}). The GEMV that "
                f"forms P*x is emitted with structure=SYMMETRIC and reads one "
                f"triangle only, so an asymmetric P would be silently symmetrised "
                f"by the hardware. Symmetrise it yourself and the generator will "
                f"not have to guess which triangle you meant.")
        if np.any(self.l > self.u):
            bad = int(np.argmax(self.l - self.u))
            raise CodegenError(
                f"infeasible bounds at row {bad}: l = {self.l[bad]:.6g} > "
                f"u = {self.u[bad]:.6g}")
        for nm in ("P", "q", "A"):
            v = getattr(self, nm)
            if not np.all(np.isfinite(v)):
                raise CodegenError(f"{nm} contains non-finite entries")
        unknown = self.varies - set(QP_BLOCKS)
        if unknown:
            raise CodegenError(
                f"varies names unknown data blocks {sorted(unknown)}; "
                f"known blocks are {list(QP_BLOCKS)}")

    # -- shape ------------------------------------------------------------

    @property
    def n_var(self) -> int:
        return int(self.P.shape[0])

    @property
    def n_con(self) -> int:
        return int(self.A.shape[0])

    @property
    def structural(self) -> tuple[str, ...]:
        """Blocks that do NOT change call to call -- the residency opportunity."""
        return tuple(b for b in QP_BLOCKS if b not in self.varies)

    @property
    def instance(self) -> tuple[str, ...]:
        return tuple(b for b in QP_BLOCKS if b in self.varies)

    @property
    def refactorises_every_call(self) -> bool:
        """True when the Cholesky cannot be hoisted out of the control loop.

        ``K = P + sigma*I + rho*A'A`` depends on P and A only, so this is exactly
        "does P or A vary". It is the single most consequential thing ``varies``
        decides, so it is named rather than left implicit.
        """
        return bool({"P", "A"} & self.varies)

    # -- constructors -----------------------------------------------------

    @classmethod
    def from_arrays(
        cls, P, q, A, l, u, *, varies: Iterable[str] = QP_BLOCKS,
        name: str = "qp", notes: Sequence[str] = (),
    ) -> "QPProblem":
        """The direct front door: hand the generator the five arrays.

        ``varies`` must be stated. For a condensed MPC QP it is
        ``("q", "l", "u")``; for a problem whose weights are retuned online it is
        larger; for a one-shot solve the default (everything) costs nothing.
        """
        return cls(P=P, q=q, A=A, l=l, u=u, varies=frozenset(varies),
                   name=name, origin="arrays", notes=tuple(notes))

    def with_instance(self, q=None, l=None, u=None) -> "QPProblem":
        """A copy with new per-tick data. Used to build validation variants."""
        return replace(
            self,
            q=self.q if q is None else np.asarray(q, dtype=np.float64).ravel(),
            l=self.l if l is None else np.asarray(l, dtype=np.float64).ravel(),
            u=self.u if u is None else np.asarray(u, dtype=np.float64).ravel(),
        )

    def summary(self) -> str:
        dp = np.count_nonzero(self.P) / max(self.n_var ** 2, 1)
        da = np.count_nonzero(self.A) / max(self.n_var * self.n_con, 1)
        return (
            f"{self.name}: n_var={self.n_var} n_con={self.n_con} "
            f"density(P)={dp:.2f} density(A)={da:.2f} "
            f"structural={list(self.structural)} per-call={list(self.instance)} "
            f"origin={self.origin}"
        )


# ---------------------------------------------------------------------------
# Precision policy
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PrecisionPolicy:
    """Per-buffer formats, from which descriptor precision fields are derived.

    **Formats belong to buffers, not to descriptors.** A descriptor's ``prec_m``
    and ``prec_v`` are then *derived* from the formats of the buffers it names,
    and a combination the format cannot encode is refused at code-generation
    time. Doing it the other way round -- picking descriptor fields and hoping
    the buffers agree -- is how limitation L4 stayed invisible: the conflict is a
    storage conflict between two descriptors, not an arithmetic one inside either.

    Three roles, because the measured numerics distinguish three:

    ``bulk``
        The matrices whose size scales with the largest product of dimensions
        (``A``, and ``P`` where it can be). The only operands worth narrowing.
    ``solve``
        The Cholesky factor and the triangular solves. PHASE0_FINDINGS 0.3
        measured the triangular solve ~3x more error-sensitive than the operands,
        and measured that keeping it at fp32 while operands go to fp16 costs
        **nothing**: fp16/fp32-solve and fp16/fp64-solve score identically.
    ``vector``
        Iterates, working vectors, bounds and convergence scalars.
    """

    bulk: Format = Format.FP32
    solve: Format = Format.FP32
    vector: Format = Format.FP32
    name: str = "fp32"

    @classmethod
    def uniform(cls, fmt: Format) -> "PrecisionPolicy":
        fmt = Format(fmt)
        return cls(bulk=fmt, solve=fmt, vector=fmt, name=f"all-{fmt}")

    @classmethod
    def fp16_operands(cls) -> "PrecisionPolicy":
        """The opt-in mixed mode: fp16 bulk operands, fp32 solve and vectors.

        Never a default. Worth 1.16-1.49x latency and 1.51-1.79x energy on the
        ADMM rows at ~2400x the operand error (8e-7 -> 1.9e-3, about 0.2% of a
        commanded control action), and it requires a macro that declares fp16 --
        **distinct from bf16**, which is 10x worse on this path at the same width.
        """
        return cls(bulk=Format.FP16, solve=Format.FP32, vector=Format.FP32,
                   name="fp16-operands/fp32-solve")

    @property
    def is_mixed(self) -> bool:
        return len({self.bulk, self.solve, self.vector}) > 1

    @property
    def uses_narrow_operands(self) -> bool:
        return self.bulk in (Format.FP16, Format.BF16, Format.INT8)

    def __str__(self) -> str:
        return self.name


class TerminationNorm(Enum):
    """Which norm the in-fabric convergence test uses.

    ``LINF`` reproduces OSQP's own criterion and needs ``NRMI`` (descriptor v0.1
    ruling R1). ``L2`` is expressible in frozen v0 via ``DOT`` and is retained so
    the cost of the substitute stays measurable.
    """

    L2 = "l2"
    LINF = "linf"


@dataclass(frozen=True)
class CodegenOptions:
    """Everything the generator decides before it emits a descriptor.

    All of it is baked in: after :func:`argmind.codegen.generate` returns, no
    value here can change without regenerating, which is the determinism claim
    stated as a property of the tool rather than as an aspiration.
    """

    rho: float = 0.125
    sigma: float = 1e-6
    alpha: float = 1.6
    eps_abs: float = 1e-6
    eps_rel: float = 1e-6
    max_iterations: int = 200
    #: Evaluate the residual block (and therefore the convergence test) every
    #: k-th iteration. **1 is correct for this workload**: QUAD-MPC converges in
    #: 6-14 iterations, so OSQP's library default of 25 means never checking.
    check_every: int = 1
    precision: PrecisionPolicy = field(default_factory=PrecisionPolicy)
    termination: TerminationNorm = TerminationNorm.LINF
    #: Carry rho and 1/rho as scratchpad scalars read by a zero-stride VMUL
    #: (descriptor v0.1 ruling R5) instead of as fp32 immediates in word 7.
    #: **Default on**: it is the encoding that makes the reciprocal mismatch
    #: ~1e-16 instead of 1.5e-8, and the mismatch is what stalls the solve.
    rho_in_scratchpad: bool = True
    #: Delete staging copies (contract rule C6). Default on. Turn it off to emit
    #: the reference-equivalent program the hand-written solver emits, which is
    #: how the two are compared descriptor for descriptor.
    elide_copies: bool = True
    #: fp16 bulk operands are an opt-in mode and this is the opt-in. A precision
    #: policy with narrow operands is refused without it.
    allow_narrow_operands: bool = False
    #: Emit the sequencer's convergence test in the infinity norm even on a
    #: strict-v0 device. Off: the generator refuses instead, which is the honest
    #: behaviour when the device cannot run what was asked for.
    name: str = "admm"

    def __post_init__(self) -> None:
        if self.check_every < 1:
            raise CodegenError("check_every must be >= 1")
        if self.max_iterations < 1:
            raise CodegenError("max_iterations must be >= 1")
        if self.rho <= 0.0 or self.sigma < 0.0:
            raise CodegenError("rho must be > 0 and sigma >= 0")
        if not 0.0 < self.alpha < 2.0:
            raise CodegenError(
                f"ADMM relaxation alpha must be in (0, 2); got {self.alpha}")

    # -- rho arithmetic, which is contract rule C1 ------------------------

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
        """``fl32(rho) * fl32(1/rho) - 1``. Zero iff rho is a power of two."""
        r = np.float32(self.rho)
        c = np.float32(1.0 / float(r))
        return float(np.float64(r) * np.float64(c) - 1.0)

    @property
    def rho_is_power_of_two(self) -> bool:
        r = float(self.rho)
        return r > 0.0 and np.frexp(r)[0] == 0.5
