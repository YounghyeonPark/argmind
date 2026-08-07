# SPDX-License-Identifier: Apache-2.0
"""Front ends: getting a :class:`~argmind.codegen.problem.QPProblem` out of a
problem a user already has.

The adoption on-ramp is that a user runs *their* problem, not a transcription of
it. CVXPY is the front door: it installs from PyPI with no compiler, it is what
control engineers prototype MPC in, and -- the part that matters here -- **it
already carries the one declaration the arrays cannot**.

Why a CVXPY ``Parameter`` is the residency declaration
------------------------------------------------------
The residency plan needs to know which QP data blocks change from call to call.
In CVXPY, data that changes between solves is a ``Parameter`` and data that does
not is a constant; that is not a convention, it is what DPP means. So the front
end derives ``varies`` by **canonicalising the problem at two different parameter
values and differencing the canonical data**. Whatever moved is per-instance;
whatever did not is structural.

That method is empirical rather than symbolic, and the tradeoff is stated rather
than hidden: it is robust across CVXPY versions (it uses only the public
``get_problem_data``), it cannot produce a false *structural* answer except on a
measure-zero coincidence of parameter draws, and it costs two canonicalisations at
code-generation time, which is free because code generation is offline. A
symbolic query of the parameter-affine map would be exact and would couple the
generator to CVXPY internals.

The two-sided-row recovery, and why it is not optional
-----------------------------------------------------
CVXPY canonicalises ``l <= Ax <= u`` into two one-sided rows, so a problem a user
would naturally write with 320 two-sided constraints arrives with 640 one-sided
ones -- and the constraint matrix is the largest object in the program. Codegen
recovers the two-sided form by pairing rows ``A_i = -A_j``, which on QUAD-MPC
recovers exactly the hand-derived 320-row form and halves both the dominant
matrix and the dominant GEMV. Without it, coming in through CVXPY costs 2x the
scratchpad and 2x the constraint-product cycles, which would make the on-ramp a
tax on using it.

CasADi is **not** wired up. It is the other natural front door for NMPC and it
would be a second extraction path onto the same ``QPProblem``, but its output is a
nonlinear problem whose QP subproblems come from an SQP loop -- that is the
ARM-MPC / interior-point path, not this one, and claiming it works when the SQP
outer loop is not generated would be a false claim about the on-ramp.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .problem import CodegenError, QPProblem

CVXPY_AVAILABLE: bool
try:  # pragma: no cover - exercised by both branches in CI, not both here
    import cvxpy as cp  # noqa: F401
    CVXPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    cp = None
    CVXPY_AVAILABLE = False

#: CasADi front end status. Recorded as data so a report can state it rather than
#: a reader having to infer it from the absence of a function.
CASADI_FRONTEND = {
    "status": "DEFERRED, not attempted",
    "why": "CasADi's natural output is a nonlinear program; its QP subproblems "
           "come from an SQP outer loop (the acados / HPIPM path). Generating "
           "the subproblem without the outer loop would not run a user's "
           "problem, so it would be a false claim about the on-ramp.",
    "what_it_needs": "the interior-point / SQP solver family, which is a "
                     "separate generator target and a separate validation "
                     "reference (acados, HPIPM).",
}


def require_cvxpy() -> None:
    if not CVXPY_AVAILABLE:
        raise CodegenError(
            "the CVXPY front end needs cvxpy installed (`pip install cvxpy`). "
            "The direct front door, QPProblem.from_arrays(P, q, A, l, u, "
            "varies=...), needs nothing beyond numpy and is fully supported.")


@dataclass
class ExtractionReport:
    """What the front end found, so the generated program can be explained."""

    n_var: int
    n_con_canonical: int
    n_con_merged: int
    n_eq: int
    parameters: tuple[str, ...]
    varies: tuple[str, ...]
    structural: tuple[str, ...]
    two_sided_pairs: int
    #: Bytes of constraint matrix saved by the two-sided recovery, at fp32.
    bytes_saved_fp32: int

    def text(self) -> str:
        return (
            f"cvxpy front end: n_var={self.n_var}, canonical constraint rows="
            f"{self.n_con_canonical} ({self.n_eq} equality), merged to "
            f"{self.n_con_merged} two-sided rows by pairing "
            f"{self.two_sided_pairs} (A_i = -A_j) row pairs -- "
            f"{self.bytes_saved_fp32} B of constraint matrix saved at fp32. "
            f"Parameters {list(self.parameters)} -> per-call blocks "
            f"{list(self.varies)}; structural {list(self.structural)}."
        )


def _osqp_form(data: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                    np.ndarray, np.ndarray, int]:
    """Assemble ``l <= Ax <= u`` from CVXPY's OSQP-target canonical data.

    CVXPY hands back the pieces (``P``, ``q``, equalities ``A x = b``,
    inequalities ``F x <= G``, optional variable bounds) rather than the stacked
    form, so the stacking convention lives here: equalities first with
    ``l = u = b``, then inequalities with ``l = -inf``, then any variable bounds
    as identity rows.
    """
    P = np.asarray(data["P"].todense(), dtype=np.float64)
    P = 0.5 * (P + P.T)
    q = np.asarray(data["q"], dtype=np.float64).ravel()
    n = P.shape[0]
    blocks, lows, ups = [], [], []
    Aeq = data.get("A")
    if Aeq is not None and Aeq.shape[0]:
        b = np.asarray(data["b"], dtype=np.float64).ravel()
        blocks.append(np.asarray(Aeq.todense(), dtype=np.float64))
        lows.append(b)
        ups.append(b.copy())
    n_eq = int(blocks[0].shape[0]) if blocks else 0
    F = data.get("F")
    if F is not None and F.shape[0]:
        G = np.asarray(data["G"], dtype=np.float64).ravel()
        blocks.append(np.asarray(F.todense(), dtype=np.float64))
        lows.append(np.full(G.shape[0], -np.inf))
        ups.append(G)
    lb, ub = data.get("lower_bounds"), data.get("upper_bounds")
    if lb is not None or ub is not None:
        blocks.append(np.eye(n))
        lows.append(np.full(n, -np.inf) if lb is None
                    else np.asarray(lb, dtype=np.float64).ravel())
        ups.append(np.full(n, np.inf) if ub is None
                   else np.asarray(ub, dtype=np.float64).ravel())
    if not blocks:
        raise CodegenError(
            "the problem has no constraints; the generated ADMM program would be "
            "an unconstrained least-squares solve, for which a single triangular "
            "solve is the right program and this generator is the wrong tool")
    A = np.vstack(blocks)
    return P, q, A, np.concatenate(lows), np.concatenate(ups), n_eq


def merge_two_sided_rows(
    A: np.ndarray, l: np.ndarray, u: np.ndarray, *, tol: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Recover ``l <= a'x <= u`` rows from pairs of one-sided rows.

    Two canonical rows ``a'x <= g1`` and ``-a'x <= g2`` are the single two-sided
    row ``-g2 <= a'x <= g1``. Pairing is done on an exact byte-level hash of the
    row and its negation, so it never merges rows that merely look similar; with
    ``tol > 0`` the rows are rounded before hashing, which trades exactness for
    reach and is off by default.

    Returns ``(A, l, u, n_pairs)``. Row order is preserved: the surviving row
    keeps the position of the first of each pair, so a caller comparing against a
    reference solver's row order can still do so.

    **Duplicate coefficient rows.** A condensed MPC matrix routinely contains
    repeated rows, and then which ``-a`` row a given ``+a`` row is paired with is
    arbitrary. That changes the individual rows but **not the feasible set**: all
    four rows constrain the same linear functional, so the intersection is
    ``[max of the lower bounds, min of the upper bounds]`` however they are
    paired. Measured on QUAD-MPC: 320 merged rows over 307 distinct coefficient
    vectors, and the intersected feasible set is identical to the hand-derived
    condensed QP's. Do not expect row-by-row equality with a reference
    formulation; do expect the same feasible set, and test for that.
    """
    m = A.shape[0]
    key = np.array(np.round(A / tol) * tol if tol > 0 else A, dtype=np.float64,
                   copy=True)
    # Canonicalise negative zero. Without this, a row containing a 0.0 entry
    # never matches its own negation, because -0.0 and +0.0 have different bytes
    # -- and on a sparse-ish condensed MPC matrix that is nearly every row, so the
    # pass would silently find almost no pairs and quietly cost 2x the footprint.
    key[key == 0.0] = 0.0

    def rowkey(row: np.ndarray) -> bytes:
        r = np.ascontiguousarray(row, dtype=np.float64).copy()
        r[r == 0.0] = 0.0
        return r.tobytes()

    index: dict[bytes, list[int]] = {}
    for i in range(m):
        index.setdefault(rowkey(key[i]), []).append(i)
    used = np.zeros(m, dtype=bool)
    keep: list[int] = []
    lo = l.astype(np.float64).copy()
    hi = u.astype(np.float64).copy()
    pairs = 0
    for i in range(m):
        if used[i]:
            continue
        used[i] = True
        keep.append(i)
        neg = rowkey(-key[i])
        partner = None
        for j in index.get(neg, ()):
            if j > i and not used[j]:
                partner = j
                break
        if partner is None:
            continue
        # row j is -row i, so l_j <= -a'x <= u_j  =>  -u_j <= a'x <= -l_j
        used[partner] = True
        lo[i] = max(lo[i], -hi[partner])
        hi[i] = min(hi[i], -lo[partner])
        pairs += 1
    keep_arr = np.array(keep, dtype=int)
    return (np.ascontiguousarray(A[keep_arr]), lo[keep_arr], hi[keep_arr], pairs)


def from_cvxpy(
    prob, *, name: str = "cvxpy", merge_two_sided: bool = True,
    probe_scale: float = 1.0, seed: int = 0,
) -> tuple[QPProblem, ExtractionReport]:
    """Extract a :class:`QPProblem` from a CVXPY problem, deriving ``varies``.

    The problem must be a QP that CVXPY can target at OSQP, and its
    call-to-call-varying data must be expressed as ``cp.Parameter`` objects --
    which is what a user writing a repeatedly-solved MPC already does, because
    that is how CVXPY caches canonicalisation.

    Only single-variable problems are supported, so that the canonical solution
    vector maps unambiguously onto the user's variable. That is a real limit and
    it is refused rather than guessed at: CVXPY stacks multiple variables in an
    order this front end has no business predicting.
    """
    require_cvxpy()
    variables = prob.variables()
    if len(variables) != 1:
        raise CodegenError(
            f"the CVXPY front end supports one Variable, found {len(variables)}. "
            f"CVXPY stacks multiple variables in a canonical order that the "
            f"generator would have to guess to unpack the solution, so this is "
            f"refused rather than approximated. Concatenate into one variable, or "
            f"use QPProblem.from_arrays with the stacking you intend.")
    params = list(prob.parameters())
    for p in params:
        if p.value is None:
            raise CodegenError(
                f"parameter {p.name()!r} has no value; codegen needs one "
                f"canonicalisation point to emit against")

    def extract() -> tuple:
        P, q, A, l, u, n_eq = _osqp_form(prob.get_problem_data(cp.OSQP)[0])
        rows = A.shape[0]
        pairs = 0
        if merge_two_sided:
            A, l, u, pairs = merge_two_sided_rows(A, l, u)
        return P, q, A, l, u, n_eq, rows, pairs

    P, q, A, l, u, n_eq, n_con_canonical, pairs = extract()

    # Derive `varies` by differencing the canonicalisation at a second parameter
    # point. Restore the user's values afterwards -- this must not be observable.
    #
    # The differencing happens on the MERGED form on purpose. Merging moves data
    # between l and u (row j = -row i contributes -u_j to l_i), so a problem whose
    # canonical l is entirely -inf -- which is every purely-inequality problem --
    # would be classified "l is structural" on the unmerged form and then have a
    # per-call l after merging. That is the kind of quiet mis-derivation that
    # leaves a stale bound resident in a control loop.
    varies: set[str] = set()
    if params:
        rng = np.random.default_rng(seed)
        saved = [np.array(p.value, copy=True) for p in params]
        try:
            for p in params:
                v = np.asarray(p.value, dtype=np.float64)
                step = probe_scale * (1.0 + np.abs(v)) * rng.standard_normal(v.shape)
                p.value = float(v + step) if v.ndim == 0 else v + step
            P2, q2, A2, l2, u2, _, _, _ = extract()
        finally:
            for p, s in zip(params, saved):
                p.value = s
        for nm, a, b in (("P", P, P2), ("q", q, q2), ("A", A, A2),
                         ("l", l, l2), ("u", u, u2)):
            if a.shape != b.shape or not _same(a, b):
                varies.add(nm)

    report = ExtractionReport(
        n_var=P.shape[0], n_con_canonical=n_con_canonical,
        n_con_merged=A.shape[0], n_eq=n_eq,
        parameters=tuple(p.name() for p in params),
        varies=tuple(sorted(varies)),
        structural=tuple(nm for nm in ("P", "q", "A", "l", "u")
                         if nm not in varies),
        two_sided_pairs=pairs,
        bytes_saved_fp32=(n_con_canonical - A.shape[0]) * P.shape[0] * 4,
    )
    problem = QPProblem(
        P=P, q=q, A=A, l=l, u=u, varies=frozenset(varies), name=name,
        origin="cvxpy",
        notes=(report.text(),),
    )
    return problem, report


def _same(a: np.ndarray, b: np.ndarray) -> bool:
    """Equality that treats matching infinities as equal."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    fa, fb = np.isfinite(a), np.isfinite(b)
    if not np.array_equal(fa, fb):
        return False
    if not np.array_equal(a[~fa], b[~fb]):
        return False
    return bool(np.allclose(a[fa], b[fb], rtol=0.0, atol=0.0))


__all__ = [
    "CASADI_FRONTEND", "CVXPY_AVAILABLE", "ExtractionReport", "from_cvxpy",
    "merge_two_sided_rows", "require_cvxpy",
]
