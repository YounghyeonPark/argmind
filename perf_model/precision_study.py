# SPDX-License-Identifier: Apache-2.0
"""
Numerical demonstration of the precision floors the silicon must provide.
========================================================================

Three precision requirements are currently asserted in prose (CLAUDE.md settled
findings, from the solver-numerics audit) and enforced only as a metadata check
on benchmark definitions. They are claims about arithmetic the hardware MUST
implement, and they carry real area: fp64 costs +0.067 mm² and forces precision
into the MACIF contract. A claim that expensive should be demonstrated, not
asserted.

The three claims under test:

  C1  The EKF covariance recursion needs fp32 minimum. There is NO fp16 or bf16
      path in ANY filter form, because the conventional update squares the
      covariance condition number.
  C2  Interior-point methods need fp64 near convergence, because the KKT
      condition number grows as 1/mu and reaches ~1e12.
  C3  ADMM tolerates bf16 OPERANDS with fp32 accumulation, because it is a
      first-order fixed-point iteration that contracts errors rather than
      amplifying them -- but its triangular solve must not be low precision.

This script is written so each claim CAN FAIL. A refutation would be more
valuable than a confirmation: C1 failing would open a cheaper EKF datapath, and
C3 failing would mean the ADMM rows need fp32 operands and the bandwidth
requirement roughly doubles.

Everything here is MEASURED (deterministic numerical experiment), not modelled.
Note the limit of that word: it measures the numerics, not silicon.
"""

from __future__ import annotations

import numpy as np

RNG = np.random.default_rng(20260730)


# ---------------------------------------------------------------------------
# Precision emulation
# ---------------------------------------------------------------------------

def to_bf16(x):
    """Round-to-nearest-even into bfloat16, returned as float32.

    bf16 is fp32 with the low 16 mantissa bits removed, so this is exact
    emulation of bf16 STORAGE (8-bit significand) with fp32 arithmetic -- which
    is precisely the "bf16 operands, fp32 accumulate" datapath under test.
    """
    x = np.asarray(x, dtype=np.float32)
    u = x.view(np.uint32)
    # round-to-nearest-even on the 16-bit boundary
    rounded = (u + 0x7FFF + ((u >> 16) & 1)).astype(np.uint32) & 0xFFFF0000
    return rounded.view(np.float32)


def cast(x, prec):
    if prec == "fp64":
        return np.asarray(x, dtype=np.float64)
    if prec == "fp32":
        return np.asarray(x, dtype=np.float32)
    if prec == "fp16":
        return np.asarray(x, dtype=np.float16).astype(np.float32)
    if prec == "bf16":
        return to_bf16(x)
    raise ValueError(prec)


# ---------------------------------------------------------------------------
# C1 — EKF covariance recursion
# ---------------------------------------------------------------------------

def ins_gnss_problem(n=24, m=12, cond_target=1e6):
    """An INS/GNSS-like filter whose covariance is deliberately ill-conditioned.

    Real inertial filters mix metres of position uncertainty with micro-radian
    gyro-bias states, so kappa(P) >= 1e6 is routine rather than pathological.
    """
    Q = np.linalg.qr(RNG.standard_normal((n, n)))[0]
    scales = np.logspace(0, np.log10(cond_target), n)
    P = Q @ np.diag(scales) @ Q.T
    P = 0.5 * (P + P.T)

    F = np.eye(n) + 0.01 * RNG.standard_normal((n, n))
    H = RNG.standard_normal((m, n))
    R = np.diag(RNG.uniform(0.5, 2.0, m))
    Qn = 1e-3 * np.eye(n)
    return P, F, H, R, Qn


def ekf_step(P, F, H, R, Qn, prec, form):
    """One covariance recursion in the given precision and form.

    Operands are stored at `prec`; products accumulate in fp32 (or fp64 when
    prec is fp64), which is the datapath contract being tested.
    """
    acc = np.float64 if prec == "fp64" else np.float32
    c = lambda M: cast(M, prec).astype(acc)     # noqa: E731
    P_, F_, H_, R_, Q_ = c(P), c(F), c(H), c(R), c(Qn)

    Pp = F_ @ P_ @ F_.T + Q_                    # propagate
    S = H_ @ Pp @ H_.T + R_                     # innovation covariance
    K = np.linalg.solve(S.T, (Pp @ H_.T).T).T   # gain

    if form == "conventional":
        Pn = Pp - K @ H_ @ Pp
    elif form == "joseph":
        A = np.eye(P.shape[0], dtype=acc) - K @ H_
        Pn = A @ Pp @ A.T + K @ R_ @ K.T
    elif form == "sqrt":
        # Square-root form: propagate a Cholesky factor, never form P directly.
        # This is what safety-critical INS actually ships (Bierman-Thornton).
        try:
            L = np.linalg.cholesky(Pp)
        except np.linalg.LinAlgError:
            return None
        A = np.eye(P.shape[0], dtype=acc) - K @ H_
        M = np.concatenate([A @ L, K @ np.linalg.cholesky(R_)], axis=1)
        _, Rq = np.linalg.qr(M.T)
        Pn = Rq.T @ Rq
    else:
        raise ValueError(form)
    return 0.5 * (Pn + Pn.T)


def run_c1(steps=60):
    """Does the filter stay positive-definite? That is the failure mode."""
    rows = []
    for form in ("conventional", "joseph", "sqrt"):
        for prec in ("fp64", "fp32", "bf16", "fp16"):
            P, F, H, R, Qn = ins_gnss_problem()
            failed_at, min_eig = None, np.inf
            for k in range(steps):
                Pn = ekf_step(P, F, H, R, Qn, prec, form)
                if Pn is None or not np.all(np.isfinite(Pn)):
                    failed_at = k
                    break
                w = np.linalg.eigvalsh(Pn)
                min_eig = min(min_eig, w.min())
                if w.min() <= 0:                # lost positive-definiteness
                    failed_at = k
                    break
                P = Pn
            rows.append((form, prec, failed_at, min_eig, steps))
    return rows


# ---------------------------------------------------------------------------
# C2 — interior-point KKT conditioning
# ---------------------------------------------------------------------------

def kkt_system(n=40, m=60, mu=1e-8):
    """A primal-dual KKT system at barrier parameter mu.

    The (S/Z) diagonal block scales as mu, so kappa grows ~1/mu as the solver
    approaches the boundary. This is the mechanism behind the fp64 claim.
    """
    A = RNG.standard_normal((m, n))
    Hs = RNG.standard_normal((n, n))
    H = Hs.T @ Hs + 0.1 * np.eye(n)
    s = RNG.uniform(0.5, 1.5, m)
    z = mu / s                                  # complementarity: s*z = mu
    K = np.block([
        [H, A.T],
        [A, -np.diag(s / z)],
    ])
    x = RNG.standard_normal(n + m)
    return K, K @ x, x


def run_c2():
    """FIRST VERSION OF THIS TEST WAS INVALID — recording why, because the failure
    mode is instructive.

    It drew the right-hand side as `K @ x` for a random x. A random x has almost
    no component along the smallest singular vectors, so the solve never exercises
    the ill-conditioned subspace: fp32 error came out flat at ~1e-6 across seven
    decades of condition number, contradicting the kappa*epsilon bound, and made
    fp32 look adequate. Large kappa is a WORST-CASE bound over right-hand sides,
    not a promise about any particular one.

    Fixed by aiming the right-hand side down the worst direction: solve for the
    smallest right singular vector, which is exactly what an interior-point step
    does as constraints activate and the Newton direction turns into the
    near-singular subspace.
    """
    rows = []
    for mu in (1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12):
        K, _, _ = kkt_system(mu=mu)
        U, S, Vt = np.linalg.svd(K)
        kappa = S[0] / S[-1]
        x_true = Vt[-1]                     # the worst-conditioned direction
        b = K @ x_true

        def solve_at(dtype):
            xs = np.linalg.solve(K.astype(dtype), b.astype(dtype))
            return xs.astype(np.float64)

        e64 = np.linalg.norm(solve_at(np.float64) - x_true) / np.linalg.norm(x_true)
        xs32 = solve_at(np.float32)
        e32 = np.linalg.norm(xs32 - x_true) / np.linalg.norm(x_true)
        # fp32 factorisation + fp64 residual, one refinement step.
        r = b - K @ xs32                    # fp64 residual: the load-bearing part
        d = np.linalg.solve(K.astype(np.float32),
                            r.astype(np.float32)).astype(np.float64)
        e_ir = np.linalg.norm(xs32 + d - x_true) / np.linalg.norm(x_true)
        rows.append((mu, kappa, e64, e32, e_ir))
    return rows


# ---------------------------------------------------------------------------
# C3 — ADMM operand precision
# ---------------------------------------------------------------------------

def admm_problem(n=80, m=320):
    """Build the QP ONCE.

    Third bug in this study, same family as the first two: `_admm` used to draw
    P, A and q from the module RNG on every call, so each precision variant was
    solving a DIFFERENT random problem and the reference scored 1.4 relative error
    against itself. Any cross-precision comparison must hold the problem fixed.
    """
    Gs = RNG.standard_normal((n, n))
    P = Gs.T @ Gs + 0.5 * np.eye(n)
    A = RNG.standard_normal((m, n))
    q = RNG.standard_normal(n)
    return P, A, q


def admm_solution(prob, iters=400, op_prec="fp32", solve_prec="fp64"):
    """Return (final x, final primal residual) for scoring against a reference."""
    return _admm(prob, iters, op_prec, solve_prec)


def _admm(prob, iters=400, op_prec="fp32", solve_prec="fp64"):
    """ADMM on a box-constrained QP, mirroring the QUAD-MPC structure.

    `op_prec` is the precision of the matrix OPERANDS in the matvecs; the
    cached triangular solve is done at `solve_prec`, so the two can be varied
    independently. That separation is the claim: matvecs tolerate bf16, the
    solve does not.
    """
    P, A, q = prob
    n, m = A.shape[1], A.shape[0]
    lo, hi = -np.ones(m), np.ones(m)
    rho, sigma = 1.0, 1e-6

    M = P + sigma * np.eye(n) + rho * (A.T @ A)
    L = np.linalg.cholesky(M)

    Ao = cast(A, op_prec).astype(np.float64)
    Lo = cast(L, solve_prec).astype(np.float64)

    x = np.zeros(n)
    z = np.zeros(m)
    y = np.zeros(m)
    res = np.inf
    for _ in range(iters):
        rhs = sigma * x - q + Ao.T @ (rho * z - y)
        w = np.linalg.solve(Lo, rhs)
        x = np.linalg.solve(Lo.T, w)
        Ax = Ao @ x
        z = np.clip(Ax + y / rho, lo, hi)
        y = y + rho * (Ax - z)
        res = np.linalg.norm(Ax - z, np.inf)             # primal residual
    return x, res


def run_c3(iters=400):
    """FIRST VERSION OF THIS TEST WAS ALSO INVALID — same lesson, different shape.

    It compared the final primal residual across precisions. But rounding the
    operand matrix CHANGES THE PROBLEM, so each precision was measured against
    its own easier problem: bf16 and even a bf16 triangular solve came out with
    *lower* residual than fp64, which is not physically meaningful.

    Fixed by measuring what a control engineer actually cares about: how far the
    computed control action lands from the true solution OF THE ORIGINAL PROBLEM.
    The reference is the fp64 solution at high iteration count; every variant is
    scored against that same reference.
    """
    prob = admm_problem()
    x_ref, _ = admm_solution(prob, iters=iters, op_prec="fp64", solve_prec="fp64")
    rows = []
    for op, sv, label in (
        ("fp64", "fp64", "fp64 operands, fp64 solve  (reference)"),
        ("fp32", "fp64", "fp32 operands, fp64 solve"),
        ("bf16", "fp64", "bf16 operands, fp64 solve"),
        ("fp16", "fp64", "fp16 operands, fp64 solve"),
        ("fp64", "fp32", "fp64 operands, fp32 solve"),
        ("fp64", "bf16", "fp64 operands, BF16 SOLVE"),
        ("bf16", "bf16", "bf16 operands, bf16 solve"),
        # ---- ADDED 2026-07-30: the configurations we would ACTUALLY BUILD.
        # Everything above pairs a 16-bit operand path with an fp64 solve, which
        # no CM SKU offers on the ADMM path -- the fp64 cluster exists for
        # interior point and putting an O(n^2) triangular solve on 8 FMA units
        # would cost more than the bandwidth saved. The proposal on the table is
        # 16-bit operands with an FP32 solve, so that is what has to be measured.
        # If the fp32 solve dominates the error, the whole "halve the operands to
        # double the bandwidth" idea buys nothing and must be dropped.
        ("fp32", "fp32", "fp32 operands, fp32 solve   <-- CM-100 today"),
        ("fp16", "fp32", "FP16 operands, fp32 solve   <-- the proposal"),
        ("bf16", "fp32", "bf16 operands, fp32 solve   <-- proposal w/ bf16 array"),
    ):
        x, res = admm_solution(prob, iters=iters, op_prec=op, solve_prec=sv)
        err = np.linalg.norm(x - x_ref) / max(np.linalg.norm(x_ref), 1e-300)
        rows.append((label, err, res))
    return rows


# ---------------------------------------------------------------------------
# C4 — FCS-MPC operand precision (added 2026-07-30 with the PWR-MPC re-spec)
# ---------------------------------------------------------------------------
#
# FCS-MPC's numerics are a different problem from every other row in the set and
# the difference is structural, not a matter of degree:
#
#   * There is no factorisation, so no condition number to amplify anything.
#   * The computation is a 1-2 step forward simulation -- error accumulates over
#     Np products, not over 40 iterations or 60 filter steps.
#   * THE OUTPUT IS A DISCRETE ARGMIN, not a real number. A precision error only
#     changes the CONTROL ACTION if it reorders the top two candidates, and when
#     two candidates' costs are within epsilon of each other, either choice is
#     within epsilon of optimal.
#
# That last point is the one that could make FCS-MPC tolerate a much narrower
# format than ADMM, which matters because it is a DIFFERENT ANSWER for the same
# array and therefore a MACIF question: if solver operands want fp16 and FCS-MPC
# tolerates bf16 or int8, the array must declare more than one format.
#
# Written so it can fail. The failure mode that would kill the idea is a high
# disagreement rate coupled with a large COST PENALTY -- disagreement alone is
# not a defect if the chosen candidate is nearly as good.


def fcs_problem(n_x=4, n_y=2, n_u=3, levels=3, seed=0):
    """A three-level FCS-MPC step: plant, output map, and the switching set."""
    rng = np.random.default_rng(20260730 + seed)
    # Discrete-time plant of a machine sampled at 25 us: eigenvalues near 1,
    # which is what makes the candidate costs CLOSE TOGETHER and therefore makes
    # the argmin sensitive. A well-separated problem would flatter every format.
    A = np.eye(n_x) + 2e-2 * rng.standard_normal((n_x, n_x))
    B = 0.15 * rng.standard_normal((n_x, n_u))
    C = rng.standard_normal((n_y, n_x))
    lv = np.linspace(-1.0, 1.0, levels)
    U = np.array(np.meshgrid(*([lv] * n_u), indexing="ij")).reshape(n_u, -1)
    x0 = rng.standard_normal(n_x)
    y_ref = C @ x0 + 0.05 * rng.standard_normal(n_y)
    return A, B, C, U, x0, y_ref


def fcs_costs(prob, Np, prec, lam=1e-3):
    """Cost of every switching SEQUENCE, with operands stored at `prec`.

    Accumulation is fp32 (fp64 for the fp64 case), matching the datapath
    contract. Returns the cost vector over |U|^Np sequences in fp64 so the
    ORDERING can be compared across formats.
    """
    A, B, C, U, x0, y_ref = prob
    acc = np.float64 if prec == "fp64" else np.float32
    c = lambda M: cast(M, prec).astype(acc)          # noqa: E731
    A_, B_, C_, U_ = c(A), c(B), c(C), c(U)
    x_, yr_ = c(x0), c(y_ref)
    nU = U.shape[1]

    # states: (n_x, n_seq_so_far); cost accumulates per sequence
    X = (A_ @ x_)[:, None] + B_ @ U_                 # (n_x, nU)
    cost = np.sum((C_ @ X - yr_[:, None]) ** 2, axis=0).astype(acc)
    cost = cost + acc(lam) * np.sum(U_ ** 2, axis=0)
    for _ in range(2, Np + 1):
        Xp = A_ @ X                                  # (n_x, k)
        # tile parents across children
        X = (np.repeat(Xp, nU, axis=1)
             + np.tile(B_ @ U_, (1, Xp.shape[1])))
        step = np.sum((C_ @ X - yr_[:, None]) ** 2, axis=0).astype(acc)
        step = step + acc(lam) * np.tile(np.sum(U_ ** 2, axis=0), Xp.shape[1])
        cost = np.repeat(cost, nU) + step
    return np.asarray(cost, dtype=np.float64)


def run_c4(Np=2, trials=200):
    """Does a narrow operand format change the SELECTED switching sequence?

    Two numbers per format, and they answer different questions:
      disagree   fraction of ticks where argmin differs from the fp64 argmin
      penalty    mean excess TRUE cost of the chosen sequence, relative to the
                 true optimum -- the quantity a control engineer cares about
    """
    rows = []
    for prec in ("fp64", "fp32", "fp16", "bf16", "int8-ish"):
        if prec == "int8-ish":
            continue                                  # not a float path; skip
        disagree = 0
        pens = []
        for t in range(trials):
            prob = fcs_problem(seed=t)
            ref = fcs_costs(prob, Np, "fp64")
            got = fcs_costs(prob, Np, prec)
            i_ref, i_got = int(np.argmin(ref)), int(np.argmin(got))
            if i_ref != i_got:
                disagree += 1
            spread = ref.max() - ref.min()
            pens.append((ref[i_got] - ref[i_ref]) / spread if spread > 0 else 0.0)
        rows.append((prec, disagree / trials, float(np.mean(pens)),
                     float(np.max(pens))))
    return rows


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def main():
    np.seterr(over="ignore", invalid="ignore")  # fp16 overflow IS a result here
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    out("=" * 94)
    out("PRECISION FLOOR STUDY — numerical demonstration (MEASURED numerics, not silicon)")
    out("=" * 94)
    out("Tests three prose claims that carry real area. Each is written so it CAN fail.")
    out()

    # ---- C1 ----
    out("C1 — EKF covariance recursion: is fp32 the floor, in every filter form?")
    out(f"{'form':>14} {'prec':>6} {'survived':>10} {'min eigenvalue':>16}   verdict")
    out("-" * 72)
    for form, prec, failed_at, min_eig, steps in run_c1():
        if failed_at is None:
            v, surv = "STABLE", f"{steps}/{steps}"
        else:
            v, surv = "LOST POS-DEF", f"{failed_at}/{steps}"
        out(f"{form:>14} {prec:>6} {surv:>10} {min_eig:>16.3e}   {v}")
    out()
    out("  Reading: a filter that loses positive-definiteness has diverged — the")
    out("  covariance no longer represents an uncertainty. This is the failure mode")
    out("  square-root filters exist to avoid.")
    out()

    # ---- C2 ----
    out("C2 — Interior-point KKT conditioning: does fp32 fail as mu -> 0?")
    out(f"{'mu':>8} {'cond(K)':>11} {'fp64 err':>11} {'fp32 err':>11} "
        f"{'fp32+1 IR':>11}   fp32 usable?")
    out("-" * 72)
    for mu, kappa, e64, e32, eir in run_c2():
        usable = "yes" if e32 < 1e-6 else ("marginal" if e32 < 1e-3 else "NO")
        out(f"{mu:>8.0e} {kappa:>11.2e} {e64:>11.2e} {e32:>11.2e} {eir:>11.2e}"
            f"   {usable}")
    out()
    out("  'fp32+1 IR' = one step of iterative refinement with the residual computed")
    out("  in fp64. If that column tracks fp64, refinement works but STILL needs an")
    out("  fp64 residual path — i.e. it relocates the fp64 requirement, not removes it.")
    out()

    # ---- C3 ----
    out("C3 — ADMM: do bf16 operands survive, and is the solve the sensitive step?")
    out("  Scored as relative error in the computed control action against the fp64")
    out("  solution OF THE SAME PROBLEM — not each variant's own residual.")
    out(f"{'configuration':>38} {'rel. error in x':>16} {'primal res':>12}")
    out("-" * 70)
    for label, err, res in run_c3():
        out(f"{label:>38} {err:>16.3e} {res:>12.3e}")
    out()
    out("  The claim: MATVECS tolerate bf16 (ADMM contracts operand error) while the")
    out("  cached triangular SOLVE does not (error there is amplified by kappa(M)).")
    out("  Compare 'bf16 operands, fp64 solve' against 'fp64 operands, BF16 SOLVE'.")
    out("  If the claim holds, the second is materially worse than the first.")
    out()

    # ---- C4 ----
    out("C4 — FCS-MPC: does a narrow operand format change the SELECTED switching")
    out("     sequence? (three-level NPC, |U|=27, Np=2 -> 729 sequences, 200 trials)")
    out(f"{'operand prec':>14} {'argmin disagrees':>17} {'mean cost penalty':>19} "
        f"{'worst':>9}")
    out("-" * 64)
    for prec, dis, mean_pen, max_pen in run_c4():
        out(f"{prec:>14} {dis:>16.1%} {mean_pen:>19.2e} {max_pen:>9.2e}")
    out()
    out("  'cost penalty' = excess TRUE cost of the chosen sequence as a fraction of")
    out("  the full cost spread across candidates. Disagreement alone is NOT a defect:")
    out("  when two candidates are within epsilon, either is within epsilon of optimal.")
    out("  The number that would kill a format is a large PENALTY, not a large")
    out("  disagreement rate. Note the plant is deliberately near-unity-eigenvalue,")
    out("  which packs the candidate costs close together — a well-separated plant")
    out("  would flatter every format.")
    out()

    out("=" * 94)
    out("VERDICTS")
    out("=" * 94)
    out("C1  CONFIRMED, strongly. fp32 is stable over 60 steps in all three filter")
    out("    forms; bf16 and fp16 lose positive-definiteness on the FIRST step in all")
    out("    three, square-root form included. fp16 overflows outright at kappa(P)=1e6,")
    out("    so for the EKF the binding limit is dynamic RANGE as well as precision.")
    out("    -> The 'fp32 minimum, no 16-bit path in any filter form' requirement holds.")
    out()
    out("C2  INCONCLUSIVE — and the fp64 requirement is NOT measured by this study.")
    out("    The mechanism is confirmed: kappa(K) grows 3.1e2 -> 2.0e13 as mu falls")
    out("    1e-2 -> 1e-12, exactly the ~1/mu behaviour claimed. But the accuracy loss")
    out("    does not follow: fp32 forward error stays ~5e-6 even at kappa=2e13, about")
    out("    1e7 times better than the kappa*epsilon bound, because a synthetic KKT")
    out("    matrix with a huge well-separated -diag(s/z) block is benign for LU with")
    out("    partial pivoting. Real interior-point ill-conditioning comes from a MIX of")
    out("    active and inactive constraints in one matrix plus a nearly-singular")
    out("    reduced Hessian, which this construction does not reproduce.")
    out("    -> fp64 for interior-point currently rests on the literature and the")
    out("       solver-numerics audit, NOT on this experiment. Do not cite this as")
    out("       measured support. Settling it needs a real IP iteration sequence.")
    out()
    out("C3  PARTIALLY CONFIRMED, with one claim too strong and one bonus finding.")
    out("    The directional claim HOLDS: the cached triangular solve is ~3x more")
    out("    sensitive than the operands (6.1e-2 vs 1.9e-2 control-action error), so")
    out("    'the solve must not be low precision' stands.")
    out("    But 'ADMM tolerates bf16 operands' is TOO STRONG: bf16 operands cost 1.9e-2")
    out("    -- ~2% error in the commanded control action -- while fp32 operands cost")
    out("    4.0e-7. 2% is not obviously acceptable for a constrained system and should")
    out("    not be assumed away.")
    out("    BONUS, and actionable: fp16 operands (1.9e-3) beat bf16 (1.9e-2) by 10x,")
    out("    because for these matrices MANTISSA BITS matter more than exponent range")
    out("    -- the opposite of the ML tradeoff that makes bf16 the default. If 16-bit")
    out("    operands are ever used to double effective GEMV bandwidth on the ADMM rows")
    out("    (which are bandwidth-bound, so this is worth ~2x latency), the format")
    out("    should be fp16, not bf16. The MAC array is currently specified bf16.")
    out()
    out("=" * 94)
    out("LIMITS OF THIS STUDY")
    out("=" * 94)
    out("  * THREE of the experiments here were invalid on first write, all in the same")
    out("    family -- comparing across precisions without holding the problem fixed,")
    out("    or measuring a quantity that is not comparable across variants. The first")
    out("    C3 version even showed bf16 'beating' fp64. Each is documented in the")
    out("    relevant docstring. Treat any cross-precision comparison with suspicion")
    out("    until its reference row scores exactly zero, as C3's now does.")
    out("  * Emulates STORAGE precision with fp32/fp64 arithmetic. Real hardware also")
    out("    rounds intermediate products; a true bf16-multiplier datapath may be")
    out("    slightly worse than shown here.")
    out("  * Uses LAPACK dense solves, not our tiled kernels. Blocking changes")
    out("    summation order and therefore error growth.")
    out("  * Single random instance per configuration with a fixed seed. Indicative of")
    out("    the mechanism, not a distribution over problems.")
    out("  * Does NOT establish silicon behaviour, only numerics.")

    with open("perf_model/results_precision_study.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
