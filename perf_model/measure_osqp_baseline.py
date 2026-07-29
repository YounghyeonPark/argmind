"""
Measure REAL OSQP behaviour on the QUAD-MPC benchmark problem.
==============================================================

Phase-0 critical-path item #1 was "measure the real CPU baselines", which we called
blocked for lack of Arm hardware. Absolute timings indeed do not transfer from this
x86 host to a Cortex-A. But two things DO transfer, and both are currently *assumed*
in every latency number we have produced:

  1. **Iteration count** — architecture-independent. The model assumes 40 ADMM
     iterations for QUAD-MPC. That is a guess. This measures it, cold and warm-started.
  2. **Problem structure** — the constraint-row count and sparsity of a real OSQP
     formulation, which the model's `n_con = 2*n_var + 2*N*n_x` may have wrong.

Everything reported as a *time* here is labelled x86 and must not be compared to the
model's Arm baselines.

Problem: quadrotor MPC, 12 states, 4 inputs, horizon N=20, condensed to the control
sequence (80 decision variables), per BENCHMARKS.md #1.
"""

from __future__ import annotations

import time

import numpy as np
import osqp
import scipy.sparse as sp

RNG = np.random.default_rng(0)


# ---------------------------------------------------------------------------
# Quadrotor model: hover linearisation, 12 states / 4 inputs
#   x = [pos(3), vel(3), rpy(3), omega(3)]
# ---------------------------------------------------------------------------

def quadrotor_model(dt=0.005, coupled=True):
    """Hover-linearised quadrotor.

    `coupled=False` reproduces the FLAWED first version of this harness, in which
    the inputs were [thrust, tau_x, tau_y, tau_z] applied one-per-state with a
    diagonal inertia and no drag. That plant decouples into four independent SISO
    channels, so the condensed Hessian came out 4-block-diagonal (nnz(P)=1600,
    25% dense) and nobody would ever solve it as one 80-variable QP -- they would
    solve four 20-variable QPs. Iteration counts measured on it are optimistic.

    `coupled=True` (default) uses the physically correct formulation: the control
    inputs are the FOUR ROTOR THRUSTS, mapped to body force and moments through the
    rotor mixing matrix, plus aerodynamic drag. Every input then touches several
    states, and the condensed Hessian becomes dense -- as it must for a
    controllable, coupled plant.
    """
    n_x, n_u = 12, 4
    m, g = 0.9, 9.81
    Ixx, Iyy, Izz = 4.9e-3, 4.9e-3, 8.8e-3
    arm, c_tau = 0.15, 0.016        # rotor arm (m), torque/thrust ratio

    Ac = np.zeros((n_x, n_x))
    Ac[0:3, 3:6] = np.eye(3)          # pos' = vel
    Ac[3, 7] = g                      # vel_x' =  g*pitch
    Ac[4, 6] = -g                     # vel_y' = -g*roll
    Ac[6:9, 9:12] = np.eye(3)         # rpy'   = omega

    Bc = np.zeros((n_x, n_u))
    if coupled:
        # Rotor thrusts f1..f4 (X configuration) -> [Fz, tau_x, tau_y, tau_z]
        Bc[5, :] = np.array([1, 1, 1, 1]) / m
        Bc[9, :] = arm * np.array([0, 1, 0, -1]) / Ixx
        Bc[10, :] = arm * np.array([-1, 0, 1, 0]) / Iyy
        Bc[11, :] = c_tau * np.array([1, -1, 1, -1]) / Izz
        # Aerodynamic drag couples translational states to themselves
        for i in range(3, 6):
            Ac[i, i] = -0.12
        # Rotor-plane / gyroscopic cross-coupling between body rates
        Ac[9, 10] = 0.05
        Ac[10, 9] = -0.05
    else:
        Bc[5, 0] = 1.0 / m
        Bc[9, 1] = 1.0 / Ixx
        Bc[10, 2] = 1.0 / Iyy
        Bc[11, 3] = 1.0 / Izz

    A = np.eye(n_x) + dt * Ac         # forward Euler is adequate at 5 ms
    B = dt * Bc
    return A, B


def build_condensed_qp(A, B, N=20):
    """Condense to the control sequence U; return P, q-builder, A_con, bounds."""
    n_x, n_u = B.shape
    n_var = N * n_u

    # X = Phi @ x0 + Gamma @ U   (stacked predicted states, N blocks)
    Phi = np.zeros((N * n_x, n_x))
    Gamma = np.zeros((N * n_x, n_var))
    Apow = np.eye(n_x)
    for i in range(N):
        Apow = Apow @ A if i > 0 else A
        Phi[i * n_x:(i + 1) * n_x, :] = Apow
        for j in range(i + 1):
            Aj = np.linalg.matrix_power(A, i - j)
            Gamma[i * n_x:(i + 1) * n_x, j * n_u:(j + 1) * n_u] = Aj @ B

    Q = np.diag([10, 10, 10, 1, 1, 1, 5, 5, 5, 0.1, 0.1, 0.1])
    R = np.diag([0.1, 0.1, 0.1, 0.1])
    Qbar = np.kron(np.eye(N), Q)
    Rbar = np.kron(np.eye(N), R)

    P = Gamma.T @ Qbar @ Gamma + Rbar
    P = 0.5 * (P + P.T)                       # symmetrise

    # Constraints, in OSQP's native form  l <= A_con U <= u :
    #   input box  (n_var rows)   and   predicted-state box  (N*n_x rows)
    A_con = sp.vstack([sp.eye(n_var), sp.csc_matrix(Gamma)], format="csc")
    # Per-rotor thrust deviation from the hover trim of m*g/4 ~= 2.2 N.
    u_lim = np.tile([2.0, 2.0, 2.0, 2.0], N)
    x_lim = np.tile([5, 5, 5, 4, 4, 4, 0.6, 0.6, 3.14, 6, 6, 6], N)

    return dict(P=sp.csc_matrix(P), Gamma=Gamma, Phi=Phi, Qbar=Qbar,
                A_con=A_con, u_lim=u_lim, x_lim=x_lim,
                n_var=n_var, n_x=n_x, n_u=n_u, N=N)


def q_and_bounds(m, x0, xref):
    """Linear cost term and constraint bounds for the current state."""
    drift = m["Phi"] @ x0
    q = m["Gamma"].T @ m["Qbar"] @ (drift - np.tile(xref, m["N"]))
    l = np.concatenate([-m["u_lim"], -m["x_lim"] - drift])
    u = np.concatenate([+m["u_lim"], +m["x_lim"] - drift])
    return q, l, u


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

X0 = np.array([2.0, -1.5, 1.0, 0, 0, 0, 0.1, -0.1, 0.2, 0, 0, 0])

# OSQP checks the termination criterion only every `check_termination` iterations
# (default 25), so reported iteration counts are quantised to multiples of 25 and
# systematically OVERSTATE the true count. Set it to 1 when the iteration count is
# the quantity being measured; leave it at the default when measuring wall-clock,
# because per-iteration residual checks are not free.
EXACT = dict(check_termination=1)


def solve_once(m, x0, xref, eps, warm=None, polish=False, extra=None):
    q, l, u = q_and_bounds(m, x0, xref)
    prob = osqp.OSQP()
    prob.setup(P=m["P"], q=q, A=m["A_con"], l=l, u=u,
               eps_abs=eps, eps_rel=eps, verbose=False, polishing=polish,
               warm_starting=warm is not None, **(extra or {}))
    if warm is not None:
        prob.warm_start(x=warm)
    res = prob.solve()
    return res


def closed_loop(m, eps, steps=200, polish=False, extra=None, scenario="settle",
                coupled=True):
    """Realistic control loop: re-solve every tick, warm-started from last solution.

    scenario:
      "settle"     -- return to hover from an initial offset, then hold. This is the
                      EASIEST case for warm-starting: once settled, consecutive QPs are
                      nearly identical and the previous solution is almost optimal.
      "aggressive" -- step reference changes plus random disturbances, which force
                      active-set changes and genuine re-convergence. This is what a
                      real-time deadline must actually be sized for.
    """
    A, B = quadrotor_model(coupled=coupled)
    x = X0.copy()
    xref = np.zeros(12)
    rng = np.random.default_rng(1)

    q, l, u = q_and_bounds(m, x, xref)
    prob = osqp.OSQP()
    prob.setup(P=m["P"], q=q, A=m["A_con"], l=l, u=u,
               eps_abs=eps, eps_rel=eps, verbose=False, polishing=polish,
               warm_starting=True, **(extra or {}))

    iters, times, statuses = [], [], []
    for k in range(steps):
        if scenario == "aggressive":
            # The scenario must stay FEASIBLE, or the measured iteration count is
            # just OSQP struggling with an ill-posed problem rather than doing
            # control work. The horizon is N*dt = 100 ms and per-rotor authority is
            # +/-2 N about hover, so setpoint steps must be reachable within that:
            # a 3 m step is not (first version of this scenario used +/-3 m and
            # drove the QP primal-infeasible, producing p95 ~900 iterations that
            # measured nothing useful).
            if k % 20 == 0:
                xref = np.zeros(12)
                xref[:3] = rng.uniform(-0.3, 0.3, 3)
            # Gust disturbance the controller can actually reject.
            x = x + np.concatenate([rng.normal(0, 0.005, 3), rng.normal(0, 0.03, 3),
                                    rng.normal(0, 0.004, 3), rng.normal(0, 0.03, 3)])
        q, l, u = q_and_bounds(m, x, xref)
        prob.update(q=q, l=l, u=u)
        t0 = time.perf_counter()
        res = prob.solve()
        times.append((time.perf_counter() - t0) * 1e6)      # us
        iters.append(int(res.info.iter))
        statuses.append(res.info.status)
        u0 = np.asarray(res.x[:m["n_u"]]).ravel()
        if not np.all(np.isfinite(u0)):
            break
        x = A @ x + B @ u0                                   # advance the plant
    return np.array(iters), np.array(times), statuses


def pct(a, p):
    return float(np.percentile(a, p))


def main():
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    A, B = quadrotor_model(coupled=True)
    m = build_condensed_qp(A, B, N=20)
    A_d, B_d = quadrotor_model(coupled=False)
    m_d = build_condensed_qp(A_d, B_d, N=20)

    out("=" * 92)
    out("MEASURED OSQP BASELINE — QUAD-MPC (BENCHMARKS.md #1)")
    out("=" * 92)
    out("Host: x86_64 (this machine). TIMES DO NOT TRANSFER TO Arm — iteration")
    out("counts and problem structure do. Compare only the iteration counts to the model.")
    out()

    # ---- structural facts -------------------------------------------------
    n_var, n_x, N = m["n_var"], m["n_x"], m["N"]
    n_con_real = m["A_con"].shape[0]
    n_con_model = 2 * n_var + 2 * N * n_x
    out("PROBLEM STRUCTURE")
    out(f"  decision variables (N*n_u)      : {n_var}")
    out(f"  constraint ROWS, real OSQP form : {n_con_real}   "
        f"(input box {n_var} + state box {N*n_x})")
    out(f"  constraint rows assumed by model: {n_con_model}   "
        f"(2*n_var + 2*N*n_x)")
    out(f"  -> model overcounts constraints by {n_con_model/n_con_real:.1f}x")
    out(f"     Cause: OSQP expresses a two-sided bound l<=Ax<=u as ONE row; the model")
    out(f"     counts two. This inflates the A@x and A.T@y terms in every solver row.")
    out()
    out(f"  nnz(P)={m['P'].nnz}  nnz(A_con)={m['A_con'].nnz}  "
        f"density(P)={m['P'].nnz/n_var**2:.2f}   [COUPLED plant, correct]")
    out(f"  nnz(P)={m_d['P'].nnz}  nnz(A_con)={m_d['A_con'].nnz}  "
        f"density(P)={m_d['P'].nnz/n_var**2:.2f}   [decoupled plant, FLAWED v1]")
    out("  The first harness version used inputs [thrust,tau_x,tau_y,tau_z] with a")
    out("  diagonal inertia and no drag, which decouples into 4 independent SISO")
    out("  channels -> 4-block-diagonal Hessian at 25% density. Nobody solves that as")
    out("  one 80-var QP. Fixed: inputs are now the 4 rotor thrusts through the mixing")
    out("  matrix, plus drag and gyroscopic coupling. Both are measured below.")
    out()

    # ---- iteration counts: the headline ----------------------------------
    out("ITERATION COUNTS   [check_termination=1, exact]")
    out(f"{'plant':>10} {'scenario':>11} {'tol':>8} {'cold':>5} {'p50':>5} {'p95':>5} "
        f"{'p99':>5} {'max':>5} {'mean':>6}   status")
    for label, mm, cpl in (("coupled", m, True), ("decoupled", m_d, False)):
        for scen in ("settle", "aggressive"):
            for eps in (1e-3, 1e-6):
                cold = solve_once(mm, X0, np.zeros(12), eps, extra=EXACT)
                it, tm, st = closed_loop(mm, eps, extra=EXACT, scenario=scen,
                                         coupled=cpl)
                uniq = ", ".join(sorted(set(st))[:2])
                bad = sum(s != "solved" for s in st)
                flag = "" if bad == 0 else f"  <-- {bad}/{len(st)} NOT SOLVED, discard"
                out(f"{label:>10} {scen:>11} {eps:>8.0e} {int(cold.info.iter):>5} "
                    f"{pct(it,50):>5.0f} {pct(it,95):>5.0f} {pct(it,99):>5.0f} "
                    f"{it.max():>5.0f} {it.mean():>6.1f}   {uniq}{flag}")
        out()
    out("  'cold' = first solve, no warm start.")
    out("  'settle' = return to hover then hold -- the EASIEST warm-start case.")
    out("  'aggressive' = reference steps + disturbances forcing active-set changes;")
    out("  this is the distribution a hard real-time deadline must be sized against,")
    out("  and the tail (p99/max), not the median, is what determines feasibility.")
    out("  Default OSQP settings would report all of these quantised to multiples of 25.")
    out()

    # ---- timing on this host (labelled, not transferable) -----------------
    it, tm, st = closed_loop(m, 1e-3)
    out("TIMING ON THIS x86 HOST (reference only — NOT an Arm baseline)")
    out(f"  solve time p50 {pct(tm,50):.1f} us   p95 {pct(tm,95):.1f} us   "
        f"max {tm.max():.1f} us")
    out(f"  per-iteration  {pct(tm,50)/max(pct(it,50),1):.2f} us/iter at eps=1e-3")
    out(f"  1 kHz deadline (1000 us) met on {100*np.mean(tm<1000):.0f}% of ticks")
    out()

    out("=" * 92)
    out("WHAT THIS CHANGES")
    out("=" * 92)
    out("  Anything above is measurement, not modelling. Two model inputs are")
    out("  now testable rather than assumed: the iteration count and the constraint")
    out("  row count. See the table above for whether 40 iterations was right.")
    out("  Still NOT measured: Arm latency, Arm energy, achieved SIMD efficiency.")

    with open("perf_model/results_osqp_measured.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
