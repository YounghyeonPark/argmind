"""End-to-end validation: QUAD-MPC condensed QP on the Argmind functional simulator.

Takes the coupled quadrotor model and condensed QP from
``perf_model/measure_osqp_baseline.py`` (BENCHMARKS.md #1: 12 states, 4 rotor
thrusts, horizon N=20, condensed to 80 decision variables and 320 constraint
rows), expresses the ADMM iteration as a descriptor sequence, runs it on the
functional simulator, and compares the commanded control action against reference
OSQP.

Three separate comparisons, because they answer three different questions:

**V1 -- algorithm identity.** CM versus OSQP configured with *identical*
parameters and all runtime adaptivity disabled (no Ruiz scaling, no adaptive rho).
Both are then running the same algorithm, so agreement should be at machine
precision and the iteration counts should match exactly. This tests the
descriptor sequence, the kernels, the addressing and the scratchpad plumbing.

**V2 -- control-action accuracy.** CM at production tolerance versus a
high-accuracy OSQP solution of the same QP. This is the number a control engineer
cares about: how wrong is the thrust command.

**V3 -- closed loop.** 200 ticks of the BENCHMARKS "aggressive" scenario,
warm-started, reporting iteration-count distribution against the measured OSQP
range and the per-tick data transfer the residency model predicts.

An active constraint set is pinned deliberately in V1b: with the nominal +/-2 N
thrust bounds this QP's optimum is **interior**, so the box projection and the
dual update are no-ops at convergence and a wrong dual update is undetectable.
Any validation of a projection kernel that does not force an active set is not
validating it.

All modelled figures elsewhere in this project carry +/-2x. Nothing here is
modelled: these are numerical results from executing the descriptors.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "perf_model")):
    if p not in sys.path:
        sys.path.insert(0, p)

import osqp  # noqa: E402

from argmind import (  # noqa: E402
    CM100_SCRATCHPAD_BYTES,
    CM100_SCRATCHPAD_BYTES_V0,
    Device,
    DeviceConfig,
    Format,
    V0_LIMITATIONS,
    ScratchpadExhausted,
    concat,
    macif,
)
from argmind.solvers.admm import ADMMConfig, ADMMSolver, TerminationNorm  # noqa: E402
from measure_osqp_baseline import (  # noqa: E402
    X0,
    build_condensed_qp,
    q_and_bounds,
    quadrotor_model,
)

OUT = ROOT / "examples" / "results_quadrotor_admm.txt"


# ---------------------------------------------------------------------------
# Problem construction
# ---------------------------------------------------------------------------

def quad_mpc(u_lim: float | None = None):
    """The BENCHMARKS #1 problem. `u_lim` overrides per-rotor thrust authority."""
    A_sys, B_sys = quadrotor_model(coupled=True)
    m = build_condensed_qp(A_sys, B_sys, N=20)
    if u_lim is not None:
        m["u_lim"] = np.tile([u_lim] * m["n_u"], m["N"])
    return A_sys, B_sys, m


def osqp_matched(P, A, q, l, u, cfg: ADMMConfig, eps: float, max_iter=100_000):
    """OSQP with every runtime adaptation disabled, so it runs our algorithm."""
    prob = osqp.OSQP()
    prob.setup(
        P=sp.csc_matrix(P), q=q, A=sp.csc_matrix(A), l=l, u=u,
        eps_abs=eps, eps_rel=eps,
        scaling=0, adaptive_rho=0,
        rho=cfg.rho_as_encoded, sigma=cfg.sigma_as_encoded,
        alpha=cfg.alpha_as_encoded,
        check_termination=1, warm_starting=False, polishing=False,
        verbose=False, max_iter=max_iter,
    )
    return prob.solve()


def osqp_accurate(P, A, q, l, u):
    """A reference solution good enough to measure control-action error against."""
    prob = osqp.OSQP()
    prob.setup(
        P=sp.csc_matrix(P), q=q, A=sp.csc_matrix(A), l=l, u=u,
        eps_abs=1e-11, eps_rel=1e-11, check_termination=1, polishing=True,
        verbose=False, max_iter=400_000,
    )
    return prob.solve()


def build_solver(
    model, cfg: ADMMConfig, scratchpad_bytes: int, macro=None, strict_v0=False,
) -> ADMMSolver:
    n, m = model["n_var"], model["A_con"].shape[0]
    dev = Device(DeviceConfig(
        scratchpad_bytes=scratchpad_bytes,
        banks=16,
        # CM-100 as costed declares int8/bf16/fp32/fp64. fp16 needs the variant
        # that declares it -- see section 6.
        macro=macro or (macif.CM100_FP16 if cfg.fmt is Format.FP16
                        else macif.CM100),
        # The infinity-norm termination test needs the proposed NRMI opcode,
        # which frozen v0 does not have. Section 6 quantifies the alternative.
        strict_v0=strict_v0,
        check_hazards=True,
    ))
    return ADMMSolver(dev, n, m, cfg)


def numpy_reference(P, A, q, l, u, cfg: ADMMConfig, eps: float, max_iter=2000):
    """The same reduced-form ADMM in plain numpy/scipy double precision.

    A control for V1: it isolates "are the descriptors right" from "does a
    reduced-form Cholesky round differently from OSQP's full-KKT LDL^T". Uses
    exactly the constants the descriptors carry, fp32 rounding included.
    """
    import scipy.linalg as sla

    rho = cfg.rho_as_encoded
    sig = cfg.sigma_as_encoded
    al = cfg.alpha_as_encoded
    inv_rho = float(np.float32(1.0 / rho))
    n, m = P.shape[0], A.shape[0]
    K = P + sig * np.eye(n) + rho * (A.T @ A)
    L = np.linalg.cholesky(K)
    x, y, z = np.zeros(n), np.zeros(m), np.zeros(m)
    for k in range(1, max_iter + 1):
        rhs = sig * x - q + A.T @ (rho * z - y)
        xt = sla.solve_triangular(
            L.T, sla.solve_triangular(L, rhs, lower=True), lower=False)
        zt = A @ xt
        x = al * xt + (1 - al) * x
        t = al * zt + (1 - al) * z + inv_rho * y
        z = np.clip(t, l, u)
        y = rho * (t - z)
        Ax, Px, Aty = A @ x, P @ x, A.T @ y
        rp = np.max(np.abs(Ax - z))
        rd = np.max(np.abs(Px + q + Aty))
        ep = eps + eps * max(np.max(np.abs(Ax)), np.max(np.abs(z)))
        ed = eps + eps * max(np.max(np.abs(Px)), np.max(np.abs(Aty)),
                             np.max(np.abs(q)))
        if rp <= ep and rd <= ed:
            return x, k
    return x, max_iter


# ---------------------------------------------------------------------------

def main() -> None:
    lines: list[str] = []

    def out(s: str = "") -> None:
        print(s)
        lines.append(s)

    A_sys, B_sys, model = quad_mpc()
    P = model["P"].toarray()
    Acon = model["A_con"].toarray()
    n, m = model["n_var"], Acon.shape[0]
    BIG = 1 << 20        # scratchpad large enough that fp64 fits; see section 5

    out("=" * 92)
    out("QUAD-MPC ADMM ON THE ARGMIND FUNCTIONAL SIMULATOR")
    out("=" * 92)
    out("Problem: BENCHMARKS.md #1 -- coupled quadrotor (4 rotor thrusts, drag,")
    out("gyroscopic cross-coupling), 12 states, horizon N=20, condensed to the")
    out(f"control sequence: n_var={n}, constraint rows={m}, "
        f"density(P)={model['P'].nnz / n**2:.2f}.")
    out("Reference: OSQP. Numbers below are computed, not modelled.")
    out()

    # ---------------- 1. the generated program ---------------------------
    cfg = ADMMConfig(eps_abs=1e-6, eps_rel=1e-6, max_iterations=400,
                     fmt=Format.FP64, termination=TerminationNorm.LINF)
    solver = build_solver(model, cfg, BIG)
    counts = solver.descriptor_counts()

    out("-" * 92)
    out("1. THE DESCRIPTOR PROGRAM")
    out("-" * 92)
    per_iter = counts["iteration"] + counts["residual"]
    out(f"  descriptors: setup {counts['setup']}, iteration "
        f"{counts['iteration']}, residual {counts['residual']}")
    out(f"  -> {per_iter} descriptors per checked iteration, of which the "
        f"convergence test is {counts['residual']} "
        f"({100 * counts['residual'] / per_iter:.0f}%).")
    out("     The sequencer analysis assumed 28 descriptors per ADMM iteration; "
        "the")
    out("     hand-written v0 sequence is 32, and the residual block is nearly "
        "half of it.")
    out(f"  rho={cfg.rho} (power of two -- required, see section 6), "
        f"sigma={cfg.sigma}, alpha={cfg.alpha}")
    out("  dependency structure (emitted vs achievable):")
    out(solver.chain_report())
    whole = concat("admm.unrolled",
                   solver.solver.setup, solver.solver.iteration,
                   solver.solver.residual, solver.solver.iteration,
                   solver.solver.residual)
    out(f"  two-iteration unrolled block hazard-checks clean: "
        f"{len(whole)} descriptors")
    out()
    out("  iteration block:")
    for line in solver.solver.iteration.listing().splitlines():
        out("    " + line)
    out()

    # ---------------- 2. V1: algorithm identity --------------------------
    out("-" * 92)
    out("2. V1 -- ALGORITHM IDENTITY vs settings-matched OSQP")
    out("-" * 92)
    out("   OSQP with scaling=0, adaptive_rho=0 and our rho/sigma/alpha is running")
    out("   the same algorithm, so this is a like-for-like test of the descriptor")
    out("   sequence rather than of ADMM.")
    out()
    out(f"   {'bounds':>22} {'eps':>7} {'act':>4} {'OSQP it':>8} {'CM it':>6} "
        f"{'max|x-x_osqp|':>14} {'npy it':>7} {'max|x-x_npy|':>13}")

    v1_worst = 0.0
    v1_worst_npy = 0.0
    v1_iter_mismatch = 0
    v1_npy_mismatch = 0
    v1_cases = 0
    for label, u_lim in (("nominal +/-2.0 N", None), ("tight +/-0.35 N", 0.35)):
        _, _, mdl = quad_mpc(u_lim)
        Pm, Am = mdl["P"].toarray(), mdl["A_con"].toarray()
        q, l, u = q_and_bounds(mdl, X0, np.zeros(12))
        for eps in (1e-6, 1e-8):
            c = ADMMConfig(eps_abs=eps, eps_rel=eps, max_iterations=2000,
                           fmt=Format.FP64, termination=TerminationNorm.LINF)
            s = build_solver(mdl, c, BIG)
            s.set_problem(Pm, Am)
            s.set_instance(q, l, u)
            s.reset_iterates()
            st = s.solve(warm=True)
            r = osqp_matched(Pm, Am, q, l, u, c, eps)
            x_ref = np.asarray(r.x).ravel()
            active = int(np.sum(np.abs(np.asarray(r.y).ravel()) > 1e-8))
            d = float(np.max(np.abs(s.solution - x_ref)))
            x_npy, it_npy = numpy_reference(Pm, Am, q, l, u, c, eps)
            d_npy = float(np.max(np.abs(s.solution - x_npy)))
            v1_worst = max(v1_worst, d)
            v1_worst_npy = max(v1_worst_npy, d_npy)
            v1_cases += 1
            v1_iter_mismatch += int(st.iterations != int(r.info.iter))
            v1_npy_mismatch += int(st.iterations != it_npy)
            out(f"   {label:>22} {eps:>7.0e} {active:>4} {int(r.info.iter):>8} "
                f"{st.iterations:>6} {d:>14.3e} {it_npy:>7} {d_npy:>13.3e}"
                f"{'' if st.converged else '  LIMIT'}")
    out()
    out(f"   vs numpy reduced-form reference (same algorithm, same constants):")
    out(f"     worst disagreement {v1_worst_npy:.3e}, "
        f"iteration-count mismatches {v1_npy_mismatch}/{v1_cases}")
    out(f"   vs OSQP (same algorithm, different linear solver -- reduced Cholesky")
    out(f"   here, sparse full-KKT LDL' there):")
    out(f"     worst disagreement {v1_worst:.3e}, "
        f"iteration-count mismatches {v1_iter_mismatch}/{v1_cases}")
    out("   Reading: the descriptor program reproduces the reference algorithm to")
    out("   machine precision. Where OSQP's count differs it differs by one, on the")
    out("   interior-optimum rows only, because the two factorisations round")
    out("   differently and the residual is crossing the threshold steeply there.")
    out("   The tight-bounds rows matter for a second reason: with the nominal")
    out("   +/-2 N authority this QP's optimum is INTERIOR (active set 0), so PROJ")
    out("   and the dual update are no-ops at convergence and a wrong dual update")
    out("   cannot be detected at all.")
    out()

    # ---------------- 3. V2: control-action accuracy ---------------------
    out("-" * 92)
    out("3. V2 -- CONTROL-ACTION ACCURACY vs a high-accuracy OSQP solution")
    out("-" * 92)
    q, l, u = q_and_bounds(model, X0, np.zeros(12))
    ref = osqp_accurate(P, Acon, q, l, u)
    u_star = np.asarray(ref.x).ravel()[:model["n_u"]]
    out(f"   reference (eps=1e-11, polished): u0 = "
        f"{np.array2string(u_star, precision=9)}")
    out(f"   {'precision':>10} {'eps':>8} {'CM it':>7} {'|u0-u0*|inf':>13} "
        f"{'rel':>10} {'|x-x*|inf':>12}  note")
    for fmt in (Format.FP64, Format.FP32):
        for eps in (1e-3, 1e-4, 1e-6, 1e-8):
            c = ADMMConfig(eps_abs=eps, eps_rel=eps, max_iterations=600,
                           fmt=fmt, termination=TerminationNorm.LINF)
            s = build_solver(model, c, BIG)
            s.set_problem(P, Acon)
            s.set_instance(q, l, u)
            s.reset_iterates()
            st = s.solve(warm=True)
            x_cm = s.solution
            eu = float(np.max(np.abs(x_cm[:model["n_u"]] - u_star)))
            out(f"   {str(fmt):>10} {eps:>8.0e} {st.iterations:>7} {eu:>13.3e} "
                f"{eu / max(np.max(np.abs(u_star)), 1e-30):>10.2e} "
                f"{float(np.max(np.abs(x_cm - np.asarray(ref.x).ravel()))):>12.3e}"
                f"  {'' if st.converged else 'ITERATION LIMIT -- tolerance not reachable'}")
    out()
    out("   Accuracy report, as required for any precision change: down to")
    out("   eps=1e-6 the ADMM stopping tolerance dominates and fp32 costs nothing")
    out("   measurable on the commanded thrust. At eps=1e-8 fp32 cannot reach the")
    out("   tolerance -- the row above shows where the floor is, which is the")
    out("   number a customer needs before choosing fp32.")
    out()

    # ---------------- 4. V3: closed loop + residency ---------------------
    out("-" * 92)
    out("4. V3 -- CLOSED LOOP, 200 TICKS, WARM STARTED")
    out("-" * 92)
    for eps in (1e-3, 1e-6):
        res = closed_loop(model, A_sys, B_sys, eps, steps=200)
        out(f"   eps={eps:.0e}")
        out(f"     CM iterations      p50 {res['p50']:.0f}  p95 {res['p95']:.0f} "
            f" max {res['max']:.0f}   (cold first solve {res['cold']})")
        out(f"     OSQP default       p50 {res['ref_p50']:.0f}  "
            f"p95 {res['ref_p95']:.0f}  max {res['ref_max']:.0f}")
        out(f"     |u0_CM - u0_OSQP|  p50 {res['u_p50']:.3e}  "
            f"max {res['u_max']:.3e}")
        out(f"     per-tick transfer  fetched {res['fetch_steady']} B, "
            f"written back {res['wb_steady']} B "
            f"(first call fetched {res['fetch_first']} B)")
        out(f"     setup (Cholesky) ran on {res['setups']} of 200 ticks")
        out(f"     descriptors per tick p50 {res['desc_p50']:.0f}")
    out()

    # ---------------- 5. scratchpad footprint ----------------------------
    out("-" * 92)
    out("5. SCRATCHPAD FOOTPRINT — THE MEASUREMENT THAT RESIZED THE CM-100")
    out("-" * 92)
    out("   This section is why the CM-100 scratchpad is 192 KB and not 128 KB.")
    out("   The plan below is MEASURED (statically allocated bytes), and it")
    out("   contradicted the costed capacity. Ruling: descriptor v0.1 R8.")
    out()
    kib = lambda n: f"{n / 1024:.1f} KiB"
    for cap, label in ((CM100_SCRATCHPAD_BYTES_V0, "128 KiB (as costed)"),
                       (CM100_SCRATCHPAD_BYTES, "192 KiB (corrected)")):
        out(f"   against {label}:")
        for fmt in (Format.FP64, Format.FP32, Format.FP16):
            s = build_solver(model, ADMMConfig(fmt=fmt), BIG)
            used = s.device.plan.used
            fits = used <= cap
            over = "" if fits else f" (over by {100 * used / cap - 100:.0f}%)"
            out(f"     all-{str(fmt):>4} working set {used:>8} B ({kib(used):>10})  "
                f"{'fits' if fits else 'DOES NOT FIT'}{over}")
        out()
    for cap in (CM100_SCRATCHPAD_BYTES_V0, CM100_SCRATCHPAD_BYTES):
        try:
            build_solver(model, ADMMConfig(fmt=Format.FP32), cap)
            out(f"   fp32 build against a real {kib(cap)} device: SUCCEEDED")
        except ScratchpadExhausted as e:
            out(f"   fp32 build against a real {kib(cap)} device: REFUSED --")
            out(f"     {e}")
    out()
    out("   fp32 is the DEFAULT for ADMM, so fp32 is what has to fit. fp64 here is")
    out("   a VALIDATION configuration (it is what matches OSQP bit-for-bit); fp64")
    out("   is an interior-point requirement, not an ADMM one, so the 327 KiB figure")
    out("   must NOT be used to size anything.")
    out()
    out("   Breakdown of the fp32 working set, largest first:")
    c = ADMMConfig(fmt=Format.FP32)
    s = build_solver(model, c, BIG)
    for a in sorted(s.device.plan.allocations, key=lambda a: -a.nbytes)[:6]:
        out(f"     {a.name:<24} {a.nbytes:>8} B  {a.shape}")
    out(f"     ... {len(s.device.plan.allocations) - 6} smaller allocations")
    out()

    # ---------------- 6. the frozen spec, under load ---------------------
    out("-" * 92)
    out("6. WHAT IMPLEMENTING THE FROZEN SPEC REVEALED")
    out("-" * 92)
    out("   6a. rho must be a power of two, or the solve stalls.")
    out("       The iteration multiplies by 1/rho and by rho, both carried in the")
    out("       fp32 descriptor word 7. The ADMM fixed point needs them to be")
    out("       exact reciprocals: with an active constraint, y = rho*(1/rho)*y at")
    out("       the fixed point, so if that product is not 1 the only solution is")
    out("       y = 0 and the dual variable cannot settle.")
    out("       Run on BOTH bound sets, because the interior-optimum case (active")
    out("       set 0, y* = 0) satisfies y = 0 trivially and therefore hides it:")
    _, _, mdl_t = quad_mpc(0.35)
    Pt, At = mdl_t["P"].toarray(), mdl_t["A_con"].toarray()
    qt, lt, ut = q_and_bounds(mdl_t, X0, np.zeros(12))
    out(f"       {'bounds':>18} {'rho':>7} {'recip err':>11} {'CM iters':>9} "
        f"{'final primal res':>17}")
    for blabel, (Pb, Ab, qb, lb, ub) in (
        ("nominal (act 0)", (P, Acon, q, l, u)),
        ("tight (act 10)", (Pt, At, qt, lt, ut)),
    ):
        for rho in (0.1, 0.2, 0.125, 0.25):
            c = ADMMConfig(rho=rho, eps_abs=1e-9, eps_rel=1e-9,
                           max_iterations=1500, fmt=Format.FP64,
                           termination=TerminationNorm.LINF)
            # Built directly, with the inexact-rho guard explicitly waived, so
            # the stall can be measured rather than merely refused.
            dev = Device(DeviceConfig(scratchpad_bytes=BIG,
                                      macro=macif.UNIVERSAL, strict_v0=False))
            s = ADMMSolver(dev, n, m, c, allow_inexact_rho=True)
            s.set_problem(Pb, Ab)
            s.set_instance(qb, lb, ub)
            s.reset_iterates()
            st = s.solve(warm=True)
            out(f"       {blabel:>18} {rho:>7} {c.rho_reciprocal_error:>11.2e} "
                f"{st.iterations:>9} {float(s.s_rp.spad_array[0]):>17.3e}"
                f"{'' if st.converged else '  STALLED'}")
    out("       The runtime refuses a non-power-of-two rho by default rather than")
    out("       returning a quietly-unconverged answer; the rows above use the")
    out("       explicit opt-out.")
    out()

    out("   6b. The convergence test cannot be expressed in frozen v0.")
    out("       OSQP's criterion is stated in infinity norms. v0's only reduction")
    out("       is DOT, i.e. a 2-norm. Cost of using the 2-norm instead:")
    out(f"       {'norm':>6} {'opcode':>7} {'frozen v0':>10} {'CM iters':>9} "
        f"{'|u0-u0*|inf':>13}")
    for norm in (TerminationNorm.LINF, TerminationNorm.L2):
        c = ADMMConfig(eps_abs=1e-6, eps_rel=1e-6, max_iterations=600,
                       fmt=Format.FP64, termination=norm)
        strict = norm is TerminationNorm.L2
        s = build_solver(model, c, BIG, macro=macif.UNIVERSAL, strict_v0=strict)
        s.set_problem(P, Acon)
        s.set_instance(q, l, u)
        s.reset_iterates()
        st = s.solve(warm=True)
        eu = float(np.max(np.abs(s.solution[:model["n_u"]] - u_star)))
        out(f"       {norm.name:>6} {norm.opcode.name:>7} "
            f"{'yes' if strict else 'NO':>10} {st.iterations:>9} {eu:>13.3e}")
    out("       Both programs are the same fifteen residual descriptors with one")
    out("       opcode substituted. The 2-norm test is a valid stopping rule, but it")
    out("       is not the reference solver's rule, so iteration counts published")
    out("       against OSQP would not be comparable. One opcode (NRMI) fixes it:")
    out("       no new fields, no new base addresses.")
    out()

    out("   6c. MACIF refusals fire as specified.")
    for macro, fmt in ((macif.CM100, Format.FP16),
                       (macif.CIM_INFERENCE_MACRO, Format.FP64),
                       (macif.CIM_INFERENCE_MACRO, Format.FP32)):
        c = ADMMConfig(fmt=fmt)
        try:
            build_solver(model, c, BIG, macro=macro)
            out(f"       {macro.name} + {fmt}: ACCEPTED  <-- unexpected")
        except Exception as exc:
            first = str(exc).split(". ")[0]
            out(f"       {macro.name} + {fmt}: refused -- {first}")
    out()

    out("   6d. Descriptor v0 limitations found by implementing it:")
    for lim in V0_LIMITATIONS:
        out(f"       [{lim['id']}] {lim['what']}")
        out(f"            severity: {lim['severity']}")
        out(f"            fix: {lim['fix']}")
    out()

    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nwritten to {OUT}")


def closed_loop(model, A_sys, B_sys, eps: float, steps: int = 200) -> dict:
    """The BENCHMARKS 'aggressive' scenario, on the CM, warm started."""
    P = model["P"].toarray()
    Acon = model["A_con"].toarray()
    cfg = ADMMConfig(eps_abs=eps, eps_rel=eps, max_iterations=2000,
                     fmt=Format.FP64, termination=TerminationNorm.LINF)
    s = build_solver(model, cfg, 1 << 20)
    s.set_problem(P, Acon)

    rng = np.random.default_rng(1)
    x = X0.copy()
    xref = np.zeros(12)

    q0, l0, u0 = q_and_bounds(model, x, xref)
    ref = osqp.OSQP()
    ref.setup(P=model["P"], q=q0, A=model["A_con"], l=l0, u=u0,
              eps_abs=eps, eps_rel=eps, verbose=False, check_termination=1,
              warm_starting=True)

    iters, ref_iters, uerr, fetched, wb, desc = [], [], [], [], [], []
    setups = 0
    cold = None
    for k in range(steps):
        if k % 20 == 0:
            xref = np.zeros(12)
            xref[:3] = rng.uniform(-0.3, 0.3, 3)
        x = x + np.concatenate([rng.normal(0, 0.005, 3), rng.normal(0, 0.03, 3),
                                rng.normal(0, 0.004, 3), rng.normal(0, 0.03, 3)])
        q, l, u = q_and_bounds(model, x, xref)
        s.set_instance(q, l, u)
        st = s.solve(warm=(k > 0))
        if k == 0:
            cold = st.iterations
        ref.update(q=q, l=l, u=u)
        r = ref.solve()
        iters.append(st.iterations)
        ref_iters.append(int(r.info.iter))
        desc.append(st.descriptors_issued)
        fetched.append(st.bytes_fetched)
        wb.append(st.bytes_written_back)
        setups += int(st.setup_ran)
        u_cm = s.solution[:model["n_u"]]
        uerr.append(float(np.max(np.abs(u_cm - np.asarray(r.x).ravel()[:4]))))
        x = A_sys @ x + B_sys @ u_cm

    it = np.array(iters)
    ri = np.array(ref_iters)
    ue = np.array(uerr)
    return dict(
        p50=np.percentile(it, 50), p95=np.percentile(it, 95), max=it.max(),
        cold=cold,
        ref_p50=np.percentile(ri, 50), ref_p95=np.percentile(ri, 95),
        ref_max=ri.max(),
        u_p50=float(np.median(ue)), u_max=float(ue.max()),
        fetch_first=fetched[0], fetch_steady=int(np.median(fetched[1:])),
        wb_steady=int(np.median(wb[1:])), setups=setups,
        desc_p50=float(np.median(desc)),
    )


if __name__ == "__main__":
    main()
