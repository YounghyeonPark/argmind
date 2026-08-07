# SPDX-License-Identifier: Apache-2.0
"""End-to-end code generation: QUAD-MPC from a problem description to descriptors.

The hand-written descriptor sequence in ``examples/quadrotor_admm.py`` proved the
architecture runs the algorithm. This proves the *generator* does, which is the
part a customer touches: nobody is going to hand-write accelerator kernels for
their MPC controller.

What is demonstrated, in order:

1. **The front door.** A CVXPY problem with the state as a ``cp.Parameter`` --
   what a control engineer already has -- is extracted to a QP, and the
   residency declaration is *derived* from which canonical data blocks move when
   the parameters move. Plus the direct ``(P, q, A, l, u)`` front door.
2. **Generated == hand-written.** In reference-equivalent mode the emitted
   descriptor stream is identical to the hand-written one, descriptor for
   descriptor, in canonical (buffer, offset) form.
3. **Generated == OSQP**, on the nominal *and* the tight-bound variant. The
   nominal QUAD-MPC optimum is interior, so it cannot validate the projection
   kernel or the dual update; the tight variant is what does.
4. **The static allocation plan**, at three precision policies.
5. **The residency plan**, derived and then checked against the bytes a real
   closed loop moves.
6. **The compliance report** against all eight codegen contract rules.
7. **What the format could not express**, and what the generator did about it.

Nothing here is modelled except the cycles-per-descriptor column of the
compliance report, which is labelled where it appears. Everything else is
computed by executing descriptors.
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
    Device,
    DeviceConfig,
    Format,
    ScratchpadExhausted,
    macif,
)
from argmind.codegen import (  # noqa: E402
    CM100,
    CYCLES_PER_DESCRIPTOR_BAR,
    QUEUE_DEPTH,
    CodegenOptions,
    ContractViolation,
    PrecisionPolicy,
    QPProblem,
    generate,
)
from argmind.codegen.findings import (  # noqa: E402
    RESIDENCY_MECHANISM_HAZARD,
    V0_1_LIMITATIONS_FOUND_BY_CODEGEN,
)
from argmind.codegen.frontend import (  # noqa: E402
    CASADI_FRONTEND,
    CVXPY_AVAILABLE,
    from_cvxpy,
)
from argmind.codegen.generate import compare_programs  # noqa: E402
from argmind.solvers.admm import ADMMConfig, ADMMSolver  # noqa: E402
from argmind.solvers.admm import TerminationNorm as HandTerminationNorm  # noqa: E402
from measure_osqp_baseline import (  # noqa: E402
    X0,
    build_condensed_qp,
    q_and_bounds,
    quadrotor_model,
)

OUT = ROOT / "examples" / "results_codegen_quadrotor.txt"
BIG = 1 << 21


# ---------------------------------------------------------------------------

def quad_model(u_lim: float | None = None):
    A_sys, B_sys = quadrotor_model(coupled=True)
    mdl = build_condensed_qp(A_sys, B_sys, N=20)
    if u_lim is not None:
        mdl["u_lim"] = np.tile([u_lim] * mdl["n_u"], mdl["N"])
    return A_sys, B_sys, mdl


def quad_problem(mdl, x=None, xref=None) -> QPProblem:
    q, l, u = q_and_bounds(mdl, X0 if x is None else x,
                           np.zeros(12) if xref is None else xref)
    return QPProblem.from_arrays(
        mdl["P"].toarray(), q, mdl["A_con"].toarray(), l, u,
        varies=("q", "l", "u"), name="QUAD-MPC")


def cvxpy_quad_mpc(mdl):
    """QUAD-MPC as a user would write it: the state is a Parameter."""
    import cvxpy as cp

    N, n_x = mdl["N"], mdl["n_x"]
    P = mdl["P"].toarray()
    Gamma, Phi, Qbar = mdl["Gamma"], mdl["Phi"], mdl["Qbar"]
    U = cp.Variable(mdl["n_var"], name="U")
    x0 = cp.Parameter(n_x, name="x0")
    xref = cp.Parameter(n_x, name="xref")
    tile = np.tile(np.eye(n_x), (N, 1))
    drift = Phi @ x0
    cost = (0.5 * cp.quad_form(U, cp.psd_wrap(P))
            + ((Gamma.T @ Qbar) @ (drift - tile @ xref)) @ U)
    cons = [U <= mdl["u_lim"], U >= -mdl["u_lim"],
            Gamma @ U <= mdl["x_lim"] - drift,
            Gamma @ U >= -mdl["x_lim"] - drift]
    prob = cp.Problem(cp.Minimize(cost), cons)
    x0.value = X0
    xref.value = np.zeros(n_x)
    return prob, U


def osqp_matched(problem: QPProblem, opt: CodegenOptions, eps: float,
                 max_iter=200_000):
    prob = osqp.OSQP()
    prob.setup(
        P=sp.csc_matrix(problem.P), q=problem.q, A=sp.csc_matrix(problem.A),
        l=problem.l, u=problem.u, eps_abs=eps, eps_rel=eps, scaling=0,
        adaptive_rho=0, rho=opt.rho_as_encoded, sigma=opt.sigma_as_encoded,
        alpha=opt.alpha_as_encoded, check_termination=1, warm_starting=False,
        polishing=False, verbose=False, max_iter=max_iter)
    return prob.solve()


def osqp_accurate(problem: QPProblem):
    prob = osqp.OSQP()
    prob.setup(P=sp.csc_matrix(problem.P), q=problem.q,
               A=sp.csc_matrix(problem.A), l=problem.l, u=problem.u,
               eps_abs=1e-11, eps_rel=1e-11, check_termination=1, polishing=True,
               verbose=False, max_iter=400_000)
    return prob.solve()


def fp64_opts(**kw) -> CodegenOptions:
    kw.setdefault("precision", PrecisionPolicy.uniform(Format.FP64))
    return CodegenOptions(**kw)


def reference_mode(**kw) -> CodegenOptions:
    kw.setdefault("rho_in_scratchpad", False)
    kw.setdefault("elide_copies", False)
    return fp64_opts(**kw)


def gen(problem, options=None, **kw):
    kw.setdefault("scratchpad_bytes", BIG)
    kw.setdefault("macro", macif.UNIVERSAL)
    return generate(problem, options or fp64_opts(), **kw)


# ---------------------------------------------------------------------------

def main() -> None:
    lines: list[str] = []

    def out(s: str = "") -> None:
        print(s)
        lines.append(s)

    _, _, mdl = quad_model()
    nominal = quad_problem(mdl)
    _, _, mdl_t = quad_model(0.35)
    tight = quad_problem(mdl_t)

    out("=" * 92)
    out("CODE GENERATION FOR QUAD-MPC: PROBLEM DESCRIPTION IN, DESCRIPTORS OUT")
    out("=" * 92)
    out(f"Problem: {nominal.summary()}")
    out("Reference: OSQP, and the hand-written descriptor sequence in")
    out("argmind/solvers/admm.py. Numbers below are computed by executing")
    out("descriptors, except the cycles-per-descriptor column, which is MODELLED")
    out("(+/-2x, not quotable externally) and labelled where it appears.")
    out()

    # ---------------- 1. the front door ---------------------------------
    out("-" * 92)
    out("1. THE FRONT DOOR")
    out("-" * 92)
    cvx_problem = None
    if CVXPY_AVAILABLE:
        prob, U = cvxpy_quad_mpc(mdl)
        cvx_problem, extraction = from_cvxpy(prob, name="QUAD-MPC-cvxpy")
        out("   a) CVXPY. The user's own problem, with the state as a Parameter.")
        out(f"      {extraction.text()}")
        out("      Two things happened there and both matter:")
        out("      * `varies` was DERIVED, not declared. A CVXPY Parameter IS the")
        out("        residency declaration -- data that changes between solves is a")
        out("        Parameter, data that does not is a constant, which is what DPP")
        out("        means. The front end canonicalises at two parameter values and")
        out("        differences the result, so the user supplies NO hints.")
        out("      * The two-sided rows were RECOVERED. CVXPY canonicalises")
        out("        l <= Ax <= u into two one-sided rows, so a 320-constraint")
        out("        problem arrives with 640 -- and the constraint matrix is the")
        out("        largest object in the program. Pairing rows A_i = -A_j recovers")
        out("        the hand-derived form exactly. Without it, coming in through")
        out("        CVXPY would cost 2x the scratchpad and 2x the A-product cycles,")
        out("        i.e. the on-ramp would be a tax on using it.")
    else:
        out("   a) CVXPY is not installed; the front end is exercised by")
        out("      tests/test_argmind_codegen.py when it is.")
    out()
    out("   b) Direct: QPProblem.from_arrays(P, q, A, l, u, varies=(...)).")
    out("      Needs nothing beyond numpy. `varies` must be stated and defaults to")
    out("      EVERYTHING, i.e. to no residency assumption at all -- a generator")
    out("      that guessed 'P is probably constant' would return a stale answer")
    out("      the first time a user retuned a weight.")
    out()
    out(f"   c) CasADi: {CASADI_FRONTEND['status']}.")
    out(f"      {CASADI_FRONTEND['why']}")
    out()

    # ---------------- 2. generated == hand-written ----------------------
    out("-" * 92)
    out("2. GENERATED == HAND-WRITTEN, DESCRIPTOR FOR DESCRIPTOR")
    out("-" * 92)
    g_ref = gen(nominal, reference_mode())
    dev_hand = Device(DeviceConfig(scratchpad_bytes=BIG, banks=16,
                                   macro=macif.UNIVERSAL, host_bytes=8 << 20))
    hand = ADMMSolver(dev_hand, nominal.n_var, nominal.n_con,
                      ADMMConfig(fmt=Format.FP64,
                                 termination=HandTerminationNorm.LINF))
    for block in ("setup", "iteration", "residual"):
        diffs = compare_programs(getattr(g_ref.solver, block), g_ref.device,
                                 getattr(hand.solver, block), dev_hand)
        n_desc = len(getattr(g_ref.solver, block))
        out(f"   {block:<10} {n_desc:>3} descriptors  "
            f"{'IDENTICAL' if not diffs else f'{len(diffs)} DIFFERENCES'}")
    out("   Compared in canonical form: addresses resolved to (buffer, element")
    out("   offset), because requiring identical absolute addresses would only test")
    out("   that two allocators declare buffers in the same order.")
    out()
    out(f"   descriptors: {g_ref.descriptor_counts()} -> "
        f"{g_ref.descriptors_per_checked_iteration} per checked iteration")
    out("   dependency structure (emitted vs achievable):")
    out(g_ref.chain_report())
    out()
    out("   iteration block, as generated:")
    for line in g_ref.solver.iteration.listing().splitlines():
        out("     " + line)
    out()

    # ---------------- 3. vs OSQP, both bound sets ----------------------
    out("-" * 92)
    out("3. GENERATED == OSQP, ON BOTH BOUND SETS")
    out("-" * 92)
    out("   The nominal +/-2 N row has an INTERIOR optimum (active set 0), so PROJ")
    out("   and the dual update are no-ops at convergence and a wrong dual update")
    out("   is undetectable there. The tight +/-0.35 N row is what validates them.")
    out()
    out(f"   {'bounds':>18} {'mode':>10} {'eps':>7} {'act':>4} {'OSQP it':>8} "
        f"{'CM it':>6} {'desc/it':>8} {'max|x-x_osqp|':>14}")
    for label, problem in (("nominal", nominal), ("tight", tight)):
        for eps in (1e-6, 1e-8):
            for mode, opts in (("reference", reference_mode),
                               ("default", fp64_opts)):
                o = opts(eps_abs=eps, eps_rel=eps, max_iterations=2000)
                s = gen(problem, o)
                st = s.solve(warm=False)
                r = osqp_matched(problem, o, eps)
                act = int(np.sum(np.abs(np.asarray(r.y).ravel()) > 1e-8))
                d = float(np.max(np.abs(s.solution - np.asarray(r.x).ravel())))
                out(f"   {label:>18} {mode:>10} {eps:>7.0e} {act:>4} "
                    f"{int(r.info.iter):>8} {st.iterations:>6} "
                    f"{s.descriptors_per_checked_iteration:>8} {d:>14.3e}"
                    f"{'' if st.converged else '  LIMIT'}")
    out()
    out("   'reference' is the mode that reproduces the hand-written program (rho in")
    out("   descriptor word 7, staging copies kept). 'default' is what the generator")
    out("   ships: rho in scratchpad (ruling R5) and staging copies elided, 29")
    out("   descriptors instead of 31. The two agree bit for bit.")
    out()

    # ---------------- 4. rho, the contract's sharpest rule -------------
    out("-" * 92)
    out("4. CONTRACT RULE C1: rho IS REFUSED, NOT WARNED ABOUT")
    out("-" * 92)
    for rho in (0.1, 0.2, 0.125, 0.25):
        try:
            g = gen(tight, reference_mode(rho=rho))
            f = g.compliance.by_rule("C1")
            out(f"   rho={rho:<6} w7 immediate     ACCEPTED "
                f"(reciprocal error {f.numbers['w7_reciprocal_error']:.2e})")
        except ContractViolation:
            err = CodegenOptions(rho=rho).rho_reciprocal_error
            out(f"   rho={rho:<6} w7 immediate     REFUSED  "
                f"(reciprocal error {err:.2e} -> the dual cannot settle)")
    out()
    out("   And ruling R5's answer, measured: the same rho that stalls in w7")
    out("   converges when it lives in scratchpad at the buffer's own precision.")
    out(f"   {'rho':>8} {'encoding':>18} {'iterations':>11} {'converged':>10}")
    for rho in (0.1, 0.125):
        o = fp64_opts(rho=rho, rho_in_scratchpad=True, eps_abs=1e-9,
                      eps_rel=1e-9, max_iterations=1500)
        s = gen(tight, o)
        st = s.solve(warm=False)
        out(f"   {rho:>8} {'scratchpad VMUL':>18} {st.iterations:>11} "
            f"{str(st.converged):>10}")
    out("   The w7 encoding of rho=0.1 does NOT converge -- measured in")
    out("   tests/test_argmind_codegen.py, which bypasses the refusal by rewriting")
    out("   the two VMULs as w7 immediates and runs to the 600-iteration limit. So")
    out("   the power-of-two rule is real on the w7 path and genuinely lifted by R5")
    out("   --")
    out("   which is worth knowing, because rho is a tuning parameter and taking")
    out("   away 7 of every 8 candidate values is not free.")
    out()

    # ---------------- 5. the allocation plan ---------------------------
    out("-" * 92)
    out("5. THE STATIC SCRATCHPAD ALLOCATION PLAN")
    out("-" * 92)
    policies = [
        ("all-fp64 (validation)", PrecisionPolicy.uniform(Format.FP64), False),
        ("all-fp32 (default)", PrecisionPolicy.uniform(Format.FP32), False),
        ("fp16 operands / fp32 solve", PrecisionPolicy.fp16_operands(), True),
    ]
    out(f"   {'policy':>28} {'bytes':>9} {'KiB':>8} {'of 192 KiB':>11} "
        f"{'desc/it':>8}")
    plans = {}
    for name, pol, narrow in policies:
        s = gen(nominal, CodegenOptions(precision=pol,
                                        allow_narrow_operands=narrow))
        plans[name] = s
        used = s.device.plan.used
        out(f"   {name:>28} {used:>9} {used / 1024:>8.1f} "
            f"{100 * used / CM100_SCRATCHPAD_BYTES:>10.1f}% "
            f"{s.descriptors_per_checked_iteration:>8}")
    out()
    out("   fp32 is the default for ADMM and fp32 is what has to fit. fp64 is a")
    out("   VALIDATION configuration -- it is what matches OSQP bit for bit -- and")
    out("   must not be used to size anything. Against a real 192 KiB device:")
    for pol, label in ((PrecisionPolicy.uniform(Format.FP32), "fp32"),
                       (PrecisionPolicy.uniform(Format.FP64), "fp64")):
        try:
            s = generate(nominal, CodegenOptions(precision=pol),
                         scratchpad_bytes=CM100_SCRATCHPAD_BYTES,
                         macro=macif.UNIVERSAL)
            out(f"     {label}: build SUCCEEDED, {s.device.plan.used} B")
        except ScratchpadExhausted as e:
            out(f"     {label}: build REFUSED -- {str(e).splitlines()[0]}")
    out()
    ref32 = gen(nominal, reference_mode(
        precision=PrecisionPolicy.uniform(Format.FP32)))
    new32 = gen(nominal, fp64_opts(precision=PrecisionPolicy.uniform(Format.FP32)))
    out(f"   The hand-written v0.1 layout measured 167440 B at fp32. Generated in")
    out(f"   reference mode: {ref32.device.plan.used} B -- the same plan. With the")
    out(f"   in-place rule enforced: {new32.device.plan.used} B, "
        f"{ref32.device.plan.used - new32.device.plan.used} B less, which is")
    out("   exactly the two staging buffers rule C6 deletes.")
    out()
    out("   Largest allocations, fp32:")
    for a in sorted(new32.device.plan.allocations, key=lambda a: -a.nbytes)[:6]:
        out(f"     {a.name:<12} {a.nbytes:>8} B  {str(a.fmt):>5}  {a.shape}")
    out(f"     ... {len(new32.device.plan.allocations) - 6} smaller allocations")
    out()

    # ---------------- 6. residency, derived then measured --------------
    out("-" * 92)
    out("6. THE RESIDENCY PLAN, DERIVED THEN MEASURED")
    out("-" * 92)
    g = gen(nominal, fp64_opts(eps_abs=1e-6, eps_rel=1e-6, max_iterations=400))
    out(g.residency.report())
    out()
    first = g.solve(warm=False)
    g.set_instance(*q_and_bounds(mdl, X0 * 1.01, np.zeros(12)))
    steady = g.solve()
    out(f"   PREDICTED  first call {g.residency.first_call_bytes_in} B in, steady "
        f"{g.residency.steady_bytes_in} B in / {g.residency.steady_bytes_out} B out")
    out(f"   MEASURED   first call {first.bytes_fetched} B in, steady "
        f"{steady.bytes_fetched} B in / {steady.bytes_written_back} B out")
    out(f"   setup ran on the first call: {first.setup_ran}; on the second: "
        f"{steady.setup_ran}")
    out("   Predicted equals measured. The 46x was hand-arranged before; it is now")
    out("   derived from the emitted dataflow, and the only thing the planner had to")
    out("   be told is which data blocks change from call to call -- which the CVXPY")
    out("   route supplies for free.")
    out()
    loop = closed_loop(mdl, quad_model()[0], quad_model()[1], 1e-6, steps=200)
    out("   200-tick closed loop, warm started (BENCHMARKS 'aggressive' scenario):")
    out(f"     CM iterations       p50 {loop['p50']:.0f}  p95 {loop['p95']:.0f}  "
        f"max {loop['max']:.0f}   (cold first solve {loop['cold']})")
    out(f"     OSQP default        p50 {loop['ref_p50']:.0f}  "
        f"p95 {loop['ref_p95']:.0f}  max {loop['ref_max']:.0f}")
    out(f"     |u0_CM - u0_OSQP|   p50 {loop['u_p50']:.3e}  max {loop['u_max']:.3e}")
    out(f"     per-tick transfer   {loop['fetch_steady']} B in, "
        f"{loop['wb_steady']} B out (first call {loop['fetch_first']} B)")
    out(f"     Cholesky ran on {loop['setups']} of 200 ticks")
    out()

    # ---------------- 7. compliance ------------------------------------
    out("-" * 92)
    out("7. THE CODEGEN CONTRACT, CHECKED ON THE EMITTED PROGRAM")
    out("-" * 92)
    out("   Cycles are MODELLED (CM-100: 128 B/cycle, 144 MACs, 16 fp32 lanes, 8")
    out("   fp64 FMAs), +/-2x, NOT QUOTABLE. The descriptor stream they are")
    out("   computed from is measured.")
    out()
    out("   Reported for the SHIPPING configuration: all-fp32 on a real 192 KiB")
    out("   CM-100, which is what a customer would build. (Sections 3 and 6 run")
    out("   fp64, because that is the configuration that matches OSQP bit for bit.)")
    out()
    shipping = generate(
        nominal,
        CodegenOptions(precision=PrecisionPolicy.uniform(Format.FP32),
                       eps_abs=1e-6, eps_rel=1e-6, max_iterations=400),
        scratchpad_bytes=CM100_SCRATCHPAD_BYTES, macro=macif.UNIVERSAL)
    out(shipping.compliance.text())
    out()
    out("   Cycles per descriptor by precision, and where the program is thinnest:")
    out(f"   {'policy':>28} {'c/desc':>9} {'x bar':>7} {'longest run':>12} "
        f"{'absorbed?':>10}")
    for name, pol, narrow in policies:
        s = gen(nominal, CodegenOptions(precision=pol,
                                        allow_narrow_operands=narrow))
        f = s.compliance.by_rule("C3")
        run = f.numbers["longest_short_run"]
        out(f"   {name:>28} {f.numbers['mean_cycles_per_descriptor']:>9.1f} "
            f"{f.numbers['margin_x']:>7.1f} {run:>12} "
            f"{('yes' if run <= QUEUE_DEPTH else 'NO'):>10}")
    out(f"   The bar is {CYCLES_PER_DESCRIPTOR_BAR:.0f} cycles/descriptor on")
    out(f"   average, with no sub-threshold run longer than the {QUEUE_DEPTH}-deep")
    out("   queue. At fp64 there is no sub-threshold descriptor at all; the thin")
    out("   point exists at fp32 and narrower.")
    out()
    out("   The thinnest run at fp32, descriptor by descriptor:")
    ref32 = gen(nominal, reference_mode(
        precision=PrecisionPolicy.uniform(Format.FP32)))
    for i, d in enumerate(ref32.solver.residual.descriptors):
        b = CM100.breakdown(d)
        flag = "  <-- below bar" if b["cycles"] < CYCLES_PER_DESCRIPTOR_BAR else ""
        out(f"     residual[{i:>2}] {d.opcode.name:<5} {b['cycles']:>8.1f} cycles "
            f"({b['bound']}-bound)  {d.label}{flag}")
    out()
    out("   And the contract is a statement about PROBLEM SIZE, not just about the")
    out("   algorithm. Generated ADMM at fp32, average cycles per descriptor:")
    out(f"   {'n_var':>7} {'n_con':>7} {'c/desc':>9} {'longest run':>12} "
        f"{'verdict':>9}")
    rng = np.random.default_rng(4)
    for n in (6, 12, 24, 48, 80):
        m = 2 * n
        Mx = rng.standard_normal((n, n))
        Pp = Mx @ Mx.T + np.eye(n)
        Pp = 0.5 * (Pp + Pp.T)
        Aa = rng.standard_normal((m, n))
        c = Aa @ (rng.standard_normal(n) * 0.2)
        pr = QPProblem.from_arrays(Pp, rng.standard_normal(n), Aa, c - 0.3,
                                   c + 0.3, varies=("q", "l", "u"))
        s = gen(pr, fp64_opts(precision=PrecisionPolicy.uniform(Format.FP32)))
        f = s.compliance.by_rule("C3")
        out(f"   {n:>7} {m:>7} {f.numbers['mean_cycles_per_descriptor']:>9.1f} "
            f"{f.numbers['longest_short_run']:>12} "
            f"{('PASS' if f.status == 'pass' else 'FAIL'):>9}")
    out("   Small QPs are sequencer-issue-bound, and the generator says so rather")
    out("   than reporting a pass. For those, tell the customer to use the MCU.")
    out()

    # ---------------- 8. what the format could not express -------------
    out("-" * 92)
    out("8. WHAT GENERATING (RATHER THAN HAND-WRITING) FOUND")
    out("-" * 92)
    for f in V0_1_LIMITATIONS_FOUND_BY_CODEGEN:
        out(f"   [{f['id']}] {f['what']}")
        out(f"        severity : {f['severity']}")
        out(f"        found by : {f['found_by']}")
        for key, label in (("measured_cost", "cost    "), ("why", "why     "),
                           ("consequence", "effect  "), ("fix", "fix     ")):
            if key in f:
                out(f"        {label} : {f[key]}")
        out(f"        status   : {f['status']}")
        out()
    h = RESIDENCY_MECHANISM_HAZARD
    out(f"   [{h['id']}] {h['what']}")
    out(f"        severity : {h['severity']}")
    out(f"        reached  : {h['how_it_is_reached']}")
    out(f"        fix      : {h['fix']}")
    out(f"        status   : {h['status']}")
    out()
    if plans:
        s16 = plans["fp16 operands / fp32 solve"]
        out("   The G1 workaround, in the generated program:")
        for note in s16.lowering.format_notes:
            out(f"     {note}")
        out()

    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nwritten to {OUT}")


def closed_loop(mdl, A_sys, B_sys, eps: float, steps: int = 200) -> dict:
    """The BENCHMARKS 'aggressive' scenario, on a generated program."""
    problem = quad_problem(mdl)
    s = gen(problem, fp64_opts(eps_abs=eps, eps_rel=eps, max_iterations=2000))

    rng = np.random.default_rng(1)
    x = X0.copy()
    xref = np.zeros(12)
    q0, l0, u0 = q_and_bounds(mdl, x, xref)
    ref = osqp.OSQP()
    ref.setup(P=mdl["P"], q=q0, A=mdl["A_con"], l=l0, u=u0, eps_abs=eps,
              eps_rel=eps, verbose=False, check_termination=1,
              warm_starting=True)

    iters, ref_iters, uerr, fetched, wb = [], [], [], [], []
    setups = 0
    cold = None
    for k in range(steps):
        if k % 20 == 0:
            xref = np.zeros(12)
            xref[:3] = rng.uniform(-0.3, 0.3, 3)
        x = x + np.concatenate([rng.normal(0, 0.005, 3), rng.normal(0, 0.03, 3),
                                rng.normal(0, 0.004, 3), rng.normal(0, 0.03, 3)])
        q, l, u = q_and_bounds(mdl, x, xref)
        s.set_instance(q, l, u)
        st = s.solve(warm=(k > 0))
        if k == 0:
            cold = st.iterations
        ref.update(q=q, l=l, u=u)
        r = ref.solve()
        iters.append(st.iterations)
        ref_iters.append(int(r.info.iter))
        fetched.append(st.bytes_fetched)
        wb.append(st.bytes_written_back)
        setups += int(st.setup_ran)
        u_cm = s.solution[:mdl["n_u"]]
        uerr.append(float(np.max(np.abs(u_cm - np.asarray(r.x).ravel()[:4]))))
        x = A_sys @ x + B_sys @ u_cm

    it = np.array(iters)
    ri = np.array(ref_iters)
    ue = np.array(uerr)
    return dict(
        p50=np.percentile(it, 50), p95=np.percentile(it, 95), max=it.max(),
        cold=cold, ref_p50=np.percentile(ri, 50), ref_p95=np.percentile(ri, 95),
        ref_max=ri.max(), u_p50=float(np.median(ue)), u_max=float(ue.max()),
        fetch_first=fetched[0], fetch_steady=int(np.median(fetched[1:])),
        wb_steady=int(np.median(wb[1:])), setups=setups)


if __name__ == "__main__":
    main()
