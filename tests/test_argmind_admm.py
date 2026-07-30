"""ADMM on the functional simulator, validated against reference OSQP.

The milestone test. Three levels, deliberately:

1. against a numpy implementation of the *same* reduced-form algorithm, which
   isolates the descriptor sequence and the addressing;
2. against OSQP configured with identical parameters and all runtime adaptivity
   disabled, which is the reference-solver comparison;
3. on the QUAD-MPC benchmark problem, warm-started over a closed loop, which is
   the deliverable.

Every problem is checked in **two** bound configurations. With QUAD-MPC's nominal
+/-2 N per-rotor authority the optimum is interior -- the active set is empty --
so the box projection and the dual update are no-ops at convergence and a wrong
dual update is undetectable. A validation that does not force an active set is
not validating the projection kernel.
"""

from __future__ import annotations

import numpy as np
import osqp
import pytest
import scipy.linalg as sla
import scipy.sparse as sp

from argmind import (
    ArgmindError,
    CM100_SCRATCHPAD_BYTES,
    Device,
    DeviceConfig,
    Format,
    ScratchpadExhausted,
    check_dependencies,
    concat,
    macif,
)
from argmind.solvers.admm import ADMMConfig, ADMMSolver, TerminationNorm
from measure_osqp_baseline import (
    X0,
    build_condensed_qp,
    q_and_bounds,
    quadrotor_model,
)

BIG_SCRATCHPAD = 1 << 20
RNG = np.random.default_rng(20260730)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def make_solver(n, m, cfg, scratchpad=BIG_SCRATCHPAD, **kw) -> ADMMSolver:
    dev = Device(DeviceConfig(scratchpad_bytes=scratchpad, banks=16,
                              macro=macif.UNIVERSAL, strict_v0=False,
                              host_bytes=8 << 20))
    return ADMMSolver(dev, n, m, cfg, **kw)


def random_qp(n=6, m=9, bound=0.4, seed=0):
    rng = np.random.default_rng(seed)
    M = rng.normal(size=(n, n))
    P = M @ M.T + 0.5 * np.eye(n)
    A = rng.normal(size=(m, n))
    q = rng.normal(size=n)
    return P, A, q, np.full(m, -bound), np.full(m, bound)


def osqp_matched(P, A, q, l, u, cfg: ADMMConfig, eps: float):
    prob = osqp.OSQP()
    prob.setup(P=sp.csc_matrix(P), q=q, A=sp.csc_matrix(A), l=l, u=u,
               eps_abs=eps, eps_rel=eps, scaling=0, adaptive_rho=0,
               rho=cfg.rho_as_encoded, sigma=cfg.sigma_as_encoded,
               alpha=cfg.alpha_as_encoded, check_termination=1,
               warm_starting=False, polishing=False, verbose=False,
               max_iter=100_000)
    return prob.solve()


def numpy_admm(P, A, q, l, u, cfg: ADMMConfig, eps: float, max_iter=5000):
    """The same reduced-form ADMM, with the same fp32-rounded constants."""
    rho, sig, al = cfg.rho_as_encoded, cfg.sigma_as_encoded, cfg.alpha_as_encoded
    inv_rho = float(np.float32(1.0 / rho))
    n, m = P.shape[0], A.shape[0]
    L = np.linalg.cholesky(P + sig * np.eye(n) + rho * (A.T @ A))
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
        if (np.max(np.abs(Ax - z))
                <= eps + eps * max(np.max(np.abs(Ax)), np.max(np.abs(z)))
                and np.max(np.abs(Px + q + Aty))
                <= eps + eps * max(np.max(np.abs(Px)), np.max(np.abs(Aty)),
                                   np.max(np.abs(q)))):
            return x, k
    return x, max_iter


def quad_mpc(u_lim=None):
    A_sys, B_sys = quadrotor_model(coupled=True)
    mdl = build_condensed_qp(A_sys, B_sys, N=20)
    if u_lim is not None:
        mdl["u_lim"] = np.tile([u_lim] * mdl["n_u"], mdl["N"])
    return A_sys, B_sys, mdl


def active_count(res) -> int:
    return int(np.sum(np.abs(np.asarray(res.y).ravel()) > 1e-8))


# ---------------------------------------------------------------------------
# The generated program
# ---------------------------------------------------------------------------

def test_program_shape_is_what_the_sequencer_analysis_must_assume():
    cfg = ADMMConfig()
    s = make_solver(80, 320, cfg)
    counts = s.descriptor_counts()
    assert counts == {"setup": 4, "iteration": 17, "residual": 15}
    # 32 descriptors per checked iteration, of which the convergence test is
    # nearly half. The sequencer analysis assumed 28 per ADMM iteration.
    assert counts["iteration"] + counts["residual"] == 32


def test_every_block_is_dependency_clean_including_across_iterations():
    s = make_solver(12, 20, ADMMConfig())
    for block in (s.solver.setup, s.solver.iteration, s.solver.residual):
        check_dependencies(block, strict=True)
    unrolled = concat("unrolled", s.solver.setup, s.solver.iteration,
                      s.solver.residual, s.solver.iteration, s.solver.residual)
    check_dependencies(unrolled, strict=True)


def test_iteration_block_uses_only_frozen_v0_opcodes():
    """The loop body must be expressible without any proposed addition."""
    s = make_solver(12, 20, ADMMConfig(termination=TerminationNorm.L2))
    assert s.solver.iteration.uses_only_frozen_v0
    assert s.solver.setup.uses_only_frozen_v0
    assert s.solver.residual.uses_only_frozen_v0
    assert s.solver.uses_only_frozen_v0


def test_the_infinity_norm_termination_test_is_NOT_expressible_in_frozen_v0():
    """Limitation L1: OSQP's own stopping rule needs a reduction v0 lacks."""
    from argmind import Opcode, UnsupportedOpcode

    s = make_solver(12, 20, ADMMConfig(termination=TerminationNorm.LINF))
    assert not s.solver.uses_only_frozen_v0
    assert s.solver.proposed_opcodes == (Opcode.NRMI,)
    # ...and only the residual block is affected.
    assert s.solver.iteration.uses_only_frozen_v0
    assert not s.solver.residual.uses_only_frozen_v0

    strict = Device(DeviceConfig(scratchpad_bytes=BIG_SCRATCHPAD, banks=16,
                                 macro=macif.UNIVERSAL, strict_v0=True,
                                 host_bytes=8 << 20))
    with pytest.raises(UnsupportedOpcode, match="NRMI"):
        ADMMSolver(strict, 12, 20, ADMMConfig(termination=TerminationNorm.LINF))


def test_a_strict_v0_device_can_still_run_the_two_norm_variant():
    strict = Device(DeviceConfig(scratchpad_bytes=BIG_SCRATCHPAD, banks=16,
                                 macro=macif.UNIVERSAL, strict_v0=True,
                                 host_bytes=8 << 20))
    P, A, q, l, u = random_qp(seed=3)
    cfg = ADMMConfig(eps_abs=1e-8, eps_rel=1e-8, max_iterations=4000,
                     termination=TerminationNorm.L2)
    s = ADMMSolver(strict, P.shape[0], A.shape[0], cfg)
    s.set_problem(P, A)
    s.set_instance(q, l, u)
    s.reset_iterates()
    stats = s.solve(warm=True)
    assert stats.converged
    ref = osqp_matched(P, A, q, l, u, cfg, 1e-9)
    assert np.max(np.abs(s.solution - np.asarray(ref.x).ravel())) < 1e-6


# ---------------------------------------------------------------------------
# Level 1: against the same algorithm in numpy
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed,bound", [(0, 0.4), (1, 0.15), (2, 5.0)])
def test_matches_a_numpy_reference_of_the_same_algorithm(seed, bound):
    P, A, q, l, u = random_qp(bound=bound, seed=seed)
    cfg = ADMMConfig(eps_abs=1e-9, eps_rel=1e-9, max_iterations=5000)
    s = make_solver(P.shape[0], A.shape[0], cfg)
    s.set_problem(P, A)
    s.set_instance(q, l, u)
    s.reset_iterates()
    stats = s.solve(warm=True)
    x_ref, it_ref = numpy_admm(P, A, q, l, u, cfg, 1e-9)
    assert stats.converged
    assert stats.iterations == it_ref
    assert np.max(np.abs(s.solution - x_ref)) < 1e-12


# ---------------------------------------------------------------------------
# Level 2: against OSQP with identical settings
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("eps", [1e-6, 1e-8])
def test_random_qp_matches_settings_matched_osqp(eps):
    P, A, q, l, u = random_qp(bound=0.15, seed=5)
    cfg = ADMMConfig(eps_abs=eps, eps_rel=eps, max_iterations=5000)
    s = make_solver(P.shape[0], A.shape[0], cfg)
    s.set_problem(P, A)
    s.set_instance(q, l, u)
    s.reset_iterates()
    stats = s.solve(warm=True)
    ref = osqp_matched(P, A, q, l, u, cfg, eps)
    assert ref.info.status == "solved"
    assert active_count(ref) > 0, "this fixture must have an active set"
    assert stats.converged
    assert abs(stats.iterations - int(ref.info.iter)) <= 1
    assert np.max(np.abs(s.solution - np.asarray(ref.x).ravel())) < 20 * eps


# ---------------------------------------------------------------------------
# Level 3: QUAD-MPC, the deliverable
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("u_lim,expect_active", [(None, False), (0.35, True)])
def test_quad_mpc_matches_settings_matched_osqp(u_lim, expect_active):
    _, _, mdl = quad_mpc(u_lim)
    P, A = mdl["P"].toarray(), mdl["A_con"].toarray()
    q, l, u = q_and_bounds(mdl, X0, np.zeros(12))
    eps = 1e-8
    cfg = ADMMConfig(eps_abs=eps, eps_rel=eps, max_iterations=2000)
    s = make_solver(mdl["n_var"], A.shape[0], cfg)
    s.set_problem(P, A)
    s.set_instance(q, l, u)
    s.reset_iterates()
    stats = s.solve(warm=True)
    ref = osqp_matched(P, A, q, l, u, cfg, eps)

    assert ref.info.status == "solved"
    assert (active_count(ref) > 0) is expect_active
    assert stats.converged
    assert abs(stats.iterations - int(ref.info.iter)) <= 1
    # The commanded control action: the number a control engineer cares about.
    n_u = mdl["n_u"]
    du = np.max(np.abs(s.solution[:n_u] - np.asarray(ref.x).ravel()[:n_u]))
    assert du < 100 * eps, du


def test_quad_mpc_active_case_matches_osqp_to_machine_precision():
    """With the same algorithm and an unambiguous threshold crossing, the two
    implementations agree to machine precision and on the iteration count."""
    _, _, mdl = quad_mpc(0.35)
    P, A = mdl["P"].toarray(), mdl["A_con"].toarray()
    q, l, u = q_and_bounds(mdl, X0, np.zeros(12))
    for eps in (1e-6, 1e-8):
        cfg = ADMMConfig(eps_abs=eps, eps_rel=eps, max_iterations=2000)
        s = make_solver(mdl["n_var"], A.shape[0], cfg)
        s.set_problem(P, A)
        s.set_instance(q, l, u)
        s.reset_iterates()
        stats = s.solve(warm=True)
        ref = osqp_matched(P, A, q, l, u, cfg, eps)
        assert stats.iterations == int(ref.info.iter)
        assert np.max(np.abs(s.solution - np.asarray(ref.x).ravel())) < 1e-13


def test_quad_mpc_control_action_matches_a_high_accuracy_reference():
    _, _, mdl = quad_mpc()
    P, A = mdl["P"].toarray(), mdl["A_con"].toarray()
    q, l, u = q_and_bounds(mdl, X0, np.zeros(12))
    prob = osqp.OSQP()
    prob.setup(P=mdl["P"], q=q, A=mdl["A_con"], l=l, u=u, eps_abs=1e-11,
               eps_rel=1e-11, check_termination=1, polishing=True,
               verbose=False, max_iter=400_000)
    ref = prob.solve()
    u_star = np.asarray(ref.x).ravel()[:mdl["n_u"]]

    cfg = ADMMConfig(eps_abs=1e-6, eps_rel=1e-6, max_iterations=2000)
    s = make_solver(mdl["n_var"], A.shape[0], cfg)
    s.set_problem(P, A)
    s.set_instance(q, l, u)
    s.reset_iterates()
    stats = s.solve(warm=True)
    assert stats.converged
    err = np.max(np.abs(s.solution[:mdl["n_u"]] - u_star))
    # Absolute error in a thrust command of order 0.7 N.
    assert err < 1e-5, err


def test_quad_mpc_warm_started_iteration_count_is_in_the_measured_range():
    """Measured OSQP on this benchmark: warm-started counts in the 6-14 band."""
    A_sys, B_sys, mdl = quad_mpc()
    P, A = mdl["P"].toarray(), mdl["A_con"].toarray()
    cfg = ADMMConfig(eps_abs=1e-6, eps_rel=1e-6, max_iterations=2000)
    s = make_solver(mdl["n_var"], A.shape[0], cfg)
    s.set_problem(P, A)

    rng = np.random.default_rng(1)
    x = X0.copy()
    xref = np.zeros(12)
    iters = []
    for k in range(40):
        if k % 20 == 0:
            xref = np.zeros(12)
            xref[:3] = rng.uniform(-0.3, 0.3, 3)
        x = x + np.concatenate([rng.normal(0, 0.005, 3), rng.normal(0, 0.03, 3),
                                rng.normal(0, 0.004, 3), rng.normal(0, 0.03, 3)])
        q, l, u = q_and_bounds(mdl, x, xref)
        s.set_instance(q, l, u)
        stats = s.solve(warm=(k > 0))
        assert stats.converged
        iters.append(stats.iterations)
        x = A_sys @ x + B_sys @ s.solution[:mdl["n_u"]]

    p50 = float(np.percentile(iters, 50))
    assert 4 <= p50 <= 14, iters
    assert max(iters) <= 20, iters


# ---------------------------------------------------------------------------
# Residency
# ---------------------------------------------------------------------------

def test_cholesky_is_computed_once_and_stays_resident_across_calls():
    A_sys, B_sys, mdl = quad_mpc()
    P, A = mdl["P"].toarray(), mdl["A_con"].toarray()
    cfg = ADMMConfig(eps_abs=1e-6, eps_rel=1e-6, max_iterations=2000)
    s = make_solver(mdl["n_var"], A.shape[0], cfg)
    s.set_problem(P, A)
    setups = 0
    for k in range(5):
        q, l, u = q_and_bounds(mdl, X0 * (1 + 0.01 * k), np.zeros(12))
        s.set_instance(q, l, u)
        stats = s.solve(warm=(k > 0))
        setups += int(stats.setup_ran)
    assert setups == 1


def test_changing_the_plant_model_re_triggers_the_factorisation():
    A_sys, B_sys, mdl = quad_mpc()
    P, A = mdl["P"].toarray(), mdl["A_con"].toarray()
    cfg = ADMMConfig(eps_abs=1e-6, eps_rel=1e-6, max_iterations=2000)
    s = make_solver(mdl["n_var"], A.shape[0], cfg)
    s.set_problem(P, A)
    q, l, u = q_and_bounds(mdl, X0, np.zeros(12))
    s.set_instance(q, l, u)
    assert s.solve(warm=False).setup_ran
    s.set_instance(q, l, u)
    assert not s.solve(warm=True).setup_ran
    s.set_problem(P * 1.01, A)              # re-linearised plant
    s.set_instance(q, l, u)
    assert s.solve(warm=True).setup_ran


def test_steady_state_transfer_is_only_the_per_tick_delta():
    A_sys, B_sys, mdl = quad_mpc()
    P, A = mdl["P"].toarray(), mdl["A_con"].toarray()
    n, m = mdl["n_var"], A.shape[0]
    cfg = ADMMConfig(eps_abs=1e-6, eps_rel=1e-6, max_iterations=2000)
    s = make_solver(n, m, cfg)
    s.set_problem(P, A)
    first = None
    steady = []
    for k in range(4):
        q, l, u = q_and_bounds(mdl, X0 * (1 + 0.01 * k), np.zeros(12))
        s.set_instance(q, l, u)
        stats = s.solve(warm=(k > 0))
        if k == 0:
            first = stats.bytes_fetched
        else:
            steady.append(stats.bytes_fetched)
    item = Format.FP64.itemsize
    expected_delta = (n + 2 * m) * item                 # exactly q, l and u
    assert set(steady) == {expected_delta}
    assert first > 40 * expected_delta
    # P, A and the factor are never re-fetched.
    assert s.P.is_resident and s.A.is_resident and s.L.is_resident


def test_warm_start_is_the_absence_of_a_transfer_not_a_special_call():
    A_sys, B_sys, mdl = quad_mpc()
    P, A = mdl["P"].toarray(), mdl["A_con"].toarray()
    cfg = ADMMConfig(eps_abs=1e-6, eps_rel=1e-6, max_iterations=2000)
    s = make_solver(mdl["n_var"], A.shape[0], cfg)
    s.set_problem(P, A)
    q, l, u = q_and_bounds(mdl, X0, np.zeros(12))
    s.set_instance(q, l, u)
    cold = s.solve(warm=False)
    s.set_instance(q, l, u)
    warm = s.solve(warm=True)
    assert warm.iterations < cold.iterations
    # A cold start is a generation bump on x, y, z -- nothing more.
    assert warm.bytes_fetched < cold.bytes_fetched


# ---------------------------------------------------------------------------
# rho must be a power of two (limitation L5)
# ---------------------------------------------------------------------------

def test_non_power_of_two_rho_is_refused_by_default():
    cfg = ADMMConfig(rho=0.1)
    assert cfg.rho_reciprocal_error != 0.0
    with pytest.raises(ArgmindError, match="power of two"):
        make_solver(6, 9, cfg)


@pytest.mark.parametrize("rho", [0.0625, 0.125, 0.25, 0.5, 1.0, 2.0])
def test_powers_of_two_are_exactly_invertible_in_the_fp32_scalar_field(rho):
    assert ADMMConfig(rho=rho).rho_reciprocal_error == 0.0
    make_solver(6, 9, ADMMConfig(rho=rho))


def test_an_inexact_rho_stalls_when_the_active_set_is_non_empty():
    """The failure the refusal exists to prevent, demonstrated.

    Also shows why it must be demonstrated on an active-set problem: at an
    interior optimum y* = 0 anyway, so the broken fixed point is satisfied and
    the stall does not appear.
    """
    P, A, q, l, u = random_qp(bound=0.15, seed=5)
    eps = 1e-9
    bad = ADMMConfig(rho=0.1, eps_abs=eps, eps_rel=eps, max_iterations=1200)
    good = ADMMConfig(rho=0.125, eps_abs=eps, eps_rel=eps, max_iterations=1200)

    ref = osqp_matched(P, A, q, l, u, good, eps)
    assert active_count(ref) > 0

    s_bad = make_solver(P.shape[0], A.shape[0], bad, allow_inexact_rho=True)
    s_bad.set_problem(P, A)
    s_bad.set_instance(q, l, u)
    s_bad.reset_iterates()
    stats_bad = s_bad.solve(warm=True)

    s_good = make_solver(P.shape[0], A.shape[0], good)
    s_good.set_problem(P, A)
    s_good.set_instance(q, l, u)
    s_good.reset_iterates()
    stats_good = s_good.solve(warm=True)

    assert not stats_bad.converged
    assert stats_good.converged
    # rho = 0.125 is not tuned for this fixture, so the count is not small --
    # what matters is that it terminates at all, where rho = 0.1 cannot.
    assert stats_good.iterations < bad.max_iterations
    # The stall floor is set by the reciprocal mismatch, not by the tolerance.
    floor = float(s_bad.s_rp.spad_array[0])
    assert floor > eps


def test_an_inexact_rho_is_harmless_when_the_optimum_is_interior():
    """Why the guard cannot be justified by the flagship benchmark alone."""
    _, _, mdl = quad_mpc()                       # nominal bounds: active set 0
    P, A = mdl["P"].toarray(), mdl["A_con"].toarray()
    q, l, u = q_and_bounds(mdl, X0, np.zeros(12))
    cfg = ADMMConfig(rho=0.1, eps_abs=1e-9, eps_rel=1e-9, max_iterations=600)
    s = make_solver(mdl["n_var"], A.shape[0], cfg, allow_inexact_rho=True)
    s.set_problem(P, A)
    s.set_instance(q, l, u)
    s.reset_iterates()
    assert s.solve(warm=True).converged


# ---------------------------------------------------------------------------
# Precision and footprint
# ---------------------------------------------------------------------------

def test_fp32_reaches_1e_6_but_not_1e_8_on_quad_mpc():
    """Accuracy report for a precision change, as required."""
    _, _, mdl = quad_mpc()
    P, A = mdl["P"].toarray(), mdl["A_con"].toarray()
    q, l, u = q_and_bounds(mdl, X0, np.zeros(12))
    outcomes = {}
    for eps in (1e-6, 1e-8):
        cfg = ADMMConfig(eps_abs=eps, eps_rel=eps, max_iterations=400,
                         fmt=Format.FP32)
        s = make_solver(mdl["n_var"], A.shape[0], cfg)
        s.set_problem(P, A)
        s.set_instance(q, l, u)
        s.reset_iterates()
        outcomes[eps] = s.solve(warm=True).converged
    assert outcomes[1e-6] is True
    assert outcomes[1e-8] is False


def test_quad_mpc_working_set_does_not_fit_the_costed_cm100_scratchpad():
    """A finding, pinned so it cannot be lost: 128 KiB is not enough at fp32."""
    _, _, mdl = quad_mpc()
    n, m = mdl["n_var"], mdl["A_con"].shape[0]

    footprints = {}
    for fmt in (Format.FP64, Format.FP32, Format.FP16):
        s = make_solver(n, m, ADMMConfig(fmt=fmt))
        footprints[fmt] = s.device.plan.used

    assert footprints[Format.FP64] > 2 * CM100_SCRATCHPAD_BYTES
    assert footprints[Format.FP32] > CM100_SCRATCHPAD_BYTES
    assert footprints[Format.FP16] < CM100_SCRATCHPAD_BYTES

    dev = Device(DeviceConfig(scratchpad_bytes=CM100_SCRATCHPAD_BYTES, banks=16,
                              macro=macif.UNIVERSAL, strict_v0=False,
                              host_bytes=8 << 20))
    with pytest.raises(ScratchpadExhausted, match="against a 128 KiB"):
        ADMMSolver(dev, n, m, ADMMConfig(fmt=Format.FP32))


def test_iterates_and_problem_data_are_persistent_and_working_vectors_too():
    from argmind import Lifetime

    s = make_solver(6, 9, ADMMConfig())
    for buf in (s.P, s.A, s.L, s.x, s.y, s.z, s.q, s.l, s.u):
        assert buf.lifetime is Lifetime.PERSISTENT, buf.name


def test_solution_read_requires_the_output_to_have_been_written_back():
    from argmind import StaleHostCopy

    P, A, q, l, u = random_qp(seed=7)
    cfg = ADMMConfig(eps_abs=1e-6, eps_rel=1e-6, max_iterations=2000)
    s = make_solver(P.shape[0], A.shape[0], cfg)
    s.set_problem(P, A)
    s.set_instance(q, l, u)
    s.reset_iterates()
    s.solve(warm=True, outputs=[])            # ask for nothing back
    with pytest.raises(StaleHostCopy):
        _ = s.solution
    s.set_instance(q, l, u)
    s.solve(warm=True)                        # default output is x
    assert s.solution.shape == (P.shape[0],)
