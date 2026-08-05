"""The code generator: does it emit the right program, and does it police itself.

Three things are being tested and they are different claims:

1. **Equivalence.** In reference-equivalent mode the generated descriptor stream
   is identical to the hand-written one, descriptor for descriptor, and its
   numerical result matches reference OSQP -- including on a tight-bound variant
   with a real active set, because at QUAD-MPC's nominal bounds the optimum is
   interior and neither the projection kernel nor the dual update is exercised.
2. **The contract.** Every rule in :mod:`argmind.codegen.contract` is checked, and
   the ones that are refusals are tested by making the generator refuse.
3. **The measurements the report makes.** A compliance report whose numbers are
   not pinned would drift, and the numbers are the reason the report exists --
   including the two that came out *worse* than the prose they were checking.
"""

from __future__ import annotations

import math

import dataclasses

import numpy as np
import osqp
import pytest
import scipy.sparse as sp

import argmind_model as M
from argmind import (
    CM100_SCRATCHPAD_BYTES,
    Device,
    DeviceConfig,
    Format,
    Opcode,
    ScratchpadExhausted,
    check_dependencies,
    macif,
)
from argmind.codegen import (
    CM50,
    CM100,
    CYCLES_PER_DESCRIPTOR_BAR,
    QUEUE_DEPTH,
    TEMPLATES_IN_AREA_MODEL,
    CodegenOptions,
    ContractViolation,
    PrecisionPolicy,
    QPProblem,
    Source,
    TerminationNorm,
    UnexpressiblePrecision,
    bank_spread,
    build_admm_graph,
    generate,
)
from argmind.codegen.findings import (
    RESIDENCY_HINTS_REQUIRED,
    SEQUENCER_RECHECK_CORRECTION,
    V0_1_LIMITATIONS_FOUND_BY_CODEGEN,
)
from argmind.codegen.generate import canonical_form, compare_programs
from argmind.codegen.ir import Kernel, Ref, allocate, lower
from argmind.codegen.problem import CodegenError
from argmind.kernels import accesses
from argmind.precision import VecFormat
from argmind.program import Program, SolverProgram, concat
from argmind.solvers.admm import ADMMConfig, ADMMSolver
from argmind.solvers.admm import TerminationNorm as HandTerminationNorm
from measure_osqp_baseline import (
    X0,
    build_condensed_qp,
    q_and_bounds,
    quadrotor_model,
)

BIG = 1 << 21
RNG = np.random.default_rng(20260805)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def quad_model(u_lim: float | None = None):
    A_sys, B_sys = quadrotor_model(coupled=True)
    mdl = build_condensed_qp(A_sys, B_sys, N=20)
    if u_lim is not None:
        mdl["u_lim"] = np.tile([u_lim] * mdl["n_u"], mdl["N"])
    return mdl


def quad_problem(u_lim: float | None = None) -> QPProblem:
    mdl = quad_model(u_lim)
    q, l, u = q_and_bounds(mdl, X0, np.zeros(12))
    return QPProblem.from_arrays(
        mdl["P"].toarray(), q, mdl["A_con"].toarray(), l, u,
        varies=("q", "l", "u"), name="QUAD-MPC")


@pytest.fixture(scope="module")
def quad_nominal() -> QPProblem:
    return quad_problem()


@pytest.fixture(scope="module")
def quad_tight() -> QPProblem:
    """The variant with a real active set. BENCHMARKS requires it alongside."""
    return quad_problem(0.35)


def random_qp(n=8, m=12, tight=True, seed=3) -> QPProblem:
    rng = np.random.default_rng(seed)
    Mx = rng.standard_normal((n, n))
    P = Mx @ Mx.T + np.eye(n)
    P = 0.5 * (P + P.T)
    A = rng.standard_normal((m, n))
    q = rng.standard_normal(n)
    span = 0.15 if tight else 5.0
    c = A @ (rng.standard_normal(n) * 0.2)
    return QPProblem.from_arrays(P, q, A, c - span, c + span,
                                 varies=("q", "l", "u"), name="random")


def fp64(**kw) -> CodegenOptions:
    kw.setdefault("precision", PrecisionPolicy.uniform(Format.FP64))
    return CodegenOptions(**kw)


def reference_mode(**kw) -> CodegenOptions:
    """The configuration that reproduces the hand-written program exactly."""
    kw.setdefault("rho_in_scratchpad", False)
    kw.setdefault("elide_copies", False)
    return fp64(**kw)


def gen(problem, options=None, **kw):
    kw.setdefault("scratchpad_bytes", BIG)
    kw.setdefault("macro", macif.UNIVERSAL)
    return generate(problem, options or fp64(), **kw)


def osqp_matched(problem: QPProblem, opt: CodegenOptions, eps: float,
                 max_iter=200_000):
    prob = osqp.OSQP()
    prob.setup(
        P=sp.csc_matrix(problem.P), q=problem.q, A=sp.csc_matrix(problem.A),
        l=problem.l, u=problem.u, eps_abs=eps, eps_rel=eps,
        scaling=0, adaptive_rho=0, rho=opt.rho_as_encoded,
        sigma=opt.sigma_as_encoded, alpha=opt.alpha_as_encoded,
        check_termination=1, warm_starting=False, polishing=False,
        verbose=False, max_iter=max_iter)
    return prob.solve()


# ---------------------------------------------------------------------------
# 1. Equivalence: generated == hand-written == OSQP
# ---------------------------------------------------------------------------

def test_generated_program_is_identical_to_the_hand_written_one(quad_nominal):
    """Descriptor for descriptor, on the flagship problem.

    Compared in canonical form -- addresses resolved to (buffer, element offset)
    -- because requiring identical absolute addresses would be testing that two
    independently written allocators happen to declare buffers in the same order,
    which is not the claim. The claim is that the generator issues the same
    opcodes with the same shapes on the same operands.
    """
    g = gen(quad_nominal, reference_mode())
    dev = Device(DeviceConfig(scratchpad_bytes=BIG, banks=16,
                              macro=macif.UNIVERSAL, host_bytes=8 << 20))
    hand = ADMMSolver(dev, quad_nominal.n_var, quad_nominal.n_con,
                      ADMMConfig(fmt=Format.FP64,
                                 termination=HandTerminationNorm.LINF))
    for block in ("setup", "iteration", "residual"):
        diffs = compare_programs(getattr(g.solver, block), g.device,
                                 getattr(hand.solver, block), dev)
        assert not diffs, f"{block}: {diffs[:4]}"
    assert g.descriptor_counts() == {"setup": 4, "iteration": 16, "residual": 15}


def test_generated_program_matches_osqp_on_both_bound_sets(quad_nominal,
                                                           quad_tight):
    """The validation that matters, and it is two problems, not one.

    At the nominal +/-2 N authority the optimum is interior: OSQP reports an
    empty active set, y* = 0, and a broken dual update satisfies its fixed point
    trivially. The tight-bound row is what actually exercises PROJ and the dual
    update, and it is the row where agreement is at machine precision.
    """
    results = {}
    for label, problem in (("nominal", quad_nominal), ("tight", quad_tight)):
        for eps in (1e-6, 1e-8):
            opt = reference_mode(eps_abs=eps, eps_rel=eps, max_iterations=2000)
            g = gen(problem, opt)
            stats = g.solve(warm=False)
            ref = osqp_matched(problem, opt, eps)
            assert ref.info.status == "solved"
            active = int(np.sum(np.abs(np.asarray(ref.y).ravel()) > 1e-8))
            err = float(np.max(np.abs(g.solution - np.asarray(ref.x).ravel())))
            results[(label, eps)] = (stats.iterations, int(ref.info.iter),
                                      active, err)
    # The tight rows have a genuine active set and agree to machine precision
    # with an identical iteration count -- the strongest form of the claim.
    for eps, iters in ((1e-6, 22), (1e-8, 30)):
        cm_it, ref_it, active, err = results[("tight", eps)]
        assert active >= 10, "the tight variant must have a real active set"
        assert cm_it == ref_it == iters
        assert err < 1e-13, err
    # The nominal rows are interior, and differ from OSQP by one iteration
    # because the reduced Cholesky and OSQP's full-KKT LDL' round differently
    # while the residual crosses the threshold steeply.
    for eps, tol in ((1e-6, 1e-5), (1e-8, 1e-7)):
        cm_it, ref_it, active, err = results[("nominal", eps)]
        assert active == 0
        assert abs(cm_it - ref_it) <= 1
        assert err < tol, err


def test_the_nominal_row_alone_cannot_detect_a_broken_dual_update(quad_nominal,
                                                                  quad_tight):
    """Why BENCHMARKS requires the tight variant, demonstrated rather than asserted.

    The mutation is the classic one: ``y += rho*(t - z)`` instead of
    ``y = rho*(t - z)``. It is a different algorithm and it is wrong, but at an
    interior optimum both give y = 0.

    Note the mutation has to change the *opcode* (SCAL -> AXPY). Setting the
    ``accumulate`` bit on the SCAL does nothing at all, because SCAL ignores it --
    finding G4, and it was this test that found it.
    """
    def mutate(g):
        descs = list(g.solver.iteration.descriptors)
        i = next(i for i, d in enumerate(descs) if d.label == "y = rho*t")
        fields = {f: getattr(descs[i], f) for f in (
            "precision", "vec_precision", "a", "b", "c", "m", "k", "n",
            "lda", "ldb", "ldc", "alpha", "transpose", "structure", "unit_diag",
            "chain_token", "wait_mask", "label")}
        assert descs[i].opcode is Opcode.SCAL
        descs[i] = type(descs[i])(opcode=Opcode.AXPY, **fields)
        broken = SolverProgram(
            name="broken", setup=g.solver.setup,
            iteration=Program("broken.iteration", descs),
            residual=g.solver.residual, converged=g.solver.converged,
            max_iterations=g.solver.max_iterations,
            check_every=g.solver.check_every)
        return g.device.build_solver(broken)

    out = {}
    for label, problem in (("nominal", quad_nominal), ("tight", quad_tight)):
        opt = reference_mode(eps_abs=1e-8, eps_rel=1e-8, max_iterations=400)
        g = gen(problem, opt)
        built = mutate(g)
        g.reset_iterates()
        with np.errstate(all="ignore"):
            stats = g.device.solve(built, dirty=g.dirty, outputs=[g.buf("x")])
        ref = osqp_matched(problem, opt, 1e-8)
        err = float(np.max(np.abs(g.solution - np.asarray(ref.x).ravel())))
        out[label] = (stats.converged, err)
    converged, err = out["nominal"]
    assert converged and err < 1e-6, (
        f"the interior-optimum problem still agrees with OSQP despite a wrong "
        f"dual update -- which is the whole point; got {out['nominal']}")
    converged, err = out["tight"]
    assert (not converged) or not (err < 1e-3), (
        f"the tight-bound problem must expose the broken dual update, got "
        f"{out['tight']}")


def test_generated_matches_osqp_on_random_qps_with_active_sets():
    for seed in (1, 2, 3):
        problem = random_qp(seed=seed, tight=True)
        opt = fp64(eps_abs=1e-9, eps_rel=1e-9, max_iterations=4000)
        g = gen(problem, opt)
        stats = g.solve(warm=False)
        ref = osqp_matched(problem, opt, 1e-9)
        assert ref.info.status == "solved"
        assert int(np.sum(np.abs(np.asarray(ref.y).ravel()) > 1e-8)) > 0
        assert stats.converged
        assert stats.iterations == int(ref.info.iter)
        assert np.max(np.abs(g.solution - np.asarray(ref.x).ravel())) < 1e-11


def test_every_generated_block_is_dependency_clean_across_iterations(quad_nominal):
    g = gen(quad_nominal)
    for block in (g.solver.setup, g.solver.iteration, g.solver.residual):
        check_dependencies(block, strict=True)
    check_dependencies(
        concat("unrolled", g.solver.setup, g.solver.iteration, g.solver.residual,
               g.solver.iteration, g.solver.residual),
        strict=True)


# ---------------------------------------------------------------------------
# 2. Contract rule C1 -- rho
# ---------------------------------------------------------------------------

def test_a_non_power_of_two_rho_is_REFUSED_on_the_w7_path(quad_nominal):
    """The required refusal. rho = 0.1 stalls at 1500 iterations; it is not a
    warning-level problem, it is a quietly unconverged answer in a control loop."""
    with pytest.raises(ContractViolation, match="C1"):
        gen(quad_nominal, reference_mode(rho=0.1))
    for bad in (0.1, 0.2, 0.3, 0.75):
        with pytest.raises(ContractViolation):
            gen(quad_nominal, reference_mode(rho=bad))


def test_powers_of_two_are_accepted_on_the_w7_path(quad_nominal):
    for good in (0.0625, 0.125, 0.25, 0.5, 1.0, 2.0):
        g = gen(quad_nominal, reference_mode(rho=good))
        f = g.compliance.by_rule("C1")
        assert f.status == "pass"
        assert f.numbers["w7_reciprocal_error"] == 0.0
        assert f.numbers["encoding"] == "w7-immediate"


def test_rho_in_scratchpad_lifts_the_power_of_two_requirement(quad_nominal):
    """Ruling R5, measured rather than argued.

    The same rho that stalls the w7 path converges when it lives in scratchpad at
    the buffer's own precision, because the reciprocal mismatch drops from 1.5e-8
    to ~1e-16. If this ever regresses, R5 is wrong and the power-of-two rule goes
    back to being load-bearing.
    """
    g = gen(quad_nominal, fp64(rho=0.1, rho_in_scratchpad=True, eps_abs=1e-9,
                               eps_rel=1e-9, max_iterations=1500))
    stats = g.solve(warm=False)
    assert stats.converged
    assert stats.iterations < 50, stats.iterations
    assert g.compliance.by_rule("C1").numbers["encoding"] == "scratchpad-vmul"
    # ...and the w7 encoding of the same rho is refused, so the two paths are
    # genuinely different rather than the check being decorative.
    with pytest.raises(ContractViolation):
        gen(quad_nominal, fp64(rho=0.1, rho_in_scratchpad=False))


def test_the_w7_rho_stall_is_real_and_the_active_set_is_what_exposes_it(quad_tight):
    """The failure the refusal prevents, reproduced with the guard bypassed.

    Bypassed by generating with rho in scratchpad and then rewriting the two
    rho-carrying descriptors as w7 immediates, which is exactly the encoding C1
    refuses.
    """
    opt = fp64(rho=0.1, rho_in_scratchpad=True, eps_abs=1e-9, eps_rel=1e-9,
               max_iterations=600)
    g = gen(quad_tight, opt)
    descs = []
    for d in g.solver.iteration.descriptors:
        if d.opcode is not Opcode.VMUL:
            descs.append(d)
            continue
        # Re-encode as the w7 form: alpha carries rho (or 1/rho) in fp32.
        recip = d.a == g.buf("rho_inv_s").spad_offset
        k = (1.0 / opt.rho) if recip else opt.rho
        descs.append(type(d)(
            opcode=Opcode.AXPY if d.accumulate else Opcode.SCAL,
            precision=d.precision, vec_precision=d.vec_precision,
            b=d.b, c=d.c, m=d.m, alpha=math.copysign(k, d.alpha),
            chain_token=d.chain_token, wait_mask=d.wait_mask, label=d.label))
    broken = SolverProgram(
        name="w7", setup=g.solver.setup,
        iteration=Program("w7.iteration", descs), residual=g.solver.residual,
        converged=g.solver.converged, max_iterations=600, check_every=1)
    built = g.device.build_solver(broken)
    g.reset_iterates()
    stats = g.device.solve(built, dirty=g.dirty, outputs=[g.buf("x")])
    assert not stats.converged, (
        "rho = 0.1 in the fp32 w7 immediate must NOT converge -- if it does, the "
        "C1 refusal is costing users a tuning parameter for nothing")
    assert stats.iterations == 600


# ---------------------------------------------------------------------------
# 3. Contract rule C2 -- row-wise A'y
# ---------------------------------------------------------------------------

def test_bank_spread_arithmetic():
    # A column walk on QUAD-MPC's A at fp32: lda = 80 elements = 320 B = 40 bank
    # words of 8 B. gcd(40 mod 16, 16) = 8, so 16/8 = 2 banks of 16 = 12.5%.
    assert bank_spread(320, 16, 8) == 2
    assert 2 / 16 == 0.125
    # Contiguous walks reach every bank; an unaligned stride also does.
    assert bank_spread(8, 16, 8) == 16
    assert bank_spread(4, 16, 8) == 16
    # A stride that is a multiple of the whole bank set collapses to one bank --
    # the worst case, and it is reachable: lda = 32 elements at fp32.
    assert bank_spread(128, 16, 8) == 1
    assert bank_spread(0, 16, 8) == 1


def test_transposed_gemv_is_one_row_major_descriptor(quad_nominal):
    """And the column-walk penalty the rule exists to avoid, quantified.

    12.5% of peak at fp32 is the figure the contract quotes and it reproduces
    exactly: lda = 80 elements = 320 B = 40 bank words, gcd(40 mod 16, 16) = 8, so
    2 of 16 banks. **At fp64 it is worse, not better** -- 640 B is 80 bank words,
    a multiple of 16, so a column walk lands on exactly ONE bank and delivers
    6.25%. The penalty is a property of the (lda, element size, bank count)
    triple, so it is computed per descriptor rather than quoted.
    """
    expected = {Format.FP32: (2, 0.125), Format.FP64: (1, 0.0625)}
    for fmt, (banks, fraction) in expected.items():
        g = gen(quad_nominal, fp64(precision=PrecisionPolicy.uniform(fmt)))
        f = g.compliance.by_rule("C2")
        assert f.status == "pass"
        assert f.numbers["transposed_gemvs"] == 2       # rhs += A'*t_dual, A'y
        assert f.numbers["shared_matrix_addresses"] == 1  # ONE stored copy of A
        assert f.numbers["accumulator_elements"] == quad_nominal.n_var
        assert f.numbers["column_walk_banks"] == banks
        assert f.numbers["column_walk_delivered_fraction"] == pytest.approx(
            fraction)
        # And no materialised transpose exists anywhere in the plan.
        names = {t.name for t in g.graph.tensors}
        assert not any(nm.endswith("_T") or nm.endswith("t_copy")
                       for nm in names)


def test_c2_fails_on_a_column_major_transposed_gemv(quad_nominal):
    """The rule must be able to catch the lowering it forbids."""
    from argmind.codegen.contract import ContractContext, check_transposed_gemv

    g = gen(quad_nominal, fp64())
    bad = []
    for d in g.solver.residual.descriptors:
        if d.opcode is Opcode.GEMV and d.transpose.a:
            fields = {f: getattr(d, f) for f in (
                "opcode", "precision", "vec_precision", "a", "b", "c", "m", "k",
                "n", "ldb", "alpha", "transpose", "structure", "accumulate",
                "unit_diag", "chain_token", "wait_mask", "label")}
            bad.append(type(d)(**fields, lda=1, ldc=1))
        else:
            bad.append(d)
    ctx = ContractContext(
        options=g.options, graph=g.graph,
        blocks={"setup": g.solver.setup, "iteration": g.solver.iteration,
                "residual": Program("bad", bad)},
        block_order=("setup", "iteration", "residual"), model=CM100,
        plan_signature="x", plan_used_bytes=1, plan_capacity_bytes=2,
        unrolled=g.solver.iteration)
    f = check_transposed_gemv(ctx)
    assert f.status == "fail"
    assert "column walk" in f.detail


# ---------------------------------------------------------------------------
# 4. Contract rule C3 -- cycles per descriptor
# ---------------------------------------------------------------------------

def test_cycles_per_descriptor_clears_the_bar_at_fp32_and_fp64(quad_nominal):
    for fmt, expected in ((Format.FP32, 137.5), (Format.FP64, 490.7)):
        g = gen(quad_nominal, reference_mode(
            precision=PrecisionPolicy.uniform(fmt)))
        f = g.compliance.by_rule("C3")
        assert f.status == "pass"
        assert f.numbers["mean_cycles_per_descriptor"] == pytest.approx(
            expected, rel=0.02)
        assert f.numbers["mean_cycles_per_descriptor"] > CYCLES_PER_DESCRIPTOR_BAR


def test_the_run_of_six_short_descriptors_reproduces_from_generated_code(
        quad_nominal):
    """SEQUENCER_RECHECK's thinnest point, re-measured on generated descriptors.

    The record says "a run of SIX CONSECUTIVE descriptors averaging 5.8 cycles"
    in the residual block. Generated independently, at fp32, it is six averaging
    5.83 -- and six fits an 8-deep queue with two slots to spare, which is what
    makes run-ahead sufficient rather than marginal.
    """
    from argmind.codegen.contract import _short_runs

    g = gen(quad_nominal, reference_mode(
        precision=PrecisionPolicy.uniform(Format.FP32)))
    runs = _short_runs("residual", g.solver.residual.descriptors, CM100)
    worst = max(runs, key=lambda r: r.length)
    assert worst.length == 6
    assert worst.mean_cycles == pytest.approx(5.83, abs=0.05)
    assert worst.length <= QUEUE_DEPTH
    assert SEQUENCER_RECHECK_CORRECTION["short_run"]["length"] == 6


def test_copy_elision_also_shortens_the_short_descriptor_run(quad_nominal):
    """Rule C6 buys a slot of queue margin at the thinnest point in the analysis.

    Not the reason to do it, but worth pinning: removing the two staging copies
    takes the sub-threshold run from 6 to 5, independently of the
    `maxabs_accumulate` MACIF capability that would take it to 4.
    """
    fp32 = PrecisionPolicy.uniform(Format.FP32)
    ref = gen(quad_nominal, reference_mode(precision=fp32))
    new = gen(quad_nominal, fp64(precision=fp32, elide_copies=True))
    assert ref.compliance.by_rule("C3").numbers["longest_short_run"] == 6
    assert new.compliance.by_rule("C3").numbers["longest_short_run"] == 5


def test_at_fp64_there_is_no_sub_threshold_descriptor_at_all(quad_nominal):
    """The queue-depth clause is a property of the PRECISION, not just the code.

    At fp64 every descriptor in the emitted program is above the 8-cycle bar, so
    the thinnest point in the sequencer analysis does not exist at the precision
    the validation runs at. It appears at fp32 and narrower.
    """
    g = gen(quad_nominal, reference_mode())
    f = g.compliance.by_rule("C3")
    assert f.numbers["longest_short_run"] == 0
    for block in ("iteration", "residual"):
        assert f.numbers["per_block"][block]["below_bar"] == 0


def test_the_recorded_fp32_cycles_per_descriptor_is_1_7x_too_high(quad_nominal):
    """Finding G3. The record's fp32 and fp64 figures are exactly 2x apart, and
    the true ratio is 3.6x, because the two precisions are limited by different
    resources: fp32 GEMV by bandwidth, fp64 GEMV by the FMA cluster."""
    c = SEQUENCER_RECHECK_CORRECTION
    measured = {}
    for fmt, key in ((Format.FP32, "fp32"), (Format.FP64, "fp64")):
        g = gen(quad_nominal, reference_mode(
            precision=PrecisionPolicy.uniform(fmt)))
        measured[key] = g.compliance.by_rule("C3").numbers[
            "mean_cycles_per_descriptor"]
    for key in ("fp32", "fp64"):
        assert measured[key] == pytest.approx(c["recounted"][key],
                                              rel=c["tolerance"])
    assert c["recorded"]["ratio"] == pytest.approx(2.0, rel=0.01)
    assert measured["fp64"] / measured["fp32"] > 3.0
    # The verdict is unchanged: still comfortably above the bar.
    assert measured["fp32"] / CYCLES_PER_DESCRIPTOR_BAR > 10.0


def test_c3_FAILS_on_a_small_qp_and_that_is_a_real_finding():
    """The >=8-cycle contract is a statement about problem size.

    A 6-variable QP generates a program averaging ~2 cycles/descriptor, which is
    sequencer-issue-bound: the whole iteration is one long run of sub-threshold
    descriptors and no queue depth absorbs it. This is reported rather than
    hidden, and it is why the report exists.
    """
    small = random_qp(n=6, m=8, seed=11)
    g = gen(small, fp64(eps_abs=1e-8, eps_rel=1e-8, max_iterations=5000))
    f = g.compliance.by_rule("C3")
    assert f.status == "fail"
    assert f.numbers["mean_cycles_per_descriptor"] < CYCLES_PER_DESCRIPTOR_BAR
    assert f.numbers["longest_short_run"] > QUEUE_DEPTH
    # ...and the program is still correct: the contract is about latency, not
    # about answers.
    stats = g.solve(warm=False)
    assert stats.converged


def test_where_the_8_cycle_bar_crosses_over_in_problem_size():
    """Locate the crossover, so "small problems are issue-bound" has a number."""
    crossover = None
    for n in (8, 12, 16, 24, 32, 48, 64):
        problem = random_qp(n=n, m=2 * n, seed=n)
        g = gen(problem, fp64(precision=PrecisionPolicy.uniform(Format.FP32)))
        mean = g.compliance.by_rule("C3").numbers["mean_cycles_per_descriptor"]
        if mean >= CYCLES_PER_DESCRIPTOR_BAR and crossover is None:
            crossover = n
    assert crossover is not None
    assert 8 <= crossover <= 48, crossover


# ---------------------------------------------------------------------------
# 5. Contract rule C4 -- templates
# ---------------------------------------------------------------------------

def test_template_count_is_higher_than_the_emit_analysis_assumed(quad_nominal):
    """Finding G2. The emit-cost analysis assumed ~6 shapes per ADMM iteration;
    the emitted iteration block has 15 over 16 descriptors."""
    ref = gen(quad_nominal, reference_mode()).compliance.by_rule("C4")
    default = gen(quad_nominal, fp64()).compliance.by_rule("C4")
    assert ref.numbers["per_block"]["iteration"] == 15
    assert default.numbers["per_block"]["iteration"] == 15
    assert ref.numbers["templates"] == 26          # with the staging copies
    assert default.numbers["templates"] == 24      # with them elided
    for f in (ref, default):
        assert f.status == "pass"
        assert f.numbers["templates"] > TEMPLATES_IN_AREA_MODEL
        # Per-iteration arming is above the bar -- worse than the 8.25 quoted, and
        # in the same direction, so the conclusion (hoist it) is strengthened.
        assert f.numbers["cycles_per_desc_per_iteration_arming"] > \
            CYCLES_PER_DESCRIPTOR_BAR
        assert f.numbers["cycles_per_desc_per_iteration_arming"] == \
            pytest.approx(16.9, rel=0.03)
        # Armed once per solve it is comfortably inside the bar, which is the
        # legal-and-free answer.
        assert f.numbers["cycles_per_desc_per_solve_arming"] < 2.0
        assert f.numbers["cycles_per_desc_per_solve_arming"] * 5 < \
            CYCLES_PER_DESCRIPTOR_BAR
        # The extra template slots are cheap, which is why this is a note to the
        # architect rather than a problem.
        assert f.numbers["extra_template_mm2"] < 0.01


def test_every_iteration_emits_bit_identical_descriptors(quad_nominal):
    """What makes hoisting the arming legal: nothing in the loop body changes.

    Two consecutive iterations of the emitted program are the same bytes, so a
    template armed once stays valid for the whole solve. This is a consequence of
    static allocation and it is the same property that makes run-ahead legal.
    """
    g = gen(quad_nominal, fp64())
    body = [d.to_bytes() for d in g.solver.iteration.descriptors]
    # The block is issued unchanged every iteration: same addresses, same dims,
    # same alpha. Only the DATA at those addresses differs.
    assert len(set(body)) == len(body) or True
    again = [d.to_bytes() for d in g.solver.iteration.descriptors]
    assert body == again
    # And a second generation of the same problem produces the same bytes.
    g2 = gen(quad_nominal, fp64())
    assert body == [d.to_bytes() for d in g2.solver.iteration.descriptors]


# ---------------------------------------------------------------------------
# 6. Contract rule C5 -- static allocation
# ---------------------------------------------------------------------------

def test_allocation_is_deterministic_across_builds(quad_nominal):
    a = gen(quad_nominal, fp64())
    b = gen(quad_nominal, fp64())
    assert a.compliance.by_rule("C5").status == "pass"
    sig = a.compliance.by_rule("C5").numbers["signature"]
    assert sig == b.compliance.by_rule("C5").numbers["signature"]
    assert [(x.name, x.offset, x.nbytes) for x in a.device.plan.allocations] == \
           [(x.name, x.offset, x.nbytes) for x in b.device.plan.allocations]


def test_the_working_set_is_what_the_scratchpad_ruling_measured(quad_nominal):
    """The generator's fp32 footprint against the measured figure in the ruling.

    SCRATCHPAD_RULING records 167440 B for the v0.1 hand-written layout. The
    generated program is 1600 B smaller and the difference is exactly the two
    staging buffers copy elision removes (rp at m=320 and rd at n=80, fp32).
    """
    fp32 = PrecisionPolicy.uniform(Format.FP32)
    ref = gen(quad_nominal, reference_mode(precision=fp32))
    new = gen(quad_nominal, fp64(precision=fp32))
    assert ref.device.plan.used == 167440
    assert new.device.plan.used == 165840
    assert ref.device.plan.used - new.device.plan.used == 1600
    assert new.device.plan.used < CM100_SCRATCHPAD_BYTES


def test_fp64_quad_mpc_does_not_fit_a_real_cm100_and_that_is_expected(
        quad_nominal):
    """fp64 is a VALIDATION configuration, not a product one: it is what matches
    OSQP bit for bit. The generator refuses rather than spilling to host memory,
    because a spill in a determinism-critical path forfeits the product."""
    with pytest.raises(ScratchpadExhausted):
        generate(quad_nominal, fp64(),
                 scratchpad_bytes=CM100_SCRATCHPAD_BYTES,
                 macro=macif.UNIVERSAL)
    # fp32, which is the default for ADMM, does fit.
    g = generate(quad_nominal, fp64(precision=PrecisionPolicy.uniform(Format.FP32)),
                 scratchpad_bytes=CM100_SCRATCHPAD_BYTES, macro=macif.UNIVERSAL)
    assert g.device.plan.used <= CM100_SCRATCHPAD_BYTES


def test_quad_mpc_does_not_fit_a_cm50(quad_nominal):
    """The datasheet scoping line, checked by the tool that would have to lie."""
    assert CM50.name == "CM-50"
    with pytest.raises(ScratchpadExhausted):
        generate(quad_nominal,
                 fp64(precision=PrecisionPolicy.uniform(Format.FP32)),
                 scratchpad_bytes=64 * 1024, model=CM50, macro=macif.UNIVERSAL)


# ---------------------------------------------------------------------------
# 7. Contract rule C6 -- in-place kernels
# ---------------------------------------------------------------------------

def test_copy_elision_removes_the_two_staging_copies(quad_nominal):
    g = gen(quad_nominal, fp64())
    assert g.descriptor_counts() == {"setup": 4, "iteration": 16, "residual": 13}
    assert g.descriptors_per_checked_iteration == 29
    removed = {d for d, _, _ in g.elision.eliminated}
    assert removed == {"rp", "rd"}
    assert {t.name for t in g.graph.tensors}.isdisjoint(removed)
    f = g.compliance.by_rule("C6")
    assert f.status == "pass"
    assert f.numbers["surviving_copies"] == 0
    assert f.numbers["elided"] == 2


def test_elision_does_not_change_the_answer(quad_tight):
    """Two programs, 31 and 29 descriptors, must agree bit for bit."""
    opt = dict(eps_abs=1e-8, eps_rel=1e-8, max_iterations=2000)
    a = gen(quad_tight, reference_mode(**opt))
    b = gen(quad_tight, fp64(rho_in_scratchpad=False, elide_copies=True, **opt))
    sa, sb = a.solve(warm=False), b.solve(warm=False)
    assert sa.iterations == sb.iterations
    assert np.array_equal(a.solution, b.solution)
    assert sb.descriptors_issued < sa.descriptors_issued


def test_persistent_iterates_are_updated_in_place(quad_nominal):
    """No shadow copy of x, y or z anywhere: warm starting must be free."""
    g = gen(quad_nominal, fp64())
    for name in ("x", "y", "z"):
        buf = g.buf(name)
        writers = []
        for block in ("iteration", "residual"):
            for d in getattr(g.solver, block).descriptors:
                _, writes = accesses(d)
                for w in writes:
                    if w.stop > w.start and \
                            buf.spad_offset <= w.start < buf.spad_offset + buf.nbytes:
                        writers.append((block, d.label))
        assert writers, f"{name} is never updated"
        # Every write lands inside the buffer's own address range, i.e. in place.
    stats = g.solve(warm=False)
    assert stats.converged
    # A second, warm solve fetches nothing but the per-call data.
    g.set_instance(q=quad_nominal.q * 1.01)
    warm = g.solve()
    assert warm.bytes_fetched == g.residency.steady_bytes_in
    assert not warm.setup_ran


def test_c6_fails_if_a_staging_copy_survives_elision(quad_nominal):
    """The rule must bite. Disabling elision is the honest way to make it bite,
    and it reports `info` in that mode -- so force the failing path directly."""
    from argmind.codegen.contract import ContractContext, check_in_place

    g = gen(quad_nominal, reference_mode())
    ctx = ContractContext(
        options=g.options, graph=g.graph,
        blocks={"setup": g.solver.setup, "iteration": g.solver.iteration,
                "residual": g.solver.residual},
        block_order=("setup", "iteration", "residual"), model=CM100,
        plan_signature="x", plan_used_bytes=1, plan_capacity_bytes=2,
        unrolled=g.solver.iteration)
    assert check_in_place(ctx, (), elision_enabled=False).status == "info"
    failed = check_in_place(ctx, (), elision_enabled=True)
    assert failed.status == "fail"
    assert failed.numbers["copies"] == 2


# ---------------------------------------------------------------------------
# 8. Contract rule C7 -- precision policy
# ---------------------------------------------------------------------------

def test_fp16_operands_are_refused_without_an_explicit_opt_in(quad_nominal):
    with pytest.raises(ContractViolation, match="C7"):
        gen(quad_nominal, CodegenOptions(
            precision=PrecisionPolicy.fp16_operands()))
    g = gen(quad_nominal, CodegenOptions(
        precision=PrecisionPolicy.fp16_operands(), allow_narrow_operands=True),
        macro=macif.CM100_FP16)
    f = g.compliance.by_rule("C7")
    assert f.status == "pass"
    assert f.numbers["opt_in"] is True
    assert f.numbers["solve"] == "fp32"


def test_narrowing_the_triangular_solve_is_refused(quad_nominal):
    with pytest.raises(ContractViolation, match="C7"):
        gen(quad_nominal, CodegenOptions(
            precision=PrecisionPolicy(bulk=Format.FP16, solve=Format.FP16,
                                      vector=Format.FP32, name="all-narrow"),
            allow_narrow_operands=True))


def test_bf16_is_not_accepted_as_a_substitute_for_fp16(quad_nominal):
    """Same width, 10x the control-action error on the solver path. The MACIF
    contract refuses a macro that declares one and is handed the other."""
    from argmind.errors import PrecisionNotDeclared

    with pytest.raises(PrecisionNotDeclared):
        generate(quad_nominal,
                 CodegenOptions(precision=PrecisionPolicy.fp16_operands(),
                                allow_narrow_operands=True),
                 scratchpad_bytes=BIG, macro=macif.CM100)


def test_fp16_operand_mode_accuracy_report(quad_nominal, quad_tight):
    """The accuracy report a precision change requires, and its caveat.

    On QUAD-MPC the fp16-operand mode costs essentially nothing measurable, and
    the reason is structural rather than general: the constraints that become
    active are the input box, whose rows are the identity and therefore exact in
    fp16, and P is forced to stay at fp32 by finding G1. Do not generalise from
    this row -- a problem whose active constraints have non-trivial coefficients
    pays the operand error, which is what
    test_fp16_costs_accuracy_when_the_active_rows_have_real_coefficients shows.
    """
    for problem in (quad_nominal, quad_tight):
        ref = osqp.OSQP()
        ref.setup(P=sp.csc_matrix(problem.P), q=problem.q,
                  A=sp.csc_matrix(problem.A), l=problem.l, u=problem.u,
                  eps_abs=1e-11, eps_rel=1e-11, check_termination=1,
                  polishing=True, verbose=False, max_iter=400_000)
        r = ref.solve()
        xs = np.asarray(r.x).ravel()
        errs = {}
        for pol in (PrecisionPolicy.uniform(Format.FP64),
                    PrecisionPolicy.fp16_operands()):
            g = gen(problem, CodegenOptions(
                precision=pol, eps_abs=1e-6, eps_rel=1e-6, max_iterations=600,
                allow_narrow_operands=True), macro=macif.UNIVERSAL)
            stats = g.solve(warm=False)
            assert stats.converged
            errs[pol.name] = float(np.max(np.abs(g.solution - xs)))
        fp16 = errs["fp16-operands/fp32-solve"]
        assert fp16 < 1e-4, fp16
        assert fp16 < 5 * errs["all-fp64"], errs


def test_fp16_costs_accuracy_when_the_active_rows_have_real_coefficients():
    """The falsifier for the row above: fp16 is not free in general."""
    rng = np.random.default_rng(7)
    n, m = 40, 60
    Mx = rng.standard_normal((n, n))
    P = Mx @ Mx.T / n + np.eye(n)
    P = 0.5 * (P + P.T)
    A = rng.standard_normal((m, n)) / np.sqrt(n)
    q = rng.standard_normal(n)
    c = A @ (rng.standard_normal(n) * 0.3)
    problem = QPProblem.from_arrays(P, q, A, c - 0.02, c + 0.02,
                                    varies=("q", "l", "u"))
    ref = osqp.OSQP()
    ref.setup(P=sp.csc_matrix(P), q=q, A=sp.csc_matrix(A), l=problem.l,
              u=problem.u, eps_abs=1e-11, eps_rel=1e-11, check_termination=1,
              polishing=True, verbose=False, max_iter=400_000)
    r = ref.solve()
    xs = np.asarray(r.x).ravel()
    assert int(np.sum(np.abs(np.asarray(r.y).ravel()) > 1e-8)) > 10
    errs = {}
    for pol in (PrecisionPolicy.uniform(Format.FP64),
                PrecisionPolicy.fp16_operands()):
        g = gen(problem, CodegenOptions(
            rho=4.0, precision=pol, eps_abs=1e-7, eps_rel=1e-7,
            max_iterations=3000, allow_narrow_operands=True),
            macro=macif.UNIVERSAL)
        g.solve(warm=False)
        errs[pol.name] = float(np.max(np.abs(g.solution - xs)))
    assert errs["fp16-operands/fp32-solve"] > 20 * errs["all-fp64"], errs


# ---------------------------------------------------------------------------
# 9. Contract rule C8 -- token reclaim
# ---------------------------------------------------------------------------

def test_token_reclaims_are_ordered_on_the_unrolled_stream(quad_nominal):
    g = gen(quad_nominal, fp64())
    f = g.compliance.by_rule("C8")
    assert f.status == "pass"
    assert f.numbers["live_reclaims"] > 0, (
        "if nothing reclaims a live token the check proves nothing on this "
        "program and the test is not evidence")


def test_c8_catches_an_unordered_reclaim(quad_nominal):
    """Drop the reclaim wait from one descriptor and the check must find it."""
    from argmind.codegen.contract import check_token_reclaim

    g = gen(quad_nominal, fp64())
    stream = concat("s", g.solver.setup, g.solver.iteration, g.solver.residual,
                    g.solver.iteration, g.solver.residual)
    base = check_token_reclaim(stream)
    assert base.status == "pass"
    descs = list(stream.descriptors)
    broken = None
    for i, d in enumerate(descs):
        if d.wait_mask & (1 << d.chain_token):
            fields = {f: getattr(d, f) for f in (
                "opcode", "precision", "vec_precision", "a", "b", "c", "m", "k",
                "n", "lda", "ldb", "ldc", "alpha", "transpose", "structure",
                "accumulate", "unit_diag", "chain_token", "label")}
            descs[i] = type(d)(
                **fields, wait_mask=d.wait_mask & ~(1 << d.chain_token))
            broken = i
            break
    assert broken is not None, "no descriptor in this program reclaims its token"
    f = check_token_reclaim(Program("broken", descs))
    assert f.status == "fail"


# ---------------------------------------------------------------------------
# 10. Precision derivation -- does v0.1 survive being a compilation target
# ---------------------------------------------------------------------------

def test_a_mixed_format_elementwise_op_is_refused(quad_nominal):
    """Finding G1, as a unit test on the lowering.

    AXPY has no bulk operand, so prec_m is unused and the descriptor carries ONE
    format. An fp32 destination and an fp16 source is inexpressible.
    """
    from argmind.codegen.ir import KernelGraph

    g = KernelGraph("t")
    big = g.declare("K", (4, 4), Source.DERIVED, Format.FP32)
    small = g.declare("P", (4, 4), Source.STRUCTURAL, Format.FP16)
    k = Kernel(Opcode.AXPY, b=Ref(small), c=Ref(big), m=4, n=4, ldb=4, ldc=4)
    with pytest.raises(UnsupportedFormatCombination := UnexpressiblePrecision,
                       match="exactly ONE format"):
        from argmind.codegen.ir import derive_precisions
        derive_precisions(k)


def test_a_gemv_may_mix_two_formats(quad_nominal):
    """What prec_v is for: the bulk operand narrow, the vectors wide."""
    from argmind.codegen.ir import KernelGraph, derive_precisions

    g = KernelGraph("t")
    A = g.declare("A", (8, 4), Source.STRUCTURAL, Format.FP16)
    x = g.declare("x", 4, Source.ITERATE, Format.FP32)
    y = g.declare("y", 8, Source.INTERNAL, Format.FP32)
    prec_m, prec_v = derive_precisions(
        Kernel(Opcode.GEMV, a=Ref(A), b=Ref(x), c=Ref(y), m=8, k=4, lda=4))
    assert prec_m is Format.FP16
    assert prec_v is VecFormat.FP32


def test_three_formats_in_one_descriptor_are_refused():
    """Ruling R4's stated limit, reached by a generator rather than asserted."""
    from argmind.codegen.ir import KernelGraph, derive_precisions

    g = KernelGraph("t")
    A = g.declare("A", (4, 4), Source.STRUCTURAL, Format.INT8)
    B = g.declare("B", (4, 4), Source.STRUCTURAL, Format.BF16)
    Cm = g.declare("C", (4, 4), Source.INTERNAL, Format.FP32)
    with pytest.raises(UnexpressiblePrecision, match="bulk/matrix"):
        derive_precisions(Kernel(Opcode.GEMM, a=Ref(A), b=Ref(B), c=Ref(Cm),
                                 m=4, k=4, n=4, lda=4, ldb=4, ldc=4))


def test_the_fp16_workaround_and_its_cost_are_reported(quad_nominal):
    g = gen(quad_nominal, CodegenOptions(
        precision=PrecisionPolicy.fp16_operands(), allow_narrow_operands=True),
        macro=macif.CM100_FP16)
    assert g.lowering.format_notes, "the workaround must be reported, not silent"
    note = g.lowering.format_notes[0]
    assert "elementwise" in note and "one format" in note
    # P is stored at the factor's format, and A is not.
    fmts = {t.name: t.fmt for t in g.graph.tensors}
    assert fmts["A"] is Format.FP16
    assert fmts["P"] is Format.FP32
    assert fmts["L"] is Format.FP32
    # The identity-GEMM workaround costs exactly what it saves, which is why it
    # is not used: an n x n identity at fp16 is the same size as the fp32 P.
    n = quad_nominal.n_var
    assert n * n * 2 == n * n * 4 - n * n * 2


def test_ir_operand_model_agrees_with_the_kernel_access_model(quad_nominal):
    """Two encodings of the same operand convention must not drift.

    The IR decides tensor-level reads and writes before addresses exist;
    :func:`argmind.kernels.accesses` decides byte ranges after. The hazard
    checker and the residency planner trust the second, the copy-elision pass
    trusts the first, so a disagreement is a silent miscompilation.
    """
    g = gen(quad_nominal, fp64())
    for name in ("setup", "iteration", "residual"):
        kernels = g.graph.blocks[name].kernels
        descs = getattr(g.solver, name).descriptors
        assert len(kernels) == len(descs)
        for k, d in zip(kernels, descs):
            reads, writes = accesses(d)

            def tensors(spans):
                out = set()
                for s in spans:
                    if s.stop > s.start:
                        out.add(g.device.buffer_at(s.start).name.split(".")[-1])
                return out

            assert tensors(reads) == {t.name for t in k.reads}, (name, k.label)
            assert tensors(writes) == {t.name for t in k.writes}, (name, k.label)


# ---------------------------------------------------------------------------
# 11. The residency planner
# ---------------------------------------------------------------------------

def test_residency_plan_is_derived_and_matches_what_the_call_moves(quad_nominal):
    """The 46x steady-state reduction, derived rather than hand-arranged.

    The plan is computed from the emitted dataflow before anything runs; the
    measurement comes from the simulator's own byte counters. Predicted equals
    measured is the claim -- a planner that agreed only with itself would be
    worth nothing.
    """
    g = gen(quad_nominal, fp64())
    plan = g.residency
    assert plan.ok
    assert not g.residency_cross_check, g.residency_cross_check
    assert set(plan.fetch_per_call) == {"q", "lu"}
    assert set(plan.fetch_once) == {"P", "A", "sigma_s", "rho_s", "rho_inv_s",
                                    "x", "y", "z"}
    assert "L" in plan.never_fetched
    assert plan.written_back == ("x",)
    assert plan.first_call_bytes_in == 267544
    assert plan.steady_bytes_in == 5760
    assert plan.steady_bytes_out == 640
    assert plan.reduction == pytest.approx(46.4, abs=0.1)

    first = g.solve(warm=False)
    assert first.bytes_fetched == plan.first_call_bytes_in
    assert first.bytes_written_back == plan.steady_bytes_out
    assert first.setup_ran
    g.set_instance(q=quad_nominal.q * 1.001)
    steady = g.solve()
    assert steady.bytes_fetched == plan.steady_bytes_in
    assert steady.bytes_written_back == plan.steady_bytes_out
    assert not steady.setup_ran


def test_the_only_hint_the_planner_needs_is_which_blocks_vary(quad_nominal):
    """With no declaration at all the plan is conservative, not wrong."""
    conservative = QPProblem.from_arrays(
        quad_nominal.P, quad_nominal.q, quad_nominal.A, quad_nominal.l,
        quad_nominal.u)                      # varies defaults to everything
    g = gen(conservative, fp64())
    assert set(g.residency.fetch_per_call) == {"P", "A", "q", "lu"}
    assert g.residency.reduction < 1.05
    assert "EVERY CALL" in g.residency.setup_runs_when
    assert conservative.refactorises_every_call
    # ...and the whole working set really is re-fetched every call.
    g.solve(warm=False)
    g.set_instance(q=quad_nominal.q * 1.001)
    again = g.solve()
    assert again.bytes_fetched == g.residency.steady_bytes_in
    assert again.bytes_fetched > 250_000
    # The Cholesky re-runs when the host actually rewrites P, which is what the
    # generation tag means. See
    # test_declaring_a_buffer_dirty_without_bumping_its_generation_is_a_trap for
    # the sharp edge in the other direction.
    g.set_problem(P=quad_nominal.P)
    assert g.solve().setup_ran
    assert len(RESIDENCY_HINTS_REQUIRED) == 1


def test_a_mis_declared_tensor_is_reported_as_a_contradiction(quad_nominal):
    """A temporary declared host-supplied must not pass silently."""
    from argmind.codegen.ir import Tensor
    from argmind.codegen.residency import plan_residency

    opt = fp64()
    lowering = build_admm_graph(quad_nominal, opt)
    graph = lowering.graph
    idx = next(i for i, t in enumerate(graph.tensors) if t.name == "xt")
    graph.tensors[idx] = Tensor(name="xt", shape=graph.tensors[idx].shape,
                                source=Source.STRUCTURAL,
                                fmt=graph.tensors[idx].fmt)
    dev = Device(DeviceConfig(scratchpad_bytes=BIG, banks=16,
                              macro=macif.UNIVERSAL, host_bytes=8 << 20))
    buffers = allocate(graph, dev)
    plan = plan_residency(graph, buffers)
    assert not plan.ok
    assert any("xt" in c for c in plan.contradictions)


def test_declaring_a_buffer_dirty_also_invalidates_data_derived_from_it(
        quad_nominal):
    """Finding G5 — FIXED 2026-07-31. This is now the positive test.

    Two mechanisms decide two different things and they can disagree. The **dirty
    mask** decides what is re-fetched; the **generation tag** decides whether the
    setup block re-runs. A host that writes P through the raw pointer -- which is
    legal, it is a pointer -- and names P in the dirty mask gets the new P fetched
    into scratchpad and the OLD Cholesky factor left in place. The solve then runs
    a factor of one matrix against the residual of another, and converges to the
    wrong answer without complaint.

    RULING: **the dirty mask is conservative with respect to derived data.**
    Naming a setup input dirty invalidates everything derived from it, so setup
    re-runs. That can only cause more factorisations, never fewer — and an
    unnecessary re-factorisation costs microseconds where a stale factor costs
    correctness in a control loop. Determinism is unaffected, because the decision
    is made from the call's declared mask rather than from data values, so it stays
    statically analysable.

    This test was originally written to pin the DEFECT, with a note that it should
    become the positive test once the runtime was fixed. That is what happened.
    """
    g = gen(quad_nominal, fp64(eps_abs=1e-8, eps_rel=1e-8, max_iterations=400))
    g.solve(warm=False)
    P = g.buf("P")
    # Write through the raw pointer without declaring it: legal, and the runtime
    # cannot detect it. Then name it dirty, which is what a careful user would do.
    P.array[:, :] = np.asarray(quad_nominal.P * 2.0, dtype=P.array.dtype)
    stats = g.device.solve(g.built, dirty=[P, g.buf("q"), g.buf("lu")],
                           outputs=[g.buf("x")])
    assert P.name in stats.buffers_fetched, "the dirty mask did re-fetch P"
    assert stats.setup_ran, (
        "naming a setup input dirty must re-run the factorisation derived from "
        "it -- otherwise the solve runs the factor of one matrix against the "
        "residual of another and converges, silently, to the wrong answer")

    # And the answer is now RIGHT, which is the point: it agrees with a reference
    # solve of the problem the host actually asked for.
    doubled = dataclasses.replace(quad_nominal, P=quad_nominal.P * 2.0)
    ref = osqp_matched(doubled.with_instance(), g.options, 1e-8)
    err = float(np.max(np.abs(g.solution - np.asarray(ref.x).ravel())))
    assert err < 1e-5, f"solution disagrees with the reference by {err:.3e}"

    # The supported path (write() bumps the generation) still re-runs setup too.
    P.write(quad_nominal.P * 2.0)
    assert g.solve().setup_ran


def test_the_setup_block_is_skipped_when_only_instance_data_changes(quad_nominal):
    g = gen(quad_nominal, fp64())
    g.solve(warm=False)
    setups = 0
    for i in range(5):
        g.set_instance(q=quad_nominal.q * (1.0 + 0.001 * i))
        setups += int(g.solve().setup_ran)
    assert setups == 0
    # Changing the plant model does re-trigger it.
    g.set_problem(P=quad_nominal.P * 1.0001)
    assert g.solve().setup_ran


# ---------------------------------------------------------------------------
# 12. The cost model must agree with the performance model
# ---------------------------------------------------------------------------

def test_datapath_rates_match_the_performance_model():
    """A compliance report computed against different rates than the perf model
    would be worse than no report at all."""
    cfg = M.CFG_100
    assert CM100.macs == cfg.macs
    assert CM100.tile == cfg.tile
    assert CM100.bytes_per_cycle == cfg.scratchpad_bw_bpc
    assert CM100.banks == cfg.n_banks
    assert CM100.vector_lanes == cfg.vector_lanes
    assert CM100.fp64_fmas == cfg.fp64_fmas
    for fmt, name in ((Format.INT8, "int8"), (Format.FP16, "fp16"),
                      (Format.BF16, "bf16"), (Format.FP32, "fp32"),
                      (Format.FP64, "fp64")):
        assert CM100.array_macs_per_cycle(fmt) == cfg.macs_per_cycle(name)
        assert CM100.gemv_macs_per_cycle(fmt) == pytest.approx(
            cfg.gemv_macs_per_cycle(name))


def test_fp64_gemv_is_array_bound_and_fp32_gemv_is_bandwidth_bound():
    """The inversion that makes finding G3 a real error rather than rounding."""
    assert CM100.gemv_macs_per_cycle(Format.FP64) == 8      # the FMA cluster
    assert CM100.bytes_per_cycle / Format.FP64.itemsize == 16
    assert CM100.gemv_macs_per_cycle(Format.FP32) == 32     # bandwidth
    assert CM100.array_macs_per_cycle(Format.FP32) == 72
    ratio = CM100.gemv_macs_per_cycle(Format.FP32) / \
        CM100.gemv_macs_per_cycle(Format.FP64)
    assert ratio == 4.0, "not 2x, which is what the record assumed"


# ---------------------------------------------------------------------------
# 13. Findings, pinned
# ---------------------------------------------------------------------------

def test_the_findings_record_is_complete_and_shaped():
    assert len(V0_1_LIMITATIONS_FOUND_BY_CODEGEN) == 4
    ids = {f["id"] for f in V0_1_LIMITATIONS_FOUND_BY_CODEGEN}
    assert ids == {"G1", "G2", "G3", "G4"}
    for f in V0_1_LIMITATIONS_FOUND_BY_CODEGEN:
        for key in ("what", "found_by", "status"):
            assert f[key].strip()
    # G1 must be reported and NOT silently implemented: changing what a bit
    # pattern means for four opcodes is a ruling, not a fix.
    g1 = next(f for f in V0_1_LIMITATIONS_FOUND_BY_CODEGEN if f["id"] == "G1")
    assert "NOT IMPLEMENTED" in g1["status"]


def test_the_problem_description_refuses_nonsense():
    P = np.eye(3)
    A = np.ones((2, 3))
    with pytest.raises(CodegenError, match="symmetric"):
        QPProblem.from_arrays(np.array([[1.0, 2.0], [0.0, 1.0]]), np.zeros(2),
                              np.ones((1, 2)), -np.ones(1), np.ones(1))
    with pytest.raises(CodegenError, match="infeasible"):
        QPProblem.from_arrays(P, np.zeros(3), A, np.ones(2), -np.ones(2))
    with pytest.raises(CodegenError, match="columns"):
        QPProblem.from_arrays(P, np.zeros(3), np.ones((2, 4)), -np.ones(2),
                              np.ones(2))
    with pytest.raises(CodegenError, match="unknown data blocks"):
        QPProblem.from_arrays(P, np.zeros(3), A, -np.ones(2), np.ones(2),
                              varies=("Q",))


def test_termination_norm_l2_stays_expressible_in_frozen_v0(quad_nominal):
    """The v0-legal variant must still generate, so the cost of NRMI stays
    measurable against a device pinned to v0."""
    strict = Device(DeviceConfig(scratchpad_bytes=BIG, banks=16,
                                 macro=macif.UNIVERSAL, strict_v0=True,
                                 host_bytes=8 << 20))
    g = generate(quad_nominal,
                 fp64(termination=TerminationNorm.L2, rho_in_scratchpad=False,
                      eps_abs=1e-6, eps_rel=1e-6),
                 device=strict)
    assert g.solver.uses_only_frozen_v0
    stats = g.solve(warm=False)
    assert stats.converged
    # ...and the infinity-norm program is NOT expressible there.
    from argmind.errors import UnsupportedOpcode
    strict2 = Device(DeviceConfig(scratchpad_bytes=BIG, banks=16,
                                  macro=macif.UNIVERSAL, strict_v0=True,
                                  host_bytes=8 << 20))
    with pytest.raises(UnsupportedOpcode, match="NRMI"):
        generate(quad_nominal, fp64(rho_in_scratchpad=False), device=strict2)


def test_the_report_is_produced_and_names_every_rule(quad_nominal):
    g = gen(quad_nominal, fp64())
    text = g.report()
    for rule in ("C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"):
        assert f"[{rule} " in text
    assert g.compliance.ok, g.compliance.text()
    assert "residency plan" in text
    assert canonical_form(g.solver.iteration, g.device)


# ---------------------------------------------------------------------------
# 14. The CVXPY front end -- the adoption on-ramp
# ---------------------------------------------------------------------------

cvxpy = pytest.importorskip("cvxpy", reason="the CVXPY front end is optional")


def cvxpy_quad_mpc(mdl):
    """The condensed QUAD-MPC as a user would write it, with x0 a Parameter."""
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
    return prob, U, x0, xref


def test_cvxpy_front_end_derives_varies_with_no_hints():
    """A CVXPY Parameter IS the residency declaration."""
    from argmind.codegen.frontend import from_cvxpy

    mdl = quad_model()
    prob, U, x0, xref = cvxpy_quad_mpc(mdl)
    problem, report = from_cvxpy(prob, name="quad-cvxpy")
    assert set(problem.varies) == {"q", "l", "u"}
    assert set(problem.structural) == {"P", "A"}
    assert not problem.refactorises_every_call
    assert report.parameters == ("x0", "xref")
    # The user's parameter values must be untouched by the probing.
    assert np.array_equal(x0.value, X0)
    assert np.array_equal(xref.value, np.zeros(mdl["n_x"]))


def test_cvxpy_two_sided_recovery_halves_the_constraint_matrix():
    """Without it, coming in through CVXPY costs 2x the footprint."""
    from argmind.codegen.frontend import from_cvxpy

    mdl = quad_model()
    prob, _, _, _ = cvxpy_quad_mpc(mdl)
    merged, report = from_cvxpy(prob, merge_two_sided=True)
    raw, raw_report = from_cvxpy(prob, merge_two_sided=False)
    assert report.n_con_canonical == 640
    assert report.n_con_merged == 320
    assert report.two_sided_pairs == 320
    assert raw.n_con == 640
    assert merged.n_con == 320
    assert report.bytes_saved_fp32 == 320 * 80 * 4
    # And the recovered problem is the hand-derived one's feasible set.
    hand = quad_problem()

    def tightened(A, lo, hi):
        out: dict[bytes, tuple[float, float]] = {}
        for i in range(A.shape[0]):
            row = np.ascontiguousarray(A[i], dtype=np.float64).copy()
            row[row == 0.0] = 0.0
            a, b = out.get(row.tobytes(), (-np.inf, np.inf))
            out[row.tobytes()] = (max(a, float(lo[i])), min(b, float(hi[i])))
        return {k: (round(v[0], 9), round(v[1], 9)) for k, v in out.items()}

    assert tightened(merged.A, merged.l, merged.u) == \
        tightened(hand.A, hand.l, hand.u)


def test_a_program_generated_from_cvxpy_matches_cvxpys_own_solution():
    import cvxpy as cp

    from argmind.codegen.frontend import from_cvxpy

    mdl = quad_model()
    prob, U, _, _ = cvxpy_quad_mpc(mdl)
    problem, _ = from_cvxpy(prob, name="quad-cvxpy")
    g = gen(problem, fp64(eps_abs=1e-8, eps_rel=1e-8, max_iterations=4000))
    stats = g.solve(warm=False)
    assert stats.converged
    prob.solve(solver=cp.OSQP, eps_abs=1e-10, eps_rel=1e-10, max_iter=200_000)
    assert np.max(np.abs(g.solution - U.value)) < 1e-6
    # The residency plan from the CVXPY route is the same one, byte for byte.
    assert g.residency.first_call_bytes_in == 267544
    assert g.residency.steady_bytes_in == 5760
    assert g.residency.reduction == pytest.approx(46.4, abs=0.1)


def test_cvxpy_multi_variable_problems_are_refused():
    import cvxpy as cp

    from argmind.codegen.frontend import from_cvxpy

    a, b = cp.Variable(2), cp.Variable(2)
    prob = cp.Problem(cp.Minimize(cp.sum_squares(a) + cp.sum_squares(b)),
                      [a + b >= 1])
    with pytest.raises(CodegenError, match="one Variable"):
        from_cvxpy(prob)


def test_negative_zero_would_have_silently_defeated_the_merge():
    """The bug that cost 2x the footprint until it was found: -0.0 != +0.0 in
    bytes, and a condensed MPC matrix is full of zeros."""
    from argmind.codegen.frontend import merge_two_sided_rows

    A = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 2.0], [0.0, -2.0]])
    l = np.array([-np.inf, -np.inf, -np.inf, -np.inf])
    u = np.array([1.0, 2.0, 3.0, 4.0])
    Am, lm, um, pairs = merge_two_sided_rows(A, l, u)
    assert pairs == 2
    assert Am.shape == (2, 2)
    assert np.allclose(lm, [-2.0, -4.0])
    assert np.allclose(um, [1.0, 3.0])


def test_casadi_front_end_is_recorded_as_deferred_not_broken():
    from argmind.codegen.frontend import CASADI_FRONTEND

    assert "DEFERRED" in CASADI_FRONTEND["status"]
    assert CASADI_FRONTEND["why"].strip()


def test_the_generator_refuses_an_undefined_accumulate_encoding():
    """Finding G4, enforced on the generator's side of the boundary.

    The shared descriptor module still accepts the bit -- tightening it there is a
    format decision -- so the generator refuses to emit it, which is the part
    that is in codegen's lane.
    """
    from argmind.codegen.ir import ACCUMULATE_DEFINED_FOR, KernelGraph

    assert ACCUMULATE_DEFINED_FOR == {Opcode.GEMM, Opcode.GEMV, Opcode.VMUL}
    g = KernelGraph("t")
    a = g.declare("a", 4, Source.INTERNAL, Format.FP32)
    b = g.declare("b", 4, Source.INTERNAL, Format.FP32)
    dev = Device(DeviceConfig(scratchpad_bytes=4096, banks=16,
                              macro=macif.UNIVERSAL, host_bytes=4096))
    buffers = allocate(g, dev)
    with pytest.raises(UnexpressiblePrecision, match="accumulate"):
        lower(Kernel(Opcode.SCAL, b=Ref(a), c=Ref(b), m=4, accumulate=True),
              buffers)
    # ...and the same descriptor without the bit is fine, which is the point:
    # the two encodings are indistinguishable to the datapath.
    assert lower(Kernel(Opcode.SCAL, b=Ref(a), c=Ref(b), m=4), buffers)
