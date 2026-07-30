"""
Invariant and regression tests for the Argmind performance model.
=================================================================

WHY THIS FILE EXISTS

CLAUDE.md states the rule "models must be able to fail" — but until now nothing
*enforced* it. Phase 0 found nine separate modelling errors, every one caught by
manual or agent review, several of which were mechanically checkable:

  1. Overhead energy undercounted (raw MAC energy only)
  2. Matrix-vector work modelled as matrix-matrix
  3. Phantom transcendental term (box projection charged as divides)
  4. Constraint rows overcounted 2x (two-sided bound counted twice)
  5. `P@x` double-counted (P is inside the cached factorisation)
  6. A treated as dense when its input-box block is the identity
  7. Scratchpad energy constant 2.3x too low (bank->datapath net omitted)
  8. `A^T A` formed in the SQP term, which no production solver does
  9. Once-per-call `q`/bounds assembly omitted entirely

Each numbered error below has a test that FAILS IF IT RETURNS. The rest are
physical invariants: things that must hold for any believable model, whose
violation means the model has drifted into flattering nonsense.

These tests deliberately do NOT pin exact numbers. Pinning outputs would just
freeze today's estimates and make every legitimate refinement look like a
failure. They pin *relationships* and *directions* instead.
"""

import pytest

import argmind_model as M


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cfg():
    return M.ArgmindCfg(name="test-16", tile=16)


def solvers():
    return [b for b in M.BENCHMARKS if b.kind == "solver"]


def by_name(name):
    return next(b for b in M.BENCHMARKS if b.name == name)


# ---------------------------------------------------------------------------
# Regression tests: one per known past error
# ---------------------------------------------------------------------------

def test_e1_energy_exceeds_raw_mac_energy(cfg):
    """ERROR 1: energy counted only the multiplier-accumulator.

    Real designs land 2-4x above datapath-only energy once clock distribution,
    register files, operand staging and control are counted. A model whose total
    energy approaches the raw MAC figure has lost that.
    """
    for b in solvers():
        r = M.model_argmind(b, cfg)
        raw_mac_uj = b.total_macs() * M.E_MAC_PJ[b.prec] / 1e6
        assert r["energy_uj"] > 1.5 * raw_mac_uj, (
            f"{b.name}: total energy {r['energy_uj']:.3f} uJ is not meaningfully "
            f"above raw MAC energy {raw_mac_uj:.3f} uJ — overhead undercounted")


def test_e2_gemv_work_is_insensitive_to_mac_count():
    """ERROR 2: matrix-vector work modelled as matrix-matrix.

    THE settled architecture finding: at batch 1 the solver inner loop has no
    operand reuse, so it is bandwidth-bound and a bigger MAC array buys nothing.
    If this test starts failing, either the model regressed or the settled
    conclusion in CLAUDE.md is wrong — both need a human, not a silent fix.
    """
    quad = by_name("QUAD-MPC")
    small = M.model_argmind(quad, M.ArgmindCfg(name="t16", tile=16))
    big = M.model_argmind(quad, M.ArgmindCfg(name="t64", tile=64))
    assert big["latency_us"] == pytest.approx(small["latency_us"], rel=0.05), (
        "16x more MACs changed GEMV-bound latency by >5% — the GEMV/GEMM split "
        "has regressed")


def test_e3_admm_has_no_transcendentals():
    """ERROR 3: the phantom transcendental term.

    ADMM's box projection is min/max — zero divides, sqrt or exp. Charging it as
    transcendentals penalised the CPU ~3x harder than the CM (12 cycles each vs
    4/vector-lane) and was single-handedly responsible for an apparent latency
    advantage that did not exist.
    """
    admm = [b for b in solvers() if b.algo == "admm"]
    assert admm, "no ADMM benchmarks found — has `algo` tagging been dropped?"
    for b in admm:
        assert b.transcendental_per_iter == 0.0, (
            f"{b.name}: ADMM projection is min/max, not divide/sqrt. "
            "Any nonzero value here re-creates the phantom-transcendental bug")
    # And the converse: interior-point DOES have real barrier transcendentals, so
    # zeroing them everywhere would be the opposite error.
    for b in (x for x in solvers() if x.algo == "ip"):
        assert b.transcendental_per_iter > 0.0, (
            f"{b.name}: interior-point has genuine barrier divides; zero here "
            "would understate its cost")


def test_algorithm_precision_floors_are_respected():
    """Precision is a property of the algorithm, per solver-numerics.

    Interior-point KKT conditioning reaches ~1e12 near convergence -> fp64 (or
    fp32 + mandatory iterative refinement, which we do not model). The EKF
    covariance recursion squares the covariance condition number -> fp32 minimum,
    and there is NO fp16/bf16 path in any filter form.
    """
    for b in M.BENCHMARKS:
        if b.algo == "ip":
            assert b.prec == "fp64", (
                f"{b.name}: interior-point modelled in {b.prec}. Without "
                "iterative refinement (not modelled) this is optimistic")
        if b.algo == "ekf":
            assert b.prec in ("fp32", "fp64"), (
                f"{b.name}: EKF covariance in {b.prec} — no fp16/bf16 path exists")


def test_gemm_energy_tracks_REUSE_DEPTH_not_mac_count():
    """ERROR 10: MAC count and operand-reuse depth were the same number.

    GEMM operand traffic is (1/Tm + 1/Tn) bytes-elements per MAC. The model used
    2/tile, which is only correct for a square array and made "more MACs" and
    "more reuse" indistinguishable -- so the energy benefit of a bigger array got
    attributed to the MACs. A 16x8 array has HALF the MACs of 16x16 but keeps
    reuse depth 16 in one dimension, so its GEMM energy must sit strictly between
    16x16 and 8x8, nowhere near either.
    """
    ekf = by_name("EKF-96")
    e = {}
    for label, tile, tile_n in (("16x16", 16, 16), ("16x8", 16, 8), ("8x8", 8, 8)):
        c = M.ArgmindCfg(name=label, tile=tile, tile_n=tile_n)
        e[label] = M.model_argmind(ekf, c)["energy_uj"]
    assert e["16x16"] < e["16x8"] < e["8x8"], (
        f"rectangular array energy {e} is not between the two square ones — "
        "reuse depth and MAC count have been conflated again")
    # And the analytic claim that makes the array-size question one-dimensional:
    # at fixed MAC count, square minimises operand fetch.
    sq = M.ArgmindCfg(name="s", tile=12, tile_n=12).gemm_operand_bytes_per_mac("fp32")
    for tm, tn in ((16, 9), (18, 8), (24, 6), (36, 4)):
        rect = M.ArgmindCfg(name="r", tile=tm,
                            tile_n=tn).gemm_operand_bytes_per_mac("fp32")
        assert sq < rect, (
            f"{tm}x{tn} fetches fewer operand bytes/MAC than 12x12 at equal MAC "
            "count — the AM-HM argument for a square array is broken")


def test_worst_case_iteration_tail_is_characterised_and_binding():
    """The typical case cannot size a hard-real-time part.

    Every solver row must carry a worst-case iteration count, and it must be the
    one used for feasibility. At p95 iterations EVERY solver row meets its
    deadline at 16 B/cycle, so a sweep run on typical iterations reports no
    binding constraint at all and would justify any configuration whatsoever.
    """
    for b in solvers():
        assert b.iters_worst >= b.iters, (
            f"{b.name}: no worst-case iteration tail (iters_worst="
            f"{b.iters_worst} < iters={b.iters}) — feasibility is being sized "
            "against the typical case")
    # The tail must actually bind somewhere, or it is decoration.
    import dataclasses
    quad = by_name("QUAD-MPC")
    narrow = M.ArgmindCfg(name="n", tile=16, scratchpad_bw_bpc=16, n_banks=2)
    wc = M.model_argmind(dataclasses.replace(quad, iters=quad.iters_wc), narrow)
    typ = M.model_argmind(quad, narrow)
    assert typ["latency_us"] <= quad.deadline_us < wc["latency_us"], (
        "the worst-case tail does not change any feasibility verdict at the "
        "narrowest bandwidth — check it is being applied")


def test_e4_constraint_rows_match_a_real_osqp_instance():
    """ERROR 4: two-sided bounds counted as two rows.

    Cross-checked against a measured OSQP problem instance built by
    perf_model/measure_osqp_baseline.py: QUAD-MPC has 320 constraint rows
    (80 input box + 240 predicted-state box), not 640.
    """
    quad = by_name("QUAD-MPC")
    n_var, n_x, N = 20 * 4, 12, 20
    expected_rows = n_var + N * n_x          # 320, as measured
    # Recover the modelled row count from the Gamma-block GEMV term.
    annz = N * n_x * n_var
    assert annz == (expected_rows - n_var) * n_var
    assert quad.gemv_macs_per_iter < 2.0 * (2 * annz), (
        "per-iteration GEMV is far above the Gamma-block cost — constraint rows "
        "or A density may be overcounted again")


def test_e5_e6_admm_gemv_matches_hand_derivation():
    """ERRORS 5 & 6: `P@x` double-counted, and A treated as dense.

    Correct per-iteration GEMV for the reduced-form condensed ADMM:
        triangular solves      n^2
      + Gamma@x and Gamma^T@y  2*nnz(Gamma)
      + residuals amortised    (n^2/2 + nnz(Gamma)) / check_term
    P does NOT appear as a separate matvec: it lives inside the cached factor.
    A's input-box rows are the identity and cost zero MACs.
    """
    n_x, n_u, N, check = 12, 4, 20, 25
    n_var = N * n_u
    annz = N * n_x * n_var
    expected = n_var ** 2 + 2 * annz + (n_var ** 2 / 2 + annz) / check
    assert by_name("QUAD-MPC").gemv_macs_per_iter == pytest.approx(expected, rel=1e-9)


def test_e7_scratchpad_energy_includes_distribution_network(cfg):
    """ERROR 7: flat 0.40 pJ/byte omitted the bank->datapath network.

    The banked model must charge more than the legacy flat constant, since it
    now counts wire and mux/staging energy explicitly.
    """
    if not cfg.use_banked_energy:
        pytest.skip("banked energy model disabled")
    assert cfg.sram_pj_per_byte() > M.E_SRAM_BYTE_PJ, (
        "banked pJ/byte is not above the legacy flat 0.40 — the distribution "
        "network is being omitted again")


def test_e8_sqp_term_does_not_form_ata():
    """ERROR 8: the SQP re-factorisation term included forming `A^T A`.

    OSQP explicitly does not form A^T A (it destroys sparsity) and no production
    MPC stack does either. The term must be the Cholesky alone, so it must scale
    as n_var^3 and be independent of the constraint count.
    """
    a = M.condensed_mpc("a", n_x=14, n_u=7, N=30, iters=1, rate=1, note="",
                        refactor=True)
    b = M.condensed_mpc("b", n_x=28, n_u=7, N=30, iters=1, rate=1, note="",
                        refactor=True)
    # Doubling n_x doubles the constraint rows but leaves n_var unchanged.
    assert a.gemm_macs_per_call == pytest.approx(b.gemm_macs_per_call, rel=1e-9), (
        "the once-per-call GEMM depends on the constraint count — an A^T A-like "
        "term has returned")


def test_e9_once_per_call_assembly_is_counted():
    """ERROR 9: rebuilding q and the state-dependent bounds was omitted.

    It depends on the new x0, so it cannot be warm-started and sits on the
    critical path before iteration 1.

    KEYED OFF `algo`, NOT THE NAME (changed 2026-07-30). It used to test
    `b.name.endswith("MPC")`, which broke the moment PWR-MPC was re-specified as
    finite-control-set MPC -- a formulation with genuinely NO once-per-call
    assembly, because x(k) arrives as n_x floats and there is no q vector or
    state-dependent bound to rebuild. This is the second time a name-keyed
    requirement has misfired after a re-spec (ARM-MPC was the first), which is
    what the `algo` field exists to prevent.
    """
    condensing = [b for b in solvers() if b.algo in ("admm", "ip")]
    assert condensing, "no condensing-MPC rows found — has `algo` tagging broken?"
    for b in condensing:
        assert b.gemv_macs_per_call > 0.0, (
            f"{b.name}: once-per-call q/bounds assembly is not counted")
    # The converse, so the exemption cannot be abused: a row that claims no
    # once-per-call assembly must also have no iteration to warm-start into.
    for b in solvers():
        if b.gemv_macs_per_call == 0.0 and b.gemm_macs_per_call == 0.0:
            assert b.iters == 1 and b.iters_worst == 1, (
                f"{b.name}: claims no once-per-call setup yet iterates "
                f"({b.iters}/{b.iters_worst}) — an iterative solver must pay for "
                f"assembling the data its first iteration consumes")


# ---------------------------------------------------------------------------
# Physical invariants
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bw", [16, 32, 64, 128, 256])
def test_more_bandwidth_never_increases_latency(bw):
    quad = by_name("QUAD-MPC")
    lo = M.model_argmind(quad, M.ArgmindCfg(name="lo", tile=16,
                                            scratchpad_bw_bpc=bw))
    hi = M.model_argmind(quad, M.ArgmindCfg(name="hi", tile=16,
                                            scratchpad_bw_bpc=bw * 2))
    assert hi["latency_us"] <= lo["latency_us"] * 1.0001


def test_more_macs_never_increases_latency():
    for b in M.BENCHMARKS:
        small = M.model_argmind(b, M.ArgmindCfg(name="s", tile=16))
        big = M.model_argmind(b, M.ArgmindCfg(name="b", tile=32))
        assert big["latency_us"] <= small["latency_us"] * 1.0001, b.name


def test_latency_decomposition_is_consistent(cfg):
    for b in M.BENCHMARKS:
        r = M.model_argmind(b, cfg)
        parts = r["compute_us"] + r["transfer_us"] + r["overhead_us"]
        assert r["latency_us"] == pytest.approx(parts, rel=1e-9), b.name
        assert r["latency_us"] > 0 and r["energy_uj"] > 0


def test_zero_copy_never_slower_than_copy_based():
    for b in M.BENCHMARKS:
        t1 = M.model_argmind(b, M.ArgmindCfg(name="t1", tile=16, tier="T1"))
        t0 = M.model_argmind(b, M.ArgmindCfg(name="t0", tile=16, tier="T0"))
        assert t1["latency_us"] <= t0["latency_us"] * 1.0001, b.name


def test_perf_w_is_not_absurd(cfg):
    """A ratio above ~100x against a real CPU means an assumption broke.

    This is the guard that would have flagged the original 128x result, which
    came from counting only raw MAC energy.
    """
    for b in solvers():
        cm = M.model_argmind(b, cfg)
        best = min(M.model_cpu(b, c)["energy_uj"] for c in M.ALL_CPUS)
        ratio = best / cm["energy_uj"]
        assert 1.0 < ratio < 100.0, (
            f"{b.name}: perf/W {ratio:.1f}x is outside the believable band — "
            "find the responsible assumption before trusting it")


# ---------------------------------------------------------------------------
# "The model must be able to fail" — enforced, not just asserted
# ---------------------------------------------------------------------------

def test_model_can_report_a_missed_deadline(cfg):
    misses = [b.name for b in M.BENCHMARKS
              if M.model_argmind(b, cfg)["latency_us"] > b.deadline_us]
    assert misses, ("no benchmark misses its deadline in any configuration — a "
                    "model that cannot report failure is not evidence")


def test_model_can_report_the_cpu_winning(cfg):
    cpu_wins = []
    for b in M.BENCHMARKS:
        cm = M.model_argmind(b, cfg)["latency_us"]
        fastest = min(M.model_cpu(b, c)["latency_us"] for c in M.ALL_CPUS)
        if fastest < cm:
            cpu_wins.append(b.name)
    assert cpu_wins, ("the CPU never wins on latency anywhere — the baselines "
                      "are probably not being given their fair advantages")


def test_kill_criterion_is_evaluated_against_most_efficient_cpu(cfg):
    """The stated rule is perf/W vs the MOST ENERGY-EFFICIENT baseline.

    Comparing against the fastest (bigger, hungrier) core instead would inflate
    the ratio several-fold. This pins the conservative choice.
    """
    for b in solvers():
        energies = {c.name: M.model_cpu(b, c)["energy_uj"] for c in M.ALL_CPUS}
        assert min(energies.values()) < max(energies.values()), (
            "all CPU baselines have identical energy — the baseline set is not "
            "actually spanning an efficiency range")


# ---------------------------------------------------------------------------
# Settled findings from CLAUDE.md, encoded so drift is visible
# ---------------------------------------------------------------------------

def test_settled_admm_inner_loop_is_pure_gemv():
    """Keyed off `algo`, not a name list (changed 2026-07-30, PWR-MPC re-spec)."""
    admm = [b for b in M.BENCHMARKS + M.OFF_SET if b.algo == "admm"]
    assert admm, "no ADMM rows left in the model at all"
    for b in admm:
        assert b.gemm_macs_per_iter == 0.0, (
            f"{b.name}: ADMM inner loop has acquired matrix-matrix work. If that "
            "is real, CLAUDE.md's settled finding needs revising deliberately")


def test_settled_fcs_mpc_is_gemm_not_gemv_and_that_is_not_a_contradiction():
    """FCS-MPC's candidate set IS the batch dimension batch-1 solvers lack.

    This looks like it contradicts the settled "solver inner loops are GEMV"
    finding and does not: the settled finding is about ITERATIVE solvers at batch
    1. Enumeration over a finite control set is a different algorithm class with a
    real batch dimension, so it is matrix-MATRIX work.

    The load-bearing half is the SECOND assertion. The batch does not make this
    row array-bound, because the reuse depth is n_y = 2 (the number of controlled
    outputs), not the tile dimension. Arithmetic intensity must therefore stay
    well ABOVE the tile-limited figure -- if it ever drops to it, someone has
    quietly assumed the array reuses these operands 12 times and the row's
    binding constraint will be reported wrongly.
    """
    fcs = [b for b in M.BENCHMARKS + M.OFF_SET if b.algo == "fcs"]
    assert fcs, "no FCS-MPC rows found"
    for b in fcs:
        assert b.gemm_macs_per_iter > 10 * b.gemv_macs_per_iter, (
            f"{b.name}: FCS-MPC enumeration should be matrix-matrix dominated")
        assert b.gemm_bytes_per_mac is not None
        tile_limited = M.CFG_100.gemm_operand_bytes_per_mac(b.prec)
        assert b.gemm_bytes_per_mac > 2.0 * tile_limited, (
            f"{b.name}: arithmetic intensity {b.gemm_bytes_per_mac:.2f} B/MAC has "
            f"collapsed to the tile-limited figure {tile_limited:.2f} — the "
            f"n_y-limited reuse depth has been lost")
        assert b.iters == 1 and b.iters_worst == 1, (
            f"{b.name}: FCS-MPC has a FIXED op count and no convergence tail")
        assert not any(o.chain_dim > 0 for o in b.kernels), (
            f"{b.name}: FCS-MPC has no dependency chain — no Cholesky, no "
            f"triangular solve. A chain here means the decomposition is wrong")


def test_fcs_mpc_binding_resource_is_the_vector_unit():
    """The architecture finding this re-spec produced, pinned.

    On the re-specified PWR-MPC the vector unit -- cost evaluation and argmin --
    costs more cycles than the MAC array and more than operand fetch. That is the
    opposite of every other solver row and it is what would drive a lane-count or
    reduction-unit change, so it must not drift silently.
    """
    b = by_name("PWR-MPC")
    cfg = M.CFG_100
    dim = M.representative_dim(b)
    gemm = b.gemm_macs_per_iter / M.gemm_macs_per_cycle_bounded(
        cfg, b.prec, dim, b.gemm_bytes_per_mac)
    vec = b.vector_ops_per_iter / cfg.vector_lanes
    assert vec > 2.0 * gemm, (
        f"PWR-MPC vector {vec:.0f} cyc vs GEMM {gemm:.0f} cyc — the vector unit "
        f"is no longer the binding resource on the FCS row; the lane-count "
        f"recommendation needs re-deriving")


def test_gemm_bandwidth_ceiling_is_inactive_on_the_pre_existing_rows():
    """The GEMM bandwidth bound was added for FCS-MPC. Prove it changed nothing
    else, and that it is not decoration -- it must actually bind somewhere."""
    cfg = M.CFG_100
    bound_bites = []
    for b in M.BENCHMARKS:
        dim = M.representative_dim(b)
        array = cfg.macs_per_cycle(b.prec) * cfg.gemm_util(b.prec, dim)
        bpm = (cfg.gemm_operand_bytes_per_mac(b.prec)
               if b.gemm_bytes_per_mac is None else b.gemm_bytes_per_mac)
        bound_bites.append((b.name, cfg.scratchpad_bw_bpc / bpm < array))
    biting = [n for n, x in bound_bites if x]
    assert biting == ["PWR-MPC"], (
        f"GEMM bandwidth ceiling binds on {biting}; expected exactly PWR-MPC. "
        f"If it now binds elsewhere the pre-existing rows have moved; if it binds "
        f"nowhere the term is decoration")


def test_settled_ekf_covariance_is_gemm_dominated():
    for name in ("EKF-24", "EKF-96"):
        b = by_name(name)
        assert b.gemm_macs_per_iter > b.gemv_macs_per_iter, (
            f"{name}: covariance recursion should be matrix-matrix dominated")


def test_settled_iteration_count_within_measured_range():
    """Measured on real OSQP: p95=7 at 1e-3, 21 at 1e-6, cold 60-73.

    The modelled design point must sit inside the measured envelope. Outside it,
    someone has re-guessed a number that has already been measured.
    """
    assert 7 <= by_name("QUAD-MPC").iters <= 73


def test_benchmark_definitions_are_self_consistent():
    for b in M.BENCHMARKS:
        assert b.iters >= 1 and b.calls_per_sec > 0
        assert b.deadline_us == pytest.approx(1e6 / b.calls_per_sec, rel=1e-6), (
            f"{b.name}: deadline does not match the invocation rate. NOTE: real "
            "hard-real-time MPC budgets only 30-60% of the control period to the "
            "solver; if that is introduced, update this test deliberately")
        assert b.persistent_bytes >= 0 and b.delta_bytes > 0
