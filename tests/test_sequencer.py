"""
Invariants for the descriptor-issue / sequencer model.
======================================================

Added 2026-07-30, when the flat `scalar_ops_per_iter` term and the fixed
50%-compute-overlap fudge were replaced by a descriptor-derived model
(perf_model/sequencer.py). Two classes of test:

  * CONSISTENCY. The kernel decomposition must account for the benchmark's
    declared operation counts exactly. Deriving descriptor counts and MAC counts
    independently is precisely the error that double-counted HPIPM GEMM work 5.7x
    in an earlier fp64 analysis, and it is the error this whole model was built
    to avoid repeating.

  * DIRECTIONALITY. The model must remain able to report failure. A sequencer
    model in which no plausible configuration ever misses a deadline would be
    decoration, not evidence -- so one test asserts that a scalar-core,
    no-run-ahead sequencer DOES miss the PWR-MPC control-period budget.

As elsewhere in this suite, relationships and directions are pinned, not
numbers, so that legitimate refinement does not look like regression.
"""

import dataclasses

import pytest

import argmind_model as M
import sequencer as Q


REC = M.CFG_100
SOLVERS = [b for b in M.BENCHMARKS if b.kind == "solver"]


def by_name(name):
    return next(b for b in M.BENCHMARKS + M.OFF_SET if b.name == name)


def setup_cfg(setup, queue=8, base=None):
    return dataclasses.replace(base or REC, seq_setup_cycles=setup,
                               seq_queue_depth=queue)


# ---------------------------------------------------------------------------
# Consistency: the decomposition must add up
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("b", M.BENCHMARKS, ids=lambda b: b.name)
def test_kernel_decomposition_accounts_for_declared_work(b):
    """THE core discipline. Descriptor counts are only meaningful if the ops they
    describe are the same ops the benchmark charges MACs for."""
    M.assert_kernels_consistent(b)


@pytest.mark.parametrize("b", M.BENCHMARKS, ids=lambda b: b.name)
def test_every_benchmark_is_decomposed(b):
    """A benchmark without a kernel list silently falls back to the legacy flat
    scalar term, which would reintroduce the very blind spot being closed."""
    assert b.kernels, f"{b.name} has no kernel decomposition"


@pytest.mark.parametrize("b", M.BENCHMARKS, ids=lambda b: b.name)
def test_descriptor_cycles_are_physical(b):
    d = M.descriptor_cost(b, REC)
    assert d["n_desc"] > 0
    assert d["exec_cycles"] > 0
    assert d["cycles_per_desc"] > 0
    assert d["n_dep"] <= d["n_desc"]
    assert d["total_cycles"] >= d["exec_cycles"]


def test_datapath_cycles_match_the_main_model():
    """The kernel decomposition must reproduce the per-iteration datapath cycle
    count the rest of the model computes from the aggregate MAC figures.

    If these drift, the sequencer analysis is describing a different machine from
    the one the latency and energy tables describe.
    """
    for b in M.BENCHMARKS:
        d = M.descriptor_cost(b, REC)
        dim = M.representative_dim(b)
        want = (M.split_gemv_cycles(b, REC, b.gemv_macs_per_iter)
                + M.gemm_like_cycles(b.gemm_macs_per_iter, REC, b.prec, dim,
                                     bytes_per_mac=b.gemm_bytes_per_mac)
                + b.vector_ops_per_iter / REC.vector_lanes
                + b.transcendental_per_iter * 4.0 / REC.vector_lanes)
        assert d["exec_cycles"] == pytest.approx(want, rel=1e-9), b.name


# ---------------------------------------------------------------------------
# The flat scalar term must no longer reach the CM path
# ---------------------------------------------------------------------------

def test_scalar_ops_per_iter_no_longer_affects_the_cm():
    """`scalar_ops_per_iter` is now a CPU-only parameter. If changing it moves CM
    latency, the flat term has crept back into the CM model."""
    for b in SOLVERS:
        base = M.model_argmind(b, REC)["latency_us"]
        infl = M.model_argmind(
            dataclasses.replace(b, scalar_ops_per_iter=b.scalar_ops_per_iter * 100),
            REC)["latency_us"]
        assert infl == pytest.approx(base, rel=1e-12), (
            f"{b.name}: the flat scalar term still drives CM latency")


def test_legacy_mode_still_reproducible():
    """The 50%-overlap path is retained for auditability and must still work."""
    legacy = dataclasses.replace(REC, use_descriptor_seq=False)
    for b in M.BENCHMARKS:
        r = M.model_argmind(b, legacy)
        assert r["latency_us"] > 0
        assert r["seq_cycles_per_iter"] == pytest.approx(
            0.5 * b.scalar_ops_per_iter / legacy.seq_ipc)


# ---------------------------------------------------------------------------
# Monotonicity: the model must respond in the right direction
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("b", SOLVERS, ids=lambda b: b.name)
def test_more_setup_cycles_never_reduces_latency(b):
    prev = 0.0
    for s in (1, 2, 4, 8, 16, 32, 64):
        lat = M.model_argmind(b, setup_cfg(s))["latency_us"]
        assert lat >= prev - 1e-9, f"{b.name}: latency fell as setup cost rose"
        prev = lat


@pytest.mark.parametrize("b", SOLVERS, ids=lambda b: b.name)
def test_run_ahead_is_never_a_penalty(b):
    """Decoupled access-execute cannot be slower than write-and-poll."""
    for s in (4, 20):
        with_ra = M.model_argmind(b, setup_cfg(s, queue=8))["latency_us"]
        without = M.model_argmind(b, setup_cfg(s, queue=0))["latency_us"]
        assert without >= with_ra - 1e-9, f"{b.name} at setup={s}"


@pytest.mark.parametrize("b", SOLVERS, ids=lambda b: b.name)
def test_smaller_chain_block_never_reduces_latency(b):
    """A narrower blocked Cholesky means more dependent descriptors and more
    drain. If latency improved, the drain term has the wrong sign."""
    prev = None
    for blk in (16, 8, 4, 2, 1):
        lat = M.model_argmind(
            dataclasses.replace(b, iters=b.iters_wc),
            dataclasses.replace(REC, chain_block_override=blk))["latency_us"]
        if prev is not None:
            assert lat >= prev - 1e-9, f"{b.name}: chain block {blk}"
        prev = lat


def test_dependency_drain_is_not_a_bandwidth_problem():
    """The central claim about the drain term: it cannot be provisioned away.

    More scratchpad bandwidth must leave the dependency-drain cycle count
    untouched. If it does not, the term has been conflated with a resource.
    """
    for b in SOLVERS:
        ref = None
        for bw in (32.0, 64.0, 128.0, 256.0):
            c = dataclasses.replace(REC, scratchpad_bw_bpc=bw,
                                    n_banks=min(16, max(1, int(bw // 8))))
            d = M.descriptor_cost(b, c)["dep_drain_cycles"]
            if ref is None:
                ref = d
            assert d == pytest.approx(ref), (
                f"{b.name}: dependency drain moved with scratchpad bandwidth")


# ---------------------------------------------------------------------------
# The findings themselves, pinned as relationships
# ---------------------------------------------------------------------------

def test_block_structured_ip_issues_far_more_descriptors_than_admm():
    """The structural finding: HPIPM's per-stage block operations produce an
    order of magnitude more descriptors per iteration than ADMM's handful of
    large GEMVs. This is the mechanism behind the whole analysis."""
    ip = M.descriptor_cost(by_name("ARM-MPC"), REC)["n_desc"]
    admm = M.descriptor_cost(by_name("QUAD-MPC"), REC)["n_desc"]
    assert ip > 10 * admm, (
        f"ARM-MPC {ip:.0f} descriptors/iter vs QUAD-MPC {admm:.0f} — the "
        f"block-structured/ADMM granularity gap has disappeared")


def test_exposure_ordering_by_cycles_per_descriptor():
    """Exposure is set by datapath cycles per descriptor, NOT by algorithm class.

    RESTATED 2026-07-30 after the PWR-MPC re-spec. The original finding -- "the
    most exposed row is PWR-MPC, and PWR-MPC is ADMM, so do not pitch this as
    'interior point is the risky one'" -- was recorded against a continuous-
    control-set ADMM formulation that is no longer the frozen row. With PWR-MPC
    re-specified as FCS-MPC the ordering CHANGES: it is no longer the most exposed
    row, because 12 descriptors command ~54 datapath cycles each and none of them
    is chained.

    What SURVIVES, and is the durable half of the finding, is the mechanism:
    exposure tracks datapath cycles per descriptor, so small problems and
    block-structured algorithms both reduce it independently. The off-set PWR-CCS
    row still demonstrates it, and is asserted here so the mechanism keeps a
    live example.
    """
    cpd = {b.name: M.descriptor_cost(b, REC)["cycles_per_desc"] for b in SOLVERS}
    cpd["PWR-CCS"] = M.descriptor_cost(by_name("PWR-CCS"), REC)["cycles_per_desc"]
    cpd["PWR-FCS1"] = M.descriptor_cost(by_name("PWR-FCS1"), REC)["cycles_per_desc"]
    # The mechanism: block-structured IP is more exposed than a large ADMM, and a
    # SMALL problem is more exposed than either -- regardless of algorithm.
    assert cpd["ARM-MPC"] < cpd["QUAD-MPC"] < cpd["EKF-96"]
    assert cpd["PWR-CCS"] < cpd["ARM-MPC"], (
        "the small-problem exposure mechanism has disappeared from PWR-CCS")
    assert cpd["PWR-FCS1"] < cpd["PWR-MPC"], (
        "a one-step FCS horizon should be MORE descriptor-exposed than a two-step "
        "one; if not, the descriptor decomposition does not scale with the batch")
    # And the re-specified frozen row is no longer the thin one.
    assert cpd["PWR-MPC"] > cpd["ARM-MPC"], (
        "the re-specified PWR-MPC is still the most exposed row — check the "
        "re-spec actually removed the dependency chains")


def test_faster_precision_increases_sequencer_exposure():
    """Corrects the logged claim that the exposure holds 'at any precision'. It
    scales with arithmetic RATE, so fp64 on the FMA cluster is the SAFEST case
    and bf16 on the array the most exposed."""
    arm = by_name("ARM-MPC")
    x = {p: Q.crossover_setup(dataclasses.replace(arm, prec=p), REC)
         for p in ("fp64", "fp32", "bf16")}
    assert x["bf16"] < x["fp32"] < x["fp64"], (
        f"crossover setup cycles by precision: {x} — the rate/exposure "
        f"relationship has inverted")


def test_ml_rows_are_insensitive_to_the_sequencer():
    """The control case: ML descriptors command tens of thousands of datapath
    cycles, so no sequencer parameter should matter there."""
    for b in M.BENCHMARKS:
        if b.kind != "ml":
            continue
        lo = M.model_argmind(b, setup_cfg(1, queue=8))["latency_us"]
        hi = M.model_argmind(b, setup_cfg(64, queue=0))["latency_us"]
        assert hi <= 1.01 * lo, f"{b.name} moved {hi / lo:.3f}x"


def test_issue_width_above_one_buys_nothing():
    """Reported conclusion: issue WIDTH is not the axis that matters; setup COST
    is. Widening issue must not change any solver verdict."""
    for b in SOLVERS:
        one = M.model_argmind(dataclasses.replace(b, iters=b.iters_wc),
                              setup_cfg(4))["latency_us"]
        wide = M.model_argmind(
            dataclasses.replace(b, iters=b.iters_wc),
            dataclasses.replace(setup_cfg(4), seq_issue_width=4.0))["latency_us"]
        assert wide == pytest.approx(one, rel=1e-9), b.name


# ---------------------------------------------------------------------------
# The model must be able to fail
# ---------------------------------------------------------------------------

def margin(b, cfg):
    """Worst-case margin against the 50%-of-control-period engineering budget."""
    wc = dataclasses.replace(b, iters=b.iters_wc)
    return (wc.deadline_us * Q.PERIOD_BUDGET
            / M.model_argmind(wc, cfg)["latency_us"])


def test_a_plausible_sequencer_misses_a_deadline_budget():
    """CLAUDE.md: "models must be able to fail". A sequencer model under which no
    plausible configuration ever misses anything is decoration.

    DOWNGRADED AND RESTATED 2026-07-30 (PWR-MPC re-spec). The negative result used
    to be: a scalar-core write-and-poll sequencer (no run-ahead) misses PWR-MPC's
    50%-of-period budget at 0.66x. That rested on PWR-MPC being a continuous-
    control-set ADMM problem. Re-specified as FCS-MPC the row has NO dependency
    chain and 54 datapath cycles per descriptor, so it survives no-run-ahead at
    6.5x and is completely insensitive to chain blocking.

    **Consequence, stated plainly because it is a downgrade: no row in the frozen
    set misses a budget from either pathology ALONE any more.** What survives is
    that you may skip at most ONE of the two -- QUAD-MPC, at its MEASURED
    worst-case iteration tail, misses at 0.37x when both are absent. Both terms
    therefore stay load-bearing and the model can still fail; the "only deadline
    miss in the analysis" framing is retired.
    """
    quad = by_name("QUAD-MPC")
    both_absent = dataclasses.replace(REC, seq_setup_cycles=20,
                                      seq_queue_depth=0, chain_block_override=1)
    assert margin(quad, both_absent) < 1.0, (
        f"QUAD-MPC now meets its worst-case budget with neither run-ahead nor "
        f"wide chain blocking ({margin(quad, both_absent):.2f}x) — the sequencer "
        f"model has gone slack and can no longer report any failure")
    assert margin(quad, REC) > 1.0, (
        "the recommended configuration misses QUAD-MPC's worst-case budget — a "
        "real architecture problem, not a test failure")
    # Each pathology alone must still cost real margin, or the requirement is
    # unfalsifiable in the other direction.
    for label, cfg in (("no run-ahead", setup_cfg(20, queue=0)),
                       ("scalar chain blocking",
                        dataclasses.replace(REC, chain_block_override=1))):
        assert margin(quad, cfg) < 0.9 * margin(quad, REC), (
            f"{label} costs QUAD-MPC no margin — that term is not load-bearing")


def test_the_respecified_pwr_row_is_insensitive_to_both_sequencer_pathologies():
    """The other half of the downgrade, asserted so it cannot be quietly undone.

    If someone re-derives a run-ahead or chain-blocking requirement from the
    motor-drive row, this fails and forces them to say which row they used.
    """
    pwr = by_name("PWR-MPC")
    base = margin(pwr, REC)
    assert margin(pwr, dataclasses.replace(REC, chain_block_override=1)) \
        == pytest.approx(base, rel=1e-9), (
        "the re-specified PWR-MPC has acquired sensitivity to chain blocking — "
        "FCS-MPC has no dependency chain, so this means a chain_dim crept in")
    assert margin(pwr, setup_cfg(20, queue=0)) > 0.7 * base, (
        "the re-specified PWR-MPC has become run-ahead-sensitive")
    assert base > 3.0, (
        f"the re-specified PWR-MPC budget margin is only {base:.2f}x — if this "
        f"falls the row is back to being a binding constraint")


def test_narrow_chain_blocking_breaks_a_deadline():
    """The RTL requirement that came out of this: a blocked Cholesky must advance
    a full datapath width per dependent step.

    Kept alive on the OFF-SET PWR-CCS row (the retired continuous-control-set
    definition), and labelled as such. A requirement whose only live example is
    off the frozen set is a weaker requirement, and the test says so rather than
    hiding it behind a frozen-set name.
    """
    ccs = by_name("PWR-CCS")
    assert margin(ccs, dataclasses.replace(REC, chain_block_override=1)) < 1.0, (
        "a scalar-sequential dependency chain no longer breaks even the off-set "
        "PWR-CCS budget — the drain term may have gone slack")
    assert margin(ccs, setup_cfg(20, queue=0)) < 1.0, (
        "no-run-ahead no longer breaks even the off-set PWR-CCS budget")
    assert margin(ccs, REC) > 1.0, (
        "the recommended configuration misses the off-set PWR-CCS budget")


def test_recommended_config_meets_every_solver_budget():
    """The positive counterpart. If this fails the part is mis-provisioned."""
    for b in SOLVERS:
        bw = dataclasses.replace(b, iters=b.iters_wc)
        lat = M.model_argmind(bw, REC)["latency_us"]
        assert lat <= b.deadline_us * Q.PERIOD_BUDGET, (
            f"{b.name}: {lat:.1f} us vs {b.deadline_us * Q.PERIOD_BUDGET:.1f} us "
            f"budget at the recommended configuration")


# ---------------------------------------------------------------------------
# Area
# ---------------------------------------------------------------------------

def test_sequencer_fits_its_area_allocation():
    """The bottom-up cost must stay inside cm100_sizing's REST_MM2 sequencer
    line, or the 0.698 mm2 core area is wrong."""
    total = Q.seq_logic_mm2(True, 8) + Q.seq_mem_mm2()
    assert total <= 0.045, (
        f"sequencer costed at {total:.4f} mm2 against a 0.045 mm2 allocation — "
        f"core area must be restated")
    # And it must not be so far under that the allocation is meaningless.
    assert total >= 0.010, "sequencer area estimate has become implausibly small"


def test_descriptor_generator_is_cheap_relative_to_the_core():
    delta = Q.seq_logic_mm2(True, 8) - Q.seq_logic_mm2(False, 0)
    assert delta / 0.698 < 0.02, (
        "the descriptor generator is no longer a rounding error against core "
        "area; the 'it is nearly free' claim needs restating")
