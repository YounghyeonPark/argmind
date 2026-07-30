"""
Invariants for the SRAM cost model and the OSQP measurement harness.
====================================================================

TESTING.md flagged both as untested. That matters more for these than for the
main model:

  * Scratchpad read energy is ~80% of modelled solver call energy, so
    `sram_cost.py` now carries the single most important constant in the project.
  * `measure_osqp_baseline.py` produces the ONLY measured data we own. Its first
    two versions were both wrong (decoupled plant, then an infeasible scenario),
    and both were caught by review rather than automatically. The plant-coupling
    test below is the invariant that would have caught the first one on the spot.
"""

import numpy as np
import pytest

import measure_osqp_baseline as H
import sram_cost as S


# ---------------------------------------------------------------------------
# SRAM cost model
# ---------------------------------------------------------------------------

def test_area_grows_with_bank_count_at_fixed_capacity():
    """The costed structural finding: bandwidth is cheap, BANK COUNT is what costs.

    Splitting the same capacity into more, smaller macros must cost area, because
    per-macro peripheral overhead stops amortising. If this inverts, the
    recommendation to prefer few wide banks over many narrow ones loses its basis.
    """
    areas = [S.SpadCfg(128, nb, 8).area_mm2()["total_mm2"] for nb in (4, 8, 16, 32)]
    assert areas == sorted(areas), f"area not monotonic in bank count: {areas}"
    assert areas[-1] > 1.2 * areas[0], (
        "32 banks costs less than 20% more than 4 banks — the small-macro penalty "
        "has gone missing, and with it the reason to prefer wide banks")


def test_array_efficiency_falls_with_smaller_macros():
    effs = [S.SpadCfg(128, nb, 8).area_mm2()["array_eff"] for nb in (4, 8, 16, 32)]
    assert effs == sorted(effs, reverse=True), f"array efficiency not falling: {effs}"


def test_density_brackets_the_measured_anchor():
    """Calibration guard: the model is anchored to a measured 16nm 6T macro at
    8.1 Mb/mm², against a 14.3 Mb/mm² bitcell limit. Mid-size configurations must
    land near the anchor, and nothing may exceed the physical bitcell limit.
    """
    for nb in (4, 8, 16):
        d = S.SpadCfg(256, nb, 8).area_mm2()["mb_per_mm2"]
        assert 3.0 < d < 14.3, f"{nb} banks: {d:.2f} Mb/mm² is outside physical bounds"


def test_narrow_bank_words_are_flagged_as_extrapolated():
    """There is NO published energy data below a 64-bit access width.

    Below that floor the model keeps predicting ever-cheaper bytes for
    ever-narrower banks, which is unphysical — per-access fixed costs stop
    amortising, so pJ/byte must eventually turn back up. The model does not
    model that turn; it flags the region instead. So the contract is:
    `extrapolated` fires below the floor, and **no shipped recommendation may
    sit in flagged territory.** This test pins both halves.
    """
    assert S.MIN_SUPPORTED_BANK_WORD_BITS >= 64

    ok = S.SpadCfg(128, 16, 8).energy_pj_per_byte()          # 64-bit word
    assert ok["extrapolated"] is False and ok["total_pj_b"] > 0

    for banks, word in ((32, 4), (64, 2)):                   # 32-bit, 16-bit
        bad = S.SpadCfg(128, banks, word).energy_pj_per_byte()
        assert bad["extrapolated"] is True, (
            f"{word*8}-bit bank word is below the data floor but is not flagged "
            "— a caller cannot tell it is unsupported")
        # Documents WHY the flag must be honoured: the model is optimistic here.
        assert bad["total_pj_b"] < ok["total_pj_b"]

    # The shipped CM-100 point must not rely on extrapolated territory.
    rec = S.SpadCfg(128, 16, 8).energy_pj_per_byte()
    assert rec["extrapolated"] is False


def test_energy_includes_distribution_not_just_macro():
    """The 2.3x error was omitting the bank->datapath network. Total must exceed
    the macro term by a clear margin, or that network has been dropped again.
    """
    e = S.SpadCfg(128, 16, 8).energy_pj_per_byte()
    assert e["total_pj_b"] > e["macro_16_pj_b"]
    assert e["wire_pj_b"] > 0 and e["mux_pj_b"] > 0
    assert e["total_pj_b"] > 1.15 * e["macro_16_pj_b"]


# ---------------------------------------------------------------------------
# OSQP measurement harness
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def coupled_qp():
    A, B = H.quadrotor_model(coupled=True)
    return H.build_condensed_qp(A, B, N=20)


@pytest.fixture(scope="module")
def decoupled_qp():
    A, B = H.quadrotor_model(coupled=False)
    return H.build_condensed_qp(A, B, N=20)


def test_coupled_plant_gives_a_DENSE_condensed_hessian(coupled_qp, decoupled_qp):
    """THE invariant that would have caught the decoupled-plant error.

    A condensed Hessian for a controllable, coupled plant must be dense: every
    input eventually influences every state, so Gamma^T Qbar Gamma has no
    structural zeros. The first harness applied inputs one-per-state with diagonal
    inertia and no drag, which decoupled into four independent SISO channels and
    produced a 4-block-diagonal Hessian at 25% density. Iteration counts measured
    on that plant were optimistic, and it took an external audit to notice.
    """
    n = coupled_qp["n_var"]
    dens = coupled_qp["P"].nnz / n ** 2
    assert dens > 0.95, (
        f"coupled-plant Hessian is only {dens:.0%} dense — the plant has "
        "decoupled again and any iteration count measured on it is optimistic")
    # And confirm the flawed configuration is still reproducible and still sparse,
    # so the contrast in the results file stays meaningful.
    assert decoupled_qp["P"].nnz / n ** 2 < 0.5


def test_rotor_mixing_couples_each_input_to_multiple_states():
    """Each rotor thrust must drive several states, not one.

    For an X-configuration quadrotor the correct count is 3 per rotor: vertical
    acceleration, one of roll/pitch, and yaw. The flawed version drove exactly one
    state per input, which is what let the problem decouple.
    """
    _, B_c = H.quadrotor_model(coupled=True)
    _, B_d = H.quadrotor_model(coupled=False)
    touched_c = (np.abs(B_c) > 0).sum(axis=0)
    touched_d = (np.abs(B_d) > 0).sum(axis=0)
    assert (touched_c >= 3).all(), (
        f"inputs touch only {touched_c} states each — rotor mixing is not wired up")
    assert (touched_c > touched_d).all(), (
        "the coupled plant is no more coupled than the flawed one")


def test_condensed_qp_dimensions_match_the_model_assumption(coupled_qp):
    """Cross-check the harness against the model: 320 constraint rows for
    QUAD-MPC (80 input box + 240 predicted-state box), one row per two-sided
    bound. This is the measured fact that corrected a 2x overcount.
    """
    assert coupled_qp["n_var"] == 80
    assert coupled_qp["A_con"].shape[0] == 320


def test_iteration_statistics_require_an_all_solved_run(coupled_qp):
    """The infeasible-scenario error: a p95 of ~900 iterations was OSQP
    struggling with a physically unreachable setpoint, not doing control work.
    Any reported statistic must come from a run where every tick solved.
    """
    iters, times, statuses = H.closed_loop(coupled_qp, 1e-3, steps=40,
                                           extra=H.EXACT, scenario="aggressive")
    assert all(s == "solved" for s in statuses), (
        f"scenario is not feasible: {sorted(set(statuses))} — fix the scenario "
        "before reading any iteration number off it")
    assert len(iters) == 40 and iters.min() >= 1


def test_warm_start_beats_cold_solve(coupled_qp):
    """Warm-starting is load-bearing for the whole thesis: it is why a control
    loop needs ~15 iterations rather than ~70, and it is what call-to-call
    residency exists to exploit.
    """
    cold = H.solve_once(coupled_qp, H.X0, np.zeros(12), 1e-3, extra=H.EXACT)
    warm, _, _ = H.closed_loop(coupled_qp, 1e-3, steps=40, extra=H.EXACT,
                               scenario="settle")
    assert np.percentile(warm, 50) < 0.5 * int(cold.info.iter)
