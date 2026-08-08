# SPDX-License-Identifier: Apache-2.0
"""
Invariants for the descriptor format and its emit cost.

These pin an artifact rather than a model: the RV32 sequences in
`seq_emit_cost.py` are the derivation behind `seq_setup_cycles`, which sets the
crossover at which the sequencer rather than the datapath becomes binding. If
someone edits those sequences, the conclusions move, and these tests say so.
"""

import pytest

import seq_emit_cost as S


def test_descriptor_is_a_power_of_two_size():
    """Queue indexing must be a shift, not a multiply."""
    assert S.DESC_BYTES == 32
    assert S.DESC_BYTES & (S.DESC_BYTES - 1) == 0
    assert S.DESC_WORDS * 4 == S.DESC_BYTES


def test_software_emit_cannot_reach_the_assumed_setup_cost():
    """The load-bearing conclusion: 4 cycles/descriptor presumes hardware.

    If this ever passes trivially — i.e. software emit gets under 4 — then the
    hardware descriptor generator stops being a precondition and the sequencer
    requirement in PHASE0_FINDINGS §0.2 should be revisited.
    """
    c = S.emit_costs(1)
    assert c["naive"] > S.ASSUMED_SETUP_CYCLES
    assert c["delta"] > S.ASSUMED_SETUP_CYCLES, (
        "a delta-encoded software emit now beats the assumed setup cost — the "
        "hardware generator may no longer be required, which is a design change")


def test_naive_emit_corroborates_the_independently_assumed_20_cycles():
    """The sequencer analysis charged write-and-poll at ~20 cycles by intuition.

    This count derives it from instructions. Agreement within a few cycles is
    the point; large divergence means one of the two is wrong.
    """
    naive = S.emit_costs(1)["naive"]
    assert 15 <= naive <= 30, f"naive emit {naive} no longer corroborates ~20"


def test_generator_benefit_depends_on_descriptors_per_template():
    """The flat assumption hid this, and it is what makes ADMM exposed."""
    sweep = S.emit_costs(540)["hwgen"]          # one shape across 30 stages
    admm = S.emit_costs(16 / 6)["hwgen"]        # ~6 shapes, 16 descriptors
    assert sweep < 0.5, "template amortisation over a long sweep has broken"
    assert admm > S.ASSUMED_SETUP_CYCLES, (
        "ADMM as decomposed no longer exceeds the assumed setup cost — if the "
        "decomposition changed, the once-per-solve arming requirement may be moot")
    assert admm > 10 * sweep


def test_breakeven_is_about_six_descriptors_per_template():
    be = S.breakeven_descriptors_per_template(S.ASSUMED_SETUP_CYCLES)
    assert 4.0 <= be <= 8.0, f"break-even moved to {be:.1f} descriptors/template"


@pytest.mark.parametrize("target,expect_more", [(1.0, True), (10.0, False)])
def test_breakeven_scales_inversely_with_target(target, expect_more):
    be = S.breakeven_descriptors_per_template(target)
    base = S.breakeven_descriptors_per_template(S.ASSUMED_SETUP_CYCLES)
    assert (be > base) is expect_more


def test_instruction_counter_ignores_comments_and_labels():
    asm = """
        # a comment
        .Lloop:
        add  a0, a0, a1     # trailing comment

        sw   a0, 0(a2)
    """
    assert S.count_instructions(asm) == 2
