# SPDX-License-Identifier: Apache-2.0
"""What a caller may and may not bound through `Guard.set_external`, held to the measurement.

The interesting half of this file is negative. `demo/policy_health.py` records a sweep in which
two reinforcement objectives failed in opposite directions, and the conclusion was that no
one-sided bound on any action statistic separates the working policies from the broken ones.
That conclusion is only worth anything if a later change that quietly contradicted it would
fail here, so the measured cases are carried as data and the separation is *computed*.
"""

from __future__ import annotations

import numpy as np
import pytest

from argmind.adapt import RecursiveLeastSquares
from argmind.guard import Bound, Guard
from demo.policy_health import action_stats, collapsed, external_signal

#: stay, left, right, up, down -- and which action undoes which. `stay` undoes nothing.
OPPOSITE = (-1, 2, 1, 4, 3)
N_ACTIONS = 5

#: Measured, not constructed: `(name, reversal, stillness, entropy, score)` from the runs
#: recorded in `policy_health.py`'s docstring -- a rigidity-weight sweep over 400-step greedy
#: trajectories at four seeds (medians), and a separate novelty-reward environment at three
#: seeds.
#:
#: Regenerate with `python -m demo.objective_sweep.sweep`, which is checked in for that purpose:
#: no test can verify that a recorded measurement is the one that was taken, and a mutant that
#: edited a score here survived the whole suite. Reproducibility is the defence, not an
#: assertion.
#:
#: `score` is the ratio against that environment's own control, and whether a policy WORKS is
#: **derived** from it rather than written down beside it. The first version carried a boolean,
#: and a mutant that relabelled `w_rigid=6.00` as working passed the whole suite -- so the
#: negative result was not protected against an edit to its own data. Deriving the label means
#: an edit has to change a measured number, which is a visible act.
#:
#: The two environments' scores are not comparable to each other in magnitude; only the
#: side of 1.0 they fall on is, which is all the derivation uses.
#: Every row of the sweep is here, including the two that fall between the outcomes. Carrying
#: only the decisive rows would be choosing the data after seeing it.
MEASURED = (
    # part tracking, by rigidity weight: reward against a zero-reward control
    ("w_rigid=0.00", 0.375, 0.182, 2.27, 1.18),
    ("w_rigid=0.25", 0.321, 0.298, 2.16, 1.05),
    ("w_rigid=0.50", 0.297, 0.289, 2.07, 1.10),
    ("w_rigid=1.00", 0.258, 0.216, 2.03, 0.98),
    ("w_rigid=2.00", 0.306, 0.298, 2.19, 1.01),
    ("w_rigid=3.00", 0.192, 0.472, 1.91, 0.69),
    ("w_rigid=6.00", 0.221, 0.263, 2.12, 0.68),
    # novelty reward, a different environment and a different agent count: reward against the
    # follow-the-blob oracle, which is the ceiling there. The wiggler exceeding it is the fault.
    ("pan oracle", 0.000, 0.614, 1.73, 1.00),
    ("pan wiggler", 0.257, 0.180, 2.24, 1.51),
)

#: A policy works when it is at least as good as its control and no better than its ceiling.
#: Both failures are outside that band, in opposite directions: the frozen learner at 0.69 and
#: the wiggler at 1.51, which beat the oracle on the oracle's own reward while looking at
#: nothing.
WORKS_LO, WORKS_HI = 0.98, 1.30


def _works(row) -> bool:
    return WORKS_LO <= row[4] <= WORKS_HI


def _separates(index: int) -> bool:
    """Is there a threshold and a direction that classifies every measured case correctly?"""
    good = [row[index] for row in MEASURED if _works(row)]
    bad = [row[index] for row in MEASURED if not _works(row)]
    return max(bad) < min(good) or max(good) < min(bad)


def test_the_measured_set_can_answer_the_question_it_is_asked():
    """A separation test over cases that are all one class passes vacuously, and this project
    has published three vacuous passes. So the shape of the data is asserted before it is used:
    both classes populated, and both environments represented on each side."""
    good = [r for r in MEASURED if _works(r)]
    bad = [r for r in MEASURED if not _works(r)]
    assert len(good) >= 2 and len(bad) >= 2, (len(good), len(bad))
    env = (lambda r: "pan" if r[0].startswith("pan") else "track")
    assert {env(r) for r in good} == {"pan", "track"}, "one environment only: no inversion"
    assert {env(r) for r in bad} == {"pan", "track"}
    # And the failures are on OPPOSITE sides of the band, which is what makes one bound
    # impossible rather than merely badly chosen.
    assert min(r[4] for r in bad) < WORKS_LO < WORKS_HI < max(r[4] for r in bad)


def test_no_action_statistic_separates_working_policies_from_broken_ones():
    """The finding. Each statistic is refuted by a different pair of cases, and the reason is
    that whether moving a lot is good is a property of the task."""
    for index, name in ((1, "reversal"), (2, "stillness"), (3, "entropy")):
        assert not _separates(index), (
            f"{name} now separates the measured cases, which contradicts "
            f"demo/policy_health.py. Re-measure and rewrite that docstring rather than "
            f"relaxing this test -- a statistic that separates is a better detector, but the "
            f"claim in the module has to change with it."
        )


def test_the_two_statistics_invert_between_the_two_environments():
    """Not merely 'does not separate': each one has the opposite sign in the two environments,
    which is why one bound cannot serve both. This is the mechanism behind the test above, and
    it fails independently if a number is edited to make the first one pass."""
    track = [r for r in MEASURED if r[0].startswith("w_rigid")]
    good_rev = [r[1] for r in track if _works(r)]
    bad_rev = [r[1] for r in track if not _works(r)]
    # part tracking: the working policies reverse MORE than the broken ones.
    assert min(good_rev) > max(bad_rev)
    # panning: the working policy reverses LESS -- the oracle never reverses at all.
    oracle = next(r for r in MEASURED if r[0] == "pan oracle")
    wiggler = next(r for r in MEASURED if r[0] == "pan wiggler")
    assert oracle[1] < wiggler[1]

    good_still = [r[2] for r in track if _works(r)]
    bad_still = [r[2] for r in track if not _works(r)]
    # part tracking: the worst row is the stillest.
    assert max(bad_still) > max(good_still)
    # panning: the working policy is the stillest, by a wide margin.
    assert oracle[2] > wiggler[2]

    # And entropy is the one that would do actual harm as a lower bound: the broken policy
    # reads higher than the working one, so the bound freezes the oracle first.
    assert wiggler[3] > oracle[3]


def test_collapse_is_detected_with_the_margin_it_was_measured_with():
    """The narrow claim that survives. A constant policy reads exactly zero bits; every policy
    that does anything reads at least 1.73. The default threshold sits in the empty middle, so
    it is not tuned to a boundary."""
    constant = action_stats([0] * 500, N_ACTIONS, opposite=OPPOSITE)
    assert constant.entropy == 0.0
    assert constant.stillness == 1.0
    assert collapsed(constant)

    lowest_working = min(row[3] for row in MEASURED if _works(row))
    assert lowest_working >= 1.73
    for row in MEASURED:
        stats = action_stats([0] * 40, N_ACTIONS)          # shape only; entropy replaced below
        stats.entropy = row[3]
        assert not collapsed(stats), f"{row[0]} is not collapsed and must not read as collapsed"


def test_a_short_stream_is_not_collapsed_merely_for_being_short():
    """A policy that has used one action twice has zero entropy and is not broken. Freezing on
    that would trip at every start-up, so `min_samples` gates it."""
    assert action_stats([1, 1], N_ACTIONS).entropy == 0.0
    assert not collapsed(action_stats([1, 1], N_ACTIONS))
    assert collapsed(action_stats([1] * 32, N_ACTIONS))
    assert not collapsed(action_stats([1] * 31, N_ACTIONS))


def test_reversal_needs_the_callers_table_and_is_zero_without_it():
    """There is no general way to know which of a caller's actions cancel, so `reversal` is not
    guessed. Without `opposite` it reports 0.0 rather than something plausible."""
    seq = [1, 2, 1, 2, 1, 2]
    assert action_stats(seq, N_ACTIONS, opposite=OPPOSITE).reversal == 1.0
    assert action_stats(seq, N_ACTIONS).reversal == 0.0
    # `stay` undoes nothing, so alternating with it is not a reversal.
    assert action_stats([1, 0, 1, 0], N_ACTIONS, opposite=OPPOSITE).reversal == 0.0


def test_statistics_are_over_the_right_denominators():
    """`reversal` is over transitions and `stillness` over samples. Getting these the same is
    an off-by-one that only shows up on short streams, which is where a start-up sits."""
    s = action_stats([1, 2, 0], N_ACTIONS, opposite=OPPOSITE)
    assert s.n == 3
    assert s.reversal == pytest.approx(1 / 2)      # one reversal in two transitions
    assert s.stillness == pytest.approx(1 / 3)     # one stay in three samples
    assert s.entropy == pytest.approx(np.log2(3))
    assert action_stats([], N_ACTIONS).n == 0      # no samples is not an error


def test_an_out_of_range_action_raises_rather_than_being_counted():
    with pytest.raises(ValueError, match="outside"):
        action_stats([0, 5], N_ACTIONS)
    with pytest.raises(ValueError, match="outside"):
        action_stats([-1], N_ACTIONS)
    with pytest.raises(ValueError, match="positive"):
        action_stats([0], 0)


def test_the_sweep_that_produced_the_table_still_runs():
    """`demo/objective_sweep/` is the evidence behind `MEASURED`, and evidence nothing executes
    rots. The full sweep takes about five minutes, so this drives a few steps of each
    environment: enough that a rename or a shape change fails here rather than the next time
    somebody tries to regenerate the table.

    Not a parity check on the figures -- reproducing them costs what the sweep costs.
    """
    from demo.objective_sweep import env

    bb, scale, n_s = env.track_backbone()
    assert n_s == bb.n_outputs + 2 * (env.N_PART - 1) + 1
    qs = env.track_train(bb, scale, n_s, 0.5, seed=1, episodes=1)
    total, streams = env.track_run(bb, scale, qs, 0.5, n=6, seed=1, collect=True)
    assert np.isfinite(total)
    assert [len(s) for s in streams] == [6] * env.N_PART
    assert all(0 <= a < len(env.ACTIONS) for s in streams for a in s)
    # The action table and its inverse must stay consistent, or `reversal` counts nonsense.
    for a, inv in enumerate(env.OPPOSITE):
        if inv >= 0:
            assert env.ACTIONS[inv] == tuple(-c for c in env.ACTIONS[a])
        else:
            assert env.ACTIONS[a] == (0, 0)

    # `onehot` is what makes Q per-action rather than one shared map, and getting it wrong is
    # silent: the environment still runs, every action scores identically, and the sweep quietly
    # produces different numbers. A mutant that wrote every action into slot 0 survived the
    # smoke test above, so the property is asserted directly.
    s = np.array([1.0, 2.0, 3.0])
    blocks = [env.onehot(s, a, 3, n_actions=4) for a in range(4)]
    assert all(len(b) == 12 for b in blocks)
    for a, b in enumerate(blocks):
        assert np.array_equal(b[a * 3:(a + 1) * 3], s)
        assert np.count_nonzero(b) == 3          # nothing written outside its own slot

    pbb, pscale, pn = env.pan_backbone()
    frames, pos = env.world(4, seed=1)
    reward, stream = env.pan_run(pbb, pscale, frames, pos, "oracle", seed=1)
    assert np.isfinite(reward) and len(stream) == 4
    assert pn == pbb.n_outputs + 1


def test_the_guard_freezes_on_a_collapsed_policy_through_an_external_slot():
    """The whole point of the statistic: the head is healthy -- it fits whatever it is given --
    and the failure is only visible in what the policy does with the output. So it reaches the
    guard as an external, and the freeze happens at the step the collapse becomes visible.
    """
    rng = np.random.default_rng(3)
    adapter = RecursiveLeastSquares(4, 1, forgetting=0.99, p0=10.0, dtype="f8")
    guard = Guard(adapter, [Bound("external_0", lo=0.5)], rollback_depth=4,
                  checkpoint_every=5, recover_after=1_000_000, n_external=1)
    window: list[int] = []
    froze_at = None
    for k in range(400):
        # After step 200 the policy stops deciding. Nothing about the regression changes.
        window.append(0 if k >= 200 else int(rng.integers(N_ACTIONS)))
        window[:] = window[-64:]
        guard.set_external(0, external_signal(action_stats(window, N_ACTIONS)))
        x = rng.standard_normal(4)
        report = guard.update(x, [float(x @ np.array([1.0, -2.0, 0.5, 3.0]))])
        if report.frozen and froze_at is None:
            froze_at = k
    assert froze_at is not None, "a collapsed policy did not freeze the guard"
    # The window is 64 long, so the freeze lands once the collapsed actions dominate it. The
    # exact step is a property of the window length, so it is bounded rather than pinned.
    assert 200 < froze_at < 200 + 64, froze_at
    assert guard.frozen
    assert np.all(np.isfinite(np.asarray(guard.state)))


def test_the_raw_statistic_freezes_the_guard_at_step_zero():
    """Why `external_signal` exists, and it was found by the test above failing rather than by
    reading. A `Bound` is evaluated from the first update and has no "not enough evidence yet";
    a one-action window carries zero bits, so the raw entropy trips immediately.

    Third occurrence of this shape in the repository, after `at_least(PTrace, 1e-9)` tripping on
    the first update and `at_most(PTrace, 1e4)` being unable ever to fire.
    """
    raw = Guard(RecursiveLeastSquares(2, 1, dtype="f8"), [Bound("external_0", lo=0.5)],
                rollback_depth=2, n_external=1)
    raw.set_external(0, action_stats([3], N_ACTIONS).entropy)      # one action: 0.0 bits
    assert raw.update([1.0, 2.0], [3.0]).frozen, (
        "the naive version no longer freezes at step 0, so external_signal's reason for "
        "existing is not demonstrated")

    gated = Guard(RecursiveLeastSquares(2, 1, dtype="f8"), [Bound("external_0", lo=0.5)],
                  rollback_depth=2, n_external=1)
    gated.set_external(0, external_signal(action_stats([3], N_ACTIONS)))
    assert not gated.update([1.0, 2.0], [3.0]).frozen
    # And the gate must not hide a real collapse once there is evidence for one.
    assert external_signal(action_stats([0] * 64, N_ACTIONS)) == 0.0


def test_the_same_bound_freezes_a_working_policy_in_the_other_environment():
    """The negative result, made operational. `pan oracle` reads 1.73 bits and works. A caller
    who tightened the bound to catch the wiggler at 2.24 bits from below would have to set it
    above 2.24, and that freezes the oracle on its first window.

    This is the same shape as the guard's `at_most(PTrace, 1e4)` finding: one bound, two
    directions, and which direction is right is not knowable from inside the guard.
    """
    # The oracle's own measured histogram over its 500-step evaluation, not a designed one.
    oracle_hist = (307, 55, 57, 37, 44)
    oracle_like = [a for a, c in enumerate(oracle_hist) for _ in range(c)]
    stats = action_stats(oracle_like, N_ACTIONS, opposite=OPPOSITE)
    assert stats.entropy == pytest.approx(1.73, abs=0.01), stats.entropy

    guard = Guard(RecursiveLeastSquares(2, 1, dtype="f8"),
                  [Bound("external_0", lo=2.3)],      # set to catch a 2.24-bit wiggler
                  rollback_depth=2, n_external=1)
    guard.set_external(0, stats.entropy)
    report = guard.update([1.0, 2.0], [3.0])
    assert report.frozen, ("the bound that catches the wiggler must be shown to freeze the "
                           "oracle, or the negative result is not demonstrated")
