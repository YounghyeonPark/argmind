# SPDX-License-Identifier: Apache-2.0
"""Caller-side statistics for a policy's own behaviour, and the measured limit on using them.

`Guard.set_external` lets the caller hand the guard a scalar it cannot compute -- and for a head
that chooses actions, the obvious candidates are statistics of the action stream. This module
provides them, and pins what a sweep measured about whether they are safe to bound.

What was measured
-----------------
Two reinforcement objectives were driven to failure in this repository's exploration, in
opposite directions. Both learned a policy that earned a large reward while doing nothing
useful:

  wiggle   reward = "how much did my view change", on a single agent panning across a wide
           frame. The learner beat a follow-the-blob oracle by **1.51x on the oracle's own
           reward** while looking at nothing, 25.7% of its actions exactly undoing the previous.
  freeze   an abstract part-tracking objective (salient / distinct / rigid, no part named) with
           the rigidity term at weight 3.0. The learner scored **0.69x** a head trained with
           its reward set to zero.

Measured over the rigidity weight on trajectories, four seeds, 400-step greedy evaluation --
which is the axis that decides the second failure. Regenerate with
`python -m demo.objective_sweep.sweep`; `score` is against a head trained with its reward set
to zero, which collapses to a constant policy and is therefore also the collapse control:

      W_RIGID  score  reversal  stillness  entropy
         0.00  1.18x     37.5%      18.2%     2.27   learning helps
         0.25  1.05x     32.1%      29.8%     2.16   learning helps
         0.50  1.10x     29.7%      28.9%     2.07   learning helps
         1.00  0.98x     25.8%      21.6%     2.03
         2.00  1.01x     30.6%      29.8%     2.19
         3.00  0.69x     19.2%      47.2%     1.91   learning HURTS
         6.00  0.68x     22.1%      26.3%     2.12   learning HURTS

      novelty reward, a different environment:
          random  1328.9     16.0%      20.0%     2.32
          oracle  1571.8      0.0%      61.4%     1.73   the ceiling
         wiggler  2369.5     25.7%      18.0%     2.24   1.51x the oracle, looking at nothing
             r=0   515.3      0.0%     100.0%     0.00   collapsed

Two of those figures moved when the sweep was checked in, and the reason is a definition rather
than noise: the exploratory version counted `stillness` over *transitions* where `action_stats`
counts it over samples, and pooled three trackers into one histogram where entropy is now taken
per tracker and averaged. Pooling inflates entropy whenever the agents differ, so the earlier
column read high. The shipped definitions are the ones a caller with a single agent would
compute, and the table follows them.

**No statistic here detects a badly specified objective, and two of them point the wrong way.**
`reversal` separates within the sweep -- but *low* is the bad side, while in the wiggle failure
*high* was the entire fault (25.7% against the oracle's 0.0%). `stillness` inverts too: 47.2% is
the worst row above, and 61.4% is the *oracle* in the wiggle environment, which correctly stops
moving once it is on the blob. And `entropy` is worst of all as a one-sided bound: the wiggler
reads **2.24 bits against the oracle's 1.73**, so a low-entropy bound freezes the working policy
before it freezes the broken one.

The general statement, which is the finding: **a policy's action distribution does not carry the
sign of its own error.** Whether moving a lot is good depends on the task, and that is exactly
the knowledge a specification-error detector was supposed to supply.

What does work, and it is much narrower
---------------------------------------
Collapse to a constant is detected with an enormous margin and needs no tuning: a head trained
with zero reward reads **0.00 bits and 100% stillness in both environments**, against >= 1.73
bits for every policy that does anything. `collapsed()` is that test, and it is the failure that
matters in a deployment -- a head that has silently stopped producing decisions -- rather than
one that is pointed at the wrong thing.

So bound `entropy` from below at zero-ish through `Guard.set_external`, and do not bound it from
above. `demo/test_policy_health.py` asserts both halves, including that no single threshold on
any of the three statistics classifies the measured cases correctly.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

__all__ = ["ActionStats", "action_stats", "collapsed", "external_signal"]


class ActionStats:
    """Three scalars describing an action stream. Plain values, no numpy, so a caller on the
    hosted tier can compute the same thing in a few lines of C++."""

    __slots__ = ("reversal", "stillness", "entropy", "n")

    def __init__(self, reversal: float, stillness: float, entropy: float, n: int) -> None:
        self.reversal = reversal
        self.stillness = stillness
        self.entropy = entropy
        self.n = n

    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        return (f"ActionStats(reversal={self.reversal:.3f}, stillness={self.stillness:.3f}, "
                f"entropy={self.entropy:.3f}, n={self.n})")


def action_stats(actions: Sequence[int], n_actions: int, *,
                 opposite: Sequence[int] | None = None,
                 hold: int = 0) -> ActionStats:
    """Summarise a sequence of action indices.

    `hold` is the index of the do-nothing action, counted by `stillness`. `opposite[a]` is the
    action that undoes `a`, or a negative value if none does; without it `reversal` is 0.0,
    because there is no general way to know which of a caller's actions cancel.

    An empty or single-element stream returns zeros with `n` set, rather than raising: a
    statistic over no samples is not an error, it is simply not yet available, and a caller
    feeding this into `set_external` needs a value on the first step.
    """
    if n_actions <= 0:
        raise ValueError(f"n_actions must be positive, got {n_actions}")
    counts = [0] * n_actions
    reversals = 0
    still = 0
    prev = -1
    for a in actions:
        if not 0 <= a < n_actions:
            raise ValueError(f"action {a} is outside [0, {n_actions})")
        counts[a] += 1
        if a == hold:
            still += 1
        if prev >= 0 and opposite is not None:
            inv = opposite[prev]
            if inv >= 0 and inv == a:
                reversals += 1
        prev = a
    total = len(actions)
    if total == 0:
        return ActionStats(0.0, 0.0, 0.0, 0)
    ent = 0.0
    for c in counts:
        if c:
            p = c / total
            ent -= p * math.log2(p)
    # `reversal` is over transitions, of which there are one fewer than samples.
    trans = total - 1
    return ActionStats(reversals / trans if trans else 0.0, still / total, ent, total)


def collapsed(stats: ActionStats, *, min_samples: int = 32, min_entropy: float = 0.5) -> bool:
    """Has the policy stopped deciding?

    True when the action distribution carries less than `min_entropy` bits over at least
    `min_samples` actions. The default is not tuned to a boundary: the measured gap is 0.00 bits
    for a collapsed head against 1.73 or more for every functioning policy across two
    environments, so anything in (0, 1.7) is the same test. 0.5 is the middle of the lower half.

    `min_samples` exists because entropy over a handful of actions is small for an ordinary
    reason -- a policy has not had the chance to use its whole repertoire yet -- and freezing on
    that would trip at every start-up.
    """
    return stats.n >= min_samples and stats.entropy < min_entropy


def external_signal(stats: ActionStats, *, min_samples: int = 32,
                    healthy: float = 8.0) -> float:
    """The value to hand `Guard.set_external`, which is **not** the raw entropy.

    A `Bound` has no notion of "not enough evidence yet": it is evaluated on every update from
    the first. Feeding raw entropy into `Bound("external_0", lo=0.5)` freezes the guard at
    **step 0**, because a one-action window carries zero bits -- measured, and pinned by
    `test_the_raw_statistic_freezes_the_guard_at_step_zero`. That is the third time in this
    repository that a reasonable-looking bound has fired on the first update, after
    `at_least(PTrace, 1e-9)` and `at_most(PTrace, 1e4)` being dead.

    So the "not established yet" case is resolved *here*, on the caller's side of the slot,
    where there is enough context to know it is not a fault. Before `min_samples` actions the
    slot reads `healthy`, which is above any achievable entropy for a small action set and
    therefore passes any lower bound a caller would write.
    """
    if stats.n < min_samples:
        return healthy
    return stats.entropy
