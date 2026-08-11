# Objective sweep — the evidence behind `demo/policy_health.py`

    python -m demo.objective_sweep.sweep          # both tables, 4m49s on 28 workers (measured)

This directory is not library code and not a benchmark. It exists because
`demo/policy_health.py` states figures, and an audit in this repository has already caught a
set of quoted sweep numbers that came from a run leaving no trace.

## The question

The owner's scope decision asks for several objective functions declared **abstractly** — track
eyes, nose and mouth without knowing which is which, because a frozen backbone cannot know. The
abstract form measured here is *salient / distinct / rigid*, with no part named, and it works as
a specification: its optimum is one tracker per part, and 4,000 random placements did not beat
it.

What this sweep asks is the next question, and it is a different one. **A learner acts over
trajectories, and the objective was validated over static placements.** The rigidity term means
"the trackers keep their distances" when scoring a placement; over a trajectory the cheapest way
to satisfy it is *don't move*.

## What it measures

`sweep.py --track` sweeps the rigidity weight and scores the learner against a head trained
with its reward set to zero. That control matters twice over: it is the "did the reward drive
anything" check, and it also happens to be the collapsed policy — it picks `stay` on 100% of
steps, which is why its reward is identical at every weight (a policy that never changes a
pairwise distance pays no rigidity penalty at any weight).

    W_RIGID  0.00  0.25  0.50 | 1.00  2.00 | 3.00  6.00
    score   1.18x 1.05x 1.10x | 0.98x 1.01x | 0.69x 0.68x

So the safe region is roughly 0.0–0.5 and the weight this project had been using — 3.0, raised
deliberately to push a distractor away — is in the region where **learning is worse than not
learning**. That is a specification error, not a learning failure.

`sweep.py --pan` runs the other failure: a novelty reward on a single panning agent, which the
learner maximises by oscillating. It scores **1.51× the follow-the-blob oracle on the oracle's
own reward** while looking at nothing.

## The finding

Both failures are visible in the action stream, and **neither is detectable by a bound on it**,
because the two environments give every statistic the opposite sign. `demo/policy_health.py`
carries the table and the argument; `demo/test_policy_health.py` fails if any statistic starts
separating the cases, so the claim cannot go stale quietly.

What *is* detectable, with no threshold tuning, is collapse to a constant: 0.00 bits against
≥1.73 for every policy that does anything.

## Cost

Measured: 4m49s wall for both tables at `--jobs 28` on this machine (32 cores). Not a target
figure and not a benchmark — it is stated so that anyone deciding whether to regenerate the
table knows what it costs.
