# SPDX-License-Identifier: Apache-2.0
"""Regenerates the two tables quoted in `demo/policy_health.py`.

    python -m demo.objective_sweep.sweep            # both, ~8 min on 30 cores
    python -m demo.objective_sweep.sweep --track    # the rigidity-weight sweep only
    python -m demo.objective_sweep.sweep --pan      # the novelty-reward environment only

Every figure in `policy_health.py`'s docstring and in `MEASURED` in
`demo/test_policy_health.py` comes from here. The tables are printed rather than written to a
file: they are prose in a docstring, and a generated file that nothing reads is how a table goes
stale. What keeps them honest is that the run is reproducible, not that it is automated --
`test_policy_health.py` cannot verify that a recorded measurement is the one that was taken, and
no test can, which is exactly why the script is checked in.

Seeded throughout, and the seeds are stated in the output.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys

import numpy as np

from demo.objective_sweep import env
from demo.policy_health import action_stats

WEIGHTS = (0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 6.0)
TRACK_SEEDS = (1, 2, 3, 4)
PAN_SEEDS = (1, 2, 3)


def _track_job(job):
    """(w_rigid, seed, zero_reward) -> (reward, reversal, stillness, entropy). One process."""
    w_rigid, seed, zero = job
    bb, scale, n_s = env.track_backbone()
    qs = env.track_train(bb, scale, n_s, w_rigid, seed, zero=zero)
    total, streams = env.track_run(bb, scale, qs, w_rigid, seed=seed, collect=True)
    # One statistic over the three trackers' streams concatenated: they share a policy class and
    # the question is about the policy, not about tracker 2.
    stats = [action_stats(s, len(env.ACTIONS), opposite=env.OPPOSITE) for s in streams]
    return (total,
            float(np.mean([s.reversal for s in stats])),
            float(np.mean([s.stillness for s in stats])),
            float(np.mean([s.entropy for s in stats])))


def track(pool) -> None:
    jobs = [(w, s, z) for w in WEIGHTS for s in TRACK_SEEDS for z in (False, True)]
    res = dict(zip(jobs, pool.map(_track_job, jobs)))
    print(f"part tracking: {len(TRACK_SEEDS)} seeds {TRACK_SEEDS}, 10 training episodes, "
          f"400-step greedy evaluation, medians")
    print(f"  {'W_RIGID':>9}{'learned':>10}{'r=0':>9}{'score':>9}"
          f"{'reversal':>10}{'stillness':>11}{'entropy':>9}   verdict")
    for w in WEIGHTS:
        lrn = [res[(w, s, False)] for s in TRACK_SEEDS]
        tl = np.median([r[0] for r in lrn])
        tz = np.median([res[(w, s, True)][0] for s in TRACK_SEEDS])
        score = tl / tz if tz > 0 else float("nan")
        rev, still, ent = (float(np.median([r[i] for r in lrn])) for i in (1, 2, 3))
        verdict = "helps" if score > 1.05 else "HURTS" if score < 0.98 else ""
        print(f"  {w:>9.2f}{tl:>10.1f}{tz:>9.1f}{score:>8.2f}x"
              f"{rev:>10.1%}{still:>11.1%}{ent:>9.2f}   {verdict}")
    z = [res[(w, s, True)] for w in WEIGHTS for s in TRACK_SEEDS]
    print(f"\n  the r=0 control: stillness {np.median([r[2] for r in z]):.1%}, "
          f"entropy {np.median([r[3] for r in z]):.2f} bits. Its reward is identical at every")
    print("  weight, which is only consistent if it never changes a pairwise distance -- i.e.")
    print("  all three trackers move together, which is what a collapsed policy does.")


def _pan_job(job):
    """(policy, seed) -> (reward, reversal, stillness, entropy, histogram)."""
    from argmind.adapt import RecursiveLeastSquares
    name, seed = job
    bb, scale, n_s = env.pan_backbone()
    frames, pos = env.world(500, seed=seed)
    if name in ("random", "oracle"):
        total, stream = env.pan_run(bb, scale, frames, pos, name, seed=seed)
    else:
        q = RecursiveLeastSquares(len(env.PAN_ACTIONS) * n_s, 1, forgetting=0.999, p0=10.0,
                                  dtype="f8")
        for ep in range(12):
            env.pan_run(bb, scale, frames, pos, "greedy", q=q,
                        eps=max(0.05, 0.5 * (0.8 ** ep)), learn=True,
                        zero_reward=name.endswith("r=0"), seed=seed * 100 + ep)
        total, stream = env.pan_run(bb, scale, frames, pos, "greedy", q=q, eps=0.0, seed=seed)
    st = action_stats(stream, len(env.PAN_ACTIONS), opposite=env.OPPOSITE)
    hist = [stream.count(a) for a in range(len(env.PAN_ACTIONS))]
    return total, st.reversal, st.stillness, st.entropy, hist


def pan(pool) -> None:
    names = ("random", "oracle", "learned (the wiggler)", "learned, r=0")
    jobs = [(n, s) for n in names for s in PAN_SEEDS]
    res = dict(zip(jobs, pool.map(_pan_job, jobs)))
    print(f"\nnovelty reward: {len(PAN_SEEDS)} seeds {PAN_SEEDS}, 12 training episodes, "
          f"500-step greedy evaluation, medians")
    print(f"  {'policy':>22}{'reward':>9}{'reversal':>10}{'stillness':>11}{'entropy':>9}"
          f"   histogram (seed {PAN_SEEDS[0]})")
    got = {}
    for n in names:
        rows = [res[(n, s)] for s in PAN_SEEDS]
        got[n] = [float(np.median([r[i] for r in rows])) for i in range(4)]
        print(f"  {n:>22}{got[n][0]:>9.1f}{got[n][1]:>10.1%}{got[n][2]:>11.1%}"
              f"{got[n][3]:>9.2f}   {res[(n, PAN_SEEDS[0])][4]}")
    w, o = got["learned (the wiggler)"], got["oracle"]
    print(f"\n  the wiggler scores {w[0] / o[0]:.2f}x the follow-the-blob oracle ON THE "
          f"ORACLE'S OWN REWARD,")
    print("  while looking at nothing. That is the reward being hacked, not the task solved.")
    print(f"  entropy: wiggler {w[3]:.2f} bits against the oracle's {o[3]:.2f} -- so a LOWER "
          f"bound on")
    print("  entropy freezes the working policy before the broken one. The statistic does not")
    print("  carry the sign of the error; see demo/policy_health.py.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--track", action="store_true", help="the rigidity-weight sweep only")
    ap.add_argument("--pan", action="store_true", help="the novelty-reward environment only")
    ap.add_argument("--jobs", type=int, default=28, help="worker processes")
    args = ap.parse_args(argv)
    both = not (args.track or args.pan)
    with mp.Pool(max(1, args.jobs)) as pool:
        if both or args.track:
            track(pool)
        if both or args.pan:
            pan(pool)
    return 0


if __name__ == "__main__":
    sys.exit(main())
