# SPDX-License-Identifier: Apache-2.0
"""The cycle: look, remember, act — and ask only when remembering failed.

    observe ──▶ describe ──▶ cache.lookup ──┬── hit  ──▶ replay the stored actions
                                            └── miss ──▶ brain.decide ──▶ act ──▶ cache.store

**Why the cache is consulted before the brain and not beside it.** The owner's requirement is that
repeated work gets faster, and the only way that happens is if the expensive step is *skipped*, not
raced. A hit costs one matrix product against the memory; a miss costs a model round trip.

**Every step is recorded, because "it got faster" is a claim.** `history` carries, per step, whether
it hit, what it did, and how long the deciding took. `report()` turns that into the two numbers that
either support the claim or refute it: the hit rate, and the wall-clock the cache skipped.

**A hit is not trusted forever.** `verify` is an optional callable the caller supplies; when it says
the world did not end up where the entry promised, the entry is invalidated and the next encounter
is a miss that asks the brain again. Without that, a cache is a bet placed once and never settled.
"""

from __future__ import annotations

import time

__all__ = ["AgentLoop", "Step"]


class Step:
    """One cycle, kept so the claim can be audited rather than believed."""

    __slots__ = ("n", "hit", "slot", "score", "actions", "decide_s", "total_s", "why")

    def __init__(self, n, hit, slot, score, actions, decide_s, total_s, why) -> None:
        self.n = int(n)
        self.hit = bool(hit)
        self.slot = int(slot)
        self.score = float(score)
        self.actions = list(actions)
        self.decide_s = float(decide_s)
        self.total_s = float(total_s)
        self.why = str(why)

    def __repr__(self) -> str:
        mark = "HIT " if self.hit else "miss"
        return (f"Step({self.n} {mark} slot={self.slot} score={self.score:.4f} "
                f"{len(self.actions)} actions, decide {self.decide_s * 1000:.0f} ms)")


class AgentLoop:
    """Ties the senses, the situation descriptor, the cache, the brain and the actuator together.

    Nothing here knows what a screen is or what an action means. That is deliberate: the pieces were
    measured separately and this file is the only place they meet, so a change to any one of them
    does not have to be reasoned about here.
    """

    def __init__(self, *, screen, situation, cache, brain, actuator, verify=None,
                 webcam=None, pointer=None) -> None:
        self.screen = screen
        #: **Extra eyes, and deliberately NOT part of the cache key.** Measured over 119 real
        #: observations: keying on screen alone recurs at 21.0 %, screen+webcam at 17.6 %, and the
        #: webcam ALONE at 61.3 % — with nine distinct entries for the whole run. That last number
        #: looks like the best key and is the worst: a key that matches almost everything cannot
        #: tell situations apart, which is the global-pooling failure already recorded in
        #: `docs/findings/vocabulary-and-labelling.md` arriving in a new place. So the webcam is
        #: context for the brain and never a key. (The reason is DISCRIMINATION; a first version of
        #: this comment said the webcam moves too much to be stable, and the measurement says the
        #: opposite — it moves less than the screen.)
        self.webcam = webcam
        self.pointer = pointer
        self.situation = situation
        self.cache = cache
        self.brain = brain
        self.actuator = actuator
        #: `verify(step, frame_after) -> bool`. False invalidates the entry that produced the step.
        self.verify = verify
        self.history: list[Step] = []

    def step(self, goal: str, *, context: str = "") -> Step:
        t0 = time.perf_counter()
        frame = self.screen.read()
        situation = self.situation.describe(frame, t=t0)

        hit = self.cache.lookup(situation.key)
        frames = {"screen": frame}
        if hit is None:
            #: Read only on a miss. The extra eyes cost a grab each and nothing downstream of a hit
            #: looks at them, so paying for them on the fast path would spend the cache's saving.
            if self.webcam is not None:
                frames["webcam"] = self.webcam.read()
            if self.pointer is not None:
                state = self.pointer.read()
                context = (f"{context}\ncursor at {state['cursor']}, "
                           f"buttons: {'left ' if state['left'] else ''}"
                           f"{'right ' if state['right'] else ''}"
                           f"{'none' if not (state['left'] or state['right']) else ''}").strip()
        if hit is not None:
            actions, why, decide_s, slot, score = hit.value, "cached", 0.0, hit.slot, hit.score
            self.cache.note_hit_saved(hit.slot)
        else:
            decision = self.brain.decide(goal, frames, context)
            #: **Validated before it is stored.** A malformed action executed once is a bad turn; a
            #: malformed action CACHED is a bad turn that repeats faster every time it hits. The
            #: brain is a model and models return odd things, so the seam is checked here.
            for action in decision.actions:
                self.actuator.validate(action)
            actions, why, decide_s = decision.actions, decision.why, decision.seconds
            #: Stored **before** acting, so an action that changes the screen cannot make the key
            #: it was decided from unrecoverable. The key describes the situation the decision was
            #: FOR, which is the only thing a later look-up can match against.
            slot = self.cache.store(situation.key, actions, cost_s=decision.seconds)
            score = 1.0

        self.actuator.run(actions)
        step = Step(len(self.history), hit is not None, slot, score, actions, decide_s,
                    time.perf_counter() - t0, why)
        self.history.append(step)

        if self.verify is not None and not self.verify(step, self.screen.read()):
            #: The entry promised something the world did not do. Drop the value and leave the key,
            #: so the same situation asks the brain again instead of repeating a mistake fast.
            self.cache.invalidate(step.slot)
        return step

    def run(self, goal: str, steps: int, *, context: str = "", settle: float = 0.0) -> list[Step]:
        out = []
        for _ in range(int(steps)):
            out.append(self.step(goal, context=context))
            if settle:
                time.sleep(settle)
        return out

    # ---- the evidence -------------------------------------------------------------------------

    def report(self) -> dict:
        """The two numbers the cache has to earn, plus what it cost to earn them."""
        hits = [s for s in self.history if s.hit]
        misses = [s for s in self.history if not s.hit]
        spent = sum(s.decide_s for s in misses)
        return {
            "steps": len(self.history),
            "hits": len(hits),
            "misses": len(misses),
            "hit_rate": (len(hits) / len(self.history)) if self.history else 0.0,
            "brain_seconds_spent": round(spent, 3),
            "brain_seconds_skipped": self.cache.stats()["brain_seconds_skipped"],
            "mean_step_ms_hit": round(1000 * _mean(s.total_s for s in hits), 1),
            "mean_step_ms_miss": round(1000 * _mean(s.total_s for s in misses), 1),
        }


def _mean(xs) -> float:
    xs = list(xs)
    return (sum(xs) / len(xs)) if xs else 0.0
