# SPDX-License-Identifier: Apache-2.0
"""The agent app: does the cache actually make repeated work cheaper, and can it be wrong?

**Everything here runs with no API key, no network, no camera and no display.** That is not a
convenience — it is the only way the claim *"반복적인 작업은 점점 빠르게"* can be checked at all, since
checking it against a real model would measure the model's latency on the day it ran.

The claim decomposes into four things a test can fail on:

* a repeated situation must **not reach the brain** — the deterministic half, counted rather than timed
* a *different* situation must reach it, or the cache is matching everything and the first half is
  vacuous
* the entry must be **droppable**, because a cache that cannot be wrong is a bet
* nothing may touch the real mouse unless somebody asked for that in so many words

The one timing assertion is written with a wide margin and is secondary; the counted assertions are
what hold the result. A wall-clock figure is not something this suite depends on.
"""

from __future__ import annotations

import numpy as np
import pytest

from agent import (ActionCache, ActionRefused, Actuator, AgentLoop, MicrophoneSense, ScriptedBrain,
                   SituationSense)


class FakeScreen:
    """Returns frames from a list, holding the last one once exhausted.

    Holding rather than raising because the loop reads the screen twice on a verified step, and a
    fixture that ran out between those two reads would fail for a reason that is not the finding.
    """

    def __init__(self, frames) -> None:
        self.frames = list(frames)
        self.reads = 0

    def read(self):
        frame = self.frames[min(self.reads, len(self.frames) - 1)]
        self.reads += 1
        return frame


def screen_of(seed: int, h: int = 72, w: int = 128):
    """A synthetic screen: structured blocks rather than noise, because a UI is mostly flat regions
    and noise would make every descriptor equidistant from every other."""
    g = np.random.default_rng(seed)
    f = np.zeros((h, w, 3), dtype=np.float64) + 0.15
    for _ in range(6):
        y, x = int(g.integers(0, h - 12)), int(g.integers(0, w - 20))
        f[y:y + 12, x:x + 20] = g.uniform(0.2, 0.95, 3)
    return f


def build(brain, frames, *, verify=None, threshold=0.995):
    sense = SituationSense()
    return AgentLoop(screen=FakeScreen(frames), situation=sense,
                     cache=ActionCache(sense.n_features, capacity=32, threshold=threshold, k=16),
                     brain=brain, actuator=Actuator(live=False), verify=verify)


CLICK = [{"do": "click", "button": "left", "x": 10, "y": 20}]


def test_a_REPEATED_situation_never_reaches_the_brain():
    """The requirement, counted rather than timed. Ten encounters with one screen must cost exactly
    one decision — everything after the first is served from the memory."""
    brain = ScriptedBrain(default=CLICK)
    same = screen_of(1)
    loop = build(brain, [same] * 10)
    loop.run("do the thing", 10)

    assert len(brain.asked) == 1, (
        f"the brain was asked {len(brain.asked)} times for one repeated situation; the cache is "
        "not skipping the expensive step, which is the only thing it is for")
    report = loop.report()
    assert report["hits"] == 9 and report["misses"] == 1, report
    assert report["brain_seconds_skipped"] >= 0.0
    assert all(s.actions == CLICK for s in loop.history), "a hit replayed something else"


def test_a_DIFFERENT_situation_still_reaches_it_or_the_cache_matches_everything():
    """The control. Without it, a cache that returned the first entry for every key would pass the
    test above perfectly."""
    brain = ScriptedBrain(default=CLICK)
    loop = build(brain, [screen_of(i) for i in range(1, 7)])
    loop.run("do the thing", 6)

    assert len(brain.asked) == 6, (
        f"six unrelated screens produced {len(brain.asked)} decisions; the descriptor is matching "
        "situations that are not the same one, and a false hit clicks in the wrong place")
    assert loop.report()["hit_rate"] == 0.0


def test_the_cache_can_be_WRONG_and_the_entry_is_dropped():
    """A cache that cannot be invalidated is a bet placed once and never settled. `verify` says the
    world did not end up where the entry promised; the next encounter must ask again."""
    brain = ScriptedBrain(default=CLICK)
    same = screen_of(2)
    loop = build(brain, [same] * 4, verify=lambda step, after: not step.hit)

    loop.run("do the thing", 4)
    assert len(brain.asked) == 3, (
        f"the brain was asked {len(brain.asked)} times; with every hit rejected the pattern must be "
        "miss, hit-then-dropped, miss, hit-then-dropped — an entry that survives rejection would "
        "repeat a mistake faster and faster")


def test_a_hit_is_cheaper_than_a_miss_IN_THE_PART_THE_CACHE_TOUCHES():
    """The timing claim, and the first version of it was measuring the wrong thing.

    Comparing whole *steps* compared the actuator's rate limiter, which is charged to hits and misses
    alike — and on a miss the brain's own latency absorbs part of the wait, so a hit came out
    **slower**: 100.4 ms against 53.8 ms. That number was real and said nothing about the cache.

    What the cache affects is the deciding, so that is what is compared. Written with a wide margin
    and stated as secondary — the counted assertions above are what hold the result, and a
    wall-clock figure is not something this suite depends on.
    """
    brain = ScriptedBrain(default=CLICK, latency=0.05)
    same = screen_of(3)
    loop = build(brain, [same] * 6)
    loop.run("do the thing", 6)

    hits = [s for s in loop.history if s.hit]
    misses = [s for s in loop.history if not s.hit]
    assert hits and misses, "the fixture produced only one kind of step"
    assert max(s.decide_s for s in hits) < min(s.decide_s for s in misses), (
        f"deciding on a hit cost {max(s.decide_s for s in hits):.4f} s against "
        f"{min(s.decide_s for s in misses):.4f} s on a miss; the cache is not skipping the model")
    assert loop.report()["brain_seconds_skipped"] > 0.0, (
        "no brain time was credited as skipped, so the cache cannot claim to have saved any")


def test_the_actuator_is_DRY_RUN_until_somebody_says_otherwise():
    """This types into whatever window has focus and there is no undo. The default has to be the
    safe one, and the record has to exist either way so a dry run is reviewable."""
    a = Actuator()
    assert a.live is False, "actuation defaulted to live; that moves the real mouse"
    a.do({"do": "click", "x": 1, "y": 2})
    assert a.performed == [{"do": "click", "x": 1, "y": 2, "live": False}]


@pytest.mark.parametrize("action,fragment", [
    ({"do": "fly"}, "unknown action"),
    ({"do": "key", "name": "f4"}, "unknown key"),
    ({"nope": 1}, "'do' key"),
])
def test_a_malformed_or_unknown_action_is_REFUSED_rather_than_guessed(action, fragment):
    """Guessing a keystroke is how an agent sends Alt+F4 by accident. `KEYS` is short on purpose and
    a name outside it is an error, not a lookup miss."""
    with pytest.raises(ActionRefused, match=fragment):
        Actuator().do(action)


def test_a_brake_stops_the_current_burst_and_not_the_next_one():
    """Checked per action rather than per batch: "the human touched the mouse" has to stop what is
    running, not what runs after it."""
    seen = []
    a = Actuator(guard=lambda action: seen.append(action) or len(seen) < 3, max_per_second=1000.0)
    with pytest.raises(ActionRefused, match="brake"):
        a.run([{"do": "move", "x": i, "y": i} for i in range(5)])
    assert len(a.performed) == 2, f"{len(a.performed)} ran before the brake; it is checked per action"


def test_AUDIO_is_absent_and_says_so_rather_than_returning_silence():
    """This project's own distinction. A microphone that returned zeros would look like a working
    sense with a quiet room."""
    with pytest.raises(NotImplementedError, match="no audio capture library"):
        MicrophoneSense()


def test_the_cache_SURVIVES_a_restart(tmp_path):
    """"Progressively faster" across a session is a smaller claim than across restarts, and the
    second one is what somebody using this every day would mean."""
    sense = SituationSense()
    frame = screen_of(4)
    key = sense.describe(frame).key

    first = ActionCache(sense.n_features, capacity=32, threshold=0.995, k=16)
    first.store(key, CLICK, cost_s=1.5)
    first.save(tmp_path / "cache")

    second = ActionCache(sense.n_features, capacity=32, threshold=0.995, k=16)
    assert second.lookup(key) is None
    second.load(tmp_path / "cache")
    hit = second.lookup(key)
    assert hit is not None and hit.value == CLICK, "the cache did not survive being written and read"


def test_a_cache_saved_under_a_DIFFERENT_descriptor_is_refused(tmp_path):
    """A key means nothing across a descriptor it was not computed with, so loading one is refused
    rather than reshaped — the same refusal the persistence record makes elsewhere in this repo."""
    import json

    cache = ActionCache(8, capacity=4, threshold=0.99, k=4)
    np.save(tmp_path / "c.keys.npy", np.asarray(cache.memory.state))
    (tmp_path / "c.values.json").write_text(json.dumps({
        "threshold": 0.99, "capacity": 999, "n_features": 8,
        "uses": [], "cost_of": {}, "values": [],
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="different descriptor"):
        cache.load(tmp_path / "c")


def test_a_MALFORMED_decision_is_refused_before_it_becomes_a_cache_entry():
    """A bad action run once is a bad turn. A bad action *stored* is a bad turn that repeats faster
    every time it hits, which is the one failure mode a cache adds that nothing else has."""
    brain = ScriptedBrain(default=[{"do": "key", "name": "f4"}])
    loop = build(brain, [screen_of(5)] * 2)
    with pytest.raises(ActionRefused, match="unknown key"):
        loop.step("do the thing")
    assert loop.cache.stats()["entries"] == 0, (
        "the malformed decision was cached anyway; the next matching screen would replay it")

# --------------------------------------------------------------------------------------------
# The extra eyes: what belongs in the KEY and what belongs in the CONTEXT -- 2026-09-08
#
# The spec lists screen, webcam, keyboard and mouse as inputs, and the default reading is that the
# situation is all of them together. Measured over 119 real observations at 2 Hz instead:
#
#     key             entries   any hit   RECURRENT
#     screen               23     80.7%       21.0%
#     screen+webcam        44     63.0%       17.6%
#     webcam                9     92.4%       61.3%
#
# **The webcam alone looks like the best key and is the worst one.** Nine entries for a 75-second
# run means it can barely tell one moment from another -- high invariance, no discrimination, which
# is the global-pooling failure already recorded in `docs/findings/vocabulary-and-labelling.md`
# turning up somewhere new. A key that matches almost everything replays the same action whatever is
# on screen.
#
# **My stated reason was wrong and is withdrawn.** I predicted the webcam would be too UNSTABLE to
# key on -- a face never holds still. It is *more* stable than the screen here: the camera sees a
# mostly static room while the terminal scrolls. The right criterion was never stability; it is
# whether the key separates situations that need different actions.


class FakeWebcam:
    """Hands out a different frame every read, so "the webcam changed" is the default."""

    def __init__(self) -> None:
        self.reads = 0

    def read(self):
        self.reads += 1
        return screen_of(1000 + self.reads)


class FakePointer:
    def __init__(self) -> None:
        self.reads = 0

    def read(self):
        self.reads += 1
        return {"cursor": (7, 9), "left": True, "right": False, "middle": False,
                "shift": False, "ctrl": False, "alt": False, "t": 0.0}


def test_the_WEBCAM_reaches_the_brain_but_never_the_cache_key():
    """The measured design decision, held. Ten encounters with one screen and a webcam that changes
    every single frame must still cost exactly one decision — if the webcam were in the key, every
    frame would be new and the cache would never hit at all."""
    brain = ScriptedBrain(default=CLICK)
    cam = FakeWebcam()
    loop = build(brain, [screen_of(7)] * 10)
    loop.webcam = cam

    loop.run("do the thing", 10)
    assert len(brain.asked) == 1, (
        f"a changing webcam produced {len(brain.asked)} decisions for one repeated screen; it has "
        "got into the cache key, and the measurement says that costs hits and buys nothing")
    assert brain.saw == ["screen", "webcam"], (
        f"the brain was given {brain.saw}; the webcam is context and has to actually arrive, or the "
        "agent cannot see the person it is supposed to be interacting with")


def test_the_extra_eyes_are_read_only_on_a_MISS():
    """A hit is supposed to be cheap. Grabbing a camera frame on the fast path spends part of the
    saving the cache exists to produce."""
    brain = ScriptedBrain(default=CLICK)
    cam, ptr = FakeWebcam(), FakePointer()
    loop = build(brain, [screen_of(8)] * 6)
    loop.webcam, loop.pointer = cam, ptr

    loop.run("do the thing", 6)
    assert cam.reads == 1 and ptr.reads == 1, (
        f"the webcam was read {cam.reads} times and the pointer {ptr.reads} for one miss and five "
        "hits; they are being read on the fast path")


def test_the_POINTER_state_reaches_the_brain_as_context():
    """Where the cursor is and what is held is what turns "click there" into something a person is
    part of. Passed as text rather than as an image because that is what it is."""
    seen = {}

    class Recording(ScriptedBrain):
        def decide(self, goal, frames, context=""):
            seen["context"] = context
            return super().decide(goal, frames, context)

    brain = Recording(default=CLICK)
    loop = build(brain, [screen_of(9)])
    loop.pointer = FakePointer()
    loop.step("do the thing")

    assert "cursor at (7, 9)" in seen["context"], seen
    assert "left" in seen["context"], "a held button is part of what the person is doing"
