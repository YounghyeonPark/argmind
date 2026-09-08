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
                   SituationSense, differs, patch_signature, signatures)


class FakeScreen:
    """A surface showing one frame, not a queue handing them out.

    **A queue was the first version and it was the wrong model.** The loop reads the screen more than
    once in a step -- a second time on a miss, to tell whether the click target is animating -- and a
    queue turns that into "the screen changed between the two reads", which is a fixture artefact
    that broke two tests for a reason unrelated to what they were checking. A real screen is a
    surface you can look at twice; this is that.
    """

    def __init__(self, frame=None) -> None:
        self.current = frame
        self.reads = 0

    def show(self, frame):
        self.current = frame
        return self

    def read(self):
        self.reads += 1
        return self.current


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
    """Returns the loop; `loop.screen` is the surface, and `show_and_step` drives it."""
    sense = SituationSense()
    screen = FakeScreen(frames[0] if isinstance(frames, list) else frames)
    loop = AgentLoop(screen=screen, situation=sense,
                     cache=ActionCache(sense.n_features, capacity=32, threshold=threshold, k=16),
                     brain=brain, actuator=Actuator(live=False), verify=verify)
    loop._frames = list(frames) if isinstance(frames, list) else [frames]
    return loop


def drive(loop, goal="do the thing", frames=None):
    """One step per frame, the frame put on the surface first — the order a real loop sees."""
    for frame in (frames if frames is not None else loop._frames):
        loop.screen.show(frame)
        loop.step(goal)
    return loop.history


CLICK = [{"do": "click", "button": "left", "x": 10, "y": 20}]


def test_a_REPEATED_situation_never_reaches_the_brain():
    """The requirement, counted rather than timed. Ten encounters with one screen must cost exactly
    one decision — everything after the first is served from the memory."""
    brain = ScriptedBrain(default=CLICK)
    same = screen_of(1)
    loop = build(brain, [same] * 10)
    drive(loop)

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
    drive(loop)

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

    drive(loop)
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
    drive(loop)

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

    drive(loop)
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

    drive(loop)
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

# --------------------------------------------------------------------------------------------
# The screen key cannot see what a click lands on -- 2026-09-08
#
# Measured on a real 2560x1440 screen against the 0.995 threshold the cache uses:
#
#     a 100x30 region blanked  (a button's worth)   1.0000   the key does not move
#     two 200x100 regions swapped                   1.0000   the key does not move AT ALL
#     a 400x300 region blanked (a dialog)           0.9975   still a hit
#     a 800x600 region blanked                      0.9904   finally a miss
#     the whole screen shifted 10 px                0.9950   a miss
#
# **Two buttons can exchange places and the key is unchanged**, so a cached click lands on the wrong
# one. The recurrence numbers elsewhere -- 58.3 % on a real screen -- say nothing about this: they
# measure whether the screen looks the same GLOBALLY, and a click depends on what is locally there.
#
# The mechanism is arithmetic, not a surprise: the descriptor sees a 128x72 decimation of the screen,
# so one descriptor pixel is 20x20 screen pixels, and a 3x3 grid pool averages over that again. No
# threshold on that key recovers local detail -- the 800x600 row is what it takes to move the score,
# and by then a third of the screen is different.
#
# So a hit re-checks the full-resolution pixels around each click target before replaying, and these
# tests use screens large enough for "local" and "global" to be different things -- the small ones
# above are the size of the descriptor's own input, where the distinction does not exist.

BIG = (720, 1280)


def big_screen(seed: int):
    g = np.random.default_rng(seed)
    f = np.zeros((*BIG, 3), dtype=np.float64) + 0.12
    for _ in range(24):
        y, x = int(g.integers(0, BIG[0] - 90)), int(g.integers(0, BIG[1] - 160))
        f[y:y + 90, x:x + 160] = g.uniform(0.2, 0.9, 3)
    return f


def with_button(frame, x, y, shade):
    out = frame.copy()
    out[y - 15:y + 15, x - 50:x + 50] = shade
    return out


TARGET = (300, 400)
CLICK_THERE = [{"do": "click", "button": "left", "x": TARGET[0], "y": TARGET[1]}]


def test_a_LOCAL_change_at_the_click_target_drops_the_hit_even_though_the_key_matches():
    """The failure the local check exists for, and the test asserts the premise rather than assuming
    it: the two screens must still match globally, or this is measuring something else."""
    sense = SituationSense()
    before = with_button(big_screen(11), *TARGET, 0.85)
    after = with_button(before, *TARGET, 0.20)

    globally = float(sense.describe(before).key @ sense.describe(after).key)
    assert globally >= 0.995, (
        f"the two screens score {globally:.4f} globally, below the cache threshold — the premise of "
        "this test is that the KEY cannot tell them apart, and here it can")

    brain = ScriptedBrain(default=CLICK_THERE)
    loop = build(brain, [before, before, after, after])
    drive(loop, "press the button")

    assert loop.report()["stale_hits_dropped"] >= 1, (
        "the button changed under the cached click and the entry was replayed anyway; that is a "
        "click on the wrong thing")
    assert len(brain.asked) >= 2, (
        f"the brain was asked {len(brain.asked)} times; after the target changed it has to be asked "
        "again rather than the stale action being repeated")


def test_a_change_AWAY_from_the_target_does_not_drop_the_hit():
    """The control. A check that vetoed on any change at all would pass the test above and make the
    cache useless — the point is that it is *local*."""
    sense = SituationSense()
    before = with_button(big_screen(12), *TARGET, 0.85)
    elsewhere = before.copy()
    #: Small enough that the KEY still matches — measured, not guessed: a 100x200 block moved it to
    #: 0.9950, just under the threshold, which would have made this test pass for the wrong reason.
    elsewhere[600:640, 1100:1180] = 0.05

    globally = float(sense.describe(before).key @ sense.describe(elsewhere).key)
    assert globally >= 0.995, f"the fixture's unrelated change already moved the key ({globally:.4f})"

    brain = ScriptedBrain(default=CLICK_THERE)
    loop = build(brain, [before, elsewhere, elsewhere])
    drive(loop, "press the button")

    assert loop.report()["stale_hits_dropped"] == 0, (
        "a change nowhere near the click target dropped the entry; the check is behaving globally "
        "and the cache will never hold anything")
    assert len(brain.asked) == 1


def test_an_OFF_SCREEN_target_counts_as_unverifiable_and_not_as_verified():
    """A coordinate outside the frame is exactly the case where a stale action would do something
    unintended, so it must not pass the check by default."""
    frame = big_screen(13)
    assert patch_signature(frame, 10_000, 10_000) is None
    assert differs([patch_signature(frame, *TARGET)], [None]), (
        "an unverifiable target compared equal; None has to mean 'cannot check', not 'checked'")


def test_actions_with_no_coordinate_are_not_given_a_fake_one():
    """Typing has nowhere on screen to verify. Signing the whole screen for it would either veto
    every keystroke or verify nothing, so those get None and the caller is told."""
    frame = big_screen(14)
    sigs = signatures(frame, [{"do": "type", "text": "x"}, {"do": "click", "x": 5, "y": 5}])
    assert sigs[0] is None and isinstance(sigs[1], list)
    assert not differs([None, sigs[1]], [None, sigs[1]])


def test_a_DIMMED_button_is_seen_where_a_difference_hash_would_miss_it():
    """The instrument change, held. A dHash compares each cell with its neighbour and is invariant to
    uniform brightness by construction — so a button greying out, the commonest UI state change there
    is, was invisible to the first version of this check. Measured on a real screen: 0.1325 at the
    weakest, against a quiet baseline of exactly 0.0000."""
    frame = big_screen(15)
    lit = with_button(frame, *TARGET, 0.85)
    dim = lit.copy()
    dim[TARGET[1] - 15:TARGET[1] + 15, TARGET[0] - 50:TARGET[0] + 50] *= 0.85

    before = patch_signature(lit, *TARGET)
    after = patch_signature(dim, *TARGET)
    assert differs([before], [after]), (
        "a uniformly dimmed button read as unchanged; that is the dHash failure this check replaced")
    assert not differs([before], [patch_signature(lit, *TARGET)]), "the same frame differed from itself"
