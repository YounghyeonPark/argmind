# SPDX-License-Identifier: Apache-2.0
"""`loop.py` written down, byte for byte — the equivalence test a port has to pass.

**`loop.py` has no equivalence test and what stands in for one cannot survive a rewrite.**
`measure/sending.py`'s `loop_now()` hashes the file, so it says *this is the same source*, which is
by construction unanswerable once the source is the thing being replaced. A fingerprint over a
compiled artefact is no better: three builds of a byte-identical Rust source, same rustc and flags,
produced three different SHA-256s. So the only form the test can take is behavioural — feed a
scripted world through a real `Session` and write down every byte that comes out — and it has to be
authored **against the Python core**, because a golden file first blessed against a reimplementation
records the reimplementation.

## What is pinned here rather than somewhere else

The value of this file is the branches nothing else exercises, so it aims at those:

* **all four channels of `remember_doing()`.** Only `offered` and `turned_down` are fed anywhere in
  `measure/`, and nothing feeds `could not`; the sentences are read by the next model call and by
  `converse.still_open()`, and the tense in them is the only thing separating an offer from a fact.
  Every channel is fed every kind of effect here, including a kind `VERBS` does not list.
* **`recent()` at the boundaries**, including the two settings whose meanings are not
  interchangeable — `0` is none and `None` is all — and the lines that exist before any turn has
  opened.
* **`find(name, role)`**, including the `Font :` label in front of the `combo box` of the same name,
  and the tiebreak's fallback when nothing was ever seen.
* **`Exchange`, `Done`, `Intent` and `Trigger` fields as they come out**, `trigger.asked` among them.
* **`poll()` and the `MIN_GAP` floor**, against a clock this file moves.
* **`perform()`'s re-resolution**, the vanished target, the unprompted downgrade, and the effect
  kind that reaches the `else`.

What is **not** pinned, and each omission is a decision:

* **The rendered prompt.** `measure/choosing.py`'s `prompt_record()` owns that and tags it `r322a4c`;
  a second copy here would be a second fingerprint of the same bytes, forking on every prompt edit
  and inviting somebody to re-bless one of them. What this file pins is the *handoff* — the window
  `recent()` produced — and it checks, live and format-free, that the window's lines reached the
  prompt and that the dropped ones did not.
* **`think_s` and `act_s` as durations.** They are pinned as *numbers* only because the clock below
  is fake; what that pins is which region of `think()` each one spans, which is behaviour. Nothing
  here is a latency measurement and no number in it belongs in a finding.
* **`__repr__`.** Debugging output, deterministic but not behaviour. Pinning it would make a port
  answerable for the shape of its own debug strings, and a golden file whose entries are not all
  equally authoritative is one somebody will argue with a line at a time.
* **`recent(history, "2")`.** The `int(turns)` coercion accepts a string; that is an accident of
  Python rather than a decision, and pinning an accident obliges a port to reproduce it.

## Re-blessing, and what has to be true first

    ARGMIND_BLESS_LOOP=re-bless python -m pytest tests/test_loop_golden.py

**A diff in the golden file is a behaviour change in the loop.** Not a formatting difference, not a
refactor that came out the same shape: this file renders behaviour and nothing else, so if the
transcript moved, what the agent writes into its own conversation, or hands the model, or does to
the world, moved with it. The honest reasons to accept one are that a decision was taken and
recorded, or that this file was extended to cover more. **A behaviour change and a re-bless must not
be one commit** — the record of what the loop used to do is the whole asset, and a golden file
rewritten in the same breath as the code pins the new behaviour while hiding that it is new. Nor is
a re-bless the way to make a port pass: the port is what has to match, in the direction the
transcript already points.

Two mechanisms keep that from happening quietly. The blessing run **fails** after writing, so no
invocation both re-blesses and reports green. And the file carries a tag over its own body, so a
hand edit of one line fails as a hand edit rather than as a diff, and a legitimate re-bless moves
the tag onto the first line of the diff, where a reviewer meets it before the change itself.

## What it needs to run

No display, no model, no network, no device, no toolchain. `time` is replaced in both core modules
that read it, so the wall clock is never consulted: the substitution is at the module attribute
because injecting a clock would be a change to `loop.py`, and this file is not allowed to move the
behaviour it is recording. Everything with a dict or a set in it is rendered in sorted order.
"""

from __future__ import annotations

import hashlib
import os
import pathlib
import re

import pytest

from argmind.core import intent as intent_module
from argmind.core import loop as loop_module
from argmind.core.contract import Widget
from argmind.core.intent import (MIN_GAP, Appeared, Effect, Intent, Perceived, Said, Vanished,
                                 WindowChanged)
from argmind.core.llm import Reply
from argmind.core.loop import (ING, NOTHING_DONE, PAST, PLAIN, REMEMBER_TURNS, SPOKE, TURN_OPENS,
                               VERBS, Done, Session, recent)

GOLDEN = pathlib.Path(__file__).resolve().parent / "golden" / "loop-transcript.txt"

#: The one way to overwrite the file. Long enough not to be typed by accident, and named in the
#: header of the file it rewrites.
BLESS = ("ARGMIND_BLESS_LOOP", "re-bless")

SENTINEL = "#---- transcript ----"

#: The fake environment's answers. Powers of two, so every duration below is exact in binary and a
#: rendered float carries no rounding of its own.
LOOK_S = 0.125
TICK = 0.25
START = 1000.0


# ---- a world that does not move unless this file moves it ---------------------------------------

class Clock:
    """The wall clock, replaced. **`time()` never advances on its own and `perf_counter()` always
    does**, and the asymmetry is deliberate: the floor a watch keeps is a claim about how much time
    passed, so this file states it outright; the duration of a region of code is a claim about
    *which* region, so a fixed tick per read pins the region and nothing else."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.now = START
        self.ticks = 0

    def time(self) -> float:
        return self.now

    def perf_counter(self) -> float:
        self.ticks += 1
        return self.ticks * TICK

    def advance(self, seconds) -> None:
        self.now += float(seconds)


class ScriptedEyes:
    """One screen per read, the last repeating. **A turn reads up to three times** — to ask, to
    re-resolve immediately before acting, and to confirm — and the whole design exists because those
    three reads disagree, so a fake that cannot disagree cannot express the thing under test."""

    def __init__(self, *screens) -> None:
        self.screens = [list(s) for s in screens] or [[]]
        self.reads = 0

    def foreground(self):
        screen = self.screens[min(self.reads, len(self.screens) - 1)]
        self.reads += 1
        return list(screen)

    def timed_foreground(self):
        return self.foreground(), LOOK_S


class ScriptedHands:
    """Declares no vocabulary, which is the fallback case: every key is kept."""

    def __init__(self, live=False) -> None:
        self.live = live
        self.done = []

    def click(self, widget, button="left"):
        self.done.append(f"click {widget.role} {widget.name!r} at {widget.centre}")
        return {"do": "click"}

    def type_text(self, text):
        self.done.append(f"type {text!r}")
        return {"do": "type"}

    def press(self, key):
        self.done.append(f"key {key!r}")
        return {"do": "key"}


class DeclaringHands(ScriptedHands):
    """An environment that publishes its key names — and refuses anything else, so a broken filter
    fails on the press rather than on an assertion nobody wrote."""

    def __init__(self, *names, live=False) -> None:
        super().__init__(live)
        self.names = tuple(names)

    def keys(self):
        return self.names

    def press(self, key):
        if key not in self.names:
            raise AssertionError(f"the core pressed {key!r}, which this environment refuses")
        return super().press(key)


class ScriptedModel:
    """Canned answers in order, and every prompt kept. Asserts the core hands it no image."""

    def __init__(self, *texts) -> None:
        self.texts = list(texts)
        self.prompts = []

    def ask(self, prompt, images=None):
        assert images is None, "the core must never hand a model an image"
        self.prompts.append(prompt)
        text = self.texts.pop(0) if self.texts else '{"say": "", "do": []}'
        return Reply(text, 0.001, 0.002, 100, 10, "stop")


def says(text, *actions):
    """The model's answer, in the shape the prompt asks for."""
    import json                                                          # noqa: PLC0415

    return json.dumps({"say": text, "do": list(actions)}, ensure_ascii=False)


def w(name, role="button", rect=(10, 20, 40, 30), window="Win"):
    return Widget(name, role, rect, window)


DESK = [w("닫기", rect=(10, 20, 40, 30)),
        w("Share", rect=(60, 20, 40, 30)),
        w("Mute (m)", rect=(110, 20, 40, 30)),
        w("File", role="menu item", rect=(35, 0, 36, 35))]


def sitting(model=None, *screens, hands=None, **kw):
    """A `Session` over a scripted world. Nothing here knows what platform it is on."""
    return Session(eyes=ScriptedEyes(*(screens or (DESK,))),
                   model=model if model is not None else ScriptedModel(),
                   hands=hands if hands is not None else ScriptedHands(), **kw)


# ---- how a thing is written down ----------------------------------------------------------------

class Transcript:
    """Lines, and a heading for each thing being pinned. The commentary goes in the file itself, so
    a diff arrives beside the reason the line is there."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def section(self, title) -> None:
        self.lines += ["", f"=== {title} ===", ""]

    def note(self, text) -> None:
        self.lines.append(f"# {text}")

    def line(self, text="") -> None:
        self.lines.append(text)


def _widget(widget) -> str:
    if widget is None:
        return "(none)"
    x, y, width, height = widget.rect
    return (f"{widget.role} {widget.name!r} @ {x},{y} {width}x{height} "
            f"window={widget.window!r} centre={widget.centre}")


def _effect(effect) -> str:
    because = effect.because.kind if effect.because is not None else None
    return (f"{effect.kind} target={effect.target!r} role={effect.role!r} "
            f"text={effect.text!r} because={because} unprompted={effect.unprompted}")


def _done(done) -> str:
    return f"{_effect(done.effect)} | target={_widget(done.target)} | confirmed={done.confirmed}"


def _history(transcript, history, indent="  ") -> None:
    """Every history line, quoted. **Quoted because one of them ends in a space** — an effect of a
    kind `VERBS` does not list is described as a press of nothing — and a golden file carrying
    invisible whitespace is one that any editor's trim-on-save can break while looking untouched.
    `repr` is injective, so nothing is lost by making it visible."""
    if not history:
        transcript.line(f"{indent}(empty)")
    for who, text in history:
        transcript.line(f"{indent}{who} | {text!r}")


def _exchange(transcript, exchange) -> None:
    trigger = exchange.trigger
    transcript.line(f"  n={exchange.n}")
    transcript.line(f"  trigger kind={trigger.kind!r} asked={trigger.asked} "
                    f"at={trigger.at:.6f} text={trigger.text!r}")
    transcript.line(f"  said={exchange.said!r}")
    transcript.line(f"  n_widgets={exchange.n_widgets}")
    transcript.line(f"  look_s={exchange.look_s:.6f} think_s={exchange.think_s:.6f} "
                    f"act_s={exchange.act_s:.6f} total_s={exchange.total_s:.6f}")
    for name, items in (("did", exchange.did), ("proposed", exchange.proposed),
                        ("refused", exchange.refused)):
        if not items:
            transcript.line(f"  {name}: (none)")
        for item in items:
            transcript.line(f"  {name}: {_done(item) if name == 'did' else _effect(item)}")


def _report(transcript, session) -> None:
    report = session.report()
    for key in sorted(report):
        transcript.line(f"  {key} = {report[key]}")


def _counters(transcript, session) -> None:
    transcript.line(f"  blind={session.blind} vanished={session.vanished} "
                    f"unknown_keys={session.unknown_keys} reads={session.eyes.reads}")


# ---- the transcript -----------------------------------------------------------------------------

def _constants(transcript, _clock) -> None:
    transcript.section("0. the constants the loop is written around")
    transcript.note("A port that changes one of these has changed a derived number, not a name.")
    transcript.note("MOST_AT_ONCE and MAX_LISTED are converse.py's and are pinned by r322a4c.")
    transcript.line(f"  REMEMBER_TURNS = {REMEMBER_TURNS}")
    transcript.line(f"  MIN_GAP = {MIN_GAP}")
    transcript.line(f"  SPOKE, NOTICED = {SPOKE!r}, {TURN_OPENS[1]!r}")
    transcript.line(f"  TURN_OPENS = {TURN_OPENS!r}")
    transcript.line(f"  NOTHING_DONE = {NOTHING_DONE!r}")
    transcript.line(f"  PAST, PLAIN, ING = {PAST}, {PLAIN}, {ING}")
    for kind in sorted(VERBS):
        transcript.line(f"  VERBS[{kind!r}] = {VERBS[kind]!r}")


CHANNEL_EFFECTS = (
    ("click with a role", Effect("click", target="Copy", role="button")),
    ("click with no role", Effect("click", target="Copy")),
    ("type", Effect("type", text="hello")),
    ("key", Effect("key", text="enter")),
    ("a kind VERBS does not list", Effect("scroll", target="Pane", role="pane")),
)


def _channels(transcript, _clock) -> None:
    transcript.section("1. remember_doing(): four channels, and the tense carries the state")
    transcript.note("The past tense belongs to `did` and to nothing else; every other line says in")
    transcript.note("words that nothing happened, so a reader that ignores the label still cannot")
    transcript.note("conclude a click landed. Only `offered` and `turned_down` are fed anywhere in")
    transcript.note("measure/, which makes these the least protected sentences in the file.")
    for channel in ("offered", "refused", "turned_down"):
        transcript.line()
        transcript.line(f"-- {channel} --")
        for what, effect in CHANNEL_EFFECTS:
            session = sitting()
            session.remember_doing(**{channel: [effect]})
            transcript.line(f"  [{what}]")
            _history(transcript, session.history, indent="    ")

    transcript.line()
    transcript.line("-- did --")
    transcript.note("`did` is the one channel handed the widget as it was AT THE MOMENT OF ACTING,")
    transcript.note("so its role comes from the world while the other three print the role the")
    transcript.note("model chose. Here the model chose `button` and the world says `combo box`.")
    for what, effect in CHANNEL_EFFECTS:
        session = sitting()
        target = w("Copy", role="combo box") if effect.kind == "click" else None
        session.remember_doing(did=[Done(effect, target)])
        transcript.line(f"  [{what}]")
        _history(transcript, session.history, indent="    ")

    transcript.line()
    transcript.line("-- the order within one call --")
    transcript.note("Source order: offered, did, could not, they said no to. It is not cosmetic —")
    transcript.note("converse.still_open() closes every open offer on any settling line, so an")
    transcript.note("offer written in the same call as a settlement is already closed by the time")
    transcript.note("the next prompt is built.")
    session = sitting()
    session.remember_doing(did=[Done(Effect("click", target="Cut", role="button"), w("Cut"))],
                           refused=[Effect("type", text="bye")],
                           offered=[Effect("click", target="Copy", role="button")],
                           turned_down=[Effect("key", text="tab")])
    _history(transcript, session.history)

    transcript.line()
    transcript.line("-- nothing to record --")
    transcript.note("Four empty channels is the default, so a turn that did nothing writes nothing.")
    session = sitting()
    session.remember_doing()
    _history(transcript, session.history)


WINDOW_HISTORY = [
    ("offered", 'to click the button "Copy" — nothing was done; they have not accepted'),
    ("them", "alpha"), ("you", "reply alpha"),
    ("them", "bravo"), ("you", "reply bravo"), ("did", 'clicked the button "Share"'),
    ("noticed", "'Build succeeded' appeared"), ("you", "reply charlie"),
    ("offered", 'to click the button "Open" — nothing was done; they have not accepted'),
    ("them", "delta"), ("you", "reply delta"),
    ("could not", 'click the button "닫기" — it had gone by then; nothing was done'),
]


def _window(transcript, _clock) -> None:
    transcript.section("2. recent(): a window measured in turns, cut where a turn opens")
    transcript.note("A turn opens on `them` or `noticed` and on nothing else. 0 is none and None is")
    transcript.note("all: `history[-0:]` was the whole list, so the one value a person reaches for")
    transcript.note("to turn memory off used to turn it all the way up.")
    transcript.line("-- the history being windowed --")
    _history(transcript, WINDOW_HISTORY)
    for turns in (None, 0, -1, 1, 2, 3, 4, 99):
        transcript.line()
        window = recent(WINDOW_HISTORY, turns)
        transcript.line(f"-- turns={turns!r} -> {len(window)} lines --")
        _history(transcript, window)

    transcript.line()
    transcript.line("-- a history no turn has opened --")
    transcript.note("Lines written before anybody spoke are legitimate: remember_doing() on a fresh")
    transcript.note("session produces them. They are carried while they are all there is.")
    orphans = WINDOW_HISTORY[:1] + [("did", 'clicked the button "Share"')]
    for turns in (None, 0, 1, 12):
        transcript.line(f"  turns={turns!r} -> {len(recent(orphans, turns))} lines")
        _history(transcript, recent(orphans, turns), indent="    ")
    transcript.line(f"  an empty history, turns=12 -> {recent([], 12)}")


def _window_live(transcript, _clock) -> None:
    transcript.section("3. the window as decide() hands it over, turn by turn")
    transcript.note("remember_turns=2. Turn one writes three lines and turn three writes three, so")
    transcript.note("a two-turn window is four lines at one boundary and six at another — which is")
    transcript.note("the whole reason the unit is turns: the expensive turns are the ones a")
    transcript.note("follow-up refers back to, and a line window shrank the memory exactly there.")
    model = ScriptedModel(says("reply one", {"click": 1}), says("reply two"),
                          says("reply three", {"click": 0}), says("reply four"))
    session = sitting(model, DESK, DESK, DESK, DESK, [x for x in DESK if x.name != "닫기"],
                      remember_turns=2)
    for n, heard in enumerate(("line one", "line two", "line three", "line four"), start=1):
        #: What `decide` is about to pass: the history does not change between here and the call.
        window = session.recent()
        session.exchange(heard)
        transcript.line(f"-- before turn {n} ({heard!r}) --")
        _history(transcript, window)
        #: Live, and free of any prompt format: the window reached the prompt and the lines the
        #: window dropped did not. Pinning the rendered block would be a second copy of r322a4c.
        prompt = session.model.prompts[-1]
        carried = {text for _who, text in window}
        for text in carried:
            assert text in prompt, f"a line in the window never reached the prompt: {text!r}"
        for _who, text in session.history:
            if text not in carried and text != heard:
                assert text not in prompt, f"a dropped line reached the prompt: {text!r}"
    transcript.line()
    transcript.line("-- the whole history afterwards --")
    _history(transcript, session.history)
    transcript.line()
    transcript.line(f"-- recent() afterwards -> {len(session.recent())} lines --")
    _history(transcript, session.recent())


def _finding(transcript, _clock) -> None:
    transcript.section("4. find(): the world as it is now, and the role is half of the key")
    transcript.note("Character Map, measured: a `text` reading 'Font :' sits in front of a `combo")
    transcript.note("box` of the same name. The name alone returns the label, the click lands on a")
    transcript.note("piece of text and does nothing — indistinguishable from a click that missed.")
    fonts = [w("Font :", role="text", rect=(34, 107, 53, 20)),
             w("Font :", role="combo box", rect=(86, 106, 321, 30))]
    session = sitting(None, fonts)
    session.observe()
    for name, role in (("Font :", "combo box"), ("Font :", "text"), ("Font :", ""),
                       ("Font :", "button"), ("Nothing", ""), ("Nothing", "button")):
        transcript.line(f"  find({name!r}, {role!r}) -> {_widget(session.find(name, role))}")
    transcript.line(f"  reads consumed: {session.eyes.reads} (one per find, plus the observe)")

    transcript.line()
    transcript.line("-- the tiebreak among namesakes of the same role --")
    transcript.note("Nearest to where a thing of that role was last seen: a thing moves rather than")
    transcript.note("teleporting past its own copies.")
    seen = [w("Item", role="list item", rect=(0, 100, 40, 20)),
            w("Item", role="list item", rect=(0, 400, 40, 20))]
    scrolled = [w("Item", role="list item", rect=(0, 300, 40, 20)),
                w("Item", role="list item", rect=(0, 130, 40, 20))]
    session = sitting(None, seen, scrolled)
    session.observe()
    transcript.line(f"  last seen at {[x.centre for x in seen]}, now at "
                    f"{[x.centre for x in scrolled]}")
    transcript.line(f"  find('Item', 'list item') -> {_widget(session.find('Item', 'list item'))}")

    transcript.line()
    transcript.line("-- the tiebreak with nothing ever seen --")
    transcript.note("`find` is public because an interface resolving an action a second way would")
    transcript.note("be a second answer, so it has to work before any observe: the fallback is the")
    transcript.note("first candidate, not a crash and not None.")
    session = sitting(None, scrolled)
    transcript.line(f"  seen={session.seen}")
    transcript.line(f"  find('Item', 'list item') -> {_widget(session.find('Item', 'list item'))}")

    transcript.line()
    transcript.line("-- the role filters the candidates, it does not merely break a tie --")
    transcript.note("A mutation found this: with the role dropped from the filter the Character Map")
    transcript.note("case still passed, because the nearest thing of that role was the right one.")
    transcript.note("Only when the chosen thing MOVES does a namesake of another role end up nearer")
    transcript.note("to where it used to be, and win.")
    label = w("Item", role="text", rect=(0, 100, 40, 20))
    seen = [label, w("Item", role="list item", rect=(0, 200, 40, 20))]
    scrolled = [label, w("Item", role="list item", rect=(0, 320, 40, 20))]
    session = sitting(None, seen, scrolled)
    session.observe()
    transcript.line(f"  find('Item', 'list item') -> {_widget(session.find('Item', 'list item'))}")
    transcript.line(f"  find('Item', '')          -> {_widget(session.find('Item'))}")


def _one_turn(transcript, clock) -> None:
    transcript.section("5. one turn, every field of what comes out of it")
    transcript.note("Three turns on one session: one that acts, one that only speaks, one that does")
    transcript.note("three things. look_s is the environment's own number; think_s and act_s come")
    transcript.note("from the fake clock and pin WHICH REGION is timed, not how long it takes.")
    model = ScriptedModel(says("닫을게요", {"click": 0}),
                          says("세 개 열려 있어요"),
                          says("검색할게요", {"click": 1}, {"type": "argmind"}, {"key": "enter"}))
    session = sitting(model, DESK, hands=ScriptedHands(live=True))
    for heard in ("닫아줘", "뭐가 열려 있지?", "share에 argmind 치고 엔터"):
        clock.advance(10.0)
        exchange = session.exchange(heard)
        transcript.line(f"-- {heard!r} --")
        _exchange(transcript, exchange)
        transcript.line(f"  hands: {session.hands.done[-3:] or '(nothing)'}")
        _counters(transcript, session)
    transcript.line()
    transcript.line("-- the conversation it wrote for itself --")
    _history(transcript, session.history)
    transcript.line()
    transcript.line("-- report() --")
    _report(transcript, session)
    transcript.line()
    transcript.line("-- report() on a session that has not thought yet --")
    _report(transcript, sitting())


def _unprompted(transcript, clock) -> None:
    transcript.section("6. noticing, and the permission that is not in the utterance")
    transcript.note("An unprompted effect without standing permission is downgraded to speech, not")
    transcript.note("refused: the refusal path is conversational, and the default is that way")
    transcript.note("because a wrong click has no undo.")
    for allowed in (False, True):
        clock.advance(10.0)
        model = ScriptedModel(says("빌드 끝났어요. 열까요?", {"click": 1}))
        session = sitting(model, DESK, hands=ScriptedHands(live=True),
                          may_act_unprompted=allowed)
        exchange = session.think(Perceived("'Build succeeded' appeared"))
        transcript.line(f"-- may_act_unprompted={allowed} --")
        _exchange(transcript, exchange)
        transcript.line(f"  hands: {session.hands.done or '(nothing)'}")
        transcript.line("  history:")
        _history(transcript, session.history, indent="    ")
        _report(transcript, session)
        transcript.line()
    transcript.note("THE OFFER IS NOT WRITTEN DOWN BY THE LOOP. think() records `did` and `refused`")
    transcript.note("and nothing else, so a downgraded effect leaves no `offered` line and")
    transcript.note("converse.still_open() cannot see it. Only an interface — app.py's proposal")
    transcript.note("row — feeds that channel. Recorded here as it stands, not fixed here.")

    transcript.line()
    transcript.line("-- an effect nobody caused runs whoever is asking --")
    transcript.note("`unprompted` is false when `because` is None, so an effect built by hand is")
    transcript.note("not treated as the agent's own idea. That is the permission model's open edge")
    transcript.note("and it is written down here rather than argued about.")
    session = sitting(None, DESK, hands=ScriptedHands(live=True))
    session.observe()
    did, proposed, refused = session.perform(Intent("", [Effect("key", text="enter")]))
    transcript.line(f"  did={[_done(d) for d in did]}")
    transcript.line(f"  proposed={proposed} refused={refused}")
    transcript.line(f"  hands: {session.hands.done}")


def _watching(transcript, clock) -> None:
    transcript.section("7. poll(), and the floor a watch keeps")
    transcript.note("Perception changes 13 seconds in 14 and a model call costs 0.3-2 s, so an")
    transcript.note("unfloored watch is a model that never stops and an agent that never listens.")
    transcript.note("The clock below is moved by this file; nothing here reads the wall.")
    win = [w("닫기"), w("Share"), w("Mute (m)")]
    mail = [w("Send", window="Mail"), w("Build succeeded", window="Mail")]
    back = [w("닫기"), w("Build again"), w("Mute (m)")]
    session = sitting(None, win, mail, back, mail, back,
                      watches=[WindowChanged(), Appeared("build"), Vanished("mute")])
    transcript.line(f"  watches: WindowChanged(gap={MIN_GAP}), Appeared('build'), "
                    f"Vanished('mute')")
    plan = [(0.0, "first poll: there is no BEFORE, so nothing whatever happened"),
            (0.0, "the window changed, something appeared, something went away"),
            (1.0, "all three would fire and all three are inside the floor"),
            (2.5, "3.5 s after the LAST FIRING, not 2.5 s after the last check"),
            (3.0, "exactly at the floor, which is not inside it")]
    for advance, why in plan:
        clock.advance(advance)
        fired = session.poll()
        transcript.line()
        transcript.line(f"-- t={clock.now:.6f} (+{advance}) : {why} --")
        transcript.line(f"  seen now: {[x.name for x in session.seen]}")
        if not fired:
            transcript.line("  fired: (nothing)")
        for trigger in fired:
            transcript.line(f"  fired: kind={trigger.kind!r} asked={trigger.asked} "
                            f"at={trigger.at:.6f} text={trigger.text!r}")

    transcript.line()
    transcript.line("-- a session with no watches still looks --")
    transcript.note("poll() observes whether or not anything is watching, so `seen` is fresh for")
    transcript.note("the next decision. Returning [] because it never looked is a different bug.")
    session = sitting(None, win, mail)
    for _ in range(2):
        transcript.line(f"  poll() -> {session.poll()}  seen={[x.name for x in session.seen]} "
                        f"reads={session.eyes.reads}")


def _acting(transcript, clock) -> None:
    transcript.section("8. perform(): resolved against the world AS IT IS NOW")
    transcript.note("Thirteen times in fourteen, over the second a call takes, something listed when")
    transcript.note("the question was asked is gone when the answer arrives. Clicking a remembered")
    transcript.note("rectangle is clicking where something used to be.")

    clock.advance(10.0)
    moved = [w("닫기", rect=(999, 500, 40, 30))] + DESK[1:]
    session = sitting(ScriptedModel(says("닫을게요", {"click": 0})), DESK, moved, moved,
                      hands=ScriptedHands(live=True))
    exchange = session.exchange("close it")
    transcript.line("-- the target moved between the question and the answer --")
    _exchange(transcript, exchange)
    transcript.line(f"  hands: {session.hands.done}")

    clock.advance(10.0)
    gone = [x for x in DESK if x.name != "닫기"]
    session = sitting(ScriptedModel(says("닫을게요", {"click": 0})), DESK, gone,
                      hands=ScriptedHands(live=True))
    exchange = session.exchange("close it")
    transcript.line()
    transcript.line("-- the target had gone by then --")
    _exchange(transcript, exchange)
    transcript.line(f"  hands: {session.hands.done or '(nothing)'}")
    _counters(transcript, session)
    transcript.line("  history:")
    _history(transcript, session.history, indent="    ")

    transcript.line()
    transcript.line("-- confirmation asks the widget layer, and only when something was clicked --")
    transcript.note("None in a dry run: nothing was clicked, so nothing should have changed, and")
    transcript.note("False there would read as a failure that never happened.")
    other = [w("Send", window="Mail")]
    for what, screens, live in (("dry run", (DESK,), False),
                                ("still exactly where it was", (DESK,), True),
                                ("the thing is gone", (DESK, DESK, gone), True),
                                ("the window changed", (DESK, DESK, other), True)):
        clock.advance(10.0)
        session = sitting(ScriptedModel(says("ok", {"click": 0})), *screens,
                          hands=ScriptedHands(live=live))
        exchange = session.exchange("close it")
        transcript.line(f"  [{what}] confirmed={exchange.did[0].confirmed} "
                        f"target={_widget(exchange.did[0].target)}")

    transcript.line()
    transcript.line("-- an effect of a kind perform() has no arm for --")
    transcript.note("Refused, not attempted and not raised. converse() cannot build one, so this is")
    transcript.note("reachable only by an interface constructing an Intent — which is a thing an")
    transcript.note("interface is allowed to do.")
    session = sitting(None, DESK, hands=ScriptedHands(live=True))
    session.observe()
    did, proposed, refused = session.perform(
        Intent("", [Effect("scroll", target="Pane", role="pane"),
                    Effect("click", target="Share", role="button")]))
    transcript.line(f"  did={[_done(d) for d in did]}")
    transcript.line(f"  proposed={proposed}")
    transcript.line(f"  refused={[_effect(e) for e in refused]}")
    transcript.line(f"  hands: {session.hands.done}")
    transcript.note("`vanished` counts a target that went, and an unknown kind is not that.")
    _counters(transcript, session)


def _keys(transcript, clock) -> None:
    transcript.section("9. decide(): a key the environment does not have")
    transcript.note("The model has emitted {'key': 'zoom in'}, a key on no platform. `press` raises")
    transcript.note("and nothing between it and think() catches, so one invented keystroke used to")
    transcript.note("cost the whole turn — including the sentence already composed.")
    for hands, what in ((DeclaringHands("enter", "tab", "escape"), "declares three keys"),
                        (ScriptedHands(), "declares nothing, so keeps every key")):
        clock.advance(10.0)
        session = sitting(ScriptedModel(says("확대할게요", {"key": "zoom in"})), DESK, hands=hands)
        exchange = session.exchange("확대해줘")
        transcript.line(f"-- {what} --")
        _exchange(transcript, exchange)
        transcript.line(f"  hands: {session.hands.done or '(nothing)'}")
        _counters(transcript, session)
        transcript.line("  history:")
        _history(transcript, session.history, indent="    ")
        transcript.line()
    transcript.note("A dropped key is not filed as something that WENT AWAY: `could not` means it")
    transcript.note("had gone by then, and that sentence would be false in the core's own context")
    transcript.note("on the very next call.")

    transcript.line()
    transcript.line("-- decide() looks first if nothing has been observed --")
    transcript.note("A decision needs a world, and the first read is not the interface's job to")
    transcript.note("remember. The count is how a port shows it did the same thing.")
    session = sitting(ScriptedModel(says("ok")), DESK)
    transcript.line(f"  before: seen={session.seen} reads={session.eyes.reads}")
    intent = session.decide(Said("뭐가 보여?"))
    transcript.line(f"  after:  n_seen={len(session.seen)} reads={session.eyes.reads} "
                    f"say={intent.say!r} effects={intent.effects}")
    transcript.line(f"  intent.trigger kind={intent.trigger.kind!r} asked={intent.trigger.asked}")


SCENES = (_constants, _channels, _window, _window_live, _finding, _one_turn, _unprompted,
          _watching, _acting, _keys)


def render(clock) -> str:
    """The whole transcript, from a clock that starts where it always starts."""
    clock.reset()
    transcript = Transcript()
    for scene in SCENES:
        scene(transcript, clock)
    return "\n".join(transcript.lines).strip("\n") + "\n"


# ---- the golden file ----------------------------------------------------------------------------

def tag(body: str) -> str:
    """A short fingerprint of the transcript, in the shape `measure/` already uses."""
    return "t" + hashlib.sha256(body.encode("utf-8")).hexdigest()[:6]


def header(body: str) -> str:
    return ("# argmind — the golden transcript of argmind/core/loop.py.\n"
            "# GENERATED. Do not edit by hand: the tag below is taken over the body, and a hand\n"
            "# edit fails as a hand edit rather than as a difference in behaviour.\n"
            "#\n"
            f"# tag {tag(body)}\n"
            "#\n"
            f"# Re-bless:  {BLESS[0]}={BLESS[1]} python -m pytest tests/test_loop_golden.py\n"
            "# A diff below is a behaviour change in the loop. Read the docstring of\n"
            "# tests/test_loop_golden.py before accepting one, and never commit it together with\n"
            "# the change it records.\n"
            f"{SENTINEL}\n")


def split(text: str) -> tuple[str, str]:
    head, sentinel, body = text.partition(SENTINEL + "\n")
    assert sentinel, f"{GOLDEN} has no {SENTINEL!r} line; it is not a transcript this can read"
    return head, body


@pytest.fixture
def clock(monkeypatch):
    """`time` replaced in both core modules that read it — `intent` for the wall clock a trigger
    stamps itself with and a watch measures its floor against, `loop` for the three regions of a
    thought. Replaced at the module attribute because injecting a clock would be a change to the
    file this test exists to record."""
    fake = Clock()
    monkeypatch.setattr(intent_module, "time", fake)
    monkeypatch.setattr(loop_module, "time", fake)
    return fake


def test_the_LOOP_writes_the_TRANSCRIPT_IT_WAS_BLESSED_WITH(clock):
    """**The equivalence test `loop.py` did not have.** A hash of the source cannot survive the
    source being replaced, which is exactly what a port does to it; this compares what the loop
    *writes* — every history line, every field of an `Exchange`, every resolution and every refusal
    — against a file recorded when the behaviour was the behaviour somebody decided on.

    A failure here is not a broken test. It is the loop doing something else, and the first
    differing line says which of those things.
    """
    fresh = render(clock)
    if os.environ.get(BLESS[0]) == BLESS[1]:
        was = tag(split(GOLDEN.read_text(encoding="utf-8"))[1]) if GOLDEN.exists() else "(none)"
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(header(fresh) + fresh, encoding="utf-8", newline="\n")
        pytest.fail(f"re-blessed {GOLDEN.name}: {was} -> {tag(fresh)}. This run does not pass, on "
                    "purpose — read the diff, and commit it apart from whatever changed the "
                    "behaviour it records.")
    assert GOLDEN.exists(), (
        f"{GOLDEN} is missing. It is the record of what the loop does; create it with "
        f"{BLESS[0]}={BLESS[1]} python -m pytest {pathlib.Path(__file__).name}")
    stored = split(GOLDEN.read_text(encoding="utf-8"))[1]
    if stored != fresh:
        pairs = zip(stored.splitlines(), fresh.splitlines())
        first = next((n for n, (a, b) in enumerate(pairs, 1) if a != b),
                     min(len(stored.splitlines()), len(fresh.splitlines())) + 1)
        assert stored == fresh, (
            f"the loop no longer writes what {GOLDEN.name} records — first difference at line "
            f"{first}, stored {tag(stored)} against {tag(fresh)} now. If the change was decided, "
            "re-bless in a commit of its own; if it was not, this is the regression.")


def test_the_TRANSCRIPT_HAS_NOT_BEEN_EDITED_BY_HAND():
    """**A golden file that is easy to overwrite quietly is how a rewrite launders a change.** One
    line of the body edited to match new behaviour would otherwise be indistinguishable from a line
    that was always there. The tag is taken over the body, so an edit that does not go through the
    blessing path forks the two and says so — and a blessing that does moves the tag to the top of
    the diff, where a reviewer meets it before the change itself.
    """
    assert GOLDEN.exists(), f"{GOLDEN} is missing"
    head, body = split(GOLDEN.read_text(encoding="utf-8"))
    stated = re.search(r"^# tag (t[0-9a-f]{6})$", head, re.MULTILINE)
    assert stated, f"{GOLDEN.name} carries no tag line; it cannot vouch for its own body"
    assert stated.group(1) == tag(body), (
        f"{GOLDEN.name} says it is {stated.group(1)} and its body is {tag(body)} — the file has "
        "been edited by hand. Do not correct the tag: re-bless, so that the change is a diff "
        "somebody chose rather than a line somebody typed.")
    assert BLESS[0] in head and BLESS[1] in head, (
        "the file no longer says how to re-bless it, which is the one thing its reader needs")


def test_the_TRANSCRIPT_IS_DETERMINISTIC_or_it_pins_nothing(clock):
    """**A golden file that renders differently twice is a test that fails at random and gets
    deleted.** Two renders in one process, so anything reading the wall clock, iterating a set, or
    depending on a dict's insertion order being incidental shows up here rather than on somebody
    else's machine six weeks from now.
    """
    assert render(clock) == render(clock)


def test_the_TRANSCRIPT_COVERS_EVERY_CHANNEL_of_remember_doing():
    """**The coverage claim, read off the function rather than believed.** Two of the four channels
    are fed by nothing anywhere in `measure/`, so a fifth arriving unexercised here would leave the
    least protected thing in the file exactly as unprotected as it was. The channels come from the
    signature, so a new one announces itself.
    """
    import inspect                                                       # noqa: PLC0415

    channels = [n for n in inspect.signature(Session.remember_doing).parameters if n != "self"]
    labels = {"offered": "offered", "did": "did", "refused": "could not",
              "turned_down": "they said no to"}
    assert set(channels) == set(labels), (
        f"remember_doing() now takes {channels}; the transcript feeds {sorted(labels)}, and a "
        "channel it does not feed is a channel this file does not protect")
    assert GOLDEN.exists(), f"{GOLDEN} is missing"
    body = split(GOLDEN.read_text(encoding="utf-8"))[1]
    for channel, label in labels.items():
        assert f"{label} | " in body, f"the transcript records no line on the {channel!r} channel"
