# SPDX-License-Identifier: Apache-2.0
"""The voice seam, tested with no microphone and no speaker — which is the point of it being a seam.

Two findings are held here against the source rather than against a docstring, because a rule that
lives only in prose is a rule that gets quietly undone:

> **The deterministic side says WHERE, and WHETHER. The probabilistic side says WHAT.**

WHERE is `test_core.py`'s. WHETHER is this file's, and it is the one voice threatens: a spoken
*"응"* must not be able to run an action, because a transcript is a recogniser's opinion about a
waveform and the gate on an irreversible click is a physical device event.

The second finding is that **`argmind/core/` does not change**, and that is protected the same way:
if a later commit grows an `Ears` protocol beside `Eyes`, one of these fails.
"""

from __future__ import annotations

import ast
import io
import pathlib

import pytest

from argmind.core import Said, Session
from argmind.voice import Heard, Listener, Speaker

ARGMIND = pathlib.Path(__file__).resolve().parents[1] / "argmind"
CORE = ARGMIND / "core"
SEAM = ARGMIND / "voice.py"

#: Everything the core reaches `perform` through. `think()` and `exchange()` are on the list because
#: they *call* `perform`, and a front end written the natural way — `session.exchange(heard.text)` —
#: acts on a transcript with no button anywhere while containing no call to `perform` at all.
REACHES_AN_ACTION = ("perform", "think", "exchange")


def _modules(root):
    return [p for p in sorted(root.rglob("*.py")) if "__pycache__" not in p.parts]


def _tree(path):
    return ast.parse(io.open(path, encoding="utf-8").read(), filename=str(path))


def _names(tree):
    """Every identifier the module actually uses — not its prose.

    A line-based scan is unusable here: `converse.py` and `loop.py` discuss speech, utterances and
    speakers at length in their docstrings, and that is description rather than dependence, exactly
    as `contract.py`'s platform comparison is.
    """
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            found.add(node.id)
        elif isinstance(node, ast.Attribute):
            found.add(node.attr)
        elif isinstance(node, ast.arg):
            found.add(node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            found.add(node.name)
        elif isinstance(node, ast.alias):
            found.add(node.asname or node.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return found


# ---- WHETHER: a transcript may start a thought and may not finish one ---------------------------

def test_the_SEAM_cannot_reach_an_ACTION_because_it_cannot_name_a_SESSION():
    """The structural half, and the strong one: `argmind/voice.py` imports nothing from the core.

    A protocol module that cannot name a `Session` cannot perform through one however it is later
    edited, so this is a property rather than an enumeration of today's call sites."""
    tree = _tree(SEAM)
    reached = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            reached |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            reached.add("." * node.level + (node.module or ""))
    assert not [m for m in reached if m.startswith((".core", "argmind"))], (
        f"the voice seam imports {sorted(reached)}; a transcript is one line from running an "
        "action once this module can name a Session")
    names = _names(tree)
    leaked = sorted(names & set(REACHES_AN_ACTION) | (names & {"Session"}))
    assert not leaked, f"the voice seam names {leaked}; speech may start a thought, never finish one"


def test_ONLY_THE_APP_AND_THE_CLI_can_reach_an_action_outside_the_core():
    """The assertion `test_app.py`'s is defeated by relocation rather than by edit.

    `test_app.py` counts calls to `.perform(` **in `app.py`**, and the app's whole safety argument
    is that there is exactly one. A voice front end in a new file would satisfy that test by not
    being in that file. This is the same claim made where it cannot be sidestepped: across the whole
    of `argmind/` outside the core, only two modules may reach an action at all.

    `__main__.py` is on the list because `talk`, `say` and `watch` call `exchange()` and `think()`,
    which perform directly and are gated by `--live` rather than by a button. The test below keeps
    that path and voice from ever meeting."""
    offenders = []
    for path in _modules(ARGMIND):
        if path.is_relative_to(CORE) or path.name in ("app.py", "__main__.py"):
            continue
        for node in ast.walk(_tree(path)):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr in REACHES_AN_ACTION):
                offenders.append(f"{path.name}:{node.lineno}: .{node.func.attr}()")
    assert not offenders, (
        f"{offenders} reach an action outside the app's confirmation button. Only app.py (one call "
        "to perform, behind 실행) and __main__.py (behind --live) may, and a module that a "
        "microphone feeds may never be either")


def test_the_CLI_and_the_VOICE_SEAM_never_meet():
    """`__main__.py` performs without a button — `talk` calls `exchange()` and `--live` is the only
    gate. That is fine for a developer tool driven from a keyboard and is exactly what must not be
    handed a microphone: a `--voice` flag on `talk` would let a mis-transcription click something
    with no confirmation anywhere. The two are kept exclusive here rather than remembered."""
    source = io.open(ARGMIND / "__main__.py", encoding="utf-8").read()
    assert "voice" not in source.lower(), (
        "the CLI has grown a voice; it reaches perform() through exchange() with no button, so "
        "speech there would confirm an action. Voice belongs to app.py, which has one")


def test_the_app_still_has_EXACTLY_ONE_way_to_run_an_action():
    """Stated again from this side because the two tests protect different things: `test_app.py`
    protects the app's design, this protects the *voice* argument, which assumes it."""
    tree = _tree(ARGMIND / "app.py")
    performs = [n for n in ast.walk(tree)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "perform"]
    reaches = [n for n in ast.walk(tree)
               if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
               and n.func.attr in ("think", "exchange")]
    assert len(performs) == 1, f"{len(performs)} calls to perform() in app.py; voice assumes one"
    assert not reaches, (
        "app.py has started calling think()/exchange(), which perform without reaching the button")


# ---- the core does not change --------------------------------------------------------------

def test_the_core_GREW_NO_EARS_AND_NO_MOUTH():
    """The finding this whole design is: voice attaches outside `Session`, so no core module knows
    a microphone exists.

    Identifiers only. `converse.py` and `loop.py` discuss speech and speakers in prose, and prose is
    description — the same distinction `test_the_core_NAMES_NO_PLATFORM` draws for the platform
    table in `contract.py`."""
    banned = ("voice", "microphone", "speaker", "listener", "audio", "transcript",
              "utterance", "whisper", "sapi", "sounddevice", "tts")
    offenders = []
    for path in _modules(CORE):
        for name in _names(_tree(path)):
            low = name.lower()
            for word in banned:
                if word in low:
                    offenders.append(f"{path.name}: {name}")
    assert not offenders, (
        f"the core has grown a notion of hearing or speaking: {offenders}. Session.exchange() takes "
        "a str and Intent.say is a str; voice converts either side of those and changes nothing "
        "here. A protocol in core/ that no core module calls is what contract.py refuses")


def test_the_SEAM_needs_no_third_party_package():
    """`argmind/core/` is importable with nothing but the standard library and so is this. Anything
    that needs `sounddevice`, torch or `comtypes` lives in `argmind/env/`, behind the seam."""
    stdlib = {"__future__", "typing", "time", "dataclasses", "enum", "abc"}
    imported = set()
    for node in ast.walk(_tree(SEAM)):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.add(node.module.split(".")[0])
    assert imported <= stdlib, f"the voice seam imports {sorted(imported - stdlib)}"


# ---- the type at the seam ------------------------------------------------------------------

def test_a_transcript_carries_NO_CONFIDENCE_and_that_is_deliberate():
    """A negative result protected by a test, which is this project's standard.

    Whisper publishes no calibrated confidence, so the field would be `None` on the only recogniser
    in sight — and a `None` confidence invites a threshold nobody can set. If a recogniser that
    publishes one is adopted, the field arrives with the measurement that sets the threshold."""
    assert Heard.__slots__ == ("text", "seconds", "at"), (
        "Heard has grown a field. If it is `confidence`, the measurement that sets its threshold "
        "belongs in the same commit; if it is `language`, its only consumer is a sentence in the "
        "prompt and this project is 0 for 5 on those")
    assert not hasattr(Heard("hi", 1.0), "confidence")


def test_PACE_is_the_one_signal_that_a_transcript_was_INVENTED():
    """Near-silence does not make Whisper quiet, it makes it produce a fixed phrase. Forty
    characters over a third of a second is not a bad transcription, it is a fabrication, and the
    ratio is how an implementation sees it — computed in one place so three callers cannot disagree
    about what "too fast to be speech" means."""
    speech = Heard("지우개 눌러줘", 2.0)
    fabrication = Heard("감사합니다.", 0.3)
    assert speech.pace == pytest.approx(3.5)
    #: Not an absolute cut-off — the seam names none, because the only pace measured on this machine
    #: is a synthesiser's. What it shows is the separation the cut-off would live in: a fixed phrase
    #: over a third of a second of room tone reads several times faster than a person.
    assert fabrication.pace > 5 * speech.pace
    #: No audio at all is not "infinitely slow"; it is a transcript with nothing behind it.
    assert Heard("감사합니다.", 0.0).pace == float("inf")
    assert Heard("", 2.0).pace == 0.0


# ---- fakeable the way Eyes and Hands are ----------------------------------------------------

class FakeListener:
    """Scripted utterances. **No microphone anywhere in this suite**, the way no test needs a
    display, a model or a network."""

    def __init__(self, *utterances) -> None:
        self.queue = list(utterances)
        self.capturing = False
        self.holds = []

    def hold(self, down: bool) -> None:
        self.capturing = bool(down)
        self.holds.append(bool(down))

    def heard(self):
        return self.queue.pop(0) if (self.queue and not self.capturing) else None

    def release(self) -> None:
        self.queue.clear()


class FakeSpeaker:
    """Records what was said aloud and whether it was cut off."""

    def __init__(self) -> None:
        self.spoken = []
        self.stopped = 0
        self._busy = False

    def say(self, text: str) -> None:
        self.spoken.append(text)
        self._busy = True

    def stop(self) -> None:
        self.stopped += 1
        self._busy = False

    def speaking(self) -> bool:
        return self._busy

    def release(self) -> None:
        self._busy = False


def test_a_fake_satisfies_the_protocols_so_nothing_here_needs_a_DEVICE():
    assert isinstance(FakeListener(), Listener)
    assert isinstance(FakeSpeaker(), Speaker)


def test_holding_the_key_is_what_CAPTURES_and_nothing_arrives_while_it_is_HELD():
    """Push-to-talk is the design and not a placeholder: the key is the floor an open microphone has
    no equivalent of, and it makes self-hearing impossible because the microphone is shut while the
    agent talks."""
    ear = FakeListener(Heard("지우개 눌러줘", 1.4))
    ear.hold(True)
    assert ear.heard() is None, "an utterance arrived while the person was still speaking"
    ear.hold(False)
    assert ear.heard().text == "지우개 눌러줘"
    assert ear.holds == [True, False]


def test_BARGE_IN_is_the_talk_key_stopping_the_MOUTH():
    """One line, because push-to-talk bought it: no voice activity detection, no echo cancellation,
    no half-duplex state machine. The utterance is cut and not resumed — the agent's speech is a
    courtesy and the person's is the input."""
    mouth, ear = FakeSpeaker(), FakeListener()
    mouth.say("빨간색을 선택할 수 있어요. 그리고…")
    assert mouth.speaking()
    #: What `app.py` owes on the talk key going down, in the order it owes it.
    mouth.stop()
    ear.hold(True)
    assert not mouth.speaking() and mouth.stopped == 1
    assert len(mouth.spoken) == 1, "the interrupted utterance was resumed"


# ---- a spoken turn is an ordinary turn -------------------------------------------------------

class OneWidget:
    def foreground(self):
        return []

    def timed_foreground(self):
        return [], 0.0


class SilentModel:
    def ask(self, prompt, images=None):
        from argmind.core.llm import Reply                              # noqa: PLC0415

        assert images is None
        return Reply('{"say": "네", "do": []}', 0.01, 0.02, 10, 5, "stop")


class NoHands:
    live = False

    def click(self, widget, button="left"):
        raise AssertionError("nothing spoken may reach an action")

    def type_text(self, text):
        raise AssertionError("nothing spoken may reach an action")

    def press(self, key):
        raise AssertionError("nothing spoken may reach an action")


def test_a_SPOKEN_turn_is_Said_like_any_other_and_stays_ASKED():
    """**No new `Trigger` kind, and the reason is a one-word failure.** `Trigger.asked` turns on
    `kind == "said"`, and it is what decides whether an effect is downgraded to speech. A `Spoke()`
    trigger with its own kind would make every spoken request *unprompted* — the permission model
    inverted by a spelling. A person speaking is a person asking."""
    heard = Heard("이거 뭐야", 1.1)
    trigger = Said(heard.text)
    assert trigger.asked, "a spoken request stopped counting as somebody asking"
    assert trigger.kind == "said"

    session = Session(eyes=OneWidget(), model=SilentModel(), hands=NoHands())
    exchange = session.think(trigger)
    assert exchange.said == "네"
    assert exchange.trigger.asked
    #: The interface reads exactly this to decide whether to speak the reply aloud. Nothing in the
    #: core has to learn a new field for it.
    assert session.history[0] == ("them", "이거 뭐야")


def test_an_UNPROMPTED_reply_is_TELLABLE_APART_so_the_room_can_stay_QUIET():
    """A `Watch` can make the agent speak with nobody having addressed it, and with a speaker
    attached that is noise in the owner's room. The downgrade is the interface's — `app.py` owes a
    `may_speak_unprompted` flag defaulting to False — and this is the field it turns on.

    **It is not inverted the way `may_act_unprompted` is.** The app sets that one True because it
    has a stronger gate downstream, a person pressing a button. There is no downstream gate on a
    sentence: you cannot un-hear one."""
    from argmind.core import Perceived                                  # noqa: PLC0415

    session = Session(eyes=OneWidget(), model=SilentModel(), hands=NoHands())
    exchange = session.think(Perceived("the build finished"))
    assert not exchange.trigger.asked
    assert session.history[0][0] == "noticed"
