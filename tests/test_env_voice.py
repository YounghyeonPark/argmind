# SPDX-License-Identifier: Apache-2.0
"""The Windows voice: SAPI and Whisper, tested with **no microphone, no speaker, no GPU, no model**.

Three things are held here that a docstring cannot hold on its own:

* **nothing loads at import.** `python -m pytest` must stay free of GPU work and of a sound card,
  so torch, transformers, `sounddevice`, numpy and `comtypes` are all reached inside methods, and
  an AST walk says so rather than a convention;
* **the guards are pinned to the transcripts that produced them.** Every degenerate string below
  came out of `openai/whisper-large-v3-turbo` on this machine on 2026-09-13 — reproducibly, two
  identical runs each — and every clean one is a real round trip through SAPI. A guard tuned on
  nothing is a guess, and a guard whose evidence lives only in a comment is a guess a year later;
* **the negative result about `llm.py` is protected.** Its stop and this one are the same shape and
  its constants catch 1 of these 8. Somebody will one day propose collapsing the two functions;
  this is the measurement that answers them.

**Not one test here calls `hold(True)`**, deliberately: the owner is at this machine and the first
recording is arranged with them. `offer()` is the injection point instead, and it is the same path.
"""

from __future__ import annotations

import ast
import io
import pathlib
import sys
import time

import pytest

from argmind.env.sapi import SapiSpeaker, needs_korean
from argmind.env.whisper import (
    MODEL, REPEAT_FLOOR, REPEAT_PROBE, REPEAT_TIMES, WhisperListener, _repeating, _wordless,
    why_rejected)
from argmind.voice import Heard, Listener, Speaker

ENV = pathlib.Path(__file__).resolve().parents[1] / "argmind" / "env"

#: Verbatim from the runs recorded in `argmind/env/whisper.py`, and the two long ones are the same
#: strings rebuilt from their cycle: 63 copies cut mid-phrase, and 20 of the 40 that Whisper emitted.
DEGENERATE = [
    "Thank you. Thank you.",
    "... ... ... ... ... ... ...",
    "... ... ... ... ... ... ... ... ...",
    "うんご視聴ありがとうございましたご視聴ありがとうございましたご視聴ありがとうございました",
    "Tchau, tchau. Tchau, tchau. Tchau, tchau. Tchau, tchau.",
    "Tchau, tchau. Se inscreva no canal e ative o sininho para receber notificações de novos vídeos.",
    "곱창 눌러줘 " + "한글자막 by " * 20,
    "Thank you. " + "I'm going to make sure that " * 63 + "I'm going",
    #: A second run, on a different draw of the same noise. Here because what reproduces is that it
    #: degenerates, not what it says — the guard has to be a property of the text's shape.
    "Mm-hmm Thank you. Thank you. Thank you.",
    "속삭이냐 좋아 다음 영상에서 만나요. 한글자막 by 한효정",
]

#: The seven that are actually repetitions. The other three are hallucinations occurring once each;
#: they belong to the silence finding, and this guard does not claim them.
REPETITIONS = [t for t in DEGENERATE if t not in (
    "Thank you. Thank you.",
    "Tchau, tchau. Se inscreva no canal e ative o sininho para receber notificações de novos vídeos.",
    "속삭이냐 좋아 다음 영상에서 만나요. 한글자막 by 한효정",
)]

#: Real transcripts of real synthesised speech, from the round trip in `whisper.py`'s docstring.
CLEAN = [
    "지우개 눌러줘.", "빨간색을 선택할 수 있어요.", "네, 이미지를 저장하고 창을 닫을게요.",
    "화면에 20개의 버튼이 있어요", "공번창에서 특수문자를 선택한 후 3번 버튼을 클릭해 선택하고 4번 버튼으로 복사하세요.",
    "Okay, I'll select the red color.", "copy is number four",
    "I don't have a history of actions, you'll need to describe what you are trying to accomplish.",
    "BIOS version slash date is October 22nd, 2025.",
]

#: Hand-written and adversarial rather than recorded — **the weakest denominator in this file, and
#: labelled**. A person repeating themselves is the false positive that would cost a real turn.
REPETITIVE_HUMAN = [
    "네 네 네", "아니 아니 아니 아니", "지우개 눌러줘 지우개 눌러줘",
    "복사 눌러줘 복사 눌러줘 복사 눌러줘", "아니 그거 말고 아니 그거 말고 아니 그거 말고",
    "no no no no", "wait wait wait wait wait", "하나 둘 셋 하나 둘 셋 하나 둘 셋",
    "빨간색 빨간색 빨간색 빨간색 빨간색 빨간색",
]


def _top_level_imports(path):
    imported = set()
    for node in ast.parse(io.open(path, encoding="utf-8").read(), filename=str(path)).body:
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.add(node.module.split(".")[0])
    return imported


# ---- nothing loads at import -----------------------------------------------------------------

@pytest.mark.parametrize("name", ["sapi.py", "whisper.py"])
def test_a_voice_module_imports_NO_DEVICE_AND_NO_MODEL_at_module_level(name):
    """The suite runs with no GPU, no sound card and no network, and it must keep doing so.

    A top-level `import torch` costs seconds and a top-level `import comtypes` fails on a machine
    that has none — and both would be invisible until the day somebody ran the suite somewhere
    else. The heavy imports live inside the methods that need them."""
    heavy = {"torch", "transformers", "sounddevice", "comtypes", "numpy", "_ctypes"}
    assert not (_top_level_imports(ENV / name) & heavy), (
        f"{name} imports a device or a model at module level; the whole suite pays for it")


def test_importing_the_voice_modules_PULLS_IN_NOTHING_HEAVY():
    """The claim above, checked against what actually happened rather than against the source: both
    modules are imported at the top of this file, so if either dragged torch in it is here now."""
    for heavy in ("torch", "transformers", "sounddevice"):
        assert heavy not in sys.modules, (
            f"importing argmind.env voice modules loaded {heavy}; pytest now needs a GPU")


def test_a_LISTENER_and_a_SPEAKER_satisfy_the_protocols():
    """`object.__new__` rather than a constructor on purpose: constructing a `SapiSpeaker` opens the
    audio device, and a test that needs a speaker is a test that cannot run on a build machine."""
    assert isinstance(object.__new__(SapiSpeaker), Speaker)
    ear = WhisperListener(recognise=lambda samples, rate: "")
    try:
        assert isinstance(ear, Listener)
    finally:
        ear.release()


# ---- the guards, pinned to what produced them --------------------------------------------------

def test_the_REPETITION_GUARD_catches_every_loop_that_was_reproduced():
    """7 of 7, and the three it lets through are named because they are not repetitions.

    The one that matters is the third: a real command buried at 0 dB SNR came back as
    `곱창 눌러줘 한글자막 by …` — action-shaped, confident and forty copies long."""
    caught = [t for t in REPETITIONS if _repeating(t) or _wordless(t)]
    assert len(caught) == len(REPETITIONS) == 7, (
        f"{[t[:24] for t in REPETITIONS if t not in caught]} got through the guard")
    assert _repeating("곱창 눌러줘 " + "한글자막 by " * 20)
    #: Not repetitions: two copies is a person saying 네, 네, and the outro occurs once.
    assert not _repeating("Thank you. Thank you.")
    assert not _repeating(
        "Tchau, tchau. Se inscreva no canal e ative o sininho para receber notificações "
        "de novos vídeos.")


def test_the_guard_rejects_NONE_of_the_354_strings_that_must_survive():
    """The false-positive side, which is the one that costs a turn — and a rejected turn is silent,
    while a fabricated one at least appears in the transcript where the person can see it."""
    for text in CLEAN + REPETITIVE_HUMAN:
        assert not _repeating(text), f"the guard would have thrown away {text!r}"
        assert not _wordless(text)


def test_a_repeated_command_SHORTER_than_the_floor_survives_and_that_is_the_floor_s_job():
    """`아니 그거 말고` three times is 29 characters and a person could say it. It is under
    `REPEAT_FLOOR`, and that — not a margin in the counting — is what protects it. Said plainly
    because it is the guard's weakest edge: the same sentence four times over would be rejected."""
    short = "아니 그거 말고 아니 그거 말고 아니 그거 말고"
    assert len(short) < REPEAT_FLOOR and not _repeating(short)
    assert _repeating(short + " 아니 그거 말고 아니 그거 말고")


def test_the_guard_CONSTANTS_are_the_measured_ones():
    """A floor lowered to make something pass is the failure mode `test_suite_size.py` warns about,
    one level down. 32/12/3 came out of 8 degenerate transcripts against 354 that must survive."""
    assert (REPEAT_FLOOR, REPEAT_PROBE, REPEAT_TIMES) == (32, 10, 3)
    #: 8 catches the same 7 and takes 2 of the 193 recorded replies with it; 12 stays clean and
    #: misses one. The grid was measured, not picked.
    assert _repeating("Mm-hmm Thank you. Thank you. Thank you.")


def test_LLM_S_OWN_STOP_would_catch_ONE_of_these_TEN():
    """**The negative result, protected.** `_looping()` and `_repeating()` are the same idea, and
    the constants are not transferable: `LOOP_FLOOR = 256` was derived from a JSON envelope of about
    160 characters, while the longest clean transcript measured here is 63.

    It is not wrong — 0 of the clean transcripts — it is aimed at a different length of text. This
    is the number that answers "why not just call the one in `llm.py`"."""
    from argmind.core.llm import _looping                                    # noqa: PLC0415

    assert sum(_looping(t) for t in DEGENERATE) == 1
    assert sum(_repeating(t) or _wordless(t) for t in DEGENERATE) == 7
    assert not any(_looping(t) for t in CLEAN)


def test_a_transcript_of_PURE_PUNCTUATION_is_not_a_turn():
    """25 seconds of room tone produced `... ... ...` nine times over, twice. No threshold decides
    this one: there is no language in which that is something a person said."""
    assert _wordless("... ... ... ... ... ... ... ... ...")
    assert _wordless(".")
    assert not _wordless("네")
    assert not _wordless("22")


# ---- the pace cut-off is deliberately unset ------------------------------------------------------

def test_PACE_rejects_NOTHING_until_somebody_measures_a_person():
    """`voice.py` says no threshold is named because the only pace on this machine is a
    synthesiser's. The measurement here made that worse rather than better: silence transcribes to
    `Thank you.` **120 times out of 120**, and over a 2 s press that is 5.0 characters a second
    against Heami's measured 5.69 for real Korean. Pace cannot separate them.

    So the default rejects nothing, and this test is what fails if a cut-off is quietly added."""
    fabrication = Heard("Thank you.", 0.5)
    slow_fabrication = Heard("Thank you.", 2.0)
    real = Heard("빨간색을 선택할 수 있어요.", 2.87)
    assert why_rejected(fabrication.text, fabrication.seconds) == ""
    assert real.pace == pytest.approx(5.23, abs=0.01)
    assert slow_fabrication.pace == pytest.approx(5.0)
    #: 5.0 against 5.23 for that sentence and 5.69 for the whole Korean corpus. Any cut-off that
    #: takes the fabrication takes real Korean with it, which is why the audio to set it has to
    #: come from a person rather than from a synthesiser.
    assert abs(slow_fabrication.pace - real.pace) < 0.3


def test_a_pace_limit_WORKS_when_a_measurement_finally_sets_one():
    """Unset is not unimplemented. The mechanism is here and tested so that the commit that brings
    the recorded audio brings only a number."""
    assert why_rejected("Thank you.", 0.5, pace_limit=12.0) == "pace"
    assert why_rejected("지우개 눌러줘", 2.0, pace_limit=12.0) == ""


def test_WHY_a_transcript_was_rejected_is_reported_and_not_merely_THAT():
    """A single `rejected` total cannot tell a recogniser problem from a threshold problem."""
    assert why_rejected("") == "empty"
    assert why_rejected("   ") == "empty"
    assert why_rejected("...") == "wordless"
    assert why_rejected("곱창 눌러줘 " + "한글자막 by " * 20) == "repeated"
    assert why_rejected("지우개 눌러줘", 1.4) == ""


# ---- the listener, driven with no device ----------------------------------------------------------

def _wait_for(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


def test_an_utterance_arrives_as_HEARD_and_the_caller_never_waits_for_it():
    """The whole path with the recogniser injected: audio in on one thread, `Heard` out on another.

    `heard()` returning `None` first is not a failure, it is the contract — the caller is the thread
    already waiting on the model and recognition takes 234 ms."""
    ear = WhisperListener(recognise=lambda samples, rate: " 지우개 눌러줘. ")
    try:
        assert ear.heard() is None
        ear.offer([0.0] * 16000, 16000, at=1_000_000.0)
        assert _wait_for(lambda: ear.counts["heard"] == 1)
        said = ear.heard()
        assert said.text == "지우개 눌러줘."
        assert said.seconds == pytest.approx(1.0)
        assert said.at == 1_000_000.0
        assert ear.heard() is None
    finally:
        ear.release()


def test_a_FABRICATION_is_rejected_rather_than_returned_and_the_count_says_which():
    """`voice.py`: near-silence must produce `None`, not `감사합니다`. A rejection is not an error,
    so nothing raises and nothing arrives — but a silent rejection with no counter is how a build
    goes deaf without anybody noticing."""
    ear = WhisperListener(recognise=lambda samples, rate: "곱창 눌러줘 " + "한글자막 by " * 20)
    try:
        ear.offer([0.0] * 32000)
        assert _wait_for(lambda: ear.counts["rejected_repeated"] == 1)
        assert ear.heard() is None
        assert ear.counts["heard"] == 0
    finally:
        ear.release()


def test_EMPTY_AUDIO_never_reaches_the_recogniser():
    """A tapped key produces no samples. Whisper given none of them still answers `Thank you.`"""
    calls = []
    ear = WhisperListener(recognise=lambda samples, rate: calls.append(rate) or "Thank you.")
    try:
        ear.offer([])
        assert _wait_for(lambda: ear.counts["empty_capture"] == 1)
        assert calls == []
        assert ear.heard() is None
    finally:
        ear.release()


def test_a_RECOGNISER_THAT_RAISES_leaves_the_agent_deaf_but_ALIVE():
    """Failure #1 of this package, one floor down: an exception on the worker thread would kill the
    thread, and every later utterance would vanish with nothing raised anywhere a person can see."""
    def broken(samples, rate):
        raise RuntimeError("the device went away")

    ear = WhisperListener(recognise=broken)
    try:
        ear.offer([0.0] * 1600)
        assert _wait_for(lambda: ear.counts["failed"] == 1)
        assert isinstance(ear.last_error, RuntimeError)
        #: still listening
        ear._recognise = lambda samples, rate: "네"
        ear.offer([0.0] * 1600)
        assert _wait_for(lambda: ear.counts["heard"] == 1)
        assert ear.heard().text == "네"
    finally:
        ear.release()


def test_RELEASING_TWICE_is_not_an_error_and_hold_False_alone_does_nothing():
    """`release()` is called from a window closing, which can happen twice; `hold(False)` arrives on
    a key coming up that may have gone down before the listener existed.

    **`hold(True)` is not called here or anywhere in this suite** — it opens the microphone."""
    ear = WhisperListener(recognise=lambda samples, rate: "")
    ear.hold(False)
    ear.release()
    ear.release()
    assert ear.counts["captured"] == 0 and ear.counts["held"] == 0


def test_the_MODEL_is_named_but_never_loaded():
    """The default is the one every number was measured on, and naming it must not fetch it."""
    assert MODEL == "openai/whisper-large-v3-turbo"
    ear = WhisperListener(recognise=lambda samples, rate: "")
    try:
        ear.warm()
        ear.offer([0.0] * 1600)
        assert _wait_for(lambda: ear.counts["heard"] + ear.counts["rejected_empty"] == 1)
        assert ear.load_seconds is None, "an injected recogniser still loaded Whisper"
        assert "torch" not in sys.modules
    finally:
        ear.release()


# ---- the voice is chosen from the text ------------------------------------------------------------

@pytest.mark.parametrize("text, korean", [
    ("지우개는 22번 버튼입니다.", True),
    ("빨간색을 선택할 수 있어요.", True),
    ("Character Grid이 열려 있어요.", True),
    ("Okay, I'll select the red colour.", False),
    ("BIOS Version/Date is 10/22/2025", False),
    ("22", False),
    ("", False),
    ("ㄱㄴㄷ", True),
])
def test_ANY_HANGUL_picks_the_korean_voice(text, korean):
    """**The measured asymmetry, and the reason this is `any` and not `most`.** An English voice
    given pure Korean emits 0 bytes (79 of 79 strings) and given Korean with a number in it emits
    the number alone (46 of 46, and 0 of 20 survive a round trip: `지우개는 22번 버튼입니다.` comes back
    as `03, 04.`). Heami given English is merely accented — 57 of 57 audible, understood 10 of 30
    against Zira's 25 of 30 — so the mixed case costs far less on the Korean voice."""
    assert needs_korean(text) is korean


def test_the_speaker_falls_back_to_the_DEFAULT_voice_and_never_to_an_ENGLISH_one():
    """What `voice.py` requires, read off the source because constructing the class opens the audio
    device. There is exactly one fallback token and it is the system default; the count is what
    makes the resulting silence visible instead of silent."""
    source = io.open(ENV / "sapi.py", encoding="utf-8").read()
    tree = ast.parse(source)
    chooser = next(n for n in ast.walk(tree)
                   if isinstance(n, ast.FunctionDef) and n.name == "_token_for")
    returned = {ast.unparse(n.value) for n in ast.walk(chooser) if isinstance(n, ast.Return)}
    assert returned == {"self._default_token", "self._korean_token"}, (
        f"_token_for can return {sorted(returned)}; the only fallback voice.py allows is the "
        "system default")
    assert 'self._bump("silent_no_korean_voice")' in source, (
        "a machine with no Korean voice speaks silence or a bare number, and nothing raises; the "
        "only honest thing left is to count it")
