# SPDX-License-Identifier: Apache-2.0
"""Hearing and speaking. **An interface, not an environment — and `argmind/core/` does not change.**

    python -m argmind app --voice        (once app.py and env/ have built their halves)

This module is the whole seam: two protocols, one value type, no third-party import, and no import
of `argmind.core` at all. Everything that needs a sound card, a GPU or SAPI lives behind it in
`argmind/env/`; everything that wires it to a conversation lives in `argmind/app.py`.

## The finding, stated first because it is the answer

**The core already speaks text and voice only converts audio to text and back, on either side of it.
Not one line of `argmind/core/` changes, and no protocol is added beside `Eyes` and `Hands`.**

That is a conclusion, not a dodge, and here is the test that produced it. `Eyes` and `Hands` are in
`core/contract.py` **because the core calls them**: `Session.observe()` calls
`eyes.timed_foreground()`, `Session.perform()` calls `hands.click()`. Grep `core/` for a line that
would call an `Ears` or a `Mouth` and there is none, because there cannot be one —
`Session.exchange()` takes a `str` and `Intent.say` is a `str`, and both ends of a voice channel
attach *outside* those. A protocol in `core/` that nothing in `core/` calls is precisely what
`contract.py` refuses for `attending_title()`, `at()` and `pointing_is_exact()`: a method every
future environment must implement to satisfy a caller that may not exist on that platform.

**`Eyes` and `Hands` are how the agent touches the machine's other applications. A microphone and a
speaker are how the agent touches the person.** Those are different surfaces, and the reason voice
*looks* like it belongs beside `Eyes` and `Hands` is that all four are I/O. Being I/O is not the
criterion; being called by the core is.

**Why not in `core/` even as a platform-free protocol.** It would be the first file in `core/` that
no core module imports, and it would quietly redefine `core/` from *the agent* to *the bin for
things with no platform in them*. `test_the_core_NAMES_NO_PLATFORM` would then be scanning a file
that has nothing to do with the claim it is making. **Why not inside `app.py`.** Importing `app.py`
imports `tkinter`, so an `env` module implementing `Speaker` would need a GUI toolkit to reach its
own base class — and `app.py` is owned by one agent while this seam has three sides.

**A robot pulls the other way and it does not win, for a reason worth recording.** Voice would be a
robot's primary channel, so "voice is interface" sounds like it makes voice unavailable there. It
does not: it makes the robot's *interface* a voice interface, in place of a window. What a robot
lacks is a deterministic `Eyes` — the question already open in `contract.py` — and, separately, the
thing the next section is about.

## The second rule, which the app already obeys and nobody had written down

    The deterministic side says WHERE, and WHETHER. The probabilistic side says WHAT.

WHERE is the measured one: a vision model took 24-82 s for a pixel and never answered on a dense
screen, UI Automation answers in 41-266 ms. WHETHER has been true in this repository since the app
was built and has only ever been justified in passing: **`Session.perform` is reached from exactly
one place, a person pressing 실행**, and a button press is an event Windows attributes to a physical
device. No model interprets it. `tests/test_app.py` asserts the call site; nothing states the
property the call site exists to protect.

Voice is what forces it to be written, because voice offers an obvious feature that breaks it.

**Spoken confirmation does not exist. Saying *"응"* out loud cannot run an action.** A transcript is
a recogniser's opinion about a waveform; making it the gate on an irreversible click replaces a
device event with a model output in the one slot this project has decided must stay exact. It is the
`Eyes`-for-a-robot failure in a new coat: a probabilistic component filling a deterministic slot, at
the seam, quietly. The cost is one extra motion per action and the person is already at the keyboard
that the action is about to drive.

So the division is one line: **voice may START a thought and may REPORT one. It may not FINISH one.**
`Heard` becomes `Said(text)` like anything typed; consent stays with the button.

**What the tests should assert afterwards** — the second one is new and is the one that matters,
because the first is defeated by relocation rather than by edit:

* `argmind/app.py` still contains exactly one call to `.perform(` — the existing assertion;
* **no module outside `argmind/core/` other than `app.py` and `__main__.py` calls `perform`,
  `think` or `exchange` on a session**, and this module names none of the three. `think()` and
  `exchange()` reach `perform` through `loop.py`, so a headless voice front end written the
  natural way — `session.exchange(heard.text)`, which is what `python -m argmind talk` does — would
  act on a transcript with no button anywhere. Counting `.perform(` alone would not see it.

`tests/test_voice.py` holds both, by walking the AST, for the reason `test_core.py` gives: a rule
that lives in a docstring is a rule that gets quietly undone.

**The corollary is that voice requires the window, and it is a real limit rather than a tidy one.**
A voice-only front end has no button, so it would have to either accept spoken confirmation or run
with standing permission — the two things refused above. **This is the robot failing to fit for a
second, independent reason**: it has no deterministic source of WHERE *and* no deterministic source
of WHETHER. Same root, no accessibility layer and no button in a room, and it is more evidence that
the robot wants a different core rather than a bad `Eyes`.

## The type at the seam, and the field that is deliberately missing

`Heard` carries the transcript, how long the person spoke, and when they stopped. **It does not
carry a confidence.** Whisper does not produce a calibrated one, and a field that is `None` on the
only recogniser in sight is a field that invites a threshold nobody can set — this project's
standard is that a number carries its denominator and its conditions, and a confidence with neither
is worse than no confidence. If a recogniser that publishes one is adopted, the field arrives in the
same commit as the measurement that sets the threshold, and not before.

**`seconds` is here because it is the one cheap signal that a transcript is invented.** Whisper
hallucinates fixed phrases on near-silence, so a 0.3 s utterance transcribing to forty characters is
not a bad transcription, it is a fabrication — and the ratio is how you see it. `pace` computes that
ratio in one place for the implementation that rejects it, the interface that shows it and the
harness that counts it, exactly as `Widget.centre` is the one place a name becomes a coordinate.

**No language field.** Its only consumer would be a sentence in the prompt, and five sentences have
been added to `SYSTEM` over this project's life with all five sliding the model between acting and
refusing without moving the line. If a measurement wants the detected language it asks the
recogniser through `getattr`, which is how `keys()`, `at()` and `attending_title()` are already
resolved.

## Push-to-talk, and what barge-in therefore means

**Hold a key to talk. Always-listening is not built and is not merely deferred for politeness.**
Push-to-talk is not the timid option here; it is the one that makes three separate problems
disappear at once, and each would otherwise need its own mechanism:

* **the floor.** `Watch` has `MIN_GAP = 3.0` because perception changes 13 seconds in 14 and an
  unfloored watch is a model that never stops. An open microphone has the same shape and no such
  floor exists for it. Under push-to-talk the floor is a physical key, which is stronger than a
  timer because it is the person's intent rather than an estimate of it;
* **self-hearing.** A speaker and an open microphone in one room means the agent transcribes itself.
  Under push-to-talk the microphone is shut while the agent talks, so it cannot happen at all;
* **who was being addressed.** An always-open microphone in a room with a phone call in it produces
  turns nobody meant to take. Nothing here can tell those apart.

**Barge-in is therefore one line: pressing the talk key stops the mouth.** The utterance is cut mid
word and is not resumed — the agent's speech is a courtesy and the person's is the input. No voice
activity detection, no echo cancellation, no half-duplex state machine; those are what always-
listening would cost and they are the size of the thing being avoided.

**Barge-in during *thinking* is not implemented and `llm.py` gains no cancellation hook.** The median
call is 0.61 s, which is not long enough to interrupt on purpose; the pathological call is already
bounded by the two shape stops added on 2026-09-11; and abandoning the read would not stop the
server generating, so it would buy the person's patience and not the machine's time. **What would
refute this is a count, and it is nearly free**: `app.py` already ignores input while `busy`, so it
can tally the talk-key presses it swallowed. A tally that stays near zero says the lockout is
invisible; one that does not is the first evidence for cancellation.

## A bad transcript is another turn to repair, and needs no channel of its own

The conversation side already handles *"아니 그거 말고"* — 3 of 4 corrections land, n=4, which cannot
reach p<0.05 under any repair and is the honest size of that claim. A mis-transcription produces at
worst a wrong *proposal*, which is the failure mode the button already exists for and which arrives
at the existing rate. The one obligation is that **what was heard is written into the transcript
verbatim**, as the person's own line: if it says 지우개 굴러 when they said 지우개 눌러, the odd answer
explains itself and the correction has something to point at. A transcript the person cannot see is
a misunderstanding with no evidence in the room.

**What is unmeasured, and it is the claim to attack.** A transcript is out-of-distribution text
beside a typed line — no punctuation, run-together clauses, a wrong homophone. Whether that raises
the spurious-action rate is not known. `measure/talking.py` reads 31 of 34 conversational turns
producing no action; the refuting measurement is that harness re-run over transcripts of the same
prompts spoken aloud, and a drop there is a finding about voice and not about the prompt.

## The numbers, measured on this machine 2026-09-13

**Speaking costs nothing and is already installed.** `Microsoft Heami Desktop - Korean` is present
beside David and Zira, and SAPI is reachable through `comtypes`, which is already a dependency —
probed, not assumed. Synthesised into a memory stream rather than to the sound card, so synthesis
time and audio duration are separated:

    reply                    chars   audio      synthesis
    지우개는 22번 버튼입니다.        15    3.05 s      24.0 ms
    네, 이미지를 저장하고 창을…      21    3.92 s      11.3 ms
    빨간색을 선택할 수 있어요.       15    2.87 s       6.0 ms
    화면에 20개의 버튼이 있어요…     30    5.17 s       9.7 ms
    Okay, I'll select the…       52    4.00 s      43.2 ms
    Total Physical Memory is…    75    7.38 s      20.3 ms

**Synthesis is 6-43 ms, 179-294x realtime, at 22.05 kHz 16-bit mono.** Measured, n=6 utterances, one
pass, this machine. Against a 0.61 s median model call it is two orders of magnitude below the
thing the person is already waiting for, so **time to first sound is the model plus ~25 ms** and
output voice does not enter the latency budget at all.

**Speech pace: 5.4 characters per second Korean** (Heami, 81 chars over 15.01 s, n=4) and **11.2
English** (Zira, 127 chars over 11.38 s, n=2). Small n, one pass, default rate.

**Replies are short, and that is re-derivable.** Every non-empty `say` recorded in `measure/*.json`
— **660 of them**, from the `saying`, `reoffer` and `repair` harnesses on fixed fixtures with
`qwen3:8b` thinking off, so a claim about those harnesses and not about real use — runs to a
**median of 17 characters, p75 24, p90 35, p99 52, max 75**.

**Computed from those two: a median reply is ~3.1 s of Korean speech and a p90 reply ~6.5 s.**

**That inverts the shape of a turn, and it is the one thing here arguing with a settled decision.**
The round trip this project economises is 0.61 s; the utterance built from it is five times longer.
It does **not** license a second model call — time to first sound would double, and silence before
the first word is what a conversation is judged on.

**It does reopen streaming, and the first version of this paragraph said otherwise on a wrong
denominator.** That version compared ~250 ms against the 3.1 s *utterance* and called it 8%. The
utterance is not the wait: the person hears it. The wait is the silence between their last word and
the first sound, which is the model call plus synthesis — measured, `say` completes at 232 ms of a
478 ms reply and synthesis costs ~25 ms, so **speaking on completion of `say` rather than of the
reply takes that silence from ~503 ms to ~257 ms**, roughly halving the only interval a listener
experiences as a gap. A rate carries its denominator, and this one had the wrong one. Two things
also moved in streaming's favour since it was declined: shipped is now **`say` first**, so the
string to extract from partial JSON is the one that arrives first, and there is now a p90 reply of
6.5 s whose tail can be synthesised while its head is being heard. It is still not free — partial
JSON, and a `say` that turns out to be wrong cannot be unsaid — so this is a reopened question and
not a decision. The cheaper thing to try first remains an audible acknowledgement at the moment the
key is released, which costs no extraction and no model change.

**The number nobody has is transcription.** `sounddevice`, `faster-whisper` and `ffmpeg` are not
installed and nothing here has measured them. The end-to-end spoken turn is

    the person talks (~2 s) + end-of-speech + TRANSCRIBE + model 0.61 s + synthesis 0.03 s

so transcription is the only unbounded term and it decides whether spoken input is usable at all.
**That is `environment`'s first debt and it should be paid before anything else is written**: a
single measured latency for the model actually chosen, on real utterances of the lengths above,
reported with its denominator.

## What each side owes

**`argmind/env/` — the implementations, and the platform is named only there.** A `Listener` over
`sounddevice` plus Whisper on the 4090, and a `Speaker` over SAPI, selecting Heami by name and
falling back to the default voice rather than to an English one. Three obligations that are not
obvious from the protocol:

* **`heard()` must not block.** Transcription is hundreds of milliseconds on a GPU and the caller is
  the agent thread, which is already the thread that waits for the model. Capture and recognition
  run on the implementation's own thread and `heard()` reports what has finished;
* **reject the fabrication rather than return it.** Near-silence must produce `None`, not
  *"감사합니다"*. The threshold is a `pace` above which a transcript is not speech, and it is
  **derived from recorded audio, not guessed** — the 5.4 chars/s above is a synthesiser's pace and
  is the wrong denominator for a person;
* **each capability owes its measurement**, which is that team's standing rule and applies here to
  transcription latency, to the hallucination rate on silence, and to whether Korean and English
  round-trip at the rates `docs/findings/interaction.md` records for text.

**`argmind/app.py` — the wiring, and it is two attachments rather than a new front end.** The window
already has an input path (entry box → `Agent.ask`) and an output path (the `say` event → the
transcript). Voice adds a second source into the first and a second sink from the second. Nothing
else moves, and in particular the proposal rows and the 실행 button are untouched, which is what
keeps the permission argument intact. What it owes:

* **a `may_speak_unprompted` flag, defaulting to False**, and **it is not inverted the way
  `may_act_unprompted` is.** The app sets that one to True because it has a stronger gate
  downstream — a person pressing a button. There is no downstream gate on a sentence: you cannot
  un-hear one. So a `Watch` firing with a speaker attached writes to the transcript and stays
  silent unless the flag is on, which is the same downgrade `loop.py` already applies to an
  unprompted effect, one level out;
* **if that flag is turned on, the watches need a larger `Watch.gap`.** `MIN_GAP = 3.0` was derived
  for a line of text appearing in a window, and an agent narrating every alt-tab out loud is a
  different thing. **The number is unmeasured and should not be invented**: the measurement is how
  often a watch fires during an ordinary working hour, and it needs no model at all — poll the
  watches, count the firings, never call `converse()`;
* **write what was heard into the transcript as the person's line**, for the repair reason above;
* **keep the voice injectable**, the way `Agent.build` already injects eyes, hands and model, so
  that `tests/test_app.py` keeps running with no microphone and no speaker.

**`measure/` — three instruments, in this order.** Transcription latency and its hallucination rate
on silence (env's debt, above); `talking.py` re-run over spoken transcripts, which is the refutation
test for the whole design; and the watch-firing rate per working hour, which is what would let
unprompted speech be switched on with a number behind it.

**One existing cost gets worse and it is worth saying out loud.** The index leak — 5 of 32 replies
carrying an internal widget number, *"지우개는 22번 버튼입니다"* — was accepted on 2026-09-10 as the
price of rule-following, 36 of 36 probes against 14. Read on a screen it is noise. **Spoken, it is
3.05 s of a voice saying a number that means nothing to the listener** — measured above, it is the
first row of the table. That does not reverse the trade, which was the owner's and was made with
both columns in front of them; it raises what fixing the leak is worth.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

__all__ = ["Heard", "Listener", "Speaker"]


class Heard:
    """One finished utterance, as text.

    `seconds` is how long the person spoke and `at` is when they stopped, from the same clock
    `Trigger` uses. **There is no confidence field and the module docstring says why** — the
    recogniser in sight does not produce a calibrated one, and an always-`None` field invites a
    threshold nobody can set.
    """

    __slots__ = ("text", "seconds", "at")

    def __init__(self, text: str, seconds: float = 0.0, at: float = 0.0) -> None:
        self.text = str(text)
        self.seconds = float(seconds)
        self.at = float(at)

    @property
    def pace(self) -> float:
        """Characters of transcript per second of audio, `inf` if the audio had no length.

        **The one place this ratio is computed**, for the reason `Widget.centre` is the one place a
        name becomes a coordinate: the implementation rejects a fabrication with it, the interface
        may show it, and a harness counts it, and three copies would be three chances to disagree
        about what "too fast to be speech" means.

        It exists because near-silence does not produce silence from Whisper, it produces a fixed
        phrase — *"감사합니다"*, *"Thank you."* — and a plausible sentence over a third of a second of
        room tone is the one junk transcript that reads as a real turn. **No threshold is named
        here.** The only pace measured on this machine is a synthesiser's (5.4 characters a second
        for Korean, 11.2 for English), which is the wrong denominator for a person, and a guessed
        cut-off that silently drops real turns is worse than none.
        """
        return float("inf") if self.seconds <= 0 else len(self.text) / self.seconds

    def __repr__(self) -> str:
        return f"Heard({self.text[:32]!r}, {self.seconds:.1f}s)"


@runtime_checkable
class Listener(Protocol):
    """A microphone that produces finished utterances. **It never reaches a `Session`.**

    Three methods, and the shape is chosen so that push-to-talk today and voice-activity detection
    later are the *same* protocol: an implementation that decides for itself when speech begins
    simply ignores `hold` and returns utterances from `heard()`. That is the property to want here —
    the thing most likely to change is when capture starts, and it is the thing not encoded.
    """

    def hold(self, down: bool) -> None:
        """Capture while `down`. Called on the talk key going down and coming up.

        **Push-to-talk, and it is the load-bearing choice rather than the cautious one**: the key is
        the floor an open microphone has no equivalent of, it makes self-hearing impossible because
        the microphone is shut while the agent talks, and it is the only thing here that can tell a
        turn addressed to the agent from a phone call in the same room.
        """
        ...

    def heard(self) -> Heard | None:
        """A finished utterance, or `None` — and it **must not block**.

        `None` is the ordinary answer and means nothing has finished: no speech, or speech still
        being recognised. It is also the answer for a transcript the implementation has judged to be
        a fabrication, which is a rejection and not an error, the same way `Watch.look()` returns
        `None` rather than raising.

        Transcription costs hundreds of milliseconds on a GPU and the caller is the thread that
        already waits for the model, so recognition belongs on the implementation's own thread and
        this reports what has finished. A blocking version would put an unmeasured wait on the one
        thread the person is watching.
        """
        ...

    def release(self) -> None:
        """Give the device back. Named as `llm.Local.release()` is, because it is the same job."""
        ...


@runtime_checkable
class Speaker(Protocol):
    """A voice. Says a string and can be cut off mid-word.

    **`say()` returns immediately and that is forced by barge-in**: a blocking version cannot be
    interrupted, and a median reply measured at ~3.1 s of Korean speech against a 0.61 s model call
    is long enough that being unable to stop it is the difference between a conversation and a
    lecture.
    """

    def say(self, text: str) -> None:
        """Begin speaking. Returns at once; use `speaking()` to find out whether it has finished.

        Synthesis measured at **6-43 ms for 15-75 character replies** on this machine, 179-294x
        realtime, so the cost of starting is negligible beside the model call that produced the
        text — output voice does not enter the latency budget.
        """
        ...

    def stop(self) -> None:
        """Cut the current utterance off mid-word and do not resume it.

        **This is what barge-in is.** The talk key going down calls it: the agent's speech is a
        courtesy and the person's is the input, so there is no arbitration and no queue. Resuming
        would be worse than dropping — the person interrupted because the answer had stopped being
        the one they wanted.
        """
        ...

    def speaking(self) -> bool:
        """Whether an utterance is still being spoken. Cheap; it is polled."""
        ...

    def release(self) -> None:
        """Give the device back."""
        ...
