# SPDX-License-Identifier: Apache-2.0
"""Windows again: SAPI behind `voice.Speaker`. **The voice is chosen per utterance, from the text.**

This is the mouth. `argmind/voice.py` is the protocol and says why voice is an interface rather than
a contract the core knows about; this file is the half that knows what SAPI is, and it is the only
place in the project that does.

## The failure that decides the whole design, measured here on 2026-09-13

**An English voice given Korean text does not speak it, and does not say so.** Synthesised into a
memory stream — no sound card, nothing audible — over the **125 unique Korean `say` strings** and
**57 unique English ones (>= 3 alphabetic words)** recovered from `measure/*.json`:

    voice   text                                    result
    David   pure Korean, n=79                       0 bytes, 79 of 79
    Zira    pure Korean, n=79                       0 bytes, 79 of 79
    David   Korean containing digits/Latin, n=46    audio, 46 of 46 -- of the digits ONLY
    Heami   Korean, n=125                           audio, 125 of 125
    Heami   English, n=57                           audio, 57 of 57

**The second row is worse than the first and it is the one nobody had seen.** Nothing is silent
there: `지우개는 22번 버튼입니다.` through David is 1.32 s of audio that a recogniser reads back as
`03, 04.` — the reply's *widget index*, read aloud, with the sentence around it gone. 46 of the 125
Korean replies (37%) contain a digit or a Latin word, so a one-voice build would speak a bare number
in more than a third of them. `argmind/voice.py` closes on the index leak costing 3.05 s of a voice
saying a number that means nothing; through the wrong voice the leak is *all that is left*.

**The reverse hole was measured because nobody had, and it is real but different.** Heami given
English is never silent — 57 of 57 produce audio — but it is Korean phonetics: `Click the Font combo
box` comes back from Whisper as `클릭 더 판트 컴보 박스`. Round-tripped through
`openai/whisper-large-v3-turbo`, **Heami's English is understood 10 of 30 against Zira's 25 of 30 on
the same 30 strings**. That is a recogniser standing in for a listener, so it is a *ceiling* on how
bad it is and not a measurement of intelligibility to a person. It is enough to settle the rule:
**any Hangul in the text picks the Korean voice, everything else stays on the default one.**

**The fallback is the system default voice and never an English one**, which `voice.py` requires.
On a machine with no Korean voice installed there is nothing honest left to do, so the text goes to
the default voice and `counts["silent_no_korean_voice"]` records it — the failure is then *counted*
rather than silent, which is the only part of it this file can fix. What the interface does with
that count is the app's decision, not this file's.

## The rest of the numbers, same day, same machine

**Synthesis into memory: median 3.6 ms, max 42.9 ms** (Heami, n=182 strings: the 125 Korean and the
57 English). Against the 0.61 s median model call that produced the text, starting to speak is free.

**Speech pace, on much larger denominators than the n=4 and n=2 in `voice.py`: Heami 5.69 characters
per second of Korean** (3,148 characters over 553.1 s of audio, 125 strings), **Heami 9.53 on
English** (3,160 over 331.7 s, 57), **Zira 11.92** (3,160 over 265.0 s), **David 11.76** (3,160 over
268.6 s), all at the default rate.

**`Speak(SPF_ASYNC)` returns in 40-84 ms on the worker thread and a purge costs 62-106 ms** (n=2
apartments, one pass each). Both are why neither runs on the caller's thread: `say()` and `stop()`
put a command on a queue and return, and the caller is the thread already waiting on the model.

**COM apartment: whichever one the worker can get, and it is not a free choice.** Importing
`comtypes` calls `CoInitialize()` for the importing thread, so a worker that imports it first is
already an STA and the MTA is then refused with `RPC_E_CHANGED_MODE` — which crashed the first
version of this file. Both were measured speaking, purging and speaking again, in the STA with no
message pump as well as in the MTA, and nothing here uses the SAPI events an unpumped STA would
break. `apartment` records which one a given run got.

**Measured with `Volume = 0`.** The owner is at this machine; every number above was taken with the
device open and the volume at zero, so the room stayed quiet.

## Two things this file decides and says out loud

**A second `say()` replaces the first rather than queueing behind it.** `Speaker.speaking()` is one
boolean and `stop()` is one purge, so a queue would be a second thing to interrupt and a stale
sentence could still arrive after the person had moved on. Each `say()` carries `SPF_PURGEBEFORESPEAK`.

**A worker that dies makes the agent mute with no error anywhere**, which is failure #1 of this
package in a new place: the loop catches, counts in `counts["failed"]`, keeps the last exception in
`last_error` and re-opens the voice once. Raising on the worker thread would take the mouth away
silently, which is exactly what the whole file is about.
"""

from __future__ import annotations

import queue
import re
import threading

__all__ = ["SapiSpeaker", "needs_korean", "SAPI_UNAVAILABLE"]

SAPI_UNAVAILABLE = (
    "SAPI could not be reached. It ships with Windows, so this usually means comtypes is missing "
    "(pip install comtypes) or this process has no audio session.")

#: Hangul syllables, conjoining jamo and compatibility jamo. **Script, not language**: what decides
#: the voice is whether SAPI has anything to pronounce, and that is a property of the characters.
_HANGUL = re.compile(r"[가-힣ᄀ-ᇿ㄰-㆏]")

#: SAPI's own constants, named rather than spelled as integers at the call site.
#: `SPF_ASYNC` returns before the sound does; `SPF_PURGEBEFORESPEAK` drops whatever is queued.
_SPF_ASYNC, _SPF_PURGE = 1, 2

#: `SAFT16kHz16BitMono`. Only `speaks()` uses it — the sound-card path takes the voice's own format.
_FORMAT_16K_16BIT_MONO = 18

#: The primary language of ko-KR as SAPI reports it from `GetAttribute("Language")`: an LCID in hex,
#: `412` here, `409` for both en-US voices. Matched on the whole attribute and on its parts, because
#: the attribute is documented to be a semicolon-separated list.
_KOREAN_LCID = "412"


def needs_korean(text: str) -> bool:
    """Does this text need the Korean voice? True if it contains any Hangul at all.

    **Any Hangul, not most of it**, and that asymmetry is measured: a Korean sentence through an
    English voice loses everything but its digits (46 of 46 such strings, and 0 of 20 survive a
    round trip), while English through Heami stays audible and merely accented (57 of 57 produce
    audio; 10 of 30 round-trip against Zira's 25 of 30). The two errors are not the same size, so
    the mixed case goes to the voice that can pronounce both alphabets.
    """
    return bool(_HANGUL.search(text or ""))


class SapiSpeaker:
    """A voice over SAPI. Satisfies `argmind.voice.Speaker`; everything else here is off the protocol.

    `voices()`, `speaks()` and `counts` are on the class and not on `Speaker`, for the reason
    `contract.py` gives for `attending_title()` and `at()`: a caller that wants them resolves them
    with `getattr` and has an honest fallback, and a future environment is not made to implement
    three methods for a caller that may not exist on that platform.

    Construction opens the device — it starts the worker, takes a COM apartment, creates the voice
    and reads the installed tokens, **254 ms measured** — and raises `RuntimeError(SAPI_UNAVAILABLE)`
    if any of that failed. Deliberate: a mouth that cannot speak should say so when it is built,
    not go quiet later.
    """

    def __init__(self, *, volume: int | None = None, rate: int | None = None,
                 poll: float = 0.02, timeout: float = 10.0) -> None:
        #: `volume=0` is what every number in this file's docstring was measured with.
        self.volume, self.rate, self.poll = volume, rate, float(poll)
        self.counts = {
            "said": 0, "stopped": 0, "replaced": 0, "dropped": 0, "empty": 0,
            "by_korean": 0, "by_default": 0, "silent_no_korean_voice": 0, "failed": 0,
        }
        self.last_error: BaseException | None = None
        #: "mta" or "sta"; see `_run`. Not a setting — a record of what the thread was given.
        self.apartment = ""
        self._q: queue.Queue = queue.Queue()
        self._speaking = threading.Event()
        self._ready = threading.Event()
        self._lock = threading.Lock()
        self._serial = 0
        self._cancel_through = 0
        self._error: BaseException | None = None
        self._voice = None
        self._default_token = None
        self._korean_token = None
        self._names: tuple[str, ...] = ()
        self._thread = threading.Thread(target=self._run, name="argmind-sapi", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout):
            raise RuntimeError(SAPI_UNAVAILABLE + " (the worker did not start)")
        if self._error is not None:
            raise RuntimeError(SAPI_UNAVAILABLE) from self._error

    # ---- Speaker ------------------------------------------------------------------------------

    def say(self, text: str) -> None:
        """Begin speaking, on the worker. Returns immediately — measured at a queue put.

        An empty or blank string is counted and dropped rather than handed to SAPI: `Intent.say` can
        be empty when the model chose to act without speaking, and an empty utterance would leave
        `speaking()` true until the worker noticed nothing was happening.
        """
        text = "" if text is None else str(text)
        if not text.strip():
            self._bump("empty")
            return
        if not self._thread.is_alive():
            #: Released already, or the worker never came up. Queueing here would leave
            #: `speaking()` true for ever, and the app polls it to decide whether to shut up.
            self._bump("dropped")
            return
        with self._lock:
            self._serial += 1
            serial = self._serial
            if self._speaking.is_set():
                self.counts["replaced"] += 1
            self.counts["said"] += 1
        #: Set here rather than on the worker: `say()` returns before the worker has started, and a
        #: caller that polls `speaking()` immediately must not be told the utterance already ended.
        self._speaking.set()
        self._q.put(("say", serial, text))

    def stop(self) -> None:
        """Cut the utterance off mid-word. **This is barge-in**, and it is one purge.

        `speaking()` is false the moment this returns, before the worker has run: the caller is
        about to open the microphone and the answer to "is the agent still talking" is already no.
        """
        with self._lock:
            self._cancel_through = self._serial
            self.counts["stopped"] += 1
        self._speaking.clear()
        self._q.put(("stop",))

    def speaking(self) -> bool:
        """Whether an utterance is still going. A flag read, so it can be polled at frame rate."""
        return self._speaking.is_set()

    def release(self) -> None:
        """Give the device back and stop the worker. Safe to call twice."""
        if self._thread.is_alive():
            self._speaking.clear()
            self._q.put(("release",))
            self._thread.join(timeout=5.0)

    # ---- off the protocol, resolved by the caller with getattr --------------------------------

    def voices(self) -> tuple[str, ...]:
        """The installed voices, in SAPI's order. `('Microsoft David Desktop - English (United
        States)', 'Microsoft Zira…', 'Microsoft Heami Desktop - Korean')` on this machine."""
        return self._names

    def has_korean(self) -> bool:
        """Whether a Korean voice was found. False means every Korean reply is 0 bytes or digits."""
        return self._korean_token is not None

    def speaks(self, text: str, timeout: float = 5.0) -> bool:
        """Would the voice this text selects actually produce audio? Synthesised into memory.

        **The self-check for the failure at the top of this file**, and the same shape as
        `WindowsHands.pointing_is_exact()`: it answers the question that has no other answer,
        because a voice with nothing to pronounce reports success and emits nothing.

        Unlike everything else here it **blocks** the caller until the worker answers — **20-52 ms
        measured through this call, n=3**, of which the synthesis itself is 3.6 ms median; the rest
        is a second `SpVoice` and a round trip through the worker's queue. So it belongs at start-up
        or in a diagnostic, and not on the path a person is waiting on.
        """
        if not (text or "").strip():
            return False
        answer: queue.Queue = queue.Queue(maxsize=1)
        self._q.put(("speaks", text, answer))
        try:
            return bool(answer.get(timeout=timeout))
        except queue.Empty:
            return False

    # ---- the worker: the only thread that touches COM ------------------------------------------

    def _bump(self, key: str, n: int = 1) -> None:
        with self._lock:
            self.counts[key] = self.counts.get(key, 0) + n

    def _run(self) -> None:
        try:
            import comtypes                                                  # noqa: PLC0415
            import comtypes.client                                           # noqa: PLC0415
        except Exception as exc:                                             # noqa: BLE001
            self._error = exc
            self._ready.set()
            return
        #: **The apartment is not ours to pick, and finding that out cost a crash.** Importing
        #: `comtypes` calls `CoInitialize()` for the *importing* thread, so a worker that imports it
        #: first is already an STA and asking for the MTA afterwards fails with
        #: `OSError [WinError -2147417850] Cannot change thread mode after it is set`
        #: (`RPC_E_CHANGED_MODE`). Whether that happens depends on something invisible from here —
        #: whether anything else in the process had imported `comtypes` before this thread started —
        #: so both outcomes are legal and `apartment` records which one this run got. Measured, one
        #: pass each: in the STA with no message pump and in the MTA, `Speak(SPF_ASYNC)`, a purge and
        #: a following `Speak` all completed. Nothing here uses SAPI events, which is the feature an
        #: unpumped STA would break.
        ours = False
        try:
            comtypes.CoInitializeEx(comtypes.COINIT_MULTITHREADED)
            self.apartment, ours = "mta", True
        except OSError:
            self.apartment = "sta"
        self._client = comtypes.client
        try:
            self._open()
        except Exception as exc:                                             # noqa: BLE001
            self._error = exc
            self._ready.set()
            if ours:
                comtypes.CoUninitialize()
            return
        self._ready.set()
        try:
            self._loop()
        finally:
            try:
                self._purge()
            except Exception:                                                # noqa: BLE001, S110
                pass
            self._voice = None
            #: Only the initialisation this thread added is undone. The one `comtypes` performed at
            #: import is its own, and unbalancing it would close COM under whatever else uses it.
            if ours:
                comtypes.CoUninitialize()

    def _open(self) -> None:
        voice = self._client.CreateObject("SAPI.SpVoice")
        if self.volume is not None:
            voice.Volume = int(self.volume)
        if self.rate is not None:
            voice.Rate = int(self.rate)
        tokens = voice.GetVoices()
        names, korean = [], None
        for i in range(tokens.Count):
            token = tokens.Item(i)
            names.append(token.GetDescription())
            #: The attribute is a hex LCID and is documented as a possible list, so `412` is looked
            #: for among the parts rather than compared to the whole string.
            langs = str(token.GetAttribute("Language") or "").lower().replace(",", ";").split(";")
            if korean is None and _KOREAN_LCID in [p.strip() for p in langs]:
                korean = token
        self._voice, self._names = voice, tuple(names)
        self._default_token, self._korean_token = voice.Voice, korean

    def _loop(self) -> None:
        while True:
            try:
                cmd = self._q.get(timeout=self.poll if self._speaking.is_set() else None)
            except queue.Empty:
                cmd = None
            try:
                if cmd is None:
                    #: Nothing to do but notice the utterance ended. SAPI is the only thing that
                    #: knows, and `WaitUntilDone(0)` asks without waiting.
                    if self._speaking.is_set() and self._finished():
                        self._speaking.clear()
                elif cmd[0] == "release":
                    return
                elif cmd[0] == "stop":
                    self._purge()
                elif cmd[0] == "say":
                    self._say(cmd[1], cmd[2])
                elif cmd[0] == "speaks":
                    cmd[2].put(self._synthesised(cmd[1]) > 0)
            except Exception as exc:                                         # noqa: BLE001
                #: Failure #1 of this package, one floor down: a window closing, a device
                #: disappearing or a voice being uninstalled must not take the mouth away in
                #: silence. Count it, keep it, re-open once, carry on.
                self.last_error = exc
                self._bump("failed")
                self._speaking.clear()
                if cmd is not None and cmd[0] == "speaks":
                    cmd[2].put(False)
                try:
                    self._open()
                except Exception as reopen:                                  # noqa: BLE001
                    self.last_error = reopen

    def _say(self, serial: int, text: str) -> None:
        with self._lock:
            stale = serial <= self._cancel_through
        if stale:
            #: `stop()` happened after this was queued. Barge-in must beat a sentence already in
            #: flight, or the talk key would start the agent talking.
            self._bump("dropped")
            return
        self._voice.Voice = self._token_for(text, spoken=True)
        self._voice.Speak(text, _SPF_ASYNC | _SPF_PURGE)

    def _token_for(self, text: str, *, spoken: bool = False):
        #: `spoken` keeps `speaks()` out of the routing counts. A probe that never reached the sound
        #: card must not read as an utterance the person heard — a counter that counts two different
        #: things is a counter nobody can quote.
        if not needs_korean(text):
            if spoken:
                self._bump("by_default")
            return self._default_token
        if self._korean_token is None:
            #: The honest fallback `voice.py` asks for: the system default, never an English voice
            #: chosen on purpose — and a count, because what the person hears is silence or a bare
            #: number and neither raises anything.
            if spoken:
                self._bump("silent_no_korean_voice")
                self._bump("by_default")
            return self._default_token
        if spoken:
            self._bump("by_korean")
        return self._korean_token

    def _synthesised(self, text: str) -> int:
        """Bytes this text produces through the voice it selects. No sound card is involved."""
        stream = self._client.CreateObject("SAPI.SpMemoryStream")
        stream.Format.Type = _FORMAT_16K_16BIT_MONO
        probe = self._client.CreateObject("SAPI.SpVoice")
        probe.Voice = self._token_for(text)
        probe.AudioOutputStream = stream
        probe.Speak(text, 0)
        return len(bytearray(stream.GetData()))

    def _finished(self) -> bool:
        return bool(self._voice.WaitUntilDone(0))

    def _purge(self) -> None:
        if self._voice is not None:
            self._voice.Speak("", _SPF_ASYNC | _SPF_PURGE)
