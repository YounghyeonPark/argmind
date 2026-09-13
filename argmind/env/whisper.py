# SPDX-License-Identifier: Apache-2.0
"""Whisper behind `voice.Listener`: capture on a key, recognise on a thread, reject the fabrications.

The ear. `argmind/voice.py` is the protocol; this is the half that knows what a sound card and a GPU
are. **Nothing here loads at import** — torch, transformers, `sounddevice` and numpy are all reached
inside methods, so `python -m pytest` still does no GPU work and needs no device.

## Transcription, which was the number nobody had

`openai/whisper-large-v3-turbo`, transformers 5.14.1, torch 2.13.0+cu126, fp16, RTX 4090 Laptop,
audio synthesised by SAPI rather than recorded — **the microphone was never opened, and that is a
condition of every number in this file, not an aside.**

    n=140 utterances, median audio 2.77 s      median 234 ms, p90 300 ms, max 645 ms
    the first call after loading                        645 ms
    model load, cold process, n=3                      4.31-4.32 s
    weights resident (torch.cuda.memory_allocated)      1,544 MiB

**It scales with the transcript, not with the audio**, which is why a long hold is not the expensive
case: 184 ms median under 10 characters (n=27), 227 ms at 10-19 (n=49), 272 ms at 30-39 (n=14),
288 ms at 60-69 (n=6) — and a 30 s clip of noise transcribing to two words still costs ~200 ms.

Against the spoken turn `voice.py` lays out — talk, end of speech, TRANSCRIBE, model 0.61 s,
synthesis 0.03 s — **transcription is about a third of the machine's part of the wait**, and the
answer to whether spoken input is usable at all is yes.

## Two failures this file exists to survive

**1. Silence is not transcribed as silence. It is transcribed as a sentence, every single time.**
120 clips of generated silence and generated room tone — digital zeros and Gaussian noise at -60,
-50 and -40 dBFS, at 0.5, 1.0 and 2.0 s, ten of each — produced **120 non-empty transcripts out of
120**, almost all of them `Thank you.`, with `Gracias.` and `Okay.` among them. A second draw of the
same 120 through the finished class gave **119 of 120 surviving every guard in this file** (117
`Thank you.`, one `Okay.`, one Russian subtitle credit, one rejected for having no letters in it).
**Digital zeros hallucinate as readily as room tone**, 30 of 30, so this is not about the noise
floor of a microphone.

That is worse than the hazard `voice.py` describes and it lands on the guard it specifies. **`pace`
does not separate this case.** `Thank you.` is 10 characters: over 0.5 s that is 20 characters a
second and over 2.0 s it is 5.0, while Korean speech through Heami measures **5.69 characters a
second** over 3,148 characters. So a pace cut-off catches a fabrication on a short press and cannot
catch one on a two-second press without also rejecting real Korean. **The cut-off therefore stays
unset — `pace_limit=None` rejects nothing** — and this paragraph is the reason rather than caution.

**The measurement that would set it**: the owner holding the talk key and speaking, so that a real
distribution of characters per second exists for a person on this microphone; and the same key held
in a quiet room with nothing said, so that the two distributions can be looked at side by side. The
audio for both is arranged with the owner and does not exist yet. Until it does, the honest defence
is push-to-talk itself: a fabrication needs somebody to have held the key.

**2. It degenerates on hard input, the same way `llm.py` does, and it was reproduced here.**

    45 s of -40 dBFS noise, long-form    "Thank you. I'm going to make sure that…"   1,784 chars, 3.5 s
    "복사 눌러줘" at 0 dB SNR + 60 s tone  "곱창 눌러줘 한글자막 by 한글자막 by…"          495 chars, 3.4 s
    91 s of a word plus room tone        "うんご視聴ありがとうございました" x3                44 chars
    120 s of noise                       "Tchau, tchau." x4                            55 chars
    25 s of noise, short-form            "... ... ..." x9                              35 chars

The second row is the one that matters: **action-shaped, confident and repeated**, from a real
command buried in noise. It is `_looping()`'s failure in another process, and `llm.py`'s remark that
a wrongly stopped turn produces what the turn already produced applies unchanged.

**What reproduces is the shape and not the sentence, and the first version of this file said
otherwise.** Each row above was run twice and came back byte-identical, which says the decode is
deterministic — it does not say the phenomenon repeats, because both runs were the *same clip*. Run
again on a different draw of the same noise, the same four conditions gave
`Thank you. Thank you.`, `속삭이냐 좋아 다음 영상에서 만나요. 한글자막 by 한효정`,
`Mm-hmm Thank you. Thank you. Thank you.` and one rejection. **Long low-information audio
degenerates; what it degenerates into is a coin toss**, which is the reason the guard is a property
of the text's structure rather than a list of phrases.

**What did NOT reproduce, said because it is a withdrawal.** The case this file was told to defend
against — a bare `응` at 20 dB SNR coming back as `안녕히계세요.` forty-two times — did not happen
once here. Across 100 transcriptions of 12 utterances at SNR 30, 20, 10 and 0 dB of white noise the
longest transcript was 21 characters. Every long loop above needed **more than 30 seconds of audio**,
which is the long-form path. Short-form degeneration exists but is small: `... ...` and `哥哥哥哥`.
So the guard is justified by loops that were reproduced on this machine, not by that one.

## The guard, and why `llm.py`'s could not simply be called

`_looping()` in `argmind/core/llm.py` is the same *shape* — count a probe taken off the tail — and
its constants cannot be borrowed, which is a measurement rather than an opinion: run unchanged over
the ten degenerate transcripts collected here it **catches 1 of 10** (0 of 152 clean transcripts,
so it is not wrong, it is aimed elsewhere). `LOOP_FLOOR = 256` was derived from a complete JSON envelope of
about 160 characters, and a spoken turn is nothing like that long — the longest clean transcript in
the whole round trip here is 63 characters. Sharing it would mean either editing `core/`, which this
package may not do, or passing constants into a private function whose docstring argues for the ones
it has. **So it is re-derived here and named after what it is**, and a test pins the 1 of 8 so that
a later reader does not "simplify" one into the other.

Two changes fall out of the measurement:

* **counted anywhere, not off the tail.** The 495-character Korean loop ends `한글자. 한글자막 by
  한효정` — the cycle drifts in its last copy, so a tail probe finds itself once and `str.count`
  returns 1. The most-repeated 10-character shingle in that same string occurs **60 times**. Tail
  anchoring is what missed it, and counting every position is O(n) on a transcript this short;
* **`REPEAT_FLOOR = 32` characters**, because a transcript is not a JSON envelope.

Against **354 strings that must not be rejected** — 152 clean transcripts from the round trip, 193
recorded `say` strings, and 9 hand-written repetitive human utterances (`아니 그거 말고` three times,
`wait wait wait wait wait`) — the settings below fire **0 times**, and they catch **7 of the 7
repetitive degenerations** in the 10 junk transcripts collected here. The three they miss are not
repetitions: `Thank you. Thank you.` (two copies, and three is the floor because a person says
`네, 네`), a Portuguese YouTube outro, and a hallucinated Korean subtitle credit — each occurring
once. Those are hallucinations rather than loops, they belong to failure 1 above, and **the guard
does not claim them**; 119 of 120 silence clips still produce a transcript with it in place.

**The 9 human strings are hand-written and adversarial, not recorded speech.** They are the weakest
part of that denominator and they are labelled so. The floor does the work for them: all 9 are
shorter than 32 characters, so what protects a person repeating themselves is that they were brief.

## Two paths, both counted

**Short-form under 30 s, long-form over it**, because over 30 s the pipeline does not degrade, it
raises: `ValueError: You have passed more than 3000 mel input features (> 30 seconds) which
automatically enables long-form generation which requires the model to predict timestamp tokens`.
A person holding the talk key for 31 seconds would otherwise lose the turn to a traceback on a
worker thread. `by_short` and `by_long` count which ran, because long-form is where every long loop
above came from and a build that quietly starts taking it should be visible.

**The flag goes on the pipeline call, not in `generate_kwargs`.** Passed inside `generate_kwargs`
it is silently ignored and the `ValueError` above is raised anyway; that cost one debugging pass
here and is written down rather than remembered.

`by_native` and `by_resampled` are the other pair: this machine's input device opens at 16 kHz, so
the resampling path is **unmeasured and linear-interpolated with no anti-alias filter**. It is
counted so that a machine taking it is visible rather than quietly worse.
"""

from __future__ import annotations

import collections
import queue
import threading
import time
import unicodedata

from ..voice import Heard

__all__ = ["WhisperListener", "why_rejected", "MODEL",
           "REPEAT_FLOOR", "REPEAT_PROBE", "REPEAT_TIMES"]

#: Chosen because it is already on this machine and because it answers in 234 ms; `faster-whisper`
#: and `ffmpeg` are deliberately not installed, since the transformers pipeline already clears the
#: bar the end-to-end turn sets.
MODEL = "openai/whisper-large-v3-turbo"

#: The repetition guard. See the module docstring for every number behind these three: 6 of 6
#: repetitive degenerations caught, 0 of 354 strings that must not be rejected.
#:
#: **`REPEAT_PROBE` is 10 and not 32** because the cycles that were actually observed are short —
#: `한글자막 by ` is 9 characters, `ご視聴ありがとうございました` is 14, `Thank you. ` is 11 — and a probe
#: longer than the cycle cannot count it. The grid was measured rather than picked: at 8 the same 7
#: are caught **and two of the 193 recorded replies go with them** (both long English sentences of
#: the `The screen shows a Character Grid pane with…` kind, which is a thing a person could say);
#: at 12 the false-positive record is still clean but `Mm-hmm Thank you. Thank you. Thank you.`
#: survives. 10 is the setting that catches the most with nothing false.
REPEAT_FLOOR, REPEAT_PROBE, REPEAT_TIMES = 32, 10, 3

#: Above this the pipeline needs long-form generation and raises without it. 30 s exactly, in
#: samples, at the rate Whisper wants.
WHISPER_RATE = 16000
LONG_FORM_SAMPLES = 30 * WHISPER_RATE


def _wordless(text: str) -> bool:
    """Is there no letter and no digit in this at all?

    `... ... ... ... ...` is what 25 seconds of room tone produced, twice, and `.` is what five
    seconds produced. **No threshold is involved**: a transcript with nothing but punctuation in it
    is not a thing a person said, at any pace, in any language. 2 of 2 such transcripts rejected,
    0 of 152 clean ones and 0 of 193 recorded replies touched.
    """
    return not any(unicodedata.category(ch)[0] in "LN" for ch in text)


def _repeating(text: str) -> bool:
    """Is a short cycle repeating itself inside this transcript?

    The shape is `argmind.core.llm._looping()`'s and the constants are not, for the reason the
    module docstring measures: that function catches 1 of the 8 degenerate transcripts recorded
    here, because its floor was derived from a JSON envelope and a spoken turn is a tenth the size.

    Counted at every position rather than off the tail, because the one loop that mattered — a real
    command buried in noise, coming back as `곱창 눌러줘 한글자막 by` forty times — drifts in its final
    copy, which defeats a tail probe (1 occurrence) and not a shingle count (60). 7 of 7 repetitive
    degenerations, 0 of 354 strings that had to survive.
    """
    squeezed = " ".join(text.split())
    if len(squeezed) < REPEAT_FLOOR:
        return False
    shingles = collections.Counter(
        squeezed[i:i + REPEAT_PROBE] for i in range(len(squeezed) - REPEAT_PROBE + 1))
    return max(shingles.values()) >= REPEAT_TIMES


def why_rejected(text: str, seconds: float = 0.0, pace_limit: float | None = None) -> str:
    """Why this transcript is not a turn, or `""` if it is one. **One place, three callers.**

    A reason rather than a bool so the count says which guard fired: a build where
    `rejected_repeated` climbs has a recogniser problem, one where `rejected_pace` climbs has a
    threshold problem, and a single `rejected` total could not tell them apart.

    `pace_limit=None` — the default, and what the listener ships with — **rejects nothing on pace**.
    """
    if not text or not text.strip():
        return "empty"
    if _wordless(text):
        return "wordless"
    if _repeating(text):
        return "repeated"
    if pace_limit is not None and Heard(text, seconds).pace > pace_limit:
        return "pace"
    return ""


class WhisperListener:
    """A microphone and Whisper, satisfying `argmind.voice.Listener`.

    `hold()` opens and closes the device; `heard()` returns what has finished and **never blocks**,
    because the caller is the thread already waiting on the model. Recognition, and the 4.3 s model
    load, happen on this object's own thread.

    **`recognise` is injectable** — a callable taking `(samples, samplerate)` and returning a string
    — the way `Agent.build` injects eyes, hands and a model. That is what lets the guards, the
    queue, the counters and `heard()` be tested with no GPU, no device and no network, and it is how
    the fabrications above were replayed against the guard.
    """

    def __init__(self, *, model: str = MODEL, device: str | None = None,
                 samplerate: int = WHISPER_RATE, pace_limit: float | None = None,
                 recognise=None, blocksize: int = 1600) -> None:
        self.model, self.device = model, device
        self.samplerate = int(samplerate)
        #: Unset on purpose. The module docstring names the audio that would set it and says why
        #: that audio does not exist yet; a guessed cut-off silently drops real turns.
        self.pace_limit = pace_limit
        self.blocksize = int(blocksize)
        self.counts = {
            "held": 0, "captured": 0, "offered": 0, "heard": 0, "empty_capture": 0,
            "rejected_empty": 0, "rejected_wordless": 0, "rejected_repeated": 0,
            "rejected_pace": 0, "by_short": 0, "by_long": 0, "by_native": 0,
            "by_resampled": 0, "failed": 0,
        }
        self.last_error: BaseException | None = None
        self.load_seconds: float | None = None
        #: (audio seconds, transcription seconds) for the last 64 utterances — what the numbers in
        #: the docstring are re-derived from, and what an interface can show without a harness.
        self.recent: collections.deque = collections.deque(maxlen=64)
        self._recognise = recognise
        self._pipe = None
        self._q: queue.Queue = queue.Queue()
        self._done: collections.deque = collections.deque()
        self._stream = None
        self._blocks: list = []
        self._capture_rate = self.samplerate
        self._working = False
        self._thread = threading.Thread(target=self._run, name="argmind-whisper", daemon=True)
        self._thread.start()

    # ---- Listener -----------------------------------------------------------------------------

    def hold(self, down: bool) -> None:
        """Capture while the key is down. **Push-to-talk is the design**, not a placeholder.

        Going down also asks the worker to load the model, so the 4.3 s cold load overlaps the
        person speaking instead of being added to their first answer.
        """
        if down:
            self._begin()
        else:
            self._end()

    def heard(self) -> Heard | None:
        """A finished utterance, or `None`. A deque pop — no lock, no wait, no device call.

        `None` also means rejected: a fabrication is not an error, the same way `Watch.look()`
        returning `None` is not. `counts` says which guard fired.
        """
        try:
            return self._done.popleft()
        except IndexError:
            return None

    def release(self) -> None:
        """Close the device, stop the worker, drop the weights. Safe to call twice."""
        self._end(discard=True)
        if self._thread.is_alive():
            self._q.put(("release",))
            self._thread.join(timeout=30.0)

    # ---- off the protocol -----------------------------------------------------------------------

    def warm(self) -> None:
        """Load the model now, on the worker. Costs 4.31-4.32 s once and nothing afterwards.

        Off the protocol because a `Listener` on a phone may have nothing to warm; an interface that
        wants it reaches for it with `getattr`, and doing nothing is an honest fallback — the first
        utterance simply pays the load.
        """
        self._q.put(("warm",))

    def pending(self) -> int:
        """Utterances captured and not yet answered for. 0 or 1 in practice; median 234 ms of 1."""
        return self._q.qsize() + (1 if self._working else 0)

    def offer(self, samples, samplerate: int | None = None, at: float | None = None) -> None:
        """Hand audio in without a microphone — the same path `hold(False)` takes.

        **Every number in this file came through here**, from SAPI's memory stream and from
        generated noise. It is also what a measurement harness uses, and what lets a test drive the
        whole queue with a list of floats and an injected recogniser.
        """
        self.counts["offered"] += 1
        self._q.put(("audio", samples, int(samplerate or self.samplerate),
                     time.time() if at is None else float(at)))

    # ---- capture ---------------------------------------------------------------------------------

    def _begin(self) -> None:
        if self._stream is not None:
            return
        self.counts["held"] += 1
        self._q.put(("warm",))
        try:
            import sounddevice                                               # noqa: PLC0415
        except Exception as exc:                                             # noqa: BLE001
            self.last_error = exc
            self.counts["failed"] += 1
            return
        rate = self.samplerate
        try:
            sounddevice.check_input_settings(samplerate=rate, channels=1, dtype="float32")
            self.counts["by_native"] += 1
        except Exception:                                                    # noqa: BLE001
            #: The device will not give 16 kHz. Take its own rate and resample on the way out —
            #: counted, because it is the unmeasured path.
            rate = int(sounddevice.query_devices(kind="input")["default_samplerate"])
            self.counts["by_resampled"] += 1
        self._blocks, self._capture_rate = [], rate
        try:
            self._stream = sounddevice.InputStream(
                samplerate=rate, channels=1, dtype="float32", blocksize=self.blocksize,
                callback=lambda data, frames, at, status: self._blocks.append(data.copy()))
            self._stream.start()
        except Exception as exc:                                             # noqa: BLE001
            self.last_error = exc
            self.counts["failed"] += 1
            self._stream = None

    def _end(self, *, discard: bool = False) -> None:
        stream, self._stream = self._stream, None
        if stream is None:
            return
        try:
            stream.stop()
            stream.close()
        except Exception as exc:                                             # noqa: BLE001
            self.last_error = exc
            self.counts["failed"] += 1
        blocks, self._blocks = self._blocks, []
        if discard or not blocks:
            if not discard:
                self.counts["empty_capture"] += 1
            return
        import numpy                                                         # noqa: PLC0415

        samples = numpy.concatenate(blocks).reshape(-1)
        self.counts["captured"] += 1
        self._q.put(("audio", samples, self._capture_rate, time.time()))

    # ---- the worker ------------------------------------------------------------------------------

    def _run(self) -> None:
        while True:
            command = self._q.get()
            if command[0] == "release":
                self._unload()
                return
            self._working = True
            try:
                if command[0] == "warm":
                    self._load()
                elif command[0] == "audio":
                    self._transcribe(*command[1:])
            except Exception as exc:                                         # noqa: BLE001
                #: Failure #1 of this package again: an exception here would kill the thread and
                #: the agent would go deaf with nothing raised anywhere the person can see.
                self.last_error = exc
                self.counts["failed"] += 1
            finally:
                self._working = False

    def _load(self) -> None:
        if self._pipe is not None or self._recognise is not None:
            return
        import torch                                                         # noqa: PLC0415
        from transformers import pipeline                                    # noqa: PLC0415

        device = self.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        began = time.perf_counter()
        self._pipe = pipeline("automatic-speech-recognition", model=self.model,
                              dtype=torch.float16 if "cuda" in device else torch.float32,
                              device=device)
        self.load_seconds = time.perf_counter() - began

    def _unload(self) -> None:
        #: Nothing loaded, nothing to give back — and importing torch to discover that would make
        #: shutting down an injected fake require a GPU-capable install. Caught by the suite.
        if self._pipe is None:
            return
        self._pipe = None
        try:
            import torch                                                     # noqa: PLC0415

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:                                                    # noqa: BLE001, S110
            pass

    def _transcribe(self, samples, rate: int, at: float) -> None:
        seconds = len(samples) / float(rate or self.samplerate)
        if not len(samples):
            self.counts["empty_capture"] += 1
            return
        began = time.perf_counter()
        text = (self._recognise or self._whisper)(samples, rate)
        self.recent.append((seconds, time.perf_counter() - began))
        reason = why_rejected(text, seconds, self.pace_limit)
        if reason:
            self.counts["rejected_" + reason] += 1
            return
        self.counts["heard"] += 1
        self._done.append(Heard(text.strip(), seconds, at))

    def _whisper(self, samples, rate: int) -> str:
        import numpy                                                         # noqa: PLC0415

        self._load()
        audio = numpy.asarray(samples, dtype=numpy.float32).reshape(-1)
        if rate != WHISPER_RATE:
            audio = _resampled(audio, rate, WHISPER_RATE)
        #: `return_timestamps` belongs to the pipeline call. Put it in `generate_kwargs` and it is
        #: ignored, and the ValueError about 3000 mel features is raised anyway.
        extra = {}
        if audio.size > LONG_FORM_SAMPLES:
            extra["return_timestamps"] = True
            self.counts["by_long"] += 1
        else:
            self.counts["by_short"] += 1
        answer = self._pipe({"array": audio, "sampling_rate": WHISPER_RATE},
                            generate_kwargs={"task": "transcribe"}, **extra)
        return str(answer.get("text", ""))


def _resampled(samples, src: int, dst: int):
    """Linear interpolation, **no anti-alias filter and unmeasured** — this machine's device gives
    16 kHz natively, so nothing here has ever run. `counts["by_resampled"]` is how a machine that
    does take it becomes visible instead of quietly worse."""
    import numpy                                                             # noqa: PLC0415

    if src == dst or len(samples) == 0:
        return samples
    wanted = int(round(len(samples) * dst / float(src)))
    return numpy.interp(numpy.linspace(0, len(samples) - 1, wanted),
                        numpy.arange(len(samples)), samples).astype(numpy.float32)
