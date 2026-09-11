# SPDX-License-Identifier: Apache-2.0
"""The local model, with every setting that was measured the hard way baked in.

**Six things silently broke a measurement of this exact model before any of them was understood.**
They are here as code rather than as advice, because each one produced a plausible zero that looked
like a model failure:

| what | symptom | what it would have been reported as |
|---|---|---|
| `format: "json"` | 366 ms, empty output | "the model cannot answer" |
| `num_predict` too small | thinks, then truncates | "the model cannot answer" |
| reading `response` | the text is on `message.content` | "the model cannot answer" |
| **`num_ctx` default 4096** | prompt 3,661 + reply 435 ≈ 4,096, `done_reason: length` | "it cannot work at native resolution" |
| **instructing the coordinate convention** | 13,000 characters arguing with itself, no answer | "it cannot read a dense screen" |
| warm vs cold prefill | 25 ms warm, 0.9–5.5 s cold | "images are free, no cache needed" |

Three of those would have killed a direction that works. One would have kept a direction that does
not. Every one of them pointed the way I was already leaning, which is why they are written down.

**Thinking cannot be turned off** on `qwen3-vl:8b` here — `think: false` and a `/no_think` suffix
both leave ~400 thinking chunks — so the budget has to be large enough to think *and* answer.

**A loaded model is resident memory, and `num_ctx` is part of the bill.** Measured with `ollama ps`:
`qwen2.5:7b` occupies **4.7 GB at ctx 4096 and 5.1 GB at 8192**, and `qwen3-vl:8b` is 6.1 GB before
its context. Ollama holds a model for **five minutes after the last call** by default, so alternating
between two models keeps both — about 11 GB — and a 16 GB GPU then spills into system RAM.

That is written here because it was learned from a machine running out of memory and rebooting. The
python process that actually exhausted it held **82 GB** and was never identified, so this is not a
confession of cause; but a `llama-server` of mine was still holding **5.9 GB fifty-four minutes after
my last call**, which is pressure I put on a machine that was already thrashing, and there was no
reason for it. Hence `keep_alive` below, and `release()`.

**Measured latencies on this machine** (RTX 4090 Laptop, 16 GB):

    qwen2.5:7b   text, warm      67–85 ms to first token, 156–200 ms total
    qwen3-vl:8b  text, warm      2.7 s
    qwen3-vl:8b  image + pixels  24–82 s, and on a dense screen no answer at all

The first row is why `choose.py` asks a text model a text question.

**The ceiling is a backstop, not an instrument — decided 2026-09-11 with both sides measured.**
`num_predict` stays at **4096**. Lowering it is the obvious move and it does not solve the problem:
a reply cut off at 256 tokens parses to nothing in exactly the way a reply cut off at 4,096 does, so
the turn still produces no speech and no action, and a four-second silence is still a silence. What
a lower ceiling *would* do is reintroduce row two of the table above on the arm that needs the room
- `--think on` spends **3,854 characters of reasoning and 277 output tokens** on goals where
thinking off peaks at 39, and reasoning counts against `num_predict`, so any ceiling small enough to
bite the conversational loop is small enough to truncate that arm mid-thought. The ceiling has one
job, which is to stop a runaway before it costs a minute, and 4096 is where it does that without
binding on a supported configuration.

**What bounds the loop instead is two stop conditions on the SHAPE of what is coming back**, because
the quantity that separates a loop from a long answer is generated length and not seconds — a sweep
run while this machine was busy shows 6-14 s stalls at 17-39 output tokens on perfectly healthy
calls, so a time budget would kill the healthy ones first:

    the envelope closed    stop GRACE chunks later   the reader only ever reads the first object
    the tail is repeating  stop                      LOOP_FLOOR characters, none of them closing one

Measured over **1,845 calls** at the shipped prompt with a conversation in it
(`measure/conversation-qwen3-8b-off-qa62805-s5020f7-f041650-d1.3.5.8.12@r12-x3.json`, whose
`llm_print` is this file as it stood before the change): `done_reason` **stop 1,827, length 18**;
output tokens **median 26, p95 47, max 4096**. The 18 are one `{"key": "tab"}` repetition at
**69-71 s each — 47% of the run's model time in 1.0% of its turns**, and every one of them is a turn
the person hears nothing back from. Twenty-seven more closed a usable object and then kept talking:
**962 output tokens for a 119-character sentence, 192 for a 31-character one**.

**What the two stops are worth is ESTIMATED, and the estimate is arithmetic over that same file, not
a run.** Healthy calls there fit `seconds = 0.150 + 0.01796 * out_tokens` (n = 1,815), and that fit
predicts the twelve 962-token calls at 17.4 s against 16.2 measured and the eighteen 4,096-token
ones at 73.7 against 69.0, so it is not fitted to the tail it is being used on. Closing early
recovers **~226 s over 27 calls**; stopping a loop at 256 characters recovers **~1,210 s over 18**;
together **~54% of that run's 2,649 s of model time**. **None of it is measured** until that harness
is run again — `Local.ask` names the arm that would refute it.
"""

from __future__ import annotations

import base64
import io
import json
import time
import urllib.error
import urllib.request

__all__ = ["Local", "Reply", "OLLAMA", "NOT_RUNNING", "CLOSED", "DEGENERATE"]

OLLAMA = "http://127.0.0.1:11434"
NOT_RUNNING = (
    f"no Ollama server answered at {OLLAMA}. Start it, or point Local(host=...) somewhere else. "
    "This project runs the model locally on purpose: no per-call cost, no network, no key.")

#: What ended the call. The first two are the server's own `done_reason` and are untouched; the
#: second two are this class stopping the read, and they are separate values rather than a reuse of
#: `stop` because **a reader whose totals stop adding up needs to know who ended the call**. A
#: result file carries `reason` verbatim, so a run taken after 2026-09-11 says on its own face which
#: mechanism was in force.
FINISHED, CEILING, CLOSED, DEGENERATE = "stop", "length", "closed", "degenerate"

#: Content chunks read *after* the reply's JSON object has closed, before the read stops.
#:
#: **It is not zero, and the reason is a measurement rather than caution.** Breaking on the closing
#: brace itself would mean never receiving the server's final message, and that message is where
#: `prompt_eval_count` and `eval_count` live — so every healthy call in every result file would
#: trade two real token counts for a local estimate, in order to save the one token a finished reply
#: has left to emit. A model that is done sends its end marker immediately after the brace, so the
#: grace costs nothing at all on the calls that behave and 24 chunks (**0.43 s** at the 55.7
#: tokens/s that run's own fit implies, which is derived and not itself a measurement) on
#: the ones that do not. The asymmetry says err large: too
#: large costs half a second on a call already wasting fifteen, too small silently degrades every
#: token count in a 1,845-call result file.
GRACE = 24

#: The loop detector. `LOOP_PROBE` characters off the tail, counted within the last `LOOP_WINDOW`,
#: is degeneration at `LOOP_TIMES` non-overlapping occurrences.
#:
#: **`LOOP_FLOOR` is where the false-positive argument is spent**: nothing is examined at all until
#: the reply is longer than any complete reply this project has recorded. The longest sentence in
#: the 1,845-call run is 119 characters and the longest under an ordinary reply is 88, so a complete
#: envelope runs to roughly 160 characters; 256 is comfortably past that, and past it the reply has
#: also failed to close an object that a real answer closes inside 26 tokens. A cycle of up to
#: `LOOP_WINDOW // LOOP_TIMES` = 128 characters is caught, and the measured one is sixteen.
#:
#: **The cost of a false positive is bounded by the benefit**, which is what makes erring toward
#: firing safe here: a wrongly stopped turn produces nothing, and that is what the same turn already
#: produces today after 71 s. The case it would take is a person genuinely asking for a long
#: repeated string to be typed. A `degenerate` in `report()` that a human recognises as a real
#: request is what would say the floor is too low.
LOOP_FLOOR, LOOP_WINDOW, LOOP_PROBE, LOOP_TIMES = 256, 512, 32, 4


class _Balanced:
    """Where the first *balanced* JSON object starts and ends, fed either a whole reply or a stream.

    **One scanner with two callers, which is the reason it is a class at all.** `Reply.json()` needs
    it to extract the object; `Local.ask` needs it to know an answer is complete while tokens are
    still arriving. Written twice, the two would eventually disagree about what complete means, and
    the stop would then cut a reply off at a point the extractor refuses — a silent zero of exactly
    the shape this file exists to record. This repository has the receipt: a headless harness that
    re-implemented a push instead of calling it hid a real bug for a day.

    Quotes and escapes are tracked because a brace inside a string is not a brace — a reply saying
    `{"say": "press } to close"}` would otherwise close one level early.
    """

    __slots__ = ("start", "end", "_depth", "_in_string", "_escaped", "_at")

    def __init__(self) -> None:
        self.start = self.end = -1
        self._depth = self._at = 0
        self._in_string = self._escaped = False

    @property
    def closed(self) -> bool:
        return self.end >= 0

    def feed(self, chunk: str) -> bool:
        """Consume more text; returns whether a whole object has now been seen.

        Indices are into everything fed so far, so a caller streaming a reply and a caller handing
        over the finished text arrive at the same two numbers.
        """
        for ch in chunk:
            at, self._at = self._at, self._at + 1
            if self.end >= 0:
                continue
            if self.start < 0:
                if ch != "{":
                    continue
                self.start = at
            if self._in_string:
                if self._escaped:
                    self._escaped = False
                elif ch == "\\":
                    self._escaped = True
                elif ch == '"':
                    self._in_string = False
            elif ch == '"':
                self._in_string = True
            elif ch == "{":
                self._depth += 1
            elif ch == "}":
                self._depth -= 1
                if self._depth == 0:
                    self.end = at
        return self.end >= 0


def _looping(text: str) -> bool:
    """Is the tail of this reply a short cycle repeating itself?

    Characters and not tokens, because characters are what the extractor reads and a chunk boundary
    is the server's business. Non-overlapping occurrences of the tail rather than an exact period
    over the window, so a cycle that drifts slightly — a changing index, a stray space — is still
    caught; the exact-period test is the tidier one and it is brittle in the direction that matters.
    """
    if len(text) < LOOP_FLOOR:
        return False
    return text[-LOOP_WINDOW:].count(text[-LOOP_PROBE:]) >= LOOP_TIMES


class Reply:
    """What came back, and what it cost — both, always. A latency figure that is not attached to the
    call it describes is how a stale number gets published."""

    __slots__ = ("text", "ttft", "total", "prompt_tokens", "out_tokens", "reason", "thought")

    def __init__(self, text, ttft, total, prompt_tokens, out_tokens, reason, thought=0) -> None:
        self.text = text
        self.ttft = ttft
        self.total = float(total)
        #: `None` whenever `reason` is one of this class's own stops: the server puts
        #: `prompt_eval_count` only in its final message, and a read that ended early never saw one.
        #: Absent rather than zero, because a prompt of unknown size and a prompt of no size are not
        #: the same fact and the second one is impossible.
        self.prompt_tokens = prompt_tokens
        #: **Whose count this is depends on `reason`, and that is the only place it is written
        #: down.** Under `stop` or `length` it is the server's `eval_count`. Under `closed` or
        #: `degenerate` the server's final message never arrived, so it is the number of streamed
        #: chunks counted here — one per token as this server streams, which is a description of
        #: what arrived and not a token count the server confirmed. The distinction is legible from
        #: `reason` alone rather than carried in a second field, because a second field is one more
        #: thing a harness has to remember to record.
        self.out_tokens = out_tokens
        self.reason = reason
        #: Characters of reasoning the model produced and this class threw away. **Discarding it
        #: was right and hiding its cost was not**: `qwen3:8b` answered a twelve-token question with
        #: 201 output tokens and 1,506 characters of thinking, at 2.95 s to first token, and nothing
        #: in the reply said why. Counted so the bill is visible.
        self.thought = int(thought)

    @property
    def truncated(self) -> bool:
        """`length` means it ran out of room, not that it finished. Treating the two the same is how
        an empty answer gets read as a refusal.

        **Deliberately NOT widened to cover `degenerate`, on 2026-09-11.** The tempting move is to
        make this mean "did not finish", so the loop detector's catches keep showing up in the count
        the harnesses already print. It is wrong twice. `measure/`'s own summary prints this beside
        the words *stopped at the token ceiling ... done_reason 'length'*, which would become a lie
        for every degenerate call; and a column that means one thing before a commit and another
        after is this project's most-recorded way of losing a comparison. The count of degenerate
        calls is not hidden by the narrow reading — `spend_summary` prints `reasons` verbatim, so a
        run after the change reads `{'stop': ..., 'degenerate': 18}` on the same line.

        `unfinished` is the wide question, and it is a different one.
        """
        return self.reason == CEILING

    @property
    def unfinished(self) -> bool:
        """No finished answer arrived: the ceiling, or a loop that was stopped.

        **The question a caller with a person waiting actually has**, and the one nothing in this
        repository asks yet. `truncated` answers *did the budget run out*, which is a question about
        this file's settings; a turn that produced no speech and no action wants to know only
        whether there is anything to read, and both of these mean there is not. What to do about it
        is not this file's call: the sentence *"that came out garbled — say it again?"* belongs
        wherever the turn is assembled, and it needs this property and `reason` to tell the two
        apart. Exposed here, unconsumed, on purpose — the alternative was to guess at the wording
        inside a model wrapper.
        """
        return self.reason in (CEILING, DEGENERATE)

    def json(self) -> dict | None:
        r"""The first *balanced* JSON object in the reply, or `None`.

        Extracted rather than parsed whole, because a model asked for JSON usually returns JSON and
        occasionally returns JSON with a sentence in front of it.

        **Balanced, and that is the seventh pitfall.** This was `re.search(r"\{.*?\}")`, which is
        correct for exactly as long as the reply contains no nested object — and it stopped being
        correct the moment the answer grew a list of things to do: `{"say": "ok", "do": [{"click":
        0}]}` matched up to the *inner* brace, failed to parse, and returned `None`. The model was
        answering perfectly and every action silently disappeared. Same shape as the other six: a
        plausible zero produced by the harness rather than by the model.

        **The scan itself moved into `_Balanced` on 2026-09-11 and did not change.** `Local.ask`
        stops reading once an object has closed, and a stop that used its own copy of this logic
        could cut a reply off at a point this method then refuses. One scanner cannot disagree with
        itself; two eventually do.
        """
        scan = _Balanced()
        if not scan.feed(self.text):
            return None
        try:
            return json.loads(self.text[scan.start:scan.end + 1])
        except json.JSONDecodeError:
            return None

    def __repr__(self) -> str:
        t = f"{self.ttft * 1000:.0f}ms" if self.ttft else "--"
        return f"Reply({self.text[:40]!r}, ttft={t}, total={self.total * 1000:.0f}ms)"


class Local:
    """One call to a local model. No streaming callbacks, no session state — the loop owns those."""

    def __init__(self, model: str, *, host: str = OLLAMA, ctx: int = 8192,
                 max_tokens: int = 4096, keep_alive: str = "60s", think=None,
                 stop_when_answered: bool = True, stop_when_looping: bool = True) -> None:
        self.model = model
        self.host = host.rstrip("/")
        #: How long the server keeps the weights resident after a call. **A real trade, not a
        #: tidy-up**: Ollama's default is five minutes, which holds 5-6 GB long after a
        #: conversation ends; unloading immediately would make every exchange pay the cold load,
        #: measured at 5.8-8.4 s, which is the opposite of what this project is for. A minute
        #: covers a pause in conversation and releases an idle machine.
        #:
        #: The real fix is `release()` at the end, and the caller is expected to call it.
        self.keep_alive = keep_alive
        #: **8192, not the 4096 default.** A 2560x1440 screenshot is ~3,660 prompt tokens and this
        #: model thinks for ~400 more; at 4096 the answer is cut off and `done_reason` says `length`
        #: while the content is empty. Measured: at 8192 the same call answers correctly.
        self.ctx = int(ctx)
        self.max_tokens = int(max_tokens)
        #: `False` switches a reasoning model's thinking off; `None` leaves the field out entirely,
        #: which is what a model that has never heard of it needs. Measured: `qwen3:8b` takes
        #: **8.29 s and 336 tokens** to answer `{"say": "hi"}` with thinking on and **1.70 s and 7
        #: tokens** with it off. The agent asks a model to pick a name from a list it can already
        #: see; there is nothing there to reason about for 1,500 characters.
        self.think = think
        #: The two stops, **switchable because a brake that cannot be switched off cannot be
        #: measured**. Both off reproduces this file's behaviour before 2026-09-11 exactly, request
        #: for request, so a harness can run one arm each way in a single session and compare
        #: against the same machine, the same residency and the same warm cache instead of against
        #: a result file taken on another day.
        self.stop_when_answered = bool(stop_when_answered)
        self.stop_when_looping = bool(stop_when_looping)

    def ask(self, prompt: str, images=None) -> Reply:
        """One call, read until there is nothing left worth reading.

        **The request is byte-identical to what it was before the stops existed** — same messages,
        same `num_predict`, same `num_ctx`, same temperature. Only the reading stops early. So the
        tokens the model produces up to the stop are the tokens it would have produced anyway, and
        **nothing this project measures about what the model CHOOSES can move**: accuracy, the rule
        rows, the leak, the refusal rate. What moves is latency, and the token accounting on the few
        calls that end early.

        **The measurement that would refute that** is `measure/conversation.py` at the fingerprints
        of the run quoted at the top of this file, two arms in one session -
        `Local(..., stop_when_answered=False, stop_when_looping=False)` against the defaults. Same
        verdicts required; the run is only 82% deterministic call for call (565 of 615 turns give an
        identical token count across its three repeats), so the comparison has to be at the verdict
        the harness already scores and not at the token. **If a verdict moves, the claim that the
        request is unchanged is wrong** and the stops go back behind a flag. What the arms should
        show if this is right: `reasons` gains `closed` and `degenerate`, `seconds_max` falls from
        ~70 s toward ~2 s, and `out_tokens_max` falls from 4096 to a few hundred.

        A second thing it would refute: this assumes the server abandons generation when the client
        disconnects. If it does not, the work continues off-stream and the saving shows up nowhere -
        the run's wall clock would be unchanged while every per-call `seconds` fell, which is a
        disagreement the same two arms would make obvious.
        """
        content = {"role": "user", "content": prompt}
        if images:
            content["images"] = [self._b64(im) for im in images]
        payload = {
            "model": self.model,
            "messages": [content],
            "stream": True,
            "keep_alive": self.keep_alive,
            #: No `format: "json"`. It silences this model completely -- 366 ms, empty output. The
            #: shape is asked for in the prompt and pulled out of the text by `Reply.json()`.
            "options": {"num_predict": self.max_tokens, "num_ctx": self.ctx, "temperature": 0},
        }
        if self.think is not None:
            payload["think"] = self.think
        body = json.dumps(payload).encode()
        req = urllib.request.Request(f"{self.host}/api/chat", data=body,
                                     headers={"Content-Type": "application/json"})
        t0, ttft, text, last, thinking = time.perf_counter(), None, "", {}, 0
        envelope, spare, arrived, ended = _Balanced(), GRACE, 0, None
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                for line in r:
                    if not line.strip():
                        continue
                    d = json.loads(line)
                    #: The answer is `message.content`. `message.thinking` is the model reasoning and
                    #: is deliberately discarded: it is not an answer, and reading `response` (the
                    #: other endpoint's field) returns nothing at all. **Its size is kept**, because
                    #: throwing the text away also threw away the only sign of what it cost.
                    reasoning = (d.get("message") or {}).get("thinking") or ""
                    thinking += len(reasoning)
                    chunk = (d.get("message") or {}).get("content")
                    #: Counted, not estimated afterwards from characters: this is what `out_tokens`
                    #: falls back to when the server's final message never arrives.
                    arrived += 1 if (reasoning or chunk) else 0
                    if chunk:
                        if ttft is None:
                            ttft = time.perf_counter() - t0
                        text += chunk
                        #: **Both stops watch the content and neither watches the reasoning.** A
                        #: thinking model legitimately circles a point while composing, and the
                        #: 13,000-character argument this model once had with itself was reasoning,
                        #: not answer — stopping that is a different decision on different evidence,
                        #: and the `--think on` arm is supported. So the arm reaches the answer
                        #: exactly as it did, and the stops apply to the answer only.
                        if envelope.closed:
                            spare -= 1
                            if self.stop_when_answered and spare <= 0:
                                ended = CLOSED
                                break
                        elif not envelope.feed(chunk) and self.stop_when_looping and _looping(text):
                            ended = DEGENERATE
                            break
                    if d.get("done"):
                        last = d
        except urllib.error.URLError as exc:                              # pragma: no cover
            raise RuntimeError(NOT_RUNNING) from exc
        if ended is None:
            return Reply(text.strip(), ttft, time.perf_counter() - t0,
                         last.get("prompt_eval_count"), last.get("eval_count"),
                         last.get("done_reason"), thinking)
        return Reply(text.strip(), ttft, time.perf_counter() - t0, None, arrived, ended, thinking)

    def release(self) -> bool:
        """Ask the server to unload the weights now. Returns whether it answered.

        Not a nicety. A model left resident is 5-6 GB held by a process nobody is talking to, and
        this project put 5.9 GB of exactly that on a machine for fifty-four minutes after its last
        call, on a day the machine ran out of memory and rebooted. Whatever else was to blame, that
        part was avoidable and free to avoid.

        Failure is swallowed on purpose: this runs on the way out, and a server that has already
        gone is the outcome being asked for anyway.
        """
        body = json.dumps({"model": self.model, "messages": [], "keep_alive": 0}).encode()
        req = urllib.request.Request(f"{self.host}/api/chat", data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30):
                return True
        except Exception:                                                  # noqa: BLE001
            return False

    @staticmethod
    def _b64(frame) -> str:
        from PIL import Image                                             # noqa: PLC0415

        if isinstance(frame, str):
            return frame
        im = frame if isinstance(frame, Image.Image) else Image.fromarray(frame)
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return base64.standard_b64encode(buf.getvalue()).decode("ascii")
