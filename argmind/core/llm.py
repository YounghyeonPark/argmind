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
"""

from __future__ import annotations

import base64
import io
import json
import time
import urllib.error
import urllib.request

__all__ = ["Local", "Reply", "OLLAMA", "NOT_RUNNING"]

OLLAMA = "http://127.0.0.1:11434"
NOT_RUNNING = (
    f"no Ollama server answered at {OLLAMA}. Start it, or point Local(host=...) somewhere else. "
    "This project runs the model locally on purpose: no per-call cost, no network, no key.")


class Reply:
    """What came back, and what it cost — both, always. A latency figure that is not attached to the
    call it describes is how a stale number gets published."""

    __slots__ = ("text", "ttft", "total", "prompt_tokens", "out_tokens", "reason", "thought")

    def __init__(self, text, ttft, total, prompt_tokens, out_tokens, reason, thought=0) -> None:
        self.text = text
        self.ttft = ttft
        self.total = float(total)
        self.prompt_tokens = prompt_tokens
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
        an empty answer gets read as a refusal."""
        return self.reason == "length"

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

        Quotes are tracked because a brace inside a string is not a brace — a reply saying
        `{"say": "press } to close"}` would otherwise close one level early.
        """
        start = self.text.find("{")
        if start < 0:
            return None
        depth, in_string, escaped = 0, False, False
        for i in range(start, len(self.text)):
            ch = self.text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
            elif ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(self.text[start:i + 1])
                    except json.JSONDecodeError:
                        return None
        return None

    def __repr__(self) -> str:
        t = f"{self.ttft * 1000:.0f}ms" if self.ttft else "--"
        return f"Reply({self.text[:40]!r}, ttft={t}, total={self.total * 1000:.0f}ms)"


class Local:
    """One call to a local model. No streaming callbacks, no session state — the loop owns those."""

    def __init__(self, model: str, *, host: str = OLLAMA, ctx: int = 8192,
                 max_tokens: int = 4096, keep_alive: str = "60s", think=None) -> None:
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

    def ask(self, prompt: str, images=None) -> Reply:
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
                    thinking += len((d.get("message") or {}).get("thinking") or "")
                    chunk = (d.get("message") or {}).get("content")
                    if chunk:
                        if ttft is None:
                            ttft = time.perf_counter() - t0
                        text += chunk
                    if d.get("done"):
                        last = d
        except urllib.error.URLError as exc:                              # pragma: no cover
            raise RuntimeError(NOT_RUNNING) from exc
        return Reply(text.strip(), ttft, time.perf_counter() - t0,
                     last.get("prompt_eval_count"), last.get("eval_count"),
                     last.get("done_reason"), thinking)

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
