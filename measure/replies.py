# SPDX-License-Identifier: Apache-2.0
"""Real model replies, kept **chunk by chunk**, so `llm.py`'s stop logic can be replayed offline.

    python -m measure.replies --help
    python -m measure.replies --plan                        # what would be recorded, no model
    python -m measure.replies --arm off --only ref-eraser   # one scene, while writing
    python -m measure.replies --arm off              # the shipped configuration's stream shapes
    python -m measure.replies --arm on               # --think on, the densest ceiling source
    python -m measure.replies --arm small            # qwen3:0.6b, for replies that do not parse
    python -m measure.replies --arm stress           # asked for a loop, because none turns up now

`docs/findings/porting-the-core.md` names this as one half of Stage 0 and says why it is a blocker:
*"Every harness stores aggregates and no per-reply record survives"*, so `Reply.json()`'s balanced
scanner, `_Balanced`, `_looping`, `GRACE` and the truncation logic could only be ported against
streams somebody imagined. `tests/test_llm_replay.py` is the equivalence test this feeds; the other
half, `tests/test_loop_golden.py`, is done.

## Why the record is a list of chunks and not a string

Read `argmind/core/llm.py:374-412` before changing the format. The stop logic does not run on the
finished text:

* `envelope.feed(chunk)` consumes **chunk by chunk** and `spare` counts *chunks* after the envelope
  closes, not characters. **The same final text split differently stops in a different place**, so a
  record of the joined text cannot exercise `GRACE` at all.
* `arrived` counts stream lines where `reasoning` **or** `chunk` was non-empty, so whether a line
  carried thinking matters even when it carried no content. It is what `out_tokens` falls back to
  when a stop fires and the server's final message never arrives.
* the final `done` message carries `prompt_eval_count`, `eval_count` and `done_reason`, and it is
  the only line that does.

So a row here is the ordered sequence of stream lines, one entry each, in the order they arrived.

## Recorded with the stops OFF, and that is the one thing that would waste the run

`Local(..., stop_when_answered=False, stop_when_looping=False)`. With them on the stream is cut at
the stop and everything after it is never seen — the corpus would then contain exactly the prefixes
the shipped logic already produces and could never exercise the logic it exists to test. The flags
are in the filename and in the JSON, and `--stops-on` writes `STOPSON` into the name rather than
letting the two kinds of file share one.

## What is NOT stored, and why each is safe

* **The reasoning text.** A thinking chunk is stored as its length. `Local.ask` reads `reasoning`
  in exactly two ways — `len(reasoning)` and its truthiness — so a substitute of the same length is
  behaviourally identical, and `llm.py` throws the text away anyway. It is 3,854 characters a call
  on the `--think on` arm and it would otherwise be most of the corpus by weight.
  `tests/test_llm_replay.py` reads that claim off `ask`'s syntax tree rather than trusting this
  paragraph, so a version of `ask` that looked at the reasoning *text* would fail the suite.
* **The prompt.** Its length and its sha are stored, with the case, the arm, the distance, the
  repeat and the turn index, which is what is needed to rebuild it: the scenes below are
  `measure/conversation.py`'s own cases driven through its own `drive()`, so that harness renders
  the same bytes. Storing 6 KB of widget list per row would be three quarters of the file.
* **The joined text.** Its length, its sha and its first 120 characters are stored; the text itself
  is the concatenation of the entries, and the replay reconstructs it. Storing both would let two
  copies of one fact disagree.

## The selection is deliberate, stratified, and NOT a sample

**No rate may be computed from this file.** The interesting rows are the ones that broke something
and they are now rare — the shipped stops removed the tail — so the scenes below over-sample them on
purpose, from the three sources this project has measured:

* the `{"key": "tab"}` repetition to the 4,096-token ceiling, **18 of 1,845 turns** in
  `measure/conversation-qwen3-8b-off-qa62805-s5020f7-f041650-d1.3.5.8.12@r12-x3.json`, every one of
  them the `no-copy` case on `character-map` at turn 4, at 69-71 s each. The `without` arm produced
  it 12 times in 12 there and the `with` arm 6 in 12, so both are recorded;
* the envelope that closes and then keeps talking, **962 output tokens for a 119-character
  sentence** (`rule-close`, turn 3) and **192 for a 31-character one** (`no-colour`, turn 7). These
  are the only rows that can exercise `GRACE` at all, since a reply whose closing brace is its last
  content chunk stops in the same place at every grace above zero;
* `--think on`, **5 replies at the ceiling in 168 calls** in
  `measure/repair-qwen3-8b-on-qa62805-s5020f7-f339a3d-h09102d-x3.json`, on three prompts named in
  `SINGLES` below. This is the one of the three that still reproduces: see `STRESS`, which exists
  because **at the shipped prompt the other two do not**, and which asks for the shapes outright.

The healthy rows come free: every other turn of those same conversations is recorded too, so the
common shape — median 26 output tokens, the envelope closed by the last content chunk — is the bulk
of the file.

**What that costs.** A rate over these rows would be meaningless, and a reader who wants one has
1,845 calls to read it off instead. And the sources are recorded failures, so a failure nobody has
recorded is not here: this corpus contains no reply with a brace inside a string and none with prose
in front of the JSON, which is why `tests/test_llm_stream.py`'s synthetic cases stay where they are.
The replay test says which mutants only they can catch.

## How it is recorded, and why there is no hook in `Local.ask`

`urllib.request.urlopen` is replaced for the duration of one call by a wrapper that hands `ask` the
real response and keeps every line as it goes past — the same seam `tests/test_llm_stream.py` fakes
the server at. **Nothing in `argmind/` changes**, which is the point: a recorder that edited the
thing it records would be recording the recorder. It also means the record is the wire and not this
file's idea of the wire.

**Every row is checked at record time.** Immediately after each live call the encoded entries are
replayed through a second `Local`, stops still off, and the replayed `Reply` must match the live one
on text, reason, both token counts and `thought`; the decoded line count must match the number of
lines that came off the socket; and a fingerprint over the per-line `(thinking length, content,
done)` projection must match. A run that cannot reproduce its own call raises and writes nothing.

## What the rows are a function of

Model, `think`, the two stop flags, `num_predict`, `num_ctx`, temperature, the rendered prompt
(`q`/`r`), the fixture screens (`s`), `measure/conversation.py`'s case list (`f`) and its driving
code (`c`), this file (`h`), `argmind/core/loop.py` (`l`, the history writer), `argmind/core/llm.py`
(`m`, the request), the repeat count — **and the server**, which is in the JSON and not in the name.

**What it would fail to notice changing**: the Ollama build, the GPU, and anything else that alters
how the server splits a token stream into lines. A corpus recorded on another machine replays
identically, because the replay is offline; but a *re-recording* there could differ chunk for chunk
while every fingerprint in the name stayed put. `ollama_version` and the residency are recorded for
that reason.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import hashlib
import json
import pathlib
import sys
import time
import urllib.request

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from argmind.core import llm as llm_module                               # noqa: E402
from argmind.core.llm import OLLAMA, Local                               # noqa: E402
from argmind.core.loop import REMEMBER_TURNS                             # noqa: E402
from measure import conversation as conv                                 # noqa: E402
from measure import sending                                              # noqa: E402
from measure.capture import load                                         # noqa: E402
from measure.choosing import (prompt_fields, prompt_tag, print_spend,    # noqa: E402
                              spend_summary)

HERE = pathlib.Path(__file__).resolve().parent
CORE = pathlib.Path(llm_module.__file__).resolve().parent

#: The scenes, written before anything ran, each one a case of `measure/conversation.py` driven
#: through its own `drive()`: `(case, distance, arm, repeats, what it is here for)`.
#:
#: **The turn indices in the reasons are quoted from the recorded run, and nothing here selects a
#: turn** — whole conversations are recorded and the interesting one is simply among them. `script()`
#: is imported rather than copied, so if the filler or a case moves, the line at that index moves
#: with it and `f` in the filename forks.
CONVERSATIONS = [
    ("no-copy", 3, "without", 3,
     'the 4,096-token {"key": "tab"} loop at turn 4 — 12 of 12 in the recorded run'),
    ("no-copy", 3, "with", 2,
     "the same turn on the other arm — 6 of 12 there, so a clean draw here is expected"),
    ("rule-close", 3, "with", 2,
     "turn 3 closed its envelope and then produced 962 output tokens for a 119-character "
     "sentence: the only shape that can exercise GRACE"),
    ("no-colour", 5, "with", 1,
     "turn 7, the smaller version of the same thing — 192 output tokens for 31 characters"),
    ("ref-eraser", 1, "with", 2, "the healthy median, on the 123-widget screen"),
    ("ref-parts", 1, "with", 1, "the healthy median, on the 152-widget screen"),
]

#: One-line turns with no history, on the screen `measure/repair.py` uses for language. The first
#: three are the prompts that ran to the ceiling under `--think on` in
#: `repair-qwen3-8b-on-qa62805-s5020f7-f339a3d-h09102d-x3.json` — rows 6, 18 and 28, at 1, 2 and 1
#: of 3 repeats, so a repeat that comes back clean is the expected thing and not a miss.
SINGLES = [
    ("character-map", "여기서 뭘 할 수 있어?", "ceiling under --think on, 1 of 3 repeats"),
    ("character-map", "클릭도 돼?", "ceiling under --think on, 2 of 3 repeats"),
    ("character-map", "화면에 뭐가 보여?", "ceiling under --think on, 1 of 3 repeats"),
    ("character-map", "고마워", "the social turn: the shortest reply this agent produces"),
]

#: How many times each `SINGLES` prompt is asked away from the `off` arm. Three, because the ceiling
#: fires on one draw in three and a single draw would report whichever way it went.
SINGLE_REPEATS = 3

#: Prompts written **to elicit a shape**, and the only thing here not drawn from a measured case
#: list. They exist because of a result this file produced on its first full run and did not
#: expect: **at the shipped prompt `r322a4c`, none of the three recorded pathologies reproduced.**
#: 61 replies over the exact scenes that carried them at `qa62805` — the `no-copy` turn that ran to
#: the ceiling 12 times in 12, the `rule-close` turn that spent 962 output tokens for a
#: 119-character sentence, the `no-colour` turn that spent 192 — came back at 14 to 60 output
#: tokens, every one of them `stop`, the slowest 4.5 s and that one a cold load. The prompt moved
#: (`qa62805` -> `r322a4c`, the standing-offer slot) and so did `loop.py`'s offer tense, and
#: whatever produced the tail went with one of them. That is worth knowing on its own, and it
#: leaves the corpus with only the easy half.
#:
#: So these ask for the shapes outright: a repeated keystroke, more actions than `MOST_AT_ONCE`
#: allows, and a long answer that is not a loop. **Whether they work is reported, not assumed** —
#: the run prints what each one produced, and a prompt that comes back at 30 tokens is one that
#: elicited nothing and says so on its own line.
#:
#: The third is deliberately the *false positive* `LOOP_FLOOR` exists to avoid: a person asking for
#: something long and repetitive is the case `llm.py` names as what would say the floor is too low.
STRESS = [
    ("untitled---paint", "탭 키를 계속 눌러줘",
     'the {"key": "tab"} repetition, asked for outright — the shape of all 18 ceiling turns'),
    ("system-information", "여기 있는 항목 전부 눌러줘",
     "more actions than MOST_AT_ONCE allows, on the 152-widget screen"),
    ("untitled---paint", "1부터 200까지 세어줘",
     "a long answer that is NOT degenerate — the false positive the floor is set to avoid"),
    ("character-map", "화면에 있는 것들 이름을 전부 말해줘",
     "a long say on a 21-widget screen: the shape that closes an envelope and keeps talking"),
    ("system-information", "화면에 있는 것들 이름을 전부 말해줘", "the same request on 152 widgets"),
]

#: `(model, think, what the arm is for)`. **Four arms, four files, the model in each name.**
#:
#: **`think=False`, not `None`, and the difference cost the first recording of this file.** `None`
#: leaves the field out of the request, which on `qwen3:8b` means the model thinks: the smoke run
#: came back with 4,032 characters of reasoning and 194 output tokens on a turn
#: `measure/conversation.py` recorded at 28, under a filename that said `off`. That is
#: `docs/findings/harness-pitfalls.md`'s unset-flag row, live, in the first six calls.
#: `measure/conversation.py:916` maps its `--think off` to `False` and this maps to the same thing,
#: so the two arms are the same arm.
ARMS = {
    "off": ("qwen3:8b", False,
            "the shipped configuration: thinking off, the conversation scenes"),
    "on": ("qwen3:8b", True,
           "--think on: reasoning lines, the only way a line counts toward `arrived` while "
           "carrying no content"),
    "small": ("qwen3:0.6b", False,
              "a model small enough to answer badly, for replies that do not parse"),
    "stress": ("qwen3:8b", False,
               "the shipped model asked for the shapes the shipped prompt stopped producing"),
}

#: What each arm records: `(the conversations, the singles repeated, the stress prompts repeated)`.
#: **A table rather than branches**, because an arm that quietly recorded a different scene list
#: than its name says is the same defect as a filename that omits a condition.
COMPOSITION = {
    "off": (True, 1, 0),
    "on": (False, SINGLE_REPEATS, 0),
    "small": (True, 1, 1),
    "stress": (False, 0, 3),
}

#: What each `think` setting is called in a filename. **Three values, not two**: `None` is a real
#: third state — the field is absent from the request and the model decides — and folding it in with
#: `off` is how a corpus comes to be named for a condition it was not recorded under.
THINK_LABEL = {True: "on", False: "off", None: "unset"}

#: The recording configuration. Named, because a corpus taken without it is worthless and the one
#: mistake that produces one is forgetting a keyword.
OFF = {"stop_when_answered": False, "stop_when_looping": False}


# ---- fingerprints -----------------------------------------------------------------------------

def harness_print() -> str:
    """Six characters over this file. `h` in the name."""
    return sending.harness_print(__file__)


def ollama_version():
    """The server's own version. **Not in the filename, and that is a residual, not a decision I
    can make alone**: it is the condition most likely to move the chunking without moving anything
    else, and no other file in this directory has a letter for it. Recorded in the JSON so a corpus
    that suddenly chunks differently can be told apart from one that was asked something else."""
    try:
        with urllib.request.urlopen(f"{OLLAMA}/api/version", timeout=5) as r:
            return json.loads(r.read()).get("version")
    except Exception:                                                     # noqa: BLE001
        return None


# ---- the wire ---------------------------------------------------------------------------------

class Wire:
    """One HTTP response with every line kept as it goes past.

    A proxy and not a copy: `ask` iterates the real socket and gets the real bytes, and this sees
    them on the way. Anything that made the caller read out of a buffer would be recording a stream
    this file had assembled.
    """

    def __init__(self, inner, sink) -> None:
        self._inner, self._sink = inner, sink

    def __enter__(self):
        self._inner.__enter__()
        return self

    def __exit__(self, *exc):
        return self._inner.__exit__(*exc)

    def __iter__(self):
        for line in self._inner:
            self._sink.append(line)
            yield line


@contextlib.contextmanager
def tapped():
    """`urlopen` replaced for the duration of one call, and restored in a `finally`.

    Around one call rather than around the run, because `measure/conversation.py`'s `residency()`
    calls `.read()` on a response `Wire` does not proxy — and a tap that broke the residency read
    would take the record of what else was on the GPU with it.
    """
    real = urllib.request.urlopen
    sink = []

    def spy(*a, **kw):
        return Wire(real(*a, **kw), sink)

    urllib.request.urlopen = spy
    try:
        yield sink
    finally:
        urllib.request.urlopen = real


class Watched:
    """The model, with the raw stream of every call kept beside the `Reply`.

    Sits **inside** `measure/conversation.py`'s `Recorder`, so a retried call records the attempt
    that answered rather than the one that raised: `Recorder` calls this again and each call taps
    afresh.
    """

    def __init__(self, inner) -> None:
        self.inner = inner
        self.calls = []

    def ask(self, prompt, images=None):
        with tapped() as sink:
            t0 = time.perf_counter()
            reply = self.inner.ask(prompt, images) if images else self.inner.ask(prompt)
            took = time.perf_counter() - t0
        self.calls.append({"prompt": prompt, "reply": reply, "raw": list(sink),
                           "seconds": round(took, 3)})
        return reply


# ---- the format -------------------------------------------------------------------------------

def project(line: bytes):
    """One stream line as `Local.ask` sees it: `(thinking length, content, done)`.

    `content` is `None` for a line `ask` skips before parsing — a blank one — which is a different
    thing from a line that carried an empty string, and the two count differently toward `arrived`.
    """
    if not line.strip():
        return (0, None, False)
    d = json.loads(line)
    msg = d.get("message") or {}
    return (len(msg.get("thinking") or ""), msg.get("content") or "", bool(d.get("done")))


def stream_print(lines) -> str:
    """Twelve characters over the per-line projection of a whole stream.

    **This is what pins the chunk boundaries.** The text, both token counts and `thought` all
    survive two adjacent content chunks being joined, and joining them moves where `GRACE` stops —
    so the one quantity a corpus must not lose is the one nothing else here would notice losing.
    """
    blob = json.dumps([list(project(line)) for line in lines], ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def encode(line: bytes):
    """One stream line, reduced to what `ask` reads.

    A bare string is the common case — a content chunk and nothing else on the line — so the common
    case costs its own characters and two quotes. Everything else is an object.
    """
    if not line.strip():
        return {"blank": True}
    d = json.loads(line)
    msg = d.get("message") or {}
    thinking, content = msg.get("thinking") or "", msg.get("content") or ""
    if not thinking and not d.get("done"):
        return content
    entry = {}
    if thinking:
        #: The length, not the text. `ask` reads `len(reasoning)` and its truthiness and nothing
        #: else; the module docstring says why, and the replay test asserts it off the syntax tree
        #: rather than off the prose.
        entry["t"] = len(thinking)
    if content:
        entry["c"] = content
    if d.get("done"):
        entry["done"] = {"done_reason": d.get("done_reason"),
                         "prompt_eval_count": d.get("prompt_eval_count"),
                         "eval_count": d.get("eval_count")}
    return entry


def decode(entry) -> bytes:
    """One entry back into a line the real reader parses. The inverse of `encode` **as far as `ask`
    can tell**, which is the only sense in which anything here claims to be an inverse."""
    if isinstance(entry, str):
        return json.dumps({"message": {"role": "assistant", "content": entry},
                           "done": False}).encode()
    if entry.get("blank"):
        return b"\n"
    msg = {"role": "assistant", "content": entry.get("c", "")}
    if entry.get("t"):
        msg["thinking"] = "x" * int(entry["t"])
    out = {"message": msg, "done": bool(entry.get("done"))}
    out.update(entry.get("done") or {})
    return json.dumps(out).encode()


# ---- replaying, for this file's own check -------------------------------------------------------

class Canned:
    """The recorded stream played back to `ask` as if it were the server, counting what was read."""

    def __init__(self, lines) -> None:
        self.lines, self.read = list(lines), 0

    def __enter__(self):
        return self

    def __exit__(self, *_exc) -> bool:
        return False

    def __iter__(self):
        for line in self.lines:
            self.read += 1
            yield line


def play(entries, *, module=llm_module, **kw):
    """Replay one recorded reply. Returns `(Reply, lines read)`. No network: `urlopen` never runs.

    `tests/test_llm_replay.py` has its own copy of this and of `decode`, deliberately — the test
    must not depend on `measure/`, and each copy is checked against the same ground truth, which is
    what the live call actually returned.
    """
    stream = Canned([decode(e) for e in entries])
    real = urllib.request.urlopen
    urllib.request.urlopen = lambda *_a, **_k: stream
    try:
        return module.Local("replayed", **kw).ask("(replayed)"), stream.read
    finally:
        urllib.request.urlopen = real


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def check(row) -> None:
    """Replay the row just recorded and refuse it if it does not reproduce the call.

    **Three things, because a codec can lose exactly one of them at a time.** The text and the token
    counts catch a dropped or garbled chunk and a thinking line recorded as a content one; the line
    count catches a dropped line; and `stream_print` catches two chunks joined, which is the only
    loss that changes nothing else and still moves where `GRACE` stops.
    """
    reply, read = play(row["lines"], **OFF)
    was = row["recorded"]
    got = {"reason": reply.reason, "text_chars": len(reply.text), "text_sha": sha(reply.text),
           "prompt_tokens": reply.prompt_tokens, "out_tokens": reply.out_tokens,
           "thought": reply.thought}
    mine = {k: was[k] for k in got}
    if got != mine:
        raise RuntimeError(f"{row['key']}: the replay does not reproduce the call. "
                           f"recorded {mine}, replayed {got}")
    if read != was["stream_lines"] or len(row["lines"]) != was["stream_lines"]:
        raise RuntimeError(f"{row['key']}: {was['stream_lines']} lines came off the socket, "
                           f"{len(row['lines'])} were stored, {read} were replayed")
    again = stream_print([decode(e) for e in row["lines"]])
    if again != was["stream_print"]:
        raise RuntimeError(f"{row['key']}: the chunk boundaries did not survive the round trip — "
                           f"{was['stream_print']} recorded against {again} decoded")


# ---- recording --------------------------------------------------------------------------------

def row_of(call, key: str, **about) -> dict:
    """One recorded reply: what was asked, what came back, and every line of how it came back."""
    raw, reply = call["raw"], call["reply"]
    seen = [project(line) for line in raw]
    row = {
        "key": key,
        **about,
        "prompt_chars": len(call["prompt"]),
        "prompt_sha": sha(call["prompt"]),
        "recorded": {
            "reason": reply.reason,
            "text_chars": len(reply.text),
            "text_sha": sha(reply.text),
            "text_head": reply.text[:120],
            "prompt_tokens": reply.prompt_tokens,
            "out_tokens": reply.out_tokens,
            "thought": reply.thought,
            #: Counted here the way `ask` counts it, off the wire. It is not on `Reply` — with the
            #: stops off `out_tokens` is the server's `eval_count` — and it becomes observable from
            #: outside only when a stop fires, which is the arm the replay test asserts it in.
            "arrived": sum(1 for t, c, _d in seen if t or c),
            "stream_lines": len(raw),
            "stream_bytes": sum(len(line) for line in raw),
            "stream_print": stream_print(raw),
            "seconds": call["seconds"],
            "ttft": None if reply.ttft is None else round(reply.ttft, 3),
        },
        "lines": [encode(line) for line in raw],
    }
    check(row)
    return row


def record_conversation(model, screens, case, distance, arm, repeat, why) -> list:
    """One scripted conversation, every turn of it recorded.

    Driven by `measure/conversation.py`'s own `drive()`, so the prompts are that harness's prompts
    and not a second rendering of them. The captured calls are matched to the turns by position and
    the count is checked, because a retry that slipped a call in would otherwise shift every row's
    label by one.
    """
    watched = model.inner
    lines = conv.script(case, distance, arm)
    turns = conv.drive(model, screens[case["screen"]], lines, REMEMBER_TURNS)
    if len(watched.calls) != len(turns):
        raise RuntimeError(f"{len(turns)} turns and {len(watched.calls)} captured calls")
    rows = []
    for turn, call in zip(turns, watched.calls):
        rows.append(row_of(
            call, f"{case['id']}@d{distance}/{arm}/r{repeat}/t{turn['i']}",
            scene="conversation", case=case["id"], kind=case["kind"], screen=case["screen"],
            distance=distance, arm=arm, repeat=repeat, turn=turn["i"], said=turn["said"],
            history_lines=turn["history_lines"], window_lines=turn["window_lines"], why=why))
    del watched.calls[:]
    return rows


def record_single(model, screens, screen, said, repeat, why, scene="single") -> list:
    """One turn with no conversation behind it — `measure/repair.py`'s language shape, and the
    shape the `stress` prompts use. `scene` is part of the key, so a stress row can never be read
    as a measured one."""
    watched = model.inner
    turns = conv.drive(model, screens[screen], [said], REMEMBER_TURNS)
    if len(watched.calls) != 1:
        raise RuntimeError(f"one turn and {len(watched.calls)} captured calls")
    row = row_of(watched.calls[0], f"{scene}/{screen}/{said}/r{repeat}",
                 scene=scene, case=scene,
                 kind="language" if scene == "single" else "stress", screen=screen,
                 distance=0, arm="none", repeat=repeat, turn=0, said=said,
                 history_lines=turns[0]["history_lines"],
                 window_lines=turns[0]["window_lines"], why=why)
    del watched.calls[:]
    return [row]


def scenes(arm: str, only):
    """What this arm records, as `(label, what)` pairs, resolved before the model loads."""
    conversations, singles, stress = COMPOSITION[arm]
    plan = []
    if conversations:
        for case_id, distance, side, repeats, why in CONVERSATIONS:
            case = next((c for c in conv.CASES if c["id"] == case_id), None)
            if case is None:
                raise RuntimeError(f"measure/conversation.py has no case {case_id!r} — the scene "
                                   "list and the case list have drifted apart")
            for repeat in range(1 if arm == "small" else repeats):
                plan.append((f"{case_id}@d{distance}/{side}/r{repeat}",
                             ("conversation", case, distance, side, repeat, why)))
    for screen, said, why in SINGLES:
        for repeat in range(singles):
            plan.append((f"single/{screen}/{said}/r{repeat}",
                         ("single", screen, said, repeat, why)))
    for screen, said, why in STRESS:
        for repeat in range(stress):
            plan.append((f"stress/{screen}/{said}/r{repeat}",
                         ("stress", screen, said, repeat, why)))
    if only:
        plan = [(label, what) for label, what in plan if only in label]
        if not plan:
            raise RuntimeError(f"--only {only!r} selected no scene")
    return plan


def wanted_screens(plan) -> set:
    return {(what[1]["screen"] if what[0] == "conversation" else what[1]) for _l, what in plan}


# ---- the file ---------------------------------------------------------------------------------

def result_path(a, think, stops_off: bool) -> pathlib.Path:
    """Where this arm writes, resolved before the model is loaded.

    **The stop flags are in the name**, because a corpus recorded with them on is a corpus of the
    prefixes the shipped logic already produces and must never be mistaken for this one. **So is
    the arm**, and it was not for one draft: `off` and `stress` are the same model at the same
    think setting over different scenes, so they produced the same path, and the second would have
    overwritten the first. That is the fifth instance of this shape in this directory and it was
    caught by `--plan` rather than by the refusal, which only fires once a file exists. The rest
    follows the conventions of the other result files — model, think, `q`, `s`, `f`, `h`, `l` — with
    two additions: `c` for `measure/conversation.py`, which supplies both the cases and the driving,
    and `m` for `argmind/core/llm.py`, which is the request itself. `l` stays `loop.py`, because two
    other harnesses already spend that letter on it and a letter meaning one file here and another
    there is how a comparison gets lost.
    """
    parts = [a.model.replace(":", "-"), THINK_LABEL[think], a.arm, prompt_tag(),
             f"s{conv.screen_print()}", f"f{conv.data_print()}", f"h{harness_print()}",
             f"c{sending.harness_print(conv.__file__)}",
             f"l{sending.norm_print(CORE / 'loop.py')}",
             f"m{sending.norm_print(CORE / 'llm.py')}",
             "stopsoff" if stops_off else "STOPSON", f"x{a.repeat}"]
    if a.only:
        parts.append("".join(ch for ch in a.only if ch.isalnum() or ch in "-_")[:24])
    return HERE / ("replies-" + "-".join(parts) + ".json")


def coverage(rows) -> dict:
    """Which shapes the corpus reaches, counted by replaying each row through the **shipped** stops.

    Printed rather than asserted here: one arm is not expected to reach all of them, and it is the
    union across the files that has to. `tests/test_llm_replay.py` is where that is a failure.
    """
    got = collections.Counter()
    for row in rows:
        shipped, _read = play(row["lines"])
        got["shipped: " + str(shipped.reason)] += 1
        if row["recorded"]["reason"] == "length":
            got["ran to the ceiling with the stops off"] += 1
        if shipped.json() is None:
            got["no object to parse"] += 1
        if row["recorded"]["thought"]:
            got["carried reasoning"] += 1
        if row["recorded"]["arrived"] < row["recorded"]["stream_lines"] - 1:
            got["a line carrying neither"] += 1
    return dict(got)


def disagreement(rows) -> list:
    """Prompts asked more than once, and whether the answer came back the same.

    **The worst of N is not a summary this file can print** — it stores replies, not verdicts — so
    what it owes a reader instead is how many repeats of one prompt disagreed. A ceiling that fires
    on one draw in three is exactly the thing a single draw would report wrongly either way.
    """
    by_prompt = collections.defaultdict(list)
    for row in rows:
        by_prompt[(row["case"], row["arm"], row["distance"], row["turn"], row["said"])].append(row)
    out = []
    for what, group in sorted(by_prompt.items(), key=lambda kv: str(kv[0])):
        if len(group) < 2:
            continue
        out.append({"prompt": f"{what[0]}/{what[1]}/d{what[2]}/t{what[3]} {what[4]}",
                    "said": what[4],
                    "n": len(group),
                    "identical_streams": len({r["recorded"]["stream_print"] for r in group}) == 1,
                    "reasons": [r["recorded"]["reason"] for r in group],
                    "out_tokens": [r["recorded"]["out_tokens"] for r in group]})
    return out


def spend_of(row) -> dict:
    """`measure/choosing.py`'s `spend()` shape, off a recorded row rather than off a `Reply`."""
    r = row["recorded"]
    return {"seconds": r["seconds"], "out_tokens": r["out_tokens"], "reason": r["reason"],
            "truncated": r["reason"] == "length", "thought_chars": r["thought"]}


def show(row) -> None:
    r = row["recorded"]
    print(f"    {row['key'][:46]:<46} {str(r['reason']):<10} out {str(r['out_tokens']):>5}  "
          f"lines {r['stream_lines']:>5}  {r['seconds']:>6.2f}s  {r['text_chars']:>5} chars")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m measure.replies",
                                description=__doc__.split("\n\n")[0])
    p.add_argument("--arm", choices=tuple(ARMS), default="off")
    p.add_argument("--model", default=None, help="overrides the arm's model; goes in the name")
    p.add_argument("--repeat", type=int, default=1,
                   help="whole-plan repeats. The per-scene counts in CONVERSATIONS are part of the "
                        "experiment; this multiplies them")
    p.add_argument("--only", default=None, metavar="TEXT",
                   help="record only scenes whose label contains TEXT — one case while writing")
    p.add_argument("--plan", action="store_true", help="print what would be recorded and stop")
    p.add_argument("--stops-on", action="store_true",
                   help="record with the shipped stops IN FORCE. **That produces a corpus which "
                        "cannot test the stops**: the stream is cut at the stop and everything "
                        "after it is never seen. Here so that the flag is a decision somebody "
                        "typed, and the file it writes says STOPSON in its name")
    a = p.parse_args(argv)
    model_name, think, why_arm = ARMS[a.arm]
    a.model = a.model or model_name
    stops_off = not a.stops_on

    plan = scenes(a.arm, a.only) * a.repeat
    out = result_path(a, think, stops_off)
    print(f"\n  RECORDING REPLIES — {a.arm}: {why_arm}")
    print(f"    model {a.model}   think {think}   stops "
          f"{'OFF (the whole point)' if stops_off else '** ON — this corpus tests nothing **'}")
    for label, _what in plan:
        print(f"      {label}")
    print(f"\n    writes {out.name}")
    if a.plan:
        return 0
    if out.exists():
        print(f"\n  ** REFUSING TO WRITE: {out.name} already exists. **")
        print("     Every condition this file puts in a name matches a corpus already on disk.")
        print("     Move it aside rather than deleting it: tests/test_llm_replay.py replays every")
        print("     replies-*.json it finds and its golden names the rows it expects.")
        return 2

    screens = {name: widgets for name, _window, widgets in load()}
    missing = sorted(wanted_screens(plan) - set(screens))
    if missing:
        print(f"  no fixture {missing} — run python -m measure.capture")
        return 2

    local = Local(a.model, think=think, **({} if a.stops_on else OFF))
    watched = Watched(local)
    model = conv.Recorder(watched)
    before, version = conv.residency(), ollama_version()
    started = time.time()
    rows = []
    print()
    for label, what in plan:
        if what[0] == "conversation":
            _kind, case, distance, side, repeat, why = what
            made = record_conversation(model, screens, case, distance, side, repeat, why)
        else:
            kind, screen, said, repeat, why = what
            made = record_single(model, screens, screen, said, repeat, why, scene=kind)
        for row in made:
            show(row)
        rows.extend(made)
    took = round(time.time() - started, 1)
    after = conv.residency()
    local.release()

    summary = spend_summary([(r["key"], spend_of(r)) for r in rows], ceiling=local.max_tokens)
    reached = coverage(rows)
    repeats = disagreement(rows)
    print_spend(summary)
    print("\n  WHAT THE CORPUS REACHES — every row replayed through the shipped stops")
    for shape, n in sorted(reached.items()):
        print(f"    {shape:<40} {n}")
    print("\n  REPEATS OF ONE PROMPT — identical streams, or not")
    for d in repeats:
        print(f"    {d['prompt'][:38]:<38} n={d['n']}  "
              f"{'identical' if d['identical_streams'] else 'DIFFERENT'}  "
              f"reasons {d['reasons']}  out {d['out_tokens']}")

    body = {
        "what": "recorded model replies, chunk by chunk, for tests/test_llm_replay.py",
        "arm": a.arm, "why_arm": why_arm, "model": a.model, "think": think,
        "stop_when_answered": local.stop_when_answered,
        "stop_when_looping": local.stop_when_looping,
        "max_tokens": local.max_tokens, "ctx": local.ctx, "temperature": 0,
        **prompt_fields(),
        "screens_print": conv.screen_print(), "data_print": conv.data_print(),
        "harness_print": harness_print(),
        "conversation_print": sending.harness_print(conv.__file__),
        "loop_print": sending.norm_print(CORE / "loop.py"),
        "llm_print": sending.norm_print(CORE / "llm.py"),
        "remember_turns": REMEMBER_TURNS,
        "ollama_version": version, "ollama_before": before, "ollama_after": after,
        "seconds": took, "calls": model.calls, "retried": model.retried,
        "scenes": [label for label, _w in plan],
        "composition": dict(zip(("conversations", "singles", "stress"), COMPOSITION[a.arm])),
        "conversations": CONVERSATIONS, "singles": SINGLES, "stress": STRESS,
        "spend": summary, "coverage": reached, "repeats": repeats,
        "rows": rows,
    }
    out.write_text(json.dumps(body, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n  wrote {out.name}  ({out.stat().st_size / 1e6:.2f} MB, {len(rows)} replies, "
          f"{took} s)")
    print("  Next: ARGMIND_BLESS_REPLAY=re-bless python -m pytest tests/test_llm_replay.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
