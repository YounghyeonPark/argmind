# SPDX-License-Identifier: Apache-2.0
"""Recorded replies, replayed through the stop logic — the equivalence test `llm.py` did not have.

**No model, no server, no network.** Every byte comes off disk: `measure/replies-*.json` holds real
`qwen3:8b` and `qwen3:0.6b` streams, recorded chunk by chunk with the stops switched **off**, so the
whole stream survives and the shipped reader can be run over it here. `measure/replies.py` records
them and says why the stops have to be off; this file is the other end.

A corpus is named `replies-<model>-<think>-<arm>-<q>-s-f-h-c-l-m-stopsoff-x<n>.json`, and the
second and third segments repeat a word on two of the four files: the `off` arm is recorded with
thinking off, so it reads `qwen3-8b-off-off`, and the thinking arm reads `qwen3-8b-on-on`. They are
two conditions that happen to share a word — `--arm stress` is thinking-off as well, and `--arm
small` is a different model at the same setting — and dropping either segment would lose a
condition. **This paragraph lives here and not in `measure/replies.py` because that file's sha is
in every corpus name**: a comment added to it after a recording makes the recording's own `h`
underivable, which is the failure `measure/conversation.py`'s `LEGACY` block records.

`docs/findings/porting-the-core.md` Stage 0: `tests/test_loop_golden.py` is the golden transcript for
`loop.py` and this is the corpus for `llm.py`. Stage 2 — reply parsing — was blocked on it.

## The arms, and which of them is blessed

1. **The replay reproduces the call.** Feed the recorded chunks back with the stops off and the
   `Reply` must equal the one the live call returned, field for field, down to the sha of the text.
   Nothing is blessed here: the comparison is with what the server actually did, recorded beside the
   stream. This also validates the decoder below — the corpus is encoded by `measure/replies.py` and
   decoded here, deliberately twice, because a test that imported the harness would be checking a
   codec against itself.
2. **The shipped stops do what `tests/golden/llm-replay.txt` says.** Reason, text, where it stopped,
   how many lines it read, both token counts, and what `Reply.json()` parses out. This is the golden
   arm, it was blessed off the Python implementation, and **it is what a Rust port has to
   reproduce.**
3. **The answered stop on its own, on the rows where it lands differently.** `Local` has two flags
   and the golden's second section is the second configuration, kept because of a fact about the
   corpus: **no recorded reply reaches `GRACE` under the shipped pair.** Two do reach it with the
   loop stop switched off — a reply that closes its object and then produces 505 more content chunks
   — and without that section `GRACE` would be pinned by nothing at all. The section holds only the
   rows where the two configurations disagree, so the *set* of such rows is part of the golden too.
4. **The stops change no answer.** `Reply.json()` under the shipped stops equals `Reply.json()` with
   them off, on every recorded reply. `tests/test_llm_stream.py` asserts that over eight strings
   somebody wrote; this asserts it over 128 real streams and 29,710 real stream
   lines.

## What is asserted behaviourally, and the one thing that is not

A port has to be able to pass this without being Python, so everything is asked of `Local` from
outside: `reply.text`, `reply.reason`, `reply.out_tokens`, `reply.prompt_tokens`, `reply.thought`,
`reply.json()`, and **how many lines were read off the socket**, which is the transport's business
rather than the language's.

`arrived` is the exception, and it is worth stating exactly. It is a local of `ask` with no
accessor, so it is observable **only through `out_tokens` when one of the two stops fires** — which
is precisely when the server's `eval_count` never arrives and `arrived` becomes the reported count.
On a reply that runs to the end nothing outside can see it, and this file does not pretend to check
it there. `test_ARRIVED_...` asserts it where it is visible and says so.

## Blessing

    ARGMIND_BLESS_REPLAY=re-bless python -m pytest tests/test_llm_replay.py

**A diff in the golden is a change in what the stops do.** The blessing run fails after writing, so
no invocation both re-blesses and reports green, and the file carries a tag over its own body so a
hand edit fails as a hand edit rather than as a difference in behaviour. Re-blessing is not how a
port is made to pass: the port is what has to match. Adding a corpus is not a re-bless of anything
either — new rows arrive in the diff with nothing beside them, which is what a reviewer should see.

## Can any of it fail? Fifteen mutants, and the survivors are named

`test_WHICH_MUTANTS_THE_CORPUS_CATCHES` rewrites `llm.py` in memory — `GRACE`, the four loop
constants, the scanner's string and escape tracking, the `arrived` counter, both stop flags, the
extractor's bounds — and replays the whole corpus against each on all three arms. A corpus of
well-formed replies would let most of them through, which is the flattering result this directory
exists to refuse. **Three survive**, each is a real gap in what recorded output contains, and
`SURVIVORS` says what would kill it. Nothing is written into `argmind/`: a mutant is a compiled
string.
"""

from __future__ import annotations

import ast
import functools
import hashlib
import json
import os
import pathlib
import re
import types
import urllib.request

import pytest

from argmind.core import llm as shipped
from argmind.core.llm import CEILING, CLOSED, DEGENERATE, FINISHED

ROOT = pathlib.Path(__file__).resolve().parents[1]
MEASURE = ROOT / "measure"
SOURCE = pathlib.Path(shipped.__file__).resolve()
GOLDEN = ROOT / "tests" / "golden" / "llm-replay.txt"

BLESS = ("ARGMIND_BLESS_REPLAY", "re-bless")
SENTINEL = "#---- expectations ----"

#: The two sections of the golden. The second exists because the first cannot reach `GRACE`.
SHIPPED_ARM = "under the shipped stops"
GRACE_ARM = "under the answered stop alone, on the rows where that differs"

#: Everything `measure/replies.py` preserves off a stream line. `test_the_CORPUS_KEEPS_...` reads the
#: keys `ask` actually looks up out of its syntax tree and checks them against this, so a `llm.py`
#: that started reading an eighth field fails here rather than being quietly unrecorded.
RECORDED_KEYS = {"message", "thinking", "content", "done",
                 "done_reason", "prompt_eval_count", "eval_count"}

#: A reply longer than this is pinned by its length and its sha rather than by its text. The bound
#: is a readability choice and nothing rests on it: the sha is over the whole string either way.
SHOWN = 400


def safe(text) -> str:
    """A printable form of a row key. **A test whose report crashes under `-s` is a test somebody
    turns off**: half these keys carry the Korean line that produced the reply, `print` writes to a
    cp1252 console when pytest is not capturing, and the escape is cheaper than either losing the
    key or reconfiguring somebody else's stdout from inside a test."""
    return str(text).encode("ascii", "backslashreplace").decode("ascii")


#: The fields the live call was recorded on, and therefore the ones arm 1 can compare.
LIVE = ("reason", "text_chars", "text_sha", "prompt_tokens", "out_tokens", "thought",
        "stream_lines")


# ---- the corpus ---------------------------------------------------------------------------------

def decode(entry) -> bytes:
    """One stored entry back into the line the server sent, as far as `ask` can tell.

    **A second implementation on purpose.** `measure/replies.py` holds the encoder and its own copy
    of this; `tests/` must not import a harness, and two copies checked against the same ground
    truth — what the live call returned — is stronger here than one copy checked against itself.
    """
    if isinstance(entry, str):
        return json.dumps({"message": {"role": "assistant", "content": entry},
                           "done": False}).encode()
    if entry.get("blank"):
        return b"\n"
    msg = {"role": "assistant", "content": entry.get("c", "")}
    if entry.get("t"):
        #: The reasoning was recorded as its length. `test_THE_REASONING_TEXT_...` keeps that
        #: honest: `ask` may read `len(reasoning)` and its truthiness, and nothing else.
        msg["thinking"] = "x" * int(entry["t"])
    out = {"message": msg, "done": bool(entry.get("done"))}
    out.update(entry.get("done") or {})
    return json.dumps(out).encode()


def project(line: bytes):
    """One line as `ask` sees it: `(thinking length, content, done)`. `content` is `None` for a line
    it skips before parsing, which is a different thing from a line carrying an empty string."""
    if not line.strip():
        return (0, None, False)
    d = json.loads(line)
    msg = d.get("message") or {}
    return (len(msg.get("thinking") or ""), msg.get("content") or "", bool(d.get("done")))


def stream_print(lines) -> str:
    """The chunk-boundary fingerprint `measure/replies.py` took off the wire, recomputed off the
    decode. Two adjacent content chunks joined is the one loss that changes no other field here and
    still moves where `GRACE` stops."""
    blob = json.dumps([list(project(line)) for line in lines], ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


class Canned:
    """The recorded stream, played back as if it were the server, counting what was read.

    **The count is half the instrument**, exactly as in `tests/test_llm_stream.py`: every other
    assertion here would pass if `ask` read the whole stream and threw the tail away, and where it
    stopped is the thing under test.
    """

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


def play(lines, *, module=shipped, **kw):
    """Replay one recorded reply through `module.Local`. Returns `(Reply, lines read)`."""
    stream = Canned(lines)
    real = urllib.request.urlopen
    urllib.request.urlopen = lambda *_a, **_k: stream
    try:
        return module.Local("replayed", **kw).ask("(replayed)"), stream.read
    finally:
        urllib.request.urlopen = real


OFF = {"stop_when_answered": False, "stop_when_looping": False}
ANSWERED_ONLY = {"stop_when_looping": False}


def _load():
    """Every corpus on disk as `(stem, body, rows)`, with any recorded **under** the stops excluded
    by their own contents and cross-checked against their names."""
    out, refused = [], []
    for path in sorted(MEASURE.glob("replies-*.json")):
        body = json.loads(path.read_text(encoding="utf-8"))
        off = not body["stop_when_answered"] and not body["stop_when_looping"]
        named = "STOPSON" not in path.name
        assert off == named, (
            f"{path.name} says the stops were {'off' if off else 'on'} inside and "
            f"{'off' if named else 'on'} in its name. One of the two is a lie and neither can be "
            "trusted until somebody says which")
        (out if off else refused).append((path.stem, body, body["rows"]))
    return out, refused


CORPORA, RECORDED_WITH_STOPS_ON = _load()
#: Decoded once: the ceiling rows are ~4,100 lines each and the mutation table replays every row
#: fifteen times over three arms.
LINES = {(stem, row["key"]): [decode(e) for e in row["lines"]]
         for stem, _body, rows in CORPORA for row in rows}
ROWS = [(stem, row) for stem, _body, rows in CORPORA for row in rows]


def outcome(stem, row, *, module=shipped, **kw) -> dict:
    """What one recorded stream produces under one reader configuration."""
    reply, read = play(LINES[(stem, row["key"])], module=module, **kw)
    got = reply.json()
    rendered = None if got is None else json.dumps(got, ensure_ascii=False, sort_keys=True)
    return {
        "reason": reply.reason,
        "read": [read, len(row["lines"])],
        "out_tokens": reply.out_tokens,
        "prompt_tokens": reply.prompt_tokens,
        "thought": reply.thought,
        "text_chars": len(reply.text),
        "text_sha": hashlib.sha256(reply.text.encode("utf-8")).hexdigest()[:16],
        "text": reply.text if len(reply.text) <= SHOWN else None,
        "json": got if rendered is not None and len(rendered) <= SHOWN else None,
        "json_sha": None if rendered is None
        else hashlib.sha256(rendered.encode("utf-8")).hexdigest()[:16],
    }


def as_live(stem, row, *, module=shipped) -> dict:
    """The stops-off replay, in the shape the live call was recorded in."""
    reply, read = play(LINES[(stem, row["key"])], module=module, **OFF)
    return {"reason": reply.reason, "text_chars": len(reply.text),
            "text_sha": hashlib.sha256(reply.text.encode("utf-8")).hexdigest()[:16],
            "prompt_tokens": reply.prompt_tokens, "out_tokens": reply.out_tokens,
            "thought": reply.thought, "stream_lines": read}


def expectations(*, module=shipped) -> dict:
    """Both golden sections. The second holds only the rows the two configurations disagree on."""
    ship = {stem: {row["key"]: outcome(stem, row, module=module) for row in rows}
            for stem, _body, rows in CORPORA}
    grace = {}
    for stem, row in ROWS:
        alone = outcome(stem, row, module=module, **ANSWERED_ONLY)
        if alone != ship[stem][row["key"]]:
            grace[f"{stem}/{row['key']}"] = alone
    return {SHIPPED_ARM: ship, GRACE_ARM: grace}


@functools.lru_cache(maxsize=1)
def golden_now() -> dict:
    """The shipped expectations, computed once. Several tests and fifteen mutants ask for them."""
    return expectations()


# ---- the corpus is there, and it is the right kind ------------------------------------------------

def test_the_CORPUS_IS_ON_DISK_and_was_recorded_with_the_STOPS_OFF():
    """**The one mistake that would make everything below vacuous.** A corpus recorded with the
    stops in force holds only the prefixes the shipped logic already produces, so every assertion
    would pass while exercising nothing. The flags are inside each file and repeated in its name;
    `_load()` refuses a file whose two disagree, and this refuses an empty directory.
    """
    assert CORPORA, (
        f"no corpus in {MEASURE}. Record one: python -m measure.replies --arm off. Without it this "
        "file asserts nothing and llm.py's stop logic has no equivalence test at all")
    assert not RECORDED_WITH_STOPS_ON, (
        "these corpora were recorded with the stops in force and are being ignored: "
        f"{[stem for stem, _b, _r in RECORDED_WITH_STOPS_ON]}. They cannot exercise the stop "
        "logic — the stream was cut at the stop — so they are not replayed here. Delete them or "
        "move them out of measure/ rather than leaving a corpus that looks like evidence")
    ceiling = shipped.Local("x").max_tokens
    for stem, body, rows in CORPORA:
        assert rows, f"{stem} carries no replies"
        assert body["stop_when_answered"] is False and body["stop_when_looping"] is False
        assert body["max_tokens"] == ceiling, (
            f"{stem} was recorded at num_predict {body['max_tokens']} and the shipped ceiling is "
            f"{ceiling}; a corpus taken at another ceiling truncates somewhere else")


#: What the corpus has to reach, and what it does not reach — **both, because the second list is
#: what the first one is worth.** Each entry is a predicate over `(row, shipped outcome)`.
MUST_REACH = {
    "the server finished on its own": lambda r, s: s["reason"] == FINISHED,
    "it ran to the 4,096-token ceiling": lambda r, s: r["recorded"]["reason"] == CEILING,
    "the loop detector stopped it": lambda r, s: s["reason"] == DEGENERATE,
    "there was no object to parse": lambda r, s: s["json_sha"] is None,
    "the reply carried reasoning": lambda r, s: r["recorded"]["thought"] > 0,
}

#: **Absent, and each absence is a measurement.** `closed` was 27 of 1,845 turns at prompt `qa62805`
#: — an envelope closed, then hundreds of tokens of chatter — and at `r322a4c` neither model does it
#: any more: `measure/replies.py`'s stress arm asked for it outright, on both models, and
#: `qwen3:8b` emitted **nothing at all** after its closing brace on any prompt tried, while
#: `qwen3:0.6b` emitted exactly one chunk. `GRACE` is 24, so under the shipped pair it is never
#: reached, which is why the golden carries a second section for the one configuration that does
#: reach it. If this assertion starts failing, the shape has come back and the second section can go.
MUST_NOT_REACH = {
    "the envelope closed and it kept talking past the grace":
        lambda r, s: s["reason"] == CLOSED,
}


def test_the_CORPUS_REACHES_THE_SHAPES_IT_EXISTS_FOR():
    """**A cheap corpus of well-formed replies is the flattering result available here**, and this
    refuses it. Every shape below is a branch of `ask` or of `Reply` that nothing else in this suite
    reaches with real output.

    A shape going missing means either a recording that did not reach it or a corpus file that was
    moved; both weaken the mutation table, and the mutation table is what says these assertions
    bite. A shape in the second list arriving means the model's output has changed shape, which is
    worth a run of `measure/replies.py` rather than an edit here.
    """
    got = dict.fromkeys(list(MUST_REACH) + list(MUST_NOT_REACH), 0)
    for stem, row in ROWS:
        shape = golden_now()[SHIPPED_ARM][stem][row["key"]]
        for name, hit in {**MUST_REACH, **MUST_NOT_REACH}.items():
            got[name] += bool(hit(row, shape))
    missing = [name for name in MUST_REACH if not got[name]]
    arrived = [name for name in MUST_NOT_REACH if got[name]]
    assert not missing, (
        f"the corpus reaches none of {missing}. Counts: {got}. The ceiling comes from `--arm on` "
        "and `--arm small`, the loop from `--arm stress`, reasoning from `--arm on`")
    assert not arrived, (
        f"{arrived} is in the corpus now and the comment on MUST_NOT_REACH says it was not. Read "
        "that comment: the second section of the golden exists only because of the absence")
    print("\n  shapes:", got)


# ---- arm 1: the replay reproduces the call --------------------------------------------------------

def test_EVERY_RECORDED_REPLY_REPLAYS_AS_THE_CALL_THAT_WAS_RECORDED():
    """**Nothing here is blessed.** The comparison is against what the live server did, stored
    beside the stream by `measure/replies.py` at record time — so this is the arm that says the
    corpus is a recording and not a fixture somebody wrote, and it is what validates the decoder.

    A failure means the stored chunks no longer produce the reply they produced on the machine that
    recorded them: either the corpus was edited, or `ask` reads a stream differently now.
    """
    bad = []
    for stem, row in ROWS:
        got = as_live(stem, row)
        want = {k: row["recorded"][k] for k in LIVE}
        if got != want:
            bad.append(f"{stem}/{row['key']}: recorded {want}, replayed {got}")
    assert not bad, "the corpus no longer replays as itself:\n  " + "\n  ".join(bad[:8])


def test_THE_CHUNK_BOUNDARIES_SURVIVED_being_written_down():
    """The one loss that changes no other field. Text, token counts and `thought` are all identical
    when two adjacent content chunks are joined, and joining them moves where `GRACE` stops — so
    `measure/replies.py` fingerprinted the per-line projection off the wire and this recomputes it
    off the decode."""
    bad = [f"{stem}/{row['key']}" for stem, row in ROWS
           if stream_print(LINES[(stem, row["key"])]) != row["recorded"]["stream_print"]]
    assert not bad, ("the chunk boundaries on disk are not the ones that were recorded, so GRACE "
                     f"would be exercised in the wrong places: {[safe(b) for b in bad[:8]]}")
    #: The recorder counted `arrived` off the wire the way `ask` counts it. Recomputing it here
    #: from the stored lines is the other half of that claim, and it is the number the stopped
    #: rows above report as `out_tokens`.
    miscounted = [f"{stem}/{row['key']}" for stem, row in ROWS
                  if sum(1 for t, c, _d in map(project, LINES[(stem, row["key"])]) if t or c)
                  != row["recorded"]["arrived"]]
    assert not miscounted, (
        f"the recorded `arrived` is not what these lines carry: {[safe(m) for m in miscounted]}")


# ---- arm 2: the golden ----------------------------------------------------------------------------

def tag(body: str) -> str:
    return "v" + hashlib.sha256(body.encode("utf-8")).hexdigest()[:6]


def header(body: str) -> str:
    return ("# argmind — what argmind/core/llm.py's stops do to every recorded reply in\n"
            "# measure/replies-*.json. GENERATED. Do not edit by hand: the tag below is taken over\n"
            "# the body, so a hand edit fails as a hand edit and not as a change in behaviour.\n"
            "#\n"
            f"# tag {tag(body)}\n"
            "#\n"
            f"# Re-bless:  {BLESS[0]}={BLESS[1]} python -m pytest tests/test_llm_replay.py\n"
            "# A diff below is a change in what the stops do. It is what a port has to reproduce.\n"
            "# `read` is [lines consumed, lines in the stream]; `text` and `json` are written out\n"
            f"# when they are at most {SHOWN} characters and pinned by their sha either way.\n"
            f"{SENTINEL}\n")


def split(text: str) -> tuple[str, str]:
    head, sentinel, body = text.partition(SENTINEL + "\n")
    assert sentinel, f"{GOLDEN} has no {SENTINEL!r} line; it is not a golden this can read"
    return head, body


def render(what: dict) -> str:
    return json.dumps(what, indent=1, sort_keys=True, ensure_ascii=False) + "\n"


def test_the_SHIPPED_STOPS_DO_WHAT_THE_GOLDEN_SAYS():
    """**The equivalence test a port has to pass**, and the only thing in this file that is blessed
    rather than measured. Per recorded reply: what the reader ended on, the text it kept, how much
    of the stream it consumed, both token counts, and what `Reply.json()` parsed out.

    A failure is not a broken test. It is the stop logic doing something else to a stream that has
    not changed, and the first differing line says which reply and which field.
    """
    fresh = render(expectations())
    if os.environ.get(BLESS[0]) == BLESS[1]:
        was = tag(split(GOLDEN.read_text(encoding="utf-8"))[1]) if GOLDEN.exists() else "(none)"
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(header(fresh) + fresh, encoding="utf-8", newline="\n")
        pytest.fail(f"re-blessed {GOLDEN.name}: {was} -> {tag(fresh)}. This run does not pass, on "
                    "purpose — read the diff, and commit it apart from whatever changed the "
                    "behaviour it records.")
    assert GOLDEN.exists(), (
        f"{GOLDEN} is missing. It is the record of what the stops do; create it with "
        f"{BLESS[0]}={BLESS[1]} python -m pytest {pathlib.Path(__file__).name}")
    stored = split(GOLDEN.read_text(encoding="utf-8"))[1]
    if stored != fresh:
        pairs = zip(stored.splitlines(), fresh.splitlines())
        first = next((n for n, (a, b) in enumerate(pairs, 1) if a != b),
                     min(len(stored.splitlines()), len(fresh.splitlines())) + 1)
        assert stored == fresh, (
            f"the stops no longer do what {GOLDEN.name} records — first difference at line "
            f"{first}, stored {tag(stored)} against {tag(fresh)} now. If the change was decided, "
            "re-bless in a commit of its own; if it was not, this is the regression.")


def test_the_GOLDEN_HAS_NOT_BEEN_EDITED_BY_HAND():
    """One line edited to match new behaviour would otherwise be indistinguishable from a line that
    was always there. The tag is over the body, so an edit outside the blessing path forks the two
    and says so."""
    assert GOLDEN.exists(), f"{GOLDEN} is missing"
    head, body = split(GOLDEN.read_text(encoding="utf-8"))
    stated = re.search(r"^# tag (v[0-9a-f]{6})$", head, re.MULTILINE)
    assert stated, f"{GOLDEN.name} carries no tag line; it cannot vouch for its own body"
    assert stated.group(1) == tag(body), (
        f"{GOLDEN.name} says it is {stated.group(1)} and its body is {tag(body)} — it has been "
        "edited by hand. Do not correct the tag: re-bless, so the change is a diff somebody chose.")
    assert BLESS[0] in head and BLESS[1] in head


def test_the_GOLDEN_COVERS_EVERY_REPLY_IN_EVERY_CORPUS():
    """A corpus file moved out of `measure/` would otherwise make this suite quietly smaller: the
    golden would still match, over fewer rows. So the row sets are compared both ways."""
    stored = json.loads(split(GOLDEN.read_text(encoding="utf-8"))[1])[SHIPPED_ARM]
    on_disk = {stem: {row["key"] for row in rows} for stem, _body, rows in CORPORA}
    assert set(stored) == set(on_disk), (
        f"the golden covers {sorted(set(stored) - set(on_disk))}, which is not on disk, and misses "
        f"{sorted(set(on_disk) - set(stored))}, which is")
    for stem, keys in on_disk.items():
        assert set(stored[stem]) == keys, (
            f"{stem}: the golden and the corpus disagree about which replies exist — "
            f"{sorted(set(stored[stem]) ^ keys)[:6]}")


def test_GRACE_IS_PINNED_BY_THE_ROWS_THE_LOOP_STOP_REACHES_FIRST():
    """**Without this the corpus would say nothing about `GRACE` at all**, and `GRACE` is one of the
    three things it exists to pin.

    Two recorded replies close their object and then produce 505 more content chunks — a person
    asking `qwen3:8b` to count from 1 to 200 — and under the shipped pair the loop detector reaches
    them first, at 107 lines, so `closed` never happens. With the loop stop switched off the same
    streams stop at the grace, and that is the second section of the golden. The count of such rows
    is asserted here so that the section cannot quietly become empty.
    """
    grace = golden_now()[GRACE_ARM]
    reached = [key for key, got in grace.items() if got["reason"] == CLOSED]
    assert reached, (
        "no recorded reply reaches GRACE under any configuration, so the three GRACE mutants below "
        "are unkillable and the constant is pinned by tests/test_llm_stream.py's synthetic stream "
        "alone. Record a reply that closes its object and keeps talking")
    print(f"\n  GRACE reached on {len(reached)} of {len(ROWS)} replies: "
          f"{[safe(k) for k in reached]}")


# ---- arm 4: the stops change no answer ------------------------------------------------------------

def test_the_STOPS_CHANGE_NO_ANSWER_ON_ANY_RECORDED_REPLY():
    """**`Local.ask`'s load-bearing claim, on real output for the first time.** Its docstring says
    the request is byte-identical and only the reading stops early, so nothing about what the model
    *chooses* can move. `tests/test_llm_stream.py` asserts that over eight strings somebody wrote;
    this asserts it over every recorded stream, including the three that ran to the ceiling.

    Where it would break: a stop that cut a reply at a point `Reply.json()` then refuses. It cannot
    here because both use `_Balanced` — which is a claim about one scanner, and the mutation table
    attacks that scanner directly.
    """
    bad = []
    for stem, row in ROWS:
        with_stops, _r = play(LINES[(stem, row["key"])])
        without, _r2 = play(LINES[(stem, row["key"])], **OFF)
        if with_stops.json() != without.json():
            bad.append(f"{stem}/{row['key']}: stopped -> {with_stops.json()!r}, "
                       f"whole -> {without.json()!r}")
    assert not bad, ("a stop changed the answer, which is latency bought with accuracy:\n  "
                     + "\n  ".join(bad[:8]))


def test_ARRIVED_is_what_out_tokens_falls_back_to_WHERE_IT_IS_OBSERVABLE():
    """`arrived` is a local of `ask` with no accessor, so it is visible from outside **only when a
    stop fires** — that is when the server's final message never arrives and `arrived` becomes the
    reported `out_tokens`. This asserts it exactly there, over the lines the reader consumed, and
    asserts nothing about the replies that ran to the end, where nothing outside can see it.

    **A port cannot be checked on the unobservable half either**, which is the honest limit of this
    corpus rather than a gap in it.
    """
    seen = 0
    for stem, row in ROWS:
        lines = LINES[(stem, row["key"])]
        for kw in ({}, ANSWERED_ONLY):
            reply, read = play(lines, **kw)
            if reply.reason not in (CLOSED, DEGENERATE):
                continue
            seen += 1
            expected = sum(1 for t, c, _d in map(project, lines[:read]) if t or c)
            assert reply.out_tokens == expected, (
                f"{stem}/{row['key']}: the stop reported {reply.out_tokens} tokens and {expected} "
                f"of the {read} lines it read carried anything")
            assert reply.prompt_tokens is None, (
                f"{stem}/{row['key']}: a count for a final message that never arrived")
    assert seen, "no recorded reply was stopped early, so this asserted nothing"


# ---- what the corpus preserves --------------------------------------------------------------------

def _ask_tree() -> ast.FunctionDef:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "ask":
            return node
    raise AssertionError("llm.py has no ask()")


def test_the_CORPUS_KEEPS_EVERY_FIELD_ask_READS():
    """**Read off the syntax tree, not off the format's docstring.** The corpus stores a projection
    of each stream line, so a `llm.py` that started reading an eighth field would be replaying
    against lines that never carry it — and every assertion above would still pass. The keys come
    from `ask`'s own `.get(...)` calls, so a new one announces itself here.
    """
    keys = {a.value for node in ast.walk(_ask_tree())
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get" and node.args
            for a in node.args[:1] if isinstance(a, ast.Constant) and isinstance(a.value, str)}
    assert keys <= RECORDED_KEYS, (
        f"ask() now reads {sorted(keys - RECORDED_KEYS)} off a stream line and the corpus does not "
        "record it. Re-record, and add the field to measure/replies.py's encode() and decode()")
    assert not RECORDED_KEYS - keys, (
        f"the corpus records {sorted(RECORDED_KEYS - keys)}, which ask() no longer reads — "
        "harmless, but the two lists have drifted and one of them is wrong about the other")


def test_THE_REASONING_TEXT_IS_ONLY_EVER_MEASURED_never_read():
    """**The claim the corpus's size rests on.** A thinking chunk is stored as its length, so the
    replay hands `ask` a substitute of the same length. That is exact for as long as `ask` reads
    `reasoning` in only two ways: `len(reasoning)`, and its truthiness in the `arrived` count.

    If a future `ask` looks at the reasoning *text* — matches it, logs it, parses it — every
    `--think on` row becomes a replay of the wrong stream, silently, and that arm is 33,281
    characters of reasoning. So the rule is read off the tree rather than remembered.
    """
    fn = _ask_tree()
    parents = {}
    for node in ast.walk(fn):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    uses = [n for n in ast.walk(fn)
            if isinstance(n, ast.Name) and n.id == "reasoning" and isinstance(n.ctx, ast.Load)]
    assert uses, "ask() no longer has a `reasoning` local; this rule is reading nothing"
    bad = []
    for use in uses:
        parent = parents[use]
        measured = (isinstance(parent, ast.Call) and isinstance(parent.func, ast.Name)
                    and parent.func.id == "len")
        tested = isinstance(parent, (ast.BoolOp, ast.UnaryOp, ast.Compare))
        if not (measured or tested):
            bad.append(f"line {use.lineno} inside {type(parent).__name__}")
    assert not bad, (
        "ask() reads the reasoning TEXT at " + ", ".join(bad) + ". The corpus records the length "
        "only, so every --think on row would be replaying a substitute. Re-record with the text "
        "kept, or those rows stop being evidence")


# ---- can any of this fail? -------------------------------------------------------------------------

#: `(label, pattern, replacement, what it would break in the shipped agent)`. Each is applied to
#: `llm.py`'s **source text**, compiled in memory and replayed — nothing is written into `argmind/`,
#: which is the mechanism `measure/choosing.py`'s `--prompt-check` already uses on `converse.py`.
MUTANTS = (
    ("GRACE 24 -> 1", r"^GRACE = 24$", "GRACE = 1",
     "a finished reply would lose the server's own token counts"),
    ("GRACE 24 -> 23", r"^GRACE = 24$", "GRACE = 23", "off by one in the grace"),
    ("GRACE 24 -> 25", r"^GRACE = 24$", "GRACE = 25", "off by one the other way"),
    ("LOOP_FLOOR 256 -> 128", r"^LOOP_FLOOR, LOOP_WINDOW, LOOP_PROBE, LOOP_TIMES = 256",
     "LOOP_FLOOR, LOOP_WINDOW, LOOP_PROBE, LOOP_TIMES = 128",
     "a real reply judged degenerate"),
    ("LOOP_FLOOR 256 -> 1024", r"^LOOP_FLOOR, LOOP_WINDOW, LOOP_PROBE, LOOP_TIMES = 256",
     "LOOP_FLOOR, LOOP_WINDOW, LOOP_PROBE, LOOP_TIMES = 1024",
     "seventy seconds of loop before it fires"),
    ("LOOP_WINDOW 512 -> 256", r"LOOP_TIMES = 256, 512", "LOOP_TIMES = 256, 256",
     "a longer cycle stops being caught"),
    ("LOOP_PROBE 32 -> 16", r"= 256, 512, 32, 4$", "= 256, 512, 16, 4",
     "a shorter tail matches more often"),
    ("LOOP_TIMES 4 -> 3", r"= 256, 512, 32, 4$", "= 256, 512, 32, 3",
     "three repetitions is emphasis, not a loop"),
    ("LOOP_TIMES 4 -> 5", r"= 256, 512, 32, 4$", "= 256, 512, 32, 5",
     "the loop is allowed one more turn of the cycle"),
    ("the scanner forgets it is in a string",
     "^ {12}elif ch == '\"':\n {16}self\\._in_string = True$",
     "            elif False:\n                self._in_string = True",
     'a brace inside a string closes the object early: {"say": "press } to close"}'),
    ("the scanner forgets an escape",
     '^ {16}elif ch == "\\\\\\\\":\n {20}self\\._escaped = True$',
     '                elif False:\n                    self._escaped = True',
     "a backslash-quote inside a string ends it and the next brace is miscounted"),
    ("arrived counts every line", r"arrived \+= 1 if \(reasoning or chunk\) else 0",
     "arrived += 1",
     "out_tokens over-reports on a stopped call by however many empty lines came past"),
    ("stop_when_answered is ignored", r"if self\.stop_when_answered and spare <= 0",
     "if True and spare <= 0",
     "the arm that would refute the whole change stops being available"),
    ("stop_when_looping is ignored", r"and self\.stop_when_looping and _looping\(text\)",
     "and True and _looping(text)", "the same, for the other stop"),
    ("the extractor is off by one", r"self\.text\[scan\.start:scan\.end \+ 1\]",
     "self.text[scan.start:scan.end]", "every reply parses to None"),
)

#: The mutants this corpus cannot kill, and what would kill each. **A corpus that no mutant escapes
#: is as suspicious here as anywhere else**; these three are the honest shape of what 128 recorded
#: replies contain and `tests/test_llm_stream.py`'s eight written ones do.
SURVIVORS = {
    "the scanner forgets it is in a string":
        "no recorded reply contains a brace inside a string. test_llm_stream.py covers it "
        'synthetically — {"say": "press } to close"} — and a corpus cannot until a model writes '
        "one. What would kill it: any real reply whose say quotes JSON, or a brace",
    "the scanner forgets an escape":
        "the same gap one level in: no recorded reply carries a backslash escape inside its first "
        'object. The synthetic case is {"say": "a \\" quote", "do": []}',
    "arrived counts every line":
        "equivalent under this server, as test_llm_stream.py already records: a read that ends "
        "early ends inside a line that carried content, and no empty line precedes it. Killing it "
        "would take a keep-alive message Ollama does not send, and a test written against a "
        "fabricated server protects nothing",
}


def _mutant(label, pattern, repl):
    src = SOURCE.read_text(encoding="utf-8")
    new, n = re.subn(pattern, repl, src, count=1, flags=re.MULTILINE)
    assert n == 1, (
        f"the mutation {label!r} did not apply to llm.py — the line it edits has moved, so this "
        "row proves nothing and has been passing silently")
    module = types.ModuleType("llm_mutant")
    module.__file__ = str(SOURCE)
    exec(compile(new, f"<mutant: {label}>", "exec"), module.__dict__)   # noqa: S102
    return module


def test_WHICH_MUTANTS_THE_CORPUS_CATCHES():
    """**The corpus's own negative control.** Every mutant in the table is compiled in memory and
    replayed over every recorded reply on all three arms: against the golden with the shipped stops,
    against the golden's grace section with the loop stop off, and against what the live server did
    with both off. All three are needed — a mutant that ignores `stop_when_answered` behaves exactly
    like the shipped reader while the flag is true and is only visible on the arm that switches it
    off, which is the arm `llm.py`'s docstring names as the one that would refute the whole change.

    Escapes are named in `SURVIVORS` with what would kill each. An escape that is not on the list is
    a gap; a listed one that starts being caught means the corpus grew a reply nobody described.

    The first assertion is that each mutation **applies**. A regex whose line has moved would make
    the whole table pass while testing nothing, which is how a mutation suite fails silently.
    """
    want = golden_now()
    caught, escaped = [], []
    for label, pattern, repl, why in MUTANTS:
        module = _mutant(label, pattern, repl)
        differs = None
        for stem, row in ROWS:
            key = f"{stem}/{row['key']}"
            if outcome(stem, row, module=module) != want[SHIPPED_ARM][stem][row["key"]]:
                differs = f"stops on, {key}"
            elif as_live(stem, row, module=module) != {k: row["recorded"][k] for k in LIVE}:
                differs = f"stops off, {key}"
            elif (key in want[GRACE_ARM]
                  and outcome(stem, row, module=module, **ANSWERED_ONLY) != want[GRACE_ARM][key]):
                differs = f"answered stop alone, {key}"
            if differs:
                break
        (caught if differs else escaped).append((label, differs or why))
    print(f"\n  MUTANTS — {len(caught)} of {len(MUTANTS)} caught by {len(ROWS)} recorded replies")
    for label, where in caught:
        print(f"    caught   {label:<42} first at {safe(where)}")
    for label, why in escaped:
        print(f"    ESCAPED  {label:<42} {safe(why)}")
    assert sorted(label for label, _w in escaped) == sorted(SURVIVORS), (
        f"the mutants this corpus lets through are {sorted(l for l, _w in escaped)} and the ones "
        f"written down are {sorted(SURVIVORS)}. Either a gap opened or one closed, and both change "
        "what these tests are worth")


def test_THE_GOLDEN_IS_NOT_A_TAUTOLOGY():
    """**A golden blessed from the implementation can only fail if the implementation moves**, so
    the one thing it must not be is insensitive. This is the cheapest demonstration that it is not:
    the extractor's bounds moved by one character, over the real corpus, and the golden says so on
    the first reply that parses.

    It duplicates one row of the table above on purpose — that table asserts a *set* of survivors
    and would pass if everything escaped and `SURVIVORS` had been widened to match.
    """
    module = _mutant("the extractor is off by one", r"self\.text\[scan\.start:scan\.end \+ 1\]",
                     "self.text[scan.start:scan.end]")
    want = golden_now()[SHIPPED_ARM]
    moved = [f"{stem}/{row['key']}" for stem, row in ROWS
             if outcome(stem, row, module=module) != want[stem][row["key"]]]
    assert moved, ("one character of Reply.json() changed and no recorded reply noticed. The "
                   "golden is pinning nothing a port could get wrong")
def test_RAISING_THE_FLOOR_ABOVE_THE_LONGEST_COMPLETE_REPLY_STOPS_THE_SAME_THREE():
    """**Why `LOOP_FLOOR` stays at 256 now that it no longer clears every real reply.**

    The floor was derived so that no complete answer is examined for repetition at all, off a run
    whose longest sentence was 119 characters. This corpus has complete replies of 2,357 — a listing
    of everything on a screen — so the derivation is dated and the detector does look at real
    output. The obvious repair is to raise the floor above the longest complete reply on record.

    **It does not work, and that is the whole reason this test exists rather than a sentence.** At
    `LOOP_FLOOR = 2560` the same three replies are stopped as degenerate, just later and thousands
    of tokens further in. So raising the floor buys no false positive back and costs the true one
    its saving, which is a decision somebody could otherwise reverse on plausibility alone.

    A failure here means the corpus grew a reply that the two floors judge differently — which is
    the evidence that would reopen it, and it should be read rather than blessed away.
    """
    raised = 2560
    module = _mutant(f"LOOP_FLOOR 256 -> {raised}",
                     r"^LOOP_FLOOR, LOOP_WINDOW, LOOP_PROBE, LOOP_TIMES = 256",
                     f"LOOP_FLOOR, LOOP_WINDOW, LOOP_PROBE, LOOP_TIMES = {raised}")
    want = golden_now()[SHIPPED_ARM]
    shipped_rows, raised_rows, later = {}, {}, []
    for stem, row in ROWS:
        was, now = want[stem][row["key"]], outcome(stem, row, module=module)
        key = f"{stem}/{row['key']}"
        if was["reason"] == DEGENERATE:
            shipped_rows[key] = was
        if now["reason"] == DEGENERATE:
            raised_rows[key] = now
            later.append(f"{safe(row['key'])}: {was['read'][0]} -> {now['read'][0]} of "
                         f"{now['read'][1]} lines")
    assert set(shipped_rows) == set(raised_rows), (
        f"a floor of {raised} changes WHICH replies are judged degenerate: only the shipped floor "
        f"stops {sorted(set(shipped_rows) - set(raised_rows))} and only the raised one stops "
        f"{sorted(set(raised_rows) - set(shipped_rows))}. llm.py says raising it changes no verdict")
    assert shipped_rows, "no reply is judged degenerate at either floor, so this asserted nothing"
    for key in shipped_rows:
        assert raised_rows[key]["read"][0] > shipped_rows[key]["read"][0], (
            f"{safe(key)} is stopped no later at a floor ten times higher, which cannot be right")
        assert raised_rows[key]["json_sha"] is None and shipped_rows[key]["json_sha"] is None, (
            f"{safe(key)} yields an answer at one of the two floors. The floor stays at 256 "
            "because raising it recovers nothing; a row that parses at 2560 is that claim "
            "failing, and it is llm.py's sentence that has to change, not this number")
    examined = sum(1 for stem, row in ROWS
                   if want[stem][row["key"]]["reason"] == FINISHED
                   and want[stem][row["key"]]["text_chars"] >= shipped.LOOP_FLOOR)
    print()
    print(f"  complete replies past LOOP_FLOOR={shipped.LOOP_FLOOR}, examined and "
          f"spared by LOOP_TIMES: {examined}")
    print("  the same rows stop later at " + str(raised) + ": " + "; ".join(later))
