# SPDX-License-Identifier: Apache-2.0
"""What `Local` reads off the stream, and when it stops reading.

**No model, no network.** The server is a list of lines and the test asserts how many of them were
consumed, because "it stopped early" is a claim about reading and not about the answer — and the
answer is the thing that must not change.

The decision under test, taken 2026-09-11: **`num_predict` stays at 4096 and the loop is bounded by
the shape of what comes back instead.** Measured over 1,845 calls at the shipped prompt with a
conversation in it — median 26 output tokens, 18 replies at the 4,096 ceiling, each 69–71 s, each a
turn the person hears nothing back from, together 47% of the run's model time in 1.0% of its turns.
A lower ceiling turns a 71-second silence into a four-second one and is still a silence; worse, it
puts back the second entry of `docs/findings/harness-pitfalls.md` on the `--think on` arm, which
spends 3,854 characters of reasoning against a 39-token peak with thinking off.

So pitfall 2 is prevented **by not lowering the ceiling at all**, and `test_the_CEILING_IS_STILL_...`
below is what would notice if a later change lowered it quietly.

**Twenty mutants, nineteen caught, and the two that survived the first pass were both real.** One
was that `stop_when_looping` could be ignored entirely: the control arm only ever ran a stream whose
envelope closed, so nothing exercised the flag on the path it exists for —
`test_WITHOUT_THE_LOOP_STOP_...` is the repair and it is the better test, because a control arm that
cannot reproduce the failure is not a control. The survivor that remains is **equivalent under the
server this file models**: counting every streamed line rather than only the ones carrying text
gives the same number, because a read that ends early ends inside a line that carried content and
this server sends no empty ones before its last. Killing it would mean inventing a keep-alive
message Ollama does not send, and a test written against a fabricated server protects nothing.
"""

from __future__ import annotations

import json

import pytest

from argmind.core.llm import (CEILING, CLOSED, DEGENERATE, FINISHED, GRACE, LOOP_FLOOR,
                              Local, Reply, _Balanced, _looping)


# ---- a server that is a list of lines -----------------------------------------------------------

class Stream:
    """One `/api/chat` response, and a count of how much of it the caller actually read.

    **The count is the whole instrument.** Every other assertion here would pass just as well if
    `ask` read the stream to the end and threw the tail away, which saves nothing — the cost being
    removed is the model generating tokens nobody will read, and the only proxy a test can hold for
    that is the caller hanging up.
    """

    def __init__(self, lines) -> None:
        self.lines, self.read, self.exited = lines, 0, False

    def __enter__(self):
        return self

    def __exit__(self, *_exc) -> bool:
        self.exited = True
        return False

    def __iter__(self):
        for line in self.lines:
            self.read += 1
            yield line

    @property
    def finished(self) -> bool:
        return self.read == len(self.lines)


def serve(content=(), *, thinking=(), reason="stop", prompt_tokens=1234, out_tokens=None):
    """The lines a streaming Ollama chat sends: reasoning, then content, then one final message
    carrying the counts. Only that last message has them, which is why hanging up costs them."""
    lines = [json.dumps({"message": {"thinking": t}}).encode() for t in thinking]
    lines += [json.dumps({"message": {"content": c}}).encode() for c in content]
    lines.append(json.dumps({
        "message": {"content": ""}, "done": True, "done_reason": reason,
        "prompt_eval_count": prompt_tokens,
        "eval_count": len(thinking) + len(content) if out_tokens is None else out_tokens,
    }).encode())
    return Stream(lines)


@pytest.fixture
def serving(monkeypatch):
    """`serving(stream, **kw)` -> a `Local` whose next call reads that stream and nothing else.

    Patched at `urllib.request.urlopen` rather than injected, because **the point is that `ask` uses
    the ordinary client**: a wrapper that took a stream as an argument would test a code path the
    app never runs.
    """
    import urllib.request                                                 # noqa: PLC0415

    def give(stream, **kw):
        monkeypatch.setattr(urllib.request, "urlopen", lambda *_a, **_k: stream)
        return Local("fake", **kw)

    return give


def chunks(text, size=4):
    """One reply, cut into token-sized pieces. The cut points are arbitrary on purpose: a stop that
    only works when a brace lands at the start of a chunk is not a stop."""
    return [text[i:i + size] for i in range(0, len(text), size)]


ANSWER = '{"say": "네, 열게요", "do": [{"click": 3}]}'
LOOP = '{"say": "ok", "do": [' + '{"key": "tab"}, ' * 400


# ---- the ceiling stays where it is --------------------------------------------------------------

def test_the_CEILING_IS_STILL_4096_and_a_truncated_reply_still_says_so(serving, monkeypatch):
    """**Pitfall 2 of `docs/findings/harness-pitfalls.md`, held by not moving the thing it is about.**
    Too small a `num_predict` truncates the model mid-thought, `done_reason` says `length`, the
    content is empty and it reads as a refusal. Three separate claims are pinned here because
    lowering the ceiling is the obvious response to a 4,096-token loop and it is the wrong one: the
    number is unchanged, it is what the request carries, and a reply that does hit it is still
    reported as having hit it rather than being folded into the two new stops.

    If this fails because somebody lowered the ceiling, the question to answer first is what now
    prevents a `--think on` call — 3,854 characters of reasoning, and reasoning counts against
    `num_predict` — from being cut off mid-thought.
    """
    sent = {}

    def spy(req, **_kw):
        sent.update(json.loads(req.data))
        return serve(chunks(ANSWER))

    model = serving(serve(chunks(ANSWER)))
    import urllib.request                                                 # noqa: PLC0415
    monkeypatch.setattr(urllib.request, "urlopen", spy)
    assert model.max_tokens == 4096
    model.ask("hi")

    assert sent["options"]["num_predict"] == 4096, (
        "the ceiling left the request; pitfall 2 was prevented by this number being large")
    hit = Reply("", None, 1.0, 10, 4096, CEILING)
    assert hit.truncated and hit.unfinished


def test_truncated_stays_NARROW_and_unfinished_is_the_WIDE_question():
    """**A column that means one thing before a commit and another after is how a comparison gets
    lost**, and `measure/`'s summary prints `truncated` beside the words *stopped at the token
    ceiling ... done_reason 'length'*. So the loop detector's catches do not join that count; they
    show up in `reasons`, which the same summary prints verbatim.

    `unfinished` is the question a caller with a person waiting has — is there anything to read —
    and it covers both.
    """
    assert Reply("", None, 1.0, 1, 4096, CEILING).truncated
    assert not Reply("", None, 1.0, None, 300, DEGENERATE).truncated
    assert Reply("", None, 1.0, None, 300, DEGENERATE).unfinished
    assert not Reply("{}", None, 1.0, 1, 8, FINISHED).unfinished
    assert not Reply("{}", None, 1.0, None, 30, CLOSED).unfinished, (
        "a closed object is a finished answer that this class stopped reading past, not a failure")


# ---- the stop that is free ----------------------------------------------------------------------

def test_a_reply_that_FINISHES_ON_ITS_OWN_KEEPS_THE_SERVER_S_OWN_COUNTS(serving):
    """**The grace exists for this test and this test only.** Breaking on the closing brace would
    save the one token a finished reply has left to emit and would cost `prompt_eval_count` and
    `eval_count` on every healthy call in every result file, because the server sends them once, at
    the end. 1,827 of 1,845 calls in the measured run are this case.
    """
    stream = serve(chunks(ANSWER))
    reply = serving(stream).ask("open it")
    assert stream.finished, "a well-behaved reply was cut short and its counts went with it"
    assert reply.reason == FINISHED and not reply.unfinished
    assert reply.prompt_tokens == 1234 and reply.out_tokens == len(chunks(ANSWER))
    assert reply.json() == {"say": "네, 열게요", "do": [{"click": 3}]}


def test_the_read_STOPS_once_the_answer_S_OBJECT_HAS_CLOSED(serving):
    """The 962-token case: a usable envelope, then 800 more tokens of talking. Measured twelve times
    in the run, at 16.6 s each for a 119-character sentence; and 192 output tokens for a 31-character
    one, twenty-seven such calls in all.

    **It is free because `Reply.json()` reads the first balanced object and nothing else** — the
    trailing text was already being thrown away, so declining to generate it cannot change an
    answer. That equivalence is asserted directly below rather than argued for here.
    """
    stream = serve(chunks(ANSWER + " " + "and also here is some more chatter. " * 60))
    reply = serving(stream).ask("open it")
    assert not stream.finished, "it read to the end, so nothing was saved"
    assert reply.reason == CLOSED
    assert reply.json() == {"say": "네, 열게요", "do": [{"click": 3}]}
    assert reply.prompt_tokens is None, (
        "the server's final message never arrived; reporting a count for it would be inventing one")
    assert reply.out_tokens == stream.read, "the count reported is the count of what arrived"
    assert reply.out_tokens == len(chunks(ANSWER)) + GRACE, (
        "it read the closing chunk and exactly the grace after it")


@pytest.mark.parametrize("text", [
    ANSWER,
    '{"say": "ok", "do": []}',
    'sure, here you go: {"say": "ok", "do": [{"click": 0}]} hope that helps',
    '{"say": "press } to close", "do": [{"click": 0}]}',
    '{"say": "a \\" quote", "do": []} trailing {"say": "second"}',
    'no json here at all, just a sentence',
    '{"say": "unclosed", "do": [',
    '{not json at all}',
])
def test_the_STOP_RETURNS_EXACTLY_WHAT_READING_TO_THE_END_WOULD_HAVE(serving, text):
    """**The load-bearing claim.** A stop that changed the answer would be a latency saving paid for
    with accuracy, and this project has no way to notice that except by measuring choosing again.

    It holds because the stop and the extractor are the same scanner: `Local.ask` stops when
    `_Balanced` says an object closed, and `Reply.json()` asks `_Balanced` where it was. The cases
    cover the ones this file has been wrong about before — a nested object, a brace inside a string,
    prose in front of the answer — plus the two where there is nothing to find, since a stop that
    fires on an unparseable reply would be worse than the ceiling.
    """
    with_stops = serving(serve(chunks(text)))
    without = serving(serve(chunks(text)), stop_when_answered=False, stop_when_looping=False)
    assert with_stops.ask("x").json() == without.ask("x").json()


def test_BOTH_STOPS_OFF_reproduces_the_reading_they_replaced(serving):
    """**A brake that cannot be switched off cannot be measured.** The arm that would refute the
    whole change is this configuration against the default in one session, so it has to exist and it
    has to be exactly the old behaviour: read everything, take the server's word for all of it.
    """
    stream = serve(chunks(ANSWER + " " + "chatter " * 200))
    reply = serving(stream, stop_when_answered=False, stop_when_looping=False).ask("x")
    assert stream.finished and reply.reason == FINISHED
    assert reply.prompt_tokens == 1234


def test_WITHOUT_THE_LOOP_STOP_the_control_arm_reproduces_the_MEASURED_FAILURE(serving):
    """The other half of the control, and the more important half: with the loop stop off, the same
    stream produces exactly what the run recorded — read to the very end, `done_reason: length`,
    `Reply.json()` returning `None`, the turn producing nothing at all, 18 times in 1,845.

    **A control arm that cannot reproduce the failure is not a control.** This is also what stops
    the flag from being decorative: without it the suite passes whether or not `stop_when_looping`
    is consulted, and a mutant that ignored the flag survived until this was written.
    """
    stream = serve(chunks(LOOP), reason="length")
    reply = serving(stream, stop_when_looping=False).ask("확대해줘")
    assert stream.finished, "the control arm stopped early, so it is not the old behaviour"
    assert reply.reason == CEILING and reply.truncated
    assert reply.json() is None and reply.out_tokens == len(chunks(LOOP))


# ---- the stop that is not free, and how much it can cost ----------------------------------------

def test_a_LOOP_is_stopped_LONG_before_the_ceiling(serving):
    """The 18 turns that produced nothing: `{"key": "tab"}` repeated to 4,096 tokens, 69–71 s each.

    The envelope never closes, so the free stop cannot reach it — that is why there are two. What is
    left is the repetition itself, which is visible in the stream hundreds of tokens before the
    ceiling. **This does not make the turn produce an answer**; it makes the nothing cost about two
    seconds instead of seventy, and it labels it, so the layer that assembles the turn can say
    something rather than going quiet.
    """
    stream = serve(chunks(LOOP))
    reply = serving(stream).ask("확대해줘")
    assert reply.reason == DEGENERATE and reply.unfinished
    assert reply.json() is None, "the point is that this reply was never going to parse"
    assert len(reply.text) < LOOP_FLOOR + 64, (
        f"stopped at {len(reply.text)} characters, which is late enough to be worth checking")
    assert stream.read < len(stream.lines) / 4


def test_a_LONG_BUT_HONEST_reply_is_not_mistaken_for_a_loop(serving):
    """**The false positive this can have, deliberately provoked.** A reply four times longer than
    anything the measured run produced, saying different things all the way, must run to the end —
    a detector that fires on length is a ceiling wearing a different hat, and the whole argument
    above is that a ceiling is the wrong instrument.
    """
    long_say = " ".join(f"항목{i}은 화면 {i * 7 % 13}번째 줄에 있습니다" for i in range(60))
    text = json.dumps({"say": long_say, "do": []}, ensure_ascii=False)
    assert len(text) > 4 * LOOP_FLOOR
    stream = serve(chunks(text))
    reply = serving(stream).ask("전부 설명해줘")
    assert reply.reason == FINISHED and reply.json()["say"] == long_say


def test_REASONING_IS_NOT_WATCHED_BY_EITHER_STOP(serving):
    """**The `--think on` arm is supported and is the accuracy end of the frontier** — 34 of 37
    correct decisions against 28 — and it spends 3,854 characters of reasoning to get there. Neither
    stop may touch it: a model circling a point while composing is what reasoning is, and the
    13,000-character argument this model once had with itself was reasoning rather than answer, so
    stopping that is a different decision resting on different evidence.

    Both stops therefore read `message.content` only. The reasoning is counted, as it already was.
    """
    reasoning = chunks("생각해보자. " * 400)
    stream = serve(chunks(ANSWER), thinking=reasoning)
    reply = serving(stream, think=True).ask("open it")
    assert stream.finished, "a stop reached into the reasoning; the thinking arm is not that arm"
    assert reply.reason == FINISHED and reply.thought == sum(len(c) for c in reasoning)
    assert reply.json() == {"say": "네, 열게요", "do": [{"click": 3}]}


# ---- the two pieces on their own -----------------------------------------------------------------

@pytest.mark.parametrize("cut", [1, 3, 7, 40])
def test_the_scanner_gives_the_SAME_ANSWER_however_the_stream_is_CHOPPED(cut):
    """A stop that depended on where the server split its tokens would fire correctly in a test and
    intermittently in use."""
    text = 'x {"say": "a } b", "do": [{"click": 1}]} tail'
    whole, streamed = _Balanced(), _Balanced()
    whole.feed(text)
    for piece in chunks(text, cut):
        streamed.feed(piece)
    assert (whole.start, whole.end) == (streamed.start, streamed.end)
    assert json.loads(text[whole.start:whole.end + 1]) == {"say": "a } b", "do": [{"click": 1}]}


#: The longest sentence the model produced in the 1,845-call run, measured, and the longest under a
#: reply that was not itself degenerating. An envelope around the first is about 160 characters.
LONGEST_SAY, LONGEST_ORDINARY_SAY = 119, 88

UNIT = "the same forty character sentence here!!!"[:40]


@pytest.mark.parametrize("text, looping", [
    ('{"key": "tab"}, ' * 40, True),
    ("\n" * 400, True),
    ('{"key": "tab"}, ' * 2, False),
    ("네, 열게요", False),
    ("항목이 여러 개 있습니다. " * 3, False),
    #: **The boundary, from both sides.** Three occurrences of a phrase is emphasis and four is a
    #: loop; a detector that fired at three would take a person asking for something repeated.
    ("a" * 200 + UNIT * 3, False),
    ("a" * 200 + UNIT * 4, True),
])
def test_the_LOOP_DETECTOR_looks_at_nothing_shorter_than_a_real_reply(text, looping):
    """Below `LOOP_FLOOR` nothing is examined at all, which is where the false-positive argument is
    spent; above it, `LOOP_TIMES` occurrences of the tail is the line."""
    assert _looping(text) is looping
    assert not _looping(text[:LOOP_FLOOR - 1]), "something shorter than any real reply was judged"


def test_the_FLOOR_is_above_ANY_ORDINARY_COMPLETE_REPLY():
    """**The derivation, pinned, because a constant with its reasoning only in a comment drifts.**
    Lowering the floor below twice the longest sentence measured is a decision about false positives
    and it needs a number behind it.

    **This test was called `..._ABOVE_ANY_COMPLETE_REPLY_THIS_PROJECT_HAS_MEASURED` and that name
    stopped being true.** At prompt `r322a4c`, 5 of the 122 complete replies in
    `measure/replies-*.json` are longer than the floor — the longest 2,357 characters, with a 507
    character `say` — so the detector does examine real output, and what spares it is `LOOP_TIMES`
    rather than the floor. The assertion below is unchanged and still holds: it pins the clearance
    over an *ordinary* reply, which is the case the floor was derived from. The wider claim is now
    checked where it can be, against recorded output, in `tests/test_llm_replay.py`.

    The number this docstring used to ask for — a `degenerate` on a turn a person recognises as a
    real request — **has been produced**, elicited by `measure/replies.py`'s stress arm, and the
    floor stayed at 256. `argmind/core/llm.py` carries the decision and why raising it would have
    bought nothing.
    """
    assert LONGEST_ORDINARY_SAY < LONGEST_SAY
    assert LOOP_FLOOR >= 2 * LONGEST_SAY, (
        f"the floor is {LOOP_FLOOR} and this project has measured a complete reply of "
        f"{LONGEST_SAY} characters; that is not enough clearance to call length degeneration")
