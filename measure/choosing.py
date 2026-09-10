# SPDX-License-Identifier: Apache-2.0
"""Step 4: does it pick the right thing? The number that decides whether the rebuild is worth doing.

    python -m measure.choosing              every goal, three times each
    python -m measure.choosing --repeat 1   one pass, for a quick look at a prompt change

**Everything else here is measured and this is not.** The environment answers in 41-266 ms; the
model answers in 67-85 ms; a name survives a moving screen. None of that matters if the thing it
picks is the wrong thing, and the only evidence so far is casual: two live attempts, one right and
one wrong — *"open the file menu"* selected the document.

## Four outcomes, not two

                        it clicked something        it clicked nothing
    a right answer      RIGHT or WRONG              MISSED
    no right answer     WRONG — the bad one         DECLINED — correct

`WRONG` on the bottom row is the outcome that matters most and the one a two-outcome score hides. A
model pushed to always act will always act, and on a screen that does not contain what was asked for
it clicks whatever is nearest. **Nine of the thirty-seven goals have no right answer** for exactly
that reason: without them the measurement cannot produce a negative result, and an agent that clicks
something every time would score the same as one that chooses well.

## Three runs, not one

Temperature is 0, so this ought to be deterministic; *ought to* is not a measurement, and this
project has withdrawn two conclusions that were single draws. The repeat count exists to report
whether the answers agree with themselves, not to average them into looking better.

## What this deliberately does not measure

Perception and actuation. The screens are fixtures captured from real windows, so nothing here can
fail because a window was not in front or a click missed — which is the point: a bad number then
points at choosing, and only at choosing.

## The prompt fingerprint every harness here imports — `q`, from 2026-09-10

    python -m measure.choosing --prompt-record    what is sent, verbatim, and its six characters
    python -m measure.choosing --prompt-check     that it moves on the prompt and not on prose

**This number is a function of the prompt that was sent, and until today nothing here measured
that.** `prompt_print()` hashed `converse.py` — the file — which failed in both directions at once.
It over-triggered: a comment forked it, and an agent that drafted a rationale into that file's
docstring watched `5461db` become `b2c3d3` and **reverted its own writing** rather than lose the
baselines. It under-covered: the files stored `prompt_text = SYSTEM`, so the reply-shape line — the
one literal whose reordering took `saying.py` from 6 leaks of 32 to 0 — appears in **no** result
file written before today.

`q` is a hash of what `converse()` actually renders, over three probes that between them exercise
every branch of its assembly, plus `MAX_LISTED` and `MOST_AT_ONCE` because `converse.py` itself
records that the ordering and `MOST_AT_ONCE` are the only two things ever measured to move the line.
The rendered text is stored verbatim as `prompt_record`, so two runs are compared by diffing rather
than by trusting six characters to have meant the same thing twice.

**The migration is the letter.** Old files say `p<hex>` and new ones say `q<hex>`; the two hash
different things and can collide on the hex, never on the letter. Nothing on disk was renamed,
because a recorded measurement is not editable. A reader who lines `choosing-index-…-p5461db.json`
up against a `q` file and treats them as a matched pair sees the letters differ, sees `prompt_scheme`
spelled out in the new file, and sees a `prompt_record` where the old file has a `prompt_text` that
holds only the system prompt — three refusals, of which one is in the filename. To ask the real
question, whether two runs shared a library, compare `converse_file_print`: it is the old `p` value,
recorded in every new file and kept out of every new name.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import random
import sys
import zlib
import time

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from argmind.core import Perceived, Said, Widget, converse               # noqa: E402
#: `SYSTEM` only. `MAX_LISTED` and `MOST_AT_ONCE` are read off the *resolved module* inside
#: `prompt_record()` instead, so that `--prompt-check` can measure a mutant's values rather than the
#: shipped ones — a check that read the constants from here would have reported "MOST_AT_ONCE does
#: not fork the print" and been right about the wrong module.
from argmind.core.converse import SYSTEM                                 # noqa: E402
import hashlib                                                          # noqa: E402
from argmind.core.llm import Local, Reply                                # noqa: E402
from measure.capture import load                                         # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
RIGHT, WRONG, MISSED, DECLINED = "right", "wrong", "missed", "declined"


# ---- what the model is actually sent -------------------------------------------------------------

#: The letter every filename carries for the current scheme, and **the letter IS the scheme**. Old
#: files say `p<hex>` and were a hash of `converse.py`; new ones say `q<hex>` and are a hash of the
#: rendered prompt. Two schemes can produce the same six hex characters by accident and can never
#: produce the same letter, so a reader who ignores everything else still cannot line a `p` file up
#: against a `q` one and believe they share a prompt.
PROMPT_LETTER = "q"
PROMPT_SCHEME = ("q = sha256(prompt_record())[:6] — the prompt as converse() actually renders it, "
                 "over three probes, plus MAX_LISTED and MOST_AT_ONCE. Superseded p = "
                 "sha256(converse.py)[:6] on 2026-09-10. A p-named file and a q-named file are NOT "
                 "comparable by fingerprint; compare converse_file_print instead.")

#: **The probe.** Two widgets, two lines of history, one request — fixed literal data whose only job
#: is to make `converse()` render every branch it has: asked and unprompted, with history and
#: without, with widgets and without. It is **not** a fixture and must not be tuned: editing
#: anything below forks every fingerprint in this directory, which is the price of hashing something
#: rendered rather than something read off disk.
#:
#: Short, dull, ASCII strings on purpose. The record is stored verbatim in every result file and is
#: meant to be read by eye and diffed by `diff`.
PROBE_WIDGETS = (Widget("one", "button", (0, 0, 1, 1)),
                 Widget("two", "combo box", (0, 0, 1, 1)))
PROBE_HISTORY = (("them", "a"), ("you", "b"))
PROBE_TEXT = "probe"


class _Capture:
    """A model that keeps the prompt and answers nothing.

    Standing in for `Local` rather than re-implementing the assembly, because a harness that
    rebuilds the thing under test is a second copy that can agree with itself while both are wrong —
    the same reason `conversation.Recorder` exists. The reply is a syntactically valid empty intent
    so `converse()` runs to the end down its ordinary path.
    """

    def __init__(self) -> None:
        self.sent = None

    def ask(self, prompt, images=None) -> Reply:
        self.sent = prompt
        return Reply('{"do": [], "say": ""}', 0.0, 0.0, 0, 0, "stop")


def _render(speak, label: str, trigger, widgets, history) -> tuple[str, str]:
    """One probe, rendered by `speak` (`converse`, or a mutant of it under `--prompt-check`).
    Returns `(the labelled block, the bare prompt)`."""
    cap = _Capture()
    speak(cap, list(widgets), trigger, history=list(history))
    if cap.sent is None:
        #: Not a `return None`. If `converse()` ever answers without asking, the thing this
        #: fingerprints was never built, and hashing the empty string would give six stable
        #: characters that mean nothing at all — a clean zero produced by the harness.
        raise RuntimeError("converse() returned without calling model.ask() — there is no prompt "
                           "to fingerprint, and a hash of nothing is not a fingerprint")
    return f"--- {label} ---\n{cap.sent}\n", cap.sent


def prompt_record(module=None) -> str:
    """**What is actually sent**, verbatim — stored beside every result rather than hashed away
    into six characters.

    `module` is `argmind.core.converse` unless `--prompt-check` hands in a mutant of it; taking it
    as an argument is what lets this file prove its own claim without writing a byte into
    `argmind/`.

    ## Why this replaced a hash of `converse.py` (2026-09-10)

    The old print hashed the file, so it failed in both directions at once.

    *It over-triggered.* A comment forked it. That is not a pedantic cost: one agent drafted a
    rationale into `converse.py`'s docstring, watched the fingerprint fork `5461db -> b2c3d3`, and
    **reverted its own writing** to keep the baselines comparable — the measurement was editing the
    library's prose. A correction to that same docstring's stale tallies is blocked by the same
    mechanism.

    *It under-covered.* The baselines stored `prompt_text = SYSTEM` and nothing else, so **no file in
    this directory contains the reply-shape line** — and that line is the exact literal the last kept
    change moved (`{"say": ..., "do": ...}` to `{"do": ..., "say": ...}`, leak 6 of 32 to 0 of 32).
    `saying-...-p5461db-...` and `saying-...-pbcfe81-...` differ by that change, differ on the
    headline, and their two `prompt_text` fields are **byte-identical**. The thing most worth
    recording was the thing not recorded.

    ## What is in it

    Three renders through the real `converse()`, covering every branch of its assembly — asked /
    unprompted (so `UNPROMPTED` is in the record), history / none (so the `Recently:` header and the
    `  who: text` line format are), widgets / none (so `describe()`'s `0: role "name"` numbering is,
    and so is the empty-surface line). Plus `MAX_LISTED` and `MOST_AT_ONCE` as bare numbers, because
    they are conditions of the result that never appear in the prompt: `converse.py`'s own docstring
    records that of everything ever tried, **`MOST_AT_ONCE` and the field ordering are the only two
    things that ever moved the measured line**, so a print blind to one of them is blind to half of
    what is known to work.

    ## What it would fail to notice changing

    `_said()`'s recovery of a misfiled utterance, `_effect()`'s validation, and anything else
    `converse()` does to the reply *after* it arrives: those are not sent, so they are not here.
    They are real conditions and this print does not cover them. Also outside it, and covered
    elsewhere: which fixtures were used (`s`), which cases (`f`), the classifier (`h`), and the
    model wrapper's `num_ctx` / `num_predict` / `temperature` (`llm_print`).
    """
    import importlib                                                     # noqa: PLC0415

    #: **`import_module`, not `import argmind.core.converse as shipped`.** The second form binds the
    #: result of `getattr(argmind.core, "converse")`, and `argmind/core/__init__.py` re-exports a
    #: *function* of that name over the *module* of that name — so it silently hands back the
    #: function and every attribute read here fails, or worse, succeeds on the wrong object. The
    #: scheme this replaced carried the same warning in its docstring and this still caught me.
    m = module or importlib.import_module("argmind.core.converse")
    blocks = [f"MAX_LISTED = {m.MAX_LISTED}\nMOST_AT_ONCE = {m.MOST_AT_ONCE}\n"]
    texts = []
    for label, trigger, widgets, history in (
            ("asked, with history, widgets listed", Said(PROBE_TEXT), PROBE_WIDGETS, PROBE_HISTORY),
            ("unprompted, no history, widgets listed", Perceived(PROBE_TEXT), PROBE_WIDGETS, ()),
            ("asked, no history, nothing on screen", Said(PROBE_TEXT), (), ())):
        block, text = _render(m.converse, label, trigger, widgets, history)
        blocks.append(block)
        texts.append(text)
    #: **The probes must still be probing three different things.** If two renders come back
    #: identical, a branch this print claims to cover has stopped existing or stopped being reached,
    #: and the fingerprint would go on looking healthy while covering less than it says. Refusing to
    #: run beats a print that quietly narrows.
    if len(set(texts)) != len(texts):
        raise RuntimeError(
            "two of the three probes rendered identically, so a branch of converse()'s prompt "
            "assembly this fingerprint claims to cover is no longer distinguishable — the probes in "
            "measure/choosing.py need rewriting before any number is quoted")
    return "\n".join(blocks)


def prompt_print(module=None) -> str:
    """Six characters over `prompt_record()` — the prompt as sent, not the file that builds it.

    **A comment cannot fork this** because a comment is not sent. Neither can a docstring, an
    import, a rename, a type annotation, or any of the reasoning `converse.py` carries about why its
    prompt is shaped the way it is. What forks it is a changed word in `SYSTEM` or `UNPROMPTED`, a
    changed reply-shape line, a changed widget-line format, a changed trigger phrasing, a changed
    history-line format, and a changed `MAX_LISTED` or `MOST_AT_ONCE`. That list is the whole claim.
    """
    return hashlib.sha256(prompt_record(module).encode("utf-8")).hexdigest()[:6]


def prompt_tag() -> str:
    """The filename segment: the scheme letter and the print. Every harness calls this rather than
    formatting the letter itself, because the migration is exactly a change of letter, and four
    files formatting it by hand is four chances for one of them to keep the old one."""
    return f"{PROMPT_LETTER}{prompt_print()}"


def converse_file_print() -> str:
    """The **old** scheme's six characters — sha256 of `converse.py` — recorded, and never in a name.

    This is the bridge across the migration and the only reason it survives. Recorded baselines are
    named `p5461db` (say-first) and `pbcfe81` (the shipped do-first tree); a new file's `q` cannot be
    compared with either, but its `converse_file_print` can, and it says which of them the run was
    made at. A reader who wants to know whether a new run and an old one shared a library reads this
    field. A reader who puts it back in a filename has re-created the defect that made an agent
    revert its own writing to protect a hash.
    """
    import argmind.core                                                  # noqa: PLC0415

    source = pathlib.Path(argmind.core.__file__).with_name("converse.py")
    return hashlib.sha256(source.read_bytes()).hexdigest()[:6]


def by_name_record() -> str:
    """The `--by-name` prompt, rendered over the same probe.

    **`q` cannot cover this and must not appear to.** `by_name()` does not call `converse()` — it
    assembles its own prompt here, out of `SYSTEM` plus five instruction lines that live in this
    file — so a `choosing-name-…` result carrying `q<hex>` would be advertising a fingerprint of a
    prompt it did not use. Two of those five lines were rewritten once already, after the first
    wording scored five correct answers as zero; that rewrite left no mark on any filename.

    Rendered rather than hashed off the source for the same reason as `prompt_record()`: a comment
    in this file is not part of the question the model was asked.
    """
    lines = _by_name_lines(list(PROBE_WIDGETS), PROBE_TEXT)
    return chr(10).join(lines) + "\n"


def by_name_print() -> str:
    """Six characters over `by_name_record()`. In the filename **only when `--by-name` is used**,
    because on the index path it is not a condition of anything."""
    return hashlib.sha256(by_name_record().encode("utf-8")).hexdigest()[:6]


def prompt_fields() -> dict:
    """The four keys every result file in this directory carries, written once so that four
    harnesses cannot drift apart on them.

    `prompt_text` is **gone, not renamed**: it held `SYSTEM` alone, `prompt_record` holds something
    else entirely, and one key meaning two different things either side of a migration is how a
    reader gets misled by a field that parses perfectly.
    """
    return {"prompt_scheme": PROMPT_SCHEME, "prompt": prompt_print(),
            "prompt_record": prompt_record(), "converse_file_print": converse_file_print()}


# ---- what a call cost ----------------------------------------------------------------------------

def spend(reply) -> dict:
    """What one model call cost, off the `Reply` itself.

    **The instrument this directory was missing.** A degeneration was found by hand and by hand
    only: the model repeated `{"key": "tab"}` for **4,096 output tokens and 71 s** with
    `done_reason: length`, so `Reply.json()` returned `None` and the turn produced **nothing at
    all**. In the result files it was a slightly larger `seconds` — and `seconds` is the one figure
    that cannot tell it apart from a busy machine, since a sweep run while this PC was working shows
    6-14 s stalls at 17-39 output tokens on a perfectly healthy baseline. **The quantity that
    separates them is generated length**, and nothing recorded it.

    `None` in gives all-`None` out rather than zeros: a call that produced no reply is a fact, and a
    zero here would average into a rate.
    """
    if reply is None:
        return {"seconds": None, "out_tokens": None, "reason": None, "truncated": None,
                "thought_chars": None}
    return {"seconds": round(reply.total, 3), "out_tokens": reply.out_tokens,
            "reason": reply.reason, "truncated": reply.truncated,
            "thought_chars": reply.thought}


def spend_summary(labelled, ceiling=None) -> dict:
    """`labelled` is `(label, spend)` pairs. The worst call, named — never a mean.

    A mean output length hides one 4,096-token turn among forty short ones, which is the entire
    event this exists to make visible. So: the maximum, **what produced it**, and every reply that
    hit the ceiling listed by name.
    """
    labelled = list(labelled)
    got = [(lab, s) for lab, s in labelled if s.get("out_tokens") is not None]
    timed = [(lab, s) for lab, s in labelled if s.get("seconds") is not None]
    thought = [(lab, s) for lab, s in labelled if s.get("thought_chars")]
    worst = max(got, key=lambda ls: ls[1]["out_tokens"], default=None)
    slowest = max(timed, key=lambda ls: ls[1]["seconds"], default=None)
    thinker = max(thought, key=lambda ls: ls[1]["thought_chars"], default=None)
    return {
        "calls": len(labelled), "with_counts": len(got),
        "max_tokens_setting": ceiling,
        "out_tokens_max": worst[1]["out_tokens"] if worst else None,
        "out_tokens_max_on": worst[0] if worst else None,
        "truncated": sum(1 for _lab, s in labelled if s.get("truncated")),
        "truncated_on": [lab for lab, s in labelled if s.get("truncated")],
        "reasons": dict(collections.Counter(str(s.get("reason")) for _lab, s in labelled)),
        "seconds_max": slowest[1]["seconds"] if slowest else None,
        "seconds_max_on": slowest[0] if slowest else None,
        "thought_chars_max": thinker[1]["thought_chars"] if thinker else 0,
        "thought_chars_max_on": thinker[0] if thinker else None,
        "thought_chars_total": sum(s.get("thought_chars") or 0 for _lab, s in labelled),
    }


def print_spend(s: dict) -> None:
    """The cost of the run, as a delta rather than a boolean: the longest reply beside the ceiling
    it was allowed, and the case that produced it."""
    ceiling = s["max_tokens_setting"]
    room = "" if not ceiling else f" of a {ceiling} ceiling"
    print(f"\n  WHAT IT COST — n = {s['calls']} calls")
    print(f"    output tokens  max {s['out_tokens_max']}{room}  "
          f"on {str(s['out_tokens_max_on'])[:44]!r}")
    print(f"    slowest call   {s['seconds_max']} s  on {str(s['seconds_max_on'])[:44]!r}")
    if s["thought_chars_total"]:
        #: `--think on` is a supported arm and its bill was invisible in every result here. Printed
        #: whenever it is non-zero, so a run meant to measure a prompt that measured the prompt
        #: *and* the reasoning says so on its own summary line rather than only in `seconds`.
        print(f"    thinking       {s['thought_chars_total']} characters over the run, "
              f"max {s['thought_chars_max']} on {str(s['thought_chars_max_on'])[:36]!r}")
    if s["truncated"]:
        n = s["truncated"]
        print(f"    ** TRUNCATED   {n} repl{'y' if n == 1 else 'ies'} stopped at the token ceiling "
              f"rather than finishing — done_reason 'length'. **Usually** that means "
              f"Reply.json() found no complete object and the turn produced nothing at all; it can "
              f"still parse if a whole object happened to land before the cut, so read the reply "
              f"before calling it empty. Either way it is not merely slow.")
        for lab in s["truncated_on"][:10]:
            print(f"         {str(lab)[:70]!r}")
    elif not s["with_counts"]:
        #: **Not zero — unknown.** No call reported an `eval_count`, so "nothing hit the ceiling" is
        #: not something this run measured, and printing a clean 0 for it would be the harness
        #: reporting an absence of evidence as evidence of absence.
        print(f"    truncated      unknown — no call reported a token count "
              f"(reasons seen: {s['reasons']})")
    else:
        print(f"    truncated      0 of {s['with_counts']} counted — no reply ran to the ceiling "
              f"(reasons seen: {s['reasons']})")


# ---- can the fingerprint fail in both directions? -------------------------------------------------

#: `(label, from, to, must_fork)`. **Written before the scheme was run**, and half of them assert
#: that the print does NOT move — a fingerprint that only ever proves it can change has proved the
#: easy half. `from` is a literal that must exist in the shipped `converse.py`; a mutant whose
#: substitution matched nothing would silently pass every row, so that is checked separately.
#:
#: The first row is not hypothetical: it is the correction to `converse.py`'s docstring that the old
#: scheme was blocking on the day this was written (the environment declares fifteen names over
#: fourteen distinct virtual keys — `esc` and `escape` are both 0x1B).
PROMPT_MUTANTS = (
    ("the blocked docstring correction", "fourteen named keys",
     "fifteen names over fourteen distinct virtual keys", False),
    ("a stale tally removed from a docstring", "`enter` (41), `tab` (24) and `zoom in` (3)",
     "`enter`, `tab` and `zoom in`", False),
    ("a plain comment", "from .intent import Effect, Intent",
     "# a comment that must not fork a baseline\nfrom .intent import Effect, Intent", False),
    ("the reply-shape line", "Reply with only JSON:", "Reply with JSON only:", True),
    ("the field ordering", '{"do": [ ... ], "say": "<what you say back>"}',
     '{"say": "<what you say back>", "do": [ ... ]}', True),
    ("a word of SYSTEM", "a wrong click is worse", "a wrong click is far worse", True),
    ("a word of UNPROMPTED", "Nobody asked you anything", "Nobody asked you a thing", True),
    ("the widget-line numbering", '{i}: {w.role} "{w.name}"', '{i}. {w.role} "{w.name}"', True),
    ("the history-line format", '"  {who}: {text}"', '"  {who} said: {text}"', True),
    ("MOST_AT_ONCE", "MOST_AT_ONCE = 3", "MOST_AT_ONCE = 8", True),
    ("MAX_LISTED", "MAX_LISTED = 200", "MAX_LISTED = 150", True),
)


def _mutant_module(source: str):
    """`converse.py`'s source, altered and executed **in memory**. Nothing is written.

    `measure/` may not edit `argmind/`, and it should not want to: a check that edits the library to
    test itself is one interrupted run away from leaving the library edited. The relative import
    inside `converse.py` resolves because `__package__` is set and `argmind.core` is already loaded.
    """
    import argmind.core                                                  # noqa: PLC0415, F401
    import types                                                         # noqa: PLC0415

    mod = types.ModuleType("argmind.core._prompt_check_mutant")
    mod.__package__ = "argmind.core"
    mod.__file__ = "<mutant of converse.py, never written to disk>"
    exec(compile(source, mod.__file__, "exec"), mod.__dict__)            # noqa: S102
    return mod


def prompt_check() -> int:
    """Prove the claim in `prompt_print()`'s docstring, both halves, and return 0 or 1.

        python -m measure.choosing --prompt-check

    **The claim is two-sided and so is this**: a comment, a docstring and a withdrawn tally must
    leave the print exactly where it was, and the reply shape, the field ordering, either prompt
    text, the numbering, the history format, `MOST_AT_ONCE` and `MAX_LISTED` must every one of them
    move it. A scheme that only demonstrates the second half is a scheme nobody has shown to be
    better than the one it replaced, which hashed the whole file and moved on everything.
    """
    import argmind.core                                                  # noqa: PLC0415

    path = pathlib.Path(argmind.core.__file__).with_name("converse.py")
    source = path.read_text(encoding="utf-8")
    before = path.read_bytes()
    base = prompt_print()
    print(f"  shipped converse.py -> {PROMPT_LETTER}{base}   "
          f"(old scheme's p{converse_file_print()})\n")
    print(f"  {'':<3} {'edit':<40} {'old p':<8} {'new q':<8} verdict")
    bad = 0
    for label, old, new, must_fork in PROMPT_MUTANTS:
        if source.count(old) != 1:
            #: A substitution that matches nothing changes nothing, so every "did not fork" row
            #: would pass for the wrong reason and the check would report a clean sweep while
            #: testing nothing. That is this project's own recurring defect, so it is fatal here.
            print(f"  {'!!':<3} {label:<40} literal appears {source.count(old)} times in "
                  f"converse.py — this row tests nothing")
            bad += 1
            continue
        text = source.replace(old, new)
        got = prompt_print(_mutant_module(text))
        legacy = hashlib.sha256(text.encode("utf-8")).hexdigest()[:6]
        forked = got != base
        ok = forked == must_fork
        bad += not ok
        want = "must fork" if must_fork else "must NOT fork"
        print(f"  {'ok ' if ok else 'FAIL':<3} {label:<40} p{legacy:<7} "
              f"{PROMPT_LETTER}{got:<7} {'forked' if forked else 'unchanged':<9} — {want}")
    #: The library is read and never written, and this says so out loud rather than trusting the
    #: sentence above. `measure/` owning a check that mutates `argmind/` would be one crash away
    #: from a library edited by a measurement.
    if path.read_bytes() != before:
        print("\n  FAIL: converse.py changed on disk during a check that only reads it")
        bad += 1
    n = len(PROMPT_MUTANTS)
    forks = sum(1 for _l, _o, _n, f in PROMPT_MUTANTS if f)
    print(f"\n  {n - bad} of {n} as labelled ({n - forks} that must not fork, {forks} that must)"
          f"{'' if not bad else '   <- the scheme does not do what its docstring says'}")
    print("  the old scheme (sha256 of converse.py) forks on every row above, including the three "
          "that are only prose — that is what it was replaced for")
    return 1 if bad else 0


def by_name(model, widgets, goal):
    """The same question with the answer given as an exact NAME instead of a list index.

    **Why this variant exists.** Six of the seven wrong answers in the first run sat 1-10 places from
    a correct one and five within four — *"창 닫아"* chose index 121 where `Close` is 122. That is
    the shape of arithmetic going slightly wrong, not of a model that misunderstood the request, and
    it is the same failure the vision model had: it could see the answer and could not compute the
    number for it.

    The index was chosen because a number can be range-checked, so the model cannot invent a target.
    **A name can be checked exactly as strictly** — it must appear in the list, character for
    character — so nothing is given up by asking for one.

    Returns `(the accepted name or None, the raw string it asked for, the Reply)`. The `Reply` is
    handed back rather than dropped because a declined answer and a reply that ran to the token
    ceiling are indistinguishable from the first two alone.
    """
    listed = list(widgets)
    reply = model.ask(chr(10).join(_by_name_lines(listed, goal)))
    got = reply.json() or {}
    want = got.get("click")
    if not isinstance(want, str) or not want:
        return None, None, reply
    #: Exact match only. A name the list does not contain is an invention, and the right answer to
    #: an invention is to drop it — the same rule the index path applies to an out-of-range number.
    #:
    #: **The rejected string is handed back rather than swallowed**, because "it declined" and "it
    #: named something that is not there" are different failures and the first version of this
    #: reported both as nothing. One of them is the model being careful and the other is this
    #: protocol not working.
    return (want if any(w.name == want for w in listed) else None), want, reply


def _by_name_lines(listed, goal) -> list:
    """The `--by-name` prompt, as a list of lines. **Split out of `by_name()` so it can be rendered
    without a model**, which is the only way its fingerprint can be taken.

    **The role goes BEFORE the quoted name and the instruction says so twice**, because the first
    version of this prompt read "the exact name of one thing above" and the model answered
    `button "Copy"` — the whole line, quotes and role included, which is a literal reading of what it
    was told. Every one of five answers was right and every one was scored zero. That is the harness
    lying again, and the fix is the prompt rather than a lenient match on the answer: matching
    leniently would let a genuine invention through as well.
    """
    lines = [SYSTEM, "", "On screen now — one per line, role first, then the name in quotes:"]
    lines += [f'{w.role}: "{w.name}"' for w in listed]
    lines += ["", f"They said: {goal}", "",
              'Reply with only JSON: {"say": "<what you say back>", "click": "<a name>"}.',
              'For "click", give ONLY what is inside the quotes on one line above — not the role '
              'word, not the quotes. So for the line   button: "Copy"   you would write '
              '"click": "Copy".',
              'Use null for "click" if nothing listed above does this.']
    return lines


def outcome(chosen, accept) -> str:
    """One goal's result. `chosen` is the name it clicked, or `None`."""
    if accept is None:
        return DECLINED if chosen is None else WRONG
    if chosen is None:
        return MISSED
    return RIGHT if chosen in accept else WRONG


def run(model_name: str, repeat: int, only: str | None, *,
        named: bool = False, shuffle: int | None = None, drop=(), think=None) -> int:
    screens = {name: widgets for name, _window, widgets in load()}
    if drop:
        #: **Leaving something out of the list is a change to what the model can see**, not a
        #: cleanup. A `text` widget cannot usefully be clicked — one of the recorded wrong answers
        #: clicked `\\Device\\HarddiskVolume1`, and a `text` label named `Font :` sitting in front
        #: of a combo box of the same name caused a defect that took a whole round of work to find.
        #: But the same widgets are what the model reads when it is asked *what is on screen*, so
        #: this has to be measured rather than assumed to be free.
        before = sum(len(v) for v in screens.values())
        screens = {k: [w for w in v if w.role not in drop] for k, v in screens.items()}
        after = sum(len(v) for v in screens.values())
        print(f"  dropped {before - after} of {before} widgets with role in {drop}")
    goals = json.loads((HERE / "goals.json").read_text(encoding="utf-8"))["goals"]
    if only:
        goals = [g for g in goals if only in g["screen"]]
    if not goals or not screens:
        print("  nothing to run — capture some screens first (python -m measure.capture)")
        return 2

    model = Local(model_name, think=think)
    rows, tally, spends = [], collections.Counter(), []
    how = ("by NAME" if named else "by number") + (
        "" if shuffle is None else f", shuffled {shuffle}") + (
        "" if not drop else f", without {'/'.join(drop)}") + (
        "" if think is None else f", thinking {'on' if think else 'off'}")
    print(f"  {model_name} | {len(goals)} goals x {repeat} | {how} | "
          f"{len(screens)} screens, {sum(len(w) for w in screens.values())} widgets\n")
    print(f"  {'outcome':<9} {'screen':<14} {'goal':<28} chose")
    t_start = time.perf_counter()
    for g in goals:
        widgets = screens.get(g["screen"])
        if widgets is None:
            print(f"  {'NO SCREEN':<9} {g['screen'][:14]:<14} {g['goal'][:28]:<28}")
            continue
        if shuffle is not None:
            #: **The discriminator.** Index proximity and MEANING proximity are confounded on a real
            #: toolbar — related controls sit next to each other — so a small gap does not by itself
            #: say which one caused the error. Shuffling breaks the tie: if the cause is arithmetic
            #: the gaps stay small in the new order, and if it is meaning the same wrong items get
            #: chosen wherever they are.
            widgets = list(widgets)
            #: **crc32, not hash().** Python randomises `hash()` of a string per process, so the
            #: first version of this produced an order that could not be reproduced afterwards —
            #: and the analysis of it silently scored against an order that was never shown to the
            #: model. A measurement whose conditions cannot be reconstructed is not a measurement.
            random.Random(shuffle * 1000003 + zlib.crc32(g["goal"].encode())).shuffle(widgets)
        seen, rejected, counts, costs = [], [], [], []
        for _ in range(repeat):
            if named:
                got, raw, reply = by_name(model, widgets, g["goal"])
                seen.append(got)
                rejected.append(None if got is not None else raw)
                costs.append(spend(reply))
                continue
            intent = converse(model, widgets, Said(g["goal"]))
            clicks = [e.target for e in intent.effects if e.kind == "click"]
            #: **How many, not only which.** The talking harness showed the spurious actions come in
            #: threes and eights while a real request wants one, and that comparison was impossible
            #: because this file kept the first click and dropped the count.
            counts.append(len(intent.effects))
            #: **What the call cost, every repeat.** A MISSED with 4,096 output tokens behind it and
            #: a MISSED from a model that answered `{"do": []}` in twenty are the same row here and
            #: are not the same event: the first is `done_reason: length`, an unparseable reply and
            #: a turn that produced nothing. Recorded so the next one is in the file.
            costs.append(spend(intent.reply))
            seen.append(clicks[0] if clicks else None)
        results = [outcome(c, g["accept"]) for c in seen]
        #: Positions are recorded rather than reconstructed later, for the same reason: the order
        #: the model was shown is a condition of the result and belongs with it.
        names = [x.name for x in widgets]
        at = [names.index(c) if c in names else None for c in seen]
        right_at = [i for i, n in enumerate(names) if g["accept"] and n in g["accept"]]
        #: The WORST of the repeats is what gets counted. Taking the best would be scoring the
        #: model's luckiest draw, and an agent you use once gets one draw.
        worst = sorted(results, key=[RIGHT, DECLINED, MISSED, WRONG].index)[-1]
        tally[worst] += 1
        agreed = len(set(seen)) == 1
        rows.append({"screen": g["screen"], "goal": g["goal"], "accept": g["accept"],
                     "chose": seen, "results": results, "counted": worst, "agreed": agreed,
                     "chose_at": at, "right_at": right_at, "n_listed": len(names),
                     "rejected": rejected, "effects": counts, "spend": costs})
        spends += [(f"{g['screen']} / {g['goal']}", c) for c in costs]
        mark = "" if agreed else "  (disagreed across runs)"
        print(f"  {worst.upper():<9} {g['screen'][:14]:<14} {g['goal'][:28]:<28} "
              f"{str(seen[0])[:34]}{mark}")

    took = time.perf_counter() - t_start
    answerable = [r for r in rows if r["accept"] is not None]
    unanswerable = [r for r in rows if r["accept"] is None]
    print(f"\n  {len(rows)} goals in {took:.0f} s ({took / max(1, len(rows) * repeat):.2f} s a call)")
    print(f"    right      {tally[RIGHT]:>3} / {len(answerable)} answerable")
    print(f"    wrong      {tally[WRONG]:>3}   ({sum(1 for r in unanswerable if r['counted'] == WRONG)}"
          f" of them on a screen with no right answer — the bad one)")
    print(f"    missed     {tally[MISSED]:>3}   (a right answer existed and it did nothing)")
    print(f"    declined   {tally[DECLINED]:>3} / {len(unanswerable)} unanswerable")
    disagreed = [r for r in rows if not r["agreed"]]
    print(f"    disagreed  {len(disagreed):>3}   across {repeat} runs at temperature 0"
          f"{' — so a single run is not this number' if disagreed else ''}")
    cost = spend_summary(spends, ceiling=model.max_tokens)
    print_spend(cost)

    #: **The model and the prompt are both part of the filename, and every model gets one.** Two
    #: separate versions of this have destroyed a baseline: first a 14b run overwrote the 7b file
    #: because the name carried only the protocol, and then the fix — keeping the historic name for
    #: the 7b — let a *7b* run overwrite it again. There is no exception now.
    stem = model_name.replace(":", "-").replace("/", "-")
    tag = (("name" if named else "index") + ("" if shuffle is None else f"-shuffled{shuffle}")
           + ("" if not drop else "-without-" + "-".join(drop)) + f"-{stem}")
    tag += "" if think is None else ("-thinking" if think else "-nothink")
    #: **`q`, and the letter is the migration.** Every file here older than 2026-09-10 carries
    #: `p<hex>`, which was a hash of `converse.py` — a different quantity under six characters of
    #: the same shape. Two hashes can collide on the hex; they cannot collide on the letter, so a
    #: `p` file and a `q` file cannot be mistaken for a matched pair however carelessly they are
    #: read. `converse_file_print` in the JSON is what bridges them.
    tag += f"-{prompt_tag()}"
    if named:
        #: The `--by-name` arm asks a different question, assembled in this file, and `q` describes
        #: only the one `converse()` asks. `n` is that second prompt, and it is absent from every
        #: index run because there it is a condition of nothing.
        tag += f"-n{by_name_print()}"
    if only:
        #: **`--only` belongs in the name.** It did not, so a one-screen look wrote the same
        #: filename as the full 37-goal run and silently replaced it — the fourth instance in this
        #: directory of a name that carries some conditions and not others, and the cheapest to
        #: have avoided. `saying.py` already hashes its own `--only` for exactly this reason.
        tag += "-only" + hashlib.sha256(only.encode()).hexdigest()[:6]
    out = HERE / f"choosing-{tag}.json"
    out.write_text(json.dumps({
        "model": model_name, "repeat": repeat, "seconds": round(took, 1),
        "how": how, "named": named, "shuffle": shuffle, "think": think, "only": only,
        #: **Which prompt produced this**, and the prompt itself, verbatim. Four rows of a
        #: model-comparison table were measured under one reply shape and a fifth under another,
        #: and nothing in the files said so — the table read as a comparison of models and was
        #: partly a comparison of prompts. The predecessor of this line stored `SYSTEM` alone, so
        #: the reply-shape line that produced the largest measured change in this project was in no
        #: result file anywhere; `prompt_record` is what was actually sent.
        **prompt_fields(),
        #: Only on the arm that uses it, so an index run cannot be read as having asked it.
        **({"by_name_prompt": by_name_print(), "by_name_record": by_name_record()} if named else {}),
        "max_tokens": model.max_tokens, "spend": cost,
        "screens": {k: len(v) for k, v in screens.items()},
        "tally": dict(tally), "rows": rows,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n  every answer -> {out}")
    model.release()
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m measure.choosing", description=__doc__)
    p.add_argument("--model", default="qwen3:8b",
                   help="what ships. The baseline every early table compares against is qwen2.5:7b")
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--only", default=None, metavar="SCREEN", help="substring of a screen name")
    p.add_argument("--by-name", action="store_true",
                   help="ask for the exact name instead of a list index")
    p.add_argument("--shuffle", type=int, default=None, metavar="SEED",
                   help="shuffle each screen's order — separates an index error from a meaning one")
    p.add_argument("--think", choices=("on", "off"), default="off",
                   help="a reasoning model's thinking. **Defaults to off, which is what ships** — "
                        "leaving it unset used to omit the field, so qwen3:8b thought, and a run "
                        "meant to measure a prompt change measured the prompt AND the thinking "
                        "(5.63 s a call instead of 0.45 gave it away)")
    p.add_argument("--drop", default=None, metavar="ROLES",
                   help="comma-separated roles to leave out of the list, e.g. 'text'. Order is "
                        "preserved, because shuffling was measured to cost accuracy")
    p.add_argument("--prompt-record", action="store_true",
                   help="print the prompt exactly as converse() renders it, and its fingerprint, "
                        "then stop. No model, no screens, nothing written")
    p.add_argument("--prompt-check", action="store_true",
                   help="prove the fingerprint moves on the eight things that are the prompt and "
                        "stays put on the three that are only prose. Reads argmind/, writes "
                        "nothing, calls no model")
    a = p.parse_args(argv)
    if a.prompt_record:
        print(prompt_record())
        print(f"{PROMPT_LETTER}{prompt_print()}   (old scheme: p{converse_file_print()})")
        return 0
    if a.prompt_check:
        return prompt_check()
    return run(a.model, a.repeat, a.only, named=a.by_name, shuffle=a.shuffle,
               drop=tuple(r.strip() for r in a.drop.split(",")) if a.drop else (),
               think={"on": True, "off": False}.get(a.think))


if __name__ == "__main__":
    sys.exit(main())
