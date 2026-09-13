# SPDX-License-Identifier: Apache-2.0
"""What each harness actually sends, and therefore what it cannot see move.

Nothing here calls a model. Import it, send every turn through `turn()`, put `caveat()` on the
headline, and merge what `note()` returns into the result JSON - `note()` prints the block and hands
back the record, so there is no way to store this without having printed it.

## The defect this exists to make impossible

`measure/choosing.py`, `measure/saying.py` and `measure/talking.py` called
`converse(model, widgets, Said(text))` with no `history=`. `converse()` defaults it to `None`, so no
`Recently:` block was ever rendered and **nothing any of the three measures is a function of the
conversation**. That is not a gap in coverage anybody weighed; it is three call sites that omitted a
keyword argument, and it cost two wrong conclusions in two days:

* A reply-shape change was committed as *"improvement, all three harnesses same or better, no
  slide"*. It had taken rule-following from **36 of 36 probes to 14 of 36**. The three reproduced
  their baselines digit for digit under a prompt that makes the agent click the eraser when greeted.
* *"No reply in a 111-call sweep exceeds 36 output tokens"* was offered as evidence that closing the
  key list ended a degeneration. That sweep was `choosing.py`. With a conversation in the prompt,
  **18 of 1,845 turns still ran to the 4,096-token ceiling** at 69-71 s each - 47% of the model time
  in 1.0% of the turns.

Both are the same shape: a number that is not a function of X, quoted as evidence about X.

## What was NOT done, and why

**These three were not given a history arm.** Sending a conversation changes the prompt, so it
changes what they measure, and the four baselines at `qa62805` - a leak of 5 of 32, 21 of 28 right,
28 of 37 correct, 31 of 34 quiet - were expensive and would all have had to be re-taken to buy an
axis `measure/conversation.py` already covers properly, with controls these three do not have. A
change that quietly redefines `choosing` is worse than the blindness.

So the blindness is kept and made **structural** instead:

1. `turn()` is the only way into `converse()` from a harness, and its `history` is keyword-only with
   **no default**. A new harness cannot be blind by omission; it has to write the word.
2. Every turn is counted as it goes out, so the line printed at the end is a **measurement of this
   run** - "68 of 68 turns carried no conversation" - and not a sentence someone wrote once.
3. `note()` prints beside the headline, and `caveat()` goes **on** the headline, because the
   headline is the line that gets copied into a commit message.
4. The deferral names a file on disk. "`measure/conversation.py` covers this" is a docstring; "no
   result in this directory carries `qa62805`, the newest carries `q1ec131`" is a fact that goes
   stale by itself the moment the prompt moves.
5. `python -m measure.choosing --prompt-check` fails if any of 1-4 is dropped: an AST walk over
   `measure/*.py`, plus a byte-for-byte proof that `turn()` sends what a bare `converse()` sent.

## What this still would not notice

It is a **coverage** instrument, not a measurement. It cannot tell you the history behaviour is
fine; it can only refuse to let a history-free run be read as though it had. Specifically:

* A conversation result at the right `q` says a run happened, **not that it passed** - `note()`
  prints the filename and does not open it. Nothing here reads a verdict out of it.
* `q` covers the prompt and **not** `argmind/core/loop.py` - the window, `remember_turns`, and what
  a history line says. `deferral()` compares that file's hash as a second condition since
  2026-09-13, so a run at the right prompt under a different history writer no longer prints as
  coverage; what it still cannot see is a change to any *other* file of `argmind/` that moves
  behaviour without moving the rendered prompt. `measure/reoffer.py` records the whole tree for
  exactly that reason and this does not.
* The reverse blindness is not covered here at all: `conversation.py` sends conversations and never
  sends a 152-widget screen with no history, so *it* is blind to what these three measure. That is
  stated, not instrumented, because no run has yet been misread in that direction.
* A harness that imports `turn` and then calls `model.ask()` by hand sends whatever it likes. The
  AST rule catches a call to `converse`; it does not catch a prompt assembled from scratch, which is
  what `choosing.by_name()` legitimately does and why `by_name_print()` exists separately.
"""

from __future__ import annotations

import collections
import hashlib
import json
import pathlib
import re

from argmind.core import converse
#: Imported for its **file**, not for anything it does: `loop.py` decides what the history says and
#: how much of it is carried, and the deferral below is a claim about the history axis. See
#: `loop_now()`.
from argmind.core import loop as _loop

HERE = pathlib.Path(__file__).resolve().parent

#: Recorded here because `note()` prints this run's truncation count beside it, and a rate with
#: nothing to compare it against is a bare number. From `measure/conversation.py`'s d-sweep:
#: 3 of 1,845 at `q1ec131` ("do" first), 18 of 1,845 at `qa62805` ("say" first).
WITH_HISTORY_TRUNCATED = "18 of 1,845 turns (measure/conversation.py --axis d, qa62805, say-first)"

#: The harness whose result on disk `deferral()` goes looking for. **One entry, and the judgement
#: that reduced it to one is the paragraph below.** Data, not prose, because `note()` prints it and
#: `--prompt-check` asserts every file named here exists.
COVERS_HISTORY = {
    #: **The cost, measured rather than remembered.** This line said "~24 min" and the recorded run
    #: it describes took 2,671 s — 44.5 minutes for 1,872 calls at qa62805. A wrong cost is how a
    #: re-run gets deferred as unaffordable or scheduled as cheap; it is one field of the JSON.
    "conversation.py": "python -m measure.conversation --axis d   1,872 calls, ~45 min "
                       "(measured 2,671 s at qa62805)",
}

#: **`repair.py` was in the dict above until 2026-09-13 and `deferral()` never looked at it.**
#: `exercised_at_this_prompt` is computed from `conversation-*.json` alone, so three harnesses
#: printed *"covered by measure/repair.py"* against a claim nothing in this file checked. That is
#: the shape of defect this module exists to prevent, one level up: a coverage instrument with a
#: line in it that is not instrumented.
#:
#: Fixed by dropping the claim rather than by wiring up the check, and the judgement is:
#: **a distance-1 harness with no without-history arm is not coverage for "no slide".** The
#: blindness being deferred is that a prompt change can move what the agent does *mid-conversation*
#: while three history-free numbers hold digit for digit — and that is what took rule-following from
#: 36 of 36 probes to 14 of 36. Catching it needs an anchor far enough back to be truncated and a
#: control arm that says the anchor was what mattered. These two send a conversation and have
#: neither: every second turn sits one line after its offer, and there is no without-history cell to
#: compare against, so a slide at distance 8 leaves both of them unmoved. They are listed because
#: they *are* the two files whose numbers would move under a change to `loop.py`'s history writing —
#: `reoffer.py` exists to measure exactly that — and a reader deciding what to re-run should see
#: them. Listed, printed, and explicitly not counted.
EXERCISES_HISTORY = {
    "repair.py": "4 second turns per repeat, all at distance 1, no without-history arm",
    "reoffer.py": "44 scored turns per repeat, 40 of them one turn after their offer and 4 five "
                  "turns after it; no without-history arm",
}

#: Every file named anywhere above. `--prompt-check` walks this, so a harness renamed out from under
#: either dict fails a check rather than printing a filename nobody can open.
NAMED_HARNESSES = {**COVERS_HISTORY, **EXERCISES_HISTORY}


class _NoHistory:
    """The sentinel a harness passes to say it is sending no conversation **on purpose**.

    A distinct object rather than `None` or `()`, so the three states are three things: a real
    conversation, a deliberate absence, and a forgotten argument - which is now a `TypeError` at the
    call site instead of a silently empty `Recently:` block.
    """

    def __repr__(self) -> str:
        return "NO_HISTORY"

    def __bool__(self) -> bool:
        #: Falsy, so `if history:` inside `converse()` would behave as it does today. It never
        #: reaches there - `turn()` converts it to `None` - but a sentinel that would change
        #: behaviour if it leaked is a trap set for the one path that forgets to convert it.
        return False


NO_HISTORY = _NoHistory()

#: `{"with": n, "without": n}` over this process. Module-level on purpose: a harness is one run in
#: one process, and a counter the harness has to remember to increment is a counter that will
#: eventually disagree with what was sent.
_SENT = collections.Counter()


def turn(model, widgets, trigger, *, history, limit=None):
    """One turn through `converse()`, counted. `history` is keyword-only and has **no default**.

    Pass `NO_HISTORY` to mean "this harness deliberately sends none"; pass a sequence of
    `(who, text)` pairs to send one.

    **This must be prompt-identical to calling `converse()` directly**, or a refactor has re-defined
    every baseline in this directory. `--prompt-check` proves that byte for byte over the three
    fingerprint probes rather than asserting it here.
    """
    if history is NO_HISTORY:
        _SENT["without"] += 1
        real = None
    else:
        real = list(history) if history else None
        _SENT["with" if real else "without"] += 1
    #: `limit` is forwarded only when a caller sets one, so the default stays `converse()`'s own
    #: `MAX_LISTED` and this wrapper cannot pin a value the library later changes.
    extra = {} if limit is None else {"limit": limit}
    return converse(model, widgets, trigger, history=real, **extra)


def transparent(probes, capture) -> list:
    """Is `turn()` byte-identical to a bare `converse()`? `[(label, True/False)]`.

    **This lives here, not in the checker**, for a reason the checker itself found: `choosing.py`
    ran this comparison inline, which made `choosing.py` a file that calls `converse()` directly —
    and "no harness calls `converse()` directly" is one of the rules the checker enforces. It failed
    on itself within a minute of being written. A rule its own author has to be exempted from is a
    rule with a hole in it, so the exemption was removed instead of widened.

    `probes` are `(label, trigger, widgets, history)` and `capture` is a class whose instances
    record the prompt in `.sent` — `choosing._Capture`, passed in rather than rebuilt, because a
    second capturing stub is a second thing that can be wrong.
    """
    out = []
    for label, trigger, widgets, history in probes:
        direct, wrapped = capture(), capture()
        converse(direct, list(widgets), trigger, history=list(history) or None)
        turn(wrapped, list(widgets), trigger,
             history=list(history) if history else NO_HISTORY)
        out.append((label, direct.sent == wrapped.sent))
    #: The probes are not turns anybody measured, so they must not appear in this run's tally.
    reset()
    return out


def harness_print(path) -> str:
    """Six characters over **the harness file that produced a result**.

    One definition, here, because three of the five harnesses recorded this and two did not, and a
    condition that three files agree on and two omit is the drift `measure/` keeps being bitten by.
    `f` says the case list moved and `q` says the prompt moved; this says the code that turns a
    reply into a verdict moved, and a run under different code is a different experiment.

    The cost is that editing a comment gets a new filename. That is the cheaper direction here:
    three baselines in this directory were destroyed by a name that carried some conditions and not
    others, and no amount of cosmetic churn has ever cost anything comparable.
    """
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()[:6]


def norm_print(path) -> str:
    """Six characters over a source file with `\\r\\n` normalised to `\\n`. **Not `harness_print`.**

    `core.autocrlf=true` in this clone: the working tree is CRLF and anything `git archive` writes
    is LF, so a byte hash says every file of a package differs between an exported tree and this
    one, and none does after normalising. `measure/reoffer.py` needs that to state *which* file its
    two arms differ in; `measure/conversation.py` needs it so the `l` in its filename is the same
    quantity `reoffer.py` puts in its own.

    **The two quantities are deliberately named apart and must stay apart.** `reoffer.py`'s JSON
    calls the normalised one `loop_print` and the byte one `loop_print_raw`;
    `conversation.py`'s JSON calls the byte one `loop_print` — the key it has carried since
    2026-09-10, which is what the recorded files hold — and the normalised one `loop_print_norm`.
    One letter in a filename (`l`) means the normalised value in both, and a reader comparing a
    JSON key across the two files has to read this paragraph. That is the cost of not rewriting
    recorded results to match a later convention.
    """
    return hashlib.sha256(
        pathlib.Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()[:6]


def tally() -> dict:
    """What this run actually sent. `{"with": n, "without": n}`."""
    return {"with": _SENT["with"], "without": _SENT["without"]}


def reset() -> None:
    """For a check that runs two harnesses in one process. No harness calls this."""
    _SENT.clear()


# ---- is anything covering the axis this run is blind to? -----------------------------------------

def _prompt_of(path: pathlib.Path) -> str | None:
    """The **tagged** fingerprint a result file was written at - `q1ec131`, `p5461db` - or `None`.

    A filename is a claim and the JSON is the record; where they disagree the JSON wins. Only the
    head is read - these run to 1.9 MB and `prompt` sits fourth - with a full parse as the fallback,
    so a reordered writer degrades this to slow rather than to wrong.

    **The letter is reconstructed here and that is load-bearing.** The JSON stores `prompt` as six
    bare hex characters under both schemes, so comparing it against `prompt_tag()`'s `qa62805` by
    stripping the letter would let a pre-2026-09-10 `p` file - a hash of `converse.py`, a different
    quantity entirely - register as coverage on a hex collision. `prompt_scheme` was added with `q`
    and is absent from every `p` file, which is exactly the discriminator: no key, no `q`.
    """
    try:
        head = path.read_bytes()[:16384].decode("utf-8", "ignore")
        got = re.search(r'"prompt":\s*"([0-9a-f]{6})"', head)
        scheme = re.search(r'"prompt_scheme":\s*"(\w)', head)
        if not got:
            full = json.loads(path.read_text(encoding="utf-8"))
            got = full.get("prompt")
            letter = str(full.get("prompt_scheme", "p"))[:1]
            return None if not got else f"{letter}{got}"
    except (ValueError, OSError):
        return None
    return f"{scheme.group(1) if scheme else 'p'}{got.group(1)}"


def _loop_of(path: pathlib.Path) -> str | None:
    """The `loop_print` a conversation result recorded — the **byte** hash of `argmind/core/loop.py`
    as that run saw it — or `None` for a file written before the key existed.

    Same head-then-full-parse shape as `_prompt_of`, and the same rule: the JSON is the record.
    `None` means *not recorded*, which is a third state and must never print as *matches* — a run
    from before 2026-09-10 carries no `loop_print` at all.
    """
    try:
        head = path.read_bytes()[:65536].decode("utf-8", "ignore")
        got = re.search(r'"loop_print":\s*"([0-9a-f]{6})"', head)
        if got:
            return got.group(1)
        return json.loads(path.read_text(encoding="utf-8")).get("loop_print")
    except (ValueError, OSError):
        return None


def loop_now() -> str:
    """The byte hash of the `argmind/core/loop.py` **this process imported**.

    Resolved off the module object rather than off a path rebuilt from the package name: the second
    form is relative to the working directory and would quietly record the wrong file — or `None` —
    whenever a harness is run from anywhere but the repository root.

    **Why the deferral needs it at all.** `q` covers the prompt `converse.py` renders. It does not
    cover `loop.py`, which decides what a history line *says* and how many turns are carried, and
    on 2026-09-13 that file changed: an offer is no longer written in the past tense. Every
    conversation result on disk predates that. Without this, three harnesses would go on printing
    *"the axis was run at this prompt"* against a file taken under a different history writer, for
    as long as the prompt happened not to move — which is the same shape as the filename that
    carried the protocol and not the model.
    """
    return harness_print(_loop.__file__)


def conversation_runs() -> list:
    """Every `conversation-*.json` in this directory as `(path, q, loop_print, mtime)`, newest
    first."""
    got = [(p, _prompt_of(p), _loop_of(p), p.stat().st_mtime)
           for p in sorted(HERE.glob("conversation-*.json"))]
    #: `t[3]`, the mtime. It was `t[2]` and the tuple grew a member in front of it — the whole
    #: reason this line is commented is that a silent re-sort by hex string would have changed
    #: which run `newest_run` names without changing anything a reader could see.
    return sorted(got, key=lambda t: t[3], reverse=True)


def deferral(tag: str) -> dict:
    """Whether any conversation result on disk was taken at `tag` **and at this `loop.py`**, and
    which.

    **It reports existence, never a verdict.** A run at the right prompt and the right history
    writer says the axis was exercised at this tree; whether it passed is in that file and is not
    read here. The states print differently and none of them is a tick.

    **Two conditions, not one, since 2026-09-13.** `exercised_at_this_prompt` is kept because it is
    what the recorded JSONs in this directory hold and dropping it would silently redefine a field;
    `exercised_at_this_tree` is the one the caveat is now built on. A run at the same prompt under
    a different `loop.py` is a real reading of a real tree — it is just not a reading of *this*
    one, and the two must not print the same.
    """
    runs = conversation_runs()
    here = loop_now()
    #: Compared **with** the scheme letter. See `_prompt_of` - two schemes can collide on the six
    #: hex characters and can never collide on the letter, and a `p` file counting as coverage for
    #: a `q` prompt is the one way this instrument could print a tick it has not earned.
    at = [p for p, q, _l, _m in runs if q == tag]
    #: `l is not None` is load-bearing: a file from before the key existed has an unknown history
    #: writer, and unknown must not count as matching.
    at_tree = [p for p, q, l, _m in runs if q == tag and l is not None and l == here]
    return {
        "axis": "the conversation - a 'Recently:' block in the prompt",
        "sent_by_this_run": False,
        "covered_by": COVERS_HISTORY,
        #: Named and **not** counted — see the constant. Stored so a reader of the JSON gets the
        #: same distinction the printed block makes, rather than a shorter list with no reason.
        "exercises_only": EXERCISES_HISTORY,
        "prompt": tag,
        "loop_print": here,
        "runs_at_this_prompt": [p.name for p in at],
        "runs_at_this_prompt_and_loop": [p.name for p in at_tree],
        #: What the runs at the right prompt were taken under, so a reader sees the mismatch itself
        #: rather than a bare `False`. Never a bare boolean.
        "loop_prints_at_this_prompt": sorted({str(l) for p, q, l, _m in runs if q == tag}),
        "newest_run_at": (runs[0][1] if runs else None),
        "newest_run_loop": (runs[0][2] if runs else None),
        "newest_run": (runs[0][0].name if runs else None),
        #: The whole point of the field. `False` here is what makes "no slide" unsupported.
        "exercised_at_this_prompt": bool(at),
        "exercised_at_this_tree": bool(at_tree),
    }


def caveat(tag: str) -> str:
    """The clause that goes **on** the headline, short enough to survive being copied out of it.

    **Three states, since the history writer became a condition.** The middle one — a run at this
    prompt taken under a different `loop.py` — used to print as the first, and it is the state this
    repository is actually in today.
    """
    d = deferral(tag)
    if d["exercised_at_this_tree"]:
        return f"   [no conversation sent; conversation.py has run at {tag} + loop {d['loop_print']}]"
    if d["exercised_at_this_prompt"]:
        return (f"   [no conversation sent; conversation.py has run at {tag} but under a different "
                f"loop.py ({'/'.join(d['loop_prints_at_this_prompt'])} -> {d['loop_print']})]")
    return f"   [no conversation sent; NOTHING has run the history axis at {tag}]"


def note(tag: str, truncated=None, of=None) -> dict:
    """The block, printed after the summary, **returning the JSON record to store**.

    There is deliberately no way to record this without printing it, and no second entry point that
    recomputes it: a verdict printed beside a number taken from somewhere else is a defect this
    directory has shipped twice, so the harness writes back exactly the dict that was printed.

    `truncated` / `of` are this run's own token-ceiling count, printed beside the rate measured
    **with** a conversation in the prompt - the delta, because "0 of 68" means nothing on its own
    when the failure needs a history to appear at all.
    """
    d = deferral(tag)
    sent = tally()
    n = sent["with"] + sent["without"]
    print("\n  THE CONVERSATION - NOT EXERCISED BY THIS HARNESS")
    print(f"    {sent['without']} of {n} turns went out with history=NO_HISTORY; no 'Recently:' "
          f"block reached the model.")
    print("    So this number cannot move when a prompt change only changes what the agent does")
    print("    mid-conversation. That is not hypothetical: a reply-shape change held all three")
    print("    history-free harnesses and took rule-following from 36 of 36 probes to 14 of 36.")
    if truncated is not None and of:
        print(f"    token ceiling, this run: {truncated} of {of} replies. With a conversation in "
              f"the prompt: {WITH_HISTORY_TRUNCATED}.")
    for name, how in COVERS_HISTORY.items():
        print(f"    covered by     measure/{name:<18} {how}")
    for name, how in EXERCISES_HISTORY.items():
        print(f"    NOT coverage:  measure/{name:<18} {how}")
    print("    ** the second list sends a conversation and nothing here checks its result: "
          "distance 1,")
    print("       no without-history arm. A 'no slide' reading rests on the first list alone. **")
    if d["exercised_at_this_prompt"]:
        print(f"    at this prompt ({tag}): {len(d['runs_at_this_prompt'])} conversation result(s) "
              f"exist -")
        for name in d["runs_at_this_prompt"][:4]:
            print(f"        {name}")
        print("    ** that says the axis was RUN at this prompt, not that it passed. Nothing here")
        print("       reads its verdict. Open it before reporting anything as 'no slide'. **")
        #: **The second condition, added 2026-09-13.** `q` covers the prompt and not `loop.py`,
        #: which decides what a history line says; the tense of an offer changed that morning and
        #: every conversation result on disk predates it. Printed as the two hashes, not as a
        #: boolean, because "which loop.py" is the thing a reader has to go and check.
        if not d["exercised_at_this_tree"]:
            print(f"    ** and NONE of them at this loop.py: they were taken at "
                  f"{', '.join(d['loop_prints_at_this_prompt'])} and this tree is "
                  f"{d['loop_print']}.")
            print("       loop.py is what writes the history the prompt renders, so the axis is "
                  "unmeasured at")
            print("       THIS tree. Re-run measure/conversation.py before reading the above as "
                  "coverage. **")
        else:
            print(f"    {len(d['runs_at_this_prompt_and_loop'])} of them at this loop.py "
                  f"({d['loop_print']}) as well as this prompt.")
    else:
        newest = f"{d['newest_run_at']} ({d['newest_run']})" if d["newest_run"] else "none on disk"
        print(f"    at this prompt ({tag}): ** NO conversation result exists. Newest is "
              f"{newest}. **")
        print("    ** The history axis is UNMEASURED at this prompt, so a 'same or better, no "
              "slide'")
        print("       reading of the number above is supported by nothing in this directory. **")
    return {"history": {**d, "turns": sent, "printed": True}}
