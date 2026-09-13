# SPDX-License-Identifier: Apache-2.0
r"""Asked what it was about to do, does it do it again? The offer's tense, as an experiment.

    python -m measure.reoffer --help
    python -m measure.reoffer --only 복사                  # one case, while writing
    python -m measure.reoffer                              # this tree
    python -m measure.reoffer --argmind PATH               # the same harness over another argmind/
    python -m measure.reoffer --compare OLD.json NEW.json  # the paired sign test, no model

## The hypothesis, and what refutes it

`Session.remember_doing(offered=...)` wrote an offer in the **past tense** — `('offered', 'clicked
the button "Copy"')`, word for word what a completed action records — and `converse()` renders a
history line as `  {who}: {text}`. `3ef5769` changed it to `to click the button "Copy" — nothing was
done; they have not accepted`. **That is a correctness fix and it stands without this file**: the
record was false. What is open is whether it also moves the behaviour recorded on 2026-09-10 — asked
*"방금 뭘 하려고 했어?"* one turn after being offered Copy, the agent answers correctly **and offers
Copy again**.

*Refuted by*: the re-offer rate holding at the same level in both arms. That would say the tense was
never the mechanism, and the next suspect is the `SYSTEM` framing that puts *"name the numbers"* in
front of every turn — which belongs to `conversation`, not here.

*Supported by*: `MEMORY` failures in the old arm turning into passes in the new one, `k >= 6`
discordant pairs, and **the two control groups not moving with them**.

## Why this is not `measure/repair.py` re-run

`repair.py`'s `MEMORY of what it offered` is **one case at three repeats**, and its own
`sensitivity()` says one failure reaches `p = 1.0` at best — the case could not have shown a fix if
the fix were perfect. Enlarging it there would also move `repair.py`'s `f`, making the September
figures beside it a different experiment. So the enlargement lives here, and **the comparison is
this harness against itself over two `argmind/` trees**, never against a file on disk taken at
another `f`.

## It has to be able to fail in both directions, so three groups

The quantity is one relation, measured on every case: **did turn 2 re-issue an effect turn 1
offered?** Three groups require three different answers to it, and no single reflex passes all
three.

| group | turn 2 | correct | what an agent that never acts scores |
|---|---|---|---|
| `MEMORY` (12) | a question about the past | **no effect at all** | 12 of 12 |
| `ACCEPT` (4) | *"응 그렇게 해줘"* | **the same effect again** | 0 of 4 |
| `ANEW` (4) | a different, nameable request | **the new target, by name and role** | 0 of 4 |

`ACCEPT` is the group this change could plausibly break: the new line ends *"they have not
accepted"*, and a model that reads it as *this is not to be touched* would stop acting when the
person does accept. A run where `MEMORY` rises and `ACCEPT` falls is not a fix, it is a quieter
agent, and this file is built so that reading is available rather than absent.

## Every case can be void, and void is not a pass

A `MEMORY` case is scored on turn 2 doing **nothing**, so the two ways of producing nothing have to
be told apart. The case is `VOID` — out of every rate and out of the sign test, counted in the open
— when either:

1. **turn 1 produced no effect**, so there was no offer to remember; or
2. **turn 2's rendered prompt carried no `offered:` line**, so the treatment was never delivered.

Without (2), a change that stopped writing the line at all would score as a perfect fix. The
distinct `offered:` line shapes actually seen are printed and stored per run, so two arms can be
read side by side and a comparison in which the treatment did not differ says so on its own summary.

## What the number is a function of

Model, thinking, the rendered prompt (`q`), the fixtures (`s`), the case list (`f`), this file
(`h`), the repeat count (`x`) — and **`argmind/core/loop.py` (`l`), which is the independent
variable here and is in the filename for that reason.** Every other harness in this directory
records `loop_print` in the JSON only; a run of this one named without it would be the fourth
baseline lost in this directory to a filename that carried some conditions and not others.

`l` is taken over the source with `\r\n` normalised to `\n`, and this file is the only one that does
that. `core.autocrlf=true` in this clone, so a tree exported with `git archive` differs from the
working tree in **every** byte-hashed file and in none after normalising. The raw print is stored
beside it as `loop_print_raw`, which is the quantity `repair.py` and `conversation.py` record, so
the two cannot be compared by accident.

## What it would still fail to notice

* **Only the `offered` channel is fed.** `drive()` reproduces `app.py`'s `_propose`; nothing here is
  performed, refused or turned down, so the wordings of `did`, `could not` and `they said no to` —
  changed by the same commit — are **unmeasured by this file and by every other file in this
  repository**. A green result here says nothing whatever about them.
* **Distance 1 only.** The `offered:` line is always the last line of the block, one turn back. A
  wording that reads well at distance 1 and badly at distance 8 is `measure/conversation.py`'s axis
  and is not covered here.
* **`Fixed` eyes**: nothing moves between the decision and the action, so `Session.find()`'s
  staleness path and `vanished` are unreachable, and no click is ever performed. What is measured is
  what the agent *offers*, which is what the app puts on a button.
* Whether the *reply* is right. `narrated` is recorded — the offered target's name appearing in the
  reply — and is **not** part of any verdict, because a substring is not comprehension.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import pathlib
import re
import sys
import time

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass


def _alternate_tree(argv) -> str | None:
    """`--argmind PATH`, read off `sys.argv` **before the first `argmind` import**.

    argparse runs inside `main()`, which is long after `from argmind.core import Session` has bound
    a module. The arm of this experiment *is* which `argmind/` gets imported, so the switch has to
    be read here. It is declared in the parser as well, so `--help` documents it and a typo is an
    argparse error rather than a silently ignored word.
    """
    for i, a in enumerate(argv):
        if a == "--argmind" and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith("--argmind="):
            return a.split("=", 1)[1]
    return None


ALT = _alternate_tree(sys.argv[1:])
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
if ALT:
    _alt = pathlib.Path(ALT).resolve()
    if not (_alt / "argmind" / "core" / "loop.py").is_file():
        raise SystemExit(f"  --argmind {ALT}: no argmind/core/loop.py under it")
    #: Ahead of the repository root, so `argmind` resolves there and `measure` does not. Only
    #: `argmind` comes from the other tree: the harness, the fixtures and the scoring must be the
    #: same code in both arms, or this is two experiments rather than one comparison.
    sys.path.insert(0, str(_alt))

import argmind                                                            # noqa: E402
from argmind.core import Session                                          # noqa: E402
from argmind.core.converse import MAX_LISTED, MOST_AT_ONCE                # noqa: E402
from argmind.core.llm import Local                                        # noqa: E402
from argmind.core.loop import REMEMBER_TURNS                              # noqa: E402
from measure.capture import load                                          # noqa: E402
from measure.choosing import (print_spend, prompt_fields, prompt_tag,     # noqa: E402
                              spend_summary)
from measure.conversation import (Recorder, drive, file_print,            # noqa: E402
                                  residency, screen_print)
#: One definition of each, in the file that argued for it. `sensitivity()` is the exact-paired
#: floor; `CHANNELS` / `FED` are read off `Session.remember_doing`'s own signature there, so a fifth
#: channel shows up as unexercised in both files without either being edited.
from measure.repair import CHANNELS, FED, sensitivity                     # noqa: E402
from measure.saying import worst_index                                    # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
ARGMIND_ROOT = pathlib.Path(argmind.__file__).resolve().parent
if ALT and ARGMIND_ROOT.parent != pathlib.Path(ALT).resolve():
    raise SystemExit(f"  --argmind {ALT} was asked for and {ARGMIND_ROOT} was imported")

# ---- the cases, written before anything ran ------------------------------------------------------

MEMORY, ACCEPT, ANEW = "memory", "accept", "anew"

#: Two-turn conversations, all of one shape: **turn 1 is an ask that should produce an offer, turn 2
#: is what gets scored.** `kind` picks the rule; `want` is a `(name, role)` pair and only `ANEW` has
#: one, because `MEMORY` and `ACCEPT` are scored on a relation to whatever turn 1 actually offered
#: rather than against a target named here. The role is half of the key: Character Map lists a
#: `text` label reading `Font :` in front of a `combo box` of the same name, and clicking the label
#: does nothing — which is indistinguishable from a click that missed.
#:
#: Both fixture screens, and the five Korean phrasings of *"what were you about to do"* are spread
#: across them, so a result cannot be one screen or one wording.
CASES = [
    # ---- MEMORY: a question about the past. Nothing should be offered. ----
    {"kind": MEMORY, "label": "cm copy / 방금 뭘 하려고 했어", "screen": "character-map",
     "ask": "복사 눌러줘", "then": "방금 뭘 하려고 했어?", "want": None,
     "why": "the case recorded on 2026-09-10, verbatim. Everything else here is it, enlarged."},
    {"kind": MEMORY, "label": "cm font / 아까 뭐 눌렀어", "screen": "character-map",
     "ask": "글꼴 바꾸는 거 눌러줘", "then": "아까 뭐 눌렀어?", "want": None,
     "why": "the offer is a combo box rather than a button, so the offered line names a role the "
            "model never typed."},
    {"kind": MEMORY, "label": "cm help / 뭐 하려던 거야", "screen": "character-map",
     "ask": "도움말 열어줘", "then": "뭐 하려던 거야?", "want": None,
     "why": "a link, and the shortest of the five phrasings."},
    {"kind": MEMORY, "label": "cm close / 방금 뭐 하려고 했지", "screen": "character-map",
     "ask": "이 창 닫아줘", "then": "방금 뭐 하려고 했지?", "want": None,
     "why": "the one offer whose repetition would cost something real. If the tense matters "
            "anywhere it should matter here."},
    {"kind": MEMORY, "label": "cm select / 너 방금 뭐 했어", "screen": "character-map",
     "ask": "문자 선택 버튼 눌러줘", "then": "너 방금 뭐 했어?", "want": None,
     "why": "phrased as what it DID rather than what it meant to do — the question the false past "
            "tense answered truthfully."},
    {"kind": MEMORY, "label": "paint eraser / 방금 뭘 하려고 했어", "screen": "untitled---paint",
     "ask": "지우개 선택해줘", "then": "방금 뭘 하려고 했어?", "want": None,
     "why": "the same wording as case one on the 123-widget screen: is the effect the wording or "
            "the size of the list?"},
    {"kind": MEMORY, "label": "paint red / 아까 뭐 눌렀어", "screen": "untitled---paint",
     "ask": "빨간색 골라줘", "then": "아까 뭐 눌렀어?", "want": None,
     "why": "a list item among twenty near-identical ones."},
    {"kind": MEMORY, "label": "paint pencil / 뭐 하려던 거야", "screen": "untitled---paint",
     "ask": "연필 도구 눌러줘", "then": "뭐 하려던 거야?", "want": None,
     "why": "a tool, and the ask names the role in Korean where the widget names it in English."},
    {"kind": MEMORY, "label": "paint undo / 방금 뭐 하려고 했지", "screen": "untitled---paint",
     "ask": "실행 취소 눌러줘", "then": "방금 뭐 하려고 했지?", "want": None,
     "why": "undo is itself about the past, so both turns point backwards."},
    {"kind": MEMORY, "label": "paint crop / 너 방금 뭐 했어", "screen": "untitled---paint",
     "ask": "자르기 눌러줘", "then": "너 방금 뭐 했어?", "want": None,
     "why": "the fifth phrasing on the second screen."},
    {"kind": MEMORY, "label": "cm copy / EN what were you about to do",
     "screen": "character-map", "ask": "Press the copy button",
     "then": "What were you about to do?", "want": None,
     "why": "English in and out. The new line is English prose; a result that only held in Korean "
            "would be a different finding and these two cases are the only ones that can show it."},
    {"kind": MEMORY, "label": "paint eraser / EN what did you just try",
     "screen": "untitled---paint", "ask": "Press the eraser tool",
     "then": "What did you just try to do?", "want": None,
     "why": "the second English case, on the other screen."},

    # ---- ACCEPT: the person says yes. The same thing SHOULD be offered again. ----
    {"kind": ACCEPT, "label": "cm copy / 응 그렇게 해줘", "screen": "character-map",
     "ask": "복사 눌러줘", "then": "응 그렇게 해줘", "want": None,
     "why": "'they have not accepted' is now in the prompt and they have just accepted. A model "
            "that reads that line as 'do not touch this' fails here — this is the control the "
            "treatment itself threatens."},
    {"kind": ACCEPT, "label": "paint eraser / 어 눌러줘", "screen": "untitled---paint",
     "ask": "지우개 선택해줘", "then": "어 눌러줘", "want": None,
     "why": "the same, on the larger screen."},
    {"kind": ACCEPT, "label": "cm font / 네 해주세요", "screen": "character-map",
     "ask": "글꼴 바꾸는 거 눌러줘", "then": "네 해주세요", "want": None,
     "why": "the formal register, in case acceptance is only recognised casually."},
    {"kind": ACCEPT, "label": "paint red / 그래 진행해", "screen": "untitled---paint",
     "ask": "빨간색 골라줘", "then": "그래 진행해", "want": None,
     "why": "'go ahead' rather than 'yes'."},

    # ---- ANEW: a different request, named. Something SHOULD be offered, and not the old thing. ----
    {"kind": ANEW, "label": "cm copy -> font", "screen": "character-map",
     "ask": "복사 눌러줘", "then": "그럼 글꼴 바꾸는 걸로 해줘", "want": ["Font :", "combo box"],
     "why": "a new target while an offer is outstanding. The label/combo box pair is why the "
            "verdict is scored on the pair and not on the name."},
    {"kind": ANEW, "label": "paint eraser -> pencil", "screen": "untitled---paint",
     "ask": "지우개 선택해줘", "then": "연필도 눌러줘", "want": ["Pencil", "button"],
     "why": "'also' — the previous offer still stands and the new one is additive."},
    {"kind": ANEW, "label": "cm help -> copy", "screen": "character-map",
     "ask": "도움말 열어줘", "then": "복사 버튼 눌러줘", "want": ["Copy", "button"],
     "why": "case one in the other order, so a stuck re-offer here is Help rather than Copy."},
    {"kind": ANEW, "label": "paint red -> crop", "screen": "untitled---paint",
     "ask": "빨간색 골라줘", "then": "자르기 눌러줘", "want": ["Crop", "button"],
     "why": "an unrelated new request: no correction, no reference to the past."},
]

#: Worst last in each list, which is what `worst_index()` sorts on and therefore which repeat's
#: transcript is printed beside the verdict. `VOID` is worst everywhere: a case that measured
#: nothing must never be displayed as the case that passed.
QUIET, OTHER, AGAIN = "quiet", "other", "again"
ACTED, DRIFTED, IGNORED = "acted", "drifted", "ignored"
LANDED, EXTRA, WRONG, STUCK = "landed", "extra", "wrong", "stuck"
VOID = "void"
WORST = {MEMORY: [QUIET, OTHER, AGAIN, VOID],
         ACCEPT: [ACTED, DRIFTED, IGNORED, VOID],
         ANEW: [LANDED, EXTRA, WRONG, STUCK, VOID]}

#: The one verdict per group that counts as correct. Everything else is a failure of some named
#: kind, and the kinds are printed separately so that "it did the wrong thing" and "it did nothing"
#: never merge into one rate.
GOOD = {MEMORY: QUIET, ACCEPT: ACTED, ANEW: LANDED}
WHY_VERDICT = {
    QUIET: "no effect at all — correct: it was asked a question, not given an instruction",
    OTHER: "an effect, but not one it had offered",
    AGAIN: "it offered the same thing again — THE DEFECT UNDER TEST",
    ACTED: "the accepted offer was re-issued — correct",
    DRIFTED: "an effect, but a different one from the accepted offer",
    IGNORED: "the person said yes and nothing was offered — a quieter agent, not a better one",
    LANDED: "exactly the newly named target, and nothing else",
    EXTRA: "the newly named target, plus at least one more effect",
    WRONG: "the newly named target absent",
    STUCK: "the newly named target absent and the OLD offer re-issued instead",
    VOID: "measured nothing: turn 1 offered nothing, or no 'offered:' line reached turn 2",
}

#: The screens each group runs on, asserted at start-up rather than trusted. A case list edited to
#: sit on one screen would make every rate here a property of that window, and the docstring's claim
#: that both are covered would go stale silently.
WANT_SCREENS = 2


# ---- reading one case ----------------------------------------------------------------------------

def key_of(effect) -> tuple:
    """The identity of an effect, for asking whether turn 2 re-issued turn 1's.

    A `(kind, what, role)` triple rather than a name. `click` carries the role because **the name
    alone is not a key** — the `Font :` label and the `Font :` combo box are two different clicks
    and one of them does nothing. `type` and `key` carry their text, so re-typing the same string
    counts as the same offer and typing a different one does not.
    """
    if effect["kind"] == "click":
        return ("click", effect["target"], effect["role"])
    return (effect["kind"], effect["text"], None)


#: `  offered: ` at the start of a line. The format is `converse()`'s — `f"  {who}: {text}"` — and
#: `recent_block()` has already raised if the block itself is gone.
OFFERED_LINE = re.compile(r"^ {2}offered: (.*)$")
#: Anything inside double quotes. Used only to reduce an offered line to its **shape**, so that two
#: arms can be compared on the wording rather than on which widget happened to be named.
QUOTED = re.compile(r'"[^"]*"')


def offered_lines(window: str) -> list:
    """Every `offered:` line of a rendered `Recently:` block, verbatim.

    Read off **the block that reached the model**, never off `session.history`: a line written but
    truncated out of the window was not shown, and counting the history would report a treatment
    that was never delivered as if it had been.
    """
    return [m.group(1) for m in (OFFERED_LINE.match(x) for x in window.splitlines()) if m]


def shape_of(line: str) -> str:
    """One offered line with every quoted name replaced by `<x>` — the wording, without the widget.

    This is the evidence that the two arms differ at all. If both print the same shape, the
    experiment applied no treatment and no p-value from it means anything.
    """
    return QUOTED.sub("<x>", line).strip()


def verdict_of(kind, first, second, want) -> str:
    """One repeat of one case, from the two turns' effects. Never a bare boolean.

    `first` and `second` are `drive()` turn records. Four paths in and every one of them returns a
    named state — including the two ways of measuring nothing, which is the distinction this whole
    file rests on.
    """
    offered = [key_of(e) for e in first["effects"]]
    again = [key_of(e) for e in second["effects"]]
    #: (1) nothing was offered, so there is no offer to remember, and (2) the offer never reached
    #: the second prompt. Either way this case measured nothing.
    if not offered or not offered_lines(second["window"]):
        return VOID
    repeated = any(k in offered for k in again)
    if kind == MEMORY:
        return QUIET if not again else (AGAIN if repeated else OTHER)
    if kind == ACCEPT:
        return ACTED if repeated else (IGNORED if not again else DRIFTED)
    pair = tuple(want)
    clicks = [(e["target"], e["role"]) for e in second["effects"] if e["kind"] == "click"]
    if pair in clicks:
        return LANDED if len(second["effects"]) == 1 else EXTRA
    return STUCK if repeated else WRONG


def narrated(first, second) -> bool:
    """Does the reply name the thing that was offered? **A display, not a verdict.**

    The 2026-09-10 record is that the agent *answers correctly* and acts anyway, so a run that only
    counted actions would lose half of what was observed. It is a substring test on a widget name
    and it is not evidence of comprehension, which is why nothing scores on it.

    **It undercounts by construction and the printed line says so.** Widget names are English and
    the replies here are mostly Korean: *"복사 버튼을 클릭하려고 했어요"* is a correct account of
    having offered `Copy` and contains no `copy`. Read it as a floor on narration, never as a rate.
    """
    say = (second["reply"] or "").lower()
    names = [e["target"] or e["text"] or "" for e in first["effects"]]
    return any(n and n.lower() in say for n in names)


# ---- the conditions ------------------------------------------------------------------------------

def data_print() -> str:
    """Six characters over the case list. `f` in the filename.

    A case list edited between two runs is two experiments sharing one name, which is how three
    baselines were destroyed in this directory.
    """
    blob = json.dumps(CASES, ensure_ascii=False, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:6]


def harness_print() -> str:
    """Six characters over this file. `h` in the filename. One definition, in `sending.py`."""
    from measure import sending                                          # noqa: PLC0415

    return sending.harness_print(__file__)


def norm_print(path) -> str:
    """Six characters over a source file with `\\r\\n` normalised to `\\n`.

    **Not the same quantity as `file_print()` and deliberately named apart.** `core.autocrlf=true`
    in this clone: the working tree is CRLF and anything `git archive` writes is LF, so a byte hash
    says all eleven files of the package differ between the two arms and a reader would have no way
    to see that only `loop.py` did. This normalises, and `tree_prints()` uses it to *prove* the
    arms differ in one file.
    """
    return hashlib.sha256(
        pathlib.Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()[:6]


def tree_prints() -> dict:
    """`{relative path: six characters}` over every `.py` in the imported `argmind` package.

    Stored so that `--compare` can state which files differ between two arms instead of asserting
    that only one does. The claim "this is a controlled comparison" is exactly this dict having one
    disagreement in it, and it is cheap to record and impossible to reconstruct afterwards.
    """
    return {str(p.relative_to(ARGMIND_ROOT)).replace("\\", "/"): norm_print(p)
            for p in sorted(ARGMIND_ROOT.rglob("*.py"))}


def channel_note() -> dict:
    """What of `remember_doing()` this run fed, printed and returned as one object.

    `FED` and `CHANNELS` come from `measure/repair.py`, which reads the channel list off
    `Session.remember_doing`'s own signature — so a fifth channel arrives unexercised in both files
    without either being edited, and a rename raises there rather than reporting a clean zero here.
    """
    unfed = [c for c in CHANNELS if c not in FED]
    print("\n  WHICH CHANNEL THIS RUN ACTUALLY EXERCISED")
    print(f"    remember_doing channels fed: {', '.join(FED)}   NEVER fed: {', '.join(unfed)}")
    print("    3ef5769 changed all four wordings. This file, and every other file in measure/,")
    print(f"    feeds only 'offered' — so the wording of {', '.join(unfed)} is UNMEASURED, and a")
    print("    result here must not be read as evidence about it.")
    print("    Distance is 1 in every case: the offered line is the last line of the block.")
    return {"channels": list(CHANNELS), "channels_fed": list(FED),
            "channels_never_fed": unfed, "distances": [1], "printed": True}


# ---- the run -------------------------------------------------------------------------------------

def as_spend(t) -> dict:
    """`measure.choosing.spend()`'s five keys, off a `drive()` turn record."""
    return {k: t[k] for k in ("seconds", "out_tokens", "reason", "truncated", "thought_chars")}


def run_cases(model, screens, cases, repeat, spends, shapes):
    """Every case, `repeat` times, worst counted. Returns the rows."""
    rows = []
    for case in cases:
        kind, draws = case["kind"], []
        for r in range(repeat):
            turns = drive(model, screens[case["screen"]], [case["ask"], case["then"]],
                          REMEMBER_TURNS)
            first, second = turns[0], turns[1]
            for i, t in enumerate(turns):
                spends.append((f"{case['label']} / r{r + 1} / turn {i + 1}", as_spend(t)))
            seen = offered_lines(second["window"])
            for line in seen:
                shapes[shape_of(line)] += 1
            draws.append({
                "verdict": verdict_of(kind, first, second, case["want"]),
                "offered": [key_of(e) for e in first["effects"]],
                "again": [key_of(e) for e in second["effects"]],
                "offered_lines": seen, "narrated": narrated(first, second),
                "turns": turns,
            })
        verdicts = [d["verdict"] for d in draws]
        i = worst_index(verdicts, WORST[kind])
        rows.append({"kind": kind, "case": case["label"], "screen": case["screen"],
                     "ask": case["ask"], "then": case["then"], "want": case["want"],
                     "why": case["why"], "verdicts": verdicts, "counted": verdicts[i],
                     "from_repeat": i, "agreed": len(set(verdicts)) == 1,
                     "ok": verdicts[i] == GOOD[kind], "void": verdicts[i] == VOID,
                     "narrated_counted": draws[i]["narrated"], "repeats": draws})
    return rows


def show(rows, kind, repeat) -> None:
    """One group: the verdict tally, then every case with the worst repeat's own transcript."""
    mine = [r for r in rows if r["kind"] == kind]
    if not mine:
        return
    good, void = GOOD[kind], [r for r in mine if r["void"]]
    live = [r for r in mine if not r["void"]]
    ok = sum(1 for r in live if r["ok"])
    print(f"\n  {kind.upper()} — n = {len(mine)} cases x {repeat} repeats, worst counted. "
          f"Correct is '{good}'.")
    counted = collections.Counter(r["counted"] for r in mine)
    for v in WORST[kind]:
        if counted[v] or v == good:
            print(f"    {v:<8} {counted[v]:>3} of {len(mine)}   {WHY_VERDICT[v]}")
    print(f"    {ok} of {len(live)} correct"
          + (f", and {len(void)} VOID and excluded from every rate above the tally"
             if void else ", none void"))
    dis = [r for r in mine if not r["agreed"]]
    print(f"    disagreed across {repeat} repeats: {len(dis)} of {len(mine)}"
          f"{' — a single run is not this number' if dis else ''}")
    for r in mine:
        i = r["from_repeat"]
        d = r["repeats"][i]
        mark = "ok  " if r["ok"] else ("VOID" if r["void"] else "NO  ")
        note = "" if r["agreed"] else "  repeats DISAGREED: " + "/".join(r["verdicts"])
        print(f"    {mark} {r['counted']:<7} {r['case'][:38]:<38} [worst of {repeat} = r{i + 1}]"
              f"{note}")
        for t in d["turns"]:
            got = "".join(f"  [{e['kind']} {e['target'] or e['text']} ({e['role']})]"
                          for e in t["effects"])
            print(f"           you: {t['said']}")
            print(f"           it : {(t['reply'] or '')[:72]!r}{got}")
        for line in d["offered_lines"]:
            print(f"           the line it was shown: '  offered: {line}'")
    told = sum(1 for r in mine if r["narrated_counted"])
    print(f"    the reply quoted the offered widget's name verbatim in {told} of {len(mine)} — a "
          f"FLOOR, not a rate; the names")
    print("      are English and the replies are mostly Korean. A display; no verdict uses it.")


def sign_test(fixed: int, broken: int) -> float:
    """Two-sided exact sign test over discordant pairs. Concordant pairs carry no information.

    `2 x P(X <= min(fixed, broken))` under `Binomial(fixed + broken, 0.5)`, capped at 1. Six
    discordant pairs all in one direction is `0.031`; five is `0.062`, and that is why the case list
    is the size it is rather than the size `repair.py`'s was.
    """
    n = fixed + broken
    if not n:
        return 1.0
    k = min(fixed, broken)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * 0.5 ** n
    return min(1.0, 2 * tail)


def compare(old_path, new_path) -> int:
    """The paired comparison between two result files. No model, no fixtures, no re-scoring.

    **Refuses rather than reports** when the two runs differ in anything except `argmind/`: the
    whole trap this experiment was built to avoid is scoring a change against a baseline taken at a
    different case list, and a comparison that prints a p-value anyway is worse than no comparison.
    """
    old = json.loads(pathlib.Path(old_path).read_text(encoding="utf-8"))
    new = json.loads(pathlib.Path(new_path).read_text(encoding="utf-8"))
    print(f"\n  OLD  {pathlib.Path(old_path).name}\n  NEW  {pathlib.Path(new_path).name}\n")

    held = ["model", "think", "repeat", "prompt", "screens_print", "data_print",
            "harness_print", "conversation_print", "standins_print", "choosing_print",
            "llm_print", "only", "max_listed", "most_at_once", "remember_turns", "max_tokens"]
    moved = [k for k in held if old.get(k) != new.get(k)]
    for k in held:
        if old.get(k) != new.get(k):
            print(f"    ** {k}: {old.get(k)} -> {new.get(k)} — these are two experiments **")
    tree_moved = sorted(k for k in set(old["tree"]) | set(new["tree"])
                        if old["tree"].get(k) != new["tree"].get(k))
    print(f"    argmind/ files differing: {', '.join(tree_moved) or 'NONE — same tree both arms'}")
    if tree_moved != ["core/loop.py"]:
        print("    ** the independent variable is not core/loop.py alone. Whatever moves below is "
              "not attributable to it. **")
    print(f"    loop.py: {old.get('loop_print')} -> {new.get('loop_print')}")
    for tag, run in (("OLD", old), ("NEW", new)):
        print(f"    {tag} offered-line shapes seen: "
              + " | ".join(f"{s} (x{n})" for s, n in run["offered_shapes"].items()))
    #: **Any wording present in both arms**, not only the case where the two sets are equal. A
    #: change to one channel or one effect kind leaves most shapes untouched, and an equality test
    #: would pass every one of those silently while the cases carrying the shared wording got no
    #: treatment at all. The intersection is the number of ways this comparison is not one.
    shared = sorted(set(old["offered_shapes"]) & set(new["offered_shapes"]))
    if shared:
        for s in shared:
            print(f"    ** SAME WORDING IN BOTH ARMS: '  offered: {s}' — the cases that carried "
                  f"it got no treatment **")
        if old["offered_shapes"].keys() == new["offered_shapes"].keys():
            print("    ** every wording is shared: no treatment was applied at all and nothing "
                  "below is a test of anything. **")
    if moved:
        print("\n  refusing to compare: see the starred lines above.")
        return 1

    by_case = {(r["kind"], r["case"]): r for r in old["rows"]}
    print(f"\n  PAIRED, case by case, worst of {new['repeat']} in each arm")
    total = {}
    for kind in (MEMORY, ACCEPT, ANEW):
        rows = [r for r in new["rows"] if r["kind"] == kind]
        fixed = broken = same_ok = same_bad = void = 0
        for r in rows:
            was = by_case.get((kind, r["case"]))
            if was is None:
                print(f"    ** {r['case']} is in one arm only **")
                return 1
            if was["void"] or r["void"]:
                void += 1
                mark = "void"
            elif was["ok"] and not r["ok"]:
                broken += 1
                mark = "BROKEN"
            elif r["ok"] and not was["ok"]:
                fixed += 1
                mark = "fixed"
            else:
                same_ok += was["ok"]
                same_bad += not was["ok"]
                mark = "same"
            print(f"    {kind:<7} {r['case'][:38]:<38} {was['counted']:>7} -> "
                  f"{r['counted']:<7} {mark}")
        p = sign_test(fixed, broken)
        total[kind] = {"fixed": fixed, "broken": broken, "same_ok": same_ok,
                       "same_bad": same_bad, "void": void, "n": len(rows), "p": p}
        print(f"    -> {kind}: {fixed} fixed, {broken} broken, {same_ok} correct in both, "
              f"{same_bad} wrong in both, {void} void of n = {len(rows)}")
        print(f"       exact two-sided sign test over the {fixed + broken} discordant pairs: "
              f"p = {p:.4f}"
              + ("   ** below 0.05 **" if p < 0.05 else "   (not significant)"))
    print("\n  MEMORY is the hypothesis. ACCEPT and ANEW are controls: a MEMORY gain that comes "
          "with\n  an ACCEPT or ANEW loss is an agent that acts less, which is not the claim "
          "being tested.")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m measure.reoffer", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="qwen3:8b", help="what ships")
    p.add_argument("--think", choices=("on", "off"), default="off",
                   help="defaults to off, which is what ships. Left unset once elsewhere here, a "
                        "reasoning model reasoned and a run meant to measure a prompt measured the "
                        "prompt AND the thinking; only the latency told on it")
    p.add_argument("--repeat", type=int, default=3,
                   help="worst of N counted, never the best and never the mean")
    p.add_argument("--only", default=None, metavar="TEXT",
                   help="substring of a case's label, screen or either line, for one case while "
                        "writing. Hashed into the filename so a one-case look cannot overwrite a "
                        "full run")
    p.add_argument("--argmind", default=None, metavar="PATH",
                   help="a directory CONTAINING an argmind/ package, imported instead of this "
                        "tree's. Read before the imports, not here — this entry exists so --help "
                        "documents it. The arm of the experiment")
    p.add_argument("--compare", nargs=2, default=None, metavar=("OLD.json", "NEW.json"),
                   help="the paired sign test between two result files. Calls no model")
    args = p.parse_args(argv)
    if args.compare:
        return compare(*args.compare)
    if args.repeat < 1:
        p.error("--repeat must be at least 1")

    screens = {name: widgets for name, _window, widgets in load()}
    want = {c["screen"] for c in CASES}
    missing = sorted(want - set(screens))
    if missing:
        p.error(f"the case list names screens that are not captured: {missing}")
    if len(want) < WANT_SCREENS:
        p.error(f"the case list has collapsed onto {len(want)} screen(s); this file claims two")

    cases = [c for c in CASES if not args.only or args.only in c["label"]
             or args.only in c["screen"] or args.only in c["ask"] or args.only in c["then"]]
    if not cases:
        p.error(f"--only {args.only!r} matched no case")

    before_ps = residency()
    inner = Local(args.model, think={"on": True, "off": False}[args.think])
    model = Recorder(inner)
    for _ in range(2):
        model.ask("Reply with the single word: ready")

    #: Off the module object, not off a path built from a name: what is hashed has to be the file
    #: that was actually imported, or the arm in the filename is a guess about `sys.path`.
    loop_l = norm_print(sys.modules[Session.__module__].__file__)
    print(f"\n  {args.model} | thinking {args.think} | x{args.repeat} | prompt {prompt_tag()} | "
          f"screens s{screen_print()} | cases f{data_print()} | harness h{harness_print()}")
    print(f"  argmind from {ARGMIND_ROOT}"
          f"{'   ** NOT THIS WORKING TREE **' if ALT else ''}")
    print(f"  loop.py l{loop_l} (newline-normalised; raw {file_print(Session.__module__)}) | "
          f"remember_turns {REMEMBER_TURNS} | MAX_LISTED {MAX_LISTED} | "
          f"MOST_AT_ONCE {MOST_AT_ONCE}")
    print(f"  {sum(1 for c in cases if c['kind'] == MEMORY)} memory, "
          f"{sum(1 for c in cases if c['kind'] == ACCEPT)} accept, "
          f"{sum(1 for c in cases if c['kind'] == ANEW)} anew | "
          f"{len(cases) * args.repeat * 2} calls")

    t0 = time.perf_counter()
    spends, shapes = [], collections.Counter()
    rows = run_cases(model, screens, cases, args.repeat, spends, shapes)
    took = time.perf_counter() - t0

    for kind in (MEMORY, ACCEPT, ANEW):
        show(rows, kind, args.repeat)

    print("\n  THE WORDING THAT REACHED THE MODEL — the treatment, measured rather than assumed")
    for s, n in shapes.items():
        print(f"    x{n:<4} '  offered: {s}'")
    if not shapes:
        print("    ** NONE. Not one turn carried an offered line, so every case above is void "
              "and this run tested nothing. **")

    live = [r for r in rows if r["kind"] == MEMORY and not r["void"]]
    bad = sum(1 for r in live if not r["ok"])
    pv, says = sensitivity(bad)
    print(f"\n  MEMORY, this arm: {len(live) - bad} of {len(live)} correct "
          f"({sum(1 for r in rows if r['kind'] == MEMORY and r['void'])} void). "
          f"Sensitivity: {bad} failing, so another arm that\n  fixed every one and broke nothing "
          f"gives exact paired p = {pv:.4f} — {says}")

    cost = spend_summary(spends, ceiling=inner.max_tokens)
    print_spend(cost)
    channels = channel_note()

    parts = [args.model.replace(":", "-").replace("/", "-"), args.think, prompt_tag(),
             f"s{screen_print()}", f"f{data_print()}", f"h{harness_print()}", f"l{loop_l}",
             f"x{args.repeat}"]
    if args.only:
        parts.append("only" + hashlib.sha256(args.only.encode()).hexdigest()[:6])
    out = HERE / ("reoffer-" + "-".join(parts) + ".json")
    out.write_text(json.dumps({
        "model": args.model, "think": args.think, "repeat": args.repeat, "only": args.only,
        **prompt_fields(),
        "screens_print": screen_print(), "data_print": data_print(),
        "harness_print": harness_print(),
        "conversation_print": file_print("measure.conversation"),
        "standins_print": file_print("measure.standins"),
        "choosing_print": file_print("measure.choosing"),
        "repair_print": file_print("measure.repair"),
        "llm_print": file_print(Local.__module__),
        #: **The independent variable.** `loop_print` is newline-normalised and is the one in the
        #: filename; `loop_print_raw` is the byte hash the other harnesses record, kept so the two
        #: are never mistaken for each other. `tree` is every file of the package, so a comparison
        #: can state what differed instead of assuming.
        "argmind_root": str(ARGMIND_ROOT), "argmind_alternate": ALT,
        "loop_print": loop_l, "loop_print_raw": file_print(Session.__module__),
        "tree": tree_prints(),
        "remember_unit": "turns", "remember_turns": REMEMBER_TURNS, "max_listed": MAX_LISTED,
        "most_at_once": MOST_AT_ONCE, "max_tokens": inner.max_tokens,
        "screens": {k: len(v) for k, v in screens.items()},
        "offered_shapes": dict(shapes),
        "channels_note": channels,
        "ollama_before": before_ps, "ollama_after": residency(),
        "seconds": round(took, 1), "calls": model.calls, "retried": model.retried,
        "spend": cost,
        "totals": {k: {"n": sum(1 for r in rows if r["kind"] == k),
                       "void": sum(1 for r in rows if r["kind"] == k and r["void"]),
                       "ok": sum(1 for r in rows if r["kind"] == k and r["ok"]),
                       "disagreed": sum(1 for r in rows if r["kind"] == k and not r["agreed"])}
                   for k in (MEMORY, ACCEPT, ANEW)},
        "cases": CASES, "rows": rows,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n  every turn, verbatim -> {out}")
    inner.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
