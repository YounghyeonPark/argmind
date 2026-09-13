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

## It has to be able to fail in both directions, so six groups

The quantity is one relation, measured on every case: **did turn 2 re-issue an effect turn 1
offered?** Six groups require three different answers to it, and no single reflex passes them all.

| group | turn 2 | correct | never acts | acts always |
|---|---|---|---|---|
| `MEMORY` (24) | a question about the past | **no effect at all** | 24 of 24 | 0 of 24 |
| `REFUSE` (12) | *"아니 됐어"* | **no effect at all** | 12 of 12 | 0 of 12 |
| `ACCEPT` (40) | *"응"*, *"응 그렇게 해줘"*, *"yes"* | **the same effect again** | 0 of 40 | 40 of 40 |
| `FARACCEPT` (12) | the same, **five turns after the offer** | the same effect again | 0 of 12 | 12 of 12 |
| `RESTATE` (8) | the offered target **named** | that target, by name and role | 0 of 8 | part of 8 |
| `ANEW` (8) | a different, nameable request | the new target, by name and role | 0 of 8 | part of 8 |

Two reflexes and both are punished. An agent that never acts scores **36 of 104** and zero on the
four groups that ask for an action; an agent that acts on everything after an offer scores **52 of
104** plus whatever the two named-target groups give it, and zero on the two that ask for silence.
That second one is exactly what a fix for `ACCEPT` risks producing. `MEMORY` and `REFUSE` are the
controls that make the difference visible; `ANEW` and `RESTATE` say whether it can still act on
something *else*.

**Every group sits on all four captured screens, in the same proportions.** 12, 21, 123 and 152
widget lines; `main()` refuses to run if any (group, screen) pair is empty, and `ACCEPT` carries
exactly 6 phrased and 4 bare, 8 Korean and 2 English, **on each** of the four. Until 2026-09-13 two
screens carried every case and *123 lines* and *Paint* were one variable with two names.

## Pre-registered, 2026-09-13, before any fix to `ACCEPT` exists

**The open defect.** Saying yes to an offer did nothing in **2 of 4** cases at `f842432`, identical
in both tense arms: *"응 그렇게 해줘"* and *"네 해주세요"* acted, *"어 눌러줘"* and *"그래 진행해"*
were ignored. It costs more than the re-offer defect does — a re-offer costs a row somebody ignores,
this one means the thing they asked for does not happen and they have to say it again.

**Four is not scoreable and this file's own `sensitivity()` says so**: `k` discordant pairs give
`2 x 0.5^k`, so a perfect repair of two failures reaches `p = 0.5`. The set is enlarged **now,
before the attempt**, so that nobody has seen which cases a change helps.

* **Primary endpoint**: `ACCEPT`, 16 cases at distance 1, worst of 3 repeats, exact two-sided paired
  sign test over the discordant pairs. **`k = 6` all in one direction reaches `p = 0.031`; `k = 5`
  gives `0.062` and does not.** At the observed half-failing rate 16 cases give about 8 failures to
  work with, so a repair of six of eight is enough and a perfect one gives `0.0078`.
* **The fix is not accepted on that alone.** `MEMORY` and `REFUSE` must not lose more cases than
  they gain (`broken <= fixed` in each), or the agent has become one that acts on anything following
  an offer — which scores well here and is worse to use.
* **Secondary, reported separately and pooled into nothing**: `FARACCEPT` (does a yes five turns
  after the offer still connect to it — nobody has asked), and `RESTATE` (naming the target may
  work for a different reason entirely: it is a fresh instruction, not an acceptance).
* **What refutes the defect rather than the fix**: `ACCEPT` coming back at 15 or 16 of 16 on this
  baseline. Then *"saying yes does nothing, 2 of 4"* was four cases of noise, there is nothing to
  repair, and any later improvement is regression to the mean.

`ACCEPT` is also the group the **tense** change could have broken — the offered line now ends
*"they have not accepted"*, and a model reading that as *do not touch this* would stop acting when
the person does accept. It did not move: 2 of 4 in both arms.

## The second pre-registration, 2026-09-13, and what it does not get to claim

The plan above was scored at `96d8456` and **the criterion was not met**: `ACCEPT` 8 of 15 -> 12 of
15, four discordant pairs all one way, `p = 0.1250` against a floor that needs six. The change is
committed on trial and is not called a fix.

**The binding constraint is baseline failures, not cases.** The sign test can only see a repair of
something that failed; 16 cases gave 7 failures, and `k >= 6` needed six of those seven to be
repaired. So the set is enlarged — `ACCEPT` 16 -> 40 and `FARACCEPT` 4 -> 12 — and both arms are
re-run at the new `f` and the new `h`. The four committed result files stay exactly as they are.

**Both plans are printed at the top of every run and both are in the JSON.** The second one says
this in its own text, and it belongs here too: *scoring a fix that already exists against a case set
chosen now is not a pre-registration in the strong sense.* What is genuinely fixed in advance is the
criterion, the test, the controls and all 104 cases, hashed into `f` before either arm ran; what is
not is that the set was chosen by somebody who had seen which four cases the change repaired. **All
44 original cases are kept verbatim**, the three `ACCEPT` failures it did not repair included, so
the enlargement cannot be read as having replaced the cases that did not go its way. And:

> **This is one enlargement. The result stands whatever it says, and there is no third set.**

`ANEW` broke on one case at `96d8456` and is **not** in the criterion, because it was not in the
first one. Moving it in now would be choosing a criterion after seeing the data — in the direction
that fails the change, but chosen the same way. It is printed beside the criterion on every run.

## Two channels, and the treatment is measured in both

`96d8456` changed no wording in the transcript. It renders the unanswered offer **again**, between
the widget list and the person's words, and the `offered:` shapes are byte-identical in both arms —
so the shape check said *"every wording is shared: no treatment was applied at all"* beside a
comparison whose treatment was real. `slot_lines()` reads the other region off the prompt that was
actually sent, generically: it does not know what the fix says, it reports whatever is between the
widget list and `They said:`, which is **nothing** in the arm that predates the change. Both
channels are counted per turn, stored, and printed with `n` by `--compare`.

## Every case can be void, and void is not a pass

A `MEMORY` case is scored on turn 2 doing **nothing**, so the two ways of producing nothing have to
be told apart. The case is `VOID` — out of every rate and out of the sign test, counted in the open
— when either:

1. **turn 1 produced no effect**, so there was no offer to remember; or
2. **turn 1's own offer did not reach turn 2's rendered prompt** - no `offered:` line at all, or
   none naming what turn 1 offered - so the treatment was never delivered. Distance 1 always
   carries it; the twelve `FARACCEPT` cases sit five turns back and are the ones that could lose
   it, which is why the check is on the rendered block and never on `session.history`.

Without (2), a change that stopped writing the line at all would score as a perfect fix. The
distinct `offered:` line shapes actually seen are printed and stored per run, so two arms can be
read side by side and a comparison in which the treatment did not differ says so on its own summary.

## What the number is a function of

Model, thinking, the rendered prompt (`q`), the fixtures (`s`), the case list (`f`), this file
(`h`), the repeat count (`x`) — and **`argmind/core/loop.py` (`l`), which is the independent
variable here and is in the filename for that reason.** Every other harness in this directory
records `loop_print` in the JSON only; a run of this one named without it would be the fourth
baseline lost in this directory to a filename that carried some conditions and not others.

`l` is taken over the source with CRLF normalised to LF. `core.autocrlf=true` in this clone, so a
tree exported with `git archive` differs from the working tree in **every** byte-hashed file and in
none after normalising. The raw print is stored beside it as `loop_print_raw`, which is the
quantity `repair.py` records, so the two cannot be compared by accident. The definition moved to
`measure/sending.py` on 2026-09-13, when `conversation.py` grew an `l` of its own: one function,
so that two files cannot put two different quantities under one letter. **Their JSON keys still
disagree** - `conversation.py` stores the byte hash as `loop_print`, because every recorded
result of its own holds it under that key, and the normalised one as `loop_print_norm`.
See `sending.norm_print`.

## What it would still fail to notice

* **Only the `offered` channel is fed.** `drive()` reproduces `app.py`'s `_propose`; nothing here is
  performed, refused or turned down, so the wordings of `did`, `could not` and `they said no to` —
  changed by the same commit — are **unmeasured by this file and by every other file in this
  repository**. A green result here says nothing whatever about them.
* **The scored turn is one turn after the offer, except in twelve cases where it is five.**
  `FARACCEPT` puts four ordinary filler turns in between; everything else sits directly after its
  offer. A wording that reads
  well at distance 1 and badly at distance 8 is `measure/conversation.py`'s axis and is not covered
  here.
* **The screen is four points, not a variable that can be decomposed.** `chrome-remote-desktop` is
  the shortest prompt here *and* the only screen whose widgets are named in Korean; a result on that
  screen alone cannot say which of the two it is. The four are 12, 21, 123 and 152 widget lines with
  one application each, so *"152 lines"* and *"System Information"* are still one thing — four
  points on the line instead of two, which is enough to see a monotone trend and not enough to
  attribute it.
* **The utterances are not a random sample of anything.** They are what one person judged a
  plausible Korean or English acceptance, and the enlargement was written by somebody who had
  read the failing replies. Two arms of the same 104 remove the instrument from the comparison;
  they do not make the set representative of how anybody talks.
* **`Fixed` eyes**: nothing moves between the decision and the action, so `Session.find()`'s
  staleness path and `vanished` are unreachable, and no click is ever performed. What is measured is
  what the agent *offers*, which is what the app puts on a button.
* Whether the *reply* is right. `narrated` is recorded — the offered target's name appearing in the
  reply — and is **not** part of any verdict, because a substring is not comprehension. The one
  place the reply nearly mattered is the `ANEW` break, where the words were the baseline's exactly
  and the action slot was empty; `SILENT` names that failure now, and it names it off the effects,
  not off the text.
* **A `silent` verdict says the action slot was empty, not why.** Say-first drops the second field
  sometimes and a model can also decide to do nothing; this file cannot tell those apart, and
  `measure/saying.py` is where the field order is measured.
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
from measure import sending                                               # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
ARGMIND_ROOT = pathlib.Path(argmind.__file__).resolve().parent
if ALT and ARGMIND_ROOT.parent != pathlib.Path(ALT).resolve():
    raise SystemExit(f"  --argmind {ALT} was asked for and {ARGMIND_ROOT} was imported")

# ---- the cases, written before anything ran ------------------------------------------------------

MEMORY, ACCEPT, ANEW = "memory", "accept", "anew"
#: Added 2026-09-13, pre-registered. `FARACCEPT` is `ACCEPT` with four turns in between and is a
#: **separate kind, not a form of it**, so that the primary endpoint's sign test cannot be diluted
#: or rescued by a question nobody has asked before. `RESTATE` is scored by `ANEW`'s rule — the
#: person names a target, so there is a target to check — and it is separate for the same reason:
#: naming the thing may work because it is a fresh instruction rather than because an acceptance
#: was understood, and pooling the two would hide exactly that.
FARACCEPT, RESTATE, REFUSE = "faraccept", "restate", "refuse"
#: The order every table, tally and paired test walks. One list, so a group added here cannot be
#: silently missing from a summary — which is how a control gets dropped from a comparison.
GROUPS = (MEMORY, REFUSE, ACCEPT, FARACCEPT, RESTATE, ANEW)

#: Four ordinary turns between the offer and the yes, for `FARACCEPT`. **In the case list**, not
#: imported from `measure/conversation.py`'s `FILLER`, so `data_print()` covers them: filler edited
#: between two runs is two experiments under one `f`.
#:
#: Nothing here names a widget, asks about the past, or asks for the screen to be described — the
#: three things that would put the offer back into the window and turn a distance-4 probe into a
#: distance-1 one. `filler_check()` asserts the first of those against the real fixtures rather than
#: trusting this paragraph.
FILLER_FAR = [
    "오늘 날씨 어때?",
    "커피 한 잔 마시고 싶다",
    "너 이름이 뭐야?",
    "수고했어",
]

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
    {"kind": MEMORY, "lang": "en", "label": "cm copy / EN what were you about to do",
     "screen": "character-map", "ask": "Press the copy button",
     "then": "What were you about to do?", "want": None,
     "why": "English in and out. The new line is English prose; a result that only held in Korean "
            "would be a different finding and these two cases are the only ones that can show it."},
    {"kind": MEMORY, "lang": "en", "label": "paint eraser / EN what did you just try",
     "screen": "untitled---paint", "ask": "Press the eraser tool",
     "then": "What did you just try to do?", "want": None,
     "why": "the second English case, on the other screen."},

    #: **Six on each of the two screens this directory had captured and never used**, added
    #: 2026-09-13 with the `ACCEPT` enlargement. Until then `character-map` (21 widget lines) and
    #: `untitled---paint` (123) carried every `MEMORY` and `ACCEPT` case between them, so *123
    #: lines* and *Paint* were one variable wearing two names — the confound `96d8456`'s own message
    #: names as the reason for a larger set. `system-information` (152) and `chrome-remote-desktop`
    #: (12) put four points on that axis instead of two.
    #:
    #: **The five Korean phrasings and the two English ones are the same on all four screens**, so
    #: the wording is held while the screen moves. Six per screen, five Korean and one English,
    #: which is what the original twelve already were.
    {"kind": MEMORY, "label": "si find / 방금 뭘 하려고 했어", "screen": "system-information",
     "ask": "찾기 버튼 눌러줘", "then": "방금 뭘 하려고 했어?", "want": None,
     "why": "case one's wording on the longest prompt captured — 152 widget lines between the "
            "offer and the question."},
    {"kind": MEMORY, "label": "si hardware / 아까 뭐 눌렀어", "screen": "system-information",
     "ask": "하드웨어 리소스 눌러줘", "then": "아까 뭐 눌렀어?", "want": None,
     "why": "a tree item — a role neither of the first two screens offers, so the offered line "
            "names a kind of thing the set had never carried."},
    {"kind": MEMORY, "label": "si minimize / 뭐 하려던 거야", "screen": "system-information",
     "ask": "이 창 최소화해줘", "then": "뭐 하려던 거야?", "want": None,
     "why": "an offer whose repetition costs something visible, on the long screen."},
    {"kind": MEMORY, "label": "si edit menu / 방금 뭐 하려고 했지", "screen": "system-information",
     "ask": "편집 메뉴 눌러줘", "then": "방금 뭐 하려고 했지?", "want": None,
     "why": "a menu item on a screen that also lists a menu bar and a menu item both called "
            "'System' — the name is not the key here either."},
    {"kind": MEMORY, "label": "si summary / 너 방금 뭐 했어", "screen": "system-information",
     "ask": "시스템 요약 눌러줘", "then": "너 방금 뭐 했어?", "want": None,
     "why": "'System Summary' is three widgets — a tree item, a list and a header — so the offered "
            "line names one of three things wearing one name."},
    {"kind": MEMORY, "lang": "en", "label": "si find / EN what were you about to do",
     "screen": "system-information", "ask": "Press the Find button",
     "then": "What were you about to do?", "want": None,
     "why": "the English pair of case one, on the longest prompt."},
    {"kind": MEMORY, "label": "crd close / 방금 뭘 하려고 했어",
     "screen": "chrome-remote-desktop---qvb-flfm-dev1",
     "ask": "닫기 눌러줘", "then": "방금 뭘 하려고 했어?", "want": None,
     "why": "case one's wording on the shortest prompt captured — 12 widget lines. With the 152 of "
            "system-information it brackets the two screens the old set had."},
    {"kind": MEMORY, "label": "crd extensions / 아까 뭐 눌렀어",
     "screen": "chrome-remote-desktop---qvb-flfm-dev1",
     "ask": "확장 프로그램 눌러줘", "then": "아까 뭐 눌렀어?", "want": None,
     "why": "**the widget's own name is Korean**, so the offered line quotes Korean inside English "
            "prose. No other screen here can produce that, and `narrated` can finally be true of a "
            "Korean reply."},
    {"kind": MEMORY, "label": "crd minimize / 뭐 하려던 거야",
     "screen": "chrome-remote-desktop---qvb-flfm-dev1",
     "ask": "최소화 눌러줘", "then": "뭐 하려던 거야?", "want": None,
     "why": "the shortest phrasing on the shortest screen — the least of everything, and the case "
            "most likely to work if anything does."},
    {"kind": MEMORY, "label": "crd password / 방금 뭐 하려고 했지",
     "screen": "chrome-remote-desktop---qvb-flfm-dev1",
     "ask": "비밀번호 저장 눌러줘", "then": "방금 뭐 하려고 했지?", "want": None,
     "why": "saving a password is the one offer here a person would mind being repeated."},
    {"kind": MEMORY, "label": "crd settings / 너 방금 뭐 했어",
     "screen": "chrome-remote-desktop---qvb-flfm-dev1",
     "ask": "원격 데스크톱 설정 열어줘", "then": "너 방금 뭐 했어?", "want": None,
     "why": "the longest widget name in the fixtures — 'Chrome Remote Desktop 맞춤설정 및 제어', "
            "mixed script, quoted whole into the offered line."},
    {"kind": MEMORY, "lang": "en", "label": "crd extensions / EN what did you just try",
     "screen": "chrome-remote-desktop---qvb-flfm-dev1", "ask": "Press the extensions button",
     "then": "What did you just try to do?", "want": None,
     "why": "English in, and the name it has to hand back is Korean."},

    # ---- ACCEPT: the person says yes. The same thing SHOULD be offered again. ----
    #: **The original four first and verbatim**, as `repair.py`'s enlargement kept its original six:
    #: the two that failed and the two that passed are still in the set, in their own words, so the
    #: enlargement cannot be read as having quietly replaced the cases that showed the defect.
    #: `form` splits the set for display only — `bare`, `phrase` — and no verdict reads it.
    {"kind": ACCEPT, "form": "phrase", "label": "cm copy / 응 그렇게 해줘",
     "screen": "character-map", "ask": "복사 눌러줘", "then": "응 그렇게 해줘", "want": None,
     "why": "'they have not accepted' is now in the prompt and they have just accepted. A model "
            "that reads that line as 'do not touch this' fails here — this is the control the "
            "treatment itself threatens. ACTED at f842432 in both tense arms."},
    {"kind": ACCEPT, "form": "phrase", "label": "paint eraser / 어 눌러줘",
     "screen": "untitled---paint", "ask": "지우개 선택해줘", "then": "어 눌러줘", "want": None,
     "why": "the same, on the larger screen. IGNORED 3 of 3 in both arms — one of the two failures "
            "this enlargement exists to make scoreable."},
    {"kind": ACCEPT, "form": "phrase", "label": "cm font / 네 해주세요", "screen": "character-map",
     "ask": "글꼴 바꾸는 거 눌러줘", "then": "네 해주세요", "want": None,
     "why": "the formal register, in case acceptance is only recognised casually. ACTED."},
    {"kind": ACCEPT, "form": "phrase", "label": "paint red / 그래 진행해",
     "screen": "untitled---paint", "ask": "빨간색 골라줘", "then": "그래 진행해", "want": None,
     "why": "'go ahead' rather than 'yes'. IGNORED as the worst of 3 (acted once) — the second "
            "failure, and the one case in the group that disagreed with itself."},

    #: **Bare acceptance**: one or two syllables, no verb, nothing to parse as an instruction. If
    #: anything distinguishes the two that worked from the two that did not it should show here —
    #: *"응 그렇게 해줘"* carries 해줘 (*do it*) and *"어 눌러줘"* carries 눌러줘 (*press it*), so the
    #: pair that failed is not the one with less imperative in it, and a bare row is the control for
    #: that reading.
    {"kind": ACCEPT, "form": "bare", "label": "cm copy / 응", "screen": "character-map",
     "ask": "복사 눌러줘", "then": "응", "want": None,
     "why": "the shortest acceptance in the language, on the case that already works with a "
            "phrase. If this fails and the phrase passes, the fix is about parsing, not memory."},
    {"kind": ACCEPT, "form": "bare", "label": "paint eraser / 어", "screen": "untitled---paint",
     "ask": "지우개 선택해줘", "then": "어", "want": None,
     "why": "the same pairing on the case that already fails with a phrase."},
    {"kind": ACCEPT, "form": "bare", "label": "cm help / ㅇㅇ", "screen": "character-map",
     "ask": "도움말 열어줘", "then": "ㅇㅇ", "want": None,
     "why": "typed rather than spoken — two consonants, no vowel. A tokeniser's worst case and "
            "what people actually type."},
    {"kind": ACCEPT, "form": "bare", "label": "paint pencil / 그래", "screen": "untitled---paint",
     "ask": "연필 도구 눌러줘", "then": "그래", "want": None,
     "why": "'sure', which is also an ordinary conversational filler — the acceptance most easily "
            "read as small talk."},
    {"kind": ACCEPT, "form": "bare", "lang": "en", "label": "cm copy / EN yes", "screen": "character-map",
     "ask": "Press the copy button", "then": "yes", "want": None,
     "why": "English in and out. The offered line is English prose; a defect that only exists in "
            "Korean is a different finding and these four English cases are what can show it."},
    {"kind": ACCEPT, "form": "bare", "lang": "en", "label": "paint eraser / EN go ahead",
     "screen": "untitled---paint", "ask": "Press the eraser tool", "then": "go ahead",
     "want": None, "why": "the second English bare case, on the other screen."},

    #: Six more phrasings, spread over four offers this set has not yet accepted — a link, a check
    #: on the one destructive offer, a split button and an undo — so the group is not four asks
    #: wearing sixteen coats.
    {"kind": ACCEPT, "form": "phrase", "label": "cm close / 응 그거 해줘", "screen": "character-map",
     "ask": "이 창 닫아줘", "then": "응 그거 해줘", "want": None,
     "why": "the one acceptance whose repetition costs something real. If a fix makes the agent "
            "act on anything after an offer, this is where it hurts."},
    {"kind": ACCEPT, "form": "phrase", "label": "paint crop / 네 그렇게 해주세요",
     "screen": "untitled---paint", "ask": "자르기 눌러줘", "then": "네 그렇게 해주세요",
     "want": None, "why": "formal register on the larger screen; the pair of case 3."},
    {"kind": ACCEPT, "form": "phrase", "label": "cm select / 좋아 그렇게 하자",
     "screen": "character-map", "ask": "문자 선택 버튼 눌러줘", "then": "좋아 그렇게 하자",
     "want": None, "why": "'let's do that' — first person plural, no imperative at all."},
    {"kind": ACCEPT, "form": "phrase", "label": "paint undo / 어 그거로 해줘",
     "screen": "untitled---paint", "ask": "실행 취소 눌러줘", "then": "어 그거로 해줘",
     "want": None, "why": "'that one' — a pronoun pointing at the offer rather than a name."},
    {"kind": ACCEPT, "form": "phrase", "lang": "en", "label": "cm close / EN yes, please do that",
     "screen": "character-map", "ask": "Close this window", "then": "yes, please do that",
     "want": None, "why": "the English phrase against the English bare 'yes' on case 9."},
    {"kind": ACCEPT, "form": "phrase", "lang": "en", "label": "paint crop / EN okay, do it",
     "screen": "untitled---paint", "ask": "Press the crop button", "then": "okay, do it",
     "want": None, "why": "the fourth English case, on the other screen."},

    #: ---- the 2026-09-13 enlargement: 16 acceptances -> 40, and four screens instead of two ----
    #:
    #: **The binding constraint was never the number of cases, it was the number of cases that
    #: FAIL at baseline.** 16 gave 15 live and 7 failures, and the sign test can only see a repair
    #: of a failure: `k = 4` was reachable and `k >= 6` was not. 40 at the same failure rate gives
    #: about 19, and a repair as good as the one already observed — 4 of 7 — reaches `k >= 6` with
    #: room to spare. `planned_sensitivity()` prints that arithmetic before a single call is made.
    #:
    #: **Ten on every screen, and the same ten shapes on every screen**: 6 phrased and 4 bare, 8
    #: Korean and 2 English, and within those, phrased = 5 Korean + 1 English and bare = 3 Korean +
    #: 1 English. The original sixteen were already 5 phrased / 3 bare and 6 Korean / 2 English on
    #: **each** of the two screens they sat on; that property is why the two new screens get exactly
    #: that profile plus one phrased-Korean and one bare-Korean, and why `cm` and `paint` each get
    #: one of each as well. **Neither form nor language can be a stand-in for the screen**, which is
    #: the whole reason the new screens are here.
    #:
    #: The wordings cross the screens deliberately: *"응 그렇게 해줘"* was a Character Map case and
    #: *"어 눌러줘"* a Paint one, and both now appear on `si` and `crd` too. The four that remain on
    #: one screen only — *"응 그거 해줘"*, *"좋아 그렇게 하자"*, *"네 그렇게 해주세요"*, *"어 그거로
    #: 해줘"* — stay where they were because the original sixteen are kept verbatim.
    {"kind": ACCEPT, "form": "phrase", "label": "cm advanced / 응 그렇게 부탁해",
     "screen": "character-map", "ask": "고급 보기 체크해줘", "then": "응 그렇게 부탁해",
     "want": None, "why": "a check box, the one role on this screen no acceptance case had; and "
                          "the wording is the one shared by all four screens."},
    {"kind": ACCEPT, "form": "bare", "label": "cm minimize / 네", "screen": "character-map",
     "ask": "최소화해줘", "then": "네", "want": None,
     "why": "the bare polite yes — one syllable, and the register the phrased '네 해주세요' "
            "already passes in. If '네' fails where '네 해주세요' passes, length is the variable."},
    {"kind": ACCEPT, "form": "phrase", "label": "paint magnifier / 응 그렇게 부탁해",
     "screen": "untitled---paint", "ask": "돋보기 눌러줘", "then": "응 그렇게 부탁해",
     "want": None, "why": "the same wording as the Character Map case above, 102 widget lines "
                          "further down the prompt."},
    {"kind": ACCEPT, "form": "bare", "label": "paint save / 네", "screen": "untitled---paint",
     "ask": "저장 눌러줘", "then": "네", "want": None,
     "why": "the bare '네' on the long screen, and Save is an app bar button — a role the group "
            "carried nowhere else."},

    #: `system-information`, 152 widget lines: **the far end of the prompt-distance axis** the
    #: post-hoc split in `96d8456` could not test, because no acceptance case was on it.
    {"kind": ACCEPT, "form": "phrase", "label": "si find / 응 그렇게 해줘",
     "screen": "system-information", "ask": "찾기 버튼 눌러줘", "then": "응 그렇게 해줘",
     "want": None, "why": "the wording that passes on Character Map, on the longest prompt. If it "
                          "fails here and passes there, distance is doing the work."},
    {"kind": ACCEPT, "form": "phrase", "label": "si hardware / 어 눌러줘",
     "screen": "system-information", "ask": "하드웨어 리소스 눌러줘", "then": "어 눌러줘",
     "want": None, "why": "the wording that fails on Paint 3 of 3 — the case that literally asked "
                          "'which button?'. On a third screen it is no longer one screen's."},
    {"kind": ACCEPT, "form": "phrase", "label": "si minimize / 네 해주세요",
     "screen": "system-information", "ask": "이 창 최소화해줘", "then": "네 해주세요",
     "want": None, "why": "the formal register at 152 lines."},
    {"kind": ACCEPT, "form": "phrase", "label": "si edit menu / 그래 진행해",
     "screen": "system-information", "ask": "편집 메뉴 눌러줘", "then": "그래 진행해",
     "want": None, "why": "'go ahead' rather than 'yes', on a menu item."},
    {"kind": ACCEPT, "form": "phrase", "label": "si summary / 응 그렇게 부탁해",
     "screen": "system-information", "ask": "시스템 요약 눌러줘", "then": "응 그렇게 부탁해",
     "want": None, "why": "the offer names one of three widgets called 'System Summary', so "
                          "accepting it has to re-pick the same role as well as the same name."},
    {"kind": ACCEPT, "form": "phrase", "lang": "en", "label": "si find / EN okay, do it",
     "screen": "system-information", "ask": "Press the Find button", "then": "okay, do it",
     "want": None, "why": "English phrase, longest prompt. Pairs with the Paint case of the same "
                          "words so the English row is two wordings on four screens, not four."},
    {"kind": ACCEPT, "form": "bare", "label": "si find / 응", "screen": "system-information",
     "ask": "찾기 버튼 눌러줘", "then": "응", "want": None,
     "why": "the shortest acceptance in the language on the longest prompt — the far corner of "
            "both axes at once, and the cell most likely to fail."},
    {"kind": ACCEPT, "form": "bare", "label": "si components / 그래",
     "screen": "system-information", "ask": "구성 요소 눌러줘", "then": "그래", "want": None,
     "why": "'sure', which is also ordinary small talk, at 152 lines' distance from the offer."},
    {"kind": ACCEPT, "form": "bare", "label": "si software env / ㅇㅇ",
     "screen": "system-information", "ask": "소프트웨어 환경 눌러줘", "then": "ㅇㅇ", "want": None,
     "why": "two consonants and no vowel — the typed yes. On Character Map it came back with "
            "SYSTEM's own fallback sentence."},
    {"kind": ACCEPT, "form": "bare", "lang": "en", "label": "si minimize / EN yes",
     "screen": "system-information", "ask": "Minimize this window", "then": "yes", "want": None,
     "why": "the bare English yes on the long screen."},

    #: `chrome-remote-desktop`, 12 widget lines: **the near end of the same axis**, and the only
    #: screen in the fixtures whose widgets are named in Korean. Two things move at once here and
    #: that is stated rather than hidden — a result on this screen alone cannot separate a short
    #: prompt from a prompt whose target names are in the language of the request.
    {"kind": ACCEPT, "form": "phrase", "label": "crd close / 응 그렇게 해줘",
     "screen": "chrome-remote-desktop---qvb-flfm-dev1", "ask": "닫기 눌러줘",
     "then": "응 그렇게 해줘", "want": None,
     "why": "the Character Map wording on the shortest prompt there is. If acceptance is a "
            "distance effect this is the cell that should pass first."},
    {"kind": ACCEPT, "form": "phrase", "label": "crd extensions / 어 눌러줘",
     "screen": "chrome-remote-desktop---qvb-flfm-dev1", "ask": "확장 프로그램 눌러줘",
     "then": "어 눌러줘", "want": None,
     "why": "the Paint wording at 12 lines instead of 123, and the offered name is Korean. Failing "
            "here would say the wording, not the distance."},
    {"kind": ACCEPT, "form": "phrase", "label": "crd minimize / 네 해주세요",
     "screen": "chrome-remote-desktop---qvb-flfm-dev1", "ask": "최소화 눌러줘",
     "then": "네 해주세요", "want": None, "why": "the formal register on the short screen."},
    {"kind": ACCEPT, "form": "phrase", "label": "crd password / 그래 진행해",
     "screen": "chrome-remote-desktop---qvb-flfm-dev1", "ask": "비밀번호 저장 눌러줘",
     "then": "그래 진행해", "want": None,
     "why": "the acceptance on this screen that would cost something if it were dropped and the "
            "person had to say it twice."},
    {"kind": ACCEPT, "form": "phrase", "label": "crd settings / 응 그렇게 부탁해",
     "screen": "chrome-remote-desktop---qvb-flfm-dev1", "ask": "원격 데스크톱 설정 열어줘",
     "then": "응 그렇게 부탁해", "want": None,
     "why": "the four-screen wording again, on the longest widget name in the fixtures."},
    {"kind": ACCEPT, "form": "phrase", "lang": "en", "label": "crd close / EN yes, please do that",
     "screen": "chrome-remote-desktop---qvb-flfm-dev1", "ask": "Close this window",
     "then": "yes, please do that", "want": None,
     "why": "English in, a Korean widget name offered back, and an English acceptance of it."},
    {"kind": ACCEPT, "form": "bare", "label": "crd close / 응",
     "screen": "chrome-remote-desktop---qvb-flfm-dev1", "ask": "닫기 눌러줘", "then": "응",
     "want": None, "why": "the near corner against `si find / 응`'s far one: same two turns, same "
                          "two words, 140 widget lines of difference."},
    {"kind": ACCEPT, "form": "bare", "label": "crd restore / 그래",
     "screen": "chrome-remote-desktop---qvb-flfm-dev1", "ask": "창 복구해줘", "then": "그래",
     "want": None, "why": "'sure' on the short screen; the ask names the action where the widget "
                          "names the state ('복구')."},
    {"kind": ACCEPT, "form": "bare", "label": "crd extensions / ㅇㅇ",
     "screen": "chrome-remote-desktop---qvb-flfm-dev1", "ask": "확장 프로그램 눌러줘",
     "then": "ㅇㅇ", "want": None, "why": "the typed yes on the short screen."},
    {"kind": ACCEPT, "form": "bare", "lang": "en", "label": "crd extensions / EN go ahead",
     "screen": "chrome-remote-desktop---qvb-flfm-dev1", "ask": "Press the extensions button",
     "then": "go ahead", "want": None,
     "why": "the fourth English bare case; the pair of Paint's, on the other end of the axis."},

    # ---- FARACCEPT: the same yes, with four ordinary turns in between. ----
    #: **Cheap, and nobody has asked it.** Every other case in this file sits one line after its
    #: offer. If a yes at distance 4 is not connected to the offer, "acceptance" is a
    #: last-line-of-the-prompt effect and the fix belongs somewhere else entirely. The window is 12
    #: turns, so the offer is still shown — `offer_reached()` checks that per run rather than
    #: assuming it, and a case where it was truncated is VOID, not a failure.
    {"kind": FARACCEPT, "label": "cm copy far / 응 해줘", "screen": "character-map",
     "ask": "복사 눌러줘", "filler": FILLER_FAR, "then": "응 해줘", "want": None,
     "why": "the case that accepts correctly at distance 1: if it fails here, distance is the "
            "variable and nothing about the wording is."},
    {"kind": FARACCEPT, "label": "paint eraser far / 응 해줘", "screen": "untitled---paint",
     "ask": "지우개 선택해줘", "filler": FILLER_FAR, "then": "응 해줘", "want": None,
     "why": "the case that fails at distance 1. Failing here as well is the null result."},
    {"kind": FARACCEPT, "label": "cm font far / 그래 해줘", "screen": "character-map",
     "ask": "글꼴 바꾸는 거 눌러줘", "filler": FILLER_FAR, "then": "그래 해줘", "want": None,
     "why": "a combo box rather than a button, four turns back."},
    {"kind": FARACCEPT, "lang": "mixed", "label": "paint red far / EN yes go ahead",
     "screen": "untitled---paint",
     "ask": "빨간색 골라줘", "filler": FILLER_FAR, "then": "yes go ahead", "want": None,
     "why": "English acceptance of a Korean ask, at distance. Two things at once, deliberately: "
            "this one is a probe, not a control."},

    #: **Four to twelve, three on each screen, two Korean and one mixed each.** `FARACCEPT` was 0
    #: of 4 at baseline and 3 of 4 after `96d8456`, which is the sharpest single reading this file
    #: has produced and also its smallest group. It stays secondary and in no criterion; it is
    #: enlarged because a group that moved 0 -> 3 on four cases is a group that deserves a
    #: denominator, and because at four it sat on two screens like everything else did.
    {"kind": FARACCEPT, "lang": "mixed", "label": "cm close far / EN yes go ahead",
     "screen": "character-map", "ask": "이 창 닫아줘", "filler": FILLER_FAR,
     "then": "yes go ahead", "want": None,
     "why": "the mixed case Character Map lacked, so all four screens now carry one."},
    {"kind": FARACCEPT, "label": "paint crop far / 그래 해줘", "screen": "untitled---paint",
     "ask": "자르기 눌러줘", "filler": FILLER_FAR, "then": "그래 해줘", "want": None,
     "why": "the second Korean far case on Paint, on an offer that accepts correctly at "
            "distance 1."},
    {"kind": FARACCEPT, "label": "si find far / 응 해줘", "screen": "system-information",
     "ask": "찾기 버튼 눌러줘", "filler": FILLER_FAR, "then": "응 해줘", "want": None,
     "why": "distance 5 on the longest prompt — conversational distance and prompt distance at "
            "their maxima together."},
    {"kind": FARACCEPT, "label": "si minimize far / 그래 해줘", "screen": "system-information",
     "ask": "이 창 최소화해줘", "filler": FILLER_FAR, "then": "그래 해줘", "want": None,
     "why": "the second Korean far case on the long screen."},
    {"kind": FARACCEPT, "lang": "mixed", "label": "si hardware far / EN yes go ahead",
     "screen": "system-information", "ask": "하드웨어 리소스 눌러줘", "filler": FILLER_FAR,
     "then": "yes go ahead", "want": None,
     "why": "English acceptance of a Korean ask, five turns back, on a tree item."},
    {"kind": FARACCEPT, "label": "crd close far / 응 해줘",
     "screen": "chrome-remote-desktop---qvb-flfm-dev1", "ask": "닫기 눌러줘",
     "filler": FILLER_FAR, "then": "응 해줘", "want": None,
     "why": "distance 5 on the shortest prompt: if the offer is still reachable anywhere, here."},
    {"kind": FARACCEPT, "label": "crd extensions far / 그래 해줘",
     "screen": "chrome-remote-desktop---qvb-flfm-dev1", "ask": "확장 프로그램 눌러줘",
     "filler": FILLER_FAR, "then": "그래 해줘", "want": None,
     "why": "the second Korean far case on the short screen."},
    {"kind": FARACCEPT, "lang": "mixed", "label": "crd password far / EN yes go ahead",
     "screen": "chrome-remote-desktop---qvb-flfm-dev1", "ask": "비밀번호 저장 눌러줘",
     "filler": FILLER_FAR, "then": "yes go ahead", "want": None,
     "why": "the mixed far case on the fourth screen; the offered name is Korean and the "
            "acceptance is not."},

    # ---- RESTATE: the person names the offered target instead of saying yes. ----
    #: **Scored by ANEW's rule, on a named target, and separate from ACCEPT on purpose.** Saying
    #: *"닫기 눌러줘"* after being offered 닫기 may work for a reason that has nothing to do with
    #: acceptance: it is a complete instruction on its own. If `RESTATE` passes where `ACCEPT` fails,
    #: the agent is not failing to act — it is failing to read a yes as referring to anything, and
    #: pooling the two into one rate would have hidden that.
    #:
    #: `STUCK` is reachable and is the interesting failure: turn 1 offered something else and turn 2
    #: re-issued that instead of the thing the person just named.
    {"kind": RESTATE, "label": "cm / 글자 복사하고 싶어 -> 복사 눌러줘", "screen": "character-map",
     "ask": "글자 복사하고 싶어", "then": "복사 눌러줘", "want": ["Copy", "button"],
     "why": "a vague want, then the target by name. The ask is deliberately not the restatement."},
    {"kind": RESTATE, "label": "paint / 지우고 싶어 -> 지우개 눌러줘", "screen": "untitled---paint",
     "ask": "지우고 싶어", "then": "지우개 눌러줘", "want": ["Eraser", "button"],
     "why": "the same shape on the 123-widget screen."},
    {"kind": RESTATE, "label": "cm / 이 창 닫고 싶어 -> 닫기 눌러줘", "screen": "character-map",
     "ask": "이 창 닫고 싶어", "then": "닫기 눌러줘", "want": ["Close", "button"],
     "why": "the example in the request that pre-registered this group, word for word."},
    {"kind": RESTATE, "label": "paint / 자르고 싶어 -> 자르기 눌러줘", "screen": "untitled---paint",
     "ask": "자르고 싶어", "then": "자르기 눌러줘", "want": ["Crop", "button"],
     "why": "the fourth, and the Korean name is a verb where the widget's is a noun."},

    #: Two on each new screen, same shape: a vague want, then the offered thing by name.
    {"kind": RESTATE, "label": "si / 하드웨어 쪽 보고 싶어 -> 하드웨어 리소스 눌러줘",
     "screen": "system-information", "ask": "하드웨어 쪽 보고 싶어",
     "then": "하드웨어 리소스 눌러줘", "want": ["Hardware Resources", "tree item"],
     "why": "the first named target on a tree item, and on the longest prompt."},
    {"kind": RESTATE, "label": "si / 시스템 정보 보고 싶어 -> 시스템 요약 눌러줘",
     "screen": "system-information", "ask": "시스템 정보 찾아보고 싶어",
     "then": "시스템 요약 눌러줘", "want": ["System Summary", "tree item"],
     "why": "**three widgets are called 'System Summary'** — tree item, list, header. The pair is "
            "the key and the label is not, which is the Font : defect on a third screen."},
    {"kind": RESTATE, "label": "crd / 이 창 닫고 싶어 -> 닫기 눌러줘",
     "screen": "chrome-remote-desktop---qvb-flfm-dev1", "ask": "이 창 닫고 싶어",
     "then": "닫기 눌러줘", "want": ["닫기", "button"],
     "why": "the Character Map case word for word, on a screen whose Close button is named in "
            "Korean — so the restatement and the widget name are the same string."},
    {"kind": RESTATE, "label": "crd / 확장 프로그램 보고 싶어 -> 확장 프로그램 눌러줘",
     "screen": "chrome-remote-desktop---qvb-flfm-dev1", "ask": "확장 프로그램 보고 싶어",
     "then": "확장 프로그램 눌러줘", "want": ["확장 프로그램", "button"],
     "why": "the fourth screen's second named target."},

    # ---- REFUSE: the person says no. Nothing should be offered. ----
    #: **The control `ACCEPT` needs and did not have.** A fix that made the agent act on anything
    #: following an offer would score 16 of 16 on acceptance and be worse to live with than the
    #: defect. Every line here is an unambiguous *stop*: none of them is *"아니 그거 말고"*, which
    #: asks for an alternative and would make an offer of something else the correct answer.
    {"kind": REFUSE, "label": "cm copy / 아니 됐어", "screen": "character-map",
     "ask": "복사 눌러줘", "then": "아니 됐어", "want": None,
     "why": "the plainest refusal, on the case that accepts correctly. AGAIN here means it "
            "re-offered the very thing that was just turned down."},
    {"kind": REFUSE, "label": "paint eraser / 아니야 하지 마", "screen": "untitled---paint",
     "ask": "지우개 선택해줘", "then": "아니야 하지 마", "want": None,
     "why": "an explicit prohibition rather than a decline."},
    {"kind": REFUSE, "label": "cm font / 됐어 안 해도 돼", "screen": "character-map",
     "ask": "글꼴 바꾸는 거 눌러줘", "then": "됐어 안 해도 돼", "want": None,
     "why": "'it's fine, you don't have to' — polite, and the one most easily read as assent."},
    {"kind": REFUSE, "lang": "en", "label": "paint crop / EN no, never mind", "screen": "untitled---paint",
     "ask": "Press the crop button", "then": "no, never mind", "want": None,
     "why": "English refusal against the English acceptances, so a language effect in one group "
            "can be read against the other."},

    #: **Four to twelve, three on each screen, two Korean and one English.** `REFUSE` is half of
    #: the pair that says whether a rise in `ACCEPT` is comprehension or an agent that acts on
    #: anything following an offer, and at n = 4 it could only ever have caught a total collapse.
    #: The wordings repeat across screens on purpose: the same *no* on four screens is the cleanest
    #: reading of the screen there is. Still nothing here asks for an alternative.
    {"kind": REFUSE, "lang": "en", "label": "cm copy / EN no, don't do that",
     "screen": "character-map", "ask": "Press the copy button", "then": "no, don't do that",
     "want": None, "why": "the English refusal Character Map lacked; an explicit prohibition, so "
                          "the English row carries both a decline and a prohibition."},
    {"kind": REFUSE, "label": "paint undo / 아니 지금은 하지 마", "screen": "untitled---paint",
     "ask": "실행 취소 눌러줘", "then": "아니 지금은 하지 마", "want": None,
     "why": "'not now' — a refusal with a time in it, which is the one most easily read as "
            "'later, then'."},
    {"kind": REFUSE, "label": "si close / 아니 됐어", "screen": "system-information",
     "ask": "이 창 닫아줘", "then": "아니 됐어", "want": None,
     "why": "the plainest refusal on the longest prompt, against the same words on Character Map."},
    {"kind": REFUSE, "label": "si edit menu / 아니야 하지 마", "screen": "system-information",
     "ask": "편집 메뉴 눌러줘", "then": "아니야 하지 마", "want": None,
     "why": "an explicit prohibition at 152 lines."},
    {"kind": REFUSE, "lang": "en", "label": "si find / EN no, never mind",
     "screen": "system-information", "ask": "Press the Find button", "then": "no, never mind",
     "want": None, "why": "the English refusal on the long screen, the same words as Paint's."},
    {"kind": REFUSE, "label": "crd close / 아니 됐어",
     "screen": "chrome-remote-desktop---qvb-flfm-dev1", "ask": "닫기 눌러줘", "then": "아니 됐어",
     "want": None, "why": "the plainest refusal on the shortest prompt — the same two turns as "
                          "`crd close / 응` and the opposite correct answer."},
    {"kind": REFUSE, "label": "crd password / 됐어 안 해도 돼",
     "screen": "chrome-remote-desktop---qvb-flfm-dev1", "ask": "비밀번호 저장 눌러줘",
     "then": "됐어 안 해도 돼", "want": None,
     "why": "'you don't have to' — polite, and about the one offer here worth refusing."},
    {"kind": REFUSE, "lang": "en", "label": "crd close / EN no, never mind",
     "screen": "chrome-remote-desktop---qvb-flfm-dev1", "ask": "Close this window",
     "then": "no, never mind", "want": None,
     "why": "the English refusal on the fourth screen; two wordings over four screens, like the "
            "English acceptances."},

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

    #: Two on each new screen. `ANEW` is the group that broke under `96d8456` — one case, and the
    #: reply was word for word the baseline's with an empty action slot beside it — so it is the
    #: control this enlargement most needs a denominator for. Eight is still small and it is a
    #: control, not an endpoint: `broken <= fixed` is not asked of it, and the split verdict below
    #: is what makes its failures readable.
    {"kind": ANEW, "label": "si find -> edit menu", "screen": "system-information",
     "ask": "찾기 버튼 눌러줘", "then": "그럼 편집 메뉴로 해줘", "want": ["Edit", "menu item"],
     "why": "'then make it the edit menu instead' — the same shape as the case that broke, on the "
            "longest prompt."},
    {"kind": ANEW, "label": "si minimize -> software env", "screen": "system-information",
     "ask": "이 창 최소화해줘", "then": "소프트웨어 환경도 눌러줘",
     "want": ["Software Environment", "tree item"],
     "why": "'also' — the previous offer stands and the new one is additive, on a tree item."},
    {"kind": ANEW, "label": "crd close -> minimize",
     "screen": "chrome-remote-desktop---qvb-flfm-dev1", "ask": "닫기 눌러줘",
     "then": "그럼 최소화로 해줘", "want": ["최소화", "button"],
     "why": "a new target while an offer stands, on the shortest prompt. Close and Minimize sit "
            "next to each other in the list and both are named in Korean."},
    {"kind": ANEW, "label": "crd password -> extensions",
     "screen": "chrome-remote-desktop---qvb-flfm-dev1", "ask": "비밀번호 저장 눌러줘",
     "then": "확장 프로그램도 눌러줘", "want": ["확장 프로그램", "button"],
     "why": "the additive form on the fourth screen."},
]

#: Worst last in each list, which is what `worst_index()` sorts on and therefore which repeat's
#: transcript is printed beside the verdict. `VOID` is worst everywhere: a case that measured
#: nothing must never be displayed as the case that passed.
QUIET, OTHER, AGAIN = "quiet", "other", "again"
ACTED, DRIFTED, IGNORED = "acted", "drifted", "ignored"
#: `SILENT` was split out of `WRONG` on 2026-09-13, **before either arm of the enlarged run**, and
#: both arms are scored under it. The `ANEW` break at `96d8456` was recorded as `wrong`, and what
#: happened was that the reply was word for word the baseline's — `글꼴을 변경하려면 'Font :'
#: 콤보박스를 클릭해 주세요.` — with an empty action slot beside it. *Said the right thing and did
#: nothing* and *did a different thing* have different causes: the first is the say-first field
#: order dropping the second field, the second is a choice. `ACCEPT` has had that distinction since
#: it was written (`IGNORED` against `DRIFTED`) and `MEMORY` has it (`QUIET` against `OTHER`); this
#: is the same split arriving in the third vocabulary. Neither is correct and the pass rate does not
#: move — only the name of the failure does.
LANDED, EXTRA, WRONG, STUCK, SILENT = "landed", "extra", "wrong", "stuck", "silent"
VOID = "void"
#: Three vocabularies over six groups, and **which group uses which is the pre-registration**:
#: `REFUSE` is scored by `MEMORY`'s rule (nothing should happen), `FARACCEPT` by `ACCEPT`'s (the
#: offer again), `RESTATE` by `ANEW`'s (a named target). A group with its own private vocabulary
#: would be a group whose failures cannot be compared with anything.
WORST = {MEMORY: [QUIET, OTHER, AGAIN, VOID],
         REFUSE: [QUIET, OTHER, AGAIN, VOID],
         ACCEPT: [ACTED, DRIFTED, IGNORED, VOID],
         FARACCEPT: [ACTED, DRIFTED, IGNORED, VOID],
         RESTATE: [LANDED, EXTRA, SILENT, WRONG, STUCK, VOID],
         ANEW: [LANDED, EXTRA, SILENT, WRONG, STUCK, VOID]}

#: The one verdict per group that counts as correct. Everything else is a failure of some named
#: kind, and the kinds are printed separately so that "it did the wrong thing" and "it did nothing"
#: never merge into one rate.
GOOD = {MEMORY: QUIET, REFUSE: QUIET, ACCEPT: ACTED, FARACCEPT: ACTED,
        RESTATE: LANDED, ANEW: LANDED}
WHY_VERDICT = {
    QUIET: "no effect at all — correct: it was asked a question, or told to stop",
    OTHER: "an effect, but not one it had offered",
    AGAIN: "it offered the same thing again — THE DEFECT UNDER TEST (memory), or it re-offered "
           "what was just refused",
    ACTED: "the accepted offer was re-issued — correct",
    DRIFTED: "an effect, but a different one from the accepted offer",
    IGNORED: "the person said yes and nothing was offered — THE DEFECT THE ACCEPT SET IS SIZED FOR",
    LANDED: "exactly the named target, and nothing else",
    EXTRA: "the named target, plus at least one more effect",
    SILENT: "the named target absent and NOTHING was offered at all — it said something and did "
            "nothing. Split out of 'wrong' on 2026-09-13; both arms are scored under the split",
    WRONG: "the named target absent and a DIFFERENT effect offered instead",
    STUCK: "the named target absent and the OLD offer re-issued instead",
    VOID: "measured nothing: turn 1 offered nothing, or turn 1's offer never reached turn 2",
}

#: The screens each group runs on, asserted at start-up rather than trusted. A case list edited to
#: sit on one screen would make every rate here a property of that window, and the docstring's claim
#: that all four are covered would go stale silently. **Two until 2026-09-13**, when the other two
#: captured screens were brought in: at two, "123 widget lines" and "Paint" were the same variable.
WANT_SCREENS = 4

#: The four captured screens, **ordered by the number of widget lines the prompt carries**, which
#: is the axis this enlargement exists to put four points on instead of two. The counts are the
#: fixtures' own, asserted in `main()` rather than trusted: a re-capture that changed a window's
#: size would otherwise leave every label here carrying its old number, which is exactly how a
#: display comes to disagree with the run beside it.
SCREEN_LINES = (("chrome-remote-desktop---qvb-flfm-dev1", "crd", 12),
                ("character-map", "cm", 21),
                ("untitled---paint", "paint", 123),
                ("system-information", "si", 152))

#: Every group must sit on every screen, and this is checked rather than described. Balance is the
#: whole point of the enlargement — a group that quietly lost a screen would put the screen back
#: into every rate as a confound, which is the state this file was in until 2026-09-13.
WANT_EVERY_GROUP_ON_EVERY_SCREEN = True


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

    This is the evidence that the two arms differ **in the transcript**. If both print the same
    shape, the `offered:` lines got no treatment; see `slot_lines()` for the rest of the prompt,
    because a change can be delivered somewhere the transcript never sees.
    """
    return QUOTED.sub("<x>", line).strip()


#: The two literals that bound the region between the widget list and the person's words, and the
#: shape of a widget line. Taken from `converse()`'s own assembly and **not imported from it**: the
#: without-fix arm is a `converse.py` that has no slot at all, so an import would make the check
#: impossible to run in the one arm that has to return nothing.
ON_SCREEN = "On screen now:"
UTTERANCE = ("\nThey said: ", "\nYou noticed: ")
WIDGET_LINE = re.compile(r"^\d+: ")
#: The line `describe()` renders when a surface exposes nothing. Not a slot line; no fixture here
#: produces it, and a check that silently counted it would report a treatment that is a placeholder.
NO_WIDGETS = "(nothing named"


def slot_lines(prompt: str) -> list:
    """**Whatever the prompt puts between the widget list and the person's words**, verbatim.

    Nothing, in the arm that predates `96d8456`. The standing-offer block, in the arm that carries
    it. This is the `offer_reached()` check applied to the *other* channel: the treatment under test
    is a prompt region, and a comparison that cannot show the treatment was delivered is a
    comparison that can score a change which never reached the model.

    **Generic on purpose — it does not know what the fix says.** It reports whatever is there, so a
    later change that renders something else in the same place is measured by the same code and a
    change that renders nothing reads as nothing rather than as absence of evidence.

    Raises rather than returning `[]` when the prompt cannot be bounded, because "the slot was
    empty" and "this harness can no longer find the region" must not print the same thing.
    """
    if ON_SCREEN not in prompt:
        raise RuntimeError("no 'On screen now:' in the prompt — converse.py's format has moved "
                           "and this harness is reading nothing")
    tail = prompt.split(ON_SCREEN, 1)[1]
    cut = [tail.split(mark, 1)[0] for mark in UTTERANCE if mark in tail]
    if not cut:
        raise RuntimeError("no 'They said:' or 'You noticed:' after the widget list — the region "
                           "between them cannot be bounded")
    return [x.rstrip() for x in min(cut, key=len).splitlines()
            if x.strip() and not WIDGET_LINE.match(x) and NO_WIDGETS not in x]


#: What `slot_shapes` counts a turn with an empty region under. **A named key, not a missing one**:
#: an arm that renders nothing has to be visible in the tally with its own count beside it, or the
#: two arms are read as one recorded and one not.
NO_SLOT = "(nothing between the widget list and the turn)"


def slot_shape(lines) -> str:
    """One turn's region as a single key: the wording, with the widget names taken out."""
    return " | ".join(shape_of(x) for x in lines) if lines else NO_SLOT


def offer_reached(first, last) -> bool:
    """Did **turn 1's own offer** survive into the prompt the scored turn was given?

    Stricter than *"there is an `offered:` line"*, and the twelve `FARACCEPT` cases are why. Four
    filler turns sit between the offer and the yes, and any one of them can produce an offer of its
    own — so a block containing *some* offered line says nothing about whether the treatment this
    case is about was delivered. The names are matched against the rendered lines, which is what the
    model saw.

    At distance 1 this is the same test as the old one and cannot fail on its own; at distance 4 it
    is the difference between a failure and a case that was never asked.
    """
    names = [e["target"] or e["text"] or "" for e in first["effects"]]
    lines = offered_lines(last["window"])
    return bool(lines) and any(n and any(n in line for line in lines) for n in names)


def verdict_of(kind, first, second, want) -> str:
    """One repeat of one case, from the two scored turns' effects. Never a bare boolean.

    `first` is the offer turn and `second` is the **last** turn — one later for every group but
    `FARACCEPT`, five later for that one. Six paths in and every one returns a named state,
    including the two ways of measuring nothing, which is the distinction this file rests on.
    """
    offered = [key_of(e) for e in first["effects"]]
    again = [key_of(e) for e in second["effects"]]
    #: (1) nothing was offered, so there is no offer to accept, refuse or remember, and (2) that
    #: offer never reached the scored turn's prompt. Either way this case measured nothing.
    if not offered or not offer_reached(first, second):
        return VOID
    repeated = any(k in offered for k in again)
    if kind in (MEMORY, REFUSE):
        return QUIET if not again else (AGAIN if repeated else OTHER)
    if kind in (ACCEPT, FARACCEPT):
        return ACTED if repeated else (IGNORED if not again else DRIFTED)
    #: `ANEW` and `RESTATE`: a target was named, so it is checked by name **and role** — the
    #: `Font :` label sits in front of the `Font :` combo box and clicking the label does nothing.
    pair = tuple(want)
    clicks = [(e["target"], e["role"]) for e in second["effects"] if e["kind"] == "click"]
    if pair in clicks:
        return LANDED if len(second["effects"]) == 1 else EXTRA
    if repeated:
        return STUCK
    #: **Three ways of missing the named target, not two.** `SILENT` is *nothing at all was
    #: offered* — the reply may be word-perfect and the action slot empty, which is what the one
    #: `ANEW` break at `96d8456` actually was. `WRONG` is *something else was offered*. They have
    #: different causes and the old single name covered both.
    return SILENT if not again else WRONG


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
    return sending.harness_print(__file__)


def norm_print(path) -> str:
    """Six characters over a source file with CRLF normalised to LF. **Not `file_print()`.**

    One definition, in `measure/sending.py`, since `conversation.py` grew an `l` of its own — two
    files hashing two slightly different things under one letter is the drift this directory keeps
    being bitten by. Kept as a name here because `tree_prints()` and the arm check below read like
    prose with it and because it is what the JSON key is called.
    """
    return sending.norm_print(path)


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
    print("    Distance is 1 — the offered line is the last line of the block — in every case "
          "but the")
    print("    twelve FARACCEPT ones, where it is five turns back with four ordinary turns "
          "after it.")
    return {"channels": list(CHANNELS), "channels_fed": list(FED),
            "channels_never_fed": unfed, "distances": [1, 5], "printed": True}


def filler_check(screens) -> list:
    """Does any `FILLER_FAR` line name a widget? Returns the offenders, empty for a clean set.

    **The confound the far cases exist to avoid, checked rather than asserted in prose.** A filler
    line that says *"Eraser"* puts the offer's name back into the window and turns a distance-4
    probe into a distance-1 one — a harness that cannot fail. `measure/conversation.py` refuses to
    run on the same finding; this returns the list so `main()` can refuse in the same way.

    Names of one or two characters are skipped: `A`, `OK` and `Up` match half of any language, and
    a check that fires on everything is a check nobody can keep.
    """
    names = {w.name for widgets in screens.values() for w in widgets
             if w.name and len(w.name) > 2}
    return [(line, n) for line in FILLER_FAR for n in sorted(names)
            if n.lower() in line.lower()]


#: The pair every projection in `planned_sensitivity()` is computed from: the two committed arms at
#: the 44-case set. **Named, not globbed** — a glob would silently pick up a later file and change a
#: printed projection without anybody editing anything.
BASELINE_WITHOUT = "reoffer-qwen3-8b-off-r9352d2-s5020f7-fa5771b-h57e605-lbc9cc0-x3.json"
BASELINE_WITH = "reoffer-qwen3-8b-off-r322a4c-s5020f7-fa5771b-h57e605-lbc9cc0-x3.json"

#: Six discordant pairs one way is `2 x 0.5^6 = 0.031`; five is `0.0625`. This is the whole reason
#: the case list was enlarged, and it is a property of the exact test, not of the effect size.
K_FOR_SIGNIFICANCE = 6


def measured_rates(path):
    """`{group: (live, failing)}` off a stored result file, or `None` if it is not there.

    Reads `rows`, not `totals`: `totals` counts `ok` over every case including the void ones, and a
    projection built on that would be a rate whose denominator has cases in it that measured
    nothing.
    """
    if not path.is_file():
        return None
    run = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for kind in GROUPS:
        live = [r for r in run["rows"] if r["kind"] == kind and not r["void"]]
        out[kind] = (len(live), sum(1 for r in live if not r["ok"]))
    return out


def planned_sensitivity(cases) -> dict:
    """**What the planned set could show, printed before a single call is made.**

    The number that decides whether a run is worth its calls, and this directory has published a
    Fisher `p = 0.070` on `n = 28` after working out too late that the design could not reach 0.05.
    `k` discordant pairs give `2 x 0.5^k`, so a group with fewer than six **failures in the baseline
    arm** cannot show a repair as significant however perfect the repair is — the binding constraint
    is baseline failures, never cases.

    Three quantities, and the labels are the point:

    * **measured** — the live count and the failure count of each group in the two committed arms at
      the 44-case set (`f a5771b`).
    * **estimated** — the planned group size times that measured failure rate. It assumes the two
      screens that carried no acceptance case behave like the two that did, **which is exactly what
      nobody knows**, and is the reason four screens are here. It can be wrong in either direction.
    * **computed** — `2 x 0.5^k` at the estimated numbers, and how many repairs reach `k = 6`.

    Returns the projection so the JSON carries it: a sensitivity nobody can find afterwards is a
    sensitivity that gets recomputed to taste.
    """
    without = measured_rates(HERE / BASELINE_WITHOUT)
    with_fix = measured_rates(HERE / BASELINE_WITH)
    planned = {k: sum(1 for c in cases if c["kind"] == k) for k in GROUPS}
    print("\n  WHAT THE PLANNED SET COULD SHOW — before any call, from the committed 44-case pair")
    if without is None or with_fix is None:
        print(f"    ** {BASELINE_WITHOUT} or {BASELINE_WITH} is not in {HERE}, so no projection "
              f"can be made. **")
        print("    The run is not refused for it — but then nobody has said in advance what it "
              "could show, which is")
        print("    the thing this print exists to stop.")
        return {"available": False, "planned": planned}
    print("    measured: the WITHOUT arm's live cases and failures, and how many of those the WITH "
          "arm repaired.")
    print("    estimated: that failure rate over the planned n.  computed: 2 x 0.5^k, the exact "
          "paired floor.")
    out = {"available": True, "planned": planned, "baseline_without": BASELINE_WITHOUT,
           "baseline_with": BASELINE_WITH, "k_for_significance": K_FOR_SIGNIFICANCE, "groups": {}}
    for kind in GROUPS:
        live, bad = without[kind]
        _live_new, bad_new = with_fix[kind]
        n = planned[kind]
        if not live or not n:
            continue
        rate = bad / live
        #: Estimated, and rounded **down**: a projection that rounds up in its own favour is the
        #: kind of correction this project is required to scrutinise harder than one that does not.
        expect = int(n * rate)
        pv, _says = sensitivity(expect)
        repaired = max(0, bad - bad_new)
        share = repaired / bad if bad else 0.0
        expect_fixed = int(expect * share)
        star = "   <- PRIMARY ENDPOINT" if kind == ACCEPT else ""
        print(f"    {kind:<10} n {n:>3} planned | measured {bad} of {live} failing "
              f"({rate:.0%}) | estimated {expect} failures{star}")
        if expect:
            print(f"               a perfect repair of {expect} gives p = {pv:.4f}; k >= "
                  f"{K_FOR_SIGNIFICANCE} needs {K_FOR_SIGNIFICANCE} of them repaired, "
                  f"{K_FOR_SIGNIFICANCE / expect:.0%} of the estimate")
            print(f"               at the repair share already observed ({repaired} of {bad} = "
                  f"{share:.0%}) that is an estimated k = {expect_fixed}, p = "
                  f"{sign_test(expect_fixed, 0):.4f}")
        else:
            print("               nothing is estimated to fail, so there is nothing here for a "
                  "repair to move")
        out["groups"][kind] = {
            "planned_n": n, "measured_live": live, "measured_failing": bad,
            "measured_failure_rate": round(rate, 4), "estimated_failures": expect,
            "p_if_all_repaired": round(pv, 6), "observed_repair_share": round(share, 4),
            "estimated_k_at_observed_share": expect_fixed,
            "p_at_observed_share": round(sign_test(expect_fixed, 0), 6)}
    print("    ** The estimate assumes system-information and chrome-remote-desktop fail like the "
          "two screens that")
    print("       carried every case before today. Nobody has measured that — it is the reason "
          "they are here — and it")
    print("       can be wrong in either direction. These are what the design COULD show, not a "
          "prediction. **")
    return out


def plan_note() -> dict:
    """The pre-registered plans, printed at the top of every run and stored in the JSON.

    **Two of them, and the first is kept verbatim.** The first was written before any attempt at the
    `ACCEPT` defect existed and is what `96d8456` was scored against; the second enlarges the case
    set so that the same question can be answered at all, and is honest in its own text about what
    that costs. Printed rather than left in the docstring because the thing that gets copied out of
    a run is the log, and a criterion nobody read is a criterion that gets chosen afterwards.
    """
    first = {
        "written": "2026-09-13, before any fix to ACCEPT existed",
        "primary": "ACCEPT, 16 cases, the yes one turn after the offer, worst of 3, "
                   "exact two-sided paired sign test",
        "significance": "k = 6 discordant pairs one way gives p = 0.031; k = 5 gives 0.062",
        "controls": "MEMORY (12) and REFUSE (4) must not lose more than they gain",
        "secondary": "FARACCEPT (4, the yes five turns after the offer) and RESTATE (4), "
                     "reported separately, in no criterion",
        "refuted_by": "ACCEPT at 15 or 16 of 16 on this baseline — then the 2 of 4 was noise",
        "outcome": "NOT MET at 96d8456. ACCEPT 8 of 15 -> 12 of 15, k = 4 all one way, "
                   "p = 0.1250; the change is committed on trial and is not called a fix.",
    }
    second = {
        "written": "2026-09-13, AFTER 96d8456 exists and after its result is known",
        "what_this_is": "an enlargement of the case set, because the first plan could not answer "
                        "its own question: 16 ACCEPT cases gave 7 baseline failures, and k >= 6 "
                        "was barely reachable even by a perfect repair. The endpoint, the test "
                        "and the controls are unchanged.",
        "honest_about_itself": "SCORING A FIX THAT ALREADY EXISTS AGAINST A CASE SET CHOSEN NOW IS "
                               "NOT A PRE-REGISTRATION IN THE STRONG SENSE. Fixed in advance: the "
                               "criterion, the test, the controls, and all 104 cases, written and "
                               "hashed into f before either arm ran. NOT fixed in advance: the "
                               "set was designed by somebody who had seen which four cases the "
                               "fix repaired. Every one of the original 44 is kept verbatim, "
                               "including the three ACCEPT failures it did not repair, so the "
                               "enlargement cannot be read as having replaced the cases that did "
                               "not go its way.",
        "standing": "THIS IS ONE ENLARGEMENT. THE RESULT STANDS WHATEVER IT SAYS, AND THERE IS NO "
                    "THIRD SET.",
        "primary": "ACCEPT, 40 cases over four screens, the yes one turn after the offer, worst "
                   "of 3 repeats, exact two-sided paired sign test over the discordant pairs",
        "significance": "k = 6 discordant pairs one way gives p = 0.031; k = 5 gives 0.0625 and "
                        "does not. Unchanged from the first plan.",
        "controls": "MEMORY (24) and REFUSE (12): broken <= fixed in each, or the agent has become "
                    "one that acts on anything following an offer — which scores well here and is "
                    "worse to live with",
        "criterion": "MET only if ACCEPT p < 0.05 AND fixed > broken AND both controls hold. "
                     "Nothing else counts, and three of four is not met.",
        "secondary": "FARACCEPT (12) and RESTATE (8), reported separately and pooled into nothing",
        "not_pooled": "ACCEPT + FARACCEPT pooled reaches 0.05 far more easily and IS NOT the "
                      "endpoint. 96d8456 recorded that pooling at p = 0.0156 and refused it; it "
                      "is refused here again, in advance and in writing.",
        "anew_is_not_in_it": "ANEW (8) broke on one case at 96d8456 and is printed beside the "
                             "criterion every run. It is NOT in the criterion, because it was not "
                             "in the first one and adding it now would be choosing a criterion "
                             "after seeing the data — in the direction that fails the fix, but "
                             "chosen the same way. Read it; it does not score.",
        "refuted_by": "ACCEPT with k small, or with fixed no greater than broken. Then the "
                      "standing-offer block is not the mechanism, and the next suspect is "
                      "SYSTEM's two-way partition of the turn — asked to do something, or just "
                      "talking — which a bare 'ㅇㅇ' is neither of. That belongs to conversation.",
        "also_refuted": "ACCEPT at 38 of 40 or better in the WITHOUT arm. Then there is no defect "
                        "at this size to repair, and 8 of 15 was a property of the 16 cases.",
        "scoring_change": "the 'wrong' verdict is split into 'silent' (nothing offered at all) and "
                          "'wrong' (a different effect offered) for RESTATE and ANEW. Both arms "
                          "are run and scored under the split, so it is consistent; no pass rate "
                          "moves, only the name of a failure.",
    }
    print("\n  PRE-REGISTERED (1 of 2) — the plan 96d8456 was scored against, kept verbatim")
    for k, v in first.items():
        print(f"    {k:<20} {v}")
    print("\n  PRE-REGISTERED (2 of 2) — this run's plan, and what it does not get to claim")
    for k, v in second.items():
        print(f"    {k:<20} {v}")
    return {"first": first, "second": second}


# ---- the run -------------------------------------------------------------------------------------

def as_spend(t) -> dict:
    """`measure.choosing.spend()`'s five keys, off a `drive()` turn record."""
    return {k: t[k] for k in ("seconds", "out_tokens", "reason", "truncated", "thought_chars")}


class Tapped(Recorder):
    """`Recorder`, keeping **every** prompt rather than only the last one.

    `Recorder.ask` is called exactly once per turn — its retry loop lives inside that single call —
    so the list is one prompt per turn, and `run_cases()` refuses rather than lining them up by
    guess if it ever is not. The prompts are read and dropped per case; none is stored, because 768
    prompts carrying 152 widget lines each is a result file nobody can open.
    """

    def __init__(self, inner, tries: int = 4) -> None:
        super().__init__(inner, tries)
        self.prompts = []

    def ask(self, prompt, images=None):
        self.prompts.append(prompt)
        return super().ask(prompt, images)


def run_cases(model, screens, cases, repeat, spends, shapes, slots):
    """Every case, `repeat` times, worst counted. Returns the rows.

    A case is `[ask] + filler + [then]`, and `filler` is empty for every group but `FARACCEPT`. The
    scored pair is **the first turn and the last**; the turns in between are recorded, counted and
    printed, never scored — an effect proposed during small talk is `measure/talking.py`'s
    measurement and it is reported here because it is free and because a `DRIFTED` verdict at
    distance 4 reads differently when the drift is to something the agent offered two turns ago.
    """
    rows = []
    for case in cases:
        kind, draws = case["kind"], []
        filler = list(case.get("filler", ()))
        for r in range(repeat):
            #: **Cleared per case, checked per case.** The two warm-up calls and any earlier case's
            #: prompts are not this case's, and a list that quietly ran long would line the wrong
            #: prompt up against the wrong turn — which is the display defect this directory has
            #: shipped twice.
            del model.prompts[:]
            turns = drive(model, screens[case["screen"]], [case["ask"], *filler, case["then"]],
                          REMEMBER_TURNS)
            if len(model.prompts) != len(turns):
                raise RuntimeError(f"{len(model.prompts)} prompts for {len(turns)} turns on "
                                   f"{case['label']}: they cannot be paired, so the slot check "
                                   f"would be reading somebody else's prompt")
            first, second = turns[0], turns[-1]
            for i, t in enumerate(turns):
                spends.append((f"{case['label']} / r{r + 1} / turn {i + 1}", as_spend(t)))
            seen = offered_lines(second["window"])
            for line in seen:
                shapes[shape_of(line)] += 1
            #: The other channel: what the prompt put between the widget list and the person's
            #: words, on **every** turn, counted the same way the offered lines are.
            slot = [slot_lines(x) for x in model.prompts]
            for one in slot:
                slots[slot_shape(one)] += 1
            middle = turns[1:-1]
            draws.append({
                "verdict": verdict_of(kind, first, second, case["want"]),
                "offered": [key_of(e) for e in first["effects"]],
                "again": [key_of(e) for e in second["effects"]],
                #: The two presences, apart. `offer_reached` is the treatment; the count of lines
                #: is how much else was in the block beside it.
                "offer_reached": offer_reached(first, second),
                "offered_lines": seen, "narrated": narrated(first, second),
                #: **The treatment, on the scored turn, verbatim** — and the count over the whole
                #: conversation, so a slot that appeared on the filler turns and not on the scored
                #: one is not read as a slot that was there.
                "slot": slot[-1], "slot_turns": sum(1 for one in slot if one),
                #: Never a bare boolean: what the filler turns did, so a drift at distance 4 can be
                #: read against what else was on offer by then.
                "filler_effects": sum(len(t["effects"]) for t in middle),
                "filler_offers": [key_of(e) for t in middle for e in t["effects"]],
                "turns": turns,
            })
        verdicts = [d["verdict"] for d in draws]
        i = worst_index(verdicts, WORST[kind])
        rows.append({"kind": kind, "case": case["label"], "screen": case["screen"],
                     #: **Turns from the offer to the scored turn, 1 = adjacent.** Four filler
                     #: turns therefore read as 5, not 4: the yes is the sixth turn of the
                     #: conversation and the offer was the first. `measure/conversation.py` counts
                     #: filler lines instead, so its "distance 4" and this file's are two things.
                     "form": case.get("form"), "distance": 1 + len(filler),
                     #: Default "ko": the set is Korean unless a case says otherwise, and the
                     #: default is in one place rather than in a hundred case dicts.
                     "lang": case.get("lang", "ko"),
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
    #: **The breakdown by wording, with its own n beside it.** Four of sixteen is not a rate and
    #: this directory has published one; the split is printed so a reader can see that a bare *"응"*
    #: and a formal *"네 해주세요"* are four cases each, not a trend.
    #: **The screen breakdown, with its own n beside every rate.** The screen was a perfect
    #: confound in this file until 2026-09-13 — two screens, one of them carrying every long
    #: prompt — and it is printed first because it is the factor the enlargement was for. Ordered
    #: short prompt to long, so a distance effect reads down the column.
    for _screen, tag, lines in SCREEN_LINES:
        at = [r for r in mine if r["screen"] == _screen and not r["void"]]
        voids = sum(1 for r in mine if r["screen"] == _screen and r["void"])
        if at or voids:
            print(f"      of which {tag:<5} ({lines:>3} widget lines) "
                  f"{sum(1 for r in at if r['ok'])} of {len(at)}"
                  + (f", {voids} void" if voids else ""))
    forms = sorted({r["form"] for r in mine if r["form"]})
    for form in forms:
        at = [r for r in mine if r["form"] == form and not r["void"]]
        if at:
            print(f"      of which {form:<7} {sum(1 for r in at if r['ok'])} of {len(at)}")
    #: **Read off the case, not guessed from its label.** A substring test on the label is how a
    #: display comes to disagree with the thing beside it; `lang` is written in the case list before
    #: anything runs, and `mixed` is a third value because one case deliberately is.
    for lang in ("ko", "en", "mixed"):
        live_at = [r for r in mine if r["lang"] == lang and not r["void"]]
        if live_at:
            print(f"      of which {lang:<7} {sum(1 for r in live_at if r['ok'])} of "
                  f"{len(live_at)}")
    dis = [r for r in mine if not r["agreed"]]
    print(f"    disagreed across {repeat} repeats: {len(dis)} of {len(mine)}"
          f"{' — a single run is not this number' if dis else ''}")
    for r in mine:
        i = r["from_repeat"]
        d = r["repeats"][i]
        mark = "ok  " if r["ok"] else ("VOID" if r["void"] else "NO  ")
        note = "" if r["agreed"] else "  repeats DISAGREED: " + "/".join(r["verdicts"])
        far = ("" if r["distance"] == 1
               else f" [{r['distance']} turns after the offer, {r['distance'] - 1} in between]")
        print(f"    {mark} {r['counted']:<7} {r['case'][:38]:<38} [worst of {repeat} = r{i + 1}]"
              f"{far}{note}")
        for n, t in enumerate(d["turns"]):
            got = "".join(f"  [{e['kind']} {e['target'] or e['text']} ({e['role']})]"
                          for e in t["effects"])
            #: The filler turns are printed on one line each and marked, because a transcript that
            #: shows six turns without saying which two were scored is a display that invites the
            #: wrong two to be read.
            lead = "     " if n in (0, len(d["turns"]) - 1) else "  fil"
            print(f"        {lead} you: {t['said']}")
            print(f"        {lead} it : {(t['reply'] or '')[:72]!r}{got}")
        if d["filler_effects"]:
            print(f"           {d['filler_effects']} effect(s) proposed during the filler turns: "
                  f"{d['filler_offers']}")
        #: **Which of the two void reasons it was**, not that it was void. They have different
        #: fixes: one is a fixture that never set the question up, the other is a window that
        #: dropped the treatment, and a single message for both is a function with two failure
        #: paths returning the same thing.
        if not d["offer_reached"]:
            why = ("turn 1 offered nothing, so there was nothing to accept, refuse or remember"
                   if not d["offered"] else
                   "turn 1's offer never reached the last turn's prompt")
            print(f"           ** {why} — VOID, not a failure **")
        for line in d["offered_lines"]:
            print(f"           the line it was shown: '  offered: {line}'")
        #: **The other half of the treatment, beside the verdict it may have caused.** Printed for
        #: every case in both arms, including when it is empty: an arm that renders nothing has to
        #: say so on its own transcript, or the two arms read as one measured and one not.
        for line in d["slot"]:
            print(f"           and beside the turn itself: '{line}'")
        if not d["slot"]:
            print("           and beside the turn itself: nothing — this arm renders no such "
                  "region")
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


#: **The conditions a comparison holds fixed, and the four of them a caller may declare as the
#: thing being changed.** Anything in `held` differing refuses the comparison; anything in
#: `DECLARABLE` differing refuses it *unless* `--changed` names it, and refuses it the other way
#: round — declared and unmoved — because a declaration that is not true of the files is a label,
#: not a design.
#:
#: **Only `argmind/` conditions are declarable.** `data_print`, `harness_print`,
#: `conversation_print`, `standins_print`, `choosing_print`, `model`, `think`, `repeat` are the
#: instrument and the protocol: a comparison across a changed instrument is two experiments however
#: loudly it is declared, and there is no flag for it here.
#:
#: `prompt` is on the list because the fix this file was enlarged to score will most likely be an
#: edit to `converse.py`, which moves `q` — and without a declaration `--compare` would refuse
#: every such attempt and the pre-registration would score nothing.
DECLARABLE = ("prompt", "loop_print", "remember_turns", "max_listed", "most_at_once")


def compare(old_path, new_path, changed=()) -> int:
    """The paired comparison between two result files. No model, no fixtures, no re-scoring.

    **Refuses rather than reports** when the two runs differ in anything except the declared
    independent variable: the whole trap this experiment was built to avoid is scoring a change
    against a baseline taken at a different case list, and a comparison that prints a p-value anyway
    is worse than no comparison.

    `changed` is what the caller declares it changed — a subset of `DECLARABLE`. Declaring moves a
    condition out of the refusal and into a starred line of its own; it never removes it from the
    output, and declaring something that did **not** move refuses just as hard, because that is a
    comparison whose label is false.
    """
    old = json.loads(pathlib.Path(old_path).read_text(encoding="utf-8"))
    new = json.loads(pathlib.Path(new_path).read_text(encoding="utf-8"))
    print(f"\n  OLD  {pathlib.Path(old_path).name}\n  NEW  {pathlib.Path(new_path).name}\n")

    held = ["model", "think", "repeat", "prompt", "screens_print", "data_print",
            "harness_print", "conversation_print", "standins_print", "choosing_print",
            "llm_print", "only", "max_listed", "most_at_once", "remember_turns", "max_tokens"]
    declared = list(changed)
    bad_declaration = [k for k in declared if k not in DECLARABLE]
    for k in bad_declaration:
        print(f"    ** --changed {k}: not declarable. Only {', '.join(DECLARABLE)} are, and the "
              f"rest are the instrument, not the experiment **")
    #: Declared **and it did not move**: the two files are the same experiment under a label that
    #: says they are not. Refused, because that reading is worse than no reading.
    idle = [k for k in declared if k in DECLARABLE and old.get(k) == new.get(k)]
    for k in idle:
        print(f"    ** --changed {k}: declared as the independent variable and it is identical in "
              f"both arms ({old.get(k)}) — nothing was changed **")
    moved = [k for k in held if old.get(k) != new.get(k) and k not in declared]
    for k in held:
        if old.get(k) == new.get(k):
            continue
        if k in declared and k in DECLARABLE:
            print(f"    ** {k}: {old.get(k)} -> {new.get(k)} — DECLARED as the independent "
                  f"variable of this comparison **")
        else:
            print(f"    ** {k}: {old.get(k)} -> {new.get(k)} — these are two experiments **")
    tree_moved = sorted(k for k in set(old["tree"]) | set(new["tree"])
                        if old["tree"].get(k) != new["tree"].get(k))
    print(f"    argmind/ files differing: {', '.join(tree_moved) or 'NONE — same tree both arms'}")
    #: **One file, whichever file.** This used to name `core/loop.py`, which was the independent
    #: variable of the run this harness was written for; the set is now scored against whatever
    #: change is being attempted, so the rule is *exactly one file moved* and the star says which.
    #: Zero moved is the loudest case of all: two arms of the same tree.
    if len(tree_moved) == 1:
        print(f"    the independent variable in argmind/ is {tree_moved[0]} alone")
    else:
        print(f"    ** {len(tree_moved)} argmind/ files differ, not one. Whatever moves below is "
              f"not attributable to any single change. **")
    print(f"    loop.py: {old.get('loop_print')} -> {new.get('loop_print')}")
    for tag, run in (("OLD", old), ("NEW", new)):
        print(f"    {tag} offered-line shapes seen: "
              + " | ".join(f"{s} (x{n})" for s, n in run["offered_shapes"].items()))
    #: **The second channel, printed beside the first.** The `offered:` wording lives in the
    #: transcript; the region between the widget list and the person's words is where `96d8456` put
    #: its change, and a comparison that could only see the first would call that change untreated.
    #: `None` for a file written before 2026-09-13 — **not zero**, because "this arm rendered
    #: nothing there" and "this arm did not look" are different facts about a run.
    for tag, run in (("OLD", old), ("NEW", new)):
        shapes = run.get("slot_shapes")
        if shapes is None:
            print(f"    {tag} region beside the turn: not recorded (this file predates the check)")
        else:
            total = sum(shapes.values())
            carried = sum(n for s, n in shapes.items() if s != NO_SLOT)
            print(f"    {tag} region beside the turn: {carried} of {total} turns carried one — "
                  + " | ".join(f"{s} (x{n})" for s, n in shapes.items()))
    #: **Any wording present in both arms**, not only the case where the two sets are equal. A
    #: change to one channel or one effect kind leaves most shapes untouched, and an equality test
    #: would pass every one of those silently while the cases carrying the shared wording got no
    #: treatment *in that channel*. The intersection is the number of ways this comparison is not
    #: one — in the transcript. It is not, on its own, evidence that nothing was applied: that
    #: sentence is only true when the other channel and the prompt fingerprint agree with it, and
    #: this used to print it regardless, beside a run whose treatment was real.
    shared = sorted(set(old["offered_shapes"]) & set(new["offered_shapes"]))
    slots_known = old.get("slot_shapes") is not None and new.get("slot_shapes") is not None
    slots_moved = slots_known and old["slot_shapes"] != new["slot_shapes"]
    prompt_moved = old.get("prompt") != new.get("prompt")
    if shared:
        for s in shared:
            print(f"    ** SAME OFFERED WORDING IN BOTH ARMS: '  offered: {s}' — whatever else "
                  f"differs, THIS wording did not **")
        if old["offered_shapes"].keys() == new["offered_shapes"].keys():
            if slots_moved:
                print("    -> every offered wording is shared, and the region beside the turn is "
                      "NOT: the treatment in")
                print("       this comparison is that region, and no case's transcript wording "
                      "was touched.")
            elif prompt_moved and not slots_known:
                print("    ** every offered wording is shared and one arm did not record the "
                      "other region, so what")
                print("       differs cannot be named here. The prompt fingerprint moved; that is "
                      "the only evidence of a")
                print("       treatment this comparison has. **")
            elif not prompt_moved:
                print("    ** every wording is shared, the region beside the turn is identical "
                      "and the prompt fingerprint")
                print("       did not move: no treatment was applied at all and nothing below is "
                      "a test of anything. **")
    if moved or bad_declaration or idle:
        print("\n  refusing to compare: see the starred lines above.")
        return 1
    if declared:
        print(f"\n  DECLARED INDEPENDENT VARIABLE: {', '.join(declared)}")
        print("    Everything else this file records is identical in both arms; nothing "
              "below is attributable to anything else.")

    by_case = {(r["kind"], r["case"]): r for r in old["rows"]}
    print(f"\n  PAIRED, case by case, worst of {new['repeat']} in each arm")
    total = {}
    for kind in GROUPS:
        rows = [r for r in new["rows"] if r["kind"] == kind]
        #: **A group neither arm has is not a group that agreed.** The pre-2026-09-13 result files
        #: carry three kinds, and printing `0 fixed, 0 broken, p = 1.0` for the other three reads
        #: as three controls holding when in fact three controls were never run.
        if not rows and not any(r["kind"] == kind for r in old["rows"]):
            print(f"    -> {kind}: in neither arm — this pair predates the group. Not scored, not "
                  f"a control.")
            total[kind] = {"fixed": 0, "broken": 0, "same_ok": 0, "same_bad": 0, "void": 0,
                           "n": 0, "p": 1.0}
            continue
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
    #: **The pre-registered reading, printed rather than left to the reader**, and printed from
    #: the same numbers the table above was printed from. A verdict beside a number taken from
    #: somewhere else is a defect this directory has shipped twice.
    #: `--only` can leave a group out, and a criterion computed over a missing control is
    #: a criterion that cannot fail. Say so and stop rather than print three of five.
    if not {ACCEPT, MEMORY, REFUSE} <= set(total):
        print("\n  no pre-registered reading: this comparison is missing one of "
              "ACCEPT, MEMORY, REFUSE (a --only run). The paired tables above stand; the "
              "criterion does not apply.")
        return 0
    acc, mem, ref = total[ACCEPT], total[MEMORY], total[REFUSE]
    #: **A criterion computed over an empty group cannot fail**, and both controls exist in files
    #: written after 2026-09-13 only. An older pair is a real comparison of the three groups it
    #: has; it is not this criterion, and saying so is cheaper than a control that "holds" on n=0.
    empty = [n for n, t in (("ACCEPT", acc), ("MEMORY", mem), ("REFUSE", ref)) if not t["n"]]
    if empty:
        print(f"\n  NO PRE-REGISTERED READING: {', '.join(empty)} has no cases in this pair,")
        print("    so the criterion would be scored over a group that cannot move. The "
              "paired tables above stand on their own.")
        return 0
    print(f"\n  THE PRE-REGISTERED READING — measure/reoffer.py, 2026-09-13, before any fix existed")
    print(f"    primary   ACCEPT   {acc['fixed']} fixed, {acc['broken']} broken of n = "
          f"{acc['n']}, k = {acc['fixed'] + acc['broken']} discordant, p = {acc['p']:.4f}"
          + ("   ** below 0.05 **" if acc["p"] < 0.05 else "   (0.05 needs k >= 6 one way)"))
    for name, t in (("MEMORY", mem), ("REFUSE", ref)):
        held_ok = t["broken"] <= t["fixed"]
        note = ("holds" if held_ok else
                "** LOST MORE THAN IT GAINED: a noisier agent, not a better one **")
        print(f"    control   {name:<8} {t['fixed']} fixed, {t['broken']} broken of n = "
              f"{t['n']} — {note}")
    passed = (acc["p"] < 0.05 and acc["fixed"] > acc["broken"]
              and mem["broken"] <= mem["fixed"] and ref["broken"] <= ref["fixed"])
    print(f"    -> the pre-registered criterion is {'MET' if passed else 'NOT met'}. "
          f"FARACCEPT and RESTATE are")
    print("       secondary, reported above, and in no criterion.")
    print("\n  ACCEPT is the hypothesis now. MEMORY and REFUSE are the controls "
          "that say whether a gain is comprehension or an agent that acts on everything "
          "after an offer; ANEW says it can still act on something else.")
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
    p.add_argument("--changed", default="", metavar="KEY[,KEY]",
                   help="with --compare: declare which condition IS the change being scored. "
                        f"One or more of {', '.join(DECLARABLE)} — all of them inside argmind/. "
                        "Without this, --compare refuses when the prompt differs, which is every "
                        "fix that edits converse.py. Declaring a condition that did not actually "
                        "move refuses as well")
    args = p.parse_args(argv)
    if args.compare:
        return compare(*args.compare,
                       changed=[k.strip() for k in args.changed.split(",") if k.strip()])
    if args.changed:
        p.error("--changed is only meaningful with --compare")
    if args.repeat < 1:
        p.error("--repeat must be at least 1")

    screens = {name: widgets for name, _window, widgets in load()}
    want = {c["screen"] for c in CASES}
    missing = sorted(want - set(screens))
    if missing:
        p.error(f"the case list names screens that are not captured: {missing}")
    if len(want) < WANT_SCREENS:
        p.error(f"the case list has collapsed onto {len(want)} screen(s); this file claims "
                f"{WANT_SCREENS}")
    #: **The widget counts in `SCREEN_LINES` are asserted, not trusted.** They are printed beside
    #: every rate this file produces, and a re-capture that changed a window's size would otherwise
    #: leave four labels quietly describing the fixtures of a previous week.
    wrong = [(n, want_n, len(screens[n])) for n, _tag, want_n in SCREEN_LINES
             if n in screens and len(screens[n]) != want_n]
    if wrong:
        p.error(f"SCREEN_LINES disagrees with the fixtures: {wrong} (name, claimed, actual)")
    #: **Every group on every screen**, checked rather than described. This is the whole content of
    #: the 2026-09-13 enlargement, and a case list edited back onto two screens would put the screen
    #: back into every rate as a confound while the docstring went on claiming four.
    if WANT_EVERY_GROUP_ON_EVERY_SCREEN and not args.only:
        gaps = [(k, n) for k in GROUPS for n, _tag, _l in SCREEN_LINES
                if not any(c["kind"] == k and c["screen"] == n for c in CASES)]
        if gaps:
            p.error(f"these (group, screen) pairs have no case, so the screen is a confound for "
                    f"the group: {gaps}")

    cases = [c for c in CASES if not args.only or args.only in c["label"]
             or args.only in c["screen"] or args.only in c["ask"] or args.only in c["then"]]
    if not cases:
        p.error(f"--only {args.only!r} matched no case")
    dirty = filler_check(screens)
    if dirty:
        p.error(f"a FILLER_FAR line names a widget, which is the confound the far cases measure: "
                f"{dirty}")

    #: **Both of these print before the model is even opened.** A criterion nobody read before the
    #: run is a criterion chosen afterwards, and a sensitivity computed after the fact is a number
    #: that arrives too late to stop a run that could not have shown anything.
    plan = plan_note()
    projection = planned_sensitivity(cases)

    before_ps = residency()
    inner = Local(args.model, think={"on": True, "off": False}[args.think])
    model = Tapped(inner)
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
    #: The group sizes and **the call count derived from the cases**, not from a constant: four
    #: cases carry four filler turns each and a line that said `len(cases) * repeat * 2` would
    #: under-report by 96 calls on the full set.
    turns_total = sum(2 + len(c.get("filler", ())) for c in cases) * args.repeat
    print("  " + ", ".join(f"{sum(1 for c in cases if c['kind'] == k)} {k}" for k in GROUPS)
          + f" | {len(cases)} cases | {turns_total} calls")

    t0 = time.perf_counter()
    spends, shapes, slots = [], collections.Counter(), collections.Counter()
    rows = run_cases(model, screens, cases, args.repeat, spends, shapes, slots)
    took = time.perf_counter() - t0

    for kind in GROUPS:
        show(rows, kind, args.repeat)

    print("\n  THE WORDING THAT REACHED THE MODEL — the treatment, measured rather than assumed")
    for s, n in shapes.items():
        print(f"    x{n:<4} '  offered: {s}'")
    if not shapes:
        print("    ** NONE. Not one turn carried an offered line, so every case above is void "
              "and this run tested nothing. **")

    #: **The second channel, counted the same way.** The wording above lives in the transcript; this
    #: is the region of the prompt between the widget list and the person's words, which is where
    #: `96d8456` put its change and where the transcript cannot see it. Both are printed with `n`
    #: because "the treatment differed" is a claim about counts, not a feeling.
    print("\n  THE REGION BESIDE THE TURN — the other channel, over every turn of every repeat")
    for shape, n in slots.items():
        print(f"    x{n:<4} {shape}")

    #: **Every group's sensitivity, before any comparison exists.** `k` discordant pairs give
    #: `2 x 0.5^k`, so a group with fewer than six failures in this arm cannot show a repair as
    #: significant however good the repair is. Printed for all six because a control that cannot
    #: move is as much a design fact as a hypothesis that cannot.
    print(f"\n  WHAT THIS ARM COULD SHOW — sensitivity, printed before any comparison")
    for kind in GROUPS:
        live = [r for r in rows if r["kind"] == kind and not r["void"]]
        if not live:
            continue
        bad = sum(1 for r in live if not r["ok"])
        pv, says = sensitivity(bad)
        void = sum(1 for r in rows if r["kind"] == kind and r["void"])
        star = "   <- PRIMARY ENDPOINT" if kind == ACCEPT else ""
        print(f"    {kind:<10} {len(live) - bad} of {len(live)} correct ({void} void), "
              f"{bad} failing -> a perfect repair gives p = {pv:.4f}{star}")
        if bad and pv >= 0.05:
            print(f"      {says}")

    cost = spend_summary(spends, ceiling=inner.max_tokens)
    print_spend(cost)
    channels = channel_note()

    parts = [args.model.replace(":", "-").replace("/", "-"), args.think, prompt_tag(),
             f"s{screen_print()}", f"f{data_print()}", f"h{harness_print()}", f"l{loop_l}",
             f"x{args.repeat}"]
    if args.only:
        parts.append("only" + hashlib.sha256(args.only.encode()).hexdigest()[:6])
    out = HERE / ("reoffer-" + "-".join(parts) + ".json")
    #: **Never overwrite a result.** Two arms of this experiment wrote one path once, and the
    #: second took a committed baseline with it — 33 minutes to get it back out of git. Every
    #: condition this file knows about is in the name, and the name is still not proof: `capture.py`
    #: and `sending.py` are recorded in the JSON and not in it.
    if out.exists():
        print(f"\n  ** REFUSING TO WRITE: {out.name} already exists. Move or rename it and "
              f"re-run; do not delete it. **")
        print("     If this is a second arm, something that should differ does not: check "
              "loop_print,")
        print("     prompt and tree in the existing JSON against the header printed above.")
        inner.release()
        return 3
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
        #: The second channel, and **stored under its own key** rather than merged into
        #: `offered_shapes`: they are two regions of the prompt and a comparison has to be able to
        #: say which one moved.
        "slot_shapes": dict(slots),
        "channels_note": channels, "plan": plan, "planned_sensitivity": projection,
        "filler_far": FILLER_FAR,
        "ollama_before": before_ps, "ollama_after": residency(),
        "seconds": round(took, 1), "calls": model.calls, "retried": model.retried,
        "spend": cost,
        "totals": {k: {"n": sum(1 for r in rows if r["kind"] == k),
                       "void": sum(1 for r in rows if r["kind"] == k and r["void"]),
                       "ok": sum(1 for r in rows if r["kind"] == k and r["ok"]),
                       "disagreed": sum(1 for r in rows if r["kind"] == k and not r["agreed"])}
                   for k in GROUPS},
        "cases": CASES, "rows": rows,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n  every turn, verbatim -> {out}")
    inner.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
