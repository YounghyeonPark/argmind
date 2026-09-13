# SPDX-License-Identifier: Apache-2.0
"""Can you correct it? Can it tell you what it did? Does it answer in your language?

    python -m measure.repair --help
    python -m measure.repair                       # the shipped arm: qwen3:8b, thinking off, x3
    python -m measure.repair --only 고마워          # one case, while writing

**The owner's priority is interaction, not accuracy** — *"약간의 실수는 괜찮음. 계속 고쳐나가게 하면
됨. 중요한건 상호작용 및 의사소통임."* Everything else in this directory measures choosing. This
measures whether the turn can be repaired and whether you are answered in the language you used.

## The three claims, and what would refute each

**Repair** — it picks the wrong thing, you say so, and you name the right one. Does the *second*
turn land? Going in, the prediction was that it could not: the history recorded what you said and
what it said, and **not what it offered**, so the *that* in "not that one" referred to nothing.
`Session.remember_doing()` now writes the offer into the conversation, which is what this exercises.
*Refuted by*: a run where the second turn does not click the named target. Two of the four cases
want a specific widget and **two want nothing clicked at all**, so an agent that clicks on every
turn and an agent that never clicks both score 2 of 4. A set where every case wanted a click could
be passed by clicking always.

**Memory of its own actions** — you ask what it just did. `MEMORY of what it offered` is one of the
four repair cases and it has been failing since 2026-09-10 for a known reason: a question about the
past makes it propose the action again, correctly narrating it while doing so. That is a **recorded
open defect** (`docs/findings/interaction.md`), deliberately not chased — in the app the effect is a
proposal with a button, which is the "a few mistakes are fine" case. A 3-of-4 here is that defect,
not a fault in this file.

**Language** — a Korean question answered in English is a communication defect, not a taste.
*Refuted by*: a Korean-in case answered with no Hangul. And **the mirror is measured too**, because
a prompt line that simply forces Korean would score 36 of 36 on the Korean set while making the
agent unusable in English: `ENGLISH` is twelve English turns where a Korean reply is the failure.
The first six of those mirror the first six Korean cases word for word. Citing one half without the
other is incomplete by construction, exactly as with `choosing` and `talking`.

## Why the sets are the size they are

The `KOREAN` set was **six cases** until 2026-09-13 and could not carry the figure that was in
`CLAUDE.md`. A later run is paired with this one case by case, so a repair is an exact paired
(sign) test on the cases that change: fixing every one of `k` failures and breaking nothing is
`p = 2 x 0.5^k`, which needs **k >= 6** to reach 0.05. Six cases showing one or two failures could
reach `p = 0.5` at best — the set could not have shown a fix even if a fix were perfect. At the
1-in-6 failure rate observed in September, **36 cases** puts the expected number of discordant pairs
at 6, which is the minimum that can reach 0.05.

**The original six are the first six and they are unchanged**, and their sub-rate is printed
separately, so the September figure still has something to be compared against. The thirty added
were written before this was run and cover the kinds the six did not: social, about itself,
requests, the past, mixed Korean-with-Latin input, the formal register, chit-chat and one long
sentence. They were **not** chosen for being easy — the run below reports the rate per kind, so a
kind that only passes is visible as such.

## What this number is a function of

Model, thinking, the rendered prompt (`q`), the fixture screens (`s`), the case lists (`f`), this
harness (`h`), the repeat count (`x`) — all seven in the filename — and, in the JSON, `loop.py`,
`llm.py`, `converse.py`'s `MAX_LISTED` / `MOST_AT_ONCE`, `remember_turns`, and what else was
resident on the GPU. **Until 2026-09-13 the filename carried the model and the thinking flag and
nothing else**, so two prompt experiments overwrote the one baseline this file had.

`measure/repair-qwen3-8b-off.json` is that file and **nothing in it is a measurement of any tree
that has ever shipped**. Reconstructed from the commit graph: it was written at 10:14 on
2026-09-10, two minutes before `38aa39c`, when `converse.py` was `pd307bb` — a prompt so much older
than the current scheme that `prompt_print()` raises `AttributeError` on it, because `MOST_AT_ONCE`
did not exist yet — and when `Session.remember` counted **lines**, sliced `history[-12:]`, and
opened 10.7% of prompts in the middle of a turn. Its `korean 4 of 6` is worse still: per
`docs/findings/interaction.md` it is the run taken **under an added `SYSTEM` line that was reverted
the same day** for breaking `고마워`, so its prompt is in no commit at all. The `5 of 6` quoted in
`CLAUDE.md` belongs to the run that file overwrote. Neither figure was ever true of the shipped
agent, and only one of the two can even be re-derived — by re-running, which is what the results
named `-qa62805-` beside it are.

## What it would still fail to notice changing

* **Three of the four channels of `Session.remember_doing()` are never fed.** This harness only ever
  `offered`; nothing is performed, refused or turned down, so a change to the wording of `did`,
  `could not` or `they said no to` cannot move any number here. The run counts and prints this off
  `remember_doing`'s own signature, so a fifth channel would announce itself.
* **Only the second turn of the four repair cases carries a conversation.** Every language case is
  one turn, so most of this run's turns render no `Recently:` block at all — printed as a count of
  this run rather than claimed here. `measure/sending.py` names this file as covering the history
  axis for the three harnesses that send none; at four second-turns per repeat, all at distance 1,
  that is a thin claim and the block below says so in numbers.
* **Two of the four captured screens**, and `Fixed` eyes: nothing moves between the decision and the
  action, so `Session.find()`'s staleness path and `vanished` are unreachable here.
* Whether a reply is a *sentence* — `measure/saying.py` covers that, and it is where the index leak
  is counted. A reply here can be Korean, on topic, and still read like a debug line.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import inspect
import json
import pathlib
import re
import sys
import time

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from argmind.core import Session                                          # noqa: E402
from argmind.core.converse import MAX_LISTED, MOST_AT_ONCE                # noqa: E402
from argmind.core.llm import Local                                        # noqa: E402
from argmind.core.loop import REMEMBER_TURNS                              # noqa: E402
from measure.capture import load                                          # noqa: E402
from measure.choosing import (prompt_fields, prompt_tag, print_spend,     # noqa: E402
                              spend_summary)
#: `drive()` is **the app's bookkeeping reproduced by hand** — say, reply, then
#: `remember_doing(offered=...)` — and this file used to reproduce it a second time, in a local
#: `talk()`. Two hand copies of `argmind/app.py`'s `_think` + `_propose` can drift apart silently and
#: one of them would then be measuring a loop the app does not run. There is one copy now, in the
#: file that documents it. `Fixed` and `Nothing` moved out of here to `measure/standins.py` at the
#: same time; importing them from a leaf harness is what made this import a cycle.
from measure.conversation import (Recorder, drive, file_print,            # noqa: E402
                                  residency, screen_print)
from measure.saying import worst_index                                    # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent

# ---- the cases, written before anything ran ------------------------------------------------------

#: `want` is a **name and a role**, or `None` for "nothing should be clicked".
#:
#: **The role is half of the key and scoring without it flattered.** Character Map lists a `text`
#: label reading `Font :` at index 8 and a `combo box` of the same name at index 9; the click that
#: matters is the combo box and the click on the label does nothing, which is indistinguishable from
#: a click that missed. Until 2026-09-13 this file compared names only, so either one scored as a
#: repair that landed. Both verdicts are reported below, with the delta between them.
CASES = [
    {"label": "REPAIR names the right thing", "screen": "untitled---paint",
     "lines": ["빨간색 골라줘", "아니 그거 말고 Red 를 눌러줘"],
     "want": ["Red", "list item"],
     "why": "the correction names the target in as many words. If this fails, nothing repairs."},
    {"label": "REPAIR without naming it", "screen": "untitled---paint",
     "lines": ["동그라미 그리고 싶어", "아니야 그거 말고"],
     "want": None,
     "why": "'not that one' with no replacement. The right answer is to stop and ask, not to "
            "guess again — so this is one of the two cases an always-clicking agent fails."},
    {"label": "REPAIR after a vague ask", "screen": "character-map",
     "lines": ["뭔가 눌러줘", "글꼴 바꾸는 걸로 해줘"],
     "want": ["Font :", "combo box"],
     "why": "the first turn cannot be answered, the second narrows it. The label/combo box pair "
            "is here on purpose: this is the case where a name-only verdict cannot tell a click "
            "that would work from one that would do nothing."},
    {"label": "MEMORY of what it offered", "screen": "character-map",
     "lines": ["복사 눌러줘", "방금 뭘 하려고 했어?"],
     "want": None,
     "why": "a question about the past. KNOWN OPEN DEFECT since 2026-09-10: it answers correctly "
            "and clicks Copy a second time. Kept as the standing record of that defect."},
]

#: Korean in, Korean expected out. One turn each, on the smallest screen so the answer is short.
#: `(kind, text)`. **The first six are the original set, in their original order** — see the
#: docstring for why the other thirty exist and why they are not there to raise a score.
ORIGINAL_KOREAN = 6
KOREAN = [
    ("screen", "뭐가 열려 있어?"),
    ("request", "이 창 닫아줘"),
    ("screen", "여기서 뭘 할 수 있어?"),
    ("social", "고마워"),
    ("screen", "지금 화면 설명해줘"),
    ("screen", "복사 버튼 어디 있어?"),
    # ---- added 2026-09-13, for power. Written before the run. ----
    ("social", "안녕"),
    ("social", "수고했어"),
    ("social", "잘 부탁해"),
    ("social", "미안해"),
    ("social", "반가워"),
    ("itself", "너 뭘 할 수 있어?"),
    ("itself", "너 누구야?"),
    ("itself", "한국어 할 줄 알아?"),
    ("itself", "클릭도 돼?"),
    ("itself", "너 이름이 뭐야?"),
    ("request", "복사 버튼 눌러줘"),
    ("request", "글꼴 바꿔줘"),
    ("request", "아무거나 하나 눌러줘"),
    ("request", "도움말 열어줘"),
    ("request", "문자 하나 선택해줘"),
    ("screen", "버튼이 몇 개야?"),
    ("screen", "이 창 이름이 뭐야?"),
    ("screen", "지금 뭐가 선택돼 있어?"),
    ("screen", "화면에 뭐가 보여?"),
    ("past", "방금 뭐 했어?"),
    ("past", "아까 뭐 눌렀어?"),
    #: Korean with a Latin word in it. A plausible mechanism for an English answer is that a Latin
    #: token in the question tips the reply's language, and nothing measured it.
    ("mixed-in", "Red 버튼 있어?"),
    ("mixed-in", "Copy 가 어디야?"),
    ("mixed-in", "Font 어떻게 바꿔?"),
    #: The formal register. A model that answers 해요체 fluently can still drop to English when
    #: addressed in 합쇼체, and the app's users are not all going to be casual.
    ("formal", "이 창을 닫아 주시겠습니까?"),
    ("formal", "지금 무엇을 할 수 있습니까?"),
    ("chit-chat", "오늘 날씨 어때?"),
    ("chit-chat", "심심해"),
    ("chit-chat", "배고파"),
    ("long", "이 창에서 특수 문자를 하나 골라서 복사하고 싶은데 어떻게 하면 좋을까?"),
]

#: **The other direction.** English in, English expected out. Without this set a prompt line reading
#: "always answer in Korean" would score 36 of 36 above and break the agent for anybody typing
#: English. The first six mirror the original six Korean cases so the pair is readable side by side.
ENGLISH = [
    ("screen", "What's open?"),
    ("request", "Close this window"),
    ("screen", "What can I do here?"),
    ("social", "Thanks"),
    ("screen", "Describe the screen"),
    ("screen", "Where is the copy button?"),
    ("itself", "Who are you?"),
    ("screen", "How many buttons are there?"),
    ("past", "What did you just do?"),
    ("request", "Press the copy button"),
    ("chit-chat", "How's the weather?"),
    ("request", "Can you change the font for me?"),
]

#: The screen every language case runs on. Named rather than picked by size: `sorted(...)[0]` would
#: silently move the whole language measurement to another window the day a smaller one is captured.
LANGUAGE_SCREEN = "character-map"

# ---- reading a reply -----------------------------------------------------------------------------

#: Worst last in each list. On the Korean set an English answer is the defect under test; on the
#: English set a Korean answer is. `OTHER` — letters in a third script — ranks below both, because
#: an English fallback is at least a language the person can be assumed to reach, and the observed
#: `OTHER` is Chinese answering a Korean apology. `NEITHER` is worst on all counts because it is not
#: a wrong-language reply at all: nothing was said, and `spend` says whether the turn ran to the
#: token ceiling. Four states, and the run below has reached every one of them.
KOREAN_OUT, ENGLISH_OUT, OTHER, NEITHER = "korean", "english", "other", "neither"
LANGUAGE_WORST = {"korean": [KOREAN_OUT, ENGLISH_OUT, OTHER, NEITHER],
                  "english": [ENGLISH_OUT, KOREAN_OUT, OTHER, NEITHER]}

#: Worst last. `EXTRA` exists because "it clicked the right thing" and "it clicked the right thing
#: and two others" were the same verdict under the old rule, which took the **first** click and
#: ignored the rest — so a reply that landed Red and also cleared the canvas read as a clean repair.
LANDED, EXTRA, WRONG = "landed", "extra", "wrong"
REPAIR_WORST = [LANDED, EXTRA, WRONG]


def script_of(text: str) -> tuple[str, bool]:
    """`(korean | english | other | neither, mixed)` for one reply.

    **Four outcomes, and each of the last three was reached by a real reply in the first run of
    this file.** The predecessor was `is_korean(text) -> bool`, and two of the four collapse into
    its `False`:

    * `is_korean("")` was `False`, so on the English control set **a reply that never arrived would
      have scored as a correct English answer**. A turn that runs to the token ceiling produces
      `say = ""`, and the `--think on` arm did it five times in 168 calls.
    * `other` exists because the `--think on` arm answered *미안해* with **`没关系，有什么需要帮忙的吗？`**
      — Chinese. No Hangul and no ASCII letters, so a three-way version printed it under "no
      letters at all — empty, digits or punctuation only", which is a display disagreeing with the
      reply beside it. Wrong language and no language are different defects with different causes.

    Any Hangul at all counts as Korean — a reply that mixes is still speaking to the person — but
    `mixed` is returned beside it so "answered in Korean" and "answered in Korean and quoted an
    English widget name" are distinguishable in the record rather than merged into the rate.
    """
    t = text or ""
    hangul = any("가" <= c <= "힣" or "ᄀ" <= c <= "ᇿ"
                 or "㄰" <= c <= "㆏" for c in t)
    latin = any(c.isascii() and c.isalpha() for c in t)
    if hangul:
        return KOREAN_OUT, latin
    if latin:
        return ENGLISH_OUT, False
    #: `str.isalpha()` is True for Hangul and for CJK alike; Hangul has already been taken, so what
    #: reaches here is a letter in some third script.
    return (OTHER if any(c.isalpha() for c in t) else NEITHER), False


def repair_verdict(clicks, want):
    """`landed | extra | wrong` for one repeat of one repair case.

    `clicks` is `[(name, role)]` in the order the model produced them; `want` is `(name, role)` or
    `None`. Scored on the **pair**, because a name is not a key — see `CASES`.
    """
    if want is None:
        return LANDED if not clicks else WRONG
    if tuple(want) not in clicks:
        return WRONG
    return LANDED if len(clicks) == 1 else EXTRA


def old_rule(clicks, want) -> bool:
    """The pre-2026-09-13 verdict: the **first** click's name, role ignored, extras ignored.

    Kept and reported only so the new headline can be read against the `3 of 4` that is quoted in
    `CLAUDE.md` and `docs/findings/interaction.md`. Never the score.
    """
    first = clicks[0][0] if clicks else None
    return first == (want[0] if want else None)


def sensitivity(bad: int) -> tuple[float, str]:
    """Two-sided exact paired p **if a later run fixed every failure and broke nothing**, and the
    plain-language version.

    Printed before a comparison is ever made, not after: this project has published a Fisher
    `p = 0.070` on `n = 28` and only worked out afterwards that the design could not have reached
    0.05. `k` discordant pairs give `2 x 0.5^k`, so six failures is the floor for significance.
    """
    if not bad:
        return 1.0, "nothing failed, so there is nothing a repair could move"
    p = 2 * 0.5 ** bad
    if p < 0.05:
        return p, "a perfect repair of this set would be significant"
    return p, (f"** this set CANNOT show a repair as significant — {bad} failure"
               f"{'' if bad == 1 else 's'} gives {p:.3f} at best, and 6 are needed for 0.05 **")


# ---- the conditions ------------------------------------------------------------------------------

def data_print() -> str:
    """Six characters over all three case lists.

    `f` in the filename. A case list edited between two runs is two experiments sharing one name,
    which is how three baselines were destroyed in this directory — and how the only baseline this
    file ever had came to describe a prompt that was reverted.
    """
    blob = json.dumps([CASES, KOREAN, ENGLISH, LANGUAGE_SCREEN, ORIGINAL_KOREAN],
                      ensure_ascii=False, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:6]


def harness_print() -> str:
    """Six characters over this file. `h` in the filename.

    Delegated to `sending.harness_print` so there is one definition of it; this file is not a
    `sending.turn()` caller and owes that module nothing else — see `history_note()`.
    """
    from measure import sending                                          # noqa: PLC0415

    return sending.harness_print(__file__)


#: The channels `Session.remember_doing()` accepts, read off the function rather than listed here,
#: so a fifth one added to `loop.py` shows up as unexercised without anybody editing this file.
#: `self` is dropped because the signature is read off the unbound function; every other parameter
#: is a channel.
CHANNELS = tuple(n for n in inspect.signature(Session.remember_doing).parameters if n != "self")
#: The one `drive()` feeds, reproducing `app.py`'s `_propose`. Checked against `CHANNELS` at run
#: time: if it is ever renamed, this refuses to run rather than reporting a clean zero.
FED = ("offered",)


def history_note(turns_with, turns_total, kinds) -> dict:
    """**The inverse of `sending.note()`**, printed and returned as one thing.

    `measure/sending.py` exists because three harnesses sent no conversation and nothing said so.
    This harness is one of the two that *do*, so it owes the opposite disclosure and it owes it in
    the same shape: measured off this run, printed beside the headline, and stored as the object
    that was printed rather than recomputed for the file.

    What it will not do is print a tick. `sending.COVERS_HISTORY` names this file as covering the
    history axis for `choosing`, `saying` and `talking`; the numbers below are what that claim is
    actually worth, and at four second-turns per repeat it is thin. The distance axis and the
    controls are in `measure/conversation.py` and are not here.
    """
    missing = [c for c in FED if c not in CHANNELS]
    if missing:
        raise RuntimeError(
            f"Session.remember_doing() no longer takes {missing} — its channels are now {CHANNELS}. "
            f"This harness feeds one of them by way of measure.conversation.drive(); until that is "
            f"re-read, every 'the conversation carried the offer' reading here is unsupported.")
    unfed = [c for c in CHANNELS if c not in FED]
    pct = 100 * turns_with / turns_total if turns_total else 0.0
    print("\n  THE CONVERSATION — EXERCISED, THINLY, AND HERE IS HOW MUCH")
    print(f"    {turns_with} of {turns_total} turns rendered a 'Recently:' block "
          f"({pct:.0f}%); the rest opened a fresh conversation.")
    if turns_with:
        print("    Every one of those is the second turn of a repair case, so the only distance "
              "measured here is 1.")
    else:
        print("    ** NONE. This run sent no conversation at all, so it covers the history axis "
              "for nothing. **")
    print(f"    history lines the model was actually shown, by kind: "
          f"{', '.join(f'{k} {v}' for k, v in sorted(kinds.items())) or 'none'}")
    print(f"    remember_doing channels fed: {', '.join(FED)}   "
          f"NEVER fed: {', '.join(unfed)}")
    print("    ** so a change to the wording of those channels, or to anything that only appears")
    print("       after an effect is performed, cannot move a number in this file. **")
    print("    the distance axis, the with/without controls and the truncation count are in")
    print("    measure/conversation.py — this file is one conversation per case, no control arm.")
    return {"history": {"turns_with_block": turns_with, "turns": turns_total,
                        "line_kinds_shown": dict(kinds), "distances": [1],
                        "channels": list(CHANNELS), "channels_fed": list(FED),
                        "channels_never_fed": unfed, "printed": True}}


# ---- the run -------------------------------------------------------------------------------------

def as_spend(turn) -> dict:
    """`measure.choosing.spend()`'s five keys, off a `drive()` turn record.

    `drive()` already stores each of them; re-asking the `Reply` for them would mean keeping the
    reply objects alive across the run for no gain, and a second reader of the same fields is a
    second thing that can disagree.
    """
    return {k: turn[k] for k in ("seconds", "out_tokens", "reason", "truncated", "thought_chars")}


#: `  who: ` at the very start of a line and nowhere else. `who` may contain spaces — `they said
#: no to` is one of `remember_doing`'s labels — but never a colon.
LINE = re.compile(r"^ {2}([^\s:][^:]{0,23}): ")


def block_kinds(window: str, into) -> None:
    """Count the `  who: text` lines of a rendered `Recently:` block, by `who`.

    Read off **the block that reached the model**, not off `session.history`: a line written but
    truncated out of the window was not shown, and counting the history would report it as if it
    had been. The format is `converse()`'s — `f"  {who}: {text}"` — and `recent_block()` has already
    raised if it is gone. **Matched on exactly that shape rather than on "the text before the first
    colon"**: a reply is stored in the history verbatim, so a reply containing a newline puts
    unindented continuation lines in the block, and one of those carrying a colon was counted as a
    fifth kind of history line that `loop.py` cannot produce.
    """
    for line in window.splitlines():
        got = LINE.match(line)
        if got:
            into[got.group(1)] += 1


def run_repair(model, screens, cases, repeat, spends, kinds):
    """The repair set. Returns `(rows, turns_seen, turns_with_block)`."""
    rows, seen, withblock = [], 0, 0
    if not cases:
        return rows, seen, withblock
    print(f"  REPAIR AND MEMORY — n = {len(cases)} cases x {repeat} repeats, worst counted\n")
    for case in cases:
        want = tuple(case["want"]) if case["want"] else None
        draws = []
        for r in range(repeat):
            turns = drive(model, screens[case["screen"]], case["lines"], REMEMBER_TURNS)
            for i, t in enumerate(turns):
                seen += 1
                if t["window"].strip():
                    withblock += 1
                    block_kinds(t["window"], kinds)
                spends.append((f"{case['label']} / r{r + 1} / turn {i + 1}", as_spend(t)))
            last = turns[-1]
            clicks = list(zip(last["clicked"], last["clicked_roles"]))
            draws.append({"turns": turns, "clicks": clicks,
                          "verdict": repair_verdict(clicks, want),
                          "old_rule": old_rule(clicks, want)})
        verdicts = [d["verdict"] for d in draws]
        i = worst_index(verdicts, REPAIR_WORST)
        agreed = len(set(verdicts)) == 1
        #: The worst repeat, and **the transcript printed is that same repeat's**. Printing repeat
        #: one's reply beside a worst-of-three verdict has shipped twice in this project.
        show = draws[i]
        mark = {LANDED: "OK  ", EXTRA: "EXTRA", WRONG: "NO  "}[verdicts[i]]
        print(f"    {mark:<5} {case['label']:<28} wanted {str(want):<22} "
              f"got {show['clicks'] or 'nothing'}   [worst of {repeat} = r{i + 1}"
              f"{'' if agreed else ', repeats DISAGREED: ' + '/'.join(verdicts)}]")
        for t in show["turns"]:
            print(f"           you: {t['said']}")
            got = "".join(f'  [{n} ({ro})]' for n, ro in zip(t["clicked"], t["clicked_roles"]))
            print(f"           it : {t['reply'][:74]!r}{got}"
                  + (f"   <{t['n_effects']} effects>" if t["n_effects"] > 1 else ""))
        rows.append({"set": "repair", "case": case["label"], "screen": case["screen"],
                     "lines": case["lines"], "want": case["want"], "why": case["why"],
                     "verdicts": verdicts, "counted": verdicts[i], "from_repeat": i,
                     "agreed": agreed, "old_rule_each": [d["old_rule"] for d in draws],
                     "old_rule_counted": all(d["old_rule"] for d in draws),
                     "repeats": draws})
    return rows, seen, withblock


def run_language(model, widgets, cases, which, repeat, spends):
    """One-turn cases. `which` is `"korean"` or `"english"` and selects the worst-ordering only —
    the classifier does not know which set it is looking at."""
    rows = []
    for kind, line in cases:
        draws = []
        for r in range(repeat):
            turns = drive(model, widgets, [line], REMEMBER_TURNS)
            t = turns[0]
            spends.append((f"{which} / {line[:28]} / r{r + 1}", as_spend(t)))
            got, mixed = script_of(t["reply"])
            draws.append({"say": t["reply"], "got": got, "mixed": mixed,
                          "clicked": t["clicked"], "n_effects": t["n_effects"],
                          "truncated": t["truncated"], "reason": t["reason"]})
        verdicts = [d["got"] for d in draws]
        i = worst_index(verdicts, LANGUAGE_WORST[which])
        rows.append({"set": which, "kind": kind, "said": line, "verdicts": verdicts,
                     "counted": verdicts[i], "from_repeat": i,
                     "agreed": len(set(verdicts)) == 1,
                     "mixed_counted": draws[i]["mixed"], "repeats": draws})
    return rows


def show_language(rows, which, want, repeat, note: str) -> None:
    """The four-way tally, per kind, and the worst repeat's own text beside its own verdict."""
    n = len(rows)
    if not n:
        return
    counted = collections.Counter(r["counted"] for r in rows)
    print(f"\n  {note} — n = {n} cases x {repeat} repeats, worst counted")
    for kind in LANGUAGE_WORST[which]:
        why = {KOREAN_OUT: "some Hangul in the reply",
               ENGLISH_OUT: "Latin letters, no Hangul",
               OTHER: "letters, but neither Hangul nor Latin — Chinese has been seen here",
               NEITHER: "no letters at all — empty, digits or punctuation only"}[kind]
        #: Named separately. A wrong-language reply is the defect this set is for; a reply with no
        #: letters in it is a turn that did not happen, and merging the two under one arrow is how
        #: a truncation gets counted as a language failure.
        flag = ("" if kind == want else
                "   <- the defect this set measures" if kind != NEITHER else
                "   <- not a language failure: nothing was said. See the truncation line below")
        print(f"    {kind:<8} {counted[kind]:>3} of {n}  ({100 * counted[kind] / n:>3.0f}%)  "
              f"{why}{flag}")
    mixed = sum(1 for r in rows if r["counted"] == KOREAN_OUT and r["mixed_counted"])
    if want == KOREAN_OUT:
        print(f"    of the {counted[KOREAN_OUT]} Korean replies, {mixed} also contained Latin "
              f"letters — a widget name quoted, usually")
    kinds = list(dict.fromkeys(r["kind"] for r in rows))
    print("    by kind: " + "   ".join(
        f"{k} {sum(1 for r in rows if r['kind'] == k and r['counted'] == want)}"
        f"/{sum(1 for r in rows if r['kind'] == k)}" for k in kinds))
    for r in rows:
        if r["counted"] != want:
            i = r["from_repeat"]
            print(f"      {r['counted'].upper():<8} {r['said'][:30]:<30} r{i + 1} "
                  f"{r['repeats'][i]['say'][:56]!r}")
    dis = [r for r in rows if not r["agreed"]]
    print(f"    disagreed across {repeat} repeats: {len(dis)} of {n}"
          f"{' — a single run is not this number' if dis else ''}")
    bad = n - counted[want]
    p, says = sensitivity(bad)
    print(f"    sensitivity: {bad} failing, so a later run that fixed all of them and broke "
          f"nothing gives exact paired p = {p:.3f} — {says}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m measure.repair", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="qwen3:8b", help="what ships")
    p.add_argument("--think", choices=("on", "off"), default="off",
                   help="defaults to off, which is what ships. Left unset once elsewhere here, a "
                        "reasoning model reasoned and a run meant to measure a prompt measured the "
                        "prompt AND the thinking; only the latency told on it")
    p.add_argument("--repeat", type=int, default=3,
                   help="worst of N counted, never the best and never the mean. At temperature 0 "
                        "in one process repeats have come back identical 26 of 26 elsewhere here, "
                        "so this usually costs time and buys a printed zero — which is the "
                        "evidence that one draw was enough, and is not the same as assuming it")
    p.add_argument("--only", default=None, metavar="TEXT",
                   help="substring of a case's text, label or screen, for one case while writing. "
                        "Hashed into the filename so a one-case look cannot overwrite a full run")
    args = p.parse_args(argv)
    if args.repeat < 1:
        p.error("--repeat must be at least 1")

    screens = {name: widgets for name, _window, widgets in load()}
    cases = [c for c in CASES if not args.only
             or args.only in c["label"] or args.only in c["screen"]
             or any(args.only in line for line in c["lines"])]
    korean = [c for c in KOREAN if not args.only or args.only in c[1] or args.only in c[0]]
    english = [c for c in ENGLISH if not args.only or args.only in c[1] or args.only in c[0]]
    if not (cases or korean or english):
        p.error(f"--only {args.only!r} matched no case in any of the three sets")

    before_ps = residency()
    inner = Local(args.model, think={"on": True, "off": False}[args.think])
    model = Recorder(inner)
    for _ in range(2):
        model.ask("Reply with the single word: ready")

    print(f"\n  {args.model} | thinking {args.think} | x{args.repeat} | prompt {prompt_tag()} | "
          f"screens s{screen_print()} | cases f{data_print()} | harness h{harness_print()}")
    print(f"  {len(cases)} repair cases, {len(korean)} Korean, {len(english)} English control | "
          f"remember_turns {REMEMBER_TURNS} | MAX_LISTED {MAX_LISTED} | "
          f"MOST_AT_ONCE {MOST_AT_ONCE}\n")

    t0 = time.perf_counter()
    spends, kinds = [], collections.Counter()
    rows, seen, withblock = run_repair(model, screens, cases, args.repeat, spends, kinds)

    landed = sum(1 for r in rows if r["counted"] == LANDED)
    if rows:
        counted = collections.Counter(r["counted"] for r in rows)
        n = len(rows)
        print()
        for kind in REPAIR_WORST:
            why = {LANDED: "exactly the wanted click, and nothing else",
                   EXTRA: "the wanted click, plus at least one more",
                   WRONG: "the wanted click absent, or a click where none was wanted"}[kind]
            print(f"    {kind:<7} {counted[kind]:>3} of {n}   {why}")
        old = sum(1 for r in rows if r["old_rule_counted"])
        print(f"    {landed} of {n} landed. By the pre-2026-09-13 rule — first click's NAME only, "
              f"role and extras ignored — {old} of {n}, a delta of {old - landed}.")
        dis = [r for r in rows if not r["agreed"]]
        print(f"    disagreed across {args.repeat} repeats: {len(dis)} of {n}")
        pv, says = sensitivity(n - landed)
        print(f"    sensitivity: {n - landed} failing, exact paired p = {pv:.3f} — {says}")
        print("    of the failures, 'MEMORY of what it offered' is a KNOWN OPEN DEFECT recorded on "
              "2026-09-10 and\n    deliberately not chased — see docs/findings/interaction.md. It "
              "is not a regression and not a fault here.")

    small = screens[LANGUAGE_SCREEN]
    krows = run_language(model, small, korean, "korean", args.repeat, spends)
    erows = run_language(model, small, english, "english", args.repeat, spends)
    seen += (len(krows) + len(erows)) * args.repeat
    rows += krows + erows

    show_language(krows, "korean", KOREAN_OUT, args.repeat,
                  f"LANGUAGE — Korean in, Korean out, on {LANGUAGE_SCREEN} ({len(small)} widgets)")
    six = krows[:ORIGINAL_KOREAN] if not args.only else []
    if six:
        ok6 = sum(1 for r in six if r["counted"] == KOREAN_OUT)
        print(f"    the original six, unchanged and first in the list — {ok6} of {len(six)}. "
              f"This is the row comparable\n      with the September figure, and it carries "
              f"n = {len(six)}: at best it could show a repair at p = 0.500.")
    show_language(erows, "english", ENGLISH_OUT, args.repeat,
                  "CONTROL — English in, English out. A rule that forced Korean would show here")
    print("  the Korean set and this control are one measurement in two halves; citing either "
          "alone is incomplete.")

    took = time.perf_counter() - t0
    cost = spend_summary(spends, ceiling=inner.max_tokens)
    print_spend(cost)
    blind = history_note(withblock, seen, kinds)

    #: **Every condition in the name.** Model, thinking, prompt, screens, cases, harness, repeats.
    #: This file wrote `repair-{model}-{think}.json` until 2026-09-13 and its single baseline was
    #: overwritten twice by prompt experiments; what survived described a `SYSTEM` line that was
    #: reverted the same day, and it is the number quoted in `CLAUDE.md`.
    parts = [args.model.replace(":", "-").replace("/", "-"), args.think, prompt_tag(),
             f"s{screen_print()}", f"f{data_print()}", f"h{harness_print()}", f"x{args.repeat}"]
    if args.only:
        #: Hashed, not stripped: `--only 고마워` and `--only 안녕` both reduce to nothing under an
        #: ASCII filter, and two one-case runs would then share a filename.
        parts.append("only" + hashlib.sha256(args.only.encode()).hexdigest()[:6])
    out = HERE / ("repair-" + "-".join(parts) + ".json")
    out.write_text(json.dumps({
        "model": args.model, "think": args.think, "repeat": args.repeat, "only": args.only,
        **prompt_fields(), **blind,
        "screens_print": screen_print(), "data_print": data_print(),
        "harness_print": harness_print(),
        #: `drive()`, `Recorder` and `recent_block` live there, so its source is a condition of this
        #: number in the same way this file's own is.
        "conversation_print": file_print("measure.conversation"),
        "standins_print": file_print("measure.standins"),
        "choosing_print": file_print("measure.choosing"),
        "llm_print": file_print(Local.__module__),
        #: **The windowing is a condition and only `conversation.py` recorded it.** This harness
        #: goes through a real `Session` — `recent()`, `remember_doing()`, the unknown-key filter in
        #: `decide()` — so `loop.py` moving moves this number and no fingerprint above would see it.
        #: The old baseline here was taken when `remember` counted LINES, not turns.
        "loop_print": file_print(Session.__module__), "remember_unit": "turns",
        "remember_turns": REMEMBER_TURNS, "max_listed": MAX_LISTED,
        "most_at_once": MOST_AT_ONCE, "max_tokens": inner.max_tokens,
        "language_screen": LANGUAGE_SCREEN,
        "screens": {k: len(v) for k, v in screens.items()},
        "ollama_before": before_ps, "ollama_after": residency(),
        "seconds": round(took, 1), "calls": model.calls, "retried": model.retried,
        "spend": cost,
        "repair": {"n": len(cases), "landed": landed,
                   "old_rule": sum(1 for r in rows
                                   if r["set"] == "repair" and r["old_rule_counted"])},
        "korean": {"n": len(krows),
                   **{k: sum(1 for r in krows if r["counted"] == k)
                      for k in LANGUAGE_WORST["korean"]},
                   "original_six": sum(1 for r in krows[:ORIGINAL_KOREAN]
                                       if r["counted"] == KOREAN_OUT) if not args.only else None},
        "english_control": {"n": len(erows),
                            **{k: sum(1 for r in erows if r["counted"] == k)
                               for k in LANGUAGE_WORST["english"]}},
        "cases": CASES, "korean_cases": KOREAN, "english_cases": ENGLISH,
        "rows": rows,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n  every turn, verbatim -> {out}")
    inner.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
