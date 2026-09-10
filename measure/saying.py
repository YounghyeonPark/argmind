# SPDX-License-Identifier: Apache-2.0
"""Is the reply something you would say to a person?

    python -m measure.saying                 both sets, three repeats, worst counted
    python -m measure.saying --repeat 1       one pass, for a quick look
    python -m measure.saying --only 지우개     one case

**The widget list is numbered and the person never sees the numbers.** They are an addressing
scheme for one round trip, and they leak into the speech. Eight requests probed by hand on the
shipped configuration: two answered with a bare index list (`"3, 122"`, `"0"`), one answered in
fluent Korean and told the person an internal index (*"지우개는 22번 버튼입니다."*), five were fine.
Nothing here measures it. `measure/choosing.py` records **which widget was chosen**;
`measure/talking.py` records **whether an action was produced**; neither stores `intent.say` at all,
so the whole defect is invisible to both.

## Why this is not a regex for digits

*"버튼이 21개 있어요"* is a correct answer to *"여기 버튼이 몇 개야?"*. *"복사 버튼은 4번입니다"* is a
leak. A harness that flags every digit scores the first as a defect and has no power; one that flags
nothing is worse. So there are two labelled sets and the harness has to survive both:

* **`LEAKING` — requests whose correct reply contains no number at all.** Any number that is not read
  off the screen is a defect here. This is the set that can show the leak. It was **enlarged from 14
  cases to 32 after the first four runs, for power and not for score**: the rate was 3 of 14 in every
  one of them, and three leaks is three discordant pairs against a later run — an exact paired test
  on three pairs cannot reach 0.05 no matter how well a repair works. The original fourteen are
  still the first fourteen entries, so the earlier number stays readable from any JSON, and the run
  prints what significance this set could reach.
* **`NUMBERS` — utterances where a number is legitimate**: counts, sizes, a value the model reads on
  screen. A number *missing* is the defect here. This is the set that stops the eventual fix from
  being "ban digits" — a prompt that bans them scores perfectly on the first set and collapses on
  this one, and that collapse is the number `conversation` will need.

**The label is a property of the correct REPLY, not of the correct action.** *"크롬 열어줘"* on Paint
is labelled here without deciding what it should click, because whatever the right answer is, it
contains no number. That is why these cases could be labelled confidently where `goals.json` had to
argue about which widget names count as right.

## What would refute the claim this file was written for

The claim is *"internal indices reach the person's ear"*. It is refuted if the leak set comes back
`clean`/`quoted` across three repeats — the hand probe was eight turns and eight turns is an
anecdote. It is **also** refuted, differently, if the leaked numbers turn out not to be indices: a
number is only *conclusively* an index when it equals the position of a widget the same reply
clicked, so that subset is counted and reported separately from the headline rate.

## The four ways a number gets classified, and the one that is a judgement call

    clean     no digits at all                                            not a leak
    quoted    every digit sits in a token that stands as a WHOLE token     not a leak
              in a widget name the model was shown — '1536', '10.0.26200'
              and 'i9-13900HX' pass; '22' inside '10/22/2025' does not
    index     a digit run that is a valid index AND is the position of a   LEAK, conclusive
              widget this same reply clicked
    stray     a valid index, nothing clicked there                        LEAK, not conclusive
    off-list  neither on the screen nor a valid index — invented           LEAK, not conclusive

**Only `index` is conclusive, and the first version of this docstring got that wrong.** It said a
legitimate count of actions — *"2가지 하겠습니다"* — would land in `off-list` where it could be
discounted. It does not: on a 21-widget screen 2 is a valid index, so it lands in `stray`, and on a
123-widget screen so does every small count. There is no list length at which the distinction works.
What is done instead is cheaper and honest: each number records `equals_effect_count`, and the
summary says how many of the non-conclusive leaks had a number equal to the number of effects the
same reply produced. So the headline rate, the conclusive subset and the arguable remainder are all
readable from one run, and none of them is a judgement made by this file.

**Single digits are never attributed to the screen.** `'1'` is a substring of `Color 1: Black`, so
attributing them would let every leak of index 0-9 through as "quoted". The cost is stated rather
than hidden: a legitimate single-digit value read off the screen would be counted as a leak, which
is exactly why no leak-set request asks for one — a self-check below refuses to run if any leak-set
request even contains a digit.

## The criterion's own failure mode, found by running it

**`index` can be a coincidence, and the control set caught one on the first run.** Asked *"여기
버튼이 몇 개야?"* on Character Map the model answered *"20개의 버튼이 있어요"* and clicked Minimize,
Maximize and Close — indexes 18, 19 and **20**. The count and a clicked index collided, and by the
rule above that is `index`. It is scored as nothing, because control-set numbers are never scored as
leaks, but it is proof that the "conclusive" label is defeasible rather than a hypothesis about it.

The headline rate does not rest on it: on the leak set the premise is that **no** number belongs, so
`22번 버튼입니다` and `3을 클릭해 주세요` are leaks whatever the index matching says. What the
matching adds is the demonstration that the number *is* the addressing scheme, and that is the claim
the coincidence weakens. Both figures are printed for that reason.

## The other half: a reply that says nothing

Five shapes, all decidable without judgement, so all five are counted:

    empty      nothing, or only punctuation — "…"
    numeric    nothing but digits and separators — "3, 122" is this AND an index leak, and on
               the control set a bare "10/22/2025" is this and is a correct terse answer, which
               is why every count below is printed split by set
    echo       a restatement of the request, normalised
    bare-ack   the whole reply is one acknowledgement token — "ok", "네", "알겠습니다"
    bare-name  the whole reply is one widget's name, verbatim — asked *"확대해줘"* it replies
               `Zoom in`, the interface's English label handed to the person

`bare-ack` and `bare-name` are the soft two, and `bare-name` is soft for a reason worth stating
because it was found by running the classifier rather than by thinking about it: asked *"메모리가
얼마나 있어?"* the model answers `32.0 GB`, which is **a widget's name verbatim and the correct
answer at the same time**. On a request for an action a bare label is the interface talking; on a
request for a value it is the value. So the counts are **printed split by set**, and the total is
printed both with and without the soft classes. Nothing is folded silently. **What this cannot
see is a fluent reply
with no content**, and no cheap test finds that; it is left unmeasured rather than guessed at. One
shape it does not even try: a reply that *claims* an action while the intent carries none — seen at
least once here (*"소프트웨어 환경을 열고 창을 최대화합니다"* with zero effects) and only visible
because `say` and the effects are stored side by side.

## What three repeats do and do not buy, measured on this harness

**Repeats inside one process mostly agree with themselves, and that is not the same as stability.**
Three identical replies on **26 of 26** cases in one run and **24 of 26** in the next — and in both
of the disagreeing cases it was repeat 1 that differed from repeats 2 and 3, which is the shape of a
warm cache rather than of sampling. Meanwhile running the whole file a second time changed the
wording of control replies that had been stable within each run: *"바이오스 날짜가 언제야?"* answered
`'BIOS 날짜는 10/22/2025입니다.'` in one process and `'BIOS Version/Date is 10/22/2025'` in the next,
from identical fingerprints. Most of the variation lives **between** processes, not within them, so
a within-run repeat count is cheap and reports less than it appears to.

What did hold across the two independent runs is every verdict on both axes, 26 of 26 — that is the
evidence for stability, and it costs a second invocation rather than a larger `--repeat`.

**`--reload` was added to buy independent draws inside one run and does not.** It unloads the weights
between repeats; on the one case tried it produced three identical replies, and their wording matched
the *first* run rather than the second, so the resident weights are not what moves between processes
and **I do not know what is**. The flag is kept because it eliminates one candidate and because a run
made under it is a different run, which is why it is in the filename. It is off by default: it pays a
model load per repeat and, on the evidence so far, buys nothing.

That leaves a condition **outside every fingerprint in the name** — `p`, `s`, `f` and `h` are all
identical across those two runs, and the text still moved. Anyone comparing two of these files should
expect the wording to differ and the verdicts not to.

## What this number is a function of

The model, the thinking flag, **the prompt as it is actually sent**, the fixture screens, the case
text, and the classifier in this file — all six are in the filename: `q` over the rendered prompt,
`s`/`f` as in `measure/conversation.py`, and `h` over this file itself, because the rule that turns
a reply into a verdict is as much a condition of the number as the prompt is. It is **not** a
function of a live desktop: fixtures, nothing clicked, no window touched.

**`q` replaced `p` on 2026-09-10 and the letter is the migration.** `p` was a hash of `converse.py`,
which moved when a comment moved and — the part that mattered — never carried the reply-shape line.
The two files `saying-…-p5461db-…` and `saying-…-pbcfe81-…` are this harness's 6-of-32 and 0-of-32
leak baselines, they differ by exactly that line, and their stored `prompt_text` fields are
byte-identical. `q` hashes what `converse()` renders and `prompt_record` stores it verbatim, so the
difference is now in the file rather than absent from it. A `p`-named file and a `q`-named file
cannot be compared by fingerprint at all; `converse_file_print` inside the JSON says which
`converse.py` a `q` run was made at, and that is the only bridge.

Three conditions it would fail to notice changing: the **language** of the reply (a leak in English
counts the same as one in Korean, and every case here is Korean, so an English-language leak rate is
unmeasured — `measure/repair.py` owns that question), **whether the widget the number points at was
the right one** (`choosing.py` owns that; a reply may be clean and wrong), and what `converse()`
does to a reply **after** it arrives — `_said()`'s de-misfiling and `_effect()`'s validation are not
sent to the model, so `q` cannot see them move.

One condition it *does* check rather than assume: that the model was shown `0:`-based numbering at
all. `describe()`'s output is asserted to be a substring of the prompt actually sent, so a change to
the numbering aborts this harness instead of quietly reporting no leaks.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import re
import string
import sys
import time

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from argmind.core import Said, converse                                   # noqa: E402
from argmind.core.converse import MAX_LISTED, describe                    # noqa: E402
from argmind.core.llm import Local                                        # noqa: E402
from measure.capture import load                                          # noqa: E402
from measure.choosing import (prompt_fields, prompt_tag, print_spend,     # noqa: E402
                              spend, spend_summary)
#: Reused rather than copied — `screen_print` and `residency` exist once, in the file that first
#: needed them, for the same reason `Fixed` and `Nothing` live in `measure/repair.py`.
from measure.conversation import Recorder, file_print, residency, screen_print   # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent

#: ---- the leak set -----------------------------------------------------------------------------
#: `(group, screen, what the person says)`. **Labelled before anything ran**, and the label is one
#: claim: *a correct reply to this contains no number*. Not which widget is right — that is
#: `goals.json`'s question and it is a harder one.
#:
#: Groups exist because "it leaks on a long list" and "it leaks when asked for two things" are
#: different problems, and an average over them would hide whichever is rarer. The **five** cases
#: marked `probed` — three that leaked, two that were fine — are the hand probes this file was
#: written for, kept so the rate can be compared against the anecdote it replaces. (This comment
#: said "three" while marking five, which is the stale-tally shape this project keeps finding in
#: its own prose.)
LEAKING = [
    # Paint — 123 widgets, the case that produced "3, 122" and "지우개는 22번 버튼입니다."
    ("one action", "untitled---paint", "지우개로 바꿔줘"),                       # probed: leaked
    ("one action", "untitled---paint", "빨간색으로 바꿔줘"),
    ("one action", "untitled---paint", "확대해줘"),
    ("two actions", "untitled---paint", "이 그림 저장하고 창 닫아줘"),            # probed: leaked
    ("two actions", "untitled---paint", "최소화하고 닫아줘"),                    # probed: fine
    ("nothing there", "untitled---paint", "크롬 열어줘"),
    # Character Map — 21 widgets. A short list is the control on list length: an index it cannot
    # get wrong is still an index it should not say out loud.
    ("one action", "character-map", "복사 눌러줘"),                             # probed: fine
    ("one action", "character-map", "글꼴 바꿔줘"),
    ("nothing there", "character-map", "볼륨 좀 줄여줘"),
    # System Information — 152 widgets, the longest list here. Produced "0".
    ("one action", "system-information", "하드웨어 리소스 열어줘"),               # probed: leaked
    ("one action", "system-information", "찾기 눌러줘"),
    ("two actions", "system-information", "소프트웨어 환경 열고 창 최대화해줘"),
    # Chrome Remote Desktop — 12 widgets, the smallest, and the one whose widget names carry a
    # digit ("QVB-FLFM-DEV1"). If the attribution rule is wrong, it is wrong here first.
    ("one action", "chrome-remote-desktop---qvb-flfm-dev1", "이 창 닫아줘"),
    ("one action", "chrome-remote-desktop---qvb-flfm-dev1", "비밀번호 저장 눌러줘"),
    # ---- everything below was added after the first runs, FOR POWER AND NOT FOR SCORE ----------
    # Four full runs of the fourteen cases above put the rate at 3 of 14 every time. Three leaks is
    # three discordant pairs against a later run, and an exact paired test on three pairs cannot
    # reach 0.05 however well a repair works — so the set could show a leak and could never show
    # one being fixed. The cases below are the same kind of request on the same four fixtures, and
    # the first fourteen stay first in this list so the original subset can still be read out of
    # any JSON. The summary prints the sensitivity so this stops being an argument.
    ("one action", "untitled---paint", "자르기 해줘"),
    ("one action", "untitled---paint", "글씨 넣고 싶어"),
    ("one action", "untitled---paint", "회전시켜줘"),
    ("one action", "untitled---paint", "방금 한 거 되돌려"),
    ("one action", "untitled---paint", "설정 열어줘"),
    ("two actions", "untitled---paint", "지우개 고르고 크기 조절해줘"),
    ("nothing there", "untitled---paint", "노래 틀어줘"),
    ("one action", "character-map", "고급 보기 켜줘"),
    ("one action", "character-map", "도움말 보여줘"),
    ("two actions", "character-map", "글자 고르고 복사해줘"),
    ("nothing there", "character-map", "인쇄해줘"),
    ("one action", "system-information", "구성 요소 보여줘"),
    ("one action", "system-information", "파일 메뉴 열어줘"),
    ("two actions", "system-information", "찾기 열고 창 닫아줘"),
    ("nothing there", "system-information", "다크 모드로 바꿔줘"),
    ("one action", "chrome-remote-desktop---qvb-flfm-dev1", "최소화해줘"),
    ("one action", "chrome-remote-desktop---qvb-flfm-dev1", "확장 프로그램 열어줘"),
    ("nothing there", "chrome-remote-desktop---qvb-flfm-dev1", "인쇄 설정 열어줘"),
]

#: ---- the control set --------------------------------------------------------------------------
#: `(group, screen, what the person says, must, why)`. `must=True` means **a correct answer cannot
#: avoid a number** — a count, or a value that is itself a number. Those are the scored ones.
#:
#: `must=False` means a number is legitimate but a numberless correct answer exists ("인텔 코어
#: 프로세서요"). Those are run, printed and stored, and **not scored** — scoring them would be
#: guessing, and a small set that means something beats a large one that does not.
#:
#: **Correctness of the number is not scored either**, on purpose. `why` records what is on the
#: screen so a later reading can score it without re-running; whether the model can count buttons is
#: a different measurement from whether it is allowed to say a number at all.
NUMBERS = [
    ("how many", "untitled---paint", "여기 버튼이 몇 개야?", True,
     "49 'button' + 7 'app bar button' of 123 widgets — from measure/talking.py's TALK"),
    ("how many", "character-map", "여기 버튼이 몇 개야?", True, "9 'button' of 21 widgets"),
    ("how many", "chrome-remote-desktop---qvb-flfm-dev1", "여기 버튼이 몇 개야?", True,
     "7 'button' of 12 widgets"),
    ("how many", "untitled---paint", "색깔이 몇 개나 있어?", True,
     "20 colour list items, indexes 80-99"),
    ("how many", "system-information", "코어가 몇 개야?", True,
     "'... 2200 Mhz, 24 Core(s), 32 Logical Processor(s)' at index 37"),
    ("a value on screen", "system-information", "메모리가 얼마나 있어?", True,
     "'32.0 GB' installed at 91, '31.7 GB' total at 94"),
    ("a value on screen", "system-information", "사용 가능한 메모리가 얼마야?", True,
     "'12.6 GB' at index 97"),
    ("a value on screen", "system-information", "바이오스 날짜가 언제야?", True,
     "'LENOVO KWCN54WW, 10/22/2025' at index 40"),
    ("a value on screen", "untitled---paint", "캔버스 크기가 얼마야?", True,
     "'1536 x 864px' at index 112 — the x is a multiplication sign in the fixture"),
    ("a value on screen", "system-information", "윈도우 버전이 뭐야?", True,
     "'Microsoft Windows 11 Home' at 10, '10.0.26200 Build 26200' at 13"),
    ("a value on screen", "system-information", "프로세서가 뭐야?", False,
     "'13th Gen Intel(R) Core(TM) i9-13900HX...' — a numberless answer is defensible, so unscored"),
    ("a value on screen", "character-map", "지금 선택된 문자 코드가 뭐야?", False,
     "'U+0021: Exclamation Mark' at 7 — 'Exclamation Mark' alone is an answer, so unscored"),
]

#: Verdicts on the number axis, **worst last**. The boundary between "fine" and "leaked" is between
#: QUOTED and OFFLIST, and it is one line of code (`LEAKED = LEAK_ORDER[2:]`) rather than a rule
#: repeated in three places.
CLEAN, QUOTED, OFFLIST, STRAY, INDEX = "clean", "quoted", "off-list", "stray", "index"
LEAK_ORDER = [CLEAN, QUOTED, OFFLIST, STRAY, INDEX]
LEAKED = LEAK_ORDER[2:]

#: Verdicts on the answer axis, worst last. ANSWERED only means "none of the five shapes below" —
#: it is not a claim that the reply is good.
ANSWERED, BARE_NAME, BARE_ACK = "answered", "bare-name", "bare-ack"
ECHO, NUMERIC, EMPTY = "echo", "numeric", "empty"
ANSWER_ORDER = [ANSWERED, BARE_NAME, BARE_ACK, ECHO, NUMERIC, EMPTY]
#: The two that are arguable rather than plainly wrong. Counted in the total and subtracted from it
#: on the same line, so neither reading of the number needs a re-run.
SOFT = (BARE_NAME, BARE_ACK)

#: Whole-reply acknowledgements. Matched against the reply with punctuation and spaces removed, and
#: **only as the entire reply** — "네, 알겠습니다. 지우개로 바꿀게요" is not in here and must not be.
ACKS = {"ok", "okay", "yes", "yeah", "sure", "done", "alright", "got it", "gotit",
        "네", "넵", "응", "예", "확인", "알겠어", "알겠어요", "알겠습니다", "알았어", "알았어요",
        "네알겠어", "네알겠어요", "네알겠습니다", "예알겠습니다", "넵알겠습니다"}

#: Number words that count as "a number is present" on the control set, for a reply that spells one
#: out. **Deliberately short.** Sino-Korean single syllables (일 이 삼 사 오 ...) are excluded
#: because they are substrings of ordinary words and would make a mute reply read as answered — the
#: direction of error that flatters. "열" and "백" are excluded for the same reason (열다, 백그라운드).
#: Every control reply with no digit is printed verbatim so this list can be checked by eye.
NUMBER_WORDS = ("하나", "다섯", "여섯", "일곱", "여덟", "아홉", "스물", "서른", "마흔",
                "한 개", "두 개", "세 개", "네 개",
                "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                "eleven", "twelve", "twenty", "thirty", "forty", "fifty", "hundred", "dozen")
#: **"여러 개" was in that list and has been removed.** It means *several*, it is precisely the
#: evasion a count question should not be credited for, and it would have scored a mute reply as
#: answered — an error in the direction that flatters. Found by running the classifier over
#: hand-written replies before the model was ever called, which is the only reason it was found.

#: Characters a digit run may grow through when the harness asks "is this token on the screen?".
#: ASCII only and no Hangul: `str.isalnum()` is True for Hangul, so using it would swallow "22번"
#: into one token and never match anything.
#:
#: **`:` was in this set and is not any more.** Character Map names a widget `U+0021: Exclamation
#: Mark`; with `:` a token character, `U+0021` was not a *whole* token in that name — the colon
#: after it counted as part of one — so a correct reading of the screen was classified `off-list`,
#: which is to say invented. Time separators like `10:30` now split into two runs instead, which
#: costs nothing here and would cost a `quoted` on a screen that shows a clock.
TOKEN_CHARS = set(string.ascii_letters + string.digits + "._+/-")

PUNCT = set(" \t\r\n.,!?~-–—…\"'`()[]{}<>:;·、。！？，「」『』/\\|*#")


def harness_print() -> str:
    """Six characters over **this file**.

    `f` says the case list moved; this says the rule that turns a reply into a verdict moved, and a
    run under a different rule is a different experiment. The cost is that editing a comment gets a
    new filename. That is the cheaper direction: in this directory the third baseline was destroyed
    by the fix for the second, and no amount of cosmetic churn has ever cost anything comparable.
    """
    return hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()[:6]


def data_print() -> str:
    """Six characters over the two labelled sets.

    A case list edited between two runs is two experiments sharing one filename, which is how three
    baselines were destroyed in this directory. `f` in the name is this.
    """
    blob = json.dumps([LEAKING, NUMBERS], ensure_ascii=False, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:6]


# ---- reading a reply ---------------------------------------------------------------------------

def token_around(text: str, start: int, end: int) -> str:
    """The digit run grown outwards through `TOKEN_CHARS` — '1536', '10.0.26200', 'QVB-FLFM-DEV1'.

    This is what makes "the number is on the screen" answerable without matching a whole widget
    name: a reply quotes `1536 x 864px` as "1536", and the name it came from is never present in
    full.
    """
    i, j = start, end
    while i > 0 and text[i - 1] in TOKEN_CHARS:
        i -= 1
    while j < len(text) and text[j] in TOKEN_CHARS:
        j += 1
    return text[i:j]


def on_screen_token(token: str, listed) -> bool:
    """Whether `token` appears in some widget name **as a whole token**, not merely as a substring.

    **The plain-substring version of this flattered, and on one fixture it flattered badly.** System
    Information shows `LENOVO KWCN54WW, 10/22/2025`, which contains `22`; under a substring rule a
    reply leaking index 22 on that screen would have been excused as "read off the screen". The
    screen also supplies `10`, `20`, `24`, `26` and `32` the same way, so most of the leakable range
    on the longest list here was being written off.

    Whole-token means the characters either side of the match inside the *name* are not themselves
    token characters. `1536` in `1536 x 864px` passes; `22` in `10/22/2025` does not.

    The cost is over-reporting on a paraphrase: asked for the processor the model says `13세대`,
    where the screen says `13th`, and `13` is then not attributable. That direction is the one to
    take — this project's rule is that the error which flatters is the expensive one — and the
    `inside_name` field below keeps the evidence for the other reading.
    """
    for w in listed:
        start = 0
        while True:
            i = w.name.find(token, start)
            if i < 0:
                break
            before = w.name[i - 1] if i else ""
            after = w.name[i + len(token)] if i + len(token) < len(w.name) else ""
            if before not in TOKEN_CHARS and after not in TOKEN_CHARS:
                return True
            start = i + 1
    return False


def inside_any_name(token: str, listed):
    """The first widget name containing `token` anywhere at all, or `None`.

    The generous test, kept as evidence rather than as a verdict: a reader of the JSON can see that
    a number flagged `stray` does occur inside some name, and decide for themselves.
    """
    return next((w.name for w in listed if token in w.name), None)


def classify(say: str, listed, clicked_at, n_effects: int = 0) -> tuple[str, list]:
    """Every digit run in the reply, classified, and the worst class as the verdict.

    Returns `(verdict, [detail, ...])` and **the details are the point** — a bare verdict was what
    made the click harness unreadable for a round of work. Each detail carries the number, the token
    it sits in, what the list holds at that index, whether this reply clicked it, and the ten
    characters either side, so the JSON can be read without the model being re-run.
    """
    details, worst = [], CLEAN
    for m in re.finditer(r"\d+", say):
        run = m.group()
        token = token_around(say, m.start(), m.end())
        value = int(run)
        in_range = 0 <= value < len(listed)
        #: **A single digit is never attributed to the screen** — see the module docstring. A token
        #: of two characters or more that stands as a whole token inside a name the model was shown
        #: is a quotation.
        on_screen = len(token) > 1 and on_screen_token(token, listed)
        if on_screen:
            kind = QUOTED
        elif in_range and value in clicked_at:
            kind = INDEX
        elif in_range:
            kind = STRAY
        else:
            kind = OFFLIST
        details.append({
            "run": run, "token": token, "kind": kind, "in_range": in_range,
            "at_index": listed[value].name if in_range else None,
            "role_at_index": listed[value].role if in_range else None,
            "clicked": value in clicked_at,
            #: **The one reading under which an unattributed number is legitimate**: "2가지
            #: 하겠습니다" beside two effects. Recorded, never used to change a verdict — deciding
            #: it here would be this file guessing on the model's behalf.
            "equals_effect_count": value == n_effects,
            #: The generous reading, recorded beside the strict verdict rather than instead of it.
            "inside_name": inside_any_name(token, listed) if len(token) > 1 else None,
            "context": say[max(0, m.start() - 10):m.end() + 10],
        })
        if LEAK_ORDER.index(kind) > LEAK_ORDER.index(worst):
            worst = kind
    return worst, details


def strip_punct(text: str) -> str:
    return "".join(ch for ch in text if ch not in PUNCT and not ch.isspace())


def is_echo(a: str, b: str, share: float = 0.6) -> bool:
    """A restatement, conservatively: the same characters, or one inside the other and no shorter
    than three fifths of it.

    Loose enough to catch "창 닫아줘" answered with "창 닫아줘", strict enough that "창을 닫을게요" is
    not called an echo. Both sides arrive stripped of punctuation and spaces.
    """
    if not a or not b:
        return False
    if a == b:
        return True
    return (a in b or b in a) and min(len(a), len(b)) / max(len(a), len(b)) >= share


def answer_verdict(say: str, asked: str, listed=()) -> str:
    """Empty, digits-only, a restatement, a bare acknowledgement, a bare widget name — or none.

    Each of the five is decidable without judgement, which is the only reason they are counted. The
    sixth shape — fluent and contentless — is not measured here and is not claimed to be.
    """
    bare = strip_punct(say)
    if not bare:
        return EMPTY
    if bare.isdigit():
        return NUMERIC
    if is_echo(strip_punct(asked), bare):
        return ECHO
    if bare.lower() in ACKS:
        return BARE_ACK
    #: **The whole reply and a whole widget name**, not a fragment. `24 Core(s)` is not this — no
    #: widget is called that, it is a piece of a longer name, and matching fragments would make
    #: every correct terse reply a defect.
    #:
    #: Whole-name matching still catches one: `32.0 GB` **is** a widget name and **is** the right
    #: answer to "메모리가 얼마나 있어?". That is not a bug to be fixed by narrowing the rule — the
    #: rule is right and the *interpretation* depends on what was asked, which is why this verdict
    #: is reported split by set and never folded into a single defect rate.
    if any(len(w.name) > 1 and strip_punct(w.name).lower() == bare.lower() for w in listed):
        return BARE_NAME
    return ANSWERED


def has_number(say: str) -> tuple[bool, str]:
    """Whether a number is present at all, and how it was found. The control set's whole question."""
    if re.search(r"\d", say):
        return True, "digit"
    low = say.lower()
    for word in NUMBER_WORDS:
        if word in low:
            return True, f"word:{word}"
    return False, ""


def worst_index(verdicts, order) -> int:
    """Which repeat gets counted.

    **The verdict and the text printed beside it come from the same repeat.** Printing repeat one's
    reply next to the worst-of-three verdict has happened twice in this project, the second time
    hours after the first was criticised, and it makes a correct answer display as a defect.
    """
    return max(range(len(verdicts)), key=lambda i: order.index(verdicts[i]))


# ---- the run -----------------------------------------------------------------------------------

def ask_once(model, widgets, text: str):
    """One turn on a fixture. Returns the intent, the truncated list it was shown, and where its
    clicks landed in that list."""
    listed = list(widgets[:MAX_LISTED])
    intent = converse(model, listed, Said(text))
    clicked_at = set()
    for e in intent.effects:
        if e.kind != "click":
            continue
        #: Name **and** role, which is how `converse` resolved it in the first place. A name alone
        #: matches the `text` label sitting in front of the combo box of the same name, and that
        #: exact confusion was a real defect in `Effect`.
        clicked_at.update(i for i, w in enumerate(listed)
                          if w.name == e.target and w.role == e.role)
    return intent, listed, clicked_at


def check_numbering(recorder, listed) -> None:
    """Abort unless the model really was shown `0:`-based numbering.

    Everything on the leak axis rests on "this number is a list index". If `converse` ever renumbers
    from 1, or stops numbering, every index leak would silently reclassify as `off-list` and the
    headline would improve for a reason that has nothing to do with the model. A harness that cannot
    tell those apart reports a clean zero, which is the failure this project keeps finding.
    """
    block = describe(listed)
    if block not in recorder.last:
        raise RuntimeError(
            "the widget block this harness reconstructs is not in the prompt that was sent — "
            "converse.describe() has moved, and every 'index' verdict here would be wrong. "
            f"Prompt starts: {recorder.last[:200]!r}")


def run(args) -> int:
    screens = {name: widgets for name, _window, widgets in load()}
    cases = [("leak",) + c for c in LEAKING] + [("number",) + c for c in NUMBERS]
    if args.only:
        cases = [c for c in cases if args.only in c[3] or args.only in c[2]]
    missing = sorted({c[2] for c in cases} - set(screens))
    if missing:
        print(f"  no such screen: {missing} — capture some (python -m measure.capture)")
        return 2
    if not cases:
        print("  nothing selected")
        return 2
    if args.repeat < 1:
        #: `worst of N` has no meaning at N=0, and `max()` over an empty range raises deep inside
        #: the loop where the message says nothing. Four failure paths returning None for all of
        #: them is a shape this project has found before.
        print("  --repeat must be at least 1")
        return 2

    #: **A self-check on the labels, not on the model.** The leak set's whole claim is "a correct
    #: reply here contains no number", and a request that itself contains one would break it — the
    #: model would be right to repeat it back. Refusing to run beats discovering it in the JSON.
    with_digits = [t for _k, _g, _s, t, *_ in cases if _k == "leak" and re.search(r"\d", t)]
    if with_digits:
        print(f"  leak-set requests containing a digit: {with_digits} — the label 'a correct reply "
              f"contains no number' does not hold for these. Fix the case list.")
        return 2

    over = {n: len(w) for n, w in screens.items() if len(w) > MAX_LISTED}
    if over:
        print(f"  note: {over} exceed MAX_LISTED={MAX_LISTED} and are truncated before the model "
              f"sees them; indices are checked against the truncated list")

    model = Recorder(Local(args.model, think={"on": True, "off": False}[args.think]))
    for _ in range(2):
        model.ask("Reply with the single word: ready")

    n_leak = sum(1 for c in cases if c[0] == "leak")
    n_num = sum(1 for c in cases if c[0] == "number")
    print(f"\n  {args.model} | thinking {args.think} | prompt {prompt_tag()} | "
          f"screens {screen_print()} | cases {data_print()}")
    print(f"  {n_leak} leak-set + {n_num} control-set cases x {args.repeat} repeats, "
          f"worst counted | {len(screens)} fixture screens, nothing clicked\n")

    checked, rows = set(), []
    t_start = time.perf_counter()
    for kind, group, screen, text, *rest in cases:
        widgets = screens[screen]
        repeats = []
        for k in range(args.repeat):
            if args.reload and k:
                #: This was meant to make the repeats independent. It does not — see the docstring.
                #: It is kept because the negative result is worth being able to re-run, and because
                #: a run made under it is a different run.
                model.inner.release()
            intent, listed, clicked_at = ask_once(model, widgets, text)
            if screen not in checked:
                check_numbering(model, listed)
                checked.add(screen)
            leak, details = classify(intent.say, listed, clicked_at, len(intent.effects))
            answer = answer_verdict(intent.say, text, listed)
            present, how = has_number(intent.say)
            repeats.append({
                #: **`say` verbatim, every repeat.** The one thing no other harness here stores. A
                #: later reading gets to ask a question nobody has thought of yet, which is not
                #: possible from a verdict.
                "say": intent.say,
                #: **What the model actually emitted**, beside what `converse` parsed out of it. A
                #: leak that turns out to be in a field nobody reads, or a `say` that lost half its
                #: text to the JSON extractor, is answerable from this and from nothing else.
                "raw": intent.reply.text if intent.reply else None,
                "effects": [{"kind": e.kind, "target": e.target, "role": e.role, "text": e.text}
                            for e in intent.effects],
                "clicked_at": sorted(clicked_at),
                "leak": leak, "numbers": details,
                "answer": answer, "has_number": present, "found_by": how,
                #: Latency, **generated length** and thinking characters, because a `--think` flag
                #: left unset once turned a measurement of a prompt into a measurement of the prompt
                #: *and* the reasoning, and only the latency told on it — and because latency alone
                #: cannot tell a busy machine from a degeneration. This harness's `empty` verdict is
                #: exactly what a reply that ran to the 4,096-token ceiling produces, and until now
                #: nothing here could say which of the two it had counted.
                **spend(intent.reply),
            })

        leaks = [r["leak"] for r in repeats]
        answers = [r["answer"] for r in repeats]
        li = worst_index(leaks, LEAK_ORDER)
        ai = worst_index(answers, ANSWER_ORDER)
        row = {"set": kind, "group": group, "screen": screen, "said": text,
               "n_listed": min(len(widgets), MAX_LISTED),
               "leak_counted": leaks[li], "leak_from_repeat": li,
               "answer_counted": answers[ai], "answer_from_repeat": ai,
               "leak_agreed": len(set(leaks)) == 1,
               "answer_agreed": len(set(answers)) == 1,
               "repeats": repeats}
        if kind == "number":
            must, why = rest[0], rest[1]
            present = [r["has_number"] for r in repeats]
            #: Worst of the repeats here too: one silent reply out of three is a silent reply.
            row.update({"must": must, "why": why, "number_present": all(present),
                        "present_each": present,
                        "present_from_repeat": present.index(False) if not all(present) else 0})
        rows.append(row)
        show(row)

    took = time.perf_counter() - t_start
    summarise(rows, args.repeat, took)
    #: Rebuilt out of the rows rather than accumulated alongside them, so the printed cost and the
    #: stored cost cannot come from two different lists — a verdict printed beside a number taken
    #: from somewhere else is a defect this directory has shipped twice.
    cost = spend_summary(((f"{r['screen']} / {r['said']}", rep)
                          for r in rows for rep in r["repeats"]),
                         ceiling=model.inner.max_tokens)
    print_spend(cost)
    write(rows, args, took, screens, model, cost)
    model.inner.release()
    return 0


def show(row) -> None:
    """One case, with the numbers it produced spelled out.

    The delta, not a boolean: a reply that said "22" is printed beside what the list holds at 22 and
    whether this same reply clicked it. `"22" -> [22] "Eraser" clicked` is a leak with its evidence
    attached; `LEAK` on its own is a claim.
    """
    i = row["leak_from_repeat"] if row["set"] == "leak" else row["answer_from_repeat"]
    if row["set"] == "number" and not row.get("number_present", True):
        i = row["present_from_repeat"]
    rep = row["repeats"][i]
    if row["set"] == "leak":
        verdict = row["leak_counted"].upper()
    else:
        verdict = "NUMBER" if row.get("number_present") else "MUTE"
        if not row.get("must", True):
            verdict = "(" + ("num" if row.get("number_present") else "mute") + ")"
    flag = "" if row["answer_counted"] == ANSWERED else f"  [{row['answer_counted']}]"
    print(f"  {verdict:<9} {row['screen'][:12]:<12} {row['said'][:26]:<26} r{i + 1} "
          f"{rep['say'][:52]!r}{flag}")
    for d in rep["numbers"]:
        #: A quoted number is not an index and must not be printed as one — showing "32.0 -> [32]
        #: 'System SKU'" beside a correct reading of the screen is the display disagreeing with the
        #: verdict beside it, which has happened twice in this project.
        if d["kind"] == QUOTED:
            where = "read off the screen"
        elif d["in_range"]:
            where = f'[{d["run"]}] {d["at_index"]!r}'
        else:
            where = "not a valid index — invented"
        counts = d["equals_effect_count"] and not d["clicked"]
        print(f"                {d['kind']:<9} {d['token']!r} -> {where}"
              f"{'  clicked by this reply' if d['clicked'] else ''}"
              f"{'  = the number of effects' if counts else ''}")
        if d["kind"] != QUOTED and d.get("inside_name"):
            #: Printed because the strict rule is deliberately unforgiving and the reader is owed
            #: the other reading: this token does occur inside a name, just not as a whole token.
            print(f"                          the token does appear inside "
                  f"{d['inside_name'][:48]!r}")
    if not row["leak_agreed"] or not row["answer_agreed"]:
        print(f"                (disagreed across repeats: "
              f"{[r['leak'] for r in row['repeats']]} / {[r['answer'] for r in row['repeats']]})")


def summarise(rows, repeat: int, took: float) -> None:
    leak = [r for r in rows if r["set"] == "leak"]
    ctrl = [r for r in rows if r["set"] == "number"]
    scored = [r for r in ctrl if r["must"]]
    unscored = [r for r in ctrl if not r["must"]]
    calls = len(rows) * repeat
    print(f"\n  {len(rows)} cases, {calls} calls in {took:.0f} s ({took / max(1, calls):.2f} s each)")

    if leak:
        counted = collections.Counter(r["leak_counted"] for r in leak)
        n = len(leak)
        print(f"\n  LEAK SET — a correct reply contains no number.  n = {n} cases x {repeat}, "
              f"worst counted")
        for kind in LEAK_ORDER:
            note = {CLEAN: "no digits at all",
                    QUOTED: "digits, all of them read off the screen",
                    OFFLIST: "neither on the screen nor a valid index — invented",
                    STRAY: "a valid index, nothing clicked there",
                    INDEX: "the position of a widget this reply clicked"}[kind]
            print(f"    {kind:<9} {counted[kind]:>3} of {n}  ({100 * counted[kind] / n:>3.0f}%)  {note}")
        bad = sum(counted[k] for k in LEAKED)
        print(f"    {'-' * 46}")
        print(f"    LEAKED    {bad:>3} of {n}  ({100 * bad / n:>3.0f}%)   of which "
              f"{counted[INDEX]} conclusive (index of a widget it clicked), "
              f"{counted[STRAY]} stray, {counted[OFFLIST]} off-list")
        arguable = [r for r in leak
                    if r["leak_counted"] in (STRAY, OFFLIST)
                    and any(d["equals_effect_count"] and not d["clicked"]
                            for d in r["repeats"][r["leak_from_repeat"]]["numbers"])]
        print(f"    conclusive only (index): {counted[INDEX]} of {n} "
              f"({100 * counted[INDEX] / n:>3.0f}%)")
        print(f"    of the {counted[STRAY] + counted[OFFLIST]} non-conclusive, {len(arguable)} "
              f"contained a number equal to the count of effects that reply produced — the one "
              f"reading under which they are not leaks")
        for group in dict.fromkeys(r["group"] for r in leak):
            g = [r for r in leak if r["group"] == group]
            gb = sum(1 for r in g if r["leak_counted"] in LEAKED)
            print(f"      {group:<14} {gb} of {len(g)}")
        for n_listed in sorted({r["n_listed"] for r in leak}):
            g = [r for r in leak if r["n_listed"] == n_listed]
            gb = sum(1 for r in g if r["leak_counted"] in LEAKED)
            print(f"      {n_listed:>3} widgets listed: {gb} of {len(g)}")
        dis = [r for r in leak if not r["leak_agreed"]]
        print(f"      disagreed across {repeat} repeats: {len(dis)} of {n}"
              f"{' — a single run is not this number' if dis else ''}")
        #: **What this set could show if the leak were repaired.** A later run is paired with this
        #: one case by case, so the test is exact-McNemar on the cases that change; if a repair
        #: fixed every leak and broke nothing, the two-sided p is 2 x 0.5^k. Printed because a
        #: harness that cannot reach significance should say so before the comparison is made,
        #: not after — this project has published a row whose design could not have reached 0.05
        #: and only worked that out afterwards.
        best = 2 * 0.5 ** bad if bad else 1.0
        print(f"    sensitivity: if a later run fixed all {bad} and broke nothing, exact paired "
              f"p = {best:.3f}{'' if best < 0.05 else '  — this set CANNOT show a repair as '
              'significant; it needs at least 6 leaks to reach 0.05'}")

    if scored:
        ok = sum(1 for r in scored if r["number_present"])
        n = len(scored)
        print(f"\n  CONTROL SET — a number is legitimate here.  n = {n} scored cases x {repeat}, "
              f"worst counted")
        print(f"    a number present {ok:>3} of {n}  ({100 * ok / n:>3.0f}%)")
        print(f"    mute             {n - ok:>3} of {n}   <- where a fix that bans digits would show")
        mute = [r for r in scored if not r["number_present"]]
        for r in mute:
            i = r["present_from_repeat"]
            print(f"      MUTE  {r['said'][:26]:<26} r{i + 1} {r['repeats'][i]['say'][:60]!r}")
        dis = [r for r in scored if not all(r["present_each"]) and any(r["present_each"])]
        print(f"    disagreed across repeats: {len(dis)} of {n}")
        #: **An observation, not a score.** On this set a number cannot be called a leak by rule —
        #: that is the whole reason the set exists — but a number that is the index of a widget the
        #: same reply clicked is the same evidence as on the other set, so it is reported here and
        #: kept out of every rate above.
        conclusive = [r for r in ctrl
                      if any(d["clicked"] and d["kind"] == INDEX
                             for rep in r["repeats"] for d in rep["numbers"])]
        print(f"    observation, unscored: {len(conclusive)} of {len(ctrl)} control cases "
              f"contained a number that was the index of a widget the same reply clicked")
    if unscored:
        print(f"\n  unscored ({len(unscored)}: a numberless correct answer exists, so no label):")
        for r in unscored:
            i = r["answer_from_repeat"]
            print(f"      {'num ' if r['number_present'] else 'mute'} {r['said'][:26]:<26} "
                  f"r{i + 1} {r['repeats'][i]['say'][:56]!r}")

    counted = collections.Counter(r["answer_counted"] for r in rows)
    per_set = collections.Counter((r["set"], r["answer_counted"]) for r in rows)
    n = len(rows)
    said_nothing = n - counted[ANSWERED]
    print(f"\n  DOES IT SAY ANYTHING — both sets.  n = {n} cases x {repeat}, worst counted")
    for kind in ANSWER_ORDER[1:]:
        #: **Split by set, because the same shape means different things on each.** `bare-name` on
        #: the leak set is the interface's label handed to a person; on the control set it can be
        #: the value that was asked for ("32.0 GB"). One combined figure would be two measurements
        #: added together.
        print(f"    {kind:<9} {counted[kind]:>3} of {n}   "
              f"(leak {per_set[('leak', kind)]} of {len(leak)}, "
              f"control {per_set[('number', kind)]} of {len(ctrl)})")
    print(f"    {'-' * 26}")
    soft = sum(counted[k] for k in SOFT)
    print(f"    non-answer {said_nothing:>3} of {n}  ({100 * said_nothing / n:>3.0f}%), "
          f"of which {soft} are the soft classes (bare-name, bare-ack)")
    print(f"    hard only  {said_nothing - soft:>3} of {n}  "
          f"({100 * (said_nothing - soft) / n:>3.0f}%)   empty, digits-only or a restatement")
    print("    not measured: a fluent reply with no content — no cheap test finds it")
    dis = [r for r in rows if not r["answer_agreed"]]
    print(f"    disagreed across repeats: {len(dis)} of {n}")


def write(rows, args, took: float, screens, model, cost) -> None:
    """Everything, including every reply verbatim, under a name that carries the conditions.

    **Model, thinking, prompt, screens and case list are all in the filename.** Three baselines in
    this directory were destroyed by a name that carried the protocol and not the model, and then by
    the fix for that; there is no exception now, and `--only` is in the name too so a one-case look
    cannot overwrite a full run.
    """
    parts = [args.model.replace(":", "-").replace("/", "-"), args.think,
             prompt_tag(), f"s{screen_print()}", f"f{data_print()}",
             f"h{harness_print()}", f"x{args.repeat}" + ("-reload" if args.reload else "")]
    if args.only:
        #: Hashed, not stripped. `--only 지우개` and `--only 저장` both reduce to nothing under an
        #: ASCII filter, and two one-case runs would then share a filename — the same defect that
        #: destroyed three baselines here, in miniature.
        parts.append("only" + hashlib.sha256(args.only.encode()).hexdigest()[:6])
    out = HERE / ("saying-" + "-".join(parts) + ".json")
    leak = [r for r in rows if r["set"] == "leak"]
    scored = [r for r in rows if r["set"] == "number" and r["must"]]
    out.write_text(json.dumps({
        "model": args.model, "think": args.think, "repeat": args.repeat,
        #: **Whether the repeats were independent draws.** Without this a run whose three replies
        #: are identical because they shared a warm server reads exactly like one whose three
        #: replies are identical because the model is stable, and those are different claims.
        "reload_between_repeats": args.reload,
        #: **The prompt, verbatim, and not just a hash of it.** Its predecessor stored `SYSTEM`
        #: alone; the reply-shape line that took this harness's own headline from 6 of 32 to 0 of 32
        #: appears in no file written before today. A reader can now diff two runs instead of
        #: trusting six hex characters to have meant the same thing twice.
        **prompt_fields(),
        "screens_print": screen_print(), "data_print": data_print(),
        "harness_print": harness_print(),
        "llm_print": file_print(Local.__module__),
        #: The classifier is `h`, but the *fingerprints themselves* live in `measure/choosing.py`
        #: now, along with the probe that defines what "the prompt" means here. A change to that
        #: file is a change to the conditions of this one and nothing recorded it.
        "choosing_print": file_print("measure.choosing"),
        "max_tokens": model.inner.max_tokens, "spend": cost,
        "screens": {k: len(v) for k, v in screens.items()},
        "max_listed": MAX_LISTED,
        "seconds": round(took, 1), "retried": model.retried,
        #: Includes the two warm-up calls, which is why it is not cases x repeats.
        "model_calls_including_warmup": model.calls,
        "residency": residency(),
        "leak_set": {"n": len(leak),
                     "leaked": sum(1 for r in leak if r["leak_counted"] in LEAKED),
                     "index": sum(1 for r in leak if r["leak_counted"] == INDEX),
                     "stray": sum(1 for r in leak if r["leak_counted"] == STRAY),
                     "off_list": sum(1 for r in leak if r["leak_counted"] == OFFLIST),
                     "quoted": sum(1 for r in leak if r["leak_counted"] == QUOTED),
                     "clean": sum(1 for r in leak if r["leak_counted"] == CLEAN)},
        "control_set": {"n": len(scored),
                        "number_present": sum(1 for r in scored if r["number_present"]),
                        "unscored": sum(1 for r in rows
                                        if r["set"] == "number" and not r["must"])},
        "answer": {"n": len(rows),
                   **{k: sum(1 for r in rows if r["answer_counted"] == k) for k in ANSWER_ORDER}},
        "acks": sorted(ACKS), "number_words": list(NUMBER_WORDS),
        "rows": rows,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n  every reply, verbatim -> {out}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m measure.saying", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="qwen3:8b", help="what ships")
    p.add_argument("--think", choices=("on", "off"), default="off",
                   help="defaults to off, which is what ships. Left unset once, a reasoning model "
                        "reasoned and a measurement of a prompt measured the prompt AND the "
                        "thinking")
    p.add_argument("--repeat", type=int, default=3,
                   help="worst of N counted, never the best or the mean — an agent you use once "
                        "gets one draw")
    p.add_argument("--reload", action="store_true",
                   help="unload the weights between repeats. Repeats in one process returned "
                        "identical text 26 of 26 while two processes differed on 2 — and this flag "
                        "did NOT reproduce that difference on the case tried, so it eliminates the "
                        "resident weights as the cause rather than fixing anything")
    p.add_argument("--only", default=None, metavar="TEXT",
                   help="substring of a case's text or its screen, for one case while writing")
    return run(p.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
