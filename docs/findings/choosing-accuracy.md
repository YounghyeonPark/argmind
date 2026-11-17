# Does it pick the right thing? — step 4, measured 2026-09-09

Everything else about this design is measured. The environment answers in 41–266 ms, the model
answers in well under a second, a name survives a screen that moves under it. **None of that matters
if the thing it picks is the wrong thing**, and until today the only evidence was casual: two live
attempts, one right and one wrong — *"open the file menu"* selected the document.

This is the number that gates deleting anything (`REBUILD.md`, step 5).

## Conditions

| | |
|---|---|
| model | `qwen2.5:7b` via Ollama, `temperature 0`, `num_ctx 8192` |
| screens | 4 real windows captured from this desktop: 12, 21, 123 and 152 named widgets (308 total) |
| goals | 37, written and labelled **before anything ran**; 9 of them have no right answer |
| repeats | 3 per goal; the **worst** of the three is what is counted |
| what is measured | choosing, and only choosing — the screens are fixtures, so nothing here can fail because a window was not in front or a click missed |

`measure/goals.json` holds the goals and their accepted answers, `measure/screens/` the captured
windows, `measure/choosing*.json` every individual answer. The Chrome captures that came with the
first sweep were **deleted rather than checked in**: they contained the owner's search history and
email address. `system-information.json` still carries the machine name and user name, which is the
one privacy cost of keeping a real fixture, and it can be dropped without changing the shape of the
result.

**Four outcomes, not two.**

|  | it clicked something | it clicked nothing |
|---|---|---|
| a right answer exists | RIGHT or WRONG | MISSED |
| no right answer exists | **WRONG** — the bad one | DECLINED — correct |

Nine goals with no right answer are there so the measurement can produce a negative result. Without
them an agent that clicks something every time scores the same as one that chooses well.

## The result

**18 of 28 answerable goals right (64%), and 8 of 9 correct declines.** Per-run, scored separately:
19 / 18 / 18 right — one goal out of 37 disagreed across the three passes, so this is not a
single-draw number.

| protocol | right / 28 | wrong | *of those, on a screen with no right answer* | missed | declined / 9 |
|---|---|---|---|---|---|
| **index, interface order** (what is built) | **18** | 8 | 1 | 3 | 8 |
| index, order shuffled (seed 7) | 15 | 13 | 2 | 2 | 7 |
| name instead of number | 15 | 3 | 2 | 12 | 7 |

Every wrong answer, in the shipped protocol:

    지우개로 바꿔줘            -> Fill                     (Eraser)
    확대해줘                -> Fit to window            (Zoom in)
    배경 지워줘              -> Share                    (Remove background)
    이미지 크기 바꿔줘          -> Crop                    (Resize and skew)
    창 닫아                -> Maximize                 (Close)
    make the brush bigger -> Vertical Small Increase  (Size)
    찾기 눌러줘              -> \Device\HarddiskVolume1  (Find)
    이 그림을 메일로 보내줘       -> Share                    (nothing — it should have declined)

## The hypothesis I had, and the measurement that refuted it

Six of the seven wrong answers with a right answer available sat **1–10 places** from a correct one
and five within four: *"창 닫아"* chose index 121 where `Close` is 122. That looked exactly like the
vision model's failure in a new guise — it can see the answer and cannot compute the number for it —
and it pointed at a fix I would have liked, since asking for a name instead of an index is a small
change.

**Shuffling each screen's order refutes it.**

| | median gap to the nearest right answer | as a fraction of the list | within 5 places |
|---|---|---|---|
| interface order | 4 | 0.033 | 15 of 21 |
| shuffled | 33 | 0.268 | 3 of 28 |

If the cause were arithmetic, the errors would still land near the target's *new* position. They do
not — the gaps expand to roughly what picking at random gives. **The small gaps in interface order
are semantic proximity, not counting error**: a toolbar puts `Fill` next to `Eraser` and `Maximize`
next to `Close`, so an error onto a related control is also an error onto a neighbouring index. The
two were confounded and the shuffle separated them.

That correction favoured me and it does not survive. Recorded because it is the fourth time on this
project that an appealing mechanism explained a pattern that a control refuted.

## Two things the shuffle established that were not the question

**The interface's own order carries information.** Shuffling costs 18 → 15 right and drives wrong
from 8 to 13. So the widget list must be presented in the order the environment gives it —
re-sorting it alphabetically, or by size, or by anything else, is not a neutral change.

**Asking for a name instead of a number is safer and less capable.** Wrong falls 8 → 3; missed rises
3 → 12; right falls 18 → 15. Of the misses, most are genuine `null` answers rather than a broken
match — only three distinct goals produced a name that was not in the list, and one of those was
`Find what` for a widget called `Find what:`, which is exact matching being brittle about a colon.
Nothing was changed on this basis: it is a real trade, not an improvement, and which side is right
depends on whether the agent is about to click.

## Density: a direction, not a rate

| widgets | screen | right | wrong | missed | declined |
|---|---|---|---|---|---|
| 12 | Chrome Remote Desktop | 3 / 4 | 0 | 1 | 2 / 2 |
| 21 | Character Map | 5 / 5 | 0 | 0 | 2 / 2 |
| 123 | Paint | 7 / 13 | 7 | 0 | 2 / 3 |
| 152 | System Information | 3 / 6 | 1 | 2 | 2 / 2 |

Small screens 8 of 9, large screens 10 of 19 — **one-sided Fisher exact p = 0.070, n = 28.** That is
not significant, and the two small screens are also the two *simpler applications*, so density and
complexity are confounded here. The direction is worth acting on only to the extent of not claiming
the 64% transfers to a dense professional interface; establishing it needs more screens at each size.

## What this means for the design

**It is not good enough to click unsupervised, and it is good enough to propose.** Eight wrong
answers in 37 goals, one of them on a screen where the right response was to do nothing, is a rate
that would produce a bad click most working hours. It is also plainly not random — 18 right and 8 of
9 correct declines is a long way from chance on lists of 12 to 152.

That is the same conclusion the permission design reached for a different reason. `Session` refuses
to act on its own judgement by default and turns an unprompted effect into a sentence instead; that
default was chosen because an agent acting unwatched has no undo, and this measurement says the
choosing is not accurate enough to justify relaxing it. **The safety argument and the accuracy
measurement agree**, which is worth noting precisely because they could have disagreed.

## The prompt is at a local optimum, and two attempts to improve it were withdrawn (2026-09-09)

Building the PC app exposed a real defect: the agent sometimes answers a Korean question in English
— *"Sure."* — even though `SYSTEM` already says to reply "in the language they used". Two fixes were
tried and **both were withdrawn, because both cost accuracy.**

|  | right / 28 | wrong | *on a screen with no right answer* | missed | declined / 9 |
|---|---|---|---|---|---|
| as it stands | **18** | 8 | 1 | 3 | **8** |
| a stronger rule in the first paragraph | 19 | 15 | **6** | 0 | **3** |
| the same rule as a separate line after the refusal | 17 | 12 | 1 | 0 | 8 |

**The first attempt looked like an improvement and was the worse of the two.** Right went *up* by
one, which is the only column a careless reading checks — and declines collapsed from 8 of 9 to 3 of
9, with wrong clicks on screens that had no right answer going from 1 to 6. Emphasising a rule about
language in the paragraph that also carries *"a wrong click is worse than an honest I can't see that
here"* diluted the refusal.

**The second attempt kept the refusals and converted every miss into a wrong answer.** Declines held
at 8 of 9, but `missed` went 3 → 0 and `wrong` went 8 → 12: on screens where an answer existed the
model stopped declining and started guessing.

So **any imperative added to the system prompt pushed the model toward acting**, in both placements
and in both directions of emphasis. The cause is not established — what is established is the
effect, three times over, since the baseline was re-measured after each revert and reproduced
exactly (18 / 8 / 1 / 3 / 8-of-9 each time).

**Nothing was changed.** The agent still occasionally replies in English and that is a real defect,
recorded rather than traded for a worse click. The operational rule this leaves: **a prompt edit is
a change to a measured system and has to be re-measured** — `python -m measure.choosing` takes 50
seconds, and both of these looked obviously harmless.

## Leaving the un-clickable widgets out changes nothing, and three interventions now say the same

`text` widgets are the obvious thing to drop from the list. They cannot usefully be clicked; one of
the eight recorded wrong answers clicked `\Device\HarddiskVolume1`, a `text` value in System
Information; and a `text` label named `Font :` sitting in front of a `combo box` of the same name
caused a defect that took a whole round of work to find. Three independent reasons, all pointing the
same way.

**Measured, and it is a wash.** 308 widgets to 217 — a 30% smaller list — over the same 37 goals:

| | right / 28 | wrong | *on a screen with no right answer* | missed | declined / 9 | correct decisions |
|---|---|---|---|---|---|---|
| every widget | 18 | 8 | 1 | 3 | 8 | **26 of 37** |
| without `text` | 19 | 10 | 2 | 1 | 7 | **26 of 37** |

Five goals moved, two better and three worse:

    지우개로 바꿔줘             wrong    -> right    Eraser
    카테고리 이름만 검색하게 해줘   missed   -> right
    복사                     right    -> wrong    Select
    램 좀 늘려줘               declined -> wrong    Installed Physical Memory (RAM)
    닫아                     missed   -> wrong    REMOTE-PC

Identical on correct decisions, and **slightly worse where it counts**: two more harmful outcomes
(a click on the wrong thing) in exchange for two fewer harmless ones (doing nothing). `복사` is a
regression on a goal that worked, and `램 좀 늘려줘` went from correctly declining to clicking a row
of hardware information. Not adopted.

It also weakens the density direction further. A third off the list, and the total does not move.

### The model is where the gain was, and the newer 8B beats the bigger 14B

Four configurations, same 37 goals, same prompt, three repeats each:

| model | right / 28 | wrong | *no-answer* | missed | declined / 9 | correct | a call |
|---|---|---|---|---|---|---|---|
| `qwen2.5:7b` | 18 | 8 | 1 | 3 | 8 | 26 of 37 | 0.45 s |
| **`qwen3:8b`, thinking off** | 23 | 5 | 2 | 2 | 7 | **30 of 37** | **0.45 s** |
| `qwen2.5:14b` | 25 | 5 | 2 | 0 | 7 | 32 of 37 | 2.39 s |
| **`qwen3:8b`, thinking on** | **27** | 3 | 2 | 0 | 7 | **34 of 37** | 3.75 s |

**Row one to row two is the only free change measured on this project.** Same latency to two decimal
places, four more correct decisions. Everything else here has been a trade.

**One 5 GB model covers both ends of the frontier.** `qwen3:8b` at 0.45 s is the fast point and the
same model thinking is the accurate one — 27 of 28 is 96%, above the criterion's nine in ten. The
14B sits between them, costs 10.4 GB, and needs a second download to reach a place one flag already
goes. Adopted: **`qwen3:8b` with `think=False`, and `--think` for the other end.**

**Neither end passes the whole criterion.** `REBUILD.md` fails the design if the right-thing rate is
below nine in ten *or* a reply routinely takes more than a second. Thinking off is 82% and 0.45 s;
thinking on is 96% and 3.75 s. The shipped default takes the latency clause, because a wrong
proposal costs a glance and a "no" while a wait is paid on every turn including pure conversation —
and the app makes a person confirm every action anyway.

**Bigger is not monotonically better.** The 14B gets two goals wrong that both 8B models get right:
`동그라미 그리고 싶어` → `Rectangle`, and `빨간색 골라줘` → `Orange`.

**Unmeasured, and the obvious next thing**: thinking only when an action is at stake. Answering
*"뭐가 열려 있어?"* needs no reasoning and choosing a widget to click might; `converse` does both in
one call, so separating them means two calls and the round trip is the budget.

### The tenth way this harness produced a confident wrong number

`llm.py` discards `message.thinking` — correctly, it is not an answer — and **that hid what it
cost**. `qwen3:8b` answered a twelve-token question with 201 output tokens and 1,112 characters of
reasoning, 2.95 s to first token, and nothing in the `Reply` said why. The text is still discarded
and its length is now recorded as `Reply.thought`, so the bill is visible.

### A bigger model moved the line first — and breaks the latency clause

`qwen2.5:14b`, same 37 goals, same prompt, same three repeats, **run twice and identical both
times** (25/5/0/7, 262 s and 265 s).

| | right / 28 | wrong | *no-answer screens* | missed | declined / 9 | correct | a call |
|---|---|---|---|---|---|---|---|
| `qwen2.5:7b` | 18 | 8 | 1 | 3 | 8 | 26 of 37 | **0.44 s** |
| `qwen2.5:14b` | **25** | 5 | 2 | **0** | 7 | **32 of 37** | 2.38 s |

**Thirteen goals moved, eleven of them for the better.** Everything the 7b missed entirely is gone
(3 → 0), and the errors that remain are a different kind:

    동그라미 그리고 싶어    run 1 Oval (right), runs 2 and 3 Rectangle   — it disagrees with itself
    빨간색 골라줘         Orange x3                                  — a plain error, Red is there
    이 그림을 메일로 보내줘   Share x3                                   — should have declined
    이 문자를 파일로 저장해줘  Copy x3                                    — should have declined
    보기 메뉴 열어         Application x3                             — the menu BAR, not the menu item

It is still on the same trade — bad clicks on screens with no right answer went 1 → 2 — but it gains
far more than it gives up, which is what "moving the line" means and what the prompt and list edits
never did.

**The cost is the criterion.** `REBUILD.md` says the design has failed *"if a reply routinely takes
more than a second"*, and 2.38 s is not close. The model is 10.4 GB and sits **entirely in this
machine's 16 GB of VRAM**, so that is what a 14B costs here rather than a spill that could be tuned
away. On the other clause it is much better placed: 25 of 28 is 89%, against a required nine in ten.

**Adopted as the default with the conflict written next to it**, because the accuracy clause is the
one the project said would decide whether the rebuild was worth doing, and `--model qwen2.5:7b` is
one flag back. What has *not* been tried: a newer model of the 7b's size, which would test whether
the gain needs the latency at all.

### The same shape three times: anything that raises "right" by one costs a refusal

| intervention | right | declined / 9 | correct decisions |
|---|---|---|---|
| as it stands | 18 | 8 | 26 |
| a stronger language rule in the first paragraph | 19 | 3 | 22 |
| the same rule as a separate line | 17 | 8 | 25 |
| dropping `text` from the list | 19 | 7 | 26 |

**Every intervention that moved `right` up by one moved refusals down.** Two were prompt edits in
different places and one was a change to what the model can see, so this is not one mechanism —
and none of them beat 26. The model appears to sit on a trade between acting and refusing, and
these nudges slide along it rather than move it.

What that suggests, and what has *not* been measured: the way past 26 is probably not persuading
this model to be more willing, but changing what it chooses **with** — a larger model is the obvious
candidate and nothing suitable is installed — or changing the cost of being wrong, which the app
already does by making a person press the button.

## Two harness faults found while measuring, both of the familiar shape

**The name protocol scored 0 of 5 while the model was answering correctly.** The prompt said "the
exact name of one thing above" and listed things as `- button "Copy"`, so the model replied
`button "Copy"` — a literal reading of the instruction. Every answer was right and every answer was
scored zero. Fixed in the prompt, not by matching leniently: a lenient match would also let a genuine
invention through.

**The shuffle seed used `hash()` on the goal string**, which Python randomises per process, so the
order could not be reconstructed afterwards — and the first analysis of the gaps silently scored
against an order the model was never shown, reporting a median gap of 27 that meant nothing. Now
`crc32`, and the positions are recorded at run time rather than reconstructed. A measurement whose
conditions cannot be rebuilt is not a measurement.

These bring the running count of ways this local-model harness has produced a confident wrong number
to nine. Every one so far has been invisible from the outside: the agent simply did nothing, or did
something plausible.

## Not measured

Perception and actuation, deliberately — they are separated so a bad number points at one part.
End-to-end accuracy on a live desktop is therefore **not** 64%; it is 64% times whatever the other
two contribute, and the click half is unmeasured. The goals are also mine, which is the honest limit
of this set: someone else's thirty-seven would not be these thirty-seven.
