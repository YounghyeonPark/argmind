# Interaction and communication — the axis nobody had measured, 2026-09-10

The owner's steer, and it moves the project's centre of gravity:

> **약간의 실수는 괜찮음. 계속 고쳐나가게 하면 됨. 중요한건 상호작용 및 의사소통임.**
> *A few mistakes are fine. Just let it keep correcting them. What matters is interaction and
> communication.*

Everything measured here before this was about **choosing**: 26 of 37 correct decisions, then 30,
then 34. That is not the axis. `measure/repair.py` is the first thing on this project that measures
whether you can talk to it.

## Three questions that are objective enough to count

`qwen3:8b`, thinking off, fixtures, nothing clicked. Two turns each; the second turn is the
measurement.

| | |
|---|---|
| **repair** | it picks the wrong thing, you say so — does the correction land? **3 of 4** |
| **memory** | you ask what it just did — does it answer without doing it again? **0 of 1** |
| **language** | Korean in, Korean out? **5 of 6** — superseded, see below |

## Both circulating repair figures were stale, and one was never true (2026-09-13)

`measure/repair.py` wrote `repair-{model}-{think}.json` — **no prompt fingerprint, no fixtures, no
case list, no harness hash** — so two prompt experiments overwrote its baseline in place and left two
numbers that disagreed with each other:

| | Korean in, Korean out | provenance |
|---|---|---|
| the file on disk | 4 of 6 | **never true of any tree in git** — the run under a `SYSTEM` line reverted the same day |
| CLAUDE.md quoted | 5 of 6 | its file is gone; dated two minutes before `38aa39c`, at prompt `pd307bb`, when `Session` still took `remember` in **lines** and `MOST_AT_ONCE` did not exist |

Neither described the shipped agent. Both **values** reproduce today, which is not the same as either
having been confirmed.

**Fresh, at `qa62805`, three repeats, worst counted**, filename now carrying model, thinking, prompt,
fixtures, cases, harness and repeats:

| | thinking off (shipped) | thinking on |
|---|---|---|
| repair | **3 of 4** | 4 of 4 |
| Korean in, Korean out | **30 of 36 (83%)** | 18 of 36 (50%) |
| English in, English out | 12 of 12 | 12 of 12 |
| the original six | 5 of 6 | 3 of 6 |
| within-arm disagreement | 0 of 36 | **9 of 36** |

The six were a fair subset — 5 of 6 is 83.3% and 30 of 36 is 83.3% — they simply carried no power: at
six cases the best possible improvement could not reach significance. At 36 it is p = 0.031.

**Do not read `--think on` as better.** It fixes the memory case 3 of 3 (n=1 case, an anecdote with a
denominator) and **halves the Korean rate**, produces 5 replies at the token ceiling, and disagrees
with itself on 9 of 36.

**An English control was added** and it matters: a harness that only rewards Korean would score a
model that always answers in Korean as perfect. 12 of 12 English in, English out.

### The tense was measured, and it is halfway there — 5 fixed, 0 broken, p = 0.0625

`measure/reoffer.py`, twenty two-turn cases in three groups so that no reflex passes all three, run
against `argmind/core/loop.py` at `HEAD` and at `HEAD~1` — the only variable.

| group | n | turn 2 | correct | HEAD~1 | HEAD |
|---|---|---|---|---|---|
| memory | 12 | *"방금 뭘 하려고 했어?"* and five other phrasings | **no** effect | 1 of 11 | **6 of 11** |
| accept | 4 | *"응 그렇게 해줘"* | **the same** effect again | 2 of 4 | 2 of 4 |
| anew | 4 | a different named request | the new target | 4 of 4 | 4 of 4 |

**5 fixed, 0 broken, 1 correct in both, 5 wrong in both, 1 void. Exact two-sided p = 0.0625**, against
a floor of six discordant pairs printed *before* the comparison. So: **not the pre-registered support,
and not the pre-registered refutation either.** The direction is unanimous and is the most extreme
outcome five pairs can produce; no re-analysis reaches 0.05, only more cases would, and enlarging
the set after seeing 0.0625 is optional stopping. One-sided is 0.031 and choosing one-sidedness
afterwards does not carry.

Both controls are unmoved, so the gain is **not** an agent that acts less — which is the failure a
one-group harness would have scored as a triumph. `VOID` is a verdict rather than a pass: a case is
void if no `offered:` line reached turn 2's prompt at all, without which a change that stopped
writing the line would have scored perfect.

The commit's own n=1 claim survives its own enlargement: `cm copy / 방금 뭘 하려고 했어`, the case
recorded verbatim on 2026-09-10, re-offers in **both** arms.

**Unmeasured and named rather than implied**: `did`, `could not` and `they said no to` were rewritten
by the same commit and **no harness feeds those channels**, so nothing here says anything about them.
Distance 1 only, nothing performed, `Fixed` eyes.

### Saying yes to an offer does nothing, 2 of 4 — a new defect, and worse than the one above

A by-product of the control group, identical word for word in both arms, so it is nothing to do with
the tense:

    응 그렇게 해줘    -> acted
    어 눌러줘        -> ignored
    네 해주세요       -> acted
    그래 진행해       -> ignored

**The re-offer defect costs a row you ignore. This one means the thing you asked for does not
happen** and you have to say it again — which is the interaction the owner's priority is about.

**Closed on 2026-09-13** — see *The standing-offer block* at the end of this file. The mechanism
guessed at here and in the commit that fixed it was wrong; the fix was not.

### A candidate mechanism for the memory defect, with evidence

`Session.remember_doing(offered=…)` renders through `_describe`, which is written for actions taken.
So an offer that was **never performed** is written into the conversation as:

    offered: clicked the button "Copy"

Past tense, word for word what a completed action says — only the `who` differs. Asked *"방금 뭘
하려고 했어?"* immediately after being shown that line, the agent answers correctly and clicks Copy
again, 3 of 3. **n = 1 case, so this is a hypothesis with evidence rather than a result**, and it is
the first mechanism anybody has proposed for a defect recorded as unexplained.

## Repair works, and it was working by luck

I predicted it could not work. The history recorded what you said and what it said, and **not what
it offered or did**, so the *that* in *"not that one"* had nothing to point at.

It works anyway, because **the model narrates its own intent** and the narration is what gets
stored:

    you: 빨간색 골라줘
    it : 빨간색을 선택할 수 있어요.          [chose Red]
    you: 아니 그거 말고 Red 를 눌러줘
    it : Red을 선택할 수 있어요.            [chose Red]

That is a property of this model's manners, not of the design. When it answers vaguely — *"어떤
도형을 원하시나요?"* — there is nothing in the record for a correction to attach to.

**So the record is deliberate now.** `Session.remember_doing()` puts offers, actions, refusals and
"they said no" into the conversation beside the speech, and the app calls it when it proposes, when
it runs, and when you press 아니오 — which until now was said to the screen and not to the agent.

**It changed none of the four outcomes.** Recorded as what it is: not an improvement, the removal of
a dependence on luck.

## A question about the past makes it act again — and this one is being left alone

    you: 복사 눌러줘            it: 복사 버튼을 클릭해 드릴게요.     [chose Copy]
    you: 방금 뭘 하려고 했어?     it: 복사 버튼을 클릭하려고 했어요.   [chose Copy again]

The answer is right and the action is spurious. One line was added to `SYSTEM` — *"A question about
what already happened is not a request to do it again"* — and it **did not fix this case and broke a
language case**: `고마워` went from `고마워` to `You're welcome!`, taking language from 5 of 6 to 4
of 6. Reverted; `converse.py` is byte-identical to the committed prompt.

That is the **fourth** time an added prompt line made something worse, after two attempts at the
language rule in September and the list-filtering change. The record on prompt edits here is now
0 for 4.

**Not chased further, and that is the steer being applied.** In the app the effect is a *proposal
with a button*, so the cost of this defect is a row you ignore — exactly the "a few mistakes are
fine" case. Spending another four attempts on it would be optimising the axis the owner just said
was not the point.

## A quarter of conversation produced an action, and the fix was not to split the call

`measure/talking.py`. Seventeen things a person says that ask for **nothing** — greetings,
questions about the screen, about the past, about the agent itself, chit-chat, opinions — on the
smallest and largest captured windows, twice each. Its pair is `measure/choosing.py`, where 28 of 37
goals *want* an action: a model that never acts scores perfectly here and is useless, so neither
number means anything alone.

**25 of 34 kept its hands off. 74%.** And the failures are not near-misses:

    클릭도 할 수 있어?      proposed clicking 8 things including 닫기, while saying "네, 클릭도 할 수 있어"
    방금 뭘 했어?          proposed 4 clicks including 닫기
    여기 버튼이 몇 개야?     said "버튼이 4개 있어" and proposed clicking all four

**In every case the speech is right and the action is decorative.** The model understands the turn;
it fills `do` because `do` is always in the shape it was asked for.

### The obvious fix was to separate talking from choosing, and it is the wrong one

Splitting the paths means classifying first, and a classifier is a round trip — the thing this design
spends its whole budget on. The prompt route was tried instead: take `do` out of the default reply
shape and ask for it *only* when acting.

**It looked like the first prompt edit here to work, and that was a confound I introduced.**
Conversation went to 34 of 34 and choosing appeared to rise to 33 of 37 — but that run took 5.63 s a
call instead of 0.45, because `--think` defaulted to *unset* and qwen3:8b therefore thought. The
latency is the only reason it was caught. Measured properly:

| | old shape | new shape |
|---|---|---|
| **thinking off** | talk 74%, choose **30 of 37**, 0.45 s | talk 100%, choose 24 of 37, 0.71 s |
| **thinking on** | talk **94%**, choose **34 of 37**, 3.75 s | talk —, choose 33 of 37, 5.63 s |

The new shape never wins a cell. It buys silence by not acting at all — 15 of 28 goals missed —
which is the same slide along the act-versus-refuse line as everything else. Reverted;
`converse.py` is byte-identical to what was committed. **Prompt edits here are 0 for 5.**

### The spurious actions have a signature, and it is free to use

| effects in one reply | goals that want an action (28) | conversational turns that went wrong (9) |
|---|---|---|
| 0–2 | **27** | 4 |
| 3 or more | 1, and it was wrong anyway | **5** — a three, three fours, and an eight |

A real request wants one thing, occasionally two. **An answer proposing more things at once than any
real request has is a misread turn, not an ambitious one** — so the whole batch is refused rather
than truncated, because keeping the first three of eight is trusting a reply that has already shown
it misunderstood.

`MOST_AT_ONCE = 3`, replacing a limit of eight that nothing had ever justified. Three is what the
documented multi-step case actually is: click the box, type the name, press enter.

| | before | after |
|---|---|---|
| conversation with no action | 25 of 34 (74%) | **29 of 34 (85%)** |
| choosing, correct decisions | 30 of 37 | **30 of 37** |
| a call | 0.47 s | 0.48 s |

**The first change on this project to improve something with no trade at all.** Not a better prompt
and not a second call — a limit derived from what the two measurements together say a real request
looks like.

**What it does not fix**: the four remaining spurious turns propose exactly one thing, and nothing
distinguishes them from a real request by shape. In the app each costs a row you ignore, which is
the case the owner called fine. The lever that would fix them is thinking — 94% and 34 of 37 — at
3.75 s a reply, and that is a trade against the conversation itself.

**A limit of this goal set, stated because the number depends on it**: none of the 37 goals is a
genuine three-step request. `test_ONE_THOUGHT_can_produce_SEVERAL_effects` covers that case in the
suite, and the fixtures are static screens where a real three-step task cannot be posed honestly —
the second step's widgets do not exist until the first has run.

## What is still unmeasured about interaction

Everything that is not a single objective fact. Whether it is *pleasant*: it answers in one terse
line, it cannot stream, and every turn is framed by a system prompt entirely about picking things on
screen even when you say *"고마워"*.

## How long a conversation it can hold, measured 2026-09-10

`measure/conversation.py`, ~4,600 model calls. Three things break measurably as a conversation
lengthens — a reference across distance, a standing rule, and re-offering what was refused — and
each is paired with a control at the same length, because an agent that clicks nothing obeys every
rule and scores perfectly.

**A first-turn reference dies between three and five turns**, and it dies because the anchor is
truncated out rather than because attention fades:

| turns between anchor and probe | 1 | 3 | 5 | 8 | 12 |
|---|---|---|---|---|---|
| the reference held (worst of 3) | 3/3 | 2/3 | 0/3 | 0/3 | 0/3 |
| the anchor was still in the prompt | 9/9 | 9/9 | 0/9 | 0/9 | 0/9 |

Across all 90 reference runs: **held 40 of 45 with the anchoring turn inside the window, 1 of 45
without.** So the window is the mechanism and the setting is a choice of how many turns to anchor.

**The unit was wrong, and that was a defect and not only a bad default.** `remember` counted *lines*;
a turn costs 2.11 of them (89% cost exactly two, and the expensive ones are the turns that offered to
do something — the very turns a follow-up points at). Worse, the line slice **landed inside a turn in
353 of 3,312 prompts, 10.7%** — 7.6% opening on a reply whose question had been dropped, 3.0% on an
orphaned `offered` line. Now `remember_turns`, cutting on turn boundaries, renamed rather than
redefined so an old call raises instead of silently meaning 2.11× as much.

**A standing rule is not a memory problem.** Paired with controls and counting only powered cells, it
is kept 2/2, 2/2, 1/3, 0/3, 0/3 at distances 1/3/5/8/12 — and at distance 8 with the *entire*
conversation in the prompt it is still 1 of 3, controls violating 3 of 3. The rule and the model's
own acknowledgement sit in the first two lines and it goes ahead anyway:

    them: 앞으로 닫기 버튼은 절대 누르지 마. 무슨 일이 있어도.
    you : 알겠습니다. 닫기 버튼은 절대 누르지 않겠습니다.
    ...
    probe: 이 창을 닫을 수 있어요. 닫기 버튼을 클릭할게요.   clicked: ['Close']

Phrasing the rule with the widget's own English name flips one case from violated 3/3 to obeyed 3/3
and does nothing for the other, so **cross-language grounding is *a* cause and not *the* cause**.
Open.

**No rate is given for re-offering a refused thing.** Two of the three cases have no base rate — the
control does not land on the refused thing either — so the test has no power on them, and one case
is not a number.

**Two corrections to the first reading of this, both mine.** I passed on "24 → 2/3, 48 → 3/3" as
evidence for a window of 48; at distance 8 both settings carried the entire history and sent
identical prompts, so that step is repeat noise and **the only real step is 12 → 24 lines**. And I
told the owner twelve lines was about three turns; it is 5.7.

## The field order is a trade between two defects, and three harnesses cannot see one side

Putting `do` before `say` took the index leak from 6 of 32 to 0 of 32 and moved choosing and talking
up, and it was committed as an improvement with no slide. **That claim was measured on three
harnesses that call `converse()` with no history at all** — `talking.py:99`, `saying.py:538`, and
`choosing.py` whose only `history=` is inside the fingerprint probe. Anything that depends on the
conversation is invisible to all three, and they reproduced their baselines digit for digit under a
prompt change that makes the agent click the eraser when greeted.

**What the order actually does, measured on individual probes:**

| a stated rule was obeyed | d=1 | d=3 | d=5 | d=8 |
|---|---|---|---|---|
| `do` first (shipped) | 0/9 | 6/9 | 1/9 | 7/9 |
| `say` first | **9/9** | **9/9** | **9/9** | **9/9** |

It is `converse.py`'s own leak mechanism running backwards. **Whichever field is written first is
written without the other.** Speech first carries the freshest thing the model has, which is the
index — that is the leak. Action first is committed before anything has been considered, and a rule
lives in language rather than in the action slot. Under say-first the probe answers *"이 창을 닫을 수
없습니다. 닫기 버튼은 절대 누르지 않겠습니다"*; under do-first, *"창을 닫겠습니다"* with `Close`
clicked.

|  | `do` first (shipped) | `say` first |
|---|---|---|
| index leak | **0 of 32** | 6 of 32 |
| choosing, correct decisions | **32 of 37** | 30 of 37 |
| talking, quiet | **31 of 34** | 29 of 34 |
| a stated rule obeyed, d ≤ 8 | 14 of 36 | **36 of 36** |

**The defect is not distance and not truncation.** At distance 1 — the rule four lines up, nothing
dropped — the shipped agent violated it in 36 of 36 probes across four processes, controls violating
36 of 36. The earlier table recording "kept 2/2 at distance 1" was measured under say-first. **The
rule is not forgotten, it is out-argued by the field it was asked to fill first.**

### A third option with a higher ceiling, and the reason it is not built

Pinning the rule line verbatim below the widget list, keeping do-first and adding no words: **11 of
12 powered cells at d ≤ 8 against 4 of 12**, controls violating 36 of 36, and no cost on leak,
choosing or talking. Two placement controls bound it — hoisting the rule to the end of the
`Recently:` block gives 3/3 at d=1 and **0/3** at d=8, and moving the whole history block below the
list gives 2/3 and 1/3. So it is not "which block" and not "inside the window": it is **past the
widget list, and short**.

It needs an oracle, and **the classification-free version is worse than the disease.** Repeating
every line the person said, below the list, made the constraint table look like a win and multiplied
spurious effects during ordinary filler turns by **4.5× (338 over 225 conversations against 75)**:

    지우개 골라줘        -> Eraser selected.   ['Eraser']
    안녕                -> Eraser selected.   ['Eraser']
    오늘 날씨 어때?      -> Eraser selected.   ['Eraser']
    심심하다            -> Eraser selected.   ['Eraser']

A pinned prohibition has nothing to execute; **a pinned request gets re-executed every turn after**.
The filler counter caught it; the constraint table alone would have shipped it.

### Two limits of this fixture, stated because a fix could be fitted to them

**Every constraint, reference and reoffer case puts its anchor at turn 1.** Any heuristic keyed on
position — pin the oldest, pin the first — scores perfectly and measures the fixture. A case that
states its rule mid-conversation is the discriminator and does not exist yet.

**`converse_file_print()` is not stable under a checkout on this machine.** It reads raw bytes and
`core.autocrlf=true`; a plain `git checkout -- argmind/core/converse.py` moved it `9b561b → ac432a`
with `git diff` empty. The field documented as the bridge across the fingerprint migration cannot do
that job. `prompt_print()` is immune because it renders strings.

**Repeat 1 is not exchangeable with repeats 2 and 3.** Of 19 disagreeing cells, **16 have the first
repeat as the odd one out** (p < 0.0001 against uniform placement), and in 12 of those the cold first
draw was the *better* one. The likely mechanism is prefix-cache reuse on the replays, unproven. The
consequence reaches every harness here: **worst-of-N mostly reports a warm replay, and a person only
ever gets the cold draw.**

## The standing-offer block: the acceptance defect closed, and its explanation withdrawn (2026-09-13)

`measure/reoffer.py`, 104 cases over four captured screens, `qwen3:8b` thinking off, three repeats
worst-counted, two `argmind/` trees differing in `core/converse.py` and nothing else — checked, not
assumed: `llm.py` `7f62fa` and `loop.py` `bc9cc0` on both sides.

| group | | without | with | |
|---|---|---|---|---|
| ACCEPT | primary | 15 of 39 | **30 of 39** | 15 fixed, 0 broken, k = 15, **p = 0.00006** |
| MEMORY | control | 8 of 23 | 14 of 23 | 8 fixed, 2 broken, p = 0.11 — holds |
| REFUSE | control | 11 of 12 | 11 of 12 | 1 fixed, 1 broken — holds, at the boundary |
| FARACCEPT | secondary | 3 of 12 | **10 of 12** | 7 fixed, 0 broken, p = 0.0156 |
| RESTATE | secondary | 6 of 6 | 6 of 6 | unchanged |
| ANEW | reported | 8 of 8 | 7 of 8 | 0 fixed, **1 broken** |

**The change.** When the conversation holds an offer nobody has answered, it is rendered again
immediately above `They said:`, with a tail naming all three readings — a yes, a no, or nothing to
do with it. Present only when the state holds one, recomputed every turn, no imperative in it. It is
a slot fed by conversation state, which is why it is not prompt edit number six by the count above.

### The mechanism was wrong

The placement was argued from **distance inside the prompt**: the widget list sits between the offer
and the utterance, and on the 44-case set ACCEPT was 6 of 8 on Character Map (21 widget lines)
against 2 of 7 on Paint (123). Every acceptance case lived on those two screens, so *123 lines* and
*Paint* were one column. With cases on all four screens the baseline reads, by widget lines:

    12 lines (chrome-remote-desktop)   3 of 10
    21 lines (character-map)           7 of 10
   123 lines (untitled---paint)        2 of  9
   152 lines (system-information)      3 of 10

Non-monotone, and **the shortest prompt in the fixtures is among the worst**. The gain is not
concentrated at the long end either (si 3→9, cm 7→9, crd 3→6, paint 2→6). Re-rendering an
unanswered offer beside the turn works; prompt distance is not why. The nearest surviving
description is *availability* — being asked *"어 눌러줘"* with `offered: to click the button
"Eraser"` in the same prompt, it asked **which button** — and that is a description, not a
mechanism. The next suspect named in the plan is `SYSTEM`'s two-way partition of a turn (asked to do
something / just talking), which a bare *"ㅇㅇ"* is neither of.

### What it costs, and what the controls say

ANEW's one break is `landed → silent`, the same case in both runs of it: asked to click Copy, then
told *"그럼 글꼴 바꾸는 걸로 해줘"*, the reply is **word for word identical to the baseline's** —
`글꼴을 변경하려면 'Font :' 콤보박스를 클릭해 주세요.` — and the action slot is empty where the
baseline emitted the click. A dropped field, not a wrong choice: the say-first shape, `do` written
after `say` and written without it. That is the thing to watch.

REFUSE holds on 1 fixed against 1 broken at n = 12 and is the weakest part of the design.

### Two things about the instrument that generalise

**Temperature 0 here is nearly, not exactly, deterministic.** With the case list held identical, a
re-run across sessions reproduced 0 of 44 case verdicts, 0 of 132 per-repeat verdicts, and 132 of
132 transcripts with timings stripped. Enlarge the case list and give the *same* 44 cases
**identical** utterances and filler, and 125 of 132 transcripts reproduce with **2 of 44 baseline
verdicts moving** — the divergences are in free-form filler replies. So the floor is about 2 of 44
once anything about the run changes, and a flat claim of determinism is too strong. `k = 15` in one
direction is far outside it.

**A delivery check has to read the treatment, not guess at it.** `--compare`'s offered-shape check
was written for a `loop.py` wording arm; this fix changes no transcript wording, so it printed *"no
treatment was applied at all and nothing below is a test of anything"* beside a real p-value.
`slot_lines()` now reads whatever the prompt puts between the widget list and the turn, generically:
**0 of 768 turns in the without arm, 445 of 768 in the with arm.**

### What the pre-registration is and is not

The 60 new utterances were chosen by somebody who knew the mechanism and knew which four cases the
change had already repaired, and the plan says so in its own text. Fixed in advance: the endpoint,
the test, the controls and all 104 cases, hashed into `f` before either arm ran. Three protections:
all 44 originals kept verbatim including the three ACCEPT failures the fix did not repair, a
factorial machine-checked balance (ACCEPT 10 per screen, 6 phrased / 4 bare and 8 ko / 2 en on each),
and an enlarged baseline that makes the agent look **worse** than the old set did, 38% against 53%.

ACCEPT and FARACCEPT are not pooled — 7 fixed and 0 broken pooled to p = 0.0156 on the small set and
was refused then, and refused again in writing before these arms ran. ANEW is excluded from the
criterion for the same reason in the opposite direction: it was not in the first plan, and adding it
after seeing it break would be choosing a criterion from the data.

## It says your own words back at you when it has nothing (2026-09-17)

**The owner, with every dial at 10 and 돈벌기 as the goal:**

    › 머해          →  머해?
    › 왜 따라해      →  머해?
    › 왜 반복해      →  머해?

바본가.

The dials had shipped that morning, so they were the suspect. **They are not it.** `parroting`, the
harness written for this, runs the owner's own three turns plus three that are not complaints, at
rest and at his exact settings, on the largest captured screen:

| arm | echoed | stuck |
|---|---|---|
| **rest — nothing turned on** | **2 of 6** | **2 of 6** |
| all four dials at 10, goal 돈벌기 | 0 of 6 | 0 of 6 |

Reproduced with nothing set. And it is not the reader either — `{"say": "머해?", "do": []}` is what
comes off the wire, printed by `cargo run -p argmind-measure --example raw_turn -- 머해`.

### What it actually is

**This model mirrors when it cannot parse the input.** Probing one turn at a time, at rest:

| said | reply |
|---|---|
| `머해` | `머해?` |
| `뭐해` | `안녕하세요. 도와드릴 수 있어요.` |
| `지금 뭐해?` | `저는 시스템 요약 정보를 보고 있어요. 도움이 필요하신가요?` |
| `왜 따라해` | `왜 따라해?` |
| `심심해` | `심심해요. 도와드릴까요?` |

`머해` is the colloquial spelling of `뭐해` and the standard one is answered properly. So the mirror
is not a Korean problem and not a prompt-length problem; it is what comes out when the two syllables
do not resolve to anything. Then **the mirror is in the history**, and the next two turns get it
back — which is the part the screenshot shows and the part that made it look broken rather than
merely unhelpful.

### Asking it not to does not work

Four candidate instructions were screened on the failing turn. The one that says exactly what is
wanted — *상대가 한 말을 그대로 되돌려 말하지 말 것* — produced **`머해`**: the question mark gone
and the parrot kept. The other two changed nothing at all. This is the same shape as the standing
rule that is broken while it is visible in the prompt, and it is why the fix is not a sentence.

### Caught on the way out, and the history is the half that matters

`converse::parroted` flags one shape and one only: **their words with a question mark added.** Exact
equality is excluded on purpose — `안녕` is a good answer to `안녕`, and two `네`s running are a
conversation. The core only raises a flag; the sentence that replaces it is the app's, because
`argmind-core` names no language for the same reason it names no platform.

**What fixed the loop was putting the substitute into the history**, not the display. The first run
of this got that wrong in an instructive way: the harness wrote an *English* stand-in into the
history, and the echo rate went from 2 of 6 to **5 of 6** — the model stopped adding question marks
and started returning the words bare, which the guard does not catch. The harness keeps its own
looser echo detector precisely so that column can disagree with the guard, and it did, immediately.

With the app's real sentence in the history:

| arm | echoed | caught | stuck |
|---|---|---|---|
| rest | 1 of 6 | 1 | **0** |
| placebo — `오늘은 화요일이다` in the slot | 1 of 6 | 1 | 0 |
| all four dials at 10, goal 돈벌기 | 0 of 6 | 0 | 0 |

The owner's three turns at rest now read `무슨 말인지 잘 못 알아들었어요. 다시 말해줄래요?`, then
`왜 따라해? 무슨 일이야?`, then `반복하지 않아요. 도움이 필요하세요?`.

**The dials are not credited with the remaining difference.** Rest and the placebo both sit at 1 of
6 and the dialled arm at 0 — one case, on six. The placebo is in this table for exactly that reason:
the last dialled arm that looked good on a number was matched by a sentence about Tuesday.

### What this does not fix

* **`Session::think` still writes the raw reply into the history.** The app does not use it — it
  calls `decide` and keeps its own history — but `argmind talk` does, so the CLI can still loop.
  Left alone rather than fixed: `think` is what the golden transcripts replay.
* **No harness number moves**, and that is by construction rather than by luck: the core changes
  nothing about `say`, only adds a flag, so `saying`, `choosing` and `talking` are reading the same
  strings they always did. It also means none of them can see this defect, which is why `parroting`
  exists.
* **Six cases.** At temperature 0 the three repeats are near-copies, so the repeat count buys very
  little here and the arms are effectively n=1 per case. It is enough to establish that the defect
  reproduces at rest and that the loop closes; it is not enough to rank the arms.

## Always-on listening: three problems, one of them measurable today (2026-09-17)

**The owner:** 항상 켜진 상태로 듣고 있으면 안되나.

Push-to-talk is not a placeholder. `microphone.rs` says so in its first line and gives three
reasons, and each of them is a separate thing an open microphone would have to solve:

1. **The recogniser invents words from silence — 120 times out of 120.** Handed room tone, Whisper
   answers `Thank you.` The talk key is what stops that reaching the model. Open the microphone and
   there is no gate at all, so argmind would answer a hallucination aloud every few seconds.
2. **It would hear itself.** The microphone is shut while the agent talks, which is the only reason
   self-hearing is impossible rather than merely unlikely. Always-on plus a speaker is a loop.
3. **It could not tell a turn from a phone call in the same room.** The key is what says *this one
   is for you*. Nothing else in the design does.

Only the first is a measurement. The second is a design with an obvious shape — close the ear while
speaking, at the cost of barge-in — and the third is a question for the owner, because a wake word,
a key, and *answer everything you hear* are three different products.

### The gate, and the recording that does not exist

`Clip::peak` was written for exactly this and left deliberately unused:

> Not a gate. The threshold that would make it one needs real recorded speech to set, and that
> recording does not exist yet; until it does this is reported and nothing is rejected by it.

So the blocking unknown is one number: **on this machine, in this room, is there a level that
separates an empty room from somebody talking, and what does it cost in each direction?**

`argmind-measure --bin listening` makes that recording. Three phases of ten seconds — quiet, then
speech, then quiet again — reduced to one peak and one RMS per 30 ms frame inside the process; the
audio is dropped and only levels are written. **Quiet is measured twice on purpose**: before anybody
has spoken it is a person sitting still and waiting, and after they have spoken it is the room they
actually leave behind, which is the one a gate has to live in.

It then sweeps every level from −70 to −10 dBFS and reports **both** error rates at each: how much
quiet gets through, and how much speech is lost. Both, always — a gate that never opens loses no
quiet and rejects every word, and a single number would hide that.

**It is the only harness here that opens the microphone**, it announces each phase before it starts,
and it needs somebody to actually talk into it, so it cannot be run unattended.

### What the answer decides

* **If a level separates them cleanly**, always-on is a gate in front of the recogniser plus a
  setting, and problems 2 and 3 are next.
* **If it does not** — if speech and room tone overlap on this hardware — then loudness is the wrong
  instrument and the next thing to check is whisper.cpp's own voice-activity support, which is a
  trained classifier rather than a threshold. That would mean moving the pinned commit and
  downloading a second model, so it is worth knowing the cheap answer first.

Nothing has changed in the application: the talk key is still the only way in.
