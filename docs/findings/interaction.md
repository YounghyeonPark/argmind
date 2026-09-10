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
| **language** | Korean in, Korean out? **5 of 6** |

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
line, it cannot stream, every turn is framed by a system prompt entirely about picking things on
screen even when you say *"고마워"*, and nothing has ever been measured about a conversation longer
than three turns.
