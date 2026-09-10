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

## What is still unmeasured about interaction

Everything that is not a single objective fact. Whether it is *pleasant*: it answers in one terse
line, it cannot stream, every turn is framed by a system prompt entirely about picking things on
screen even when you say *"고마워"*, and nothing has ever been measured about a conversation longer
than three turns.
