# What it is trying to be

**The owner, 2026-09-17:** 설정에서 생각 목표설정이 가능하고 수치를 조정할수 있음. 재치, 유머,
봉사, 헌신 등등.

Asked immediately after establishing that argmind does not think, it answers: a turn's prompt is the
system sentence, twelve turns of history, the screen now, and an unanswered offer — and `Session`
holds no goal, belief or plan, so nothing survives a turn except *what was said*, never *what it
concluded*. Four dials and a sentence are not a mind. They are the first thing in this program that
persists across turns because a person put it there.

## The rule that made it safe to build

Prompt edits here are **0 for 5** — each raised one measured number and lowered another — and the
sixth that paid was a *slot* rather than a sentence, present only when there was something to put in
it. This is built to the same shape, and one guarantee is nailed down before anything else:

> **With every dial at rest, the prompt is byte-for-byte what it was before `mind.rs` existed.**

`tests/equivalence.rs` holds it from outside. The six probes render through the new parameter with a
resting `Mind` and still hash to `r322a4c` at 5,676 bytes — a tag and a byte count recorded from the
Python, long before a dial existed. That is a check against an outside record, not against itself.
`tests/mind.rs` holds the other direction: turning a dial up and back down restores the prompt
exactly, which is what catches a slot that leaves a blank line behind when it empties.

So the feature cannot cost anything until somebody turns it on.

## Does a dial do anything?

**A dial that moves nothing is worse than no dial** — it is an interface making a claim it cannot
keep. So the first question was not *is it better*, it was *is it wired*.

`saying`, 44 Korean requests over three captured screens, `qwen3:8b` thinking off, temperature 0, one
repeat unless said otherwise. At temperature 0 an identical prompt gives an identical reply, so any
difference in the sentences is caused by the slot.

| arm | Korean out | leaked | replies differing from rest |
|---|---|---|---|
| **rest — no line at all** | 28 / 44 | 5 / 32 | — |
| 재치=7, line in English | 30 / 44 | 1 / 32 | 39 / 44 |
| 유머=7, line in English | 30 / 44 | 0 / 32 | 40 / 44 |
| 봉사=7, line in English | 31 / 44 | 0 / 32 | 27 / 44 |
| 헌신=7, line in English | 28 / 44 | 0 / 32 | 20 / 44 |

Wired: 20 to 40 of 44 sentences change, and they change in the direction on the label — 재치 and 유머
drop the formal `-입니다` for `-야` and `-어요!`, 봉사 answers *I'll do it*, 헌신 answers *I did it*.

## The first reading was wrong, and the control is why

The leak column above fell from 5 of 32 to 0 or 1 in **all four** arms. A uniformly tidy result is
evidence the measurement is broken, and a flattering one means find the responsible assumption
first, so the obvious suspect was that *any* line in that slot does it, whatever it says.

**A placebo — one content-free sentence in the same slot, in the same shape:**

| arm | Korean out | leaked |
|---|---|---|
| `목표=Today is Tuesday.` | **19 / 44** | 2 / 32 |
| `목표=오늘은 화요일이다` | **42 / 44** | 1 / 32 |
| the same, repeated | **41 / 44** | 1 / 32 |

The leak improvement is the slot, not the dials: a sentence about Tuesday buys it too. Nothing about
wit or humour fixed anything.

And the column nobody was looking at moved instead. **The language of this slot decides the language
of the reply.** An English placebo *costs* nine cases of a defect this project already records —
*the agent still answers some Korean in English* — and a Korean placebo buys fourteen, reproduced on
a second run. The four English dial lines sat at 28–31 because they were English: they moved the
wording and left the language where they found it.

The comment in the first version of `mind.rs` said the line was English "because the rest of the
prompt is English and mixing languages inside one instruction is a thing this project has not
measured." It is measured now and the argument was backwards.

## What shipped, and what it trades

The line is Korean: `이런 태도를 목표로 하세요: 재치(빠르고 영리하게) 7/10.` plus the counterweight
`그렇다고 해서 할 말이 없는데 말하거나, 부탁받지 않은 일을 할 이유가 되지는 않습니다.` — said last,
because every one of these dials is a reason to say more than is wanted and 31 of 34 conversational
turns are measured as ones that should produce no action at all.

| arm | Korean out | leaked |
|---|---|---|
| 재치=7 | 41 / 44 | 7 / 32 |
| 유머=7 | 44 / 44 | 2 / 32 |
| 봉사=7 | 41 / 44 | 3 / 32 |
| 헌신=7 | 40 / 44 | 8 / 32 |

**At three repeats, worst counted**, against the baseline on the same harness print:

| | Korean out | leaked | control set |
|---|---|---|---|
| rest | 28 / 44 | 5 / 32 | 12 / 12 |
| 헌신=7, the worst arm above | **39 / 44** | **7 / 32** | 12 / 12 |

So: **+11 of 44 on the language axis, −2 of 32 on the leak axis.** The same trade as every prompt
edit in this project, and the owner's stated priority decides which way round it goes —
*중요한건 상호작용 및 의사소통임*, and answering a Korean question in Korean is that. The leak
movement is 2 discordant cases and is not a number anybody should quote as a regression.

**And it is paid only by somebody who turns a dial.** The resting prompt is untouched, which is the
whole reason this was allowed to be a prompt edit at all.

## What this does not say

* **Nothing about English speakers.** All 44 requests are Korean. The measurement cannot say what
  the right slot language is for somebody who talks to argmind in English, and a Korean line would
  presumably do to them what the English one did here.
* **Nothing about where the slot belongs.** It sits next to the trigger, which is inside the
  distance where a stated rule is obeyed 36 of 36 times — but that measurement is about rules and
  this is a manner. The last time a placement was argued from prompt distance the argument was
  withdrawn by the run that confirmed the change. Settling it means running the slot at the top and
  at the bottom; it has not been done.
* **Nothing about `choosing` or `talking`.** Neither harness has been run with a dial turned, so
  whether a dial makes the agent click things it should not is unmeasured. `talk 2 "유머=7"` and
  `choose 3 "봉사=7"` are the two commands, and the filename carries an `m` component so a dialled
  run cannot be confused with a resting one.
* **Nothing about whether a person likes it.** Four dials were chosen because each one is a line in
  every prompt for the rest of the program's life and a long list is how a prompt stops being read.
  Which four, and whether ten would be better, is taste.

## Where it lives

    native/argmind-core/src/mind.rs      DIALS, MOST, Mind, and the one line it renders
    native/argmind-serve/src/mind.rs     %LOCALAPPDATA%\argmind\mind.json — the third file of
                                         its shape, beside decisions.json and devices.json
    ui/src/index.html  #aiming           four sliders, a goal box, and the line as it is sent

The page prints **the exact sentence the model will be shown**, underneath the sliders. A settings
panel that renders four numbers and keeps what it does with them to itself is asking to be trusted
about the one part it could simply show; printing it also means the interface cannot quietly stop
matching the prompt, because it is the same string.
