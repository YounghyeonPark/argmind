# Is the person looking at the screen

**Two ways of answering it, measured on the same person at the same desk.** 2026-09-23,
`argmind-measure --bin attending`, `harness 646cb3`.

| | what it costs | what it gave |
|---|---|---|
| **YuNet landmarks**, `argmind-env-face` | **99 ms**, no model | looking **0.20 / 0.40**, turned **15.05 / 24.08** — ranges nowhere near touching |
| **`qwen3-vl:8b`** at 320×180 | **2,312 ms** | **nothing, this time — the question was leading and the arm is void** |

**So the cheap path separates a turned head from a facing one, hugely, and the expensive path was
not measured.** Both halves of that sentence matter and the second is the harness's fault.

## Conditions

`"Integrated Camera"` 1920×1080, settling in 3,024 ms. Detector fed 640×360 at a score threshold of
0.6; model fed 320×180, the size `seeing` found is the smallest that reads a gaze. Four phases of
ten seconds, **both orders**, 311 usable samples and 382 detector passes. One person, one desk, one
light.

## What the cheap path gave

| phase | 10% | median | 90% | n | face | unreadable |
|---|---|---|---|---|---|---|
| 화면 보기 (1) | 0.37 | **0.40** | 0.43 | 94 | 100 px | 0 |
| 고개 돌리기 (1) | 12.08 | **15.05** | 27.53 | 74 | 94 px | 22 |
| 고개 돌리기 (2) | 18.58 | **24.08** | 36.47 | 48 | 95 px | 48 |
| 화면 보기 (2) | 0.14 | **0.20** | 0.24 | 95 | 102 px | 1 |

Looking's 90th percentile is **0.41**; turned's 10th is **12.70**. Thirty times apart at the tails,
not at the medians — which is the comparison that matters, because two medians far apart with
overlapping ranges is a threshold that fires on the wrong phase several times a minute.

**Both orders gave the same verdict.** That control exists because `presence` once separated a still
room from a still room 2.17 against 0.65, three to one in the wrong direction, **purely by which
phase ran first**. Running one order here would have been worth nothing.

**The face stayed the same size**, 94–102 px, so all four phases were looking at the same subject.
That column is in the output for exactly the case where it is not.

## The threshold is 1.0, and it is geometry rather than the middle of the gap

`HEAD_ASIDE = 1.0`. **The nose has reached one of the eyes** — a statement about a face.

The obvious alternative was the geometric middle of the measured gap, 2.29, the way `camera::MOVING`
took 0.15 from between 0.05 and 0.38. It is the wrong move here, and the run is what shows why.

**`head_on` is not bounded, and this repository's own comment said it was.** *0 facing the camera,
±1 fully turned* was written before anybody ran it. The quantity is the nose's offset **divided by
the eye separation**, and at profile the eyes converge, so the divisor goes to zero and the ratio
runs away rather than saturating — 15 and 24 measured, against 0.2 and 0.4 facing. 2.29 sits in the
part of that range where the divisor is collapsing and the value is unstable.

So: **anything from about 0.5 to 10 separates this data**, and what picks 1.0 out of that interval is
that it is the only value in it with a meaning. Said out loud because a constant that looks derived
and is not is how a number stops being questioned.

## Unreadable is a signal, and it makes the gap an understatement

`head_on` returns nothing when the eyes land within a pixel of each other. That happened **0 and 1
times of 95 while looking, and 22 of 96 and 48 of 96 while turned.**

Near profile the detector still finds the face and the two eyes collapse, so *unreadable* is most of
the way to *turned* on its own. It also means the samples that survived in the turned arm are the
**less** turned ones — half of round two was dropped — so the measured separation is smaller than
the real one. A measurement that errs toward the boring answer is the direction to err in.

Nothing acts on it yet. It is recorded because a caller that wants the extreme angles has a second
signal sitting in what currently looks like a dropped sample.

## The expensive arm is void, and the fault is the question

All four phases came back **보고 있다**, two words, 2.3 s each — including both phases with the head
turned far enough that half the detector samples could not find two eyes.

The first reading was *the model failed*. It is withdrawn. What changed between this harness and
`seeing`, which read the same fact correctly 4 of 4 across two room states, is the question:

| | asked | took |
|---|---|---|
| `seeing` | *무엇이 있는지 한두 문장으로 … 사람이 있는지, 화면 쪽을 보고 있는지, 눈에 띄는 물건 하나* | 4–5 s |
| `attending`, first run | *'보고 있다' 또는 '보고 있지 않다' 로 시작해서 한 문장으로만* | 2.3 s |

**A forced choice with the affirmative listed first is a leading question**, and four identical
two-word answers is what the standards here call a uniformly tidy result: evidence the measurement
is broken, not evidence about the subject. The seconds say the same thing from the other side —
`seeing` established that the time is generation length, and 2.3 s is a model that stopped looking.

The harness now asks the open question. **Until it is re-run, nothing is established about the
expensive path on this axis** beyond what `seeing` already said.

## Wired, 2026-09-23 — and what was deliberately not wired

`설정 → 화면을 보고 있는지 알기`, the fourteenth decidable key, `attention`. Three switches now sit
under the camera and they are three decisions:

| key | what it allows | what it leaves behind |
|---|---|---|
| `camera` | the light stays on, movement is reported | nothing |
| **`attention`** | **where the head is pointed** | **nothing — a word with an age** |
| `faces` | the same face is recognised | a row per person, until deleted |

`attention` is its own key because it is its own fact: `camera` says *something moved* and this says
*a person is pointed at the screen*, which is more about them. It sits above `faces` in the list and
below it in what it gives away — the roster outlives the session and this evaporates when the device
closes.

**The reading rides the movement loop.** Two threads on one device is two lights' worth of nothing,
which is the reason `switch_camera` already refuses to start a second watcher. A face pass runs every
`FACE_PASS = 1.0 s` while the switch is on — about three readings per movement window — and the
window is classified at its edge by `pointed()`, a median, because **the constant was derived from
medians and using it on one frame would be using it against a statistic it was not derived from.**

What that costs: 99 ms of every 1,000 rather than of every 3,000. A movement window needs six looks
to judge and has room for about ninety at 32 ms each, so it loses roughly a tenth of them.

**Three readings, and the third is not a vote for the second.** `화면 쪽을 봄` / `고개를 돌림` /
`고개 방향을 읽지 못함`. The last one is where the eyes converged — measured 0 and 1 of 95 looking
against 22 and 48 of 96 turned, so it correlates with turning, and *correlates* is all that was
measured. `Facing` has four states rather than an `Option<bool>` for the same reason: *nobody asked*
and *it could not be read* are different facts, and collapsing them is the pitfall that cost this
project a round of work once already.

**Nothing is added at rest, and a test holds it.** `camera_line` was lifted out of the camera
thread's closure so the claim can be run: with the switch off the blackboard line is byte for byte
what it was before any of this existed. That is the ground the dials were allowed on and it is the
only ground a fifteenth prompt slot could have had — prompt edits here are 1 for 7.

**And it gates nothing.** The measurement says a turned head reads thirty times a facing one; it does
not say argmind should behave differently when it does. `somebody_about` still asks only about
movement, `notice` is still what it gates, and wiring attention into that decision would be shipping
a behaviour nobody measured. Reporting and acting are two steps and one of them is earned.

## Asked to judge, and it cannot — 2026-09-28

The owner: *말을 걸지말지는 상황에 맞게 판단하면 됨.* Not a rule that goes quiet when somebody turns
away, but argmind judging the situation. The natural way to give a model that judgement is to tell
it the fact where it decides to speak first — beside the watch's trigger — and let silence be its
answer. **That rested on one assumption: that silence is an answer this model gives on this path.**
`argmind-measure --bin turned` checked it, with the failure conditions written first.

28 unprompted moments over the four captured screens, each in four arms that differ only in one
appended sentence, built by the same `watch::regarding` the application would use:

| | spoke | talked about the person | wanted to click |
|---|---|---|---|
| **shipped instruction** — *Mentioning it is usually enough* | | | |
| 없음 (today) | 28 / 28 | — | 5 / 28 |
| 자리에 있음 (control) | 28 / 28 | | 1 / 28 |
| 보고 있음 | 28 / 28 | | 4 / 28 |
| 고개 돌림 | 28 / 28 | | 4 / 28 |
| **counterfactual** — *say something only if it is worth their attention; otherwise leave "say" empty* | | | |
| 없음 | 28 / 28 | 0 / 28 | 9 / 28 |
| 자리에 있음 (control) | 28 / 28 | **18 / 28** | 9 / 28 |
| 보고 있음 | 28 / 28 | 9 / 28 | 12 / 28 |
| 고개 돌림 | 28 / 28 | 12 / 28 | 9 / 28 |

**224 of 224 spoke, and zero pairs discordant in either run.** A uniformly tidy result is evidence
the measurement is broken, so the responsible assumption was looked for before anything was
concluded: the reader does not manufacture a sentence — `say` is the model's field or empty — so the
uniformity is the model. It is also what `noticing` recorded long before, 8 of 8.

**Three things, and none of them is the pre-registered success.**

1. **It never chooses silence here**, not even when the instruction says to leave `say` empty. On
   this path the silence a person gets comes from `said_before`, which is deterministic — the model
   was never supplying it.
2. **A sentence about the person becomes a sentence to the person.** *You turned your head, I
   noticed.* *사용자가 고개를 돌리고 있어요.* The **control** drew the most of it, 18 of 28 — *You're
   here* — so it is the presence of a line about them, not what the line says. Wired as built,
   argmind would have announced that somebody looked away. That is the worst of the outcomes and it
   was caught before it shipped.
3. **Telling it to be quiet made it want to click**, 5 → 9 of 28 at baseline — `noticing`'s old
   finding, reproduced on a different axis.

So the judgement the owner asked for **cannot be delegated to this model on this path**, and the
sentence is not wired. `watch::regarding` stays, as the instrument's shared builder, with the reason
written on it where anybody would reach for it. The shipped prompt was never changed: the
counterfactual swapped the line inside the harness only, so `r322a4c` still describes what runs.

## What this does not answer

**It is a head, not a gaze.** Somebody who holds their head still and looks to the side reads under
the threshold and is not looking at the screen. The model can see that case and this cannot, which
is the one thing the expensive path would buy — and it is precisely the case the void arm was
supposed to price.

**It cannot reach p < 0.05 as a count**, the way `seeing` could not: the phases are two states. What
carries the result is the size of the separation and the order control, not a p-value.

**One person, one desk, one camera, one light**, and a threshold from one session.

**And whether attention is worth anything at all is still unmeasured.** `camera::MOVING` answers *is
somebody at the desk* in 32 ms with no model and no face detector; this answers *are they pointed at
the screen* in 99 ms. Nothing here says the second fact changes what argmind should do on a 30 s
tick, and the cheap price is a reason to find out rather than an answer.

## Re-running it

```text
cd native
cargo run -p argmind-measure --bin attending --release
cargo run -p argmind-measure --bin attending --release -- --check   # opens no camera
```

`--check` is the one to run first: it reads both models, runs the detector on a drawn gradient and
expects to find **no** face there. The full run wants a person doing both things on purpose, and
`--quick` drops the order control, which was the most valuable thing the audit added.
