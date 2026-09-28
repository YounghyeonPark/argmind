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
