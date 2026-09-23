# Seeing a room

**What a vision model can say about the actual room the camera is in, and at what size.** Measured
2026-09-23 with `argmind-measure --bin seeing`, `qwen3-vl:8b`, one machine, one room, one person.

The short version, and it is three different answers:

| what is asked | the smallest size that gives it | how it read |
|---|---|---|
| is somebody there | **160×90**, 43 kB | 16 of 16 |
| a named object — a yellow cup | **160×90**, 43 kB | 16 of 16 |
| are they looking at the screen | **320×180**, 173 kB | 4 of 4 at 320 and above; **1 of 4** at 160×90 |
| are their eyes open | **no size measured gives it** | 1920×1080 said *눈을 감고* on 2 of 3 asks and the eyes were open |

**So the number is 320×180** — and the arm above it is not the safe choice, it is the one that made
something up.

## Why this measurement needed a door built for it

`camera.rs` reduced every frame to 64×36 greys and had **no accessor for anything larger**. That was
not a promise in a comment, it was a property of the type: [`Frame`] holds `WIDE`×`TALL` bytes and
there is nothing else to ask for. At that size a room is light and shadow, which is all
`camera::MOVING` ever needed.

Asking this question means giving that up. `Camera::photograph(wide, tall)` exists now, **takes the
size as an argument** so a reader can see in the source how much of a room a caller wants, and what
stands where the type stood is a rule read off the source —
`argmind-rules/tests/the_app_takes_no_photograph.rs`: **the application cannot take a photograph at
all**, only a harness a person runs on purpose. Two code breaks are caught, a `photograph(` and a
bare `::shrink(`; a commented-out one is not, deliberately, because a rule that forbids writing down
what it forbids is one nobody can document.

**Nothing is written to disk.** The captions reach the terminal and the picture dies with the
process. A stored photograph is exactly what would let somebody check a caption later, and that is
the artefact this is trying not to make — which also means the only ground truth available is the
person who was in front of the lens, and both runs were read back to them.

## The conditions

`"Integrated Camera"`, 1920×1080. Opening it takes **3,312–3,344 ms** of which **3,004–3,007 ms** is
`settle()` waiting for the picture to stop moving; the photograph itself is **42–44 ms**.

**One photograph per run, shrunk for every arm.** Four captures would be four instants and *the
small arm missed it* would be indistinguishable from *the person moved*. The top arm is the sensor's
own size, read at runtime: a fixed 1920×1080 against a smaller sensor would be an **upscale**, every
arm would be the same picture at different byte counts, and *resolution does not help* would come
out true for free.

| run | photograph | asks | the room, as its occupant reported it |
|---|---|---|---|
| 1 | one | 4 (one per size) | looking at the screen |
| 2 | one | 12 (three per size) | **not** looking at the screen, eyes open |

The second state was luck — the owner happened to look away — and that is the weakness to fix before
this is leaned on. A deliberate pair is ten seconds of somebody's time and has not been spent.

| size | PNG, stored blocks | `prompt_tokens` |
|---|---|---|
| 1920×1080 | 6,222,418 B | **2102** |
| 640×360 | 691,678 B | 1137 |
| 320×180 | 173,058 B | 1137 |
| 160×90 | 43,358 B | 1137 |

**The tokens do not fall with the pixels below 640×360** — 1137 for all three, reproducing the
earlier reading of 1110 from 640×360 down to 80×46. Only the full-size arm buys more tiles, and what
it buys with them is below.

## The negative control

**0 of 3.** The same question about three pictures with no room in them — flat grey, uniform noise,
and the four-coloured-square test card the original 8 s figure was measured on. It opens no camera,
so it runs with nobody present.

| picture | seconds, warm | what it said |
|---|---|---|
| flat grey | 6.8 | 사람이 **없으며**, 눈에 띄는 물건은 회색 배경 |
| uniform noise | 3.0 | 사람이나 눈에 띄는 물건이 전혀 **보이지 않으며** |
| four squares | 5.9 | 사람이 **없으며**, 빨강·녹색·파랑·노랑으로 나뉜 깃발 |

This control exists because **most of a correct caption is guessable.** The question names the
things to look for, and in front of a webcam *somebody is there, facing the screen* is what a model
would answer whether or not it can see. Without this, the room run says nothing.

The test card naming its four colours correctly — and reading them as a flag, which is not a wrong
reading — is the other half: the model is not merely declining, it is describing.

**The harness's own counter was wrong and the run showed it.** It knew one Korean negation, `없`, so
*사람이나 눈에 띄는 물건이 전혀 보이지 않습니다* was scored as a person and the control printed **1 of
3**. The sentence was printed beside the count, which is the only reason a wrong count was visible;
a harness that printed the tally alone would have reported a failed control. The negations are
written out now.

## What the two runs found

**Presence and the object hold everywhere.** A person: 16 of 16. A yellow cup, which the occupant
confirms was there: 16 of 16. Neither *yellow* nor *cup* is in the question — it asks for "one
noticeable object" — so this is the token that could not be guessed, and 43 kB is enough for it.

**The gaze is where the sizes part.**

| size | run 1 (looking) | run 2 (not looking, ×3) | correct |
|---|---|---|---|
| 1920×1080 | 보고 있다 | 보고 있지 않다 ×3 | 4 of 4 |
| 640×360 | 보고 있다 | 보고 있지 않다 ×3 | 4 of 4 |
| 320×180 | 보고 있다 | 보고 있지 않다 ×3 | 4 of 4 |
| **160×90** | 보고 있다 | **보고 있습니다 ×3** | **1 of 4** |

**160×90 gave the same answer in both states.** A size that answers identically whether or not the
thing is true is not reading it, and the one case it scored is the one where the prior happened to
match. Everything from 320×180 up tracked the change.

**And the largest arm invented a detail.** 1920×1080 added *눈을 감고* — eyes closed — on 2 of 3
asks. **The eyes were open.** It is the only claim in sixteen captions that is false, it appears
only at the size with twice the tokens, and it is confidently specific. More pixels did not buy more
truth here; they bought a sentence with more in it, and the extra was wrong.

This is the result that decides the recommendation. 320×180 gets the gaze right and does not reach
for the eyes; 1920×1080 costs 36× the bytes, 1.85× the tokens, and adds a fabrication.

## What it does not cost

**Not the wire, and not the resolution.** Warm, every size lands in the same band:

| size | seconds, warm |
|---|---|
| 1920×1080 | 4.1, 4.5 |
| 640×360 | 5.2, 4.2, 4.2 |
| 320×180 | 5.1, 5.4, 5.5 |
| 160×90 | 4.6, 4.1, 4.1 |

Full resolution at 6 MB and 2102 tokens is **not distinguishable** from 160×90 at 43 kB. The spread
inside one size is as large as the spread between sizes, so the seconds are how much the model
generates, not how much picture it was handed. The control says the same from the other side: three
pictures at an identical 1137 tokens took 3.0, 5.9 and 6.8 s.

**An earlier reading of this output was wrong and is withdrawn.** The first run read 17.2 s at full
resolution and the second 13.9 s, and that was written up as the doubled token count. It was the
model arriving on the graphics card: the arms run largest first, so the first row carried the load,
and the same arm's own second and third rounds read 4.1 and 4.5. `--control` already had a throwaway
warming ask and the room path did not; it does now, and prints what the load cost — **2.3 s**.

## What this cannot claim

**It cannot reach p < 0.05 and no arithmetic will fix that.** The gaze question is binary, so a coin
scores 4 of 4 one time in sixteen. The strongest honest statement is the *pattern*: three sizes
tracked a change of state and one did not, on one photograph per state.

**One room, one person, one device, one light.** Nothing here says what happens with two people, a
dark room, or a laptop camera pointed at a ceiling.

**The second state was accidental.** A deliberate pair — hold still looking at the screen, then look
away, a photograph in each — is what would turn this from a pattern into a measurement, and it is
the same ten seconds `presence` already asks for.

**Nothing measures the thing argmind would actually use.** *Is somebody at the desk* is answered by
`camera::MOVING` at 32 ms and 64×36 greys; this route costs 4–5 seconds and a model call. What it
adds is **attention**, and whether attention is worth 5 s on a 30 s tick is a different question
that no number here touches.

## How to re-run it

```text
cd native
cargo run -p argmind-measure --bin seeing --release
cargo run -p argmind-measure --bin seeing --release -- --repeat 3
cargo run -p argmind-measure --bin seeing --release -- --control      # opens no camera
cargo run -p argmind-measure --bin seeing --release -- 320x180
```

`--release` because the largest arm hands 6 MB to a hand-written base64, and in a debug build that
shows up in the seconds as if the model had spent them. **Stop argmind first**: it holds `qwen3:8b`
on the same card, which is what cost `crowding` thirty-five minutes.
