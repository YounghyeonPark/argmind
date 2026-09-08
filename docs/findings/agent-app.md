# The agent app: an LLM on the PC, with argmind's memory as its cache

Asked 2026-09-08: an agent that works with an LLM inside the I/O environment a PC gives — screen,
webcam, microphone, keyboard, mouse in; screen, keyboard, mouse, speaker out — that interacts with a
person, and *"캐시를 만들어서 반복적인 작업은 점점 빠르게 동작할수 있게끔 할것."*

`agent/` is that app. The LLM supplies language and judgement and nothing here competes with it.
What it does not supply is a memory that outlives a call and gets **cheaper** with repetition, and
that is the part argmind already had.

## The cache key was measured before the cache was built

A situation has to come back looking like itself or there is nothing to cache. 117 real screen
captures over 60 s, three candidate keys:

| key | consecutive frames | > 10 s apart | gap |
|---|---|---|---|
| **backbone, 144-d** | 0.9882 ± 0.0620 | 0.7579 ± 0.1501 | +0.2303 |
| 32×32 grey pixels, 1024-d | 0.9917 ± 0.0411 | 0.7290 ± 0.1665 | +0.2627 |
| 64-bit dHash | 0.9604 ± 0.1479 | 0.1817 ± 0.4407 | +0.7787 |

The means are not the finding — the **tail** is. Of pairs more than ten seconds apart, **20.4 %
score ≥ 0.995**, and the share barely moves from 0.95 to 0.995 (28.0 → 23.5 → 20.4 %). The
distribution is two-humped, so a threshold exists between the humps.

**This is the opposite of what the same machinery does on a camera.** The same store, the same
descriptor family, a moving object under load: **0 of 25**. A screen is a domain where this works,
and the difference is that windows repeat pixel for pixel while a mug on a desk never does.

The backbone and the raw pixels measured within noise of each other, so choosing the backbone is
architectural and not empirical — 144 values against 1,024 is what `ExemplarMemory` stores per
entry. dHash separates far better and only 1 % of far pairs reach 0.98: it tolerates nothing, and a
cache that only hits on an exact repeat is a dictionary.

## The threshold is 0.995 for an asymmetry, not an accuracy

A miss costs a second of latency. A **false hit moves the real mouse and types into the real
window**. Tightening from 0.95 to 0.995 gives up about a quarter of the hits and buys most of the
safety, which is only a sensible trade because those two costs are not comparable.

## What belongs in the KEY and what belongs in the CONTEXT — and my reason was wrong

The spec lists the webcam as an input, so the default reading is that the situation is screen and
webcam together. Measured over 119 observations at 2 Hz instead:

| key | entries | any hit | recurrent (> 10 s) |
|---|---|---|---|
| screen | 23 | 80.7 % | **21.0 %** |
| screen + webcam | 44 | 63.0 % | 17.6 % |
| webcam | 9 | 92.4 % | **61.3 %** |

**The webcam alone looks like the best key and is the worst.** Nine entries for a 75-second run
means it can barely tell one moment from another — high invariance, no discrimination, which is the
global-pooling failure recorded in `vocabulary-and-labelling.md` turning up in a new place. Keying
on it would replay the same action whatever was on screen.

**My stated reason is withdrawn.** I predicted the webcam would be too *unstable* to key on, because
a face never holds still. It is **more** stable than the screen here: the camera sees a mostly static
room while the terminal scrolls. The criterion was never stability — it is whether the key separates
situations that need different actions. The conclusion survived; the reasoning behind it did not, and
those are different things.

So the webcam and the pointer state are **context for the brain and never part of the key**, and
they are read **only on a miss** — grabbing a camera frame on the fast path spends the saving the
cache exists to produce.

## Measured on the real screen, through the app itself

    103 observations over 60 s, 5 distinct situations remembered
    any hit         95.1%   includes the screen simply not changing
    RECURRENT hit   58.3%   matched something first seen over 10 s ago

**The first number flatters and the tool used to print only that.** A hit on the frame half a second
ago is a still screen, not a repeated task; an idle terminal reads 94.6 % and means nothing.
`python -m agent --measure` now prints both and says which one matters. Both are upper bounds: a
cached decision also has to still be the *right* one, which no observation-only mode can tell you.

## Two bugs the tests found, and both were the same shape

**Validation lived inside the live branch**, so a dry run accepted `{"do": "key", "name": "f4"}`
silently — and the loop *stores what it accepted*, so a malformed action became a cache entry that
would fail the first time somebody ran with `--live`. A bad action executed once is a bad turn; a bad
action cached is a bad turn that repeats faster every time it hits. Validation is now independent of
execution and the loop refuses a decision **before** storing it.

**And the first timing test measured the brake, not the cache.** Comparing whole steps compared the
actuator's rate limiter, which is charged to hits and misses alike — and on a miss the brain's own
latency absorbs part of that wait, so a hit came out **slower**: 100.4 ms against 53.8 ms. The number
was real and said nothing about the cache. The comparison is now on the deciding, which is the part
the cache touches, and the counted assertions are what hold the result.

## Ten mutants, nine caught, one equivalent

Caught: the webcam joining the key, the extra eyes read on every step, a decision cached without
validation, a `verify` that never invalidates, a look-up that always matches, an `invalidate` that
keeps the value, a value written to the wrong slot, actuation defaulting to live, and a brake checked
per batch instead of per action.

Survived and **recorded rather than contrived around**: removing the normalisation in
`SituationSense.describe`. `ExemplarMemory._sparse` unit-normalises the top-k of both the stored slot
and the query, so the memory is scale-invariant and no cache decision can turn on that norm. It stays
because `Situation.key` is public and callers compare keys directly — the measurements above did.

## What is absent rather than unfinished

| | |
|---|---|
| screen in | built (`PIL.ImageGrab`, 34 ms for 2560×1440 here) |
| webcam in | built (`cv2`) |
| keyboard / mouse in | built (`user32`, sampled — not hooked; an agent that swallowed the user's input would be a keylogger with extra steps) |
| keyboard / mouse out | built (`user32` `SendInput`), **dry-run by default** |
| **microphone in** | **absent** — no capture library in this interpreter |
| **speaker out** | **absent** — same reason |
| LLM | `ClaudeBrain` written against the official SDK and **never executed**: no package, no key here |

`ClaudeBrain` is untested code and says so in its own docstring. It imports the official SDK lazily
and fails with the install line rather than reaching for a raw-HTTP shim because `httpx` happens to
be installed — a hand-rolled client is a second thing to keep correct against an API that changes.

Actuation defaults to `live=False` behind three brakes: the flag, a rate limit, and a guard callable
checked before *every* action rather than once per batch, so "the human touched the mouse" stops the
burst that is running.

## What has not been measured

Whether a cached decision is still the **right** one. Every number above is about whether a situation
recurs, not whether the stored action still applies to it — and those come apart exactly when a
window has moved but still looks the same. `verify` is the hook for finding out and nothing here
supplies one. The recurrence figures are also one screen on one machine over one minute; a working
day is a different number, which is why `--measure` exists.
