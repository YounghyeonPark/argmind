# `demo/` — argmind driven by a webcam, on a PC

**Not part of the library.** Nothing under `argmind/` or `cpp/` imports anything here, and
this directory has dependencies the library must never acquire — OpenCV for the camera and
the window. That separation is the point: argmind is a library that allocates nothing and talks
to no hardware, and a demo that blurred that line would be arguing against the design it is
demonstrating.

`python -m pytest` at the repository root **does** run the tests here. This line used to say it
did not, and that was true and wrong: `testpaths` listed `tests` alone, so `test_pointer.py` had
never executed under the suite since the demo landed — present, importable, and reached by
nothing, which is the same shape as `select_newest` being compiled and called by no binary. The
tests import no OpenCV (`pointer.py` takes `import cv2` inside `main()`), so a clone without it
still collects them.

```
pip install -r demo/requirements.txt
python -m demo.pointer            # from the repository root
```

## What it shows, and why each part is here

A dot follows your hand. That is not the interesting part — a frozen model can do that. What
a frozen model **cannot** do is every line of this table, and each one exercises a different
piece of the library:

| what you see | what it is |
|---|---|
| you hold the button on yourself and drag, and it starts following | online adaptation from labels that arrive when *you* supply them |
| it also teaches itself from the input, on a declared cadence | an intermittent aiding sensor with a learned head between its runs — see the last section, including where that is *not* worth doing |
| the camera runs at ~30 Hz and your clicks do not | `Aligner` — two independent clocks, which is what it was built for |
| move the laptop or change the lighting and it re-converges | drift tracking, the original reason this library exists |
| **cover the camera** and it freezes instead of being destroyed | `Guard` — and it says which check tripped |
| uncover it and it thaws | bounded rollback and explicit recovery |
| quit and restart: it still works | `persist`, two slots with a CRC and a configuration fingerprint |

Press `h` for the keys.

## Measured

| | |
|---|---|
| backbone + pipeline, host, numpy | **931 fps** — a 30 Hz camera is not the constraint |
| position error after ~400 taught frames | **0.035**, against **0.24** for guessing the centre |
| head state | **1,160 B** (NLMS, 145 features, two outputs) |
| backbone weights | **4,960 B** (3 layers, 3x3 grid, 144 features) |
| covered camera, frame std | **0.000** against 0.194-0.197 for every legitimate scene |

`demo/test_pointer.py` drives all of it headlessly with synthetic frames and runs in the root
suite — no camera and no OpenCV, because `cv2` is imported inside `main()`. **A demo nobody
runs rots.**

## What is honest about it

**The backbone is untrained, so the accuracy is poor.** Its weights are random. Measured on a
synthetic moving blob, a shallow random backbone predicts position to about **0.23 of the
target's own standard deviation** — the dot follows you roughly, not precisely. Importing
trained weights would fix that and is a separate piece of work; the file format is already
there, since `Backbone` keeps its weights in one flat block that `persist` carries.

**The architecture is shallow and grid-pooled on purpose, and both were measured.** Random
convolutional features work shallow and carry nothing deep: predicting brightness reads 0.009
at one layer and **1.119 at twenty-seven**, which is worse than a constant. And global average
pooling discards *where* anything was, so a position task reads 1.076 globally against 0.231 on
a 3×3 grid. A deep globally-pooled stack — which is what a real MobileNet is — would make this
demo impossible with untrained weights.

**The demo's job is the library's behaviour, not the backbone's accuracy.** All six rows above
are visible whether the dot tracks well or badly.

**The guard's external check watches the raw frame's standard deviation, and two other
statistics were tried first and rejected.** Distance from a baseline embedding saturates — a
covered camera reads 1.000 against a dim room's 0.877, which is not a detector — and the
embedding's norm overlaps outright, 0.604 covered against 0.578 for a dim room. Frame standard
deviation separates completely: **0.000 covered against 0.194-0.197 for every legitimate
scene**. It is also the right *kind* of statistic for `set_external`, because the head
genuinely cannot see it: a flat grey frame produces a perfectly plausible embedding that the
head fits happily.

**You teach by dragging, not clicking, and that is a measurement rather than a preference.**
NLMS at 145 features reaches a radial error of 0.058 after 300 updates and gets nowhere after
50 — and nobody clicks fifty times.

## Keys

| key | |
|---|---|
| hold and drag | teach: "I am here now", one label per frame while held |
| `f` | freeze by hand |
| `t` | thaw |
| `r` | reset the head and forget everything |
| `s` | save to `demo/state/`, `l` load |
| `q` | quit (it saves) |

## `policy_health.py` — what a caller may *not* bound, measured

The demo's external check works because frame standard deviation separates completely. That is
not a general result, and the second thing in this directory exists to say so.

`demo/objective_sweep/` drives two reinforcement objectives to failure in opposite directions —
a rigidity term that makes *not moving* optimal, and a novelty reward that makes *oscillating*
optimal, the latter scoring **1.51× a follow-the-blob oracle on the oracle's own reward while
looking at nothing**. Both failures are plainly visible in the action stream. Neither is
detectable by a bound on it:

| statistic | part tracking | panning |
|---|---|---|
| `reversal` | working policies reverse **more** (29.7–37.5% vs 19.2–22.1%) | the working oracle reverses **0.0%**, the wiggler 25.7% |
| `stillness` | the worst row is the stillest (47.2%) | the working oracle is the stillest (61.4%) |
| `entropy` | overlapping | the wiggler reads **2.24 bits against the oracle's 1.73** |

So a low-entropy bound freezes the working policy *before* the broken one. **A policy's action
distribution does not carry the sign of its own error** — whether moving a lot is good is a
property of the task, which is exactly what a specification-error detector was supposed to
supply. Same shape as the guard's `at_most(PTrace, 1e4)` finding: one bound, two directions, and
which direction is right is not knowable from inside the guard.

What *is* detectable, with no threshold tuning, is **collapse to a constant**: 0.00 bits and 100%
stillness against ≥1.73 bits for every policy that does anything, in both environments. That is
the failure that matters in a deployment — a head that has silently stopped deciding — and
`collapsed()` is it.

One implementation note that a test found rather than a reading: the value handed to
`set_external` is **not** the raw statistic. A `Bound` is evaluated from the first update and has
no "not enough evidence yet", so raw entropy freezes the guard at **step 0** on a one-action
window. `external_signal()` resolves that on the caller's side of the slot. Third occurrence of
that shape here, after `at_least(PTrace, 1e-9)` tripping immediately and `at_most(PTrace, 1e4)`
being dead.

## Teaching from the input instead of from the mouse

    python -m demo.pointer                       # both sources; the detector runs every 10 frames
    python -m demo.pointer --teach auto           # no mouse at all
    python -m demo.pointer --teach mouse          # the original behaviour
    python -m demo.pointer --detect-every 5       # a cadence you can afford

`demo/label_sources.py` supplies the label from a second, independent computation on the same
input, so nothing has to keep dragging. The **pure** version — learn from the video alone with no
external signal — was measured and does **not** work on an untrained backbone: a linear map to the
next embedding reads a diagonal of 0.004 against an identity's 1.0, and frozen held-out error is
2–7× worse than repeating the last value. A label has to come from somewhere; it just does not
have to come from a human.

**The rule that decides whether the learned head is worth anything, measured both ways.** Scored
only on the frames where the label source is blind — the ones a caller with no learned model would
have to guess, over a moving blob:

| label source | blind by | head | hold last | centre | |
|---|---|---|---|---|---|
| `MotionLabeller` | nothing moving | 0.090 | **0.078** | 0.316 | 0.87×, the head **loses** |
| `CadenceDetector(10)` | a declared interval | **0.101** | 0.133 | 0.231 | 1.31×, the head wins |

**An aiding sensor earns a learned model only if it can be blind while the estimand changes.**
Motion energy fails that structurally, not for want of tuning: it is blind exactly when the target
is not moving, so holding its last output is right whenever it is unavailable. It is kept because
the negative result is the useful half, and because it is the first thing anyone reaches for.

**And the control that scopes all of it: a detector you can run every frame scores 0.010** —
eight times better than the head. The head is not better than its own teacher, it interpolates
between the teacher's runs. On this host the exhaustive detector costs 0.10 ms/frame against the
backbone-plus-pipeline's 0.58 ms, so for *this* task on *this* machine the right answer is to run
the detector. The case argmind is built for is a heavy detector on an NPU at 10–30 Hz with a
24-byte head between its runs, and that has not been measured here, so it is not claimed.

**Two design choices inside the detector came out of measurements that rejected the first
attempt.** It reads the **single-step** frame difference, not an accumulation between runs: an
accumulated map is about the *interval*, so its centroid is the middle of the arc travelled and
its error grows with the window — 0.048 / 0.074 / 0.121 / 0.160 at cadence 2 / 5 / 10 / 15 against
a flat ~0.036. Re-stamping the accumulated label at the interval midpoint does **not** fix it
(identical to three decimals), because with a fast reference `hold_last` pairs it with the current
features either way — the label is wrong about every instant, not mis-aligned. And it reads the
difference rather than the frame, which no synthetic-blob fixture can justify since there the mover
*is* the brightest thing; with a brighter static lamp in the scene the two separate completely,
and that fixture exists because a mutant reading the raw frame survived everything else.

The first fixture for the motion result was rigged, and the label count gave it away: the blob
teleported into its parked position, a teleport is a motion event, so the detector got one free
reading at exactly the right place.
## More than one object

    python -m demo.pointer --objects 2            # two moving things, leftmost first

**What you saw with two people in frame was not a recognition failure.** With one output pair the
label is one centroid, and a difference map with two movers has its centroid *between* them — the
demo was tracking that midpoint, and tracking it accurately (0.067 against the midpoint, 0.262
against either object). Two separate things had to be fixed.

**The label had to stop aggregating**, which is arithmetic: `find_peaks` instead of a centroid.

**And the head is permutation-symmetric**, which is the interesting half. With 2K outputs nothing
distinguishes output pair 1 from pair 2, so unless the labels arrive in the same order every frame
the least-squares answer for both pairs is the same thing. Measured over two movers, against a
centre-guess control:

| labels | crossing | scored as | head | centre | |
|---|---|---|---|---|---|
| sorted by x | no | role | **0.129** | 0.312 | 2.43× |
| sorted by x | yes | role | **0.137** | 0.222 | 1.62× |
| sorted by x | yes | **identity** | 0.251 | 0.222 | **0.88× — fails** |
| as found | no | role | 0.256 | 0.312 | 1.22× |
| as found | yes | role | 0.236 | 0.222 | **0.94× — fails** |

So peaks are returned **sorted by x**, and what that buys is **roles, not identities**: the head
learns "where is the leftmost object", a well-defined function of the current frame. It cannot
follow *which* object is which through a crossing — worse than guessing the centre — and that is
structural, not a tuning failure. Association is a question about the input's *history*, which is
exactly what the first section of `CLAUDE.md` says this model class cannot represent. Fixing it
means a tracker with state, which is a different component.

I predicted crossings would break role tracking too. They do not (0.137 against 0.129); the metric
was what mattered.

**The detector cannot count, and that is measured.** Frame differencing finds *change*, and one
object in motion is two changes — `|current − previous|` has a lobe where the object was and one
where it is. Asking for three peaks on a single blob of σ = 8.6 px:

| displacement | 0.5 px | 1.9 px | 4.8 px | 9.6 px | 19.2 px |
|---|---|---|---|---|---|
| `suppress=0.18` (default) | 3 | 3 | 2 | 2 | 2 |
| `suppress=0.45` | 3 | 1 | 1 | 1 | 1 |

The lobe separation scales with the object's **size**, not with how far it moved, so one radius
cannot serve every object size. The left column is a different failure — at 0.5 px the difference
*is* noise, and a floor relative to the peak cannot catch that because the peak is noise too.
`min_energy` covers that half and **ships disabled**, because a still scene reads 0.0129–0.0131
against 0.0177–0.0190 for slow real motion: a separation of only **1.35×**, so a margin of 2.0
refuses real motion outright. A peak-to-median gate was measured as the alternative and is
**worse** (0.97×, overlapping), because a second mover raises the map's own median.

**So `--objects K` is scoped: it separates objects that are far apart, and it cannot count.** The
two-object accuracy above is for movers half a frame apart.
