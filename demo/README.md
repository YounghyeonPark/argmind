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
## Telling the objects apart — by what they look like

    python -m demo.pointer --objects 2 --identify

Sorting labels by x gave **roles**: "leftmost" is a function of the current frame, so a head can
learn it (0.063, 3.53× a centre guess). **Identity** is not — which object this is depends on which
it *was* — and the sort rule duly failed at it (0.234 against a centre guess of 0.222).

**Appearance removes the history.** If the objects look different, the current frame says which is
which, so the head's target becomes a function of the frame again and the association moves to the
caller's side of the label. Measured end to end through the real composition:

| assignment | scored as | head | centre | |
|---|---|---|---|---|
| sorted by x | identity | 0.234 | 0.222 | 0.95× — fails |
| **appearance** | **identity** | **0.041** | 0.222 | **5.36×** |
| sorted by x | role | 0.063 | 0.222 | 3.53× |
| appearance | role | 0.242 | 0.222 | 0.92× — fails |

Each rule answers one question and fails the other, which is why both ship. A caller who wants
"leftmost" should not be given the appearance matcher.

**Four things had to hold, and finding which one was broken was the work:**

| | measured |
|---|---|
| a **random** backbone's crop embeddings separate the objects | 0.206 within/between (identical pair: 0.915) |
| nearest-template matching assigns them | **100%** correct, at every crossing speed |
| the head can express "where is the red one" | 0.072 with perfect labels — better than roles |
| the **locator** puts one detection on each object | 20.3% → **96.6%** |

The last row was the whole failure. **A frame difference is a temporal derivative**, so a mover
leaves a dipole and two peaks usually land on one object. No suppression radius fixes it here: the
dipole spacing is ~17 px for a blob of σ 8.6 px while the two objects come within ~21.6 px of each
other. Differencing against a running **background** instead pairs at 96.6%, and it is also
immune to the bright static lamp that ruled out locating on the raw frame.

**The control that says this is appearance and not something leaking.** Two *identical* objects
must fall to chance. They reach 74.3% at different heights and **50.6%** at the same height — the
residual was the crop carrying position, because a crop near a frame edge is clamped. That is why
the honest control puts them level, and why `CROP` is documented as carrying position.

**And two limits of the background locator, both measured.** It **cannot count** — `k` is a
declaration of how many slots the caller wants, not a detection, so two slots on a one-object scene
gives a second slot chasing noise. And a **stationary object is absorbed into the background and
lost at every rate**:

| px/frame | rate=0.02 | 0.05 | 0.15 | 0.4 |
|---|---|---|---|---|
| 0.0 | **0%** | **0%** | **0%** | **0%** |
| 0.5 | 91% | 93% | 0% | 0% |
| 1.0 | **100%** | 98% | 13% | 0% |
| 1.9 | 27% | 27% | 24% | 4% |
| 4.8 | 0% | 0% | 0% | 0% |

So `rate` is not a free parameter. An earlier version of that measurement reported 100% at every
rate because it asked whether *a* peak existed rather than whether the peak was on the object — a
check that could not produce a negative result. Reading the coordinates caught it: (0.89, 0.82) for
an object at (0.35, 0.50). **Three measurement bugs in this thread were all caught by reading the
numbers** rather than the code — that one, a uniform 2.00 in every cell of a swept table, and a
0.89 error on a frame 1.0 wide.

## Separating objects inside one appearance cluster — 2026-08-12

Two things were reported together: the label id changed whenever anything moved, and the regions
were not separated from each other. They had different causes.

**The flickering ids were the display's cluster-to-slot pairing, and the first thing I blamed was
wrong.** The suspicion was `FrameSegmenter.descriptors`, which standardises **per frame** while the
cluster centres are carried **across** frames, so the space moves under them. Measured over a scene
with a large subject crossing *and* a global brightness drift: **0.0% switch rate**, and identical
for a running or a fixed standardisation. Refuted, and kept as a test because it is the obvious next
thing to blame.

The real cause: `labels` was indexed by *cluster* and `slot_ids` by *slot*, and the chain between
them destroys the correspondence — the background is dropped **by area**, the survivors are
re-sorted **by x**, and the appearance matcher **permutes** them again. Cluster *n* was coloured
with the id of slot *n*. `slot_label_map` carries the cluster through, and the dropped background
cluster now stays 0 instead of being painted, which is also half of "the regions are not separated":
a cluster map tiles the whole frame by construction.

**The separation itself needed a new mechanism, and connectivity is most of it.** A cluster is a
region of similar appearance, not an object, so two objects that look alike land in one cluster and
its centroid sits *between* them. `split_regions` splits each cluster into its spatially separate
pieces. Mean best IoU against ground truth, two same-coloured discs held apart: **0.50 by cluster,
1.00 by region.**

**The edge gate earns its place only where `k` is under-declared, which is the normal condition** —
`k` is a declaration in every locator here and nothing counts the appearance groups in a room.
Scored on the two objects a caller declared, with distractors eating the cluster budget:

| scene, caller declared 2 so k=3 | clusters | +connected | +edge-gated | k it would have needed |
|---|---|---|---|---|
| 2 targets touching + 1 distractor | 0.83 | 0.83 | 0.83 | 3 |
| 2 targets touching + 2 distractors | 0.60 | 0.60 | **0.83** | 5 |
| 2 targets **apart** + 2 distractors | 0.53 | 0.53 | **1.00** | 4 |
| 2 targets touching, no distractor | 0.83 | 0.83 | 0.83 | 3 |

The third row is the mechanism: a target that shares a cluster **with the background** is connected
to it, so connectivity alone does nothing and the step at the object's own rim is what severs it.
**My first conclusion was that the gate buys nothing over raising `k`** — 0.82 against 0.83 — and it
came from a fixture set with no distractors, where `k` was adequate by construction. The controls in
the test are what stop that recurring: the gate must *not* help where `k` is already enough.

**The end-to-end number is less flattering than the IoU table, and both are recorded.** Distance
from each slot's taught centre to the target it should track:

| scene | clusters | +connected | +edge-gated |
|---|---|---|---|
| same colour apart | 0.213 | **0.000** | 0.000 |
| same colour touching | **0.053** | 0.188 | 0.187 |
| touch + 2 distractors | 0.183 | 0.183 | 0.183 |
| apart + 2 distractors | 0.179 | 0.179 | 0.179 |
| mean over seven | 0.092 | **0.081** | 0.081 |

**Splitting costs on the case nothing can separate** — two identical touching discs are one region,
so the second slot gets a background fragment: 0.053 becomes 0.188. And **the edge gate does not
improve what the head is taught at all.** It takes region IoU from 0.53 to 1.00 and moves the taught
centre by 0.000, because it changes a region's *shape* and not which region wins by area. It is a
segmentation-picture improvement and is claimed as nothing more — which is the right claim, since
the picture is what was reported as wrong.

**What nothing here separates: two identical objects that touch.** 0.60 for clusters, connectivity
and the gate alike, at every `k` tried. There is no edge between them to find.

**A fixture caught a real defect in my own selection rule.** Dropping the largest *region* as
background is right while regions are clusters, and false once they split: on two same-coloured
discs at k=3 the noisy wall occupies two clusters and fragments into **twelve regions of 76–191 px**
while each object is **353 px**, so the objects became the largest regions and the rule dropped one
of them. The background is dropped by **cluster** now. The limit that leaves is stated rather than
tuned around: a background of two large contiguous surfaces puts one of them in a slot at
`k = objects + 1`.

**Two things about the implementation, both measured after being claimed wrongly first.**
Components are found by hooking to the smallest open neighbour and then **jumping pointers**, which
is 4 rounds on a disc and 20 on a region that snakes across the frame; plain neighbour propagation
needs one pass per pixel of the *diameter* — 167 — and lost to a Python union-find. And the split
runs **one pass over the whole field** rather than one per cluster, since "same cluster and not
gated" is the same adjacency; that is verified region-for-region against a per-cluster reference and
is **1.5–2.1× faster on five of seven fixtures and a wash on the other two**, not the third I
predicted, because the cost is hook rounds and sharing passes does not reduce them.

Cost, this host, 72x96, gated: **1.5–2.0 ms** on five fixtures and **13.4–14.3 ms** on the two with
a noise-carved background — 43% of a 33 ms frame at worst, still clearing 30 Hz with the segmenter's
2.5 ms. `scipy.ndimage.label` is 0.05 ms and is deliberately not used: it cannot express a gated
join, so it would be a second implementation for the ungated case only.

**And the whole selection path was inside `main()`, where nothing could reach it** — the fourth
occurrence of that shape in one day, after `select_newest` compiled and called by no binary,
`drive`'s `teach_every` that no caller ever passed, and a test that reimplemented the assembly it
was checking. `slot_label_map` and `regions_for_slots` are module-level now and the tests drive
them. Eighteen mutants, all caught; two of them were real gaps rather than confirmations — a
region's centre computed from its whole cluster survived every mask-IoU test in the file, because
the centres are what the head is taught with and nothing compared them, and a rule keeping the
*smallest* regions survived every fixture where `k = objects + 1` makes the two rules equivalent.

## The picture is not the head's slots — 2026-08-12

"전보단 나은데 라벨수가 적은것 같음". `--objects 3` bounds the head at three because that is its
output width. A picture has no such limit, and the resolution recorded for the vocabulary question is
*unbounded vocabulary, bounded concurrency* — so `label_field` paints every region and
`RegionTracker` gives each one a label that survives between frames.

| identity | coverage | purity | distinct | ids per object |
|---|---|---|---|---|
| raw region index | 1.00 | 0.66 | 1.00 | 2.7 |
| cluster index | 1.00 | **1.00** | **0.00** | 1.0 |
| **tracked** | 1.00 | **1.00** | **1.00** | **1.0** |

Three objects, two the same colour, k=4. **The cluster index scores a perfect purity and is
useless** — two objects wearing one id are each perfectly pure — which is why distinctness is
measured beside it and why purity alone cannot fail.

**The labels come from splitting, and `k` is close to irrelevant** — five objects give 5.0 labels at
purity 1.00 over 5 minted ids, identically at k = 4, 6, 8 and 10, because splitting separates the
objects that share a cluster. An earlier version of this section said raising `k` was *worse* (3.0
labels at 0.45 over 43 ids); that was the background-selection defect below and is **withdrawn**.

**Two floors on one split, and neither number can do both jobs.** The split's floor feeds the head:
0.01 → 0.03 takes the taught centre on two same-coloured objects from 0.000 to **0.222**. The
picture's floor keeps fragments out: 0.01 paints 7.0 labels at 47% on-object, 0.03 paints 3.2 at 95%,
0.05 paints only objects but loses them (coverage 0.98) and re-mints labels (purity 0.83).
`--seg-min-frac` and `--seg-show-frac`.

**A minimum age was the mechanism I expected and it does not work**: it helps only when the floor is
too low, and once the floor is right it costs coverage for precision. Not offered.

**Two of my own fixtures collided while a comment claimed they did not** — 0.148 against a touching
threshold of 0.220, and 0.112 against 0.180. Every number here was re-measured on separate orbits and
the fixture asserts its own separation now. Nine mutants, all caught.

## Keeping a label across an object hiding and coming back — 2026-08-13

A vanished label waits `--label-keep` frames in a dormant pool and is reclaimed by what a region
looks like. Both failure directions are scored together, because either alone can look perfect:

| | position coasting | appearance |
|---|---|---|
| hides, returns to the same place | recovers | recovers |
| hides, returns **somewhere else** | **fails** | recovers |
| both hide and **swap places**, distinguishable | **0/8** | **8/8** |
| a different-looking object arrives in its place | **steals the label** | refuses |
| an identical object arrives in its place | steals | steals |

**Position coasting is refused rather than unbuilt.** It recovers only the case where nothing moved,
and it hands a dead object's label to whatever arrives there — the reason `RegionTracker` held a
single frame to begin with.

**The limit: two identical objects that hide together and swap come back exchanged, 1/8** — worse
than the 4/8 a coin flip gives, because tied descriptors are broken by the assignment's ordering,
which follows position. Identity through a swap while both are hidden is history-dependent and there
is no history to have; the feedback bank cannot help either, since during the blackout there is
nothing to integrate.

`--label-keep` is a real forget window: at a gap past it, nothing recovers. `--label-tol` is how
different a region may look and still reclaim its label.

**The descriptor is masked, not a crop.** A crop near a frame edge is clamped and carries position,
which inflated an identity control from 50.6% to 74.3% here once before. The same object at the edge
and in the middle now describes 0.0000 apart.

**Two measurement errors of mine, both caught by a row that could not be true.** The
false-inheritance fixture put the newcomer at the hidden object's place *during* the gap, so nothing
ever vanished and every mechanism — including no memory at all — read as stealing. And I called the
identical-twin recovery chance from one non-monotone row; over eight seeds it is 6/8, 7/8, 7/8,
because the twin never vanished and one-to-one assignment has already consumed it. Seven mutants, all
caught.

## Why a region goes unlabelled — 2026-08-13

"라벨링이 안되는 영역이 있는데 라벨링갯수가 고정되어서 잘 안되는거같음". **The count was never fixed** —
it paints 4 labels for a declared 3 and 5 for five objects at k=4. A region was dropped for two
reasons that have nothing to do with counting: sitting in the **background cluster**, which happens as
soon as a scene has more appearance groups than `k` allows, and being under the **area floor**.

Four mechanisms measured, three refused:

| mechanism | result |
|---|---|
| raise or adapt `k` | **changes nothing**: covered 0.80 at k=4, 6 and 8 alike. (An earlier row here claimed it was non-monotone and *worse*; that was the background defect below, and is withdrawn) |
| merge close clusters | folds the small object back into the wall, covered 1.00 → 0.80 |
| drop the largest *region* rather than the background *cluster* | rescues it (0.80 → 1.00), destroys a noisy scene (precision 0.95 → **0.26**) |
| **ring contrast** | covered **0.80 → 1.00**; precision 0.99 → **0.92** on a noise-carved wall |

Contrast is the only *local* statistic in the list, which is why it is the one that works: a real
object differs from the band just outside it, a wall fragment is a piece of the wall's own
distribution. Object floor **0.136** against non-wall ceiling **0.118** over three scenes — a
**1.15×** interval, so `--seg-contrast` carries its numbers, and 0 restores the old rule exactly.

Inside the background cluster a region is admitted only by contrast, never if it is the largest piece
of that cluster. Outside it, the floor and contrast are alternatives.

**Two measurement errors of mine.** The contrast probe ran without the noise the sweep added, so it
claimed a 0.5-against-0.01 gap where matched noise gives 0.519 against 0.118 — the conclusion
survived, the margin did not. And the first reproduction scored a uniform 1.00 for every scheme, which
is the signal that the measurement was broken: three distinct objects separate cleanly whatever you
do. Nine mutants; the two survivors were real gaps, including `MIN_REGION_PX` being dead at the
shipped floors.

## Does the count really vary? — 2026-08-13

Yes, and asking found a defect with a common trigger. Objects entering and leaving give a true count
of 1,2,3,4,3,2,1,0 and the painted count agrees on **30 of 30 frames**, minting exactly **4** ids for
4 objects. `k` and `--objects` are fixed for the run; the count is not.

**The defect**: `label_field` picked its background from the clusters that had produced a surviving
region. On a clean wall the wall's clusters produce none, so the object's own cluster became "the
largest present" and was dropped — a frame with exactly one region, the object's own at contrast
0.519, painted nothing. A lone object on a plain background is now labelled on 12 of 12 frames.

**It also invalidated a recorded finding.** "Raising `k` is worse" was this defect, not `k`. Five
objects now give 5.0 labels at purity 1.00 over 5 ids at every k from 4 to 10, and the crowded scene
covers 0.80 without contrast and 1.00 with it *at every k*. So `k` is close to irrelevant and
contrast is the mechanism — the same conclusion, for a correct reason.
