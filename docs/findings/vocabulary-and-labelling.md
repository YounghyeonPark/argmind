<!-- Moved out of CLAUDE.md on 2026-08-16 so the file loaded into every session stays
     small. The text is verbatim; nothing was summarised or dropped. CLAUDE.md carries a
     one-line pointer to each of these. -->

# A vocabulary that grows automatically, and everything that blocked it — 2026-08-11/13

## A vocabulary that grows automatically — assessed and partly built, 2026-08-11/12

**The owner's proposal**: the network should *memorise* objects, segment them perceptually, and the
recognisable-label count should then scale automatically. Two thirds of that is supported by
measurement; the third collides with a constraint and the way round it was **refuted**.

**The constraint that shapes everything here**: `RecursiveLeastSquares.widened()` **refuses** to
change `n_outputs` — theta is row-major over features, so a different width interleaves the old
coefficients. A growing label set therefore cannot be a growing head. So the resolution is
**unbounded vocabulary, bounded concurrency**: a caller-side dictionary that grows freely, and a
head with a declared number of slots. `ObjectMemory` is that dictionary.

**Putting the identity on the head's INPUT instead does not work, and one refutation was my own
prediction going the wrong way.** Against a per-slot head at **1.71–1.83x** a centre guess, flat
from 2 to 5 objects:

| form | 2 objects | 3 | 4 | 5 |
|---|---|---|---|---|
| per-slot, 2K outputs | 1.73x | 1.83x | 1.72x | 1.71x |
| `[embedding, query]` concatenated | 1.10x | 1.03x | 1.01x | 1.00x |
| query correlated against the feature map | 1.08x | 1.04x | 1.01x | 1.01x |

The concatenation was *predicted* to fail — locating a pattern is a multiplicative interaction and
a linear map on a concatenation has no product term. **The correlation was predicted to work and
does not**, and the cause is measured rather than assumed: the correlation map's argmax lands on the
queried object only **4–50%** of the time under any similarity (cosine 17.5/10.8/4.2%, plain dot
2.5/5.8/50.0%, negative distance 40.0/15.8/18.3%). A random backbone's per-cell channel signature
does not identify an object. **So per-object segmentation by correlating a memory against the
feature map is refused here, not merely unbuilt** — and it is the *fourth* independent line of work
this session to end at "this needs trained backbone weights".

**What the dictionary does, measured over 1200 frames**: 2 objects → 4 ids, **96.7%** correct
matching; 3 → 7 ids, **66.5%**; 5 → 12 ids, **43.1%**. It works for two, is mixed at three, degrades
past that, and **inflates about 2x** because the same object seen where its crop clips at a frame
edge describes differently. Automatic growth is real, not free, and not exact.

**`tol` is tuned on the outcome and the first version of its docstring claimed otherwise.** The
derivation would be the gap between within-object and between-object descriptor distance, but that
gap **moves with the scene** — measured on the crops the registry actually sees, (0.081, 0.225) at
two objects and (0.131, 0.288) at three. There is no single interval, so 0.25 is a compromise over a
2/3/5 sweep. A test pins *why* it cannot be derived, and it was a failing assertion — the claim that
0.25 sat inside "the" gap — that caught the overreach. What is principled is the shape of failure
either side: at 0.04 it mints **101/282/917** ids for 2/3/5 objects, one per detection; at 0.60
everything collapses to one id.

**Two more of my own errors, both caught by numbers rather than code.** A registry accuracy of
**0–30%** across every threshold, which was a *metric* bug: it compared the id index against the
true object index, and those coincide only when ids are minted in order and there are exactly as
many. Tracking which object each id was minted from gives 43–97%. And an exploratory 85.3% at three
objects became **66.5%** on the shipped path, because the scratch script only updated the background
on cadence frames — the higher number came from a slower-adapting background, not better matching.

**And a mutation check found a real aliasing defect**: `observe` used `np.asarray`, which returns the
*same object* for a float64 input, so the memory aliased the caller's descriptor and anything they
did to it afterwards silently rewrote the store. The surviving mutant was `descriptor()` handing out
the live array — it survived because the test compared against an array the store was also holding.
Same class as the `shares_memory` defect recorded for `Adapter.state`. Both copy now, six mutants
caught.

**Bottom-up grouping of the minted ids into higher-order labels — the owner's next proposal,
measured and refused on this backbone (2026-08-12).** The idea is right in shape and is the way to
an automatically scaling vocabulary without trained weights, and there was a sharp test for it
already in the data: `ObjectMemory` inflates ~2x, so if those ids merge on their own the proposal
works *and* the inflation is fixed. Five things were measured and four are negative.

**Grouping helps at two objects and hurts at three.** Clustered at the true k: **98.1%** purity at
two against the online rule's 90.4%, but **65.4%** at three against the online rule's **91.5%**. So
the instances of one object do not form one cluster — they form several *individually pure*
fragments, and a fragment of one object can sit closer to a fragment of another than to its own.
The idea is not wrong in principle; the descriptor is not good enough for it.

**The number of groups cannot be found automatically**, which is the part "scalable" needs. A single
distance cutoff, groups found (purity): at 0.10 → 19 (99%) / 60 (97%) / 133 (82%) for true 2/3/5; at
0.20 → 4 (98%) / 11 (95%) / 12 (60%); at 0.25 → 1 / 5 / 4. Nothing reads 2 and 3. High-purity
cutoffs give an order of magnitude too many groups; low ones destroy the distinction.

**More observations make it worse, which inverts the assumption the proposal rests on.** Purity at
n = 50/100/200/400: two objects 100.0/99.0/97.5/98.8%, three objects 94.0/68.0/95.5/**65.2%**. More
views mean more edge-clipped ones, so within-object spread grows faster than evidence does.

**The "commonality" mechanism specifically does not hold**: a group centroid is no better than one
instance — 1.01x, 1.00x, 0.94x. Averaging view-dependent descriptors averages the views.

**And co-occurrence, an appearance-free cue, fails for a reason that names the real bottleneck.**
Two ids never seen together are candidates for being one object; measured, same-object pairs
co-occur **57.2%** (max 98.7%) against different-object pairs at 72.8% (min 0.0%) — overlapping. The
premise is violated by the **detector**, not by the world: it cannot count, so it puts two
detections on one object and both ids appear in the same frame legitimately.

**The through-line, and it is now strong enough to state as a conclusion.** Five independent routes
— per-object correlation segmentation, identity on the head's input, an unbounded id registry,
bottom-up grouping, and co-occurrence merging — are all blocked at the same two places: **an
untrained backbone's crop descriptor conflates identity with view, and the classical detector
cannot count.** Nothing in the adapter, the guard, the pipeline, the aligner or the head is the
constraint. **Importing trained backbone weights is the next piece of work**, and the format for it
already exists — `Backbone` keeps its weights in one flat block with `state`/`load_state`, and the
layouts are TFLite's for exactly this reason.

**Raw pixels beat the random backbone on every axis this demo exercises (2026-08-12), which is
not what I expected.** The owner asked why the frame is not simply fed in. Position, one object,
perfect labels:

| input | features | head state | ms/frame | error | vs centre |
|---|---|---|---|---|---|
| backbone | 145 | 1.1 KB | 0.94 | 0.064 | 4.55x |
| **raw /8** | **109** | **0.9 KB** | **0.42** | **0.063** | **4.58x** |
| raw rgb | 20,737 | 162.0 KB | 0.73 | 0.062 | 4.72x |

A 9x12 block-mean downsample is **fewer numbers** than the backbone's 144, half the time, and no
worse. **Frozen** — which separates tracking from generalising — raw reaches **0.001 (215.8x a
centre guess)** against the backbone's 0.027 (10.6x). At three objects raw is 0.070/4.00x against
0.077/3.65x, and raw RGB crops separate appearance at **0.096** against the crop backbone's 0.177.

The reason is mechanical: position is a linear readout of a spatial map, and `sum_i w_i·pixel_i`
with coordinate weights is the centroid times the total brightness. The backbone pools to 3x3 and
mixes channels randomly, which for this task can only lose information.

**What a backbone buys is invariance, and a random one does not buy it.** Trained under one
condition, frozen, then tested under another: brightness x1.6 gives 0.9x for the backbone against
0.7x for raw, a +0.25 offset 1.0x against 0.8x — marginally better and **both at a centre guess**,
i.e. both failed. Under clutter raw wins outright, 1.6x to 1.4x. So this strengthens the
trained-weights conclusion from a new direction: the case for a backbone is *entirely* conditional
on it being trained. `--input raw|backbone`, and the source is in the persistence fingerprint.

**The first invariance fixture measured nothing**: with NLMS the frozen control read 0.89x, worse
than guessing, so the freeze dominated the condition change. RLS converges to the least-squares
solution and its control reads 10.6x and 215.8x. Fifth broken fixture caught by a control today.

**Label regions are visible now** (`--segments`), and they are a partition of a mask rather than
learned segmentation — the distinction is measured, not stylistic. `BackgroundLocator` already
computed `|frame - background|` and discarded it; keeping it costs nothing. Against ground truth,
one object: IoU **0.76** at `floor=0.40` (precision 1.00, recall 0.76), 0.71 at 0.25, 0.42 at 0.15
(recall 0.99, precision 0.42) — a trade with no free setting, so it is a flag. Quality falls to
0.55 at two objects and 0.46 at three, and a **stationary object has no region at all**, which is
the locator's absorbed-into-the-background limit made visible. Learned per-object segmentation
stays refused at 4–50% localisation; nothing here claims otherwise. Four mutants, all caught.

**"Why does the segmentation only label motion edges?" — the owner's observation, and it has a
precise cause with a two-sided fix.** The background is a running mean, so it **absorbs** anything
that stops moving: the interior stops differing from the model and only the boundary, where the
object is newly arriving, is left. Measured on a flat-topped object at 0.05 px/frame, coverage of
the object's **core** is **0.00** plain against **0.31** with a selective update, and of the whole
object 0.10 against 0.40.

**The blob fixture could not show this at all** — a Gaussian has no uniform interior to absorb, and
core/rim stayed at ~0.9 whatever the rate. A flat disc was needed, and that is why the first
measurement of this question came back saying nothing was wrong.

**The predicted objection was ghosting and it is measured to be wrong**, which is worth keeping so
it is not raised again: sixty frames after an object leaves, the frame fraction still labelled is
**0.234 plain against 0.082 selective** — the plain model has a half-absorbed copy to unlearn, the
selective one never absorbed it. A slow leak inside the mask was tried as the mitigation and made
both worse.

**What actually bites is the PATH**, and it is why `selective` ships off:

| | plain | selective |
|---|---|---|
| region IoU, 1 object | **0.71** | 0.54 |
| region IoU, 2 / 3 objects | 0.55 / 0.46 | **0.62 / 0.52** |
| peaks paired to objects, 2 / 3 | **96% / 80%** | 71% / **9%** |
| vocabulary purity, 2 objects | **90%** | 47% |
| objects represented, 3 | 2/3 | **3/3** |
| stationary object detected | **0%** | **23%** |

A background that never learns inside the mask never forgets where the object has *been*, so a
traverser leaves a permanent trail and the peaks land on it. Selective is right for a mostly-still
subject whose extent you want and wrong for anything crossing the scene; **neither setting is right
for both**, and every other measurement in the package was taken with it off, so the default cannot
change quietly. `--bg-selective` exposes it. Two mutants caught: an inert flag, and flipping the
default.

**"raw면 텍스쳐랑 컬러도 볼텐데" — the owner's observation, and it caught a real defect I had
just shipped.** `raw_features` averaged the channels away, so the head input was greyscale while
the caller's appearance memory was RGB. That was not a simplification: the demo's own descriptor
measurement had already shown colour doing all the work — raw crop RGB separates two objects at
**0.096** against raw crop grey at **0.984**, which is chance.

Measured on three objects, frozen, with slot 0 always the same object whatever its position:

| input | features | roles | identity |
|---|---|---|---|
| backbone | 145 | 4.61x | 4.01x |
| raw /8 grey (what shipped) | 109 | **12.09x** | **1.71x** |
| raw /8 RGB | 325 | 10.09x | **133.60x** |
| **raw /12 RGB (now the default)** | **145** | 8.34x | **118.87x** |

**Greyscale nearly destroys identity** (1.71x) while being the *best* at roles (12.09x), where
position is the whole question and colour is one more thing to fit. So grey stays reachable as
`--raw-grey` rather than being removed.

**And the default is deliberately 144 numbers — the backbone's own width — so the comparison is
like-for-like rather than a trade.** At identical cost it is **118.87x against 4.01x on identity**,
thirty times better at the harder task. The architectural consequence is larger than the number:
`AppearanceMatcher` exists because the head could not express identity, and with colour in the
input it can — the matcher is still needed to keep *labels* consistent while learning, but the
head's representation has stopped being the bottleneck. Two mutants: averaging the channels away
again, and moving the factor off the backbone's width.

**Feeding the head's own output back into its input — the owner's proposal, and it works, with a
cost that decides how it must be used (2026-08-12).** It also corrects a framing I had been using
loosely: the first section of this file says no choice of *instantaneous* features represents a
history-dependent quantity, and I had been reading that as "this model class cannot do identity".
It is a statement about the **basis**. `y_t = f(x_t, y_{t-1}, ...)` is still linear in the
parameters, and "the caller supplies the basis" is exactly the licence to put lagged outputs in it.

**Measured on the only fixture that can test it**: two *identical* objects orbiting 180° apart,
whose image is **bit-identical** under exchanging their labels — so no instantaneous property
whatever can say which is which. Free-running, against a centre guess and against copying the
previous *true* position:

| basis | error | x centre | x persistence |
|---|---|---|---|
| frame only | 0.2504 | **1.00x** | 0.04x |
| + previous output, teacher-forced | 0.0002 | 1572x | 70.5x |
| + previous output, free-running | 0.0195 | 12.8x | **0.57x** |
| + three lags, free-running | 0.0035 | 71.0x | 3.18x |
| + **EWMA bank .5/.15/.03**, free-running | 0.0018 | **141.7x** | **6.36x** |

Frame-only reads **exactly a centre guess**, which is what says the question is real. And the
short-to-long part of the proposal is not decoration: **one lag free-running is worse than
persistence** (0.57x) because its own errors compound; three lags reach 3.18x and the bank 6.36x.
Teacher forcing reads 1572x and is the ceiling, not a shippable number — an 11x gap, and only the
free-running figure describes a system anyone can deploy.

**It cannot recover from a dropout, and that is the cost.** Sixty blind frames take the error from
0.0009 to 0.70, and nine hundred frames later it is still **0.46**. Not divergence — the prediction
stays bounded at 1.24 — a **stable wrong fixed point**, and it follows from the same property that
made feedback necessary: a frame that cannot disambiguate cannot correct either. A four-rate bank
behaves identically, so it is not a tuning problem. **The guard cannot catch it**: the head is
healthy and its residual is low because it fits the wrong assignment well — the same shape as the
two-rate finding this file already records.

**So the two mechanisms are complements, not alternatives**: feedback carries identity through
ambiguity, appearance (`ObjectMemory`, `AppearanceMatcher`) re-anchors it after a break, and neither
alone is enough. `--feedback`, and the source is in the persistence fingerprint. Three mutants
caught: collapsing the bank to one rate, removing the leak, and not widening the basis.

**Three rigged fixtures on the way, all mine, each caught by a number that could not be true.**
Objects at fixed different heights (height identifies them); different colours (appearance already
solves it, measured at 118x last turn); and a trajectory crossing only in x, which never swapped
anything. The shipped fixture asserts its own swap-invariance for that reason.

**Launching it with `--feedback` crashed, and the bug is one this file already records — arriving
by the same door for the second time.** `ValueError: features: expected shape (162,), got (144,)`
on the first calibrated frame. The calibration branch **recomputes the basis** after setting the
scale, so the feedback bank — concatenated at the call site rather than inside the extraction —
was silently dropped on that one frame. A single extraction path was introduced precisely to stop
this, and adding a term outside it reintroduced it.

**And the first test written for it caught neither of two deliberate bugs, because it reimplemented
the assembly instead of calling it.** That is the third occurrence today of the same shape:
`select_newest` compiled and called by no binary, `drive`'s `teach_every` that no caller ever
passed, and now a test rebuilding the logic it was meant to check. `basis()` and `frame_width()` are
module-level functions now and the test drives *them*.

**Two more mutants then survived because a value was computed and never compared** — the scale
derived from the zero-padded whole, and the scale never applied at all. Both passed a test that
asserted only `scale != 1.0`. What calibration is *for* is that the frame features come out at unit
scale, so that is what is asserted now. Same class as the recorded `gain_norm` gap: present,
decoded, and never looked at.

**"덩어리로 나오는데 움직이는 형상을 잡아내지 못한다" — the trail, measured, and it does not tune
away.** With `--bg-selective` the object *is* fully covered (recall 0.98), and an **equal area is
painted where it was**: precision pinned at 0.50, area exactly **2.05x** the object. That second
blob is what makes the shape illegible.

| background | precision | recall | area/object |
|---|---|---|---|
| running mean | 0.46 | 0.37 | 0.80 |
| mean, selective | 0.48 | **0.98** | **2.05** |
| median of 600 frames | 0.60 | 0.94 | 1.57 |
| *the true wall* | **1.00** | **1.00** | **1.00** |

The last row is the control, and it was run because a precision pinned near 0.5 across every method
at every setting is exactly what this project treats as a broken measurement. It is not broken:
given the true wall the metric reports 1.00/1.00, so 0.5 really is the trail.

**A leak inside the mask is the obvious fix and it fails.** Swept 0.000 → 0.030, precision stays
0.42–0.49 while recall collapses 0.95 → 0.37. The tension is structural: a background must be
**slow** not to absorb the object and **fast** to forget its trail, and for a moving object those
are the same timescale.

**A median escapes absorption in principle, and its requirement is quantitative**: a pixel must be
background for *more than half* the history, so the history must exceed **twice the object's transit
across its own width**. For a 27 px subject drifting at 0.08 px/frame that is ~700 frames — over
twenty seconds at 30 Hz. At 600 frames it still reads 0.60/0.94/1.57.

So none of the shipped options gives a clean silhouette of a slowly-moving subject, and that is
stated rather than tuned around. Two mutants caught (a leak inside the mask, an inverted hold); one
survivor was my own dud that slowed the update *outside* the mask, which is not the property under
test.

**Segment each frame on its own, by colour and optionally texture — the owner's proposal, and it
removes both background-model failures at once (2026-08-12).** The reasoning is theirs and it is
sound: **a per-frame segmentation has no memory, so it cannot have a trail, and cannot lose a
stationary object.** Those were the two failures measured, and both are properties of the mechanism
rather than of the scene. `--locate texture`, and it needs no cadence — every frame produces labels.

Measured IoU against ground truth on single frames, with the shipped class:

| fixture | descriptor | moving | still |
|---|---|---|---|
| colour differs | colour only | **1.00** | **1.00** |
| colour differs | colour+texture | **0.25** | 0.25 |
| same mean colour, texture differs | colour only | **0.32** | 0.32 |
| same mean colour, texture differs | colour+texture | **0.63** | 0.62 |

`still` equals `moving` in every row, which is the whole point — background subtraction detects a
stationary object 0% of the time at every rate. And colour alone reaches an **exact** segmentation
where the background locator paints 2.05x the object's area.

**Texture is off by default because it is actively harmful when colour suffices**: 1.00 → **0.25**,
worse than the control. It is not dilution, it is domination — the stripes-versus-smooth axis carves
the frame somewhere else. It is also the *only* thing that works when object and background share a
mean brightness (0.32 → 0.63), which is the fixture built to make the texture claim testable rather
than a restatement of the colour result. The third row is the control that says the fixture is
honest.

**That 0.25 was 0.71 in the exploratory script**, and the difference is the shipped class: it
**warm-starts its centres** and seeds deterministically where the script re-seeded at random every
frame. Warm-starting is what makes a cluster's *index* stable — measured, the same cluster holds the
object on all twelve frames — and it costs accuracy in exactly the configuration that was already
the wrong one to use. The class's figures are the ones recorded.

**Three iterations are a measured floor, not a preference**: IoU 1.00 at three against **0.16** at
one, so the constructor refuses fewer than two and says why. Cost on this host at 72x96: 1.07 ms at
k=2, 2.47 at k=3, 3.90 at k=4 — affordable against a 33 ms frame, and a host figure rather than a
target one. `k` is `objects + 1` because one cluster is the background, and the background is
dropped **by area** rather than by index so it keeps working when the subject moves.

Four mutants caught: dropping the texture statistics, re-seeding every frame, skipping
standardisation, and letting an empty cluster shift the others along.

**The label count was capped at four, and the cap was mine rather than the problem's.**
`AppearanceMatcher` refused `k > 4` because it solved the assignment by enumerating `k!`
permutations. It solves them exactly now with `scipy.optimize.linear_sum_assignment` in O(k^3), and
without SciPy the old refusal stands rather than degrading silently — a **greedy** nearest-template
match is never used, because it is wrong precisely in the ambiguous cases the matcher exists for.

**The other limits are different and worth separating**: the head has none (2K outputs for any K),
`FrameSegmenter` has none, region colours wrap at eight, and `ObjectMemory`'s vocabulary ceiling is
64. Only the matcher refused.

**And raising the cap does not make more objects work**, which has to be said in the same breath:
vocabulary purity is 90% at two objects, 63% at three, 29% at five; region IoU 0.71 / 0.55 / 0.46.
The refusal was artificial, the degradation is not.

**Two mutants survived the first test and one was a real gap.** A greedy match survived, because
the fixture gave each descriptor a template it exactly equalled — where grabbing the nearest happens
to be optimal. Greedy only fails when two detections want the same template, so a fixture now builds
that: greedy pays 11 where the optimum pays 9, and it is caught. The other survivor is **equivalent**
— raising `PERMUTATION_LIMIT` to 99 makes k=8 enumerate 40,320 permutations and return the same
answer, slower; recorded rather than pinned with a timing assertion this project would refuse to
gate on.

## Recognition memory: the crop is load-bearing, not the learning — 2026-08-19

**The proposal** was to raise perception by holding each label's characteristic features in a
long-term memory and re-assigning the label when the features reappear. It is well aimed:
`demo/test_wean.py` had just measured self-training degrading the head 0.938 → 0.804, and a
descriptor matched against a *stored exemplar* is an anchor rather than a moving target — the
training signal that is not the head's own output, which 2026-08-14 requires.

**First, the recorded next step was tried and it fails.** `tests/test_learned_features.py` records
that a *position* objective leaves single-unit selectivity at random-init level (0.511 against
0.532) and names "an objective that requires identity" as what to try. Posed as three-way shape
classification with each object rendered **alone at the centre** — so position cannot answer it —
and measured **paired**, same initialisation before and after, four seeds, at a rate where both
objectives train:

| objective | paired change in mean d-prime | every seed positive |
|---|---|---|
| position | +0.031 | no (worst −0.025) |
| identity | **−0.061** | no (worst −0.170) |

An objective that requires identity does not produce object-selective units either. The paired
design is an improvement on the recorded one, which compared means across seeds where the random
control's own spread (0.339–0.757) was wider than any effect.

**A learning-rate collapse had to be diagnosed first**, and it is not monotonic: at lr 0.005 every
unit dies (alive 0.00, residual 0.8175 = the constant predictor), at 0.02 most do (alive 0.17), and
at 0.001 the task trains cleanly (alive 1.00, residual 0.272). A first run at 0.02 produced d-prime
0.000 and would have been recorded as "identity destroys selectivity"; it measured a dead network.

**Then the instrument was wrong, and the correction is the finding.** Single-unit selectivity is not
the question — whether the *population* carries shape is. Linear decoding, held out by position:

| descriptor | frozen random backbone | features learned too |
|---|---|---|
| whole frame, hold-out sharing positions | 0.822–1.000 | 0.989–1.000 |
| whole frame, **positions disjoint** | **0.367** | 0.342 |
| **24×24 crop centred on the object, positions disjoint** | **0.996** | — |

Chance is 0.333. **The first row was nearly recorded as "the descriptor already exists, no learning
needed".** Its hold-out shared phases with training — the orbit is periodic mod 61, so it differed
only in noise, and a linear head with 109 coefficients over 600 samples memorises that. Splitting
the positions properly takes it to chance.

**So a whole-frame embedding carries no position-invariant shape, and no objective gives it one.**
Grid pooling entangles *what* with *where* by construction, which is the honest reading of every
d-prime result here: units are not object-selective because the representation is not
object-factored.

**A crop removes position by construction, and then an untrained backbone is already enough** —
0.996 at positions it never saw, no gradient having touched it. That is also why the colour
re-identification result (0.048 within against 0.232 between) worked: it was measured on crops.

**The consequence.** Recognition memory needs no representation learning. It needs *where to look* —
**attention**, one of the eight faculties `tests/test_what_this_is_not.py` records as absent. The
proposal's memory half is sound and largely already built; the half that is missing is the one that
selects a region to describe, and that is what would have to be learned rather than hand-written for
the 2026-08-14 decision to be met.

`tests/test_recognition_memory.py` holds both halves and their contrast, 30 s. Three mutants caught,
including the split silently ceasing to be disjoint — the exact error this entry had to correct.

### Attention by recognition: a stored exemplar can say where to look — 2026-08-19

The entry above ends by naming *where to look* as the missing half. Measured the same day, and the
signal exists: score candidate crop centres by similarity to the nearest stored exemplar, take the
best, on an **untrained** backbone.

| memory | distractor in the same frame | lands on the remembered object |
|---|---|---|
| red disc | blue disc | 1.000 (other 0.000) |
| blue disc | red disc | 1.000 (other 0.000) |
| disc | bar, **same colour** | 1.000 |
| bar | disc, **same colour** | 1.000 |
| square | bar, **same colour** | 1.000 |

**Every figure is 1.000, so the controls are the finding.** A tidy result here is evidence the
measurement is broken, and the first two versions of this experiment were:

- **v1 scored 1.000 against a 0.153 chance rate** on a frame with *one* object on a plain wall,
  where "not wall" already answers it. It could not distinguish recognition from saliency.
- **v2 put two objects in the frame but always the remembered one on the left**, and the exemplars
  came from a left-band training split — so matching *position* would score 1.000 too. This is the
  same confound that turned the earlier "the descriptor already exists" result into position
  memorisation, recurring two hours later in a new fixture.

Alternating the sides leaves it at 1.000, and swapping which colour is remembered **flips** the
answer 1.000/0.000 in both directions. So it is reading the memory, not the wall and not the place.

**What this is not, stated so the result is not over-read.** It is a brute-force scan of ~117
candidate crops per frame — *search*, not attention; no cheap proposal mechanism exists. And the
exemplars were seeded from ground-truth-centred crops, so the bootstrap question is untouched:
where the first exemplar comes from without the classical segmenter is exactly the 2026-08-14
problem and this does not answer it.

**What it does establish** is that the matching signal discriminates on colour *and* on shape at
equal colour — the condition under which single-unit selectivity stayed at random-init level under
both a position and an identity objective. Attention has something to be learned *from*.

`tests/test_attention_by_memory.py`, 9 s. Two mutants survive and are recorded as such; a third,
dropping the distractor, was a real gap and is closed.

### Bootstrap: the first exemplar can be acquired with no labels — 2026-08-19

The entry above left one thing open: the exemplars were seeded from ground-truth-centred crops, so
the loop still rested on the classical segmenter. **Rarity closes it.** Score every candidate crop by
its mean similarity to all the others, take the minimum — the crop least like the rest of the
picture. No labels, no segmenter, no trained weights.

| background | rarest crop lands on an object | chance |
|---|---|---|
| plain wall | 1.000 | 0.101 |
| sinusoidal stripes | 1.000 | 0.101 |
| 40 small clutter blobs | 1.000 | 0.101 |

Two objects in every frame, on the demo's own 72×96 geometry.

**A confound was found and removed.** The first version picked a border crop on **8 of 8** frames:
`crop_at` pads outside the frame with background, so a crop straddling the edge has an unusual
composition and is genuinely the rarest thing there — an artefact of cropping, not a fact about
objects. Candidates are restricted to crops lying entirely inside the frame, which is also the
honest restriction: a crop you cannot take is not a candidate.

**And a claim was withdrawn while the test for it was being written.** An exploratory run scored
**0.000** against a 0.444 chance rate on a 36×48 frame, and the explanation offered was scale — a
24×24 crop covers two thirds of that height, so no candidate can hold one object surrounded by
background. The geometric half is true and is asserted. The 0.000 is **not reproducible**: the test
fixture, differing in object radius and using unpadded crops, reaches 1.000 on the same frame size.
So the failure was real in that run and its cause was never isolated. Asserting the scale limit
would have been asserting a mechanism that cannot be reproduced, which is the error this repository
keeps finding in its own records — this time caught by writing the test rather than after.

**What the four measurements together say.** The owner's proposal — long-term memory of each label's
features, re-assigned when the features reappear — is measurable end to end and needs **no
representation learning at any step**:

| step | mechanism | measured |
|---|---|---|
| bootstrap | rarest crop | 1.000 against 0.101, three backgrounds |
| descriptor | crop embedding, untrained backbone | 0.996 shape decoding at unseen positions |
| attention | nearest stored exemplar | 1.000, flips when the memory is swapped |
| recognition | same match, same label | 1.000 on shape at equal colour |

**What is still not built.** Every step is a full scan — 247 candidates and ~61,000 similarities per
frame — so this is *search*, not attention, and a cheap proposal mechanism is the thing that would
have to be learned. The clutter tested is uniform in appearance and becomes background
statistically; a distractor that looks like a different *object* is untested. And nothing here has
been through `Adapter`, `Guard` or `Pipeline` — it is a measurement, not a component.

`tests/test_bootstrap.py`, 20 s. Three mutants caught, including rarity silently becoming
commonality and border crops being allowed back in.

### Distilling the scan into a proposal: a map works partly, a point not at all — 2026-08-19

The bootstrap entry left the loop as *search* — 247 crop embeddings and ~61,000 similarities per
frame. This asks whether a head can predict where the scan would have looked, from one whole-frame
pass. It is not the trap `demo/test_wean.py` found: rarity is a fixed function of the frame, so the
teacher does not drift as the student learns.

| proposal | lands on an object |
|---|---|
| full scan (the teacher) | 0.933 |
| learned **map**, one frame pass | **0.617** |
| learned **point**, one frame pass | 0.267 |
| chance (a grid cell) | 0.062 |

**The point fails for a measured reason.** With two objects the teacher picks one or the other, so
least squares learns their average: **8 of 10** proposals land between them, and every predicted `y`
sits at 0.42–0.49 — the mean of the two bands. That is this file's own recorded "a difference map
with two movers puts its centroid between them; a point cannot stand in for a field", and the
recorded remedy is what works: a map output has nothing to permute, nothing to count and no
assignment to make. It more than doubles the score.

**The shortfall to the teacher is the head's, not the grid's.** Asked before concluding: an oracle
choosing the best cell of the 6×8 grid scores **1.000** at this tolerance, so quantisation is not
the limit and the map head reaches 62% of what the grid can express.

**A hold-out was corrected mid-experiment, and the reason matters more than the number.** The first
split trained on objects in the left band and tested in the right. That is the correct hold-out for
*recognition* — does the descriptor carry identity independently of place — and the wrong one for
*localisation*, where the position **is** the output: it demands extrapolation beyond any value the
head has ever emitted. Measured, it scores **0.000**. The split was changed because the demand was
incoherent, not because the number was bad, and both numbers are recorded.

**Where this leaves the owner's proposal.** Every step is measured and none needs representation
learning, but the cheap-proposal step is only two-thirds of the way: a single linear map over a
whole-frame embedding recovers most of the structure the point form throws away and still loses a
third of the teacher. Closing that is the open engineering problem; the scan remains the fallback,
and it works.

`tests/test_proposal.py`, 55 s at 80 training frames, where the gap narrows to 0.500 against 0.300 —
the assertions are set to what is reproducible at the size they run, not to the 160-frame figures
quoted above. Three mutants caught.

### Closing the distillation's missing third: it is resolution, and RLS's n² is the price — 2026-08-19

The map proposal reached 0.617 against a teacher's 0.933. Sweeping one axis — the frame embedding's
spatial resolution, since the target is a 6×8 map and the embedding pooled to **3×3**, so the head
was being asked to upsample:

| pool | embedding | proposal (RLS, 160 frames) |
|---|---|---|
| 2 | 64 | 0.517 |
| 3 | 144 | 0.617 |
| 4 | 256 | 0.717 |
| 6 | 576 | 0.817 |
| 9 | 1,296 | **0.850** |

**It is structural, not data.** Tripling the training frames buys almost nothing: 0.617 → 0.667 at
pool 3, 0.817 → 0.833 at pool 6. So the missing third was the head being asked to invent spatial
detail its input had already thrown away, and raising the pool recovers it to **91% of the teacher**.

**And 9 is a hard ceiling for this plan, refused rather than truncated.** The last feature map is
9×12, so `grid_avg_pool` at 12 raises "a 12x12 grid needs a feature map at least that big". The
refusal happens at **call** time, not construction — a `Backbone(pool=12)` builds happily and fails
when it sees a frame. Asserted as it behaves; the first version of the test expected the constructor
to catch it.

**The price is the wall this repository already named.** Under RLS the covariance is n²:

| pool | head | proposal | state, f4 |
|---|---|---|---|
| 6 | RLS | 0.833 | **1.44 MB** |
| 6 | NLMS | 0.733 | **111 KB** |
| 9 | RLS | 0.850 | ~6.7 MB — 480 training frames **did not finish in 560 s** |
| 9 | NLMS | 0.783 | **249 KB** |

So NLMS gives up 0.07–0.10 accuracy for a 6–13× smaller state, which is exactly the trade it was
added for (330,544 B against 2,872 B at 128 features, recorded). And the RLS timeout is a
measurement, not an inconvenience: at pool 9 the n² cost is prohibitive in *time* as well as memory.

**Where the distillation now stands.** 0.617 → **0.850** against a 0.933 teacher, at 1,296 features
under RLS or 0.783 at 249 KB under NLMS. The scan remains the fallback and still works.

**And the last 9% is not the head's — that was checked before varying it, and the check made the
variation unnecessary.** The student imitates the argmax of the teacher's *block-averaged* 6×8 map,
not the teacher's raw argmax over 247 candidates. Those are different, measured over four test
seeds:

| test seed | raw argmax over 247 | argmax of the 6×8 map |
|---|---|---|
| 77 | 0.933 | 0.833 |
| 101 | 0.900 | 0.800 |
| 202 | 0.900 | 0.850 |
| 303 | 0.900 | 0.850 |

Averaging costs about **0.075**, and the student measured 0.850 — **at that ceiling within seed
noise**. So the distillation is closed at the student's level. Going further means changing the
*target representation* — a finer grid, or regressing an offset within the chosen cell — and not the
head. Varying the head's form would have been chasing a gap that does not exist, which is the fifth
time today that asking "can this metric even reach it" before concluding has changed the answer.

`tests/test_proposal.py` pins the monotone relation and the ceiling, 1 m 43 s. The teacher's targets
are computed once and shared across pool settings — the rarity map does not depend on the student's
pooling, and recomputing it per setting doubled the file's runtime for nothing. Five mutants caught.
