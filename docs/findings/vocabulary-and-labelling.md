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

*(Bookkeeping: this ceiling result shipped inside commit `ae22245`, whose message describes only the
resolution sweep. Two commits were launched in the background concurrently and the first one's
`git add -A` swept up the second's files. Nothing was lost — the test and this text are both in that
commit — but `git log` under-describes it, which matters in a repository where the messages carry
the findings. Do not run two `git add -A` commits concurrently.)*

`tests/test_proposal.py` pins the monotone relation and the ceiling, 1 m 43 s. The teacher's targets
are computed once and shared across pool settings — the rarity map does not depend on the student's
pooling, and recomputing it per setting doubled the file's runtime for nothing. Five mutants caught.

### A sparse descriptor: sixteen of 108 dimensions are free — 2026-08-19

The crop embedding is 108-d with 100 non-zeros and its energy is spread — top-8 holds 37%, top-16
holds 60% — so keeping only the largest `k` discards real signal. Measured across all three
capabilities:

| k | recognition | attention | bytes per exemplar |
|---|---|---|---|
| 4 | 0.967 | 0.875 | 24 |
| 8 | 0.978 | 0.938 | 48 |
| **16** | **1.000** | **1.000** | **96** |
| 32 | 1.000 | 1.000 | 192 |
| 108 (dense) | 1.000 | 1.000 | 432 |

Bytes are `(value f4, index u16)` pairs against a dense f4 vector. **k = 16 costs nothing** — both
tasks sit at the dense score for 4.5× less storage and a sixth of the multiply-adds per match. And
the degradation below that is graceful: four dimensions out of 108 still recognise at 0.967, which
says the code is distributed rather than brittle.

**The bootstrap number looked erratic, and the cause is the metric.** Scored as "is the single rarest
crop within 0.10 of an object centre" it reads 0.167 / 1.000 / 0.000 / 0.750 / 1.000 across those k
— non-monotonic, and reproducible at three times the sample, so not noise. Two explanations were
tested and the first was mine, named in advance:

- **support mismatch**, where two independently sparsified vectors share few active dimensions and
  the dot product collapses. **Refuted**: at k ≥ 16, 0.000 of candidate pairs share no support.
- **the ranking**, which turns out to be sound at every k and to improve monotonically:

| k | mean rarity rank, on-object | off-object |
|---|---|---|
| 4 | 43.2 | 132.0 |
| 8 | 29.4 | 133.5 |
| 16 | 25.1 | 134.0 |
| 32 | 21.2 | 134.5 |
| 108 | 21.2 | 134.5 |

25 of 247 candidates are on an object. **So sparsity does not break the bootstrap signal; taking a
single argmin and demanding its centre fall within 0.10 does.** The crop at the argmin covers only
0.20–0.28 of its area with object — it contains one without being centred on it.

**That is also a finding about `tests/test_bootstrap.py`'s own metric**, which uses the same
argmin-within-tolerance rule. It was not wrong at the dense setting, but it is fragile, and the rank
separation is the sound measure. Recorded rather than silently switched, because the bootstrap
figures already published rest on the argmin form.

`tests/test_sparse.py`, 7 s. Three mutants caught, including `topk` silently keeping the *smallest*
entries.

### Making the suite affordable, and the claim got stronger doing it — 2026-08-20

`tests/test_proposal.py` had grown to **161 s** and pushed the pre-commit hook past ten minutes,
which is a defect in itself: this repository's discipline rests on the suite running at every
commit, and a hook that times out is a hook people switch off. Measured, the slowest single test in
the suite is not one of these (`test_mlp.py`'s fp16 comparison, 70.9 s) and several C++ target tests
sit at 35–41 s — the suite was already heavy and this file pushed it over.

**Coarsening the candidate grid left the teacher untouched.** 247 crops per frame down to 70, and
the teacher scores **0.938 either way**. What did change was the student's map-over-point advantage,
which fell from +0.283 to +0.100 — and that turned out not to be the grid:

| training frames | map | point | gap | seconds |
|---|---|---|---|---|
| 80 | 0.567 | 0.467 | +0.100 | 12 |
| 200 | 0.583 | 0.300 | **+0.283** | 39 |
| 400 | 0.667 | 0.300 | +0.367 | 109 |

**The 80-frame setting was under-trained, and it was flattering the wrong side**: the *point* head
scores 0.467 on noise it loses with more data, settling at 0.300. So the coarser grid paid for the
frames that sharpen the contrast, and the assertion threshold went **up**, 0.15 → 0.2.

| | before | after |
|---|---|---|
| candidates per frame | 247 | 70 |
| training frames | 80 | 200 |
| map − point | +0.100 | **+0.283** |
| file runtime | 161 s | **75.5 s** |

`interior()` takes a `step` parameter so this trade does not disturb `test_bootstrap.py`, whose
published figures are at the default density and rest on it.

**And the bootstrap gained the rank-separation assertion** that `test_sparse.py` showed was the
sound measure. Added rather than substituted: the argmin tests are kept because the figures already
published for them are computed that way, and swapping the measure silently would leave those
numbers describing something no longer computed.

**One mutant is equivalent and recorded as such**: reverting both call sites to the dense grid
survives, because a denser grid only makes the file slower and its assertions stricter. It is not a
gap.

### `ExemplarMemory`: the recognition loop becomes a component — 2026-08-20

Everything measured over 2026-08-19 was a *measurement*: crop descriptors recognise, a stored
exemplar steers attention, rarity bootstraps without labels, sixteen of 108 dimensions suffice.
None of it went through a contract. `argmind/memory.py` does.

| requirement | why it is not a formality | measured |
|---|---|---|
| one flat contiguous state block | persistence is a copy; the C ABI needs no serialisation | 8 slots × 108 features at k=16 → **1,056 B** |
| fixed capacity | the target tier has no heap, and the demo's `ObjectMemory` grows a Python list | a full memory replaces, never grows |
| `state`/`load_state`/`n_features`/`n_outputs`/`dtype` | what `persist.pack` reads | goes through `Store` unchanged, and an 8-slot record is **refused** by a 16-slot memory |

**It is deliberately not an `Adapter`**, and that is recorded as a decision. `update(x, y)` presumes
a target; an exemplar is admitted, not fitted, and a `y` no caller could supply would be the
half-built shape this project refuses. `n_outputs` **is** declared, as the capacity — what it would
emit read as a layer, and what makes the record's shape check mean something.

**Indices live in the same float block as the values**, which is exact: f4 represents integers below
2²⁴ without loss. A second integer array would make the state two blocks and break the one property
the persistence path depends on.

**Two defects, both found by tests rather than by reading.**

The replacement test caught `look_up` refreshing the age of the **nearest** slot rather than the
**matched** one. A near miss then counts as a recognition, every age converges, and the tie-break
evicts whichever came first — which is the exemplar most recently recognised, exactly the one to
keep. Measured: storing a, b, c then touching a and admitting d evicted **a**.

And a mutant storing the *smallest* `k` coefficients instead of the largest **survived every test**,
because they all used random Gaussian descriptors where any consistent subset self-matches exactly.
Real crop embeddings carry structure and keeping the wrong half destroys discrimination while
leaving self-matching intact. The suite now stores a disc and queries a same-coloured bar against
it; the mutant is caught.

`tests/test_memory.py`, 3 s, twelve tests. Six mutants caught.

### Can a SEARCH enter the contract? Audited 2026-08-20 — yes, with one declaration

The recognition loop's remaining steps are a scan: crop N candidates, score them, keep one. The
`Pipeline` contract has no place for "N updates per frame", so the question was whether that puts
the whole loop outside the library. **Answered by building it rather than by reading**, the way the
`Stack` audit was.

**The shape that works is a layer.** A layer's contract is `features(image) -> vector`, and *how* it
produces that vector is its own business — so a layer that scans candidates and returns the winner's
descriptor is expressible. Measured, in order:

1. `Stack` **accepts** it at construction — `n_features` 20,737, the frame plus the constant
2. `predict()` **works end to end**, and the layer reports looking at (0.292, 0.389) with the disc at
   (0.30, 0.40)
3. `update()` **dies at `backward`** — an argmax has no gradient with respect to the frame

**So exactly one call blocked it, and the fix is a declaration rather than a workaround.**
`Stack(..., feature_lr=None)` states at construction that the layers are frozen, and no backward
pass is attempted. Failing on live data for a property knowable at build time is the shape this
project refuses, and the failure above arrives on the *first sample*.

`0.0` is refused rather than silently meaning frozen: `apply_gradients` rejects a non-positive rate,
so a caller writing `feature_lr=0` would get a construction that dies later — the same defect one
level down.

**It also fixes a case the demo already had.** A frozen backbone has nothing to descend, so
`--input backbone` could not be wrapped in a `Stack` without paying for a pass that changes nothing.
It can now.

**And the scanning stack goes through `Guard` and `persist` unchanged** — the test every component
here has to pass. Four mutants caught, including a frozen stack descending anyway and `None` being
coerced to a rate.

**One test failed on the way and it was the right one failing.** Adding `argmind/memory.py` tripped
`test_what_this_is_not.py`'s assertion that eight cognitive faculties are absent, because the keyword
`memory` now matches a module. That test's own message says to update it deliberately if a faculty
has genuinely been built. It has, **partly**: `ExemplarMemory` is a bounded content-addressable store
of appearance and it recognises what it has seen. It is **not** episodic memory — which binds *what*
to *where* and *when* — and it is not working memory, which maintains items against a task. The
faculty list is narrowed to `episodic`/`recollect`/`workingmemory`, and a new test holds the line the
narrowing costs: `ExemplarMemory` must keep `admit`/`look_up`/`observe` and must **not** grow
`timestamp`, `episode`, `context` or `sequence`, nor report a time on a match.

### The five pieces wired together — 2026-08-20

`RecognitionLayer` composes them: crop every candidate, describe each, then recognise if something
in memory matches and otherwise admit the rarest. It is a `Stack` layer, frozen (`feature_lr=None`),
because selection is an argmax. Driven over **appear → absent → a different object → the same object
returns**:

| candidate grid | first appearance | on return | same label |
|---|---|---|---|
| step 6, 117 candidates | slot 0 | slot 0 **and 2** | no |
| **step 2, 925 candidates** | **slot 0** | **slot 0** | **yes** |

**So the claim the proposal was made for holds** — a returning object gets its old label back —
**and the cost is a fine candidate grid.**

**Two failures on the way, and the first was predicted and then built anyway.** When the proposal was
first assessed the note read: *"잘못된 descriptor가 기억에 들어가면 그 오차가 영구히 고정됩니다…
무엇을 기억에 쓸 자격이 있는지의 관문이 기제만큼 중요합니다."* The first version had no gate. On an
empty frame the rarest crop is a patch of **wall**, and every wall crop then matches it forever:
measured, the object recognised as slot 0 came back as **slot 1** after a three-frame absence, with
the layer looking at (0.81, 0.58) and (0.88, 0.83) — background.

The gate is a margin on rarity, and it works because the separation is not marginal: `median − min`
of the candidates' mean similarity is **0.0007–0.0011** on an empty wall against **0.126–0.143** with
an object present, two orders of magnitude. `admit_margin` is **required**, not defaulted, for the
reason `ExemplarMemory.threshold` is. A frame that admits nothing reports slot **−1** rather than
reusing the last one, because "there was nothing here" is a real answer.

**The second failure is crop centring and it is recorded as a cost rather than tuned away.** With
the gate in place a coarse grid still lost the identity, and the reason is measured directly: the
best match score available anywhere on the grid is **0.8835** at steps 12, 6 and 3 and **0.9990** at
step 1, against a 0.85 threshold. A coarse grid leaves the score on the decision boundary where
frame noise decides. Every recognition figure in this repository was measured on crops centred on
the object by construction, so the sensitivity was there all along and only appeared once the centre
came from a grid. Lowering the threshold would have hidden it.

`tests/test_recognise.py`, 52 s, seven tests including two controls that must fail: a coarse grid
must lose the identity, and opening the gate must reproduce the background-exemplar failure. Four
mutants caught.

### Centre refinement: the same identity at a sixth of the crops — 2026-08-20

Identity across an absence needed 925 crops per frame, which is not a rate anything can run. A
coarse pass to pick the region, then a fine pass around the winner to decide there:

| variant | crops per frame | identity across the absence |
|---|---|---|
| plain step 2 | 925 | yes |
| plain step 8 | 70 | **no** |
| **step 8 + refine(8, 2)** | **147** | **yes** |
| step 12 + refine(12, 2) | 188 | yes |

**6.3× fewer crops for the same answer.**

**The first version refined in the wrong place and the numbers said so immediately.** It re-centred
*after* the slot had been assigned, so it changed the reported centre and nothing else — 70 crops,
slots [0, 1] → [0, 3], byte-identical to no refinement. The point of a better centre is to look
*that* descriptor up, so the refinement has to run before the memory is consulted. A test asserts
the refining layer and a plain one disagree about what they attended to.

**And a mutant refining around a fixed frame centre survived**, because on this fixture a
neighbourhood of the middle still overlaps the object — the identity test is insensitive to it. The
locality property is pinned directly instead: the refined centre must lie within the declared span
of the coarse choice. That is the fixture being weak, not the code, and saying which is which
matters.

### The 21× speedup is not available, and the reason is the method's own precision — 2026-08-20

At 147 crops per frame the composed loop runs at **6.5 fps** (153.8 ms), and the time is entirely
the backbone: one forward pass on a 24×24 crop is **1.05 ms**, so 147 of them is 154 ms.
**These two figures are the test fixture's narrower plan, not the demo's, and were published as the
demo's — corrected below to 119.7 ms → 8.4 fps at 0.763 ms a crop.** Left standing rather than
rewritten, because the correction's subject is this sentence. Cropping
(0.67 ms), the 147×147 similarity matrix (0.18 ms) and a memory look-up (0.026 ms) are noise beside
it.

The obvious fix is to run the backbone **once** on the whole frame and pool descriptors out of the
feature map: a 72×96 pass costs **7.5 ms**, a **21×** saving.

**The descriptors agree well enough** — cosine 0.91–0.96 between a crop run alone and the
corresponding region pooled from the whole-frame map, against a 0.85 match threshold. The difference
is padding: a crop run alone pads with zeros where the whole frame has neighbours.

**And it does not work.** Driven through the same appear → absent → return sequence with region
pooling used consistently for both storing and querying, identity fails at every candidate density:

| candidate density | first appearance | on return | same | ms/frame |
|---|---|---|---|---|
| 70 regions | slots 0 **and 1** | slot 2 | no | 22.0 |
| 247 regions | slots 0 and 1 | slot 2 | no | 39.5 |
| 925 regions | slots 0 and 1 | slot 2 | no | 140.8 |

The first appearance already takes **two** slots, so the object is not even recognised frame to
frame — with per-crop descriptors it took one.

**The mechanism is the one every measurement today has been circling.** A region's boundary on the
feature map can only be placed to the nearest **map cell**, and this plan's cells are 4 frame pixels
apart. Centre refinement measured the precision the method needs at **2 pixels**: at step 8 the best
available match score is 0.8835 against a 0.85 threshold and identity is lost, at step 2 it holds.
**The speedup and the required precision come from the same stride**, so buying one spends the
other.

**So the cost is intrinsic to the method as it stands, not an implementation inefficiency.** Saying
that is worth more than a tuned number: a cheaper descriptor, or one less sensitive to centring, is
what would change it — not a faster loop.

### Running it found the wiring broken, for the second time this session — 2026-08-20

`--locate recognise` crashed on launch: `NameError: name 'RecognitionLayer' is not defined`. The
imports were never added, and **three checks passed anyway**:

- `import demo.pointer` — the names are only referenced inside `main()`, so nothing touches them
- the demo suite — my test for the locator **built the layer itself** rather than calling `main()`'s
  construction
- the full suite, for the same reason

**The proximate cause was a guard against double-editing that prevented editing at all.** The script
adding the import was conditioned on `if "ExemplarMemory" not in s`, and that name was already in
the file — in the code that *needed* the import. A safety check on the wrong string.

**The real cause is the one this file has now recorded twice.** `demo/test_pointer.py`'s harness
re-implemented the assembly instead of calling it, exactly as it did for `taught.push` before
`teach_from_click` was hoisted. A harness that rebuilds the thing it is checking cannot see the thing
break. The construction is now `build_recogniser()` at module level — the **ninth** block moved out
of this `main()` for that reason — and the test calls it. Deleting the import fails the test.

**And this is the second time running the demo found a defect no fixture visited**, after the
`--output map` mouse-teaching crash. Both were in `main()`; both were found by launching it.

### The descriptor's centre sensitivity is a frontier, not a bug — 2026-08-20

The 6.5 fps and the missing 21× have one cause: the descriptor tolerates a 2-pixel shift and the
feature map's stride is 4. So — is there a descriptor that tolerates 4 and still tells shapes apart?
Criteria fixed first, from what the method needs: cosine **≥ 0.97** at a 4-pixel shift, and a
same-versus-cross-shape margin **≥ 0.10**.

| descriptor | dims | 2 px | 4 px | 6 px | margin |
|---|---|---|---|---|---|
| global pool (1×1) | 12 | 0.9993 | **0.9988** | 0.9879 | **+0.0040** |
| hard 2×2 | 48 | 0.9834 | 0.9484 | 0.9331 | +0.0071 |
| **hard 3×3 (in use)** | 108 | 0.9723 | 0.9109 | 0.8598 | **+0.0989** |
| soft 2×2 | 48 | 0.9884 | 0.9646 | 0.9430 | +0.0071 |
| soft 3×3 | 108 | 0.9810 | 0.9427 | 0.9081 | +0.0353 |
| soft 4×4 | 192 | 0.9691 | 0.9161 | 0.8732 | +0.0619 |

**Nothing reaches the corner.** Global pooling wins invariance outright and cannot tell a disc from a
bar — margin 0.004. The grid in use discriminates and fails invariance.

**And soft binning — the standard remedy for exactly this quantisation — does not break the
tension.** It moves along the same frontier and slightly worse: soft 4×4 matches hard 3×3's
invariance (0.9161 against 0.9109) with a smaller margin (0.0619 against 0.0989). So the trade is a
property of spatial pooling *as such*, not of the binning's hardness: discrimination here **comes
from** encoding where inside the crop the content sits, which is precisely what a shift disturbs.

**Which closes the loop on the whole line of inquiry.** The 147 crops a frame are buying *explicit
centring*, and centring is the only thing measured that gets both. Escaping the frontier needs a
descriptor outside this family — oriented-gradient histograms, max instead of mean within cells, a
learned invariance — and none of those has been measured here.

`tests/test_descriptor_frontier.py`, 1.8 s. Four mutants caught, including lowering the criteria
until something passes, and the margin quietly comparing an object with itself.

### The demo showed only the camera — the decision was wired and the display was not, 2026-08-20

Owner feedback after running it: *"카메라영상만 나오고 인식하는 내용이 안나와서 앱이 어떤식으로
동작하는지 판단하기 어려움."* Correct, and a real defect. `--locate recognise` decided a slot every
frame and drew **nothing about it** — the slot, whether it was recognised or freshly admitted, and
how full the memory is were all invisible, so nothing on screen distinguished this mode from any
other.

The recogniser's state is now carried out of the loop into `state["recognise"]`, and the overlay
shows it as the **attended crop box** — drawn as the box it actually is, because the crop is a third
of the frame height and contains the object without being centred on it, which is confusing if only
a point is shown — plus a status line reading `slot 0 RECOGNISED   memory 1/16`, `slot 2 new --
admitted`, or `nothing here -- stored nothing`. Green for a recognition, amber for an admission,
grey for neither.

**Third defect in this demo found by launching it rather than by a fixture, and the third that lived
in `main()`.** The other two were the `--output map` mouse-teaching crash and the missing
`RecognitionLayer` import.

**And one of the three mutants for it survived the first pass**, for a reason this file has already
recorded once: the test searched `main()` for the string `"recognise"`, which also appears in
`elif how == "recognise":`, so a mutant that stopped assigning the state passed. Checked through the
**AST** now — an `ast.Assign` whose target is `state["recognise"]` — exactly as the `learn_features`
check had to be after it was fooled by a help string.

### The admission gate fires on every real frame, and no fixture here can show it — 2026-08-20

The owner ran the demo and it read **`memory 13/16`** after ~1,900 frames, against one or two
exemplars on the synthetic scene. Measured on 40 real camera frames: `median - min` of the
candidates' mean similarity is **0.235–0.306**, above the 0.02 gate on **100%** of them — and above
even the synthetic *object present* range of 0.126–0.143.

**The threshold is not wrong.** It does exactly what it was measured to do. The **design assumption**
is wrong: it treats "the frame is featureless" as "there is nothing worth remembering", and a real
room is never featureless. A chair, a bookshelf, a window and a face are all genuinely unlike their
surroundings, so rarity fires every frame — correctly.

**And no fixture in this repository can demonstrate that.** Measured: synthetic clutter reads
**0.0070** and stripes **0.0092**, both *under* the gate, against a real scene's 0.235. The textured
backgrounds here are two orders of magnitude too uniform to stand in for a room — they become
background statistically, which `tests/test_bootstrap.py` already records as an untested limit. So
the real-camera figure is a **recorded measurement with its conditions**, in the same class as this
project's frame rates, and the suite pins only the *inadequacy*: a test asserts those backgrounds
stay under the gate, and says to update it deliberately if they ever stop.

**What admission needs is a criterion about the object, not about the frame** — recurrence rather
than momentary distinctness. That is not built.

### Recognition is parallel now — the serial version was an implementation constraint, 2026-08-20

Owner's objection after watching the demo: *"인지를 한번에 한개씩만 함? 본인은 시각정보 판단할때
한개만 봄? 여러개를 동시에 인지해야 하는거 아닌가?"* Correct on every count.

**The reason it was serial was not a principle.** `features()` must emit a fixed-width vector for the
head above, and one descriptor is fixed width — so an implementation constraint was allowed to decide
the architecture. Foveal *selection* is largely serial; *recognition* is not, and a viewer knows a
chair, a bookshelf and a face are present without visiting each in turn.

**And it cost nothing to fix.** The scan already computed every candidate's descriptor and looked
each one up; the serial version threw all but the best away. A look-up is 0.026 ms.
`emit="presence"` returns **one score per memory slot** instead — exactly what
`ExemplarMemory.n_outputs = capacity` had anticipated. Measured: a disc at (0.30, 0.40) and a bar at
(0.70, 0.62) are both reported, at (0.33, 0.44) and (0.71, 0.61), from the third pass onward.

**The deeper half was still serial after the first fix, and a measurement caught it.** With a disc in
memory and a disc *and* a bar in the frame, the bar was never admitted — the recognition path
returned as soon as the disc matched. **Recognising in parallel and admitting serially is still a
one-object system.** The rule is *explain the scene, then remember what is left over*: candidates
within one crop side of an already-matched centre are accounted for, and the rarity gate runs on the
remainder.

**The demo orders its outputs by slot, not by position**, and that is the point. Every classical
locator here returns centres leftmost-first because that is a function of the current frame and
therefore learnable; this one can do better, because the slot *is* the identity — output pair *i* is
always the same object, across an absence. A frame where only some requested slots are present
teaches **nothing**: an absent object has no location and inventing one is the "a point cannot stand
in for a field" error a level up.

**The earlier refusal of `--objects 2` is withdrawn.** It read "selecting twice needs a rule for
suppressing the first choice that has not been measured". The rule is measured now.

Five mutants on the layer, two on the locator. One survived the first pass and needed the test
strengthened: truncating the presence list to its strongest entry was **masked by parallel
admission**, which refilled the list from the unexplained region while the memory kept growing.
Occupancy holding steady across frames with nothing new is what distinguishes recognising two things
from recognising one and admitting another.

### The retention criterion: spells, not frames — 2026-08-20

The gate on *admission* fires on every real frame, so what was missing is a rule for what **stays**.
The obvious version of "it keeps appearing" is worse than useless: counting **frames** makes the wall
the best-remembered thing in the room, because static scenery appears in every one, perfectly, while
a person who steps out loses ground.

**A spell is one unbroken run of presence.** A bookshelf accumulates a single long spell however many
frames it filled; a person who comes and goes accumulates many. Measured over twelve frames, one slot
present throughout and one present every other pair: **1 spell against 3.**

**And it is a retention criterion, not an admission one** — you cannot notice that something left and
came back until it is already stored. So admission stays liberal and **eviction takes the fewest
spells**, with age as the tie-break. Age alone was the wrong rule: it throws out whichever exemplar
has waited longest, which on a real scene is an object that has stepped out — exactly the one the
memory exists to hold.

The block grows by two scalars per slot, `spells` and `here`: 8 slots at k=16 goes **280 values,
1,120 B** from 264/1,056. `ExemplarMemory.end_frame(present)` closes a frame, called by the caller
because a *frame* is not something the memory can see — `look_up` runs many times within one. Only
`emit="presence"` can call it, since the descriptor path reports at most one slot and cannot say
which others were absent rather than merely not the strongest.

**Three fixture weaknesses had to be fixed before the mutants were caught, and one exposed a real
defect underneath.**

- A mutant reverting eviction to `argmin(age)` survived because `end_frame` did not touch the ages —
  so between two admissions **every age was identical** and "least recently matched" meant nothing.
  The clock now advances per frame. That was a genuine hole in the LRU built earlier, and the earlier
  test for it passed for a weaker reason than it claimed.
- A mutant removing the age refresh from `end_frame` survived twice: first because unequal spells
  never reach the tie-break, then because the admission order already put the frequently-seen slot
  ahead. The tie-break has its own test now, with the **frequent slot admitted first** so only the
  refresh can reverse the order.

**Named in advance, not measured**: a panning camera gives scenery spells too, and a slot can be
evicted during a temporary absence — which is the failure this was built to prevent, arriving by
another door.

### Long-term memory should be unbounded, and the per-slot loop was what stopped it — 2026-08-20

The owner's judgement on capacity: *"동시에 보는게 4개인건 괜찮은데 장기기억이 무한대여야 하는건
맞는듯."* Right, and the two are different quantities that this implementation had collapsed into one
number. `capacity` was both the store size **and** the width of the presence vector — the same
conflation as the serial-recognition one, and caught the same way.

Human figures, as standardly reported and **needing their sources verified before being quoted
here**: simultaneous tracking ~4–5 (Pylyshyn & Storm), visual working memory ~4 (Luck & Vogel; Cowan
revising Miller), subitizing ~4 — against **long-term visual memory in the thousands** (Brady et al.,
~2,500 objects with detail after one exposure). So a bounded store is the wrong shape for the
long-term half.

**What actually stopped it was the look-up.** `look_up` walks the slots, so it is linear in capacity:
**0.018 ms at 8 slots, 2.033 ms at 1,000**, and 147 candidates of the latter is 479 ms a frame —
3.35 fps from look-ups alone. `look_up_many` does it as one matrix product:

| slots | loop ×147 | batched | speed-up |
|---|---|---|---|
| 8 | 3 ms | 1.0 ms | 3× |
| 128 | 34 ms | 1.2 ms | 28× |
| 1,000 | 479 ms | 8.8 ms | 54× |
| 4,000 | 2,479 ms | 9.4 ms | **263×** |

**Nearly flat from 1,000 to 4,000**, because what remains is sparsifying the queries rather than the
slot count. The dense matrix it needs is **derived, not state**: a 4,000-slot cache is 3.5 MB of
float64 and would triple the record, so it is rebuilt on admit, `load_state` and `reset`.

The demo's capacity goes **16 → 256** (35 KB at f4). The 8 KB core tier still caps near 32 slots, and
that is a memory budget rather than a design choice — said so in the code.

**And a figure I published was wrong.** The recorded "6.5 fps" for `--locate recognise` was measured
with the *test fixture's* backbone plan (8 then 12 channels), not the demo's (12 then 16), and before
the batched look-up. Measured on the demo's own construction: **119.7 ms → 8.4 fps**, of which the
backbone is 112 ms at 0.763 ms a crop. So *"the time is entirely the backbone"* holds, but the
number was from a different configuration and was presented as the demo's.

**Four mutants, and one needed a case none of the others could reach**: removing the empty-slot mask
survived, because an empty slot scores exactly 0 and that beats only *negative* scores. A query
anti-correlated with everything stored is the only thing that sees it.

### A configurable report cap, and a recall bug from composing two good features — 2026-08-20

The owner's call on the ~4 limit: make it a setting. `RecognitionLayer(max_reported=N)` and
`--recognise-max N`, defaulting to no limit.

**It is a setting and not a cognitive claim, and the code says so.** Human simultaneous tracking is
around four, but *"people manage four so this should"* is the borrowed-assumption error this project
keeps catching. Measured, the cap buys **nothing**: the batched look-up reports every slot for the
same cost. What a cap is genuinely for is a fixed output width for a head above and a readable
overlay.

**And the invariant that makes it safe**: truncation touches *reporting* only, never `end_frame`. A
slot present but ranked fifth is still present, and letting the cap decide would cost it a spell —
corrupting the retention signal with a display setting. A mutant moving the truncation before
`end_frame` is caught.

**Two fixture errors and one real bug, in that order.**

A first cap measurement was meaningless: the six-object scene was **two identities**, because it
alternated disc and bar and the descriptor is shape-based, so the third disc correctly matched the
first disc's exemplar. Rebuilt with five distinct colours.

Then, with a proper fixture, only **one** slot was reported at every cap. The cause was real:
**refinement had re-serialised the parallel path.** It re-described the neighbourhood of the coarse
choice and ran the presence pass on *that*, so only one object could ever be seen. Neither feature
was wrong alone; composing them without checking the interaction was.

Refining every seed instead of one did not fix it either — the seeds come from the coarse pass, so a
slot the coarse grid mis-centred badly enough to miss was never refined and never recovered:
**4 of 4 reported without refinement, 2 of 3 with it.** The fix is that refinement **adds** candidates
rather than replacing them, which restores 3 of 3 and is sound because presence takes the best score
per slot. A coarse candidate can only lose to its own refinement.

**And one thing is kept on principle rather than on evidence, stated as such.** Refining every seed
rather than the first survives as a mutant: once refinement is additive the coarse candidates cover
the rest, and no fixture here shows a benefit. The comparison is confounded — the two versions admit
differently, report different slot counts, and their scores are not comparable. The cost is one extra
neighbourhood of crops per reported slot.

### "It keeps wandering" — the observation is right, the morphing hypothesis is not the cause, 2026-08-20

Owner, watching `--locate recognise --recognise-max 4` on a real room: boxes do appear in several
places, but *"계속 헤매는 것으로 보여지는데 아마 피사체의 morphing을 동일 객체적 특징을 놓치는게
아닌가"*. The wandering is real and now quantified. The explanation is a different one.

**The wandering, measured** over 120 real frames through the shipped locator:

| | |
|---|---|
| the top slot changed on | **30% of frames** (36 of 119) |
| distinct top slots | **8** |
| mean reported score | **0.989** |
| motion under the reported box | **0.92×** the whole-frame mean |

**The morphing hypothesis is in the right direction and is not the dominant cause.** Descriptor
self-similarity at a location, against frame lag, on a real capture:

| lag (frames) | static locations | moving locations | moving pairs below the 0.85 threshold |
|---|---|---|---|
| 1 | 1.0000 | 0.9976 | 0% |
| 10 | 0.9999 | 0.9727 | 0% |
| 40 | 0.9998 | 0.9510 | 1% |
| 80 (~9 s) | 0.9998 | **0.9219** | **8%** |

So a deforming subject **does** drift — exactly as predicted, and static scenery does not — but only 8%
of nine-second pairs fall below threshold, which cannot account for 27 admissions in 1,867 frames. At
lag 1 it is **0 of 70 locations**, so the first version of this measurement, frame to frame, was
asking at the wrong timescale and found nothing.

**What the numbers point at instead.** The reported box sits where motion is **0.92× the frame mean**
— that is, *not* on the person — and matches at **0.989**. Three of the four boxes in the owner's
screenshot are on blank wall, one reading `#23 1.00`. The memory is filling with background, which
this file already measured: the admission gate's rarity spread is **0.235–0.306 on every real frame**
against a 0.02 threshold, because a real room is never featureless.

**And interchangeable background patches all match near-perfectly, so which one wins is arbitrary.**
That is what 30% switching at score 0.989 is: not lost tracking, but a tie. The wandering is the
absence of anything distinguishing the candidates, not the loss of a subject's features.

**So the fix is upstream of the descriptor.** Morphing would need a tolerant descriptor; this needs
admission to stop storing wall. `spells` was built for exactly that and helps — occupancy settled at
**27/256** where the earlier ungated run reached 13/16 and kept replacing — but it is a *retention*
rule, and what is still missing is an *admission* criterion about the object rather than the frame.

### An admission criterion about the object: descriptor CHANGE — 2026-08-20

The rarity gate asks *"is anything here unlike its surroundings"*, and on a real scene the answer is
always yes. What was missing is a question about the object. **Descriptor change** is one, and unlike
rarity it separates on a real scene.

**It is not a frame difference.** A pixel difference of a moving edge gives a dipole, which this
repository records as a failure of motion-based location. This is one scalar per fixed crop centre:
how far the descriptor there has moved from an exponential baseline.

Measured on real frames, per candidate:

| | quietest 10 | busiest 10 | ratio |
|---|---|---|---|
| lag 5 frames | 0.00024 | 0.05893 | **243×** |
| EMA, rate 0.05 | 0.00005 | 0.01893 | **348×** |

The EMA form is what shipped: one descriptor set of storage rather than a ring of `lag` of them —
40 KB at f4 for 70 candidates, derived and never persisted.

**And the first design was wrong in an instructive way.** It kept rarity as the chooser and added
change as a *gate* on top. Measured on real frames, it admitted **nothing at any threshold** — 0
exemplars at 0.002, 0.005 and 0.01 against 4–6 with the gate off. Rarity picks quiet, distinctive
places, a patch of blank wall among other blank wall, and the change test then rejects exactly what
rarity chose. **A gate cannot improve a wrong choice, only refuse it.** Change has to *select*.

**With change selecting, measured on 120 real frames:**

| selector | occupied | switching | motion under the box |
|---|---|---|---|
| rarity | 8 | **29%** | 1.04× frame mean |
| change ≥ 0.002 | 10 | **16%** | 0.96× |
| change ≥ 0.005 | 8 | 16% | 0.98× |
| change ≥ 0.01 | 6 | 19% | 1.00× |

**The wandering nearly halves** — 29% of frames switching top slot down to 16%. **And attention still
does not prefer moving regions**: the motion under the reported box stays at the frame mean. So this
fixes what is admitted, not where recognition then looks, and saying so matters more than the
improvement.

`--recognise-change D` in the demo. Named in advance and untested: an object that is *still* when
first seen is never admitted, and a panning camera makes everything change so the criterion stops
discriminating.

Five mutants. One needed a fixture nothing else had: a frozen baseline reads correctly while things
keep moving, so only a scene where something **moves once and then stops** shows that the baseline
must forget.

### A match map, because boxes and slot numbers were unreadable — 2026-08-20

Owner, after running it: *"segment 라벨링이 안나오니까 뭐가 어떻게 돌아가는지 하나도 파악이 안됨."*
Fair. Four rectangles and four slot numbers on a photograph of a room say **where the layer looked**
and nothing about how it is carving the scene, so there is no way to judge whether the mechanism is
working or flailing.

**The information was already being computed and thrown away.** The presence pass finds a best
matching slot for every candidate; `last_field` keeps it, as an integer map over the coarse grid —
`slot + 1` where the match cleared the threshold, `0` where it did not. That plugs straight into the
demo's existing `--segments` window, which already colours an integer label field by slot.

**It is a field over the candidate grid, not a pixel segmentation**, and it looks like what it is:
7×10 cells for the demo's 70 candidates. This locator *selects regions*; it does not partition the
frame, and drawing it as though it did would be the more legible lie.

`--locate recognise --segments` now opens that window. Four mutants caught, including unmatched cells
being marked as slot 0 + 1 — which would fill the grid and tell a viewer nothing while looking
busier.

**One small correction on the way**: the locator read `self.layer.last_field` directly, which broke
an existing test's layer double. A layer is not required to publish a field — a descriptor-emitting
one has no per-candidate match to report — so it is a `getattr` with a default.

### The match map is a patchwork, and my measurements do not reach the state that produces it

Owner, on the segment window: *"뭐가뭔지 모르겠음."* The picture is a scatter of coloured cells
unrelated to anything in the room. Two hypotheses were tested and **neither is confirmed**, and the
reason the second could not be is the important part.

**Ties — refuted.** If the best and second-best slots for a cell were within noise, its colour would
be arbitrary. Measured on a real capture: of 70 cells, 20 cleared the 0.85 threshold, and among those
the best scored **0.8960** against a second-best **0.7218** — a gap of **0.174**, with a mean of
**1.0** slot within 0.05 of the winner out of 9. Every cell decides confidently. The patchwork is not
indecision.

**Adjacent disagreement — untested, because the state could not be reached.** The next hypothesis was
that neighbouring cells decisively pick *different* slots, which the frontier measurement makes
plausible: the descriptor's self-similarity is 0.9109 at a 4-pixel shift and 0.8598 at 6, against a
0.85 threshold and an 8-pixel grid step, so one step already looks like a different stored patch. The
run to test it matched **1 cell of 70** — no adjacent pairs to compare, and nothing measured.

**That is the limitation worth recording.** The owner's screenshot shows **88 exemplars after 2,967
frames**. A 60-frame capture reaches **9**. The patchwork is a property of a memory that has
accumulated for thousands of frames, and every measurement in this section samples a state two orders
of magnitude away from it. So the explanations offered for the wandering and the scatter rest on short
captures, and the regime the owner is actually looking at is **unmeasured**.

**What is established** stays established: rarity fires on every real frame (0.235–0.306 against
0.02); change-based selection halves the switching (29% → 16%) and does not move attention onto
moving regions; the descriptor's invariance and discrimination trade off along a frontier no pooling
escapes. **What is not** is why 88 accumulated exemplars paint the scene the way they do — and
answering it needs a long run instrumented from inside, not another mechanism.

**Nine mechanisms have been added along this line** — rarity gate, spells, parallel recognition,
parallel admission, refinement, additive refinement, report cap, change selection, match map — each
measured, each helping a little, and the behaviour on a real room is still not explained. That
pattern is itself the finding: the additions are working around a descriptor that does not support
scene-level identity there, and the frontier measurement says this descriptor family cannot.

### Not downsampling the input: no measurable gain, and 16x the working set — 2026-08-24

The owner asked whether the input and output resolution could simply be left alone instead of being
downsampled to 72x96. The mechanism is real and worth stating before the numbers, because it is why
the question is a good one: the crop is fixed at 24x24 by the backbone's `in_shape`, so at a higher
frame resolution the same crop covers a *smaller* part of the scene — object-sized rather than
scene-sized — and that costs nothing per crop.

**Three readings were measured. The first two were single-seed draws and are withdrawn.**

**(1) Crop fixed, frame raised.** Adjacent-crop against far-crop similarity, real camera frames:

| frame | crop covers | adjacent | far | coherence |
|---|---|---|---|---|
| 72x96 | 33% | 0.8745 | 0.8161 | **+0.0584** |
| 144x192 | 17% | 0.8974 | 0.8640 | +0.0333 |
| 288x384 | 8% | 0.9220 | 0.9176 | +0.0044 |
| 432x576 | 6% | 0.9383 | 0.9352 | **+0.0031** |

Neighbours do become more alike, exactly as the proposal predicts — but distant crops become more
alike faster, so the coherence collapses 19-fold. It even had a mechanism: a 24 px window on a
432x576 frame is mostly featureless surface. **Withdrawn.** Across ten backbone seeds — on the synthetic 1/f scene, not the camera — the ratio
is mean **1.07, sd 1.37**, and three of the ten have *negative* coherence at 72x96. That fixture is
the weaker one and it is named as such, but a sign-unstable sweep is enough to withdraw a monotone
19-fold collapse: there was no effect, there was one draw with a story attached to it.

**(2) Crop scaled with the frame** — the proposal read properly, the same part of the scene at more
detail, the backbone's input growing with it. Tolerance-to-shift minus confusability read
+0.1536 -> +0.0789, a halving, at backbone seed 1 on camera frames. On the synthetic scene at seed
23 the same measurement **triples** (+0.1029 -> +0.3189) — and reading (3)'s own camera sweep below
puts seed 1 at ratio 0.59 against a max of 1.59 over thirty seeds, so the halving was the low end of
a wide spread rather than the effect. Sign-unstable on one draw, and that is what sent the third
reading to a sweep before it was believed. **Withdrawn.**

**(3) The swept reading, which is the only one that says anything.** Tolerance to a physical
displacement of 3% of the frame, against confusability with a different place — this is the recorded
descriptor frontier with resolution as the axis instead of the pooling scheme, and it is where the
proposal could have won, since a 3% shift is 3 px in a 24 px crop and 12 px in a 96 px crop, the same
fraction of the crop either way.

    72x96 -> 288x384, crop 24 -> 96, the demo's plan, 3 real camera frames, THIRTY backbone seeds
    margin ratio: mean 0.87, sd 0.26, min 0.47, max 1.59
    22 of 30 seeds DOWN  ->  two-sided sign test p ~ 0.016

**Established at p<0.05, and small: not downsampling makes the descriptor about 13% less
discriminative.** Tolerance rises, as the proposal reasons it should (0.919 -> 0.984 on the first
frames measured) — and confusability rises faster (0.766 -> 0.905). It is the same frontier as the
soft-binning result, reached along a different axis, which is now twice-measured evidence that the
trade is a property of spatial pooling as such.

**The cost needs no clock and no seed, and it is what decides it.**
`Backbone.peak_activation_values()` goes **2,160 -> 34,560, exactly 16x, for the same 144 numbers
out** — 135 KB at fp32 against the declared 8 KB tier, and 33% more than the whole-frame backbone's
own 25,920, which `docs/ARCHITECTURE.md` already names as the figure a target tier has to hold. A
certain 16x working set for a measured 13% loss. **The answer is no, and the reason is the cost side
rather than the quality side.**

**The methodological result is the one worth carrying.** Two clean, mechanistically explicable
numbers died in a ten-seed sweep, and the second one changed *sign*. The seed spread here is larger
than the effect, so on this question a single-seed measurement is not evidence in either direction —
and both of my first two answers were exactly that. This is the "a uniformly tidy result is evidence
the measurement is broken" rule arriving in a form I did not recognise: the tidiness was not in the
table, it was in the monotonic 19-fold collapse that a second seed does not reproduce.

`tests/test_descriptor_frontier.py` holds it, camera-free on a 1/f synthetic scene, and the fixture's
limit is stated in the file rather than implied: the synthetic scene reproduces the seed **spread**
but not the **direction** (ten seeds split 4/6, mean 0.99), so the direction rests on camera frames
no test here can capture. Three mutants; one survived and was a real gap — a shift that stopped
scaling with the frame measured a four-times-easier displacement at high resolution and *raised* the
margin, which is precisely the flattering answer the block would otherwise have reported. A fourth
test closes it.

**What was not measured.** Whether a *larger crop at the same frame resolution* helps — that is a
different axis (more scene context per descriptor, not more detail) and it costs the same 16x. And
the demo was not run at high resolution end to end; the descriptor is the bottleneck measured here,
not the display.

### The match map on the camera frame, and a crash `--input learned` had been carrying — 2026-08-24

The map in its own window failed at the one thing it was added for. The owner's report after watching
it: *"뭐가뭔지 모르겠음."* Right — a 44x upscaled grid of coloured cells beside the camera view has
nothing tying a cell to a place in the room, so it says no more than the boxes did.

**Drawn on the camera frame instead**, at the display's own resolution, so a cell sits on the thing
it matched. `RecognitionLayer.grid_x` / `grid_y` are the missing piece and are now public: the field
is an integer map with **no geometry of its own**, and its own window needed none, which is exactly
why it could not be placed. Cells are tinted at 0.45 so the scene stays visible underneath — the
point is to see *which thing* a cell landed on, which needs the thing.

**A cell marks a crop CENTRE, not the crop's extent**, and the code says so because the picture
implies otherwise: the crop is 24 px of the 72x96 analysis frame, a third of its height, so the
tinted square is far smaller than what the descriptor saw. Tinting the crop would cover the frame in
overlapping squares and say less. The boxes carry the extent; the tint carries the carving. Colours
come from `ID_COLOURS` indexed by slot, which is what `segment_image` already does for the same
field, so the overlay and the window cannot disagree about what colour a slot is.

**On by default under `--locate recognise`, with `--no-map-overlay` to turn it off.** Default-on
because the alternative was measured against its purpose and failed.

**Eight mutants, one survived and it was a real gap.** Applying the tint to the whole frame instead
of only the painted cells dims the entire scene, and every test missed it: a black fixture frame
stays black however much you scale it, and an all-zero field returns before the blend. A non-black
frame with one populated cell is the only thing that sees it. Also caught: an off-by-one slot colour,
dropping slot 0, crossing the two half-steps (which moves no centre, so the *extent* had to be
asserted as well), transposing the field, ignoring the flag, and handing out the axes unsorted or
swapped.

**Running it found a crash that had nothing to do with the overlay.** `--locate recognise --input
learned` died with `AttributeError: 'Stack' object has no attribute '_e'`: `build()` returns the
**learner** under the name `head`, so with `--input learned` that name holds a `Stack`, and `main()`
was reading `head._e` for the status line. Every other `--input` gives an adapter that has it. This
is the **eleventh** block hoisted out of `main()` for the same reason as the previous ten — no
fixture reaches that line, because it needs a guard-watched update from a real teach signal — and
`head_residual()` is now called by `main()` and by a test that exercises all three sources. A second
test asserts `main()` calls it and no longer reaches for `_e` itself, **via the AST**, the same check
this file already applies to `learn_features`.

**The residual still comes out of a private buffer, and that is stated rather than dressed up.**
There is deliberately no adapter-level accessor: it is a per-update quantity on `UpdateReport`, which
is what `Guard` evaluates, and `Pipeline.step()` does not forward it. Surfacing it properly means a
new field on `StepReport`, which the C++ port mirrors — a library change with a parity cost, and not
a status line's business.

**And `main()`'s argument parser is out too**, as the tenth hoist: it sat *after* `import cv2`, so
checking a flag's default needed a camera library, and a flag can be accepted by the parser and never
reach the thing it controls with no test able to tell. `demo/test_pointer.py` now drives `draw()`
itself against a stub `cv2` — **nothing tested `draw()` at all before this** — and the stub's state
dict is checked against every key `draw`, `status_lines` and `_overlay` actually read, by walking
their ASTs, so a display that grows a field fails there instead of raising `KeyError` inside a test
that was passing yesterday.

**What this does not do.** It does not make the layer attend to more regions, change any descriptor,
or alter what is recognised: the same field that went to the window goes to the frame. It is a
display change, and the wandering the owner reported is measured elsewhere and unaffected.

### 2,005 frames of camera: what the patchwork is, and what it costs there — 2026-08-24

Every capture here had stopped at 60 frames and 9 exemplars while the owner's screen showed 88 after
2,967, which is why the patchwork was recorded as *untested* rather than explained. Run properly,
through `build_recogniser()` so it cannot measure a different assembly than the demo runs:

    2,005 frames, 781 s, 48 exemplars admitted, capacity 256
    last 500 frames:  19.6 of 43.3 slots PRESENT every frame  — 45% of the memory
                      51.4 of 70 candidate cells claimed      — 73% of the frame
                      13.6 distinct slots painted in the map
    cell coverage saturates by frame 250 (52 cells) while `occupied` goes on 20 -> 48

**The memory is holding the room, not the objects in it.** After frame 250 a new exemplar buys no
coverage — only another claimant for cells that were already claimed. Thirteen or fourteen slot
colours tile three quarters of the frame every frame, which is the patchwork, and it is not one
object fragmenting: it is the whole scene being remembered.

**On admission, what did the memory already know?** `look_up` on the descriptor about to be stored,
47 admissions with a memory to compare against:

| best existing similarity | count | share |
|---|---|---|
| [0.00, 0.50) | 2 | 4% |
| [0.50, 0.70) | 9 | 19% |
| [0.70, 0.80) | 9 | 19% |
| **[0.80, 0.85)** — just below threshold | **19** | **40%** |
| [0.85, 1.01) — already matched | 8 | 17% |

**57% of admissions are 0.80-or-more similar to something already stored**, and the dominant band is
*just below* the 0.85 threshold. `_pick_to_admit` selects by rarity **within the frame** and never
asks the memory what it holds, so a crop the memory almost recognises is stored as if it were new.

**My own prediction was wrong and is withdrawn.** I predicted that duplication through the claim
filter would dominate. It is 17%; the near-threshold band is 40%, more than twice as large. The
mechanism I named is real and secondary.

Where it does happen it is fully accounted for: of the eight duplicates admitted at or above
threshold, **seven cleared the claim filter on x, three on y, two on both, and none on neither** —
`claimed = [c for _s, _sc, c in self.last_present]` holds one position per slot, so the same
remembered thing seen far enough from that one position is still `free`. No unexplained residue.

**And the published frame rate describes a memory holding one object.** Crops described per frame,
counted rather than timed, on camera frames:

| reported slots | 2 | 5 | 8 | 16 | 33 |
|---|---|---|---|---|---|
| with refinement | 95 | 338 | 565 | 935 | **995** |

The coarse pass alone is 70. **147 crops — the recorded figure — is one reported slot.** At the ~20
slots this run settles on it is ~995, and the measured rate is **2.6 fps against a published 8.4**
(that 8.4 was measured on a nearly-empty memory; a 60-frame run at 8 exemplars reads 10.9 fps, a
different run and stated as such). Timed at matched reported-slot counts, refinement costs **10-13x**:
726 ms against 72 ms at 17 present, 671 against 51 at 33.

**The memory look-up is not the cause**, and the recorded "nearly flat from 1,000 to 4,000 slots"
stands: without refinement the pass stays 13-80 ms across the same range. The cost is refinement's
one neighbourhood per reported slot, and the saturation at ~995 is those neighbourhoods overlapping
until their union covers the frame.

This bears directly on a note already in the code: *"Refining every seed rather than one is kept on
principle, not on evidence… a mutant refining only the first seed **survives**: no fixture here shows
a benefit."* The principle now has a measured price — 14x the crops, 10-13x the time, in the regime
the demo actually reaches. **Nothing was changed on that basis**; it is the owner's call, and the
benefit is still unmeasured, which is the half that matters.

`tests/test_recognise.py` holds three of these camera-free: the near-threshold admission (the
fixture lands on 0.8132 against 0.85 by itself, which is why it is the test), the above-threshold
duplicate getting in through the claim filter, and refinement's per-slot cost **as a count, not a
clock** — a wall-clock figure is not something this suite gets to assert.

**The claim-filter test needed THREE copies of the object and the reason is the finding restated.**
With two, the layer recognises both from one slot and the reported position lands on whichever scores
higher, so nothing is left far enough away to be `free` — measured, and the first version of the test
failed on exactly that. A third copy is what a moving camera scene supplies continuously: more of the
remembered thing than one reported position can cover. The threshold is lowered to 0.80 there, said
plainly, and the separation is asserted against `_claim_x` so the fixture cannot quietly stop showing
the mechanism.

**A fourth test was written and removed: the DISTRIBUTION cannot be held here.** No synthetic scene
in that file reproduces the 57%-at-0.80 share — twelve frames of `render(...)` at every texture
setting give **two** admissions with best priors 0.49–0.68, because these scenes are near-static and
the admission gate barely fires. That is the same fixture limit already recorded above under *the
gate fires on every real frame and no fixture here can show it*. Forcing it with a tuned threshold
would have been a test measuring its own fixture, so the shares stay camera-measured and unprotected,
and the tests hold the mechanisms instead.

Five mutants, all caught: admission consulting the memory, the claim filter excluding every candidate
near any reported position, refinement skipped, refinement running for the first seed only (which the
weaker first version of the cost test let through — a total-count assertion cannot see it, so a
one-slot scene is measured beside the many-slot one), and a look-up that always reports a match. The
last survives the three tests here and is caught by `test_a_COARSE_grid_loses_the_identity`; a mutant
surviving a `-k` subset is not a gap, and checking that rather than reporting it was the difference.

**What is still not measured.** Whether the 88-exemplar state differs in kind from 48 — this run
stopped at 48 in 781 s and the growth was still linear, so the answer is probably "more of the same",
but that is an expectation and not a measurement. And what refinement *buys*, which is the number
that would justify its price.
