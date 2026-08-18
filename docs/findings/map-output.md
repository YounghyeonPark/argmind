<!-- Moved out of CLAUDE.md on 2026-08-16 so the file loaded into every session stays
     small. The text is verbatim; nothing was summarised or dropped. CLAUDE.md carries a
     one-line pointer to each of these. -->

# A segmentation map as the output, and a pyramid of past maps as input — 2026-08-12/15

## A segmentation map as the output, and a pyramid of past maps as input — 2026-08-12

**The owner's redesign, and it dissolves three measured problems rather than working around them.**
Every workaround in the demo existed because the head emits *coordinates*: 2K outputs have no
intrinsic order, so labels had to arrive sorted (0.129) or appearance-matched (0.041) against 0.256
for arbitrary order; `k` is a declaration in every locator because none can count; and an assignment
algorithm exists only to decide which slot a detection belongs in. **A map output has no slots.**
The output is deliberately coarser than the input — 9x12 against 72x96 — because a segmentation does
not need per-pixel precision and the parameter count is `n_features x n_outputs`.

**The pyramid is the multi-timescale memory already measured here, organised spatially**: past
output maps as leaky integrators, each at its own rate *and* its own resolution — recent sharp,
distant coarse. The rates and resolutions move together on purpose, since a band averaging over a
long time has less to say about exactly where.

**And it produces the first result in this session where a head beats the detector it learns from.**
IoU at the output resolution, free-running:

| scene | frame only | + pyramid | the segmenter (control) |
|---|---|---|---|
| one object | 0.99 | **1.00** | 1.00 |
| plus a **same-colour twin** | 0.41 | **0.98** | **0.50** |

A per-frame segmenter has nothing to tell identical twins apart with; the pyramid carries which one
was the target.

**Most of that 0.98 is persistence, and saying so is the point**: repeating the previous *true* map
scores **0.92**, so the learned part adds 0.06 IoU. What is decisively better is the comparison
against the segmenter, not against doing nothing.

**And it fails absorbingly — worse than the coordinate version's stable wrong fixed point.** A
60-frame blackout takes it from 0.98 to **0.00, permanently**: a blank frame drives the map to zero,
the pyramid fills with zeros, and zero is a fixed point. The frame-only head **recovers** (0.13 →
0.62) precisely because it has no memory to corrupt. **Resetting the pyramid does not help** — the
head trained with a populated pyramid has never seen an empty one, so a reset lands in a
distribution that never occurred. Exposure bias again.

**What fixes it is training-time dropout of the pyramid, and the trade has no free lunch:**

| dropout | while healthy | after the blackout |
|---|---|---|
| none | **0.98** | **0.14** (never recovers) |
| 0.05–0.30 | ~0.49 | **~0.68** |
| 0.50 | 0.11 | 0.10 |

A caller chooses between a head that is excellent until something goes wrong and one that is
mediocre and survives. The recovered 0.68 still beats the segmenter's 0.50, so the memory keeps
paying — just less. Nothing here resolves that trade and it is recorded rather than tuned away.
`demo/map_head.py`, four mutants caught.

**Not wired into the live demo.** The window still tracks points; making the map the demo's output
means a 108-output head, a 108-wide `taught` stream and a different overlay, which is a larger
change than measuring the architecture was. The measurement is the deliverable here.

**"움직임이 있을때마다 라벨id가 변경됨" — a real defect, and the first thing I blamed was wrong.**
The suspicion was that `FrameSegmenter.descriptors` standardises **per frame** while the cluster
centres are carried **across** frames, so the space moves under them. Measured over a scene with a
large subject crossing *and* a global brightness drift: **0.0% switch rate**, and identical for a
running or a fixed standardisation. Refuted, and kept as a test because it is the obvious next thing
to blame.

**The actual cause was the display's cluster-to-slot pairing.** `labels` was indexed by *cluster*
and `slot_ids` by *slot*, and the chain between them destroys the correspondence: the background is
dropped **by area**, the survivors are re-sorted **by x**, and the appearance matcher **permutes**
them again. So cluster *n* was coloured with the id of slot *n* — an arbitrary pairing that shifts
whenever anything moves. `slot_label_map` carries the cluster index through, and the dropped
background cluster now stays 0 instead of being painted, which is also the other half of "영역간
구분이 잘 안되는" — a cluster map tiles the whole frame by construction, so painting every cluster
makes the regions read as a partition of the picture rather than as objects.

**And it was in `main()`, where nothing could see it — the fourth time today.** `slot_label_map` is
module-level now and the test drives it; two mutants are caught (keying on the slot, and painting
every cluster). The three before it were `select_newest` compiled and called by no binary, `drive`'s
`teach_every` that no caller ever passed, and a test that reimplemented the assembly it was checking.

**Separating objects inside one appearance cluster — the other half of that report, 2026-08-12.**
`영역간 구분이 잘 안되는건 있고, 에지를 통한 오브젝트를 구분하는 것 또한 필요해보임`. A cluster is a
region of similar appearance, not an object, so two objects that look alike land in one cluster and
its centroid sits *between* them. `split_regions` splits each cluster into its spatially separate
pieces: mean best IoU on two same-coloured discs held apart, **0.50 by cluster against 1.00 by
region**.

**The edge gate earns its place only where `k` is under-declared, which is the normal condition** —
`k` is a declaration in every locator here and nothing counts the appearance groups in a room. With
distractors eating the cluster budget, scored on the two objects the caller declared: **0.53 by
connectivity against 1.00 gated** (2 targets apart + 2 distractors), and 0.60 against 0.83 touching.
The mechanism is the third row: a target that shares a cluster **with the background** is connected
to it, so connectivity alone does nothing and the step at the object's own rim is what severs it.
**My first conclusion — that the gate buys nothing over raising `k`, 0.82 against 0.83 — came from a
fixture set with no distractors, where `k` was adequate by construction.** The controls are what
stop that recurring: the gate must *not* help where `k` is already enough.

**The end-to-end number is less flattering and both are kept.** Distance from each slot's taught
centre to its target: `same colour apart` **0.213 → 0.000**, but `same colour touching` **0.053 →
0.188** — two identical touching discs are one region, so the second slot gets a background
fragment. Mean over seven fixtures 0.092 → 0.081. **And the edge gate does not improve what the head
is taught at all**: region IoU 0.53 → 1.00, taught centre moved by 0.000, because it changes a
region's *shape* and not which region wins by area. A picture improvement, claimed as nothing more —
which is the right claim, since the picture is what was reported wrong. **What nothing separates is
two identical objects that touch**: 0.60 for clusters, connectivity and the gate alike, at every `k`.

**A fixture caught a real defect in my own selection rule.** Dropping the largest *region* as
background is right while regions are clusters and false once they split: on two same-coloured discs
at k=3 the noisy wall occupies two clusters and fragments into **twelve regions of 76–191 px** while
each object is **353 px**, so the objects became the largest regions and the rule dropped one of
them. It drops by **cluster** now, and the limit that leaves is stated rather than tuned around.

**Two implementation claims were measured after being asserted wrongly.** Components hook to the
smallest open neighbour and then **jump pointers** — 4 rounds on a disc, 20 on a region snaking
across the frame — where plain propagation needs one pass per pixel of the *diameter*, 167, and lost
to a Python union-find. And the split runs **one pass over the whole field** rather than one per
cluster, verified region-for-region against a per-cluster reference: **1.5–2.1× faster on five of
seven fixtures and a wash on the other two**, not the third I predicted, because the cost is hook
rounds and sharing passes does not reduce them. Worst case **14.3 ms of a 33 ms frame**;
`scipy.ndimage.label` is 0.05 ms and is not used because it cannot express a gated join.

**The whole selection path was inside `main()` again** — `regions_for_slots` is module-level now.
Eighteen mutants, all caught, and two were real gaps: a region's centre computed from its whole
cluster survived every mask-IoU test, because the centres are what the head is taught with and
nothing compared them; and a rule keeping the *smallest* regions survived every fixture where
`k = objects + 1` makes the two rules equivalent.

**"전보단 나은데 라벨수가 적은것 같음" — the picture was showing the head's slots, 2026-08-12.**
`--objects 3` bounds the head at three because that is its output width; a picture has no such
limit, and this repository's own resolution of the vocabulary question is *unbounded vocabulary,
bounded concurrency*. `label_field` paints every region, and `RegionTracker` gives each one a label
that survives between frames. Three objects, two of them the same colour, k=4:

| identity | coverage | purity | distinct | ids per object |
|---|---|---|---|---|
| raw region index | 1.00 | 0.66 | 1.00 | 2.7 |
| cluster index | 1.00 | **1.00** | **0.00** | 1.0 |
| **tracked** | 1.00 | **1.00** | **1.00** | **1.0** |

**The cluster index scores a perfect purity and is useless**, which is why distinctness is measured
beside it: two objects wearing one id are each perfectly pure, so purity alone cannot fail. The raw
region index is what the display would use for free, and it flickers because a region's number comes
out of a raster-order component pass.

**The labels come from splitting, and `k` turns out to be close to irrelevant.** Five objects give
**5.0 labels at purity 1.00 over 5 minted ids, identically at k = 4, 6, 8 and 10** — five labels for
five objects out of a scene declared with four clusters, because splitting separates the objects that
share one. **An earlier version of this paragraph said raising `k` was worse (3.0 labels at 0.45 over
43 ids) and that is WITHDRAWN**: it was an artefact of the background-selection defect recorded
below, not a property of `k`. A version before that said `k` did not matter at all, from a fixture
whose objects collided — so this measurement has been wrong twice in opposite directions, and what
survives is the splitting half, which never depended on either.

**One area floor cannot serve the head and the picture, measured in both directions.** The split's
floor feeds the head, and raising it 0.01 → 0.03 takes the taught centre on two same-coloured
objects from **0.000 to 0.222** because a slot loses its region entirely. The picture's floor keeps
background fragments out: 0.01 paints 7.0 labels of which **47%** are on a real object, 0.03 paints
3.2 of which **95%** are, and 0.05 paints only objects but starts losing them (coverage 0.98) and
re-minting labels when a region dips under (purity 0.83). So `--seg-min-frac` and `--seg-show-frac`
are two thresholds on one split, and 0.03 is a middle rather than an optimum.

**A minimum age was the mechanism I expected and it is measured not to work.** It helps only when
the floor is too low (precision 0.44 → 0.85 at floor 0.01) and once the floor is right it is worse
than nothing, costing coverage 1.00 → 0.90 for precision 0.98 → 0.90. Not offered.

**And I gated a test on an absolute timing, which this project had already paid to learn once.**
The split's cost test bounded the worst fixture at 33 ms; it passed standalone at 14.3 and **failed
inside the hour at 34.1** under the pre-commit hook, which runs the whole suite alongside it. It
asserts a within-run **ratio** now — a noise-carved background must cost more than twice a clean
scene at the same frame size and the same `k`, which is the hook-round explanation itself — and the
milliseconds stay in the docstring as recorded figures rather than as a gate.

**Two fixtures of mine collided while claiming in a comment that they did not** — minimum separation
0.148 against a touching threshold of 0.220, and 0.112 against 0.180 — so objects merged near the end
of every run and the merge read as a labelling failure. Every number above was re-measured on
separate orbits, and the fixture now **asserts** its own separation. Nine mutants, all caught.

**"how to keep the memory of object if the network appear/hide subsequently" — built, and the
mechanism is appearance while the obvious one is measured wrong (2026-08-13).** A vanished label now
waits `keep` frames in a dormant pool and is reclaimed by what a region *looks like*. Scored on both
directions at once, because either alone can look perfect:

| | position coasting | appearance |
|---|---|---|
| hides, returns to the same place | recovers | recovers |
| hides, returns **somewhere else** | **fails** | recovers |
| both hide and **swap places**, distinguishable | **0/8** | **8/8** |
| a different-looking object arrives in its place | **steals the label** | refuses |
| an identical object arrives in its place | steals | steals |

**Position coasting is refused, not merely unbuilt**: it recovers only the case where nothing moved
and it hands a dead object's label to whatever arrives there — which is the reason `RegionTracker`
held a single frame in the first place, now measured instead of argued.

**The limit is two identical objects that hide together and swap: 1/8, which is worse than the 4/8
a coin flip gives.** Tied descriptors are broken by the assignment's ordering, which follows
position, so it is systematically wrong rather than randomly. Identity through a swap while both are
hidden is history-dependent and there is no history to have — the output-feedback bank cannot help
either, since during the blackout there is nothing to integrate. Distinguishable objects swap fine,
8/8, and that is the whole case for matching on appearance rather than place.

**The descriptor is masked, not a crop**, which is a correction this package already paid for: a crop
near a frame edge is clamped and carries *position*, and that leak inflated an identity control from
50.6% to 74.3% until the fixture was levelled. Measured, the same object at the frame edge and in the
middle describes 0.0000 apart. Colour does the work — raw RGB separates two objects at 0.096 against
greyscale's 0.984, which is chance.

**And the block pairing a slot to its region was still inside `main()`**, where nothing could reach
it — `follow_permutation` is module-level now, with the case that motivated it pinned: a slot the
frame had no region for carries the frame centre, so two entries can be equal and `before.index(pt)`
hands both of them the same region. Sixth thing in two days to need moving out for that reason; four
mutants caught.

**Two of my own measurement errors, both caught by a row that could not be true.** The
false-inheritance fixture put the newcomer at the hidden object's position *during* the gap, so a
region sat there continuously and merely changed colour — position matching linked them with no
vanish at all, and every mechanism including "no memory" read as stealing. And I reported the
identical-twin recovery as chance from a non-monotone 3-gap row; over eight seeds it is 6/8, 7/8, 7/8,
and the reason is mechanical rather than lucky — the twin never vanished, so one-to-one assignment
has already consumed it and only the returning object is free. The case that is genuinely
unrecoverable is both twins hiding together. Seven mutants, all caught.

**"라벨링이 안되는 영역이 있는데 라벨링갯수가 고정되어서 잘 안되는거같음" — the count was never
fixed, and what admits a region is CONTRAST (2026-08-13).** The label count already scales: 4 labels
for a declared 3, 5 for five objects at k=4. A region went unlabelled for two reasons that have
nothing to do with counting — sitting in the **background cluster**, which happens as soon as a scene
has more appearance groups than `k` allows, and being under the **area floor**. Both were reproduced:
a small faint object among four bright ones reads **covered 0.80**, and it is always the same object.

**Four mechanisms measured, three refused.**

| mechanism | result |
|---|---|
| raise or adapt `k` | **non-monotone and scene-dependent in both directions**: five large objects want k=4 (purity 1.00 against 0.45 at k≥6) while the small one needs k=8. The two fixtures disagree, so no `k` serves both |
| merge clusters whose centres are close | folds the small object back into the wall, covered 1.00 → **0.80** at tol 1.5 |
| drop the largest *region* instead of the background *cluster* | rescues the absorbed object (0.80 → 1.00) and **destroys** a noisy scene, precision 0.95 → **0.26** |
| **ring contrast** | covered **0.80 → 1.00** on the failing scene; precision 0.99 → **0.92** on a noise-carved one |

**Contrast is the only local statistic in the list, and that is why it is the one that works.** A real
object differs from the band of pixels just outside it; a fragment of the wall is a piece of the
wall's own distribution. Measured over three scenes, the smallest contrast any real object had was
**0.136** and the largest any non-wall region had was **0.118** — a separating value exists and the
interval is **1.15× wide**, thinner than the 1.35× margin that made `min_energy` ship disabled here.
It ships **on** anyway, at 0.125, because the alternative is an object that is never labelled at all,
and `--seg-contrast 0` restores the old rule exactly (asserted, not described).

The rule: inside the background cluster a region is admitted **only** by contrast and never if it is
the largest piece of that cluster — that piece is the wall. Outside it, the area floor and contrast
are **alternatives**, since a small object with contrast is an object and a large patch without any
is not.

**Two fixtures of mine measured different scenes while being compared.** The contrast probe ran
without the noise the sweep added, so it reported a three-order-of-magnitude gap (0.5 against 0.01)
where the matched-noise probe gives 0.519 against 0.118 — the conclusion survived, the margin did not.
And the first reproduction attempt scored a **uniform 1.00 for every scheme**, which is this project's
own signal that the measurement rather than the world was broken: three distinct objects separate
cleanly whatever you do, and the failure needs *more appearance groups than `k` allows*.

**Nine mutants, and the two that survived were both real gaps.** A small high-contrast object still
vetoed by the area floor survived everything, because in every fixture the small object was *inside*
the background cluster and that branch was never reached. And `MIN_REGION_PX` is **dead as the demo
is configured** — the split's own floor is 27 pixels against its 12 — the same shape as
`at_most(PTrace, 1e4)` being configured and unable to fire, and as `min_dtype` being unreachable for
every shipped adapter. It is kept because a caller lowering `--seg-min-frac` reaches it immediately,
so it is exercised with a floor that does and its unreachability at the default is asserted.

**"레이블 갯수가 상황에 따라 변하는거 맞음?" — yes, and asking found a defect with a very common
trigger (2026-08-13).** Objects entering and leaving a scene give a true count of 1,2,3,4,3,2,1,0 and
the painted count agrees on **30 of 30 frames**, minting exactly **4** ids for 4 objects — so it
follows the scene rather than re-minting its way to the right number. `k` and `--objects` are fixed
for the whole run; the count is not.

**The defect: `label_field` chose its background from the clusters that had produced a surviving
region.** On a clean wall the wall's clusters produce none — their fragments sit under the split's
floor — so the object's own cluster became "the largest present" and was dropped. Measured, a frame
containing exactly one region, the object's own at contrast 0.519, painted **nothing**; a lone object
on a plain background is now labelled on 12 of 12 frames. The background is the largest cluster **in
the frame** now, whether or not it produced a region. Same shape as the `pick_regions` correction of
the day before, which also identified the background from the wrong population.

**And it invalidated a finding this file had recorded twice.** "Raising `k` makes the labelling
worse — 3.0 labels at purity 0.45 over 43 minted ids from k=6" was the defect, not `k`: at high `k`
the wall splits into clusters whose fragments all fall under the floor, so the background was
mis-identified and real objects were dropped and re-minted. Re-measured, five objects give **5.0
labels at purity 1.00 over 5 ids at every k from 4 to 10**, and the crowded scene covers 0.80 with
contrast off and 1.00 with it on, *at every k*. So `k` is close to irrelevant and **contrast is the
mechanism** — which is a cleaner statement of the same conclusion, reached for a correct reason
rather than a lucky one. Three mutants, including one restoring the exact defect, all caught.

**Bottom-up grouping, refused on 2026-08-12, is now measured to WORK — and the refusal was right
at the time (2026-08-13).** The owner restated the strategy: cluster low, group similar labels at a
deeper level. It was refused a day earlier, and re-tested rather than restated, because **the
refusal's stated cause had since been removed**: it was "an untrained backbone's *crop* descriptor
conflates identity with view", and `region_descriptor` now reads the region's own pixels in raw RGB,
where the same object at the frame edge and in the middle describes 0.0000 apart.

| | grouping | the online rule (`RegionTracker`) |
|---|---|---|
| purity where the descriptor separates | **1.00** at any cutoff in 0.10–0.30 | 1.00 |
| number of groups | **found**, not declared: 2/4/6 for 2/3/5 objects | n/a |
| labels per object, continuous run | 1.0 | 1.0 |
| labels per object, **absence longer than the forget window** | **1.0** | **3.0** |
| purity where the descriptor does NOT separate | 0.76–0.80 | 0.95–1.00 |

**It buys nothing where continuity already works and everything where continuity cannot reach.** Three
objects vanishing together for 40 frames with a 30-frame forget window: the tracker mints **3.0**
labels per object, grouping consolidates them to **1.0** at purity 1.00. **Purity cannot be the metric
here** — a fresh label per episode is perfectly pure — which is the same metric hole this file already
records for distinctness.

**And the two results are one finding read at two points, not a contradiction.** Grouping works iff
within-object descriptor distance stays below between-object distance. The 2026-08-12 descriptor was
in the overlapping regime — (0.081, 0.225) at two objects, (0.131, 0.288) at three — and a fixture
built to overlap now (similar colours under a 0.9 illumination gradient, within 0.243 against between
0.014) reproduces the failure exactly: 0.76 against the online rule's 1.00.

**Forcing the group count is what hurt.** Constrained to exactly `n` groups purity falls to 0.80 at
three objects and 0.86 at five, because single linkage must then merge two objects to hit the number.
Letting a cutoff decide is the proposal's whole point, and the cutoff is insensitive over a 3× band.

**One fixture bug of mine, caught by the project's own rule.** The first version scored a **uniform
1.00 everywhere** — a flat disc on a flat wall has *no* within-object variation, so a mean-RGB
descriptor describes it identically every frame (within 0.002 against between 0.85) and no scheme can
fail. The shipped fixture adds an illumination gradient, a size cycle and an occluder, and
`test_the_fixture_gives_the_same_object_something_to_vary` asserts the variation rather than assuming
it.

**Deliberately not wired into the demo.** It is measured to add nothing on a continuous run, and what
it would take is re-pointing `ObjectMemory` at `region_descriptor` — its 96.7/66.5/43.1% figures were
measured on the crop descriptor and are now suspect for the same reason the refusal was.
`demo/test_grouping_again.py` is the evidence, protected both ways.

## The two-harness discrepancy, localised to one axis — 2026-08-17

The open item was that `demo/test_map_output.py` free-runs at **0.09–0.16** while
`demo/test_map_head.py` reaches **0.96–0.98**, with three explanations already refuted (exposure
bias, dropout, and "the head"). It is not unlocalised any more, and one of those refutations is now
suspect.

**Axes checked and eliminated, by reading the two harnesses rather than by reasoning:**

| axis | `test_map_head.py` | `test_map_output.py` | |
|---|---|---|---|
| basis source | raw block-mean pixels | `build(1, "raw", ...)` — also raw | **same** |
| input grid | `IN_H, IN_W = 6, 8` ×3 = 144 | `RAW_FACTOR` 12 → 6×8×3 = 144 | **same** |
| output grid | `OUT_H, OUT_W = 9, 12` | `MAP_H, MAP_W = 9, 12` | **same** |
| **head** | `RecursiveLeastSquares(..., forgetting=1.0, p0=100.0, dtype="f8")` | `NormalisedLeastMeanSquares(mu=0.3, dtype="f4")` from `build()` | **differs** |

**So the difference is the head, and it is the explanation this file already records as refuted.**
That refutation — "a bare RLS reads 0.02, worse" — is the thing to re-derive rather than trust: RLS
converges to the least-squares solution, so 0.02 is not a result about RLS, it is a symptom of how
the substitution was made. The candidates are that the swap kept `dtype="f4"`, kept the default
`p0`, or left the composition's `Bound("external_0", lo=0.35)` unfed, which freezes on the first
update and has caught three separate fixtures in this repository already.

**An earlier hypothesis of mine is refuted here and kept so it is not tried again**: that the
demo's raw downsample is too coarse to express the output — a 6×8 grid read off as a 9×12 field.
Both harnesses have exactly that shape, so it cannot be the cause of a difference between them.

**Not measured yet, and the wean number stays unquotable until it is**: an RLS head, f8, `p0=100`,
`forgetting=1.0`, driven through `build()`'s composition with the external supplied. If that
reaches 0.9+, the discrepancy is NLMS's convergence rate at 108 outputs — which this repository
already measured degrading with width (0.3525 relative error at 128→64 after 400 samples) — and not
a defect in either harness.

### Measured 2026-08-17: the cause is DROPOUT, and both of my localisations were wrong

The section above localised the discrepancy to the head. **That was wrong, and so was the earlier
refutation it called suspect.** Measured, not argued — every axis driven through the failing harness
with one variable moved at a time:

| axis moved | free-running IoU | verdict |
|---|---|---|
| NLMS → RLS (f8, p0=100, λ=1.0) | 0.110 → **0.135** | **not the cause.** The original refutation was right |
| pyramid never teacher-forced | 0.135 → 0.270 | matters for RLS, but the good harness teacher-forces *identically* (`observe(y if train else pred)`), so it is not a difference between them |
| `map_teacher` → `map_target` | 0.135 → **0.135** | the two are **bit-identical** on the same truth (mean 0.071, 15 non-zero cells); not an axis at all |
| Guard → bare head | 0.135 → **0.135** | **zero difference to three decimals.** The guard never intervenes on this fixture |
| **dropout 0.15 → 0.00** | 0.135 → **0.359** | **2.7x. This is the axis.** |

**And it reproduces in the other harness, which is what makes it a finding rather than a fixture
quirk.** `demo/test_map_head.py`'s `drive(memory=True, twin=True)` returns **0.984** today, exactly
the recorded 0.96–0.98 — so that number is live, not stale. Adding `dropout=0.15` to *it* gives
**0.458**, a 2.1x loss in the same direction.

So `MAP_DROPOUT = 0.15` — blanking the memory on 15% of teaching frames, added to stop the head
leaning on the pyramid — is the dominant cause of the gap, in both harnesses independently. **This
file previously recorded dropout as refuted. That refutation is withdrawn.**

**What is still open, stated so the gap is not called closed**: at matched dropout the two harnesses
are 0.359 against 0.984, and 0.135 against 0.458 — a residual factor of about 2.7 that none of the
axes above explains. The discrepancy is no longer 0.09-vs-0.98; it is 0.36-vs-0.98, with the
ordering consistent. The residual is the next thing to localise, and the remaining untested
differences are the free-run length (400 steps against 1,200) and the basis *ordering* that
`map_basis` produces against a plain `concatenate`.

**Three of my own refutations in this area have now been overturned by measurement** — the head one
was right and I doubted it, the dropout one was wrong and I trusted it. The pattern is that each was
a single-variable check written after the conclusion; the table above moves one axis at a time
against a fixed harness, which is what should have been done first.

### What `MAP_DROPOUT` actually buys — ablated 2026-08-17

Dropout was added on the reasoning that a head with a pyramid in its basis will lean on the memory
and stop reading the frame. It costs 2.7x free-running IoU. **Nobody had measured what it prevents**,
so the axes were fixed before running rather than after: teach-time dropout in {0.00, 0.15}, and at
test time the basis ablated three ways — whole, memory zeroed, frame zeroed.

| teach dropout | both | frame only | memory only |
|---|---|---|---|
| 0.00 | 0.359 | 0.115 | **0.349** |
| 0.15 | 0.135 | 0.073 | 0.080 |

**The diagnosis dropout was built for is confirmed.** At 0.00 the head is running almost entirely on
its own memory: memory-only reads 0.349 against a whole-basis 0.359, and knocking the frame out
costs it 0.010. The frame is nearly decorative. That is a free-running loop feeding on its own
output, and it is the same mechanism as this file's recorded blackout result — a 60-frame blank
takes it from 0.98 to 0.00 **permanently**, because zero is a fixed point of a self-sustaining map.

**And the cure does not work.** Dropout does break the dependence — memory-only falls 0.349 → 0.080,
which is what it was for. But the head does not pick the frame up instead: frame-only goes 0.115 →
**0.073**, slightly *worse*. Both halves collapse together. **Dropout removes the crutch without
producing the capability**, and the 2.7x is paid for nothing.

So the honest statement of the map head as it stands is that it does not read the frame in either
configuration, and its apparent 0.36 free-running score at dropout 0 is memory autocorrelation on a
smoothly moving scene rather than perception. A control this file already carries makes the same
point from the other side: a plain persistence baseline reads 0.921 on the good harness, against the
head's 0.984.

**Not turned into a test**, and the reason is stated rather than left as an omission: each row is a
1,400-step run and the pair takes minutes, which is too slow for a suite that runs on every commit.
It is a recorded measurement with its conditions, reproducible from
`docs/findings/map-output.md`'s description; if the wean experiment comes to depend on it, it earns
a test then.

**What this changes about the next step.** The wean experiment — withdrawing the classical labeller
— cannot be read against a head that is not using its input. Whether the head can be made to read
the frame is now the prior question, and the ablation above is the instrument for answering it: the
number to move is **frame-only**, not *both*.
