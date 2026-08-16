<!-- Moved out of CLAUDE.md on 2026-08-16 so the file loaded into every session stays
     small. The text is verbatim; nothing was summarised or dropped. CLAUDE.md carries a
     one-line pointer to each of these. -->

# Objectives declared abstractly, and the detector that does not exist — 2026-08-10/11

## Objectives declared abstractly — measured 2026-08-10/11, and the detector does not exist

**The owner's question**: an application should declare several objective functions abstractly —
track eyes, nose, mouth — when no pre-trained backbone knows which part is which. **The
specification works and the automation does not**, and the split is the finding.

**The abstract form is sound.** *salient / distinct / rigid*, with no part named, has its optimum
at one-tracker-per-part: 4,000 random rigid-triangle placements did not beat the truth, and a
distractor that is bright but drifts independently is refused by `rigid` rather than by anything
that knows what a face is. Measured over static placements.

**But the learner acts over trajectories, and that is where the weight was never validated.**
`W_RIGID` had been raised 1.0 → 3.0 deliberately, to widen the distractor's margin (only +0.039
at 1.0). Over trajectories the cheapest way to keep distances constant is **not to move**, so at
3.0 a policy that always picks `stay` beats the learner. Swept on trajectories, four seeds, 400
greedy steps, scored against a zero-reward control — measured, `demo/objective_sweep/`:

| `W_RIGID` | 0.00 | 0.25 | 0.50 | 1.00 | 2.00 | 3.00 | 6.00 |
|---|---|---|---|---|---|---|---|
| score | **1.18x** | **1.05x** | **1.10x** | 0.98x | 1.01x | **0.69x** | **0.68x** |

So the safe region is 0.0–0.5 and the weight I chose is in the region where **learning is worse
than not learning**. A specification error of mine, found by sweeping the axis rather than by
reasoning about it: *an objective validated over placements is not validated for a learner.*

**The detector I proposed does not work, and that is the substantive negative result.** The plan
was to put the action-reversal rate and the stay rate into `set_external` so the guard freezes a
degenerate policy. Measured across the sweep *and* a second, unrelated environment (a novelty
reward on one panning agent, which the learner maximises by oscillating — **1.51x a
follow-the-blob oracle on the oracle's own reward while looking at nothing**):

| statistic | part tracking | panning | verdict |
|---|---|---|---|
| `reversal` | working reverses **more**, 29.7–37.5% vs 19.2–22.1% | working oracle reverses **0.0%**, wiggler 25.7% | sign inverts |
| `stillness` | worst row is the stillest, 47.2% | working oracle is the stillest, 61.4% | sign inverts |
| `entropy` | overlapping | wiggler **2.24 bits** vs oracle **1.73** | a lower bound freezes the *working* policy first |

**A policy's action distribution does not carry the sign of its own error** — whether moving a lot
is good is a property of the task, which is exactly what the detector was supposed to supply.
Same shape as `at_most(PTrace, 1e4)` being dead while `at_least(PTrace, 1e-9)` trips immediately:
one bound, two directions, and which is right is not knowable from inside the guard. Held out
rather than fitted — `entropy` was chosen on the freeze failure and then tested on the wiggle
failure, where it points the wrong way.

**What does work is much narrower and needs no tuning: collapse to a constant.** A head trained
with zero reward reads **0.00 bits and 100% stillness in both environments**, against ≥1.73 bits
for every policy that does anything. `demo/policy_health.py` is that, `collapsed()` is the test,
and it catches the deployment failure that matters — a head that has silently stopped deciding —
not one pointed at the wrong thing.

**And the value handed to `set_external` is not the raw statistic**, which a failing test found
rather than a reading: a `Bound` is evaluated from the first update and has no "not enough
evidence yet", so raw entropy on a one-action window **freezes the guard at step 0**.
`external_signal()` resolves it on the caller's side of the slot. **Third occurrence** of that
shape here. Both halves are pinned, including that the naive version still fails.

**Two earlier figures are corrected by the shipped definitions rather than by a new run.** The
exploratory code counted `stillness` over *transitions* where `action_stats` counts it over
samples, and pooled three trackers into one histogram where entropy is now per tracker; pooling
inflates entropy whenever the agents differ, so that column had read high (e.g. 2.26 → 2.03).
Every reward and score reproduced exactly. The wiggler's reversal rate, quoted as 45% from an
earlier scratch computation, is **25.7%** under `action_stats`.

**The sweep is checked in because no test can protect this.** Twelve mutants, eleven caught. The
survivor edits a *recorded measurement* — a score in the table — which falsifies an input rather
than a computation, and no assertion reaches it. The first version of the table carried a boolean
"works" label and that mutant was worse: relabelling a row passed the whole suite. The label is
**derived** from the measured score now, so an edit has to change a number, and
`python -m demo.objective_sweep.sweep` regenerates every figure in 4m49s at `--jobs 28` on this
machine (measured; not a target figure). Same remedy as the `persist` sweeps, for the same reason.

**A separate finding, from adding the tests: `testpaths` was `["tests"]`, so `demo/test_pointer.py`
had never run under `python -m pytest` since the demo landed** — while `demo/README.md` said in
so many words that it "runs in the root suite". Present, importable, reached by nothing: the same
shape as `select_newest` being compiled and called by no binary. `testpaths` is `["tests",
"demo"]` now and the demo's tests import no OpenCV, so a clone without it still collects them.
Had this not been fixed, every test above would have protected nothing.

**Running the demo on a webcam found two caller bugs and one real library defect** (2026-08-11),
and the owner found all three by asking why the dot only moved while dragging.

**The demo had the aligner's reference backwards.** `reference="taught"` — the *label* stream —
and frames are built at reference instants, so a prediction existed only when a label arrived.
Measured on the headless fixture: **0 predictions out of 140 unlabelled frames** against
**140/140** with `reference="features"`, error 0.085. The library's own recorded answer was the
one the demo did not use — `hold_last` plus the fastest stream as the reference — and the same
run measured why the other half must stay: with `require_fresh_target` off the held label is
relearned once per camera frame and the error goes **0.085 → 0.238**.

**The suite could not see it**, and the reason is the shape this project keeps finding: `drive`
has had a `teach_every` parameter since it was written and **no caller ever passed one**, so
every case fed a label on every frame — both streams then arrive at the same instant,
`hold_last` always finds an exact match, and the reference choice has no observable
consequence. The knob existed and was never turned.

**And the overlay's `updates` counter was hardwired to zero.** The line read `guard.updates if
hasattr(guard, "updates") else state["updates"]` and `Guard` has no such attribute — so the one
number that answers "is it learning right now" never moved, which is why the question had to be
asked at all. Counted from `report.updated` now, with the overlay stating *learning* against
*predicting only*, because there is no self-supervision here and that is measured rather than
unfinished. Both halves are pinned, including that the attribute is genuinely absent.

**The library defect: `require_fresh_target` leaked, and the leak was not bounded.**
The gate is an identity check on the target sample's timestamp — deliberately, since a tolerance
would put back the threshold the design removed — but it *reconstructed* that timestamp as
`t_reference + staleness`, and that addition is not bit-stable across reference instants. After
a gap in the target stream the same physical sample rebuilt an ulp away from the stored value and
read as new. **52 of 120 frames relearned a 14-second-old target**, which is precisely the
failure the gate exists to prevent and whose cost this file already quantifies. **It needs
timestamps that are not exactly representable** — accumulation or a clock read, i.e. the normal
case; every existing fixture used `k / 30.0`, which round-trips, which is why nothing caught it.

The first version of this note said **exactly one** leaked and would not repeat. Withdrawn: one
was what the *demo's* arithmetic produced, and generalising it made a property of one path into
a property of the mechanism. Fixed by carrying the value instead of deriving it — `Frame.sample_t`
in the reference and `Pick::sample_t` in both C++ tiers, since **all three had the same
reconstruction**. Every pipeline grew 32 bytes (`double sample_t_[MaxStreams]`) and only
`AM_PIPE_F64_6_1_MAX_BYTES` crossed its bound, 1016 → 1048, so the `static_assert` fired for the
fourth time and the bounds were re-measured per shape rather than adjusted by a delta; target
flash moved +20 to +44 B with **static RAM unchanged**, and `SIZES.md` was regenerated.

**Mutation found the coverage gap that mattered more than the fix.** The reference mutant was
caught, but the core's was caught *only by `SIZES.md` noticing the flash had moved* — an accident
— and the hosted port's **survived outright**: `runtime_shape_pipeline.hpp`'s freshness gate had
no behavioural case at all. Both harnesses now carry a gap case with accumulated timestamps,
and all three mutants are caught on host and target. One fixture bug of mine on the way: the
hosted suite puts the target on stream 0 and the reference *is* the target stream, where the
gate cannot fire — copying its mapping while pushing the core suite's arrangement gave 81
updates where 9 were expected, and a 1786-ulp state divergence that was the swap, not the port.

**Teaching from the input instead of from the mouse — built, and the rule it turns on is the
finding.** The owner asked for it after being told the head learns only while dragging. Pure
self-supervision was already measured not to work here (a linear map to the next embedding reads a
diagonal of 0.004 against an identity's 1.0), so the label comes from a second, independent
computation on the same input — an **aiding sensor**, which is the structure this library was
designed against. `demo/label_sources.py`, `--teach auto|mouse|both`, `--detect-every N`.

**An aiding sensor earns a learned model only if it can be blind while the estimand changes.**
Scored only on the frames where the source is blind, over a moving blob:

| source | blind by | head | hold last | centre | |
|---|---|---|---|---|---|
| `MotionLabeller` | nothing moving | 0.090 | **0.078** | 0.316 | **0.87× — the head loses** |
| `CadenceDetector(10)` | a declared interval | **0.101** | 0.133 | 0.231 | 1.31× (1.26× at 5, 1.41× at 15) |

Motion energy fails *structurally*, not for want of tuning: it is blind exactly when the target is
not moving, so holding its last output is right whenever it is unavailable. It ships anyway,
because the negative result is the useful half and it is the first thing anyone reaches for.

**And the unflattering control scopes the whole thing: a detector run on every frame scores 0.010**
— eight times better than the head, which is an interpolator between its teacher's runs and cannot
beat it. On this host the exhaustive detector costs **0.10 ms/frame against the backbone-plus-
pipeline's 0.58 ms** (measured), so for this task on this machine the right answer is to run the
detector. The case argmind is for is a heavy detector on an NPU at 10–30 Hz with a 24-byte head
between its runs, and **that is not measured, so it is not claimed.** My original premise — that
the detector is the expensive one — was wrong on the host, and the measurement said so.

**Two detector choices came out of measurements that rejected the first attempt.** It reads the
**single-step** frame difference rather than accumulating between runs: an accumulated map is
about the *interval*, so its centroid is the middle of the arc travelled and its error grows with
the window — 0.048 / 0.074 / 0.121 / 0.160 at cadence 2/5/10/15 against a flat ~0.036. **Stamping
the accumulated label at the interval's midpoint does not fix it** (identical to three decimals),
because with a fast reference `hold_last` pairs it with the current features either way — the
label is wrong about every instant rather than mis-aligned, which is the one case where this
project's recorded stamping lesson is *not* the answer. And it reads the difference rather than the
frame, which no synthetic-blob fixture can justify since there the mover *is* the brightest thing;
a mutant locating on the raw frame survived everything until a scene with a brighter static lamp
was added, where the two separate completely.

**The first fixture was rigged and the label count gave it away**: the blob *teleported* into its
parked position, a teleport is a motion event, so the detector got one free reading at exactly the
right place and holding was correct by construction. Nine mutants, all caught after that fixture
was added.

**More than one object: the head does it, the detector cannot count — both measured 2026-08-11.**
The owner reported multi-object tracking not working. It was not a recognition failure: with one
output pair the label is one centroid, and a difference map with two movers has its centroid
*between* them, so the demo tracked that midpoint **accurately** — 0.067 against the midpoint,
0.262 against either object. Two independent causes, and separating them was the work.

**The head is permutation-symmetric, and an assignment rule on the caller's side settles it.**
With 2K outputs nothing distinguishes output pair 1 from pair 2, so unless labels arrive in the
same order every frame the least-squares answer for both pairs is the same thing:

| labels | crossing | scored as | head | centre | |
|---|---|---|---|---|---|
| sorted by x | no | role | **0.129** | 0.312 | 2.43x |
| sorted by x | yes | role | **0.137** | 0.222 | 1.62x |
| sorted by x | yes | *identity* | 0.251 | 0.222 | **0.88x — fails** |
| as found | no | role | 0.256 | 0.312 | 1.22x |
| as found | yes | role | 0.236 | 0.222 | **0.94x — fails** |

**What it buys is roles, not identities.** "Leftmost" is a well-defined function of the current
frame and a head can learn it. *Which* object is which through a crossing is worse than guessing
the centre, and that is **the first section of this file arriving in a concrete task**: association
depends on the input's history, and no choice of instantaneous features represents it. **I
predicted crossings would break role tracking too, and that was wrong** — 0.137 against 0.129; the
metric was what mattered, not the scene.

**The detector cannot count, and the reason is frame differencing itself.** One object in motion is
*two* changes — a lobe where it was and one where it is. Asking three peaks of a single blob at
sigma 8.6 px: `suppress=0.18` gives 3/3/2/2/2 and `suppress=0.45` gives 3/1/1/1/1 at displacements
0.5/1.9/4.8/9.6/19.2 px. The separation scales with the object's **size**, not its displacement, so
one radius cannot serve every size. The 0.5 px column is a *different* failure — the difference is
noise, and a floor relative to the peak cannot catch that because the peak is noise too.

**`min_energy` covers that half and ships disabled, because its safe margin is 1.35x**: a still
scene reads 0.0129–0.0131 against 0.0177–0.0190 for slow real motion, so a margin of 2.0 refuses
real motion outright — found by it refusing everything. A peak-to-median gate was measured as the
principled alternative and is **worse**: 0.97x, overlapping, because a second mover raises the
map's own median. Recorded so it is not tried again.

So `--objects K` is scoped: **it separates objects that are far apart and it cannot count.** Nine
mutants; the two that survived were both in `CadenceDetector`, because the multi-object tests
called `find_peaks` directly and did their own sorting — the component that ships was bypassed.
Both closed, plus two more on the energy floor, which had no test at all when it was added.

**Cognitive distinction by appearance works, and it makes identity expressible — 2026-08-11.**
The owner asked whether the two tracked objects could be told apart by their features. They can,
and the reason it is not merely a nicety: sorting by x tracks **roles** ("leftmost" is a function of
the current frame, so a head can learn it) and fails **identity**, because which object this is
depends on which it *was*. **Appearance removes the history** — if they look different, the current
frame says which is which, so the head's target is a function of the frame again and the association
sits on the caller's side. Measured end to end: **0.041 against a centre-guess 0.222, 5.36x**, and
better than the sort rule ever managed on roles (0.063). Each rule fails the other's question
(appearance on roles: 0.242, 0.92x), so both ship.

**Four things had to hold and finding the broken one was the work.** A **random** backbone's crop
embeddings separate the objects (0.206 within/between, identical pair 0.915); nearest-template
matching is **100%** correct at every speed tried; the head expresses "where is the red one" at
0.072 with perfect labels. The failure was the **locator**: a frame difference is a temporal
derivative, so a mover leaves a dipole and two peaks land on one object — **20.3%** correct pairing
against **96.6%** for a running background, which is also immune to the static lamp that ruled out
locating on the raw frame. No suppression radius fixes the derivative: dipole spacing ~17 px for a
sigma-8.6 px blob against a closest approach of ~21.6 px.

**The control is what makes the appearance claim mean anything.** Two *identical* objects must fall
to chance: 74.3% at different heights, **50.6%** at the same height. The residual was the crop
carrying **position**, since a crop near a frame edge is clamped — so the honest control levels
them, and `CROP` is documented as position-sensitive.

**Two limits of the background locator, measured.** It **cannot count** — `k` is a declaration, not
a detection. And a **stationary object is absorbed into the background and lost at every rate**
(0% at every rate tried), while the usable window is narrow: 100% at 1.0 px/frame and rate 0.02,
0% at 4.8 px/frame. `rate` is not free.

**Three measurement bugs in this thread, all caught by reading numbers rather than code.** A
crossing fixture that sent its objects to x=2.0, off the picture, reading 0.89 error on a frame
1.0 wide. A swept table that read a uniform 2.00 in every cell — this project's own rule that a
uniformly tidy result is a broken measurement, applied to my own. And a stationary-object check
that asked whether *a* peak existed rather than whether it was **on** the object, so it could not
produce a negative result and duly reported 100% everywhere; the peak was at (0.89, 0.82) for an
object at (0.35, 0.50). A fourth was a scoring error, not a measurement one: identity scored
against "object A is slot 0" read 0.430 — roughly the distance between the objects — because the
slot names are arbitrary and the matcher seeds from whichever object is leftmost first. A constant
swap read as a tracking failure.

**And running it once more found the worst-shaped bug yet, which is now twice for this demo: a
camera failure that reported success and destroyed the saved model.** `cap.read()` returned
not-ok on the very first frame — the camera was still held by another process — so the loop broke,
the demo saved an **untrained** head over the trained one, printed "saved", and exited **0**,
indistinguishable from a normal quit. The only sign was an OpenCV warning on stderr. Fixed both
ways: `should_save(updates, restored)` refuses to write when the session learned nothing, and a
dead camera now says so and returns 2. Same family as the earlier "printed saved and saved nothing
where it said" — **the failures that matter in this demo are the ones that report success.** Two
mutants, both caught.
