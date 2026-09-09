# Argmind — project context

**A sensor-agnostic library for on-device online adaptation and inference.**
Started 2026-08-08. This is the active project.

**One algorithm: recursive least squares, linear in the parameters.** The first line here used
to read "Multi-sensor AI software: fusion, on-device learning, and inference", which oversold
what is built — `RecursiveLeastSquares` and a `Guard` that wraps it are the only two `Adapter`
implementations in the repository. At n=6 the learned model is **24 bytes of θ** and inference
is **six multiply-adds**; of a composed pipeline's 3,264 B of RAM, model plus covariance is 168
(5.1%) and the other 95% is the guard's checkpoint ring, the sensor rings and the flash slots.
The library is small because the model is small, not because the implementation is clever.

**The caller supplies the basis, and nothing with memory of its own input can be fitted.**
Hysteresis, stiction and backlash depend on the input's *history*; no choice of instantaneous
features represents them. What RLS is genuinely for is tracking a **drift** in something whose
form is already known. DESIGN.md's *What it is not* carries the full statement — it was added
after a hardware-scoping answer proposed an application this model class cannot support, citing
a source that said so in the same paragraph.

Author: Younghyeon Park <ypark.dev@gmail.com>. Apache-2.0 (`LICENSE`, `NOTICE`).
Side project — open source, not a product, no support commitment.

## Working scope

The whole repository. There is one project here.

**A previous, unrelated project — a hardware accelerator IP study — was stopped and removed from the
working tree on 2026-08-08.** It lives in history under the tag `accelerator-final`, which is where
an archive belongs; it is out of the tree because forty files of a different project tax every
search from here on. **Do not restore it to continue it** — everything worth carrying forward was
extracted before removal.

    git show accelerator-final:accelerator/README.md      # browse

Two of its results are still worth knowing, because they were measured and they bear on estimation
work: **the EKF covariance recursion needs fp32 minimum, in every filter form**, and **precision
floors are a property of the algorithm, not of the platform** — three distinct floors appeared
across one workload set, so no single precision policy works everywhere.

Re-derived *here* for RLS on 2026-08-08 rather than inherited, because carrying a conclusion across
algorithms is the borrowed-assumption error: **fp16 fails the RLS covariance recursion even at κ=1**
(refused at construction), and **fp32 survives but does not hold positive-definiteness exactly**.
Both halves have tests. Full detail: `docs/findings/retired-accelerator-project.md`.


## Standards of evidence — carried over deliberately

These came from the previous project, where seventeen modelling errors were found and most of
them flattered the design. They cost real time to learn and they apply to anything measured.

- **Distinguish measured from estimated, every time.** Say which is which in the same sentence
  as the number.
- **A model must be able to fail.** If an analysis cannot produce a negative result, it is not
  evidence. Where a result is negative, protect it with a test so a later change cannot quietly
  optimise it away.
- **A flattering result means find the responsible assumption first.** The most common shapes:
  a model that cannot express a cost, so an axis looks free; a comment claiming a term the code
  does not contain; a swept axis that moves only one side of a comparison.
- **A uniformly tidy result is evidence the measurement is broken**, not that the world is
  simple. This recurred often enough to be a rule.
- **Scrutinise corrections that favour you as hard as ones that do not.**
- Accuracy figures carry the dataset, the split, and the conditions. "95%" without those is not
  a number.
- Cite URLs for external claims. "Not published" is a valid finding; never present an estimate
  as someone else's specification.

## Conventions

**Location: `C:\dev\argmind`.** Moved off Google Drive on 2026-08-08 — Drive sync can corrupt a
git repository by uploading partial writes or creating conflict copies of `.git/index` and loose
objects. Backup belongs on a remote, not on a syncing filesystem.

Windows, PowerShell primary; Python 3.14; git repo. `python -m pytest` before quoting any
number. A tracked pre-commit hook in `.githooks/` runs the suite and blocks on failure —
install once per clone with `git config core.hooksPath .githooks`.

## Status — what is built, and the decisions that shape it

**Built and verified**: `argmind/` (`stream`, `align`, `adapt`, `guard`, `pipeline`, `persist`,
`bench`, `backbone`) and the whole runtime library across to C++ in `cpp/` — a freestanding core
with compile-time shapes behind a C ABI, and a hosted tier (`cpp/include/argmind/host/`) whose
`RuntimeShaped*` classes take their shapes at construction. 1,212 Python tests and 5,653 C++ checks
pass; every suite is mutation-checked. Compilation, linking **and execution** are verified on the
declared floor — the full C++ self-check suite runs under QEMU on Cortex-M4 and Cortex-M55,
**re-measured 2026-08-25 at 5,653 on both cores, exactly the host count**.

**The C++ count is pinned by an assertion AND gated in the documents now (2026-08-25).** It had
drifted 4,652 → 5,653 unnoticed because both gates — host and target — were `>=` floors, and a
floor is blind to a count *rising*.
`test_every_DOCUMENT_stating_the_check_count_states_the_CURRENT_one` reads every `.md` here and
fails on any check count that is neither the current figure nor marked as a dated record, so the
prose is now coupled to the number rather than merely describing it.

**Three corrections to that account, from an independent audit of the commit that made it.**
*"The cause is one mechanism"* is **withdrawn**: only the C++ figure had the floor cause. The
Python figures had no gate at all, and `ExemplarMemory` had a test asserting the right value and
prose that never followed. Three unrelated lapses were collapsed into one fixable cause, which is
the self-flattering shape this project's standards exist to catch. *"Adding checks fails the suite
until the documents are updated"* was **false when written** — both assertions compared the binary
to a literal in a test and nothing read a document; it is true now because the gate above was
added afterwards. And the tally was wrong: the sweep behind it grepped for `4,652` and **missed
three further sites sitting at 4,650**, so *"four for four"* was an undercount, and it was attached
to the `ExemplarMemory` sentence when it belongs to the C++ figure's own history.

**The Python figure above is floored, not pinned.** An equality would fail on every commit that
adds a test, but a floor costs nothing and catches deletion — which the earlier version of this
paragraph argued against by proving that floors miss *rises*, a different direction. Re-derive the
exact number with `python -m pytest`.

**Three adapters**: `RecursiveLeastSquares` (O(n²), covariance), `NormalisedLeastMeanSquares`
(O(n), none) and `MultilayerPerceptron`. `Guard` **is** an `Adapter`, so "no guard" is not
expressible; `Pipeline` and `Guard` are templates over the contract and a second and third algorithm
went through both unchanged.

**The adapter contract, since everything fits into it**: state is one flat contiguous block so
persistence is a copy and the C ABI needs no serialisation; `update()` always returns an
`UpdateReport`; shape errors raise but bad *values* are rejected-and-counted, since a sensor NaN is
a runtime event; covariance windup is reported rather than clamped, because bounding it inside the
adapter would make the guard optional.

**Target floor — settled 2026-08-08, unchanged**: Cortex-M4F class, ARMv7E-M with a
single-precision hardware FPU, 8 KB RAM for the library, 32 KB flash, C++17 freestanding, no
exceptions/RTTI/heap. **Any on-target *timing* claim stays absent** — QEMU models the instruction
set, not the pipeline, and nobody has run this on hardware.

### Active decisions, in the order they were made

| date | decision |
|---|---|
| 2026-08-09 | Scope widened beyond RLS-only: an SBC-class tier alongside the core, more than one model class |
| 2026-08-10 | argmind is **not in the motor loop** — it produces an estimate the robot's stack consumes. All learning and inference run at runtime; a runtime-shape layer sits above the compile-time core |
| 2026-08-13 | Purpose: a 24/7 training and inferencing brain; the **thinking structure** matters most; it must learn without a trained initial state; outputs are whatever the caller configures |
| 2026-08-14 | **argmind must do the learning and inferencing itself** — the demo's classical perception is scaffolding to be replaced, not a component to improve |
| 2026-08-16 | **Learning applies to all parts**, deep network structures are usable, and low-level through high-level cognition must be expressible |

### What is NOT built, so it is not mistaken for deferred work

No audio path, no actuator or speech path, no trained backbone weights imported, and the
hosted-vs-embedded inference decision is open. **argmind is not a cognitive brain as it stands** —
the demo's perception is 1,087 lines of hand-written classical code, what argmind learns is one
linear map, and eight faculties usually taken to be needed are absent rather than unfinished.
`tests/test_what_this_is_not.py` holds that under test rather than in prose.

**The restructure begun 2026-08-16: the seam exists.** The re-audit found one real constraint — an
`Adapter` reported nothing a level below it could descend on — and `Adapter.input_gradient()` is
that quantity: `d(loss)/d(input)` for the last **applied** update, at the **pre-update** parameters,
in a preallocated buffer, zero when the update was refused. All three adapters implement it and it
is **checked against central differences** (~1e-10 for RLS/NLMS, ~1e-9 for the network) across three
shapes and three architectures, because a wrong gradient still trains. The base declares it and
*refuses* rather than leaving it abstract, so an adapter that genuinely cannot differentiate its
input can say so.

**And `Stack` uses it (2026-08-17).** One layer below a head, presented as one `Adapter`, so
**nothing above it changed** — that was the test, not that it learns. One flat state block covers
every level, so a checkpoint checkpoints the layers, a rollback rolls them back (measured against an
unguarded control), a freeze stops them learning, and **one** persistence record carries them. It
goes through `Pipeline` unchanged because its input is flat by design: `FrameMap` builds a 1-D
regressor, and the first `prod(in_shape)` entries are reshaped while the rest are **extras** that
bypass the layers.

**Two real limits, refused rather than half-built.** More than one layer is refused at construction:
`backward` returns no gradient with respect to its own input, so layers cannot be chained — depth
belongs *inside* a layer, and a `Backbone` plan is already a stack. And a `Stack`'s own
`input_gradient()` refuses, because its input is the raw image.

**`Stack.state` is a snapshot, not live storage** — the components own their arrays — which diverges
from every other adapter here and is pinned by a test rather than described. Safe for the paths that
matter, since a guard copies out and calls `load_state`.

**Eight mutants; three survived the first pass and two were real gaps.** Nothing checked that the
extras reach the head, and nothing checked the gradient slice took the *front* of the head's input.
The third was the finding: descending on a refused sample looked equivalent — `input_gradient()` is
zero — and is **not**, because the forward pass has already put a NaN in the image buffer and
`0 * nan` is `nan`. The `if report.applied` check is load-bearing, and only a fixture supplying a
NaN rather than merely a rejected step can see it.

**Running the demo found a crash `--output map` had left behind (2026-08-17).** `taught: expected
shape (108,), got (6,)` — widening the stream to 108 updated the *automatic* label branch and not
the *mouse* one, so one click pushed six values into a 108-wide stream. The fix is a **refusal**,
not a conversion: a map target is a foreground mask over the whole frame, so building one from a
single point asserts that every other cell is background. Same shape as this demo's own finding that
a difference map with two movers puts its centroid between them — a point cannot stand in for a
field. **Eighth block to need hoisting out of `main()`**, and the surviving mutant exposed the
deeper cause: the headless harness **re-implemented** `taught.push` instead of calling it, so both
sides now go through `teach_from_click`. The limit that remains is stated rather than implied — a
change bypassing `main()`'s call site entirely still survives the suite, and this bug was found by
*running* the demo, not by a test.

**The demo is on `Stack` for `--input learned` (2026-08-17)**, so the backbone is inside the guard:
its state block goes **870 → 2,110**, a rollback rolls the convolutional weights back, a freeze
stops them, and **one** record carries both where there used to be a second `Store` with a second
fingerprint. `main()` no longer runs a chain rule — `Stack` owns it, gated on the head's own step
rather than a hand-written `report.updated` check.

**The cost is that the feature stream carries the FRAME**, 20,736 values against 144, because a
`Stack`'s basis is its *layer's* input; capacity drops 64 → 8 to match, the camera being the
reference stream. Only this source: a frozen backbone has nothing to descend and `--input raw` has
no layer, so wrapping either would pay for a backward pass that changes nothing.

**Three tests failed, and the failure was progress** — they pinned properties of the arrangement
this replaced. One asserted `main()`'s hand-written gate; it now asserts the stronger thing, that
the guard's block covers the backbone. One asserted a *second* store for the weights; it now
asserts one record restores both. One asserted the learned basis reads the backbone like the frozen
source — inverted, since the basis is now the frame.

**`learn_features` is kept and is no longer reachable from the loop.** Ten comparison fixtures use
it as the reference implementation for learned-against-frozen, and rewriting those would put
recorded measurements at risk. A test asserts `main()` does not call it — **via the AST, not a
substring**, because `main()` mentions it inside a help string and the text search duly failed on
that. Present-and-called-by-nothing would be the `select_newest` defect; both halves are asserted.
Four mutants, all caught. Moving the demo on to `Stack` is the next
step, and `state_size` is not on every adapter (`NormalisedLeastMeanSquares` has none) although
this file's statement of the contract lists it — a small doc/code mismatch found on the way.

**The recognition memory the owner proposed is built, wired and measured (2026-08-19/20).** Five
pieces, none of which needs representation learning: rarity finds an object with no labels (1.000
against a 0.101 chance rate), a crop descriptor from an **untrained** backbone decodes shape at 0.996
across unseen positions, a stored exemplar steers attention to the remembered object (1.000, and it
flips when the memory is swapped), sixteen of 108 dimensions carry all of it, and a distilled map
proposal reaches its target's own ceiling. `ExemplarMemory` is the bounded store — 8 slots in
**1,120 B**, one flat block, `Store` unchanged — and `RecognitionLayer` composes them as a *frozen*
`Stack` layer, which needed the one contract extension `feature_lr=None`.

**It answers the question it was proposed for**: over appear → absent → a different object → the same
object returns, the returning object gets **its old slot back**. `demo/pointer.py --locate recognise`
runs it, so the region and the identity come from the library rather than from the classical
labeller.

**Two costs are the method's own and are recorded rather than tuned away.** It runs at **8.4 fps** on the demo's own
plan (147 crop embeddings at 0.763 ms; an earlier **6.5 fps was measured on the test fixture's
smaller plan and presented as the demo's** — corrected 2026-08-20), and the obvious 21× — one whole-frame pass, descriptors pooled from
the feature map — **loses the identity**: a pooled region can only be centred to the map's 4-pixel
stride and the descriptor needs 2, so the speedup and the precision come from the same place. And it
attends to **one region per frame**, so more than one object is refused rather than approximated.

**And the sensitivity is a frontier, not a bug (2026-08-20).** Asked whether a descriptor exists that
tolerates the map's 4-pixel stride *and* still separates shapes: **none of six does.** Global pooling
is invariant at 0.9988 and cannot tell a disc from a bar (margin 0.004); the 3×3 grid in use
discriminates at 0.099 and is invariant only to 0.911. **Soft bilinear binning, the standard remedy
for this quantisation, does not escape it** — soft 4×4 matches hard 3×3's invariance with a smaller
margin. Discrimination *comes from* encoding where inside the crop the content sits, which is what a
shift disturbs. So the 147 crops buy explicit centring, and centring is the only thing measured that
gets both.

**Not downsampling the input was measured 2026-08-24 and the answer is no** — but the reason is the
cost: `peak_activation_values()` goes **2,160 → 34,560, exactly 16×, for the same 144 numbers out**,
against a measured **13% loss** in discrimination (30 seeds, ratio 0.87 ± 0.26, 22/30 down, sign-test
p~0.016). **My first two answers were single-seed draws and are withdrawn** — a 19-fold coherence
collapse reads 1.07 ± 1.37 over ten seeds, and a margin that halved at seed 1 *triples* at seed 23.
Carry-forward: **on this question the seed spread exceeds the effect, so one seed is not evidence**,
and the tidiness that fooled me was a monotone collapse rather than a round number.

**The match map is drawn ON the camera frame now (2026-08-24)**, not only in its own 44x-upscaled
window — `RecognitionLayer.grid_x`/`grid_y` are public because a field with no geometry cannot be
placed, and a cell marks a crop *centre*, not the crop's extent, which the code says because the
picture implies otherwise. On by default under `--locate recognise`; `--no-map-overlay` turns it off.
Eight mutants, one survived and was real: tinting the whole frame instead of the matched cells dims
the scene, and a black fixture frame cannot see it. **Running it also found a crash the overlay had
nothing to do with** — `--input learned` died on `head._e` because `build()` returns the *learner*
under that name and a `Stack` has no `_e`. That is the **eleventh** hoist out of `main()`
(`head_residual`), the parser is the tenth, and `draw()` now has tests at all — it had none.

**The long run happened (2026-08-24) and the patchwork is not fragmentation — it is the room.**
2,005 frames, 48 exemplars: **19.6 of 43.3 slots present every frame** (45% of the memory), **51.4 of
70 cells claimed** (73% of the frame), 13.6 slot colours tiling it, and coverage saturates by frame
250 while `occupied` goes on 20 → 48. On admission the memory already scores **≥0.80 for 57%** of
what it stores, dominated by the band *just below* the 0.85 threshold (40%) rather than by outright
duplicates (17%) — **I predicted the opposite ranking and withdraw it**; the claim filter's
one-position-per-slot explains all eight of the 17% and leaves no residue. **And the published rate
is for ONE reported slot**: 147 crops is one slot, ~995 at the ~20 slots this reaches, measured
**2.6 fps against a published 8.4**, because refinement costs a neighbourhood per reported slot
(10-13× timed, 14× counted). The look-up is not the cause and its "nearly flat" stands. Refinement's
*benefit* is still unmeasured and nothing was changed on that basis.

**Refinement's benefit was measured 2026-08-25 and is not established.** On the file's own criterion
with the seed swept (harness agreeing with `drive()` 25/25): **22/25 with refine against 19/25
without** on a clean scene — the recorded categorical claim is true at the fixture's default seed 23
and is **+3 of 25, McNemar p=0.375**. With scene clutter the difference vanishes exactly (16 vs 16,
10 vs 10 at load 8) and at load 24 it inverts (2 vs 4). The reason is geometry: neighbourhoods
overlap, so refinement's union grows into the step-2 grid it exists to avoid — ~7% of it at one
reported slot, **~80% at the ~20 the demo reaches**, ~95% at 33. **The 6.3× saving is a one-slot
property** and that docstring now says so. Those percentages are **computed** for slots placed
uniformly at random, and real slots cluster; an earlier version of this sentence read "100% at 33,
where 70+925=995 is exactly the measured crop count" and that is **withdrawn** — the model says
94.9%, so the measured 995 is a five-point disagreement with it rather than a confirmation of it.
Nothing was changed: not-significant is not zero — though with 5 discordant pairs **no outcome
could have reached 0.05**, so that row reports the design's sensitivity, not refinement's effect.

**The retention collapse recorded here (22/25 → 16 → 10 → 2–4) is WITHDRAWN as a measurement of
retention.** Two independent reconstructions disagree with the loaded rows and one disagrees in
*sign*; the construction that decides it was never checked in. And the criterion is the deeper
problem: `back <= first` is silent about *which* slot, and under load the object is **never
admitted** — a background exemplar already clears the 0.85 threshold, so it is recognised as
background and the criterion passes because it rides that same slot twice. Separating genuine
passes from that: **22/25 at load 0 (22 genuine), 15/25 at noise load 8 (5 genuine), and 23/25 at a
load taken from 900 real camera frames (0 genuine).** The real-camera load makes the recorded
criterion look *better* while the object is stored zero times out of 25 — more degeneracy, not
less, and a first reading of it as "the real load is gentler" is withdrawn too.

**"The mechanism is the threshold" is WITHDRAWN — it is structure, and the threshold sweep is what
refuted it.** Swept 0.60 → 0.98, 25 seeds: genuine retention is **0/25 at every value** under a real
camera load, on both paths, so no exemplar-vs-object score can be the cause. In descriptor mode
`features()` takes the max over **every candidate against every slot** and admits only if that
matched nothing, so one crop anywhere in a 70-candidate frame suppresses admission frame-wide. That
max saturates at **0.998** on a real load (0.881 at noise load 8, measured through `look_up_many`
rather than recomputed — an earlier 0.921 of mine is withdrawn), so the threshold would have to sit
above 0.998, where nothing is recognised at all. `emit="presence"` does **not** have this defect —
its `claimed`/`free` split is the recorded "admission has to be parallel too" fix, **which never
reached the descriptor branch**. And **the demo runs presence while every recorded retention number
here is descriptor**.

Asked fairly of the parallel path — is the object's slot reported *at all* — the demo's path retains
**10/25 at load 8 and 2/25 at load 24**, so a further claim of mine, "0/25 everywhere on presence",
is withdrawn as a criterion artefact. The collapse is driven by **load, not threshold**: across
0.60–0.95 that cell moves only 7 → 10 → 10 → 7, and the long run sits at ~20 slots. The owner's
*"계속 헤맨다"* is real and worse than the withdrawn numbers said. Nothing was changed: the obvious
repair is probably necessary and provably not sufficient, since the path that already has it still
reads 2/25 in the demo's own regime.

**An agent app was asked for and built, 2026-09-08 — `agent/`.** An LLM on the PC, with
`ExemplarMemory` as a fuzzy-keyed cache so repeated work skips the model. **The key was measured
before the cache was built**: of real screen captures more than ten seconds apart, **20.4 % score
≥ 0.995**, and the 0.95→0.995 share barely moves, so the distribution is two-humped and a threshold
exists. That is the **opposite of the camera**, where the same machinery retains 0/25 under load —
windows repeat pixel for pixel and a mug on a desk never does. Through the app on a real screen:
**any hit 95.1 %, recurrent hit 58.3 %**, and the first number flatters (an idle terminal reads
94.6 % and means nothing), so `--measure` prints both and says which matters. **The webcam is context
and never a key** — measured, screen 21.0 % recurrent against screen+webcam 17.6 %, while the webcam
alone hits 92.4 % on **nine** entries, which is the global-pooling failure again: it cannot tell
situations apart. **My reason for that was wrong and is withdrawn** — I predicted the webcam was too
unstable to key on; it is *more* stable than the screen, and the criterion was discrimination, not
stability. Two bugs the tests found, both the same shape: validation lived inside the live branch so
a dry run cached a malformed action, and the first timing test measured the actuator's rate limiter
rather than the cache (a hit read **slower**, 100.4 ms against 53.8 ms). Ten mutants, nine caught,
one equivalent and recorded. **Absent rather than unfinished**: microphone and speaker (no capture
library here), and `ClaudeBrain` is written against the official SDK and **has never been executed** —
no package, no key. Actuation is `live=False` by default behind three brakes.

**argmind is being rebuilt around a conversational core, 2026-09-09 — `argmind/core/` and
`argmind/env/`.** Not a new project: the same repository, the same name, a different shape. The rule
it is arranged around was measured before a line of it was written — **the deterministic side says
WHERE, the probabilistic side says WHAT, and a model never emits a coordinate.** Asking a vision
model for a pixel took 24–82 s and on a dense screen never answered; Windows UI Automation answers
exactly in 41–266 ms; a text model chooses among named things in well under a second. The core is
four separable pieces — `observe`, `poll`, `decide`, `perform` — because **acting is not a stage
that always runs, it is something decided**, and a thought can start from a `Watch` noticing
something rather than from being spoken to. A watch carries a floor, since perception changes in 13
seconds of 14 while a model call costs 0.3–2 s. An effect stores a **name**, never a rectangle, and
is re-resolved immediately before acting: over the second a call takes, a listed widget is gone 13
times in 14. Nothing in `core/` names a platform, and an AST-walking test holds that.

**An unprompted effect without standing permission is downgraded to speech, not refused.** Every
brake built for "you asked, it does one thing" — `live`, a rate limit, a guard — fails to cover "it
decided, repeatedly, while you were reading something else", so the agent says *"the build finished
— want me to open it?"* instead of clicking. Off by default; `--allow-unprompted` is a separate CLI
word from `--live`.

**Choosing accuracy is measured and it is 18 of 28 answerable goals, with 8 of 9 correct declines**
(37 goals, 4 real captured screens of 12–152 widgets, labelled before running, 3 repeats, worst
counted; per-run 19/18/18). **Not good enough to click unsupervised, good enough to propose** —
which is where the permission default already sat for safety reasons. **My explanation of the errors
was refuted by its own control**: they sat 1–10 places from a correct answer, which looked like the
vision model's arithmetic failure, and shuffling the order expanded the median gap from 4 places to
33 — the neighbours were *semantic*, because toolbars group related controls. Shuffling also costs
accuracy (18 → 15 right, 8 → 13 wrong), so the interface's own order carries information and the
list must not be re-sorted. Asking for a name instead of an index is safer and less capable (wrong
8 → 3, missed 3 → 12); nothing was changed on it. Two more harness faults were found on the way,
both invisible from outside: a prompt that scored five correct answers as zero, and a shuffle seeded
with `hash()`, which Python randomises per process, so the first gap analysis scored against an order
the model was never shown. **`docs/findings/choosing-accuracy.md`.**

**Nothing of the old tree is deleted and that is deliberate** — `REBUILD.md` step 5, gated on this
number, is still open, and `argmind-v1` stands at `5e52f61`.

**Open numbers nobody has**: backward-pass throughput on an M55; whether fp32 holds past ten million
updates on a non-stationary target; and why the map head's two harnesses disagree on free-running
IoU — **measured 2026-08-17 to be `MAP_DROPOUT = 0.15`**, which costs 0.359 → 0.135 here and
0.984 → 0.458 in the other harness, reproducing in both. Head, basis, grids, target function and the
guard are each eliminated by moving one axis at a time. A residual factor of ~2.7 is still open, and
**the previously recorded refutation of dropout is withdrawn**.

**Ablated the same day, and the result is negative for dropout.** At dropout 0 the head runs almost
entirely on its own memory — memory-only 0.349 against a whole-basis 0.359, so removing the frame
costs 0.010 — which confirms the failure dropout was built to prevent. But dropout does not fix it:
it breaks the dependence (memory-only 0.349 → 0.080) without the head picking the frame up instead
(frame-only 0.115 → **0.073**, slightly worse). **That conclusion is WITHDRAWN the same day**: it ran on the *twin*
scene, where the frame provably cannot resolve which of two identical blobs is the target — the
recorded per-frame segmenter control there is 0.50. It measured the fixture, not the model. On a
scene where the previous map predicts nothing, the same head reads **0.928 from the frame alone**
against memory-only **0.000** and a collapsed persistence baseline of **0.126**; on a smooth orbit
it leans on memory because persistence alone scores 0.920 there. **The head uses whichever half
carries the information**, which is least squares working. `demo/test_map_perception.py` holds it,
gated on the persistence baseline so the scene cannot quietly stop discriminating. **The wean
experiment then ran, and is negative**: self-training on the head's own hardened output falls 0.938
→ 0.839 against a frozen control, declines monotonically to 0.717 over 4,000 steps and is still
declining, while a teacher ceiling reads 0.943 — so the 0.10 is the cost of the *labels*, not of
updating. It is **not** the zero-fixed-point collapse recorded elsewhere: the mask spreads into the
wrong cells while its area moves *toward* the truth, which a degeneracy detector would not catch.
**The classical labeller cannot be deleted by self-training**, and a signal that is not the head's
own output is what 2026-08-14 requires; nothing here provides one. `demo/test_wean.py`, 3.9 s. A second correction: "too slow for a per-commit suite" was a runtime
asserted without measuring — it is 2.6 s.

## Where the detail lives

Every measurement, correction and withdrawal is recorded verbatim under `docs/findings/`. This file
was 2,954 lines and is loaded into every session; the detail was moved out on 2026-08-16 for that
reason, and nothing was summarised away. **Read the relevant one before changing that area** — most
of these findings are corrections of a plausible first answer, and several are of *mine*.

| file | what is in it |
|---|---|
| `docs/findings/scope-and-adapters.md` | the 2026-08-09 widening; RLS/NLMS/MLP; precision floors, fp16 refused, fp32 to ten million updates; target scaling; `widened()` |
| `docs/findings/platform-and-tiers.md` | the layer question; runtime shapes; packaging; board migration; exporting a model across boards |
| `docs/findings/objectives.md` | abstract objective specification, the weight sweep, and the degeneracy detector that does **not** work |
| `docs/findings/vocabulary-and-labelling.md` | segmentation, splitting, contrast admission, region tracking, object memory, bottom-up grouping |
| `docs/findings/map-output.md` | the map head, the pyramid, the wean experiment, `--output map`, the persistence fingerprint |
| `docs/findings/purpose-and-decisions.md` | the stated purpose, the self-learning decision, the constraint re-audit, and the "is this a brain" audit |
| `DESIGN.md` + `docs/design/` | the design document — what it is, what it is not, the abstractions; the dated log moved to `docs/design/status-log.md` and `basis-and-scope.md` |
| `docs/ARCHITECTURE.md` | **the block diagrams** — the forward stack layer by layer with every tensor shape and weight count, the backward edge added 2026-08-16 and what it replaced, and the `Stack` that is not built yet. Checked against the running code by `tests/test_architecture_doc.py`, not generated |
| `docs/findings/runtime-library-and-cpp.md` | the C++ core, the C ABI, parity, target build and sizes, verification limits |
| `docs/findings/agent-app.md` | the `agent/` app — the cache key measurement, the threshold's asymmetry, key vs context, the two bugs and the equivalent mutant |
| `docs/findings/choosing-accuracy.md` | the rebuilt core's step 4 — 18/28, the refuted index hypothesis, the shuffle control, name vs number, and the two harness faults found measuring it |
| `REBUILD.md` | the rebuild plan itself — what is decided, what is still open (the old library's disposition), and the staged path |


## Sub-agent team

Five project-scoped agents in `.claude/agents/`, checked in for the same reason `.githooks/`
is — the discipline belongs to the project, not to one machine. Each covers work that was
done by hand at least twice, with this project's actual mistakes encoded in it. See
`.claude/agents/README.md` for why these five and what was deliberately left out.

| agent | when to reach for it |
|---|---|
| `evidence-auditor` | before a commit stating a number; after editing DESIGN.md or CLAUDE.md; when a result looks good |
| `numerics-reviewer` | before declaring a precision floor or a tolerance; when an estimator misbehaves |
| `mutation-tester` | after writing a module and its tests; before claiming a suite protects something |
| `parity-verifier` | after any change to `cpp/` or the Python adapters; when a tolerance is under pressure |
| `embedded-reviewer` | after any change under `cpp/`; before adding an ABI entry point |

After a change under `cpp/`, run `embedded-reviewer` and `parity-verifier` together: does it
still fit the target, and does it still compute the right thing. Independent questions.

A sub-agent's conclusion is a claim about the repository, and claims get verified.

**They are also a dependency, which the pipeline port found out.** The embedded review and the
independent parity verification for it were both launched and both terminated on an API spend
limit before reporting; every figure in that section was measured directly instead. The numbers
are real, but the independence is not — the mutation list, the sweep and the memory figures were
designed by the same hand that wrote the code, and this project's history says that is exactly
the arrangement under which a flattering result survives. Re-run both before the next component.

## Working state

The `origin` remote is the private GitHub repository `YounghyeonPark/argmind`; commits use
`ypark.dev@gmail.com`. The repository was moved off Google Drive on 2026-08-08 — Drive sync
can corrupt a git repository by uploading partial writes. Backup belongs on a remote, not on
a syncing filesystem.
