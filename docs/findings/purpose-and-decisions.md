<!-- Moved out of CLAUDE.md on 2026-08-16 so the file loaded into every session stays
     small. The text is verbatim; nothing was summarised or dropped. CLAUDE.md carries a
     one-line pointer to each of these. -->

# The stated purpose, the self-learning decision, and the constraint re-audit — 2026-08-13/16

## The stated purpose, 2026-08-13 — and it is larger than what is built

**The owner's words**: the project is to make a **24/7 training and inferencing brain** that gives a
robot "a heart mind and thinking mechanism"; the **thinking structure** matters most, especially that
**without a trained initial state it can be learned from input sensor data**; **recognition works in
the middle hidden network**; and **the output can be any of the robot's outputs, configured by the
user**. Performance is explicitly deferred.

Recorded rather than argued, as the 2026-08-09 widening was. What follows is only which parts the
repository already supports, which are open, and which are blocked *and by what* — because three of
this file's own recorded claims bear on it and one of them was wrong.

**Already supported.** Arbitrary configured outputs is close to done: `MultilayerPerceptron(n, m,
hidden=...)` takes its widths at construction, `RuntimeShapedMlp<T>` does the same across the C ABI,
and `FrameMap` binds declared sensor indices to features and targets with a fingerprint that refuses
a block restored against a different order. Continuous adaptation with a guard, rollback, freezing and
persistence is the part that is finished and verified on the declared floor.

**Not blocked, and I had it recorded as blocked.** Learning the hidden layers on the second tier is a
**compute** question, not a memory one — 2.59 MB against 4.2, see the withdrawal above. The blocking
fact is that the NPU is an inference part, so a backward pass is software on the M55, and **nobody has
measured that throughput**. That is the first number this vision needs.

**Open, and the 24/7 requirement is what opens it.** fp32 is measured exact against fp64 to **ten
million** updates on a stationary target, inconclusive on a non-stationary one, and its own step
underflow climbs 0.1% → 38.7% across five checkpoints **without saturating**. At 50 Hz a device
reaches ten million in under three days and **one hundred million in twenty-three days**; a year is
1.6 billion. So a 24/7 brain spends essentially all its life two orders of magnitude beyond where the
measurement stops, on the axis that is already trending. Nothing has cost anything yet and nobody has
run it.

**What is measured *against* one part of it, and the scoping matters.** "Recognition in the middle
hidden network" is representation learning, and every negative result in this file about hidden
features is about **frozen random** ones — an untrained stack emits a 6.8e-4 embedding on which every
head fails, a random backbone buys no invariance, per-object correlation segmentation localises 4–50%.
**None of that is evidence against learning the layers**, which is what the owner is asking for; it is
evidence against not learning them. The distinction was not drawn in this file before today.

**What is genuinely absent**, so it is not mistaken for deferred work: any audio path, and any
actuator or speech path. The other two items this paragraph used to list — backprop through a
convolutional stack, and a measurement of whether a learned-from-scratch stack becomes
object-selective — are done as of the same day, and the measurement came back negative; see the
section below.

**The convolutional stack is differentiable now, and the first experiment it enabled returns a
negative result (2026-08-13).** `Backbone` was forward-only, so "recognition in the middle hidden
network" was not expressible. `conv2d_backward`, `depthwise_conv2d_backward`, `relu6_backward` and
`grid_avg_pool_backward` are the four gradients; `Backbone.backward` composes them and
`apply_gradients` is one SGD step, kept separate so `features()` stays a pure function of the state.
Verified against **central differences** per operation (1e-11) and over the composed stack (1e-8 over
five plans, two pool settings and two output scales) — the only check worth having, since a wrong
backward pass still trains, just worse.

**The reference had a trap that had to be diagnosed rather than assumed.** The composed check
disagreed by 1.5e-02 to 7.6e-01 on three plans, and the cause is `relu6`'s corner, not the gradient:
**biases initialise to exactly zero** and `relu6` emits exact zeros, so an all-zero receptive field
puts a pre-activation precisely on the kink, where a central difference averages slope 0 and slope 1.
Confirmed by counting — the plans that disagree have 8 and 3 pre-activations at exactly zero, the plan
agreeing to 3.9e-10 has none — and by nudging the biases, which takes the same configurations to
6e-11. A test asserts the un-nudged version *still* disagrees, so it is not rediscovered as a bug.

**The experiment: learning the hidden layers from sensor data does NOT make units object-selective,
and the stack does learn.** That pairing is the finding. Trained end to end on a position task the
stack reaches 0.010–0.016 against 0.53–0.56 with the labels shuffled, so the gradient works and the
layers move — while object selectivity stays where random initialisation left it:

| run | mean d-prime | per seed |
|---|---|---|
| random init | **0.532** | 0.339, 0.499, 0.757 |
| trained | **0.511** | 0.423, 0.313, 0.796 |

The direction differs by seed and the random control's own spread is wider than any effect.
**Recognition is not a by-product of an objective that does not need it**: position is a linear readout
of a spatial map, so the gradient's cheapest route is to preserve spatial layout rather than to become
selective for what is there. The next thing to try is therefore an objective that *requires* identity,
not more layers or more updates.

**Two measurement errors of mine, both inside one turn.** A **single seed said the opposite** — at
seed 7 alone, random 0.676 against trained 1.157, which I reported as "1.7× the control" before
running the other seeds; at seed 11 training made it *worse*, 2.276 → 1.995. Fourth single-seed or
aggregation artifact recorded here. And **the first metric could not fail**: with objects of different
*colours* a random filter has a random colour preference for free and scores **1.532 before any
training**, 3–9 of 12 units already above 1.0. The shipped fixture gives all three objects the same
colour and different shapes, and a test keeps the colour version's triviality measured rather than
described.

**The shuffled-label control is vacuous and is recorded as such** so it is not quoted later: its 0.090
comes with unit liveness of 0–58% against 75–100%, so training on noise kills the units and the low
score is the dead-unit explanation rather than evidence about learning.

Eleven mutants on the backward pass, one real gap: `output_scale` defaults to 1.0, so dropping it from
the backward pass was invisible until the check swept that axis — the same "computed and never
compared" shape as the `gain_norm` gap found twice before.

**The demo runs on the new structure now — `--input learned`, and the trade cuts both ways
(2026-08-14).** The differentiable backbone made a third input source expressible: the same
convolutional stack the demo already built, with its weights moved by the head's own residual.
Measured on a three-object tracking fixture with an RLS head, against the two sources that already
shipped:

| source | tracking error | frozen, then the scene brightens ×1.6 | ms/frame |
|---|---|---|---|
| raw pixels | **0.0010** (510×) | 0.6506 — **0.78×, worse than guessing** | 0.07 |
| frozen random backbone | 0.0241 (21×) | **0.0816 — 6.19×** | 0.68 |
| **learned backbone** | **0.0163** (31×) | 0.1658 — 3.05× | 0.70 + **2.42 backward** |

**Learning the features beats freezing them at tracking and costs generalisation.** The learned stack
fits the condition it trained under more tightly than a random projection does and transfers worse
when the light changes; raw pixels win tracking by a wide margin and fail the condition change
outright. None of the three dominates, so it is a flag and the default does not move.

**And it corrects a claim recorded here twice.** *"What a backbone buys is invariance, and a random
one does not buy it"* was measured with an **NLMS** head whose frozen control read 0.89× — worse than
guessing — so it compared two failures. With RLS the frozen control is meaningful and the random
backbone reaches **6.19×** against raw pixels' **0.78×**. A random backbone *does* buy invariance; the
earlier fixture could not see it. Same shape as the recorded note that the first invariance fixture
"measured nothing" for exactly that reason — the correction was applied to the fixture then and not
to the conclusion.

**`learn_features` is module-level**, for the reason six other things in `demo/pointer.py` are, and
the caller gates it on `report.updated` — which also stops it while the guard is frozen, since a
frozen guard applies nothing. A fault is precisely when you do not want it pushed into the
representation. The learned weights persist through the **same two-slot record format** as the head,
in their own store: `pack` needs only `state`, `dtype`, `n_features` and `n_outputs`, all of which
`Backbone` already had.

**Two properties found by testing rather than by reading.** At initialisation `theta` is all zeros,
so there is **no gradient into the features at all** until the head has learned something — real, not
a defect, and pinned. And a test that builds the composition without calling `set_external` sees a
head that never learns: `Bound("external_0", lo=0.35)` is evaluated from the first update, reads 0,
and freezes before anything is applied. Third occurrence of that shape here.

**Eight mutants, and the two survivors were both real gaps.** Taking `theta[1:]` instead of
`theta[:-1]` — dropping the first feature's row rather than the constant's — **still trains**, just
against a shifted objective, and survived every accuracy assertion; it is caught now by checking the
applied step against central differences of the real loss, the same discipline the convolutional
gradients themselves are held to. And a mutant letting `learned` fall through to raw features was
invisible to width, because `RAW_FACTOR` was deliberately chosen to make the raw downsample 144 —
the backbone's own width. Values discriminate where widths cannot, and `--raw-grey` is what separates
them for `frame_width`.

**Running it found a defect I had just introduced, and fixing it naively made things worse
(2026-08-14).** The calibration branch tested `source == "backbone"`, so adding a third source that
also reads the backbone left `--input learned` with the scale **computed, printed, and applied
nowhere** — `output_scale` stayed 1.0 and `basis()` ignores its `scale` argument on that branch. The
run printed `feature scale 19.6` while nothing used it.

**Applying it made the result five times worse**, and the mechanism is the finding: `output_scale`
multiplies the **backward** pass as well as the forward one, so calibrating without touching the rate
takes an effective feature step 29× too large. Measured, three objects, RLS head, 900 updates:

| calibration | feature lr | vs a centre guess |
|---|---|---|
| not applied (what shipped) | 0.02 | 31.00× |
| applied, rate unchanged | 0.02 | **1.81×** |
| **applied, rate ÷ output_scale** | 0.00068 | **36.21×** |

So the two halves belong together and both now ship: `calibration_scales` routes the scale per source
and `feature_step` divides the rate by it. Dividing by `output_scale²` reads 37.51× and is **not**
done — one factor is what the mechanism explains and the second would be fitted.

**This is the recorded rule "scale the input, it is not optional" meeting a case it was never
measured on**: it was derived for a *frozen* backbone, where nothing propagates back through the
scale. Seventh piece of logic to need moving out of `main()`'s loop, and three mutants — including
one restoring the exact defect — are caught.

## argmind must do the learning and inferencing itself — decided 2026-08-14

**The owner's words**: performance will keep improving as the architecture is updated, and the
important thing is that **argmind must do the learning and inferencing by itself**. Recorded as a
decision, as the 2026-08-09 widening and the 2026-08-13 purpose were, and the current numbers are
explicitly not the argument against it.

**What it rules out, concretely.** The demo's perception is **1,396 lines of classical code** in
`demo/label_sources.py` — k-means segmentation, connected-component splitting, ring contrast,
appearance matching, region tracking, an object memory — plus the assembly in `pointer.py`. argmind
never learns any of it: those algorithms *produce the labels* and the 145 → 6 head consumes them. The
block diagram made that visible and it is what the decision is aimed at. Under it, that code is
scaffolding to be replaced, not a component to be improved.

**What is already built toward it.** The pieces exist and two are measured:

- **The map output.** `demo/map_head.py` makes the head emit a segmentation map rather than
  coordinates, with past maps as a resolution pyramid. Measured **IoU 0.98 against a per-frame
  segmenter's 0.50** on same-colour twins — the first result here where a head beat the classical
  detector it learns from. It is **not wired into the demo**: that needs a 108-output head, a
  108-wide `taught` stream and a different overlay.
- **A differentiable backbone.** `Backbone.backward` exists and is verified against central
  differences, so the features can be learned rather than projected.
- **Arbitrary configured outputs.** `MultilayerPerceptron(n, m, hidden=...)` and
  `RuntimeShapedMlp<T>` already take their widths at construction on both sides of the C ABI.

**The one obstacle that is measured rather than guessed: where the training signal comes from.**
If argmind is to learn the labelling itself, something must supply the target, and the three
candidates are not equal:

| source of the signal | measured |
|---|---|
| self-supervision (predict the next embedding) | **fails** — a linear map to the next embedding reads a diagonal of 0.004 against an identity's 1.0 |
| an aiding sensor that is blind sometimes | `CadenceDetector` **1.31×** a hold-last baseline; `MotionLabeller` **0.87×**, i.e. worse than holding |
| the classical segmenter as a bootstrap teacher | works, and is what the map-head result used — but it is the thing the decision wants removed |

So the honest shape of the work is **bootstrap then wean**: let the classical labeller teach the map
head, then measure whether the head still holds up as the teacher is withdrawn. That is a
falsifiable programme rather than a hope, and the withdrawal is the experiment.

**And today's negative result tells us which knob to turn, not that the goal is wrong.** Learning the
hidden layers on a *position* task left object selectivity where random initialisation put it, 0.511
against 0.532. Position is a linear readout of a spatial map, so the gradient's cheapest route is to
preserve layout rather than to become selective. **An objective that requires identity is the
untested case**, and a map output is exactly such an objective — which is the second reason to wire
it in.

**The first step under that decision is built: `--output map` (2026-08-14).** The head emits a
9x12 segmentation instead of coordinates, with past maps fed back as a resolution pyramid, and the
classical labeller supplies the target rather than the answer. A map output has **no slots** —
nothing to permute, nothing to count, no assignment — which removes three measured problems at once.
Shapes: 295 -> 108 under a guard, `taught` widened to 108, `map_teacher` isolated as one named
function precisely because it is the thing to be withdrawn. The points head is untouched.

**The wean is the experiment and its result is NOT yet usable, which is recorded rather than
glossed.** With the teacher the head reaches IoU **0.80-0.84** against its own labeller; withdrawn,
it free-runs at **0.09-0.16**, against 0.92 for simply repeating the previous true map. **But this
harness disagrees with `test_map_head.py`'s, which free-runs at 0.96-0.98 on the same scene, and I
have not localised the difference.** Three explanations tested and refuted: *exposure bias* (training
on the head's own prediction gives an identical 0.80/0.10), *the head* (a bare RLS reads 0.02 free,
worse), and *my dropout default* (0.0, 0.05 and 0.15 all read 0.10). So the composition wires up and
learns, and the wean number cannot be quoted until the two harnesses agree.

**And a mechanism I proposed for the blackout failure is refuted, with the reason.** `map_head.py`
records that free-running through a blackout goes to 0.00 and that *resetting* the pyramid does not
help; the guard already knows a frame is untrustworthy, so **holding** the pyramid looked like the
missing option. It makes no difference at all. Measured against blackout length, the object orbiting
in 160 frames: 5 frames gives 0.72 either way, 15 gives 0.24/0.22, 40 and 60 give **0.00 for both**.
**The failure scales with how far the object moved, not with what the memory was fed** — so the
memory's problem is staleness rather than corruption, and the recorded "a reset lands in a
distribution that never occurred" is not the whole story, since an in-distribution held memory fails
identically.

**"how to set the target map? and is the training just a substitution between two maps?" — the
second question is right, and the answer corrects the map head's own headline (2026-08-14).**

The target is three steps and no more: the classical labeller's `field`, then `field > 0` for a
boolean foreground mask at 72x96 — **object identity is discarded** — then `block_mean` to 9x12, so
each of the 108 cells is the *fraction* of its 8x8 block that is foreground. Soft, not binary.

**And the training is effectively a map-to-map substitution.** Ablating the basis by zeroing halves
of it — same model, same widths, only what it is told differs:

| input | head IoU |
|---|---|
| frame + memory (what ships) | 0.98 |
| **memory only, frame zeroed** | **0.98** |
| frame only, memory zeroed | 0.41 |
| repeating the previous true map | 0.92 |

**Zeroing the camera costs nothing on that fixture — and I then overstated what it meant, which the
next question caught.** "The frame adds 0.00" is true of the **twin** scene only, where the target is
one of two *identical* blobs and a single frame genuinely cannot say which; the memory is not a
shortcut there, it is the only available source of the answer. Two controls settle it:

| scene | frame only | memory only | persistence |
|---|---|---|---|
| orbit, twin | **0.41** | 0.98 | 0.92 |
| orbit, no twin | **0.99** | — | 0.92 |
| **jump** — teleports every frame, no twin | **0.95** | **0.00** | 0.10 |

The jump row is decisive: with the previous map carrying no information whatever, the head reaches
**0.95 from pixels** and the memory contributes nothing. **So it does learn a perception map**, and
the honest statement is narrower than "it ignores the camera": the head uses whichever input can
answer the question it is asked, and leans entirely on memory where one frame is underdetermined —
which is also why free-running collapses over a long gap.

What survives of the caution: `map_head.py`'s "most of that 0.98 is persistence" stands, and the twin
headline is a *tracking* result rather than a perception one. What is withdrawn is my own stronger
claim that the frame is ignored in general. Both controls are checked in, so the scope cannot be lost
again.

**And a timing test of this project's own preferred kind failed anyway, which sharpens the rule.**
`test_the_gate_is_cheaper_per_frame_than_adapting` asserts a **ratio measured inside one run** —
exactly the form adopted after the absolute-millisecond gate failed earlier — and it read 0.23 alone
and **0.78 inside the full suite**, failing its 0.7 bound. A ratio cancels the machine's *speed* and
not its *load*: the gated path is dominated by fixed per-step overhead and the adapting path by
per-frame arithmetic, so contention inflates them unequally and the ratio drifts toward 1. The
**estimator** was changed, not the threshold — `min` over three repeats, which is the standard
estimate of an uncontended cost, and "the gate skips work" is a claim about work rather than about the
scheduler.

**"the core blackbox need to any to any and the initial start runs without any pre-trained
background" — both checked rather than asserted, 2026-08-14.** Two owner requirements, and the
measurement is in `tests/test_any_to_any.py`, driven through the *whole* composition — `Stream` →
`Aligner` → `FrameMap` → `Guard` → adapter under `Pipeline.step()` — so a shape that only worked in a
bare adapter would fail.

| n → m | RLS | NLMS | MLP(8) |
|---|---|---|---|
| 1 → 1 | 0.0000 | 0.0000 | 0.92 |
| 1 → 7 | 0.0000 | 0.0000 | 0.26 |
| 6 → 3 | 0.0000 | 0.0000 | 0.20 |
| 40 → 12 | 0.0000 | 0.0821 | 0.62 |
| 128 → 64 | 0.0000 | 0.3525 | 0.86 |
| 12 → 200 | 0.0000 | 0.0009 | 0.66 |

relative error after 400 samples. **Any-to-any holds**: every shape builds and runs with no special
case and no refusal, both directions — 12 → 200 is sixteen times more outputs than inputs and is the
shape a segmentation map takes. **RLS reaches the exact least-squares solution at every one of them.**
NLMS degrades with width, which is its recorded convergence rate rather than a shape limit, and 400
samples is far too few for SGD, which this file already records as worse than a constant in its first
few hundred.

**Cold start holds everywhere, and "no pre-trained background" is reported rather than assumed to be
zero** — for two of the three adapters it is not: NLMS starts at exactly **0.000**, RLS at **100.000**
(θ is zero; the 100 is `p0` on the covariance diagonal), the network at **~0.5–1.0** of seeded Glorot,
because zero weights would make every hidden unit identical with identical gradients. Nothing anywhere
imports a trained weight.

**The one place any-to-any stops is the C ABI**, and it is a declared list rather than a hidden
limit: 25 `AM_DECLARE_*` instantiations, because the freestanding core keeps shapes in the *type* so
the memory is declared. Adding one is three lines and a rebuild. The hosted tier has no list —
`RuntimeShapedRls/Nlms/Mlp/Guard/Pipeline` all take their shapes at construction. So any-to-any is
available at the tier that can afford a heap and enumerated at the tier that cannot, which is the
trade `argmind.h` already documents.

**And adding the mode left the persistence fingerprint blind to it (2026-08-15).** A points record
and a map record hashed **identically** — `0x224c2fa3` both — because `fingerprint()` covered the
backbone plan, input shape, pooling, embedding width, object count and input source, and not the
readout. What refused a cross-mode load was `unpack`'s **shape** check, "record is for 145x6 but this
adapter is 295x108", and that protection is **accidental**: it holds only while the two state sizes
differ, and the fingerprint exists precisely for the case where the bytes are valid and mean
something else. A map head's basis also carries 150 pyramid values whose meaning depends on the band
layout, which no length check can see. The mode, the grid and the memory width are all in it now.

**The first test for it passed for the wrong reason, and the mutants said so.** `fingerprint` puts
the mode in *and* appends the map geometry on the map branch, so **either alone** makes a points and
a map hash differ — a mutant dropping the mode survived, and so did one dropping the geometry, each
masked by the other. The two contributions are asserted separately now, the mode against a
hypothetical third readout. **Two more survivors were real gaps**: dropping `map_memory_width()`
while keeping the grid, which a re-split band layout at the same total width would walk straight
through; and `main()`'s call site **never passing the mode at all** — the argument has a default, so
the omission was silent while every direct-call test still passed. `fingerprint_for` is that call
site, module-level for the reason seven other blocks in this `main()` are. Five mutants, all caught.

**Still undecided:** whether inference is *hosted* (an interface to ONNX Runtime/TFLite) or
*embedded*, and whether the tiers ship as one library or two. Nothing on the camera/audio/speech
path is built.

## Learning applies everywhere, deep structures allowed, low to high — decided 2026-08-16

**The owner's words**: `제약사항이 있다면 재검토바람. 학습은 모든부분에 적용됨. 딥러닝네트워크
구조를 사용가능하고, 저수준 인지부터 고수준 인지까지 가능한 형태여야함`. Recorded as a decision,
like the 2026-08-09 widening and the 2026-08-13 purpose. What follows is the re-audit it asked for,
and **most of the constraints I had been reporting turn out to be mine rather than the problem's** —
the same finding this file already records for `AppearanceMatcher`'s `k > 4` refusal and for the
withdrawn "training does not fit under 4.2 MB".

**Withdrawn — not constraints:**

- **Depth is not limited by anything.** Measured: `MultilayerPerceptron` at 16 hidden layers is
  67,600 parameters in **one contiguous state block**, and a 6-layer network goes through `Guard`
  unchanged — checkpointed, rolled back and persisted like any other adapter. The contract never
  mentioned depth; I had been arguing from the *core tier's* 8 KB, which is a different question.
- **RLS refusing to widen `n_outputs` was load-bearing in the wrong argument.** It is what produced
  "unbounded vocabulary, bounded concurrency", and it is a property of **RLS's row-major theta**, not
  of the architecture. `MultilayerPerceptron` has both `widened()` and `deepened()`. With a network
  head that constraint is simply absent.
- **"Recognition does not emerge in the hidden layers"** is a result about the **objective**, not the
  architecture. It was measured on *position*, which is a linear readout of a spatial map, so the
  gradient's cheapest route is to preserve layout. This file already scoped it on 2026-08-13; it had
  been reading as an architectural limit and it is not one.
- **The 8 KB / Cortex-M4F floor is the core tier only.** The hosted tier takes shapes at construction
  and has a heap; an STM32N6 has 4.2 MB against training arithmetic of 2.59 MB.

**What is real, and it is one thing stated precisely.** The `Adapter` contract is
`update(x, y) -> UpdateReport`, and `UpdateReport` carries `applied`, `residual`, `residual_post`,
`gain_norm`, `p_trace`, `p_min_diag`, `reason` — **no gradient term**. So:

- a **monolithic** deep learner fits perfectly, because it owns all its layers and needs no term for
  passing anything between them — which is why 16 layers work;
- a **hierarchy of separately-guarded levels does not**, because a hidden level has no `y` to be
  given and returns nothing a level below it could descend on.

**And the conv stack is not an adapter at all.** `Backbone` has `state`, `load_state`, `n_features`,
`n_outputs` and `state_size`, and is **missing `update`, `predict`, `reset` and `health_fields`** — so
it cannot go under a `Guard`, cannot be checkpointed or rolled back, cannot be persisted through the
same record path, and cannot sit in a `Pipeline`. Feature learning is **42 lines inside the demo**
(`demo/pointer.py::learn_features`), not in the library. There is also no composite type — no
`Sequential`, `Stack` or `Hierarchy` anywhere in `argmind`.

**So "학습은 모든 부분에 적용됨" is already true inside one adapter and false across two**, and the
gap is a contract extension rather than a relaxed number. That is a deliberate design change and it
is recorded here before being made, because the contract is what `Guard`, `Pipeline`, `persist` and
the C ABI are all written against — this file's own note that "if the guard needed one change for a
network, the contract would not be one" is the standard the extension has to meet.

**Nothing here is a measurement of whether it works, only of what is expressible.** The backward
throughput on an M55 is still the number nobody has.

## Is this a cognitive brain? Asked 2026-08-16 — no, and the gap is specific

**The owner's question**: `여전히 의심스러운 부분이 있음. 인지적 지능기능을 가진 두뇌를 만드는게
맞는지 확인바람`. The suspicion is correct, and the answer is recorded here because nothing in this
repository said what it is *not* at the system level — DESIGN.md's *What it is not* is about the
model class, one level down. `tests/test_what_this_is_not.py` protects it, since a negative result
left in prose is the kind this project has already had to withdraw twice.

**Measured, not argued.** The demo's perception is **1,087 lines of hand-written classical code**
(`demo/label_sources.py`) plus 1,013 of assembly; what argmind learns is **one linear map** — 145→6
at 870 parameters for points, 295→108 at 31,860 (124.5 KB) for the map — and `predict` is literally
`np.dot(arr, self._theta)`. The backbone's 1,240 parameters are **frozen** unless `--input learned`.
So the perception is not learned; it is consumed. That is exactly what the 2026-08-14 decision is
aimed at, and the test states the starting position so the wean is visible as a *failing* test
rather than as a claim.

**Eight faculties a cognitive brain is usually taken to need are absent**, not unfinished: a goal the
system holds for itself, action selection, episodic or working memory, attention, concept formation,
language, audio input, a self-model.

**And the measurement I reached for first was broken in the flattering direction.** A keyword grep
over the sources reported four of those as PRESENT. Every hit was a false positive — `policy` is the
*aligner's* synchronisation policy, `action` is a substring of `subtraction`, `mic ` of `dynamic
range`, `language` of "the language decision in DESIGN.md", and `reward`/`salien` are the retired
objective-sweep harness. Checked against the library's public API instead, all eight are absent. The
substring trap is pinned by a test so the same grep is not quoted again.

**Two of my own test defects on the way, both the shape this file keeps recording.** The control that
was supposed to show the network is *not* linear passed **vacuously on NaN** — an identity-activation
mutant made it diverge, and `nan != approx(nan)` is True, so the mutant survived while the assertion
"held". And the first fix made the test brittle to *training stability*: forcing `relu`, a legitimate
activation, diverged at that rate and failed a test it has no business failing. The control is an
**untrained** network now, because the claim is about the form of the map, which initialisation
already settles. Six mutants, all behaving as intended including the one that must **not** be caught.

**What is not in question**, and is deliberately not argued down: the online adaptation, guard,
rollback, persistence, any-to-any shapes, cold start with nothing pretrained, and execution on the
declared floor are measured and hold. The gap is between that and the stated purpose of 2026-08-13,
not in the built part.

**Four things would change the answer, and all four are open.** The wean must succeed — it currently
free-runs at 0.09–0.16 while a second harness reads 0.96–0.98 on the same scene and the discrepancy
is unlocalised. An objective that *requires* identity, since position provably does not (d-prime
0.511 trained against 0.532 random). A throughput figure for a backward pass on the M55, which has
never been measured. And memory, goal and attention, which do not exist at all.

**The naming risk is this file's own precedent one level up.** CLAUDE.md's first line was already
corrected once for overselling — "Multi-sensor AI software: fusion, on-device learning, and
inference" against a repository containing one algorithm. "24/7 brain" is the same shape of claim
about the whole system. It stands as the **stated purpose**, recorded as a decision on 2026-08-13;
it is not a description of what is built, and the two must not be quoted as if they were the same
sentence.
