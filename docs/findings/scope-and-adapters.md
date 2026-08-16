<!-- Moved out of CLAUDE.md on 2026-08-16 so the file loaded into every session stays
     small. The text is verbatim; nothing was summarised or dropped. CLAUDE.md carries a
     one-line pointer to each of these. -->

# Scope, adapters and precision — the 2026-08-09 widening and what it cost

## Status — scope widened 2026-08-09, and most of the widening is not built

**The 2026-08-08 scope was deliberately reopened by the project owner on 2026-08-09.** The old
table said the decisions were "not to be re-litigated without a reason"; the reason given is a
target application that needs **camera and microphone input, situation assessment, inference,
actuator output and speech output**. A 24-byte linear model does not do any of that, and saying
so did not change the decision, so the decision stands and the scope moves.

| axis | 2026-08-08 | now |
|---|---|---|
| platform | embedded / edge | **embedded / edge *and* an SBC-class tier** — two floors, not one |
| learning | on-device online adaptation | unchanged, **but no longer one algorithm** |
| model class | (unstated — the defect below) | **was RLS only; must admit others, including hosted inference — and since 2026-08-10 the platform generates the basis itself rather than receiving it** |
| sensors | sensor-agnostic | unchanged, and now genuinely tested against it |
| deliverable | a library, not an application | unchanged — **speech and situation logic are a caller's, not this library's** |
| language | C++ core behind a C ABI | unchanged for the core tier |
| target floor | Cortex-M4F class | **unchanged for the core tier**; the new tier has its own, undecided |

**What survives is most of it, and that is measured rather than hoped.** `Guard` and `Pipeline`
are templates over `Adapter`, and the contract they require is small — `update`, `predict`,
`state`, `load_state`, `reset`, `health`, `frozen`, plus `n_features`/`n_outputs`/`state_size`/
`scalar`. A second adapter with a **different algorithm and no covariance matrix at all**
(normalised LMS, O(n) against RLS's O(n²), state `N*M` rather than `N*M + N*N`) goes through
both templates **unchanged, compiling clean at `-Wall -Wextra` on the host and on Cortex-M4F
freestanding**, learns correctly through the guard, and the guard shrinks with it: 432 bytes
against RLS's 1,144 at the same depth. `cpp/tests/test_core.cpp` carries it, so the claim the
widening rests on is protected rather than asserted.

**The one place the contract is not model-agnostic, found by that probe.** `health()` is two
*covariance* scalars, and `Field::PTrace`/`Field::PMinDiag` are two of the guard's five
checkable fields. An adapter without a covariance must report zeros, and then the same bound
breaks in both directions: `at_most(PTrace, 1e4)` is **dead** — configured, reporting healthy
on every sample, unable ever to fire — and `at_least(PTrace, 1e-9)`, a reasonable sanity bound
on RLS, **trips on the first update and freezes the adapter**. Both are pinned by a test with
an RLS control. **Closed the same day**: health is an **adapter-declared set** now, and a check
on a field the adapter cannot report is refused where the caller can act on it — `add_bound` in
C++, `Guard.__init__` in Python. `Divergence` is covered too (it watches `residual_norm` alone,
and it is the check most likely to be enabled by someone who does not know the problem's scale);
`health()` stops being mandatory, since the guard's calls sit behind `if constexpr` and an
adapter declaring neither covariance field need not implement it; and a `Check` now declares
what it `watches`. That last one closed a hole nobody was looking for — `WATCHABLE` admitted
`applied`/`reason`/`residual`/`residual_post` and `Bound.evaluate` does `float(getattr(...))`,
so `Bound("reason")` used to construct fine and crash at evaluation. **No ABI query was added**:
every guard the C ABI exports wraps `Rls`, so it could only ever return one constant, and this
project has already shipped two entry points nothing called. Three mutants confirm the fix (5
Python failures, 3 C++, 1 more for divergence) rather than the suite merely passing.

**Also exposed by the reopening: the model class was never written down.** "Linear in the
parameters" appeared nowhere in the repository while four bullets of DESIGN.md's *What it is
not* explained that it is not a CNN. That is fixed; see the first section of this file.

**The second tier's floor is decided, and the answer is that it does not move.** The obvious
reading — camera and audio mean an SBC — is wrong, because of a part class that did not exist
when the floor was set: an **MCU with an NPU**. The **Cortex-M55** is ARMv8.1-M with *native
scalar and vector fp32* ([Arm](https://developer.arm.com/documentation/101273/latest/Cortex-M55-Processor-level-components-and-system-registers---Reference-Material/Floating-point-and-MVE-support)),
so it is **strictly above** the declared floor; **STM32N6** pairs it at 800 MHz with a 600 GOPS
NPU at 3 TOPS/W, 4.2 MB SRAM and a MIPI CSI-2 camera pipeline
([ST](https://www.st.com/en/microcontrollers-microprocessors/stm32n6-series.html)). Measured, no
source changes: the composed pipeline is **smaller** on M55 than on M4F at identical RAM (the
two figures are in `cpp/target/SIZES.md`, because the pair quoted here went stale in a day) —
and **all 4,650 self-checks run and pass
under QEMU's `mps3-an547` Cortex-M55 board**. `tests/test_target_run.py` is parametrised over
both cores, mutation-checked two ways.

**The fork is what the words mean, not what the silicon can do.** React like an animal (see a
person, hear a call, answer with a sound) fits an MCU+NPU at ~1 W and the floor is unchanged;
*converse* (speech recognition, dialogue, natural TTS) needs an SBC at 10–25 W, and that costs
**40–63 % of standing endurance** — **my arithmetic**, not tomcat's figures: a 41.8 Wh battery
and ~14.9 W standing draw are *derived* from its stated NFR6 (83.6 W trotting for 30 min, 168
min standing, in `docs/REQUIREMENTS.md` of `YounghyeonPark/tomcat`), which itself carries an
`[assumed]` 175 Wh/kg.

**Learning and inference can be integrated, at one forced place: the head.** The NPU does not
learn — Arm calls Ethos-U55 an *inference* accelerator and its path is TFLite **Micro**
([Arm](https://developer.arm.com/Processors/Ethos-U55)) — and backprop's cost is retained
*activations*, not weights. **The sentence that used to follow this one said end-to-end on-device
training "is not available under 4.2 MB and no heap", and that is WITHDRAWN as of 2026-08-13**: the
arithmetic is 2.59 MB against an STM32N6's 4.2 (weights 821.6 KB + gradients 821.6 + retained
activations 1,012.5, at 96×96 α=0.25 in fp32), so it fits with room to spare, and it is 224×224 that
does not at 6.9 MB. `Backbone.training_activation_values()` reports it rather than leaving it in
prose, and a test carries the tier arithmetic. **What blocks it is compute, and that has never been
measured**: the NPU cannot train, so a backward pass runs on the M55 in software and no throughput
figure for that exists anywhere here. What is available is a frozen int8 backbone on the NPU whose feature vector **is** the
regressor for an `Adapter` under a `Guard`: inference and learning share the computation rather
than duplicating it, and one flat state block still covers persistence. Measured on both cores,
a 128-wide head: `Guard<Rls<f32,128,1>,4,4>` is **330,544 B** against `Guard<Nlms<...>>` at
**2,872 B** — **115×**, which is arithmetic on the state sizes. An accuracy comparison stood
beside it ("~17,000×") and is **withdrawn**: it came from a scratch fixture never checked in,
an audit reconstructing it got the opposite ordering, and NLMS's μ — which sets its
steady-state error — was never swept off its default. **The wall
is RLS's n²**: head 128 costs 7.5% of the SRAM, 256 costs 30%, and **512 does not fit at 119%**,
so a wide head forces a square-root form, a shallower ring, or NLMS. What it does *not* give:
no representation learning, the backbone is frozen, and "situation assessment" is the backbone's
job. What it does give is the reason argmind belongs here — an online-adapted head is the only
part still moving after deployment, so freezing, rollback, checkpointing and the record format
all transfer unchanged.

**The label arrives after the features that earned it, and getting that wrong is silent.** Honest
self-supervision on a robot is late: predicted at `t`, confirmed at `t + Δ` by something slower
and more physical. `allow_non_causal` does *not* express this — it permits a policy to pick a
future sample, it does not declare a lag. **No library change is needed**: make the outcome the
reference so instants are label arrivals, and **stamp each feature with the instant it is about
rather than when it was computed**. Measured at Δ=0.30 s, features 50 Hz / labels 10 Hz —
stamping by computation time gives relative error **1.105**, a different map entirely; stamping
by the instant it is about gives **1.56e−04**, which is not error but the `P0⁻¹` prior term,
confirmed by the 1/p0 law across four decades (×9.85, ×9.98, ×10.00, ×10.00). **No guard catches
the wrong one** — the adapter is healthy and `residual_norm` is low, because it fits the wrong
map well; it is being asked the wrong question, and a guard cannot see that. Both halves are in
`tests/test_pipeline.py`, mutation-checked.

**The guard can watch the caller's own scalars now, which is the middle path made real.**
Every check watched the *adapter's* health, and for a frozen backbone feeding an adapted head
that is the wrong place to look — the head stays healthy because it fits whatever it is given,
so an input shift reaches an early layer's statistics long before the head's residual. The
guard cannot compute those numbers; it can watch them. `Guard<Adapter, Depth, MaxBounds,
NExternal>` takes up to four, `set_external(i, v)` supplies them, and `Bound` on
`external_field(i)` (`"external_i"` in Python) freezes with a cause, a counter and automatic
recovery. Declaring none costs **zero bytes** — held as a member the empty slots measured
1144 → 1152, so they are an empty private *base* and a `static_assert` checks the optimisation
rather than trusting it. Externals are **inputs, not state**: a rollback does not restore them,
or a trip would resurrect the value that caused it. Non-finite is accepted at the call and
unhealthy at the check, because a NaN out of a runtime is the event the guard exists for.
Both sides freeze at exactly the shift step; six mutants confirm it (3 C++, 3 Python).

**In the C ABI now, and the recorded question — how many slots — is answered with four**, the
same number `Depth` and `MaxBounds` already are and for the reason the header already gives
for them: they are the memory. A C caller cannot choose a template parameter, so declaring
zero would leave the feature unreachable from the one boundary it exists to cross; the C++
side keeps its zero-cost default for anyone who can pick. `AM_FIELD_EXTERNAL_0..3`,
`am_guard_*_set_external` and `am_pipe_*_set_external`, with a `static_assert` tying the
header's numbering to `Field::External0` — a C caller bounding a different slot than it set
would be silent otherwise.

Measured end to end in C99 on the backbone shape: an external bound is accepted, an
out-of-range slot is refused rather than wrapped, and the pipeline **freezes at exactly the
step the caller's scalar shifts** while the head's own residual never moves. The cost is
**32 bytes per guard**; the target rows moved 64 because the exported `MAX_BYTES` round to the
next multiple of 64, and `f32_12_1` composed is now at **97.4% of the declared 8 KB floor**.
The `MAX_BYTES` `static_assert` caught every stale bound at the first build, for the third
time. Three mutants: wrapping an out-of-range slot, narrowing `valid_field` back, and
declaring no slots — all caught.

**There is a `demo/` now, and it is deliberately outside the library.** `python -m
demo.pointer` runs argmind off a webcam: a dot follows your hand after you teach it by
dragging, and the same window shows the two-rate alignment, drift re-convergence, the guard
freezing when you cover the camera, thawing when you uncover it, and persistence across a
restart. **Nothing under `argmind/` or `cpp/` imports it**, and it has a dependency the library
must never acquire — OpenCV. A demo that blurred that line would argue against the design it
demonstrates.

Measured: **931 fps** for backbone plus pipeline on the host, so a 30 Hz camera is not the
constraint; position error **0.035** after ~400 taught frames against **0.24** for guessing the
centre; head state **1,160 B**.

**Three of its design choices came out of measurements that rejected the first attempt.** The
architecture is shallow and grid-pooled because random convolutional features **work shallow
and die deep** — brightness reads 0.009 at one layer and **1.119 at twenty-seven**, worse than
a constant — and because global average pooling discards *where*: a position task reads 1.076
globally against **0.231** on a 3x3 grid. `grid_avg_pool` is in the library for that reason, in
both implementations, with parity cases including a grid that does not divide the map.

The guard's external check watches the **raw frame's standard deviation**, after two other
statistics were measured not to separate: distance from a baseline embedding saturates
(covered 1.000, a dim room 0.877) and embedding norm overlaps outright (0.604 against 0.578).
Frame std is 0.000 covered against 0.194-0.197 for every legitimate scene — and it is the right
*kind* of statistic, because a flat grey frame gives a plausible embedding the head fits
happily. Setting the first threshold by eye froze the demo on its second frame. And you teach
by dragging rather than clicking because NLMS at 145 features needs a few hundred updates:
0.058 at 300, nowhere at 50.

**`demo/test_pointer.py` drives the whole composition headlessly** with synthetic frames and
runs in the root suite, needing no camera and no OpenCV — `cv2` is imported inside `main()`.
It carries the control the fault test needs: the same covered camera destroys an *unguarded*
head, so freezing prevented something. **And it walked into the preallocated-prediction trap
for the third time in this repository**, reporting that teaching barely worked (0.256 → 0.227)
on a composition that reaches 0.035.

**Running it found a demo bug of the worst shape: it printed "saved" and saved nothing where
it said.** A `Store` takes a base *path* — its slots are `<path>.0` and `<path>.1`, and its
docstring says exactly that — and the demo passed the *directory*, so the head landed as
`demo/state.0` beside it while the message named `demo/state`. The library was right and the
caller was wrong, which is the reading that matters: it **saved correctly and reported the
wrong place**, which is worse than failing. Pinned by a test, and the mutant that points
`Store` at a directory again is caught.

**The platform has its own backbone now** (`argmind/backbone.py`), and it exists because a
previous answer of mine narrowed the owner's decision without the authority to. Asked whether
camera hardware needs a HAL, I said no *and* that a backbone belongs to the vendor's runtime —
the second half was not the decision of 2026-08-10, which said the platform needs **its own
inference network**. The owner asked. It was wrong.

**Two things I had conflated.** Using an NPU does require the vendor toolchain: the Neural-ART's
instruction encoding is proprietary and its weights go through ST's compiler. Implementing a
convolution requires nothing. So both paths to the same feature vector are supported — run the
backbone here, or push a vendor runtime's embedding into a `Stream` like any other sensor — and
the seam is the vector.

**The tier was decided by arithmetic before a line was written.** The smallest useful vision
backbone, MobileNet v1 α=0.25 at 96×96, holds two adjacent activation buffers and the largest
pair is **54 KB in int8, 216 KB in fp32**, against a declared core floor of **8 KB for the
whole library**. At 224×224 it is 294 KB. No implementation changes that, and
`peak_activation_values()` reports it rather than leaving it in prose.

**Convolution is checked against the definition written as four nested loops**, which shares no
implementation trick with the im2col-and-matmul version — the same discipline as RLS against
the closed-form solution and the MLP against central differences. Agreement is 1e-13 relative
over six shapes, and depthwise is exact.

**The finding that came out of building it corrects something I claimed the same morning.** An
untrained deep stack emits a full-rank but tiny embedding — magnitude **6.8e-4** — and on it
*every* head fails: RLS, NLMS and the network all score ~1.0 against a constant predictor.
Scaling it fixes RLS decisively (0.998 → 0.0015). **`p0` has units**: the gain is
`p0·x/(1 + p0·‖x‖²)`, so at `‖x‖ ~ 1e-4` the data term is `1e-8·p0` and the default 100 is
eight orders of magnitude short — raising `p0` to 1e10 recovers it exactly. **Scaling the
target is free for RLS and NLMS; scaling the input is not**, and I had generalised the morning's
target-scaling result into a prediction that they would be invariant here. Wrong, and both
directions are now pinned by a test.

**L2-normalising each embedding is measured and deliberately not the answer**: it makes ‖x‖
exactly 1 and conditions the prior by construction, but it discards the magnitude, which costs
**150×** against a single global scale on a target that depends on it. So `calibrate_scale()`
*reports* a constant and `output_scale` applies it, rather than the library guessing. A trained
backbone with batch normalisation has none of this problem — that is what batch normalisation
is for — so it is a property of an untrained or naively-converted stack.

**Eight mutants, two of which were real gaps.** The SAME-padding impulse fixture could not
discriminate: at 4×4 with a 3×3 kernel the extra row of padding lands where an impulse at (0,0)
is inside the first window either way, so a top-heavy mutant survived. 5×5 with a 4×4 kernel
pads by three and separates them — one window against four. And a mutant forcing stride 1
survived everything, because `features()` pools its map away so the output *shape* and the
feature count are identical; it is caught now by composing the layer independently.

**What is deliberately absent**: int8, because quantised inference is defined by a scheme
(zero points, per-axis scales, accumulator width, requantisation rounding) and two
implementations that disagree on any of it are two different models; and on-device training of
the backbone, which was already measured unavailable since backprop retains activations and the
forward pass alone is the peak above.

**The backbone is across** (`cpp/include/argmind/host/runtime_shape_backbone.hpp`,
`cpp/tools/bbcheck.cpp`), into the hosted tier for the memory reason above. Measured over six
plans — one conv, one depthwise, a separable block, stride at every layer, a 5×5 kernel, a
non-square input — at both precisions: **within 400 ulp of the reference**, and the full
27-layer MobileNet at 96×96 agrees too. Eight mutants, all caught: top-heavy padding, a
column-major kernel index, a depthwise that mixes channels, relu6 without its upper clip, an
ignored stride, a dropped `output_scale`, a pool dividing by the wrong count, and a
zero-channel plan accepted.

**Classifying the divergence took two corrections and the second one is the finding.** Through
the whole stack the deviation does *not* grow with the sum length — 1.18, 1.88, 0.77 ulp at
pointwise widths of 4, 32, 256 — which is not what reduction order looks like. The first
explanation offered was `relu6` saturating at 6, and it is **measured to be wrong: 0.0% of
activations reach the clip at any width**. What is true is that **55% are clipped to exactly
zero**, bit-identical whatever order produced them, and global average pooling then averages
errors of random sign over H×W. Isolating a single pointwise layer with no pooling and no
stack, the growth is plain: **0.52, 0.50, 1.31, 3.94, 4.17 ulp** at cin of 4, 16, 64, 256,
1024. So it is reduction order, and the stack absorbs most of it — and a test now pins the
*rejected* explanation too, so it is not offered again.

**And it runs on ARM.** `bbcheck` is parametrised `[host]`/`[target]` like every other parity
tool here, so the six plans at both precisions execute under QEMU on the declared floor as well
as on the host, and agree with the reference there too. Convolution is the first thing in this
library whose arithmetic is dense enough for a soft-float or contraction difference to compound
through twenty-seven layers, and it does not.

**Three cases are host-only and the reason is the harness, not the library.** The QEMU image
takes a 64 KB heap out of `.bss` — the AN547's DTCM is 512 KB and a larger array made the M55
produce no output at all — so a 96×96 MobileNet at 840 KB of weights plus 216 KB of activations
cannot run there, and neither can the 256-wide classification case at 147 KB of activations in
f8. **That limit announced itself rather than being assumed**: the wide case was in the target
set first and failed with exit 1 while the host passed, which is also the evidence that the
target path is a genuinely different binary. An STM32N6 has 4.2 MB.

**One harness bug that fails silently:** `struct` does not insert the padding a C++ compiler
puts before a `double`, so the Python encoder packed 28 bytes where `bbcheck` static_asserts
32, and every case came back "not a case file" with nothing on stderr.

**There are three adapters now, and the third is a network** — `MultilayerPerceptron`, the
2026-08-10 scope decision made real. Python only so far. Three of nine mutants exposed real
gaps rather than confirming the suite, and two design choices were measured rather than taken.

**SGD, because Adam did not earn its state.** Adam keeps two moments per weight, so it stores
3x. Compared at *equal bytes* — the comparison that matters when memory is the whole reason a
network is here — SGD wins: at ~28 KB, relu SGD at (56,56) scores **0.1314** against Adam's
**0.1410** at (32,32), with less seed spread. Adam wins at a *fixed architecture*, which is the
comparison that flatters it.

**`tanh` by default, and it is the less accurate one.** At (32,32) over 40k samples relu reaches
0.1391 against tanh's 0.2315. The default is tanh because at `lr=0.1` tanh beats a constant on
all six seeds (0.751 against 0.764) and **relu beats it on none** (0.772–0.831, plus one
divergence); at 0.05 tanh wins all six; at 0.2 both die, so tanh is not claimed safe, only
later-failing. **Two earlier versions of that table were aggregation artifacts** — a mean over
three seeds hid one case, and calling a row "diverged" because *any* seed did turned one seed in
six into a property. It is per-seed now. Third aggregation artifact of the session.

**The gradient is checked against central differences, not against "it learns"** — a wrong
backward pass still trains, just worse. 1.5e-8 to 3.7e-11 over five architectures and both
activations. And at `hidden=()` there is an exact law, the analogue of NLMS's: the a posteriori
error is exactly `(1 - 2*lr*(x.x + 1))` times the a priori one, 1.7e-13 over 600 cases. **The
`+1` is the bias** and the first version omitted it, failing by 20x — the law is tight enough to
catch a missing term, which is what makes it worth having, and the test asserts the version
without the bias does *not* hold.

**Scale the target — this outranks the precision question and was found while chasing it.**
An un-normalised target defeats *every* precision. On `50·sin(0.5s₀) + 20·s₁s₂` at 200k
updates: raw, **fp64** scores 0.7326 of a constant predictor with ‖w‖=3617 and **96.2% of its
steps below the representable spacing**; divided by the target's own standard deviation it
scores **0.0144** with ‖w‖=8.7 and 0.0%. **51× better, in double precision.** It bites at the
other end too — a target with output ~0.001 is fitted four times *worse* than a constant in
every precision until it is scaled up. The docstring now leads with this, because none of the
precision figures mean anything without it.

**It is gradient descent's failure and not the library's**, which is the scoping the first
version of this paragraph left implicit. On the same raw target **RLS reports 0.0% step
underflow and *identical* accuracy scaled or not** — 0.6547 both — and so does NLMS, 0.7691
both: a least-squares solution is scale-invariant, so scaling `y` scales `theta` and nothing
else. Only the network cares (63.6% against 0.0%, 0.7156 against 0.1534). `argmind.bench.
step_underflow` measures it for any adapter, and those two zeros are its control.

**It is a bring-up diagnostic and deliberately not a guard check.** The condition is a
*configuration* error, present from the first update and unchanging, not the developing fault
a guard exists to catch — the distinction the owner's correction of 2026-08-10 drew. **A cheap
runtime proxy was measured and does not work**: `gain_norm / (eps·‖state‖)` correlates with the
true rate at only +0.70 on a log scale and the orderings overlap (1.24e13 for a 74% run against
2.98e11 for a 0% one), because a norm is dominated by the largest weights and gradients — the
ones still moving — while the rate is about the stuck minority. The separation the real
measure gives is total and needs **five probes**: raw 37–97%, scaled exactly 0.0% in every run
of a six-seed sweep.

**And the diagnostic has a limit that had to be measured rather than reasoned about: it is not interpretable on `relu`.** A unit that is off has no gradient, so its weights sit still for a reason unrelated to scaling, and the reading **inverts** — relu gives 0.0% raw and 57.1% scaled where tanh gives 64.4% and 0.0%. Pinned by a test. "Stuck" is `movement == 0`, exact under round-to-nearest and cheaper than a `spacing()` call; comparing against the spacing can over-count a downward step across a binade boundary, measured at 0.00% and kept as the exact form anyway.

**fp32 to TEN million updates: exact on a stationary target, and the non-stationary answer is
not available.** A device adapting at 50 Hz reaches ten million in under three days, so the
length is a deployment figure. 24 runs, 2 targets × 2 architectures × 3 seeds:

- **stationary: `f4/f8` is 1.0000 at every one of five checkpoints**, at (16,) and at (32,32),
  while f4's own step underflow climbs to **11.3%** and **2.6%**. So underflow at those rates
  costs *nothing measurable* — the claim that fp32 is the floor holds ten times further out
  than it did.
- **non-stationary: no conclusion, and that is the finding.** Both precisions degrade late —
  f8 itself goes 0.0386 → 0.0629 at (16,) and 0.0228 → 0.0415 at (32,32) between 6M and 10M —
  and the ratio swings **both ways**: 1.2211 at (16,), **0.6019** at (32,32), where f4 is 40%
  *better*. Two trajectories separating on a moving target, three seeds, and a swing far
  larger than any precision effect would be. **Reporting either number as a precision result
  would be picking a side of a coin.**

**What does hold, and is the reason the concern stays open: fp32's underflow keeps climbing
and does not saturate.** On the drifting deep case it runs 0.1% → 0.6% → 6.3% → 20.6% →
**38.7%** across the five checkpoints. Nothing has cost anything yet. Nobody has run a
hundred million.

**fp32 is indistinguishable from fp64, and that is now measured to a million updates.** Four
targets × two architectures × three seeds × 1,000,000 updates, 48 runs in 296 s: `f4/f8` is
1.0000 at 100k, 250k and 500k and within ±4% at 1M — residue that falls on *both* sides of one,
so it is two trajectories drifting apart rather than degradation. What does grow is fp32's own
step underflow, 0.0% → up to 22% over a million updates on a non-stationary target: it costs
nothing yet, it is the same mechanism that kills fp16, and it is recorded rather than dismissed.

**fp16 is refused, and the claim has been narrowed twice.** The first version said it *worked*
(one target, one architecture, one rate — the most favourable of twenty-four). The second said
it cost **4.0×** on a small-output target; **withdrawn**, because fp64 scores 4.2× worse than a
constant on that fixture — nothing learns it, so the comparison was between two failures, and
normalised, fp16 is *better* there (0.817×). Fourth vacuous comparison of the session.

What survives is measured on a **well-scaled** target and is a *rate*, not a ratio: at (16,),
lr 0.002, fp32 halves its error between 50k and 100k updates (0.0907 → 0.0478) while fp16 moves
4% (0.0853 → 0.0821). **fp16 has not become less accurate — it has stopped**, with 67% of its
steps below the spacing at the weight's own magnitude against fp32's 0.1%. Separately, on a
badly-scaled target it goes non-finite where fp32 survives on the same data and the same
starting weights — kept, and labelled, because fp64 fails there too and a badly-scaled problem
is not an indictment of a precision on its own. Both halves are tested, and the tests assert the
*controls* (fp32 must keep improving; fp32 must not underflow) so a fixture that stops
discriminating fails rather than passes.

That measurement also exposed a real defect — the shared dtype refusal gave *every* adapter RLS's
covariance reason, including two that have no covariance. It picks the right reason now, and both
branches are tested.

**And a mutation check on that correction found `min_dtype` to be a dead check** as the repository
stands: `SUPPORTED_DTYPES` is `('f4','f8')` and all three adapters declare `f4`, so the itemsize
comparison can never be true — the allowlist refuses `f2` first and nothing else is smaller. Same
shape as the guard's `at_most(PTrace, 1e4)` finding. It is not dead *code*, since an adapter needing
double precision would declare `f8`, so it is exercised with a synthetic one that does, and a
separate assertion fails if it ever becomes reachable for a shipped adapter.

**A diverging step is deliberately not clamped**, following the rule RLS sets by reporting
covariance windup rather than bounding it: a safety mechanism inside the adapter makes the guard
optional. A test drives a real divergence, asserts the unguarded adapter goes non-finite, and
asserts the guard catches it and leaves a finite state. And `reset()` **stops** a network rather
than restarting it — zero weights make every hidden unit identical with identical gradients, a
symmetry it cannot learn out of; pinned by a test because it is surprising.

**Three mutants that survived and were findings.** `state` returning a copy survived both an
equality check *and* an allocation count, because the copy is discarded immediately so live
blocks return to baseline — it needs `shares_memory` plus a count over *retained* results. The
same defect has been in this repository before, on the path the guard takes at every checkpoint.
`gain_norm` reporting the raw gradient instead of the step survived because nothing compared its
*value*; it is now exact against the distance the state moved. The third was my own dud.

**The network is across** (`cpp/include/argmind/host/runtime_shape_mlp.hpp`), into the *hosted*
tier and not the core, by arithmetic rather than preference: the guard's ring is
`Depth × state_size`, which is 47.5 KB for a 3-32-32-1 network at depth 4 against 1.6 KB for RLS
at six features, and it needs `tanh` from libm, which the freestanding core does not link.

**Initialisation deliberately does not match the reference, and that is the design.** The
reference draws Glorot weights from numpy's PCG64 with a ziggurat normal; matching that
bit-for-bit means reimplementing both, and a generator that *nearly* matches is worse than one
that plainly does not. So the port uses its own stated generator and every parity case hands
over the reference's own starting weights — which is what a device restoring a trained model
does anyway, so the harness exercises the real path rather than a special one.

**`mlpcheck` reuses `xcheck`'s result format byte for byte** — same `ResultHeader`, same 56-byte
`Row`, `static_assert`s copied — so one Python reader serves both. The *case* format is new and
could not be reused: `xcheck`'s header is a fixed 184 bytes carrying `lambda` and `p0`, and a
network's configuration is a variable-length list of widths plus a weight block.

**Measured across five architectures no instantiation list contains** (`()`, `(5,)`, `(8,6)`,
`(4,4,4)`, `(16,)` at 3×1 to 7×3), both activations, both precisions: applied/reason exact,
every reported scalar and the state block inside 400 ulp. **The `hidden=()` case carries the
exact law on both sides** — a posteriori error exactly `(1 − 2·lr·(x·x + 1))` times a priori,
to 1e-9 — which is the strongest statement available about a port.

**`tanh` was the divergence to worry about and it is not one.** Reduction order is declared
divergence 3 and every adapter carries it, but numpy's `tanh` and libm's disagreeing would be a
**fourth**, and worse in kind: a transcendental enters at every hidden unit rather than at a
summation. `relu` is exact in both, so comparing the two activations at the same architecture
separates them — and a test asserts that separation rather than a bare bound.

**Eleven mutants on the port, one survivor and it was a finding**: `gain_norm` without the
learning rate. The rows were *collected* and only `applied`/`reason` were compared, so the field
was present, decoded, and never looked at — the quietest way for a field to go untested, and the
**second time in two days** for this exact field. Every reported scalar is compared now. The ten
caught include column-major weights, a dropped bias, updating the weights before computing the
next delta, a wrong `tanh` derivative, a transposed delta propagation, and the activation applied
to the output layer.

**And the composition is pinned in C++, not only in Python**: a test compiles
`RuntimeShapedGuard<RuntimeShapedMlp<double>>`, checks the guard accepts a residual bound and
**refuses a `p_trace` bound**, drives a latched-sensor fault and asserts it freezes at exactly
the fault step leaving a finite state. If the guard needed one change for a network, the
contract would not be one.

**And the network goes through the whole composition on both sides** — streams, aligner, frame
map, guard, persist — `tests/test_mlp_pipeline.py`. That is the claim the scope decision rests
on, and it was the one piece still untested: `Guard` and `Pipeline` being templates over an
`Adapter` had been demonstrated with NLMS, a different algorithm but still one matrix. A network
has layers.

**Full pipeline parity for this combination is deliberately not run, and the reason is pinned
rather than asserted.** A pipeline's outputs are bookkeeping — frames, updated, lost_frames,
staleness, the gates — and all of it is driven by shapes and clocks, not by the adapter's
arithmetic; that is already compared for two adapters, and the network's arithmetic is already
compared by `mlpcheck`. So only the *seam* is new. `test_the_seam_is_shape_driven` runs the same
event log through a network and through RLS and requires **every counter to agree exactly**,
with its own guards that the run reached overflow, a refused sample and non-zero staleness. If
that premise ever breaks the test fails, instead of the reasoning quietly going stale.

**Four fixture bugs, and three of them were the failure this project keeps finding.** A `drive`
helper that restarted its timebase, so `Stream.push` refused every sample on a second run —
monotonic time enforced at the source, working exactly as designed. A seam fixture where nothing
was ever skipped and staleness was zero throughout, because both streams arrived at the same
instant and `hold_last` always found an exact match. And a fault case asserting 2x damage where
the fixture can only produce 1.53x, because at 400 steps of warm-up the network has barely
learned and there is little to destroy — **the assertion now says the magnitude is a property of
the fixture and the full-scale figure lives elsewhere**, rather than being loosened silently.

**The fourth was the more interesting one: the test walked into the trap its own docstring
warns about.** `StepReport.prediction` is a preallocated buffer the next `step()` overwrites —
documented, and deliberate, because a fresh array per step allocates once per step. Collecting
the reports and reading them afterwards gives one value repeated, and the test then reported
"the prediction stopped moving" about a pipeline that was working. It reads the value during the
loop now, and separately asserts the aliasing *still holds*, so the documented behaviour is
pinned rather than worked around.

**Eight mutants across the two sides, all caught** after one dud of mine that only added a
comment: prediction as a fresh array, never predicting, predicting before the update, dropping
the skipped counters, freezing without rolling back, the C++ step never reporting frozen, and
`set_constant` ignored.

**There are two adapters in C++** (`cpp/include/argmind/nlms.hpp`, `NormalisedLeastMeanSquares`
in the reference). Normalised LMS is O(n) in memory and time and carries no covariance, which is
the point: at a network head's width RLS's n² is the binding constraint — 330,544 B against
2,872 at 128 features, and 512 features do not fit an STM32N6's SRAM. The Python was written
first, as RLS's was, and `xcheck` gained an adapter axis in the byte that was padding so the two
are compared rather than trusted for being short.

**Measured: bit-identical at f64 with one output**, at every μ including the 2.0 stability
boundary — RLS cannot manage that, having two declared reciprocal-multiply divergences, and NLMS
adds none. At f32 it is 0.0–1.5 ulp up to μ=1.5 and 7.5–13.1 at μ=2, where the recursion is
marginally stable so a reduction-order difference persists instead of decaying. **That claim is
scoped to m=1 because that is where it was measured** — at (6,3) the two differ by 1.3–7.6 ulp,
numpy taking a matvec path that sums differently, which is declared divergence 3 and not a new
one. The first version said "at f64" and was generalised from one shape.

**The exact law is what verifies it, not a tolerance**: after an update the a posteriori error is
exactly `(1 − μ·x·x/(ε + x·x))` times the a priori one, so μ=1 with negligible ε projects the
sample out completely. Both sides assert the law. `ε` guards the zero regressor — an all-zero
input does nothing instead of dividing by zero — and μ is refused outside (0, 2].

**One tolerance in that suite could not fail and was caught by perturbing a state block by 1%
and watching it pass**: `scale_deviation` returns a *fraction* and the assertion compared it
against `400`. It goes through `ulps()` now. Guarded NLMS has no parity harness and `xcheck`
**exits non-zero** rather than silently substituting RLS, which is tested.
