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

**A previous, unrelated project — a hardware accelerator IP study — was stopped and removed from
the working tree on 2026-08-08.** It is not deleted: it lives in history under the tag
`accelerator-final`, which is where an archive belongs. It is out of the tree because carrying
forty files and ten documents of a different project taxes every search and every exploration
from here on, and makes the repository look like two projects to anyone reading it.

    git show accelerator-final:accelerator/README.md      # browse
    git checkout accelerator-final -- accelerator/        # restore, if ever needed

**Do not restore it to continue it.** Everything worth carrying forward was extracted into this
file before removal.

Two things from it are still worth knowing, because they were measured rather than assumed and
they bear on sensor/estimation work:

- **The EKF covariance recursion needs fp32 minimum. There is no fp16/bf16 path in any filter
  form** — conventional, Joseph and square-root all lose positive-definiteness on the *first*
  step, and fp16 additionally overflows at κ(P)=1e6. Measured numerically.
- **Precision floors are a property of the algorithm, not of the platform.** Three distinct
  floors appeared across one workload set. Do not assume a single precision policy works
  everywhere.

Measured *here*, on 2026-08-08, for RLS — deliberately re-derived rather than inherited from the
above, because carrying a conclusion across algorithms is the borrowed-assumption error:

- **fp16 fails the RLS covariance recursion even at κ=1**, losing positive-definiteness by step
  115 with every regressor inside fp16's range, so it is arithmetic and not dynamic range. It is
  refused at construction.
- **fp32 survives but does not hold positive-definiteness exactly** — a negative eigenvalue at
  roundoff scale (≈-5e-7 against tr(P)~1e2) on the first update when `p0·‖x‖² >> 1`. The fix if
  it ever bites is a square-root/UD form, not more precision. Both halves have tests.

Full detail lives in the tag, not in this tree: `git show accelerator-final:accelerator/PHASE0_FINDINGS.md`.

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
*activations*, not weights, so end-to-end on-device training is not available under 4.2 MB and
no heap. What is available is a frozen int8 backbone on the NPU whose feature vector **is** the
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

## Platformisation — the layer question, settled 2026-08-10

**argmind is not in the motor loop.** The owner's words: the robot's own stack drives the
motors and argmind "controls the signal" — it produces an estimate or a correction that the
robot's control stack consumes. That settles a question three earlier answers had circled:

- The latency budget is the **estimation slot**, not the motor loop. On tomcat's own NFR12
  that is **5.0 ms** of a 7.5 ms pipeline, against a whole loop of ~45–49 ms of which 37–41 is
  the leg physically moving. Enormous for an estimator.
- So there is no latency argument for firmware **here**. Structurally firmware is still the
  floor — argmind allocates nothing and `step()` is bounded, so on bare metal its latency *is*
  the code path, while on an OS the same code runs plus the scheduler's jitter, which can only
  be equal or worse. It just does not matter against 5 ms.
- **argmind does not force the choice either way.** Freestanding C++ behind a C ABI links into
  firmware and into a Linux process unchanged; the tier is a property of the caller's system.
- What would force firmware is position in the loop, and tomcat's ADR-0005 already answers it:
  per-motor driver (FOC 15–30 kHz) is firmware without discussion, the RT tier (≥1 kHz hard) is
  firmware or PREEMPT_RT, and perception/planning is an OS. Signal-level estimation is the last
  of those.

**Runtime shapes, decided the same day** — "모든 학습, 추론은 런타임에서 돌아감". Worth recording
that compile-time *dimensions* never meant learning happened at compile time: both designs run
at runtime, and the question was only whether `n_features`/`n_outputs` are template parameters
or arguments. Measured before deciding: **runtime shapes would not save flash.** `--gc-sections`
already means a firmware pays only for the shapes it calls — 2,308 B for bare RLS against 38,770
for all fifteen instantiations, and a per-shape shim is 102–116 B. What it costs is inlining,
the declared memory, and the `SIZES.md` gate; what it buys is not rebuilding. At OS level that
trade flips, which is why the two answers are consistent.

**So: the verified core stays, and a runtime-shape layer goes above it.** Nothing measured today
is discarded, and the core remains what lets argmind move *down* into a driver if that is ever
wanted.

**First platform work landed 2026-08-10.** There was no `README.md` and no way to consume the
library at all — `find_package(argmind)` did not exist and the only integration path was copying
sources. Now: install rules, an export set, `argmind::argmind`, and a version policy that
`AM_VERSION_REQUIRE(major, minor)` turns into a compile error rather than a comment. The record
format is versioned **separately and says so**, because a device's saved state has to outlive a
library upgrade. All of it executed by `tests/test_packaging.py`, which installs to a prefix and
then builds and runs a *separate* project against it — and checks the README's own C example
compiles, after grepping the header for its entry points reported every one of them missing
(they are macro-generated, so only compiling means anything).

**The runtime-shape layer exists** (`cpp/include/argmind/host/runtime_shape.hpp`). The name
says what is runtime about it — the *shape* — after one commit as `dyn.hpp`/`DynRls`, renamed
because the first person to read it asked what `dyn` meant. That question was the answer:
an abbreviation that does not name the varying thing is a bad name in a public API, and in
Rust `dyn` means dynamic dispatch, which this is not. `RuntimeShapedRls<T>` and
`RuntimeShapedNlms<T>` take `n_features`/`n_outputs` at construction. **The path is the boundary** — these
use the heap, no core header may include one, and both halves are checked: a core header
reaching into `host/` fails, and a hosted header that started compiling freestanding would fail
too. That second check was weak at first, satisfied by *any* compile error including a syntax
error; it now builds the header hosted first and requires the freestanding failure to name a
hosted include.

**It is a port of the reference, not a new algorithm** — `argmind/adapt.py` has always been
runtime-shaped, so `RuntimeShapedRls` is a *closer* transcription of the specification than `Rls<T,N,M>`
is. One deliberate difference: it **divides** where the core multiplies by a reciprocal, since
divergences 1 and 2 buy `VDIV.F32` cycles that do not exist above an OS. **That removes two of
three divergences and not the third**, which is worth saying because the header first claimed it
"turns a tolerance into an equality" and the measurement then did not. Reduction order remains.

`cpp/tools/shapecheck.cpp` reads `xcheck`'s format unchanged — the case header already carried `n`
and `m` as `uint16_t` — so the harness reuses one encoder and reader. **Measured at shapes the
instantiation list does not contain**, which is the entire point: 5×1, 7×1, 7×2, 50×1, 6×4, both
adapters, both precisions → **0.5 to 4.0 ulp**, growing with the sum length exactly as reduction
order should, and the same order at f32 as at f64. A test asserts that *classification*, not
just the number.

**`RuntimeShapedGuard` closes the layer** (`host/runtime_shape_guard.hpp`). `Guard` holds
`ckpt_[Depth][state_size]`, so depth, bound capacity and the external slots become constructor
arguments and the checkpoint ring goes on the heap; the *decisions* are matched exactly. Guarded
cases now run through `shapecheck` and are compared at 7×1, 7×2, 5×1 and 50×1: same freeze step,
same rollback steps, **same set of tripped checks**, state within 400 ulp.

**Parity found a real bug in the port, and it is the one this project has now made three times.**
The runtime guard projects its report on to a one-output shell before evaluating checks, and the
first version copied the scalars but not `frozen`. `Divergence` holds its slow average while
frozen — `if (!r.frozen) slow += ...` — so the baseline tracked the very fault it was holding
out against and **the guard thawed itself in the middle of one**: the reference thawed at step
218 and the port at 166, with the fault running to 200. An uninitialised field reintroduced a
design error the reference had already been fixed for.

**And the first guard fixture caught none of it.** One bound, depth 1, no divergence, no rejected
samples — four deliberate bugs survived it, each for a different reason: rolling back to the
newest checkpoint is identical at depth 1, stopping at the first failing check is identical with
one check, resetting divergence has nothing to reset when it is off, and no sample was ever
rejected. Widened to depth 4, two bounds, divergence on, and a NaN in the data, plus a separate
fault-that-ends case for the thaw. Six mutants, all caught — verified individually, because the
batch count was itself unreliable and said two survived when they did not.

One fixture is deliberately artificial and says so: nothing else crosses its bound where
`p_trace` does (`p_min_diag` is not monotone and dips at step 40; `gain_norm` peaked at 0.50 long
before), both measured, so "every check runs" is reached with two bounds on the same quantity.

**`RuntimeShapedPipeline` finishes the hosted tier** (`host/runtime_shape_pipeline.hpp`).
**Half of it needed no porting**: `Aligner` already takes its policy, reference and staleness
limit as values, and `StreamView<T>`/`Ring<T>` already carry a runtime `dim` — all made that way
for the C ABI, so the hosted tier inherits them. Only `FrameMap` and `Pipeline` fixed anything
at compile time. `shapepipecheck` reads `pipecheck`'s job and result formats **byte for byte**,
the structs copied with their size `static_assert`s intact, so the Python harness reuses its own
`Case`/`both`/`compare`.

**Two port bugs, both found by widening a fixture rather than by reading.** The first
comparison ran on cases with **no events at all** — `steps=0`, four agreements out of four, the
third vacuous pass this session. With events driven in, three mutants still survived because
nothing ever *declined* an instant; an aiding stream that starts late reaches that path, and it
immediately exposed the second bug: **`n_staleness` was derived from `ok` rather than from
whether a frame was built**, disagreeing with the reference on the very first step.

One mutant is **equivalent and recorded rather than contrived around**: adding `&& frame_ready_`
back to the prediction condition changes nothing here, because this port took the fixed shape
where `build_frame` never clears the flag. Reintroducing the core's old bug needs both halves.

**Swapping boards, measured 2026-08-10.** Every parity claim had compared an implementation
against the *reference*, which answers "is the port right" and not the question an integrator
asks before a hardware change: **move this firmware from a Cortex-M4F to a Cortex-M55 and do the
numbers change?** Measured over four shapes and both precisions: **0.0 to 3.5 ulp between the
two boards**, against 0.5 to 3.0 for either board against the reference — so a swap costs no more
than the port already does, and `tests/test_board_migration.py` asserts *that comparison* rather
than a bare bound, since a swap ten times worse than the port would still pass a loose one.

**The f64 path differs more than f32, which is the opposite of the naive guess** — at 6×1 and 6×3
the two are **bit-identical at f32** and differ by 3.0 and 3.3 ulp at f64. The M55 has hardware
double precision and the M4F does not, so f64 is soft-float on one and not the other.

**And a harness bug had to be found first: `malloc` returns NULL on QEMU's `mps3-an547`.**
`rdimon`'s crt0 takes its heap from `SYS_HEAPINFO`, which works on `mps2-an386` and does not
there. Nothing says so — `fopen` returns NULL so a parity tool reports "cannot open <file>" and
reads as a missing file, and `printf("%f")` dies inside `dtoa` on a failed REENT malloc. Both
were chased before the cause was found. `startup.c` supplies `_sbrk` over a static array now, so
the heap comes from `.bss` and depends on no semihosting answer; at 256 KB it silently overflowed
the AN547's 512 KB DTCM and the image produced *no output at all*, which is quieter than the
failure it fixed, so it is 64 KB and the reason is recorded. The self-check suite never noticed
because it allocates nothing.

**Exporting the learned model across boards: done and verified in both directions.** The
record format already claimed "the writer and the reader may be different machines" and had
been checked host-against-Cortex-M4F; **board-against-board is the case an integrator actually
performs and had never been run.** `cpp/tools/braincheck.cpp` trains, packs and writes on one
machine and reads, loads and predicts on another. M4F → M55 and M55 → M4F both reproduce the
exporting board's prediction **to the last printed digit**, as does a record the *Python*
reference wrote. `persist::pack` already took `count` as a runtime argument, so runtime shapes
needed no special case — an earlier note here saying otherwise was wrong.

**And the new board refuses what it must.** A flipped bit and a truncated write are caught by
the CRC and the length. The third case is the one that matters and the CRC cannot see it: a
record that is **perfectly valid but written under a different `FrameMap` order**, which would
put the right numbers on the wrong features. The configuration fingerprint refuses it. That case
is built with the Python reference so the CRC is genuinely correct — forging it by editing bytes
only re-tests the CRC, which this project has been caught doing before. Both halves
mutation-checked: removing the fingerprint check and skipping `load_state` each fail a test.

**The network can keep its learning while gaining INPUTS, and cannot usefully gain hidden
units — both measured.** `MultilayerPerceptron.widened(n_features)` fills the new rows of `W0`
with **zeros**, which preserves the function (the new inputs contribute nothing whatever their
values) *and* still learns, because the gradient into a new row is `x_new · delta` and `delta`
comes from a trained downstream. Measured against starting over, three inputs becoming six on a
truth needing all six: **0.915 grafted against 1.134 fresh at 100 updates, 0.651 against 1.004
at 2,000, 0.383 against 0.559 at 30,000** — the advantage does not close, and on the *original*
three inputs at 100 updates it is 0.481 against 0.929, which is the direct statement that the
learning survived.

**Random rows were measured and are not offered**: a tie on accuracy (2 of 6 seeds, medians
0.3779 against 0.3960 inside seed spreads of 1.51× and 1.66×) and they break continuity, moving
the output by **2.0 to 4.6** on a target whose spread is 1.4. Zero moves it by 0.0–2.3 ulp.

**`deepened()` exists only to refuse, and the refusal is the measurement.** Extra hidden units
buy nothing on any target that could be constructed — single layer, 60,000 updates, four seeds:
`smooth` 0.2368/0.2469/0.2391/0.2305 and `wiggly` 1.0918/1.0918/1.1386/1.1404 at h=8/16/32/64,
so **eight units already reach the ceiling and sixty-four are sometimes worse**. The binding
constraint on these problems is SGD's convergence, not capacity. The first widening fixture
could not see this: all three strategies landed within 0.2% of each other *including the one
whose new units are provably dead*, which is a fixture that cannot tell them apart rather than
three answers agreeing.

The refusal carries the recipe anyway, because it is the trap for anyone hand-editing a state
block: a new hidden unit with **both** new blocks zero is **permanently dead** — measured, its
weights move by exactly 0.0, since its outgoing weights gate the gradient into its incoming ones
and `tanh(0)=0` gates the reverse; **both random** jumps the output by 0.095; **incoming random,
outgoing zero** is the one that works.

**The port carries it too** (`RuntimeShapedMlp::widened`), in the hosted tier's own idiom: a narrower request returns an adapter whose `valid()` is false rather than throwing, matching `RuntimeShapedRls`. There is no `deepened()` on that side — the reference's exists only to refuse and carries the measurement, and a second copy of a refusal buys nothing. **Parity is exact rather than a tolerance**, for the reason `persist`'s is: widening moves bytes, it does not compute, so a tolerance would let a transposed copy through whenever the numbers happened to be close. Four architectures, `np.array_equal` on the whole block. The narrowing mutant survived until a case asked for one.

**And the metric needed the correction this repository has now made four times.** The widened
output differs from the old one by up to **367 ulp against each probe's own value** — which is
the prediction passing near zero, not the networks differing. Against the *run's* scale it is
**0.0 to 2.3 ulp** over five architectures, exactly 0 with no hidden layer. The test asserts
both, so the reason the per-value metric is wrong is pinned rather than described. Seven mutants,
all caught.

**Keeping the learning while adding capacity: `widened()`, on both sides.** A bigger board does
not make a bigger model — `n_features` is a design choice — but a wider model can now inherit a
narrower one's learning instead of starting over. `theta` keeps its coefficients and the new ones
start at zero; `P` keeps its earned block in the corner, the new diagonal gets `p0` and the cross
terms are zero. That is principled rather than a copy: `P` is the inverse information matrix, so
the corner is the confidence earned, the diagonal is "nothing known yet", and zero cross terms
say no evidence relates the two sets.

**Appending only**, and that is a constraint. Feature order is parameter order, so appending is
the one widening that permutes nothing; rearranging needs a mapping the caller supplies, and
guessing is the failure `persist`'s fingerprint exists to prevent. Narrowing and changing the
output count are refused with the reason.

**There are two reasons to widen and they want OPPOSITE answers**, which the owner named:
expanding inputs/outputs, or expanding the *model* at the same inputs by enriching the basis. So
`keep_confidence` is **keyword-only with no default** — a default would be a trap.

| | fresh | keep_confidence=True | False |
|---|---|---|---|
| new independent inputs, error on the old ones @10 | 0.410 | **0.0010** | 0.327 |
| richer basis, same three sensors, error @2000 | 0.00001 | **0.242** | 0.00001 |

**New inputs want it carried**, and the *confidence* is what buys the head start — θ alone gives
almost nothing (0.327 against a fresh 0.410). **A richer basis wants it dropped**, because a
misspecified fit is **biased, not merely incomplete**: fitting a truth with an `0.8·s₀²` term
using a linear basis lands the constant at **1.136** instead of 0.3, having absorbed
`E[0.8·s₀²]`, with `tr(P)` down to 0.0032 — carrying that certainty makes the adapter sure of
wrong values and it resists correction for thousands of updates.

**Two predictions of mine were measured to be wrong here.** That grafting would matter *more*
when new features are rarely excited — it does not, the old features are excited either way.
And that carrying `P` is simply a benefit — in the basis case it is the whole problem. Both are
why `widened()` has no default. Seven mutants caught across the two implementations, and the two
widenings are compared against each other.

**Still undecided:** whether inference is *hosted* (an interface to ONNX Runtime/TFLite) or
*embedded*, and whether the tiers ship as one library or two. Nothing on the camera/audio/speech
path is built.

The original design driver, still the one the core tier's API is shaped against: an inertial +
aiding-sensor estimator with online bias calibration. Sensor-agnostic in the API, specific in
what it must handle.

**Built:** `argmind/stream.py` (fixed-capacity ring buffers, counted drops, monotonic time
enforced at the source), `argmind/align.py` (four explicit alignment policies, signed staleness,
causal/non-causal declared), `argmind/adapt.py` (the `Adapter` contract and
`RecursiveLeastSquares`), `argmind/guard.py` (`Check`/`Bound`/`Divergence`, freeze, bounded
rollback, explicit recovery), `argmind/pipeline.py` (`FrameMap`, input gate, bounded `step()`),
`argmind/persist.py` (two-slot atomic store, CRC, configuration fingerprint, declared cadence),
`argmind/bench.py` (fixed workload, correctness digest, leak detector, regression gate), and
`cpp/` (the C++ core: `mat.hpp`, `rls.hpp`, `guard.hpp`, `persist.hpp`, `stream.hpp`,
`align.hpp`, `pipeline.hpp`, the C ABI, four parity tools
`xcheck`/`pcheck`/`scheck`/`pipecheck`).
**The runtime library is fully across to C++** — stream, align, adapt, guard, persist and
pipeline. `argmind/bench.py` has no C++ counterpart and is not getting one: it is a host
measurement harness, and DESIGN.md already records that Python timings are not a baseline the
C++ can be compared against.
507 Python tests and 4,652 C++ checks pass; several exist to prove failures are visible
rather than silent. Every suite is mutation-checked, and surviving mutants are recorded where
they live rather than tallied here — including the ones that turn out to be **equivalent on a
64-bit host**, a distinction worth keeping rather than rounding away. Cumulative counts are
avoided on purpose: the two that used to sit in this paragraph were both wrong until an audit
measured them.

**The adapter contract, since everything after it fits into it** (full reasoning in DESIGN.md):
state is one flat contiguous block so persistence is a copy and the C ABI needs no serialisation;
`update()` always returns an `UpdateReport` because the `Guard` is written against those fields;
shape errors raise but bad *values* are rejected-and-counted, since a sensor NaN is a runtime
event and not a programming error; and covariance windup is reported rather than clamped, because
bounding it inside the adapter would make the guard optional.

RLS was verified against the closed-form exponentially weighted least squares solution, not
against itself. Two apparent failures turned out to be real properties worth pinning: the residual
error on noiseless data is the `P0⁻¹` prior term and scales as exactly `1/p0` across four decades,
and the unexcited covariance direction grows as exactly `p0·λ⁻ᵏ` (12 digits). The tests assert
those laws rather than loose tolerances.

**The guard wraps the adapter and *is* an `Adapter`** — a deliberate change from the API sketch,
which had them side by side in the pipeline. Side by side leaves "no guard" expressible, and a
safety mechanism that can be left out gets left out. Checks run after the update (the quantities
worth checking do not exist until then), which is why rollback exists rather than prevention.

Two design errors were made here and caught by tests, both about what a stateful detector does
while frozen: resetting checks on a trip made the guard **thaw itself in the middle of a fault**,
and the first fix — holding the baseline per sample — ratcheted it down until ordinary noise
tripped it. The rule that works is per-*freeze*. The tidy-looking version was the broken one,
twice.

**The pipeline maps a `Frame` to `(x, y)` through a declared index table, not a callable** — a
callable is the obvious Python answer and does not cross a C ABI. `FrameMap` resolves convenient
forms once at construction. Feature order is parameter order, so a persisted state block restored
against a different order is wrong undetectably.

**Two gates at two different times**, and they must not merge: the pipeline checks *input* health
(staleness, sensor validity) before the update; the guard checks *model* health after it, because
those quantities do not exist until the update is computed.

A test found a real observability gap here: 200 frames against a 64-deep ring adapted 64 times and
reported nothing wrong. The first fix was wrong the other way — `Stream.dropped` counts
*overwrites*, which a healthy ring does on every push, so it false-alarms on every working system.
`StepReport.lost_frames` counts what is actually knowable: reference samples overwritten before
they could become frames. `Stream.dropped`'s docstring was overclaiming and is corrected.

**Persistence is two slots written alternately**, each with a sequence number and CRC, so a write
interrupted by power loss destroys at most the slot not in use. The reference writes files but
models the raw-flash scheme faithfully, because the C side has no `os.replace`. A configuration
fingerprint (`crc32`, never Python's salted `hash()`) refuses a block whose `FrameMap` order
differs — the one corruption nothing downstream can detect. Cadence is declared, because a sector
rated for 100k erase cycles written at 100 Hz is gone in seventeen minutes.

**The guard-state question is answered**: checkpoints and check EWMAs describe a run and are not
persisted; the frozen flag and its cause are, because they are a conclusion. What a reboot does
with a freeze is the caller's call (`on_frozen="thaw"|"keep"|"refuse"`) — thawing risks re-learning
into the same trouble, keeping risks stranding a device a power cycle would have fixed.

**The benchmark refuses to commit absolute timings as a gate.** A baseline is a statement about
one machine; as a repository gate it fails on everyone else's laptop and gets switched off. Record
locally (`python -m argmind.bench --record`), compare with `--compare`, which exits non-zero. The
suite gates only what survives moving machines: exact counters, the state digest, the memory leak
rate, and ratios measured inside one run. The tolerance is widened by *measured* run-to-run
spread, not a guessed constant.

**The benchmark covers all three adapters now, and it carries a transferable number beside
the timings** (`adapter_scaling`, `update_flops`, `--adapters`). There was no benchmark for the
network at all, and this project's own rule is that an optimisation with no way to fail is a
hope.

**`update_flops` is derived from the shapes, not timed**, which is what makes it usable off
this machine: RLS is `4n² + 4nm + 2n`, NLMS is `4nm + 2n` with no n² term at all, and a network
is `8W` over its weight count, because the forward pass runs *twice* — once for the prediction
and once for the a posteriori residual. **At equal state a network costs about twice an RLS**,
since RLS's `4n²` sits against a state of `n² + nm` while a network's `8W` sits against `W`.
Measured on the host and it holds: 34.6 µs against 60.3 at ~4,200 elements, 206.8 against 460.3
at ~66,000, against an interleaved spread of 1.07–1.18×.

**And the timings are proved to be dispatch rather than arithmetic, quantitatively.** `ns/flop`
runs from **75 down to 0.7 — a factor of 112**, and only the largest adapters approach the
arithmetic, where RLS and a network converge on the same 0.8 ns/flop and the network simply does
twice the flops. A constant would have meant arithmetic-bound. So the span *is* the evidence for
DESIGN.md's existing rule that these figures are not a baseline the C++ core can be compared
against, and nothing here is a regression gate for the same reason.

**One number was measured and is deliberately not reported.** The equal-*accuracy* comparison —
MLP (32,32) at 62.6 µs against RFF-128 through RLS at 78.9 — reads 1.26× and sits inside
run-to-run spreads of 1.62× and 1.33×. It did not survive its own error bars. The memory side of
the same comparison is 9.5 KB against 131.0 KB and is not close. **A 2.3× outlier was caught the
same way**: a single non-interleaved run read MLP (256,256) at 169 µs where eight interleaved
repeats give 460. Consecutive repeats measure a drifting mean, which this repository has already
paid to learn once.

**Two measured results from building it**, both of which contradicted a plausible first guess:

- **RLS update cost is flat from n=4 to n=32** (~22 µs) and only climbs at n=64. RLS is O(n²), so
  a naive scaling test asserting ×64 would have failed. At the design driver's n=6 the time is
  *all* call overhead — the regime the C++ decision was made for, now measured here rather than
  inherited.
- **Memory does not grow with uptime**, but proving it needed two corrections. Fitting a trend
  line to successive batches reports a settling transient as growth; the right tool is a
  discriminator across batch *sizes*, fitting `cost = fixed + leak × frames`. And `tracemalloc`
  allocates while measuring, so its own bookkeeping swamped the residual — `sys.getallocatedblocks`
  does not. The measurement found one real thing: `Adapter.state` built a fresh view per call, on
  a path the guard takes every checkpoint. Fixed. A positive control (an adapter that hoards every
  report) keeps the detector honest.

**The C++ core — started 2026-08-08, RLS across.** `cpp/include/argmind/{mat,rls}.hpp`,
`argmind.h` + `cpp/src/abi.cpp`, `cpp/tools/xcheck.cpp`, `cpp/tests/test_core.cpp`, `CMakeLists.txt`.
Build and run it through `python -m pytest` — `tests/test_cpp_parity.py` invokes the compiler
directly and skips if there is none, so a clone without a toolchain still tests the Python.

**The C ABI's real tension**: compile-time dimensions are why C++ was chosen, and a C ABI wants
runtime arguments. Runtime `n` would discard the reason, so the ABI exports an **explicit
instantiation list** — adding a dimension is three lines plus a rebuild. Storage is the caller's;
`AM_*_MAX_BYTES` is an upper bound held to being one by a `static_assert`.

**The ABI now exports the guard and the record format too (2026-08-09), and that started as an
inconsistency rather than a feature.** The rule the guard was designed around is that it *is*
an adapter so "no guard" is not expressible — and the C boundary exported bare `Rls`, making
the guard unreachable from C. Likewise `am_rls_*_state()` handed out the block a device must
survive a power cut with and there was no way to frame it. `Depth` and `MaxBounds` are fixed
at 4 in the ABI because they are the memory; `checkpoint_every` stays runtime. Measured on
x86-64 (a shape, not a target figure): the guarded adapters are 1144/1384/3424/2000 bytes, the
checkpoint ring being the majority at every shape — **`f32_12_1` at 3424 is 42% of the
declared 8 KB budget for one adapter**, which is arithmetic on the budget and worth seeing
before a target build rather than after.

The `MAX_BYTES` `static_assert` earned its keep on the first build: the bounds were written
before measuring and `AM_GUARD_F32_12_1_MAX_BYTES` was 3328 against a real 3488 (both
figures historical: 3456 against 3424 since the ring shrank). And the C99
gate turned out weaker than it read — it compiled the header and stopped, which says nothing
about symbols being callable. A C99 program now compiles with gcc, links against the C++ core
and runs end to end: configure a guard, drive a fault, pack the state, restore it.

Seventeen mutants on the shim, sixteen caught. The survivor is **equivalent, not untested**:
`am_guard_*_state(nullptr)` checks for null, and dropping the check is undetectable because
`state()` sits at offset 0 of the guard — measured. It stays because a null dereference is
undefined whatever the address arithmetic produces.

**NLMS is in the C ABI now, and the reason is a measurement rather than symmetry.** The
question that produced it was whether a **HAL** is needed for camera/NPU hardware. It is not,
and that was verified rather than argued: a 127-wide backbone embedding goes into `am_ring_f32`
— whose `dim` is a runtime value, decided for the instantiation-explosion problem years of
commits ago — through the frame map into a guarded pipeline, and predicts. **No new API.** What
differs per vendor (model format, load, invoke, dequantisation) all happens *before* the push,
and it belongs to the vendor SDK: taking a dependency on ST's, Arm's and Hailo's runtimes would
end the freestanding, zero-dependency property that lets this library link into anything.

**But the check exposed a real gap that a HAL would not have fixed.** Every
`AM_DECLARE_*` was RLS-backed, and at 128 features a guarded RLS pipeline is **332,328 bytes**
— forty times the whole declared 8 KB floor — where NLMS is **4,656 and fits it**. So a C
caller wiring a backbone got the adapter this project's own measurement says not to use. That
reverses the "wait for a caller" note above: this is the caller.

Exported at 6×1 and 128×1: the bare adapter, a guard over it, and a guarded pipeline.
**Deliberately no `am_guard_nlms_*_health`** — an adapter with no covariance could only answer
zeros, a bound on a field an adapter cannot report was measured to break in both directions at
once, and C has no `health_fields` to query, so the *missing symbol* is how a C caller is told.
`add_bound(AM_FIELD_P_TRACE)` returns `AM_ERR_CONFIG`, which is the same refusal reaching the
same boundary by the only means it has.

**The guard and pipeline macros were derived from the RLS ones by substitution, not
transcribed** — two hand-copied 200-line macros drift — and the guarded pipeline's shared tail
is factored into `_REST` so only `init`, whose signature genuinely depends on the adapter,
exists twice.

**The size gate fired again and the +16 bytes on every row are not NLMS code.** `--gc-sections`
still drops the shims a configuration never calls. A second adapter gave the compiler a reason
to *outline* helpers it had been inlining per instantiation: `fill_report<float,1>` and
`fill_guard_report<float,1>` became real functions and `am_guard_f32_6_1_update` shrank from
0x82 to 0x32. Diagnosed from the symbol table, because "a size moved" and "a size moved for a
reason you can state" are different claims.

Three mutants. One survived and was a real gap — dropping `params_valid` from the *bare*
`am_nlms_*_init`, which the guarded tests never reach; closed. One is equivalent and recorded:
weakening a `static_assert` cannot be detected until the invariant it guards actually breaks.

**The composed handle followed the same day, and everything the library does is now reachable
from C.** The instantiation explosion is avoided by moving the two stream axes out of the type
entirely: **the caller owns the sample storage**, so `am_ring_f32`'s `dim` and `capacity` are
runtime values and only the adapter stays in the type. Same trade `StreamView` already makes
inside the core.

**One push, not two — and it took two attempts and a measurement.** A C ring and a
`Stream<T,Dim,Capacity>` would hold two copies of the monotonicity and finiteness refusals, so
both now call a shared `ring_push`. Taking a `Ring<T>*` cost **+32 bytes** of code at `-Os`
(160→192 for `Stream<float,6,64>::push`); passing members separately still cost +16, because
the rewritten wrap used compare-and-branch where `% Capacity` folds to an AND on a
power-of-two capacity. Restoring the modulo lands back on exactly 160. The comment saying the
extraction was free was written before measuring and was wrong twice. That modulo is a
division where `slot()` avoids one, and the two do not conflict: this is once per *push*, that
was twice per element of every *scan*.

Pipelines, x86-64 shapes: bare 616 / 952 bytes, guarded 1632 / 1896 / 3952 / 2640 — a guarded
pipeline is its guard plus 424 to 576, not "about 400" as this line first said, and 16 to 32
bytes larger than first published because the freshness gate added three members and nothing
pinned the figure more tightly than its `MAX_BYTES` bound. An audit caught both. `map_mutable()` **clears the configuration** on purpose:
the map is what `configure()` bounds-checked, so mutating it and staying configured would let
a caller widen a component past a sensor's width after the check.

**Sixteen mutants, two survived long enough to be findings.** `am_ring_*_init` setting `last_t`
to 0 rather than `-inf` survived — a **real defect the C path had and the C++ path did not**,
silently refusing the first sample of any device whose timebase is milliseconds-since-boot.
And zeroing `lost_frames` survived because nothing on that side ever overflowed a reference
ring: the field was present, compared, and compared *against zero* — the quietest way for a
field to go untested. Both closed.

**A fixture that agreed while testing nothing**, caught by three assertions at the end: the
pipeline ABI comparison mapped the reference stream's own value — the *target* — into the
regressor, so the model read the answer off its input, the residual sat at zero, and the
injected fault did nothing while everything else matched perfectly.

**Parity: ~1.5 ulp on the state block, and the same ulp count at f32 as at f64** — which is what
says the difference is algorithmic rounding, not a precision-dependent bug. Worst quantity is the a
posteriori residual at 597 ulp, and that is cancellation, not error: it is `y - x'θ` after fitting
to that sample. Three divergences are deliberate (two reciprocal multiplies from the port notes,
plus BLAS reduction order).

**The parity metric took two corrections**, both non-obvious and both recorded in
`tests/test_cpp_parity.py`: element-wise relative error is meaningless on a block with
cancellation-produced near-zero entries (reads 1e-13/1e-4), and the a posteriori residual does not
belong in a *relative* comparison at all — nor does its ratio to the a priori one, which inherits
the same cancellation (1.06e5 ulp). It is bounded absolutely against the target scale instead.

**Verification limits, as they now stand.** Compilation, linking and *execution* all happen on
the declared floor: every core header builds for Cortex-M4F, the ABI links into four
firmware-shaped images so flash and RAM come from a link map, and the whole self-check suite
runs on an emulated Cortex-M4. What is left is **timing on real silicon**. An emulator models
the instruction set, not the pipeline, so **on-target timing claims stay absent** — the same
refusal as before, for a smaller reason.

**The Guard is across too** (`cpp/include/argmind/guard.hpp`). `Guard<Adapter, Depth,
MaxBounds>` — the checkpoint ring and check capacity live in the type, so the memory is
declared: `Guard<Rls<float,6,1>, 4, 4>` is 1144 bytes of which 672 is the ring.

**One structural difference, forced not chosen**: the Python's list of polymorphic `Check`
objects needs virtuals and a heap, and the core has neither. Checks become concrete members
with declared capacity and their identities become an enum, which is what a `tripped` bitmask
is built from. The cost is that a user cannot supply their own check type — the same trade the
C ABI makes for compile-time dimensions.

A guard *decides* rather than computes, so parity here is about decisions and a decision that
differs is not a rounding difference. Matching exactly at both precisions, across depths
1/2/4/10 and cadences 1 and 5: freeze step, thaw step, the whole frozen vector, which steps
rolled back, the set of failing checks, and which checkpoint a trip restores.

**The checkpoint ring holds nothing but state blocks now, and that was recorded as redundant
before it was measured to be.** Each slot used to carry the two covariance scalars true at
that state. They are recoverable from the block itself — `health()` is a pure function of the
state, and for RLS both scalars come out of `P`, which the block contains — so a rollback
recomputes them, **bit-identically, verified over sixty checkpoints**. `checkpoint_bytes` is
now exactly `Depth × state_size`. Measured on the target: the guarded configuration's static
RAM falls **1296 → 1168 B** and the full composed pipeline's **3264 → 3136**, on both M4F and
M55; on x86-64 the guards go 1208/1448/3488/2064 → 1144/1384/3424/2000, and NLMS 472 → 432,
which is not a uniform 64 because alignment absorbs some of it. The exported `AM_GUARD_*` and
`AM_PIPE_G_*` upper bounds are tightened to match — leaving them would have meant a caller who
sizes storage with the macro sees none of the saving, which was the point.

**And `SIZES.md` told readers to re-run the size test with `--regen-sizes`, which did not
exist.** That file is the one this repository created *because* a hand-copied size table went
stale three times, so a broken instruction in it is the same failure returning by another
door. The option exists now, rewrites only the numbers so the prose cannot drift from the
reasoning it carries, and **still compares afterwards** — a regenerator that is not checked
against the same parser the gate uses can write a file the gate cannot read.

**A mutation check on the port found a bug in the reference.** `Guard.freeze()` read its health
scalars from the newest checkpoint. Exact after a *trip* — the state has just been restored to
that checkpoint, and `_probe`'s docstring justified it on that ground — but after a *manual*
freeze nothing was restored, so the value could be `checkpoint_every` updates stale: 300.0
reported against a live 104.1, three updates into a cadence of five. A `Bound` on `p_trace`
during a manual freeze read a state the adapter was no longer in. Both implementations now take
the live values. That leaves the ring's stored health scalars redundant; removing them would
drop the declared cost to `Depth * state_size`, and it is **recorded in DESIGN.md rather than
half-done.**

**Two flattering results had to be withdrawn, both from one broken fixture.** With `p0=100` and
six features `tr(P)` starts at 600, so a `p_trace` bound below that trips on the *first* update
— a sweep over four bounds compared step 0 against step 0 four times. The fixture is now a
windup scenario where `tr(P)` grows as exactly `p0·λ^-k`, and the tests pin the *predicted* step
(`ln(hi)/ln(1/λ)`: 45/90/135 predicted, 44/89/134 measured). With that broken fixture the
restored states also came out bit-identical, which looked like proof that a rollback copies
rather than recomputes — it proved nothing, since both rolled back to the *construction* state.
At a real trip step they differ by ~41 ulp. What is verifiable is *which* checkpoint each side
chose, tested by comparing each side against **its own** unguarded trajectory.

**How to write these tests, learned the expensive way.** Eight of eighteen mutants survived the
first run because there were no Guard tests in the C++ unit suite at all — a parity harness can
only test what the tool exercises. The four that belonged in the parity suite were then designed
*against their mutants*, and the first attempt at all four failed to discriminate: a second
spike one sample later lands while still frozen; a second fault 240 samples after a thaw arrives
once the EWMAs have re-converged; bounds a rejected report satisfies anyway make a leaked
rejection identical (the observable that works is the *checkpoint* it stores); and one clean
sample in three never lets the fast average fall below the threshold. **A test written without
checking that its bug would fail it is a guess.**

**Persistence is across** (`cpp/include/argmind/persist.hpp`). Format only — pack, validate,
unpack, arbitrate two slots. It does **not** write flash: erase granularity, write width and
the driver API are platform-specific, and guessing at them would be wrong everywhere.

**Parity here is byte identity, not a tolerance**, because serialisation has no rounding.
The sweeps are checked in so the figures are reproducible: 840 pack configurations, every
single-bit flip in a 78-byte record (624), every truncation length (78), and 400 seeded
fuzz blobs — zero differences. They also pin *check ordering* (version, dtype and length all
before the CRC), which needed saying because a mutant moving the C++ version check after the
CRC passed the whole suite: every test touching versioning repaired the CRC first.

**The first real run of the sub-agent team happened here, and it earned its place.** Three
findings neither the tests nor I had:

- **The reference destroyed the record it had just loaded.** `Store.__init__` starts at slot 0
  and `load()` updated only the sequence, so after an *odd* number of writes in the previous
  session the first write after a reboot landed on the newest record. Measured: three saves
  leave `{0: 3, 1: 2}`, and the next save destroys 3. Exactly the failure two slots prevent.
- **A 32-bit overflow that cannot be reproduced on the host.** `count` is untrusted and
  `size_t` is 32 bits on the target, so `count * sizeof(T)` can wrap. The CRC still catches
  it, which made the safety depend on an unstated invariant. **The rule that host figures are
  not target figures, applied to behaviour rather than timing.** Guarding `unpack` and not
  `pack` was an omission a later audit caught — the same multiplication, unguarded, on the
  path that writes.
- **`select_newest` was compiled but never called by any binary the suite built**, and a bounds
  check added to the parity *tool* had made the library's own clamp unreachable. Both closed.

**And the auditor falsified my own summary of all this.** I had written "exactly one mutant
survives anywhere"; it built a second one in minutes — moving the C++ version check after the
CRC — which passes the whole suite and is not host-equivalent. It also found that the sweep
figures I quoted came from a run that left no trace, that `persist.hpp` had **zero coverage**
in the C++ unit suite and was in neither the freestanding nor the C99 gate, that `next_slot`
had no caller anywhere, and four tests whose assertions could not fail. All fixed; the sweeps
are checked in, and `persist.hpp` is now in the freestanding gate and the C++ unit suite —
not the C99 gate, which covers the ABI header alone and cannot cover a C++ header that has no
ABI surface.

**Three operational lessons about the team**, all now in `.claude/agents/`: an agent ran
`git checkout -- <file>` to undo a mutant and discarded uncommitted work (it recovered, but
the instruction was missing and is now explicit); one report's cumulative mutant count was
wrong while every finding in it was correct; and **the most valuable single run was the one
aimed at my own prose**, which is the argument for running it before every commit that states
a number rather than after.

**`Stream` and `Aligner` are across** (`cpp/include/argmind/{stream,align}.hpp`). The
structural decision, and it was challenged and upheld in review: an aligner must see streams
of *different* widths at once, and this core has no virtuals for a base class and cannot name
a variadic `Aligner<Streams...>` from a C ABI. So `Stream<T, Dim, Capacity>` keeps its shape
in the type and hands out a **`StreamView` with a runtime `dim`** to anything that treats
several streams uniformly. `dim` only ever drives an O(dim) copy at the slow stream's rate;
the compile-time shape stays where it buys inlining, in `rls.hpp`.

**The port found a third reference bug, from a single disagreeing push.** `Stream.push`'s
ordering check cannot catch a NaN — `NaN <= last` is False, so the sample was accepted,
`_last_t` became NaN, and *every later comparison was False too*: monotonicity was gone for
the life of the stream, leaving rings like `[1.0, 1.5, nan, 0.1]`. That is not cosmetic —
every alignment policy scans the ring assuming order, so one NaN can make `hold_last`,
`nearest` and `interpolate` all return the wrong sample. Non-finite timestamps now have their
own check, and a test demonstrates the hazard by corrupting the ring directly.

**Parity is decisions, not arithmetic.** The checked-in sweeps compare bit-exactly — staleness
by bit pattern, values by raw bytes — and the negative control says the three *distinguishable*
policies differ at 292 of 300 instants. `drop_if_stale` is not one of the three: it shares a
branch with `hold_last` and never reads the staleness limit, so an earlier claim of "four
policies, 872 of 942" was wrong twice over, and the decision that actually distinguishes it was
compared nowhere until an audit found that. Interpolation's float64-then-narrow blend is
bit-identical, its precondition is recorded (an FMA contraction breaks f64 exactness, which the
M4F cannot do but an M7 can), and so is the cost the target *does* pay: that widening is soft
float on a single-precision FPU — the largest unstated cost in the header until the audit named
it.

**The `Pipeline` is across, and it was the last piece** (`cpp/include/argmind/pipeline.hpp`).
What has to agree at this level is *bookkeeping*, not arithmetic — a pipeline is driven by two
independent clocks, samples arriving and `step()` being called, and every figure it reports is a
difference between counters taken at different moments. So the parity job file is an **event
log** with `push` and `step` interleaved, which is the only way to reach a ring that overflowed
between two steps or a guard that froze half-way through one. Zero disagreements across the
checked-in suite and a 120-case randomised sweep, everything but the state block compared
exactly; the state block's worst deviation is 149.8 ulp against 400 allowed.

**Frames are never materialised** — the reference's list of `Frame` objects needs a heap, so
alignment writes straight into the preallocated `x`/`y` one instant at a time. That keeps one
`Pick` alive rather than `MaxStreams` of them (280 bytes against 1,120 for a `double` pipeline
at four streams) and costs one thing: a frame that fails half-way leaves `x` mixing two
instants, so `frame_ready()` is false and `predict_last()` refuses. **Views are passed to
`step()`, never stored**, because a `StreamView` snapshots head and count and a cached one
would align against a ring frozen at construction forever. And **every stream in one C++
pipeline shares a scalar type**, which the Python does not require — a real reduction, so the
parity cases only ever use matched dtypes.

**Sixteen of seventeen mutants caught, and the survivor is the finding.** The `lost_frames`
clamp survives because it is *dead*: each visited instant increments exactly one of
frames/no_data/dropped_stale, and every unconsumed reference sample arrived after the last
successful step, so `considered <= owed` holds by construction — the reference's `max(0, ...)`
is dead for the same reason. It stays because the C++ subtraction is unsigned and a wrap would
report ~1.8e19, and it is **recorded as unreachable rather than described as tested**. Two
tests written to kill that mutant failed to, which is the more useful half: the deltas are
sampled *inside* `step()`, so advancing the aligner between calls cannot reach the clamp.
**A test written without checking that its bug would fail it is a guess** — twice now.

The run did find a real gap next door: using the aligner's *absolute* counters instead of the
step's delta survived the whole suite, because no case had both declined instants and an
overflowing ring in one run. One was added and it is caught.

**Low latency and honest weighting pull against each other, and both halves are now
measured.** With the fastest stream as the reference — which is what latency wants —
`hold_last` and `drop_if_stale` add **0 ms**; `nearest` **degenerates to `hold_last`** (same
sample on all 190 frames, since its future sample has not arrived); and `interpolate` builds
**zero frames**, not merely late ones, because bracketing needs a sample after the instant and
`drain` consumes the instant anyway. So the answer is `hold_last` + fastest reference, and
`step()` waits for nothing: output latency is the caller's call cadence.

The cost that hides: a slow target is relearned once per fast frame — 191 frames from 20
measurements — and `tr(P)` ends 10x (Python) to 12x (C++) below where that information should
leave it. **`max_stale_s` cannot fix it**: it is a threshold on *time* where the question is
*identity*. Measured, 0.10 s adapts exactly 20 times at ±20 ms of jitter and 32 times without
it — no value is right for both. **`require_fresh_target`** compares the target sample's own
timestamp instead and needs no number; off by default. Edges tested: `interpolate` (target
genuinely new every frame), target-on-the-reference (gate cannot fire), and a first target at
**exactly t = 0** — the second finding this session with that shape, after `am_ring_*_init`.

**`StepReport` carries the output now**, present even on a frame the gate refused: refusing to
*learn* is not a reason to stop *predicting*, and that separation is what lets a fast output
run on a slow target. Extraction moved ahead of the input gates in the reference to match the
port. The buffer is preallocated and read-only, with `sys.getallocatedblocks` over 500 steps
proving no per-step allocation crept back in.

**Comparing that output took three attempts at the metric.** Bits fail (θ carries rls.hpp's
declared divergences). Element-wise relative fails at **217 ulp at f64**, on a step where the
prediction is near zero — against the run's own scale it is **14.0 ulp**. Third quantity in
this project to need that treatment, after the state block and the a posteriori residual:
**whenever a quantity can pass through zero, a relative comparison measures the cancellation
rather than the implementations.** A NaN feature yields a NaN output on both sides, which is
the honest answer rather than a stale number.

Fifteen mutants, thirteen caught. One survivor was *equivalent* — a redundant initialiser in
`target_is_new()` — and was deleted rather than recorded. The other was real: zeroing the
prediction in the ABI shim survived, because everything compared `has_prediction` and nothing
compared what it pointed at.

**And `step()`'s cost is now decomposed, because a single per-step figure hides the thing
latency turns on.** `step_cost` sweeps cost against frames drained and fits
`fixed + per_frame x frames` — `fit_leak`'s technique, reused because a per-frame cost at one
batch size cannot tell overhead from work. Measured: **~11.4 us fixed + ~41.4 us per frame**,
so roughly a quarter of a one-frame step is overhead on that run. **The 0.5% run-to-run
stability this line first claimed was not real**: three consecutive runs agree that closely,
ten interleaved ones move 10.9% and the overhead fraction ranges 19.9-37.2%. A drifting mean
impersonating a stable one, which is what sampling three times in a row measures. The
intercept is worse still, being an extrapolation, which is why nothing compares two intercepts — an early `gate_saving` did and it was mostly fit conditioning. `fit_step`
also returns **NaN** rather than a line through one point: numpy fits it with a warning nobody
reads, and a fabricated slope gets quoted.

`require_fresh_target` measured in one run: **x0.22 per frame where it can help** (it skips
the update, which is nearly all the work) and **x1.07 where it cannot** — 1.00 inside the
harness's spread. Two different questions, conflated into one number in the first version.

All of it is Python, so ~41 us/frame is interpreter and numpy dispatch, not arithmetic —
`scaling` already showed RLS flat from n=4 to n=32. Not a baseline for the C++ core, for the
reason DESIGN.md already gives.

**The cross-compiler exists now (2026-08-09) and the target build is real.** Arm GNU
Toolchain 14.2.Rel1, installed per-user from Arm's own zip with the published SHA-256
verified — `winget`'s package is a machine-scope NSIS installer needing elevation. Flash and
RAM now come from a **link map**, as promised, and the link found three things a host build
cannot:

- **`norm()` was calling into libm**, and `-ffreestanding` was not why. GCC will not emit
  `VSQRT.F32` for `sqrtf` unless told it need not set `errno`. Measured: three call sites and
  zero `VSQRT.F32` without `-fno-math-errno`; zero calls and five inlined instructions with
  it. `Rls::update` calls `norm()` three times. The flag is now in the embedded profile; it is
  **not** fast-math and changes no result.
- **The core needs `memcpy` and `memset` from its environment** — 48 and 69 call sites, which
  GCC may emit even freestanding. Nothing here said so before. Beyond those, only libgcc's
  soft-float double helpers, which confirms `align.hpp`'s documented cost as a symbol list.
  The whole external set is pinned in `tests/test_target_build.py`.
- **The checked-in CMake cross recipe did not work.** Written before a toolchain existed and
  therefore never run: `project()`'s compiler check links an executable, which fails
  bare-metal. Same category as `select_newest` being compiled but never called.

**Measured sizes, Cortex-M4F, `-Os`, newlib-nano, from the map.** `--gc-sections` means there
is no one "size of the library", so four firmware-shaped configurations are linked separately:
**the figures live in `cpp/target/SIZES.md`**, generated and gated by
`tests/test_target_build.py`. The composed pipeline is about **32% of the 32 KB flash budget
and 40% of the 8 KB RAM**; the guarded row grew 208 bytes once an audit noticed `usage.c`
never called freeze/thaw/reset/counters/frozen_by, so `--gc-sections` had stripped six entry
points out of the configuration called "what the field should run".

**That table went stale three times**, which is why it is generated now rather than restated.
It drifted at `47a4dd2` (+40 on the composed row), at `17cff2b` (−16/−16/−24, and the note
recording this said "±16 on three rows" — wrong on the third), and at `0ada177`, **the commit
that documented the first two**. Two independent audits caught the third. Bounding against the
budget instead of pinning was deliberate — a compiler upgrade should not be a failing test —
and three drifts say it cost more than it saved. One method note worth keeping: summing the
whole link map, rather than only what follows "Linker script and memory map", reads 41,072
bytes for a 2,340-byte image.

**Every byte of that RAM is the caller's**: the library defines zero `.data` and zero `.bss`,
72 B of `.rodata`. The storage-is-the-caller's design, confirmed and now tested.

**And the code now RUNS on a Cortex-M4 (2026-08-09), which closes the last verification
limit.** `cpp/tests/test_core.cpp` executes under QEMU's `mps2-an386` — the declared floor —
and all 4,652 checks pass. QEMU was extracted rather than installed, its NSIS installer being
`requireAdministrator`; getting an extractor meant unpacking 7-Zip's own SFX with `7zr.exe`
first.

**What running it bought.** `persist.hpp`'s wraparound guard was documented as an *equivalent
mutant on a 64-bit host* — threshold 2.3e18 against a `uint32_t count`. On a 32-bit target it
is 536,870,876. A test now forges a `count` of `1 + 2^29`, chosen so the wrapped product makes
the length check *agree* with the real buffer, and repairs the CRC so it is rejected for its
length rather than for corruption. Measured both ways: removing the guard gives **2 failures on ARM and 0 on x86-64**. A guard
that could only be argued for is tested, and the claim of host-untestability is itself
verified. **That covered one of two identical guards** — `pack` carries the same check with
the same comment, and removing *it* still passed on ARM until an audit found it. Both are
covered now, and the write path matters more: its `count` arrives from a caller across the C
ABI rather than out of a CRC-protected record.

**Three harness bugs the host build would have hidden**, each a way a cross build that
compiles perfectly still does not run: a two-entry vector table (a fault has nowhere to go, so
it locks up with PC=0 and no cause); a fault reporter that called `printf` when the fault was
*inside* printf; and **nothing copying `.data`** — newlib's crt0 zeroes `.bss` and assumes a
loader placed `.data`, which is false for a ROM image, so `_impure_ptr` came up as garbage and
`__sflush_r` branched through stdout's uninitialised vtable. Diagnosing the third needed the
first two fixed, which is the argument for a startup that reports rather than one that is
short.

Also confirmed: `VSQRT.F32` of 2.0f returns `0x3fb504f3` exactly, and `int`/`long`/`void*`/
`size_t` are all 4 bytes.

**Still absent, and now for a narrower reason: any on-target timing.** QEMU models the
instruction set, not the pipeline or flash wait states, so a cycle count here would be fiction
dressed as measurement. An *instruction* count would be honest, being architectural rather
than temporal, but the Windows build ships no TCG plugins. The claim stays absent — what
changed is that the reason is now "nobody has measured it on hardware" rather than "nothing
can run the code".

**Parity now crosses architectures too.** Every parity claim had been x86-64 against
x86-64; the record format's own declaration that "the writer and the reader may be different
machines" had only ever been checked on one. All four tools now build for Cortex-M4 and run
under QEMU against the same Python reference — unchanged, because semihosting carries argv in
and file I/O out, so `tests/test_target_parity.py` reuses the existing harnesses' encoders and
comparators rather than growing a second set that could drift.

Holding on the target: record bytes identical at f4/f8 and 6x1/12x3, a host-written record
unpacking with every field intact and a bit-flip still refused; RLS to the same ulp at both
precisions; the guard's freeze step and applied/blocked sequence exact over 400 steps; the
aligner's picks bit-identical under three policies; the pipeline step-for-step including the
freshness gate. **A curated subset, not the whole suite** — a QEMU invocation is about a
second and the host sweeps run to hundreds of configurations — and which cases run is stated
rather than left to be assumed.

**The parity audit found a real bug and an undeclared divergence, and left four things
open.** Fixed: `Pipeline::step` emitted its prediction only `if (frames && frame_ready_)`,
and `build_frame` cleared that flag on entry — so a step whose *later* instant failed reported
frames and no output, while the reference, which only sees frames that exist, predicted.
**118 disagreeing steps in 250 randomised cases**, host and target alike. Extraction builds
into scratch and commits on success now, which removes the hazard the flag was marking rather
than guarding it. The parity case added for it catches both halves of the old behaviour and
**not** the gate alone — that is an equivalent mutant now, and checking which mutant a test
kills is a separate act from writing it.

Declared: **a fourth divergence in `rls.hpp`**, target-only and nobody's choice. GCC contracts
`a*b + c` into `VFMA.F32`, which VFPv4 has and SSE2 does not, so target f32 figures are not
host f32 figures (state 2.0 → 5.0 ulp, gain_norm 87.4 → 134.0); `-ffp-contract=off` reproduces
the host exactly. `align.hpp` said the M4F "cannot" contract — true of its double blend, false
of the f32 core, and the two sentences read as one.

Open, all measured: the 400-ulp state tolerance is exceeded (1008 f32 / 472 f64) over the
harness's expressible grid — *not* the pipeline, identical through plain RLS, so it is the
declared divergence at conditions the 1.5-ulp headline never swept (`excited()` has SNR ≈ 200).
The prediction metric normalises by the run's largest prediction where the project's own rule
says `max_j|x_j θ_j|`. ~~The cross-architecture subset discriminates nothing and its cost argument was wrong.~~
**Closed the same day: there is no subset.** `tests/_toolbuild.py` builds a tool for either
architecture and the four suites are parametrised `[host]` / `[target]`, so every parity case
runs on both — 510 tests in about 150 s, skipping cleanly without the toolchains. What is left in
`test_target_parity.py` is the question the suites cannot ask, since each compares one build
against the reference and never the two against each other: the target's deviation must stay
within **4×** the host's. Not 1×, because `VFMA.F32` contraction legitimately costs ~2.5× at
f32; and verified rather than assumed — the round-toward-zero fault was rebuilt and the check
reports **37.5×** and fails. f64 is asserted identical between builds as the control. Smaller: the
baseline ulp table reproduces at λ=0.99 only (gain_norm 110 vs 58 at 0.97), and `pipecheck`
instantiates only M=1 so no multi-output case has run through the freshness gate.

**Both remaining tolerance findings are closed, and they were the same finding twice: a bound
that looked comfortable because every fixture behind it was quiet.** The prediction metric
normalised by the largest *prediction* in a run where the project's rule is the scale of the
values being subtracted, `max_j |x_j theta_j|`; correcting it made the comparison **stricter**
(a case reading 134 ulp now reads 401). And `excited()` puts 0.01*randn on a signal of 3, so
there is now a fixture where the noise is the size of the signal. Re-derived across SNR 300 to
1 and p0 1 to 1000, on both architectures: state 1.5 -> 249 ulp, prediction 4.8 -> 947, with
p0=1000 dominating both. **One number cannot cover that, so there is not one** — the defaults
(400 / 200) hold for any sane prior and a deliberately bad one declares its own (500 / 1500)
with the measurement beside it. Loosening the default would have made it meaningless for the
other twelve cases. The deviation is conditioning, not the port: the same data through plain
RLS gives the identical figure, so the "1.5 ulp on theta" headline is a property of
`excited()`. The host-to-target ratio bound moved 4 -> 8 for the same reason in miniature: 4
came from one fixture at 2.5x and the sweep found 4.7 on a noisier one.

**The last two audit items are closed.** The baseline ulp table said its maxima came from
"λ=0.97 and 0.99" and reproduced only at 0.99 — `gain_norm` 58 where 0.97 gives 110, `p_trace`
1.4 where it gives 10.0; an axis named in the prose and not swept in the measurement. It is
re-measured over every shape `xcheck` instantiates, both λ, and now both architectures. That
also exposed uneven headroom in `FIELD_ULP`: twenty times for `p_trace`, **1.5 times** for
`gain_norm` at f32 on the target, where DESIGN.md had said "three to eight". And `pipecheck`
now instantiates three outputs, so the freshness gate finally runs with more than one — the
reference compares a tuple over unique target streams and the core iterates outputs, which had
been checked by reading and never by running. The first version of that fixture put the
targets on the reference stream so the gate could not fire; it failed its own guard, the third
time this session.

**And the stack had never been measured**, which a hardware question exposed. `am_*_bytes()`
reports the object; these functions also use the caller's stack. With `-fstack-usage` for
Cortex-M4 at `-Os` the hot path is **312-328 bytes** and `init` is a transient spike — but
before the fix `am_pipe_g_f32_12_1_init` wanted **8,296 bytes, more than the whole 8 KB RAM
floor**, because it built the object as a temporary and copied it. On the target this library
names, that entry point could not be called. Placement new brought it to 4,288 (and 1,536 for
`g_f32_6_1`); the header comment that had ruled placement new out said `<new>` was unavailable
freestanding, and it is not. One adapter-sized temporary remains, because `Pipeline` takes its
adapter by value — recorded, not half-done. Both figures are now bounded by a test.

**And the two halves had to be added up, which nothing did until a second hardware question.**
The object and the frame that constructs it share the same RAM, so the integrator's figure is
`AM_PIPE_G_*_MAX_BYTES` + the `init` frame. That first read **8,384 at `f32_12_1`** and this
file concluded "the declared 8 KB floor is a six-feature floor". **Withdrawn the same day: the
cause was the by-value adapter, not the shape.** `Pipeline` and `Guard` each copied an adapter
the caller had already built, the residue `argmind.h` recorded as "one adapter-sized temporary
… recorded rather than done". Adding the 128-wide network-head shape made it absurd —
`am_pipe_g_f32_128_1_init` wanted **397,192 bytes of stack to build a 332,392-byte object**, an
entry point callable on nothing. Both constructors now take an `in_place_adapter_t` tag and
build the adapter from its own arguments, nesting so a guarded pipeline has no temporary at any
level, with `Rls::params_valid()` split out so the ABI can refuse a bad configuration without
constructing something to ask. Measured: **397,192 → 504**, `f32_12_1` **4,288 → 168**,
`f32_6_1` **1,536 → 152**, guards **32**. Every shape benefited, not just the one that exposed
it, and `f32_12_1` now fits the floor at 4,264. `MAX_INIT_STACK` tightened 5 KB → 1 KB. **The
test forced the correction** — its failure message said "update the claim rather than this
test". What remains is a tier boundary, not a defect: `f32_128_1`'s *object* is 332 KB, so it is
not offered on this floor, and the test separately asserts the reason is the object and not the
frame.
**The per-shape flash/RAM sweep is back, generated this time.** It was withdrawn once
(10,404 / 10,664 / 10,380) for coming from an unrecorded hand-edit of a `usage.c` that had no
shape knob, with one row disagreeing with this project's own table. `usage.c` has the knob now
— `-DAM_SHAPE/-DAM_N/-DAM_M`, with the map wiring written against the dimensions rather than
against a shape, which is what stops the next hand-edit — and the rows are in
`cpp/target/SIZES.md` under the same gate as everything else there.

Measured, composed pipeline, Cortex-M4F: **flash 10,324 / 10,508 / 10,200** at 6×1, 6×3 and
12×1 — a **3% spread, and 12×1 is the *smallest* of the three** — against **RAM 3,408 / 3,872 /
7,912**. The reasoning that survived the withdrawal is now the measurement: compile-time
dimensions change how far a loop runs, not how much code exists. **Size the RAM, not the
flash** — and 12×1 sits at **96.6% of the declared 8 KB floor**, at the edge rather than
comfortably inside it.

Two things moved in the process, both wiring rather than regressions: the composed row's RAM
3,136 → 3,408, because the map now takes `AM_N-1` features from the reference stream and the
constant where the hand-written version took three, one and a repeat; and the record slots are
sized `state*4 + 96` rather than a flat 256, after the first version put 12×1 **40 bytes over
the floor** by over-reserving them.

**Next:** hardware, for the timing question. Everything else the project set out to build is
built and verified on the declared floor.

**Before any optimisation lands, a benchmark that can report a regression must exist.** An
optimisation with no way to fail is not an optimisation, it is a hope.

**Target range — settled 2026-08-08. The floor is Cortex-M4F class**: ARMv7E-M with a
single-precision hardware FPU (M4F/M33/M7, or RISC-V with F). fp32 FPU required, 8 KB RAM for the
library, 32 KB flash, C++17 freestanding with no exceptions/RTTI/heap, reference flags
`-mcpu=cortex-m4 -mfpu=fpv4-sp-d16 -mfloat-abi=hard`. That closes the matrix-type question: **own
minimal fixed-size types, no Eigen.**

**What decided it was the precision floor, not size or speed, which was not the expected answer.**
Computed from the reference's declared shapes: the driver configuration needs 3.3 KB of RAM, and an
analytic flop count puts n=6 at 5 Hz at 1.3 kflop/s — even n=30 at 200 Hz is ~2% of a 48 MHz M4F
(that ratio an estimate from published cycle counts, not measured). Both comfortable. The binding
constraint is that every covariance operation is fp32 with no fp16 path, so without an FPU it is
soft float at 20–100 cycles per op on data-dependent paths, and this project bounds the *worst*
case. An FPU is what makes the timing analysable.

**M0+/M3 are not supported and not claimed** — they would probably work (n=6 computes to 0.27 ms
per soft-float update, 0.14% duty at 5 Hz) but nobody has run it on one, and writing untested
support down is how a claim becomes a fiction. No target hardware exists on this machine, though a cross-compiler and QEMU arrived on 2026-08-09: flash and RAM now come from a link map and execution is verified on emulated Cortex-M4 and Cortex-M55, so what is left is **any
on-target timing claim stays absent rather than interpolated from host numbers.**

**Two port notes** from that review, deliberately not applied to Python where call overhead hides
them: `P /= lambda` is n² divisions (36 at n=6; `VDIV.F32` is 14 cycles on an M4F, so ~500 cycles
for one line) and must become one reciprocal plus n² multiplies, as must `k = Px / den`. And the
Python timings are **not** a baseline for the C++ — they are interpreter and numpy dispatch cost,
which the port does not pay, so comparing them would flatter it for reasons unrelated to the port.

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

No git remote is configured — deliberately, so a stray `push` cannot write to the retired Google
Drive location. Add one when publishing.
