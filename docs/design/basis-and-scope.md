<!-- Moved out of DESIGN.md on 2026-08-16 to keep that file a design document rather than
     a running log. The text is verbatim; nothing was summarised or dropped. -->

# Scope decisions of 2026-08-09/10 — the platform generates the basis, and what widening cost

## The platform generates the basis — scope decision, 2026-08-10

The answer above ended by saying the basis comes from outside and that this library is not a
deep-learning runtime. The owner's response is that **on a platform, generating the basis is
the platform's job, and a self-contained learning and inference network belongs in it too.**
Taken. This reverses *"the caller supplies the basis"* and the fourth bullet of *What it is
not*, and it is the largest scope change the project has made.

**The structural question was asked before the decision was recorded, because it could be
measured.** `Guard` and `Pipeline` were shown model-agnostic with NLMS — a different algorithm
with no covariance. A backpropagating network is a much harder case: it has *layers*, its state
is weights, it needs a backward pass, and it has no closed form for anything. A three-layer MLP
wearing the `Adapter` contract, run against the framework unchanged:

- **`persist` packed it with no special case** — 1,217 weights in one flat contiguous block, in
  layer order, 9,766 bytes, unpacking to the same count. The flat-block rule was written for a
  covariance and holds for a network.
- **`Guard` accepted it** and built its ring at `Depth × state_size`.
- **A `p_trace` bound was refused**, exactly as for any adapter with no covariance. The
  adapter-declared `health_fields` mechanism, added for NLMS, covers this case without change.

So **the framework does not have to move; only the model class does.** That is a real result and
not a comfortable one — it means the cost is arithmetic rather than architecture, and the
arithmetic is unflattering in two of three directions.

### What learning the basis buys, and what it costs

Same target as above, `sin(3s₀)·exp(−s₁²) + |s₂| − 0.4`, 2,000 held-out points. A constant
predictor scores 0.7641. RMSE against the number of samples *seen*, so sample cost and final
accuracy are separated — a table of final accuracies alone would hide the whole trade:

| model | state | 200 | 1k | 5k | 20k | 60k |
|---|---|---|---|---|---|---|
| RFF D=32, RLS | 8.8 KB | 0.5679 | 0.4911 | 0.4788 | 0.4757 | 0.4756 |
| RFF D=128, RLS | **131 KB** | **0.4656** | **0.3613** | **0.2605** | 0.2521 | 0.2458 |
| MLP 3-16-1, SGD | 0.6 KB | 0.7769 | 0.6175 | 0.5194 | 0.4921 | 0.5340 |
| MLP 3-32-32-1, SGD | **9.5 KB** | 0.7822 | 0.7206 | 0.3873 | 0.2455 | **0.2484** |

**Memory: the network wins by 14×.** 0.2484 at 9.5 KB against 0.2458 at 131 KB — the same
accuracy, and RLS's covariance is `n²` while a network's weights are not. This is the argument
for the decision and it is a strong one.

**Samples: the network loses by ~4×.** RFF reaches 0.2605 at 5,000 samples; the MLP needs 20,000
to match it, and at 200 samples it is *worse than a constant* while RFF is already at 0.4656.
RLS is the exact recursive solution of a least-squares problem and SGD is a walk downhill. On a
device that must be useful in its first minute, that difference is the whole story.

**The small network's 0.4921 at 20k and 0.5340 at 60k looked like a network getting worse with
more data.** It was written up here as "the finding that matters most" and it is **withdrawn**:
it is two points picked out of a flat noisy series. The audit, ten runs over five initialisations
and two data streams, RMSE fitted over `k = 20k…60k` rather than compared at the endpoints:

- **the slope is positive in 1 of 10 runs — the one seed that had been quoted.** Mean −0.0250
  per 10k samples, sd 0.0140: nine of ten keep improving.
- in that one run the slope is **+0.0026 against a within-run scatter about its own trend line
  of 0.0216** — the "trend" is an eighth of the noise it sits in.

The rule this breaks is one this repository already had (*a uniformly tidy result is evidence
the measurement is broken*) applied to its mirror image: **a result that is interesting because
two endpoints disagree needs the points in between.** The commit that introduced it quoted the
non-monotonicity from a two-checkpoint run and never plotted the middle. What caught it was
asking a *different* question — whether the residual tracks held-out error — which printed the
series at twelve checkpoints, and the series was flat.

### The question was the wrong one, and the owner named it

Two days of this drifted from one question into another without noticing:

| | asks | needs | is it the guard's job |
|---|---|---|---|
| **validation** | is this model any good? | data the adapter has not trained on | **no** |
| **fault detection** | did something just break? | the stream itself | **yes** |

Searching for "a signal that sees a slow loss of quality" is the first one. Online adaptation
**improves a model by repetition; it does not pre-judge whether it will work** — the owner's
correction, and it dissolves the problem rather than solving it. A guard is reactive by
construction: checks run *after* the update, because the quantities worth checking do not exist
until then. Nothing in this design ever claimed to predict.

**So the honest measurement is whether the guard does its real job on a network.** A sensor
latched high for 200–400 samples, held-out RMSE against the model's own quality just before the
fault, unguarded, with an RLS control beside it:

| clean samples since the fault ended | +0 | +100 | +500 | +2,000 | +5,000 | +20,000 |
|---|---|---|---|---|---|---|
| MLP 3-32-32-1, SGD | 27.4× | 7.6× | 3.9× | 3.8× | 3.4× | **3.33×** |
| RLS, rich basis | 9.4× | 9.0× | 7.4× | 3.6× | 1.3× | **1.00×** |

**RLS recovers completely and SGD does not.** Once clean data resumes the least-squares recursion
pulls back to the correct fit; a walk downhill from a corrupted position does not retrace, and it
stalls at 3.3× for as long as the run continues. So the guard's value differs by model class in
kind rather than degree: **for RLS it buys time, and for a network it is what prevents permanent
damage.**

**Two claims were made here and withdrawn before they were written down**, both from reading one
configuration as a property:

- *"The network is 4.5× more damaged than RLS (1.03×)"* — that was a 30,000-sample run in which
  RLS had 4,600 clean samples to recover in and the comparison was made at the end. It was
  measuring recovery and calling it vulnerability.
- *"The network is more vulnerable at impact"* — at +0 it is 27.4× against 9.4× in that run, but
  at a shorter warm-up with `forgetting=0.995` the same fault gives 29.0× against RLS's **32.7×**.
  **The ordering flips with the forgetting factor**, so vulnerability at impact is not a property
  of the model class and no version of that sentence is kept.

Guarded, at the same fault: `Divergence` freezes at **exactly the fault step** in every case, on
`residual_norm` alone — a field every adapter reports — and the damage does not happen. The NaN
case never freezes and takes no damage either, because the adapter rejects a non-finite target at
the input and counts it, so nothing reaches the model; that is correct, and worth naming, because
"no freeze" reads as a miss.

**One flattering reading declined.** The guarded MLP ends *below* where it was before the fault,
which looks like the guard improving the model. It is not: rollback restores a checkpoint up to
800 samples old, and this run's held-out RMSE fluctuates over 0.21–0.41 across its own
checkpoints, so both figures sit inside the noise. The claim is damage prevented, not improvement.

**What this corrects.** "The model class arrives with the guard's best signal missing and a new
failure mode that needs one" was written above and is **too strong**. `p_trace` is genuinely
absent and genuinely useful — per-direction confidence, which for RLS catches windup in a
direction nothing has excited. But it answers a validation-shaped question, and the guard's own
job is answered by the residual for both model classes. What remains missing is narrower and is
now stated narrowly: **there is no confidence-per-direction for a backpropagated weight, so a
windup-shaped fault has no analogue and no check.**

### The residual still does not substitute for `p_trace`, and no check can be built on it

The concern the withdrawn claim was standing in for is real and unresolved. Measured three ways:

**No configuration of the existing checks detects a slow loss of quality.** `Divergence`'s own
docstring says a fault the baseline can track will never trip; swept, it is worse than that.
At its default `ratio=5.0` it never fires; tightened to 1.05 it fires at step ~201, immediately
after warmup, on ordinary noise. `Bound("residual_norm")` fires at step **6** for every
`hi` in 0.8…2.0 — the untrained model's noise — and never at 3.0. **There is no setting between
blind and instantly wrong.**

**Two new designs failed their negative controls.** A long EWMA against a baseline latched at
`settle`, and the same with the baseline tracking the best seen so far, both to remove
`Divergence`'s flaw of a reference that follows the drift. At `ratio=1.10` the tracking variant
fires on the degrading network (24,583) — and also on a network that is **improving** (26,424)
and on **stationary RLS** (23,717). At 1.25 and above nothing fires anywhere. No discrimination
at any threshold.

**Because the quantity does not carry the signal.** EWMA of `‖residual‖` at α=1e-4 against
held-out RMSE, twelve checkpoints:

| | EWMA range | held-out range | correlation |
|---|---|---|---|
| MLP 3-16-1, misspecified for its capacity | 1.023× (**falling**) | 1.115× (rising) | **+0.112** |
| MLP 3-32-32-1, improving | 1.857× | 1.937× | +0.631 |
| RLS, linear basis on a nonlinear truth | **1.518×** | **1.002×** | **−0.287** |

The last row is the clearest: the smoothed residual moves by half while the thing it is supposed
to proxy is flat to two parts in a thousand, and it moves the *wrong way*. The responsible
assumption is **misspecification** — a control with a well-specified RLS gives +0.895 — and
misspecification is the normal condition for an adapted head on a frozen backbone. A prequential
residual is optimistic exactly when the model is fitting the samples it sees and not the
population, which is the failure a quality signal exists to catch.

**So the check was not built**, and after the correction above it should not be: a mechanism for
a failure mode that could not be reproduced would be a mechanism justified by its own fixture,
and the question it answered was the validation one. What a validation signal would need is data
the adapter has *not* trained on — the guard withholding a fraction of samples — which is a real
design with a real cost, and belongs to whoever wants validation rather than to the guard.

**The guard's ring is `Depth × state_size` and that scales with the model, not with the
sensors.** At depth 4: RLS n=6 is 1.6 KB total, the MLP 47.5 KB, a 3-64-64-1 net 175 KB, and
RFF-128 through RLS **655 KB**. Comfortable on the SBC/NPU tier, impossible on the 8 KB core
floor — so this is a second-tier capability by arithmetic, not by preference, and the core tier
keeps the model class it has.

### What this does not settle

Training a *deep* backbone on-device is still not available and the reason is unchanged: backprop
retains activations, not weights. What the probe shows is that a small head-sized network trains
inside the contract. Between "adapt a linear head" and "train a backbone" there is a range, and
where in it this platform stops is not decided by this measurement.

## "Is linear not too stupid?", measured 2026-08-10

Asked directly, and the honest answer has three parts.

**Linear in the PARAMETERS is not linear in the inputs.** The regressor is whatever the caller
puts in it, so a polynomial, spline, Fourier or random-feature basis gives a nonlinear model
while the learning stays an exact recursive least-squares solution. Demonstrated: a truth
containing `0.8·s₀²` is unfittable by a linear basis and fitted to 1e-5 by a quadratic one, at
the same three sensors.

**But on a genuinely hard function, a bare linear basis is exactly as good as a constant.**
Against `sin(3s₀)·exp(−s₁²) + |s₂| − 0.4`, over 2,000 held-out points:

| basis | n | RMSE | vs a constant | state |
|---|---|---|---|---|
| constant predictor | — | 0.7641 | 1.0× | — |
| linear | 4 | **0.7640** | **1.0×** | 80 B |
| quadratic | 10 | 0.5201 | 1.5× | 440 B |
| cubic | 20 | 0.5169 | 1.5× | 1.7 KB |
| random Fourier, D=32 | 33 | 0.4654 | 1.6× | 4.5 KB |
| random Fourier, D=128 | 129 | 0.1474 | 5.2× | 67 KB |
| random Fourier, D=512 | 513 | 0.1071 | 7.1× | **1.05 MB** |

So the challenge lands. A linear basis learns nothing here, and buying accuracy with a *generic*
basis costs `n²` — half a megabyte of covariance to reach 7×, and cubic barely improves on
quadratic because the truth is not polynomial at all.

**The third part is where the answer actually is: it is the basis that matters, not the width.**
An oracle basis of eight terms — the truth's own building blocks — fits it to machine precision
in **288 bytes**, beating five hundred random features at 1/17,000th of the state.

Nobody has that oracle. **A trained network is the machine that finds those eight from data**,
and that is the whole argument for the frozen-backbone-plus-adapted-head design: the backbone
supplies the representation, argmind adapts on top of it, and the system is as expressive as its
backbone rather than as its adapter. Calling the adapter linear describes the last layer, not the
model.

**What linear-in-the-parameters buys, and it is not nothing.** No local minima and no learning
rate. Memory and time known before the device ships. `p_trace` is a real per-direction confidence
— there is no cheap equivalent for a backpropagated weight — and it is what the guard's bounds
and rollback are built on. Those properties are why the adapter is this and not something else,
and they are given up together the moment it is not.

**What it cannot do, unchanged from the model-class note above.** It cannot learn the basis: if
you do not know which features matter, no amount of data supplies them. It scales badly in the
input dimension: a degree-2 basis on 128 inputs is 8,256 terms and `n²` on that is 68 million
covariance entries. And nothing whose output depends on the input's *history* can be fitted by
instantaneous features at all.

## A bigger board does not make a bigger model, 2026-08-10

Asked directly, and the answer is no: the model's size is `n_features` and `n_outputs`, which
are a **design choice**, not a property of the silicon. A larger part lets you *choose* a wider
one; nothing widens by itself.

**What actually grows is the machinery, not the model.** `theta` is `n·m` values — that is the
whole of what was learned. `P` is `n²` and is how the learning is *done*. At n=6 the learned
model is 24 bytes against 144 of covariance; at n=128 it is 512 against 65,536. **Six times over
at the small end and a hundred and twenty-eight times at the large one**, which is why the
memory table climbs the way it does and why a second adapter without a covariance exists at all.

**Keeping the same shape on a better board already works and is verified** — `braincheck` moves
a trained model between Cortex-M4F and Cortex-M55 in both directions and the prediction is
identical to the last printed digit.

**Changing the shape is refused, and that is correct.** The configuration fingerprint has no way
to know which old feature belongs in which new slot, and guessing would put the right numbers on
the wrong features — the one corruption nothing downstream detects.

**What a deliberate widening would buy, measured.** Six sensors become twelve; the old six keep
their meaning. Grafting the trained `theta` and its `P` block into the corner of a fresh n=12
model, against starting from zero, on the error in `theta` against the truth:

| updates | from zero | grafted |
|---|---|---|
| 10 | 1.0919 | **0.0586** |
| 20 | 0.0152 | 0.0095 |
| 400 | 0.0015 | 0.0013 |

So it is a **head start of roughly ten to twenty samples, not a permanent advantage** — RLS
re-converges quickly once the new features are excited. Worth building only where updates are
expensive: a self-supervised label at a few hertz, or new features that are rarely excited. It
is **not built**, and the measurement is here so the decision is priced rather than guessed.

## Scope widened, 2026-08-09

Everything above describes the library as built. **The owner reopened the scope on 2026-08-09**
for a target application needing camera and microphone input, situation assessment, inference,
actuator output and speech. The concern was raised that a linear model does none of that and
that the power cost of a perception board is severe on a small legged robot; the decision was
made with that stated, so it stands.

**The framework survives; the adapter is the narrow part, and that is measured.** A second
adapter — normalised LMS, O(n), **no covariance matrix at all**, state `N*M` rather than
`N*M + N*N` — passes through `Guard` and `Pipeline` with **no change to either template**,
compiling clean at `-Wall -Wextra` on the host and for Cortex-M4F freestanding, learning
correctly through the guard, and shrinking it to 432 bytes from RLS's 1,144 at the same depth.
So `Stream`, `Aligner`, `FrameMap`, `Guard`, `Pipeline` and the record format are the reusable
contribution, and it was never RLS that made them worth having. `cpp/tests/test_core.cpp`
carries the probe so this is protected, not asserted.

**One leak, and it has to be closed first.** `health()` hands out two *covariance* scalars, and
`Field::PTrace`/`Field::PMinDiag` are two of the guard's five checkable fields. An adapter with
no covariance reports zeros, and the same bound then fails in **both** directions:

| bound | on RLS | on a covariance-free adapter |
|---|---|---|
| `at_most(PTrace, 1e4)` | a real ceiling | **dead** — 0.0 ≤ 1e4 forever; configured, always healthy, cannot fire |
| `at_least(PTrace, 1e-9)` | a sane sanity check | **fatal** — trips on the first update and freezes |

Silently dead one way and instantly bricking the other is worse than either alone, and the type
system permitted both.

**Closed the same day.** Health is now an **adapter-declared set**: an adapter names the fields
it can report, and a check on anything else is refused where the caller can still act on it —
at `add_bound` in C++, at `Guard.__init__` in Python. `Rls` declares all five; the probe
adapter declares three.

- **`Divergence` is covered too, and it is the one that mattered most.** It watches
  `residual_norm` alone, so an adapter that cannot report it would have the check comparing a
  constant against itself — and divergence is the check that needs no units, so it is the one
  a sensor-agnostic caller is most likely to enable and least likely to re-derive. `valid()`
  refuses that combination.
- **`health()` is no longer mandatory.** The guard's calls sit behind
  `if constexpr (needs_health(Adapter::health_fields))`, so for an adapter declaring neither
  covariance field the branch is never instantiated and the method need not exist. In Python
  it stops being abstract and the base returns zeros nothing can read.
- **A `Check` now declares what it `watches`**, abstract rather than defaulted, so a new check
  has to say. That closed a second hole nobody was looking for: `WATCHABLE` admitted `applied`,
  `reason`, `residual` and `residual_post`, and `Bound.evaluate` does `float(getattr(...))` —
  so `Bound("reason")` constructed fine and crashed at evaluation. It is refused at
  construction now, because `Rls` does not declare those four.
- **The `Adapter.health` docstring was self-contradictory** and the wrong half was the
  instruction. It said inventing a number "is precisely what this project's standards forbid"
  and then, two sentences later, told an adapter with no covariance to "report the nearest
  analogue". There is no analogue of a covariance trace in an algorithm without a covariance.
  Declare fewer fields instead.
- **No ABI query, deliberately.** Every guard the C ABI exports wraps `Rls`, so
  `am_guard_*_health_fields()` could only ever return one constant. This project has already
  shipped two entry points nothing called; a third is not worth the surface. Recorded in
  `argmind.h`, to be exported when a non-RLS adapter reaches the boundary.

Three mutants confirm it rather than the suite merely passing: removing the Python refusal
fails five tests, removing the C++ `add_bound` mask check fails three, and removing the
divergence guard fails one.

**Undecided, and each changes the work materially:**

- ~~**The second tier's floor.**~~ **Answered 2026-08-09, and the answer is that the floor does
  not move.** See below.
- **Hosted or embedded inference.** ~~An interface to ONNX Runtime / TFLite keeps this library's
  size and auditability; embedding a runtime does not.~~ **That argument was wrong and is
  withdrawn** — see below. Still undecided, but on different grounds.
- **One library or two.** Sharing the `Adapter` contract across tiers is the argument for one;
  the core tier's no-heap/no-exceptions/no-libm constraints are the argument for two.

**Nothing on the camera/audio/speech path is built.** This section exists so the documents stop
saying the scope is settled when it is not.

### The floor does not move — measured, not argued

The obvious reading of "camera, microphone, situation assessment, speech" is that the
Cortex-M4F floor has to be abandoned for an SBC, which would discard the link-map figures, the
QEMU execution and the freestanding gate. That reading is wrong, and what makes it wrong is a
class of part that did not exist when the floor was set: **an MCU with an NPU.**

- The **Cortex-M55** implements ARMv8.1-M with *native scalar and vector fp32*, plus scalar
  fp64 ([Arm, Floating-point and MVE
  support](https://developer.arm.com/documentation/101273/latest/Cortex-M55-Processor-level-components-and-system-registers---Reference-Material/Floating-point-and-MVE-support)).
  It is therefore **strictly above** the declared ARMv7E-M + fp32 floor, not beside it.
- **STM32N6** pairs that core at 800 MHz with ST's Neural-ART NPU at 1 GHz — ~300 MAC units,
  **600 GOPS**, **3 TOPS/W** — 4.2 MB of contiguous SRAM, and a camera pipeline with MIPI CSI-2
  and an ISP ([ST, STM32N6
  series](https://www.st.com/en/microcontrollers-microprocessors/stm32n6-series.html);
  [STM32N647Z0](https://www.st.com/en/microcontrollers-microprocessors/stm32n647z0.html)).
  At ST's own efficiency figure the NPU at full rate is **0.2 W** — that is arithmetic on
  600 GOPS ÷ 3 TOPS/W, the accelerator only, not a system power figure, and ST's parametric
  pages resist automated fetching so it has not been checked against a datasheet table.

**Measured here, same source, no changes:** the composed pipeline is **smaller** on Cortex-M55
than on Cortex-M4F at identical static RAM — moving up this way costs nothing and saves flash.
The two figures are in [`cpp/target/SIZES.md`](cpp/target/SIZES.md) rather than in this
sentence, because the pair that used to be here (7,876 / 10,396) went stale within a day and
no test could see it. And it **runs**: the whole self-check
suite, all 4,650 checks, executes under QEMU's `mps3-an547` Cortex-M55 board and passes.
`tests/test_target_run.py` is parametrised over both cores, so this is a test rather than a note;
two mutants confirm the case is real (breaking the check-count bound fails both cores, and
pointing the M55 case at the M4 board produces no result line at all).

**What that buys and what it does not.** 600 GOPS with 4.2 MB of SRAM is enough for small vision
— detection, pose, presence — and for audio keyword spotting. It is **not** enough for large
speech recognition or for anything an "LLM" would be a fair description of, and the SRAM bounds
model size hard. So the fork is not technical, it is about what the words mean:

| what "situation assessment and speech" means | part class | power | this floor |
|---|---|---|---|
| react like an animal: see a person, hear a call, answer with a sound | MCU + NPU (STM32N6 class) | ~1 W order | **unchanged** |
| converse: speech recognition, dialogue, natural TTS | SBC + accelerator, or Jetson | 10–25 W | abandoned |

The second row is expensive in a way that is specific to a small legged robot. The figures below
are **derived by me** from tomcat's stated NFR6 — 83.6 W trotting for 30 min and 168 min
standing, in `docs/REQUIREMENTS.md` of the `YounghyeonPark/tomcat` repository, which itself
labels its 175 Wh/kg pack density `[assumed]`. Neither the battery capacity nor the standing
draw is stated there: **41.8 Wh** is 83.6 W × 0.5 h and **14.9 W** is 41.8 Wh ÷ 2.8 h. Adding 10 W takes standing from
168 min to **101** (−40 %); adding 25 W takes it to **63** (−63 %). Trotting is barely affected
(30 → 26.8 → 23.1 min) because 83.6 W dominates. NVIDIA's Jetson Orin Nano runs 15 W, 25 W and
an uncapped MAXN SUPER mode ([NVIDIA, JetPack 6.2 Super
Mode](https://developer.nvidia.com/blog/nvidia-jetpack-6-2-brings-super-mode-to-nvidia-jetson-orin-nano-and-jetson-orin-nx-modules/));
Hailo's M.2 module is 26 TOPS ([Hailo-8
M.2](https://hailo.ai/products/ai-accelerators/hailo-8-m2-ai-acceleration-module/)) but Hailo's
own page states no wattage, and the 2.5 W figure in circulation could not be traced to a primary
source here, so it is **not** quoted as one. Raspberry Pi 5 idle figures of 2.2–2.7 W come from
Raspberry Pi *forum* threads rather than a specification, and are flagged the same way.

**So the recommendation is a second tier that is still an MCU**, with the core tier's floor left
exactly where it is and the higher tier declaring Cortex-M55 + NPU. That keeps no-heap,
no-exceptions, no-libm and the whole verification chain, and it is the only option of the three
that has been *run* rather than reasoned about.

### Can learning and inference be integrated, or only placed side by side?

Reviewed 2026-08-09, because a network foundation written from scratch could in principle put
them together rather than bolting one onto the other. **They can be integrated, at exactly one
place, and the place is forced rather than chosen.**

**The NPU does not learn.** Arm describes Ethos-U55 as an *inference* accelerator, its software
path is TensorFlow Lite **Micro** — an inference runtime — and the documentation makes no
provision for training or backward operators
([Arm, Ethos-U55](https://developer.arm.com/Processors/Ethos-U55)). ST presents Neural-ART the
same way. So "learning on an NPU-equipped MCU" is never the NPU learning; it is the CPU learning
beside it. That is a constraint on the design, not a preference within it.

**Backprop's cost is activations, not weights, and that is what closes the door on the rest.**
Forward-only inference can release each layer's activations as it moves; a backward pass must
retain them. Against 4.2 MB of SRAM and a no-heap contract, end-to-end on-device training of a
backbone is not available at this floor, and no amount of care in a new foundation changes it.

**Which leaves the head — and the head is where argmind already is.** Frozen backbone on the NPU
in int8, forward only, emitting a feature vector; the last layer adapted online by an `Adapter`
under a `Guard`. That is not a fallback shape, it is the standard one, and it happens to be
exactly the contract this library already has. The integration is *organic* in the sense that
matters: the forward pass's output **is** the regressor, so inference and learning share the
computation rather than duplicating it, and one flat state block still holds everything
persistence needs.

**Measured on the target cores rather than argued.** A 128-wide head — the range a small vision
backbone emits — driven 4,000 samples through the full guard:

| | guard object |
|---|---|
| `Guard<Rls<float,128,1>,4,4>` | **330,544 B** |
| `Guard<Nlms<float,128,1>,4,4>` | **2,872 B** |

Both instantiate and run on Cortex-M4F under `mps2-an386` and on Cortex-M55 under
`mps3-an547`. The sizes are exact and independently reproduced.

**An accuracy comparison stood here and is withdrawn.** It read "113× the memory for
~17,000× the accuracy", from a scratch fixture that was never checked in — the failure this
document already names elsewhere as "the sweep figures I quoted came from a run that left no
trace". An audit reconstructing the fixture got the **opposite ordering**, which does not
prove the published pair wrong but does prove it unsupported. Two things have to be fixed
before it can be re-stated: the fixture must be in the repository, and NLMS's step size must
be swept — μ sets its steady-state error and only μ=0.5 was ever tried, so the losing side of
a one-point comparison had its only tuning knob left at a default. What survives is the
memory ratio, which is arithmetic on the state sizes and needs no fixture.

**The wall is RLS's n², and head width is the design variable.** State is `N·M + N²`, and the
guard's ring multiplies it by `Depth`:

| head N | RLS state + ring(4) | of STM32N6's 4.2 MB | NLMS state + ring(4) |
|---|---|---|---|
| 32 | 21 KB | 0.5 % | 704 B |
| 128 | 330 KB | 7.5 % | 2.6 KB |
| 256 | 1.32 MB | 30 % | 5.2 KB |
| 512 | 5.25 MB | **119 % — does not fit** | 10 KB |

So a head wider than about 256 forces either a square-root/UD form, a lower rollback depth, or
NLMS. That is a real design boundary and it should be recorded before anyone picks a backbone.

**Getting it to run found a harness fact worth keeping:** the AN547's DTCM at `0x20000000` is
512 KB, not the 4 MB `mps2.ld` declares for the AN386, so a 662 KB `.bss` locked the board up
with `SP=0x200a1c50` and "can't escalate 3 to HardFault". The regions were then probed by
writing and reading back rather than taken from a document: `0x00000000`, `0x01000000`,
`0x20000000`, `0x21000000`, `0x60000000`, `0x70000000` and `0x80000000` are backed and
`0x28000000` is not, with `0x60000000` good for at least 64 MB. A head-sized image has to place
its data there.

**What this does not give, said plainly so nobody expects it.** No representation learning: the
backbone is frozen, and if the world changes in a way its features do not capture, no head can
recover it. "Situation assessment" is the backbone's job and the backbone does not learn on
device. And the head is linear in the parameters unless a different adapter is written.

**What it does give is the reason argmind belongs in this at all.** An online-adapted head is
precisely the component that drifts silently — it is the only part still moving after
deployment. Freezing, bounded rollback, checkpointing, the two-slot record, the staleness gate
and the refusal to watch a field the adapter cannot report all transfer unchanged. A frozen
network with an unguarded adapted head is a system that can learn garbage and keep answering
confidently, which is the failure this library was built around.

#### Hosted or embedded — and the size argument I first gave was wrong

**Hosted** means argmind never sees the network: the caller runs the backbone on whatever
runtime it has and pushes the resulting feature vector, and argmind's contract stays "give me
`x` and `y`". **Embedded** means argmind contains tensor types, operator kernels, a graph
format, a loader and quantisation semantics, and runs the backbone itself.

**The size argument does not survive contact with the numbers.** This document said embedding
a runtime would cost the library its size and auditability. TFLite Micro's core runtime is
**16 KB on a Cortex-M3**, 22 KB with enough operators for a keyword-spotting model, and it
requires **no dynamic allocation** — it works from a caller-supplied arena
([Google, LiteRT for Microcontrollers](https://ai.google.dev/edge/litert/microcontrollers/overview)).
Against argmind's own 10.4 KB that is a tripling, not an explosion, and the arena model is
*aligned* with storage-is-the-caller's rather than opposed to it. The argument was made from an
assumption and it was wrong.

**What actually separates them, in order of weight:**

1. **The NPU driver, and this one is structural.** Neural-ART needs ST's runtime; Ethos-U needs
   Arm's. They are vendor-specific. Embedding means either binding a vendor-neutral Apache-2.0
   library to one vendor, or reimplementing the accelerator path — and reimplementing it means
   running on the CPU, which discards the entire reason for the tier. Hosted sidesteps this:
   the vendor's runtime runs, argmind consumes its output.
2. **The verification model would break.** This project's claim is 4,650 checks, byte-identical
   record parity, ~1.5 ulp against a Python oracle, mutation-tested, executing on the declared
   floor. A network runtime cannot be held to that by a few-hundred-line reference. Embedding
   would mean shipping a component whose correctness argument is somebody else's test suite —
   good engineering, but not *this* claim, and a claim is only as strong as its weakest part.
3. **The external dependency set stops being pinnable.** argmind needs exactly fourteen symbols
   — `memcpy`, `memset`, `sqrt` and eleven libgcc soft-float helpers — and a test asserts the
   *set*, so a new one is a decision rather than an accident. TFLM plus CMSIS-NN plus a vendor
   driver ends that. It is a small test and it is what makes "freestanding" checkable.
4. **Format ownership.** argmind owns one format and holds it to every single-bit flip in a
   78-byte record, every truncation length, 400 fuzz blobs, byte-identical across
   architectures. That does not scale to a model file with a graph, quantisation parameters and
   versioning, and owning a second format at a lower standard than the first is worse than not
   owning it.

**The fair case for embedding, which is real and is not about bytes:**

1. **The guard could see inside the network** — and this is the strongest argument, because it
   answers a limit found the same day. No guard catches a misaligned target, since the guard
   watches the *head's* health and the head is healthy. Per-layer activation statistics are a
   different signal: an input distribution shift shows up in an early layer long before it
   reaches the head's residual. Hosted gives none of that.
2. **Partial backprop.** Owning the kernels makes fine-tuning the last convolution possible
   rather than only a linear head — a materially larger model class, still bounded by
   activation memory.
3. One artifact, one arena, and no version skew between backbone and head.

**A middle path that should be considered before either.** Host the execution *and* declare a
health interface: the caller runs the backbone and also pushes a few scalars from inside it —
an early-layer activation norm, an output entropy — as an ordinary `Stream`. The guard then
watches them through `Bound`, and the `health_fields` mechanism added the same day is exactly
what declares which scalars an adapter can report. That recovers most of embedding's only
compelling advantage without owning a single kernel.

**Measured, for the hosted side.** `am_ring_f32` already takes a runtime `dim` over caller
storage, so a 128-wide backbone output pushes through the existing API with no new concepts.
Adding the matching pipeline shape is **three lines** — a declaration in `argmind.h`, a definition in `abi.cpp` and a `MAX_BYTES` — which is one more than this document claimed until the compiler refused the two-line version —
built for Cortex-M55 freestanding at `-Wall -Wextra`, clean on the first attempt, **+4,876
bytes** of text and rodata before `--gc-sections`. That is what the boundary costs.

#### The label arrives after the features that earned it, and getting that wrong is silent

The head needs a `y`, and on a robot the honest ones are *self-supervised and late*: something
is predicted at `t` and a slower, more trustworthy sensor confirms it at `t + Δ` — foot contact
against a predicted step, measured torque against a predicted load, a second modality against
the first. The camera is fast and cheap; the thing that grades it is slow and physical.

`allow_non_causal` does **not** express this. It says a policy may pick a sample after the
instant; it does not say "these features go with the target that arrives Δ later". Nothing in
the pipeline declares a lag.

**No library change is needed, and the arrangement that works is worth writing down because the
one that does not is silent.** Make the *outcome* stream the reference, so instants are label
arrivals and `hold_last` reaches back for features, and **stamp each feature vector with the
instant it is about rather than the instant it was computed**. Measured, n=4, Δ=0.30 s,
features at 50 Hz and labels at 10 Hz:

| feature timestamp | relative error in θ |
|---|---|
| when it was computed (`t`) | **1.105** — a different map entirely |
| when it is about (`t + Δ`) | **1.56e−04** |

**And 1.56e−04 is not error.** It is the `P0⁻¹` prior term this project measured elsewhere,
confirmed here across four decades — p0 of 1/10/100/1000/10000 gives 1.53e−02, 1.56e−03,
1.56e−04, 1.56e−05, 1.56e−06, ratios ×9.85, ×9.98, ×10.00, ×10.00. The head learns the map
exactly.

**The part that matters for a safety story: no guard catches the wrong one.** Freezing, rollback
and the divergence detector all watch the *adapter's* health, and the adapter is perfectly
healthy — `residual_norm` is low, because it fits the wrong map well. It is being asked the
wrong question, and a guard cannot tell the difference between a good answer to the wrong
question and a good answer. Both halves are pinned in `tests/test_pipeline.py`, including the
guard that the naive arrangement still fails, and mutation-checked: undoing the stamp shift
fails both tests, and disconnecting `p0` fails the law test.
