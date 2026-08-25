<!-- Moved out of CLAUDE.md on 2026-08-16 so the file loaded into every session stays
     small. The text is verbatim; nothing was summarised or dropped. CLAUDE.md carries a
     one-line pointer to each of these. -->

# The runtime library, the C++ core, the C ABI and the target — as built

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
1,013 Python tests and 4,652 C++ checks passed **when this was written**; on 2026-08-25 an
audit measured **1,191 and 5,653**, so both figures here had gone stale unnoticed — the C++
gate is a `>=` floor and cannot see a count *grow*. Several checks exist to prove failures
are visible rather than silent. Every suite is mutation-checked, and surviving mutants are
recorded where they live rather than tallied here — including the ones that turn out to be
**equivalent on a 64-bit host**, a distinction worth keeping rather than rounding away. Cumulative counts are
avoided on purpose: the two that used to sit in this paragraph were both wrong until an audit
measured them, and the two that replaced them went stale the same way — **four for four**.
A count in prose is not protected by anything; the assertion in `tests/test_cpp_parity.py`
is.

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
and all 4,652 checks pass. **Re-measured 2026-08-25 as the suite has since grown: 5,653
checks, 0 failures, on `cortex-m4` AND `cortex-m55`, which is exactly the host count** —
so "the whole suite runs on target" is current and not a claim inherited from 08-09.
QEMU was extracted rather than installed, its NSIS installer being
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
