<!-- Moved out of CLAUDE.md on 2026-08-16 so the file loaded into every session stays
     small. The text is verbatim; nothing was summarised or dropped. CLAUDE.md carries a
     one-line pointer to each of these. -->

# Platformisation, runtime shapes and the two tiers — 2026-08-10

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
