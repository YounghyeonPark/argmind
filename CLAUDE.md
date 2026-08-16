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
`RuntimeShaped*` classes take their shapes at construction. 1,024 Python tests and 4,652 C++ checks
pass; every suite is mutation-checked. Compilation, linking **and execution** are verified on the
declared floor — the full C++ self-check suite runs under QEMU on Cortex-M4 and Cortex-M55.

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

**Not yet built**: a `Stack` that owns the chain rule and is itself an `Adapter` — which is what makes
Guard, Pipeline and persist cover every level at once — and `Backbone` formally satisfying the layer
protocol it already meets in substance (`features`/`backward`/`apply_gradients`). Until then feature
learning still lives in 42 lines of the demo.

**Open numbers nobody has**: backward-pass throughput on an M55; whether fp32 holds past ten million
updates on a non-stationary target; and why the map head's two harnesses disagree on free-running
IoU (0.09–0.16 against 0.96–0.98 on the same scene).

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
