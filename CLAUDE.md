# Argmind — project context

**Multi-sensor AI software: fusion, on-device learning, and inference.**
Started 2026-08-08. This is the active project.

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

## Status — settled 2026-08-08

**Requirements are settled. Building has started.** The four scope decisions, made explicitly
and not to be re-litigated without a reason:

| axis | decision |
|---|---|
| platform | **embedded / edge** |
| learning | **on-device online adaptation** — not offline training |
| sensors | **sensor-agnostic**, general framework |
| deliverable | **a library**, not an application |
| language | **C++ core behind a C ABI**, optimised for speed (see DESIGN.md) |

The design driver, chosen so the API is shaped against something real: an inertial + aiding-sensor
estimator with online bias calibration. Sensor-agnostic in the API, specific in what it must
handle.

**Built:** `argmind/stream.py` (fixed-capacity ring buffers, counted drops, monotonic time
enforced at the source) and `argmind/align.py` (four explicit alignment policies, signed
staleness, causal/non-causal declared). 16 tests pass; several exist to prove failures are
visible rather than silent.

**Next, in order** — the ordering is deliberate and the reasoning is in DESIGN.md:

1. `Adapter`, starting with recursive least squares with a forgetting factor. Python reference
   first; it sets the contract every later method must fit.
2. **`Guard` second, not fifth.** Half the reason this library exists is that online adaptation
   diverges in the field, so the freeze path must be the default shape of the API rather than
   something a user remembers to add. Building it late would quietly make it optional.
3. `Pipeline` — stream to align to adapt to predict, behind one bounded `step()`.
4. Persistence, because adaptation state that does not survive a power cycle is useless in the
   field.

**Before any optimisation lands, a benchmark that can report a regression must exist.** An
optimisation with no way to fail is not an optimisation, it is a hope.

**One question still open:** whether the target range reaches down to MCUs. It decides the matrix
types (own minimal types vs Eigen) and the `-mcpu` set. It does not block the Python reference.

## Working state

No git remote is configured — deliberately, so a stray `push` cannot write to the retired Google
Drive location. Add one when publishing.
