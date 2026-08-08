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

Windows, PowerShell primary; Python 3.14; git repo. `python -m pytest` before quoting any
number. A tracked pre-commit hook in `.githooks/` runs the suite and blocks on failure —
install once per clone with `git config core.hooksPath .githooks`.

## Status

**Requirements are not yet settled.** Scope, target platform, sensor set and what "learning"
means on-device are open. Do not start building until they are — the design space is wide enough
that guessing wastes more than asking.
