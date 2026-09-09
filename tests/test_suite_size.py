# SPDX-License-Identifier: Apache-2.0
"""The Python suite's own size, floored.

**Why a floor and not an equality, and why anything at all.** The C++ check count is pinned
exactly (`tests/test_cpp_parity.py`) because it is quoted in prose and prose is protected by
nothing. The Python count cannot be: a suite gains tests constantly and an equality would fail
on every commit that adds one.

CLAUDE.md said as much on 2026-08-25 and then concluded that the Python figure should carry no
protection at all -- **an argument an independent audit correctly called half honest.** The
commit had just finished proving that a `>=` floor is blind to a count *rising*, and used that
to justify having nothing; but the direction a floor *does* catch is a count **falling**, which
is deletion, and that is the failure worth catching here. A floor costs nothing.

So: this asserts the collected count has not fallen. It says nothing about the exact figure, and
the figure in CLAUDE.md remains explicitly approximate and explicitly re-derivable.
"""

from __future__ import annotations

import pytest

#: The floor, not the count. Raise it deliberately when the suite has grown enough that the gap
#: stops meaning anything -- and never lower it to make a deletion pass.
#:
#: **Lowered twice on 2026-09-09, and this is the case the comment above warns about**, so it says
#: why rather than merely doing it. Step 5 of `REBUILD.md` removes the old tree, after step 4
#: measured the rebuilt core's choosing accuracy, and it goes in stages:
#:
#:     1,252  before                    the whole tree
#:       743  1150 -> 700   the C++ port, the target build, the QEMU runs, every parity module
#:       499   700 -> 460   demo/, agent/, and the architecture document they were checked against
#:
#: The floor is not being fitted to what is left. Each value sits ~40 below the count that follows
#: it, which is the margin 1150 held against 1191, so an unplanned module vanishing still trips it.
COLLECTED_FLOOR = 460


def test_the_suite_has_not_SHRUNK(request):
    """A deleted or accidentally-unimported test module is silent otherwise: `pytest` reports
    what it collected and nothing compares that to what it collected last time.

    This is the one direction a floor is good for. It is deliberately loose -- the point is to
    catch a module vanishing, not to track the count.

    **It has to skip on a subset run, and the first version of this test did not** -- it read
    `session.testscollected`, which is what THIS invocation collected, so `pytest tests/foo.py`
    failed it immediately. That is the `-k` trap this repository has already recorded once:
    a check whose meaning depends on the whole suite being present must say so rather than fire
    on a partial run. Skipped, not passed, so the difference stays visible.
    """
    cfg = request.config
    if cfg.option.keyword or cfg.option.markexpr:
        pytest.skip("-k/-m selects a subset; this floor is a claim about the whole suite")
    if list(cfg.args) != list(cfg.getini("testpaths")):
        pytest.skip(f"invoked on {list(cfg.args)} rather than the configured testpaths; "
                    "this floor is a claim about the whole suite")

    collected = request.session.testscollected
    assert collected >= COLLECTED_FLOOR, (
        f"the suite collected {collected} tests, below the floor of {COLLECTED_FLOOR}. Tests "
        "have been deleted or a module has stopped being imported. If the removal was "
        "deliberate, lower this floor in the same commit and say why; do not lower it to get "
        "green.")
