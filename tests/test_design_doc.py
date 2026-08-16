# SPDX-License-Identifier: Apache-2.0
"""`DESIGN.md`'s claims about the repository, checked against the repository.

**Why.** An audit on 2026-08-16 found the file carrying a Python usage example that **did not run** —
written before `FrameMap` existed, so it omitted a required argument. That is not a stale number; it
is a sample demonstrating an API the library does not have, in the document a reader goes to first.
Same family as `SIZES.md` instructing readers to use a `--regen-sizes` option that did not exist,
and as `select_newest` being compiled and called by no binary.

Two claims are pinned here, both of which go stale by someone doing good work:

1. **The adaptation-methods list says which of the four are built.** Building an EKF should fail this
   test, because the list would then be wrong — and the failure message says so.
2. **No `python` code fence sits in the file un-executed.** The fix for the broken sample was to
   remove it and point at examples that are *run* by the suite; this keeps a new one from drifting
   in unchecked.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

DOC = Path(__file__).resolve().parent.parent / "DESIGN.md"


@pytest.fixture(scope="module")
def text():
    return DOC.read_text(encoding="utf-8", errors="replace")


def test_the_adaptation_method_list_reports_what_is_actually_built(text):
    """The list is a plan, and the plan did not hold — 3 was built before 2. That is the reason to
    keep it, so the labels have to stay true."""
    import argmind.adapt as adapt
    import argmind.guard as guard

    built = {n for n in dir(adapt) if n[0].isupper()}
    section = text.split("## Adaptation methods")[1].split("## Open questions")[0]

    assert "RecursiveLeastSquares" in built and "**BUILT**" in section
    assert {"NormalisedLeastMeanSquares", "MultilayerPerceptron"} <= built, (
        "the gradient methods are gone; item 3's BUILT label is now wrong")
    assert "**BUILT, out of order**" in section, (
        "the list no longer records that item 3 arrived before item 2 — which is the only thing a "
        "kept plan is evidence of")

    kalman = {n for n in built if "Kalman" in n or n.startswith("Ekf") or n.startswith("EKF")}
    assert not kalman, (
        f"a Kalman-family adapter exists now ({kalman}) and DESIGN.md still says item 2 is NOT "
        "BUILT. That is the decision being executed, not a regression: update the list.")
    assert "**NOT BUILT.**" in section

    # Item 4 is the subtle one: `Divergence` looks like drift detection and is not.
    assert hasattr(guard, "Divergence"), "Divergence is gone; item 4's wording needs re-deriving"
    assert "is a guard *check*" in section, (
        "the list no longer distinguishes Divergence from a drift detector that decides *when* to "
        "adapt — they are easy to confuse and the note is what prevents it")


def test_no_python_example_sits_in_the_document_unexecuted(text):
    """The removed sample did not run. Rather than fix it in place — where it would drift again the
    next time the API moved — the document points at examples the suite *executes*.

    A `python` fence appearing here is not forbidden in principle; it is forbidden **silently**. If
    one is added, run it in this test and delete this assertion.
    """
    fences = re.findall(r"```(\w*)", text)
    assert "python" not in fences, (
        "DESIGN.md has grown a python code fence again. The last one predated FrameMap and was "
        "missing a required argument. Either execute it from this test, or point at "
        "tests/test_packaging.py and tests/test_pipeline.py, which are run.")

    for pointer in ("tests/test_packaging.py", "tests/test_pipeline.py"):
        assert pointer in text, f"the document no longer points at {pointer}, which is executed"


def test_the_standards_section_does_not_re_state_CLAUDE_md(text):
    """It used to be seven lines whose body was 'See CLAUDE.md'. What survives is the one rule that
    is specific to this level, and the duplication is gone — checked, because duplication grows
    back."""
    section = text.split("## Standards")[1]
    assert len(section.strip().splitlines()) < 12, (
        "the Standards section is growing again; the evidence rules live in CLAUDE.md")
    assert "anything that can report only success is not evidence" in section
    claude = (DOC.parent / "CLAUDE.md").read_text(encoding="utf-8", errors="replace")
    assert "Standards of evidence" in claude, (
        "CLAUDE.md no longer carries the evidence rules, so DESIGN.md deferring to it is now a "
        "dangling pointer")
