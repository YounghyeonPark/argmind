# SPDX-License-Identifier: Apache-2.0
"""What argmind is not, measured — asked by the project owner on 2026-08-16.

The question was `인지적 지능기능을 가진 두뇌를 만드는게 맞는지 확인바람` — is this actually
building a brain with cognitive-intelligence functions. The answer is **no, not as it stands**, and
this file is that answer protected by assertions rather than left in prose, because this repository's
own rule is that a negative result gets a test so a later change cannot quietly erase it.

Three claims, each of which can fail:

1. **What learns is one linear map.** The head satisfies superposition exactly; the network is the
   control that does *not*, so the check discriminates rather than passing on everything.
2. **The demo's perception is hand-written and argmind learns none of it.** The named algorithms —
   k-means segmentation, connected components, ring contrast, appearance matching, region tracking —
   all live in `demo/`, and `argmind/` imports nothing from there. That is the 2026-08-14 decision's
   *starting* state, so when this test fails the decision is being executed.
3. **The faculties a cognitive brain is usually taken to need are absent** — a goal the system holds
   for itself, action selection, episodic memory, attention, concept formation, language, audio, a
   self-model. Not unfinished: absent.

**A keyword grep reported four of those as PRESENT and was wrong on every one**, which is recorded
here because it is a broken measurement that flattered the answer, and this project treats that as
the most important kind to catch. `policy` was the aligner's synchronisation policy, `action` was a
substring of `subtraction`, `mic ` of `dynamic`, `language` of "the language decision in DESIGN.md".
`test_the_keyword_search_that_got_this_wrong_still_gets_it_wrong` pins the trap.

What is *not* in question, and is deliberately not argued down here: the online adaptation, the
guard, persistence, any-to-any shapes, cold start with nothing pretrained, and execution on the
declared floor are measured and hold. This file is about the gap between that and the stated purpose
of 2026-08-13, not about whether the built part works.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from argmind.adapt import MultilayerPerceptron, RecursiveLeastSquares

ROOT = Path(__file__).resolve().parent.parent

#: The perception algorithms the demo runs. Every one is classical, hand-written, and in `demo/`.
#: argmind learns none of them — that is what the 2026-08-14 decision is aimed at.
CLASSICAL = ("FrameSegmenter", "connected_regions", "split_regions", "ring_contrast",
             "AppearanceMatcher", "RegionTracker", "BackgroundLocator", "ObjectMemory")


def _trained(adapter, n=6, steps=200, seed=0):
    g = np.random.default_rng(seed)
    truth = g.standard_normal((n + 1, 1)) * 0.5
    for _ in range(steps):
        x = np.concatenate([g.standard_normal(n), [1.0]])
        adapter.update(x, x @ truth)
    return adapter


def test_what_learns_is_one_linear_map_and_the_network_is_the_control():
    """**Claim 1.** `RecursiveLeastSquares.predict` is `x @ theta` — so a prediction on a blend of
    two feature vectors is the same blend of the two predictions, exactly.

    The network is here as the control. If it satisfied superposition too, this check would be
    passing on everything rather than measuring anything — the same reason every fixture in this
    repository carries a case that must *not* hold.
    """
    rng = np.random.default_rng(1)
    x1 = np.concatenate([rng.standard_normal(6), [1.0]])
    x2 = np.concatenate([rng.standard_normal(6), [1.0]])
    a = 0.3

    rls = _trained(RecursiveLeastSquares(7, 1, dtype="f8"))
    blend = float(rls.predict(a * x1 + (1 - a) * x2)[0])
    parts = a * float(rls.predict(x1)[0]) + (1 - a) * float(rls.predict(x2)[0])
    assert blend == pytest.approx(parts, rel=1e-12), (
        f"the head stopped being linear in its features: {blend} vs {parts}")

    # **Untrained on purpose.** The claim is about the *form* of the map, which the activation
    # settles at initialisation — training is irrelevant to it and made the check brittle: forcing
    # `relu` (a legitimate activation) diverged at this rate, tripped the finiteness assertion, and
    # the test failed on a change it has no business failing on.
    mlp = MultilayerPerceptron(7, 1, hidden=(8,), dtype="f8", seed=3)
    m_blend = float(mlp.predict(a * x1 + (1 - a) * x2)[0])
    m_parts = a * float(mlp.predict(x1)[0]) + (1 - a) * float(mlp.predict(x2)[0])
    # **Finiteness first, and it is not defensive tidying.** A mutant replacing the activation with
    # the identity made the network diverge to NaN, and `nan != approx(nan)` is *True* — so the
    # control below held while measuring nothing, and the mutant survived. The control has to be
    # able to fail before it can be a control.
    assert np.isfinite(m_blend) and np.isfinite(m_parts), (
        f"the network's predictions are not finite ({m_blend}, {m_parts}); the inequality below "
        "would hold vacuously, which is how this check was silently disabled once already")
    assert m_blend != pytest.approx(m_parts, rel=1e-6), (
        "the network satisfies superposition too, so this check discriminates nothing — its hidden "
        "units have gone linear")


def test_the_demo_perception_is_hand_written_and_the_library_learns_none_of_it():
    """**Claim 2, and it is the one the 2026-08-14 decision is aimed at.**

    Every perception algorithm the demo runs is classical code in `demo/`. argmind consumes the
    labels those algorithms produce; it does not produce them. Measured rather than described: the
    named classes are found in `demo/` and **not** in `argmind/`, and no library module imports the
    demo.

    **When this test fails, that is the decision being executed rather than a regression** — the
    message says so, because a test whose failure is progress must not read as a break.
    """
    lib = "\n".join(p.read_text(encoding="utf-8", errors="replace")
                    for p in sorted((ROOT / "argmind").glob("*.py")))
    labeller = (ROOT / "demo" / "label_sources.py").read_text(encoding="utf-8", errors="replace")

    for name in CLASSICAL:
        assert f"class {name}" in labeller or f"def {name}" in labeller, (
            f"{name} is no longer in demo/label_sources.py; if it moved into argmind/, this test is "
            "the record of that and should be updated deliberately")
        assert f"class {name}" not in lib and f"def {name}" not in lib, (
            f"{name} is now implemented inside argmind/ — the library has taken over a piece of "
            "perception it used to consume. That is the 2026-08-14 decision, not a regression: "
            "update this test and CLAUDE.md's 'what this is not' section together")

    for p in sorted((ROOT / "argmind").glob("*.py")):
        text = p.read_text(encoding="utf-8", errors="replace")
        for ln in text.splitlines():
            s = ln.strip()
            assert not (s.startswith("import demo") or s.startswith("from demo")), (
                f"{p.name} imports the demo; the dependency direction that keeps OpenCV out of the "
                "library has been reversed")


def test_the_faculties_a_cognitive_brain_needs_are_absent_from_the_library():
    """**Claim 3.** Eight faculties, none of them present. Checked against the library's *public
    API* rather than against a text search, because a text search is what got this wrong.

    Absent is not the same as unfinished, and the distinction is the whole point of the test: a
    reader of CLAUDE.md's stated purpose could reasonably assume these are in progress somewhere.
    """
    import argmind

    exported = {n.lower() for n in dir(argmind)}
    for module in sorted((ROOT / "argmind").glob("*.py")):
        exported |= {module.stem.lower()}

    forbidden = {
        "a goal the system holds for itself": ("goal", "objective", "drive", "motivation"),
        "action selection or planning":       ("planner", "plan", "act", "actuator"),
        "episodic or working memory":         ("episodic", "memory", "recollect"),
        "attention":                          ("attention", "attend", "salience"),
        "concept formation":                  ("concept", "abstraction", "symbol"),
        "language":                           ("language", "speech", "tts", "asr"),
        "audio input":                        ("audio", "microphone", "spectrogram"),
        "a self-model":                       ("selfmodel", "introspection", "metacognition"),
    }
    present = {faculty: sorted(n for n in names if n in exported)
               for faculty, names in forbidden.items()}
    found = {k: v for k, v in present.items() if v}
    assert not found, (
        f"argmind now exposes {found}. If a faculty has genuinely been built, this test is where "
        "the 'what this is not' claim was recorded and both should move together")


def test_the_keyword_search_that_got_this_wrong_still_gets_it_wrong():
    """**The broken measurement, pinned so it is not repeated.**

    Grepping the sources for `action`, `mic `, `language` and `policy` reported four faculties
    PRESENT. Every hit was a false positive, and the failure mode is worth keeping executable
    because it is the shape this repository keeps finding: a measurement that agrees with the
    answer you were hoping for.
    """
    traps = {
        "action": "subtraction",                # argmind/adapt.py, the rank-1 subtraction
        "mic ": "dynamic range",                # argmind/adapt.py, fp16's dynamic range
        "language": "the language decision",    # argmind/bench.py, C++ vs Python
    }
    for needle, host in traps.items():
        assert needle in host, (
            f"'{needle}' no longer matches inside '{host}', so the substring trap this test records "
            "has changed and the grep result it explains needs re-deriving")

    align = (ROOT / "argmind" / "align.py").read_text(encoding="utf-8", errors="replace")
    assert "policy" in align.lower(), "align.py no longer says 'policy'"
    assert "hold_last" in align, (
        "align.py's 'policy' is no longer the synchronisation policy — the word may now mean action "
        "selection, which is exactly the reading the false positive produced")


def test_depth_is_not_a_constraint_and_the_contract_never_limited_it():
    """**Re-audit of 2026-08-16.** I had been reporting depth as constrained. Measured, it is not:
    a 16-hidden-layer network is one contiguous state block and goes through the `Guard` unchanged.

    This is the *withdrawal* half of the constraint audit, and it gets a test for the same reason
    every withdrawn claim here does — so the constraint is not quietly reasserted later.
    """
    from argmind.guard import Bound, Guard

    deep = MultilayerPerceptron(64, 16, hidden=(64,) * 16, dtype="f8")
    assert deep.state.ndim == 1 and deep.state.flags["C_CONTIGUOUS"], (
        "a deep network no longer keeps one flat contiguous state block, which is what makes "
        "persistence a copy and the C ABI need no serialisation")
    assert deep.state.size > 60000

    guard = Guard(MultilayerPerceptron(20, 5, hidden=(64,) * 6, dtype="f8"),
                  [Bound("residual_norm", hi=1e6)], rollback_depth=4, checkpoint_every=10)
    rng = np.random.default_rng(0)
    for _ in range(50):
        guard.update(rng.standard_normal(20), rng.standard_normal(5))
    assert not guard.frozen, "a six-layer network no longer survives the guard it is wrapped in"


def test_the_real_constraint_is_that_the_contract_has_no_term_for_a_GRADIENT():
    """**The one constraint that survived the audit, pinned so it is not confused with a size limit.**

    `Adapter.update(x, y)` needs a *target*, and `UpdateReport` carries no gradient. A monolithic deep
    learner needs neither — it owns all its layers, which is why depth is free above. A hierarchy of
    separately-guarded levels needs both: a hidden level has no `y` to be given, and gets back nothing
    a level below it could descend on.

    When the contract grows that term, this test is where the old shape was recorded.
    """
    from argmind.adapt import Adapter, UpdateReport

    fields = set(UpdateReport.__dataclass_fields__)
    assert not (fields & {"gradient", "grad", "d_input", "backward"}), (
        f"UpdateReport now carries a gradient term ({fields}); the hierarchy constraint recorded on "
        "2026-08-16 has been lifted and CLAUDE.md's audit should move with it")

    import inspect
    assert list(inspect.signature(Adapter.update).parameters)[1:3] == ["x", "y"], (
        "Adapter.update no longer takes a supervised pair; the 'a hidden level has no y' finding "
        "needs re-deriving")

    from argmind.backbone import Backbone
    bb = Backbone([("conv", 12, 5, 4), ("dw", 3, 2), ("pw", 16)], (32, 32, 3), pool=2)
    absent = [m for m in ("update", "predict", "reset", "health_fields") if not hasattr(bb, m)]
    assert absent == ["update", "predict", "reset", "health_fields"], (
        f"the conv stack has gained {set(('update','predict','reset','health_fields')) - set(absent)}"
        " — it is becoming a first-class learner, which is the 2026-08-16 decision being executed "
        "rather than a regression. Update this test and CLAUDE.md's audit together")
    assert hasattr(bb, "backward") and hasattr(bb, "apply_gradients"), (
        "the backbone lost its gradients; learning the features is no longer expressible at all")
