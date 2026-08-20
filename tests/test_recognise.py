# SPDX-License-Identifier: Apache-2.0
"""The five measured pieces, wired together: does a returning object get its old label back?

That question is what the project owner proposed on 2026-08-19 — *"장기기억을 통해서 각 라벨이 갖는
고유의 특징을 찾고, 그걸 기억으로 해서 다음에 같은 특징이 보이면 동일 라벨로 인식"*. Each piece was
measured separately and each needs no representation learning. `RecognitionLayer` composes them and
this file drives the composition end to end.

Measured, over a sequence of appear → absent → a different object → the same object returns:

| candidate grid | slots for the first appearance | on return | same label |
|---|---|---|---|
| step 6, 117 candidates | 0 | 0 **and 2** | no |
| **step 2, 925 candidates** | **0** | **0** | **yes** |

**The cost of the coarse grid is crop centring, not the idea.** Measured directly: the best match
score available anywhere on the grid is 0.8835 at steps 12, 6 and 3, and **0.9990 at step 1** — so a
coarse grid leaves the score sitting just above the 0.85 threshold, where frame noise flips it. The
descriptor is sensitive to where the crop is centred, which is unsurprising given every recognition
figure here was measured on crops centred on the object by construction.

**And the admission gate is load-bearing, which was found by it being absent.** The first version
admitted the rarest crop of every frame. On an empty frame that is a patch of *wall*, and every wall
crop then matches it forever: measured, an object recognised as slot 0 came back as slot 1 after a
three-frame absence, with the layer looking at (0.81, 0.58) and (0.88, 0.83) — background. The
failure was predicted when the design was proposed and the gate was left out anyway.

The gate is a margin on rarity, and it works because the separation is not marginal: `median - min`
of the candidates' mean similarity is **0.0007–0.0011** on an empty wall and **0.126–0.143** with an
object present.
"""

from __future__ import annotations

import numpy as np
import pytest

from argmind.backbone import Backbone
from argmind.memory import ExemplarMemory
from argmind.recognise import RecognitionLayer
from tests.test_bootstrap import H, W, render
from tests.test_recognition_memory import CROP, PLAN, POOL


def grid(step):
    return [(x / W, y / H) for y in range(12, H - 11, step) for x in range(12, W - 11, step)]


def scene(phase, k):
    if phase == "A":
        return render([("disc", 0.30 + 0.02 * (k % 3), 0.40)], seed=100 + k, texture="none")
    if phase == "gone":
        return render([], seed=200 + k, texture="none")
    if phase == "B":
        return render([("bar", 0.70, 0.62)], seed=300 + k, texture="none")
    return render([("disc", 0.34 - 0.02 * (k % 3), 0.44)], seed=400 + k, texture="none")


def layer(step=2, *, admit_margin=0.02, threshold=0.85, capacity=8, seed=23):
    bb = Backbone(PLAN, (CROP, CROP, 3), dtype="f8", seed=seed, pool=POOL)
    mem = ExemplarMemory(capacity, bb.n_outputs, 16, threshold=threshold)
    return RecognitionLayer(bb, mem, (H, W, 3), grid(step), admit_margin=admit_margin)


def drive(L):
    seq = []
    for phase, n in (("A", 3), ("gone", 2), ("B", 2), ("again", 3)):
        for k in range(n):
            L.features(scene(phase, k))
            seq.append((phase, L.last_slot, L.last_recognised))
    return seq


def test_a_returning_object_gets_its_OLD_label_back():
    """The claim the whole proposal was made for."""
    seq = drive(layer(step=2))
    first = {s for p, s, _r in seq if p == "A" and s >= 0}
    back = {s for p, s, _r in seq if p == "again" and s >= 0}
    assert first, "the first appearance was never stored"
    assert back <= first, (
        f"the object first took slots {sorted(first)} and came back as {sorted(back)} — identity "
        "across an absence is what this composition exists to provide")


def test_an_EMPTY_frame_stores_nothing_and_says_so():
    """`-1` rather than silently reusing the last slot: "there was nothing here" is a real answer,
    and admitting the background is what broke the first version."""
    L = layer(step=2)
    seq = drive(L)
    gone = [s for p, s, _r in seq if p == "gone"]
    assert all(s == -1 for s in gone), f"an empty frame produced slots {gone}"
    assert L.memory.occupied <= 2, (
        f"{L.memory.occupied} exemplars stored for two objects; the gate is admitting background")


def test_WITHOUT_the_gate_the_background_becomes_an_exemplar_and_wins_forever():
    """The control that shows the gate is load-bearing rather than decorative. A margin small enough
    to admit anything reproduces the original failure exactly."""
    seq = drive(layer(step=2, admit_margin=1e-9))
    gone = [s for p, s, _r in seq if p == "gone"]
    assert any(s >= 0 for s in gone), (
        "even with the gate open nothing was admitted on an empty frame; this control no longer "
        "reproduces the failure it exists to demonstrate")
    first = {s for p, s, _r in seq if p == "A" and s >= 0}
    back = {s for p, s, _r in seq if p == "again" and s >= 0}
    assert not (back and back <= first), (
        "identity survived with the gate open, so the gate is not what fixed it and this file's "
        "explanation should be re-derived")


def test_a_COARSE_grid_loses_the_identity_and_the_reason_is_crop_centring():
    """Recorded as a measured cost, not hidden. The score is on the decision boundary at a coarse
    grid — 0.8835 available against a 0.85 threshold — so noise decides."""
    seq = drive(layer(step=6))
    first = {s for p, s, _r in seq if p == "A" and s >= 0}
    back = {s for p, s, _r in seq if p == "again" and s >= 0}
    assert not back <= first, (
        f"the coarse grid now keeps identity ({sorted(first)} → {sorted(back)}); if that became "
        "true the centring explanation is wrong and the fine grid is no longer needed")


def test_the_admission_margin_separates_an_empty_frame_from_one_worth_remembering():
    """The measurement the gate's threshold rests on, asserted rather than quoted."""
    L = layer(step=6)
    def spread(img):
        d = L._describe(img)
        S = d @ d.T
        ms = (S.sum(axis=1) - np.diag(S)) / (len(L.centres) - 1)
        return float(np.median(ms) - ms.min())
    empty, occupied = spread(scene("gone", 0)), spread(scene("A", 0))
    assert empty < 0.01 < occupied, (
        f"empty {empty:.4f} against occupied {occupied:.4f}; the gate needs these two orders of "
        "magnitude apart and they no longer are")


def test_a_non_positive_margin_is_refused():
    with pytest.raises(ValueError, match="admit_margin must be positive"):
        layer(admit_margin=0.0)


def test_it_refuses_a_memory_that_cannot_hold_the_backbone_s_descriptor():
    bb = Backbone(PLAN, (CROP, CROP, 3), dtype="f8", seed=0, pool=POOL)
    with pytest.raises(ValueError, match="must match"):
        RecognitionLayer(bb, ExemplarMemory(4, bb.n_outputs + 1, 8, threshold=0.8),
                         (H, W, 3), grid(12), admit_margin=0.02)


def counted(L):
    """Wrap `_describe` so a test can state the cost as a number instead of asserting it is small."""
    L._n = 0
    inner = L._describe

    def _describe(img):
        out = inner(img)
        L._n += len(L.centres)
        return out
    L._describe = _describe
    return L


def test_local_REFINEMENT_keeps_the_identity_at_a_sixth_of_the_crops():
    """**The cost of a fine grid, paid down.** A coarse pass picks the region and a fine pass around
    the winner decides there:

    | variant | crops per frame | identity across the absence |
    |---|---|---|
    | plain step 2 | 925 | yes |
    | plain step 8 | 70 | **no** |
    | **step 8 + refine(8, 2)** | **147** | **yes** |
    | step 12 + refine(12, 2) | 188 | yes |

    6.3× fewer crops than the fine grid, for the same answer.
    """
    L = counted(layer(step=8))
    L.refine = (8, 2)
    seq = drive(L)
    first = {s for p, s, _r in seq if p == "A" and s >= 0}
    back = {s for p, s, _r in seq if p == "again" and s >= 0}
    assert back and back <= first, (
        f"refinement lost the identity: {sorted(first)} → {sorted(back)}")
    per_frame = L._n / 10
    assert per_frame < 200, f"{per_frame:.0f} crops per frame; the point of refining is to spend less"
    assert per_frame > len(L.centres), "the fine pass never ran — only the coarse grid was counted"


def test_refinement_must_happen_BEFORE_the_memory_is_consulted():
    """**A design error caught by measurement, recorded so it is not reintroduced.**

    The first version refined the centre *after* the slot had been assigned. It changed the reported
    centre and nothing else, and the identity results were byte-identical to no refinement at all —
    70 crops, slots [0, 1] → [0, 3], still wrong. The point of a better centre is to look *that*
    descriptor up.

    Asserted structurally: the refined pass must change which descriptor is handed to the memory, so
    a refining layer and a plain one must disagree about what they attended to on the same frame.
    """
    plain, fine = layer(step=8), layer(step=8)
    fine.refine = (8, 2)
    img = scene("A", 0)
    plain.features(img)
    fine.features(img)
    assert fine.last_centre != plain.last_centre, (
        "the refining layer attended to exactly the coarse grid point; refinement is not changing "
        "what gets looked up")

    # **And it must be LOCAL to the coarse choice.** A mutant refining around a fixed frame centre
    # survived the identity test, because on this fixture a neighbourhood of the middle still
    # overlaps the object — the fixture is insensitive to it, so the property is pinned directly.
    span_x = 8 / W
    span_y = 8 / H
    assert abs(fine.last_centre[0] - plain.last_centre[0]) <= span_x + 1e-9, (
        f"the refined centre {fine.last_centre} is outside the neighbourhood of the coarse choice "
        f"{plain.last_centre}; refinement is searching somewhere it was not asked to")
    assert abs(fine.last_centre[1] - plain.last_centre[1]) <= span_y + 1e-9


def test_a_malformed_refine_specification_is_refused():
    with pytest.raises(ValueError, match=r"\(span, step\)"):
        layer(step=8).__class__(
            Backbone(PLAN, (CROP, CROP, 3), dtype="f8", seed=0, pool=POOL),
            ExemplarMemory(4, 108, 16, threshold=0.8), (H, W, 3), grid(12),
            admit_margin=0.02, refine=(0, 2))
