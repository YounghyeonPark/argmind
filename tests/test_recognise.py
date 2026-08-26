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

    **That saving is for ONE reported slot, and the demo does not stay there** — measured
    2026-08-25 and qualified here rather than left standing. Refinement describes a neighbourhood
    per reported slot and the neighbourhoods overlap, so its union grows into the fine grid it was
    invented to replace: **7% of the step-2 grid at one slot, 45% at eight, 72% at sixteen, 95% at
    thirty-three**, and full coverage needs about fifty. See
    `test_REFINEMENT_DEGENERATES_into_the_fine_grid_it_was_invented_to_replace` below, whose
    docstring carries the assumption those percentages depend on — **they are computed for slots
    placed UNIFORMLY at random, and real reported slots cluster on objects.**
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


def test_the_gate_fires_on_every_REAL_frame_and_no_fixture_here_can_show_it():
    """**What the admission gate actually distinguishes, and the fixture that cannot see it.**

    Measured on a real camera scene, 40 frames: `median - min` of the candidates' mean similarity is
    **0.235-0.306** — above the 0.02 gate on **100%** of frames, and above even the synthetic
    "object present" range of 0.126-0.143. The demo showed `memory 13/16` after ~1,900 frames as a
    direct consequence.

    **The threshold is not wrong.** It does exactly what it was measured to do. The *design
    assumption* is wrong: it treats "the frame is featureless" as "there is nothing worth
    remembering", and a real room is never featureless. A chair, a bookshelf, a window and a face are
    all genuinely unlike their surroundings, so rarity fires every frame — correctly.

    **And no fixture in this repository can demonstrate that**, which is the point of this test.
    Measured below: synthetic clutter reads 0.0070 and stripes 0.0092, both *under* the gate, against
    a real scene's 0.235. The textured backgrounds here are far too uniform to stand in for a room —
    they become background statistically, which `tests/test_bootstrap.py` already records as an
    untested limit. So the real-camera figure is a **recorded measurement with its conditions**, in
    the same class as this project's frame rates, and not something the suite protects.

    What admission needs is a criterion about the *object* — recurrence rather than momentary
    distinctness — and that is not built.
    """
    L = layer(step=6)

    def spread(img):
        d = L._describe(img)
        S = d @ d.T
        ms = (S.sum(axis=1) - np.diag(S)) / (len(L.centres) - 1)
        return float(np.median(ms) - ms.min())

    empty = spread(render([], seed=1, texture="none"))
    one = spread(render([("disc", 0.30, 0.40)], seed=1, texture="none"))
    clutter = spread(render([], seed=2, texture="clutter"))
    stripes = spread(render([], seed=3, texture="stripes"))

    assert empty < L.admit_margin, (
        f"an empty wall reads {empty:.4f}, at or above the {L.admit_margin} gate")
    assert one > L.admit_margin * 3, f"one object reads only {one:.4f}"

    # **The inadequacy, asserted.** If these ever rose above the gate, the fixture would at last be
    # able to reproduce the real-scene failure -- and this test's message should change with it.
    assert clutter < L.admit_margin and stripes < L.admit_margin, (
        f"synthetic clutter ({clutter:.4f}) or stripes ({stripes:.4f}) now trip the gate. That would "
        "mean a fixture here can finally show the real-camera failure of 0.235-0.306, which this "
        "test currently records as unreproducible -- update it deliberately")
    assert clutter < one / 10, (
        f"clutter ({clutter:.4f}) is within a factor of ten of a real object ({one:.4f}); the claim "
        "that these backgrounds are too uniform to stand in for a room would then be weaker")




def _presence_layer(step=4, capacity=8, threshold=0.85, seed=23, refine=None):
    from argmind.backbone import Backbone
    bb = Backbone(PLAN, (CROP, CROP, 3), dtype="f8", seed=seed, pool=POOL)
    mem = ExemplarMemory(capacity, bb.n_outputs, 16, threshold=threshold)
    return RecognitionLayer(bb, mem, (H, W, 3), grid(step), admit_margin=0.02, emit="presence",
                            refine=refine)


def test_MORE_THAN_ONE_object_is_recognised_in_the_SAME_frame():
    """**The owner's objection, answered.** *"인식을 한번에 한개씩만 함? … 여러개를 동시에 인지해야
    하는거 아닌가?"* — correct, and the serial version was not a principle.

    `features()` must emit a fixed-width vector for the head above, and one descriptor is fixed
    width, so an implementation constraint was allowed to decide the architecture. `emit="presence"`
    returns one score per memory slot instead — parallel by construction, and exactly what
    `ExemplarMemory.n_outputs = capacity` had already anticipated.

    Measured: a disc at (0.30, 0.40) and a bar at (0.70, 0.62) are both reported, at (0.33, 0.44)
    and (0.71, 0.61), from the third pass onward.
    """
    L = _presence_layer()
    assert L.n_outputs == L.memory.capacity, (
        "presence mode must emit one score per slot; that is what makes it fixed width")
    img = render([("disc", 0.30, 0.40), ("bar", 0.70, 0.62)], seed=1, texture="none")
    for _ in range(4):
        scores = L.features(img)

    assert len(L.last_present) >= 2, (
        f"only {len(L.last_present)} object(s) reported with two in the frame — recognition is "
        "still serial")
    assert int((np.asarray(scores) > 0).sum()) >= 2, (
        "the emitted vector marks fewer than two slots present, so a head above it cannot learn "
        "about both")
    seen = {s for s, _sc, _c in L.last_present}
    assert len(seen) >= 2, f"two reports but only slots {seen}"

    # **Both must be RECOGNISED, not one recognised and one freshly admitted.** A mutant truncating
    # the presence list to its strongest entry survived until this existed: parallel admission then
    # refilled the list from the unexplained region, so the length recovered while the memory kept
    # growing. Occupancy holding steady is what distinguishes the two.
    settled = L.memory.occupied
    for _ in range(4):
        L.features(img)
        assert len(L.last_present) >= 2, "a later pass dropped back to one object"
    assert L.memory.occupied == settled == 2, (
        f"occupancy went {settled} -> {L.memory.occupied} on frames with nothing new in them; the "
        "system is re-admitting what it already knows")
    for _s, _sc, (cx, cy) in L.last_present:
        assert (abs(cx - 0.30) < 0.12 and abs(cy - 0.40) < 0.12) or \
               (abs(cx - 0.70) < 0.12 and abs(cy - 0.62) < 0.12), (
            f"a reported centre ({cx:.2f}, {cy:.2f}) is at neither object")


def test_the_DESCRIPTOR_mode_reports_at_most_one_and_that_is_the_contrast():
    """Either number alone says nothing; the difference on the same frame is the finding."""
    serial = layer(step=4)
    parallel = _presence_layer(step=4)
    img = render([("disc", 0.30, 0.40), ("bar", 0.70, 0.62)], seed=1, texture="none")
    for _ in range(4):
        serial.features(img)
        parallel.features(img)
    assert len(serial.last_present) <= 1, (
        f"descriptor mode reported {len(serial.last_present)} objects; if it became parallel too, "
        "the two modes no longer differ and this file's argument should change")
    assert len(parallel.last_present) > len(serial.last_present)


def test_ADMISSION_is_parallel_too_or_the_memory_cannot_grow_while_anything_is_known():
    """**The half that was still serial after the first fix**, and it is the deeper one.

    With a disc in memory and a disc *and* a bar in the frame, the bar was never admitted: the
    recognition path returned as soon as the disc matched. Recognising in parallel and admitting
    serially is still a one-object system. The rule is "explain the scene, then remember what is
    left over" — candidates near an already-matched slot are accounted for, and the rarity gate runs
    on the remainder.
    """
    L = _presence_layer()
    L.features(render([("disc", 0.30, 0.40)], seed=1, texture="none"))
    assert L.memory.occupied == 1, "the first object was not admitted"

    both = render([("disc", 0.30, 0.40), ("bar", 0.70, 0.62)], seed=1, texture="none")
    L.features(both)
    assert L.memory.occupied == 2, (
        f"the memory holds {L.memory.occupied} after a second object appeared beside a known one — "
        "admission is still gated on nothing being recognised")


def test_an_unknown_emit_mode_is_refused():
    from argmind.backbone import Backbone
    bb = Backbone(PLAN, (CROP, CROP, 3), dtype="f8", seed=0, pool=POOL)
    with pytest.raises(ValueError, match="emit must be"):
        RecognitionLayer(bb, ExemplarMemory(4, bb.n_outputs, 16, threshold=0.8), (H, W, 3),
                         grid(12), admit_margin=0.02, emit="both")


COLOURS = [(0.85, 0.25, 0.20), (0.20, 0.35, 0.85), (0.25, 0.80, 0.30),
           (0.85, 0.80, 0.20), (0.70, 0.25, 0.80)]
SPOTS = [(0.20, 0.30), (0.50, 0.30), (0.80, 0.30), (0.30, 0.72), (0.70, 0.72)]


def coloured(specs, seed):
    """Genuinely distinct objects. A disc/bar fixture looks like five objects and is **two
    identities** — the descriptor is shape-based, so the third disc matches the first disc's
    exemplar, correctly. That mistake made an earlier cap measurement meaningless."""
    yy, xx = np.mgrid[0:H, 0:W]
    img = np.full((H, W, 3), 0.35)
    for col, (cx, cy) in specs:
        img[np.hypot(xx - cx * W, yy - cy * H) <= 0.075 * W] = col
    g = np.random.default_rng(seed)
    return np.clip(img + g.standard_normal(img.shape) * 0.01, 0, 1).astype("f8")


def _seeded(cap=None, refine=(8, 2)):
    """Five distinct objects shown one at a time, then all together."""
    from argmind.backbone import Backbone
    bb = Backbone(PLAN, (CROP, CROP, 3), dtype="f8", seed=23, pool=POOL)
    L = RecognitionLayer(bb, ExemplarMemory(32, bb.n_outputs, 16, threshold=0.85),
                         (H, W, 3), grid(8), admit_margin=0.02, refine=refine,
                         emit="presence", max_reported=cap)
    for i, (col, spot) in enumerate(zip(COLOURS, SPOTS)):
        for k in range(2):
            L.features(coloured([(col, spot)], seed=100 * i + k))
    L.features(coloured(list(zip(COLOURS, SPOTS)), seed=0))
    return L


def test_REFINEMENT_ADDS_candidates_rather_than_replacing_them():
    """**A recall bug from composing two features that were each fine alone.**

    Refinement re-described the neighbourhood of the coarse choice and ran the presence pass on
    *that*, so a slot the coarse grid mis-centred badly enough to miss was never refined and never
    recovered. Measured on five distinct objects: **4 of 4 reported without refinement, 2 of 3 with
    it.** Concatenating instead restores it to 3 of 3, and is sound because presence takes the best
    score per slot — a coarse candidate can only lose to its own refinement.
    """
    fine, plain = _seeded(refine=(8, 2)), _seeded(refine=None)
    assert len(fine.last_present) >= min(3, fine.memory.occupied), (
        f"refinement reported {len(fine.last_present)} of {fine.memory.occupied} remembered "
        "objects; it is losing regions the coarse pass missed instead of adding to them")
    assert len(plain.last_present) >= 3, (
        f"the unrefined control reported only {len(plain.last_present)}; without it there is no "
        "baseline to say refinement lost anything")


def test_the_report_CAP_truncates_what_you_are_told_and_NOT_what_is_remembered():
    """**The setting, and the invariant that makes it safe.**

    A cap is a caller preference — a fixed output width for a head, a readable overlay — and not a
    cognitive claim: measured, it buys nothing, because the batched look-up reports every slot for
    the same cost. What it must never do is decide what counted as *present*, or a slot ranked fifth
    would lose a spell and the retention signal would be corrupted by a display setting.
    """
    unlimited = _seeded(cap=None)
    available = len(unlimited.last_present)
    assert available >= 2, f"the fixture only produced {available} recognitions; nothing to truncate"

    spells_seen = set()
    for cap in (1, 2, available):
        L = _seeded(cap=cap)
        assert len(L.last_present) == min(cap, len(L.last_present) or 1) <= cap, (
            f"cap {cap} reported {len(L.last_present)}")
        assert len(L.last_present) <= cap
        spells_seen.add(int((np.asarray(L.memory.spells) > 0).sum()))

    assert len(spells_seen) == 1, (
        f"the number of slots with spells varies with the report cap: {sorted(spells_seen)}. "
        "Truncation is reaching end_frame, so a display setting is deciding what the memory "
        "considers present")


def test_a_report_cap_below_one_is_refused():
    from argmind.backbone import Backbone
    bb = Backbone(PLAN, (CROP, CROP, 3), dtype="f8", seed=0, pool=POOL)
    with pytest.raises(ValueError, match="max_reported"):
        RecognitionLayer(bb, ExemplarMemory(4, bb.n_outputs, 16, threshold=0.8), (H, W, 3),
                         grid(12), admit_margin=0.02, emit="presence", max_reported=0)


def _moving_scene(k, n=10):
    """A static object and a moving one. The static one never changes; the mover crosses."""
    return coloured([(COLOURS[0], (0.25, 0.30)),
                     (COLOURS[1], (0.35 + 0.35 * (k % n) / n, 0.70))], seed=k)


def _change_layer(admit_change, step=8):
    from argmind.backbone import Backbone
    bb = Backbone(PLAN, (CROP, CROP, 3), dtype="f8", seed=23, pool=POOL)
    return RecognitionLayer(bb, ExemplarMemory(16, bb.n_outputs, 16, threshold=0.85),
                            (H, W, 3), grid(step), admit_margin=0.02, emit="presence",
                            admit_change=admit_change, change_rate=0.05)


def test_change_SELECTS_what_to_admit_it_does_not_merely_veto_it():
    """**A gate on a wrong choice cannot improve it, only refuse it.**

    A first version kept rarity as the chooser and added descriptor change as a gate on top. On real
    frames it admitted **nothing at any threshold** — 0 exemplars at 0.002, 0.005 and 0.01 against
    4–6 with the gate off — because rarity picks quiet, distinctive places (a patch of blank wall
    among other blank wall) and the change test then rejects exactly what rarity chose.

    With change as the *selector* the admissions happen and land on the thing that moved.
    """
    L = _change_layer(0.002)
    for k in range(12):
        L.features(_moving_scene(k))
    assert L.memory.occupied >= 1, (
        "nothing was admitted with change as the selector; the veto-only failure has returned")

    where = [c for _s, _sc, c in L.last_present]
    assert where, "nothing recognised after twelve frames"
    # The mover's band is y ~ 0.70; the static object sits at y ~ 0.30.
    assert any(abs(cy - 0.70) < 0.15 for _cx, cy in where), (
        f"reported centres {where} are all in the static object's band; change-based selection is "
        "not finding the thing that moves")


def test_the_change_signal_separates_a_STATIC_object_from_a_MOVING_one():
    """The measurement the selector rests on, on a fixture rather than on a camera.

    On real frames the separation is **348×** at `change_rate=0.05` — quietest ten locations 0.00005
    against busiest ten 0.01893. That figure needs a camera; this asserts the ordering, which does
    not.
    """
    L = _change_layer(0.002)
    for k in range(12):
        L.features(_moving_scene(k))

    ch = L._change
    assert ch is not None and len(ch) == len(L.centres)
    static = [i for i, (cx, cy) in enumerate(L.centres)
              if abs(cx - 0.25) < 0.10 and abs(cy - 0.30) < 0.10]
    mover = [i for i, (_cx, cy) in enumerate(L.centres) if abs(cy - 0.70) < 0.10]
    assert static and mover, "the fixture's bands do not land on the candidate grid"
    assert max(ch[i] for i in mover) > max(ch[i] for i in static) * 3, (
        f"the mover's band peaks at {max(ch[i] for i in mover):.5f} against the static object's "
        f"{max(ch[i] for i in static):.5f}; the signal the selector uses is not separating them")


def test_a_non_positive_change_threshold_or_rate_is_refused():
    from argmind.backbone import Backbone
    bb = Backbone(PLAN, (CROP, CROP, 3), dtype="f8", seed=0, pool=POOL)
    mem = lambda: ExemplarMemory(4, bb.n_outputs, 16, threshold=0.8)
    with pytest.raises(ValueError, match="admit_change"):
        RecognitionLayer(bb, mem(), (H, W, 3), grid(12), admit_margin=0.02, admit_change=0.0)
    with pytest.raises(ValueError, match="change_rate"):
        RecognitionLayer(bb, mem(), (H, W, 3), grid(12), admit_margin=0.02, change_rate=0.0)


def test_the_baseline_ADVANCES_so_a_thing_that_moved_once_stops_reading_as_changed():
    """A frozen baseline compares against frame zero forever, so a chair that was pushed once and
    then left alone keeps scoring as movement — and keeps earning admissions.

    An exponential baseline forgets: after the motion stops, change decays back toward zero. A mutant
    that stops advancing the baseline survived every other test here, because their fixtures keep
    moving and a frozen baseline reads that correctly.
    """
    L = _change_layer(0.002)
    for k in range(8):                                  # it moves
        L.features(_moving_scene(k))
    band = [i for i, (_cx, cy) in enumerate(L.centres) if abs(cy - 0.70) < 0.10]
    while_moving = max(L._change[i] for i in band)

    settled = _moving_scene(7)                          # then stops, in one place
    for _ in range(30):
        L.features(settled)
    after_stopping = max(L._change[i] for i in band)

    assert while_moving > 0.002, f"the fixture never registered movement: {while_moving:.5f}"
    assert after_stopping < while_moving / 5, (
        f"change stayed at {after_stopping:.5f} against {while_moving:.5f} while moving — the "
        "baseline is not advancing, so anything that ever moved keeps earning admissions forever")

def test_the_grid_AXES_place_every_painted_cell_where_the_layer_says_it_LOOKED():
    """`last_field` is an integer map with no geometry of its own, and `grid_x` / `grid_y` are the
    geometry. Without them the field can only be shown as a picture in the abstract — which is
    exactly what made it unreadable, since nothing tied a cell to a place in the room.

    The property an overlay on the camera frame depends on, and the only one that can turn it into a
    lie: a cell painted for a slot must sit within one candidate step of the centre the layer
    *reports* for that slot. If those disagreed the tint would contradict the box drawn beside it —
    the same unreadability the field was added to fix, arriving with more colour this time.

    Within one step and not exactly equal, deliberately: `_field` paints only the **coarse**
    candidates, so a slot whose best match was a refined position is reported up to the refinement
    span away from its cell.
    """
    L = _presence_layer()
    assert L.last_field is None, "a layer that has seen nothing must not claim a field"
    img = render([("disc", 0.30, 0.40), ("bar", 0.70, 0.62)], seed=1, texture="none")
    for _ in range(4):
        L.features(img)

    assert L.last_field.shape == (len(L.grid_y), len(L.grid_x)), (
        f"the field is {L.last_field.shape} and the axes are "
        f"{(len(L.grid_y), len(L.grid_x))}; a drawing indexes one by the other")
    step_x, step_y = L.grid_x[1] - L.grid_x[0], L.grid_y[1] - L.grid_y[0]
    assert L.last_present, "nothing was recognised, so the placement claim would be vacuous"

    checked = 0
    for slot, _score, (cx, cy) in L.last_present:
        cells = np.argwhere(L.last_field == slot + 1)
        if not len(cells):
            continue                      # its best match was a refined position; see the docstring
        checked += 1
        off = min(max(abs(L.grid_x[c] - cx) / step_x, abs(L.grid_y[r] - cy) / step_y)
                  for r, c in cells)
        assert off <= 1.0, (
            f"slot {slot} is reported at ({cx:.3f}, {cy:.3f}) but its nearest painted cell is "
            f"{off:.2f} candidate steps away — an overlay would tint one place and box another")
    assert checked, (
        "no reported slot had a painted cell, so nothing above was actually compared; the field "
        "and the presence report have stopped agreeing about anything")


def test_the_axes_are_the_SORTED_UNIQUE_candidate_coordinates():
    """Stated as the round trip, because "the axes" is only meaningful if a caller can go from a
    candidate to its cell and back. A duplicated or unsorted axis would still have the right
    *length* and would place every cell wrongly."""
    L = _presence_layer()
    assert list(L.grid_x) == sorted(set(L.grid_x)) and list(L.grid_y) == sorted(set(L.grid_y))
    assert set(L.centres) == {(x, y) for x in L.grid_x for y in L.grid_y}, (
        "the candidate set is not the product of the two axes, so a rectangular field cannot "
        "represent it and some cell in any drawing means nothing")

# --------------------------------------------------------------------------------------------
# What 2,005 frames of camera showed, and what it costs there -- 2026-08-24
#
# Every earlier capture here stopped at 60 frames and 9 exemplars while the owner's screen showed 88
# after 2,967, so the patchwork had been recorded as untested rather than explained. Run through
# `build_recogniser()`, so it cannot measure a different assembly than the demo runs:
#
#     2,005 frames, 781 s, 48 exemplars admitted, capacity 256
#     last 500 frames:  19.6 of 43.3 slots PRESENT every frame  -- 45% of the memory
#                       51.4 of 70 candidate cells claimed      -- 73% of the frame
#                       13.6 distinct slots painted in the map
#     cell coverage saturates by frame 250 (52 cells) while `occupied` goes on 20 -> 48
#
# **The memory is holding the room, not the objects in it.** After frame 250 a new exemplar buys no
# coverage, only another claimant for cells already claimed. That is the patchwork, and it is not one
# object fragmenting -- it is the whole scene being remembered.
#
# On admission, what did the memory already know? `look_up` on the descriptor about to be stored, 47
# admissions with a memory to compare against:
#
#     [0.00,0.50)  4%   [0.50,0.70) 19%   [0.70,0.80) 19%   [0.80,0.85) 40%   >= 0.85  17%
#
# **57% are 0.80-or-more similar to something already stored**, dominated by the band JUST BELOW the
# 0.85 threshold. `_pick_to_admit` selects by rarity within the frame and never asks the memory what
# it holds, so a crop the memory almost recognises is stored as if it were new.
#
# **I predicted the opposite ranking and withdraw it.** I expected duplication through the claim
# filter to dominate; it is 17% against the near-threshold band's 40%. Where it does happen it is
# fully accounted for: seven of eight cleared the claim filter on x, three on y, and **none on
# neither** -- `claimed` holds ONE position per slot.
#
# **And the published frame rate describes a memory holding one object.** Crops described per frame,
# counted rather than timed, on camera frames:
#
#     reported slots     2     5     8    16    33
#     with refinement   95   338   565   935   995        (the coarse pass alone is 70)
#
# 147 crops -- the recorded figure -- is ONE reported slot. At the ~20 slots this run settles on it is
# ~995, and the measured rate is **2.6 fps against a published 8.4**. Timed at matched slot counts,
# refinement costs 10-13x: 726 ms against 72 at 17 present, 671 against 51 at 33. The memory look-up
# is not the cause and its recorded "nearly flat from 1,000 to 4,000 slots" stands.
#
# That bears on a note already in this code: refining every seed rather than one is *"kept on
# principle, not on evidence"*, and a mutant refining only the first seed survives. The principle now
# has a measured price. **Nothing was changed on that basis** -- the benefit is still unmeasured,
# which is the half that matters.
#
# **The DISTRIBUTION is not held here and cannot be.** A test asserting the 57%-at-0.80 share was
# written and removed: no synthetic scene in this file reproduces it. Twelve frames of `render(...)`
# at every texture setting give **two** admissions, best prior scores 0.49-0.68, because these scenes
# are near-static and the admission gate barely fires -- the same fixture limit already recorded in
# `test_the_gate_fires_on_every_REAL_frame_and_no_fixture_here_can_show_it`. So the shares rest on
# camera frames, and what the three tests below hold is the MECHANISM.
#
# Costs are held as COUNTS, not clocks: a wall-clock figure is not something this suite gets to assert.


def _second_copy():
    """One disc, then the same disc twice -- the second far enough to clear the claim filter on x --
    with the crop at that second position, so a test can ask what the memory makes of it."""
    two = render([("disc", 0.25, 0.35), ("disc", 0.80, 0.75)], seed=1, texture="none")
    y, x = int(0.75 * H), int(0.80 * W)
    crop = np.asarray(two)[y - CROP // 2:y + CROP // 2, x - CROP // 2:x + CROP // 2]
    return render([("disc", 0.25, 0.35)], seed=1, texture="none"), two, crop


def _describe_with(layer, crop):
    d = np.asarray(layer.backbone.features(crop), dtype=np.float64)
    return d / max(float(np.linalg.norm(d)), 1e-12)


def _watch_admissions(layer):
    """The long run's own instrumentation, as a fixture: what did the memory already say about each
    descriptor at the moment it was stored?

    `look_up` before the write, which is public API -- the store has no public accessor and needs
    none for this. Returns the list the run collected 47 of.
    """
    seen = []
    real = layer.memory.admit

    def watched(descriptor, _s=seen, _r=real, _m=layer.memory):
        prior = _m.look_up(descriptor) if _m.occupied else None
        _s.append(None if prior is None else (float(prior.score), bool(prior.matched)))
        return _r(descriptor)

    layer.memory.admit = watched
    return seen


def test_ADMISSION_stores_what_the_memory_NEARLY_recognises():
    """**The dominant mechanism over 2,005 camera frames**, and the one I did not predict: 40% of
    admissions sit in [0.80, 0.85) against something already stored, more than twice the 17% that
    clear the threshold outright.

    `_pick_to_admit` selects by rarity WITHIN THE FRAME and never asks the memory what it holds, so
    a crop the memory *almost* recognises is stored as if it were new. This fixture lands in that
    band on its own -- 0.8132 against a 0.85 threshold -- which is why it is the test.

    A record of current behaviour, not an endorsement: it is why the owner saw a patchwork.
    """
    L = _presence_layer(step=4)
    one, two, crop = _second_copy()
    for _ in range(3):
        L.features(one)
    assert L.memory.occupied >= 1, "nothing was admitted at all"

    known = L.memory.look_up(_describe_with(L, crop))
    assert 0.75 <= known.score < L.memory.threshold, (
        f"the second copy scores {known.score:.4f} against threshold {L.memory.threshold}; this test "
        "is about the band just BELOW threshold and the fixture has left it")

    before = L.memory.occupied
    for _ in range(3):
        L.features(two)
    assert L.memory.occupied > before, (
        f"occupied stayed at {before} -- admission has started consulting the memory, which would be "
        "a real improvement and not the behaviour recorded here")


def test_a_duplicate_ABOVE_threshold_gets_in_through_the_claim_filter():
    """The secondary mechanism, 17% of admissions on camera, and the one with no unexplained residue:
    seven of eight cleared the claim filter on x, three on y, **none on neither**.

    `claimed = [c for _s, _sc, c in self.last_present]` holds ONE position per slot, so the same
    remembered thing seen far enough from where its slot was reported is still `free`.

    Asserted with the run's own instrumentation -- `look_up` on the descriptor about to be stored --
    because "occupied grew" is too weak: the admitted candidate is the rarest FREE one and need not
    be the duplicate. A mutant making admission consult the memory SURVIVED that weaker form.

    The threshold is lowered so this scene reaches the regime the camera reaches on its own. Stated
    rather than hidden, and it is a constructor parameter, not a fixture trick.
    """
    L = _presence_layer(step=4, threshold=0.80, capacity=16)
    seen = _watch_admissions(L)
    #: THREE copies, not two. With two the layer recognises both from one slot and the reported
    #: position lands on whichever scores higher, so nothing is left far enough away -- measured, and
    #: it is why the first version of this test failed. A third copy is what the camera's moving
    #: scene supplies continuously: more of the remembered thing than one reported position covers.
    here = [("disc", 0.20, 0.30), ("disc", 0.80, 0.30), ("disc", 0.50, 0.80)]
    dx = abs(0.80 - 0.20)
    assert dx > L._claim_x, (
        f"the copies are {dx:.3f} apart and the claim filter is {L._claim_x:.3f}; they would all be "
        "excluded and this test would prove nothing")

    for _ in range(3):
        L.features(render([here[0]], seed=1, texture="none"))
    for _ in range(5):
        L.features(render(here, seed=1, texture="none"))

    scored = [p for p in seen if p is not None]
    assert scored, "only the very first admission happened, so nothing was compared"
    matched = [sc for sc, m in scored if m]
    assert matched, (
        f"no admission was of something the memory already matched -- best prior scores were "
        f"{sorted(round(sc, 4) for sc, _m in scored)[-4:]} against threshold {L.memory.threshold}. "
        "Admission consulting the memory would be a real improvement, not the behaviour recorded")
    assert max(matched) >= L.memory.threshold


def test_REFINEMENT_costs_one_neighbourhood_per_REPORTED_slot():
    """Why the published 8.4 fps describes a memory holding one object.

    Counted, not timed: a wall-clock figure is not something this suite may assert, and the count is
    what the clock was following anyway -- 95 crops at 2 reported slots against 995 at 33 on camera,
    alongside timings of 10-13x at matched slot counts.
    """
    def described(layer, image):
        total = [0]
        real = layer._describe

        def counted(img, _t=total, _r=real, _L=layer):
            _t[0] += len(_L.centres)
            return _r(img)

        layer._describe = counted
        layer.features(image)
        return total[0]

    img = render([("disc", 0.25, 0.35), ("bar", 0.72, 0.70)], seed=1, texture="none")

    flat = _presence_layer(step=4)
    for _ in range(4):
        flat.features(img)
    coarse_only = described(flat, img)

    refined = _presence_layer(step=4, refine=(4, 2))
    for _ in range(4):
        refined.features(img)
    with_refine = described(refined, img)
    n = len(refined.last_present)

    assert coarse_only == len(flat.centres), (
        f"without refinement a pass must describe exactly the candidate grid; got {coarse_only} for "
        f"{len(flat.centres)} candidates")
    assert n >= 1, "nothing was reported, so there is no per-slot cost to measure"
    assert with_refine > coarse_only, (
        "refinement described no more candidates than the coarse pass, so it is not running")

    #: **The cost must SCALE with reported slots**, which is the whole claim -- 95 crops at 2 slots
    #: against 995 at 33. A mutant refining only the first seed passes any assertion on the total, so
    #: a one-slot scene is measured beside the many-slot one. The neighbourhoods overlap, so the
    #: growth is sublinear and 1.5x is the bar rather than n-fold.
    lone = _presence_layer(step=4, refine=(4, 2))
    solo = render([("disc", 0.25, 0.35)], seed=1, texture="none")
    for _ in range(4):
        lone.features(solo)
    added_one = described(lone, solo) - len(lone.centres)
    assert len(lone.last_present) < n, (
        f"the one-object scene reports {len(lone.last_present)} slots and the two-object scene {n}; "
        "they are not distinguishable, so the scaling below is not being measured")
    added_many = with_refine - coarse_only
    assert added_many > added_one * 1.5, (
        f"refinement added {added_many} candidates at {n} reported slots and {added_one} at "
        f"{len(lone.last_present)}; it is documented as one neighbourhood PER REPORTED SLOT and this "
        "says it runs for one seed regardless")


def test_REFINEMENT_DEGENERATES_into_the_fine_grid_it_was_invented_to_replace():
    """**What refinement buys, asked in the regime where it costs 14x — and the answer is nothing.**

    Measured 2026-08-25 on the file's own criterion (`back and back <= first`), 25 seeds, with the
    harness agreeing with `drive()` on 25/25 seeds for the clean row -- the one row anyone else can
    reproduce, because the harness itself is not in the repository:

    | load | scene | refine on | refine off | discordant | McNemar | smallest p reachable |
    |---|---|---|---|---|---|---|
    | 0 | clean | **22/25** | 19/25 | 4 on / 1 off | p = 0.375 | 0.0625 |
    | 0 | clutter | 16/25 | 16/25 | 5 / 5 | p = 1.000 | 0.0625 |
    | 8 | clutter | 10/25 | 10/25 | 4 / 4 | p = 1.000 | 0.125 |
    | 24 | clutter | 2/25 | 4/25 | 1 on / 3 off | p = 0.625 | 0.125 |

    **The last column is the honest reading and it was added after an audit: with 5 discordant
    pairs no outcome could have reached 0.05, and with 4 none could reach 0.125.** So "not
    significant" reports this design's sensitivity, not a property of refinement -- 25 seeds is too
    few to decide it either way, and the recorded categorical claim (plain step 8 loses the
    identity, refine keeps it) is true at the fixture's default seed 23 and is +3 of 25 across
    seeds. That is also why a mutant refining only the first seed survives; an independent audit
    measured that mutant at **22/25, identical to full refinement**, against 19/25 with refinement
    off -- so the whole per-slot 14x buys nothing over a single neighbourhood on this criterion.

    **THE THREE LOADED ROWS ARE WITHDRAWN AS A MEASUREMENT OF RETENTION.** Two independent
    reconstructions disagree with them and one disagrees in *sign* (load improving retention), and
    the construction that decides it -- capacity, what "noise" means, which scene -- was recorded in
    one clause and the harness was never checked in. Worse, the criterion does not measure retention
    under load at all: see
    `test_under_LOAD_the_retention_criterion_PASSES_FOR_THE_WRONG_REASON`, where the object is never
    admitted and the pass comes from riding a background slot. The rows are kept here as the record
    of what was claimed, not as evidence. The clean row stands.

    **And here is the reason, which is geometry and needs no images at all.** Each reported slot gets
    a neighbourhood, the neighbourhoods overlap, and their union grows into the step-2 grid that
    refinement exists to avoid. At full coverage `refine=(8, 2)` *is* the plain step-2 configuration,
    which the table above says also keeps the identity — so the two arms converge, and they do.

    Held as the union: deterministic, and no forward pass, no frames and no clock. (A
    `Backbone` is constructed, because a `RecognitionLayer` needs one; nothing is run
    through it. The earlier wording here said "no backbone", which was false as written.)
    """
    L = _presence_layer(step=8, refine=(8, 2))
    fine = {(x / W, y / H) for y in range(CROP // 2, H - CROP // 2 + 1, 2)
            for x in range(CROP // 2, W - CROP // 2 + 1, 2)}

    def covered(n, draws=8):
        g = np.random.default_rng(0)
        out = []
        for _ in range(draws):
            seeds = [L.centres[i] for i in g.choice(len(L.centres), n, replace=False)]
            union = set()
            for sx, sy in seeds:
                union.update(L._neighbourhood(sx, sy))
            out.append(len(union) / len(fine))
        return float(np.mean(out))

    one, eight, sixteen = covered(1), covered(8), covered(16)
    everything = covered(len(L.centres), draws=1)

    assert one < 0.15, (
        f"a single reported slot already refines {one:.1%} of the fine grid; the 6.3x saving in "
        "`test_local_REFINEMENT_keeps_the_identity_at_a_sixth_of_the_crops` depends on this being "
        "small")
    assert eight > 3 * one, (
        f"coverage barely grew with reported slots ({one:.1%} at one, {eight:.1%} at eight), so the "
        "cost is no longer per-slot and the measured 95 -> 995 crops has another cause")
    assert sixteen > 0.6, (
        f"at sixteen reported slots refinement covers only {sixteen:.1%} of the fine grid; the "
        "measured figure is ~73% and the whole point is that it approaches the grid it replaces")
    assert everything > 0.99, (
        f"refining every candidate covers {everything:.1%} of the step-2 grid rather than all of it;"
        " if that became false the neighbourhoods no longer tile and the argument above is wrong")


def _loaded(seed, load, refine=None, capacity=None):
    """A layer whose memory is already holding `load` background exemplars.

    One descriptor per slot, each from a DISTINCT image, admitted through the memory's own
    `admit` -- so what varies between this and the camera measurement recorded in
    docs/findings/vocabulary-and-labelling.md is the load's SOURCE and nothing else.
    """
    bb = Backbone(PLAN, (CROP, CROP, 3), dtype="f8", seed=seed, pool=POOL)
    mem = ExemplarMemory(capacity or load + 8, bb.n_outputs, 16, threshold=0.85)
    L = RecognitionLayer(bb, mem, (H, W, 3), grid(8), admit_margin=0.02, refine=refine)
    rng = np.random.default_rng(1000 + seed)
    for _ in range(load):
        cx, cy = L.centres[int(rng.integers(len(L.centres)))]
        v = np.asarray(L.backbone.features(L._crop(rng.random((H, W, 3)), cx, cy)),
                       dtype=np.float64)
        L.memory.admit(v / float(np.linalg.norm(v)))
    return L


def test_the_0_85_THRESHOLD_HAS_NO_HEADROOM_against_a_populated_memory():
    """**Why identity fails under load, and it is the threshold rather than the load's source.**

    Measured 2026-08-25, and the figures below are the ones THIS test computes -- 10 backbone
    seeds, 24 crops each. The object against ITSELF across the absence scores **0.875** against
    a 0.85 threshold: 0.025 of headroom, and its worst seed is 0.603. Arbitrary crops that have
    nothing to do with it clear the same threshold **48.8%** of the time. So a memory holding a
    handful of background exemplars almost always contains one that matches the object as well
    as the object does: `look_up` returns it, and the object is *recognised as background*
    instead of being stored.

    That is a property of an untrained backbone's descriptor rather than of the scene. Measured
    on the sweep's own banks at load 8, the fraction above threshold is **56.5% for synthetic
    noise and 67.0% for descriptors taken from 900 real camera frames** -- real crops are
    slightly WORSE here, not better, which is why the camera load admits the object less often
    rather than more. See `test_under_LOAD_the_retention_criterion_PASSES_FOR_THE_WRONG_REASON`.

    Deterministic and cheap: no drive(), no clock.
    """
    above, selfsim = [], []
    for seed in range(10):
        L = _loaded(seed, 0, capacity=8)
        rng = np.random.default_rng(500 + seed)

        def d(img, cx, cy):
            v = np.asarray(L.backbone.features(L._crop(img, cx, cy)), dtype=np.float64)
            return v / float(np.linalg.norm(v))

        obj = d(scene("A", 0), 0.30, 0.40)
        selfsim.append(float(d(scene("again", 0), 0.30, 0.40) @ obj))
        sims = []
        for _ in range(24):
            cx, cy = L.centres[int(rng.integers(len(L.centres)))]
            sims.append(float(d(rng.random((H, W, 3)), cx, cy) @ obj))
        above.append(float(np.mean(np.asarray(sims) >= 0.85)))

    mean_self, mean_above = float(np.mean(selfsim)), float(np.mean(above))
    assert mean_self > 0.85, (
        f"the object no longer clears its own threshold ({mean_self:.3f}); if that became true "
        "the failure below is ordinary rejection and not a collision, and the explanation in "
        "this docstring should be re-derived")
    assert mean_above > 0.35, (
        f"only {mean_above:.1%} of arbitrary crops clear the 0.85 threshold against the object, "
        "against a measured 48.8%. If the descriptor became this discriminative the collision "
        "explanation is wrong -- and the load results below should have improved with it")


def test_under_LOAD_the_retention_criterion_PASSES_FOR_THE_WRONG_REASON():
    """**The recorded load table does not measure identity retention, and this is the control
    that shows it.** Measured 2026-08-25.

    `back <= first` is this file's criterion and it is silent about *which* slot. Under load the
    object is never admitted -- a background exemplar already clears the threshold, so
    `look_up` matches it -- and the criterion then passes because the object rides that same
    background slot on both appearances. A pass, for the opposite of the reason intended.

    Background occupies slots 0..load-1 and `ExemplarMemory.admit` takes the first free slot,
    so a slot >= `load` did not exist until the object arrived. That is the separation:

    | load | source | kept | of which own slot | of which rode background |
    |---|---|---|---|---|
    | 0 | -- | 22/25 | **22** | 0 |
    | 8 | noise | 15/25 | 5 | **10** |
    | 8 | 900 real camera frames | 23/25 | **0** | **23** |
    | 24 | noise | 12/25 | 2 | 10 |
    | 24 | 900 real camera frames | 19/25 | **0** | 19 |

    **The real-camera arm is the finding.** Replacing the constructed load with descriptors from
    900 frames of an actual scene -- the assumption `docs/findings/vocabulary-and-labelling.md`
    flagged as the one to distrust -- makes the recorded criterion look BETTER (23/25 against
    15/25) while the object is admitted **0 times out of 25**. A first reading of that as "the
    real load is gentler" is withdrawn: it is more degeneracy, not less.

    Genuine retention is therefore **22/25 at load 0, 5/25 at noise load 8, and 0/25 at real
    load 8** -- a sharper collapse than the recorded 22 -> 16 -> 10 -> 2-4, and it is the
    owner's *"계속 헤맨다"* with its mechanism attached.

    Only the noise arm is held here: it needs no camera and no stored frames. ~5 s.
    """
    kept = own = rode = 0
    for seed in range(25):
        L = _loaded(seed, 8)
        before = L.memory.occupied
        seq = drive(L)
        first = {s for p, s, _r in seq if p == "A" and s >= 0}
        back = {s for p, s, _r in seq if p == "again" and s >= 0}
        if first and back <= first:
            kept += 1
            if any(s >= before for s in first):
                own += 1
            else:
                rode += 1

    assert kept > own, (
        f"all {kept} passing seeds admitted the object as its own exemplar, so the criterion is "
        "no longer blind to which slot it got. If that is a real improvement, the load table in "
        "docs/findings/vocabulary-and-labelling.md should be re-measured and this test retired")
    assert rode > own, (
        f"under a load of 8 the criterion passed {kept}/25 times, {own} on the object's own slot "
        f"and {rode} riding a background slot; the recorded measurement is 5 own against 10 rode "
        "and the whole point is that the majority of passes are degenerate")
    assert own < 12, (
        f"{own}/25 seeds stored the object as its own exemplar under load, against a measured 5. "
        "Admission under load working would be a real fix and would invalidate the table above")
