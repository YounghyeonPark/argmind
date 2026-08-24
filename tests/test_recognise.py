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




def _presence_layer(step=4, capacity=8, threshold=0.85, seed=23):
    from argmind.backbone import Backbone
    bb = Backbone(PLAN, (CROP, CROP, 3), dtype="f8", seed=seed, pool=POOL)
    mem = ExemplarMemory(capacity, bb.n_outputs, 16, threshold=threshold)
    return RecognitionLayer(bb, mem, (H, W, 3), grid(step), admit_margin=0.02, emit="presence")


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
