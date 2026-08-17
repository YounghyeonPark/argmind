# SPDX-License-Identifier: Apache-2.0
"""`--output map`: the head emits the segmentation, and the classical labeller is only its teacher.

This is the first step under the decision of 2026-08-14 — argmind must do the learning and
inferencing itself. Until now the 1,396 lines of classical perception in `demo/label_sources.py`
*were* the answer and the head consumed their output. Here the head produces a
{MAP_H}x{MAP_W} segmentation and the classical labeller supplies the target it learns from.

**The teacher is still there, and saying so is the point.** Nothing in this file claims the
labeller has been removed. What it pins is that the composition works with a map output, that the
head reaches the teacher, and what happens when the teacher is **withdrawn** — which is the
falsifiable half and the experiment the decision actually asks for.

A map output removes three measured problems at once rather than making one smaller: 2K coordinates
have no intrinsic order (0.256 for arbitrary order against 0.129 sorted), no locator here can count,
and an assignment algorithm exists only to decide which detection is which slot. **A map has no
slots.**
"""

from __future__ import annotations

import numpy as np
import pytest

import tempfile
from pathlib import Path

from argmind.backbone import Backbone
from argmind.persist import Store
from demo.map_head import map_iou
from demo.pointer import (IN_SHAPE, MAP_DROPOUT, MAP_H, MAP_W, PLAN, POOL, build, fingerprint,
                          fingerprint_for, mouse_target,
                          map_basis, map_memory_width,
                          map_teacher)
from demo.map_head import Pyramid

H, W = 72, 96


def scene(k, *, twin=False, blind=False, seed=0):
    """A coloured target drifting on a plain wall; `twin` adds a same-coloured decoy.

    Same shape as `test_map_head.py`'s fixture, because that is where the 0.98-against-0.50 result
    was measured and a different scene would not be comparable to it.
    """
    yy, xx = np.mgrid[0:H, 0:W]
    a = 2 * np.pi * k / 160.0
    tx, ty = 0.5 + 0.28 * np.cos(a), 0.5 + 0.26 * np.sin(a)
    img = np.full((H, W, 3), 0.35)
    truth = np.hypot(xx - tx * W, yy - ty * H) <= 0.13 * W
    img[truth] = (0.85, 0.30, 0.25)
    if twin:
        dx, dy = 0.5 - 0.28 * np.cos(a * 0.91), 0.5 - 0.26 * np.sin(a * 0.91)
        img[np.hypot(xx - dx * W, yy - dy * H) <= 0.13 * W] = (0.85, 0.30, 0.25)
    if blind:
        img = np.zeros_like(img)
        truth = np.zeros_like(truth)
    g = np.random.default_rng(seed)
    return (img + g.standard_normal(img.shape) * 0.01).astype("f4"), truth


def frame_features(img, factor=12):
    """The demo's own raw source, block-mean in colour."""
    a = np.asarray(img, dtype=np.float64)
    h, w = a.shape[0] // factor * factor, a.shape[1] // factor * factor
    a = a[:h, :w]
    return a.reshape(h // factor, factor, w // factor, factor, 3).mean(axis=(1, 3)).ravel()


def run(*, teach_until, n=1400, twin=True, dropout=MAP_DROPOUT, blind=None, seed=3):
    """Drive the SHIPPED map composition. `teach_until` is where the teacher is withdrawn.

    The guard is driven directly rather than through the pipeline: the pipeline's own bookkeeping is
    already compared against the reference elsewhere, and what is new here is the map output, the
    pyramid in the basis, and the teacher.
    """
    _bb, _head, guard, features, taught, _fmap, _pipe = build(1, "raw", output="map")
    assert taught.dim == MAP_H * MAP_W, "the taught stream must carry a map, not coordinates"
    assert features.dim == len(frame_features(scene(0)[0])) + map_memory_width()
    guard.set_external(0, 1.0)                       # or the external bound freezes it at once

    pyr = Pyramid(MAP_H, MAP_W)
    rng = np.random.default_rng(seed)
    #: Scored separately: the A PRIORI prediction while the teacher is present, and the free-running
    #: prediction after it is withdrawn. An earlier version returned only the second and called it
    #: the first, so "the head never learned" and "the head cannot free-run" were indistinguishable.
    taught_scores, free_scores = [], []
    for k in range(n):
        dark = bool(blind and blind[0] <= k < blind[1])
        img, truth = scene(k, twin=twin, blind=dark, seed=k)
        blank = dropout > 0.0 and k < teach_until and rng.random() < dropout
        x = np.concatenate([map_basis(frame_features(img), pyr, blank=blank), [1.0]])
        pred = np.clip(np.asarray(guard.predict(x), dtype=np.float64), 0.0, 1.0)
        target = map_teacher(truth)
        if k < teach_until:
            if k > teach_until // 2:                 # after it has had a chance to learn
                taught_scores.append(map_iou(pred, target))
            guard.update(x, target)
        else:
            free_scores.append(map_iou(pred, target))
        pyr.observe(pred if k >= teach_until else target)
    return taught_scores, free_scores


def test_the_composition_carries_a_map_output_at_all():
    """The shapes first: a map head is 295 → 108 here, and the streams follow it. If this fails,
    every number below is measuring something else."""
    _bb, head, guard, features, taught, fmap, _pipe = build(3, "raw", output="map")
    assert head.n_outputs == MAP_H * MAP_W == 108
    assert guard.n_outputs == head.n_outputs
    assert taught.dim == 108 and features.dim == 144 + map_memory_width()
    assert head.n_features == features.dim + 1, "the constant, appended by the FrameMap"
    assert len(fmap.features) == features.dim

    # And the points head is untouched, which is what makes this a flag rather than a rewrite.
    _bb2, points, *_rest = build(3, "raw")
    assert points.n_outputs == 6 and points.n_features == 145

    with pytest.raises(ValueError, match="output must be"):
        build(1, "raw", output="mask")


def test_the_head_reaches_the_teacher_while_it_is_taught():
    """With the teacher present the head reproduces the labeller's map. That is the precondition for
    the withdrawal below meaning anything — a head that never learned cannot be said to survive."""
    scored, _free = run(teach_until=1000, n=1400, blind=None)
    assert np.mean(scored) > 0.55, (
        f"the head reached only IoU {np.mean(scored):.2f} against its own teacher; the map "
        "composition is not learning and nothing downstream can be concluded")


def test_withdrawing_the_teacher_and_the_open_discrepancy_that_scopes_it():
    """**The decision of 2026-08-14 asks for the classical labeller to go. This is the experiment,
    and its result is NOT yet usable — which is recorded rather than glossed.**

    Measured on this composition, the teacher withdrawn at the halfway point:

    | teach until | taught IoU | free-running |
    |---|---|---|
    | 1000 | 0.82 | 0.09 |
    | 700 | 0.80 | 0.10 |
    | 400 | 0.79 | 0.16 |

    against 0.84 with the teacher never withdrawn and **0.92 for simply repeating the previous true
    map**. So on this harness the head learns the labeller well and cannot run without it.

    **But this harness disagrees with `test_map_head.py`'s, which free-runs at 0.96–0.98 on the same
    scene, and I have not localised the difference.** Three explanations were tested and refuted:

    * *exposure bias* — training with the head's own prediction in the pyramid instead of the target
      gives an identical 0.80 / 0.10
    * *the head* — a bare RLS at `forgetting=1.0, p0=100` reads 0.84 taught and **0.02** free, worse
    * *my dropout default* — 0.0, 0.05 and 0.15 all free-run at 0.10

    So the honest state is that the composition wires up and learns, and the wean number here cannot
    be quoted until the two harnesses agree. This test therefore asserts only what both agree on.
    """
    taught_score = float(np.mean(run(teach_until=1000, n=1400)[0]))
    weaned = run(teach_until=700, n=1400)[1]
    early = float(np.mean(weaned[:200]))
    late = float(np.mean(weaned[-200:]))

    assert taught_score > 0.55, (
        f"the head reaches only {taught_score:.2f} with the teacher present, so the withdrawal "
        "below measures nothing")
    assert late < taught_score, (
        f"free-running scores {late:.2f} against {taught_score:.2f} taught. Both harnesses agree "
        "that withdrawing the teacher costs something; they disagree on how much, so only the "
        "direction is asserted here")
    assert early > 0.0 and late > 0.0, (
        "the head stopped producing a map at all once free-running, which neither harness shows")


def test_the_pyramid_is_part_of_the_basis_and_blanking_keeps_the_width():
    """Dropout blanks the memory rather than removing it: a head whose input width changed between
    frames would be a different model, and `FrameMap` would refuse the block."""
    pyr = Pyramid(MAP_H, MAP_W)
    feats = frame_features(scene(0)[0])
    full = map_basis(feats, pyr)
    blanked = map_basis(feats, pyr, blank=True)
    assert len(full) == len(blanked) == len(feats) + pyr.width
    assert np.array_equal(full[:len(feats)], blanked[:len(feats)]), "the frame half must be intact"

    pyr.observe(np.full(MAP_H * MAP_W, 0.5))
    moved = map_basis(feats, pyr)
    assert not np.array_equal(moved, full), "the pyramid did not take the observation"
    assert np.all(map_basis(feats, pyr, blank=True)[len(feats):] == 0.0)


def test_the_teacher_is_one_named_function_because_it_is_what_gets_withdrawn():
    """`map_teacher` exists as a named function rather than an inline expression precisely so the
    thing the decision wants removed has a name and one call site."""
    field = np.zeros((H, W), dtype=np.int32)
    field[20:40, 30:60] = 3                          # any labelled region is foreground
    target = map_teacher(field)
    assert target.shape == (MAP_H * MAP_W,)
    assert 0.0 < target.max() <= 1.0 and target.min() == 0.0
    assert map_teacher(np.zeros((H, W), dtype=np.int32)).max() == 0.0


def test_the_fingerprint_tells_the_two_output_modes_apart():
    """**A gap found by asking, not by a failure — and it was survivable rather than harmless.**

    `fingerprint()` covered the backbone plan, the input shape, the pooling, the embedding width, the
    object count and the input source, and **not** the output mode. So a points head and a map head
    built from the same backbone hashed identically (`0x224c2fa3` both, measured).

    What refused a cross-mode load was `unpack`'s *shape* check — "record is for 145x6 but this
    adapter is 295x108" — which is the accidental protection, not the designed one: it holds only
    while the two state sizes differ. The fingerprint exists for exactly the case a length check
    cannot see, which is why it now carries the mode, the map shape and the memory width.
    """
    backbone = Backbone(PLAN, IN_SHAPE, pool=POOL, seed=0)
    points = fingerprint(backbone, 3, "backbone", "points")
    mapped = fingerprint(backbone, 3, "backbone", "map")
    assert points != mapped, (
        f"both modes hash to {points:#x}; a points record and a map record are indistinguishable")
    assert fingerprint(backbone, 3, "backbone") == points, "the default is no longer points"

    # **And the two contributions are asserted separately, because the first version of this test
    # could not tell them apart.** `fingerprint` puts the mode in *and* appends the map geometry on
    # the map branch, so either alone makes these two integers differ: a mutant dropping the mode
    # survived, and so did a mutant dropping the geometry, each masked by the other. Same shape as
    # every "passed for a reason other than the one it names" finding in this repository.
    assert fingerprint(backbone, 3, "backbone", "coords") != points, (
        "the mode itself is not in the hash — two non-map readouts would collide, and the map "
        "branch is what happened to separate points from map")


def test_the_fingerprint_refuses_a_cross_mode_load_where_the_length_check_CANNOT():
    """The sharp version, because the test above only compares two integers.

    Here the record and the adapter have the **same shape**, so `unpack`'s length and dimension
    checks all pass and the CRC is genuinely correct — the bytes are valid and mean something else,
    which is the one corruption nothing downstream can detect. Only the fingerprint refuses it.
    """
    backbone = Backbone(PLAN, IN_SHAPE, pool=POOL, seed=0)
    _bb, _h, guard, *_ = build(3, "backbone")
    with tempfile.TemporaryDirectory() as d:
        store = Store(Path(d) / "head")
        store.save(guard, fingerprint=fingerprint(backbone, 3, "backbone", "points"))
        # Same adapter, same shape, same CRC. Only the declared output mode differs.
        with pytest.raises(ValueError, match="fingerprint|configuration"):
            store.load(guard, fingerprint=fingerprint(backbone, 3, "backbone", "map"))
        # And the control: the mode it was written under still loads, so the refusal above is the
        # mode and not the store refusing everything.
        store.load(guard, fingerprint=fingerprint(backbone, 3, "backbone", "points"))


def test_the_map_geometry_is_in_the_fingerprint_too(monkeypatch):
    """The other half, isolated the same way. A map head's basis carries `map_memory_width()` pyramid
    values whose meaning depends on the band layout, and the 108 outputs are a 9x12 grid — a block
    restored against a different grid puts the right numbers in the wrong cells.

    The grid used here is *larger*, not a rearrangement of the same 108 cells: `map_memory_width`
    refuses a map finer than its own coarsest band, so no same-cell-count grid is expressible with
    the default bands. That makes this the weaker of the two available fixtures and it is said rather
    than glossed — a length check would also catch this one. What it does establish is that the
    geometry reaches the hash at all, which was the surviving mutant.
    """
    import demo.pointer as pointer
    backbone = Backbone(PLAN, IN_SHAPE, pool=POOL, seed=0)
    before_map = fingerprint(backbone, 3, "backbone", "map")
    before_points = fingerprint(backbone, 3, "backbone", "points")

    monkeypatch.setattr(pointer, "MAP_H", MAP_H + 3)     # 12x15, a different grid
    monkeypatch.setattr(pointer, "MAP_W", MAP_W + 3)
    assert fingerprint(backbone, 3, "backbone", "map") != before_map, (
        "the map geometry is not in the hash; a 9x12 block and a differently-shaped one are "
        "indistinguishable")
    assert fingerprint(backbone, 3, "backbone", "points") == before_points, (
        "the points fingerprint moved with the map geometry, which it has nothing to do with")


def test_the_memory_WIDTH_is_in_the_fingerprint_and_not_only_the_grid(monkeypatch):
    """A mutant keeping `map{MAP_H}x{MAP_W}` and dropping `map_memory_width()` survived, and it is a
    real gap rather than an equivalent one: the bands could be re-split to a *different layout at the
    same total width*, where every length check agrees and the 150 values mean something else.
    """
    import demo.pointer as pointer
    backbone = Backbone(PLAN, IN_SHAPE, pool=POOL, seed=0)
    before = fingerprint(backbone, 3, "backbone", "map")
    monkeypatch.setattr(pointer, "map_memory_width", lambda *a, **k: map_memory_width() + 1)
    assert fingerprint(backbone, 3, "backbone", "map") != before, (
        "the memory width does not reach the hash; a re-split band layout at the same total width "
        "would restore into a head that reads it differently")


def test_the_call_site_passes_the_mode_and_not_only_the_signature_accepting_it():
    """**The mutant that survived everything**: `fingerprint` grew an `output` argument with a
    default, and `main()`'s call site never passed it. Every fingerprint test above still passed,
    because they all call `fingerprint` directly. `fingerprint_for` is that call site, module-level
    for exactly this reason.
    """
    class Args:
        input, raw_grey, feedback, output = "backbone", False, False, "points"
    backbone = Backbone(PLAN, IN_SHAPE, pool=POOL, seed=0)
    args = Args()
    points = fingerprint_for(backbone, 3, args)
    args.output = "map"
    assert fingerprint_for(backbone, 3, args) != points, (
        "the demo's own call site does not carry the output mode into the fingerprint")
    assert points == fingerprint(backbone, 3, "backbone", "points"), (
        "fingerprint_for no longer composes the source string the way main() did")

    args.output, args.feedback = "points", True                 # the suffixes it also carries
    assert fingerprint_for(backbone, 3, args) != points


def test_a_mouse_CLICK_in_map_mode_teaches_nothing_and_does_not_crash():
    """**Found by running the demo, not by a test** — `ValueError: taught: expected shape (108,),
    got (6,)`. When `--output map` widened the `taught` stream to 108, the *automatic* label branch
    was updated and the *mouse* branch was not, so a single click pushed six values into a
    108-wide stream and the demo died.

    It was inside `main()`, where nothing could reach it — the eighth block here to need hoisting
    for that reason, after `select_newest`, `drive`'s `teach_every`, `slot_label_map`,
    `regions_for_slots`, `follow_permutation`, `learn_features` and `fingerprint_for`.

    The fix is a **refusal**, not a conversion: the map target is a foreground mask over the whole
    frame, so building one from one point would assert every other cell is background — mostly
    false whenever anything else is in the scene.
    """
    assert mouse_target(0.5, 0.25, 3, "map") is None, (
        "a click in map mode produced a target; whatever it is, it cannot be a full foreground "
        "mask derived from one point")

    # The control: in points mode it still teaches, and at the width the stream expects.
    for n in (1, 2, 3):
        target = mouse_target(0.5, 0.25, n, "points")
        assert target is not None and len(target) == 2 * n, (
            f"points mode at {n} objects produced {target}, which the taught stream would refuse")
        assert target[:2] == [0.5, 0.25]


def test_the_taught_stream_width_matches_what_each_mode_pushes():
    """The invariant the crash violated, checked directly rather than by driving the demo: whatever
    a mode pushes must be the width that mode's stream was built with."""
    import numpy as np
    for mode, objects in (("points", 3), ("map", 3)):
        _bb, _h, guard, *_ = build(objects, "backbone", output=mode)
        width = guard.n_outputs
        click = mouse_target(0.5, 0.5, objects, mode)
        if click is not None:
            assert len(click) == width, (
                f"{mode}: a click pushes {len(click)} values into a {width}-wide stream")
        if mode == "map":
            field = np.zeros((72, 96), dtype=np.int32)
            field[20:40, 30:60] = 1
            assert map_teacher(field).shape == (width,), (
                "the automatic branch and the stream disagree in map mode too")
