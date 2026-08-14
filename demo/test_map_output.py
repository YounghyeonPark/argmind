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

from demo.map_head import map_iou
from demo.pointer import (MAP_DROPOUT, MAP_H, MAP_W, build, map_basis, map_memory_width,
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
