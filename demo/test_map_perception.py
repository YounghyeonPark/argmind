# SPDX-License-Identifier: Apache-2.0
"""Does the map head read the frame, or coast on its own memory? Measured on scenes that can tell.

**This file exists because a conclusion recorded on 2026-08-17 was wrong.** An ablation on the
*twin* scene read frame-only 0.115 against memory-only 0.349 and I wrote that the head "does not
read the frame in either configuration". The twin scene is one where the frame **provably cannot**
resolve the target: two identical same-coloured blobs, and this repository's own recorded control is
that a per-frame segmenter reads 0.50 there. So that ablation measured a task the frame cannot do,
not a head that will not use it — the fixture, not the model.

Measured across three scenes, one axis at a time, `dropout=0`, `p0=100`, `forgetting=1.0`:

| scene | persistence | both | frame only | memory only |
|---|---|---|---|---|
| smooth orbit, twin | — | 0.359 | 0.115 | 0.349 |
| smooth orbit, single | 0.920 | 0.994 | 0.177 | 0.685 |
| new position every 7 frames | 0.873 | 0.910 | 0.882 | 0.000 |
| **new position every frame** | **0.126** | 0.949 | **0.928** | **0.000** |

**The head uses whichever half carries the information**, which is least squares doing its job, not
a defect. On a scene where the previous map predicts the next one it leans on memory, because
persistence alone scores 0.920 there. On a scene where memory is worthless it reads the frame and
scores 0.928 from the frame alone, against a persistence baseline that has collapsed to 0.126.

Each run below is ~1 second, which is also a correction: the record for the earlier ablation said it
was "too slow for a per-commit suite" and the pair actually takes **2.6 s**. That justification was
asserted without being measured, in a commit that was careful about everything else.
"""

from __future__ import annotations

import numpy as np
import pytest

from argmind.adapt import RecursiveLeastSquares
from demo.map_head import Pyramid, map_iou
from demo.pointer import MAP_H, MAP_W, build, map_basis, map_teacher
from demo.test_map_output import H, W, frame_features

NF = 144                     #: the raw-frame part of the basis; the rest is the pyramid


def scene(k, *, jump_every, seed=0):
    """A single coloured blob. `jump_every=None` is the smooth orbit; an integer teleports it to a
    fresh draw that often, which is what makes the previous map worthless as a predictor.

    Single, not twinned, on purpose: a same-coloured twin cannot be told from its decoy by anything
    looking at one frame, so including it would ask the frame to do the provably impossible.
    """
    yy, xx = np.mgrid[0:H, 0:W]
    if jump_every:
        g = np.random.default_rng(1000 + k // jump_every)
        tx, ty = 0.22 + 0.56 * g.random(), 0.22 + 0.56 * g.random()
    else:
        a = 2 * np.pi * k / 160.0
        tx, ty = 0.5 + 0.28 * np.cos(a), 0.5 + 0.26 * np.sin(a)
    img = np.full((H, W, 3), 0.35)
    truth = np.hypot(xx - tx * W, yy - ty * H) <= 0.13 * W
    img[truth] = (0.85, 0.30, 0.25)
    g = np.random.default_rng(seed)
    return (img + g.standard_normal(img.shape) * 0.01).astype("f4"), truth


def ablate(*, jump_every, teach_until=300, n=400):
    """Free-running IoU with the basis whole, with the memory zeroed, and with the frame zeroed —
    plus a persistence baseline, which is what says whether the scene can discriminate at all."""
    _bb, _h, _g, features, *_rest = build(1, "raw", output="map")
    head = RecursiveLeastSquares(features.dim + 1, MAP_H * MAP_W,
                                 forgetting=1.0, p0=100.0, dtype="f8")
    pyr, prev = Pyramid(MAP_H, MAP_W), None
    out = {"both": [], "frame": [], "memory": [], "persistence": []}
    for k in range(n):
        img, truth = scene(k, jump_every=jump_every, seed=k)
        x = np.concatenate([map_basis(frame_features(img), pyr), [1.0]])
        target = map_teacher(truth)
        if k < teach_until:
            head.update(x, target)
        else:
            for name, cut in (("both", None), ("frame", slice(NF, -1)), ("memory", slice(0, NF))):
                xx_ = x.copy()
                if cut is not None:
                    xx_[cut] = 0.0
                p = np.clip(np.asarray(head.predict(xx_), dtype=np.float64), 0.0, 1.0)
                out[name].append(map_iou(p, target))
            if prev is not None:
                out["persistence"].append(map_iou(prev, target))
        pred = np.clip(np.asarray(head.predict(x), dtype=np.float64), 0.0, 1.0)
        pyr.observe(pred if k >= teach_until else target)
        prev = target
    return {k: float(np.mean(v)) for k, v in out.items()}


def test_the_head_DOES_read_the_frame_when_the_frame_is_what_carries_the_answer():
    """**The correction.** On a scene whose previous map predicts nothing, the head scores from the
    frame alone and scores nothing from memory alone."""
    r = ablate(jump_every=1)
    assert r["persistence"] < 0.3, (
        f"the persistence baseline reads {r['persistence']:.3f}; this scene has not collapsed it, "
        "so it cannot tell a perceiving head from a coasting one and nothing below means anything")
    assert r["frame"] > 0.7, f"frame-only reads {r['frame']:.3f}; the head is not using its input"
    assert r["memory"] < 0.1, (
        f"memory-only reads {r['memory']:.3f}; if memory still scores here the scene is not as "
        "unpredictable as it looks")
    assert r["frame"] > r["memory"], "frame-only must beat memory-only or the claim is not made"


def test_on_a_PREDICTABLE_scene_it_leans_on_memory_instead_and_that_is_not_a_defect():
    """The control that shows the *fixture* is what moves the answer. Persistence alone scores 0.92
    on a smooth orbit, so least squares putting its weight there is correct behaviour, not a failure
    to perceive."""
    r = ablate(jump_every=None)
    assert r["persistence"] > 0.8, (
        f"persistence reads {r['persistence']:.3f} on the smooth orbit; it used to read 0.92 and "
        "the comparison this test draws depends on memory being nearly sufficient here")
    assert r["memory"] > r["frame"], (
        f"memory-only {r['memory']:.3f} no longer beats frame-only {r['frame']:.3f} on the smooth "
        "scene — the contrast with the jump scene is what makes this a finding about fixtures")


@pytest.mark.parametrize("jump_every,expect_frame_wins", [(1, True), (7, True), (None, False)])
def test_the_ordering_moves_with_the_scene_and_only_with_the_scene(jump_every, expect_frame_wins):
    """Same head, same basis, same rates — only the trajectory differs. If the ordering did not
    follow the scene, the two results above would be about something else."""
    r = ablate(jump_every=jump_every)
    assert (r["frame"] > r["memory"]) is expect_frame_wins, (
        f"jump_every={jump_every}: frame {r['frame']:.3f} vs memory {r['memory']:.3f}")
