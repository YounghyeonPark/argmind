# SPDX-License-Identifier: Apache-2.0
"""Does recognition appear in the hidden layers when they are learned from sensor data?

The falsifiable form of "recognition works in the middle hidden network", run with no trained initial
state: Glorot random, then SGD through `Backbone.backward` on a position task.

**The answer is no, and the stack does learn.** That pairing is the finding. Trained end to end the
stack reaches a position error of 0.010-0.016 against 0.53-0.56 with the labels shuffled, so the
gradient works and the layers move — and object selectivity stays exactly where random
initialisation left it:

| run | mean d-prime | per seed |
|---|---|---|
| random init | **0.532** | 0.339, 0.499, 0.757 |
| trained | **0.511** | 0.423, 0.313, 0.796 |

The sign of the change differs by seed, and the seed spread of the random control alone (0.339 to
0.757) is larger than any effect. **Recognition is not a by-product of an objective that does not
need it**: position is a linear readout of a spatial map, so the cheapest route to low error is to
preserve spatial layout, not to become selective for what is there.

Two measurement errors were made getting here and both are pinned below.

**A single seed said the opposite.** At seed 7 alone the numbers were random 0.676, trained 1.157 --
"1.7x the control" -- because that initialisation happened to be unusually unselective. At seed 11
training made it *worse*, 2.276 to 1.995. Fourth single-seed or aggregation artifact in this
project's history, and the reason the table above is per seed.

**And the first metric could not fail.** With objects of different *colours*, a random filter has a
random colour preference for free and scores a mean d-prime of 1.532 before any training at all. The
fixture here gives all three objects the **same colour** and different shapes, so selectivity has to
come from shape. `test_the_colour_fixture_cannot_fail` keeps that reasoning measured rather than
described.

The shuffled-label control is **not** cited as evidence: its d-prime of 0.090 comes with unit
liveness of 0-58% against 75-100%, so training on noise kills the units and the low score is the
dead-unit explanation. `test_the_shuffled_control_is_vacuous` records that so it is not quoted later.
"""

from __future__ import annotations

import numpy as np
import pytest

from argmind.adapt import RecursiveLeastSquares
from argmind.backbone import Backbone, conv2d, depthwise_conv2d, relu6

H, W = 36, 48
PLAN = [("conv", 8, 5, 2), ("dw", 3, 2), ("pw", 12)]        # 36x48 -> 18x24 -> 9x12, 12 channels
POOL = 3
WALL = 0.35
SAME = (0.85, 0.25, 0.20)
CENTRES = [(0.24, 0.30), (0.74, 0.32), (0.48, 0.74)]
PERIOD = 61.0
STEPS = 200


def frame_at(t, seed=0, r=0.13, colours=None, shapes=("disc", "square", "bar")):
    """Three objects on their own orbits. `colours` defaults to all the SAME, which is what stops a
    random colour preference from scoring."""
    colours = colours or [SAME, SAME, SAME]
    yy, xx = np.mgrid[0:H, 0:W]
    img = np.full((H, W, 3), WALL)
    masks, truth = [], []
    for i, (colour, shape) in enumerate(zip(colours, shapes)):
        a = 2 * np.pi * t / PERIOD + 2.1 * i
        cx, cy = CENTRES[i]
        p = (cx + 0.10 * float(np.cos(a)), cy + 0.10 * float(np.sin(a)))
        if shape == "disc":
            m = np.hypot(xx - p[0] * W, yy - p[1] * H) <= r * W
        elif shape == "square":
            m = (np.abs(xx - p[0] * W) <= r * W * 0.8) & (np.abs(yy - p[1] * H) <= r * W * 0.8)
        else:
            m = (np.abs(xx - p[0] * W) <= r * W * 1.6) & (np.abs(yy - p[1] * H) <= r * W * 0.35)
        img[m] = colour
        masks.append(m)
        truth += [p[0], p[1]]
    g = np.random.default_rng(seed)
    return (np.clip(img + g.standard_normal(img.shape) * 0.01, 0, 1).astype("f8"), masks,
            np.asarray(truth))


PROBE = [(t, 9000 + t) for t in range(0, 61, 10)]           # held-out seeds over one orbit


def last_map(bb, img):
    x = np.asarray(img, dtype=bb.dtype)
    for entry, w, b in zip(bb.plan, bb._w, bb._b):
        if entry[0] == "conv":
            z = conv2d(x, w, b, stride=entry[3])
        elif entry[0] == "pw":
            z = conv2d(x, w, b, stride=1)
        else:
            z = depthwise_conv2d(x, w, b, stride=entry[2])
        x = relu6(z)
    return x


def shrink(mask, shape):
    h, w = shape
    fh, fw = mask.shape[0] / h, mask.shape[1] / w
    out = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            tile = mask[int(i * fh):max(int((i + 1) * fh), int(i * fh) + 1),
                        int(j * fw):max(int((j + 1) * fw), int(j * fw) + 1)]
            out[i, j] = tile.mean() if tile.size else 0.0
    return out > 0.5


def selectivity(bb, **kw):
    """Per unit, the best d-prime between one object and the other two. Scale-free, and a dead unit
    scores 0 rather than NaN."""
    acc = None
    for t, seed in PROBE:
        img, masks, _truth = frame_at(t, seed=seed, **kw)
        m = last_map(bb, img)
        h, w, c = m.shape
        small = [shrink(mk, (h, w)) for mk in masks]
        if acc is None:
            acc = [[[] for _ in masks] for _ in range(c)]
        for u in range(c):
            plane = m[:, :, u]
            for o, mk in enumerate(small):
                if mk.any():
                    acc[u][o].append(plane[mk].ravel())
    out = []
    for u in range(len(acc)):
        best = 0.0
        for o in range(len(acc[u])):
            if not acc[u][o]:
                continue
            on = np.concatenate(acc[u][o])
            others = [np.concatenate(acc[u][p]) for p in range(len(acc[u])) if p != o and acc[u][p]]
            if not others:
                continue
            rest = np.concatenate(others)
            sd = np.sqrt(0.5 * (on.var() + rest.var()))
            if sd > 1e-12:
                best = max(best, abs(on.mean() - rest.mean()) / sd)
        out.append(best)
    return np.asarray(out)


def alive(bb, **kw):
    sds = [last_map(bb, frame_at(t, seed=s, **kw)[0]).reshape(-1, 12).std(axis=0)
           for t, s in PROBE]
    return float((np.mean(sds, axis=0) > 1e-6).mean())


def theta_of(head, n_outputs):
    """The head's coefficients, with the layout asserted rather than assumed."""
    n = head.n_features
    theta = np.asarray(head.state[:n * n_outputs]).reshape(n, n_outputs)
    probe = np.arange(1.0, n + 1.0) / n
    assert np.allclose(probe @ theta, np.asarray(head.predict(probe)), rtol=1e-9, atol=1e-12), (
        "RLS's theta is not (n_features, n_outputs) row-major; a transposed read would give a "
        "gradient that trains something else")
    return theta


def train(bb, steps=STEPS, *, shuffle=False, lr=0.02, **kw):
    """Learn the stack end to end on the position task, with an RLS head above it.

    RLS because it is the exact least-squares solution, so an improvement measured here is the
    stack's rather than a head optimiser catching up.
    """
    head = RecursiveLeastSquares(bb.n_outputs + 1, 6, forgetting=0.999, p0=100.0, dtype="f8")
    rng = np.random.default_rng(1234)
    errs = []
    for t in range(steps):
        img, _masks, truth = frame_at(t, seed=t, **kw)
        if shuffle:
            truth = truth[rng.permutation(len(truth))]
        x = np.concatenate([np.asarray(bb.features(img)), [1.0]])
        report = head.update(x, truth)
        errs.append(float(np.linalg.norm(report.residual)))
        # `residual` is the A PRIORI error y - prediction, so pred - y is its negation.
        d_feat = 2.0 * (theta_of(head, 6)[:-1] @ -np.asarray(report.residual))
        dws, dbs = bb.backward(img, d_feat)
        bb.apply_gradients(dws, dbs, lr)
    return float(np.mean(errs[-steps // 5:]))


SEEDS = (7, 23)


def test_the_stack_really_does_learn_the_task():
    """The precondition for the negative result below: if the gradient did not train, "no
    selectivity appeared" would say nothing. Trained reaches 0.010-0.016 against 0.53-0.56 with the
    labels shuffled."""
    for seed in SEEDS:
        bb = Backbone(PLAN, (H, W, 3), dtype="f8", seed=seed, pool=POOL)
        real = train(bb)
        bb2 = Backbone(PLAN, (H, W, 3), dtype="f8", seed=seed, pool=POOL)
        noise = train(bb2, shuffle=True)
        assert real < 0.10, f"seed {seed}: the stack did not learn the task, error {real:.4f}"
        assert noise > 5 * real, (
            f"seed {seed}: shuffled labels reached {noise:.4f} against {real:.4f}. If noise trains "
            "as well as signal, the task is being solved by something other than the input")


def test_learning_the_hidden_layers_does_not_make_units_object_selective():
    """**The finding.** Object selectivity after training is where random initialisation left it:
    0.511 against 0.532 as a mean over three seeds, and the direction differs by seed.

    Recognition is not a by-product of an objective that does not need it. Position is a linear
    readout of a spatial map, so the gradient's cheapest route is to preserve spatial layout rather
    than to become selective for what is there.

    Asserted as a *comparison against the same seed's own random init*, not against a constant,
    because the random spread (0.339 to 0.757) is wider than any training effect -- which is how a
    single seed made this look like a 1.7x improvement.
    """
    moved = []
    for seed in SEEDS:
        bb = Backbone(PLAN, (H, W, 3), dtype="f8", seed=seed, pool=POOL)
        before = selectivity(bb).mean()
        train(bb)
        after = selectivity(bb).mean()
        moved.append(after - before)
        assert after < 3.0 * before + 0.5, (
            f"seed {seed}: selectivity went {before:.3f} -> {after:.3f}. If training now produces "
            "object-selective units on this task, that is a new capability and this file's central "
            "claim is stale -- re-measure it over seeds before believing it")
    assert min(moved) < 0.0 or max(moved) < 0.35, (
        f"selectivity moved {[round(m, 3) for m in moved]} on every seed in the same direction and "
        "by a margin; the no-effect finding needs re-deriving")


def test_the_colour_fixture_cannot_fail_which_is_why_shapes_are_used():
    """The first version of this measurement gave the three objects different **colours**, and a
    random filter has a random colour preference for free: mean d-prime **1.532 before any training**,
    with 3 to 9 of 12 units already above 1.0. A metric that scores that high on an untrained network
    is measuring the input, not the learning.

    Same-coloured objects of different shapes score far lower at random, which is what leaves room
    for training to show an effect -- and it still does not.
    """
    colours = [(0.88, 0.18, 0.18), (0.18, 0.78, 0.22), (0.18, 0.22, 0.88)]
    bb = Backbone(PLAN, (H, W, 3), dtype="f8", seed=11, pool=POOL)
    by_colour = selectivity(bb, colours=colours).mean()
    by_shape = selectivity(bb).mean()
    assert by_colour > 2.0 * by_shape, (
        f"an untrained stack scores {by_colour:.3f} on differently coloured objects and "
        f"{by_shape:.3f} on same-coloured ones. If that gap has closed, the colour fixture is no "
        "longer the trivial one and this file's choice of shapes needs re-justifying")


def test_the_shuffled_control_is_vacuous_and_is_not_cited():
    """Training on shuffled labels scores a d-prime of 0.090 -- and **kills the units**, liveness
    0-58% against 75-100%. So the low score is the dead-unit explanation, and the control cannot be
    quoted as evidence that the trained selectivity is learned. Recorded here so nobody quotes it."""
    seed = 23
    bb = Backbone(PLAN, (H, W, 3), dtype="f8", seed=seed, pool=POOL)
    live_before = alive(bb)
    train(bb, shuffle=True)
    live_after = alive(bb)
    assert live_before > 0.6, f"the random init is already {live_before:.0%} alive; nothing to lose"
    assert live_after < live_before, (
        f"shuffled training left liveness at {live_after:.0%} from {live_before:.0%}. If it no "
        "longer kills units then the control is meaningful after all and should be used")


def test_backward_and_apply_gradients_are_what_makes_this_expressible():
    """A guard against the experiment silently not using the new code: without a backward pass the
    stack cannot change at all, and every number above would be the random control twice."""
    bb = Backbone(PLAN, (H, W, 3), dtype="f8", seed=7, pool=POOL)
    before = np.array(bb.state)
    train(bb, steps=5)
    assert not np.array_equal(bb.state, before), "training moved no weights"
    with pytest.raises(ValueError, match="expected d_features"):
        bb.backward(frame_at(0)[0], np.zeros(bb.n_outputs + 1))
