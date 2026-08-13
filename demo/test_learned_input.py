# SPDX-License-Identifier: Apache-2.0
"""`--input learned`: the demo's features adapt instead of being frozen or raw.

The backbone became differentiable, so a third input source is expressible — the same convolutional
stack the demo already builds, with its weights moved by the head's own residual. This file measures
whether it earns its place and pins the trade, which cuts both ways.

Measured on a three-object tracking fixture with an RLS head, against the two sources that already
ship:

| source | tracking error | frozen, then the scene brightens x1.6 | ms/frame |
|---|---|---|---|
| raw pixels | **0.0010** | 0.6506 — **0.78x, worse than guessing** | 0.07 |
| frozen random backbone | 0.0241 | **0.0816 — 6.19x** | 0.68 |
| **learned backbone** | **0.0163** | 0.1658 — 3.05x | 0.70 + **2.42 backward** |

**None of the three dominates**, which is why this is a flag and not a new default. Learning the
features beats the frozen backbone at tracking and *costs* generalisation: the learned stack fits the
condition it trained under more tightly than a random projection does. Raw pixels win tracking by a
wide margin and fail the condition change outright.

**And it corrects a claim this project had recorded.** "What a backbone buys is invariance, and a
random one does not buy it" was measured with an NLMS head whose frozen control read 0.89x — worse
than guessing — so the comparison was between two failures. With an RLS head the frozen control is
meaningful and the random backbone reaches **6.19x** against raw pixels' **0.78x**. A random backbone
does buy invariance; the earlier measurement could not see it.
"""

from __future__ import annotations

import numpy as np
import pytest

from argmind.adapt import RecursiveLeastSquares
from argmind.backbone import Backbone
from demo.pointer import (FEATURE_LR, IN_SHAPE, PLAN, POOL, RAW_FACTOR, build, head_theta,
                          learn_features, raw_features)

H, W = IN_SHAPE[0], IN_SHAPE[1]
WALL = 0.35
COLOURS = [(0.85, 0.20, 0.20), (0.20, 0.75, 0.25), (0.20, 0.25, 0.85)]
CENTRES = [(0.24, 0.30), (0.74, 0.32), (0.48, 0.74)]
PERIOD = 97.0                                    # not a divisor of any run length below


def frame_at(t, seed=0, r=0.11, bright=1.0):
    yy, xx = np.mgrid[0:H, 0:W]
    img = np.full((H, W, 3), WALL)
    truth = []
    for i in range(3):
        a = 2 * np.pi * t / PERIOD + 2.1 * i
        cx, cy = CENTRES[i]
        p = (cx + 0.13 * float(np.cos(a)), cy + 0.13 * float(np.sin(a)))
        img[np.hypot(xx - p[0] * W, yy - p[1] * H) <= r * W] = COLOURS[i]
        truth += [p[0], p[1]]
    g = np.random.default_rng(seed)
    return (np.clip(img * bright + g.standard_normal(img.shape) * 0.01, 0, 1).astype("f4"),
            np.asarray(truth, dtype=np.float64))


def drive(source, *, steps=700, freeze_at=None, bright_after=1.0, lr=FEATURE_LR, seed=1):
    """Run one source with an RLS head. RLS because it is the exact least-squares solution, so a
    difference measured here belongs to the features rather than to a head optimiser catching up --
    and because with NLMS the frozen control is worse than guessing, which is what made an earlier
    version of this comparison vacuous."""
    bb = Backbone(PLAN, IN_SHAPE, dtype="f4", seed=seed, pool=POOL)
    width = bb.n_outputs if source in ("backbone", "learned") else len(
        raw_features(frame_at(0)[0], RAW_FACTOR, colour=True))
    head = RecursiveLeastSquares(width + 1, 6, forgetting=0.999, p0=100.0, dtype="f4")
    err, guess = [], []
    for t in range(steps):
        frozen = freeze_at is not None and t >= freeze_at
        img, truth = frame_at(t, seed=t, bright=bright_after if frozen else 1.0)
        f = (np.asarray(bb.features(img)) if source in ("backbone", "learned")
             else raw_features(img, RAW_FACTOR, colour=True))
        x = np.concatenate([f, [1.0]]).astype(np.float64)
        guess.append(float(np.linalg.norm(np.full(6, 0.5) - truth)))
        if frozen:
            err.append(float(np.linalg.norm(np.asarray(head.predict(x)) - truth)))
            continue
        report = head.update(x, truth)
        err.append(float(np.linalg.norm(report.residual)))
        if source == "learned":
            # Exactly what `main()` does: theta from the head, residual as prediction - target.
            learn_features(bb, img, head_theta(head), -np.asarray(report.residual), lr)
    n = max(len(err) // 5, 1)
    return float(np.mean(err[-n:])), float(np.mean(guess[-n:]))


def test_learning_the_features_beats_freezing_them_at_tracking():
    """0.0163 against the frozen backbone's 0.0241 — the case for the flag existing at all."""
    learned, _g = drive("learned")
    frozen, _g2 = drive("backbone")
    assert learned < frozen, (
        f"learned {learned:.4f} against frozen {frozen:.4f}; if learning the features no longer "
        "helps tracking there is no reason to offer it")
    assert learned < 0.05, f"learned reached only {learned:.4f}"


def test_it_costs_generalisation_which_is_the_other_half():
    """The unflattering half, asserted so it cannot be quietly dropped. Frozen after 500 updates and
    the scene brightened x1.6, the **random** backbone reaches 6.19x a centre guess and the learned
    one 3.05x: the learned stack fits the condition it trained under more tightly."""
    learned, lg = drive("learned", steps=700, freeze_at=500, bright_after=1.6)
    frozen, fg = drive("backbone", steps=700, freeze_at=500, bright_after=1.6)
    assert fg / frozen > lg / learned, (
        f"the frozen random backbone scores {fg / frozen:.2f}x and the learned one "
        f"{lg / learned:.2f}x under a condition change. If learning has stopped costing "
        "generalisation that is a real improvement — measure it over seeds before believing it")
    assert fg / frozen > 2.0, "the frozen control is no longer meaningfully above a centre guess"


def test_raw_pixels_still_win_tracking_and_still_fail_the_condition_change():
    """Neither of the three dominates, which is the whole reason `--input` has three values.

    This also **corrects a recorded claim**: "a random backbone does not buy invariance" was measured
    with an NLMS head whose frozen control read 0.89x, i.e. between two failures. With RLS the
    backbone reaches 6.19x where raw pixels read 0.78x — worse than guessing.
    """
    raw, _g = drive("raw")
    learned, _g2 = drive("learned")
    assert raw < learned, f"raw {raw:.4f} no longer beats learned {learned:.4f} at tracking"

    raw_frozen, rg = drive("raw", steps=700, freeze_at=500, bright_after=1.6)
    assert rg / raw_frozen < 1.0, (
        f"raw pixels now score {rg / raw_frozen:.2f}x under a brightness change; they used to be "
        "worse than a centre guess, which is what makes the backbone's 6.19x worth having")


def test_the_learner_is_gated_on_the_head_having_updated():
    """`main()` calls this only when `report.updated` is non-zero, which means it also stops while
    the guard is frozen — a fault is exactly when you do not want it pushed into the features.

    Pinned here as the *property* the gate provides, since the gate itself lives in the loop."""
    bb, head, guard, _f, _t, _m, _p = build(objects=3, source="learned")
    #: The composition carries `Bound("external_0", lo=0.35)`, and a bound is evaluated from the
    #: FIRST update -- so with no external supplied it reads 0, trips, and freezes before anything is
    #: applied. `main()` sets it after calibration; a test that forgets sees a head that never learns
    #: and blames the learner. This is the same "a bound has no not-enough-evidence-yet" shape the
    #: project recorded for the policy-collapse detector.
    guard.set_external(0, 1.0)
    img, truth = frame_at(0, seed=0)

    # At initialisation theta is all zeros, so there is no gradient into the features at all. That
    # is a real property of starting from a zero head, not a defect: the head has to learn first.
    assert np.all(head_theta(guard) == 0)
    assert learn_features(bb, img, head_theta(guard), np.full(6, 0.1)) == 0.0

    x = np.concatenate([np.asarray(bb.features(img)), [1.0]])
    guard.update(x, truth)
    theta = head_theta(guard)
    assert np.abs(theta).max() > 0.0, "one update left the head at zero"
    moved = learn_features(bb, img, theta, np.full(6, 0.1))
    assert moved > 0.0, "with a non-zero head the features still did not move"


def test_the_learner_refuses_shapes_and_rates_it_cannot_use():
    bb, _h, guard, *_rest = build(objects=3, source="learned")
    img, _truth = frame_at(0, seed=0)
    theta = np.ones((bb.n_outputs + 1, 6))
    with pytest.raises(ValueError, match="learning_rate"):
        learn_features(bb, img, theta, np.zeros(6), 0.0)
    with pytest.raises(ValueError, match="rows for a backbone"):
        learn_features(bb, img, np.ones((bb.n_outputs, 6)), np.zeros(6))
    with pytest.raises(ValueError, match="expected a residual"):
        learn_features(bb, img, theta, np.zeros(5))
    with pytest.raises(ValueError, match="row-major"):
        head_theta(_FakeHead(bb.n_outputs + 1, 6))


class _FakeHead:
    """A head whose `predict` disagrees with its own state block, which is what `head_theta`'s
    assertion exists to catch: a transposed read still produces a gradient, just for another model."""

    def __init__(self, n, m):
        self.n_features, self.n_outputs = n, m
        self.state = np.arange(n * m, dtype=np.float64)

    def predict(self, _x):
        return np.zeros(self.n_outputs) + 1.0


def test_the_learned_source_builds_and_persists_like_any_other():
    """The backbone's weights become state worth surviving a power cut only when they are learned,
    and they go through the same record format as the head — `pack` needs only `state`, `dtype`,
    `n_features` and `n_outputs`, all of which `Backbone` already had."""
    from argmind.persist import Store, unpack

    bb, _head, guard, feat, _taught, _fmap, _pipe = build(objects=3, source="learned")
    assert feat.dim == bb.n_outputs, "the learned source must feed the head the backbone's width"
    assert guard.n_features == bb.n_outputs + 1, "plus the constant"

    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        store = Store(Path(d) / "features")
        img, truth = frame_at(3, seed=3)
        guard.set_external(0, 1.0)               # or the bound freezes it on the first update
        x = np.concatenate([np.asarray(bb.features(img)), [1.0]])
        guard.update(x, truth)
        learn_features(bb, img, head_theta(guard), np.full(6, 0.2))
        moved = np.array(bb.state)
        store.save(bb, fingerprint=1234)

        bb2 = Backbone(PLAN, IN_SHAPE, dtype="f4", seed=1, pool=POOL)
        assert not np.array_equal(bb2.state, moved), "a fresh backbone already matches the learned one"
        assert store.load(bb2, fingerprint=1234) is not None
        assert np.array_equal(bb2.state, moved), "the learned weights did not come back"
        # And a record written under one configuration is refused against another -- LOUDLY.
        # It raises rather than returning None, because a block that loads and means something else
        # is the one corruption nothing downstream can detect.
        with pytest.raises(ValueError, match="fingerprint"):
            store.load(bb2, fingerprint=9999)


def test_the_applied_step_is_the_loss_gradient_and_not_merely_a_step():
    """**Accuracy cannot catch a misaligned gradient.** A mutant taking `theta[1:]` instead of
    `theta[:-1]` — dropping the first feature's row rather than the constant's — still trains, just
    against a shifted objective, and survived every accuracy assertion in this file.

    So the step is checked against central differences of the real loss, the same discipline the
    convolutional gradients themselves are held to: `apply_gradients` subtracts, so the weight change
    must be anti-parallel to the numeric gradient.
    """
    bb = Backbone(PLAN, IN_SHAPE, dtype="f8", seed=4, pool=POOL)
    rng = np.random.default_rng(5)
    img, _truth = frame_at(2, seed=2)
    img = img.astype("f8")
    theta = rng.standard_normal((bb.n_outputs + 1, 6)) * 0.3
    target = rng.standard_normal(6) * 0.2

    def loss():
        x = np.concatenate([np.asarray(bb.features(img)), [1.0]])
        return float(((x @ theta) - target) ** 2 @ np.ones(6))

    resid = (np.concatenate([np.asarray(bb.features(img)), [1.0]]) @ theta) - target
    base = np.array(bb.state, dtype=np.float64)

    # Central differences over a sample of weights: the full block is 1,240 and this is enough to
    # pin the direction, while a bad row alignment shows up in every coordinate at once.
    picks = rng.choice(base.size, size=40, replace=False)
    numeric = np.zeros(base.size)
    for i in picks:
        p = base.copy(); p[i] = base[i] + 1e-6; bb.load_state(p); hi = loss()
        p[i] = base[i] - 1e-6; bb.load_state(p); lo = loss()
        numeric[i] = (hi - lo) / 2e-6
    bb.load_state(base)

    learn_features(bb, img, theta, resid, 0.01)
    delta = np.array(bb.state, dtype=np.float64) - base

    g, d = numeric[picks], delta[picks]
    cos = float(g @ -d / max(np.linalg.norm(g) * np.linalg.norm(d), 1e-18))
    assert cos > 0.99, (
        f"the applied step is only {cos:.3f} anti-parallel to the loss gradient; it is descending "
        "something, but not this loss — check which row of theta is dropped")


def test_the_learned_source_feeds_the_backbone_and_not_raw_pixels():
    """A mutant letting `learned` fall through to `raw_features` is **invisible to width**, because
    the raw downsample was deliberately sized to the backbone's own 144. It is visible only in the
    values, so that is what is compared."""
    from demo.pointer import basis, frame_width

    bb = Backbone(PLAN, IN_SHAPE, dtype="f4", seed=1, pool=POOL)
    img, _truth = frame_at(1, seed=1)

    learned = np.asarray(basis(img, backbone=bb, source="learned"))
    frozen = np.asarray(basis(img, backbone=bb, source="backbone"))
    raw = np.asarray(basis(img, backbone=bb, source="raw"))

    assert len(learned) == len(raw) == 144, "the widths are equal, which is why values must be used"
    assert np.array_equal(learned, frozen), "learned must read the backbone, like the frozen source"
    assert not np.allclose(learned, raw), "learned is returning the raw downsample"
    assert frame_width(bb, "learned") == bb.n_outputs
    # `colour=False` is what discriminates: raw drops to 48 there while the backbone's width does
    # not depend on colour at all. At the default the two are both 144 -- RAW_FACTOR was chosen to
    # make them equal -- so a `frame_width` that forgets the learned source is invisible without it.
    assert frame_width(bb, "raw", colour=False) != bb.n_outputs
    assert frame_width(bb, "learned", colour=False) == bb.n_outputs
