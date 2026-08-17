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


def test_the_learned_source_puts_the_BACKBONE_INSIDE_THE_GUARD():
    """**Rewritten 2026-08-17, and the failure was progress.** This used to pin a property of
    `main()`'s hand-written gate: `learn_features` was called only when `report.updated`, so a
    frozen guard also stopped the features moving. The composition no longer works that way —
    `--input learned` builds a `Stack`, which *is* the adapter, so the guard covers the
    convolutional weights and the head as one state block and cannot forget to.

    The claim is therefore stronger and needs no gate at the call site at all.
    """
    from argmind.stack import Stack

    bb, learner, guard, feat, _t, _m, _p = build(objects=3, source="learned")
    assert isinstance(learner, Stack), "--input learned no longer builds a Stack"
    assert guard.state.size == bb.state.size + 870, (
        f"the guard's block is {guard.state.size}; it must cover the backbone's "
        f"{bb.state.size} weights as well as the head's")
    assert feat.dim == int(np.prod(IN_SHAPE)), (
        "the feature stream must carry the frame under this source — a Stack's basis is its "
        "layer's input, not the embedding")

    #: The bound is evaluated from the FIRST update, so with no external supplied it reads 0 and
    #: freezes before anything is applied. Third occurrence of that shape in this repository.
    guard.set_external(0, 1.0)
    img, truth = frame_at(0, seed=0)
    # + the constant, which `FrameMap(constant=True)` appends in the pipeline and a direct
    # `guard.update()` must supply itself. It arrives as a Stack *extra*: it bypasses the layers,
    # because the head's gradient with respect to a constant is not the backbone's to receive.
    x = np.concatenate([np.asarray(img, dtype=np.float32).ravel(), [1.0]])
    before = np.asarray(bb.state).copy()
    for _ in range(12):
        guard.update(x, truth)
    assert not np.array_equal(np.asarray(bb.state), before), (
        "the convolutional weights never moved through the guard")

    # And a frozen guard stops them, without the caller gating anything.
    guard.freeze("test")
    held = np.asarray(bb.state).copy()
    for _ in range(10):
        guard.update(x, truth)
    assert np.array_equal(np.asarray(bb.state), held), (
        "the layers kept learning while frozen — a fault is precisely when you do not want it "
        "pushed into the representation")


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


def test_ONE_record_carries_the_backbone_and_the_head_together():
    """**The other thing `Stack` buys, and this test used to assert the opposite.** The backbone's
    learned weights needed a second `Store` with a second fingerprint, and `main()` had to keep the
    two in step. One adapter means one record.
    """
    import tempfile
    from pathlib import Path as _P

    from argmind.persist import Store

    bb, learner, guard, _f, _t, _m, _p = build(objects=3, source="learned")
    guard.set_external(0, 1.0)
    img, truth = frame_at(3, seed=3)
    # + the constant, which `FrameMap(constant=True)` appends in the pipeline and a direct
    # `guard.update()` must supply itself. It arrives as a Stack *extra*: it bypasses the layers,
    # because the head's gradient with respect to a constant is not the backbone's to receive.
    x = np.concatenate([np.asarray(img, dtype=np.float32).ravel(), [1.0]])
    for _ in range(10):
        guard.update(x, truth)
    trained = np.asarray(guard.state).copy()
    assert not np.array_equal(trained[:bb.state.size], np.zeros(bb.state.size))

    with tempfile.TemporaryDirectory() as d:
        store = Store(_P(d) / "stack")
        store.save(guard, fingerprint=1234)

        _bb2, _l2, fresh, *_rest = build(objects=3, source="learned")
        assert not np.array_equal(np.asarray(fresh.state), trained)
        assert store.load(fresh, fingerprint=1234) is not None
        assert np.array_equal(np.asarray(fresh.state), trained), (
            "one record did not restore both levels")
        with pytest.raises(ValueError, match="fingerprint"):
            store.load(fresh, fingerprint=9999)


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


def test_the_learned_basis_is_the_FRAME_because_the_backbone_moved_inside():
    """**Inverted 2026-08-17.** This used to assert that `learned` reads the backbone like the
    frozen source does, and that a mutant falling through to `raw_features` is invisible to width
    (both are 144 by construction) so values must be compared.

    Under a `Stack` the adapter's basis is its *layer's input* — the frame — and extracting the
    embedding here would run the forward pass twice and hand the head a vector it cannot descend
    through. The width discriminates now, and the values still do: a raw downsample is 144 numbers,
    the frame is 20,736, and neither is the embedding.
    """
    from demo.pointer import basis, frame_width

    bb = Backbone(PLAN, IN_SHAPE, dtype="f4", seed=1, pool=POOL)
    img, _truth = frame_at(1, seed=1)

    learned = np.asarray(basis(img, backbone=bb, source="learned"))
    frozen = np.asarray(basis(img, backbone=bb, source="backbone"))
    raw = np.asarray(basis(img, backbone=bb, source="raw"))

    assert learned.size == int(np.prod(IN_SHAPE)), "the learned basis is not the frame"
    assert np.array_equal(learned, np.asarray(img, dtype=learned.dtype).ravel()), (
        "the learned basis is not the frame the layer consumes")
    assert frozen.size == raw.size == 144 and not np.allclose(frozen, raw), (
        "the other two sources moved; this test's controls are about them not changing")

    assert frame_width(bb, "learned") == int(np.prod(IN_SHAPE))
    assert frame_width(bb, "backbone") == bb.n_outputs
    assert frame_width(bb, "raw", colour=False) != bb.n_outputs


def test_the_calibration_is_applied_for_every_source_that_reads_the_backbone():
    """**The defect this test exists for.** The calibration branch tested `source == "backbone"`,
    so adding a third source that also reads the backbone left `--input learned` with the scale
    computed, printed, and applied nowhere — `Backbone.output_scale` stayed 1.0 and `basis()`
    ignores its `scale` argument on that branch.

    Exactly one of the two scales is ever not 1.0, which is what makes the routing checkable."""
    from demo.pointer import calibration_scales

    emb = np.linspace(0.0, 0.1, 144)
    for source in ("backbone", "learned"):
        out_scale, basis_scale = calibration_scales(source, emb, 144)
        assert out_scale > 1.0, f"{source}: the backbone was left unscaled"
        assert basis_scale == 1.0, f"{source}: the scale would be applied twice"
    out_scale, basis_scale = calibration_scales("raw", emb, 144)
    assert out_scale == 1.0 and basis_scale > 1.0, "raw features are scaled at the call site"


def test_calibrating_without_correcting_the_rate_is_worse_than_not_calibrating():
    """**Both halves belong together, and that was measured after the naive fix made things worse.**
    `output_scale` multiplies the backward pass as well as the forward one, so calibrating without
    dividing the rate takes an effective step 29x too large:

    | calibration | feature lr | vs a centre guess |
    |---|---|---|
    | not applied | 0.02 | 31.00x |
    | applied, rate unchanged | 0.02 | **1.81x** |
    | applied, rate / output_scale | 0.00068 | **36.21x** |
    """
    from demo.pointer import calibration_scales, feature_step

    def drive_scaled(correct_rate):
        bb = Backbone(PLAN, IN_SHAPE, dtype="f4", seed=1, pool=POOL)
        warm = np.asarray(bb.features(frame_at(0)[0]))
        bb.output_scale, _ = calibration_scales("learned", warm, bb.n_outputs)
        head = RecursiveLeastSquares(bb.n_outputs + 1, 6, forgetting=0.999, p0=100.0, dtype="f4")
        lr = feature_step(FEATURE_LR, bb.output_scale) if correct_rate else FEATURE_LR
        err, guess = [], []
        for t in range(500):
            img, truth = frame_at(t, seed=t)
            x = np.concatenate([np.asarray(bb.features(img)), [1.0]]).astype(np.float64)
            r = head.update(x, truth)
            err.append(float(np.linalg.norm(r.residual)))
            guess.append(float(np.linalg.norm(np.full(6, 0.5) - truth)))
            learn_features(bb, img, head_theta(head), -np.asarray(r.residual), lr)
        n = max(len(err) // 5, 1)
        return float(np.mean(err[-n:])), float(np.mean(guess[-n:]))

    bad, bg = drive_scaled(False)
    good, gg = drive_scaled(True)
    assert gg / good > 5.0 * (bg / bad), (
        f"corrected {gg / good:.2f}x against uncorrected {bg / bad:.2f}x. If the two are close now, "
        "output_scale has stopped multiplying the backward pass and this correction is unnecessary")

    assert feature_step(0.02, 1.0) == 0.02, "an unscaled backbone must leave the rate alone"
    with pytest.raises(ValueError, match="base_lr"):
        feature_step(0.0, 1.0)


def test_nothing_in_the_demo_loop_calls_the_old_chain_rule():
    """`Stack` owns the chain rule now, so `learn_features` must not be reachable from `main()`.

    It is deliberately **kept**, because the comparison fixtures above measure learned-against-
    frozen-against-raw with it and rewriting those would put recorded measurements at risk. What
    is not acceptable is leaving it wired in *and* the Stack doing the same job — nor leaving it
    present with nothing at all using it, which is the `select_newest` defect this repository
    found once already. This assertion is the difference between the two.
    """
    import inspect

    import demo.pointer as pointer

    # **The AST, not a substring.** `main()` mentions `learn_features()` inside a help string, and
    # a text search cannot tell a mention from a call — which it duly failed on.
    import ast
    tree = ast.parse(inspect.getsource(pointer.main))
    called = {n.func.id for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "learn_features" not in called, (
        "main() still calls learn_features; the Stack and the caller-side chain rule would both "
        "be stepping the same weights")
    assert callable(pointer.learn_features), (
        "learn_features was deleted — the comparison fixtures in this file use it as the "
        "reference implementation and would need rewriting with it")
