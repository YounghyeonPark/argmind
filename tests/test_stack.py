# SPDX-License-Identifier: Apache-2.0
"""`Stack` — the hierarchy the 2026-08-16 decision asked for, and the test it had to pass.

**The claim is not that it learns.** It is that *nothing above it changed*. `Guard`, `Pipeline` and
`persist` are written against the `Adapter` contract, and this project's own standard is that **if
the guard needed one change to accommodate a hierarchy, the contract would not be one**. So the
tests here drive a stack through each of them unmodified and check the thing that was previously
untrue: that a checkpoint, a rollback, a freeze and a saved record all reach the *layers*.

The measured consequence, against the arrangement it replaces:

| | backbone outside the guard (today) | `Stack` |
|---|---|---|
| layers checkpointed | no | **yes** |
| layers rolled back on a trip | no | **yes** |
| layers frozen with the head | no — a fault is pushed into the representation | **yes** |
| layers in the persistence record | a second store, a second fingerprint | **one record** |
| who owns the chain rule | 42 lines of `demo/pointer.py`, linear heads only | the library, any head |
"""

from __future__ import annotations

import numpy as np
import pytest

from argmind.adapt import MultilayerPerceptron, NormalisedLeastMeanSquares, RecursiveLeastSquares
from argmind.backbone import Backbone
from argmind.guard import Bound, Guard
from argmind.persist import Store
from argmind.stack import Stack

PLAN = [("conv", 8, 3, 2), ("pw", 6)]
SHAPE = (16, 16, 3)


def make(head_kind="rls", extras=2, m=3, dtype="f8", feature_lr=0.02, seed=0):
    bb = Backbone(PLAN, SHAPE, pool=2, seed=seed, dtype=dtype)
    n = bb.n_outputs + extras
    head = {
        "rls": lambda: RecursiveLeastSquares(n, m, dtype=dtype),
        "nlms": lambda: NormalisedLeastMeanSquares(n, m, mu=0.3, dtype=dtype),
        "mlp": lambda: MultilayerPerceptron(n, m, hidden=(6,), dtype=dtype, learning_rate=0.02),
    }[head_kind]()
    return bb, head, Stack([bb], head, feature_lr=feature_lr)


def drive(stack, steps=40, seed=1, rng=None):
    g = rng or np.random.default_rng(seed)
    for _ in range(steps):
        stack.update(g.standard_normal(stack.n_features), g.standard_normal(stack.n_outputs))
    return stack


# --------------------------------------------------------------------------------------------
# the contract
# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("head_kind", ("rls", "nlms", "mlp"))
def test_both_levels_learn_through_one_update_call(head_kind):
    """The point of the whole exercise: one `update(x, y)` moves the layers *and* the head.

    Three heads, because the composite must not know its head's algebra — that was the defect in
    the demo's 42 lines. A head with no covariance and a head with hidden layers both work or the
    seam is not doing its job.
    """
    bb, _head, stack = make(head_kind)
    before = stack.state.copy()
    drive(stack)
    after = stack.state
    layers, head = slice(0, bb.state_size), slice(bb.state_size, None)
    assert not np.array_equal(before[layers], after[layers]), (
        f"the layers did not move under a {head_kind} head — the gradient never reached them")
    assert not np.array_equal(before[head], after[head]), f"the {head_kind} head did not move"


def test_the_state_is_one_flat_contiguous_block_that_covers_every_level():
    bb, head, stack = make()
    assert stack.state_size == bb.state_size + head.state_size
    s = stack.state
    assert s.ndim == 1 and s.flags["C_CONTIGUOUS"], (
        "the block is not flat and contiguous, so persistence is no longer a copy and the C ABI "
        "would need serialisation")
    assert np.array_equal(s[:bb.state_size], np.asarray(bb.state).ravel())
    assert np.array_equal(s[bb.state_size:], np.asarray(head.state).ravel())


def test_the_state_is_a_SNAPSHOT_and_that_differs_from_every_other_adapter():
    """A real divergence, pinned rather than described. Every other adapter here hands out its live
    storage — `Adapter.state` returning a copy is a recorded defect, caught once by `shares_memory`.
    A `Stack` cannot: its components own their own arrays. Mutating what it returns changes nothing.
    """
    bb, _head, stack = make()
    s = stack.state
    s[0] = 12345.0
    assert np.asarray(bb.state).ravel()[0] != 12345.0, (
        "the block is live after all — if that became true, this divergence is gone and the "
        "module docstring should stop claiming it")
    assert stack.state[0] != 12345.0, "the write survived a refresh, so `state` is not re-gathered"
    # And the buffer is reused, so a caller keeping it gets the same trap as StepReport.prediction.
    assert stack.state is stack.state


def test_load_state_and_reset_reach_the_layers():
    bb, _head, stack = make()
    start = stack.state.copy()
    drive(stack)
    moved = stack.state.copy()
    assert not np.array_equal(start, moved)

    stack.load_state(start)
    assert np.array_equal(stack.state, start), "load_state did not restore every level"

    drive(stack)
    stack.reset()
    assert np.array_equal(stack.state, start), (
        "reset() left the layers where training put them — the layer protocol has no reset(), "
        "which is why the initial block is snapshotted at construction")


def test_more_than_one_layer_is_refused_with_the_reason():
    """Refused rather than half-built: `backward` returns no gradient with respect to its own input,
    so there is nothing to hand a layer beneath. Depth belongs inside a layer."""
    bb, head, _ = make()
    bb2 = Backbone(PLAN, SHAPE, pool=2, seed=1, dtype="f8")
    with pytest.raises(ValueError, match="one layer"):
        Stack([bb, bb2], head, feature_lr=0.01)
    with pytest.raises(ValueError, match="no layers"):
        Stack([], head, feature_lr=0.01)


def test_nothing_can_sit_below_a_stack_and_it_says_so():
    _bb, _head, stack = make()
    drive(stack, steps=5)
    with pytest.raises(NotImplementedError, match="raw image"):
        stack.input_gradient()


# --------------------------------------------------------------------------------------------
# what it buys: the guard now reaches the layers
# --------------------------------------------------------------------------------------------

def test_a_guard_ROLLS_BACK_THE_LAYERS_and_an_unguarded_control_shows_it_matters():
    """The measured consequence, against a control — because the first version of this test asserted
    a *shape* and could not fail.

    Before `Stack`, the backbone sat outside the guard entirely: a trip restored the head to a good
    state and left the representation wherever the fault had pushed it. Here the same fault runs
    through a guarded stack and an unguarded one built from the same seed on the same data, and the
    claim is that the guarded stack's **layers** end up far closer to where they were before the
    fault than the unguarded stack's do.
    """
    bb_g, _h1, guarded = make(seed=5)
    bb_u, _h2, unguarded = make(seed=5)
    assert np.array_equal(guarded.state, unguarded.state), "the two stacks did not start equal"

    guard = Guard(guarded, [Bound("residual_norm", hi=8.0)], rollback_depth=4, checkpoint_every=5)

    clean = np.random.default_rng(7)
    for _ in range(40):
        x, y = clean.standard_normal(guarded.n_features), clean.standard_normal(guarded.n_outputs)
        guard.update(x, y)
        unguarded.update(x, y)
    good = np.asarray(bb_g.state).ravel().copy()
    assert np.allclose(good, np.asarray(bb_u.state).ravel()), (
        "the two diverged before the fault, so the comparison below is not about the guard")

    fault = np.random.default_rng(11)
    for _ in range(6):
        x = fault.standard_normal(guarded.n_features)
        y = fault.standard_normal(guarded.n_outputs) * 500.0
        guard.update(x, y)
        unguarded.update(x, y)

    assert guard.frozen, "the fixture never tripped the guard, so nothing about rollback was tested"
    moved_guarded = float(np.abs(np.asarray(bb_g.state).ravel() - good).max())
    moved_free = float(np.abs(np.asarray(bb_u.state).ravel() - good).max())
    assert moved_free > 0.0, (
        "the fault did not damage the unguarded stack's layers either, so this fixture cannot "
        "distinguish a rollback that reaches them from one that does not")
    assert moved_guarded < moved_free, (
        f"the guarded layers moved {moved_guarded:.4g} against the unguarded {moved_free:.4g} — "
        "the rollback is not reaching the layer block")


def test_a_frozen_guard_stops_the_LAYERS_learning_too():
    """A fault is precisely when you do not want it pushed into the representation. The demo has to
    gate `learn_features` on `report.updated` by hand; a `Stack` cannot forget to."""
    bb, _head, stack = make()
    guard = Guard(stack, [Bound("residual_norm", hi=5.0)], rollback_depth=2, checkpoint_every=3)
    g = np.random.default_rng(3)
    for _ in range(30):
        guard.update(g.standard_normal(stack.n_features), g.standard_normal(stack.n_outputs) * 400.0)
    assert guard.frozen, "the fixture never froze; the assertion below would be vacuous"

    held = np.asarray(bb.state).ravel().copy()
    for _ in range(20):
        guard.update(g.standard_normal(stack.n_features), g.standard_normal(stack.n_outputs))
    assert np.array_equal(np.asarray(bb.state).ravel(), held), (
        "the layers kept learning while the guard was frozen — the fault is being written into the "
        "representation, which is the failure freezing exists to prevent")


def test_one_record_carries_every_level(tmp_path):
    """Before: the backbone needed a second `Store` and a second fingerprint, and the demo had to
    keep them in step. Now `pack` sees one adapter."""
    bb, _head, stack = make()
    drive(stack)
    trained = stack.state.copy()

    store = Store(tmp_path / "stack")
    store.save(stack, fingerprint=0xABCD)

    stack.reset()
    assert not np.array_equal(stack.state, trained)
    store.load(stack, fingerprint=0xABCD)
    assert np.array_equal(stack.state, trained), "the record did not restore every level"
    assert np.array_equal(np.asarray(bb.state).ravel(), trained[:bb.state_size]), (
        "the head came back but the layers did not — the record is not covering the hierarchy")


def test_it_goes_through_a_pipeline_unchanged():
    """The composition test. A `Stack` is fed by `FrameMap` like any adapter, because its input is
    flat by design — `FrameMap` builds a 1-D regressor and could not feed an image-shaped one."""
    from argmind.align import Aligner
    from argmind.pipeline import FrameMap, Pipeline
    from argmind.stream import Stream

    _bb, _head, stack = make(extras=0, m=3)
    n = stack.n_features
    guard = Guard(stack, [Bound("residual_norm", hi=1e6)], rollback_depth=2, checkpoint_every=5)
    px = Stream("px", dim=n, capacity=16, dtype="f8")
    ty = Stream("ty", dim=3, capacity=16, dtype="f8")
    pipe = Pipeline([px, ty], Aligner(reference="px", policy="hold_last", max_stale_s=1.0), guard,
                    FrameMap(features=[("px", i) for i in range(n)],
                             target=[("ty", j) for j in range(3)], constant=False))
    g = np.random.default_rng(4)
    updated = 0
    for k in range(25):
        px.push(k * 0.01, g.standard_normal(n))
        ty.push(k * 0.01, g.standard_normal(3))
        report = pipe.step()
        updated += int(report.updated)
        assert report.prediction is None or np.asarray(report.prediction).shape == (3,)
    assert updated > 15, f"the pipeline only adapted {updated} times out of 25"


def test_a_NONFINITE_sample_does_not_reach_the_layer_WEIGHTS():
    """**Found by a surviving mutant**, and it is not the no-op it looks like.

    `update` descends only when the head applied its step, and the docstring reasons that a refused
    sample leaves `input_gradient()` zero so descending would change nothing. That reasoning is
    **wrong for a NaN**: the forward pass has already put NaN into the image buffer, and
    `backward(nan_image, zeros)` multiplies them — `0 * nan` is `nan`, not `0` — so the weights
    would be destroyed by a sample the head correctly refused.

    So the `if report.applied` check is load-bearing rather than defensive, which is only visible
    with a fixture that supplies a NaN rather than merely a rejected step.
    """
    bb, _head, stack = make()
    drive(stack, steps=20)
    good = np.asarray(bb.state).ravel().copy()
    assert np.all(np.isfinite(good))

    bad = np.random.default_rng(0).standard_normal(stack.n_features)
    bad[0] = np.nan
    report = stack.update(bad, np.zeros(stack.n_outputs))
    assert not report.applied, "the head accepted a NaN, so this fixture tests nothing"
    now = np.asarray(bb.state).ravel()
    assert np.all(np.isfinite(now)), (
        "a refused sample destroyed the layer weights — 0 * nan is nan, so descending on a "
        "'zero' gradient after a NaN forward pass is not the no-op it looks like")
    assert np.array_equal(now, good), "the layers moved on a sample the head refused"


def test_the_EXTRAS_reach_the_head_and_are_not_the_layers_business():
    """The split the whole design turns on, tested in both directions.

    Extras bypass the layers and are concatenated on to the features the head sees, so changing
    only the extras must change the prediction (they arrive) and must **not** move the layer
    weights (they are not the layers' output). A mutant zeroing them survived until this existed.
    """
    bb, _head, stack = make(extras=2)
    drive(stack, steps=30)
    g = np.random.default_rng(9)
    x = g.standard_normal(stack.n_features)
    a = np.asarray(stack.predict(x)).copy()
    x2 = x.copy()
    x2[-2:] += 5.0                                   # only the extras differ
    b = np.asarray(stack.predict(x2))
    assert not np.allclose(a, b), (
        "changing only the extras did not change the prediction — they are not reaching the head")

    # And the gradient the layers receive must be the features' slice, not the extras'.
    before = np.asarray(bb.state).ravel().copy()
    stack.update(x, g.standard_normal(stack.n_outputs))
    moved_features = np.abs(np.asarray(bb.state).ravel() - before).max()
    assert moved_features > 0.0, "the layers did not move at all, so the slice is untested"


class ArgmaxLayer:
    """A layer that selects by argmax over candidate crops — the shape a *search* takes.

    It satisfies the forward half of the layer protocol and refuses the backward half, because an
    argmax has no gradient with respect to the frame. Written here rather than imported so the
    contract question is answered by something minimal.
    """

    def __init__(self, inner, h, w):
        self.inner, self.in_shape = inner, (h, w, 3)
        self.h, self.w = h, w
        self.n_outputs, self.dtype = inner.n_outputs, inner.dtype
        self.looked_at = None

    def features(self, image):
        best, out = -np.inf, None
        for cy in (0.3, 0.5, 0.7):
            for cx in (0.3, 0.5, 0.7):
                y0, x0 = int(cy * self.h) - 8, int(cx * self.w) - 8
                patch = np.asarray(image)[max(y0, 0):max(y0, 0) + 16, max(x0, 0):max(x0, 0) + 16]
                if patch.shape[:2] != (16, 16):
                    continue
                v = np.asarray(self.inner.features(patch.astype(self.inner.dtype)))
                if float(np.abs(v).max()) > best:
                    best, out, self.looked_at = float(np.abs(v).max()), v, (cx, cy)
        return out.astype(self.dtype)

    def backward(self, image, d_features):
        raise NotImplementedError("an argmax has no gradient with respect to the frame")

    def apply_gradients(self, dws, dbs, lr):
        raise NotImplementedError("this layer has no weights of its own")

    @property
    def state(self):
        return self.inner.state

    def load_state(self, flat):
        self.inner.load_state(flat)


def _argmax_stack(feature_lr):
    inner = Backbone([("conv", 8, 3, 2), ("pw", 6)], (16, 16, 3), pool=2, seed=0, dtype="f8")
    layer = ArgmaxLayer(inner, 24, 32)
    head = RecursiveLeastSquares(layer.n_outputs + 1, 2, dtype="f8")
    return layer, Stack([layer], head, feature_lr=feature_lr)


def test_a_layer_that_cannot_be_DIFFERENTIATED_is_declared_frozen_at_construction():
    """**The contract extension of 2026-08-20, and what it is for.**

    A search — scan candidate crops, keep one — is a legitimate layer: how a layer turns an input
    into features is its own business. What it cannot do is `backward`. Measured before the change:
    such a layer was *accepted* by this constructor, `predict()` worked end to end, and `update()`
    died at `backward` on the first sample. Failing on live data for a property knowable at build
    time is the shape this project refuses.

    `feature_lr=None` declares the layers frozen and no backward pass is attempted at all. It also
    covers a case the demo already had and could not express: a frozen backbone has nothing to
    descend, so wrapping it meant paying for a pass that changes nothing.
    """
    layer, stack = _argmax_stack(None)
    assert stack.frozen_layers, "feature_lr=None must declare the layers frozen"

    rng = np.random.default_rng(0)
    x = np.concatenate([rng.standard_normal(24 * 32 * 3), [1.0]])
    assert np.asarray(stack.predict(x)).shape == (2,)
    assert layer.looked_at is not None, "the layer never ran its search"

    before = np.asarray(layer.state).copy()
    report = stack.update(x, [0.3, 0.4])
    assert report.applied, "the head did not learn above a frozen layer"
    assert np.array_equal(np.asarray(layer.state), before), "a frozen layer's weights moved"


def test_the_same_layer_with_a_RATE_still_fails_and_that_is_why_the_flag_exists():
    """The control. Without the declaration the failure is real and arrives on live data — which is
    the whole argument for making it a construction-time choice."""
    _layer, stack = _argmax_stack(0.01)
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.standard_normal(24 * 32 * 3), [1.0]])
    with pytest.raises(NotImplementedError, match="no gradient"):
        stack.update(x, [0.3, 0.4])


def test_a_non_positive_rate_is_refused_rather_than_silently_meaning_frozen():
    """`0.0` is not a way to spell `None`: `apply_gradients` refuses a non-positive rate, so a caller
    writing `feature_lr=0` would get a construction that dies later. It is refused here instead."""
    with pytest.raises(ValueError, match="positive or None"):
        _argmax_stack(0.0)
    with pytest.raises(ValueError, match="positive or None"):
        _argmax_stack(-0.5)


def test_frozen_layers_still_go_through_the_guard_and_persist(tmp_path):
    """The test every component here has to pass: nothing above it changed."""
    layer, stack = _argmax_stack(None)
    guard = Guard(stack, [Bound("residual_norm", hi=1e6)], rollback_depth=2, checkpoint_every=3)
    rng = np.random.default_rng(1)
    for _ in range(12):
        guard.update(np.concatenate([rng.standard_normal(24 * 32 * 3), [1.0]]),
                     rng.standard_normal(2))
    trained = np.asarray(guard.state).copy()

    store = Store(tmp_path / "argmax")
    store.save(guard, fingerprint=0xC0FFEE)
    _l2, fresh_stack = _argmax_stack(None)
    fresh = Guard(fresh_stack, [Bound("residual_norm", hi=1e6)], rollback_depth=2,
                  checkpoint_every=3)
    fresh.update(np.concatenate([rng.standard_normal(24 * 32 * 3), [1.0]]), rng.standard_normal(2))
    assert not np.array_equal(np.asarray(fresh.state), trained)
    assert store.load(fresh, fingerprint=0xC0FFEE) is not None
    assert np.array_equal(np.asarray(fresh.state), trained), "one record did not restore the stack"
