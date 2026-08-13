# SPDX-License-Identifier: Apache-2.0
"""The backward pass through a convolutional stack, checked against central differences.

**Why this is the only check worth having.** A wrong backward pass still trains, just worse, so
"it learns" proves nothing. This project has already paid for that lesson once:
`MultilayerPerceptron`'s exact a-posteriori law caught a missing bias term that failed by 20x and
that no accuracy comparison had noticed.

**And the reference itself has a trap that had to be diagnosed rather than assumed.** `relu6` has a
corner at 0 and another at 6, and a central difference taken across a corner averages the slope on
each side — reporting 0.5 where the analytic derivative is 0. It bites here rather than in theory,
because **biases initialise to exactly zero** and `relu6` emits exact zeros, so a receptive field of
clipped zeros gives a pre-activation of precisely 0. Measured on three plans:

| plan | pre-activations exactly at 0 | rel err on the kink | off it |
|---|---|---|---|
| one conv | 0 | 3.9e-10 | 3.9e-10 |
| separable block | 8 of 325 | **1.5e-02** | 6.5e-11 |
| the demo's plan | 3 of 220 | **5.7e-02** | 1.7e-10 |
| five layers | — | **7.6e-01** | 7.4e-11 |

So the tests below move the biases off zero before differencing, and one of them asserts that the
un-moved version *does* disagree — otherwise the next reader rediscovers this as a gradient bug.
"""

from __future__ import annotations

import numpy as np
import pytest

from argmind.backbone import (Backbone, conv2d, conv2d_backward, depthwise_conv2d,
                              depthwise_conv2d_backward, grid_avg_pool, grid_avg_pool_backward,
                              relu6, relu6_backward)


def central(loss, a, eps=1e-6):
    """Central differences in place, over the array the loss closes over.

    Passing a *copy* here is the mistake that made a first version of this read all zeros: the
    closure kept reading the original while the perturbations went somewhere nothing looked at.
    """
    g = np.zeros_like(a)
    it = np.nditer(a, flags=["multi_index"])
    while not it.finished:
        i = it.multi_index
        old = a[i]
        a[i] = old + eps
        hi = loss()
        a[i] = old - eps
        lo = loss()
        a[i] = old
        g[i] = (hi - lo) / (2 * eps)
        it.iternext()
    return g


def rel(got, ref):
    return float(np.abs(np.asarray(got) - ref).max() / max(np.abs(ref).max(), 1e-12))


def test_conv2d_gradients_match_central_differences():
    """Input, filter and bias, over shapes that exercise padding and stride separately."""
    rng = np.random.default_rng(0)
    for h, w, cin, cout, k, st in ((5, 7, 3, 4, 3, 1), (6, 6, 2, 3, 4, 2), (5, 5, 3, 2, 5, 3)):
        x = rng.standard_normal((h, w, cin))
        wt = rng.standard_normal((cout, k, k, cin)) * 0.3
        b = rng.standard_normal(cout) * 0.1
        d = rng.standard_normal(conv2d(x, wt, b, stride=st).shape)

        def loss():
            return float((conv2d(x, wt, b, stride=st) * d).sum())

        dx, dw, db = conv2d_backward(x, wt, d, stride=st)
        for name, got, ref in (("dx", dx, central(loss, x)), ("dw", dw, central(loss, wt)),
                               ("db", db, central(loss, b))):
            assert rel(got, ref) < 1e-7, f"{h}x{w} k{k} s{st}: {name} rel {rel(got, ref):.2e}"


def test_depthwise_gradients_match_central_differences():
    """The channels must stay separate in the backward direction too, which is what a mutant that
    mixes them breaks — and mixing is invisible to any single-channel fixture."""
    rng = np.random.default_rng(1)
    for h, w, c, k, st in ((5, 7, 3, 3, 1), (6, 6, 2, 3, 2), (7, 5, 4, 5, 2)):
        x = rng.standard_normal((h, w, c))
        wt = rng.standard_normal((k, k, c)) * 0.4
        b = rng.standard_normal(c) * 0.1
        d = rng.standard_normal(depthwise_conv2d(x, wt, b, stride=st).shape)

        def loss():
            return float((depthwise_conv2d(x, wt, b, stride=st) * d).sum())

        dx, dw, db = depthwise_conv2d_backward(x, wt, d, stride=st)
        for name, got, ref in (("dx", dx, central(loss, x)), ("dw", dw, central(loss, wt)),
                               ("db", db, central(loss, b))):
            assert rel(got, ref) < 1e-7, f"{h}x{w}x{c} k{k} s{st}: {name} rel {rel(got, ref):.2e}"
    assert c > 1, "a single-channel fixture cannot see channel mixing"


def test_relu6_gradient_is_zero_at_both_ends():
    """Both clips, and the upper one is the half a mutant drops. A `relu` backward pass passes every
    fixture whose activations never reach 6, which is most of them — this project measured 0.0% of
    activations reaching the upper clip at any width."""
    x = np.array([-1.0, 0.5, 3.0, 5.9, 6.5, 7.0]).reshape(1, 6, 1)
    d = np.random.default_rng(2).standard_normal(x.shape)

    def loss():
        return float((relu6(x) * d).sum())

    got = relu6_backward(x, d)
    assert rel(got, central(loss, x)) < 1e-6

    # Dropping the upper clip must be visible against the same reference.
    lower_only = d * (x > 0)
    assert rel(lower_only, central(loss, x)) > 0.5, (
        "a backward pass that forgets the upper clip agrees with central differences; this fixture "
        "has stopped putting any activation above 6 and the mutant would survive")

    # Exactly on a corner the convention is zero, stated rather than left to be guessed.
    assert relu6_backward(np.array([[[0.0, 6.0]]]), np.ones((1, 1, 2))).sum() == 0.0


def test_pool_gradient_matches_and_respects_the_remainder_tile():
    """The tiling must match the forward pass, remainder-to-the-last-tile included: a gradient that
    divides by the wrong count is the same mutant this project caught going forward."""
    rng = np.random.default_rng(3)
    divides, does_not = 0, 0
    for h, w, c, grid in ((6, 9, 2, 3), (5, 5, 3, 2), (4, 4, 2, 1), (7, 5, 2, 3)):
        x = rng.standard_normal((h, w, c))
        d = rng.standard_normal(grid_avg_pool(x, grid).shape)

        def loss():
            return float((grid_avg_pool(x, grid) * d).sum())

        got = grid_avg_pool_backward(x.shape, d, grid)
        assert rel(got, central(loss, x)) < 1e-7, f"{h}x{w} grid {grid}"
        if h % grid or w % grid:
            does_not += 1
        else:
            divides += 1
    assert divides and does_not, "both a dividing and a non-dividing grid have to be covered"
    with pytest.raises(ValueError, match="grid must be"):
        grid_avg_pool_backward((4, 4, 2), np.zeros(2), 0)
    with pytest.raises(ValueError, match="at least that big"):
        grid_avg_pool_backward((2, 2, 1), np.zeros(9), 3)


#: Each plan with input shapes and pool settings that are actually VALID for it -- a plan whose
#: total stride reduces the map below `pool` cannot be tiled, and `Backbone` accepts that
#: configuration at construction and raises on the first forward pass. Kept small on purpose: the
#: composed check differences every weight, so it is two forward passes per parameter.
PLANS = {
    "one conv": ([("conv", 4, 3, 1)], (((10, 12, 3), 2), ((9, 9, 3), 3))),
    "conv stride 2": ([("conv", 3, 3, 2)], (((10, 12, 3), 2), ((12, 12, 3), 3))),
    "separable block": ([("conv", 4, 3, 2), ("dw", 3, 1), ("pw", 5)],
                        (((10, 12, 3), 2), ((12, 18, 3), 3))),
    "the demo's plan": ([("conv", 12, 5, 4), ("dw", 3, 2), ("pw", 16)],
                        (((16, 16, 3), 2), ((24, 24, 3), 3))),
    "five layers": ([("conv", 4, 3, 2), ("dw", 3, 2), ("pw", 6), ("dw", 3, 1), ("pw", 4)],
                    (((12, 12, 3), 2), ((24, 24, 3), 3))),
}


def stack_gradient(bb, img, d):
    dws, dbs = bb.backward(img, d)
    return np.concatenate([np.concatenate([np.asarray(w).ravel(), np.asarray(b).ravel()])
                           for w, b in zip(dws, dbs)])


def central_over_state(bb, img, d, eps=1e-6):
    base = np.array(bb.state, dtype=np.float64)
    g = np.zeros(base.size)
    for i in range(base.size):
        p = base.copy()
        p[i] = base[i] + eps
        bb.load_state(p)
        hi = float((bb.features(img) * d).sum())
        p[i] = base[i] - eps
        bb.load_state(p)
        lo = float((bb.features(img) * d).sum())
        g[i] = (hi - lo) / (2 * eps)
    bb.load_state(base)
    return g


def off_kink(plan, shape, pool, seed=3, output_scale=1.0):
    """A stack whose pre-activations are not sitting on `relu6`'s corner.

    Biases initialise to **exactly zero**, and `relu6` emits exact zeros, so a receptive field of
    clipped zeros gives a pre-activation of precisely 0 — where a central difference is meaningless.
    Nudging every weight and bias moves them off it; `test_the_kink_is_why...` asserts that this
    nudge is doing real work rather than papering over a wrong gradient.
    """
    bb = Backbone(plan, shape, dtype="f8", seed=seed, pool=pool, output_scale=output_scale)
    st = np.array(bb.state, dtype=np.float64)
    bb.load_state(st + np.random.default_rng(11).standard_normal(st.size) * 0.05)
    return bb


@pytest.mark.parametrize("name", list(PLANS))
def test_the_composed_stack_agrees_with_central_differences(name):
    """Per-operation agreement is not enough: the composition is where an index convention shared
    between two layers goes wrong. Five plans, both pool settings, agreement to 1e-8 relative."""
    plan, cases = PLANS[name]
    #: `output_scale` defaults to 1.0, so a backward pass that drops it is INVISIBLE at the default
    #: -- a mutant removing it survived every case until this axis was added. Same shape as the
    #: `gain_norm` gap this project found twice: present, computed, never compared.
    for shape, pool in cases:
        for scale in (1.0, 7.5):
            bb = off_kink(plan, shape, pool, output_scale=scale)
            rng = np.random.default_rng(7)
            img = rng.standard_normal(shape)
            d = rng.standard_normal(bb.n_outputs)
            got = stack_gradient(bb, img, d)
            ref = central_over_state(bb, img, d)
            assert rel(got, ref) < 1e-8, (
                f"{name} at {shape} pool {pool} output_scale {scale}: rel {rel(got, ref):.2e}")


def test_the_kink_is_why_the_reference_needs_moving_and_not_a_wrong_gradient():
    """**The diagnosis, pinned so it is not rediscovered as a bug.** With biases left at their
    zero initialisation the composed check disagrees by 1.5e-02 to 7.6e-01; nudging them off the
    corner takes the same configurations to 6e-11. If this test ever fails it means the
    initialisation stopped producing exact zeros, and the nudge is then unnecessary rather than
    load-bearing."""
    plan, shape, pool = PLANS["five layers"][0], (12, 12, 3), 2
    rng = np.random.default_rng(7)
    img = rng.standard_normal(shape)

    on = Backbone(plan, shape, dtype="f8", seed=3, pool=pool)
    d = rng.standard_normal(on.n_outputs)
    on_err = rel(stack_gradient(on, img, d), central_over_state(on, img, d))

    off = off_kink(plan, shape, pool)
    off_err = rel(stack_gradient(off, img, d), central_over_state(off, img, d))

    assert off_err < 1e-8, f"off the kink the gradient still disagrees, {off_err:.2e}"
    assert on_err > 1e-4, (
        f"on the kink the reference now agrees too ({on_err:.2e}). Either the initialisation stopped "
        "putting pre-activations at exactly zero, or relu6 changed — in which case `off_kink` is "
        "doing nothing and this file's central claim needs re-deriving")


def test_a_gradient_step_moves_the_state_and_reduces_the_loss():
    """The gradients are usable, not merely correct: repeated steps on one target must descend.

    Not a proof of learning — one sample, one target — and deliberately so. What it catches is a
    sign error, which agreement with central differences cannot: differencing checks the gradient of
    the loss, and `apply_gradients` has to *subtract* it.
    """
    bb = Backbone(PLANS["separable block"][0], (12, 12, 3), dtype="f8", seed=5, pool=2)
    rng = np.random.default_rng(9)
    img = rng.standard_normal((12, 12, 3))
    want = rng.standard_normal(bb.n_outputs) * 0.1

    def loss():
        return float(((bb.features(img) - want) ** 2).sum())

    first = loss()
    moved = 0.0
    for _ in range(30):
        d = 2.0 * (np.asarray(bb.features(img)) - want)
        dws, dbs = bb.backward(img, d)
        moved += bb.apply_gradients(dws, dbs, 0.05)
    assert moved > 0.0, "thirty steps moved the state by exactly nothing"
    assert loss() < first * 0.9, f"loss went {first:.4f} -> {loss():.4f}; check the sign"

    with pytest.raises(ValueError, match="learning_rate"):
        bb.apply_gradients(dws, dbs, 0.0)


def test_backward_refuses_shapes_it_cannot_use():
    bb = Backbone(PLANS["one conv"][0], (8, 8, 3), dtype="f4", seed=1, pool=2)
    with pytest.raises(ValueError, match="expected an image"):
        bb.backward(np.zeros((7, 8, 3)), np.zeros(bb.n_outputs))
    with pytest.raises(ValueError, match="expected d_features"):
        bb.backward(np.zeros((8, 8, 3)), np.zeros(bb.n_outputs + 1))
    with pytest.raises(ValueError, match="expected dout"):
        conv2d_backward(np.zeros((5, 5, 2)), np.zeros((3, 3, 3, 2)), np.zeros((4, 4, 3)))
    with pytest.raises(ValueError, match="expected dout"):
        depthwise_conv2d_backward(np.zeros((5, 5, 2)), np.zeros((3, 3, 2)), np.zeros((4, 4, 2)))


def test_forward_is_still_frozen():
    """`features()` promises to change nothing, and `backward()` must not have broken that."""
    bb = Backbone(PLANS["the demo's plan"][0], (24, 32, 3), dtype="f4", seed=2, pool=3)
    before = np.array(bb.state)
    img = np.random.default_rng(4).standard_normal((24, 32, 3)).astype("f4")
    bb.features(img)
    bb.backward(img, np.ones(bb.n_outputs, dtype="f4"))
    assert np.array_equal(bb.state, before), "backward() changed the state; it must only report"
