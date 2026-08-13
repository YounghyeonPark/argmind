"""The platform's own backbone: a frozen convolutional feature extractor.

The correctness anchor is external, as everywhere here. RLS is checked against the closed-form
weighted solution and the MLP against central differences; convolution is checked against
**the definition written as four nested loops**, which shares no implementation trick with the
im2col-and-matmul version it verifies.
"""
import math

import numpy as np
import pytest

from argmind.adapt import NormalisedLeastMeanSquares, RecursiveLeastSquares
from argmind.backbone import (Backbone, conv2d, conv2d_naive, depthwise_conv2d,
                              depthwise_conv2d_naive, global_avg_pool, grid_avg_pool,
                              mobilenet_v1, relu6)
from argmind.persist import pack, unpack


# -- the arithmetic, against the definition -----------------------------------------------

@pytest.mark.parametrize("h,w,cin,cout,k,s", [
    (7, 7, 3, 4, 3, 1), (8, 8, 5, 6, 3, 2), (9, 7, 2, 3, 5, 1),
    (6, 6, 4, 4, 1, 1), (5, 5, 1, 2, 3, 2), (4, 9, 3, 2, 3, 1),
])
def test_conv2d_matches_the_four_loop_definition(h, w, cin, cout, k, s):
    g = np.random.default_rng(h * 100 + w * 10 + k)
    x = g.standard_normal((h, w, cin))
    f = g.standard_normal((cout, k, k, cin))
    b = g.standard_normal(cout)
    fast, slow = conv2d(x, f, b, s), conv2d_naive(x, f, b, s)
    assert fast.shape == slow.shape
    scale = max(float(np.max(np.abs(slow))), 1e-30)
    assert float(np.max(np.abs(fast - slow))) / scale < 1e-13
    # A control: without it, two all-zero results would agree perfectly.
    assert scale > 1e-6


@pytest.mark.parametrize("h,w,c,k,s", [
    (7, 7, 4, 3, 1), (8, 8, 3, 3, 2), (9, 9, 5, 5, 2), (6, 4, 2, 1, 1),
])
def test_depthwise_matches_the_four_loop_definition(h, w, c, k, s):
    g = np.random.default_rng(h * 7 + c)
    x = g.standard_normal((h, w, c))
    f = g.standard_normal((k, k, c))
    b = g.standard_normal(c)
    fast, slow = depthwise_conv2d(x, f, b, s), depthwise_conv2d_naive(x, f, b, s)
    assert fast.shape == slow.shape
    scale = max(float(np.max(np.abs(slow))), 1e-30)
    assert float(np.max(np.abs(fast - slow))) / scale < 1e-13
    assert scale > 1e-6


def test_a_depthwise_convolution_does_not_mix_channels():
    """The property that makes the cost `k*k*C` instead of `k*k*C*C`, asserted rather than
    assumed from the loop bounds: changing one input channel must move only its own output."""
    g = np.random.default_rng(4)
    x = g.standard_normal((6, 6, 5))
    f = g.standard_normal((3, 3, 5))
    a = depthwise_conv2d(x, f, None, 1)
    x2 = x.copy()
    x2[:, :, 2] += 10.0
    b = depthwise_conv2d(x2, f, None, 1)
    moved = np.abs(a - b).max(axis=(0, 1))
    assert moved[2] > 1.0
    assert np.all(np.delete(moved, 2) == 0.0), "a depthwise filter mixed channels"


def test_same_padding_puts_the_extra_row_at_the_bottom():
    """TensorFlow's asymmetry: when the total padding is odd the extra row goes at the
    BOTTOM. Getting it backwards shifts every feature by a pixel and produces a network that
    still looks plausible, so it is pinned with an impulse.

    **The shape has to be chosen to discriminate.** A 4x4 input with a 3x3 kernel at stride 2
    pads by one, and an impulse at (0,0) lands inside the first window under either
    convention -- the first version of this test used exactly that and a top-heavy mutant
    survived it. 5x5 with a 4x4 kernel pads by three, which is where the two part.
    """
    x = np.zeros((5, 5, 1))
    x[0, 0, 0] = 1.0
    out = conv2d(x, np.ones((1, 4, 4, 1)), None, stride=2)[:, :, 0]
    assert out.shape == (3, 3)
    # Bottom-heavy: top pad 1, so only the first window sees the impulse. Top-heavy would put
    # two rows above it and four windows would.
    assert float(out.sum()) == 1.0, (
        f"the impulse reached {out.sum():.0f} windows; with the extra padding on top it "
        "reaches four, which is the mutant this fixture exists to kill")
    assert out[0, 0] == 1.0


def test_relu6_clips_at_both_ends():
    x = np.array([-3.0, 0.0, 2.5, 6.0, 9.0])
    assert np.array_equal(relu6(x), np.array([0.0, 0.0, 2.5, 6.0, 6.0]))


def test_global_average_pool_turns_a_map_into_a_vector():
    x = np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4)
    v = global_avg_pool(x)
    assert v.shape == (4,)
    assert np.allclose(v, x.mean(axis=(0, 1)))


# -- the backbone -------------------------------------------------------------------------

def test_the_weights_are_one_flat_block_and_persist_takes_it():
    """The same promise every adapter makes, so a backbone crosses the record format with no
    special case -- which is what makes "ship the frozen model as a record" free."""
    b = Backbone([("conv", 8, 3, 2), ("dw", 3, 1), ("pw", 12)], (16, 16, 3), dtype="f8")
    want = (8 * 3 * 3 * 3 + 8) + (3 * 3 * 8 + 8) + (12 * 1 * 1 * 8 + 12)
    assert b.state_size == want == b.state.size
    assert b.n_outputs == 12
    assert b.n_features == 16 * 16 * 3

    rec = pack(b, sequence=1, fingerprint=0xC0FFEE)
    assert unpack(rec).state_size == want
    assert np.array_equal(unpack(rec).state, b.state)

    other = Backbone([("conv", 8, 3, 2), ("dw", 3, 1), ("pw", 12)], (16, 16, 3),
                     dtype="f8", seed=99)
    assert not np.allclose(other.state, b.state)
    other.load_state(b.state)
    img = np.random.default_rng(2).standard_normal((16, 16, 3))
    assert np.array_equal(other.features(img), b.features(img))

    with pytest.raises(ValueError, match="state block must have"):
        b.load_state(np.zeros(want + 1))
    bad = np.array(b.state)
    bad[0] = np.nan
    with pytest.raises(ValueError, match="not finite"):
        b.load_state(bad)


def test_the_backbone_is_frozen():
    """It has no `update`, and running it must not move a weight. A feature extractor that
    drifted would make every checkpoint in the guard describe a different model."""
    b = Backbone([("conv", 6, 3, 2), ("dw", 3, 2), ("pw", 8)], (16, 16, 3), dtype="f8")
    assert not hasattr(b, "update")
    before = np.array(b.state)
    g = np.random.default_rng(1)
    for _ in range(5):
        b.features(g.standard_normal((16, 16, 3)))
    assert np.array_equal(np.asarray(b.state), before)
    assert not b.state.flags.writeable


def test_the_peak_activation_pair_is_why_this_is_not_core_tier():
    """Decided by arithmetic before any code: the smallest useful vision backbone needs
    54 KB of activation working set in int8 against a declared 8 KB floor for the whole
    library. Reported by the module so the figure is not folded into prose."""
    m = mobilenet_v1(0.25, (96, 96, 3), dtype="f4")
    peak = m.peak_activation_values()
    assert peak == 55296
    assert peak * 1 > 6 * 8 * 1024, "int8 activations would now fit the core floor"
    assert m.n_outputs == 256

    wide = mobilenet_v1(0.25, (224, 224, 3), dtype="f4")
    assert wide.peak_activation_values() > 5 * peak


def test_mobilenet_produces_a_full_rank_embedding():
    """An untrained deep stack could emit a constant, which is a basis of rank one and would
    make everything downstream fit noise. It does not: the magnitude decays with depth but
    the directions stay distinct."""
    m = mobilenet_v1(0.25, (96, 96, 3), dtype="f8", seed=1)
    g = np.random.default_rng(3)
    v = np.array([m.features(g.standard_normal((96, 96, 3))) for _ in range(6)])
    assert v.shape == (6, 256)
    assert np.all(np.isfinite(v))
    assert np.linalg.matrix_rank(v) == 6, "the embedding is degenerate across images"
    assert np.mean(v.max(axis=0) == 0.0) < 0.2, "too many units are dead for every image"


# -- the scale, which is the finding that came out of building this -----------------------

def test_the_embedding_must_be_scaled_and_the_module_reports_the_constant():
    """`p0` has units, and an untrained backbone's embedding is ~1e-4.

    Measured: raw, an RLS head scores 0.998 of a constant predictor -- it learns nothing.
    Scaled by the calibrated constant it reaches 0.0015. The failure is the PRIOR against the
    input scale rather than the algorithm: the gain is `p0*x/(1 + p0*|x|^2)`, so at
    `|x| ~ 1e-4` the data term is `1e-8 * p0` and the default 100 is eight orders of
    magnitude short.

    Scaling the *target* is free for RLS and NLMS; scaling the *input* is not, and a
    prediction of mine that conflated the two was wrong.
    """
    m = mobilenet_v1(0.25, (96, 96, 3), dtype="f8", seed=1)
    g = np.random.default_rng(5)
    imgs = g.standard_normal((40, 96, 96, 3))
    e = np.array([m.features(im) for im in imgs])
    assert float(np.abs(e).max()) < 1e-2, (
        "the untrained embedding is no longer small, so this fixture cannot show the effect")

    scale = m.calibrate_scale(imgs)
    assert scale > 100.0
    m.output_scale = scale
    scaled = np.array([m.features(im) for im in imgs])
    assert abs(float(scaled.std()) - 1.0) < 0.05

    # ...and it is the difference between learning and not.
    w = np.random.default_rng(9).standard_normal(m.n_outputs)
    y = e @ w

    def head_error(x):
        feats = np.concatenate([x, np.ones((len(x), 1))], axis=1)
        a = RecursiveLeastSquares(feats.shape[1], 1, forgetting=1.0, p0=100.0, dtype="f8")
        for _ in range(6):
            for xi, yi in zip(feats, y):
                a.update(xi, [yi])
        pred = np.array([float(a.predict(xi)[0]) for xi in feats])
        return float(np.sqrt(np.mean((pred - y) ** 2)) / y.std())

    assert head_error(e) > 0.9, (
        "the raw embedding trained fine, so the scale finding is not demonstrated here")
    assert head_error(scaled) < 0.1, (
        "scaling did not rescue the head; the calibration constant is not doing its job")

    with pytest.raises(ValueError, match="does not vary"):
        m.calibrate_scale([imgs[0]] * 3)
    with pytest.raises(ValueError, match="output_scale must be positive"):
        Backbone([("conv", 4, 3, 1)], (8, 8, 3), output_scale=0.0)


def test_the_input_scale_matters_and_the_target_scale_does_not():
    """The separation the prediction got wrong, pinned in both directions.

    `p0` is a prior on the coefficients and the gain is `p0*x/(1 + p0*|x|^2)`, so a small
    INPUT starves it while a small TARGET is absorbed into theta. Measured at 1e-4: target
    scaling changes nothing at all, input scaling takes RLS from 2.6e-6 to 0.996 -- and
    raising `p0` to 1e10 puts it back to 2.6e-6, which is what says the algorithm is fine
    and the prior was wrong.
    """
    n, count = 12, 1500
    g = np.random.default_rng(4)
    x0 = g.standard_normal((count, n))
    w = g.standard_normal(n)
    y0 = x0 @ w

    def err(xs, ys, p0):
        a = RecursiveLeastSquares(n, 1, forgetting=1.0, p0=p0, dtype="f8")
        for xi, yi in zip(xs, ys):
            a.update(xi, [yi])
        pred = xs @ np.asarray(a.state)[:n]
        return float(np.sqrt(np.mean((pred - ys) ** 2)) / max(ys.std(), 1e-300))

    base = err(x0, y0, 100.0)
    assert base < 1e-4
    assert abs(err(x0, y0 * 1e-4, 100.0) - base) < 1e-6, (
        "scaling the target changed the fit, so it is not free after all")
    assert err(x0 * 1e-4, y0 * 1e-4, 100.0) > 0.5, (
        "scaling the input did not starve the prior; the fixture no longer shows the effect")
    assert err(x0 * 1e-4, y0 * 1e-4, 1e10) < 1e-4, (
        "raising p0 did not recover it, so the cause is not the prior against the scale")

    # NLMS divides by the input power, so it should survive what RLS does not.
    def nlms_err(xs, ys):
        a = NormalisedLeastMeanSquares(n, 1, mu=0.5, dtype="f8")
        for _ in range(4):
            for xi, yi in zip(xs, ys):
                a.update(xi, [yi])
        pred = xs @ np.asarray(a.state)[:n]
        return float(np.sqrt(np.mean((pred - ys) ** 2)) / max(ys.std(), 1e-300))

    assert nlms_err(x0 * 1e-4, y0) < 1e-3, "NLMS is not input-scale robust after all"


def test_the_backbone_feeds_the_adapters_the_library_already_has():
    """The seam the whole scope decision turns on, at the width that decided the C ABI: the
    embedding is 256 wide, which is 1 KB of NLMS state and would be 259 KB of RLS."""
    m = mobilenet_v1(0.25, (96, 96, 3), dtype="f4", seed=1)
    g = np.random.default_rng(11)
    imgs = g.standard_normal((30, 96, 96, 3))
    m.output_scale = m.calibrate_scale(imgs)

    n = m.n_outputs + 1
    head = NormalisedLeastMeanSquares(n, 1, mu=0.5, dtype="f4")
    assert head.state.nbytes == n * 4
    assert head.state.nbytes < 2 * 1024
    rls_state = (n + n * n) * 4
    assert rls_state > 200 * 1024, (
        f"RLS at this width is {rls_state} bytes; if that has shrunk the argument for "
        "putting NLMS in the C ABI needs re-deriving")

    for im in imgs:
        x = np.concatenate([m.features(im), [1.0]])
        r = head.update(x, [float(x[:3].sum())])
        assert r.applied
    assert head.updates == len(imgs)


def test_the_layer_plan_is_declared_and_bad_entries_are_refused():
    with pytest.raises(ValueError, match="unknown layer"):
        Backbone([("attention", 8)], (8, 8, 3))
    with pytest.raises(ValueError, match="in_shape must be"):
        Backbone([("conv", 4, 3, 1)], (8, 8))
    b = Backbone([("conv", 4, 3, 1)], (8, 8, 3), dtype="f8")
    with pytest.raises(ValueError, match="expected an image of"):
        b.features(np.zeros((8, 8, 4)))
    with pytest.raises(ValueError, match="input channels"):
        conv2d(np.zeros((5, 5, 3)), np.zeros((2, 3, 3, 4)))
    with pytest.raises(ValueError, match="depthwise filter expects"):
        depthwise_conv2d(np.zeros((5, 5, 3)), np.zeros((3, 3, 4)))
    assert math.isclose(float(np.sum(b.features(np.zeros((8, 8, 3))) >= 0.0)), 4.0)


def test_the_plan_stride_reaches_the_convolution():
    """`features()` pools its map away, so a stride the plan declares and the forward pass
    ignores produces the SAME output shape and the same number of features -- a mutant
    forcing stride 1 survived every other test here. Pinned by composing the layer
    independently and by the spatial size the plan implies.
    """
    g = np.random.default_rng(6)
    img = g.standard_normal((16, 16, 3))
    b = Backbone([("conv", 5, 3, 2)], (16, 16, 3), dtype="f8", seed=2)
    w = np.asarray(b.state)[:5 * 3 * 3 * 3].reshape(5, 3, 3, 3)
    bias = np.asarray(b.state)[5 * 3 * 3 * 3:]

    want = global_avg_pool(relu6(conv2d(img, w, bias, stride=2)))
    assert np.allclose(b.features(img), want)
    # ...and stride 1 over the same weights gives a different answer, so the comparison is
    # not passing because the stride happens not to matter on this input.
    other = global_avg_pool(relu6(conv2d(img, w, bias, stride=1)))
    assert not np.allclose(want, other)

    # The declared memory follows the stride too, which is the figure that decides the tier.
    assert (Backbone([("conv", 5, 3, 2)], (16, 16, 3)).peak_activation_values()
            < Backbone([("conv", 5, 3, 1)], (16, 16, 3)).peak_activation_values())


# -- the readout, which decides what a spatial task can do at all --------------------------

def test_grid_pooling_keeps_where_and_global_pooling_does_not():
    """Not a refinement. Global average pooling **discards position by construction**, and on
    a shallow random backbone a moving blob reads **1.076** of its own standard deviation
    globally -- worse than a constant predictor -- against **0.231** on a 3x3 grid. Brightness
    is unaffected either way (0.009 against 0.023) because it is a global quantity.

    Shortened here to the property rather than the regression: the tiles must differ from each
    other when the input has structure, and their mean must be the global answer.
    """
    g = np.random.default_rng(1)
    x = g.standard_normal((12, 12, 4))
    x[:6, :, :] += 5.0                       # a top-heavy image

    glob = grid_avg_pool(x, 1)
    tiles = grid_avg_pool(x, 3).reshape(9, 4)
    assert glob.shape == (4,) and tiles.shape == (9, 4)
    assert np.allclose(tiles.mean(axis=0), glob), (
        "the tiles do not average to the global answer, so one of the two is wrong")
    assert tiles[0, 0] > tiles[8, 0] + 1.0, (
        "the top and bottom tiles agree on a top-heavy image, so the grid is not spatial")

    # ...and the global readout cannot tell the image from its vertical flip, which is the
    # whole reason a spatial task needs the grid.
    flipped = x[::-1, :, :]
    assert np.allclose(grid_avg_pool(flipped, 1), glob)
    assert not np.allclose(grid_avg_pool(flipped, 3), grid_avg_pool(x, 3))

    b = Backbone([("conv", 6, 3, 2)], (16, 16, 3), dtype="f8", seed=1, pool=3)
    assert b.n_outputs == 6 * 9
    img = g.standard_normal((16, 16, 3))
    assert b.features(img).shape == (54,)
    with pytest.raises(ValueError, match="grid must be"):
        grid_avg_pool(x, 0)
    with pytest.raises(ValueError, match="at least that big"):
        grid_avg_pool(np.zeros((2, 2, 3)), 3)


def test_a_random_backbone_dies_with_depth():
    """The measurement that decided the demo's architecture, and it is not obvious: random
    convolutional features work SHALLOW and carry nothing deep.

    Predicting overall brightness from a random-weight stack, RMSE over the target's own
    standard deviation: **0.009 at one layer**, 0.011 at two, 0.010 at four, 0.027 at eight,
    and **1.119 at twenty-seven** -- worse than a constant. Each relu6 halves the signal and
    the random mixing decorrelates what is left. An untrained deep backbone is not a basis.
    """
    rng = np.random.default_rng(0)
    n, hh = 60, 48

    def scene(bright):
        return np.full((hh, hh, 3), bright) + rng.standard_normal((hh, hh, 3)) * 0.05

    bright = rng.uniform(0.1, 0.9, n)
    imgs = [scene(b) for b in bright]

    def brightness_error(plan):
        bb = Backbone(plan, (hh, hh, 3), dtype="f8", seed=1, pool=1)
        x = np.array([bb.features(im) for im in imgs])
        x = x / max(float(x.std()), 1e-30)
        x = np.concatenate([x, np.ones((n, 1))], axis=1)
        m = NormalisedLeastMeanSquares(x.shape[1], 1, mu=0.5, dtype="f8")
        for _ in range(40):
            for xi, yi in zip(x[:45], bright[:45]):
                m.update(xi, [yi])
        p = np.array([float(m.predict(xi)[0]) for xi in x[45:]])
        return float(np.sqrt(np.mean((p - bright[45:]) ** 2)) / bright[45:].std())

    shallow = brightness_error([("conv", 16, 5, 4)])
    deep_plan = [("conv", 8, 3, 2)]
    for i in range(8):
        deep_plan += [("dw", 3, 2 if i in (0, 2) else 1), ("pw", 16)]
    deep = brightness_error(deep_plan)

    assert shallow < 0.2, f"a one-layer random backbone cannot read brightness: {shallow:.3f}"
    assert deep > shallow * 3.0, (
        f"depth no longer destroys the signal ({shallow:.3f} against {deep:.3f}); if random "
        "deep features have become usable the demo's architecture should be re-derived")


def test_learning_the_hidden_layers_is_a_compute_question_not_a_memory_one():
    """**This test exists because it corrects a claim CLAUDE.md had recorded** -- that end-to-end
    on-device training "is not available under 4.2 MB and no heap". The arithmetic says otherwise for
    the smallest useful vision backbone in fp32:

    | | values | fp32 |
    |---|---|---|
    | weights | 210,336 | 821.6 KB |
    | weight gradients | 210,336 | 821.6 KB |
    | retained activations | 259,200 | 1,012.5 KB |
    | **total** | | **2.59 MB** against an STM32N6's 4.2 MB |

    What blocks it is that the NPU cannot train, so a backward pass runs on the M55 in software, and
    no throughput figure for that exists. The memory objection was wrong; the timing one is open.
    """
    m = mobilenet_v1(0.25, (96, 96, 3), dtype="f4")
    infer = m.peak_activation_values()
    train = m.training_activation_values()
    assert train > infer, "retaining every activation cannot cost less than the largest pair"
    assert train / infer == pytest.approx(4.7, abs=0.3), (
        f"the training-to-inference activation ratio is {train / infer:.1f}x, not the 4.7x recorded; "
        "the plan changed and the tier arithmetic needs redoing")

    total_bytes = (2 * m.state.size + train) * 4
    assert total_bytes < 4.2e6, (
        f"learning this backbone needs {total_bytes / 1e6:.2f} MB, over an STM32N6's 4.2 MB. If that "
        "is real then the memory objection is back and CLAUDE.md's correction is wrong")
    assert total_bytes > 2.0e6, (
        f"only {total_bytes / 1e6:.2f} MB; if it has become cheap, say by how much rather than "
        "leaving the 2.59 MB figure stale")

    # And at 224x224 it does not fit, which is what makes the 96x96 result a real constraint rather
    # than a blanket permission.
    wide = mobilenet_v1(0.25, (224, 224, 3), dtype="f4")
    wide_bytes = (2 * wide.state.size + wide.training_activation_values()) * 4
    assert wide_bytes > 4.2e6, (
        f"224x224 training now fits in {wide_bytes / 1e6:.2f} MB; the resolution ceiling has moved")

    # The demo's own shallow plan is nearly free to train, because its first buffer dominates.
    shallow = Backbone([("conv", 12, 5, 4), ("dw", 3, 2), ("pw", 16)], (72, 96, 3),
                       dtype="f4", seed=1, pool=3)
    ratio = shallow.training_activation_values() / shallow.peak_activation_values()
    assert ratio == pytest.approx(1.1, abs=0.15), (
        f"the shallow plan's ratio is {ratio:.2f}x, not the 1.1x recorded")
