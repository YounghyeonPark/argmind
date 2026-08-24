# SPDX-License-Identifier: Apache-2.0
"""Shift-invariance against discrimination: a frontier no spatial pooling escapes.

**Why this was measured.** The composed recognition loop runs at 8.4 fps because it takes 147 crop
embeddings a frame, and the 21× available from one whole-frame pass **loses the identity**: a region
pooled from the feature map centres only to the map's 4-pixel stride while the descriptor needs 2.
So the question is whether a descriptor exists that tolerates 4 pixels and still tells shapes apart.

**Criteria fixed before the numbers existed**, and they are what the method needs, not what it can
reach: cosine ≥ 0.97 at a 4-pixel crop shift, and a same-shape-versus-cross-shape margin ≥ 0.10.

| descriptor | dims | 2 px | 4 px | 6 px | same | cross | margin |
|---|---|---|---|---|---|---|---|
| global pool (1×1) | 12 | 0.9993 | **0.9988** | 0.9879 | 0.9997 | 0.9957 | **+0.0040** |
| hard 2×2 | 48 | 0.9834 | 0.9484 | 0.9331 | 0.9949 | 0.9877 | +0.0071 |
| **hard 3×3 (in use)** | 108 | 0.9723 | 0.9109 | 0.8598 | 0.9897 | 0.8907 | **+0.0989** |
| soft 2×2 | 48 | 0.9884 | 0.9646 | 0.9430 | 0.9958 | 0.9887 | +0.0071 |
| soft 3×3 | 108 | 0.9810 | 0.9427 | 0.9081 | 0.9938 | 0.9585 | +0.0353 |
| soft 4×4 | 192 | 0.9691 | 0.9161 | 0.8732 | 0.9899 | 0.9280 | +0.0619 |

**Nothing reaches the corner.** Global pooling wins invariance and cannot tell a disc from a bar —
margin 0.004. The 3×3 grid in use discriminates and fails invariance. And **soft binning, the
standard remedy for exactly this quantisation, does not break the tension** — it moves along the same
frontier and slightly *worse*: soft 4×4 matches hard 3×3's invariance (0.9161 against 0.9109) with a
smaller margin (0.0619 against 0.0989).

**So the trade is a property of spatial pooling as such, not of the binning's hardness.**
Discrimination here *comes from* encoding where inside the crop the content sits, which is exactly
what a shift disturbs. Escaping it needs something outside this descriptor family — or explicit
centring, which is what the 147 crops a frame are buying.
"""

from __future__ import annotations

import numpy as np
import pytest

from argmind.backbone import Backbone, conv2d, depthwise_conv2d, relu6
from tests.test_bootstrap import H, W, render
from tests.test_recognition_memory import CROP, PLAN

INVARIANCE_NEEDED, MARGIN_NEEDED = 0.97, 0.10


def feature_map(bb, image):
    x = np.asarray(image, dtype=bb.dtype)
    for entry, w, b in zip(bb.plan, bb._w, bb._b):
        if entry[0] == "conv":
            x = relu6(conv2d(x, w, b, stride=entry[3]))
        elif entry[0] == "dw":
            x = relu6(depthwise_conv2d(x, w, b, stride=entry[2]))
        else:
            x = relu6(conv2d(x, w, b, stride=1))
    return np.asarray(x, dtype=np.float64)


def soft_pool(fm, grid):
    """Bilinear spatial binning: each map position is spread across the cells it falls between, so a
    shift moves the weights continuously instead of reassigning them."""
    h, w, c = fm.shape
    out, wsum = np.zeros((grid, grid, c)), np.zeros((grid, grid))
    for i, gy in enumerate((np.arange(h) + 0.5) / h * grid - 0.5):
        y0, fy = int(np.floor(gy)), gy - np.floor(gy)
        for j, gx in enumerate((np.arange(w) + 0.5) / w * grid - 0.5):
            x0, fx = int(np.floor(gx)), gx - np.floor(gx)
            for dy, wy in ((0, 1 - fy), (1, fy)):
                for dx, wx in ((0, 1 - fx), (1, fx)):
                    cy, cx = y0 + dy, x0 + dx
                    if 0 <= cy < grid and 0 <= cx < grid and wy * wx > 0:
                        out[cy, cx] += wy * wx * fm[i, j]
                        wsum[cy, cx] += wy * wx
    return (out / np.maximum(wsum, 1e-9)[..., None]).ravel()


def _crop(img, cx, cy, dx=0, dy=0):
    y0, x0 = int(cy * H) - CROP // 2 + dy, int(cx * W) - CROP // 2 + dx
    return np.asarray(img)[y0:y0 + CROP, x0:x0 + CROP].astype("f8")


def _unit(v):
    v = np.asarray(v, dtype=np.float64).ravel()
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def frontier_point(kind, grid, seed=23):
    """`(invariance at 4 px, discrimination margin)` for one descriptor."""
    bb = Backbone(PLAN, (CROP, CROP, 3), dtype="f8", seed=seed,
                  pool=grid if kind == "hard" else 3)
    disc = render([("disc", 0.40, 0.50)], seed=1, texture="none")
    bar = render([("bar", 0.40, 0.50)], seed=1, texture="none")
    other = render([("disc", 0.44, 0.54)], seed=7, texture="none")

    def describe(img, cx, cy, dx=0, dy=0):
        patch = _crop(img, cx, cy, dx, dy)
        return _unit(bb.features(patch) if kind == "hard"
                     else soft_pool(feature_map(bb, patch), grid))

    base = describe(disc, 0.40, 0.50)
    return (float(base @ describe(disc, 0.40, 0.50, 4, 4)),
            float(base @ describe(other, 0.44, 0.54)) - float(base @ describe(bar, 0.40, 0.50)))


@pytest.mark.parametrize("kind,grid", [("hard", 1), ("hard", 2), ("hard", 3),
                                       ("soft", 2), ("soft", 3), ("soft", 4)])
def test_NO_spatial_pooling_reaches_both_criteria(kind, grid):
    """The frontier, one point at a time. Each of these fails at least one, and the file's docstring
    carries the full table — so if one ever passes both, this test is where that is noticed."""
    inv, margin = frontier_point(kind, grid)
    assert not (inv >= INVARIANCE_NEEDED and margin >= MARGIN_NEEDED), (
        f"{kind} {grid}x{grid} reaches invariance {inv:.4f} and margin {margin:+.4f} — it meets both "
        "criteria, so a whole-frame pass is now usable and the 8.4 fps is no longer necessary")


def test_the_two_ENDS_of_the_frontier_are_where_they_were_measured():
    """The controls that make the negative result mean something: one end must win invariance and
    lose discrimination, the other the reverse. Without both, "nothing works" is indistinguishable
    from "the measurement is broken"."""
    g_inv, g_margin = frontier_point("hard", 1)
    assert g_inv > 0.99, f"global pooling's invariance fell to {g_inv:.4f}"
    assert g_margin < 0.02, (
        f"global pooling now discriminates at {g_margin:+.4f}; if that became true it is the "
        "descriptor this whole line of measurement was looking for")

    h_inv, h_margin = frontier_point("hard", 3)
    assert h_margin > 0.05, f"the 3x3 grid in use discriminates at only {h_margin:+.4f}"
    assert h_inv < 0.95, f"the 3x3 grid is now invariant at {h_inv:.4f}; the 4-pixel stride is usable"


def test_SOFT_binning_does_not_break_the_tension_and_is_worse_at_matched_invariance():
    """Soft binning is the standard remedy for this quantisation and it does not escape the trade.
    Asserted as the comparison that matters: at the same invariance it discriminates less."""
    soft_inv, soft_margin = frontier_point("soft", 4)
    hard_inv, hard_margin = frontier_point("hard", 3)
    assert abs(soft_inv - hard_inv) < 0.02, (
        f"soft 4x4 ({soft_inv:.4f}) and hard 3x3 ({hard_inv:.4f}) are no longer at comparable "
        "invariance, so the comparison below is not like for like")
    assert soft_margin < hard_margin, (
        f"soft binning now discriminates better at matched invariance ({soft_margin:+.4f} against "
        f"{hard_margin:+.4f}) — it would then be a real improvement and not a move along the same "
        "frontier")

# --------------------------------------------------------------------------------------------
# Not downsampling: measured, and the first two answers were one draw of a random seed -- 2026-08-24
#
# The owner asked whether the input and output could simply not be downsampled. Three readings were
# measured on real camera frames, and the first two produced clean-looking numbers that DID NOT
# SURVIVE a seed sweep:
#
#   1. Crop fixed at 24x24, frame raised. Adjacent-minus-far coherence read +0.0584 -> +0.0031 at
#      backbone seed 1 -- a collapse, with a mechanism to explain it (a 24 px window on a 432x576
#      frame covers 6% of the scene instead of a third, so it sees surface rather than structure).
#      Across ten seeds ON THE SYNTHETIC SCENE: ratio mean **1.07, sd 1.37**, and three of the ten
#      have NEGATIVE coherence at the low resolution -- a weak fixture, said plainly, but a
#      sign-unstable one is enough to withdraw a 19-fold monotone collapse. No effect; a draw with a
#      story attached.
#   2. Crop scaled with the frame -- the proposal read properly, the same part of the scene at more
#      detail. Tolerance-to-shift minus confusability read +0.1536 -> +0.0789 at seed 1, a halving.
#      At seed 23, synthetic scene, the same measurement TRIPLES (+0.1029 -> +0.3189).
#      Sign-unstable on one draw, which is what sent the third reading to the sweep.
#
# So the third reading is the swept one, and it is the only one that says anything. Tolerance to a
# physical displacement of 3% of the frame, against confusability with a different place -- this
# file's own frontier with resolution as the axis instead of the pooling scheme, since a 3% shift is
# 3 px in a 24 px crop and 12 px in a 96 px crop, the same fraction either way:
#
#     72x96 -> 288x384, crop 24 -> 96, demo plan, 3 real camera frames, THIRTY backbone seeds
#     margin ratio: mean 0.87, sd 0.26, min 0.47, max 1.59
#     22 of 30 seeds DOWN  ->  two-sided sign test p ~ 0.016
#
# **Established at p<0.05 and small: not downsampling makes the descriptor about 13% LESS
# discriminative on average**, because tolerance and confusability both rise and confusability rises
# faster. It is the same frontier as the pooling result, reached along a different axis.
#
# The cost needs no clock and no seed. `peak_activation_values()` goes **2,160 -> 34,560, exactly
# 16x, for the same 144 numbers out** -- 135 KB at fp32 against the declared 8 KB tier, and 33% more
# than the whole-frame backbone's own 25,920. That is the figure that decides it: a certain 16x
# working set for a measured 13% loss.
#
# **What the camera-free fixture can and cannot hold.** The synthetic 1/f scene reproduces the seed
# SPREAD -- which is the methodological finding, and the tests below assert it -- but NOT the
# direction (ten seeds split 4/6, mean 0.99). It is the weaker fixture and saying so is the point:
# the direction rests on camera frames that no test here can capture.

RES_BASE = (72, 96, 24, 8, 3)          # frame h, w, crop, candidate step, physical shift
RES_SEEDS = tuple(range(1, 11))


def natural_scene(h, w, seed=0):
    """A scene whose detail survives being sampled up: 1/f noise plus a few shapes.

    Built once at the largest size and *sampled down* to each resolution, so a higher resolution
    genuinely reveals more of the same scene rather than interpolating a smaller one. Hard-edged
    shapes alone gain nothing from more pixels, so a fixture built from them would move for a reason
    unrelated to the question.
    """
    g = np.random.default_rng(seed)
    HH, WW = 288, 384
    f = np.zeros((HH, WW, 3))
    ky, kx = np.fft.fftfreq(HH)[:, None], np.fft.fftfreq(WW)[None, :]
    k = np.sqrt(ky ** 2 + kx ** 2)
    k[0, 0] = 1.0
    for c in range(3):
        phase = g.normal(size=(HH, WW)) + 1j * g.normal(size=(HH, WW))
        f[:, :, c] = np.real(np.fft.ifft2(phase / k))
    f = (f - f.min()) / (f.max() - f.min())
    yy, xx = np.mgrid[0:HH, 0:WW]
    for cy, cx, r in [(80, 120, 34), (200, 300, 26), (120, 290, 20)]:
        f[(yy - cy) ** 2 + (xx - cx) ** 2 < r * r] = [0.95, 0.2, 0.15]
    ys = np.linspace(0, HH - 1, h).astype(int)
    xs = np.linspace(0, WW - 1, w).astype(int)
    return np.ascontiguousarray(f[np.ix_(ys, xs)])


def resolution_margin(scale, bseed, scale_the_shift=True):
    """Tolerance to a 3%-of-frame shift minus confusability with a different place, at one
    resolution and one initialisation.

    Everything scales together -- frame, crop, candidate step and the shift -- so the displacement is
    the same fraction of the crop at every point and the comparison is like for like. The demo's own
    plan, because this file has already published one figure that was the narrower fixture plan's.

    A mutant that stopped scaling the shift SURVIVED the first two tests below, so the third one
    exists: without it the high-resolution point is measured against a four-times-easier
    displacement, and every number in this block would be measuring that instead.
    """
    from demo.pointer import PLAN as DEMO_PLAN, POOL as DEMO_POOL

    h0, w0, c0, s0, d0 = RES_BASE
    h, w, C, step = h0 * scale, w0 * scale, c0 * scale, s0 * scale
    #: The displacement scales with the frame, so it is the same FRACTION of the crop at every
    #: resolution -- which is the whole basis of the comparison. `scale_the_shift=False` is the
    #: control that shows it is load-bearing, not a decoration: a fixed 3 px against a 96 px crop is
    #: a quarter of the displacement, and tolerance rises for that reason and not the descriptor's.
    shift = d0 * scale if scale_the_shift else d0
    bb = Backbone(DEMO_PLAN, (C, C, 3), dtype="f8", seed=bseed, pool=DEMO_POOL)
    frame = natural_scene(h, w)

    def describe(y, x):
        return _unit(bb.features(frame[y - C // 2:y + C // 2, x - C // 2:x + C // 2]))

    ys = range(C // 2 + shift, h - C // 2 - shift + 1, step)
    xs = list(range(C // 2 + shift, w - C // 2 - shift + 1, step))
    here = {(y, x): describe(y, x) for y in ys for x in xs}
    tolerance = [float(v @ describe(y, x + shift)) for (y, x), v in here.items()]
    keys = list(here)
    g = np.random.default_rng(0)
    confuse = [float(here[keys[a]] @ here[keys[b]])
               for a, b in g.integers(0, len(keys), size=(200, 2)) if a != b]
    return float(np.mean(tolerance)) - float(np.mean(confuse))


def _resolution_ratios():
    return np.array([resolution_margin(4, s) / resolution_margin(1, s) for s in RES_SEEDS])


def test_RESOLUTION_gives_no_RELIABLE_gain_across_initialisations():
    """Not downsampling does not reliably improve discrimination, which is what decides it against a
    16x working set. Asserted on the mean across seeds rather than on one, because one is what went
    wrong: seed 1 said the margin halves and seed 23 said it triples."""
    r = _resolution_ratios()
    assert r.mean() < 1.2, (
        f"4x resolution now improves the margin by {r.mean():.2f}x across {len(RES_SEEDS)} seeds - "
        "not downsampling would then be a real gain, and worth re-testing against its measured 16x "
        "peak-activation cost")


def test_the_seed_SPREAD_is_what_killed_the_single_seed_result():
    """The methodological finding, and the reason this block exists at all: the spread across random
    initialisations is larger than the effect, so no single-seed measurement can decide the question
    in either direction. If this ever stops being true, one seed becomes enough and the two withdrawn
    numbers above would deserve another look."""
    r = _resolution_ratios()
    assert r.std() > abs(r.mean() - 1.0), (
        f"the seed spread (sd {r.std():.2f}) no longer exceeds the effect (|mean-1| = "
        f"{abs(r.mean() - 1.0):.2f}); a single-seed measurement would now be admissible here")
    assert (r > 1).sum() not in (0, len(r)), (
        f"all {len(r)} seeds now agree on the direction; on this fixture they split, and unanimity "
        "means the fixture changed")


def test_the_COST_of_not_downsampling_is_16x_and_needs_no_clock():
    """Peak activation, not a host timing - this project's own rule is that a target figure is not a
    host figure, and `peak_activation_values()` is the same number on either.

    The point is the ratio against what comes *out*: 16x the working set for an unchanged 144-value
    descriptor.
    """
    from demo.pointer import PLAN as DEMO_PLAN, POOL as DEMO_POOL

    small = Backbone(DEMO_PLAN, (24, 24, 3), dtype="f8", seed=1, pool=DEMO_POOL)
    large = Backbone(DEMO_PLAN, (96, 96, 3), dtype="f8", seed=1, pool=DEMO_POOL)
    assert small.n_outputs == large.n_outputs == 144, (
        "the descriptor width changed, so the cost ratio below is no longer for the same output")
    assert small.peak_activation_values() == 2160
    assert large.peak_activation_values() == 34560
    assert large.peak_activation_values() == 16 * small.peak_activation_values()
    assert large.peak_activation_values() * 4 > 8 * 1024, (
        "a 96x96 crop's peak activation now fits the declared 8 KB tier at fp32; it did not")

def test_the_shift_must_SCALE_or_the_comparison_is_not_like_for_like():
    """The surviving mutant, closed. Holding the displacement at 3 px while the crop grows to 96
    makes it a quarter of the crop, so tolerance rises for a reason that has nothing to do with
    resolution — and the margin rises with it, which is exactly the flattering answer this block
    would otherwise report."""
    seed = RES_SEEDS[0]
    fair = resolution_margin(4, seed)
    flattering = resolution_margin(4, seed, scale_the_shift=False)
    assert flattering > fair, (
        f"an unscaled 3 px shift no longer flatters the high-resolution point ({flattering:+.4f} "
        f"against {fair:+.4f}); if that is true the shift scaling stopped mattering and the mutant "
        "this test was written for is no longer worth catching")

