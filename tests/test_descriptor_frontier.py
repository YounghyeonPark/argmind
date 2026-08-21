# SPDX-License-Identifier: Apache-2.0
"""Shift-invariance against discrimination: a frontier no spatial pooling escapes.

**Why this was measured.** The composed recognition loop runs at 6.5 fps because it takes 147 crop
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
        "criteria, so a whole-frame pass is now usable and the 6.5 fps is no longer necessary")


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
