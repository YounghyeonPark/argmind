# SPDX-License-Identifier: Apache-2.0
"""The label regions: what they are, what they are worth, and what they are not.

`BackgroundLocator` already computes `|frame - background|` and used to throw it away. Keeping it
is what lets a caller *see* the regions the detections came from rather than only their
coordinates, and `partition` splits that mask by nearest detection.

**It is a partition of a mask, not learned segmentation.** The mask is real; the per-object split
inherits every limit already measured for the locator. Learned per-object segmentation — correlating
a remembered descriptor against the backbone's feature map — was measured and refused at 4-50%
localisation, and nothing here claims otherwise.
"""

from __future__ import annotations

import numpy as np
import pytest

from demo.label_sources import BackgroundLocator, partition
from demo.pointer import ID_COLOURS, segment_image
from demo.test_object_memory import PALETTE, orbit, scene

H, W = 72, 96


def true_mask(pts, i, sigma=0.075, thresh=0.35):
    """Ground truth: pixels where object `i` contributes at least `thresh` of its own peak."""
    yy, xx = np.mgrid[0:H, 0:W]
    px, py = pts[i]
    d = ((xx - px * W) ** 2 + (yy - py * H) ** 2) / (2 * (sigma * W) ** 2)
    return np.exp(-d) >= thresh


def region_quality(n_true, floor, frames=600, warm=100):
    """Mean best-matching IoU, recall and precision per object."""
    loc = BackgroundLocator(n_true, rate=0.03, warmup=15, suppress=0.10)
    ious, recs, precs, used = [], [], [], 0
    for k in range(frames):
        pts = [orbit(k, i, n_true) for i in range(n_true)]
        got = loc(scene(pts, PALETTE[:n_true], seed=k))
        if got is None or k < warm:
            continue
        lab = partition(loc.foreground, got, floor=floor)
        used += 1
        for i in range(n_true):
            t = true_mask(pts, i)
            best = rec = prec = 0.0
            for j in range(1, len(got) + 1):
                r = lab == j
                union = (r | t).sum()
                v = float((r & t).sum() / union) if union else 0.0
                if v > best:
                    best = v
                    rec = (r & t).sum() / max(t.sum(), 1)
                    prec = (r & t).sum() / max(r.sum(), 1)
            ious.append(best)
            recs.append(rec)
            precs.append(prec)
    assert used > 100, f"only {used} frames produced regions"
    return float(np.mean(ious)), float(np.mean(recs)), float(np.mean(precs))


def test_the_locator_keeps_the_foreground_map_it_used_to_discard():
    """The map is computed on every call anyway. Keeping it costs nothing and is the whole
    material for showing regions rather than points."""
    loc = BackgroundLocator(2, rate=0.03, warmup=10)
    assert loc.foreground is None
    for k in range(30):
        loc(scene([orbit(k, i, 2) for i in range(2)], PALETTE[:2], seed=k))
    assert loc.foreground is not None
    assert loc.foreground.shape == (H, W)
    assert float(loc.foreground.max()) > 0.0
    loc.reset()
    assert loc.foreground is None, "reset left a stale map behind"


def test_the_regions_are_real_but_rough_and_the_floor_trades_recall_for_precision():
    """Measured against ground truth, one object:

    | floor | IoU | recall | precision |
    |---|---|---|---|
    | 0.15 | 0.42 | **0.99** | 0.42 |
    | 0.25 | 0.71 | 0.94 | 0.74 |
    | 0.40 | **0.76** | 0.76 | **1.00** |

    A low floor covers the object and a great deal else; a high one is exact about what it claims
    and misses a quarter of the object. That is a trade with no free setting, which is why it is a
    flag rather than a constant.
    """
    lo = region_quality(1, 0.15)
    mid = region_quality(1, 0.25)
    hi = region_quality(1, 0.40)

    assert lo[1] > 0.9, f"a low floor no longer covers the object: recall {lo[1]:.2f}"
    assert hi[2] > 0.9, f"a high floor is no longer precise: {hi[2]:.2f}"
    assert lo[2] < hi[2] and lo[1] > hi[1], "the recall/precision trade has gone"
    assert mid[0] > 0.5, f"the default floor gives IoU {mid[0]:.2f}"


def test_region_quality_degrades_with_object_count_and_that_is_stated():
    """IoU 0.71 at one object, 0.55 at two, 0.46 at three, at the default floor. The regions
    inherit the locator's limits — it cannot count, so extra detections split one object's mask."""
    one = region_quality(1, 0.25)[0]
    three = region_quality(3, 0.25)[0]
    assert one > three * 1.3, (
        f"three objects now reach {three:.2f} against one at {one:.2f}; if the degradation has "
        "gone, the scope claimed in demo/pointer.py is understated")
    assert three > 0.3, f"three objects collapsed to IoU {three:.2f}"


def test_a_stationary_object_has_no_region_at_all():
    """The same limit that loses a stationary object from the *detector* loses it from the picture,
    and it is worth seeing there: a background estimate absorbs anything that stops moving, so its
    region simply disappears rather than degrading."""
    loc = BackgroundLocator(1, rate=0.05, warmup=15, suppress=0.10)
    covered = []
    for k in range(120):
        got = loc(scene([(0.35, 0.5)], PALETTE[:1], seed=k))     # never moves
        if got is None or k < 60:
            continue
        lab = partition(loc.foreground, got, floor=0.25)
        t = true_mask([(0.35, 0.5)], 0)
        covered.append(float(((lab > 0) & t).sum() / max(t.sum(), 1)))
    assert covered, "the fixture produced no regions at all"
    assert float(np.mean(covered)) < 0.25, (
        f"a stationary object now keeps {np.mean(covered):.0%} of its region; if a background "
        "estimate no longer absorbs it, the rate knob behaves differently than measured")


def test_partition_refuses_to_paint_where_it_should_not():
    """Three guards, and the radius one matters most: without it every noise pixel above the floor
    joins some region and the picture reads as confident where it is not."""
    fg = np.zeros((20, 30))
    fg[9:11, 4:6] = 1.0                       # a real blob near x=0.17
    fg[2, 27] = 0.9                           # a noise spike far away
    lab = partition(fg, [(0.17, 0.5)], floor=0.25, radius=0.20)
    assert lab[9, 4] == 1
    assert lab[2, 27] == 0, "a distant noise pixel was painted into the region"
    # Everything below the floor stays background whatever the radius.
    assert (lab[fg < 0.25] == 0).all()
    # And the degenerate inputs answer rather than raising.
    assert not partition(np.zeros((4, 4)), [(0.5, 0.5)]).any()
    assert not partition(fg, []).any()


def test_the_picture_colours_by_remembered_id_not_by_slot():
    """A region's colour has to follow the vocabulary, or it says nothing: a colour that changes
    when the slots reshuffle is worse than no colour."""
    lab = np.zeros((6, 8), dtype=int)
    lab[1:3, 1:3] = 1
    lab[3:5, 5:7] = 2

    a = segment_image(lab, ids=[3, 0], scale=1)
    b = segment_image(lab, ids=[0, 3], scale=1)      # the same regions, slots swapped
    assert tuple(a[2, 2]) == ID_COLOURS[3 % len(ID_COLOURS)]
    assert tuple(a[4, 6]) == ID_COLOURS[0 % len(ID_COLOURS)]
    assert tuple(b[2, 2]) == tuple(a[4, 6]), "the colour follows the slot, not the id"

    assert tuple(a[0, 0]) == (26, 26, 26)            # background stays dark
    assert segment_image(lab, ids=[0, 1], scale=3).shape == (18, 24, 3)
    # An id beyond the palette wraps rather than raising, because the vocabulary is unbounded.
    assert tuple(segment_image(lab, ids=[len(ID_COLOURS) + 1, 0], scale=1)[2, 2]) == \
        ID_COLOURS[(len(ID_COLOURS) + 1) % len(ID_COLOURS)]


def test_the_segments_window_requires_the_background_locator():
    """A frame difference gives the leading and trailing edges of one object, not its extent, so
    painting regions from it would look like segmentation and be a dipole."""
    # `main()` imports cv2 before it parses anything, and this file must keep the property the
    # README claims: the demo's tests run in a clone with no OpenCV.
    pytest.importorskip("cv2")
    from demo.pointer import main
    with pytest.raises(SystemExit, match="--segments needs --locate background"):
        main(["--segments", "--locate", "motion"])
