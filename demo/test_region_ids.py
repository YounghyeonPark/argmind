# SPDX-License-Identifier: Apache-2.0
"""Every region gets a label that survives between frames, so the picture is not the head's slots.

The reported symptom was that too few labels appeared. The cause was the display painting only the
regions the head tracks -- a head with 2K outputs has K slots because that is its output width, and a
picture has no such limit. This repository's resolution of the vocabulary question is "unbounded
vocabulary, bounded concurrency"; this file is the unbounded half.

What it measures, and each number is here because none of them can fail alone:

* **the raw region index flickers** -- purity 0.66 and 2.7 ids per object, because a region's number
  comes out of a raster-order component pass
* **the cluster index cannot separate objects that look alike** -- distinctness **0.00**, invisible
  to purity, which reads a perfect 1.00 because two objects wearing one id are each perfectly pure
* **raising `k` makes the labelling worse, not better** -- five objects give 5.0 labels at purity
  1.00 with k=4, and 3.0 at purity 0.45 with k=6 or more
* **one area floor cannot serve the head and the picture** -- the split needs 0.01 or a slot loses
  its region, and the picture needs 0.03 or half its labels are background fragments

Synthetic discs on a noisy wall, so the *ordering* of the schemes is the claim, not the absolutes.
"""

from __future__ import annotations

import numpy as np
import pytest

from demo.label_sources import FrameSegmenter, RegionTracker, split_regions
from demo.pointer import label_field

H, W = 72, 96
RED = (0.85, 0.20, 0.20)
BLUE = (0.20, 0.25, 0.85)
GREEN = (0.20, 0.75, 0.25)
YELLOW = (0.85, 0.80, 0.20)

#: Each object walks its **own small orbit**, so every one of them is genuinely moving and no two
#: ever approach each other. Two earlier versions of this fixture used straight drifts and claimed in
#: a comment that they did not collide; measured, the minimum separation was 0.148 against a touching
#: threshold of 0.220 for one and 0.112 against 0.180 for the other, so objects merged near the end
#: of every run and the merge read as a labelling failure. `test_the_fixture_keeps_them_apart`
#: asserts the separation rather than a comment claiming it.
#:
#: Entries are (centre, orbit radius, phase, colour). The first two objects share a colour, which is
#: what makes the distinctness column a test of anything.
THREE = [((0.20, 0.25), 0.04, 0.0, RED),
         ((0.78, 0.25), 0.04, 2.1, RED),
         ((0.49, 0.75), 0.04, 4.2, BLUE)]
FIVE = [((0.18, 0.22), 0.04, 0.0, RED),
        ((0.80, 0.22), 0.04, 1.3, RED),
        ((0.18, 0.78), 0.04, 2.5, GREEN),
        ((0.80, 0.78), 0.04, 3.8, BLUE),
        ((0.49, 0.50), 0.04, 5.0, YELLOW)]
FRAMES = 40
PERIOD = 40.0


def positions(t, spec):
    out = []
    for (cx, cy), orbit, phase, _colour in spec:
        a = phase + 2 * np.pi * t / PERIOD
        out.append((cx + orbit * float(np.cos(a)), cy + orbit * float(np.sin(a))))
    return out


def disc(p, r):
    yy, xx = np.mgrid[0:H, 0:W]
    return np.hypot(xx - p[0] * W, yy - p[1] * H) <= r * W


def frame_at(t, spec, r=0.11, seed=0):
    img = np.full((H, W, 3), 0.35)
    masks = []
    for p, (_c, _o, _ph, colour) in zip(positions(t, spec), spec):
        m = disc(p, r)
        img[m] = colour
        masks.append(m)
    g = np.random.default_rng(seed)
    return (img + g.standard_normal(img.shape) * 0.01).astype("f4"), masks


def test_the_fixture_keeps_them_apart_and_inside_the_frame():
    """Asserted, not commented. Two earlier versions of this fixture let objects merge, and a merge
    is indistinguishable from a labelling failure in every metric below."""
    import itertools
    for spec, r in ((THREE, 0.11), (FIVE, 0.09)):
        worst = 9.9
        for t in range(FRAMES):
            pts = positions(t, spec)
            for a, b in itertools.combinations(pts, 2):
                worst = min(worst, float(np.hypot(a[0] - b[0], a[1] - b[1])))
            for x, y in pts:
                assert r < x < 1 - r and r < y < 1 - r, f"an object left the frame at t={t}"
        assert worst > 2 * r + 0.02, (
            f"objects come within {worst:.3f} of each other where {2 * r:.3f} is touching; they "
            "merge, and a merge reads as a labelling failure in every metric here")


def sequence(scheme, spec=THREE, k=4, radius=0.11, min_frac=0.01, show_frac=0.03):
    """Run a drifting scene and report what label each true object wore on each frame."""
    seg = FrameSegmenter(k)
    tracker = RegionTracker()
    carried = [[] for _ in spec]
    shown, on_object = [], []
    for t in range(FRAMES):
        img, masks = frame_at(t, spec, radius, seed=t)
        seg(img)
        rmap, centres, of_cluster = split_regions(seg.labels, k, img, edges=0.95,
                                                  min_frac=min_frac)
        field, ids = label_field(rmap, of_cluster, seg.labels, tracker, min_frac=show_frac)
        if scheme == "region index":
            ids = list(range(1, len(ids) + 1))
        elif scheme == "cluster index":
            ids = [100 + int(of_cluster[int(np.bincount(rmap[field == s + 1]).argmax()) - 1])
                   for s in range(len(ids))]
        shown.append(len(ids))
        hit = 0
        for s in range(1, len(ids) + 1):
            piece = field == s
            if any((piece & mk).sum() > 0.3 * piece.sum() for mk in masks):
                hit += 1
        on_object.append(hit / max(len(ids), 1))
        for o, mk in enumerate(masks):
            best, area = None, 0
            for s in range(1, len(ids) + 1):
                ov = ((field == s) & mk).sum()
                if ov > area:
                    best, area = s, ov
            carried[o].append(ids[best - 1] if best is not None else None)
    return carried, float(np.mean(shown)), float(np.mean(on_object)), tracker.minted


def score(carried):
    """Coverage, purity and distinctness. All three, because none of them can fail alone."""
    cov, pur, inflation = [], [], []
    for seq in carried:
        seen = [v for v in seq if v is not None]
        cov.append(len(seen) / len(seq))
        if not seen:
            pur.append(0.0)
            inflation.append(0)
            continue
        _vals, counts = np.unique(seen, return_counts=True)
        pur.append(counts.max() / len(seen))
        inflation.append(len(_vals))
    both = [(a, b) for a, b in zip(carried[0], carried[1]) if a is not None and b is not None]
    return dict(coverage=float(np.mean(cov)), purity=float(np.mean(pur)),
                distinct=sum(1 for a, b in both if a != b) / max(len(both), 1),
                ids=float(np.mean(inflation)))


def test_a_tracked_label_is_the_only_scheme_that_is_both_stable_and_distinct():
    """Three objects, two of them the same colour, k=4:

    | identity | coverage | purity | distinct | ids per object |
    |---|---|---|---|---|
    | raw region index | 1.00 | 0.66 | 1.00 | 2.7 |
    | cluster index | 1.00 | **1.00** | **0.00** | 1.0 |
    | **tracked** | 1.00 | **1.00** | **1.00** | **1.0** |

    **The cluster index scores a perfect purity and is useless**, which is the whole reason
    distinctness is measured beside it: two objects wearing one id are each perfectly pure, so purity
    alone cannot fail. Inflation cannot be dropped either -- a scheme minting a fresh label every
    frame would score perfect distinctness.
    """
    got = {name: score(sequence(name)[0])
           for name in ("region index", "cluster index", "tracked")}

    assert got["region index"]["purity"] < 0.85, (
        f"the raw region index now holds an id {got['region index']['purity']:.0%} of the time; if "
        "the component numbering has become stable the tracker's case needs re-measuring")
    assert got["region index"]["ids"] > 2.0, "the raw index stopped inflating"

    assert got["cluster index"]["distinct"] < 0.10, (
        f"the cluster index separates two same-coloured objects "
        f"{got['cluster index']['distinct']:.0%} of the time, which it cannot do by construction -- "
        "the fixture has stopped giving them the same colour")
    assert got["cluster index"]["purity"] > 0.95, (
        "the cluster index is supposed to be the case where purity looks perfect and the scheme is "
        "still useless; if its purity has fallen, this test no longer demonstrates that")

    tracked = got["tracked"]
    assert tracked["purity"] > 0.95, f"tracked purity fell to {tracked['purity']:.2f}"
    assert tracked["distinct"] > 0.95, f"tracked distinctness fell to {tracked['distinct']:.2f}"
    assert tracked["ids"] < 1.5, f"the tracker mints {tracked['ids']:.1f} ids per object"
    assert tracked["coverage"] > 0.95


def test_more_labels_come_from_splitting_and_not_from_raising_k():
    """**More clusters is worse, which is the opposite of what I assumed before measuring it.** Five
    objects in a scene, `k` swept with the head's slot count irrelevant:

    | k | labels shown | on a real object | purity | ids minted |
    |---|---|---|---|---|
    | **4** | **5.0** | 1.00 | **1.00** | **5** |
    | 6 | 3.0 | 1.00 | 0.45 | 43 |
    | 8 | 3.0 | 1.00 | 0.45 | 43 |
    | 10 | 3.0 | 1.00 | 0.45 | 43 |

    So five labels for five objects come out of a scene declared with **four** clusters -- splitting
    separates the objects that share one -- and raising `k` carves the objects across clusters
    instead, losing two of the five labels and minting 43 ids where 5 would do.

    That is the answer to "there are too few labels": paint every region rather than the head's
    slots, and leave `k` at `objects + 1`. An earlier version of this test asserted the opposite --
    that `k` did not matter at all -- from a run on a fixture whose objects collided and with the
    split's floor standing in for the picture's.

    The control is here because a flat row is this project's own signal of a broken measurement: the
    clusters must genuinely differ across `k`, and at k=10 all ten are occupied.
    """
    seen = {}
    for k in (4, 6, 10):
        carried, shown, precision, minted = sequence("tracked", spec=FIVE, k=k, radius=0.09)
        seen[k] = (shown, score(carried)["purity"], minted)
        assert precision > 0.90, f"k={k}: only {precision:.0%} of labels sat on a real object"

    assert seen[4][0] > 4.5, (
        f"k=4 shows only {seen[4][0]:.1f} labels for five objects; splitting is no longer separating "
        "the objects that share a cluster, which is what makes the whole field worth painting")
    assert seen[4][1] > 0.95 and seen[4][2] <= 8, (
        f"k=4 purity {seen[4][1]:.2f} over {seen[4][2]} minted ids")
    assert seen[10][0] < seen[4][0] - 1.0 and seen[10][1] < 0.70, (
        f"raising k no longer costs anything: {seen}. If that is real then `k` is free after all and "
        "the recommendation to leave it at objects+1 needs re-deriving")

    img, _masks = frame_at(10, FIVE, 0.09, seed=10)
    seg = FrameSegmenter(10)
    for _ in range(3):
        seg(img)
    occupied = sum(1 for j in range(10) if (seg.labels == j).any())
    assert occupied >= 8, f"only {occupied} of 10 clusters are occupied; k is not real here"


def test_the_head_and_the_picture_need_different_floors():
    """One number cannot serve both, measured in both directions.

    The **split's** floor feeds the head: raising it 0.01 -> 0.03 takes the taught centre on two
    same-coloured objects from 0.000 to **0.222**, because a slot loses its region entirely. The
    **picture's** floor keeps background fragments out. Swept on three objects:

    | show floor | labels shown | on a real object | coverage | purity |
    |---|---|---|---|---|
    | 0.01 | 7.0 | **0.47** | 1.00 | 1.00 |
    | 0.02 | 3.5 | 0.89 | 1.00 | 1.00 |
    | **0.03** | 3.2 | **0.95** | 1.00 | 1.00 |
    | 0.05 | 3.0 | 1.00 | 0.98 | **0.83** |

    0.03 is a middle rather than an optimum: 0.05 paints nothing but objects and starts losing them,
    and the purity fall at 0.05 is a label being dropped and re-minted when a region dips under.
    """
    low = sequence("tracked", show_frac=0.01)
    high = sequence("tracked", show_frac=0.03)
    assert low[1] > high[1] + 1.5, (
        f"the low floor now paints {low[1]:.1f} labels against {high[1]:.1f}; if the difference has "
        "gone then one floor would do and this pair of knobs is not justified")
    assert low[2] < 0.70 and high[2] > 0.90, (
        f"labels on a real object: {low[2]:.2f} at floor 0.01 and {high[2]:.2f} at 0.03. The whole "
        "reason the picture has its own floor is that gap")
    # And a floor set too high costs, so 0.03 is a middle rather than "higher is better".
    too_high = sequence("tracked", show_frac=0.05)
    assert score(too_high[0])["purity"] < score(high[0])["purity"], (
        "a floor of 0.05 no longer costs purity; if higher really is free then the default should "
        "move and this sweep needs redoing")

    # And the split's own floor must stay low, which is the other direction of the same finding.
    import demo.test_edge_split as edge
    assert edge.taught_centres("same colour apart", split=True, edges=0.95, min_frac=0.01) < 0.02
    assert edge.taught_centres("same colour apart", split=True, edges=0.95, min_frac=0.03) > 0.15


def test_the_tracker_follows_a_reordering_and_refuses_a_jump():
    """`limit` and `area_tol` are gates rather than costs: a candidate too far away or too different
    in size is refused outright, so a region that vanished and one that appeared are not linked
    merely because nothing better was on offer."""
    t = RegionTracker()
    assert t([(0.1, 0.1), (0.9, 0.9)], [100, 100]) == [1, 2]
    # The same two regions, listed in the other order: the labels must follow the regions.
    assert t([(0.91, 0.91), (0.11, 0.11)], [100, 100]) == [2, 1]
    # A jump past `limit` is a new region, not a continuation.
    assert t([(0.11, 0.11), (0.5, 0.11)], [100, 100]) == [1, 3]
    # An area change past `area_tol` is likewise refused.
    assert t([(0.11, 0.11)], [1000]) == [4]
    assert t.minted == 4

    t.reset()
    assert t.minted == 0 and t([(0.5, 0.5)], [10]) == [1]

    for bad in ({"limit": 0.0}, {"limit": -1.0}, {"area_tol": 0.5}):
        with pytest.raises(ValueError):
            RegionTracker(**bad)
    with pytest.raises(ValueError, match="centres against"):
        RegionTracker()([(0.5, 0.5)], [1, 2])


def test_the_unsplit_path_has_nothing_to_add_and_says_so():
    """Without splitting the clusters *are* the regions, so there is no field to build: the map
    passes through and the caller keeps using the slot ids it already had."""
    img, _masks = frame_at(0, THREE, seed=0)
    seg = FrameSegmenter(3)
    seg(img)
    field, ids = label_field(seg.labels, None, seg.labels, RegionTracker())
    assert np.array_equal(field, seg.labels) and ids == []
