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


def test_more_labels_come_from_splitting_and_k_barely_matters():
    """**Five labels for five objects come out of a scene declared with four clusters**, because
    splitting separates the objects that share one. `k` swept, with the head's slot count irrelevant:

    | k | labels | on a real object | purity | ids minted |
    |---|---|---|---|---|
    | 4 | 5.0 | 1.00 | 1.00 | 5 |
    | 6 | 5.0 | 1.00 | 1.00 | 5 |
    | 8 | 5.0 | 1.00 | 1.00 | 5 |
    | 10 | 5.0 | 1.00 | 1.00 | 5 |

    **An earlier version of this test asserted that raising `k` was WORSE -- 3.0 labels at purity
    0.45 over 43 minted ids from k=6 -- and that was an artefact of a defect, not a property of
    `k`.** `label_field` chose its background from the clusters that had produced a surviving region;
    at high `k` the wall splits into clusters whose fragments all fall under the split's floor, so the
    background was mis-identified and real objects were dropped and re-minted. With the background
    taken from the largest cluster in the frame, `k` is flat across 4 to 10.

    A flat row is this project's own signal of a broken measurement, so the control is still here:
    the clusters must genuinely differ across `k`, and at k=10 all ten are occupied. What is flat is
    the *outcome*, not the input.
    """
    seen = {}
    for k in (4, 6, 10):
        carried, shown, precision, minted = sequence("tracked", spec=FIVE, k=k, radius=0.09)
        seen[k] = (shown, score(carried)["purity"], minted)
        assert precision > 0.90, f"k={k}: only {precision:.0%} of labels sat on a real object"
        assert shown > 4.5, (
            f"k={k} shows only {shown:.1f} labels for five objects; splitting is no longer "
            "separating the objects that share a cluster")
        assert seen[k][1] > 0.95 and seen[k][2] <= 8, (
            f"k={k}: purity {seen[k][1]:.2f} over {seen[k][2]} minted ids. If raising k has started "
            "costing again, check the background selection before blaming k -- that is what it was "
            "the first time")

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


# --- Keeping a label across an object hiding and coming back -------------------------------------


def render(objs, seed=0, r=0.11):
    """`objs` is a list of (centre, colour); returns the frame and one mask each."""
    img = np.full((H, W, 3), 0.35)
    masks = []
    for p, colour in objs:
        m = disc(p, r)
        img[m] = colour
        masks.append(m)
    g = np.random.default_rng(seed)
    return (img + g.standard_normal(img.shape) * 0.01).astype("f4"), masks


def play(script, *, appearance, keep=30, tol=0.25, show_frac=0.03, seed0=0):
    """Run a scripted scene and report the label each tagged object wore on each frame."""
    k = max(len(objs) for objs, _tags in script) + 1
    seg = FrameSegmenter(k)
    tracker = RegionTracker(keep=keep, tol=tol)
    worn = {}
    for t, (objs, tags) in enumerate(script):
        img, masks = render(objs, seed=t + seed0)
        seg(img)
        rmap, centres, of_cluster = split_regions(seg.labels, k, img, edges=0.95, min_frac=0.01)
        # `label_field` is the shipped assembly; passing the frame is what turns appearance on.
        field, ids = label_field(rmap, of_cluster, seg.labels, tracker, min_frac=show_frac,
                                 frame=img if appearance else None)
        for tag, mk in zip(tags, masks):
            best, area = None, 0
            for slot in range(1, len(ids) + 1):
                ov = ((field == slot) & mk).sum()
                if ov > area:
                    best, area = slot, ov
            worn.setdefault(tag, {})[t] = ids[best - 1] if best is not None else None
        for tag in worn:
            worn[tag].setdefault(t, None)
    return worn


def hide_and_return(gap, *, place, substitute=None, before=12, after=12):
    """'a' is present, hides for `gap` frames, then something comes back. The tags say what."""
    rest = [((0.75, 0.30), GREEN), ((0.50, 0.80), BLUE)]
    home, away = (0.22, 0.30), (0.28, 0.72)
    script = [([(home, RED)] + rest, ["a", "b", "c"]) for _ in range(before)]
    #: A genuine ABSENCE. An earlier version of this fixture put the newcomer at `home` during the
    #: gap, so a region sat there continuously and merely changed colour -- position matching linked
    #: them with no vanish at all, and every mechanism including "no memory" read as stealing.
    script += [(list(rest), ["b", "c"]) for _ in range(gap)]
    if substitute is not None:
        script += [([(home, substitute)] + rest, ["x", "b", "c"]) for _ in range(after)]
    else:
        script += [([(home if place == "home" else away, RED)] + rest, ["a", "b", "c"])
                   for _ in range(after)]
    return script, before, gap


def came_back_as_itself(worn, before, gap, tag="a"):
    was = worn[tag].get(before - 1)
    seen = [worn[tag].get(t) for t in range(before + gap, before + gap + 12)]
    seen = [v for v in seen if v is not None]
    return bool(seen) and all(v == was for v in seen)


def stole_a_dead_label(worn, before, gap):
    was = worn["a"].get(before - 1)
    return any(worn.get("x", {}).get(t) == was for t in range(before + gap, before + gap + 12))


def test_a_hidden_object_gets_its_own_label_back_wherever_it_reappears():
    """Appearance, not position. Recovery of 'a' after hiding for `gap` frames:

    | returns | no memory | appearance |
    |---|---|---|
    | to the same place | no | **yes** |
    | somewhere else | no | **yes** |
    | after a gap past `keep` | no | no |

    The "no memory" column is the control and is what shipped before: the tracker held one frame, so
    a vanished label was gone.
    """
    for place in ("home", "elsewhere"):
        for gap in (1, 3, 10):
            script, before, g = hide_and_return(gap, place=place)
            assert came_back_as_itself(play(script, appearance=True), before, g), (
                f"returning {place} after {gap} frames did not reclaim the label")
            assert not came_back_as_itself(play(script, appearance=False), before, g), (
                f"a frame-to-frame tracker recovered a {gap}-frame absence returning {place}; it "
                "has no memory to do that with, so this fixture is not testing what it claims")

    # `keep` is a forget window and a real one, or the pool would grow without bound.
    script, before, g = hide_and_return(20, place="home")
    assert came_back_as_itself(play(script, appearance=True, keep=30), before, g)
    assert not came_back_as_itself(play(script, appearance=True, keep=5), before, g), (
        "a 20-frame absence recovered with a 5-frame forget window; `keep` is not being enforced")


def test_a_newcomer_does_not_inherit_a_dead_label_unless_it_is_indistinguishable():
    """The other direction, and the reason position coasting is not offered. A different object
    arrives where 'a' was:

    | newcomer | position coasting (not shipped) | appearance |
    |---|---|---|
    | a yellow disc | **steals the label** | refuses |
    | an identical red disc | steals | steals |

    The second row is a limit rather than a defect: an identical object arriving where an identical
    object was is indistinguishable by construction, and this says so instead of pretending.
    """
    for gap in (3, 10):
        script, before, g = hide_and_return(gap, place="home", substitute=YELLOW)
        assert not stole_a_dead_label(play(script, appearance=True), before, g), (
            f"a yellow disc took the red one's label after a {gap}-frame gap; re-identification is "
            "by appearance, so a different colour must not match")

    script, before, g = hide_and_return(3, place="home", substitute=RED)
    assert stole_a_dead_label(play(script, appearance=True), before, g), (
        "an identical disc no longer inherits the label. If that is real then appearance is "
        "separating things it cannot possibly separate, and the descriptor is leaking something")


def swap_script(gap, *, twin, swap, before=12, after=12):
    """Two objects hide **together** and both return, optionally with their places exchanged."""
    left, right = (0.22, 0.30), (0.75, 0.30)
    other = RED if twin else GREEN
    rest = [((0.50, 0.80), BLUE)]
    script = [([(left, RED), (right, other)] + rest, ["a", "b", "c"]) for _ in range(before)]
    script += [(list(rest), ["c"]) for _ in range(gap)]
    ends = [(right, RED), (left, other)] if swap else [(left, RED), (right, other)]
    script += [(ends + rest, ["a", "b", "c"]) for _ in range(after)]
    return script, before, gap


def test_two_objects_can_swap_places_while_hidden_unless_they_are_identical():
    """Both hide together, both return, over eight seeds:

    | objects | swapped | appearance | position coasting |
    |---|---|---|---|
    | different-looking | no | 8/8 | 8/8 |
    | different-looking | **yes** | **8/8** | **0/8** |
    | identical | no | 7/8 | 8/8 |
    | identical | **yes** | **1/8** | 0/8 |

    **The last row is worse than chance** -- a coin flip on two objects gives 4/8 -- because tied
    descriptors are broken by the assignment's ordering, which follows position, so it is
    systematically wrong rather than randomly. Identity through a swap while both are hidden is
    history-dependent and there is no history to have: the output-feedback bank cannot help either,
    because during the blackout there is nothing to integrate.
    """
    def hits(twin, swap):
        n = 0
        for s in range(8):
            script, before, gap = swap_script(6, twin=twin, swap=swap)
            worn = play(script, appearance=True, seed0=s * 7)
            n += (came_back_as_itself(worn, before, gap, "a")
                  and came_back_as_itself(worn, before, gap, "b"))
        return n

    assert hits(False, False) >= 7, "distinguishable objects no longer recover a plain absence"
    assert hits(False, True) >= 7, (
        "distinguishable objects no longer survive swapping places while hidden, which is the whole "
        "claim for matching by appearance rather than by position")
    assert hits(True, True) <= 2, (
        "two IDENTICAL objects now come back correctly after swapping while both were hidden. "
        "Nothing here can know that -- if it passes, the fixture is leaking a difference between "
        "them and the identical-object claim is untested")


def test_the_dormant_pool_empties_and_needs_descriptors_to_fill():
    """Held labels are a bounded resource, and without descriptors there is nothing to re-match on --
    so the pool stays empty rather than silently keeping position-matched labels alive."""
    red = np.array([0.8, 0.2, 0.2, 0.01, 0.01, 0.01])
    green = np.array([0.2, 0.8, 0.2, 0.01, 0.01, 0.01])

    t = RegionTracker(keep=2)
    assert t([(0.2, 0.3), (0.7, 0.3)], [300, 300], [red, green]) == [1, 2]
    assert t([], [], []) == [] and t.dormant == 2
    # Returned with their positions exchanged: the labels follow appearance, not place.
    assert t([(0.7, 0.3), (0.2, 0.3)], [300, 300], [red, green]) == [1, 2]
    assert t.dormant == 0

    # Past the forget window the pool empties and a return is a new label.
    t = RegionTracker(keep=2)
    t([(0.2, 0.3)], [300], [red])
    for _ in range(3):
        t([], [], [])
    assert t.dormant == 0
    assert t([(0.2, 0.3)], [300], [red]) == [2]

    # No descriptors means nothing to re-match on, so nothing goes dormant.
    t = RegionTracker(keep=30)
    t([(0.2, 0.3)], [300])
    t([], [])
    assert t.dormant == 0, "a label went dormant with no descriptor to reclaim it by"

    for bad in ({"keep": -1}, {"tol": 0.0}, {"tol": -1.0}):
        with pytest.raises(ValueError):
            RegionTracker(**bad)
    with pytest.raises(ValueError, match="descriptors against"):
        RegionTracker()([(0.5, 0.5)], [10], [red, green])


def test_the_descriptor_reads_the_region_and_not_its_neighbourhood():
    """Masked rather than a bounding-box crop, which is a correction rather than a preference: this
    package measured a crop carrying *position*, because a crop near a frame edge is clamped, and
    that leak inflated an identity control from 50.6% to 74.3% until the fixture was levelled."""
    from demo.label_sources import region_descriptor

    img, masks = render([((0.22, 0.30), RED), ((0.75, 0.30), GREEN)], seed=0)
    a = region_descriptor(img, masks[0])
    b = region_descriptor(img, masks[1])
    assert len(a) == 6, "six numbers at RGB: three means and three spreads"
    assert np.linalg.norm(a - b) > 0.25, (
        f"two differently coloured regions describe {np.linalg.norm(a - b):.3f} apart, inside the "
        "default tol; colour is what does the work here")

    # The same object at the frame's edge must describe the same as in the middle -- a crop would
    # not, because it gets clamped and carries where it is.
    mid, edge = render([((0.50, 0.50), RED)], seed=1), render([((0.12, 0.50), RED)], seed=1)
    gap = np.linalg.norm(region_descriptor(mid[0], mid[1][0])
                         - region_descriptor(edge[0], edge[1][0]))
    assert gap < 0.02, f"the same object describes {gap:.4f} apart at the edge; position is leaking"

    # An empty region answers rather than raising, and a greyscale frame gives four numbers.
    assert not region_descriptor(img, np.zeros(img.shape[:2], dtype=bool)).any()
    assert len(region_descriptor(img.mean(axis=2), masks[0])) == 2


def test_the_slot_to_region_pairing_follows_the_matchers_permutation():
    """The appearance matcher hands back the same points in a different order, and which region each
    slot came from has to follow. Getting it wrong colours a region with another slot's label, which
    is the defect that made the ids flicker in the first place.

    The interesting case is **duplicate points**: a slot the frame had no region for carries the
    frame centre, so two entries can be equal, and `before.index(pt)` would hand both of them the
    same region. This lived inside `main()`'s loop where nothing could reach it.
    """
    from demo.pointer import follow_permutation

    before = [(0.2, 0.5), (0.5, 0.5), (0.8, 0.5)]
    assert follow_permutation(before, before, [7, 8, 9]) == [7, 8, 9]
    assert follow_permutation(before, [before[2], before[0], before[1]], [7, 8, 9]) == [9, 7, 8]

    # Two slots at the frame centre, which is what an absent region carries.
    dup = [(0.5, 0.5), (0.2, 0.5), (0.5, 0.5)]
    got = follow_permutation(dup, [dup[1], dup[0], dup[2]], ["a", "b", "c"])
    assert got == ["b", "a", "c"], (
        f"got {got}; with two identical points the pairing must still be one-to-one, and "
        "`before.index` would give both of them the first match")
    assert sorted(follow_permutation(dup, [dup[2], dup[0], dup[1]], ["a", "b", "c"])) == \
        ["a", "b", "c"], "a value was used twice or dropped"

    assert follow_permutation([], [], []) == []
    with pytest.raises(ValueError, match="values"):
        follow_permutation(before, before, [1, 2])
    with pytest.raises(ValueError, match="not one of the points"):
        follow_permutation(before, [(0.9, 0.9), before[0], before[1]], [1, 2, 3])


# --- Why a region goes unlabelled, and what admits it -------------------------------------------

WALL = 0.35
#: Deliberately close to the wall AND small: this is the object that goes unlabelled. Four large
#: distinct objects beside it mean the scene has six appearance groups where `k = objects + 1` is 4,
#: so two must share clusters -- and whichever shares with the wall is dropped entirely.
FAINT = (0.44, 0.42, 0.43)
CROWD = [((0.16, 0.24), 0.03, 0.0, RED, 0.10),
         ((0.80, 0.24), 0.03, 1.3, GREEN, 0.10),
         ((0.16, 0.76), 0.03, 2.5, BLUE, 0.10),
         ((0.80, 0.76), 0.03, 3.8, YELLOW, 0.10),
         ((0.48, 0.50), 0.03, 5.0, FAINT, 0.055)]


def crowd_frame(t, seed=0):
    yy, xx = np.mgrid[0:H, 0:W]
    img = np.full((H, W, 3), WALL)
    masks = []
    for (cx, cy), orbit, phase, colour, r in CROWD:
        a = phase + 2 * np.pi * t / PERIOD
        p = (cx + orbit * float(np.cos(a)), cy + orbit * float(np.sin(a)))
        m = np.hypot(xx - p[0] * W, yy - p[1] * H) <= r * W
        img[m] = colour
        masks.append(m)
    g = np.random.default_rng(seed)
    return (img + g.standard_normal(img.shape) * 0.01).astype("f4"), masks


def noisy_three(t, seed=0):
    img, masks = frame_at(t, THREE, 0.11, seed=seed)
    g = np.random.default_rng(1000 + seed)
    return np.clip(img + g.standard_normal(img.shape).astype("f4") * 0.03, 0, 1), masks


def field_quality(maker, k, contrast, frames=24, show_frac=0.03):
    """Coverage and precision of the painted field. Both, because either alone cannot fail."""
    seg, tracker = FrameSegmenter(k), RegionTracker(keep=30)
    cov, prec, count = [], [], []
    for t in range(frames):
        img, masks = maker(t, seed=t)
        seg(img)
        rmap, centres, of_cluster = split_regions(seg.labels, k, img, edges=0.95, min_frac=0.004)
        field, ids = label_field(rmap, of_cluster, seg.labels, tracker, min_frac=show_frac,
                                 frame=img, contrast=contrast)
        count.append(len(ids))
        slots = range(1, len(ids) + 1)
        on = sum(1 for s in slots
                 if any(((field == s) & m).sum() > 0.3 * (field == s).sum() for m in masks))
        prec.append(on / max(len(ids), 1))
        hit = sum(1 for m in masks
                  if ids and max(((field == s) & m).sum() for s in slots) > 0.3 * m.sum())
        cov.append(hit / len(masks))
    return (float(np.mean(cov)), float(np.mean(prec)), float(np.mean(count)))


def test_contrast_is_what_admits_an_object_the_other_rules_drop():
    """The reported symptom was areas with no label and a count that looked fixed. **The count was
    never fixed** -- the row below paints 4 labels for a declared 3 and 5 for five objects -- and the
    cause was two drops that have nothing to do with counting: sitting in the *background cluster*,
    which happens as soon as a scene has more appearance groups than `k` allows, and being under the
    area floor.

    Measured through the shipped `label_field`:

    | scene | contrast off | contrast 0.125 |
    |---|---|---|
    | small faint object among four | covered **0.80**, precision 1.00, 4.0 labels | covered **1.00**, precision 1.00, 5.0 |
    | five large objects | 1.00 / 1.00 / 5.0 | 1.00 / 1.00 / 5.0 |
    | three alike, noisy wall | 1.00 / **0.99** / 3.0 | 1.00 / **0.92** / 3.3 |

    The cost is stated rather than hidden: seven points of precision on a noise-carved wall buys the
    difference between an object never being labelled and being labelled.
    """
    off = field_quality(crowd_frame, 4, 0.0)
    on = field_quality(crowd_frame, 4, 0.125)
    assert off[0] < 0.9, (
        f"the crowded scene already covers {off[0]:.2f} with contrast off; it is supposed to lose "
        "the faint object to the background cluster, so this fixture is not reproducing the report")
    assert on[0] > 0.95, f"contrast recovered only {on[0]:.2f} of the objects"
    assert on[2] > off[2], "contrast admitted no extra label at all"

    # It must not disturb a scene that already worked.
    assert field_quality(frame_at_five, 4, 0.125)[:2] == pytest.approx((1.0, 1.0), abs=0.05)

    # And the cost on a noisy wall is real, so it is asserted rather than described.
    clean = field_quality(noisy_three, 4, 0.0)
    admitted = field_quality(noisy_three, 4, 0.125)
    assert admitted[1] < clean[1] - 0.03, (
        f"contrast costs nothing on a noise-carved wall now ({clean[1]:.2f} -> {admitted[1]:.2f}). "
        "If that is real it should be on unconditionally and the docstring's trade is stale")
    assert admitted[0] > 0.95, "and it must not lose objects while doing it"


def frame_at_five(t, seed=0):
    return frame_at(t, FIVE, 0.09, seed=seed)


def test_raising_k_does_not_fix_coverage_at_any_value():
    """**`k` is not the mechanism, and this is the cleanest statement of it**: with contrast off the
    crowded scene covers 0.80 at k=4, k=6 and k=8 alike, and with contrast on it covers 1.00 at all
    three. Raising `k` separates the faint object's *cluster* -- measured, at k=6 it has its own --
    and the area floor drops it anyway, so nothing changes.

    An earlier version of this test also asserted that raising `k` *damaged* another fixture, and
    that was an artefact of the background-selection defect rather than a property of `k`. The claim
    withdrawn; what survives is the one below, which never depended on it.
    """
    for k in (4, 6, 8):
        assert field_quality(crowd_frame, k, 0.0)[0] < 0.9, (
            f"k={k} with contrast off now covers the faint object; if raising k is sufficient then "
            "contrast is not needed")
        assert field_quality(crowd_frame, k, 0.125)[0] > 0.95, (
            f"k={k} with contrast on does not cover it either, so the mechanism is not what this "
            "file claims")


def test_ring_contrast_reads_the_boundary_and_not_the_frame_edge():
    """A region touching the frame edge gets a **thinner** ring rather than a clamped one: there is
    genuinely less outside to compare against, and clamping would make a region look like an object
    for being at the edge -- the position leak that inflated an identity control here from 50.6% to
    74.3% until the fixture was levelled.

    Measured across three scenes: the smallest contrast any real object had was **0.136** and the
    largest any non-wall region had was **0.118**, so a separating value exists and the interval is
    only **1.15x** wide. That thinness is why it is a documented knob.
    """
    from demo.label_sources import ring_contrast

    img = np.full((H, W, 3), WALL, dtype="f4")
    yy, xx = np.mgrid[0:H, 0:W]
    disc_mid = np.hypot(xx - 0.5 * W, yy - 0.5 * H) <= 8
    img[disc_mid] = (0.9, 0.2, 0.2)
    assert ring_contrast(img, disc_mid) > 0.5, "a bright disc on a plain wall has no contrast"

    # A patch of the wall itself has none, whatever its size or place.
    flat = np.zeros((H, W), dtype=bool)
    flat[4:14, 4:14] = True
    assert ring_contrast(img, flat) < 0.02, "a patch of plain wall reads as a boundary"

    # The same object at the frame edge still reads as an object -- a thinner ring, not a wrong one.
    edge = np.hypot(xx - 4, yy - 0.5 * H) <= 8
    img2 = np.full((H, W, 3), WALL, dtype="f4")
    img2[edge] = (0.9, 0.2, 0.2)
    assert ring_contrast(img2, edge) > 0.4, "an object at the frame edge lost its contrast"

    # Degenerate inputs answer rather than raising; a mask covering everything has no ring.
    assert ring_contrast(img, np.zeros((H, W), dtype=bool)) == 0.0
    assert ring_contrast(img, np.ones((H, W), dtype=bool)) == 0.0
    with pytest.raises(ValueError, match="width"):
        ring_contrast(img, disc_mid, width=0)


def test_contrast_zero_restores_the_previous_behaviour_exactly():
    """The knob has to be a true off switch, or "it used to work like this" stops being checkable."""
    seg, k = FrameSegmenter(4), 4
    img, _masks = crowd_frame(3, seed=3)
    seg(img)
    rmap, centres, of_cluster = split_regions(seg.labels, k, img, edges=0.95, min_frac=0.004)
    a = label_field(rmap, of_cluster, seg.labels, RegionTracker(keep=30), frame=img, contrast=0.0)
    b = label_field(rmap, of_cluster, seg.labels, RegionTracker(keep=30), frame=None,
                    contrast=0.125)
    assert np.array_equal(a[0], b[0]), (
        "contrast=0 and no frame at all disagree; one of them is not the old rule")


#: Three distinct colours so each gets its own cluster at k=4, and the third is SMALL. That puts it
#: outside the background cluster and under the area floor -- the branch the crowded fixture never
#: reaches, because there the faint object is inside the background cluster instead. A mutant
#: dropping it survived every other test in this file.
SPARSE = [((0.22, 0.30), 0.03, 0.0, RED, 0.10),
          ((0.76, 0.30), 0.03, 2.1, GREEN, 0.10),
          ((0.48, 0.72), 0.03, 4.2, BLUE, 0.045)]


def sparse_frame(t, seed=0):
    yy, xx = np.mgrid[0:H, 0:W]
    img = np.full((H, W, 3), WALL)
    masks = []
    for (cx, cy), orbit, phase, colour, r in SPARSE:
        a = phase + 2 * np.pi * t / PERIOD
        p = (cx + orbit * float(np.cos(a)), cy + orbit * float(np.sin(a)))
        m = np.hypot(xx - p[0] * W, yy - p[1] * H) <= r * W
        img[m] = colour
        masks.append(m)
    g = np.random.default_rng(seed)
    return (img + g.standard_normal(img.shape) * 0.01).astype("f4"), masks


def test_a_small_object_in_its_own_cluster_is_admitted_by_contrast_too():
    """**The area floor and contrast are alternatives, not both required.** Outside the background
    cluster a small object still has to get past the floor, and the crowded fixture never reaches
    that branch because its faint object is *inside* the background cluster. A mutant that kept the
    floor as a veto there survived every other test here.

    This object is about 0.9% of the frame against a show floor of 3%, and it is bright, so it is
    exactly the case the rule exists for.
    """
    img, masks = sparse_frame(0, seed=0)
    seg = FrameSegmenter(4)
    for _ in range(3):
        seg(img)
    small = masks[2]
    background = max(range(4), key=lambda j: (seg.labels == j).sum())
    holds = int(np.bincount(seg.labels[small], minlength=4).argmax())
    assert holds != background, (
        "the small object landed in the background cluster, so this fixture is a copy of the "
        "crowded one and does not reach the branch it was written for")
    assert small.sum() < 0.03 * small.size, "the object is over the show floor and proves nothing"

    off = field_quality(sparse_frame, 4, 0.0)
    on = field_quality(sparse_frame, 4, 0.125)
    assert off[0] < 0.9, f"the floor no longer drops it ({off[0]:.2f}); the fixture is not testing"
    assert on[0] > 0.95, f"contrast admitted only {on[0]:.2f} of the objects"
    assert on[1] > 0.9, f"and it brought junk with it, precision {on[1]:.2f}"


def test_the_pixel_minimum_is_unreachable_at_the_shipped_floors_and_still_works():
    """`MIN_REGION_PX` is **dead as the demo is configured** -- the split's own floor is 0.004 of a
    72x96 frame, which is 27 pixels, so nothing smaller than 12 ever reaches `label_field`. Same
    shape as the `at_most(PTrace, 1e4)` bound this repository found configured and unable to fire,
    and as `min_dtype` being unreachable for every shipped adapter.

    It is kept because it is not dead *code* -- a caller lowering `--seg-min-frac` reaches it
    immediately -- so it is exercised with a floor that does, and the unreachability at the default
    is asserted rather than assumed.
    """
    from demo.pointer import MIN_REGION_PX

    # A noise-carved wall, because that is the only thing that produces specks at all -- a clean
    # scene has contiguous clusters and cannot exercise this in either direction.
    seg = FrameSegmenter(4)
    img, _masks = noisy_three(1, seed=1)
    for _ in range(3):
        seg(img)

    # At the shipped floor, nothing small enough to reach the guard exists.
    rmap, _c, _of = split_regions(seg.labels, 4, img, edges=0.95, min_frac=0.004)
    smallest = min((rmap == i).sum() for i in range(1, rmap.max() + 1)) if rmap.max() else 0
    assert smallest >= MIN_REGION_PX, (
        f"a {smallest}-pixel region reached label_field at the shipped floor, so the guard is live "
        "after all and this test's premise is wrong")

    # Below it, specks appear and the guard is what keeps them out of the picture.
    rmap, _c, of = split_regions(seg.labels, 4, img, edges=0.95, min_frac=0.0004)
    tiny = sum(1 for i in range(1, rmap.max() + 1) if (rmap == i).sum() < MIN_REGION_PX)
    assert tiny > 0, "no sub-minimum region even at a floor of 0.0004; the guard cannot be tested"
    field, ids = label_field(rmap, of, seg.labels, RegionTracker(keep=30), min_frac=0.03,
                             frame=img, contrast=0.125)
    painted = [int((field == s).sum()) for s in range(1, len(ids) + 1)]
    assert painted and min(painted) >= MIN_REGION_PX, (
        f"a region of {min(painted)} pixels was painted; the pixel minimum is not being applied")


# --- Does the count actually follow the scene? ---------------------------------------------------


def entering_frame(t, seed=0):
    """Four objects, each present only between frames 4i and 4i+14, so the true count rises 1..4
    and falls back to 0. Nothing about `k` or `--objects` changes across the run."""
    yy, xx = np.mgrid[0:H, 0:W]
    img = np.full((H, W, 3), WALL)
    masks = []
    for i, (cx, cy, colour) in enumerate(((0.22, 0.28, RED), (0.78, 0.28, GREEN),
                                          (0.22, 0.74, BLUE), (0.78, 0.74, YELLOW))):
        if 4 * i <= t < 4 * i + 14:
            m = np.hypot(xx - cx * W, yy - cy * H) <= 0.10 * W
            img[m] = colour
            masks.append(m)
    g = np.random.default_rng(seed)
    return (img + g.standard_normal(img.shape) * 0.01).astype("f4"), masks


def lone_frame(t, seed=0):
    """One object on a clean wall -- the case that exposed the background-selection defect."""
    yy, xx = np.mgrid[0:H, 0:W]
    img = np.full((H, W, 3), WALL)
    m = np.hypot(xx - (0.30 + 0.02 * t) * W, yy - 0.5 * H) <= 0.10 * W
    img[m] = RED
    g = np.random.default_rng(seed)
    return (img + g.standard_normal(img.shape) * 0.01).astype("f4"), [m]


def label_counts(maker, k=4, frames=30, contrast=0.125):
    seg, tracker = FrameSegmenter(k), RegionTracker(keep=30)
    true, got = [], []
    for t in range(frames):
        img, masks = maker(t, seed=t)
        seg(img)
        rmap, _c, of_cluster = split_regions(seg.labels, k, img, edges=0.95, min_frac=0.004)
        _field, ids = label_field(rmap, of_cluster, seg.labels, tracker, min_frac=0.03,
                                  frame=img, contrast=contrast)
        true.append(len(masks))
        got.append(len(ids))
    return true, got, tracker.minted


def test_the_label_count_follows_the_scene_frame_by_frame():
    """`k` and `--objects` are fixed for the whole run; the count is not.

    Objects enter and leave so the true count goes 1, 2, 3, 4, 3, 2, 1, 0 and the painted count
    **agrees on all 30 frames**, minting exactly **4** labels for 4 objects -- so it is following the
    scene rather than re-minting its way to the right number.
    """
    true, got, minted = label_counts(entering_frame)
    assert len(set(true)) >= 4, "the fixture no longer varies its object count"
    agree = sum(a == b for a, b in zip(true, got))
    assert agree >= 29, f"the count matched on only {agree}/30 frames: {list(zip(true, got))}"
    assert minted <= 6, (
        f"{minted} labels minted for four objects; the count is being reached by re-minting rather "
        "than by following the scene")


def test_a_lone_object_on_a_clean_wall_is_labelled_on_every_frame():
    """**The defect this test was written for, found by asking whether the count really varies.**
    `label_field` chose its background from the clusters that had produced a surviving region. On a
    clean wall the wall's clusters produce none -- their fragments are under the split's floor -- so
    the object's own cluster became "the largest present" and was dropped as background.

    Measured before the fix: a frame with exactly one region, the object's own, contrast 0.519,
    painted **nothing**. The background is the largest cluster in the *frame* now, whether or not it
    produced a region.
    """
    true, got, minted = label_counts(lone_frame, frames=12)
    assert got == [1] * 12, f"a lone object was not labelled on every frame: {got}"
    assert minted == 1, f"{minted} labels minted for one object that never left"


def test_the_background_is_the_biggest_cluster_in_the_frame_not_the_biggest_that_has_a_region():
    """The defect in isolation: a frame where only the object's cluster has a surviving region."""
    img, masks = lone_frame(3, seed=3)
    seg = FrameSegmenter(4)
    for _ in range(3):
        seg(img)
    rmap, _c, of_cluster = split_regions(seg.labels, 4, img, edges=0.95, min_frac=0.004)

    survivors = sorted(set(int(j) for j in of_cluster))
    biggest = int(np.argmax([(seg.labels == j).sum() for j in range(4)]))
    assert biggest not in survivors, (
        "the wall's cluster still has a region here, so this frame no longer isolates the defect")

    _field, ids = label_field(rmap, of_cluster, seg.labels, RegionTracker(keep=30), min_frac=0.03,
                              frame=img, contrast=0.125)
    assert len(ids) == 1, (
        f"{len(ids)} labels where the object is the only region in the frame; picking the "
        "background from the clusters that have regions drops the object itself")
