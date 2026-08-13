# SPDX-License-Identifier: Apache-2.0
"""Grouping low-level labels into higher-order ones: refused in 2026-08-12, and now measured to work.

The strategy is the owner's: cluster low, then group similar labels together at a deeper level. It
was measured on 2026-08-12 and **refused** -- grouping helped at two objects (98.1% purity) and hurt
at three (65.4% against the online rule's 91.5%), no distance cutoff read the right number of groups,
and more observations made it worse.

**The refusal's stated cause has since been removed, which is why it is re-measured rather than
restated.** It was "an untrained backbone's CROP descriptor conflates identity with view": a crop near
a frame edge is clamped, so it carried position, and the recorded within/between distances overlapped
-- (0.081, 0.225) at two objects, (0.131, 0.288) at three. `region_descriptor` now reads the region's
**own pixels** in raw RGB, and the same object at the frame edge and in the middle describes 0.0000
apart.

What this file pins, both halves:

* **grouping works where the descriptor separates** -- purity 1.00 with any cutoff in 0.10-0.30, and
  the number of groups is found rather than declared
* **it buys nothing where continuity already works** -- 1.0 labels per object either way
* **it earns its place across a gap longer than the tracker's forget window** -- the online rule mints
  3.0 labels per object there, grouping consolidates them to 1.0
* **it fails where the descriptor does not separate** -- 0.76-0.80 against the online rule's
  0.95-1.00, which is the regime the 2026-08-12 measurement was in, so both results are consistent

Every figure is measured on this host with these fixtures, and the fixtures are synthetic, so the
*ordering* is the claim. Nothing here is wired into the demo: this is the evidence for a decision, not
a feature.
"""

from __future__ import annotations

import numpy as np
import pytest

from demo.label_sources import FrameSegmenter, RegionTracker, region_descriptor, split_regions
from demo.pointer import MIN_REGION_PX, label_field

H, W = 72, 96
WALL = 0.35
PERIOD = 40.0
COLOURS = [(0.85, 0.20, 0.20), (0.20, 0.75, 0.25), (0.20, 0.25, 0.85),
           (0.85, 0.80, 0.20), (0.75, 0.30, 0.80)]
CENTRES = [(0.18, 0.24), (0.80, 0.24), (0.18, 0.76), (0.80, 0.76), (0.49, 0.50)]


def scene(t, n, *, seed=0, r=0.09, colours=None, gradient=0.56, gap=0, block=20):
    """`n` objects on their own orbits, with the appearance VARIATION that makes grouping a question.

    A flat disc on a flat wall describes identically every frame -- within-object distance 0.002
    against between-object 0.85 -- so a fixture without variation cannot fail. Three sources here,
    all of them things a camera really does: an illumination gradient across the frame, a size cycle
    so a region is sometimes a small part of the object, and (with `gap`) whole absences.
    """
    yy, xx = np.mgrid[0:H, 0:W]
    img = np.full((H, W, 3), WALL)
    masks = []
    present = gap == 0 or (t % (block + gap)) < block
    if present:
        for i in range(n):
            a = 2 * np.pi * t / PERIOD + 1.3 * i
            cx, cy = CENTRES[i]
            p = (cx + 0.16 * float(np.cos(a)), cy + 0.16 * float(np.sin(a)))
            rr = r * (1.0 + 0.35 * float(np.sin(a)))
            m = np.hypot(xx - p[0] * W, yy - p[1] * H) <= rr * W
            img[m] = (COLOURS[i] if colours is None else colours[i])
            masks.append(m)
    img *= ((1.0 - gradient / 2) + gradient * xx / (W - 1))[:, :, None]
    g = np.random.default_rng(seed)
    return np.clip(img + g.standard_normal(img.shape) * 0.01, 0, 1).astype("f4"), masks


def observe(n, frames=40, *, keep=30, **kw):
    """Run the shipped path; return (descriptor, true object, online label) for every painted region."""
    seg, tracker = FrameSegmenter(n + 1), RegionTracker(keep=keep)
    rows = []
    for t in range(frames):
        img, masks = scene(t, n, seed=t, **kw)
        seg(img)
        rmap, _c, of_cluster = split_regions(seg.labels, n + 1, img, edges=0.95, min_frac=0.004)
        field, ids = label_field(rmap, of_cluster, seg.labels, tracker, min_frac=0.03, frame=img,
                                 contrast=0.125)
        for slot, label in enumerate(ids, start=1):
            piece = field == slot
            if piece.sum() < MIN_REGION_PX:
                continue
            owner, best = None, 0
            for o, m in enumerate(masks):
                overlap = (piece & m).sum()
                if overlap > best:
                    owner, best = o, overlap
            if owner is None or best < 0.3 * piece.sum():
                owner = -1                            # a region that is not on any object
            rows.append((region_descriptor(img, piece), owner, label))
    return rows, tracker.minted


def group(desc, cutoff):
    """Single-linkage grouping at a distance cutoff: transitive by construction, no count declared."""
    n = len(desc)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    d = np.linalg.norm(np.asarray(desc)[:, None, :] - np.asarray(desc)[None, :, :], axis=2)
    for i in range(n):
        for j in range(i + 1, n):
            if d[i, j] <= cutoff:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[max(ri, rj)] = min(ri, rj)
    roots = {}
    return [roots.setdefault(find(i), len(roots)) for i in range(n)]


def purity(labels, owners):
    """Weighted fraction of each group belonging to that group's dominant object.

    **Cannot fail on its own** -- a scheme minting a fresh label per observation scores a perfect
    1.00 -- which is why `per_object` is measured beside it everywhere below.
    """
    total, hit = 0, 0
    for g in set(labels):
        members = [o for gg, o in zip(labels, owners) if gg == g]
        _v, counts = np.unique(members, return_counts=True)
        hit += counts.max()
        total += len(members)
    return hit / max(total, 1)


def per_object(labels, owners, n):
    """How many distinct labels each true object wore. 1.0 is the ideal; inflation is the failure."""
    return float(np.mean([len({v for v, o in zip(labels, owners) if o == i}) for i in range(n)]))


def separation(rows):
    """The precondition: within-object distance against between-object distance."""
    desc = np.asarray([r[0] for r in rows])
    owners = [r[1] for r in rows]
    d = np.linalg.norm(desc[:, None, :] - desc[None, :, :], axis=2)
    same, diff = [], []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            if owners[i] < 0 or owners[j] < 0:
                continue
            (same if owners[i] == owners[j] else diff).append(d[i, j])
    return max(same), min(diff)


def test_the_fixture_gives_the_same_object_something_to_vary():
    """The guard that makes everything below capable of failing. Without view variation the same disc
    describes identically -- within 0.002 against between 0.85 -- and every scheme scores 1.00. That
    is what the first version of this measurement did, and it is this project's own signal of a broken
    fixture."""
    rows, _minted = observe(3)
    within, between = separation(rows)
    assert within > 0.05, (
        f"within-object distance is only {within:.3f}; the fixture has stopped varying the view and "
        "nothing below can fail")
    assert within < between, (
        f"within {within:.3f} exceeds between {between:.3f}; this fixture is the SEPARATED case and "
        "the overlapping one is tested separately")


def test_grouping_works_where_the_descriptor_separates_and_finds_its_own_count():
    """Purity 1.00 at every cutoff from 0.10 to 0.30 -- a **3x-wide** band -- and the number of groups
    is found rather than declared: 2, 4 and 6 for two, three and five objects, the extra one being the
    background fragments that are not on any object.

    **Forcing the count is what hurt.** Constrained to exactly `n` groups, purity falls to 0.80 at
    three objects and 0.86 at five, because single linkage then has to merge two objects to hit the
    number. Letting the cutoff decide is the whole point of the proposal.
    """
    for n in (2, 3, 5):
        rows, _minted = observe(n)
        desc, owners = [r[0] for r in rows], [r[1] for r in rows]
        for cutoff in (0.10, 0.20, 0.30):
            g = group(desc, cutoff)
            assert purity(g, owners) > 0.95, (
                f"n={n} cutoff {cutoff}: purity {purity(g, owners):.2f}")
            assert n <= len(set(g)) <= n + 3, (
                f"n={n} cutoff {cutoff}: {len(set(g))} groups for {n} objects plus background")


def test_grouping_buys_nothing_where_continuity_already_works():
    """The unflattering half, and the reason this is not wired into the demo. With no absences the
    tracker's own labels are already 1.0 per object at purity 1.00, so grouping has nothing to
    consolidate."""
    rows, _minted = observe(3, gap=0)
    online = [r[2] for r in rows]
    owners = [r[1] for r in rows]
    grouped = group([r[0] for r in rows], 0.20)
    assert per_object(online, owners, 3) == pytest.approx(1.0, abs=0.2), (
        "the online rule no longer holds one label per object on a continuous run, so the comparison "
        "below is not against a working baseline")
    assert per_object(grouped, owners, 3) == pytest.approx(per_object(online, owners, 3), abs=0.2), (
        "grouping now differs from continuity on a continuous run; if it is better, it should be "
        "wired in, and if worse, it should not be offered")


def test_grouping_earns_its_place_across_a_gap_longer_than_the_forget_window():
    """**The one case where it beats what ships.** Three objects vanish together for 40 frames every
    20, with a forget window of 30 -- so `RegionTracker` cannot reclaim them and mints fresh labels:

    | gap | keep | online labels per object | grouped |
    |---|---|---|---|
    | 0 | 30 | 1.0 | 1.0 |
    | 40 | 30 | **3.0** | **1.0** |
    | 60 | 30 | 2.0 | **1.0** |
    | 40 | 5 | **3.0** | **1.0** |

    Purity is 1.00 for both throughout, which is exactly why it cannot be the metric: a fresh label
    per episode is perfectly pure. Inflation is what discriminates.
    """
    for gap, keep in ((40, 30), (40, 5)):
        rows, _minted = observe(3, frames=160, gap=gap, keep=keep)
        owners = [r[1] for r in rows]
        online = per_object([r[2] for r in rows], owners, 3)
        grouped = per_object(group([r[0] for r in rows], 0.20), owners, 3)
        assert online > 1.8, (
            f"gap {gap} keep {keep}: the tracker minted only {online:.1f} labels per object, so it "
            "is reclaiming them after all and there is nothing for grouping to consolidate")
        assert grouped < online - 0.8, (
            f"gap {gap} keep {keep}: grouping gives {grouped:.1f} labels per object against the "
            f"tracker's {online:.1f}; the consolidation this proposal is for has stopped happening")
        assert purity(group([r[0] for r in rows], 0.20), owners) > 0.95


def test_grouping_fails_where_the_descriptor_does_not_separate():
    """**The precondition, and the reason the 2026-08-12 refusal was right at the time.** Objects of
    similar colour under a strong illumination gradient put within-object variation *above*
    between-object separation, and grouping then scores 0.76-0.80 against the online rule's 0.95-1.00:

    | colour step | gradient | within max | between min | grouped | online |
    |---|---|---|---|---|---|
    | 0.05 | 0.56 | 0.168 | 0.045 | **0.79** | 1.00 |
    | 0.05 | 0.90 | 0.261 | 0.040 | 0.80 | 0.95 |
    | 0.02 | 0.90 | 0.243 | 0.014 | **0.76** | 1.00 |

    The old measurement's descriptor was in exactly this regime -- within/between of (0.131, 0.288) at
    three objects -- so that refusal and this file's positive result are the same finding read at two
    points, not a contradiction.
    """
    base = np.array([0.70, 0.35, 0.30])
    similar = [tuple(np.clip(base + 0.02 * i, 0, 1)) for i in range(3)]
    rows, _minted = observe(3, colours=similar, gradient=0.90)
    within, between = separation(rows)
    assert within > between, (
        f"within {within:.3f} against between {between:.3f}: the fixture no longer overlaps, so it "
        "cannot demonstrate the failure it exists for")

    owners = [r[1] for r in rows]
    grouped = purity(group([r[0] for r in rows], 0.05), owners)
    online = purity([r[2] for r in rows], owners)
    assert grouped < 0.9, f"grouping scores {grouped:.2f} on descriptors that do not separate"
    assert online > grouped + 0.1, (
        f"the online rule ({online:.2f}) no longer beats grouping ({grouped:.2f}) here; if grouping "
        "has become robust to an overlapping descriptor, the precondition above is wrong")
