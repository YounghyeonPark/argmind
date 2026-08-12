# SPDX-License-Identifier: Apache-2.0
"""Splitting an appearance cluster into objects: by connectivity, and by the frame's own edges.

A cluster is a region of similar appearance, not an object, and `FrameSegmenter`'s docstring has
always said so. This file measures what that costs and what fixes it -- two same-coloured objects
held apart read IoU **0.50** as one cluster and **1.00** as two regions -- along with the two limits
that must not be lost: **two identical objects that touch cannot be separated by anything here**,
and the edge gate earns its place only where `k` is under-declared, which is the normal condition
because `k` is a declaration and nothing counts the appearance groups in a room.

Every number here is measured on this host with these fixtures. The scenes are synthetic discs, so
the *ordering* of the methods is the claim, not the absolute IoU.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from demo.label_sources import (FrameSegmenter, connected_regions, edge_gate, pick_regions,
                                split_regions)
from demo.pointer import slot_label_map

H, W = 72, 96
RED = (0.80, 0.20, 0.20)
GREEN = (0.20, 0.75, 0.25)
BLUE = (0.20, 0.25, 0.80)
YELLOW = (0.85, 0.80, 0.15)


def disc(p, r=0.11):
    yy, xx = np.mgrid[0:H, 0:W]
    return np.hypot(xx - p[0] * W, yy - p[1] * H) <= r * W


def scene(objs, base=0.35, seed=0):
    """`objs` is a list of (centre, colour). Returns the frame and one mask per object."""
    img = np.full((H, W, 3), base)
    masks = []
    for p, c in objs:
        m = disc(p)
        img[m] = c
        masks.append(m)
    g = np.random.default_rng(seed)
    return (img + g.standard_normal(img.shape) * 0.01).astype("f4"), masks


#: Each entry is (objects, which of them the caller declared, k). `k` is `declared + 1` throughout,
#: which is the demo's own rule -- one cluster is the background.
SCENES = {
    "same colour apart": ([((0.28, 0.5), RED), ((0.72, 0.5), RED)], [0, 1], 3),
    "same colour touching": ([((0.44, 0.5), RED), ((0.56, 0.5), RED)], [0, 1], 3),
    "two colours apart": ([((0.28, 0.5), RED), ((0.72, 0.5), GREEN)], [0, 1], 3),
    "one object": ([((0.5, 0.5), RED)], [0], 2),
    "touch + 1 distractor": ([((0.40, 0.5), RED), ((0.52, 0.5), GREEN), ((0.80, 0.25), BLUE)],
                             [0, 1], 3),
    "touch + 2 distractors": ([((0.40, 0.5), RED), ((0.52, 0.5), GREEN), ((0.80, 0.25), BLUE),
                              ((0.20, 0.80), YELLOW)], [0, 1], 3),
    "apart + 2 distractors": ([((0.30, 0.5), RED), ((0.62, 0.5), GREEN), ((0.85, 0.20), BLUE),
                               ((0.15, 0.82), YELLOW)], [0, 1], 3),
}


def flood_reference(mask, gate=None):
    """Components by explicit flood fill, which shares no step with hook-and-jump.

    The same discipline as RLS against the closed-form solution and the backbone against four
    nested loops: a fast implementation is verified against the definition, not against itself.
    """
    m = np.asarray(mask, dtype=bool)
    g = None if gate is None else np.asarray(gate, dtype=bool)
    out = np.zeros(m.shape, dtype=np.int32)
    nxt = 0
    for sy in range(m.shape[0]):
        for sx in range(m.shape[1]):
            if not m[sy, sx] or out[sy, sx]:
                continue
            nxt += 1
            stack = [(sy, sx)]
            out[sy, sx] = nxt
            while stack:
                y, x = stack.pop()
                blocked = g is not None and g[y, x]
                for dy, dx, mine in ((0, -1, blocked), (-1, 0, blocked),
                                     (0, 1, g is not None and x + 1 < m.shape[1] and g[y, x + 1]),
                                     (1, 0, g is not None and y + 1 < m.shape[0] and g[y + 1, x])):
                    # A gated pixel does not join across its OWN left/upper edge, so a step to the
                    # right or down is blocked by the gate on the pixel being stepped INTO.
                    ny, nx = y + dy, x + dx
                    if not (0 <= ny < m.shape[0] and 0 <= nx < m.shape[1]):
                        continue
                    if mine or not m[ny, nx] or out[ny, nx]:
                        continue
                    out[ny, nx] = nxt
                    stack.append((ny, nx))
    return out, nxt


def segment(objs, k, seed=0, reps=3):
    img, masks = scene(objs, seed=seed)
    seg = FrameSegmenter(k)
    for _ in range(reps):
        seg(img)                                     # warm the centres, as the demo's loop does
    return img, masks, seg


def best_iou(regions, masks):
    out = []
    for mk in masks:
        best = 0.0
        for r in regions:
            u = (r | mk).sum()
            if u:
                best = max(best, float((r & mk).sum() / u))
        out.append(best)
    return float(np.mean(out))


def measure(name, *, split, edges, min_frac=0.01):
    """Mean best IoU over the objects the caller declared, and the number of regions painted."""
    objs, targets, k = SCENES[name]
    img, masks, seg = segment(objs, k)
    if not split:
        regions = [seg.labels == j for j in range(k) if (seg.labels == j).any()]
    else:
        lab, centres, _of = split_regions(seg.labels, k, img, edges=edges, min_frac=min_frac)
        regions = [lab == i for i in range(1, len(centres) + 1)]
    return best_iou(regions, [masks[i] for i in targets]), len(regions)


def test_the_fast_components_agree_with_a_flood_fill():
    """Verified against the definition rather than against itself, over every scene, both gate
    settings, and every cluster -- the same discipline as RLS against the closed form."""
    checked = 0
    for name, (objs, _t, k) in SCENES.items():
        for q in (None, 0.95):
            img, _masks, seg = segment(objs, k)
            gate = edge_gate(img, q) if q is not None else None
            for j in range(k):
                mk = seg.labels == j
                if not mk.any():
                    continue
                ref, n = flood_reference(mk, gate)
                got, ids = connected_regions(mk, gate)
                assert len(ids) == n, (
                    f"{name} q={q} cluster {j}: {len(ids)} regions against the flood fill's {n}")
                for i in ids:
                    assert any(np.array_equal(got == i, ref == c) for c in range(1, n + 1)), (
                        f"{name} q={q} cluster {j}: region {i} is not any flood-filled component")
                checked += 1
    assert checked > 20, f"only {checked} clusters compared; the grid has stopped covering anything"


def test_connectivity_separates_two_objects_that_look_alike():
    """The measurement that justifies splitting at all. Two same-coloured discs land in one
    cluster, so its centroid sits *between* them -- the midpoint failure this package measured at
    0.067 against the midpoint and 0.262 against either object.

    Mean best IoU: **0.50 by cluster, 1.00 by region.** The first number is the control; if
    clustering alone ever reaches the objects here, this fixture has stopped being about
    connectivity.
    """
    clusters, _n = measure("same colour apart", split=False, edges=None)
    regions, _m = measure("same colour apart", split=True, edges=None)
    assert clusters < 0.65, (
        f"one cluster already scores {clusters:.2f} on two same-coloured objects; the fixture no "
        "longer merges them and the splitting claim is untested")
    assert regions > 0.95, f"splitting scores only {regions:.2f} where 1.00 was measured"

    # And it costs nothing where colour already separates everything.
    assert measure("two colours apart", split=True, edges=None)[0] > 0.95
    assert measure("one object", split=True, edges=None)[0] > 0.95


def test_a_region_is_taught_its_own_centre_and_not_its_clusters():
    """**The centres are what the head is taught with, so they are the point of all of this.** A
    cluster spanning two objects has its centroid *between* them -- measured elsewhere in this
    package at 0.067 against the midpoint and 0.262 against either object -- and a mutant computing
    a region's centre from its whole cluster survived every mask-IoU test here. Present, computed,
    and never compared: the same gap this repository has found in `gain_norm` twice.
    """
    objs, _t, k = SCENES["same colour apart"]
    img, masks, seg = segment(objs, k)
    lab, centres, of_cluster = split_regions(seg.labels, k, img, min_frac=0.01)
    ids, points = pick_regions(seg.labels, lab, centres, of_cluster, 2)

    yy, xx = np.mgrid[0:H, 0:W]
    truth = [(float(xx[m].mean()) / (W - 1), float(yy[m].mean()) / (H - 1)) for m in masks]
    for (x, y), (tx, ty) in zip(points, truth):
        assert abs(x - tx) < 0.03 and abs(y - ty) < 0.03, (
            f"a taught centre is at ({x:.3f}, {y:.3f}) for an object at ({tx:.3f}, {ty:.3f})")

    # The control that says the fixture is about this: the CLUSTER those regions came from has its
    # centroid between the two objects, which is what the head used to be taught.
    merged = seg.labels == max(range(k), key=lambda j: ((seg.labels == j) & (masks[0] | masks[1])).sum())
    mx = float(xx[merged].mean()) / (W - 1)
    midpoint = (truth[0][0] + truth[1][0]) / 2
    assert abs(mx - midpoint) < 0.06, (
        f"the merged cluster's centroid is at x={mx:.3f}, not the midpoint {midpoint:.3f}; the "
        "fixture no longer merges the two objects into one cluster and proves nothing")
    assert all(abs(mx - tx) > 0.15 for tx, _ty in truth), (
        "the midpoint is close to an object, so being taught the midpoint would not be a failure")


def test_two_identical_objects_that_touch_cannot_be_separated_by_any_of_this():
    """The negative result, protected. There is no edge between two identical touching objects, so
    neither connectivity nor the gate can find one: **0.60 for all three**, at every `k` tried.

    A test that only measured the wins would let a later change claim this case works.
    """
    got = {
        "clusters": measure("same colour touching", split=False, edges=None)[0],
        "connected": measure("same colour touching", split=True, edges=None)[0],
        "gated": measure("same colour touching", split=True, edges=0.95)[0],
    }
    for how, v in got.items():
        assert v < 0.75, (
            f"{how} now scores {v:.2f} on two identical touching objects. If that is real it is a "
            f"new capability and this docstring is wrong; all three measured 0.60. {got}")


def test_the_edge_gate_earns_its_place_only_where_k_is_under_declared():
    """`k` is a declaration -- `objects + 1` -- and nothing counts the appearance groups in a room,
    so under-declaration is the normal condition. Scored on the two objects a caller declared:

    | scene, k=3 | clusters | +connected | +edge-gated | k it would have needed |
    |---|---|---|---|---|
    | touch + 1 distractor | 0.83 | 0.83 | 0.83 | 3 |
    | touch + 2 distractors | 0.60 | 0.60 | **0.83** | 5 |
    | apart + 2 distractors | 0.53 | 0.53 | **1.00** | 4 |

    The third row is the mechanism: a target sharing a cluster **with the background** is connected
    to it, so connectivity alone does nothing, and the step at the object's own rim is what severs
    it.

    My first conclusion was that the gate buys nothing over raising `k` -- 0.82 against 0.83 -- and
    that came from a fixture set with no distractors, where `k` was adequate by construction. The
    controls below are what keep that from happening again: the gate must NOT help where `k` is
    already enough.
    """
    for name in ("touch + 2 distractors", "apart + 2 distractors"):
        plain = measure(name, split=True, edges=None)[0]
        gated = measure(name, split=True, edges=0.95)[0]
        assert gated > plain + 0.15, (
            f"{name}: gated {gated:.2f} against connected {plain:.2f}. The gate is the only thing "
            "measured to separate an object from a background cluster it belongs to; if the gap "
            "has closed, either the fixture stopped under-declaring k or the gate stopped working")

    # The controls. Where k is adequate the gate must be neither the reason it works nor a cost.
    for name in ("touch + 1 distractor", "two colours apart", "one object"):
        plain = measure(name, split=True, edges=None)[0]
        gated = measure(name, split=True, edges=0.95)[0]
        assert gated == pytest.approx(plain, abs=0.05), (
            f"{name}: the gate moved a scene it should not touch, {plain:.2f} -> {gated:.2f}")


def test_the_size_floor_changes_the_count_and_not_the_accuracy():
    """**A best-IoU metric is blind to this, which is why the count is measured beside it.** At
    k=4 on a two-object scene the floor makes no difference to IoU at all -- 1.00 either way, the
    objects are found -- and takes the region count from **924 to 13**. Without it the picture is a
    speckle field with the right answer buried in it.
    """
    objs, targets, _k = SCENES["two colours apart"]
    img, masks, seg = segment(objs, 4)
    scored = {}
    for frac in (0.0, 0.01):
        lab, centres, _of = split_regions(seg.labels, 4, img, min_frac=frac)
        regions = [lab == i for i in range(1, len(centres) + 1)]
        scored[frac] = (best_iou(regions, [masks[i] for i in targets]), len(centres))
    assert scored[0.0][0] == pytest.approx(scored[0.01][0], abs=0.05), (
        f"the floor moved the IoU, {scored[0.0][0]:.2f} -> {scored[0.01][0]:.2f}; it is supposed "
        "to drop speckle, so if it is dropping accuracy the fraction is too large")
    assert scored[0.0][1] > 10 * scored[0.01][1], (
        f"the floor only takes {scored[0.0][1]} regions to {scored[0.01][1]}; if speckle has "
        "stopped appearing then this floor is no longer earning its place")


def test_a_slot_with_no_region_says_so_rather_than_pointing_at_the_background():
    """A head with 2n outputs needs 2n numbers whatever the frame contained, so `pick_regions`
    always returns `n` of each. The missing ones carry `None`, not 0: region ids are 1-based and 0
    is the background, so painting id 0 would colour the whole wall as an object."""
    flat = np.full((H, W, 3), 0.4, dtype="f4")
    seg = FrameSegmenter(3)
    seg(flat)
    lab, centres, of_cluster = split_regions(seg.labels, 3, flat, min_frac=0.01)
    ids, points = pick_regions(seg.labels, lab, centres, of_cluster, 3)
    assert len(ids) == 3 and len(points) == 3
    assert None in ids, f"a featureless frame produced three real regions: {ids}"
    assert all(0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 for x, y in points)

    # And the painter must skip the missing slot rather than claim the background for it.
    painted = slot_label_map(lab, [None, None, None])
    assert not painted.any(), "a slot with no region painted something"


def test_dropping_the_largest_region_generalises_the_rule_it_replaced():
    """With `k = objects + 1` and no splitting there are exactly `objects + 1` regions, so "all but
    the biggest" and "the biggest n after the biggest" are the same set -- which is why the change
    is safe. They stop being the same as soon as one cluster splits into more pieces than that."""
    objs, _t, k = SCENES["two colours apart"]
    img, _masks, seg = segment(objs, k)
    centres = seg(img)
    sizes = [(seg.labels == j).sum() for j in range(k)]
    old = sorted(((j, centres[j]) for j in sorted(range(k), key=lambda j: sizes[j])[:2]),
                 key=lambda jp: jp[1][0])

    lab, rcentres, of_cluster = split_regions(seg.labels, k, img, min_frac=0.01)
    ids, points = pick_regions(seg.labels, lab, rcentres, of_cluster, 2)
    for (x, y), (_j, (ox, oy)) in zip(points, old):
        assert x == pytest.approx(ox, abs=0.02) and y == pytest.approx(oy, abs=0.02), (
            f"the new rule picked {points} where the old one picked {[p for _j, p in old]}")

    # The largest region really is the one dropped, and it is found by area rather than by index.
    biggest = max(range(1, len(rcentres) + 1), key=lambda i: (lab == i).sum())
    assert biggest not in ids, "the largest region survived; on a plain wall that region IS the wall"


def test_the_two_rules_diverge_once_a_cluster_splits_into_more_pieces_than_slots():
    """The equivalence above holds only at `k = objects + 1`, and a mutant that keeps the *smallest*
    `n` regions survives every fixture where it does -- there, "all but the biggest" and "the
    smallest n" are the same set.

    Three same-coloured discs of different sizes give four regions for two slots, and the two rules
    then disagree: the shipped one takes the two largest after the background, the other takes the
    tiny dot.
    """
    img = np.full((H, W, 3), 0.35)
    big, mid, tiny = disc((0.22, 0.5), 0.16), disc((0.60, 0.5), 0.11), disc((0.88, 0.8), 0.06)
    for m in (big, mid, tiny):
        img[m] = RED
    img = (img + np.random.default_rng(3).standard_normal(img.shape) * 0.01).astype("f4")

    seg = FrameSegmenter(2)
    for _ in range(3):
        seg(img)
    lab, centres, of_cluster = split_regions(seg.labels, 2, img, min_frac=0.01)
    assert len(centres) > 3, (
        f"only {len(centres)} regions; the fixture no longer produces more pieces than slots and "
        "cannot tell the two selection rules apart")

    ids, _points = pick_regions(seg.labels, lab, centres, of_cluster, 2)
    kept = [(lab == i).sum() for i in ids if i is not None]
    dropped = [(lab == i).sum() for i in range(1, len(centres) + 1) if i not in ids]
    assert min(kept) > min(dropped), (
        f"the rule kept regions of {sorted(kept)} pixels and dropped {sorted(dropped)}; it is "
        "taking the smallest rather than the largest-after-the-background")
    assert max(kept) < max(dropped), "the background survived the drop"


def test_the_gate_is_relative_and_refuses_a_quantile_that_is_not_one():
    """Relative to the frame's own gradients, for the reason every threshold here is relative: the
    strongest edge in a scene is the boundary between two things, whatever the lighting. Swept
    0.70-0.98 the effect is monotone, so 0.95 is a cost choice and not an optimum."""
    img, _masks = scene(SCENES["two colours apart"][0])
    marked = [float(edge_gate(img, q).mean()) for q in (0.70, 0.90, 0.98)]
    assert marked[0] > marked[1] > marked[2], f"the quantile stopped ordering the gate: {marked}"
    assert marked[1] == pytest.approx(0.10, abs=0.06), (
        f"q=0.90 marks {marked[1]:.0%} of the frame; a quantile should mark about a tenth, so the "
        "gradient has become degenerate")
    # Twice the brightness must not change which pixels are marked -- that is what relative means.
    assert np.array_equal(edge_gate(img, 0.95), edge_gate(img * 2.0, 0.95))
    for bad in (0.0, 1.0, -0.5, 2.0):
        with pytest.raises(ValueError, match="quantile"):
            edge_gate(img, bad)
    with pytest.raises(ValueError, match="edges needs the frame"):
        split_regions(np.zeros((4, 4), dtype=int), 2, None, edges=0.9)
    with pytest.raises(ValueError, match="gate shape"):
        connected_regions(np.ones((4, 4), dtype=bool), np.ones((5, 5), dtype=bool))


def per_cluster_reference(cluster_labels, k, frame, *, edges=None, min_frac=0.01):
    """`split_regions` the obvious way: components of each cluster's mask, one call per cluster.

    The shipped version does **one** pass over the whole field, because a region is a maximal
    connected set of same-cluster pixels and "same cluster and not gated" is the same adjacency.
    That is an argument, so this is here to check it by running it.
    """
    lab = np.asarray(cluster_labels)
    gate = edge_gate(frame, edges) if edges is not None else None
    floor = max(int(min_frac * lab.size), 1)
    out = np.zeros(lab.shape, dtype=np.int32)
    h, w = lab.shape
    yy, xx = np.mgrid[0:h, 0:w]
    centres, clusters = [], []
    for j in range(k):
        m = lab == j
        if not m.any():
            continue
        comp, ids = connected_regions(m, gate)
        for i in ids:
            piece = comp == i
            if piece.sum() < floor:
                continue
            centres.append((float(xx[piece].mean()) / max(w - 1, 1),
                            float(yy[piece].mean()) / max(h - 1, 1)))
            clusters.append(j)
            out[piece] = len(centres)
    return out, centres, clusters


def test_one_pass_over_the_field_gives_the_same_regions_as_one_pass_per_cluster():
    """Identical label map, identical centres, identical cluster ownership, over every fixture and
    both gate settings -- **including the numbering**, which both the display and the head key on."""
    for name, (objs, _t, k) in SCENES.items():
        for q in (None, 0.95):
            img, _masks, seg = segment(objs, k)
            ref = per_cluster_reference(seg.labels, k, img, edges=q)
            got = split_regions(seg.labels, k, img, edges=q, min_frac=0.01)
            assert np.array_equal(ref[0], got[0]), f"{name} q={q}: the label maps differ"
            assert ref[1] == got[1], f"{name} q={q}: the centres differ"
            assert ref[2] == got[2], f"{name} q={q}: the cluster ownership differs"


def test_it_costs_a_few_milliseconds_and_the_worst_case_is_a_shape_not_a_size():
    """Measured on this host at 72x96, gated: **1.5-2.0 ms** on five of these seven fixtures and
    **13.4-14.3 ms** on the two where the background is carved into interleaved noise clusters. The
    difference is the number of hook rounds -- 20 for a region that snakes across the frame, 4 for a
    disc -- so the cost follows the largest region's *shape*, not the pixel count.

    The worst case is 43% of a 33 ms frame, and with the segmenter's 2.5 ms and the backbone plus
    pipeline's 0.58 the demo still clears 30 Hz. Host figures, not target ones.

    **What is asserted is the ratio, not the milliseconds.** The first version of this test bounded
    the worst fixture at 33 ms and failed inside the hour on this same machine, at 34.1, because the
    pre-commit hook runs the whole suite alongside it -- which is precisely the failure this project
    already recorded once for `argmind.bench`: an absolute timing is a statement about one machine at
    one moment, and as a gate it fails on someone else's laptop and gets switched off. A ratio taken
    inside one run survives moving machines.
    """
    cost = {}
    for name, (objs, _t, k) in SCENES.items():
        img, _masks, seg = segment(objs, k)
        split_regions(seg.labels, k, img, edges=0.95, min_frac=0.01)      # warm
        t0 = time.perf_counter()
        for _ in range(4):
            split_regions(seg.labels, k, img, edges=0.95, min_frac=0.01)
        cost[name] = (time.perf_counter() - t0) / 4 * 1e3

    # The cost follows the largest region's SHAPE. Every fixture is the same 72x96 with the same k,
    # so a pixel count cannot explain the spread -- and if it ever flattens, the hook-round
    # explanation in the docstring is wrong and the figures above need re-deriving.
    sprawling = cost["same colour apart"]
    compact = cost["apart + 2 distractors"]
    assert sprawling > 2.0 * compact, (
        f"a noise-carved background now costs {sprawling:.1f} ms against a clean scene's "
        f"{compact:.1f} at the same frame size and the same k. The spread IS the hook-round "
        "explanation; without it the cost model in this docstring is unsupported")

def taught_centres(name, *, split, edges, min_frac=0.01):
    """Where each slot's label lands, scored against the DECLARED targets, slot i vs target i.

    Both are leftmost-first, so this scores a **role** rather than an identity -- the distinction
    this package measured at 0.129 for roles against 0.251 for identity through a crossing.
    """
    objs, targets, k = SCENES[name]
    img, masks, seg = segment(objs, k)
    yy, xx = np.mgrid[0:H, 0:W]
    truth = [(float(xx[masks[i]].mean()) / (W - 1), float(yy[masks[i]].mean()) / (H - 1))
             for i in targets]
    if not split:
        sizes = [(seg.labels == j).sum() for j in range(k)]
        keep = sorted(range(k), key=lambda j: sizes[j])[:len(targets)]
        centres = seg(img)
        points = sorted((centres[j] for j in keep), key=lambda p: p[0])
    else:
        lab, centres, of_cluster = split_regions(seg.labels, k, img, edges=edges,
                                                 min_frac=min_frac)
        _ids, points = pick_regions(seg.labels, lab, centres, of_cluster, len(targets))
    return float(np.mean([np.hypot(p[0] - t[0], p[1] - t[1]) for p, t in zip(points, truth)]))


def test_what_the_head_is_taught_improves_in_one_case_and_costs_in_another():
    """**The end-to-end number, and it is less flattering than the IoU table.** Distance from each
    slot's taught centre to the target it is supposed to track:

    | scene | clusters | +connected | +edge-gated |
    |---|---|---|---|
    | same colour apart | 0.213 | **0.000** | 0.000 |
    | same colour touching | **0.053** | 0.188 | 0.187 |
    | touch + 2 distractors | 0.183 | 0.183 | 0.183 |
    | apart + 2 distractors | 0.179 | 0.179 | 0.179 |
    | mean over all seven | 0.092 | **0.081** | 0.081 |

    Two things this says that the IoU table does not.

    **Splitting costs on the case nothing can separate.** Two identical touching discs are one
    region, so the second slot gets a background fragment instead of the pair's shared centroid:
    0.053 becomes 0.188. That is the documented limit of `pick_regions` -- only one cluster is
    dropped as background -- arriving as a number.

    **The edge gate does not improve what the head is taught, at all.** It takes region IoU from
    0.53 to 1.00 on the distractor scenes and moves the taught centre by 0.000. It changes a
    region's *shape*, not which region wins by area, so it is a segmentation-picture improvement and
    is claimed as nothing more.
    """
    assert taught_centres("same colour apart", split=False, edges=None) > 0.15
    assert taught_centres("same colour apart", split=True, edges=None) < 0.02, (
        "splitting no longer puts the taught label on each of two same-coloured objects, which is "
        "the whole reason it is on by default")

    cost = taught_centres("same colour touching", split=True, edges=None)
    assert cost > taught_centres("same colour touching", split=False, edges=None), (
        "splitting no longer costs anything on two identical touching objects. If that is real it "
        "is good news and this docstring's 0.053 -> 0.188 is stale -- check it rather than "
        "loosening this")

    for name in ("touch + 2 distractors", "apart + 2 distractors"):
        plain = taught_centres(name, split=True, edges=None)
        gated = taught_centres(name, split=True, edges=0.95)
        assert gated == pytest.approx(plain, abs=0.02), (
            f"{name}: the gate moved the taught centre {plain:.3f} -> {gated:.3f}. It is claimed to "
            "improve the picture only; if it now helps the head, say so where the claim is made")

    every = list(SCENES)
    means = [float(np.mean([taught_centres(n, split=s, edges=e) for n in every]))
             for s, e in ((False, None), (True, None))]
    assert means[1] < means[0], (
        f"splitting is no longer a net improvement over the scene set: {means[0]:.3f} -> "
        f"{means[1]:.3f}. The win is concentrated in one case, so a change to any other could "
        "flip this")


def test_the_wiring_the_demo_uses_is_the_wiring_under_test():
    """`regions_for_slots` is what `main()` calls, and it lives at module level for the reason
    `slot_label_map` does: the version inside the loop coloured every region with the wrong id and
    no test could reach it. Four defects in this package have had that shape in one day.

    Both paths are driven here, and the `split=False` one must reproduce the centres the segmenter
    itself returns -- it recomputes them from the labels rather than calling the detector again,
    because a second call would advance the warm start the whole index-stability claim rests on.
    """
    from demo.pointer import regions_for_slots

    objs, targets, k = SCENES["same colour apart"]
    img, masks, seg = segment(objs, k)
    from_detector = seg(img)                          # the centres the class itself reports

    rmap, of_cluster, ids, points = regions_for_slots(seg, img, 2, split=False)
    assert np.array_equal(rmap, seg.labels), "the un-split path stopped painting from the clusters"
    assert of_cluster is None, "the un-split path claimed regions have clusters; they ARE clusters"
    assert all(i in range(k) for i in ids)
    for i, p in zip(ids, points):
        assert p == pytest.approx(from_detector[i], abs=1e-9), (
            f"slot centre {p} for cluster {i} disagrees with the segmenter's own {from_detector[i]}")

    split_map, split_of, split_ids, split_points = regions_for_slots(
        seg, img, 2, split=True, edges=0.95)
    assert split_map.max() > seg.labels.max(), "splitting produced no more regions than clusters"
    assert len(split_ids) == 2 and len(split_points) == 2

    # The two paths disagree here, which is the point: this is the fixture splitting was built for.
    yy, xx = np.mgrid[0:H, 0:W]
    truth = [(float(xx[m].mean()) / (W - 1), float(yy[m].mean()) / (H - 1)) for m in masks]
    err = [float(np.mean([np.hypot(p[0] - t[0], p[1] - t[1]) for p, t in zip(pts, truth)]))
           for pts in (points, split_points)]
    assert err[1] < err[0] / 5, (
        f"split {err[1]:.3f} against unsplit {err[0]:.3f}; the wiring is not carrying the "
        "improvement the functions were measured to give")
