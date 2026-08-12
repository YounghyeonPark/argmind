# SPDX-License-Identifier: Apache-2.0
"""Segmenting a single frame by colour and texture, with no motion and no background model.

Every segmentation failure measured in this package came from the background model, and both are
properties of the mechanism rather than of the scene: it absorbs a stationary object (0% detected at
every rate) and it leaves a trail the same size as the object (precision 0.50, area 2.05x).

**A per-frame segmentation cannot have a trail and cannot lose a stationary object.** Those two
facts are what this file measures, along with the two things that make the shipped defaults what
they are: texture is harmful when colour suffices, and one k-means iteration is not a cheaper answer
but a wrong one.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from demo.label_sources import FrameSegmenter

H, W = 72, 96


def disc_mask(p, r=0.16):
    yy, xx = np.mgrid[0:H, 0:W]
    return np.hypot(xx - p[0] * W, yy - p[1] * H) <= r * W


def scene_colour(p, seed=0, base=0.35, bright=0.80):
    """Brightness differs, texture does not. Colour alone should segment this exactly."""
    m = disc_mask(p)
    img = np.full((H, W, 3), base) + m[:, :, None] * (bright - base)
    g = np.random.default_rng(seed)
    return (img + g.standard_normal(img.shape) * 0.01).astype("f4"), m


def scene_texture(p, seed=0, base=0.50, amp=0.30, period=3.0):
    """The **same mean brightness** inside and out; the object is finely striped, the wall smooth.

    This is what makes the texture claim testable rather than a restatement of the colour result.
    The stripes average to the wall's level, so a descriptor that only reads one pixel's value has
    nothing to go on -- which is the control below.
    """
    m = disc_mask(p)
    yy, xx = np.mgrid[0:H, 0:W]
    img = np.full((H, W), base) + m * (np.sin(2 * np.pi * xx / period) * amp)
    g = np.random.default_rng(seed)
    out = np.repeat((img + g.standard_normal(img.shape) * 0.01)[:, :, None], 3, axis=2)
    return out.astype("f4"), m


def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


def quality(scene, texture, *, moving=True, frames=8, k=2, **kw):
    seg = FrameSegmenter(k, texture=texture, **kw)
    got = []
    for i in range(frames):
        p = (0.30 + (0.04 * i if moving else 0.0), 0.5)
        img, mask = scene(p, seed=i)
        seg(img)
        got.append(max(iou(seg.labels == j, mask) for j in range(k)))
    return float(np.mean(got))


def test_the_texture_fixture_cannot_be_solved_by_colour():
    """The control that makes every texture number below mean something. If a colour-only
    descriptor succeeds here, the fixture is leaking brightness."""
    assert quality(scene_texture, texture=False) < 0.45, (
        "a colour-only descriptor now segments the same-brightness fixture; it is leaking and the "
        "texture claim is untested")
    # And the object really is there to be found, which the texture descriptor demonstrates.
    assert quality(scene_texture, texture=True) > 0.5


def test_a_single_frame_is_enough_and_stillness_costs_nothing():
    """The whole point against a background model. Measured IoU:

    | fixture | descriptor | moving | still |
    |---|---|---|---|
    | colour differs | colour only | **1.00** | **1.00** |
    | same mean colour | colour+texture | **0.63** | 0.62 |

    A background model detects a stationary object 0% of the time at every rate. Here `still`
    equals `moving`, because there is nothing in the method that could tell the difference.
    """
    for scene, texture in ((scene_colour, False), (scene_texture, True)):
        moving = quality(scene, texture, moving=True)
        still = quality(scene, texture, moving=False)
        assert still == pytest.approx(moving, abs=0.05), (
            f"still {still:.2f} against moving {moving:.2f}; a per-frame method has no way to "
            "know, so a gap means state has crept in somewhere")
    assert quality(scene_colour, texture=False) > 0.95, "colour no longer segments exactly"


def test_texture_is_harmful_when_colour_suffices_which_is_why_it_is_off():
    """1.00 falls to **0.25** -- worse than the control -- when texture is added to a fixture
    colour already solves. Two more dimensions do not merely dilute the separation, they dominate
    it: the stripes-versus-smooth axis carves the frame somewhere else entirely.

    That figure was 0.71 in an exploratory script that re-seeded k-means at random every frame. The
    shipped class warm-starts its centres, which is what makes a cluster's index stable, and it
    costs accuracy in exactly the configuration that was already the wrong one to use.
    """
    assert FrameSegmenter(2).texture is False
    clean = quality(scene_colour, texture=False)
    with_texture = quality(scene_colour, texture=True)
    assert with_texture < clean * 0.5, (
        f"texture now costs only {clean / with_texture:.2f}x on a colour fixture ({clean:.2f} -> "
        f"{with_texture:.2f}); if it has become harmless the default could change and the "
        "docstring needs re-measuring")


def test_a_cluster_index_stays_with_the_same_object():
    """Warm-started centres are what make the index stable. k-means labels are arbitrary per run,
    so a fresh start every frame would permute them and the head's slots would reshuffle -- which
    this package measured at 0.256 against 0.129 for consistently ordered labels."""
    seg = FrameSegmenter(2)
    holding = []
    for i in range(12):
        img, mask = scene_colour((0.30 + 0.03 * i, 0.5), seed=i)
        seg(img)
        holding.append(int(np.argmax([(seg.labels == j)[mask].sum() for j in range(2)])))
    assert len(set(holding)) == 1, (
        f"the object moved between cluster indices {sorted(set(holding))} across a run; the "
        "warm start is not holding and the head would see its slots swap")

    # And a reset really does forget, or "warm" would be an overstatement.
    seg.reset()
    assert seg.labels is None


def test_one_iteration_is_a_wrong_answer_not_a_cheap_one():
    """IoU 0.16 at one iteration against 1.00 at three, so the floor is measured rather than
    chosen. The constructor refuses fewer than two and says why."""
    assert quality(scene_colour, texture=False, iters=3) > 0.95
    with pytest.raises(ValueError, match="iters must be at least 2"):
        FrameSegmenter(2, iters=1)
    with pytest.raises(ValueError, match="k must be at least 2"):
        FrameSegmenter(1)
    with pytest.raises(ValueError, match="radius must be at least 1"):
        FrameSegmenter(2, radius=0)


def test_it_costs_a_few_milliseconds_of_a_frame():
    """Measured on this host at 72x96: 1.07 ms at k=2, 2.47 at k=3, 3.90 at k=4, warm-started. A
    30 Hz frame is 33 ms, so this is affordable -- and it is a host figure, not a target one."""
    img, _m = scene_colour((0.4, 0.5), seed=1)
    seg = FrameSegmenter(3)
    seg(img)                                     # warm up, as the loop would be
    t0 = time.perf_counter()
    for _ in range(20):
        seg(img)
    ms = (time.perf_counter() - t0) / 20 * 1e3
    assert ms < 15.0, (
        f"segmenting takes {ms:.1f} ms of a 33 ms frame; if that is real on this host the demo "
        "cannot run it every frame and the cadence-free claim has to go")


def test_an_empty_cluster_keeps_its_slot():
    """Shifting the others along would renumber the head's outputs mid-run, which is the
    permutation failure in its quietest form."""
    flat = np.full((H, W, 3), 0.4, dtype="f4")
    seg = FrameSegmenter(3)
    got = seg(flat)
    assert len(got) == 3, "a degenerate frame returned fewer centres than clusters"
    assert all(0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 for x, y in got)
    assert seg.labels.shape == (H, W)

def test_the_cluster_a_region_came_from_decides_its_colour():
    """The reported bug: the label id changed whenever anything moved.

    The display painted cluster *n* with the id of slot *n*, and those are different orderings --
    the background is dropped **by area**, the survivors are re-sorted **by x**, and the appearance
    matcher **permutes** them again. So the pairing was arbitrary and shifted every frame.

    `slot_label_map` closes it by carrying which cluster each slot came from. It also drops the
    background cluster to 0: a cluster map tiles the whole frame by construction, so painting every
    cluster makes the regions read as a partition of the picture rather than as objects, which is
    the other half of what was reported.
    """
    from demo.pointer import slot_label_map

    clusters = np.array([[0, 0, 1, 1],
                         [2, 2, 1, 1],
                         [2, 2, 0, 0]])

    # Slot 0 came from cluster 2, slot 1 from cluster 1. Cluster 0 is the dropped background.
    out = slot_label_map(clusters, [2, 1])
    assert (out[clusters == 2] == 1).all(), "slot 0's cluster is not painted 1"
    assert (out[clusters == 1] == 2).all(), "slot 1's cluster is not painted 2"
    assert (out[clusters == 0] == 0).all(), "the dropped cluster was painted as a region"

    # The same clusters with the slots swapped must produce the swapped map, or the colour is
    # following the cluster index rather than the slot -- which is the bug.
    swapped = slot_label_map(clusters, [1, 2])
    assert (swapped[clusters == 1] == 1).all()
    assert (swapped[clusters == 2] == 2).all()
    assert not np.array_equal(out, swapped)

    # Degenerate inputs answer rather than raising: no slots means everything is background.
    assert not slot_label_map(clusters, []).any()
    assert not slot_label_map(clusters, None).any()


def test_the_background_cluster_is_the_largest_and_is_dropped_by_area():
    """`k` is `objects + 1` because one cluster is the background, and it must be identified by
    **area** rather than by index -- a cluster's index is stable but which cluster is the
    background is not, so dropping index 0 would drop a real object as soon as the subject moved
    to a different part of the frame."""
    from demo.pointer import slot_label_map

    seg = FrameSegmenter(3)
    img, mask = scene_colour((0.35, 0.5), seed=1)
    seg(img)
    sizes = [(seg.labels == j).sum() for j in range(seg.k)]
    keep = sorted(range(seg.k), key=lambda j: sizes[j])[:2]
    assert int(np.argmax(sizes)) not in keep, (
        "the largest cluster survived the drop; on a plain wall that cluster IS the wall")

    painted = slot_label_map(seg.labels, keep)
    covered = float(((painted > 0) & mask).sum() / max(mask.sum(), 1))
    assert covered > 0.6, (
        f"the kept clusters cover only {covered:.0%} of the object; dropping by area removed "
        "something that was not the background")
    # And the picture is not a tiling: most of a plain wall must stay unpainted.
    assert float((painted == 0).mean()) > 0.4


def test_cluster_indices_survive_a_scene_whose_statistics_move():
    """The first suspicion for the flickering ids was that `descriptors` standardises **per frame**
    while the centres are carried **across** frames, so the space moves under them.

    Measured and **refuted**: over a scene with a large subject crossing *and* a global brightness
    drift, the cluster holding the subject never changes index -- a 0.0% switch rate, and the same
    for a running or a fixed standardisation. The instability was the display's cluster-to-slot
    pairing, not the clustering. Kept because it is the obvious next thing to blame.
    """
    seg = FrameSegmenter(3)
    holding = []
    for k in range(120):
        drift = 0.06 * np.sin(2 * np.pi * k / 90.0)
        a = 2 * np.pi * k / 100.0
        cx = 0.5 + 0.30 * np.cos(a)
        img, mask = scene_colour((cx, 0.5), seed=k, base=0.34 + drift)
        seg(img)
        if k < 20:
            continue
        holding.append(int(np.argmax([(seg.labels == j)[mask].sum() for j in range(seg.k)])))
    switches = sum(1 for a, b in zip(holding, holding[1:]) if a != b)
    assert switches == 0, (
        f"the subject's cluster changed index {switches} times over {len(holding)} frames; if the "
        "clustering itself has become unstable, the per-frame standardisation is worth revisiting "
        "after all")
