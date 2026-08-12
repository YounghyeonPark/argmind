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
