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

def flat_disc(p, r=0.14, bright=0.85, base=0.35, seed=0, present=True):
    """A flat-topped object: uniform interior, hard edge — a person in a plain shirt, which is
    what a Gaussian blob is not. The blob fixture cannot show this effect at all, because it has
    no uniform interior to absorb."""
    yy, xx = np.mgrid[0:H, 0:W]
    d = (np.hypot(xx - p[0] * W, yy - p[1] * H) <= r * W) if present else np.zeros((H, W), bool)
    img = np.full((H, W, 3), base) + d[:, :, None] * (bright - base)
    return (img + np.random.default_rng(seed).standard_normal(img.shape) * 0.01).astype("f4"), d


def coverage(selective, speed, frames=300, warm=60, floor=0.25):
    """How much of the object, and of its eroded core, the mask keeps."""
    from scipy.ndimage import binary_erosion
    loc = BackgroundLocator(1, rate=0.03, warmup=15, suppress=0.10, selective=selective)
    whole, core = [], []
    for k in range(frames):
        p = (0.25 + speed * k, 0.5)
        img, mask = flat_disc(p, seed=k)
        loc(img)
        if loc.foreground is None or k < warm:
            continue
        fg = loc.foreground
        keep = fg >= floor * max(float(fg.max()), 1e-9)
        inner = binary_erosion(mask, iterations=3)
        whole.append(float((keep & mask).sum() / max(mask.sum(), 1)))
        core.append(float((keep & inner).sum() / max(inner.sum(), 1)))
    assert whole, "no frames measured"
    return float(np.mean(whole)), float(np.mean(core))


def test_a_plain_background_labels_only_the_edges_of_a_still_object():
    """The reported symptom, reproduced and explained.

    A running mean absorbs anything that stops moving, so the interior stops differing from the
    background and only the boundary is left. Measured on a flat-topped object at 0.05 px/frame:

    | update | whole object | **core** |
    |---|---|---|
    | plain | 0.10 | **0.00** |
    | selective | 0.40 | **0.31** |

    A Gaussian-blob fixture cannot show this — it has no uniform interior to absorb — which is why
    this file carries a flat disc.
    """
    pytest.importorskip("scipy")
    plain_whole, plain_core = coverage(False, 0.0005)
    sel_whole, sel_core = coverage(True, 0.0005)
    assert plain_core < 0.1, (
        f"a plain background now keeps {plain_core:.2f} of a still object's core; if that is real "
        "the absorption this file explains has stopped happening")
    assert sel_core > plain_core * 3, (sel_core, plain_core)
    assert sel_whole > plain_whole


def test_selective_update_is_off_by_default_and_the_reason_is_the_path():
    """It is better for a still subject and much worse for a moving one, so it is a flag.

    Selective never forgets where an object has *been*, so a traverser leaves a permanent trail and
    the peaks land on it: pairing falls from 80% to 9% at three objects. Every other measurement in
    this package was taken with it off, which is the other reason the default cannot change quietly.
    """
    assert BackgroundLocator(1).selective is False

    def pairing(n_true, selective, frames=600, warm=100):
        loc = BackgroundLocator(n_true, rate=0.03, warmup=15, suppress=0.10, selective=selective)
        ok = tot = 0
        for k in range(frames):
            pts = [orbit(k, i, n_true) for i in range(n_true)]
            got = loc(scene(pts, PALETTE[:n_true], seed=k))
            if got is None or k < warm:
                continue
            d = np.array([[np.hypot(f[0] - p[0], f[1] - p[1]) for p in pts] for f in got])
            tot += 1
            ok += (d.min(axis=0) < 0.06).all() and len(set(np.argmin(d, axis=1))) == n_true
        return ok / max(tot, 1)

    plain, sel = pairing(3, False), pairing(3, True)
    assert plain > 0.6, f"plain pairing fell to {plain:.0%}"
    assert sel < plain * 0.5, (
        f"selective now pairs at {sel:.0%} against plain's {plain:.0%}; if the path trail has "
        "stopped hurting, selective may be the better default and this needs re-measuring")


def test_the_expected_cost_of_selective_update_is_not_the_one_that_bites():
    """Ghosting was the predicted objection and it is measured to be wrong, which is worth keeping
    so it is not raised again: sixty frames after an object leaves, the frame fraction still
    labelled is **0.234 plain against 0.082 selective**. The plain model has a half-absorbed copy
    to unlearn; the selective one never absorbed it."""
    def ghost(selective, n=400, gone_at=250, floor=0.25):
        loc = BackgroundLocator(1, rate=0.03, warmup=15, suppress=0.10, selective=selective)
        after = []
        for k in range(n):
            img, _m = flat_disc((0.30 + 0.0006 * k, 0.5), seed=k, present=k < gone_at)
            loc(img)
            if loc.foreground is None or k < gone_at + 60:
                continue
            fg = loc.foreground
            after.append(float((fg >= floor * max(float(fg.max()), 1e-9)).mean()))
        return float(np.mean(after))

    plain, sel = ghost(False), ghost(True)
    assert sel < plain, (
        f"selective now ghosts more than plain ({sel:.3f} vs {plain:.3f}), which is what I "
        "originally predicted; if that has become true the docstring must be corrected")

def test_half_of_what_a_difference_detects_is_where_the_object_WAS():
    """Why a filled region is not the whole story, and it is geometry rather than a defect.

    For a translating object, `|frame - background|` is a crescent: half of it sits where the
    object now is, and half where it just left. Measured on a flat-topped object -- `on-object` is
    **exactly half** of `painted` in every configuration:

    | background | floor | painted | on-object | above-floor |
    |---|---|---|---|---|
    | plain | 0.10 | 197.7 | 98.8 | 197.7 |
    | plain | 0.40 | 79.0 | 39.6 | 79.0 |
    | selective | 0.10 | 479.9 | 239.9 | 480.1 |
    | selective | 0.40 | 479.9 | 239.9 | 480.1 |

    That uniform 0.5 was checked rather than trusted, because this project treats a uniformly tidy
    column as a broken measurement. It is real: `painted` tracks `above-floor` exactly, so the
    floor is doing its job and the halving is the dipole in another guise.

    And note the floor does **nothing** under selective -- 480 above it at both 0.10 and 0.40 --
    because a background that never learns the object leaves a roughly *uniform* difference over
    it, so any peak-relative threshold in that range selects the same pixels.
    """
    pytest.importorskip("scipy")
    from demo.label_sources import BackgroundLocator, partition

    def counts(selective, floor, frames=300, warm=80):
        loc = BackgroundLocator(1, rate=0.03, warmup=15, suppress=0.10, selective=selective)
        rows = []
        for k in range(frames):
            p = (0.25 + 0.0005 * k, 0.5)
            img, mask = flat_disc(p, seed=k)
            got = loc(img)
            if loc.foreground is None or k < warm:
                continue
            lab = partition(loc.foreground, got or [p], floor=floor)
            painted = lab > 0
            above = (loc.foreground >= floor * loc.foreground.max()).sum()
            rows.append((painted.sum(), (painted & mask).sum(), above))
        a = np.array(rows, dtype=float)
        return a[:, 0].mean(), a[:, 1].mean(), a[:, 2].mean()

    for selective in (False, True):
        for floor in (0.10, 0.40):
            painted, on_object, above = counts(selective, floor)
            assert painted == pytest.approx(above, rel=0.02), (
                f"painted {painted:.0f} does not track above-floor {above:.0f}; the radius clamp "
                "is dominating and the floor measurement means nothing")
            assert on_object / painted == pytest.approx(0.5, abs=0.08), (
                f"{selective}/{floor}: on-object is {on_object / painted:.2f} of painted, not the "
                "half that a translating object's crescent implies")

    # The floor is inert under selective, which is what a uniform difference over the object means.
    lo = counts(True, 0.10)[2]
    hi = counts(True, 0.40)[2]
    assert lo == pytest.approx(hi, rel=0.02), (
        f"the floor now changes what selective selects ({lo:.0f} vs {hi:.0f}); the difference over "
        "the object has stopped being uniform and the flag's docstring needs re-measuring")

def test_selective_covers_the_object_and_paints_its_trail_in_equal_measure():
    """Why a filled region still does not read as a shape, measured.

    Selective covers the whole object -- recall 0.98 -- and paints an **equal area where it was**:
    precision pinned at 0.50, area exactly 2x the object. The trail cannot heal, because a
    background that never re-learns inside the mask never forgets.

    | background | precision | recall | area/object |
    |---|---|---|---|
    | running mean | 0.46 | 0.37 | 0.80 |
    | mean, selective | 0.48 | **0.98** | **2.05** |
    | *the true wall* | **1.00** | **1.00** | **1.00** |

    The last row is the control that makes the others readable: the metric reports 1.00 when the
    reference is right, so 0.5 is the trail and not a broken measurement. It was checked because a
    precision pinned near 0.5 across every method and every setting is exactly the shape this
    project treats as evidence the measurement is broken.
    """
    from demo.label_sources import _grey

    def track(selective, speed=0.002, frames=400, floor=0.25, warm=60):
        loc = BackgroundLocator(1, rate=0.03, warmup=15, suppress=0.10, selective=selective)
        prec, rec, area = [], [], []
        for k in range(frames):
            p = (0.20 + speed * k, 0.5)
            img, mask = flat_disc(p, seed=k)
            loc(img)
            if loc.foreground is None or k < warm:
                continue
            fg = loc.foreground
            keep = fg >= floor * max(float(fg.max()), 1e-9)
            prec.append((keep & mask).sum() / max(keep.sum(), 1))
            rec.append((keep & mask).sum() / max(mask.sum(), 1))
            area.append(keep.sum() / max(mask.sum(), 1))
        return float(np.mean(prec)), float(np.mean(rec)), float(np.mean(area))

    # The control first: a perfect reference must score 1.00, or nothing below means anything.
    wall = np.full((H, W), 0.35)
    hits = []
    for k in range(90, 400, 40):
        p = (0.20 + 0.002 * k, 0.5)
        img, mask = flat_disc(p, seed=k)
        fg = np.abs(_grey(img).astype(np.float64) - wall)
        keep = fg >= 0.25 * max(float(fg.max()), 1e-9)
        hits.append(((keep & mask).sum() / max(keep.sum(), 1),
                     (keep & mask).sum() / max(mask.sum(), 1)))
    ideal = np.mean(hits, axis=0)
    assert ideal[0] > 0.95 and ideal[1] > 0.95, (
        f"a perfect reference scores precision {ideal[0]:.2f} recall {ideal[1]:.2f}; the metric "
        "is broken and every number in this test is meaningless")

    plain = track(False)
    sel = track(True)
    assert sel[1] > 0.85, f"selective recall fell to {sel[1]:.2f}; it should cover the object"
    assert plain[1] < sel[1] * 0.6, f"plain recall {plain[1]:.2f} against selective {sel[1]:.2f}"
    assert sel[2] > 1.7, (
        f"selective now paints {sel[2]:.2f}x the object area; if the trail has stopped being an "
        "equal-sized second blob, the explanation in label_sources.py needs re-measuring")
    assert sel[0] == pytest.approx(0.5, abs=0.1), (
        f"selective precision is {sel[0]:.2f}, not the ~0.5 an equal-area trail implies")


def test_a_leak_inside_the_mask_does_not_heal_the_trail():
    """The obvious fix, refused by measurement.

    Swept from fully selective to fully plain, precision stays 0.42-0.49 while recall collapses
    from 0.95 to 0.37. The tension is structural: a background must be **slow** not to absorb the
    object and **fast** to forget its trail, and for a moving object those are the same timescale.
    """
    from demo.label_sources import _grey

    def track(leak, rate=0.03, speed=0.002, frames=400, floor=0.25, warm=60):
        bg = None
        prec, rec = [], []
        for k in range(frames):
            p = (0.20 + speed * k, 0.5)
            img, mask = flat_disc(p, seed=k)
            g = _grey(img).astype(np.float64)
            if bg is None:
                bg = g.copy()
                continue
            fg = np.abs(g - bg)
            keep = fg >= floor * max(float(fg.max()), 1e-9)
            bg += np.where(keep, leak, rate) * (g - bg)
            if k >= warm:
                prec.append((keep & mask).sum() / max(keep.sum(), 1))
                rec.append((keep & mask).sum() / max(mask.sum(), 1))
        return float(np.mean(prec)), float(np.mean(rec))

    swept = [track(leak) for leak in (0.0, 0.005, 0.01, 0.02, 0.03)]
    best = max(p for p, _r in swept)
    assert best < 0.6, (
        f"a leak now reaches precision {best:.2f}; if the trail can be healed this way it is "
        "worth shipping and the docstring in label_sources.py is wrong")
    assert swept[0][1] > swept[-1][1] * 2, (
        "the recall collapse across the sweep has gone, so the trade this test describes is not "
        "the trade being measured")
