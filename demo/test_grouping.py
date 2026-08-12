# SPDX-License-Identifier: Apache-2.0
"""Do minted ids group into higher-order labels on their own? Measured: not on this backbone.

The proposal is right in shape — keep doing low-level recognition, let the commonalities between
labels emerge, and let them merge into higher-order groups. It is the way to an automatically
scaling vocabulary without trained weights, and there is a sharp test for it sitting in the data:
`ObjectMemory` **inflates**, so two true objects become three or four ids. If those merge on their
own, the proposal works and the inflation is fixed at the same time.

It does not, and the cause is specific rather than a shortfall of tuning: **a random backbone's
crop descriptor conflates identity with view**, so the instances of one object form several pure
fragments rather than one cluster, and more observations add more spread rather than more evidence.

Everything here is a negative result, which is why it is a test: the next person to have this idea
should find the measurement rather than repeat the work.
"""

from __future__ import annotations

import numpy as np
import pytest

from argmind.backbone import Backbone
from demo.label_sources import BackgroundLocator, ObjectMemory
from demo.pointer import CROP, PLAN, crop_at
from demo.test_object_memory import PALETTE, orbit, scene, unit

linkage = pytest.importorskip("scipy.cluster.hierarchy").linkage
fcluster = pytest.importorskip("scipy.cluster.hierarchy").fcluster


@pytest.fixture(scope="module")
def crop_backbone():
    return Backbone(PLAN, (CROP, CROP, 3), dtype="f4", seed=1, pool=2)


@pytest.fixture(scope="module")
def observed(crop_backbone):
    """Instances with their true provenance, from the real path.

    Provenance is used for **scoring only** — the grouping never sees it, which is the point.
    Detections that are not on any object are dropped rather than labelled: a detection on noise
    has no true label and counting it either way would be inventing one.
    """
    def describe(img, p):
        return unit(crop_backbone.features(crop_at(img, p)).astype(np.float64))

    out = {}
    for n_true in (2, 3):
        colours = PALETTE[:n_true]
        loc = BackgroundLocator(n_true, rate=0.03, warmup=15, suppress=0.10)
        mem = ObjectMemory(ObjectMemory.DEFAULT_TOL, max_ids=4096)
        inst, truth, ids = [], [], []
        for k in range(1200):
            pts = [orbit(k, i, n_true) for i in range(n_true)]
            img = scene(pts, colours, seed=k)
            if k % 4:
                loc(img)
                continue
            found = loc(img)
            if found is None:
                continue
            for p in found:
                near = int(np.argmin([np.hypot(p[0] - q[0], p[1] - q[1]) for q in pts]))
                if np.hypot(p[0] - pts[near][0], p[1] - pts[near][1]) > 0.08:
                    continue
                d = describe(img, p)
                i, _ = mem.observe(d)
                inst.append(d)
                truth.append(near)
                ids.append(i)
        out[n_true] = (np.array(inst), np.array(truth), np.array(ids), len(mem))
        assert len(inst) > 300, f"only {len(inst)} instances at {n_true} objects"
    return out


def purity(labels, truth):
    return sum(np.bincount(truth[labels == c]).max() for c in np.unique(labels)) / len(truth)


def test_grouping_helps_at_two_objects_and_hurts_at_three(observed):
    """The result, and it is not uniform — which is what makes it informative.

    | objects | clustered at the true k | the online ids |
    |---|---|---|
    | 2 | **98.1%** | 90.4% |
    | 3 | **65.4%** | 91.5% |

    At two objects grouping beats the online rule, so the idea is not wrong in principle. At three
    it is *worse* than the six ids the online rule minted, which are individually 91.5% pure. So
    the instances of one object do not form one cluster: they form several pure fragments, and a
    fragment of one object can sit closer to a fragment of another than to its own.
    """
    inst2, truth2, ids2, _ = observed[2]
    inst3, truth3, ids3, _ = observed[3]

    g2 = purity(fcluster(linkage(inst2, "average"), 2, criterion="maxclust"), truth2)
    g3 = purity(fcluster(linkage(inst3, "average"), 3, criterion="maxclust"), truth3)
    online3 = purity(ids3, truth3)

    assert g2 > 0.9, f"grouping at two objects fell to {g2:.1%}"
    assert g3 < online3, (
        f"grouping at three objects ({g3:.1%}) now beats the online ids ({online3:.1%}). That "
        "would mean one object's instances have become one cluster, and the whole reason this "
        "approach is refused would need re-measuring")


def test_the_number_of_groups_cannot_be_found_automatically(observed):
    """The part that "automatically scalable" actually needs, and it is the one that fails hardest.

    A single distance cutoff applied at every object count, groups found (and purity):

    | cutoff | true=2 | true=3 | true=5 |
    |---|---|---|---|
    | 0.10 | 19 (99%) | 60 (97%) | 133 (82%) |
    | 0.20 | 4 (98%) | 11 (95%) | 12 (60%) |
    | 0.25 | 1 (50%) | 5 (66%) | 4 (52%) |
    | 0.40 | 1 (33%) | 1 (33%) | 1 (28%) |

    No cutoff reads 2 and 3. The ones that keep purity give an order of magnitude too many groups,
    and the ones that give few groups have destroyed the distinction. A cutoff that has to be set
    per scene is another knob, not automatic discovery.
    """
    inst2, _t2, _i2, _n2 = observed[2]
    inst3, _t3, _i3, _n3 = observed[3]
    for cut in (0.10, 0.15, 0.20, 0.25, 0.30, 0.40):
        k2 = len(np.unique(fcluster(linkage(inst2, "average"), cut, criterion="distance")))
        k3 = len(np.unique(fcluster(linkage(inst3, "average"), cut, criterion="distance")))
        assert not (k2 == 2 and k3 == 3), (
            f"a cutoff of {cut} now recovers the true object count at both 2 and 3. That is the "
            "automatic discovery this file says does not exist -- re-measure at 5 objects and "
            "rewrite the claim")


def test_more_observations_make_it_worse_not_better(observed):
    """The assumption the proposal rests on — keep doing it and you learn — inverts here.

    Purity against the number of instances: 2 objects 100.0 / 99.0 / 97.5 / 98.8%, and 3 objects
    94.0 / 68.0 / 95.5 / **65.2%** at n = 50 / 100 / 200 / 400. More views mean more of the
    edge-clipped ones, so the within-object spread grows faster than the evidence does.
    """
    inst3, truth3, _i, _n = observed[3]
    early = purity(fcluster(linkage(inst3[:50], "average"), 3, criterion="maxclust"), truth3[:50])
    late = purity(fcluster(linkage(inst3[:400], "average"), 3, criterion="maxclust"), truth3[:400])
    assert late < early, (
        f"three objects now improve with observations ({early:.1%} -> {late:.1%}). If that is "
        "real the descriptor has become view-invariant, which would most likely mean the backbone "
        "weights are no longer random -- and this whole file should be re-run")


def test_a_group_centroid_is_no_better_than_one_instance(observed):
    """The specific mechanism in the proposal — that the *commonality* of a group describes the
    object better than any one sighting — measured and not supported: 1.01x, 1.00x, 0.94x at 2, 3
    and 5 objects. Averaging view-dependent descriptors averages the views, not the object."""
    inst, truth, _i, _n = observed[3]
    lab = fcluster(linkage(inst, "average"), 3, criterion="maxclust")
    groups = np.unique(lab)
    cent = np.array([inst[lab == c].mean(axis=0) for c in groups])
    cent /= np.maximum(np.linalg.norm(cent, axis=1, keepdims=True), 1e-12)
    first = np.array([inst[lab == c][0] for c in groups])
    major = {c: np.bincount(truth[lab == c]).argmax() for c in groups}

    scores = []
    for ref in (first, cent):
        pick = np.argmin(np.linalg.norm(inst[:, None, :] - ref[None, :, :], axis=2), axis=1)
        scores.append(float((np.array([major[groups[p]] for p in pick]) == truth).mean()))
    assert scores[1] < scores[0] * 1.15, (
        f"a centroid now beats a first instance by {scores[1] / scores[0]:.2f}x; if that holds at "
        "five objects too, the commonality mechanism works and ObjectMemory should hold centroids")


def test_co_occurrence_cannot_merge_the_fragments_either(crop_backbone):
    """The appearance-free cue, and why it fails: an object cannot be in two places at once, so two
    ids never seen together are candidates for merging. Measured, same-object pairs co-occur
    **57.2%** of the time (max 98.7%) against different-object pairs at 72.8% (min 0.0%) —
    overlapping, and in the wrong direction at the extremes.

    The premise is violated by the **detector**, not by the world: it cannot count, so it puts two
    detections on one object and the two ids appear in the same frame legitimately. That is the
    fifth independent route blocked at the same place.
    """
    def describe(img, p):
        return unit(crop_backbone.features(crop_at(img, p)).astype(np.float64))

    n_true = 3
    colours = PALETTE[:n_true]
    loc = BackgroundLocator(n_true, rate=0.03, warmup=15, suppress=0.10)
    mem = ObjectMemory(ObjectMemory.DEFAULT_TOL, max_ids=4096)
    frames, owner = [], {}
    for k in range(1600):
        pts = [orbit(k, i, n_true) for i in range(n_true)]
        img = scene(pts, colours, seed=k)
        if k % 4:
            loc(img)
            continue
        found = loc(img)
        if found is None:
            continue
        here = []
        for p in found:
            near = int(np.argmin([np.hypot(p[0] - q[0], p[1] - q[1]) for q in pts]))
            if np.hypot(p[0] - pts[near][0], p[1] - pts[near][1]) > 0.08:
                continue
            i, minted = mem.observe(describe(img, p))
            if minted:
                owner[i] = near
            here.append(i)
        if here:
            frames.append(set(here))

    n = len(mem)
    co = np.zeros((n, n))
    seen = np.zeros(n)
    for f in frames:
        for i in f:
            seen[i] += 1
            for j in f:
                if i != j:
                    co[i, j] += 1
    same, diff = [], []
    for i in range(n):
        for j in range(i + 1, n):
            if seen[i] < 5 or seen[j] < 5:
                continue
            rate = co[i, j] / min(seen[i], seen[j])
            (same if owner.get(i) == owner.get(j) else diff).append(rate)
    assert same and diff, "the fixture produced no comparable pairs"
    assert max(same) > min(diff), (
        f"co-occurrence now separates cleanly (same-object max {max(same):.1%} below "
        f"different-object min {min(diff):.1%}). That would make automatic merging possible "
        "without appearance, and it is worth building -- most likely it means the detector "
        "stopped putting two detections on one object")
