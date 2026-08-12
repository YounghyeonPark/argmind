# SPDX-License-Identifier: Apache-2.0
"""A vocabulary that grows automatically — what it does, and the two designs it replaces.

The goal is a recognisable-label count that scales on its own. The constraint is that
`RecursiveLeastSquares.widened()` **refuses** to change `n_outputs`, so a growing label set cannot
be a growing head. The resolution is **unbounded vocabulary, bounded concurrency**: a caller-side
dictionary that grows freely, and a head with a declared number of slots.

Two ways of putting the identity on the head's *input* instead were measured and both fail, and one
of those refutations was a prediction of mine going the wrong way. Both are asserted here, because
they are the reason the shipped design is what it is.
"""

from __future__ import annotations

import numpy as np
import pytest

from argmind.adapt import NormalisedLeastMeanSquares, RecursiveLeastSquares
from argmind.backbone import Backbone, conv2d, depthwise_conv2d, relu6
from demo.label_sources import BackgroundLocator, ObjectMemory
from demo.pointer import CROP, IN_SHAPE, PLAN, crop_at

H, W = IN_SHAPE[0], IN_SHAPE[1]
PALETTE = ((0.95, 0.25, 0.20), (0.20, 0.35, 0.95), (0.25, 0.90, 0.30),
           (0.90, 0.85, 0.20), (0.80, 0.25, 0.85))


def scene(pts, colours, seed=0, bright=0.45, sigma=0.075):
    yy, xx = np.mgrid[0:H, 0:W]
    img = np.full((H, W, 3), bright)
    for (px, py), c in zip(pts, colours):
        d = ((xx - px * W) ** 2 + (yy - py * H) ** 2) / (2 * (sigma * W) ** 2)
        img += np.exp(-d)[:, :, None] * np.asarray(c, dtype=float)[None, None, :]
    return (img + np.random.default_rng(seed).standard_normal(img.shape) * 0.02).astype("f4")


def orbit(k, i, n, period=90.0):
    a = 2 * np.pi * (k / period) + 2 * np.pi * i / n
    return (0.5 + 0.30 * np.cos(a), 0.5 + 0.28 * np.sin(a * (1 + 0.13 * i)))


def feature_map(bb, image):
    """The stack's output before pooling, which `features()` throws away.

    Local to this file: it exists only to run the refuted correlation experiment below, and adding
    it to `Backbone` would be adding an entry point for something measured not to work.
    """
    x = np.asarray(image, dtype=bb.dtype)
    for entry, w, b in zip(bb.plan, bb._w, bb._b):        # noqa: SLF001
        if entry[0] == "conv":
            x = relu6(conv2d(x, w, b, stride=entry[3]))
        elif entry[0] == "pw":
            x = relu6(conv2d(x, w, b, stride=1))
        else:
            x = relu6(depthwise_conv2d(x, w, b, stride=entry[2]))
    return x


@pytest.fixture(scope="module")
def crop_backbone():
    return Backbone(PLAN, (CROP, CROP, 3), dtype="f4", seed=1, pool=2)


def unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def describer(crop_backbone):
    def describe(img, p):
        return unit(crop_backbone.features(crop_at(img, p)).astype(np.float64))
    return describe


def test_the_head_cannot_grow_its_outputs_which_is_what_forces_this_design():
    """The constraint, asserted rather than cited. If this ever becomes possible the whole
    vocabulary-vs-slots split should be revisited, so it fails here first."""
    rls = RecursiveLeastSquares(4, 2, dtype="f8")
    wider = rls.widened(6, keep_confidence=False)          # more features: allowed
    assert wider.n_features == 6 and wider.n_outputs == 2
    with pytest.raises(ValueError, match="cannot change the number of outputs"):
        rls.widened(6, keep_confidence=False, n_outputs=4)


def registry_run(crop_backbone, n_true, tol, perfect, n=1200, every=4):
    """Mint ids automatically and score whether a match lands on the object its id came from."""
    describe = describer(crop_backbone)
    colours = PALETTE[:n_true]
    loc = BackgroundLocator(n_true, rate=0.03, warmup=15, suppress=0.10)
    mem = ObjectMemory(tol, max_ids=4096)
    owner, right, matched = [], 0, 0
    for k in range(n):
        pts = [orbit(k, i, n_true) for i in range(n_true)]
        img = scene(pts, colours, seed=k)
        if k % every:
            if not perfect:
                loc(img)                       # the background must keep updating
            continue
        found = list(pts) if perfect else loc(img)
        if found is None:
            continue
        for p in found:
            near = int(np.argmin([np.hypot(p[0] - q[0], p[1] - q[1]) for q in pts]))
            i, minted = mem.observe(describe(img, p))
            if minted:
                owner.append(near)
            else:
                matched += 1
                right += owner[i] == near
    assert matched > 100, f"only {matched} matches; the fixture is not exercising this"
    return len(mem), right / matched


def test_the_vocabulary_grows_automatically_and_mostly_correctly(crop_backbone):
    """The claim. Minting an id when nothing stored is within `tol`, over 1200 frames:

    | true objects | ids | match correct |
    |---|---|---|
    | 2 | 4 | 96.7% |
    | 3 | 7 | 66.5% |

    It **inflates** — about twice as many ids as objects — because the same object seen where its
    crop clips against a frame edge describes differently. Automatic growth is real, not free, and
    not exact.

    The three-object figure was 85.3% in an exploratory script whose background only updated on
    cadence frames. Updating it every frame is the correct path and gives 66.5%; the higher number
    came from a slower-adapting background, not from better matching.
    """
    ids2, acc2 = registry_run(crop_backbone, 2, ObjectMemory.DEFAULT_TOL, perfect=False)
    assert acc2 > 0.8, f"two objects matched correctly only {acc2:.1%} of the time"
    assert 2 <= ids2 <= 8, f"{ids2} ids for two objects"

    ids3, acc3 = registry_run(crop_backbone, 3, ObjectMemory.DEFAULT_TOL, perfect=False)
    assert acc3 > 0.6, f"three objects matched correctly only {acc3:.1%} of the time"
    assert acc3 < acc2, "three objects are no longer harder than two"
    assert ids3 > 3, (
        "the vocabulary no longer inflates. That would be an improvement and the docstring's "
        "'about twice as many ids as objects' would need re-measuring")


def test_it_degrades_with_object_count_and_that_is_stated_not_hidden(crop_backbone):
    """Five objects: 43.1% from detected crops, 57.0% from perfect ones. So the honest scope is
    "a few objects", and the degradation is the measurement rather than an adjective."""
    _ids2, acc2 = registry_run(crop_backbone, 2, ObjectMemory.DEFAULT_TOL, perfect=False)
    _ids5, acc5 = registry_run(crop_backbone, 5, ObjectMemory.DEFAULT_TOL, perfect=False)
    assert acc5 < acc2 * 0.75, (
        f"five objects now match at {acc5:.1%} against two at {acc2:.1%}; if the degradation has "
        "gone, the scope in demo/label_sources.py is understated")
    assert acc5 > 0.25, f"five objects collapsed to {acc5:.1%}, worse than recorded"


def test_the_threshold_has_a_measured_window_and_collapses_outside_it(crop_backbone):
    """Either side of the default the behaviour is a collapse, not a degradation: 101 ids for two
    objects at 0.04 -- one per detection -- and a single id at 0.60.

    The default itself is **tuned on the outcome**, and this test pins why it has to be: the
    within/between gap that would justify a derivation moves with the object count.
    """
    tight, _ = registry_run(crop_backbone, 2, 0.04, perfect=False)
    loose, _ = registry_run(crop_backbone, 2, 0.60, perfect=False)
    assert tight > 40, (
        f"a threshold below the within-object distance minted only {tight} ids; it is supposed to "
        "mint one per detection, which is the failure mode of automatic growth")
    assert loose == 1, f"a threshold above the between-object distance gave {loose} ids, not 1"
    # And the reason the default cannot be *derived*: the gap it would come from moves with the
    # object count. Measured on the crops the registry sees, (0.081, 0.225) at two objects and
    # (0.131, 0.288) at three -- 0.25 is above the first and inside the second, so there is no
    # single interval. The first version of this test asserted it sat inside "the" gap and failed,
    # which is how the claim was caught.
    assert not (0.081 < ObjectMemory.DEFAULT_TOL < 0.225), (
        "the default now sits inside the two-object gap; if the gaps have converged, a derived "
        "threshold may be possible and the docstring's admission of tuning should be revisited")


def test_the_ceiling_holds_and_says_so():
    """`max_ids` exists because unbounded growth is this design's failure mode. Past the ceiling
    the nearest id is returned however far away, and `full` reports it — a caller that silently
    stops learning new objects is worse than one that can be asked."""
    m = ObjectMemory(0.1, max_ids=2)
    a, b, c = np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([-1.0, -1.0])
    assert m.observe(a) == (0, True)
    assert m.observe(a + 0.01) == (0, False)
    assert m.observe(b) == (1, True)
    assert m.full and len(m) == 2
    far, minted = m.observe(c)
    assert not minted and far in (0, 1), "the ceiling minted an id anyway"
    assert len(m) == 2

    with pytest.raises(ValueError, match="tol must be positive"):
        ObjectMemory(0.0)
    with pytest.raises(ValueError, match="max_ids"):
        ObjectMemory(0.1, max_ids=0)
    with pytest.raises(ValueError, match="1-D"):
        ObjectMemory().observe(np.zeros((2, 2)))


def test_descriptors_are_held_and_handed_out_as_copies():
    """Averaging a matched descriptor into its id lets it drift onto a neighbour, which is the
    same shape as the guard's baseline tracking the fault it was holding out against."""
    m = ObjectMemory(0.5)
    a = np.array([1.0, 0.0])
    expected = a.copy()                      # an INDEPENDENT reference to compare against
    m.observe(a)
    m.observe(np.array([0.9, 0.4]))          # a match, well inside tol
    assert np.array_equal(m.descriptor(0), expected), "the stored descriptor moved"

    # `observe` must copy: np.asarray on a float64 array returns the same object, so the store
    # used to alias the caller's array. Comparing against `a` rather than `expected` hid both
    # this and the aliasing below, because mutating one changed the other.
    a[0] = 42.0
    assert np.array_equal(m.descriptor(0), expected), "the store aliased the caller's array"

    got = m.descriptor(0)
    got[0] = 99.0
    assert np.array_equal(m.descriptor(0), expected), "descriptor() handed out the live array"


def one_head_form(form, n_obj, crop_backbone, n=1200, every=4):
    """Score a fixed-output head that takes the object's identity on its INPUT.

    Labels are supplied perfectly, so this measures the architecture and nothing else.
    """
    bb = Backbone(PLAN, IN_SHAPE, dtype="f4", seed=1, pool=3)
    colours = PALETTE[:n_obj]
    cal = np.array([bb.features(scene([(0.5, 0.5)], [colours[0]], seed=i)) for i in range(8)])
    bb.output_scale = 1.0 / max(float(cal.std()), 1e-9)
    describe = describer(crop_backbone)

    img0 = scene([orbit(0, j, n_obj) for j in range(n_obj)], colours, seed=999)
    pooled = [describe(img0, orbit(0, i, n_obj)) for i in range(n_obj)]
    chan = [unit(feature_map(crop_backbone, crop_at(img0, orbit(0, i, n_obj)))
                 .reshape(-1, 16).mean(axis=0)) for i in range(n_obj)]

    def heat(fmap, q):
        f = fmap.reshape(-1, fmap.shape[2]).astype(np.float64)
        f = f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-12)
        return (f @ q).reshape(fmap.shape[0], fmap.shape[1])

    probe = feature_map(bb, img0)
    width = (bb.n_outputs + len(pooled[0]) + 1 if form == "concat"
             else probe.shape[0] * probe.shape[1] + 1)
    head = NormalisedLeastMeanSquares(width, 2, mu=0.3, dtype="f4")

    errs, ctrl = [], []
    for k in range(n):
        pts = [orbit(k, i, n_obj) for i in range(n_obj)]
        img = scene(pts, colours, seed=k)
        emb = bb.features(img)
        fmap = feature_map(bb, img) if form == "correlate" else None
        for i in range(n_obj):
            x = (np.concatenate([emb, pooled[i], [1.0]]) if form == "concat"
                 else np.concatenate([heat(fmap, chan[i]).ravel(), [1.0]]))
            p = head.predict(x).copy()
            if k % every == 0:
                head.update(x, [pts[i][0], pts[i][1]])
            if k > n // 2:
                errs.append(np.hypot(p[0] - pts[i][0], p[1] - pts[i][1]))
                ctrl.append(np.hypot(0.5 - pts[i][0], 0.5 - pts[i][1]))
    return float(np.mean(errs)), float(np.mean(ctrl))


def test_putting_the_identity_on_the_input_does_not_work(crop_backbone):
    """The two refutations that make the shipped design necessary rather than merely convenient.

    Against a per-slot head at 1.71-1.83x a centre guess:

    | form | 2 objects | 5 |
    |---|---|---|
    | `[embedding, query]` concatenated | 1.10x | 1.00x |
    | query correlated against the feature map | 1.08x | 1.01x |

    The concatenation was **predicted** to fail: locating a specific pattern is a multiplicative
    interaction between query and image, and a linear map on a concatenation has no product term.

    The correlation was predicted to **work** and does not, which is the more useful half. Its
    cause is measured: the correlation map's argmax lands on the queried object only 4-50% of the
    time under any similarity — see `test_a_correlation_map_does_not_localise_the_queried_object`.
    """
    for form in ("concat", "correlate"):
        err, centre = one_head_form(form, 2, crop_backbone)
        assert centre < err * 1.4, (
            f"{form} now reaches {centre / err:.2f}x a centre guess at two objects. If that is "
            "real, the identity CAN go on the input and ObjectMemory's docstring is wrong")


def test_a_correlation_map_does_not_localise_the_queried_object(crop_backbone):
    """Why the correlation form fails, isolated from any learning.

    Fraction of frames where the map's argmax lands on the queried object, three queries:

    | similarity | q0 | q1 | q2 |
    |---|---|---|---|
    | cosine | 17.5% | 10.8% | 4.2% |
    | plain dot product | 2.5% | 5.8% | 50.0% |
    | negative distance | 40.0% | 15.8% | 18.3% |

    A random backbone's per-cell channel signature does not identify an object; that is what a
    *trained* embedding is for. Cosine was tried first and is also the trap this project already
    records for embeddings — normalising discards the magnitude.
    """
    bb = Backbone(PLAN, IN_SHAPE, dtype="f4", seed=1, pool=3)
    n_obj = 3
    colours = PALETTE[:n_obj]
    img0 = scene([orbit(0, j, n_obj) for j in range(n_obj)], colours, seed=999)
    queries = [feature_map(crop_backbone, crop_at(img0, orbit(0, i, n_obj)))
               .reshape(-1, 16).mean(axis=0) for i in range(n_obj)]

    def maps(fm, q, kind):
        f = fm.reshape(-1, fm.shape[2]).astype(np.float64)
        if kind == "cosine":
            f = f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-12)
            m = f @ unit(q)
        elif kind == "dot":
            m = f @ q
        else:
            m = -np.linalg.norm(f - q[None, :], axis=1)
        return m.reshape(fm.shape[:2])

    best = 0.0
    for kind in ("cosine", "dot", "distance"):
        for i in range(n_obj):
            hits = total = 0
            for k in range(0, 360, 6):
                pts = [orbit(k, j, n_obj) for j in range(n_obj)]
                m = maps(feature_map(bb, scene(pts, colours, seed=k)), queries[i], kind)
                y, x = divmod(int(np.argmax(m)), m.shape[1])
                got = (x / (m.shape[1] - 1), y / (m.shape[0] - 1))
                d = [np.hypot(got[0] - p[0], got[1] - p[1]) for p in pts]
                total += 1
                hits += int(np.argmin(d)) == i and min(d) < 0.15
            best = max(best, hits / total)
    assert best < 0.75, (
        f"a correlation map now localises the queried object {best:.1%} of the time. If that is "
        "real then per-object segmentation by correlation works and is worth building -- which "
        "would most likely mean the backbone weights are no longer random")

def test_the_default_threshold_loses_objects_and_a_tight_one_over_segments():
    """The two failure modes are not symmetric, and reading the trace is what showed it.

    | objects | ids | objects represented | purity |
    |---|---|---|---|
    | 3 at the default 0.25 | 8 | **2 of 3** | 62.6% |
    | 3 at 0.10 | 72 | **3 of 3** | **95.1%** |

    At the default an object can end up with **no id of its own** — its detections are absorbed
    into another id and can never be right, which is most of the purity gap. At a tight threshold
    every object is represented and the fragments are pure; the cost is 18x inflation.

    **Over-segmentation is the better failure**: pure fragments can in principle be merged, whereas
    an object with no id is information already lost. That is where a merging step should be aimed,
    and `test_grouping.py` measured merging at the default threshold instead — where the objects
    were already gone.
    """
    from demo.label_trace import trace

    loose = trace(objects=3, frames=1200, verbose=False, summary=False)
    tight = trace(objects=3, frames=1200, tol=0.10, verbose=False, summary=False)

    assert loose["objects_represented"] < 3, (
        f"the default threshold now represents all {loose['objects_represented']} objects; if "
        "that is real the purity gap has another cause and this file needs re-measuring")
    assert tight["objects_represented"] == 3, (
        f"a tight threshold now loses an object too ({tight['objects_represented']} of 3), so "
        "over-segmentation is no longer the recoverable failure this claims")
    assert tight["purity"] > loose["purity"] * 1.3, (
        f"tight {tight['purity']:.1%} against loose {loose['purity']:.1%}")
    assert tight["ids"] > loose["ids"] * 5, "the tight threshold no longer over-segments"


def test_a_good_fraction_of_the_vocabulary_is_minted_from_noise():
    """Ids that came from a detection on nothing at all: 1 of 3 at two objects, 3 of 8 at three,
    7 of 11 at five. The detector's inability to count shows up here as vocabulary pollution, and
    a caller cannot tell those ids from real ones without the ground truth this fixture has."""
    from demo.label_trace import trace

    got = trace(objects=5, frames=1200, verbose=False, summary=False)
    junk = got["ids"] - got["ids_from_objects"]
    assert junk > 0, "no id came from noise; the detector has stopped firing on nothing"
    assert junk >= got["ids_from_objects"] * 0.5, (
        f"only {junk} of {got['ids']} ids came from noise; if the pollution has dropped that far, "
        "the detector improved and the scope here is understated")
