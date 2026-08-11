"""The demo's composition, driven headlessly.

The camera and the window are not tested — those are OpenCV's. What is tested is everything
the README claims, driven with synthetic frames: that it learns from clicks, that it survives
a covered camera by freezing rather than being destroyed, that it thaws, and that a restart
gets the same predictions back.

**A demo nobody runs rots.** This runs in the root suite and needs no camera and no OpenCV,
which is why `cv2` is imported inside `main()` rather than at module scope.
"""
import numpy as np
import pytest

from argmind.persist import Store
from demo.pointer import IN_SHAPE, STATE_DIR, STATE_PATH, build, fingerprint


def frame_at(x, y, bright=0.45, seed=0):
    """A synthetic 'hand': a bright blob at (x, y) in fractions of the frame."""
    h, w = IN_SHAPE[0], IN_SHAPE[1]
    yy, xx = np.mgrid[0:h, 0:w]
    d = ((xx - x * w) ** 2 + (yy - y * h) ** 2) / (2 * (0.12 * w) ** 2)
    img = np.full((h, w, 3), bright) + np.exp(-d)[:, :, None] * 0.9
    return (img + np.random.default_rng(seed).standard_normal(img.shape) * 0.02).astype("f4")


def drive(parts, path, steps=260, covered=(), t0=0.0, teach_every=1):
    """Push a frame per step and teach on some of them. Returns the reports and the clock."""
    backbone, head, guard, features, taught, _fmap, pipe = parts
    g = np.random.default_rng(11)
    out, t = [], t0
    for k in range(steps):
        t += 1.0 / 30.0
        x, y = float(g.uniform(0.2, 0.8)), float(g.uniform(0.2, 0.8))
        img = np.zeros(IN_SHAPE, dtype="f4") if covered and covered[0] <= k < covered[1] \
            else frame_at(x, y, seed=k)
        emb = backbone.features(img)
        # The caller's own statistic, exactly as the demo computes it: the raw frame's
        # standard deviation relative to calibration. A covered camera reads 0.
        guard.set_external(0, float(np.std(img)) / max(path["frame_std"], 1e-6))
        features.push(t, emb)
        if k % teach_every == 0:
            taught.push(t, [x, y])
        r = pipe.step()
        # The prediction must be read NOW. `StepReport.prediction` is a preallocated buffer
        # the next step overwrites, so a list of reports read afterwards holds one value
        # repeated -- **the third time this trap has been walked into in this repository**,
        # and the first version of this helper did it too: it reported that teaching barely
        # worked (0.256 to 0.227) on a composition that reaches 0.058.
        out.append((bool(r.frozen), tuple(r.frozen_by),
                    None if r.prediction is None else (float(r.prediction[0]),
                                                       float(r.prediction[1])),
                    (x, y)))
    return out, t


@pytest.fixture
def ready():
    """A built composition with the backbone calibrated, which is what the demo does after
    its first fifteen frames -- an uncalibrated embedding is ~1e-3 and no head learns on it."""
    parts = build()
    backbone = parts[0]
    sample = np.array([backbone.features(frame_at(0.5, 0.5, seed=i)) for i in range(8)])
    backbone.output_scale = 1.0 / max(float(sample.std()), 1e-9)
    return parts, {"frame_std": float(np.std(frame_at(0.5, 0.5, seed=0)))}


def _error(rows):
    got = [(p[0], p[1], tx, ty) for _fz, _by, p, (tx, ty) in rows if p is not None]
    assert got, "the pipeline never produced a prediction"
    a = np.array(got)
    return float(np.sqrt(np.mean((a[:, 0] - a[:, 2]) ** 2 + (a[:, 1] - a[:, 3]) ** 2)))


def test_it_learns_where_you_are_from_clicks(ready):
    parts, path = ready
    reports, _t = drive(parts, path, steps=400)
    early, late = _error(reports[:60]), _error(reports[-60:])
    assert late < early * 0.8, (
        f"teaching did not improve it: {early:.3f} then {late:.3f}")
    # A control: a constant predictor at the centre of the sampled range would score about
    # 0.24, so beating it is the claim rather than merely improving.
    assert late < 0.24, f"it never beat a centre-guess: {late:.3f}"


def test_covering_the_camera_freezes_it_instead_of_destroying_it(ready):
    parts, path = ready
    guard = parts[2]
    before, t = drive(parts, path, steps=400)
    good = _error(before[-60:])

    covered, t = drive(parts, path, steps=90, covered=(0, 90), t0=t)
    assert any(fz for fz, _by, _p, _t in covered), (
        "a covered camera never froze the guard; the head fits whatever it is given, so "
        "nothing else in the composition would have noticed")
    # `frozen_by` is reported on the TRIP and empty on the blocked steps after it, which is
    # the library working as designed -- the guard still holds it.
    trip = next(by for fz, by, _p, _t in covered if fz and by)
    assert "external" in " ".join(trip), (
        f"frozen by {trip}, not by the caller's scalar -- if the head's own residual caught "
        "this, the external check is not what the test claims it is")
    assert guard.frozen_by == trip

    guard.thaw()
    after, _t = drive(parts, path, steps=400, t0=t)
    recovered = _error(after[-60:])
    assert recovered < good * 2.0, (
        f"the head did not survive the fault: {good:.3f} before, {recovered:.3f} after")


def test_an_unguarded_head_is_destroyed_by_the_same_fault(ready):
    """The control without which the test above shows nothing: the fault has to be capable of
    destroying an unguarded head, or freezing prevented nothing."""
    parts, path = ready
    backbone, head, _guard, features, taught, _fmap, _pipe = parts
    g = np.random.default_rng(11)
    t = 0.0
    for phase, covered in (("teach", False), ("covered", True)):
        for k in range(240 if phase == "teach" else 90):
            t += 1.0 / 30.0
            x, y = float(g.uniform(0.2, 0.8)), float(g.uniform(0.2, 0.8))
            img = np.zeros(IN_SHAPE, dtype="f4") if covered else frame_at(x, y, seed=k)
            emb = np.concatenate([backbone.features(img), [1.0]])
            head.update(emb, [x, y] if not covered else [9.0, -9.0])
        if phase == "teach":
            probe = [(frame_at(0.3, 0.7, seed=1), 0.3, 0.7),
                     (frame_at(0.7, 0.3, seed=2), 0.7, 0.3)]
            clean = np.mean([abs(float(head.predict(
                np.concatenate([backbone.features(im), [1.0]]))[0]) - tx)
                for im, tx, _ty in probe])
    dirty = np.mean([abs(float(head.predict(
        np.concatenate([backbone.features(im), [1.0]]))[0]) - tx)
        for im, tx, _ty in probe])
    assert dirty > clean * 3.0, (
        f"the fault did not damage an unguarded head ({clean:.3f} then {dirty:.3f}), so the "
        "guard in the test above prevented nothing")


def test_it_remembers_across_a_restart(ready, tmp_path):
    parts, path = ready
    backbone, _head, guard, _f, _t, _m, _p = parts
    drive(parts, path, steps=400)

    store = Store(tmp_path)
    fp = fingerprint(backbone)
    store.save(guard, fingerprint=fp)

    fresh = build()
    fresh[0].output_scale = backbone.output_scale
    rec = Store(tmp_path).load(fresh[2], fingerprint=fp)
    assert rec is not None

    img = frame_at(0.35, 0.65, seed=5)
    x = np.concatenate([backbone.features(img), [1.0]])
    assert np.allclose(guard.predict(x), fresh[2].predict(x)), (
        "the restored head does not predict what the saved one did")

    # ...and a head saved under a different backbone must be refused, which the CRC cannot
    # see: those bytes are perfectly valid, they just mean something else.
    other = build()
    with pytest.raises(Exception):
        Store(tmp_path).load(other[2], fingerprint=fp ^ 0xFFFF)


def test_the_state_path_is_a_base_path_and_the_slots_land_under_it(tmp_path):
    """"Saved" printed while nothing was written is the defect this pins.

    A `Store` takes a base PATH -- its slots are `<path>.0` and `<path>.1` -- and its
    docstring says so. The first version of the demo passed the *directory*, so the slots
    landed as `demo/state.0` beside it while the demo announced `demo/state`. It saved
    correctly and told the user the wrong place, which is worse than failing.
    """
    parts = build()
    guard = parts[2]
    base = tmp_path / "state" / "head"
    base.parent.mkdir(parents=True)
    store = Store(base)
    store.save(guard, fingerprint=fingerprint(parts[0]))

    written = sorted(p.name for p in base.parent.iterdir())
    assert written, "save() wrote nothing under the directory it was pointed at"
    assert all(n.startswith("head.") for n in written), written
    assert not list(tmp_path.glob("state.*")), (
        "a slot landed beside the directory instead of inside it")

    # ...and the demo's own constants have to agree with that shape, or the module ships
    # the bug this test describes.
    assert STATE_PATH.parent == STATE_DIR
    assert STATE_PATH.name and not STATE_PATH.name.endswith("/")
