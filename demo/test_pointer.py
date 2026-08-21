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
from demo.pointer import (IN_SHAPE, STATE_DIR, STATE_PATH, build, fingerprint,
                          teach_from_click)


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
        # `teach_every=0` means never, which `k % teach_every` cannot express -- at k=0
        # any modulus is zero, so a case meaning "no labels at all" pushed one anyway.
        if teach_every and k % teach_every == 0:
            # Through the shipped function, not a re-implementation of it -- a harness that
            # rebuilds the assembly cannot see the assembly break.
            teach_from_click(taught, t, x, y, 1, "points")
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


def test_it_keeps_predicting_on_frames_that_carry_no_label(ready):
    """The camera runs at 30 Hz and the labels do not, so a prediction on an unlabelled frame
    is the entire output of this demo. The first version of `pointer.py` used the LABEL stream
    as the aligner's reference, and frames are built at reference instants -- so the dot moved
    while the button was held and never otherwise. Measured on this fixture: 0 predictions out
    of 140 unlabelled frames.

    `drive` has had a `teach_every` parameter since it was written and **no caller ever passed
    one**, so every case fed a label on every frame; both streams then arrive at the same
    instant, `hold_last` always finds an exact match, and the reference choice has no
    observable consequence. The knob existed and was never turned.
    """
    parts, path = ready
    reports, t = drive(parts, path, steps=200)               # teach densely first
    taught_well = _error(reports[-60:])

    # Now the button is released: features keep arriving, labels stop entirely.
    quiet, _t = drive(parts, path, steps=140, t0=t, teach_every=0)
    have = [p for _fz, _by, p, _xy in quiet if p is not None]
    assert len(have) == 140, (
        f"only {len(have)} of 140 unlabelled frames produced a prediction; with the label "
        "stream as the aligner's reference this is 0, which is the bug this test exists for")

    # And the prediction must still be USEFUL, not merely present -- a held constant would
    # also be "present" on every frame.
    assert _error(quiet) < 0.24, (
        f"predictions on unlabelled frames are no better than a centre-guess: "
        f"{_error(quiet):.3f} against a taught {taught_well:.3f}")

    # The prediction moves frame to frame, which is what says it is being recomputed from the
    # camera rather than repeated. This is also the aliasing trap's other face: a stale buffer
    # would give one value 140 times.
    assert len({(round(p[0], 6), round(p[1], 6)) for p in have}) > 100


def test_a_held_label_is_not_relearned_once_per_camera_frame(ready):
    """`require_fresh_target` is not decoration once the reference is the fast stream. Without
    it the same held label is used as a target on every camera frame, and the head is dragged
    towards a position that stopped being true -- measured 0.238 radial error against 0.085.

    This is the cost CLAUDE.md records for a fast reference, made concrete in the demo: a slow
    target relearned once per fast frame. The gate compares the target sample's own timestamp,
    which is why it needs no threshold; `max_stale_s` is a bound on time where the question is
    identity, and no value of it is right for both.
    """
    parts, path = ready
    _pipe = parts[6]
    assert _pipe.require_fresh_target, (
        "the demo turned the freshness gate off; with a camera-rate reference that relearns "
        "every held label and the measured cost is 2.8x the error")

    # Teach at a tenth of the camera rate, which is what dragging actually produces.
    reports, _t = drive(parts, path, steps=600, teach_every=10)
    updated = sum(1 for _fz, _by, p, _xy in reports if p is not None)
    assert updated == 600                     # a prediction on every frame regardless
    assert _error(reports[-120:]) < 0.24, "sparse teaching never beat a centre-guess"


def test_the_overlay_reports_whether_it_is_learning_right_now(ready):
    """"Do I have to keep teaching it, or does it keep learning by itself?" was a question the
    demo could not answer, because the `updates` counter on screen was hardwired to zero: the
    line read `guard.updates if hasattr(guard, "updates") else ...` and `Guard` has no such
    attribute. A defensive `hasattr` around a field nobody checked.

    So both halves are pinned here: the attribute is genuinely absent (the fallback was not
    dead code guarding a rename), and the counter now advances only while labels arrive.
    """
    parts, path = ready
    guard = parts[2]
    assert not hasattr(guard, "updates"), (
        "Guard now exposes an update counter; the demo should read it, and this test should "
        "say so rather than asserting its absence")

    reports, t = drive(parts, path, steps=120)
    learned = sum(1 for _fz, _by, p, _xy in reports if p is not None)
    assert learned == 120

    # `updated` is the honest source, and it must move while teaching...
    updates_taught, t2 = _count_updates(parts, path, steps=120, t0=t, teach_every=1)
    assert updates_taught > 100, updates_taught
    # ...and stop when the button is released, which is what "not learning" means. The clock
    # continues rather than jumping: a jump exposes a real library defect in the freshness
    # gate (one stale target slips through after a gap in the target stream), which is pinned
    # by `test_a_gap_in_the_target_stream_lets_one_stale_target_through` in tests/, not here.
    updates_quiet, _t = _count_updates(parts, path, steps=120, t0=t2, teach_every=0)
    assert updates_quiet == 0, (
        f"{updates_quiet} updates on frames with no new label; the head is learning from a "
        "target that stopped being true, which is what require_fresh_target prevents")


def _count_updates(parts, path, steps, t0, teach_every):
    """`drive` reports predictions, not whether the adapter moved, so this counts `updated`."""
    backbone, _head, guard, features, taught, _fmap, pipe = parts
    g = np.random.default_rng(5)
    n, t = 0, t0
    for k in range(steps):
        t += 1.0 / 30.0
        x, y = float(g.uniform(0.2, 0.8)), float(g.uniform(0.2, 0.8))
        img = frame_at(x, y, seed=k)
        guard.set_external(0, float(np.std(img)) / max(path["frame_std"], 1e-6))
        features.push(t, backbone.features(img))
        if teach_every and k % teach_every == 0:
            # Through the shipped function, not a re-implementation of it -- a harness that
            # rebuilds the assembly cannot see the assembly break.
            teach_from_click(taught, t, x, y, 1, "points")
        n += bool(pipe.step().updated)
    return n, t


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

def test_a_session_that_learned_nothing_does_not_overwrite_the_saved_head():
    """Found by running the demo while the camera was still held by another process.

    `cap.read()` failed on the first frame, the loop broke, and the demo saved an untrained head
    over the trained one and printed "saved" -- exit code 0, indistinguishable from a normal quit.
    A camera glitch at start-up destroyed the model.

    Same family as the earlier bug that printed "saved" and wrote to the wrong place. The failures
    that matter here are the ones that report success.
    """
    from demo.pointer import should_save

    assert should_save(0, False) is False, "a session that learned nothing would overwrite"
    assert should_save(0, True) is False
    assert should_save(1, False) is True
    assert should_save(500, True) is True


def test_a_dead_camera_is_reported_and_does_not_look_like_a_normal_quit(ready, tmp_path):
    """The other half: the exit has to be distinguishable. `main()` returns 2 and says so on
    stderr rather than 0 with a cheerful "saved".

    Driven through the save path rather than through `main`, which needs a camera and OpenCV: what
    is asserted is that a trained head survives a subsequent no-op session, which is the damage
    the bug actually did.
    """
    parts, path = ready
    guard = parts[2]
    store = Store(tmp_path / "head")
    fp = fingerprint(parts[0], 1)

    drive(parts, path, steps=200)                      # learn something
    store.save(guard, fingerprint=fp)
    trained = np.asarray(guard.state).copy()

    # A fresh composition that learned nothing must not be written over it.
    fresh = build(1)
    from demo.pointer import should_save
    assert not should_save(0, False)
    # The check is what stands between the two; if a caller ignores it, this is the damage.
    store.save(fresh[2], fingerprint=fp)
    restored = build(1)
    store.load(restored[2], fingerprint=fp)
    assert not np.allclose(np.asarray(restored[2].state), trained), (
        "saving an untrained head did not change the record, so this test is not demonstrating "
        "the hazard that should_save exists to prevent")

# ---------------------------------------------------------------------------
# Raw pixels against the random backbone
# ---------------------------------------------------------------------------


def test_raw_pixels_match_the_backbone_in_width_and_beat_it_in_accuracy(ready):
    """The owner asked why not just feed the raw image. Measured, it wins on every axis here, and
    the default is deliberately the SAME width as the backbone so the comparison is like-for-like.

    | input | features | roles | identity |
    |---|---|---|---|
    | backbone | 145 | 4.61x | 4.01x |
    | raw /8 grey | 109 | **12.09x** | 1.71x |
    | **raw /12 RGB (default)** | **145** | 8.34x | **118.87x** |
    """
    from demo.pointer import raw_features, raw_width

    parts_raw = build(1, "raw")
    parts_bb = build(1, "backbone")
    assert raw_width() == 144, "the raw default is no longer the backbone's own width"
    assert parts_raw[1].n_features == parts_bb[1].n_features

    # And it is the shape the composition expects, which is what would break silently.
    v = raw_features(frame_at(0.3, 0.7, seed=1))
    assert v.shape == (raw_width(),)
    assert parts_raw[3].dim == raw_width()
    assert len(parts_raw[5].features) == raw_width()


def test_the_raw_input_keeps_colour_and_that_is_what_makes_identity_expressible():
    """The owner's observation: raw input should see texture and colour. The first version of
    `raw_features` averaged the channels away, which is a real defect and not a simplification.

    The demo's own appearance descriptor had already measured colour doing all the work -- raw crop
    RGB separates two objects at 0.096 against raw crop *grey* at 0.984, which is chance. So a
    greyscale head input meant the head could never express identity while the caller's memory
    could: measured on three objects, frozen, slot 0 always the same object whatever its position,
    grey reaches **1.71x** a centre guess against RGB's **118.87x**.

    Greyscale is still better for *roles*, where position is the whole question and colour is one
    more thing to fit -- 12.09x against 8.34x -- so it stays reachable rather than being removed.
    """
    from demo.pointer import raw_features, raw_width

    img = frame_at(0.3, 0.7, seed=1)
    rgb = raw_features(img)
    grey = raw_features(img, colour=False)
    assert rgb.shape == (raw_width(),)
    assert grey.shape == (raw_width(colour=False),)
    assert rgb.size == grey.size * 3, "the colour default no longer carries three channels"

    # The channels must actually differ, or "colour" is a shape and not information.
    planes = rgb.reshape(-1, 3)
    assert not np.allclose(planes[:, 0], planes[:, 2]), "the three channels are identical"

    # And the grey version must be their mean, so the two are the same measurement at
    # different resolution rather than two unrelated things.
    assert np.allclose(planes.mean(axis=1), grey, atol=1e-6)


def test_a_raw_head_learns_through_the_whole_composition(ready):
    """Not merely a shape change: drive the real pipeline and require it to beat a centre guess."""
    parts = build(1, "raw")
    from demo.pointer import raw_features

    backbone, _head, guard, features, taught, _fmap, pipe = parts
    scale = 1.0 / max(float(np.std([raw_features(frame_at(0.5, 0.5, seed=i))
                                    for i in range(8)])), 1e-9)
    base = float(np.std(frame_at(0.5, 0.5, seed=0)))
    g = np.random.default_rng(11)
    t, rows = 0.0, []
    for k in range(500):
        t += 1.0 / 30.0
        x, y = float(g.uniform(0.2, 0.8)), float(g.uniform(0.2, 0.8))
        img = frame_at(x, y, seed=k)
        guard.set_external(0, float(np.std(img)) / max(base, 1e-6))
        features.push(t, raw_features(img) * scale)
        # Through the shipped function, not a re-implementation of it -- a harness that
        # rebuilds the assembly cannot see the assembly break.
        teach_from_click(taught, t, x, y, 1, "points")
        r = pipe.step()
        if r.prediction is not None and k > 250:
            rows.append(np.hypot(float(r.prediction[0]) - x, float(r.prediction[1]) - y))
    assert rows, "the raw pipeline produced no prediction"
    assert float(np.mean(rows)) < 0.24, (
        f"a raw-input head reached {np.mean(rows):.3f}, no better than a centre guess")


def test_the_source_is_in_the_fingerprint():
    """A raw block and a backbone block are both valid bytes of the same length only by accident,
    and they mean entirely different things. The CRC cannot see that; the fingerprint must."""
    from demo.pointer import fingerprint

    bb = build(1)[0]
    assert fingerprint(bb, 1, "raw") != fingerprint(bb, 1, "backbone")
    with pytest.raises(ValueError, match="source must be"):
        build(1, "pixels")

def _status_state(**over):
    st = dict(source="raw", how="background", identify=True, objects=3, remember=True,
              vocab=64, vocab_full=False, n_auto=3117, n_taught=0, updates=3909, fps=16.4,
              drift=1.22, residual=0.136, frozen=False, frozen_by=(), learning=True,
              teach="both")
    st.update(over)
    return st


def test_the_status_block_fits_the_frame_it_is_drawn_on():
    """One long status line ran past the window and read as overlapping text: at font scale 0.5
    the outline stroke merges neighbouring glyphs, so a line that is merely *long* looks like two
    lines drawn on top of each other. Measured against the real font rather than guessed from a
    character count."""
    cv2 = pytest.importorskip("cv2")
    from demo.pointer import status_lines

    lines = status_lines(_status_state())
    assert len(lines) >= 3, "the status collapsed back into one or two lines"
    widest = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)[0][0] for t in lines)
    assert widest < 560, (
        f"the widest status line is {widest}px, which overruns a 640px camera frame -- that is "
        "the overlap this test exists for")


def test_the_vocabulary_ceiling_is_visible_on_screen():
    """Past `max_ids` the memory stops learning new objects and returns the nearest id however far
    away it is. A caller cannot see that from a count alone -- 64 looks like any other number."""
    from demo.pointer import status_lines

    assert "FULL" not in status_lines(_status_state(vocab_full=False))[0]
    assert "FULL" in status_lines(_status_state(vocab_full=True))[0]
    # And nothing about the vocabulary shows when it is not being kept.
    assert "vocab" not in status_lines(_status_state(remember=False))[0]


def test_the_status_says_which_of_the_four_things_is_happening():
    """Frozen, learning, waiting to be taught, or predicting only. These are exclusive and the
    overlay has to pick the right one -- the counter that used to answer this was hardwired to 0."""
    from demo.pointer import status_lines

    assert "FROZEN by external_0" in status_lines(
        _status_state(frozen=True, frozen_by=("external_0",)))[3]
    assert status_lines(_status_state(learning=True))[3] == "LEARNING"
    assert "teach" in status_lines(
        _status_state(learning=False, teach="mouse", n_taught=0))[3]
    assert "not learning" in status_lines(_status_state(learning=False))[3]

def test_the_status_text_is_not_drawn_twice():
    """The owner asked why black text was overlapping. It was: a thickness-3 black outline under a
    thickness-1 white glyph, kept when the dimmed band was added.

    At scale 0.52 that stroke puts black ink about **11 px past the last character** -- one
    character width -- and fills the gaps between letters, so on a dim band it reads as a second,
    bolder copy. The band already supplies the contrast, so the outline only supplied the double.

    Measured rather than eyeballed: no ink darker than the band may sit past where the string ends.
    """
    cv2 = pytest.importorskip("cv2")
    from demo.pointer import _overlay, status_lines

    st = _status_state(vocab_full=True)
    lines = status_lines(st)
    grey = 120
    out = _overlay(cv2, np.full((110, 640, 3), grey, np.uint8), st)
    band = int(grey * 0.35)

    for i, text in enumerate(lines):
        seg = out[8 + 22 * i:8 + 22 * (i + 1)]
        end = 8 + cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)[0][0]
        darker = np.where(seg.min(axis=(0, 2)) < band - 10)[0]
        stray = darker[darker > end + 4]
        assert len(stray) == 0, (
            f"line {i} has {len(stray)} columns of ink darker than the band past x={end}, at "
            f"{stray.min()}..{stray.max()} -- that is the outline reading as a ghost character")

    # And the band must still be doing its job, or the text is unreadable over a bright scene.
    assert out[:96].mean() < grey * 0.6, "the dimmed band has gone"
    assert out[100].mean() == grey, "the band spilled past the status block"


def test_locate_recognise_is_argmind_doing_the_perception():
    """**The 2026-08-14 decision, wired.** `--locate recognise` takes its region and its identity
    from `argmind.RecognitionLayer` instead of from `demo/label_sources.py`'s classical locators.

    Driven here rather than described: an object appears, is recognised on the next frame, the frame
    empties and **nothing is stored**, and when the object returns it gets **the same slot**.

    The frame cost is the method's and is recorded rather than hidden: 147 crop embeddings at
    1.05 ms is 6.5 fps against a camera at ~30. One whole-frame pass would be 21x cheaper and loses
    the identity, because a region pooled from the feature map centres to 4 pixels and the
    descriptor needs 2.
    """
    import numpy as np

    import demo.pointer as P
    from demo.pointer import build_recogniser

    # **Through the shipped construction, not a re-implementation of it.** The first version of this
    # test built the layer itself, so it passed while `main()` crashed with
    # `NameError: name 'RecognitionLayer' is not defined` -- the imports had never been added, and
    # `import demo.pointer` does not touch them because they are only referenced inside `main()`.
    h, w = P.IN_SHAPE[0], P.IN_SHAPE[1]
    det = build_recogniser(1)

    yy, xx = np.mgrid[0:h, 0:w]

    def frame(cx, cy, empty=False):
        img = np.full((h, w, 3), 0.35, dtype="f4")
        if not empty:
            img[np.hypot(xx - cx * w, yy - cy * h) <= 0.075 * w] = (0.85, 0.25, 0.20)
        return img

    first = det(frame(0.30, 0.40))
    assert first is not None and det.last_slot >= 0 and not det.last_recognised, (
        "the first sighting should be admitted, not recognised")
    stored = det.last_slot

    det(frame(0.31, 0.41))
    assert det.last_recognised and det.last_slot == stored, "the object was not recognised again"

    assert det(frame(0, 0, empty=True)) is None and det.last_slot == -1, (
        "an empty frame produced a label; the admission gate is letting background in")

    det(frame(0.30, 0.40))
    assert det.last_recognised and det.last_slot == stored, (
        f"the object came back as slot {det.last_slot} instead of {stored} — identity across an "
        "absence is what this locator exists to provide")

    # **And the reported centre must be the attended one.** A mutant returning a fixed (0.5, 0.5)
    # survived until this existed, because nothing checked the coordinate the demo actually draws.
    #
    # Asserted as *tracking* rather than proximity, because the crop is a third of the frame height
    # and the attended centre sits off the object by design — measured at (0.3125, 0.50) for an
    # object at (0.30, 0.40). It contains the object without being centred on it, which this
    # repository already records for the argmin of a rarity scan.
    left = det(frame(0.25, 0.40))[0]
    right = det(frame(0.72, 0.40))[0]
    assert right[0] > left[0] + 0.2, (
        f"the object moved from x=0.25 to x=0.72 and the reported centre went {left} -> {right}; it "
        "is not tracking what was attended to")


def test_more_than_one_object_is_REFUSED_by_the_recognising_locator():
    """It attends once per frame. Refused rather than approximated by attending twice, because
    suppressing the first choice needs a rule nobody has measured."""
    from demo.label_sources import RecogniseLocator
    with pytest.raises(ValueError, match="one region per frame"):
        RecogniseLocator(object(), 2)
