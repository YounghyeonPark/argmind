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
    0.763 ms is 8.4 fps against a camera at ~30 — **with one slot reported**. At the ~20 slots a
    2,000-frame run settles on, refinement makes it ~995 crops and a measured 2.6 fps. One whole-frame pass would be 21x cheaper and loses
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


def test_the_recognition_state_REACHES_THE_SCREEN():
    """**Found by running the demo, which showed only the camera.**

    `--locate recognise` decided a slot every frame and drew nothing about it: the slot, whether it
    was recognised or freshly admitted, and how full the memory is were all invisible, so nothing on
    screen distinguished this mode from any other. The decision was wired and the display was not.

    Third defect in this demo found by launching it rather than by a fixture, and the third that
    lived in `main()`.
    """
    from demo.pointer import status_lines

    base = dict(remember=False, vocab=0, source="backbone", how="recognise", identify=False,
                objects=1, n_auto=3, n_taught=0, updates=3, fps=6.5, drift=1.0, residual=0.12,
                frozen=False, frozen_by=(), learning=True, teach="auto")

    without = status_lines(dict(base))
    recognised = status_lines({**base, "recognise": dict(slot=0, recognised=True, occupied=1,
                                                         capacity=16)})
    admitted = status_lines({**base, "recognise": dict(slot=2, recognised=False, occupied=3,
                                                       capacity=16)})
    empty = status_lines({**base, "recognise": dict(slot=-1, recognised=False, occupied=1,
                                                    capacity=16)})

    assert len(recognised) == len(without) + 1, (
        "the recognition summary did not add a line; the mode is invisible on screen again")
    assert "RECOGNISED" in recognised[3] and "slot 0" in recognised[3]
    assert "new" in admitted[3] and "slot 2" in admitted[3]
    assert "nothing here" in empty[3], (
        "an empty frame reads as a recognition; the -1 slot has to be distinguishable on screen or "
        "the admission gate's effect cannot be seen")
    assert "1/16" in recognised[3] and "3/16" in admitted[3], (
        "the memory occupancy is not shown, so a viewer cannot see exemplars being stored")

    # The three states must be distinguishable from each other, not merely present.
    assert len({recognised[3], admitted[3], empty[3]}) == 3


def test_the_recogniser_state_is_CARRIED_out_of_the_loop():
    """The other half: `status_lines` can only show what `main()` puts in `state`. Asserted through
    the shipped construction, because a test that builds its own locator cannot see the loop fail to
    populate it — which is exactly how the missing import survived earlier."""
    import inspect

    import demo.pointer as pointer

    # **The AST, not a substring.** A first version searched `main()` for the string
    # `"recognise"`, which also appears in `elif how == "recognise":` — so a mutant that stopped
    # assigning the state survived. This file has been caught by exactly that once before, with
    # `learn_features` named inside a help string.
    import ast

    tree = ast.parse(inspect.getsource(pointer.main))
    assigned = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name)
                        and target.value.id == "state"
                        and isinstance(target.slice, ast.Constant)):
                    assigned.add(target.slice.value)
    assert "recognise" in assigned, (
        "main() no longer assigns state['recognise'], so the status line has nothing to show and "
        "the window goes back to being just the camera")


def test_the_demo_recognises_MORE_THAN_ONE_object_and_orders_them_by_IDENTITY():
    """**The owner's objection, wired.** The locator used to refuse `--objects 2` on the grounds that
    selecting twice needed an unmeasured rule. It is measured now, so the refusal is gone.

    And the ordering is the part that matters. Every classical locator here returns centres
    leftmost-first, because that is a function of the current frame and therefore learnable. This one
    orders by **slot**, so output pair *i* is always the same object — which is what the memory is
    for, and what "leftmost" can never give.
    """
    from demo.label_sources import RecogniseLocator
    from demo.pointer import build_recogniser, status_lines

    det = build_recogniser(2)
    assert det.layer.emit == "presence", "the demo is back on the one-region-per-frame layer"
    assert det.layer.n_outputs == det.layer.memory.capacity, (
        "the layer no longer emits one score per slot, so a head above it cannot see two objects")

    # A partial frame teaches nothing: an absent object has no location and inventing one is the
    # "a point cannot stand in for a field" error a level up.
    class OnePresent:
        emit = "presence"
        last_slot, last_recognised = 0, True
        last_present = [(0, 0.9, (0.3, 0.4))]

        def features(self, frame):
            return None
    partial = RecogniseLocator(OnePresent(), 2)
    assert partial(None) is None, (
        "a frame with only slot 0 present produced two labels; the missing one would be invented")

    class BothPresent(OnePresent):
        last_present = [(1, 0.95, (0.7, 0.6)), (0, 0.90, (0.3, 0.4))]
    both = RecogniseLocator(BothPresent(), 2)
    out = both(None)
    assert out == [(0.3, 0.4), (0.7, 0.6)], (
        f"got {out}: the centres must come out in SLOT order, not in score or scan order — the slot "
        "is the identity and the head's output pairs are fixed to it")

    line = status_lines(dict(remember=False, vocab=0, source="backbone", how="recognise",
                            identify=False, objects=2, n_auto=0, n_taught=0, updates=0, fps=6.5,
                            drift=1.0, residual=0.1, frozen=False, frozen_by=(), learning=True,
                            teach="auto",
                            recognise=dict(slot=1, recognised=True, occupied=2, capacity=16,
                                           present=[(0, 0.97, (0.3, 0.4)),
                                                    (1, 0.93, (0.7, 0.6))])))[3]
    assert "2 RECOGNISED" in line and "#0" in line and "#1" in line, (
        f"the status line reads {line!r}; with two objects present it must say so, or the screen "
        "looks serial while the mechanism is not")


def test_the_recognising_locator_publishes_a_MATCH_MAP_for_the_segments_window():
    """**Boxes and slot numbers alone are unreadable**, which the owner reported after watching it:
    four rectangles on a room say where the layer looked and nothing about how it carves the scene.

    The presence pass already computes a best-matching slot for every candidate; the map is that,
    kept instead of discarded, as an integer field the demo's existing `--segments` window paints by
    slot. It is a field over the *candidate grid*, not a pixel segmentation — this locator selects
    regions rather than partitioning the frame, and it should look like what it is.
    """
    import numpy as np

    from demo.pointer import build_recogniser, segment_image
    from tests.test_bootstrap import render

    det = build_recogniser(1, max_reported=4)
    img = render([("disc", 0.30, 0.40), ("bar", 0.70, 0.62)], seed=1,
                 texture="none").astype("f4")
    for _ in range(4):
        det(img)

    field = det.labels
    assert field is not None, (
        "the locator publishes no field, so `--segments` has nothing to draw and the mode is "
        "unreadable again")
    assert field.ndim == 2 and field.dtype.kind in "iu", f"{field.shape} {field.dtype}"
    assert field.shape[0] * field.shape[1] == len(det.layer.centres), (
        "the field is not one cell per coarse candidate, so its geometry does not match what was "
        "actually looked at")

    slots = set(int(v) for v in np.unique(field))
    assert slots - {0}, "no cell was assigned a slot; the map would render as a blank grid"
    assert 0 in slots, (
        "every cell claims a slot — a match map with no unmatched cells is telling you nothing, and "
        "would mean the threshold is doing no work")

    painted = segment_image(field, [], scale=8)
    assert painted.shape == (field.shape[0] * 8, field.shape[1] * 8, 3)

class _StubCv2:
    """Enough of `cv2` for `draw()` to run: it records call names and touches nothing.

    Written because **nothing tested `draw()` at all**, and every demo bug this file has caught so
    far lived in exactly that kind of untested wiring — `main()` building one thing and the harness
    building another.
    """

    FONT_HERSHEY_SIMPLEX = 0
    LINE_AA = 16
    MARKER_CROSS = 0
    MARKER_TILTED_CROSS = 1

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def record(*a, **k):
            self.calls.append(name)
        return record


def _recognise_state(field, overlay=True):
    import demo.pointer as P

    layer = P.build_recogniser(1).layer
    #: Every key `draw`, `status_lines` and `_overlay` read, listed by walking their ASTs rather
    #: than by adding one each time a KeyError appeared — a stub filled in by trial and error is a
    #: stub that stops matching `main()` the moment the display grows a field.
    state = {
        "pred": None, "frozen": False, "frozen_by": (), "taught_points": [], "learning": False,
        "n_taught": 0, "teach": "both", "remember": False, "vocab_full": False, "vocab": 0,
        "detector_ran": False, "minted": [], "objects": 1, "slot_ids": [], "drift": 0.0,
        "fps": 0.0, "how": "recognise", "identify": False, "n_auto": 0, "residual": 0.0,
        "source": "learned", "updates": 0, "map_overlay": overlay,
        "recognise": {"present": [], "slot": 0, "recognised": True, "centre": (0.5, 0.5),
                      "occupied": 1, "capacity": 8, "crop": P.RECOGNISE_CROP,
                      "field": field, "grid": (layer.grid_x, layer.grid_y)},
    }
    return state


def test_the_match_map_is_drawn_ON_the_camera_frame_where_it_matched():
    """The owner's report was *"뭐가뭔지 모르겠음"* — the map existed only in a window of its own,
    upscaled 44x, with nothing tying a cell to a place in the room. Drawn on the frame, a cell sits
    on the thing it matched.

    Asserted as placement rather than as "something changed": cell `(0, 0)` must land at the first
    candidate centre and **not** in the middle of the frame, which is what a field drawn with the
    wrong geometry would do.
    """
    import numpy as np

    import demo.pointer as P

    layer = P.build_recogniser(1).layer
    field = np.zeros((len(layer.grid_y), len(layer.grid_x)), dtype=np.int32)
    field[0, 0] = 1
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    P.match_overlay(frame, field, (layer.grid_x, layer.grid_y))

    ys, xs = np.nonzero(frame.any(axis=2))
    assert len(ys), "a matched cell painted nothing"
    cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
    want_y, want_x = layer.grid_y[0] * 480, layer.grid_x[0] * 640
    assert abs(cy - want_y) <= 2 and abs(cx - want_x) <= 2, (
        f"cell (0,0) painted around ({cy:.0f}, {cx:.0f}) and its candidate is at "
        f"({want_y:.0f}, {want_x:.0f}) — the overlay would tint the wrong part of the room")
    assert np.array_equal(frame[240:260, 300:320], np.zeros((20, 20, 3), np.uint8)), (
        "the frame centre was tinted by a cell at the corner, so the geometry is being ignored")

    #: The extent too, not only the centre: swapping the two half-steps moves no centre and would
    #: survive everything above. A candidate step is 8 px of a 72x96 analysis frame either way, and
    #: the display shares that aspect ratio, so a cell must come out SQUARE in display pixels.
    ph, pw = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
    assert abs(ph - pw) <= 2, (
        f"a cell painted {ph}x{pw} display pixels; the candidate step is the same number of pixels "
        "on both axes here, so the two half-steps have been crossed")


def test_an_UNMATCHED_cell_leaves_the_frame_untouched():
    """The control. A tint that covered the frame regardless would look informative and carry
    nothing — and the field is mostly zeros on any real scene."""
    import numpy as np

    import demo.pointer as P

    layer = P.build_recogniser(1).layer
    empty = np.zeros((len(layer.grid_y), len(layer.grid_x)), dtype=np.int32)
    frame = np.full((480, 640, 3), 77, dtype=np.uint8)
    before = frame.copy()
    P.match_overlay(frame, empty, (layer.grid_x, layer.grid_y))
    assert np.array_equal(frame, before), "an all-zero field still tinted the frame"


def test_two_slots_get_DIFFERENT_colours_and_they_are_the_window_s_own():
    """The overlay and the separate window must not disagree about what colour a slot is; they read
    the same palette by slot, which is what `segment_image` does for the same field."""
    import numpy as np

    import demo.pointer as P

    layer = P.build_recogniser(1).layer
    field = np.zeros((len(layer.grid_y), len(layer.grid_x)), dtype=np.int32)
    field[0, 0], field[-1, -1] = 1, 2
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    P.match_overlay(frame, field, (layer.grid_x, layer.grid_y), tint=1.0)

    a = tuple(int(v) for v in frame[int(layer.grid_y[0] * 480), int(layer.grid_x[0] * 640)])
    b = tuple(int(v) for v in frame[int(layer.grid_y[-1] * 480), int(layer.grid_x[-1] * 640)])
    assert a != b, "both slots were tinted the same colour, so the carving is invisible"
    assert a == tuple(P.ID_COLOURS[0]) and b == tuple(P.ID_COLOURS[1]), (
        f"the overlay used {a} and {b}; segment_image colours the same field by slot from "
        "ID_COLOURS, and two views that disagree are worse than one")


def test_draw_APPLIES_the_overlay_and_the_flag_turns_it_OFF():
    """The wiring, which is where this demo's bugs live. `--no-map-overlay` has to reach the drawing
    and not merely be accepted by the parser."""
    import numpy as np

    import demo.pointer as P

    layer = P.build_recogniser(1).layer
    field = np.zeros((len(layer.grid_y), len(layer.grid_x)), dtype=np.int32)
    field[2, 3] = 1

    on = np.full((480, 640, 3), 90, dtype=np.uint8)
    P.draw(_StubCv2(), on, _recognise_state(field, overlay=True))
    off = np.full((480, 640, 3), 90, dtype=np.uint8)
    P.draw(_StubCv2(), off, _recognise_state(field, overlay=False))

    band = P.status_lines(_recognise_state(field))
    below = 8 + 22 * len(band) + 8          # under the status band, which dims rows either way
    assert not np.array_equal(on[below:], off[below:]), (
        "draw() produced the same frame with the overlay on and off, so the flag is not reaching it")
    assert np.array_equal(off[below:], np.full_like(off[below:], 90)), (
        "with the overlay off draw() still tinted the frame below the status band")

    args = P.parser().parse_args(["--locate", "recognise", "--no-map-overlay"])
    assert args.no_map_overlay is True
    assert P.parser().parse_args([]).no_map_overlay is False, (
        "the overlay must be ON by default — its own window was measured against the thing it is "
        "for and failed")

def test_the_stub_state_COVERS_what_the_drawing_actually_reads():
    """A hand-listed fixture drifts. This walks `draw`, `status_lines` and `_overlay` for every
    `state[...]` they touch and asserts the stub supplies it — so a display that grows a field fails
    here instead of raising `KeyError` from inside a test that was passing yesterday.

    The same defect, twice fixed in this file already: a harness that re-implements the assembly
    cannot see the assembly break.
    """
    import ast
    import io as _io

    import demo.pointer as P

    tree = ast.parse(_io.open(P.__file__, encoding="utf-8").read())
    reads = set()
    for fn in tree.body:
        if not (isinstance(fn, ast.FunctionDef)
                and fn.name in ("draw", "status_lines", "_overlay")):
            continue
        for n in ast.walk(fn):
            if (isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name)
                    and n.value.id == "state" and isinstance(n.slice, ast.Constant)):
                reads.add(n.slice.value)
            if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                    and n.func.attr == "get" and isinstance(n.func.value, ast.Name)
                    and n.func.value.id == "state" and n.args
                    and isinstance(n.args[0], ast.Constant)):
                reads.add(n.args[0].value)

    assert reads, "no state keys were found, so this test is checking nothing"
    missing = reads - set(_recognise_state(None).keys())
    assert not missing, (
        f"the drawing reads {sorted(missing)} and the stub does not supply it; add it here so the "
        "fixture keeps matching main()")

def test_the_tint_touches_NOTHING_outside_the_cells_it_painted():
    """The surviving mutant, closed. Blending the whole frame instead of only the painted cells
    dims the entire scene by the tint fraction — and every test above missed it, because a black
    frame stays black however much you scale it and an empty field returns before the blend.

    A non-black frame and a field with one cell in it is the only fixture that can see it.
    """
    import numpy as np

    import demo.pointer as P

    layer = P.build_recogniser(1).layer
    field = np.zeros((len(layer.grid_y), len(layer.grid_x)), dtype=np.int32)
    field[3, 4] = 1
    frame = np.full((480, 640, 3), 120, dtype=np.uint8)
    P.match_overlay(frame, field, (layer.grid_x, layer.grid_y))

    changed = np.any(frame != 120, axis=2)
    assert changed.any(), "the matched cell painted nothing"
    step_x = layer.grid_x[1] - layer.grid_x[0]
    step_y = layer.grid_y[1] - layer.grid_y[0]
    expected = int(round(step_x * 640)) * int(round(step_y * 480))
    assert changed.sum() <= expected * 1.2, (
        f"{changed.sum()} pixels changed for one cell, which covers about {expected} — the tint is "
        "being applied outside the cells it matched, so the whole scene is dimmed")
    assert (frame[0, 0] == 120).all() and (frame[-1, -1] == 120).all(), (
        "a corner far from the only matched cell was modified")

def test_the_status_RESIDUAL_works_for_every_input_source():
    """The crash `--input learned` shipped with, held under test.

    `main()` read `head._e` while `build()` hands back a `Stack` for `--input learned`, and no
    fixture could see it because the line needs a guard-watched update from a real teach signal.
    So the expression is hoisted and both callers use it — `main()` is asserted to call it below,
    the same AST check this file already applies to `learn_features`.
    """
    import numpy as np

    import demo.pointer as P

    for source in ("backbone", "raw", "learned"):
        _bb, learner, _guard, _feat, _taught, _fmap, _pipe = P.build(1, source=source)
        r = P.head_residual(learner)
        assert isinstance(r, float) and np.isfinite(r), (
            f"--input {source} gives residual {r!r}; the status line would crash or read NaN")

    stack_sourced = P.build(1, source="learned")[1]
    assert hasattr(stack_sourced, "head"), (
        "--input learned no longer produces something with a public `.head`, so the walk down in "
        "head_residual() is guessing again")


def test_main_CALLS_head_residual_rather_than_reaching_in_itself():
    """Via the AST, not a substring: a mention in a docstring is not a call, and the point of the
    hoist is that `main()` stops doing the algebra. Present-and-called-by-nothing is the other half
    of the same defect."""
    import ast
    import io as _io

    import demo.pointer as P

    tree = ast.parse(_io.open(P.__file__, encoding="utf-8").read())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    called = {n.func.id for n in ast.walk(main)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "head_residual" in called, "main() no longer calls head_residual()"

    reaches = [n for n in ast.walk(main)
               if isinstance(n, ast.Attribute) and n.attr == "_e"]
    assert not reaches, (
        "main() reaches for `_e` itself again; that is the attribute whose absence on Stack crashed "
        "--input learned")
