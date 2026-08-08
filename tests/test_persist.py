# SPDX-License-Identifier: Apache-2.0
"""Tests for persisting adaptation state across a power cycle.

Every failure exercised here is one a device actually produces: a write cut short
by power loss, a bit flipped in storage, a slot that was never written, a block
restored into a configuration it does not belong to. The one that matters most is
the torn write, because it is the one where a naive implementation destroys
working state while trying to save it.
"""

import struct
import subprocess
import sys

import numpy as np
import pytest

from argmind import (Aligner, Bound, FrameMap, Guard, Pipeline,
                     RecursiveLeastSquares, Store)
from argmind.persist import (FORMAT_VERSION, MAGIC, Record, fingerprint_of,
                             pack, unpack)


def trained(n=4, updates=50, seed=71, dtype="f8"):
    a = RecursiveLeastSquares(n_features=n, forgetting=0.99, dtype=dtype)
    rng = np.random.default_rng(seed)
    for _ in range(updates):
        x = rng.standard_normal(n)
        a.update(x, [float(x @ np.arange(1, n + 1))])
    return a


# ---------------------------------------------------------------------------
# Round trip
# ---------------------------------------------------------------------------

def test_a_saved_adapter_comes_back_identical(tmp_path):
    a = trained()
    store = Store(tmp_path / "state")
    store.save(a)

    fresh = RecursiveLeastSquares(n_features=4, forgetting=0.99, dtype="f8")
    rec = store.load(fresh)
    assert rec is not None
    assert np.array_equal(fresh.state, np.array(a.state)), (
        "restored state is not bit-identical to what was saved")
    probe = np.array([0.3, -1.2, 0.7, 2.0])
    assert np.array_equal(fresh.predict(probe), a.predict(probe))


def test_a_power_cycle_mid_learning_resumes_where_it_left_off(tmp_path):
    """The whole point, end to end: learn, save, throw everything away, come
    back, and carry on as though nothing happened."""
    rng = np.random.default_rng(83)
    theta = np.array([1.5, -2.0, 0.5])
    xs = rng.standard_normal((200, 3))
    ys = xs @ theta

    store = Store(tmp_path / "state")
    live = RecursiveLeastSquares(3, forgetting=0.99, dtype="f8")
    for x, y in zip(xs[:100], ys[:100]):
        live.update(x, [y])
    store.save(live)
    for x, y in zip(xs[100:], ys[100:]):
        live.update(x, [y])

    rebooted = RecursiveLeastSquares(3, forgetting=0.99, dtype="f8")
    assert store.load(rebooted) is not None
    for x, y in zip(xs[100:], ys[100:]):
        rebooted.update(x, [y])

    assert np.array_equal(rebooted.state, np.array(live.state)), (
        "a rebooted adapter diverged from the one that never stopped")


def test_pack_and_unpack_are_inverses_without_touching_the_disk():
    a = trained()
    rec = unpack(pack(a, sequence=7, fingerprint=0xDEADBEEF))
    assert rec.version == FORMAT_VERSION
    assert rec.sequence == 7
    assert rec.fingerprint == 0xDEADBEEF
    assert (rec.n_features, rec.n_outputs) == (4, 1)
    assert rec.dtype == np.dtype("f8")
    assert rec.frozen is False
    assert np.array_equal(rec.state, np.array(a.state))
    assert isinstance(rec, Record)


@pytest.mark.parametrize("dtype", ["f4", "f8"])
def test_both_supported_precisions_round_trip(tmp_path, dtype):
    a = trained(dtype=dtype)
    store = Store(tmp_path / f"s_{dtype}")
    store.save(a)
    fresh = RecursiveLeastSquares(4, forgetting=0.99, dtype=dtype)
    store.load(fresh)
    assert np.array_equal(fresh.state, np.array(a.state))


# ---------------------------------------------------------------------------
# The failures a device actually produces
# ---------------------------------------------------------------------------

def test_a_write_interrupted_by_power_loss_does_not_destroy_the_last_good_state(
        tmp_path):
    """The reason there are two slots at all.

    A single-slot store overwrites the only copy it has; lose power during that
    and there is nothing left. Here the interrupted write lands in the slot that
    is *not* in use, so the previous record survives untouched.
    """
    store = Store(tmp_path / "state")
    old = trained(updates=50)
    store.save(old)                      # slot 0
    new = trained(updates=200, seed=5)
    store.save(new)                      # slot 1

    # power fails partway through the next write, which targets slot 0 again
    victim = store.slot_path(0)
    victim.write_bytes(victim.read_bytes()[:40])

    fresh = RecursiveLeastSquares(4, forgetting=0.99, dtype="f8")
    rec = store.load(fresh)
    assert rec is not None, "a torn write took the whole store with it"
    assert np.array_equal(fresh.state, np.array(new.state)), (
        "recovered a record, but not the newest intact one")

    _, problems = store.read()
    assert any("cut short" in p or "too short" in p for p in problems), (
        f"the torn slot was skipped without saying why: {problems}")


def test_a_single_flipped_bit_is_caught(tmp_path):
    store = Store(tmp_path / "state")
    store.save(trained())
    p = store.slot_path(0)
    blob = bytearray(p.read_bytes())
    blob[-9] ^= 0x01                    # inside the payload, not the CRC
    p.write_bytes(bytes(blob))

    with pytest.raises(ValueError, match="CRC mismatch"):
        unpack(bytes(blob))
    fresh = RecursiveLeastSquares(4, forgetting=0.99, dtype="f8")
    assert store.load(fresh) is None, "a corrupt slot was loaded anyway"


def test_no_state_at_all_is_a_return_value_not_an_exception(tmp_path):
    """First boot is normal. It must be distinguishable from a fault, and it
    must not be an exception the caller has to wrap every start-up in."""
    store = Store(tmp_path / "nothing")
    fresh = RecursiveLeastSquares(4, dtype="f8")
    assert store.load(fresh) is None
    rec, problems = store.read()
    assert rec is None
    assert problems == ["slot 0: not written", "slot 1: not written"]


def test_both_slots_corrupt_reports_both_reasons(tmp_path):
    store = Store(tmp_path / "state")
    store.save(trained())
    store.save(trained(seed=2))
    for i in (0, 1):
        p = store.slot_path(i)
        p.write_bytes(p.read_bytes()[:30])

    rec, problems = store.read()
    assert rec is None
    assert len(problems) == 2, (
        f"lost the diagnosis for one of the slots: {problems}")


def test_a_record_from_a_future_format_is_refused_rather_than_guessed_at():
    blob = bytearray(pack(trained()))
    struct.pack_into("<H", blob, 4, FORMAT_VERSION + 1)
    with pytest.raises(ValueError, match="newer than this build"):
        unpack(bytes(blob))


def test_something_that_is_not_a_record_is_rejected():
    with pytest.raises(ValueError, match="bad magic"):
        unpack(b"NOPE" + bytes(64))
    with pytest.raises(ValueError, match="too short"):
        unpack(b"")
    assert MAGIC == b"ARGM"


# ---------------------------------------------------------------------------
# Restoring into the wrong thing
# ---------------------------------------------------------------------------

def test_a_block_restored_into_the_wrong_shape_or_precision_is_refused(tmp_path):
    store = Store(tmp_path / "state")
    store.save(trained(n=4))

    with pytest.raises(ValueError, match="4x1 but this adapter is 3x1"):
        store.load(RecursiveLeastSquares(3, dtype="f8"))
    with pytest.raises(ValueError, match="record is float64, adapter is float32"):
        store.load(RecursiveLeastSquares(4, dtype="f4"))


def test_the_fingerprint_catches_a_reordered_frame_map(tmp_path):
    """The dangerous case, because nothing downstream can catch it.

    Swapping two features produces a state block that loads cleanly, has the
    right shape and precision, and is silently wrong: parameter 0 now multiplies
    a different sensor.
    """
    def pipe_with(order):
        streams = [_stream(n) for n in ("raw", "temp", "ref")]
        return Pipeline(streams=streams, aligner=Aligner.hold_last("ref"),
                        adapter=RecursiveLeastSquares(3, dtype="f8"),
                        frame_map=FrameMap(order, "ref", constant=True))

    a = pipe_with(["raw", "temp"])
    b = pipe_with(["temp", "raw"])
    fa = fingerprint_of("rls", 3, 1, "float64", a.frame_map.slots)
    fb = fingerprint_of("rls", 3, 1, "float64", b.frame_map.slots)
    assert fa != fb, "reordering the features did not change the fingerprint"

    store = Store(tmp_path / "state")
    store.save(a.adapter, fingerprint=fa)
    with pytest.raises(ValueError, match="fingerprint"):
        store.load(b.adapter, fingerprint=fb)

    # and the matching configuration still loads
    assert store.load(a.adapter, fingerprint=fa) is not None


def _stream(name):
    from argmind import Stream
    s = Stream(name, dim=1, capacity=8, dtype="f8")
    s.push(t=1.0, values=[1.0])
    return s


def test_fingerprints_are_stable_across_processes():
    """Python's hash() is salted per process. A fingerprint built on it would
    mismatch on the next boot and reject every valid state block — which is a
    failure that only shows up in the field, on the second run."""
    expr = ("from argmind.persist import fingerprint_of; "
            "print(fingerprint_of('rls', 3, 1, 'float64', (('raw', 0), ('t', 1))))")
    here = fingerprint_of("rls", 3, 1, "float64", (("raw", 0), ("t", 1)))
    out = subprocess.run([sys.executable, "-c", expr], capture_output=True,
                         text=True, cwd="."). stdout.strip()
    assert int(out) == here, "the fingerprint changed between processes"


def test_a_zero_fingerprint_means_do_not_check(tmp_path):
    """So that a caller who has nothing to fingerprint is not forced to invent
    one, and an existing store is not invalidated by adding the argument later."""
    store = Store(tmp_path / "state")
    store.save(trained())
    assert store.load(RecursiveLeastSquares(4, forgetting=0.99, dtype="f8")) is not None


# ---------------------------------------------------------------------------
# The guard's frozen flag: carried, and what to do with it is the caller's call
# ---------------------------------------------------------------------------

def frozen_guard():
    g = Guard(RecursiveLeastSquares(4, forgetting=0.99, dtype="f8"),
              [Bound("p_trace", hi=1e-9, name="trace")])
    rng = np.random.default_rng(11)
    g.update(rng.standard_normal(4), [1.0])
    assert g.frozen
    return g


def test_the_frozen_flag_and_its_cause_survive_the_write(tmp_path):
    store = Store(tmp_path / "state")
    store.save(frozen_guard())
    rec, _ = store.read()
    assert rec.frozen is True
    assert rec.frozen_by == ("trace",), (
        "the state came back frozen with no record of why")


@pytest.mark.parametrize("policy,expect_frozen", [("thaw", False), ("keep", True)])
def test_on_frozen_policy_decides_what_a_reboot_means(tmp_path, policy,
                                                      expect_frozen):
    """There is no obviously correct default here, so it is the caller's choice.
    `thaw` risks re-learning into the same trouble; `keep` risks stranding a
    device that a power cycle would have fixed."""
    store = Store(tmp_path / "state")
    store.save(frozen_guard())

    fresh = Guard(RecursiveLeastSquares(4, forgetting=0.99, dtype="f8"),
                  [Bound("p_trace", hi=1e9)])
    store.load(fresh, on_frozen=policy)
    assert fresh.frozen is expect_frozen


def test_on_frozen_refuse_raises_and_names_the_cause(tmp_path):
    store = Store(tmp_path / "state")
    store.save(frozen_guard())
    fresh = Guard(RecursiveLeastSquares(4, forgetting=0.99, dtype="f8"),
                  [Bound("p_trace", hi=1e9)])
    with pytest.raises(ValueError, match=r"frozen by \['trace'\]"):
        store.load(fresh, on_frozen="refuse")


def test_keeping_a_freeze_needs_something_to_keep_it_in(tmp_path):
    """A guarded run saved, an unguarded run restoring. Silently dropping the
    freeze would be the worst outcome; it says so instead."""
    store = Store(tmp_path / "state")
    store.save(frozen_guard())
    with pytest.raises(ValueError, match="no freeze path"):
        store.load(RecursiveLeastSquares(4, forgetting=0.99, dtype="f8"),
                   on_frozen="keep")


def test_an_unknown_on_frozen_policy_is_refused(tmp_path):
    with pytest.raises(ValueError, match="thaw/keep/refuse"):
        Store(tmp_path / "s").load(RecursiveLeastSquares(2, dtype="f8"),
                                   on_frozen="ignore")


# ---------------------------------------------------------------------------
# Cadence, because flash wears out
# ---------------------------------------------------------------------------

def test_cadence_by_updates_writes_only_when_due(tmp_path):
    store = Store(tmp_path / "state", every_updates=25)
    a = trained(updates=0)
    wrote = [store.maybe_save(a, updates=n) for n in range(101)]
    assert wrote[0] is True, "no record existed until the first interval elapsed"
    assert sum(wrote) == 5, f"wrote {sum(wrote)} times in 101 updates at every 25"
    assert store.writes == 5


def test_cadence_by_time_writes_only_when_due(tmp_path):
    store = Store(tmp_path / "state", every_seconds=10.0)
    a = trained(updates=0)
    wrote = [store.maybe_save(a, t=float(i)) for i in range(61)]
    assert sum(wrote) == 7           # t = 0, 10, 20, 30, 40, 50, 60
    assert store.writes == 7


def test_a_hundred_hertz_stream_would_destroy_a_sector_without_cadence(tmp_path):
    """The arithmetic that makes cadence a required decision rather than a
    nicety: at 100 Hz, saving every update is 8.6 million writes a day against a
    sector rated for perhaps 100k erase cycles."""
    store = Store(tmp_path / "state", every_updates=1000)
    a = trained(updates=0)
    for n in range(10_000):
        store.maybe_save(a, updates=n)
    assert store.writes == 10
    naive = 10_000
    assert store.writes * 1000 == naive, "cadence did not reduce writes as declared"


def test_a_store_with_no_cadence_refuses_to_guess(tmp_path):
    store = Store(tmp_path / "state")
    with pytest.raises(ValueError, match="no cadence"):
        store.maybe_save(trained(), updates=1)


def test_cadence_needs_the_input_it_was_configured_for(tmp_path):
    with pytest.raises(ValueError, match="no updates count"):
        Store(tmp_path / "a", every_updates=10).maybe_save(trained(), t=1.0)
    with pytest.raises(ValueError, match="no time given"):
        Store(tmp_path / "b", every_seconds=10).maybe_save(trained(), updates=1)


@pytest.mark.parametrize("kwargs", [{"every_updates": 0}, {"every_seconds": 0.0},
                                    {"every_seconds": -1.0}])
def test_invalid_cadence_is_refused(tmp_path, kwargs):
    with pytest.raises(ValueError):
        Store(tmp_path / "s", **kwargs)


# ---------------------------------------------------------------------------
# Slot mechanics
# ---------------------------------------------------------------------------

def test_saves_alternate_slots_and_the_sequence_increases(tmp_path):
    store = Store(tmp_path / "state")
    a = trained()
    seqs = [store.save(a) for _ in range(4)]
    assert seqs == [1, 2, 3, 4]
    assert unpack(store.slot_path(0).read_bytes()).sequence == 3
    assert unpack(store.slot_path(1).read_bytes()).sequence == 4


def test_the_newest_valid_slot_wins_regardless_of_which_file_it_is(tmp_path):
    store = Store(tmp_path / "state")
    first = trained(updates=10)
    store.save(first)
    second = trained(updates=200, seed=3)
    store.save(second)

    fresh = RecursiveLeastSquares(4, forgetting=0.99, dtype="f8")
    rec = store.load(fresh)
    assert rec.sequence == 2
    assert np.array_equal(fresh.state, np.array(second.state))


def test_loading_continues_the_sequence_rather_than_restarting_it(tmp_path):
    """A reboot that restarted numbering at 1 would make the stale slot look
    newer on the write after next, and the store would load backwards."""
    first = Store(tmp_path / "state")
    a = trained()
    first.save(a)
    first.save(a)
    assert first.sequence == 2

    after_reboot = Store(tmp_path / "state")
    after_reboot.load(RecursiveLeastSquares(4, forgetting=0.99, dtype="f8"))
    assert after_reboot.sequence == 2
    assert after_reboot.save(a) == 3


def test_the_record_is_little_endian_whatever_the_host_is():
    """The writer and the reader may be different machines, so the byte order is
    declared rather than inherited.

    Note the limit of this test, because it is worth being explicit about: on a
    little-endian host, ``<`` and ``=`` produce byte-identical output, so no
    amount of comparing bytes can tell a declared byte order from an inherited
    one. Everything below the format assertions would pass on x86 either way. The
    declaration itself is therefore what is pinned — the only check that can fail
    here, and the one that matters when the C++ core has to read these blocks
    from a device that may not share the host's endianness.
    """
    from argmind.persist import _CRC, _HEADER

    assert _HEADER.format.startswith("<"), (
        f"header byte order is {_HEADER.format[0]!r}, not declared little-endian")
    assert _CRC.format.startswith("<")

    a = RecursiveLeastSquares(2, n_outputs=1, dtype="f4")
    blob = pack(a)
    magic, version = struct.unpack_from("<4sH", blob, 0)
    assert magic == MAGIC and version == FORMAT_VERSION
    # p0 defaults to 100.0; the first covariance element must be there in LE f4
    assert struct.pack("<f", 100.0) in blob
    assert unpack(blob).state.dtype.byteorder in ("=", "<")


def test_a_reboot_never_writes_over_the_record_it_just_loaded(tmp_path):
    """A bug found by the parity verifier while checking the C++ port, not by review.

    `__init__` starts at slot 0 and `load()` used to update only the sequence number, so
    after any ODD number of writes in the previous session the newest record sat in slot 0
    and the first write after a reboot landed on top of it. Measured before the fix: three
    saves leave sequences {slot 0: 3, slot 1: 2}, and the next save destroys 3.

    That is precisely the failure two slots exist to prevent. The scheme's promise is that
    an interrupted write can destroy at most the copy *not* in use; aimed at the copy that
    is, it destroys a generation more than it should.
    """
    a = trained()
    store = Store(tmp_path / "state")
    for _ in range(3):                          # odd, so the newest lands back in slot 0
        store.save(a)
    seqs = {i: unpack(store.slot_path(i).read_bytes()).sequence for i in (0, 1)}
    assert seqs == {0: 3, 1: 2}, f"the fixture no longer sets this up: {seqs}"

    rebooted = Store(tmp_path / "state")
    rec = rebooted.load(RecursiveLeastSquares(4, forgetting=0.99, dtype="f8"))
    assert rec.sequence == 3
    assert rebooted.live_slot == 0, "the record in use came from slot 0"

    rebooted.save(a)
    after = {i: unpack(store.slot_path(i).read_bytes()).sequence for i in (0, 1)}
    assert 3 in after.values(), (
        f"the write after the reboot destroyed the record it had just loaded: {after}")
    assert after[1] == 4, f"the new record should have gone to slot 1: {after}"


@pytest.mark.parametrize("writes", [1, 2, 3, 4, 5])
def test_the_newest_record_survives_a_reboot_and_a_write_at_any_parity(tmp_path, writes):
    """The general form. Whichever slot the newest record is in, the next write must go to
    the other one — so a power failure during it leaves the newest intact.

    Only the odd cases discriminate: after an even number of writes the newest record is
    already in slot 1 and the naive slot-0 start happens to be right. They are kept because
    a fix that broke the even case would be a different bug, and this is where it would
    show. An audit pointed out that the name says "any parity" while half the cases cannot
    fail — so the asymmetry is written down rather than implied.
    """
    a = trained()
    store = Store(tmp_path / f"s{writes}")
    for _ in range(writes):
        store.save(a)
    newest = max(unpack(store.slot_path(i).read_bytes()).sequence
                 for i in (0, 1) if store.slot_path(i).exists())

    rebooted = Store(tmp_path / f"s{writes}")
    rebooted.load(RecursiveLeastSquares(4, forgetting=0.99, dtype="f8"))
    target = rebooted._slot
    held = unpack(store.slot_path(target).read_bytes()).sequence \
        if store.slot_path(target).exists() else None
    assert held != newest, (
        f"after {writes} writes the next write targets slot {target}, which holds the "
        f"newest record (sequence {newest})")


def test_live_slot_reports_nothing_before_anything_is_written(tmp_path):
    store = Store(tmp_path / "fresh")
    assert store.live_slot == -1
    store.load(RecursiveLeastSquares(4, forgetting=0.99, dtype="f8"))
    assert store.live_slot == -1, "a first boot has no live slot to report"


def test_a_corrupted_dtype_code_is_rejected():
    """A gap a mutation check found: removing the check left `_CODE_DTYPE[code]` to raise
    `KeyError` two lines later, and nothing in the suite distinguished that from the
    `ValueError` the format contract promises. A caller catching `ValueError` around a
    load — which is what every other refusal here raises — would not have caught it."""
    import zlib
    blob = bytearray(pack(trained()))
    struct.pack_into("<B", blob, 16, 99)          # neither 4 nor 8
    struct.pack_into("<I", blob, len(blob) - 4,
                     zlib.crc32(bytes(blob[:-4])) & 0xFFFFFFFF)
    with pytest.raises(ValueError, match="unknown dtype code"):
        unpack(bytes(blob))


def test_an_overlong_freeze_cause_is_truncated_rather_than_crashing():
    """The cause length is one byte in the header, so `pack` truncates at 255. Without
    that clamp `struct.pack` raises `struct.error` — a crash while trying to save state,
    which is the worst possible moment for one. No test reached this path before, because
    the only over-length case went through the C++ side."""
    class Frozen:
        def __init__(self, inner):
            self.inner = inner
            self.frozen = True
            self.frozen_by = tuple(f"check_number_{i}" for i in range(40))
        n_features = property(lambda s: s.inner.n_features)
        n_outputs = property(lambda s: s.inner.n_outputs)
        dtype = property(lambda s: s.inner.dtype)
        state = property(lambda s: s.inner.state)

    frozen = Frozen(trained())
    assert len(",".join(frozen.frozen_by)) > 255, "the fixture no longer overflows"

    rec = unpack(pack(frozen))
    assert rec.frozen is True
    assert len(",".join(rec.frozen_by)) <= 255
    assert rec.frozen_by[0] == "check_number_0", "truncation lost the leading causes"


def test_a_store_that_writes_before_it_reads_does_not_clobber_the_newest(tmp_path):
    """The other half of the reboot bug, and the half whose fix had no test.

    An audit found that deleting `_arbitrate_once()` from `save()` left the entire suite
    green: `load()` was protected and `save()` was not. Writing without loading first is an
    ordinary thing for a caller to do — a supervisor that only ever persists, say — and
    before the fix it destroyed the newest record *and regressed the sequence*, so from
    then on the older record won arbitration and the device silently reverted a generation.

    Measured before the fix: three saves leave {0: 3, 1: 2}; a fresh Store calling save()
    with no load() leaves {0: 1, 1: 2}.
    """
    a = trained()
    store = Store(tmp_path / "state")
    for _ in range(3):                          # odd, so the newest lands in slot 0
        store.save(a)
    assert {i: unpack(store.slot_path(i).read_bytes()).sequence
            for i in (0, 1)} == {0: 3, 1: 2}, "the fixture no longer sets this up"

    writer = Store(tmp_path / "state")           # a reboot that never loads
    assert writer.live_slot == -1, "it has not looked at the slots yet"
    seq = writer.save(a)

    after = {i: unpack(store.slot_path(i).read_bytes()).sequence for i in (0, 1)}
    assert 3 in after.values(), (
        f"the write destroyed the newest record without ever reading it: {after}")
    assert seq == 4, f"the sequence regressed to {seq} instead of advancing to 4"
    assert after == {0: 3, 1: 4}, after


def test_maybe_save_also_arbitrates_before_its_first_write(tmp_path):
    """`maybe_save` goes through `save`, so the same protection has to reach it — a cadence
    that writes on its first call must not aim that write at the live slot either."""
    a = trained()
    store = Store(tmp_path / "state")
    for _ in range(3):
        store.save(a)

    writer = Store(tmp_path / "state", every_updates=10)
    assert writer.maybe_save(a, updates=0) is True, "the first cadence call should write"
    after = {i: unpack(store.slot_path(i).read_bytes()).sequence for i in (0, 1)}
    assert after == {0: 3, 1: 4}, after
