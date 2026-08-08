# SPDX-License-Identifier: Apache-2.0
"""The C++ persistence format against the Python reference.

Serialisation has no rounding, so the claim here is stronger than anywhere else in this
repository: **the two implementations produce byte-identical records**, and decode each
other's. Nothing in this file compares to a tolerance, because nothing here needs one.

That strength is why the format is worth testing this hard. A state block is the one thing
that crosses between a device and a host, and between one version of this library and the
next. A disagreement in the recursion shows up as a slightly different number; a
disagreement in the format shows up as a device that will not boot, or worse, one that
boots on a block it has misread.

Three properties this file exists to pin, each because the reference chose something that a
port could plausibly get wrong:

- The header is 26 bytes and Python's `struct` packs it without padding, so two of its
  three `uint32_t` fields land unaligned: `fingerprint` at offset 18 and `count` at 22.
  (`sequence` at offset 8 happens to be aligned.) A C++ struct laid over the buffer would
  insert padding and produce a longer record.
- The CRC is zlib's, which is the reflected IEEE 802.3 polynomial. A plain CRC-32 with the
  same polynomial but no reflection produces different bytes for the same input.
- Element bytes are little-endian by declaration rather than by inheriting the host's
  order.

If no C++ compiler is available these tests skip, for the same reason the recursion's parity
tests do: the Python reference must stay usable on a clone with no toolchain.
"""

import shutil
import struct
import subprocess
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

from argmind.adapt import RecursiveLeastSquares
from argmind.persist import (FORMAT_VERSION, MAGIC, Store, fingerprint_of, pack,
                             unpack)

REPO = Path(__file__).resolve().parent.parent
CPP = REPO / "cpp"

_JOB = struct.Struct("<4sHBBHHIIIII")     # 32 bytes; see cpp/tools/pcheck.cpp
_DECODED = struct.Struct("<iIIIIIII")     # 32 bytes

#: Status codes from argmind::persist::Status.
STATUS = {0: "ok", 1: "too_short", 2: "bad_magic", 3: "future_version",
          4: "bad_dtype", 5: "length_mismatch", 6: "crc_mismatch"}


def _compiler() -> str | None:
    for name in ("g++", "clang++"):
        if shutil.which(name):
            return name
    return None


@pytest.fixture(scope="session")
def pcheck(tmp_path_factory) -> Path:
    cxx = _compiler()
    if cxx is None:
        pytest.skip("no C++ compiler on PATH; the Python reference stands alone")
    out = tmp_path_factory.mktemp("cpp_p") / ("pcheck.exe" if sys.platform == "win32"
                                              else "pcheck")
    done = subprocess.run(
        [cxx, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Wpedantic", "-fno-exceptions",
         "-fno-rtti", "-I", str(CPP / "include"), str(CPP / "tools" / "pcheck.cpp"),
         "-o", str(out)], capture_output=True, text=True)
    if done.returncode != 0:
        pytest.fail(f"pcheck did not build:\n{done.stdout}\n{done.stderr}")
    return out


def cpp_pack(pcheck: Path, tmp_path: Path, state, *, n_features, n_outputs,
             sequence=0, fingerprint=0, frozen=False, cause=b"", dtype="f8") -> bytes:
    dt = np.dtype(dtype)
    arr = np.ascontiguousarray(state, dtype=dt)
    job = tmp_path / "job.bin"
    out = tmp_path / "out.bin"
    with open(job, "wb") as fh:
        fh.write(_JOB.pack(b"AMPC", 1, dt.itemsize, 0, n_features, n_outputs,
                           arr.size, sequence, fingerprint, 1 if frozen else 0,
                           len(cause)))
        fh.write(cause)
        fh.write(arr.tobytes())
    done = subprocess.run([str(pcheck), str(job), str(out)], capture_output=True,
                          text=True)
    assert done.returncode == 0, f"pcheck pack failed: {done.stdout}\n{done.stderr}"
    return out.read_bytes()


def cpp_unpack(pcheck: Path, tmp_path: Path, blob: bytes, dtype="f8"):
    """Returns (status_name, fields, cause_bytes, state_array)."""
    dt = np.dtype(dtype)
    job = tmp_path / "ujob.bin"
    out = tmp_path / "uout.bin"
    with open(job, "wb") as fh:
        fh.write(_JOB.pack(b"AMPC", 1, dt.itemsize, 1, 0, 0, 0, 0, 0, 0, 0))
        fh.write(blob)
    done = subprocess.run([str(pcheck), str(job), str(out)], capture_output=True,
                          text=True)
    assert done.returncode == 0, f"pcheck unpack failed: {done.stderr}"
    raw = out.read_bytes()
    (status, seq, fp, frozen, nf, no, count, cause_len) = _DECODED.unpack_from(raw, 0)
    fields = {"sequence": seq, "fingerprint": fp, "frozen": bool(frozen),
              "n_features": nf, "n_outputs": no, "count": count}
    if STATUS[status] != "ok":
        return STATUS[status], fields, b"", np.empty(0, dtype=dt)
    off = _DECODED.size
    cause = raw[off:off + cause_len]
    off += cause_len
    state = np.frombuffer(raw, dtype=dt, count=count, offset=off)
    return "ok", fields, cause, np.array(state)


def cpp_unpack_batch(pcheck: Path, tmp_path: Path, blobs, dtype="f8"):
    """Decode many blobs in one invocation.

    The sweeps below are thousands of decodes, and one process each put this file at 53
    seconds — essentially all of it process-spawn cost on Windows. A pre-commit hook that
    slow is a hook people learn to bypass, and shrinking the sweeps would have weakened
    exactly the claims they exist to make true. So the tool batches instead.
    """
    dt = np.dtype(dtype)
    job = tmp_path / "bjob.bin"
    out = tmp_path / "bout.bin"
    with open(job, "wb") as fh:
        fh.write(_JOB.pack(b"AMPC", 1, dt.itemsize, 3, 0, 0, 0, 0, 0, 0, 0))
        fh.write(struct.pack("<I", len(blobs)))
        for b in blobs:
            fh.write(struct.pack("<I", len(b)))
            fh.write(b)
    done = subprocess.run([str(pcheck), str(job), str(out)], capture_output=True,
                          text=True)
    assert done.returncode == 0, f"pcheck batch unpack failed: {done.stderr}"
    raw = out.read_bytes()
    assert len(raw) == len(blobs) * _DECODED.size, "short batch result"
    results = []
    for i in range(len(blobs)):
        status, seq, fp, frozen, nf, no, count, cause_len = _DECODED.unpack_from(
            raw, i * _DECODED.size)
        results.append((STATUS[status], {"sequence": seq, "fingerprint": fp,
                                         "frozen": bool(frozen), "n_features": nf,
                                         "n_outputs": no, "count": count}))
    return results


def cpp_pack_batch(pcheck: Path, tmp_path: Path, state, *, n_features, n_outputs,
                   records, dtype="f8"):
    """Pack many records sharing one state block. `records` is a sequence of
    (sequence, fingerprint, frozen, cause)."""
    dt = np.dtype(dtype)
    arr = np.ascontiguousarray(state, dtype=dt)
    job = tmp_path / "bpjob.bin"
    out = tmp_path / "bpout.bin"
    with open(job, "wb") as fh:
        fh.write(_JOB.pack(b"AMPC", 1, dt.itemsize, 4, n_features, n_outputs,
                           arr.size, 0, 0, 0, 0))
        fh.write(arr.tobytes())
        fh.write(struct.pack("<I", len(records)))
        for sequence, fingerprint, frozen, cause in records:
            fh.write(struct.pack("<IIII", sequence, fingerprint, 1 if frozen else 0,
                                 len(cause)))
            fh.write(cause)
    done = subprocess.run([str(pcheck), str(job), str(out)], capture_output=True,
                          text=True)
    assert done.returncode == 0, f"pcheck batch pack failed: {done.stderr}"
    raw = out.read_bytes()
    blobs, off = [], 0
    for _ in range(len(records)):
        (length,) = struct.unpack_from("<I", raw, off)
        off += 4
        blobs.append(raw[off:off + length])
        off += length
    assert off == len(raw), "trailing bytes in the batch result"
    return blobs


def trained(n=6, m=1, updates=40, dtype="f8", seed=5):
    a = RecursiveLeastSquares(n_features=n, n_outputs=m, forgetting=0.99, dtype=dtype)
    rng = np.random.default_rng(seed)
    for _ in range(updates):
        x = rng.standard_normal(n)
        a.update(x, rng.standard_normal(m))
    return a


# ---------------------------------------------------------------------------
# Byte identity, which is the whole point
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", ["f4", "f8"])
@pytest.mark.parametrize("n,m", [(6, 1), (6, 3), (12, 1), (2, 1)])
def test_the_two_implementations_produce_identical_bytes(pcheck, tmp_path, dtype, n, m):
    a = trained(n=n, m=m, dtype=dtype)
    py = pack(a, sequence=7, fingerprint=0xDEADBEEF)
    cpp = cpp_pack(pcheck, tmp_path, np.array(a.state), n_features=n, n_outputs=m,
                   sequence=7, fingerprint=0xDEADBEEF, dtype=dtype)

    assert len(py) == len(cpp), (
        f"records differ in length: python {len(py)}, c++ {len(cpp)}. A C++ struct laid "
        "over the buffer would pad the 26-byte header; this is what that looks like")
    if py != cpp:
        first = next(i for i, (x, y) in enumerate(zip(py, cpp)) if x != y)
        pytest.fail(f"first differing byte at offset {first}: python {py[first]:#04x}, "
                    f"c++ {cpp[first]:#04x}")


def test_the_frozen_flag_and_its_cause_are_identical_too(pcheck, tmp_path):
    """The conclusion the guard reached travels with the block, so it has to survive the
    format the same way the parameters do."""
    a = trained()
    cause = b"p_trace,divergence"

    class Frozen:
        """Minimal stand-in: `pack` reads `frozen`/`frozen_by` off whatever it is given."""
        def __init__(self, inner):
            self.inner = inner
            self.frozen = True
            self.frozen_by = ("p_trace", "divergence")
        n_features = property(lambda s: s.inner.n_features)
        n_outputs = property(lambda s: s.inner.n_outputs)
        dtype = property(lambda s: s.inner.dtype)
        state = property(lambda s: s.inner.state)

    py = pack(Frozen(a), sequence=3)
    cpp = cpp_pack(pcheck, tmp_path, np.array(a.state), n_features=6, n_outputs=1,
                   sequence=3, frozen=True, cause=cause)
    assert py == cpp


def test_an_empty_cause_is_encoded_the_same_as_no_cause(pcheck, tmp_path):
    a = trained()
    py = pack(a, sequence=1)
    cpp = cpp_pack(pcheck, tmp_path, np.array(a.state), n_features=6, n_outputs=1,
                   sequence=1, cause=b"")
    assert py == cpp
    assert cpp[17] == 0, "the cause-length byte should be zero"


@pytest.mark.parametrize("dtype", ["f4", "f8"])
def test_the_cpp_decodes_a_python_record(pcheck, tmp_path, dtype):
    a = trained(dtype=dtype)
    fp = fingerprint_of("rls", 6, 1, np.dtype(dtype).name, (("raw", 0),))
    blob = pack(a, sequence=42, fingerprint=fp)

    status, fields, cause, state = cpp_unpack(pcheck, tmp_path, blob, dtype=dtype)
    assert status == "ok"
    assert fields["sequence"] == 42
    assert fields["fingerprint"] == fp
    assert fields["n_features"] == 6 and fields["n_outputs"] == 1
    assert fields["count"] == a.state.size
    assert np.array_equal(state, np.array(a.state)), (
        "the payload decoded to different numbers")


def test_the_python_decodes_a_cpp_record(pcheck, tmp_path):
    a = trained()
    blob = cpp_pack(pcheck, tmp_path, np.array(a.state), n_features=6, n_outputs=1,
                    sequence=11, fingerprint=0x1234, frozen=True, cause=b"trace")
    rec = unpack(blob)
    assert rec.sequence == 11
    assert rec.fingerprint == 0x1234
    assert rec.frozen is True
    assert rec.frozen_by == ("trace",)
    assert np.array_equal(rec.state, np.array(a.state))


# ---------------------------------------------------------------------------
# The details byte identity actually depends on
# ---------------------------------------------------------------------------

def test_the_crc_is_zlibs_reflected_crc32(pcheck, tmp_path):
    """The CRC has to be the *reflected* IEEE 802.3 one. A plain CRC-32 over the same
    polynomial without reflection is a different function that looks equally reasonable in
    source, and would only ever be caught by a cross-check like this."""
    a = trained()
    blob = cpp_pack(pcheck, tmp_path, np.array(a.state), n_features=6, n_outputs=1,
                    sequence=2)
    body, stored = blob[:-4], struct.unpack("<I", blob[-4:])[0]
    assert stored == zlib.crc32(body) & 0xFFFFFFFF, (
        "the C++ CRC is not the one zlib computes, so the two will reject each other's "
        "records")


def test_the_header_is_twenty_six_bytes_with_unaligned_fields(pcheck, tmp_path):
    """The property a struct-based port would silently break. Python's `struct` packs
    without padding, so `sequence`, `fingerprint` and `count` sit at offsets 8, 18 and 22
    — and 18 and 22 are not four-byte aligned."""
    a = trained(n=2)
    blob = cpp_pack(pcheck, tmp_path, np.array(a.state), n_features=2, n_outputs=1,
                    sequence=0x11223344, fingerprint=0x55667788)
    assert blob[:4] == MAGIC
    assert struct.unpack_from("<H", blob, 4)[0] == FORMAT_VERSION
    assert struct.unpack_from("<I", blob, 8)[0] == 0x11223344
    assert struct.unpack_from("<I", blob, 18)[0] == 0x55667788
    assert struct.unpack_from("<I", blob, 22)[0] == a.state.size
    # Derived from the layout the reference actually uses, so this fails when the format
    # moves. Two earlier versions did not: the first compared two integer literals
    # (`18 % 4 != 0`), and the second rebuilt the field table inline in the test, which is
    # the same constants rearranged and passes whatever either implementation does.
    from argmind.persist import _HEADER

    offsets, at = {}, 0
    for name, fmt in zip(
            ("magic", "version", "flags", "sequence", "n_features", "n_outputs",
             "dtype", "cause_len", "fingerprint", "count"),
            ("4s", "H", "H", "I", "H", "H", "B", "B", "I", "I")):
        offsets[name] = at
        at += struct.calcsize("<" + fmt)
    assert _HEADER.format == "<4sHHIHHBBII", (
        f"the reference header layout changed to {_HEADER.format}; this test's field "
        "table no longer describes it")
    assert at == _HEADER.size == 26, f"the header is {_HEADER.size} bytes, not 26"
    assert offsets["fingerprint"] == 18 and offsets["count"] == 22
    assert offsets["fingerprint"] % 4 != 0 and offsets["count"] % 4 != 0, (
        "the header no longer places a uint32 unaligned, so nothing here needs "
        "byte-wise access and the header comment should change with it")
    assert offsets["sequence"] % 4 == 0, (
        "sequence is aligned; the comment in persist.hpp says so and would be wrong "
        "if this changed")
    assert len(blob) == 26 + a.state.size * 8 + 4


def test_the_element_bytes_are_little_endian_by_declaration(pcheck, tmp_path):
    """Not by inheriting the host's order. This cannot be distinguished by comparing
    output on an x86 host — both choices produce the same bytes — so what is checked is
    that a known value appears in the known order."""
    state = np.array([1.0, 2.0], dtype=np.float32)
    blob = cpp_pack(pcheck, tmp_path, state, n_features=1, n_outputs=1, dtype="f4")
    payload = blob[26:26 + 8]
    assert payload == struct.pack("<ff", 1.0, 2.0)
    assert payload != struct.pack(">ff", 1.0, 2.0)


# ---------------------------------------------------------------------------
# The failures a device actually produces, refused identically
# ---------------------------------------------------------------------------

def test_a_torn_write_is_refused_by_both(pcheck, tmp_path):
    a = trained()
    blob = pack(a, sequence=1)
    truncated = blob[:40]

    status, _, _, _ = cpp_unpack(pcheck, tmp_path, truncated)
    assert status == "length_mismatch", f"c++ said {status}"
    with pytest.raises(ValueError, match="cut short|too short"):
        unpack(truncated)


def test_a_flipped_bit_is_refused_by_both(pcheck, tmp_path):
    a = trained()
    blob = bytearray(pack(a, sequence=1))
    blob[40] ^= 0x01

    status, _, _, _ = cpp_unpack(pcheck, tmp_path, bytes(blob))
    assert status == "crc_mismatch", f"c++ said {status}"
    with pytest.raises(ValueError, match="CRC mismatch"):
        unpack(bytes(blob))


def test_a_future_version_is_refused_by_both(pcheck, tmp_path):
    a = trained()
    blob = bytearray(pack(a, sequence=1))
    struct.pack_into("<H", blob, 4, FORMAT_VERSION + 1)
    # The CRC has to be repaired or this tests truncation instead of versioning.
    struct.pack_into("<I", blob, len(blob) - 4,
                     zlib.crc32(bytes(blob[:-4])) & 0xFFFFFFFF)

    status, _, _, _ = cpp_unpack(pcheck, tmp_path, bytes(blob))
    assert status == "future_version", f"c++ said {status}"
    with pytest.raises(ValueError, match="newer than this build"):
        unpack(bytes(blob))


def test_something_that_is_not_a_record_is_refused_by_both(pcheck, tmp_path):
    junk = b"NOPE" + bytes(64)
    status, _, _, _ = cpp_unpack(pcheck, tmp_path, junk)
    assert status == "bad_magic"
    with pytest.raises(ValueError, match="bad magic"):
        unpack(junk)


def test_a_record_of_the_wrong_precision_is_refused(pcheck, tmp_path):
    """A block written at f8 and read back by an f4 build. Shape and CRC are both fine, so
    only the dtype code catches it — and continuing would reinterpret the bytes."""
    a = trained(dtype="f8")
    blob = pack(a, sequence=1)
    status, _, _, _ = cpp_unpack(pcheck, tmp_path, blob, dtype="f4")
    assert status == "bad_dtype", f"c++ said {status}"


# ---------------------------------------------------------------------------
# Findings from the sub-agent review, each pinned so it cannot come back
# ---------------------------------------------------------------------------

def test_a_cause_longer_than_the_field_is_clamped_identically(pcheck, tmp_path):
    """The cause length is one byte in the header, so both sides truncate at 255.

    This was unreachable through the harness before: `pcheck` read `cause_len` straight
    into a 256-byte stack buffer without clamping, which was a silent out-of-bounds write
    at 257 bytes and an access violation at 300. The library always clamped; the tool did
    not, so the truncation path could not be exercised at all.
    """
    a = trained(n=2)
    state = np.array(a.state)
    for length in (0, 1, 128, 254, 255, 256, 300, 1000):
        cause = b"x" * length
        cpp = cpp_pack(pcheck, tmp_path, state, n_features=2, n_outputs=1,
                       sequence=1, frozen=True, cause=cause)
        assert cpp[17] == min(length, 255), (
            f"cause_len {length} encoded as {cpp[17]}")
        rec = unpack(cpp)
        assert len(",".join(rec.frozen_by)) == min(length, 255)


def test_a_forged_count_that_would_wrap_on_a_32_bit_target_is_refused(pcheck, tmp_path):
    """`count` is read straight from the record and `size_t` is 32 bits on the target, so
    `count * sizeof(T)` can wrap and make a corrupt length look correct.

    On this 64-bit host the wrap does not happen, so the guard cannot be *triggered* here —
    what is checked is that the forged values are still refused, and that the guard did not
    break the ordinary path. The behaviour it protects is target-only, which is the point:
    a host cannot reach it, exactly as host timings cannot stand in for target timings.
    """
    a = trained(n=6)
    blob = bytearray(pack(a, sequence=1))
    true_count = a.state.size
    for forged in (0x40000000 + true_count, 0x80000000 + true_count, 0xFFFFFFFF):
        candidate = bytearray(blob)
        struct.pack_into("<I", candidate, 22, forged)
        struct.pack_into("<I", candidate, len(candidate) - 4,
                         zlib.crc32(bytes(candidate[:-4])) & 0xFFFFFFFF)
        status, _, _, _ = cpp_unpack(pcheck, tmp_path, bytes(candidate))
        assert status == "length_mismatch", (
            f"count {forged:#x} was accepted with status {status}")

    status, fields, _, _ = cpp_unpack(pcheck, tmp_path, bytes(blob))
    assert status == "ok" and fields["count"] == true_count, (
        "the overflow guard rejected a valid record")


_ARBITRATION = struct.Struct("<iiiIii")   # 24 bytes; see cpp/tools/pcheck.cpp


def cpp_select(pcheck: Path, tmp_path: Path, slot_a: bytes, slot_b: bytes,
               dtype="f8"):
    """Drive `argmind::persist::select_newest` directly.

    This exists because a mutation check found the function was compiled but never
    called from any binary the suite built. The arbitration test below it decoded each
    slot separately and compared against Python's own arbitration, which is a different
    code path — so mutants that made `select_newest` pick the lower sequence, or prefer
    an invalid slot, both survived.
    """
    dt = np.dtype(dtype)
    job = tmp_path / "sjob.bin"
    out = tmp_path / "sout.bin"
    with open(job, "wb") as fh:
        fh.write(_JOB.pack(b"AMPC", 1, dt.itemsize, 2, 0, 0, 0, 0, 0, 0, 0))
        fh.write(struct.pack("<I", len(slot_a)))
        fh.write(slot_a)
        fh.write(struct.pack("<I", len(slot_b)))
        fh.write(slot_b)
    done = subprocess.run([str(pcheck), str(job), str(out)], capture_output=True,
                          text=True)
    assert done.returncode == 0, f"pcheck select failed: {done.stderr}"
    winner, status_a, status_b, sequence, nxt, _pad = _ARBITRATION.unpack(
        out.read_bytes())
    return {"winner": winner, "status_a": STATUS[status_a],
            "status_b": STATUS[status_b], "sequence": sequence, "next": nxt}


def test_select_newest_picks_the_same_slot_the_reference_does(pcheck, tmp_path):
    """Which slot to believe is the decision the two-slot scheme exists to make, and it
    has to be the same decision on both sides or a device and a host disagree about which
    record is current."""
    a = trained()
    older = pack(a, sequence=4)
    newer = pack(a, sequence=9)
    torn = newer[:20]

    cases = [
        ("newer in slot 1", older, newer, 1, 9),
        ("newer in slot 0", newer, older, 0, 9),
        ("slot 1 torn", newer, torn, 0, 9),
        ("slot 0 torn", torn, newer, 1, 9),
        ("slot 1 missing", newer, b"", 0, 9),
        ("slot 0 missing", b"", newer, 1, 9),
        ("equal sequences", newer, newer, 0, 9),
    ]
    for label, sa, sb, want_winner, want_seq in cases:
        got = cpp_select(pcheck, tmp_path, sa, sb)
        assert got["winner"] == want_winner, f"{label}: c++ chose slot {got['winner']}"
        assert got["sequence"] == want_seq, f"{label}: sequence {got['sequence']}"
        assert got["next"] == 1 - want_winner, (
            f"{label}: the next write should target slot {1 - want_winner}, not "
            f"{got['next']}")

        # ...and the reference agrees, through its own arbitration.
        store = Store(tmp_path / label.replace(" ", "_"))
        store.slot_path(0).parent.mkdir(parents=True, exist_ok=True)
        for i, blob in ((0, sa), (1, sb)):
            if blob:
                store.slot_path(i).write_bytes(blob)
            elif store.slot_path(i).exists():
                store.slot_path(i).unlink()
        rec, _ = store.read()
        assert rec is not None and rec.sequence == want_seq, f"{label}: python"


def test_select_newest_refuses_when_neither_slot_validates(pcheck, tmp_path):
    """First boot, and total corruption, must be distinguishable from a valid record and
    from each other — the statuses come back rather than being collapsed into a bool."""
    a = trained()
    full = pack(a, sequence=1)
    # Truncation has two regimes and they report differently, which is worth pinning
    # rather than papering over: below a header plus a CRC there is nothing to inspect,
    # so it is `too_short`; above that the header parses and describes a longer record
    # than arrived, which is `length_mismatch`. A first version of this test expected the
    # latter for a 20-byte slot and was simply wrong about which regime that is.
    stub = full[:20]                      # < 26 + 4, so nothing can be read
    torn = full[:40]                      # header intact, body cut short

    empty = cpp_select(pcheck, tmp_path, b"", b"")
    assert empty["winner"] == -1
    assert empty["status_a"] == "too_short" and empty["status_b"] == "too_short"

    stubs = cpp_select(pcheck, tmp_path, stub, stub)
    assert stubs["winner"] == -1
    assert stubs["status_a"] == "too_short" and stubs["status_b"] == "too_short"

    both_torn = cpp_select(pcheck, tmp_path, torn, torn)
    assert both_torn["winner"] == -1
    assert both_torn["status_a"] == "length_mismatch"
    assert both_torn["status_b"] == "length_mismatch"

    junk = cpp_select(pcheck, tmp_path, b"NOPE" + bytes(60), torn)
    assert junk["winner"] == -1
    assert junk["status_a"] == "bad_magic"
    assert junk["status_b"] == "length_mismatch"


# ---------------------------------------------------------------------------
# The sweeps, checked in so the numbers are reproducible
# ---------------------------------------------------------------------------
#
# An audit caught these being quoted in DESIGN.md from an ephemeral sub-agent run against a
# harness that no longer existed — "measured across 1008 configurations and 80,000 fuzz
# blobs" while the suite actually did eight configurations, one bit flip and two
# truncations. The honest options were to relabel the numbers as reported-by-a-tool or to
# check the sweeps in. They are cheap, so they are checked in and the documentation now
# quotes what these produce.


def pack_configurations():
    """Every combination the byte-identity claim covers. Yields kwargs for cpp_pack."""
    for dtype in ("f4", "f8"):
        for n, m in ((6, 1), (6, 3), (12, 1), (2, 1)):
            for sequence in (0, 1, 7, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF, 0xDEADBEEF):
                for fingerprint in (0, 1, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF):
                    for cause in (b"", b"trace", b"p_trace,divergence"):
                        yield dtype, n, m, sequence, fingerprint, cause


def test_byte_identity_across_every_declared_configuration(pcheck, tmp_path):
    """840 configurations: two precisions, four shapes, seven sequence values including
    both ends and the high-bit-set range, five fingerprints, three cause lengths.
    (2 x 4 x 7 x 5 x 3 = 840; a first version of this docstring said 1680, which is what
    happens when a count is written down rather than computed.)

    High-bit values are in there because a sign-extension asymmetry between Python's
    arbitrary-precision ints and C++'s uint32_t would show up nowhere else.
    """
    class Frozen:
        def __init__(self, inner, by):
            self.inner, self.frozen = inner, bool(by)
            self.frozen_by = tuple(by.decode().split(",")) if by else ()
        n_features = property(lambda s: s.inner.n_features)
        n_outputs = property(lambda s: s.inner.n_outputs)
        dtype = property(lambda s: s.inner.dtype)
        state = property(lambda s: s.inner.state)

    grouped = {}
    for dtype, n, m, sequence, fingerprint, cause in pack_configurations():
        grouped.setdefault((dtype, n, m), []).append((sequence, fingerprint, cause))

    checked = 0
    for (dtype, n, m), records in grouped.items():
        a = trained(n=n, m=m, dtype=dtype)
        cpp_blobs = cpp_pack_batch(
            pcheck, tmp_path, np.array(a.state), n_features=n, n_outputs=m, dtype=dtype,
            records=[(seq, fp, bool(cause), cause) for seq, fp, cause in records])
        for (sequence, fingerprint, cause), cpp in zip(records, cpp_blobs):
            py = pack(Frozen(a, cause), sequence=sequence, fingerprint=fingerprint)
            if py != cpp:
                first = next(i for i, (x, y) in enumerate(zip(py, cpp)) if x != y)
                pytest.fail(f"{dtype} n={n} m={m} seq={sequence:#x} "
                            f"fp={fingerprint:#x} cause={cause!r}: first differing byte "
                            f"at {first}")
            checked += 1
    assert checked == 840, f"the sweep covered {checked} configurations, not 840"


def test_every_single_bit_flip_in_a_record_is_refused_identically(pcheck, tmp_path):
    """A record has 8 * len(blob) bits. Flip each one and require both sides to reach the
    same verdict — which also pins CHECK ORDERING: a flip in the version field must report
    `future_version` even though it has broken the CRC as well.
    """
    a = trained(n=2)                       # a 78-byte record: 624 bits
    blob = pack(a, sequence=1)
    assert len(blob) == 78, f"the record is {len(blob)} bytes, so the count below is wrong"

    candidates = []
    for bit in range(8 * len(blob)):
        c = bytearray(blob)
        c[bit // 8] ^= 1 << (bit % 8)
        candidates.append(bytes(c))
    cpp_results = cpp_unpack_batch(pcheck, tmp_path, candidates)

    seen = {}
    for bit, (candidate, (cpp_status, _)) in enumerate(zip(candidates, cpp_results)):
        try:
            unpack(bytes(candidate))
            py_status = "ok"
        except ValueError as exc:
            text = str(exc)
            py_status = ("bad_magic" if "magic" in text else
                         "future_version" if "newer than" in text else
                         "bad_dtype" if "dtype code" in text else
                         "length_mismatch" if "cut short" in text else
                         "crc_mismatch" if "CRC" in text else "other")
        assert py_status == cpp_status, (
            f"bit {bit} (byte {bit // 8}): python said {py_status}, c++ said {cpp_status}")
        seen[cpp_status] = seen.get(cpp_status, 0) + 1

    assert seen.get("future_version", 0) > 0, (
        "no flip reached the version field, so this does not pin check ordering")
    assert seen.get("bad_dtype", 0) > 0
    assert seen.get("crc_mismatch", 0) > 0
    assert sum(seen.values()) == 624


def test_every_truncation_length_is_refused_identically(pcheck, tmp_path):
    """Truncation has two regimes and both sides must agree on where the boundary is:
    below a header plus a CRC there is nothing to read, above it the header parses and
    describes more bytes than arrived."""
    a = trained(n=2)
    blob = pack(a, sequence=1)
    prefixes = [blob[:length] for length in range(len(blob))]
    cpp_results = cpp_unpack_batch(pcheck, tmp_path, prefixes)
    for length, (prefix, (cpp_status, _)) in enumerate(zip(prefixes, cpp_results)):
        try:
            unpack(prefix)
            py_status = "ok"
        except ValueError as exc:
            py_status = "too_short" if "too short" in str(exc) else "length_mismatch"
        assert py_status == cpp_status, (
            f"truncated to {length}: python {py_status}, c++ {cpp_status}")
    assert cpp_unpack(pcheck, tmp_path, blob)[0] == "ok", "the full record must still load"


def test_differential_fuzz_finds_no_disagreement(pcheck, tmp_path):
    """Random corruption of a valid record, on the theory that the shapes worth finding are
    the ones nobody thought to construct. Seeded, so a failure is reproducible."""
    a = trained(n=2)
    blob = pack(a, sequence=1)
    rng = np.random.default_rng(4242)
    ok_seen = 0

    candidates = []
    for i in range(400):
        c = bytearray(blob)
        if i % 4 == 0:
            # A quarter of the corpus stays VALID: perturb a header field and repair the
            # CRC. Without this every blob was refused and the field comparison below never
            # executed — measured on the seeded corpus, zero `ok` results in 400 — so the
            # test compared refusal reasons only and never a decoded value.
            struct.pack_into("<I", c, 8, int(rng.integers(0, 2**32)))     # sequence
            struct.pack_into("<I", c, 18, int(rng.integers(0, 2**32)))    # fingerprint
            struct.pack_into("<I", c, len(c) - 4, zlib.crc32(bytes(c[:-4])) & 0xFFFFFFFF)
        else:
            for _ in range(int(rng.integers(1, 5))):
                c[int(rng.integers(0, len(c)))] = int(rng.integers(0, 256))
            if rng.random() < 0.2:
                c = c[:int(rng.integers(0, len(c) + 1))]
        candidates.append(bytes(c))
    cpp_results = cpp_unpack_batch(pcheck, tmp_path, candidates)

    for trial, (candidate, (cpp_status, cpp_fields)) in enumerate(
            zip(candidates, cpp_results)):
        try:
            rec = unpack(bytes(candidate))
            py_status, py_seq, py_fp = "ok", rec.sequence, rec.fingerprint
        except ValueError as exc:
            text = str(exc)
            py_status = ("too_short" if "too short" in text else
                         "bad_magic" if "magic" in text else
                         "future_version" if "newer than" in text else
                         "bad_dtype" if "dtype code" in text else
                         "length_mismatch" if "cut short" in text else
                         "crc_mismatch" if "CRC" in text else "other")
            py_seq = py_fp = None
        assert py_status == cpp_status, (
            f"trial {trial}: python {py_status}, c++ {cpp_status}, "
            f"bytes {bytes(candidate).hex()[:80]}")
        if py_status == "ok":
            ok_seen += 1
            assert py_seq == cpp_fields["sequence"], f"trial {trial}: sequence differs"
            assert py_fp == cpp_fields["fingerprint"], f"trial {trial}: fingerprint differs"

    assert ok_seen >= 90, (
        f"only {ok_seen} of 400 fuzz blobs decoded successfully; the comparison of decoded "
        "values is then barely exercised, which is what this corpus was reshaped to fix")


def test_the_version_check_runs_before_the_crc_on_both_sides(pcheck, tmp_path):
    """Pinned directly, because a mutant that moved the C++ version check after the CRC
    passed the entire suite. Every test that touched versioning repaired the CRC first,
    which is exactly what removes the ordering from what they test.

    A record from a future version whose CRC is *also* wrong must still report the version,
    because that is the more useful diagnosis: a newer writer, not a corrupt slot.
    """
    a = trained(n=2)
    blob = bytearray(pack(a, sequence=1))
    struct.pack_into("<H", blob, 4, FORMAT_VERSION + 1)      # CRC deliberately NOT repaired

    cpp_status, _, _, _ = cpp_unpack(pcheck, tmp_path, bytes(blob))
    assert cpp_status == "future_version", (
        f"c++ reported {cpp_status}; the CRC check is running before the version check")
    with pytest.raises(ValueError, match="newer than this build"):
        unpack(bytes(blob))


def test_the_dtype_check_runs_before_the_length_check_on_both_sides(pcheck, tmp_path):
    """Same reasoning, one field along: a record of the wrong precision must be reported as
    such rather than as a length mismatch, even though the wrong element size also makes
    the length wrong."""
    a = trained(n=2, dtype="f8")
    blob = pack(a, sequence=1)
    cpp_status, _, _, _ = cpp_unpack(pcheck, tmp_path, blob, dtype="f4")
    assert cpp_status == "bad_dtype", f"c++ reported {cpp_status}"
