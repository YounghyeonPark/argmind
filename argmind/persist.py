# SPDX-License-Identifier: Apache-2.0
"""Persisting adaptation state across a power cycle.

Adaptation state that does not survive a power cycle is useless in the field: a
device that spends an hour learning its own biases and then forgets them every
time someone cycles the ignition has not been calibrated, it has been kept busy.

The format was already decided by the `Adapter` contract — the state is one flat
contiguous block. What this module adds is everything around that block, and each
piece is here because of a specific way storing state goes wrong on a device.

**A write can be interrupted.** Power fails mid-erase and the sector is left as
neither the old value nor the new one. So there are two slots, written
alternately, each carrying its own sequence number and CRC. Loading takes the
highest-sequence slot that validates. An interrupted write can destroy at most the
slot being written, which is never the one currently in use. This is the standard
scheme for raw flash and it is modelled faithfully here even though the reference
writes files: the C side will not have `os.replace`, so a design that leaned on
rename would not survive the port.

**A block can be restored into the wrong thing.** Shape and precision are checked,
which catches the crude cases. The dangerous case is subtler: `FrameMap` order is
parameter order, so a block restored after someone reordered two features is
numerically valid, silently wrong, and undetectable downstream. `fingerprint` is
the defence — a stable hash of the configuration that produced the state, refused
on mismatch. Use `fingerprint_of` rather than Python's `hash()`, which is salted
per process and would differ every run.

**Flash wears out.** A sector rated for 100k erase cycles, written on every update
at 100 Hz, is gone in about seventeen minutes. Cadence is therefore a declared
policy rather than a decision left to the caller's discipline: `Store.maybe_save`
takes `every_updates` and `every_seconds` and does nothing until one is due.

**Endianness and width are declared, not inherited.** Everything is little-endian
with explicit widths, because the writer and the reader may be different machines
and "whatever this compiler does" is not a format.

What is not persisted, deliberately
-----------------------------------
The guard's checkpoints and the EWMAs inside its checks. Those describe a run, and
a run ends at power-off; restoring them would assert a continuity that does not
exist.

The frozen flag *is* carried, because it is a conclusion rather than working
state, together with what tripped. What to do with it on load is a real decision
with no obviously correct default, so it is the caller's: `on_frozen="thaw"`
comes back adapting, `"keep"` comes back frozen, `"refuse"` raises. The default is
`"thaw"` and the argument against it is worth knowing — a device that froze for
cause will re-learn its way back into the same trouble. The argument for it is
that a persisted freeze can strand a device that a power cycle would have fixed,
with no way back except a site visit. Choose per deployment; the record tells you
which happened either way.
"""

from __future__ import annotations

import os
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .adapt import Adapter

__all__ = ["Record", "Store", "pack", "unpack", "fingerprint_of",
           "FORMAT_VERSION", "MAGIC"]

MAGIC = b"ARGM"
FORMAT_VERSION = 1

#: ``<`` is not decoration. The writer and the reader may be different machines,
#: and a format that inherits the host's byte order is not a format.
_HEADER = struct.Struct("<4sHHIHHBBII")
_CRC = struct.Struct("<I")

_DTYPE_CODE = {np.dtype("f4"): 4, np.dtype("f8"): 8}
_CODE_DTYPE = {v: k for k, v in _DTYPE_CODE.items()}

_FLAG_FROZEN = 1 << 0


@dataclass(frozen=True, slots=True)
class Record:
    """One decoded slot.

    `frozen` and `frozen_by` describe the guard that wrote it, and are carried
    even when loading into a bare adapter — a caller that dropped the guard still
    deserves to know the state it is restoring was frozen when it was written.
    """

    version: int
    sequence: int
    n_features: int
    n_outputs: int
    dtype: np.dtype
    fingerprint: int
    frozen: bool
    frozen_by: tuple[str, ...]
    state: np.ndarray

    @property
    def state_size(self) -> int:
        return int(self.state.size)


def fingerprint_of(*parts) -> int:
    """A stable 32-bit hash of whatever identifies a configuration.

    Stable is the whole point: Python's `hash()` is salted per process, so a
    fingerprint built from it would mismatch on the next run and reject every
    valid state block. This is `crc32` over a canonical text rendering, which is
    the same in every process and on every machine.

    Pass whatever must match for a restored block to be meaningful. For a
    pipeline that means the resolved frame map, since feature order is parameter
    order:

        fingerprint_of("rls", 3, 1, "float64", pipe.frame_map.slots)
    """
    text = "|".join(_canonical(p) for p in parts)
    return zlib.crc32(text.encode("utf-8")) & 0xFFFFFFFF


def _canonical(p) -> str:
    if isinstance(p, (list, tuple)):
        return "(" + ",".join(_canonical(x) for x in p) + ")"
    if isinstance(p, np.dtype):
        return p.name
    return str(p)


def pack(adapter: Adapter, sequence: int = 0, fingerprint: int = 0) -> bytes:
    """Serialise an adapter's state into one self-validating slot.

    Reads `frozen`/`frozen_by` off the adapter if it has them, so passing a
    `Guard` records that it was frozen without the caller having to remember.
    """
    dt = np.dtype(adapter.dtype)
    if dt not in _DTYPE_CODE:
        raise ValueError(f"cannot persist dtype {dt.name!r}; supported: "
                         f"{[d.name for d in _DTYPE_CODE]}")

    frozen = bool(getattr(adapter, "frozen", False))
    by = ",".join(getattr(adapter, "frozen_by", ()) or ()).encode("utf-8")
    if len(by) > 255:
        by = by[:255]

    state = np.ascontiguousarray(adapter.state, dtype=dt.newbyteorder("<"))
    header = _HEADER.pack(
        MAGIC, FORMAT_VERSION, _FLAG_FROZEN if frozen else 0,
        int(sequence) & 0xFFFFFFFF, adapter.n_features, adapter.n_outputs,
        _DTYPE_CODE[dt], len(by), int(fingerprint) & 0xFFFFFFFF, state.size)
    body = header + by + state.tobytes()
    return body + _CRC.pack(zlib.crc32(body) & 0xFFFFFFFF)


def unpack(blob: bytes) -> Record:
    """Decode one slot, or raise `ValueError` describing why it cannot be.

    Every failure here is one a device actually produces: a slot that was never
    written, a write cut short by power loss, a bit flipped in storage, or a
    block written by a future version of this library.
    """
    if len(blob) < _HEADER.size + _CRC.size:
        raise ValueError(f"slot is {len(blob)} bytes, too short to be a record")
    (magic, version, flags, sequence, n_features, n_outputs, code, by_len,
     fingerprint, count) = _HEADER.unpack_from(blob, 0)
    if magic != MAGIC:
        raise ValueError(f"bad magic {magic!r}; this is not an argmind record")
    if version > FORMAT_VERSION:
        raise ValueError(
            f"record format version {version} is newer than this build "
            f"understands ({FORMAT_VERSION}); refusing to guess at its layout")
    if code not in _CODE_DTYPE:
        raise ValueError(f"unknown dtype code {code}")

    dt = _CODE_DTYPE[code]
    expect = _HEADER.size + by_len + count * dt.itemsize + _CRC.size
    if len(blob) != expect:
        raise ValueError(
            f"slot is {len(blob)} bytes, the header describes {expect}; the "
            "write was cut short or the slot is padded with something else")

    stored = _CRC.unpack_from(blob, expect - _CRC.size)[0]
    actual = zlib.crc32(blob[:expect - _CRC.size]) & 0xFFFFFFFF
    if stored != actual:
        raise ValueError(f"CRC mismatch: stored {stored:#010x}, computed "
                         f"{actual:#010x}; the slot is corrupt")

    by = blob[_HEADER.size:_HEADER.size + by_len].decode("utf-8", "replace")
    start = _HEADER.size + by_len
    state = np.frombuffer(blob, dtype=dt.newbyteorder("<"), count=count,
                          offset=start).astype(dt, copy=True)
    return Record(version=version, sequence=sequence, n_features=n_features,
                  n_outputs=n_outputs, dtype=dt, fingerprint=fingerprint,
                  frozen=bool(flags & _FLAG_FROZEN),
                  frozen_by=tuple(s for s in by.split(",") if s), state=state)


class Store:
    """Two alternating slots on disk, each self-validating.

    Parameters
    ----------
    path : str | Path
        Base path. Slots are ``<path>.0`` and ``<path>.1``.
    every_updates, every_seconds : int | float | None
        Cadence for `maybe_save`. At least one must be set, or `maybe_save` would
        either write on every call or never — both of which are better said out
        loud. `save` ignores cadence entirely and is the right call for shutdown.
    """

    __slots__ = ("path", "every_updates", "every_seconds", "_sequence",
                 "_slot", "_last_updates", "_last_t", "_writes", "_arbitrated")

    def __init__(self, path, every_updates: int | None = None,
                 every_seconds: float | None = None) -> None:
        if every_updates is not None and every_updates < 1:
            raise ValueError(f"every_updates must be >= 1, got {every_updates}")
        if every_seconds is not None and every_seconds <= 0:
            raise ValueError(f"every_seconds must be positive, got {every_seconds}")
        self.path = Path(path)
        self.every_updates = every_updates
        self.every_seconds = every_seconds
        self._sequence = 0
        self._slot = 0
        self._last_updates: int | None = None
        self._last_t: float | None = None
        self._writes = 0
        self._arbitrated = False

    def slot_path(self, i: int) -> Path:
        return self.path.with_name(self.path.name + f".{i}")

    @property
    def writes(self) -> int:
        """Physical writes performed. The number to watch against a flash
        endurance budget, and the one `maybe_save` exists to keep small."""
        return self._writes

    @property
    def sequence(self) -> int:
        return self._sequence

    # -- writing -------------------------------------------------------------

    def _arbitrate_once(self) -> None:
        """Find out which slot is live, if this instance has not already.

        A `Store` that writes before it reads knows nothing about what is on disk: it
        starts at slot 0 with sequence 0. Measured before this existed — three saves leave
        {slot 0: 3, slot 1: 2}, and a fresh Store calling `save()` without `load()` leaves
        {slot 0: 1, slot 1: 2}. The newest record is destroyed *and* the sequence counter
        regresses, so from then on the older record wins arbitration and the device
        silently reverts a generation.

        `load()` fixing its own path was not enough, because writing without loading is a
        perfectly ordinary thing for a caller to do.
        """
        if self._arbitrated:
            return
        self._arbitrated = True
        rec, from_slot, _ = self._newest()
        if rec is not None:
            self._sequence = max(self._sequence, rec.sequence)
            self._slot = 1 - from_slot

    def save(self, adapter: Adapter, fingerprint: int = 0) -> int:
        """Write to the slot not holding the newest record, and return the sequence
        number written.

        On the first call this consults the slots if nothing has been loaded, so the
        promise above holds for a `Store` that writes before it reads.
        """
        self._arbitrate_once()
        self._sequence += 1
        blob = pack(adapter, sequence=self._sequence, fingerprint=fingerprint)
        target = self.slot_path(self._slot)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "wb") as fh:
            fh.write(blob)
            fh.flush()
            os.fsync(fh.fileno())
        self._slot ^= 1
        self._writes += 1
        return self._sequence

    def maybe_save(self, adapter: Adapter, updates: int | None = None,
                   t: float | None = None, fingerprint: int = 0) -> bool:
        """Write only if the declared cadence says it is due. Returns whether it
        wrote.

        `updates` is a monotonic count (`Pipeline.stats["updated"]` fits) and `t`
        is a clock in seconds. Whichever is configured and has elapsed triggers a
        write; the first call with a configured cadence always writes, so there is
        a valid record from the start rather than only after the first interval.
        """
        if self.every_updates is None and self.every_seconds is None:
            raise ValueError(
                "this Store has no cadence, so maybe_save cannot decide anything. "
                "Set every_updates or every_seconds, or call save() explicitly.")
        due = False
        if self.every_updates is not None:
            if updates is None:
                raise ValueError("every_updates is set but no updates count given")
            due |= (self._last_updates is None
                    or updates - self._last_updates >= self.every_updates)
        if self.every_seconds is not None:
            if t is None:
                raise ValueError("every_seconds is set but no time given")
            due |= (self._last_t is None
                    or t - self._last_t >= self.every_seconds)
        if not due:
            return False
        self.save(adapter, fingerprint=fingerprint)
        self._last_updates = updates
        self._last_t = t
        return True

    # -- reading -------------------------------------------------------------

    def _newest(self) -> tuple[Record | None, int, list[str]]:
        """The newest valid record, the slot it came from, and why the others failed."""
        best: Record | None = None
        best_slot = -1
        problems: list[str] = []
        for i in (0, 1):
            p = self.slot_path(i)
            try:
                blob = p.read_bytes()
            except FileNotFoundError:
                problems.append(f"slot {i}: not written")
                continue
            try:
                rec = unpack(blob)
            except ValueError as exc:
                problems.append(f"slot {i}: {exc}")
                continue
            if best is None or rec.sequence > best.sequence:
                best, best_slot = rec, i
        return best, best_slot, problems

    def read(self) -> tuple[Record | None, list[str]]:
        """The newest valid record and why the rejected slots were rejected.

        The reasons come back rather than being logged and forgotten: "it started
        from scratch" and "it started from scratch because slot 1 had a CRC
        mismatch" are very different mornings.
        """
        best, _slot, problems = self._newest()
        return best, problems

    @property
    def live_slot(self) -> int:
        """The slot this `Store` is treating as live, or -1 if it has not looked yet.

        Deliberately does not go and read the slots: a fresh `Store` over a path full of
        valid records reports -1 until it loads or saves, because until then it genuinely
        does not know. An earlier docstring said "the slot holding the record currently in
        use", which claimed knowledge this property does not have — and a first attempt at
        fixing it changed the condition instead of the sentence, which then reported a live
        slot after a load that found nothing.
        """
        return 1 - self._slot if self._sequence else -1

    def load(self, adapter: Adapter, fingerprint: int = 0,
             on_frozen: str = "thaw") -> Record | None:
        """Restore the newest valid record into `adapter`.

        Returns the record, or None if no slot validated — which is a legitimate
        first-boot outcome, so it is a return value rather than an exception. A
        record that validates but does not *fit* is an exception, because that is
        a configuration error and continuing from a mismatched block would be
        worse than not starting.
        """
        if on_frozen not in ("thaw", "keep", "refuse"):
            raise ValueError(f"on_frozen must be thaw/keep/refuse, got "
                             f"{on_frozen!r}")
        self._arbitrated = True
        rec, from_slot, _ = self._newest()
        if rec is None:
            return None

        if (rec.n_features, rec.n_outputs) != (adapter.n_features,
                                               adapter.n_outputs):
            raise ValueError(
                f"record is for {rec.n_features}x{rec.n_outputs} but this adapter "
                f"is {adapter.n_features}x{adapter.n_outputs}")
        if rec.dtype != np.dtype(adapter.dtype):
            raise ValueError(f"record is {rec.dtype.name}, adapter is "
                             f"{np.dtype(adapter.dtype).name}")
        if fingerprint and rec.fingerprint != fingerprint:
            raise ValueError(
                f"configuration fingerprint {rec.fingerprint:#010x} does not match "
                f"this pipeline's {fingerprint:#010x}. The block would load and be "
                "silently wrong — feature order is parameter order.")
        if rec.frozen and on_frozen == "refuse":
            raise ValueError(
                f"the stored state was frozen by {list(rec.frozen_by)} and "
                "on_frozen='refuse'")

        adapter.load_state(rec.state)
        self._sequence = max(self._sequence, rec.sequence)
        # Write to the OTHER slot next. Without this the first write after a reboot lands
        # on the record just loaded, because __init__ starts at slot 0 and nothing moved
        # it — so after any odd number of writes in the previous session, the newest
        # record is the one destroyed. That is precisely the failure two slots exist to
        # prevent: an interrupted write must never be aimed at the copy in use.
        self._slot = 1 - from_slot
        if rec.frozen and on_frozen == "keep":
            freeze = getattr(adapter, "freeze", None)
            if freeze is None:
                raise ValueError(
                    "the stored state was frozen and on_frozen='keep', but this "
                    "adapter has no freeze path to restore it into")
            freeze(rec.frozen_by[0] if rec.frozen_by else "restored")
        return rec

    def __repr__(self) -> str:
        return (f"Store({str(self.path)!r}, every_updates={self.every_updates}, "
                f"every_seconds={self.every_seconds}, writes={self._writes})")
