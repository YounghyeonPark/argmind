# SPDX-License-Identifier: Apache-2.0
"""Bounded sensor streams.

A `Stream` is a named source of timestamped samples with a fixed capacity. The
capacity is declared once and never grows: on an embedded target, a buffer whose
size depends on uptime is a bug rather than a tuning parameter, so this module
allocates at construction and never again.

Overwriting rather than blocking is deliberate. A sensor does not stop producing
because the consumer is late, and a framework that blocks the producer converts a
latency problem into a correctness problem. When the buffer is full the oldest
sample is dropped and the drop is *counted*, because silently discarding data is
the failure mode that makes multi-sensor systems impossible to debug.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Sample", "Stream"]


@dataclass(frozen=True, slots=True)
class Sample:
    """One observation from one stream.

    `t` is seconds on that stream's own timebase. Streams are not assumed to share
    a clock — reconciling them is the aligner's job, not the stream's.

    `valid` carries the sensor's own opinion of the reading: a GNSS fix with no
    satellites, a thermocouple that is open-circuit, a range finder returning its
    saturation value. Invalid samples are *kept*, not dropped, because "the sensor
    said it was bad" and "no sample arrived" are different situations and a
    consumer usually needs to tell them apart.
    """

    t: float
    values: np.ndarray
    valid: bool = True


class Stream:
    """A fixed-capacity ring buffer of samples from one source.

    Parameters
    ----------
    name : str
    dim : int
        Width of each sample vector.
    capacity : int
        Number of samples retained. Memory is ``capacity * dim`` elements, fixed.
    rate_hz : float | None
        *Nominal* rate, used only to reason about staleness and to let callers
        size buffers. Nothing enforces it; real sensors miss their nominal rate
        and the framework must not assume otherwise.
    dtype : str
        Storage precision. See the precision note in DESIGN.md — an algorithm's
        precision floor is a property of the algorithm, so adapters declare what
        they need and the pipeline refuses unsafe combinations at construction.
    """

    __slots__ = ("name", "dim", "capacity", "rate_hz", "dtype",
                 "_t", "_v", "_valid", "_head", "_count", "_dropped", "_last_t")

    def __init__(self, name: str, dim: int, capacity: int = 64,
                 rate_hz: float | None = None, dtype: str = "f4") -> None:
        if dim < 1:
            raise ValueError(f"{name}: dim must be >= 1, got {dim}")
        if capacity < 1:
            raise ValueError(f"{name}: capacity must be >= 1, got {capacity}")
        if rate_hz is not None and rate_hz <= 0:
            raise ValueError(f"{name}: rate_hz must be positive, got {rate_hz}")

        self.name = name
        self.dim = dim
        self.capacity = capacity
        self.rate_hz = rate_hz
        self.dtype = np.dtype(dtype)

        # Preallocated. Nothing here grows for the lifetime of the stream.
        self._t = np.zeros(capacity, dtype=np.float64)   # time stays fp64 always
        self._v = np.zeros((capacity, dim), dtype=self.dtype)
        self._valid = np.zeros(capacity, dtype=bool)
        self._head = 0          # next write position
        self._count = 0         # samples currently held, <= capacity
        self._dropped = 0       # overwritten because the consumer fell behind
        self._last_t = -np.inf

    # -- properties ---------------------------------------------------------

    def __len__(self) -> int:
        return self._count

    @property
    def dropped(self) -> int:
        """Samples overwritten before being consumed. Non-zero means the buffer
        is too small or the consumer too slow; either way the caller should know."""
        return self._dropped

    @property
    def latest_t(self) -> float | None:
        return self._last_t if self._count else None

    @property
    def nominal_period_s(self) -> float | None:
        return 1.0 / self.rate_hz if self.rate_hz else None

    # -- writing ------------------------------------------------------------

    def push(self, t: float, values, valid: bool = True) -> None:
        """Append a sample. Overwrites the oldest if full.

        Rejects non-monotonic timestamps. A sample arriving before one already
        stored means either a clock that went backwards or samples delivered out
        of order, and both break every alignment policy downstream. Failing here
        localises the bug to the source instead of producing a subtly wrong fused
        estimate somewhere far away.
        """
        if t <= self._last_t:
            raise ValueError(
                f"{self.name}: timestamps must increase, got t={t!r} after "
                f"{self._last_t!r}. Out-of-order or backward-stepping time breaks "
                "alignment; fix it at the source rather than downstream.")

        arr = np.asarray(values, dtype=self.dtype)
        if arr.shape != (self.dim,):
            raise ValueError(
                f"{self.name}: expected shape ({self.dim},), got {arr.shape}")

        if self._count == self.capacity:
            self._dropped += 1

        self._t[self._head] = t
        self._v[self._head] = arr
        self._valid[self._head] = valid
        self._head = (self._head + 1) % self.capacity
        self._count = min(self._count + 1, self.capacity)
        self._last_t = t

    # -- reading ------------------------------------------------------------

    def _index(self, i: int) -> int:
        """Map age index (0 = oldest held) to a physical slot."""
        oldest = (self._head - self._count) % self.capacity
        return (oldest + i) % self.capacity

    def at(self, i: int) -> Sample:
        """Sample by age: 0 is the oldest retained, len-1 the newest."""
        if not -self._count <= i < self._count:
            raise IndexError(f"{self.name}: index {i} out of range "
                             f"({self._count} samples held)")
        if i < 0:
            i += self._count
        s = self._index(i)
        return Sample(float(self._t[s]), self._v[s].copy(), bool(self._valid[s]))

    def latest(self) -> Sample | None:
        return self.at(self._count - 1) if self._count else None

    def times(self) -> np.ndarray:
        """Held timestamps, oldest first. A copy — callers must not alias the
        ring, or a later push would mutate what they are holding."""
        if not self._count:
            return np.empty(0, dtype=np.float64)
        idx = [self._index(i) for i in range(self._count)]
        return self._t[idx].copy()

    def newest_at_or_before(self, t: float) -> Sample | None:
        """The most recent sample not later than `t`, or None.

        This is the primitive behind hold-last alignment. Linear scan from the
        newest end: correct regardless of buffer wrap, and cheap because
        alignment queries are near the head in any real-time use.
        """
        for i in range(self._count - 1, -1, -1):
            s = self._index(i)
            if self._t[s] <= t:
                return Sample(float(self._t[s]), self._v[s].copy(),
                              bool(self._valid[s]))
        return None

    def nearest(self, t: float) -> Sample | None:
        """The sample closest in time to `t`, before or after."""
        if not self._count:
            return None
        best_i, best_d = 0, np.inf
        for i in range(self._count):
            d = abs(float(self._t[self._index(i)]) - t)
            if d < best_d:
                best_i, best_d = i, d
        return self.at(best_i)

    def bracketing(self, t: float) -> tuple[Sample, Sample] | None:
        """The pair (before, after) straddling `t`, for interpolation.

        Returns None when `t` falls outside the held range — deliberately, rather
        than extrapolating. Extrapolation past the end of a sensor stream is a
        guess, and a fusion framework that guesses without saying so is how a
        quiet wrong answer gets shipped.
        """
        prev = None
        for i in range(self._count):
            cur = self.at(i)
            if cur.t == t:
                return cur, cur
            if cur.t > t:
                return (prev, cur) if prev is not None else None
            prev = cur
        return None

    def __repr__(self) -> str:
        return (f"Stream({self.name!r}, dim={self.dim}, "
                f"held={self._count}/{self.capacity}, dropped={self._dropped})")
