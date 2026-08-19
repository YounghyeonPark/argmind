# SPDX-License-Identifier: Apache-2.0
"""`ExemplarMemory` — a bounded store of sparse descriptors, with one flat state block.

Measured on 2026-08-19, recognition of a previously-seen object needs **no representation
learning**: a crop embedding from an untrained backbone decodes shape at 0.996 across positions it
has never seen, a stored exemplar steers attention to the remembered object at 1.000, and sixteen
of 108 dimensions carry it at the dense score. What was missing was somewhere to *put* the
exemplars that this library could carry.

**Bounded, because the target tier has no heap.** Capacity is fixed at construction and a full
memory admits by replacing, never by growing. That is the difference between a component and a
measurement: the demo's `ObjectMemory` grows a Python list, which is fine on a host and expressible
nowhere else.

**One flat contiguous block**, so `persist` is a copy and the C ABI needs no serialisation — the
same reason every adapter here is laid out that way. Indices live in that block as floats, which is
exact: f4 represents integers below 2^24 without loss and a descriptor index is at most a few
thousand. Storing them in a second integer array would make the state two blocks and break the one
property the persistence path depends on.

**What it is not.** It is not an `Adapter`: `update(x, y)` presumes a target and this has none — an
exemplar is admitted, not fitted. Forcing it into that contract would put a `y` in the signature
that no caller could supply, and this project refuses rather than half-builds. What it does supply
is `state`/`load_state`/`state_size`, which is what `Store` actually requires, so it persists
alongside adapters without pretending to be one.
"""

from __future__ import annotations

import numpy as np

__all__ = ["ExemplarMemory", "MatchReport"]


class MatchReport:
    """What a lookup found. A value object so a caller need not unpack a tuple in the right order."""

    __slots__ = ("slot", "score", "matched")

    def __init__(self, slot: int, score: float, matched: bool) -> None:
        self.slot = slot
        self.score = score
        self.matched = matched

    def __repr__(self) -> str:
        return f"MatchReport(slot={self.slot}, score={self.score:.4f}, matched={self.matched})"


class ExemplarMemory:
    """`capacity` sparse descriptors of `n_features` dimensions, `k` non-zeros each.

    `threshold` is the similarity below which `look_up` reports no match. It is a caller decision
    and has no safe default that generalises, so it is required rather than guessed.
    """

    #: index, value, and one "age" scalar per slot -- the age is what makes replacement defined
    #: without a heap-allocated queue.
    def __init__(self, capacity: int, n_features: int, k: int, *, threshold: float,
                 dtype: str = "f4") -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be at least 1, got {capacity}")
        if not 1 <= k <= n_features:
            raise ValueError(f"k must be in 1..{n_features}, got {k}")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold is a cosine similarity, so 0..1; got {threshold}")
        self.capacity = int(capacity)
        self.n_features = int(n_features)
        self.k = int(k)
        self.threshold = float(threshold)
        self.dtype = np.dtype(dtype)

        #: One similarity per slot, which is what this would emit if it were read as a layer. It
        #: is declared because `persist.pack` requires it, and it is the honest value rather than a
        #: placeholder: it makes the record's shape check meaningful, so an 8-slot memory refuses to
        #: load into a 16-slot one.
        self.n_outputs = self.capacity

        #: [ indices (capacity*k) | values (capacity*k) | ages (capacity) ]
        self.state_size = 2 * self.capacity * self.k + self.capacity
        self._block = np.zeros(self.state_size, dtype=self.dtype)
        self._idx = self._block[:self.capacity * self.k].reshape(self.capacity, self.k)
        self._val = self._block[self.capacity * self.k:2 * self.capacity * self.k] \
            .reshape(self.capacity, self.k)
        self._age = self._block[2 * self.capacity * self.k:]
        self._idx[:] = -1.0                      # -1 marks an empty slot
        self._clock = 0

    # ---- state -------------------------------------------------------------------------------

    @property
    def state(self) -> np.ndarray:
        """The live block, not a snapshot — the views above write straight into it."""
        return self._block

    def load_state(self, flat) -> None:
        arr = np.asarray(flat, dtype=self.dtype).ravel()
        if arr.size != self.state_size:
            raise ValueError(f"state is {self.state_size} values, got {arr.size}")
        self._block[:] = arr

    def reset(self) -> None:
        self._block[:] = 0.0
        self._idx[:] = -1.0
        self._clock = 0

    @property
    def occupied(self) -> int:
        return int((self._idx[:, 0] >= 0).sum())

    # ---- the two operations ------------------------------------------------------------------

    def _sparse(self, descriptor):
        """Top-`k` by magnitude, unit-normalised. Sparsifying here rather than at the call site is
        what makes a stored slot and a query commensurable."""
        v = np.asarray(descriptor, dtype=np.float64).ravel()
        if v.size != self.n_features:
            raise ValueError(f"expected {self.n_features} features, got {v.size}")
        idx = np.argpartition(np.abs(v), -self.k)[-self.k:] if self.k < v.size \
            else np.arange(v.size)
        vals = v[idx]
        n = float(np.linalg.norm(vals))
        return idx, (vals / n if n > 1e-12 else vals)

    def look_up(self, descriptor) -> MatchReport:
        """Nearest stored exemplar by cosine similarity on the shared support.

        Empty slots are skipped rather than scoring 0, because a zero slot is not "very different";
        it is absent, and letting it compete would make an empty memory report a best match.
        """
        idx, vals = self._sparse(descriptor)
        best, best_slot = -1.0, -1
        query = np.zeros(self.n_features, dtype=np.float64)
        query[idx] = vals
        for s in range(self.capacity):
            if self._idx[s, 0] < 0:
                continue
            si = self._idx[s].astype(np.int64)
            score = float(np.dot(query[si], self._val[s].astype(np.float64)))
            if score > best:
                best, best_slot = score, s
        if best_slot < 0:
            return MatchReport(-1, 0.0, False)
        matched = best >= self.threshold
        if matched:
            # **Only a real match refreshes the age.** Refreshing the nearest slot regardless was a
            # defect the replacement test caught: a near miss then counts as a recognition, every
            # slot's age converges, and the tie-break evicts whichever came first -- which is the
            # exemplar most recently *recognised*, exactly the one to keep.
            self._age[best_slot] = self._clock
        return MatchReport(best_slot, best, matched)

    def admit(self, descriptor) -> int:
        """Store a descriptor and return its slot. A full memory replaces the least recently
        matched one — bounded by construction, so this never allocates."""
        self._clock += 1
        idx, vals = self._sparse(descriptor)
        empty = np.flatnonzero(self._idx[:, 0] < 0)
        slot = int(empty[0]) if empty.size else int(np.argmin(self._age))
        self._idx[slot] = idx.astype(self.dtype)
        self._val[slot] = np.asarray(vals, dtype=self.dtype)
        self._age[slot] = self._clock
        return slot

    def observe(self, descriptor) -> MatchReport:
        """Look up, and admit if nothing matched. The recognition loop in one call."""
        report = self.look_up(descriptor)
        if not report.matched:
            return MatchReport(self.admit(descriptor), report.score, False)
        return report
