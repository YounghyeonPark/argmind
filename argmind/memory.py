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

        #: [ indices (c*k) | values (c*k) | ages (c) | spells (c) | here (c) ]
        #:
        #: **`spells` is the retention criterion**, and it exists because the obvious one is worse
        #: than useless. Counting *frames* a slot was seen in makes the wall the best-remembered
        #: thing in the room: static scenery appears in every frame, perfectly, while a person who
        #: steps out loses ground. A **spell** is one unbroken run of presence, so a bookshelf that
        #: never moves accumulates a single long spell and a person who comes and goes accumulates
        #: many. That is what separates the two, and it is the phenomenon the owner asked about:
        #: appear, be absent, appear again.
        #:
        #: `here` is the previous frame's presence, which is all that is needed to spot the
        #: absent-to-present transition without storing a history.
        c, ck = self.capacity, self.capacity * self.k
        self.state_size = 2 * ck + 3 * c
        self._block = np.zeros(self.state_size, dtype=self.dtype)
        self._idx = self._block[:ck].reshape(c, self.k)
        self._val = self._block[ck:2 * ck].reshape(c, self.k)
        self._age = self._block[2 * ck:2 * ck + c]
        self._spells = self._block[2 * ck + c:2 * ck + 2 * c]
        self._here = self._block[2 * ck + 2 * c:]
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

    @property
    def spells(self) -> np.ndarray:
        """Separate appearances per slot. A view, so it follows `load_state`."""
        return self._spells

    def end_frame(self, present) -> None:
        """Close a frame: `present` is the slots recognised in it.

        Called by the caller rather than inferred, because a *frame* is not something this class can
        see — `look_up` runs many times within one. A slot that was absent and is now present starts
        a new spell; one that stays present continues the spell it is in.
        """
        #: **The clock advances here too.** It used to move only on `admit`, so between two
        #: admissions every age was identical and "least recently matched" meant nothing — a mutant
        #: reverting eviction to `argmin(age)` survived because age could not distinguish anything.
        #: Found by building a fixture that needed the two rules to disagree.
        self._clock += 1
        seen = np.zeros(self.capacity, dtype=bool)
        for s in present:
            if 0 <= int(s) < self.capacity:
                seen[int(s)] = True
        self._spells[seen & (self._here < 0.5)] += 1.0
        self._here[:] = seen.astype(self.dtype)
        self._age[seen] = self._clock

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
        if empty.size:
            slot = int(empty[0])
        else:
            #: **Evict the fewest spells, not the oldest.** Age alone throws out whichever exemplar
            #: has waited longest, which on a real scene is an object that has stepped out — exactly
            #: the one the memory exists to hold. Spells rank scenery below objects: a bookshelf has
            #: one long spell however many frames it filled. Age remains the tie-break.
            fewest = np.flatnonzero(self._spells == self._spells.min())
            slot = int(fewest[np.argmin(self._age[fewest])])
        self._idx[slot] = idx.astype(self.dtype)
        self._val[slot] = np.asarray(vals, dtype=self.dtype)
        self._age[slot] = self._clock
        #: One spell in progress, not zero: it is present now. Zero would make every new admission
        #: the next thing evicted.
        self._spells[slot] = 1.0
        self._here[slot] = 1.0
        return slot

    def observe(self, descriptor) -> MatchReport:
        """Look up, and admit if nothing matched. The recognition loop in one call."""
        report = self.look_up(descriptor)
        if not report.matched:
            return MatchReport(self.admit(descriptor), report.score, False)
        return report
