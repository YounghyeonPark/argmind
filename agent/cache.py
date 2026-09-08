# SPDX-License-Identifier: Apache-2.0
"""The cache that makes repeated work faster: a fuzzy-keyed memo over the brain's decisions.

**The requirement, in the owner's words:** *"캐시를 만들어서 반복적인 작업은 점점 빠르게 동작할수
있게끔 할것."* So the loop asks the brain what to do only when it has not seen this situation before.
A hit costs a matrix product; a miss costs an LLM round trip, which is the thing this exists to skip.

**It is `ExemplarMemory`, because that is already a bounded fuzzy-keyed store** — content-addressed,
threshold-matched, one flat state block, and its batched look-up was measured nearly flat from 1,000
to 4,000 slots. What is added here is the *value* side and the honesty about eviction: the memory
hands back a slot, and when a full memory reuses one, whatever was stored against it is gone. That is
not a leak to paper over, it is what "bounded" means, and `values[slot]` being overwritten in the
same call is what keeps the two halves from drifting apart.

**The threshold defaults to 0.995 and the reason is asymmetry, not accuracy.** A miss costs a second
of latency. A **false hit moves the real mouse and types into the real window**, which is not
symmetric with that at all. The measurement behind the number: of screen pairs more than ten seconds
apart, 28.0 % match at 0.95 and 20.4 % at 0.995 — the distribution is two-humped, so tightening from
0.95 to 0.995 gives up about a quarter of the hits and buys most of the safety. See
`agent/situation.py` for the table.

**What makes "progressively faster" observable rather than asserted**: every entry counts its own
hits, and `stats()` reports hits, misses and the brain time skipped. A cache nobody can audit is a
cache nobody should let touch a mouse.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from argmind.memory import ExemplarMemory

__all__ = ["ActionCache", "Hit"]


class Hit:
    """A cache hit: what to do, which entry said so, and how sure it was."""

    __slots__ = ("value", "slot", "score", "uses")

    def __init__(self, value, slot: int, score: float, uses: int) -> None:
        self.value = value
        self.slot = int(slot)
        self.score = float(score)
        self.uses = int(uses)

    def __repr__(self) -> str:
        return f"Hit(slot={self.slot}, score={self.score:.4f}, uses={self.uses})"


class ActionCache:
    """Keys are situation descriptors; values are whatever the brain decided to do.

    The value is opaque here on purpose. This class must not know what an action is, or it becomes
    the place every new action type has to be taught about.
    """

    def __init__(self, n_features: int, *, capacity: int = 512, threshold: float = 0.995,
                 k: int = 16) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError(f"threshold is a cosine similarity in (0, 1], got {threshold}")
        self.memory = ExemplarMemory(capacity, int(n_features), int(k),
                                     threshold=float(threshold), dtype="f4")
        #: Indexed by slot, so eviction cannot leave a value pointing at somebody else's key.
        self.values: list = [None] * int(capacity)
        self.uses = np.zeros(int(capacity), dtype=np.int64)
        self.threshold = float(threshold)
        self._hits = 0
        self._misses = 0
        self._skipped_s = 0.0
        #: What each entry cost to learn, so a hit can be credited with what it skipped.
        self._cost_of: dict[int, float] = {}

    # ---- the two operations -------------------------------------------------------------------

    def lookup(self, key) -> Hit | None:
        """`None` means ask the brain. Never raises on an empty cache — the first frame is a miss."""
        if not self.memory.occupied:
            self._misses += 1
            return None
        report = self.memory.look_up(np.asarray(key, dtype=np.float64))
        if not report.matched or self.values[report.slot] is None:
            #: A matched slot with no value is possible and is not an error: the key survived an
            #: `invalidate`, which drops the value and leaves the key to be overwritten in turn.
            self._misses += 1
            return None
        self._hits += 1
        self.uses[report.slot] += 1
        return Hit(self.values[report.slot], report.slot, report.score, int(self.uses[report.slot]))

    def store(self, key, value, *, cost_s: float = 0.0) -> int:
        """Remember that this situation calls for this action, and what asking cost.

        `cost_s` is the brain time this entry will save on every later hit. Recorded rather than
        assumed, because "the cache made it faster" is a claim and this is the only evidence for it.
        """
        slot = int(self.memory.admit(np.asarray(key, dtype=np.float64)))
        self.values[slot] = value
        self.uses[slot] = 0
        self._cost_of[slot] = float(cost_s)
        return slot

    def invalidate(self, slot: int) -> None:
        """Forget what this entry said to do, keeping the key so the same mistake is a miss.

        Exists because **a cache that cannot be wrong is not a cache, it is a bet.** The loop calls
        this when a replayed action did not leave the world where the entry said it would.
        """
        if not 0 <= int(slot) < len(self.values):
            raise IndexError(f"slot {slot} is outside a cache of {len(self.values)}")
        self.values[int(slot)] = None

    # ---- evidence -----------------------------------------------------------------------------

    def note_hit_saved(self, slot: int) -> None:
        """Credit a hit with the brain time it skipped, using what that entry cost to learn."""
        self._skipped_s += self._cost_of.get(int(slot), 0.0)

    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": (self._hits / total) if total else 0.0,
            "entries": int(self.memory.occupied),
            "capacity": int(self.memory.capacity),
            "brain_seconds_skipped": round(self._skipped_s, 3),
        }

    # ---- across restarts ----------------------------------------------------------------------

    def save(self, path) -> None:
        """Keys as the memory's own flat block, values as JSON beside it.

        Two files rather than one because they are two kinds of thing: the block is a float array
        with a fixed layout, and the values are whatever the brain returned. Pretending otherwise
        would mean serialising the block into JSON and losing exactness for tidiness.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path.with_suffix(".keys.npy"), np.asarray(self.memory.state))
        path.with_suffix(".values.json").write_text(json.dumps({
            "threshold": self.threshold,
            "capacity": int(self.memory.capacity),
            "n_features": int(self.memory.n_features),
            "uses": self.uses.tolist(),
            "cost_of": {str(k): v for k, v in self._cost_of.items()},
            "values": self.values,
        }), encoding="utf-8")

    def load(self, path) -> None:
        path = Path(path)
        meta = json.loads(path.with_suffix(".values.json").read_text(encoding="utf-8"))
        if meta["capacity"] != self.memory.capacity or meta["n_features"] != self.memory.n_features:
            raise ValueError(
                f"saved cache is {meta['capacity']}x{meta['n_features']}, this one is "
                f"{self.memory.capacity}x{self.memory.n_features}; a key means nothing across a "
                "different descriptor, so this is refused rather than reshaped")
        self.memory.load_state(np.load(path.with_suffix(".keys.npy")))
        self.values = list(meta["values"])
        self.uses = np.asarray(meta["uses"], dtype=np.int64)
        self._cost_of = {int(k): float(v) for k, v in meta["cost_of"].items()}
