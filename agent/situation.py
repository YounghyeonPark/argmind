# SPDX-License-Identifier: Apache-2.0
"""What "the same situation, again" means for a screen — the cache's key.

**Measured before it was chosen.** 117 real screen captures over 60 s, three candidate keys, and the
question a cache actually asks: does a situation seen a while ago come back looking the same, while
an unrelated one does not?

| key | consecutive frames | > 10 s apart | gap |
|---|---|---|---|
| **backbone, 144-d** | 0.9882 ± 0.0620 | 0.7579 ± 0.1501 | +0.2303 |
| 32×32 grey pixels, 1024-d | 0.9917 ± 0.0411 | 0.7290 ± 0.1665 | +0.2627 |
| 64-bit dHash | 0.9604 ± 0.1479 | 0.1817 ± 0.4407 | +0.7787 |

The means are not the finding — the **tail** is. Of pairs more than ten seconds apart, **20.4 % score
at or above 0.995**: screens genuinely recur, pixel for pixel. And the share barely moves from 0.95
to 0.995 (28.0 → 23.5 → 20.4 %), so the distribution is two-humped and a threshold exists between
them. That is the whole basis for caching anything here.

**This is the opposite of what the camera does.** The same store, the same descriptor family and a
moving object retains **0 of 25** under load (`docs/findings/vocabulary-and-labelling.md`). A screen
is a domain where this mechanism works, and the difference is that windows repeat exactly while a
mug on a desk never does.

**Why the backbone and not the pixels.** They measured within noise of each other, so this is an
architectural choice and not an empirical one, said plainly: the backbone is 144 values against
1,024, which is what `ExemplarMemory` stores per entry, and it is the library's own. dHash separates
far better but only 1 % of far pairs reach 0.98 — it is a hash, so it tolerates nothing, and a cache
that only hits on an exact repeat is a dictionary.

**What was not measured.** The capture was one screen — a terminal with output scrolling through it.
A recurrence rate on somebody's real working day is a different number, and *task-level* repetition,
which is what the cache is for, was not measured at all: 20 % of frame pairs recurring within one
minute is not the same claim.
"""

from __future__ import annotations

import numpy as np

from argmind.backbone import Backbone

__all__ = ["Situation", "SituationSense", "PLAN", "POOL", "SHAPE"]

#: The same plan the demo uses. Untrained and stays untrained: the recognition work here measured a
#: seeded backbone decoding shape at 0.996 across unseen positions, so there is nothing to train
#: away from, and training it would make the cache key move under the cache.
PLAN = [("conv", 12, 5, 4), ("dw", 3, 2), ("pw", 16)]
POOL = 3
#: 16:9, and small on purpose — a screen grab costs ~34 ms and everything after it should be noise
#: beside that. Measured on this machine: 2560×1440 grabbed in 34 ms, describing costs ~1 ms.
SHAPE = (72, 128, 3)


class Situation:
    """One observation, reduced to the vector the cache is keyed on.

    Carries the descriptor and the frame it came from. The frame is kept because the brain needs to
    *see* the situation on a miss, and re-grabbing would be a different moment.
    """

    __slots__ = ("key", "frame", "t")

    def __init__(self, key: np.ndarray, frame: np.ndarray, t: float) -> None:
        self.key = key
        self.frame = frame
        self.t = float(t)

    def __repr__(self) -> str:
        return f"Situation(t={self.t:.3f}, key[{self.key.size}])"


class SituationSense:
    """Turns a screen frame into a `Situation`.

    Separate from the sensing itself so a test can feed it frames and never touch a display — the
    same rule `demo/pointer.py` follows, and the reason its composition test can run at all.
    """

    def __init__(self, *, seed: int = 1, dtype: str = "f4") -> None:
        self.backbone = Backbone(PLAN, SHAPE, dtype=dtype, seed=seed, pool=POOL)
        self.n_features = int(self.backbone.n_outputs)

    def describe(self, frame, t: float = 0.0) -> Situation:
        """`frame` is HxWx3 in [0, 1]; any size, resized to `SHAPE` by decimation.

        Decimation rather than interpolation because it is exact, cheap and has no library
        dependency in this direction — the caller already paid for the grab.
        """
        a = np.asarray(frame, dtype=np.float64)
        if a.ndim != 3 or a.shape[2] != 3:
            raise ValueError(f"a frame is HxWx3, got {a.shape}")
        if a.shape[:2] != SHAPE[:2]:
            ys = np.linspace(0, a.shape[0] - 1, SHAPE[0]).astype(int)
            xs = np.linspace(0, a.shape[1] - 1, SHAPE[1]).astype(int)
            a = a[np.ix_(ys, xs)]
        v = np.asarray(self.backbone.features(a), dtype=np.float64)
        n = float(np.linalg.norm(v))
        #: **The cache does not need this, and a mutant removing it survives.** Recorded rather than
        #: contrived around: `ExemplarMemory._sparse` unit-normalises the top-k of both the stored
        #: slot and the query, so the memory is scale-invariant and no cache decision can turn on
        #: the norm here. It stays because `Situation.key` is a public value and callers compare
        #: keys directly -- the measurements in this module's docstring did exactly that -- and a
        #: key that is only *sometimes* a unit vector is a trap for the next such caller.
        return Situation(v / n if n > 1e-12 else v, a, t)
