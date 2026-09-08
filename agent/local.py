# SPDX-License-Identifier: Apache-2.0
"""The local check: the screen key cannot see what a click lands on, so something else has to.

**Why this exists — measured on a real 2560×1440 screen, against the 0.995 the cache uses:**

| change | key score | cache calls them the same? |
|---|---|---|
| a 100×30 region blanked — a button's worth | **1.0000** | **yes** |
| **two 200×100 regions swapped** | **1.0000** | **yes** |
| a 400×300 region blanked — a dialog appearing | 0.9975 | yes |
| a 800×600 region blanked | 0.9904 | no |
| the whole screen shifted 10 px | 0.9950 | no |

**Two buttons can exchange places and the key does not move at all.** A cached click then lands on
the wrong one, and the recurrence figures elsewhere (58.3 % on a real screen) say nothing about it —
they measure whether the screen looks the same *globally*. The mechanism is arithmetic: the
descriptor sees a 128×72 decimation, so one descriptor pixel is 20×20 screen pixels, and a 3×3 grid
pool averages over that again. No threshold on that key recovers local detail.

**A difference hash was tried first and was the wrong instrument.** dHash compares each cell with its
neighbour, so it is *invariant to uniform brightness* by construction — and a button greying out when
it becomes disabled is exactly a uniform brightness change, which is the commonest UI state change
there is. It could not see it. What is used instead is an 8×8 mean-pooled patch, which sees both that
and a content move.

**The tolerance is measured, and the measurement changed the design.** Forty patch locations, six
grabs of an idle screen 0.4 s apart:

    p50 0.0000   p75 0.4398   p90 0.5851   p100 0.9246
    18 of 40 locations moved at all; the other 22 read exactly 0.0000

The distribution is **two-humped with nothing in between**: a location is either perfectly static or
animating hard. So no single tolerance separates "idle" from "changed" everywhere, and the honest
structure is not a better threshold — it is that **a click target sitting in an animating region is
not verifiable, and therefore must not be cached at all**. At the quiet locations the weakest real
change measured 0.1325 against a baseline of exactly zero, so `TOLERANCE = 0.05` sits with a 2.6×
margin under the signal and well over the noise, which is nil.
"""

from __future__ import annotations

import numpy as np

__all__ = ["patch_signature", "signatures", "differs", "unstable", "PATCH", "CELLS",
           "TOLERANCE", "STABLE"]

#: The neighbourhood around a click that has to still look the same, in screen pixels — six
#: descriptor pixels across. Wide enough to hold a button and its surroundings, narrow enough that a
#: change elsewhere in the window does not veto a good entry.
PATCH = 128
#: The patch is reduced to CELLS×CELLS means. 8×8 of a 128×128 patch is one cell per 16 px.
CELLS = 8
#: Largest per-cell difference that still counts as the same thing. Quiet locations read 0.0000 and
#: the weakest real change measured 0.1325.
TOLERANCE = 0.05
#: How much a target may move *between two frames at decision time* before it is called animated and
#: refused. The same scale as `TOLERANCE`, and deliberately the same number: a target that moves by
#: more than the check can tolerate is one the check could never pass.
STABLE = 0.05


def patch_signature(frame, x: int, y: int, size: int = PATCH) -> list | None:
    """`CELLS`×`CELLS` mean-pooled greyscale around `(x, y)`, flattened. `None` when off-frame.

    `None` means *cannot verify*, and every caller here treats that as a failed check rather than a
    passed one — an off-screen coordinate is precisely where a stale cached action would do something
    unintended.
    """
    a = np.asarray(frame, dtype=np.float64)
    h, w = a.shape[:2]
    if not (0 <= y < h and 0 <= x < w):
        return None
    half = size // 2
    patch = a[max(0, y - half):min(h, y + half), max(0, x - half):min(w, x + half)]
    if patch.size == 0:
        return None
    grey = patch.mean(axis=2) if patch.ndim == 3 else patch
    ys = np.linspace(0, grey.shape[0], CELLS + 1).astype(int)
    xs = np.linspace(0, grey.shape[1], CELLS + 1).astype(int)
    out = []
    for i in range(CELLS):
        for j in range(CELLS):
            cell = grey[ys[i]:ys[i + 1], xs[j]:xs[j + 1]]
            out.append(float(cell.mean()) if cell.size else -1.0)
    return out


def signatures(frame, actions) -> list:
    """One signature per action that names a coordinate; `None` for the ones that do not.

    A `type` or a `key` action has nowhere on screen to check. Hashing the whole screen for it would
    either veto every keystroke or verify nothing, so it gets `None` and the caller is told.
    """
    return [patch_signature(frame, int(a["x"]), int(a["y"])) if ("x" in a and "y" in a) else None
            for a in actions]


def differs(stored, current, tolerance: float = TOLERANCE) -> bool:
    """True when any checkable target no longer looks the way it did when the action was decided."""
    if stored is None or current is None or len(stored) != len(current):
        return True
    for a, b in zip(stored, current):
        if a is None and b is None:
            continue
        if a is None or b is None or len(a) != len(b):
            return True
        if float(np.abs(np.asarray(a) - np.asarray(b)).max()) > tolerance:
            return True
    return False


def unstable(first, second, stable: float = STABLE) -> bool:
    """True when a target is moving on its own, from two signatures taken moments apart.

    **An animating target is not cacheable**, and that is the measurement talking rather than
    caution: 18 of 40 sampled locations on an idle screen moved by 0.44 or more between grabs, so an
    entry keyed there could never pass its own check and would be dropped on every hit — paying for
    a look-up and a model call instead of just the model call.
    """
    return differs(first, second, tolerance=stable)
