# SPDX-License-Identifier: Apache-2.0
"""Labels that come out of the input instead of out of the mouse — and what that is worth.

Why not self-supervision
------------------------
The pure version — learn from the video alone with no external signal — was measured in this
project and **does not work** on an untrained backbone: a linear map from random features to the
*next* embedding learned essentially nothing (the learned map's diagonal reads 0.004 against an
identity's 1.0), and once frozen its held-out error was 2–7× worse than simply repeating the last
value. Without a grounded target, a self-referential objective is satisfied by any constant.

So the label has to come from something. What it does **not** have to come from is a human: a
second, independent computation on the same input can supply it. That makes it an *aiding sensor*,
which is the structure this library was designed against — "an inertial + aiding-sensor estimator
with online bias calibration". The aiding sensor is authoritative and intermittent; the learned
model is fast and always on.

The rule that decides whether a learned head pays, measured both ways
---------------------------------------------------------------------
**An aiding sensor earns a learned model only if it can be blind while the estimand changes.**
Blindness that is anti-correlated with change is already covered by holding the last value.

Scored on the frames where the source is blind — the ones a caller with no learned model would
have to guess — over a moving blob, 660 frames, the second half after learning:

| label source | blind by | head | hold last | centre | verdict |
|---|---|---|---|---|---|
| `MotionLabeller` | nothing moving | 0.090 | **0.078** | 0.316 | 0.87×, the head **loses** |
| `CadenceDetector` | a declared interval | **0.079** | 0.116 | 0.231 | 1.46×, the head wins |

`MotionLabeller` fails structurally, not for want of tuning: motion energy is blind exactly when
the target is not moving, so it cannot be blind while the estimand changes. It is kept here
because the negative result is the useful half, and because it is the first thing anyone reaches
for.

The control that is not flattering, and it decides when to use any of this
-------------------------------------------------------------------------
Running the detector on **every** frame scores **0.010** — eight times better than the learned
head. So the head is not more accurate than its own teacher; its value is conditional on the
teacher being unavailable. And on this host the exhaustive detector costs **0.10 ms/frame against
the backbone-plus-pipeline's 0.58 ms** (measured), so it is not the expensive one either.

That is the honest scope: **automatic labelling removes the hand from the mouse, and the learned
head pays only where the label source genuinely cannot run at the output rate.** On a PC with a
cheap detector, run the detector. The case argmind is built for is the other one — a heavy
detector on an NPU at 10–30 Hz and a 24-byte head between its runs — and *that* has not been
measured here, so it is not claimed.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

__all__ = ["MotionLabeller", "CadenceDetector", "brightest_window", "motion_centroid"]


def _grey(frame) -> np.ndarray:
    a = np.asarray(frame, dtype=np.float32)
    return a.mean(axis=2) if a.ndim == 3 else a


def _centroid(weight: np.ndarray, floor: float) -> tuple[float, float] | None:
    """The centroid of `weight` after zeroing everything below `floor` of its peak.

    The floor is not cosmetic. Without it, sensor noise spread over the whole frame drags the
    centroid towards the middle — which is a bias towards exactly the answer a constant
    predictor gives, i.e. towards the one control this has to be able to beat.
    """
    peak = float(weight.max())
    if peak <= 0.0:
        return None
    w = np.where(weight >= floor * peak, weight, 0.0)
    total = float(w.sum())
    if total <= 0.0:
        return None
    h, wd = w.shape
    cy = float((w.sum(axis=1) @ np.arange(h, dtype=np.float64)) / total)
    cx = float((w.sum(axis=0) @ np.arange(wd, dtype=np.float64)) / total)
    return cx / max(wd - 1, 1), cy / max(h - 1, 1)


def brightest_window(frame, half: int = 6) -> tuple[float, float]:
    """Exhaustive search for the brightest (2·half+1)² window, by summed-area table.

    Knows nothing about hands, which is the point — it is a stand-in for any authoritative
    detector. On a real camera it finds the brightest thing in view, which is usually a window
    or a lamp rather than a person; `motion_centroid` is the better default there and this is
    kept because it is what the measurements in the module docstring used.
    """
    g = _grey(frame).astype(np.float64)
    c = np.pad(g.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    k = 2 * half + 1
    h, w = g.shape
    if h <= k or w <= k:
        return 0.5, 0.5
    box = c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]
    y, x = divmod(int(np.argmax(box)), box.shape[1])
    return (x + half) / (w - 1), (y + half) / (h - 1)


def motion_centroid(accumulated: np.ndarray, floor: float = 0.35) -> tuple[float, float] | None:
    """Where movement happened, from an accumulated absolute-difference map."""
    return _centroid(accumulated, floor)


class MotionLabeller:
    """Frame-to-frame motion energy, and its centroid when there is enough of it.

    **Measured not to earn a learned head** — see the module docstring. It is blind exactly when
    nothing moves, so holding its last output is correct whenever it is unavailable (0.87×
    against the head). Kept because that is a useful negative result and because this is the
    first thing anyone reaches for.

    Also useful on its own terms as a *component*: `CadenceDetector` uses the same difference
    map, but decides its availability by cadence instead of by content.
    """

    __slots__ = ("min_energy", "floor", "_prev", "energy")

    def __init__(self, min_energy: float = 0.004, floor: float = 0.35) -> None:
        self.min_energy = float(min_energy)
        self.floor = float(floor)
        self._prev: np.ndarray | None = None
        self.energy = 0.0

    def reset(self) -> None:
        self._prev = None
        self.energy = 0.0

    def calibrate(self, frames, quantile: float = 0.6, margin: float = 1.5) -> float:
        """Set `min_energy` from a still sample of the caller's own camera, and return it.

        Measured rather than assumed: a camera's noise floor is a property of the sensor and
        the lighting. Setting this demo's *other* threshold by eye froze it on frame two.
        """
        self.reset()
        seen = []
        for f in frames:
            self(f)                       # for the side effect on `energy`
            seen.append(self.energy)
        self.reset()
        if len(seen) < 2:
            return self.min_energy
        self.min_energy = float(np.quantile(seen[1:], quantile) * margin)
        return self.min_energy

    def __call__(self, frame) -> tuple[float, float] | None:
        """The motion centroid in frame fractions, or None when the detector is blind.

        None is the honest answer for a still frame, and it is what makes this an intermittent
        sensor rather than one that invents a position. A caller pushes nothing on a None.
        """
        cur = _grey(frame)
        if self._prev is None or self._prev.shape != cur.shape:
            self._prev = cur
            self.energy = 0.0
            return None
        diff = np.abs(cur - self._prev)
        self._prev = cur
        self.energy = float(diff.mean())
        if self.energy < self.min_energy:
            return None
        return _centroid(diff, self.floor)


class CadenceDetector:
    """An authoritative locator the caller runs only every `every` frames.

    This is the version that earns a learned head, because its blindness is set by a **declared
    interval** rather than by the scene: the target can and does move while it is unavailable,
    which is precisely what holding the last value cannot cover.

    `every` is the caller's declaration of what it can afford, and it is deliberately not
    guessed. On this host the detector is cheaper than the backbone, so `every=1` is the right
    answer for a PC and the whole arrangement is redundant; the case it is built for is a heavy
    detector that cannot run at the output rate, which is not measured here.

    By default it locates the centroid of the **single-step** frame difference at the cadence
    instant — so it finds what is moving *now*, which on a real camera is a person rather than a
    window. Pass `locate` for something else; `brightest_window` is what the module's headline
    measurements used.

    **Accumulating the difference between runs was the first design and is measured out of it.**
    An accumulated map is about the *interval*, so its centroid is the middle of the arc the
    target travelled and its error grows with the window — measured against the truth at the
    cadence instant, with no learning involved:

    | locator | every=2 | every=5 | every=10 | every=15 |
    |---|---|---|---|---|
    | accumulated | 0.048 | 0.074 | 0.121 | 0.160 |
    | single-step | 0.036 | 0.037 | 0.036 | **0.021** |
    | `brightest_window` | 0.010 | 0.010 | 0.010 | 0.010 |

    Re-stamping the accumulated label at the interval's midpoint does **not** fix it — measured,
    identical to three decimals — because with a fast reference `hold_last` pairs the label with
    the current features either way. It is the label that is wrong about every instant, not the
    alignment. That distinction is worth keeping: this project's recorded two-rate lesson is
    about *stamping*, and this is the one case where stamping is not the answer.
    """

    __slots__ = ("every", "floor", "locate", "_prev", "_k", "runs", "last")

    def __init__(self, every: int = 10, *, floor: float = 0.35,
                 locate: Callable[[np.ndarray], tuple[float, float] | None] | None = None) -> None:
        if every < 1:
            raise ValueError(f"every must be at least 1, got {every}")
        self.every = int(every)
        self.floor = float(floor)
        self.locate = locate
        self.reset()

    def reset(self) -> None:
        self._prev: np.ndarray | None = None
        self._k = 0
        self.runs = 0
        self.last: tuple[float, float] | None = None

    def __call__(self, frame) -> tuple[float, float] | None:
        """A location on the cadence instants, None on every other frame.

        Holding the previous frame costs a copy; that is not the expensive part. The point of
        the cadence is that the `locate` call is what a caller cannot afford at output rate.
        """
        cur = _grey(frame)
        if self._prev is None or self._prev.shape != cur.shape:
            self._prev = cur
            self._k = 0
            return None
        step = np.abs(cur - self._prev)
        self._prev = cur
        self._k += 1
        if self._k % self.every != 0:
            return None

        self.runs += 1
        got = self.locate(cur) if self.locate is not None \
            else motion_centroid(step, self.floor)
        if got is not None:
            self.last = got
        return got
