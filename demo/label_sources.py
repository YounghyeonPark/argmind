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

__all__ = ["MotionLabeller", "CadenceDetector", "brightest_window",
           "motion_centroid", "find_peaks", "BackgroundLocator", "AppearanceMatcher",
           "ObjectMemory", "partition"]


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


def find_peaks(weight: np.ndarray, k: int, *, floor: float = 0.35,
               suppress: float = 0.18) -> list[tuple[float, float]]:
    """The `k` strongest separated maxima of `weight`, each refined to its own local centroid.

    Greedy with non-maximum suppression, and it knows nothing about what the objects are. Returns
    fewer than `k` — possibly none — when the map does not contain that many peaks above `floor`
    of its own maximum, which is the honest answer and is what lets a caller push nothing.

    `suppress` is the exclusion radius as a fraction of the frame's longer side. It is what stops
    one bright object being reported as two, and it bounds how close two objects may be before
    they merge — a real limit of this detector, not of the head that learns from it.
    """
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    m = np.array(weight, dtype=np.float64, copy=True)
    peak = float(m.max()) if m.size else 0.0
    if peak <= 0.0:
        return []
    r = max(1, int(suppress * max(m.shape)))
    out: list[tuple[float, float]] = []
    h, wd = m.shape
    for _ in range(k):
        y, x = divmod(int(np.argmax(m)), wd)
        if m[y, x] < floor * peak:
            break
        y0, y1 = max(0, y - r), min(h, y + r + 1)
        x0, x1 = max(0, x - r), min(wd, x + r + 1)
        local = _centroid(m[y0:y1, x0:x1], 0.5)
        if local is None:
            out.append((x / max(wd - 1, 1), y / max(h - 1, 1)))
        else:
            out.append(((x0 + local[0] * max(x1 - x0 - 1, 1)) / max(wd - 1, 1),
                        (y0 + local[1] * max(y1 - y0 - 1, 1)) / max(h - 1, 1)))
        m[y0:y1, x0:x1] = 0.0
    return out


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

    Tracking `k > 1` objects needs one more thing, and it is not in the detector: **a consistent
    assignment rule**, because a head with 2k outputs is permutation-symmetric and nothing
    distinguishes output pair 1 from pair 2. Peaks are therefore returned **sorted by x**, so
    slot 0 is the leftmost. Measured over two movers against a centre-guess control:

    | labels | crossing | scored as | head | centre | |
    |---|---|---|---|---|---|
    | sorted by x | no | role | **0.129** | 0.312 | 2.43× |
    | sorted by x | yes | role | **0.137** | 0.222 | 1.62× |
    | sorted by x | yes | *identity* | 0.251 | 0.222 | **0.88×, fails** |
    | as found | no | role | 0.256 | 0.312 | 1.22× |
    | as found | yes | role | 0.236 | 0.222 | **0.94×, fails** |

    The sort rule is what makes it work at all, and **what it buys is roles, not identities**: the
    head learns "where is the leftmost object", a well-defined function of the current frame. It
    cannot follow *which* object is which through a crossing — 0.88×, worse than guessing the
    centre — and that is structural. Association depends on the input's *history*, and DESIGN.md's
    *What it is not* already says no choice of instantaneous features represents a
    history-dependent quantity.

    An earlier draft of this note predicted the crossing would break role tracking too. It does
    not: 0.137 with crossings against 0.129 without. The metric was what mattered, not the scene.

    Two limits of frame differencing itself, both measured, and neither is the head's:

    **A moving object leaves a dipole.** `|current - previous|` has a lobe where the object was
    and one where it is, so at a displacement comparable to the object's own size one mover reads
    as two peaks. Measured on a blob of sigma 8.6 px, asking for 3 peaks:

    | displacement | 0.5 px | 1.9 px | 4.8 px | 9.6 px | 19.2 px |
    |---|---|---|---|---|---|
    | `suppress=0.18` | 3 | 3 | 2 | 2 | 2 |
    | `suppress=0.45` | 3 | 1 | 1 | 1 | 1 |

    The separation scales with the object's **size**, not with how far it moved, so one suppression
    radius cannot serve every object size. `suppress` is therefore a stated knob rather than a
    solved problem.

    **And a floor relative to the maximum admits noise when the maximum IS noise** — the left
    column above, where the object barely moved. `min_energy` fixes that half: below it the frame
    is declared to carry no motion and the answer is None. Set it from the caller's own camera
    with `calibrate`, because a noise floor is a property of the sensor and the lighting.

    So `k > 1` is offered with the limit stated: **it separates objects that are far apart and it
    cannot count.** The measured two-object result above is for movers half a frame apart.
    """

    __slots__ = ("every", "floor", "k", "min_energy", "suppress", "locate",
                 "_prev", "_k", "runs", "last", "energy")

    def __init__(self, every: int = 10, *, k: int = 1, floor: float = 0.35,
                 min_energy: float = 0.0, suppress: float = 0.18,
                 locate: Callable[[np.ndarray], list[tuple[float, float]]] | None = None) -> None:
        if every < 1:
            raise ValueError(f"every must be at least 1, got {every}")
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")
        self.every = int(every)
        self.k = int(k)
        self.floor = float(floor)
        self.min_energy = float(min_energy)
        self.suppress = float(suppress)
        self.locate = locate
        self.reset()

    def calibrate(self, frames, quantile: float = 0.9, margin: float = 1.2) -> float:
        """Set `min_energy` from a still sample of the caller's own camera, and return it.

        **The margin is small because the separation is small**, and that is measured rather than
        assumed. Mean absolute frame difference: a still scene reads 0.0129-0.0131 and a fixture
        whose blobs move ~2 px per frame reads 0.0177-0.0190 — a gap of only **1.35x**. A margin
        of 2.0 refuses real motion outright, which is how this was found.

        So `min_energy` catches a scene that is *not moving at all*; it cannot be set to
        distinguish slow motion from noise, and the constructor leaves it at 0.0 so that nothing
        is refused unless a caller opts in.

        A peak-to-median ratio was measured as the alternative, on the theory that noise is
        uniform while a mover is localised. **It is worse**: 0.97x, the two distributions overlap,
        because a second mover raises the map's own median. Recorded so it is not tried again.
        """
        self.reset()
        seen = []
        for f in frames:
            cur = _grey(f)
            if self._prev is not None and self._prev.shape == cur.shape:
                seen.append(float(np.abs(cur - self._prev).mean()))
            self._prev = cur
        self.reset()
        if seen:
            self.min_energy = float(np.quantile(seen, quantile) * margin)
        return self.min_energy

    def reset(self) -> None:
        self._prev: np.ndarray | None = None
        self._k = 0
        self.runs = 0
        self.energy = 0.0
        self.last: list[tuple[float, float]] | None = None

    def __call__(self, frame) -> list[tuple[float, float]] | None:
        """`k` locations on the cadence instants, None on every other frame.

        Always a list, even at `k == 1`: a return type that changes with a constructor argument is
        a trap for the caller, and the length *is* the declared object count.

        Fewer than `k` peaks means the scene does not contain that many movers, and the honest
        answer is None. A partial list would silently reassign slots, which is exactly the
        permutation problem the sort rule exists to prevent.

        Holding the previous frame costs a copy; that is not the expensive part. The point of the
        cadence is that the `locate` call is what a caller cannot afford at output rate.
        """
        cur = _grey(frame)
        if self._prev is None or self._prev.shape != cur.shape:
            self._prev = cur
            self._k = 0
            return None
        step = np.abs(cur - self._prev)
        self._prev = cur
        self.energy = float(step.mean())
        self._k += 1
        if self._k % self.every != 0:
            return None
        if self.energy < self.min_energy:
            # Nothing moved enough to be distinguishable from the sensor's own noise. A floor
            # relative to the peak cannot catch this, because the peak is noise too.
            return None

        self.runs += 1
        got = self.locate(cur) if self.locate is not None \
            else find_peaks(step, self.k, floor=self.floor, suppress=self.suppress)
        if got is None or len(got) < self.k:
            return None
        self.last = sorted(got[:self.k], key=lambda p: p[0])
        return self.last


class BackgroundLocator:
    """Locate objects against a running background estimate, not against the previous frame.

    **Why this exists, measured.** A frame difference is a temporal *derivative*, so a moving
    object leaves a dipole — a lobe where it was and one where it is — and asking for two peaks
    usually returns two lobes of the same object. Over two crossing movers, the two peaks landed
    one per object only **20.3%** of the time; against a running background it is **96.6%**. No
    suppression radius fixes the derivative version here: the dipole spacing is ~17 px for a blob
    of sigma 8.6 px, and the two objects come within ~21.6 px of each other, so the window between
    "merge the lobes" and "merge the objects" is 17–21.6 px.

    It is also what survives a bright *static* distractor, which is the failure that ruled out
    locating on the raw frame: a lamp is in the background model, so with one in the corner this
    still reads 96.6% where frame differencing reads 20.3% and `brightest_window` sits on the lamp.

    `rate` is the background's adaptation rate per frame. It is the one real knob: too slow and a
    lighting change becomes a phantom object, too fast and a stationary object is absorbed into the
    background and disappears. Nothing here measures where that boundary is, so it is stated as a
    knob rather than as a solved problem.
    """

    __slots__ = ("rate", "k", "floor", "suppress", "warmup", "_bg", "_n", "foreground")

    def __init__(self, k: int = 1, *, rate: float = 0.05, floor: float = 0.35,
                 suppress: float = 0.10, warmup: int = 10) -> None:
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")
        if not 0.0 < rate <= 1.0:
            raise ValueError(f"rate must be in (0, 1], got {rate}")
        self.k = int(k)
        self.rate = float(rate)
        self.floor = float(floor)
        self.suppress = float(suppress)
        self.warmup = int(warmup)
        self.reset()

    def reset(self) -> None:
        self._bg: np.ndarray | None = None
        self._n = 0
        #: The last `|frame - background|` map, or None before the first call. It is computed on
        #: every call anyway and was thrown away; keeping it is what lets a caller *see* the
        #: regions the detections came from rather than only their coordinates.
        self.foreground: np.ndarray | None = None

    @property
    def ready(self) -> bool:
        return self._n >= self.warmup

    def __call__(self, frame) -> list[tuple[float, float]] | None:
        """`k` locations, or None until the background has settled or if fewer than `k` are found.

        The background updates on every call including the ones that return None, because it is
        an estimate of the scene rather than a product of the detection.
        """
        cur = _grey(frame)
        if self._bg is None or self._bg.shape != cur.shape:
            self._bg = cur.astype(np.float64)
            self._n = 1
            return None
        self._bg += self.rate * (cur - self._bg)
        self._n += 1
        self.foreground = np.abs(cur - self._bg)
        if not self.ready:
            return None
        got = find_peaks(self.foreground, self.k, floor=self.floor,
                         suppress=self.suppress)
        if len(got) < self.k:
            return None
        return sorted(got, key=lambda p: p[0])


class AppearanceMatcher:
    """Assign detections to slots by what they LOOK like, rather than by where they are.

    This is what makes identity expressible at all. Sorting by x gives *roles* — "leftmost" is a
    function of the current frame, so a head can learn it — but which object is which through a
    crossing depends on which it *was*, and DESIGN.md says no instantaneous basis represents a
    history. Appearance removes the history: if the objects look different, the current frame says
    which is which, and the head's target is a function of the frame again.

    **Measured on an untrained random backbone**, which is the surprising part. Crop embeddings of
    two differently-coloured objects: within-object distance 0.048 against between-object 0.232, a
    ratio of **0.206**. Given detections that land on the objects, nearest-template matching was
    **100% correct** at every crossing speed tried (2/2, 6/6, 5/5, 12/12 assignments).

    **And the control that says it is appearance and not something leaking**: two *identical*
    objects give a ratio of 0.915 and matching at **52.0%**, which is chance. It must fail there,
    and it does.

    Templates are seeded from the first detection, ordered left to right once, and then **held**.
    Updating them each frame would let a template drift onto the other object during a crossing,
    which is the same shape as the guard's baseline tracking the fault it was holding out against.
    """

    __slots__ = ("k", "describe", "_templates")

    def __init__(self, k: int, describe: Callable[[np.ndarray, tuple[float, float]],
                                                  np.ndarray]) -> None:
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")
        self.k = int(k)
        self.describe = describe
        self.reset()

    def reset(self) -> None:
        self._templates: list[np.ndarray] | None = None

    @property
    def seeded(self) -> bool:
        return self._templates is not None

    def __call__(self, frame, points: list[tuple[float, float]]) -> list[tuple[float, float]]:
        """`points` reordered so slot i holds the object matching template i.

        Exhaustive over permutations, which is exact and is fine for the k a caller can label:
        k! is 2, 6, 24. A larger k wants the Hungarian algorithm, and this refuses rather than
        pretending — a greedy fallback would be silently wrong in exactly the ambiguous cases.
        """
        if len(points) != self.k:
            raise ValueError(f"expected {self.k} points, got {len(points)}")
        if self.k > 4:
            raise ValueError(
                f"k={self.k} needs an assignment algorithm rather than {self.k}! permutations; "
                "this refuses instead of falling back to a greedy match that would be wrong "
                "precisely when the objects are hard to tell apart")
        desc = [self.describe(frame, p) for p in points]
        if self._templates is None:
            order = sorted(range(self.k), key=lambda i: points[i][0])
            self._templates = [desc[i] for i in order]
            return [points[i] for i in order]

        best, best_cost = None, float("inf")
        for perm in _permutations(range(self.k)):
            cost = sum(float(np.linalg.norm(desc[perm[i]] - self._templates[i]))
                       for i in range(self.k))
            if cost < best_cost:
                best, best_cost = perm, cost
        return [points[i] for i in best]


def _permutations(seq):
    seq = list(seq)
    if len(seq) <= 1:
        yield tuple(seq)
        return
    for i, v in enumerate(seq):
        for rest in _permutations(seq[:i] + seq[i + 1:]):
            yield (v,) + rest


class ObjectMemory:
    """A growing dictionary of object appearances: the vocabulary, separate from the head.

    **Why this is the shape the constraint forces.** Giving each object its own output pair means
    K objects need a 2K-output head, and `RecursiveLeastSquares.widened()` **refuses** to change
    `n_outputs` — "theta is stored row-major over features, so a different width interleaves the
    old coefficients differently and there is no correct answer without a mapping". A growing
    label set therefore cannot be a growing head.

    So: **unbounded vocabulary, bounded concurrency.** This dictionary grows freely on the
    caller's side; the head has a declared number of slots and tracks whichever remembered objects
    are present. Fixed state, fixed flops, no heap on the core tier's side of the boundary.

    Two alternatives that put the identity on the head's *input* were measured and both fail,
    against a per-slot head at 1.71-1.83x a centre guess:

    | form | 2 objects | 3 | 4 | 5 |
    |---|---|---|---|---|
    | per-slot, 2K outputs | 1.73x | 1.83x | 1.72x | 1.71x |
    | `[embedding, query]` concatenated | 1.10x | 1.03x | 1.01x | 1.00x |
    | query correlated against the feature map | 1.08x | 1.04x | 1.01x | 1.01x |

    The concatenation was predicted to fail: locating a specific pattern is a multiplicative
    interaction and a linear map on a concatenation has no product term. **The correlation was
    predicted to work and does not**, and the reason is measured rather than guessed: the
    correlation map's argmax lands on the queried object only 4-50% of the time, whichever
    similarity is used — cosine 17.5/10.8/4.2%, plain dot 2.5/5.8/50.0%, negative distance
    40.0/15.8/18.3% for three queries. A random backbone's per-cell channel signature does not
    identify an object. That is what a *trained* embedding is for.

    So per-object segmentation by correlating a memory against the feature map is **refused here**,
    not merely absent.

    What this dictionary is measured to do
    --------------------------------------
    Minting a new id when nothing stored is within `tol`, over a 1200-frame run:

    | true objects | ids minted | match correct |
    |---|---|---|
    | 2 | 4 | **96.7%** |
    | 3 | 7 | **66.5%** |
    | 5 | 12 | **43.1%** |

    So it works for two, is mixed at three, and degrades past that. It also **inflates**: about
    twice as many ids as objects, because the same object seen where its crop clips against a frame
    edge describes differently. Automatic growth is real; it is not free and it is not exact.

    **`tol` is tuned on the outcome and there is no way around that here**, which is worth stating
    plainly because the first version of this docstring claimed otherwise. The obvious derivation
    is the gap between within-object and between-object descriptor distance — but that gap **moves
    with the scene**: measured on the crops this actually sees, (0.081, 0.225) at two objects and
    (0.131, 0.288) at three. There is no single interval for a threshold to sit inside, so 0.25 is
    a compromise chosen across a 2/3/5-object sweep, not a derived value.

    What *is* principled is the shape of the failure either side of it, and it is a collapse rather
    than a degradation: at 0.04 this mints **101, 282 and 917** ids for 2, 3 and 5 objects — one per
    detection — and at 0.60 everything becomes a single id.
    """

    __slots__ = ("tol", "max_ids", "_store")

    #: Chosen across a 2/3/5-object sweep, i.e. **tuned on the outcome**. See the class
    #: docstring: the within/between gap that would justify it moves with the object count, so
    #: there is no interval to derive it from.
    DEFAULT_TOL = 0.25

    def __init__(self, tol: float = DEFAULT_TOL, *, max_ids: int = 64) -> None:
        if not tol > 0.0:
            raise ValueError(f"tol must be positive, got {tol}")
        if max_ids < 1:
            raise ValueError(f"max_ids must be at least 1, got {max_ids}")
        self.tol = float(tol)
        self.max_ids = int(max_ids)
        self.reset()

    def reset(self) -> None:
        self._store: list[np.ndarray] = []

    def __len__(self) -> int:
        return len(self._store)

    @property
    def full(self) -> bool:
        return len(self._store) >= self.max_ids

    def observe(self, descriptor) -> tuple[int, bool]:
        """`(id, minted)` for this descriptor: an existing id, or a new one.

        `max_ids` is a hard ceiling, and it exists because "automatically scalable" has a failure
        mode: at a `tol` below the measured within-object distance this mints an id per detection —
        917 of them for five objects. Past the ceiling the nearest id is returned however far away
        it is, and `full` says so, because a caller that silently stops learning new objects is
        worse than one that can be asked.

        Descriptors are **held**, never averaged into. Updating an id's descriptor from each match
        lets it drift onto a neighbour, which is the same shape as the guard's baseline tracking
        the fault it was holding out against.
        """
        # Copied, not referenced. `np.asarray` on a float64 array returns the *same object*, so
        # the store aliased the caller's descriptor and anything they did to it afterwards silently
        # rewrote the memory. Found by a mutation check: `descriptor()` handing out the live array
        # survived, because the test compared it against an array the store was also holding.
        # Same class as the `shares_memory` defect this project records for `Adapter.state`.
        d = np.array(descriptor, dtype=np.float64, copy=True)
        if d.ndim != 1:
            raise ValueError(f"descriptor must be 1-D, got shape {d.shape}")
        if self._store:
            dist = [float(np.linalg.norm(d - s)) for s in self._store]
            j = int(np.argmin(dist))
            if dist[j] <= self.tol or self.full:
                return j, False
        elif self.full:                     # max_ids cannot be 0, so this is unreachable
            raise AssertionError("empty store reported full")
        self._store.append(d)
        return len(self._store) - 1, True

    def descriptor(self, i: int) -> np.ndarray:
        """The held descriptor for an id. A copy, so a caller cannot make it drift."""
        return self._store[i].copy()


def partition(foreground: np.ndarray, points, *, floor: float = 0.25,
              radius: float = 0.35) -> np.ndarray:
    """Label every foreground pixel with the detection nearest to it. 0 means background.

    **This is a partition of a mask, not learned segmentation, and the difference matters.** The
    mask is real — it is where the frame differs from its own background — but the per-object
    split is nearest-detection assignment, so it inherits every limit already measured for the
    locator: it cannot count, a stationary object is absorbed into the background and vanishes,
    and a region is only as good as the detection at its centre.

    Learned per-object segmentation was measured separately and **refused**: correlating a
    remembered descriptor against the backbone's feature map localises the queried object only
    4-50% of the time. This function makes no such claim; it shows what the detector actually saw.

    `floor` is a fraction of the map's own peak, so it follows the scene's contrast rather than
    fixing an absolute. `radius` bounds how far a pixel may be from its detection, in fractions of
    the frame's longer side — without it every noise pixel above the floor joins some region, and
    the picture reads as confident where it is not.
    """
    fg = np.asarray(foreground, dtype=np.float64)
    out = np.zeros(fg.shape, dtype=np.int32)
    peak = float(fg.max()) if fg.size else 0.0
    if peak <= 0.0 or not len(points):
        return out
    mask = fg >= floor * peak
    if not mask.any():
        return out
    h, w = fg.shape
    yy, xx = np.mgrid[0:h, 0:w]
    lim = radius * max(h, w)
    best = np.full(fg.shape, np.inf)
    for i, (px, py) in enumerate(points, start=1):
        d = np.hypot(xx - px * max(w - 1, 1), yy - py * max(h - 1, 1))
        closer = mask & (d < best) & (d <= lim)
        best = np.where(closer, d, best)
        out = np.where(closer, i, out)
    return out
