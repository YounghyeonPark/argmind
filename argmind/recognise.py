# SPDX-License-Identifier: Apache-2.0
"""`RecognitionLayer` — the five measured pieces wired into one layer of a `Stack`.

Measured separately over 2026-08-19/20 and each without any representation learning: a crop
descriptor from an untrained backbone recognises shape at unseen positions (0.996); rarity finds an
object with no labels (1.000 against a 0.101 chance rate); a stored exemplar steers attention to the
remembered object (1.000, and it flips when the memory is swapped); sixteen of 108 dimensions carry
all of it. `ExemplarMemory` made the store a component. This is the composition.

**What it does, per frame.** Crop every candidate, describe each one, then:

- if something in memory matches, look there and report **that slot** — recognition;
- otherwise take the rarest crop and admit it — bootstrap.

The emitted features are the winning crop's descriptor, so a head above learns from what was
attended to rather than from the whole frame. `slot` is the identity the caller asked for: the same
object leaving and returning gets the same number, which is the claim `tests/test_recognise.py`
measures rather than asserts.

**It is a layer, and it is frozen.** Selection is an argmax over candidates, which has no gradient
with respect to the frame, so a `Stack` must be built with `feature_lr=None`. That is declared at
construction rather than discovered on the first sample — see `argmind/stack.py`.

**One flat state block**, `[ backbone | memory ]`, so persistence is a copy and one record carries
the weights *and* what has been seen. The block is a snapshot refreshed on read, as `Stack`'s is,
because the two components own their own arrays.
"""

from __future__ import annotations

import numpy as np

from .backbone import Backbone
from .memory import ExemplarMemory

__all__ = ["RecognitionLayer"]


class RecognitionLayer:
    """Scan, recognise or admit, and emit the attended crop's descriptor.

    `centres` is the caller's candidate list in frame fractions. It is supplied rather than derived
    because the density is a cost decision — 247 candidates per frame against 70 changes runtime by
    3.4x and leaves the selection unchanged, measured — and this class should not make it.
    """

    def __init__(self, backbone: Backbone, memory: ExemplarMemory, in_shape, centres, *,
                 admit_margin: float, refine: tuple[int, int] | None = None,
                 emit: str = "descriptor"):
        if backbone.n_outputs != memory.n_features:
            raise ValueError(
                f"the backbone emits {backbone.n_outputs} features and the memory stores "
                f"{memory.n_features}; they must match or a descriptor cannot be looked up")
        self.backbone, self.memory = backbone, memory
        self.in_shape = tuple(in_shape)
        self.crop = tuple(backbone.in_shape[:2])
        self.centres = [(float(cx), float(cy)) for cx, cy in centres]
        if not self.centres:
            raise ValueError("no candidate centres; there is nothing to look at")
        #: **The gate on what may be written.** A crop is admitted only if it is a genuine outlier:
        #: the winner's mean similarity to the rest must be at least this far below the median.
        #:
        #: Without it the first empty frame stores a patch of *wall*, and every wall crop then
        #: matches it forever. Measured before this existed: an object recognised as slot 0 came
        #: back as slot 1 after a three-frame absence, because a wall exemplar admitted during the
        #: gap outscored it everywhere. The failure was predicted when the design was proposed and
        #: built anyway; the measurement is what caught it.
        #:
        #: The separation is not marginal, which is why a threshold works at all — measured over
        #: six frames, `median - min` is **0.0007-0.0011** on an empty wall and **0.126-0.143**
        #: with an object present, two orders of magnitude apart. Required rather than defaulted,
        #: for the reason `ExemplarMemory.threshold` is: the scale belongs to the caller's scene.
        if admit_margin <= 0.0:
            raise ValueError(
                f"admit_margin must be positive, got {admit_margin}. Without a gate the first "
                "featureless frame stores its background and then recognises it everywhere")
        self.admit_margin = float(admit_margin)
        #: `(span, step)` in pixels: after the coarse pass picks a region, look again around it at
        #: this density and decide there. `None` disables it.
        #:
        #: **It has to happen before the memory is consulted, not after.** A first version refined
        #: the centre once the slot was already assigned, which changed the reported centre and
        #: nothing else — the numbers were identical to no refinement at all. The point of a better
        #: centre is to look *that* descriptor up.
        #:
        #: The cost it exists to avoid is measured: the best match score available on a coarse grid
        #: is 0.8835 against a 0.85 threshold, so noise decides, while a step-1 grid reaches 0.9990
        #: at 3,577 crops per frame.
        self.refine = None if refine is None else (int(refine[0]), int(refine[1]))
        if self.refine is not None and (self.refine[0] < 1 or self.refine[1] < 1):
            raise ValueError(f"refine is (span, step) in pixels and both must be >= 1, got {refine}")
        #: What the layer emits, and it is an architectural choice rather than a formatting one.
        #:
        #: `"descriptor"` returns the attended crop's descriptor: **one region per frame**, which is
        #: what the first version did. The owner's objection is the right one — nothing about vision
        #: says only one thing may be recognised at a time. Foveal *selection* is largely serial;
        #: *recognition* is not, and a viewer knows a chair, a bookshelf and a face are present
        #: without visiting each in turn.
        #:
        #: The reason it was serial was not a principle. `features()` must emit a fixed-width vector
        #: for the head above, and one descriptor is fixed width — so an implementation constraint
        #: was allowed to decide the architecture. `"presence"` fixes it at no cost: the scan already
        #: computes every candidate's descriptor and looks each one up, and the serial version threw
        #: all but the best away. A look-up is 0.026 ms, so nothing was being saved.
        #:
        #: `"presence"` returns **one score per memory slot** — the best match found anywhere in the
        #: frame for that exemplar, zero where the slot is empty or unmatched. Fixed width by
        #: construction, and it is exactly what `ExemplarMemory.n_outputs = capacity` anticipated.
        if emit not in ("descriptor", "presence"):
            raise ValueError(f"emit must be 'descriptor' or 'presence', got {emit!r}")
        self.emit = emit
        #: How far from a matched centre a candidate counts as "already accounted for". One crop
        #: side, in frame fractions: two objects closer than that share a crop and cannot be
        #: separated by this mechanism anyway.
        self._claim_x = backbone.in_shape[1] / float(in_shape[1])
        self._claim_y = backbone.in_shape[0] / float(in_shape[0])
        self.n_outputs = backbone.n_outputs if emit == "descriptor" else memory.capacity
        self.dtype = np.dtype(backbone.dtype)

        #: What the last forward pass did, for a caller that wants to draw or log it.
        self.last_centre = None
        self.last_slot = -1
        self.last_recognised = False
        #: `[(slot, score, (cx, cy)), ...]` for every slot recognised this frame, strongest first.
        #: Empty when nothing was. Under `emit="descriptor"` it holds at most one entry, which is
        #: the whole difference between the two modes made visible to a caller.
        self.last_present: list = []

        n_bb = int(np.asarray(backbone.state).size)
        self._split = n_bb
        self.state_size = n_bb + int(memory.state_size)
        self._block = np.zeros(self.state_size, dtype=self.dtype)

    # ---- state -------------------------------------------------------------------------------

    @property
    def state(self) -> np.ndarray:
        """A snapshot, refreshed on read — the components own their arrays, as in `Stack`."""
        self._block[:self._split] = np.asarray(self.backbone.state).ravel()
        self._block[self._split:] = np.asarray(self.memory.state).ravel()
        return self._block

    def load_state(self, flat) -> None:
        arr = np.asarray(flat).ravel()
        if arr.size != self.state_size:
            raise ValueError(f"state is {self.state_size} values, got {arr.size}")
        self.backbone.load_state(arr[:self._split])
        self.memory.load_state(arr[self._split:])

    # ---- the layer protocol ------------------------------------------------------------------

    def _crop(self, image, cx, cy):
        h, w = self.in_shape[0], self.in_shape[1]
        ch, cw = self.crop
        y0, x0 = int(cy * h) - ch // 2, int(cx * w) - cw // 2
        return np.asarray(image)[y0:y0 + ch, x0:x0 + cw].astype(self.backbone.dtype)

    def _describe(self, image):
        out = []
        for cx, cy in self.centres:
            v = np.asarray(self.backbone.features(self._crop(image, cx, cy)), dtype=np.float64)
            n = float(np.linalg.norm(v))
            out.append(v / n if n > 1e-12 else v)
        return np.asarray(out)

    def _choose(self, described):
        """Index of the crop to attend to: the best remembered one, or the rarest. `matched` says
        which rule fired."""
        if self.memory.occupied:
            best_j, best_score = -1, -np.inf
            for j, d in enumerate(described):
                report = self.memory.look_up(d)
                if report.matched and report.score > best_score:
                    best_j, best_score = j, report.score
            if best_j >= 0:
                return best_j, True
        S = described @ described.T
        mean_sim = (S.sum(axis=1) - np.diag(S)) / max(len(described) - 1, 1)
        return int(np.argmin(mean_sim)), False

    def _neighbourhood(self, cx, cy):
        span, step = self.refine
        h, w, (ch, cw) = self.in_shape[0], self.in_shape[1], self.crop
        return [(x / w, y / h)
                for y in range(int(cy * h) - span, int(cy * h) + span + 1, step)
                for x in range(int(cx * w) - span, int(cx * w) + span + 1, step)
                if ch // 2 <= y <= h - ch // 2 and cw // 2 <= x <= w - cw // 2]

    def _presence(self, described, centres) -> np.ndarray:
        """Best match per slot, over the whole frame. One pass, every slot, in parallel."""
        out = np.zeros(self.memory.capacity, dtype=np.float64)
        where: dict = {}
        #: **One matrix product for every candidate against every slot.** A loop over `look_up`
        #: costs 2.033 ms at 1,000 slots, so 147 candidates of it is 479 ms a frame; batched it is
        #: 8.8 ms, and 4,000 slots is 9.4 ms — nearly flat, because the cost is the queries'
        #: sparsification rather than the slot count. That is what makes a large memory usable.
        for j, report in enumerate(self.memory.look_up_many(described)):
            if report.slot >= 0 and report.matched and report.score > out[report.slot]:
                out[report.slot] = report.score
                where[report.slot] = centres[j]
        self.last_present = sorted(
            ((s, float(out[s]), where[s]) for s in where), key=lambda t: -t[1])
        return out

    def features(self, image) -> np.ndarray:
        """Look, then either recognise or admit.

        Returns the attended crop's descriptor under `emit="descriptor"`, or one score per memory
        slot under `emit="presence"` — see the note on `emit`.
        """
        centres = self.centres
        described = self._describe(image)

        if self.refine is not None:
            j, _how = self._choose(described)
            local = self._neighbourhood(*centres[j])
            if local:
                keep = self.centres
                self.centres = local
                try:
                    described = self._describe(image)
                finally:
                    self.centres = keep
                centres = local

        if self.emit == "presence":
            scores = self._presence(described, centres)
            self.last_recognised = bool(self.last_present)

            #: **Admission has to be parallel too**, or the memory can only grow while the scene is
            #: empty of everything it already knows. Measured before this: with a disc in memory and
            #: a disc *and* a bar in the frame, the bar was never admitted -- the early return fired
            #: on the disc's match and nothing else was considered. Recognising in parallel and
            #: admitting serially is still a one-object system.
            #:
            #: The rule is "explain the scene, then remember what is left over": candidates near a
            #: slot already matched are accounted for, and the rarity gate runs on the remainder.
            claimed = [c for _s, _sc, c in self.last_present]
            free = [j for j, c in enumerate(centres)
                    if all(abs(c[0] - q[0]) > self._claim_x or abs(c[1] - q[1]) > self._claim_y
                           for q in claimed)]
            if free and self.memory.occupied < self.memory.capacity:
                sub = described[free]
                S = sub @ sub.T
                mean_sim = (S.sum(axis=1) - np.diag(S)) / max(len(sub) - 1, 1)
                k = int(np.argmin(mean_sim))
                if float(np.median(mean_sim) - mean_sim[k]) >= self.admit_margin:
                    slot = self.memory.admit(sub[k])
                    scores[slot] = 1.0
                    self.last_present.append((slot, 1.0, centres[free[k]]))
                    self.last_present.sort(key=lambda t: -t[1])
            if self.last_present:
                self.last_slot, _s, self.last_centre = self.last_present[0]
            else:
                self.last_slot, self.last_centre = -1, centres[0]
            #: **Close the frame**, which is what turns per-frame presence into the retention
            #: signal. Only this mode can: the descriptor path reports at most one slot, so it
            #: cannot say which of the others were absent rather than merely not the strongest.
            self.memory.end_frame([sl for sl, _sc, _c in self.last_present])
            return scores.astype(self.dtype)

        best_j, best_score = -1, -np.inf
        if self.memory.occupied:
            for j, d in enumerate(described):
                report = self.memory.look_up(d)
                if report.matched and report.score > best_score:
                    best_j, best_score = j, report.score

        if best_j >= 0:
            self.last_recognised = True
            self.last_slot = self.memory.look_up(described[best_j]).slot
        else:
            # Nothing known is here. Fall back to rarity, which needs no labels -- but admit only
            # what stands out, or the background becomes an exemplar and wins forever.
            S = described @ described.T
            mean_sim = (S.sum(axis=1) - np.diag(S)) / max(len(self.centres) - 1, 1)
            best_j = int(np.argmin(mean_sim))
            self.last_recognised = False
            if float(np.median(mean_sim) - mean_sim[best_j]) >= self.admit_margin:
                self.last_slot = self.memory.admit(described[best_j])
            else:
                #: Looked, recognised nothing, stored nothing. Reported as -1 rather than silently
                #: reusing the last slot, because "there was nothing here" is a real answer.
                self.last_slot = -1

        self.last_centre = centres[best_j]
        self.last_present = ([(self.last_slot, float(max(best_score, 0.0)), self.last_centre)]
                             if self.last_slot >= 0 and self.last_recognised else [])
        return described[best_j].astype(self.dtype)

    def backward(self, image, d_features):
        raise NotImplementedError(
            "selection is an argmax over candidates, which has no gradient with respect to the "
            "frame; build the Stack with feature_lr=None to declare these layers frozen")

    def apply_gradients(self, d_weights, d_biases, learning_rate):
        raise NotImplementedError("a RecognitionLayer has no weights of its own to step")
