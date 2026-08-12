# SPDX-License-Identifier: Apache-2.0
"""Learn a segmentation MAP instead of coordinates, with a resolution pyramid of past outputs.

Why a map rather than points
----------------------------
Every workaround in this package exists because the head emits *coordinates*:

| problem | what it forced |
|---|---|
| 2K outputs have no intrinsic order | labels sorted by x (0.129) or matched by appearance (0.041), against 0.256 for arbitrary order |
| a scene does not come with a number of objects | `k` is a declaration in every locator here, and none of them can count |
| which detection belongs in which slot | an assignment algorithm, exact but O(k^3) |

**A map output has no slots.** There is nothing to permute, nothing to count, and no assignment to
make. That is not a smaller version of the same design; it removes three measured problems at once.

The output is deliberately **lower resolution than the input** — 9x12 here against 72x96 — because
a segmentation does not need per-pixel precision to say where something is, and the head's parameter
count is `n_features x n_outputs`.

The pyramid
-----------
`Pyramid` holds past output maps as leaky integrators, each at its own rate **and its own
resolution**: recent past sharp, distant past coarse. The rates and the resolutions move together on
purpose — a band averaging over a long time has less to say about exactly *where*, so full
resolution there would be paying for noise.

That is the multi-timescale memory already measured to work in this package, organised spatially
rather than as a flat vector: an EWMA bank over past outputs reached 141.7x a centre guess where a
frame-only head read 1.00x, and a single lag was *worse* than persistence.

What is measured, including the two results that scope it
--------------------------------------------------------
IoU at the output resolution, free-running:

| scene | frame only | + pyramid | the segmenter (control) |
|---|---|---|---|
| one object | 0.99 | **1.00** | 1.00 |
| plus a **same-colour twin** | 0.41 | **0.98** | **0.50** |

The second row is the first time in this package that a head has **beaten the detector it learns
from**: a per-frame segmenter cannot tell identical twins apart, and the pyramid carries which one
was the target.

**But most of that 0.98 is persistence**, and saying so is the honest scoping: repeating the
previous *true* map scores **0.92** on the same fixture. The learned part adds 0.06 IoU. What is
decisively better is the comparison against the segmenter, not against doing nothing.

**And it fails absorbingly.** A 60-frame blackout takes it from 0.98 to **0.00, permanently**: a
blank frame drives the map to zero, the pyramid fills with zeros, and zero is a fixed point. The
frame-only head *recovers* (0.13 to 0.62) precisely because it has no memory to corrupt. Resetting
the pyramid on an unhealthy frame does **not** fix it either — the head trained with a populated
pyramid has never seen an empty one, so a reset lands in a distribution that never occurred.

What does fix it is **training-time dropout of the pyramid**, and it is a trade with no free lunch:

| dropout | while healthy | after the blackout |
|---|---|---|
| none | **0.98** | **0.14** (never recovers) |
| 0.05 – 0.30 | ~0.49 | **~0.68** |
| 0.50 | 0.11 | 0.10 |

So a caller chooses between a head that is excellent until something goes wrong, and one that is
mediocre and survives. The recovered 0.68 still beats the segmenter's 0.50, so the memory keeps
paying — just less. Nothing here resolves that trade, and it is recorded rather than tuned away.
"""

from __future__ import annotations

import numpy as np

__all__ = ["Pyramid", "block_mean", "map_target", "map_iou"]


def block_mean(a, out_h: int, out_w: int) -> np.ndarray:
    """Block-mean down to `(out_h, out_w)`, preserving a trailing channel axis if present.

    Refuses a target larger than the source rather than interpolating: upsampling here would
    invent detail, and every caller wants the opposite direction.
    """
    x = np.asarray(a, dtype=np.float64)
    if out_h < 1 or out_w < 1:
        raise ValueError(f"output must be at least 1x1, got {out_h}x{out_w}")
    if out_h > x.shape[0] or out_w > x.shape[1]:
        raise ValueError(
            f"cannot block-mean {x.shape[0]}x{x.shape[1]} up to {out_h}x{out_w}; this only "
            "reduces, because the pyramid's whole point is that coarse is cheaper")
    fh, fw = x.shape[0] // out_h, x.shape[1] // out_w
    x = x[:out_h * fh, :out_w * fw]
    if x.ndim == 2:
        return x.reshape(out_h, fh, out_w, fw).mean(axis=(1, 3))
    return x.reshape(out_h, fh, out_w, fw, x.shape[2]).mean(axis=(1, 3))


def map_target(mask, out_h: int, out_w: int) -> np.ndarray:
    """A boolean mask as the flat, soft, low-resolution map the head learns."""
    return block_mean(np.asarray(mask, dtype=np.float64), out_h, out_w).ravel()


def map_iou(predicted, target, thresh: float = 0.4) -> float:
    """Intersection over union of two flat maps at a shared threshold.

    Soft maps on both sides, so the threshold applies to both: comparing a soft prediction against
    a hard target would make the number depend on how much of a cell the object covers.
    """
    a = np.asarray(predicted).ravel() >= thresh
    b = np.asarray(target).ravel() >= thresh
    union = int((a | b).sum())
    return float((a & b).sum() / union) if union else 0.0


class Pyramid:
    """Past output maps as leaky integrators at falling rate and falling resolution.

    Each band is `(rate, height, width)`. The default spans roughly one frame, seven frames and
    thirty: sharp and recent, then coarser and longer. See the module docstring for what this buys
    (0.98 against 0.50 for a per-frame segmenter on an ambiguous scene) and what it costs (a
    permanent 0.00 after a dropout unless the head was trained with the memory sometimes blanked).
    """

    __slots__ = ("bands", "out_h", "out_w", "_maps")

    DEFAULT_BANDS = ((0.6, 9, 12), (0.15, 5, 6), (0.03, 3, 4))

    def __init__(self, out_h: int = 9, out_w: int = 12, bands=DEFAULT_BANDS) -> None:
        if not bands:
            raise ValueError("a pyramid needs at least one band")
        for rate, h, w in bands:
            if not 0.0 < rate <= 1.0:
                raise ValueError(f"every band rate must be in (0, 1], got {rate}")
            if h > out_h or w > out_w:
                raise ValueError(
                    f"band {h}x{w} is larger than the output map {out_h}x{out_w}; a band is a "
                    "*coarsening* of the output, so it cannot be finer than it")
        self.bands = tuple((float(r), int(h), int(w)) for r, h, w in bands)
        self.out_h, self.out_w = int(out_h), int(out_w)
        self.reset()

    def reset(self) -> None:
        self._maps = [np.zeros((h, w)) for _r, h, w in self.bands]

    @property
    def width(self) -> int:
        return sum(h * w for _r, h, w in self.bands)

    def features(self) -> np.ndarray:
        """The bands, flattened and concatenated: this is what goes into the head's basis."""
        return np.concatenate([m.ravel() for m in self._maps])

    def observe(self, flat_map) -> None:
        """Fold an output map into every band.

        The caller decides whether `flat_map` is the head's own prediction or a supplied label.
        During learning the label is the honest thing to feed; free-running with the prediction is
        what ships, and the gap between them is the exposure-bias gap this package measures
        elsewhere at an order of magnitude.
        """
        v = np.asarray(flat_map, dtype=np.float64).ravel()
        if v.size != self.out_h * self.out_w:
            raise ValueError(f"expected {self.out_h * self.out_w} values, got {v.size}")
        full = v.reshape(self.out_h, self.out_w)
        for i, (rate, h, w) in enumerate(self.bands):
            self._maps[i] += rate * (block_mean(full, h, w) - self._maps[i])
