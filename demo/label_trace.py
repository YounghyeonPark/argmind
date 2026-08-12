# SPDX-License-Identifier: Apache-2.0
"""Watch the label vocabulary grow, minute by minute, on a synthetic scene.

    python -m demo.label_trace                 # three objects, ~20 s
    python -m demo.label_trace --objects 2 --frames 2000
    python -m demo.label_trace --tol 0.15      # a tighter threshold: watch it inflate

Needs no camera and no OpenCV. What it prints is the thing the measurements summarise but do not
show: **when each id is minted, what it was minted from, and how often it is right afterwards.**

Reading it
----------
`minted` lines are the moments the vocabulary grew. The `on` column is which true object the
detection was actually on -- available here because the scene is synthetic, and used for scoring
only; the memory never sees it.

The summary tells you the two things that decide whether automatic growth is working:

  inflation  ids per true object. 1.0 is exact; measured ~2x, because the same object seen where
             its crop clips a frame edge describes differently.
  purity     of the detections that matched an existing id, how often that id was minted from the
             object the detection was on. 96.7% at two objects, 66.5% at three, 43.1% at five.

Measured over 1200 frames:

| objects | ids | from an object | from noise | objects represented | inflation | purity |
|---|---|---|---|---|---|---|
| 2 | 3 | 2 | 1 | **2 of 2** | 1.00x | **90.1%** |
| 3 | 8 | 5 | 3 | **2 of 3** | 1.67x | 62.6% |
| 5 | 11 | 4 | 7 | **3 of 5** | 0.80x | 29.1% |
| 3 at `--tol 0.10` | 72 | 54 | 18 | **3 of 3** | 18.0x | **95.1%** |

**The failure at the default threshold is not inflation — it is objects that never get an id.** At
three objects only two are represented; the third's detections are absorbed into another id and can
never be right, which is most of the purity gap. Inflation reading *below* 1.0 at five objects says
the same thing more starkly: fewer ids came from real objects than there are objects.

**The tight threshold fails the opposite way, and that is the better failure.** At `--tol 0.10` all
three objects are represented and the fragments are 95.1% pure — the cost is 72 ids for 3 objects.
Over-segmentation is recoverable in principle, because pure fragments can be merged;
under-segmentation is not, because an object with no id of its own is information already lost.

That reframes where a merging step should be aimed: at the over-segmented regime, not at the
compromise one. `demo/test_grouping.py` measured merging at the default threshold, where the
objects were already lost.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from argmind.backbone import Backbone
from demo.label_sources import BackgroundLocator, ObjectMemory
from demo.pointer import CROP, IN_SHAPE, PLAN, crop_at

H, W = IN_SHAPE[0], IN_SHAPE[1]
PALETTE = ((0.95, 0.25, 0.20), (0.20, 0.35, 0.95), (0.25, 0.90, 0.30),
           (0.90, 0.85, 0.20), (0.80, 0.25, 0.85))
NAMES = ("red", "blue", "green", "yellow", "purple")


def scene(pts, colours, seed=0, bright=0.45, sigma=0.075):
    yy, xx = np.mgrid[0:H, 0:W]
    img = np.full((H, W, 3), bright)
    for (px, py), c in zip(pts, colours):
        d = ((xx - px * W) ** 2 + (yy - py * H) ** 2) / (2 * (sigma * W) ** 2)
        img += np.exp(-d)[:, :, None] * np.asarray(c, dtype=float)[None, None, :]
    return (img + np.random.default_rng(seed).standard_normal(img.shape) * 0.02).astype("f4")


def orbit(k, i, n, period=90.0):
    a = 2 * np.pi * (k / period) + 2 * np.pi * i / n
    return (0.5 + 0.30 * np.cos(a), 0.5 + 0.28 * np.sin(a * (1 + 0.13 * i)))


def trace(objects=3, frames=1200, every=4, tol=None, verbose=True, summary=True):
    """Run the real path and report every id event. Returns the summary as a dict.

    `verbose` prints the mint events and the per-id table; `summary` prints the totals. They are
    separate because `--quiet` used to set one flag that suppressed both, so a run documented as
    "summary only" printed nothing at all.
    """
    tol = ObjectMemory.DEFAULT_TOL if tol is None else tol
    crop_bb = Backbone(PLAN, (CROP, CROP, 3), dtype="f4", seed=1, pool=2)
    colours = PALETTE[:objects]
    loc = BackgroundLocator(objects, rate=0.03, warmup=15, suppress=0.10)
    mem = ObjectMemory(tol, max_ids=256)

    owner: dict[int, int] = {}          # id -> the true object it was minted from (scoring only)
    hits: dict[int, list[int]] = {}     # id -> [correct, total] afterwards
    events, matched, right = [], 0, 0

    for k in range(frames):
        pts = [orbit(k, i, objects) for i in range(objects)]
        img = scene(pts, colours, seed=k)
        if k % every:
            loc(img)                    # the background must keep updating between cadence frames
            continue
        found = loc(img)
        if found is None:
            continue
        for p in found:
            near = int(np.argmin([np.hypot(p[0] - q[0], p[1] - q[1]) for q in pts]))
            on_object = np.hypot(p[0] - pts[near][0], p[1] - pts[near][1]) <= 0.08
            i, minted = mem.observe(crop_desc(crop_bb, img, p))
            label = NAMES[near] if on_object else "nothing"
            if minted:
                owner[i] = near if on_object else -1
                hits[i] = [0, 0]
                events.append((k, i, label, p))
                if verbose:
                    print(f"  frame {k:>5}   MINTED id {i:<3} from {label:<8} "
                          f"at ({p[0]:.2f}, {p[1]:.2f})   vocabulary now {len(mem)}")
            elif on_object:
                matched += 1
                ok = owner.get(i, -1) == near
                right += ok
                hits[i][1] += 1
                hits[i][0] += ok

    real = [i for i in owner if owner[i] >= 0]
    out = {
        "ids": len(mem),
        "ids_from_objects": len(real),
        "inflation": len(real) / objects if objects else float("nan"),
        "purity": right / matched if matched else float("nan"),
        "matched": matched,
        "events": events,
        "objects_represented": len({owner[i] for i in owner if owner[i] >= 0}),
    }
    if verbose:
        print(f"\n  {'id':>4}{'minted from':>14}{'used':>8}{'correct':>10}")
        for i in sorted(hits):
            src = NAMES[owner[i]] if owner.get(i, -1) >= 0 else "nothing"
            c, t = hits[i]
            acc = f"{c / t:.0%}" if t else "-"
            print(f"  {i:>4}{src:>14}{t:>8}{acc:>10}")
    if summary:
        print(f"\n  {objects} objects -> {out['ids']} ids "
              f"({out['ids_from_objects']} from a real object, "
              f"{out['ids'] - out['ids_from_objects']} from noise), "
              f"{out['objects_represented']} of {objects} objects represented")
        print(f"  inflation {out['inflation']:.2f}x   "
              f"purity {out['purity']:.1%} over {matched} matches")
    if verbose:
        print("\n  Inflation above 1.0 is the same object minted twice: a crop clipped at a frame")
        print("  edge describes differently. An object with NO id of its own is the worse case --")
        print("  its detections are absorbed into another object's id and can never be right.")
    return out


def crop_desc(crop_bb, img, p):
    v = crop_bb.features(crop_at(img, p)).astype(np.float64)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m demo.label_trace",
                                 description=__doc__.splitlines()[0])
    ap.add_argument("--objects", type=int, default=3)
    ap.add_argument("--frames", type=int, default=1200)
    ap.add_argument("--tol", type=float, default=None,
                    help=f"match threshold (default {ObjectMemory.DEFAULT_TOL}). Below the "
                         "within-object distance it mints an id per detection")
    ap.add_argument("--quiet", action="store_true", help="summary only")
    args = ap.parse_args(argv)
    trace(args.objects, args.frames, tol=args.tol, verbose=not args.quiet)
    return 0


if __name__ == "__main__":
    sys.exit(main())
