# SPDX-License-Identifier: Apache-2.0
"""A dot follows your hand, having been taught by you, and survives you covering the camera.

Run from the repository root:

    pip install -r demo/requirements.txt
    python -m demo.pointer

See `demo/README.md` for what each behaviour demonstrates and what is honest about the
accuracy. Nothing in `argmind/` imports this file; the dependency runs one way only.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from argmind.adapt import NormalisedLeastMeanSquares
from argmind.align import Aligner
from argmind.backbone import Backbone
from argmind.guard import Bound, Divergence, Guard
from argmind.persist import Store, fingerprint_of
from argmind.pipeline import FrameMap, Pipeline
from argmind.stream import Stream

#: Small on purpose. Random convolutional features work shallow and carry nothing deep --
#: brightness reads 0.009 at one layer and 1.119 at twenty-seven -- and a 3x3 grid is what
#: keeps *where* anything was: a position task reads 1.076 globally against 0.231 gridded.
PLAN = [("conv", 12, 5, 4), ("dw", 3, 2), ("pw", 16)]
IN_SHAPE = (72, 96, 3)          # the camera frame, resized; H then W
POOL = 3
STATE_DIR = Path(__file__).resolve().parent / "state"


def build():
    """The whole composition, and every piece of it is load-bearing here."""
    backbone = Backbone(PLAN, IN_SHAPE, dtype="f4", seed=1, pool=POOL)
    n = backbone.n_outputs + 1                       # + the constant

    head = NormalisedLeastMeanSquares(n, 2, mu=0.3, dtype="f4")
    guard = Guard(
        head,
        [
            # The head's own residual, in the units the target is in: screen fractions, so a
            # residual above 3 means it is pointing off the screen and further every step.
            Bound("residual_norm", hi=3.0),
            # Growth against recent history, which needs no units -- the check for a fault
            # whose scale nobody knows in advance.
            Divergence(ratio=6.0, floor=0.05, warmup=40),
            # The caller's own scalar, and the whole reason `set_external` exists: the
            # standard deviation of the raw frame, relative to what it was while teaching.
            # A covered camera reads 0.000 where every legitimate scene reads 0.19 -- and
            # the HEAD cannot see it, because a flat grey frame still produces a perfectly
            # plausible embedding (norm 0.60 against a reference 1.00) that the head fits
            # happily. `lo`, not `hi`: too little is the fault.
            #
            # The threshold is set from calibration rather than by eye. Two statistics were
            # tried first and neither separates: distance from a baseline embedding
            # saturates (covered 1.000, a dim room 0.877) and embedding norm overlaps
            # outright (covered 0.604, a dim room 0.578).
            Bound("external_0", lo=0.35),
        ],
        rollback_depth=8,
        checkpoint_every=15,
        recover_after=20,
        n_external=1,
    )

    features = Stream("features", dim=backbone.n_outputs, capacity=64, dtype="f4")
    taught = Stream("taught", dim=2, capacity=16, dtype="f4")
    fmap = FrameMap(features=[("features", i) for i in range(backbone.n_outputs)],
                    target=[("taught", 0), ("taught", 1)], constant=True)
    pipe = Pipeline([features, taught],
                    Aligner(reference="taught", policy="hold_last", max_stale_s=0.4),
                    guard, fmap, require_fresh_target=True)
    return backbone, head, guard, features, taught, fmap, pipe


def fingerprint(backbone):
    """Refuses a saved head restored against a different backbone or readout. The CRC cannot
    see that -- the bytes are perfectly valid, they just mean something else."""
    return fingerprint_of("pointer", str(PLAN), str(IN_SHAPE), f"pool{POOL}",
                          f"n{backbone.n_outputs}")


def draw(cv2, frame, state):
    h, w = frame.shape[:2]
    if state["pred"] is not None:
        px, py = int(state["pred"][0] * w), int(state["pred"][1] * h)
        colour = (0, 0, 255) if state["frozen"] else (0, 220, 0)
        cv2.circle(frame, (px, py), 16, colour, 3)
        cv2.circle(frame, (px, py), 2, colour, -1)
    for tx, ty in state["taught_points"][-8:]:
        cv2.drawMarker(frame, (int(tx * w), int(ty * h)), (255, 180, 0),
                       cv2.MARKER_TILTED_CROSS, 12, 1)

    lines = [
        f"taught {state['n_taught']}   updates {state['updates']}   {state['fps']:.0f} fps",
        f"frame std x{state['drift']:.2f}   residual {state['residual']:.3f}",
    ]
    if state["frozen"]:
        lines.append("FROZEN by " + (", ".join(state["frozen_by"]) or "?"))
    elif state["n_taught"] < 4:
        lines.append("hold the mouse button on yourself and move around to teach it")
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (8, 20 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, text, (8, 20 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def main(argv=None):
    # Imported here, not at module scope, so `build()` and the composition test it drives
    # need nothing but argmind. The demo's dependency must not become the library's, and a
    # test that could not run without a camera is a demo that rots.
    try:
        import cv2
    except ImportError:  # pragma: no cover - the demo's dependency, not the library's
        sys.exit("this demo needs OpenCV: pip install -r demo/requirements.txt")

    p = argparse.ArgumentParser(prog="python -m demo.pointer", description=__doc__)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--load", action="store_true", help="restore a saved head at startup")
    args = p.parse_args(argv)

    backbone, head, guard, features, taught, fmap, pipe = build()
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    store = Store(STATE_DIR)
    fp = fingerprint(backbone)
    if args.load:
        rec = store.load(guard, fingerprint=fp)
        if rec is None:
            print("nothing saved yet")
        else:
            print(f"restored a head from sequence {rec.sequence}")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit(f"cannot open camera {args.camera}")

    # Hold and drag to teach, rather than click. Measured: NLMS at 145 features reaches a
    # radial error of 0.058 after 300 updates and gets nowhere after 50, and nobody clicks
    # fifty times. Dragging supplies one label per frame for as long as the button is down.
    click = {"xy": None, "down": False}

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            click["down"] = True
        elif event == cv2.EVENT_LBUTTONUP:
            click["down"] = False
        if click["down"]:
            click["xy"] = (x, y)

    cv2.namedWindow("argmind")
    cv2.setMouseCallback("argmind", on_mouse)

    state = {"pred": None, "frozen": False, "frozen_by": (), "n_taught": 0,
             "updates": 0, "fps": 0.0, "drift": 0.0, "residual": 0.0,
             "taught_points": []}
    # Calibrated from the first second: the scale of the caller's own statistic and of the
    # backbone's output. Both are numbers the library refuses to guess.
    frame_std = None
    calibrated = False
    t0 = time.perf_counter()
    frames = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (IN_SHAPE[1], IN_SHAPE[0])).astype(np.float32) / 255.0
        emb = backbone.features(small)

        if not calibrated and frames > 15:
            backbone.output_scale = 1.0 / max(float(np.std(emb)), 1e-6)
            frame_std = float(np.std(small))
            calibrated = True
            emb = backbone.features(small)
            print(f"calibrated: output_scale {backbone.output_scale:.1f}, "
                  f"frame std {frame_std:.3f}")
        if calibrated:
            state["drift"] = float(np.std(small)) / max(frame_std, 1e-6)
            guard.set_external(0, state["drift"])

        now = time.perf_counter() - t0
        features.push(now, emb)
        if click["xy"] is not None:
            cx, cy = click["xy"]
            click["xy"] = None
            taught.push(now, [cx / w, cy / h])
            state["n_taught"] += 1
            state["taught_points"].append((cx / w, cy / h))

        report = pipe.step()
        if report.prediction is not None:
            state["pred"] = (float(report.prediction[0]), float(report.prediction[1]))
        state["frozen"] = bool(report.frozen)
        state["frozen_by"] = tuple(report.frozen_by)
        state["updates"] = guard.updates if hasattr(guard, "updates") else state["updates"]
        if report.updated:
            state["residual"] = float(np.linalg.norm(head._e))   # noqa: SLF001

        frames += 1
        if frames % 15 == 0:
            state["fps"] = frames / max(time.perf_counter() - t0, 1e-6)

        cv2.imshow("argmind", draw(cv2, frame, state))
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("f"):
            guard.freeze("by hand")
        if key == ord("t"):
            guard.thaw()
        if key == ord("r"):
            guard.reset()
            state["taught_points"].clear()
            state["n_taught"] = 0
        if key == ord("s"):
            store.save(guard, fingerprint=fp)
            print("saved")
        if key == ord("l"):
            rec = store.load(guard, fingerprint=fp)
            if rec is not None:
                print(f"restored sequence {rec.sequence}")
        if key == ord("h"):
            print(__doc__)

    store.save(guard, fingerprint=fp)
    cap.release()
    cv2.destroyAllWindows()
    print(f"saved to {STATE_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
