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
from demo.label_sources import (AppearanceMatcher, BackgroundLocator, CadenceDetector,
                                ObjectMemory)

#: Small on purpose. Random convolutional features work shallow and carry nothing deep --
#: brightness reads 0.009 at one layer and 1.119 at twenty-seven -- and a 3x3 grid is what
#: keeps *where* anything was: a position task reads 1.076 globally against 0.231 gridded.
PLAN = [("conv", 12, 5, 4), ("dw", 3, 2), ("pw", 16)]
IN_SHAPE = (72, 96, 3)          # the camera frame, resized; H then W
POOL = 3
STATE_DIR = Path(__file__).resolve().parent / "state"
#: A `Store` takes a base PATH, not a directory -- its slots are `<path>.0` and
#: `<path>.1`. Its docstring says so and the first version of this file passed the
#: directory, so the slots landed as `demo/state.0` NEXT TO it while the demo printed
#: "saved to demo/state". It saved correctly and told the user the wrong place.
STATE_PATH = STATE_DIR / "head"
#: The appearance window around a detected object, in pixels of the resized frame. It has to be
#: big enough to contain an object and small enough not to contain its neighbour -- and it carries
#: POSITION as well as appearance whenever it is clamped at a frame edge, which is measured: two
#: identical objects at different heights are still told apart 74.3% of the time through their
#: crops, falling to 50.6% (chance) when they sit at the same height.
CROP = 20


def should_save(updates: int, restored: bool) -> bool:
    """Is there anything worth writing over the saved head?

    **No, if this session learned nothing.** Found by running the demo when the camera was still
    held by another process: `cap.read()` failed on the very first frame, the loop broke, and the
    demo saved an *untrained* head over the trained one and printed "saved" -- exit code 0, looking
    exactly like a normal quit. A camera glitch at start-up destroyed the model.

    Same family as the earlier bug where it printed "saved" and wrote to the wrong place: the
    dangerous failures here are the ones that report success.

    `restored` is accepted and deliberately does not change the answer: re-saving a freshly loaded
    head writes the same bytes and buys nothing, and taking it into account would mean a session
    that loaded and then learned nothing still spends an erase cycle. It is a parameter so the
    reasoning is visible rather than absent.
    """
    return updates > 0


def crop_at(frame, point):
    """The appearance window around `point`, clamped to the frame."""
    h, w = frame.shape[0], frame.shape[1]
    x = int(round(point[0] * (w - 1)))
    y = int(round(point[1] * (h - 1)))
    half = CROP // 2
    x0 = max(0, min(w - CROP, x - half))
    y0 = max(0, min(h - CROP, y - half))
    return frame[y0:y0 + CROP, x0:x0 + CROP, :]


def build(objects: int = 1):
    """The whole composition, and every piece of it is load-bearing here.

    `objects` is how many things to track: the head gets `2 * objects` outputs, and the labels
    arrive in a **declared order** (leftmost first), because a head with 2k outputs is
    permutation-symmetric and learns the mean of the objects if the order wanders. Measured over
    two movers: sorted labels reach 0.129 against a centre-guess 0.312, and labels in whatever
    order the detector found them reach 0.256 — which is the centre guess with extra steps. See
    `demo/label_sources.py` for the table and for what this does *not* buy, which is identity
    through a crossing.
    """
    if objects < 1:
        raise ValueError(f"objects must be at least 1, got {objects}")
    backbone = Backbone(PLAN, IN_SHAPE, dtype="f4", seed=1, pool=POOL)
    n = backbone.n_outputs + 1                       # + the constant

    head = NormalisedLeastMeanSquares(n, 2 * objects, mu=0.3, dtype="f4")
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
    taught = Stream("taught", dim=2 * objects, capacity=16, dtype="f4")
    fmap = FrameMap(features=[("features", i) for i in range(backbone.n_outputs)],
                    target=[("taught", i) for i in range(2 * objects)], constant=True)
    # The reference is the CAMERA, not the labels, and getting this wrong was the demo's
    # third caller-side bug: with `reference="taught"` a frame exists only at a label
    # instant, so the dot moved while dragging and never otherwise -- 0 predictions out of
    # 140 unlabelled frames, measured. The library's own recorded answer is this one:
    # `hold_last` plus the fastest stream as the reference, so `step()` waits for nothing
    # and output latency is the caller's call cadence. Refusing to *learn* is not a reason
    # to stop *predicting*.
    #
    # `require_fresh_target` is what stops the held label being relearned once per camera
    # frame, and it is not optional here: measured on the same data, 0.085 radial error with
    # it against 0.238 without. It compares the target sample's own timestamp, so it needs
    # no threshold -- `max_stale_s` cannot do this job, being a bound on time where the
    # question is identity.
    pipe = Pipeline([features, taught],
                    Aligner(reference="features", policy="hold_last", max_stale_s=0.4),
                    guard, fmap, require_fresh_target=True)
    return backbone, head, guard, features, taught, fmap, pipe


def fingerprint(backbone, objects: int = 1):
    """Refuses a saved head restored against a different backbone or readout. The CRC cannot
    see that -- the bytes are perfectly valid, they just mean something else.

    `objects` is in it because it changes what the outputs *mean*, not only how many there are:
    a two-object block restored into a one-object head would put the leftmost object's y into
    the x slot of nothing.
    """
    return fingerprint_of("pointer", str(PLAN), str(IN_SHAPE), f"pool{POOL}",
                          f"n{backbone.n_outputs}", f"obj{objects}")


def draw(cv2, frame, state):
    h, w = frame.shape[:2]
    if state["pred"] is not None:
        colour = (0, 0, 255) if state["frozen"] else (0, 220, 0)
        # One circle per tracked object, labelled by SLOT rather than by identity: slot 0 is
        # "leftmost", and that is what the head can learn. It is not "the object you were
        # looking at a moment ago" -- see demo/label_sources.py.
        pred = state["pred"]
        for i in range(len(pred) // 2):
            px, py = int(pred[2 * i] * w), int(pred[2 * i + 1] * h)
            cv2.circle(frame, (px, py), 16, colour, 3)
            cv2.circle(frame, (px, py), 2, colour, -1)
            # The slot number, and the remembered id it is currently matched to. They are
            # different things: the slot is a fixed budget, the id is the growing vocabulary.
            tag = str(i)
            ids = state.get("slot_ids") or []
            if i < len(ids):
                tag = f"{i}:#{ids[i]}"
            if len(pred) > 2 or state.get("remember"):
                cv2.putText(frame, tag, (px + 20, py + 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, colour, 2, cv2.LINE_AA)
    for tx, ty in state["taught_points"][-8:]:
        cv2.drawMarker(frame, (int(tx * w), int(ty * h)), (255, 180, 0),
                       cv2.MARKER_TILTED_CROSS, 12, 1)

    if state.get("minted"):
        # A new label was just minted. Worth showing, because it is the moment the vocabulary
        # grew and it is otherwise invisible.
        cv2.putText(frame, "NEW LABEL #" + ", #".join(str(i) for i in state["minted"]),
                    (8, frame.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 200, 255), 2, cv2.LINE_AA)
    if state.get("detector_ran"):
        # Where the automatic labels just landed, so the two sources are distinguishable on
        # screen rather than both arriving as an unexplained jump in the dots.
        for px, py in state["taught_points"][-state.get("objects", 1):]:
            cv2.circle(frame, (int(px * w), int(py * h)), 9, (0, 200, 255), 2)

    lines = [
        f"{state['how']}{'+id' if state['identify'] else ''} x{state['objects']}"
        f"{'   vocab ' + str(state['vocab']) if state['remember'] else ''}   "
        f"auto {state['n_auto']}   mouse {state['n_taught']}   "
        f"updates {state['updates']}   {state['fps']:.0f} fps",
        f"frame std x{state['drift']:.2f}   residual {state['residual']:.3f}",
    ]
    if state["frozen"]:
        lines.append("FROZEN by " + (", ".join(state["frozen_by"]) or "?"))
    elif state["learning"]:
        lines.append("LEARNING")
    elif state["teach"] == "mouse" and state["n_taught"] < 4:
        lines.append("hold the mouse button on yourself and move around to teach it")
    else:
        # The answer to "do I have to keep teaching it?" -- with `--teach mouse`, yes; with a
        # detector wired in, the input teaches it and this line says which.
        lines.append("predicting only, not learning - move, or hold the button to teach")
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
    p.add_argument("--teach", choices=("both", "auto", "mouse"), default="both",
                   help="where labels come from: an automatic detector, your mouse, or both")
    p.add_argument("--objects", type=int, default=1, metavar="K",
                   help="how many moving things to track. The head gets 2K outputs and the "
                        "labels arrive leftmost-first: it learns ROLES, not identities, and "
                        "cannot follow which object is which through a crossing")
    p.add_argument("--locate", choices=("motion", "background"), default=None,
                   help="how the automatic labels find things. `motion` differences consecutive "
                        "frames; `background` differences against a running background estimate. "
                        "Defaults to background above one object, because a frame difference is a "
                        "temporal derivative and leaves a DIPOLE -- measured, two peaks land one "
                        "per object 20.3%% of the time against 96.6%% for background")
    p.add_argument("--identify", action="store_true",
                   help="assign the slots by APPEARANCE instead of left-to-right. Tracks identity "
                        "through a crossing (0.041 against a centre-guess 0.222) where the sort "
                        "rule cannot (0.234). Needs the objects to LOOK different: with identical "
                        "ones the assignment falls to chance, measured 50.6%%")
    p.add_argument("--remember", action="store_true",
                   help="keep a growing dictionary of object appearances and show it. Every "
                        "detection gets an id; new ones are minted when nothing stored matches. "
                        "The vocabulary is unbounded, the tracked slots are not")
    p.add_argument("--tol", type=float, default=ObjectMemory.DEFAULT_TOL, metavar="D",
                   help=f"how close a detection must be to a remembered object to be it "
                        f"(default {ObjectMemory.DEFAULT_TOL}). Lower over-segments -- more ids, "
                        "each purer; higher loses objects into each other, which is worse")
    p.add_argument("--detect-every", type=int, default=10, metavar="N",
                   help="run the automatic detector every N frames (default 10). This is a "
                        "declaration of what the caller can afford, and it is what the learned "
                        "head is covering; N=1 needs no learned head at all")
    args = p.parse_args(argv)

    n_objects = args.objects
    if n_objects > 1 and args.teach == "mouse":
        sys.exit("--objects > 1 needs the automatic detector: one mouse position cannot say "
                 "where two objects are. Use --teach auto or --teach both.")
    backbone, head, guard, features, taught, fmap, pipe = build(n_objects)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    store = Store(STATE_PATH)
    fp = fingerprint(backbone, n_objects)
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

    # `background` above one object is not a preference: a frame difference is a temporal
    # derivative, so a mover leaves a lobe where it was and one where it is, and asking for two
    # peaks returns two lobes of the same object 80% of the time. Measured 20.3% correct pairing
    # against 96.6%. No suppression radius fixes it here -- the dipole spacing (~17 px for a blob
    # of sigma 8.6) and the closest approach of the two objects (~21.6 px) leave no window.
    how = args.locate or ("background" if n_objects > 1 else "motion")
    if args.teach == "mouse":
        detector = None
    elif how == "background":
        detector = BackgroundLocator(n_objects, rate=0.05)
    else:
        detector = CadenceDetector(args.detect_every, k=n_objects)

    # Both `--identify` and `--remember` describe crops, so the describer is built once for
    # either. It was built inside each branch at first, with the second branch testing `dir()` for
    # a name the first may or may not have bound -- a construction that works and is a trap.
    describe = None
    if args.identify or args.remember:
        if how != "background":
            sys.exit("--identify and --remember need --locate background: appearance is only as "
                     "good as the crops it describes, and motion differencing puts those on the "
                     "leading and trailing edges of one object rather than on two objects.")
        crop_backbone = Backbone(PLAN, (CROP, CROP, 3), dtype="f4", seed=1, pool=2)

        def describe(frame, point, _bb=crop_backbone):
            v = _bb.features(crop_at(frame, point)).astype(np.float64)
            n = float(np.linalg.norm(v))
            return v / n if n > 0 else v

    matcher = AppearanceMatcher(n_objects, describe) if args.identify else None
    memory = ObjectMemory(args.tol, max_ids=64) if args.remember else None
    state = {"pred": None, "frozen": False, "frozen_by": (), "n_taught": 0,
             "updates": 0, "fps": 0.0, "drift": 0.0, "residual": 0.0,
             "taught_points": [], "learning": False, "last_update_frame": -999,
             "n_auto": 0, "detector_ran": False, "teach": args.teach,
             "every": args.detect_every, "objects": n_objects, "how": how,
             "identify": args.identify, "vocab": 0, "slot_ids": [], "minted": [],
             "remember": args.remember}
    # Calibrated from the first second: the scale of the caller's own statistic and of the
    # backbone's output. Both are numbers the library refuses to guess.
    frame_std = None
    calibrated = False
    camera_failed = False
    t0 = time.perf_counter()
    frames = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            # A dead camera and a user pressing `q` used to leave by the same door, so a
            # start-up failure looked like a normal quit -- and then overwrote the saved head.
            camera_failed = True
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

        # Labels out of the INPUT, on a declared cadence. The detector is authoritative and
        # unavailable between runs, which is what gives the head something to do -- and the
        # blindness has to be independent of the target's motion or holding the last detection
        # covers it. Measured: a motion-energy detector, which is blind exactly when nothing
        # moves, LOSES to holding (0.87x); this one wins (1.26-1.42x over cadences 5 to 15).
        # And running a detector every frame beats both at 0.010 against the head's ~0.08, so
        # this arrangement is about the detector being unaffordable, not about accuracy.
        auto = detector(small) if detector is not None else None
        if auto is not None and matcher is not None:
            # Slots by appearance rather than by position. This is what makes IDENTITY a function
            # of the current frame -- "which object is which" otherwise depends on which it was,
            # and DESIGN.md's *What it is not* says no instantaneous basis represents a history.
            auto = matcher(small, auto)
        if auto is not None and memory is not None:
            # One id per detection, minted when nothing remembered is close enough. This is the
            # vocabulary growing in front of you; the head's slots are a separate, fixed budget.
            seen = [memory.observe(describe(small, p)) for p in auto]
            state["slot_ids"] = [i for i, _ in seen]
            state["minted"] = [i for i, m in seen if m]
            state["vocab"] = len(memory)
        state["detector_ran"] = auto is not None
        if auto is not None:
            # Sorted leftmost-first by the detector, and flattened in that order. The order is
            # the whole reason this works at all: a head with 2k outputs is permutation-
            # symmetric, and labels in whatever order they were found reach 0.256 against a
            # centre-guess 0.312 -- i.e. learn nothing.
            taught.push(now, [c for xy in auto for c in xy])
            state["n_auto"] += 1
            state["taught_points"].extend(auto)
        if click["xy"] is not None:
            cx, cy = click["xy"]
            click["xy"] = None
            # Pushed a hair later so a mouse label always wins the instant it shares with an
            # automatic one: `hold_last` takes the newest at or before the reference, and a
            # correction the user made by hand should not be the one that gets dropped.
            # One mouse position teaches every slot the same point, which is only sensible
            # at --objects 1; with more, the automatic detector is the label source and the
            # mouse is a nudge. Stated rather than silently doing something clever.
            taught.push(now + 1e-6, [cx / w, cy / h] * n_objects)
            state["n_taught"] += 1
            state["taught_points"].append((cx / w, cy / h))

        report = pipe.step()
        if report.prediction is not None:
            state["pred"] = tuple(float(v) for v in report.prediction)
        state["frozen"] = bool(report.frozen)
        state["frozen_by"] = tuple(report.frozen_by)
        # Counted from the reports, because `Guard` exposes no update counter. This line used
        # to read `guard.updates if hasattr(guard, "updates") else state["updates"]`, and
        # `Guard` has no such attribute -- so the number on screen was hardwired to 0 for the
        # life of the demo, and it is the one number that answers "is it learning right now".
        # A defensive `hasattr` around a field nobody checked is how that stays invisible.
        if report.updated:
            state["updates"] += 1
            state["last_update_frame"] = frames
            state["residual"] = float(np.linalg.norm(head._e))   # noqa: SLF001
        # Learning is not continuous and the demo must say so: the head adapts only on a frame
        # that carries a NEW label, which means only while the button is held. There is no
        # self-supervision here, and that is measured rather than unfinished -- predicting the
        # next embedding from an untrained backbone learns nothing (a learned map's diagonal
        # reads 0.004 against an identity's 1.0, and frozen held-out error is 2-7x worse than
        # simply repeating the last value).
        state["learning"] = frames - state["last_update_frame"] < 8

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

    cap.release()
    cv2.destroyAllWindows()
    if should_save(state["updates"], args.load):
        store.save(guard, fingerprint=fp)
        print(f"saved to {store.slot_path(0)} and .1")
    else:
        print(f"nothing learned this session ({state['updates']} updates), so the saved head "
              f"at {store.slot_path(0)} is left alone")
    if camera_failed:
        print(f"camera {args.camera} stopped returning frames. If another process is holding it, "
              "close that first; OpenCV reports this on stderr and would otherwise look like a "
              "normal exit.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
