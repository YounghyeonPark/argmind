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
from argmind.memory import ExemplarMemory
from argmind.recognise import RecognitionLayer
from argmind.guard import Bound, Divergence, Guard
from argmind.persist import Store, fingerprint_of
from argmind.pipeline import FrameMap, Pipeline
from argmind.stack import Stack
from argmind.stream import Stream
from demo.label_sources import (RecogniseLocator,AppearanceMatcher, BackgroundLocator, CadenceDetector,
                                FrameSegmenter, ObjectMemory, RegionTracker, partition,
                                pick_regions, region_descriptor, ring_contrast, split_regions)
from demo.map_head import Pyramid, block_mean, map_iou, map_target

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
#: The learned backbone's own record, under `--input learned`. A separate two-slot store rather than
#: one combined block, because the two have different shapes and a caller switching `--input` should
#: be refused by the fingerprint rather than handed a block that unpacks and means something else.
FEATURES_PATH = STATE_DIR / "features"
#: The segmentation map the head emits under `--output map`, deliberately coarser than the frame:
#: a segmentation does not need per-pixel precision to say where something is, and the head's
#: parameter count is `n_features x n_outputs`.
MAP_H, MAP_W = 9, 12

#: `--locate recognise`. The crop side and the coarse step are the two that were measured: identity
#: across an absence holds at a 2-pixel effective centre and is lost at 8, and a coarse pass with
#: local refinement reaches it at 147 crops a frame instead of 925.
RECOGNISE_CROP, RECOGNISE_STEP = 24, 8
RECOGNISE_SLOTS, RECOGNISE_K = 16, 16
#: A cosine similarity, and the margin on rarity below which nothing is admitted. Both measured
#: rather than guessed: `median - min` of the candidates' mean similarity is 0.0007-0.0011 on an
#: empty wall against 0.126-0.143 with an object present.
RECOGNISE_THRESHOLD, RECOGNISE_MARGIN = 0.85, 0.02
#: How often the pyramid is blanked while learning. **The only thing measured to survive a blackout**
#: -- holding the pyramid and resetting it both fail identically, because the memory's problem after
#: a gap is that it is about a world that has moved on, not that it was corrupted. It costs peak
#: accuracy: 0.98 healthy without it against ~0.38, for 0.00 after a blackout against ~0.54.
MAP_DROPOUT = 0.15


def map_memory_width(out_h=None, out_w=None) -> int:
    """How many inputs the pyramid contributes, so `build` and `main` agree on the width without
    one of them constructing the other's object."""
    return Pyramid(out_h or MAP_H, out_w or MAP_W).width


def map_basis(frame_features, pyramid, *, blank=False):
    """The map head's whole input: this frame's features, then the pyramid of past output maps.

    Module-level for the reason seven other things in this file are. `blank` is the training-time
    dropout: it zeroes the memory rather than removing it, so the width never changes -- a head whose
    input width moved between frames would be a different model, and `FrameMap` would refuse it.
    """
    mem = np.zeros(pyramid.width) if blank else pyramid.features()
    return np.concatenate([np.asarray(frame_features, dtype=np.float64), mem])


def mouse_target(x: float, y: float, n_objects: int, output: str):
    """What a mouse click teaches, or `None` if it teaches nothing. Module-level so a test can
    reach it -- the eighth block in this `main()` to be moved out for that reason, and this one was
    found by the demo **crashing**: the automatic branch was updated for `--output map` and this one
    was not, so a click pushed 6 values into a 108-wide stream.

    **In map mode a click teaches nothing, and that is a refusal rather than an omission.** The
    target is a full foreground mask over the whole frame, so making one from a single point would
    assert that every other cell is background -- a strong claim, and mostly a false one whenever
    anything else is in the scene. The same shape as this demo's recorded finding that a difference
    map with two movers puts its centroid between them: a point cannot stand in for a field.

    In points mode it teaches every slot the same point, which is only sensible at `--objects 1`;
    with more, the detector is the label source and the mouse is a nudge. Stated rather than
    silently doing something clever.
    """
    if output == "map":
        return None
    return [x, y] * int(n_objects)


def teach_from_click(taught, t: float, x: float, y: float, n_objects: int, output: str) -> bool:
    """Push what a click teaches, if it teaches anything. Returns whether it pushed.

    The *push* lives here and not only the decision, so the demo and the headless harness go
    through the same code. `demo/test_pointer.py`'s `drive` used to build the vector itself, which
    is the "a test that reimplements the assembly instead of calling it" defect this package has
    already been caught by once.

    **What this still cannot cover, stated rather than implied**: a change to `main()`'s call site
    that bypasses this function entirely. A mutant doing exactly that survives the suite, and the
    original bug was found by *running the demo*, not by a test. The demo's own failures have been
    the ones that report success or crash on a path no fixture visits.
    """
    target = mouse_target(x, y, n_objects, output)
    if target is None:
        return False
    taught.push(t, target)
    return True


def map_teacher(field):
    """The target map, from whatever produced the label field. Foreground is any labelled region.

    **This is the teacher the decision of 2026-08-14 wants withdrawn.** It is the classical
    labeller -- k-means, connected components, contrast -- and the head learns to reproduce it. The
    measurement that matters is what survives when it is taken away, which is why this is one small
    function with a name rather than an inline expression.
    """
    return map_target(np.asarray(field) > 0, MAP_H, MAP_W)
#: The appearance window around a detected object, in pixels of the resized frame. It has to be
#: big enough to contain an object and small enough not to contain its neighbour -- and it carries
#: POSITION as well as appearance whenever it is clamped at a frame edge, which is measured: two
#: identical objects at different heights are still told apart 74.3% of the time through their
#: crops, falling to 50.6% (chance) when they sit at the same height.
CROP = 20
#: Block-mean downsample factor for `--input raw`. 12 turns a 72x96 frame into 6x8, and in
#: colour that is 6*8*3 = 144 numbers -- **exactly the backbone's width**, which makes the
#: comparison a like-for-like one rather than a trade.
RAW_FACTOR = 12


def raw_features(frame, factor=RAW_FACTOR, colour=True):
    """A block-mean downsample of the frame, flattened. **In colour by default.**

    **Measured better than the random backbone on every axis this demo exercises**, which is not
    what I expected. Position, one object, perfect labels:

    | input | features | head state | ms/frame | error | vs centre |
    |---|---|---|---|---|---|
    | backbone | 145 | 1.1 KB | 0.94 | 0.064 | 4.55x |
    | raw /8 grey | **109** | **0.9 KB** | **0.42** | **0.063** | **4.58x** |

    Frozen, which separates tracking from generalising, raw reaches **0.001 (215.8x a centre
    guess)** against the backbone's 0.027 (10.6x).

    **Colour is not optional, and the first version of this function threw it away.** The demo's
    appearance descriptor had already measured colour doing all the work -- raw crop RGB separates
    two objects at 0.096 against raw crop *grey* at 0.984, which is useless -- so a greyscale head
    input meant the head could never express identity while the caller's memory could. Measured on
    three objects, frozen, with slot 0 always the same object whatever its position:

    | input | features | roles | identity |
    |---|---|---|---|
    | backbone | 145 | 4.61x | 4.01x |
    | raw /8 grey | 109 | **12.09x** | **1.71x** |
    | raw /8 RGB | 325 | 10.09x | **133.60x** |
    | **raw /12 RGB** | **145** | 8.34x | **118.87x** |

    Grey is best for *roles*, where position is the whole question and colour is one more thing to
    fit. It nearly fails at identity. And `/12` in colour is 144 numbers -- the backbone's own
    width -- for 118.87x against the backbone's 4.01x, so the default is like-for-like and thirty
    times better at the harder task.

    The reason position works at all is mechanical: `sum_i w_i * pixel_i` with weights proportional
    to a pixel's coordinate *is* the centroid times the total brightness, so a linear head computes
    it almost exactly from pixels. The backbone pools to a 3x3 grid and mixes channels randomly,
    which for this task can only lose information.

    **What a backbone buys is invariance, and a random one does not buy it.** Trained under one
    condition and frozen, then tested under another: brightness x1.6 gives 0.9x for the backbone
    against 0.7x for raw, a +0.25 offset 1.0x against 0.8x -- marginally better and *both at a
    centre guess*, i.e. both failed. Under clutter raw wins outright, 1.6x to 1.4x. A trained
    backbone with batch normalisation is a different proposition; this one is random.
    """
    a = np.asarray(frame, dtype=np.float32)
    if a.ndim == 2:
        a = a[:, :, None]
    if not colour:
        a = a.mean(axis=2, keepdims=True)
    h, w = a.shape[0] // factor * factor, a.shape[1] // factor * factor
    a = a[:h, :w]
    return a.reshape(h // factor, factor, w // factor, factor,
                     a.shape[2]).mean(axis=(1, 3)).ravel()


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


#: Leak rates for the output-feedback bank: short, medium and long term. Three, because one is
#: measured to be *worse than persistence* free-running and three is where it starts paying.
FEEDBACK_RATES = (0.5, 0.15, 0.03)


def basis(small, *, backbone, source, colour=True, scale=1.0, output_memory=None):
    """The head's whole basis for one frame: the frame's features, then the feedback bank.

    **Module-level on purpose.** This lived as a closure inside `main()`, and the tests that were
    supposed to cover it reimplemented the same concatenation locally -- so two deliberate bugs in
    the real function survived the suite. Same shape as `select_newest` being compiled and called
    by no binary, and as `drive`'s `teach_every` that no caller ever passed. A test that rebuilds
    the logic it is checking is not a test.

    The bank belongs *inside* here rather than at the call site, and that is not tidiness: the
    calibration branch recomputes the basis after setting the scale, so anything concatenated
    outside is silently dropped on that one frame. It cost `expected shape (162,), got (144,)` on
    the first calibrated frame -- which is the same defect a single extraction path was introduced
    to fix, arriving a second time by the same door.
    """
    if source == "learned":
        #: The **frame itself**, because under this source the backbone lives inside the adapter
        #: (`Stack`) and its input is the basis. Extracting the embedding here would compute the
        #: forward pass twice and hand the head a vector it cannot descend through.
        base = np.asarray(small, dtype=backbone.dtype).ravel()
    elif source == "backbone":
        base = backbone.features(small)
    else:
        base = raw_features(small, colour=colour) * scale
    if output_memory is None:
        return base
    return np.concatenate([base, output_memory.features().astype(base.dtype)])


#: Default step for the feature learner. Small, because the head is adapting underneath it: the
#: features move the ground the head stands on, and 0.05 was measured to oscillate.
FEATURE_LR = 0.02


def head_theta(head):
    """The head's coefficient matrix, `(n_features, n_outputs)`, with the layout checked.

    Both linear adapters lay `theta` out row-major over features and put it first in the state
    block, which is what makes this a reshape rather than an accessor. It is *asserted* rather than
    assumed, because a transposed read would still produce a gradient -- one that trains something
    else, silently. The tolerance is f32's over a few hundred terms; 1e-6 was too tight and that was
    my error rather than a layout error.
    """
    n, m = head.n_features, head.n_outputs
    theta = np.asarray(head.state[:n * m]).reshape(n, m)
    probe = np.arange(1.0, n + 1.0, dtype=np.float64) / n
    if not np.allclose(probe @ theta, np.asarray(head.predict(probe)), rtol=1e-3, atol=1e-4):
        raise ValueError("the head's theta is not (n_features, n_outputs) row-major")
    return theta


def learn_features(backbone, small, theta, residual, learning_rate=FEATURE_LR):
    """**No longer the demo's path (2026-08-17).** `--input learned` builds a `Stack`, which owns
    the chain rule, so `main()` calls this nowhere. It survives as the *reference implementation*
    the comparison fixtures in `demo/test_learned_input.py` measure learned-against-frozen with,
    and `test_nothing_in_the_demo_loop_calls_the_old_chain_rule` asserts that it stays out of the
    loop — otherwise this would be the "compiled and called by no binary" defect this repository
    already found once in `select_newest`.

    Backpropagate the head's residual into the backbone. Returns how far the weights moved.

    **This is the whole of what makes the features learned rather than frozen**, and it is a
    module-level function for the reason six other things in this file are: the version that lived
    inside `main()`'s loop could not be tested, and every one of those turned out to be wrong.

    `residual` is `prediction - target`, so `dL/dfeatures` for the squared error is
    `2 * theta[:-1] @ residual` -- the last row of `theta` belongs to the constant, which the
    backbone does not produce.

    **Measured, and the trade is real in both directions.** Against the same demo with the backbone
    frozen, on a three-object tracking fixture with an RLS head:

    | source | tracking error | frozen, then the scene brightens x1.6 | ms/frame |
    |---|---|---|---|
    | raw pixels | **0.0010** | 0.6506 — **0.78x, worse than guessing** | 0.07 |
    | frozen random backbone | 0.0241 | **0.0816 — 6.19x** | 0.68 |
    | **learned backbone** | **0.0163** | 0.1658 — 3.05x | 0.70 + **2.42 backward** |

    So learning the features **beats the frozen backbone at tracking** and **costs generalisation**:
    the learned stack fits the condition it was trained under more tightly than a random projection
    does, and transfers worse when the light changes. Raw pixels win at tracking by a wide margin and
    fail the condition change outright. None of the three dominates, which is why this is a flag.

    The caller gates it on the head having actually updated -- which also means it stops while the
    guard is frozen, since a frozen guard applies nothing. That is the behaviour you want: a fault is
    exactly when you do not wish to push it into the features.
    """
    if learning_rate <= 0.0:
        raise ValueError(f"learning_rate must be positive, got {learning_rate}")
    theta = np.asarray(theta)
    resid = np.asarray(residual, dtype=np.float64)
    if theta.shape[0] != backbone.n_outputs + 1:
        raise ValueError(
            f"theta has {theta.shape[0]} rows for a backbone of {backbone.n_outputs} features "
            "plus the constant")
    if resid.shape != (theta.shape[1],):
        raise ValueError(f"expected a residual of {(theta.shape[1],)}, got {resid.shape}")
    d_feat = 2.0 * (theta[:-1] @ resid)
    dws, dbs = backbone.backward(small, d_feat.astype(backbone.dtype))
    return backbone.apply_gradients(dws, dbs, learning_rate)


def calibration_scales(source, embedding, n_frame):
    """Where the measured feature scale has to be applied, per source. Returns
    `(backbone_output_scale, basis_scale)` — exactly one of them is ever not 1.0.

    **Scaling the input is not optional and this project measured why**: `p0` has units, so the RLS
    gain is `p0·x/(1 + p0·‖x‖²)`, and an untrained stack's embedding at a magnitude of 6.8e-4 puts
    the data term eight orders of magnitude below the default prior — every head then scores about
    1.0 against a constant predictor until it is scaled.

    Module level because the version inside `main()`'s loop tested `source == "backbone"` and I added
    a third source that also reads the backbone. Under `--input learned` the scale was therefore
    computed, printed, and **applied nowhere**: `Backbone.output_scale` stayed 1.0 and `basis()`
    ignores its `scale` argument on the backbone branch. Seventh piece of logic in this file to need
    moving out of that loop, and the same "computed and never used" shape the project has now found
    in `gain_norm` twice and in `output_scale`'s own backward pass.
    """
    sd = float(np.std(np.asarray(embedding)[:n_frame]))
    scale = 1.0 / max(sd, 1e-6)
    if source in ("backbone", "learned"):
        return scale, 1.0                    # the backbone applies it inside `features()`
    return 1.0, scale                        # raw features are multiplied at the call site


def feature_step(base_lr, output_scale):
    """The feature learning rate, corrected for the calibration applied to the same features.

    **`Backbone.output_scale` multiplies the backward pass as well as the forward one**, so a
    calibration that makes the features unit-scale also multiplies every gradient into the stack by
    the same factor. Leaving the rate alone then takes an effective step of 0.02 x 29 and the
    learning oscillates. Measured, three objects, RLS head, 900 updates:

    | calibration | feature lr | error | vs a centre guess |
    |---|---|---|---|
    | not applied | 0.02 | 0.0171 | 31.00x |
    | applied, rate unchanged | 0.02 | 0.2920 | **1.81x** |
    | **applied, rate / output_scale** | 0.00068 | **0.0146** | **36.21x** |

    So the two halves belong together: applying the calibration without this correction is worse than
    not calibrating at all, and applying both is better than either. Dividing by `output_scale`
    squared is marginally better still (37.51x) and is *not* done, because one factor is the one the
    mechanism explains and the second would be fitted.

    This is the rule "scale the input, it is not optional" meeting a case it was not measured on: it
    was derived for a FROZEN backbone, where nothing propagates back through the scale.
    """
    if base_lr <= 0.0:
        raise ValueError(f"base_lr must be positive, got {base_lr}")
    return base_lr / max(float(output_scale), 1e-9)


def frame_width(backbone, source, colour=True) -> int:
    """How much of a basis is the frame, as opposed to the feedback bank.

    Calibration needs this: the bank starts at zero, so scaling from the whole vector would be
    scaling from something a third of which is padding.

    **`learned` is the frame itself**, not the embedding: under that source the backbone lives
    inside a `Stack`, so the adapter's input is what the *layer* consumes. This used to return
    `backbone.n_outputs` for it and is now wrong by 144x if it does.
    """
    if source == "learned":
        return int(np.prod(IN_SHAPE))
    return (backbone.n_outputs if source == "backbone" else raw_width(colour=colour))


def feedback_width(objects: int, rates=FEEDBACK_RATES) -> int:
    """How many extra basis terms an `OutputMemory` contributes, so `build` and `main` agree
    without one of them constructing the other's object."""
    return len(rates) * 2 * objects


class OutputMemory:
    """A bank of leaky integrators over the head's own past output, for the head's own input.

    **Why this is allowed at all.** DESIGN.md says no choice of *instantaneous* features represents
    a history-dependent quantity, and which object this is depends on which it was. That is a
    statement about the BASIS, not the model class: `y_t = f(x_t, y_{t-1}, ...)` is still linear in
    the parameters, and "the caller supplies the basis" is the licence to put lagged outputs in it.

    **What it buys, measured on a fixture where nothing else can work.** Two identical objects
    orbiting 180 degrees apart: swapping their labels produces a **bit-identical** image, so no
    instantaneous property whatever can say which is which. Identity there is purely historical.

    | basis | error | x centre | x persistence |
    |---|---|---|---|
    | frame only | 0.2504 | **1.00x** | 0.04x |
    | + previous output, teacher-forced | 0.0002 | 1572x | 70.5x |
    | + previous output, free-running | 0.0195 | 12.8x | **0.57x** |
    | + three lags, free-running | 0.0035 | 71.0x | 3.18x |
    | + **this bank**, free-running | 0.0018 | **141.7x** | **6.36x** |

    Frame-only reads exactly a centre guess, which is what says the question is real. And the
    multi-timescale part is not decoration: **one lag free-running is worse than simply copying the
    previous true position** (0.57x), because its own errors compound. Teacher forcing reads 1572x
    and is the ceiling, not a shippable number -- the figure that ships is the free-running one.

    **And it cannot recover from a dropout, which is the cost.** Sixty blind frames take the error
    from 0.0009 to 0.70, and nine hundred frames later it is still **0.46** -- it never comes back.
    That is not divergence (the prediction stays bounded at 1.24); it is a **stable wrong fixed
    point**, and it follows from the same property that made feedback necessary: a frame that
    cannot disambiguate also cannot correct. More time constants do not help -- a four-rate bank
    behaves identically to three.

    **So this needs an external re-anchor and the guard is not it**: the head is healthy and its
    residual is low, because it fits the wrong assignment well. What re-anchors it is appearance,
    when the objects differ -- which is what `ObjectMemory` and `AppearanceMatcher` are. Feedback
    carries identity through ambiguity; appearance recovers it after a break. Neither alone.
    """

    __slots__ = ("rates", "n_outputs", "_bank")

    def __init__(self, n_outputs: int, rates=FEEDBACK_RATES) -> None:
        if n_outputs < 1:
            raise ValueError(f"n_outputs must be at least 1, got {n_outputs}")
        if not rates or any(not 0.0 < r <= 1.0 for r in rates):
            raise ValueError(f"every rate must be in (0, 1], got {rates}")
        self.rates = tuple(float(r) for r in rates)
        self.n_outputs = int(n_outputs)
        self.reset()

    def reset(self) -> None:
        self._bank = [np.zeros(self.n_outputs, dtype=np.float64) for _ in self.rates]

    @property
    def width(self) -> int:
        return len(self.rates) * self.n_outputs

    def features(self) -> np.ndarray:
        return np.concatenate(self._bank)

    def observe(self, y) -> None:
        """Fold an output into every band. The caller decides whether `y` is the head's own
        prediction or a supplied label -- during learning the label is the honest thing to feed,
        and the gap between the two is the teacher-forcing gap the docstring quantifies."""
        v = np.asarray(y, dtype=np.float64).ravel()
        if v.size != self.n_outputs:
            raise ValueError(f"expected {self.n_outputs} outputs, got {v.size}")
        for i, r in enumerate(self.rates):
            self._bank[i] += r * (v - self._bank[i])


def raw_width(factor=RAW_FACTOR, shape=IN_SHAPE, colour=True):
    return (shape[0] // factor) * (shape[1] // factor) * (shape[2] if colour else 1)


#: One stable BGR colour per label id, so a region keeps its colour as the vocabulary grows.
#: Deliberately not random per frame: a colour that changes is a colour that says nothing.
ID_COLOURS = ((60, 200, 255), (255, 160, 60), (120, 240, 120), (200, 120, 255),
              (80, 220, 240), (240, 120, 160), (160, 255, 200), (255, 220, 120))


def regions_for_slots(detector, frame, n, *, split=True, edges=0.95, min_frac=0.01):
    """Pick the `n` regions a `FrameSegmenter` frame offers, for the head and for the display.

    **Module level for the reason `slot_label_map` is**: this ran inside `main()`'s loop, where no
    test could reach it, and the last thing that lived there coloured every region with the wrong
    id for weeks. Four separate defects in this package have had that shape in one day.

    Returns `(region_map, ids, points)`. `region_map` is what `slot_label_map` paints from, `ids`
    is the value in it each slot came from, and `points` is what the head is taught -- always `n`
    of the last two.

    `split=False` is the older rule: clusters are regions, and the largest cluster is the
    background. It is kept because every other figure in this package was measured with it, and
    because splitting **costs** on one measured case -- two identical touching objects give one
    region, so the second slot gets a background fragment and the taught centre goes 0.053 -> 0.188.
    What splitting buys is the case it was built for: two same-coloured objects held apart, 0.213 ->
    0.000.
    """
    if not split:
        sizes = [(detector.labels == j).sum() for j in range(detector.k)]
        keep = sorted(range(detector.k), key=lambda j: sizes[j])[:n]
        centres = detector(frame) if detector.labels is None else _cluster_centres(detector)
        pairs = sorted(((j, centres[j]) for j in keep), key=lambda jp: jp[1][0])
        return detector.labels, None, [j for j, _p in pairs], [p for _j, p in pairs]
    region_map, centres, of_cluster = split_regions(detector.labels, detector.k, frame,
                                                    edges=edges, min_frac=min_frac)
    ids, points = pick_regions(detector.labels, region_map, centres, of_cluster, n)
    return region_map, of_cluster, ids, points


def follow_permutation(before, after, values):
    """Reorder `values` the same way `after` reorders `before`.

    The appearance matcher returns the same points in a different order, and whatever was indexed by
    the old order -- here, which region each slot came from -- has to follow it. Get this wrong and
    the picture colours a region with another slot's label, which is exactly the defect that made the
    ids flicker in the first place.

    **Matched by first UNUSED equal value, not by `before.index(pt)`.** A slot the frame had no
    region for carries the frame centre, so two entries can be identical, and `index` would hand both
    of them the same region. Module level because that reasoning was a comment inside `main()`'s loop
    where nothing could reach it -- the fifth thing in this package to need moving out for that
    reason in one day.
    """
    if len(before) != len(after) or len(before) != len(values):
        raise ValueError(f"{len(before)} before, {len(after)} after, {len(values)} values")
    taken = set()
    out = []
    for pt in after:
        match = next((i for i, q in enumerate(before) if i not in taken and q == pt), None)
        if match is None:
            raise ValueError(f"{pt} is not one of the points that went in: {before}")
        taken.add(match)
        out.append(values[match])
    return out


#: A region smaller than this many pixels is noise whatever its contrast. Not a fraction, because
#: it is about there being enough pixels for a mean to mean anything, not about the scene's scale.
MIN_REGION_PX = 12


def label_field(region_map, of_cluster, cluster_labels, tracker, *, min_frac=0.03, frame=None,
                contrast=0.125):
    """Every region worth painting, with a label that survives between frames.

    **The label count is not the head's slot count**, and conflating them is why the picture showed
    three regions when the scene had more. A head with 2K outputs has K slots because that is its
    output width; a picture has no such limit. This repository's own resolution of the vocabulary
    question is "unbounded vocabulary, bounded concurrency", and this is the unbounded half.

    **The labels come from splitting, and `k` is close to irrelevant.** Five objects, the head's
    slot count irrelevant: 5.0 labels at purity 1.00 over 5 minted ids, identically at k = 4, 6, 8
    and 10. Five labels for five objects come out of a scene declared with **four** clusters, because
    splitting separates the objects that share one.

    An earlier version of this docstring said raising `k` was **worse** -- 3.0 labels at purity 0.45
    over 43 ids -- and that was an artefact of the background-selection defect described above, not a
    property of `k`. Withdrawn and re-measured.

    **`min_frac` here is NOT the split's floor, and one number cannot serve both.** The split's floor
    must be low enough that every object has a region, because that is what the head is taught from:
    raising it from 0.01 to 0.03 takes the taught centre on two same-coloured objects from **0.000
    to 0.222**, losing a slot's region entirely. The picture's floor must be high enough to keep
    background fragments out: at 0.01 it paints 6.8 labels of which **44%** sit on a real object, and
    at 0.03 it paints 2.8 of which **98%** do. So there are two thresholds on one split.

    Swept, three objects: floor 0.01 paints 7.0 labels at 0.47 precision, 0.02 paints 3.5 at 0.89,
    0.03 paints 3.2 at 0.95, and 0.05 paints 3.0 at 1.00 but starts losing objects (coverage 0.98)
    and re-minting labels when a region dips under it (purity 0.83). 0.03 is a middle, not an
    optimum.

    `frame` is what lets a label survive an object hiding and coming back: with it, each region
    gets a masked appearance descriptor and `RegionTracker` can re-identify a dormant label. Without
    it the tracker is frame-to-frame only and a hidden object returns as a new label. The measurement
    for why re-identification is by appearance and never by position is in `RegionTracker`.

    **`contrast` is why an object stops going unlabelled, and the report that produced it was
    "some areas get no label and the count looks fixed".** The count was never fixed -- it already
    paints 4 labels for a declared 3 and 5 for five objects -- but a region could be dropped two
    ways: for sitting in the *background cluster*, which happens as soon as a scene has more
    appearance groups than `k` allows, and for being under the area floor. Neither is about counting.

    Four mechanisms were measured and three are refused:

    | mechanism | result |
    |---|---|
    | raise `k` | **changes nothing**: covered 0.80 at k=4, 6 and 8 alike. It separates the faint object's cluster and the area floor drops it anyway |
    | merge clusters whose centres are close | folds the small object back into the wall, covered 1.00 -> 0.80 |
    | drop the largest *region* instead of the background *cluster* | rescues the absorbed object, and destroys a noisy scene: precision 0.95 -> **0.26** |
    | **ring contrast** | covered **0.80 -> 1.00** on the failing scene, precision 0.99 -> **0.92** on a noisy one |

    So contrast is the only one that separates a real object from a fragment of the wall, because that
    is a *local* property and the others are all global. What it costs is stated rather than hidden:
    seven points of precision on a noise-carved wall, and `contrast=0` restores the old behaviour
    exactly.

    The rule: inside the background cluster a region is admitted **only** by contrast, and never if it
    is the largest piece of that cluster -- that piece is the wall. Outside it the area floor and
    contrast are *alternatives*, since a small object with contrast is an object and a large patch
    without any is not.

    A minimum *age* was measured as the alternative and is not offered: it only helps when the floor
    is too low -- precision 0.44 -> 0.85 at floor 0.01 -- and once the floor is right it is worse
    than nothing, costing coverage 1.00 -> 0.90 for precision 0.98 -> 0.90.
    """
    lab = np.asarray(region_map)
    if of_cluster is None:                       # clusters are the regions; there is nothing to add
        return lab, []
    cl = np.asarray(cluster_labels)
    #: The background is the largest cluster **in the frame**, not the largest among the clusters
    #: that happened to produce a surviving region. Restricting it to those was a real defect with
    #: a common trigger: one object on a clean wall leaves the wall's clusters with no region above
    #: the split's floor, so the object's own cluster became "the largest present" and was dropped.
    #: Measured, a lone object on a plain background got NO label on the frame where that happened.
    counts = np.bincount(cl.ravel())
    background = int(counts.argmax()) if counts.size else None
    floor = max(int(min_frac * lab.size), 1)
    sizes = {i: int((lab == i).sum()) for i in range(1, len(of_cluster) + 1)}
    in_bg = [i for i in sizes if of_cluster[i - 1] == background]
    wall = max((sizes[i] for i in in_bg), default=0)
    live, centres, areas = [], [], []
    for i in range(1, len(of_cluster) + 1):
        piece = lab == i
        n = sizes[i]
        if contrast and frame is not None and n >= MIN_REGION_PX:
            step = ring_contrast(frame, piece)
            if of_cluster[i - 1] == background:
                if step < contrast or n >= wall:
                    continue
            elif n < floor and step < contrast:
                continue
        elif of_cluster[i - 1] == background or n < floor:
            continue
        yy, xx = np.nonzero(piece)
        live.append(i)
        centres.append((float(xx.mean()) / max(lab.shape[1] - 1, 1),
                        float(yy.mean()) / max(lab.shape[0] - 1, 1)))
        areas.append(n)
    descs = None
    if frame is not None:
        descs = [region_descriptor(frame, lab == i) for i in live]
    got = tracker(centres, areas, descs)
    out = np.zeros(lab.shape, dtype=np.int32)
    for slot, i in enumerate(live):
        out[lab == i] = slot + 1
    return out, got


def _cluster_centres(detector):
    """Each cluster's centroid in frame fractions, from the labels the detector already produced."""
    lab = detector.labels
    h, w = lab.shape
    yy, xx = np.mgrid[0:h, 0:w]
    out = []
    for j in range(detector.k):
        m = lab == j
        out.append((0.5, 0.5) if not m.any() else
                   (float(xx[m].mean()) / max(w - 1, 1), float(yy[m].mean()) / max(h - 1, 1)))
    return out


def slot_label_map(cluster_labels, slot_cluster):
    """Repaint a cluster map so each kept cluster carries its SLOT number, background 0.

    **Module level because the version inside `main()` was wrong and nothing could see it.** The
    display painted cluster *n* with the id of slot *n*, and those are different orderings: the
    background is dropped by area, the survivors are re-sorted by x, and the appearance matcher
    permutes them again. The pairing was arbitrary and shifted every frame, which is exactly the
    "label id changes whenever something moves" a caller sees.

    Dropping the background to 0 rather than colouring it is the other half: a cluster map tiles
    the whole frame by construction, so painting every cluster makes the regions read as a
    partition of the picture instead of as objects.
    """
    lab = np.asarray(cluster_labels)
    out = np.zeros(lab.shape, dtype=np.int32)
    for slot, j in enumerate(slot_cluster or []):
        if j is None:                                # a slot the frame had no region for; the head
            continue                                 # still needs its two outputs, the picture does
        out[lab == j] = slot + 1                     # not need a lie
    return out


def segment_image(labels, ids=(), scale=6):
    """An RGB picture of a label map: background dark, each region in its id's colour.

    **This shows what the detector saw, not a learned segmentation**, and the distinction is
    measured rather than stylistic. The mask is real -- it is where the frame differs from its own
    background -- and the per-object split is nearest-detection assignment. Against ground truth on
    a synthetic scene, IoU is 0.76 for one object at `floor=0.40` (precision 1.00, recall 0.76) and
    falls to 0.46 at three.

    Learned per-object segmentation, by correlating a remembered descriptor against the backbone's
    feature map, was measured separately and refused: it localises the queried object 4-50% of the
    time. Nothing here claims otherwise.
    """
    lab = np.asarray(labels)
    out = np.zeros((lab.shape[0], lab.shape[1], 3), dtype=np.uint8)
    out[lab == 0] = (26, 26, 26)
    for j in range(1, int(lab.max()) + 1 if lab.size else 1):
        region = lab == j
        if not region.any():
            continue
        # Colour by the remembered id when there is one, so the picture and the vocabulary agree.
        key = ids[j - 1] if j - 1 < len(ids) else j - 1
        out[region] = ID_COLOURS[key % len(ID_COLOURS)]
    if scale > 1:
        out = np.repeat(np.repeat(out, scale, axis=0), scale, axis=1)
    return out


def build_recogniser(n_objects: int = 1):
    """`main()`'s own construction of the recognising locator, hoisted so a test can reach it.

    The **ninth** block in this `main()` to be moved out for that reason, and this one was found by
    the demo crashing with `NameError: name 'RecognitionLayer' is not defined` — the imports were
    never added, `import demo.pointer` passed because the names are only referenced inside `main()`,
    and the test for `--locate recognise` built the layer itself instead of calling this. That is the
    same "a harness that re-implements the assembly cannot see the assembly break" defect this file
    already fixed once for `teach_from_click`.
    """
    crop_bb = Backbone(PLAN, (RECOGNISE_CROP, RECOGNISE_CROP, 3), dtype="f4", seed=1, pool=POOL)
    h, w = IN_SHAPE[0], IN_SHAPE[1]
    half = RECOGNISE_CROP // 2
    centres = [(x / w, y / h)
               for y in range(half, h - half + 1, RECOGNISE_STEP)
               for x in range(half, w - half + 1, RECOGNISE_STEP)]
    return RecogniseLocator(
        RecognitionLayer(crop_bb,
                         ExemplarMemory(RECOGNISE_SLOTS, crop_bb.n_outputs, RECOGNISE_K,
                                        threshold=RECOGNISE_THRESHOLD, dtype="f4"),
                         IN_SHAPE, centres, admit_margin=RECOGNISE_MARGIN,
                         refine=(RECOGNISE_STEP, 2)),
        n_objects)


def build(objects: int = 1, source: str = "backbone", *, colour: bool = True,
          feedback: bool = False, output: str = "points"):
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
    if source not in ("backbone", "raw", "learned"):
        raise ValueError(f"source must be 'backbone', 'raw' or 'learned', got {source!r}")
    if output not in ("points", "map"):
        raise ValueError(f"output must be 'points' or 'map', got {output!r}")
    # The backbone is built either way: `--identify` and `--remember` describe crops with it, and
    # the guard's calibration uses it. `source` decides only what reaches the HEAD.
    backbone = Backbone(PLAN, IN_SHAPE, dtype="f4", seed=1, pool=POOL)
    width = (backbone.n_outputs if source in ("backbone", "learned")
             else raw_width(colour=colour))
    # Sized here, CONSTRUCTED by the caller -- like the detector, the matcher and the object
    # memory. `build` owns the shapes; `main` owns the caller-side state.
    fb = feedback_width(objects) if feedback else 0
    #: Under `--output map` the head emits a segmentation instead of coordinates, and its input
    #: carries the pyramid of past output maps. A map output has no slots: nothing to permute,
    #: nothing to count, and no assignment to make -- which is three measured problems removed at
    #: once rather than one problem made smaller.
    #: Sized here, CONSTRUCTED by the caller -- the convention this file already uses for the
    #: detector, the matcher and the object memory. `build` owns the shapes; `main` owns the state.
    extra = map_memory_width() if output == "map" else 0
    n = width + fb + extra + 1                       # + the constant
    m = MAP_H * MAP_W if output == "map" else 2 * objects

    head = NormalisedLeastMeanSquares(n, m, mu=0.3, dtype="f4")
    #: **`--input learned` puts the backbone INSIDE the adapter (2026-08-17).** A `Stack` is itself
    #: an `Adapter`, so one flat state block covers the convolutional weights and theta together:
    #: the guard checkpoints both, a rollback rolls both back, a freeze stops both learning, and
    #: one persistence record carries both. None of that was true while the backbone sat outside,
    #: where `learn_features` had to be gated on `report.updated` by hand and its weights lived in
    #: a second store with a second fingerprint.
    #:
    #: Only this source. A frozen backbone has nothing to descend, and `--input raw` has no layer
    #: at all -- wrapping either in a `Stack` would pay for a backward pass that changes nothing.
    #: The cost is that the feature stream then carries the **frame** rather than the embedding,
    #: because a Stack's basis is its layer's input; capacity drops to match, since the camera is
    #: the reference stream and a frame is consumed at the step it arrives.
    learner = (Stack([backbone], head, feature_lr=FEATURE_LR / max(backbone.output_scale, 1e-9))
               if source == "learned" else head)
    guard = Guard(
        learner,
        [
            # The head's own residual, in the units the target is in: screen fractions, so a
            # residual above 3 means it is pointing off the screen and further every step.
            # In frame fractions for points; for a map the residual is over 108 soft cells, so the
            # same bound would be meaningless -- scaled by sqrt(m) to keep it a per-output statement.
            Bound("residual_norm", hi=3.0 * (m / 6) ** 0.5),
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

    # The feedback bank rides in the SAME stream as the frame features: it is part of the
    # basis, not a second sensor, and giving it its own stream would make the aligner responsible
    # for something that has no independent clock.
    stream_width = (int(np.prod(IN_SHAPE)) if source == "learned" else width) + fb + extra
    features = Stream("features", dim=stream_width,
                      capacity=8 if source == "learned" else 64, dtype="f4")
    taught = Stream("taught", dim=m, capacity=16, dtype="f4")
    fmap = FrameMap(features=[("features", i) for i in range(stream_width)],
                    target=[("taught", i) for i in range(m)], constant=True)
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
    return backbone, learner, guard, features, taught, fmap, pipe


def fingerprint_for(backbone, objects, args):
    """`main()`'s own call, hoisted out of it so a test can reach it.

    A mutant that read `args.output` in `fingerprint` and never passed it here survived the whole
    suite — the argument had a default, so the omission was silent. That is the same shape as
    `select_newest` compiled and called by no binary, `drive`'s `teach_every` that no caller ever
    passed, and the six other blocks moved out of this `main()` for the same reason.
    """
    return fingerprint(backbone, objects,
                       args.input + ("-grey" if getattr(args, "raw_grey", False) else "")
                       + ("-fb" if getattr(args, "feedback", False) else ""),
                       getattr(args, "output", "points"))


def fingerprint(backbone, objects: int = 1, source: str = "backbone", output: str = "points"):
    """Refuses a saved head restored against a different backbone or readout. The CRC cannot
    see that -- the bytes are perfectly valid, they just mean something else.

    `objects` is in it because it changes what the outputs *mean*, not only how many there are:
    a two-object block restored into a one-object head would put the leftmost object's y into
    the x slot of nothing.

    `output` joined it when `--output map` did, and the gap is worth stating because it was
    survivable rather than harmless: a points block and a map block hashed **identically**, and what
    refused a cross-load was `unpack`'s *shape* check -- "record is for 145x6 but this adapter is
    295x108". That protection is accidental. It holds only while the two sizes differ, and the
    fingerprint exists precisely for the case where the bytes are valid and mean something else. A
    map head's basis also carries 150 pyramid values whose meaning depends on the band layout, which
    no length check can see, so the map shape and memory width go in beside the mode.
    """
    parts = ["pointer", str(PLAN), str(IN_SHAPE), f"pool{POOL}", f"n{backbone.n_outputs}",
             f"obj{objects}", f"src{source}", f"out{output}"]
    if output == "map":
        parts.append(f"map{MAP_H}x{MAP_W}x{map_memory_width()}")
    return fingerprint_of(*parts)


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
    rec = state.get("recognise")
    if rec is not None:
        #: The attended crop, drawn as the box it actually is rather than as a point: the crop is a
        #: third of the frame height and contains the object without being centred on it, which is a
        #: measured property and confusing if only its centre is shown.
        known = rec["slot"] >= 0
        colour = (0, 220, 0) if rec["recognised"] else ((0, 200, 255) if known else (120, 120, 120))
        if rec["centre"] is not None and known:
            cw = rec["crop"] * w // IN_SHAPE[1]
            ch = rec["crop"] * h // IN_SHAPE[0]
            cx, cy = int(rec["centre"][0] * w), int(rec["centre"][1] * h)
            cv2.rectangle(frame, (cx - cw // 2, cy - ch // 2), (cx + cw // 2, cy + ch // 2),
                          colour, 2)
            cv2.drawMarker(frame, (cx, cy), colour, cv2.MARKER_CROSS, 10, 1)

    for tx, ty in state["taught_points"][-8:]:
        cv2.drawMarker(frame, (int(tx * w), int(ty * h)), (255, 180, 0),
                       cv2.MARKER_TILTED_CROSS, 12, 1)

    if state.get("minted"):
        # A new label was just minted. Worth showing, because it is the moment the vocabulary
        # grew and it is otherwise invisible.
        banner = "NEW LABEL #" + ", #".join(str(i) for i in state["minted"])
        cv2.putText(frame, banner, (8, frame.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, banner, (8, frame.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 200, 255), 2, cv2.LINE_AA)
    if state.get("detector_ran"):
        # Where the automatic labels just landed, so the two sources are distinguishable on
        # screen rather than both arriving as an unexplained jump in the dots.
        for px, py in state["taught_points"][-state.get("objects", 1):]:
            cv2.circle(frame, (int(px * w), int(py * h)), 9, (0, 200, 255), 2)

    return _overlay(cv2, frame, state)


def status_lines(state):
    """The overlay text, as short lines rather than one long one.

    Split because a single status line ran past the window and read as overlapping text: at font
    scale 0.5 the outline stroke merges neighbouring glyphs, so a line that is merely *long* looks
    like two lines drawn on top of each other. Returned separately from the drawing so the
    wrapping can be tested without a camera.
    """
    vocab = ""
    if state["remember"]:
        vocab = f"   vocab {state['vocab']}"
        if state.get("vocab_full"):
            # The ceiling is not a detail: past it the memory stops learning new objects and
            # returns the nearest id however far away it is. A caller cannot see that otherwise.
            vocab += " FULL"
    mode = (f"{state['source']}/{state['how']}"
            f"{'+id' if state['identify'] else ''} x{state['objects']}{vocab}")
    counts = (f"auto {state['n_auto']}   mouse {state['n_taught']}   "
              f"updates {state['updates']}")
    health = (f"{state['fps']:.0f} fps   std x{state['drift']:.2f}   "
              f"residual {state['residual']:.3f}")

    if state["frozen"]:
        note = "FROZEN by " + (", ".join(state["frozen_by"]) or "?")
    elif state["learning"]:
        note = "LEARNING"
    elif state["teach"] == "mouse" and state["n_taught"] < 4:
        note = "hold the button on yourself and move to teach it"
    else:
        # The answer to "do I have to keep teaching it?" -- with `--teach mouse`, yes; with a
        # detector wired in, the input teaches it and this line says which.
        note = "predicting only, not learning"
    rec = state.get("recognise")
    if rec is None:
        return [mode, counts, health, note]
    #: **The line that was missing.** With `--locate recognise` the screen showed only the camera:
    #: the slot, whether it was recognised or freshly admitted, and how full the memory is were all
    #: invisible, so nothing distinguished this mode from any other. Found by running it, which is
    #: the third defect in this demo found that way and the third that lived in `main()`.
    if rec["slot"] < 0:
        seen = "nothing here -- stored nothing"
    elif rec["recognised"]:
        seen = f"slot {rec['slot']} RECOGNISED"
    else:
        seen = f"slot {rec['slot']} new -- admitted"
    return [mode, counts, health, f"{seen}   memory {rec['occupied']}/{rec['capacity']}", note]


def _overlay(cv2, frame, state):
    """Draw the status block on a dimmed band, which is what makes it readable over any scene.

    An outline stroke alone was not enough: over a bright window the white text washed out, and
    over a dark chair the black outline vanished. A band is one blend and does not depend on what
    is behind it.

    **And with the band the outline has to go**, which is a bug I shipped by adding one and
    keeping the other. A thickness-3 black stroke under a thickness-1 white glyph puts black ink
    **11 px past the last character** at this scale -- about one character width -- and fills the
    gaps between letters, so on a dim band it reads as a second, bolder copy of the text. That is
    the "why is there black text overlapping" the owner saw. The band supplies the contrast; the
    outline only supplies the double.
    """
    lines = status_lines(state)
    scale, step, pad = 0.52, 22, 8
    height = pad + step * len(lines)
    band = frame[:height].astype(np.float32) * 0.35
    frame[:height] = band.astype(frame.dtype)
    for i, text in enumerate(lines):
        y = pad + step * (i + 1) - 6
        cv2.putText(frame, text, (pad, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
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
    p.add_argument("--locate", choices=("motion", "background", "texture", "recognise"),
                   default=None,
                   help="how the automatic labels find things. `motion` differences consecutive "
                        "frames; `background` differences against a running background estimate. "
                        "Defaults to background above one object, because a frame difference is a "
                        "temporal derivative and leaves a DIPOLE -- measured, two peaks land one "
                        "per object 20.3%% of the time against 96.6%% for background. `texture` "
                        "clusters each frame on its own, with no motion and no background model: "
                        "IoU 1.00 against background subtraction's area of 2.05x the object, and "
                        "it does not lose a stationary subject")
    p.add_argument("--texture-stats", action="store_true",
                   help="add local contrast and gradient to --locate texture. Measured HARMFUL "
                        "when colour already separates the objects (IoU 1.00 -> 0.25) and the only "
                        "thing that works when they share a mean brightness (0.32 -> 0.63)")
    p.add_argument("--identify", action="store_true",
                   help="assign the slots by APPEARANCE instead of left-to-right. Tracks identity "
                        "through a crossing (0.041 against a centre-guess 0.222) where the sort "
                        "rule cannot (0.234). Needs the objects to LOOK different: with identical "
                        "ones the assignment falls to chance, measured 50.6%%")
    p.add_argument("--input", choices=("backbone", "raw", "learned"), default="backbone",
                   help="what reaches the head: the frozen random backbone's 144 features, a "
                        "block-mean downsample of the raw frame in colour (also 144), or the same "
                        "backbone with its weights LEARNED from the head's residual. Raw is 8.34x a "
                        "centre guess on roles against the backbone's 4.61x and 118.87x on identity "
                        "against 4.01x; learned beats the frozen backbone at tracking (0.0163 "
                        "against 0.0241) and costs generalisation (3.05x against 6.19x under a "
                        "brightness change) -- see raw_features() and learn_features()")
    p.add_argument("--output", choices=("points", "map"), default="points",
                   help="what the HEAD emits. 'points' is 2 coordinates per object; 'map' is a "
                        f"{MAP_H}x{MAP_W} segmentation the head learns from the classical labeller, "
                        "with past maps fed back as a resolution pyramid. A map has no slots -- "
                        "nothing to permute, nothing to count, no assignment -- and on same-colour "
                        "twins it reaches IoU 0.98 against a per-frame segmenter's 0.50, the only "
                        "case here where a head beats the detector it learns from")
    p.add_argument("--map-dropout", type=float, default=MAP_DROPOUT, metavar="P",
                   help=f"how often the pyramid is blanked while learning (default {MAP_DROPOUT}). "
                        "The only remedy measured to survive a blackout: holding the memory and "
                        "resetting it both fail identically, because after a gap the memory is about "
                        "a world that has moved on. It costs peak accuracy, 0.98 against ~0.38")
    p.add_argument("--feature-lr", type=float, default=FEATURE_LR, metavar="R",
                   help=f"step for the feature learner under --input learned (default "
                        f"{FEATURE_LR}). Small on purpose: the features move the ground the head "
                        "stands on")
    p.add_argument("--raw-grey", action="store_true",
                   help="drop colour from --input raw. Better for roles (12.09x against 8.34x, "
                        "with a third of the features) and nearly useless for identity (1.71x "
                        "against 118.87x), because colour is what tells the objects apart")
    p.add_argument("--segments", action="store_true",
                   help="open a second window showing the label regions: the background-difference "
                        "mask partitioned by nearest detection, coloured by remembered id. Needs "
                        "--locate background, which is where the mask comes from")
    p.add_argument("--feedback", action="store_true",
                   help="put the head's own past output back into its input, as a bank of leaky "
                        "integrators at three time constants. The only thing that works when two "
                        "objects are indistinguishable by appearance -- and it cannot recover "
                        "from a dropout, because the frame that cannot disambiguate cannot "
                        "correct either. See OutputMemory")
    p.add_argument("--bg-selective", action="store_true",
                   help="stop the background learning where the mask says there is an object. "
                        "Fills the regions instead of outlining them -- core coverage 0.00 to "
                        "0.31 on a mostly-still object -- but never forgets the PATH a moving "
                        "one swept, and peak pairing falls 80%% to 9%% at three objects")
    p.add_argument("--no-split-regions", action="store_true",
                   help="do not split an appearance cluster into its spatially separate pieces. "
                        "Splitting is on because a cluster is not an object: two same-coloured "
                        "objects held apart read IoU 0.50 as one cluster and 1.00 as two regions, "
                        "and a cluster spanning both puts its centroid between them")
    p.add_argument("--no-edge-split", action="store_true",
                   help="do not let the frame's strongest intensity steps break a region. The edge "
                        "is what separates an object from a background it happens to resemble: "
                        "with two distractors eating the cluster budget, two targets read 0.53 by "
                        "connectivity alone and 1.00 with the edge. Costs 1.00 to 0.99 where "
                        "colour already separates everything")
    p.add_argument("--edge-quantile", type=float, default=0.95, metavar="Q",
                   help="gradient quantile above which a pixel may not join its neighbour "
                        "(default 0.95). Relative to the frame's own gradients, so it follows the "
                        "scene's contrast; swept 0.70-0.98, monotone, and 0.95 is where the cost "
                        "on easy scenes has gone")
    p.add_argument("--seg-min-frac", type=float, default=0.01, metavar="F",
                   help="smallest region the SPLIT keeps, as a fraction of the frame (default "
                        "0.01). It does not change IoU -- it changes the region count, 924 pieces "
                        "against 13 on one fixture. Keep it low: it is what the head is taught "
                        "from, and 0.03 costs a slot its region entirely, 0.000 -> 0.222")
    p.add_argument("--seg-show-frac", type=float, default=0.03, metavar="F",
                   help="smallest region the PICTURE paints (default 0.03), a separate threshold "
                        "on the same split because one number cannot serve both: at 0.01 the "
                        "picture shows 6.8 labels of which 44%% are on a real object, at 0.03 it "
                        "shows 2.8 of which 98%% are")
    p.add_argument("--seg-contrast", type=float, default=0.125, metavar="C",
                   help="how much a region must differ from the ring of pixels around it to earn a "
                        "label the area floor and the background cluster would have denied it "
                        "(default 0.125; 0 disables). This is what stops an object going unlabelled "
                        "for resembling the wall: covered 0.80 -> 1.00 on such a scene, at a cost of "
                        "precision 0.99 -> 0.92 on a noise-carved one. The separating interval is "
                        "only 1.15x wide, so on a scene unlike these it may need moving")
    p.add_argument("--label-keep", type=int, default=30, metavar="N",
                   help="how many frames a label waits for its object to come back (default 30, "
                        "one second at 30 Hz). A real forget window: past it nothing recovers. "
                        "Re-identification is by appearance only -- matching by where the object "
                        "was hands its label to whatever arrives there, measured in every "
                        "substitution case tried")
    p.add_argument("--label-tol", type=float, default=0.25, metavar="D",
                   help="how different a region may look and still reclaim a dormant label "
                        "(default 0.25). Distinguishable objects reclaim theirs even after swapping "
                        "places, 8/8; two IDENTICAL objects that hide together and swap come back "
                        "exchanged, 1/8, and nothing here fixes that")
    p.add_argument("--seg-floor", type=float, default=0.25, metavar="F",
                   help="mask threshold as a fraction of the map's own peak (default 0.25). "
                        "Measured on one object: 0.15 gives recall 0.99 at precision 0.42, "
                        "0.40 gives recall 0.76 at precision 1.00")
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
    pyramid = Pyramid(MAP_H, MAP_W) if args.output == "map" else None
    backbone, head, guard, features, taught, fmap, pipe = build(
        n_objects, args.input, colour=not args.raw_grey, feedback=args.feedback,
        output=args.output)
    output_memory = OutputMemory(2 * n_objects) if args.feedback else None
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    store = Store(STATE_PATH)
    #: Only under `--input learned` do the backbone's weights become state worth surviving a power
    #: cut; frozen ones are reproduced from the seed and belong in the fingerprint instead.
    feature_store = Store(FEATURES_PATH) if args.input == "learned" else None
    fp = fingerprint_for(backbone, n_objects, args)
    if args.load:
        rec = store.load(guard, fingerprint=fp)
        if rec is None:
            print("nothing saved yet")
        else:
            print(f"restored a head from sequence {rec.sequence}")
        if feature_store is not None:
            frec = feature_store.load(backbone, fingerprint=fp)
            print(f"restored learned features from sequence {frec.sequence}" if frec is not None
                  else "no learned features saved yet")

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
    elif how == "recognise":
        #: **argmind doing the perception**, rather than consuming a classical labeller's output.
        #: The region and its identity come from `RecognitionLayer`: crop the candidates, recognise
        #: what is remembered or admit what stands out. `--objects 1` only, and it is refused above
        #: that rather than approximated -- see `RecogniseLocator`.
        #:
        #: 147 crops a frame at 1.05 ms each is 6.5 fps, measured. That is the method's rate and not
        #: the wiring's: one whole-frame pass would be 21x cheaper and loses the identity, because a
        #: region pooled from the feature map can only be centred to 4 pixels and the descriptor
        #: needs 2.
        detector = build_recogniser(n_objects)
    elif how == "texture":
        # Every frame produces labels, so there is no cadence and nothing to be blind about --
        # which is the other half of dropping the background model. `k` is objects + 1 because
        # one cluster is the background: a two-object scene needs three regions, not two.
        detector = FrameSegmenter(n_objects + 1, texture=args.texture_stats)
    elif how == "background":
        detector = BackgroundLocator(n_objects, rate=0.05, selective=args.bg_selective)
    else:
        detector = CadenceDetector(args.detect_every, k=n_objects)

    # Both `--identify` and `--remember` describe crops, so the describer is built once for
    # either. It was built inside each branch at first, with the second branch testing `dir()` for
    # a name the first may or may not have bound -- a construction that works and is a trap.
    describe = None
    if args.identify or args.remember:
        if how == "motion":
            sys.exit("--identify and --remember need --locate background or texture: appearance "
                     "is only as good as the crops it describes, and motion differencing puts "
                     "those on the leading and trailing edges of one object rather than on two.")
        crop_backbone = Backbone(PLAN, (CROP, CROP, 3), dtype="f4", seed=1, pool=2)

        def describe(frame, point, _bb=crop_backbone):
            v = _bb.features(crop_at(frame, point)).astype(np.float64)
            n = float(np.linalg.norm(v))
            return v / n if n > 0 else v

    if args.segments and how == "motion":
        sys.exit("--segments needs --locate background or --locate texture: a frame difference "
                 "gives the leading and trailing edges of one object rather than its extent.")
    matcher = AppearanceMatcher(n_objects, describe) if args.identify else None
    memory = ObjectMemory(args.tol, max_ids=64) if args.remember else None
    #: One label per region that survives between frames. Only the picture uses it: the head's slots
    #: come from `pick_regions`, which is a different and bounded question.
    tracker = RegionTracker(keep=args.label_keep, tol=args.label_tol)
    state = {"pred": None, "frozen": False, "frozen_by": (), "n_taught": 0,
             "updates": 0, "fps": 0.0, "drift": 0.0, "residual": 0.0,
             "taught_points": [], "learning": False, "last_update_frame": -999,
             "n_auto": 0, "detector_ran": False, "teach": args.teach,
             "every": args.detect_every, "objects": n_objects, "how": how,
             "identify": args.identify, "vocab": 0, "slot_ids": [], "minted": [],
             "remember": args.remember, "source": args.input, "vocab_full": False}
    # Calibrated from the first second: the scale of the caller's own statistic and of the
    # backbone's output. Both are numbers the library refuses to guess.
    frame_std = None
    calibrated = False
    camera_failed = False
    feature_scale = 1.0

    def extract(small, blank=False):
        #: The pyramid rides in the SAME extraction path as everything else. Adding a term outside
        #: it is how `--feedback` broke on its first launch: the calibration branch recomputes the
        #: basis, so anything concatenated at the call site is silently dropped on that one frame.
        frame_part = basis(small, backbone=backbone, source=args.input,
                           colour=not args.raw_grey, scale=feature_scale,
                           output_memory=output_memory)
        if pyramid is None:
            return frame_part
        return map_basis(frame_part, pyramid, blank=blank)
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
        # One extraction path, so calibration cannot silently switch the source. The first
        # version recomputed `backbone.features(small)` after calibrating, which would have
        # ignored `--input raw` on that frame and every frame after it.
        #: Training-time dropout of the pyramid. The head is always learning here, so "while
        #: training" is every frame it updates -- and this is the only remedy measured to survive a
        #: blackout, at a real cost to peak accuracy.
        blank = (pyramid is not None and args.map_dropout > 0.0
                 and np.random.random() < args.map_dropout)
        emb = extract(small, blank=blank)

        if not calibrated and frames > 15:
            # Scaling the input matters and is not optional: `p0` has units, and this project
            # measured an unscaled embedding at 6.8e-4 defeating every head until it was scaled.
            n_frame = frame_width(backbone, args.input, not args.raw_grey)
            backbone.output_scale, feature_scale = calibration_scales(args.input, emb, n_frame)
            frame_std = float(np.std(small))
            calibrated = True
            emb = extract(small, blank=blank)
            applied = max(backbone.output_scale, feature_scale)
            print(f"calibrated: feature scale {applied:.1f}, frame std {frame_std:.3f}")
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
        if isinstance(detector, RecogniseLocator):
            #: **What the recogniser just did, so the window can show it.** Wiring the decision and
            #: not the display left the demo showing only the camera: the slot, whether it was
            #: recognised or freshly admitted, and where it looked were all invisible, so nothing on
            #: screen distinguished this mode from any other. Found by running it.
            state["recognise"] = {
                "slot": detector.last_slot,
                "recognised": detector.last_recognised,
                "centre": detector.layer.last_centre,
                "occupied": detector.layer.memory.occupied,
                "capacity": detector.layer.memory.capacity,
                "crop": RECOGNISE_CROP,
            }
        #: Which cluster each surviving label came from, in slot order. Carried because the
        #: display used to colour cluster *n* with the id of slot *n*, and those are different
        #: orderings: `keep` sorts by area, the x-sort below re-sorts, and the matcher permutes
        #: again. The pairing was arbitrary and shifted every frame, which is what made the
        #: region colours flicker whenever anything moved.
        slot_cluster = None
        region_map = None
        if auto is not None and how == "texture":
            # A cluster is a region of similar appearance, not an object. Splitting it into
            # spatially separate pieces is what separates two objects that look alike, and the edge
            # gate is what separates an object from a background it resembles.
            region_map, of_cluster, slot_cluster, auto = regions_for_slots(
                detector, small, n_objects,
                split=not args.no_split_regions,
                edges=None if args.no_edge_split else args.edge_quantile,
                min_frac=args.seg_min_frac)
        if auto is not None and matcher is not None and slot_cluster is not None:
            # The matcher permutes the points, so the cluster list has to follow it. Matched by
            # value, which is safe because two clusters with an identical centre would already be
            # one cluster.
            before = list(auto)
            auto = matcher(small, auto)
            slot_cluster = follow_permutation(before, auto, slot_cluster)
        elif auto is not None and matcher is not None:
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
            state["vocab_full"] = memory.full
        field_ids, labels = [], None
        #: The map head needs the label field as its TEACHER, so it is computed whenever either the
        #: picture or the teacher wants it -- not only when `--segments` is drawing.
        want_field = (args.segments or args.output == "map") and auto is not None
        if want_field:
            if how == "texture":
                # Every region gets a label, not only the head's slots: the head has K of them
                # because that is its output width, and a picture has no such limit. The dropped
                # background cluster stays 0 and is drawn dark, which is what makes the regions
                # read as separate rather than tiling the whole frame.
                labels, field_ids = label_field(region_map, of_cluster, detector.labels, tracker,
                                                min_frac=args.seg_show_frac, frame=small,
                                                contrast=args.seg_contrast)
            elif getattr(detector, "foreground", None) is not None:
                labels = partition(detector.foreground, auto, floor=args.seg_floor)
            if labels is not None and args.segments:
                cv2.imshow("argmind - segments",
                           segment_image(labels, field_ids if field_ids else
                                         (state.get("slot_ids") or [])))
        state["detector_ran"] = auto is not None
        if auto is not None:
            # Sorted leftmost-first by the detector, and flattened in that order. The order is
            # the whole reason this works at all: a head with 2k outputs is permutation-
            # symmetric, and labels in whatever order they were found reach 0.256 against a
            # centre-guess 0.312 -- i.e. learn nothing.
            if args.output == "map":
                #: The classical labeller is the TEACHER here, not the output. It produces the map
                #: the head learns to reproduce -- and withdrawing it is the experiment the
                #: 2026-08-14 decision asks for, not a step already taken.
                if labels is not None:
                    taught.push(now, map_teacher(labels))
                    state["n_auto"] += 1
            else:
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
            if teach_from_click(taught, now + 1e-6, cx / w, cy / h, n_objects, args.output):
                state["n_taught"] += 1
            state["taught_points"].append((cx / w, cy / h))

        report = pipe.step()
        if report.prediction is not None and pyramid is not None:
            #: FREE-RUNNING: the pyramid sees the head's own prediction, not the label. Teacher
            #: forcing reads an order of magnitude better and describes no system anyone can deploy.
            pred_map = np.clip(np.asarray(report.prediction, dtype=np.float64), 0.0, 1.0)
            pyramid.observe(pred_map)
            state["map_iou"] = (map_iou(pred_map, map_teacher(labels))
                                if labels is not None else None)
            if args.segments:
                shown = (pred_map.reshape(MAP_H, MAP_W) >= 0.4).astype(np.int32)
                cv2.imshow("argmind - head output",
                           segment_image(np.repeat(np.repeat(shown, 8, 0), 8, 1), [], scale=1))
        if report.prediction is not None and pyramid is None:
            state["pred"] = tuple(float(v) for v in report.prediction)
        #: Learn the FEATURES, not only the head. Gated on the head having actually updated, which
        #: also stops it while the guard is frozen -- a fault is exactly when you do not want it
        #: pushed into the representation. The target is the label pushed this frame, which is exact
        #: under `--locate texture` (one label per frame) and an approximation otherwise.
        #: **The feature step used to be here, in 42 lines of caller code.** `Stack` owns the
        #: chain rule now: one `update(x, y)` moves the convolutional weights and theta together,
        #: gated on the head's own step rather than on a hand-written `report.updated` check, and
        #: the guard covers both. What is left is bookkeeping.
        if args.input == "learned" and report.updated and not report.frozen:
            state["feat_steps"] = state.get("feat_steps", 0) + 1
        state["frozen"] = bool(report.frozen)
        state["frozen_by"] = tuple(report.frozen_by)
        # Counted from the reports, because `Guard` exposes no update counter. This line used
        # to read `guard.updates if hasattr(guard, "updates") else state["updates"]`, and
        # `Guard` has no such attribute -- so the number on screen was hardwired to 0 for the
        # life of the demo, and it is the one number that answers "is it learning right now".
        # A defensive `hasattr` around a field nobody checked is how that stays invisible.
        if output_memory is not None and state["pred"] is not None:
            # Fed with the head's own output, which is the free-running case the docstring's
            # numbers are for. Feeding the label instead is teacher forcing and reads 11x better
            # while measuring a system nobody can deploy.
            output_memory.observe(state["pred"])
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
            if feature_store is not None:
                feature_store.save(backbone, fingerprint=fp)
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
        if feature_store is not None:
            feature_store.save(backbone, fingerprint=fp)
        print(f"saved to {store.slot_path(0)} and .1"
              + (f", features to {feature_store.slot_path(0)} and .1"
                 if feature_store is not None else ""))
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
