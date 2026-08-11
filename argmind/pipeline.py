# SPDX-License-Identifier: Apache-2.0
"""Wiring: streams to frames to an adapted model, behind one bounded `step()`.

This is where the pieces meet, and the meeting point is where the two questions
the other modules could avoid have to be answered.

**How does a `Frame` become ``(x, y)``?** The framework does not know which stream
is a feature and which is a target — that is the user's problem domain, and a
library that guesses gets it wrong. So it is declared, and declared as an *index
table* rather than as a callable. A callable would be the obvious Python answer
and it does not cross a C ABI; a pair of ``(stream, component)`` lists is two
integer arrays on the other side. `FrameMap` accepts convenient forms and
resolves them once at construction into that flat table, so the convenience costs
nothing at run time and nothing at the boundary.

**Where does staleness stop being a number nobody reads?** Here. The aligner
reports it and the adapter cannot see it, so the pipeline is the only place that
can act on it. Note the split, because it is deliberate:

- The pipeline gates on **input health** — staleness, sensor validity — *before*
  the update. Bad input should not reach the learner at all.
- The `Guard` gates on **model health** — residual, gain, covariance — *after* it,
  because those quantities do not exist until the update has been computed.

Two gates, two different times, two different failures. Collapsing them into one
would mean either checking the model before it has moved or letting known-bad
data into it first.

`step()` is pull-based and does bounded work: at most the reference stream's
retained sample count, then it returns. No threads, no callbacks, no event loop.
The caller owns timing, which is what an embedded integrator needs and what makes
the whole thing deterministically testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .adapt import Adapter
from .align import NON_CAUSAL, Aligner
from .stream import Stream

__all__ = ["FrameMap", "Pipeline", "StepReport"]


def _resolve(spec, streams: dict[str, Stream], what: str) -> tuple[tuple[str, int], ...]:
    """Turn one declaration into a flat ``((stream, component), ...)`` table.

    Accepts a bare stream name (all of its components), ``(name, i)``,
    ``(name, [i, j])`` or ``(name, slice)``. Everything resolves here, at
    construction, against the streams that actually exist — so a typo or an index
    past the end of a sensor fails now rather than on the first frame at 3am.
    """
    out: list[tuple[str, int]] = []
    items = spec if isinstance(spec, (list, tuple)) and not _is_pair(spec) else [spec]
    for item in items:
        if isinstance(item, str):
            name, sel = item, None
        else:
            name, sel = item
        if name not in streams:
            raise KeyError(f"{what}: no stream named {name!r}; have {sorted(streams)}")
        dim = streams[name].dim
        if sel is None:
            idx = list(range(dim))
        elif isinstance(sel, slice):
            idx = list(range(dim))[sel]
        elif isinstance(sel, (int, np.integer)):
            idx = [int(sel)]
        else:
            idx = [int(i) for i in sel]
        for i in idx:
            if not 0 <= i < dim:
                raise IndexError(
                    f"{what}: component {i} is outside stream {name!r}, which has "
                    f"dim={dim}")
            out.append((name, i))
    if not out:
        raise ValueError(f"{what}: resolved to nothing")
    return tuple(out)


def _is_pair(spec) -> bool:
    """True for a single ``(name, selector)`` declaration, as opposed to a list of
    them. Needed because both are sequences and confusing them silently would
    build the wrong feature vector."""
    return (len(spec) == 2 and isinstance(spec[0], str)
            and not isinstance(spec[1], str))


class FrameMap:
    """Which parts of a frame are the regressor, and which are the target.

    Parameters
    ----------
    features, target
        A stream name (all components), ``(name, i)``, ``(name, [i, j])``, a
        ``(name, slice)``, or a list of any of those. Order is the order of the
        vector, which matters: the parameters the adapter learns are indexed by
        it, and a persisted state block restored against a different order is
        wrong in a way nothing downstream can detect.
    constant : bool
        Append a constant 1.0 to the regressor. Not a convenience — with it the
        adapter learns an offset term, which for online *bias* calibration is the
        whole point of the exercise.
    """

    __slots__ = ("features", "target", "constant", "_feat", "_targ")

    def __init__(self, features, target, constant: bool = False) -> None:
        self.features = features
        self.target = target
        self.constant = bool(constant)
        self._feat: tuple[tuple[str, int], ...] = ()
        self._targ: tuple[tuple[str, int], ...] = ()

    def resolve(self, streams: dict[str, Stream]) -> None:
        self._feat = _resolve(self.features, streams, "features")
        self._targ = _resolve(self.target, streams, "target")

    @property
    def n_features(self) -> int:
        return len(self._feat) + (1 if self.constant else 0)

    @property
    def n_outputs(self) -> int:
        return len(self._targ)

    @property
    def slots(self) -> tuple[tuple[tuple[str, int], ...], tuple[tuple[str, int], ...]]:
        """The resolved tables, for inspection and for the C side later."""
        return self._feat, self._targ

    def __repr__(self) -> str:
        return (f"FrameMap(features={self._feat}, target={self._targ}, "
                f"constant={self.constant})")


@dataclass(frozen=True, slots=True)
class StepReport:
    """What one `step()` did.

    Observability is the return value, not a logging afterthought — same rule as
    `UpdateReport`, one level up.

    Attributes
    ----------
    frames : int
        Frames the aligner produced this step.
    updated : int
        Adaptations that actually stand. Never more than `frames`, and the gap is
        the interesting part.
    lost_frames : int
        Reference-stream samples that were overwritten before they could become
        frames — input the pipeline will never see. Non-zero means `step()` is
        being called too rarely for the ring size.

        This exists because a test found the gap: a 200-frame workload against a
        64-deep ring adapted 64 times, reported nothing amiss, and produced a
        perfectly plausible fit from a third of the data.

        It counts the reference stream only, and that is the whole of what can
        honestly be counted. `Stream.dropped` is *overwrites*, which in steady
        state happen constantly and mean nothing; and a non-reference sample
        being overwritten is normal under `hold_last`, which only ever wants the
        newest one at or before the reference instant. A reference sample that
        is overwritten, by contrast, is a frame that will never exist.
    skipped : dict[str, int]
        Why the rest did not, by reason. Stable tokens: ``stale``, ``invalid``,
        ``stale_target`` come from the pipeline's input gate; ``frozen``,
        ``guard:<check>``, ``nonfinite_input``, ``nonpositive_denominator`` come
        back from the adapter or the guard.
    frozen : bool
        Is adaptation frozen *now*. A guarded pipeline that froze mid-step
        reports True even though earlier frames in the same step applied.
    frozen_by : tuple[str, ...]
        Checks that tripped during this step, if any.
    staleness_s : dict[str, float]
        Per-stream staleness of the most recent frame — the current picture, not
        an average over the step. Signed, as the aligner reports it.
    worst_staleness_s : float
        Worst absolute staleness seen anywhere in this step. This is the one to
        alarm on; the dict above is for working out which stream to blame.
    t : float | None
        Reference time of the last frame, or None if there were none.
    prediction : np.ndarray | None
        The model's output for the **most recent frame**, computed after every
        adaptation this step applied — so it is the current best estimate for the
        newest input, which is what a control loop wants back.

        Present whenever a frame existed, including one the input gate refused to
        learn from. That is the point: refusing to *learn* from a sample is not a
        reason to stop *predicting* on it, and separating the two is what lets a
        fast output run on a slow target.

        **The array is preallocated and overwritten by the next `step()`.** It is
        handed out read-only so the overwrite cannot be mistaken for a caller's
        own copy. A fresh array per step would allocate once per step, which
        `bench.memory_growth` is built to notice and `Adapter.state` was already
        fixed for.
    """

    frames: int = 0
    updated: int = 0
    lost_frames: int = 0
    skipped: dict[str, int] = field(default_factory=dict)
    frozen: bool = False
    frozen_by: tuple[str, ...] = ()
    staleness_s: dict[str, float] = field(default_factory=dict)
    worst_staleness_s: float = 0.0
    t: float | None = None
    prediction: np.ndarray | None = None

    @property
    def idle(self) -> bool:
        """No frames were available. Distinct from 'frames arrived and none were
        used', which is a fault and this is not."""
        return self.frames == 0

    @property
    def healthy(self) -> bool:
        """Every available frame was adapted to, nothing was dropped, nothing is
        frozen. One call for 'is this pipeline actually working'."""
        return (self.updated == self.frames and not self.lost_frames
                and not self.skipped and not self.frozen)


class Pipeline:
    """Streams to alignment to adaptation to prediction, behind one `step()`.

    Parameters
    ----------
    streams : sequence of Stream
    aligner : Aligner
    adapter : Adapter
        Usually a `Guard` wrapping one — a `Guard` *is* an `Adapter`, so nothing
        here distinguishes them. Unguarded is permitted rather than blocked,
        because the Python reference has to be runnable unguarded to serve as the
        oracle the C++ core is verified against. `guarded` says which you have.
    frame_map : FrameMap
    max_stale_s : float | None
        Frames whose worst staleness exceeds this do not adapt. They are still
        counted and still reported. None means no input gate on staleness, which
        is rarely right: a stream that quietly died then contributes its last
        value to every update forever.
    require_valid : bool
        Skip adaptation on frames where any contributing sensor set its own
        validity flag false. Default True — the sensor said the reading was bad
        and it is the only thing in the system that knows.
    require_fresh_target : bool
        Adapt only when the target sample has actually changed since the last
        attempt. Default False, which preserves the behaviour every earlier
        version had.

        This exists because low latency and honest weighting pull in opposite
        directions. Latency wants the *fastest* stream as the reference, so a
        frame — and an output — exists at every fast sample. But when the target
        rides a slower stream, hold-last then hands the same measurement to the
        learner over and over: measured, a 10 Hz reference against a 1 Hz target
        produced 191 frames from 20 measurements, each used up to ten times. RLS
        weights that measurement ten times, so ``P`` shrinks faster than the
        information justifies and the covariance ends up over-confident.

        The obvious workaround is ``max_stale_s``, and it is a *time* threshold
        rather than an identity check, so it breaks under jitter in both
        directions: measured with ±20 ms of jitter, a 0.05 s gate adapted 8 times
        out of 20 and a 0.15 s gate adapted 25.

        Freshness is decided on the **sample timestamp** of every stream a target
        component comes from, which is exactly the identity ``max_stale_s`` was
        standing in for. Two notes on the edges. Under ``interpolate`` the target
        is synthesised at the reference instant, so it is new every frame and this
        gate never fires — correctly, since the value really does differ. And if
        the target rides the reference stream itself there is a new target in
        every frame, so the gate costs nothing and does nothing.
    allow_non_causal : bool
        `nearest` and `interpolate` consult samples from after the reference
        instant. That is legitimate offline and a correctness bug in a causal
        real-time loop, which is what a pipeline is, so it must be said out loud
        here even though the aligner alone permits it.
    """

    __slots__ = ("streams", "aligner", "adapter", "frame_map", "max_stale_s",
                 "require_valid", "require_fresh_target", "_x", "_y", "_stats",
                 "_last_report", "_arrived", "_pred", "_target_streams",
                 "_last_target_t")

    def __init__(self, streams, aligner: Aligner, adapter: Adapter,
                 frame_map: FrameMap, max_stale_s: float | None = None,
                 require_valid: bool = True,
                 allow_non_causal: bool = False,
                 require_fresh_target: bool = False) -> None:
        streams = list(streams)
        self.streams = {s.name: s for s in streams}
        if len(self.streams) != len(streams):
            raise ValueError("two streams share a name; they would be "
                             "indistinguishable in every frame and every map")
        if aligner.reference not in self.streams:
            raise KeyError(f"reference stream {aligner.reference!r} is not among "
                           f"{sorted(self.streams)}")
        if aligner.policy in NON_CAUSAL and not allow_non_causal:
            raise ValueError(
                f"aligner policy {aligner.policy!r} consults samples from after the "
                "reference instant. In a causal loop that is a correctness bug, not "
                "a tuning choice. Pass allow_non_causal=True if this pipeline is "
                "being run offline over recorded data.")
        if max_stale_s is not None and max_stale_s <= 0:
            raise ValueError(f"max_stale_s must be positive, got {max_stale_s}")

        frame_map.resolve(self.streams)
        if frame_map.n_features != adapter.n_features:
            raise ValueError(
                f"the frame map produces {frame_map.n_features} features "
                f"(constant={frame_map.constant}) but the adapter expects "
                f"{adapter.n_features}")
        if frame_map.n_outputs != adapter.n_outputs:
            raise ValueError(
                f"the frame map produces {frame_map.n_outputs} outputs but the "
                f"adapter expects {adapter.n_outputs}")

        self.aligner = aligner
        self.adapter = adapter
        self.frame_map = frame_map
        self.max_stale_s = max_stale_s
        self.require_valid = bool(require_valid)
        self.require_fresh_target = bool(require_fresh_target)
        # Resolved once: which streams a target component comes from. Order does not
        # matter here, only membership, so a tuple of unique names is enough.
        self._target_streams = tuple(dict.fromkeys(
            name for name, _ in frame_map.slots[1]))
        self._last_target_t: tuple[float, ...] | None = None

        # Preallocated. Extracting a regressor from a frame allocates nothing.
        self._x = np.zeros(adapter.n_features, dtype=adapter.dtype)
        self._y = np.zeros(adapter.n_outputs, dtype=adapter.dtype)
        # Preallocated for the same reason as _x and _y: a fresh array per step is an
        # allocation per step, and bench.memory_growth exists to catch exactly that.
        self._pred = np.zeros(adapter.n_outputs, dtype=adapter.dtype)
        self._pred.flags.writeable = False
        self._stats = {"steps": 0, "frames": 0, "updated": 0, "lost_frames": 0}
        self._last_report = StepReport()
        # Starts at zero, not at what the stream already holds: samples pushed
        # before this pipeline existed are still owed a frame, and if the ring
        # overflowed before the first step then that loss is real and should be
        # reported on it.
        self._arrived = 0

    # -- properties ----------------------------------------------------------

    @property
    def guarded(self) -> bool:
        """Whether the adapter carries a freeze path. False is legitimate for
        offline reference runs and is a bad idea in the field."""
        from .guard import Guard
        return isinstance(self.adapter, Guard)

    @property
    def frozen(self) -> bool:
        return bool(getattr(self.adapter, "frozen", False))

    @property
    def stats(self) -> dict[str, int]:
        return dict(self._stats)

    @property
    def last(self) -> StepReport:
        """The most recent step's report, for callers that stepped somewhere they
        could not conveniently keep the return value."""
        return self._last_report

    # -- the work ------------------------------------------------------------

    def _extract(self, frame) -> None:
        feat, targ = self.frame_map.slots
        for j, (name, i) in enumerate(feat):
            self._x[j] = frame.values[name][i]
        if self.frame_map.constant:
            self._x[-1] = 1.0
        for j, (name, i) in enumerate(targ):
            self._y[j] = frame.values[name][i]

    def step(self) -> StepReport:
        """Consume every frame the streams can currently produce, and report.

        Bounded: the aligner emits at most the reference stream's retained sample
        count, and every frame costs one adapter update. Nothing here waits and
        nothing allocates per sample.
        """
        skipped: dict[str, int] = {}

        # Loss is arrivals minus what the aligner got to look at. Nothing is ever
        # removed from a ring, so `dropped + len` is the total ever pushed; the
        # aligner considered one reference instant per frame it emitted plus one
        # per instant it declined to build. Anything unaccounted for was
        # overwritten before it could become a frame.
        ref = self.streams[self.aligner.reference]
        arrived = ref.dropped + len(ref)
        before = self.aligner.stats
        frames = self.aligner.drain(self.streams)
        after = self.aligner.stats
        considered = (len(frames)
                      + after["no_data"] - before["no_data"]
                      + after["dropped_stale"] - before["dropped_stale"])
        lost = max(0, (arrived - self._arrived) - considered)
        self._arrived = arrived
        updated = 0
        tripped: list[str] = []
        worst = 0.0
        staleness: dict[str, float] = {}
        last_t: float | None = None

        def note(reason: str) -> None:
            skipped[reason] = skipped.get(reason, 0) + 1

        for frame in frames:
            last_t = frame.t
            staleness = frame.staleness_s
            worst = max(worst, frame.worst_staleness_s)

            # Extracted for EVERY frame, before the gates, so that a frame the learner
            # refuses still produces an output. The C++ port already did it in this order;
            # the two were unobservably different until `prediction` gave `_x` a reader.
            self._extract(frame)

            # Input gate, before the learner sees anything.
            if self.max_stale_s is not None and frame.worst_staleness_s > self.max_stale_s:
                note("stale")
                continue
            if self.require_valid and not frame.all_valid():
                note("invalid")
                continue
            if self.require_fresh_target:
                # The sample's own timestamp, carried by the aligner. This used to
                # reconstruct it as `frame.t + frame.staleness_s[name]`, and that addition
                # is not bit-stable across reference instants: after a gap in the target
                # stream the same physical sample rebuilt an ulp away from the stored value
                # and read as new. Measured, 52 of 120 frames relearned a 14-second-old
                # target -- the exact failure this gate exists to prevent. An identity check
                # needs the identity, not a value equal to it to within rounding.
                stamps = tuple(frame.sample_t[name] for name in self._target_streams)
                if stamps == self._last_target_t:
                    note("stale_target")
                    continue
                # Recorded on the ATTEMPT, not on success. A target the adapter then
                # rejects is not retried on the next nine frames: it would be rejected
                # nine more times and the counters would say nine faults where there was
                # one bad measurement.
                self._last_target_t = stamps

            report = self.adapter.update(self._x, self._y)
            if report.applied:
                updated += 1
            else:
                note(report.reason or "rejected")
            for name in getattr(report, "tripped", ()):
                tripped.append(name)

        # The output, once per step rather than once per frame: the newest frame's
        # regressor against the parameters this step finished with. O(n*m) — six
        # multiply-adds at the design driver's shape, which is not a cost worth making
        # optional.
        prediction = None
        if frames:
            self._pred.flags.writeable = True
            self._pred[:] = self.adapter.predict(self._x)
            self._pred.flags.writeable = False
            prediction = self._pred

        self._stats["steps"] += 1
        self._stats["frames"] += len(frames)
        self._stats["updated"] += updated
        self._stats["lost_frames"] += lost
        self._last_report = StepReport(
            frames=len(frames), updated=updated, lost_frames=lost, skipped=skipped,
            frozen=self.frozen, frozen_by=tuple(tripped),
            staleness_s=dict(staleness), worst_staleness_s=worst, t=last_t,
            prediction=prediction)
        return self._last_report

    def predict(self, x) -> np.ndarray:
        """Straight through to the adapter. Unaffected by freezing — that is the
        point of freezing."""
        return self.adapter.predict(x)

    def predict_frame(self, frame) -> np.ndarray:
        """Predict from a frame, using the same map `step()` uses.

        Exists so that a caller cannot get the regressor layout right in training
        and wrong at inference, which is a mistake that produces plausible numbers
        and no error.
        """
        self._extract(frame)
        return self.adapter.predict(self._x)

    def __repr__(self) -> str:
        return (f"Pipeline(streams={sorted(self.streams)}, "
                f"aligner={self.aligner.policy!r}, guarded={self.guarded}, "
                f"frozen={self.frozen}, steps={self._stats['steps']})")
