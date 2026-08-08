# SPDX-License-Identifier: Apache-2.0
"""Aligning multi-rate, asynchronous sensor streams into frames.

This is the part most multi-sensor code gets wrong silently. Sensors do not share
a clock or a rate; samples arrive late, jittered, or not at all. Zipping arrays
together implies a synchronisation policy without stating it, and the choice of
policy changes the answer.

So the policy is explicit, it is recorded in the frame it produced, and every
frame reports how stale each contributing sample was. A caller who does not look
at staleness has still been told.

Policies
--------
`hold_last`    take the newest sample at or before the reference time. The honest
               default for control: it never invents data and never looks ahead,
               but it is biased by up to one sample period of the slower stream.
`nearest`      take the closest sample in time, before or after. Lower average
               error than hold-last, but it USES FUTURE DATA relative to the
               reference instant, so it is legitimate offline and a bug in a
               causal real-time loop. The framework will not stop you; it will
               record it.
`interpolate`  linearly interpolate between the bracketing samples. Best accuracy
               for smooth signals; meaningless for anything discrete, and it also
               requires a future sample.
`drop_if_stale` like hold_last, but yields no frame at all when any stream is
               older than its limit. Use when a stale input is worse than no
               output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .stream import Sample, Stream

__all__ = ["Frame", "Aligner", "Policy"]

Policy = Literal["hold_last", "nearest", "interpolate", "drop_if_stale"]

#: Policies that may consult a sample later than the reference instant. Using one
#: of these inside a causal loop is a correctness error, not a tuning choice, so
#: the aligner names them rather than leaving the user to work it out.
NON_CAUSAL: frozenset[str] = frozenset({"nearest", "interpolate"})


@dataclass(frozen=True, slots=True)
class Frame:
    """One synchronised observation across streams.

    Attributes
    ----------
    t : float
        The reference instant this frame represents.
    values : dict[str, np.ndarray]
        Per-stream aligned values.
    staleness_s : dict[str, float]
        How far each contributing sample sat from `t`. Signed: negative means the
        sample predates the reference, positive means it came after (only
        possible under a non-causal policy). Interpolated entries report 0.0.
    valid : dict[str, bool]
        The sensor's own validity flag, carried through untouched.
    policy : str
    """

    t: float
    values: dict[str, np.ndarray]
    staleness_s: dict[str, float]
    valid: dict[str, bool] = field(default_factory=dict)
    policy: str = "hold_last"

    @property
    def worst_staleness_s(self) -> float:
        return max((abs(v) for v in self.staleness_s.values()), default=0.0)

    def all_valid(self) -> bool:
        return all(self.valid.values()) if self.valid else True


class Aligner:
    """Combines streams into frames on a reference stream's timebase.

    One stream is the *reference*: its samples define the instants at which frames
    exist. That is the right shape for sensor fusion, where a fast stream (say an
    IMU) sets the cadence and slower aiding sources are resampled onto it.

    Parameters
    ----------
    reference : str
    policy : Policy
    max_stale_s : float | None
        A sample older than this contributes, but the frame is marked stale. Under
        `drop_if_stale` no frame is produced at all. None disables the check,
        which is rarely what anyone wants — a stream that silently died then
        contributes its last value forever.
    """

    __slots__ = ("reference", "policy", "max_stale_s", "_consumed_to",
                 "_frames", "_dropped_stale", "_no_data")

    def __init__(self, reference: str, policy: Policy = "hold_last",
                 max_stale_s: float | None = None) -> None:
        if policy not in ("hold_last", "nearest", "interpolate", "drop_if_stale"):
            raise ValueError(f"unknown policy {policy!r}")
        if max_stale_s is not None and max_stale_s <= 0:
            raise ValueError(f"max_stale_s must be positive, got {max_stale_s}")

        self.reference = reference
        self.policy = policy
        self.max_stale_s = max_stale_s
        self._consumed_to = -np.inf   # reference time already emitted
        self._frames = 0
        self._dropped_stale = 0
        self._no_data = 0

    # -- convenience constructors, so the policy is visible at the call site --

    @classmethod
    def hold_last(cls, reference: str, max_stale_s: float | None = None) -> "Aligner":
        return cls(reference, "hold_last", max_stale_s)

    @classmethod
    def nearest(cls, reference: str, max_stale_s: float | None = None) -> "Aligner":
        return cls(reference, "nearest", max_stale_s)

    @classmethod
    def interpolate(cls, reference: str,
                    max_stale_s: float | None = None) -> "Aligner":
        return cls(reference, "interpolate", max_stale_s)

    @classmethod
    def drop_if_stale(cls, reference: str, max_stale_s: float) -> "Aligner":
        return cls(reference, "drop_if_stale", max_stale_s)

    # -- statistics, because a silent aligner is an undebuggable one ---------

    @property
    def is_causal(self) -> bool:
        """False if this policy may consult samples later than the reference."""
        return self.policy not in NON_CAUSAL

    @property
    def stats(self) -> dict[str, int]:
        return {"frames": self._frames,
                "dropped_stale": self._dropped_stale,
                "no_data": self._no_data}

    # -- the work -----------------------------------------------------------

    def _pick(self, s: Stream, t: float) -> tuple[Sample, float] | None:
        """Return (sample, signed staleness) for stream `s` at reference time `t`."""
        if self.policy in ("hold_last", "drop_if_stale"):
            got = s.newest_at_or_before(t)
            return (got, got.t - t) if got else None

        if self.policy == "nearest":
            got = s.nearest(t)
            return (got, got.t - t) if got else None

        pair = s.bracketing(t)
        if pair is None:
            return None
        a, b = pair
        if b.t == a.t:
            return a, 0.0
        w = (t - a.t) / (b.t - a.t)
        blended = (1.0 - w) * a.values.astype(np.float64) \
            + w * b.values.astype(np.float64)
        return Sample(t, blended.astype(a.values.dtype),
                      a.valid and b.valid), 0.0

    def align(self, streams: dict[str, Stream], t: float) -> Frame | None:
        """Build one frame at reference instant `t`, or None if it cannot be built."""
        values: dict[str, np.ndarray] = {}
        staleness: dict[str, float] = {}
        validity: dict[str, bool] = {}

        for name, s in streams.items():
            picked = self._pick(s, t)
            if picked is None:
                self._no_data += 1
                return None
            sample, gap = picked
            if (self.max_stale_s is not None
                    and abs(gap) > self.max_stale_s
                    and self.policy == "drop_if_stale"):
                self._dropped_stale += 1
                return None
            values[name] = sample.values
            staleness[name] = gap
            validity[name] = sample.valid

        self._frames += 1
        return Frame(t=t, values=values, staleness_s=staleness,
                     valid=validity, policy=self.policy)

    def drain(self, streams: dict[str, Stream]) -> list[Frame]:
        """Emit a frame for every unconsumed reference sample.

        Advances an internal watermark so repeated calls do not re-emit. Bounded
        work: at most the reference stream's retained sample count.
        """
        if self.reference not in streams:
            raise KeyError(f"reference stream {self.reference!r} not among "
                           f"{sorted(streams)}")

        ref = streams[self.reference]
        out: list[Frame] = []
        for t in ref.times():
            if t <= self._consumed_to:
                continue
            frame = self.align(streams, float(t))
            self._consumed_to = float(t)
            if frame is not None:
                out.append(frame)
        return out

    def __repr__(self) -> str:
        return (f"Aligner(reference={self.reference!r}, policy={self.policy!r}, "
                f"max_stale_s={self.max_stale_s}, causal={self.is_causal})")
