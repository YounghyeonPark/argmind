# SPDX-License-Identifier: Apache-2.0
"""Health checks, freezing and rollback for online adaptation.

Half the reason this library exists is that online adaptation diverges in the
field. A learner that keeps updating from a sensor that has drifted, saturated or
died will happily destroy an estimate that was working, and it will do it
quietly. So the freeze path is not a feature here; it is the shape of the API.

**A `Guard` wraps an `Adapter` and is itself an `Adapter`.** That is a deliberate
departure from the API sketch in DESIGN.md, which passed `adapter=` and `guard=`
to the pipeline side by side. Side by side leaves "no guard" expressible, and a
safety mechanism you can leave out is one that gets left out. Wrapping makes the
guarded object interchangeable with the raw one, so a `Pipeline` — or any caller
— holds exactly one adapter and cannot tell whether it is guarded, except by the
extra fields on the report it gets back.

What a guard actually does
--------------------------
Checks run *after* the adapter has updated, because the quantities worth checking
— the residual, the gain, the covariance trace — only exist once the update has
been computed. That is why rollback exists rather than pure prevention: the state
has already moved by the time anything is known about it, so the guard restores a
checkpoint instead of vetoing in advance.

The sequence per sample:

1. The adapter updates. If it rejects the sample itself (a NaN from the sensor),
   that is passed through unchanged — it is not a guard trip and must not be
   counted as one, or the two failure modes become indistinguishable in the logs.
2. Every check evaluates the report. All of them run; the guard does not stop at
   the first, because knowing that three things went wrong together is diagnostic.
3. If any tripped: restore the deepest retained checkpoint and freeze. Stateful
   checks keep their history — it is the only reference the frozen probes have
   for deciding whether the world has come back. They are reset on `thaw()`, so
   adaptation resumes with a clean detector.
4. If none tripped: take a checkpoint.

Freezing stops adaptation, not prediction. That is the whole point: the estimate
carries on from the last parameters known to be healthy, which is what a vehicle
needs, while the learner stops making things worse.

Rollback depth is memory, and memory is declared
------------------------------------------------
Checkpoints cost ``rollback_depth * (state_size + 2)`` elements, fixed at
construction — consistent with the rest of this library, where memory is a
function of configuration and never of uptime. The guard rolls back to the
*deepest* retained checkpoint rather than the most recent, because divergence is
detected after it starts; rolling back one update lands inside the bad stretch.
Depth is therefore "how far back can this undo", and the honest default is 1 with
the expectation that the caller raises it once they know their detection lag.

Recovery
--------
Frozen is terminal until `thaw()`, unless `recover_after` is set. While frozen the
guard still evaluates a *probe* report each sample — the a priori residual is
real, the gain is genuinely zero because nothing was applied, and the covariance
scalars are those of the state the guard is sitting on, which is exact because
adaptation is stopped. `recover_after` consecutive healthy probes thaw the guard.

The caveat that follows from that, stated rather than discovered later: a check on
`gain_norm` with a positive lower bound can never pass while frozen, so a guard
configured that way plus `recover_after` will never recover. `Guard` refuses that
combination at construction rather than deadlocking in the field.

What is and is not persisted
----------------------------
`state` and `load_state` delegate to the wrapped adapter, so the flat block that
crosses the C ABI stays the adapter's and nothing else. The checkpoints and the
EWMAs inside stateful checks are **not** in it and are not persisted: they are
descriptions of a run, and a run ends at power-off.

The frozen flag is different, because it is a *conclusion* rather than working
state, and `argmind.persist` carries it alongside the block with `frozen_by`. What
to do with it on load is the caller's decision and not a default this library
should make silently — see `persist.load` and its `on_frozen` policy.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

import numpy as np

from .adapt import Adapter, UpdateReport

__all__ = ["Check", "Bound", "Divergence", "Guard", "GuardReport"]

#: Report attributes a `Bound` may watch. Checked at construction so that a typo
#: fails loudly instead of producing a guard that silently never fires — which is
#: worse than no guard, because it looks like protection.
WATCHABLE: frozenset[str] = frozenset(UpdateReport.__slots__) | {
    "residual_norm", "residual_post_norm"}

#: How many caller-supplied scalars a guard can watch. Matches `kMaxExternal` in
#: `cpp/include/argmind/guard.hpp`, where the limit is the `Field` enum's room.
MAX_EXTERNAL = 4

#: Names for them. These are NOT report fields -- nothing in `UpdateReport` produces
#: them, and the adapter cannot: they are numbers the caller computes about the world
#: outside the adapter, which is the whole point. See `Guard.set_external`.
EXTERNAL: frozenset[str] = frozenset(f"external_{i}" for i in range(MAX_EXTERNAL))


def external_index(field: str) -> int | None:
    """The slot a field name refers to, or None if it is not an external."""
    if field in EXTERNAL:
        return int(field.rsplit("_", 1)[1])
    return None


@dataclass(frozen=True, slots=True)
class GuardReport(UpdateReport):
    """An `UpdateReport` plus what the guard decided about it.

    A subclass rather than a wrapper so that a guarded adapter still satisfies the
    `Adapter` contract exactly: `update()` returns an `UpdateReport`, and callers
    that do not care about guarding never have to know.

    `applied` keeps its meaning — the adaptation stands — so a rolled-back update
    reports `applied=False` even though the adapter did apply it before the guard
    took it back.
    """

    frozen: bool = False
    tripped: tuple[str, ...] = ()
    rolled_back: bool = False


class Check(abc.ABC):
    """One health test over a report.

    `evaluate` returns True for healthy, and always receives a `GuardReport` —
    including on the applied path, where the guard builds one before running the
    checks specifically so that `report.frozen` is always available. A stateful
    check needs it: `frozen` marks a probe, meaning the adapter did not move and
    the sample is being looked at rather than learned from, and a detector that
    keeps folding those into its own reference adapts to the fault it is meant to
    be holding out against.

    Stateful checks implement `reset`, which the guard calls on `thaw()` and
    whenever the adapter's state is replaced underneath it — not on a trip, where
    the history is the only reference the frozen probes have.
    """

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name

    @abc.abstractmethod
    def evaluate(self, report: "GuardReport") -> bool: ...

    @property
    @abc.abstractmethod
    def watches(self) -> frozenset[str]:
        """The report fields this check reads.

        Abstract rather than defaulted, so a new check has to say. `Guard` compares
        it against the adapter's `health_fields` and refuses a check the adapter
        cannot feed — the alternative is a check that either never fires or trips
        at once, both of which look like a configured guard.
        """

    def reset(self) -> None:
        """Stateless by default."""

    #: False for checks that cannot meaningfully pass while adaptation is frozen.
    #: Used to refuse a `recover_after` configuration that could never recover.
    usable_when_frozen: bool = True

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name!r})"


class Bound(Check):
    """A scalar report field must stay within ``[lo, hi]``.

    Non-finite is unhealthy regardless of the bounds — a NaN compares False
    against everything, so without saying this explicitly a NaN would slip
    through every comparison as "not out of range".
    """

    __slots__ = ("field", "lo", "hi", "_ext")

    def __init__(self, field: str, lo: float | None = None,
                 hi: float | None = None, name: str | None = None) -> None:
        if field not in WATCHABLE and field not in EXTERNAL:
            raise ValueError(
                f"{field!r} is not a report field a guard can watch; choose from "
                f"{sorted(WATCHABLE)} or {sorted(EXTERNAL)}. A guard watching a field "
                "that does not exist would never fire and would look like protection.")
        if lo is None and hi is None:
            raise ValueError(f"Bound({field!r}) sets neither lo nor hi, so it can "
                             "never fail. A check that cannot fail is not a check.")
        if lo is not None and hi is not None and not lo <= hi:
            raise ValueError(f"Bound({field!r}): lo={lo} exceeds hi={hi}")
        super().__init__(name or f"{field}_bound")
        self.field = field
        self.lo = lo
        self.hi = hi
        #: Set by `Guard.__init__` for an external field, so that `evaluate` keeps its
        #: one-argument signature. The C++ passes the slots to `Bound::evaluate` instead;
        #: either way the check reads the guard's live value and the decision is the same.
        #: Doing it here rather than widening `Check.evaluate` avoids breaking every
        #: check anyone has already written, including the ones in this repository's own
        #: tests.
        self._ext: list[float] | None = None

    @property
    def watches(self) -> frozenset[str]:
        return frozenset({self.field})

    @property
    def usable_when_frozen(self) -> bool:
        # gain_norm is identically zero while frozen, so a positive lower bound on
        # it can never be satisfied and recovery would never happen.
        return not (self.field == "gain_norm" and self.lo is not None and self.lo > 0)

    def evaluate(self, report: UpdateReport) -> bool:
        idx = external_index(self.field)
        if idx is not None:
            if self._ext is None:
                raise RuntimeError(
                    f"{self.name!r} watches {self.field!r} but was never bound to a "
                    "guard's slots; Guard.__init__ does that, so this check was "
                    "evaluated outside one")
            v = float(self._ext[idx])
        else:
            v = float(getattr(report, self.field))
        if not np.isfinite(v):
            return False
        if self.lo is not None and v < self.lo:
            return False
        return not (self.hi is not None and v > self.hi)

    def __repr__(self) -> str:
        return f"Bound({self.field!r}, lo={self.lo}, hi={self.hi})"


class Divergence(Check):
    """Fires when the residual grows against its own recent history.

    A fixed bound on the residual needs to know the scale of the problem, which
    for a sensor-agnostic library is exactly what is not known. This compares a
    fast exponential average of ``||residual||`` against a slow one, so it detects
    *growth* rather than magnitude and needs no units.

    `floor` matters more than it looks: on near-noiseless data the slow average
    approaches zero and any residual at all is then an infinite ratio. The floor
    is the residual magnitude below which the check declines to have an opinion,
    and leaving it at zero on clean data produces false trips.

    `warmup` samples pass unconditionally, because a ratio between two averages
    that have each seen three samples is noise.

    **The slow average is held while the guard is frozen** (``report.frozen``).
    Without that the detector adapts to the very thing it is detecting: a residual
    that jumps 80-fold and then stays there is growth for a few samples, after
    which the baseline catches up and a permanently broken sensor reads as normal.
    That was not hypothetical — it let the guard thaw itself in the middle of a
    fault during development.

    Holding it *per sample* instead — refusing any sample the check called
    unhealthy — was the first attempt and is worse. The baseline then only ever
    absorbs the smaller residuals, so it ratchets downward until ordinary noise
    trips it. Freezing is the right granularity: during normal operation the
    baseline sees everything and stays unbiased; during a fault it sees nothing
    and stays a reference.

    What this check cannot do, stated so nobody expects it to: a fault that comes
    on slowly enough for the baseline to track it will never trip. This detects
    growth against recent history. Absolute limits are `Bound`'s job.
    """

    __slots__ = ("ratio", "fast_alpha", "slow_alpha", "floor", "warmup",
                 "_fast", "_slow", "_seen")

    usable_when_frozen = True

    #: `residual_norm` and nothing else. An adapter that cannot report it would leave
    #: this comparing a constant against itself — and this is the check most likely to
    #: be enabled by a caller who does not know the scale of the problem, since it is
    #: the one that needs no units.
    watches = frozenset({"residual_norm"})

    def __init__(self, ratio: float = 5.0, fast_alpha: float = 0.25,
                 slow_alpha: float = 0.02, floor: float = 0.0,
                 warmup: int = 20, name: str = "divergence") -> None:
        if ratio <= 1.0:
            raise ValueError(f"ratio must exceed 1, got {ratio}; at or below 1 the "
                             "check fires on ordinary noise")
        for a, label in ((fast_alpha, "fast_alpha"), (slow_alpha, "slow_alpha")):
            if not 0.0 < a <= 1.0:
                raise ValueError(f"{label} must be in (0, 1], got {a}")
        if fast_alpha <= slow_alpha:
            raise ValueError(
                f"fast_alpha ({fast_alpha}) must exceed slow_alpha ({slow_alpha}); "
                "otherwise the 'fast' average is the slower of the two and the "
                "check detects the opposite of what it claims to")
        if floor < 0.0:
            raise ValueError(f"floor must be non-negative, got {floor}")
        super().__init__(name)
        self.ratio = float(ratio)
        self.fast_alpha = float(fast_alpha)
        self.slow_alpha = float(slow_alpha)
        self.floor = float(floor)
        self.warmup = int(warmup)
        self._fast = 0.0
        self._slow = 0.0
        self._seen = 0

    def reset(self) -> None:
        self._fast = 0.0
        self._slow = 0.0
        self._seen = 0

    @property
    def ewma(self) -> tuple[float, float]:
        """(fast, slow). Exposed for tests and for anyone debugging a trip."""
        return self._fast, self._slow

    def evaluate(self, report: UpdateReport) -> bool:
        r = report.residual_norm
        if not np.isfinite(r):
            return False
        self._fast += self.fast_alpha * (r - self._fast)
        self._seen += 1
        if not report.frozen:
            # Held while frozen so the baseline cannot drift onto the fault. See
            # the class docstring for why this is per-freeze and not per-sample.
            self._slow += self.slow_alpha * (r - self._slow)
        if self._seen <= self.warmup:
            return True
        reference = max(self._slow, self.floor)
        return reference <= 0.0 or self._fast <= self.ratio * reference

    def __repr__(self) -> str:
        return (f"Divergence(ratio={self.ratio}, fast_alpha={self.fast_alpha}, "
                f"slow_alpha={self.slow_alpha}, floor={self.floor})")


class Guard(Adapter):
    """An adapter that gates its own adaptation. See the module docstring.

    Parameters
    ----------
    adapter : Adapter
        The learner being guarded. The guard owns it: callers should go through
        the guard, since updating the inner adapter directly bypasses every check
        and leaves the checkpoints describing a state that no longer exists.
    checks : sequence of Check
        Run in order, all of them, after every applied update. An empty sequence
        is refused — a guard with no checks is a wrapper pretending to be a safety
        mechanism.
    rollback_depth : int
        Checkpoints retained. Memory is ``depth * (state_size + 2)`` elements,
        allocated once. Rolling back goes to the deepest retained one.
    checkpoint_every : int
        Take a checkpoint every N healthy updates. With `rollback_depth`, this
        sets how far back a rollback reaches: ``depth * checkpoint_every``
        updates.
    recover_after : int | None
        Consecutive healthy probes needed to thaw automatically. None means the
        only way out is `thaw()`, which is the right default for anything where a
        human or a supervisor should decide.
    """

    __slots__ = ("adapter", "checks", "rollback_depth", "checkpoint_every",
                 "recover_after", "_ckpt", "_ckpt_health", "_ckpt_head",
                 "_ckpt_count", "_since_ckpt", "_frozen", "_healthy_run",
                 "_frozen_health", "_frozen_by", "_stats",
                 "n_external", "_external")

    def __init__(self, adapter: Adapter, checks, rollback_depth: int = 1,
                 checkpoint_every: int = 1, recover_after: int | None = None,
                 n_external: int = 0) -> None:
        checks = tuple(checks)
        if not checks:
            raise ValueError(
                "a Guard with no checks cannot ever freeze, which makes it a "
                "wrapper that looks like a safety mechanism. Give it at least one.")
        for c in checks:
            if not isinstance(c, Check):
                raise TypeError(f"{c!r} is not a Check")
        # A check the adapter cannot feed is refused here rather than misbehaving later.
        # `p_trace` and `p_min_diag` are covariance quantities; an adapter without a
        # covariance used to return zeros, and the same Bound then broke in BOTH
        # directions -- `hi=1e4` could never fire while looking configured, and
        # `lo=1e-9` tripped on the first update and froze the adapter for the run.
        # Measured on a second adapter before this check existed.
        if not 0 <= n_external <= MAX_EXTERNAL:
            raise ValueError(f"n_external must be in [0, {MAX_EXTERNAL}], "
                             f"got {n_external}")
        self.n_external = int(n_external)
        self._external = [0.0] * self.n_external
        declared = frozenset(f"external_{i}" for i in range(self.n_external))
        reportable = adapter.health_fields | declared
        for c in checks:
            missing = sorted(c.watches - reportable)
            if missing:
                raise ValueError(
                    f"{c.name!r} watches {missing}, which {type(adapter).__name__} "
                    f"does not report (it declares {sorted(reportable)}). A check on "
                    "a field its adapter cannot produce is not a check: depending on "
                    "the direction of the bound it either can never fire or trips "
                    "immediately, and both look like a configured guard.")
        if rollback_depth < 1:
            raise ValueError(f"rollback_depth must be >= 1, got {rollback_depth}")
        if checkpoint_every < 1:
            raise ValueError(f"checkpoint_every must be >= 1, got {checkpoint_every}")
        if recover_after is not None:
            if recover_after < 1:
                raise ValueError(f"recover_after must be >= 1, got {recover_after}")
            unusable = [c.name for c in checks if not c.usable_when_frozen]
            if unusable:
                raise ValueError(
                    f"recover_after is set but {unusable} cannot pass while frozen, "
                    "so the guard could never thaw. Either drop recover_after and "
                    "thaw explicitly, or remove those checks.")

        for c in checks:
            for f in c.watches & EXTERNAL:
                # Bind before the first evaluate; `Bound.evaluate` raises if this was
                # skipped, rather than silently reading a zero.
                c._ext = self._external
        self.adapter = adapter
        self.checks = checks
        self.rollback_depth = int(rollback_depth)
        self.checkpoint_every = int(checkpoint_every)
        self.recover_after = recover_after

        n = adapter.state.size
        # Preallocated ring of checkpoints. Each carries the state block and the
        # covariance scalars that were true at that state, so a probe report built
        # after a rollback reports the truth rather than a stale number.
        self._ckpt = np.zeros((self.rollback_depth, n), dtype=adapter.state.dtype)
        self._ckpt_health = np.zeros((self.rollback_depth, 2), dtype=np.float64)
        self._ckpt_head = 0
        self._ckpt_count = 0
        self._since_ckpt = 0
        self._frozen = False
        self._healthy_run = 0
        self._frozen_health = adapter.health()
        self._frozen_by: tuple[str, ...] = ()
        self._stats = {"applied": 0, "trips": 0, "rollbacks": 0,
                       "blocked_frozen": 0, "rejected_by_adapter": 0, "thaws": 0}
        self._store_checkpoint(*adapter.health())

    # -- Adapter contract, delegated ----------------------------------------

    @property
    def n_features(self) -> int:
        return self.adapter.n_features

    @property
    def n_outputs(self) -> int:
        return self.adapter.n_outputs

    @property
    def health_fields(self) -> frozenset[str]:
        """What the wrapped adapter reports. The guard *is* an adapter, so a guard
        wrapping a guard has to see through to the one that actually computes.

        Externals are deliberately absent: they are this guard's configuration, not
        something it can report on another's behalf.
        """
        return self.adapter.health_fields

    def set_external(self, i: int, value: float) -> None:
        """Supply one of the caller's scalars, before `update()`.

        Every check the guard has watches the **adapter's** health, and for a frozen
        backbone feeding an adapted head that is the wrong place to look: the head stays
        healthy while the world moves under it, because it fits whatever it is given. An
        input distribution shift reaches an early layer's activation statistics long
        before it reaches the head's residual, and only the caller -- who is running the
        backbone -- can compute that. The guard cannot produce the number, but it can
        watch it, and then a shift freezes adaptation with a cause, a counter and
        automatic recovery.

        Held until overwritten, so a backbone slower than the head need not invent a
        value per sample. Non-finite is accepted here on purpose: `Bound` already treats
        non-finite as unhealthy, and a NaN out of a runtime is a runtime event the guard
        should act on rather than an argument error the caller should have caught.
        """
        if not 0 <= i < self.n_external:
            raise IndexError(
                f"external {i} was not declared; this guard has {self.n_external}")
        self._external[i] = float(value)

    def external(self, i: int) -> float:
        if not 0 <= i < self.n_external:
            raise IndexError(
                f"external {i} was not declared; this guard has {self.n_external}")
        return self._external[i]

    @property
    def dtype(self) -> np.dtype:
        return self.adapter.dtype

    @property
    def state(self) -> np.ndarray:
        """The wrapped adapter's block. The guard's own state is not in here —
        see the module docstring, this is a stated boundary and not an oversight."""
        return self.adapter.state

    def load_state(self, flat) -> None:
        self.adapter.load_state(flat)
        self._rearm()

    def reset(self) -> None:
        self.adapter.reset()
        self._rearm()

    def predict(self, x) -> np.ndarray:
        """Unaffected by freezing. Freezing stops adaptation, not the estimate."""
        return self.adapter.predict(x)

    def health(self) -> tuple[float, float]:
        """Straight through. Correct while frozen too: the state is not moving,
        so the wrapped adapter's own answer is still the current one."""
        return self.adapter.health()

    def _rearm(self) -> None:
        """Drop guard history and re-arm against whatever state the adapter is in
        now. Called when the state is replaced underneath the guard, because every
        checkpoint and every EWMA describes a state that no longer exists."""
        for c in self.checks:
            c.reset()
        self._ckpt_head = 0
        self._ckpt_count = 0
        self._since_ckpt = 0
        self._frozen = False
        self._healthy_run = 0
        self._frozen_by = ()
        self._frozen_health = self.adapter.health()
        self._store_checkpoint(*self._frozen_health)

    # -- freezing ------------------------------------------------------------

    @property
    def frozen(self) -> bool:
        return self._frozen

    @property
    def frozen_by(self) -> tuple[str, ...]:
        """What froze it last — the checks that tripped, or the reason given to
        `freeze()`. Kept after `thaw()` as well, because "why did this device
        freeze yesterday" is the question someone actually asks."""
        return self._frozen_by

    @property
    def stats(self) -> dict[str, int]:
        """A guard nobody can inspect is a guard nobody will trust. `trips` and
        `rejected_by_adapter` are kept apart deliberately: divergence and a bad
        sensor sample are different faults and must not be summed."""
        return dict(self._stats)

    def freeze(self, reason: str = "manual") -> None:
        """Stop adaptation without rolling back. For a supervisor that knows something the
        checks cannot see — a calibration in progress, a bay door open, a known-bad sensor.

        The health scalars come from the adapter as it stands, not from the newest
        checkpoint. That was a bug, and it was found by a mutation check on the C++ port
        rather than by review: a manual freeze does not move the state, so the live values
        are exactly right, while the newest checkpoint can be up to `checkpoint_every`
        updates stale — measured at 300.0 against a live 104.1 three updates into a cadence
        of five. A `Bound` on `p_trace` evaluated during a manual freeze therefore read a
        number describing a state the adapter was no longer in.
        """
        self._frozen = True
        self._healthy_run = 0
        self._frozen_by = (reason,)
        self._frozen_health = self.adapter.health()

    def thaw(self) -> None:
        """Resume adaptation from the current parameters."""
        if self._frozen:
            self._stats["thaws"] += 1
        self._frozen = False
        self._healthy_run = 0
        for c in self.checks:
            c.reset()

    # -- checkpoints ---------------------------------------------------------

    def _newest(self) -> int:
        return (self._ckpt_head - 1) % self.rollback_depth

    def _oldest(self) -> int:
        return (self._ckpt_head - self._ckpt_count) % self.rollback_depth

    def _store_checkpoint(self, p_trace: float, p_min_diag: float) -> None:
        self._ckpt[self._ckpt_head] = self.adapter.state
        self._ckpt_health[self._ckpt_head] = (p_trace, p_min_diag)
        self._ckpt_head = (self._ckpt_head + 1) % self.rollback_depth
        self._ckpt_count = min(self._ckpt_count + 1, self.rollback_depth)

    def _roll_back(self) -> tuple[float, float]:
        i = self._oldest()
        self.adapter.load_state(self._ckpt[i])
        health = (float(self._ckpt_health[i, 0]), float(self._ckpt_health[i, 1]))
        # Everything retained is at or after the state just restored, so keeping it
        # would let a later trip roll "back" to a moment that came afterwards.
        self._ckpt_count = 1
        self._ckpt_head = (i + 1) % self.rollback_depth
        self._since_ckpt = 0
        return health

    # -- the work ------------------------------------------------------------

    def _failing(self, report: UpdateReport) -> tuple[str, ...]:
        """Every failing check, not just the first — three faults at once is
        diagnostic information and collapsing it to one throws that away."""
        return tuple(c.name for c in self.checks if not c.evaluate(report))

    def _probe(self, x, y) -> GuardReport:
        """What the sample looks like without adapting to it.

        The residual is real. `gain_norm` is zero because nothing was applied. The
        covariance scalars describe the state the guard is sitting on — the restored
        checkpoint after a trip, the live state after a manual `freeze()` — and are exact
        in both cases, because adaptation is stopped and the state is not moving.

        An earlier version justified reading the stored checkpoint value by saying the
        state had not moved *since it was restored*. That is true after a trip and false
        after a manual freeze, where nothing was restored at all.
        """
        ya = np.asarray(y, dtype=self.adapter.dtype).reshape(-1)
        if ya.size != self.n_outputs:
            raise ValueError(f"y must have {self.n_outputs} outputs, got {ya.size}")
        resid = ya - self.adapter.predict(x)
        trace, min_diag = self._frozen_health
        return GuardReport(applied=False, residual=resid, residual_post=resid,
                           gain_norm=0.0, p_trace=trace, p_min_diag=min_diag,
                           reason="frozen", frozen=True)

    def update(self, x, y) -> UpdateReport:
        if self._frozen:
            self._stats["blocked_frozen"] += 1
            probe = self._probe(x, y)
            if not self._failing(probe):
                self._healthy_run += 1
            else:
                self._healthy_run = 0
            if (self.recover_after is not None
                    and self._healthy_run >= self.recover_after):
                self.thaw()
            return probe

        report = self.adapter.update(x, y)
        if not report.applied:
            # The adapter refused the data itself. Not a guard trip: no checkpoint,
            # no rollback, and counted separately so the two never blur together.
            self._stats["rejected_by_adapter"] += 1
            return GuardReport(
                applied=False, residual=report.residual,
                residual_post=report.residual_post, gain_norm=report.gain_norm,
                p_trace=report.p_trace, p_min_diag=report.p_min_diag,
                reason=report.reason, frozen=False)

        # Built before the checks run, not after, so that every check sees a
        # GuardReport and `frozen` is always meaningful rather than sometimes
        # absent. It is returned unchanged if nothing trips.
        provisional = GuardReport(
            applied=True, residual=report.residual,
            residual_post=report.residual_post, gain_norm=report.gain_norm,
            p_trace=report.p_trace, p_min_diag=report.p_min_diag,
            reason=None, frozen=False)

        tripped = self._failing(provisional)
        if tripped:
            health = self._roll_back()
            self._frozen = True
            self._frozen_health = health
            self._frozen_by = tripped
            self._healthy_run = 0
            # Stateful checks are deliberately NOT reset here. Their history is
            # what tells the frozen probes whether the world has come back, and
            # wiping it leaves the guard with no reference except the fault it
            # just froze on. They are reset on thaw instead, so adaptation
            # resumes with a clean detector.
            self._stats["trips"] += 1
            self._stats["rollbacks"] += 1
            return GuardReport(
                applied=False, residual=report.residual,
                residual_post=report.residual_post, gain_norm=report.gain_norm,
                p_trace=health[0], p_min_diag=health[1],
                reason=f"guard:{tripped[0]}", frozen=True, tripped=tripped,
                rolled_back=True)

        self._stats["applied"] += 1
        self._since_ckpt += 1
        if self._since_ckpt >= self.checkpoint_every:
            self._store_checkpoint(report.p_trace, report.p_min_diag)
            self._since_ckpt = 0
        return provisional

    def __repr__(self) -> str:
        return (f"Guard({self.adapter!r}, checks={[c.name for c in self.checks]}, "
                f"rollback_depth={self.rollback_depth}, frozen={self._frozen})")
