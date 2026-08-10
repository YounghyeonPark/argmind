# SPDX-License-Identifier: Apache-2.0
"""Online adaptation: the `Adapter` contract, and recursive least squares.

An `Adapter` is the online learner. It sees `(x, y)` pairs one at a time, keeps
bounded state, and answers `predict(x)`. Everything later in this library —
Kalman-family parameter estimation, online gradient methods — has to fit the
shape declared here, so the shape is chosen for the constraints that actually
bind rather than for Python's convenience:

- **Dimensions are fixed at construction.** Nothing about the state depends on how
  many samples have been seen.
- **The state is one flat, contiguous block.** `adapter.state` is the whole thing.
  That is what makes persistence a memcpy and what lets the same state cross a C
  ABI later without a serialisation format in the middle.
- **`update()` returns a report, never nothing.** The `Guard` that comes next
  decides whether to freeze on what a report says, so an adapter that only
  mutates state silently would make the guard impossible to write.
- **A rejected update leaves the state untouched**, and says why it was rejected.
  In the field a NaN sample must not take the estimator with it, and must not
  vanish either.

One honest caveat about "no allocation in the steady state": the *recursion* here
allocates nothing — every intermediate is preallocated scratch and every operation
writes through ``out=``. The boundary is not allocation-free, because `np.asarray`
at the input and the small copies inside `UpdateReport` do allocate. That is a
property of the Python reference, not of the contract; the C++ core takes the
report by pointer and has no boundary conversion to do. Do not read the Python
timings as embedded numbers.

Precision
---------
DESIGN.md carries a measured finding from the parked project: *precision floors
are a property of the algorithm, not of the platform*. That was measured for the
EKF covariance recursion, so it is not evidence about this one — inheriting the
conclusion would be exactly the borrowed-assumption error this project's
standards exist to catch. It was therefore re-measured here, for RLS, and the
numbers are in `tests/test_adapt.py::test_fp16_refusal_is_backed_by_measurement`:

- **fp16 fails outright**, and not because of dynamic range. With every regressor
  held inside fp16's range and the regressor correlation perfectly conditioned
  (κ=1), the conventional recursion still lost positive-definiteness by step 115;
  at κ=10 it went by step 3, and the parameter error reached NaN. fp16 is
  refused at construction.
- **fp32 survives, but does not maintain positive-definiteness exactly.** With
  ``p0=1e2`` and well-conditioned regressors of magnitude ≥1e2, fp32 showed a
  negative eigenvalue on the *first* update — at roundoff scale (≈-5e-7 against a
  trace of order 1e2), not a blowup, and the parameter estimates stayed sound.
  fp64 never lost it across the same sweep up to magnitude 1e6.

So fp32 is the floor and it is a real floor, not a comfortable one: the
conventional form is the weak one, which is why square-root and UD factorisations
exist. If the roundoff-scale indefiniteness ever matters for a target, the fix is
a factored form, not more precision. `p_min_diag` is reported on every update so
that condition is observable rather than theoretical.

Note that ``p0`` interacts with the regressor scale: it is a prior covariance, so
``p0 * ||x||^2 >> 1`` is what puts the first update in the cancellation regime
above. Scale it to the problem rather than reaching for a large default.
"""

from __future__ import annotations

import abc
import math
from dataclasses import dataclass

import numpy as np

__all__ = ["Adapter", "UpdateReport", "RecursiveLeastSquares"]

#: Storage precisions an adapter may be built with. fp16 is absent deliberately;
#: see the module docstring — it was measured failing, not assumed to.
SUPPORTED_DTYPES: tuple[str, ...] = ("f4", "f8")


@dataclass(frozen=True, slots=True)
class UpdateReport:
    """What one `update()` did, and what the state looks like afterwards.

    This is the observability surface the `Guard` will consume, so it carries the
    quantities a health check needs rather than the ones an adapter finds
    convenient to expose.

    Attributes
    ----------
    applied : bool
        False means the state was not touched. `reason` says why.
    reason : str | None
        None when applied. Otherwise a short, stable token — these are matched
        on by guards and by tests, so they are part of the API.
    residual : np.ndarray
        The *a priori* error ``y - predict(x)``, computed before the update. This
        is the honest one: the a posteriori error always looks better because the
        adapter has already seen the answer.
    residual_post : np.ndarray
        The same error recomputed after the update. The pair is diagnostic: a gap
        that stays wide means the model cannot express the data; a gap that
        collapses every step means the adapter is chasing noise.
    gain_norm : float
        ``||k||``. A gain that grows over time is the signature of covariance
        windup under poor excitation.
    p_trace : float
        ``tr(P)``. Grows like ``lambda**-k`` in directions the data does not
        excite. This is the classic RLS failure and it is left visible rather
        than clamped — see `RecursiveLeastSquares`.
    p_min_diag : float
        Smallest diagonal of ``P``. Negative proves ``P`` is no longer positive
        semi-definite. It is an O(n) proxy for the O(n^3) eigenvalue test, chosen
        because this has to be affordable on every update on an MCU.
    """

    applied: bool
    residual: np.ndarray
    residual_post: np.ndarray
    gain_norm: float
    p_trace: float
    p_min_diag: float
    reason: str | None = None

    @property
    def residual_norm(self) -> float:
        return float(np.linalg.norm(self.residual))

    @property
    def residual_post_norm(self) -> float:
        return float(np.linalg.norm(self.residual_post))


class Adapter(abc.ABC):
    """The online learner contract. See the module docstring for why this shape."""

    #: Declared empty so that subclasses declaring `__slots__` actually get them.
    #: Without this the base contributes a `__dict__` and every "slotted" adapter
    #: silently carries one, which on the reference side is only untidy but sets
    #: the wrong expectation for a type that is supposed to be a fixed block.
    __slots__ = ()

    #: The coarsest storage precision this method may be built with. Declared per
    #: adapter because the floor belongs to the algorithm, not to the platform.
    min_dtype: str = "f4"

    #: Which report fields this adapter can actually report, so that a `Guard` can
    #: refuse to watch one it cannot.
    #:
    #: Deliberately **not** given a default. A subclass that forgets it raises
    #: `AttributeError` the first time a guard wraps it, which is loud; a default of
    #: "all of them" would silently hand back the defect this attribute exists to fix.
    #:
    #: `p_trace` and `p_min_diag` are *covariance* quantities. Before this existed, an
    #: adapter without a covariance had to invent two numbers, and zeros broke a bound
    #: in both directions at once — `hi=1e4` became a check that could never fire while
    #: looking configured, and `lo=1e-9` tripped on the first update and froze the
    #: adapter for the whole run. Both measured on a second adapter in the C++ suite.
    health_fields: frozenset[str]

    n_features: int
    n_outputs: int
    dtype: np.dtype

    @abc.abstractmethod
    def update(self, x, y) -> UpdateReport:
        """Take one observation. Must not raise on bad *data* — reject and report."""

    @abc.abstractmethod
    def predict(self, x) -> np.ndarray:
        """Current model output for `x`, shape ``(n_outputs,)``."""

    def health(self) -> tuple[float, float]:
        """``(p_trace, p_min_diag)`` for the state as it stands, without updating.

        The same two scalars `UpdateReport` carries, available without consuming a
        sample. A `Guard` needs them at moments when no update happened — when it
        checkpoints the construction state, and while it is frozen — and inventing
        a plausible-looking zero there would be fabricating a number, which is
        precisely what this project's standards forbid.

        **Required only if `health_fields` declares `p_trace` or `p_min_diag`.**
        This used to be abstract, and its docstring told an adapter with no
        covariance to "report the nearest analogue" — two sentences after saying
        that inventing a number is forbidden. It was a contradiction, and the
        instruction was the wrong half: there is no analogue of a covariance trace
        in an algorithm that has no covariance, and a probe adapter measured what
        the fabricated zeros do to a guard. Declare fewer fields instead; the base
        returns zeros that nothing can read, because a `Guard` refuses a `Bound` on
        an undeclared field.
        """
        return 0.0, 0.0

    @property
    @abc.abstractmethod
    def state(self) -> np.ndarray:
        """The entire adaptation state as one flat, contiguous, read-only array.

        Read-only because this is a *view*, not a copy: writing through it would
        edit the live adapter behind its own validation. Use `load_state` to put
        one back.
        """

    @abc.abstractmethod
    def load_state(self, flat) -> None:
        """Restore a block previously read from `state`."""

    @abc.abstractmethod
    def reset(self) -> None:
        """Return to the construction-time state."""

    # -- shared validation ---------------------------------------------------

    @classmethod
    def _check_dtype(cls, dtype) -> np.dtype:
        dt = np.dtype(dtype)
        if dt.str[1:] not in SUPPORTED_DTYPES:
            raise ValueError(
                f"{cls.__name__}: dtype {dt.name!r} is not supported; use one of "
                f"{SUPPORTED_DTYPES}. Half precision is refused because the "
                "covariance recursion was measured losing positive-definiteness "
                "in fp16 even with perfectly conditioned regressors — see the "
                "argmind.adapt module docstring. This is a property of the "
                "algorithm; it is not a portability limitation to work around.")
        if np.dtype(dt).itemsize < np.dtype(cls.min_dtype).itemsize:
            raise ValueError(
                f"{cls.__name__}: requires at least {cls.min_dtype!r}, got {dt.name!r}")
        return dt


class RecursiveLeastSquares(Adapter):
    """Exponentially weighted recursive least squares.

    Minimises ``sum_i lambda^(k-i) ||y_i - theta^T x_i||^2`` plus the prior term
    ``lambda^k * theta^T P0^-1 theta``, updated one sample at a time in ``O(n^2)``
    time and ``O(n^2)`` memory that never grows. It is first in this library
    because its stability can be argued about, and because online calibration of
    a sensor bias is literally this problem.

    Multiple outputs share one ``P``. That is not a shortcut: when several outputs
    are driven by the *same* regressor — three bias axes against one temperature
    model, say — the correlation matrix genuinely is common, and sharing it makes
    the multi-output case exactly equivalent to running independent single-output
    adapters, which is asserted in the tests.

    Parameters
    ----------
    n_features : int
        Length of the regressor `x`.
    n_outputs : int
        Length of `y`. Default 1.
    forgetting : float
        ``lambda`` in (0, 1]. 1.0 is infinite memory: the estimate converges and
        then stops responding, which is right for a constant and wrong for
        anything that drifts. Below 1 the effective memory is about
        ``1 / (1 - lambda)`` samples — see `memory_horizon_samples`. That number
        is the tuning knob worth thinking in; ``lambda`` itself is not intuitive.
    p0 : float
        Initial covariance ``P = p0 * I``. Large means "no prior confidence", so
        the first samples move `theta` a long way. Read the precision note in the
        module docstring before making it large: ``p0 * ||x||^2 >> 1`` is what
        drives the first update into fp32 cancellation.
    dtype : str
        Storage and arithmetic precision. fp32 minimum, measured.

    Divergence is left visible, deliberately
    ----------------------------------------
    Under poor excitation — a stationary vehicle, a constant temperature — ``P``
    grows like ``lambda**-k`` in the unexcited directions, the gain grows with it,
    and the first sample that finally arrives off that direction produces a burst.
    This is the classic RLS failure and it is the reason the design driver names
    "must not diverge while the vehicle is stationary".

    This class does **not** clamp it. Bounding the trace here would make the
    `Guard` optional, which is precisely the mistake DESIGN.md warns against:
    freezing is meant to be the default shape of the API, not a bonus feature
    buried in one adapter. So windup is reported (`p_trace`, `gain_norm`) on every
    update and there is a test that watches it grow; deciding what to do about it
    belongs to the guard.
    """

    min_dtype = "f4"

    #: All five. The first three come out of any update; the last two are this
    #: algorithm's covariance, which it has. `argmind.guard.WATCHABLE` is the set of
    #: fields that exist at all, and a test ties the two together.
    health_fields = frozenset({"residual_norm", "residual_post_norm", "gain_norm",
                               "p_trace", "p_min_diag"})

    __slots__ = ("n_features", "n_outputs", "dtype", "forgetting", "p0",
                 "_state", "_state_ro", "_theta", "_p", "_px", "_k", "_e", "_e_post",
                 "_yhat", "_pn", "_tn", "_psym",
                 "_updates", "_rejected_nonfinite", "_rejected_denominator")

    def __init__(self, n_features: int, n_outputs: int = 1,
                 forgetting: float = 0.999, p0: float = 1e2,
                 dtype: str = "f4") -> None:
        if n_features < 1:
            raise ValueError(f"n_features must be >= 1, got {n_features}")
        if n_outputs < 1:
            raise ValueError(f"n_outputs must be >= 1, got {n_outputs}")
        if not 0.0 < forgetting <= 1.0:
            raise ValueError(
                f"forgetting must be in (0, 1], got {forgetting}. Above 1 amplifies "
                "old data instead of discounting it; at or below 0 it is not a "
                "weighting at all.")
        if not p0 > 0.0 or not math.isfinite(p0):
            # `not p0 > 0.0` alone lets `+inf` through, and an independent parity check
            # found it: the C++ `Rls` has always tested `is_finite(p0)` and the reference
            # never did, so the two disagreed on exactly one cell of a 108-point sweep.
            #
            # What it produces is worse than a wrong number. `eye(n) * inf` is `inf` on the
            # diagonal and `0 * inf = NaN` off it, so the adapter is born with a non-finite
            # covariance, every denominator is then NaN, and **every update for the rest of
            # its life is rejected** as `nonpositive_denominator`. A dead adapter that said
            # nothing at construction. numpy does emit a RuntimeWarning, which is exactly
            # the kind of warning nobody reads.
            #
            # `forgetting` needs no such clause: `not 0.0 < forgetting <= 1.0` already
            # rejects both infinities and NaN through the range test.
            raise ValueError(
                f"p0 must be positive and finite, got {p0}")

        self.dtype = self._check_dtype(dtype)
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.forgetting = float(forgetting)
        self.p0 = float(p0)

        n, m = n_features, n_outputs
        # One block: theta (n*m) then P (n*n). Persisting the adapter is a copy of
        # this and nothing else, and it is what crosses the C ABI later.
        self._state = np.zeros(n * m + n * n, dtype=self.dtype)
        self._theta = self._state[:n * m].reshape(n, m)      # views, not copies
        self._p = self._state[n * m:].reshape(n, n)
        self._state_ro = self._state.view()
        self._state_ro.flags.writeable = False

        # Scratch, preallocated. The recursion below allocates nothing.
        self._px = np.zeros(n, dtype=self.dtype)
        self._k = np.zeros(n, dtype=self.dtype)
        self._e = np.zeros(m, dtype=self.dtype)
        self._e_post = np.zeros(m, dtype=self.dtype)
        self._yhat = np.zeros(m, dtype=self.dtype)
        self._pn = np.zeros((n, n), dtype=self.dtype)
        self._tn = np.zeros((n, m), dtype=self.dtype)
        self._psym = np.zeros((n, n), dtype=self.dtype)

        self._updates = 0
        self._rejected_nonfinite = 0
        self._rejected_denominator = 0
        self.reset()

    # -- properties ----------------------------------------------------------

    @property
    def theta(self) -> np.ndarray:
        """Current parameters, shape ``(n_features, n_outputs)``. A copy — a
        caller holding these must not have them mutated by the next update."""
        return self._theta.copy()

    @property
    def state(self) -> np.ndarray:
        """The persistence block, read-only.

        The view is built once at construction and handed out unchanged. Building
        a fresh one per call allocated a small object every time, on a path the
        guard takes on every checkpoint — the sort of thing that is invisible
        until something measures it. It stays valid because `reset` and
        `load_state` write *into* the buffer rather than replacing it.
        """
        return self._state_ro

    @property
    def state_size(self) -> int:
        """Elements in the persistence block. Fixed at construction, by design."""
        return self._state.size

    @property
    def memory_horizon_samples(self) -> float:
        """Roughly how many past samples still carry weight. ``inf`` at
        ``lambda == 1``, which is the honest answer: nothing is ever forgotten."""
        return float("inf") if self.forgetting >= 1.0 else 1.0 / (1.0 - self.forgetting)

    @property
    def stats(self) -> dict[str, int]:
        """Rejections are counted, not just reported per-call. A stream that is
        10% NaN produces a working-looking adapter and a wrong one."""
        return {"updates": self._updates,
                "rejected_nonfinite": self._rejected_nonfinite,
                "rejected_denominator": self._rejected_denominator}

    # -- state ---------------------------------------------------------------

    def reset(self) -> None:
        self._theta[:] = 0
        self._p[:] = np.eye(self.n_features, dtype=self.dtype) * self.dtype.type(self.p0)

    def load_state(self, flat) -> None:
        """Restore a block from `state`. Validates rather than trusting it.

        Persisted state comes back from flash after a power cycle, which is
        exactly where a truncated write or a bit flip shows up. Loading a corrupt
        block silently would move the failure to somewhere unrecognisable.
        """
        arr = np.asarray(flat, dtype=self.dtype).reshape(-1)
        if arr.size != self._state.size:
            raise ValueError(
                f"state block is {arr.size} elements, expected {self._state.size} "
                f"for n_features={self.n_features}, n_outputs={self.n_outputs}")
        if not np.all(np.isfinite(arr)):
            raise ValueError("state block contains non-finite values; refusing to "
                             "load. A stored NaN would silently poison every "
                             "subsequent update.")
        self._state[:] = arr

    # -- the work ------------------------------------------------------------

    def predict(self, x) -> np.ndarray:
        arr = self._as_x(x)
        return np.dot(arr, self._theta)

    def _as_x(self, x) -> np.ndarray:
        arr = np.asarray(x, dtype=self.dtype).reshape(-1)
        if arr.size != self.n_features:
            raise ValueError(
                f"x must have {self.n_features} features, got {arr.size}")
        return arr

    def _as_y(self, y) -> np.ndarray:
        arr = np.asarray(y, dtype=self.dtype).reshape(-1)
        if arr.size != self.n_outputs:
            raise ValueError(f"y must have {self.n_outputs} outputs, got {arr.size}")
        return arr

    def health(self) -> tuple[float, float]:
        return (float(np.trace(self._p.astype(np.float64))),
                float(np.min(np.diagonal(self._p))))

    def _report(self, applied: bool, reason: str | None) -> UpdateReport:
        trace, min_diag = self.health()
        return UpdateReport(
            applied=applied,
            residual=self._e.copy(),
            residual_post=self._e_post.copy(),
            gain_norm=float(np.linalg.norm(self._k)),
            p_trace=trace,
            p_min_diag=min_diag,
            reason=reason)

    def update(self, x, y) -> UpdateReport:
        """Take one observation and report what happened.

        Shape errors raise — those are programming mistakes and should fail at the
        call site. Bad *values* do not raise: they are rejected, counted and
        reported, because a NaN from a sensor is a runtime event and taking the
        estimator down with it is the wrong response in a deployed system.
        """
        xa = self._as_x(x)
        ya = self._as_y(y)

        if not (np.all(np.isfinite(xa)) and np.all(np.isfinite(ya))):
            self._rejected_nonfinite += 1
            self._e[:] = 0
            self._e_post[:] = 0
            return self._report(False, "nonfinite_input")

        # a priori residual, before the state moves. Computed first because it is
        # the only error figure that means anything for monitoring.
        np.dot(xa, self._theta, out=self._yhat)
        np.subtract(ya, self._yhat, out=self._e)

        np.dot(self._p, xa, out=self._px)
        den = self.dtype.type(self.forgetting) + np.dot(xa, self._px)

        if not np.isfinite(den) or den <= 0:
            # Reachable when P has lost positive-definiteness, or when p0 and the
            # regressor scale overflow the declared precision. Either way the
            # gain would be meaningless; refuse rather than propagate it.
            self._rejected_denominator += 1
            self._e_post[:] = self._e
            return self._report(False, "nonpositive_denominator")

        np.divide(self._px, den, out=self._k)

        np.outer(self._k, self._e, out=self._tn)
        np.add(self._theta, self._tn, out=self._theta)

        np.outer(self._k, self._px, out=self._pn)
        np.subtract(self._p, self._pn, out=self._p)
        np.divide(self._p, self.dtype.type(self.forgetting), out=self._p)

        # Symmetrise. The rank-1 subtraction is symmetric in exact arithmetic and
        # is not in floating point, and the asymmetry compounds. Cheap at these
        # sizes; it does not restore positive-definiteness, which is why
        # p_min_diag is reported rather than assumed.
        np.add(self._p, self._p.T, out=self._psym)
        np.multiply(self._psym, self.dtype.type(0.5), out=self._psym)
        self._p[:] = self._psym

        np.dot(xa, self._theta, out=self._yhat)
        np.subtract(ya, self._yhat, out=self._e_post)

        self._updates += 1
        return self._report(True, None)


    def widened(self, n_features: int, n_outputs: int | None = None,
                p0: float | None = None) -> "RecursiveLeastSquares":
        """A wider adapter that keeps everything this one learned.

        **Appending only, and that is a constraint rather than a simplification.**
        Feature order is parameter order, and a persisted block restored against a
        different order is wrong undetectably -- which is why `persist` carries a
        configuration fingerprint at all. Appending is the one widening that cannot
        permute anything: old feature `j` is still feature `j`. A widening that
        rearranged would need a mapping the caller supplies, and guessing one is exactly
        the failure the fingerprint exists to prevent.

        What carries over, and why it is principled rather than a copy:

        - ``theta`` keeps its learned coefficients; the new ones start at zero, which is
          what "no evidence yet" means for a coefficient.
        - ``P`` keeps its learned block in the corner and the new diagonal gets ``p0``.
          ``P`` is the inverse information matrix, so the old block carries the
          confidence this adapter *earned* and the new diagonal says "nothing is known
          about these yet". Cross terms are zero: no evidence yet relates the two sets.

        **This does not make the model class richer.** It adds inputs, not
        expressiveness -- the adapter is still linear in the parameters and the caller
        still supplies the basis.

        Measured, six features becoming twelve, error against the truth on the original
        six: at 10 updates 0.3278 from zero against 0.0010 grafted, at 50 0.0039 against
        0.0010, at 600 0.0014 against 0.0007. A large head start that decays to a small
        persistent one. A prediction of mine that it would matter MORE when the new
        features are rarely excited was **wrong and measured to be wrong**: the old
        features are excited either way, so a fresh adapter relearns them just as fast.

        `p0` defaults to this adapter's, and `forgetting` and `dtype` always carry over --
        a widening that quietly changed the forgetting factor would look like the graft
        failing.
        """
        n_outputs = self.n_outputs if n_outputs is None else int(n_outputs)
        p0 = self.p0 if p0 is None else float(p0)
        if n_features < self.n_features:
            raise ValueError(
                f"widened() only appends: {n_features} is narrower than "
                f"{self.n_features}. Dropping a feature needs a mapping this cannot "
                "guess, and guessing would put learned values on the wrong features.")
        if n_outputs != self.n_outputs:
            raise ValueError(
                f"widened() cannot change the number of outputs "
                f"({self.n_outputs} to {n_outputs}): theta is stored row-major over "
                "features, so a different width interleaves the old coefficients "
                "differently and there is no correct answer without a mapping.")

        out = RecursiveLeastSquares(n_features, n_outputs, forgetting=self.forgetting,
                                    p0=p0, dtype=self.dtype.name)
        n_old, m = self.n_features, self.n_outputs
        out._theta[:n_old, :] = self._theta
        out._p[:n_old, :n_old] = self._p
        return out

    def __repr__(self) -> str:
        return (f"RecursiveLeastSquares(n_features={self.n_features}, "
                f"n_outputs={self.n_outputs}, forgetting={self.forgetting}, "
                f"dtype={self.dtype.name!r}, updates={self._updates})")

class NormalisedLeastMeanSquares(Adapter):
    """Normalised LMS: the cheap adapter, for when RLS's covariance will not fit.

    ``theta += (mu / (eps + x.x)) * outer(x, y - x.theta)``

    **Why a second adapter exists at all.** RLS carries an ``n x n`` covariance, so its
    state is ``n*m + n*n`` and a `Guard`'s checkpoint ring multiplies that by its depth.
    At the handful of sensor channels this library was designed against that is nothing;
    at the width a network embedding has it is the binding constraint — 128 features cost
    330 KB against this adapter's 2.6 KB, and 512 features do not fit an STM32N6's SRAM at
    all. NLMS is ``O(n)`` in both memory and time and has no covariance whatsoever.

    **What it gives up.** RLS is a recursive exact solution of a weighted least-squares
    problem; this is gradient descent with a step normalised by the input power. It
    converges more slowly, its steady-state error is set by ``mu`` rather than driven to
    the prior term, and it has no notion of per-direction confidence — which is also why
    it cannot wind up the way RLS does under poor excitation, and why it reports no
    ``p_trace``. The trade is priced rather than asserted: see `health_fields`.

    **The one exact property, and the tests assert it rather than a tolerance.** After an
    update the a posteriori error is ``(1 - mu * x.x / (eps + x.x))`` times the a priori
    one, exactly, for any data. At ``mu = 1`` and ``eps`` negligible that is zero: one
    sample is projected out completely. This is the analogue of RLS being checked against
    the closed-form weighted solution rather than against itself.

    ``mu`` must be in ``(0, 2]``: outside that the recursion is unstable for the standard
    reason, and 2 is the boundary rather than a safe value.

    Precision: ``min_dtype`` is ``f4``, the same as RLS, but **for a different reason and
    a weaker one**. RLS's floor was measured — fp16 loses positive-definiteness in the
    covariance recursion by step 115 even at a condition number of 1. This adapter has no
    covariance and that argument does not transfer, so fp16 might well work. Nobody has
    measured it, and writing down untested support is how a claim becomes a fiction.
    """

    min_dtype = "f4"

    #: Three of five. `p_trace` and `p_min_diag` are covariance quantities and there is
    #: no covariance here, so they are declared absent rather than reported as zero --
    #: a `Guard` then refuses a bound on them at construction. Reporting zeros instead
    #: was measured to break the same bound in both directions at once.
    health_fields = frozenset({"residual_norm", "residual_post_norm", "gain_norm"})

    __slots__ = ("n_features", "n_outputs", "dtype", "mu", "eps",
                 "_state", "_state_ro", "_theta", "_e", "_e_post", "_yhat", "_tn",
                 "_updates", "_rejected_nonfinite", "_rejected_denominator")

    def __init__(self, n_features: int, n_outputs: int = 1,
                 mu: float = 0.5, eps: float = 1e-6, dtype: str = "f4") -> None:
        if n_features < 1:
            raise ValueError(f"n_features must be >= 1, got {n_features}")
        if n_outputs < 1:
            raise ValueError(f"n_outputs must be >= 1, got {n_outputs}")
        if not 0.0 < mu <= 2.0 or not math.isfinite(mu):
            raise ValueError(
                f"mu must be in (0, 2], got {mu}. Outside that the recursion is "
                "unstable; 2 is the boundary, not a safe value.")
        if not eps > 0.0 or not math.isfinite(eps):
            # Guards the zero regressor. Without it a silent stretch of all-zero input
            # divides by zero rather than doing nothing, which is what it should do.
            raise ValueError(f"eps must be positive and finite, got {eps}")

        self.dtype = self._check_dtype(dtype)
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.mu = float(mu)
        self.eps = float(eps)

        # One flat contiguous block, theta only. The same layout RLS uses for its
        # leading `n*m` elements, so `persist` needs no special case.
        self._state = np.zeros(n_features * n_outputs, dtype=self.dtype)
        self._state_ro = self._state.view()
        self._state_ro.flags.writeable = False
        self._theta = self._state.reshape(n_features, n_outputs)

        self._e = np.zeros(n_outputs, dtype=self.dtype)
        self._e_post = np.zeros(n_outputs, dtype=self.dtype)
        self._yhat = np.zeros(n_outputs, dtype=self.dtype)
        self._tn = np.zeros((n_features, n_outputs), dtype=self.dtype)
        self._updates = 0
        self._rejected_nonfinite = 0
        self._rejected_denominator = 0

    # -- Adapter contract ----------------------------------------------------

    @property
    def state(self) -> np.ndarray:
        return self._state_ro

    @property
    def theta(self) -> np.ndarray:
        return self._theta

    @property
    def updates(self) -> int:
        return self._updates

    def load_state(self, flat) -> None:
        arr = np.asarray(flat, dtype=self.dtype).reshape(-1)
        if arr.size != self._state.size:
            raise ValueError(
                f"state block must have {self._state.size} elements, got {arr.size}")
        if not np.all(np.isfinite(arr)):
            raise ValueError("refusing a state block that is not finite")
        self._state[:] = arr

    def reset(self) -> None:
        self._state[:] = 0
        self._e[:] = 0
        self._e_post[:] = 0
        self._updates = 0
        self._rejected_nonfinite = 0
        self._rejected_denominator = 0

    def predict(self, x) -> np.ndarray:
        return np.asarray(self._as_x(x) @ self._theta, dtype=self.dtype)

    def _as_x(self, x) -> np.ndarray:
        arr = np.asarray(x, dtype=self.dtype).reshape(-1)
        if arr.size != self.n_features:
            raise ValueError(
                f"x must have {self.n_features} features, got {arr.size}")
        return arr

    def _as_y(self, y) -> np.ndarray:
        arr = np.asarray(y, dtype=self.dtype).reshape(-1)
        if arr.size != self.n_outputs:
            raise ValueError(f"y must have {self.n_outputs} outputs, got {arr.size}")
        return arr

    def _report(self, applied: bool, reason: str | None,
                gain_norm: float) -> UpdateReport:
        return UpdateReport(
            applied=applied,
            residual=self._e.copy(),
            residual_post=self._e_post.copy(),
            gain_norm=gain_norm,
            # Declared absent in `health_fields`, so a Guard refuses to watch these and
            # nothing can read them. Zero because the field has to hold something.
            p_trace=0.0,
            p_min_diag=0.0,
            reason=reason)

    def update(self, x, y) -> UpdateReport:
        """Take one observation and report what happened.

        Same contract as RLS: shape errors raise, bad values are rejected and counted.
        """
        xa = self._as_x(x)
        ya = self._as_y(y)

        if not (np.all(np.isfinite(xa)) and np.all(np.isfinite(ya))):
            self._rejected_nonfinite += 1
            self._e[:] = 0
            self._e_post[:] = 0
            return self._report(False, "nonfinite_input", 0.0)

        np.dot(xa, self._theta, out=self._yhat)
        np.subtract(ya, self._yhat, out=self._e)

        power = self.dtype.type(self.eps) + np.dot(xa, xa)
        if not np.isfinite(power) or power <= 0:
            # Reachable with FINITE input: a regressor of 1e30 squares to infinity in
            # f32. The same vocabulary RLS uses, so a guard's counters mean the same
            # thing whichever adapter is underneath.
            self._rejected_denominator += 1
            self._e_post[:] = self._e
            return self._report(False, "nonpositive_denominator", 0.0)

        g = self.dtype.type(self.mu) / power
        np.outer(xa, self._e, out=self._tn)
        np.multiply(self._tn, g, out=self._tn)
        np.add(self._theta, self._tn, out=self._theta)

        np.dot(xa, self._theta, out=self._yhat)
        np.subtract(ya, self._yhat, out=self._e_post)

        self._updates += 1
        # The gain VECTOR is `g * x`, so its norm is what compares against RLS's
        # ||k||. Reporting `g` alone would make the two adapters' `gain_norm` mean
        # different things, and a `Guard` bound is written against the name.
        return self._report(True, None, float(np.linalg.norm(g * xa)))

    def __repr__(self) -> str:
        return (f"NormalisedLeastMeanSquares(n_features={self.n_features}, "
                f"n_outputs={self.n_outputs}, mu={self.mu}, "
                f"dtype={self.dtype.name!r}, updates={self._updates})")

