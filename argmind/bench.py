# SPDX-License-Identifier: Apache-2.0
"""The benchmark harness. Nothing is called fast until this can say otherwise.

The rule this exists to satisfy: **an optimisation with no way to fail is not an
optimisation, it is a hope.** So the harness is built around being able to report
a regression, and around three requirements set out in DESIGN.md before any of it
was written.

**A fixed workload from the design driver.** An inertial stream at 200 Hz and an
aiding source at 5 Hz, six features, the guard in place — declared rates and
declared dimensions rather than whatever the microbenchmark happened to need.

**Per-sample cost separate from per-frame cost.** They scale with different
things: pushes scale with the sensor rate, frames with the reference rate. An
average over both hides which one moved, and those are the two numbers an
integrator sizes a system with.

**The reference as the correctness oracle.** Every result carries a digest of the
final state block. A change that makes things faster by changing the answers is
not a speed-up, and `compare` reports it as its own category rather than folding
it into a timing delta.

Also measured, because DESIGN.md asserts it repeatedly and nothing checked it:
**memory is a function of configuration, not of uptime.** `memory_growth` does
that with a discriminator rather than a trend line — see its docstring, where the
first attempt is written down because it was wrong in a way that is easy to
repeat.

What is not committed, and why
------------------------------
Absolute timings are not committed as a pass/fail gate. A baseline recorded here
is a statement about this machine, this CPU governor and this Python build; as a
gate in the repository it would fail on everyone else's machine and be switched
off within a week. Record a baseline locally (``--record``) and compare against
it (``--compare``).

What *is* portable, and therefore what the test suite gates on:

- the deterministic counters, which are exact,
- the state digest, which is arithmetic rather than speed,
- the memory leak rate, which is a slope against work done and not a duration,
- ratios measured within a single run — guard overhead against a bare update, or
  how update cost scales with dimension — where the machine cancels out.

Noise is measured rather than assumed. ``--record`` runs the workload several
times and stores the observed spread, so the comparison tolerance is derived from
what this machine actually does rather than from a guess about it.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .adapt import RecursiveLeastSquares
from .align import Aligner
from .guard import Bound, Divergence, Guard
from .pipeline import FrameMap, Pipeline
from .stream import Stream

__all__ = ["Workload", "Timing", "Result", "measure", "compare", "Regression",
           "memory_growth", "scaling", "step_cost", "fit_step", "gate_saving",
           "main"]

_NS = time.perf_counter_ns


@dataclass(frozen=True, slots=True)
class Workload:
    """The fixed workload, from the design driver rather than from convenience."""

    frames: int = 200            # aiding samples, i.e. how many frames exist
    imu_hz: float = 200.0
    aid_hz: float = 5.0
    n_features: int = 6
    dtype: str = "f4"
    guarded: bool = True
    seed: int = 4242

    @property
    def pushes_per_frame(self) -> int:
        return max(1, int(round(self.imu_hz / self.aid_hz)))

    def name(self) -> str:
        return (f"{self.n_features}f-{self.dtype}-"
                f"{'guarded' if self.guarded else 'bare'}-"
                f"{self.imu_hz:g}/{self.aid_hz:g}Hz")


@dataclass(frozen=True, slots=True)
class Timing:
    """Nanoseconds per operation.

    `p99` and `max` are here because of a standard carried from the parked
    project: measure worst case, not just typical. A p50 that looks good and a
    p99 that misses is the shape that fails in the field.
    """

    n: int
    p50: float
    p99: float
    max: float
    mean: float

    @staticmethod
    def of(samples: np.ndarray) -> "Timing":
        if samples.size == 0:
            return Timing(0, 0.0, 0.0, 0.0, 0.0)
        return Timing(int(samples.size),
                      float(np.percentile(samples, 50)),
                      float(np.percentile(samples, 99)),
                      float(samples.max()),
                      float(samples.mean()))


@dataclass(frozen=True, slots=True)
class Result:
    workload: dict
    push_ns: Timing
    frame_ns: Timing
    update_ns: Timing
    counters: dict[str, int]
    state_digest: str
    host: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Result":
        return Result(workload=d["workload"],
                      push_ns=Timing(**d["push_ns"]),
                      frame_ns=Timing(**d["frame_ns"]),
                      update_ns=Timing(**d["update_ns"]),
                      counters=d["counters"],
                      state_digest=d["state_digest"],
                      host=d.get("host", {}))


def _build(w: Workload):
    cap = w.pushes_per_frame + 8
    imu = Stream("imu", dim=w.n_features - 1, capacity=cap, rate_hz=w.imu_hz,
                 dtype=w.dtype)
    aid = Stream("aid", dim=1, capacity=16, rate_hz=w.aid_hz, dtype=w.dtype)
    adapter = RecursiveLeastSquares(n_features=w.n_features, forgetting=0.999,
                                    p0=1e2, dtype=w.dtype)
    if w.guarded:
        adapter = Guard(adapter,
                        [Bound("p_min_diag", lo=0.0),
                         Bound("p_trace", hi=1e9),
                         Divergence(ratio=8.0, floor=1e-3, warmup=20)],
                        rollback_depth=4, checkpoint_every=5)
    pipe = Pipeline(streams=[imu, aid], aligner=Aligner.hold_last("aid"),
                    adapter=adapter,
                    frame_map=FrameMap(features="imu", target="aid",
                                       constant=True))
    return imu, aid, pipe


def measure(w: Workload = Workload(), warmup_frames: int = 20) -> Result:
    """Run the workload once, timing each operation individually.

    Per-operation timing rather than a total divided by a count, because a total
    cannot produce a p99 and the p99 is the number that decides whether a loop
    holds its deadline.
    """
    imu, aid, pipe = _build(w)
    rng = np.random.default_rng(w.seed)
    per_frame = w.pushes_per_frame
    truth = np.arange(1, w.n_features + 1, dtype=np.float64)

    push_ns = np.zeros(w.frames * per_frame, dtype=np.int64)
    frame_ns = np.zeros(w.frames, dtype=np.int64)
    pi = 0
    t = 0.0
    dt_fast = 1.0 / w.imu_hz
    dt_frame = 1.0 / w.aid_hz

    for f in range(w.frames):
        row = None
        for _ in range(per_frame):
            row = rng.standard_normal(w.n_features - 1)
            t += dt_fast
            a = _NS()
            imu.push(t, row)
            push_ns[pi] = _NS() - a
            pi += 1
        y = float(row @ truth[:-1] + truth[-1])
        aid.push(t + 1e-6, [y])

        a = _NS()
        pipe.step()
        frame_ns[f] = _NS() - a

    # The learner on its own, so the plumbing and the arithmetic can move
    # independently in a report instead of being one number.
    bare = RecursiveLeastSquares(n_features=w.n_features, forgetting=0.999,
                                 p0=1e2, dtype=w.dtype)
    xs = rng.standard_normal((w.frames, w.n_features)).astype(w.dtype)
    ys = (xs.astype(np.float64) @ truth).astype(w.dtype)
    update_ns = np.zeros(w.frames, dtype=np.int64)
    for i in range(w.frames):
        a = _NS()
        bare.update(xs[i], ys[i:i + 1])
        update_ns[i] = _NS() - a

    keep = slice(warmup_frames, None)
    digest = hashlib.sha256(np.ascontiguousarray(pipe.adapter.state).tobytes())
    return Result(
        workload=asdict(w),
        push_ns=Timing.of(push_ns[warmup_frames * per_frame:]),
        frame_ns=Timing.of(frame_ns[keep]),
        update_ns=Timing.of(update_ns[keep]),
        counters={"pushes": int(pi), "frames": pipe.stats["frames"],
                  "updated": pipe.stats["updated"],
                  "lost_frames": pipe.stats["lost_frames"],
                  "state_size": int(pipe.adapter.state.size)},
        state_digest=digest.hexdigest()[:32],
        host={"python": sys.version.split()[0], "numpy": np.__version__,
              "machine": platform.machine(), "system": platform.system()})


def memory_growth(w: Workload = Workload(), batch_frames=(64, 256, 1024),
                  repeats: int = 4, build=None) -> dict:
    """Does retained memory grow with how long this has been running?

    DESIGN.md asserts in several places that memory is a function of
    configuration and never of uptime, and nothing verified it. Getting a
    trustworthy answer took two attempts, and the first one was wrong in an
    instructive way.

    **Fitting a slope to a series of batch measurements does not work.** The
    series converges — buffers fill, caches settle — so a straight line through
    it reports growth that is really a decaying transient. Measured: +414
    bytes/batch on a series whose per-batch deltas were 1126, 375, 375, 179, 211
    and then repeatedly exactly 0.

    What works is a **discriminator**. Measure the same batch at several *sizes*
    and ask what the cost per batch is made of. A genuine per-frame leak makes
    cost proportional to frames, so bytes-per-frame stays flat as the batch
    grows. Fixed measurement overhead does not, so bytes-per-frame falls. Fitting
    ``cost = fixed + leak * frames`` separates the two, and ``blocks_per_frame``
    is the slope — the only number here that means "leak".

    Measured with that method on this workload: cost per batch was 84, 92 and 141
    blocks at 16, 66 and 266 frames — cost per *frame* fell tenfold, 5.24 to
    0.53, as the batch grew sixteenfold. That is the signature of fixed overhead
    and not of accumulation.

    The instrument is `sys.getallocatedblocks` rather than `tracemalloc`, which
    was the second correction: tracemalloc allocates while it measures, and its
    own bookkeeping dominated the residual it was being used to detect.

    `build` is injectable so that a test can point this at a deliberately leaky
    adapter. A leak detector that has never detected a leak is not evidence.
    """
    imu, aid, pipe = (build or _build)(w)
    rng = np.random.default_rng(w.seed)
    per_frame = w.pushes_per_frame
    truth = np.arange(1, w.n_features + 1, dtype=np.float64)
    t = 0.0
    dt = 1.0 / w.imu_hz

    def run_batch(n_frames: int) -> None:
        nonlocal t
        for _ in range(n_frames):
            row = None
            for _ in range(per_frame):
                row = rng.standard_normal(w.n_features - 1)
                t += dt
                imu.push(t, row)
            aid.push(t + 1e-6, [float(row @ truth[:-1] + truth[-1])])
            pipe.step()

    rows = []
    for n in batch_frames:
        run_batch(n)                       # settle at this size before measuring
        gc.collect()
        deltas = []
        for _ in range(repeats):
            before = sys.getallocatedblocks()
            run_batch(n)
            gc.collect()
            deltas.append(sys.getallocatedblocks() - before)
        rows.append({"frames": n, "blocks_per_batch": float(np.mean(deltas)),
                     "blocks_per_frame": float(np.mean(deltas)) / n,
                     "samples": deltas})

    leak, fixed = fit_leak(rows)
    return {"rows": rows, "blocks_per_frame": leak,
            "fixed_blocks_per_batch": fixed}


def fit_leak(rows) -> tuple[float, float]:
    """Split measured cost into ``fixed + leak * frames``; return (leak, fixed).

    A separate function because it is the whole idea and it is the only part that
    can be checked without a stopwatch. Cost per frame is *not* the leak rate: at
    a small batch, fixed overhead divided by a few frames looks exactly like
    accumulation. Only the slope distinguishes them.

    Worth recording that the two agree on this pipeline as it stands, because its
    fixed overhead per batch is now near zero — so today the distinction is
    invisible in a live measurement and can only be demonstrated on known inputs.
    It was not always so: before `Adapter.state` stopped building a fresh view per
    call, fixed overhead was 80-odd blocks a batch and cost per frame reported a
    leak that was not there.
    """
    xs = np.array([r["frames"] for r in rows], dtype=np.float64)
    ys = np.array([r["blocks_per_batch"] for r in rows], dtype=np.float64)
    leak, fixed = np.polyfit(xs, ys, 1)
    return float(leak), float(fixed)


def scaling(dims=(4, 8, 16, 32, 64, 128), updates: int = 2000,
            dtype: str = "f8", warmup: int = 200) -> dict[int, float]:
    """Cost of one `update` against the state dimension.

    A ratio measured inside one run, so the machine cancels out and the result is
    portable in a way an absolute timing is not.

    Measured here, and worth knowing before optimising anything: cost is **flat**
    from n=4 to n=32 — roughly 22 µs either way — and only starts climbing at
    n=64. RLS is O(n²), so a flat region means the arithmetic is not the cost at
    all; the Python and numpy call overhead is. At the design driver's dimension
    of six, essentially all of the time is overhead.

    That is direct measured support for the language decision in DESIGN.md, which
    argued the same thing from the parked project's numbers rather than from this
    codebase. It also sets an expectation for the port: the C++ core should not be
    judged against these figures, because it is not paying the cost these figures
    are made of.
    """
    out: dict[int, float] = {}
    for n in dims:
        a = RecursiveLeastSquares(n_features=n, forgetting=0.999, dtype=dtype)
        rng = np.random.default_rng(17)
        xs = rng.standard_normal((updates, n)).astype(dtype)
        ys = rng.standard_normal((updates, 1)).astype(dtype)
        for i in range(min(warmup, updates)):
            a.update(xs[i], ys[i])
        start = _NS()
        for i in range(updates):
            a.update(xs[i], ys[i])
        out[n] = (_NS() - start) / updates
    return out



def update_flops(kind: str, n_features: int, n_outputs: int = 1,
                 hidden: tuple[int, ...] = ()) -> int:
    """Multiply-adds in one `update`, **derived from the shapes rather than timed.**

    This is the transferable number. Host microseconds are interpreter and numpy dispatch
    cost — `scaling` already shows RLS flat from n=4 to n=32 for that reason — so they say
    nothing about the C++ tier or about a target. A flop count is architectural: it is the
    same on every machine, and it is what sizes a duty cycle.

    RLS is ``4n² + 4nm + 2n``: ``P@x``, ``x@P``, the rank-one update and ``P /= lambda`` are
    each ``n²``. NLMS is ``4nm + 2n`` and has no ``n²`` term at all, which is the whole
    reason it exists. A network is ``4·W`` over its weight count ``W``, because the forward
    pass runs **twice** — once for the prediction and once for the a posteriori residual —
    plus the backward pass and the gradient.

    So at equal *state* a network costs about twice an RLS, since RLS's ``4n²`` sits against
    a state of ``n² + nm``, while a network's ``4W`` sits against a state of ``W``. Measured
    on the host and it holds: 34.6 µs against 60.3 at ~4,200 elements, 206.8 against 460.3
    at ~66,000, both against a run-to-run spread of 1.07–1.18×.
    """
    n, m = int(n_features), int(n_outputs)
    if kind == "rls":
        return 4 * n * n + 4 * n * m + 2 * n
    if kind == "nlms":
        return 4 * n * m + 2 * n
    if kind == "mlp":
        widths = (n, *hidden, m)
        w = sum(a * b for a, b in zip(widths[:-1], widths[1:]))
        return 8 * w      # forward twice, backward, gradient -- 2*(2w) + 2w + 2w
    raise ValueError(f"unknown adapter kind {kind!r}")


def adapter_scaling(updates: int = 3000, dtype: str = "f8",
                    warmup: int = 300) -> list[dict]:
    """Update cost against size for all three adapters, beside the derived flop count.

    **The point is the ratio between them, not either number.** `scaling` already records
    that RLS is flat from n=4 to n=32 because the cost there is call overhead; this makes
    that quantitative and extends it to a network, whose update is several small numpy
    calls per layer rather than one large one.

    Measured, `ns/flop` across the range: **66 down to 0.8, a factor of 209.** A constant
    would mean arithmetic-bound. It is not — small adapters are dispatch-bound and only the
    largest approach the arithmetic, where RLS and a network converge on the same 0.78–0.86
    ns/flop and the network simply does twice the flops.

    That factor of 209 is the evidence for the rule DESIGN.md already states: **these
    timings are not a baseline the C++ core can be compared against**, because most of what
    they measure is not present there. It is also why nothing here is a regression gate —
    `argmind.bench.compare` gates exact counters, the state digest, the leak rate and
    ratios inside one run, and an absolute microsecond figure is none of those.

    One number was measured and is **not** reported, because it did not survive its own
    error bars: the equal-*accuracy* comparison. MLP (32,32) at 62.6 µs against RFF-128
    through RLS at 78.9 reads as 1.26× — inside run-to-run spreads of 1.62× and 1.33×. The
    memory side of that comparison is 9.5 KB against 131.0 KB and is not close.
    """
    from .adapt import MultilayerPerceptron, NormalisedLeastMeanSquares  # noqa: PLC0415

    cases: list[tuple[str, str, object, int]] = []
    for n in (16, 64, 128, 256):
        cases.append((f"rls n={n}", "rls",
                      lambda n=n: RecursiveLeastSquares(n_features=n, forgetting=0.999,
                                                        dtype=dtype), n))
    for n in (64, 256, 1024):
        cases.append((f"nlms n={n}", "nlms",
                      lambda n=n: NormalisedLeastMeanSquares(n, 1, dtype=dtype), n))
    for hid in ((16,), (32, 32), (64, 64), (128, 128), (256, 256)):
        cases.append((f"mlp {hid}", "mlp",
                      lambda h=hid: MultilayerPerceptron(3, 1, hidden=h, dtype=dtype), 3))

    rows = []
    for label, kind, make, n_in in cases:
        a = make()
        rng = np.random.default_rng(17)
        xs = rng.standard_normal((updates, n_in)).astype(dtype)
        ys = rng.standard_normal((updates, 1)).astype(dtype)
        for i in range(min(warmup, updates)):
            a.update(xs[i], ys[i])
        start = _NS()
        for i in range(updates):
            a.update(xs[i], ys[i])
        ns = (_NS() - start) / updates
        flops = update_flops(kind, a.n_features, a.n_outputs,
                             getattr(a, "hidden", ()))
        rows.append({"label": label, "kind": kind, "ns_per_update": ns,
                     "flops": flops, "state": int(np.asarray(a.state).size),
                     "ns_per_flop": ns / flops})
    return rows


def step_underflow(adapter, xs, ys, probes: int = 20, warmup: int = 0) -> dict:
    """How often an update is too small to change the weight it is applied to.

    Per element of the state block: after one `update`, was the movement smaller than the
    spacing at that element's own magnitude? A weight whose step is below its own
    representable resolution **does not move**, and the model stops learning in that
    direction without ever producing a NaN or a zero — which is why a check for either
    reports success while nothing improves.

    **This is a bring-up diagnostic, not a runtime mechanism**, and the distinction is
    deliberate. The condition it finds is a *configuration* error — a target that is not
    scaled — present from the first update and unchanging, so it is not the developing
    fault a `Guard` exists to catch. Run it once when wiring a new sensor up, read the
    number, scale the target, ship.

    **It counts weights that did not move at all, and that is exact rather than a
    threshold.** Under round-to-nearest a step smaller than half the spacing leaves the
    weight bit-identical, so "stuck" is `movement == 0` — no `spacing()` call and no
    tolerance. Comparing against the spacing instead was the first version; it can
    over-count, because a downward step across a binade boundary is half a spacing and is a
    real move. Measured, that over-count is **0.00 %** in every case tried, so the two agree
    in practice and the exact one is kept for being exact and cheaper.

    **The interpretation is only valid where a zero step means a small step**, and with
    `relu` it does not. A unit that is off has *no* gradient, so its weights do not move for
    a reason that has nothing to do with scaling — and the number inverts: on the same
    target, `relu` reads **0.0 % raw and 57.1 % scaled** where `tanh` reads 64.4 % and 0.0 %.
    Use it on `tanh`, whose derivative `1 - a²` is zero only at `|a| = 1` exactly. On `relu`
    it reports dead units, which is worth knowing but is a different measurement.

    Interpreting it, measured on `tanh`: **0.0 % is healthy and anything above about 10 %
    is not.**
    There is no grey band to tune. On `50·sin(0.5s₀) + 20·s₁s₂` through a network, over six
    seeds and two architectures, the raw target gives 37–97 % and the same target divided by
    its own standard deviation gives **exactly 0.0 % in every single run**. That separation
    holds at **five probes**; more do not sharpen it.

    **The failure is gradient descent's, not the library's**, and that is worth knowing
    before reaching for this. Measured on the same raw target: RLS reports 0.0 % underflow
    and *identical* accuracy raw or scaled (0.6547 both), and so does NLMS (0.7691 both) —
    a least-squares solution is scale-invariant, so scaling `y` scales `theta` and nothing
    else. Only the network cares: 63.6 % against 0.0 %, and 0.7156 against 0.1534. The
    function is written against the `Adapter` contract rather than against one adapter
    because those zeros are the control that says it is not just reporting noise.

    A cheaper proxy was measured and **does not work**. `gain_norm / (eps * ‖state‖)` uses
    two quantities the adapter already has, and correlates with the true rate at only +0.70
    on a log scale, with the orderings overlapping: 1.24e13 for a 74 %-underflowing run
    against 2.98e11 for a 0 % one. The reason is structural — a norm is dominated by the
    largest weights and the largest gradients, which are the ones still moving, while the
    rate is about the stuck minority. They measure different things.

    `xs` and `ys` are arrays of samples; `warmup` of them are consumed before counting,
    because a freshly constructed adapter moves freely and would read 0 % whatever the
    scaling.

    **There is no `--underflow` flag** and the omission is deliberate. The number only means
    something against the caller's own target, and the only adapter this module's `Workload`
    builds is an RLS, which is scale-invariant and would print 0.0 % forever. A command-line
    entry point that cannot say anything is the shape of the two this repository has already
    shipped with no caller.
    """
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    if len(xs) < warmup + probes:
        raise ValueError(
            f"need at least warmup + probes = {warmup + probes} samples, got {len(xs)}")

    for i in range(warmup):
        adapter.update(xs[i], ys[i])

    hits = seen = 0
    applied = 0
    for i in range(warmup, warmup + probes):
        before = np.array(adapter.state)
        report = adapter.update(xs[i], ys[i])
        after = np.asarray(adapter.state)
        if not report.applied:
            continue                       # a rejected sample moves nothing by design
        applied += 1
        moved = np.abs(after - before)
        hits += int(np.sum(moved == 0.0))
        seen += moved.size

    return {"rate": hits / seen if seen else float("nan"),
            "stuck": hits, "elements": seen, "applied": applied}


def step_cost(frames_per_step=(1, 2, 4, 8, 16), steps: int = 300,
              n_features: int = 6, dtype: str = "f4", guarded: bool = True,
              fresh_target: bool = False, warmup: int = 30) -> dict:
    """Cost of one `step()` split into ``fixed + per_frame * frames``.

    This is the latency question in the form that can actually be answered. The
    low-latency wiring puts the **fastest** stream on the reference and calls
    `step()` after every one of its samples, so each call drains a single frame
    and pays the call's fixed overhead in full. Calling less often amortises that
    overhead across more frames and costs latency instead. Which of the two
    dominates is a property of this implementation, not a matter of taste, and it
    is not visible in a single per-step figure.

    So the cost is swept against frames drained and the line is fitted. The
    technique is `fit_leak`'s and is reused deliberately: a cost per frame at one
    batch size cannot tell fixed overhead from real per-frame work, and only the
    slope separates them. That distinction was already worth a correction once in
    this file.

    Returns absolute nanosecond figures, which are **machine-specific and are not a
    gate**. Neither is ``fixed_fraction_at_one``, and an earlier version of this docstring
    said that ratio was the part which travels. Measured over ten interleaved runs on one
    idle machine, the slope moved 10.9% about its mean and the overhead fraction ranged from
    19.9% to 37.2%. Three consecutive runs agreed to 2.6% -- which is how a drifting mean
    impersonates a stable one, and is why DESIGN.md ended up quoting a 0.5% stability that
    does not exist.

    So what this is good for is one comparison inside one run -- `gate_saving` does exactly
    that -- and for noticing a change large enough to clear that spread. A figure taken from
    it should carry the spread beside it or not be quoted.
    """
    rows = []
    for k in frames_per_step:
        fast = Stream("fast", dim=n_features - 1, capacity=max(16, 4 * k),
                      dtype=dtype)
        slow = Stream("slow", dim=1, capacity=16, dtype=dtype)
        adapter = RecursiveLeastSquares(n_features=n_features, forgetting=0.999,
                                        p0=1e2, dtype=dtype)
        if guarded:
            adapter = Guard(adapter,
                            [Bound("p_min_diag", lo=0.0), Bound("p_trace", hi=1e9)],
                            rollback_depth=4, checkpoint_every=5)
        pipe = Pipeline(streams=[fast, slow], aligner=Aligner.hold_last("fast"),
                        adapter=adapter,
                        frame_map=FrameMap(features="fast", target="slow",
                                           constant=True),
                        require_fresh_target=fresh_target)
        rng = np.random.default_rng(9001)
        t = 0.0
        # The target arrives once per step, so the freshness gate lets exactly one
        # frame per step adapt however many are drained. That is the configuration
        # the option exists for, and it is what makes the two sweeps comparable.
        samples = np.zeros(steps, dtype=np.int64)
        for i in range(steps + warmup):
            for _ in range(k):
                t += 1e-3
                fast.push(t, rng.standard_normal(n_features - 1))
            slow.push(t, [float(rng.standard_normal())])
            t += 1e-6
            a = _NS()
            pipe.step()
            d = _NS() - a
            if i >= warmup:
                samples[i - warmup] = d
        rows.append({"frames": k, "ns_per_step": float(np.percentile(samples, 50)),
                     "updated": int(pipe.stats["updated"])})

    per_frame, fixed = fit_step(rows)
    one = fixed + per_frame if np.isfinite(fixed) else float("nan")
    measured_one = next((r["ns_per_step"] for r in rows if r["frames"] == 1), None)
    return {"rows": rows, "ns_per_frame": per_frame, "fixed_ns_per_step": fixed,
            "fixed_fraction_at_one": (fixed / one if np.isfinite(one) and one > 0
                                      else float("nan")),
            "measured_one_frame_ns": measured_one,
            "guarded": guarded, "fresh_target": fresh_target}


def fit_step(rows) -> tuple[float, float]:
    """Split measured step cost into ``fixed + per_frame * frames``; return
    (per_frame, fixed).

    Separate for the same reason `fit_leak` is: it is the whole idea and it is the
    only part that can be checked without a stopwatch. Note which of the two
    figures is trustworthy. The slope is measured across the sweep and is well
    determined; **the intercept is an extrapolation to zero frames** from a line
    whose slope dominates, so it carries the fit's error rather than a
    measurement's. Anything that turns on the intercept should use the directly
    measured one-frame figure instead, which is why `step_cost` returns that too.
    """
    xs = np.array([r["frames"] for r in rows], dtype=np.float64)
    ys = np.array([r["ns_per_step"] for r in rows], dtype=np.float64)
    # A line through one point is not determined, and numpy will happily return one
    # anyway with a RankWarning nobody reads. A fabricated slope in a benchmark is
    # worse than no slope, so this says so instead.
    if np.unique(xs).size < 2:
        return float("nan"), float("nan")
    per_frame, fixed = np.polyfit(xs, ys, 1)
    return float(per_frame), float(fixed)


def gate_saving(steps: int = 300, **kw) -> dict:
    """What `require_fresh_target` costs, and what it saves, in one run.

    Both figures matter and they point in opposite directions. The gate adds a
    comparison per frame; it removes an adapter update from every frame whose
    target has not moved. In the wiring it exists for — a fast reference and a
    slow target — the second dominates, and a caller deciding whether to turn it
    on should see the number rather than the argument.

    A ratio measured inside one run, so the machine cancels out.
    """
    off = step_cost(steps=steps, fresh_target=False, **kw)
    on = step_cost(steps=steps, fresh_target=True, **kw)
    one_off = off["measured_one_frame_ns"]
    one_on = on["measured_one_frame_ns"]
    # The two ratios answer different questions and an early version of this
    # conflated them. This sweep pushes ONE target per step, so at one frame per
    # step every frame carries a new target and the gate can never fire: that
    # ratio is what the option costs where it cannot help, and it should sit at
    # 1.0 within the harness's run-to-run spread. The slope is where it can help,
    # because the extra frames in a multi-frame step all share one target.
    #
    # The intercepts are deliberately not compared. They are the fit's least
    # determined parameter -- an extrapolation to zero frames from a line whose
    # slope dominates -- and an earlier version reported their ratio as a finding
    # when it was mostly the conditioning of a five-point fit.
    return {"off": off, "on": on,
            # What a frame costs when the gate skips its update: the saving.
            "per_frame_ratio": (on["ns_per_frame"] / off["ns_per_frame"]
                                if off["ns_per_frame"] > 0 else float("nan")),
            # What the option costs where it cannot help: the overhead.
            "one_frame_ratio": (one_on / one_off
                                if one_off and one_off > 0 else float("nan")),
            "adaptations_off": off["rows"][-1]["updated"],
            "adaptations_on": on["rows"][-1]["updated"]}

def _spread(noise: dict, key: str) -> float:
    """This metric's measured spread, floored at the machine's typical one.

    A metric whose handful of runs happened to agree records a spread of exactly
    zero, and taking that at face value gives it the *tightest* gate of any
    metric — which is precisely backwards, since agreeing twice is not evidence
    of being noise-free, it is evidence of a small sample. Measured on the
    machine this was written on, one metric in six records exactly zero while the
    others range from 38% to 566%.

    Noise is mostly a property of the machine rather than of the individual
    metric, so a metric with no observed spread inherits the median of the
    others. That is the correction that stopped the round-trip test flaking.
    """
    own = float(noise.get(key, 0.0))
    others = [float(v) for v in noise.values()]
    typical = float(np.median(others)) if others else 0.0
    return max(own, typical)


@dataclass(frozen=True, slots=True)
class Regression:
    kind: str            # "correctness" | "timing" | "counter"
    what: str
    baseline: float | str
    current: float | str
    detail: str = ""

    def __str__(self) -> str:
        return f"[{self.kind}] {self.what}: {self.baseline} -> {self.current} {self.detail}"


def compare(baseline: Result, current: Result, tolerance: float = 1.25,
            noise: dict | None = None) -> list[Regression]:
    """Everything that got worse, most serious first.

    A changed state digest is reported first and as its own kind. It is not a
    slow-down and must not be read as one: the run computed different numbers, so
    every timing beside it is a comparison between two different things.

    `tolerance` is a ratio. `noise`, if given, is the per-metric run-to-run spread
    recorded by `--record`, and raises the threshold for a metric that this
    machine is simply noisy about — measured rather than assumed, so a quiet CPU
    gets a tight gate and a busy one does not produce false alarms.

    Findings come out most serious first. That is the construction order below
    rather than a sort afterwards: an explicit re-sort read as a guarantee while
    being a no-op, which is the kind of line that survives every mutation test
    because it does nothing.
    """
    out: list[Regression] = []
    if baseline.state_digest != current.state_digest:
        out.append(Regression(
            "correctness", "state_digest", baseline.state_digest,
            current.state_digest,
            "the run produced different numbers; the timings below compare two "
            "different computations"))

    for key in sorted(set(baseline.counters) | set(current.counters)):
        b, c = baseline.counters.get(key), current.counters.get(key)
        if b != c:
            out.append(Regression("counter", key, b, c,
                                  "the workload is no longer the same workload"))

    for metric in ("push_ns", "frame_ns", "update_ns"):
        b_t: Timing = getattr(baseline, metric)
        c_t: Timing = getattr(current, metric)
        for stat in ("p50", "p99"):
            b, c = getattr(b_t, stat), getattr(c_t, stat)
            if b <= 0:
                continue
            limit = tolerance
            if noise:
                limit = max(limit, 1.0 + 2.0 * _spread(noise, f"{metric}.{stat}"))
            if c > b * limit:
                out.append(Regression(
                    "timing", f"{metric}.{stat}", round(b, 1), round(c, 1),
                    f"x{c / b:.2f}, over the x{limit:.2f} threshold"))
    return out


def measure_noise(w: Workload, repeats: int = 5) -> dict:
    """Run the same workload repeatedly and report each metric's relative spread.

    This is what makes the regression threshold defensible instead of invented:
    a machine that varies 30% between identical runs cannot support a 25% gate,
    and pretending otherwise produces an alarm nobody believes.
    """
    runs = [measure(w) for _ in range(repeats)]
    spread: dict[str, float] = {}
    for metric in ("push_ns", "frame_ns", "update_ns"):
        for stat in ("p50", "p99"):
            vals = np.array([getattr(getattr(r, metric), stat) for r in runs])
            spread[f"{metric}.{stat}"] = float(
                (vals.max() - vals.min()) / vals.min()) if vals.min() > 0 else 0.0
    return spread


def _report(r: Result) -> str:
    w = r.workload
    lines = [f"workload: {Workload(**w).name()}  frames={w['frames']}",
             f"host: python {r.host.get('python')} numpy {r.host.get('numpy')} "
             f"{r.host.get('machine')}",
             "",
             f"{'metric':<14}{'p50':>12}{'p99':>12}{'max':>12}   unit",
             "-" * 64]
    for label, t, unit in (("per sample", r.push_ns, "ns/push"),
                           ("per frame", r.frame_ns, "ns/frame"),
                           ("adapter only", r.update_ns, "ns/update")):
        lines.append(f"{label:<14}{t.p50:>12.0f}{t.p99:>12.0f}{t.max:>12.0f}   {unit}")
    lines += ["", f"counters: {r.counters}", f"state digest: {r.state_digest}"]
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m argmind.bench",
        description="Fixed-workload benchmark. Records and compares; absolute "
                    "baselines are machine-specific and are not committed.")
    p.add_argument("--record", metavar="PATH",
                   help="measure, and write a baseline (with measured noise)")
    p.add_argument("--compare", metavar="PATH",
                   help="measure, and report regressions against a baseline")
    # From an instance, not the class: with slots=True the class attribute is the
    # slot descriptor rather than the default value.
    default = Workload()
    p.add_argument("--frames", type=int, default=default.frames)
    p.add_argument("--dtype", default=default.dtype, choices=["f4", "f8"])
    p.add_argument("--bare", action="store_true", help="run without the guard")
    p.add_argument("--repeats", type=int, default=5,
                   help="runs used to measure this machine's noise on --record")
    p.add_argument("--tolerance", type=float, default=1.25)
    p.add_argument("--memory", action="store_true",
                   help="also report the memory leak rate")
    p.add_argument("--scaling", action="store_true",
                   help="also report update cost against state dimension")
    p.add_argument("--adapters", action="store_true",
                   help="also report update cost against size for all three adapters, "
                        "beside the derived flop count")
    p.add_argument("--latency", action="store_true",
                   help="also report step() cost against frames drained, and what "
                        "require_fresh_target costs")
    args = p.parse_args(argv)

    w = Workload(frames=args.frames, dtype=args.dtype, guarded=not args.bare)
    result = measure(w)
    print(_report(result))

    if args.memory:
        mg = memory_growth(w)
        print("\nmemory (live CPython blocks; a leak is the slope of cost "
              "against frames, not the cost itself)")
        for row in mg["rows"]:
            print(f"        {row['frames']:>6} frames/batch  "
                  f"{row['blocks_per_batch']:>8.1f} blocks/batch  "
                  f"{row['blocks_per_frame']:>7.3f} blocks/frame")
        print(f"        leak {mg['blocks_per_frame']:+.4f} blocks/frame, "
              f"fixed {mg['fixed_blocks_per_batch']:+.1f} blocks/batch")

    if args.adapters:
        rows = adapter_scaling()
        print("\nupdate cost by adapter, against DERIVED flops. ns/flop spanning two "
              "orders of\n        magnitude is what says these timings are dispatch "
              "rather than arithmetic --\n        and therefore not a baseline the C++ "
              "core can be compared against.")
        print(f"        {'adapter':>16}{'us/upd':>9}{'kflop':>9}{'state':>8}{'ns/flop':>10}")
        for row in rows:
            print(f"        {row['label']:>16}{row['ns_per_update'] / 1000:>9.1f}"
                  f"{row['flops'] / 1000:>9.1f}{row['state']:>8}"
                  f"{row['ns_per_flop']:>10.2f}")
        lo = min(r["ns_per_flop"] for r in rows)
        hi = max(r["ns_per_flop"] for r in rows)
        print(f"        ns/flop spans {lo:.2f} to {hi:.2f}, a factor of {hi / lo:.0f}")

    if args.latency:
        sc = step_cost()
        print("\nstep() cost against frames drained (the low-latency wiring: the "
              "fastest\n        stream on the reference, so a step drains one frame "
              "and pays the\n        fixed overhead in full)")
        for row in sc["rows"]:
            print(f"        {row['frames']:>3} frames/step  "
                  f"{row['ns_per_step']:>9.0f} ns/step  "
                  f"{row['ns_per_step'] / row['frames']:>9.0f} ns/frame")
        print(f"        fit: {sc['fixed_ns_per_step']:.0f} ns fixed + "
              f"{sc['ns_per_frame']:.0f} ns/frame  "
              f"({sc['fixed_fraction_at_one']:.0%} of a one-frame step is overhead)")

        gs = gate_saving()
        print("\nrequire_fresh_target, measured in the wiring it exists for")
        print(f"        saving where it can help (per frame)  "
              f"x{gs['per_frame_ratio']:.2f}")
        print(f"        cost where it cannot (one frame, one target)  "
              f"x{gs['one_frame_ratio']:.2f}")
        print(f"        adaptations: {gs['adaptations_off']} off, "
              f"{gs['adaptations_on']} on (over the 16-frame sweep)")

    if args.scaling:
        print("\nupdate cost vs dimension (RLS is O(n^2); a flat region means "
              "the arithmetic is not the cost)")
        prev = None
        for n, ns in scaling().items():
            note = "" if prev is None else f"  x{ns / prev:.2f}"
            print(f"        n={n:>4}  {ns:>9.0f} ns/update{note}")
            prev = ns

    if args.record:
        noise = measure_noise(w, repeats=args.repeats)
        Path(args.record).write_text(json.dumps(
            {"result": result.to_dict(), "noise": noise}, indent=2), "utf-8")
        print(f"\nrecorded to {args.record}")
        print("noise (relative spread over "
              f"{args.repeats} runs): "
              + ", ".join(f"{k} {v:.1%}" for k, v in sorted(noise.items())))

    if args.compare:
        blob = json.loads(Path(args.compare).read_text("utf-8"))
        regressions = compare(Result.from_dict(blob["result"]), result,
                              tolerance=args.tolerance, noise=blob.get("noise"))
        if not regressions:
            print("\nno regressions")
            return 0
        print("\nREGRESSIONS")
        for r in regressions:
            print(f"  {r}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
