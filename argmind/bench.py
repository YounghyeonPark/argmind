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
           "memory_growth", "scaling", "main"]

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
