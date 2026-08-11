# SPDX-License-Identifier: Apache-2.0
"""The C++ `Pipeline` against the Python reference.

What has to agree here is **bookkeeping under interleaving**, not arithmetic. `xcheck`
already settled that RLS and the guard compute the same numbers, and `scheck` that the
aligner picks the same samples. What is new at this level is that a pipeline is driven by two
independent clocks — samples arriving, and `step()` being called — and every quantity it
reports is a difference between counters taken at different moments. `lost_frames` is
`arrived - considered` where both terms move; the skip counters partition frames that existed
into reasons they did not adapt; the drain watermark decides which reference instants are
still owed a frame.

So the job file is an **event log**. `push` and `step` interleave in whatever order a case
wants, which is the only way to reach the states that matter: a ring that overflowed between
two steps, a step with nothing new, a guard that froze half-way through a step.

Everything except the model state is compared **exactly** — the counters are integers and the
staleness figures are compared by bit pattern, so a signed zero shows. The state block is the
one place a tolerance is right, because it is RLS arithmetic and `cpp/include/argmind/rls.hpp`
lists three deliberate divergences that produce it.

One limit, stated because "parity holds" should not be read as covering it: the C++ pipeline
takes `StreamView<T>*`, so every stream in one pipeline shares a scalar type. The Python
allows a mixture. These cases only ever use matched dtypes, because the C++ side cannot
express the other thing.
"""

import math
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import _toolbuild

from argmind.adapt import RecursiveLeastSquares
from argmind.align import Aligner
from argmind.guard import Bound, Divergence, Guard
from argmind.pipeline import FrameMap, Pipeline
from argmind.stream import Stream

REPO = Path(__file__).resolve().parent.parent
CPP = REPO / "cpp"

# Sizes are asserted against the tool's own static_asserts. Python's `struct` with `<` packs
# without padding and a C++ struct does not; the explicit `4x` before the first double in the
# job header is that difference, and getting it wrong has already cost this project two
# debugging sessions on other tools.
_JOB = struct.Struct("<4sHBBHHHHHHHBBBBBBIIIIIdddd")  # 80 bytes, then bounds + divergence
_BOUND = struct.Struct("<ddII")                       # 24
_DIV = struct.Struct("<ddddII")                       # 40
_SPEC = struct.Struct("<HHHH")                        # 8
_MAP = struct.Struct("<HHI")                          # 8
_EVENT = struct.Struct("<BBB5xd")                     # 16
_STEP = struct.Struct("<4BIIII8Ii8x2d3d")             # 104
_TAIL = struct.Struct("<7QII")                        # 64
assert _JOB.size + 4 * _BOUND.size + _DIV.size == 216, "cpp/tools/pipecheck.cpp: Job"
assert (_BOUND.size, _DIV.size, _SPEC.size, _MAP.size) == (24, 40, 8, 8)
assert (_EVENT.size, _STEP.size, _TAIL.size) == (16, 104, 64)

POLICIES = {"hold_last": 0, "nearest": 1, "interpolate": 2, "drop_if_stale": 3}
FIELDS = {"residual_norm": 0, "residual_post_norm": 1, "gain_norm": 2,
          "p_trace": 3, "p_min_diag": 4}

#: The C++ `SkipReason` enum, in order. The Python reports strings; `guard:<check>` collapses
#: onto one slot here because the C++ carries the failing checks in the `frozen_by` bitmask
#: instead — which names *more* of them than the string does, since the Python only ever
#: reports the first.
SKIP = ["stale", "invalid", "frozen", "guard", "nonfinite_input",
        "nonpositive_denominator", "rejected", "stale_target"]


def _bits(x: float) -> int:
    """A float's bit pattern, so -0.0 and 0.0 compare unequal and a NaN payload shows."""
    return struct.unpack("<Q", struct.pack("<d", float(x)))[0]


def _compiler() -> str | None:
    for name in ("g++", "clang++"):
        if shutil.which(name):
            return name
    return None


@pytest.fixture(scope="session", params=_toolbuild.ARCHES)
def pipecheck(request, tmp_path_factory) -> Path:
    """The parity tool, built for the host and for the declared target.

    Both, on every case: a review measured a QEMU invocation at 157 ms and showed
    that the hand-picked subset which used to stand in for the target passed with
    the host binaries substituted, and passed again with the target FPU set to
    round-toward-zero. See tests/_toolbuild.py.
    """
    work = tmp_path_factory.mktemp("cpp_pl_" + request.param)
    return _toolbuild.build_tool("pipecheck", request.param, work)


class Case:
    """One pipeline configuration plus the event log to drive it with.

    A plain holder rather than a dataclass so that the two runners below read the same
    fields with no adaptation layer between them — an adaptation layer is where a harness
    starts comparing itself.
    """

    def __init__(self, dims, capacity, features, target, *, constant=False,
                 n_features=None, n_outputs=None, policy="hold_last", reference=0,
                 dtype="f8",
                 max_stale_pipeline=None, max_stale_aligner=0.0, require_valid=True,
                 allow_non_causal=False, lam=0.98, p0=10.0, guard=None,
                 require_fresh_target=False):
        self.dims = list(dims)
        self.capacity = capacity
        self.features = list(features)      # [(stream, component)]
        # One `(stream, component)` or a list of them. A list is how the freshness gate gets
        # exercised with more than one output at all: the reference compares a tuple over
        # unique target streams and the core iterates outputs, and an audit could check that
        # equivalence by reading but not by running.
        self.targets = list(target) if isinstance(target, list) else [target]
        self.target = self.targets[0]
        self.n_outputs = n_outputs or len(self.targets)
        self.constant = constant
        self.n_features = n_features or (len(features) + (1 if constant else 0))
        self.policy = policy
        self.reference = reference
        self.dtype = dtype
        self.max_stale_pipeline = max_stale_pipeline
        self.max_stale_aligner = max_stale_aligner
        self.require_valid = require_valid
        self.allow_non_causal = allow_non_causal
        self.require_fresh_target = require_fresh_target
        self.lam = lam
        self.p0 = p0
        self.guard = guard                  # None, or a dict
        self.events = []                    # ("push", s, t, values, valid) | ("step",)

    def push(self, s, t, values, valid=True):
        self.events.append(("push", s, float(t), list(values), bool(valid)))
        return self

    def step(self):
        self.events.append(("step",))
        return self


def run_cpp(pipecheck: Path, tmp_path: Path, case: Case):
    dt = np.dtype(case.dtype)
    g = case.guard or {}
    bounds = g.get("bounds", [])
    div = g.get("divergence")
    job = tmp_path / "job.bin"
    out = tmp_path / "out.bin"
    with open(job, "wb") as fh:
        fh.write(_JOB.pack(
            b"AMPL", 1, dt.itemsize, 1 if case.guard else 0, case.n_features,
            case.n_outputs, len(case.dims), case.capacity, POLICIES[case.policy],
            case.reference, len(case.features), 1 if case.constant else 0,
            1 if case.require_valid else 0,
            0 if case.max_stale_pipeline is None else 1,
            1 if case.allow_non_causal else 0,
            1 if case.require_fresh_target else 0, 0,
            len(case.events), g.get("depth", 1), g.get("checkpoint_every", 1),
            g.get("recover_after", 0), len(bounds),
            case.max_stale_pipeline or 0.0, case.max_stale_aligner,
            case.lam, case.p0))
        for i in range(4):
            if i < len(bounds):
                field, lo, hi = bounds[i]
                flags = (1 if lo is not None else 0) | (2 if hi is not None else 0)
                fh.write(_BOUND.pack(lo or 0.0, hi or 0.0, FIELDS[field], flags))
            else:
                fh.write(_BOUND.pack(0.0, 0.0, 0, 0))
        if div:
            fh.write(_DIV.pack(div["ratio"], div["fast_alpha"], div["slow_alpha"],
                               div["floor"], div["warmup"], 1))
        else:
            fh.write(_DIV.pack(5.0, 0.25, 0.02, 0.0, 20, 0))
        for d in case.dims:
            fh.write(_SPEC.pack(d, 0, 0, 0))
        for s, c in case.features:
            fh.write(_MAP.pack(s, c, 0))
        for t in case.targets:
            fh.write(_MAP.pack(t[0], t[1], 0))
        for ev in case.events:
            if ev[0] == "push":
                _, s, t, values, valid = ev
                fh.write(_EVENT.pack(0, s, 1 if valid else 0, t))
                fh.write(np.ascontiguousarray(values, dtype=dt).tobytes())
            else:
                fh.write(_EVENT.pack(1, 0, 0, 0.0))

    done = subprocess.run([str(pipecheck), str(job), str(out)], capture_output=True,
                          text=True)
    assert done.returncode == 0, f"pipecheck failed: {done.stdout}\n{done.stderr}"

    raw = out.read_bytes()
    off = 0
    accepted, steps = [], []
    n_streams = len(case.dims)
    for ev in case.events:
        if ev[0] == "push":
            accepted.append(raw[off])
            off += 1
            continue
        row = _STEP.unpack_from(raw, off)
        off += _STEP.size
        stale = list(struct.unpack_from(f"<{n_streams}d", raw, off))
        off += 8 * n_streams
        steps.append({
            "ok": bool(row[0]), "frozen": bool(row[1]), "has_t": bool(row[2]),
            "has_prediction": bool(row[3]),
            "frames": row[4], "updated": row[5], "lost_frames": row[6],
            "frozen_by": row[7], "skipped": list(row[8:16]),
            "n_staleness": row[16], "worst": row[17], "t": row[18],
            "prediction": list(row[19:22]), "staleness": stale})
    tail = _TAIL.unpack_from(raw, off)
    off += _TAIL.size
    state_size = tail[7]
    state = np.frombuffer(raw, dtype=dt, count=state_size, offset=off).copy()
    off += state_size * dt.itemsize
    assert off == len(raw), f"unread bytes in the result: {len(raw) - off}"
    return {"accepted": accepted, "steps": steps, "state": state,
            "totals": {"steps": tail[0], "frames": tail[1], "updated": tail[2],
                       "lost": tail[3], "al_frames": tail[4],
                       "al_dropped_stale": tail[5], "al_no_data": tail[6]},
            "frozen": bool(tail[8])}


def run_python(case: Case):
    streams = [Stream(f"s{i}", dim=d, capacity=case.capacity, dtype=case.dtype)
               for i, d in enumerate(case.dims)]
    adapter = RecursiveLeastSquares(case.n_features, case.n_outputs,
                                    forgetting=case.lam, p0=case.p0, dtype=case.dtype)
    if case.guard:
        g = case.guard
        checks = [Bound(field, lo, hi, name=f"b{i}")
                  for i, (field, lo, hi) in enumerate(g.get("bounds", []))]
        if g.get("divergence"):
            d = g["divergence"]
            checks.append(Divergence(ratio=d["ratio"], fast_alpha=d["fast_alpha"],
                                     slow_alpha=d["slow_alpha"], floor=d["floor"],
                                     warmup=d["warmup"]))
        adapter = Guard(adapter, checks, rollback_depth=g.get("depth", 1),
                        checkpoint_every=g.get("checkpoint_every", 1),
                        recover_after=g.get("recover_after") or None)

    fmap = FrameMap([(f"s{s}", c) for s, c in case.features],
                    [(f"s{t[0]}", t[1]) for t in case.targets], constant=case.constant)
    aligner = Aligner(f"s{case.reference}", case.policy,
                      case.max_stale_aligner if case.max_stale_aligner > 0 else None)
    pipe = Pipeline(streams, aligner, adapter, fmap,
                    max_stale_s=case.max_stale_pipeline,
                    require_valid=case.require_valid,
                    allow_non_causal=case.allow_non_causal,
                    require_fresh_target=case.require_fresh_target)

    dt = np.dtype(case.dtype)
    accepted, steps = [], []
    for ev in case.events:
        if ev[0] == "push":
            _, s, t, values, valid = ev
            try:
                streams[s].push(t=t, values=np.asarray(values, dtype=dt), valid=valid)
                accepted.append(1)
            except ValueError:
                accepted.append(0)
            continue
        rep = pipe.step()
        buckets = [0] * len(SKIP)
        for token, n in rep.skipped.items():
            key = "guard" if token.startswith("guard:") else token
            buckets[SKIP.index(key)] += n
        pred = ([float(v) for v in rep.prediction] + [0.0, 0.0])[:3] \
            if rep.prediction is not None else [0.0, 0.0, 0.0]
        # The scale the cancellation happens at: the largest term summed to make the
        # prediction, not the prediction itself. See PREDICTION_ULP.
        n_theta = case.n_features * case.n_outputs
        theta = np.asarray(adapter.state[:n_theta], dtype=np.float64)
        terms = np.abs(np.outer(np.asarray(pipe._x, dtype=np.float64),
                                np.ones(case.n_outputs)).ravel()
                       * theta.reshape(case.n_features, case.n_outputs).ravel())
        # Finite terms only: a NaN feature makes the whole product NaN, and a NaN scale
        # turns every later comparison into `0.0 <= nan`, which is False. The floor in
        # `compare()` covers the case where nothing is finite.
        finite = terms[np.isfinite(terms)]
        term_scale = float(finite.max()) if (rep.prediction is not None
                                             and finite.size) else 0.0
        steps.append({
            "ok": True, "frozen": rep.frozen, "has_t": rep.t is not None,
            "has_prediction": rep.prediction is not None, "prediction": pred,
            "term_scale": term_scale,
            "frames": rep.frames, "updated": rep.updated,
            "lost_frames": rep.lost_frames,
            "frozen_by": _mask(rep.frozen_by, case),
            "skipped": buckets,
            "n_staleness": len(rep.staleness_s),
            "worst": rep.worst_staleness_s, "t": rep.t if rep.t is not None else 0.0,
            "staleness": [rep.staleness_s.get(f"s{i}", 0.0)
                          for i in range(len(case.dims))]})
    st = pipe.stats
    return {"accepted": accepted, "steps": steps, "state": np.array(adapter.state),
            "totals": {"steps": st["steps"], "frames": st["frames"],
                       "updated": st["updated"], "lost": st["lost_frames"],
                       "al_frames": aligner.stats["frames"],
                       "al_dropped_stale": aligner.stats["dropped_stale"],
                       "al_no_data": aligner.stats["no_data"]},
            "frozen": pipe.frozen}


def _mask(names, case: Case) -> int:
    """The Python's check names as the C++'s bitmask: bounds in declaration order, the
    divergence check at bit 31, exactly as `guard.hpp` assigns them."""
    out = 0
    for name in names:
        if name == "divergence":
            out |= 1 << 31
        else:
            out |= 1 << int(name[1:])
    return out


#: How far a prediction may differ, in ulp of **the scale the cancellation happens at**.
#:
#: Three metrics were tried and the first two were wrong. Bit equality fails at f32 on the
#: first frame that carries an output, because a prediction is `x'theta` and theta already
#: carries `rls.hpp`'s declared divergences. Element-wise relative error fails at 217 ulp at
#: f64 on a step where the prediction happens to be near zero — the familiar cancellation
#: artefact, which had already retired element-wise comparisons of the state block and of the
#: a posteriori residual.
#:
#: The third was normalising by the largest *prediction* in the run, and a parity review
#: pointed out that is still the cancelled quantity's own scale. This project's rule for a
#: cancelled quantity is the scale of the values being subtracted, which here is
#: `max_j |x_j theta_j|` — the largest term summed to make the prediction. The two coincide
#: when theta and x are O(1), which is why the quiet fixtures never showed the difference, and
#: diverge under covariance windup.
#:
#: **The default is for well-posed cases and there is no single number that covers every
#: regime.** Measured under the corrected metric, on both architectures, across an SNR sweep
#: from 300 down to 1: 4.8 ulp at SNR 300, 48 at SNR 3, and 152 at SNR 1 -- so 200 holds for
#: anything with a sane prior. Start the covariance far larger than the data warrants
#: (`p0 = 1000`) and the same code reaches **947 ulp**, because that is where the cancellation
#: is worst.
#:
#: A single bound would therefore either fail the ill-conditioned cases or be useless on the
#: rest. `compare()` takes the bound as an argument and the cases that deliberately choose a
#: bad prior declare their own, with the measurement beside it. A tolerance is a property of
#: the regime, not of the port, and pretending otherwise is how one gets loosened until it
#: stops meaning anything.
PREDICTION_ULP = 200.0

#: For a prior chosen far larger than the data warrants. Measured worst 947 ulp on the
#: prediction and 249 on the state block, both at `p0 = 1000` with the noise the size of the
#: signal, on both architectures.
ILL_CONDITIONED_PREDICTION_ULP = 1500.0
ILL_CONDITIONED_STATE_ULP = 500.0


def compare(case: Case, py, cpp, *, state_ulps=400.0, prediction_ulps=PREDICTION_ULP):
    """Everything but the model state and the prediction, exactly. Staleness by bit
    pattern; those two carry `rls.hpp`'s three declared divergences and get a tolerance."""
    assert py["accepted"] == cpp["accepted"], "the two accepted different pushes"
    assert len(py["steps"]) == len(cpp["steps"])
    eps = float(np.finfo(np.dtype(case.dtype)).eps)
    pred_scale = max([abs(v) for st in py["steps"] if st["has_prediction"]
                      for v in st["prediction"]] + [1e-300])
    for i, (a, b) in enumerate(zip(py["steps"], cpp["steps"])):
        assert b["ok"], f"step {i}: the C++ pipeline refused its configuration"
        for key in ("frames", "updated", "lost_frames", "frozen", "has_t",
                    "frozen_by", "skipped", "n_staleness", "has_prediction"):
            assert a[key] == b[key], f"step {i}: {key}: python {a[key]} != cpp {b[key]}"
        assert _bits(a["worst"]) == _bits(b["worst"]), f"step {i}: worst staleness"
        assert _bits(a["t"]) == _bits(b["t"]), f"step {i}: reference time"
        for s in range(len(case.dims)):
            assert _bits(a["staleness"][s]) == _bits(b["staleness"][s]), (
                f"step {i}: staleness of stream {s}")
        # The output, on every case rather than only the one named for it. Not bits: a
        # prediction is x'theta and theta already carries rls.hpp's three declared
        # divergences, so the output inherits them.
        if a["has_prediction"]:
            # The term scale, floored well below the run scale so a step whose terms all
            # vanish does not divide by nothing. Both come from the Python side, so the two
            # implementations are never compared against a scale one of them chose.
            scale = max(a.get("term_scale", 0.0), pred_scale * 1e-6, 1e-300)
            for j in range(len(a["prediction"])):
                pa, pb = a["prediction"][j], b["prediction"][j]
                # A frame carrying a NaN feature must produce a NaN output on BOTH sides.
                # The adapter refuses to learn from it, but refusing to learn is not a
                # reason to hand back a number: a stale value presented as this frame's
                # answer is worse than a NaN a caller can test for. Compared as a property
                # rather than by subtraction, since nan - nan is nan.
                if not (math.isfinite(pa) and math.isfinite(pb)):
                    assert math.isnan(pa) == math.isnan(pb) and (
                        math.isnan(pa) or pa == pb), (
                        f"step {i}: prediction[{j}] {pa!r} vs {pb!r}")
                    continue
                assert abs(pa - pb) <= prediction_ulps * eps * scale, (
                    f"step {i}: prediction[{j}] {pa!r} != {pb!r}, which is "
                    f"{abs(pa - pb) / eps / scale:.1f} ulp of the term scale")
    assert py["totals"] == cpp["totals"], "the cumulative counters disagree"
    assert py["frozen"] == cpp["frozen"]

    a, b = np.asarray(py["state"], np.float64), np.asarray(cpp["state"], np.float64)
    assert a.shape == b.shape
    scale = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))), 1e-300)
    dev = float(np.max(np.abs(a - b))) / scale
    tol = state_ulps * float(np.finfo(np.dtype(case.dtype)).eps)
    assert dev <= tol, (
        f"the learned state diverged by {dev / float(np.finfo(np.dtype(case.dtype)).eps):.1f}"
        f" ulp, over the {state_ulps} allowed for rls.hpp's three declared divergences")
    return dev


def both(pipecheck, tmp_path, case: Case, **kw):
    py = run_python(case)
    cpp = run_cpp(pipecheck, tmp_path, case)
    compare(case, py, cpp, **kw)
    return py, cpp


# -- case builders ----------------------------------------------------------------------


def simple(dtype="f8", capacity=64, policy="hold_last", n=40, **kw):
    """Two streams at different rates: s0 the 1 Hz reference carrying the target, s1 a
    2-wide sensor at ~3 Hz. The shape the design driver has — a fast aiding source resampled
    onto a slower reference — rather than a single-rate fixture, which cannot tell any two
    policies apart.

    The fast stream is **phase-offset and jittered**, and both are load-bearing rather than
    realism. The first version ran it at exactly 1/3 s from zero, so a sample landed within
    1e-16 of every reference instant; `nearest` and `interpolate` then returned the same
    value to sixteen digits and the discrimination test below caught it. Jitter is what makes
    staleness vary, without which `drop_if_stale` either abandons every frame or none.
    """
    case = Case([1, 2], capacity, [(1, 0), (1, 1)], (0, 0), constant=True,
                policy=policy, dtype=dtype, **kw)
    rng = np.random.default_rng(7)
    fast = 0
    for k in range(1, n + 1):
        # Up to and just past this reference instant, so the non-causal policies have a
        # future sample to consult and can differ from hold-last at all.
        while fast / 3.0 + 0.15 < k + 0.25:
            case.push(1, fast / 3.0 + 0.15 + float(rng.uniform(-0.08, 0.08)),
                      rng.normal(size=2))
            fast += 1
        case.push(0, float(k), [float(rng.normal())])
        case.step()
    return case



def noisy(dtype="f8", noise=3.0, p0=100.0, n=60, seed=7):
    """The design driver's shape at a realistic signal-to-noise ratio.

    A parity review measured that every fixture in this suite is far quieter than anything a
    sensor produces: `excited()` puts `0.01*randn` on a signal of about 3, so its SNR is
    around 200, and the "about 1.5 ulp on theta" headline the whole tolerance rests on is a
    property of that. At SNR ≈ 1 the same code disagrees by tens of ulp, and over a wider
    grid than either suite sweeps it reaches four figures.

    That is not the port: replaying the same `(x, y)` through plain RLS gives the identical
    figure to the last digit. It is the declared RLS divergence at conditions the baseline
    never visited. So the fixture set gets one case where the noise is the same size as the
    signal, and the tolerances are checked against it rather than only against the quiet ones.
    """
    case = Case([1, 3, 2], 64, [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1)], (0, 0),
                constant=True, n_features=6, dtype=dtype, p0=p0)
    rng = np.random.default_rng(seed)
    truth = np.linspace(1.0, 2.0, 5)
    for k in range(1, n + 1):
        a = rng.standard_normal(3)
        b = rng.standard_normal(2)
        y = float(np.concatenate([a, b]) @ truth) + 0.3 + noise * float(rng.standard_normal())
        case.push(1, k - 0.4, list(a))
        case.push(2, k - 0.2, list(b))
        case.push(0, float(k), [y])
        case.step()
    return case


@pytest.mark.parametrize("dtype", ["f4", "f8"])
def test_the_two_agree_when_the_noise_is_the_size_of_the_signal(pipecheck, tmp_path, dtype):
    """SNR ≈ 1, which is a sensor rather than a demonstration.

    Measured across an SNR sweep from 300 down to 1 and `p0` from 1 to 1000, on both
    architectures: worst state deviation 99 ulp and worst prediction 134 ulp, against the 400
    and 200 the suite allows. Both bounds hold here with room, and now they hold somewhere
    that is not 200x quieter than the problem.
    """
    py, cpp = both(pipecheck, tmp_path, noisy(dtype=dtype))
    assert py["totals"]["updated"] >= 55, "this fixture is meant to adapt"


@pytest.mark.parametrize("dtype", ["f4", "f8"])
@pytest.mark.parametrize("p0", [1.0, 1000.0])
def test_a_badly_chosen_prior_still_agrees_within_its_own_tolerance(pipecheck, tmp_path,
                                                                    p0, dtype):
    """`p0 = 1000` starts the covariance far larger than the data warrants. It is a prior a
    user can choose, and it is where the prediction's cancellation is worst -- so it is where
    a metric bug hides, and it is why the old metric (normalise by the run's largest
    prediction, not by the largest term summed to make it) looked fine for a day.

    It gets its own tolerance rather than loosening the default, because the deviation is a
    property of the conditioning and not of the port: measured 947 ulp on the prediction and
    249 on the state, against 152 and 28 at the same noise with `p0 = 100`. Replaying the same
    data through plain RLS gives the identical figure, which is what says conditioning.
    """
    tolerances = ({} if p0 <= 100.0 else
                  {"state_ulps": ILL_CONDITIONED_STATE_ULP,
                   "prediction_ulps": ILL_CONDITIONED_PREDICTION_ULP})
    both(pipecheck, tmp_path, noisy(dtype=dtype, p0=p0), **tolerances)

# -- the tests --------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", ["f4", "f8"])
@pytest.mark.parametrize("capacity", [4, 64])
def test_the_ordinary_case_agrees_step_for_step(pipecheck, tmp_path, dtype, capacity):
    py, _ = both(pipecheck, tmp_path, simple(dtype=dtype, capacity=capacity))
    assert py["totals"]["updated"] > 20, "this fixture is meant to adapt, not to idle"


@pytest.mark.parametrize("policy", ["hold_last", "nearest", "interpolate",
                                    "drop_if_stale"])
def test_every_policy_agrees(pipecheck, tmp_path, policy):
    non_causal = policy in ("nearest", "interpolate")
    case = simple(policy=policy, allow_non_causal=non_causal,
                  max_stale_aligner=0.2 if policy == "drop_if_stale" else 0.0)
    both(pipecheck, tmp_path, case)


def test_the_policies_are_distinguishable_on_this_fixture(pipecheck, tmp_path):
    """The negative control, and it is the number to distrust: if every policy produced the
    same trajectory then all four tests above would pass on a port that collapsed them.

    `drop_if_stale` is compared by what only it can do — abandon a frame — because it shares
    `_pick` with `hold_last` and cannot differ in the sample it returns. That confusion
    produced a wrong figure in the aligner's own suite and is not repeated here.
    """
    states = {}
    for policy in ("hold_last", "nearest", "interpolate"):
        case = simple(policy=policy, allow_non_causal=policy != "hold_last")
        states[policy] = run_python(case)["state"]
    pairs = [("hold_last", "nearest"), ("hold_last", "interpolate"),
             ("nearest", "interpolate")]
    for a, b in pairs:
        assert not np.allclose(states[a], states[b], rtol=1e-6), (
            f"{a} and {b} learned the same model on this fixture, so it cannot "
            "distinguish them and the parity tests above prove less than they look")

    dropped = run_python(simple(policy="drop_if_stale", max_stale_aligner=0.2))
    assert dropped["totals"]["al_dropped_stale"] > 0, (
        "drop_if_stale abandoned no frame, so nothing here exercises the only decision "
        "that separates it from hold_last")


def test_a_ring_that_overflows_between_steps_reports_lost_frames(pipecheck, tmp_path):
    """The gap a test found in the reference: 200 frames against a 64-deep ring adapted 64
    times and reported nothing amiss. Here the reference stream is 4 deep and ten samples
    arrive between steps, so six frames per step can never exist."""
    case = Case([1, 1], 4, [(1, 0)], (0, 0), constant=True)
    t = 0.0
    for _ in range(5):
        for _ in range(10):
            t += 1.0
            case.push(0, t, [math.sin(t)])
            case.push(1, t, [math.cos(t)])
        case.step()
    py, cpp = both(pipecheck, tmp_path, case)
    assert py["totals"]["lost"] > 0, "the fixture no longer overflows"
    assert cpp["totals"]["lost"] == py["totals"]["lost"]
    assert not any(s["frames"] > 4 for s in cpp["steps"]), (
        "a step emitted more frames than the reference ring can hold")


def test_declined_instants_and_an_overflowing_ring_in_one_run(pipecheck, tmp_path):
    """Both terms of the loss arithmetic moving at once, which nothing else here does.

    `lost = (arrived - arrived_prev) - considered`, and `considered` is a *delta* of the
    aligner's counters taken inside the step. Every other case has either no declined
    instants or no overflow, so the delta and the absolute count are equal and an
    implementation that used the absolute count would pass the whole suite. A mutation check
    found exactly that: baselining `before_no_data` at zero survived everything. This case is
    what kills it.

    Phase one declines four instants because s1 has nothing yet; phase two frames four while
    six more are overwritten. If `considered` were cumulative, phase two would count phase
    one's four declines again and report two lost instead of six.
    """
    case = Case([1, 1], 4, [(1, 0)], (0, 0), constant=True)
    for k in range(1, 11):
        case.push(0, float(k), [float(k)])          # reference only; s1 is silent
    case.step()
    case.push(1, 10.5, [1.0])
    for k in range(11, 21):
        case.push(0, float(k), [float(k)])
    case.step()

    py, cpp = both(pipecheck, tmp_path, case)
    first, second = py["steps"]
    assert (first["frames"], first["lost_frames"]) == (0, 6), (
        f"phase one should decline four and lose six, got {first}")
    assert (second["frames"], second["lost_frames"]) == (4, 6), (
        f"phase two should frame four and lose six, got {second}")
    assert py["totals"]["al_no_data"] == 4


def test_a_step_with_nothing_new_is_idle_on_both_sides(pipecheck, tmp_path):
    case = Case([1, 1], 8, [(1, 0)], (0, 0), constant=True)
    case.push(0, 1.0, [1.0]).push(1, 1.0, [2.0]).step().step().step()
    py, cpp = both(pipecheck, tmp_path, case)
    assert [s["frames"] for s in cpp["steps"]] == [1, 0, 0]
    assert py["totals"]["frames"] == 1


def test_the_input_gate_skips_for_the_same_reason(pipecheck, tmp_path):
    """Both halves of the gate, and the reasons must not blur: a stale input and an invalid
    one are different faults and land in different counters."""
    case = Case([1, 1], 64, [(1, 0)], (0, 0), constant=True, max_stale_pipeline=1.5)
    case.push(1, 0.5, [1.0], valid=False)
    for k in range(1, 9):
        case.push(0, float(k), [float(k)])
        if k in (3, 6):
            case.push(1, float(k) - 0.1, [2.0], valid=(k == 6))
    case.step()
    py, cpp = both(pipecheck, tmp_path, case)
    totals = [sum(s["skipped"][i] for s in cpp["steps"]) for i in range(len(SKIP))]
    assert totals[SKIP.index("stale")] > 0, "no frame was gated for staleness"
    assert totals[SKIP.index("invalid")] > 0, "no frame was gated for validity"
    assert py["steps"][0]["skipped"] == cpp["steps"][0]["skipped"]


def test_a_nan_from_a_sensor_is_rejected_identically(pipecheck, tmp_path):
    """A sensor NaN is a runtime event, not a programming error: the adapter must count it
    and carry on, and the count must land in `nonfinite_input` on both sides rather than in
    the catch-all."""
    case = Case([1, 1], 64, [(1, 0)], (0, 0), constant=True)
    for k in range(1, 9):
        case.push(0, float(k), [float(k)])
        case.push(1, float(k), [float("nan") if k in (3, 5) else 1.0 * k])
        case.step()
    py, cpp = both(pipecheck, tmp_path, case)
    nonfinite = sum(s["skipped"][SKIP.index("nonfinite_input")] for s in cpp["steps"])
    assert nonfinite == 2, f"expected two rejected frames, got {nonfinite}"
    assert sum(s["skipped"][SKIP.index("rejected")] for s in py["steps"]) == 0, (
        "a reason landed in the catch-all, which means the mapping lost it")


@pytest.mark.parametrize("depth", [1, 4])
@pytest.mark.parametrize("checkpoint_every", [1, 3])
def test_a_guarded_pipeline_freezes_at_the_same_frame(pipecheck, tmp_path, depth,
                                                      checkpoint_every):
    """The guard trips inside a step, and a step that froze half-way reports frozen even
    though earlier frames in it applied. That is the interaction only this level has."""
    case = Case([1, 1], 64, [(1, 0)], (0, 0), constant=True, p0=1.0, lam=0.995,
                guard={"depth": depth, "checkpoint_every": checkpoint_every,
                       "bounds": [("residual_norm", None, 5.0)]})
    rng = np.random.default_rng(3)
    for k in range(1, 41):
        x = float(rng.normal())
        y = 2.0 * x + 1.0
        if k == 25:
            y += 500.0            # the fault, mid-step
        case.push(1, float(k), [x])
        case.push(0, float(k), [y])
        if k % 4 == 0:
            case.step()
    py, cpp = both(pipecheck, tmp_path, case)
    assert py["frozen"], "the fixture never tripped, so it tests nothing about freezing"
    guard_skips = sum(s["skipped"][SKIP.index("guard")] for s in cpp["steps"])
    frozen_skips = sum(s["skipped"][SKIP.index("frozen")] for s in cpp["steps"])
    assert guard_skips == 1, f"expected exactly one trip, got {guard_skips}"
    assert frozen_skips > 0, "nothing was blocked after the freeze"
    tripped = [s["frozen_by"] for s in cpp["steps"] if s["frozen_by"]]
    assert tripped == [1], f"the tripped mask should name bound 0 once, got {tripped}"


def test_the_divergence_check_agrees_through_a_pipeline(pipecheck, tmp_path):
    """A stateful check, driven through the pipeline rather than directly. Its EWMAs are
    held across a freeze and reset on thaw, and a pipeline that fed it a different set of
    frames would drift out of step with the reference within a few samples."""
    case = Case([1, 1], 64, [(1, 0)], (0, 0), constant=True, p0=1.0, lam=0.99,
                guard={"depth": 4, "checkpoint_every": 2,
                       "divergence": {"ratio": 4.0, "fast_alpha": 0.4,
                                      "slow_alpha": 0.05, "floor": 1e-3,
                                      "warmup": 10}})
    rng = np.random.default_rng(11)
    for k in range(1, 61):
        x = float(rng.normal())
        y = 3.0 * x - 0.5 + (80.0 if k == 45 else 0.0)
        case.push(1, float(k), [x])
        case.push(0, float(k), [y])
        if k % 3 == 0:
            case.step()
    py, cpp = both(pipecheck, tmp_path, case)
    assert py["frozen"], "the divergence check never fired"
    assert any(s["frozen_by"] == (1 << 31) for s in cpp["steps"]), (
        "the divergence bit is not what the C++ reported")


def test_a_three_stream_map_scatters_components_to_the_same_places(pipecheck, tmp_path):
    """Feature order is parameter order, and the C++ extracts by scattering from one stream
    at a time rather than gathering per feature. Interleaving the map across streams is what
    makes a transposed scatter visible: a map that took every component from one stream
    would agree under either implementation."""
    case = Case([1, 3, 2], 64,
                [(1, 2), (2, 0), (1, 0), (2, 1), (1, 1)], (0, 0),
                constant=True, n_features=6)
    rng = np.random.default_rng(23)
    for k in range(1, 31):
        case.push(1, float(k) - 0.3, list(rng.normal(size=3)))
        case.push(2, float(k) - 0.1, list(rng.normal(size=2)))
        case.push(0, float(k), [float(rng.normal())])
        case.step()
    py, cpp = both(pipecheck, tmp_path, case)
    assert py["totals"]["updated"] >= 29


def test_a_permuted_feature_order_learns_a_different_model(pipecheck, tmp_path):
    """The negative control for the test above. If the map's order did not reach the
    regressor, a scatter that ignored it would still pass parity — both sides would be
    equally wrong.

    This is the failure `persist.py`'s configuration fingerprint exists for: order is not
    recoverable from a state block, so getting it wrong is undetectable downstream.
    """
    order = [(1, 2), (2, 0), (1, 0), (2, 1), (1, 1)]
    base = None
    for perm in (order, [order[i] for i in (1, 0, 3, 2, 4)]):
        case = Case([1, 3, 2], 64, perm, (0, 0), constant=True, n_features=6)
        rng = np.random.default_rng(23)
        for k in range(1, 31):
            case.push(1, float(k) - 0.3, list(rng.normal(size=3)))
            case.push(2, float(k) - 0.1, list(rng.normal(size=2)))
            case.push(0, float(k), [float(rng.normal())])
            case.step()
        state = run_python(case)["state"]
        if base is None:
            base = state
        else:
            assert not np.allclose(base[:6], state[:6]), (
                "permuting the feature order changed nothing, so this fixture cannot "
                "detect a scatter that ignores it")


def test_the_drain_watermark_is_not_reoffered(pipecheck, tmp_path):
    """An instant the aligner declined is still an instant it has seen. If the watermark
    advanced only on success, a reference sample that could not be framed would be retried
    on every later step and `no_data` would climb without bound."""
    case = Case([1, 1], 8, [(1, 0)], (0, 0), constant=True)
    for k in range(1, 4):
        case.push(0, float(k), [float(k)])     # reference only; s1 has nothing yet
    case.step().step()
    case.push(1, 3.5, [1.0])
    case.step()
    py, cpp = both(pipecheck, tmp_path, case)
    assert py["totals"]["al_no_data"] == 3, (
        f"expected three declined instants, got {py['totals']['al_no_data']}")
    assert cpp["steps"][1]["frames"] == 0 and cpp["steps"][2]["frames"] == 0, (
        "a declined instant came back")


def test_the_freshness_gate_agrees_across_the_boundary(pipecheck, tmp_path):
    """The gate that lets a fast output run on a slow target, compared step for step.

    A 1-wide reference at 10 Hz carrying features and a 1 Hz stream carrying the target: the
    configuration low latency asks for and the one where hold-last hands the same measurement
    to the learner ten times over. Both implementations decide freshness from the sample's own
    timestamp, so they have to agree on every frame or one of them is learning from a different
    set of measurements.

    This case cannot see the reconstruction defect that
    `test_a_gap_in_the_target_stream_carries_across_the_boundary` covers, and the reason is
    worth stating: `k / 10.0` is exactly representable and there is no gap in the target
    stream, so rebuilding the timestamp from staleness round-trips and the two mechanisms are
    indistinguishable here.
    """
    case = Case([1, 1], 64, [(0, 0)], (1, 0), constant=True,
                require_fresh_target=True)
    j = 0
    for k in range(1, 61):
        case.push(0, k / 10.0, [float(k % 7)])
        if k % 10 == 0:
            j += 1
            case.push(1, k / 10.0, [2.0 + 0.01 * k])
        case.step()
    py, cpp = both(pipecheck, tmp_path, case)
    gated = sum(s["skipped"][SKIP.index("stale_target")] for s in cpp["steps"])
    assert py["totals"]["updated"] == j, (
        f"expected one adaptation per distinct target, got {py['totals']['updated']} "
        f"against {j} measurements")
    assert gated > 0, "no frame was gated, so this compares nothing"
    assert gated + py["totals"]["updated"] == py["totals"]["frames"]


def test_a_gap_in_the_target_stream_carries_across_the_boundary(pipecheck, tmp_path):
    """The case the harness was missing, and it was missing for two reasons at once.

    The freshness gate used to rebuild the target's timestamp as `t_reference + staleness` on
    all three implementations. That addition is not bit-stable across reference instants, so
    after a gap in the target stream the same physical sample read as new: measured in the
    reference, **52 of 120 frames relearned a 14-second-old target**. The fix is `sample_t`
    carried on the pick rather than derived.

    Every existing case here misses it because every one of them uses exactly representable
    timestamps (`k / 10.0`) and keeps the target stream running. This one accumulates, as a
    real caller does, and then goes quiet on the target while the reference keeps arriving.

    Verified against its own mutant: putting the reconstruction back into `pipeline.hpp` makes
    this fail, where before it was caught only by `SIZES.md` noticing the flash had moved --
    which is an accident, not coverage.
    """
    case = Case([1, 1], 64, [(0, 0)], (1, 0), constant=True, dtype="f8",
                require_fresh_target=True)
    t = 0.0
    for k in range(1, 61):
        t += 1.0 / 30.0                      # accumulated, so t is not exactly representable
        case.push(0, t, [float(k % 7)])
        case.push(1, t, [2.0 + 0.01 * k])
        case.step()
    gap = t
    for k in range(1, 61):
        case.push(0, gap + 10.0 + k / 30.0, [float(k % 7)])
        case.step()

    py, cpp = both(pipecheck, tmp_path, case)
    gated = sum(s["skipped"][SKIP.index("stale_target")] for s in cpp["steps"])
    assert py["totals"]["frames"] == 120
    assert py["totals"]["updated"] == 60, (
        f"{py['totals']['updated']} adaptations against 60 distinct targets: a stale target "
        "reached the learner, which is what carrying `sample_t` prevents")
    assert gated == 60, f"{gated} frames gated, expected the 60 after the gap"
    # The fixture's own guard: without accumulation this case is the tidy one again and proves
    # nothing, so assert the timestamps really are inexact.
    assert gap != 2.0 and gap == pytest.approx(2.0), gap


def test_the_prediction_crosses_the_boundary_on_every_frame(pipecheck, tmp_path):
    """Including frames the gate refused to learn from, which is the whole point of having
    the output separate from the adaptation.

    Compared with a tolerance, and the first two attempts at that tolerance were both wrong.
    A prediction is `x'θ`, and θ already differs between the two by the three divergences
    `rls.hpp` declares, so the output inherits them and a **bit** comparison fails on the
    tenth frame at f32. Switching to element-wise relative error then failed too, at 217 ulp
    on a step where the prediction happens to be small — the same cancellation artefact that
    already retired element-wise comparisons of the state block and of the a posteriori
    residual. Against the run's own scale that step is 14.0 ulp.

    What *is* exact is the widening: the report carries the prediction as a double so one
    record shape covers both precisions, and float32 to double is lossless. Anything beyond
    what θ's divergence explains is a plumbing bug and this tolerance would catch it.
    """
    for dtype in ("f4", "f8"):
        case = Case([1, 1], 64, [(0, 0)], (1, 0), constant=True, dtype=dtype,
                    require_fresh_target=True)
        for k in range(1, 41):
            case.push(0, k / 10.0, [float(k % 5) - 2.0])
            if k % 10 == 0:
                case.push(1, k / 10.0, [1.5 + 0.02 * k])
            case.step()
        py, cpp = both(pipecheck, tmp_path, case)
        # 31 of 40, and the nine missing ones are honest: the target stream is silent
        # until k=10, so those steps have no frame at all and there is nothing to predict
        # on. This assertion was written as ">= 35" before being measured and was wrong --
        # the same mistake three times in this session, which is why the figure is exact
        # now and carries its derivation.
        with_pred = [s for s in cpp["steps"] if s["has_prediction"]]
        assert len(with_pred) == 31, (
            f"{len(with_pred)} of {len(cpp['steps'])} steps carried an output")
        assert all(not s["has_prediction"] for s in cpp["steps"][:9])
        # ...and a gated frame is among the ones that did produce an output, which is the
        # property this test exists for rather than a count.
        assert any(s["has_prediction"] and s["skipped"][SKIP.index("stale_target")]
                   for s in cpp["steps"]), (
            "no step both refused to learn and still produced an output")
        for i, (a, b) in enumerate(zip(py["steps"], cpp["steps"])):
            assert a["has_prediction"] == b["has_prediction"], f"step {i}"
            if not a["has_prediction"]:
                # A step with no frame must carry no output on either side, and the padding
                # past `n_outputs` must be zero rather than whatever was there before.
                assert b["prediction"] == [0.0, 0.0, 0.0], f"step {i}: {b['prediction']}"
                continue
            assert b["prediction"][1:] == [0.0, 0.0], (
                f"step {i}: this instantiation has one output; the rest must be zeroed")
        # The values themselves are compared by `compare()` above, on every case rather than
        # only here; what this test adds is that the output exists on the frames the gate
        # refused, which no other case exercises.


def test_a_step_whose_last_instant_fails_still_reports_its_output(pipecheck, tmp_path):
    """The shape that found a real divergence, kept so it cannot come back.

    One step drains two reference instants; the first builds a frame and the second does not.
    The C++ used to clear its frame-ready flag on the failed build and then withhold the
    step's output, while the reference -- which only ever sees frames that exist -- predicted
    normally. A parity review reproduced it 118 times in 250 randomised cases, identically on
    the host and the target, so it was logic rather than rounding.

    `drop_if_stale` is the reachable way to make a later instant fail: streams are monotonic,
    so a second instant cannot predate the first stream's data, but a policy can abandon it.

    What this catches, checked rather than assumed: restoring **both** halves of the old
    behaviour -- clearing the flag on entry to `build_frame` and gating the output on it --
    fails this test. Restoring the gate alone does not, because the fix also stopped a failed
    build from touching the flag, so the gate now reads a value that is always true once a
    frame exists. The gate alone is an equivalent mutant, and saying which mutant a test kills
    matters more than saying that it kills one.
    """
    case = Case([1, 1], 8, [(1, 0)], (0, 0), constant=True,
                policy="drop_if_stale", max_stale_aligner=0.4)
    case.push(1, 1.0, [2.0])
    case.push(0, 1.1, [3.0])          # staleness -0.1: kept
    case.push(0, 1.9, [3.0])          # staleness -0.9: abandoned, and it is the last
    case.step()
    py, cpp = both(pipecheck, tmp_path, case)
    assert py["steps"][0]["frames"] == 1
    assert py["steps"][0]["has_prediction"], "the reference should predict on the good frame"
    assert cpp["steps"][0]["has_prediction"], (
        "the port withheld the output because a later instant failed")
    assert py["totals"]["al_dropped_stale"] == 1, "the fixture no longer abandons anything"


def test_a_refused_push_is_visible_to_the_comparison(pipecheck, tmp_path):
    """`pipecheck` calls a refusal "data", and until now it was data no case produced: an
    audit instrumented the harness and found `accepted` held the single value 1 across the
    entire suite, so a tool that hardcoded it would have passed everything."""
    case = Case([1, 1], 8, [(1, 0)], (0, 0), constant=True)
    case.push(1, 1.0, [2.0])
    case.push(0, 1.0, [3.0])
    case.push(0, 1.0, [4.0])                        # equal timestamp: refused
    case.push(0, 0.5, [5.0])                        # backwards: refused
    case.push(0, float("nan"), [6.0])               # not finite: refused
    case.push(0, 2.0, [7.0])                        # and ordering still works after
    case.step()
    py, cpp = both(pipecheck, tmp_path, case)
    assert py["accepted"] == cpp["accepted"]
    assert py["accepted"] == [1, 1, 0, 0, 0, 1], py["accepted"]


@pytest.mark.parametrize("fresh", [False, True])
def test_three_outputs_through_the_freshness_gate(pipecheck, tmp_path, fresh):
    """The axis that had never been executed.

    The gate decides from the target sample's timestamp, and the two implementations get there
    differently: the reference builds a tuple over the *unique target streams* and compares it
    whole, the core iterates *outputs* and ORs "changed". Equivalent by construction — and
    `pipecheck` instantiated only `Rls<T, N, 1>`, so an audit could check that by reading and
    not by running. It can now.

    Two of the three targets share a stream and the third has its own, which is the case where
    "unique streams" and "per output" are genuinely different loops. The reference is the fast
    feature stream, because that is the wiring the gate exists for -- with a slow reference
    every frame carries a new target and the gate can never fire, which is how the first
    version of this fixture managed to test nothing.
    """
    case = Case([2, 1, 3], 64,
                [(2, 0), (2, 1), (2, 2), (0, 0), (1, 0)],
                [(0, 0), (0, 1), (1, 0)], constant=True, n_features=6, n_outputs=3,
                dtype="f4", reference=2, require_fresh_target=fresh)
    rng = np.random.default_rng(19)
    for k in range(1, 41):
        case.push(2, k - 0.3, list(rng.normal(size=3)))
        if k % 5 == 0:                       # the two-target stream, slower
            case.push(0, k - 0.1, list(rng.normal(size=2)))
        if k % 3 == 0:                       # the one-target stream, slower still
            case.push(1, k - 0.15, [float(rng.normal())])
        case.push(2, float(k), list(rng.normal(size=3)))
        case.step()
    py, cpp = both(pipecheck, tmp_path, case)
    gated = sum(s["skipped"][SKIP.index("stale_target")] for s in cpp["steps"])
    if fresh:
        assert gated > 0, "no frame was gated, so the multi-output path is untested"
    else:
        assert gated == 0


def test_the_totals_are_the_sum_of_the_steps(pipecheck, tmp_path):
    """A cheap internal consistency check on both sides at once. It has caught nothing yet;
    it is here because `lost_frames` is a difference of counters and a per-step figure that
    did not add up to the cumulative one would mean one of them is computed twice."""
    case = simple(n=25, capacity=4)
    py, cpp = both(pipecheck, tmp_path, case)
    for side in (py, cpp):
        assert side["totals"]["frames"] == sum(s["frames"] for s in side["steps"])
        assert side["totals"]["updated"] == sum(s["updated"] for s in side["steps"])
        assert side["totals"]["lost"] == sum(s["lost_frames"] for s in side["steps"])
        assert side["totals"]["steps"] == len(side["steps"])
