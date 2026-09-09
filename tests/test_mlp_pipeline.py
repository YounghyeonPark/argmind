"""A network through the whole composition: streams, aligner, frame map, guard, persist.

`test_mlp.py` shows the adapter is right and `test_mlp_parity.py` shows the port matches it.
Neither shows the piece the scope decision actually rests on: that a model which learns its
own basis drops into the machinery already built, **without the machinery changing**. That is
what `Guard` and `Pipeline` being templates over an `Adapter` was for, and it was demonstrated
once with NLMS — a different algorithm but still one matrix. A network has layers.

Full C++/Python parity for this combination is **deliberately not run**, and the reason is
checkable rather than an excuse: a pipeline's job is bookkeeping — counters, gates, staleness —
and every one of those is driven by shapes and clocks, not by the adapter's arithmetic. That is
already compared for two adapters in `test_runtime_shape_pipeline_parity.py`, and the network's
arithmetic is already compared in `test_mlp_parity.py`. What would be new is the *seam*, and
the seam is what the tests below exercise on both sides. `test_the_seam_is_shape_driven` pins
the premise so that if it ever stops holding, this decision fails rather than the reasoning
quietly going stale.
"""
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from argmind.adapt import MultilayerPerceptron, RecursiveLeastSquares
from argmind.align import Aligner
from argmind.guard import Bound, Guard
from argmind.persist import fingerprint_of, pack, unpack
from argmind.pipeline import FrameMap, Pipeline
from argmind.stream import Stream

REPO = Path(__file__).resolve().parents[1]
CPP = REPO / "cpp"


def build(hidden=(8,), lr=0.02, seed=1, depth=4, cadence=10):
    imu = Stream("imu", dim=3, capacity=64, dtype="f8")
    tgt = Stream("tgt", dim=1, capacity=64, dtype="f8")
    net = MultilayerPerceptron(4, 1, hidden=hidden, learning_rate=lr, dtype="f8", seed=seed)
    guard = Guard(net, [Bound("residual_norm", hi=8.0)],
                  rollback_depth=depth, checkpoint_every=cadence)
    fmap = FrameMap(features=[("imu", 0), ("imu", 1), ("imu", 2)],
                    target=("tgt", 0), constant=True)
    pipe = Pipeline([imu, tgt], Aligner(reference="tgt", policy="hold_last",
                                        max_stale_s=0.5), guard, fmap)
    return imu, tgt, net, guard, fmap, pipe


def drive(imu, tgt, pipe, steps=600, fault=None, seed=3, t0=0.0, invalid_every=0):
    """Returns (reports, t) so a second run continues the timebase. Restarting it makes
    `Stream.push` refuse every sample -- monotonic time is enforced at the source."""
    g = np.random.default_rng(seed)
    t, reports = t0, []
    for k in range(steps):
        t += 0.01
        s = g.standard_normal(3)
        imu.push(t, s, valid=not (invalid_every and k % invalid_every == 0))
        if k % 2 == 0:
            y = float(np.sin(2 * s[0]) + 0.3 * s[1] * s[2])
            if fault is not None and fault[0] <= k < fault[1]:
                y = 40.0
            tgt.push(t, [y])
        r = pipe.step()
        # The prediction must be read NOW. It is a view into a buffer the next step
        # overwrites, so a list of reports read afterwards holds one value repeated -- the
        # first version of this helper did exactly that and the test then reported "the
        # prediction stopped moving" about a pipeline that was working.
        reports.append((r, None if r.prediction is None else float(r.prediction[0])))
    return reports, t


def test_a_network_runs_through_the_whole_composition():
    imu, tgt, net, guard, fmap, pipe = build()
    pairs, _t = drive(imu, tgt, pipe)
    reports = [r for r, _p in pairs]

    frames = sum(r.frames for r in reports)
    assert frames > 250, f"only {frames} frames were built; the fixture is not exercising it"
    assert sum(r.updated for r in reports) == frames
    assert not guard.frozen

    # `StepReport.prediction` is documented as a preallocated buffer the NEXT step
    # overwrites, so it must be copied to be kept. The first version of this test collected
    # the views and read 300 identical predictions -- the trap that docstring warns about,
    # walked into by the test written to exercise it.
    preds = [p for _r, p in pairs if p is not None]
    assert len(preds) == frames
    assert len(set(preds[-20:])) > 1, "the prediction stopped moving"
    held = [r.prediction for r in reports if r.prediction is not None]
    assert len({float(p[0]) for p in held}) == 1, (
        "the report handed out a fresh array per step; that allocates once per step, which "
        "the pipeline is measured not to do -- if this changed on purpose, so must the "
        "docstring that promises a preallocated buffer")

    # ...and it must actually have learned, or the composition ran while doing nothing.
    early = np.mean([abs(p) for _r, p in pairs[:40] if p is not None])
    probe = np.random.default_rng(99).standard_normal((300, 3))
    want = np.array([np.sin(2 * s[0]) + 0.3 * s[1] * s[2] for s in probe])
    got = np.array([guard.predict([s[0], s[1], s[2], 1.0])[0] for s in probe])
    rmse = float(np.sqrt(np.mean((got - want) ** 2)))
    assert rmse < float(np.std(want)) * 0.85, (
        f"the network learned nothing through the pipeline: RMSE {rmse:.4f} against a "
        f"constant predictor's {float(np.std(want)):.4f}")
    assert early > 0.0


def test_the_composed_state_survives_a_record_round_trip():
    """`persist` needs no special case for a network, and the fingerprint still refuses a
    block written under a different `FrameMap` order — which is the corruption nothing
    downstream can detect, and it does not care what the adapter is."""
    imu, tgt, net, guard, fmap, pipe = build()
    drive(imu, tgt, pipe, steps=300)

    fp = fingerprint_of(*[f"{s}:{c}" for s, c in fmap.features], "const", "tgt:0")
    rec = pack(guard, sequence=1, fingerprint=fp)
    back = unpack(rec)
    assert back.state_size == guard.state.size
    assert back.fingerprint == fp

    restored = MultilayerPerceptron(4, 1, hidden=(8,), learning_rate=0.02,
                                    dtype="f8", seed=99)
    restored.load_state(back.state)
    x = [0.3, -1.0, 2.0, 1.0]
    assert np.allclose(restored.predict(x), guard.predict(x)), (
        "a restored network does not predict what the saved one did")

    # A different feature ORDER is the same bytes with a different meaning; the CRC cannot
    # see it and the fingerprint must.
    other = fingerprint_of(*[f"{s}:{c}" for s, c in reversed(fmap.features)],
                           "const", "tgt:0")
    assert other != fp


def test_a_fault_through_the_pipeline_freezes_and_the_guard_limits_the_damage():
    """A fault arriving by the real path -- pushed to a stream, aligned, mapped into a
    frame -- rather than handed straight to `update()`.

    **The damage figure here is a property of this fixture, not of the model class.** At
    this scale the network has only 400 steps of warm-up and has barely learned (RMSE
    0.6555 against a constant predictor's 0.743), so there is little to destroy: 200
    corrupted samples cost 1.53x where a well-trained one at full scale costs 2.93x. That
    full-scale figure is measured in `test_model_class.py`; what is load-bearing *here* is
    the **path**, and the assertions are set at what this fixture actually produces rather
    than at the number the property deserves.
    """
    def run(guarded):
        imu, tgt, net, guard, fmap, pipe = build()
        if not guarded:
            pipe = Pipeline([imu, tgt], Aligner(reference="tgt", policy="hold_last",
                                                max_stale_s=0.5), net, fmap)
        _warm, t1 = drive(imu, tgt, pipe, steps=400)
        probe = np.random.default_rng(99).standard_normal((300, 3))
        want = np.array([np.sin(2 * s[0]) + 0.3 * s[1] * s[2] for s in probe])
        model = guard if guarded else net

        def rmse():
            got = np.array([model.predict([s[0], s[1], s[2], 1.0])[0] for s in probe])
            return float(np.sqrt(np.mean((got - want) ** 2)))

        before = rmse()
        pairs, _t2 = drive(imu, tgt, pipe, steps=500, fault=(0, 400), seed=5, t0=t1)
        froze = next((i for i, (r, _p) in enumerate(pairs) if r.frozen), None)
        return before, rmse(), froze

    b0, a0, never = run(False)
    assert never is None, "an unguarded pipeline reported frozen"
    assert a0 > b0 * 1.4, (
        f"the fault did not damage the unguarded pipeline ({a0/b0:.2f}x), so this fixture "
        "cannot show a guard preventing anything")

    b1, a1, froze = run(True)
    assert froze is not None, (
        "the guard never froze -- a fault that reaches the adapter through streams, the "
        "aligner and the frame map must trip the same checks as one handed to update()")
    assert froze < 5, f"the guard took {froze} steps to notice a target of 40.0"
    assert a1 < b1 * 1.1, f"the guard did not limit the damage: {b1:.4f} -> {a1:.4f}"


def test_the_seam_is_shape_driven_which_is_why_full_parity_is_not_run():
    """The premise behind not building a pipeline-plus-network parity harness.

    A pipeline's own outputs -- frames, updated, lost_frames, staleness, the gates -- must
    depend on shapes and clocks and **not** on which adapter is underneath. If that holds,
    the existing pipeline parity (two adapters) plus the existing MLP parity covers this
    combination, and only the seam is new. If it ever stops holding, this test fails and the
    decision recorded in the module docstring has to be revisited rather than quietly
    inherited.
    """
    def counters(make_adapter):
        imu = Stream("imu", dim=3, capacity=8, dtype="f8")   # small, so it overflows
        tgt = Stream("tgt", dim=1, capacity=8, dtype="f8")
        fmap = FrameMap(features=[("imu", 0), ("imu", 1), ("imu", 2)],
                        target=("tgt", 0), constant=True)
        pipe = Pipeline([imu, tgt], Aligner(reference="tgt", policy="hold_last",
                                            max_stale_s=0.02), make_adapter(),
                     fmap, max_stale_s=0.015)
        g = np.random.default_rng(3)
        out, t = [], 0.0
        for k in range(200):
            t += 0.01
            if k % 5 == 0:                       # imu SLOWER than the reference, so
                imu.push(t, g.standard_normal(3),   # hold_last has to reach back and
                         valid=(k % 35 != 0))       # the pick is genuinely stale
            tgt.push(t + 0.002, [float(g.standard_normal())])
            r = pipe.step() if k % 4 else None      # skipped steps overflow the rings
            if r is not None:
                out.append((r.frames, r.updated, r.lost_frames,
                            tuple(sorted(r.skipped.items())),
                            round(r.worst_staleness_s, 12)))
        return out

    net = counters(lambda: MultilayerPerceptron(4, 1, hidden=(8,), learning_rate=0.02,
                                                dtype="f8", seed=1))
    rls = counters(lambda: RecursiveLeastSquares(4, 1, forgetting=0.99, p0=100.0,
                                                 dtype="f8"))
    assert net == rls, (
        "the pipeline's bookkeeping differs between a network and RLS, so it is NOT "
        "adapter-independent and a pipeline-plus-network parity harness is needed after all")
    # ...and the run has to reach the interesting paths, or the equality is vacuous.
    assert any(f for f, _u, _l, _s, _w in net), "no frames were built"
    assert any(s for _f, _u, _l, s, _w in net), "nothing was ever skipped"
    assert any(w > 0 for *_rest, w in net), "staleness was zero throughout"


CPP_COMPOSE = r"""
#include <stdio.h>
#include <cmath>
#include "argmind/host/runtime_shape_pipeline.hpp"
#include "argmind/host/runtime_shape_guard.hpp"
#include "argmind/host/runtime_shape_mlp.hpp"
using namespace argmind;
using namespace argmind::host;

int main() {
    using Net = RuntimeShapedMlp<double>;
    using Guarded = RuntimeShapedGuard<Net>;

    Net net(4, 1, {8}, Activation::Tanh, 0.02, 7);
    Guarded guarded(std::move(net), 4, 4, 0, 10);
    Bound b; b.field = Field::ResidualNorm; b.hi = 8.0; b.has_hi = true;
    if (!guarded.add_bound(b)) { printf("bound 0\n"); return 1; }

    RuntimeShapedFrameMap map;
    for (int i = 0; i < 3; ++i) map.add_feature(0, i);
    map.set_constant(true);
    map.add_target(1, 0);

    RuntimeShapedPipeline<Guarded> pipe(std::move(guarded),
                                        Aligner(Policy::HoldLast, 0.5), map,
                                        /*reference=*/1);
    pipe.set_max_stale_s(0.5);

    Stream<double, 3, 64> imu;
    Stream<double, 1, 64> tgt;
    StreamView<double> views[2] = {imu.view(), tgt.view()};
    printf("configured %d\n", (int)pipe.configure(views, 2));

    unsigned s = 1; double x[3], y[1], t = 0.0;
    int frames = 0, froze = -1;
    for (int k = 0; k < 600; ++k) {
        t += 0.01;
        for (int i = 0; i < 3; ++i) {
            s = s * 1664525u + 1013904223u;
            x[i] = (double)(s >> 8) / 8388608.0 - 1.0;
        }
        imu.push(t, x, true);
        if (k % 2 == 0) {
            y[0] = (k >= 400 && k < 460) ? 40.0
                 : std::sin(2 * x[0]) + 0.3 * x[1] * x[2];
            tgt.push(t, y, true);
        }
        views[0] = imu.view();
        views[1] = tgt.view();
        const RuntimeShapedStep<double> r = pipe.step(views);
        frames += (int)r.frames;
        if (r.frozen && froze < 0) froze = k;
    }
    printf("frames %d\n", frames);
    printf("froze %d\n", (froze >= 400 && froze < 470) ? 1 : 0);
    printf("state %d\n", pipe.adapter().state_size());
    int finite = 1;
    for (int i = 0; i < pipe.adapter().state_size(); ++i) {
        const double v = pipe.adapter().state()[i];
        if (!(v == v)) finite = 0;
    }
    printf("finite %d\n", finite);
    return 0;
}
"""
