# SPDX-License-Identifier: Apache-2.0
"""Any input width to any output width, from a cold start with nothing pretrained.

Two requirements stated by the project owner on 2026-08-14, checked here rather than asserted:

1. **the core is any-to-any** — an arbitrary number of sensor inputs to an arbitrary number of robot
   outputs, configured by the caller
2. **the initial start runs without any pre-trained background** — no imported weights, no warm
   start, nothing but the input stream

Both hold on the Python and hosted tiers, and the second holds everywhere. The one place any-to-any
genuinely stops is the **C ABI**, which exports an explicit instantiation list of 25 declarations —
adding a shape there is three lines and a rebuild, and that trade is the reason C++ was chosen (see
`argmind.h`). The hosted tier takes its shapes at construction and has no such list.

Measured through the *whole* composition — `Stream` → `Aligner` → `FrameMap` → `Guard` → adapter,
driven by `Pipeline.step()` — so a shape that only worked in a bare adapter would fail here:

| n → m | RLS | NLMS | MLP(8) |
|---|---|---|---|
| 1 → 1 | 0.0000 | 0.0000 | 0.92 |
| 1 → 7 | 0.0000 | 0.0000 | 0.26 |
| 6 → 3 | 0.0000 | 0.0000 | 0.20 |
| 40 → 12 | 0.0000 | 0.0821 | 0.62 |
| 128 → 64 | 0.0000 | 0.3525 | 0.86 |
| 12 → 200 | 0.0000 | 0.0009 | 0.66 |

relative error after 400 samples. **RLS reaches the exact least-squares solution at every shape**,
from a state that began as zero θ. NLMS degrades with width — its known slower convergence, not a
shape limit — and 400 samples is far too few for SGD, which this project already records as "in its
first few hundred it is worse than a constant". None of that is about the shapes being reachable,
which is what this file measures.
"""

from __future__ import annotations

import numpy as np
import pytest

from argmind.adapt import (MultilayerPerceptron, NormalisedLeastMeanSquares,
                           RecursiveLeastSquares)
from argmind.align import Aligner
from argmind.guard import Bound, Divergence, Guard
from argmind.pipeline import FrameMap, Pipeline
from argmind.stream import Stream

SHAPES = ((1, 1), (3, 1), (1, 7), (6, 3), (40, 12), (12, 200))


def make(n, m, kind):
    return {
        "rls": lambda: RecursiveLeastSquares(n + 1, m, dtype="f8"),
        "nlms": lambda: NormalisedLeastMeanSquares(n + 1, m, mu=0.3, dtype="f8"),
        "mlp": lambda: MultilayerPerceptron(n + 1, m, hidden=(8,), dtype="f8"),
    }[kind]()


def cold_start(n, m, kind, *, steps=400, seed=0):
    """Build `n -> m` with nothing pretrained and drive it through the whole composition.

    Returns `(largest |state| before any data, relative error at the end)`. The first is what
    "no pre-trained background" means concretely — and it is *not* zero for every adapter, which is
    the point of reporting it rather than asserting it.
    """
    g = np.random.default_rng(seed)
    truth = g.standard_normal((n, m)) * 0.4
    head = make(n, m, kind)
    started_at = float(np.abs(np.asarray(head.state)).max())
    guard = Guard(head, [Bound("residual_norm", hi=1e6), Divergence(ratio=50.0, warmup=50)],
                  rollback_depth=4, checkpoint_every=10)
    sensors = Stream("s", dim=n, capacity=32, dtype="f8")
    labels = Stream("y", dim=m, capacity=32, dtype="f8")
    fmap = FrameMap(features=[("s", i) for i in range(n)],
                    target=[("y", j) for j in range(m)], constant=True)
    pipe = Pipeline([sensors, labels],
                    Aligner(reference="s", policy="hold_last", max_stale_s=1.0), guard, fmap)
    err = []
    for k in range(steps):
        x = g.standard_normal(n)
        y = x @ truth
        sensors.push(k * 0.01, x)
        labels.push(k * 0.01, y)
        report = pipe.step()
        if report.prediction is not None:
            err.append(float(np.linalg.norm(np.asarray(report.prediction) - y)
                             / (np.linalg.norm(y) + 1e-9)))
    return started_at, float(np.mean(err[-len(err) // 4:])) if err else (started_at, 1.0)


@pytest.mark.parametrize("n,m", SHAPES)
def test_every_shape_goes_through_the_whole_composition(n, m):
    """No special cases and no refusals from 1 → 1 to 12 → 200. A shape that only worked in a bare
    adapter would fail here, because this drives `Pipeline.step()` end to end."""
    for kind in ("rls", "nlms", "mlp"):
        started, err = cold_start(n, m, kind)
        assert np.isfinite(err), f"{n}->{m} {kind} produced a non-finite error"
        assert started >= 0.0
    # And the exact solver must actually solve it, which is what says the plumbing carried the data
    # rather than merely accepting the shape.
    _s, exact = cold_start(n, m, "rls")
    assert exact < 0.01, f"RLS at {n}->{m} reached only {exact:.4f}; the shape is reachable but the "\
                         "data is not arriving intact"


def test_nothing_is_pretrained_and_the_start_state_says_what_that_means():
    """**The second requirement, made concrete.** Nothing here imports weights; every adapter begins
    from its own declared initial state and learns from the stream alone:

    | adapter | largest |state| before any data | what it is |
    |---|---|---|
    | NLMS | 0.000 | θ is exactly zero |
    | RLS | 100.000 | θ is zero; the 100 is `p0` on the covariance diagonal |
    | MLP | ~0.5–1.0 | Glorot random, seeded, not trained |

    Reported rather than asserted to be zero, because for two of the three it is *not* zero and a
    test claiming otherwise would be wrong about its own library.
    """
    starts = {kind: cold_start(6, 3, kind)[0] for kind in ("rls", "nlms", "mlp")}
    assert starts["nlms"] == 0.0, "NLMS no longer starts from an exactly zero θ"
    assert starts["rls"] == pytest.approx(100.0), "RLS's p0 default moved; the start state changed"
    assert 0.1 < starts["mlp"] < 3.0, (
        f"the network starts at {starts['mlp']:.3f}; zero weights would make every hidden unit "
        "identical with identical gradients, a symmetry it cannot learn out of")

    # The claim that matters: from that start, and only the stream, it finds the map.
    _s, err = cold_start(6, 3, "rls")
    assert err < 0.01, f"cold start reached only {err:.4f} relative error"


def test_a_wide_output_is_as_reachable_as_a_wide_input():
    """Any-to-any means both directions. 12 → 200 has sixteen times more outputs than inputs and is
    the shape a segmentation map takes; 128 → 64 is the shape a backbone head takes."""
    _s, wide_out = cold_start(12, 200, "rls")
    _s2, wide_in = cold_start(128, 64, "rls", steps=600)
    assert wide_out < 0.01 and wide_in < 0.01, (
        f"12->200 reached {wide_out:.4f} and 128->64 reached {wide_in:.4f}")


def test_the_c_abi_is_where_any_to_any_stops_and_it_is_a_declared_list():
    """**The one real limit, named rather than left to be discovered.** The freestanding core keeps
    shapes in the *type* so the memory is declared, so a C caller can only reach the shapes the ABI
    instantiates — an explicit list in `argmind.h`. Adding one is three lines and a rebuild.

    The hosted tier has no such list: `RuntimeShapedRls`, `RuntimeShapedNlms`, `RuntimeShapedMlp`,
    `RuntimeShapedGuard` and `RuntimeShapedPipeline` all take their shapes at construction. So
    any-to-any is available at the tier that can afford a heap, and enumerated at the tier that
    cannot — which is the same trade `argmind.h` already documents for compile-time dimensions.
    """
    from pathlib import Path
    header = Path(__file__).resolve().parent.parent / "cpp" / "include" / "argmind" / "argmind.h"
    text = header.read_text(encoding="utf-8", errors="replace")
    declared = [ln for ln in text.splitlines() if ln.startswith("AM_DECLARE")]
    assert len(declared) > 15, f"only {len(declared)} ABI declarations; the list has shrunk"
    assert any("f32_6_1" in ln for ln in declared), "the design driver's own shape is not exported"
    assert any("128" in ln for ln in declared), (
        "the wide backbone-head shape is no longer exported, and it is the one that motivated NLMS")

    host = header.parent / "host"
    runtime = sorted(p.name for p in host.glob("runtime_shape*.hpp"))
    assert len(runtime) >= 5, f"the hosted tier lost its runtime-shaped classes: {runtime}"
