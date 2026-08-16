# SPDX-License-Identifier: Apache-2.0
"""`Adapter.input_gradient()` — the seam a hierarchy is built on, checked against central differences.

**Why this exists.** The 2026-08-16 re-audit found exactly one real constraint behind "학습은 모든
부분에 적용됨": `update(x, y)` needs a target and `UpdateReport` carries no gradient, so a level with
no `y` of its own has nothing to descend on. Everything an `Adapter` reports describes *itself*;
`input_gradient()` is the one quantity that describes what it wants of whatever produced its input.

**Verified against central differences, not against "it learns"** — the same discipline
`Backbone.backward` is held to, and for the same reason: a wrong gradient still trains, just worse,
and the demo's 42-line `learn_features` was never checked this way at all.

Measured, `d(loss)/dx` against `(L(x+h) - L(x-h)) / 2h` on the pre-update parameters:

| adapter | shapes | max relative error |
|---|---|---|
| RLS | 4→2, 7→1, 6→3 | ~1e-10 |
| NLMS | 4→2, 7→1, 6→3 | ~1e-10 |
| MLP `(8,)`, `(6,5)`, `()` | 4→2, 7→1, 6→3 | ~1e-9 |

The MLP is the case that could actually have been wrong: its backward loop stopped one layer short
of the input, so the last hop (`delta @ w[0].T`, with **no** activation derivative, because `acts[0]`
is `x` and the network did not produce it) is new code rather than a rearrangement.
"""

from __future__ import annotations

import numpy as np
import pytest

from argmind.adapt import (Adapter, MultilayerPerceptron, NormalisedLeastMeanSquares,
                           RecursiveLeastSquares)

SHAPES = ((4, 2), (7, 1), (6, 3))


def build(kind, n, m):
    return {
        "rls": lambda: RecursiveLeastSquares(n, m, dtype="f8"),
        "nlms": lambda: NormalisedLeastMeanSquares(n, m, mu=0.3, dtype="f8"),
        "mlp0": lambda: MultilayerPerceptron(n, m, hidden=(), dtype="f8", learning_rate=0.01),
        "mlp1": lambda: MultilayerPerceptron(n, m, hidden=(8,), dtype="f8", learning_rate=0.01),
        "mlp2": lambda: MultilayerPerceptron(n, m, hidden=(6, 5), dtype="f8", learning_rate=0.01),
    }[kind]()


def warm(adapter, n, m, seed=0, steps=25):
    """Move the parameters off their initial state, so the gradient is not evaluated somewhere
    degenerate — at `theta = 0` an RLS input gradient is exactly zero and any sign error passes."""
    g = np.random.default_rng(seed)
    for _ in range(steps):
        adapter.update(g.standard_normal(n), g.standard_normal(m))
    return adapter


def central_difference(adapter, x, y, h=1e-6):
    """`dL/dx` of `L = ||y - predict(x)||^2` at the parameters as they stand, by finite differences.

    Shares no algebra with the implementations — it only calls `predict`, which is the point.
    """
    out = np.zeros_like(x)
    for i in range(x.size):
        hi, lo = x.copy(), x.copy()
        hi[i] += h
        lo[i] -= h
        l_hi = float(np.sum((y - np.asarray(adapter.predict(hi))) ** 2))
        l_lo = float(np.sum((y - np.asarray(adapter.predict(lo))) ** 2))
        out[i] = (l_hi - l_lo) / (2.0 * h)
    return out


@pytest.mark.parametrize("kind", ("rls", "nlms", "mlp0", "mlp1", "mlp2"))
@pytest.mark.parametrize("n,m", SHAPES)
def test_the_input_gradient_agrees_with_central_differences(kind, n, m):
    """The only check worth having. A wrong backward pass still trains."""
    a = warm(build(kind, n, m), n, m)
    g = np.random.default_rng(99)
    x, y = g.standard_normal(n), g.standard_normal(m)

    reference = central_difference(a, x, y)      # BEFORE the update: pre-update parameters
    report = a.update(x, y)
    assert report.applied
    got = np.asarray(a.input_gradient())

    scale = max(float(np.abs(reference).max()), 1e-12)
    err = float(np.abs(got - reference).max()) / scale
    assert err < 1e-6, f"{kind} {n}->{m}: input gradient off by {err:.2e}\n{got}\nvs\n{reference}"
    # And it must not be trivially zero, or the comparison above is vacuous.
    assert scale > 1e-6, f"{kind} {n}->{m}: the reference gradient is ~0, so nothing was compared"


def test_it_is_the_PRE_update_gradient_and_that_is_not_the_same_number():
    """The definition names the pre-update parameters. That is a real choice, not a formality: the
    post-update gradient is a *different* vector, and a stack differentiating at two different
    parameter points is not doing backprop.

    Asserted by showing the two disagree — otherwise "pre-update" would be untested wording.
    """
    n, m = 6, 3
    a = warm(build("rls", n, m), n, m)
    g = np.random.default_rng(7)
    x, y = g.standard_normal(n), g.standard_normal(m)

    a.update(x, y)
    got = np.asarray(a.input_gradient()).copy()
    after = central_difference(a, x, y)           # same x, y, but the parameters have moved
    assert not np.allclose(got, after, rtol=1e-3), (
        "the pre- and post-update input gradients are indistinguishable on this fixture, so it "
        "cannot tell which one the implementation returns")


@pytest.mark.parametrize("kind", ("rls", "nlms", "mlp1"))
def test_a_refused_update_reports_a_zero_gradient(kind):
    """A level below must not descend on a step its own head refused to take. NaN in, zero out —
    and the control is that a *good* sample immediately afterwards is non-zero again, so this is
    not passing because the buffer is always zero."""
    n, m = 6, 3
    a = warm(build(kind, n, m), n, m)
    report = a.update(np.full(n, np.nan), np.zeros(m))
    assert not report.applied
    assert np.all(np.asarray(a.input_gradient()) == 0.0), f"{kind} leaked a gradient from a "\
                                                          "rejected sample"

    g = np.random.default_rng(3)
    good = a.update(g.standard_normal(n), g.standard_normal(m))
    assert good.applied
    assert np.any(np.asarray(a.input_gradient()) != 0.0), (
        f"{kind} reports zero after an accepted update too — the zero above measured nothing")


def test_the_buffer_is_preallocated_and_the_next_update_overwrites_it():
    """The documented contract, and the trap. `StepReport.prediction` has exactly this shape and
    this repository has walked into it three times — a caller collecting the arrays and reading
    them afterwards gets one value repeated."""
    n, m = 6, 3
    a = warm(build("rls", n, m), n, m)
    g = np.random.default_rng(5)

    first = a.input_gradient()
    a.update(g.standard_normal(n), g.standard_normal(m))
    second = a.input_gradient()
    assert first is second, "a fresh array per call allocates once per step on the hot path"

    kept = np.asarray(a.input_gradient()).copy()
    a.update(g.standard_normal(n), g.standard_normal(m))
    assert not np.allclose(kept, np.asarray(a.input_gradient())), (
        "the buffer did not change across two different updates, so the aliasing above is not "
        "actually being exercised")


def test_an_adapter_that_cannot_differentiate_its_input_says_so():
    """The base declares the method and refuses, rather than leaving it abstract. An adapter with no
    differentiable input is a legitimate thing to write; being unable to *define* it would not be.
    """
    class Opaque(Adapter):
        n_features, n_outputs = 3, 1

        def update(self, x, y): raise NotImplementedError
        def predict(self, x): raise NotImplementedError
        def reset(self): raise NotImplementedError
        def load_state(self, flat): raise NotImplementedError

        @property
        def state(self): raise NotImplementedError

    with pytest.raises(NotImplementedError, match="input gradient"):
        Opaque().input_gradient()
