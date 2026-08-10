"""`MultilayerPerceptron`: the adapter the platform-generates-the-basis decision requires.

The accuracy claims live in `test_model_class.py`, against a fixture deliberately *outside*
the library — a contract that only fits adapters shipped with it is not a contract. What is
tested here is that this implementation is right, which is a different question: **a backward
pass that is subtly wrong still trains, just worse**, so "it learns" is not evidence.
"""
import gc
import math
import sys

import numpy as np
import pytest

import argmind.adapt as adapt_module
from argmind.adapt import MultilayerPerceptron, RecursiveLeastSquares
from argmind.guard import Bound, Divergence, Guard
from argmind.persist import pack, unpack


# -- correctness of the implementation, before anything about accuracy ------------------

@pytest.mark.parametrize("hidden,activation", [
    ((), "tanh"), ((5,), "tanh"), ((4, 3), "tanh"), ((4, 3), "relu"), ((6, 5, 4), "tanh"),
])
def test_the_gradient_matches_central_differences(hidden, activation):
    """The primary correctness test. `learning_rate` is set to 1e-9 so the step *is* the
    gradient scaled, and the whole flat block is differenced element by element."""
    m = MultilayerPerceptron(3, 2, hidden=hidden, activation=activation,
                             learning_rate=1e-9, dtype="f8", seed=3)
    g = np.random.default_rng(1)
    x, y = g.standard_normal(3), g.standard_normal(2)
    w0 = np.array(m.state)

    def loss(w):
        m.load_state(w)
        return float(np.sum((y - m.predict(x)) ** 2))

    m.load_state(w0)
    m.update(x, y)
    taken = (w0 - np.array(m.state)) / 1e-9

    h, num = 1e-6, np.zeros_like(w0)
    for i in range(w0.size):
        p, q = w0.copy(), w0.copy()
        p[i] += h
        q[i] -= h
        num[i] = (loss(p) - loss(q)) / (2 * h)

    rel = np.linalg.norm(taken - num) / np.linalg.norm(num)
    assert rel < 1e-6, f"analytic gradient disagrees with finite differences: {rel:.3e}"
    assert np.linalg.norm(num) > 1e-6, (
        "the numerical gradient is ~zero, so this comparison would pass for an "
        "implementation that does nothing at all")


def test_the_exact_law_with_no_hidden_layer():
    """A single linear layer makes the update closed-form, so this is an equality.

    `e_post == (1 - 2*lr*(x.x + 1)) * e`. The `+ 1` is the bias, which acts as a constant
    input of one — the first version of this law omitted it, was wrong by a factor of 20,
    and that is why the law is worth having: one tight enough to catch a missing term.
    """
    worst = 0.0
    for lr in (0.001, 0.01, 0.1):
        for t in range(60):
            m = MultilayerPerceptron(4, 2, hidden=(), learning_rate=lr, dtype="f8", seed=t)
            g = np.random.default_rng(1000 + t)
            x, y = g.standard_normal(4), g.standard_normal(2)
            r = m.update(x, y)
            want = (1.0 - 2.0 * lr * (float(x @ x) + 1.0)) * r.residual
            worst = max(worst, float(np.max(np.abs(r.residual_post - want))
                                     / np.max(np.abs(want))))
    assert worst < 1e-12, f"the law is not exact: worst relative deviation {worst:.3e}"

    # ...and the law without the bias term must NOT hold, or it discriminates nothing.
    m = MultilayerPerceptron(4, 1, hidden=(), learning_rate=0.01, dtype="f8", seed=1)
    g = np.random.default_rng(7)
    x, y = g.standard_normal(4), g.standard_normal(1)
    r = m.update(x, y)
    naive = (1.0 - 2.0 * 0.01 * float(x @ x)) * r.residual
    assert abs(float(r.residual_post[0] - naive[0])) > 1e-6 * abs(float(naive[0]))


# -- the Adapter contract ----------------------------------------------------------------

def test_the_state_block_is_flat_and_persist_needs_no_special_case():
    m = MultilayerPerceptron(3, 2, hidden=(4, 5), dtype="f8", seed=1)
    want = (3 * 4 + 4) + (4 * 5 + 5) + (5 * 2 + 2)
    assert m.state.size == m.state_size == want

    rec = pack(m, sequence=1, fingerprint=0xABCD)
    back = unpack(rec)
    assert back.state_size == want and np.array_equal(back.state, m.state)

    m.update([1.0, 2.0, 3.0], [0.5, -0.5])
    assert not np.array_equal(np.asarray(m.state), back.state), (
        "the weights are a copy rather than a view into the state block, so `state` can "
        "drift from what the adapter is actually using")

    m2 = MultilayerPerceptron(3, 2, hidden=(4, 5), dtype="f8", seed=99)
    m2.load_state(m.state)
    assert np.allclose(m2.predict([0.3, -1.0, 2.0]), m.predict([0.3, -1.0, 2.0]))


def test_state_is_a_standing_view_and_not_a_fresh_array():
    """`state` must be one standing read-only view, not a copy per call.

    Equality cannot catch a copy -- it holds the same numbers. Nor can counting live
    allocation blocks around bare `m.state` calls: **the copy is discarded immediately, so
    the counter returns to baseline**. That was the first version of this test and it let
    the mutant through. Two things do catch it: whether the array shares memory with the
    block, and counting blocks while the results are *retained*.

    This exact defect has been in this repository before -- the memory work found
    `Adapter.state` building a fresh view per call, on the path the guard takes at every
    checkpoint.
    """
    m = MultilayerPerceptron(3, 1, hidden=(8,), dtype="f8", seed=1)
    a, b = m.state, m.state
    assert a is b, "`state` returns a different object each call"
    assert np.shares_memory(a, np.asarray(m.state)), "`state` does not alias the block"
    assert not a.flags.writeable, "`state` is writable, so a caller can corrupt the model"

    m.update([1.0, 2.0, 3.0], [0.5])
    assert np.array_equal(np.asarray(a), np.asarray(m.state)), (
        "the view held from before the update did not see it, so `state` is a snapshot")

    gc.collect()
    before = sys.getallocatedblocks()
    keep = [m.state for _ in range(2000)]        # RETAINED: a copy cannot hide here
    grew = sys.getallocatedblocks() - before
    assert grew < 100, f"2000 retained `state` results cost {grew} blocks; it is copying"

    gc.collect()
    before = sys.getallocatedblocks()
    keep = [np.array(m.state) for _ in range(2000)]     # positive control
    assert sys.getallocatedblocks() - before > 1000, (
        "2000 deliberate copies did not register, so this counter cannot see an allocation "
        "and the check above proves nothing")
    del keep


def test_gain_norm_is_the_step_taken_and_not_the_gradient():
    """Exact, because it can be: the step is `-lr * grad`, so `||delta state||` is
    `lr * ||grad||`, which is what `gain_norm` reports. NLMS reports the gain *vector* for
    the same reason — a `Guard` bound is written against the name, so the quantity has to
    mean the same thing whichever adapter is underneath, and reporting the raw gradient
    would make it `1/lr` times larger."""
    for lr in (0.001, 0.05):
        m = MultilayerPerceptron(4, 2, hidden=(6,), learning_rate=lr, dtype="f8", seed=2)
        g = np.random.default_rng(11)
        for _ in range(30):
            x, y = g.standard_normal(4), g.standard_normal(2)
            before = np.array(m.state)
            r = m.update(x, y)
            moved = float(np.linalg.norm(np.asarray(m.state) - before))
            assert abs(r.gain_norm - moved) <= 1e-12 * max(moved, 1e-30), (
                f"gain_norm {r.gain_norm:.6e} is not the distance the state moved "
                f"({moved:.6e})")
            assert moved > 0.0, "nothing moved, so the comparison is 0 == 0"


def test_the_guard_takes_it_and_refuses_the_field_it_cannot_report():
    m = MultilayerPerceptron(3, 1, hidden=(8,), dtype="f8")
    assert m.health_fields == {"residual_norm", "residual_post_norm", "gain_norm"}
    g = Guard(m, [Bound("residual_norm", hi=50.0)], rollback_depth=4)
    assert g._ckpt.shape == (4, m.state_size)
    with pytest.raises(ValueError):
        Guard(MultilayerPerceptron(3, 1, hidden=(8,), dtype="f8"),
              [Bound("p_trace", hi=1e6)])
    Guard(RecursiveLeastSquares(3, 1, dtype="f8"), [Bound("p_trace", hi=1e6)])  # control


def test_bad_values_are_rejected_and_counted_and_shapes_raise():
    m = MultilayerPerceptron(3, 1, hidden=(4,), dtype="f8", seed=1)
    before = np.array(m.state)
    r = m.update([1.0, float("nan"), 3.0], [0.5])
    assert not r.applied and r.reason == "nonfinite_input"
    assert m.rejected_nonfinite == 1 and np.array_equal(np.asarray(m.state), before)
    r = m.update([1.0, 2.0, 3.0], [float("inf")])
    assert not r.applied and m.rejected_nonfinite == 2

    with pytest.raises(ValueError):
        m.update([1.0, 2.0], [0.5])
    with pytest.raises(ValueError):
        m.update([1.0, 2.0, 3.0], [0.5, 0.5])


def test_the_constructor_refuses_what_it_cannot_do():
    for kw in ({"hidden": (4, 0)}, {"activation": "sigmoid"}, {"learning_rate": 0.0},
               {"learning_rate": -1.0}, {"learning_rate": float("inf")}):
        with pytest.raises(ValueError):
            MultilayerPerceptron(3, 1, dtype="f8", **kw)
    with pytest.raises(ValueError):
        MultilayerPerceptron(0, 1, dtype="f8")
    with pytest.raises(ValueError):
        MultilayerPerceptron(3, 0, dtype="f8")
    # activation is keyword-only along with everything after n_outputs
    with pytest.raises(TypeError):
        MultilayerPerceptron(3, 1, (4,))


def test_reset_stops_the_network_rather_than_restarting_it():
    """Documented and surprising, so pinned: `reset` zeroes the weights, and a network of
    zero weights has every hidden unit computing the same function with the same gradient.
    It cannot learn out of that. RLS's zero theta is a fresh start; this is not."""
    m = MultilayerPerceptron(3, 1, hidden=(8,), learning_rate=0.05, dtype="f8", seed=1)
    g = np.random.default_rng(2)
    for _ in range(500):
        s = g.standard_normal(3)
        m.update(s, [float(s[0] * 2 - s[1])])
    m.reset()
    assert np.all(np.asarray(m.state) == 0.0) and m.updates == 0
    for _ in range(2000):
        s = g.standard_normal(3)
        m.update(s, [float(s[0] * 2 - s[1])])
    hidden_w = np.asarray(m.state)[:24].reshape(3, 8)
    assert np.allclose(hidden_w, hidden_w[:, :1]), (
        "the hidden units differentiated after a reset, so symmetry is not permanent and "
        "the docstring's warning is wrong")


# -- the two choices that were measured rather than taken on taste ------------------------

def test_relu_is_offered_but_is_not_the_default_and_the_measurement_says_why():
    """`relu` reaches a better optimum than `tanh` and is still not the default.

    Measured at (16,), 8,000 samples, **per seed** over six, against a constant predictor's
    0.7641:

    ==========  ========  ==================================  ================
    lr             0.2                    0.1                       0.05
    ==========  ========  ==================================  ================
    ``tanh``    all div    0.751 on all six, below 0.764       0.458 - 0.630
    ``relu``    all div    0.772 - 0.831 and one divergence     0.592 - 0.715
    ==========  ========  ==================================  ================

    At ``lr=0.1`` ``tanh`` learns something on every seed and ``relu`` learns **nothing** on
    every seed, plus one outright divergence. At 0.05 ``tanh`` wins on all six. At 0.2 both
    die, so ``tanh`` is not being claimed unconditionally safe. A separate 40,000-sample run
    at (32,32) has ``relu`` at 0.1391 against ``tanh``'s 0.2315, so its advantage is real
    where a caller can afford to find it.

    **Two earlier versions of this table were aggregation artifacts** and are why this one
    is per-seed: a mean over three seeds hid that ``relu`` at 0.02 was fine, and marking a
    row "diverged" when *any* seed diverged turned one seed in six into a property.
    """
    assert MultilayerPerceptron(3, 1, dtype="f8").activation == "tanh"
    CONSTANT = 0.7641

    def final(act, lr, seed, n=8000):
        m = MultilayerPerceptron(3, 1, hidden=(16,), activation=act,
                                 learning_rate=lr, dtype="f8", seed=seed)
        r = np.random.default_rng(5)
        with np.errstate(all="ignore"):        # divergence is the subject, not an accident
            for _ in range(n):
                s = r.standard_normal(3)
                m.update(s, [math.sin(3 * s[0]) * math.exp(-s[1] ** 2) + abs(s[2]) - 0.4])
        if not np.all(np.isfinite(np.asarray(m.state))):
            return float("inf")
        p = np.random.default_rng(0).standard_normal((400, 3))
        return float(np.sqrt(np.mean(
            [(m.predict(s)[0] - (math.sin(3 * s[0]) * math.exp(-s[1] ** 2)
                                 + abs(s[2]) - 0.4)) ** 2 for s in p])))

    SEEDS = (1, 2, 3)
    tanh_01 = [final("tanh", 0.1, s) for s in SEEDS]
    relu_01 = [final("relu", 0.1, s) for s in SEEDS]
    assert all(v < CONSTANT for v in tanh_01), (
        f"tanh failed to beat a constant at lr=0.1 ({tanh_01}), so the contrast is not "
        "about relu and this fixture must be rebuilt")
    assert all(v >= CONSTANT for v in relu_01), (
        f"relu learned at lr=0.1 on some seed ({relu_01}); the docstring's table is stale "
        "and should be re-measured rather than the threshold nudged")

    # ...and tanh is not claimed unconditionally safe.
    assert all(final(a, 0.2, s) == float("inf") for a in ("tanh", "relu") for s in (1, 2)), (
        "something survived lr=0.2, so 'both die at 0.2' overstates the caveat")


def test_fp16_is_refused_because_it_was_measured_to_fail():
    """A negative result, protected so a later change cannot quietly reverse it.

    An earlier version of this test asserted the opposite -- that fp16 *works* and is
    refused only for want of evidence -- and it passed, because it ran the one
    configuration where fp16 is fine. A sweep of 216 runs (four targets, three
    architectures, two rates, three seeds, 100,000 updates) found that configuration to be
    the most favourable of twenty-four.

    Reproduced here in 2,500 updates: on a large-output target at (32,32) the fp16 weights
    go non-finite and the fp32 control on the same data and the same starting weights does
    not. Both halves matter -- without the control this would only show that the target is
    hard.
    """
    with pytest.raises(ValueError, match="no covariance"):
        MultilayerPerceptron(3, 1, hidden=(4,), dtype="f2")
    with pytest.raises(ValueError, match="positive-definiteness"):
        RecursiveLeastSquares(3, 1, dtype="f2")

    def truth(s):
        return 50.0 * math.sin(0.5 * s[0]) + 20.0 * s[1] * s[2]

    def died_at(dtype, seed=2, n=2500):
        ref = MultilayerPerceptron(3, 1, hidden=(32, 32), learning_rate=0.02,
                                   dtype="f8", seed=seed)
        init = np.array(ref.state, dtype="f8")
        orig = adapt_module.SUPPORTED_DTYPES
        try:
            adapt_module.SUPPORTED_DTYPES = ("f2", "f4", "f8")
            MultilayerPerceptron.min_dtype = "f2"
            m = MultilayerPerceptron(3, 1, hidden=(32, 32), learning_rate=0.02,
                                     dtype=dtype, seed=seed)
            m.load_state(init.astype(dtype))
            g = np.random.default_rng(1002)
            with np.errstate(all="ignore"):
                for k in range(n):
                    s = g.standard_normal(3)
                    m.update(s, [truth(s)])
                    if not np.all(np.isfinite(np.asarray(m.state))):
                        return k
            return None
        finally:
            adapt_module.SUPPORTED_DTYPES = orig
            MultilayerPerceptron.min_dtype = "f4"

    assert died_at("f2") is not None, (
        "fp16 survived the case it was measured to fail; if that is a real improvement the "
        "docstring's sweep must be re-run rather than this threshold nudged")
    assert died_at("f4") is None, (
        "fp32 died too, so this fixture shows a hard target rather than a precision floor")


def test_the_fp16_failure_is_step_underflow_and_not_a_nan():
    """The mechanism, because the obvious detector cannot see it.

    Nothing is driven to zero -- 0.0% of weights in every run of the sweep. What happens is
    that `lr*grad` falls below the fp16 spacing *at that weight's magnitude*, so the weight
    does not move: up to 92% of updates on a small-output target. **A network can stop
    learning in half precision without ever producing a NaN**, and a test looking for NaNs
    or zeros would report success. Measured here on the small-output case.
    """
    def truth(s):
        return 0.001 * (math.sin(3 * s[0]) + s[1] * s[2])

    def underflow_rate(dtype, n=1500):
        orig = adapt_module.SUPPORTED_DTYPES
        try:
            adapt_module.SUPPORTED_DTYPES = ("f2", "f4", "f8")
            MultilayerPerceptron.min_dtype = "f2"
            m = MultilayerPerceptron(3, 1, hidden=(32, 32), learning_rate=0.002,
                                     dtype=dtype, seed=1)
            g = np.random.default_rng(1001)
            hits = seen = 0
            with np.errstate(all="ignore"):
                for k in range(n):
                    s = g.standard_normal(3)
                    before = np.array(m.state)
                    m.update(s, [truth(s)])
                    if k % 50 == 0:
                        moved = np.abs(np.asarray(m.state) - before)
                        spacing = np.spacing(np.abs(before).astype(dtype)).astype("f8")
                        hits += int(np.sum(moved < spacing))
                        seen += moved.size
                zeros = float(np.mean(np.asarray(m.state) == 0.0))
            return hits / max(seen, 1), zeros
        finally:
            adapt_module.SUPPORTED_DTYPES = orig
            MultilayerPerceptron.min_dtype = "f4"

    half, half_zeros = underflow_rate("f2")
    single, _ = underflow_rate("f4")
    assert half > 0.4, (
        f"only {half:.1%} of fp16 steps underflowed, so this fixture no longer shows the "
        "mechanism the docstring describes")
    assert single < half / 2.0, (
        f"fp32 underflowed at {single:.1%} against fp16's {half:.1%} -- if the two are "
        "comparable this measures the problem rather than the precision")
    assert half_zeros == 0.0, (
        "fp16 drove weights to zero after all, so a zero-check WOULD have caught it and "
        "the claim that the obvious detector is blind is wrong")


# -- what is deliberately NOT clamped ------------------------------------------------------

def test_a_divergence_is_not_clamped_inside_the_adapter_and_the_guard_catches_it():
    """Following the rule RLS already sets by reporting covariance windup rather than
    bounding it: a safety mechanism inside the adapter makes the `Guard` optional, and a
    guard that can be left out gets left out."""
    def drive(guarded):
        m = MultilayerPerceptron(3, 1, hidden=(16,), activation="relu",
                                 learning_rate=0.5, dtype="f8", seed=1)
        a = Guard(m, [Divergence(ratio=4.0, floor=0.05, warmup=50)],
                  rollback_depth=4, checkpoint_every=10) if guarded else m
        r = np.random.default_rng(5)
        with np.errstate(all="ignore"):   # overflow IS the behaviour under test
            for k in range(1, 3001):
                s = r.standard_normal(3) * 3.0
                rep = a.update(s, [float(s[0] * s[1] * s[2])])
                if guarded and rep.frozen:
                    return k, np.all(np.isfinite(np.asarray(a.state)))
        return None, np.all(np.isfinite(np.asarray(a.state)))

    step, finite = drive(False)
    assert step is None and not finite, (
        "the unguarded adapter stayed finite, so this fixture does not drive a divergence "
        "and cannot show that the adapter declines to clamp it")

    step, finite = drive(True)
    assert step is not None, "the guard did not catch a divergence that destroyed the model"
    assert finite, "the guard froze but the restored state is not finite"
