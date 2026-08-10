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


SD_LARGE = 29.5409       # the target's own std, measured over 4,000 draws


def _large(s, normalised=True):
    v = 50.0 * math.sin(0.5 * s[0]) + 20.0 * s[1] * s[2]
    return v / SD_LARGE if normalised else v


def _run(dtype, hidden, lr, n, normalised=True, seed=1, probe_n=400):
    """A run at `dtype`, started from the SAME weights an f8 run would start from, so a
    difference is the recursion and not two initialisations. Returns
    (RMSE/target-std or inf, step-underflow rate, ||w||)."""
    ref = MultilayerPerceptron(3, 1, hidden=hidden, learning_rate=lr, dtype="f8", seed=seed)
    init = np.array(ref.state, dtype="f8")
    orig = adapt_module.SUPPORTED_DTYPES
    try:
        adapt_module.SUPPORTED_DTYPES = ("f2", "f4", "f8")
        MultilayerPerceptron.min_dtype = "f2"
        m = MultilayerPerceptron(3, 1, hidden=hidden, learning_rate=lr, dtype=dtype,
                                 seed=seed)
        m.load_state(init.astype(dtype))
        g = np.random.default_rng(2001)
        hits = seen = 0
        with np.errstate(all="ignore"):
            for k in range(n):
                s = g.standard_normal(3)
                if k % 100 == 0:
                    before = np.array(m.state)
                    m.update(s, [_large(s, normalised)])
                    moved = np.abs(np.asarray(m.state) - before)
                    spacing = np.spacing(np.abs(before).astype(dtype)).astype("f8")
                    hits += int(np.sum(moved < spacing))
                    seen += moved.size
                else:
                    m.update(s, [_large(s, normalised)])
                if not np.all(np.isfinite(np.asarray(m.state))):
                    return float("inf"), hits / max(seen, 1), float("inf")
        probe = np.random.default_rng(0).standard_normal((probe_n, 3))
        want = np.array([_large(s, normalised) for s in probe])
        got = np.array([float(m.predict(s)[0]) for s in probe])
        rel = float(np.sqrt(np.mean((got - want) ** 2)) / np.std(want))
        return rel, hits / max(seen, 1), float(np.linalg.norm(np.asarray(m.state, "f8")))
    finally:
        adapt_module.SUPPORTED_DTYPES = orig
        MultilayerPerceptron.min_dtype = "f4"


def test_an_unnormalised_target_defeats_every_precision():
    """The finding that outranks the precision question, and the reason the docstring leads
    with it.

    Raw ``50·sin(0.5s₀) + 20·s₁s₂`` drives the weights to ``‖w‖ ≈ 3600``, where the spacing
    at a weight's magnitude exceeds ``lr*grad`` on **96 % of updates in fp64** and the model
    scores 0.73 of a constant predictor. Dividing by the target's own standard deviation
    gives 0.0144 and 0.0 % — **51× better, in double precision**.
    """
    raw_rel, raw_undf, raw_w = _run("f8", (32, 32), 0.01, 30_000, normalised=False)
    nrm_rel, nrm_undf, nrm_w = _run("f8", (32, 32), 0.01, 30_000, normalised=True)

    assert raw_rel > nrm_rel * 10.0, (
        f"normalising bought little ({raw_rel:.4f} to {nrm_rel:.4f}), so this fixture no "
        "longer shows the effect the docstring leads with")
    assert raw_undf > 0.5 and nrm_undf < 0.05, (
        f"underflow {raw_undf:.1%} raw against {nrm_undf:.1%} normalised -- the mechanism "
        "stated in the docstring is not what this fixture is showing")
    assert raw_w > nrm_w * 20.0, f"weights did not grow: {raw_w:.1f} against {nrm_w:.1f}"


def test_fp32_is_indistinguishable_from_fp64():
    """`min_dtype = 'f4'` is a floor, not a compromise. Measured to a million updates in a
    separate sweep (f4/f8 = 1.0000 at 100k, 250k and 500k); reproduced here at a length a
    test can afford, with the same starting weights so only the recursion differs."""
    for hidden, lr in (((16,), 0.002), ((32, 32), 0.02)):
        f4, u4, _ = _run("f4", hidden, lr, 20_000)
        f8, u8, _ = _run("f8", hidden, lr, 20_000)
        assert abs(f4 / f8 - 1.0) < 0.02, (
            f"{hidden} lr={lr}: f4/f8 = {f4 / f8:.4f}; fp32 is no longer the floor and "
            "min_dtype should be re-derived rather than this bound loosened")
        assert u4 < 0.1 and u8 < 0.1, (
            f"underflow {u4:.1%}/{u8:.1%} -- this fixture is not well scaled, so it cannot "
            "say anything about precision")


def test_fp16_stops_learning_where_fp32_keeps_going():
    """A negative result, protected, and **narrower than two earlier versions of this test**.

    The first asserted "4.0x the fp32 error on a small-output target". Its fixture was a raw
    ~0.001 target on which **fp64 itself scores 4.2 times worse than a constant predictor**
    -- nothing learns it, so the comparison was between two failures, and normalised, fp16
    is *better* there. The second asserted a ratio at 20,000 updates, where there is no
    difference at all (1.00x).

    What is real is visible only over time, and as a **rate** rather than a ratio: on a
    well-scaled target at (16,), fp32 halves its error between 50,000 and 100,000 updates
    (0.0907 -> 0.0478) while fp16 moves 4% (0.0853 -> 0.0821). It is not less accurate; it
    has **stopped**, because 67% of its steps are below the spacing at the weight's own
    magnitude against fp32's 0.1%.
    """
    with pytest.raises(ValueError, match="no covariance"):
        MultilayerPerceptron(3, 1, hidden=(4,), dtype="f2")
    with pytest.raises(ValueError, match="positive-definiteness"):
        RecursiveLeastSquares(3, 1, dtype="f2")

    mid4, _, _ = _run("f4", (16,), 0.002, 50_000)
    end4, u4, _ = _run("f4", (16,), 0.002, 100_000)
    mid2, _, _ = _run("f2", (16,), 0.002, 50_000)
    end2, u2, _ = _run("f2", (16,), 0.002, 100_000)

    assert u4 < 0.05, (
        f"fp32 underflowed {u4:.1%}, so this fixture is not well scaled and cannot "
        "attribute anything to the precision")
    assert u2 > 0.4, (
        f"only {u2:.1%} of fp16 steps underflowed; if that is a real improvement the "
        "docstring's table must be re-measured rather than this threshold nudged")
    assert end4 < mid4 * 0.7, (
        f"fp32 did not keep improving ({mid4:.4f} to {end4:.4f}), so there is nothing for "
        "fp16 to fail to keep up with and this fixture proves nothing")
    assert end2 > mid2 * 0.9, (
        f"fp16 kept learning ({mid2:.4f} to {end2:.4f}) -- re-derive min_dtype rather than "
        "keeping a refusal the evidence no longer supports")
    # ...and the mechanism is NOT a NaN or a zeroed weight, which is why the obvious
    # detector is blind: this run finishes finite.
    assert np.isfinite(end2), "fp16 diverged, so this run shows the other failure mode"


def test_fp16_diverges_where_fp32_survives_on_a_badly_scaled_problem():
    """The second, cruder fp16 failure, kept and **labelled**: the target here is raw, and
    a badly-scaled problem is not an indictment of a precision on its own — fp64 fails on it
    too, differently (96 % underflow, 0.73 of a constant). What is fair to say is that fp16
    is strictly less robust: it goes non-finite where fp32, on the same data and the same
    starting weights, does not."""
    f2, _u, _w = _run("f2", (32, 32), 0.02, 2500, normalised=False, seed=1)
    f4, _u, _w = _run("f4", (32, 32), 0.02, 2500, normalised=False, seed=1)
    assert not np.isfinite(f2), "fp16 survived the case it was measured to diverge on"
    assert np.isfinite(f4), (
        "fp32 diverged too, so this shows a hard problem rather than a precision floor")


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


# -- keeping the learning while adding inputs ---------------------------------------------

def _six(s):
    """Needs all six inputs, and the new three are not explained by the old three."""
    return (math.sin(3 * s[0]) + 0.5 * s[1] * s[2]
            + 0.8 * math.tanh(2 * s[3]) - 0.6 * s[4] * s[5])


def test_widening_the_input_moves_the_output_by_at_most_one_ulp():
    """The requirement for a device that is mid-flight when a sensor is installed: the
    prediction must not jump. A zero row means the new inputs contribute nothing whatever
    their values, so the function is preserved *mathematically*.

    **It is not preserved bit-for-bit, and the first version of this test asserted that it
    was.** numpy groups a sum of six differently from a sum of three, and three of the terms
    being exactly zero does not restore the order of the other three. Declared divergence 3,
    in a new place.

    **Measuring it took the correction this repository has now made four times.** Against
    each probe's own value the worst reads **367 ulp** at (8,) -- which is the prediction
    passing near zero, not the implementations differing. Against the run's scale, which is
    the rule already applied to the state block, the a posteriori residual and the pipeline's
    prediction, it is **0.0 to 2.3 ulp** over five architectures: exactly 0 with no hidden
    layer, 0.8-2.3 with one to three.

    The contrast that matters is unaffected: filling the new rows randomly instead moves the
    output by **2.0 to 4.6** on a target whose own spread is 1.4.
    """
    old = MultilayerPerceptron(3, 2, hidden=(8, 5), learning_rate=0.02, dtype="f8", seed=1)
    g = np.random.default_rng(4)
    for _ in range(400):
        s = g.standard_normal(3)
        old.update(s, [float(np.sin(2 * s[0])), float(s[1] * s[2])])

    wide = old.widened(6)
    assert wide.state.size == old.state.size + 3 * 8
    assert wide.n_features == 6 and wide.hidden == (8, 5) and wide.n_outputs == 2

    probe = np.random.default_rng(3).standard_normal((200, 6)) * 5.0   # new inputs LARGE
    a = np.array([np.asarray(old.predict(x[:3])) for x in probe])
    b = np.array([np.asarray(wide.predict(x)) for x in probe])
    ulp = float(np.finfo(np.float64).eps)
    run_scale = max(float(np.max(np.abs(a))), 1e-30)
    worst = float(np.max(np.abs(a - b))) / run_scale / ulp
    assert worst < 8.0, (
        f"widening moved the output by {worst:.1f} ulp of the run's scale (measured 0.0-2.3 "
        "over five architectures); the new rows are not zero, or something other than the "
        "summation grouping changed")
    # The per-value metric is deliberately NOT used and the reason is asserted: it reads up
    # to 367 ulp on a prediction that passes near zero, which measures the cancellation
    # rather than the two networks.
    per_value = max(float(np.max(np.abs(p - q))) / max(float(np.max(np.abs(p))), 1e-30)
                    for p, q in zip(a, b)) / ulp
    assert per_value > worst, (
        "the per-value metric is no longer inflated by cancellation here, so this fixture "
        "does not demonstrate why the run's scale is the right denominator")
    # ...and the new rows really are zero, so the bound above is not passing for the wrong
    # reason -- a tiny random fill would also land inside a loose tolerance.
    assert np.all(np.asarray(wide.state)[:6 * 8].reshape(6, 8)[3:, :] == 0.0)

    # ...and everything after the first layer must be carried across untouched, or the
    # equality above could hold for a network that had lost its later layers.
    assert np.array_equal(np.asarray(old.state)[3 * 8:], np.asarray(wide.state)[6 * 8:])


def test_the_new_input_rows_learn_from_the_first_update():
    """Zeros that cannot move would be the `reset()` trap again. They do move here, because
    the gradient into a new input row is `x_new * delta` and `delta` comes from a trained
    downstream -- which is exactly what a new hidden UNIT does not have."""
    old = MultilayerPerceptron(3, 1, hidden=(8,), learning_rate=0.02, dtype="f8", seed=1)
    g = np.random.default_rng(4)
    for _ in range(400):
        s = g.standard_normal(3)
        old.update(s, [float(np.sin(2 * s[0]))])

    wide = old.widened(6)
    before = np.array(wide.state)
    s = g.standard_normal(6)
    wide.update(s, [_six(s)])
    after = np.asarray(wide.state)
    new_rows_before = before[:6 * 8].reshape(6, 8)[3:, :]
    new_rows_after = after[:6 * 8].reshape(6, 8)[3:, :]
    assert np.all(new_rows_before == 0.0)
    assert float(np.max(np.abs(new_rows_after))) > 1e-6, (
        "the new input rows did not move on the first update, so they are dead weight")


def test_grafting_beats_starting_over_when_a_sensor_arrives():
    """Measured, three inputs becoming six on a truth needing all six: RMSE over the whole
    input space is 0.915 grafted against 1.134 fresh at 100 updates, 0.651 against 1.004 at
    2,000 and 0.383 against 0.559 at 30,000 -- the advantage does not close. Shortened here,
    and the assertion is on the *early* separation, which is what "the learning survived"
    means."""
    def build():
        m = MultilayerPerceptron(3, 1, hidden=(16,), learning_rate=0.02, dtype="f8", seed=2)
        g = np.random.default_rng(502)
        for _ in range(10000):
            s = g.standard_normal(6)
            s[3:] = 0.0
            m.update(s[:3], [_six(s)])
        return m, g

    old, g = build()
    grafted = old.widened(6)
    fresh = MultilayerPerceptron(6, 1, hidden=(16,), learning_rate=0.02, dtype="f8", seed=9)

    probe = np.random.default_rng(0).standard_normal((600, 6))
    want = np.array([_six(s) for s in probe])

    def rmse(m):
        got = np.array([float(m.predict(s)[0]) for s in probe])
        return float(np.sqrt(np.mean((got - want) ** 2)))

    # 1,500 rather than 400: the ratio is NOT monotone in the number of post-widening
    # updates -- 0.819 at 200, 0.871 at 400, 0.719 at 1500 -- so 400 lands in a transient
    # and would pin a bounce rather than the effect. At 1,500 it is 0.63-0.72 across every
    # warm-up length tried.
    gg = np.random.default_rng(77)
    for _ in range(1500):
        s = gg.standard_normal(6)
        y = [_six(s)]
        grafted.update(s, y)
        fresh.update(s, y)

    assert rmse(grafted) < rmse(fresh) * 0.85, (
        f"the graft did not beat starting over ({rmse(grafted):.4f} against "
        f"{rmse(fresh):.4f}), so nothing here shows the learning being kept")
    # A control: without it, a fresh network that simply failed would pass the above.
    assert rmse(fresh) < float(np.std(want)) * 2.0


def test_deepening_is_refused_with_the_measurement_as_the_reason():
    """Not a missing feature. Extra hidden units were measured to buy nothing on any target
    that could be constructed -- h=8 already reaches the ceiling and h=64 does not beat it,
    because SGD's convergence binds before capacity does."""
    m = MultilayerPerceptron(3, 1, hidden=(8,), dtype="f8")
    with pytest.raises(NotImplementedError, match="measured to buy nothing"):
        m.deepened((16,))
    with pytest.raises(ValueError, match="only appends"):
        m.widened(2)


def test_the_widened_network_still_satisfies_the_contract():
    """A widened adapter is an adapter: `persist` must take it, a `Guard` must wrap it, and
    the covariance fields must still be refused."""
    old = MultilayerPerceptron(3, 1, hidden=(8,), dtype="f8", seed=1)
    wide = old.widened(5)
    rec = pack(wide, sequence=1, fingerprint=0xBEEF)
    assert unpack(rec).state_size == wide.state.size
    g = Guard(wide, [Bound("residual_norm", hi=50.0)], rollback_depth=2)
    assert g.n_features == 5
    with pytest.raises(ValueError):
        Guard(old.widened(5), [Bound("p_trace", hi=1e6)])
