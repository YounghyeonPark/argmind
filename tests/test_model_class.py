"""The framework must hold a model that learns its own basis.

The scope decision of 2026-08-10 — *the platform generates the basis* — rests on a claim about
this repository rather than about machine learning in general: that `Guard`, `persist` and the
`Adapter` contract accept a backpropagating network **without being changed for it**. That claim
is load-bearing, so it is tested rather than asserted, and the network here is deliberately not
part of the library: if the contract only fits adapters that ship with it, it is not a contract.

`Mlp` below is a fixture. It is the smallest thing that is honestly a network — layers, a
backward pass, weights rather than a covariance — and every assertion about it is one the
framework would have to satisfy for any such model.
"""
import numpy as np
import pytest

from argmind.adapt import Adapter, RecursiveLeastSquares, UpdateReport
from argmind.guard import Bound, Guard
from argmind.persist import pack, unpack


class Mlp(Adapter):
    """A dense tanh network trained by SGD, wearing the `Adapter` contract.

    State is every weight and bias in one flat contiguous block in layer order, which is the
    same promise `Rls` makes and the reason `persist` needs no special case.

    `health_fields` is the residual three and there is no `health()`: **a backpropagated weight
    has no cheap per-direction confidence**, so `p_trace` is the guard signal this model class
    gives up. That is a property of the class, not a gap in the fixture.
    """

    min_dtype = "f4"
    health_fields = frozenset({"residual_norm", "residual_post_norm", "gain_norm"})

    def __init__(self, n_features, n_outputs=1, hidden=(32, 32), lr=0.02, dtype="f8", seed=1):
        self.n_features, self.n_outputs = int(n_features), int(n_outputs)
        self.dtype, self.lr = np.dtype(dtype), float(lr)
        widths = [self.n_features, *hidden, self.n_outputs]
        self._shapes = list(zip(widths[:-1], widths[1:]))
        self._state = np.zeros(sum(a * b + b for a, b in self._shapes), dtype=self.dtype)
        self._state_ro = self._state.view()
        self._state_ro.flags.writeable = False
        g, off = np.random.default_rng(seed), 0
        for a, b in self._shapes:                       # Xavier, so a bad init is not the test
            self._state[off:off + a * b] = (
                g.standard_normal((a, b)) * np.sqrt(2.0 / (a + b))).ravel()
            off += a * b + b
        self._e = np.zeros(self.n_outputs, dtype=self.dtype)
        self._e_post = np.zeros(self.n_outputs, dtype=self.dtype)

    def _layers(self):
        off = 0
        for a, b in self._shapes:
            W = self._state[off:off + a * b].reshape(a, b)
            off += a * b
            yield W, self._state[off:off + b]
            off += b

    @property
    def state(self):
        return self._state_ro

    def load_state(self, flat):
        arr = np.asarray(flat, dtype=self.dtype).reshape(-1)
        if arr.size != self._state.size:
            raise ValueError(f"state block must have {self._state.size} elements")
        if not np.all(np.isfinite(arr)):
            raise ValueError("refusing a state block that is not finite")
        self._state[:] = arr

    def reset(self):
        self._state[:] = 0

    def _forward(self, x):
        acts, layers = [x], list(self._layers())
        for i, (W, b) in enumerate(layers):
            z = acts[-1] @ W + b
            acts.append(np.tanh(z) if i < len(layers) - 1 else z)
        return acts

    def predict(self, x):
        return self._forward(np.asarray(x, dtype=self.dtype).reshape(-1))[-1]

    def update(self, x, y):
        x = np.asarray(x, dtype=self.dtype).reshape(-1)
        y = np.asarray(y, dtype=self.dtype).reshape(-1)
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
            self._e[:] = self._e_post[:] = 0
            return self._report(False, "nonfinite_input", 0.0)

        acts = self._forward(x)
        self._e[:] = y - acts[-1]
        layers = list(self._layers())
        delta, grads = -2.0 * self._e, []
        for i in range(len(layers) - 1, -1, -1):
            grads.append((np.outer(acts[i], delta), delta.copy()))
            if i:
                delta = (delta @ layers[i][0].T) * (1.0 - acts[i] ** 2)   # tanh'
        grads.reverse()
        gn = 0.0
        for (W, b), (gw, gb) in zip(layers, grads):
            W -= self.dtype.type(self.lr) * gw          # writes through into the flat block
            b -= self.dtype.type(self.lr) * gb
            gn += float(np.sum(gw * gw) + np.sum(gb * gb))
        self._e_post[:] = y - self._forward(x)[-1]
        return self._report(True, None, float(np.sqrt(gn)) * self.lr)

    def _report(self, applied, reason, gain):
        return UpdateReport(applied=applied, residual=self._e.copy(),
                            residual_post=self._e_post.copy(), gain_norm=gain,
                            p_trace=0.0, p_min_diag=0.0, reason=reason)


def _truth(s):
    """Deliberately NOT polynomial, so a richer fixed basis is not a way out either."""
    return np.sin(3 * s[0]) * np.exp(-s[1] ** 2) + np.abs(s[2]) - 0.4


@pytest.fixture(scope="module")
def probe():
    rng = np.random.default_rng(0)
    pts = rng.standard_normal((2000, 3))
    return pts, np.array([_truth(s) for s in pts])


def _rmse(f, probe):
    pts, want = probe
    return float(np.sqrt(np.mean([(f(s) - _truth(s)) ** 2 for s in pts])))


def test_the_flat_state_block_makes_persist_work_unchanged():
    net = Mlp(3, 1)
    rec = pack(net, sequence=1, fingerprint=0xABCD)
    back = unpack(rec)
    assert back.state_size == net.state.size == 1217
    assert np.array_equal(back.state, net.state)
    # ...and the block is genuinely the weights, not a copy that drifted from them.
    net.update([1.0, 2.0, 3.0], [0.5])
    assert not np.array_equal(np.asarray(net.state), back.state)


def test_the_guard_takes_a_network_and_refuses_the_field_it_cannot_report():
    net = Mlp(3, 1)
    g = Guard(net, [Bound("residual_norm", hi=50.0)], rollback_depth=4, checkpoint_every=25)
    assert g._ckpt.shape == (4, net.state.size), "the ring is Depth x state, model-sized"
    with pytest.raises(ValueError):
        Guard(Mlp(3, 1), [Bound("p_trace", hi=1e6)])
    # An RLS control, so this is about the adapter's declaration and not about the bound.
    Guard(RecursiveLeastSquares(3, 1, dtype="f8"), [Bound("p_trace", hi=1e6)])


def test_a_network_learns_what_no_linear_basis_can(probe):
    """The measurement the scope decision rests on.

    A linear basis on this target scores 0.7640 against a constant predictor's 0.7641 — it
    learns *nothing*, which is why the platform generating its own basis is worth the cost.
    Thresholds are set where 5,000 updates put the network (measured 0.3873), far from both
    the linear ceiling and any tolerance that could be met by accident.
    """
    pts, want = probe
    constant = float(np.std(want))
    assert 0.76 < constant < 0.77

    lin = RecursiveLeastSquares(4, 1, forgetting=1.0, p0=100.0, dtype="f8")
    net = Mlp(3, 1, hidden=(32, 32), lr=0.02, dtype="f8", seed=1)
    g = np.random.default_rng(5)
    for _ in range(5000):
        s = g.standard_normal(3)
        y = [_truth(s)]
        lin.update([s[0], s[1], s[2], 1.0], y)
        net.update(s, y)

    e_lin = _rmse(lambda s: lin.predict([s[0], s[1], s[2], 1.0])[0], probe)
    e_net = _rmse(lambda s: net.predict(s)[0], probe)
    assert e_lin > constant * 0.98, (
        f"the linear basis learned something ({e_lin:.4f} against a constant's {constant:.4f}), "
        "so this target does not demonstrate the ceiling and the whole comparison is void")
    assert e_net < 0.45, f"the network did not beat the linear ceiling: {e_net:.4f}"
    assert e_net < e_lin / 1.5


def test_the_residual_does_not_substitute_for_the_covariance_signal():
    """The gap this model class arrives with, pinned so it is not quietly assumed away.

    A predecessor of this test asserted that SGD *got worse with more data* (0.4921 at 20k,
    0.5340 at 60k) and passed, because both figures are deterministic under fixed seeds. It
    was pinning **two points out of a flat noisy series**: over ten runs the slope is positive
    in one — that seed — and mean −0.0250 per 10k against a within-run scatter of 0.0216. A
    deterministic test can pin noise, and a passing test is not evidence that what it describes
    is real.

    What is real, and is what that test was reaching for: `p_trace` warns that RLS is losing
    confidence, a backpropagated weight has no equivalent, and **the residual does not
    substitute**. On a misspecified fit the smoothed residual and held-out error move in
    opposite directions — RLS on a linear basis over a nonlinear truth gives EWMA range 1.518×
    against a held-out range of 1.002×, correlation −0.287, while a well-specified control gives
    +0.895. Misspecification is the normal condition for an adapted head.
    """
    net = Mlp(3, 1)
    assert net.health_fields.isdisjoint({"p_trace", "p_min_diag"}), (
        "if this adapter reports a covariance, the gap this test describes is gone and the "
        "claim in DESIGN.md is stale")

    # The measurement, at a scale that runs in a test: a misspecified RLS, where the residual
    # is *provably* the wrong proxy because the held-out error cannot move at all.
    true = np.array([2.0, -1.5, 0.5, 0.3])
    probe_x = np.random.default_rng(0).standard_normal((500, 4))
    probe_x[:, 3] = 1.0
    m = RecursiveLeastSquares(4, 1, forgetting=1.0, p0=100.0, dtype="f8")
    r = np.random.default_rng(5)
    ew, rows = 0.0, []
    for k in range(1, 20001):
        s = r.standard_normal(3)
        x = np.array([s[0], s[1], s[2], 1.0])
        e = float(np.linalg.norm(m.update(x, [_truth(s)]).residual))
        ew = e if k == 1 else ew + 1e-4 * (e - ew)
        if k % 2000 == 0:
            p = probe_x @ np.asarray(m.state)[:4]
            rows.append((ew, float(np.sqrt(np.mean((p - probe_x @ true) ** 2)))))

    ews = np.array([a for a, _ in rows])
    rms = np.array([b for _, b in rows])
    assert ews.max() / ews.min() > 1.3, (
        f"the smoothed residual barely moved ({ews.max()/ews.min():.3f}x), so this fixture no "
        "longer shows it wandering and cannot demonstrate that it is a bad proxy")
    assert rms.max() / rms.min() < 1.05, (
        f"the held-out error moved {rms.max()/rms.min():.3f}x, so the residual has something "
        "real to track here and the comparison is not the one this test claims to make")
    assert np.corrcoef(ews, rms)[0, 1] < 0.5, "the residual tracked held-out error after all"


def test_no_existing_check_can_see_a_slow_loss_of_quality():
    """Swept rather than argued: the useful window between blind and instantly wrong is empty.

    `Divergence`'s docstring already said a fault its baseline can track will never trip. This
    measures the other end too, because "then tighten it" is the obvious reply.
    """
    from argmind.guard import Divergence

    def froze_at(check, n=3000):
        net = Mlp(3, 1, hidden=(16,), lr=0.02, dtype="f8", seed=1)
        g = Guard(net, [check], rollback_depth=4, checkpoint_every=100)
        r = np.random.default_rng(5)
        for k in range(1, n + 1):
            s = r.standard_normal(3)
            if g.update(s, [_truth(s)]).frozen:
                return k
        return None

    assert froze_at(Divergence(ratio=5.0, floor=0.05, warmup=200)) is None, (
        "the default Divergence fired, so it is not blind here and the sweep is void")
    tight = froze_at(Divergence(ratio=1.05, fast_alpha=0.02, slow_alpha=0.0002,
                                floor=0.05, warmup=200))
    assert tight is not None and tight < 400, (
        f"tightening Divergence did not produce an immediate false trip (froze at {tight}), so "
        "the claim that no useful setting exists between the two is not demonstrated")

    assert froze_at(Bound("residual_norm", hi=2.0)) is not None
    assert froze_at(Bound("residual_norm", hi=2.0)) < 100, (
        "a residual bound loose enough to survive the untrained model's noise would be the "
        "counterexample to this whole test")
    assert froze_at(Bound("residual_norm", hi=3.0)) is None
