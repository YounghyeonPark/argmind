// SPDX-License-Identifier: Apache-2.0
//
// A dense feed-forward network, trained by SGD, wearing the same adapter contract.
//
// Why it is here and not in the core
// ----------------------------------
// Arithmetic, not preference. A `Guard`'s checkpoint ring is `Depth * state_size`, and a
// network's state is its weights: 1.6 KB total for RLS at six features, **47.5 KB** for a
// 3-32-32-1 network at depth 4. That does not fit the declared 8 KB core floor. It also
// needs `tanh` from libm, which the freestanding core deliberately does not link.
//
// The layer question this settles
// -------------------------------
// The 2026-08-10 scope decision is that generating the basis is the platform's job. The
// measurement behind it is memory: against a target no fixed basis suits, a network
// reaches the same accuracy at 9.5 KB that random Fourier features through RLS need
// 131 KB for, because a covariance is n^2 and weights are not. What it costs is samples,
// about 4x, and in its first few hundred it is worse than a constant.
//
// What this gives up, stated because a `Guard` is written against it
// ------------------------------------------------------------------
// There is no covariance, so no `p_trace`: `health_fields` is the residual three, and a
// guard refuses a bound on a covariance field at the point it is configured. **There is no
// confidence-per-direction for a backpropagated weight**, so a windup-shaped fault -- a
// direction nothing has excited drifting away -- has no analogue here and no check.
//
// Everything else a guard does works, and it matters more here than for RLS: measured, a
// sensor latched high for 400 samples leaves RLS at 1.00x its pre-fault quality twenty
// thousand clean samples later and this at **3.33x**. The least-squares recursion pulls
// back to the correct fit; a walk downhill from a corrupted position does not retrace. For
// RLS a guard buys time; for this it is what keeps a fault from being permanent.
//
// Initialisation does NOT match the reference, on purpose
// -------------------------------------------------------
// The reference draws Glorot weights from numpy's PCG64 with a ziggurat normal. Matching
// that bit-for-bit would mean reimplementing both, and a generator that *nearly* matches
// is worse than one that plainly does not. So this uses its own stated generator
// (splitmix64 with Box-Muller) and says so. **To reproduce a reference run, load the
// reference's state block** -- which is what the parity harness does, and what a device
// restoring a trained model does anyway.

#ifndef ARGMIND_HOST_RUNTIME_SHAPE_MLP_HPP
#define ARGMIND_HOST_RUNTIME_SHAPE_MLP_HPP

#include <stdint.h>

#include <cmath>
#include <vector>

#include "argmind/mat.hpp"
#include "argmind/rls.hpp"   // Report and Reason: one vocabulary for every adapter
#include "argmind/host/runtime_shape.hpp"   // RuntimeShapedReport

namespace argmind {
namespace host {

enum class Activation : uint8_t { Tanh = 0, Relu = 1 };

template <typename T>
class RuntimeShapedMlp {
    // fp16 refused on measurement: on a WELL-SCALED target it underflows 42-82% of its
    // steps against fp32's 0.0-0.8% and costs 1.06x to 2.13x the error, because `lr*grad`
    // falls below the spacing at the weight's own magnitude. Nothing is driven to zero, so
    // a NaN-or-zero check misses it. fp32 against fp64 is 1.0000 to a million updates.
    static_assert(sizeof(T) >= 4, "fp16 measured to fail; see the reference's docstring");

public:
    using scalar = T;

    /// Three of five, matching `Nlms` and the reference: no covariance.
    static constexpr uint32_t health_fields = 0x07u;

    static bool params_valid(T learning_rate) {
        return learning_rate > T(0) && is_finite(learning_rate);
    }

    /// `hidden` may be empty, and that case is worth having: a single linear layer makes
    /// the update closed-form, so the a posteriori error is exactly
    /// `(1 - 2*lr*(x.x + 1))` times the a priori one. The `+ 1` is the bias, which acts
    /// as a constant input of one -- the reference's first version of that law omitted it
    /// and was wrong by a factor of twenty.
    RuntimeShapedMlp(int n_features, int n_outputs, const std::vector<int>& hidden,
                     Activation activation, T learning_rate, uint64_t seed = 0)
        : n_(n_features), m_(n_outputs), act_(activation), lr_(learning_rate) {
        std::vector<int> widths;
        widths.push_back(n_features);
        for (int h : hidden) widths.push_back(h);
        widths.push_back(n_outputs);

        size_t total = 0;
        for (size_t i = 0; i + 1 < widths.size(); ++i) {
            const int a = widths[i], b = widths[i + 1];
            shapes_.push_back({a, b});
            offsets_.push_back(total);
            total += static_cast<size_t>(a) * b + b;
            valid_ = valid_ && a >= 1 && b >= 1;
        }
        state_.assign(total, T(0));

        // Widest layer boundary, so the forward buffers are allocated once.
        int widest = 0;
        for (int w : widths) widest = w > widest ? w : widest;
        acts_.assign(widths.size() * static_cast<size_t>(widest), T(0));
        act_stride_ = static_cast<size_t>(widest);
        delta_.assign(static_cast<size_t>(widest), T(0));
        delta_next_.assign(static_cast<size_t>(widest), T(0));
        pred_a_.assign(static_cast<size_t>(widest), T(0));
        pred_b_.assign(static_cast<size_t>(widest), T(0));
        e_.assign(static_cast<size_t>(n_outputs), T(0));
        e_post_.assign(static_cast<size_t>(n_outputs), T(0));

        glorot(seed);
    }

    bool valid() const {
        return valid_ && n_ >= 1 && m_ >= 1 && params_valid(lr_);
    }

    int n_features() const { return n_; }
    int n_outputs() const { return m_; }
    int state_size() const { return static_cast<int>(state_.size()); }
    uint32_t updates() const { return updates_; }
    uint32_t rejected_nonfinite() const { return rejected_nonfinite_; }
    Activation activation() const { return act_; }

    const T* state() const { return state_.data(); }
    T* state_mutable() { return state_.data(); }

    bool load_state(const T* flat) {
        for (size_t i = 0; i < state_.size(); ++i) {
            if (!is_finite(flat[i])) return false;
        }
        for (size_t i = 0; i < state_.size(); ++i) state_[i] = flat[i];
        return true;
    }

    /// Zero weights, matching the reference. **This stops the network rather than
    /// restarting it**: every hidden unit then computes the same function with the same
    /// gradient, a symmetry it cannot learn out of. RLS's zero theta is a fresh start;
    /// this is not. Construct a new adapter to restart.
    void reset() {
        for (T& v : state_) v = T(0);
        for (T& v : e_) v = T(0);
        for (T& v : e_post_) v = T(0);
        updates_ = 0;
        rejected_nonfinite_ = 0;
    }

    /// Allocates nothing. The scratch is `mutable` and sized at construction, which costs
    /// thread-safety on a single object -- stated rather than discovered, and the same
    /// trade every adapter here already makes by holding its report buffers.
    void predict(const T* x, T* out) const {
        T* a = pred_a_.data();
        T* b = pred_b_.data();
        for (int i = 0; i < n_; ++i) a[i] = x[i];
        for (size_t l = 0; l < shapes_.size(); ++l) {
            layer(l, a, b, l + 1 < shapes_.size());
            T* t = a; a = b; b = t;
        }
        for (int j = 0; j < m_; ++j) out[j] = a[j];
    }


    /// A network with more INPUTS that keeps everything this one learned.
    ///
    /// A new input is a new **row** of `W0`, filled with zeros. The new inputs then
    /// contribute nothing whatever their values, so the widened network computes the old
    /// function of the old inputs -- measured at 0.0 to 2.3 ulp of the run's scale over
    /// five architectures, against 2.0 to 4.6 for filling them randomly on a target whose
    /// own spread is 1.4. For a device mid-flight when a sensor is installed, that
    /// continuity is the requirement.
    ///
    /// **And the new rows still learn**, because the gradient into one is `x_new * delta`
    /// and `delta` comes from an already-trained downstream. That is exactly what a new
    /// hidden UNIT does not have, which is why there is no `deepened()` here: extra hidden
    /// units were measured to buy nothing on any target that could be constructed (h=8
    /// already reaches the ceiling, h=64 is sometimes worse), because SGD's convergence
    /// binds before capacity does. The reference's `deepened()` exists only to refuse and
    /// carries that measurement; nothing is gained by a second copy of a refusal.
    ///
    /// Returns an invalid adapter if `n_features` is narrower -- checked with `valid()`
    /// rather than thrown, matching `RuntimeShapedRls::widened`.
    RuntimeShapedMlp<T> widened(int n_features) const {
        std::vector<int> hidden;
        for (size_t l = 0; l + 1 < shapes_.size(); ++l) hidden.push_back(shapes_[l].out);
        if (n_features < n_) {
            return RuntimeShapedMlp<T>(0, 0, hidden, act_, lr_);
        }

        RuntimeShapedMlp<T> out(n_features, m_, hidden, act_, lr_);
        const int h0 = shapes_[0].out;
        T* w0 = out.state_.data();
        for (size_t i = 0; i < static_cast<size_t>(n_features) * h0; ++i) w0[i] = T(0);
        for (int i = 0; i < n_; ++i) {
            for (int j = 0; j < h0; ++j) {
                w0[static_cast<size_t>(i) * h0 + j] =
                    state_[static_cast<size_t>(i) * h0 + j];
            }
        }
        // Everything from `b0` onward is unchanged, and the two blocks have the same
        // length there because only `W0` grew.
        const size_t tail_src = static_cast<size_t>(n_) * h0;
        const size_t tail_dst = static_cast<size_t>(n_features) * h0;
        for (size_t k = 0; k + tail_src < state_.size(); ++k) {
            out.state_[tail_dst + k] = state_[tail_src + k];
        }
        return out;
    }

    RuntimeShapedReport<T> update(const T* x, const T* y) {
        RuntimeShapedReport<T> r;
        for (int i = 0; i < n_; ++i) {
            if (!is_finite(x[i])) return reject(r);
        }
        for (int j = 0; j < m_; ++j) {
            if (!is_finite(y[j])) return reject(r);
        }

        forward(x);
        const T* out = act_row(shapes_.size());
        for (int j = 0; j < m_; ++j) e_[j] = y[j] - out[j];

        // d(||y - yhat||^2) / d(pre-activation of the output layer).
        for (int j = 0; j < m_; ++j) delta_[static_cast<size_t>(j)] = T(-2) * e_[j];

        double sq = 0.0;
        for (size_t l = shapes_.size(); l-- > 0;) {
            const int in = shapes_[l].in, outw = shapes_[l].out;
            const T* a = act_row(l);
            T* w = state_.data() + offsets_[l];
            T* bias = w + static_cast<size_t>(in) * outw;

            if (l) {
                // Compute the next delta BEFORE the weights move, or the chain rule uses
                // the updated matrix. The reference computes every gradient first and
                // then applies them, so this ordering is what matches it.
                for (int k = 0; k < in; ++k) {
                    T acc = T(0);
                    for (int j = 0; j < outw; ++j)
                        acc += delta_[static_cast<size_t>(j)] *
                               w[static_cast<size_t>(k) * outw + j];
                    delta_next_[static_cast<size_t>(k)] = acc * dact(a[k]);
                }
            }

            for (int k = 0; k < in; ++k) {
                for (int j = 0; j < outw; ++j) {
                    const T g = a[k] * delta_[static_cast<size_t>(j)];
                    w[static_cast<size_t>(k) * outw + j] -= lr_ * g;
                    sq += static_cast<double>(g) * static_cast<double>(g);
                }
            }
            for (int j = 0; j < outw; ++j) {
                const T g = delta_[static_cast<size_t>(j)];
                bias[j] -= lr_ * g;
                sq += static_cast<double>(g) * static_cast<double>(g);
            }
            if (l) delta_.swap(delta_next_);
        }

        // Straight through `forward` rather than `predict`, because the reference does
        // exactly that (`self._forward(xa)[-1]`) and a second code path here would be a
        // place for the two to diverge silently.
        forward(x);
        const T* post = act_row(shapes_.size());
        for (int j = 0; j < m_; ++j) e_post_[j] = y[j] - post[j];

        ++updates_;
        return finish(r, true, Reason::None,
                      static_cast<double>(lr_) * std::sqrt(sq));
    }

private:
    struct Shape { int in, out; };

    T actv(T z) const {
        return act_ == Activation::Tanh ? std::tanh(z) : (z > T(0) ? z : T(0));
    }

    /// Both derivatives are expressible in the ACTIVATION rather than the pre-activation,
    /// which is why only the activations are stored and the pre-activations are not.
    T dact(T a) const {
        return act_ == Activation::Tanh ? T(1) - a * a : (a > T(0) ? T(1) : T(0));
    }

    void layer(size_t l, const T* in, T* out, bool nonlinear) const {
        const int ni = shapes_[l].in, no = shapes_[l].out;
        const T* w = state_.data() + offsets_[l];
        const T* bias = w + static_cast<size_t>(ni) * no;
        for (int j = 0; j < no; ++j) {
            T acc = T(0);
            for (int i = 0; i < ni; ++i) acc += in[i] * w[static_cast<size_t>(i) * no + j];
            const T z = acc + bias[j];
            out[j] = nonlinear ? actv(z) : z;
        }
    }

    T* act_row(size_t l) { return acts_.data() + l * act_stride_; }
    const T* act_row(size_t l) const { return acts_.data() + l * act_stride_; }

    void forward(const T* x) {
        T* first = act_row(0);
        for (int i = 0; i < n_; ++i) first[i] = x[i];
        for (size_t l = 0; l < shapes_.size(); ++l)
            layer(l, act_row(l), act_row(l + 1), l + 1 < shapes_.size());
    }

    /// splitmix64 and Box-Muller. **Deliberately not numpy's** -- see the header comment.
    void glorot(uint64_t seed) {
        uint64_t s = seed + 0x9E3779B97F4A7C15ull;
        auto next_u64 = [&s]() {
            uint64_t z = (s += 0x9E3779B97F4A7C15ull);
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
            return z ^ (z >> 31);
        };
        auto uniform = [&next_u64]() {
            return (static_cast<double>(next_u64() >> 11) + 0.5) * (1.0 / 9007199254740992.0);
        };
        for (size_t l = 0; l < shapes_.size(); ++l) {
            const int a = shapes_[l].in, b = shapes_[l].out;
            const double sd = std::sqrt(2.0 / (a + b));
            T* w = state_.data() + offsets_[l];
            for (size_t i = 0; i < static_cast<size_t>(a) * b; ++i) {
                const double u1 = uniform(), u2 = uniform();
                w[i] = static_cast<T>(std::sqrt(-2.0 * std::log(u1)) *
                                      std::cos(6.283185307179586 * u2) * sd);
            }
        }
    }

    RuntimeShapedReport<T>& reject(RuntimeShapedReport<T>& r) {
        for (T& v : e_) v = T(0);
        for (T& v : e_post_) v = T(0);
        ++rejected_nonfinite_;
        return finish(r, false, Reason::NonFiniteInput, 0.0);
    }

    RuntimeShapedReport<T>& finish(RuntimeShapedReport<T>& r, bool applied,
                                   Reason why, double gain) {
        r.applied = applied;
        r.reason = why;
        r.residual = e_;
        r.residual_post = e_post_;
        r.residual_norm = detail::norm_of(e_);
        r.residual_post_norm = detail::norm_of(e_post_);
        r.gain_norm = gain;
        r.p_trace = 0.0;
        r.p_min_diag = 0.0;
        return r;
    }

    int n_, m_;
    Activation act_;
    T lr_;
    bool valid_ = true;
    std::vector<Shape> shapes_;
    std::vector<size_t> offsets_;
    std::vector<T> state_;
    std::vector<T> acts_;
    size_t act_stride_ = 0;
    std::vector<T> delta_, delta_next_, e_, e_post_;
    mutable std::vector<T> pred_a_, pred_b_;
    uint32_t updates_ = 0;
    uint32_t rejected_nonfinite_ = 0;
};

}  // namespace host
}  // namespace argmind

#endif  // ARGMIND_HOST_RUNTIME_SHAPE_MLP_HPP
