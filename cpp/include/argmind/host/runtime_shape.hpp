// SPDX-License-Identifier: Apache-2.0
//
// Runtime-shaped adapters, for the hosted tier.
//
// The name says what is runtime about it: the SHAPE -- `n_features` and `n_outputs`,
// nothing else. These were called `DynRls`/`DynNlms` in `dyn.hpp` for one commit and
// renamed after the first person to read it asked what `dyn` meant. That question is
// the answer: an abbreviation that does not name the varying thing is a bad name in a
// public API, and in Rust `dyn` means dynamic dispatch, which this is not.
//
// Why this exists, and why it is a separate directory
// ---------------------------------------------------
// The core takes its dimensions as template parameters. That is the reason C++ was chosen
// -- the compiler unrolls, keeps operands in registers and vectorises -- and the C ABI
// pays for it with an explicit instantiation list, so adding a shape is three lines and a
// rebuild. On a microcontroller that trade is obviously right: firmware is rebuilt anyway.
//
// It is the wrong trade above an OS, where a platform wants a shape from a configuration
// file. So `argmind/host/` holds the other side of it. **The path is the boundary**: these
// headers use the heap and are not freestanding, the core headers must never include one,
// and `tests/test_target_build.py` checks both halves rather than trusting the convention.
//
// Measured before deciding, because the obvious argument for runtime shapes is wrong:
// **it does not save flash.** `--gc-sections` already means a firmware pays only for the
// shapes it calls -- 2,308 bytes for bare RLS against 38,770 for all fifteen
// instantiations, and a per-shape shim is 102 to 116 bytes. What runtime shapes cost is
// inlining and declared memory; what they buy is not rebuilding.
//
// This is a port of the reference, not a new algorithm
// ----------------------------------------------------
// `argmind/adapt.py` has always been runtime-shaped: `RecursiveLeastSquares(n, m, ...)`
// takes its dimensions as arguments. So `RuntimeShapedRls` is a *closer*
// transcription of the
// specification than `Rls<T,N,M>` is, and parity against it can be exact rather than
// bounded in ulp.
//
// Which is why there is **one deliberate difference from `rls.hpp`**: it divides where the
// core multiplies by a reciprocal. Divergences 1 and 2 there buy `VDIV.F32` cycles on a
// Cortex-M4F, and above an OS there is nothing to buy, so matching the reference is worth
// more than saving a divide.
//
// **That removes two of the three divergences and not the third**, which is worth saying
// because the first version of this comment claimed it "turns a tolerance into an
// equality" and then the measurement did not. Reduction order still differs -- numpy
// dispatches its dots to BLAS and these are ordered loops -- so the comparison stays a
// tolerance, just a tight one that grows with the sum length exactly as reduction order
// should. Measured at shapes the instantiation list does not contain, both adapters, both
// precisions: 0.5 ulp at 5x1, 1.0 at 7x1, and 2.5 to 4.0 at 50x1.

#ifndef ARGMIND_HOST_RUNTIME_SHAPE_HPP
#define ARGMIND_HOST_RUNTIME_SHAPE_HPP

#include <stdint.h>

#include <vector>

#include "argmind/mat.hpp"   // is_finite, and nothing that needs a shape
#include "argmind/rls.hpp"   // Report and Reason: one vocabulary for every adapter

namespace argmind {
namespace host {

/// The runtime-shaped report. `Report<T, M>` fixes its residual arrays at compile time,
/// so it cannot carry a shape decided at construction.
template <typename T>
struct RuntimeShapedReport {
    bool applied = false;
    Reason reason = Reason::None;
    std::vector<T> residual;
    std::vector<T> residual_post;
    double residual_norm = 0.0;
    double residual_post_norm = 0.0;
    double gain_norm = 0.0;
    double p_trace = 0.0;
    double p_min_diag = 0.0;
};

namespace detail {

template <typename T>
double norm_of(const std::vector<T>& v) {
    T acc = T(0);
    for (T e : v) acc += e * e;
    return static_cast<double>(sqrt_of(acc));
}

}  // namespace detail

/// Exponentially weighted RLS with the shape decided at construction.
///
/// State layout is byte-identical to `Rls<T,N,M>`: theta (n*m) then P (n*n), row-major.
/// So a block written by one is readable by the other at the same shape, and
/// `persist.hpp`'s record format needs no special case.
template <typename T>
class RuntimeShapedRls {
    static_assert(sizeof(T) >= 4,
                  "fp32 is the measured floor for this recursion; there is no "
                  "half-precision path in any filter form");

public:
    using scalar = T;

    /// All five, matching `Rls`. Spelled out rather than naming `ALL_FIELDS`, because
    /// `guard.hpp` includes this file's dependencies and not the reverse; the
    /// runtime-shaped guard static-asserts the tie.
    static constexpr uint32_t health_fields = 0x1Fu;

    static bool params_valid(T lambda, T p0) {
        return lambda > T(0) && lambda <= T(1) && p0 > T(0) &&
               is_finite(lambda) && is_finite(p0);
    }

    RuntimeShapedRls(int n_features, int n_outputs, T lambda, T p0)
        : n_(n_features), m_(n_outputs), lambda_(lambda), p0_(p0),
          state_(static_cast<size_t>(n_features) * n_outputs +
                     static_cast<size_t>(n_features) * n_features, T(0)),
          px_(n_features, T(0)), k_(n_features, T(0)),
          e_(n_outputs, T(0)), e_post_(n_outputs, T(0)), yhat_(n_outputs, T(0)) {
        reset();
    }

    bool valid() const {
        return n_ >= 1 && m_ >= 1 && params_valid(lambda_, p0_);
    }

    int n_features() const { return n_; }
    int n_outputs() const { return m_; }
    int state_size() const { return static_cast<int>(state_.size()); }
    T lambda() const { return lambda_; }
    T p0() const { return p0_; }
    uint32_t updates() const { return updates_; }

    const T* state() const { return state_.data(); }
    T* state_mutable() { return state_.data(); }
    T* theta() { return state_.data(); }
    const T* theta() const { return state_.data(); }
    T* p() { return state_.data() + static_cast<size_t>(n_) * m_; }
    const T* p() const { return state_.data() + static_cast<size_t>(n_) * m_; }

    bool load_state(const T* flat) {
        for (size_t i = 0; i < state_.size(); ++i) {
            if (!is_finite(flat[i])) return false;
        }
        for (size_t i = 0; i < state_.size(); ++i) state_[i] = flat[i];
        return true;
    }

    void reset() {
        for (T& v : state_) v = T(0);
        for (int i = 0; i < n_; ++i) p()[static_cast<size_t>(i) * n_ + i] = p0_;
        for (T& v : e_) v = T(0);
        for (T& v : e_post_) v = T(0);
        updates_ = 0;
        rejected_nonfinite_ = 0;
        rejected_denominator_ = 0;
    }

    void predict(const T* x, T* out) const {
        for (int j = 0; j < m_; ++j) {
            T acc = T(0);
            for (int i = 0; i < n_; ++i)
                acc += x[i] * theta()[static_cast<size_t>(i) * m_ + j];
            out[j] = acc;
        }
    }

    /// A wider adapter that keeps everything this one learned.
    ///
    /// **Appending only.** Feature order is parameter order, and a persisted block
    /// restored against a different order is wrong undetectably -- which is why
    /// `persist` carries a configuration fingerprint. Appending is the one widening
    /// that permutes nothing: old feature `j` is still feature `j`. Rearranging would
    /// need a mapping the caller supplies, and guessing one is the failure the
    /// fingerprint exists to prevent.
    ///
    /// `theta` keeps its coefficients and the new ones start at zero. `P` keeps its
    /// learned block in the corner and the new diagonal gets `p0`, with zero cross
    /// terms: `P` is the inverse information matrix, so the corner carries the
    /// confidence this adapter earned, the diagonal says nothing is known about the new
    /// features yet, and no evidence yet relates the two sets.
    ///
    /// **It adds inputs, not expressiveness.** Still linear in the parameters.
    ///
    /// Returns an invalid adapter if `n_features` is narrower -- checked with `valid()`
    /// rather than thrown, matching the rest of this library.
    RuntimeShapedRls<T> widened(int n_features, T p0) const {
        if (n_features < n_) return RuntimeShapedRls<T>(0, 0, lambda_, p0);
        RuntimeShapedRls<T> out(n_features, m_, lambda_, p0);
        for (int i = 0; i < n_; ++i) {
            for (int j = 0; j < m_; ++j) {
                out.theta()[static_cast<size_t>(i) * m_ + j] =
                    theta()[static_cast<size_t>(i) * m_ + j];
            }
        }
        for (int i = 0; i < n_; ++i) {
            for (int j = 0; j < n_; ++j) {
                out.p()[static_cast<size_t>(i) * n_features + j] =
                    p()[static_cast<size_t>(i) * n_ + j];
            }
        }
        return out;
    }

    RuntimeShapedRls<T> widened(int n_features) const {
        return widened(n_features, p0_);
    }

    void health(double* p_trace, double* p_min_diag) const {
        double tr = 0.0;
        double lo = static_cast<double>(p()[0]);
        for (int i = 0; i < n_; ++i) {
            const double d = static_cast<double>(p()[static_cast<size_t>(i) * n_ + i]);
            tr += d;
            if (d < lo) lo = d;
        }
        *p_trace = tr;
        *p_min_diag = lo;
    }

    RuntimeShapedReport<T> update(const T* x, const T* y) {
        RuntimeShapedReport<T> r;
        for (int i = 0; i < n_; ++i) {
            if (!is_finite(x[i])) return reject(r, Reason::NonFiniteInput);
        }
        for (int j = 0; j < m_; ++j) {
            if (!is_finite(y[j])) return reject(r, Reason::NonFiniteInput);
        }

        // a priori residual, before the state moves
        predict(x, yhat_.data());
        for (int j = 0; j < m_; ++j) e_[j] = y[j] - yhat_[j];

        // px = P x
        for (int i = 0; i < n_; ++i) {
            T acc = T(0);
            for (int j = 0; j < n_; ++j)
                acc += p()[static_cast<size_t>(i) * n_ + j] * x[j];
            px_[i] = acc;
        }
        T den = lambda_;
        for (int i = 0; i < n_; ++i) den += x[i] * px_[i];

        if (!is_finite(den) || den <= T(0)) {
            for (int j = 0; j < m_; ++j) e_post_[j] = e_[j];
            ++rejected_denominator_;
            return finish(r, false, Reason::NonPositiveDenominator, 0.0);
        }

        // DIVIDE, matching the reference. `rls.hpp` multiplies by a reciprocal to save
        // VDIV.F32 cycles on a Cortex-M4F; there is nothing to save above an OS, and
        // matching exactly turns a tolerance into an equality.
        for (int i = 0; i < n_; ++i) k_[i] = px_[i] / den;

        for (int i = 0; i < n_; ++i) {
            for (int j = 0; j < m_; ++j) {
                theta()[static_cast<size_t>(i) * m_ + j] += k_[i] * e_[j];
            }
        }
        for (int i = 0; i < n_; ++i) {
            for (int j = 0; j < n_; ++j) {
                T& pij = p()[static_cast<size_t>(i) * n_ + j];
                pij = (pij - k_[i] * px_[j]) / lambda_;
            }
        }
        // Symmetrise. The rank-1 subtraction is symmetric in exact arithmetic and is not
        // in floating point, and the asymmetry compounds.
        for (int i = 0; i < n_; ++i) {
            for (int j = 0; j < n_; ++j) {
                const size_t a = static_cast<size_t>(i) * n_ + j;
                const size_t b = static_cast<size_t>(j) * n_ + i;
                const T avg = (p()[a] + p()[b]) * T(0.5);
                sym_[a] = avg;
            }
        }
        for (int i = 0; i < n_ * n_; ++i) p()[i] = sym_[i];

        predict(x, yhat_.data());
        for (int j = 0; j < m_; ++j) e_post_[j] = y[j] - yhat_[j];

        ++updates_;
        return finish(r, true, Reason::None, detail::norm_of(k_));
    }

private:
    RuntimeShapedReport<T>& reject(RuntimeShapedReport<T>& r, Reason why) {
        for (T& v : e_) v = T(0);
        for (T& v : e_post_) v = T(0);
        ++rejected_nonfinite_;
        return finish(r, false, why, 0.0);
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
        health(&r.p_trace, &r.p_min_diag);
        return r;
    }

    int n_, m_;
    T lambda_, p0_;
    std::vector<T> state_;
    std::vector<T> px_, k_, e_, e_post_, yhat_;
    std::vector<T> sym_ = std::vector<T>(static_cast<size_t>(n_) * n_, T(0));
    uint32_t updates_ = 0;
    uint32_t rejected_nonfinite_ = 0;
    uint32_t rejected_denominator_ = 0;
};

/// Normalised LMS with the shape decided at construction. State is theta only.
template <typename T>
class RuntimeShapedNlms {
    static_assert(sizeof(T) >= 4,
                  "fp16 is unmeasured for this recursion, not known to work");

public:
    using scalar = T;

    /// Three of five, matching `Nlms`: no covariance, so a guard refuses a bound on
    /// `p_trace` or `p_min_diag` at the point it is configured.
    static constexpr uint32_t health_fields = 0x07u;

    static bool params_valid(T mu, T eps) {
        return mu > T(0) && mu <= T(2) && eps > T(0) &&
               is_finite(mu) && is_finite(eps);
    }

    RuntimeShapedNlms(int n_features, int n_outputs, T mu, T eps)
        : n_(n_features), m_(n_outputs), mu_(mu), eps_(eps),
          state_(static_cast<size_t>(n_features) * n_outputs, T(0)),
          e_(n_outputs, T(0)), e_post_(n_outputs, T(0)), yhat_(n_outputs, T(0)) {}

    bool valid() const { return n_ >= 1 && m_ >= 1 && params_valid(mu_, eps_); }

    int n_features() const { return n_; }
    int n_outputs() const { return m_; }
    int state_size() const { return static_cast<int>(state_.size()); }
    uint32_t updates() const { return updates_; }

    const T* state() const { return state_.data(); }
    T* state_mutable() { return state_.data(); }

    bool load_state(const T* flat) {
        for (size_t i = 0; i < state_.size(); ++i) {
            if (!is_finite(flat[i])) return false;
        }
        for (size_t i = 0; i < state_.size(); ++i) state_[i] = flat[i];
        return true;
    }

    void reset() {
        for (T& v : state_) v = T(0);
        for (T& v : e_) v = T(0);
        for (T& v : e_post_) v = T(0);
        updates_ = 0;
        rejected_nonfinite_ = 0;
        rejected_denominator_ = 0;
    }

    void predict(const T* x, T* out) const {
        for (int j = 0; j < m_; ++j) {
            T acc = T(0);
            for (int i = 0; i < n_; ++i)
                acc += x[i] * state_[static_cast<size_t>(i) * m_ + j];
            out[j] = acc;
        }
    }

    RuntimeShapedReport<T> update(const T* x, const T* y) {
        RuntimeShapedReport<T> r;
        for (int i = 0; i < n_; ++i) {
            if (!is_finite(x[i])) return reject(r, Reason::NonFiniteInput);
        }
        for (int j = 0; j < m_; ++j) {
            if (!is_finite(y[j])) return reject(r, Reason::NonFiniteInput);
        }

        predict(x, yhat_.data());
        for (int j = 0; j < m_; ++j) e_[j] = y[j] - yhat_[j];

        T power = eps_;
        for (int i = 0; i < n_; ++i) power += x[i] * x[i];
        if (!is_finite(power) || power <= T(0)) {
            for (int j = 0; j < m_; ++j) e_post_[j] = e_[j];
            ++rejected_denominator_;
            return finish(r, false, Reason::NonPositiveDenominator, 0.0);
        }

        const T g = mu_ / power;
        for (int i = 0; i < n_; ++i) {
            for (int j = 0; j < m_; ++j) {
                // `(x*e)*g`, matching numpy's `np.outer(x, e) * g`; the other association
                // rounds differently.
                state_[static_cast<size_t>(i) * m_ + j] += (x[i] * e_[j]) * g;
            }
        }

        predict(x, yhat_.data());
        for (int j = 0; j < m_; ++j) e_post_[j] = y[j] - yhat_[j];

        ++updates_;
        std::vector<T> gx(static_cast<size_t>(n_));
        for (int i = 0; i < n_; ++i) gx[i] = g * x[i];
        return finish(r, true, Reason::None, detail::norm_of(gx));
    }

private:
    RuntimeShapedReport<T>& reject(RuntimeShapedReport<T>& r, Reason why) {
        for (T& v : e_) v = T(0);
        for (T& v : e_post_) v = T(0);
        ++rejected_nonfinite_;
        return finish(r, false, why, 0.0);
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
        // No covariance, so these are declared absent the same way `Nlms` declares them.
        r.p_trace = 0.0;
        r.p_min_diag = 0.0;
        return r;
    }

    int n_, m_;
    T mu_, eps_;
    std::vector<T> state_;
    std::vector<T> e_, e_post_, yhat_;
    uint32_t updates_ = 0;
    uint32_t rejected_nonfinite_ = 0;
    uint32_t rejected_denominator_ = 0;
};

}  // namespace host
}  // namespace argmind

#endif  // ARGMIND_HOST_RUNTIME_SHAPE_HPP
