// SPDX-License-Identifier: Apache-2.0
//
// Normalised LMS, ported from argmind/adapt.py.
//
// The Python is the specification and the oracle; this is the implementation. Written
// second, in the order RLS took, and everything about *when* it acts is matched exactly.
//
// Why a second adapter exists
// ---------------------------
// `Rls` carries an N x N covariance, so its state is `N*M + N*N` and a `Guard`'s
// checkpoint ring multiplies that by `Depth`. At the handful of sensor channels this
// library was designed against that is nothing. At the width a network embedding has it
// is the binding constraint: measured, a guarded 128-feature RLS head is 330,608 bytes
// against this adapter's 2,912, and at 512 features RLS does not fit an STM32N6's whole
// SRAM. This is O(n) in memory and time and has no covariance at all.
//
// What it gives up, said plainly: RLS is a recursive exact solution of a weighted
// least-squares problem, and this is gradient descent with the step normalised by input
// power. It converges more slowly and its steady-state error is set by `mu` rather than
// driven down to the prior term. It also cannot wind up the way RLS does under poor
// excitation, which is the same fact seen from the other side.
//
// The one exact property, which is what the tests assert
// -----------------------------------------------------
// After an update the a posteriori error is exactly `(1 - mu*x.x/(eps + x.x))` times the
// a priori one, for any data. At `mu = 1` with negligible `eps` that is zero: the sample
// is projected out completely. This is the analogue of RLS being checked against the
// closed-form weighted solution rather than against itself, and it is a law rather than
// a tolerance.
//
// Deliberate divergences from the Python: none beyond what `rls.hpp` already lists for
// reduction order and target FMA contraction. In particular the update is written to
// match numpy's association -- `(x[i] * e[j]) * g`, not `(g * x[i]) * e[j]` -- because
// those round differently and parity here is meant to be arithmetic, not approximate.

#ifndef ARGMIND_NLMS_HPP
#define ARGMIND_NLMS_HPP

#include <stdint.h>

#include "argmind/mat.hpp"
#include "argmind/rls.hpp"   // Report and Reason: one vocabulary for every adapter

namespace argmind {

template <typename T, int N, int M = 1>
class Nlms {
    static_assert(N >= 1, "n_features must be >= 1");
    static_assert(M >= 1, "n_outputs must be >= 1");
    // The same floor as `Rls`, for a WEAKER reason, and the difference is worth stating.
    // RLS's fp16 refusal was measured -- positive-definiteness lost by step 115 at a
    // condition number of 1. There is no covariance here so that argument does not
    // transfer, and fp16 might well work. Nobody has measured it, and writing down
    // untested support is how a claim becomes a fiction.
    static_assert(sizeof(T) >= 4,
                  "fp16 is unmeasured for this recursion, not known to work");

public:
    using scalar = T;

    static constexpr int n_features = N;
    static constexpr int n_outputs = M;
    /// Theta only. A quarter of RLS's state at n=4 and a hundredth at n=128.
    static constexpr int state_size = N * M;

    /// Three of five. The two omitted are covariance quantities and there is no
    /// covariance, so a `Guard` refuses a bound on them at the point it is configured
    /// rather than comparing against a fabricated zero. Spelled out rather than using
    /// `RESIDUAL_FIELDS` because `guard.hpp` includes this header and not the reverse; a
    /// `static_assert` there ties the two together.
    static constexpr uint32_t health_fields = 0x07u;

    Nlms() : mu_(T(0.5)), eps_(T(1e-6)) { reset(); }
    Nlms(T mu, T eps) : mu_(mu), eps_(eps) { reset(); }

    /// Whether these parameters would produce a usable adapter, without building one.
    ///
    /// `mu` in (0, 2] is the standard stability bound, and 2 is the boundary rather than
    /// a safe value. `eps` guards the zero regressor: without it a stretch of all-zero
    /// input divides by zero instead of doing nothing, which is what it should do.
    static bool params_valid(T mu, T eps) {
        return mu > T(0) && mu <= T(2) && eps > T(0) &&
               is_finite(mu) && is_finite(eps);
    }

    bool valid() const { return params_valid(mu_, eps_); }
    T mu() const { return mu_; }
    T eps() const { return eps_; }

    void reset() {
        fill<T, state_size>(state_, T(0));
        fill<T, M>(e_, T(0));
        fill<T, M>(e_post_, T(0));
        updates_ = 0;
        rejected_nonfinite_ = 0;
        rejected_denominator_ = 0;
    }

    const T* state() const { return state_; }
    T* state_mutable() { return state_; }
    const T* theta() const { return state_; }

    bool load_state(const T* __restrict flat) {
        if (!all_finite<T, state_size>(flat)) return false;
        copy<T, state_size>(flat, state_);
        return true;
    }

    void predict(const T* __restrict x, T* __restrict out) const {
        vec_mat<T, N, M>(x, state_, out);
    }

    bool frozen() const { return false; }
    uint32_t updates() const { return updates_; }
    uint32_t rejected_nonfinite() const { return rejected_nonfinite_; }
    uint32_t rejected_denominator() const { return rejected_denominator_; }

    // No `health()`. It is required only of an adapter declaring a covariance field, and
    // the guard's calls sit behind `if constexpr (needs_health(...))`, so for this type
    // the branch is never instantiated. That this compiles is the proof.

    Report<T, M> update(const T* __restrict x, const T* __restrict y) {
        Report<T, M> r;

        if (!all_finite<T, N>(x) || !all_finite<T, M>(y)) {
            fill<T, M>(e_, T(0));
            fill<T, M>(e_post_, T(0));
            ++rejected_nonfinite_;
            return finish(r, false, Reason::NonFiniteInput, T(0));
        }

        // a priori residual, before the state moves
        vec_mat<T, N, M>(x, state_, yhat_);
        for (int j = 0; j < M; ++j) e_[j] = y[j] - yhat_[j];

        const T power = eps_ + dot<T, N>(x, x);
        if (!is_finite(power) || power <= T(0)) {
            // Reachable with FINITE input: a regressor of 1e30 squares to infinity in
            // f32. Same vocabulary as `Rls`, so a guard's counters mean the same thing
            // whichever adapter is underneath it.
            for (int j = 0; j < M; ++j) e_post_[j] = e_[j];
            ++rejected_denominator_;
            return finish(r, false, Reason::NonPositiveDenominator, T(0));
        }

        const T g = mu_ / power;
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < M; ++j) {
                // `(x*e)*g`, matching numpy's `np.outer(x, e) * g`. The other
                // association rounds differently and this is meant to be bit-comparable.
                state_[i * M + j] += (x[i] * e_[j]) * g;
            }
        }

        vec_mat<T, N, M>(x, state_, yhat_);
        for (int j = 0; j < M; ++j) e_post_[j] = y[j] - yhat_[j];

        ++updates_;
        // The gain VECTOR is `g*x`, so its norm is what compares against RLS's ||k||.
        // Reporting `g` alone would make `gain_norm` mean different things in the two
        // adapters, and a `Guard` bound is written against the name rather than the type.
        T gx[N];
        for (int i = 0; i < N; ++i) gx[i] = g * x[i];
        return finish(r, true, Reason::None, norm<T, N>(gx));
    }

private:
    Report<T, M>& finish(Report<T, M>& r, bool applied, Reason reason, T gain) {
        r.applied = applied;
        r.reason = reason;
        copy<T, M>(e_, r.residual);
        copy<T, M>(e_post_, r.residual_post);
        r.residual_norm = static_cast<double>(norm<T, M>(e_));
        r.residual_post_norm = static_cast<double>(norm<T, M>(e_post_));
        r.gain_norm = static_cast<double>(gain);
        // Declared absent, so a `Guard` refuses to watch these and nothing reads them.
        r.p_trace = 0.0;
        r.p_min_diag = 0.0;
        return r;
    }

    T mu_;
    T eps_;
    T state_[state_size];
    T e_[M];
    T e_post_[M];
    T yhat_[M];
    uint32_t updates_;
    uint32_t rejected_nonfinite_;
    uint32_t rejected_denominator_;
};

}  // namespace argmind

#endif  // ARGMIND_NLMS_HPP
