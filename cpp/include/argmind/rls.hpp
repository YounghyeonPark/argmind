// SPDX-License-Identifier: Apache-2.0
//
// Exponentially weighted recursive least squares, ported from argmind/adapt.py.
//
// The Python is the specification and the oracle; this is the implementation. Where
// they can differ they are made to differ for a stated reason and nowhere else, and
// the differences are listed at the bottom of this comment so that a parity failure
// can be read as either "a known divergence grew" or "something is wrong".
//
// The contract is the one adapt.py sets out:
//   - dimensions fixed at construction, here at compile time
//   - the whole state is one flat contiguous block: theta (N*M) then P (N*N),
//     row-major, byte-identical to what the Python writes and what
//     argmind/persist.py stores
//   - update() returns a report, never nothing, because the guard is written
//     against its fields
//   - a rejected update leaves the state untouched and says why: a NaN from a
//     sensor is a runtime event, not a programming error, and must not take the
//     estimator with it
//
// No exceptions, no RTTI, no heap, no virtuals. Nothing here allocates and nothing
// here can throw, so a caller in a freestanding environment needs no runtime.
//
// Known, deliberate divergences from the Python reference:
//
//   1. `P /= lambda` becomes a multiply by a precomputed reciprocal. On an M4F
//      VDIV.F32 is 14 cycles and that statement runs N*N times. Costs one extra
//      rounding.
//   2. `k = Px / den` likewise becomes a reciprocal and N multiplies.
//   3. Reductions are ordered loops. numpy dispatches these to BLAS, which may
//      block or vectorise them and therefore sum in a different order. Neither is
//      more correct; they simply are not bit-identical.
//   4. **On a target with a fused multiply-add, GCC contracts `a*b + c` into one.**
//      Added after a parity review measured it rather than after anyone chose it:
//      VFPv4 on the Cortex-M4F has `VFMA.F32`, the x86-64 SSE2 baseline does not, and
//      the target's f32 figures are therefore not the host's. Measured over sixty
//      configurations -- state 2.0 ulp on the host against 5.0 on the target,
//      residual_norm 14.2 against 38.0, gain_norm 87.4 against 134.0 -- and
//      `-ffp-contract=off` on the target build reproduces the host figures *exactly*,
//      which is what identifies the cause. One rounding instead of two is not worse
//      arithmetic; it is different arithmetic, and a divergence list that omitted it
//      would make a target-versus-host ulp count look like a bug.
//
//      f64 is untouched: doubles are soft float on a single-precision FPU and there is
//      nothing to contract. `align.hpp` records the f64 half of this and, until the
//      measurement above, said the M4F "cannot" contract -- true of the double blend it
//      was describing and false of everything in this file.
//
// Everything else — the order of the steps, where the a priori residual is taken,
// the symmetrisation, the trace accumulated in double, the rejection conditions —
// is matched exactly, on purpose.

#ifndef ARGMIND_RLS_HPP
#define ARGMIND_RLS_HPP

#include <stdint.h>

#include "argmind/mat.hpp"

namespace argmind {

/// Why an update did not apply. Values are part of the ABI and match the string
/// tokens the Python reference reports.
enum class Reason : uint8_t {
    None = 0,
    NonFiniteInput = 1,
    NonPositiveDenominator = 2,
};

/// What one update did. Mirrors argmind.adapt.UpdateReport.
///
/// The scalars are double even when T is float, because that is what the Python
/// hands a guard and the guard's thresholds are written against those numbers.
/// `p_trace` is genuinely accumulated in double; `gain_norm` is computed in T and
/// then widened, because numpy's norm of a float32 array is a float32 computation.
template <typename T, int M>
struct Report {
    bool applied;
    Reason reason;
    T residual[M];       ///< a priori: y - predict(x), taken before the state moves
    T residual_post[M];  ///< the same error after it moved
    double residual_norm;
    double residual_post_norm;
    double gain_norm;
    double p_trace;
    double p_min_diag;
};

/// Exponentially weighted RLS with a shared covariance across outputs.
///
/// `T` is the storage and arithmetic precision. fp16 is absent by construction
/// rather than by a check: it was measured losing positive-definiteness in this
/// recursion even with perfectly conditioned regressors, so there is no half
/// instantiation to select. See the precision note in argmind/adapt.py.
template <typename T, int N, int M = 1>
class Rls {
    static_assert(N >= 1, "n_features must be >= 1");
    static_assert(M >= 1, "n_outputs must be >= 1");
    static_assert(sizeof(T) >= 4,
                  "fp32 is the measured floor for this recursion; there is no "
                  "half-precision path in any filter form");

public:
    /// Exposed so that wrappers (the guard) can name the precision they are
    /// wrapping without being told it separately.
    using scalar = T;

    static constexpr int n_features = N;
    static constexpr int n_outputs = M;
    /// Elements in the persistence block. Fixed at compile time, by design.
    static constexpr int state_size = N * M + N * N;

    Rls() : lambda_(T(0.999)), inv_lambda_(T(1) / T(0.999)), p0_(T(100)) {
        reset();
    }

    /// `lambda` in (0, 1]; `p0` positive. Both are checked by `valid()` rather
    /// than by throwing, since this core has no exceptions: a caller that ignores
    /// `valid()` gets a defined object that reports itself unusable.
    Rls(T lambda, T p0)
        : lambda_(lambda), inv_lambda_(T(1) / lambda), p0_(p0) {
        reset();
    }

    bool valid() const {
        return lambda_ > T(0) && lambda_ <= T(1) && p0_ > T(0) &&
               is_finite(lambda_) && is_finite(p0_);
    }

    T lambda() const { return lambda_; }
    T p0() const { return p0_; }

    /// Roughly how many past samples still carry weight; infinite at lambda == 1,
    /// which is the honest answer rather than a large number.
    double memory_horizon_samples() const {
        const double l = static_cast<double>(lambda_);
        return l >= 1.0 ? __builtin_inf() : 1.0 / (1.0 - l);
    }

    void reset() {
        fill<T, N * M>(theta(), T(0));
        set_scaled_identity<T, N>(p(), p0_);
    }

    // -- state -------------------------------------------------------------

    /// The entire adaptation state, one flat contiguous block: theta then P.
    ///
    /// A single array with views into it, exactly as the Python reference does,
    /// rather than two adjacent members. Two members happen to be contiguous on
    /// every compiler anyone will use and the standard does not say so, and
    /// indexing past the first would be reaching outside its bounds. Persistence
    /// and the C ABI both depend on this block, so it is the one place worth
    /// being pedantic.
    const T* state() const { return state_; }
    T* state_mutable() { return state_; }

    const T* theta() const { return state_; }
    T* theta() { return state_; }
    const T* p() const { return state_ + N * M; }
    T* p() { return state_ + N * M; }

    /// Restore a block previously read from `state()`. Returns false and changes
    /// nothing if it is not finite: persisted state comes back from flash, which
    /// is exactly where a torn write or a bit flip shows up, and loading a NaN
    /// silently would poison every later update.
    bool load_state(const T* __restrict flat) {
        if (!all_finite<T, state_size>(flat)) return false;
        copy<T, state_size>(flat, state_);
        return true;
    }

    /// (p_trace, p_min_diag) without consuming a sample. The guard needs these
    /// where no update happened — at construction and while frozen — and a
    /// plausible-looking zero there would be a fabricated number.
    void health(double* __restrict p_trace, double* __restrict p_min_diag) const {
        *p_trace = trace_f64<T, N>(p());
        *p_min_diag = static_cast<double>(min_diagonal<T, N>(p()));
    }

    // -- the work ----------------------------------------------------------

    void predict(const T* __restrict x, T* __restrict out) const {
        vec_mat<T, N, M>(x, theta(), out);
    }

    Report<T, M> update(const T* __restrict x, const T* __restrict y) {
        Report<T, M> r;

        if (!all_finite<T, N>(x) || !all_finite<T, M>(y)) {
            fill<T, M>(r.residual, T(0));
            fill<T, M>(r.residual_post, T(0));
            finish(r, /*applied=*/false, Reason::NonFiniteInput, /*gain=*/T(0));
            return r;
        }

        // a priori residual, before the state moves. The only error figure that
        // means anything for monitoring: the a posteriori one always looks better
        // because the adapter has already seen the answer.
        T yhat[M];
        vec_mat<T, N, M>(x, theta(), yhat);
        for (int j = 0; j < M; ++j) r.residual[j] = y[j] - yhat[j];

        T px[N];
        mat_vec<T, N, N>(p(), x, px);
        const T den = lambda_ + dot<T, N>(x, px);

        if (!is_finite(den) || !(den > T(0))) {
            // Reachable when P has lost positive-definiteness, or when p0 and the
            // regressor scale overflow the declared precision. Either way the gain
            // would be meaningless; refuse rather than propagate it.
            copy<T, M>(r.residual, r.residual_post);
            finish(r, /*applied=*/false, Reason::NonPositiveDenominator, T(0));
            return r;
        }

        T k[N];
        const T inv_den = T(1) / den;
        scale_to<T, N>(px, inv_den, k);

        outer_add<T, N, M>(k, r.residual, theta());
        rank1_sub_scaled<T, N>(k, px, inv_lambda_, p());
        symmetrise<T, N>(p());

        vec_mat<T, N, M>(x, theta(), yhat);
        for (int j = 0; j < M; ++j) r.residual_post[j] = y[j] - yhat[j];

        ++updates_;
        finish(r, /*applied=*/true, Reason::None, norm<T, N>(k));
        return r;
    }

    uint32_t updates() const { return updates_; }
    uint32_t rejected_nonfinite() const { return rejected_nonfinite_; }
    uint32_t rejected_denominator() const { return rejected_denominator_; }

private:
    void finish(Report<T, M>& r, bool applied, Reason reason, T gain) {
        r.applied = applied;
        r.reason = reason;
        r.residual_norm = static_cast<double>(norm<T, M>(r.residual));
        r.residual_post_norm = static_cast<double>(norm<T, M>(r.residual_post));
        r.gain_norm = static_cast<double>(gain);
        health(&r.p_trace, &r.p_min_diag);
        if (reason == Reason::NonFiniteInput) ++rejected_nonfinite_;
        if (reason == Reason::NonPositiveDenominator) ++rejected_denominator_;
    }

    // One block, theta then P, so that `state()` hands out the whole thing and
    // persistence is a memcpy. The layout is shared with the Python reference and
    // with the on-flash format, so it is not free to change.
    T state_[state_size];

    T lambda_;
    T inv_lambda_;
    T p0_;
    uint32_t updates_ = 0;
    uint32_t rejected_nonfinite_ = 0;
    uint32_t rejected_denominator_ = 0;
};

}  // namespace argmind

#endif  // ARGMIND_RLS_HPP
