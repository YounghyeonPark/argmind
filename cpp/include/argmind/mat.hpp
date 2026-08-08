// SPDX-License-Identifier: Apache-2.0
//
// Minimal fixed-size linear algebra: exactly what the adapters need, nothing else.
//
// Why not Eigen: the target floor is Cortex-M4F (see DESIGN.md). Eigen is header
// bloat and a C++14+ requirement for a core that has to build freestanding, and at
// the dimensions this library lives at — 3 to 30 — it would buy nothing anyway.
// That last part is measured, not assumed: the Python reference's cost at n<=32 is
// call overhead rather than arithmetic, so what matters here is that the compiler
// can see the sizes and inline everything, which is what these types are for.
//
// Rules this file keeps, from DESIGN.md:
//   - dimensions are template parameters, so loops unroll and bounds vanish
//   - no allocation, no exceptions, no RTTI, no virtuals, nothing from libm
//   - pointers into distinct objects are marked __restrict so the optimiser may
//     assume no aliasing, which is a precondition for auto-vectorisation
//   - row-major throughout, matching the Python reference's numpy layout exactly,
//     because the state block is shared between them
//
// Auto-vectorisation is not assumed here. DESIGN.md requires it be verified with
// -fopt-info-vec or the emitted assembly rather than hoped for; that check belongs
// with the build, not with this header.

#ifndef ARGMIND_MAT_HPP
#define ARGMIND_MAT_HPP

#include <stddef.h>
#include <stdint.h>

namespace argmind {

// -- IEEE-754 predicates without libm ---------------------------------------
//
// A bit-pattern test rather than <cmath>: this header must be usable freestanding,
// and `v != v` style tricks are exactly what -ffast-math is allowed to delete.

inline bool is_finite(float v) {
    uint32_t bits;
    __builtin_memcpy(&bits, &v, sizeof(bits));
    return (bits & 0x7F800000u) != 0x7F800000u;
}

inline bool is_finite(double v) {
    uint64_t bits;
    __builtin_memcpy(&bits, &v, sizeof(bits));
    return (bits & 0x7FF0000000000000ull) != 0x7FF0000000000000ull;
}

// Maps to VSQRT.F32 on an M4F and to sqrtss on x86-64; no libm call either way.
inline float  sqrt_of(float v)  { return __builtin_sqrtf(v); }
inline double sqrt_of(double v) { return __builtin_sqrt(v); }

// -- primitives --------------------------------------------------------------
//
// Free functions over raw contiguous storage rather than an expression-template
// system. The operations the recursion needs are few and fixed, and writing them
// out is what lets the fused ones below be fused at all.

/// out = A * x, with A row-major R x C.
template <typename T, int R, int C>
inline void mat_vec(const T* __restrict a, const T* __restrict x,
                    T* __restrict out) {
    for (int i = 0; i < R; ++i) {
        T acc = T(0);
        const T* row = a + i * C;
        for (int j = 0; j < C; ++j) acc += row[j] * x[j];
        out[i] = acc;
    }
}

/// out = x^T * A, with A row-major R x C. out has C entries.
///
/// The accumulation runs over rows so that the inner loop walks a row of A
/// contiguously. It is also the order numpy's own reduction takes for this shape,
/// which matters only because the two implementations are compared.
template <typename T, int R, int C>
inline void vec_mat(const T* __restrict x, const T* __restrict a,
                    T* __restrict out) {
    for (int j = 0; j < C; ++j) out[j] = T(0);
    for (int i = 0; i < R; ++i) {
        const T xi = x[i];
        const T* row = a + i * C;
        for (int j = 0; j < C; ++j) out[j] += xi * row[j];
    }
}

template <typename T, int N>
inline T dot(const T* __restrict a, const T* __restrict b) {
    T acc = T(0);
    for (int i = 0; i < N; ++i) acc += a[i] * b[i];
    return acc;
}

template <typename T, int N>
inline T norm(const T* __restrict a) {
    return sqrt_of(dot<T, N>(a, a));
}

/// out[i] = a[i] * s
template <typename T, int N>
inline void scale_to(const T* __restrict a, T s, T* __restrict out) {
    for (int i = 0; i < N; ++i) out[i] = a[i] * s;
}

/// A += u (x) v, A row-major R x C.
template <typename T, int R, int C>
inline void outer_add(const T* __restrict u, const T* __restrict v,
                      T* __restrict a) {
    for (int i = 0; i < R; ++i) {
        const T ui = u[i];
        T* row = a + i * C;
        for (int j = 0; j < C; ++j) row[j] += ui * v[j];
    }
}

/// P <- (P - u (x) v) * s, in place, P row-major N x N.
///
/// `s` is a reciprocal supplied by the caller rather than a divisor. That is a
/// deliberate divergence from the Python reference, which divides: VDIV.F32 costs
/// 14 cycles on an M4F and this line runs N*N times, so at N=6 dividing would be
/// about 500 cycles for one statement against 14 plus 36 multiplies. The cost is
/// one extra rounding on the reciprocal, and it is declared here because it is a
/// real reason the two implementations cannot agree to the last bit.
template <typename T, int N>
inline void rank1_sub_scaled(const T* __restrict u, const T* __restrict v, T s,
                             T* __restrict p) {
    for (int i = 0; i < N; ++i) {
        const T ui = u[i];
        T* row = p + i * N;
        for (int j = 0; j < N; ++j) row[j] = (row[j] - ui * v[j]) * s;
    }
}

/// P <- (P + P^T) * 0.5, in place.
///
/// The rank-1 subtraction above is symmetric in exact arithmetic and is not in
/// floating point, and the asymmetry compounds. Both halves of each pair are read
/// before either is written, so this needs no scratch. It does not restore
/// positive-definiteness, which is why the caller reports the smallest diagonal.
template <typename T, int N>
inline void symmetrise(T* __restrict p) {
    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            const T s = (p[i * N + j] + p[j * N + i]) * T(0.5);
            p[i * N + j] = s;
            p[j * N + i] = s;
        }
    }
}

/// Trace, accumulated in double whatever T is.
///
/// Deliberate: the Python reference computes `trace(P.astype(float64))`, so at f32
/// storage the sum is still taken in double. A float accumulation here would
/// disagree with it for reasons that have nothing to do with the recursion.
template <typename T, int N>
inline double trace_f64(const T* __restrict p) {
    double acc = 0.0;
    for (int i = 0; i < N; ++i) acc += static_cast<double>(p[i * N + i]);
    return acc;
}

template <typename T, int N>
inline T min_diagonal(const T* __restrict p) {
    T m = p[0];
    for (int i = 1; i < N; ++i) {
        const T d = p[i * N + i];
        if (d < m) m = d;
    }
    return m;
}

template <typename T, int N>
inline void fill(T* __restrict a, T v) {
    for (int i = 0; i < N; ++i) a[i] = v;
}

/// A <- v * I, row-major N x N.
template <typename T, int N>
inline void set_scaled_identity(T* __restrict a, T v) {
    for (int i = 0; i < N * N; ++i) a[i] = T(0);
    for (int i = 0; i < N; ++i) a[i * N + i] = v;
}

template <typename T, int N>
inline bool all_finite(const T* __restrict a) {
    for (int i = 0; i < N; ++i) {
        if (!is_finite(a[i])) return false;
    }
    return true;
}

template <typename T, int N>
inline void copy(const T* __restrict src, T* __restrict dst) {
    for (int i = 0; i < N; ++i) dst[i] = src[i];
}

}  // namespace argmind

#endif  // ARGMIND_MAT_HPP
