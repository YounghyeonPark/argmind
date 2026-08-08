// SPDX-License-Identifier: Apache-2.0
//
// The C ABI implementation: one macro, one line per instantiation.
//
// Everything here is plumbing. The rules it enforces are the ones the header
// promises: no allocation, no exceptions, no throwing, null and size checks at the
// boundary rather than trust, and a static assertion per instantiation that the
// declared upper bound is still an upper bound.
//
// Placement into caller storage uses a union-aligned cast rather than placement new
// so that <new> is not needed in a freestanding build. `Rls` is trivially
// destructible and has no virtuals, so constructing it by assignment into the
// caller's bytes is well defined for these types.

#include "argmind/argmind.h"

#include "argmind/rls.hpp"

namespace {

template <typename T, int M>
void fill_report(const argmind::Report<T, M>& src, am_report* dst,
                 T* residual_out) {
    dst->applied = src.applied ? 1 : 0;
    dst->reason = static_cast<int32_t>(src.reason);
    dst->residual_norm = src.residual_norm;
    dst->residual_post_norm = src.residual_post_norm;
    dst->gain_norm = src.gain_norm;
    dst->p_trace = src.p_trace;
    dst->p_min_diag = src.p_min_diag;
    if (residual_out) {
        for (int j = 0; j < M; ++j) residual_out[j] = src.residual[j];
    }
}

}  // namespace

#define AM_DEFINE_RLS(NAME, SCALAR, N, M, MAX_BYTES)                            \
    using Rls_##NAME = argmind::Rls<SCALAR, N, M>;                              \
    static_assert(sizeof(Rls_##NAME) <= (MAX_BYTES),                            \
                  "AM_RLS_" #NAME "_MAX_BYTES is no longer an upper bound on "  \
                  "sizeof(Rls); raise the macro in argmind.h");                 \
    static_assert(alignof(Rls_##NAME) <= alignof(am_rls_storage),               \
                  "am_rls_storage no longer satisfies this instantiation's "     \
                  "alignment");                                                 \
    static_assert(Rls_##NAME::state_size == (N) * (M) + (N) * (N),              \
                  "state layout changed; persist.py and the C ABI both depend "  \
                  "on theta (N*M) then P (N*N)");                               \
                                                                                \
    extern "C" size_t am_rls_##NAME##_bytes(void) {                             \
        return sizeof(Rls_##NAME);                                              \
    }                                                                           \
    extern "C" int am_rls_##NAME##_state_size(void) {                           \
        return Rls_##NAME::state_size;                                          \
    }                                                                           \
    extern "C" int am_rls_##NAME##_init(void* obj, size_t obj_bytes,            \
                                        SCALAR lambda, SCALAR p0) {             \
        if (!obj) return AM_ERR_NULL;                                           \
        if (obj_bytes < sizeof(Rls_##NAME)) return AM_ERR_TOO_SMALL;             \
        Rls_##NAME probe(lambda, p0);                                           \
        if (!probe.valid()) return AM_ERR_CONFIG;                               \
        *static_cast<Rls_##NAME*>(obj) = probe;                                 \
        return AM_OK;                                                           \
    }                                                                           \
    extern "C" int am_rls_##NAME##_reset(void* obj) {                           \
        if (!obj) return AM_ERR_NULL;                                           \
        static_cast<Rls_##NAME*>(obj)->reset();                                 \
        return AM_OK;                                                           \
    }                                                                           \
    extern "C" int am_rls_##NAME##_update(void* obj, const SCALAR* x,           \
                                          const SCALAR* y, am_report* report,   \
                                          SCALAR* residual_out) {               \
        if (!obj || !x || !y || !report) return AM_ERR_NULL;                     \
        fill_report<SCALAR, M>(static_cast<Rls_##NAME*>(obj)->update(x, y),      \
                               report, residual_out);                            \
        return AM_OK;                                                           \
    }                                                                           \
    extern "C" int am_rls_##NAME##_predict(const void* obj, const SCALAR* x,     \
                                           SCALAR* out) {                       \
        if (!obj || !x || !out) return AM_ERR_NULL;                              \
        static_cast<const Rls_##NAME*>(obj)->predict(x, out);                    \
        return AM_OK;                                                           \
    }                                                                           \
    extern "C" int am_rls_##NAME##_health(const void* obj, double* p_trace,      \
                                          double* p_min_diag) {                  \
        if (!obj || !p_trace || !p_min_diag) return AM_ERR_NULL;                 \
        static_cast<const Rls_##NAME*>(obj)->health(p_trace, p_min_diag);        \
        return AM_OK;                                                           \
    }                                                                           \
    extern "C" const SCALAR* am_rls_##NAME##_state(const void* obj) {            \
        return obj ? static_cast<const Rls_##NAME*>(obj)->state() : nullptr;     \
    }                                                                           \
    extern "C" int am_rls_##NAME##_load_state(void* obj, const SCALAR* flat) {    \
        if (!obj || !flat) return AM_ERR_NULL;                                   \
        return static_cast<Rls_##NAME*>(obj)->load_state(flat) ? AM_OK           \
                                                              : AM_ERR_STATE;    \
    }                                                                           \
    extern "C" uint32_t am_rls_##NAME##_updates(const void* obj) {               \
        return obj ? static_cast<const Rls_##NAME*>(obj)->updates() : 0u;        \
    }

AM_DEFINE_RLS(f32_6_1, float, 6, 1, AM_RLS_F32_6_1_MAX_BYTES)
AM_DEFINE_RLS(f32_6_3, float, 6, 3, AM_RLS_F32_6_3_MAX_BYTES)
AM_DEFINE_RLS(f32_12_1, float, 12, 1, AM_RLS_F32_12_1_MAX_BYTES)
AM_DEFINE_RLS(f64_6_1, double, 6, 1, AM_RLS_F64_6_1_MAX_BYTES)
