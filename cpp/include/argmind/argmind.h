/* SPDX-License-Identifier: Apache-2.0
 *
 * The C ABI. Valid C99; no C++ in this header.
 *
 * Two reasons this boundary exists, from DESIGN.md: a plain-C firmware, Python,
 * Rust or MATLAB can call in without knowing C++ is inside, and callers are
 * insulated from C++ ABI instability across compilers and versions, which is a
 * genuine hazard for a library shipped as a binary.
 *
 * The tension this header has to resolve
 * --------------------------------------
 * Compile-time dimensions are the reason C++ was chosen: they are what lets the
 * compiler unroll, keep operands in registers and vectorise. A C ABI wants runtime
 * arguments. Taking `n` at runtime would resolve the tension by throwing away the
 * reason for the choice, so it is resolved the other way: **the ABI exports an
 * explicit list of instantiations**, and adding a dimension means adding one line
 * to that list and rebuilding. That is a real cost and it is the honest one.
 *
 * Storage is the caller's
 * -----------------------
 * Nothing here allocates. `am_rls_*_bytes()` reports how much storage an instance
 * needs and the caller supplies it — statically, on its own stack, or from its own
 * pool.
 *
 * `AM_*_MAX_BYTES` is a compile-time **upper bound** for callers that must size an
 * array before they can call anything. It is not the exact size: that depends on
 * the C++ layout, which a C header cannot compute. The implementation carries a
 * static assertion that the real object fits inside the bound, so the bound cannot
 * silently become wrong, and `init()` rejects storage smaller than the true size.
 * Use `am_rls_*_bytes()` when a runtime value will do.
 *
 * Alignment: the storage must be aligned for double, i.e. 8 bytes. Use
 * `am_rls_storage` if in doubt.
 */

#ifndef ARGMIND_H
#define ARGMIND_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define AM_VERSION_MAJOR 0
#define AM_VERSION_MINOR 1

/* Return codes. Zero is success; negative is failure. Nothing here throws. */
#define AM_OK               0
#define AM_ERR_NULL        (-1)
#define AM_ERR_CONFIG      (-2)  /* lambda outside (0,1], p0 not positive       */
#define AM_ERR_STATE       (-3)  /* a state block that is not finite            */
#define AM_ERR_TOO_SMALL   (-4)  /* caller storage smaller than *_bytes()       */

/* Why an update did not apply. Matches argmind.adapt's report tokens. */
#define AM_REASON_NONE                    0
#define AM_REASON_NONFINITE_INPUT         1
#define AM_REASON_NONPOSITIVE_DENOMINATOR 2

/* What one update did. Fixed size regardless of the number of outputs, so that
 * this struct is stable ABI: the residual *vector* is retrieved through an
 * optional out-parameter instead of living here. The scalars are the ones a guard
 * is written against. */
typedef struct {
    int32_t applied;   /* 0 or 1; 0 means the state was not touched */
    int32_t reason;    /* AM_REASON_*                               */
    double residual_norm;
    double residual_post_norm;
    double gain_norm;
    double p_trace;
    double p_min_diag;
} am_report;

/* Storage with the right alignment for any instantiation below. Declare one of
 * these of the right size and pass it in:
 *
 *     static am_rls_storage store[AM_RLS_F32_6_1_MAX_BYTES / sizeof(am_rls_storage)];
 *     am_rls_f32_6_1_init(store, sizeof(store), 0.999f, 100.0f);
 */
typedef union {
    double d;
    void* p;
    uint64_t u;
} am_rls_storage;

/* --------------------------------------------------------------------------
 * The instantiation list.
 *
 * Each entry is (name, scalar type, N, M). Chosen for the design driver rather
 * than for completeness: six features with one output is the inertial + aiding
 * calibration case, three outputs is the same model against three bias axes, and
 * the f64 entry exists because the Python reference is verified at f64 and the
 * parity harness needs somewhere to meet it.
 *
 * AM_RLS_*_STATE is the element count of the persistence block, which is also the
 * layout argmind/persist.py stores: theta (N*M) then P (N*N), row-major.
 * -------------------------------------------------------------------------- */

#define AM_DECLARE_RLS(NAME, SCALAR)                                           \
    size_t am_rls_##NAME##_bytes(void);                                        \
    int am_rls_##NAME##_state_size(void);                                      \
    int am_rls_##NAME##_init(void* obj, size_t obj_bytes, SCALAR lambda,        \
                             SCALAR p0);                                       \
    int am_rls_##NAME##_reset(void* obj);                                      \
    int am_rls_##NAME##_update(void* obj, const SCALAR* x, const SCALAR* y,     \
                               am_report* report, SCALAR* residual_out);       \
    int am_rls_##NAME##_predict(const void* obj, const SCALAR* x,               \
                                SCALAR* out);                                  \
    int am_rls_##NAME##_health(const void* obj, double* p_trace,                \
                               double* p_min_diag);                            \
    const SCALAR* am_rls_##NAME##_state(const void* obj);                       \
    int am_rls_##NAME##_load_state(void* obj, const SCALAR* flat);              \
    uint32_t am_rls_##NAME##_updates(const void* obj);

#define AM_RLS_F32_6_1_STATE  (6 * 1 + 6 * 6)   /* 42 elements, 168 bytes */
#define AM_RLS_F32_6_3_STATE  (6 * 3 + 6 * 6)   /* 54 elements, 216 bytes */
#define AM_RLS_F32_12_1_STATE (12 * 1 + 12 * 12)
#define AM_RLS_F64_6_1_STATE  (6 * 1 + 6 * 6)

AM_DECLARE_RLS(f32_6_1, float)
AM_DECLARE_RLS(f32_6_3, float)
AM_DECLARE_RLS(f32_12_1, float)
AM_DECLARE_RLS(f64_6_1, double)

/* Compile-time upper bounds on instance size, for statically declared storage.
 * The implementation static-asserts that each real object fits, so these cannot
 * drift into being wrong without failing the build. */
#define AM_RLS_F32_6_1_MAX_BYTES  256
#define AM_RLS_F32_6_3_MAX_BYTES  320
#define AM_RLS_F32_12_1_MAX_BYTES 704
#define AM_RLS_F64_6_1_MAX_BYTES  448

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* ARGMIND_H */
