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

#include <stddef.h>

#include "argmind/guard.hpp"
#include "argmind/persist.hpp"
#include "argmind/pipeline.hpp"
#include "argmind/rls.hpp"
#include "argmind/stream.hpp"

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

// -- the guard --------------------------------------------------------------------------
//
// Depth and MaxBounds are fixed here, not per instantiation, for the reason the header
// gives: they are the memory and they would otherwise be two more axes in the list.

namespace {

constexpr int kGuardDepth = 4;
constexpr int kGuardBounds = 4;

template <typename T, int M>
void fill_guard_report(const argmind::GuardReport<T, M>& src, am_guard_report* dst,
                       T* residual_out) {
    dst->applied = src.applied ? 1 : 0;
    dst->reason = static_cast<int32_t>(src.reason);
    dst->residual_norm = src.residual_norm;
    dst->residual_post_norm = src.residual_post_norm;
    dst->gain_norm = src.gain_norm;
    dst->p_trace = src.p_trace;
    dst->p_min_diag = src.p_min_diag;
    dst->frozen = src.frozen ? 1 : 0;
    dst->rolled_back = src.rolled_back ? 1 : 0;
    dst->tripped = src.tripped;
    dst->guard_reason = static_cast<int32_t>(src.guard_reason);
    if (residual_out) {
        for (int j = 0; j < M; ++j) residual_out[j] = src.residual[j];
    }
}

/// Reject a field code that is not one of the five, rather than casting it into the enum
/// and getting a check that watches nothing. The C++ side cannot make this mistake — the
/// enum is the parameter — and the C side can, which is exactly what a boundary is for.
bool valid_field(int32_t field) {
    return field >= 0 && field <= static_cast<int32_t>(argmind::Field::PMinDiag);
}

}  // namespace

// One entry point below carries an **equivalent mutant**, recorded rather than tallied
// because the distinction is worth keeping. `am_guard_*_state(nullptr)` tests for null
// before dereferencing; removing that test survives the whole suite. It is not an untested
// guard, it is an undetectable one: measured, `state()` sits at offset 0 of the guard,
// because `state_` is the first member of `Rls` and `Rls` is the first member of `Guard`,
// so the mutant returns null for a null argument too and no test can separate them. It
// stays because dereferencing a null pointer is undefined whatever the address arithmetic
// happens to produce, and because that offset is a layout accident rather than a
// guarantee. `am_rls_*_state` has the same shape for the same reason.

#define AM_DEFINE_GUARD(NAME, SCALAR, N, M, MAX_BYTES)                          \
    using Guard_##NAME =                                                        \
        argmind::Guard<argmind::Rls<SCALAR, N, M>, kGuardDepth, kGuardBounds>;   \
    static_assert(sizeof(Guard_##NAME) <= (MAX_BYTES),                          \
                  "AM_GUARD_" #NAME "_MAX_BYTES is no longer an upper bound; "  \
                  "raise the macro in argmind.h");                              \
    static_assert(alignof(Guard_##NAME) <= alignof(am_rls_storage),             \
                  "am_rls_storage no longer satisfies this guard's alignment"); \
                                                                                \
    extern "C" size_t am_guard_##NAME##_bytes(void) {                           \
        return sizeof(Guard_##NAME);                                            \
    }                                                                           \
    extern "C" int am_guard_##NAME##_state_size(void) {                         \
        return Guard_##NAME::state_size;                                        \
    }                                                                           \
    extern "C" int am_guard_##NAME##_checkpoint_bytes(void) {                   \
        return Guard_##NAME::checkpoint_bytes;                                  \
    }                                                                           \
    extern "C" int am_guard_##NAME##_init(void* obj, size_t obj_bytes,          \
                                          SCALAR lambda, SCALAR p0,             \
                                          uint32_t checkpoint_every,            \
                                          uint32_t recover_after) {             \
        if (!obj) return AM_ERR_NULL;                                           \
        if (obj_bytes < sizeof(Guard_##NAME)) return AM_ERR_TOO_SMALL;          \
        argmind::Rls<SCALAR, N, M> inner(lambda, p0);                           \
        if (!inner.valid()) return AM_ERR_CONFIG;                               \
        Guard_##NAME probe(inner, checkpoint_every, recover_after);             \
        *static_cast<Guard_##NAME*>(obj) = probe;                               \
        return AM_OK;                                                           \
    }                                                                           \
    extern "C" int am_guard_##NAME##_add_bound(void* obj, int32_t field,        \
                                               int32_t has_lo, double lo,       \
                                               int32_t has_hi, double hi) {     \
        if (!obj) return AM_ERR_NULL;                                           \
        if (!valid_field(field)) return AM_ERR_CONFIG;                          \
        argmind::Bound b;                                                       \
        b.field = static_cast<argmind::Field>(field);                           \
        b.has_lo = has_lo != 0;                                                 \
        b.has_hi = has_hi != 0;                                                 \
        b.lo = lo;                                                              \
        b.hi = hi;                                                              \
        if (!b.valid()) return AM_ERR_CONFIG;                                   \
        return static_cast<Guard_##NAME*>(obj)->add_bound(b) ? AM_OK            \
                                                             : AM_ERR_FULL;     \
    }                                                                           \
    extern "C" int am_guard_##NAME##_set_divergence(                            \
        void* obj, double ratio, double fast_alpha, double slow_alpha,          \
        double floor, uint32_t warmup) {                                        \
        if (!obj) return AM_ERR_NULL;                                           \
        argmind::Divergence d;                                                  \
        d.enabled = true;                                                       \
        d.ratio = ratio;                                                        \
        d.fast_alpha = fast_alpha;                                              \
        d.slow_alpha = slow_alpha;                                              \
        d.floor = floor;                                                        \
        d.warmup = warmup;                                                      \
        if (!d.valid()) return AM_ERR_CONFIG;                                   \
        static_cast<Guard_##NAME*>(obj)->set_divergence(d);                     \
        return AM_OK;                                                           \
    }                                                                           \
    extern "C" int am_guard_##NAME##_valid(const void* obj) {                   \
        if (!obj) return AM_ERR_NULL;                                           \
        return static_cast<const Guard_##NAME*>(obj)->valid() ? AM_OK           \
                                                              : AM_ERR_UNUSABLE;\
    }                                                                           \
    extern "C" int am_guard_##NAME##_reset(void* obj) {                         \
        if (!obj) return AM_ERR_NULL;                                           \
        static_cast<Guard_##NAME*>(obj)->reset();                               \
        return AM_OK;                                                           \
    }                                                                           \
    extern "C" int am_guard_##NAME##_update(void* obj, const SCALAR* x,         \
                                            const SCALAR* y,                    \
                                            am_guard_report* report,            \
                                            SCALAR* residual_out) {             \
        if (!obj || !x || !y || !report) return AM_ERR_NULL;                    \
        fill_guard_report<SCALAR, M>(                                           \
            static_cast<Guard_##NAME*>(obj)->update(x, y), report, residual_out);\
        return AM_OK;                                                           \
    }                                                                           \
    extern "C" int am_guard_##NAME##_predict(const void* obj, const SCALAR* x,  \
                                             SCALAR* out) {                     \
        if (!obj || !x || !out) return AM_ERR_NULL;                             \
        static_cast<const Guard_##NAME*>(obj)->predict(x, out);                 \
        return AM_OK;                                                           \
    }                                                                           \
    extern "C" int am_guard_##NAME##_health(const void* obj, double* p_trace,   \
                                            double* p_min_diag) {               \
        if (!obj || !p_trace || !p_min_diag) return AM_ERR_NULL;                \
        static_cast<const Guard_##NAME*>(obj)->health(p_trace, p_min_diag);     \
        return AM_OK;                                                           \
    }                                                                           \
    extern "C" const SCALAR* am_guard_##NAME##_state(const void* obj) {         \
        return obj ? static_cast<const Guard_##NAME*>(obj)->state() : nullptr;  \
    }                                                                           \
    extern "C" int am_guard_##NAME##_load_state(void* obj, const SCALAR* flat) { \
        if (!obj || !flat) return AM_ERR_NULL;                                  \
        return static_cast<Guard_##NAME*>(obj)->load_state(flat) ? AM_OK        \
                                                                 : AM_ERR_STATE;\
    }                                                                           \
    extern "C" int am_guard_##NAME##_freeze(void* obj) {                        \
        if (!obj) return AM_ERR_NULL;                                           \
        static_cast<Guard_##NAME*>(obj)->freeze();                              \
        return AM_OK;                                                           \
    }                                                                           \
    extern "C" int am_guard_##NAME##_thaw(void* obj) {                          \
        if (!obj) return AM_ERR_NULL;                                           \
        static_cast<Guard_##NAME*>(obj)->thaw();                                \
        return AM_OK;                                                           \
    }                                                                           \
    extern "C" int am_guard_##NAME##_frozen(const void* obj) {                  \
        return obj && static_cast<const Guard_##NAME*>(obj)->frozen() ? 1 : 0;  \
    }                                                                           \
    extern "C" uint32_t am_guard_##NAME##_frozen_by(const void* obj) {          \
        return obj ? static_cast<const Guard_##NAME*>(obj)->frozen_by() : 0u;   \
    }                                                                           \
    extern "C" int am_guard_##NAME##_counters(const void* obj,                  \
                                              am_guard_counters* out) {         \
        if (!obj || !out) return AM_ERR_NULL;                                   \
        const Guard_##NAME* g = static_cast<const Guard_##NAME*>(obj);          \
        out->applied = g->applied();                                            \
        out->trips = g->trips();                                                \
        out->rollbacks = g->rollbacks();                                        \
        out->blocked_frozen = g->blocked_frozen();                              \
        out->rejected_by_adapter = g->rejected_by_adapter();                    \
        out->thaws = g->thaws();                                                \
        return AM_OK;                                                           \
    }

AM_DEFINE_GUARD(f32_6_1, float, 6, 1, AM_GUARD_F32_6_1_MAX_BYTES)
AM_DEFINE_GUARD(f32_6_3, float, 6, 3, AM_GUARD_F32_6_3_MAX_BYTES)
AM_DEFINE_GUARD(f32_12_1, float, 12, 1, AM_GUARD_F32_12_1_MAX_BYTES)
AM_DEFINE_GUARD(f64_6_1, double, 6, 1, AM_GUARD_F64_6_1_MAX_BYTES)

// -- the record format ------------------------------------------------------------------

extern "C" size_t am_persist_record_bytes(uint32_t count, uint32_t element_size,
                                          uint32_t cause_len) {
    return argmind::persist::record_bytes(count, element_size, cause_len);
}

extern "C" uint32_t am_persist_crc32(const uint8_t* data, size_t len) {
    return data ? argmind::persist::crc32(data, len) : 0u;
}

extern "C" int am_persist_next_slot(int newest) {
    return argmind::persist::next_slot(newest);
}

namespace {

template <typename T>
void fill_record(const argmind::persist::Record<T>& src, am_record* dst) {
    dst->cause = src.cause;
    dst->sequence = src.sequence;
    dst->fingerprint = src.fingerprint;
    dst->count = src.count;
    dst->cause_len = src.cause_len;
    dst->version = src.version;
    dst->n_features = src.n_features;
    dst->n_outputs = src.n_outputs;
    dst->frozen = src.frozen ? 1u : 0u;
}

}  // namespace

#define AM_DEFINE_PERSIST(NAME, SCALAR)                                         \
    extern "C" size_t am_persist_pack_##NAME(                                   \
        const SCALAR* state, uint32_t count, uint16_t n_features,               \
        uint16_t n_outputs, uint32_t sequence, uint32_t fingerprint,            \
        int32_t frozen, const char* cause, size_t cause_len, uint8_t* out,      \
        size_t out_bytes) {                                                     \
        return argmind::persist::pack<SCALAR>(state, count, n_features,         \
                                              n_outputs, sequence, fingerprint, \
                                              frozen != 0, cause, cause_len,    \
                                              out, out_bytes);                  \
    }                                                                           \
    extern "C" int am_persist_unpack_##NAME(const uint8_t* in, size_t in_bytes,  \
                                            am_record* rec) {                   \
        if (!rec) return AM_PERSIST_TOO_SHORT;                                  \
        argmind::persist::Record<SCALAR> r;                                     \
        const argmind::persist::Status st =                                     \
            argmind::persist::unpack<SCALAR>(in, in_bytes, &r);                 \
        if (st == argmind::persist::Status::Ok) fill_record<SCALAR>(r, rec);    \
        return static_cast<int>(st);                                            \
    }                                                                           \
    extern "C" int am_persist_read_state_##NAME(const uint8_t* in,              \
                                                size_t in_bytes, SCALAR* out,   \
                                                uint32_t out_count) {           \
        if (!out) return AM_ERR_NULL;                                           \
        argmind::persist::Record<SCALAR> r;                                     \
        const argmind::persist::Status st =                                     \
            argmind::persist::unpack<SCALAR>(in, in_bytes, &r);                 \
        if (st != argmind::persist::Status::Ok) return static_cast<int>(st);    \
        return argmind::persist::read_state<SCALAR>(r, out, out_count)          \
                   ? AM_PERSIST_OK                                              \
                   : AM_PERSIST_LENGTH_MISMATCH;                                \
    }                                                                           \
    extern "C" int am_persist_select_newest_##NAME(                             \
        const uint8_t* a, size_t a_bytes, const uint8_t* b, size_t b_bytes,     \
        am_record* out, int32_t* status_a, int32_t* status_b) {                 \
        argmind::persist::Record<SCALAR> r;                                     \
        argmind::persist::Status sa = argmind::persist::Status::TooShort;       \
        argmind::persist::Status sb = argmind::persist::Status::TooShort;       \
        const int which = argmind::persist::select_newest<SCALAR>(              \
            a, a_bytes, b, b_bytes, &r, &sa, &sb);                              \
        if (status_a) *status_a = static_cast<int32_t>(sa);                     \
        if (status_b) *status_b = static_cast<int32_t>(sb);                     \
        if (which >= 0 && out) fill_record<SCALAR>(r, out);                     \
        return which;                                                           \
    }

AM_DEFINE_PERSIST(f32, float)
AM_DEFINE_PERSIST(f64, double)

// The status codes are two enums in two languages and they have to agree, so say so once
// here rather than trusting that nobody renumbers one of them.
static_assert(static_cast<int>(argmind::persist::Status::Ok) == AM_PERSIST_OK, "");
static_assert(static_cast<int>(argmind::persist::Status::TooShort) ==
                  AM_PERSIST_TOO_SHORT, "");
static_assert(static_cast<int>(argmind::persist::Status::BadMagic) ==
                  AM_PERSIST_BAD_MAGIC, "");
static_assert(static_cast<int>(argmind::persist::Status::FutureVersion) ==
                  AM_PERSIST_FUTURE_VERSION, "");
static_assert(static_cast<int>(argmind::persist::Status::BadDtype) ==
                  AM_PERSIST_BAD_DTYPE, "");
static_assert(static_cast<int>(argmind::persist::Status::LengthMismatch) ==
                  AM_PERSIST_LENGTH_MISMATCH, "");
static_assert(static_cast<int>(argmind::persist::Status::CrcMismatch) ==
                  AM_PERSIST_CRC_MISMATCH, "");
static_assert(static_cast<int>(argmind::Field::PMinDiag) == AM_FIELD_P_MIN_DIAG, "");
static_assert(static_cast<int>(argmind::GuardReason::Tripped) == AM_GUARD_TRIPPED, "");
static_assert(argmind::DIVERGENCE_BIT == AM_TRIPPED_DIVERGENCE, "");

// -- caller-owned rings -----------------------------------------------------------------
//
// `am_ring_*` is layout-identical to `argmind::Ring<T>` and is reinterpreted rather than
// copied field by field, so the two cannot drift into disagreeing about what a ring is. The
// assertions below are what makes that reinterpretation legal to rely on rather than a habit
// that happens to work on this compiler.

#define AM_ASSERT_RING_LAYOUT(CNAME, SCALAR)                                    \
    static_assert(sizeof(CNAME) == sizeof(argmind::Ring<SCALAR>),               \
                  #CNAME " has drifted from argmind::Ring");                    \
    static_assert(alignof(CNAME) == alignof(argmind::Ring<SCALAR>), "");        \
    static_assert(offsetof(CNAME, values) ==                                    \
                      offsetof(argmind::Ring<SCALAR>, values), "");             \
    static_assert(offsetof(CNAME, times) ==                                     \
                      offsetof(argmind::Ring<SCALAR>, times), "");              \
    static_assert(offsetof(CNAME, valid) ==                                     \
                      offsetof(argmind::Ring<SCALAR>, valid), "");              \
    static_assert(offsetof(CNAME, dim) == offsetof(argmind::Ring<SCALAR>, dim), \
                  "");                                                          \
    static_assert(offsetof(CNAME, capacity) ==                                  \
                      offsetof(argmind::Ring<SCALAR>, capacity), "");           \
    static_assert(offsetof(CNAME, head) ==                                      \
                      offsetof(argmind::Ring<SCALAR>, head), "");               \
    static_assert(offsetof(CNAME, count) ==                                     \
                      offsetof(argmind::Ring<SCALAR>, count), "");              \
    static_assert(offsetof(CNAME, dropped) ==                                   \
                      offsetof(argmind::Ring<SCALAR>, dropped), "");            \
    static_assert(offsetof(CNAME, last_t) ==                                    \
                      offsetof(argmind::Ring<SCALAR>, last_t), "");

AM_ASSERT_RING_LAYOUT(am_ring_f32, float)
AM_ASSERT_RING_LAYOUT(am_ring_f64, double)

#define AM_DEFINE_RING(NAME, SCALAR)                                            \
    extern "C" int am_ring_##NAME##_init(am_ring_##NAME* r, SCALAR* values,      \
                                         double* times, uint8_t* valid,          \
                                         int32_t dim, int32_t capacity) {       \
        if (!r || !values || !times || !valid) return AM_ERR_NULL;               \
        if (dim < 1 || capacity < 1) return AM_ERR_CONFIG;                       \
        r->values = values;                                                      \
        r->times = times;                                                        \
        r->valid = valid;                                                        \
        r->dim = dim;                                                            \
        r->capacity = capacity;                                                  \
        r->head = 0;                                                             \
        r->count = 0;                                                            \
        r->dropped = 0;                                                          \
        r->last_t = -__builtin_inf();                                            \
        return AM_OK;                                                            \
    }                                                                            \
    extern "C" int am_ring_##NAME##_reset(am_ring_##NAME* r) {                   \
        if (!r) return AM_ERR_NULL;                                              \
        r->head = 0;                                                             \
        r->count = 0;                                                            \
        r->dropped = 0;                                                          \
        r->last_t = -__builtin_inf();                                            \
        return AM_OK;                                                            \
    }                                                                            \
    extern "C" int am_ring_##NAME##_push(am_ring_##NAME* r, double t,            \
                                         const SCALAR* values, int32_t valid) {  \
        if (!r || !values || !r->values) return AM_ERR_NULL;                     \
        return argmind::ring_push<SCALAR>(                                       \
                   reinterpret_cast<argmind::Ring<SCALAR>*>(r), t, values,       \
                   valid != 0)                                                   \
                   ? AM_OK                                                       \
                   : AM_ERR_CONFIG;                                              \
    }

AM_DEFINE_RING(f32, float)
AM_DEFINE_RING(f64, double)

// -- the composed pipeline --------------------------------------------------------------

namespace {

constexpr int kPipeStreams = AM_MAX_STREAMS;

static_assert(AM_SKIP_COUNT == argmind::SKIP_REASON_COUNT,
              "the C skip-reason list and SkipReason have diverged");
static_assert(static_cast<int>(argmind::SkipReason::Rejected) == AM_SKIP_REJECTED, "");
static_assert(static_cast<int>(argmind::SkipReason::StaleTarget) == AM_SKIP_STALE_TARGET,
              "");
static_assert(static_cast<int>(argmind::Policy::DropIfStale) == AM_POLICY_DROP_IF_STALE,
              "");

template <typename Pipe, typename Ring>
int views_from(const Ring* rings, int32_t n, argmind::StreamView<typename Pipe::T>* out) {
    if (!rings || n < 1 || n > kPipeStreams) return AM_ERR_CONFIG;
    for (int32_t i = 0; i < n; ++i) {
        if (!rings[i].values || !rings[i].times || !rings[i].valid) return AM_ERR_NULL;
        out[i] = argmind::ring_view<typename Pipe::T>(
            *reinterpret_cast<const argmind::Ring<typename Pipe::T>*>(&rings[i]));
    }
    return AM_OK;
}

template <typename Pipe>
void fill_step(const typename Pipe::Step& src, am_step_report* dst) {
    dst->ok = src.ok ? 1 : 0;
    dst->frozen = src.frozen ? 1 : 0;
    dst->has_t = src.has_t ? 1 : 0;
    dst->n_staleness = src.n_staleness;
    dst->frames = src.frames;
    dst->updated = src.updated;
    dst->lost_frames = src.lost_frames;
    dst->frozen_by = src.frozen_by;
    for (int i = 0; i < AM_SKIP_COUNT; ++i) dst->skipped[i] = src.skipped[i];
    dst->pad = 0;
    dst->worst_staleness = src.worst_staleness;
    dst->t = src.t;
    for (int i = 0; i < AM_MAX_STREAMS; ++i) dst->staleness[i] = src.staleness[i];
    dst->has_prediction = src.has_prediction ? 1 : 0;
    dst->n_outputs = Pipe::n_outputs;
    // Widened to double so the report is one shape across every instantiation. Exact for
    // f32, and the alternative -- a scalar-typed report per instantiation -- would put the
    // scalar in the struct name and make a caller's code depend on which pipeline it holds.
    for (int i = 0; i < AM_MAX_OUTPUTS; ++i) {
        dst->prediction[i] =
            i < Pipe::n_outputs ? static_cast<double>(src.prediction[i]) : 0.0;
    }
}

bool valid_policy(int32_t p) {
    return p >= AM_POLICY_HOLD_LAST && p <= AM_POLICY_DROP_IF_STALE;
}

}  // namespace

#define AM_DEFINE_PIPE_COMMON(NAME, SCALAR, RING, PIPE, MAX_BYTES)              \
    static_assert(sizeof(PIPE) <= (MAX_BYTES),                                  \
                  "AM_PIPE_" #NAME "_MAX_BYTES is no longer an upper bound; "   \
                  "raise the macro in argmind.h");                              \
    static_assert(alignof(PIPE) <= alignof(am_rls_storage), "");                \
    static_assert(PIPE::max_streams == AM_MAX_STREAMS, "");                     \
    static_assert(PIPE::n_outputs <= AM_MAX_OUTPUTS,                             \
                  "am_step_report cannot carry this instantiation's outputs; "   \
                  "raise AM_MAX_OUTPUTS in argmind.h");                          \
                                                                                \
    extern "C" size_t am_pipe_##NAME##_bytes(void) { return sizeof(PIPE); }     \
    extern "C" int am_pipe_##NAME##_n_features(void) {                          \
        return PIPE::n_features;                                                \
    }                                                                           \
    extern "C" int am_pipe_##NAME##_state_size(void) {                          \
        return PIPE::n_features * PIPE::n_outputs +                             \
               PIPE::n_features * PIPE::n_features;                             \
    }                                                                           \
    extern "C" int am_pipe_##NAME##_add_feature(void* obj, int32_t stream,       \
                                                int32_t component) {            \
        if (!obj) return AM_ERR_NULL;                                           \
        return static_cast<PIPE*>(obj)->map_mutable().add_feature(stream,        \
                                                                 component)      \
                   ? AM_OK                                                      \
                   : AM_ERR_FULL;                                               \
    }                                                                           \
    extern "C" int am_pipe_##NAME##_add_target(void* obj, int32_t stream,        \
                                               int32_t component) {             \
        if (!obj) return AM_ERR_NULL;                                           \
        return static_cast<PIPE*>(obj)->map_mutable().add_target(stream,         \
                                                                component)       \
                   ? AM_OK                                                      \
                   : AM_ERR_FULL;                                               \
    }                                                                           \
    extern "C" int am_pipe_##NAME##_set_constant(void* obj, int32_t on) {        \
        if (!obj) return AM_ERR_NULL;                                           \
        static_cast<PIPE*>(obj)->map_mutable().set_constant(on != 0);           \
        return AM_OK;                                                           \
    }                                                                           \
    extern "C" int am_pipe_##NAME##_set_max_stale_s(void* obj, int32_t enable,   \
                                                    double limit) {             \
        if (!obj) return AM_ERR_NULL;                                           \
        if (enable) {                                                           \
            static_cast<PIPE*>(obj)->set_max_stale_s(limit);                    \
        } else {                                                                \
            static_cast<PIPE*>(obj)->clear_max_stale_s();                       \
        }                                                                       \
        return AM_OK;                                                           \
    }                                                                           \
    extern "C" int am_pipe_##NAME##_set_require_valid(void* obj, int32_t on) {   \
        if (!obj) return AM_ERR_NULL;                                           \
        static_cast<PIPE*>(obj)->set_require_valid(on != 0);                    \
        return AM_OK;                                                           \
    }                                                                           \
    extern "C" int am_pipe_##NAME##_set_allow_non_causal(void* obj, int32_t on) {\
        if (!obj) return AM_ERR_NULL;                                           \
        static_cast<PIPE*>(obj)->set_allow_non_causal(on != 0);                 \
        return AM_OK;                                                           \
    }                                                                           \
    extern "C" int am_pipe_##NAME##_set_require_fresh_target(void* obj,          \
                                                             int32_t on) {      \
        if (!obj) return AM_ERR_NULL;                                           \
        static_cast<PIPE*>(obj)->set_require_fresh_target(on != 0);             \
        return AM_OK;                                                           \
    }                                                                           \
    extern "C" int am_pipe_##NAME##_configure(void* obj, const RING* rings,      \
                                              int32_t n) {                      \
        if (!obj) return AM_ERR_NULL;                                           \
        argmind::StreamView<SCALAR> views[kPipeStreams];                        \
        const int rc = views_from<PIPE, RING>(rings, n, views);                 \
        if (rc != AM_OK) return rc;                                             \
        return static_cast<PIPE*>(obj)->configure(views, n) ? AM_OK             \
                                                            : AM_ERR_CONFIG;    \
    }                                                                           \
    extern "C" int am_pipe_##NAME##_step(void* obj, const RING* rings, int32_t n,\
                                         am_step_report* out) {                 \
        if (!obj || !out) return AM_ERR_NULL;                                   \
        argmind::StreamView<SCALAR> views[kPipeStreams];                        \
        const int rc = views_from<PIPE, RING>(rings, n, views);                 \
        if (rc != AM_OK) return rc;                                             \
        PIPE* p = static_cast<PIPE*>(obj);                                      \
        if (!p->configured()) return AM_ERR_UNCONFIGURED;                       \
        fill_step<PIPE>(p->step(views), out);                                   \
        return AM_OK;                                                           \
    }                                                                           \
    extern "C" int am_pipe_##NAME##_predict(const void* obj, const SCALAR* x,    \
                                            SCALAR* out) {                      \
        if (!obj || !x || !out) return AM_ERR_NULL;                             \
        static_cast<const PIPE*>(obj)->predict(x, out);                         \
        return AM_OK;                                                           \
    }                                                                           \
    extern "C" const SCALAR* am_pipe_##NAME##_state(const void* obj) {           \
        return obj ? static_cast<const PIPE*>(obj)->adapter().state() : nullptr; \
    }                                                                           \
    extern "C" int am_pipe_##NAME##_load_state(void* obj, const SCALAR* flat) {  \
        if (!obj || !flat) return AM_ERR_NULL;                                  \
        return static_cast<PIPE*>(obj)->adapter_mutable().load_state(flat)       \
                   ? AM_OK                                                      \
                   : AM_ERR_STATE;                                              \
    }                                                                           \
    extern "C" int am_pipe_##NAME##_frozen(const void* obj) {                    \
        return obj && static_cast<const PIPE*>(obj)->frozen() ? 1 : 0;          \
    }

#define AM_DEFINE_PIPE_GUARDED(NAME, SCALAR, N, M, RING, MAX_BYTES)             \
    using Adapter_##NAME =                                                      \
        argmind::Guard<argmind::Rls<SCALAR, N, M>, kGuardDepth, kGuardBounds>;   \
    using Pipe_##NAME = argmind::Pipeline<Adapter_##NAME, AM_MAX_STREAMS>;       \
    extern "C" int am_pipe_##NAME##_init(void* obj, size_t obj_bytes,            \
                                         SCALAR lambda, SCALAR p0,              \
                                         uint32_t checkpoint_every,             \
                                         uint32_t recover_after, int32_t policy, \
                                         double max_stale_s, int32_t reference) {\
        if (!obj) return AM_ERR_NULL;                                           \
        if (obj_bytes < sizeof(Pipe_##NAME)) return AM_ERR_TOO_SMALL;            \
        if (!valid_policy(policy)) return AM_ERR_CONFIG;                        \
        argmind::Rls<SCALAR, N, M> inner(lambda, p0);                           \
        if (!inner.valid()) return AM_ERR_CONFIG;                               \
        const Adapter_##NAME g(inner, checkpoint_every, recover_after);          \
        const argmind::Aligner al(static_cast<argmind::Policy>(policy),          \
                                  max_stale_s);                                 \
        *static_cast<Pipe_##NAME*>(obj) =                                       \
            Pipe_##NAME(g, al, argmind::FrameMap<N, M>(), reference);            \
        return AM_OK;                                                           \
    }                                                                           \
    extern "C" int am_pipe_##NAME##_add_bound(void* obj, int32_t field,          \
                                              int32_t has_lo, double lo,         \
                                              int32_t has_hi, double hi) {      \
        if (!obj) return AM_ERR_NULL;                                           \
        if (!valid_field(field)) return AM_ERR_CONFIG;                          \
        argmind::Bound b;                                                       \
        b.field = static_cast<argmind::Field>(field);                           \
        b.has_lo = has_lo != 0;                                                 \
        b.has_hi = has_hi != 0;                                                 \
        b.lo = lo;                                                              \
        b.hi = hi;                                                              \
        if (!b.valid()) return AM_ERR_CONFIG;                                   \
        return static_cast<Pipe_##NAME*>(obj)->adapter_mutable().add_bound(b)    \
                   ? AM_OK                                                      \
                   : AM_ERR_FULL;                                               \
    }                                                                           \
    extern "C" int am_pipe_##NAME##_set_divergence(                             \
        void* obj, double ratio, double fast_alpha, double slow_alpha,          \
        double floor, uint32_t warmup) {                                        \
        if (!obj) return AM_ERR_NULL;                                           \
        argmind::Divergence d;                                                  \
        d.enabled = true;                                                       \
        d.ratio = ratio;                                                        \
        d.fast_alpha = fast_alpha;                                              \
        d.slow_alpha = slow_alpha;                                              \
        d.floor = floor;                                                        \
        d.warmup = warmup;                                                      \
        if (!d.valid()) return AM_ERR_CONFIG;                                   \
        static_cast<Pipe_##NAME*>(obj)->adapter_mutable().set_divergence(d);     \
        return AM_OK;                                                           \
    }                                                                           \
    extern "C" int am_pipe_##NAME##_frozen_by(const void* obj, uint32_t* out) {  \
        if (!obj || !out) return AM_ERR_NULL;                                   \
        *out = static_cast<const Pipe_##NAME*>(obj)->adapter().frozen_by();      \
        return AM_OK;                                                           \
    }                                                                           \
    extern "C" int am_pipe_##NAME##_counters(const void* obj,                    \
                                             am_guard_counters* out) {          \
        if (!obj || !out) return AM_ERR_NULL;                                   \
        const Adapter_##NAME& g =                                               \
            static_cast<const Pipe_##NAME*>(obj)->adapter();                    \
        out->applied = g.applied();                                             \
        out->trips = g.trips();                                                 \
        out->rollbacks = g.rollbacks();                                         \
        out->blocked_frozen = g.blocked_frozen();                               \
        out->rejected_by_adapter = g.rejected_by_adapter();                     \
        out->thaws = g.thaws();                                                 \
        return AM_OK;                                                           \
    }                                                                           \
    AM_DEFINE_PIPE_COMMON(NAME, SCALAR, RING, Pipe_##NAME, MAX_BYTES)

// The ring type is a macro parameter rather than pasted from the scalar name: `am_ring_##float`
// would be `am_ring_float`, which is not a type. A macro that is clever about types is a macro
// whose error messages are useless.
#define AM_DEFINE_PIPE_BARE(NAME, SCALAR, N, M, RING, MAX_BYTES)                \
    using Pipe_##NAME = argmind::Pipeline<argmind::Rls<SCALAR, N, M>,           \
                                          AM_MAX_STREAMS>;                      \
    extern "C" int am_pipe_##NAME##_init(void* obj, size_t obj_bytes,            \
                                         SCALAR lambda, SCALAR p0,              \
                                         int32_t policy, double max_stale_s,     \
                                         int32_t reference) {                   \
        if (!obj) return AM_ERR_NULL;                                           \
        if (obj_bytes < sizeof(Pipe_##NAME)) return AM_ERR_TOO_SMALL;            \
        if (!valid_policy(policy)) return AM_ERR_CONFIG;                        \
        argmind::Rls<SCALAR, N, M> inner(lambda, p0);                           \
        if (!inner.valid()) return AM_ERR_CONFIG;                               \
        const argmind::Aligner al(static_cast<argmind::Policy>(policy),          \
                                  max_stale_s);                                 \
        *static_cast<Pipe_##NAME*>(obj) = Pipe_##NAME(                          \
            inner, al, argmind::FrameMap<N, M>(), reference);                    \
        return AM_OK;                                                           \
    }                                                                           \
    AM_DEFINE_PIPE_COMMON(NAME, SCALAR, RING, Pipe_##NAME, MAX_BYTES)

AM_DEFINE_PIPE_BARE(f32_6_1, float, 6, 1, am_ring_f32, AM_PIPE_F32_6_1_MAX_BYTES)
AM_DEFINE_PIPE_BARE(f64_6_1, double, 6, 1, am_ring_f64, AM_PIPE_F64_6_1_MAX_BYTES)
AM_DEFINE_PIPE_GUARDED(g_f32_6_1, float, 6, 1, am_ring_f32,
                       AM_PIPE_G_F32_6_1_MAX_BYTES)
AM_DEFINE_PIPE_GUARDED(g_f32_6_3, float, 6, 3, am_ring_f32,
                       AM_PIPE_G_F32_6_3_MAX_BYTES)
AM_DEFINE_PIPE_GUARDED(g_f32_12_1, float, 12, 1, am_ring_f32,
                       AM_PIPE_G_F32_12_1_MAX_BYTES)
AM_DEFINE_PIPE_GUARDED(g_f64_6_1, double, 6, 1, am_ring_f64,
                       AM_PIPE_G_F64_6_1_MAX_BYTES)
