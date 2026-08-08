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

/* ==========================================================================
 * The guard.
 *
 * This section exists because of an inconsistency rather than a feature request.
 * DESIGN.md's rule is that the guard *is* an adapter, deliberately, so that "no
 * guard" is not expressible -- a safety mechanism that can be left out gets left
 * out. Until now the C ABI exported bare `Rls` and nothing else, so from C the
 * guard was not merely optional, it was unreachable. The boundary contradicted
 * the design principle it was supposed to carry.
 *
 * `Depth` and `MaxBounds` are fixed at 4 in the ABI. They are the memory, they are
 * reported by `am_guard_*_bytes()`, and choosing them per call would put them in
 * the type and back into the instantiation list. A caller who needs a different
 * depth uses the C++ header; the ABI is a curated surface, not the whole library.
 * `checkpoint_every` stays a runtime argument, and it is the tunable that matters:
 * rollback reaches `depth * checkpoint_every` updates back.
 * ========================================================================== */

/* Fields a bound may watch. A name that does not exist is rejected at add time
 * rather than becoming a check that silently never fires. */
#define AM_FIELD_RESIDUAL_NORM      0
#define AM_FIELD_RESIDUAL_POST_NORM 1
#define AM_FIELD_GAIN_NORM          2
#define AM_FIELD_P_TRACE            3
#define AM_FIELD_P_MIN_DIAG         4

/* Why the guard blocked an update, when it was not the adapter's own rejection. */
#define AM_GUARD_NONE     0
#define AM_GUARD_FROZEN   1  /* frozen; this sample was only looked at            */
#define AM_GUARD_TRIPPED  2  /* a check failed on this sample and it was rolled back */

/* Bit 31 of `tripped` is the divergence check; bits 0..3 are the bounds, in the
 * order they were added. A bitmask rather than a list of names because names need
 * storage this core does not have. */
#define AM_TRIPPED_DIVERGENCE 0x80000000u

#define AM_ERR_FULL        (-5)  /* no room for another check                     */
#define AM_ERR_UNUSABLE    (-6)  /* a configuration that could never work         */

/* `am_report` with what the guard decided. Flattened rather than nested, matching
 * the C++ `GuardReport : Report` and the Python subclass: a caller that does not
 * care about guarding reads it as the plain report it also is. */
typedef struct {
    int32_t applied;
    int32_t reason;         /* AM_REASON_*      */
    double residual_norm;
    double residual_post_norm;
    double gain_norm;
    double p_trace;
    double p_min_diag;
    int32_t frozen;
    int32_t rolled_back;
    uint32_t tripped;       /* AM_TRIPPED_* and bound bits */
    int32_t guard_reason;   /* AM_GUARD_*       */
} am_guard_report;

/* Counters. A silent guard is an undebuggable one, and an adapter rejection and a
 * guard trip are different faults that must never be added together. */
typedef struct {
    uint32_t applied;
    uint32_t trips;
    uint32_t rollbacks;
    uint32_t blocked_frozen;
    uint32_t rejected_by_adapter;
    uint32_t thaws;
} am_guard_counters;

#define AM_DECLARE_GUARD(NAME, SCALAR)                                         \
    size_t am_guard_##NAME##_bytes(void);                                      \
    int am_guard_##NAME##_state_size(void);                                    \
    int am_guard_##NAME##_checkpoint_bytes(void);                              \
    int am_guard_##NAME##_init(void* obj, size_t obj_bytes, SCALAR lambda,      \
                               SCALAR p0, uint32_t checkpoint_every,            \
                               uint32_t recover_after);                        \
    int am_guard_##NAME##_add_bound(void* obj, int32_t field, int32_t has_lo,   \
                                    double lo, int32_t has_hi, double hi);     \
    int am_guard_##NAME##_set_divergence(void* obj, double ratio,               \
                                         double fast_alpha, double slow_alpha,  \
                                         double floor, uint32_t warmup);       \
    int am_guard_##NAME##_valid(const void* obj);                              \
    int am_guard_##NAME##_reset(void* obj);                                    \
    int am_guard_##NAME##_update(void* obj, const SCALAR* x, const SCALAR* y,   \
                                 am_guard_report* report, SCALAR* residual_out);\
    int am_guard_##NAME##_predict(const void* obj, const SCALAR* x, SCALAR* out);\
    int am_guard_##NAME##_health(const void* obj, double* p_trace,              \
                                 double* p_min_diag);                          \
    const SCALAR* am_guard_##NAME##_state(const void* obj);                     \
    int am_guard_##NAME##_load_state(void* obj, const SCALAR* flat);            \
    int am_guard_##NAME##_freeze(void* obj);                                   \
    int am_guard_##NAME##_thaw(void* obj);                                     \
    int am_guard_##NAME##_frozen(const void* obj);                             \
    uint32_t am_guard_##NAME##_frozen_by(const void* obj);                      \
    int am_guard_##NAME##_counters(const void* obj, am_guard_counters* out);

AM_DECLARE_GUARD(f32_6_1, float)
AM_DECLARE_GUARD(f32_6_3, float)
AM_DECLARE_GUARD(f32_12_1, float)
AM_DECLARE_GUARD(f64_6_1, double)

/* Upper bounds, static-asserted in the implementation exactly as the RLS ones are.
 * These are large because the checkpoint ring is: at depth 4 it is
 * 4 * (state_size * sizeof(T) + 16) bytes, which `am_guard_*_checkpoint_bytes()`
 * reports so a caller can see where the memory went rather than infer it.
 *
 * Measured on x86-64 -- a shape, not a target figure, and the layout will differ
 * under AAPCS. f32_6_1: 1208 bytes of which 736 is the ring. f32_6_3: 1448 / 928.
 * f32_12_1: 3488 / 2560 -- worth looking at twice against the declared 8 KB RAM
 * budget, since one guarded twelve-feature adapter is over 40% of it. f64_6_1:
 * 2064 / 1408. The bounds below carry a small margin over these because they are a
 * compile-time promise a C header makes without being able to compute the layout;
 * the first draft was written before measuring and f32_12_1's bound was too small,
 * which the static assertion caught. */
#define AM_GUARD_F32_6_1_MAX_BYTES  1280
#define AM_GUARD_F32_6_3_MAX_BYTES  1536
#define AM_GUARD_F32_12_1_MAX_BYTES 3584
#define AM_GUARD_F64_6_1_MAX_BYTES  2176

/* ==========================================================================
 * The record format.
 *
 * `am_rls_*_state()` hands out the block a device has to survive a power cut with,
 * and until now there was no way to frame it from C -- the state block dead-ended
 * at the boundary. This is that framing: pack, validate, arbitrate two slots.
 *
 * It does **not** write flash. Erase granularity, write width and the driver API
 * are platform-specific and guessing at them would be wrong everywhere. What is
 * here is the part that is the same on every device: a header, a CRC, a sequence
 * number, and the rule that two slots are written alternately so a write
 * interrupted by power loss destroys at most the slot not in use.
 *
 * No instantiation list. Everything but the scalar type is a runtime argument.
 * ========================================================================== */

/* Why a slot was rejected. Ok is 0; a caller that only checks for zero is right. */
#define AM_PERSIST_OK              0
#define AM_PERSIST_TOO_SHORT       1
#define AM_PERSIST_BAD_MAGIC       2
#define AM_PERSIST_FUTURE_VERSION  3  /* newer than this build; refuse, do not guess */
#define AM_PERSIST_BAD_DTYPE       4
#define AM_PERSIST_LENGTH_MISMATCH 5  /* the header describes a different size: torn */
#define AM_PERSIST_CRC_MISMATCH    6

/* A decoded slot. `cause` points into the caller's buffer and is valid only as long
 * as that buffer is; it is not NUL-terminated. Nothing here copies, because on the
 * target there is nowhere to copy to. */
typedef struct {
    const char* cause;
    uint32_t sequence;
    uint32_t fingerprint;
    uint32_t count;         /* elements in the state block */
    uint32_t cause_len;
    uint16_t version;
    uint16_t n_features;
    uint16_t n_outputs;
    uint16_t frozen;
} am_record;

/* Exactly how many bytes a record occupies. Size a slot with this -- and note the
 * constraint it implies, because it bites on real flash: `unpack` compares the length
 * it is given against the one the header describes EXACTLY, so a caller must pass the
 * record's own length, not the capacity of the buffer it was read into. That refusal is
 * the right one (a torn write looks exactly like a short record) but it means a slot
 * rounded up to an erase page cannot be handed in whole. Keep the record length
 * alongside the slot, or re-derive it with this function from the configuration. */
size_t am_persist_record_bytes(uint32_t count, uint32_t element_size,
                               uint32_t cause_len);

/* CRC-32, reflected IEEE 802.3, zlib-compatible. Exposed because a caller needs it
 * to build a configuration fingerprint, and because the fingerprint MUST NOT come
 * from a per-process hash: Python's `hash()` is salted, so one built from it would
 * mismatch on the next boot and reject every valid block.
 *
 * A C caller interoperating with argmind/persist.py must reproduce that module's
 * canonical text rendering byte for byte; this function is the hash, not the
 * rendering, and the rendering is the part that is easy to get subtly wrong. */
uint32_t am_persist_crc32(const uint8_t* data, size_t len);

/* The slot the next write should go to, given which is newest. -1 (nothing valid)
 * starts at slot 0. Writing to the newest slot is exactly the failure two slots
 * exist to prevent, and the reference had that bug. */
int am_persist_next_slot(int newest);

#define AM_DECLARE_PERSIST(NAME, SCALAR)                                       \
    size_t am_persist_pack_##NAME(                                             \
        const SCALAR* state, uint32_t count, uint16_t n_features,               \
        uint16_t n_outputs, uint32_t sequence, uint32_t fingerprint,            \
        int32_t frozen, const char* cause, size_t cause_len, uint8_t* out,      \
        size_t out_bytes);                                                     \
    int am_persist_unpack_##NAME(const uint8_t* in, size_t in_bytes,            \
                                 am_record* rec);                              \
    int am_persist_read_state_##NAME(const uint8_t* in, size_t in_bytes,        \
                                     SCALAR* out, uint32_t out_count);         \
    int am_persist_select_newest_##NAME(const uint8_t* a, size_t a_bytes,       \
                                        const uint8_t* b, size_t b_bytes,       \
                                        am_record* out, int32_t* status_a,      \
                                        int32_t* status_b);

AM_DECLARE_PERSIST(f32, float)
AM_DECLARE_PERSIST(f64, double)

/* ==========================================================================
 * What a C caller still cannot do, said here rather than left to be discovered.
 *
 * There are no `am_stream_*`, `am_align_*` or `am_pipeline_*` entry points. That is
 * a scope decision, not an oversight, and the reason is the same tension this header
 * opens with. `Pipeline::step()` takes an array of `StreamView<T>`, which is a C++
 * template; exposing it needs a C mirror of that struct plus a stream ABI, and
 * `Stream<T, Dim, Capacity>` puts two more axes into an instantiation list that would
 * then multiply against the adapter's. The shape that avoids the explosion is one
 * composed handle per wiring -- streams, aligner, guard and adapter chosen together
 * and exported as a single instantiation -- and that is a design decision worth
 * making deliberately rather than tacking onto this list.
 *
 * So a C caller today gets: a guarded or unguarded adapter, and the record format to
 * persist it. It must do its own buffering and alignment. That is a real limit on the
 * declared deliverable -- "a library, not an application" -- and it is written down.
 * ========================================================================== */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* ARGMIND_H */
