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
 * explicit list of instantiations**, and adding a dimension means adding THREE lines
 * -- a declaration here, a definition in `abi.cpp`, and a `MAX_BYTES` -- and
 * rebuilding. This said "one line" until someone tried it and the compiler refused
 * the two-line version. That is a real cost and it is the honest one.
 *
 * Storage is the caller's, and so is the stack
 * --------------------------------------------
 * Nothing here allocates. `am_rls_*_bytes()` reports how much storage an instance
 * needs and the caller supplies it — statically, on its own stack, or from its own
 * pool.
 *
 * **`*_bytes()` is the object, not the whole cost, and the difference bit once.**
 * These functions also use the caller's stack, and until 2026-08-09 nobody had
 * measured how much. With `-fstack-usage` for Cortex-M4 at `-Os`:
 *
 *   steady state    am_pipe_*_step        312 to 328 bytes
 *                   am_guard_*_update      88 to 104
 *                   am_rls_*_update        72 to 88
 *   initialisation  am_pipe_*_init        136 to 168 (first tier)
 *                   am_pipe_g_f32_128_1_init     504 (the head shape)
 *                   am_guard_*_init             32 to 40
 *                   am_rls_*_init               24 to 32
 *
 * `cpp/target/SIZES.md` carries these and a test regenerates them, because the
 * figures in this comment were wrong three times before that existed.
 *
 * **`init` is no longer a spike, and getting there took two goes.** It used to build
 * the whole object as a temporary and copy it, costing `am_pipe_g_f32_12_1_init`
 * 8,296 bytes — more than the 8 KB floor, so that entry point could not be called on
 * the target this library names. Placement new fixed that and left one residue, which
 * this comment recorded as "one adapter-sized temporary … recorded here rather than
 * done": `Pipeline` and `Guard` each took their adapter BY VALUE, so a whole `Rls` and
 * a whole `Guard` were alive on the stack before the copy.
 *
 * Adding the 128-feature head shape made the residue absurd —
 * `am_pipe_g_f32_128_1_init` wanted **397,192 bytes of stack to build a 332,392-byte
 * object**. Both constructors now take an `in_place_adapter_t` tag and build the
 * adapter from its own arguments, nesting so a guarded pipeline has no temporary at
 * any level, and `Rls::params_valid()` lets this ABI refuse a bad configuration
 * without constructing something to ask. Measured: 397,192 → 504 at 128 features,
 * 4,288 → 168 at twelve, 1,536 → 152 at six. Every shape benefited.
 *
 * `tests/test_target_build.py` bounds all of it, so it cannot grow back quietly.
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

/* Version, and what it promises -- which until 2026-08-10 was nothing at all: the two
 * macros existed and no document said what a consumer could rely on.
 *
 *   MAJOR  The C ABI. Within one major version, a program compiled against an older
 *          header links and runs against a newer library: entry points are added at the
 *          end of the instantiation list, struct fields are appended into existing
 *          padding or at the end, and enum values are never renumbered. A major bump may
 *          break any of that.
 *
 *   MINOR  Additions. New instantiations, new entry points, new struct fields at the end.
 *          A minor bump never removes or renumbers.
 *
 * **The persisted record format is versioned separately and independently** -- see
 * `persist.hpp`. A device's saved state has to outlive a library upgrade, so its
 * compatibility is not this number's business. Its own version byte and configuration
 * fingerprint decide whether a block is accepted, and a mismatched `FrameMap` order is
 * refused rather than silently misread.
 *
 * `AM_VERSION_REQUIRE(major, minor)` makes a consumer's expectation a compile error
 * instead of a link-time surprise. A policy nothing enforces is a comment. */
#define AM_VERSION_MAJOR 0
#define AM_VERSION_MINOR 1

#define AM_VERSION_REQUIRE(major, minor)                                       \
    typedef char am_version_check_##major##_##minor[                           \
        ((AM_VERSION_MAJOR) == (major) && (AM_VERSION_MINOR) >= (minor))       \
            ? 1 : -1]

/* Zero while the API is still moving. At 0.x a MINOR bump is allowed to break things,
 * which is the usual convention and is stated here rather than assumed. The promises
 * above take effect at 1.0. */

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
 * rather than becoming a check that silently never fires.
 *
 * `add_bound` also refuses a field the *adapter* cannot report. In C++ that is the
 * adapter's `health_fields` mask, and the last two here are covariance quantities an
 * adapter without a covariance does not have. **There is deliberately no
 * `am_guard_*_health_fields()` to query it**: every guard this ABI exports wraps `Rls`,
 * which declares all five, so the function could only ever return one constant. This
 * project has twice shipped an entry point nothing called (`next_slot`, and
 * `select_newest` compiled but reached by no binary), and a third is not worth the
 * surface. Export it when a non-RLS adapter reaches this boundary. */
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
#define AM_GUARD_F32_6_1_MAX_BYTES  1152
#define AM_GUARD_F32_6_3_MAX_BYTES  1408
#define AM_GUARD_F32_12_1_MAX_BYTES 3456
#define AM_GUARD_F64_6_1_MAX_BYTES  2048

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
 * Streams and the pipeline: the composed handle.
 *
 * The previous version of this header ended by saying a C caller could not reach
 * the pipeline, and why: `Pipeline::step()` takes an array of `StreamView<T>`, and
 * `Stream<T, Dim, Capacity>` would put two more axes into an instantiation list
 * that then multiplies against the adapter's.
 *
 * The resolution is to move the two axes out of the type entirely: **the caller
 * owns the sample storage**. `am_ring_f32` is a plain C descriptor over buffers the
 * caller declares, so `dim` and `capacity` are runtime values on this side and
 * nothing about a stream's shape reaches the instantiation list. What is left in
 * the type is the adapter -- which is where the compile-time shape was earning its
 * keep all along, in the arithmetic -- plus `AM_MAX_STREAMS`.
 *
 * That is the same trade `StreamView` already makes inside the C++ core, made once
 * more at the boundary. There is no second copy of the ring logic: `am_ring_*_push`
 * calls the same `ring_push` that `Stream<T, Dim, Capacity>::push` does, because a
 * refusal implemented twice is one that will eventually disagree with itself.
 *
 * The cost, stated: the C ring's push loop is bounded by a runtime `dim` and its
 * wrap is a runtime modulo, where the C++ `Stream<>` folds both to constants. That
 * is one integer division and an unfoldable loop per push, at the sensor's rate.
 * ========================================================================== */

/* Streams per pipeline. In the type, because the staleness array and the shape
 * table are, so the memory stays declared. */
#define AM_MAX_STREAMS 4

/* Outputs a step report can carry back. Three, because the widest instantiation in
 * the list below is `6_3`. */
#define AM_MAX_OUTPUTS 3

/* Alignment policies. `nearest` and `interpolate` consult samples from AFTER the
 * reference instant: legitimate offline, a correctness bug in a causal loop, which
 * is what a pipeline is. `am_pipe_*_configure` refuses them unless
 * `set_allow_non_causal` said so out loud. */
#define AM_POLICY_HOLD_LAST     0
#define AM_POLICY_NEAREST       1
#define AM_POLICY_INTERPOLATE   2
#define AM_POLICY_DROP_IF_STALE 3

/* Why a frame that existed did not adapt the model. Indices into
 * `am_step_report.skipped`. `AM_SKIP_REJECTED` is unreachable with the adapters
 * this library ships and exists for a future one. */
#define AM_SKIP_STALE                    0
#define AM_SKIP_INVALID                  1
#define AM_SKIP_FROZEN                   2
#define AM_SKIP_GUARD_TRIPPED            3
#define AM_SKIP_NONFINITE_INPUT          4
#define AM_SKIP_NONPOSITIVE_DENOMINATOR  5
#define AM_SKIP_REJECTED                 6
/* Only produced when `am_pipe_*_set_require_fresh_target` is on. Appended rather
 * than filed with the other input-gate reasons, which is where it belongs
 * semantically: these are array indices, so inserting it would silently renumber
 * everything after it for a caller already compiled against the old header. */
#define AM_SKIP_STALE_TARGET             7
#define AM_SKIP_COUNT                    8

#define AM_ERR_UNCONFIGURED (-7)  /* step() before a successful configure()        */

/* A caller-owned ring of timestamped samples.
 *
 * Declare the three buffers, call `am_ring_*_init`, and never touch the fields
 * afterwards -- they are visible because this is C and the struct has to be
 * declarable, not because they are an interface. Sizes:
 *
 *     values[capacity * dim]   times[capacity]   valid[capacity]
 *
 * `dropped` counts samples overwritten because the ring was full. Note what it is
 * NOT: a ring has no idea whether anyone read a sample before it was overwritten,
 * so this cannot mean "lost". In steady state a correctly sized ring overwrites on
 * every push and this climbs forever with nothing wrong. What can honestly be
 * called lost is `am_step_report.lost_frames`. */
typedef struct {
    float* values;
    double* times;
    uint8_t* valid;
    int32_t dim;
    int32_t capacity;
    int32_t head;
    int32_t count;
    uint64_t dropped;
    double last_t;
} am_ring_f32;

typedef struct {
    double* values;
    double* times;
    uint8_t* valid;
    int32_t dim;
    int32_t capacity;
    int32_t head;
    int32_t count;
    uint64_t dropped;
    double last_t;
} am_ring_f64;

#define AM_DECLARE_RING(NAME, SCALAR)                                          \
    int am_ring_##NAME##_init(am_ring_##NAME* r, SCALAR* values, double* times, \
                              uint8_t* valid, int32_t dim, int32_t capacity);   \
    int am_ring_##NAME##_reset(am_ring_##NAME* r);                             \
    int am_ring_##NAME##_push(am_ring_##NAME* r, double t, const SCALAR* values,\
                              int32_t valid);

AM_DECLARE_RING(f32, float)
AM_DECLARE_RING(f64, double)

/* What one `step()` did. Observability is the return value, not a logging
 * afterthought.
 *
 * `lost_frames` is reference-stream samples overwritten before they could become
 * frames -- input the pipeline will never see. Non-zero means `step()` is being
 * called too rarely for the ring size. It counts the reference stream only, and
 * that is the whole of what can honestly be counted. */
typedef struct {
    int32_t ok;              /* 0 when unconfigured or the rings do not match     */
    int32_t frozen;
    int32_t has_t;
    int32_t n_staleness;
    uint32_t frames;
    uint32_t updated;
    uint32_t lost_frames;
    uint32_t frozen_by;      /* checks that tripped this step; 0 when unguarded   */
    uint32_t skipped[AM_SKIP_COUNT];
    uint32_t pad;
    double worst_staleness;  /* the one to alarm on                               */
    double t;
    double staleness[AM_MAX_STREAMS];  /* signed, of the most recent frame        */
    /* The model's output for the most recent frame, after every adaptation this
     * step applied. Present whenever a frame existed -- INCLUDING one the input
     * gate refused to learn from, which is the point: refusing to learn from a
     * sample is not a reason to stop predicting on it, and separating the two is
     * what lets a fast output run on a slow target.
     *
     * A double array regardless of the pipeline's scalar type, so this struct stays
     * one shape across every instantiation. The widening is exact for f32. */
    int32_t has_prediction;
    int32_t n_outputs;
    double prediction[AM_MAX_OUTPUTS];
} am_step_report;

/* The shared surface of every pipeline instantiation, guarded or not. */
#define AM_DECLARE_PIPE_COMMON(NAME, SCALAR, RING)                             \
    size_t am_pipe_##NAME##_bytes(void);                                       \
    int am_pipe_##NAME##_n_features(void);                                     \
    int am_pipe_##NAME##_state_size(void);                                     \
    int am_pipe_##NAME##_add_feature(void* obj, int32_t stream,                 \
                                     int32_t component);                       \
    int am_pipe_##NAME##_add_target(void* obj, int32_t stream,                  \
                                    int32_t component);                        \
    int am_pipe_##NAME##_set_constant(void* obj, int32_t on);                   \
    int am_pipe_##NAME##_set_max_stale_s(void* obj, int32_t enable,             \
                                         double limit);                        \
    int am_pipe_##NAME##_set_require_valid(void* obj, int32_t on);              \
    int am_pipe_##NAME##_set_allow_non_causal(void* obj, int32_t on);               int am_pipe_##NAME##_set_require_fresh_target(void* obj, int32_t on);       \
    int am_pipe_##NAME##_configure(void* obj, const RING* rings, int32_t n);    \
    int am_pipe_##NAME##_step(void* obj, const RING* rings, int32_t n,          \
                              am_step_report* out);                            \
    int am_pipe_##NAME##_predict(const void* obj, const SCALAR* x, SCALAR* out); \
    const SCALAR* am_pipe_##NAME##_state(const void* obj);                      \
    int am_pipe_##NAME##_load_state(void* obj, const SCALAR* flat);             \
    int am_pipe_##NAME##_frozen(const void* obj);

/* Unguarded. Legitimate for an offline reference run over recorded data and a bad
 * idea in the field, which is why the guarded names are the shorter ones to reach
 * for. */
#define AM_DECLARE_PIPE_BARE(NAME, SCALAR, RING)                               \
    int am_pipe_##NAME##_init(void* obj, size_t obj_bytes, SCALAR lambda,       \
                              SCALAR p0, int32_t policy, double max_stale_s,    \
                              int32_t reference);                              \
    AM_DECLARE_PIPE_COMMON(NAME, SCALAR, RING)

/* Guarded. `Depth` and `MaxBounds` are 4, as for `am_guard_*` and for the same
 * reason: they are the memory. */
#define AM_DECLARE_PIPE_GUARDED(NAME, SCALAR, RING)                            \
    int am_pipe_##NAME##_init(void* obj, size_t obj_bytes, SCALAR lambda,       \
                              SCALAR p0, uint32_t checkpoint_every,             \
                              uint32_t recover_after, int32_t policy,           \
                              double max_stale_s, int32_t reference);          \
    int am_pipe_##NAME##_add_bound(void* obj, int32_t field, int32_t has_lo,     \
                                   double lo, int32_t has_hi, double hi);      \
    int am_pipe_##NAME##_set_divergence(void* obj, double ratio,                \
                                        double fast_alpha, double slow_alpha,   \
                                        double floor, uint32_t warmup);        \
    int am_pipe_##NAME##_frozen_by(const void* obj, uint32_t* out);             \
    int am_pipe_##NAME##_counters(const void* obj, am_guard_counters* out);     \
    AM_DECLARE_PIPE_COMMON(NAME, SCALAR, RING)

AM_DECLARE_PIPE_BARE(f32_6_1, float, am_ring_f32)
AM_DECLARE_PIPE_BARE(f64_6_1, double, am_ring_f64)
AM_DECLARE_PIPE_GUARDED(g_f32_6_1, float, am_ring_f32)
AM_DECLARE_PIPE_GUARDED(g_f32_6_3, float, am_ring_f32)
AM_DECLARE_PIPE_GUARDED(g_f32_12_1, float, am_ring_f32)
AM_DECLARE_PIPE_GUARDED(g_f32_128_1, float, am_ring_f32)
AM_DECLARE_PIPE_GUARDED(g_f64_6_1, double, am_ring_f64)

/* Upper bounds, static-asserted in the implementation. A guarded pipeline is its
 * guard plus about 400 bytes; the guard is mostly its checkpoint ring. Measured on
 * x86-64 and therefore a shape rather than a target figure. */
#define AM_PIPE_F32_6_1_MAX_BYTES     768
#define AM_PIPE_F64_6_1_MAX_BYTES     1024
#define AM_PIPE_G_F32_6_1_MAX_BYTES   1664
#define AM_PIPE_G_F32_6_3_MAX_BYTES   1920
#define AM_PIPE_G_F32_12_1_MAX_BYTES  4032

/* The second tier's shape: a learnable HEAD on a frozen network, whose input is an
 * embedding rather than a handful of sensor channels. Present because the scope widened
 * on 2026-08-09 toward camera and audio work, where the integration point is the last
 * layer -- the NPU runs the backbone forward-only and this adapts what follows it.
 *
 * **It is enormous next to the others, and that is RLS's n^2, not an accident.** State is
 * `N*M + N*N`, so 128 features carry a 128x128 covariance, and the guard's ring multiplies
 * it by `Depth`. Measured 332,392 bytes against the first tier's 1,792. It does NOT belong
 * on the 8 KB floor and is not offered there; `--gc-sections` means a firmware that never
 * touches it pays nothing. At 256 features the same arithmetic reaches 30% of an STM32N6's
 * whole SRAM and at 512 it does not fit at all, which is the boundary to know before
 * choosing a backbone's output width. */
#define AM_PIPE_G_F32_128_1_MAX_BYTES 335744
#define AM_PIPE_G_F64_6_1_MAX_BYTES   2688

/* Six instantiations is a lot of code for a caller who wants one. Build with
 * `-ffunction-sections -fdata-sections` and link with `--gc-sections`, which the
 * embedded CMake path already sets, and the unused ones do not reach flash. */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* ARGMIND_H */
