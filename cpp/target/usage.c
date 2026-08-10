/* SPDX-License-Identifier: Apache-2.0
 *
 * A firmware-shaped caller, for measuring what this library costs on the declared target.
 *
 * There is no single "size of the library": what reaches flash depends on which entry
 * points a caller touches, because the build uses -ffunction-sections and --gc-sections.
 * So this file has four configurations selected by -DAM_CFG=n, from the smallest useful one
 * to the whole composed handle, and each is linked separately. Reporting one number would
 * be reporting whichever configuration happened to be measured.
 *
 * Everything is stored through `volatile` sinks so the optimiser cannot delete the calls
 * and hand back a smaller, dishonest figure.
 *
 * This is NOT a board support file. There is no vector table, no clock setup, no startup
 * code -- `_am_entry` exists only to give the linker a root. A real firmware would replace
 * it and would add its own drivers, so these figures are the library's contribution to a
 * flash budget rather than an image that could be flashed.
 */

#include "argmind/argmind.h"

#ifndef AM_CFG
#define AM_CFG 4
#endif

/* The SHAPE knob, for AM_CFG=4 only.
 *
 * A per-shape flash/RAM table was published once and withdrawn: it came from an
 * unrecorded hand-edit of this file, which had no knob, and one row disagreed with the
 * project's own table for the same configuration. Redoing it needed this, so here it is.
 *
 * `AM_SHAPE` names an instantiation the ABI exports; `AM_SHAPE_UC` is the same in the case
 * the MAX_BYTES macros use, because the preprocessor cannot change case. `AM_N`/`AM_M` are
 * the dimensions, and the wiring below is written against them rather than against a
 * particular shape -- which is what stops the next person hand-editing it.
 */
#ifndef AM_SHAPE
#define AM_SHAPE    f32_6_1
#define AM_SHAPE_UC F32_6_1
#define AM_N 6
#define AM_M 1
#endif

/* The one thing the size gate cannot check, stated because a mutation test found it:
 * wiring three features here instead of AM_IMU_DIM links, runs the same code and produces
 * the SAME SIZES, so the table would record a mis-wired map without complaint. Sizes are
 * what that gate measures; correctness at these shapes is not covered anywhere yet. The
 * loop below is written against the macro for that reason -- a literal is the hand-edit
 * this knob exists to prevent. */
#define AM_JOIN_(a, b, c) a##b##c
#define AM_JOIN(a, b, c)  AM_JOIN_(a, b, c)
#define PIPE_G(suffix)    AM_JOIN(am_pipe_g_, AM_SHAPE, suffix)
#define AM_BYTES_         AM_JOIN(AM_PIPE_G_, AM_SHAPE_UC, _MAX_BYTES)
#define AM_STATE_         AM_JOIN(AM_RLS_, AM_SHAPE_UC, _STATE)

volatile int am_sink_i;
volatile float am_sink_f;
volatile double am_sink_d;

/* The caller's storage, which is the point of the ABI taking it rather than allocating.
 * It lands in .bss and is reported separately from the library's own footprint. */
#if AM_CFG == 1
static am_rls_storage obj[AM_RLS_F32_6_1_MAX_BYTES / sizeof(am_rls_storage)];
#elif AM_CFG == 2
static am_rls_storage obj[AM_GUARD_F32_6_1_MAX_BYTES / sizeof(am_rls_storage)];
#elif AM_CFG == 3
static am_rls_storage obj[AM_GUARD_F32_6_1_MAX_BYTES / sizeof(am_rls_storage)];
static unsigned char slot_a[256];
static unsigned char slot_b[256];
static float restored[AM_RLS_F32_6_1_STATE];
#else
/* The rings are sized from the shape: the reference stream carries every feature the map
 * does not take from the aiding one, and the aiding stream carries the targets. Wiring
 * them by hand is how the withdrawn table went wrong. */
#define AM_IMU_DIM (AM_N - 1)          /* one feature is the constant */
_Static_assert(AM_IMU_DIM + 1 == AM_N, "the map does not cover AM_N features");
_Static_assert(AM_STATE_ == AM_N * AM_M + AM_N * AM_N,
               "AM_N/AM_M do not match the shape's declared state size");
static am_rls_storage obj[AM_BYTES_ / sizeof(am_rls_storage)];
static float imu_v[32 * AM_IMU_DIM];
static double imu_t[32];
static unsigned char imu_k[32];
static float aid_v[8 * AM_M];
static double aid_t[8];
static unsigned char aid_k[8];
/* A slot holds one record: the fixed header plus the state block. 96 bytes of margin over
 * the header keeps this honest at every shape without over-reserving at the small ones --
 * the first version reserved 256 and put the 12x1 configuration 40 bytes over the floor. */
static unsigned char slot_a[AM_STATE_ * 4 + 96];
static unsigned char slot_b[AM_STATE_ * 4 + 96];
static float restored[AM_STATE_];
#endif

void _am_entry(void) __attribute__((section(".am_entry"), used, noreturn));

void _am_entry(void) {
    float x[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 1.0f};
    float y[1] = {7.0f};
    float out[1];

#if AM_CFG == 1
    /* 1: the bare adapter. The smallest thing that is still this library. */
    am_sink_i = am_rls_f32_6_1_init(obj, sizeof(obj), 0.999f, 100.0f);
    for (;;) {
        am_report r;
        am_sink_i = am_rls_f32_6_1_update(obj, x, y, &r, 0);
        am_sink_d = r.residual_norm + r.p_trace;
        am_sink_i = am_rls_f32_6_1_predict(obj, x, out);
        am_sink_f = out[0];
    }

#elif AM_CFG == 2
    /* 2: guarded, which is the configuration DESIGN.md says the field should run. */
    am_sink_i = am_guard_f32_6_1_init(obj, sizeof(obj), 0.999f, 100.0f, 5, 0);
    am_sink_i = am_guard_f32_6_1_add_bound(obj, AM_FIELD_P_TRACE, 0, 0.0, 1, 1e9);
    am_sink_i = am_guard_f32_6_1_set_divergence(obj, 8.0, 0.25, 0.02, 1e-3, 20);
    am_sink_i = am_guard_f32_6_1_valid(obj);
    for (;;) {
        am_guard_report r;
        am_guard_counters c;
        am_sink_i = am_guard_f32_6_1_update(obj, x, y, &r, 0);
        am_sink_d = r.residual_norm + r.p_trace;
        am_sink_i = (int)r.tripped + am_guard_f32_6_1_frozen(obj);
        am_sink_i = am_guard_f32_6_1_predict(obj, x, out);
        am_sink_f = out[0];
        /* The rest of the guard's control surface. A review pointed out that leaving these
           out let --gc-sections strip six entry points and still call the result "the
           configuration DESIGN.md says the field should run" -- and DESIGN.md's own
           on_frozen policy and its "a silent guard is an undebuggable one" both assume a
           fielded device reads the counters and can thaw. Measured cost of putting them
           back: 232 bytes, about 4% of this configuration. */
        am_sink_i = (int)am_guard_f32_6_1_frozen_by(obj);
        am_sink_i = am_guard_f32_6_1_counters(obj, &c);
        am_sink_i = (int)(c.trips + c.rollbacks + c.blocked_frozen + c.thaws);
        if (am_sink_i < 0) {
            am_sink_i = am_guard_f32_6_1_freeze(obj);
            am_sink_i = am_guard_f32_6_1_thaw(obj);
            am_sink_i = am_guard_f32_6_1_reset(obj);
        }
    }

#elif AM_CFG == 3
    /* 3: guarded plus the record format -- a device that survives a power cut. */
    am_record rec;
    am_sink_i = am_guard_f32_6_1_init(obj, sizeof(obj), 0.999f, 100.0f, 5, 0);
    am_sink_i = am_guard_f32_6_1_add_bound(obj, AM_FIELD_P_TRACE, 0, 0.0, 1, 1e9);
    am_sink_i = am_persist_select_newest_f32(slot_a, sizeof(slot_a), slot_b,
                                             sizeof(slot_b), &rec, 0, 0);
    if (am_sink_i >= 0) {
        am_sink_i = am_persist_read_state_f32(slot_a, sizeof(slot_a), restored,
                                              AM_RLS_F32_6_1_STATE);
        am_sink_i = am_guard_f32_6_1_load_state(obj, restored);
    }
    for (;;) {
        am_guard_report r;
        am_sink_i = am_guard_f32_6_1_update(obj, x, y, &r, 0);
        am_sink_d = r.p_trace;
        am_sink_i = (int)am_persist_pack_f32(am_guard_f32_6_1_state(obj),
                                             AM_RLS_F32_6_1_STATE, 6, 1, 1, 0xFEEDu,
                                             am_guard_f32_6_1_frozen(obj), 0, 0,
                                             slot_a, sizeof(slot_a));
        am_sink_i = am_persist_next_slot(0);
    }

#else
    /* 4: the whole composed handle -- two sensors, alignment, guard, adapter, persistence.
     * The configuration an integrator actually wires up. */
    am_ring_f32 rings[2];
    am_step_report step;
    am_record rec;
    int i;
    am_sink_i = am_ring_f32_init(&rings[0], imu_v, imu_t, imu_k, AM_IMU_DIM, 32);
    am_sink_i = am_ring_f32_init(&rings[1], aid_v, aid_t, aid_k, AM_M, 8);
    am_sink_i = PIPE_G(_init)(obj, sizeof(obj), 0.999f, 100.0f, 5, 0,
                              AM_POLICY_HOLD_LAST, 0.0, 0);
    for (i = 0; i < AM_IMU_DIM; ++i) {
        am_sink_i = PIPE_G(_add_feature)(obj, 0, i);
    }
    for (i = 0; i < AM_M; ++i) {
        am_sink_i = PIPE_G(_add_target)(obj, 1, i);
    }
    am_sink_i = PIPE_G(_set_constant)(obj, 1);
    am_sink_i = PIPE_G(_add_bound)(obj, AM_FIELD_P_TRACE, 0, 0.0, 1, 1e9);
    am_sink_i = PIPE_G(_set_max_stale_s)(obj, 1, 0.5);
    am_sink_i = PIPE_G(_set_require_fresh_target)(obj, 1);
    am_sink_i = PIPE_G(_configure)(obj, rings, 2);
    am_sink_i = am_persist_select_newest_f32(slot_a, sizeof(slot_a), slot_b,
                                             sizeof(slot_b), &rec, 0, 0);
    if (am_sink_i >= 0) {
        am_sink_i = am_persist_read_state_f32(slot_a, sizeof(slot_a), restored,
                                              AM_STATE_);
        am_sink_i = PIPE_G(_load_state)(obj, restored);
    }
    for (;;) {
        am_sink_i = am_ring_f32_push(&rings[0], am_sink_d, x, 1);
        am_sink_i = am_ring_f32_push(&rings[1], am_sink_d, y, 1);
        am_sink_i = PIPE_G(_step)(obj, rings, 2, &step);
        am_sink_f = (float)(step.prediction[0] + step.worst_staleness);
        am_sink_i = (int)(step.frames + step.updated + step.lost_frames +
                          step.skipped[AM_SKIP_STALE_TARGET]);
        am_sink_i = (int)am_persist_pack_f32(PIPE_G(_state)(obj), AM_STATE_,
                                             AM_N, AM_M, 1, 0xFEEDu,
                                             PIPE_G(_frozen)(obj), 0, 0,
                                             slot_a, sizeof(slot_a));
    }
#endif
}
