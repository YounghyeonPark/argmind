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
static am_rls_storage obj[AM_PIPE_G_F32_6_1_MAX_BYTES / sizeof(am_rls_storage)];
static float imu_v[32 * 3];
static double imu_t[32];
static unsigned char imu_k[32];
static float aid_v[8];
static double aid_t[8];
static unsigned char aid_k[8];
static unsigned char slot_a[256];
static unsigned char slot_b[256];
static float restored[AM_RLS_F32_6_1_STATE];
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
    am_sink_i = am_ring_f32_init(&rings[0], imu_v, imu_t, imu_k, 3, 32);
    am_sink_i = am_ring_f32_init(&rings[1], aid_v, aid_t, aid_k, 1, 8);
    am_sink_i = am_pipe_g_f32_6_1_init(obj, sizeof(obj), 0.999f, 100.0f, 5, 0,
                                       AM_POLICY_HOLD_LAST, 0.0, 0);
    am_sink_i = am_pipe_g_f32_6_1_add_feature(obj, 0, 0);
    am_sink_i = am_pipe_g_f32_6_1_add_feature(obj, 0, 1);
    am_sink_i = am_pipe_g_f32_6_1_add_feature(obj, 0, 2);
    am_sink_i = am_pipe_g_f32_6_1_add_feature(obj, 1, 0);
    am_sink_i = am_pipe_g_f32_6_1_add_feature(obj, 0, 0);
    am_sink_i = am_pipe_g_f32_6_1_add_target(obj, 1, 0);
    am_sink_i = am_pipe_g_f32_6_1_set_constant(obj, 1);
    am_sink_i = am_pipe_g_f32_6_1_add_bound(obj, AM_FIELD_P_TRACE, 0, 0.0, 1, 1e9);
    am_sink_i = am_pipe_g_f32_6_1_set_max_stale_s(obj, 1, 0.5);
    am_sink_i = am_pipe_g_f32_6_1_set_require_fresh_target(obj, 1);
    am_sink_i = am_pipe_g_f32_6_1_configure(obj, rings, 2);
    am_sink_i = am_persist_select_newest_f32(slot_a, sizeof(slot_a), slot_b,
                                             sizeof(slot_b), &rec, 0, 0);
    if (am_sink_i >= 0) {
        am_sink_i = am_persist_read_state_f32(slot_a, sizeof(slot_a), restored,
                                              AM_RLS_F32_6_1_STATE);
        am_sink_i = am_pipe_g_f32_6_1_load_state(obj, restored);
    }
    for (;;) {
        am_sink_i = am_ring_f32_push(&rings[0], am_sink_d, x, 1);
        am_sink_i = am_ring_f32_push(&rings[1], am_sink_d, y, 1);
        am_sink_i = am_pipe_g_f32_6_1_step(obj, rings, 2, &step);
        am_sink_f = (float)(step.prediction[0] + step.worst_staleness);
        am_sink_i = (int)(step.frames + step.updated + step.lost_frames +
                          step.skipped[AM_SKIP_STALE_TARGET]);
        am_sink_i = (int)am_persist_pack_f32(am_pipe_g_f32_6_1_state(obj),
                                             AM_RLS_F32_6_1_STATE, 6, 1, 1, 0xFEEDu,
                                             am_pipe_g_f32_6_1_frozen(obj), 0, 0,
                                             slot_a, sizeof(slot_a));
    }
#endif
}
