/* SPDX-License-Identifier: Apache-2.0
 *
 * What "hosted inference" means, as code that runs.
 *
 * Checked in because "how do you host a network alongside this?" is a question prose
 * answers badly and a compiling, converging program answers well. `tests/test_examples.py`
 * builds and runs it, so it cannot rot into an illustration that no longer works.
 *
 * The caller owns the network. Some vendor runtime -- TFLite Micro on the CPU, ST's
 * Neural-ART runtime, Arm's Ethos-U driver -- executes a frozen backbone and leaves an
 * output tensor somewhere. argmind never calls it, never links against it, and never sees
 * a graph. The entire boundary is **one float array and one timestamp**.
 *
 * So there is no callback, no function pointer, no plugin interface and no header
 * dependency in either direction. That is the property being demonstrated: the two halves
 * do not know about each other.
 *
 * The backbone below is a stand-in, deliberately shaped like a real one -- it writes int8
 * into its own output tensor with a quantisation scale and zero point, because that is what
 * an NPU hands back.
 */

#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include "argmind/argmind.h"

#define N        128          /* backbone embedding width                              */
#define CAP_F     64          /* how many feature vectors the ring retains             */
#define CAP_O     16
#define DELAY   0.30          /* seconds until the slow sensor grades a prediction     */
#define FEAT_DT 0.02          /* camera at 50 Hz                                       */
#define OUT_DT  0.10          /* outcome sensor at 10 Hz                               */
#ifndef FRAMES
#define FRAMES 60000        /* 128 parameters need far more than 128 labels          */
#endif

/* ---------------------------------------------------------------- the caller's memory */
/* Every byte here is the integrator's. argmind allocates nothing, which is why these are
   file-scope arrays and not a pointer to something. */
static am_rls_storage pipe_obj[AM_PIPE_G_F32_128_1_MAX_BYTES / sizeof(am_rls_storage)];

static float   feat_v[CAP_F * N];   /* ring storage: values                             */
static double  feat_t[CAP_F];       /*               timestamps                         */
static uint8_t feat_k[CAP_F];       /*               per-sample validity                */
static float   out_v[CAP_O * 1];
static double  out_t[CAP_O];
static uint8_t out_k[CAP_O];

static float   feat[N];             /* one dequantised embedding                        */
static am_ring_f32 rings[2];        /* 0 = features, 1 = outcome                        */

/* ------------------------------------------------------------- the "vendor runtime"  */
/* Stands in for interpreter->Invoke(). Writes int8 with a scale and zero point, as an NPU
   does. argmind never sees any of this. */
static int8_t backbone_out[N];
static const float BACKBONE_SCALE = 0.0234f;
static const int   BACKBONE_ZERO  = -7;

static void backbone_invoke(int frame) {
    uint32_t s = (uint32_t)frame * 2654435761u;
    for (int i = 0; i < N; ++i) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        backbone_out[i] = (int8_t)((s >> 3) & 0x7F) - 64;
    }
}

/* The one thing the integrator must write: int8 -> fp32. argmind's precision floor is fp32
   and there is no fp16 or int8 path in the covariance recursion, measured rather than
   assumed, so this conversion is not optional. It is N multiply-adds. */
static void dequantise(void) {
    for (int i = 0; i < N; ++i)
        feat[i] = ((float)backbone_out[i] - (float)BACKBONE_ZERO) * BACKBONE_SCALE;
}

/* The label: what the slow, trustworthy sensor eventually measures. Here a linear function
   of the embedding, so the run has a right answer to converge to. */
static float grade(const float* x) {
    return 2.0f * x[3] - 1.5f * x[70] + 0.25f * x[11];
}

int main(void) {
    /* --- configure once ------------------------------------------------------------- */
    if (am_ring_f32_init(&rings[0], feat_v, feat_t, feat_k, N, CAP_F) != AM_OK) return 1;
    if (am_ring_f32_init(&rings[1], out_v, out_t, out_k, 1, CAP_O) != AM_OK) return 1;

    /* Reference is stream 1, the OUTCOME: instants are label arrivals, and `hold_last`
       reaches back for the features that earned them. */
    if (am_pipe_g_f32_128_1_init(pipe_obj, sizeof(pipe_obj),
                                 /*lambda*/ 0.999f, /*p0*/ 100.0f,
                                 /*checkpoint_every*/ 8, /*recover_after*/ 0,
                                 AM_POLICY_HOLD_LAST, /*max_stale_s*/ 0.0,
                                 /*reference*/ 1) != AM_OK) return 2;
    for (int i = 0; i < N; ++i)
        if (am_pipe_g_f32_128_1_add_feature(pipe_obj, 0, i) != AM_OK) return 3;
    if (am_pipe_g_f32_128_1_add_target(pipe_obj, 1, 0) != AM_OK) return 4;
    /* The guard. Without at least one check `valid()` is false by design: a guard that can
       never freeze is a wrapper that looks like a safety mechanism. */
    if (am_pipe_g_f32_128_1_add_bound(pipe_obj, AM_FIELD_P_TRACE, 0, 0.0, 1, 1e12) != AM_OK)
        return 5;
    if (am_pipe_g_f32_128_1_configure(pipe_obj, rings, 2) != AM_OK) return 6;

    /* --- the loop ------------------------------------------------------------------- */
    double t = 0.0, next_out = DELAY;
    float  pend_y[64];
    double pend_t[64];
    unsigned pend_n = 0;          /* a RING. The first draft stopped recording at 64 and
                                     every later label silently reused a stale entry. */
    unsigned updates = 0, predictions = 0;
    double  last_pred = 0.0, last_truth = 0.0;
    double  lbl_t[64];
    float   lbl_y[64];
    unsigned lbl_n = 0;

    for (int frame = 0; frame < FRAMES; ++frame) {
        /* 1. the vendor's runtime runs. argmind is not involved. */
        backbone_invoke(frame);

        /* 2. int8 -> fp32, the integrator's three lines */
        dequantise();

        /* 3. push, stamped with the instant this embedding is ABOUT rather than the instant
              it was computed. The label for it arrives DELAY later, and stamping by
              computation time silently trains the wrong map -- measured, and there is a
              test for it, because no guard can catch it. */
        am_ring_f32_push(&rings[0], t + DELAY, feat, 1);
        pend_y[pend_n & 63] = grade(feat); pend_t[pend_n & 63] = t + DELAY; ++pend_n;

        /* 4. when the slow sensor reports, push what it measured */
        while (next_out <= t) {
            for (unsigned j = 0; j < 64u && j < pend_n; ++j) {
                const unsigned i = (pend_n - 1u - j) & 63u;
                if (pend_t[i] <= next_out) {
                    am_ring_f32_push(&rings[1], next_out, &pend_y[i], 1);
                    /* remember what was pushed, so the check below compares the
                       prediction against the label of the frame it was made for -- not
                       against the embedding that happens to be in hand now. Getting that
                       wrong is the same class of mistake as mis-stamping the features,
                       and it read as a convergence failure until it was chased down. */
                    lbl_t[lbl_n & 63] = next_out;
                    lbl_y[lbl_n & 63] = pend_y[i];
                    ++lbl_n;
                    break;
                }
            }
            next_out += OUT_DT;
        }

        /* 5. one bounded step. Learns from whatever frames the rings can now form, and
              reports. Never allocates, never blocks, never loops unboundedly. */
        am_step_report r;
        if (am_pipe_g_f32_128_1_step(pipe_obj, rings, 2, &r) != AM_OK) return 7;
        updates += r.updated;

        /* 6. the output is in the report -- present even on a frame the input gate refused
              to learn from, and even while the guard is frozen. Refusing to LEARN is not a
              reason to stop PREDICTING, which is what lets a fast output run on a slow
              label. */
        if (r.has_prediction && r.has_t) {
            for (unsigned i = 0; i < lbl_n && i < 64u; ++i) {
                if (lbl_t[i] == r.t) {
                    ++predictions;
                    last_pred = r.prediction[0];
                    last_truth = (double)lbl_y[i];
                    break;
                }
            }
        }

        t += FEAT_DT;
    }

    const int frozen = am_pipe_g_f32_128_1_frozen(pipe_obj);
    printf("hosted integration, %d-wide frozen backbone + adapted head\n", N);
    printf("  caller's storage: pipeline %u B, rings %u B, embedding %u B\n",
           (unsigned)am_pipe_g_f32_128_1_bytes(),
           (unsigned)(sizeof(feat_v) + sizeof(feat_t) + sizeof(feat_k) +
                      sizeof(out_v) + sizeof(out_t) + sizeof(out_k)),
           (unsigned)sizeof(feat));
    printf("  updates %u   predictions %u   guard frozen %d\n",
           updates, predictions, frozen);
    printf("  last prediction %.6f   truth %.6f   error %.3e\n",
           last_pred, last_truth, last_pred - last_truth);
    const double e = last_pred - last_truth;
    printf("%s\n", (e < 1e-3 && e > -1e-3) ? "HOSTED OK" : "HOSTED FAILED");
    return 0;
}
