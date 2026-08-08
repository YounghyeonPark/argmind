// SPDX-License-Identifier: Apache-2.0
//
// Aligning multi-rate, asynchronous streams, ported from argmind/align.py.
//
// This is the part most multi-sensor code gets wrong silently. Sensors do not share a clock
// or a rate; samples arrive late, jittered, or not at all. Zipping arrays together implies a
// synchronisation policy without stating it, and the choice of policy changes the answer.
//
// So the policy is explicit, it is recorded in the frame it produced, and every frame reports
// how stale each contributing sample was. A caller who does not look at staleness has still
// been told.
//
// One arithmetic detail carried across exactly
// --------------------------------------------
// Interpolation blends in **double** and then narrows back to the stream's own type, even
// for an f32 stream, because the Python does: `(1-w) * a.astype(float64) + w * b.astype(
// float64)` and then `.astype(a.values.dtype)`. Blending in T instead would be a defensible
// choice and a different one, and it would show up as a parity failure that looks like a bug
// in the search rather than in the arithmetic.
//
// That exactness has a precondition worth stating: **the blend must not be contracted into
// an FMA.** Measured — rebuilding the parity tool with `-march=x86-64-v3 -ffp-contract=fast`
// lets the compiler fuse `(1-w)*a + w*b`, and 1506 of 9995 f64 blends then differ by one ulp
// of the bracket's scale. f32 is unaffected, because narrowing absorbs it, and the M4F's
// double path is soft float and cannot contract either — but an M7 with `fpv5-d16`, or any
// host build with `-march=native`, would diverge on an f64 stream. If that day comes the
// answer is `-ffp-contract=off` for this translation unit, not a tolerance.
//
// (The per-element ulp reading on that measurement peaks at 384, which is the familiar
// cancellation artefact rather than error: the worst bracket crosses zero, so the result is
// small while the operands are not. Against the bracket's own scale it is one ulp.)
//
// The cost that widening imposes on the target, stated because it is on the axis the target
// decision was made on
// ---------------------------------------------------------------------------------------
// The declared floor is `-mfpu=fpv4-sp-d16`: a **single**-precision FPU. So every
// interpolated pick runs one double division for the weight plus two double multiplies and
// a double add per element **in software** on an M4F. Interpolation is the only policy that
// pays this — the other three copy a pointer — and it runs at the reference stream's rate,
// not the fast sensor's. It is still the most expensive thing in this header on the target,
// and matching the reference's arithmetic is why. Nobody has measured it on hardware; the
// point here is that the cost exists and is not hidden, not that it has been quantified.

#ifndef ARGMIND_ALIGN_HPP
#define ARGMIND_ALIGN_HPP

#include <stdint.h>

#include "argmind/stream.hpp"

namespace argmind {

/// How to pick a sample for a reference instant.
///
/// `HoldLast`   the newest sample at or before the reference time. The honest default for
///              control: it never invents data and never looks ahead, but it is biased by up
///              to one sample period of the slower stream.
/// `Nearest`    the closest in time, before or after. Lower average error, but it USES
///              FUTURE DATA relative to the reference instant, so it is legitimate offline
///              and a bug in a causal real-time loop.
/// `Interpolate` linear between the bracketing samples. Best accuracy for smooth signals;
///              meaningless for anything discrete, and it also requires a future sample.
/// `DropIfStale` like HoldLast, but yields no frame at all when any stream is older than its
///              limit. Use when a stale input is worse than no output.
enum class Policy : uint8_t {
    HoldLast = 0,
    Nearest = 1,
    Interpolate = 2,
    DropIfStale = 3,
};

/// True for the policies that never consult a sample later than the reference instant.
///
/// The two that do — `Nearest` and `Interpolate` — are legitimate offline and a correctness
/// error inside a causal loop, which is why this is named rather than left for a user to
/// work out.
inline bool is_causal(Policy p) {
    return p != Policy::Nearest && p != Policy::Interpolate;
}

/// Why a pick produced nothing. A configuration mistake and a quiet sensor must not look
/// alike in field data: the first is a wiring error that will never fix itself, the second
/// is a runtime condition that may. They used to share both the return value and the
/// `present == false` state, so a stream wired in too wide read as a dead sensor.
enum class PickStatus : uint8_t {
    Ok = 0,
    NoSample = 1,     ///< nothing in the ring answers for this instant
    DimTooLarge = 2,  ///< wider than `Pick::MAX_DIM`, and this policy has to blend
};

/// What one stream contributed to one frame.
template <typename T>
struct Pick {
    /// The widest stream this aligner can *interpolate*.
    ///
    /// Only interpolation needs it: every other policy hands back a pointer into the ring
    /// and copies nothing. Interpolation has to synthesise a value and this core has no
    /// heap, so the buffer is here and the limit is real — but it binds only on that one
    /// path, which is where the check now sits. It used to sit ahead of the policy
    /// dispatch, refusing wide streams under `HoldLast` and `Nearest` as well, for a buffer
    /// those policies never touch.
    ///
    /// 32 is **chosen, not derived**, and it is worth being plain about that. The
    /// reasoning is that DESIGN.md puts sensor-fusion state dimensions at 3 to 30 (itself
    /// an unsourced "typically"), and the design driver's widest sensor is a 6-axis IMU —
    /// so 32 has headroom over everything this library is aimed at. It is not a bound
    /// anything forces: a multi-channel ADC or a wide feature vector would exceed it, and
    /// such a stream simply cannot be interpolated by this aligner. `PickStatus::DimTooLarge`
    /// exists so that limit announces itself instead of looking like a dead sensor.
    static constexpr int MAX_DIM = 32;

    PickStatus status = PickStatus::NoSample;
    bool present = false;
    bool valid = false;
    /// Signed: negative means the sample predates the reference, positive means it came
    /// after — only possible under a non-causal policy. Interpolated picks report 0.
    double staleness = 0.0;
    const T* values = nullptr;    ///< borrowed, unless `blended` was written
    T blended[MAX_DIM];           ///< interpolation writes here; `values` then points at it
};

/// Combines streams onto a reference stream's timebase.
///
/// Stateless apart from the drain watermark, so a caller may hold one per pipeline and
/// nothing here allocates.
class Aligner {
public:
    Aligner() = default;
    Aligner(Policy policy, double max_stale_s)
        : policy_(policy), max_stale_s_(max_stale_s) {}

    Policy policy() const { return policy_; }
    bool causal() const { return is_causal(policy_); }
    double max_stale_s() const { return max_stale_s_; }
    /// A non-positive limit means "no limit". That is how `None` behaves in the reference,
    /// but it is not the same thing: `Aligner.__init__` there *raises* on a non-positive
    /// `max_stale_s` and only `None` disables the check. This core has no exceptions and
    /// folds the two into one sentinel, so a caller that passes 0 meaning "zero tolerance"
    /// gets "no tolerance check" instead. Stated because it is a real difference in what
    /// the two accept, not a formality.
    bool has_stale_limit() const { return max_stale_s_ > 0.0; }

    // 64-bit for the same reason `Stream::dropped` is: `frames` advances once per frame,
    // so at 1 kHz a 32-bit counter wraps on the same 49.7-day clock, and a diagnostic that
    // silently restarts is worse than one that is merely large.
    uint64_t frames() const { return frames_; }
    uint64_t dropped_stale() const { return dropped_stale_; }
    uint64_t no_data() const { return no_data_; }

    void reset() {
        consumed_to_ = -__builtin_inf();
        frames_ = 0;
        dropped_stale_ = 0;
        no_data_ = 0;
    }

    double consumed_to() const { return consumed_to_; }
    void set_consumed_to(double t) { consumed_to_ = t; }

    /// Pick one stream's contribution at reference instant `t`.
    ///
    /// `out->values` borrows the ring for every policy except interpolation, which has to
    /// synthesise a value and writes it into `out->blended`. That is why `Pick` carries the
    /// buffer: a caller that copied the pointer and then let the `Pick` die would otherwise
    /// have a dangling read, and there is nowhere to allocate one instead.
    template <typename T>
    bool pick(const StreamView<T>& s, double t, Pick<T>* __restrict out) const {
        // Reset the scalars only. `*out = Pick<T>()` cleared the blend buffer too — 280
        // bytes for `Pick<double>`, on every pick, including the three policies that never
        // write into it and hand back a pointer into the ring instead. The buffer is
        // written before it is read on the one path that uses it.
        out->status = PickStatus::NoSample;
        out->present = false;
        out->valid = false;
        out->staleness = 0.0;
        out->values = nullptr;

        if (policy_ == Policy::HoldLast || policy_ == Policy::DropIfStale) {
            const SampleRef<T> got = s.newest_at_or_before(t);
            if (!got.present) return false;
            out->status = PickStatus::Ok;
            out->present = true;
            out->valid = got.valid;
            out->staleness = got.t - t;
            out->values = got.values;
            return true;
        }

        if (policy_ == Policy::Nearest) {
            const SampleRef<T> got = s.nearest(t);
            if (!got.present) return false;
            out->status = PickStatus::Ok;
            out->present = true;
            out->valid = got.valid;
            out->staleness = got.t - t;
            out->values = got.values;
            return true;
        }

        // Only interpolation writes into `blended`, so only interpolation is bounded by it.
        if (s.dim > Pick<T>::MAX_DIM) {
            out->status = PickStatus::DimTooLarge;
            return false;
        }
        SampleRef<T> a;
        SampleRef<T> b;
        if (!s.bracketing(t, &a, &b)) return false;
        out->status = PickStatus::Ok;
        out->present = true;
        out->valid = a.valid && b.valid;
        out->staleness = 0.0;
        if (b.t == a.t) {
            out->values = a.values;
            return true;
        }
        // Blend in double and narrow back, exactly as the reference does.
        const double w = (t - a.t) / (b.t - a.t);
        for (int j = 0; j < s.dim; ++j) {
            const double blended = (1.0 - w) * static_cast<double>(a.values[j]) +
                                   w * static_cast<double>(b.values[j]);
            out->blended[j] = static_cast<T>(blended);
        }
        out->values = out->blended;
        return true;
    }

    /// Whether a frame at `t` should be abandoned for staleness. Only `DropIfStale` does;
    /// the others contribute the stale sample and report how stale it was.
    bool refuses_stale(double staleness) const {
        if (policy_ != Policy::DropIfStale || !has_stale_limit()) return false;
        const double a = staleness < 0.0 ? -staleness : staleness;
        return a > max_stale_s_;
    }

    void note_frame() { ++frames_; }
    void note_dropped_stale() { ++dropped_stale_; }
    void note_no_data() { ++no_data_; }

private:
    Policy policy_ = Policy::HoldLast;
    double max_stale_s_ = 0.0;
    double consumed_to_ = -__builtin_inf();
    uint64_t frames_ = 0;
    uint64_t dropped_stale_ = 0;
    uint64_t no_data_ = 0;
};

}  // namespace argmind

#endif  // ARGMIND_ALIGN_HPP
