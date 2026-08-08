// SPDX-License-Identifier: Apache-2.0
//
// Wiring: streams to frames to an adapted model, behind one bounded `step()`.
// Ported from argmind/pipeline.py.
//
// The two questions the other modules could avoid have to be answered here, and the answers
// are the same ones the reference gives.
//
// **How does a frame become (x, y)?** By a declared index table, not a callable. The Python
// says so in a docstring and then resolves names to `(stream, component)` pairs at
// construction precisely so this side would have something to port; `FrameMap` here is that
// resolved table with the names already gone. Feature order is parameter order, so a state
// block restored against a different order is wrong undetectably — which is why
// `persist.hpp`'s fingerprint exists.
//
// **Where does staleness stop being a number nobody reads?** Here. The aligner reports it and
// the adapter cannot see it, so this is the only place that can act on it. The split is
// deliberate and is not collapsed:
//
//   - the pipeline gates on **input** health — staleness, sensor validity — *before* the
//     update, so known-bad data never reaches the learner;
//   - the `Guard` gates on **model** health — residual, gain, covariance — *after* it,
//     because those quantities do not exist until the update has been computed.
//
// Three structural differences from the reference, all forced rather than chosen
// -----------------------------------------------------------------------------
// **1. Frames are never materialised.** The Python builds a list of `Frame` objects in
// `Aligner.drain()` and then loops over it. A list needs a heap and a `Frame` needs a dict of
// arrays, so this fuses the two: one reference instant is aligned straight into the
// preallocated `x`/`y` and updated, then the next. That is observationally identical as long
// as nothing pushes to a stream mid-step — alignment does not mutate a stream and neither
// does an update — and it is the difference between one `Pick` buffer alive at a time and
// `MaxStreams` of them. For `Pick<double>` that is 280 bytes against 1,120 at four streams,
// on a target budgeting 8 KB.
//
// The cost of writing into `x` before the frame is known to exist: a frame that fails
// half-way leaves `x` holding some of this instant's streams and some of the last one's.
// Predicting from that would produce a plausible wrong answer and no error, which is the
// exact failure mode this project keeps finding. `frame_ready()` is false after a partial
// write and `predict_last()` refuses.
//
// **2. Views are passed to `step()`, never stored.** A `StreamView` snapshots `head` and
// `count`. A pipeline that captured one at construction would hold a picture of the ring
// frozen at that moment and go on aligning against it forever, which is a bug that looks
// like a sensor that stopped. So the caller hands them in per step and the pipeline owns
// nothing it can hold stale.
//
// **3. Every stream in one pipeline shares a scalar type.** The Python allows an f8 stream
// and an f4 stream in the same frame; here the aligner is `pick<T>` and the array is
// `StreamView<T>*`. Supporting a mixture needs either a tagged union with a switch on every
// element copy or a variadic pipeline type that a C ABI cannot name. It is a real reduction
// in what this side accepts and the parity harness therefore only exercises matched dtypes —
// stated so that "parity holds" is not read as covering a case nobody ran.
//
// One thing deliberately *not* repeated from `align.hpp`
// ------------------------------------------------------
// `Aligner` folds "no limit" and "a limit of zero" into one non-positive sentinel, which
// means a caller passing 0 meaning zero-tolerance silently gets no check. That is recorded
// there as a real difference. This gate does not repeat it: `gate_stale_` is a separate
// flag, so `None` and `0.0` stay distinguishable and a zero limit is *refused* by `valid()`
// exactly as the reference's constructor raises on it.

#ifndef ARGMIND_PIPELINE_HPP
#define ARGMIND_PIPELINE_HPP

#include <stdint.h>

#include "argmind/align.hpp"
#include "argmind/guard.hpp"
#include "argmind/rls.hpp"
#include "argmind/stream.hpp"

// `guard.hpp` is included for `GuardReason`, not because a pipeline requires a guard — it does
// not, and `guarded` is a trait precisely so an unguarded one stays constructible. `SkipReason`
// enumerates guard outcomes alongside adapter ones, so the names have to be in scope even in a
// translation unit that never instantiates a `Guard`. The Python does the same thing with
// strings and gets it for free; here it costs a header. Both are header-only and neither
// emits code that is not used.

namespace argmind {

/// Why a frame that existed did not adapt the model.
///
/// The Python reports these as strings in a dict; strings need storage this core does not
/// have, so they become an enum indexing a fixed array of counters. The mapping is exact
/// except for one collapse: the Python distinguishes `guard:<check>` per check name, and
/// here every guard trip lands in `GuardTripped` with the failing checks in the report's
/// `frozen_by` bitmask instead. No information is lost — the mask names more checks than the
/// string did, since the Python only ever reports the *first* one.
enum class SkipReason : uint8_t {
    Stale = 0,                   ///< input gate: worst staleness over the pipeline's limit
    Invalid = 1,                 ///< input gate: a sensor set its own validity flag false
    Frozen = 2,                  ///< the guard is frozen; the sample was only looked at
    GuardTripped = 3,            ///< a check failed on this sample and it was rolled back
    NonFiniteInput = 4,          ///< the adapter refused: NaN or infinity in x or y
    NonPositiveDenominator = 5,  ///< the adapter refused: the gain would be meaningless
    /// Not applied and no reason given. **Unreachable with the adapters this core ships**:
    /// `Rls::update` refuses only for the two reasons above, and `Guard` either passes one
    /// of those through or sets `guard_reason`. It exists for a future adapter, and the
    /// reference has the same unreachable `or "rejected"` fallback. A test asserting it
    /// stays at zero would be an assertion that cannot fail, so there is not one.
    Rejected = 6,
};
static constexpr int SKIP_REASON_COUNT = 7;

namespace detail {

/// Never defined; used only inside `decltype`. A hand-rolled `declval` so that this header
/// keeps the same zero-include discipline as the rest of the core — `<type_traits>` is
/// freestanding-conforming in C++17, but not depending on it at all is cheaper than
/// depending on a toolchain having got that right.
template <typename U>
const U& as_const_ref();

using yes = char;
struct no {
    char pad[2];
};

/// Does this adapter carry a freeze path? The Python asks the same question with
/// `getattr(self.adapter, "frozen", False)`, and for the same reason: an *unguarded*
/// pipeline has to be constructible, because the reference has to be runnable unguarded to
/// serve as the oracle this core is verified against. Guarding is the right default in the
/// field and is not enforced here.
template <typename A>
class HasFrozen {
    template <typename U>
    static auto probe(int) -> decltype(as_const_ref<U>().frozen(), yes());
    template <typename U>
    static no probe(...);

public:
    static constexpr bool value = sizeof(probe<A>(0)) == sizeof(yes);
};

/// Does this report come from a guard? Mirrors the Python's `getattr(report, "tripped", ())`.
template <typename R>
class HasGuardFields {
    template <typename U>
    static auto probe(int) -> decltype(as_const_ref<U>().guard_reason, yes());
    template <typename U>
    static no probe(...);

public:
    static constexpr bool value = sizeof(probe<R>(0)) == sizeof(yes);
};

}  // namespace detail

/// Which parts of a frame are the regressor and which are the target.
///
/// `N` and `M` are the adapter's own dimensions, so a map that produces the wrong number of
/// features is a compile error here where the reference raises at construction. What is left
/// for `valid()` is the count that cannot be expressed in the type: a map may declare fewer
/// features than `N` only if `constant` makes up the difference.
template <int N, int M>
class FrameMap {
    static_assert(N >= 1, "a regressor with no features is not a regressor");
    static_assert(M >= 1, "n_outputs must be >= 1");

public:
    static constexpr int n_features = N;
    static constexpr int n_outputs = M;

    /// Append one regressor component. Order is parameter order — the adapter's parameters
    /// are indexed by it, and a persisted state block restored against a different order is
    /// wrong in a way nothing downstream can detect.
    bool add_feature(int stream, int component) {
        if (n_feat_ >= N || stream < 0 || component < 0) return false;
        if (stream > 255 || component > 65535) return false;
        f_stream_[n_feat_] = static_cast<uint8_t>(stream);
        f_comp_[n_feat_] = static_cast<uint16_t>(component);
        ++n_feat_;
        return true;
    }

    bool add_target(int stream, int component) {
        if (n_targ_ >= M || stream < 0 || component < 0) return false;
        if (stream > 255 || component > 65535) return false;
        t_stream_[n_targ_] = static_cast<uint8_t>(stream);
        t_comp_[n_targ_] = static_cast<uint16_t>(component);
        ++n_targ_;
        return true;
    }

    /// Append a constant 1.0 to the regressor. Not a convenience: with it the adapter learns
    /// an offset term, which for online *bias* calibration is the whole point.
    void set_constant(bool on) { constant_ = on; }
    bool constant() const { return constant_; }

    int declared_features() const { return n_feat_; }
    int declared_targets() const { return n_targ_; }

    int feature_stream(int j) const { return f_stream_[j]; }
    int feature_component(int j) const { return f_comp_[j]; }
    int target_stream(int j) const { return t_stream_[j]; }
    int target_component(int j) const { return t_comp_[j]; }

    /// The declarations have to fill the adapter exactly. A short map would leave elements of
    /// `x` holding whatever the previous frame put there, which is the silent-wrong-answer
    /// shape again rather than a missing feature.
    bool valid() const {
        return n_feat_ + (constant_ ? 1 : 0) == N && n_targ_ == M;
    }

private:
    uint8_t f_stream_[N] = {};
    uint16_t f_comp_[N] = {};
    uint8_t t_stream_[M] = {};
    uint16_t t_comp_[M] = {};
    int n_feat_ = 0;
    int n_targ_ = 0;
    bool constant_ = false;
};

/// What one `step()` did. Observability is the return value, not a logging afterthought —
/// the same rule as `Report`, one level up.
template <int MaxStreams>
struct StepReport {
    /// False when the pipeline was not configured, or the views handed in do not match the
    /// shapes it was configured against. Distinct from `idle()`: this one means nothing was
    /// even attempted.
    bool ok = false;
    uint32_t frames = 0;
    uint32_t updated = 0;
    /// Reference-stream samples overwritten before they could become frames — input the
    /// pipeline will never see. Non-zero means `step()` is being called too rarely for the
    /// ring size.
    ///
    /// It counts the reference stream only, and that is the whole of what can honestly be
    /// counted. `Stream::dropped` is *overwrites*, which a healthy ring does on every push;
    /// a non-reference sample being overwritten is normal under hold-last, which only ever
    /// wants the newest one at or before the reference instant. A reference sample that is
    /// overwritten, by contrast, is a frame that will never exist.
    uint32_t lost_frames = 0;
    uint32_t skipped[SKIP_REASON_COUNT] = {};
    bool frozen = false;
    /// Checks that tripped during this step, OR-ed across frames. Zero for an unguarded
    /// pipeline, which cannot trip.
    uint32_t frozen_by = 0;
    /// Per-stream staleness of the most recent frame — the current picture, not an average.
    /// Signed, as the aligner reports it.
    double staleness[MaxStreams] = {};
    int n_staleness = 0;
    /// Worst absolute staleness anywhere in this step. This is the one to alarm on; the
    /// array above is for working out which stream to blame.
    double worst_staleness = 0.0;
    bool has_t = false;
    double t = 0.0;

    uint32_t total_skipped() const {
        uint32_t n = 0;
        for (int i = 0; i < SKIP_REASON_COUNT; ++i) n += skipped[i];
        return n;
    }

    /// No frames were available. Distinct from "frames arrived and none were used", which is
    /// a fault and this is not.
    bool idle() const { return frames == 0; }

    /// Every available frame adapted, nothing dropped, nothing frozen. One call for "is this
    /// pipeline actually working".
    bool healthy() const {
        return ok && updated == frames && lost_frames == 0 && total_skipped() == 0 &&
               !frozen;
    }
};

/// Streams to alignment to adaptation to prediction, behind one bounded `step()`.
///
/// `MaxStreams` is in the type because the staleness array and the shape table are, so the
/// memory is declared. The adapter is held **by value**, as `Guard` holds its own: on a
/// target where everything is statically allocated that removes a lifetime question rather
/// than answering it.
///
/// `step()` is pull-based and does bounded work: at most the reference stream's retained
/// sample count, then it returns. No threads, no callbacks, no event loop. The caller owns
/// timing, which is what an embedded integrator needs and what makes the whole thing
/// deterministically testable.
template <typename Adapter, int MaxStreams = 4>
class Pipeline {
    static_assert(MaxStreams >= 1, "a pipeline needs at least the reference stream");

public:
    using scalar = typename Adapter::scalar;
    using T = scalar;
    static constexpr int n_features = Adapter::n_features;
    static constexpr int n_outputs = Adapter::n_outputs;
    static constexpr int max_streams = MaxStreams;
    using Map = FrameMap<n_features, n_outputs>;
    using Step = StepReport<MaxStreams>;

    /// Whether this pipeline's adapter carries a freeze path. False is legitimate for an
    /// offline reference run and a bad idea in the field.
    static constexpr bool guarded = detail::HasFrozen<Adapter>::value;

    /// `sizeof(Pick<T>)` — the whole pick record, of which `MAX_DIM * sizeof(T)` is the
    /// interpolation scratch buffer and the rest is the status, the staleness and the
    /// borrowed pointer. Measured on x86-64: 152 bytes at `float`, 280 at `double`.
    ///
    /// It is `sizeof` and not a hand-summed figure on purpose. `stream.hpp` shipped a
    /// `bytes` constant that added up the payload, told callers to budget from it, and was
    /// 24 bytes short of the object; a unit check now pins this one against `sizeof` so the
    /// two cannot drift.
    ///
    /// One of these is held, not `MaxStreams` of them — that is what fusing the frame loop
    /// bought, and at four streams it is 280 bytes against 1,120 for a `double` pipeline.
    static constexpr int pick_bytes = static_cast<int>(sizeof(Pick<T>));

    Pipeline(const Adapter& adapter, const Aligner& aligner, const Map& map, int reference)
        : adapter_(adapter), aligner_(aligner), map_(map), reference_(reference) {}

    // -- configuration ------------------------------------------------------

    /// Frames whose worst staleness exceeds `limit` do not adapt. They are still counted and
    /// still reported. Not setting it means no input gate on staleness, which is rarely
    /// right: a stream that quietly died then contributes its last value to every update
    /// forever.
    void set_max_stale_s(double limit) {
        gate_stale_ = true;
        max_stale_s_ = limit;
    }
    void clear_max_stale_s() { gate_stale_ = false; }

    /// Skip adaptation on frames where any contributing sensor set its own validity flag
    /// false. On by default — the sensor said the reading was bad and it is the only thing
    /// in the system that knows.
    void set_require_valid(bool on) { require_valid_ = on; }

    /// `Nearest` and `Interpolate` consult samples from after the reference instant. That is
    /// legitimate offline and a correctness bug in a causal real-time loop, which is what a
    /// pipeline is, so it has to be said out loud here even though the aligner alone permits
    /// it.
    void set_allow_non_causal(bool on) { allow_non_causal_ = on; }

    /// Validate the wiring against the streams that actually exist, once. Returns false
    /// instead of raising; a caller that ignores it gets a `step()` that reports `ok=false`
    /// and does nothing, rather than an out-of-range read.
    ///
    /// A typo or an index past the end of a sensor fails here rather than on the first frame
    /// at 3am, which is the reason the reference resolves at construction too.
    bool configure(const StreamView<T>* __restrict views, int n_streams) {
        configured_ = false;
        if (views == nullptr || n_streams < 1 || n_streams > MaxStreams) return false;
        if (reference_ < 0 || reference_ >= n_streams) return false;
        if (!aligner_.causal() && !allow_non_causal_) return false;
        if (gate_stale_ && !(max_stale_s_ > 0.0)) return false;
        if (!map_.valid()) return false;
        for (int i = 0; i < n_streams; ++i) {
            if (views[i].dim < 1) return false;
        }
        for (int j = 0; j < map_.declared_features(); ++j) {
            const int s = map_.feature_stream(j);
            if (s >= n_streams || map_.feature_component(j) >= views[s].dim) return false;
        }
        for (int j = 0; j < n_outputs; ++j) {
            const int s = map_.target_stream(j);
            if (s >= n_streams || map_.target_component(j) >= views[s].dim) return false;
        }
        n_streams_ = n_streams;
        for (int i = 0; i < n_streams; ++i) dims_[i] = views[i].dim;
        configured_ = true;
        return true;
    }

    bool configured() const { return configured_; }

    // -- inspection ---------------------------------------------------------

    const Adapter& adapter() const { return adapter_; }
    Adapter& adapter_mutable() { return adapter_; }
    const Aligner& aligner() const { return aligner_; }
    Aligner& aligner_mutable() { return aligner_; }
    const Map& map() const { return map_; }

    /// Mutable access to the frame map, for a caller that builds it in pieces — which the C
    /// ABI has to, since a C caller cannot construct a `FrameMap<N, M>` and hand it to a
    /// constructor.
    ///
    /// It **clears the configuration**, and that is the point rather than a side effect. The
    /// map is exactly what `configure()` bounds-checked against the streams; leaving the
    /// pipeline configured would let a caller widen a component past a sensor's width after
    /// the check had passed, and the next `step()` would read off the end of a ring.
    Map& map_mutable() {
        configured_ = false;
        return map_;
    }
    int reference() const { return reference_; }

    /// Cumulative across every `step()`. 64-bit for the same reason `Stream::dropped` is:
    /// these advance once per frame and a diagnostic that silently restarts after 49.7 days
    /// at 1 kHz is worse than one that is merely large.
    uint64_t steps() const { return steps_; }
    uint64_t total_frames() const { return total_frames_; }
    uint64_t total_updated() const { return total_updated_; }
    uint64_t total_lost_frames() const { return total_lost_; }

    bool frozen() const { return frozen_now(); }

    /// The regressor and target the last **complete** frame produced. Meaningless — and
    /// declared so — after a frame that failed part-way, because the extraction writes as it
    /// goes and a failed frame leaves a mixture of two instants behind.
    bool frame_ready() const { return frame_ready_; }
    const T* last_x() const { return x_; }
    const T* last_y() const { return y_; }
    double last_frame_t() const { return frame_t_; }

    // -- the work -----------------------------------------------------------

    /// Consume every frame the streams can currently produce, and report.
    ///
    /// Bounded: at most the reference stream's retained sample count, one adapter update per
    /// frame. Nothing here waits and nothing allocates.
    Step step(const StreamView<T>* __restrict views) {
        Step rep;
        if (!configured_ || views == nullptr) return rep;
        for (int i = 0; i < n_streams_; ++i) {
            // A view whose dimension changed is a different stream wearing the same index,
            // and the frame map's components were bounds-checked against the old one.
            if (views[i].dim != dims_[i]) return rep;
        }
        rep.ok = true;

        const StreamView<T>& ref = views[reference_];

        // Loss is arrivals minus what the aligner got to look at. Nothing is ever removed
        // from a ring, so `dropped + count` is the total ever pushed; the aligner considered
        // one reference instant per frame it emitted plus one per instant it declined to
        // build. Anything unaccounted for was overwritten before it could become a frame.
        const uint64_t arrived = ref.dropped + static_cast<uint64_t>(ref.count);
        const uint64_t before_no_data = aligner_.no_data();
        const uint64_t before_stale = aligner_.dropped_stale();

        uint32_t frames = 0;
        uint32_t updated = 0;
        uint32_t tripped = 0;
        double worst = 0.0;
        bool have_stale = false;
        double last_t = 0.0;
        bool have_t = false;

        for (int age = 0; age < ref.count; ++age) {
            const double t = ref.times[ref.slot(age)];
            if (t <= aligner_.consumed_to()) continue;
            const bool built = build_frame(views, t, /*count=*/true);
            // Advanced whether or not a frame came out, exactly as `drain()` does: an
            // instant the aligner declined is still an instant it has now seen, and
            // re-offering it next step would build the same nothing again.
            aligner_.set_consumed_to(t);
            if (!built) continue;

            aligner_.note_frame();
            ++frames;
            last_t = t;
            have_t = true;
            have_stale = true;
            for (int i = 0; i < n_streams_; ++i) rep.staleness[i] = stale_[i];
            rep.n_staleness = n_streams_;
            if (frame_worst_ > worst) worst = frame_worst_;

            // Input gate, before the learner sees anything.
            if (gate_stale_ && frame_worst_ > max_stale_s_) {
                ++rep.skipped[static_cast<int>(SkipReason::Stale)];
                continue;
            }
            if (require_valid_ && !frame_all_valid_) {
                ++rep.skipped[static_cast<int>(SkipReason::Invalid)];
                continue;
            }

            const auto r = adapter_.update(x_, y_);
            if (r.applied) {
                ++updated;
            } else {
                ++rep.skipped[static_cast<int>(classify(r))];
            }
            tripped |= tripped_of(r);
        }

        const uint64_t considered = static_cast<uint64_t>(frames) +
                                    (aligner_.no_data() - before_no_data) +
                                    (aligner_.dropped_stale() - before_stale);
        const uint64_t owed = arrived - arrived_;
        // The clamp matches the reference's `max(0, ...)` and is **dead on both sides**.
        // Each visited instant increments exactly one of frames/no_data/dropped_stale, so
        // `considered` is the number of instants visited; and every unconsumed reference
        // sample in the ring arrived after the last successful step, so `considered <= owed`
        // holds by construction. A mutation check confirms it — removing the clamp survives
        // the whole suite, including a test written specifically to kill it. It stays
        // because the subtraction is unsigned and a wrap would report about 1.8e19 lost
        // frames rather than a small wrong number. Recorded rather than removed, and
        // recorded rather than claimed to be tested.
        const uint64_t lost = owed > considered ? owed - considered : 0;
        arrived_ = arrived;

        rep.frames = frames;
        rep.updated = updated;
        rep.lost_frames = static_cast<uint32_t>(lost);
        rep.frozen = frozen_now();
        rep.frozen_by = tripped;
        rep.worst_staleness = worst;
        rep.has_t = have_t;
        rep.t = last_t;
        if (!have_stale) rep.n_staleness = 0;

        ++steps_;
        total_frames_ += frames;
        total_updated_ += updated;
        total_lost_ += lost;
        return rep;
    }

    /// Straight through to the adapter. Unaffected by freezing — that is the point of
    /// freezing.
    void predict(const T* __restrict x, T* __restrict out) const {
        adapter_.predict(x, out);
    }

    /// Predict from the frame at reference instant `t`, using the same map `step()` uses.
    ///
    /// Exists so a caller cannot get the regressor layout right in training and wrong at
    /// inference, which is a mistake that produces plausible numbers and no error.
    ///
    /// It does **not** advance the drain watermark and does **not** move `no_data` or
    /// `dropped_stale`, because a prediction consumes nothing. The reference's
    /// `predict_frame` takes a `Frame` that `drain()` already counted, so this matches its
    /// behaviour rather than diverging from it — but it has to be said, because this call
    /// does the aligning itself and the reference's does not.
    bool predict_at(const StreamView<T>* __restrict views, double t, T* __restrict out) {
        if (!configured_ || views == nullptr) return false;
        for (int i = 0; i < n_streams_; ++i) {
            if (views[i].dim != dims_[i]) return false;
        }
        if (!build_frame(views, t, /*count=*/false)) return false;
        adapter_.predict(x_, out);
        return true;
    }

    /// Predict from the last complete frame. Refuses after a frame that failed part-way.
    bool predict_last(T* __restrict out) const {
        if (!frame_ready_) return false;
        adapter_.predict(x_, out);
        return true;
    }

private:
    /// Align every stream at `t` and scatter into `x_`/`y_`. False if the frame cannot be
    /// built, which is the same set of conditions the reference's `align()` returns None on:
    /// a stream with nothing to offer, or — under `DropIfStale` only — a stream past its
    /// limit.
    ///
    /// `count` is false for prediction, where nothing is being consumed.
    bool build_frame(const StreamView<T>* __restrict views, double t, bool count) {
        frame_ready_ = false;
        frame_all_valid_ = true;
        frame_worst_ = 0.0;
        for (int i = 0; i < n_streams_; ++i) {
            if (!aligner_.pick<T>(views[i], t, &pick_)) {
                if (count) aligner_.note_no_data();
                return false;
            }
            if (aligner_.refuses_stale(pick_.staleness)) {
                if (count) aligner_.note_dropped_stale();
                return false;
            }
            stale_[i] = pick_.staleness;
            const double a = pick_.staleness < 0.0 ? -pick_.staleness : pick_.staleness;
            if (a > frame_worst_) frame_worst_ = a;
            if (!pick_.valid) frame_all_valid_ = false;

            // Scatter rather than gather: the pick is alive only for this stream, so
            // everything the frame map wants from it has to be taken now. O(streams x N)
            // with both small, against holding every stream's `Pick` at once.
            for (int j = 0; j < map_.declared_features(); ++j) {
                if (map_.feature_stream(j) == i) {
                    x_[j] = pick_.values[map_.feature_component(j)];
                }
            }
            for (int j = 0; j < n_outputs; ++j) {
                if (map_.target_stream(j) == i) {
                    y_[j] = pick_.values[map_.target_component(j)];
                }
            }
        }
        // Last, as the reference puts it: `self._x[-1] = 1.0`.
        if (map_.constant()) x_[n_features - 1] = T(1);
        frame_t_ = t;
        frame_ready_ = true;
        return true;
    }

    bool frozen_now() const {
        if constexpr (detail::HasFrozen<Adapter>::value) {
            return adapter_.frozen();
        } else {
            return false;
        }
    }

    template <typename R>
    static SkipReason classify(const R& r) {
        if constexpr (detail::HasGuardFields<R>::value) {
            if (r.guard_reason == GuardReason::Frozen) return SkipReason::Frozen;
            if (r.guard_reason == GuardReason::Tripped) return SkipReason::GuardTripped;
        }
        switch (r.reason) {
            case Reason::NonFiniteInput: return SkipReason::NonFiniteInput;
            case Reason::NonPositiveDenominator: return SkipReason::NonPositiveDenominator;
            case Reason::None: break;
        }
        return SkipReason::Rejected;
    }

    template <typename R>
    static uint32_t tripped_of(const R& r) {
        if constexpr (detail::HasGuardFields<R>::value) {
            return r.tripped;
        } else {
            (void)r;
            return 0u;
        }
    }

    Adapter adapter_;
    Aligner aligner_;
    Map map_;
    int reference_ = 0;

    bool gate_stale_ = false;
    double max_stale_s_ = 0.0;
    bool require_valid_ = true;
    bool allow_non_causal_ = false;

    bool configured_ = false;
    int n_streams_ = 0;
    int dims_[MaxStreams] = {};

    // Preallocated. Extracting a regressor from a frame allocates nothing.
    T x_[n_features] = {};
    T y_[n_outputs] = {};
    double stale_[MaxStreams] = {};
    Pick<T> pick_;

    bool frame_ready_ = false;
    bool frame_all_valid_ = true;
    double frame_worst_ = 0.0;
    double frame_t_ = 0.0;

    /// Starts at zero, not at what the streams already hold: samples pushed before this
    /// pipeline existed are still owed a frame, and if the ring overflowed before the first
    /// step then that loss is real and should be reported on it.
    uint64_t arrived_ = 0;
    uint64_t steps_ = 0;
    uint64_t total_frames_ = 0;
    uint64_t total_updated_ = 0;
    uint64_t total_lost_ = 0;
};

}  // namespace argmind

#endif  // ARGMIND_PIPELINE_HPP
