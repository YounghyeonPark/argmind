// SPDX-License-Identifier: Apache-2.0
//
// The pipeline, with its shape decided at construction. Hosted tier.
//
// Half of this was already runtime and did not need porting, which is worth saying
// because it is why this file is shorter than `pipeline.hpp`. `Aligner` takes its policy,
// reference and staleness limit as ordinary values; `StreamView<T>` carries a runtime
// `dim`, and `Ring<T>` a runtime `dim` and `capacity` over caller storage. Both were made
// that way for the C ABI, so the hosted tier inherits them unchanged. Only `FrameMap<N,M>`
// and `Pipeline<Adapter, MaxStreams>` fixed anything at compile time.
//
// What a pipeline has to agree about is BOOKKEEPING, not arithmetic. It is driven by two
// independent clocks -- samples arriving and `step()` being called -- and every figure it
// reports is a difference between counters taken at different moments. So the properties
// matched here are the ones `pipeline.hpp` lists:
//
//   - the reference stream's instants drive frames; the aligner advances past an instant
//     it declined, exactly as `drain()` does, because re-offering it would build the same
//     nothing again
//   - two gates at two different times: the pipeline checks INPUT health before the
//     update and the guard checks MODEL health after it
//   - a frame is built into scratch and committed on success, so an instant that fails
//     part-way leaves the previous complete frame intact
//   - the prediction is present whenever a frame existed, including one the input gate
//     refused to learn from
//
// Frames are still never materialised as objects: alignment writes straight into the
// regressor one instant at a time. That is a heap tier and it could afford a list of
// them; it does not, because the reference does not either and the difference would show
// up as a parity failure in `lost_frames` rather than as a design note.

#ifndef ARGMIND_HOST_RUNTIME_SHAPE_PIPELINE_HPP
#define ARGMIND_HOST_RUNTIME_SHAPE_PIPELINE_HPP

#include <stdint.h>

#include <vector>

#include "argmind/align.hpp"
#include "argmind/pipeline.hpp"   // SkipReason and the skip-count vocabulary
#include "argmind/stream.hpp"

namespace argmind {
namespace host {

/// Which stream and component each feature and target comes from, sized at construction.
///
/// Feature ORDER is parameter order, the same rule the core states: a persisted state
/// block restored against a different order is wrong undetectably, which is why
/// `persist`'s configuration fingerprint exists.
class RuntimeShapedFrameMap {
public:
    RuntimeShapedFrameMap() = default;

    bool add_feature(int stream, int component) {
        if (stream < 0 || component < 0) return false;
        feat_.push_back({stream, component});
        return true;
    }
    bool add_target(int stream, int component) {
        if (stream < 0 || component < 0) return false;
        targ_.push_back({stream, component});
        return true;
    }
    /// Appends a 1 as the LAST feature, matching the reference's `self._x[-1] = 1.0`.
    void set_constant(bool on) { constant_ = on; }

    bool constant() const { return constant_; }
    int declared_features() const { return static_cast<int>(feat_.size()); }
    int n_features() const { return declared_features() + (constant_ ? 1 : 0); }
    int n_outputs() const { return static_cast<int>(targ_.size()); }

    int feature_stream(int j) const { return feat_[static_cast<size_t>(j)].stream; }
    int feature_component(int j) const { return feat_[static_cast<size_t>(j)].component; }
    int target_stream(int j) const { return targ_[static_cast<size_t>(j)].stream; }
    int target_component(int j) const { return targ_[static_cast<size_t>(j)].component; }

private:
    struct Entry {
        int stream;
        int component;
    };
    std::vector<Entry> feat_;
    std::vector<Entry> targ_;
    bool constant_ = false;
};

/// What one `step()` did.
template <typename T>
struct RuntimeShapedStep {
    bool ok = false;
    bool frozen = false;
    bool has_t = false;
    uint32_t frames = 0;
    uint32_t updated = 0;
    uint64_t lost_frames = 0;
    uint32_t frozen_by = 0;
    /// Indexed by `SkipReason`. Eight, matching `AM_SKIP_COUNT` in `argmind.h` -- the
    /// enum has no `Count` member and inventing one here would be a second definition
    /// of the same number.
    uint32_t skipped[8] = {};
    /// How much of `staleness` the pipeline actually filled in. **Zero on a step that
    /// built no frame**, which is not the same as `!ok`: the first version of this port
    /// derived it from `ok` and disagreed with the reference on the very first step of
    /// a run whose aiding stream started late.
    int n_staleness = 0;
    double worst_staleness = 0.0;
    double t = 0.0;
    std::vector<double> staleness;
    bool has_prediction = false;
    std::vector<double> prediction;
};

/// Streams in, one bounded step, a report out -- with the shape decided at construction.
template <typename Adapter>
class RuntimeShapedPipeline {
public:
    using T = typename Adapter::scalar;

    RuntimeShapedPipeline(Adapter adapter, Aligner aligner, RuntimeShapedFrameMap map,
                          int reference)
        : adapter_(adapter), aligner_(aligner), map_(map), reference_(reference) {}

    void set_max_stale_s(double limit) {
        gate_stale_ = true;
        max_stale_s_ = limit;
    }
    void clear_max_stale_s() { gate_stale_ = false; }
    void set_require_valid(bool on) { require_valid_ = on; }
    void set_allow_non_causal(bool on) { allow_non_causal_ = on; }
    void set_require_fresh_target(bool on) { require_fresh_target_ = on; }

    /// **Clears the configuration on purpose.** The map is what `configure()`
    /// bounds-checked, so mutating it and staying configured would let a caller widen a
    /// component past a sensor's width after the check.
    RuntimeShapedFrameMap& map_mutable() {
        configured_ = false;
        return map_;
    }
    const RuntimeShapedFrameMap& map() const { return map_; }

    /// Bind to a set of streams: bounds-check every map entry against the widths those
    /// streams actually have, and size the working buffers.
    bool configure(const StreamView<T>* views, int n_streams) {
        configured_ = false;
        if (views == nullptr || n_streams < 1) return false;
        if (reference_ < 0 || reference_ >= n_streams) return false;
        if (map_.n_features() != adapter_.n_features()) return false;
        if (map_.n_outputs() != adapter_.n_outputs()) return false;
        if (!aligner_.causal() && !allow_non_causal_) return false;

        for (int j = 0; j < map_.declared_features(); ++j) {
            const int s = map_.feature_stream(j);
            if (s < 0 || s >= n_streams) return false;
            if (map_.feature_component(j) >= views[s].dim) return false;
        }
        for (int j = 0; j < map_.n_outputs(); ++j) {
            const int s = map_.target_stream(j);
            if (s < 0 || s >= n_streams) return false;
            if (map_.target_component(j) >= views[s].dim) return false;
        }

        n_streams_ = n_streams;
        dims_.assign(static_cast<size_t>(n_streams), 0);
        for (int i = 0; i < n_streams; ++i) dims_[static_cast<size_t>(i)] = views[i].dim;
        stale_.assign(static_cast<size_t>(n_streams), 0.0);
        sample_t_.assign(static_cast<size_t>(n_streams), 0.0);
        const size_t nf = static_cast<size_t>(adapter_.n_features());
        const size_t nm = static_cast<size_t>(adapter_.n_outputs());
        x_.assign(nf, T(0));
        y_.assign(nm, T(0));
        sx_.assign(nf, T(0));
        sy_.assign(nm, T(0));
        last_target_t_.assign(nm, 0.0);
        have_last_target_ = false;
        frame_ready_ = false;
        arrived_ = 0;
        configured_ = true;
        return true;
    }

    bool configured() const { return configured_; }
    bool frame_ready() const { return frame_ready_; }

    /// Cumulative counters, the same set the core exposes. Part of the contract rather
    /// than diagnostics: a parity harness compares the totals as well as each step,
    /// because a per-step comparison that agreed while the totals did not would mean the
    /// two had disagreed somewhere the harness never looked.
    uint64_t steps() const { return steps_; }
    uint64_t total_frames() const { return total_frames_; }
    uint64_t total_updated() const { return total_updated_; }
    uint64_t total_lost_frames() const { return total_lost_; }
    const Aligner& aligner() const { return aligner_; }
    bool frozen() const { return frozen_of(); }
    const Adapter& adapter() const { return adapter_; }
    Adapter& adapter_mutable() { return adapter_; }
    double last_frame_t() const { return frame_t_; }

    RuntimeShapedStep<T> step(const StreamView<T>* views) {
        RuntimeShapedStep<T> rep;
        rep.staleness.assign(static_cast<size_t>(n_streams_), 0.0);
        if (!configured_ || views == nullptr) return rep;
        ++steps_;
        for (int i = 0; i < n_streams_; ++i) {
            // A view whose dimension changed is a different stream wearing the same
            // index, and the map's components were bounds-checked against the old one.
            if (views[i].dim != dims_[static_cast<size_t>(i)]) return rep;
        }
        rep.ok = true;

        const StreamView<T>& ref = views[reference_];
        const uint64_t arrived = ref.dropped + static_cast<uint64_t>(ref.count);
        const uint64_t before_no_data = aligner_.no_data();
        const uint64_t before_stale = aligner_.dropped_stale();

        uint32_t frames = 0;
        double worst = 0.0;

        for (int age = 0; age < ref.count; ++age) {
            const double t = ref.times[ref.slot(age)];
            if (t <= aligner_.consumed_to()) continue;
            const bool built = build_frame(views, t);
            aligner_.set_consumed_to(t);
            if (!built) continue;

            aligner_.note_frame();
            ++frames;
            rep.t = t;
            rep.has_t = true;
            for (int i = 0; i < n_streams_; ++i) {
                rep.staleness[static_cast<size_t>(i)] = stale_[static_cast<size_t>(i)];
            }
            rep.n_staleness = n_streams_;
            if (frame_worst_ > worst) worst = frame_worst_;

            // Input gate, before the learner sees anything. Model health is the guard's
            // job and happens after the update, because those quantities do not exist
            // until then -- the two must not merge.
            if (gate_stale_ && frame_worst_ > max_stale_s_) {
                ++rep.skipped[static_cast<int>(SkipReason::Stale)];
                continue;
            }
            if (require_valid_ && !frame_all_valid_) {
                ++rep.skipped[static_cast<int>(SkipReason::Invalid)];
                continue;
            }
            if (require_fresh_target_ && !target_is_new()) {
                ++rep.skipped[static_cast<int>(SkipReason::StaleTarget)];
                continue;
            }

            const auto r = adapter_.update(x_.data(), y_.data());
            if (r.applied) {
                ++rep.updated;
            } else {
                ++rep.skipped[static_cast<int>(classify(r))];
            }
            rep.frozen_by |= tripped_of(r);
        }

        const uint64_t considered = static_cast<uint64_t>(frames) +
                                    (aligner_.no_data() - before_no_data) +
                                    (aligner_.dropped_stale() - before_stale);
        const uint64_t owed = arrived - arrived_;
        // Dead on both sides and kept for the same reason the core keeps it: each visited
        // instant increments exactly one of frames/no_data/dropped_stale, so
        // `considered <= owed` holds by construction. The subtraction is unsigned and a
        // wrap would report about 1.8e19 lost frames rather than a small wrong number.
        rep.lost_frames = owed > considered ? owed - considered : 0;
        arrived_ = arrived;

        rep.frames = frames;
        rep.worst_staleness = worst;
        total_frames_ += frames;
        total_updated_ += rep.updated;
        total_lost_ += rep.lost_frames;
        rep.frozen = frozen_of();

        // `frames` alone, not `frames && frame_ready_`. The second condition was a bug in
        // the core that a parity review caught: `build_frame` cleared the flag on entry,
        // so a step whose LATER instant failed reported frames and no output while the
        // reference, which only sees frames that exist, predicted. 118 disagreeing steps
        // in 250.
        //
        // **Adding `&& frame_ready_` back here is an EQUIVALENT mutant, and that is worth
        // recording rather than contriving a case for.** This port took the fixed shape:
        // `build_frame` never clears the flag, only sets it on a successful commit, so
        // once any frame has been built the extra condition is always true. Reintroducing
        // the bug needs both halves -- the clear AND the condition -- and a mutation check
        // confirms the condition alone changes nothing, including on the case with
        // declined instants that was added specifically to reach it.
        if (frames) {
            rep.prediction.assign(static_cast<size_t>(adapter_.n_outputs()), 0.0);
            std::vector<T> out(static_cast<size_t>(adapter_.n_outputs()));
            adapter_.predict(x_.data(), out.data());
            for (int j = 0; j < adapter_.n_outputs(); ++j) {
                rep.prediction[static_cast<size_t>(j)] = static_cast<double>(out[j]);
            }
            rep.has_prediction = true;
        }
        return rep;
    }

private:
    /// The same mapping `pipeline.hpp` makes, including the guard-reason branch: a
    /// frozen sample and a tripped one are different skips, and collapsing them would
    /// make a guarded pipeline's counters disagree with the reference's.
    template <typename R>
    static SkipReason classify(const R& r) {
        // `detail::HasGuardFields`, not a C++20 `requires` expression. The rest of the
        // library is C++17 and CMake sets that standard; the first draft here used
        // `requires` and quietly needed -std=c++20 to build.
        if constexpr (argmind::detail::HasGuardFields<R>::value) {
            if (r.guard_reason == GuardReason::Frozen) return SkipReason::Frozen;
            if (r.guard_reason == GuardReason::Tripped) return SkipReason::GuardTripped;
        }
        switch (r.reason) {
            case Reason::NonFiniteInput: return SkipReason::NonFiniteInput;
            case Reason::NonPositiveDenominator:
                return SkipReason::NonPositiveDenominator;
            case Reason::None: break;
        }
        // Unreachable with the adapters shipped here, exactly as in the core, and the
        // reference has the same fallback. No test asserts it stays zero, because that
        // assertion could not fail.
        return SkipReason::Rejected;
    }

    template <typename R>
    static uint32_t tripped_of(const R& r) {
        if constexpr (argmind::detail::HasGuardFields<R>::value) {
            return r.tripped;
        } else {
            return 0u;
        }
    }

    bool frozen_of() const {
        if constexpr (argmind::detail::HasFrozen<Adapter>::value) {
            return adapter_.frozen();
        } else {
            return false;
        }
    }

    bool build_frame(const StreamView<T>* views, double t) {
        frame_all_valid_ = true;
        frame_worst_ = 0.0;
        for (int i = 0; i < n_streams_; ++i) {
            if (!aligner_.template pick<T>(views[i], t, &pick_)) {
                aligner_.note_no_data();
                return false;
            }
            if (aligner_.refuses_stale(pick_.staleness)) {
                aligner_.note_dropped_stale();
                return false;
            }
            stale_[static_cast<size_t>(i)] = pick_.staleness;
            sample_t_[static_cast<size_t>(i)] = pick_.sample_t;
            const double a = pick_.staleness < 0.0 ? -pick_.staleness : pick_.staleness;
            if (a > frame_worst_) frame_worst_ = a;
            if (!pick_.valid) frame_all_valid_ = false;

            // Scatter rather than gather: the pick is alive only for this stream.
            for (int j = 0; j < map_.declared_features(); ++j) {
                if (map_.feature_stream(j) == i) {
                    sx_[static_cast<size_t>(j)] = pick_.values[map_.feature_component(j)];
                }
            }
            for (int j = 0; j < map_.n_outputs(); ++j) {
                if (map_.target_stream(j) == i) {
                    sy_[static_cast<size_t>(j)] = pick_.values[map_.target_component(j)];
                }
            }
        }
        if (map_.constant()) sx_[sx_.size() - 1] = T(1);
        // Commit. Scratch until here, so an instant that fails part-way leaves the
        // previous complete frame intact rather than a mixture of two.
        x_ = sx_;
        y_ = sy_;
        frame_t_ = t;
        frame_ready_ = true;
        return true;
    }

    /// Has the target sample changed? Decided on the sample's own timestamp, which the
    /// aligner carries on the pick. Recorded on the ATTEMPT, not on success, so a target the
    /// adapter rejects is not retried nine times and counted as nine faults.
    ///
    /// It used to rebuild the timestamp as `t_reference + staleness`, which is not bit-stable
    /// across reference instants; the same physical sample then read as new after a gap in
    /// the target stream. Measured in the reference: 52 of 120 frames relearned a
    /// 14-second-old target.
    bool target_is_new() {
        bool changed = false;
        for (int j = 0; j < map_.n_outputs(); ++j) {
            const int i = map_.target_stream(j);
            const double stamp = sample_t_[static_cast<size_t>(i)];
            if (!have_last_target_ || last_target_t_[static_cast<size_t>(j)] != stamp) {
                changed = true;
            }
            last_target_t_[static_cast<size_t>(j)] = stamp;
        }
        have_last_target_ = true;
        return changed;
    }

    Adapter adapter_;
    Aligner aligner_;
    RuntimeShapedFrameMap map_;
    int reference_ = 0;

    bool configured_ = false;
    int n_streams_ = 0;
    std::vector<int> dims_;
    std::vector<double> stale_;
    /// Each stream's contributing sample's own timestamp, carried from the pick rather
    /// than rebuilt as `frame_t_ + stale_[i]`; see `target_is_new`.
    std::vector<double> sample_t_;
    std::vector<T> x_, y_, sx_, sy_;
    std::vector<double> last_target_t_;
    bool have_last_target_ = false;

    Pick<T> pick_;
    bool frame_ready_ = false;
    bool frame_all_valid_ = true;
    double frame_worst_ = 0.0;
    double frame_t_ = 0.0;

    bool gate_stale_ = false;
    double max_stale_s_ = 0.0;
    bool require_valid_ = true;
    bool allow_non_causal_ = false;
    bool require_fresh_target_ = false;

    uint64_t arrived_ = 0;
    uint64_t steps_ = 0;
    uint64_t total_frames_ = 0;
    uint64_t total_updated_ = 0;
    uint64_t total_lost_ = 0;
};

}  // namespace host
}  // namespace argmind

#endif  // ARGMIND_HOST_RUNTIME_SHAPE_PIPELINE_HPP
