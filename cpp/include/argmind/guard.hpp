// SPDX-License-Identifier: Apache-2.0
//
// Health checks, freezing and bounded rollback. Ported from argmind/guard.py.
//
// The Python is the specification. Everything about *when* the guard acts is matched
// exactly; what changes is how the checks are held, and that change is forced by the
// target contract rather than chosen.
//
// How the checks are held, and why it is not a list of polymorphic objects
// ------------------------------------------------------------------------
// The Python holds a list of `Check` subclasses and calls `evaluate` on each. That
// needs virtual dispatch and heap-allocated objects, and the core has neither: no
// virtuals, no allocation, no exceptions, no RTTI. So the checks become concrete
// members with declared capacity — an array of `Bound` up to `MaxBounds`, plus one
// optional `Divergence`. Their identities become an enum instead of a string, which
// is what a `tripped` bitmask can be built from.
//
// The cost is real and worth naming: a user cannot supply their own check type the way
// they can in Python. Adding a kind of check means adding it here. That is the same
// trade the C ABI makes for compile-time dimensions, and it is made for the same
// reason — a design that took a function pointer per check would restore the
// flexibility and give up the inlining that is the point of this core.
//
// What is matched exactly
// ----------------------
//   - checks run *after* the adapter updates, because the quantities they watch do not
//     exist until then, which is why there is rollback rather than prevention
//   - every check runs; the guard does not stop at the first failure
//   - a trip rolls back to the *deepest* retained checkpoint and freezes
//   - stateful checks keep their history through a freeze and are reset on thaw. Both
//     halves of that were design errors in the Python before they were fixed, and
//     getting them backwards here would reintroduce a guard that thaws itself in the
//     middle of a fault
//   - an adapter rejection passes through and is counted separately: a NaN sample and a
//     diverging learner are different faults
//   - while frozen the guard evaluates a probe whose residual is real, whose gain is
//     genuinely zero, and whose covariance scalars are those of the restored checkpoint

#ifndef ARGMIND_GUARD_HPP
#define ARGMIND_GUARD_HPP

#include <stdint.h>

#include "argmind/mat.hpp"
#include "argmind/rls.hpp"

namespace argmind {

/// A report field a `Bound` may watch. The enum replaces the Python's string field
/// name; a name that does not exist becomes a compile error rather than a guard that
/// silently never fires.
enum class Field : uint8_t {
    ResidualNorm = 0,
    ResidualPostNorm = 1,
    GainNorm = 2,
    PTrace = 3,
    PMinDiag = 4,
};

/// The bit a `Field` occupies in an adapter's `health_fields` mask.
constexpr uint32_t field_bit(Field f) { return 1u << static_cast<uint32_t>(f); }

/// For an adapter that can report every field, which `Rls` can because the last two
/// are its covariance.
constexpr uint32_t ALL_FIELDS =
    field_bit(Field::ResidualNorm) | field_bit(Field::ResidualPostNorm) |
    field_bit(Field::GainNorm) | field_bit(Field::PTrace) | field_bit(Field::PMinDiag);

/// The three any adapter can report, because they come out of the update itself rather
/// than out of a particular algorithm's internals.
constexpr uint32_t RESIDUAL_FIELDS = field_bit(Field::ResidualNorm) |
                                     field_bit(Field::ResidualPostNorm) |
                                     field_bit(Field::GainNorm);

/// Whether an adapter reports anything that needs `health()` at all. An adapter that
/// declares neither covariance field need not implement it: the guard's calls sit
/// behind `if constexpr`, so the branch is never instantiated.
constexpr bool needs_health(uint32_t mask) {
    return (mask & (field_bit(Field::PTrace) | field_bit(Field::PMinDiag))) != 0;
}

// `rls.hpp` cannot name `ALL_FIELDS` -- this header includes it, not the reverse -- so it
// spells its mask out. Tie the two together here so they cannot drift apart silently.
static_assert(Rls<float, 1, 1>::health_fields == ALL_FIELDS,
              "Rls declares a health_fields mask that is no longer all five Fields");

/// Why the guard blocked an update, when it was not the adapter's own rejection.
enum class GuardReason : uint8_t {
    None = 0,
    Frozen = 1,      ///< adaptation is frozen; this sample was only looked at
    Tripped = 2,     ///< a check failed on this sample and it was rolled back
};

/// `Report` plus what the guard decided. Inheritance rather than composition so that
/// field access stays flat and a caller that does not care about guarding can read it
/// as the plain report it also is — the same reasoning as the Python's subclass.
template <typename T, int M>
struct GuardReport : Report<T, M> {
    bool frozen = false;
    bool rolled_back = false;
    /// Bit i is set if check i failed. Bit 31 is the divergence check. A bitmask
    /// rather than a list of names because names need storage this core does not have.
    uint32_t tripped = 0;
    GuardReason guard_reason = GuardReason::None;
};

static constexpr uint32_t DIVERGENCE_BIT = 1u << 31;

/// A scalar report field must stay within [lo, hi]. Non-finite is unhealthy regardless
/// of the bounds: a NaN compares false against everything, so without saying so it
/// would slip through every comparison as "not out of range".
struct Bound {
    Field field = Field::PTrace;
    double lo = 0.0;
    double hi = 0.0;
    bool has_lo = false;
    bool has_hi = false;

    static Bound at_least(Field f, double lo) {
        Bound b;
        b.field = f;
        b.lo = lo;
        b.has_lo = true;
        return b;
    }
    static Bound at_most(Field f, double hi) {
        Bound b;
        b.field = f;
        b.hi = hi;
        b.has_hi = true;
        return b;
    }
    static Bound between(Field f, double lo, double hi) {
        Bound b;
        b.field = f;
        b.lo = lo;
        b.hi = hi;
        b.has_lo = true;
        b.has_hi = true;
        return b;
    }

    /// A check that sets neither bound can never fail, and a check that cannot fail is
    /// not a check.
    bool valid() const { return (has_lo || has_hi) && !(has_lo && has_hi && lo > hi); }

    /// `gain_norm` is identically zero while frozen, so a positive lower bound on it
    /// can never be satisfied and automatic recovery could never happen. The guard
    /// refuses that combination rather than deadlocking in the field.
    bool usable_when_frozen() const {
        return !(field == Field::GainNorm && has_lo && lo > 0.0);
    }

    template <typename T, int M>
    bool evaluate(const GuardReport<T, M>& r) const {
        double v = 0.0;
        switch (field) {
            case Field::ResidualNorm: v = r.residual_norm; break;
            case Field::ResidualPostNorm: v = r.residual_post_norm; break;
            case Field::GainNorm: v = r.gain_norm; break;
            case Field::PTrace: v = r.p_trace; break;
            case Field::PMinDiag: v = r.p_min_diag; break;
        }
        if (!is_finite(v)) return false;
        if (has_lo && v < lo) return false;
        return !(has_hi && v > hi);
    }
};

/// Fires when the residual grows against its own recent history.
///
/// A fixed bound needs to know the scale of the problem, which for a sensor-agnostic
/// library is exactly what is not known. This compares a fast exponential average of
/// the residual norm against a slow one, so it detects growth and needs no units.
///
/// The slow average is **held while the guard is frozen**. Without that the detector
/// adapts to the very thing it detects: a residual that jumps eightyfold and stays
/// there is growth for a few samples, after which the baseline catches up and a
/// permanently broken sensor reads as normal. That was a real failure in the Python,
/// where it let the guard thaw itself mid-fault. Holding it *per sample* instead —
/// refusing any sample the check disliked — was the first fix and was worse: the
/// baseline then only ever absorbs the smaller residuals and ratchets downward until
/// ordinary noise trips it.
///
/// What this cannot do, said so nobody expects it to: a fault that arrives slowly
/// enough for the baseline to track it will never trip. Absolute limits are `Bound`'s
/// job.
struct Divergence {
    bool enabled = false;
    double ratio = 5.0;
    double fast_alpha = 0.25;
    double slow_alpha = 0.02;
    /// The residual below which the check declines to have an opinion. On near-noiseless
    /// data the slow average approaches zero and every ratio becomes infinite.
    double floor = 0.0;
    uint32_t warmup = 20;

    double fast = 0.0;
    double slow = 0.0;
    uint32_t seen = 0;

    bool valid() const {
        return !enabled ||
               (ratio > 1.0 && fast_alpha > 0.0 && fast_alpha <= 1.0 &&
                slow_alpha > 0.0 && slow_alpha <= 1.0 && fast_alpha > slow_alpha &&
                floor >= 0.0);
    }

    void reset() {
        fast = 0.0;
        slow = 0.0;
        seen = 0;
    }

    template <typename T, int M>
    bool evaluate(const GuardReport<T, M>& r) {
        const double v = r.residual_norm;
        if (!is_finite(v)) return false;
        fast += fast_alpha * (v - fast);
        ++seen;
        if (!r.frozen) slow += slow_alpha * (v - slow);
        if (seen <= warmup) return true;
        const double reference = slow > floor ? slow : floor;
        return reference <= 0.0 || fast <= ratio * reference;
    }
};

/// An adapter that gates its own adaptation.
///
/// `Depth` is the rollback depth and therefore the memory: `Depth * (state_size + 2)`
/// elements, fixed at compile time. Rolling back goes to the *deepest* retained
/// checkpoint, because divergence is detected after it starts and undoing one update
/// lands inside the bad stretch. So `Depth * checkpoint_every` is how far back it
/// reaches.
template <typename Adapter, int Depth = 1, int MaxBounds = 4>
class Guard {
    static_assert(Depth >= 1, "rollback depth must be >= 1");
    static_assert(MaxBounds >= 1, "a guard with no room for checks cannot ever freeze");

public:
    using scalar = typename Adapter::scalar;
    using T = scalar;
    static constexpr int n_features = Adapter::n_features;
    static constexpr int n_outputs = Adapter::n_outputs;
    static constexpr int state_size = Adapter::state_size;
    static constexpr int max_bounds = MaxBounds;
    static constexpr int depth = Depth;

    /// Memory the checkpoints occupy, in bytes. Declared so a caller can budget it.
    static constexpr int checkpoint_bytes =
        Depth * (state_size * static_cast<int>(sizeof(T)) + 2 * 8);

    explicit Guard(const Adapter& adapter, uint32_t checkpoint_every = 1,
                   uint32_t recover_after = 0)
        : adapter_(adapter),
          checkpoint_every_(checkpoint_every ? checkpoint_every : 1),
          recover_after_(recover_after) {
        rearm();
    }

    /// Construct the wrapped adapter **in place** from its own constructor arguments,
    /// instead of copying one the caller already built.
    ///
    /// The copying constructor above needs a whole `Adapter` alive on the stack at the
    /// same time as this one. At six features that is an annoyance; at a hundred and
    /// twenty-eight -- the width a network head has -- it is fatal. Measured before this
    /// existed: `am_pipe_g_f32_128_1_init` wanted **397,192 bytes of stack**, more than
    /// the 332,392-byte object it was building, because an `Rls` temporary and a `Guard`
    /// temporary were both live before the copy. This is the residue of the placement-new
    /// fix, which `argmind.h` recorded as "one adapter-sized temporary ... recorded here
    /// rather than done"; adding a wide shape is what made it urgent.
    ///
    /// `static_cast<A&&>` rather than `std::forward` so that `<utility>` is not needed.
    template <typename... A>
    explicit Guard(in_place_adapter_t, uint32_t checkpoint_every, uint32_t recover_after,
                   A&&... args)
        : adapter_(static_cast<A&&>(args)...),
          checkpoint_every_(checkpoint_every ? checkpoint_every : 1),
          recover_after_(recover_after) {
        rearm();
    }

    // -- configuration -----------------------------------------------------
    //
    // Checks are added after construction rather than passed in, because a variadic
    // constructor would put the check list in the type and the C ABI has to name the
    // type. `valid()` is what refuses a configuration that cannot work.

    /// Refuses a bound on a field this adapter cannot report, which is the difference
    /// between a check and something that looks like one.
    ///
    /// `Field::PTrace` and `Field::PMinDiag` are *covariance* quantities. An adapter
    /// without a covariance used to have to return zeros from `health()`, and then the
    /// same bound broke in both directions: `at_most(PTrace, 1e4)` was **dead** — always
    /// satisfied, unable ever to fire, and configured, so it read as protection — while
    /// `at_least(PTrace, 1e-9)`, a sane sanity bound on RLS, **tripped on the first
    /// update and froze the adapter for the rest of the run**. Both measured on a
    /// second adapter before this existed. Silently dead one way and instantly bricking
    /// the other is worse than either alone, so the configuration is refused at the
    /// point it is made rather than misbehaving later.
    bool add_bound(const Bound& b) {
        if (n_bounds_ >= MaxBounds || !b.valid()) return false;
        if ((Adapter::health_fields & field_bit(b.field)) == 0) return false;
        bounds_[n_bounds_++] = b;
        return true;
    }

    void set_divergence(const Divergence& d) {
        divergence_ = d;
        divergence_.enabled = true;
        divergence_.reset();
    }

    /// False when the configuration cannot work. Checked rather than thrown, since this
    /// core has no exceptions.
    ///
    /// A guard with no checks can never freeze, which makes it a wrapper that looks like
    /// a safety mechanism. And `recover_after` combined with a check that cannot pass
    /// while frozen would never thaw — refused here rather than deadlocking on a device.
    bool valid() const {
        if (n_bounds_ == 0 && !divergence_.enabled) return false;
        if (!divergence_.valid()) return false;
        // The divergence check watches `residual_norm` and nothing else. An adapter that
        // does not declare it would give the check a constant to compare against itself,
        // which is the same dead-check failure `add_bound` refuses -- and worse here,
        // because divergence is the check that needs no units and is therefore the one a
        // sensor-agnostic caller is most likely to enable and least likely to re-derive.
        if (divergence_.enabled &&
            (Adapter::health_fields & field_bit(Field::ResidualNorm)) == 0) {
            return false;
        }
        for (int i = 0; i < n_bounds_; ++i) {
            if (!bounds_[i].valid()) return false;
        }
        if (recover_after_ > 0) {
            for (int i = 0; i < n_bounds_; ++i) {
                if (!bounds_[i].usable_when_frozen()) return false;
            }
        }
        return true;
    }

    // -- Adapter surface, delegated ----------------------------------------

    const T* state() const { return adapter_.state(); }
    const Adapter& adapter() const { return adapter_; }
    Adapter& adapter_mutable() { return adapter_; }

    void predict(const T* __restrict x, T* __restrict out) const {
        adapter_.predict(x, out);
    }

    /// The guard *is* an adapter, so it reports what the one it wraps reports.
    static constexpr uint32_t health_fields = Adapter::health_fields;

    /// Straight through. Correct while frozen too: the state is not moving, so the
    /// wrapped adapter's own answer is still the current one.
    ///
    /// Behind `if constexpr`, so an adapter declaring neither covariance field need not
    /// implement `health()` at all — the branch is never instantiated. Zeros are safe in
    /// the other branch precisely because `add_bound` refuses to watch a field that is
    /// not declared, so nothing can read them.
    void health(double* __restrict p_trace, double* __restrict p_min_diag) const {
        if constexpr (needs_health(Adapter::health_fields)) {
            adapter_.health(p_trace, p_min_diag);
        } else {
            *p_trace = 0.0;
            *p_min_diag = 0.0;
        }
    }

    bool load_state(const T* __restrict flat) {
        if (!adapter_.load_state(flat)) return false;
        rearm();
        return true;
    }

    void reset() {
        adapter_.reset();
        rearm();
    }

    // -- freezing ----------------------------------------------------------

    bool frozen() const { return frozen_; }
    uint32_t frozen_by() const { return frozen_by_; }

    /// Stop adaptation without rolling back, for a supervisor that knows something the
    /// checks cannot see.
    void freeze() {
        frozen_ = true;
        healthy_run_ = 0;
        // The live state, not the newest checkpoint: a manual freeze does not move the
        // state, so the live values are exactly right, while the newest checkpoint can be
        // up to checkpoint_every updates stale. Reading the checkpoint here was a bug in
        // the Python reference, found by a mutation check on this port and fixed in both.
        health(&frozen_trace_, &frozen_min_diag_);
    }

    void thaw() {
        if (frozen_) ++thaws_;
        frozen_ = false;
        healthy_run_ = 0;
        divergence_.reset();
    }

    uint32_t applied() const { return applied_; }
    uint32_t trips() const { return trips_; }
    uint32_t rollbacks() const { return rollbacks_; }
    uint32_t blocked_frozen() const { return blocked_frozen_; }
    uint32_t rejected_by_adapter() const { return rejected_by_adapter_; }
    uint32_t thaws() const { return thaws_; }

    // -- the work ----------------------------------------------------------

    GuardReport<T, n_outputs> update(const T* __restrict x, const T* __restrict y) {
        if (frozen_) {
            ++blocked_frozen_;
            GuardReport<T, n_outputs> probe = make_probe(x, y);
            if (failing(probe) == 0) {
                ++healthy_run_;
            } else {
                healthy_run_ = 0;
            }
            if (recover_after_ > 0 && healthy_run_ >= recover_after_) thaw();
            return probe;
        }

        const Report<T, n_outputs> inner = adapter_.update(x, y);
        GuardReport<T, n_outputs> r = wrap(inner);

        if (!inner.applied) {
            // The adapter refused the data itself. Not a guard trip: no checkpoint, no
            // rollback, and counted separately so the two never blur together.
            ++rejected_by_adapter_;
            return r;
        }

        const uint32_t tripped = failing(r);
        if (tripped != 0) {
            roll_back();
            frozen_ = true;
            frozen_by_ = tripped;
            healthy_run_ = 0;
            // Stateful checks are deliberately NOT reset here: their history is the only
            // reference the frozen probes have for deciding whether the world came back.
            ++trips_;
            ++rollbacks_;
            r.applied = false;
            r.frozen = true;
            r.rolled_back = true;
            r.tripped = tripped;
            r.guard_reason = GuardReason::Tripped;
            r.p_trace = frozen_trace_;
            r.p_min_diag = frozen_min_diag_;
            return r;
        }

        ++applied_;
        if (++since_ckpt_ >= checkpoint_every_) {
            store_checkpoint(r.p_trace, r.p_min_diag);
            since_ckpt_ = 0;
        }
        return r;
    }

private:
    GuardReport<T, n_outputs> wrap(const Report<T, n_outputs>& src) const {
        GuardReport<T, n_outputs> r;
        r.applied = src.applied;
        r.reason = src.reason;
        for (int j = 0; j < n_outputs; ++j) {
            r.residual[j] = src.residual[j];
            r.residual_post[j] = src.residual_post[j];
        }
        r.residual_norm = src.residual_norm;
        r.residual_post_norm = src.residual_post_norm;
        r.gain_norm = src.gain_norm;
        r.p_trace = src.p_trace;
        r.p_min_diag = src.p_min_diag;
        return r;
    }

    /// What the sample looks like without adapting to it. The residual is real, the gain
    /// is zero because nothing was applied, and the covariance scalars are those of the
    /// checkpoint the guard is sitting on — exact rather than approximate, since the
    /// state has not moved since it was restored.
    GuardReport<T, n_outputs> make_probe(const T* __restrict x,
                                         const T* __restrict y) const {
        GuardReport<T, n_outputs> r;
        r.applied = false;
        r.reason = Reason::None;
        r.frozen = true;
        r.guard_reason = GuardReason::Frozen;
        T yhat[n_outputs];
        adapter_.predict(x, yhat);
        for (int j = 0; j < n_outputs; ++j) {
            r.residual[j] = y[j] - yhat[j];
            r.residual_post[j] = r.residual[j];
        }
        r.residual_norm = static_cast<double>(norm<T, n_outputs>(r.residual));
        r.residual_post_norm = r.residual_norm;
        r.gain_norm = 0.0;
        r.p_trace = frozen_trace_;
        r.p_min_diag = frozen_min_diag_;
        return r;
    }

    /// Every failing check, not just the first: three faults at once is diagnostic
    /// information and collapsing it to one throws that away.
    uint32_t failing(GuardReport<T, n_outputs>& r) {
        uint32_t mask = 0;
        for (int i = 0; i < n_bounds_; ++i) {
            if (!bounds_[i].evaluate(r)) mask |= (1u << i);
        }
        if (divergence_.enabled && !divergence_.evaluate(r)) mask |= DIVERGENCE_BIT;
        return mask;
    }

    int newest() const { return (ckpt_head_ + Depth - 1) % Depth; }
    int oldest() const { return (ckpt_head_ + Depth - ckpt_count_) % Depth; }

    void store_checkpoint(double trace, double min_diag) {
        copy<T, state_size>(adapter_.state(), ckpt_[ckpt_head_]);
        ckpt_health_[ckpt_head_][0] = trace;
        ckpt_health_[ckpt_head_][1] = min_diag;
        ckpt_head_ = (ckpt_head_ + 1) % Depth;
        if (ckpt_count_ < Depth) ++ckpt_count_;
    }

    void roll_back() {
        const int i = oldest();
        adapter_.load_state(ckpt_[i]);
        frozen_trace_ = ckpt_health_[i][0];
        frozen_min_diag_ = ckpt_health_[i][1];
        // Everything retained is at or after the state just restored, so keeping it
        // would let a later trip roll "back" to a moment that came afterwards.
        ckpt_count_ = 1;
        ckpt_head_ = (i + 1) % Depth;
        since_ckpt_ = 0;
    }

    /// Drop guard history and re-arm against whatever state the adapter is in now.
    /// Called when the state is replaced underneath the guard, because every checkpoint
    /// and every EWMA describes a state that no longer exists.
    void rearm() {
        divergence_.reset();
        ckpt_head_ = 0;
        ckpt_count_ = 0;
        since_ckpt_ = 0;
        frozen_ = false;
        frozen_by_ = 0;
        healthy_run_ = 0;
        health(&frozen_trace_, &frozen_min_diag_);
        store_checkpoint(frozen_trace_, frozen_min_diag_);
    }

    Adapter adapter_;
    Bound bounds_[MaxBounds];
    int n_bounds_ = 0;
    Divergence divergence_;

    // Preallocated ring. Each slot carries the state block and the covariance scalars
    // that were true at that state, so a probe built after a rollback reports the truth
    // rather than a stale number.
    T ckpt_[Depth][state_size];
    double ckpt_health_[Depth][2];
    int ckpt_head_ = 0;
    int ckpt_count_ = 0;
    uint32_t since_ckpt_ = 0;

    uint32_t checkpoint_every_ = 1;
    uint32_t recover_after_ = 0;
    bool frozen_ = false;
    uint32_t frozen_by_ = 0;
    uint32_t healthy_run_ = 0;
    double frozen_trace_ = 0.0;
    double frozen_min_diag_ = 0.0;

    uint32_t applied_ = 0;
    uint32_t trips_ = 0;
    uint32_t rollbacks_ = 0;
    uint32_t blocked_frozen_ = 0;
    uint32_t rejected_by_adapter_ = 0;
    uint32_t thaws_ = 0;
};

}  // namespace argmind

#endif  // ARGMIND_GUARD_HPP
