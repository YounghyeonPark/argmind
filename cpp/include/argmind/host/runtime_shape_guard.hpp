// SPDX-License-Identifier: Apache-2.0
//
// The guard, with its shape decided at construction. Hosted tier.
//
// Why this had to exist before the layer was usable
// -------------------------------------------------
// `Guard<Adapter, Depth, MaxBounds, NExternal>` holds `ckpt_[Depth][state_size]`, so it
// needs the state size as a compile-time constant and a runtime-shaped adapter cannot
// give it one. Until this header, the hosted tier could adapt with **no guard at all** --
// which is the exact failure this library is built around, and `shapecheck` refused
// guarded cases rather than let it look supported.
//
// What is matched exactly, because a guard is decisions and a decision that differs is
// not a rounding difference:
//
//   - checks run *after* the update, which is why there is rollback rather than prevention
//   - every check runs; the guard does not stop at the first failure
//   - a trip rolls back to the *deepest* retained checkpoint and freezes
//   - stateful checks keep their history through a freeze and are reset on thaw
//   - an adapter rejection passes through and is counted separately: a NaN sample and a
//     diverging learner are different faults
//   - while frozen the guard evaluates a probe whose residual is real, whose gain is
//     genuinely zero, and whose covariance scalars are those of the restored checkpoint
//
// What is NOT the same, and is the point: `depth`, `max_bounds` and the number of
// caller-supplied scalars are constructor arguments rather than template parameters, and
// the checkpoint ring is on the heap. The declared-memory property the core has is given
// up here deliberately -- above an OS there is nothing to declare it to.

#ifndef ARGMIND_HOST_RUNTIME_SHAPE_GUARD_HPP
#define ARGMIND_HOST_RUNTIME_SHAPE_GUARD_HPP

#include <stdint.h>

#include <vector>

#include "argmind/guard.hpp"                // Bound, Divergence, Field, GuardReason
#include "argmind/host/runtime_shape.hpp"

namespace argmind {
namespace host {

/// A `RuntimeShapedReport` plus what the guard decided.
template <typename T>
struct RuntimeShapedGuardReport : RuntimeShapedReport<T> {
    bool frozen = false;
    bool rolled_back = false;
    uint32_t tripped = 0;
    GuardReason guard_reason = GuardReason::None;
};

/// An adapter that gates its own adaptation, with the shape decided at construction.
///
/// `Adapter` is `RuntimeShapedRls<T>` or `RuntimeShapedNlms<T>`, or anything with the
/// same surface. Templated rather than type-erased: the core makes the same choice, and
/// a virtual interface here would buy a uniform container nobody has asked for.
template <typename Adapter>
class RuntimeShapedGuard {
public:
    using T = typename Adapter::scalar;
    /// Re-exported, because the guard IS an adapter and anything wrapping it -- the
    /// runtime-shaped pipeline, for one -- asks for `scalar` by name. Omitted at
    /// first, and the pipeline would not compile over a guarded adapter.
    using scalar = T;

    /// `depth` is the rollback reach in checkpoints, so `depth * checkpoint_every` is how
    /// far back a trip can undo. `n_external` reserves caller-supplied slots -- see
    /// `set_external`, and `guard.hpp` for why the guard should see them at all.
    RuntimeShapedGuard(Adapter adapter, int depth = 1, uint32_t checkpoint_every = 1,
                       uint32_t recover_after = 0, int max_bounds = 4,
                       int n_external = 0)
        : adapter_(adapter),
          depth_(depth < 1 ? 1 : depth),
          checkpoint_every_(checkpoint_every ? checkpoint_every : 1),
          recover_after_(recover_after),
          max_bounds_(max_bounds < 1 ? 1 : max_bounds),
          n_external_(n_external < 0 ? 0
                      : (n_external > kMaxExternal ? kMaxExternal : n_external)),
          external_(static_cast<size_t>(n_external_), 0.0),
          ckpt_(static_cast<size_t>(depth_ < 1 ? 1 : depth_) *
                    static_cast<size_t>(adapter.state_size()), T(0)),
          ckpt_health_(static_cast<size_t>(depth_ < 1 ? 1 : depth_) * 2, 0.0) {
        rearm();
    }

    // -- configuration -----------------------------------------------------

    bool add_bound(const Bound& b) {
        if (static_cast<int>(bounds_.size()) >= max_bounds_ || !b.valid()) return false;
        if (is_external(b.field)) {
            if (external_index(b.field) >= n_external_) return false;
        } else if ((Adapter::health_fields & field_bit(b.field)) == 0) {
            return false;
        }
        bounds_.push_back(b);
        return true;
    }

    void set_divergence(const Divergence& d) {
        divergence_ = d;
        divergence_.enabled = true;
        divergence_.reset();
    }

    bool set_external(int i, double v) {
        if (i < 0 || i >= n_external_) return false;
        external_[static_cast<size_t>(i)] = v;
        return true;
    }

    double external(int i) const {
        return (i >= 0 && i < n_external_) ? external_[static_cast<size_t>(i)] : 0.0;
    }
    int n_external() const { return n_external_; }

    /// False when the configuration cannot work. Checked rather than thrown, matching the
    /// core -- a guard with no checks can never freeze, and `recover_after` with a check
    /// that cannot pass while frozen would never thaw.
    bool valid() const {
        if (bounds_.empty() && !divergence_.enabled) return false;
        if (!divergence_.valid()) return false;
        if (divergence_.enabled &&
            (Adapter::health_fields & field_bit(Field::ResidualNorm)) == 0) {
            return false;
        }
        for (const Bound& b : bounds_) {
            if (!b.valid()) return false;
        }
        if (recover_after_ > 0) {
            for (const Bound& b : bounds_) {
                if (!b.usable_when_frozen()) return false;
            }
        }
        return adapter_.valid();
    }

    // -- adapter surface, delegated ----------------------------------------

    static constexpr uint32_t health_fields = Adapter::health_fields;

    int n_features() const { return adapter_.n_features(); }
    int n_outputs() const { return adapter_.n_outputs(); }
    int state_size() const { return adapter_.state_size(); }
    const T* state() const { return adapter_.state(); }
    const Adapter& adapter() const { return adapter_; }
    Adapter& adapter_mutable() { return adapter_; }

    void predict(const T* x, T* out) const { adapter_.predict(x, out); }

    /// `if constexpr`, not `if` -- an adapter declaring neither covariance field need not
    /// implement `health()` at all, and a plain `if` instantiates the branch anyway. The
    /// core makes the same choice for the same reason; this port used a plain `if` first
    /// and `RuntimeShapedNlms` would not compile through the guard.
    void health(double* p_trace, double* p_min_diag) const {
        if constexpr (needs_health(Adapter::health_fields)) {
            adapter_.health(p_trace, p_min_diag);
        } else {
            *p_trace = 0.0;
            *p_min_diag = 0.0;
        }
    }

    bool load_state(const T* flat) {
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
    /// checks cannot see. Reads the LIVE health scalars, not the newest checkpoint: a
    /// manual freeze moves nothing, so the checkpoint can be `checkpoint_every` updates
    /// stale. Reading it here was a bug in the reference, found by a mutation check on
    /// the original port and fixed in both.
    void freeze() {
        frozen_ = true;
        healthy_run_ = 0;
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

    RuntimeShapedGuardReport<T> update(const T* x, const T* y) {
        if (frozen_) {
            ++blocked_frozen_;
            RuntimeShapedGuardReport<T> probe = make_probe(x, y);
            if (failing(probe) == 0) {
                ++healthy_run_;
            } else {
                healthy_run_ = 0;
            }
            if (recover_after_ > 0 && healthy_run_ >= recover_after_) thaw();
            return probe;
        }

        const RuntimeShapedReport<T> inner = adapter_.update(x, y);
        RuntimeShapedGuardReport<T> r = wrap(inner);

        if (!inner.applied) {
            // The adapter refused the data itself. Not a guard trip: no checkpoint, no
            // rollback, counted separately so the two never blur together.
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
    RuntimeShapedGuardReport<T> wrap(const RuntimeShapedReport<T>& src) const {
        RuntimeShapedGuardReport<T> r;
        r.applied = src.applied;
        r.reason = src.reason;
        r.residual = src.residual;
        r.residual_post = src.residual_post;
        r.residual_norm = src.residual_norm;
        r.residual_post_norm = src.residual_post_norm;
        r.gain_norm = src.gain_norm;
        r.p_trace = src.p_trace;
        r.p_min_diag = src.p_min_diag;
        return r;
    }

    RuntimeShapedGuardReport<T> make_probe(const T* x, const T* y) const {
        RuntimeShapedGuardReport<T> r;
        r.applied = false;
        r.reason = Reason::None;
        r.frozen = true;
        r.guard_reason = GuardReason::Frozen;
        const int m = adapter_.n_outputs();
        std::vector<T> yhat(static_cast<size_t>(m));
        adapter_.predict(x, yhat.data());
        r.residual.resize(static_cast<size_t>(m));
        for (int j = 0; j < m; ++j) r.residual[static_cast<size_t>(j)] = y[j] - yhat[j];
        r.residual_post = r.residual;
        r.residual_norm = detail::norm_of(r.residual);
        r.residual_post_norm = r.residual_norm;
        r.gain_norm = 0.0;
        r.p_trace = frozen_trace_;
        r.p_min_diag = frozen_min_diag_;
        return r;
    }

    /// Every failing check, not just the first: three faults at once is diagnostic
    /// information and collapsing it to one throws that away.
    uint32_t failing(RuntimeShapedGuardReport<T>& r) {
        // `Bound::evaluate` reads a `GuardReport<T, M>`, whose residual arrays are fixed
        // at compile time. Only the scalars are ever read, so the report is projected on
        // to a one-output shell rather than making `Bound` a template over two report
        // types -- which would be the same rule written twice.
        GuardReport<T, 1> shell;
        shell.residual_norm = r.residual_norm;
        shell.residual_post_norm = r.residual_post_norm;
        shell.gain_norm = r.gain_norm;
        shell.p_trace = r.p_trace;
        shell.p_min_diag = r.p_min_diag;
        // **`frozen` is not decoration here and leaving it out was a real bug.**
        // `Divergence::evaluate` holds its slow average while the guard is frozen --
        // `if (!r.frozen) slow += ...` -- so a shell that always reads false lets the
        // baseline track the very fault it is supposed to be holding out against. Found
        // by parity, not by reading: the reference thawed at step 218 and this thawed at
        // 166, **in the middle of a fault that ran to 200**. That is the same design
        // error the reference itself made once and CLAUDE.md records; the port
        // reintroduced it through an uninitialised field.
        shell.frozen = r.frozen;
        shell.rolled_back = r.rolled_back;
        shell.applied = r.applied;
        shell.reason = r.reason;
        shell.guard_reason = r.guard_reason;

        uint32_t mask = 0;
        for (size_t i = 0; i < bounds_.size(); ++i) {
            if (!bounds_[i].evaluate(shell, external_.data())) {
                mask |= (1u << i);
            }
        }
        if (divergence_.enabled && !divergence_.evaluate(shell)) mask |= DIVERGENCE_BIT;
        return mask;
    }

    int newest() const { return (ckpt_head_ + depth_ - 1) % depth_; }
    int oldest() const { return (ckpt_head_ + depth_ - ckpt_count_) % depth_; }

    T* slot(int i) {
        return ckpt_.data() + static_cast<size_t>(i) * adapter_.state_size();
    }

    void store_checkpoint(double trace, double min_diag) {
        const int n = adapter_.state_size();
        const T* src = adapter_.state();
        T* dst = slot(ckpt_head_);
        for (int i = 0; i < n; ++i) dst[i] = src[i];
        ckpt_health_[static_cast<size_t>(ckpt_head_) * 2] = trace;
        ckpt_health_[static_cast<size_t>(ckpt_head_) * 2 + 1] = min_diag;
        ckpt_head_ = (ckpt_head_ + 1) % depth_;
        if (ckpt_count_ < depth_) ++ckpt_count_;
    }

    void roll_back() {
        const int i = oldest();
        adapter_.load_state(slot(i));
        frozen_trace_ = ckpt_health_[static_cast<size_t>(i) * 2];
        frozen_min_diag_ = ckpt_health_[static_cast<size_t>(i) * 2 + 1];
        // Everything retained is at or after the state just restored, so keeping it would
        // let a later trip roll "back" to a moment that came afterwards.
        ckpt_count_ = 1;
        ckpt_head_ = (i + 1) % depth_;
        since_ckpt_ = 0;
    }

    /// Drop guard history and re-arm against whatever state the adapter is in now.
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
    int depth_;
    uint32_t checkpoint_every_;
    uint32_t recover_after_;
    int max_bounds_;
    int n_external_;

    std::vector<Bound> bounds_;
    Divergence divergence_;
    std::vector<double> external_;

    std::vector<T> ckpt_;
    std::vector<double> ckpt_health_;
    int ckpt_head_ = 0;
    int ckpt_count_ = 0;
    uint32_t since_ckpt_ = 0;

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

}  // namespace host
}  // namespace argmind

#endif  // ARGMIND_HOST_RUNTIME_SHAPE_GUARD_HPP
