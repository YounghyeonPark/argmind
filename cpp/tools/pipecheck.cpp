// SPDX-License-Identifier: Apache-2.0
//
// The pipeline parity tool.
//
// What has to agree here is not arithmetic — `xcheck` already settled that RLS and the guard
// compute the same numbers — but **bookkeeping under interleaving**. A pipeline is a state
// machine driven by two independent clocks: samples arriving and `step()` being called. Every
// interesting quantity it reports is a difference between counters taken at different moments,
// so the way to make it fail is to interleave pushes and steps adversarially, not to push a
// thousand samples and step once.
//
// The job file is therefore an **event log**, not a sample list. `push` and `step` are
// separate events and the Python emits them in whatever order the case wants, so the same
// tool covers "step every sample", "step once at the end", "let the ring overflow between
// steps", and "step when nothing arrived" without a mode flag for each.
//
// What is compared, per step: the whole `StepReport` — frames, updated, lost_frames, all
// seven skip counters, frozen, the tripped bitmask, per-stream staleness, worst staleness,
// and the reference time. Then at the end: the aligner's three counters, the pipeline's four
// totals, and the adapter's state block. The state block matters because everything above
// could match while the two fed *different data* to the learner.
//
// This tool drives the shipped `Stream<T, Dim, Capacity>` rather than a hand-built ring, which
// `scheck` could not do — its aligner has to see several widths at once. Here the widths are
// fixed per job, so the real class is instantiated for each and the harness has no copy of
// `push` to drift from. That matters more than usual: `lost_frames` is derived from
// `dropped + count`, so a harness with its own overwrite accounting would be marking its own
// homework.
//
// Little-endian, naturally aligned, every struct a multiple of eight bytes — same rules as the
// other three tools and for the same reason.

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "argmind/align.hpp"
#include "argmind/guard.hpp"
#include "argmind/pipeline.hpp"
#include "argmind/rls.hpp"
#include "argmind/stream.hpp"

namespace {

constexpr uint16_t kVersion = 1;
constexpr int kMaxStreams = 3;
constexpr int kMaxDim = 3;
constexpr int kMaxFeatures = 8;

// The instantiation list, which is the same trade the C ABI makes and made again for the
// same reason: the pipeline's dimensions are compile-time because the adapter's are, and a
// runtime-dimensioned pipeline would discard the reason the core is C++ at all. Adding an
// axis value is one line plus a rebuild. The values here are chosen to move each axis
// independently rather than to cover a product: two precisions, a small and a large
// regressor, a ring that overflows quickly and one that does not, and a rollback depth of 1
// and 4.
//
//   dtype       f4, f8
//   n_features  2, 3, 6
//   n_outputs   1, and 3 at n_features 6
//   capacity    4, 8, 64
//   guard       absent, depth 1, depth 4
//
// Three outputs exists at six features alone, and only because the freshness gate decides per
// output: the reference compares a tuple over unique target streams while the core iterates
// outputs, and until this axis existed that equivalence had been read rather than run.

struct BoundSlot {
    double lo;
    double hi;
    uint32_t field;
    uint32_t flags;  // bit 0 has_lo, bit 1 has_hi
};
static_assert(sizeof(BoundSlot) == 24, "job format: BoundSlot must be 24 bytes");

struct DivergenceCfg {
    double ratio;
    double fast_alpha;
    double slow_alpha;
    double floor;
    uint32_t warmup;
    uint32_t enabled;
};
static_assert(sizeof(DivergenceCfg) == 40, "job format: DivergenceCfg must be 40 bytes");

struct Job {
    char magic[4];  // "AMPL"
    uint16_t version;
    uint8_t dtype;    // 4 or 8
    uint8_t guarded;  // 0 = bare adapter
        uint16_t n_features;
    uint16_t n_outputs;
    uint16_t n_streams;
    uint16_t capacity;
    uint16_t policy;
    uint16_t reference;
    uint16_t n_map_features;  // features declared by the map; + constant must equal n_features
    uint8_t constant;
    uint8_t require_valid;
    uint8_t gate_stale;
    uint8_t allow_non_causal;
        uint8_t require_fresh_target;
    uint8_t pad0;
    uint32_t n_events;
    uint32_t depth;
    uint32_t checkpoint_every;
    uint32_t recover_after;
    uint32_t n_bounds;
    double max_stale_pipeline;
    double max_stale_aligner;
    double lambda;
    double p0;
    BoundSlot bounds[4];
    DivergenceCfg divergence;
};
// 216, and the number is here rather than in a comment because the Python writes this header
// with `struct`, which packs `<` without padding while a C++ struct pads. Every size in this
// file is asserted on both sides; the pad before `max_stale_pipeline` is the one that bites.
// Still 216: the new flag and its padding land in the four bytes that were already
// implicit padding before `n_events`. Measured by the assertion, not assumed.
static_assert(sizeof(Job) == 216, "job format: header must be 216 bytes");

struct StreamSpec {
    uint16_t dim;
    uint16_t pad[3];
};
static_assert(sizeof(StreamSpec) == 8, "job format: stream spec must be 8 bytes");

/// One entry of the resolved frame map: which stream, which component of it.
struct MapEntry {
    uint16_t stream;
    uint16_t component;
    uint32_t pad;
};
static_assert(sizeof(MapEntry) == 8, "job format: map entry must be 8 bytes");

/// `kind` 0 pushes `dim` values into a stream, 1 calls `step()`. Push payload follows.
struct Event {
    uint8_t kind;
    uint8_t stream;
    uint8_t valid;
    uint8_t pad[5];
    double t;
};
static_assert(sizeof(Event) == 16, "job format: event must be 16 bytes");

/// One `StepReport`, flattened. Staleness follows, `n_streams` doubles of it — always the
/// full width, so the Python reads a fixed record and `n_staleness` says how much of it the
/// pipeline actually filled in.
struct StepOut {
    uint8_t ok;
    uint8_t frozen;
    uint8_t has_t;
    uint8_t has_prediction;
    uint32_t frames;
    uint32_t updated;
    uint32_t lost_frames;
    uint32_t frozen_by;
    uint32_t skipped[argmind::SKIP_REASON_COUNT];
    int32_t n_staleness;
    uint32_t pad2[2];   // explicit: the eighth skip counter pushed the doubles out
    double worst_staleness;
    double t;
    /// Widened to double so one record shape covers f32 and f64 jobs, exactly as
    /// `am_step_report` does. Only `n_outputs` of these are meaningful.
    double prediction[3];
};
static_assert(sizeof(StepOut) == 104, "result format: step record must be 104 bytes");

/// The tail: what the run added up to. Written after every step record.
struct Tail {
    uint64_t steps;
    uint64_t total_frames;
    uint64_t total_updated;
    uint64_t total_lost;
    uint64_t aligner_frames;
    uint64_t aligner_dropped_stale;
    uint64_t aligner_no_data;
    uint32_t state_size;
    uint32_t frozen;
};
static_assert(sizeof(Tail) == 64, "result format: tail must be 64 bytes");

argmind::Bound bound_from(const BoundSlot& s) {
    argmind::Bound b;
    b.field = static_cast<argmind::Field>(s.field);
    b.lo = s.lo;
    b.hi = s.hi;
    b.has_lo = (s.flags & 1u) != 0;
    b.has_hi = (s.flags & 2u) != 0;
    return b;
}

/// One ring per (stream, width). Wasteful — every stream carries storage for every width this
/// tool supports — and host-only, so it costs nothing that matters. The alternative is a
/// dispatch table over `Stream<T, Dim, Capacity>` instantiations, which is the same thing with
/// more machinery.
template <typename T, int Cap>
struct Rings {
    argmind::Stream<T, 1, Cap> d1[kMaxStreams];
    argmind::Stream<T, 2, Cap> d2[kMaxStreams];
    argmind::Stream<T, 3, Cap> d3[kMaxStreams];

    bool push(int i, int dim, double t, const T* v, bool valid) {
        if (dim == 1) return d1[i].push(t, v, valid);
        if (dim == 2) return d2[i].push(t, v, valid);
        return d3[i].push(t, v, valid);
    }

    argmind::StreamView<T> view(int i, int dim) const {
        if (dim == 1) return d1[i].view();
        if (dim == 2) return d2[i].view();
        return d3[i].view();
    }
};

/// Replay the event log against one configured pipeline.
template <typename Pipe, typename T, int Cap>
int drive(FILE* in, FILE* out, const Job& job, const StreamSpec* specs, Pipe& pipe) {
    static Rings<T, Cap> rings;
    rings = Rings<T, Cap>();

    argmind::StreamView<T> views[kMaxStreams];
    for (uint16_t i = 0; i < job.n_streams; ++i) views[i] = rings.view(i, specs[i].dim);
    if (!pipe.configure(views, job.n_streams)) {
        fprintf(stderr, "pipecheck: the pipeline refused this configuration\n");
        return 6;
    }

    for (uint32_t e = 0; e < job.n_events; ++e) {
        Event ev;
        if (fread(&ev, sizeof(ev), 1, in) != 1) return 4;
        if (ev.kind == 0) {
            if (ev.stream >= job.n_streams) return 5;
            const int dim = specs[ev.stream].dim;
            T v[kMaxDim];
            if (fread(v, sizeof(T), dim, in) != static_cast<size_t>(dim)) return 4;
            const uint8_t accepted =
                rings.push(ev.stream, dim, ev.t, v, ev.valid != 0) ? 1u : 0u;
            // A refusal is data: the reference raises on a non-monotonic timestamp and this
            // returns false, and if the two refuse different samples then every later
            // comparison is against a different history.
            if (fwrite(&accepted, 1, 1, out) != 1) return 3;
            continue;
        }

        // Views are re-taken here, never cached: a `StreamView` snapshots head and count, so
        // one held across a push describes a ring that has moved on.
        for (uint16_t i = 0; i < job.n_streams; ++i) views[i] = rings.view(i, specs[i].dim);
        const auto rep = pipe.step(views);

        StepOut o;
        memset(&o, 0, sizeof(o));
        o.ok = rep.ok ? 1u : 0u;
        o.frozen = rep.frozen ? 1u : 0u;
        o.has_t = rep.has_t ? 1u : 0u;
        o.frames = rep.frames;
        o.updated = rep.updated;
        o.lost_frames = rep.lost_frames;
        o.frozen_by = rep.frozen_by;
        for (int k = 0; k < argmind::SKIP_REASON_COUNT; ++k) o.skipped[k] = rep.skipped[k];
        o.n_staleness = rep.n_staleness;
        o.has_prediction = rep.has_prediction ? 1u : 0u;
        o.worst_staleness = rep.worst_staleness;
        o.t = rep.t;
        for (int j = 0; j < 3; ++j) {
            o.prediction[j] = j < Pipe::n_outputs
                                  ? static_cast<double>(rep.prediction[j])
                                  : 0.0;
        }
        if (fwrite(&o, sizeof(o), 1, out) != 1) return 3;
        double stale[kMaxStreams];
        for (int i = 0; i < kMaxStreams; ++i) stale[i] = rep.staleness[i];
        if (fwrite(stale, sizeof(double), job.n_streams, out) !=
            static_cast<size_t>(job.n_streams)) {
            return 3;
        }
    }

    Tail tail;
    memset(&tail, 0, sizeof(tail));
    tail.steps = pipe.steps();
    tail.total_frames = pipe.total_frames();
    tail.total_updated = pipe.total_updated();
    tail.total_lost = pipe.total_lost_frames();
    tail.aligner_frames = pipe.aligner().frames();
    tail.aligner_dropped_stale = pipe.aligner().dropped_stale();
    tail.aligner_no_data = pipe.aligner().no_data();
    tail.state_size = Pipe::n_features * Pipe::n_outputs +
                      Pipe::n_features * Pipe::n_features;
    tail.frozen = pipe.frozen() ? 1u : 0u;
    if (fwrite(&tail, sizeof(tail), 1, out) != 1) return 3;
    if (fwrite(pipe.adapter().state(), sizeof(T), tail.state_size, out) !=
        static_cast<size_t>(tail.state_size)) {
        return 3;
    }
    return 0;
}

template <typename T, int N, int M>
argmind::FrameMap<N, M> build_map(const Job& job, const MapEntry* feats,
                                  const MapEntry* targs, bool* ok) {
    argmind::FrameMap<N, M> map;
    map.set_constant(job.constant != 0);
    *ok = true;
    for (uint16_t j = 0; j < job.n_map_features; ++j) {
        if (!map.add_feature(feats[j].stream, feats[j].component)) *ok = false;
    }
    for (int j = 0; j < M; ++j) {
        if (!map.add_target(targs[j].stream, targs[j].component)) *ok = false;
    }
    if (!map.valid()) *ok = false;
    return map;
}

template <typename T, int N, int M, int Cap, int Depth>
int run_guarded(FILE* in, FILE* out, const Job& job, const StreamSpec* specs,
                const MapEntry* feats, const MapEntry* targs) {
    bool ok = false;
    const auto map = build_map<T, N, M>(job, feats, targs, &ok);
    if (!ok) {
        fprintf(stderr, "pipecheck: the frame map does not fill the adapter\n");
        return 6;
    }
    argmind::Rls<T, N, M> rls(static_cast<T>(job.lambda), static_cast<T>(job.p0));
    if (!rls.valid()) return 6;
    argmind::Guard<argmind::Rls<T, N, M>, Depth, 4> guard(rls, job.checkpoint_every,
                                                          job.recover_after);
    for (uint32_t i = 0; i < job.n_bounds && i < 4; ++i) {
        if (!guard.add_bound(bound_from(job.bounds[i]))) return 6;
    }
    if (job.divergence.enabled) {
        argmind::Divergence d;
        d.ratio = job.divergence.ratio;
        d.fast_alpha = job.divergence.fast_alpha;
        d.slow_alpha = job.divergence.slow_alpha;
        d.floor = job.divergence.floor;
        d.warmup = job.divergence.warmup;
        guard.set_divergence(d);
    }
    if (!guard.valid()) {
        fprintf(stderr, "pipecheck: the guard refused this configuration\n");
        return 6;
    }
    const argmind::Aligner aligner(static_cast<argmind::Policy>(job.policy),
                                   job.max_stale_aligner);
    argmind::Pipeline<argmind::Guard<argmind::Rls<T, N, M>, Depth, 4>, kMaxStreams> pipe(
        guard, aligner, map, job.reference);
    if (job.gate_stale) pipe.set_max_stale_s(job.max_stale_pipeline);
    pipe.set_require_valid(job.require_valid != 0);
    pipe.set_allow_non_causal(job.allow_non_causal != 0);
    pipe.set_require_fresh_target(job.require_fresh_target != 0);
    return drive<decltype(pipe), T, Cap>(in, out, job, specs, pipe);
}

template <typename T, int N, int M, int Cap>
int run_shape(FILE* in, FILE* out, const Job& job, const StreamSpec* specs,
              const MapEntry* feats, const MapEntry* targs) {
    if (job.guarded) {
        if (job.depth == 1) return run_guarded<T, N, M, Cap, 1>(in, out, job, specs, feats, targs);
        if (job.depth == 4) return run_guarded<T, N, M, Cap, 4>(in, out, job, specs, feats, targs);
        fprintf(stderr, "pipecheck: depth %u is not in the instantiation list\n", job.depth);
        return 2;
    }
    bool ok = false;
    const auto map = build_map<T, N, M>(job, feats, targs, &ok);
    if (!ok) {
        fprintf(stderr, "pipecheck: the frame map does not fill the adapter\n");
        return 6;
    }
    argmind::Rls<T, N, M> rls(static_cast<T>(job.lambda), static_cast<T>(job.p0));
    if (!rls.valid()) return 6;
    const argmind::Aligner aligner(static_cast<argmind::Policy>(job.policy),
                                   job.max_stale_aligner);
    argmind::Pipeline<argmind::Rls<T, N, M>, kMaxStreams> pipe(rls, aligner, map,
                                                               job.reference);
    if (job.gate_stale) pipe.set_max_stale_s(job.max_stale_pipeline);
    pipe.set_require_valid(job.require_valid != 0);
    pipe.set_allow_non_causal(job.allow_non_causal != 0);
    pipe.set_require_fresh_target(job.require_fresh_target != 0);
    return drive<decltype(pipe), T, Cap>(in, out, job, specs, pipe);
}

template <typename T, int N, int M>
int run_capacity(FILE* in, FILE* out, const Job& job, const StreamSpec* specs,
                 const MapEntry* feats, const MapEntry* targs) {
    if (job.capacity == 4) return run_shape<T, N, M, 4>(in, out, job, specs, feats, targs);
    if (job.capacity == 8) return run_shape<T, N, M, 8>(in, out, job, specs, feats, targs);
    if (job.capacity == 64) return run_shape<T, N, M, 64>(in, out, job, specs, feats, targs);
    fprintf(stderr, "pipecheck: capacity %u is not in the instantiation list\n",
            job.capacity);
    return 2;
}

template <typename T>
int run(FILE* in, FILE* out, const Job& job, const StreamSpec* specs,
        const MapEntry* feats, const MapEntry* targs) {
    if (job.n_outputs == 3) {
        if (job.n_features == 6) return run_capacity<T, 6, 3>(in, out, job, specs, feats,
                                                              targs);
        fprintf(stderr, "pipecheck: three outputs exists at six features only\n");
        return 2;
    }
    if (job.n_features == 2) return run_capacity<T, 2, 1>(in, out, job, specs, feats, targs);
    if (job.n_features == 3) return run_capacity<T, 3, 1>(in, out, job, specs, feats, targs);
    if (job.n_features == 6) return run_capacity<T, 6, 1>(in, out, job, specs, feats, targs);
    fprintf(stderr, "pipecheck: n_features %u is not in the instantiation list\n",
            job.n_features);
    return 2;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: pipecheck <job-file> <out-file>\n");
        return 1;
    }
    FILE* in = fopen(argv[1], "rb");
    if (!in) return 1;
    Job job;
    if (fread(&job, sizeof(job), 1, in) != 1 || job.magic[0] != 'A' ||
        job.magic[1] != 'M' || job.magic[2] != 'P' || job.magic[3] != 'L') {
        fprintf(stderr, "pipecheck: not a job file\n");
        fclose(in);
        return 1;
    }
    if (job.version != kVersion) {
        fprintf(stderr, "pipecheck: job format v%u, this build speaks v%u\n", job.version,
                kVersion);
        fclose(in);
        return 1;
    }
    if (job.n_streams < 1 || job.n_streams > kMaxStreams ||
        job.n_map_features > kMaxFeatures) {
        fclose(in);
        return 5;
    }
    StreamSpec specs[kMaxStreams];
    for (uint16_t i = 0; i < job.n_streams; ++i) {
        if (fread(&specs[i], sizeof(StreamSpec), 1, in) != 1) {
            fclose(in);
            return 4;
        }
        if (specs[i].dim < 1 || specs[i].dim > kMaxDim) {
            fclose(in);
            return 5;
        }
    }
    MapEntry feats[kMaxFeatures];
    for (uint16_t j = 0; j < job.n_map_features; ++j) {
        if (fread(&feats[j], sizeof(MapEntry), 1, in) != 1) {
            fclose(in);
            return 4;
        }
    }
    MapEntry targs[3];
    const uint16_t n_targ = job.n_outputs ? job.n_outputs : 1;
    if (n_targ > 3) {
        fclose(in);
        return 5;
    }
    for (uint16_t j = 0; j < n_targ; ++j) {
        if (fread(&targs[j], sizeof(MapEntry), 1, in) != 1) {
            fclose(in);
            return 4;
        }
    }
    FILE* out = fopen(argv[2], "wb");
    if (!out) {
        fclose(in);
        return 1;
    }
    const int rc = job.dtype == 4 ? run<float>(in, out, job, specs, feats, targs)
                                  : run<double>(in, out, job, specs, feats, targs);
    fclose(in);
    fclose(out);
    return rc;
}
