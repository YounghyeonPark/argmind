// SPDX-License-Identifier: Apache-2.0
//
// The stream and alignment parity tool.
//
// Two modes, because the two halves fail differently. Mode 0 drives the ring itself —
// whether a push is accepted, what gets overwritten, what the retained timestamps are — and
// mode 1 drives the four alignment policies, which is where the subtle behaviour lives:
// signed staleness, refusing to extrapolate, and the fact that the choice of policy changes
// the answer at all.
//
// Mode 1 builds `StreamView`s over flat buffers rather than instantiating `Stream<>` for
// every shape. That is deliberate: the view is what the aligner actually consumes, so this
// exercises the code under test without an instantiation table that would have to grow every
// time a test wanted a new dimension. `Stream<>`'s own mechanics are mode 0's job and
// cpp/tests/test_core.cpp's.
//
// Little-endian, naturally aligned, every struct a multiple of eight bytes — same rules as
// the other tools, and for the same reason.

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "argmind/align.hpp"
#include "argmind/stream.hpp"

namespace {

constexpr uint16_t kVersion = 1;
constexpr int kMaxStreams = 4;
// Host-only tool. These bounds cost about 1.8 MB of BSS, which is nothing on a development
// machine and is not a target figure.
constexpr int kMaxSamples = 1024;
// Matches the aligner's own limit rather than sitting below it. At 8 the tool refused every
// stream wider than that before `Aligner::pick` ever saw it, so the boundary the library
// actually enforces was unreachable from the harness that exists to test it.
constexpr int kMaxDim = argmind::Pick<double>::MAX_DIM;

struct Job {
    char magic[4];  // "AMSS"
    uint16_t version;
    uint8_t dtype;   // 4 or 8
    uint8_t mode;    // 0 = ring, 1 = alignment
    uint16_t policy;
    uint16_t n_streams;
    uint32_t n_pushes;
    uint32_t n_queries;
    double max_stale;
};
static_assert(sizeof(Job) == 32, "job header must be 32 bytes");

struct StreamSpec {
    uint16_t dim;
    uint16_t capacity;
    uint32_t pad;
};
static_assert(sizeof(StreamSpec) == 8, "stream spec must be 8 bytes");

/// One push, as the job file describes it. Values follow, `dim` of them.
struct PushHeader {
    uint32_t stream;
    uint32_t valid;
    double t;
};
static_assert(sizeof(PushHeader) == 16, "push header must be 16 bytes");

/// What one stream contributed to one query.
struct PickOut {
    uint8_t present;
    uint8_t valid;
    /// Whether this pick's staleness exceeds the aligner's limit, i.e. whether
    /// `DropIfStale` would abandon the frame. `pick()` itself never consults the limit —
    /// `HoldLast` and `DropIfStale` share one branch — so without this the fourth policy
    /// was definitionally identical to the first in everything the harness compared, and
    /// the only decision that distinguishes it went unchecked.
    uint8_t refused;
    uint8_t pad[5];
    double staleness;
};
static_assert(sizeof(PickOut) == 16, "pick result must be 16 bytes");

/// Rings built by hand so that any (dim, capacity) works without an instantiation table.
template <typename T>
struct Ring {
    T values[kMaxSamples * kMaxDim];
    double times[kMaxSamples];
    uint8_t valid[kMaxSamples];
    int dim = 0;
    int capacity = 0;
    int head = 0;
    int count = 0;
    uint64_t dropped = 0;
    double last_t = 0.0;
    bool started = false;

    void init(int d, int cap) {
        dim = d;
        capacity = cap;
        head = count = 0;
        dropped = 0;
        started = false;
    }

    /// Mirrors Stream::push, including the monotonicity refusal.
    ///
    /// A copy, and copies drift. `Stream<T, Dim, Capacity>` cannot be used here because the
    /// aligner has to see streams of several widths at once and those widths are template
    /// parameters — the same tension `StreamView` exists to resolve. So the shipped class is
    /// exercised by `cpp/tests/test_core.cpp` instead, and this note is here so nobody reads
    /// the tests below as covering `Stream::push` itself. They cover `StreamView` and the
    /// aligner, which is what they are for.
    bool push(double t, const T* v, bool ok) {
        if (started && !(t > last_t)) return false;
        if (!argmind::is_finite(t)) return false;
        if (count == capacity) ++dropped;
        for (int j = 0; j < dim; ++j) values[head * dim + j] = v[j];
        times[head] = t;
        valid[head] = ok ? 1 : 0;
        head = (head + 1) % capacity;
        if (count < capacity) ++count;
        last_t = t;
        started = true;
        return true;
    }

    argmind::StreamView<T> view() const {
        argmind::StreamView<T> s;
        s.times = times;
        s.values = values;
        s.valid = valid;
        s.dim = dim;
        s.capacity = capacity;
        s.head = head;
        s.count = count;
        return s;
    }
};

template <typename T>
Ring<T>& ring(int i) {
    static Ring<T> rings[kMaxStreams];
    return rings[i];
}

template <typename T>
int load_streams(FILE* in, const Job& job, StreamSpec* specs) {
    for (uint16_t i = 0; i < job.n_streams; ++i) {
        if (fread(&specs[i], sizeof(StreamSpec), 1, in) != 1) return 4;
        if (specs[i].dim > kMaxDim || specs[i].capacity > kMaxSamples) return 5;
        ring<T>(i).init(specs[i].dim, specs[i].capacity);
    }
    return 0;
}

template <typename T>
int run(FILE* in, FILE* out, const Job& job) {
    StreamSpec specs[kMaxStreams];
    if (job.n_streams > kMaxStreams) return 5;
    if (int rc = load_streams<T>(in, job, specs)) return rc;

    // Replay the pushes, reporting which were accepted. A refusal is data: the reference
    // raises on non-monotonic time and this returns false, and the two have to refuse the
    // same samples or every later comparison is against a different history.
    for (uint32_t p = 0; p < job.n_pushes; ++p) {
        PushHeader h;
        if (fread(&h, sizeof(h), 1, in) != 1) return 4;
        if (h.stream >= job.n_streams) return 5;
        T v[kMaxDim];
        const int dim = specs[h.stream].dim;
        if (fread(v, sizeof(T), dim, in) != static_cast<size_t>(dim)) return 4;
        const uint8_t accepted =
            ring<T>(h.stream).push(h.t, v, h.valid != 0) ? 1u : 0u;
        if (fwrite(&accepted, 1, 1, out) != 1) return 3;
    }

    // Ring state: count, dropped, then the retained timestamps oldest-first.
    for (uint16_t i = 0; i < job.n_streams; ++i) {
        const Ring<T>& r = ring<T>(i);
        const uint32_t stats[2] = {static_cast<uint32_t>(r.count),
                                   static_cast<uint32_t>(r.dropped)};
        if (fwrite(stats, sizeof(uint32_t), 2, out) != 2) return 3;
        const argmind::StreamView<T> v = r.view();
        for (int age = 0; age < r.count; ++age) {
            const double t = v.times[v.slot(age)];
            if (fwrite(&t, sizeof(double), 1, out) != 1) return 3;
        }
    }
    if (job.mode == 0) return 0;

    // Alignment: for each reference instant, every stream's pick.
    const argmind::Aligner aligner(static_cast<argmind::Policy>(job.policy),
                                   job.max_stale);
    for (uint32_t q = 0; q < job.n_queries; ++q) {
        double t = 0.0;
        if (fread(&t, sizeof(t), 1, in) != 1) return 4;
        for (uint16_t i = 0; i < job.n_streams; ++i) {
            argmind::Pick<T> pick;
            const bool got = aligner.pick<T>(ring<T>(i).view(), t, &pick);
            PickOut o;
            memset(&o, 0, sizeof(o));
            o.present = got ? 1u : 0u;
            o.valid = (got && pick.valid) ? 1u : 0u;
            o.staleness = got ? pick.staleness : 0.0;
            o.refused = (got && aligner.refuses_stale(pick.staleness)) ? 1u : 0u;
            if (fwrite(&o, sizeof(o), 1, out) != 1) return 3;
            const int dim = specs[i].dim;
            if (got) {
                if (fwrite(pick.values, sizeof(T), dim, out) !=
                    static_cast<size_t>(dim)) return 3;
            } else {
                T zeros[kMaxDim];
                for (int j = 0; j < dim; ++j) zeros[j] = T(0);
                if (fwrite(zeros, sizeof(T), dim, out) !=
                    static_cast<size_t>(dim)) return 3;
            }
        }
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: scheck <job-file> <out-file>\n");
        return 1;
    }
    FILE* in = fopen(argv[1], "rb");
    if (!in) return 1;
    Job job;
    if (fread(&job, sizeof(job), 1, in) != 1 || job.magic[0] != 'A' ||
        job.magic[1] != 'M' || job.magic[2] != 'S' || job.magic[3] != 'S') {
        fprintf(stderr, "scheck: not a job file\n");
        fclose(in);
        return 1;
    }
    if (job.version != kVersion) {
        fprintf(stderr, "scheck: job format v%u, this build speaks v%u\n", job.version,
                kVersion);
        fclose(in);
        return 1;
    }
    FILE* out = fopen(argv[2], "wb");
    if (!out) {
        fclose(in);
        return 1;
    }
    const int rc = job.dtype == 4 ? run<float>(in, out, job) : run<double>(in, out, job);
    fclose(in);
    fclose(out);
    return rc;
}
