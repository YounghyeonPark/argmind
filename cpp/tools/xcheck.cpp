// SPDX-License-Identifier: Apache-2.0
//
// The parity tool. Reads a case file the Python reference generated, runs the same
// updates through the C++ core, writes what happened, and lets the Python compare.
//
// It writes the whole *trajectory* rather than only the final state. An endpoint
// comparison can pass while the two implementations disagree in compensating ways, and a
// divergence that appears on step one is a different fault from one that grows over a
// run. Both are worth being able to tell apart.
//
// The format is binary because a text round trip loses bits, and the whole point is to
// compare bits. Little-endian with explicit widths, for the same reason
// argmind/persist.py is: the writer and the reader may be different machines.
//
// Every field is placed at its natural alignment and every struct is a multiple of eight
// bytes, so these can be read with a single fread on either side without packing
// pragmas. If a field is added, keep that property — a misaligned double read through a
// packed struct is undefined behaviour on the target even where it works on x86.
//
// One case format covers both bare and guarded runs. A guard with no bounds and no
// divergence enabled is simply not applied, which keeps the tool to one code path
// instead of two that can drift apart.

#include <stdint.h>
#include <stdio.h>

#include "argmind/guard.hpp"
#include "argmind/rls.hpp"

namespace {

constexpr uint16_t kFormatVersion = 2;

struct BoundSlot {
    double lo;
    double hi;
    uint32_t field;
    uint32_t flags;  // bit 0 has_lo, bit 1 has_hi
};

struct DivergenceCfg {
    double ratio;
    double fast_alpha;
    double slow_alpha;
    double floor;
    uint32_t warmup;
    uint32_t enabled;
};

struct CaseHeader {
    char magic[4];  // "AMXC"
    uint16_t version;
    uint8_t dtype;  // 4 or 8
    uint8_t pad;
    uint16_t n;
    uint16_t m;
    uint32_t count;
    double lambda;
    double p0;
    uint32_t depth;
    uint32_t checkpoint_every;
    uint32_t recover_after;
    uint32_t n_bounds;
    BoundSlot bounds[4];
    DivergenceCfg divergence;
};
static_assert(sizeof(BoundSlot) == 24, "case format: BoundSlot must be 24 bytes");
static_assert(sizeof(DivergenceCfg) == 40, "case format: DivergenceCfg must be 40 bytes");
static_assert(sizeof(CaseHeader) == 184, "case format: header must be 184 bytes");

struct ResultHeader {
    char magic[4];  // "AMXR"
    uint16_t version;
    uint8_t dtype;
    uint8_t pad;
    uint16_t n;
    uint16_t m;
    uint32_t count;
};
static_assert(sizeof(ResultHeader) == 16, "result format: header must be 16 bytes");

// One row per update, fixed width so the Python reads it with one struct.
struct Row {
    int32_t applied;
    int32_t reason;
    uint32_t tripped;
    uint32_t flags;  // bit 0 frozen, bit 1 rolled_back
    double residual_norm;
    double residual_post_norm;
    double gain_norm;
    double p_trace;
    double p_min_diag;
};
static_assert(sizeof(Row) == 56, "result format: row must be 56 bytes");

bool configure(const CaseHeader& h, argmind::Divergence* d) {
    if (!h.divergence.enabled) return false;
    d->enabled = true;
    d->ratio = h.divergence.ratio;
    d->fast_alpha = h.divergence.fast_alpha;
    d->slow_alpha = h.divergence.slow_alpha;
    d->floor = h.divergence.floor;
    d->warmup = h.divergence.warmup;
    return true;
}

argmind::Bound bound_from(const BoundSlot& s) {
    argmind::Bound b;
    b.field = static_cast<argmind::Field>(s.field);
    b.lo = s.lo;
    b.hi = s.hi;
    b.has_lo = (s.flags & 1u) != 0;
    b.has_hi = (s.flags & 2u) != 0;
    return b;
}

template <typename Report>
void fill_row(const Report& r, Row* row) {
    row->applied = r.applied ? 1 : 0;
    row->reason = static_cast<int32_t>(r.reason);
    row->tripped = 0;
    row->flags = 0;
    row->residual_norm = r.residual_norm;
    row->residual_post_norm = r.residual_post_norm;
    row->gain_norm = r.gain_norm;
    row->p_trace = r.p_trace;
    row->p_min_diag = r.p_min_diag;
}

int write_header(FILE* out, const CaseHeader& h) {
    ResultHeader rh = {{'A', 'M', 'X', 'R'}, kFormatVersion, h.dtype, 0, h.n, h.m,
                       h.count};
    return fwrite(&rh, sizeof(rh), 1, out) == 1 ? 0 : 3;
}

template <typename T, int N, int M>
int run_bare(FILE* in, FILE* out, const CaseHeader& h) {
    argmind::Rls<T, N, M> rls(static_cast<T>(h.lambda), static_cast<T>(h.p0));
    if (!rls.valid()) {
        fprintf(stderr, "xcheck: lambda/p0 rejected by the core\n");
        return 2;
    }
    if (int rc = write_header(out, h)) return rc;

    for (uint32_t i = 0; i < h.count; ++i) {
        T x[N];
        T y[M];
        if (fread(x, sizeof(T), N, in) != N) return 4;
        if (fread(y, sizeof(T), M, in) != M) return 4;
        Row row;
        fill_row(rls.update(x, y), &row);
        if (fwrite(&row, sizeof(row), 1, out) != 1) return 3;
    }
    const int size = argmind::Rls<T, N, M>::state_size;
    return fwrite(rls.state(), sizeof(T), size, out) == static_cast<size_t>(size) ? 0 : 3;
}

template <typename T, int N, int M, int Depth>
int run_guarded(FILE* in, FILE* out, const CaseHeader& h) {
    using Adapter = argmind::Rls<T, N, M>;
    Adapter inner(static_cast<T>(h.lambda), static_cast<T>(h.p0));
    if (!inner.valid()) {
        fprintf(stderr, "xcheck: lambda/p0 rejected by the core\n");
        return 2;
    }
    argmind::Guard<Adapter, Depth, 4> guard(inner, h.checkpoint_every, h.recover_after);
    for (uint32_t b = 0; b < h.n_bounds && b < 4; ++b) {
        if (!guard.add_bound(bound_from(h.bounds[b]))) {
            fprintf(stderr, "xcheck: bound %u rejected by the core\n", b);
            return 2;
        }
    }
    argmind::Divergence d;
    if (configure(h, &d)) guard.set_divergence(d);
    if (!guard.valid()) {
        fprintf(stderr, "xcheck: guard configuration rejected by the core\n");
        return 2;
    }
    if (int rc = write_header(out, h)) return rc;

    for (uint32_t i = 0; i < h.count; ++i) {
        T x[N];
        T y[M];
        if (fread(x, sizeof(T), N, in) != N) return 4;
        if (fread(y, sizeof(T), M, in) != M) return 4;
        const argmind::GuardReport<T, M> r = guard.update(x, y);
        Row row;
        fill_row(r, &row);
        row.tripped = r.tripped;
        row.flags = (r.frozen ? 1u : 0u) | (r.rolled_back ? 2u : 0u);
        if (fwrite(&row, sizeof(row), 1, out) != 1) return 3;
    }
    const int size = Adapter::state_size;
    return fwrite(guard.state(), sizeof(T), size, out) == static_cast<size_t>(size) ? 0
                                                                                   : 3;
}

// The instantiation lists below are dispatch tables rather than a runtime `n`, which is
// the same trade the C ABI makes and for the same reason: compile-time dimensions are
// what the core exists for. Adding a shape means adding a line.

template <typename T, int N, int M>
int dispatch_depth(FILE* in, FILE* out, const CaseHeader& h) {
    const bool guarded = h.n_bounds > 0 || h.divergence.enabled;
    if (!guarded) return run_bare<T, N, M>(in, out, h);
    switch (h.depth) {
        case 1: return run_guarded<T, N, M, 1>(in, out, h);
        case 2: return run_guarded<T, N, M, 2>(in, out, h);
        case 4: return run_guarded<T, N, M, 4>(in, out, h);
        case 10: return run_guarded<T, N, M, 10>(in, out, h);
        default:
            fprintf(stderr, "xcheck: no instantiation for rollback depth %u; add one "
                            "to dispatch_depth in xcheck.cpp\n", h.depth);
            return 5;
    }
}

template <typename T>
int dispatch(FILE* in, FILE* out, const CaseHeader& h) {
    if (h.m == 1 && h.n == 2) return dispatch_depth<T, 2, 1>(in, out, h);
    if (h.m == 1 && h.n == 3) return dispatch_depth<T, 3, 1>(in, out, h);
    if (h.m == 1 && h.n == 6) return dispatch_depth<T, 6, 1>(in, out, h);
    if (h.m == 1 && h.n == 12) return dispatch_depth<T, 12, 1>(in, out, h);
    if (h.m == 3 && h.n == 6) return dispatch_depth<T, 6, 3>(in, out, h);
    fprintf(stderr, "xcheck: no instantiation for n=%u m=%u; add one to the dispatch "
                    "table in xcheck.cpp\n", h.n, h.m);
    return 5;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: xcheck <case-file> <result-file>\n");
        return 1;
    }
    FILE* in = fopen(argv[1], "rb");
    if (!in) {
        fprintf(stderr, "xcheck: cannot open %s\n", argv[1]);
        return 1;
    }
    CaseHeader h;
    if (fread(&h, sizeof(h), 1, in) != 1 || h.magic[0] != 'A' || h.magic[1] != 'M' ||
        h.magic[2] != 'X' || h.magic[3] != 'C') {
        fprintf(stderr, "xcheck: %s is not a case file\n", argv[1]);
        fclose(in);
        return 1;
    }
    if (h.version != kFormatVersion) {
        fprintf(stderr, "xcheck: case format version %u, this build speaks %u\n",
                h.version, kFormatVersion);
        fclose(in);
        return 1;
    }
    FILE* out = fopen(argv[2], "wb");
    if (!out) {
        fprintf(stderr, "xcheck: cannot write %s\n", argv[2]);
        fclose(in);
        return 1;
    }

    int rc;
    if (h.dtype == 4) {
        rc = dispatch<float>(in, out, h);
    } else if (h.dtype == 8) {
        rc = dispatch<double>(in, out, h);
    } else {
        fprintf(stderr, "xcheck: unsupported dtype code %u\n", h.dtype);
        rc = 6;
    }
    fclose(in);
    fclose(out);
    return rc;
}
