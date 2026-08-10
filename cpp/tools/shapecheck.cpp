// SPDX-License-Identifier: Apache-2.0
//
// Parity for the runtime-shaped adapters, against the same Python reference.
//
// It reads and writes `xcheck`'s file format unchanged -- the case header already carries
// `n` and `m` as `uint16_t`, and the only thing stopping `xcheck` from accepting an
// arbitrary pair is its dispatch table. So the Python harness reuses its own encoder and
// reader, and nothing here can drift from them.
//
// **The point of this tool is the shapes `xcheck` cannot reach.** Comparing the runtime
// layer at 6x1 would mostly re-test the reference against itself; what has to be checked
// is n=5, n=7, n=50, m=4 -- the shapes a configuration file would produce and no
// instantiation list contains.
//
// Guarded cases run through `RuntimeShapedGuard`. They were refused before it existed,
// because a tool that quietly dropped the guard would have made that gap invisible --
// and the gap mattered more than the arithmetic, since "no guard" being expressible is
// the failure this library is built around.

#include <stdint.h>
#include <stdio.h>

#include <vector>

#include "argmind/host/runtime_shape_guard.hpp"

namespace {

constexpr uint16_t kFormatVersion = 2;

struct BoundSlot {
    double lo, hi;
    uint32_t field, flags;
};

struct DivergenceCfg {
    double ratio, fast_alpha, slow_alpha, floor;
    uint32_t warmup, enabled;
};

struct CaseHeader {
    char magic[4];
    uint16_t version;
    uint8_t dtype;
    uint8_t adapter;   // 0 = Rls, 1 = Nlms
    uint16_t n, m;
    uint32_t count;
    double lambda, p0;
    uint32_t depth, checkpoint_every, recover_after, n_bounds;
    BoundSlot bounds[4];
    DivergenceCfg divergence;
};
static_assert(sizeof(CaseHeader) == 184, "must match xcheck's header exactly");

struct ResultHeader {
    char magic[4];
    uint16_t version;
    uint8_t dtype, pad;
    uint16_t n, m;
    uint32_t count;
};
static_assert(sizeof(ResultHeader) == 16, "must match xcheck's result header");

struct Row {
    int32_t applied, reason;
    uint32_t tripped, flags;
    double residual_norm, residual_post_norm, gain_norm, p_trace, p_min_diag;
};
static_assert(sizeof(Row) == 56, "must match xcheck's row");

argmind::Bound bound_from(const BoundSlot& s) {
    argmind::Bound b;
    b.field = static_cast<argmind::Field>(s.field);
    b.lo = s.lo;
    b.hi = s.hi;
    b.has_lo = (s.flags & 1u) != 0;
    b.has_hi = (s.flags & 2u) != 0;
    return b;
}

/// One driver for guarded and bare alike. `Reporter` pulls the guard-only fields out,
/// because a bare report has no `tripped` and inventing a zero for it in the bare path
/// would be the same shape of lie the guard exists to prevent.
template <typename T, typename Adapter, typename Reporter>
int drive(FILE* in, FILE* out, const CaseHeader& h, Adapter& a, Reporter extra) {
    if (!a.valid()) {
        fprintf(stderr, "shapecheck: adapter parameters rejected by the core\n");
        return 2;
    }
    ResultHeader rh = {{'A', 'M', 'X', 'R'}, kFormatVersion, h.dtype, 0, h.n, h.m,
                       h.count};
    if (fwrite(&rh, sizeof(rh), 1, out) != 1) return 3;

    std::vector<T> x(h.n), y(h.m);
    for (uint32_t i = 0; i < h.count; ++i) {
        if (fread(x.data(), sizeof(T), h.n, in) != h.n) return 4;
        if (fread(y.data(), sizeof(T), h.m, in) != h.m) return 4;
        const auto rep = a.update(x.data(), y.data());
        Row row{};
        extra(rep, &row);
        row.applied = rep.applied ? 1 : 0;
        row.reason = static_cast<int32_t>(rep.reason);
        row.residual_norm = rep.residual_norm;
        row.residual_post_norm = rep.residual_post_norm;
        row.gain_norm = rep.gain_norm;
        row.p_trace = rep.p_trace;
        row.p_min_diag = rep.p_min_diag;
        if (fwrite(&row, sizeof(row), 1, out) != 1) return 3;
    }
    const size_t size = static_cast<size_t>(a.state_size());
    return fwrite(a.state(), sizeof(T), size, out) == size ? 0 : 3;
    // `a` is the guard in a guarded run, and `state()` delegates -- so this is the state
    // AFTER any rollback, which is what the reference's own `state` reports too.
}

auto bare_extra = [](const auto&, Row*) {};

auto guarded_extra = [](const auto& rep, Row* row) {
    row->tripped = rep.tripped;
    row->flags = (rep.frozen ? 1u : 0u) | (rep.rolled_back ? 2u : 0u);
};

template <typename T, typename Adapter>
int run_guarded(FILE* in, FILE* out, const CaseHeader& h, Adapter inner) {
    argmind::host::RuntimeShapedGuard<Adapter> g(
        inner, static_cast<int>(h.depth), h.checkpoint_every, h.recover_after,
        4, /*n_external=*/0);
    for (uint32_t i = 0; i < h.n_bounds && i < 4; ++i) {
        if (!g.add_bound(bound_from(h.bounds[i]))) {
            fprintf(stderr, "shapecheck: bound %u refused by the guard\n", i);
            return 2;
        }
    }
    if (h.divergence.enabled) {
        argmind::Divergence d;
        d.ratio = h.divergence.ratio;
        d.fast_alpha = h.divergence.fast_alpha;
        d.slow_alpha = h.divergence.slow_alpha;
        d.floor = h.divergence.floor;
        d.warmup = h.divergence.warmup;
        g.set_divergence(d);
    }
    if (!g.valid()) {
        fprintf(stderr, "shapecheck: the guard configuration is not usable\n");
        return 2;
    }
    return drive<T>(in, out, h, g, guarded_extra);
}

template <typename T>
int run(FILE* in, FILE* out, const CaseHeader& h) {
    const bool guarded = h.n_bounds > 0 || h.divergence.enabled;
    if (h.adapter == 1) {
        argmind::host::RuntimeShapedNlms<T> a(h.n, h.m, static_cast<T>(h.lambda),
                                              static_cast<T>(h.p0));
        if (guarded) return run_guarded<T>(in, out, h, a);
        return drive<T>(in, out, h, a, bare_extra);
    }
    argmind::host::RuntimeShapedRls<T> a(h.n, h.m, static_cast<T>(h.lambda),
                                         static_cast<T>(h.p0));
    if (guarded) return run_guarded<T>(in, out, h, a);
    return drive<T>(in, out, h, a, bare_extra);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: shapecheck <case-file> <result-file>\n");
        return 1;
    }
    FILE* in = fopen(argv[1], "rb");
    if (!in) return 4;
    CaseHeader h;
    if (fread(&h, sizeof(h), 1, in) != 1) {
        fclose(in);
        return 4;
    }
    if (h.magic[0] != 'A' || h.magic[1] != 'M' || h.magic[2] != 'X' ||
        h.magic[3] != 'C' || h.version != kFormatVersion) {
        fprintf(stderr, "shapecheck: not an AMXC v%u case file\n", kFormatVersion);
        fclose(in);
        return 4;
    }
    if (h.n < 1 || h.m < 1) {
        fclose(in);
        return 5;
    }

    FILE* out = fopen(argv[2], "wb");
    if (!out) {
        fclose(in);
        return 3;
    }
    const int rc = (h.dtype == 4)   ? run<float>(in, out, h)
                   : (h.dtype == 8) ? run<double>(in, out, h)
                                    : 5;
    fclose(in);
    fclose(out);
    return rc;
}
