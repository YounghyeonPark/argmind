// SPDX-License-Identifier: Apache-2.0
//
// The parity tool. Reads a case file the Python reference generated, runs the same
// updates through the C++ core, writes what happened, and lets the Python compare.
//
// It writes the whole *trajectory* rather than only the final state. An endpoint
// comparison can pass while the two implementations disagree in compensating ways,
// and a divergence that grows over the run is a different fault from one that is
// there on the first step. Both are worth being able to tell apart.
//
// The format is binary because a text round trip loses bits, and the whole point is
// to compare bits. Little-endian with explicit widths, for the same reason
// argmind/persist.py is: the writer and the reader may be different machines.

#include <stdint.h>
#include <stdio.h>

#include "argmind/rls.hpp"

namespace {

struct CaseHeader {
    char magic[4];      // "AMXC"
    uint16_t version;
    uint8_t dtype;      // 4 or 8
    uint8_t pad;
    uint16_t n;
    uint16_t m;
    uint32_t count;
    double lambda;
    double p0;
};

struct ResultHeader {
    char magic[4];      // "AMXR"
    uint16_t version;
    uint8_t dtype;
    uint8_t pad;
    uint16_t n;
    uint16_t m;
    uint32_t count;
};

// One row per update. Fixed width so the Python can read it with one dtype.
struct Row {
    int32_t applied;
    int32_t reason;
    double residual_norm;
    double residual_post_norm;
    double gain_norm;
    double p_trace;
    double p_min_diag;
};

template <typename T, int N, int M>
int run(FILE* in, FILE* out, const CaseHeader& h) {
    argmind::Rls<T, N, M> rls(static_cast<T>(h.lambda), static_cast<T>(h.p0));
    if (!rls.valid()) {
        fprintf(stderr, "xcheck: lambda/p0 rejected by the core\n");
        return 2;
    }

    ResultHeader rh = {{'A', 'M', 'X', 'R'}, 1, h.dtype, 0, h.n, h.m, h.count};
    if (fwrite(&rh, sizeof(rh), 1, out) != 1) return 3;

    for (uint32_t i = 0; i < h.count; ++i) {
        T x[N];
        T y[M];
        if (fread(x, sizeof(T), N, in) != N) return 4;
        if (fread(y, sizeof(T), M, in) != M) return 4;

        const argmind::Report<T, M> r = rls.update(x, y);
        Row row;
        row.applied = r.applied ? 1 : 0;
        row.reason = static_cast<int32_t>(r.reason);
        row.residual_norm = r.residual_norm;
        row.residual_post_norm = r.residual_post_norm;
        row.gain_norm = r.gain_norm;
        row.p_trace = r.p_trace;
        row.p_min_diag = r.p_min_diag;
        if (fwrite(&row, sizeof(row), 1, out) != 1) return 3;
    }

    const int size = argmind::Rls<T, N, M>::state_size;
    if (fwrite(rls.state(), sizeof(T), size, out) != static_cast<size_t>(size))
        return 3;
    return 0;
}

// The instantiation list the tool understands. Compile-time dimensions mean this
// is a dispatch table rather than a runtime `n`, which is the same trade the C ABI
// makes and for the same reason.
template <typename T>
int dispatch(FILE* in, FILE* out, const CaseHeader& h) {
    if (h.m == 1 && h.n == 2) return run<T, 2, 1>(in, out, h);
    if (h.m == 1 && h.n == 3) return run<T, 3, 1>(in, out, h);
    if (h.m == 1 && h.n == 6) return run<T, 6, 1>(in, out, h);
    if (h.m == 1 && h.n == 12) return run<T, 12, 1>(in, out, h);
    if (h.m == 3 && h.n == 6) return run<T, 6, 3>(in, out, h);
    fprintf(stderr, "xcheck: no instantiation for n=%u m=%u; add one to the "
                    "dispatch table in xcheck.cpp\n", h.n, h.m);
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
    if (fread(&h, sizeof(h), 1, in) != 1 || h.magic[0] != 'A' ||
        h.magic[1] != 'M' || h.magic[2] != 'X' || h.magic[3] != 'C') {
        fprintf(stderr, "xcheck: %s is not a case file\n", argv[1]);
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
