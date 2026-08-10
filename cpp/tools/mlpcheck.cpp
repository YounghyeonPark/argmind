// SPDX-License-Identifier: Apache-2.0
//
// Parity for `RuntimeShapedMlp`, against `argmind.adapt.MultilayerPerceptron`.
//
// **The result format is `xcheck`'s, byte for byte** -- the same `ResultHeader` and the
// same 56-byte `Row`, with their `static_assert`s copied intact -- so the Python side
// reuses one reader and nothing here can drift from it.
//
// **The case format is not, and could not be.** `xcheck`'s header is a fixed 184 bytes
// carrying `lambda` and `p0`, which is the whole configuration of an RLS. A network's
// configuration is a variable-length list of layer widths, plus an initial weight block,
// because **the two initialisations do not match**: the reference draws from numpy's
// PCG64 and this draws from splitmix64, deliberately, since a generator that nearly
// matches is worse than one that plainly does not. So the harness trains nothing into the
// C++ side by hand -- it hands over the reference's own starting weights, which is also
// what a device restoring a trained model does.

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <vector>

#include "argmind/host/runtime_shape_mlp.hpp"

namespace {

constexpr uint16_t kFormatVersion = 1;

struct CaseHeader {
    char magic[4];        // "AMMP"
    uint16_t version;
    uint8_t dtype;        // 4 or 8
    uint8_t activation;   // 0 tanh, 1 relu
    uint16_t n, m;
    uint16_t n_hidden;
    uint16_t pad;
    uint32_t count;
    uint32_t state_size;
    double learning_rate;
};
static_assert(sizeof(CaseHeader) == 32, "the Python encoder packs exactly this");

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

template <typename T>
int run(const CaseHeader& h, const std::vector<uint32_t>& hidden,
        const std::vector<T>& init, const std::vector<T>& samples, FILE* out) {
    std::vector<int> widths;
    for (uint32_t w : hidden) widths.push_back(static_cast<int>(w));

    argmind::host::RuntimeShapedMlp<T> mlp(
        h.n, h.m, widths,
        static_cast<argmind::host::Activation>(h.activation),
        static_cast<T>(h.learning_rate));

    if (mlp.state_size() != static_cast<int>(h.state_size)) {
        fprintf(stderr, "state size %d does not match the reference's %u\n",
                mlp.state_size(), h.state_size);
        return 4;
    }
    if (!mlp.valid() || !mlp.load_state(init.data())) {
        fprintf(stderr, "refused the reference's configuration or starting weights\n");
        return 5;
    }

    ResultHeader rh{};
    memcpy(rh.magic, "AMRS", 4);
    rh.version = kFormatVersion;
    rh.dtype = h.dtype;
    rh.n = h.n;
    rh.m = h.m;
    rh.count = h.count;
    if (fwrite(&rh, sizeof rh, 1, out) != 1) return 6;

    const size_t stride = static_cast<size_t>(h.n) + h.m;
    for (uint32_t k = 0; k < h.count; ++k) {
        const T* x = samples.data() + k * stride;
        const T* y = x + h.n;
        const argmind::host::RuntimeShapedReport<T> r = mlp.update(x, y);

        Row row{};
        row.applied = r.applied ? 1 : 0;
        row.reason = static_cast<int32_t>(r.reason);
        row.residual_norm = r.residual_norm;
        row.residual_post_norm = r.residual_post_norm;
        row.gain_norm = r.gain_norm;
        row.p_trace = r.p_trace;
        row.p_min_diag = r.p_min_diag;
        if (fwrite(&row, sizeof row, 1, out) != 1) return 6;
    }

    // The state block last, exactly as `xcheck` does, so one reader serves both.
    if (fwrite(mlp.state(), sizeof(T), static_cast<size_t>(mlp.state_size()), out)
        != static_cast<size_t>(mlp.state_size())) {
        return 6;
    }
    return 0;
}

template <typename T>
bool read_vec(FILE* f, size_t n, std::vector<T>& v) {
    v.resize(n);
    return n == 0 || fread(v.data(), sizeof(T), n, f) == n;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: mlpcheck <case> <result>\n");
        return 1;
    }
    FILE* in = fopen(argv[1], "rb");
    if (!in) {
        fprintf(stderr, "cannot open %s\n", argv[1]);
        return 2;
    }
    CaseHeader h{};
    if (fread(&h, sizeof h, 1, in) != 1 || memcmp(h.magic, "AMMP", 4) != 0 ||
        h.version != kFormatVersion) {
        fprintf(stderr, "not an mlpcheck case file\n");
        fclose(in);
        return 3;
    }

    std::vector<uint32_t> hidden(h.n_hidden);
    if (h.n_hidden && fread(hidden.data(), sizeof(uint32_t), h.n_hidden, in) != h.n_hidden) {
        fclose(in);
        return 3;
    }

    const size_t n_samples = static_cast<size_t>(h.count) * (static_cast<size_t>(h.n) + h.m);
    int rc = 3;
    FILE* out = nullptr;

    if (h.dtype == 4) {
        std::vector<float> init, samples;
        if (read_vec(in, h.state_size, init) && read_vec(in, n_samples, samples)) {
            fclose(in);
            in = nullptr;
            out = fopen(argv[2], "wb");
            rc = out ? run<float>(h, hidden, init, samples, out) : 6;
        }
    } else if (h.dtype == 8) {
        std::vector<double> init, samples;
        if (read_vec(in, h.state_size, init) && read_vec(in, n_samples, samples)) {
            fclose(in);
            in = nullptr;
            out = fopen(argv[2], "wb");
            rc = out ? run<double>(h, hidden, init, samples, out) : 6;
        }
    } else {
        fprintf(stderr, "dtype must be 4 or 8, got %u\n", h.dtype);
    }

    if (in) fclose(in);
    if (out) fclose(out);
    return rc;
}
