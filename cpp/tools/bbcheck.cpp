// SPDX-License-Identifier: Apache-2.0
//
// Parity for `RuntimeShapedBackbone`, against `argmind.backbone.Backbone`.
//
// The case format is its own, for the reason `mlpcheck`'s was: a backbone's configuration is
// a variable-length layer plan plus a weight block plus an image, and none of the existing
// headers carry any of that. The RESULT is just the embedding, so there is no row format to
// share -- what `mlpcheck` reuses from `xcheck` does not exist here.
//
// Weights come from the reference, as they do for `mlpcheck` and for the same reason: the two
// initialisations do not match and a generator that nearly matched would be worse than one
// that plainly does not. A device restoring a trained model takes the same path.

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <vector>

#include "argmind/host/runtime_shape_backbone.hpp"

namespace {

constexpr uint16_t kFormatVersion = 1;

struct CaseHeader {
    char magic[4];      // "AMBB"
    uint16_t version;
    uint8_t dtype;      // 4 or 8
    uint8_t pad;
    uint16_t height, width, channels;
    uint16_t n_layers;
    uint32_t state_size;
    double output_scale;
};
static_assert(sizeof(CaseHeader) == 32, "the Python encoder packs exactly this");

struct LayerRec {
    uint8_t kind;       // 0 conv, 1 depthwise, 2 pointwise
    uint8_t pad[3];
    int32_t out_channels;
    int32_t k;
    int32_t stride;
};
static_assert(sizeof(LayerRec) == 16, "the Python encoder packs exactly this");

template <typename T>
int run(const CaseHeader& h, const std::vector<LayerRec>& layers,
        const std::vector<T>& weights, const std::vector<T>& image, FILE* out) {
    std::vector<argmind::host::LayerSpec> plan;
    for (const LayerRec& L : layers) {
        plan.push_back({static_cast<argmind::host::LayerKind>(L.kind),
                        L.out_channels, L.k, L.stride});
    }
    argmind::host::RuntimeShapedBackbone<T> bb(plan, h.height, h.width, h.channels);
    if (!bb.valid()) {
        fprintf(stderr, "the plan was refused\n");
        return 4;
    }
    if (bb.state_size() != static_cast<int>(h.state_size)) {
        fprintf(stderr, "state size %d against the reference's %u\n", bb.state_size(),
                h.state_size);
        return 5;
    }
    if (!bb.load_state(weights.data())) {
        fprintf(stderr, "the reference's weights were refused\n");
        return 6;
    }
    bb.set_output_scale(static_cast<T>(h.output_scale));

    std::vector<T> emb(static_cast<size_t>(bb.n_outputs()));
    bb.features(image.data(), emb.data());

    const uint32_t n = static_cast<uint32_t>(emb.size());
    if (fwrite(&n, sizeof n, 1, out) != 1) return 7;
    if (fwrite(emb.data(), sizeof(T), n, out) != n) return 7;
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
        fprintf(stderr, "usage: bbcheck <case> <result>\n");
        return 1;
    }
    FILE* in = fopen(argv[1], "rb");
    if (!in) {
        fprintf(stderr, "cannot open %s\n", argv[1]);
        return 2;
    }
    CaseHeader h{};
    if (fread(&h, sizeof h, 1, in) != 1 || memcmp(h.magic, "AMBB", 4) != 0 ||
        h.version != kFormatVersion) {
        fprintf(stderr, "not a bbcheck case file\n");
        fclose(in);
        return 3;
    }
    std::vector<LayerRec> layers(h.n_layers);
    if (h.n_layers &&
        fread(layers.data(), sizeof(LayerRec), h.n_layers, in) != h.n_layers) {
        fclose(in);
        return 3;
    }

    const size_t n_img = static_cast<size_t>(h.height) * h.width * h.channels;
    int rc = 3;
    FILE* out = nullptr;
    if (h.dtype == 4) {
        std::vector<float> w, img;
        if (read_vec(in, h.state_size, w) && read_vec(in, n_img, img)) {
            fclose(in);
            in = nullptr;
            out = fopen(argv[2], "wb");
            rc = out ? run<float>(h, layers, w, img, out) : 7;
        }
    } else if (h.dtype == 8) {
        std::vector<double> w, img;
        if (read_vec(in, h.state_size, w) && read_vec(in, n_img, img)) {
            fclose(in);
            in = nullptr;
            out = fopen(argv[2], "wb");
            rc = out ? run<double>(h, layers, w, img, out) : 7;
        }
    } else {
        fprintf(stderr, "dtype must be 4 or 8, got %u\n", h.dtype);
    }
    if (in) fclose(in);
    if (out) fclose(out);
    return rc;
}
