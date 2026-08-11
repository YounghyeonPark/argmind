// SPDX-License-Identifier: Apache-2.0
//
// A frozen convolutional feature extractor -- the platform's own backbone, ported.
//
// Why it is in the hosted tier and not the core
// ---------------------------------------------
// Arithmetic, decided before either implementation existed. MobileNet v1 alpha=0.25 at 96x96
// holds two adjacent activation buffers at once and the largest such pair is 54 KB in int8,
// **216 KB in fp32**, against a declared core floor of 8 KB for the whole library. At 224x224
// it is 294 KB. Nothing about the implementation moves that; it is the shape of the data.
//
// What it is
// ----------
// `features()` and nothing else. A backbone here is FROZEN: it produces the regressor that an
// `Adapter` under a `Guard` then adapts on top of. Training a convolutional stack on-device
// was measured unavailable -- backpropagation retains activations, not weights, and the peak
// above is the forward pass alone.
//
// Layout, matching the reference exactly because a parity harness compares blocks
// -------------------------------------------------------------------------------
// Activations are HWC. Weights are `(out, kh, kw, in)` for a regular convolution and
// `(kh, kw, channels)` for a depthwise one -- TFLite's layouts, so a converted model
// transfers without a permutation. All of it in one flat contiguous block in layer order,
// each layer's weights then its biases, which is the promise every adapter here makes and is
// what lets `persist` carry a backbone with no special case.
//
// The one divergence, and it is the declared one
// ----------------------------------------------
// numpy dispatches its convolution to a BLAS matmul and this is ordered loops, so reduction
// order differs -- divergence 3, which every ported component here carries. It is larger than
// elsewhere for a structural reason worth stating: a pointwise layer at 512 channels sums 512
// terms per output and a 3x3 regular convolution over 256 channels sums 2,304, where an RLS
// update at six features sums six.

#ifndef ARGMIND_HOST_RUNTIME_SHAPE_BACKBONE_HPP
#define ARGMIND_HOST_RUNTIME_SHAPE_BACKBONE_HPP

#include <stdint.h>

#include <algorithm>
#include <vector>

#include "argmind/mat.hpp"

namespace argmind {
namespace host {

enum class LayerKind : uint8_t { Conv = 0, Depthwise = 1, Pointwise = 2 };

struct LayerSpec {
    LayerKind kind;
    int out_channels;   ///< ignored for Depthwise, which never changes the channel count
    int k;              ///< kernel size; 1 for Pointwise
    int stride;
};

template <typename T>
class RuntimeShapedBackbone {
    static_assert(sizeof(T) >= 4, "fp16 is unmeasured here, and int8 is a different scheme");

public:
    using scalar = T;

    /// `pool` tiles the final feature map into `pool x pool` averages instead of one.
    /// `pool = 1` is global average pooling, which discards WHERE anything was --
    /// measured on a shallow random backbone, a blob's position reads 1.076 of its own
    /// standard deviation globally (worse than a constant) and 0.231 on a 3x3 grid.
    RuntimeShapedBackbone(const std::vector<LayerSpec>& plan, int height, int width,
                          int channels, int pool = 1)
        : plan_(plan), h_(height), w_(width), c_in_(channels), pool_(pool) {
        if (pool < 1) valid_ = false;
        valid_ = height >= 1 && width >= 1 && channels >= 1;
        int h = height, w = width, c = channels;
        size_t off = 0;
        size_t widest = static_cast<size_t>(h) * w * c;
        for (const LayerSpec& L : plan_) {
            const int k = (L.kind == LayerKind::Pointwise) ? 1 : L.k;
            const int s = (L.kind == LayerKind::Pointwise) ? 1 : L.stride;
            if (k < 1 || s < 1) valid_ = false;
            size_t wn, bn;
            if (L.kind == LayerKind::Depthwise) {
                wn = static_cast<size_t>(k) * k * c;
                bn = static_cast<size_t>(c);
            } else {
                if (L.out_channels < 1) valid_ = false;
                wn = static_cast<size_t>(L.out_channels) * k * k * c;
                bn = static_cast<size_t>(L.out_channels);
            }
            offsets_.push_back(off);
            off += wn + bn;
            h = (h + s - 1) / s;
            w = (w + s - 1) / s;
            if (L.kind != LayerKind::Depthwise) c = L.out_channels;
            widest = std::max(widest, static_cast<size_t>(h) * w * c);
        }
        channels_out_ = c;
        n_outputs_ = c * pool_ * pool_;
        if (h < pool_ || w < pool_) valid_ = false;
        state_.assign(off, T(0));
        // Two buffers, ping-ponged, each the size of the largest single activation. Never
        // smaller than the true peak pair and one line instead of a schedule.
        a_.assign(widest, T(0));
        b_.assign(widest, T(0));
    }

    bool valid() const { return valid_; }
    int n_outputs() const { return n_outputs_; }
    /// The flattened input, so a `persist` record describes both sides of the model.
    int n_features() const { return h_ * w_ * c_in_; }
    int state_size() const { return static_cast<int>(state_.size()); }
    size_t activation_values() const { return a_.size() + b_.size(); }

    const T* state() const { return state_.data(); }
    T* state_mutable() { return state_.data(); }

    bool load_state(const T* flat) {
        for (size_t i = 0; i < state_.size(); ++i) {
            if (!is_finite(flat[i])) return false;
        }
        for (size_t i = 0; i < state_.size(); ++i) state_[i] = flat[i];
        return true;
    }

    /// One image in, one feature vector out. Nothing here changes state: this is frozen.
    void features(const T* image, T* out) {
        int h = h_, w = w_, c = c_in_;
        T* src = a_.data();
        T* dst = b_.data();
        const size_t n_in = static_cast<size_t>(h) * w * c;
        for (size_t i = 0; i < n_in; ++i) src[i] = image[i];

        for (size_t li = 0; li < plan_.size(); ++li) {
            const LayerSpec& L = plan_[li];
            const T* wgt = state_.data() + offsets_[li];
            if (L.kind == LayerKind::Depthwise) {
                const int k = L.k, s = L.stride;
                const int oh = (h + s - 1) / s, ow = (w + s - 1) / s;
                depthwise(src, h, w, c, wgt, wgt + static_cast<size_t>(k) * k * c, k, s,
                          dst, oh, ow);
                h = oh;
                w = ow;
            } else {
                const int k = (L.kind == LayerKind::Pointwise) ? 1 : L.k;
                const int s = (L.kind == LayerKind::Pointwise) ? 1 : L.stride;
                const int co = L.out_channels;
                const int oh = (h + s - 1) / s, ow = (w + s - 1) / s;
                conv(src, h, w, c, wgt,
                     wgt + static_cast<size_t>(co) * k * k * c, co, k, s, dst, oh, ow);
                h = oh;
                w = ow;
                c = co;
            }
            T* t = src;
            src = dst;
            dst = t;
        }

        // The layer that turns a feature MAP into the feature VECTOR the rest of this
        // library consumes. Tiles get `h / pool` rows with the remainder on the LAST one,
        // matching the reference -- a difference there would move every feature.
        int o = 0;
        for (int gi = 0; gi < pool_; ++gi) {
            const int i0 = gi * h / pool_;
            const int i1 = (gi + 1 < pool_) ? (gi + 1) * h / pool_ : h;
            for (int gj = 0; gj < pool_; ++gj) {
                const int j0 = gj * w / pool_;
                const int j1 = (gj + 1 < pool_) ? (gj + 1) * w / pool_ : w;
                const T inv = T(1) / static_cast<T>((i1 - i0) * (j1 - j0));
                for (int ch = 0; ch < c; ++ch) {
                    T acc = T(0);
                    for (int i = i0; i < i1; ++i) {
                        for (int j = j0; j < j1; ++j)
                            acc += src[(static_cast<size_t>(i) * w + j) * c + ch];
                    }
                    out[o++] = acc * inv * scale_;
                }
            }
        }
    }

    void set_output_scale(T s) { scale_ = s; }
    T output_scale() const { return scale_; }

private:
    /// TensorFlow's SAME padding. When the total is odd the extra row goes at the BOTTOM --
    /// getting that backwards shifts every feature by a pixel and still produces a network
    /// that looks plausible, which is why the reference pins it with an impulse.
    static void pad_of(int in, int k, int s, int* lo, int* total) {
        const int out = (in + s - 1) / s;
        const int p = (out - 1) * s + k - in;
        *total = p > 0 ? p : 0;
        *lo = *total / 2;
    }

    void conv(const T* x, int h, int w, int c, const T* wgt, const T* bias, int co, int k,
              int s, T* out, int oh, int ow) const {
        int top, ph, left, pw;
        pad_of(h, k, s, &top, &ph);
        pad_of(w, k, s, &left, &pw);
        for (int oi = 0; oi < oh; ++oi) {
            for (int oj = 0; oj < ow; ++oj) {
                for (int o = 0; o < co; ++o) {
                    T acc = T(0);
                    for (int di = 0; di < k; ++di) {
                        const int ii = oi * s + di - top;
                        if (ii < 0 || ii >= h) continue;
                        for (int dj = 0; dj < k; ++dj) {
                            const int jj = oj * s + dj - left;
                            if (jj < 0 || jj >= w) continue;
                            const T* xp = x + (static_cast<size_t>(ii) * w + jj) * c;
                            const T* wp = wgt + ((static_cast<size_t>(o) * k + di) * k + dj) * c;
                            for (int ci = 0; ci < c; ++ci) acc += xp[ci] * wp[ci];
                        }
                    }
                    out[(static_cast<size_t>(oi) * ow + oj) * co + o] = relu6(acc + bias[o]);
                }
            }
        }
    }

    void depthwise(const T* x, int h, int w, int c, const T* wgt, const T* bias, int k, int s,
                   T* out, int oh, int ow) const {
        int top, ph, left, pw;
        pad_of(h, k, s, &top, &ph);
        pad_of(w, k, s, &left, &pw);
        for (int oi = 0; oi < oh; ++oi) {
            for (int oj = 0; oj < ow; ++oj) {
                for (int ch = 0; ch < c; ++ch) {
                    T acc = T(0);
                    for (int di = 0; di < k; ++di) {
                        const int ii = oi * s + di - top;
                        if (ii < 0 || ii >= h) continue;
                        for (int dj = 0; dj < k; ++dj) {
                            const int jj = oj * s + dj - left;
                            if (jj < 0 || jj >= w) continue;
                            acc += x[(static_cast<size_t>(ii) * w + jj) * c + ch] *
                                   wgt[(static_cast<size_t>(di) * k + dj) * c + ch];
                        }
                    }
                    out[(static_cast<size_t>(oi) * ow + oj) * c + ch] = relu6(acc + bias[ch]);
                }
            }
        }
    }

    /// Clipped at 6, which bounds the activation range so a quantised version has a known
    /// scale. Keeping it in the fp32 path is what makes an int8 one possible later.
    static T relu6(T v) { return v < T(0) ? T(0) : (v > T(6) ? T(6) : v); }

    std::vector<LayerSpec> plan_;
    std::vector<size_t> offsets_;
    std::vector<T> state_;
    std::vector<T> a_, b_;
    int h_, w_, c_in_;
    int pool_ = 1;
    int channels_out_ = 0;
    int n_outputs_ = 0;
    T scale_ = T(1);
    bool valid_ = true;
};

}  // namespace host
}  // namespace argmind

#endif  // ARGMIND_HOST_RUNTIME_SHAPE_BACKBONE_HPP
