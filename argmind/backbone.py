# SPDX-License-Identifier: Apache-2.0
"""A frozen convolutional feature extractor — the platform's own backbone.

Why this is in the library
--------------------------
The scope decision of 2026-08-10 is that **generating the basis is the platform's job**, and
that the platform needs its own inference network. `MultilayerPerceptron` made the first half
real for sensor scalars. This is the half that reads an image.

An earlier answer said a camera backbone belongs to the vendor's runtime and argmind should
take its output. That narrowed the decision on no authority, and it conflated two things:

- **Using an NPU** does require the vendor toolchain. The Neural-ART's instruction encoding is
  proprietary and its weights must go through ST's compiler. That is a property of the
  silicon, not a preference.
- **Implementing a convolution** requires nothing. This module is the demonstration.

So there are two paths to the same feature vector and the library supports both: run the
backbone here, or let a vendor runtime run one and push its embedding into a `Stream` like any
other sensor. The seam is the vector, and the rest of the library — aligner, frame map, guard,
persist — does not care which side produced it.

Which tier, decided by arithmetic before any code
--------------------------------------------------
It cannot be the core. The **smallest useful** camera backbone, MobileNet v1 α=0.25 at 96×96,
holds two adjacent activation buffers at once and the largest such pair is **54 KB in int8,
216 KB in fp32** — against a declared core floor of **8 KB for the whole library**. At 224×224
it is 294 KB and 1.15 MB. Nothing about the implementation changes that; it is the shape of
the data. This is a second-tier module, like `MultilayerPerceptron` and for the same kind of
reason.

What this is not
----------------
**It does not learn.** A backbone here is frozen: `features()` and nothing else. Training a
CNN on-device was measured to be unavailable — backpropagation retains *activations*, not
weights, and the peak above is already the forward pass alone. What adapts is the head, which
is what the rest of this library is.

**It is fp32, and int8 is deliberately absent rather than forgotten.** Quantised inference is
not "the same network in smaller types": the arithmetic is defined by a scheme (zero points,
per-axis scales, accumulator width, requantisation rounding) and two implementations that
disagree on any of it are two different models. Adding it means matching a scheme exactly and
proving it, which is its own piece of work.

Layout
------
Activations are **HWC** and weights are **(out, kh, kw, in)** for a regular convolution and
**(kh, kw, channels)** for a depthwise one — TFLite's layouts, so a converted model's weights
transfer without a permutation. Feature order is parameter order, the same rule `FrameMap`
states, and a backbone's embedding restored against a different order is wrong undetectably.

Weights live in **one flat contiguous block**, the same promise every adapter here makes, so
`persist` carries a backbone with no special case and the record format's fingerprint refuses
one saved under a different architecture.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["conv2d", "depthwise_conv2d", "relu6", "global_avg_pool",
           "conv2d_naive", "depthwise_conv2d_naive", "Backbone", "mobilenet_v1"]


def _pad_same(x, kh, kw, stride):
    """TensorFlow's SAME padding: the total is `max(k - stride, 0)` when the size divides,
    and the extra row or column goes at the BOTTOM/RIGHT. Getting that asymmetry wrong shifts
    every feature by a pixel and still produces a plausible-looking network."""
    h, w = x.shape[0], x.shape[1]
    out_h, out_w = (h + stride - 1) // stride, (w + stride - 1) // stride
    ph = max((out_h - 1) * stride + kh - h, 0)
    pw = max((out_w - 1) * stride + kw - w, 0)
    top, left = ph // 2, pw // 2
    return np.pad(x, ((top, ph - top), (left, pw - left), (0, 0)))


def conv2d(x, w, b=None, stride=1):
    """`x` is (H, W, Cin), `w` is (Cout, kh, kw, Cin). SAME padding, HWC out.

    Implemented by im2col and one matmul: the same arithmetic as the definition, in an order
    a BLAS can use. `conv2d_naive` is the definition, and the tests hold them together.
    """
    x = np.asarray(x)
    w = np.asarray(w)
    cout, kh, kw, cin = w.shape
    if x.shape[2] != cin:
        raise ValueError(f"filter expects {cin} input channels, got {x.shape[2]}")
    p = _pad_same(x, kh, kw, stride)
    oh = (x.shape[0] + stride - 1) // stride
    ow = (x.shape[1] + stride - 1) // stride

    cols = np.empty((oh * ow, kh * kw * cin), dtype=x.dtype)
    n = 0
    for i in range(oh):
        for j in range(ow):
            cols[n] = p[i * stride:i * stride + kh, j * stride:j * stride + kw, :].ravel()
            n += 1
    out = cols @ w.reshape(cout, -1).T
    if b is not None:
        out += np.asarray(b)
    return out.reshape(oh, ow, cout)


def depthwise_conv2d(x, w, b=None, stride=1):
    """One filter per input channel, no mixing. `w` is (kh, kw, C).

    This is where a MobileNet's cost goes from `k*k*Cin*Cout` to `k*k*Cin`, and it is the
    reason a convolutional backbone is reachable on a CPU at all.
    """
    x = np.asarray(x)
    w = np.asarray(w)
    kh, kw, c = w.shape
    if x.shape[2] != c:
        raise ValueError(f"depthwise filter expects {c} channels, got {x.shape[2]}")
    p = _pad_same(x, kh, kw, stride)
    oh = (x.shape[0] + stride - 1) // stride
    ow = (x.shape[1] + stride - 1) // stride
    out = np.zeros((oh, ow, c), dtype=x.dtype)
    for di in range(kh):
        for dj in range(kw):
            patch = p[di:di + (oh - 1) * stride + 1:stride,
                      dj:dj + (ow - 1) * stride + 1:stride, :]
            out += patch * w[di, dj, :]
    if b is not None:
        out += np.asarray(b)
    return out


def conv2d_naive(x, w, b=None, stride=1):
    """The definition, four loops deep. Slow on purpose: it exists to be the thing `conv2d`
    is checked against, and a fast reference that shares an implementation trick with the
    thing it checks is not a reference."""
    x, w = np.asarray(x, dtype=np.float64), np.asarray(w, dtype=np.float64)
    cout, kh, kw, cin = w.shape
    p = _pad_same(x, kh, kw, stride)
    oh = (x.shape[0] + stride - 1) // stride
    ow = (x.shape[1] + stride - 1) // stride
    out = np.zeros((oh, ow, cout))
    for i in range(oh):
        for j in range(ow):
            for o in range(cout):
                acc = 0.0
                for di in range(kh):
                    for dj in range(kw):
                        for ci in range(cin):
                            acc += p[i * stride + di, j * stride + dj, ci] * w[o, di, dj, ci]
                out[i, j, o] = acc + (0.0 if b is None else b[o])
    return out


def depthwise_conv2d_naive(x, w, b=None, stride=1):
    """The definition again, for the depthwise case."""
    x, w = np.asarray(x, dtype=np.float64), np.asarray(w, dtype=np.float64)
    kh, kw, c = w.shape
    p = _pad_same(x, kh, kw, stride)
    oh = (x.shape[0] + stride - 1) // stride
    ow = (x.shape[1] + stride - 1) // stride
    out = np.zeros((oh, ow, c))
    for i in range(oh):
        for j in range(ow):
            for ch in range(c):
                acc = 0.0
                for di in range(kh):
                    for dj in range(kw):
                        acc += p[i * stride + di, j * stride + dj, ch] * w[di, dj, ch]
                out[i, j, ch] = acc + (0.0 if b is None else b[ch])
    return out


def relu6(x):
    """Clipped at 6, which is not decoration: it bounds the activation range so a quantised
    version has a known scale. Keeping it in the fp32 reference is what makes the int8 one
    possible later."""
    return np.clip(x, 0.0, 6.0)


def global_avg_pool(x):
    """(H, W, C) to (C,). This is the layer that turns a feature MAP into a feature VECTOR,
    which is the thing the rest of this library consumes."""
    return np.asarray(x).mean(axis=(0, 1))


class Backbone:
    """A frozen stack of layers that turns an image into a feature vector.

    The layer list is declared, not built by a callable — the same reason `FrameMap` takes an
    index table: a callable does not cross a C ABI, and this is meant to port.

    Each entry is `("conv", cout, k, stride)` or `("dw", k, stride)` or `("pw", cout)`.
    `("pw", c)` is `("conv", c, 1, 1)` and is spelled separately because a depthwise-separable
    block is the unit a reader thinks in.
    """

    __slots__ = ("plan", "in_shape", "dtype", "_shapes", "_state", "_state_ro", "_w", "_b",
                 "n_features", "n_outputs", "output_scale")

    def __init__(self, plan, in_shape=(96, 96, 3), dtype="f4", seed=0,
                 output_scale=1.0):
        self.plan = tuple(tuple(p) for p in plan)
        self.in_shape = tuple(int(v) for v in in_shape)
        self.dtype = np.dtype(dtype)
        if not output_scale > 0.0 or not math.isfinite(output_scale):
            raise ValueError(
                f"output_scale must be positive and finite, got {output_scale}")
        self.output_scale = float(output_scale)
        if len(self.in_shape) != 3 or min(self.in_shape) < 1:
            raise ValueError(f"in_shape must be (H, W, C), got {self.in_shape}")

        self._shapes = []
        c = self.in_shape[2]
        for entry in self.plan:
            kind = entry[0]
            if kind == "conv":
                _, cout, k, _s = entry
                self._shapes.append(((cout, k, k, c), (cout,)))
                c = cout
            elif kind == "pw":
                _, cout = entry
                self._shapes.append(((cout, 1, 1, c), (cout,)))
                c = cout
            elif kind == "dw":
                _, k, _s = entry
                self._shapes.append((((k, k, c)), (c,)))
            else:
                raise ValueError(f"unknown layer {entry!r}")
        # `n_outputs` is the embedding width -- what this produces, and what becomes the
        # HEAD's `n_features`. `n_features` here is the flattened input, so a `persist`
        # record describes both sides of the model and a reader can check either.
        self.n_outputs = c
        self.n_features = int(np.prod(self.in_shape))

        size = sum(int(np.prod(ws)) + int(np.prod(bs)) for ws, bs in self._shapes)
        self._state = np.zeros(size, dtype=self.dtype)
        self._state_ro = self._state.view()
        self._state_ro.flags.writeable = False

        self._w, self._b, off = [], [], 0
        for ws, bs in self._shapes:
            n = int(np.prod(ws))
            self._w.append(self._state[off:off + n].reshape(ws))
            off += n
            m = int(np.prod(bs))
            self._b.append(self._state[off:off + m])
            off += m

        # Glorot again, and for the same reason as the MLP: a zero-weight backbone emits a
        # constant vector, which is a basis of rank one.
        g = np.random.default_rng(seed)
        for w, (ws, _bs) in zip(self._w, self._shapes):
            fan_in = int(np.prod(ws[1:])) if len(ws) == 4 else int(np.prod(ws[:2]))
            fan_out = ws[0] if len(ws) == 4 else 1
            w[:] = g.standard_normal(ws) * math.sqrt(2.0 / (fan_in + fan_out))

    @property
    def state(self):
        return self._state_ro

    @property
    def state_size(self):
        return int(self._state.size)

    def load_state(self, flat):
        arr = np.asarray(flat, dtype=self.dtype).reshape(-1)
        if arr.size != self._state.size:
            raise ValueError(
                f"state block must have {self._state.size} elements, got {arr.size}")
        if not np.all(np.isfinite(arr)):
            raise ValueError("refusing a state block that is not finite")
        self._state[:] = arr

    def peak_activation_values(self):
        """The largest pair of adjacent buffers, which is what a device must reserve.

        Reported rather than inferred, because it is the number that decides whether a
        configuration fits at all -- 54 KB in int8 for the smallest useful vision backbone,
        against a declared core floor of 8 KB for the whole library.
        """
        h, w, c = self.in_shape
        sizes = [h * w * c]
        for entry, (ws, _bs) in zip(self.plan, self._shapes):
            if entry[0] == "conv":
                _, cout, _k, s = entry
                h, w, c = (h + s - 1) // s, (w + s - 1) // s, cout
            elif entry[0] == "pw":
                c = entry[1]
            else:
                _, _k, s = entry
                h, w = (h + s - 1) // s, (w + s - 1) // s
            sizes.append(h * w * c)
        return max(a + b for a, b in zip(sizes[:-1], sizes[1:]))

    def features(self, image):
        """One image in, one feature vector out. No state changes -- this is frozen."""
        x = np.asarray(image, dtype=self.dtype)
        if x.shape != self.in_shape:
            raise ValueError(f"expected an image of {self.in_shape}, got {x.shape}")
        for entry, w, b in zip(self.plan, self._w, self._b):
            if entry[0] == "conv":
                x = relu6(conv2d(x, w, b, stride=entry[3]))
            elif entry[0] == "pw":
                x = relu6(conv2d(x, w, b, stride=1))
            else:
                x = relu6(depthwise_conv2d(x, w, b, stride=entry[2]))
        return global_avg_pool(x) * self.dtype.type(self.output_scale)

    def __repr__(self):
        return (f"Backbone(layers={len(self.plan)}, in_shape={self.in_shape}, "
                f"n_outputs={self.n_outputs}, weights={self.state_size}, "
                f"dtype={self.dtype.name!r}, output_scale={self.output_scale})")

    def calibrate_scale(self, images, target_std=1.0):
        """The constant `output_scale` should be, measured on representative inputs.

        **A backbone's embedding has to be scaled and the library will not guess it**, so
        this reports rather than applies. Measured on an untrained MobileNet v1 0.25 at
        96x96, whose embedding comes out at a magnitude of **6.8e-4**:

        ==================  ==============  =============
        head input          ``|x|`` median  RMSE / std
        ==================  ==============  =============
        raw                 2.2e-3          **0.998**
        L2 per vector       1.0             0.229
        divided by its std  2.1e+1          **0.0015**
        ==================  ==============  =============

        Raw fails outright, and not because RLS is fragile: **`p0` has units.** The gain is
        ``p0*x/(1 + p0*|x|^2)``, so at ``|x| ~ 1e-4`` the data term is ``1e-8 * p0`` and the
        default prior of 100 is eight orders of magnitude too small. Raising `p0` to 1e10
        recovers it exactly -- 0.996 becomes 2.6e-6 -- which says the failure is the prior
        against the input scale rather than the algorithm. Scaling the *target* is free for
        both RLS and NLMS; scaling the *input* is not, and conflating the two was a
        prediction of mine that this measurement corrected.

        **L2-normalising each vector is not offered as the answer** even though it makes
        ``|x|`` exactly 1 and conditions the prior by construction. It discards the
        embedding's magnitude, which is information: on a target that depends on it, L2
        costs 150x against a single global scale. Whether that matters is the caller's
        problem to know, which is why this returns a number instead of applying one.

        A trained backbone with batch normalisation does not have this problem at all -- that
        is what batch normalisation is for -- so this is a property of an untrained or a
        naively-converted stack rather than of convolution.
        """
        v = np.array([self.features(im) for im in images], dtype=np.float64)
        if v.ndim != 2 or len(v) < 2:
            raise ValueError("calibration needs at least two images")
        # Two different spreads, and the first version of this checked the wrong one. The
        # scale comes from the magnitude over the WHOLE block, because that is what
        # conditions a head's prior. The refusal is about variation ACROSS IMAGES, because a
        # backbone whose output does not move with its input is a basis of rank one and
        # scaling it changes nothing -- and the overall spread is non-zero even then, since
        # the features differ from each other.
        # Relative, not `> 0`. Identical rows give a std of 5e-20 rather than exactly zero,
        # because the mean subtraction rounds -- so an absolute test passes on a backbone
        # whose output never moves, which was the first version of this check.
        magnitude = float(np.abs(v).max())
        if float(v.std(axis=0).max()) <= 1e-12 * magnitude:
            raise ValueError(
                "the embedding does not vary across these images, so there is nothing to "
                "calibrate -- a backbone whose output is constant is a basis of rank one")
        spread = float(v.std())
        if not spread > 0.0:
            raise ValueError("the embedding is identically zero")
        return float(target_std) / spread * self.output_scale


def mobilenet_v1(alpha=0.25, in_shape=(96, 96, 3), dtype="f4", seed=0):
    """The depthwise-separable stack, at a width multiplier.

    α=0.25 at 96×96 is the size the MLPerf Tiny visual-wake-words task uses, and it is the
    one that matters here: **7.5 MMAC per frame**, against 568 for α=1.0 at 224×224. The
    embedding is 256 wide, which through NLMS is 1 KB of state and through RLS would be
    259 KB — the arithmetic that put NLMS in the C ABI.
    """
    w = lambda c: max(8, int(c * alpha))                       # noqa: E731
    plan = [("conv", w(32), 3, 2)]
    for cout, stride in [(64, 1), (128, 2), (128, 1), (256, 2), (256, 1), (512, 2),
                         (512, 1), (512, 1), (512, 1), (512, 1), (512, 1),
                         (1024, 2), (1024, 1)]:
        plan.append(("dw", 3, stride))
        plan.append(("pw", w(cout)))
    return Backbone(plan, in_shape=in_shape, dtype=dtype, seed=seed)
