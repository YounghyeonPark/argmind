# SPDX-License-Identifier: Apache-2.0
"""`Stack` — a hierarchy that is itself an `Adapter`, so everything above it works unchanged.

The 2026-08-16 decision was that **learning applies to all parts** and that low-level through
high-level cognition must be expressible. The re-audit that followed found one real constraint: an
`Adapter` reported nothing a level below it could descend on. `Adapter.input_gradient()` closed
that. This is what uses it.

**What it is.** A chain of *layers* below a *head*. The head is any `Adapter`; the layers implement
a small protocol the `Backbone` already satisfied in substance:

    features(image) -> vector
    backward(image, d_features) -> (d_weights, d_biases)
    apply_gradients(d_weights, d_biases, learning_rate)
    state / load_state(flat) / state_size / in_shape / n_outputs

**Why it is an `Adapter` rather than a new kind of object.** `Guard`, `Pipeline` and `persist` are
all written against that contract, and this project's own standard is that *if the guard needed one
change to accommodate a hierarchy, the contract would not be one*. Being an `Adapter` means one flat
state block covers every level, so a checkpoint checkpoints the layers, a rollback rolls back the
layers, a freeze freezes the layers, and the persistence record carries them — none of which is true
today, where the backbone sits outside the guard entirely and the chain rule runs in 42 lines of
`demo/pointer.py`.

**The input is flat, deliberately.** `update(x, y)` takes a 1-D `x` and reshapes its first
`prod(in_shape)` entries into the image, because `FrameMap` builds a 1-D regressor and a `Pipeline`
would otherwise be unable to feed a stack at all. Any remaining entries are **extras** — the demo's
memory pyramid and constant — and they bypass the layers and are concatenated on to the features the
head sees. That split is exactly the `slice [:144]` in `docs/ARCHITECTURE.md`: the extras are not the
layers' output, so the head's gradient with respect to them is not theirs to receive.

**`state` is a snapshot, and that differs from every other adapter here.** The components own their
own storage, so the block is refreshed from them when it is read rather than being the live memory.
That is a real divergence and it is pinned by a test rather than described: a caller mutating the
returned array changes nothing. It is safe for the paths that matter — a `Guard` copies out of it at
each checkpoint and calls `load_state` to restore — and it costs one copy per read, not per update,
which is why it is preferred to refreshing after every step.
"""

from __future__ import annotations

import numpy as np

from .adapt import Adapter, UpdateReport

__all__ = ["Stack"]


class Stack(Adapter):
    """Layers below a head, presented as one `Adapter`.

    `feature_lr` is the step the layers take. It is separate from the head's own rate because the
    two are not interchangeable: this repository measured that applying a calibration scale without
    dividing the feature rate by it costs 31.0x -> 1.8x, since the scale multiplies the backward
    pass as well as the forward one.
    """

    def __init__(self, layers, head: Adapter, *, feature_lr: float):
        self.layers = list(layers)
        if not self.layers:
            raise ValueError("a Stack with no layers is just its head; use the head directly")
        if len(self.layers) > 1:
            # Refused rather than half-built. `backward(image, d_features)` differentiates at the
            # layer's *own* input and returns only weight gradients -- no gradient with respect to
            # that input -- so there is nothing to hand the layer beneath. Chaining separate layer
            # objects needs the protocol to grow that return, which is a decision, not an omission.
            # Depth is available inside one layer: a `Backbone` is already a multi-layer stack.
            raise ValueError(
                f"a Stack takes one layer, got {len(self.layers)}. The layer protocol returns no "
                "gradient with respect to its input, so layers cannot be chained; put the depth "
                "inside the layer instead (a Backbone plan is itself a stack)")
        self.head = head
        self.feature_lr = float(feature_lr)

        self.in_shape = tuple(self.layers[0].in_shape)
        self._image_size = int(np.prod(self.in_shape))

        width = self.layers[-1].n_outputs
        self.n_extra = int(head.n_features) - width
        if self.n_extra < 0:
            raise ValueError(
                f"the head takes {head.n_features} features but the layers emit {width}; a head "
                "narrower than its input cannot be fed")

        self.n_features = self._image_size + self.n_extra
        self.n_outputs = int(head.n_outputs)
        self.dtype = np.dtype(head.dtype)

        # `state.size` rather than `state_size`, because that attribute is **not** on every
        # adapter: `NormalisedLeastMeanSquares` has none, while RLS and the network do. CLAUDE.md's
        # statement of the contract lists it; the Python `Adapter` ABC does not require it. Using
        # the array's own size works for all three and for any layer.
        sizes = [int(np.asarray(l.state).size) for l in self.layers]                 + [int(np.asarray(head.state).size)]
        self._bounds = np.cumsum([0] + sizes)
        self.state_size = int(self._bounds[-1])

        #: Refreshed on read, not live. See the module docstring.
        self._block = np.zeros(self.state_size, dtype=self.dtype)
        #: The head's input: layer features followed by the extras, in one preallocated buffer.
        self._h = np.zeros(int(head.n_features), dtype=self.dtype)
        self._image = np.zeros(self.in_shape, dtype=self.layers[0].dtype)
        self._initial = self._gather().copy()

    # ---- state -----------------------------------------------------------------------------

    def _gather(self) -> np.ndarray:
        parts = [l.state for l in self.layers] + [self.head.state]
        for k, part in enumerate(parts):
            self._block[self._bounds[k]:self._bounds[k + 1]] = np.asarray(part).ravel()
        return self._block

    @property
    def state(self) -> np.ndarray:
        return self._gather()

    def load_state(self, flat) -> None:
        arr = np.asarray(flat).ravel()
        if arr.size != self.state_size:
            raise ValueError(f"state is {self.state_size} values, got {arr.size}")
        for k, part in enumerate(self.layers + [self.head]):
            part.load_state(arr[self._bounds[k]:self._bounds[k + 1]])

    def reset(self) -> None:
        """Back to the state the stack was constructed in — layers included.

        Snapshotted at construction rather than re-derived, because the layer protocol has no
        `reset()` and a seeded initialisation is not reproducible from the protocol alone.
        """
        self.load_state(self._initial)

    # ---- forward ---------------------------------------------------------------------------

    def _forward(self, xa: np.ndarray) -> np.ndarray:
        self._image[...] = xa[:self._image_size].reshape(self.in_shape)
        out = self._image
        for layer in self.layers:
            out = layer.features(out)
        self._h[:out.size] = out
        if self.n_extra:
            self._h[out.size:] = xa[self._image_size:]
        return self._h

    def predict(self, x) -> np.ndarray:
        return self.head.predict(self._forward(np.asarray(x, dtype=self.dtype).ravel()))

    # ---- update ----------------------------------------------------------------------------

    def update(self, x, y) -> UpdateReport:
        """Forward through the layers, update the head, then descend the layers on what it asks for.

        The layers move **only when the head's step was applied**. A rejected sample leaves
        `input_gradient()` zero, so this would be a no-op anyway; the explicit check is here because
        a guard that freezes the head must not leave the layers still learning — a fault is
        precisely when you do not want it pushed into the representation.
        """
        xa = np.asarray(x, dtype=self.dtype).ravel()
        if xa.size != self.n_features:
            raise ValueError(f"features: expected shape ({self.n_features},), got ({xa.size},)")

        report = self.head.update(self._forward(xa), y)
        if report.applied:
            d_h = np.asarray(self.head.input_gradient())
            d_feat = d_h[:self.layers[-1].n_outputs]
            self._descend(d_feat)
        return report

    def _descend(self, d_feat: np.ndarray) -> None:
        """One backward sweep, last layer first.

        One layer, because more than one is refused at construction and the reason is there.
        """
        layer = self.layers[0]
        dws, dbs = layer.backward(self._image, d_feat.astype(layer.dtype))
        layer.apply_gradients(dws, dbs, self.feature_lr)

    # ---- the rest of the contract ----------------------------------------------------------

    @property
    def health_fields(self):
        return self.head.health_fields

    def health(self) -> tuple[float, float]:
        return self.head.health()

    def input_gradient(self) -> np.ndarray:
        raise NotImplementedError(
            "a Stack's input is the raw image and the layer protocol does not return a gradient "
            "with respect to it, so nothing can sit below a Stack")

    @property
    def updates(self) -> int:
        return self.head.updates
