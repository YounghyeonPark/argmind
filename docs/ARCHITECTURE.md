# The learning stack

Every shape and count below was read out of the running code, and **`tests/test_architecture_doc.py`
recomputes all of them and fails if this file disagrees** — so a stale figure here breaks the suite
rather than sitting quietly. That gate exists because a hand-copied table in this repository has
gone stale three times, and because `cpp/target/SIZES.md` once told readers to run a regenerator
that did not exist: a document promising a tool is the same failure arriving by another door.

There is no generator. The prose carries reasoning that a generator would flatten, so the numbers
are *checked* rather than *written* — which is the weaker half of what `SIZES.md` does, and is said
plainly rather than implied.

Plain ASCII on purpose. It renders in a terminal, in VS Code, on GitHub and in a diff, and needs no
extension installed.


## 1. The forward stack

A 72x96 colour frame becomes 144 numbers through three convolutional layers and a grid pool.
**Nothing is pretrained** — the weights are seeded and everything after that is learned from the
input stream.

```
   FORWARD                                    TENSOR              WEIGHTS
   ------------------------------------------ ------------------- --------
   camera frame                               72 x 96 x 3               --
   resized, scaled to unit magnitude
        |
        v
  +==========================================+
  | conv 12 @ 5x5, stride 4     SAME, relu6  | 18 x 24 x 12             912
  +==========================================+
        |
        v
  +==========================================+
  | depthwise 3x3, stride 2                  |  9 x 12 x 12             120
  | one kernel per channel, no mixing        |
  +==========================================+
        |
        v
  +==========================================+
  | pointwise -> 16             1x1, mixes   |  9 x 12 x 16             208
  +==========================================+
        |
        v
  +------------------------------------------+
  | grid_avg_pool 3x3                        | 3*3*16 = 144               0
  | a grid, not global -- global discards    |
  | *where*, measured 1.076 vs 0.231         |
  +------------------------------------------+
        |
        v
  +- - - - - - - - - - - - - - - - - - - - - +
  | basis = concat                           |          295               0
  | 144 frame + 150 memory + 1 constant      |
  + - - - - - - - - - - - - - - - - - - - - -+
        |
        v
  +##########################################+
  | head -- ONE LINEAR MAP                   |   295 -> 108          31,860
  | yhat = x'theta, under a Guard            |
  +##########################################+
        |
        v
   segmentation map out                       9 x 12 = 108              --
   no slots to permute, nothing to count
```

`+===+` and `+###+` hold weights; `+---+` is a pure function of its input; `+- - -+` is assembly.

**Peak activation cost is 25,920 values** — that figure, not the 1,240 weights, is what a target
tier has to hold. It is reported by `Backbone.peak_activation_values()` rather than left in prose.


## 2. The backward edge, added 2026-08-16

Learning the head needs only `update(x, y)`. Learning the **layers** needs something the head can
hand back. Until this change no adapter reported anything a level below it could descend on, so the
demo reached inside the head and did the algebra itself.

```
  BEFORE -- the demo knew the head's algebra
  ------------------------------------------------------------------------------
   +------------+  reads theta  +- - - - - - - - - - - - -+     +--------------+
   |    head    | ------------> | demo: 2*theta[:-1]@resid| --> |   backbone   |
   | theta, res |               | 42 lines                |     | backward     |
   +------------+               | LINEAR HEADS ONLY       |     | apply        |
                                | never checked           |     +--------------+
                                +- - - - - - - - - - - - -+

  AFTER -- the head reports what it wants of its input
  ------------------------------------------------------------------------------
   +------------+               +=========================+     +==============+
   |    head    | ============> |      slice [:144]       | ==> |   backbone   |
   | ANY        | input_grad()  | memory + constant are   | d_f | backward     |
   | ALGORITHM  |  295 values   | not the backbone's      | 144 | apply        |
   +------------+               +=========================+     +==============+
```

**The edge is the same shape. What moved is who owns the algebra** — and with it, whether the head
can be anything but a linear map.

The definition, which is where the choices are:

```
   input_gradient()  =  d ||y - predict(x)||^2 / dx

   * at the parameters as they stood BEFORE the update, so every level in a stack
     differentiates at the same point -- which is what ordinary backprop does
   * ZERO when the update was refused: a level below must not descend on a step
     its own head declined to take
   * a PREALLOCATED buffer, overwritten by the next update() -- the same contract,
     and the same trap, as StepReport.prediction
```

Three choices, each of which could have gone the other way:

| | chosen | rejected |
|---|---|---|
| carrier | a **method** | a field on `UpdateReport` — the report crosses the C ABI as a struct of scalars and three parity harnesses compare it byte for byte; an array field would change the record format for every adapter, including the two that will never stack |
| evaluation point | **pre-update** parameters | post-update — differentiating each level at a different point is not backprop |
| return | the **preallocated buffer** | a fresh array — one allocation per step on the hot path |
| on refusal | **zero** | leave it — the level below follows a step the head refused |


## 3. Where the parameters are

| component | learns? | points | map |
|---|---|---|---|
| conv 12@5x5 s4 | yes, since 2026-08-16 | 912 | 912 |
| depthwise 3x3 s2 | yes | 120 | 120 |
| pointwise -> 16 | yes | 208 | 208 |
| head theta | yes, every sample | 870 | 31,860 |
| **total** | | **2,110** | **33,100** |
| head shape | | 145 -> 6 | 295 -> 108 |

**A points head is 41% of its stack's parameters; a map head is 96%.** The backbone is the same
1,240 weights either way — so what the tier arithmetic turns on is the head's *algorithm*, not the
backbone's depth. That is this repository's recorded "the wall is RLS's n²" showing up in the
structure rather than in a sentence.


## 4. The hierarchy: `Stack` is itself an `Adapter`

Built 2026-08-17. The claim is **not** that it learns — it is that nothing above it changed.

```
  BEFORE -- the backbone sat outside the guard entirely
  ------------------------------------------------------------------------------
   +------------+       +==============+       two state blocks. One is
   |  Backbone  |       |    Guard     |       checkpointed, rolled back and
   |            |       | wraps the    |       persisted; the other is not.
   | (no Guard) |       | HEAD ONLY    |       The chain rule is caller code.
   +------------+       +==============+

  NOW -- one Adapter, one state block, every level inside it
  ------------------------------------------------------------------------------
   +=========================================+     +==================+
   | Stack([layer], head, feature_lr=...)    | --> |      Guard       |
   |  +-------------+   +----------------+   |     | wraps EVERYTHING |
   |  |  Backbone   |-->|      head      |   |     +==================+
   |  +-------------+   +----------------+   |              |
   +=========================================+              v
        state = [ layer block | head block ]        Pipeline, persist,
                  one flat contiguous array          all unchanged
```

One `update(x, y)` does both levels:

```
   x  (flat: prod(in_shape) image values, then the extras)
    |
    +-- [:image] --> reshape --> layer.features() --.
    |                                               >-- concat --> head.update(h, y)
    +-- [image:] --> extras ------------------------'                     |
                                                                          v
                       layer.apply_gradients() <-- layer.backward() <-- input_gradient()
                                                        d_feat            295 values
                                                     = d_h[:144]        sliced at the
                                                                        feature width
```

**The input is flat on purpose.** `FrameMap` builds a 1-D regressor, so an image-shaped adapter
could not be fed by a `Pipeline` at all. The first `prod(in_shape)` entries are reshaped; the rest
are **extras** — the demo's memory pyramid and constant — which bypass the layers, because the
head's gradient with respect to them is not the layers' to receive.

What it buys, measured against the arrangement it replaces:

| | backbone outside the guard | `Stack` |
|---|---|---|
| layers checkpointed | no | **yes** |
| layers rolled back on a trip | no | **yes**, against an unguarded control |
| layers frozen with the head | no — the fault enters the representation | **yes** |
| layers in the persistence record | a second store, a second fingerprint | **one record** |
| who owns the chain rule | 42 lines of `demo/pointer.py`, linear heads only | the library, any head |

**Two limits, refused rather than half-built.** More than one layer is refused at construction —
`backward` returns no gradient with respect to its own input, so layers cannot be chained; depth
belongs *inside* a layer, and a `Backbone` plan is already a stack. And a `Stack`'s own
`input_gradient()` refuses, because its input is the raw image and nothing sits below it.

**`Stack.state` is a snapshot, not live storage**, which diverges from every other adapter here: the
components own their own arrays, so the block is re-gathered when read. Safe for the paths that
matter — a guard copies out of it and calls `load_state` to restore — and pinned by a test rather
than described.

**Still in the demo**: `learn_features` and its 42 lines. Moving the demo on to `Stack` is next.

## How the seam was verified

**Against central differences, not against "it learns"** — a wrong backward pass still trains, just
worse, and the demo's 42-line `learn_features` had never been checked this way at all. Max relative
error against `(L(x+h) - L(x-h)) / 2h`:

| adapter | shapes | max relative error |
|---|---|---|
| RLS | 4→2, 7→1, 6→3 | ~1e-10 |
| NLMS | 4→2, 7→1, 6→3 | ~1e-10 |
| MLP `()`, `(8,)`, `(6,5)` | 4→2, 7→1, 6→3 | ~1e-9 |

**The network's last hop is new code rather than a rearrangement**, which is why it was the case
worth doubting: its backward loop stopped one layer short of the input, so `delta @ w[0].T` is
added — with **no** activation derivative, because `acts[0]` is `x` and the network did not produce
it. A mutant applying `_dact` there is caught.

**Five mutants caught; one is equivalent and recorded rather than contrived around.** Writing
`self._w[0] @ delta` instead of `delta @ self._w[0].T` survived, and it is not a gap: the two are
**bit-identical** by definition of the transpose, measured. The five caught are dropping the −2
scaling, flipping its sign, leaking a gradient from a rejected sample, handing out a copy instead of
the preallocated buffer, and computing from the post-update `theta`.
