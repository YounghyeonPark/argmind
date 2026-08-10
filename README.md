# argmind

**On-device online adaptation and inference for multi-sensor systems.** A C++17 core
behind a C ABI, plus a Python reference that is also the specification.

Apache-2.0. A side project, open source, no support commitment.

---

## What it is

Sensors in the field drift. Bias moves with temperature, a mount shifts, a surface
changes, a cable stretches, a battery ages. The usual answer is to calibrate offline and
hope the deployment matches. argmind takes the other route: **the model keeps updating
from the sensor stream after deployment**, inside bounded memory and bounded time, with
guardrails that cannot be left out.

```c
#include "argmind/argmind.h"
AM_VERSION_REQUIRE(0, 1);

static am_rls_storage obj[AM_PIPE_G_F32_6_1_MAX_BYTES / sizeof(am_rls_storage)];

am_pipe_g_f32_6_1_init(obj, sizeof(obj), /*lambda*/0.999f, /*p0*/100.0f,
                       /*checkpoint_every*/5, /*recover_after*/0,
                       AM_POLICY_HOLD_LAST, /*max_stale_s*/0.5, /*reference*/0);
am_pipe_g_f32_6_1_add_bound(obj, AM_FIELD_P_TRACE, 0, 0.0, 1, 1e6);
am_pipe_g_f32_6_1_configure(obj, rings, 2);

/* per cycle */
am_ring_f32_push(&rings[0], t, imu, 1);
am_step_report r;
am_pipe_g_f32_6_1_step(obj, rings, 2, &r);
if (r.has_prediction) use(r.prediction[0]);
```

Nothing there allocates. `step()` is bounded. The storage is yours.

## What it is *not*

- **Not a deep-learning runtime.** It adapts *parameters and estimates*. For a CNN use
  ONNX Runtime or TFLite; argmind sits alongside one, adapting the last layer.
- **Not a training framework.** No backpropagation, no autograd, no GPU.
- **Not a driver layer or middleware.** It owns no threads, talks to no hardware and
  does not depend on ROS. You feed it samples.
- **Not a model that learns a representation.** Both adapters are *linear in the
  parameters* — you supply the basis, and nothing whose output depends on the input's
  *history* (hysteresis, stiction, backlash) can be fitted by instantaneous features.
  What this is genuinely for is tracking a **drift** in something whose form is known.

That last point is the one people get wrong, so `DESIGN.md` spells it out at length.

## What is in it

| | |
|---|---|
| `Stream` | fixed-capacity ring, counted overwrites, monotonic time enforced at the source |
| `Aligner` | four explicit policies, signed staleness, causal/non-causal declared |
| `Rls` | exponentially weighted recursive least squares, `O(n²)` state |
| `Nlms` | normalised LMS, `O(n)` state and no covariance — for wide inputs |
| `Guard` | bounds, divergence detection, freeze, bounded rollback, caller-supplied health scalars |
| `Pipeline` | frame mapping through a declared index table, two input gates, bounded `step()` |
| `persist` | two-slot record format, CRC, configuration fingerprint |

**The `Guard` *is* an `Adapter`.** Not a component beside one — because "no guard" being
expressible means it gets left out.

## Using it

```cmake
find_package(argmind 0.1 REQUIRED)
target_link_libraries(your_target PRIVATE argmind::argmind)
```

```
cmake -S cpp -B build -DCMAKE_INSTALL_PREFIX=/where
cmake --build build --target argmind && cmake --install build
```

`tests/test_packaging.py` performs exactly that and then builds a separate project
against the result, so this block is executed rather than written down.

The Python reference is importable directly (`import argmind`) and is the specification
the C++ is verified against, not a binding to it.

## Where it runs

The core is C++17 **freestanding**: no heap, no exceptions, no RTTI, no libm beyond
`memcpy`/`memset` and libgcc's soft-float double helpers, and the external symbol set is
pinned by a test. It needs an **fp32 FPU** — that is a precision floor measured in the
covariance recursion, not a performance preference.

- **Cortex-M4F and above.** Both the declared floor and Cortex-M55 run the full
  self-check suite under QEMU, and every parity case runs on both host and target.
- **Hosted.** The same sources build and run as an ordinary library on a Linux/Windows
  host; that is how the reference is compared against it.
- Sizes come from a link map and are recorded in
  [`cpp/target/SIZES.md`](cpp/target/SIZES.md), which a test regenerates and gates.

**No timing figure appears anywhere in this repository.** There is no hardware, and an
emulator models the instruction set rather than the pipeline, so any cycle count taken
here would be fiction dressed as measurement. That claim stays absent until it can be
taken on silicon.

## How it is verified

- The **Python reference is the oracle**, and the C++ is compared against it
  case-by-case: state blocks in ulp, record bytes for exactness, alignment picks by bit
  pattern, pipeline bookkeeping step-for-step. Divergences are *declared* in the headers,
  with the measurement that identified each.
- RLS is checked against the **closed-form** exponentially weighted solution, not against
  itself. NLMS is checked against its **exact one-step law**.
- Every suite is **mutation-tested**, and surviving mutants are recorded where they live
  — including the ones that turn out to be equivalent on a 64-bit host.
- `python -m pytest` runs everything, skipping cleanly without the cross toolchain or
  QEMU. A tracked pre-commit hook runs it and blocks on failure:
  `git config core.hooksPath .githooks`.

## Documents

- [`DESIGN.md`](DESIGN.md) — the design, and the reasoning behind every trade, including
  the ones that were wrong first.
- [`CLAUDE.md`](CLAUDE.md) — project context and standards of evidence.
- [`examples/hosted_inference.c`](examples/hosted_inference.c) — a frozen backbone plus
  an adapted head, built and run by the test suite.

**On the numbers in those documents.** Several are corrections of earlier ones, and the
corrections are recorded rather than quietly edited away, because how a figure was got
wrong is worth as much as the figure.
