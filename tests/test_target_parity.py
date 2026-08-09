# SPDX-License-Identifier: Apache-2.0
"""The one target-parity question the parametrised suites cannot ask.

Every parity case now runs on both architectures — the four suites take their tool from
`_toolbuild.build_tool` and are parametrised `[host]` / `[target]`. This file used to hold a
hand-picked dozen instead, justified by a QEMU invocation costing "about a second"; a review
measured **157 ms** and pointed out the subset passed unchanged when the host binaries were
substituted for the target ones. Sampling had no cost argument behind it and no discriminating
power in front of it.

What is left here is the check the suites still cannot make, because each of them compares one
build against the reference and never the two builds against each other: **the target's
disagreement must stay in the same league as the host's.** That is what catches a target whose
arithmetic is subtly not IEEE — the review set the emulated FPU to round-toward-zero, which
costs nineteen times in state deviation, and every one of the twelve subset tests passed.

The bound cannot be 1.0, and the reason is a declared difference rather than slack: GCC
contracts `a*b + c` into `VFMA.F32` on VFPv4, which the x86-64 SSE2 baseline has no equivalent
for, so target f32 legitimately differs from host f32. That is `rls.hpp`'s fourth divergence.
The bound is 8×: measured, the FMA costs 2.5× on this fixture and 4.7× on a noisier one, so a
number derived from a single point would have failed the moment the fixture set grew. A
rounding-mode fault reports 37.5× and still fails by a factor of five.

Skipped, not failed, without the cross toolchain and `qemu-system-arm` on PATH.
"""

import shutil

import numpy as np
import pytest

import _toolbuild
import test_cpp_parity as X

#: Eight, and the number moved once. Four was derived from this one fixture, where the worst
#: host-to-target ratio is 2.5; a later SNR sweep found 4.7 on a noisier case, so a bound taken
#: from a single point would have failed the moment the fixture set grew. Eight covers the
#: measured range with headroom and still fails a round-toward-zero FPU, which reports 37.5,
#: by a factor of five.
MAX_RATIO = 8.0

#: Below this the ratio is noise: two builds can differ by one ulp on a quantity where the
#: host happens to be exact, and dividing by that is not a finding.
FLOOR_ULP = 8.0


@pytest.fixture(scope="session")
def both_builds(tmp_path_factory):
    if shutil.which("qemu-system-arm") is None or shutil.which("arm-none-eabi-g++") is None:
        pytest.skip("need arm-none-eabi and qemu-system-arm on PATH")
    return {a: _toolbuild.build_tool("xcheck", a, tmp_path_factory.mktemp(f"ratio_{a}"))
            for a in ("host", "target")}


def _deviations(tool, tmp_path, dtype):
    """Per-quantity deviation from the Python reference, in ulp, for one build."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    xs, ys = X.excited(6, 300, seed=41)
    py_rows, py_state, cpp_rows, cpp_state = X.run_both(
        tool, tmp_path, xs, ys, lam=0.99, p0=100.0, dtype=dtype)
    eps = float(np.finfo(np.dtype(dtype)).eps)
    out = {"state": X.scale_deviation(py_state, cpp_state) / eps}
    for field in X.FIELD_ULP:
        a = np.array([r[field] for r in py_rows])
        b = np.array([r[field] for r in cpp_rows])
        out[field] = X.scale_deviation(a, b) / eps
    return out


@pytest.mark.parametrize("dtype", ["f4", "f8"])
def test_the_target_disagrees_no_more_than_the_host_does(both_builds, tmp_path, dtype):
    """Both builds against the same reference, then against each other.

    A target that agreed with the reference to within its own tolerance while being an order
    of magnitude worse than the host would pass every other test in this repository. This is
    the only place that ratio is looked at.
    """
    host = _deviations(both_builds["host"], tmp_path / "h", dtype)
    target = _deviations(both_builds["target"], tmp_path / "t", dtype)
    worst = None
    for field in host:
        h, t = host[field], target[field]
        if max(h, t) < FLOOR_ULP:
            continue
        ratio = t / max(h, 1e-30)
        if worst is None or ratio > worst[1]:
            worst = (field, ratio, h, t)
        assert ratio <= MAX_RATIO, (
            f"{dtype} {field}: host {h:.1f} ulp, target {t:.1f} ulp — {ratio:.1f}x. "
            "Either a new divergence appeared or the target's arithmetic is not IEEE.")
    # ...and the comparison has to have looked at something. Every quantity sitting under the
    # noise floor would make this test vacuous, which is the shape it exists to avoid.
    assert worst is not None, f"{dtype}: every quantity was below {FLOOR_ULP} ulp on both"


def test_f64_is_identical_on_both_builds(both_builds, tmp_path):
    """Doubles are soft float on a single-precision FPU, so there is nothing for the target to
    contract and nothing for a rounding mode to change: the two builds should agree with the
    reference to the same figure, not merely to the same league.

    Pinned separately because it is the control for the test above. If f64 ever drifts, that
    ratio bound is masking something rather than tolerating a known difference.
    """
    host = _deviations(both_builds["host"], tmp_path / "h64", "f8")
    target = _deviations(both_builds["target"], tmp_path / "t64", "f8")
    for field in host:
        assert host[field] == pytest.approx(target[field], rel=1e-9), (
            f"f64 {field}: host {host[field]:.3f} ulp, target {target[field]:.3f} ulp")
