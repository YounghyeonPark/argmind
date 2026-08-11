"""`RuntimeShapedBackbone` against `argmind.backbone.Backbone`.

Weights come from the reference, as they do for the network: the two initialisations do not
match on purpose, and a device restoring a trained model takes exactly this path.

What is expected to differ is reduction order -- declared divergence 3, which every ported
component here carries. It is **larger than elsewhere for a structural reason**: a pointwise
layer at 512 channels sums 512 terms per output and a 3x3 convolution over 256 channels sums
2,304, where an RLS update at six features sums six. The tests assert that the deviation
*grows with the sum length*, which is what says it is reduction order rather than a bug.
"""
import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest

import _toolbuild

from argmind.backbone import Backbone, mobilenet_v1

REPO = Path(__file__).resolve().parents[1]
CPP = REPO / "cpp"

# The `4x` is the padding a C++ compiler inserts before the double, which `struct` does
# not add for `<`. bbcheck static_asserts 32 and the first version of this packed 28.
CASE = struct.Struct("<4sHBBHHHHI4xd")   # 32 bytes; bbcheck asserts the same
LAYER = struct.Struct("<B3xiii")         # 16 bytes
KIND = {"conv": 0, "dw": 1, "pw": 2}


@pytest.fixture(scope="session", params=_toolbuild.ARCHES)
def bbcheck(request, tmp_path_factory):
    """Built for the host and for the declared target, like every other parity tool here.

    **The target cases are small on purpose and the reason is the harness, not the library.**
    `cpp/target/qemu/startup.c` gives the image a 64 KB heap out of `.bss`, because the
    AN547's DTCM is 512 KB and a larger array made the M55 image produce no output at all. A
    96x96 MobileNet needs 840 KB of weights and 216 KB of activations in fp32, so it cannot
    run there. An STM32N6 has 4.2 MB. What the target runs is the *arithmetic*, at shapes
    that fit, which is the question a port has to answer.
    """
    work = tmp_path_factory.mktemp("bb_" + request.param)
    return _toolbuild.build_tool("bbcheck", request.param, work)


def run_both(tool, tmp_path, ref, image, dtype):
    h, w, c = ref.in_shape
    job = tmp_path / "case.bin"
    with job.open("wb") as f:
        f.write(CASE.pack(b"AMBB", 1, np.dtype(dtype).itemsize, ref.pool, h, w, c,
                          len(ref.plan), ref.state_size, ref.output_scale))
        for entry in ref.plan:
            if entry[0] == "conv":
                f.write(LAYER.pack(KIND["conv"], entry[1], entry[2], entry[3]))
            elif entry[0] == "pw":
                f.write(LAYER.pack(KIND["pw"], entry[1], 1, 1))
            else:
                f.write(LAYER.pack(KIND["dw"], 0, entry[1], entry[2]))
        f.write(np.asarray(ref.state, dtype=dtype).tobytes())
        f.write(np.asarray(image, dtype=dtype).tobytes())

    out = tmp_path / "res.bin"
    done = subprocess.run([str(tool), str(job), str(out)], capture_output=True, text=True)
    assert done.returncode == 0, f"bbcheck failed ({done.returncode}): {done.stderr}"

    blob = out.read_bytes()
    n = struct.unpack_from("<I", blob, 0)[0]
    cpp = np.frombuffer(blob, dtype=dtype, count=n, offset=4)
    assert 4 + n * np.dtype(dtype).itemsize == len(blob), "trailing bytes"
    return np.asarray(ref.features(image)), cpp


def ulp(a, b, dtype):
    """Against the RUN's scale, not each element's own value -- the correction this
    repository has needed four times, for a quantity that can pass through zero."""
    scale = max(float(np.max(np.abs(a))), 1e-30)
    return float(np.max(np.abs(a - b))) / scale / float(np.finfo(np.dtype(dtype)).eps)


PLANS = [
    ("one conv", [("conv", 5, 3, 2)], (12, 12, 3)),
    ("grid pooled 2x2", [("conv", 5, 3, 2), ("dw", 3, 2)], (16, 16, 3), 2),
    ("grid pooled 3x3", [("conv", 6, 3, 2), ("dw", 3, 1), ("pw", 8)], (18, 18, 3), 3),
    ("grid that does not divide", [("conv", 4, 3, 2)], (11, 13, 3), 3),
    ("one depthwise", [("conv", 4, 3, 1), ("dw", 3, 2)], (10, 10, 2)),
    ("a separable block", [("conv", 6, 3, 2), ("dw", 3, 1), ("pw", 8)], (16, 16, 3)),
    ("stride everywhere", [("conv", 4, 3, 2), ("dw", 3, 2), ("pw", 6), ("dw", 3, 2),
                           ("pw", 10)], (20, 20, 3)),
    ("a 5x5 kernel", [("conv", 4, 5, 2), ("dw", 5, 1), ("pw", 7)], (14, 11, 3)),
    ("non-square input", [("conv", 5, 3, 1), ("dw", 3, 2), ("pw", 9)], (9, 13, 4)),
]


@pytest.mark.parametrize("dtype", ["f4", "f8"])
@pytest.mark.parametrize("case", PLANS, ids=lambda c: c[0])
def test_the_port_agrees_with_the_reference(bbcheck, tmp_path, dtype, case):
    label, plan, shape = case[0], case[1], case[2]
    pool = case[3] if len(case) > 3 else 1
    ref = Backbone(plan, shape, dtype=dtype, seed=3, pool=pool)
    img = np.random.default_rng(7).standard_normal(shape)
    py, cpp = run_both(bbcheck, tmp_path, ref, img, dtype)

    assert py.shape == cpp.shape == (ref.n_outputs,)
    dev = ulp(py, cpp, dtype)
    assert dev < 400.0, f"{label} {dtype}: embedding deviates by {dev:.1f} ulp"
    # A control: two all-zero embeddings would agree perfectly.
    assert float(np.max(np.abs(py))) > 1e-6, "the embedding is ~zero, so this compares nothing"


def test_the_deviation_is_reduction_order_and_the_stack_absorbs_it(tmp_path_factory,
                                                                   tmp_path):
    """Classifying the difference instead of accepting a small number -- and the
    classification took two corrections.

    numpy dispatches its convolution to a BLAS matmul and the port is ordered loops, so what
    is left should be reduction order, whose signature is a deviation that grows with how
    many terms each output sums. **Through the whole stack it does not grow**: 1.18, 1.88 and
    0.77 ulp at pointwise widths of 4, 32 and 256.

    The first explanation offered was `relu6` saturating at 6, and it is **measured to be
    wrong** -- 0.0% of activations reach the clip at any width. What is true: **55% are
    clipped to exactly zero**, which is bit-identical whatever order produced it, and global
    average pooling then averages errors of random sign over H*W.

    Isolating one pointwise layer, no pooling and no stack, the growth is there and plain:
    **0.52, 0.50, 1.31, 3.94, 4.17 ulp** at cin of 4, 16, 64, 256 and 1024. So the mechanism
    is reduction order, and the full stack absorbs most of it.
    """
    # Host only: the 256-wide case holds two activation buffers of 9,216 values, which is
    # 147 KB at f8 against the QEMU harness's 64 KB heap. The target runs the small plans.
    bbcheck = _toolbuild.build_tool("bbcheck", "host", tmp_path_factory.mktemp("bbwide"))

    from argmind.backbone import conv2d          # noqa: PLC0415

    def loop_pointwise(x, w, b):
        h, wd, cin = x.shape
        out = np.empty((h, wd, w.shape[0]), dtype=x.dtype)
        for i in range(h):
            for j in range(wd):
                for o in range(w.shape[0]):
                    acc = x.dtype.type(0)
                    for c in range(cin):
                        acc += x[i, j, c] * w[o, 0, 0, c]
                    out[i, j, o] = acc + b[o]
        return out

    eps = float(np.finfo(np.float64).eps)
    single = {}
    for cin in (4, 256):
        g = np.random.default_rng(cin)
        x = g.standard_normal((4, 4, cin))
        w = g.standard_normal((6, 1, 1, cin)) / np.sqrt(cin)
        b = g.standard_normal(6)
        a, c = conv2d(x, w, b, 1), loop_pointwise(x, w, b)
        single[cin] = float(np.max(np.abs(a - c))) / float(np.max(np.abs(a))) / eps
    assert single[256] > single[4] * 3.0, (
        f"a single layer no longer shows the growth ({single}), so the deviation is not "
        "reduction order and this classification is wrong")

    # ...and through the stack it flattens, for a reason that is measured rather than guessed.
    got = {}
    for width in (4, 256):
        for dtype in ("f4", "f8"):
            plan = [("conv", 8, 3, 2), ("pw", width), ("pw", 8)]
            ref = Backbone(plan, (12, 12, 3), dtype=dtype, seed=2)
            img = np.random.default_rng(5).standard_normal((12, 12, 3))
            py, cpp = run_both(bbcheck, tmp_path, ref, img, dtype)
            got[(width, dtype)] = ulp(py, cpp, dtype)
    assert got[(256, "f8")] < single[256], (
        "the stack no longer absorbs the growth, so the reason given in this docstring "
        "is stale")
    for width in (4, 256):
        lo, hi = sorted((got[(width, "f4")], got[(width, "f8")]))
        assert hi <= max(lo * 30.0, 8.0), (
            f"width {width}: f32 {got[(width, 'f4')]:.1f} against f64 "
            f"{got[(width, 'f8')]:.1f} ulp -- the same order is what says algorithmic "
            "rounding rather than a precision-dependent bug")


def test_relu6_saturation_is_not_what_absorbs_the_deviation():
    """The explanation that was offered first and measured to be wrong, pinned so it is not
    offered again: **nothing reaches the clip at 6**. What absorbs the difference is the
    clip at zero, which more than half the activations hit."""
    from argmind.backbone import conv2d, relu6   # noqa: PLC0415

    g = np.random.default_rng(5)
    img = g.standard_normal((12, 12, 3))
    ref = Backbone([("conv", 8, 3, 2), ("pw", 256), ("pw", 8)], (12, 12, 3),
                   dtype="f8", seed=2)
    s = np.asarray(ref.state)
    w0 = s[:8 * 27].reshape(8, 3, 3, 3)
    b0 = s[8 * 27:8 * 27 + 8]
    a1 = relu6(conv2d(img, w0, b0, 2))
    o = 8 * 27 + 8
    wp = s[o:o + 256 * 8].reshape(256, 1, 1, 8)
    bp = s[o + 256 * 8:o + 256 * 8 + 256]
    a2 = relu6(conv2d(a1, wp, bp, 1))

    assert float(np.mean(a2 >= 6.0)) == 0.0, (
        "activations now reach the clip at 6, so the withdrawn explanation may be live "
        "again and should be re-measured rather than assumed dead")
    at_zero = float(np.mean(a2 <= 0.0))
    assert at_zero > 0.4, (
        f"only {at_zero:.1%} of activations are clipped to zero; that was the mechanism "
        "given for the stack absorbing the reduction-order growth")


def test_the_full_mobilenet_agrees(tmp_path_factory, tmp_path):
    """The configuration the scope decision points at: 27 layers, 210,336 weights, a 256-wide
    embedding. Deep enough that a wrong padding rule or a transposed filter compounds into
    something obvious rather than staying at roundoff."""
    # Host only: 840 KB of weights and 216 KB of activations do not fit the QEMU harness's
    # 64 KB heap. That is the harness, not the library -- an STM32N6 has 4.2 MB.
    tool = _toolbuild.build_tool("bbcheck", "host", tmp_path_factory.mktemp("bbfull"))
    ref = mobilenet_v1(0.25, (96, 96, 3), dtype="f4", seed=1)
    img = np.random.default_rng(4).standard_normal((96, 96, 3))
    py, cpp = run_both(tool, tmp_path, ref, img, "f4")
    assert py.shape == (256,) and ref.state_size == 210336
    dev = ulp(py, cpp, "f4")
    assert dev < 2000.0, f"the full stack deviates by {dev:.1f} ulp"
    assert float(np.max(np.abs(py))) > 1e-8


def test_the_output_scale_crosses(bbcheck, tmp_path):
    """`output_scale` is the constant `calibrate_scale` reports, and a port that dropped it
    would produce an embedding eight orders of magnitude off with nothing else wrong."""
    plan = [("conv", 6, 3, 2), ("dw", 3, 1), ("pw", 8)]
    img = np.random.default_rng(7).standard_normal((16, 16, 3))
    ref = Backbone(plan, (16, 16, 3), dtype="f8", seed=3)
    plain, _ = run_both(bbcheck, tmp_path, ref, img, "f8")

    ref.output_scale = 1000.0
    py, cpp = run_both(bbcheck, tmp_path, ref, img, "f8")
    assert ulp(py, cpp, "f8") < 400.0
    assert np.allclose(py, plain * 1000.0, rtol=1e-12), "the reference did not apply it"
    assert not np.allclose(cpp, plain), "the port ignored output_scale"


def test_the_port_refuses_what_the_reference_refuses(bbcheck, tmp_path):
    """Both sides refuse a plan they cannot build, in their own idiom: the reference raises
    and the port returns an invalid object, matching the rest of the hosted tier."""
    with pytest.raises(ValueError, match="unknown layer"):
        Backbone([("attention", 4)], (8, 8, 3))

    ref = Backbone([("conv", 4, 3, 1)], (8, 8, 3), dtype="f8")
    job = tmp_path / "bad.bin"
    with job.open("wb") as f:
        f.write(CASE.pack(b"AMBB", 1, 8, 0, 8, 8, 3, 1, ref.state_size, 1.0))
        f.write(LAYER.pack(KIND["conv"], 0, 3, 1))       # zero output channels
        f.write(np.asarray(ref.state, dtype="f8").tobytes())
        f.write(np.zeros(8 * 8 * 3, dtype="f8").tobytes())
    done = subprocess.run([str(bbcheck), str(job), str(tmp_path / "o.bin")],
                          capture_output=True, text=True)
    assert done.returncode == 4, (
        f"the port accepted a plan with no output channels (exit {done.returncode})")

    # A control: the same encoder at a valid plan must succeed, or exit 4 could mean anything.
    py, cpp = run_both(bbcheck, tmp_path, ref, np.zeros((8, 8, 3)), "f8")
    assert py.shape == cpp.shape
