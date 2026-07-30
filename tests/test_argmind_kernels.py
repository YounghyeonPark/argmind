"""Every kernel against a numpy reference, through real descriptors.

Executed the way the hardware would: the descriptor names addresses, dimensions
and strides, and the kernel reads strided views out of the scratchpad. Nothing
here calls the kernel with numpy arrays directly, because that would not test the
addressing -- which is where the bugs are.
"""

from __future__ import annotations

import numpy as np
import pytest

from argmind import (
    Descriptor,
    Device,
    DeviceConfig,
    Format,
    Lifetime,
    Opcode,
    Program,
    Structure,
    Transpose,
    auto_chain,
    macif,
)

RNG = np.random.default_rng(20260730)


def _device(scratch=1 << 18) -> Device:
    return Device(DeviceConfig(scratchpad_bytes=scratch, banks=16,
                               macro=macif.UNIVERSAL, strict_v0=False,
                               host_bytes=1 << 22))


def _run(dev: Device, descriptors, outputs) -> None:
    """Assign dependency tokens as codegen would, then build and call.

    Tokens are assigned rather than hand-written because the device's hazard
    checker refuses a block whose tokens do not order its real dependencies --
    which is the behaviour under test in ``test_argmind_program.py``.
    """
    built = dev.build(Program("t", auto_chain(descriptors).descriptors))
    dev.call(built, outputs=outputs)


def _load(dev: Device, name: str, values, fmt=Format.FP64):
    values = np.asarray(values, dtype=np.float64)
    buf = dev.alloc(name, values.shape, fmt, Lifetime.PERSISTENT)
    buf.write(values)
    return buf


def _out(dev: Device, name: str, shape, fmt=Format.FP64):
    return dev.alloc(name, shape, fmt, Lifetime.PERSISTENT)


# ---------------------------------------------------------------------------

def test_gemv():
    dev = _device()
    m, k = 7, 5
    A = RNG.normal(size=(m, k))
    x = RNG.normal(size=k)
    a, b = _load(dev, "A", A), _load(dev, "x", x)
    y = _out(dev, "y", m)
    _run(dev, [Descriptor(Opcode.GEMV, Format.FP64, a=a.spad_offset,
                          b=b.spad_offset, c=y.spad_offset, m=m, k=k, lda=k,
                          alpha=2.0)], [y])
    assert np.allclose(y.read(), 2.0 * A @ x)


def test_gemv_transposed():
    dev = _device()
    m, k = 7, 5
    A = RNG.normal(size=(m, k))
    v = RNG.normal(size=m)
    a, b = _load(dev, "A", A), _load(dev, "v", v)
    y = _out(dev, "y", k)
    _run(dev, [Descriptor(Opcode.GEMV, Format.FP64, a=a.spad_offset,
                          b=b.spad_offset, c=y.spad_offset, m=m, k=k, lda=k,
                          alpha=1.0, transpose=Transpose.A)], [y])
    assert np.allclose(y.read(), A.T @ v)


def test_gemv_accumulate_is_in_place():
    dev = _device()
    m, k = 4, 3
    A = RNG.normal(size=(m, k))
    x = RNG.normal(size=k)
    y0 = RNG.normal(size=m)
    a, b = _load(dev, "A", A), _load(dev, "x", x)
    y = _load(dev, "y", y0)
    spad_before = y.spad_offset
    _run(dev, [Descriptor(Opcode.GEMV, Format.FP64, a=a.spad_offset,
                          b=b.spad_offset, c=y.spad_offset, m=m, k=k, lda=k,
                          alpha=1.0, accumulate=True)], [y])
    assert y.spad_offset == spad_before
    assert np.allclose(y.read(), A @ x + y0)


def test_gemv_symmetric_reads_only_the_lower_triangle():
    dev = _device()
    n = 6
    L = np.tril(RNG.normal(size=(n, n)))
    full = L + np.tril(L, -1).T
    stored = L.copy()
    stored[np.triu_indices(n, 1)] = 1e6         # garbage in the unused triangle
    x = RNG.normal(size=n)
    a, b = _load(dev, "A", stored), _load(dev, "x", x)
    y = _out(dev, "y", n)
    _run(dev, [Descriptor(Opcode.GEMV, Format.FP64, a=a.spad_offset,
                          b=b.spad_offset, c=y.spad_offset, m=n, k=n, lda=n,
                          alpha=1.0, structure=Structure.SYMMETRIC)], [y])
    assert np.allclose(y.read(), full @ x)


def test_gemm_all_transpose_combinations():
    dev = _device()
    m, k, n = 4, 5, 3
    A = RNG.normal(size=(m, k))
    B = RNG.normal(size=(k, n))
    for tag, tr, Astore, Bstore in (
        ("nn", Transpose.NONE, A, B),
        ("tn", Transpose.A, A.T, B),
        ("nt", Transpose.B, A, B.T),
        ("tt", Transpose.BOTH, A.T, B.T),
    ):
        d = _device()
        a = _load(d, "A", Astore)
        b = _load(d, "B", Bstore)
        c = _out(d, "C", (m, n))
        _run(d, [Descriptor(Opcode.GEMM, Format.FP64, a=a.spad_offset,
                            b=b.spad_offset, c=c.spad_offset, m=m, k=k, n=n,
                            lda=Astore.shape[1], ldb=Bstore.shape[1], ldc=n,
                            alpha=1.0, transpose=tr)], [c])
        assert np.allclose(c.read(), A @ B), tag


def test_gemm_gram_matrix_with_one_buffer_as_both_operands():
    """K = rho * A'A, which the ADMM setup block does with a == b."""
    dev = _device()
    m, n = 9, 4
    A = RNG.normal(size=(m, n))
    a = _load(dev, "A", A)
    c = _out(dev, "K", (n, n))
    _run(dev, [Descriptor(Opcode.GEMM, Format.FP64, a=a.spad_offset,
                          b=a.spad_offset, c=c.spad_offset, m=n, k=m, n=n,
                          lda=n, ldb=n, ldc=n, alpha=0.25,
                          transpose=Transpose.A)], [c])
    assert np.allclose(c.read(), 0.25 * A.T @ A)


def test_trsv_lower_and_upper_and_transposed():
    dev = _device()
    n = 6
    M = RNG.normal(size=(n, n))
    S = M @ M.T + n * np.eye(n)
    L = np.linalg.cholesky(S)
    rhs = RNG.normal(size=n)

    for tag, structure, tr, expected in (
        ("L", Structure.TRIL, Transpose.NONE, np.linalg.solve(L, rhs)),
        ("L'", Structure.TRIL, Transpose.A, np.linalg.solve(L.T, rhs)),
        ("U", Structure.TRIU, Transpose.NONE, np.linalg.solve(L.T, rhs)),
    ):
        d = _device()
        stored = L if structure is Structure.TRIL else L.T
        a = _load(d, "A", stored)
        b = _load(d, "b", rhs)
        x = _out(d, "x", n)
        _run(d, [Descriptor(Opcode.TRSV, Format.FP64, a=a.spad_offset,
                            b=b.spad_offset, c=x.spad_offset, m=n, lda=n,
                            structure=structure, transpose=tr)], [x])
        assert np.allclose(x.read(), expected), tag


def test_trsv_in_place_when_b_aliases_c():
    dev = _device()
    n = 5
    L = np.tril(RNG.normal(size=(n, n))) + n * np.eye(n)
    rhs = RNG.normal(size=n)
    a = _load(dev, "A", L)
    v = _load(dev, "v", rhs)
    _run(dev, [Descriptor(Opcode.TRSV, Format.FP64, a=a.spad_offset,
                          b=v.spad_offset, c=v.spad_offset, m=n, lda=n,
                          structure=Structure.TRIL)], [v])
    assert np.allclose(v.read(), np.linalg.solve(L, rhs))


def test_trsv_unit_diag_ignores_the_stored_diagonal():
    dev = _device()
    n = 4
    L = np.tril(RNG.normal(size=(n, n)), -1) + np.diag(RNG.uniform(2, 3, n))
    rhs = RNG.normal(size=n)
    unit = np.tril(L, -1) + np.eye(n)
    a, b = _load(dev, "A", L), _load(dev, "b", rhs)
    x = _out(dev, "x", n)
    _run(dev, [Descriptor(Opcode.TRSV, Format.FP64, a=a.spad_offset,
                          b=b.spad_offset, c=x.spad_offset, m=n, lda=n,
                          structure=Structure.TRIL, unit_diag=True)], [x])
    assert np.allclose(x.read(), np.linalg.solve(unit, rhs))


def test_axpy_and_scal():
    dev = _device()
    n = 11
    x0 = RNG.normal(size=n)
    y0 = RNG.normal(size=n)
    x, y = _load(dev, "x", x0), _load(dev, "y", y0)
    z = _out(dev, "z", n)
    _run(dev, [
        Descriptor(Opcode.SCAL, Format.FP64, b=x.spad_offset, c=z.spad_offset,
                   m=n, alpha=-1.5),
        Descriptor(Opcode.AXPY, Format.FP64, b=y.spad_offset, c=z.spad_offset,
                   m=n, alpha=0.25),
    ], [z])
    assert np.allclose(z.read(), -1.5 * x0 + 0.25 * y0)


def test_scal_in_place():
    dev = _device()
    n = 6
    v0 = RNG.normal(size=n)
    v = _load(dev, "v", v0)
    _run(dev, [Descriptor(Opcode.SCAL, Format.FP64, b=v.spad_offset,
                          c=v.spad_offset, m=n, alpha=3.0)], [v])
    assert np.allclose(v.read(), 3.0 * v0)


def test_axpy_with_a_strided_destination_walks_a_matrix_diagonal():
    """How the ADMM setup adds sigma*I, since v0 has no diagonal-add opcode."""
    dev = _device()
    n = 5
    K0 = RNG.normal(size=(n, n))
    K = _load(dev, "K", K0)
    sig = _load(dev, "sig", np.full(n, 0.25))
    _run(dev, [Descriptor(Opcode.AXPY, Format.FP64, b=sig.spad_offset,
                          c=K.spad_offset, m=n, ldb=1, ldc=n + 1,
                          alpha=1.0)], [K])
    assert np.allclose(K.read(), K0 + 0.25 * np.eye(n))


def test_axpy_flat_over_a_matrix():
    dev = _device()
    n = 6
    A0, B0 = RNG.normal(size=(n, n)), RNG.normal(size=(n, n))
    A, B = _load(dev, "A", A0), _load(dev, "B", B0)
    _run(dev, [Descriptor(Opcode.AXPY, Format.FP64, b=A.spad_offset,
                          c=B.spad_offset, m=n * n, alpha=1.0)], [B])
    assert np.allclose(B.read(), A0 + B0)


def test_dot_writes_one_scalar_at_address_a():
    dev = _device()
    n = 9
    x0, y0 = RNG.normal(size=n), RNG.normal(size=n)
    x, y = _load(dev, "x", x0), _load(dev, "y", y0)
    s = _out(dev, "s", 1)
    _run(dev, [Descriptor(Opcode.DOT, Format.FP64, a=s.spad_offset,
                          b=x.spad_offset, c=y.spad_offset, m=n)], [s])
    assert np.isclose(s.read()[0], x0 @ y0)


def test_dot_with_both_operands_the_same_gives_a_squared_two_norm():
    dev = _device()
    n = 7
    v0 = RNG.normal(size=n)
    v = _load(dev, "v", v0)
    s = _out(dev, "s", 1)
    _run(dev, [Descriptor(Opcode.DOT, Format.FP64, a=s.spad_offset,
                          b=v.spad_offset, c=v.spad_offset, m=n)], [s])
    assert np.isclose(s.read()[0], v0 @ v0)


def _interleaved(dev, name, lo, hi):
    """Bounds as (lo,hi) pairs -- the v0.1 PROJ operand layout (ruling R2)."""
    lu = np.empty((len(lo), 2))
    lu[:, 0], lu[:, 1] = lo, hi
    return _load(dev, name, lu)


def test_proj_is_out_of_place_in_v0_1():
    """Ruling R2: input at `a`, interleaved bounds at `b`, output at `c`.

    v0 could not express this -- a box projection needs four addresses if the
    bounds take two -- so projection was in-place-only and codegen paid a
    full-vector staging copy per iteration.
    """
    dev = _device()
    n = 8
    v0 = np.array([-3.0, -1.0, -0.2, 0.0, 0.2, 1.0, 3.0, 0.5])
    lo = np.full(n, -1.0)
    hi = np.full(n, 1.0)
    src = _load(dev, "src", v0)
    lu = _interleaved(dev, "lu", lo, hi)
    out = _out(dev, "out", n)
    _run(dev, [Descriptor(Opcode.PROJ, Format.FP64, a=src.spad_offset,
                          b=lu.spad_offset, c=out.spad_offset, m=n,
                          lda=1, ldb=2, ldc=1)], [out, src])
    assert np.allclose(out.read(), np.clip(v0, lo, hi))
    # The input is untouched: that is the whole point of out-of-place.
    assert np.allclose(src.read(), v0)


def test_proj_can_still_alias_for_an_in_place_projection():
    dev = _device()
    n = 8
    v0 = np.array([-3.0, -1.0, -0.2, 0.0, 0.2, 1.0, 3.0, 0.5])
    lo, hi = np.full(n, -1.0), np.full(n, 1.0)
    v = _load(dev, "v", v0)
    lu = _interleaved(dev, "lu", lo, hi)
    _run(dev, [Descriptor(Opcode.PROJ, Format.FP64, a=v.spad_offset,
                          b=lu.spad_offset, c=v.spad_offset, m=n,
                          lda=1, ldb=2, ldc=1)], [v])
    assert np.allclose(v.read(), np.clip(v0, lo, hi))


def test_proj_with_asymmetric_per_element_bounds():
    dev = _device()
    n = 6
    v0 = RNG.normal(size=n) * 3
    lo = RNG.uniform(-2, -1, n)
    hi = RNG.uniform(1, 2, n)
    src = _load(dev, "src", v0)
    lu = _interleaved(dev, "lu", lo, hi)
    out = _out(dev, "out", n)
    _run(dev, [Descriptor(Opcode.PROJ, Format.FP64, a=src.spad_offset,
                          b=lu.spad_offset, c=out.spad_offset, m=n,
                          lda=1, ldb=2, ldc=1)], [out])
    assert np.allclose(out.read(), np.clip(v0, lo, hi))


def test_chol_factorises_in_place_and_leaves_the_upper_triangle_alone():
    dev = _device()
    n = 6
    M = RNG.normal(size=(n, n))
    S = M @ M.T + n * np.eye(n)
    stored = np.tril(S)
    stored[np.triu_indices(n, 1)] = 1234.0       # must be ignored, and preserved
    K = _load(dev, "K", stored)
    _run(dev, [Descriptor(Opcode.CHOL, Format.FP64, a=K.spad_offset, m=n,
                          lda=n, structure=Structure.TRIL)], [K])
    got = K.read()
    assert np.allclose(np.tril(got), np.linalg.cholesky(S))
    assert np.all(got[np.triu_indices(n, 1)] == 1234.0)


def test_chol_then_two_trsv_solves_the_system():
    dev = _device()
    n = 7
    M = RNG.normal(size=(n, n))
    S = M @ M.T + n * np.eye(n)
    rhs = RNG.normal(size=n)
    K = _load(dev, "K", S)
    b = _load(dev, "b", rhs)
    x = _out(dev, "x", n)
    _run(dev, [
        Descriptor(Opcode.CHOL, Format.FP64, a=K.spad_offset, m=n, lda=n,
                   structure=Structure.TRIL, chain_token=1),
        Descriptor(Opcode.TRSV, Format.FP64, a=K.spad_offset, b=b.spad_offset,
                   c=x.spad_offset, m=n, lda=n, structure=Structure.TRIL,
                   chain_token=2, wait_mask=0b010),
        Descriptor(Opcode.TRSV, Format.FP64, a=K.spad_offset, b=x.spad_offset,
                   c=x.spad_offset, m=n, lda=n, structure=Structure.TRIL,
                   transpose=Transpose.A, chain_token=3, wait_mask=0b100),
    ], [x])
    assert np.allclose(x.read(), np.linalg.solve(S, rhs))


def test_nrmi_is_a_max_absolute_reduction():
    dev = _device()
    n = 9
    v0 = RNG.normal(size=n) * 5
    v = _load(dev, "v", v0)
    s = _out(dev, "s", 1)
    _run(dev, [Descriptor(Opcode.NRMI, Format.FP64, a=s.spad_offset,
                          b=v.spad_offset, m=n)], [s])
    assert np.isclose(s.read()[0], np.max(np.abs(v0)))


def test_vmul_is_elementwise():
    dev = _device()
    n = 8
    u0, v0 = RNG.normal(size=n), RNG.normal(size=n)
    u, v = _load(dev, "u", u0), _load(dev, "v", v0)
    w = _out(dev, "w", n)
    _run(dev, [Descriptor(Opcode.VMUL, Format.FP64, a=u.spad_offset,
                          b=v.spad_offset, c=w.spad_offset, m=n, alpha=1.0)], [w])
    assert np.allclose(w.read(), u0 * v0)


# ---------------------------------------------------------------------------
# Precision
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fmt,tol", [
    (Format.FP64, 1e-13),
    (Format.FP32, 3e-6),
    (Format.FP16, 3e-2),
    (Format.BF16, 2e-1),
])
def test_gemv_accuracy_degrades_as_the_format_narrows(fmt, tol):
    dev = _device()
    m, k = 8, 8
    A = RNG.uniform(0.5, 1.5, size=(m, k))
    x = RNG.uniform(0.5, 1.5, size=k)
    a = dev.alloc("A", (m, k), fmt, Lifetime.PERSISTENT)
    a.write(A)
    b = dev.alloc("x", k, fmt, Lifetime.PERSISTENT)
    b.write(x)
    y = dev.alloc("y", m, fmt, Lifetime.PERSISTENT)
    _run(dev, [Descriptor(Opcode.GEMV, fmt, a=a.spad_offset, b=b.spad_offset,
                          c=y.spad_offset, m=m, k=k, lda=k, alpha=1.0)], [y])
    exact = a.numbers().astype(np.float64) @ b.numbers().astype(np.float64)
    got = y.numbers().astype(np.float64)
    assert np.max(np.abs(got - exact) / np.abs(exact)) < tol


def test_fp16_is_measurably_more_accurate_than_bf16_at_the_same_width():
    """Why they are separate MACIF declarations rather than one 16-bit format."""
    m, k = 16, 16
    A = RNG.uniform(0.9, 1.1, size=(m, k))
    x = RNG.uniform(0.9, 1.1, size=k)
    exact = A @ x
    errors = {}
    for fmt in (Format.FP16, Format.BF16):
        dev = _device()
        a = dev.alloc("A", (m, k), fmt, Lifetime.PERSISTENT)
        a.write(A)
        b = dev.alloc("x", k, fmt, Lifetime.PERSISTENT)
        b.write(x)
        y = dev.alloc("y", m, fmt, Lifetime.PERSISTENT)
        _run(dev, [Descriptor(Opcode.GEMV, fmt, a=a.spad_offset,
                              b=b.spad_offset, c=y.spad_offset, m=m, k=k,
                              lda=k, alpha=1.0)], [y])
        errors[fmt] = float(np.max(np.abs(
            y.numbers().astype(np.float64) - exact) / np.abs(exact)))
    assert errors[Format.FP16] * 4 < errors[Format.BF16], errors
