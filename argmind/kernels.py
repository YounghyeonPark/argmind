"""Functional kernel implementations, one per descriptor opcode.

Every kernel takes the scratchpad byte array and a decoded descriptor, reads its
operands through strided views at the descriptor's declared addresses, and writes
its result back **in place**. No kernel copies an operand into a working buffer:
that would defeat the residency architecture it exists to serve, and a functional
model that copied would let a code generator that copies pass its tests.

Numeric behaviour
-----------------
Operands are read in the descriptor's declared format and accumulated at the
width the MACIF contract declares for that format. For fp32 and fp64 that is the
same width as the operands. For the 16-bit formats and int8 the accumulator is 32
bits, so the accumulation is performed in fp32 and the *result* is rounded back
to the operand format on store -- because descriptor v0 has one precision field
and therefore cannot say "fp16 in, fp32 out" (limitation L4).

Reductions (DOT, NRMI) write a single element in the descriptor's format at the
address in ``a``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .descriptor import Descriptor, Opcode, Structure
from .macif import MacroCapabilities
from .precision import Format, from_arithmetic, to_arithmetic


@dataclass(frozen=True)
class Access:
    """A byte range a descriptor touches, for hazard analysis."""

    start: int
    stop: int

    def overlaps(self, other: "Access") -> bool:
        return self.start < other.stop and other.start < self.stop


# ---------------------------------------------------------------------------
# Strided views into the scratchpad
# ---------------------------------------------------------------------------

def _vector(spad: np.ndarray, base: int, n: int, inc: int, fmt: Format) -> np.ndarray:
    """A strided 1-D view. `inc` is in elements, as in BLAS."""
    item = fmt.itemsize
    span = item * (1 + (n - 1) * inc) if n else 0
    raw = spad[base:base + span]
    return np.lib.stride_tricks.as_strided(
        raw.view(fmt.storage_dtype), shape=(n,), strides=(item * inc,), writeable=True
    )


def _matrix(
    spad: np.ndarray, base: int, rows: int, cols: int, ld: int, fmt: Format
) -> np.ndarray:
    """A row-major view with leading dimension `ld` (elements per row)."""
    item = fmt.itemsize
    span = item * ((rows - 1) * ld + cols) if rows else 0
    raw = spad[base:base + span]
    return np.lib.stride_tricks.as_strided(
        raw.view(fmt.storage_dtype),
        shape=(rows, cols),
        strides=(item * ld, item),
        writeable=True,
    )


def _accum_dtype(fmt: Format, macro: MacroCapabilities) -> np.dtype:
    width = macro.accumulator_width(fmt)
    return np.dtype(np.float64) if width >= 64 else np.dtype(np.float32)


def _store(dest: np.ndarray, values: np.ndarray, fmt: Format) -> None:
    dest[...] = from_arithmetic(values, fmt).reshape(dest.shape)


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------

def _op_gemm(spad, d: Descriptor, macro: MacroCapabilities) -> None:
    """C := alpha * op(A) @ op(B)  [+ C]."""
    fmt, acc = d.precision, _accum_dtype(d.precision, macro)
    ar, ac = (d.k, d.m) if d.transpose.a else (d.m, d.k)
    br, bc = (d.n, d.k) if d.transpose.b else (d.k, d.n)
    A = to_arithmetic(_matrix(spad, d.a, ar, ac, d.lda, fmt), fmt).astype(acc)
    B = to_arithmetic(_matrix(spad, d.b, br, bc, d.ldb, fmt), fmt).astype(acc)
    if d.transpose.a:
        A = A.T
    if d.transpose.b:
        B = B.T
    if d.structure is Structure.SYMMETRIC:
        A = np.tril(A) + np.tril(A, -1).T
    out = _matrix(spad, d.c, d.m, d.n, d.ldc, fmt)
    result = acc.type(d.alpha_f32) * (A @ B)
    if d.accumulate:
        result = result + to_arithmetic(out, fmt).astype(acc)
    _store(out, result, fmt)


def _op_gemv(spad, d: Descriptor, macro: MacroCapabilities) -> None:
    """y := alpha * op(A) @ x  [+ y].  A is m x k as stored."""
    fmt, acc = d.precision, _accum_dtype(d.precision, macro)
    A = to_arithmetic(_matrix(spad, d.a, d.m, d.k, d.lda, fmt), fmt).astype(acc)
    if d.structure is Structure.SYMMETRIC:
        A = np.tril(A) + np.tril(A, -1).T
    if d.transpose.a:
        A = A.T
    xn, yn = A.shape[1], A.shape[0]
    x = to_arithmetic(_vector(spad, d.b, xn, d.ldb, fmt), fmt).astype(acc)
    yv = _vector(spad, d.c, yn, d.ldc, fmt)
    result = acc.type(d.alpha_f32) * (A @ x)
    if d.accumulate:
        result = result + to_arithmetic(yv, fmt).astype(acc)
    _store(yv, result, fmt)


def _op_trsv(spad, d: Descriptor, macro: MacroCapabilities) -> None:
    """x := op(A)^-1 @ b, A triangular m x m. `c` may alias `b` (in place)."""
    fmt, acc = d.precision, _accum_dtype(d.precision, macro)
    A = to_arithmetic(_matrix(spad, d.a, d.m, d.m, d.lda, fmt), fmt).astype(acc)
    lower = d.structure is not Structure.TRIU
    A = np.tril(A) if lower else np.triu(A)
    if d.unit_diag:
        np.fill_diagonal(A, acc.type(1.0))
    if d.transpose.a:
        A = A.T
        lower = not lower
    b = to_arithmetic(_vector(spad, d.b, d.m, d.ldb, fmt), fmt).astype(acc)
    # Explicit substitution rather than a LAPACK call: the point of the
    # functional model is that it does what the descriptor says, sequentially,
    # so a wrong triangle or a wrong transpose shows up as a wrong answer.
    n = d.m
    x = np.zeros(n, dtype=acc)
    rng = range(n) if lower else range(n - 1, -1, -1)
    for i in rng:
        if lower:
            s = b[i] - A[i, :i] @ x[:i]
        else:
            s = b[i] - A[i, i + 1:] @ x[i + 1:]
        x[i] = s / A[i, i]
    _store(_vector(spad, d.c, n, d.ldc, fmt), x, fmt)


def _op_axpy(spad, d: Descriptor, macro: MacroCapabilities) -> None:
    """y := alpha * x + y, length m."""
    fmt, acc = d.precision, _accum_dtype(d.precision, macro)
    x = to_arithmetic(_vector(spad, d.b, d.m, d.ldb, fmt), fmt).astype(acc)
    yv = _vector(spad, d.c, d.m, d.ldc, fmt)
    y = to_arithmetic(yv, fmt).astype(acc)
    _store(yv, acc.type(d.alpha_f32) * x + y, fmt)


def _op_scal(spad, d: Descriptor, macro: MacroCapabilities) -> None:
    """y := alpha * x, length m. `c` may alias `b` for a true in-place scale."""
    fmt, acc = d.precision, _accum_dtype(d.precision, macro)
    x = to_arithmetic(_vector(spad, d.b, d.m, d.ldb, fmt), fmt).astype(acc)
    _store(_vector(spad, d.c, d.m, d.ldc, fmt), acc.type(d.alpha_f32) * x, fmt)


def _op_dot(spad, d: Descriptor, macro: MacroCapabilities) -> None:
    """a[0] := alpha * (x . y), length m. Result is ONE element at `a`."""
    fmt, acc = d.precision, _accum_dtype(d.precision, macro)
    x = to_arithmetic(_vector(spad, d.b, d.m, d.ldb, fmt), fmt).astype(acc)
    y = to_arithmetic(_vector(spad, d.c, d.m, d.ldc, fmt), fmt).astype(acc)
    _store(_vector(spad, d.a, 1, 1, fmt), np.array([acc.type(d.alpha_f32) * (x @ y)]),
           fmt)


def _op_proj(spad, d: Descriptor, macro: MacroCapabilities) -> None:
    """c := clip(c, a, b), length m -- box projection, INTRINSICALLY IN PLACE.

    v0 has three base addresses, and a box projection needs four (value, lower,
    upper, result). So the value must already be in the destination. See
    descriptor.V0_LIMITATIONS entry L2.
    """
    fmt, acc = d.precision, _accum_dtype(d.precision, macro)
    lo = to_arithmetic(_vector(spad, d.a, d.m, 1, fmt), fmt).astype(acc)
    hi = to_arithmetic(_vector(spad, d.b, d.m, 1, fmt), fmt).astype(acc)
    zv = _vector(spad, d.c, d.m, d.ldc, fmt)
    z = to_arithmetic(zv, fmt).astype(acc)
    _store(zv, np.clip(z, lo, hi), fmt)


def _op_chol(spad, d: Descriptor, macro: MacroCapabilities) -> None:
    """A := chol(A), lower factor written in place over the lower triangle.

    The upper triangle is left untouched, as an in-place ``potrf`` leaves it.
    Consumers must select the triangle via the descriptor `structure` field.
    """
    fmt, acc = d.precision, _accum_dtype(d.precision, macro)
    view = _matrix(spad, d.a, d.m, d.m, d.lda, fmt)
    A = to_arithmetic(view, fmt).astype(acc)
    A = np.tril(A) + np.tril(A, -1).T          # symmetrise from the lower triangle
    L = np.linalg.cholesky(A)
    out = to_arithmetic(view, fmt).astype(acc)
    idx = np.tril_indices(d.m)
    out[idx] = L[idx]
    _store(view, out, fmt)


def _op_nrmi(spad, d: Descriptor, macro: MacroCapabilities) -> None:
    """PROPOSED, not v0: a[0] := alpha * max|x_i|, length m."""
    fmt, acc = d.precision, _accum_dtype(d.precision, macro)
    x = to_arithmetic(_vector(spad, d.b, d.m, d.ldb, fmt), fmt).astype(acc)
    val = acc.type(d.alpha_f32) * (np.max(np.abs(x)) if d.m else acc.type(0.0))
    _store(_vector(spad, d.a, 1, 1, fmt), np.array([val]), fmt)


def _op_vmul(spad, d: Descriptor, macro: MacroCapabilities) -> None:
    """PROPOSED, not v0: c := alpha * (a * b) elementwise, length m."""
    fmt, acc = d.precision, _accum_dtype(d.precision, macro)
    u = to_arithmetic(_vector(spad, d.a, d.m, 1, fmt), fmt).astype(acc)
    v = to_arithmetic(_vector(spad, d.b, d.m, d.ldb, fmt), fmt).astype(acc)
    _store(_vector(spad, d.c, d.m, d.ldc, fmt), acc.type(d.alpha_f32) * u * v, fmt)


KERNELS: dict[Opcode, Callable] = {
    Opcode.GEMM: _op_gemm,
    Opcode.GEMV: _op_gemv,
    Opcode.TRSV: _op_trsv,
    Opcode.AXPY: _op_axpy,
    Opcode.PROJ: _op_proj,
    Opcode.DOT: _op_dot,
    Opcode.SCAL: _op_scal,
    Opcode.CHOL: _op_chol,
    Opcode.NRMI: _op_nrmi,
    Opcode.VMUL: _op_vmul,
}


# ---------------------------------------------------------------------------
# Static access analysis -- what each descriptor reads and writes.
# ---------------------------------------------------------------------------
#
# This is derived from the descriptor alone, so it is exactly the information a
# residency planner and a hazard checker have available in the hardware. It is
# used for (a) deciding which buffers a call must fetch and which it writes back,
# and (b) finding read-after-write pairs that the chain/wait tokens fail to order.

def _vec_span(base: int, n: int, inc: int, fmt: Format) -> Access:
    span = fmt.itemsize * (1 + (n - 1) * inc) if n else 0
    return Access(base, base + span)


def _mat_span(base: int, rows: int, cols: int, ld: int, fmt: Format) -> Access:
    span = fmt.itemsize * ((rows - 1) * ld + cols) if rows else 0
    return Access(base, base + span)


def accesses(d: Descriptor) -> tuple[list[Access], list[Access]]:
    """Return (reads, writes) as scratchpad byte ranges."""
    f = d.precision
    op = d.opcode
    if op is Opcode.GEMM:
        ar, ac = (d.k, d.m) if d.transpose.a else (d.m, d.k)
        br, bc = (d.n, d.k) if d.transpose.b else (d.k, d.n)
        reads = [_mat_span(d.a, ar, ac, d.lda, f), _mat_span(d.b, br, bc, d.ldb, f)]
        w = _mat_span(d.c, d.m, d.n, d.ldc, f)
        if d.accumulate:
            reads.append(w)
        return reads, [w]
    if op is Opcode.GEMV:
        xn = d.m if d.transpose.a else d.k
        yn = d.k if d.transpose.a else d.m
        reads = [_mat_span(d.a, d.m, d.k, d.lda, f), _vec_span(d.b, xn, d.ldb, f)]
        w = _vec_span(d.c, yn, d.ldc, f)
        if d.accumulate:
            reads.append(w)
        return reads, [w]
    if op is Opcode.TRSV:
        return ([_mat_span(d.a, d.m, d.m, d.lda, f), _vec_span(d.b, d.m, d.ldb, f)],
                [_vec_span(d.c, d.m, d.ldc, f)])
    if op is Opcode.AXPY:
        w = _vec_span(d.c, d.m, d.ldc, f)
        return [_vec_span(d.b, d.m, d.ldb, f), w], [w]
    if op is Opcode.SCAL:
        return [_vec_span(d.b, d.m, d.ldb, f)], [_vec_span(d.c, d.m, d.ldc, f)]
    if op in (Opcode.DOT,):
        return ([_vec_span(d.b, d.m, d.ldb, f), _vec_span(d.c, d.m, d.ldc, f)],
                [_vec_span(d.a, 1, 1, f)])
    if op is Opcode.NRMI:
        return [_vec_span(d.b, d.m, d.ldb, f)], [_vec_span(d.a, 1, 1, f)]
    if op is Opcode.PROJ:
        w = _vec_span(d.c, d.m, d.ldc, f)
        return ([_vec_span(d.a, d.m, 1, f), _vec_span(d.b, d.m, 1, f), w], [w])
    if op is Opcode.VMUL:
        return ([_vec_span(d.a, d.m, 1, f), _vec_span(d.b, d.m, d.ldb, f)],
                [_vec_span(d.c, d.m, d.ldc, f)])
    if op is Opcode.CHOL:
        span = _mat_span(d.a, d.m, d.m, d.lda, f)
        return [span], [span]
    raise NotImplementedError(f"no access model for {op}")
