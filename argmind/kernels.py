"""Functional kernel implementations, one per descriptor opcode.

Every kernel takes the scratchpad byte array and a decoded descriptor, reads its
operands through strided views at the descriptor's declared addresses, and writes
its result back **in place**. No kernel copies an operand into a working buffer:
that would defeat the residency architecture it exists to serve, and a functional
model that copied would let a code generator that copies pass its tests.

Numeric behaviour
-----------------
Descriptor v0.1 carries **two** formats. ``prec_m`` is the bulk / matrix operand
format; ``prec_v`` is the format of vectors and results. Accumulation happens at
the width the MACIF contract declares for ``prec_m``, and the result is rounded to
``prec_v`` on store. So "fp16 matrix, fp32 accumulate, fp32 store" -- the
configuration PHASE0_FINDINGS 0.3 recommends and v0 could not express -- is
``precision=FP16, vec_precision=VecFormat.FP32``.

Two shape rules from v0.1 (rulings R6a and R6b) are visible here:

* elementwise kernels walk **M rows x N columns**, with ``N = 0`` read as
  ``N = 1``, so a v0 flat-vector descriptor behaves identically;
* a **zero leading dimension on a read operand is a broadcast**. The strided
  views below produce that naturally, and the descriptor constructor is what
  refuses a zero stride on a *write*.

``PROJ`` takes its input at ``a``, interleaved ``(lo, hi)`` bound pairs at ``b``
with ``ldb`` elements per pair, and writes ``c``. It is out-of-place in v0.1.

Reductions (DOT, NRMI) write a single element at ``a`` in ``prec_v``, reading
their vector operands in ``prec_m``.
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
    """A strided 1-D view. `inc` is in elements, as in BLAS.

    ``inc = 0`` is a **broadcast** read of one element (v0.1 ruling R6b). Legal on
    reads; the descriptor constructor refuses it on writes.
    """
    item = fmt.itemsize
    span = item * (1 + (n - 1) * inc) if n else 0
    raw = spad[base:base + span]
    return np.lib.stride_tricks.as_strided(
        raw.view(fmt.storage_dtype), shape=(n,), strides=(item * inc,), writeable=True
    )


def _grid(spad: np.ndarray, base: int, rows: int, cols: int, ld: int,
          fmt: Format) -> np.ndarray:
    """An elementwise operand view: `rows` x `cols`, `ld` elements per row.

    v0.1 ruling R6a. With ``cols == 1`` this reduces to ``_vector(base, rows, ld)``,
    which is exactly what a v0 flat-vector or strided-diagonal descriptor meant,
    so one function serves both and the diagonal walk (``ldc = n+1``) still works.
    """
    item = fmt.itemsize
    span = item * ((rows - 1) * ld + cols) if rows else 0
    raw = spad[base:base + span]
    return np.lib.stride_tricks.as_strided(
        raw.view(fmt.storage_dtype),
        shape=(rows, cols),
        strides=(item * ld, item),
        writeable=True,
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


def _accum(d: Descriptor, macro: MacroCapabilities) -> np.dtype:
    """Accumulator dtype: the wider of what prec_m and prec_v demand.

    The accumulator is a **MACIF declaration, not a descriptor field** (ruling
    R4). But a descriptor storing fp64 must not accumulate in fp32 just because
    its matrix operand is narrow, so the width is the max of the two contracts.
    """
    wm = macro.accumulator_width(d.prec_m)
    wv = macro.accumulator_width(d.prec_v)
    return np.dtype(np.float64) if max(wm, wv) >= 64 else np.dtype(np.float32)


def _store(dest: np.ndarray, values: np.ndarray, fmt: Format) -> None:
    dest[...] = from_arithmetic(values, fmt).reshape(dest.shape)


def _symmetrise(A: np.ndarray, structure: Structure) -> np.ndarray:
    """Mirror a symmetric operand from whichever triangle holds the data."""
    if structure is Structure.SYM_LOWER:
        return np.tril(A) + np.tril(A, -1).T
    if structure is Structure.SYM_UPPER:
        return np.triu(A) + np.triu(A, 1).T
    return A


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------

def _op_gemm(spad, d: Descriptor, macro: MacroCapabilities) -> None:
    """C := alpha * op(A) @ op(B)  [+ C].  A, B in prec_m; C in prec_v."""
    fm, fv, acc = d.prec_m, d.prec_v, _accum(d, macro)
    ar, ac = (d.k, d.m) if d.transpose.a else (d.m, d.k)
    br, bc = (d.n, d.k) if d.transpose.b else (d.k, d.n)
    A = to_arithmetic(_matrix(spad, d.a, ar, ac, d.lda, fm), fm).astype(acc)
    B = to_arithmetic(_matrix(spad, d.b, br, bc, d.ldb, fm), fm).astype(acc)
    if d.transpose.a:
        A = A.T
    if d.transpose.b:
        B = B.T
    A = _symmetrise(A, d.structure)
    out = _matrix(spad, d.c, d.m, d.n, d.ldc, fv)
    result = acc.type(d.alpha_f32) * (A @ B)
    if d.accumulate:
        result = result + to_arithmetic(out, fv).astype(acc)
    _store(out, result, fv)


def _op_gemv(spad, d: Descriptor, macro: MacroCapabilities) -> None:
    """y := alpha * op(A) @ x  [+ y].  A is m x k in prec_m; x, y in prec_v.

    This is the descriptor the mixed-precision ruling exists for: A is the bulk
    operand and the only one worth narrowing (m*k elements against k), while x
    and y stay wide so that a following fp32 TRSV can read y.
    """
    fm, fv, acc = d.prec_m, d.prec_v, _accum(d, macro)
    A = to_arithmetic(_matrix(spad, d.a, d.m, d.k, d.lda, fm), fm).astype(acc)
    A = _symmetrise(A, d.structure)
    if d.transpose.a:
        A = A.T
    xn, yn = A.shape[1], A.shape[0]
    x = to_arithmetic(_vector(spad, d.b, xn, d.ldb, fv), fv).astype(acc)
    yv = _vector(spad, d.c, yn, d.ldc, fv)
    result = acc.type(d.alpha_f32) * (A @ x)
    if d.accumulate:
        result = result + to_arithmetic(yv, fv).astype(acc)
    _store(yv, result, fv)


def _op_trsv(spad, d: Descriptor, macro: MacroCapabilities) -> None:
    """x := op(A)^-1 @ b, A triangular m x m in prec_m; b, x in prec_v.

    `c` may alias `b` (in place).
    """
    fm, fv, acc = d.prec_m, d.prec_v, _accum(d, macro)
    A = to_arithmetic(_matrix(spad, d.a, d.m, d.m, d.lda, fm), fm).astype(acc)
    lower = d.structure.reads_lower or d.structure is Structure.DENSE
    A = np.tril(A) if lower else np.triu(A)
    if d.unit_diag:
        np.fill_diagonal(A, acc.type(1.0))
    if d.transpose.a:
        A = A.T
        lower = not lower
    b = to_arithmetic(_vector(spad, d.b, d.m, d.ldb, fv), fv).astype(acc)
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
    _store(_vector(spad, d.c, n, d.ldc, fv), x, fv)


def _op_axpy(spad, d: Descriptor, macro: MacroCapabilities) -> None:
    """Y := alpha * X + Y over an M x N grid (v0.1 ruling R6a).

    ``N = 0`` reads as ``N = 1``, so a v0 flat-vector AXPY is unchanged. With
    ``N > 1`` this is the matrix-add v0 lacked, which is what removes the n = 255
    cap on an ``n x n`` elementwise operation.
    """
    fv, acc = d.prec_v, _accum(d, macro)
    rows, cols = d.rows, d.cols
    x = to_arithmetic(_grid(spad, d.b, rows, cols, d.ldb, fv), fv).astype(acc)
    yv = _grid(spad, d.c, rows, cols, d.ldc, fv)
    y = to_arithmetic(yv, fv).astype(acc)
    _store(yv, acc.type(d.alpha_f32) * x + y, fv)


def _op_scal(spad, d: Descriptor, macro: MacroCapabilities) -> None:
    """Y := alpha * X over an M x N grid. `c` may alias `b` for in-place scale."""
    fv, acc = d.prec_v, _accum(d, macro)
    rows, cols = d.rows, d.cols
    x = to_arithmetic(_grid(spad, d.b, rows, cols, d.ldb, fv), fv).astype(acc)
    _store(_grid(spad, d.c, rows, cols, d.ldc, fv),
           acc.type(d.alpha_f32) * x, fv)


def _op_dot(spad, d: Descriptor, macro: MacroCapabilities) -> None:
    """a[0] := alpha * (x . y), length m. Result is ONE element at `a`, in prec_v.

    The vectors are the bulk operands and so are read in prec_m; the scalar
    result is written in prec_v, which is what lets an fp16 vector reduce into an
    fp32 scalar the sequencer can compare without a widening step.
    """
    fm, fv, acc = d.prec_m, d.prec_v, _accum(d, macro)
    x = to_arithmetic(_vector(spad, d.b, d.m, d.ldb, fm), fm).astype(acc)
    y = to_arithmetic(_vector(spad, d.c, d.m, d.ldc, fm), fm).astype(acc)
    _store(_vector(spad, d.a, 1, 1, fv), np.array([acc.type(d.alpha_f32) * (x @ y)]),
           fv)


def _op_nrmi(spad, d: Descriptor, macro: MacroCapabilities) -> None:
    """a[0] := alpha * max|x_i|, length m. The infinity norm (v0.1 ruling R1).

    OSQP's termination criterion is stated entirely in infinity norms, so without
    this the in-fabric convergence test is a DIFFERENT test from the reference
    solver's -- measured at 3.7x worse control action for the same tolerance.
    The reduction is a compare-select tree: cheaper than DOT's, which needs a
    multiplier.
    """
    fm, fv, acc = d.prec_m, d.prec_v, _accum(d, macro)
    x = to_arithmetic(_vector(spad, d.b, d.m, d.ldb, fm), fm).astype(acc)
    val = acc.type(d.alpha_f32) * (np.max(np.abs(x)) if d.m else acc.type(0.0))
    _store(_vector(spad, d.a, 1, 1, fv), np.array([val]), fv)


def _op_proj(spad, d: Descriptor, macro: MacroCapabilities) -> None:
    """C := clip(A, lo, hi) over an M x N grid -- box projection, OUT OF PLACE.

    v0.1 ruling R2. The bounds are **interleaved (lo, hi) pairs** at ``b``, with
    ``ldb`` elements between successive pairs (normally 2). That frees ``a`` for
    the input, so projection needs three addresses rather than four and no
    fourth AGU. ``c`` may still alias ``a`` if in-place is what codegen wants.
    """
    fv, acc = d.prec_v, _accum(d, macro)
    rows, cols = d.rows, d.cols
    n = rows * cols
    pair_stride = d.ldb if d.ldb else 2
    bounds = _vector(spad, d.b, 2 * n, 1, fv) if pair_stride == 2 else None
    if bounds is not None:
        pairs = to_arithmetic(bounds, fv).astype(acc).reshape(n, 2)
    else:
        # General stride: walk pair starts `pair_stride` elements apart.
        starts = to_arithmetic(_vector(spad, d.b, n, pair_stride, fv), fv).astype(acc)
        hi_v = to_arithmetic(
            _vector(spad, d.b + fv.itemsize, n, pair_stride, fv), fv).astype(acc)
        pairs = np.stack([starts, hi_v], axis=1)
    lo, hi = pairs[:, 0], pairs[:, 1]
    src = to_arithmetic(_grid(spad, d.a, rows, cols, d.lda, fv),
                        fv).astype(acc).reshape(-1)
    out = _grid(spad, d.c, rows, cols, d.ldc, fv)
    _store(out, np.clip(src, lo, hi).reshape(rows, cols), fv)


def _op_chol(spad, d: Descriptor, macro: MacroCapabilities) -> None:
    """A := chol(A), factor written in place over the triangle `structure` names.

    The other triangle is left untouched, as an in-place ``potrf`` leaves it.
    Consumers select the triangle via the descriptor `structure` field.
    """
    fm, acc = d.prec_m, _accum(d, macro)
    view = _matrix(spad, d.a, d.m, d.m, d.lda, fm)
    A = to_arithmetic(view, fm).astype(acc)
    upper = d.structure in (Structure.TRIU, Structure.SYM_UPPER)
    A = (np.triu(A) + np.triu(A, 1).T) if upper else (np.tril(A) + np.tril(A, -1).T)
    L = np.linalg.cholesky(A)
    out = to_arithmetic(view, fm).astype(acc)
    if upper:
        idx = np.triu_indices(d.m)
        out[idx] = L.T[idx]
    else:
        idx = np.tril_indices(d.m)
        out[idx] = L[idx]
    _store(view, out, fm)


def _op_vmul(spad, d: Descriptor, macro: MacroCapabilities) -> None:
    """C := alpha * (A * B) elementwise over an M x N grid (v0.1 ruling R7).

    Two jobs, both of which v0 could not do. With ``lda`` a normal stride this
    carries a **per-constraint rho vector**, which sparse-form ADMM needs. With
    ``lda = 0`` it is a broadcast (ruling R6b), so a scalar held in scratchpad at
    the buffer's OWN precision multiplies a vector -- the mechanism that takes
    the reciprocal-mismatch failure of L5 from 1.5e-8 down to ~1e-16.
    """
    fv, acc = d.prec_v, _accum(d, macro)
    rows, cols = d.rows, d.cols
    u = to_arithmetic(_grid(spad, d.a, rows, cols, d.lda, fv), fv).astype(acc)
    v = to_arithmetic(_grid(spad, d.b, rows, cols, d.ldb, fv), fv).astype(acc)
    out = _grid(spad, d.c, rows, cols, d.ldc, fv)
    result = acc.type(d.alpha_f32) * u * v
    if d.accumulate:
        result = result + to_arithmetic(out, fv).astype(acc)
    _store(out, result, fv)


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
    """Return (reads, writes) as scratchpad byte ranges.

    Derived from the descriptor alone, so this is exactly the information the
    hardware has. Note that a zero-stride read (ruling R6b) yields a span of one
    element, which is correct: the hazard checker and the residency planner both
    then see a broadcast as touching one element, not n.
    """
    fm, fv = d.prec_m, d.prec_v
    op = d.opcode
    if op is Opcode.GEMM:
        ar, ac = (d.k, d.m) if d.transpose.a else (d.m, d.k)
        br, bc = (d.n, d.k) if d.transpose.b else (d.k, d.n)
        reads = [_mat_span(d.a, ar, ac, d.lda, fm), _mat_span(d.b, br, bc, d.ldb, fm)]
        w = _mat_span(d.c, d.m, d.n, d.ldc, fv)
        if d.accumulate:
            reads.append(w)
        return reads, [w]
    if op is Opcode.GEMV:
        xn = d.m if d.transpose.a else d.k
        yn = d.k if d.transpose.a else d.m
        reads = [_mat_span(d.a, d.m, d.k, d.lda, fm), _vec_span(d.b, xn, d.ldb, fv)]
        w = _vec_span(d.c, yn, d.ldc, fv)
        if d.accumulate:
            reads.append(w)
        return reads, [w]
    if op is Opcode.TRSV:
        return ([_mat_span(d.a, d.m, d.m, d.lda, fm), _vec_span(d.b, d.m, d.ldb, fv)],
                [_vec_span(d.c, d.m, d.ldc, fv)])
    if op is Opcode.AXPY:
        w = _mat_span(d.c, d.rows, d.cols, d.ldc, fv)
        return [_mat_span(d.b, d.rows, d.cols, d.ldb, fv), w], [w]
    if op is Opcode.SCAL:
        return ([_mat_span(d.b, d.rows, d.cols, d.ldb, fv)],
                [_mat_span(d.c, d.rows, d.cols, d.ldc, fv)])
    if op is Opcode.DOT:
        return ([_vec_span(d.b, d.m, d.ldb, fm), _vec_span(d.c, d.m, d.ldc, fm)],
                [_vec_span(d.a, 1, 1, fv)])
    if op is Opcode.NRMI:
        return [_vec_span(d.b, d.m, d.ldb, fm)], [_vec_span(d.a, 1, 1, fv)]
    if op is Opcode.PROJ:
        # Bounds are interleaved (lo,hi) pairs at `b` with `ldb` elements per
        # pair, so the span covers 2 elements for every element projected.
        n = d.elements
        stride = d.ldb if d.ldb else 2
        bspan = fv.itemsize * ((n - 1) * stride + 2) if n else 0
        bounds = Access(d.b, d.b + bspan)
        w = _mat_span(d.c, d.rows, d.cols, d.ldc, fv)
        return ([_mat_span(d.a, d.rows, d.cols, d.lda, fv), bounds], [w])
    if op is Opcode.VMUL:
        w = _mat_span(d.c, d.rows, d.cols, d.ldc, fv)
        reads = [_mat_span(d.a, d.rows, d.cols, d.lda, fv),
                 _mat_span(d.b, d.rows, d.cols, d.ldb, fv)]
        if d.accumulate:
            reads.append(w)
        return reads, [w]
    if op is Opcode.CHOL:
        span = _mat_span(d.a, d.m, d.m, d.lda, fm)
        return [span], [span]
    raise NotImplementedError(f"no access model for {op}")
