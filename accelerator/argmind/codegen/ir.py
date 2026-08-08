# SPDX-License-Identifier: Apache-2.0
"""The code generator's intermediate representation, and the passes over it.

The IR is deliberately thin: a :class:`Kernel` is **a descriptor whose operands
are named tensors instead of addresses**. There is no general tensor algebra here
and there should not be -- one solver family done properly beats broad shallow
coverage, and every extra IR level is a level where the generated program can
diverge from what the format can express.

What the thinness buys is that the two hard questions are answerable *before*
addresses exist:

* **Formats.** A tensor carries a :class:`~argmind.precision.Format`; a
  descriptor carries ``prec_m`` and ``prec_v``. :func:`lower` derives the second
  from the first and **refuses** combinations descriptor v0.1 cannot encode. This
  is how a code generator finds the edges of the format instead of assuming them.
* **Copies.** Tensor-level reads and writes let :func:`elide_pure_copies` delete
  staging copies, which is contract rule C6. Doing that after allocation would be
  too late: the deleted buffer must never be allocated at all.

Allocation order is the tensor declaration order, so the address map is a pure
function of the graph. Two builds of the same graph are byte-identical, which is
contract rule C5.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import IntEnum
from typing import Iterable, Sequence

from ..descriptor import Descriptor, Opcode, Structure, Transpose
from ..memory import Buffer, Lifetime
from ..precision import Format, VecFormat
from .problem import UnexpressiblePrecision


class Source(IntEnum):
    """Where a tensor's contents come from. This is the residency classification.

    It is declared by the graph builder from the problem description and then
    **cross-checked against the emitted dataflow**: a tensor declared
    :attr:`INTERNAL` that is read before it is written is a codegen bug, and the
    check catches it (see :mod:`argmind.codegen.residency`).
    """

    #: Function of the plant model. Host writes it once; resident thereafter.
    STRUCTURAL = 0
    #: Function of the current state. Host rewrites it every call.
    INSTANCE = 1
    #: Warm-start state. Host writes on reset, the CM updates it in place.
    ITERATE = 2
    #: Produced by the setup block on the CM. Host memory never holds it, so it
    #: must never be fetched -- a fetch would overwrite the factor with zeros.
    DERIVED = 3
    #: Written before read inside the solve. Never fetched, never written back.
    INTERNAL = 4
    #: A code-generation constant placed in scratchpad (sigma, rho, 1/rho).
    CONSTANT = 5

    @property
    def host_writes(self) -> bool:
        return self in (Source.STRUCTURAL, Source.INSTANCE, Source.ITERATE,
                        Source.CONSTANT)

    @property
    def lifetime(self) -> Lifetime:
        """Every tensor in a statically scheduled solve is PERSISTENT.

        Even the temporaries: they live at a fixed scratchpad address for the life
        of the program and are never re-fetched, which is a stronger property than
        ``STREAMING`` describes. ``STREAMING`` means "fetch every call", and
        fetching a temporary would be exactly the copy the architecture exists to
        avoid.
        """
        return Lifetime.PERSISTENT


@dataclass(frozen=True)
class Tensor:
    """A named, shaped, formatted region of scratchpad, before it has an address."""

    name: str
    shape: tuple[int, ...]
    source: Source
    fmt: Format
    #: True when the host reads this back (it appears in a call's outputs).
    host_visible: bool = False
    #: Which QP data block this carries, if any. Used by the report only.
    role: str = ""

    @property
    def size(self) -> int:
        n = 1
        for s in self.shape:
            n *= int(s)
        return n

    @property
    def nbytes(self) -> int:
        return self.size * self.fmt.itemsize

    def __str__(self) -> str:
        return f"{self.name}{self.shape}:{self.fmt}"


@dataclass(frozen=True)
class Ref:
    """A reference to a tensor, optionally at an element offset into it.

    The offset exists so that a generator can address a sub-block (a stage, a
    horizon slice, the trailing rows of a stacked constraint matrix) without
    allocating a separate buffer. It is in **elements**, converted to bytes at
    lowering time using the tensor's own format.
    """

    tensor: Tensor
    offset: int = 0

    def __str__(self) -> str:
        return self.tensor.name if not self.offset else \
            f"{self.tensor.name}+{self.offset}"


#: Which operand slots each opcode reads and writes, at tensor granularity.
#: Mirrors the per-opcode operand conventions in
#: :mod:`argmind.descriptor` and is cross-checked against
#: :func:`argmind.kernels.accesses` by a test, because two encodings of the same
#: convention that drift apart is exactly the class of bug the hazard checker
#: cannot see.
_READ_SLOTS: dict[Opcode, tuple[str, ...]] = {
    Opcode.GEMM: ("a", "b"),
    Opcode.GEMV: ("a", "b"),
    Opcode.TRSV: ("a", "b"),
    Opcode.AXPY: ("b", "c"),
    Opcode.SCAL: ("b",),
    Opcode.VMUL: ("a", "b"),
    Opcode.DOT: ("b", "c"),
    Opcode.NRMI: ("b",),
    Opcode.PROJ: ("a", "b"),
    Opcode.CHOL: ("a",),
}

_WRITE_SLOT: dict[Opcode, str] = {
    Opcode.GEMM: "c",
    Opcode.GEMV: "c",
    Opcode.TRSV: "c",
    Opcode.AXPY: "c",
    Opcode.SCAL: "c",
    Opcode.VMUL: "c",
    Opcode.DOT: "a",
    Opcode.NRMI: "a",
    Opcode.PROJ: "c",
    Opcode.CHOL: "a",
}

#: Slots read at ``prec_m`` (the bulk operand format). Everything else a
#: descriptor touches is at ``prec_v``.
#:
#: Note what this table says about the elementwise opcodes: they have **no**
#: prec_m slot, so they carry exactly ONE format, not two. That is the limit
#: :func:`lower` reports on, and it is not visible from the bit layout -- the
#: field exists, it is simply unused by those opcodes.
_PREC_M_SLOTS: dict[Opcode, tuple[str, ...]] = {
    Opcode.GEMM: ("a", "b"),
    Opcode.GEMV: ("a",),
    Opcode.TRSV: ("a",),
    Opcode.CHOL: ("a",),
    Opcode.DOT: ("b", "c"),
    Opcode.NRMI: ("b",),
    Opcode.AXPY: (),
    Opcode.SCAL: (),
    Opcode.VMUL: (),
    Opcode.PROJ: (),
}

#: Opcodes whose ``prec_m`` field is not read by the datapath at all.
SINGLE_FORMAT_OPCODES: frozenset[Opcode] = frozenset(
    op for op, slots in _PREC_M_SLOTS.items() if not slots)


@dataclass(frozen=True)
class Kernel:
    """One descriptor, with tensors where the addresses will go.

    Field names match :class:`argmind.descriptor.Descriptor` exactly, so lowering
    is a substitution rather than a translation. ``n = 0`` is kept meaning
    ``n = 1`` for one-dimensional elementwise ops, as in the wire format.
    """

    opcode: Opcode
    a: Ref | None = None
    b: Ref | None = None
    c: Ref | None = None
    m: int = 0
    k: int = 0
    n: int = 0
    lda: int = 1
    ldb: int = 1
    ldc: int = 1
    alpha: float = 1.0
    transpose: Transpose = Transpose.NONE
    structure: Structure = Structure.DENSE
    accumulate: bool = False
    unit_diag: bool = False
    label: str = ""

    # -- operands ---------------------------------------------------------

    def slot(self, name: str) -> Ref | None:
        return getattr(self, name)

    @property
    def reads(self) -> tuple[Tensor, ...]:
        slots = list(_READ_SLOTS[self.opcode])
        # `accumulate` turns the destination into a read operand as well. AXPY and
        # CHOL read theirs unconditionally and are already in the table.
        dest = _WRITE_SLOT[self.opcode]
        if self.accumulate and dest not in slots:
            slots.append(dest)
        out: list[Tensor] = []
        for s in slots:
            r = self.slot(s)
            if r is not None and r.tensor not in out:
                out.append(r.tensor)
        return tuple(out)

    @property
    def writes(self) -> tuple[Tensor, ...]:
        r = self.slot(_WRITE_SLOT[self.opcode])
        return () if r is None else (r.tensor,)

    @property
    def touched(self) -> tuple[Tensor, ...]:
        seen: list[Tensor] = []
        for s in ("a", "b", "c"):
            r = self.slot(s)
            if r is not None and r.tensor not in seen:
                seen.append(r.tensor)
        return tuple(seen)

    def rename(self, old: Tensor, new: Tensor) -> "Kernel":
        """A copy with every reference to `old` pointing at `new`.

        Used by copy elision, which removes a staging buffer by redirecting its
        consumers at the buffer that was being copied *from*.
        """
        sub = {}
        for s in ("a", "b", "c"):
            r = self.slot(s)
            if r is not None and r.tensor is old:
                sub[s] = Ref(new, r.offset)
        return replace(self, **sub) if sub else self

    # -- classification ---------------------------------------------------

    @property
    def is_pure_copy(self) -> bool:
        """``SCAL`` with unit alpha into a different buffer: a staging copy.

        This is what contract rule C6 forbids. The value is not in the descriptor
        count -- one SCAL is cheap -- it is in the buffer it forces the allocator
        to reserve, and in the fact that a generator willing to emit one is a
        generator that will emit them everywhere.
        """
        return (
            self.opcode is Opcode.SCAL
            and float(self.alpha) == 1.0
            and self.b is not None and self.c is not None
            and self.b.tensor is not self.c.tensor
            and self.b.offset == self.c.offset == 0
            and self.ldb == self.ldc
            and self.b.tensor.shape == self.c.tensor.shape
            and self.b.tensor.fmt is self.c.tensor.fmt
        )

    @property
    def shape_key(self) -> tuple:
        """Everything in the descriptor **except** the three base addresses.

        This is one hardware-descriptor-generator *template*: the AGUs supply the
        addresses, so two kernels with the same shape key can be issued from one
        armed template and two with different keys cannot. Contract rule C4 counts
        these.
        """
        return (
            self.opcode, self.m, self.k, self.n, self.lda, self.ldb, self.ldc,
            float(self.alpha), self.transpose, self.structure,
            bool(self.accumulate), bool(self.unit_diag),
            tuple(
                (s, self.slot(s).tensor.fmt) for s in ("a", "b", "c")
                if self.slot(s) is not None
            ),
        )

    @property
    def shape_key_without_alpha(self) -> tuple:
        """:attr:`shape_key` ignoring the w7 immediate.

        Two kernels that differ only in alpha could share a template slot if the
        generator re-armed one word instead of the whole template. Counted so the
        cost of the cheap mitigation is a number rather than a suggestion.
        """
        k = list(self.shape_key)
        k[7] = None                      # the alpha slot; see shape_key
        return tuple(k)

    def __str__(self) -> str:
        ops = ",".join(f"{s}={self.slot(s)}" for s in ("a", "b", "c")
                       if self.slot(s) is not None)
        return f"{self.opcode.name}({ops}) {self.label}"


@dataclass
class Block:
    """An ordered list of kernels that the sequencer issues as one unit."""

    name: str
    kernels: list[Kernel] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.kernels)

    def __iter__(self):
        return iter(self.kernels)


@dataclass
class KernelGraph:
    """The whole generated solver, before allocation.

    ``tensors`` is in **declaration order**, and declaration order is allocation
    order, so it is part of the generated artefact rather than an implementation
    detail.
    """

    name: str
    tensors: list[Tensor] = field(default_factory=list)
    blocks: dict[str, Block] = field(default_factory=dict)
    block_order: tuple[str, ...] = ("setup", "iteration", "residual")

    # -- construction -----------------------------------------------------

    def declare(
        self, name: str, shape: int | Sequence[int], source: Source, fmt: Format,
        *, host_visible: bool = False, role: str = "",
    ) -> Tensor:
        shape = (int(shape),) if isinstance(shape, int) else tuple(int(s) for s in shape)
        if any(t.name == name for t in self.tensors):
            raise ValueError(f"tensor {name!r} already declared")
        t = Tensor(name=name, shape=shape, source=source, fmt=Format(fmt),
                   host_visible=host_visible, role=role)
        self.tensors.append(t)
        return t

    def block(self, name: str) -> Block:
        return self.blocks.setdefault(name, Block(name))

    def emit(self, block: str, kernel: Kernel) -> Kernel:
        self.block(block).kernels.append(kernel)
        return kernel

    # -- queries ----------------------------------------------------------

    @property
    def all_kernels(self) -> tuple[Kernel, ...]:
        out: list[Kernel] = []
        for b in self.block_order:
            if b in self.blocks:
                out.extend(self.blocks[b].kernels)
        return tuple(out)

    def kernels_using(self, t: Tensor) -> tuple[Kernel, ...]:
        return tuple(k for k in self.all_kernels if t in k.touched)

    @property
    def bytes_total(self) -> int:
        return sum(t.nbytes for t in self.tensors)

    def listing(self) -> str:
        lines = []
        for b in self.block_order:
            if b not in self.blocks:
                continue
            lines.append(f"{b}:")
            for i, k in enumerate(self.blocks[b].kernels):
                lines.append(f"  {i:>3}  {k}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pass: staging-copy elision (contract rule C6)
# ---------------------------------------------------------------------------

@dataclass
class ElisionResult:
    graph: KernelGraph
    #: (removed tensor, tensor it was aliased onto, bytes reclaimed)
    eliminated: tuple[tuple[str, str, int], ...] = ()

    @property
    def descriptors_saved(self) -> int:
        return len(self.eliminated)

    @property
    def bytes_saved(self) -> int:
        return sum(b for _, _, b in self.eliminated)


def elide_pure_copies(graph: KernelGraph) -> ElisionResult:
    """Delete ``dst = src`` staging copies by rewriting consumers onto ``src``.

    The transformation is legal exactly when

    1. the copy is a pure copy (same shape, same format, unit alpha);
    2. ``src`` is an :attr:`Source.INTERNAL` temporary -- recomputed inside the
       solve, so nothing outside depends on its value surviving;
    3. after the copy, ``src`` is not read again anywhere in the graph;
    4. ``dst`` is INTERNAL and not host-visible.

    Condition 3 is why this is a *pass* and not a peephole: on the ADMM residual
    block the copy ``rp = Ax`` is legal to remove only because ``|Ax|`` is
    computed *before* it, and a peephole looking at two adjacent descriptors
    cannot see that.

    On QUAD-MPC this removes two descriptors and two buffers. Both of them are in
    the hand-written program, which is the point: the rule is worth enforcing
    mechanically because a careful human wrote them anyway.
    """
    eliminated: list[tuple[str, str, int]] = []
    changed = True
    while changed:
        changed = False
        flat: list[tuple[str, int, Kernel]] = [
            (b, i, k)
            for b in graph.block_order if b in graph.blocks
            for i, k in enumerate(graph.blocks[b].kernels)
        ]
        for pos, (bname, idx, k) in enumerate(flat):
            if not k.is_pure_copy:
                continue
            src, dst = k.b.tensor, k.c.tensor
            if src.source is not Source.INTERNAL or dst.source is not Source.INTERNAL:
                continue
            if dst.host_visible or src.host_visible:
                continue
            later = [kk for _, _, kk in flat[pos + 1:]]
            if any(src in kk.reads or src in kk.writes for kk in later):
                continue
            # Legal. Drop the copy and point every later reference at `src`.
            graph.blocks[bname].kernels.pop(idx)
            for b in graph.block_order:
                if b not in graph.blocks:
                    continue
                ks = graph.blocks[b].kernels
                for j, kk in enumerate(ks):
                    ks[j] = kk.rename(dst, src)
            graph.tensors = [t for t in graph.tensors if t is not dst]
            eliminated.append((dst.name, src.name, dst.nbytes))
            changed = True
            break
    return ElisionResult(graph=graph, eliminated=tuple(eliminated))


# ---------------------------------------------------------------------------
# Pass: allocation
# ---------------------------------------------------------------------------

def allocate(graph: KernelGraph, device, prefix: str = "") -> dict[str, Buffer]:
    """Place every tensor, in declaration order, on `device`.

    No sizing decision is made here: the device's own
    :class:`~argmind.memory.ScratchpadPlan` is a bump allocator that raises
    :class:`~argmind.errors.ScratchpadExhausted` when the working set does not
    fit. A code generator must not silently spill to host memory in a
    determinism-critical path, so not fitting is an error at build time.
    """
    buffers: dict[str, Buffer] = {}
    for t in graph.tensors:
        buffers[t.name] = device.alloc(
            f"{prefix}{t.name}" if prefix else t.name,
            t.shape, t.fmt, t.source.lifetime,
        )
    return buffers


# ---------------------------------------------------------------------------
# Pass: lowering, and the precision derivation that goes with it
# ---------------------------------------------------------------------------

def _address(ref: Ref, buffers: dict[str, Buffer]) -> int:
    buf = buffers[ref.tensor.name]
    return buf.spad_offset + ref.offset * ref.tensor.fmt.itemsize


def derive_precisions(kernel: Kernel) -> tuple[Format, VecFormat]:
    """``(prec_m, prec_v)`` for a kernel, from its operands' formats.

    Refuses, rather than picking one, when the operands need more than the format
    can carry. Two distinct refusals, and they are different findings:

    * **Two bulk operands in different formats** (a GEMM of an fp16 and an fp32
      matrix). Genuinely three formats, which ruling R4 states as the limit.
    * **An elementwise op whose operands are in different formats.** These
      opcodes have no bulk operand, so ``prec_m`` is unused and they carry ONE
      format -- an fp32 ``+=`` fp16 matrix add is inexpressible even though only
      two formats are involved. That is not stated anywhere in the format and it
      is found here.
    """
    op = kernel.opcode
    m_slots = _PREC_M_SLOTS[op]
    m_fmts, v_fmts = [], []
    for s in ("a", "b", "c"):
        r = kernel.slot(s)
        if r is None:
            continue
        (m_fmts if s in m_slots else v_fmts).append((s, r.tensor))

    def one(items, what) -> Format:
        fmts = {t.fmt for _, t in items}
        if len(fmts) > 1:
            detail = ", ".join(f"{s}={t.name}:{t.fmt}" for s, t in items)
            raise UnexpressiblePrecision(
                f"{op.name}: {what} operands are in {len(fmts)} different "
                f"formats ({detail}). Descriptor v0.1 carries exactly two formats "
                f"(prec_m for the bulk operand, prec_v for vectors and results)"
                + (
                    f", and {op.name} is elementwise so it has no bulk operand at "
                    f"all -- prec_m is unused and the descriptor therefore carries "
                    f"exactly ONE format. A mixed-format elementwise operation is "
                    f"not expressible in v0.1."
                    if op in SINGLE_FORMAT_OPCODES else
                    ". Split the operation into two descriptors, or store one "
                    "operand at the other's format."
                )
            )
        return next(iter(fmts))

    if m_fmts:
        prec_m = one(m_fmts, "bulk/matrix")
        if v_fmts:
            prec_v_fmt = one(v_fmts, "vector/result")
            prec_v = (VecFormat.SAME if prec_v_fmt is prec_m
                      else VecFormat.of(prec_v_fmt))
        else:
            prec_v = VecFormat.SAME
    else:
        # No bulk operand: one format for everything, carried in prec_m with
        # prec_v = SAME, which is exactly what every v0 descriptor meant.
        prec_m = one(v_fmts, "elementwise")
        prec_v = VecFormat.SAME
    return prec_m, prec_v


#: Opcodes for which the ``accumulate`` bit has a defined meaning. Every other
#: opcode **silently ignores it** (see :mod:`argmind.codegen.findings` G4), so two
#: different bit patterns decode to the same operation. The generator refuses to
#: emit the meaningless one rather than relying on the ignoring.
ACCUMULATE_DEFINED_FOR: frozenset[Opcode] = frozenset(
    (Opcode.GEMM, Opcode.GEMV, Opcode.VMUL))


def lower(kernel: Kernel, buffers: dict[str, Buffer]) -> Descriptor:
    """One kernel -> one descriptor. Addresses resolved, precisions derived."""
    if kernel.accumulate and kernel.opcode not in ACCUMULATE_DEFINED_FOR:
        raise UnexpressiblePrecision(
            f"{kernel.opcode.name} with accumulate set: the bit is only defined "
            f"for {sorted(op.name for op in ACCUMULATE_DEFINED_FOR)}. Every other "
            f"opcode's kernel ignores it, so this descriptor would decode "
            f"identically to the same descriptor without the bit -- two encodings, "
            f"one meaning, and no error. AXPY already accumulates by definition; "
            f"for a scaled accumulate use AXPY, not SCAL with the bit set.")
    prec_m, prec_v = derive_precisions(kernel)
    addr = {s: (0 if kernel.slot(s) is None else _address(kernel.slot(s), buffers))
            for s in ("a", "b", "c")}
    return Descriptor(
        opcode=kernel.opcode,
        precision=prec_m,
        vec_precision=prec_v,
        a=addr["a"], b=addr["b"], c=addr["c"],
        m=kernel.m, k=kernel.k, n=kernel.n,
        lda=kernel.lda, ldb=kernel.ldb, ldc=kernel.ldc,
        alpha=kernel.alpha,
        transpose=kernel.transpose,
        structure=kernel.structure,
        accumulate=kernel.accumulate,
        unit_diag=kernel.unit_diag,
        label=kernel.label,
    )


def lower_block(block: Block, buffers: dict[str, Buffer]) -> list[Descriptor]:
    return [lower(k, buffers) for k in block.kernels]


def tensor_shapes(kernels: Iterable[Kernel]) -> dict[tuple, list[int]]:
    """Group kernel indices by :attr:`Kernel.shape_key` -- i.e. by template."""
    groups: dict[tuple, list[int]] = {}
    for i, k in enumerate(kernels):
        groups.setdefault(k.shape_key, []).append(i)
    return groups
