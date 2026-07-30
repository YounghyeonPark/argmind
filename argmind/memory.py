"""Buffers, lifetimes, generation tags, and the static scratchpad allocation plan.

Two memories, deliberately
--------------------------
The CM scratchpad is **not** hardware-coherent -- that is a design decision taken
for determinism and area. So this module models two distinct memories:

* **Host memory.** Where a user's buffer lives. ``Buffer.array`` is a real numpy
  view into it: no copy, no opaque handle. Mutate the view and the CM sees the
  mutation, because there is nothing in between. This is the property the C API
  must also have, so the simulator is built to have it rather than to fake it.
* **CM scratchpad.** A flat byte array with a *statically computed* allocation
  plan. Every offset is decided at build time, in declaration order, so two
  builds of the same program produce byte-identical address maps. There is no
  runtime allocator in the solver path and therefore no allocation-dependent
  latency.

The bridge between them is a descriptor-driven fetch, and what decides whether a
fetch happens is the **generation tag**.

Generation tags
---------------
A persistent buffer carries a monotonically increasing generation. The runtime
bumps it when the host rewrites the buffer. Each call carries a dirty mask. The
CM re-fetches a buffer only when its generation differs from the resident copy's,
or when the call names it in the dirty mask; same-generation buffers are trusted
as already resident.

The contract has a sharp edge and it is deliberately not hidden: because a buffer
is a plain pointer, **the runtime cannot detect a write it was not told about**.
There is no MMU in the way here and there is none in the C API either. Writing
through ``buffer.array`` without declaring the write leaves a stale copy resident
and the CM computes on old data -- silently, and in a control loop, dangerously.
The supported ways to write are therefore:

* :meth:`Buffer.write` -- writes and bumps the generation;
* ``with buffer.writable() as a:`` -- bumps on exit;
* :meth:`Buffer.mark_dirty` -- for when you really did write through the raw
  view and want to say so afterwards.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from enum import IntEnum
from typing import Iterator

import numpy as np

from .errors import ScratchpadExhausted, StaleHostCopy, UnmappedAddress
from .precision import Format, from_arithmetic, to_arithmetic


class Lifetime(IntEnum):
    """Allocation lifetime hint. Mirrors ``ARGMIND_STREAMING`` /
    ``ARGMIND_PERSISTENT`` in the C API."""

    #: Valid for one call. Always fetched; never trusted as resident.
    STREAMING = 0
    #: Survives across calls. Generation-tagged, fetched only on change.
    PERSISTENT = 1


#: Scratchpad allocations are aligned to the bank word. The design rule is
#: "bank word >= 64 bits", so 8 bytes is the minimum alignment that can ever be
#: correct, and using it keeps every allocation bank-aligned.
BANK_WORD_BYTES = 8


class Buffer:
    """A host allocation, addressable by the CM.

    Do not construct directly; use :meth:`argmind.device.Device.alloc`.
    """

    __slots__ = (
        "_device", "name", "fmt", "shape", "lifetime",
        "host_offset", "nbytes", "spad_offset",
        "_generation", "_resident_generation", "_resident", "_host_stale",
    )

    def __init__(
        self,
        device,
        name: str,
        fmt: Format,
        shape: tuple[int, ...],
        lifetime: Lifetime,
        host_offset: int,
        spad_offset: int,
    ) -> None:
        self._device = device
        self.name = name
        self.fmt = Format(fmt)
        self.shape = tuple(int(s) for s in shape)
        self.lifetime = Lifetime(lifetime)
        self.host_offset = int(host_offset)
        self.spad_offset = int(spad_offset)
        self.nbytes = int(np.prod(self.shape, dtype=np.int64)) * self.fmt.itemsize
        self._generation = 1
        #: Generation of the copy currently resident in scratchpad; 0 = nothing
        #: resident, so the first call always fetches.
        self._resident_generation = 0
        self._resident = False
        self._host_stale = False

    # -- the plain pointer ------------------------------------------------

    @property
    def array(self) -> np.ndarray:
        """A numpy **view** into host memory. Not a copy.

        Writing through this view is legal -- it is a pointer -- but does not
        bump the generation. See the module docstring.
        """
        raw = self._device.host_memory[self.host_offset:self.host_offset + self.nbytes]
        return raw.view(self.fmt.storage_dtype).reshape(self.shape)

    @property
    def spad_array(self) -> np.ndarray:
        """A view into the CM scratchpad copy. Exposed for inspection and tests."""
        raw = self._device.scratchpad[self.spad_offset:self.spad_offset + self.nbytes]
        return raw.view(self.fmt.storage_dtype).reshape(self.shape)

    def numbers(self) -> np.ndarray:
        """Host contents as numbers (decodes bf16 containers). Never raises."""
        return to_arithmetic(self.array, self.fmt)

    def read(self) -> np.ndarray:
        """Checked read: host contents as numbers, refusing a stale copy.

        Raises
        ------
        StaleHostCopy
            If the CM has written this buffer's scratchpad copy since the last
            writeback, i.e. the host bytes are older than the CM's. Ask for the
            buffer in a call's ``outputs`` if you want it synced.
        """
        if self.host_is_stale:
            raise StaleHostCopy(
                f"buffer {self.name!r}: the CM wrote its scratchpad copy "
                f"(resident generation {self._resident_generation}) and it has "
                f"not been written back to host memory. Name it in the call's "
                f"outputs=(...) to sync it, or use .numbers() to read the host "
                f"bytes as they are."
            )
        return to_arithmetic(self.array, self.fmt)

    @property
    def host_is_stale(self) -> bool:
        """True when scratchpad holds data the host copy does not."""
        return self._host_stale

    # -- writes that keep the generation honest ---------------------------

    def write(self, values) -> "Buffer":
        """Write `values` in place and bump the generation.

        In place: the host allocation is never reallocated and the pointer never
        changes, which is what makes the residency model work.
        """
        arr = self.array
        arr[...] = from_arithmetic(np.asarray(values), self.fmt).reshape(self.shape)
        self.mark_dirty()
        return self

    @contextmanager
    def writable(self) -> Iterator[np.ndarray]:
        """Context manager yielding the raw view; bumps the generation on exit."""
        try:
            yield self.array
        finally:
            self.mark_dirty()

    def mark_dirty(self) -> "Buffer":
        """Declare that the host has rewritten this buffer."""
        self._generation += 1
        self._host_stale = False
        return self

    # -- generation state -------------------------------------------------

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def resident_generation(self) -> int:
        return self._resident_generation

    @property
    def is_resident(self) -> bool:
        """True when a copy is in scratchpad and its generation still matches."""
        return self._resident and self._resident_generation == self._generation

    def _note_fetched(self) -> None:
        self._resident = True
        self._resident_generation = self._generation
        self._host_stale = False

    def _note_written_by_cm(self) -> None:
        """The CM wrote this buffer's scratchpad copy.

        The scratchpad copy is now newer than host memory, so both tags advance
        together: the resident copy is authoritative and must not be
        overwritten by a stale fetch on the next call.
        """
        self._generation += 1
        self._resident = True
        self._resident_generation = self._generation
        self._host_stale = True

    def _note_written_back(self) -> None:
        self._host_stale = False

    def __len__(self) -> int:
        return self.shape[0] if self.shape else 0

    def __repr__(self) -> str:
        return (
            f"<Buffer {self.name!r} {self.fmt} {self.shape} "
            f"{self.lifetime.name.lower()} spad=0x{self.spad_offset:05x} "
            f"gen={self._generation} resident_gen={self._resident_generation}>"
        )


@dataclass
class Allocation:
    """One entry in the static scratchpad allocation plan."""

    name: str
    offset: int
    nbytes: int
    fmt: Format
    shape: tuple[int, ...]
    lifetime: Lifetime

    @property
    def end(self) -> int:
        return self.offset + self.nbytes


class ScratchpadPlan:
    """The static allocation plan: a deterministic bump allocator plus a map.

    Deterministic by construction -- allocations are placed in declaration order
    with fixed alignment, so the address map is a pure function of the sequence
    of allocation requests. Nothing here runs at solve time.
    """

    def __init__(self, capacity_bytes: int, align: int = BANK_WORD_BYTES) -> None:
        self.capacity = int(capacity_bytes)
        self.align = int(align)
        self._next = 0
        self.allocations: list[Allocation] = []

    def place(
        self, name: str, nbytes: int, fmt: Format, shape: tuple[int, ...],
        lifetime: Lifetime,
    ) -> int:
        offset = self._next
        padded = (int(nbytes) + self.align - 1) // self.align * self.align
        if offset + padded > self.capacity:
            raise ScratchpadExhausted(
                f"static allocation of {name!r} ({nbytes} B, padded {padded} B) "
                f"does not fit: {offset} B already placed of {self.capacity} B "
                f"scratchpad. Working set is "
                f"{(offset + padded) / 1024:.1f} KiB against a "
                f"{self.capacity / 1024:.0f} KiB scratchpad."
            )
        self.allocations.append(
            Allocation(name, offset, int(nbytes), Format(fmt), tuple(shape), lifetime)
        )
        self._next = offset + padded
        return offset

    # -- reporting --------------------------------------------------------

    @property
    def used(self) -> int:
        return self._next

    @property
    def free(self) -> int:
        return self.capacity - self._next

    def persistent_bytes(self) -> int:
        return sum(a.nbytes for a in self.allocations
                   if a.lifetime is Lifetime.PERSISTENT)

    def lookup(self, address: int) -> Allocation:
        """Which allocation contains a scratchpad byte address."""
        for a in self.allocations:
            if a.offset <= address < a.end:
                return a
        raise UnmappedAddress(
            f"scratchpad address 0x{address:x} is in no allocation "
            f"(plan covers 0x0..0x{self.used:x})"
        )

    def report(self) -> str:
        """Human-readable residency report, as the profiler must emit."""
        lines = [
            f"scratchpad {self.capacity / 1024:.0f} KiB : "
            f"{self.used} B used ({100 * self.used / self.capacity:.1f}%), "
            f"{self.free} B free",
            f"{'offset':>9} {'bytes':>9} {'fmt':>5} {'life':>10}  name / shape",
        ]
        for a in self.allocations:
            lines.append(
                f"{a.offset:>9} {a.nbytes:>9} {str(a.fmt):>5} "
                f"{a.lifetime.name.lower():>10}  {a.name} {a.shape}"
            )
        return "\n".join(lines)


class _FlatHostMemory:
    """Adapter presenting the host heap as one indexable byte space.

    Buffers hold a single integer ``host_offset``, exactly as a C pointer would
    be a single address, so the heap is flattened into a contiguous address
    space. Blocks are pre-sized and never reallocated, so views stay valid.
    """

    def __init__(self, capacity_bytes: int) -> None:
        self._mem = np.zeros(int(capacity_bytes), dtype=np.uint8)
        self._next = 0

    def alloc(self, nbytes: int) -> int:
        nbytes = int(nbytes)
        pad = (nbytes + BANK_WORD_BYTES - 1) // BANK_WORD_BYTES * BANK_WORD_BYTES
        if self._next + pad > self._mem.size:
            raise ScratchpadExhausted(
                f"simulated host memory exhausted: needed {pad} B, "
                f"{self._mem.size - self._next} B free. Raise "
                f"Device(host_bytes=...)."
            )
        offset = self._next
        self._next += pad
        return offset

    def __getitem__(self, sl):
        return self._mem[sl]

    def __setitem__(self, sl, value) -> None:
        self._mem[sl] = value

    @property
    def used(self) -> int:
        return self._next

    @property
    def raw(self) -> np.ndarray:
        return self._mem
