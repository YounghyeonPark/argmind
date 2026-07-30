"""Argmind CM -- runtime API and functional simulator.

The runtime API is the constitution of this product: it is identical across the
simulator, the open reference core and the commercial core, so code written
against the free stack runs unmodified on licensed silicon. Changing it is a
business decision, not an implementation detail.

Quick start::

    import numpy as np
    from argmind import Device, Format, Lifetime, macif

    dev = Device()                          # CM-100 baseline configuration
    a = dev.alloc("a", (4, 4), Format.FP64, Lifetime.PERSISTENT)
    a.write(np.eye(4))                      # writes in place, bumps generation
    a.array[0, 0] = 2.0                     # legal: it is a pointer
    a.mark_dirty()                          # ...but you must say so

Modules
-------
:mod:`argmind.device`
    ``Device``, ``DeviceConfig`` -- the runtime API and the functional simulator.
:mod:`argmind.memory`
    ``Buffer`` (a plain pointer), lifetimes, generation tags, the static
    scratchpad allocation plan.
:mod:`argmind.descriptor`
    Descriptor format v0, exactly as frozen, plus what implementing it revealed
    (``V0_LIMITATIONS``).
:mod:`argmind.macif`
    The MACIF precision contract, and the refusals that enforce it.
:mod:`argmind.program`
    Descriptor blocks, dependency-token assignment, hazard checking, and the
    iterative-solver structure.
:mod:`argmind.solvers`
    Hand-written reference solver programs.
"""

from . import kernels, macif, precision
from .descriptor import (
    DESC_BYTES,
    DESC_WORDS,
    FROZEN_V0_OPCODES,
    PROPOSED_OPCODES,
    V0_LIMITATIONS,
    Descriptor,
    Opcode,
    Structure,
    Transpose,
)
from .device import CM100_SCRATCHPAD_BYTES, BuiltProgram, CallStats, Device, DeviceConfig
from .errors import (
    ArgmindError,
    ConvergenceFailure,
    DependencyViolation,
    FieldOverflow,
    MacifError,
    MissingDependency,
    PrecisionNotDeclared,
    ScratchpadExhausted,
    StaleHostCopy,
    UnsupportedOpcode,
)
from .macif import MACIF_VERSION, FormatContract, MacroCapabilities
from .memory import Buffer, Lifetime, ScratchpadPlan
from .precision import Format
from .program import Program, SolverProgram, auto_chain, check_dependencies, concat

__version__ = "0.1.0.dev0"

__all__ = [
    "ArgmindError",
    "Buffer",
    "BuiltProgram",
    "CallStats",
    "CM100_SCRATCHPAD_BYTES",
    "ConvergenceFailure",
    "DESC_BYTES",
    "DESC_WORDS",
    "DependencyViolation",
    "Descriptor",
    "Device",
    "DeviceConfig",
    "FROZEN_V0_OPCODES",
    "FieldOverflow",
    "Format",
    "FormatContract",
    "Lifetime",
    "MACIF_VERSION",
    "MacifError",
    "MacroCapabilities",
    "MissingDependency",
    "Opcode",
    "PROPOSED_OPCODES",
    "PrecisionNotDeclared",
    "Program",
    "ScratchpadExhausted",
    "ScratchpadPlan",
    "SolverProgram",
    "StaleHostCopy",
    "Structure",
    "Transpose",
    "UnsupportedOpcode",
    "V0_LIMITATIONS",
    "__version__",
    "auto_chain",
    "check_dependencies",
    "concat",
    "kernels",
    "macif",
    "precision",
]
