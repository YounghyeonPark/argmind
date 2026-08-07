# SPDX-License-Identifier: Apache-2.0
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
    Descriptor format **v0.1**: the bit layout, the ruling on each gap that
    implementing v0 exposed (``V0_LIMITATIONS`` paired with ``V0_1_RULINGS``),
    the scratchpad ruling (``SCRATCHPAD_RULING``) and the re-checked sequencer
    contract (``SEQUENCER_RECHECK``).
:mod:`argmind.macif`
    The MACIF precision contract, and the refusals that enforce it.
:mod:`argmind.program`
    Descriptor blocks, dependency-token assignment, hazard checking, and the
    iterative-solver structure.
:mod:`argmind.solvers`
    Hand-written reference solver programs.
:mod:`argmind.codegen`
    The code generator: a problem description (a QP, or a CVXPY problem) in, a
    descriptor program plus a static scratchpad allocation plan plus a residency
    plan out, with a compliance report against the codegen contract. Imported
    explicitly (``from argmind.codegen import generate``) rather than re-exported
    here, because it is a tool built *on* the runtime API and not part of it.
"""

from . import kernels, macif, precision
from .descriptor import (
    DESC_BYTES,
    DESC_WORDS,
    DESCRIPTOR_VERSION,
    FROZEN_V0_OPCODES,
    N_TOKENS,
    SCRATCHPAD_RULING,
    SEQUENCER_RECHECK,
    V0_1_ADDED_OPCODES,
    V0_1_BIT_BUDGET,
    V0_1_OPCODES,
    V0_1_RULINGS,
    V0_LIMITATIONS,
    Descriptor,
    Opcode,
    Structure,
    Transpose,
)
from .device import (
    CM100_SCRATCHPAD_BYTES,
    CM100_SCRATCHPAD_BYTES_V0,
    BuiltProgram,
    CallStats,
    Device,
    DeviceConfig,
)
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
from .precision import Format, VecFormat
from .program import (
    Program,
    SolverProgram,
    auto_chain,
    auto_chain_v0,
    check_dependencies,
    concat,
    peak_live_tokens,
)

__version__ = "0.1.0.dev0"

__all__ = [
    "ArgmindError",
    "Buffer",
    "BuiltProgram",
    "CallStats",
    "CM100_SCRATCHPAD_BYTES",
    "CM100_SCRATCHPAD_BYTES_V0",
    "ConvergenceFailure",
    "DESC_BYTES",
    "DESCRIPTOR_VERSION",
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
    "N_TOKENS",
    "Opcode",
    "PrecisionNotDeclared",
    "Program",
    "SCRATCHPAD_RULING",
    "SEQUENCER_RECHECK",
    "ScratchpadExhausted",
    "ScratchpadPlan",
    "SolverProgram",
    "StaleHostCopy",
    "Structure",
    "Transpose",
    "UnsupportedOpcode",
    "V0_1_ADDED_OPCODES",
    "V0_1_BIT_BUDGET",
    "V0_1_OPCODES",
    "V0_1_RULINGS",
    "V0_LIMITATIONS",
    "VecFormat",
    "__version__",
    "auto_chain",
    "auto_chain_v0",
    "check_dependencies",
    "concat",
    "peak_live_tokens",
    "kernels",
    "macif",
    "precision",
]
