# SPDX-License-Identifier: Apache-2.0
"""Exception hierarchy for the Argmind runtime.

Every failure mode that the hardware would detect (or that a code generator must
be prevented from producing) has its own exception type, so a caller can react
to a precision refusal differently from a scratchpad overflow.

Design rule: **nothing here ever degrades silently.** A macro that cannot honour
a requested precision raises; it does not substitute a nearby format. A
descriptor field that does not fit its bit width raises; it does not truncate.
"""

from __future__ import annotations


class ArgmindError(Exception):
    """Base class for all Argmind runtime errors."""


# --------------------------------------------------------------------------
# MACIF / precision
# --------------------------------------------------------------------------

class MacifError(ArgmindError):
    """Violation of the Macro-Agnostic Compute InterFace contract."""


class PrecisionNotDeclared(MacifError):
    """The compute macro does not declare the requested numeric format.

    Raised at *kernel build time*, never at execution time, and never resolved
    by substituting a different format of the same width.
    """


# --------------------------------------------------------------------------
# Descriptors
# --------------------------------------------------------------------------

class DescriptorError(ArgmindError):
    """Malformed descriptor."""


class FieldOverflow(DescriptorError):
    """A descriptor field value does not fit the bit width frozen in format v0."""


class UnsupportedOpcode(DescriptorError):
    """Opcode is not part of the frozen v0 set and strict-v0 mode is enabled."""


# --------------------------------------------------------------------------
# Memory
# --------------------------------------------------------------------------

class MemoryError_(ArgmindError):
    """Base class for allocation / addressing failures."""


class ScratchpadExhausted(MemoryError_):
    """The static allocation plan does not fit the configured scratchpad."""


class UnmappedAddress(MemoryError_):
    """A descriptor references a scratchpad address outside every allocation."""


class StaleHostCopy(MemoryError_):
    """The CM has newer data in scratchpad than the host copy holds.

    Raised by :meth:`argmind.memory.Buffer.read`, which is the checked way to
    look at a buffer. The raw ``.array`` view deliberately does not raise: it is
    a pointer, and a pointer shows you the bytes that are there.
    """


# --------------------------------------------------------------------------
# Dependencies / execution
# --------------------------------------------------------------------------

class DependencyError(ArgmindError):
    """Base class for chain/wait-token problems."""


class DependencyViolation(DependencyError):
    """A descriptor waited on a token that no earlier descriptor produced."""


class MissingDependency(DependencyError):
    """Two descriptors have a read-after-write hazard with no token ordering it.

    This is the bug class that only bites once the hardware runs descriptors
    ahead of each other, so it is silent on an in-order model unless explicitly
    checked. The simulator checks it by default.
    """


class ConvergenceFailure(ArgmindError):
    """An iterative program hit its iteration limit without meeting tolerance."""
