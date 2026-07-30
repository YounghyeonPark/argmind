"""Hand-written solver programs.

These are v0 reference sequences, written by hand and checked against a reference
solver. They exist so that the code generator has a validated target to reproduce
-- the generator comes next and emits exactly this shape of program.
"""

from .admm import ADMMConfig, ADMMSolver, TerminationNorm

__all__ = ["ADMMConfig", "ADMMSolver", "TerminationNorm"]
