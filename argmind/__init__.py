# SPDX-License-Identifier: Apache-2.0
"""Argmind — sensor-agnostic on-device online adaptation and inference.

See DESIGN.md. Status: the Python reference is complete — streams, alignment,
recursive least squares, the guard, the pipeline and persistence. The benchmark
harness and the C++ core come next.
"""

from .adapt import Adapter, RecursiveLeastSquares, UpdateReport
from .align import Aligner, Frame
from .guard import Bound, Check, Divergence, Guard, GuardReport
from .persist import Record, Store, fingerprint_of
from .pipeline import FrameMap, Pipeline, StepReport
from .stream import Sample, Stream

__version__ = "0.0.1.dev0"
__all__ = ["Sample", "Stream", "Aligner", "Frame",
           "Adapter", "UpdateReport", "RecursiveLeastSquares",
           "Guard", "GuardReport", "Check", "Bound", "Divergence",
           "Pipeline", "FrameMap", "StepReport",
           "Store", "Record", "fingerprint_of"]
