# SPDX-License-Identifier: Apache-2.0
"""Argmind — sensor-agnostic on-device online adaptation and inference.

See DESIGN.md. Status: the reference and the C++ core are both built — streams,
alignment, two adapters, the guard, the pipeline and persistence — and the core runs
on emulated Cortex-M4 and Cortex-M55. What is absent is timing on real silicon.
"""

from .backbone import Backbone, mobilenet_v1
from .adapt import (Adapter, MultilayerPerceptron, NormalisedLeastMeanSquares,
                    RecursiveLeastSquares,
                    UpdateReport)
from .align import Aligner, Frame
from .guard import Bound, Check, Divergence, Guard, GuardReport
from .persist import Record, Store, fingerprint_of
from .pipeline import FrameMap, Pipeline, StepReport
from .stack import Stack
from .memory import ExemplarMemory, MatchReport
from .stream import Sample, Stream

__version__ = "0.0.1.dev0"
__all__ = ["ExemplarMemory", "MatchReport", "Stack", "Sample", "Stream", "Aligner", "Frame",
           "Adapter", "UpdateReport", "RecursiveLeastSquares",
           "NormalisedLeastMeanSquares", "MultilayerPerceptron",
           "Backbone", "mobilenet_v1",
           "Guard", "GuardReport", "Check", "Bound", "Divergence",
           "Pipeline", "FrameMap", "StepReport",
           "Store", "Record", "fingerprint_of"]
