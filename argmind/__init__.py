# SPDX-License-Identifier: Apache-2.0
"""Argmind — sensor-agnostic on-device online adaptation and inference.

See DESIGN.md. Status: early. Streams, alignment and the first adapter exist;
the guard, the pipeline and persistence do not yet.
"""

from .adapt import Adapter, RecursiveLeastSquares, UpdateReport
from .align import Aligner, Frame
from .stream import Sample, Stream

__version__ = "0.0.1.dev0"
__all__ = ["Sample", "Stream", "Aligner", "Frame",
           "Adapter", "UpdateReport", "RecursiveLeastSquares"]
