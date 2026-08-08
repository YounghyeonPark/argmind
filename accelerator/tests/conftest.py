# SPDX-License-Identifier: Apache-2.0
"""Make perf_model importable without turning it into an installed package."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "perf_model"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
