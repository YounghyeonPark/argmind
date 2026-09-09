# SPDX-License-Identifier: Apache-2.0
"""Argmind — a local-model agent you talk to while you work.

    from argmind.core import Session          the conversation, with no platform in it
    from argmind.env.windows import ...       one environment's eyes and hands

Two packages and nothing else. `core/` knows what to say and what to do; `env/` knows how to
read a screen and move a mouse. An AST-walking test holds that the first names none of the
second, because "one core, an interface per environment" is a slogan otherwise.

**The rule everything is arranged around**, measured before a line of it was written:

> **The deterministic side says WHERE. The probabilistic side says WHAT.
> A model never emits a coordinate.**

Between 2026-08-08 and 2026-09-09 this name meant something else — a recursive-least-squares
library for on-device adaptation, ported to C++ for Cortex-M, with a classical-perception demo
on top. That tree was removed on 2026-09-09 rather than left unused, and it is intact under
the tags `before-removal-2026-09-09` and `argmind-v1`. `REBUILD.md` says why.
"""

__version__ = "0.1.0.dev0"
__all__ = ["__version__"]
