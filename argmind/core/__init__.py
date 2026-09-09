# SPDX-License-Identifier: Apache-2.0
"""`argmind.core` — the conversational agent, with no idea what platform it is on.

**Nothing in here names Windows, Android or a robot.** That is the test of whether "one core, an
interface per environment" is a real design or a slogan: if the core could not be written and tested
without naming a platform, the split would be imaginary, and finding that out costs one commit
instead of a restructure.

An environment supplies two small things — `Eyes` and `Hands` in `contract.py` — and this package
supplies everything above them:

| | |
|---|---|
| `contract.py` | `Widget`, `Eyes`, `Hands` — what an environment must provide |
| `llm.py` | the local model, with six settings each learned from a silent failure |
| `intent.py` | what starts a thought (`Trigger`, `Watch`) and what one produces (`Intent`) |
| `converse.py` | one call: what to say, and what — if anything — to do |
| `loop.py` | observe, poll, decide, perform — separately, because acting is decided |

**The rule everything is arranged around**, measured on this machine before a line of it was written:

> **The deterministic side says WHERE. The probabilistic side says WHAT.
> A model never emits a coordinate.**

Asking a vision model for a pixel took **24–82 s** and on a dense screen never answered; the OS
answered exactly in **41–266 ms**; a text model chose among named things in **67–85 ms**. The vision
model could *see* — its reasoning log read every 14-pixel filename — it simply could not do
arithmetic on what it saw.
"""

from .contract import Eyes, Hands, Widget
from .converse import converse, describe
from .intent import (Appeared, Effect, Intent, Perceived, Said, Trigger, Vanished, Watch,
                     WindowChanged)
from .llm import Local, Reply
from .loop import Done, Exchange, Session

__all__ = ["Appeared", "Done", "Effect", "Exchange", "Eyes", "Hands", "Intent", "Local",
           "Perceived", "Reply", "Said", "Session", "Trigger", "Vanished", "Watch", "Widget",
           "WindowChanged", "converse", "describe"]
