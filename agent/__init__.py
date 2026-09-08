# SPDX-License-Identifier: Apache-2.0
"""`argmind.agent` — an LLM-driven agent for the PC, with argmind's memory as its cache.

**What the owner asked for**, 2026-09-08: an agent that works with an LLM inside the I/O environment
a PC provides — screen, webcam, microphone, keyboard and mouse in; screen, keyboard, mouse and
speaker out — and *"캐시를 만들어서 반복적인 작업은 점점 빠르게 동작할수 있게끔 할것."*

**The cache is the part that is argmind's.** An LLM supplies language and judgement; nothing here
competes with that. What it does not supply is a memory that outlives a call and gets *cheaper* with
repetition, and `ExemplarMemory` is already a bounded, content-addressed, threshold-matched store
with one flat state block. `agent/cache.py` adds the value side and the eviction honesty.

**It was measured before it was built.** A screen situation has to come back looking like itself or
there is nothing to cache. On 117 real captures over 60 s, **20.4 % of frame pairs more than ten
seconds apart score ≥ 0.995**, and the share barely moves from 0.95 to 0.995 — the distribution is
two-humped, so a threshold exists. That is the opposite of what the same machinery does on a camera,
where a moving object retains **0 of 25** under load. The table and its limits are in
`agent/situation.py`.

**What is built, and what is absent rather than unfinished:**

| | |
|---|---|
| screen in | built (`PIL.ImageGrab`, 34 ms for 2560×1440 on this machine) |
| webcam in | built (`cv2`) |
| keyboard / mouse in | built (`user32`, sampled — not hooked) |
| keyboard / mouse out | built (`user32` `SendInput`), **dry-run by default** |
| **microphone in** | **absent** — no capture library in this interpreter |
| **speaker out** | **absent** — same reason |
| LLM | `ClaudeBrain` written against the official SDK and **never executed**: no package, no key |

Actuation defaults to `live=False`. This drives the real mouse and types into whatever window has
focus; there is no undo, so the default records what it would have done and `--live` is a decision
somebody makes on purpose.
"""

from .act import ActionRefused, Actuator
from .brain import Brain, ClaudeBrain, Decision, ScriptedBrain
from .cache import ActionCache, Hit
from .loop import AgentLoop, Step
from .senses import KeyboardMouseSense, MicrophoneSense, ScreenSense, WebcamSense
from .situation import Situation, SituationSense

__all__ = [
    "ActionCache", "ActionRefused", "Actuator", "AgentLoop", "Brain", "ClaudeBrain", "Decision",
    "Hit", "KeyboardMouseSense", "MicrophoneSense", "ScreenSense", "ScriptedBrain", "Situation",
    "SituationSense", "Step", "WebcamSense",
]
