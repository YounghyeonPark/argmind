# SPDX-License-Identifier: Apache-2.0
"""The slow, deliberative half: what to do when the cache has never seen this before.

**Three loops at three rates**, which is the shape the measurements in this repository keep pointing
at: a reflex loop in milliseconds, a perceptual loop in tens of milliseconds, and *this* one in
seconds. The cache exists so the seconds-loop runs as rarely as possible; this module is what it
calls when it must.

**`ScriptedBrain` is not a placeholder for testing convenience.** It is how the loop, the cache and
the actuators are exercised end to end with no API key, no network and no spend — the same rule the
camera demo follows, that *a test which cannot run without the hardware is a component that rots*.
Every measurement of the cache's hit rate can be made against it, because the cache does not care
where a decision came from.

**`ClaudeBrain` is written and has never run.** There is no `anthropic` package in this interpreter
and no key in this environment, so it is **untested code** and says so here rather than in a commit
message. It imports the official SDK lazily and fails with the install line, instead of reaching for
a raw-HTTP shim because `httpx` happens to be installed — a hand-rolled client is a second thing to
keep correct against an API that changes.
"""

from __future__ import annotations

import base64
import io
import time

import numpy as np

from .act import VALID

__all__ = ["Brain", "ScriptedBrain", "ClaudeBrain", "Decision", "ACTION_SCHEMA"]

#: Opus 5. Thinking is on by default there, and `effort` is what trades depth against latency —
#: `low` because this sits in an interaction loop where a second is already the thing being paid.
MODEL = "claude-opus-5"

SYSTEM = """You drive a Windows PC to help the person using it.

You may be shown more than one image, each labelled: [screen] is the display, [webcam] is the person
in front of it. The webcam is there so you can tell whether they are present and what they are doing;
it is never what you act on.

You see the images and a short goal. Reply with the actions to take, as a list. Keep it to the
smallest number of actions that makes progress — you will be asked again after they run.

Coordinates are in screen pixels, origin top-left. If you cannot tell what to do from the
screenshot, return an empty list rather than guessing: a wrong click is worse than no click."""

#: What the model must return. `strict` is set on the tool so the arguments validate exactly, which
#: matters more here than usual: these values are replayed from a cache later, so a malformed action
#: is not one bad turn, it is a bad turn stored and repeated.
ACTION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["actions"],
    "properties": {
        "actions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["do"],
                "properties": {
                    "do": {"type": "string", "enum": list(VALID)},
                    "x": {"type": "integer"},
                    "y": {"type": "integer"},
                    "button": {"type": "string", "enum": ["left", "right", "middle"]},
                    "text": {"type": "string"},
                    "name": {"type": "string"},
                    "seconds": {"type": "number"},
                },
            },
        },
        "why": {"type": "string"},
    },
}


class Decision:
    """What the brain decided, and what asking cost — the cache stores both."""

    __slots__ = ("actions", "why", "seconds")

    def __init__(self, actions, why: str = "", seconds: float = 0.0) -> None:
        self.actions = list(actions)
        self.why = str(why)
        self.seconds = float(seconds)

    def __repr__(self) -> str:
        return f"Decision({len(self.actions)} actions, {self.seconds:.2f}s)"


class Brain:
    """The seam. Anything with `decide()` can drive the loop.

    `frames` is a mapping — `{"screen": ..., "webcam": ...}` — rather than one image, because the
    agent has more than one eye and the brain should see all of them. **Only the screen goes into
    the cache key**, and that is measured rather than assumed: see `agent/situation.py`.
    """

    def decide(self, goal: str, frames: dict, context: str = "") -> Decision:
        raise NotImplementedError


class ScriptedBrain(Brain):
    """Answers from a list, and records what it was asked.

    `latency` is simulated on purpose: without it the cache's whole reason for existing — that a hit
    is cheaper than a miss — cannot be measured, only asserted.
    """

    def __init__(self, script=None, *, latency: float = 0.0, default=None) -> None:
        self.script = list(script or [])
        self.latency = float(latency)
        self.default = list(default or [])
        self.asked: list[str] = []
        #: Which eyes the last call was given, so a test can check the wiring rather than trust it.
        self.saw: list[str] = []

    def decide(self, goal: str, frames: dict, context: str = "") -> Decision:
        t0 = time.perf_counter()
        if self.latency:
            time.sleep(self.latency)
        self.asked.append(goal)
        self.saw = sorted(frames)
        actions = self.script.pop(0) if self.script else self.default
        return Decision(actions, why="scripted", seconds=time.perf_counter() - t0)


class ClaudeBrain(Brain):
    """**Untested.** Written against the official SDK; never executed — no package, no key here.

    Structured through a `strict` tool rather than by asking for JSON in the prompt, because the
    reply is *stored and replayed*: a shape that drifts once becomes a cache entry that drifts every
    time it hits.
    """

    def __init__(self, *, model: str = MODEL, effort: str = "low", max_tokens: int = 16000,
                 client=None) -> None:
        if client is None:
            try:
                import anthropic                                   # noqa: PLC0415
            except ImportError as exc:                             # pragma: no cover - no package
                raise ImportError(
                    "ClaudeBrain needs the official SDK: pip install anthropic. It is deliberately "
                    "not hand-rolled over httpx — a second HTTP client is a second thing to keep "
                    "correct against an API that changes.") from exc
            client = anthropic.Anthropic()
        self.client = client
        self.model = model
        self.effort = effort
        self.max_tokens = int(max_tokens)

    @staticmethod
    def _png(frame) -> str:
        from PIL import Image                                      # noqa: PLC0415

        a = np.asarray(frame)
        if a.dtype != np.uint8:
            a = np.clip(a * 255.0, 0, 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(a).save(buf, format="PNG")
        return base64.standard_b64encode(buf.getvalue()).decode("ascii")

    def decide(self, goal: str, frames: dict, context: str = "") -> Decision:
        t0 = time.perf_counter()
        #: Each eye labelled, because "the screen" and "the webcam" call for different readings and
        #: an unlabelled pair of images invites the model to treat them as one scene.
        content = []
        for name, frame in frames.items():
            content.append({"type": "text", "text": f"[{name}]"})
            content.append({"type": "image",
                            "source": {"type": "base64", "media_type": "image/png",
                                       "data": self._png(frame)}})
        content.append({"type": "text", "text": f"Goal: {goal}\n{context}".strip()})
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM,
            output_config={"effort": self.effort},
            tools=[{
                "name": "act",
                "description": "The actions to take now.",
                "input_schema": ACTION_SCHEMA,
                "strict": True,
            }],
            tool_choice={"type": "tool", "name": "act"},
            messages=[{"role": "user", "content": content}],
        )
        if response.stop_reason == "refusal":
            detail = getattr(response, "stop_details", None)
            raise RuntimeError(
                f"the model declined this request "
                f"({getattr(detail, 'category', None)}): {getattr(detail, 'explanation', '')}")
        for block in response.content:
            if block.type == "tool_use" and block.name == "act":
                return Decision(block.input.get("actions", []), block.input.get("why", ""),
                                time.perf_counter() - t0)
        return Decision([], why="no tool call in the reply", seconds=time.perf_counter() - t0)
