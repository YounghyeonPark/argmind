# SPDX-License-Identifier: Apache-2.0
"""What starts a thought, and what a thought produces.

The first loop welded four things together: one utterance produced one reading, one decision and at
most one click. That sequence is an artefact of the loop, not a property of the problem — **acting is
not a stage that always runs, it is something decided**. So a decision has a *trigger* and yields an
*intent*, and an intent may contain nothing to do, or several things.

## The measured constraint that shapes the trigger side

Perception changes constantly. Measured on a real desktop while somebody works, over the second a
model call takes: **something changed in 13 samples of 14**, and even the rarest thing measured — the
foreground window itself — changed in 6 of 14.

A model call costs 0.3–2 s. So *"think whenever the screen changes"* is not a design, it is a
saturated model and an agent that never listens. **A trigger therefore needs a filter and a floor on
how often it may fire**, and that is part of the design rather than a tuning detail.

## The safety consequence, stated because it is the reason this file exists

An agent that acts *because it was asked* is watched by the person who asked. An agent that acts **on
its own judgement** may not be. The three brakes built for the first case — a `live` flag, a rate
limit, a guard before every action — were designed for "you asked, it does one thing", and they do
not cover "it decided, repeatedly, while you were reading something else".

So an effect carries **what caused it**, and an unprompted effect without standing permission is not
refused — it is **downgraded to speech**. The agent says *"the build finished, want me to open it?"*
instead of opening it. The refusal path is conversational, which is what a thing you talk to should
do, and it is the default because a wrong click has no undo.
"""

from __future__ import annotations

import time

__all__ = ["Trigger", "Said", "Perceived", "Effect", "Intent", "Watch", "WindowChanged",
           "Appeared", "Vanished", "MIN_GAP"]

#: The floor on how often any one watch may start a thought. A model call is 0.3–2 s and the screen
#: changes far faster than that; without a floor the agent is permanently mid-thought and deaf.
MIN_GAP = 3.0


class Trigger:
    """Why a decision is being made. Carried into the intent so an effect knows what caused it."""

    __slots__ = ("kind", "text", "at")

    def __init__(self, kind: str, text: str) -> None:
        self.kind = kind
        self.text = text
        self.at = time.time()

    @property
    def asked(self) -> bool:
        """Whether a person asked for this. **The whole permission question turns on it.**"""
        return self.kind == "said"

    def __repr__(self) -> str:
        return f"{self.kind}({self.text[:40]!r})"


def Said(text: str) -> Trigger:                                            # noqa: N802
    """The person spoke. The only trigger that carries its own permission."""
    return Trigger("said", text)


def Perceived(what: str) -> Trigger:                                       # noqa: N802
    """Something on screen changed in a way a watch cared about."""
    return Trigger("perceived", what)


class Effect:
    """One thing to do, and what caused it.

    `target` is a **name**, not a widget and not a rectangle. The thing is looked up again at the
    moment of acting, because over the second a decision takes, a widget that was there is gone
    thirteen times in fourteen — the name still refers to the world and the rectangle does not.
    """

    __slots__ = ("kind", "target", "text", "because")

    def __init__(self, kind: str, *, target: str = "", text: str = "", because=None) -> None:
        self.kind = kind
        self.target = target
        self.text = text
        self.because = because

    @property
    def unprompted(self) -> bool:
        return self.because is not None and not self.because.asked

    def __repr__(self) -> str:
        what = self.target or self.text
        return f"Effect({self.kind} {what[:32]!r}{' unprompted' if self.unprompted else ''})"


class Intent:
    """What a thought came to: something to say, and zero or more things to do.

    Zero is the common case and the important one. An agent that always produces an action will
    produce one on a screen that does not contain what was asked for.
    """

    __slots__ = ("say", "effects", "trigger", "reply")

    def __init__(self, say: str = "", effects=None, trigger=None, reply=None) -> None:
        self.say = say
        self.effects = list(effects or [])
        self.trigger = trigger
        self.reply = reply

    def __bool__(self) -> bool:
        return bool(self.say or self.effects)

    def __repr__(self) -> str:
        return f"Intent(say={self.say[:28]!r}, {len(self.effects)} effects)"


# ---- watches ----------------------------------------------------------------------------------

class Watch:
    """A filter on perception change, with a floor on how often it may fire.

    The floor is not politeness. Measured: something on screen changes in 13 seconds out of 14, and
    a model call takes 0.3–2 s, so an unfloored watch is a model that never stops and an agent that
    never listens.
    """

    def __init__(self, gap: float = MIN_GAP) -> None:
        self.gap = float(gap)
        self._last = 0.0

    def look(self, before, after) -> str | None:
        """Override. Return a short description of what happened, or `None` for nothing worth a
        thought."""
        raise NotImplementedError

    def fired(self, before, after) -> Trigger | None:
        now = time.time()
        if now - self._last < self.gap:
            return None
        what = self.look(before, after)
        if not what:
            return None
        self._last = now
        return Perceived(what)


class WindowChanged(Watch):
    """The person moved to a different window.

    The rarest change measured — 6 of 14 over a second, 2 of 14 over 0.3 s — and still frequent
    enough that the floor above does real work.
    """

    def look(self, before, after) -> str | None:
        was = before[0].window if before else ""
        now = after[0].window if after else ""
        if was != now and now:
            return f"the window changed to {now!r}"
        return None


class Appeared(Watch):
    """Something whose name contains `needle` showed up. The useful shape of "tell me when X"."""

    def __init__(self, needle: str, gap: float = MIN_GAP) -> None:
        super().__init__(gap)
        self.needle = needle.lower()

    def look(self, before, after) -> str | None:
        had = {w.name for w in before}
        new = [w.name for w in after if self.needle in w.name.lower() and w.name not in had]
        return f"{new[0]!r} appeared" if new else None


class Vanished(Watch):
    """Something whose name contains `needle` went away."""

    def __init__(self, needle: str, gap: float = MIN_GAP) -> None:
        super().__init__(gap)
        self.needle = needle.lower()

    def look(self, before, after) -> str | None:
        now = {w.name for w in after}
        gone = [w.name for w in before if self.needle in w.name.lower() and w.name not in now]
        return f"{gone[0]!r} went away" if gone else None
