# SPDX-License-Identifier: Apache-2.0
"""What an environment must provide, and nothing else.

**The core is written against these two protocols and never against a platform.** That is not a
style preference — it is the only way to find out whether "one core, an interface per environment"
is real. If the core cannot be written and tested without naming Windows, the split is imaginary and
one commit says so, which is cheaper than a restructure later.

**The interface is determined, not invented.** Both accessibility trees a GUI ships already return
the same four things, so the shape below is a description of what exists rather than a guess about
what might be wanted:

| | Windows UI Automation | Android `uiautomator` |
|---|---|---|
| what it is called | `Name` | `text` / `content-desc` |
| what kind of thing | `LocalizedControlType` | `class` |
| where it is | `BoundingRectangle` | `bounds` |
| is it visible | `IsOffscreen` | `visible-to-user` |

**Why an accessibility tree at all.** Measured on this machine: asking a vision model where a button
is took **24–82 s** and on a dense screen never answered, while the OS answered exactly in
**41–266 ms**. The model was not failing to *see* — its reasoning log read every 14-pixel filename
correctly — it was failing at arithmetic on what it saw. So:

> **The deterministic side says WHERE. The probabilistic side says WHAT.
> A model never emits a coordinate.**

**A room is not a GUI, and `Eyes` should not be forced onto one.** Anything filling it for a robot
would be a detector guessing names and boxes, which puts probabilistic output through an interface
whose entire contract is that it is exact — the rule inverted quietly, at the seam. The previous
argmind spent a month establishing that a room hands you no ground truth; that lesson is not worth
burying inside a `robot.py` that implements this badly.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

__all__ = ["Widget", "Eyes", "Hands"]


class Widget:
    """One thing an environment can name and point at.

    `rect` is `(left, top, width, height)` in whatever coordinate space the environment acts in.
    Left and top may be **negative** — a second monitor to the left is normal and the numbers come
    back that way — so nothing here clamps them.
    """

    __slots__ = ("name", "role", "rect", "window")

    def __init__(self, name: str, role: str, rect, window: str = "") -> None:
        self.name = str(name)
        self.role = str(role)
        self.rect = tuple(int(v) for v in rect)
        self.window = str(window)

    @property
    def centre(self) -> tuple[int, int]:
        """The one place a name becomes a coordinate. Two such places would be two chances to
        disagree about what "the middle of this thing" means."""
        x, y, w, h = self.rect
        return x + w // 2, y + h // 2

    def __eq__(self, other) -> bool:
        return (isinstance(other, Widget) and self.name == other.name
                and self.role == other.role and self.rect == other.rect)

    def __hash__(self) -> int:
        return hash((self.name, self.role, self.rect))

    def __repr__(self) -> str:
        x, y, w, h = self.rect
        return f"Widget({self.role} {self.name[:32]!r} @ {x},{y} {w}x{h})"


@runtime_checkable
class Eyes(Protocol):
    """Where things are. Exact, from the environment itself — never from a model."""

    def foreground(self) -> list[Widget]:
        """Every named, placed, visible widget the person is currently looking at.

        Scoped to what is in front of them because that is what "click share" refers to, and because
        the whole desktop was measured at **22.5 s** against 0.2 s for one window.

        An empty list is a real answer, not a failure: some surfaces — canvas, video, games,
        custom-drawn UI — expose no widgets at all, and a caller has to be able to tell that from
        an error.
        """
        ...

    def timed_foreground(self) -> tuple[list[Widget], float]:
        """The same, with the seconds it took. Required rather than optional because a latency claim
        should be measured where it is made, not recalled from a README."""
        ...


@runtime_checkable
#: **What an interactive interface wants beyond `Eyes`, and why it is not in the protocol.**
#:
#: An app that runs in a window on the screen it reads needs three things the core never asks for,
#: and each is optional because a core that only answers questions works without any of them:
#:
#:     attending_title() -> str        the window the person is working in, named cheaply (0.04 ms)
#:     at(x, y) -> (name, role)        what the environment thinks is at a point — a second opinion
#:     pointing_is_exact() -> bool     do the coordinates written match the coordinates read
#:
#: They are on `WindowsEyes` and `WindowsHands` and absent from the protocols on purpose. Putting
#: them here would make every future environment implement three methods to satisfy a caller that
#: may not exist on that platform — a phone has no "window the person is in" and no cursor at all.
#: `argmind/app.py` resolves them once with `getattr` and falls back to answers that are honest
#: rather than invented: no title, nothing known at that point, and pointing assumed exact.
#:
#: **A fourth joined them on 2026-09-10, and it is the case that tests the pattern rather than
#: repeating it**, because the caller is the *core* and not an interface:
#:
#:     keys() -> (name, ...)           every name `Hands.press` accepts on this platform
#:
#: The argument above does not transfer as written — anything implementing `press` has *some*
#: vocabulary, so "the platform may not have this at all" is not the objection. The objection that
#: does hold is the other one, and it is stronger here: **the whole vocabulary is the wrong quantity
#: for the caller.** Android publishes some 280 `KEYCODE_*` names; a protocol member would oblige
#: every environment to hand over all of them so that `Session.knows_key()` could consult a handful,
#: and a core that rendered what it was handed would be back at the open key list measured at
#: **22-71 s and a turn that produced nothing**. So the protocol stays at three methods, `Hands` is
#: untouched, and the fallback is that an environment declaring nothing keeps every key — safe in a
#: way the `Eyes` fallbacks are not required to be, since over-advertising a key costs a refusal
#: from `press` while over-advertising a coordinate would cost a click on something never chosen.
#:
#: **What a model is TOLD about is a different question and gets a different answer.** It is a
#: budget the core sets against a measured degeneration hazard, not a capability the environment
#: reports, and the two disagree today on purpose: fifteen names accepted, three advertised.


class Hands(Protocol):
    """Doing things. A click takes a `Widget`, never a coordinate — the rectangle came from `Eyes`,
    so the click lands where the environment says the thing is."""

    live: bool

    def click(self, widget: Widget, button: str = "left") -> dict: ...

    def type_text(self, text: str) -> dict: ...

    def press(self, key: str) -> dict: ...
