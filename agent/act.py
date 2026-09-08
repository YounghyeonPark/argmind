# SPDX-License-Identifier: Apache-2.0
"""What the agent can do to the machine — and, by default, does not.

The action vocabulary is plain dicts because the **cache persists them**: a value that cannot be
written to JSON cannot survive a restart, and a cache that forgets on restart is not the thing the
owner asked for.

    {"do": "move",  "x": 100, "y": 200}
    {"do": "click", "button": "left", "x": 100, "y": 200}   # x/y optional; omitted means "here"
    {"do": "type",  "text": "hello"}
    {"do": "key",   "name": "enter"}
    {"do": "wait",  "seconds": 0.2}
    {"do": "say",   "text": "..."}                          # NOT BUILT, see MISSING_AUDIO

**`live=False` is the default and that is not timidity.** This moves the real mouse and types into
whatever window has focus; there is no sandbox and no undo. The general rule this repository works
under is that hard-to-reverse actions get confirmed rather than assumed, and a dry run that records
what it *would* have done is what makes the agent's behaviour reviewable before it is granted.

**Three brakes, because one is a single point of failure.**

* `live` must be set explicitly.
* a **rate limit** — an agent in a loop with a bad cache entry can emit thousands of clicks a second,
  and the first thing anyone would want is for that to be slow enough to interrupt.
* a **guard callable** the loop supplies. It is checked before *every* action, not once per batch,
  so "the human touched the mouse" stops the current burst rather than the next one.
"""

from __future__ import annotations

import ctypes
import time

from .senses import MISSING_AUDIO

__all__ = ["Actuator", "ActionRefused", "VALID"]

VALID = ("move", "click", "type", "key", "wait", "say")

#: Virtual-key codes for the named keys. Deliberately short: a name that is not here is refused
#: rather than guessed at, because guessing a keystroke is how an agent sends `Alt+F4` by accident.
KEYS = {
    "enter": 0x0D, "tab": 0x09, "escape": 0x1B, "esc": 0x1B, "space": 0x20,
    "backspace": 0x08, "delete": 0x2E, "home": 0x24, "end": 0x23,
    "up": 0x26, "down": 0x28, "left": 0x25, "right": 0x27,
}


class ActionRefused(RuntimeError):
    """Raised when an action is malformed, unknown, or blocked by a brake."""


ULONG_PTR = ctypes.c_ulonglong if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_ulong


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long), ("dy", ctypes.c_long), ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong), ("time", ctypes.c_ulong),
                ("dwExtraInfo", ctypes.POINTER(ULONG_PTR))]


class _KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort), ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong), ("time", ctypes.c_ulong),
                ("dwExtraInfo", ctypes.POINTER(ULONG_PTR))]


class _INPUTUNION(ctypes.Union):
    _fields_ = [("mi", _MOUSEINPUT), ("ki", _KEYBDINPUT)]


class _INPUT(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = [("type", ctypes.c_ulong), ("u", _INPUTUNION)]


_INPUT_MOUSE, _INPUT_KEYBOARD = 0, 1
_MOUSE = {"left": (0x0002, 0x0004), "right": (0x0008, 0x0010), "middle": (0x0020, 0x0040)}
_KEYEVENTF_KEYUP, _KEYEVENTF_UNICODE = 0x0002, 0x0004


class Actuator:
    """Executes actions, or records what it would have executed.

    `performed` is the log either way, so a dry run and a live run are compared by diffing the same
    list rather than by trusting that the dry run did the same thing.
    """

    def __init__(self, *, live: bool = False, max_per_second: float = 10.0, guard=None) -> None:
        self.live = bool(live)
        self.max_per_second = float(max_per_second)
        #: Called with the action about to run; return False to refuse. Checked per action.
        self.guard = guard
        self.performed: list[dict] = []
        self._user32 = ctypes.windll.user32 if self.live else None
        self._last = 0.0

    # ---- one action ---------------------------------------------------------------------------

    def do(self, action: dict) -> dict:
        #: **Validated whether or not it will be executed.** This used to happen inside the live
        #: branch, so a dry run accepted `{"do": "key", "name": "f4"}` silently -- and the loop
        #: stores what it accepted, so a malformed action became a CACHE ENTRY that would fail the
        #: first time somebody ran with `--live`. Found by a test, and the shape of the bug is this
        #: project's recurring one: a check on a path no fixture visits.
        self.validate(action)
        what = action["do"]
        if self.guard is not None and not self.guard(action):
            raise ActionRefused(f"a brake refused {action!r}")

        gap = 1.0 / self.max_per_second
        since = time.perf_counter() - self._last
        if since < gap:
            time.sleep(gap - since)
        self._last = time.perf_counter()

        record = dict(action)
        record["live"] = self.live
        if self.live:
            self._perform(action)
        self.performed.append(record)
        return record

    @staticmethod
    def validate(action: dict) -> None:
        """Everything that can be known about an action without performing it.

        Separate and public so the loop can refuse a decision *before* caching it: an action stored
        and replayed is worse than one rejected once.
        """
        if not isinstance(action, dict) or "do" not in action:
            raise ActionRefused(f"an action is a dict with a 'do' key, got {action!r}")
        what = action["do"]
        if what not in VALID:
            raise ActionRefused(f"unknown action {what!r}; known: {', '.join(VALID)}")
        if what == "say":
            raise NotImplementedError(f"speaker output is not built. {MISSING_AUDIO}")
        if what == "move":
            if "x" not in action or "y" not in action:
                raise ActionRefused(f"a move needs x and y, got {action!r}")
        elif what == "click":
            button = action.get("button", "left")
            if button not in _MOUSE:
                raise ActionRefused(f"unknown button {button!r}; known: {', '.join(_MOUSE)}")
            if ("x" in action) != ("y" in action):
                raise ActionRefused(f"a click gives both x and y or neither, got {action!r}")
        elif what == "key":
            name = str(action.get("name", "")).lower()
            if name not in KEYS:
                raise ActionRefused(
                    f"unknown key {name!r}. Named keys are listed explicitly rather than guessed, "
                    f"because guessing a keystroke is how an agent sends Alt+F4: "
                    f"{', '.join(sorted(KEYS))}")
        elif what == "type":
            if not isinstance(action.get("text"), str):
                raise ActionRefused(f"a type action needs text, got {action!r}")
        elif what == "wait":
            try:
                float(action.get("seconds", 0.0))
            except (TypeError, ValueError):
                raise ActionRefused(f"wait takes seconds as a number, got {action!r}") from None

    def run(self, actions) -> list[dict]:
        """A sequence, stopping at the first refusal rather than skipping past it."""
        return [self.do(a) for a in actions]

    # ---- the Windows half ---------------------------------------------------------------------

    def _perform(self, action: dict) -> None:
        what = action["do"]
        if what == "wait":
            time.sleep(max(0.0, float(action.get("seconds", 0.0))))
            return
        if what == "move":
            self._user32.SetCursorPos(int(action["x"]), int(action["y"]))
            return
        if what == "click":
            if "x" in action and "y" in action:
                self._user32.SetCursorPos(int(action["x"]), int(action["y"]))
            down, up = _MOUSE[action.get("button", "left")]
            self._send([self._mouse(down), self._mouse(up)])
            return
        if what == "key":
            vk = KEYS[str(action["name"]).lower()]
            self._send([self._key(vk=vk), self._key(vk=vk, flags=_KEYEVENTF_KEYUP)])
            return
        if what == "type":
            events = []
            for ch in str(action["text"]):
                events.append(self._key(scan=ord(ch), flags=_KEYEVENTF_UNICODE))
                events.append(self._key(scan=ord(ch), flags=_KEYEVENTF_UNICODE | _KEYEVENTF_KEYUP))
            self._send(events)

    def _mouse(self, flags: int) -> _INPUT:
        return _INPUT(type=_INPUT_MOUSE,
                      u=_INPUTUNION(mi=_MOUSEINPUT(0, 0, 0, flags, 0, None)))

    def _key(self, *, vk: int = 0, scan: int = 0, flags: int = 0) -> _INPUT:
        return _INPUT(type=_INPUT_KEYBOARD,
                      u=_INPUTUNION(ki=_KEYBDINPUT(vk, scan, flags, 0, None)))

    def _send(self, events) -> None:
        n = len(events)
        arr = (_INPUT * n)(*events)
        sent = self._user32.SendInput(n, arr, ctypes.sizeof(_INPUT))
        if sent != n:
            raise ActionRefused(
                f"SendInput accepted {sent} of {n} events (error {ctypes.GetLastError()}); a "
                "partially delivered burst is worse than none, so this is raised rather than logged")
