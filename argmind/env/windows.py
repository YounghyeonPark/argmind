# SPDX-License-Identifier: Apache-2.0
"""Windows: the environment behind `Eyes` and `Hands`.

**This is the only file in the project that knows what UI Automation is.** `argmind/core/` is written
against the two protocols and a test walks its source to keep it that way, so a second environment —
a phone, later — arrives here and nowhere else.

## Eyes — measured, and the reason the whole design is shaped this way

UI Automation returns every visible widget's **name and exact rectangle**, in the interface's own
language, across monitors including negative coordinates:

    one window (Chrome Remote Desktop)      12 widgets     41 ms
    one window (VS Code)                   282 widgets    266 ms
    the whole desktop                    3,670 widgets   22.5 s   <- why this is scoped

Against asking a vision model for the same pixel: **24–82 s**, and on a dense screen no answer at
all. The model could *see* — its reasoning log read every 14-pixel filename correctly — it could not
do arithmetic on what it saw.

**`comtypes`, not PowerShell**, and that was measured too: shelling out cost **776–1037 ms**, of
which the UIA call was 200 ms and the rest was starting a shell, and it returned Korean names as
`???`. Since the names are the entire point of this design, that route is not merely slower.

**What UIA does not see**, said here rather than discovered later: canvas content, video, games and
custom-drawn interfaces expose no widgets. `foreground()` returns an empty list, and the core is
built to tell that apart from a failure.

## Hands — three brakes, because one is a single point of failure

This clicks in whatever window has focus and there is no undo, so `live=False` is the default and the
record exists either way. The brakes are the flag, a rate limit, and a `guard` checked before *every*
action — so "the person touched the mouse" stops the burst that is running, not the next one.

Validation is independent of execution: an earlier version of this idea checked keys only inside the
live branch, so a dry run accepted `{"key": "f4"}` silently, and anything accepted got remembered.
"""

from __future__ import annotations

import ctypes
import time

from _ctypes import COMError

from ..core.contract import Widget

__all__ = ["WindowsEyes", "WindowsHands", "Refused", "KEYS", "UIA_UNAVAILABLE"]

UIA_UNAVAILABLE = (
    "UI Automation could not be reached. It ships with Windows, so this usually means comtypes is "
    "missing (pip install comtypes) or this process has no desktop access.")

#: Named keys, listed rather than guessed: guessing a keystroke is how an agent sends Alt+F4 while
#: trying to close a tab.
KEYS = {
    "enter": 0x0D, "tab": 0x09, "escape": 0x1B, "esc": 0x1B, "space": 0x20,
    "backspace": 0x08, "delete": 0x2E, "home": 0x24, "end": 0x23, "pageup": 0x21,
    "pagedown": 0x22, "up": 0x26, "down": 0x28, "left": 0x25, "right": 0x27,
}


class Refused(RuntimeError):
    """An action that was malformed, unknown, or stopped by a brake."""


# ---- Eyes -------------------------------------------------------------------------------------

class WindowsEyes:
    """`Eyes` for Windows. Build once and reuse — creating the automation object costs more than a
    query does, so a caller that makes a new one per turn pays for nothing."""

    def __init__(self) -> None:
        try:
            from comtypes.client import CreateObject, GetModule           # noqa: PLC0415

            GetModule("UIAutomationCore.dll")
            from comtypes.gen import UIAutomationClient as UIA            # noqa: PLC0415
        except Exception as exc:                                          # pragma: no cover
            raise RuntimeError(UIA_UNAVAILABLE) from exc

        self._uia_mod = UIA
        self._uia = CreateObject(UIA.CUIAutomation, interface=UIA.IUIAutomation)
        #: Offscreen elements are excluded at the source rather than filtered later: a menu that is
        #: not open still exists in the tree, and clicking it would do nothing visible.
        self._visible = self._uia.CreatePropertyCondition(UIA.UIA_IsOffscreenPropertyId, False)
        self._user32 = ctypes.windll.user32
        #: Elements that died mid-read. Counted rather than swallowed silently: a number that grows
        #: fast means the screen is churning, which is information the caller may want.
        self.vanished = 0

    def foreground(self) -> list[Widget]:
        return self.read(self._user32.GetForegroundWindow())

    def read(self, handle) -> list[Widget]:
        """The named things in one window, given its handle.

        Split out from `foreground()` for the fixture capture, which has to read a window **without
        bringing it to the front** — on a desktop somebody is using, stealing focus both disturbs
        them and captures whatever they were doing instead of what was asked for. The agent itself
        only ever reads the foreground; nothing in `core/` knows this exists.
        """
        if not handle:
            return []
        root = self._uia.ElementFromHandle(handle)
        window = root.CurrentName or ""
        found = root.FindAll(self._uia_mod.TreeScope_Descendants, self._visible)
        out = []
        for i in range(found.Length):
            try:
                element = found.GetElement(i)
                name = element.CurrentName
                r = element.CurrentBoundingRectangle
                role = element.CurrentLocalizedControlType
            except COMError:
                #: **An element can die between the enumeration and the read**, and the property
                #: access then raises `An event was unable to invoke any of the subscribers`. Found
                #: by running: reading the screen in a loop crashed within twelve iterations because
                #: something closed underneath it.
                #:
                #: Skipping is the correct answer, not a workaround: an element that vanished while
                #: it was being read is not on screen, which is exactly what this function reports.
                #: Raising would mean a window closing anywhere kills an agent that reads
                #: continuously — and reading continuously is the design.
                self.vanished += 1
                continue
            w, h = r.right - r.left, r.bottom - r.top
            #: A widget with no name cannot be asked for by name and one with no area cannot be
            #: clicked. Both are dropped here, so every `Widget` that exists is usable.
            if name and w > 0 and h > 0:
                out.append(Widget(name, role, (r.left, r.top, w, h), window))
        return out

    def timed_foreground(self) -> tuple[list[Widget], float]:
        t0 = time.perf_counter()
        widgets = self.foreground()
        return widgets, time.perf_counter() - t0


# ---- Hands ------------------------------------------------------------------------------------

_PTR = ctypes.c_ulonglong if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_ulong


class _MOUSE(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long), ("dy", ctypes.c_long), ("data", ctypes.c_ulong),
                ("flags", ctypes.c_ulong), ("time", ctypes.c_ulong),
                ("extra", ctypes.POINTER(_PTR))]


class _KEY(ctypes.Structure):
    _fields_ = [("vk", ctypes.c_ushort), ("scan", ctypes.c_ushort), ("flags", ctypes.c_ulong),
                ("time", ctypes.c_ulong), ("extra", ctypes.POINTER(_PTR))]


class _U(ctypes.Union):
    _fields_ = [("mi", _MOUSE), ("ki", _KEY)]


class _INPUT(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = [("type", ctypes.c_ulong), ("u", _U)]


_MOUSE_T, _KEY_T = 0, 1
_BUTTON = {"left": (0x0002, 0x0004), "right": (0x0008, 0x0010), "middle": (0x0020, 0x0040)}
_KEYUP, _UNICODE = 0x0002, 0x0004


class WindowsHands:
    """`Hands` for Windows. `done` is the log in a dry run and a live run alike, so the two are
    compared by diffing the same list rather than by trusting they did the same thing."""

    def __init__(self, *, live: bool = False, max_per_second: float = 4.0, guard=None) -> None:
        self.live = bool(live)
        self.max_per_second = float(max_per_second)
        self.guard = guard
        self.done: list[dict] = []
        self._user32 = ctypes.windll.user32
        self._last = 0.0

    def click(self, widget: Widget, button: str = "left") -> dict:
        if button not in _BUTTON:
            raise Refused(f"unknown button {button!r}; known: {', '.join(_BUTTON)}")
        x, y = widget.centre
        return self._do({"do": "click", "button": button, "x": x, "y": y,
                         "on": widget.name, "role": widget.role})

    def type_text(self, text: str) -> dict:
        if not isinstance(text, str) or not text:
            raise Refused(f"type needs a non-empty string, got {text!r}")
        return self._do({"do": "type", "text": text})

    def press(self, key: str) -> dict:
        name = str(key).lower()
        if name not in KEYS:
            raise Refused(
                f"unknown key {name!r}. Named keys are listed explicitly rather than guessed, "
                f"because guessing a keystroke is how an agent sends Alt+F4: {', '.join(sorted(KEYS))}")
        return self._do({"do": "key", "name": name})

    def _do(self, action: dict) -> dict:
        if self.guard is not None and not self.guard(action):
            raise Refused(f"a brake refused {action!r}")
        gap = 1.0 / self.max_per_second
        since = time.perf_counter() - self._last
        if since < gap:
            time.sleep(gap - since)
        self._last = time.perf_counter()

        record = dict(action, live=self.live)
        if self.live:
            self._perform(action)
        self.done.append(record)
        return record

    def _perform(self, action: dict) -> None:
        what = action["do"]
        if what == "click":
            self._user32.SetCursorPos(int(action["x"]), int(action["y"]))
            down, up = _BUTTON[action["button"]]
            self._send([self._mouse(down), self._mouse(up)])
        elif what == "key":
            vk = KEYS[action["name"]]
            self._send([self._key(vk=vk), self._key(vk=vk, flags=_KEYUP)])
        elif what == "type":
            events = []
            for ch in action["text"]:
                events.append(self._key(scan=ord(ch), flags=_UNICODE))
                events.append(self._key(scan=ord(ch), flags=_UNICODE | _KEYUP))
            self._send(events)

    def _mouse(self, flags: int) -> _INPUT:
        return _INPUT(type=_MOUSE_T, u=_U(mi=_MOUSE(0, 0, 0, flags, 0, None)))

    def _key(self, *, vk: int = 0, scan: int = 0, flags: int = 0) -> _INPUT:
        return _INPUT(type=_KEY_T, u=_U(ki=_KEY(vk, scan, flags, 0, None)))

    def _send(self, events) -> None:
        n = len(events)
        arr = (_INPUT * n)(*events)
        sent = self._user32.SendInput(n, arr, ctypes.sizeof(_INPUT))
        if sent != n:
            raise Refused(
                f"SendInput delivered {sent} of {n} events (error {ctypes.GetLastError()}); a "
                "half-delivered burst is worse than none, so this raises rather than logging")
