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
import ctypes.wintypes as wintypes
import time

from _ctypes import COMError

from ..core.contract import Widget

#: `EnumWindows` needs its callback kept alive for the duration of the call; building
#: the functype once here is both cheaper and safer than per-call.
_ENUM_WINDOWS = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

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

    def __init__(self, *, ignore_pid: int | None = None) -> None:
        #: **The agent's own window is on the screen it reads.** With `ignore_pid` set to the app's
        #: process, `foreground()` never returns the agent itself: it returns the last window that
        #: was in front and was somebody else's. Without this an agent app answers questions about
        #: its own text box, which is not a corner case — it is what happens every time the person
        #: types.
        self.ignore_pid = ignore_pid
        self._theirs = 0
        #: How often the person was looking at the agent and the remembered window stood in. Counted
        #: because the substitution is a **guess about what "this window" means**, and a guess that
        #: nobody counts is a guess nobody can check.
        self.looked_at_itself = 0
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
        return self.read(self.attending())

    def attending(self) -> int:
        """The window the person is working in — **not necessarily the one with focus.**

        When the foreground window belongs to `ignore_pid` the answer is the last one that did not,
        because the alternative is an agent that reads its own input box the moment you click on it.

        This is a **substitution and it can be wrong**: alt-tab twice quickly and the remembered
        window is the one before last. It is right often enough to be the only workable rule — a
        person asking an agent about "this window" has just come from it — and `looked_at_itself`
        counts how often the substitution was used at all, so the frequency is a measurement rather
        than an impression.
        """
        handle = self._user32.GetForegroundWindow()
        if handle and not self._is_ours(handle):
            self._theirs = handle
            return handle
        self.looked_at_itself += 1
        if self._theirs and self._user32.IsWindow(self._theirs):
            return self._theirs
        #: **Nothing remembered, so go and look.** An app that opens with focus has never seen
        #: anybody else's window, and without this it stays blind until the person alt-tabs —
        #: measured by watching it answer "I can't see any windows here" twice in a row while a
        #: browser with 141 widgets sat behind it. `EnumWindows` walks top-level windows in z-order,
        #: so the first visible titled window that is not ours is the one underneath.
        self._theirs = self._topmost_of_theirs()
        return self._theirs

    def _topmost_of_theirs(self) -> int:
        found = []

        def visit(handle, _param):
            if (self._user32.IsWindowVisible(handle) and not self._is_ours(handle)
                    and self._user32.GetWindowTextLengthW(handle) > 0):
                found.append(handle)
                return False
            return True

        self._user32.EnumWindows(_ENUM_WINDOWS(visit), 0)
        return found[0] if found else 0

    def _is_ours(self, handle) -> bool:
        if self.ignore_pid is None:
            return False
        pid = ctypes.c_ulong()
        self._user32.GetWindowThreadProcessId(handle, ctypes.byref(pid))
        return pid.value == self.ignore_pid

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

    def attending_title(self) -> str:
        """The title of the window `attending()` picked, without reading its whole tree.

        A full read is 41-266 ms and an agent app wants to say which window it is looking at several
        times a second. This is the cheap half of that question.
        """
        handle = self.attending()
        if not handle:
            return ""
        length = self._user32.GetWindowTextLengthW(handle)
        buffer = ctypes.create_unicode_buffer(length + 1)
        self._user32.GetWindowTextW(handle, buffer, length + 1)
        return buffer.value

    def at(self, x: int, y: int) -> tuple[str, str]:
        """What UI Automation says is at a screen point: `(name, role)`, or `("", "")`.

        **A second opinion on a coordinate, used as a hint and not as a gate.** Measured on a real
        window: a widget's own rectangle centre resolves back to that widget **73% of the time**,
        to a name containing or contained by it 14%, and to something else 12.8%.

        That last figure is an **upper bound on how often a click misses, not the miss rate.** A
        click lands on a pixel, and a parent element answering the hit test does not mean the pixel
        belongs to something else — window caption buttons are the clear case, where `Close`
        resolves to `application` and the pixel is nonetheless exactly the close button. So this
        informs the person before they confirm; it must not silently refuse, or it would block
        closing a window, which is the most common thing anybody asks for.
        """
        point = wintypes.POINT(int(x), int(y))
        try:
            element = self._uia.ElementFromPoint(point)
            return (element.CurrentName or "", element.CurrentLocalizedControlType or "")
        except (COMError, ValueError, OSError):
            return ("", "")

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

    def __init__(self, *, live: bool = False, max_per_second: float = 4.0, guard=None,
                 patterns: bool = True) -> None:
        self.live = bool(live)
        self.max_per_second = float(max_per_second)
        self.guard = guard
        self.done: list[dict] = []
        self._user32 = ctypes.windll.user32
        self._last = 0.0
        #: **Operate a control the way the environment knows how, before resorting to a pixel.**
        #: Measured 2026-09-09: asked to open a font list, the agent chose the right combo box, and
        #: the click landed exactly on it and did nothing -- a combo box is centred on its text
        #: field and the arrow that opens the list is at the edge. UI Automation exposes `Invoke`,
        #: `ExpandCollapse`, `Toggle` and `SelectionItem` for exactly this, and using them is the
        #: deterministic side doing the acting, which is the same argument that put WHERE on the
        #: environment. Not every control has a pattern, so the click stays as the fallback and
        #: `done` records which was used -- assuming the new path is better is how a regression
        #: gets shipped.
        self.patterns = bool(patterns)
        self._uia = None
        self._uia_mod = None
        #: How many actions went each way. The number that says whether this was worth adding.
        self.by_pattern = 0
        self.by_click = 0
        #: **Why the last action fell back.** `_pattern_for` returned `None` for four different
        #: reasons and nothing recorded which -- so a run that fell back to a click on a control
        #: that provably supports `ExpandCollapse` could not be diagnosed at all. Same mistake as a
        #: measurement that keeps a boolean where it needed the delta, made twice in one day.
        self.why = ""

    def pointing_is_exact(self, points=((300, 300), (900, 500))) -> bool:
        """Does the cursor go where it is sent? Checked once, before anything is clicked.

        **UIA reports physical pixels and a DPI-unaware process can have its cursor calls
        virtualised**, in which case every click is offset by the scaling factor and lands on
        whatever happens to be there. Measured on this machine: 40 of 40 points came back exactly,
        because the display is at 96 DPI — which is a property of one desk, not a guarantee. On a
        scaled display the two spaces disagree and an agent that clicks anyway is worse than one
        that does not.

        Moves the pointer and puts it back. Presses nothing.
        """
        keep = wintypes.POINT()
        self._user32.GetCursorPos(ctypes.byref(keep))
        try:
            for x, y in points:
                self._user32.SetCursorPos(int(x), int(y))
                got = wintypes.POINT()
                self._user32.GetCursorPos(ctypes.byref(got))
                if (got.x, got.y) != (int(x), int(y)):
                    return False
        finally:
            self._user32.SetCursorPos(keep.x, keep.y)
        return True

    def click(self, widget: Widget, button: str = "left") -> dict:
        if button not in _BUTTON:
            raise Refused(f"unknown button {button!r}; known: {', '.join(_BUTTON)}")
        x, y = widget.centre
        action = {"do": "click", "button": button, "x": x, "y": y,
                  "on": widget.name, "role": widget.role}
        #: A right or middle click asks for a context menu, which no pattern expresses.
        if self.patterns and button == "left":
            action["via"] = self._pattern_for(widget) or "click"
            if action["via"] == "click" and self.why:
                action["why"] = self.why
        else:
            action["via"] = "click"
        return self._do(action)

    # ---- operating a control rather than aiming at it --------------------------------------------

    def _automation(self):
        """The UIA client, made once and only if a pattern is ever wanted."""
        if self._uia is None:
            from comtypes.client import CreateObject, GetModule           # noqa: PLC0415

            GetModule("UIAutomationCore.dll")
            from comtypes.gen import UIAutomationClient as UIA            # noqa: PLC0415

            self._uia_mod = UIA
            self._uia = CreateObject(UIA.CUIAutomation, interface=UIA.IUIAutomation)
        return self._uia

    def _element_for(self, widget: Widget):
        """The element this widget names, found from its own centre and then upwards.

        **Upwards is the point.** The centre of a combo box sits on the edit control inside it, so
        `ElementFromPoint` answers with the child and the thing that expands is its parent. Walking
        up a few levels looking for the name is cheap and exact, and where the name is not found the
        answer is `None` rather than a guess -- a caption button whose centre resolves straight to
        the application is the measured case where that happens, and a synthetic click is right
        there anyway.
        """
        uia = self._automation()
        try:
            element = uia.ElementFromPoint(wintypes.POINT(*widget.centre))
        except (COMError, ValueError, OSError) as exc:
            self.why = f"nothing at the centre ({type(exc).__name__})"
            return None
        walker = uia.RawViewWalker
        for _ in range(5):
            #: **`not element`, not `element is None`.** comtypes hands back a NULL COM pointer
            #: rather than `None` when a walk runs off the top of the tree, and a NULL pointer is
            #: not None -- reading a property off it raises `ValueError: NULL COM pointer access`.
            #: Found by covering the target with another window, which is the case this lookup
            #: cannot serve anyway: it resolves by POINT, so an obstruction hands it the
            #: obstruction's own element.
            if not element:
                self.why = "walked past the top without a name and role match"
                return None
            try:
                #: Name AND role. Walking up on the name alone matched an ancestor that happened
                #: to share it -- a tab item named 'Claude Code' sitting inside a group of the same
                #: name resolved to the group, and operating the group is not what was asked for.
                if (element.CurrentName == widget.name
                        and element.CurrentLocalizedControlType == widget.role):
                    return element
                element = walker.GetParentElement(element)
            except (COMError, ValueError) as exc:
                self.why = f"the element could not be read ({type(exc).__name__})"
                return None
        self.why = "no ancestor within five hops matched the name and role"
        return None

    def _pattern_for(self, widget: Widget):
        """The pattern that operates this control, applied when live. `None` means click instead.

        Tried in a fixed order. `Invoke` first because a control that has it means "press me"; a
        combo box does not have it and falls through to `ExpandCollapse`, which is the measured
        failure. Putting expand first would make a split button drop its menu instead of doing its
        job.
        """
        self.why = ""
        element = self._element_for(widget)
        if element is None:
            return None
        uia = self._uia_mod
        wanted = (
            ("invoke", uia.UIA_InvokePatternId, uia.IUIAutomationInvokePattern, "Invoke"),
            ("expand", uia.UIA_ExpandCollapsePatternId,
             uia.IUIAutomationExpandCollapsePattern, "Expand"),
            ("toggle", uia.UIA_TogglePatternId, uia.IUIAutomationTogglePattern, "Toggle"),
            ("select", uia.UIA_SelectionItemPatternId,
             uia.IUIAutomationSelectionItemPattern, "Select"),
        )
        offered: list[str] = []
        for name, ident, interface, method in wanted:
            try:
                raw = element.GetCurrentPattern(ident)
                if not raw:
                    continue
                pattern = raw.QueryInterface(interface)
            except (COMError, ValueError, AttributeError, OSError) as exc:
                offered.append(f"{name} unavailable ({type(exc).__name__})")
                continue
            #: A dry run resolves the pattern and stops there, so `done` says what *would* have
            #: happened rather than leaving the two paths indistinguishable until something runs.
            if not self.live:
                return name
            try:
                getattr(pattern, method)()
            except COMError as exc:
                self.why = f"{name} was offered and raised {exc.args[0]:#010x}"
                return None
            return name
        self.why = ("the element offers none of invoke, expand, toggle or select"
                    + (f" [{'; '.join(offered)}]" if offered else ""))
        return None

    def type_text(self, text: str) -> dict:
        if not isinstance(text, str) or not text:
            raise Refused(f"type needs a non-empty string, got {text!r}")
        return self._do({"do": "type", "text": text})

    def keys(self) -> tuple[str, ...]:
        """Every name `press` accepts — **the vocabulary, declared instead of guessed at.**

        **Optional, and off `Hands` on purpose.** `contract.py` records the argument for
        `attending_title()` and `at()`, and it applies here with the emphasis moved. Its weak half
        does not transfer: anything implementing `press` has *some* vocabulary, so "a phone may not
        have one" is not the objection. Its strong half is stronger. **The full vocabulary is the
        wrong quantity for the caller that wants it** — Android exposes some 280 `KEYCODE_*` names,
        and a protocol member would oblige every environment to publish all of them so that the core
        could discard nearly all of them. A core that then rendered what it was handed would be back
        at the open key list measured at 22-71 s and a turn that produced nothing.

        So this answers one question and not two: **is this name a key here**, which is what the core
        needs in order to drop an invented one. It does *not* answer "what should a model be told
        about" — that is a budget the core sets and measures, and the two questions have different
        answers on this platform today: fifteen names against three advertised.

        `KEYS` and nothing else. `press` refuses against `KEYS`, so a hand-written list beside it
        would be a second spelling of one rule and a second chance for it to drift. Fifteen names
        over fourteen keystrokes — `esc` and `escape` are the same virtual key — and both spellings
        are here because the question is about names.
        """
        return tuple(sorted(KEYS))

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
        if action.get("via", "click") == "click":
            #: **Only clicks are counted here.** The first version counted every action, so three
            #: goals reported "4 through a click" -- the extra being the `escape` presses between
            #: them. A tally that silently includes keystrokes is a tally that reads as an argument
            #: for something it never measured.
            if action["do"] == "click":
                self.by_click += 1
            if self.live:
                self._perform(action)
        else:
            #: The pattern was already applied while it was being chosen -- deciding which one
            #: applies and applying it are one COM round trip, and splitting them would mean
            #: finding the element twice on a screen that moves 13 times in 14.
            self.by_pattern += 1
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
