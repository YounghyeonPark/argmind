# SPDX-License-Identifier: Apache-2.0
"""When something covers the target, does a click still work? Does a control pattern?

    python -m measure.obstruction

**This exists to decide whether the control patterns stay.** Measured 2026-09-09 on three goals:
`3 of 3` with patterns and `3 of 3` without, so they bought nothing and this project's standard says
unearned code is a cost — about ninety lines, a second COM client, an ancestor walk on every click,
and four ways to fail.

There is one concrete case that goal set never touched, and the app makes it likely rather than
exotic: **the agent's own window is always-on-top.** It is 430x560 in the bottom-right corner, and a
widget underneath it is a widget a synthetic click cannot reach — the click goes to whatever is
topmost at that pixel, which is argmind. A control pattern does not aim at a pixel at all.

So: put a window over the target and try both. Nothing here needs the model.

**What would make the patterns unearned even so**: if the click works anyway. Windows does not
route a click by z-order alone in every case, and assuming it does is the kind of thing that has
already been wrong twice today.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes as wt
import pathlib
import subprocess
import sys
import time
import tkinter as tk

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

user32 = ctypes.windll.user32
_ENUM = ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)

#: The check box, because its effect is unambiguous: advanced view adds four named things.
TARGET = ("Advanced view", "check box")
MARKER = "search"


def windows_of(pid):
    found = []

    def visit(h, _):
        out = wt.DWORD()
        user32.GetWindowThreadProcessId(h, ctypes.byref(out))
        if out.value == pid and user32.IsWindowVisible(h):
            found.append(h)
        return True

    user32.EnumWindows(_ENUM(visit), 0)
    return found


def cover(rect):
    """A borderless always-on-top window sitting exactly over `rect`. Stands in for the app."""
    x, y, w, h = rect
    root = tk.Tk()
    root.overrideredirect(True)
    root.attributes("-topmost", True)
    root.geometry(f"{max(w, 60)}x{max(h, 24)}+{x}+{y}")
    tk.Label(root, text="argmind", bg="#f2f1ee").pack(expand=True, fill="both")
    root.update()
    return root


def attempt(patterns: bool, app: str, stand_aside: bool = False) -> tuple[str, bool]:
    from argmind.env.windows import WindowsEyes, WindowsHands              # noqa: PLC0415

    proc = subprocess.Popen([app])
    time.sleep(2.5)
    handles = windows_of(proc.pid)
    if not handles:
        proc.kill()
        return ("no window", False)

    eyes = WindowsEyes()
    hands = WindowsHands(live=True, max_per_second=8.0, patterns=patterns)
    widget = next((w for w in eyes.read(handles[0])
                   if (w.name, w.role) == TARGET), None)
    if widget is None:
        proc.kill()
        return (f"no {TARGET[0]!r} in this window", False)

    lid = cover(widget.rect)
    time.sleep(0.5)
    if stand_aside:
        #: What the app now does before every click: drop out of the way, act, come back.
        lid.attributes("-topmost", False)
        lid.lower()
        lid.update()
        time.sleep(0.3)
    before = {w.name for h in windows_of(proc.pid) for w in eyes.read(h)}
    record = hands.click(widget)
    time.sleep(1.0)
    after = {w.name for h in windows_of(proc.pid) for w in eyes.read(h)}
    lid.destroy()

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    worked = any(MARKER in n.lower() for n in after - before)
    return (record.get("via", "?"), worked)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m measure.obstruction", description=__doc__)
    p.add_argument("--app", default="charmap.exe")
    args = p.parse_args(argv)

    print(f"  {TARGET[1]} {TARGET[0]!r} covered by an always-on-top window, then acted on\n")
    for label, patterns, aside in (("patterns off, covered ", False, False),
                                   ("patterns on,  covered ", True, False),
                                   ("patterns off, stood aside", False, True)):
        via, worked = attempt(patterns, args.app, aside)
        print(f"  {label:<26} via {via:<8} {'WORKED' if worked else 'did nothing'}")
        time.sleep(1.0)
    print("\n  covered rows tell you whether a pattern reaches past an obstruction that a click")
    print("  cannot; the stood-aside row tells you whether moving out of the way is enough.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
