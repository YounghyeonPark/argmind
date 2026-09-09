# SPDX-License-Identifier: Apache-2.0
"""Does acting steal the mouse pointer? Measured, because it is the one thing the patterns may buy.

    python -m measure.pointer_theft

A synthetic click moves the operating system's cursor to the widget and presses the physical button.
On a machine somebody is using — and this one is used over a remote desktop while the agent runs —
that yanks the pointer out from under them every time the agent acts. A control pattern calls a
method on the control and never touches the pointer at all.

**That is a claim about the code and it was being made as an argument.** The control patterns were
measured on task success (`3 of 3` either way) and on obstruction (`nothing` either way) and found
unearned on both, and success is not the only axis a person cares about. This puts the remaining
axis on the same footing: cursor position before and after, both ways.

Nothing here needs the model. It launches its own Character Map and closes it.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes as wt
import pathlib
import subprocess
import sys
import time

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

user32 = ctypes.windll.user32
_ENUM = ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)

#: A check box, whose effect is unambiguous and whose pattern (`Invoke`) is one the hands try first.
TARGET = ("Advanced view", "check box")
MARKER = "search"
PARKED = (700, 400)


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


def where() -> tuple[int, int]:
    point = wt.POINT()
    user32.GetCursorPos(ctypes.byref(point))
    return (point.x, point.y)


def attempt(patterns: bool, app: str):
    from argmind.env.windows import WindowsEyes, WindowsHands              # noqa: PLC0415

    proc = subprocess.Popen([app])
    time.sleep(2.5)
    handles = windows_of(proc.pid)
    if not handles:
        proc.kill()
        return ("no window", None, None, False)

    eyes = WindowsEyes()
    hands = WindowsHands(live=True, max_per_second=8.0, patterns=patterns)
    widget = next((w for w in eyes.read(handles[0]) if (w.name, w.role) == TARGET), None)
    if widget is None:
        proc.kill()
        return (f"no {TARGET[0]!r}", None, None, False)

    #: Park the pointer somewhere that is not the target, so "it did not move" is a real answer and
    #: not an accident of where the pointer already happened to be.
    user32.SetCursorPos(*PARKED)
    time.sleep(0.3)
    before_cursor = where()
    before = {w.name for h in windows_of(proc.pid) for w in eyes.read(h)}
    record = hands.click(widget)
    time.sleep(0.8)
    after_cursor = where()
    after = {w.name for h in windows_of(proc.pid) for w in eyes.read(h)}

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    worked = any(MARKER in n.lower() for n in after - before)
    return (record.get("via", "?"), before_cursor, after_cursor, worked)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m measure.pointer_theft", description=__doc__)
    p.add_argument("--app", default="charmap.exe")
    args = p.parse_args(argv)

    print(f"  acting on {TARGET[1]} {TARGET[0]!r}, pointer parked at {PARKED} first\n")
    for label, patterns in (("patterns on ", True), ("patterns off", False)):
        via, was, now, worked = attempt(patterns, args.app)
        if was is None:
            print(f"  {label}  {via}")
            continue
        moved = was != now
        print(f"  {label}  via {via:<8} {'WORKED' if worked else 'did nothing':<11} "
              f"pointer {was} -> {now}  {'MOVED' if moved else 'unmoved'}")
        time.sleep(1.0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
