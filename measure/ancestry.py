# SPDX-License-Identifier: Apache-2.0
"""Why does a widget's centre not lead back to the widget? Read-only, clicks nothing.

    python -m measure.ancestry                  Character Map, every combo box and button
    python -m measure.ancestry --app mspaint.exe

`WindowsHands` operates a control through UI Automation when it can find the element, and it finds
it by asking what is at the widget's centre and then walking **up** until the name and role match.
Measured 2026-09-09: that fails for the one case the whole mechanism was built for — a combo box
named `Font :` fell back to a synthetic click, which is what did nothing in the first place.

This prints the chain so the reason is visible instead of guessed. It launches its own window,
reads it, and closes it. **It presses nothing and clicks nothing.**
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


#: Every pattern UI Automation defines, so the answer is "here is what this control offers" rather
#: than "none of the four I happened to try". The four the hands use were chosen from a guess about
#: which controls expose what, and the combo box that motivated the whole change offered none of
#: them — a guess is not a survey.
PATTERN_NAMES = [
    "Invoke", "Selection", "Value", "RangeValue", "Scroll", "ExpandCollapse", "Grid", "GridItem",
    "MultipleView", "Window", "SelectionItem", "Dock", "Table", "TableItem", "Text", "Toggle",
    "Transform", "ScrollItem", "LegacyIAccessible", "ItemContainer", "VirtualizedItem",
    "SynchronizedInput",
]


def supported(element, uia_mod):
    out = []
    for i, name in enumerate(PATTERN_NAMES):
        try:
            if element.GetCurrentPattern(10000 + i):
                out.append(name)
        except Exception:                                                 # noqa: BLE001, S112
            continue
    return out


def windows_of(pid):
    found = []

    def visit(handle, _p):
        out = wt.DWORD()
        user32.GetWindowThreadProcessId(handle, ctypes.byref(out))
        if out.value == pid and user32.IsWindowVisible(handle):
            found.append(handle)
        return True

    user32.EnumWindows(_ENUM(visit), 0)
    return found


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m measure.ancestry", description=__doc__)
    p.add_argument("--app", default="charmap.exe")
    p.add_argument("--hops", type=int, default=8, help="how far up to look (the code stops at 5)")
    args = p.parse_args(argv)

    from argmind.env.windows import WindowsEyes                           # noqa: PLC0415

    proc = subprocess.Popen([args.app])
    time.sleep(2.5)
    handles = windows_of(proc.pid)
    if not handles:
        print(f"  {args.app} opened no visible window")
        proc.kill()
        return 2

    eyes = WindowsEyes()
    widgets = eyes.read(handles[0])
    uia = eyes._uia
    walker = uia.RawViewWalker
    print(f"  {args.app}: {len(widgets)} widgets\n")

    interesting = [w for w in widgets
                   if w.role in ("combo box", "button", "check box", "list", "edit")]
    for w in interesting:
        print(f"  {w.role} {w.name[:28]!r} centre {w.centre}")
        try:
            element = uia.ElementFromPoint(wt.POINT(*w.centre))
        except Exception as exc:                                          # noqa: BLE001
            print(f"      nothing at that point ({type(exc).__name__})")
            continue
        for hop in range(args.hops):
            if element is None:
                print(f"      {hop}: (no parent)")
                break
            try:
                name, role = element.CurrentName, element.CurrentLocalizedControlType
            except Exception:                                             # noqa: BLE001
                print(f"      {hop}: (died mid-read)")
                break
            hit = " <== name and role match" if (name == w.name and role == w.role) else ""
            print(f"      {hop}: {role[:16]:<16} {name[:34]!r}{hit}")
            if hit:
                print(f"      supports: {', '.join(supported(element, eyes._uia_mod)) or '(none)'}")
                break
            element = walker.GetParentElement(element)
        print()

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    return 0


if __name__ == "__main__":
    sys.exit(main())
