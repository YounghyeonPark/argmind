# SPDX-License-Identifier: Apache-2.0
"""Does the point we would click actually land on the thing we chose?

    python -m measure.pointing

**The other half of end-to-end accuracy, and it was never measured.** Choosing was measured at 18 of
28; the recorded limit was that "the click half is unmeasured, so 18/28 is a claim about choosing and
not about the whole loop". This closes that.

There are two ways the chain can be wrong and they are separate questions:

**(a) Is the rectangle's centre on the widget?** A `Widget.centre` is the midpoint of what UI
Automation reported. For a button that is obviously the button. For a list item that wraps, a menu
bar, or an element whose bounding box includes a child that covers it, the midpoint can sit on
something else — and the agent would then click a real thing that nobody chose. Asked with
`ElementFromPoint`, which is UIA answering its own question.

**(b) Does the cursor go where it is sent?** This process is **DPI-unaware** (`GetProcessDpiAwareness`
returns 0) on a display Windows may be scaling. UIA rectangles are physical pixels; `SetCursorPos`
from a DPI-unaware process can be virtualised. If those disagree, every click is offset by the
scaling factor and nothing lands — a failure that would look exactly like the model choosing badly.

(b) moves the real mouse pointer, so the original position is saved and put back. Nothing is
clicked: no button is ever pressed by this file.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes as wt
import collections
import pathlib
import sys

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

user32 = ctypes.windll.user32


def cursor_lands_where_it_is_sent(points) -> list[tuple]:
    """(b). Returns `(sent, got)` for each point; equal means the write path is not scaled."""
    keep = wt.POINT()
    user32.GetCursorPos(ctypes.byref(keep))
    out = []
    try:
        for x, y in points:
            user32.SetCursorPos(int(x), int(y))
            got = wt.POINT()
            user32.GetCursorPos(ctypes.byref(got))
            out.append(((int(x), int(y)), (got.x, got.y)))
    finally:
        user32.SetCursorPos(keep.x, keep.y)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m measure.pointing", description=__doc__)
    p.add_argument("--sample", type=int, default=40, metavar="N",
                   help="how many widgets to test the cursor path on (default 40)")
    args = p.parse_args(argv)

    from argmind.env.windows import WindowsEyes                          # noqa: PLC0415

    try:
        awareness = ctypes.c_int()
        ctypes.windll.shcore.GetProcessDpiAwareness(None, ctypes.byref(awareness))
        aware = awareness.value
    except Exception:                                                     # noqa: BLE001
        aware = "unknown"
    print(f"  process DPI awareness {aware}   virtual screen "
          f"{user32.GetSystemMetrics(78)}x{user32.GetSystemMetrics(79)}   "
          f"primary {user32.GetSystemMetrics(0)}x{user32.GetSystemMetrics(1)}")

    eyes = WindowsEyes()
    widgets = eyes.foreground()
    if not widgets:
        print("  nothing in the foreground window exposes a name. Put a normal window in front.")
        return 2
    print(f"  {len(widgets)} widgets in {widgets[0].window[:60]!r}\n")

    # ---- (b) the write path -------------------------------------------------------------------
    sample = widgets[:args.sample]
    pairs = cursor_lands_where_it_is_sent([w.centre for w in sample])
    off = [(s, g) for s, g in pairs if s != g]
    print(f"  (b) cursor: {len(pairs) - len(off)} of {len(pairs)} points came back exactly")
    if off:
        s, g = off[0]
        print(f"      first disagreement: sent {s}, read back {g}  -> the write path IS scaled, "
              f"and every click is offset")
    else:
        print("      the write path is not scaled — SetCursorPos and UIA share one coordinate space")

    # ---- (a) is the centre on the widget? -----------------------------------------------------
    uia = eyes._uia
    tally = collections.Counter()
    misses = []
    for w in widgets:
        pt = wt.POINT(*w.centre)
        try:
            at = uia.ElementFromPoint(pt)
            name = at.CurrentName or ""
            role = at.CurrentLocalizedControlType or ""
        except Exception:                                                 # noqa: BLE001
            tally["no element there"] += 1
            misses.append((w, "(nothing)", ""))
            continue
        if name == w.name:
            tally["the same thing"] += 1
        elif name and name in w.name or (w.name and w.name in name):
            tally["a containing or contained name"] += 1
        else:
            tally["something else"] += 1
            misses.append((w, name, role))

    print(f"\n  (a) what is actually at each widget's centre, over all {len(widgets)}:")
    for k, v in tally.most_common():
        print(f"      {v:>4}  ({100 * v / len(widgets):>4.1f}%)  {k}")
    if misses:
        print("\n      the first ten that resolve to something else:")
        for w, name, role in misses[:10]:
            print(f"        {w.role[:14]:<14} {w.name[:26]:<26} -> {role[:12]:<12} {name[:26]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
