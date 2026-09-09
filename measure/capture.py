# SPDX-License-Identifier: Apache-2.0
"""Save real widget lists so choosing can be measured without a live desktop.

    python -m measure.capture           watch for 30 s and keep every distinct window
    python -m measure.capture --for 90  longer, if you want to alt-tab through more of them

**Why a fixture and not a live run.** The question step 4 asks is *"given what the environment saw,
does the model pick the right thing?"* — one question, about choosing. A live run answers it mixed
with two others: whether the window was in front at the right moment, and whether the click landed.
Separating them is not tidiness, it is the only way a bad number points at the part that caused it.

A fixture also makes the measurement **repeatable when the prompt changes**, which it will. A number
that cannot be re-run is a number that stops being true without anybody noticing.

**These are real screens, captured from a working desktop, not screens invented to be answerable.**
A hand-written surface would contain exactly the widgets a goal needs and nothing else, and would
measure the fixture rather than the model. `--for` runs long enough to alt-tab.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

HERE = pathlib.Path(__file__).resolve().parent
SCREENS = HERE / "screens"


def _slug(title: str) -> str:
    keep = [c if c.isalnum() or c in " -_" else "-" for c in title.lower()]
    return ("".join(keep).strip().replace(" ", "-")[:48] or "untitled")


def capture(seconds: float, every: float) -> int:
    from argmind.env.windows import WindowsEyes                          # noqa: PLC0415

    SCREENS.mkdir(parents=True, exist_ok=True)
    eyes = WindowsEyes()
    kept: dict[str, list] = {}
    print(f"  watching for {seconds:.0f} s — alt-tab through the windows you want captured\n")
    end = time.time() + seconds
    while time.time() < end:
        widgets = eyes.foreground()
        if widgets:
            title = widgets[0].window
            #: Keep the RICHEST reading of a window, not the first. A window caught mid-paint
            #: exposes a fraction of its widgets, and that would quietly make every goal against it
            #: unanswerable — a failure that looks like the model choosing badly.
            if len(widgets) > len(kept.get(title, [])):
                kept[title] = widgets
                print(f"  {len(widgets):>4} widgets  {title[:70]}")
        time.sleep(every)

    for title, widgets in kept.items():
        path = SCREENS / f"{_slug(title)}.json"
        path.write_text(json.dumps({
            "window": title,
            "widgets": [{"name": w.name, "role": w.role, "rect": list(w.rect)} for w in widgets],
        }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n  {len(kept)} windows -> {SCREENS}")
    if eyes.vanished:
        print(f"  ({eyes.vanished} elements died between being listed and being read — "
              f"that is the staleness this design assumes, not an error)")
    return 0


def load():
    """Every captured screen, as `(name, window, [Widget])`."""
    from argmind.core import Widget                                      # noqa: PLC0415

    for path in sorted(SCREENS.glob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        yield (path.stem, raw["window"],
               [Widget(w["name"], w["role"], tuple(w["rect"]), raw["window"])
                for w in raw["widgets"]])


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m measure.capture", description=__doc__)
    p.add_argument("--for", dest="seconds", type=float, default=30.0, metavar="S")
    p.add_argument("--every", type=float, default=1.5, metavar="S")
    return capture(p.parse_args(argv).seconds, p.parse_args(argv).every)


if __name__ == "__main__":
    sys.exit(main())
