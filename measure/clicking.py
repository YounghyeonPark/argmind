# SPDX-License-Identifier: Apache-2.0
"""Does a confirmed click actually do the thing? The last unmeasured link.

    python -m measure.clicking

**Everything else in this chain is measured and this was not.** Choosing is 18 of 28. The cursor
goes where it is sent, 40 of 40. A widget's centre resolves back to that widget 73% of the time. And
then nobody ever pressed the button: every run of the app so far was `--dry`, so *"the person
confirms, the click lands, the thing they wanted happens"* had never once been observed.

## What makes this safe to run on a machine somebody is using

**It only ever touches a window this script launched.** Character Map is opened by this process, and
before every single click the foreground window is checked against that process's own id — a click
goes to a pixel, so if anything else has come to the front the click would land in it. When that
happens the goal is skipped and reported as skipped, not retried.

## What "it worked" means, decided before running

A click that lands and does nothing is indistinguishable from a click that missed, so each goal
carries **two** checks, both written before anything ran:

* `changed` — did the window's set of named things differ afterwards at all? Weak, but it separates
  "something happened" from "nothing happened", which is the failure this exists to catch.
* `expected` — the specific thing that should be true afterwards. Strong, and **it can be wrong
  about the application rather than about the agent**: if Character Map's advanced view does not
  expose a "Search" control under that name, this reports a failure that is mine. Both are printed
  so the two cannot be confused.
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

from argmind.core import Said, Session                                    # noqa: E402
from argmind.core.intent import Intent                                    # noqa: E402
from argmind.core.llm import Local                                        # noqa: E402

user32 = ctypes.windll.user32
_ENUM = ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)


def appears(needle):
    return (f"a name containing {needle!r} appears",
            lambda before, after, hwnd: any(needle.lower() in n.lower() for n in after - before))


def minimised():
    """**Ask Windows, do not infer from the widget list.**

    The first version of this read the window afterwards and expected *nothing*, on the assumption
    that a minimised window has no on-screen elements. It reported a failure for a click that had
    very likely worked: a minimised window still answers UI Automation with some of its tree, so the
    expectation was wrong about Windows rather than about the agent.
    """
    return ("Windows reports the window minimised",
            lambda before, after, hwnd: bool(user32.IsIconic(hwnd)))


#: Written before the run. `charmap.exe` is chosen because it is small (21 named widgets), it is
#: launched and closed by this script, and nothing it does can lose somebody's work.
GOALS = [
    ("고급 옵션 켜줘", appears("search")),
    ("글꼴 목록 열어줘", appears("arial")),
    ("이 창 최소화해", minimised()),
]


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
    p = argparse.ArgumentParser(prog="python -m measure.clicking", description=__doc__)
    p.add_argument("--model", default="qwen2.5:7b")
    p.add_argument("--app", default="charmap.exe")
    p.add_argument("--no-patterns", action="store_true",
                   help="aim a synthetic click at the widget's centre, the way it worked before "
                        "UI Automation's control patterns were used. The comparison, not a switch "
                        "to leave off")
    args = p.parse_args(argv)

    from argmind.env.windows import WindowsEyes, WindowsHands              # noqa: PLC0415

    proc = subprocess.Popen([args.app])
    time.sleep(2.5)
    mine = windows_of(proc.pid)
    if not mine:
        print(f"  {args.app} opened no visible window")
        proc.kill()
        return 2

    eyes = WindowsEyes()
    hands = WindowsHands(live=True, max_per_second=4.0, patterns=not args.no_patterns)
    if not hands.pointing_is_exact():
        print("  the cursor does not go where it is sent on this display — refusing to click")
        proc.kill()
        return 2
    session = Session(eyes=eyes, model=Local(args.model), hands=hands)

    def ours() -> bool:
        """Is the window in front the one this script opened? Checked before every click."""
        handle = user32.GetForegroundWindow()
        out = wt.DWORD()
        user32.GetWindowThreadProcessId(handle, ctypes.byref(out))
        return out.value == proc.pid

    #: Two throwaway calls, because the first two cost 4.7 s and 5.9 s and this run should not
    #: measure the weights loading.
    for _ in range(2):
        session.model.ask("Reply with the single word: ready")

    how = "centre click only" if args.no_patterns else "control patterns, click as fallback"
    print(f"  {args.app} (pid {proc.pid}) | {args.model} | live, only this window | {how}\n")
    rows = []
    for goal, (expectation, holds) in GOALS:
        user32.SetForegroundWindow(mine[0])
        time.sleep(0.4)
        if not ours():
            print(f"  SKIPPED   {goal[:22]:<22} another window came to the front")
            rows.append({"goal": goal, "outcome": "skipped"})
            continue

        session.observe()
        #: Read the same way `after` is read — every window this process owns. Comparing one window
        #: before against all of them afterwards would report every widget of a second window as
        #: newly appeared, which is a difference the measurement invented rather than found.
        before = {w.name for h in windows_of(proc.pid) for w in eyes.read(h)}
        intent = session.decide(Said(goal))
        clicks = [e for e in intent.effects if e.kind == "click"]
        if not clicks:
            print(f"  NO CLICK  {goal[:22]:<22} said {intent.say[:34]!r}")
            rows.append({"goal": goal, "outcome": "declined", "said": intent.say})
            continue

        target = session.find(clicks[0].target, clicks[0].role)
        if target is None or not ours():
            print(f"  GONE      {goal[:22]:<22} {clicks[0].target[:24]!r} was not there to click")
            rows.append({"goal": goal, "outcome": "gone"})
            continue

        before_pattern, before_click = hands.by_pattern, hands.by_click
        session.perform(Intent("", [clicks[0]], clicks[0].because))
        via = hands.done[-1].get("via", "?") if hands.done else "?"
        del before_pattern, before_click
        time.sleep(0.9)
        #: **Every window this process owns, not just the one that was launched.** A combo box's
        #: drop-down is its own top-level window, so reading only the main one reports "nothing
        #: happened" for a list that did open. The first run could not tell that apart from a click
        #: that landed on a combo's text area instead of its arrow, which is a real failure — and
        #: could not tell them apart because this line asked the narrower question.
        after = {w.name for h in windows_of(proc.pid) for w in eyes.read(h)}
        changed = after != before
        met = holds(before, after, mine[0])
        gained = sorted(after - before)[:4]
        lost = sorted(before - after)[:4]
        rows.append({"goal": goal, "chose": target.name, "changed": changed, "expected": met,
                     "gained": gained, "lost": lost, "via": via,
                     "outcome": "worked" if met else ("changed only" if changed else "nothing")})
        mark = "WORKED" if met else ("CHANGED" if changed else "NOTHING")
        print(f"  {mark:<9} {goal[:22]:<22} {via:<7} {target.name[:20]!r:<22} "
              f"| {expectation}: {met}")
        why = hands.done[-1].get("why") if hands.done else None
        if why:
            print(f"            fell back because: {why}")
        #: **The delta, always.** A boolean cannot be diagnosed after the fact; the first run left
        #: two different causes indistinguishable because only the boolean was kept.
        if gained or lost:
            print(f"            + {gained}" if gained else "", end="")
            print(f"   - {lost}" if lost else "")
        elif not changed:
            print("            nothing at all appeared or disappeared")

        hands.press("escape")
        time.sleep(0.4)

    print()
    worked = sum(1 for r in rows if r.get("outcome") == "worked")
    moved = sum(1 for r in rows if r.get("changed"))
    print(f"  {worked} of {len(GOALS)} did the expected thing; {moved} changed the window at all")
    print(f"  of {hands.by_pattern + hands.by_click} clicks, {hands.by_pattern} went through a "
          f"control pattern and {hands.by_click} through a synthetic click")
    session.model.release()
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    return 0


if __name__ == "__main__":
    sys.exit(main())
