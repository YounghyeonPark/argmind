"""What `loop.py`'s pure half does, written out for the Rust side to match.

Stage 3 of `docs/findings/porting-the-core.md`, and it is deliberately **not all of `loop.py`**.
`observe`, `poll`, `perform` and `think` call `Eyes`, `Hands` and a model — they are the seam, and
the seam is measured at 0.144 ms a crossing against 0.032 ms of work, which is an argument to have
before writing it rather than after. What is here is the half with no seam at all and the worst
protection: **three of `remember_doing()`'s four channels have never been fed by any harness**, and
`recent()` is where the line-slicing window cut a turn in half in 10.7% of prompts.

`find()` is included in its pure form. Its only I/O is `self.eyes.foreground()`; given the fresh
list and what was last seen, the resolution — name, then role, then nearest to where a same-named
thing was — is arithmetic. That decomposition is what a port would have anyway, and writing it down
is half the porting decision.

**The cases come from `tests/test_loop_golden.py`'s own scene data**, imported rather than retyped,
so this exercises the branches `architect` chose when it wrote the transcript rather than the ones
I would have thought of. Sweeps are added where a sweep is cheap and the golden pins only a few
points.

    python native/expectations/all.py && cargo test --manifest-path native/Cargo.toml
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

for s in (sys.stdout, sys.stderr):
    try:
        s.reconfigure(encoding="utf-8")
    except Exception:
        pass

from argmind.core.intent import Effect                                      # noqa: E402
from argmind.core.loop import Done, ING, PAST, PLAIN, _describe, recent    # noqa: E402
from tests.test_loop_golden import (CHANNEL_EFFECTS, DESK,                 # noqa: E402
                                    WINDOW_HISTORY, sitting, w)

OUT = pathlib.Path(__file__).resolve().parent / "loop.json"


def as_effect(e) -> dict:
    return {"kind": e.kind, "target": e.target or "", "role": e.role or "", "text": e.text or ""}


def as_widget(x) -> dict:
    #: The window is part of a `Widget` and leaving it out was the spike's shortcut. It is not
    #: cosmetic: `Widget.__eq__` ignores it while `Session.confirm` checks it separately.
    return {"name": x.name, "role": x.role, "rect": list(x.rect), "window": x.window}


# ---- 1. the four channels, and the tense that carries the state ----------------------------------
#
# The wording is the whole point: the past tense belongs to `did` and to nothing else, and every
# other line says in words that nothing happened. A port that gets a space or a dash wrong writes a
# record the model reads differently, and nothing else would catch it.

describes = []
for label, effect in CHANNEL_EFFECTS:
    for form, name in ((PAST, "past"), (PLAIN, "plain"), (ING, "ing")):
        describes.append({"label": f"{label}/{name}", "effect": as_effect(effect), "form": form,
                          "widget": None, "want": _describe(effect, form=form)})
    #: `did` is the only channel that passes a widget, and it is the resolved one, not the named
    #: one — the role in the line comes from the world rather than from what the model said.
    if effect.kind == "click":
        found = w(effect.target, role="combo box")
        describes.append({"label": f"{label}/past+widget", "effect": as_effect(effect),
                          "form": PAST, "widget": as_widget(found),
                          "want": _describe(effect, found, PAST)})

# ---- 2. recent(): a window in turns, cut where a turn opens --------------------------------------
#
# `None` is everything and `0` is nothing, and the second one is a bug fixed rather than a choice:
# `history[-0:]` is the whole list, so the setting a person reaches for to turn memory off turned it
# all the way up. Swept past both ends because that is where it went wrong.

windows = []
for turns in [None, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 99]:
    windows.append({"turns": turns, "history": [list(p) for p in WINDOW_HISTORY],
                    "want": [list(p) for p in recent(WINDOW_HISTORY, turns)]})
#: A history that opens no turn at all — every line a record, none of them a person speaking.
no_turns = [p for p in WINDOW_HISTORY if p[0] not in ("them", "noticed")]
for turns in [None, 0, 1, 3]:
    windows.append({"turns": turns, "history": [list(p) for p in no_turns],
                    "want": [list(p) for p in recent(no_turns, turns)]})
windows.append({"turns": 3, "history": [], "want": []})


# ---- 3. find(): the role is half the key, and the nearest duplicate wins --------------------------
#
# The measurement that forces this: over the second a call takes, a listed widget is gone 13 times
# in 14. And the role is there because a `text` labelled `Font :` in front of a `combo box` of the
# same name took the click and did nothing — indistinguishable from a click that missed.

def resolve(fresh, seen, name, role):
    """`Session.find`'s body with `self.eyes.foreground()` hoisted out. Kept line for line."""
    same = [x for x in fresh if x.name == name and (not role or x.role == role)]
    if not same:
        return None
    if len(same) == 1:
        return same[0]
    was = next((x.centre for x in seen if x.name == name and (not role or x.role == role)),
               same[0].centre)
    return min(same, key=lambda x: (x.centre[0] - was[0]) ** 2 + (x.centre[1] - was[1]) ** 2)


FONT_TRAP = [w("Font :", role="text", rect=(0, 0, 10, 10)),
             w("Font :", role="combo box", rect=(100, 0, 10, 10))]
TWINS = [w("Copy", rect=(0, 0, 10, 10)), w("Copy", rect=(500, 0, 10, 10)),
         w("Copy", rect=(250, 0, 10, 10))]

finds = []
for label, fresh, seen, name, role in [
    ("the desk, by name alone", DESK, DESK, "닫기", ""),
    ("the desk, name and role", DESK, DESK, "닫기", "button"),
    ("a role nothing has", DESK, DESK, "닫기", "combo box"),
    ("a name nothing has", DESK, DESK, "없는것", ""),
    ("the Font : trap, name alone", FONT_TRAP, FONT_TRAP, "Font :", ""),
    ("the Font : trap, with the role", FONT_TRAP, FONT_TRAP, "Font :", "combo box"),
    ("the role filters, it does not only break ties", FONT_TRAP, [], "Font :", "text"),
    ("duplicates, nearest to where it was", TWINS, [TWINS[1]], "Copy", ""),
    ("duplicates, nothing ever seen", TWINS, [], "Copy", ""),
    ("duplicates, seen somewhere between", TWINS, [w("Copy", rect=(240, 0, 10, 10))], "Copy", ""),
    ("nothing on screen at all", [], DESK, "닫기", ""),
]:
    got = resolve(fresh, seen, name, role)
    finds.append({"label": label, "fresh": [as_widget(x) for x in fresh],
                  "seen": [as_widget(x) for x in seen], "name": name, "role": role,
                  "want": as_widget(got) if got is not None else None})

# ---- 4. the four channels, whole — driven through the real Session ------------------------------
#
# **This section exists because the port passed without it and should not have.** With only
# `describe()` pinned, changing the offer line's *"they have not accepted"* to anything else was not
# caught: the channel wording lives in `remember_doing()`, not in `_describe()`, and it is the least
# protected text in the file — three of the four channels have never been fed by any harness.
#
# Driven through a real `Session` rather than by rebuilding the four format strings here, because
# rebuilding them would compare my copy against the Rust copy and leave the library out of it.

channels = []
for channel in ("offered", "refused", "turned_down"):
    for what, effect in CHANNEL_EFFECTS:
        session = sitting()
        session.remember_doing(**{channel: [effect]})
        channels.append({"label": f"{channel}/{what}", "channel": channel,
                         "effect": as_effect(effect), "target": None,
                         "want": [list(p) for p in session.history]})
for what, effect in CHANNEL_EFFECTS:
    session = sitting()
    target = w("Copy", role="combo box") if effect.kind == "click" else None
    session.remember_doing(did=[Done(effect, target)])
    channels.append({"label": f"did/{what}", "channel": "did", "effect": as_effect(effect),
                     "target": as_widget(target) if target is not None else None,
                     "want": [list(p) for p in session.history]})

#: All four in one call, which is where the write order shows. `offered` goes first, so a caller
#: batching an offer with a settlement has the offer closed before it is ever rendered — recorded
#: by the golden as a hazard rather than a bug, and reproduced here rather than tidied.
together = sitting()
together.remember_doing(did=[Done(Effect("click", target="Cut", role="button"), w("Cut"))],
                        refused=[Effect("type", text="bye")],
                        offered=[Effect("click", target="Copy", role="button")],
                        turned_down=[Effect("key", text="tab")])
order = {"label": "all four in one call", "want": [list(p) for p in together.history]}

OUT.write_text(json.dumps({"describe": describes, "recent": windows, "find": finds,
                           "channels": channels, "order": order},
                          ensure_ascii=False, indent=1), encoding="utf-8")
print(f"  -> {OUT.name}")
print(f"    describe  {len(describes)} cases over {len(CHANNEL_EFFECTS)} effect kinds x 3 forms")
print(f"    recent    {len(windows)} windows")
print(f"    find      {len(finds)} resolutions")
print(f"    channels  {len(channels)} whole lines, through a real Session, plus the write order")
