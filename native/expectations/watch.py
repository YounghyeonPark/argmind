"""What each watch says happened, in the words the prompt will carry.

A watch's description is not a log line — it becomes the **text of a `Trigger`**, so it goes into
the prompt and the model reads it. Two implementations that describe the same change differently
are two agents, and the difference is invisible from anywhere except here: nothing downstream
compares the wording against anything.

The obvious way to get it wrong is quoting. Python's `!r` gives `'Character Map'` and Rust's `{:?}`
gives `"Character Map"`, which is the same information in a different sentence — so the fixtures
below include names with quotes and non-ASCII in them, where the two conventions diverge further.
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

from argmind.core import Widget                                            # noqa: E402
from argmind.core.intent import Appeared, Vanished, WindowChanged, MIN_GAP  # noqa: E402


def w(name, window="Paint"):
    return Widget(name, "button", (0, 0, 10, 10), window)


#: `(label, before, after)`. Each pair is a change one of the watches is meant to notice, plus the
#: near misses that must produce nothing.
SCENES = [
    ("same window", [w("A")], [w("B")]),
    ("window changed", [w("A", "Paint")], [w("A", "Character Map")]),
    ("window went empty", [w("A", "Paint")], []),
    ("window arrived from nothing", [], [w("A", "Paint")]),
    ("appeared plain", [w("A")], [w("A"), w("Search")]),
    ("appeared, different case", [w("A")], [w("A"), w("SEARCH BOX")]),
    ("appeared, already there", [w("Search")], [w("Search")]),
    ("vanished plain", [w("A"), w("Search")], [w("A")]),
    ("vanished, nothing matching", [w("A")], [w("A")]),
    ("quotes in the name", [w("A")], [w("A"), w("say \"hello\"")]),
    ("korean in the name", [w("A")], [w("A"), w("검색 결과")]),
    ("backslash in the name", [w("A")], [w("A"), w("C:\\Search")]),
    ("window name with a quote", [w("A", "Paint")], [w("A", "it's here")]),
]

WATCHES = [
    ("window", WindowChanged()),
    ("appeared:search", Appeared("search")),
    ("appeared:검색", Appeared("검색")),
    ("vanished:search", Vanished("search")),
]

rows = []
for label, before, after in SCENES:
    for name, watch in WATCHES:
        rows.append({"scene": label, "watch": name, "said": watch.look(before, after),
                     "before": [[x.name, x.window] for x in before],
                     "after": [[x.name, x.window] for x in after]})

OUT = pathlib.Path(__file__).resolve().parent / "watch.json"
OUT.write_text(json.dumps({"min_gap": MIN_GAP, "rows": rows}, ensure_ascii=False, indent=1),
               encoding="utf-8")
print(f"  -> {OUT.name}")
print(f"    {len(rows)} scene/watch pairs, {sum(1 for r in rows if r['said'])} of them describing "
      f"something")
