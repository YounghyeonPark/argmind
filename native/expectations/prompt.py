"""The six probes the prompt fingerprint is taken over, and the tag they produce.

`measure/choosing.py` hashes the prompt as `converse()` renders it over six probes, plus
`MAX_LISTED` and `MOST_AT_ONCE`, and calls the first six hex characters `r`. **A Rust renderer that
produces the same tag has provably not changed a character of what the model is sent** — which is
the risk worth pinning here above all others, because prompt edits in this project are measured as
never free: five in a row each raised one number and lowered another, and three were nearly recorded
as improvements.

The tag is read out of the shipped Python rather than written down, so this file cannot claim a
match against a number nobody checked.
"""
import importlib
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

import measure.choosing as choosing                                         # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent / "prompt.json"

#: `import_module`, never `from argmind.core import converse` — `core/__init__.py` re-exports a
#: *function* of that name over the *module* of that name, so the second form silently hands back
#: the function. The scheme this replaced carried the same warning and it still caught me once.
converse = importlib.import_module("argmind.core.converse")


def as_widget(x) -> dict:
    return {"name": x.name, "role": x.role, "rect": list(x.rect), "window": x.window}


probes = []
for label, trigger, widgets, history in choosing.PROBES:
    probes.append({"label": label, "asked": trigger.asked, "text": trigger.text,
                   "widgets": [as_widget(x) for x in widgets],
                   "history": [list(p) for p in history]})

record = choosing.prompt_record()
OUT.write_text(json.dumps({
    "scheme": choosing.PROMPT_LETTER,
    "tag": choosing.prompt_tag(),
    "record_chars": len(record),
    "record_bytes": len(record.encode("utf-8")),
    "max_listed": converse.MAX_LISTED,
    "most_at_once": converse.MOST_AT_ONCE,
    "probes": probes,
    #: The header the record opens with, so the Rust side rebuilds the hashed text rather than
    #: guessing at its shape.
    "header": f"MAX_LISTED = {converse.MAX_LISTED}\nMOST_AT_ONCE = {converse.MOST_AT_ONCE}\n",
}, ensure_ascii=False, indent=1), encoding="utf-8")

print(f"  -> {OUT.name}")
print(f"    {len(probes)} probes, {choosing.prompt_tag()}, record {len(record.encode('utf-8'))} bytes")
