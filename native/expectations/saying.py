"""What `saying.py` classifies with, and the cases it classifies.

The constants are dumped rather than retyped. `PUNCT` and the Korean number words are exactly the
kind of table that a port gets 95% right, and the 5% is a verdict that quietly changes — so both
implementations read one copy.
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

import measure.saying as saying                                            # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent / "saying.json"
OUT.write_text(json.dumps({
    "leaking": [list(c) for c in saying.LEAKING],
    "numbers": [list(c) for c in saying.NUMBERS],
    "token_chars": "".join(sorted(saying.TOKEN_CHARS)),
    "punct": "".join(sorted(saying.PUNCT)),
    "acks": sorted(saying.ACKS),
    "number_words": list(saying.NUMBER_WORDS),
    "leak_order": saying.LEAK_ORDER,
    "answer_order": saying.ANSWER_ORDER,
    "data_print": saying.data_print(),
}, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"  -> {OUT.name}")
print(f"    {len(saying.LEAKING)} leaking cases, {len(saying.NUMBERS)} number cases, "
      f"f{saying.data_print()}")
print(f"    {len(saying.ACKS)} acks, {len(saying.NUMBER_WORDS)} number words, "
      f"{len(saying.PUNCT)} punctuation characters")
