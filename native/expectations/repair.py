"""The three case lists `repair.py` measures, and the fingerprint over them.

Dumped rather than retyped so both implementations run **the same experiment**. `data_print` is `f`
in the filename and it exists because a case list edited between two runs is two experiments
sharing one name — which is how three baselines were destroyed in this directory.

The Rust side recomputes the fingerprint from this file, so a mismatch means either the cases moved
or the serialisation did, and both are things a port gets silently wrong.
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

import measure.repair as repair                                            # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent / "repair.json"
OUT.write_text(json.dumps({
    "cases": repair.CASES,
    "korean": repair.KOREAN,
    "english": repair.ENGLISH,
    "language_screen": repair.LANGUAGE_SCREEN,
    "original_korean": repair.ORIGINAL_KOREAN,
    "data_print": repair.data_print(),
    "remember_turns": repair.REMEMBER_TURNS,
}, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"  -> {OUT.name}")
print(f"    {len(repair.CASES)} repair cases, {len(repair.KOREAN)} korean, "
      f"{len(repair.ENGLISH)} english, f{repair.data_print()}")
