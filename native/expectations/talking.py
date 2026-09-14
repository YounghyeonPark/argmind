"""The things a person says that are not requests, and the two screens they are said on.

**The other half of `choosing`.** That one measures whether it acts when it should; this measures
whether it keeps its hands off when it should not act at all — and citing either alone is
incomplete by construction, because a model that clicks nothing scores perfectly on one and zero on
the other.

The two screens are the **smallest and the largest** captured, chosen by widget count rather than
by name, so the pair moves with the fixtures instead of going stale beside them.
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

from measure.capture import load                                           # noqa: E402
import measure.talking as talking                                          # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent / "talking.json"
screens = {name: widgets for name, _window, widgets in load()}
pick = sorted(screens.items(), key=lambda kv: len(kv[1]))
where = [pick[0][0], pick[-1][0]]

OUT.write_text(json.dumps({"talk": [list(t) for t in talking.TALK], "where": where,
                           "widget_counts": {k: len(v) for k, v in screens.items()}},
                          ensure_ascii=False, indent=1), encoding="utf-8")
print(f"  -> {OUT.name}")
print(f"    {len(talking.TALK)} lines on {where}")
