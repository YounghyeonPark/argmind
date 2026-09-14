"""The recorder's scene list, and the two readings a stored corpus does not carry.

Most of what `replies.py` computes is already in the corpus: every row stores its own
`stream_print`, `arrived`, token counts and text sha, so the Rust re-derives those against the file
that recorded them and needs nothing from here.

What a corpus does **not** carry is the reading. `coverage()` replays every row through the
*shipped* stops — which is a different question from what the row was recorded under, since the
corpus was taken with the stops off — and `disagreement()` says how many repeats of one prompt came
back the same. Neither is stored anywhere, so the Python's answers are dumped and the Rust has to
match them.

The scenes are here because the recorder needs them and because a scene list edited between two
runs is two corpora sharing one name.
"""
import glob
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

import measure.replies as replies                                          # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent / "replies.json"

files = []
for path in sorted(glob.glob(str(ROOT / "measure" / "replies-*.json"))):
    d = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    rows = d["rows"]
    files.append({
        "file": pathlib.Path(path).name,
        "rows": len(rows),
        "arm": d.get("arm"),
        "coverage": replies.coverage(rows),
        "disagreement": replies.disagreement(rows),
    })

OUT.write_text(json.dumps({
    "conversations": [list(c) for c in replies.CONVERSATIONS],
    "singles": [list(s) for s in replies.SINGLES],
    "single_repeats": replies.SINGLE_REPEATS,
    "stress": [list(s) for s in replies.STRESS],
    "arms": {k: [v[0], v[1], v[2]] for k, v in replies.ARMS.items()},
    "composition": {k: list(v) for k, v in replies.COMPOSITION.items()},
    "think_label": {str(k): v for k, v in replies.THINK_LABEL.items()},
    "stops_off": replies.OFF,
    "files": files,
}, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"  -> {OUT.name}")
print(f"    {len(replies.CONVERSATIONS)} conversation scenes, {len(replies.SINGLES)} singles, "
      f"{len(replies.STRESS)} stress prompts, {len(replies.ARMS)} arms")
print(f"    {len(files)} corpora, {sum(f['rows'] for f in files)} recorded replies")
