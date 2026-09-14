"""The distance sweep: its cases, its fingerprints, and **the tables it prints**.

`judge()` and `counted` need nothing from here — every committed result carries its own verdicts,
so the Rust re-judges against the file that recorded them. What a result file does *not* carry is
the reading: `summarise()`, `paired()`, `turn_cost()` and `cost()` only ever printed, and a port
that got the paired power analysis subtly wrong would produce a plausible table nobody could check.

So the Python's own output is captured, line for line, and the Rust has to render the same
characters. That covers the arithmetic and the formatting at once — including the two things this
harness exists to keep visible: a cell whose control could not fail is **counted separately** and
never averaged in, and a point where every cell was void prints `void` rather than `0/0`.
"""
import contextlib
import glob
import io
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

import measure.conversation as conv                                        # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent / "conversation.json"
AXIS_KEY = {"d": "distance", "r": "remember", "both": "distance"}


def grab(fn, *args) -> str:
    """One table's text, exactly as the harness printed it."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn(*args)
    return buf.getvalue()


def prints_for(kinds):
    """`f` with only these kinds kept — what `--only` produced for the runs that used it."""
    keep = [c for c in conv.CASES if c["kind"] in kinds]
    blob = json.dumps([keep, conv.FILLER, conv.NO_RULE, conv.NO_REFUSAL],
                      ensure_ascii=False, sort_keys=True).encode()
    import hashlib
    return hashlib.sha256(blob).hexdigest()[:6]


files = []
for path in sorted(glob.glob(str(ROOT / "measure" / "conversation-*.json"))):
    d = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    rows = d["rows"]
    key = AXIS_KEY[d["axis"]]
    files.append({
        "file": pathlib.Path(path).name,
        "axis": d["axis"], "axis_key": key,
        "rows": len(rows),
        "repeats": sum(len(r.get("turns", [])) for r in rows),
        "turns": sum(len(t) for r in rows for t in r.get("turns", [])),
        "data_print": d.get("data_print"),
        #: Every row, in the harness's own display. The recorded defect it guards: a worst-of-three
        #: verdict printed beside repeat one's answer, so a cell that failed once read as though it
        #: had failed always.
        "show": "".join(grab(conv.show, r) for r in rows),
        "summarise": grab(conv.summarise, rows, key),
        "paired": grab(conv.paired, rows, key),
        "turn_cost": grab(conv.turn_cost, rows),
        "cost": grab(conv.cost, rows),
    })

OUT.write_text(json.dumps({
    "cases": conv.CASES,
    "filler": conv.FILLER,
    "no_rule": conv.NO_RULE,
    "no_refusal": conv.NO_REFUSAL,
    "data_print": conv.data_print(),
    "data_print_constraint": prints_for({"constraint"}),
    "data_print_reference": prints_for({"reference"}),
    "data_print_reoffer": prints_for({"reoffer"}),
    "worst": conv.WORST,
    "bad": {k: list(v) for k, v in conv.BAD.items()},
    "files": files,
}, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"  -> {OUT.name}")
print(f"    {len(conv.CASES)} cases, {len(conv.FILLER)} filler, f{conv.data_print()} "
      f"(constraint f{prints_for({'constraint'})}, reference f{prints_for({'reference'})})")
print(f"    {len(files)} committed results, {sum(f['rows'] for f in files)} rows, "
      f"{sum(f['repeats'] for f in files)} repeats, {sum(f['turns'] for f in files)} turns")
