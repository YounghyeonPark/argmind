"""The 104 cases, and the two vocabularies a recorded run does not carry as data.

Every committed `reoffer-*.json` stores its own `cases`, so the Rust scorer already re-judges
against the file that recorded them. What the results do **not** carry is the case list as the
harness holds it today — which is what a *new* run needs — nor `WORST` and `GOOD`, the orderings
that decide worst-of-N and which single verdict counts as correct.

Which group borrows which vocabulary is the pre-registration: `REFUSE` is scored by `MEMORY`'s
rule, `FARACCEPT` by `ACCEPT`'s, `RESTATE` by `ANEW`'s. Dumping it here rather than retyping it in
Rust is the point — a group quietly given its own ordering is a group whose failures can no longer
be compared with anything.

`shape_of` is dumped over every offered line in every committed result, because it is the check
that says whether two arms differ in the transcript at all, and a port that got it subtly wrong
would print "no treatment was applied" beside a real p-value.
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

import measure.reoffer as reoffer                                          # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent / "reoffer.json"

#: Every distinct offered line in every committed result, and what `shape_of` makes of it. The
#: Rust has to agree on all of them; a quoted widget name is the only thing that may disappear.
shapes = {}
for path in sorted(glob.glob(str(ROOT / "measure" / "reoffer-*.json"))):
    d = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    for row in d["rows"]:
        for rep in row["repeats"]:
            for line in rep.get("offered_lines", ()):
                shapes[line] = reoffer.shape_of(line)
            for line in rep.get("slot", ()) or ():
                shapes[line] = reoffer.shape_of(line)

#: Hand-written, because the corpus has no unbalanced quote and cannot fabricate one.
AWKWARD = ["", '  ', 'to click the button "Copy"', 'no quotes at all',
           'two "a" and "b" names', 'an unterminated "quote', '"" empty', 'trailing "']
for line in AWKWARD:
    shapes[line] = reoffer.shape_of(line)

OUT.write_text(json.dumps({
    "cases": reoffer.CASES,
    "filler_far": reoffer.FILLER_FAR,
    "groups": list(reoffer.GROUPS),
    "worst": reoffer.WORST,
    "good": reoffer.GOOD,
    "data_print": reoffer.data_print(),
    "remember_turns": reoffer.REMEMBER_TURNS,
    "no_slot": reoffer.NO_SLOT,
    "k_for_significance": reoffer.K_FOR_SIGNIFICANCE,
    "shapes": shapes,
    "slot_shape_of_nothing": reoffer.slot_shape([]),
}, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"  -> {OUT.name}")
print(f"    {len(reoffer.CASES)} cases over {len({c['screen'] for c in reoffer.CASES})} screens, "
      f"f{reoffer.data_print()}, {len(reoffer.FILLER_FAR)} far-filler lines")
print(f"    {len(shapes)} distinct offered/slot lines shaped")
