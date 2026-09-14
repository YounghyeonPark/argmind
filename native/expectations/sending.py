"""What the blindness bookkeeping says right now, so the port can be held to the same sentence.

`caveat()` has three states and the difference between them is the point — a conversation run at
this prompt and this `loop.py`, one at this prompt under a different `loop.py`, or nothing at all.
A port that collapses the middle state loses the only thing that notices `loop.py` moving, because
the prompt fingerprint does not cover it.
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

import measure.sending as sending                                          # noqa: E402
from measure.choosing import prompt_tag                                    # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent / "sending.json"
tag = prompt_tag()
d = sending.deferral(tag)
OUT.write_text(json.dumps({
    "prompt": tag,
    "loop_print": sending.loop_now(),
    "caveat": sending.caveat(tag),
    "runs_at_this_prompt": d["runs_at_this_prompt"],
    "runs_at_this_prompt_and_loop": d["runs_at_this_prompt_and_loop"],
    "loop_prints_at_this_prompt": d["loop_prints_at_this_prompt"],
    "newest_run": d["newest_run"],
    "exercised_at_this_prompt": d["exercised_at_this_prompt"],
    "exercised_at_this_tree": d["exercised_at_this_tree"],
    #: Every conversation result and what it was taken at, so the port's file scan is checked
    #: rather than only its conclusion.
    "runs": [{"file": p.name, "prompt": q, "loop_print": l}
             for p, q, l, _m in sending.conversation_runs()],
}, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"  -> {OUT.name}")
print(f"    {tag} + loop {sending.loop_now()}: {len(d['runs_at_this_prompt'])} runs at this prompt, "
      f"{len(d['runs_at_this_prompt_and_loop'])} at this tree")
print(f"    {sending.caveat(tag).strip()}")
