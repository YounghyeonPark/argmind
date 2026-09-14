"""What the shared measuring machinery computes, so the Rust版 can be held to it.

Every harness in `measure/` needs the same few things before it can measure anything: the captured
screens, a fingerprint that changes when they do, a fingerprint of the prompt, the cost of a call,
and a result path that refuses to overwrite. Porting them one harness at a time would reimplement
this five times and the five would drift — which is the defect `_Balanced`'s docstring already
records from the other direction.
"""
import hashlib
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

from measure.capture import load                                            # noqa: E402
import measure.conversation as conversation                                 # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent / "measure.json"
llm = importlib.import_module("argmind.core.llm")

screens = []
for name, window, widgets in load():
    screens.append({"name": name, "window": window, "widgets": len(widgets),
                    "first": [{"name": w.name, "role": w.role, "rect": list(w.rect)}
                              for w in widgets[:3]]})

#: One `Reply` with every field set, so the cost record's shape is pinned rather than described.
reply = llm.Reply("{\"say\": \"ok\"}", 0.036, 0.478, 373, 29, "stop", 512)
import measure.choosing as choosing                                         # noqa: E402

OUT.write_text(json.dumps({
    "screen_print": conversation.screen_print(),
    "prompt_tag": choosing.prompt_tag(),
    "screens": screens,
    "spend": choosing.spend(reply),
    #: The same call with a stop of the reader's own: `prompt_tokens` absent rather than zero.
    "spend_stopped": choosing.spend(llm.Reply("{}", 0.036, 1.2, None, 41, "degenerate", 0)),
    "spend_none": choosing.spend(None),
}, ensure_ascii=False, indent=1), encoding="utf-8")

print(f"  -> {OUT.name}")
print(f"    {len(screens)} screens, screen_print {conversation.screen_print()}, "
      f"prompt {choosing.prompt_tag()}")
print(f"    spend keys: {sorted(choosing.spend(reply))}")
