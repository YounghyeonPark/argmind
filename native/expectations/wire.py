r"""The exact bytes `Local.ask` puts on the wire, for a set of prompts.

**The transport is not unverifiable offline and I said it was.** The request body is a pure function
of the settings and the prompt, so it can be compared byte for byte without a server — and it has
two traps that only show that way. `json.dumps` defaults to `ensure_ascii=True`, so every Korean
character leaves as `\uXXXX`; and a Python dict serialises in **insertion** order, where
`serde_json`'s map is sorted. A port that gets either wrong sends a valid request that is not the
same request, and nothing downstream would notice.

What this cannot check is the socket. One live call does that, and the response side is already
covered: `Reading` is replayed against 128 recorded streams in `native/argmind-core/tests/stops.rs`.
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

llm = importlib.import_module("argmind.core.llm")
OUT = pathlib.Path(__file__).resolve().parent / "wire.json"

PROMPTS = [
    ("plain ascii", "Reply with the single word: ready"),
    ("korean", "복사 눌러줘"),
    ("mixed, with the quotes a prompt actually contains",
     'Reply with only JSON: {"say": "<what you say back>", "do": [ ... ]}'),
    ("a newline and a tab", "line one\nline two\tend"),
    ("an emoji, a surrogate pair once escaped", "괜찮아 👍"),
    ("a backslash", r'a \ b'),
    ("empty", ""),
]

SETTINGS = [
    ("shipped", {}),
    ("thinking off", {"think": False}),
    ("thinking on", {"think": True}),
    ("another model and window", {"model": "qwen3:0.6b", "ctx": 4096, "max_tokens": 512,
                                  "keep_alive": "5m"}),
]

cases = []
for label, prompt in PROMPTS:
    for sname, over in SETTINGS:
        model = over.get("model", "qwen3:8b")
        kw = {k: v for k, v in over.items() if k != "model"}
        local = llm.Local(model, **kw)
        #: Rebuilt here exactly as `ask` builds it. Kept next to the library rather than inside it
        #: because `ask` sends the body rather than returning it, and adding a hook to a file for a
        #: test to reach is how a measurement starts changing what it measures.
        payload = {"model": local.model, "messages": [{"role": "user", "content": prompt}],
                   "stream": True, "keep_alive": local.keep_alive,
                   "options": {"num_predict": local.max_tokens, "num_ctx": local.ctx,
                               "temperature": 0}}
        if local.think is not None:
            payload["think"] = local.think
        cases.append({"label": f"{label} / {sname}", "prompt": prompt, "model": model,
                      "ctx": local.ctx, "max_tokens": local.max_tokens,
                      "keep_alive": local.keep_alive, "think": local.think,
                      "body": json.dumps(payload)})

OUT.write_text(json.dumps(cases, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"  -> {OUT.name}")
print(f"    {len(cases)} bodies, {len(PROMPTS)} prompts x {len(SETTINGS)} settings")
print(f"    longest {max(len(c['body']) for c in cases)} bytes; "
      f"{sum(1 for c in cases if chr(92) + 'u' in c['body'])} carry escaped non-ASCII")
