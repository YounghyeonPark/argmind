"""What the shipped Python scanner makes of every reply, written out for the Rust side to match.

Two sources, and they cover different things.

**The corpus** — `measure/replies-*.json`, 128 real replies recorded at `732efbd`. It is the only
thing that can say the port is right about *what the model actually sends*, and it catches the trap
this port is most likely to fall into: `_Balanced` iterates a Python `str`, so `start` and `end` are
**character** indices and `Reply.json()` slices with them. 94 of the 128 are non-ASCII, mostly
Korean at three bytes a code point, so a scanner written over `as_bytes()` agrees on every ASCII
fixture and is wrong on three quarters of the real ones. Measured: that mutant fails 94 rows.

**Synthetic cases** — and they are here because the corpus is *provably* blind to them. Dropping the
scanner's string tracking entirely changes nothing across all 128 replies: no recorded reply
contains a brace or a backslash inside a string, and a corpus cannot fabricate one. `measurement`
named that survivor from the Python side; the same mutant survives on the Rust side, which is two
implementations agreeing about a gap rather than one noticing it. So the hard strings are written by
hand, their expectations come from the shipped Python like every other row, and they are labelled
`synthetic` so nobody counts them as evidence about the model.

Run this, then the crate:

    python native/reply-spike/expectations.py
    cargo run --release --manifest-path native/reply-spike/Cargo.toml -- \
        native/reply-spike/expectations.json
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

from argmind.core.llm import Reply, _Balanced                               # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent / "expectations.json"

#: The axis the corpus cannot reach. A brace inside a string is not a brace, and an escaped quote
#: does not end one — get either wrong and the object closes early, which is the failure that once
#: made every action disappear while the model was answering perfectly.
SYNTHETIC = [
    ('a brace inside a string', '{"say": "press } to close"}'),
    ('an escaped quote', '{"say": "a \\" quote", "do": []}'),
    ('a trailing escaped backslash', '{"say": "back\\\\"}'),
    ('braces both ways inside a string', '{"a": "}{"}'),
    ('nested, with prose either side', 'sure thing {"say": "ok", "do": [{"click": 0}]} done'),
    ('the nesting that broke the regex', '{"say": "ok", "do": [{"click": 0}]}'),
    ('unterminated string', '{"a": "no close'),
    ('no object at all', 'I can\'t see that here.'),
    ('the empty object', '{}'),
    ('a second object after the first', '{"a": 1} {"b": 2}'),
]


def text_of(lines) -> str:
    """Rebuild what `ask` accumulated: the content of every stream line, in order."""
    out = []
    for entry in lines:
        if isinstance(entry, str):
            out.append(entry)
        elif isinstance(entry, dict):
            out.append(entry.get("c") or "")
    return "".join(out)


def expectation(key: str, text: str, source: str) -> dict:
    scan = _Balanced()
    scan.feed(text)
    #: Through the shipped `Reply` as well as the raw scanner, so this file cannot drift from what
    #: the library actually does with the two numbers.
    return {
        "key": key,
        "source": source,
        "text": text,
        "chars": len(text),
        "bytes": len(text.encode("utf-8")),
        "start": scan.start,
        "end": scan.end,
        "reply_json_ok": Reply(text.strip(), 0.0, 0.0, 0, 0, "stop").json() is not None,
    }


cases = [expectation(f"synthetic/{name}", text, "synthetic") for name, text in SYNTHETIC]
for path in sorted(glob.glob(str(ROOT / "measure" / "replies-*.json"))):
    d = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    stem = pathlib.Path(path).stem.split("-r322a4c")[0]
    for row in d["rows"]:
        cases.append(expectation(f"{stem}/{row['key']}", text_of(row["lines"]), "recorded"))

OUT.write_text(json.dumps(cases, ensure_ascii=False), encoding="utf-8")
recorded = [c for c in cases if c["source"] == "recorded"]
wide = sum(1 for c in recorded if c["bytes"] != c["chars"])
print(f"  {len(cases)} cases -> {OUT.name}")
print(f"    recorded  {len(recorded)}, of which {wide} are non-ASCII")
print(f"    synthetic {len(cases) - len(recorded)}, covering strings and escapes the corpus has none of")
print(f"    an object was found in {sum(1 for c in cases if c['start'] >= 0)}, "
      f"Reply.json() parsed {sum(1 for c in cases if c['reply_json_ok'])}")
