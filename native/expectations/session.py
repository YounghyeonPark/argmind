"""A whole turn, driven through the real `Session`, written out for the Rust side to match.

Stage 3b — the half of `loop.py` that crosses the seam. `observe`, `poll`, `decide`, `perform`,
`think` and `exchange` all call `Eyes`, `Hands` or a model, so this is where a port stops being
arithmetic and starts being an interface. The crossing is not free: 152 widgets cost 0.144 ms
through JSON against 0.032 ms for everything the core does in a turn, and `Session` reads the
screen **up to three times a turn** — once to observe, once to re-resolve before acting, once to
confirm. That is the design's whole point and it is also the port's whole cost.

**The fakes are the golden transcript's**, imported rather than rewritten: `ScriptedEyes` returns a
different screen on each read, because the three reads disagreeing is the thing under test and a
fake that cannot disagree cannot express it. `DeclaringHands` refuses a key it never published, so
the unknown-key filter fails on the press rather than on an assertion nobody wrote.

What is compared is everything a turn produces: the `Exchange`'s fields, the history lines, the
three counters, what the hands were actually told to do, and how many times the eyes were read.
Timings are not compared — `look_s` is the environment's own number and the other two are wall
clock, which is not a property of the port.

    python native/expectations/all.py && cargo test --manifest-path native/Cargo.toml
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

from argmind.core import Said, Session                                      # noqa: E402
from argmind.core.intent import Perceived                                   # noqa: E402
from tests.test_loop_golden import (DESK, DeclaringHands, ScriptedEyes,     # noqa: E402
                                    ScriptedHands, ScriptedModel, says, w)

OUT = pathlib.Path(__file__).resolve().parent / "session.json"

GONE = [x for x in DESK if x.name != "닫기"]
MOVED = [w("닫기", rect=(300, 200, 40, 30)), *[x for x in DESK if x.name != "닫기"]]
TWO_CLOSE = [w("닫기", rect=(10, 20, 40, 30)), w("닫기", rect=(400, 20, 40, 30)),
             *[x for x in DESK if x.name != "닫기"]]
OTHER_WINDOW = [w("닫기", rect=(10, 20, 40, 30), window="Other")]
#: A `text` label in front of a `combo box` of the same name — the pair that took a click
#: and did nothing, and the reason the role is half of the key.
FONT_TRAP = [w("Font :", role="text", rect=(0, 0, 10, 10)),
             w("Font :", role="combo box", rect=(100, 0, 10, 10))]
#: The same pair after a scroll: the combo box has moved far, the label has moved to roughly
#: where the combo box used to be.
FONT_MOVED = [w("Font :", role="text", rect=(90, 0, 10, 10)),
              w("Font :", role="combo box", rect=(300, 0, 10, 10))]
#: A label and a combo box of one name, and then TWO combo boxes of that name. Which twin wins
#: depends on whose old position `find()` looks up — and that lookup has its own copy of the role.
TWIN_SEEN = [w("Font :", role="text", rect=(0, 0, 10, 10)),
             w("Font :", role="combo box", rect=(200, 0, 10, 10))]
TWIN_FRESH = [w("Font :", role="combo box", rect=(10, 0, 10, 10)),
              w("Font :", role="combo box", rect=(205, 0, 10, 10))]

#: Each scenario is a script the Rust side replays with its own fakes fed the same way.
SCENARIOS = [
    ("a plain click, dry",
     [DESK], [says("닫을게요", {"click": 0})], {"live": False, "keys": None}, True,
     [("said", "닫아줘")]),
    ("a plain click, live and the world moved",
     [DESK, DESK, GONE], [says("닫을게요", {"click": 0})], {"live": True, "keys": None}, True,
     [("said", "닫아줘")]),
    ("a plain click, live and nothing moved",
     [DESK, DESK, DESK], [says("닫을게요", {"click": 0})], {"live": True, "keys": None}, True,
     [("said", "닫아줘")]),
    ("live, and the foreground window changed underneath",
     [DESK, DESK, OTHER_WINDOW], [says("닫을게요", {"click": 0})], {"live": True, "keys": None},
     True, [("said", "닫아줘")]),
    ("the target vanished between deciding and acting",
     [DESK, GONE], [says("닫을게요", {"click": 0})], {"live": False, "keys": None}, True,
     [("said", "닫아줘")]),
    ("the target moved, and the nearest same-named one wins",
     [DESK, MOVED], [says("닫을게요", {"click": 0})], {"live": False, "keys": None}, True,
     [("said", "닫아줘")]),
    ("two of the same name, resolved against where it was seen",
     [DESK, TWO_CLOSE], [says("닫을게요", {"click": 0})], {"live": False, "keys": None}, True,
     [("said", "닫아줘")]),
    ("typing and a key, in one intent",
     [DESK], [says("입력할게요", {"type": "hello"}, {"key": "ENTER"})],
     {"live": False, "keys": None}, True, [("said", "hello 치고 엔터")]),
    ("a key the environment does not have",
     [DESK], [says("탭 누를게요", {"key": "tab"})],
     {"live": False, "keys": ["enter", "escape"]}, True, [("said", "탭 눌러줘")]),
    ("a key the environment does have",
     [DESK], [says("엔터 누를게요", {"key": "Enter"})],
     {"live": False, "keys": ["enter", "escape"]}, True, [("said", "엔터 눌러줘")]),
    ("noticed, and permitted to act",
     [DESK], [says("빌드가 끝났어요", {"click": 1})], {"live": False, "keys": None}, True,
     [("perceived", "'Build succeeded' appeared")]),
    ("noticed, and NOT permitted — downgraded to speech",
     [DESK], [says("빌드가 끝났어요", {"click": 1})], {"live": False, "keys": None}, False,
     [("perceived", "'Build succeeded' appeared")]),
    ("a reply with no action at all",
     [DESK], [says("천만에요")], {"live": False, "keys": None}, True, [("said", "고마워")]),
    ("an index nobody has",
     [DESK], [says("그건 안 보여요", {"click": 99})], {"live": False, "keys": None}, True,
     [("said", "없는거 눌러줘")]),
    ("an effect kind the core does not know",
     [DESK], [says("스크롤할게요", {"scroll": 3})], {"live": False, "keys": None}, True,
     [("said", "스크롤해줘")]),
    ("three turns, so the history and the counters accumulate",
     [DESK], [says("닫을게요", {"click": 0}), says("천만에요"), says("공유할게요", {"click": 1})],
     {"live": False, "keys": None}, True,
     [("said", "닫아줘"), ("said", "고마워"), ("said", "공유해줘")]),
    ("an empty screen",
     [[]], [says("아무것도 안 보여요")], {"live": False, "keys": None}, True,
     [("said", "뭐가 보여?")]),
    #: **Both of these were added because a mutant survived without them**, which is the only
    #: honest reason to add a case after the fact. Raising `MOST_AT_ONCE` to 10 changed nothing
    #: while no reply asked for four things, and dropping the role from `find()` changed nothing
    #: while no screen held two things of the same name.
    ("four things at once, refused as a batch",
     [DESK], [says("여러 개 할게요", {"click": 0}, {"click": 1}, {"type": "x"}, {"key": "enter"})],
     {"live": False, "keys": None}, True, [("said", "네 개 해줘")]),
    ("the Font : trap, resolved at the seam",
     [FONT_TRAP, FONT_TRAP], [says("글꼴 바꿀게요", {"click": 1})],
     {"live": False, "keys": None}, True, [("said", "글꼴 바꿔줘")]),
    #: **The role is checked in TWO places in `find()`** — the candidate filter and the lookup of
    #: where the thing was last seen — and the second masks the first. Dropping the role from the
    #: filter alone changes nothing while the right one is still at distance zero. It shows only
    #: when the target has moved and the wrong-role twin is nearer to where the right one *was*.
    ("the role filter alone, exposed by a target that moved",
     [FONT_TRAP, FONT_MOVED], [says("글꼴 바꿀게요", {"click": 1})],
     {"live": False, "keys": None}, True, [("said", "글꼴 바꿔줘")]),
    #: And the other copy of the role: the lookup of where the thing *was*. It decides which of two
    #: same-role twins wins, so it shows only when the wrong-role sibling is what that lookup would
    #: land on instead. Three role-drop mutants, three cases; none of them covers another.
    ("the role in the 'where it was' lookup, which picks between twins",
     [TWIN_SEEN, TWIN_FRESH], [says("글꼴 바꿀게요", {"click": 1})],
     {"live": False, "keys": None}, True, [("said", "글꼴 바꿔줘")]),
]


def as_widget(x) -> dict:
    return {"name": x.name, "role": x.role, "rect": list(x.rect), "window": x.window}


def as_effect(e) -> dict:
    return {"kind": e.kind, "target": e.target or "", "role": e.role or "", "text": e.text or "",
            "unprompted": e.unprompted}


def as_done(d) -> dict:
    return {"effect": as_effect(d.effect), "confirmed": d.confirmed,
            "target": as_widget(d.target) if d.target is not None else None}


cases = []
for label, screens, replies, hands_spec, may_act, turns in SCENARIOS:
    hands = (DeclaringHands(*hands_spec["keys"], live=hands_spec["live"])
             if hands_spec["keys"] is not None else ScriptedHands(live=hands_spec["live"]))
    session = Session(eyes=ScriptedEyes(*screens), model=ScriptedModel(*replies), hands=hands,
                      may_act_unprompted=may_act)
    out = []
    for kind, text in turns:
        ex = session.think(Said(text) if kind == "said" else Perceived(text))
        out.append({
            "n": ex.n, "asked": ex.trigger.asked, "trigger": ex.trigger.text, "said": ex.said,
            "n_widgets": ex.n_widgets,
            "did": [as_done(d) for d in ex.did],
            "proposed": [as_effect(e) for e in ex.proposed],
            "refused": [as_effect(e) for e in ex.refused],
            "history": [list(p) for p in session.history],
            "blind": session.blind, "vanished": session.vanished,
            "unknown_keys": session.unknown_keys,
            "hands_done": list(hands.done), "eye_reads": session.eyes.reads,
        })
    cases.append({"label": label,
                  "screens": [[as_widget(x) for x in s] for s in screens],
                  "replies": list(replies), "hands": hands_spec, "may_act_unprompted": may_act,
                  "turns": [{"asked": k == "said", "text": t} for k, t in turns],
                  "want": out})

OUT.write_text(json.dumps(cases, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"  -> {OUT.name}")
print(f"    {len(cases)} scenarios, {sum(len(c['want']) for c in cases)} turns")
print(f"    counters reached: blind {max(t['blind'] for c in cases for t in c['want'])}, "
      f"vanished {max(t['vanished'] for c in cases for t in c['want'])}, "
      f"unknown_keys {max(t['unknown_keys'] for c in cases for t in c['want'])}")
