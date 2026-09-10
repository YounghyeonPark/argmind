# SPDX-License-Identifier: Apache-2.0
"""Can you correct it? Can it tell you what it did? Does it answer in your language?

    python -m measure.repair

**Everything measured on this project so far is about choosing**, and the owner's priority is
interaction: *"a few mistakes are fine, just let it keep correcting them — what matters is
interaction and communication"*. Nothing here has ever measured that, so this does, on three things
that are objective enough to count:

**Repair** — it picks the wrong thing, you say so, and you name the right one. Does the second turn
land? This is the one the priority turns on, and the suspicion going in is that it cannot: the
history records what you said and what it said, and **not what it offered or what it did**, so the
*that* in "not that one" refers to nothing.

**Memory of its own actions** — you ask what it just did. It has no record, so it can only invent.

**Language** — a Korean question answered in English is a communication defect, not a taste. Two
attempts to fix it in the prompt were withdrawn in September because they cost choosing accuracy;
that trade looks different now, and the model has changed since, so the rate is worth having again.

Fixtures, no clicking. Two or three turns each, and the second turn is the measurement.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from argmind.core import Said, Session                                    # noqa: E402
from argmind.core.llm import Local                                        # noqa: E402
from measure.capture import load                                          # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent


class Fixed:
    """Eyes that always show the same screen. Nothing is clicked, so nothing changes."""

    def __init__(self, widgets) -> None:
        self.widgets = list(widgets)

    def foreground(self):
        return list(self.widgets)

    def timed_foreground(self):
        return list(self.widgets), 0.001


class Nothing:
    live = False

    def click(self, widget, button="left"):
        return {}

    def type_text(self, text):
        return {}

    def press(self, key):
        return {}


#: `(screen, turns, what the last turn should end up choosing)`. `None` means it should choose
#: nothing. Written before running.
CASES = [
    ("REPAIR names the right thing", "untitled---paint",
     ["빨간색 골라줘", "아니 그거 말고 Red 를 눌러줘"], "Red"),
    ("REPAIR without naming it", "untitled---paint",
     ["동그라미 그리고 싶어", "아니야 그거 말고"], None),
    ("REPAIR after a vague ask", "character-map",
     ["뭔가 눌러줘", "글꼴 바꾸는 걸로 해줘"], "Font :"),
    ("MEMORY of what it offered", "character-map",
     ["복사 눌러줘", "방금 뭘 하려고 했어?"], None),
]

#: Korean in, Korean expected out. One turn each, on the smallest screen so the answer is short.
LANGUAGE = ["뭐가 열려 있어?", "이 창 닫아줘", "여기서 뭘 할 수 있어?", "고마워",
            "지금 화면 설명해줘", "복사 버튼 어디 있어?"]


def is_korean(text: str) -> bool:
    """Any Hangul at all counts. A reply that mixes is still speaking to the person."""
    return any("가" <= ch <= "힣" for ch in text)


def talk(model, widgets, lines):
    """Run a short conversation and return every turn's `(said, chosen)`."""
    session = Session(eyes=Fixed(widgets), model=model, hands=Nothing())
    out = []
    for line in lines:
        session.observe()
        intent = session.decide(Said(line))
        chosen = next((e.target for e in intent.effects if e.kind == "click"), None)
        session.history.append(("them", line))
        if intent.say:
            session.history.append(("you", intent.say))
        #: Nothing is performed here, so what the agent WOULD have done is what goes in — which is
        #: what the app records at the moment it puts a button on the screen.
        session.remember_doing(offered=intent.effects)
        out.append((intent.say, chosen))
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m measure.repair", description=__doc__)
    p.add_argument("--model", default="qwen3:8b")
    p.add_argument("--think", choices=("on", "off"), default="off")
    args = p.parse_args(argv)

    screens = {name: widgets for name, _window, widgets in load()}
    model = Local(args.model, think={"on": True, "off": False}[args.think])
    for _ in range(2):
        model.ask("Reply with the single word: ready")

    print(f"  {args.model} | thinking {args.think}\n")
    rows = []
    print("  REPAIR AND MEMORY")
    landed = 0
    for label, screen, lines, want in CASES:
        turns = talk(model, screens[screen], lines)
        final = turns[-1][1]
        ok = (final == want)
        landed += ok
        print(f"    {'OK ' if ok else 'NO '} {label:<28} wanted {str(want):<10} got {str(final)}")
        for line, (said, chosen) in zip(lines, turns):
            print(f"         you: {line}")
            print(f"         it : {said[:70]!r}" + (f"  [{chosen}]" if chosen else ""))
        rows.append({"case": label, "lines": lines, "want": want, "turns": turns, "ok": ok})

    print(f"\n    {landed} of {len(CASES)} landed\n")

    print("  LANGUAGE — Korean in, Korean out?")
    korean = 0
    small = screens["character-map"]
    for line in LANGUAGE:
        said, _ = talk(model, small, [line])[0]
        ok = is_korean(said)
        korean += ok
        print(f"    {'OK ' if ok else 'NO '} {line[:22]:<22} -> {said[:58]!r}")
        rows.append({"case": "language", "lines": [line], "said": said, "ok": ok})
    print(f"\n    {korean} of {len(LANGUAGE)} answered in Korean")

    out = HERE / f"repair-{args.model.replace(':', '-')}-{args.think}.json"
    out.write_text(json.dumps({"model": args.model, "think": args.think,
                               "repaired": landed, "of": len(CASES),
                               "korean": korean, "of_language": len(LANGUAGE),
                               "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n  every turn -> {out}")
    model.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
