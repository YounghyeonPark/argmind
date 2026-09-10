# SPDX-License-Identifier: Apache-2.0
"""When the person is only talking, does it keep its hands off?

    python -m measure.talking

**Every turn goes through the widget list.** Say *"고마워"* and the model is handed twenty-one
widgets and a schema for clicking them. That is the shape `converse()` has always had — one call
that asks what to say and what to do together, because a round trip is the budget — and the
suspicion is that framing every utterance as a picking task makes it pick.

There is one known instance: asked *"방금 뭘 하려고 했어?"* it answers correctly **and clicks Copy a
second time**. One instance is an anecdote. This is the rate.

**It has to be able to fail in both directions.** A model that never acts would score perfectly here
and be useless, so this is only half of a pair: the other half is `measure/choosing.py`, where 28 of
the 37 goals *want* an action. Read them together or neither means anything.

Nothing is clicked. Fixtures, and the biggest screen as well as the smallest, because a longer list
is more to be tempted by.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from argmind.core import Said, converse                                   # noqa: E402
from argmind.core.llm import Local                                        # noqa: E402
from measure.capture import load                                          # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent

#: Things a person says that ask for **no action at all**. Written before running. Grouped, because
#: "it acts on small talk" and "it acts on a question about the screen" are different problems.
TALK = [
    ("greeting", "안녕"),
    ("greeting", "고마워"),
    ("greeting", "수고했어"),
    ("greeting", "잘했어"),
    ("about the screen", "뭐가 열려 있어?"),
    ("about the screen", "지금 화면 설명해줘"),
    ("about the screen", "여기 버튼이 몇 개야?"),
    ("about the screen", "복사 버튼이 어디 있어?"),
    ("about the past", "방금 뭘 했어?"),
    ("about the past", "아까 뭐 눌렀지?"),
    ("about itself", "너 뭘 할 수 있어?"),
    ("about itself", "너 이름이 뭐야?"),
    ("about itself", "클릭도 할 수 있어?"),
    ("chit-chat", "오늘 날씨 어때?"),
    ("chit-chat", "심심하다"),
    ("opinion", "이 창 좀 복잡하지 않아?"),
    ("opinion", "이거 어떻게 쓰는 거야?"),
]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m measure.talking", description=__doc__)
    p.add_argument("--model", default="qwen3:8b")
    p.add_argument("--think", choices=("on", "off"), default="off")
    p.add_argument("--repeat", type=int, default=2)
    args = p.parse_args(argv)

    screens = {name: widgets for name, _window, widgets in load()}
    #: The smallest and the largest of the captured windows, so list length is a visible axis rather
    #: than an average.
    pick = sorted(screens.items(), key=lambda kv: len(kv[1]))
    where = [pick[0], pick[-1]]

    model = Local(args.model, think={"on": True, "off": False}[args.think])
    for _ in range(2):
        model.ask("Reply with the single word: ready")

    print(f"  {args.model} | thinking {args.think} | {len(TALK)} things to say x {args.repeat}, "
          f"on {where[0][0]} ({len(where[0][1])} widgets) and {where[1][0]} "
          f"({len(where[1][1])} widgets)\n")

    rows, tally = [], collections.Counter()
    for name, widgets in where:
        print(f"  {name} — {len(widgets)} widgets")
        for kind, line in TALK:
            acted = []
            for _ in range(args.repeat):
                intent = converse(model, widgets, Said(line))
                acted.append([f"{e.kind}:{e.target or e.text}" for e in intent.effects])
            worst = max(acted, key=len)
            #: The worst of the repeats, as everywhere else here: an agent you use once gets one
            #: draw, and taking the best would score its luckiest.
            quiet = not worst
            tally[(kind, quiet)] += 1
            if not quiet:
                print(f"    ACTED  {line[:20]:<20} {worst}   said {intent.say[:40]!r}")
            rows.append({"screen": name, "kind": kind, "said": line,
                         "acted": acted, "quiet": quiet})
        n = sum(v for (_k, _q), v in tally.items())
        print()

    print("  by kind, over both screens:")
    kinds = sorted({k for k, _ in TALK}, key=[k for k, _ in TALK].index)
    total_quiet = total = 0
    for kind in kinds:
        q = tally[(kind, True)]
        n = q + tally[(kind, False)]
        total_quiet += q
        total += n
        print(f"    {kind:<18} kept its hands off {q} of {n}")
    print(f"\n  {total_quiet} of {total} conversational turns produced no action "
          f"({100 * total_quiet / total:.0f}%)")
    print("  the other half of this pair is measure/choosing.py, where 28 of 37 goals want one")

    #: **Model, thinking AND prompt in the name.** This harness had the defect that had already
    #: destroyed two baselines in `choosing.py`, and it destroyed a third: a run under a changed
    #: reply shape overwrote the run it was meant to be compared against.
    from measure.choosing import prompt_print                            # noqa: PLC0415

    out = (HERE / f"talking-{args.model.replace(':', '-')}-{args.think}"
                  f"-p{prompt_print()}.json")
    out.write_text(json.dumps({"model": args.model, "think": args.think,
                               "prompt": prompt_print(),
                               "quiet": total_quiet, "of": total, "rows": rows},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n  every turn -> {out}")
    model.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
