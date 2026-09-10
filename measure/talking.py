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

**No turn here carries a conversation** — every one goes out through `sending.turn(...,
history=NO_HISTORY)`, so nothing this file reports is a function of what was said earlier. That is
printed beside the headline and stored in the JSON rather than left here, because this exact number
was once used to support "all three harnesses same or better, no slide" about a change that took
rule-following from 36 of 36 probes to 14 of 36. `measure/sending.py` carries the whole account.
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

from argmind.core import Said                                            # noqa: E402
from argmind.core.llm import Local                                        # noqa: E402
from measure.capture import load                                          # noqa: E402
#: Imported at the top now, not inside `main()`. The lazy import dated from when only the filename
#: needed it; four of these fields belong in the JSON of every harness here and one of them is the
#: prompt itself, which is too important to reach for at the last moment.
from measure.choosing import (prompt_fields, prompt_tag, print_spend,     # noqa: E402
                             spend, spend_summary)
#: **Every turn in this file goes out with no conversation, and that is now said out loud.**
#: `measure/sending.py` carries why: the same three call sites once let a prompt change that took
#: rule-following from 36 of 36 to 14 of 36 be reported as "no slide" because all three of these
#: harnesses reproduced their baselines under it.
from measure import sending                                              # noqa: E402

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

    rows, tally, spends = [], collections.Counter(), []
    for name, widgets in where:
        print(f"  {name} — {len(widgets)} widgets")
        for kind, line in TALK:
            acted, costs = [], []
            for _ in range(args.repeat):
                intent = sending.turn(model, widgets, Said(line),
                                      history=sending.NO_HISTORY)
                acted.append([f"{e.kind}:{e.target or e.text}" for e in intent.effects])
                #: **What the call cost.** `quiet` is this harness's whole verdict, and a turn that
                #: ran to the token ceiling is quiet: `done_reason: length` means `Reply.json()`
                #: returned `None`, so there were no effects to find and nothing was said either.
                #: A perfect score built out of empty replies would read exactly like a good one.
                costs.append(spend(intent.reply))
            spends += [(f"{name} / {line}", c) for c in costs]
            worst = max(acted, key=len)
            #: The worst of the repeats, as everywhere else here: an agent you use once gets one
            #: draw, and taking the best would score its luckiest.
            quiet = not worst
            tally[(kind, quiet)] += 1
            if not quiet:
                print(f"    ACTED  {line[:20]:<20} {worst}   said {intent.say[:40]!r}")
            rows.append({"screen": name, "kind": kind, "said": line,
                         "acted": acted, "quiet": quiet, "spend": costs})
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
    #: **The caveat is ON the headline, not under it.** This is the line that gets copied into a
    #: commit message, and the sentence it was twice used to support — "all three harnesses same or
    #: better, no slide" — is exactly the sentence it cannot support on its own.
    print(f"\n  {total_quiet} of {total} conversational turns produced no action "
          f"({100 * total_quiet / total:.0f}%){sending.caveat(prompt_tag())}")
    print("  the other half of this pair is measure/choosing.py, where 28 of 37 goals want one")

    cost = spend_summary(spends, ceiling=model.max_tokens)
    print_spend(cost)
    #: Printed **and** returned from one computation, so the block on screen and the record in the
    #: file cannot disagree — a verdict printed beside a number taken from somewhere else has
    #: shipped twice in this directory.
    blind = sending.note(prompt_tag(), truncated=cost["truncated"], of=cost["calls"])

    #: **Model, thinking AND prompt in the name.** This harness had the defect that had already
    #: destroyed two baselines in `choosing.py`, and it destroyed a third: a run under a changed
    #: reply shape overwrote the run it was meant to be compared against.
    #:
    #: The prompt segment reads `q<hex>` from 2026-09-10. Files here named `p<hex>` were fingerprinted
    #: under the previous scheme — a hash of `converse.py`, which moved when a comment did and never
    #: contained the reply-shape line — and are **not** comparable to a `q` file by that value.
    #: `converse_file_print` inside the JSON is the only bridge between the two.
    #: **`h` — the harness itself, added 2026-09-11**, for the reason `choosing.py` gives at the
    #: same place: `saying.py` recorded which code produced its number and this file did not. It is
    #: also what keeps every result written before today under its own name, so the change that
    #: routed this harness through `sending.turn()` cannot overwrite a baseline it did not produce.
    out = (HERE / f"talking-{args.model.replace(':', '-')}-{args.think}"
                  f"-{prompt_tag()}-h{sending.harness_print(__file__)}.json")
    out.write_text(json.dumps({"model": args.model, "think": args.think,
                               **prompt_fields(), **blind,
                               "harness_print": sending.harness_print(__file__),
                               "sending_print": sending.harness_print(sending.__file__),
                               "max_tokens": model.max_tokens, "spend": cost,
                               "quiet": total_quiet, "of": total, "rows": rows},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n  every turn -> {out}")
    model.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
