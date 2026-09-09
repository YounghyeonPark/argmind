# SPDX-License-Identifier: Apache-2.0
"""Step 4: does it pick the right thing? The number that decides whether the rebuild is worth doing.

    python -m measure.choosing              every goal, three times each
    python -m measure.choosing --repeat 1   one pass, for a quick look at a prompt change

**Everything else here is measured and this is not.** The environment answers in 41-266 ms; the
model answers in 67-85 ms; a name survives a moving screen. None of that matters if the thing it
picks is the wrong thing, and the only evidence so far is casual: two live attempts, one right and
one wrong — *"open the file menu"* selected the document.

## Four outcomes, not two

                        it clicked something        it clicked nothing
    a right answer      RIGHT or WRONG              MISSED
    no right answer     WRONG — the bad one         DECLINED — correct

`WRONG` on the bottom row is the outcome that matters most and the one a two-outcome score hides. A
model pushed to always act will always act, and on a screen that does not contain what was asked for
it clicks whatever is nearest. **Nine of the thirty-seven goals have no right answer** for exactly
that reason: without them the measurement cannot produce a negative result, and an agent that clicks
something every time would score the same as one that chooses well.

## Three runs, not one

Temperature is 0, so this ought to be deterministic; *ought to* is not a measurement, and this
project has withdrawn two conclusions that were single draws. The repeat count exists to report
whether the answers agree with themselves, not to average them into looking better.

## What this deliberately does not measure

Perception and actuation. The screens are fixtures captured from real windows, so nothing here can
fail because a window was not in front or a click missed — which is the point: a bad number then
points at choosing, and only at choosing.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import random
import sys
import zlib
import time

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from argmind.core import Said, converse                                  # noqa: E402
from argmind.core.converse import SYSTEM                                 # noqa: E402
from argmind.core.llm import Local                                       # noqa: E402
from measure.capture import load                                         # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
RIGHT, WRONG, MISSED, DECLINED = "right", "wrong", "missed", "declined"


def by_name(model, widgets, goal):
    """The same question with the answer given as an exact NAME instead of a list index.

    **Why this variant exists.** Six of the seven wrong answers in the first run sat 1-10 places from
    a correct one and five within four — *"창 닫아"* chose index 121 where `Close` is 122. That is
    the shape of arithmetic going slightly wrong, not of a model that misunderstood the request, and
    it is the same failure the vision model had: it could see the answer and could not compute the
    number for it.

    The index was chosen because a number can be range-checked, so the model cannot invent a target.
    **A name can be checked exactly as strictly** — it must appear in the list, character for
    character — so nothing is given up by asking for one.
    """
    listed = list(widgets)
    #: **The role goes BEFORE the quoted name and the instruction says so twice**, because the first
    #: version of this prompt read "the exact name of one thing above" and the model answered
    #: `button "Copy"` — the whole line, quotes and role included, which is a literal reading of what
    #: it was told. Every one of five answers was right and every one was scored zero. That is the
    #: harness lying again, and the fix is the prompt rather than a lenient match on the answer:
    #: matching leniently would let a genuine invention through as well.
    lines = [SYSTEM, "", "On screen now — one per line, role first, then the name in quotes:"]
    lines += [f'{w.role}: "{w.name}"' for w in listed]
    lines += ["", f"They said: {goal}", "",
              'Reply with only JSON: {"say": "<what you say back>", "click": "<a name>"}.',
              'For "click", give ONLY what is inside the quotes on one line above — not the role '
              'word, not the quotes. So for the line   button: "Copy"   you would write '
              '"click": "Copy".',
              'Use null for "click" if nothing listed above does this.']
    reply = model.ask(chr(10).join(lines))
    got = reply.json() or {}
    want = got.get("click")
    if not isinstance(want, str) or not want:
        return None, None
    #: Exact match only. A name the list does not contain is an invention, and the right answer to
    #: an invention is to drop it — the same rule the index path applies to an out-of-range number.
    #:
    #: **The rejected string is handed back rather than swallowed**, because "it declined" and "it
    #: named something that is not there" are different failures and the first version of this
    #: reported both as nothing. One of them is the model being careful and the other is this
    #: protocol not working.
    return (want if any(w.name == want for w in listed) else None), want


def outcome(chosen, accept) -> str:
    """One goal's result. `chosen` is the name it clicked, or `None`."""
    if accept is None:
        return DECLINED if chosen is None else WRONG
    if chosen is None:
        return MISSED
    return RIGHT if chosen in accept else WRONG


def run(model_name: str, repeat: int, only: str | None, *,
        named: bool = False, shuffle: int | None = None, drop=()) -> int:
    screens = {name: widgets for name, _window, widgets in load()}
    if drop:
        #: **Leaving something out of the list is a change to what the model can see**, not a
        #: cleanup. A `text` widget cannot usefully be clicked — one of the recorded wrong answers
        #: clicked `\\Device\\HarddiskVolume1`, and a `text` label named `Font :` sitting in front
        #: of a combo box of the same name caused a defect that took a whole round of work to find.
        #: But the same widgets are what the model reads when it is asked *what is on screen*, so
        #: this has to be measured rather than assumed to be free.
        before = sum(len(v) for v in screens.values())
        screens = {k: [w for w in v if w.role not in drop] for k, v in screens.items()}
        after = sum(len(v) for v in screens.values())
        print(f"  dropped {before - after} of {before} widgets with role in {drop}")
    goals = json.loads((HERE / "goals.json").read_text(encoding="utf-8"))["goals"]
    if only:
        goals = [g for g in goals if only in g["screen"]]
    if not goals or not screens:
        print("  nothing to run — capture some screens first (python -m measure.capture)")
        return 2

    model = Local(model_name)
    rows, tally = [], collections.Counter()
    how = ("by NAME" if named else "by number") + (
        "" if shuffle is None else f", shuffled {shuffle}") + (
        "" if not drop else f", without {'/'.join(drop)}")
    print(f"  {model_name} | {len(goals)} goals x {repeat} | {how} | "
          f"{len(screens)} screens, {sum(len(w) for w in screens.values())} widgets\n")
    print(f"  {'outcome':<9} {'screen':<14} {'goal':<28} chose")
    t_start = time.perf_counter()
    for g in goals:
        widgets = screens.get(g["screen"])
        if widgets is None:
            print(f"  {'NO SCREEN':<9} {g['screen'][:14]:<14} {g['goal'][:28]:<28}")
            continue
        if shuffle is not None:
            #: **The discriminator.** Index proximity and MEANING proximity are confounded on a real
            #: toolbar — related controls sit next to each other — so a small gap does not by itself
            #: say which one caused the error. Shuffling breaks the tie: if the cause is arithmetic
            #: the gaps stay small in the new order, and if it is meaning the same wrong items get
            #: chosen wherever they are.
            widgets = list(widgets)
            #: **crc32, not hash().** Python randomises `hash()` of a string per process, so the
            #: first version of this produced an order that could not be reproduced afterwards —
            #: and the analysis of it silently scored against an order that was never shown to the
            #: model. A measurement whose conditions cannot be reconstructed is not a measurement.
            random.Random(shuffle * 1000003 + zlib.crc32(g["goal"].encode())).shuffle(widgets)
        seen, rejected = [], []
        for _ in range(repeat):
            if named:
                got, raw = by_name(model, widgets, g["goal"])
                seen.append(got)
                rejected.append(None if got is not None else raw)
                continue
            intent = converse(model, widgets, Said(g["goal"]))
            clicks = [e.target for e in intent.effects if e.kind == "click"]
            seen.append(clicks[0] if clicks else None)
        results = [outcome(c, g["accept"]) for c in seen]
        #: Positions are recorded rather than reconstructed later, for the same reason: the order
        #: the model was shown is a condition of the result and belongs with it.
        names = [x.name for x in widgets]
        at = [names.index(c) if c in names else None for c in seen]
        right_at = [i for i, n in enumerate(names) if g["accept"] and n in g["accept"]]
        #: The WORST of the repeats is what gets counted. Taking the best would be scoring the
        #: model's luckiest draw, and an agent you use once gets one draw.
        worst = sorted(results, key=[RIGHT, DECLINED, MISSED, WRONG].index)[-1]
        tally[worst] += 1
        agreed = len(set(seen)) == 1
        rows.append({"screen": g["screen"], "goal": g["goal"], "accept": g["accept"],
                     "chose": seen, "results": results, "counted": worst, "agreed": agreed,
                     "chose_at": at, "right_at": right_at, "n_listed": len(names),
                     "rejected": rejected})
        mark = "" if agreed else "  (disagreed across runs)"
        print(f"  {worst.upper():<9} {g['screen'][:14]:<14} {g['goal'][:28]:<28} "
              f"{str(seen[0])[:34]}{mark}")

    took = time.perf_counter() - t_start
    answerable = [r for r in rows if r["accept"] is not None]
    unanswerable = [r for r in rows if r["accept"] is None]
    print(f"\n  {len(rows)} goals in {took:.0f} s ({took / max(1, len(rows) * repeat):.2f} s a call)")
    print(f"    right      {tally[RIGHT]:>3} / {len(answerable)} answerable")
    print(f"    wrong      {tally[WRONG]:>3}   ({sum(1 for r in unanswerable if r['counted'] == WRONG)}"
          f" of them on a screen with no right answer — the bad one)")
    print(f"    missed     {tally[MISSED]:>3}   (a right answer existed and it did nothing)")
    print(f"    declined   {tally[DECLINED]:>3} / {len(unanswerable)} unanswerable")
    disagreed = [r for r in rows if not r["agreed"]]
    print(f"    disagreed  {len(disagreed):>3}   across {repeat} runs at temperature 0"
          f"{' — so a single run is not this number' if disagreed else ''}")

    tag = (("name" if named else "index") + ("" if shuffle is None else f"-shuffled{shuffle}")
           + ("" if not drop else "-without-" + "-".join(drop)))
    out = HERE / f"choosing-{tag}.json"
    out.write_text(json.dumps({
        "model": model_name, "repeat": repeat, "seconds": round(took, 1),
        "how": how, "named": named, "shuffle": shuffle,
        "screens": {k: len(v) for k, v in screens.items()},
        "tally": dict(tally), "rows": rows,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n  every answer -> {out}")
    model.release()
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m measure.choosing", description=__doc__)
    p.add_argument("--model", default="qwen2.5:7b")
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--only", default=None, metavar="SCREEN", help="substring of a screen name")
    p.add_argument("--by-name", action="store_true",
                   help="ask for the exact name instead of a list index")
    p.add_argument("--shuffle", type=int, default=None, metavar="SEED",
                   help="shuffle each screen's order — separates an index error from a meaning one")
    p.add_argument("--drop", default=None, metavar="ROLES",
                   help="comma-separated roles to leave out of the list, e.g. 'text'. Order is "
                        "preserved, because shuffling was measured to cost accuracy")
    a = p.parse_args(argv)
    return run(a.model, a.repeat, a.only, named=a.by_name, shuffle=a.shuffle,
               drop=tuple(r.strip() for r in a.drop.split(",")) if a.drop else ())


if __name__ == "__main__":
    sys.exit(main())
