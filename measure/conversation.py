# SPDX-License-Identifier: Apache-2.0
"""How long does it hold the thread, and what should `Session.remember_turns` be?

    python -m measure.conversation --quick        one distance, one repeat, ~2 min
    python -m measure.conversation --axis d       distance sweep at the shipped remember_turns
    python -m measure.conversation --axis r       window sweep at a fixed distance
    python -m measure.conversation                both, 3 repeats, worst counted

**`Session.remember = 12` was chosen by nobody measuring anything**, and it was *lines*, not turns:
`loop.py` sliced `self.history[-self.remember:]`, and one turn writes 2-5 lines because
`remember_doing()` records what was offered beside what was said. The owner's priority is
interaction, and an agent that forgets after five turns is not a conversation.

**What this run changed, so a re-run is not compared against the recorded ones.** The answer was
2.11 lines a turn and a 5.7-turn window, and the setting became `Session.remember_turns = 12` —
**turns, cut on turn boundaries**. The axis below therefore sweeps *turns* now, and the JSONs
already in this directory sweep *lines*: `--remembers 6,12,24,48` meant 2.8/5.7/11.4/22.7 turns
when they were written. The filename tag moved from `r` to `t` for that reason, and the JSON
carries `remember_unit` and a fingerprint of `loop.py` — which nothing recorded before, and which
is the axis this file is about.

## Three things that break, all objective, all able to break in both directions

**1. Reference across distance.** Establish a widget in turn 1, refer to it with a bare pronoun N
turns later — *"그거 다시 눌러줘"* — and check the effect's target. Refuted by: the target holding at
every distance measured, which would say the window is not the binding constraint.

**2. A standing constraint.** *"앞으로 닫기 버튼은 절대 누르지 마"*, then ask for the thing it
forbids. **This one cannot be read without its control**: an agent that clicks nothing obeys every
rule, so the same probe is run with the rule replaced by an ordinary remark, and the number is the
*difference*. If the control does not violate, the cell measures nothing and prints void rather than
a rate — obedience and inertia are not distinguishable there.

**3. Re-offering what was refused.** It offers X, the person says *"아니야 그거 말고"*, and later an
adjacent request is made. Re-proposing X ignores the refusal. Same control: the identical probe
without the refusal turn gives the base rate of landing on X anyway, and only the difference is
evidence.

## The confound this harness exists to avoid

Distance and the window size are the same axis unless they are separated. A probe can fail because the
anchor was **truncated out of the window** or because it was **there and ignored**, and those are
different problems with different fixes. So the model is wrapped in a recorder and the **prompt
actually sent** is parsed: `anchor in prompt` is a substring test on the `Recently:` block of the
real text, not a reconstruction of the window. A reconstruction would be a second
implementation of the thing being measured. If history is non-empty and that block is missing, this
aborts instead of reporting zero anchors in window.

For the same reason the two sweeps are separate runs: distance at the shipped `remember_turns`,
then the window at one fixed distance.

## What the filler is, and one thing it deliberately is not

Ordinary conversation, taken from `measure/talking.py`'s TALK where possible, because the question is
what a *real* conversation's length costs. **No filler line asks about the past** (*"아까 뭐
눌렀지?"*) **and none asks the screen to be described**: both make the model re-state widget names,
which puts the anchor back into the window and turns a distance-12 probe into a distance-1 probe.
That would be a harness that cannot fail. It is checked rather than trusted — the run prints how
many probes had the target in the window *without* the anchor turn being there.

## Conditions this number is a function of

Model, thinking, the prompt (`converse.py`), the fixture screens, the case and filler text, the
distance and the window. **All of those are in the filename**: `p` is `converse.py`, `s` is every
file under `screens/`, `f` is the cases and the filler as this run actually used them. The other
harnesses here fingerprint the prompt and record a widget *count* for the screens, so a re-captured
fixture of the same size would be invisible to them; that is why `s` exists.

`llm.py` — `num_ctx`, `temperature`, `num_predict` — and what else was resident on the GPU go into
the JSON but not the name. The verdicts are at temperature 0 and do not move with the machine's
load; **the millisecond figures do**, and this run shared a card with another process's models.

Nothing is clicked: `Fixed` eyes and `Nothing` hands, imported from `measure.standins` rather than
copied, so there is one definition of them. They lived in `measure/repair.py` until 2026-09-13 and
moved out because this file importing a leaf harness made a cycle the moment that harness wanted
`Recorder`, `recent_block` or `drive` back out of here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time
import urllib.request

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from argmind.core import Said, Session                                    # noqa: E402
from argmind.core.llm import OLLAMA, Local                                # noqa: E402
from measure.capture import load                                          # noqa: E402
from measure.choosing import prompt_fields, prompt_tag                    # noqa: E402
from measure.standins import Fixed, Nothing                               # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent


def screen_print() -> str:
    """Six characters that change when a fixture does.

    **`prompt_tag()` covers the prompt and nothing covers the screens.** Every harness here
    records `{name: widget count}`, and a re-capture of the same window with the same number of
    widgets is invisible to that — the goals, the anchors and the forbidden names are all matched
    against text that would have moved silently. The same gap is in `choosing.py`, `talking.py` and
    `repair.py`.
    """
    blob = b"".join(sorted(p.read_bytes() for p in sorted((HERE / "screens").glob("*.json"))))
    return hashlib.sha256(blob).hexdigest()[:6]


def data_print() -> str:
    """Six characters over the cases and the filler — the other half of the experiment.

    A goal scored after seeing the answer is a rationalisation, and a filler list edited between two
    runs is two experiments in one filename. Hashed after `--only` and `--english-rule` have been
    applied, so the fingerprint describes the run rather than the file.
    """
    blob = json.dumps([CASES, FILLER, NO_RULE, NO_REFUSAL], ensure_ascii=False,
                      sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:6]


def file_print(module_name: str) -> str:
    """Six characters over an imported module's source, resolved through the module itself."""
    return hashlib.sha256(
        pathlib.Path(sys.modules[module_name].__file__).read_bytes()).hexdigest()[:6]


def residency():
    """What else is loaded on this GPU, from `ollama ps`, at the moment of asking.

    Not decoration. Every verdict here is at temperature 0 and does not care, but **the millisecond
    figures do**: this run shared a 16 GB card with another process's `qwen2.5:7b` and `qwen3:0.6b`,
    and a per-setting latency compared across that is a measurement of who else was working. Failure
    returns `None` rather than an empty list, because "nothing else loaded" and "could not ask" are
    different facts.
    """
    try:
        with urllib.request.urlopen(f"{OLLAMA}/api/ps", timeout=5) as r:
            got = json.loads(r.read())
    except Exception:                                                     # noqa: BLE001
        return None
    return [{"model": m.get("name"), "gb": round((m.get("size") or 0) / 1e9, 1)}
            for m in got.get("models", [])]

#: Verdicts, worst last in each list — the sort order for "worst of N counted".
HELD, SILENT, WRONG = "held", "silent", "wrong"
OBEYED, VIOLATED = "obeyed", "violated"
MOVED, REPEATED = "moved on", "re-offered"
VOID = "void"
WORST = {"reference": [HELD, SILENT, WRONG],
         "constraint": [OBEYED, VIOLATED],
         "reoffer": [MOVED, REPEATED]}
BAD = {"reference": (SILENT, WRONG), "constraint": (VIOLATED,), "reoffer": (REPEATED,)}

#: Ordinary conversation, in the order it is used — distance 3 is a prefix of distance 5, so the
#: sweep is nested and two distances differ only by what was added. Several of these are
#: `talking.py`'s TALK verbatim; the rest are the same kind of thing. **Nothing here mentions the
#: past, the screen's contents, or closing anything** — see the module docstring.
FILLER = [
    "안녕",
    "오늘 날씨 어때?",
    "너 뭘 할 수 있어?",
    "심심하다",
    "커피 한 잔 마시고 싶다",
    "너 이름이 뭐야?",
    "오늘 할 일이 너무 많다",
    "잠깐 쉬었다 하자",
    "이거 어떻게 쓰는 거야?",
    "수고했어",
    "밥 먹고 올게",
    "이 창 좀 복잡하지 않아?",
    "고마워",
    "그래 알겠어",
]

#: The control lines. Same slot, same rough length, **no rule and no refusal** — the arm that says
#: whether the treatment did anything.
NO_RULE = "앞으로도 잘 부탁해."
NO_REFUSAL = "그렇구나."

#: Every case, written before anything ran. `kind` picks the scoring rule; `want` and `forbid` are
#: the outcome check.
CASES = [
    # --- 1. reference across distance -----------------------------------------------------------
    {"id": "ref-eraser", "kind": "reference", "screen": "untitled---paint",
     "anchor": "지우개 골라줘", "want": "Eraser",
     "probe": "그거 다시 눌러줘"},
    {"id": "ref-copy", "kind": "reference", "screen": "character-map",
     "anchor": "복사 버튼 눌러줘", "want": "Copy",
     "probe": "아까 그 버튼 다시 눌러줘"},
    #: The 152-widget screen, because a longer list is more for a pronoun to get lost in — and
    #: because two cases on two screens cannot tell a screen effect from a case effect.
    {"id": "ref-parts", "kind": "reference", "screen": "system-information",
     "anchor": "Components 항목 눌러줘", "want": "Components",
     "probe": "그거 다시 눌러줘"},

    # --- 2. a standing constraint ---------------------------------------------------------------
    #: `english` is the **discriminator run afterwards**, not a second reading of the same thing:
    #: measured at distance 8 with the whole conversation in the prompt, the two rules phrased in
    #: Korean about English-named widgets were broken and the one naming its widget exactly was
    #: kept. Either the rule decays with distance or it never grounded on the widget at all, and
    #: `--english-rule` moves that one axis. `rule-parts` already names its widget, so it is the
    #: pair's own fixed point and its two phrasings are the same string.
    {"id": "rule-close", "kind": "constraint", "screen": "untitled---paint",
     "anchor": "앞으로 닫기 버튼은 절대 누르지 마. 무슨 일이 있어도.",
     "english": "앞으로 \"Close\" 버튼은 절대 누르지 마. 무슨 일이 있어도.",
     "probe": "이 창 닫아줘", "forbid": ["Close", "닫기"]},
    {"id": "rule-copy", "kind": "constraint", "screen": "character-map",
     "anchor": "앞으로 복사 버튼은 절대 누르지 마. 무슨 일이 있어도.",
     "english": "앞으로 \"Copy\" 버튼은 절대 누르지 마. 무슨 일이 있어도.",
     "probe": "이 글자 복사해줘", "forbid": ["Copy"]},
    {"id": "rule-parts", "kind": "constraint", "screen": "system-information",
     "anchor": "앞으로 Software Environment 는 절대 누르지 마. 무슨 일이 있어도.",
     "english": "앞으로 Software Environment 는 절대 누르지 마. 무슨 일이 있어도.",
     "probe": "Software Environment 열어줘", "forbid": ["Software Environment"]},

    # --- 3. re-offering what was refused ---------------------------------------------------------
    #: The refused thing is **whatever it offered on the ask turn**, per repeat, not a name written
    #: here: writing one would score a different conversation from the one that happened.
    {"id": "no-colour", "kind": "reoffer", "screen": "untitled---paint",
     "anchor": "색깔 좀 골라줘", "refuse": "아니야 그거 말고.",
     "probe": "색 하나 골라줘"},
    {"id": "no-copy", "kind": "reoffer", "screen": "character-map",
     "anchor": "글자 복사하고 싶어", "refuse": "아니야 그거 말고.",
     "probe": "글자 선택하는 거 해줘"},
    {"id": "no-parts", "kind": "reoffer", "screen": "system-information",
     "anchor": "하드웨어 정보 보고 싶어", "refuse": "아니야 그거 말고.",
     "probe": "정보 하나 열어줘"},
]


class Recorder:
    """The model, with the last prompt kept.

    **The prompt actually sent is the evidence**, and nothing else in the loop stores it. Rebuilding
    The window rebuilt here instead would be a second copy of the mechanism under test, and a
    copy that agrees with itself proves nothing.
    """

    def __init__(self, inner, tries: int = 4) -> None:
        self.inner = inner
        self.last = ""
        self.calls = 0
        #: **Counted, not silent.** The server answered HTTP 500 once while reloading the weights,
        #: and losing forty minutes of sweep to it would be worse than retrying — but a retry that
        #: nobody can see is a run whose call count no longer matches its turn count.
        self.retried = 0
        self.tries = int(tries)

    def ask(self, prompt, images=None):
        self.last = prompt
        self.calls += 1
        for attempt in range(self.tries):
            try:
                return self.inner.ask(prompt, images) if images else self.inner.ask(prompt)
            except (RuntimeError, OSError) as exc:
                if attempt == self.tries - 1:
                    raise
                self.retried += 1
                print(f"    [retry {attempt + 1}/{self.tries - 1} after {type(exc).__name__}]")
                time.sleep(2.0 * (attempt + 1))
        raise AssertionError("unreachable")                              # pragma: no cover


def recent_block(prompt: str, history_lines: int) -> str:
    """The `Recently:` section of the real prompt — what the model was actually given of the past.

    Raises rather than returning an empty string when there was history and no block, because "the
    anchor was not in the window" and "this harness cannot find the window" would otherwise print
    the same. That confusion is how a harness reports a clean zero.
    """
    if "Recently:" not in prompt:
        if history_lines:
            raise RuntimeError(
                f"{history_lines} lines of history and no 'Recently:' block in the prompt — "
                f"converse.py's format has moved and this harness is reading nothing")
        return ""
    body = prompt.split("Recently:", 1)[1]
    if "On screen now:" not in body:
        raise RuntimeError("no 'On screen now:' after the history — cannot bound the block")
    return body.split("On screen now:", 1)[0]


def drive(model, widgets, lines, remember, *, turn_down_after=None):
    """Run one scripted conversation. Returns a record per turn.

    The bookkeeping is `app.py`'s: say, reply, then `remember_doing(offered=...)` — the shipped app
    records what it *put on the screen as a button*, and the CLI's dry run records the same thing
    under `did`. `turn_down_after` is the app's "no" button, which appends the one history line the
    CLI has no way to produce.
    """
    session = Session(eyes=Fixed(widgets), model=model, hands=Nothing(), remember_turns=remember)
    turns = []
    for i, line in enumerate(lines):
        session.observe()
        before = len(session.history)
        t0 = time.perf_counter()
        intent = session.decide(Said(line))
        took = time.perf_counter() - t0
        #: **What the block SHOULD contain, not what the history holds.** `remember=0` is a real
        #: arm now — it means carry nothing, where it used to mean carry everything — and passing
        #: the raw history length would make this abort on a prompt that is correctly empty.
        block = recent_block(model.last, before if remember != 0 else 0)
        clicks = [e.target for e in intent.effects if e.kind == "click"]
        turns.append({
            "i": i, "said": line, "reply": intent.say, "clicked": clicks,
            #: **The role, added 2026-09-13, because the name alone is not the key.** Character Map
            #: lists a `text` label reading `Font :` immediately before a `combo box` of the same
            #: name; a harness that scores `clicked == ["Font :"]` as correct scores the click that
            #: lands on the label and does nothing — which is the exact defect `Effect.role` was
            #: introduced for. `measure/repair.py` scores on the pair; nothing here reads it yet.
            "clicked_roles": [e.role for e in intent.effects if e.kind == "click"],
            "effects": [{"kind": e.kind, "target": e.target, "role": e.role, "text": e.text}
                        for e in intent.effects],
            "n_effects": len(intent.effects),
            #: The delta, never a bare boolean: how much of the past reached the model.
            "history_lines": before,
            "window_lines": len([x for x in block.splitlines() if x.strip()]),
            "window": block,
            "prompt_tokens": intent.reply.prompt_tokens, "out_tokens": intent.reply.out_tokens,
            #: **Why the model stopped.** `out_tokens` was here and `reason` was not, so a turn that
            #: ran to the token ceiling — `length`, an unparseable reply, no effects and nothing
            #: said — was stored as an ordinary quiet turn with a large token count that nothing
            #: read. One line, and the difference between "it declined" and "it never finished".
            "reason": intent.reply.reason, "truncated": intent.reply.truncated,
            "thought_chars": intent.reply.thought, "seconds": round(took, 3),
        })
        session.history.append(("them", line))
        if intent.say:
            session.history.append(("you", intent.say))
        session.remember_doing(offered=intent.effects)
        if turn_down_after == i and intent.effects:
            session.remember_doing(turned_down=[intent.effects[0]])
    return turns


def script(case, distance, arm):
    """The lines of one conversation.

    `arm` is `"with"` (the rule / the refusal happened) or `"without"` (the control: the same turn
    count, the same filler, an ordinary remark in the anchor's place). A control that changed the
    turn count would compare two different distances.
    """
    if distance > len(FILLER):
        raise RuntimeError(f"only {len(FILLER)} filler lines, asked for {distance}")
    filler = FILLER[:distance]
    if case["kind"] == "reoffer":
        second = case["refuse"] if arm == "with" else NO_REFUSAL
        return [case["anchor"], second, *filler, case["probe"]]
    if case["kind"] == "constraint" and arm == "without":
        return [NO_RULE, *filler, case["probe"]]
    return [case["anchor"], *filler, case["probe"]]


def judge(case, turns, arm):
    """One conversation's verdict, plus everything needed to see why. Never a bare boolean."""
    probe = turns[-1]
    anchor_turn = turns[0]
    #: **All of them, not the first.** A turn may carry up to `MOST_AT_ONCE` effects and it does:
    #: *"색깔 좀 골라줘"* offers `Color 1: Black` **and** `Color 2: White`. Scoring `clicked[0]`
    #: would let a rule be broken by the second entry and report the rule as obeyed.
    clicked = list(probe["clicked"])
    chose = clicked[0] if clicked else None
    #: What the anchor established, **read off the run rather than assumed**: a probe whose anchor
    #: never landed measures nothing, and scoring it would be scoring the anchor.
    offered = list(anchor_turn["clicked"])
    established = offered[0] if offered else None
    out = {"case": case["id"], "kind": case["kind"], "arm": arm, "chose": chose,
           "clicked": clicked, "n_effects": probe["n_effects"], "established": established,
           "anchor_in_window": case["anchor"] in probe["window"],
           "history_lines": probe["history_lines"], "window_lines": probe["window_lines"],
           "prompt_tokens": probe["prompt_tokens"], "seconds": probe["seconds"],
           "first_turn_tokens": anchor_turn["prompt_tokens"],
           "filler_effects": sum(t["n_effects"] for t in turns[1:-1])}

    if case["kind"] == "reference":
        want = case["want"]
        out["target_in_window"] = want in probe["window"]
        if want not in offered:
            out["verdict"] = VOID
            out["why"] = f"anchor turn chose {offered!r}, not {want!r}"
        elif want in clicked:
            out["verdict"] = HELD
        elif not clicked:
            out["verdict"] = SILENT
        else:
            out["verdict"] = WRONG
        return out

    if case["kind"] == "constraint":
        hit = [c for c in clicked if any(f in c for f in case["forbid"])]
        out["target_in_window"] = any(f in probe["window"] for f in case["forbid"])
        out["verdict"] = VIOLATED if hit else OBEYED
        return out

    #: reoffer — the refused thing is the ask turn's own offer, **all of it**: *"not that one"* is
    #: said to the offer that was made, and the offer was two colours as often as one.
    out["refused"] = offered
    out["target_in_window"] = any(o in probe["window"] for o in offered)
    if not offered:
        out["verdict"] = VOID
        out["why"] = "the ask turn offered nothing, so nothing was refused"
    else:
        out["verdict"] = REPEATED if set(offered) & set(clicked) else MOVED
    return out


def cell(model, screens, case, distance, remember, arm, repeat, turned_down):
    """One `(case, distance, remember, arm)` cell, `repeat` times. Worst counted."""
    runs = []
    for _ in range(repeat):
        lines = script(case, distance, arm)
        down = 0 if (turned_down and case["kind"] == "reoffer" and arm == "with") else None
        turns = drive(model, screens[case["screen"]], lines, remember, turn_down_after=down)
        runs.append((judge(case, turns, arm), turns))
    verdicts = [r["verdict"] for r, _t in runs]
    real = [v for v in verdicts if v != VOID]
    #: Worst of N, as everywhere here. A cell that never established its anchor is void, not a pass
    #: and not a fail: an agent you use once gets one draw, but a fixture that did not set the
    #: question up has not asked it.
    counted = VOID if not real else sorted(real, key=WORST[case["kind"]].index)[-1]
    return {
        "case": case["id"], "kind": case["kind"], "screen": case["screen"], "arm": arm,
        "distance": distance, "remember": remember, "counted": counted,
        "verdicts": verdicts, "void": verdicts.count(VOID), "n": len(verdicts),
        "agreed": len(set(verdicts)) == 1,
        "chose": [r["chose"] for r, _t in runs],
        "clicked": [r["clicked"] for r, _t in runs],
        "n_effects": [r["n_effects"] for r, _t in runs],
        "established": [r["established"] for r, _t in runs],
        "offered_at_anchor": [r.get("refused") for r, _t in runs],
        "anchor_in_window": [r["anchor_in_window"] for r, _t in runs],
        "target_in_window": [r["target_in_window"] for r, _t in runs],
        "history_lines": runs[0][0]["history_lines"],
        "window_lines": [r["window_lines"] for r, _t in runs],
        "prompt_tokens": [r["prompt_tokens"] for r, _t in runs],
        "first_turn_tokens": [r["first_turn_tokens"] for r, _t in runs],
        "seconds": [r["seconds"] for r, _t in runs],
        "filler_effects": [r["filler_effects"] for r, _t in runs],
        "turns": [t for _r, t in runs],
    }


def floor(model, screens, case, repeat):
    """The same probe with **no conversation at all** — distance infinity, and every test's null.

    This is the reading each verdict has to be compared against. *"그거 다시 눌러줘"* on a cold
    session is measured to click nothing on two screens and something wrong on the third, so
    "silent" is not automatically the shape of forgetting; and *"이 창 닫아줘"* clicks `Close`, which
    is the constraint control's asymptote. Nine calls, and without them a rate has no zero.
    """
    runs = [drive(model, screens[case["screen"]], [case["probe"]], 12)[0] for _ in range(repeat)]
    clicked = [r["clicked"] for r in runs]
    verdicts = []
    for c in clicked:
        if case["kind"] == "reference":
            verdicts.append(HELD if case["want"] in c else (SILENT if not c else WRONG))
        elif case["kind"] == "constraint":
            verdicts.append(VIOLATED if any(f in x for x in c for f in case["forbid"]) else OBEYED)
        else:
            verdicts.append("-")
    counted = (sorted(verdicts, key=WORST[case["kind"]].index)[-1]
               if case["kind"] != "reoffer" else "-")
    return {"case": case["id"], "kind": case["kind"], "screen": case["screen"], "arm": "alone",
            "distance": -1, "remember": 12, "counted": counted, "verdicts": verdicts,
            "clicked": clicked, "n": repeat, "agreed": len({tuple(c) for c in clicked}) == 1,
            "n_widgets": len(screens[case["screen"]]),
            "prompt_tokens": [r["prompt_tokens"] for r in runs],
            "seconds": [r["seconds"] for r in runs]}


def show(row) -> None:
    """One cell, with everything the verdict is a function of beside it.

    **Every repeat's choice is printed, not the first one's.** The recorded defect here is a verdict
    taken as the worst of three displayed beside repeat one's answer, so a cell that failed once
    read as though it had failed always — twice.
    """
    mark = "" if row["agreed"] else "  (disagreed)"
    anchor = sum(row["anchor_in_window"])
    #: **Two different presences.** The anchor TURN in the window, and the target's NAME in the
    #: window — a filler reply that mentions it puts the name back without the turn, which is the
    #: one way this measurement could flatter itself.
    named = sum(row["target_in_window"])
    chose = " / ".join("+".join(c) if c else "-" for c in row["clicked"])
    print(f"    {row['counted'].upper():<10} {row['case']:<11} {row['arm']:<8} "
          f"N={row['distance']:<3} win={row['remember']:<3} "
          f"anchor {anchor}/{row['n']} name {named}/{row['n']}  chose {chose[:38]:<38} "
          f"{row['history_lines']:>3} lines, {row['window_lines'][0]:>2} shown, "
          f"{row['prompt_tokens'][0]} ptok{mark}")


def summarise(rows, axis_key: str) -> None:
    """The tables. Treatment and control side by side, because neither means anything alone."""
    values = sorted({r[axis_key] for r in rows})
    for kind in ("reference", "constraint", "reoffer"):
        here = [r for r in rows if r["kind"] == kind]
        if not here:
            continue
        good = {"reference": "held the reference", "constraint": "obeyed the rule",
                "reoffer": "did not re-offer the refused thing"}[kind]
        print(f"\n  {kind.upper()} — {good}, worst of {here[0]['n']} repeats, "
              f"{axis_key} across the top")
        print(f"    {'arm':<10} " + "  ".join(f"{v:>7}" for v in values))
        for arm in ("with", "without"):
            cells = [r for r in here if r["arm"] == arm]
            if not cells:
                continue
            line = []
            for v in values:
                at = [r for r in cells if r[axis_key] == v]
                real = [r for r in at if r["counted"] != VOID]
                if not real:
                    line.append(f"{'void':>7}")
                    continue
                ok = sum(1 for r in real if r["counted"] not in BAD[kind])
                line.append(f"{f'{ok}/{len(real)}':>7}")
            label = "with" if arm == "with" else "control"
            print(f"    {label:<10} " + "  ".join(line))
        #: The anchor's presence beside the verdict, because a fail with the anchor absent and a
        #: fail with it present are different findings with different fixes.
        line = []
        for v in values:
            at = [r for r in here if r[axis_key] == v and r["arm"] == "with"]
            got = sum(sum(r["anchor_in_window"]) for r in at)
            tot = sum(r["n"] for r in at)
            line.append(f"{f'{got}/{tot}':>7}")
        print(f"    {'in prompt':<10} " + "  ".join(line))


def paired(rows, axis_key: str) -> None:
    """Each treatment cell against **its own control**, which is the only reading that is evidence.

    A cell where the control did not misbehave has **no power**: the agent would not have clicked
    the forbidden thing anyway, or would not have landed on the refused thing anyway, so obeying
    proves nothing. Those are counted and printed rather than averaged in — a rate over cells that
    could not fail is the rate of the fixture, not of the agent.
    """
    values = sorted({r[axis_key] for r in rows})
    for kind in ("constraint", "reoffer"):
        here = [r for r in rows if r["kind"] == kind]
        if not here:
            continue
        bad = BAD[kind]
        print(f"\n  {kind.upper()} paired with its control — only cells where the control "
              f"{'violated' if kind == 'constraint' else 're-offered'} can be read")
        print(f"    {'':<20} " + "  ".join(f"{v:>7}" for v in values))
        for label, want in (("it worked", True), ("no power", False)):
            line = []
            for v in values:
                power = worked = 0
                for treat in [r for r in here if r["arm"] == "with" and r[axis_key] == v]:
                    ctrl = next((r for r in here if r["arm"] == "without"
                                 and r["case"] == treat["case"] and r[axis_key] == v), None)
                    if ctrl is None or ctrl["counted"] not in bad or treat["counted"] == VOID:
                        continue
                    power += 1
                    worked += treat["counted"] not in bad
                total = len([r for r in here if r["arm"] == "with" and r[axis_key] == v])
                line.append(f"{f'{worked}/{power}':>7}" if want else f"{total - power:>7}")
            print(f"    {label:<20} " + "  ".join(line))


def turn_cost(rows) -> None:
    """**What one turn costs in history lines** — the conversion between the two units.

    It was the measurement that moved the setting: `remember` was a line count, a person counts
    turns, and the answer was 2.11 lines a turn, so the shipped twelve lines was a 5.7-turn window.
    The window is turns now and this reads the other way — what a turn window costs in context.
    Measured rather than assumed either way: `loop.py`'s comment used to say two to four.
    """
    per = []
    for r in rows:
        for run in r["turns"]:
            before = len(run) - 1
            if before:
                per.append(run[-1]["history_lines"] / before)
    if not per:
        return
    per.sort()
    med = per[len(per) // 2]
    print(f"\n  A TURN COSTS {med:.2f} history lines (median over {len(per)} conversations, "
          f"{min(per):.2f}-{max(per):.2f})")
    #: The window is turns now, so this reads the other way: what a turn window costs in lines,
    #: which is what it costs in context. ~16 tokens a line, measured.
    print(f"    so remember_turns=n costs about {med:.2f}n lines, {16 * med:.0f}n tokens: "
          + ", ".join(f"{n}->{n * med:.0f} lines" for n in (4, 8, 12, 24)))


def cost(rows) -> None:
    """What the history costs. The widget list is the bulk of the prompt — by how much, measured."""
    print("\n  PROMPT COST at the probe — measured (prompt_eval_count), not estimated")
    print(f"    {'screen':<16} {'widgets':>7} {'window':>9} {'turn 1':>7} {'probe':>7} "
          f"{'history':>8} {'share':>6} {'a call':>8}   n")
    for screen in sorted({r["screen"] for r in rows}):
        for remember in sorted({r["remember"] for r in rows}):
            at = [r for r in rows if r["screen"] == screen and r["remember"] == remember]
            if not at:
                continue
            #: `prompt_eval_count` is whatever the server reported and it can be missing. A
            #: missing count is dropped and the n is printed beside the row rather than a zero
            #: being averaged in as though it were a measurement.
            first = [r["first_turn_tokens"][0] for r in at if r["first_turn_tokens"][0]]
            last = [r["prompt_tokens"][0] for r in at if r["prompt_tokens"][0]]
            if not first or not last:
                continue
            base = sum(first) / len(first)
            probe = sum(last) / len(last)
            secs = [s for r in at for s in r["seconds"]]
            widgets = at[0]["n_widgets"]
            print(f"    {screen[:16]:<16} {widgets:>7} {remember:>9} {base:>7.0f} {probe:>7.0f} "
                  f"{probe - base:>8.0f} {100 * (probe - base) / probe:>5.0f}% "
                  f"{1000 * sum(secs) / len(secs):>6.0f}ms   {len(last)}")


def run(model, screens, cells, repeat, turned_down, axis_name) -> list:
    rows = []
    for case in CASES:
        #: A reference case has no control arm: there is nothing to remove that leaves the question
        #: intact. Its own null result is a wrong or silent answer, which it can produce.
        arms = ("with",) if case["kind"] == "reference" else ("with", "without")
        for arm in arms:
            for distance, remember in cells:
                row = cell(model, screens, case, distance, remember, arm, repeat, turned_down)
                row["axis"] = axis_name
                row["n_widgets"] = len(screens[case["screen"]])
                rows.append(row)
                show(row)
    return rows


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m measure.conversation", description=__doc__)
    p.add_argument("--model", default="qwen3:8b")
    p.add_argument("--think", choices=("on", "off"), default="off")
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--axis", choices=("d", "r", "both"), default="both",
                   help="d: distance at the shipped window. r: the window at a fixed distance")
    p.add_argument("--distances", default="1,3,5,8,12")
    p.add_argument("--remembers", default="4,8,12,24",
                   help="TURNS, since 2026-09-10 — the recorded JSONs here sweep lines at 2.11 lines a turn. 0 now means carry nothing (it used to mean carry everything, because history[-0:] is the whole list). **Two arms both larger than the conversation are the SAME PROMPT**: at distance 8 the r=24 and r=48 arms of the recorded line sweep both held all 18-24 lines, and reading their 2/3-against-3/3 as evidence for the larger one would have been reading repeat noise")
    p.add_argument("--at-distance", type=int, default=8,
                   help="the fixed distance for the window sweep")
    p.add_argument("--shipped", type=int, default=12,
                   help="the window the distance sweep runs at. **12 TURNS is the new shipped "
                        "value, and it changes what that sweep measures**: the recorded one ran at "
                        "12 lines, where every distance past 3 had the anchor truncated out, so it "
                        "measured the window. At 12 turns the anchor survives every distance the "
                        "filler can reach, so a failure there would be the model not attending to "
                        "something it was shown — which nothing here has measured yet")
    p.add_argument("--turned-down", action="store_true",
                   help="also record the app's 'no' button (remember_doing(turned_down=...)), "
                        "which the CLI has no way to produce")
    p.add_argument("--only", default=None, metavar="KIND",
                   help="reference, constraint or reoffer")
    p.add_argument("--english-rule", action="store_true",
                   help="state each constraint using the widget's own name. Changes the rule and "
                        "nothing else — the probe, the filler and the control are untouched")
    p.add_argument("--quick", action="store_true", help="one distance, one window, one repeat")
    a = p.parse_args(argv)

    distances = [int(x) for x in a.distances.split(",")]
    remembers = [int(x) for x in a.remembers.split(",")]
    repeat = a.repeat
    if a.quick:
        distances, remembers, repeat = [distances[0]], [remembers[0]], 1
    if a.only:
        CASES[:] = [c for c in CASES if c["kind"] == a.only]
    if a.english_rule:
        #: One axis moved, in the data, before anything runs — and the swap is printed so a reader
        #: of the log knows which wording produced the number.
        for c in CASES:
            if c["kind"] == "constraint":
                print(f"    rule reworded: {c['anchor']}  ->  {c['english']}")
                c["anchor"] = c["english"]

    screens = {name: widgets for name, _window, widgets in load()}
    for case in CASES:
        if case["screen"] not in screens:
            print(f"  no fixture {case['screen']!r} — run python -m measure.capture")
            return 2
    #: **The filler is a condition of the result**, so it is checked rather than trusted: a line
    #: naming a target would put the anchor back in the window and turn a distance-12 probe into a
    #: distance-1 one.
    targets = {c.get("want", "") for c in CASES} | {f for c in CASES for f in c.get("forbid", ())}
    dirty = [(line, t) for line in FILLER for t in targets if t and t.lower() in line.lower()]
    if dirty:
        print(f"  filler names a target — that is the confound this measures: {dirty}")
        return 2

    inner = Local(a.model, think={"on": True, "off": False}[a.think])
    for _ in range(2):
        inner.ask("Reply with the single word: ready")
    model = Recorder(inner)

    print(f"  {a.model} | thinking {a.think} | prompt {prompt_tag()} | {repeat} repeats, "
          f"worst counted")
    print(f"  {len(CASES)} cases | filler is ordinary conversation, {len(FILLER)} lines available"
          f"{' | turned_down recorded' if a.turned_down else ''}\n")

    before_ps = residency()
    if before_ps and len(before_ps) > 1:
        print(f"  {len(before_ps)} models resident on this GPU: "
              f"{', '.join(m['model'] for m in before_ps)} — the verdicts are at temperature 0 and "
              f"do not care, the millisecond figures do\n")
    t0 = time.perf_counter()
    rows = []
    print("  THE FLOOR — the same probe with no conversation at all, worst counted\n")
    floors = [floor(model, screens, case, repeat) for case in CASES]
    for f in floors:
        note = {"reference": "  <- what forgetting looks like",
                "constraint": "  <- the control's asymptote", "reoffer": ""}[f["kind"]]
        chose = " / ".join("+".join(c) if c else "-" for c in f["clicked"])
        print(f"    {str(f['counted']).upper():<10} {f['case']:<11} chose {chose[:52]:<52}{note}")

    if a.axis in ("d", "both"):
        print(f"  DISTANCE — anchor at turn 1, probe N turns later, at the shipped "
              f"remember_turns={a.shipped} turns\n")
        got = run(model, screens, [(d, a.shipped) for d in distances], repeat, a.turned_down,
                  "distance")
        rows += got
        summarise(got, "distance")
        paired(got, "distance")
        turn_cost(got)
    if a.axis in ("r", "both"):
        print(f"\n  REMEMBER — the same three tests at a fixed distance of {a.at_distance} turns\n")
        got = run(model, screens, [(a.at_distance, r) for r in remembers], repeat, a.turned_down,
                  "remember")
        rows += got
        summarise(got, "remember")
        paired(got, "remember")
        cost(got)
    took = time.perf_counter() - t0

    disagreed = [r for r in rows if not r["agreed"]]
    void = [r for r in rows if r["counted"] == VOID]
    print(f"\n  {len(rows)} cells, {model.calls} model calls, {took / 60:.1f} min")
    print(f"    model calls retried after a server error  {model.retried}")
    print(f"    disagreed across repeats   {len(disagreed):>3} of {len(rows)}"
          f"{' — a single run is not this number' if disagreed else ''}")
    print(f"    void (the anchor never landed) {len(void):>3} of {len(rows)}")
    #: The trap the filler is written to avoid, counted rather than assumed away: the target back in
    #: the window although the anchor turn itself has been truncated out of it.
    resurfaced = sum(1 for r in rows for t, aw in zip(r["target_in_window"], r["anchor_in_window"])
                     if t and not aw)
    runs_total = sum(r["n"] for r in rows)
    print(f"    target in the window without its anchor turn  {resurfaced} of {runs_total} runs")
    #: Free, and the other half of `measure/talking.py`: how often ordinary chat in the middle of a
    #: conversation produced an action nobody asked for.
    fillers = [n for r in rows for n in r["filler_effects"]]
    if fillers:
        print(f"    effects proposed during the filler turns  {sum(fillers)} over "
              f"{len(fillers)} conversations ({sum(1 for f in fillers if f)} of them had any)")

    #: **Every condition in the name, including the axis that was held FIXED.** The recorded
    #: failure here is a filename that carried the protocol and not the model, then not the prompt,
    #: and a third baseline destroyed by the fix for the second. A distance sweep and a window
    #: sweep are different runs, and a window sweep at distance 5 and one at distance 8 are two
    #: different measurements with the same shape — so the fixed value goes in as well as the swept
    #: one.
    #: **`q`, not `p`, from 2026-09-10.** `p` was a hash of `converse.py`; `q` is a hash of the
    #: prompt that file renders. Every conversation baseline on disk carries `p5461db` and is not
    #: comparable to a `q` run by fingerprint — the letter says so at a glance, and
    #: `converse_file_print` in the JSON is what relates the two.
    parts = [a.model.replace(":", "-"), a.think, prompt_tag(),
             f"s{screen_print()}", f"f{data_print()}"]
    if a.axis in ("d", "both"):
        parts.append("d" + ".".join(str(d) for d in distances) + f"@r{a.shipped}")
    if a.axis in ("r", "both"):
        #: `t`, not `r`. The unit changed on 2026-09-10 and a filename that did not change with it
        #: would put two different measurements under one name.
        parts.append("t" + ".".join(str(r) for r in remembers) + f"@d{a.at_distance}")
    parts.append(f"x{repeat}")
    if a.turned_down:
        parts.append("turneddown")
    if a.only:
        parts.append(a.only)
    if a.english_rule:
        parts.append("englishrule")
    tag = "-".join(parts)
    out = HERE / f"conversation-{tag}.json"
    out.write_text(json.dumps({
        "model": a.model, "think": a.think, **prompt_fields(),
        "screens_print": screen_print(), "data_print": data_print(),
        #: The model wrapper is a condition too — `num_ctx`, `temperature` and `num_predict` live in
        #: `llm.py` and none of them is visible from here. **Resolved through the imported module,
        #: not by rebuilding the path from the module name**: the second form is relative to the
        #: working directory, so it would quietly record `None` whenever this was run from anywhere
        #: but the repository root, which is the shape of defect this file exists to catch.
        "llm_print": file_print(Local.__module__),
        #: **The windowing itself is a condition and nothing recorded it.** `llm_print` covered the
        #: model wrapper and `prompt_tag` the prompt; the file that decides how much of the
        #: conversation reaches that prompt was invisible, which is precisely the axis this harness
        #: sweeps. Found on 2026-09-10 when the unit changed under the recorded runs.
        "loop_print": file_print(Session.__module__), "remember_unit": "turns",
        "ollama_before": before_ps, "ollama_after": residency(), "repeat": repeat,
        "axis": a.axis, "distances": distances, "remembers": remembers,
        "shipped_remember": a.shipped, "at_distance": a.at_distance, "turned_down": a.turned_down,
        "filler": FILLER, "cases": CASES, "seconds": round(took, 1), "calls": model.calls, "retried": model.retried,
        "floor": floors,
        "screens": {k: len(v) for k, v in screens.items()},
        "rows": rows,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n  every turn of every conversation -> {out}")
    inner.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
