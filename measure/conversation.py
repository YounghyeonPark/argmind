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

Model, thinking, the prompt (`q`, what `converse.py` renders), the fixture screens (`s`, every file
under `screens/`), the case and filler text (`f`, as this run used them), **this harness (`h`)**,
**`argmind/core/loop.py` (`l`)**, the distance and the window. All of those are in the filename. The
other harnesses here fingerprint the prompt and record a widget *count* for the screens, so a
re-captured fixture of the same size would be invisible to them; that is why `s` exists.

**`h` and `l` arrived on 2026-09-13 and the two conditions they cover are the two that move without
anybody typing anything.** Until then this file recorded `loop_print`, `llm_print`, `s` and `f` —
everything except itself — and put neither in the name, while being the only member of
`sending.COVERS_HISTORY`: the single file three other harnesses point at when they print *"the axis
was run at this prompt"*. Two runs of two rewritten trees wrote one path and the second overwrote a
committed baseline; it came back out of git 33 minutes later. `l` is newline-normalised, the same
quantity `reoffer.py` writes as `l` (see `sending.norm_print` for why the JSON keys disagree).

**The name is still not proof of difference, so `main()` refuses to overwrite.** `standins.py`,
`capture.py` and `choosing.py` are recorded in the JSON and not in the name; a run that differs only
in one of those produces the same path, and the refusal is what catches it. `LEGACY` below is the
one file here that predates both letters, why it was left alone, and what a reader comparing across
that boundary is looking at.

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
#: For `harness_print` and `norm_print` — one definition of each, there. This file
#: sends no turn through `sending.turn()`: every call here goes through `Session`,
#: which is the point of the harness.
from measure import sending                                                # noqa: E402

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


def harness_print() -> str:
    """Six characters over **this file**. `h` in the filename, `harness_print` in the JSON.

    It was the one condition this harness did not record. `s` said the fixtures moved, `f` the
    cases, `q` the prompt, `loop_print` the window — and the code that drives the conversation and
    turns a reply into a verdict was invisible, in the file three other harnesses defer to by name.
    Two arms of one comparison wrote the same path, and the second overwrote a committed baseline;
    it came back out of git 33 minutes later.

    **What it covers and what it does not.** This file, byte for byte, including its prose. Not
    `standins.py`, not `capture.py`, not `choosing.py`'s `prompt_fields` — those are recorded in the
    JSON beside it and are *not* in the name, so editing one of them still lets two runs collide by
    filename. That residual is why `main()` refuses to overwrite an existing result rather than
    trusting the name to be complete.

    One definition, in `sending.py`, so this and `reoffer.py` and `choosing.py` cannot drift into
    hashing three slightly different things under one letter.
    """
    from measure import sending                                          # noqa: PLC0415

    return sending.harness_print(__file__)


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

#: **The one result in this directory that predates `h` and `l`, and what can honestly be said about
#: it.** Data rather than prose, because `main()` prints it whenever that file is still on disk and
#: `--rescore` is the check behind the third claim.
#:
#: It was **left where it is** rather than renamed or re-taken, and the reasons are worth the four
#: lines: `argmind/core/llm.py` cites it by name (a file `measure/` may not edit), and three
#: committed results — `choosing-…-hc162c9`, `saying-…-h35874b`, `talking-…-h35d38f` — record that
#: name inside their own `history` block. Renaming it would make four records name a file that does
#: not exist, to gain a label this constant states exactly. Re-taking it would cost 45 minutes and
#: would **not** produce the same experiment: `loop.py` changed on 2026-09-13 (`3ef5769`, the offer
#: tense), so a run today measures a different history writer, and the two would differ in two
#: things at once.
LEGACY = {
    "file": "conversation-qwen3-8b-off-qa62805-s5020f7-f041650-d1.3.5.8.12@r12-x3.json",
    "taken": "2026-09-11 12:28",
    "produced_by": "measure/conversation.py at commit 7e1d0e5 (git blob 4f0ebd7), the state of this "
                   "file from 2026-09-10 19:08 until dd62ee5 on 2026-09-13 12:05",
    "harness_print": "NOT RECORDED and not reconstructible byte for byte — the working tree is "
                     "CRLF and nothing stored what the file hashed to on the machine that ran it. "
                     "A hash computed from the git blob today would be a different quantity wearing "
                     "the same letter, so none is written down",
    "differs_from_today": "prose; Fixed/Nothing imported from measure.standins instead of "
                          "measure.repair (the class bodies moved unchanged); and drive() now "
                          "records clicked_roles and effects per turn. Nothing on the path from a "
                          "line to a prompt, and nothing in judge()",
    "checked_by": "python -m measure.conversation --rescore FILE — today's judge() re-run over the "
                  "stored transcripts of that file. 75 cells, 225 repeats: every verdict reproduced",
    "loop_print": "d1720c — the PRE-3ef5769 loop.py, where an offer was written in the past tense. "
                  "The shipped tree is d8d312/lbc9cc0. This file is therefore not a baseline for "
                  "the shipped agent, and measure/sending.py's deferral says so to every harness "
                  "that points at it",
    "across_the_boundary": "a run today is named …-x3-h<six>-l<six> and this one is named …-x3. "
                           "The two are NOT a before/after of anything: the harness recording and "
                           "loop.py's history writing both moved between them",
}

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


def result_path(a, distances, remembers, repeat, loop_norm) -> pathlib.Path:
    """Where this run will write, resolved **before the model is loaded**.

    **Every condition in the name, including the axis that was held FIXED.** The recorded failure
    here is a filename that carried the protocol and not the model, then not the prompt, and a third
    baseline destroyed by the fix for the second. A distance sweep and a window sweep are different
    runs, and a window sweep at distance 5 and one at distance 8 are two different measurements with
    the same shape — so the fixed value goes in as well as the swept one.

    **`q`, not `p`, from 2026-09-10.** `p` was a hash of `converse.py`; `q` is a hash of the prompt
    that file renders. Every conversation baseline on disk before that date carries `p5461db` and is
    not comparable to a `q` run by fingerprint — the letter says so at a glance, and
    `converse_file_print` in the JSON is what relates the two.

    **`h` and `l` from 2026-09-13, and they close a live accident rather than a hypothetical one.**
    `h` is this file; `l` is `argmind/core/loop.py`, newline-normalised, the same quantity
    `reoffer.py` names `l`. Without them a rewritten harness over a rewritten history writer
    produced *the same path* as the baseline it was meant to be compared against, and overwrote it.
    Every other condition in the name is a choice somebody typed; these two move under a `git pull`.

    It is a pure function of the arguments and the files on disk, which is what lets `main()` refuse
    45 minutes earlier than the write.
    """
    parts = [a.model.replace(":", "-"), a.think, prompt_tag(),
             f"s{screen_print()}", f"f{data_print()}", f"h{harness_print()}", f"l{loop_norm}"]
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
    return HERE / ("conversation-" + "-".join(parts) + ".json")


def refuse(out: pathlib.Path) -> None:
    """Say why this run will not write, in the only words that stop somebody deleting the file.

    **The name is never trusted to be complete.** `h` and `l` closed the two conditions that move on
    their own; `standins.py`, `capture.py` and `choosing.py`'s `prompt_fields` are recorded in the
    JSON and are *not* in the name, so two runs can still collide. The one time this happened, two
    arms wrote one path and the second overwrote a committed baseline — 33 minutes to get it back
    out of git, and the fifth instance of that shape in this directory.
    """
    print(f"\n  ** REFUSING TO WRITE: {out.name} already exists. **")
    print("     Every condition this harness puts in a name is identical to a run already on disk.")
    print("     Either this is that run again, or something not in the name moved: standins.py,")
    print("     capture.py, choosing.py. Check the JSON's prints before assuming which.")
    print("     Move or rename the existing file and re-run. Do not delete it —")
    print("     measure/sending.py's deferral points three other harnesses at whichever file "
          "is here.")


def rescore(path) -> int:
    """Re-judge a recorded run's stored transcripts with **today's** `judge()`. Calls no model.

    The question it answers is the migration question: *did the scoring change under a result that
    is still being quoted?* A recorded row carries every turn of every repeat, so the verdicts can
    be recomputed exactly — and a harness edit that only added fields to `drive()` must reproduce
    all of them. If it does not, the file's numbers describe code that no longer exists and the
    disagreements are printed one by one rather than summarised into a rate.

    **It cannot check everything and says which half it checks.** Scoring, yes, byte for byte.
    What was *sent* — the lines, the window, the prompt — no: that would need the model again. So a
    pass here says the verdicts in the file are the verdicts today's rules give to the conversations
    that happened, and says nothing about whether those conversations would happen again.
    """
    got = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    cases = {c["id"]: c for c in got["cases"]}
    print(f"\n  RESCORING {pathlib.Path(path).name}")
    print(f"    recorded at prompt {got.get('prompt_scheme', 'p')[:1]}{got.get('prompt')}, "
          f"loop.py {got.get('loop_print')}, harness "
          f"{got.get('harness_print') or 'NOT RECORDED — this is the file the migration is about'}")
    print(f"    today's harness h{harness_print()}, judge() as it stands now\n")
    rows = got["rows"]
    #: The floor rows are scored by `floor()` and not by `judge()`; they are counted as skipped
    #: rather than silently passed, because "not checked" and "checked and agreed" are the two
    #: things this whole directory exists to keep apart.
    #: **Two tallies, not one.** A per-repeat verdict and a per-row worst-of-N are different
    #: quantities and adding them gives a denominator that matches neither — 226 out of 225
    #: repeats, which is the shape of a display disagreeing with the thing beside it.
    same = moved = missing = rows_same = rows_moved = 0
    for row in rows:
        case = cases.get(row["case"])
        if case is None or "turns" not in row:
            missing += 1
            continue
        for i, turns in enumerate(row["turns"]):
            was = row["verdicts"][i]
            now = judge(case, turns, row["arm"])["verdict"]
            if was == now:
                same += 1
                continue
            moved += 1
            print(f"    MOVED  {row['case']:<11} {row['arm']:<8} N={row['distance']:<3} "
                  f"win={row['remember']:<3} repeat {i + 1}: {was} -> {now}")
        real = [v for v in row["verdicts"] if v != VOID]
        counted = VOID if not real else sorted(real, key=WORST[row["kind"]].index)[-1]
        if counted == row["counted"]:
            rows_same += 1
        else:
            rows_moved += 1
            print(f"    MOVED  {row['case']:<11} {row['arm']:<8} worst-of-{row['n']} "
                  f"{row['counted']} -> {counted}")
    print(f"    {same} of {same + moved} stored per-repeat verdicts reproduce ({moved} moved)")
    print(f"    {rows_same} of {rows_same + rows_moved} stored worst-of-N cells reproduce "
          f"({rows_moved} moved); {missing} rows carried no transcript to re-score")
    print(f"    {len(rows)} rows, {len(got.get('floor', []))} floor rows NOT re-scored "
          f"(floor() scores those, not judge())")
    if moved or rows_moved:
        print("\n  ** the scoring has moved under this file: its numbers are not what today's "
              "rules give. **")
    return 1 if moved or rows_moved or missing else 0


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
    p.add_argument("--rescore", default=None, metavar="FILE",
                   help="re-judge a recorded run's stored transcripts with today's judge(). No "
                        "model, no fixtures. The check behind LEGACY")
    a = p.parse_args(argv)
    if a.rescore:
        return rescore(a.rescore)

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

    #: Off the module object, never off a path rebuilt from the name — the same lesson as
    #: `file_print`. `sending.norm_print` is the one definition; `reoffer.py` uses it too.
    loop_norm = sending.norm_print(sys.modules[Session.__module__].__file__)

    #: **Resolved and refused BEFORE the model is loaded.** Every part of the name is a hash of a
    #: file or a flag off the command line, so the path is knowable with no calls made — and the
    #: run this exists to prevent costs 45 minutes before it reaches the write. Checked again at
    #: the write, because two of these can be in flight at once.
    out = result_path(a, distances, remembers, repeat, loop_norm)
    if out.exists():
        refuse(out)
        return 3

    inner = Local(a.model, think={"on": True, "off": False}[a.think])
    for _ in range(2):
        inner.ask("Reply with the single word: ready")
    model = Recorder(inner)

    print(f"  {a.model} | thinking {a.think} | prompt {prompt_tag()} | {repeat} repeats, "
          f"worst counted")
    print(f"  harness h{harness_print()} | loop.py l{loop_norm} "
          f"(raw {file_print(Session.__module__)}) | screens s{screen_print()} | "
          f"cases f{data_print()}")
    print(f"  {len(CASES)} cases | filler is ordinary conversation, {len(FILLER)} lines available"
          f"{' | turned_down recorded' if a.turned_down else ''}")
    #: **Printed while the file is still there, not remembered.** The one result in this directory
    #: that carries neither `h` nor `l`; three other harnesses defer to it by name.
    if (HERE / LEGACY["file"]).exists():
        print(f"\n  THE ONE RESULT HERE THAT PREDATES h AND l — still on disk, deliberately")
        print(f"    {LEGACY['file']}")
        for key in ("produced_by", "harness_print", "differs_from_today", "checked_by",
                    "loop_print", "across_the_boundary"):
            print(f"    {key:<20} {LEGACY[key]}")
    print()

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

    #: The name was built by `result_path()` before any call was made — see there for what is in
    #: it and why. Re-checked at the write. `result_path()` already refused before the model was loaded;
    #: this is the second of the two, for two runs started a minute apart.
    if out.exists():
        refuse(out)
        inner.release()
        return 3
    out.write_text(json.dumps({
        "model": a.model, "think": a.think, **prompt_fields(),
        "screens_print": screen_print(), "data_print": data_print(),
        #: **This file, which recorded everything except itself.** See `harness_print`.
        "harness_print": harness_print(),
        #: The measure-side modules `h` does not cover. In the JSON and not in the name: they are
        #: the residual the refusal above exists for.
        "standins_print": file_print("measure.standins"),
        "capture_print": file_print("measure.capture"),
        "choosing_print": file_print("measure.choosing"),
        "sending_print": file_print("measure.sending"),
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
        "loop_print": file_print(Session.__module__),
        #: **Two hashes of one file, and they are not interchangeable.** `loop_print` is the byte
        #: hash, kept under that key because every recorded conversation result holds it and
        #: `sending.deferral()` reads it out of them. `loop_print_norm` is the same file with
        #: `\r\n` normalised — the value `reoffer.py` puts in *its* filename and, confusingly,
        #: stores under `loop_print`. `l` in a filename means the normalised one in both files.
        "loop_print_norm": loop_norm, "remember_unit": "turns",
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
