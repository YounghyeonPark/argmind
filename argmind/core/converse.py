# SPDX-License-Identifier: Apache-2.0
"""One call: what to say, and what — if anything — to do.

The model answers both in a single round trip, because latency is the budget and two calls would
double the only expensive thing in the loop. Measured on this machine with a 142-widget list:
**67–85 ms to first token**, 156–200 ms in total.

    {"do": [], "say": "..."}                           answered, nothing touched
    {"do": [{"click": 3}], "say": "..."}               answered, and one thing to do
    {"do": [{"click": 3}, {"key": "enter"}], "say": "..."}   several

**"do" comes before "say", and that ordering IS the fix for a communication defect.** Measured over
32 requests whose correct reply contains no number at all: with "say" first the model spoke the
internal index in **6 of 32** replies — *"지우개는 22번 버튼입니다"*, and twice the whole reply was the bare
number — and with "do" first, **0 of 32**, exact paired p = 0.031. The reading is that the number is
the freshest thing the model has when it reaches the first field it must fill: put the field the
number belongs in *first* and the address is discharged where it belongs, leaving the speech to be
composed after the decision rather than in place of it. It is the same de-referencing `_effect()`
already does, moved one step earlier. **This was not a sentence added to the prompt.** Five have
been added to `SYSTEM` over this project's life and all five slid the model between acting and
refusing without moving the line; the two things that ever moved it are `MOST_AT_ONCE` and this.

**A placeholder that says "<text>" gets the nearest literal typed into it.** With "say" now the last
key in the envelope, `{"type": "<text>"}` produced `{"type": "say"}` **11 times in 68 conversational
turns** — the agent typing the word *say* into the person's application. Naming the slot
`"<text to type>"` takes that to 0. The placeholder is load-bearing, not decoration.

**The key slot is closed, and "|..." in it was measured to cost 22 to 71 SECONDS.** With "do"
first, `{"key": "<enter|tab|escape|...>"}` sends the model into degenerate repetition on a request
it cannot satisfy cleanly: asked *"확대해줘"* on Paint it emitted `{"key": "tab"}` **two hundred times**,
1,338 output tokens and 22 s on one wording; on another it ran to the 4,096-token ceiling — **71 s
and `done_reason: length`, so `Reply.json()` returned None and the turn produced nothing at all**.
`MOST_AT_ONCE` refuses the batch, so nothing is pressed; what it cannot refund is the wait. With the
alternatives closed the same call is 0.4-0.6 s and 22 output tokens, and no reply in a 111-call
sweep exceeds 40. **The quantity to watch is generated LENGTH, not seconds** — a sweep run while
this machine was busy shows 6-14 s stalls at 17-39 output tokens on the **baseline** as well, so
wall-clock tails here are prefill and belong to the machine, while 1,338 and 4,096 do not.
**The trade is real**: the environment declares **fifteen key names over fourteen distinct virtual
keys** (`esc` and `escape` are both `0x1B`) and this advertises three. Whether that costs anything
is **unknown, not zero** — no goal in `measure/goals.json` needs a key beyond `enter` or `tab`, and
`session.report()["unknown_keys"]` rising in real use is what would say otherwise.

**Two tallies of what the model has actually emitted were withdrawn from this docstring on
2026-09-10.** They read `enter` (41), `tab` (24) and `zoom in` (3), and **nothing in this repository
can re-derive them**: every harness stores aggregates and no per-effect record survives. A number
that cannot be re-derived is exactly what this project's standards forbid, and it sat here for a day
because correcting a comment in this file used to fork the measurement fingerprint. It no longer
does — `measure/choosing.py` hashes the prompt as rendered, and `--prompt-check` asserts that this
very edit does not fork it.

What survives the withdrawal is the shape of the argument, which does not depend on the counts: the
open list never produced a valid key that the closed one misses, and it did produce an invented
keystroke — the exact thing `env/windows.py`'s key table exists to refuse. **The contract question it
raised has been answered elsewhere**: `WindowsHands.keys()` declares the vocabulary, `Session`
consults it and drops a key the environment has no name for, so an invented keystroke no longer
costs the turn. What the model is *shown* is still this literal, and still a decision.

**Nothing to do is the common case and the important one.** A model pushed to always act will act,
and on a screen that does not contain what was asked for, whatever it picks gets clicked. The prompt
says so.

**The model chooses by NUMBER; the effect is stored by NAME**, and that split is deliberate:

* a number can be range-checked, so the model cannot invent a target — an out-of-range index is a
  hallucination, not a near miss, and clamping it would turn an invention into a click on whatever
  sits at the edge of the list;
* a name still refers to the world when the number no longer does. Measured: over the second a call
  takes, a widget that was listed is **gone 13 times in 14**, so the list the number indexes is
  stale by the time the answer arrives.

**An unprompted thought is told that it is one.** The prompt says whether a person asked or the
screen changed, because the two deserve different answers: being asked is a reason to act, and
noticing something is usually a reason to mention it.
"""

from __future__ import annotations

from .intent import Effect, Intent

__all__ = ["converse", "describe", "MAX_LISTED", "MOST_AT_ONCE"]

#: A widget list has to fit in the context beside the conversation. 142 elements measured at ~1,630
#: prompt tokens, so this is roomy at 8192 and stops one pathological window from filling it.
MAX_LISTED = 200

#: The most things one thought may do. Click the box, type the name, press enter — that is the
#: longest sequence this design has ever needed, and an answer longer than it is a misread turn
#: rather than an ambitious one.
MOST_AT_ONCE = 3

SYSTEM = """You are a helper the person talks to while they work at their PC.

You are shown the things currently on their screen, each with a number. Reply as a person would —
briefly, in the language they used.

If they asked you to do something and the listed things can do it, name the numbers. If they were
just talking, or the thing they want is not listed, say so and do nothing: a wrong click is worse
than an honest "I can't see that here"."""

UNPROMPTED = """Nobody asked you anything — you noticed this yourself. Mentioning it is usually
enough. Only propose doing something if it is clearly wanted, and expect to be asked first."""


def describe(widgets, limit: int = MAX_LISTED) -> str:
    """The list as the model sees it: a number, a role, and the name the interface itself uses.

    Names are passed through untouched — not lowercased, not translated. Measured: the model matched
    an English request to the Korean widget "닫기" without help, which is the case a keyword matcher
    fails and the reason a model is here at all.
    """
    return "\n".join(f'{i}: {w.role} "{w.name}"' for i, w in enumerate(widgets[:limit]))


def converse(model, widgets, trigger, *, history=None, limit: int = MAX_LISTED) -> Intent:
    """One thought. Always returns an `Intent`; an empty one is a real answer."""
    listed = list(widgets[:limit])
    lines = [SYSTEM]
    if not trigger.asked:
        lines.append(UNPROMPTED)
    lines.append("")
    if history:
        lines.append("Recently:")
        lines += [f"  {who}: {text}" for who, text in history]
        lines.append("")
    lines.append("On screen now:")
    lines.append(describe(listed, limit) if listed
                 else "  (nothing named — this surface exposes no widgets)")
    lines += [
        "",
        (f"They said: {trigger.text}" if trigger.asked
         else f"You noticed: {trigger.text}"),
        "",
        'Reply with only JSON: {"do": [ ... ], "say": "<what you say back>"} where each entry of',
        '"do" is {"click": <number>}, {"type": "<text to type>"} or {"key": "<enter|tab|escape>"}.',
        'Leave "do" empty if nothing should happen.',
    ]
    reply = model.ask("\n".join(lines))
    got = reply.json() or {}

    wanted = got.get("do") or []
    said = _said(got, wanted)
    #: **More things at once than any real request has is a sign the turn was misread**, and the
    #: honest response is to distrust the batch rather than to keep the first few. Measured over 28
    #: goals that genuinely want an action: 27 produce one or two effects, and the single one that
    #: produced three was wrong anyway. Measured over 34 turns that want none — questions about the
    #: screen, about the past, about the agent itself — the spurious ones come in threes, fours and
    #: an eight: *"클릭도 할 수 있어?"* proposed clicking eight things including the close button,
    #: while saying, correctly, "yes, I can click".
    #:
    #: The limit is `MOST_AT_ONCE` and not the eight this used to allow, because three is what the
    #: documented multi-step case actually is — click the box, type the name, press enter — and
    #: eight was a number nothing had ever justified.
    if len(wanted) > MOST_AT_ONCE:
        return Intent(said, [], trigger, reply)

    effects = []
    for item in wanted:
        effect = _effect(item, listed, trigger)
        if effect is not None:
            effects.append(effect)
    return Intent(said, effects, trigger, reply)


def _said(got, wanted) -> str:
    """What the person hears — including the case where the model filed it in the wrong place.

    Asking for `"do"` first buys a clean reply at the cost of one new way to be misunderstood: the
    model sometimes answers `{"do": [{"say": "화면에 20개의 버튼이 있어요."}]}`, putting the whole utterance
    inside the list of things to do. `_effect()` correctly drops that entry — it is not an effect —
    and the person then hears **nothing at all**, which is the worst outcome available: measured at
    **3 of 132 calls** on one wording of the prompt and 4 of 132 on another, always on a question
    that had a perfectly good answer waiting inside it.

    Recovering it is de-misfiling, not repair. It is the same move `_effect()` makes when it turns
    an index into a name, and it is deliberately incapable of changing anything else:

    * it never runs when the model filled `"say"` properly, so a correct reply cannot be rewritten;
    * it never adds, drops or alters an effect — the entry stays unusable and stays dropped;
    * it prefers the envelope to the list, so a reply carrying both is read the ordinary way.

    **Not fixed in the prompt on purpose.** Of six wordings measured, two provoke the misfiling
    and four do not, with no mechanism distinguishing them; at temperature 0 that makes a quiet
    wording luck rather than a fix, and luck is not something to ship. This makes the ordering
    robust to the wording instead of dependent on it.
    """
    said = str(got.get("say", "")).strip()
    if said:
        return said
    for item in wanted:
        if isinstance(item, dict) and isinstance(item.get("say"), str) and item["say"].strip():
            return item["say"].strip()
    return ""


def _effect(item, listed, trigger) -> Effect | None:
    """One entry of `do`, or `None` if it cannot be used.

    Unusable is not repaired. An index outside the list, a key that is not a key, a `type` with no
    text — each is the model having produced something rather than chosen something, and the right
    response to that is to drop it, not to find the nearest thing it might have meant.
    """
    if not isinstance(item, dict):
        return None
    if "click" in item:
        try:
            index = int(item["click"])
        except (TypeError, ValueError):
            return None
        if not 0 <= index < len(listed):
            return None
        #: The number becomes a NAME AND A ROLE here and never travels further as an index. This
        #: one line is the staleness fix: what gets acted on is looked up again in the world as it
        #: is then. **The role is half of the key, and leaving it out was a real defect** — a window
        #: with a label and a combo box both called `Font :` resolved to the label and the click did
        #: nothing.
        return Effect("click", target=listed[index].name, role=listed[index].role, because=trigger)
    if "type" in item:
        text = item["type"]
        return Effect("type", text=str(text), because=trigger) if isinstance(text, str) and text \
            else None
    if "key" in item:
        name = item["key"]
        return Effect("key", text=str(name).lower(), because=trigger) if isinstance(name, str) \
            else None
    return None
