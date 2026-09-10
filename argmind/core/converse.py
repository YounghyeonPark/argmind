# SPDX-License-Identifier: Apache-2.0
"""One call: what to say, and what — if anything — to do.

The model answers both in a single round trip, because latency is the budget and two calls would
double the only expensive thing in the loop. Measured on this machine with a 142-widget list:
**67–85 ms to first token**, 156–200 ms in total.

    {"say": "...", "do": []}                           answered, nothing touched
    {"say": "...", "do": [{"click": 3}]}               answered, and one thing to do
    {"say": "...", "do": [{"click": 3}, {"key": "enter"}]}   several

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
        'Reply with only JSON: {"say": "<what you say back>", "do": [ ... ]} where each entry is',
        '{"click": <number>}, {"type": "<text>"} or {"key": "<enter|tab|escape|...>"}.',
        'Leave "do" empty if nothing should happen.',
    ]
    reply = model.ask("\n".join(lines))
    got = reply.json() or {}

    wanted = got.get("do") or []
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
        return Intent(str(got.get("say", "")).strip(), [], trigger, reply)

    effects = []
    for item in wanted:
        effect = _effect(item, listed, trigger)
        if effect is not None:
            effects.append(effect)
    return Intent(str(got.get("say", "")).strip(), effects, trigger, reply)


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
