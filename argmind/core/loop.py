# SPDX-License-Identifier: Apache-2.0
"""Perceiving, deciding and acting — three things, not three stages.

The first version welded them: one utterance produced one reading, one decision and at most one
click. That sequence was an artefact of the loop, not a property of the problem. **Acting is not a
stage that always runs, it is something decided**, and a decision can be started by more than being
spoken to. So the pieces are separate and callable on their own:

    observe()          read the world                41-266 ms measured, needs no model
    poll()             anything a watch cares about?  -> 0..n triggers
    decide(trigger)    one model call                 -> an Intent: what to say, 0..n to do
    perform(intent)    each effect re-resolved against the world AS IT IS NOW, or refused

`think(trigger)` runs all four and records an `Exchange`; `exchange(text)` is `think(Said(text))`,
which is the common case and not the shape.

## The world moves under every decision, and that is measured

On a real desktop while somebody works, over the second a model call takes:

    waited   anything changed   a widget that was there is GONE   the window itself changed
     0.3 s        11/14                    10/14                          2/14
     1.0 s        13/14                    13/14                          6/14
     2.0 s         6/14                     6/14                          2/14

(Not monotone — the 2 s row caught a quieter minute. The signal is the 1 s row: **thirteen times in
fourteen, something that existed when the question was asked is gone by the time the answer
arrives.**)

So an effect names its target and the name is looked up **again**, immediately before acting.
Clicking a remembered rectangle is clicking where something used to be.

**The same measurement is why a watch has a floor.** If perception changes 13 seconds in 14 and a
model call costs 0.3–2 s, "think whenever the screen changes" is a model that never stops and an
agent that never listens. `Watch.gap` is part of the design, not a tuning knob.

## Unprompted action, and why the default is to talk instead

An agent that acts *because it was asked* is watched by the person who asked. One that acts on its
own judgement may not be. The three brakes built for the first case — a `live` flag, a rate limit, a
guard before each action — all assume "you asked, it does one thing"; none of them covers "it
decided, repeatedly, while you were reading something else".

So an unprompted effect without standing permission is **downgraded to speech** rather than refused:
it becomes *"the build finished — want me to open it?"*. The refusal path is conversational, which is
what a thing you talk to should do, and it is the default because a wrong click has no undo.

**Confirmation stays cheap**, which is the second thing the environment being the source of truth
buys. A pixel agent compares screenshots and guesses; this one re-reads the widget layer and asks a
question with an answer: is the thing I clicked still exactly where it was?

**No caching.** The previous attempt made one the centrepiece; the call it would skip costs 0.16 s.
`Session.blind` counts the thoughts where widgets were listed and nothing was chosen, because that
count — not an intuition — is what would justify anything slower and cleverer.
"""

from __future__ import annotations

import time

from .converse import converse
from .intent import Said

__all__ = ["Session", "Exchange", "Done"]


class Done:
    """One effect that actually ran: what was decided, what it resolved to *then*, and whether the
    world moved.

    Three separate things on purpose. "It chose a target", "the target still existed" and "the click
    landed" have been conflated here before, and each can be true while the next is false.
    """

    __slots__ = ("effect", "target", "confirmed")

    def __init__(self, effect, target=None, confirmed=None) -> None:
        self.effect = effect
        #: The widget as it was at the moment of acting — not the one the model chose from.
        self.target = target
        #: `None` in a dry run: nothing was clicked, so nothing should have changed, and `False`
        #: there would read as a failure that never happened.
        self.confirmed = confirmed

    def __repr__(self) -> str:
        where = self.target.name[:22] if self.target is not None else self.effect.text[:22]
        return f"Done({self.effect.kind} {where!r}, confirmed={self.confirmed})"


class Exchange:
    """One thought and what came of it, kept so "fast" and "right" can both be audited."""

    __slots__ = ("n", "trigger", "said", "did", "proposed", "refused", "n_widgets",
                 "look_s", "think_s", "act_s")

    def __init__(self, n, trigger, said, did, proposed, refused, n_widgets,
                 look_s, think_s, act_s) -> None:
        self.n = int(n)
        self.trigger = trigger
        self.said = said
        self.did = list(did)
        #: Effects that would have run had the agent standing permission to act unprompted. Kept
        #: apart from `did` so "it wanted to" and "it did" are never the same field.
        self.proposed = list(proposed)
        #: Effects dropped because the target had gone by the time it was to be touched.
        self.refused = list(refused)
        #: What the model was shown. **A reply cannot be judged without it**: "there are no windows
        #: open" is right on an empty list and wrong on a full one.
        self.n_widgets = int(n_widgets)
        self.look_s = float(look_s)
        self.think_s = float(think_s)
        self.act_s = float(act_s)

    @property
    def total_s(self) -> float:
        return self.look_s + self.think_s + self.act_s

    def __repr__(self) -> str:
        return (f"Exchange({self.trigger!r} -> {self.said[:22]!r} did={len(self.did)} "
                f"proposed={len(self.proposed)} refused={len(self.refused)}, "
                f"{self.total_s * 1000:.0f}ms)")


class Session:
    """A conversation with an environment attached. Knows nothing about any platform."""

    def __init__(self, *, eyes, model, hands, limit: int = 200, remember: int = 12,
                 watches=None, may_act_unprompted: bool = False) -> None:
        self.eyes = eyes
        self.model = model
        self.hands = hands
        self.limit = int(limit)
        #: How many lines of back-and-forth to carry. **Lines, not turns** — a turn is now two to
        #: four of them, because what was done is recorded beside what was said. Twelve is about
        #: three turns. Still short on purpose: the widget list is the bulk of the prompt at ~1,600
        #: tokens for 142 widgets, and these lines are a handful of words each.
        self.remember = int(remember)
        #: What the agent may notice on its own. Empty means it only ever answers, which is the
        #: default because noticing is the half that can act while nobody is watching.
        self.watches = list(watches or [])
        #: **The permission that separates a helper from a hazard.** Off by default: an unprompted
        #: effect then becomes a sentence instead of a click.
        self.may_act_unprompted = bool(may_act_unprompted)

        self.history: list[tuple[str, str]] = []
        self.exchanges: list[Exchange] = []
        #: The world as last read. `decide` asks against it, `poll` compares against it.
        self.seen: list = []
        #: Thoughts where widgets were listed and nothing was chosen.
        self.blind = 0
        #: Effects refused because the target had gone by the time it was to be touched. Counted
        #: because it is the size of the staleness problem in real use.
        self.vanished = 0

    # ---- the pieces, each usable on its own -----------------------------------------------------

    def observe(self):
        """Read the world and keep it. Independent of any decision, so it can run at its own rate."""
        widgets, took = self.eyes.timed_foreground()
        self.seen = widgets
        return widgets, took

    def poll(self) -> list:
        """What has changed that a watch cares about, as triggers.

        Returns `[]` almost always, and that is the point: with no watches, or inside their floor,
        noticing costs one screen read and starts nothing. The first call after construction returns
        nothing whatever happens — there is no "before" to compare against, and treating an empty
        one as such would report the whole screen as newly appeared.
        """
        before = self.seen
        after, _ = self.observe()
        if not self.watches or not before:
            return []
        return [t for t in (watch.fired(before, after) for watch in self.watches) if t is not None]

    def decide(self, trigger):
        """One model call against the world as last observed. Returns an `Intent`."""
        if not self.seen:
            self.observe()
        return converse(self.model, self.seen, trigger,
                        history=self.history[-self.remember:], limit=self.limit)

    def perform(self, intent):
        """Run what may be run. Returns `(did, proposed, refused)`.

        Each effect is resolved **now**, not against the list the model chose from — thirteen times
        in fourteen that list is already wrong about where something is.
        """
        did, proposed, refused = [], [], []
        for effect in intent.effects:
            if effect.unprompted and not self.may_act_unprompted:
                #: Not an error and not silence: it becomes something to say. See the module
                #: docstring for why that is the default.
                proposed.append(effect)
                continue
            if effect.kind == "click":
                target = self.find(effect.target, effect.role)
                if target is None:
                    #: It was there when the question was asked and it is not there now. Refusing is
                    #: the only honest option: whatever occupies that rectangle now was never chosen.
                    self.vanished += 1
                    refused.append(effect)
                    continue
                self.hands.click(target)
                did.append(Done(effect, target, self._confirm(target)))
            elif effect.kind == "type":
                self.hands.type_text(effect.text)
                did.append(Done(effect))
            elif effect.kind == "key":
                self.hands.press(effect.text)
                did.append(Done(effect))
            else:
                refused.append(effect)
        return did, proposed, refused

    # ---- and tied together, for the common case -------------------------------------------------

    def exchange(self, heard: str) -> Exchange:
        """Somebody said something. Look, think, do what may be done."""
        return self.think(Said(heard))

    def think(self, trigger) -> Exchange:
        """One whole thought, timed in three parts so a regression lands on the piece that caused it
        rather than on the total."""
        _widgets, look_s = self.observe()

        t0 = time.perf_counter()
        intent = self.decide(trigger)
        think_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        did, proposed, refused = self.perform(intent)
        act_s = time.perf_counter() - t0

        #: An unprompted thought goes into the history under a different name, so the next reply can
        #: tell "you said" from "I noticed" — which is the difference between answering and
        #: interrupting.
        self.history.append(("them" if trigger.asked else "noticed", trigger.text))
        if intent.say:
            self.history.append(("you", intent.say))
        self.remember_doing(did, refused)
        if self.seen and not intent.effects:
            self.blind += 1

        ex = Exchange(len(self.exchanges), trigger, intent.say, did, proposed, refused,
                      len(self.seen), look_s, think_s, act_s)
        self.exchanges.append(ex)
        return ex

    def remember_doing(self, did=(), refused=(), offered=(), turned_down=()) -> None:
        """Put what happened into the conversation, beside what was said.

        **Repair was working by luck and this makes it deliberate.** Measured: told to pick red, it
        picks orange and says *"빨간색을 선택할 수 있어요"*; told *"not that one, press Red"*, it
        gets it right — because the sentence it spoke is in the history and the word *that* has
        something to point at. When it says something vague instead, or says nothing, the reference
        dissolves and the correction has nowhere to land.

        So the record does not depend on the model having narrated itself. It also gives an answer
        to *"what did you just do?"* that is not an invention — the measured failure being that the
        question made it do the thing a second time.
        """
        for effect in offered:
            self.history.append(("offered", _describe(effect)))
        for done in did:
            self.history.append(("did", _describe(done.effect, done.target)))
        for effect in refused:
            self.history.append(("could not", f"{_describe(effect)} — it had gone by then"))
        for effect in turned_down:
            self.history.append(("they said no to", _describe(effect)))

    # ---- resolving against the world as it is ---------------------------------------------------

    def find(self, name: str, role: str = ""):
        """The thing with this name and role, in the world as it is now — or `None` if it has gone.

        Public because an interface that shows a person what an action would touch has to resolve
        it too, and resolving it a second way would be a second answer.

        **The measurement that forces this**: over the second a model call takes, thirteen times in
        fourteen a widget present when the question was asked is absent when the answer arrives. A
        decision is made against a snapshot and executed against the world, and those are not the
        same thing.

        Matched on the name because **the rectangle is exactly what moves** — a list that scrolled, a
        window that resized. Among duplicates the one nearest to where a same-named thing was last
        seen wins, which assumes a thing moves rather than teleporting past its own copies.

        **The role is the other half of the key and it was missing.** Measured 2026-09-09 on
        Character Map: a `text` label reading `Font :` sits immediately before a `combo box` of the
        same name, the model chose the combo box, and this returned the label because the label
        comes first. The click landed on a piece of text and did nothing — which is indistinguishable
        from a click that missed, and cost a wrong diagnosis before it was found.

        An empty `role` still matches anything, so a caller that has only a name is not refused.
        """
        fresh = self.eyes.foreground()
        same = [w for w in fresh if w.name == name and (not role or w.role == role)]
        if not same:
            return None
        if len(same) == 1:
            return same[0]
        was = next((w.centre for w in self.seen
                    if w.name == name and (not role or w.role == role)), same[0].centre)
        return min(same, key=lambda w: (w.centre[0] - was[0]) ** 2 + (w.centre[1] - was[1]) ** 2)

    def _confirm(self, widget) -> bool | None:
        """Did the world move? Asked of the widget layer, not of pixels.

        `None` in a dry run: nothing was clicked, so nothing should have changed, and `False` there
        would read as a failure that never happened.
        """
        if not self.hands.live:
            return None
        after = self.eyes.foreground()
        unchanged = any(w == widget for w in after)
        window_changed = bool(after) and after[0].window != widget.window
        return window_changed or not unchanged

    # ---- evidence -------------------------------------------------------------------------------

    def report(self) -> dict:
        if not self.exchanges:
            return {"thoughts": 0}
        done = [d for e in self.exchanges for d in e.did]
        clicks = [d for d in done if d.effect.kind == "click"]
        return {
            "thoughts": len(self.exchanges),
            "asked": sum(1 for e in self.exchanges if e.trigger.asked),
            "noticed": sum(1 for e in self.exchanges if not e.trigger.asked),
            "acted": len(done),
            "spoke_only": sum(1 for e in self.exchanges if not e.did),
            #: Wanted to act unprompted and was not allowed to. **The number that says whether the
            #: permission default is costing anything**, kept apart from what it actually did.
            "proposed_only": sum(len(e.proposed) for e in self.exchanges),
            "nothing_matched": self.blind,
            "gone_before_click": self.vanished,
            "median_look_ms": round(1000 * _median(e.look_s for e in self.exchanges)),
            "median_think_ms": round(1000 * _median(e.think_s for e in self.exchanges)),
            "median_total_ms": round(1000 * _median(e.total_s for e in self.exchanges)),
            "confirmed": sum(1 for d in clicks if d.confirmed is True),
            "unconfirmed": sum(1 for d in clicks if d.confirmed is False),
        }


def _describe(effect, widget=None) -> str:
    """One short phrase for an effect, in the words the person would recognise."""
    if effect.kind == "click":
        role = widget.role if widget is not None else (effect.role or "thing")
        return f'clicked the {role} "{effect.target}"'
    if effect.kind == "type":
        return f'typed "{effect.text}"'
    return f"pressed {effect.text}"


def _median(xs) -> float:
    xs = sorted(xs)
    if not xs:
        return 0.0
    mid = len(xs) // 2
    return xs[mid] if len(xs) % 2 else (xs[mid - 1] + xs[mid]) / 2
