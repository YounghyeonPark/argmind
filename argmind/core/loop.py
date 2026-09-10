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

## What a key is, and who is allowed to know — decided 2026-09-10

The reply-shape literal in `converse.py` names three keys; `env`'s `Hands` accepts fifteen names over
fourteen keystrokes. Two separable defects sat in that gap and they get separate answers.

**The core guessing the vocabulary is a defect and it is fixed here.** `knows_key()` asks the
environment, through `getattr` and an honest fallback, exactly as `app.py` resolves `at()` and
`pointing_is_exact()` — and for the reason `contract.py` gives, sharpened: the objection to a
protocol member is not that an environment might lack a vocabulary (anything with `press` has one)
but that **the whole vocabulary is the wrong quantity for the caller**, since a phone publishes
hundreds of key names to be consulted by one line here.

**The prompt staying narrow is a decision, not the same defect.** The cost of the narrowing today is
**unknown and plausibly zero**: no goal in `measure/goals.json` needs a key beyond enter or tab, and
across the recorded reply and conversation runs the model has emitted `enter`, `tab` and `zoom in` —
the third being an invented keystroke rather than a key the narrow list denied it. Against that, the
*open* list is measured at 22-71 s and a turn that produced nothing, and fifteen names rendered into
a prompt is nearer to that than to the three that behaved. So the environment now declares and the
core now checks, and **what the model is shown does not move until something measures a request that
the three keys cannot serve.** `Session.unknown_keys` is that instrument.

**This paragraph is here rather than in `converse.py` for the reason the one below it gives** —
`prompt_print()` hashes that whole file, so writing the rationale where it belongs would fork the
fingerprint of three recorded baselines without a byte of the prompt changing.

## How much of the conversation is carried, and in what unit — measured 2026-09-10

`measure/conversation.py`, ~4,600 model calls on `qwen3:8b` with thinking off, against four captured
screens of 12-152 widgets, 3 repeats, worst counted.

**The window is the mechanism, not the model's attention.** Over 90 probes that referred with a bare
pronoun to something established earlier: with the anchoring turn inside the window the reference
held **40 of 45**; with it truncated out, **1 of 45**. Nothing else in the sweep comes close to
explaining the verdicts, so the setting below is a choice of *how many turns to anchor* and not a
tuning of how well the model attends.

**The old setting forgot in five turns.** `remember = 12` was a line count nobody had measured; a
turn costs 2.11 lines (median, n=405 conversations), so twelve lines was a **5.7-turn** window, not
the three turns this docstring used to claim. A first-turn reference held 3/3 at distance 1, 2/3 at
3, and **0/3 at 5, 8 and 12** — and at distance 12 two of three cases clicked something **wrong**
rather than nothing, which is the failure mode that matters: forgetting does not go quiet, it
proposes the wrong widget.

**Turns, not lines, and the unit change is half the finding.** A line is nobody's unit — the person
counts turns and the context counts tokens. Over the 3,312 turns that had a conversation behind them
a turn cost exactly two lines 88.9% of the time and three to five in the rest; the expensive ones
are the turns that *offered to do something*, which are exactly the turns a follow-up refers to, so
the old unit shrank the memory precisely where there was most to remember. The same 3,312 prompts —
the same set, since a turn with a history behind it is a prompt with a window in it — opened
**mid-turn 353 times (10.7%)**: 7.6% on a reply whose question had already been dropped, 3.0% on an
orphaned "offered". `recent()` cuts on turn boundaries, so that cannot happen.

**Twelve turns, derived.** It is the deepest thread the measurement reached with the reference still
holding: 7 of 9 runs and 2 of 3 cases at distance 12 with the whole conversation in the window, and
9 of 9 runs on all three cases at distance 8. Twelve is therefore the last measured point rather
than a margin above one, and a thirteenth turn is a guess in the same direction. It costs ~25 lines
at 2.11 lines a turn — about **400 prompt tokens** at the 13.5-16.6 tokens a line measured across
the three screens — against
a largest prompt observed anywhere of **2,293 in a context of 8,192**. The worst turn seen was 5
lines, so even a pathological twelve turns is ~960 tokens and still fits. **There is no second cap
on the block**, and the trigger for adding one is a measured prompt above ~4,000 tokens or a raised
`MAX_LISTED`, not a feeling that it is unbounded.

**Where the history block sits, and why it stays there.** It is pasted in *before* the widget list,
so every new line shifts every widget line after it and a server's prompt-prefix cache has nothing
to reuse past the system block. Putting the list first would make it a reusable prefix — but only
while the list is byte-identical between calls, and the measurement at the top of this file says a
listed widget is **gone 13 times in 14** over the second a call takes, so the prize is much smaller
than it looks. It is also a change to the prompt, which this project is 0 for 5 on, and it moves the
list away from the question it answers, which is where 18-of-28 choosing accuracy was measured.
Refuted by two measurements in this order: prefill shown to dominate a call, then
`measure/choosing.py` unmoved after the reorder. **This paragraph is here and not in `converse.py`
because `measure/choosing.py`'s `prompt_print()` hashes that whole file** — a comment in it forks
the fingerprint every recorded run carries, so the one file that most deserves a written rationale
is the one where writing it costs comparability.

**What is not claimed.** 24 lines and 48 lines were *the same prompt* at distance 8 — the whole
conversation was 18-24 lines, so both windows held all of it — and the 2/3-against-3/3 between them
is repeat noise, not evidence that more is better. Nothing is measured beyond 12 turns in either
direction. A bigger window did not make the agent chattier (21/30/27/27 spurious effects over 45
filler conversations each at 6/12/24/48 lines) and no rise in call time was measurable behind the
noise: the medians run 0.57-1.09 s across all four arms and the *largest* windows produced the
fastest rows, which is another process on the same card rather than an effect of memory.
"""

from __future__ import annotations

import time

from .converse import converse
from .intent import Said

__all__ = ["Session", "Exchange", "Done", "recent", "REMEMBER_TURNS"]

#: A turn opens when somebody speaks or a watch notices something. **Every other kind of line — what
#: was said back, what was offered, done, refused or turned down — belongs to the turn already in
#: progress**, so these two names are where a turn boundary is, for the writer and the reader alike.
#: Defined once because `think()` writes them and `recent()` cuts on them, and two spellings of the
#: same rule is two chances for it to drift.
SPOKE, NOTICED = "them", "noticed"
TURN_OPENS = (SPOKE, NOTICED)

#: How many turns of the conversation the next question carries. **Twelve turns, derived from the
#: sweep** — see the module docstring for the whole of it, and for why this is a count of turns and
#: not of lines. The short version: the reference held 40 of 45 times with its anchor inside the
#: window and 1 of 45 with it truncated out, so this number decides how far back a *"그거"* can point
#: and nothing else in the measurement does.
REMEMBER_TURNS = 12


def recent(history, turns):
    """The last `turns` turns of a history, cut where a turn begins and never inside one.

    `turns=None` is the whole conversation and `turns=0` is none of it. **The second one is a bug
    fixed rather than a choice**: this was `history[-remember:]`, and `history[-0:]` is the *entire*
    list, so the one setting a person reaches for to turn memory off turned it all the way up.
    Anybody tuning the number downward met it. `None` now names the unbounded case, which the sweep
    needed and previously had to fake with a number larger than the conversation — a fake that stops
    being unbounded the moment the conversation grows, which is the same blindness as a floor that
    cannot see a count rise.

    **Cutting on turn boundaries is the other half of the unit change.** Measured over 3,312 prompts
    built by the line-slicing version, 353 (10.7%) opened mid-turn: 7.6% began with a reply whose
    question had already been dropped and 3.0% with an orphaned "offered". A model shown an answer
    with no question is being handed a puzzle nobody meant to set.
    """
    if turns is None:
        return list(history)
    turns = int(turns)
    if turns <= 0:
        return []
    opens = [i for i, (who, _text) in enumerate(history) if who in TURN_OPENS]
    #: Fewer openings than the window means the whole thing fits — including any lines written
    #: before the first turn opened, which `remember_doing()` can legitimately produce.
    return list(history if len(opens) <= turns else history[opens[-turns]:])


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

    def __init__(self, *, eyes, model, hands, limit: int = 200,
                 remember_turns: int | None = REMEMBER_TURNS,
                 watches=None, may_act_unprompted: bool = False) -> None:
        self.eyes = eyes
        self.model = model
        self.hands = hands
        self.limit = int(limit)
        #: How many turns of back-and-forth to carry. **Turns, and the name says so because it used
        #: to be `remember` and used to mean lines** — a silent redefinition of a number would be
        #: the exact defect this project keeps catching, so the old spelling raises `TypeError`
        #: instead of quietly meaning 2.11 times as much.
        #:
        #: `None` carries the whole conversation; `0` carries none of it. Measured: at 12 lines
        #: (5.7 turns) a first-turn reference was gone by turn five and by turn twelve two cases in
        #: three clicked the *wrong* thing. The cost of twelve turns is ~400 prompt tokens beside a
        #: widget list that is ~1,600 for 142 widgets.
        self.remember_turns = remember_turns if remember_turns is None else int(remember_turns)
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
        #: Key effects dropped because the environment has no key by that name. **The size of the
        #: gap between what the prompt advertises and what the environment has**, which is the only
        #: honest way to find out whether the narrow advertised list is costing anything: nothing in
        #: `measure/goals.json` needs a key beyond enter or tab, so a request that wanted one has
        #: never been recorded and this is what would record it. See `decide`.
        self.unknown_keys = 0

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
        """One model call against the world as last observed. Returns an `Intent`.

        **An effect naming a key this environment does not have is dropped here.** That is what
        `converse._effect` already says happens to "a key that is not a key" and cannot do:
        `converse()` is handed a model and a list of widgets and never the environment, so the core's
        only notion of what a key is was the three names in the reply-shape literal. Recorded across
        the reply and conversation harnesses, the model has emitted `enter`, `tab` — and `zoom in`,
        which is not a key on any platform.

        **Dropped, and dropped here rather than in `perform`,** for three separate reasons:

        * `press` raises on an unknown name and nothing between it and `think()` catches, so one
          invented key costs the **whole turn** — including the sentence the model had already
          composed. That is the same outcome as the degenerate reply the key list was closed to
          avoid, reached by another road;
        * `perform`'s `refused` bucket means *it had gone by then*, and `remember_doing` writes that
          phrase into the conversation the next call reads. An invented key did not go anywhere, and
          filing it there would put a false sentence in the core's own context;
        * an interface proposes from this intent **before** `perform` is reached, so a filter in
          `perform` would still offer a person a button reading `키 zoom in` that cannot work.

        The residual is recorded rather than repaired: the model may still *say* it zoomed. That is
        the claim-without-action shape `measure/saying.py` already counts, and it is strictly better
        than pressing something invented.

        **This does not widen what the model is shown, and that is a decision.** The environment
        declaring its vocabulary and the prompt rendering it are two changes, not one; only the
        first is free. See `contract.py` and `env`'s `keys()`.
        """
        if not self.seen:
            self.observe()
        intent = converse(self.model, self.seen, trigger,
                          history=self.recent(), limit=self.limit)
        usable = [e for e in intent.effects if e.kind != "key" or self.knows_key(e.text)]
        self.unknown_keys += len(intent.effects) - len(usable)
        intent.effects = usable
        return intent

    def recent(self):
        """Exactly what the next question will carry of the past.

        Public because it is the answer to *"what does it still remember?"*, and an interface or a
        harness that reconstructed it from `history` and `remember_turns` would be a second
        implementation of the thing under test — `measure/conversation.py` reads the real prompt for
        that reason and this is the cheaper way to the same honesty.

        The call below is the *module* function of the same name, not this method: a method body
        resolves globals, never the class it sits in. Same name because it is the same idea with the
        setting already applied, and a second name would invite a second answer.
        """
        return recent(self.history, self.remember_turns)

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
        self.history.append((SPOKE if trigger.asked else NOTICED, trigger.text))
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

    def knows_key(self, name: str) -> bool:
        """Whether this environment has a key by that name.

        **The sibling of `find`, and public for the same reason**: an interface that shows a person
        what an action would do has to answer this too, and answering it a second way would be a
        second answer. `find` resolves a click's target against the world as it is; this resolves a
        key's name against the world as it is, and it is the *only* check a key gets — a key names
        nothing on screen, so there is no rectangle to look up and nothing to re-resolve.

        **Read with `getattr` and an honest fallback, not from a protocol member.** The reasoning is
        `contract.py`'s, applied to a case where its weak half does not hold: anything implementing
        `press` has some vocabulary, so this is not a capability that might be missing. It is off the
        protocol because **the whole vocabulary is the wrong quantity for the caller** — a phone
        exposes hundreds of key names, and obliging every environment to publish all of them so that
        this can consult a handful is a contract paid for by everyone and used by one line.

        An environment that declares nothing keeps every key, which is exactly the behaviour before
        this existed. The fallback is safe in a way the corresponding fallback for a *click* would
        not be: it can only over-advertise, and an over-advertised key costs a refusal from `press`,
        never a press landing on something that was not chosen.
        """
        vocabulary = getattr(self.hands, "keys", None)
        if vocabulary is None:
            return True
        return str(name).lower() in set(vocabulary())

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
            #: Key effects the environment has no name for. Zero is the expected reading and a
            #: rising one is the first evidence that the three advertised keys are too few — or
            #: that the model is inventing keystrokes, which is the same number seen from the
            #: other side and is why it is reported rather than merely counted.
            "unknown_keys": self.unknown_keys,
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
