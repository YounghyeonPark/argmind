# SPDX-License-Identifier: Apache-2.0
"""One call: what to say, and what — if anything — to do.

The model answers both in a single round trip, because latency is the budget and two calls would
double the only expensive thing in the loop. Measured on this machine with a 142-widget list:
**67–85 ms to first token**, 156–200 ms in total.

    {"say": "...", "do": []}                           answered, nothing touched
    {"say": "...", "do": [{"click": 3}]}               answered, and one thing to do
    {"say": "...", "do": [{"click": 3}, {"key": "enter"}]}   several

**WHICHEVER FIELD IS WRITTEN FIRST IS WRITTEN WITHOUT THE OTHER.** One mechanism, two opposite
defects, and it is the thing to know before editing the line that renders below. Speech first is
composed before the decision exists, so the freshest thing the model is holding — the widget's
index — lands in the sentence: *"지우개는 22번 버튼입니다"*, and sometimes the reply is the bare number.
Action first is committed before anything has been considered, so a rule the person stated a few
turns ago is never consulted — a rule lives in language and there is nowhere to put one in an
action slot — and the speech that should have carried it is written afterwards or not at all:
**68 turns of 1,845 clicked something and said NOTHING**, measured with "do" first. Reordering this
line repairs neither defect. It chooses which one to pay.

**"say" is first, and that is a MEASURED TRADE the owner settled on 2026-09-10 with both columns in
front of them — not a default and not an arbitrary ordering.** `qwen3:8b`, thinking off, the same
fixtures, worst of three repeats throughout:

                                             "do" first      "say" first        harness
    the index spoken to the person           0 of 32         5 of 32            saying
    a rule stated d turns ago obeyed         9 of 36        36 of 36            conversation, d ≤ 8
    correct decisions                       32 of 37        28 of 37            choosing
    turns that touch nothing                31 of 34        31 of 34            talking
    turns that said nothing at all      71 of 1,845     18 of 1,845             conversation
    a call, median                          0.59 s          0.60 s              conversation

The rule row is nine individual probes at each of d = 1, 3, 5, 8 — **9/9 at every one of them**
against 3/9, 3/9, 1/9, 2/9 — and its own control arm reads 3/9 at every distance, so the treatment
is what moved and not the screen. It is the largest effect anything in `measure/` has recorded for
this file.

**What it costs, at these sizes, is choosing and the leak.** Four correct decisions here, and **that
row is the noisiest thing in the table**: "say" first has read 21, 23 and 24 right of 28 across
runs — `qa62805`, `p5461db`, and a back-to-back repeat of `p5461db` — against 24 twice for "do"
first. Two to four is the size; one run is not. **Talking's two did not reproduce at all**: 31 of 34
both ways here, against 29 of 34 at `p5461db`. The leak is the dependable cost — five or six of 32
against zero, wherever it has been measured — and four of the five are conclusive, the index of a
widget the same reply clicked.

The reasoning the numbers do not settle is the owner's, and it is about interaction rather than
accuracy: **an agent that visibly disregards an instruction it was just given damages the
conversation more than one that occasionally says "22"** — an action is gated behind a button, and a
broken rule is gated by nothing.

**Neither column is a regression that somebody fixed.** Both are shipped configurations of this one
literal, measured on the same four harnesses; the fingerprints are `q1ec131` for the first and
`qa62805` for the second, and `measure/`'s result files carry them in their names. **This was never
a sentence added to the prompt.** Five have been added to `SYSTEM` over this project's life and all
five slid the model between acting and refusing without moving the line. The three changes that have
moved anything measured here are structural: `MOST_AT_ONCE`, this ordering, and closing the key list
— and only the first two touch the acting-versus-refusing line at all.

**A `say` filed inside `do` is a "do"-first failure and its recovery went with the ordering.**
`_said()` used to search `do` for an utterance the model had misfiled there — `{"do": [{"say":
"화면에 20개의 버튼이 있어요."}]}` — at 3 and 4 of 132 calls on two "do"-first wordings. Under "say"
first the envelope's own `say` was non-empty in **2,165 of the 2,165 calls that parsed**, of 2,183
logged across all four harnesses, so its precondition never occurred once. It was deleted rather
than left in as insurance, for two reasons. It never covered the failure that actually silences this
agent: `_said()` was live for all 71 silent turns above and recovered none of them, because a
"do"-first silence is a `say` that was never written, not one written in the wrong place. And a
silent repair costs something even when it never fires — `saying.py`'s `answer.empty` category is
the one place a misfile is visible, and a recovery upstream of it would take that away.

**The remaining 18 silences have a different cause and no repair here.** They ran to the 4,096-token
ceiling, so `Reply.json()` returns `None`, there is no reply to read a field out of, and the turn
produces nothing at all — see the key-slot paragraph below, which is where that failure lives.

**A placeholder that says "<text>" gets the nearest literal typed into it.** Measured with "say"
last in the envelope, `{"type": "<text>"}` produced `{"type": "say"}` **11 times in 68
conversational turns** — the agent typing the word *say* into the person's application. Naming
the slot `"<text to type>"` takes that to 0. The placeholder is load-bearing, not decoration, and
it is kept under either ordering: it cost nothing to name properly and nothing has measured the
failure to depend on where `say` sits.

**The key slot is closed, and "|..." in it was measured to cost 22 to 71 SECONDS.** Measured with
"do" first, `{"key": "<enter|tab|escape|...>"}` sends the model into degenerate repetition on a
request it cannot satisfy cleanly: asked *"확대해줘"* on Paint it emitted `{"key": "tab"}` **two
hundred times**, 1,338 output tokens and 22 s on one wording; on another it ran to the
4,096-token ceiling — **71 s and `done_reason: length`, so `Reply.json()` returned None and the
turn produced nothing at all**. `MOST_AT_ONCE` refuses the batch, so nothing is pressed; what it
cannot refund is the wait. With the alternatives closed the same call is 0.4-0.6 s and 22 output
tokens, and no reply in a 111-call sweep exceeds 40. **The quantity to watch is generated LENGTH,
not seconds** — a sweep run while this machine was busy shows 6-14 s stalls at 17-39 output tokens
on the **baseline** as well, so wall-clock tails here are prefill and belong to the machine, while
1,338 and 4,096 do not.
**Closing the list did not end the degeneration, and that correction is from 2026-09-10.** The
sweep behind "no reply exceeds 40" was `choosing.py`, which carries no conversation; with a history
in the prompt and "say" first, **18 of 1,845 turns still ran to the 4,096-token ceiling** — the same
`{"key": "tab"}` repetition, every one of them the same filler line, *"너 뭘 할 수 있어?"*, on one
screen, at 69-71 s each: **47% of that run's model time in 1.0% of its turns**. Against 3 of 1,845
with "do" first. The closed list is kept — a separate change with its own justification, which the
owner did not reverse — but what it bought is a **rate**, not an impossibility, and a reply that
hits the ceiling is a turn the person hears nothing back from. `num_predict` is where that is
bounded, and it belongs to `llm.py` rather than to this file.

**The trade is real**: the environment declares **fifteen key names over fourteen distinct virtual
keys** (`esc` and `escape` are both `0x1B`) and this advertises three. Whether that costs anything
is **unknown, not zero** — no goal in `measure/goals.json` needs a key beyond `enter` or `tab`, and
`session.report()["unknown_keys"]` rising in real use is what would say otherwise.

**Two tallies of what the model has actually emitted were withdrawn from this docstring on
2026-09-10.** They read `enter` (41), `tab` (24) and `zoom in` (3), and **nothing in this repository
can re-derive them**: every harness stores aggregates and no per-effect record survives. A number
that cannot be re-derived is exactly what this project's standards forbid, and it sat here for a day
because correcting a comment in this file used to fork the measurement fingerprint. It no longer
does — `measure/choosing.py` hashes the prompt as rendered, `--prompt-check` asserts that a prose
edit does not fork it, and **the say-first change exercised that for real**: this file's own hash
moved through six values on 2026-09-10 while `q` stayed `qa62805` from the moment the reply-shape
line was swapped, so the paragraphs above were written after the runs they report rather than
guessed before them.

What survives the withdrawal is the shape of the argument, which does not depend on the counts: the
open list never produced a valid key that the closed one misses, and it did produce an invented
keystroke — the exact thing `env/windows.py`'s key table exists to refuse. **The contract question it
raised has been answered elsewhere**: `WindowsHands.keys()` declares the vocabulary, `Session`
consults it and drops a key the environment has no name for, so an invented keystroke no longer
costs the turn. What the model is *shown* is still this literal, and still a decision.

**Nothing to do is the common case and the important one.** A model pushed to always act will act,
and on a screen that does not contain what was asked for, whatever it picks gets clicked. The prompt
says so.

## An unanswered offer is rendered where the decision is made, not only in the transcript

**The acceptance defect, diagnosed off the rendered prompt.** The person says yes to an offer and
nothing happens: `measure/reoffer.py` at `fa5771b`, `ACCEPT` **8 of 15** live cases, and **every one
of the seven failures is `ignored` rather than `drifted`** — the yes never reaches a widget at all,
so nothing is wrong with choosing. What the model actually said in those turns is the diagnosis, and
all four failure shapes are the same missing link:

    응                  -> "알겠습니다."                    an acknowledgement of small talk
    어                  -> "안녕하세요. 도와드릴게요."         read as a greeting
    ㅇㅇ                 -> "I can't see that here."       SYSTEM's own fallback clause, verbatim
    어 눌러줘             -> "어떤 버튼을 눌러야 해?"          "press it" understood, referent missing

The last one is conclusive. It parsed *press it* as a request and **asked which button** — while the
answer sat in the prompt above it, in a line reading `offered: to click the button "Eraser" —
nothing was done; they have not accepted`. So the offer was present and was not available: this is
reference, not memory and not choosing.

**Two things in the rendered prompt explain which yeses survive it.** First, `SYSTEM` partitions the
turn in two — *asked you to do something* or *just talking* — and a bare *"응"* is neither, so it
falls to the second branch by default; that is why bare acceptance is **1 of 5** and a phrased one
**7 of 10**, and why *"ㅇㅇ"* came back with the fallback sentence as its reply. Second, and no
wording explains this one, **the offer and the utterance are at opposite ends of the prompt**: the
whole widget list sits between the `Recently:` block and `They said:`. Split by screen over the same
run, `ACCEPT` is **6 of 8 live on Character Map (21 widget lines between them) against 2 of 7 live on
Paint (123)**, and the control runs the other way and by a wider margin — `MEMORY` stays quiet, which
is its correct answer, **2 of 6 on Character Map against 4 of 5 on Paint**. So the long list is not
simply making the model worse at the turn: whatever it costs, it costs acceptance specifically.

**That split is post-hoc, and it is a place to put a fix rather than a finding.** Two-sided Fisher,
computed: `ACCEPT` 6/8 against 2/7 gives **p = 0.1319**, `MEMORY` 2/6 against 4/5 gives **p =
0.2424**; neither reaches the floor. The distance variable has **two levels with one screen at each**
— `system-information` (152 lines) and `chrome-remote-desktop` (12) carry no `ACCEPT` or `MEMORY`
case at all — so *123 lines* and *Paint* are perfectly aliased, and prompt distance cannot be
separated from the application, its widget names, or the requests made of it, one of which is
*빨간색*, a colour swatch this project already has trouble choosing. And **`FARACCEPT` is 0 of 4, two
of those four on the 21-line screen**, with the offer verified present in the block every time: four
ordinary turns of conversational distance kill acceptance on the short prompt too, so prompt-line
distance is demonstrably not the only variable. What follows re-renders the offer regardless of
distance of either kind, which is what makes it worth doing on a split this weak.

So the offer is rendered **again, immediately above `They said:`** whenever the conversation carries
one nobody has answered. It is not moved: it stays in the transcript, where it is the true record of
what happened. This copy is a different thing — the state of the conversation at the moment of
deciding, recomputed every turn from the history rather than written into it, which is why it may say
*still open* where a history line may not. `loop.py` argues that one out: a line kept in the record
must stay true for as long as the record keeps it, and *"waiting for them"* is false the moment they
accept. A slot rebuilt per turn cannot go stale.

**It names the option space and gives no instruction.** *"a yes to that, a no, or nothing to do with
it"* is the whole of it. This project is 0 for 5 on sentences added to `SYSTEM`, and every one of
those slid the model along the acting/refusing line because it leaned one way; the three readings are
spelled out precisely so that this one does not — `REFUSE` and `MEMORY` need the second and third as
much as `ACCEPT` needs the first.

**A turn with no unanswered offer renders byte for byte what it rendered before.** `measure/choosing`
and `measure/talking` send no history at all, so neither can see this change — which also means
neither can vouch for it, and `measure/reoffer.py` is the only instrument that can. It has the same
blindness in its fingerprint: `q` is rendered over a probe history of `("them", "a"), ("you", "b")`,
which carries no offer, so **`q` does not move on this change** and two different prompts share
`qa62805`. Closing that means an `offered` line in `measure/choosing.py`'s `PROBE_HISTORY`, which
belongs to `measurement`.

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

__all__ = ["converse", "describe", "still_open", "MAX_LISTED", "MOST_AT_ONCE"]

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

#: The history label `Session.remember_doing()` writes an offer nobody has answered under, and the
#: three it writes a settled one under. **Duplicated across the module boundary on purpose**:
#: `loop.py` imports this module, so this module cannot import it, and a shared constant would have
#: to be a contract change. `tests/test_core.py` asserts the four spellings against what
#: `remember_doing()` actually writes, because a rename there and not here would stop the block
#: below rendering at all — a fix that evaporates in silence is worse than one that was never made.
OFFER_OPEN = "offered"
OFFER_SETTLED = ("did", "could not", "they said no to")

#: The two fixed lines of the standing-offer slot. The first localises it, the second names the
#: three things the person's next line can be. **No imperative, in either.** The offer's own words
#: come from the history between them and are not rewritten here: this module does not parse
#: `loop.py`'s prose, it repeats it.
STILL_OPEN = "Still open, from earlier:"
STILL_OPEN_TAIL = "What they say next may be a yes to that, a no, or nothing to do with it."


def describe(widgets, limit: int = MAX_LISTED) -> str:
    """The list as the model sees it: a number, a role, and the name the interface itself uses.

    Names are passed through untouched — not lowercased, not translated. Measured: the model matched
    an English request to the Korean widget "닫기" without help, which is the case a keyword matcher
    fails and the reason a model is here at all.
    """
    return "\n".join(f'{i}: {w.role} "{w.name}"' for i, w in enumerate(widgets[:limit]))


def still_open(history) -> list:
    """The offers nobody has answered yet, oldest first, in the words the history holds them in.

    An offer is opened by an `offered` line and closed by anything that settles it — the thing was
    done, it could not be done, or it was turned down. **Any settling line closes every open offer**,
    not the one it matches: matching would mean parsing `loop.py`'s sentence back into an effect, and
    under-reporting an open offer costs a re-render while over-reporting one puts a stale proposal
    next to the person's words, which is the defect this exists to fix, one turn late.

    Nothing else closes one. Four ordinary turns in between do not: `measure/reoffer.py`'s
    `FARACCEPT` cases are a yes five turns after the offer, they are **0 of 4**, and the offer is
    verified present in the rendered block every time — so distance is the thing to remove, not a
    reason to stop carrying it. The window bounds this on its own: `recent()` has already dropped
    everything older than `REMEMBER_TURNS`, and over 132 scored windows the count of `offered` lines
    reaching a prompt was 1 in 116, 2 in 7 and 0 in 9. There is no cap here because nothing has
    measured a need for one.
    """
    open_offers = []
    for who, text in history or ():
        if who == OFFER_OPEN:
            open_offers.append(text)
        elif who in OFFER_SETTLED:
            open_offers = []
    return open_offers


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
    #: **The standing offer, next to the words that may be answering it.** Only when somebody asked:
    #: the tail line is about what *they* say, and nobody said anything on an unprompted thought. A
    #: turn with no unanswered offer — every turn `measure/choosing.py` and `measure/talking.py`
    #: send, all of which carry no history — renders exactly what it rendered before this existed.
    pending = still_open(history) if trigger.asked else []
    if pending:
        lines.append("")
        lines.append(STILL_OPEN)
        lines += [f"  {text}" for text in pending]
        lines.append(STILL_OPEN_TAIL)
    lines += [
        "",
        (f"They said: {trigger.text}" if trigger.asked
         else f"You noticed: {trigger.text}"),
        "",
        'Reply with only JSON: {"say": "<what you say back>", "do": [ ... ]} where each entry of',
        '"do" is {"click": <number>}, {"type": "<text to type>"} or {"key": "<enter|tab|escape>"}.',
        'Leave "do" empty if nothing should happen.',
    ]
    reply = model.ask("\n".join(lines))
    got = reply.json() or {}

    #: `.get`, because **14 of 2,165 logged replies carried no `"do"` key at all** — the model
    #: finished the sentence and stopped. That is the same mechanism the docstring opens with, seen
    #: from the side this ordering pays it on, and it costs nothing: nothing to do is a real answer.
    wanted = got.get("do") or []
    #: The envelope's own field, and **the only place an utterance is looked for**. Under this
    #: ordering the model filled it in every one of those 2,165 replies, and the recovery that used
    #: to stand here went with the ordering it was built for — the docstring says why it was deleted
    #: rather than kept as insurance.
    said = str(got.get("say", "")).strip()
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
