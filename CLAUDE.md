# Argmind — project context

**A local-model agent you talk to while you work at your PC.** An open-source LLM on the machine,
real-time interaction, and actions taken through the operating system's own accessibility layer.
Rebuilt into this shape on 2026-09-09; the name is older than the shape.

Author: Younghyeon Park <ypark.dev@gmail.com>. Apache-2.0 (`LICENSE`, `NOTICE`).
Side project — open source, not a product, no support commitment.

**Until 2026-09-09 this repository was something else** — a recursive-least-squares library for
on-device adaptation, ported to C++ for Cortex-M, with a classical-perception demo and an agent app.
About 50,000 lines. It was **removed**, not shelved, after the rebuilt core was measured: it is
intact under `before-removal-2026-09-09` and `argmind-v1` (`5e52f61`), and
`docs/findings/what-the-old-tree-measured.md` keeps the handful of results that still bear on work
here plus the commands to reach the rest. **Do not restore it to continue it.**

## The rule everything is arranged around

> **The deterministic side says WHERE. The probabilistic side says WHAT.
> A model never emits a coordinate.**

Measured on this machine before a line of the core was written. Asking a vision model for a pixel
took **24–82 s** and on a dense screen never answered; Windows UI Automation answers the same
question exactly in **41–266 ms**; a text model chooses among named things in well under a second.
The vision model could *see* — its reasoning log read every 14-pixel filename — it could not do
arithmetic on what it saw.

`tests/test_core.py` holds this against the source by walking the AST, because a rule that lives in a
docstring is a rule that gets quietly undone: no core module may ask a model for `x` and `y`, no core
module may hand a model an image, and nothing in `core/` may name a platform.

## Standards of evidence — carried across two rewrites deliberately

These came from a project where seventeen modelling errors were found and most of them flattered the
design. They cost real time to learn and they apply to anything measured.

- **Distinguish measured from estimated, every time**, in the same sentence as the number.
- **A model must be able to fail.** An analysis that cannot produce a negative result is not
  evidence. Protect a negative result with a test so a later change cannot quietly optimise it away.
- **A flattering result means find the responsible assumption first.**
- **A uniformly tidy result is evidence the measurement is broken**, not that the world is simple.
- **Scrutinise corrections that favour you as hard as ones that do not.**
- Accuracy figures carry the dataset, the conditions and the repeat count. "64%" without those is
  not a number.
- Cite URLs for external claims. "Not published" is a valid finding.

## Conventions

**Location: `C:\dev\argmind`.** Off Google Drive since 2026-08-08 — Drive sync can corrupt a git
repository by uploading partial writes. Backup belongs on a remote, not on a syncing filesystem.

Windows, PowerShell primary; Python 3.14; git. `python -m pytest` before quoting any number. A
tracked pre-commit hook in `.githooks/` runs the suite and blocks on failure — install once per
clone with `git config core.hooksPath .githooks`.

Needs [Ollama](https://ollama.com) running with a small text model, and `comtypes` for UI Automation.

## What is built

    argmind/core/    contract.py  Widget, Eyes, Hands — what an environment must provide
                     llm.py       the local model, every setting learned from a silent failure
                     intent.py    Trigger, Effect, Intent, Watch — what starts a thought and yields
                     converse.py  one call: what to say, and what if anything to do
                     loop.py      observe / poll / decide / perform, separately
    argmind/env/     windows.py   UI Automation for eyes, SendInput for hands
    argmind/app.py                the PC app — a window you leave running while you work
    argmind/__main__.py           app / look / talk / say / watch
    measure/                      the goal set, the captured screens, every answer behind the numbers

**48 tests, and none of them needs a display, a model or a network.** Re-derive with
`python -m pytest`; `tests/test_suite_size.py` floors it and records the staircase down from 1,252.

**Four separable pieces, because acting is not a stage that always runs — it is something decided**,
and a thought can start from a `Watch` noticing something rather than from being spoken to.
`observe()` reads the world and needs no model. `poll()` turns a change into a trigger. `decide()` is
the one model call. `perform()` runs each effect against the world *as it is then*.

**An effect stores a name, never a rectangle.** Over the second a model call takes, a listed widget
is **gone 13 times in 14**, so the target is looked up again immediately before acting and a miss is
a refusal. Clicking a remembered rectangle is clicking where something used to be.

**A watch carries a floor, and it is measured rather than tuned.** Perception changes in 13 seconds
of 14 while a model call costs 0.3–2 s, so "think whenever the screen changes" is a model that never
stops and an agent that never listens. `MIN_GAP = 3.0`.

**An unprompted effect without standing permission is downgraded to speech, not refused.** Every
brake built for "you asked, it does one thing" — `live`, a rate limit, a guard — fails to cover "it
decided, repeatedly, while you were reading something else". So the agent says *"the build finished
— want me to open it?"* instead of clicking. `may_act_unprompted` is off by default and
`--allow-unprompted` is a separate CLI word from `--live`.

### The priority, stated by the owner 2026-09-10

> **약간의 실수는 괜찮음. 계속 고쳐나가게 하면 됨. 중요한건 상호작용 및 의사소통임.**
> *A few mistakes are fine. Just let it keep correcting them. What matters is interaction and
> communication.*

**This moves the centre of gravity off the accuracy numbers below.** They are still the record of
what works, and they are not the thing to optimise. `measure/repair.py` is the first measurement on
this project that asks whether you can talk to it: **repair 3 of 4, memory of its own actions 0 of
1, Korean answered in Korean 5 of 6.**

**Repair works, and it was working by luck** — the model narrates its intent, so the sentence it
spoke is what a later *"not that one"* points at. `Session.remember_doing()` now records offers,
actions, refusals and "they said no" in the conversation beside the speech, and the app calls it
when it proposes, runs, or you press 아니오. It changed none of the four outcomes: not an
improvement, the removal of a dependence on the model having good manners.

**A question about the past makes it act again** (0 of 1) and that is being left alone. A prompt
line aimed at it did not fix it and broke a language case, taking Korean from 5 of 6 to 4 of 6 —
**the fourth prompt edit here to make something worse, 0 for 4**. In the app the effect is a
proposal with a button, so the cost is a row you ignore, which is the "a few mistakes are fine"
case exactly. `docs/findings/interaction.md`.

### Choosing accuracy — the number that gated the removal

**18 of 28 answerable goals right, 8 of 9 correct declines.** 37 goals against 4 real captured
windows of 12–152 widgets, labelled before anything ran, 3 repeats, worst counted; per-run 19/18/18,
so not a single draw. Nine goals have no right answer so the measurement can fail.

**Not good enough to click unsupervised, good enough to propose** — which is where the permission
default already sat for safety reasons, and the two arguments could have disagreed.

**My explanation of the errors was refuted by its own control.** Wrong answers sat 1–10 places from a
correct one, which looked like the vision model's arithmetic failure in a new guise. Shuffling each
screen expanded the median gap from 4 places (3.3% of the list) to 33 (26.8%): the neighbours were
**semantic**, because toolbars group related controls, and index proximity was confounded with
meaning proximity. Shuffling also costs accuracy (18 → 15 right, 8 → 13 wrong), so **the interface's
own order carries information and the list must not be re-sorted.** Asking for a name instead of an
index is safer and less capable (wrong 8 → 3, missed 3 → 12); nothing was changed on it.

Density is a **direction, not a rate**: 8 of 9 on the two small screens against 10 of 19 on the two
large ones is one-sided Fisher p = 0.070, n = 28, and the small screens are also the simpler
applications. Full conditions in `docs/findings/choosing-accuracy.md`.

### The PC app — `python -m argmind app`

A small always-on-top window: a transcript, an input box, and **a row with a button for every action
it wants to take**. Three measurements shaped it, and `docs/findings/the-pc-app.md` has them.

**The agent's window is on the screen it reads.** Once you click into the input box the foreground
window is the agent, so the eyes take `ignore_pid` and answer about the last window that was in
front and was somebody else's — and the header names it, always. The first version of that fix
seeded the remembered window during setup and **lost the race**: the app answered "I can't see any
windows here" twice while a 141-widget browser sat behind it, with no way to recover because it kept
focus. `attending()` now walks top-level windows in z-order when it has nothing remembered.

**Every action is proposed and a person presses the button.** `Session.perform` is called from
exactly one place in `app.py` and a test walks the AST to keep it that way. That is also why the app
builds its session with `may_act_unprompted=True`, which is **stronger than the library default, not
weaker**: per-action human confirmation beats a session-wide flag.

**The first two questions cost 4,715 ms and 5,940 ms; the third onwards cost 331, 329, 608.** Two
calls to warm, not one. The app warms at startup and shows a running counter while thinking; `--cold`
skips it and saves the 5-6 GB until first use.

**Clicking is switched off if the coordinates disagree.** `pointing_is_exact()` runs once: this
process is DPI-unaware and UI Automation reports physical pixels, so on a scaled display every click
would be offset. Measured 40 of 40 exact here, which is one desk and not a guarantee.

**A widget's centre resolves back to that widget 73% of the time** (14% a related name, 12.8%
something else, n=304). That is an **upper bound on missed clicks, not the miss rate** — caption
buttons resolve to `application` and the pixel is still correct — so it is shown as a note beside
the button and never used to refuse.

### Three interventions, none better than 26 of 37 correct decisions

**Every change that moved `right` up by one moved refusals down**, and none beat the baseline on
correct decisions (right + correct declines, out of 37):

| | right / 28 | declined / 9 | correct |
|---|---|---|---|
| as it stands | 18 | 8 | **26** |
| a stronger language rule in `SYSTEM`'s first paragraph | 19 | 3 | 22 |
| the same rule as a separate line after the refusal | 17 | 8 | 25 |
| dropping `text` widgets from the list (308 -> 217) | 19 | 7 | 26 |

Two were prompt edits in different places and one changed what the model can see, so this is not one
mechanism. The model sits on a trade between acting and refusing and these nudges slide along it.
Dropping `text` was the best-motivated of the three — a `text` value was one of the wrong clicks,
and a `text` label named `Font :` in front of a combo box of the same name caused a whole defect —
and it is a wash: identical correct decisions, two more harmful outcomes for two fewer harmless
ones, and a regression on a goal that worked. **Not adopted.** It also weakens the density
direction: a third off the list moves nothing.

### The model is where the gain was — `qwen3:8b`, thinking off, is the default

| model | right / 28 | correct / 37 | a call |
|---|---|---|---|
| `qwen2.5:7b` | 18 | 26 | 0.45 s |
| **`qwen3:8b`, thinking off** | 23 | **30** | **0.45 s** |
| `qwen2.5:14b` | 25 | 32 | 2.39 s |
| `qwen3:8b`, thinking on | **27 (96%)** | **34** | 3.75 s |

**Row one to row two is the only free change measured on this project** — same latency, four more
correct decisions. **One 5 GB model covers both ends**: `--think` reaches 96%, above the criterion's
nine in ten, at 3.75 s. The 14B sits between them, costs 10.4 GB, and needs a second download to
reach where a flag already goes.

**Neither end passes the whole criterion** (below nine in ten *or* over a second is failure).
Thinking off is 82% at 0.45 s; thinking on is 96% at 3.75 s. The default takes the latency clause: a
wrong proposal costs a glance and a "no", a wait is paid every turn, and a person confirms every
action anyway. **Bigger is not monotonically better** — the 14B gets two goals wrong that both 8B
configurations get right.

**A tenth way this harness produced a confident wrong number**: `llm.py` discards
`message.thinking`, correctly, and that hid what it cost — 201 output tokens and 1,112 characters of
reasoning for a twelve-token answer, with nothing in the `Reply` saying why. Its length is recorded
as `Reply.thought` now. `Local(think=False)` switches reasoning off and models that never heard of
the field accept it.

### The 14B experiment that came first, and broke the latency clause

**`qwen2.5:14b` is the default now.** Same 37 goals, same prompt, three repeats, **run twice and
identical both times**:

| | right / 28 | wrong *(no-answer)* | missed | declined / 9 | correct | a call |
|---|---|---|---|---|---|---|
| `qwen2.5:7b` | 18 | 8 *(1)* | 3 | 8 | 26 of 37 | **0.44 s** |
| `qwen2.5:14b` | **25** | 5 *(2)* | **0** | 7 | **32 of 37** | 2.38 s |

Thirteen goals moved, eleven for the better, and every goal the 7b missed entirely is gone. It is
still on the same trade — bad clicks went 1 → 2 — but it gains far more than it gives up, which is
what the three nudges never did.

**It fails one clause of this project's own criterion.** `REBUILD.md` says the design has failed *"if
a reply routinely takes more than a second"*, and 2.38 s is not close. The model is 10.4 GB and sits
**entirely in this machine's 16 GB of VRAM**, so that is what a 14B costs here, not a spill to tune
away. On the accuracy clause it is much better placed: 25 of 28 is 89% against a required nine in
ten. Adopted with the conflict recorded rather than resolved; `--model qwen2.5:7b` is one flag back.

**Not tried**: a newer model at the 7b's size, which would test whether the gain needs the latency.

### The system prompt is at a local optimum, and edits must be re-measured

The agent sometimes answers a Korean question in English. **Two attempts to fix that in `SYSTEM`
were withdrawn because both cost accuracy**, in different ways: a stronger rule in the first
paragraph raised right 18 → 19 while collapsing declines 8 → 3 of 9 and wrong-on-no-answer-screens
1 → 6; the same rule as a separate line held the declines but turned every miss into a wrong answer
(right 18 → 17, wrong 8 → 12). **Any imperative added pushed the model toward acting.** The baseline
reproduced exactly after each revert, three times. `python -m measure.choosing` is 50 seconds — run
it after touching a prompt.

### Nine ways this local-model harness has produced a confident wrong number

Every one was invisible from outside — the agent simply did nothing, or did something plausible.
**Three would have killed a direction that works and one would have kept a direction that does not.**

1. `format: "json"` silences the model completely — 366 ms, empty output.
2. Too small a `num_predict` truncates it mid-thought; `done_reason` says `length` and content is
   empty.
3. The answer is on `message.content`. Reading `response` (the other endpoint's field) returns
   nothing at all.
4. `num_ctx` defaults to 4096 and a 1440p screenshot plus a reply is about 4,096 tokens.
5. **Instructing the coordinate convention** starts a 13,000-character argument with itself. The
   convention is *observed*, not instructed.
6. Prefill measured warm is 25 ms against **0.9–5.5 s cold**. A warm-cache number was published and
   withdrawn.
7. `Reply.json()` was a non-greedy brace match, correct only while the reply contained no nested
   object — the moment the answer grew a list of things to do it stopped at the inner brace and
   **every action silently disappeared**. Pinned by `test_a_NESTED_reply_is_read_whole`.
8. A prompt that said "the exact name of one thing above" over lines reading `- button "Copy"` got
   `button "Copy"` back, and **five correct answers scored zero**.
9. A shuffle seeded with `hash()` of a string, which Python randomises per process, so the first
   analysis scored against an order the model was never shown.

## What is NOT built, so it is not mistaken for deferred work

No microphone, no speaker, no phone environment, no robot environment.

**A confirmed click was measured for the first time on 2026-09-09 and it found a defect in the
core.** Three goals on a window the harness launched: two did the expected thing and the agent chose
the right widget three times of three.

**My first diagnosis of the third was wrong and is withdrawn.** I wrote that `Widget.centre` had
landed on the inert part of the right control — a combo box being centred on its text field with the
arrow at the edge. Plausible and false. What happened is that **the name lost its role on the way
back**: Character Map has a `text` label reading `Font :` immediately before a `combo box` of the
same name, the model chose the combo box, and `Session.find()` returned the label because the label
comes first. The click landed on a piece of text, which does nothing — indistinguishable from a
click that missed, which is how the wrong explanation survived a round of work. `Effect` now carries
the role and `find()` matches both; two tests pin it, the second added because a mutation showed the
first was passing for a different reason.

**The app's own window was blocking its clicks, and now steps aside.** A synthetic click goes to
whatever window is topmost at that pixel and the app is always-on-top in a corner, so a widget
underneath it was unreachable. Measured: covered → nothing, stood aside → worked. The agent asks the
window to drop `-topmost` and lower itself before performing, on a `threading.Event` handshake
capped at a second.

**`WindowsHands` also gained UI Automation's control patterns** (`Invoke`, `ExpandCollapse`,
`Toggle`, `SelectionItem`) with a click as fallback. **What they are for is not task success** — 3
of 3 both ways on the goal set, and nothing either way on a covered widget, because `_element_for`
resolves *by point* and so inherits the obstruction problem it was meant to be immune to. **They are
for not stealing the mouse**: measured with the pointer parked, a pattern leaves it at (700, 400)
and a click drags it to the widget. **108 of 207 widgets on a live window resolve to a pattern
(52%; 68% excluding `text`)**, so that share of actions never touches the cursor — which on a
machine somebody is working at is the difference between an agent you leave running and one that
fights you for the pointer. **I recommended removing this after measuring only task success; that
recommendation is withdrawn.** `LegacyIAccessible.DoDefaultAction()` is **not** a universal
fallback: every control advertises it and it raised `COMError` where `Expand` worked.
`Local.release()` exists because a model left resident holds 5–6 GB, and the caller is expected to
call it.

**Decision still open** (`REBUILD.md`): nothing about the phone and robot environments beyond
"PC first, phone later, robot after" and "one core, an interface per environment".

## Where the detail lives

| file | what is in it |
|---|---|
| `REBUILD.md` | the rebuild plan — what was decided, why, and the staged path that was followed |
| `docs/findings/choosing-accuracy.md` | step 4: 18/28, the refuted index hypothesis, the shuffle control, name vs number, and the two harness faults found while measuring |
| `docs/findings/interaction.md` | the owner's priority, and the first measurements of repair, memory and language |
| `docs/findings/the-pc-app.md` | the window: self-blindness, the pointing measurement, the cold start, and what the accuracy number forces the interface to be |
| `docs/findings/what-the-old-tree-measured.md` | the removed library's transferable results, and the `git show` commands for the full record |

## Sub-agent team

Three project-scoped agents in `.claude/agents/`, checked in for the same reason `.githooks/` is —
the discipline belongs to the project, not to one machine. `parity-verifier` and `embedded-reviewer`
were removed with the C++ port they existed to check.

| agent | when to reach for it |
|---|---|
| `evidence-auditor` | before a commit stating a number; after editing this file; when a result looks good |
| `numerics-reviewer` | when a measurement needs designing so it can produce a negative result |
| `mutation-tester` | after writing a module and its tests; before claiming a suite protects something |

A sub-agent's conclusion is a claim about the repository, and claims get verified. **They are also a
dependency**: two were once launched, both terminated on a spend limit before reporting, and the
figures were measured by the same hand that wrote the code and presented as if reviewed. The numbers
were real; the independence was not.

## Working state

`origin` is the private GitHub repository `YounghyeonPark/argmind`; commits use `ypark.dev@gmail.com`.
