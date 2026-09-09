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
the right widget three times of three. The failure was **`Widget.centre` landing on the inert part of
the right control** — a combo box's rectangle is centred on its text field, and the arrow that opens
the list is at the edge, so the click was exact and did nothing. That is sharper than the 73%
hit-test figure, which counts centres landing on a *different* element; here the element was right.
**The repair that follows is UI Automation's own control patterns** (`Invoke`, `ExpandCollapse`,
`Toggle`) with a synthetic click as fallback and a count of which was used — the deterministic side
doing the acting, which is the same argument that put WHERE on the environment. **Not built, and it
must be measured rather than assumed better**: an invoke may not set focus where a click does.
`Local.release()` exists because a model left resident holds 5–6 GB, and the caller is expected to
call it.

**Decision still open** (`REBUILD.md`): nothing about the phone and robot environments beyond
"PC first, phone later, robot after" and "one core, an interface per environment".

## Where the detail lives

| file | what is in it |
|---|---|
| `REBUILD.md` | the rebuild plan — what was decided, why, and the staged path that was followed |
| `docs/findings/choosing-accuracy.md` | step 4: 18/28, the refuted index hypothesis, the shuffle control, name vs number, and the two harness faults found while measuring |
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
