# Argmind — project context

**A local-model agent you talk to while you work at your PC.** An open-source LLM on the machine,
real-time interaction, actions taken through the OS accessibility layer. Author: Younghyeon Park
<ypark.dev@gmail.com>. Apache-2.0. A side project, not a product.

Windows, PowerShell primary, Python 3.14. Needs [Ollama](https://ollama.com) and `comtypes`.
`python -m pytest` before quoting any number; a tracked hook in `.githooks/` runs it and blocks
(`git config core.hooksPath .githooks` once per clone). `origin` is the private
`YounghyeonPark/argmind`. **Location `C:\dev\argmind`** — off Google Drive since 2026-08-08, because
sync corrupts a git repository.

**Until 2026-09-09 this was a recursive-least-squares library ported to Cortex-M**, about 50,000
lines, removed rather than shelved. Intact at `before-removal-2026-09-09` and `argmind-v1`. **Do not
restore it.** `docs/findings/what-the-old-tree-measured.md` keeps the results that still bear on
work here.

## The rule everything hangs on

> **The deterministic side says WHERE. The probabilistic side says WHAT.
> A model never emits a coordinate.**

Measured first: a vision model took **24–82 s** for a pixel and on a dense screen never answered; UI
Automation answers exactly in **41–266 ms**. `tests/test_core.py` holds this by walking the AST — no
core module asks a model for `x`/`y`, none hands it an image, nothing in `core/` names a platform.

## Standards of evidence

Carried across two rewrites. They exist because a previous project accumulated seventeen modelling
errors and **most of them flattered the design** — and because four conclusions were withdrawn in
the two days that built this, three of them flattering.

- **Measured, computed or estimated — labelled, in the same sentence as the number.**
- **A model that cannot produce a negative result is not evidence.** Protect a negative result with
  a test.
- **A flattering result means find the responsible assumption first.**
- **A uniformly tidy result is evidence the measurement is broken.**
- **Scrutinise corrections that favour you as hard as ones that do not.**
- **A rate carries its denominator and its conditions.** "82%" is not a number.
- **Choosing and talking are one measurement in two halves.** Citing one is incomplete by
  construction.

## What is built

    argmind/core/    contract.py  Widget, Eyes, Hands — what an environment must provide
                     llm.py       the local model; every setting learned from a silent failure
                     intent.py    Trigger, Effect, Intent, Watch
                     converse.py  one call: what to say, and what if anything to do
                     loop.py      observe / poll / decide / perform, separately
    argmind/env/     windows.py   UI Automation for eyes, SendInput for hands
    argmind/app.py                the window — `python -m argmind app`
    argmind/__main__.py           app / look / talk / say / watch
    measure/                      goals, captured screens, and every answer behind the numbers

**55 tests, none needing a display, a model or a network.** Re-derive with `python -m pytest`.

### The invariants a change could break without noticing

**Acting is not a stage that always runs, it is something decided.** Four separable pieces, and a
thought can start from a `Watch` rather than from being spoken to.

**An effect carries a name and a role, never a rectangle**, re-resolved immediately before acting —
a listed widget is gone **13 times in 14** over the second a call takes. The role is there because
the name alone was not a key: a `text` label reading `Font :` in front of a `combo box` of the same
name took the click and did nothing.

**A watch carries a floor** (`MIN_GAP = 3.0`). Perception changes 13 seconds in 14; a call costs
0.3–2 s.

**`MOST_AT_ONCE = 3`, derived.** 27 of 28 real requests want one effect or two; spurious ones come
in threes, fours and an eight. Three is the documented sequence — click, type, enter.

**An unprompted effect without standing permission is downgraded to speech.** The app inverts the
flag deliberately and that is *stronger* — a person confirms each action, and one AST test keeps
`perform` reachable from exactly one place there.

**One call, not two.** A round trip is the budget: 0.45 s warm, 4.7 s and 5.9 s for the first two.

## The numbers, as they stand

`qwen3:8b`, thinking off, three repeats, worst counted:

| | |
|---|---|
| choosing | **23 of 28** answerable right, 7 of 9 correct declines — **30 of 37** correct decisions |
| talking | **29 of 34** conversational turns produce no action (85%) |
| repair | 3 of 4 corrections land; memory of its own actions 0 of 1; Korean in, Korean out 5 of 6 |
| a call | **0.48 s** warm. `--think` reaches 34 of 37 and 94%, at 3.75 s |
| clicking | 3 of 3 end to end on a launched window, n=3 |

**Prompt edits here are 0 for 5** — each raised a number and lowered another, and three were nearly
recorded as improvements. The one change that moved the line was a derived limit, not a sentence.

## The owner's priority, 2026-09-10

> 약간의 실수는 괜찮음. 계속 고쳐나가게 하면 됨. 중요한건 상호작용 및 의사소통임.

Interaction and communication, not accuracy. A small drop in choosing is acceptable; a turn that
cannot be repaired is not.

## What is NOT built

No microphone, no speaker, no phone environment, no robot environment. Whether a confirmed click
achieves what the person *wanted* is unmeasured beyond n=3. The agent still answers some Korean in
English. A question about the past makes it propose the action again. No streaming; every turn is
framed by a prompt about picking widgets even when the person says *"고마워"*; nothing is measured
about a conversation longer than three turns.

**The open architectural question**: a robot has no deterministic source of WHERE. Anything filling
that slot is a *detector*, and the core would then receive probabilistic output through an interface
whose contract is that it is exact — the rule inverted at the seam. A phone fits honestly;
`uiautomator` has the same shape as UIA.

## Where the detail lives

| file | what is in it |
|---|---|
| `REBUILD.md` | the rebuild plan, the decisions and the staged path that was followed |
| `docs/findings/choosing-accuracy.md` | every choosing measurement: models, prompts, the shuffle control, the refuted hypotheses |
| `docs/findings/interaction.md` | the priority, and repair / memory / language / spurious action |
| `docs/findings/the-pc-app.md` | the window: self-blindness, pointing, cold start, the withdrawn diagnosis |
| `docs/findings/harness-pitfalls.md` | **ten ways the model harness lied and seven ways the measuring code did** — the list to check when a measurement reads zero |
| `docs/findings/what-the-old-tree-measured.md` | the removed library's transferable results, and how to reach the rest |

## Sub-agent team

**Five agents in `.claude/agents/`, all of them build**, each owning a part of the tree with no gaps
and no overlaps. **Architecture leads**: `architect` drew the boundaries and is the only one that
may change a contract.

| agent | owns |
|---|---|
| `architect` | the shape — `contract.py`, `loop.py`, `llm.py`, the AST rules, where a new thing goes |
| `conversation` | `converse.py`, `intent.py` — what it says, how it reads a turn, repair |
| `environment` | `argmind/env/` — capabilities behind Eyes and Hands |
| `app` | `argmind/app.py` and its tests |
| `measurement` | `measure/` — the instruments every decision rests on |

The checking moved **inside** rather than away: `conversation` runs both harnesses on every change,
`environment` owes each capability its measurement, `architect` must find the reading where the
contract does not change. Mutation testing is deferred, not judged. **The cost is real** — a
standard carried by the person doing the work is weaker than one carried by somebody else, and this
project's history says so. `.claude/agents/README.md` has the reasoning and two earlier versions of
the list that were wrong.

A sub-agent's conclusion is a claim about the repository, and claims get verified.
