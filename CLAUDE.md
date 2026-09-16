# Argmind — project context

**A local-model agent you talk to while you work at your PC.** An open-source LLM on the machine,
real-time interaction, actions taken through the OS accessibility layer. Author: Younghyeon Park
<ypark.dev@gmail.com>. Apache-2.0. A side project, not a product.

Windows, PowerShell primary. **Rust, since 2026-09-15** — the Python is at
`python-before-removal-2026-09-15` and is not coming back. Needs [Ollama](https://ollama.com);
everything else it wants it offers to install. `cargo test --manifest-path native/Cargo.toml`
before quoting any number; a tracked hook in `.githooks/` runs it and blocks
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
Automation answers exactly in **41–266 ms**. `argmind-rules` holds this by reading the source — no
core module asks a model for `x`/`y`, nothing rendered into the prompt mentions a rectangle, and
nothing in `argmind-core` names a platform.

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

    native/argmind-core/        contract.rs  Widget, Eyes, Hands — what an environment must provide
                                llm.rs       the local model; every setting learned from a silent failure
                                intent.rs    Trigger, Effect, Intent
                                converse.rs  one call: what to say, and what if anything to do
                                session.rs   observe / decide / perform, separately
                                watch.rs     a trigger that starts from a change, with its floor
    native/argmind-env-windows/ UI Automation for eyes, SendInput for hands, WASAPI for the
                                microphone, SAPI for the voice, a tray icon, a cover window
    native/argmind-env-ollama/  the model, over raw HTTP/1.1
    native/argmind-env-whisper/ transcription, and the guard that stops silence becoming a turn
    native/argmind-voice/       Heard / Listener / Speaker — the seam, and why it is not a contract
    native/argmind-serve/       the application: a server, a page, a tray icon, and `argmind`
    native/argmind-measure/     every harness behind the numbers
    native/argmind-rules/       the rules this repository keeps, by looking at itself
    ui/                         the front end, TypeScript with no dependency but `tsc`
    measure/                    goals, captured screens, and every answer behind the numbers
    native/expectations/        what the Python produced, committed as data
    native/golden/              two recorded transcripts the suite replays

**Nothing needs a display, a model, a network, a microphone or a speaker.** Re-derive with
`cargo test --manifest-path native/Cargo.toml`.

**Three rules are held by reading the source rather than by the compiler**, in `argmind-rules`:
the licence record, the core naming no platform and being shown no coordinate, one place calling
`perform`, and a voice seam that cannot name a `Session`. Each was checked by breaking it.

### The invariants a change could break without noticing

**Acting is not a stage that always runs, it is something decided.** Four separable pieces, and a
thought can start from a `Watch` rather than from being spoken to.

**An effect carries a name and a role, never a rectangle**, re-resolved immediately before acting —
a listed widget is gone **13 times in 14** over the second a call takes. The role is there because
the name alone was not a key: a `text` label reading `Font :` in front of a `combo box` of the same
name took the click and did nothing. **And name-and-role is still not a key**: Character Map's
advanced view has more than one button named `Open`, which cost a goal in `clicking` — the agent
opened the character set list when the font list was asked for. The tie-break anchors on the *first*
same-named widget in the previous reading, so among copies it settles nothing, and the model cannot
name the one it means either.

**A watch carries a floor** (`MIN_GAP = 3.0`). Perception changes 13 seconds in 14; a call costs
0.3–2 s.

**History is bounded in TURNS, not lines** (`remember_turns = 12`). A turn costs 2.11 lines and the
old line-based slice landed **inside a turn in 10.7% of prompts** — 7.6% opening on a reply whose
question had been dropped. `remember_turns=0` means none and `None` means all; the old `remember=0`
silently meant *everything*.

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
| choosing | **21 of 28** answerable right, 7 of 9 declines — **28 of 37** correct decisions. The noisiest row here: 21, 23 and 24 across three runs of the same configuration |
| talking | **31 of 34** conversational turns produce no action (91%) |
| saying | an index leaks into the speech in **5 of 32**; the control set keeps its legitimate numbers |
| rule-following | a stated rule is obeyed in **36 of 36** probes at distance ≤ 8 |
| memory | a first-turn reference is truncation, not attention — **40 of 45 held with the anchor in the window, 1 of 45 without** |
| accepting | a yes one turn after an offer acts in **30 of 39**, up from 15 of 39 — **15 fixed, 0 broken, p = 0.00006**, pre-registered, both controls holding, 104 cases over four screens. A yes *five turns* later, **3 of 12 → 10 of 12**. What it cost is one control case that goes `landed → silent`: the reply word for word identical and the action slot empty |
| repair | **3 of 4** corrections land (n=4, so it cannot reach p<0.05 under any repair); a question about the past still re-proposes the action **9 times in 23** — the offer wording and then the standing-offer block each moved it, neither significantly (p = 0.0625 and 0.11) and the second only as a control; **Korean in, Korean out 30 of 36 (83%)** with **English in, English out 12 of 12** as the other half. `--think on` gets repair 4 of 4 and **halves Korean to 18 of 36** |
| a call | **0.61 s** median, mean 1.43 s, and the gap was one tail: **18 of 1,845 turns ran to the 4,096-token ceiling**, 69–71 s each, 47% of the model time in 1% of the turns. The tail is gone — the same sweep now has 0 at the ceiling and costs 1,461 s against 2,671 s — **but not because the stream stops cut it: neither stop fired in that run, and an earlier version of this row credited them anyway.** Replaying 128 recorded replies through the shipped stops gives `stop` 122, `degenerate` 3, `length` 3 and **`closed` 0**: on an ordinary turn this model emits nothing after its closing brace, so the answered stop is unreachable. The stops are insurance that has not been called |
| clicking | **2 of 3, n=1** on the Rust, against the Python's 3 of 3 at n=3 on the same goals and the same condition — one Character Map for all three, so the second goal meets a screen the first has changed. One run against three and this model reads 21, 23, 24 across runs of one configuration, so it is a number rather than a regression. **The miss is the interesting part and it is not the model's**: asked for the font list it named `Open`, and advanced view has more than one button called that — what opened was the character set list. `name` and `role` together are still not a key, and the model is shown nothing else |
| obstruction | **24 of 24 arms agreed with the prediction written before the run**, 3 runs of 8. With something covering the target, *neither* path reaches it — the pattern resolves by `ElementFromPoint` and gets the cover, then falls back to a click the reach guard refuses. Stood aside or uncovered, both work. So the one case that could have earned the control patterns does not |
| the pointer | **the axis the patterns do win, 3 of 3.** Parked at `(700, 400)`: `invoke` leaves it there, a click drags it to the widget at `(130, 459)`, and both press the box. **The first version of this harness measured nothing** — it parked with `cursor_round_trip`, which puts the cursor back where it found it, so every arm read the position the previous arm's click had left and both reported *unmoved* |

**Prompt edits here were 0 for 5.** Each raised a number and lowered another, and three were nearly
recorded as improvements; the only change that moved a line was a derived limit, `MOST_AT_ONCE = 3`.
**The sixth is the first sentence-level edit that paid**, and it is a *slot fed by conversation
state* rather than a sentence: an unanswered offer re-rendered beside the turn, present only when
there is one. It cost one control case, and **the reason written down for making it was withdrawn
by the run that confirmed it** — the placement was argued from prompt distance, and once the screen
and the widget count were separated the baseline read 3 of 10 at 12 lines, 7 of 10 at 21, 2 of 9 at
123 and 3 of 10 at 152. Non-monotone, with the shortest prompt among the worst. **The fix works and
its explanation does not.**

**The reply-shape order is a measured trade, not a default.** *Whichever field is written first is
written without the other.* `say` first carries the freshest thing the model has — the index — into
the speech; `do` first commits the action before anything is considered, and a rule lives in
language rather than in the action slot. Shipped is **`say` first**, the owner's choice: it costs the
leak (0 → 5 of 32) and choosing (32 → 28 of 37) and buys rule-following (14 → 36 of 36 probes).

**Three of the four harnesses send no history** — `saying`, `choosing` and `talking` all call
`converse()` without it. They reproduced their baselines digit for digit under a change that made the
agent click the eraser when greeted. **"No slide on those three" is vacuous for anything touching the
conversation**, and only `argmind-measure`'s `conversation` harness can see it.

## The owner's priority, 2026-09-10

> 약간의 실수는 괜찮음. 계속 고쳐나가게 하면 됨. 중요한건 상호작용 및 의사소통임.

Interaction and communication, not accuracy. A small drop in choosing is acceptable; a turn that
cannot be repaired is not.

## What is NOT built

No phone environment, no robot environment. **Voice is built and is off until switched on**: hold
F9 in the window, which means you have to be looking at argmind to talk to it. Speaking is free
(4–43 ms) and already Korean; hearing **transcribes silence as words 120 times out of 120**, so a
stray key press is a turn — it cannot act on one, but it will answer it aloud. **Two of the three
guards do reject**, with no threshold in either: a transcript of nothing but punctuation is not
something a person said, and a short cycle repeating inside one is Whisper looping (7 of 7
reproduced degenerations caught, 0 of 18 strings that had to survive). **The third needs a number
nobody has** — the only pace measured on this machine is a synthesiser's, 5.69 characters a second,
which is the wrong denominator for a person, so `pace_limit` ships as `None` and rejects nothing.
Whether a confirmed click achieves what the person *wanted* is unmeasured beyond n=3, **and that
n=3 was the Python's**. The agent still answers some Korean in
English. A question about the past makes it propose the action again 9 times in 23. **It can say the
right sentence and drop the action** — the `silent` shape, one case in eight, the reply identical to
one that also clicked. Internal widget indices leak
into the speech — *"3, 122"*, *"지우개는 22번 버튼입니다"* — and nothing measures whether a reply is
a sentence. Every turn is framed by a prompt about picking widgets even when the person says
*"고마워"*. **A standing rule is broken while it is visible in the prompt**, so that one is not a
memory problem.

**Streaming was measured and declined**: warm, the first token lands at 36 ms and `say` completes at
232 ms of a 478 ms reply, so it buys ~250 ms at the cost of extracting a string from partial JSON.

**The robot is off the target list, and not for a reason a rewrite fixes.** It has no deterministic
source of WHERE — anything filling that slot is a *detector*, and the core would receive
probabilistic output through an interface whose contract is that it is exact. Since refusing spoken
confirmation it has none of WHETHER either: its only confirmation channel is the speech that was
refused. It wants a different core. A phone fits honestly; `uiautomator` has the same shape as UIA.

## The core moves to Rust — 2026-09-13, the owner's decision

> Python은 규모가 커질수록 한계가 드러남. 다른언어로 구현하는게 맞음.

**The argument is scale, and the latency measurements do not answer it** — they say a turn spends
0.032 ms in all of `converse()` against 1,019 ms of waiting, and that the FFI seam a port adds
costs *more* than the work it removes (0.144 ms to carry 152 widgets across as JSON). Both are
about speed, which is not what was claimed. What a native core does buy, in this project's own
terms: **rules currently held by AST tests become held by the compiler** — `converse()` already
never reads `.rect`, so a `Listing { name, role }` type makes "the model is never shown a
coordinate" structural rather than asserted.

**That payoff has not arrived and the port is finished.** `Listing` was never built; the core
still passes `Widget`, which carries a `rect`, and the rule is held by a test reading
`converse.rs`. What the port *did* buy is elsewhere and is real: `Reach` makes *never touch what
this harness did not start* a type, `Proposals` makes a stale button structurally unable to
confirm a later action, and a job object makes *the server dies with argmind* the operating
system's promise rather than a destructor's. Written down here because a rewrite argued from a
benefit that did not materialise should say so.

Staged so each step was verifiable, and **the instruments came first**: `measure/` was 7,489
lines against the core's 575, and it was the asset. **Done on 2026-09-15**, when the Python was
deleted at `python-before-removal-2026-09-15`. Every module has a successor, every harness has a
successor, and what the Python produced is committed as data — `native/expectations/*.json` and
`native/golden/` — so the equivalence checks outlive the implementation they were taken from.

**Deleting it found three things the port had missed**, all of them by asking what would be lost
rather than by assuming nothing would: the licence rule and the core's blindness were held only
by Python that was about to go; the transcript guard that catches Whisper's degenerate loops — 7
of 7 reproduced — had no Rust successor at all and `hear()` was returning whatever came off the
socket; and *the room stays quiet unless somebody asked* was a comment in two places rather than
a fact. `docs/findings/porting-the-core.md` has the accounting.

## Where the detail lives

| file | what is in it |
|---|---|
| `REBUILD.md` | the rebuild plan, the decisions and the staged path that was followed |
| `docs/findings/choosing-accuracy.md` | every choosing measurement: models, prompts, the shuffle control, the refuted hypotheses |
| `docs/findings/interaction.md` | the priority, and repair / memory / language / spurious action / the history window |
| `docs/findings/the-pc-app.md` | the window: self-blindness, pointing, cold start, the withdrawn diagnosis |
| `docs/findings/a-better-voice.md` | **the voice**: every open Korean TTS checked and its licence, what builds here, and the measured RTF 3.9 that rules out the CPU |
| `docs/findings/harness-pitfalls.md` | **ten ways the model harness lied and seven ways the measuring code did** — the list to check when a measurement reads zero |
| `docs/findings/porting-the-core.md` | **the move to Rust**: what speed does not decide, the robot answered, the AST rules audited one at a time, what breaks in `measure/`, and the staging |
| `native/argmind-voice/src/lib.rs` | voice is *interface*, not a contract — and why *"응"* out loud cannot run an action |
| `native/golden/loop-transcript.txt` | the behavioural record of the Python `loop.py`, kept as evidence; its subject is now covered by `expectations/loop.json` |
| `docs/findings/what-the-old-tree-measured.md` | the removed library's transferable results, and how to reach the rest |

## Sub-agent team

**Five agents in `.claude/agents/`, all of them build**, each owning a part of the tree with no gaps
and no overlaps. **Architecture leads**: `architect` drew the boundaries and is the only one that
may change a contract.

| agent | owns |
|---|---|
| `architect` | the shape — `contract.rs`, `session.rs`, `llm.rs`, the rules in `argmind-rules`, where a new thing goes |
| `conversation` | `converse.rs`, `intent.rs` — what it says, how it reads a turn, repair |
| `environment` | `native/argmind-env-*` — capabilities behind Eyes and Hands |
| `app` | `argmind-serve` and `ui/` |
| `measurement` | `argmind-measure` and `measure/` — the instruments every decision rests on |

The checking moved **inside** rather than away: `conversation` runs both harnesses on every change,
`environment` owes each capability its measurement, `architect` must find the reading where the
contract does not change. Mutation testing is deferred, not judged. **The cost is real** — a
standard carried by the person doing the work is weaker than one carried by somebody else, and this
project's history says so. `.claude/agents/README.md` has the reasoning and two earlier versions of
the list that were wrong.

A sub-agent's conclusion is a claim about the repository, and claims get verified.
