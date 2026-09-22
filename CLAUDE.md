# Argmind — project context

**A local-model agent you talk to while you work at your PC.** An open-source LLM on the machine,
real-time interaction, actions taken through the OS accessibility layer. Author: Younghyeon Park
<ypark.dev@gmail.com>. Apache-2.0. A side project, not a product.

**`argmind-app.exe` with no arguments is everything on**, since 2026-09-17: what it does is read
out of `%LOCALAPPDATA%\argmind\` — `decisions.json` for hearing, watching and speaking aloud,
`devices.json` for which microphone and speaker, `mind.json` for the dials and the goal. The flags
that remain are `--dry` (press nothing), a model name with a colon, and `--speak`/`--hear` as the
way to say yes when there is no standing answer to read.

Windows, PowerShell primary. **Rust, since 2026-09-15** — the Python is at
`python-before-removal-2026-09-15` and is not coming back. **`rust-toolchain.toml` pins
`channel = "stable"`**, which rustup resolves against the *host* — MSVC on Windows, where
`webview2-com-sys` links the WebView2 loader statically and `argmind-app.exe` needs no file beside
it. A `-gnu` build of it exits on a double-click with `0xC0000135` and no message;
`argmind-serve/tests/launchable.rs` is the rule that catches that. Needs [Ollama](https://ollama.com);
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
                                mind.rs      the dials and the goal — a slot that is absent at rest
                                thinking.rs  a note it writes for itself, revised on a timer
                                converse.rs  one call: what to say, and what if anything to do
                                session.rs   observe / decide / perform, separately
                                watch.rs     a trigger that starts from a change, with its floor
    native/argmind-env-windows/ UI Automation for eyes, SendInput for hands, WASAPI for the
                                microphone, SAPI for the voice, a cover window
    native/argmind-env-ollama/  the model, over raw HTTP/1.1
    native/argmind-env-whisper/ transcription, and the guard that stops silence becoming a turn
    native/argmind-env-qwentts/ the neural voice, over raw HTTP/1.1, streamed
    native/argmind-voice/       Heard / Listener / Speaker — the seam, and why it is not a contract
    native/argmind-serve/       the application: a server, a page, a tray icon, and `argmind`
                                senses.rs    the blackboard: a slot per sense with its age, and
                                             the one bit that makes them stand down for a turn
    native/argmind-measure/     every harness behind the numbers, and `--example raw_turn`
                                for looking at one reply whole
    native/argmind-env-linux/   eyes for Linux: AT-SPI2 over D-Bus. No hands yet
    native/argmind-env-none/    an environment that can do nothing and says so — the proof the
                                application is separable, and the list a third one must satisfy
    native/argmind-rules/       the rules this repository keeps, by looking at itself
    native/argmind-window/      the window the page lives in, and the icon in the tray
    ui/                         the front end, TypeScript with no dependency but `tsc`
    measure/                    goals, captured screens, and every answer behind the numbers
    native/expectations/        what the Python produced, committed as data
    native/golden/              two recorded transcripts the suite replays

**Nothing needs a display, a model, a network, a microphone or a speaker.** Re-derive with
`cargo test --manifest-path native/Cargo.toml`.

**Four rules are held by reading the source rather than by the compiler**, in `argmind-rules`:
the licence record, the core naming no platform and being shown no coordinate, one place calling
`perform`, a voice seam that cannot name a `Session`, and — since 2026-09-16 — **the application
naming an operating system in one file.** Each was checked by breaking it.

**One of them has an exception now, and the exception is written against what may be said.** *The
room stays quiet unless somebody asked* allowed exactly one `say` in `app.rs`; choosing a voice
means hearing it, so there is a second, and it may say only the one fixed constant `A_SAMPLE`. A
`say(&note.believes)` added to `notice` still fails. Two of the four breaks used to check it would
have walked through the **old** rule, which matched the literal text `voice.say(` — a rule about a
local variable's name.

**And one rule is held by the compiler.** `cargo test -p argmind-serve --features headless` builds
the whole application against `argmind-env-none` and runs the server against it. A `use
argmind_env_windows::` added to `app.rs` compiles fine on this machine and quietly un-ports the
program; that build is what catches it, and the pre-commit hook runs it.

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
0.3–2 s. **Watching is on by default since 2026-09-17** — the owner's decision, the cost
unchanged and now printed under the switch in 설정 rather than asked about. `notice` is the one key
in `ALLOWED_UNLESS_REFUSED`; a default is not an answer, so nothing is written to `decisions.json`
and 거부 still wins.

**A thought outlives the turn that produced it, and only just.** `thinking.rs` holds three fields —
믿는 것 (a sentence) / 할 만한 것 (**a widget the model picks by number**, never prose) / 이미 한 것 (the record) — revised every 30 s by **a call whose answer shape has no `say`
and no `do` in it**, which is what makes a timer safe. It runs only while watching. **The model is
not asked what it did**: asked once, it invented having tidied somebody's browser tabs and carried
it for three revisions, so `tried` is read off `remember_doing`'s own record instead. A blank note
renders nothing and the six probes still hash to `r322a4c`.

**History is bounded in TURNS, not lines** (`remember_turns = 12`). A turn costs 2.11 lines and the
old line-based slice landed **inside a turn in 10.7% of prompts** — 7.6% opening on a reply whose
question had been dropped. `remember_turns=0` means none and `None` means all; the old `remember=0`
silently meant *everything*.

**`MOST_AT_ONCE = 3`, derived.** 27 of 28 real requests want one effect or two; spurious ones come
in threes, fours and an eight. Three is the documented sequence — click, type, enter.

**An unprompted effect without standing permission is downgraded to speech**, and one AST test
keeps `perform` reachable from exactly one place in the app. **The app had not turned the flag on,
and `session.rs` says it should have** — *an interface with a confirm button turns this on and gates
with the button instead, which is stronger than the flag.* This one has the button and left the flag
off, so `perform` refused every effect a watch or a raised judgement produced and the 실행 beside it
did nothing at all; it was drawn, it could be pressed, and it reported `ran`. Fixed 2026-09-22: the
flag is on, set in one place, **and the automatic path refuses what the button now allows** — 확인
없이 실행 was given in advance to *the thing I just asked for*, so a turn argmind began itself still
waits for a person however that switch is set. **So argmind can now act on its own idea, and only
with a press.** **The other half of that sentence went on 2026-09-22**: *a person confirms each
action* is now a setting. 확인 없이 실행 in 설정, off by default, off written down when it is turned
off — and the owner's, asked for after watching three buttons pile up in one transcript. **The two
gates are different questions and only one moved**: *may argmind act without being asked* is in the
core, is still shut, and is what stops a watch pressing anything; *may it act without being
confirmed* is the new one and covers only turns a person started. With it on, a proposal still goes
through `Proposals` and gets a token the same way — and **the second opinion becomes a gate.** While
somebody reads the row, *그 지점에는 link Cargo.lock* is information and the button is the decision;
with nobody reading it, it is the last thing between a wrong resolution and a click, so a proposal
carrying one still waits. It stays a note otherwise, because the hit test disagrees with a correct
pixel about three times in ten.

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
| the note | **it invents what it did, if you let it.** Asked for all three fields from a session in which argmind had done nothing: *이미 메모리 사용량이 높은 탭들을 목록으로 정리해두고…* — it had not, and it carried that for three revisions. `tried` is now a record rather than a question and the re-run leaves it empty for 100 s, correctly. It also **anchored** (a file manager on `C:` still read as *exploring files related to Younghyeon Park*) and **wrote in English**; both addressed, both re-run. **`waiting` is still generic** and nothing measures whether one is ever met |
| the watch was reading argmind | **4 of 8 notices over 150 s said `the window changed to 'argmind'`** — a model call each, spent on its own transcript. `attending.rs` had written the prediction and left it: *nothing here fires on a watch yet; when something does, this is the thing to revisit.* The old rule excluded whatever was in front, so working in Chrome after coming from argmind read argmind. Now `app.py`'s original: **the newest window that is not ours**, available again since the page moved into a window this process owns. **Predicted zero before the change; the re-run gave zero** |
| the dials | **the slot's language decides the reply's**, and that is the whole finding. 44 Korean requests: no line 28 of 44 answered in Korean, a content-free **English** sentence in the slot **19**, a content-free **Korean** one **42** and 41 on a repeat. The four dials in English sat at 28–31 — they moved 20–40 of 44 sentences and left the language alone. Korean lines ship, and at x3 the worst arm reads **Korean 39 of 44 against 28, leak 7 of 32 against 5**: +11 bought, 2 discordant cases spent, paid only by somebody who turns a dial. **The first reading was that the dials fixed the leak (5 of 32 → 0 or 1 in all four arms); the placebo fixed it too** |
| parroting | **it says your own words back when it cannot parse them.** 머해 → `머해?`, then the mirror is in the history and the next two turns get it again. Reproduced **at rest** — 2 of 6 echoed and 2 stuck with every dial off — so it is not the dials, and `{"say": "머해?"}` is what comes off the wire, so it is not the reader. `뭐해` is answered properly; `머해` is the colloquial spelling of it. **Asking it not to does not work**: told 상대가 한 말을 그대로 되돌려 말하지 말 것 it replied `머해`. Caught on the way out — *their words plus a question mark*, never plain equality, because `안녕` is a good answer to `안녕` — and **the substitute goes into the history**, which is the half that closed the loop: 2 stuck → 0 |
| senses beside the conversation | **a look costs the answer about 3x**, reproduced: median 1.04 -> 3.19 s and 0.97 -> 2.76 s over two runs of ten, worst 10.81 s and 8.87 s — one answer in ten waits out most of a whole look. And only **1 and 2 looks finished** while ten answers ran, where taking turns would have fitted three, so the two are not sharing the card but obstructing each other. **Parallel structure, serial GPU**: a sense on its own thread so it cannot freeze the session, a timestamp on every reading, and a sense that stands down while a turn is in flight |
| abandoning a call | **dropping the socket frees the card.** Three arms, four rounds: an answer after a look finished **1.12 s**, after one was dropped 2 s in **1.09 s**, while one runs **3.22 s** — abandoning lands on a free card in every round, and all four looks came back `reason = "abandoned"` so the drop fired rather than the look finishing first. `Local::ask_while` asks `carry_on` before every line. **So the third rule is buildable in full**: decline to start a sense during a turn, *and* abandon one already running — the 10.81 s tail is reachable, not just the median |
| the blackboard, live | **rule 2 works completely**: driving the app with `/say` for 60 s at a time with two ticks due, **zero ticks completed**, every arm. **Rule 3 fires**: of six deliberate collisions two landed inside a tick and both were abandoned with no note produced. **The latency claim is not established** — the first attempt read a median of 2.57 s and looked like failure, and the control showed it was the screen: the prompt carries up to 200 widgets and its size follows whatever window is in front. With widget counts recorded, watching on reads 0.65 s against 0.83 and 0.84 off, at a matched median of 12 widgets. **The negative holds — watching does not cost the conversation — and nothing stronger does** |
| judging | **it narrated and waited, and three tries to fix it.** The owner: 생각이 그냥 기다리는중이면 의미가 없음. Renaming the field gave **58% full, 5 of them the idle shape** and most of the rest *도움을 줄 수 있다*. Adding `what_changed` and banning the filler gave **0%, including on a dialog appearing** — the ban read as *say nothing*. An example instead of a ban gives: full on a built *저장하지 않은 변경 내용* dialog, empty on a window that did not move, **0 of 12–24 on the captured screens**, reproduced. **Better, not clean**: the answer is grounded in the event and still ends *도움을 줄 수 있다*, and n is 2 on the discriminator. Both failure conditions were written before the runs, which is the only reason 0% read as a failure |
| the judgement became a choice | **prose failed four times, so `worth` is a widget chosen by number.** 58% full with five idle shapes, then 0% even on a dialog, then a copy of the example — all reverted. As a number out of the list it is already looking at (`choosing`: 28 of 37, 7 of 9 declines) it picked one on **12 of 12** ticks; giving the decline a cost the way `choosing` words it took that to 75%; and **6 of 8 unchanged ticks were breaking the prompt rule — 달라진 것이 없으면 null**, so the gate moved into `read_note`. **33%, and 25% on a repeat, 0 idle, both arms of the discriminator passing.** But the decline is the program's: on ticks where something changed it filled **5 of 5, twice** — it never declines when allowed to |
| a reason to start | **the one turn that begins from argmind own state.** A judgement that has stood three looks with nobody acting on it is raised by argmind itself. The gate added the day before made that impossible — it dropped a standing judgement on any quiet look, so stuck could never reach 2 — and judging was not recording stuck at all, so the runs that passed said nothing about it. A tick now concludes one of three things: nothing happened so nothing was judged and what stands stands; something happened and there is nothing to do; something happened and here is the widget. Measured on a built sequence where a dialog appears and then nothing changes: stuck 1, 2, **3 and raised**, then put down and not raised again — reproduced tick for tick. RAISE_AFTER = 3 is chosen, not measured |
| clicking | **2 of 3, n=1** on the Rust, against the Python's 3 of 3 at n=3 on the same goals and the same condition — one Character Map for all three, so the second goal meets a screen the first has changed. One run against three and this model reads 21, 23, 24 across runs of one configuration, so it is a number rather than a regression. **The miss is the interesting part and it is not the model's**: asked for the font list it named `Open`, and advanced view has more than one button called that — what opened was the character set list. `name` and `role` together are still not a key, and the model is shown nothing else |
| obstruction | **24 of 24 arms agreed with the prediction written before the run**, 3 runs of 8. With something covering the target, *neither* path reaches it — the pattern resolves by `ElementFromPoint` and gets the cover, then falls back to a click the reach guard refuses. Stood aside or uncovered, both work. So the one case that could have earned the control patterns does not |
| which voice | **a seed fixes a recording; it took a second measurement to notice that is not a person.** The talker is a *base* model, so it ships with no speakers — `/v1/audio/voices` starts empty and every built-in name is `unknown voice`, `default` included — and the timbre is whatever the sampler drew, `seed` defaulting to `-1`, the server's word for *choose for me*. Sending one fixes **one sentence** exactly: unseeded 4 of 4 different over a 21% spread in length, `seed = 7` **1 of 4** byte-identical, on the streamed path too, with four distinct seeds as the control. **And across four *different* sentences a seed held to 119 Hz of itself while different seeds on one sentence spanned 137 Hz** — 1.2x, no ratio at all — with long-term average spectra no more alike within a seed (0.821) than across seeds (0.813) and the same-seed-same-text control at exactly 1.000. Every reply is a different sentence, so the first day's fix left the complaint standing. **What holds a person is a cloned voice**: one reference drawn at a fixed seed, registered through `POST /v1/audio/voices`, and the clone then tracks its reference's pitch at **+0.99** and moves **30 Hz across five sentences against the seed's 121**. 0.69 s to make, memory-only, so it happens every launch |
| five voices and not seven | **the pitch rule was the wrong question and it still did its job.** Seventy-two seeds cloned and measured. Neighbours further apart in pitch than they wobble gave seven — and then the wobble turned out to be an n=3 artefact (4–19 Hz over three sentences, 16–33 over six) and the rule turned out to ask the wrong thing: **two voices can share a pitch and be two people, and two can differ in pitch and be one.** Long-term average spectra, which have no pitch in them, say a clone is **0.955–0.982** like itself across sentences; two of the seven pairs measured **+0.016** and **−0.004** against that and were one person under two names. Five left, weakest pair +0.042. **The first shipped list failed its own test within a minute** — `low` at 120 Hz and `lowmid` at 136 with a wobble of 17 — which is the only reason the wrong question was noticed |
| tuning a voice | **there is nothing to tune, and three different kinds of nothing.** `speed` is accepted and does nothing — five values from 0.5 to 2.0 gave one byte-identical file, and the server's own header says so: `float speed; // OAI speed, parsed then ignored (no time stretch in the ABI)`. `instructions` is refused outright: `--instruct is not supported for base models`. And the sampling knobs — `temperature`, `top_p`, `top_k`, `repetition_penalty`, `subtalker_*` — change the bytes and leave the speaker where it was, f0 moving 264→270 Hz at a fixed seed, which is noise. So the only adjustable thing is *which voice*, and no dial was shipped rather than five that move nothing |
| animal noises | **it can say 멍멍; it cannot bark.** Nine candidates in the shipped cloned voice — 멍멍, 야옹, 음메, 꼬끼오, 짹짹, 어흥, 으르렁, `Woof!`, `Meow~` — measure **92–100% voiced** at a spectral flatness of **0.031–0.069**, against ordinary speech in the same voice at 99–94% and 0.054–0.057. A bark is short and broadband; these are sustained and harmonic, which is what a person reading a syllable is. The one case that looked different, English `Woof!` at 71% voiced and 0.330 flat, was an unseeded draw and went back to 100% and 0.031 once it was the actual voice. **And `<dog barking>` ran to the token ceiling: 163.84 seconds of audio** from three words |
| starting up | **it reinstalled both servers on every launch, and two correct decisions made it.** They die with argmind — a job object, on purpose — and `present` for a server means *answering on its port*, so every launch finds them missing and hands each to its **build** fix. The offer already knew better: `title()` reads 띄울까요? when `built` is true and 빌드할까요? when it is not, so the question distinguished the two cases and the carry-out did not. Cold launch, nothing changed on the machine: **21.8 s → 7.5 s** to a voice and 8.4 → 4.8 to the recogniser, with `cmake` (4.0 s), `mingw32-make`, a `git` fetch and `glslc` regenerating Vulkan shaders (4.7 s) all gone. **What makes skipping safe was already there for another reason** — `remember_build` is written last, so a stamp means the clone, the build and the runtime all finished; that is now asserted rather than relied on |
| the pointer | **the axis the patterns do win, 3 of 3 — and they are on now.** Parked at `(700, 400)`: `invoke` leaves it there, a click drags it to the widget at `(130, 459)`, and both press the box. **The first version of this harness measured nothing** — it parked with `cursor_round_trip`, which puts the cursor back where it found it, so every arm read the position the previous arm's click had left and both reported *unmoved* |

**The one slot that is absent unless somebody asks for it.** `mind.rs` renders nothing at rest, so
the six probes still hash to `r322a4c` at 5,676 bytes — a tag recorded from the Python before a dial
existed. A prompt edit nobody has turned on cannot cost anything, which is the only ground on which
one more was allowed. `docs/findings/what-it-is-trying-to-be.md`.

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

**It runs on Linux, and there is still no Linux environment** — which sounds contradictory and is
the exact state of things. On Ubuntu 26.04 under WSLg, `argmind-app` builds and opens its own
window, WebKitGTK renders the page, and the server answers in 200 ms; the header reads
*(아무 창도 보고 있지 않음)* because `argmind-env-none` is telling the truth about having no eyes.
Running it found four things a clean cross-compile could not: the tray **panicked** and took the
application down when `libayatana-appindicator3` was absent (fixed — the window outlives a tray
that cannot be made), every Korean character was a box without `fonts-noto-cjk`, the needs list
still offers `DISM` commands on Linux, and `org.a11y.Bus` is activatable there — so AT-SPI2 work
can be done in WSL, which was the open question.

**Linux has eyes.** `argmind-env-linux` reads the active window through AT-SPI2 — 146 widgets of
`gtk3-widget-factory` in **116 ms**, against UI Automation's 41–266 ms, with real names, roles and
screen rectangles. **The contract survives the port**: `Widget { name, role, rect, window }` is
exactly what the accessibility bus hands over. `docs/findings/eyes-on-linux.md` has it.

**And hands, for clicking.** Through `org.a11y.atspi.Action` — the same choice Windows makes with
UI Automation patterns, for the same measured reason: a pattern leaves the cursor where the person
left it. Verified against the widget's own state word, `0x43000900 -> 0x43000910`, which is
`CHECKED` flipping. No new dependency: the eyes' D-Bus connection is all it takes.

**And typing**, through `EditableText.InsertText` — also without a keystroke, so it works the same
on Wayland as on X11 and asks nobody for anything. Verified by reading `안녕하세요` back out of the
widget. **Its `length` is bytes, not characters**: passing the character count put one Korean
letter in and stopped, while ASCII agreed either way — the shape of bug that passes every English
test and breaks the whole interface.

**Only named keys are refused** now. `Enter` does what a form decides and there is no honest way to
spell it here; that needs XTEST on X11 or `xdg-desktop-portal` on Wayland, and on Wayland **the
system holds the gate, not argmind**: the portal shows its own dialog, so `항상 허용` cannot be honoured from argmind's
settings there. Two gates, and the settings panel only controls one. Not decided from a WSLg
session with no window manager of its own — which is also why `ACTIVE` is unreliable there, and
why *which window is in front* has no counterpart that can be tested here.

**No macOS environment at all**, and that is most of what is missing — the application
itself is no longer in the way. The owner asked on 2026-09-16 for all three; `docs/findings/the-window.md`
has the plan and the measurement that reordered it. The short version: the port looks like
`argmind-env-windows` (1,844 lines) and was actually `argmind-serve` (2,295), which named Windows
types in six files. **That is fixed and checked** — `platform.rs` chooses, `argmind-env-none` is
what it chooses when there is nothing better, and the application builds and runs against it.

**Six crates and the whole application already compile for Linux** — `cargo check --target
x86_64-unknown-linux-gnu -p argmind-serve --no-default-features`, which the pre-commit hook runs.
That is stronger than the headless build, because it takes the other branch of every `cfg`
including the manifest's. The window is what stops there: on Linux it wants `pkg-config`,
`libdbus-1-dev`, `libgtk-3-dev`, `libwebkit2gtk-4.1-dev`, `libsoup-3.0-dev` and
`libjavascriptcoregtk-4.1-dev`, read off the build failure rather than guessed.

**The content moved behind the seam too, on 2026-09-17**: `platform::can_carry_out_fixes()` gates
every fix, because every one of them has been run on Windows and none anywhere else. Off Windows
each need still says what is missing and what to do by hand and offers no button — and the two rows
that are Windows *facts* rather than Windows *fixes*, the SAPI voice and the `winget` Vulkan SDK,
are not listed at all. Verified on Ubuntu: **없는 것 4개**, no DISM, no buttons, and the orb
rendering in WebKitGTK.

**What remains is a real environment for each, and it needs a machine.** Writing one that cannot be
run produces code that compiles and does not work. Three things are known before that starts:
**Linux is two targets** — X11 allows XTEST, Wayland routes input and window reading through
`xdg-desktop-portal` with the user's consent, which changes the permission model rather than the
code; **macOS needs one grant** in System Settings before the Accessibility API answers, which fits
`Fix::OpenPage`; and **the job object has no equivalent** — Linux has `PR_SET_PDEATHSIG`, macOS has
neither, so *the recogniser dies with argmind* is a weaker promise there and `Supervised::supervised()`
already exists to say so out loud.

**All the inputs, one at a time**, because the owner asked for 화면 · 카메라 · 음성 · 키입력 and
they are in very different states. **화면** is done and on by default. **음성** is blocked on one
measurement whose harness is written — `--bin listening`, 30 seconds and somebody talking.
**카메라 is off, and what it is blocked on is not a model.** Nothing opens it except the 시험
button in 설정; there is no `camera` key in `DECIDABLE`, and the only sense on the blackboard is
화면. The captioning route was measured — `qwen3-vl:8b`, Apache-2.0, a caption of **a drawn test
card** in **about 8 s**, too slow for the 0.7 s watch loop and fine for the 30 s tick; not video
memory (both models sit on the GPU, the answer after a look is 0.2 s) and **not resolution**
(`prompt_tokens` is 1110 from 640×360 down to 80×46) but the model thinking, 467–1,690 characters
about four coloured squares, with **`think: false` ignored** three ways over. **But that is the
wrong route.** `camera.rs` reduces every frame to 64×36 greys and returns nothing larger *as a
property of the type*, because what a camera can honestly supply is **whether somebody is there** —
the fact argmind most needs and does not have, since watching the screen will happily talk to an
empty chair. One look is **32 ms**, not 8 s.

**That is blocked on twenty seconds of the owner's time, and until 2026-09-22 it was blocked on a
broken instrument.** `measure/presence` asks a person to hold still and then move; run with nobody
moving in either phase it separated them **2.17 against 0.65**, three to one in the wrong direction,
purely by which phase ran first. `settle()` returned on the first consecutive pair under `SETTLED`,
which a warming camera offers within about five frames — it reported **329–352 ms** on every open
while the picture went on moving for most of ten seconds. **The harness's own conclusion about the
statistic had never been carried across to the settle check.** It now takes the median of a
one-second window, twice in a row, and reports **2006 ms**; the control then reads null.
**And the measurement it was blocking separates.** Ten seconds each with a person doing it on
purpose: three-second windows give still **0.05** against moving **0.38**, a **7.4x** margin, and
**8.6x** with the mean shift removed. **The centre did not move and the tail did** — the still
median was 0.03 on 2026-09-16 and 0.03 again, while the still *maximum* went 0.40 → 0.12, which is
the one second that had made the tails touch. **The earlier *the camera cannot do it* was the
instrument.** **Reproduced**: a second run gives still 0.06 against moving 0.41, 6.8x, and the four still
values across both runs and both statistics land between 0.05 and 0.06 while the moving ones span
0.33–0.46 — the still floor is the camera and the room, the moving one is how much a person
happened to move. So `camera::MOVING` is **0.15**, the geometric middle of the gap, on
three-second windows with the mean shift removed. **It says movement, not presence**: a person
reading without shifting is below it, and so is an empty chair, so what a caller may say is
*movement was last seen N seconds ago*. **And it is wired**, since the owner chose a light that stays on over one that
blinks every half minute: `camera` is the eleventh decidable key and the only one that leaves a
device running, `설정 → 카메라로 사람 알아보기` is the switch, a standing 허용 turns it on at the
first page open like the microphone and the installs, and `platform::watch_movement` holds the
device and reports every three seconds. **It does not stand down for a turn**, which every other
sense does — that rule is about two things contending for one graphics card and this never speaks
to the model. What reaches the blackboard is an age, *마지막 움직임 N초 전*, because the threshold
measures movement and a person reading without shifting is below it.

**The payoff is a gate rather than a prompt line.** The blackboard is read in one place, the
thinking tick, and the thing that talks to an empty chair is `notice` — so `somebody_about` sits
in front of it and an empty room costs **no model call at all**, rather than a call whose answer is
thrown away. `AWAY_AFTER = 300 s` is **chosen, not measured**, and biased long on purpose: too short
and argmind goes quiet while you are sitting there, too long and it says something to an empty room,
which is what it already does. **No camera never means silence** — off, refused, failing or stalled
all read *no evidence*, and a test breaks the gate two ways to keep it so. *(A caution written here off the null control — that
the still floor moves between runs, so a constant threshold looked unlikely — is withdrawn: nobody
had checked the room was empty, and its author not moving is not the same thing.)* **키입력** is implementable and deliberately not implemented — every keystroke means a
global hook, which is a keylogger capturing passwords in a program that sends text to a model; what
argmind plausibly wants is typing *activity*, which needs no characters and is a different feature
with a different permission.

No phone environment, no robot environment. **A thinking goal and four dials are built and are at rest until turned**: 재치 · 유머 · 봉사 · 헌신,
0–10, and a sentence in the person's own words, in `설정 → 생각`. At rest the prompt is what it
always was. **Voice is built and is off until switched on** — the switch is `소리 내어 말하기` in
설정, not `--speak`, since 2026-09-17. **`speaker` and `aloud` are two keys because they are two
questions**: may argmind build the synthesiser, and may it make a noise in the room. The first was
allowed and argmind still would not speak without a flag. Hearing is
F9 held in the window, which means you have to be looking at argmind to talk to it. **The settings
decide whether it listens, not the command line** — a standing `microphone: allowed` turns hearing
on with no `--hear`, and a standing 거부 turns it off with one. Two places were reporting a
standing answer and not honouring it: that, and the installs behind 항상 허용, which ran only when
somebody pressed 다시 확인 — so after every restart the page said *물어보지 않고 진행합니다* beside
a recogniser that was not running. They now run on the first page open, once per process.

**Already built means start it, not build it again — since 2026-09-21.** `can_just_start` sits at
the top of both build fixes, above the compiler check and the SDK check, because starting something
already built needs neither. **The same shape was live one row over and is fixed too**: `ollama` is
*present* when it answers on 11434 and its fix was a 700 MB installer, so a standing 항상 허용 with
Ollama installed but stopped would have reinstalled it every launch. Now two fixes sit behind that
one key — `Fix::StartOllama` when it is installed and stopped, `Fix::Install` when it is not there —
and the need says which of **three** states it is in rather than two. **`ollama.rs` is not
supervised and that is the decision**: the other two servers die with argmind by job object, and
this one is somebody else's program, installed before argmind and outliving it. **Installed is a
spawn that worked, not an exit code that was zero** — `ollama --version` with nothing to connect to
prints two warnings and still exits 0 on 0.34.2, which is not evidence about a file on disk.
Measured: **5.4 s** from `serve()` to answering, and the server outlived the process that started
it.

**Speaking has two voices and picks one at startup.** SAPI is free (4–43 ms), installed, and
`Microsoft Heami Desktop` — which the owner called 어색함 on 2026-09-16. **The neural one answers
that: 자연스러움, 2026-09-17**, judged from files of both saying the same three sentences rather
than from a demo. `cargo run -p argmind-serve --example voices` makes them again and plays
nothing. The other is a neural
synthesiser argmind builds from source and runs on the GPU: **27–238 ms to first sound** because
it streams, RTF 0.15, and it is used whenever it is answering on its port. If it is not, the old
one speaks rather than nothing. `docs/findings/a-better-voice.md` has the accounting, including
why the talker is Q8_0 and why the Vulkan SDK is asked for separately.

**And which voice, since 2026-09-21, is one of five argmind cloned for itself.** The base model
ships with no speakers and a seed holds only the sentence it is saying, so each preset is a
reference recording the model drew at a fixed seed and then had registered back — `platform.rs`
does it, because `speaker.rs` holds the presets and names no environment crate. It costs 0.69 s at
startup and again whenever the voice changes, and it happens every launch because the registry is
memory. **A failure falls back to the seed rather than to silence.** The key lives in
`devices.json` as the fourth thing that file holds and the first that is not a device; a missing,
unknown or stale one — including the bare seed the version before this wrote — is `A_VOICE`.
`설정 → 장치와 목소리` has the dropdown, ordered by measured pitch with the Hz printed beside each
name, and **choosing speaks**, because a name and a number in a list are not a voice until you have
heard one. **They are numbered rather than described**: three of the five sit within 22 Hz of each
other and are told apart by timbre, so 낮은 and 조금 낮은 would be a distinction the measurement
does not support. With SAPI answering the row greys out. **The engine still changes once per
launch** — the synthesiser dies with argmind, so startup finds no server, takes SAPI, and swaps
when the page opens one.

Hearing **transcribes silence as words 120 times out of 120**, so a stray key press is a turn —
it cannot act on one, but it will answer it aloud. **Two of the three
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
into the speech — *"3, 122"*, *"지우개는 22번 버튼입니다"*. **Whether a reply is a sentence at all is
measured now, and only just**: `parroting` is the one harness on that axis and it has six cases. Every turn is framed by a prompt about picking widgets even when the person says
*"고마워"*. **A standing rule is broken while it is visible in the prompt**, so that one is not a
memory problem.

**Control patterns ship, since 2026-09-22, and typing does not.** A pattern calls a method on the
control and leaves the pointer where the person put it; a synthetic click drags it to the widget,
measured 3 of 3. They were off because *the shipped click is confirmed by a person pressing a
button, so the cursor was already about to move to something they chose* — and 확인 없이 실행
removed that reason the day before. `platform::hands` asks for them, which is also the seam that
stopped the application building its own hands. **Typing is the half with no answer**: it is
`KEYEVENTF_UNICODE`, so it lands in whatever has focus, and UI Automation has no insert —
`ValuePattern::SetValue` replaces the whole value where AT-SPI's `EditableText.InsertText` inserts
at an offset. Keystroke-free typing on Windows would mean *replace this field*. **The narrow shape is built and
off**: `patterns::set_value` writes only into a control that is empty, takes a value pattern and is
not read-only, and falls back to keys otherwise — behind `with_typing_patterns`, a **second** switch
so that turning the measured one on does not carry the unmeasured one with it. `argmind-measure
--bin typing` has an arm on each side and answers four things: does it land, is it the same text in
Korean as in ASCII, does it refuse a box that already holds something, and what does it cost. **It
takes the foreground, so it cannot be run from the session driving it** — the same constraint
`clicking` has, and the reason there is no number yet.

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
| `docs/findings/the-window.md` | **a standalone app rather than a browser tab**: why the house style was the wrong answer, the licence sweep across 81 crates, and the DLL failure that only a double-click finds |
| `docs/findings/eyes-on-linux.md` | **AT-SPI2**: the contract surviving the port, the switch that makes toolkits publish at all, `i32::MIN` for off-screen, the hit test that answered with the container, and why D-Bus was the one wire protocol not hand-written |
| `docs/findings/a-better-voice.md` | **the voice**: every open Korean TTS checked and its licence, the measured answer — RTF 3.9 on the CPU, 0.15 on the GPU, and a K-quant that is thirty-two times faster and wrong — and **which voice**: the base model with no speakers, the seed that pins it, and the quiet-room rule widened by one sentence and broken four ways to check |
| `docs/findings/thinking-continuously.md` | **the note**: what a turn never carried, the three things the first run got wrong, and the watch that was reading argmind itself |
| `docs/findings/what-it-is-trying-to-be.md` | **the dials**: the rule that made a sixth prompt edit safe, the control that overturned the first reading, and the language of the slot |
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
