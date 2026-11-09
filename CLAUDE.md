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
                                task.rs      work carried to the end: one tool a step, the
                                             record verbatim, a `done` checked against it
                                errand.rs    work handed over: check, think, conclude — the
                                             program deciding when, the model one thing each
                                mind.rs      (also) the sentences it was asked to keep
                                rule.rs      where it may not act and where it must always ask,
                                             per app and per action — the program's, never shown
    native/argmind-env-windows/ UI Automation for eyes, SendInput for hands, WASAPI for the
                                microphone, SAPI for the voice, a cover window
    native/argmind-env-ollama/  the model, over raw HTTP/1.1
    native/argmind-env-whisper/ transcription, and the guard that stops silence becoming a turn
    native/argmind-env-qwentts/ the neural voice, over raw HTTP/1.1, streamed
    native/argmind-voice/       Heard / Listener / Speaker — the seam, and why it is not a contract
    native/argmind-serve/       the application: a server, a page, a tray icon, and `argmind`
                                senses.rs    the blackboard: a slot per sense with its age, and
                                             the one bit that makes them stand down for a turn
                                errands.rs   맡긴 일: the list on disk, a step every 5 s on the
                                             thinking thread, a report only to somebody there
                                activity.rs  한 일: what it heard, saw, said and did, read off
                                             every event the page is told — memory only
                                surface.rs   desktop or argmind's browser pane — one session,
                                             one switch, a proposal pressed where it was made
    native/argmind-measure/     every harness behind the numbers, and `--example raw_turn`
                                for looking at one reply whole
    native/argmind-env-linux/   eyes for Linux: AT-SPI2 over D-Bus. No hands yet
    native/argmind-env-browser/ eyes and hands for argmind's own browser pane, over DevTools —
                                page events only, never the person's pointer or keyboard
    native/argmind-env-mcp/     MCP servers the person configured, as a task's tools — stdio
                                JSON-RPC, std and serde_json only
    native/argmind-env-none/    an environment that can do nothing and says so — the proof the
                                application is separable, and the list a third one must satisfy
    native/argmind-rules/       the rules this repository keeps, by looking at itself
    native/argmind-window/      the window the page lives in, the icon in the tray, and
                                argmind's own browser pane — driven without a debugging port
    ui/                         the front end, TypeScript with no dependency but `tsc`
    measure/                    goals, captured screens, and every answer behind the numbers
    native/expectations/        what the Python produced, committed as data
    native/golden/              two recorded transcripts the suite replays

**Nothing needs a display, a model, a network, a microphone or a speaker.** Re-derive with
`cargo test --manifest-path native/Cargo.toml`.

**Five rules are held by reading the source rather than by the compiler**, in `argmind-rules`:
the licence record, the core naming no platform and being shown no coordinate, one place calling
`perform`, a voice seam that cannot name a `Session`, — since 2026-09-16 — **the application
naming an operating system in one file**, and — since 2026-09-23 — **the application taking no
photograph.** Each was checked by breaking it.

**The fifth one replaced a type, and that is a loss written down rather than hidden.** `camera.rs`
reduced every frame to 64×36 greys and had no accessor for anything larger, so a picture could not
leave the module however anybody felt about it. Asking whether a vision model can describe a room
means giving that up: `Camera::photograph` exists, **takes the size as an argument** so the caller
states in its own source how much of a room it wants, and what stands where the type stood is a
rule — only a harness a person runs on purpose may call it. Two *code* breaks are caught, a
`photograph(` and a bare `::shrink(`; a commented-out one is not, deliberately, for the same reason
`hands.rs` needed a day earlier — **a rule that forbids writing down what it forbids is one nobody
can document.**

**One of them has an exception now, and the exception is written against what may be said.** *The
room stays quiet unless somebody asked* allowed exactly one `say` in `app.rs`; choosing a voice
means hearing it, so there is a second, and it may say only the one fixed constant `A_SAMPLE`. A
`say(&note.believes)` added to `notice` still fails. Two of the four breaks used to check it would
have walked through the **old** rule, which matched the literal text `voice.say(` — a rule about a
local variable's name.

**One of the five was widened on 2026-09-28, deliberately and with the break checks.** *Only three
files outside the core may reach an action* now also skips `argmind-core/tests/`, on the same ground
`argmind-core/src/` was already skipped: that is the crate where `perform` lives, a test binary
there is linked into no executable, and nothing at run time can reach it. What made it necessary was
a test that pins a **refusal** — a click with two candidates must not land — and a rule that makes a
refusal harder to test is working against itself. **`argmind-serve/tests/` is still caught**, which
is the half worth keeping: a test there would reach the real hands and click whoever ran it. All
three cases were checked by breaking them.

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
opened the character set list when the font list was asked for.

**Since 2026-09-28 the lookup says so rather than guessing, and the rule came out of the comment
that recorded the failure.** *The tie-break anchors on the first same-named widget in the previous
reading, so among copies the anchor is itself arbitrary and nearest-to-it is arbitrary too* — so the
tie-break means something in exactly one case, when **the previous reading held exactly one of
them**, which makes the anchor a fact about where that thing was and *nearest to it* a statement
about where it went. With none or with several it is a coin toss with a distance calculation on top,
and `find` now returns `Found::Several` instead. `perform` refuses those and counts them in
`ambiguous`, apart from `vanished`, because a world that moved and a name that was never a key are
different failures. **The committed Python expectations confirm the rule rather than merely
permitting it**: of eleven `find` cases it reclassifies exactly the two where the previous reading
held none or several, and both duplicate cases with a real anchor still pass unchanged — so the
expectations file was not edited, and `argmind-core/tests/equivalence.rs` names the two divergences
and asserts them. **Nothing asks the model**, which is shown the same two fields and cannot name the
one it means either; the deterministic side knowing there are three buttons called `Open` is
knowledge a person can act on, which is the shape of every other answer here.

**And it asks.** *Arg* — the owner, 2026-09-23: 모르는건 물어보는 기능도 있어야함. A `Several`
raises a question with one button per candidate, each carrying its position, and **the question
replaces the offer rather than joining it**: the proposal is still filed so the action keeps its
token, but no row is drawn, because a 실행 button beside a question asking which one is two
decisions for one action. **Answering is the confirmation** — somebody who picked the second
`Open` has been at least as deliberate as somebody who pressed a button — and it still goes through
`Proposals` and the one call site that performs anything. The answer travels as `Effect.which`:
an index **and the count it was chosen against**, because three buttons can become two while
somebody reads, and *the second one* then names a different button. Re-resolving alone does not
catch that — the name still matches — so the count is checked first and a stale answer is refused
and counted in `chose_too_late`, apart from `ambiguous`: *nobody was asked* and *asking is too slow
here* say different things about whether this is worth having. **Coordinates reach the person and
never the model**, which is the rule stated precisely rather than widened. `argmind-core/tests/duplicates.rs`, 12 tests.

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

**A silent notice is not a turn — since 2026-10-21.** `noticed` opens a turn as `them` does, and
every window change was recorded before anything was known to come of it, so twelve window changes
pushed the person's own conversation out of the prompt: the owner said argmind did not remember
what they had talked about, and eleven hours of the timeline held 97 notices against a handful of
their words. A notice now enters the history only when it is said or acted on. **And the other half,
the same day**: the conversation is appended to `conversation.jsonl` and read back at start, and the
prompt no longer stops at twelve turns — `Session::context_tokens` fills what the screen and
`ANSWER_ROOM` leave of the 8,192-token context with whole turns, newest first, by a token estimate
that charges 1.0 a Hangul character and 0.4 anything else (measured 0.59–0.66 and 0.31 on qwen3's
tokenizer; over is the safe side, since an overlong prompt loses its instructions off the front).
`--bin context`, conditions written first: a first-turn fact after twenty turns recalled **3 of 3
fitted against 0 of 3 at twelve turns**, and `choosing` **31 of 37 with that history against 30 with
none**. On the headless server, a fact said, the server killed and restarted, and asked: *나비라고
했어요!* No summaries are kept — the model wrote things it had not done the one time it described
itself. 대화 기억 지우기 empties the file.

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

**A password is the person's to type, and a rule is the person's to write — since 2026-10-01,
from the dots comparison.** `Hands::into_a_secret` is asked before every `type`, and `perform`
refuses and counts `secret` apart from *gone* and *ambiguous*; on Windows it is UI Automation's
`IsPassword` of the focus and of the box just clicked, and the hands refuse on their own too, so a
harness that skips the session is still held. **Default `false`**, so an environment that cannot
tell does not stop typing — the weaker direction, chosen so typing is not Windows-only. Checked
against a hidden window this test makes, an `ES_PASSWORD` box and a plain one, both directions;
**a browser's password box is not checked.** **And a rule had to be narrowed for it**: `hands.rs`'s test forbade `GetFocusedElement` anywhere,
because typing aimed at the focus was measured and rejected; the guard *reads* the focus, since keys
land there whatever the hands decide. Allowed once, inside `typing_into_a_secret`, and broken two
ways to check — a focus read in `type_string` and a second helper beside the guard both still fail. `규칙` in 설정 holds rules by app (part of a window
title, empty = everywhere) and action (all / click / type / key): *하지 않음* is never offered and is
checked again at the press, *항상 물어봄* takes the row away from 확인 없이 실행. The strictest rule
wins in either order. None at rest, which is today exactly. **The gate inside `propose` and `act` is
not exercised end to end** — the headless build sees no widgets — only `rule::applies` beneath it.

**한 일 is recorded where the page is told, not where somebody remembers to log** — since
2026-10-01, item 5 of the dots comparison. `Listeners::send` turns every event worth a line into
one (`activity::reading`): heard, the window in front, said, noticed, offered, ran and whether the
screen moved, could not and why, asked, held back. A new path that tells the page anything is on
the timeline with no logging call written. The person's typed words are the one thing recorded
explicitly, because the page draws those itself. **Memory only, 500 lines, gone at exit** — a file
of every window title and every sentence is a privacy decision with its own switch, not a side
effect of a log. Driving the headless server to check it found a bug in the keep reader the same
day: *불러달라고 기억해줘* kept *불러달*.

**The browser pane is a surface of the one session, not a second session** — since 2026-10-20.
`one_place_acts.rs` holds that `perform` is called once, inside `act`, and a second session would be a
second place; so the eyes and hands are a pair (`surface.rs`) that reads and presses the desktop or
the pane on one switch, and the conversation, the button, the rules and the timeline are shared. **A
proposal remembers its surface** and `act` presses there behind a guard that puts the switch back on
any return — the first version wrapped `act` and the rule caught `perform` moving into the wrapper.
**Watching reads the desktop whatever the switch says**: the pane is argmind's own doing. The address
box opens an address or searches **Naver**, because Google refused the pane. **Not run inside the
application from here** — `argmind-app` takes the foreground and may open devices the person allowed;
the pane's hands were measured, the switch and the box were not.

**Each step is checked against the screen, and a step that did not go ahead stops the rest** —
the owner, 2026-10-21: *매번 화면상태를 확인하고 해야함*. With 확인 없이 실행 just turned on, *오늘
서울날씨 알려줘* on the pane's Naver login page became a click on the ID box, *서울* and Enter, one
after another: a click re-read the screen, **nothing asked where the text would go, and nothing
stopped the steps after one went wrong.** Now `Hands::typing_lands` is asked before every `type`
(the page's focus on the web; on Windows the box just clicked or the focus, by value pattern then
control type — a read-only edit is an Edit and takes nothing), `Hands::landed` after every action,
and `perform` ends the sequence at the first refusal, counting what it did not try (`unattempted`).
The application does the same across proposals: `propose` reports whether the next step may go
ahead, and a turn whose step failed says which steps it left. **And a question about now, asked with
the pane on, is not planned by clicking**: the program searches it in the pane and an errand
concludes from the page. Checked: five step tests in the core with a control where nothing fails;
edit yes, button no, read-only no in a hidden window; on a live page, a field after clicking
Wikipedia's search box and none on `example.com`. **What this does not catch is a plan that is
wrong but well-formed** — typing into a field that exists, on a screen that answered every check.

**argmind's own ideas may run without a person — on their own key, since 2026-10-21.** The owner
found asking every time tiresome; eleven hours of the timeline showed what was asking: **0 requests
of theirs and 1 proposal row, against 97 notices, 37 unprompted remarks and *Move all to Desktop* —
Google Drive's button for moving lost files — raised every minute and a half**, in Chrome and a
search page as well as in its own dialog, failing *그새 사라짐* 15 times. Asked, the owner chose *run
argmind's own ideas too* over *never ask, never do*, knowing the example. So `initiative` is the
fifteenth decidable key and the gate in `propose` reads **which** standing answer by whose idea it
was: `acting` for what was asked, `initiative` for what argmind began — 확인 없이 실행 alone still
never carries out argmind's own idea, which is what `tests/acting.rs` now pins. **Two defects went
with it**: a judgement is raised once a session, not once per three looks; and a window change is
remarked on only when the remark comes with something to do (the owner chose *fewer* over off).
What still stands between argmind's idea and a click is the screen check, the rules and the second
opinion — and `noticing` measured it wanting to click on 5–9 of 28 window changes.

**The pane opens and closes itself, by what was said — since 2026-10-21.** The owner: nobody presses
웹 by hand. `surface::web_move` is a word rule, not a model call, for the same reason everything
else that moved out of the model did: a site or an address opens it (and a request after the name —
*네이버에서 메일 열어줘* — carries on inside it), *…검색해줘* opens it on a Naver search, a question
about now opens it and is looked up, *웹 닫아* closes it, and a desktop app named while it is up
(*메모장*, *탐색기*…) takes the turn back to the desktop. *검색창 눌러줘* is a box, not a search; *다음*
is not a site. Errands looking things up in the background still do not show it. **Not run in the
application from here.**

**What a person asks for runs without a question — since 2026-11-03.** The owner, five 실행 rows on
screen: *자꾸 실행을 물어봐서 일이 진행이 안됨. 그냥 자동으로 수행해야돼.* The timeline said why: a
request answered with *click the search box, type 일본, Enter* sat behind `SURE_ENOUGH`, which was
measured on single clicks and multiplies over every token of `do` — a typed word and a key pull a
correct plan under 0.94 almost by construction, **a flaw in that measurement's design** rather than a
finding about the plan. So with 확인 없이 실행 on, a request runs past the sureness gate and the
second opinion; what still stops it is the person's own *항상 물어봄*, the *which one?* a program
cannot answer, and every screen check in `perform`. **argmind's own idea, doubtful or unsure, with 스스로
실행 on, is now neither asked nor done** — a row nobody was waiting for is what stalls the work. And
`notice` pauses while the pane is up: a switch to Snipping Tool had been planned against the pane's
list as *click Naver's search box*.

**Work is carried to the end by a loop the program runs — since 2026-11-03.** The owner: *복잡한
업무도 처리가능해야함 문서수정이나 작성 그리고 코드도 필요하면 생성해야하고 실행도 스스로.* A turn picks
at most three clicks from one look; work is `argmind_core::task` — **the model picks one tool and
its arguments a step, the program runs it and writes down what came back verbatim**, and the program
decides when to stop (25 steps, three failures, the third identical call). The workbench is
`argmind-env-windows::workspace`: files inside `%USERPROFILE%\argmind-work` with `..` and absolute
paths refused and every replaced file backed up, `run` in **`argmind-sandbox`, a WSL distro of
argmind's own with drive mounting and interop off** that mounts only that folder per command, and
`run_windows` as PowerShell in it — weaker, and said so. Chosen by the owner: both, by the work.
**And a `done` is not taken at its word**: the first run had qwen3:8b end five of six tasks after
0–2 steps with a summary it had not written and a total — *합계는 150000* — no command printed, so
`why_not_done` refuses a finish with nothing done, one naming a file that is not there, or one
holding a number of three digits or more that no tool result, the request or the date contains.
What it cannot catch is a sentence that claims and names nothing checkable — 8b's *calc.py 를
수정했습니다* after only reading it went through. `asks_for_work` is the word rule that routes a
turn there (a thing — a file name, 파일, 문서, 코드… — and a doing — 만들, 고쳐, 실행, 요약…);
**one task at a time on its own thread**, on qwen3:14b with reasoning on (`complex work`), each
step on the timeline as 작업, **its model call abandoned the moment a turn begins** and asked again
after, *멈춰* stopping it, the end said and kept. **Not run inside the application from
here**: driving the server means opening the page, and that applies standing answers that open the
camera and microphone.

**One call, not two.** A round trip is the budget: 0.45 s warm, 4.7 s and 5.9 s for the first two. **One exception since 2026-11-03**: where the word rules say *a turn*, the router asks the same model *is this work?* first — 0.15 s median, measured, because the rules left seven pieces of work in fifty-five as turns.

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
| typing without keys | **it works, it refuses what it must, and it ships — after three runs that each measured the harness instead.** Character Map, a fresh window per case: `ValuePattern` put both `hello` and `안녕하세요 반갑습니다` in correctly, and a box that already held something was appended to rather than replaced (`"먼저"` → `"먼저추가"`). **The first design wrote to the *focused* element, and getting focus there is a click** — the pointer collision it was meant to answer, so it answered nothing. It now writes at the point of the **widget argmind just clicked**: the hands hold the click, `type_string` takes it once, and an aim older than `AIM_LASTS = 1 s` is no aim — 250 ms between effects in one `perform`, 0.61 s for the model round trip it must not span. No prompt edit, because the target was never in the reply; it was in the effect before it. **What it buys, no wider than it is**: the click still moves the pointer, but in those 250 ms a person can click into a box of their own and argmind's sentence still goes where argmind aimed — plus the text arriving at once and no IME mid-string. **Three runs measured the harness before one measured the capability**: `ctrl+a` is not one of the fifteen named keys so the clear between arms was refused and every arm ran dirty; `notepad.exe` is a packaged launcher so the window belonged to another process and the reach guard would have refused it; and the fallback reason was never printed, so *1 characters already there* took four runs to become **`"
"` — the control reporting its own line terminator as its value**, which is why `blank()` trims |
| crowding | **choosing does not survive a list the size of a web page, and the sandbox's gate is where it failed.** The same 37 goals and 4 screens, padded with real widgets borrowed from the other screens, at 123 / 250 / 500 / 900: answerable goals go **22 of 28 → 3–8 of 28**, two independent padding arrangements. The control reads **29 of 37 both times** against the usual 28, so padding is the only thing that changed. **Declining is untouched** — 5 to 8 of 9 throughout, flat — so what crowding removes is finding, not the judgement to refuse. The tail is noisy and **position is not disentangled**: at 900 the answer sits near row 390 and the two seeds read 8 and 3. It rules out handing argmind a page as a widget list; it does not rule out a sandbox that narrows first, which is deterministic, allowed, and a different project |
| errands | **handed work, it finishes it only when the program makes it; and it cannot tell what only you know.** `argmind_core::errand`, from the dots comparison: an errand is thought about one model call at a time with no `say` and no `do` in any answer shape. **First design, the model choosing its own state: 0 of 3 open errands done** (all to the step cap — *make money* drifted to OBS settings), the meeting planned eight steps and never asked, and **the weather invented the date (2023년 10월 5일) and a 15~22도 forecast**. The act check caught nothing: what went wrong was facts, not acts. So the program decides — a check, at most four thoughts, then a forced conclusion with **today's date given rather than guessed**: **3 of 3 done**, weather stopped in the program's own sentence, no other year and no temperature anywhere. **The check is a choice, and it follows position**: as yes-or-no it stopped 4 of 5 (more information always helps); as three kinds it reads **A 5 of 5, B 3 of 4, C 0 of 4** on thirteen held-out errands, three runs identical; a reworded version with *nothing missing* last read **A 1 of 5** — the first option wins — and failed a keep rule written before it ran. **So C is not relied on**: the meeting concludes *10시 every weekday*, wrong and harmless only because an errand cannot act and is handed over as a draft. **Done is not useful**: the exercise routine names no exercise, and nothing measures useful. **Wired 2026-10-01**: a 맡긴 일 panel, `errands.json`, a step every 5 s on the thinking thread whether or not anybody is there, and a report said only to somebody who is — the camera's word and 눈치's. Driving the headless server found the next defect at once: **an answer from the person was ignored** — told *맑고 18도, 우산이 필요한지만*, the weather concluded *확인할 수 없어… 15~20도*. The answer sat among argmind's own notes; moved to its own block above them, present only when there is one, `--answered` went **2 of 3 → 3 of 3, twice** (*우산이 필요하지 않습니다*), with the weather criterion tightened after the baseline and the baseline re-read under it. The gift used *등산* and dropped the 5만원 budget. `--bin errands`, `--preflight`, `--answered` |
| remembering | **a kept sentence moves the model, except the one about language.** `Mind::kept` — the person's own sentences, in every prompt under the label the goal already had, absent when there are none so the probes still hash to `r322a4c`. With and without, three different turns each: *나를 대표님이라고 불러* **3 of 3 against 0 of 3**, *답은 한 문장으로만 해* **3 against 1**, *메모장 말고 VS Code를 써* **3 against 1** clicks on the editor; *항상 영어로 대답해* **0 of 3** — `the dials` again, the slot's language decides the reply's, and a Korean slot cannot ask for English. The control kept *나는 고양이를 좋아해* across six unrelated turns: **고양이 0 of 6**, actions 0 of 5 against 1 of 5 without. n = 3 a case, so no case can reach significance; both conditions were written before the run. **Kept by the program, never by the model**: *…기억해줘* is read by a word rule and stored verbatim with no model call, *이거 기억해줘* is asked back rather than stored, and a turn that reads as a correction (*말고*, *앞으로*, *하지 마*…) gets a 기억해 둘까요? row that keeps nothing until pressed. `--bin remembering` |
| browsing | **argmind's own browser pane can read the web, and what is on screen is 65–154 things.** The owner chose a pane inside argmind's window (2026-10-20) over a separate Edge window and a downloaded Chromium, so web work never meets the person's pointer, keyboard or windows. Driven through WebView2's `CallDevToolsProtocolMethod` — **no debugging port**, which every program on the machine could reach. Read-only so far: interactive, named and on screen, from the accessibility tree and a layout snapshot. **The control failed first and was right**: boxes in device pixels against a viewport in CSS pixels, two thirds apart at scale 1.5; one unit, and `example.com` reads its one link. Then Naver **68**, Wikipedia **100**, the weather service **65**, GitHub **90**, Hacker News **154**. **Google refused the pane** with its automated-traffic page, and the pre-written criterion counted that as a pass — so **4 of 5 real pages ≤ 100**, the bar with nothing to spare. All of it sits between `crowding`'s 22 of 28 and its collapse from 123, where choosing had never been measured. **And choosing there, one run: 15 of 20 answerable, 5 of 5 declined** — over the 60% bar written first, and read as flattering: the requests were written looking at the lists and the impossible ones were obvious. The misses: **a Korean request for an English label 3 times** (*글 올리는 곳* → `submit`), **off by one** at row 30, **row 66 of 68 not found**. And in the speech, **an index in 5 of 25 replies** and **English out for Korean in on English pages, 5 of 25** — the list's labels acting as the English slot did in `the dials`. `--bin browsing`, `--choosing`. **Hands, 2026-10-20 — 4 of 4 as written first**: a click in the pane took `example.com` to `iana.org` with the person's pointer at (1814, 0) before and after; *경복궁* typed into Wikipedia's search box read back from the box; a link covered by a layer in the pane's own page came back `Elsewhere` and the page did not move — the negative control; a password field read as one and a text field did not. One sample each. `argmind-env-browser`, `--hands`. **Errands that need the web now look it up** — the program searches the errand's own words on Naver in the pane, only while the 웹 switch is off, and the page text reaches the model fenced as material: **3 of 3 looked up, every number on the page**, all eight film titles on the page — **and one sentence about the films is not on it** (*기대감*, *발표*), which the number-only check passed and reading the saved page caught. The check alone missed two of the three, the weather because the **date in the prompt changed**, so words about now (*오늘*, *지금*…) now mean the web too, a rule chosen after seeing the set it fits. A planted *[AI에게]…* instruction stayed out of the result, n = 1. `--research`. **The owner's first session with the pane, 2026-10-21, found three things in its own timeline**: Naver's 메일 is `target="_blank"` and the refused new window made the click land and do nothing — new-tab links now open in the pane, and the same click reaches the login page; *오늘 서울 날씨 알려줘* got *잠시만 기다려주세요!* and nothing, since a turn cannot search — a question about now now gets a 웹에서 찾아볼까요? row that makes an errand; and the note raised the pane's own *메일* link as a judgement — thinking about the desktop pauses while the pane is on |
| sureness | **the probability of the action's tokens separates right clicks from wrong ones, held out, and it is not calibrated.** From the Jev comparison (2026-10-21): Ollama 0.40.6 returns token probabilities, and a probe found them **uncalibrated** — the errand check put *needs the web* at 0.90 on *돈 버는 방법* — **but ordered**. `converse::action_sureness` is the product over the tokens of `do`. On `choosing`'s 37 goals: AUC **0.78**, 21 of 27 actions at 1.000, and **0.94** keeps 21 of 23 right against 22 of 27 without; written before the run, met. **Held out**, the browsing pane's live pages: **15 of 16 right at or above 0.94, both below it wrong** — each clicked links nobody asked for beside the right one (0.223, 0.926), which the harness's own scoring had counted as right until that run. Two wrong actions sat at 1.000, so it cannot catch a confident mistake. `SURE_ENOUGH = 0.94` now holds back any action below it, asked for or not; asking costs **+0.02 s** a reply (0.315 → 0.336 s median, 5 each). Laya (Apache-2.0) is the next thing to measure against it. `--bin sure` |
| seeing a room | **three answers, and the top one is a negative.** `qwen3-vl:8b` on one photograph per room state, the occupant reading back both times, 16 asks. *Somebody is there* and *a yellow cup* are **16 of 16 at every size down to 160×90 and 43 kB** — neither word is in the question, so the cup is the token that could not be guessed. **The gaze is where the sizes part**: 4 of 4 from 320×180 up, **1 of 4 at 160×90**, which said 보고 있습니다 in *both* states — a size that answers the same whether or not the thing is true is not reading it. **And the largest arm invented a detail**: 1920×1080 added *눈을 감고* on 2 of 3 asks and the eyes were open, the only false claim in sixteen captions, at the one size with twice the tokens. So **320×180** — the arm above it is not the cautious choice. Costs nothing extra: warm, 6 MB at 2102 tokens is **not distinguishable** from 43 kB at 1137, and the 17.2 s first read was the model arriving on the card, written up as the token count and **withdrawn**. Negative control **0 of 3** on flat grey, noise and the four-square test card, and the harness's own tally was wrong there — it knew one Korean negation — which only the sentence printed beside it revealed. Binary question, so **it cannot reach p < 0.05**, and the second room state was luck |
| attention, cheaply | **a turned head is thirty times a facing one, and the expensive arm of the same run is void.** YuNet's five landmarks at **99 ms** and no model: looking reads medians of **0.20 and 0.40**, turned **15.05 and 24.08**, and the tails do not come near — 0.41 against 12.70 over 311 samples. **Both orders agree**, which is the control that exists because `presence` once separated a still room from itself three to one by order alone. `HEAD_ASIDE = 1.0` is **geometry, not the middle of the gap**: the run refuted this file's own claim that `head_on` is bounded — the divisor is the eye separation, which goes to zero at profile, so it read 15 and 24 rather than saturating at 1, and the midpoint 2.29 sits where the value is unstable. Anything from 0.5 to 10 separates the data and 1.0 is the only value in it that means something (*the nose has reached an eye*). **Unreadable is a signal**: 0 and 1 of 95 looking against 22 and 48 of 96 turned, so the surviving turned samples are the less turned ones and the gap is an understatement. **And the model arm says nothing** — all four phases answered 보고 있다 in two words at 2.3 s, and the reading *the expensive path failed* is **withdrawn**: the question was a forced binary with the affirmative first, where `seeing` asked openly and got 4 of 4. Four identical short answers is a broken measurement, not a result. `docs/findings/attention.md` |
| the pointer | **the axis the patterns do win, 3 of 3 — and they are on now.** Parked at `(700, 400)`: `invoke` leaves it there, a click drags it to the widget at `(130, 459)`, and both press the box. **The first version of this harness measured nothing** — it parked with `cursor_round_trip`, which puts the cursor back where it found it, so every arm read the position the previous arm's click had left and both reported *unmoved* |
| tasks | **qwen3:14b 6 of 6, qwen3:8b 0 of 6, and the 14b runs the work** — the rule written first was *14b replaces 8b only if it passes at least two more*, and the conversation stays on 8b. Six tasks checked by the program against files and records, never the model's account: write and run a script, summarise into a file, a one-line edit with the rest byte-identical, a failing test fixed (re-run by the harness in the sandbox), a CSV's sum and mean, a Windows file listing. **The first run read 1 against 1, and two of 14b's five misses were the workbench**: the sandbox kept the first task's folder mounted, so a correct fix was tested against the wrong files, and `dir /B` failing in PowerShell came back as an ordinary result. Fixed with the `done` check and a circle rule that warns on the first repeat (it had stopped at the first): **5 of 6 against 1**, reproduced, and the run between found the check itself letting an invented *150000* through on the second try, because the refusal quoting it counted as a source. **The sixth is fitted to this set**: telling `run` to put multi-line code in a file came after watching 14b pack a program into `python3 -c` three times. **8b's 0 is the check working** — every one of its finishes was refused, where 1 of 6 before was its one honest run. One run each, six tasks: a ranking of the two models on this workbench, not a rate. `--bin tasks`, `TASKS_VERBOSE=1` for every step |
| complex work | **reasoning on is the lever: 11 of 12 and no false finish, against 8 of 12 and two — at 148 s a task against 9 s.** Twelve tasks a person would hand over — three bugs with the test file to stay untouched, a printed format to meet, a CSV with a bad row to leave out and name, three edits in one README, tests that must *fail* on two wrong versions the harness swaps in, a refactor with output unchanged, a JSON merge, minutes in three sections, a zip, and three held out — each checked against files and the sandbox, every check first shown to pass on a hand-written answer and fail on the untouched files. **The first run read 8 of 18, and the repeats were identical, so 4 of 9**: six false finishes. **A change fitted to those failures did nothing**: a look at the changed files before a finish took it from 8 of 12 to 7 and made one reply unreadable — reverted; what stayed is refusing a finish after a failed run and the workbench refusing a long `python3 -c`, which turned the CSV around. **The held-out three were no test**: the old loop passed them too, so *the change generalises*, which the harness printed, meant nothing. **And two of the harness's own errors**: the CSV check wanted the keyword and the 5 on one line and failed a right report, and the number check read tool results and not what the work wrote, so a test holding `is_leap(1900)` had its finish refused. Thinking on still ends two tasks *unreadable* with the work already right. One run each. `--bin complex`, `--control`, `COMPLEX_THINK=1` **Then a plan and a person, 2026-11-04 — both built, both off, by bars written first.** A plan (`Task::planned`: 2–6 items written before the work, each closed only by a quotation the program finds in a tool result, or taken out with a reason that goes to the person) read **8 of 12** against a bar of 10 — and six of its eight passes were *stopped anyway*, the work right and an item like *the code is written correctly* open, because a plan written before looking guesses and a guess has no sentence to quote. The skip was added after seeing one such guess (*meeting_notes.md*) and is fitted to it. Asking (`task::Asking`, `ask_person`, at most two questions, the answer coming back through the conversation) **works where only the person knows**: three errands, **3 of 3 and 2 of 3 with it, 0 of 3 without** — without it the model wrote placeholders or stopped, and once invented three to-dos and was let through. **But it costs the ordinary work**: the twelve tasks read **8 and 10 of 12 with the tool on, each run with a false finish, against 12 of 12 and none with it off in the same hour** — the bar was 10 *and* no false finish; one of its two questions asked for a file it could have read. The first ask run's two WSL `Operation aborted` were nearly credited as the workbench's; the control run had `run` failing too and still read 12. `TASK_PLANS` and `TASK_ASKS` are `false`; `COMPLEX_PLAN=1`, `COMPLEX_ASK=1`, `--asking`. **Then reasoning only where it counts, and a bigger model, 2026-11-06 — both against bars written first, both refused.** Sparing (`Task::sparing`: the first step and the step after a failure reason, a quick *done* is asked again with reasoning) read **complex 9 of 12, documents 12 of 13, two false finishes, 68 s and 69 s a task** against reasoning everywhere's 12, 12, one, 110 s and 71 s — fewer reasoned calls (33 of 81) and still **725 s of the 801**, because failures and finishes are where the work is. `qwen3:30b-a3b` (Apache-2.0, 19 GB, 26% on the CPU) read **documents 13 of 13 with none false, complex 10 of 12**, 157 s and 79 s; the bar asked for more on one *and* no fewer on the other, since it cannot share the card with the conversation's 8b. **And the context is not the limit**: the largest task prompt in four runs was **2,067 tokens** of 8,192. `TASK_SPARES` is `false`; `COMPLEX_SPARING=1`, `DOCUMENTS_SPARING=1`. **Found on the way**: the application's task thread steps by itself and had never called the opening read `--bin documents` credited — `task::open` is now called there too. **And a task can press in the pane, 2026-11-06** — the first piece of joining the paths: work could read the web and not use it. `task::Pressing` gives it `look_pane`, `click`, `type_text`, `press_key` (and `read_pane` to read again): **the model picks a number from the last look and the program turns it into a name and a role**, the numbers forgotten after every press. **The pane only, the owner's choice**, since a task runs for minutes beside somebody's own desktop. In the application each press is a `Job` to the worker and goes through `propose` and `act` like a turn's — rules, password fields, screen checks, and a row the task waits on when 확인 없이 실행 is off — and what comes back is the timeline's own lines. `--bin browsing --work`, three pages that draw their answer only after a press — a form, a tab, page 3 of a list — **written into the pane** because the pane refuses 127.0.0.1, which is `is_web` keeping it off argmind's own server and stays: **3 of 3 with the tools, 0 of 3 without**, as written first, one run. The harness presses with the pane's hands directly, so **the application's gate around it is not what was measured**, and it has not been run inside the application from here. **What did ship**: `read_step` reads `{"<tool>": args}` as a call, the shape 14b wrote three times running — the control's 12 of 12 was measured with it |
| documents | **Word and Excel, made, changed and read — 12 of 13, and the bar written first missed by one false finish.** The sandbox has `python-docx` and `openpyxl` (both MIT, Ubuntu's packages, installed **without recommends**, because with them Pillow brought `libimagequant`, **GPL-3**); `read_file` returns a document's text read there, and `write_file`/`edit_file` refuse a `.docx`/`.xlsx`, which they would have written characters into. `--bin documents`, every input made and every check run in the sandbox, each check shown to pass on a hand-written answer and fail on the untouched file. **First run 3 of 6, three false finishes, one shape**: no document the request named was read before the script that changed it — the minutes guessed to start with `Task:` — then exit 0, nothing printed, done. **Showing the changed document in `run`'s result did nothing** (2 of 6, 3 of 4 held out, three false finishes): *제2조 계약 금액은 5,000,000원* sat in the result of the command meant to change it, and it finished — reverted. **What worked is the program reading first**: `Tools::opening`, the calls made before the model's first step and recorded as steps — the workbench names every document the request names, documents only, since text files it reads itself. **7 of 10 and 2 of 3 held out, then 9 of 10 and 3 of 3 once the reader stopped printing `[Heading 1] ` in front of the text**, which a script had searched for. One false finish (a regex writing `1. \1. 배경`) misses the bar of none; kept on the comparison, **5 of 10 and three false finishes without it**, and that is the author overruling their own bar, written down to be overruled back. `--bin complex` with the new tool text: **12 of 12, no false finish**, unchanged. One run each. **And what a task made is shown, 2026-11-07** — the half of D that was left: the program snapshots the work folder before a task and finds what changed after it (`argmind_serve::deliver`, not the steps, since a `.docx` is written by a script no step names), reads each through the workbench's own `read_file`, and writes them into the pane on `about:blank` as escaped text under a policy that refuses scripts. `--bin browsing --show`, no model, four checks written first: the three made files found and the untouched one not, a Markdown line and a Word paragraph on the page, **a `<script>` and an `onerror` shown as text with the title unchanged**, and the pane opening the web after — all four on the second run; the first failed the last on the check's own string. **Not run inside the application from here** |
| research into a plan | **the owner's trip request never reached the task loop, and once clicked a Git button.** *11월 일본여행 계획 2박3일로 성인2명 항공권이랑 호텔 조사해서 적절한 스케쥴을 세워줘*, three times: the word rule counted files and code as work and not research, so it went to an ordinary turn, which answered *찾아볼까요?* — and with 확인 없이 실행 on, the action beside those words was *Checkout Branch/Tag* in the editor. Now research and plans are work (`asks_for_work`), the task loop has `web_search` and `open_page` in argmind's pane (`WebTools`, `Joined`), and **a reply that asks — *…까요?* — leaves its actions as rows** (`converse::asks_first`). `--bin browsing --plan`, three requests, the owner's word for word, passing on *finished, searched, wrote a file, every number in it on a page it read*: **2 of 3, and the trip plan's hotel prices — 150,000 and 180,000 — on no page**, taken because the previous fix had made everything the work wrote a source for its own numbers. Code is a source now and a document is checked like the closing words; and a price in 만 is read as one (*48만원* is two digits and went past a rule that read three). **Re-run, 1 of 3, and both misses read differently**: the trip plan came back in the closing words with no file — legitimate for a request that named none, failed by a criterion that wanted one — with its three-digit numbers all on pages; the phone comparison wrote *추정 140만 원* and **was stopped**, the refusal working. What nothing here catches: a number that is on the page and belongs to something else — *iPhone 16, 6.7인치* is the 16 Plus. A Naver results page is not a fare search; the plan's flight line is a range from snippets. **And then it was a fare search** — the owner: *왜 조회를 못하지? 회원로그인이 필요한 부분도 아닌데.* Opened by address in the pane, Naver's flight search gave 60 price lines in 2 s and Skyscanner did not refuse; argmind had never gone there, and had it gone, the tool returned the first 1,800 characters, which on that page are the search form. So `search_flights` and `search_hotels`, **the address built by the program from values the model gives** (the click's name-and-role split again), the text read once it stops growing — **not before 10 s for hotels**, whose list held still at 2,362 characters for six seconds before its prices came and was first read there, names and no price — and results only, one fare or hotel a line. Re-run: the trip plan used them — *이스타항공 직항 369,700원*, *렘 롯폰기 496,290원*, both on the pages — **and is thin**: no flight times, and *상세히 기록 완료* with no file written, a claim nothing checks. Two more of the check's own faults, found by the phone comparison: *15.5cm* read as a file named `15`, and *114만 5천 원* as 1,140,000. Whether a fare is for one adult or two, and a hotel's for one night or both, is not read off the page by anything here |
| routing | **the word rules miss 16% of what they are given, and almost every miss is work left as a turn.** `--bin routing`, three sources reported apart: the owner's own sentences (7 — the five in `conversation.jsonl` and two from pasted timelines), borrowed (`goals.json`'s 37 and `talking.json`'s 17, whose path was a turn before this existed) and written on 2026-11-03 with their accepted paths set down first, traps included. **Rules: real 7 of 7 — fitted, since today's change was made on those very sentences — written 39 of 48 with 8 harmful, borrowed 54 of 54.** Seven pieces of work went to a turn (*블로그 글 하나 써줘*, *노트북 세 개 비교해서 추천해줘*, *사진 폴더를 날짜별로 정리해줘*…) and *문서 작성 잘하는 법 알려줘* went to work. Under the 20% bar written first, so *the rules alone will do* — read that as the bar being loose, not the rules being good. **qwen3:8b as the router**, three repeats, identical every time: alone 50 of 55 and rules-then-model **52, harmful 3 against 8 — and borrowed 48 of 54**, under the 95% that kept the turn path whole, so neither replaced the rules. The misses were apps it could not see: *글씨 쓰고 싶어* in Paint went to work, *비밀번호 저장해* to keep. **Shown the window in front, borrowed went to 52 of 54 and rules-then-model met every criterion** — an arm designed on the failures it is judged on. What it still does: *계획 세우는 거 잘해?*, *일정 관리 어떻게 해?* and *문서 작성 잘하는 법* to work, three times each — a question about how starts a task. **Then wired, the same day**: `route::routed` is the rules, then the model shown the window and **read for `work` only** — every other path has a word rule that does the moving — and never on a question about how (`task::asks_about`, *…법 알려줘* or a question with no 줘/줄 in it, which the rules obey too). Both the window and the guard were made from this harness's misses, so **fourteen held-out sentences were written after the guard and before any arm saw them**, with the criteria: **0 of 8 how-questions to work and 6 of 6 requests**, three repeats identical; real + written **54 of 55, harmful 0 against the rules' 7**, borrowed 53 of 54 (*램 좀 늘려줘* goes to work). Read the held-out set for what it is: written by the guard's author, so its questions are the guard's shape. **The cost is a second call before the turn: 0.15 s median, 0.23 s worst over 216** — the exception to *one call, not two*, taken knowingly; the timeline's 길 line prints the model's word and its seconds. `argmind_serve::route` is the one place `think` and the harness both read |
| mcp | **a task uses an MCP server's tools from the list alone, and only when there is one** — the owner, 2026-11-08: *문서작성이나 도면 설계는 코딩이나 mcp로 접근하는 방법도 있음*, on the hidden desktop's buttons being the last way into a program rather than the first. `argmind-env-mcp` starts the servers `%LOCALAPPDATA%\argmind\mcp.json` names (Claude Desktop's format; **absent at rest**) on the task's thread, and each tool becomes `server__tool` — `__`, because `blender.render` reads as a file name to the finish check. A server runs as the person, outside the sandbox: `run_windows`'s standing. **No new dependency.** `--bin mcp`, a stand-in CAD and parts server, qwen3:14b reasoning: **with it 2 of 2 through `cad__` calls, the control 0 on the parts nobody could guess** — and the control's drawing passed the coding route, an SVG by `write_file`, once the finish check stopped reading `w3.org/2000/svg` as an invented 2000. One run each. **No real server was run from here**: `npx` fetching the reference server was refused by this session's permissions, so a real program's MCP server is unmeasured |
| ways in | **the owner's order — MCP or an API, then code, then a screen — holds in 3 of 3, on the second run.** *작업 수행시 접근 우선순위는 mcp또는 api, 코드또는 스크립트 작성및 실행, 그래도 안되면 샌드박스작업 수행* (2026-11-08). `task::Tiered` lists a task's tools by rank with the rank in each description, and refuses the first `screen_open` of a task in which no API or code tool has been used — asked again, it opens. **The screen is `argmind_env_windows::screen`, a Windows desktop of argmind's own** (`CreateDesktopW`), operated through control patterns only: `screen_open` (any installed program, *설치된건 아무거나*, except one already running — a single-instance program would join the person's — and packaged launchers, whose window may open on the person's desktop: **not tried, because trying it is the harm**), `screen_look` numbered with no rectangle, `screen_press`, `screen_type` (replaces a value; a password box refused), `screen_close`. **No keys** — a save is the 저장 button. Driven by hand on `order-window.exe`: open 0.8 s, look 10–50 ms, the answer read, foreground and physical pointer unchanged. `--bin screen`, qwen3:14b reasoning, criteria first: a window whose answer appears only after 조회, the CAD server's plate, a font list with Character Map and PowerShell both there. **First run 2 of 3** — the order held in all three, and the font list was written as the record showed it, *…(앞 5521자 생략)* and the tail; a call carrying a cut result is now refused (fitted to that case) and the **re-run read 3 of 3**. **The gate on `screen_open` never fired with a model** — every task that opened a screen had tried code first. **And it can be watched, 2026-11-08** — the owner's *작업화면 보여주는 칸*: the worker draws the work screen's windows with `PrintWindow` from its own thread after every job and once a second idle, at most 800 wide, into a `View` that `platform.rs` alone holds; the page asks `/work-screen/<n>` once a second and gets a `.bmp` only when the picture changed, and a 작업 화면 fold shows it while something is open there. **The picture is for the person and never a model** — `argmind-rules/tests/the_work_screen_is_shown_only_to_the_person.rs`, one route reads it, the core never names it, four breaks caught. Checked by hand (the order window after 조회, read back as a picture, unchanged over 2.5 s idle, gone on close) and by an ignored test in `platform.rs` whose first version skipped on an open Character Map and passed in 0.02 s having checked nothing. **The fold has not been seen in the application from here.** **And a program opened there dies with argmind**, started suspended and put in the job the servers are in: three Character Maps left by killed probe runs were still running on a desktop nobody could see. `screen-drive --leave` opens one and exits without closing it, and nothing is left — the run without the job was not made, because what it would leave could not be ended from this session. **And the person's 규칙 reach it, 2026-11-08**: a screen press or write is not a `Proposal` and never reaches `perform`, so the work screen asks a `Gate` before each one with the window's title, `click` or `type`, and what it is — the application answers from `desk.rules` and `acting` (`rules::on_the_work_screen`): *하지 않음* refuses with the person's sentence as the tool result, *항상 물어봄* — **or 확인 없이 실행 off**, the standing a task's pane press already read — raises a question whose one button is yes (`About::Press`), waited on like a row, 멈춰 letting go. Checked: the decision both ways and the strictest winning in either order, two breaks caught; `screen-drive --gated` wrote the order number, had 조회 refused and the waybill never appeared, with the ungated run as control. **The question was not seen in the application from here**, and `screen_open` is not gated — a rule names a window title and a program has none until it is open. With the owner's settings today (확인 없이 실행 on, no rules) nothing changes. Not run inside the application |
| installed programs | **Blender and KiCad by their own scripting interfaces, 3 of 3 against 0 of 3 without — after two fixes fitted to the first two runs.** The owner chose *connect a real program* (2026-11-08); this machine has Blender 5.1, KiCad 10 and Office. `argmind_env_windows::programs` lists `blender_python` (a `.py` in the work folder run by `blender --background --factory-startup`) and `kicad_cli` (arguments refused when one names a place outside the folder) **only when installed**, ranked with the APIs; Office is left out because COM may join the Word the person has open. Run by hand: a 2×1×0.5 box exported in 6 s, a demo BOM in 1.7 s. `--bin programs`, qwen3:14b reasoning, every expected answer computed by running `kicad-cli` on KiCad's own demo: first **1 of 3** — the model passed the argument list bare and the tool refused six right calls; then **2 of 3** — the BOM exported and the task finished on *총 47개* without reading it, 11 being right and two digits passing the finish check; with a short text output shown in the result, **3 of 3**. **The control, with the same workbench and screen, 0 of 3 and every one an honest *cannot***: it parsed the schematic for 1,006 s and found nothing, and never went looking for the programs on Windows. One run each. Both programs run as the person outside the sandbox, `run_windows`'s standing, and every command `finish` starts now joins argmind's job. **The owner's first box in the application, 2026-11-09, failed and could not be seen failing**: *해 볼게요*, then four minutes with nothing on screen while the steps went to the closed 한 일 fold — the script called `bpy.ops.export_mesh.stl`, gone since Blender 4, got *could not be found*, and was run twice more unchanged. So every task is now a card — the owner's next word, *작업별로 진행상황이랑 완료여부를 간단하게* — the last five in memory, sent whole as a `tasks` event on every step and to a page opened mid-task: what was asked, steps, the last tool and its first line or how it ended, how long ago, 멈추기 or 치우기. **Floating over the sphere in the quiet view, top-left, and set above the input in the full view**, because nothing riding on the sphere goes in front of the conversation; the sphere carries a ring while one runs and the `acted` or `refused` jolt when it ends — what the cards say, in a shape. Checked by a headless screenshot of both views with fixed cards, not in the application. **Then run in the application, the box sent as UTF-8 to `/say`**: a card from 0 to done in 2 min 50 s, five steps — the old `box.py` failed, the hint's name went in, the old exporter's `use_selection` failed, fixed from the traceback alone, exported — and the file read back at 0.04 × 0.02 × 0.01, **metres, which an STL reader takes as millimetres**: a slicer opens a 0.04 mm box, and `--bin programs` accepted either unit. The card showed *오류: [stderr]* for both failures, the first line of every failed command, so a failed step now shows the line that names its error (`app::gist`). **The first attempt sent the sentence through Git Bash and it arrived as `?? 40mm ?? 20mm …`**: routed to a turn, answered *상자를 만들어 드릴게요*, and with 확인 없이 실행 on it pressed VS Code's *Synchronize Changes* and Chrome's *Bookmarks* — both read *화면이 바뀌지 않음* and the repository was unchanged, checked; a request nobody can read was acted on. **All three answered the same day**: a sentence whose `?` runs start words — a garbled syllable, not a question mark — at least three and a quarter of what was written, or three replacement characters, is asked again with no model call and nothing done (`converse::unreadable`, the arrived sentence as its test, *뭐해???* and *왜? 왜?* as the controls); the tool text says to build in metres and export with `global_scale=1000` when sizes come in millimetres, and **a run's result carries every new STL's sides as the program measured them from the file** — `--bin programs` now passes *box* only at 40 × 20 × 10 and adds *coin* (30 × 30 × 5), written before the run, bar 4 of 4: **4 of 4**, both in millimetres, both reading the measured sides before finishing. One run. And a 작업 폴더 button beside ⋯, and 폴더 on a finished card, open the work folder in the person's file manager (`/work-folder/open`, a POST; `platform::open_work_folder`) — the person's button, so it goes through nothing that acts. **Then code passed inline, and the tool text's oldest claim fell**: `blender_python` takes code as code (`code`, a bare string, or `script` when it is plainly not a name) and saves it to `blender_script.py` through the workbench's own `write_file` — fitted to the two steps the 4 of 4 run lost — and the next run read **2 of 4**: the text had said *빈 장면에서 시작한다* since the first day while `--factory-startup` brings a camera, a light and a 2 m cube, every passing run had deleted them itself, and inline code did not, so both STLs measured 2000 × 2000 × 2000. The hand check's first STL had been two cubes by its size, unread. **And `erc` had passed a swapped answer**, *오류 37개, 경고 0개*, since the harness took both numbers anywhere; it reads the number after each word now. The scene is emptied by the program (`read_factory_settings(use_empty=True)` before the script — objects `['Camera', 'Cube', 'Light']` without, `[]` with) and the re-run read **4 of 4, *box* and *coin* in one step each, failed Blender calls 0 against 2**. One run. **The owner's work-screen test the same afternoon failed three ways at once**: *작업 화면에서 문자표(charmap)를 열어서 ★ …* went to a turn (the router's model said *desktop*), which asked *작업 화면에서 실행할까요?*; the owner's *실행해줘* went to work **as a task whose whole request was *실행해줘***, which found the old `box.py` and made the box again; and its finish was refused — *the last command failed* — because a failed `python3 box.py` was followed by a `blender_python` that worked and only `run…` counted as running. So: naming the work screen is work (`asks_for_work`); a bare go-ahead (`converse::goes_ahead`) with a row waiting stays the turn's to accept, and sent to work carries the request the last reply asked about (`converse::offer_answered`) or starts nothing; `why_not_done` counts any program or MCP tool as running; and a request that names the work screen skips the order's first refusal (`Tiered::screen_asked_for`), since otherwise the task tries PowerShell first — which opens a program on the person's own desktop. Each checked by breaking it; `routing`'s rules unchanged at 47 of 55 and 54 of 54. **Not run in the application yet**, and a gone operator's error carries its replacement, from a table checked against Blender 5.1's own operators — none of it measured with a model |
| the seam | **a turn given work refuses it rather than promising it, so the seam was not wired — caught 0 of 23 in three repeats, harm 0.** Item A, *one path*, collides taken literally with two of the owner's decisions (a turn presses the desktop and a task only the pane; 0.6 s on 8b against seconds a step on 14b), so the owner's *진행* went to the alternative: hand a turn to the task loop when its own reply shows it cannot finish in a look. `task::promises_work` was written before the run from the three real replies that showed it (*찾아볼까요?*, *검색해볼까요?*, *잠시만 기다려주세요!*); `--bin seam` gives every sentence of `routing`'s set (now `argmind_measure::routed`, moved unchanged) to a real turn as if the router had sent it there, criteria first. **At rest every one of the 23 work sentences was declined** — *이 화면에서는 할 수 없습니다*, *도구가 보이지 않아요* — the owner's trip requests among them: the promise came from the application, with the pane up and offers already in the history. A refusal rule would be fitted to this run and must leave the nine right declines alone; not attempted. The router in front stays the only way in. `promises_work` is pinned by its test and called by nothing |
| counts | **a count in the closing words must stand whole in something run or read — it caught one invented count, and the bar written for it was missed twice by failures it took no part in.** From `--bin programs`' *총 47개* (11 were right; two digits pass the number check, which reads three and matches inside longer numbers). `task::counts` reads a number with a counting word after it (*11개*, *37건*, *5명* — not *10mm*, *3번째*, *3개월*) and `why_not_done` refuses one that no tool result, the request, the date or code printed as a whole number: *count it with a command*. Criteria written first: programs 3 of 3, complex 12 of 12 with no false finish, documents ≥ 12 of 13 with ≤ 1. **Read: documents 13 of 13, none false; complex 11 of 12, none false; programs 2 of 3.** The check never fired in complex or documents; complex's miss was the old three-digit check on *60000* in the CSV task, and programs' was *box* inventing `bpy.ops.mesh.primitive_box_add` and failing three runs — that case passed the two runs before. It fired once: *erc* closing on *오류 23개, 경고 14개*, refused, the report read, *0개 / 37개*. **Shipped against its own bar, by its author, with the reason above** — overrule it back if that reads as special pleading. The finish check still cannot catch a count that happens to be printed somewhere for something else |

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

**Every number in that paragraph is about four coloured squares, and `--bin seeing` asked the
other question on 2026-09-23.** It takes **one** photograph and shrinks it for each size, because
four captures would be four instants and *the small one missed the person* would be indistinguishable
from *the person moved*. **Nothing is saved**: the captions reach the terminal and the picture dies
with the process, since a stored photograph is what would let somebody check a caption later and
that is the artefact this is trying not to make — which also means **the only ground truth is the
person in front of the lens**, and both runs were read back to them.

**It can describe a room, and 320×180 is the size.** Presence and a named object hold at 43 kB;
**attention needs 320×180**; and **nothing measured gives eyes open or shut** — the full-resolution
arm claimed *눈을 감고* twice on open eyes. The 8 s is also gone: warm, every size is 4–5 s and the
long first read was the model loading. `docs/findings/seeing-a-room.md` has it, including the
control that had to exist first — *somebody is there, facing the screen* is guessable in front of a
webcam, so a caption is not evidence until pictures with no room in them decline, which they do
**0 of 3**. **What this does not answer is whether attention is worth 5 s on a 30 s tick** — and since
2026-09-23 there is a second way to have it. `argmind-env-face` reads **a turned head from a facing
one at 99 ms with no model**, thirty times apart at the tails, both orders agreeing; the licence
gate did the choosing, since InsightFace's weights are non-commercial and YuNet (MIT) with SFace
(Apache-2.0) under `tract` is pure Rust and adds no DLL. **It is a head and not a gaze**, which is
the one case the model can see and this cannot — and pricing that case is exactly what the leading
question voided. `camera::MOVING` still answers *is somebody at the desk* in 32 ms with neither.

**And the cheap one is wired, as a reading and not as a gate.** `attention` is the fourteenth
decidable key and `설정 → 화면을 보고 있는지 알기` is the switch; three switches sit under the camera
because they are three decisions — the light, the head, the roster — and only the third leaves
anything behind. The reading rides the movement loop rather than a second thread on the same device,
a face pass every `FACE_PASS = 1 s` giving about three readings a window, and the window is
classified by a **median**, because the constant came from medians and using it on one frame would be
using it against a statistic it was not derived from. **Three readings and the third is not a vote
for the second**: 화면 쪽을 봄 / 고개를 돌림 / 고개 방향을 읽지 못함, and `Facing` carries four states
rather than an `Option<bool>` so that *nobody asked* and *could not be read* stay apart. **Nothing is
added at rest and `camera_line` is a function so a test can say so** — with the switch off the
blackboard line is byte for byte what it was, which is the only ground a new prompt slot could have
had. **It gates nothing**: `somebody_about` still asks about movement alone, because nothing has
measured that a turned head should change what argmind does on a tick. **Confirmed live by the owner on
2026-09-30 — 화면 보고 고개돌리는건 됨** — the first run of it inside the application rather than
a harness, and it took two fixes to get there: the camera line had never been shown to anybody, and
the permission question rose above the settings panel while the switch sat deep inside it, so
ticking the box unticked it and put the question off screen.

**Asked to judge it rather than gate it, and this model cannot — measured 2026-09-28.** The owner:
말을 걸지말지는 상황에 맞게 판단하면 됨. Telling the model the fact where it decides to speak first
rested on silence being an answer it gives there, and `--bin turned` found **224 of 224 spoke**
across four arms and two instructions — including one that said *otherwise leave "say" empty*. The
reader manufactures nothing, so that is the model; `noticing` recorded the same 8 of 8 long ago.
**The silence a person gets on this path has always come from `said_before`.** Worse, a sentence
about the person became a sentence *to* them — *You turned your head, I noticed* — and the neutral
control drew the most of it, 18 of 28, so it is the line's presence and not its content. Telling it
to be quiet moved wanting-to-click 5 → 9 of 28, `noticing`'s old finding again. **Not wired**;
`watch::regarding` stays as the harness's shared builder with the reason written on it, and the
shipped prompt was never touched — the counterfactual swapped the line inside the harness only.

**So the judgement moved into the program, as 눈치** — the owner's idea, 눈치에 대한 수치를
조절하는것도 방법일듯. It sits with the four dials in `설정 → 생각` and is **not one of them**: those
reach the model as a sentence and this is never shown to it, because a 눈치 line would be the same
experiment a third time with a slider claiming it worked. It does not live in `mind.json` either,
whose writer would erase it the next time somebody turned 유머; `tact.json` is its own file. **At 0 it
is today exactly.** From 1 up, for unprompted remarks only: a remark to somebody turned away is
**held and said when they look back**, dropped if the window changed meanwhile, one at a time with
the newest winning; and **N × 10 s** separates argmind's own remarks. **No evidence never means
silence** — held only on a fresh positive *turned away*, released on anything else, and the reading
is cleared when the camera closes. Every number is chosen, and whether it feels like tact is not
something a harness here can say. `argmind-serve/src/tact.rs`, 8 tests.

**Face memory split one person into six rows, and the first cause named for it was wrong.** The
owner, 2026-09-30: *얼굴을 치웠다 다시 보이면 이름을 또 물어봄.* The roster held six rows for one
face, their prints at cosine 0.01–0.36 against a 0.363 threshold. OpenCV was read rather than
recalled — `FaceRecognizerSF::feature` uses `swapRB = true`, so SFace wants RGB and this tree fed it
BGR, nearest-sampled where `warpAffine` is bilinear — and that was fixed and **committed as the
cause**. `--bin recognising` then refuted it on the owner's own face: sitting still, **0.715 median
the new way and 0.714 the old**, 152 of 153 pairs over threshold either way. The fix matches the
reference and stays; **the causal claim is withdrawn.** What that run could not see is the owner's
actual condition — a face leaving the frame and returning, embedded every three seconds whatever
state it is in — so the harness now has three phases and a candidate gate (facing, inside the frame,
detector sure), with pass and fail counted apart. Nothing about the gate ships until it has run.

**It ran, and the gate was the wrong gate.** Facing 20 of 20, **turning the head 20 of 20**, leaning
out and back **8 of 13** — the bad frames are the arriving and leaving ones, and the gate let them
through about as often as it stopped them (81% of passing frames over threshold against 88% of
failing) while discarding sixteen good turned-head frames to do it. It did not ship. The same run
checked the other assumption and **refuted that too**: an old-way print against the new-way centre
read 0.880, 20 of 20, so dropping old rows — which would have deleted a name seen 161 times — was
reverted before it ever ran. **What the bad frames do share is that they disagree with each other**:
by then the roster held nine rows for one face, 36 of 36 pairs under 0.363, the closest 0.3628. So
`argmind_core::recognise` — core because the harness and the application must run one rule, and the
roster compiles where the face models do not — enrols a new face only when **two consecutive
unknown frames agree**, keeps their mean rather than one arbitrary frame, and lets a known face clear
whatever was held. And the question after enrolment now asks about **the row just made**, where it
had asked about the oldest unnamed one — a stranger from last week, with stray rows present.
`--bin recognising` replays every recorded frame through that rule and the old one and counts rows
for one person; that number is what the rule stands or falls on. **It held, once**: the third run
replayed 10 frames at the application's cadence and 53 at every frame, and the old rule made **2
rows** for one person both ways where the new rule made **1**. One session and a one-row difference
— the evidence that matters next is the owner not being asked again. **The same run printed *ship
the gate*** on 14 of 14 passing against 17 of 19 failing — two frames, a run after the same gate
failed 81% to 88%, and still discarding 18 turned-head frames that matched — and it did not ship;
the harness's verdict was too easy to earn and now needs a 25-point gap and never says *ship*.
Old-way prints against the new centre: 0.878, 19 of 20.

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

**Input counts as somebody there — since 2026-10-21.** The owner typed to argmind while the camera
read *601초째 아무 움직임 없음 · 얼굴 안 보임*, and a finished errand — *16.0°C, 어제보다 2.7°C 높은
편* — sat unreported, because the room was *empty*. `present` now takes the seconds since the last
keyboard or mouse input, from `GetLastInputInfo`: **a time and nothing else**, no hook and no
characters, the *typing activity* this file set apart from a keylogger. It only ever adds *present*;
with the camera off the gate is still *no evidence*.

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
| `docs/findings/seeing-a-room.md` | **what a vision model can say about the actual room**: the three sizes for three questions, the largest arm inventing eyes it could not see, the control that had to exist because most of a correct caption is guessable, and the withdrawn reading that blamed tokens for a cold start |
| `docs/findings/attention.md` | **the cheap half of *are they looking at the screen***: the thirty-fold separation and its order control, the doc comment the run refuted, why the threshold is geometry rather than a midpoint, unreadable as a signal, and the leading question that voided the arm it was meant to compare against |
| `docs/findings/thinking-continuously.md` | **the note**: what a turn never carried, the three things the first run got wrong, and the watch that was reading argmind itself |
| `docs/findings/what-it-is-trying-to-be.md` | **the dials**: the rule that made a sixth prompt edit safe, the control that overturned the first reading, and the language of the slot |
| `native/argmind-measure/src/bin/crowding.rs` | **the sandbox's gate, and it failed**: 22 of 28 down to 3–8, the three ways padding could flatter itself, the truncation that would have made it a measurement of something else, and the thirty-five minute hang that was argmind holding the model at its own context |
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
