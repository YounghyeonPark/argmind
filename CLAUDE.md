# Argmind — project context

**A local-model agent you talk to while you work at your PC.** An open-source LLM on the machine,
real-time interaction, actions taken through the OS accessibility layer. Author: Younghyeon Park
<ypark.dev@gmail.com>. Apache-2.0. A side project, not a product.

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
믿는 것 / 기다리는 것 / 이미 한 것 — revised every 30 s by **a call whose answer shape has no `say`
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
| the note | **it invents what it did, if you let it.** Asked for all three fields from a session in which argmind had done nothing: *이미 메모리 사용량이 높은 탭들을 목록으로 정리해두고…* — it had not, and it carried that for three revisions. `tried` is now a record rather than a question and the re-run leaves it empty for 100 s, correctly. It also **anchored** (a file manager on `C:` still read as *exploring files related to Younghyeon Park*) and **wrote in English**; both addressed, both re-run. **`waiting` is still generic** and nothing measures whether one is ever met |
| the watch was reading argmind | **4 of 8 notices over 150 s said `the window changed to 'argmind'`** — a model call each, spent on its own transcript. `attending.rs` had written the prediction and left it: *nothing here fires on a watch yet; when something does, this is the thing to revisit.* The old rule excluded whatever was in front, so working in Chrome after coming from argmind read argmind. Now `app.py`'s original: **the newest window that is not ours**, available again since the page moved into a window this process owns. **Predicted zero before the change; the re-run gave zero** |
| the dials | **the slot's language decides the reply's**, and that is the whole finding. 44 Korean requests: no line 28 of 44 answered in Korean, a content-free **English** sentence in the slot **19**, a content-free **Korean** one **42** and 41 on a repeat. The four dials in English sat at 28–31 — they moved 20–40 of 44 sentences and left the language alone. Korean lines ship, and at x3 the worst arm reads **Korean 39 of 44 against 28, leak 7 of 32 against 5**: +11 bought, 2 discordant cases spent, paid only by somebody who turns a dial. **The first reading was that the dials fixed the leak (5 of 32 → 0 or 1 in all four arms); the placebo fixed it too** |
| parroting | **it says your own words back when it cannot parse them.** 머해 → `머해?`, then the mirror is in the history and the next two turns get it again. Reproduced **at rest** — 2 of 6 echoed and 2 stuck with every dial off — so it is not the dials, and `{"say": "머해?"}` is what comes off the wire, so it is not the reader. `뭐해` is answered properly; `머해` is the colloquial spelling of it. **Asking it not to does not work**: told 상대가 한 말을 그대로 되돌려 말하지 말 것 it replied `머해`. Caught on the way out — *their words plus a question mark*, never plain equality, because `안녕` is a good answer to `안녕` — and **the substitute goes into the history**, which is the half that closed the loop: 2 stuck → 0 |
| clicking | **2 of 3, n=1** on the Rust, against the Python's 3 of 3 at n=3 on the same goals and the same condition — one Character Map for all three, so the second goal meets a screen the first has changed. One run against three and this model reads 21, 23, 24 across runs of one configuration, so it is a number rather than a regression. **The miss is the interesting part and it is not the model's**: asked for the font list it named `Open`, and advanced view has more than one button called that — what opened was the character set list. `name` and `role` together are still not a key, and the model is shown nothing else |
| obstruction | **24 of 24 arms agreed with the prediction written before the run**, 3 runs of 8. With something covering the target, *neither* path reaches it — the pattern resolves by `ElementFromPoint` and gets the cover, then falls back to a click the reach guard refuses. Stood aside or uncovered, both work. So the one case that could have earned the control patterns does not |
| the pointer | **the axis the patterns do win, 3 of 3.** Parked at `(700, 400)`: `invoke` leaves it there, a click drags it to the widget at `(130, 459)`, and both press the box. **The first version of this harness measured nothing** — it parked with `cursor_round_trip`, which puts the cursor back where it found it, so every arm read the position the previous arm's click had left and both reported *unmoved* |

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
**카메라** is **measured**: `qwen3-vl:8b` was already here, Apache-2.0, and a caption of a drawn
test card is accurate and takes **about 8 s** — too slow for the 0.7 s watch loop, fine for the 30 s
tick. The 24–82 s figure does not apply, it was a coordinate on a dense screen. It is not video
memory (both models sit on the GPU, the answer after a look is 0.2 s) and **not resolution**
(`prompt_tokens` is 1110 from 640×360 down to 80×46) — it is the model thinking, 467–1,690
characters of it about four coloured squares, and **`qwen3-vl:8b` ignores `think: false`** three
ways over. The lever is a non-reasoning vision model. **키입력** is implementable and deliberately not implemented — every keystroke means a
global hook, which is a keylogger capturing passwords in a program that sends text to a model; what
argmind plausibly wants is typing *activity*, which needs no characters and is a different feature
with a different permission.

No phone environment, no robot environment. **A thinking goal and four dials are built and are at rest until turned**: 재치 · 유머 · 봉사 · 헌신,
0–10, and a sentence in the person's own words, in `설정 → 생각`. At rest the prompt is what it
always was. **Voice is built and is off until switched on**: hold
F9 in the window, which means you have to be looking at argmind to talk to it. **The settings
decide whether it listens, not the command line** — a standing `microphone: allowed` turns hearing
on with no `--hear`, and a standing 거부 turns it off with one. Two places were reporting a
standing answer and not honouring it: that, and the installs behind 항상 허용, which ran only when
somebody pressed 다시 확인 — so after every restart the page said *물어보지 않고 진행합니다* beside
a recogniser that was not running. They now run on the first page open, once per process.

**Speaking has two voices and picks one at startup.** SAPI is free (4–43 ms), installed, and
`Microsoft Heami Desktop` — which the owner called 어색함 on 2026-09-16. **The neural one answers
that: 자연스러움, 2026-09-17**, judged from files of both saying the same three sentences rather
than from a demo. `cargo run -p argmind-serve --example voices` makes them again and plays
nothing. The other is a neural
synthesiser argmind builds from source and runs on the GPU: **27–238 ms to first sound** because
it streams, RTF 0.15, and it is used whenever it is answering on its port. If it is not, the old
one speaks rather than nothing. `docs/findings/a-better-voice.md` has the accounting, including
why the talker is Q8_0 and why the Vulkan SDK is asked for separately.

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
| `docs/findings/a-better-voice.md` | **the voice**: every open Korean TTS checked and its licence, and the measured answer — RTF 3.9 on the CPU, 0.15 on the GPU, and a K-quant that is thirty-two times faster and wrong |
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
