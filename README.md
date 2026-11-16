# argmind

**A local-model agent you talk to while you work at your PC.** The models run on your machine
through Ollama, so what argmind reads off your screen, hears and says is not sent to a model
service. It reads the screen through the operating system's accessibility layer, talks and listens,
carries work through to the end in a folder of its own, and acts on your desktop only as far as
the switches you set allow.

A side project by one person, not a product, with no support commitment. **Windows only for now.
The interface is Korean.** argmind answers in the language you write in, or always in Korean,
English or Chinese if you choose one in 설정 → 생각 → 대화 언어. The sentences the program writes
itself (refusals, the activity timeline) stay Korean.

## What it does

- **Talks while you work.** Type, or hold F9 in its window and speak. A reply takes about 0.6 s
  (measured median, qwen3:8b, warm) and can be spoken in a neural voice it builds and runs on your
  GPU, falling back to the Windows voice.
- **Sees the screen as names, not pixels.** UI Automation lists what is on screen, the model picks
  one by number, and the program finds it again just before acting.
- **Acts on your desktop**: clicks, typing and keys. It goes through control patterns, or a
  message posted to the program's own window, and uses your pointer and keyboard only when neither
  works.
- **Carries work to the end.** *Write and run a script*, *summarise this into a Word file* and *fix
  the failing test* go to a task loop on qwen3:14b. The model picks one tool a step, the program
  runs it and records what came back word for word, and a claimed finish is checked against the
  files and that record before it is accepted.
  - **Tools:** files in `%USERPROFILE%\argmind-work`, commands in a WSL sandbox, PowerShell,
    Blender and KiCad through their scripting interfaces when they are installed, and MCP servers
    you configure.
  - **For programs that only have buttons:** a hidden desktop of argmind's own.
- **Has its own browser pane** inside its window, driven over DevTools with no debugging port, for
  looking things up.
- **Notices things**, such as a window change or a dialog, and may say something or offer to act.
- **Works on a goal by itself** when you have set one and nothing else is going on.
- **Optional camera senses**: *movement seen N seconds ago*, and *head facing or turned*. No
  picture is kept. Off unless you turn them on.

## The one rule

> **The deterministic side says WHERE. The probabilistic side says WHAT.
> A model never emits a coordinate.**

This was measured on this machine before any of it was written:

- **A vision model asked for a pixel** took **24–82 s**, and on a dense screen never answered.
- **UI Automation** answers the same question exactly in **41–266 ms**.

So the operating system supplies the names and the rectangles, and the model only ever picks a
name. That rule is enforced by tests that read the source: nothing in the core asks a model for
`x`/`y`, and nothing rendered into a prompt mentions a rectangle.

## Safety: what it may do, and the defaults

argmind presses buttons on your computer. Read this before turning things on.

- **At rest, only watching is on.** Everything below waits for your answer in 설정 (Settings).
  Your answer is written to `%LOCALAPPDATA%\argmind\decisions.json`. Refusing always wins.
  - *확인 없이 실행*: act on what you asked for without confirmation.
  - *스스로 실행*: act on argmind's own ideas.
  - The microphone, the camera and the voice.
  - Every install it offers.
- **Until you turn on 확인 없이 실행, every action waits for a press**: it is shown as a row with a
  실행 (Run) button.
- **The person keeps some things for themselves:**
  - **Rules (규칙), per app and per action**: *하지 않음* (never) or *항상 물어봄* (always ask). The
    strictest rule wins.
  - **Passwords:** a password field is never typed into. You type it.
- **Every action is checked against the screen:**
  - It is re-resolved just before it lands.
  - A step that fails stops the steps after it.
  - A name that matches more than one control is asked about, not guessed.
  - A reply phrased as a question (*…할까요?*) leaves its action as a row instead of running it.
- **The local server** listens on loopback only and wants a token minted each run. It refuses any
  request addressed to a host name other than this machine's, so a web page cannot reach it by
  DNS rebinding.
- **Commands a task runs** are confined, one way stronger than the other:
  - Linux commands go to a WSL distro with drive mounting and Windows interop switched off, and
    only the work folder mounted.
  - PowerShell runs on a desktop and window station of argmind's own. This is weaker: a Windows
    process is not confined to a folder.

**What it cannot promise:** a plan that is wrong but well-formed gets through. Typing into a field
that exists, on a screen that passed every check, is still typing in the wrong place. And
qwen3:8b picks the right control 21 times in 28 when there is one (below).

## What leaves the machine

**Nothing is sent to a model service.** The network is used for:

- downloading the models and installers argmind asks you about;
- building the speech recogniser and the voice server from source, at pinned commits;
- pages its browser pane opens (searches go to Naver);
- whatever commands a task runs (the sandbox has a network);
- MCP servers you configure.

**What it keeps** is in `%LOCALAPPDATA%\argmind\`, as files you can read and delete:

- settings and your answers: `decisions.json`, `devices.json`, `mind.json`, `tact.json`, `rules.json`;
- the conversation, `conversation.jsonl` (*대화 기억 지우기* empties it);
- handed-over errands, `errands.json`;
- the record of work it set itself, `pursuit.jsonl`;
- face prints, `faces.json`, only when you turn on recognising faces.

The activity timeline is kept in memory only and is gone at exit.

## Requirements

- **Windows 11**, which is what it is built and run on. Windows 10 is untested.
- **A GPU with about 16 GB of memory.** On an RTX 4090 Laptop (16 GB), the conversation model,
  the work model (partly on the card), the voice and the recogniser together use **12.6 GB**
  (measured). Less memory is untested.
- **[Ollama](https://ollama.com)**, which argmind offers to install.
  - `qwen3:8b` for conversation: argmind offers to pull it (about 5 GB).
  - `qwen3:14b` for work: argmind offers to pull it too (about 9 GB).
- **Rust stable, MSVC toolchain** (pinned by `rust-toolchain.toml`). The page is prebuilt in
  `ui/dist`, so building needs nothing else.
- **Optional:** the Vulkan SDK for the neural voice, and a C/C++ toolchain for the recogniser.
  `argmind check` says what is missing and what it would do about it.
- **For tasks that run code**, WSL 2. argmind then offers to make its sandbox, a distro of its own
  called `argmind-sandbox`, and asks first. It takes 1–2 minutes here and downloads about 390 MB.
  - It installs Ubuntu 24.04 from WSL's own list into `%LOCALAPPDATA%\argmind\sandbox`.
  - It writes this `/etc/wsl.conf`:

        [automount]
        enabled = false
        mountFsTab = false

        [interop]
        enabled = false
        appendWindowsPath = false

        [user]
        default = root

  - It installs `python3-docx` and `python3-openpyxl` without recommends.
  - Before calling it made, it checks that nothing of `C:` and no Windows program can be seen from
    inside. If anything is off, it removes what it made.

  Without WSL, run `wsl --install --no-distribution` in an administrator PowerShell and restart.

## Build and run

    cargo build --release --manifest-path native/Cargo.toml
    native\target\release\argmind-app.exe

`argmind-app.exe` with no arguments starts everything you have allowed. It puts an icon by the
clock and opens a window of its own. `--dry` presses nothing.

The command line, `native\target\release\argmind.exe`:

    argmind check                   what is missing, and what it offers to do about it
    argmind setup                   fetch what is missing, asking first
    argmind look                    what is on screen right now, and how long reading it took
    argmind talk                    a conversation; it clicks only with --live
    argmind say "이 창 닫아줘"        one exchange, then exit
    argmind watch --for build       speak when something happens, unasked

**Tests** need no display, model, network, microphone or speaker:

    cargo test --manifest-path native/Cargo.toml
    git config core.hooksPath .githooks    # once per clone: the pre-commit hook runs the suite

## Measured

Every number here has its conditions beside it, and most are one run. The full record is in
`CLAUDE.md`, the project's working notebook, and in `docs/findings/`. That includes the
conclusions that were withdrawn and the ways the measuring code was wrong.

| | |
|---|---|
| choosing a control | **21 of 28** answerable goals right and **7 of 9** impossible ones declined: 28 of 37 decisions. qwen3:8b with thinking off, 4 captured windows, three repeats, the worst counted |
| crowding | a list the size of a web page breaks choosing: **22 of 28 → 3–8 of 28** at 123–900 controls |
| talking | **31 of 34** conversational turns produce no action |
| language | with a language set, **48 of 48** replies in it for Korean, English and Chinese, asked in Korean or English, on two runs. A Korean sentence you asked it to remember pulls about one reply in twelve back to Korean |
| a reply | **0.61 s** median |
| work | **11 of 12** tasks, no false finish. qwen3:14b with reasoning on, about 148 s a task, one run; each task checked against files and the sandbox, never against the model's own account |
| documents | Word and Excel made, changed and read: **13 of 13**, none false, one run |
| installed programs | Blender and KiCad through their scripting: **3 of 3**, against **0 of 3** without them; one run each, and later runs on a changed set read 2 of 3 and 4 of 4 |
| a covered target | with something lying over the target, neither the pattern nor the click reaches it; uncovered, both do. **24 of 24** arms as predicted before the run |

## Layout

    native/argmind-core/         the contract (Widget, Eyes, Hands), the model, the session, tasks,
                                 errands, rules. Names no platform
    native/argmind-env-windows/  UI Automation for eyes; patterns, window messages and SendInput for
                                 hands; WASAPI, SAPI, the work desktop, the sandbox
    native/argmind-env-*/        the browser pane, Ollama, Whisper, the voice, MCP, Linux eyes, and
                                 an environment that can do nothing (the proof the app is separable)
    native/argmind-serve/        the application: a local server, the page, the tray, the CLI
    native/argmind-window/       its window, the tray icon, the browser pane
    native/argmind-rules/        rules the repository keeps by reading its own source
    native/argmind-measure/      every harness behind the numbers
    ui/                          the page, TypeScript with no dependency but tsc
    measure/                     goals, captured screens, and every recorded answer

## History

Until 2026-09-09 this name meant a recursive-least-squares library for on-device adaptation,
ported to C++ for Cortex-M. That tree was removed rather than left unused. It is intact under the
tags `before-removal-2026-09-09` and `argmind-v1`, and `REBUILD.md` says why. Until 2026-09-15 the
agent was Python. That tree is at `python-before-removal-2026-09-15`, and what it produced is
committed as data so the equivalence checks outlived it.

## Licence

Apache-2.0. See `LICENSE` and `NOTICE`, which also asks that the numbers be quoted with their
conditions. Every dependency's licence is recorded in `docs/licences.toml` and checked by a test.
Author: Younghyeon Park.
