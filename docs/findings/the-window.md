# A window of its own — what it cost, 2026-09-16

The owner's instruction: **브라우저말고 단독앱으로 만들어줘**, and when asked how:
**멀티플랫폼으로 제작 윈도우 리눅스 macOS**.

What was there: a server on `127.0.0.1:7331`, a page, and `explorer <url>` to open it in whatever
browser the person had. A tab is not the program — closing it left a server running with nothing on
screen to say so, which is why there was a tray icon at all.

## The choice, and why the house style was the wrong answer here

Every other capability in this tree is hand-written against the platform: WASAPI, Media Foundation,
SAPI, UI Automation, raw HTTP/1.1, a chunked-transfer decoder. Three crates in the whole
repository. So the default answer was to write the WebView2 COM interfaces by hand.

**That answer is worse rather than harder**, and the measurement is what says so:

- The documented entry point is `CreateCoreWebView2EnvironmentWithOptions`, exported by
  `WebView2Loader.dll`. That DLL ships with the SDK rather than with Windows, so using it means
  argmind redistributes somebody's binary. whisper.cpp and qwentts.cpp are fetched and compiled
  here precisely so that it does not.
- The installed runtime's own `EmbeddedBrowserWebView.dll` exports exactly one thing:
  `CreateWebViewEnvironmentWithOptionsInternal`. **Measured with `objdump -p`** against runtime
  153.0.4234.32. It is what the loader calls, it is undocumented, and a program built on it breaks
  when the runtime updates under it.
- And neither of those is a window on macOS or Linux, which is half the instruction.

So `wry` (Apache-2.0 OR MIT), `tao` (Apache-2.0) and `tray-icon` (MIT OR Apache-2.0), the Tauri
project's own bindings. **81 crates enter the Windows build graph behind them.** Read across the
whole graph rather than the three: no GPL, no AGPL, no LGPL, no non-commercial. Five are MPL-2.0,
which is file-level copyleft reaching only modified copies of those files, and nothing here
modifies them. `argmind-rules` checks direct dependencies and says plainly that it does not follow
transitive ones, so that sweep was done by hand and is recorded rather than implied.

**And the redistribution argument above does not fully survive.** `webview2-com-sys` links
`WebView2LoaderStatic.lib` on MSVC — a prebuilt Microsoft static library, compiled into the
executable. Microsoft's loader code ships either way. What is actually preserved is narrower and
worth stating in its own words: **nothing has to sit beside the executable as a loose file, and the
runtime itself — the hundreds of megabytes of Chromium — is never redistributed.** It is found on
the machine, as Ollama is.

## What was verified

**The page renders and its script runs.** A throwaway server on `127.0.0.1:7777` served a page
whose only job was to `fetch('/ran')`. Both requests arrived, and the user agent it reported was
`Chrome/153.0.0.0 ... Edg/153` — the WebView2 runtime, not a browser:

    requests: ['/', '/ran?ua=...Edg/153.', '/favicon.ico']
    page fetched: True
    javascript ran: True

**It is a real top-level window owned by this process.** While it was open, one process named
`peek` with `MainWindowTitle = argmind`, and the rendering done by `msedgewebview2.exe` children
rather than by `msedge.exe`.

**Nothing is left behind.** After the process ended, `msedgewebview2.exe` processes whose command
line mentions argmind: **none**. The ones still running belonged to KakaoTalk, Google Drive,
Windows Search and Altium, which is a useful reminder that the count alone would have proved
nothing.

**The whole application, in one process**: server thread bound to 7331, window titled `argmind`,
six webview children, one PID.

## The failure that only a double-click finds

This is the part worth keeping.

`cargo run -p argmind-window --example peek` opened a window and rendered a page. Then
`argmind-app.exe`, started as a file, **exited immediately with no window, no message and nothing
in any log.** Its exit code was `-1073741515` = `0xC0000135` = *STATUS_DLL_NOT_FOUND*.

The difference is `WebView2Loader.dll`. `webview2-com-sys` links the loader statically **only when
`target_env = "msvc"`**; on a `-gnu` toolchain it links the DLL dynamically, and that DLL lives in
`target/*/build/webview2-com-sys-*/out/x64/` and nowhere a launched program looks. **`cargo run`
puts every build script's `rustc-link-search` path onto the child's `PATH`.** A double-click puts
none of them there.

**This repository already knew this defect.** `building::bring_runtime` exists because a MinGW
build of whisper.cpp ran from a shell and would have failed from the tray: *"argmind found the
server here because a shell had put them there, and a program started from the tray would have got
a missing-DLL box instead."* The same sentence, about argmind's own executable, four commits later.

And it was nearly missed twice: the first attempt to launch it printed
`api-ms-win-core-synch-l1-2-0.dll: cannot open shared object file`, which was read as noise from
running a Windows binary under a POSIX shell. It was the loader saying exactly what was wrong.

Two things came out of it:

- **`rust-toolchain.toml` pins `channel = "stable"`**, which rustup resolves against the *host* —
  `x86_64-pc-windows-msvc` here, even though the user's default toolchain is `-gnu`. Checked rather
  than assumed: `rustc -vV` reports `host: x86_64-pc-windows-msvc` with the file in place. A triple
  was deliberately not written, because that would pin Windows onto a Linux checkout.
- **`argmind-serve/tests/launchable.rs`**, which reads the built executable's imports with
  `objdump` and fails if any of them is neither beside the binary nor findable by the loader. Its
  own first version was wrong in the same family: it used `where.exe` for everything and failed a
  perfectly good binary over `api-ms-win-*`, which are **API set contract names resolved by the
  kernel and never files on disk.**

## The port, and what was actually blocking it — 2026-09-16

The owner's answer to *should this be Electron* was **아니라고 해도 멀티플랫폼에 대한 대책은
필요함**: not Electron, but parking it is not a plan either. So the plan was made, and the first
stage of it is done.

**Measured first, because the intuition was wrong.** The obvious reading is that the port is
`argmind-env-windows`, 1,844 lines, 15% of the tree. It is not:

| | lines | |
|---|---|---|
| `argmind-core` | 655 | names no platform; its only dependency is `serde_json` |
| `ui/` | 526 | HTML |
| `argmind-window` | 174 | wry already picks one of three |
| the three HTTP clients + rules | 644 | text and sockets |
| Ollama, whisper.cpp, qwentts.cpp | — | separate processes, already build on all three |
| **`argmind-env-windows`** | **1,844** | the part everyone thinks is the port |
| **`argmind-serve`** | **2,295** | **the part that actually was** |

`argmind-serve` named Windows types directly — `WindowsHands`, `SapiSpeaker`, `Attending`,
`Watcher`, `Microphone`, `argmind_env_windows::` — in **six files and twenty-one places**. Adding
an environment would not have been plugging something in; it would have been rewriting the
application. **That is what was blocking the port, and it is fixable on Windows.**

### What was done

`platform.rs` is the one file that knows. Everything else says `platform::Sight`, `platform::Touch`
and does not care. `argmind-env-none` is what it picks when there is no better answer — an
environment that sees nothing, presses nothing, makes no sound, and **refuses rather than
pretending**: `confirm` returns `false`, because with no window to ask in nobody agreed, and an
environment that answered yes would turn *cannot ask* into *said yes for you*.

**`Did` moved into `argmind-core`.** Two environments now have to report what an action did, and a
second copy of that enum is how they would come to mean different things — the argument this
project has already made twice this week, about `fix::apply` and about the tray.

### How it is checked without a second machine

This is the part that matters, because a claim about portability made on one operating system is
worth nothing.

**`--features headless` forces the empty environment even here.** `tests/portable.rs` then starts
the whole server on a port of its own, fetches the page over the socket, and reads what the build
says about itself — with no eyes, no hands, no voice and no microphone in the process. Seven tests,
all passing.

**And the compiler enumerated the interface rather than anybody guessing it.** Switching the
application over and building headless produced, one error at a time, the list of what an
environment must provide: not just `Eyes` and `Hands` but a way to know which window a person is
attending to, a hit test at a point, a microphone with an `available` flag, a supervised child, and
a yes-or-no question. **Whatever is in `argmind-env-none/src/lib.rs` is that list.**

Three rules hold it: `argmind-rules/tests/the_app_names_no_platform.rs` reads the source, the
pre-commit hook builds headless, and seven mutations were run against both — **seven caught**,
including the one that matters most, `confirm` returning `true`.

### What this did not do, said plainly

**It did not port anything.** There is still no macOS or Linux environment, and on those operating
systems argmind would start, serve its page, see nothing and press nothing. What changed is that
writing one is now adding a crate rather than rewriting the application.

**And it is the same shape of promise the Rust port made and did not keep.** That rewrite was
argued from *rules held by the compiler rather than by tests*, the `Listing` type was never built,
and `CLAUDE.md` says so. The difference here is that the claim is a binary fact a test can check —
the application builds and runs against an environment that is not Windows — so the test was
written first.

### It compiles for Linux, and that is measured rather than argued

**WSL was authorised and could not be installed.** The elevation prompt was cancelled twice; the
owner was following from another device and a UAC prompt lives on this machine's secure desktop,
where a remote reader cannot reach it. So the Linux evidence was got another way, and it turned out
to be most of what WSL would have given for this stage.

`rustup target add x86_64-unknown-linux-gnu` needs no administrator, and `cargo check --target` is
a real compile — the other branch of every `cfg` in the tree, including the manifest's
`[target.'cfg(windows)'.dependencies]`, which the `headless` feature does **not** exercise because
`cfg(windows)` is still true underneath it.

**Six crates compile for Linux unchanged**: `argmind-core`, `argmind-voice`, `argmind-env-none`,
`argmind-env-ollama`, `argmind-env-whisper`, `argmind-rules`.

**And so does the whole application** — the library and both console binaries — with the window
left out. `argmind-window` became an optional dependency behind a default-on `window` feature, and
`argmind-app` carries `required-features = ["window"]`, because only that binary ever used it. On
Windows nothing changes; with `--no-default-features` the rest compiles for Linux today.

Checked by breaking: a `use argmind_env_windows::` added to `app.rs` compiles fine here and fails
the Linux check with *unresolved import*. **That is a class the source-reading rule cannot catch**
— the rule sees names, and only a compiler knows what actually got linked. The pre-commit hook runs
both, and says so out loud when the Linux target is not installed rather than skipping quietly.

### It runs on Linux — 2026-09-17

WSL was installed on the second ask. Ubuntu 26.04.1, kernel 6.18, WSLg 1.0.73.2, which puts a Linux
window on the Windows desktop. `argmind-app` was built there from the repository — `git clone` of
the committed state rather than a copy of the working tree — and run.

**It works.** A `RAIL_WINDOW` titled `argmind (Ubuntu)`, 572×909, on the Windows desktop, with
`WebKitNetworkProcess` and `WebKitWebProcess` behind it and the page answering on 127.0.0.1:7331
**200 ms** after start. `argmind-app` is 105 MB debug, linked against `libwebkit2gtk-4.1`,
`libgtk-3`, `libsoup-3.0`, `libjavascriptcoregtk-4.1` and `libdbus-1`. The window's own tests pass
there unchanged.

The header reads **(아무 창도 보고 있지 않음)** — `argmind-env-none` answering honestly rather than
inventing a window. That is the seam working, seen rather than asserted.

### Four things that only running it found

**The tray killed the application.** `libappindicator-sys` does not return an error when
`libayatana-appindicator3.so.1` is absent — it **panics**, from inside a lazy initialiser. argmind
exited before drawing anything, four seconds after the server had already answered. The missing
library is a packaging fact; the bug was mine, because `run()` returned `Err` before the window was
shown. **The window is the program and the icon is a convenience**, and plenty of Linux desktops
have no system tray at all — that is not a broken machine. The tray is now built in its own
function, allowed to fail, and its panic is caught; with no icon the close button quits, which is
what `tray: false` already did.

**Every Korean character was a box.** The container has no CJK font and the entire interface is
Korean. `fonts-noto-cjk`, and the page renders. Nothing but running it would have said so — the
build was clean and the HTML was correct.

**Windows fixes were offered on Linux.** The `한국어 음성` row proposed
`DISM /Online /Add-Capability …` and said *Windows가 한 번 더 물어봅니다*. `needs.rs` and `fix.rs`
were behind the seam for their *types* and not for their *content*.

**Fixed the same day.** `platform::can_carry_out_fixes()` now gates every fix — reported and
offered are different things, and the difference is a button that has never worked. The two rows
that are Windows *facts* rather than Windows *fixes* are gone off Windows entirely. Re-checked on
Ubuntu: **없는 것 4개** where there were six, every row saying what to do by hand, no `DISM`, no
buttons, and the orb animating in WebKitGTK. Writing the rule found a third leak nobody had
noticed — the voice row said *Windows 음성으로 말합니다* on a machine with no voice at all.

**And AT-SPI2 is reachable here.** `org.a11y.Bus` is activatable on the session bus at
`/run/user/0/bus`, `at-spi2-core` is installed, and `pkg-config` reports `atspi-2` at 2.60.4. So
WSLg can host the eyes work and not only the window — which was the open question. *Activatable* is
not *working*: nothing has yet asked it for a widget tree, and WSLg has no window manager of its
own, so *which window is in front* is a separate question from *what is in this window*.

### What Linux wants, read off the failures rather than guessed

    build:    pkg-config  libdbus-1-dev  libgtk-3-dev  libwebkit2gtk-4.1-dev
              libsoup-3.0-dev  libjavascriptcoregtk-4.1-dev
    run:      libayatana-appindicator3-1  fonts-noto-cjk
    later:    libatspi2.0-dev (eyes)  libxdo-dev (hands)

`libdbus` and the appindicator are the tray's, not the window's — `tray-icon` speaks
StatusNotifierItem over D-Bus.

**One thing the workspace still cannot do on Linux**: `cargo build` at the root tries to build
`argmind-env-windows`, which is a workspace member, and fails with fifty errors. Naming the crates
works — `-p argmind-serve` builds everything that matters. Left as it is rather than papered over,
because the fix is a decision about workspace shape and nothing is blocked by it today.

### What the cross-check predicted, and what running it added

The cross-compile stopped at the window with

    pkg_config failed: pkg-config has not been configured to support cross-compilation
    libdbus-sys ... build.rs:25 panicked

and that list was right as far as it went — every package it named was needed. **What it
could not see was everything above**: the panic at run time, the missing font, the Windows
fixes on a Linux screen. A build that succeeds says nothing about a program that runs.

### The two things a machine is needed for

**Linux is two targets, not one.** X11 lets a program inject input with XTEST and read other
windows; **Wayland deliberately does not**, and routes both through `xdg-desktop-portal` with the
user's consent. That changes the permission model rather than the code, and argmind already has a
four-way permission system to hang it on. Which of the two to support, and what the portal
actually gives, is not something to decide from here.

**macOS needs one grant** in System Settings before the Accessibility API answers at all — which
fits `Fix::OpenPage`: open the settings pane, and the person allows it themselves.

**And the job object has no equivalent.** Linux has `PR_SET_PDEATHSIG`; macOS has neither. The
promise *the recogniser dies with argmind* is weaker off Windows, and `Supervised::supervised()`
already exists to say the weaker promise out loud — `argmind-env-none` returns `false` from it.

WSL is not installed on this machine and installing it needs administrator rights, so it was not
done.

## What is not verified

**macOS and Linux.** There is no Mac here and no WebKitGTK, and a `cargo check` for a target whose
system libraries are absent is not evidence. `argmind-window` names an operating system in exactly
one place — `attach`, because GTK puts the webview in a box inside the window and every other
backend puts it in the window — which makes the port plausible and does not make it done.

**And the agent inside the window is Windows-only regardless.** `argmind-core` names no platform;
its only dependency is `serde_json`. But the eyes are UI Automation, the hands are SendInput, the
microphone is WASAPI, the camera is Media Foundation, and both voices are Windows. On another
operating system this window opens onto a server that cannot start. **A window is the first piece
of that port and it is not the port**, and saying otherwise would be the flattering version.

## What the window does

| | |
|---|---|
| close button | hides it; the icon in the tray is what says the program is still there |
| tray, left click | shows it, un-minimises it, focuses it — all three, because a minimised window that is only made visible looks like a click that did nothing |
| tray menu | 보이기, 끝내기 |
| 끝내기 | ends the process, and the job object ends the recogniser and the voice with it |
| second launch | a second **window**, not a second argmind: no server, no icon, and closing it closes only it |
| where it may point | loopback only, checked — there is no address bar and this is not a browser |

**`tao` ends the process itself when its loop stops**, so destructors do not run. That is why the
recogniser and the voice were put behind a job object rather than a `Drop` in the first place, and
it is why the tray icon is taken down by hand on `LoopDestroyed`: Windows leaves an orphaned icon
by the clock until somebody hovers over it.

**The icon is drawn rather than shipped** — a lit circle, the same object the page draws in the
middle of itself. A file would have been an `.ico` and two `.png`s to keep in step.

## What was deleted

`argmind-env-windows/src/tray.rs`, a hand-written `Shell_NotifyIconW` tray, and its test. It worked.
It was replaced rather than kept beside the new one, because two implementations of one thing are
how the two come to behave differently — and because it could only ever be the Windows third of
what was asked for.
