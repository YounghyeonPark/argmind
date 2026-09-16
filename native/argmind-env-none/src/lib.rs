//! An environment that can do nothing, and **says so rather than pretending.**
//!
//! Every method here answers honestly: it sees no widgets, it presses nothing, it makes no sound,
//! it hears nothing. It is not a mock and not a test double — it is what argmind is on an operating
//! system nobody has written an environment for yet, which on 2026-09-16 is two of the three the
//! owner asked for.
//!
//! ## Why it exists, which is not what it looks like
//!
//! The owner asked for Windows, Linux and macOS. `argmind-core` already names no platform, and the
//! contract it asks of an environment is four methods and a struct of four fields — small enough
//! that AT-SPI2 and the macOS Accessibility API both fit it exactly. **The blocker was somewhere
//! else**: the application, `argmind-serve`, named Windows types directly in six files, so adding
//! an environment would have meant rewriting the application rather than plugging something in.
//!
//! That is a claim, and a claim about portability made on one operating system is worth nothing.
//! **This crate is how it gets checked without a second machine**: build the whole application
//! against this and run it. If anything is missing from the seam, it does not compile; if the seam
//! is a fiction, it does not run. `cargo test --features headless` is the proof, and
//! `argmind-serve/tests/portable.rs` is where it lives.
//!
//! ## And it is the list, too
//!
//! **What is in this file is exactly what a third environment must provide.** Nothing had to be
//! guessed: switching the application over and letting the compiler complain enumerated the
//! surface. It came to more than `Eyes` and `Hands` — a way to know which window a person is
//! attending, a microphone, a way to start a supervised child, and a way to ask a yes-or-no
//! question. Those are real capabilities with no contract of their own yet, and **the shape here
//! is the interface**, in the same sense `argmind-voice` says its three traits are interface
//! rather than contract.
//!
//! ## What an honest nothing looks like
//!
//! **Refusing is not the same as failing**, and the difference is the whole design of this file.
//! `confirm` returns `false` — there is no window to ask in, so nothing was approved, and an
//! environment that answered `true` would be one that installs software nobody agreed to.
//! `Hands::live` is `false`, which is the flag the rest of the program already reads to mean *the
//! hands are dry*. `Supervised` is the exception and really works: starting a child process is
//! `std::process`, the same on every operating system — what it cannot promise is that the child
//! dies with argmind, so [`Supervised::supervised`] says `false` and the caller already knows to
//! make the weaker promise out loud.

use std::path::Path;
use std::process::{Child, Command, Stdio};

use argmind_core::contract::{Eyes, Hands, Widget};
use argmind_voice::{Heard, Listener, Speaker};

/// **The rate the recogniser wants**, which is a property of whisper.cpp rather than of any
/// operating system — so it is the same number here as in the Windows microphone.
pub const WANT_HZ: u32 = 16_000;

/// Which window the person is attending to. **Here: none, ever.**
///
/// The Windows one samples the foreground and keeps the handles it has seen, so that while the
/// person is in argmind it can answer with the window they were in before it. With nothing to
/// sample there is nothing to remember, and saying so is better than answering with a window that
/// does not exist.
#[derive(Default, Clone)]
pub struct Watcher;

impl Watcher {
    pub fn saw(&self, _handle: isize) {}
    /// Always `None` — there is no window that is not in front, because there is no window.
    pub fn attended(&self, _in_front: isize) -> Option<isize> {
        None
    }
    /// **Starts no thread.** The Windows one samples every 400 ms forever; a thread that samples
    /// nothing is a thread that costs a wake-up for no answer.
    pub fn watch(&self) {}
}

/// Eyes that see nothing.
pub struct Attending {
    /// The plain eyes underneath. **The application reaches through to them** for the one question
    /// the `Eyes` trait does not carry — *what is at this point* — so the field has to be here.
    pub inner: Plain,
    pub watcher: Watcher,
}

impl Attending {
    /// **A `Result`, and always `Ok`.** The Windows one can fail — UI Automation may refuse to
    /// open — and the caller handles that. Keeping the shape means the caller does not change.
    pub fn new(watcher: Watcher) -> Result<Self, String> {
        Ok(Attending { inner: Plain, watcher })
    }

    /// The handle and title of the window being attended to. Nothing, and empty.
    pub fn window(&self) -> (isize, String) {
        (0, String::new())
    }
}

impl Eyes for Attending {
    /// **Empty, not an error.** A screen with nothing on it is a legitimate reading — the agent
    /// then has nothing to offer and says so, which is exactly right on a machine it cannot see.
    fn foreground(&mut self) -> Vec<Widget> {
        Vec::new()
    }

    fn timed_foreground(&mut self) -> (Vec<Widget>, f64) {
        (Vec::new(), 0.0)
    }
}

/// Eyes that read whatever is in front, rather than what is being attended to. **Also nothing.**
///
/// Two of them because the application really does want both: the server reads the *attended*
/// window, because the person is looking at argmind when they ask; the command line reads the
/// window *in front*, because it is being run from a terminal and there is nothing else to mean.
pub struct Plain;

impl Plain {
    pub fn new() -> Result<Self, String> {
        Ok(Plain)
    }

    /// The name and role of whatever is at a screen point. **Nothing is at any point.**
    ///
    /// Not part of the `Eyes` trait, and the one place the application reaches past it: the page
    /// shows what a proposed click would land on, and answering that needs a hit test rather than
    /// a listing. Both AT-SPI2 and the macOS Accessibility API have one.
    pub fn at_point(&self, _x: i64, _y: i64) -> Option<(String, String)> {
        None
    }
}

impl Default for Plain {
    fn default() -> Self {
        Plain
    }
}

impl Eyes for Plain {
    fn foreground(&mut self) -> Vec<Widget> {
        Vec::new()
    }
    fn timed_foreground(&mut self) -> (Vec<Widget>, f64) {
        (Vec::new(), 0.0)
    }
}

/// Hands that touch nothing.
///
/// **`done` is still recorded**, because the application reads it to explain what happened, and an
/// empty log would make every refusal look like a success.
#[derive(Default)]
pub struct Still {
    pub done: Vec<argmind_core::contract::Did>,
}

impl Still {
    pub fn unrestricted() -> Self {
        Still::default()
    }
    /// **Arming changes nothing here**, and that is not a quiet lie: [`Hands::live`] stays `false`,
    /// which is the flag the rest of the program reads. Nothing can believe these hands are armed.
    pub fn arm(self) -> Self {
        self
    }
}

impl Hands for Still {
    fn live(&self) -> bool {
        false
    }

    fn click(&mut self, _widget: &Widget) {
        self.done.push(argmind_core::contract::Did::Sent { live: false });
    }

    fn type_text(&mut self, _text: &str) {
        self.done.push(argmind_core::contract::Did::Sent { live: false });
    }

    fn press(&mut self, _key: &str) {
        self.done.push(argmind_core::contract::Did::Sent { live: false });
    }
}

/// A voice that makes no sound.
#[derive(Default)]
pub struct Mute {
    pub asked: u64,
}

impl Mute {
    /// A `Result` for the same reason [`Attending::new`] is one.
    pub fn new() -> Result<Self, String> {
        Ok(Mute::default())
    }
    /// **`false`, which is the truth and is also what stops it being used.** The router refuses to
    /// hand Korean to a voice that has none rather than emitting zero bytes and reporting success —
    /// a measured Windows failure, and the right behaviour here too.
    pub fn has_korean(&self) -> bool {
        false
    }
    pub fn korean_is_old(&self) -> bool {
        false
    }
}

impl Speaker for Mute {
    fn say(&mut self, _text: &str) {
        self.asked += 1;
    }
    fn stop(&mut self) {}
    fn speaking(&self) -> bool {
        false
    }
    fn release(&mut self) {}
}

/// A microphone that is never open.
///
/// **Not `None` at the call site but a device that hears nothing**, because the application already
/// has a `None` for *the person did not ask to listen*, and the two mean different things: one is a
/// choice and this is a limitation.
#[derive(Default)]
pub struct Deaf {
    held: bool,
    /// **`false`, and said once rather than discovered on every key press** — the same contract the
    /// Windows microphone keeps when a machine has no input device.
    pub available: bool,
}

impl Deaf {
    pub fn new() -> Self {
        Deaf::default()
    }
    /// The talk key is still tracked, so the interface behaves the way it looks — held is held.
    /// **Nothing is recorded.**
    pub fn hold(&mut self, down: bool) {
        self.held = down;
    }
    pub fn held(&self) -> bool {
        self.held
    }
    /// Always `None`: there is never a clip, so a turn never starts from one.
    pub fn take(&mut self) -> Option<Clip> {
        None
    }
}

impl Listener for Deaf {
    fn hold(&mut self, down: bool) {
        self.held = down;
    }
    fn heard(&mut self) -> Option<Heard> {
        None
    }
    fn release(&mut self) {}
}

/// What a microphone would hand over. Never produced here; the type exists so the caller compiles.
pub struct Clip {
    pub samples: Vec<f32>,
}

impl Clip {
    pub fn worth_hearing(&self) -> bool {
        false
    }
}

/// Asking a yes-or-no question when there is nowhere to ask it.
pub mod ask {
    /// **Always `false`, and that is a refusal rather than a failure.**
    ///
    /// Everything this program guards with a question — installing software, running somebody
    /// else's build, opening the microphone — is guarded because a person has to agree. With no
    /// window to put the question in, nobody agreed. Answering `true` here would turn *the machine
    /// cannot ask* into *the machine said yes on your behalf*, which is the worst failure in this
    /// program's whole design.
    pub fn confirm(title: &str, body: &str) -> bool {
        //: Said somewhere, so a person reading a log can see the question that was never put.
        eprintln!("  asked with nowhere to ask, so refused: {title}\n{body}");
        false
    }
}

/// Starting a child process and knowing whether it is still there.
pub mod supervise {
    use super::*;

    /// A child, started and watched — **and this one really works.**
    ///
    /// `std::process` is the same on all three operating systems, so there is nothing to stub. What
    /// is missing is the guarantee: the Windows version puts the child in a job object so the
    /// operating system kills it when argmind goes, **even when argmind is killed or calls
    /// `exit`.** There is no portable equivalent — Linux has `PR_SET_PDEATHSIG` and macOS has
    /// neither — so [`Self::supervised`] is `false` and the caller says the weaker promise out
    /// loud. That mechanism already exists because the Windows one can fail too.
    pub struct Supervised {
        child: Child,
    }

    impl Supervised {
        pub fn start(exe: &Path, args: &[&str], from: Option<&Path>, log: Option<&Path>)
                     -> Result<Self, String> {
            let mut command = Command::new(exe);
            command.args(args).stdin(Stdio::null());
            if let Some(at) = from {
                command.current_dir(at);
            }
            match log {
                Some(to) => {
                    let file = std::fs::File::create(to).map_err(|e| format!("{e}"))?;
                    let also = file.try_clone().map_err(|e| format!("{e}"))?;
                    command.stdout(file).stderr(also);
                }
                None => { command.stdout(Stdio::null()).stderr(Stdio::null()); }
            }
            let child = command.spawn().map_err(|e| format!("{e}"))?;
            Ok(Supervised { child })
        }

        /// **`false`, said plainly.** Nothing here promises the child dies with argmind.
        pub fn supervised(&self) -> bool {
            false
        }

        pub fn alive(&mut self) -> bool {
            matches!(self.child.try_wait(), Ok(None))
        }

        pub fn finished(&mut self) -> bool {
            !self.alive()
        }

        pub fn stop(&mut self) {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }

    impl Drop for Supervised {
        /// **The only cleanup there is here**, and it is weaker than a job object in exactly the
        /// way that matters: `std::process::exit` and a kill from outside both skip it.
        fn drop(&mut self) {
            self.stop();
        }
    }
}

/// The window in front, right now. There is none.
///
/// # Safety
///
/// Nothing unsafe happens. The signature is `unsafe` because the Windows one is, and the caller is
/// one piece of code rather than two.
pub unsafe fn foreground_handle_now() -> isize {
    0
}

/// The title of a window. There are no windows, so there are no titles.
pub fn window_title(_handle: isize) -> String {
    String::new()
}
