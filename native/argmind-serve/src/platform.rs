//! **The one file in this crate that knows which operating system this is.**
//!
//! Everything else in `argmind-serve` says `platform::Sight` and `platform::Touch` and does not
//! know or care what they are. That is the same discipline `argmind-core` keeps — and until
//! 2026-09-16 the application did not keep it: `app.rs`, `needs.rs`, `recogniser.rs`, `speaker.rs`
//! and both binaries named Windows types directly, in twenty-one places. **Adding a second
//! environment would have meant rewriting the application rather than plugging something in**, and
//! that, rather than the 1,844 lines of `argmind-env-windows`, was what actually blocked the port
//! the owner asked for.
//!
//! `argmind-rules` holds this by reading the source, the way it holds the core's blindness.
//!
//! ## What the choice is
//!
//! Windows gets `argmind-env-windows`. Everything else gets [`argmind_env_none`], which sees
//! nothing, presses nothing and says so — because there is no macOS or Linux environment yet and
//! **pretending is worse than refusing.**
//!
//! `--features headless` forces the empty one even on Windows. That is not a convenience: it is
//! how a claim about portability gets checked on a machine that only has one operating system.
//! `tests/portable.rs` builds and runs the whole application that way, so if this seam is a
//! fiction it does not compile.
//!
//! ## The names are neutral and the shapes are not a contract
//!
//! `Sight`, `Touch`, `Mouth`, `Ear`, `Watching` — `Eyes` and `Hands` are traits in the core, and
//! these are the concrete types behind them plus the capabilities that have no trait yet: knowing
//! which window a person is attending to, a microphone, starting a supervised child, and asking a
//! yes-or-no question.
//!
//! **Those four are interface rather than contract**, in the same sense `argmind-voice` says its
//! three traits are. Nothing yet justifies promoting them — a contract argued from one
//! implementation is a contract shaped like that implementation, and there will not be a second
//! opinion until somebody writes `argmind-env-linux`. What is written down instead is the list:
//! **whatever is in `argmind-env-none/src/lib.rs` is what a third environment must provide**, and
//! it was not guessed — the compiler enumerated it.

//: **Re-exported from the core rather than from an environment.** Two environments now report what
//: an action did, and a second copy of that enum is how they would come to mean different things.
pub use argmind_core::contract::Did;

#[cfg(all(windows, not(feature = "headless")))]
mod chosen {
    pub use argmind_env_windows::ask::confirm;
    pub use argmind_env_windows::microphone::WANT_HZ;
    pub use argmind_env_windows::supervise::Supervised;
    pub use argmind_env_windows::{foreground_handle_now, window_title};
    pub use argmind_env_windows::{Attending as Sight, Microphone as Ear, SapiSpeaker as Mouth,
                                  Watcher as Watching, WindowsEyes as Plain,
                                  WindowsHands as Touch};

    /// Is there a neural voice to prefer over the built-in one?
    ///
    /// **Its own function because the answer is not the same everywhere.** The neural voice needs
    /// a loudspeaker, and the only one written is WASAPI — so on an environment with no audio
    /// there is nothing to prefer and the question is answered at compile time rather than by a
    /// socket timeout.
    pub fn neural_voice(host: &str) -> Option<Box<dyn argmind_voice::Speaker>> {
        argmind_env_qwentts::QwenVoice::answering(host)
            .then(|| Box::new(argmind_env_qwentts::QwenVoice::new(crate::speaker::VOICE))
                as Box<dyn argmind_voice::Speaker>)
    }
}

#[cfg(any(not(windows), feature = "headless"))]
mod chosen {
    pub use argmind_env_none::ask::confirm;
    pub use argmind_env_none::supervise::Supervised;
    pub use argmind_env_none::WANT_HZ;
    pub use argmind_env_none::{foreground_handle_now, window_title};
    pub use argmind_env_none::{Attending as Sight, Deaf as Ear, Mute as Mouth, Plain,
                               Still as Touch, Watcher as Watching};

    /// **Never.** The synthesiser may well be answering on its port — it is a server and it builds
    /// on all three — but nothing here can play what it sends back. Saying no is the honest
    /// answer; asking the socket and then being unable to make a sound would not be.
    pub fn neural_voice(_host: &str) -> Option<Box<dyn argmind_voice::Speaker>> {
        None
    }
}

pub use chosen::*;

/// What this build can actually do, in one line, for a person reading a log or a page.
///
/// **Said out loud rather than discovered.** An agent that sees nothing and presses nothing looks
/// broken; an agent that says *I have no eyes on this operating system* is a program somebody can
/// reason about.
pub fn describe() -> &'static str {
    if cfg!(all(windows, not(feature = "headless"))) {
        "windows"
    } else if cfg!(windows) {
        //: Windows underneath, deliberately blinded — this is the measurement build.
        "headless (on windows)"
    } else {
        "headless — no environment for this operating system yet"
    }
}

/// Can this build see the screen and press things at all?
pub fn embodied() -> bool {
    cfg!(all(windows, not(feature = "headless")))
}
