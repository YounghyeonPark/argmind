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

    /// What speaks when the neural voice is not there, in the words a person would recognise.
    pub fn fallback_voice() -> Option<&'static str> {
        Some("Windows 음성")
    }

    /// Hand a URL to whatever the person chose as their browser.
    pub fn open_url(url: &str) -> Result<(), String> {
        //: `explorer` is a shell command, not a downloader — nothing is fetched or run here.
        std::process::Command::new("explorer").arg(url).spawn()
            .map(|_| ()).map_err(|e| format!("{e}"))
    }

    /// Run a documented command with administrator rights, so Windows asks as well.
    pub fn run_elevated(command: &str) -> Result<bool, String> {
        let done = std::process::Command::new("powershell")
            .args(["-NoProfile", "-Command",
                   &format!("Start-Process -Verb RunAs -Wait -FilePath cmd.exe                              -ArgumentList '/c {command}'")])
            .status().map_err(|e| format!("{e}"))?;
        Ok(done.success())
    }

    pub fn neural_voice(host: &str) -> Option<Box<dyn argmind_voice::Speaker>> {
        argmind_env_qwentts::QwenVoice::answering(host)
            .then(|| Box::new(argmind_env_qwentts::QwenVoice::new(crate::speaker::VOICE))
                as Box<dyn argmind_voice::Speaker>)
    }
}

#[cfg(all(target_os = "linux", feature = "eyes", not(feature = "headless")))]
mod chosen {
    //: **Eyes, and nothing else yet.** `argmind-env-linux` re-exports the rest from
    //: `argmind-env-none`, so the hands are still dry and the voice still silent — see that crate
    //: for why hands on Linux are a permission question before they are a coding one.
    pub use argmind_env_linux::ask::confirm;
    pub use argmind_env_linux::supervise::Supervised;
    pub use argmind_env_linux::WANT_HZ;
    pub use argmind_env_linux::{foreground_handle_now, window_title};
    pub use argmind_env_linux::{Attending as Sight, Deaf as Ear, Mute as Mouth, Plain,
                                Still as Touch, Watcher as Watching};

    /// **Not yet.** The synthesiser is a server and runs here; nothing plays what it sends back.
    pub fn neural_voice(_host: &str) -> Option<Box<dyn argmind_voice::Speaker>> {
        None
    }

    /// `xdg-open` is the right answer and has not been run from here yet.
    pub fn open_url(_url: &str) -> Result<(), String> {
        Err("이 운영체제에서 페이지를 여는 방법이 아직 확인되지 않았습니다".into())
    }

    pub fn run_elevated(_command: &str) -> Result<bool, String> {
        Err("이 운영체제에서는 관리자 권한 실행이 아직 준비되지 않았습니다".into())
    }

    /// Nothing speaks here yet — `speech-dispatcher` exists and is not written.
    pub fn fallback_voice() -> Option<&'static str> {
        None
    }
}

//: Everything else: no environment has been written for it, or this build was asked not to
//: use the one there is.
#[cfg(any(feature = "headless",
          all(not(windows),
              any(not(target_os = "linux"), not(feature = "eyes")))))]
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

    /// **Nothing speaks here.** `speech-dispatcher` on Linux and `AVSpeechSynthesizer` on macOS
    /// both exist; neither is written. Saying *Windows 음성으로 말합니다* on Ubuntu, which is what
    /// this replaced, is worse than saying nothing does.
    pub fn fallback_voice() -> Option<&'static str> {
        None
    }

    /// **Refused rather than guessed.** `xdg-open` and `open` exist and would probably work; that
    /// is not the same as having been run, and nothing here offers a fix that would reach this.
    pub fn open_url(_url: &str) -> Result<(), String> {
        Err("이 운영체제에서 페이지를 여는 방법이 아직 확인되지 않았습니다".into())
    }

    /// There is no tested way to ask this operating system for administrator rights.
    pub fn run_elevated(_command: &str) -> Result<bool, String> {
        Err("이 운영체제에서는 관리자 권한 실행이 아직 준비되지 않았습니다".into())
    }

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

/// Can this build actually carry out the things it offers to do?
///
/// **Every fix argmind offers has been run on Windows and none of them anywhere else.** The
/// installers are `.exe`, the elevation is `powershell -Verb RunAs`, the hash check is `certutil`,
/// the builds want `mingw32-make` and `objdump`, and the Vulkan SDK arrives through `winget`.
///
/// So off Windows the needs are **reported and not offered** — which is a state `needs.rs` already
/// has a word for, *argmind가 할 수 있는 게 없습니다*, and which is the honest one. Running on
/// Linux is what made this urgent: the page was offering
/// `DISM /Online /Add-Capability …` and saying *Windows가 한 번 더 물어봅니다*, on Ubuntu.
///
/// **This is a statement about argmind, not about the operating system.** Ollama installs on Linux
/// and macOS, whisper.cpp builds there, and a Vulkan SDK is a package away. None of it is written
/// here yet, and a button that has never been run is worse than a sentence saying what to do.
pub fn can_carry_out_fixes() -> bool {
    cfg!(all(windows, not(feature = "headless")))
}
