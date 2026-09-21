//! **Ollama: the one server argmind does not own.**
//!
//! The recogniser and the synthesiser are argmind's — it clones them, compiles them, starts them
//! and takes them down with it, and [`argmind_env_windows::supervise::Supervised`] makes that last
//! part the operating system's promise rather than a destructor's. This is the opposite in every
//! one of those respects. Somebody installed Ollama before argmind existed, other programs may be
//! talking to it, and it was running before this process started and should be running after it
//! stops. **So nothing here goes in the job object**, which is the whole reason this is a module
//! and not two lines in [`crate::needs`].
//!
//! ## Installed and running are two questions, and conflating them cost a reinstall a launch
//!
//! `present` for Ollama is *answering on 11434*, which is right — what a person needs to know is
//! whether there is a model to ask, not whether there is an executable somewhere. But the fix
//! offered against that was [`crate::fix::Fix::Install`], a 700 MB download and an installer, so a
//! standing 항상 허용 on `ollama` with Ollama installed and merely stopped would fetch and run the
//! installer **on every launch**. That is the same defect the two build fixes had on 2026-09-21 —
//! *not answering* read as *not there* — found while fixing those, in the row next to them.
//!
//! Nobody had hit it: Ollama normally starts itself, and the owner's `decisions.json` has no
//! `ollama` key. It was written down rather than half-fixed, and then fixed.

use std::process::Stdio;
use std::time::{Duration, Instant};

/// How long to wait for a server that has been asked to start.
///
/// **Longer than the port check and shorter than a person's patience.** Ollama opens its socket
/// quickly and then loads nothing until something is asked of it, so this is a socket appearing
/// rather than a model reaching a card — unlike [`crate::speaker::serve`], which waits out a
/// gigabyte.
const WAIT: Duration = Duration::from_secs(20);

/// Is Ollama on this machine at all?
///
/// **The question is whether the program ran, not whether it succeeded**, and that distinction is
/// the point of writing this out rather than reusing [`crate::building::missing`]. For a compiler,
/// *exits non-zero* means *not usable*. For Ollama it means almost nothing: `ollama --version`
/// with no server to reach prints two warnings and — on 0.34.2, measured — still exits 0, but
/// there is no promise it always will, and an exit code about a server that is admittedly down is
/// not evidence about a file on disk. A spawn that fails at all is the honest signal: the
/// executable was not found.
pub fn here() -> bool {
    crate::platform::child("ollama")
        .arg("--version")
        .stdout(Stdio::null()).stderr(Stdio::null()).stdin(Stdio::null())
        .status()
        .is_ok()
}

/// Start the Ollama that is already installed, and wait until it answers.
///
/// **Not supervised, and that is the decision rather than an omission.** See the module note: this
/// is somebody else's program. Taking it down when argmind closes would stop a server other things
/// on the machine may be using, and would be argmind claiming ownership of something it merely
/// found.
///
/// **Waiting is the point**, the same as for the two servers argmind does own: reporting success on
/// a process that has started but is not listening produces a first question that fails for no
/// visible reason.
pub fn serve(say: &mut dyn FnMut(&str)) -> Result<(), String> {
    if awake() {
        return Ok(());
    }
    if !here() {
        return Err("Ollama가 이 컴퓨터에 없습니다".into());
    }
    say("Ollama를 실행합니다");
    //: `serve` rather than the tray application, because it is the documented way and it is the
    //: same command on every operating system this could run on.
    crate::platform::child("ollama")
        .arg("serve")
        .stdout(Stdio::null()).stderr(Stdio::null()).stdin(Stdio::null())
        .spawn()
        .map_err(|e| format!("Ollama를 실행하지 못했습니다 ({e})"))?;

    let by = Instant::now() + WAIT;
    while Instant::now() < by {
        if awake() {
            return Ok(());
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    //: **Named as a timeout rather than as a failure.** A machine where this is slow is a machine
    //: where it may yet come up, and the next page open will find it.
    Err(format!("Ollama가 {} 초 안에 응답하지 않았습니다", WAIT.as_secs()))
}

/// Is it answering? The same call [`crate::needs`] makes, named here so this module reads whole.
pub fn awake() -> bool {
    argmind_env_ollama::manage::awake(argmind_env_ollama::OLLAMA)
}
