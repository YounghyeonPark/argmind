//! Does starting a console program from a windowed one open a console? **A/B, on this machine.**
//!
//!     cargo run -p argmind-serve --example quiet
//!
//! The owner's complaint on 2026-09-17 was that console windows appear while argmind is working.
//! Every fix shells out â€” `git`, `cmake`, `objdump`, `curl.exe`, `certutil`, `reg`, `where` â€” and
//! so does every check of what is missing, which is five `--version` calls each time the page
//! refreshes. From a process with no console of its own, Windows gives each of them one.
//!
//! **`#![windows_subsystem = "windows"]` is the whole point of this file**: a console binary hands
//! its own console to its children and nothing flashes, so measuring this from `cargo test` would
//! have said there was no problem. It has to be a windowed program to reproduce at all.
//!
//! Three phases, each held long enough for something else to count windows: a slow console program
//! spawned plainly, the same one spawned through `platform::child`, and then **the real path** â€”
//! `needs::every`, which is the five `--version` calls and the registry read that run every time
//! the page asks what is missing. Nothing is installed and nothing is changed.

#![windows_subsystem = "windows"]

use std::process::{Command, Stdio};
use std::time::Duration;

fn main() {
    let log = std::env::temp_dir().join("argmind-quiet.log");
    let say = |line: &str| {
        use std::io::Write;
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&log) {
            let _ = writeln!(f, "{line}");
        }
    };
    let _ = std::fs::remove_file(&log);
    say("start");

    //: Three seconds of doing nothing, which is long enough for a poller to catch a window.
    let slow = ["-n", "4", "127.0.0.1"];

    say("plain: spawning");
    let child = Command::new("ping").args(slow)
        .stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null())
        .spawn().expect("ping");
    let _ = { let mut c = child; c.wait() };
    say("plain: done");

    std::thread::sleep(Duration::from_secs(2));

    say("quiet: spawning");
    let mut command = Command::new("ping");
    command.args(slow).stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null());
    argmind_serve::platform::without_a_console(&mut command);
    let child = command.spawn().expect("ping");
    let _ = { let mut c = child; c.wait() };
    say("quiet: done");

    std::thread::sleep(Duration::from_secs(2));

    //: **The real path, not a stand-in.** Asking what is missing runs five `--version` calls and a
    //: registry read, every time the page refreshes â€” which is where the owner was seeing this.
    say("needs: asking");
    let found = argmind_serve::needs::every("qwen3:8b");
    say(&format!("needs: {} rows", found.len()));
    std::thread::sleep(Duration::from_secs(2));
    say("needs: done");
    say("end");
}

