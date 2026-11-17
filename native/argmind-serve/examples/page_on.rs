//! **The page, served by an argmind that can do nothing, on a port of its own** — for looking at it
//! in a browser a harness started, never at the person's own argmind.
//!
//! ```text
//!   LOCALAPPDATA=<an empty folder> cargo run -p argmind-serve --features headless --example page_on -- 7741
//! ```
//!
//! Built with `headless`, the environment is `argmind-env-none`: there is no camera, microphone,
//! speaker or pair of hands in the process at all. **Point `LOCALAPPDATA` at an empty folder**,
//! because it reads and writes `%LOCALAPPDATA%\argmind\` exactly as the real one does — the
//! decisions, the conversation, and the page's language among them — and a look at the page must
//! not change what the person set. Made for the interface language on 2026-11-17, to see the page
//! written again in each language rather than to believe a table.

#[cfg(feature = "headless")]
fn main() {
    let port: u16 = std::env::args().nth(1).and_then(|p| p.parse().ok()).unwrap_or(7741);
    assert_ne!(port, argmind_serve::app::PORT, "that port is the person's argmind");
    let how = argmind_serve::app::How {
        dry: true, speak: false, hear: false, model_name: "qwen3:8b".into(), port, web: None,
    };
    println!("  {}", argmind_serve::app::address(port));
    argmind_serve::app::run(how);
}

#[cfg(not(feature = "headless"))]
fn main() {
    eprintln!("build it with --features headless, so nothing in it can open a device");
    std::process::exit(2);
}
