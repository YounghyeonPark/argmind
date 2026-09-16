//! **The application names one operating system, in one file.**
//!
//! `argmind-core` has never named a platform and a rule already holds that. The *application* did:
//! until 2026-09-16 `argmind-serve` said `WindowsHands`, `SapiSpeaker`, `Attending` and
//! `argmind_env_windows::` in six files and twenty-one places. That is what actually blocked the
//! port the owner asked for — not the 1,844 lines of `argmind-env-windows`, which is a thing you
//! write once per operating system, but the application, which you would have had to write again.
//!
//! So: `platform.rs` chooses, and nothing else in the crate knows. **The rule is here rather than
//! in a comment because the last one like it was a comment**, and `docs/findings/porting-the-core.md`
//! records what that was worth — *"the room stays quiet unless somebody asked"* was a note in two
//! places rather than a fact, and it was wrong in both.
//!
//! **This is the third rule of its kind and the weakest to write, so it is checked by breaking.**
//! A test that reads source and finds nothing is a test that passes on an empty file; this one
//! asserts it looked at the right files and found the seam where it should be.
//!
//! Needs no network and no model: it reads text files.

use std::path::{Path, PathBuf};

use argmind_rules::{code_lines, root, rust_files};

/// The file allowed to know, and the binary that owns the window.
///
/// `argmind-app` is on the list for one line — it puts the webview's cache beside the person's
/// files, and picking that folder means knowing which operating system this is. It says so where
/// it does it.
const MAY_KNOW: [&str; 2] = ["platform.rs", "argmind-app.rs"];

/// Names that belong to one operating system.
///
/// **The crate path and the type names**, because either alone would miss the other: an
/// `argmind_env_windows::` import is obvious and a bare `SapiSpeaker` after a `use` is not.
const WINDOWS_ONLY: [&str; 9] = [
    "argmind_env_windows", "argmind_env_qwentts",
    "WindowsHands", "WindowsEyes", "SapiSpeaker", "Microphone", "Attending", "Watcher",
    "QwenVoice",
];

fn serve_sources() -> Vec<PathBuf> {
    rust_files(&root().join("native").join("argmind-serve").join("src"))
}

fn named(file: &Path) -> String {
    file.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default()
}

/// Every file but `platform.rs` is free of Windows.
#[test]
fn only_one_file_in_the_application_knows_which_operating_system_this_is() {
    let files = serve_sources();
    //: **The test looked at something.** A glob that matched nothing would pass silently, which is
    //: how a rule quietly stops being one.
    assert!(files.len() >= 8, "only {} sources found under argmind-serve/src", files.len());

    let mut guilty: Vec<String> = Vec::new();
    for file in &files {
        let short = named(file);
        if MAY_KNOW.contains(&short.as_str()) {
            continue;
        }
        let source = std::fs::read_to_string(file).expect("a source reads");
        for (at, line) in code_lines(&source) {
            for name in WINDOWS_ONLY {
                if line.contains(name) {
                    guilty.push(format!("{short}:{at}: {}", line.trim()));
                }
            }
        }
    }
    assert!(guilty.is_empty(),
            "these name an operating system outside platform.rs — say `platform::Sight` and let \
             one file choose, or a second environment means rewriting the application: {guilty:#?}");
}

/// **And `platform.rs` really is the seam**, rather than an empty file the rule passes over.
#[test]
fn the_one_file_that_knows_offers_both_answers() {
    let at = root().join("native").join("argmind-serve").join("src").join("platform.rs");
    let source = std::fs::read_to_string(&at).expect("platform.rs is the seam and has to be there");

    //: Both arms, and the feature that forces the empty one where a real one exists. Without that
    //: last part the claim could never be checked on a machine with one operating system.
    assert!(source.contains("argmind_env_windows"), "platform.rs offers no Windows");
    assert!(source.contains("argmind_env_none"), "platform.rs offers no fallback");
    assert!(source.contains("feature = \"headless\""),
            "platform.rs cannot be made to take the fallback, so nothing can test it");

    //: The names the rest of the crate uses. If one of these disappears the rule above would start
    //: passing for the wrong reason — nothing left to name.
    for neutral in ["Sight", "Touch", "Mouth", "Ear", "Watching", "Plain", "Supervised",
                    "confirm", "WANT_HZ"] {
        assert!(source.contains(neutral), "platform.rs does not offer {neutral:?}");
    }
}

/// The empty environment names no platform either — it is what a Linux checkout compiles.
#[test]
fn the_environment_that_can_do_nothing_depends_on_nothing_platform_specific() {
    let manifest = root().join("native").join("argmind-env-none").join("Cargo.toml");
    let text = std::fs::read_to_string(&manifest).expect("a manifest reads");
    for platform in ["windows", "objc", "gtk", "webkit", "core-foundation"] {
        assert!(!text.contains(platform),
                "argmind-env-none depends on {platform:?}, which is the one thing it must not");
    }
    for file in rust_files(&root().join("native").join("argmind-env-none").join("src")) {
        let source = std::fs::read_to_string(&file).expect("a source reads");
        for (at, line) in code_lines(&source) {
            assert!(!line.contains("target_os"),
                    "argmind-env-none branches on the operating system at line {at}: {}", line.trim());
        }
    }
}
