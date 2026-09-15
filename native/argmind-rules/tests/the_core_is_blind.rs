//! The rule everything hangs on, checked against the source.
//!
//! > **The deterministic side says WHERE. The probabilistic side says WHAT.
//! > A model never emits a coordinate.**
//!
//! Measured first, and that is why it is a rule: a vision model took **24–82 s** for a pixel and on
//! a dense screen never answered; UI Automation answers exactly in **41–266 ms**. A prompt that
//! asks a model for an `x` is that trade being undone without anybody deciding to.
//!
//! **The Python held this by walking its AST and the port was supposed to make it structural.**
//! `docs/findings/porting-the-core.md` argued that a `Listing { name, role }` type would make *the
//! model is never shown a coordinate* a thing the compiler knows rather than a thing a test
//! asserts. **That type was never built** — the core still passes `Widget`, which carries a `rect`.
//! So the rule is still held by looking, and this is the thing doing the looking; the promise is
//! recorded here rather than quietly dropped, because a rewrite justified by a payoff that did not
//! arrive should say so somewhere.
//!
//! Text, not syntax. Rust has no `ast` module in its standard library and this crate takes no
//! dependencies, so a comment can defeat these by containing the wrong word — which is why only
//! code lines are read, and why the failure message prints the line.

use argmind_rules::{code_lines, root, rust_files};

fn core() -> std::path::PathBuf {
    root().join("native").join("argmind-core").join("src")
}

/// **The core names no platform.** The split is real only if the core can be written without one.
#[test]
fn the_core_names_no_platform() {
    //: The same list as the Python's, plus the two the Rust could grow that the Python could not.
    const BANNED: [&str; 12] = ["uiautomation", "comtypes", "win32", "windll", "sendinput", "adb",
                                "uiautomator", "hwnd", "windows::", "winapi", "wasapi", "hresult"];
    let mut offenders = Vec::new();
    for path in rust_files(&core()) {
        let text = std::fs::read_to_string(&path).expect("a core file reads");
        let name = path.file_name().unwrap_or_default().to_string_lossy().to_string();
        for (n, line) in code_lines(&text) {
            let low = line.to_lowercase();
            //: A doc comment on the same line as code, and the `//:` notes this project uses, are
            //: prose about the rule rather than a breach of it.
            let code = low.split("//").next().unwrap_or("");
            for word in BANNED {
                if code.contains(word) {
                    offenders.push(format!("{name}:{n}: {}", line.trim()));
                }
            }
        }
    }
    assert!(offenders.is_empty(), "the core has grown a platform dependency: {offenders:#?}");
}

/// **No core module asks a model for a coordinate.**
///
/// The Python looked for a string literal containing both `"x"` and `"y"`, which is what a prompt
/// asking for a point looks like. Same here, and for the same reason: the prompt is the only place
/// this could be undone, and the prompt is a string.
#[test]
fn no_core_module_asks_a_model_for_a_coordinate() {
    let mut offenders = Vec::new();
    for path in rust_files(&core()) {
        let text = std::fs::read_to_string(&path).expect("a core file reads");
        let name = path.file_name().unwrap_or_default().to_string_lossy().to_string();
        for (n, line) in code_lines(&text) {
            let low = line.to_lowercase();
            let code = low.split("//").next().unwrap_or("");
            //: Two ways a quoted key reaches a prompt in Rust: escaped inside an ordinary
            //: literal, and bare inside a raw string. **The second was found by trying it** — the
            //: first version of this check looked only for the escaped form, and
            //: `r#"answer with {"x": 0}"#` walked straight past it.
            let escaped = code.contains("\\\"x\\\"") && code.contains("\\\"y\\\"");
            let raw = code.contains("r#\"") && code.contains("\"x\"") && code.contains("\"y\"");
            if escaped || raw {
                offenders.push(format!("{name}:{n}"));
            }
        }
    }
    assert!(offenders.is_empty(),
            "a coordinate is being asked of a model at {offenders:?}; the environment supplies it");
}

/// **What is rendered into the prompt never mentions a rectangle.**
///
/// Narrower than the whole core and that narrowness is the point. `session.rs` uses coordinates
/// on purpose — `find()` picks the nearest same-named widget by distance, which is the
/// deterministic side doing exactly its job. The rule is not *the core never touches a rectangle*;
/// it is *the model is never shown one*, and `converse.rs` is where that could stop being true.
///
/// **The first version of this test asserted the wrong thing** and failed on `find()`. Keeping the
/// wider assertion would have meant either deleting a correct tiebreak or carrying a rule nobody
/// could satisfy — and a rule that has to be argued with is one somebody eventually turns off.
#[test]
fn what_is_rendered_into_the_prompt_never_mentions_a_rectangle() {
    let at = core().join("converse.rs");
    let text = std::fs::read_to_string(&at).expect("converse.rs reads");
    let offenders: Vec<String> = code_lines(&text).into_iter()
        .filter(|(_, l)| {
            let code = l.split("//").next().unwrap_or("");
            code.contains(".rect") || code.contains(".centre()")
        })
        .map(|(n, l)| format!("converse.rs:{n}: {}", l.trim()))
        .collect();
    assert!(offenders.is_empty(),
            "the prompt has reached for a coordinate: {offenders:#?}");
    //: And it is genuinely the renderer being read, not an empty file.
    assert!(text.contains("fn render"), "converse.rs no longer renders the prompt");
}

/// The test is looking at something. **A rule checked against an empty list passes forever.**
#[test]
fn there_is_a_core_to_check() {
    let files = rust_files(&core());
    assert!(files.len() >= 5, "only {} files in the core — is the path right?", files.len());
    let session = std::fs::read_to_string(core().join("session.rs")).expect("session.rs");
    assert!(!code_lines(&session).is_empty());
}
