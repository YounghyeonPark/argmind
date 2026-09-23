//! **The application cannot take a photograph. Only a harness a person runs on purpose can.**
//!
//! Until 2026-09-23 this was held by a type: `camera.rs` reduced every frame to 64×36 greys and had
//! no accessor for anything larger, so the picture could not leave the module however anybody felt
//! about it. The owner asked for a vision path, so `Camera::photograph` exists — and the guarantee
//! the type was carrying has to go somewhere or it is simply gone.
//!
//! **This is where it went.** It is weaker than a type and it is checkable, which is the trade
//! being made rather than a trade being hidden. `camera.rs` has said since it was written that a
//! vision model would need *a different call, a different permission and a different argument*; the
//! call and the argument exist, and **the permission does not yet, because nothing in the
//! application asks for one.** When something does, this rule is what has to be deliberately
//! changed — and changing it is the moment the permission has to be built.
//!
//! Checked by breaking it: a `photograph(` added to `app.rs` fails here.

use std::path::{Path, PathBuf};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap().to_path_buf()
}

/// Every `.rs` under a directory, skipping `target`.
fn sources(under: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![under.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).into_iter().flatten().filter_map(Result::ok) {
            let path = entry.path();
            if path.is_dir() {
                if path.file_name().unwrap_or_default() != "target" {
                    stack.push(path);
                }
            } else if path.extension().is_some_and(|e| e == "rs") {
                out.push(path);
            }
        }
    }
    out
}

#[test]
fn nothing_in_the_application_takes_a_photograph() {
    let native = root().join("native");
    let app = native.join("argmind-serve");
    assert!(app.is_dir(), "the application moved; recheck this rule");

    let mut offenders = Vec::new();
    for path in sources(&app) {
        let source = std::fs::read_to_string(&path).unwrap_or_default();
        for (n, line) in source.lines().enumerate() {
            //: Comments may say the word — a rule that forbids writing down what it forbids is one
            //: nobody can document. The same correction `hands.rs` needed a day earlier.
            let code = line.split("//").next().unwrap_or("");
            if code.contains("photograph(") || code.contains("::shrink(") {
                offenders.push(format!("{}:{}: {}",
                                       path.strip_prefix(&native).unwrap_or(&path).display(),
                                       n + 1, code.trim()));
            }
        }
    }
    assert!(offenders.is_empty(),
            "the application takes a picture: {offenders:#?}\n\
             `camera.rs` reduces to 64x36 greys for everything the app does. A real picture needs \
             the permission that file's note has always said it would — build that, then move this \
             rule deliberately.");
}

/// **And the door it is guarding is still there.** A rule about a function that has been renamed
/// passes for ever and protects nothing.
#[test]
fn the_door_this_rule_guards_still_exists() {
    let camera = std::fs::read_to_string(
        root().join("native/argmind-env-windows/src/camera.rs")).expect("camera.rs");
    assert!(camera.contains("pub fn photograph(&mut self, wide: usize, tall: usize)"),
            "photograph has moved or changed shape; this rule is watching nothing");
    //: **The size is an argument, which is the other half of the design.** A call that handed back
    //: whatever the sensor had would make *how much of a room* a runtime accident instead of
    //: something a reader can see.
    assert!(!camera.contains("pub fn photograph(&mut self)"),
            "photograph no longer makes the caller name a size");
    //: And the cheap path is untouched: `look` still gives the 64x36 greys and nothing else.
    assert!(camera.contains("pub fn look(&mut self) -> Result<Frame, String>"),
            "the reduced path changed shape; MOVING was measured against it");
}
