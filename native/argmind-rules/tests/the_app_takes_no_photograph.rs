//! **One place in the application may take a photograph, and it is `platform.rs`.**
//!
//! Until 2026-09-23 this was held by a type: `camera.rs` reduced every frame to 64×36 greys and had
//! no accessor for anything larger, so the picture could not leave the module however anybody felt
//! about it. The owner asked for a vision path, so `Camera::photograph` exists — and the guarantee
//! the type was carrying had to go somewhere or be simply gone. It went here.
//!
//! ## This rule has already been moved once, deliberately, and that is the point
//!
//! The first version said **nothing in the application may take one at all** — only a harness a
//! person runs on purpose. It said, in as many words: *when something does, this rule is what has
//! to be deliberately changed, and changing it is the moment the permission has to be built.*
//!
//! Later the same day, `argmind-env-face` arrived and `platform.rs` needed a real picture to hand a
//! face detector. **The rule caught it on the first run of the suite**, which is the whole of what
//! it is for, and the permission was there to point at: `face-models` for the files and `faces` for
//! keeping anything — two keys, because downloading weights and writing down a person are two
//! decisions. So the rule narrowed rather than went away.
//!
//! **What is still guaranteed**, and it is most of what was:
//!
//! * every other file in `argmind-serve` is exactly as blind as before — `app.rs` cannot reach a
//!   picture, and neither can `people.rs`, which is the file that writes something down;
//! * `platform.rs` takes **one**, in one place, which is the camera watcher;
//! * what comes back out of that function is 128 numbers, never pixels — `people.rs` has a test of
//!   its own that nothing resembling an image is ever written to disk.
//!
//! **What is no longer guaranteed**: that the application cannot hold a frame at all. It can, for
//! about as long as it takes `tract` to look at it. That is a real loss and it is written down here
//! rather than quietly absorbed.
//!
//! Checked by breaking it: a `photograph(` added to `app.rs` fails, a second one added to
//! `platform.rs` fails, and a bare `::shrink(` anywhere fails.

use std::path::{Path, PathBuf};

/// The one file that may. **Named here rather than inferred**, so moving the call somewhere else is
/// a decision somebody makes in this file instead of a diff nobody reads.
const THE_ONE_PLACE: &str = "platform.rs";

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

/// The lines of code — **comments stripped**, because a rule that forbids writing down what it
/// forbids is one nobody can document. The same correction `hands.rs` needed a day earlier.
fn code(source: &str) -> Vec<(usize, String)> {
    source.lines().enumerate()
        .map(|(n, line)| (n + 1, line.split("//").next().unwrap_or("").to_string()))
        .collect()
}

#[test]
fn only_one_file_in_the_application_takes_a_photograph() {
    let native = root().join("native");
    let app = native.join("argmind-serve");
    assert!(app.is_dir(), "the application moved; recheck this rule");

    let mut elsewhere = Vec::new();
    let mut here = Vec::new();
    for path in sources(&app) {
        let source = std::fs::read_to_string(&path).unwrap_or_default();
        let allowed = path.file_name().is_some_and(|f| f == THE_ONE_PLACE);
        for (n, line) in code(&source) {
            //: `::shrink(` is the other way out of that module — the function `photograph` is built
            //: on — and it is refused everywhere including the one place, which has no use for it.
            let takes = line.contains("photograph(");
            if line.contains("::shrink(") || (takes && !allowed) {
                elsewhere.push(format!("{}:{n}: {}",
                                       path.strip_prefix(&native).unwrap_or(&path).display(),
                                       line.trim()));
            } else if takes {
                here.push(format!("{n}: {}", line.trim()));
            }
        }
    }

    assert!(elsewhere.is_empty(),
            "only {THE_ONE_PLACE} may take a picture, and these do: {elsewhere:#?}\n\
             Everything else in the application sees 64x36 greys. If this call really belongs \
             where it now is, move the rule on purpose and say what permission covers it.");

    //: **One, not merely few.** Two calls is two places a frame can be held, and the second one is
    //: always the one nobody reviewed.
    assert_eq!(here.len(), 1,
               "{THE_ONE_PLACE} should take exactly one picture, it takes {}: {here:#?}",
               here.len());
}

/// **The permission this rule was narrowed against exists**, and it is two keys rather than one.
///
/// Written as a test because the module note above claims it, and a note claiming a permission that
/// was never built is worse than no note.
#[test]
fn the_permission_that_allowed_the_narrowing_is_real() {
    let decided = std::fs::read_to_string(
        root().join("native/argmind-serve/src/decided.rs")).expect("decided.rs");
    for key in ["\"face-models\"", "\"faces\""] {
        assert!(decided.contains(key),
                "{key} is gone from the decidable list, so this rule is narrower than the \
                 permission that justified narrowing it");
    }
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
