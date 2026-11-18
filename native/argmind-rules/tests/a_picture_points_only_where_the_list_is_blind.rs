//! **A model says where in exactly one shape, from exactly one place, and never presses by itself.**
//!
//! The rule everything hangs on — *the deterministic side says WHERE, the probabilistic side says
//! WHAT, a model never emits a coordinate* — gave way on 2026-11-18, by the owner's choice, inside
//! one container: where the list cannot see into a place (a remote desktop, a character grid), a
//! vision model is shown a picture of it and its box becomes a point **inside the named place**
//! (`argmind_core::intent::Inside`), re-resolved by name and role at the press. `--bin looking`
//! has why: the picture found 5–6 of 6 things the list is blind to and was less exact than the list
//! wherever the list had the thing.
//!
//! An exception is only as narrow as what holds it, so this reads the source for five things:
//!
//! * **one picture, one call** — outside `platform.rs` the window's picture is asked for once,
//!   and a model is shown a picture once, both inside `app.rs`'s `look_inside`, which one turn
//!   calls;
//! * **one maker** — a point inside is constructed only in `argmind-serve/src/looking.rs`, never in
//!   the core, where the reader of a model's reply lives;
//! * **always a press** — `propose` counts a point inside among the things that must wait for a
//!   person, whatever 확인 없이 실행 says;
//! * **the marked picture goes to the person** — since 2026-11-19 the picture the model was shown,
//!   with its point drawn on it, waits beside the row: it is made once, in `look_inside`, after the
//!   model has answered, and read once, by the route that writes it to the page;
//! * **the door still exists** — the names this rule watches have not been renamed out from under
//!   it.
//!
//! Checked by breaking it: a second `look_at(` in `app.rs`, a `window_picture(` in `route.rs`, an
//! `inside: Some(` in `converse.rs`, and `|| effect.inside.is_some()` taken off `must_confirm` each
//! fail here; so do a `looking::shown(` in `think` and a `marked_picture(` in `route.rs`.

use argmind_rules::{code_lines, root, rust_files};

fn serve() -> std::path::PathBuf {
    root().join("native").join("argmind-serve").join("src")
}

/// The code of each line — what is left of it before a `//`.
fn code(line: &str) -> &str {
    line.split("//").next().unwrap_or("")
}

/// The name of the function a line sits in, by the nearest `fn` above it.
fn inside_fn(source: &str, n: usize) -> String {
    source.lines().take(n).collect::<Vec<_>>().into_iter().rev()
        .find_map(|l| {
            let t = l.trim_start();
            let t = t.strip_prefix("pub ").unwrap_or(t);
            t.strip_prefix("fn ").map(|r| r.split(|c: char| c == '(' || c == '<').next().unwrap_or("").to_string())
        })
        .unwrap_or_default()
}

#[test]
fn a_model_is_shown_a_picture_in_one_place() {
    let mut pictures = Vec::new();
    let mut looks = Vec::new();
    let mut callers = Vec::new();
    for path in rust_files(&serve()) {
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        let source = std::fs::read_to_string(&path).unwrap();
        for (n, line) in code_lines(&source) {
            let c = code(line);
            if name != "platform.rs" && c.contains("window_picture(") {
                pictures.push((name.clone(), inside_fn(&source, n), n));
            }
            if c.contains("look_at(") || c.contains("ask_with(") {
                looks.push((name.clone(), inside_fn(&source, n), n));
            }
            if c.contains("look_inside(") && !c.contains("fn look_inside") {
                callers.push((name.clone(), inside_fn(&source, n), n));
            }
        }
    }
    let only = |found: &[(String, String, usize)], what: &str| {
        assert_eq!(found.len(), 1, "{what} in more than one place, or none: {found:?}");
        assert_eq!((found[0].0.as_str(), found[0].1.as_str()), ("app.rs", "look_inside"), "{what}: {found:?}");
    };
    only(&pictures, "the window's picture is asked for");
    only(&looks, "a model is shown a picture");
    assert_eq!(callers.len(), 1, "look_inside is reached from more than one place: {callers:?}");
    assert_eq!((callers[0].0.as_str(), callers[0].1.as_str()), ("app.rs", "think"),
               "a look is something a turn does: {callers:?}");
}

#[test]
fn a_point_inside_is_made_in_one_file_and_never_by_the_core() {
    let mut made = Vec::new();
    for path in rust_files(&serve()) {
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        let source = std::fs::read_to_string(&path).unwrap();
        for (n, line) in code_lines(&source) {
            if code(line).contains("inside: Some(") {
                made.push(format!("{name}:{n}: {}", line.trim()));
            }
        }
    }
    assert_eq!(made.len(), 1, "a point inside is made in more than one place: {made:#?}");
    assert!(made[0].starts_with("looking.rs:"), "{made:#?}");
    let core = root().join("native").join("argmind-core").join("src");
    for path in rust_files(&core) {
        let source = std::fs::read_to_string(&path).unwrap();
        for (n, line) in code_lines(&source) {
            assert!(!code(line).contains("inside: Some("),
                    "the core makes a point inside at {}:{n} — the reader of a model's reply lives here: {}",
                    path.display(), line.trim());
        }
    }
}

#[test]
fn a_point_inside_always_waits_for_a_person() {
    let app = std::fs::read_to_string(serve().join("app.rs")).unwrap();
    let line = code_lines(&app).into_iter().map(|(_, l)| code(l))
        .find(|l| l.contains("let must_confirm ="))
        .expect("propose no longer decides what must wait");
    assert!(line.contains("effect.inside.is_some()"),
            "a point the picture found could run without a press: {}", line.trim());
    //: And `must_confirm` is still what holds the automatic path back.
    assert!(app.contains("if !must_confirm &&"), "must_confirm no longer gates the automatic run");
}

#[test]
fn the_marked_picture_is_made_after_the_look_and_read_by_its_route() {
    let mut made = Vec::new();
    let mut read = Vec::new();
    for path in rust_files(&serve()) {
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        let source = std::fs::read_to_string(&path).unwrap();
        for (n, line) in code_lines(&source) {
            let c = code(line);
            if name != "platform.rs" && c.contains("marked_picture(") {
                made.push((name.clone(), inside_fn(&source, n), n));
            }
            if c.contains("looking::shown(") {
                read.push((name.clone(), n));
            }
        }
    }
    assert_eq!(made.len(), 1, "the marked picture is made in more than one place, or none: {made:?}");
    assert_eq!((made[0].0.as_str(), made[0].1.as_str()), ("app.rs", "look_inside"), "{made:?}");
    let app = std::fs::read_to_string(serve().join("app.rs")).unwrap();
    let lines: Vec<&str> = app.lines().collect();
    //: **After the model answered**: the one `look_at(` comes before the mark, so the picture a
    //: model was shown is never one with a point already drawn on it.
    let asked = lines.iter().position(|l| code(l).contains("look_at(")).expect("look_inside asks no model");
    assert!(asked + 1 < made[0].2, "the picture is marked before the model is shown it");
    assert!(!code(lines[asked]).contains("marked") && !code(lines[asked]).contains("looked"),
            "a model is shown the marked picture: {}", lines[asked].trim());
    //: **And read by its route**, which writes it to the page.
    assert_eq!(read.len(), 1, "the marked picture is read in more than one place, or none: {read:?}");
    assert_eq!(read[0].0, "app.rs", "{read:?}");
    let arm = lines[..read[0].1].iter().rev().find(|l| l.trim_start().starts_with("Route::")).copied().unwrap_or("");
    assert!(arm.contains("Route::Looked"), "the marked picture is read outside its route, under `{}`", arm.trim());
}

/// **The door**: a rule about names that were renamed watches nothing.
#[test]
fn the_door_this_rule_guards_still_exists() {
    let looking = std::fs::read_to_string(serve().join("looking.rs")).unwrap();
    assert!(looking.contains("pub fn effect(") && looking.contains("pub fn landed("));
    let platform = std::fs::read_to_string(serve().join("platform.rs")).unwrap();
    assert!(platform.contains("pub fn window_picture(") && platform.contains("pub fn marked_picture("));
    assert!(looking.contains("pub fn shown("));
    let intent = std::fs::read_to_string(root().join("native/argmind-core/src/intent.rs")).unwrap();
    assert!(intent.contains("pub struct Inside") && intent.contains("pub inside: Option<Inside>"));
}
