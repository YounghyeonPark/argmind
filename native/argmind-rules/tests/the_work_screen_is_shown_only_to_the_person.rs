//! **The work screen's picture goes to the person, through one route, and to no model.**
//!
//! Since 2026-11-08 argmind's own desktop (`argmind_env_windows::screen`) draws itself into a
//! `View` so the owner can watch the work — *작업화면 보여주는 칸*. That is a picture inside the
//! application, and the rule everything here hangs on is about what a model is shown: a picture of
//! a screen is coordinates by another name. So what is held is where it can go:
//!
//! * `platform.rs` is the one file that makes a `Screen::shown(` or holds the `View`;
//! * `work_picture(` is called exactly once outside it, in `app.rs`, and that line is the route
//!   that writes it to the page;
//! * the core — where prompts, tasks and tool results are made — never names the picture's type.
//!
//! **A different thing from the camera's rule**, which is about the room (`the_app_takes_no_photograph`):
//! this picture is of a desktop the person never sees otherwise, made by argmind, and holds nobody.
//!
//! Checked by breaking it: a second `work_picture(` in `app.rs`, one in `route.rs`, and a
//! `Screen::shown(` in `app.rs` each fail; so does `screen::Picture` named in `argmind-core`.

use std::path::{Path, PathBuf};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf()
}

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

/// Code only — a rule that forbids writing down what it forbids is one nobody can document.
fn code(source: &str) -> Vec<(usize, String)> {
    source.lines().enumerate().map(|(n, l)| (n + 1, l.split("//").next().unwrap_or("").to_string())).collect()
}

#[test]
fn one_route_reads_the_work_screen_and_it_writes_to_the_page() {
    let app = root().join("argmind-serve");
    let mut reads = Vec::new();
    let mut makes = Vec::new();
    for path in sources(&app) {
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        let source = std::fs::read_to_string(&path).unwrap_or_default();
        for (n, line) in code(&source) {
            if name == "platform.rs" {
                continue;
            }
            if line.contains("work_picture(") {
                reads.push((name.clone(), n, line.trim().to_string()));
            }
            if line.contains("Screen::shown(") || line.contains("screen::View") || line.contains("screen::Picture") {
                makes.push(format!("{name}:{n}: {}", line.trim()));
            }
        }
    }
    assert!(makes.is_empty(), "only platform.rs may make or hold the work screen's picture: {makes:#?}");
    assert_eq!(reads.len(), 1, "the work screen's picture should be read in one place: {reads:#?}");
    let (file, n, _) = &reads[0];
    assert_eq!(file, "app.rs", "{reads:#?}");
    //: **And that place is the route**: the arm it sits in is `Route::WorkScreen`.
    let source = std::fs::read_to_string(app.join("src/app.rs")).unwrap();
    let before: Vec<&str> = source.lines().take(*n).collect();
    let arm = before.iter().rev().find(|l| l.trim_start().starts_with("Route::")).copied().unwrap_or("");
    assert!(arm.contains("Route::WorkScreen"), "work_picture is read outside its route, under `{}`", arm.trim());
}

#[test]
fn the_core_never_names_the_picture() {
    for path in sources(&root().join("argmind-core")) {
        let source = std::fs::read_to_string(&path).unwrap_or_default();
        for (n, line) in code(&source) {
            assert!(!line.contains("screen::Picture") && !line.contains("screen::View") && !line.contains("work_picture"),
                    "{}:{n} names the work screen's picture in the core: {}", path.display(), line.trim());
        }
    }
}

/// The door: a rule about names that were renamed watches nothing.
#[test]
fn the_door_this_rule_guards_still_exists() {
    let screen = std::fs::read_to_string(root().join("argmind-env-windows/src/screen.rs")).unwrap();
    assert!(screen.contains("pub fn shown(view: View) -> Self") && screen.contains("pub struct Picture"));
    let platform = std::fs::read_to_string(root().join("argmind-serve/src/platform.rs")).unwrap();
    assert!(platform.contains("pub fn work_picture(seen: u64)") && platform.contains("Screen::shown("));
}
