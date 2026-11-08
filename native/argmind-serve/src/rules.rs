//! **규칙** — where argmind may not act, and where it must always ask. What a rule is and how one
//! is matched is `argmind_core::rule`; this is where the list lives.
//!
//! `%LOCALAPPDATA%\argmind\rules.json`, its own file like `tact.json`, readable and editable by
//! hand. **No file is no rules**, which is today's behaviour exactly.

use argmind_core::rule::Rule;
use std::path::PathBuf;

pub fn at() -> PathBuf {
    let mut path = std::env::var("LOCALAPPDATA").map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir());
    path.push("argmind");
    path.push("rules.json");
    path
}

pub fn remembered() -> Vec<Rule> {
    std::fs::read_to_string(at()).map(|t| argmind_core::rule::parse_all(&t)).unwrap_or_default()
}

pub fn keep(rules: &[Rule]) -> Result<(), String> {
    let path = at();
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    std::fs::write(&path, argmind_core::rule::render_all(rules)).map_err(|e| format!("{path:?}: {e}"))
}

/// What a press or a write on argmind's work screen may do.
#[derive(Clone, Debug, PartialEq)]
pub enum OnTheWorkScreen {
    Go,
    /// Ask the person first — a *항상 물어봄* rule, or 확인 없이 실행 not on.
    Ask,
    /// A *하지 않음* rule; the person's sentence for it.
    Refuse(String),
}

/// **The person's 규칙 reach the work screen the way they reach the desktop** — matched on the
/// window's title and the kind, the strictest one winning — and so does 확인 없이 실행: a task is
/// something a person asked for, so `acting` is the standing answer it reads, the same as a
/// task's press in the web pane, which goes through `propose`.
pub fn on_the_work_screen(rules: &[Rule], window: &str, kind: &str, acting: bool) -> OnTheWorkScreen {
    use argmind_core::rule::{applies, told, Then};
    match applies(rules, window, kind) {
        Some(Then::Never) => OnTheWorkScreen::Refuse(told(rules, window, kind)),
        Some(Then::Confirm) => OnTheWorkScreen::Ask,
        None if !acting => OnTheWorkScreen::Ask,
        None => OnTheWorkScreen::Go,
    }
}
