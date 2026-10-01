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
