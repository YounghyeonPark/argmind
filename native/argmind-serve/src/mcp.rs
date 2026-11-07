//! **The MCP servers a task may use** — `%LOCALAPPDATA%\argmind\mcp.json`, since 2026-11-08.
//!
//! The person's file, in the format other MCP clients read, so a server set up for one is set up
//! for argmind by copying a block: `{"mcpServers": {"name": {"command", "args", "env"}}}`. **Absent
//! at rest**, and then a task is exactly what it was. Started on the task's own thread when a task
//! begins and stopped when it ends, because a server is a program running as the person and
//! nothing asked it to outlive the work. `argmind-env-mcp` has what a server is.

use std::path::PathBuf;

use argmind_env_mcp::{configured_text, Servers};

pub fn at() -> PathBuf {
    let mut path = std::env::var("LOCALAPPDATA").map(PathBuf::from).unwrap_or_else(|_| std::env::temp_dir());
    path.push("argmind");
    path.push("mcp.json");
    path
}

/// The servers, started, and a line for the timeline about each that was not — or nothing at all
/// when the file is not there. **A file that does not read is said, not ignored**: the person wrote
/// it meaning something, and silence would read as *argmind has no such tools*.
pub fn started() -> (Option<Servers>, Vec<String>) {
    let Ok(text) = std::fs::read_to_string(at()) else { return (None, Vec::new()) };
    let (list, mut notes) = match configured_text(&text) {
        Ok(c) => c,
        Err(e) => return (None, vec![format!("mcp.json 을 읽지 못했습니다 — {e}")]),
    };
    if list.is_empty() {
        return (None, notes);
    }
    let (set, failed) = Servers::start(&list);
    notes.extend(failed);
    for s in &set.servers {
        let n = s.specs().len();
        let more = if s.dropped > 0 { format!(" (나머지 {}개는 너무 많아 뺐습니다)", s.dropped) } else { String::new() };
        notes.push(format!("MCP {}: 도구 {n}개{more}", s.name));
    }
    (if set.servers.is_empty() { None } else { Some(set) }, notes)
}
