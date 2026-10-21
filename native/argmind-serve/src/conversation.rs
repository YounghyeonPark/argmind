//! **The conversation, kept** — the owner, 2026-10-21: *대화는 파일로 기록하고 모델에 맞는 크기의
//! 컨텍스트를 유지하면서*.
//!
//! `%LOCALAPPDATA%\\argmind\\conversation.jsonl`, one line a turn-line, appended as it happens and
//! read back at start, so closing argmind no longer ends the conversation. **What reaches the model
//! is a separate decision**: `Session::context_tokens` fills the prompt with as many whole turns as
//! fit, and the file holds more than that. `--bin context` measured the pair: a fact from the first
//! turn recalled 3 of 3 after twenty turns, against 0 of 3 under the old twelve-turn cut, and
//! choosing 31 of 37 with that history against 30 with none.
//!
//! **It is the person's words in a file**, which is why the panel says where it is and has a button
//! that empties it. Nothing a model wrote about the conversation is kept — no summaries: the one
//! time this model was asked to describe itself it described things it had not done.

use std::io::Write;
use std::path::PathBuf;

/// How many lines are read back at start. Far more than any prompt carries; the prompt takes what
/// fits from the newest end.
pub const READ_BACK: usize = 2000;
/// Past this many lines the file is cut back to [`READ_BACK`] when argmind starts.
pub const MOST_LINES: usize = 20_000;

pub fn at() -> PathBuf {
    let mut path = std::env::var("LOCALAPPDATA").map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir());
    path.push("argmind");
    path.push("conversation.jsonl");
    path
}

/// The conversation so far, oldest first — and the file trimmed when it has grown too long.
pub fn remembered() -> Vec<(String, String)> {
    let Ok(text) = std::fs::read_to_string(at()) else { return Vec::new() };
    let lines = text.lines().count();
    if lines > MOST_LINES {
        let keep: Vec<&str> = text.lines().skip(lines - READ_BACK).collect();
        let _ = std::fs::write(at(), keep.join("\n") + "\n");
    }
    argmind_core::session::history_from(&text, READ_BACK)
}

/// Add these lines to the end of the file.
pub fn append(lines: &[(String, String)]) -> Result<(), String> {
    if lines.is_empty() {
        return Ok(());
    }
    let path = at();
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let now = crate::activity::now();
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(&path)
        .map_err(|e| format!("{path:?}: {e}"))?;
    for (who, text) in lines {
        writeln!(f, "{}", argmind_core::session::history_line(now, who, text))
            .map_err(|e| format!("{path:?}: {e}"))?;
    }
    Ok(())
}

/// Empty it.
pub fn forget() -> Result<(), String> {
    match std::fs::remove_file(at()) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(format!("{e}")),
    }
}
