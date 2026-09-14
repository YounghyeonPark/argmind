//! What a harness did **not** send, made structural instead of remembered.
//!
//! **Three of the four harnesses here send no history at all.** They call the model with a screen
//! and one line, and they reproduced their baselines digit for digit under a change that made the
//! agent click the eraser when greeted. So *"no slide on those three"* is vacuous for anything
//! touching the conversation, and only the distance sweep can see it.
//!
//! That was a thing people remembered until it was not. Now every sender counts what it sent, and
//! prints a caveat naming which conversation run — if any — covers the axis it is blind to. The
//! caveat has **three states** and the difference between them is the point:
//!
//! * a run exists at this prompt **and** this `loop.py`;
//! * a run exists at this prompt but under a different `loop.py`, so the window and the wording of
//!   a history line have moved underneath it;
//! * nothing has run the history axis at this prompt at all.
//!
//! The middle state exists because a prompt fingerprint covers the prompt and **not** `loop.py`,
//! which decides how much of the conversation reaches it and what each line says.

use std::path::{Path, PathBuf};

use crate::print6;

/// Six characters over a source file with `\r\n` normalised to `\n`.
///
/// **Not the byte hash.** The same file checked out on Windows and on Linux differs in every line
/// ending and is the same file; a fingerprint that moves under a checkout is one that fires on
/// something nobody changed. `l` in a filename means this one.
pub fn norm_print(path: &Path) -> Option<String> {
    let bytes = std::fs::read(path).ok()?;
    let normalised: Vec<u8> = String::from_utf8_lossy(&bytes).replace("\r\n", "\n").into_bytes();
    Some(print6(&normalised))
}

/// How many turns a harness sent with a conversation, and how many without.
///
/// Counted rather than declared: a harness that says it sends history and does not is exactly the
/// failure this module exists for.
#[derive(Default, Debug, PartialEq)]
pub struct Sent {
    pub with: usize,
    pub without: usize,
}

impl Sent {
    pub fn turn(&mut self, history: Option<&[(String, String)]>) {
        match history {
            Some(h) if !h.is_empty() => self.with += 1,
            _ => self.without += 1,
        }
    }
}

/// One recorded conversation run: the file, the prompt tag it was taken at, and the `loop.py` it
/// ran under.
#[derive(Debug, Clone)]
pub struct Run {
    pub path: PathBuf,
    pub prompt: Option<String>,
    pub loop_print: Option<String>,
    pub modified: std::time::SystemTime,
}

fn field(head: &str, key: &str) -> Option<String> {
    //: Read out of the first stretch of the file rather than by parsing it. These results run to
    //: two megabytes and this is called on every harness's last line.
    let needle = format!("\"{key}\":");
    let at = head.find(&needle)? + needle.len();
    let rest = &head[at..];
    let start = rest.find('"')? + 1;
    let end = rest[start..].find('"')? + start;
    Some(rest[start..end].to_string())
}

/// Every `conversation-*.json` in the directory, newest first.
pub fn conversation_runs(dir: &Path) -> Vec<Run> {
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else { return out };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = path.file_name().unwrap_or_default().to_string_lossy().to_string();
        if !name.starts_with("conversation-") || !name.ends_with(".json") {
            continue;
        }
        let Ok(bytes) = std::fs::read(&path) else { continue };
        let head = String::from_utf8_lossy(&bytes[..bytes.len().min(65536)]).to_string();
        let prompt = field(&head, "prompt").map(|hex| {
            let letter = field(&head, "prompt_scheme")
                .and_then(|s| s.chars().next()).unwrap_or('p');
            format!("{letter}{hex}")
        });
        let modified = entry.metadata().and_then(|m| m.modified())
            .unwrap_or(std::time::UNIX_EPOCH);
        out.push(Run { path, prompt, loop_print: field(&head, "loop_print"), modified });
    }
    out.sort_by(|a, b| b.modified.cmp(&a.modified));
    out
}

/// Which conversation runs, if any, cover the history axis for this prompt and this `loop.py`.
#[derive(Debug)]
pub struct Deferral {
    pub prompt: String,
    pub loop_print: String,
    pub at_this_prompt: Vec<String>,
    pub at_this_prompt_and_loop: Vec<String>,
    pub loop_prints_at_this_prompt: Vec<String>,
    pub newest_run: Option<String>,
}

impl Deferral {
    pub fn exercised_at_this_prompt(&self) -> bool {
        !self.at_this_prompt.is_empty()
    }
    pub fn exercised_at_this_tree(&self) -> bool {
        !self.at_this_prompt_and_loop.is_empty()
    }
}

pub fn deferral(dir: &Path, tag: &str, loop_print: &str) -> Deferral {
    let runs = conversation_runs(dir);
    let named = |r: &Run| r.path.file_name().unwrap().to_string_lossy().to_string();
    let at: Vec<String> = runs.iter()
        .filter(|r| r.prompt.as_deref() == Some(tag)).map(named).collect();
    let at_tree: Vec<String> = runs.iter()
        .filter(|r| r.prompt.as_deref() == Some(tag) && r.loop_print.as_deref() == Some(loop_print))
        .map(named).collect();
    let mut prints: Vec<String> = runs.iter()
        .filter(|r| r.prompt.as_deref() == Some(tag))
        .map(|r| r.loop_print.clone().unwrap_or_else(|| "None".into())).collect();
    prints.sort();
    prints.dedup();
    Deferral { prompt: tag.into(), loop_print: loop_print.into(), at_this_prompt: at,
               at_this_prompt_and_loop: at_tree, loop_prints_at_this_prompt: prints,
               newest_run: runs.first().map(named) }
}

/// The sentence a history-free harness prints beside its headline.
///
/// **It is not decoration.** The number above it was produced without a conversation in the prompt,
/// and a reader who does not know that will read it as covering something it cannot see.
pub fn caveat(dir: &Path, tag: &str, loop_print: &str) -> String {
    let d = deferral(dir, tag, loop_print);
    if d.exercised_at_this_tree() {
        return format!("   [no conversation sent; conversation.py has run at {tag} + loop {}]",
                       d.loop_print);
    }
    if d.exercised_at_this_prompt() {
        return format!("   [no conversation sent; conversation.py has run at {tag} but under a \
                        different loop.py ({} -> {})]",
                       d.loop_prints_at_this_prompt.join("/"), d.loop_print);
    }
    format!("   [no conversation sent; NOTHING has run the history axis at {tag}]")
}
