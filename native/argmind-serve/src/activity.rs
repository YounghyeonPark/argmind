//! **한 일** — what argmind heard, saw, said and did, in the order it happened.
//!
//! Item 5 of the dots comparison: an agent that acts on a desktop owes the person an account of
//! itself — *what did you do while I was away?* — that does not depend on the model remembering.
//! The transcript is the conversation; this is everything else beside it: the window it was looking
//! at, what it noticed, what it offered, what ran and whether the screen moved, what it refused and
//! why, what it held back.
//!
//! ## Where it comes from
//!
//! **From what the page is told, in the one place it is told.** Every event goes through
//! `Listeners::send`, and [`reading`] turns the ones worth keeping into a line. A new code path
//! that tells the page something is in the timeline without anybody remembering to log it — which
//! is the property a log meant to account for a program has to have, because the path nobody
//! remembered is the one somebody will ask about.
//!
//! The one thing not sent is what the person typed — the page draws that itself — so it is
//! recorded beside the turn, explicitly.
//!
//! ## Where it lives
//!
//! **In memory, the last [`KEPT`] entries, gone when argmind closes.** Writing the timeline to disk
//! would put the person's words and every window title in a file, and that is a decision about
//! privacy with its own switch, not a side effect of a log. What survives a restart already does
//! so on purpose: the errands, the kept sentences, the rules.

use std::collections::VecDeque;

/// How many entries are kept. Chosen: at a window change every few seconds and a turn a minute this
/// is an afternoon, and a list longer than that is not one anybody reads.
pub const KEPT: usize = 500;

#[derive(Clone, Debug, PartialEq)]
pub struct Entry {
    /// Unix seconds. **The page turns it into a time**, in the person's own zone.
    pub at: u64,
    /// The program's word for what happened — never the model's.
    pub kind: &'static str,
    pub text: String,
}

/// **Which events become a line, and what the line says.** Everything else the page is told — the
/// busy flag, the device lists, the settings echoes — is the interface talking to itself.
pub fn reading(event: &str, data: &str) -> Option<(&'static str, String)> {
    let data = data.trim();
    Some(match event {
        "them" => ("들음", data.to_string()),
        "say" if !data.is_empty() => ("말함", data.to_string()),
        "noticed" => ("알아챔", data.to_string()),
        "note" => ("알림", data.to_string()),
        "attending" if !data.is_empty() => ("보는 창", data.to_string()),
        "keep-offer" => ("기억 제안", data.to_string()),
        "lookup-offer" => ("찾아보기 제안", data.to_string()),
        "propose" => ("제안", field(data, "label")?),
        "ran" => {
            let how = match data.contains("\"confirmed\":true") {
                true => "화면이 바뀜",
                false if data.contains("\"confirmed\":false") => "화면이 바뀌지 않음",
                false => "확인하지 못함",
            };
            ("실행", format!("{} — {how}", field(data, "label")?))
        }
        "gone" => ("못 함", format!("{} — 그새 사라짐", field(data, "what")?)),
        "wonder" if data != "null" => ("물음", field(data, "asked")?),
        "held" if data == "1" => ("보류", "할 말을 미뤄 둠".to_string()),
        _ => return None,
    })
}

/// One string field out of JSON this program wrote itself. **Not a JSON reader** — this crate writes
/// its JSON by hand and takes no parser — so it reads exactly the escapes `app::escape` writes.
pub fn field(json: &str, key: &str) -> Option<String> {
    let start = json.find(&format!("\"{key}\":\""))? + key.len() + 4;
    let mut out = String::new();
    let mut chars = json[start..].chars();
    while let Some(c) = chars.next() {
        match c {
            '"' => return Some(out),
            '\\' => match chars.next()? {
                'n' => out.push('\n'),
                'r' => out.push('\r'),
                't' => out.push('\t'),
                'u' => {
                    let hex: String = chars.by_ref().take(4).collect();
                    out.push(char::from_u32(u32::from_str_radix(&hex, 16).ok()?)?);
                }
                other => out.push(other),
            },
            c => out.push(c),
        }
    }
    None
}

/// The timeline itself.
#[derive(Default)]
pub struct Activity(VecDeque<Entry>);

impl Activity {
    /// Add one. **The same window twice in a row is one line** — the header is re-sent whenever a
    /// turn starts, and a timeline that says *보는 창: 메모장* five times has said it once.
    pub fn push(&mut self, at: u64, kind: &'static str, text: &str) -> Option<&Entry> {
        let text: String = text.trim().chars().take(400).collect();
        if text.is_empty() {
            return None;
        }
        if kind == "보는 창" && self.0.iter().rev().find(|e| e.kind == "보는 창")
            .is_some_and(|e| e.text == text) {
            return None;
        }
        self.0.push_back(Entry { at, kind, text });
        while self.0.len() > KEPT {
            self.0.pop_front();
        }
        self.0.back()
    }

    pub fn clear(&mut self) {
        self.0.clear();
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// For the page, oldest first, on one line.
    pub fn json(&self) -> String {
        format!("[{}]", self.0.iter().map(entry_json).collect::<Vec<_>>().join(","))
    }
}

pub fn entry_json(e: &Entry) -> String {
    format!("{{\"at\":{},\"kind\":\"{}\",\"text\":\"{}\"}}", e.at, e.kind, escape(&e.text))
}

fn escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

pub fn now() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs()).unwrap_or(0)
}
