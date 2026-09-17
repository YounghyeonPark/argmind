//! The thinking goal and the dials, remembered between runs.
//!
//! The third file of this shape, beside [`crate::decided`] (may argmind do this?) and
//! [`crate::chosen`] (with which microphone?). **A third file rather than a third section**, for
//! the reason the second one is separate from the first: *잊어버려* on a permission, *기본 장치로*
//! on a headset and *재치를 0으로* are three different undos, and a single file makes them look
//! like one.
//!
//! ## What is written down
//!
//! ```json
//! {
//!   "재치": 7,
//!   "목표": "코드 리뷰 중에 놓친 것만 말해줘"
//! }
//! ```
//!
//! Flat, hand-written and hand-read, because this crate takes no third-party dependency and the
//! other two files here are read the same way. **A dial at zero is not written**: the file says
//! what was turned, and an absent name is the resting value rather than a missing one.
//!
//! ## What a broken file means
//!
//! Rest. A name nobody knows is dropped, a number that will not parse is dropped, and a file that
//! will not read at all is no file. That is the same rule the other two keep, and here it has an
//! extra edge: **the resting mind renders the prompt argmind shipped with, byte for byte**, so the
//! worst a corrupt file can do is turn the feature off.

use std::path::PathBuf;

use argmind_core::mind::{Mind, DIALS, MOST};

/// The key the sentence is written under. Not a dial, and checked before the dial list so a future
/// dial called 목표 could not shadow it silently.
pub const AIM: &str = "목표";

/// Where the file lives — beside the decisions and the devices.
pub fn at() -> PathBuf {
    let mut path = std::env::var("LOCALAPPDATA").map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir());
    path.push("argmind");
    path.push("mind.json");
    path
}

/// What was set. **An unreadable file is a resting mind rather than an error.**
pub fn remembered() -> Mind {
    std::fs::read_to_string(at()).map(|t| parse(&t)).unwrap_or_else(|_| Mind::resting())
}

/// Is this a dial anybody can turn?
pub fn is_a_dial(name: &str) -> bool {
    DIALS.iter().any(|(k, _)| *k == name)
}

/// Hand-read, like the other two.
///
/// The sentence is taken as everything between the first quote after the colon and the **last**
/// quote on the line, then unescaped — a person's goal can contain a colon, a comma and a quote,
/// and splitting on punctuation would eat the first of those.
///
/// **One pair per line, which is what [`render`] writes and is the known limit of reading it this
/// way.** A hand-typed `{"재치": 7}` all on one line does work — the braces come off below — but a
/// hand-typed `{"재치": 7, "유머": 3}` keeps only the first. Said out loud rather than discovered,
/// because the cost of being wrong here is a dial that silently stays at rest.
pub fn parse(text: &str) -> Mind {
    let mut mind = Mind::resting();
    for line in text.lines() {
        let line = line.trim().trim_start_matches('{').trim().trim_end_matches(',');
        let Some((left, right)) = line.split_once(':') else { continue };
        let key = left.trim().trim_matches('"');
        let right = right.trim();
        if key == AIM {
            if let Some((inside, _)) = right.strip_prefix('"').and_then(|r| r.rsplit_once('"')) {
                mind.aim = unescape(inside);
            }
            continue;
        }
        if !is_a_dial(key) {
            continue;
        }
        //: A hand-edited `7,` has already lost its comma above; a hand-edited `"7"` is still a
        //: number to a person, so the quotes come off before the parse — and so does the closing
        //: brace of a file somebody wrote on one line.
        if let Ok(n) = right.trim_end_matches('}').trim().trim_matches('"').parse::<u8>() {
            mind.turn(key, n.min(MOST));
        }
    }
    mind
}

pub fn render(mind: &Mind) -> String {
    let mut rows: Vec<String> = DIALS.iter().filter_map(|(korean, _)| {
        //: **Zero is absent, not written.** The file then says only what somebody did.
        let n = mind.of(korean);
        (n > 0).then(|| format!("  \"{korean}\": {n}"))
    }).collect();
    let aim = mind.aim.trim();
    if !aim.is_empty() {
        rows.push(format!("  \"{AIM}\": \"{}\"", escape(aim)));
    }
    if rows.is_empty() {
        return "{}\n".to_string();
    }
    format!("{{\n{}\n}}\n", rows.join(",\n"))
}

fn escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n").replace('\r', "")
}

fn unescape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('n') => out.push('\n'),
            Some('"') => out.push('"'),
            Some('\\') => out.push('\\'),
            //: Anything else keeps both characters. This is not a JSON reader and pretending
            //: otherwise would silently eat a Windows path somebody pasted into their goal.
            Some(other) => {
                out.push('\\');
                out.push(other);
            }
            None => out.push('\\'),
        }
    }
    out
}

/// Turn one dial and write the file. **An unknown name is refused rather than ignored** — the page
/// cannot invent a dial, and a 400 says so while it still means something.
pub fn turn(name: &str, to: u8) -> Result<Mind, String> {
    if !is_a_dial(name) {
        return Err(format!("{name} 라는 항목은 없습니다"));
    }
    let mut mind = remembered();
    mind.turn(name, to);
    write(&mind)?;
    Ok(mind)
}

/// Write the goal. An empty one clears it.
pub fn aim(text: &str) -> Result<Mind, String> {
    let mut mind = remembered();
    //: A goal is a sentence, and **the length is capped where a sentence stops being one**. Nothing
    //: here is measured; it is a guard against a paste of a whole file ending up in every prompt.
    mind.aim = text.trim().chars().take(400).collect();
    write(&mind)?;
    Ok(mind)
}

fn write(mind: &Mind) -> Result<(), String> {
    let where_to = at();
    if let Some(parent) = where_to.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("{e}"))?;
    }
    std::fs::write(&where_to, render(mind))
        .map_err(|e| format!("{} 에 쓰지 못했습니다 ({e})", where_to.display()))
}
