//! Which microphone, which speaker, which camera — remembered between runs.
//!
//! **The same shape as [`crate::decided`] and deliberately a different file.** A permission is a
//! yes or a no about something argmind may do; this is a preference about which piece of hardware
//! it does it with. Putting them in one file would mean *forget this device* and *ask me again
//! about the microphone* were the same button, and they are not.
//!
//! ## What is written down
//!
//! `{"microphone": "{0.0.1.00000000}.{b264…}", "camera": "Integrated Camera"}` — an endpoint id for
//! the two audio ones and a name for the camera. That asymmetry is
//! [`argmind_env_windows::devices`]'s and the reasoning is there: an id survives a driver update
//! where a name may not, and Media Foundation has no id a person could pick out of a list.
//!
//! **A device that is written down and gone is not an error.** The environment falls back to
//! whatever Windows prefers and says it fell back; a headset unplugged after being chosen should
//! leave argmind audible, not silent. Nothing here deletes the choice on that account, because the
//! headset will be plugged in again.

use std::collections::BTreeMap;
use std::path::PathBuf;

/// The three kinds. **A fixed list**, for the same reason `DECIDABLE` is one: a hand-edited file
/// with a typo should be ignored rather than carry a setting nobody can name.
pub const KINDS: [&str; 3] = ["microphone", "speaker", "camera"];

/// What to call it in the page.
pub fn what(kind: &str) -> &'static str {
    match kind {
        "microphone" => "마이크",
        "speaker" => "스피커",
        "camera" => "카메라",
        _ => "알 수 없는 장치",
    }
}

pub fn is_a_kind(kind: &str) -> bool {
    KINDS.contains(&kind)
}

/// Where the file lives — beside the decisions, and not in it.
pub fn at() -> PathBuf {
    let mut path = std::env::var("LOCALAPPDATA").map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir());
    path.push("argmind");
    path.push("devices.json");
    path
}

/// What has been chosen. **A file that cannot be read is no choices rather than an error** — the
/// absence of it is the ordinary first run, and a corrupt one should mean *use the defaults*.
pub fn remembered() -> BTreeMap<String, String> {
    std::fs::read_to_string(at()).map(|t| parse(&t)).unwrap_or_default()
}

/// The one chosen for this kind, if any.
pub fn chosen(kind: &str) -> Option<String> {
    remembered().get(kind).cloned()
}

/// Hand-read, like the licence record and the decisions, because this crate takes no third-party
/// dependency.
///
/// **An endpoint id contains braces and dots and no quotes**, which is what makes reading it this
/// simply safe — and a camera name could contain anything, so the value is taken as everything
/// between the first pair of quotes after the colon rather than split on further punctuation.
pub fn parse(text: &str) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    for line in text.lines() {
        let line = line.trim().trim_end_matches(',');
        let Some((left, right)) = line.split_once(':') else { continue };
        let kind = left.trim().trim_matches('"');
        if !is_a_kind(kind) {
            continue;
        }
        let right = right.trim();
        let Some(inside) = right.strip_prefix('"').and_then(|r| r.rsplit_once('"')) else {
            continue;
        };
        if !inside.0.is_empty() {
            out.insert(kind.to_string(), inside.0.to_string());
        }
    }
    out
}

pub fn render(all: &BTreeMap<String, String>) -> String {
    if all.is_empty() {
        return "{}\n".to_string();
    }
    let rows: Vec<String> = all.iter().map(|(k, v)| format!("  \"{k}\": \"{}\"", escape(v)))
        .collect();
    format!("{{\n{}\n}}\n", rows.join(",\n"))
}

/// A camera's name is whatever its driver says, so it goes through this on the way out.
fn escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

/// Choose one, or `None` to go back to whatever the operating system prefers.
pub fn keep(kind: &str, id: Option<&str>) -> Result<(), String> {
    if !is_a_kind(kind) {
        return Err(format!("{kind} 라는 장치 종류는 없습니다"));
    }
    let mut all = remembered();
    match id.filter(|i| !i.is_empty()) {
        Some(id) => all.insert(kind.to_string(), id.to_string()),
        None => all.remove(kind),
    };
    let where_to = at();
    if let Some(parent) = where_to.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("{e}"))?;
    }
    std::fs::write(&where_to, render(&all))
        .map_err(|e| format!("{} 에 쓰지 못했습니다 ({e})", where_to.display()))
}
