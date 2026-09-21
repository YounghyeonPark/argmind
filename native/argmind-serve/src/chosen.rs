//! Which microphone, which speaker, which camera, which voice — remembered between runs.
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

/// The fourth thing this file holds, and **it is not a device.**
///
/// A preference about how argmind sounds, which is what this file is for, kept out of [`KINDS`]
/// because there is no hardware to enumerate and nothing to open and test the way a microphone is
/// tested.
///
/// **It holds a preset key, and it used to hold a seed.** Both were the answer to *which voice*, a
/// day apart: a seed reproduces one recording exactly and turned out not to name a speaker at all
/// across different sentences, so what is remembered now is one of [`crate::speaker::PRESETS`],
/// each of which is a *cloned* voice that does. A file written by the earlier version holds a bare
/// number, which is not a preset key and falls back to the default — see
/// [`crate::speaker::preset_or_default`].
pub const VOICE: &str = "voice";

/// Everything this file may hold. `KINDS` is what the page offers a list for; this is what
/// [`parse`] will read back.
pub const KEYS: [&str; 4] = ["microphone", "speaker", "camera", VOICE];



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

/// Is this something the file may hold? Wider than [`is_a_kind`] by exactly [`VOICE`].
pub fn is_a_key(key: &str) -> bool {
    KEYS.contains(&key)
}

/// Which voice. **Never `None`** — a missing, unknown or hand-mangled one is the default preset,
/// because the one answer this must not give is *whatever the sampler feels like*.
pub fn voice() -> &'static crate::speaker::Preset {
    voice_in(&remembered())
}

/// The same, over a map somebody hands in. **Split out so it can be tested without a file**, the
/// way `attended_where` is: the interesting part is what a missing or stale value falls back to,
/// and reaching it through `devices.json` would mean a test that writes the owner's settings.
pub fn voice_in(all: &BTreeMap<String, String>) -> &'static crate::speaker::Preset {
    crate::speaker::preset_or_default(all.get(VOICE).map(String::as_str).unwrap_or("").trim())
}

/// Remember a voice, by its preset key.
pub fn keep_voice(key: &str) -> Result<(), String> {
    if crate::speaker::preset(key).is_none() {
        return Err(format!("{key} 라는 목소리는 없습니다"));
    }
    keep(VOICE, Some(key))
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
        if !is_a_key(kind) {
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
    if !is_a_key(kind) {
        return Err(format!("{kind} 라는 설정은 없습니다"));
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
