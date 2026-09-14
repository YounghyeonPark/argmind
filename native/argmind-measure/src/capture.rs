//! Saving real widget lists, so choosing can be measured without a live desktop.
//!
//! **Why a fixture and not a live run.** The question is *"given what the environment saw, does the
//! model pick the right thing?"* — one question, about choosing. A live run answers it mixed with
//! two others: whether the window was in front at the right moment, and whether the click landed. A
//! fixture also makes the measurement repeatable when the prompt changes, which it will, and a
//! number that cannot be re-run is a number that stops being true without anybody noticing.
//!
//! **These are real screens from a working desktop, not screens invented to be answerable.** A
//! hand-written surface would contain exactly the widgets a goal needs and nothing else, and would
//! measure the fixture rather than the model.
//!
//! The bytes matter here in a way they do not elsewhere. `screen_print` — `s` in every harness's
//! filename — is a hash of these files as they sit on disk, so a writer that produced the same
//! *content* in a different *encoding* would move `s` on a re-capture and make every result look
//! incomparable with every other. The committed fixtures are CRLF with no trailing newline, because
//! Python's `write_text` translates on Windows; [`screen_json`] reproduces that exactly, and the
//! test round-trips all four committed files byte for byte rather than taking this paragraph's word
//! for it.

use argmind_core::contract::Widget;

/// A window title as a filename: lower case, alphanumerics and spaces kept, everything else a
/// hyphen, spaces to hyphens, 48 characters.
pub fn slug(title: &str) -> String {
    let kept: String = title.to_lowercase().chars()
        .map(|c| if c.is_alphanumeric() || c == ' ' || c == '-' || c == '_' { c } else { '-' })
        .collect();
    let out: String = kept.trim().replace(' ', "-").chars().take(48).collect();
    if out.is_empty() { "untitled".into() } else { out }
}

/// One captured screen, in the bytes the committed fixtures are written in.
///
/// `indent=1`, `ensure_ascii=False`, **CRLF**, and no trailing newline — see the module note.
pub fn screen_json(window: &str, widgets: &[Widget]) -> Vec<u8> {
    let mut out = String::from("{\n \"window\": ");
    out.push_str(&serde_json::to_string(window).unwrap());
    out.push_str(",\n \"widgets\": [");
    if widgets.is_empty() {
        out.push(']');
    } else {
        for (i, w) in widgets.iter().enumerate() {
            out.push_str(if i == 0 { "\n  {\n" } else { ",\n  {\n" });
            out.push_str(&format!("   \"name\": {},\n", serde_json::to_string(&w.name).unwrap()));
            out.push_str(&format!("   \"role\": {},\n", serde_json::to_string(&w.role).unwrap()));
            out.push_str("   \"rect\": [\n");
            let r = [w.rect.0, w.rect.1, w.rect.2, w.rect.3];
            for (j, v) in r.iter().enumerate() {
                out.push_str(&format!("    {v}{}\n", if j == 3 { "" } else { "," }));
            }
            out.push_str("   ]\n  }");
        }
        out.push_str("\n ]");
    }
    out.push_str("\n}");
    //: **The line endings are part of the fingerprint.** Written last, over the whole document, so
    //: no branch above can forget one.
    out.replace('\n', "\r\n").into_bytes()
}

/// The richest reading of each window seen, by title.
///
/// **Keep the richest, not the first.** A window caught mid-paint exposes a fraction of its
/// widgets, and that would quietly make every goal against it unanswerable — a failure that looks
/// like the model choosing badly.
#[derive(Default)]
pub struct Kept {
    pub windows: std::collections::BTreeMap<String, Vec<Widget>>,
}

impl Kept {
    /// Offer a reading. Returns `true` when it replaced what was there, which is what the live
    /// capture prints.
    pub fn offer(&mut self, widgets: Vec<Widget>) -> bool {
        let Some(first) = widgets.first() else { return false };
        let title = first.window.clone();
        if widgets.len() > self.windows.get(&title).map_or(0, Vec::len) {
            self.windows.insert(title, widgets);
            return true;
        }
        false
    }
}
