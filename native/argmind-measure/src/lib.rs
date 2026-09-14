//! The machinery every harness needs before it can measure anything.
//!
//! Five harnesses in `measure/` each need the captured screens, a fingerprint that changes when
//! those do, a fingerprint of the prompt, the cost of one call, and a result path that refuses to
//! overwrite. Porting them one at a time would reimplement this five times and the five would
//! drift — the same defect `Balanced`'s own docstring records from the other direction: *one
//! scanner with two callers, because written twice the two eventually disagree*.
//!
//! **Every fingerprint here is checked against the Python that produced the committed results.** A
//! port that computes a different `screen_print` does not fail loudly; it writes a result file
//! under a name nobody can compare with anything, which is the failure this project has already
//! had five times.

pub mod choosing;

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use argmind_core::contract::Widget;
use argmind_core::converse::render;
use argmind_core::intent::Trigger;
use sha2::{Digest, Sha256};

/// Six characters, which is what every fingerprint in this project is.
pub fn print6(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes)).chars().take(6).collect()
}

/// A harness's own source, hashed. `h` in a filename.
///
/// **The one condition a harness did not record about itself**, until two arms of one comparison
/// wrote the same path and the second overwrote a committed baseline. Call it with
/// `include_str!(file!())`: the source is hashed at compile time, so unlike the built artifact —
/// three builds of one source gave three different hashes, measured — it is reproducible.
pub fn source_print(source: &str) -> String {
    print6(source.as_bytes())
}

/// One captured screen.
pub struct Screen {
    pub name: String,
    pub window: String,
    pub widgets: Vec<Widget>,
}

/// Every captured screen, by name.
///
/// Read rather than re-captured: a fixture is what makes a measurement repeatable, and a harness
/// that reads the live desktop measures whatever somebody happened to be doing.
pub fn screens(dir: &Path) -> BTreeMap<String, Screen> {
    let mut out = BTreeMap::new();
    let mut paths: Vec<PathBuf> = fs::read_dir(dir).map(|entries| {
        entries.flatten().map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "json")).collect()
    }).unwrap_or_default();
    paths.sort();
    for path in paths {
        let Ok(raw) = fs::read_to_string(&path) else { continue };
        let Ok(d) = serde_json::from_str::<serde_json::Value>(&raw) else { continue };
        let window = d["window"].as_str().unwrap_or("").to_string();
        let widgets = d["widgets"].as_array().map(|a| a.iter().map(|w| {
            let r = w["rect"].as_array().unwrap();
            Widget::new(w["name"].as_str().unwrap_or(""), w["role"].as_str().unwrap_or(""),
                        (r[0].as_i64().unwrap(), r[1].as_i64().unwrap(),
                         r[2].as_i64().unwrap(), r[3].as_i64().unwrap()), &window)
        }).collect()).unwrap_or_default();
        let name = path.file_stem().unwrap().to_string_lossy().to_string();
        out.insert(name.clone(), Screen { name, window, widgets });
    }
    out
}

/// Six characters that change when a fixture does. `s` in a filename.
///
/// **The prompt fingerprint covers the prompt and nothing covered the screens.** Recording
/// `{name: widget count}` is blind to a re-capture of the same window with the same number of
/// widgets, and the goals, anchors and forbidden names are all matched against text that would have
/// moved silently.
///
/// The bytes are sorted **by content**, not by path, which is what the Python does — so a fixture
/// renamed without being changed does not move it.
pub fn screen_print(dir: &Path) -> String {
    let mut blobs: Vec<Vec<u8>> = fs::read_dir(dir).map(|entries| {
        entries.flatten().map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "json"))
            .filter_map(|p| fs::read(p).ok()).collect()
    }).unwrap_or_default();
    blobs.sort();
    print6(&blobs.concat())
}

// ---- the prompt fingerprint ----------------------------------------------------------------------

/// The six probes the prompt tag is taken over, and the record they build.
///
/// Six and not three since 2026-09-13: `converse` grew a block rendered only when the history holds
/// an unanswered offer, the three probes carried none, and **the tag did not move** — two prompts,
/// one name, and a scored arm would have made every model call and then refused to write onto the
/// committed baseline's own path. Probe 5 settles the offer so the closing rule is covered and
/// probe 6 makes the turn unprompted so the gate is: *a fingerprint built only from probes that
/// make a feature appear cannot see the feature being made to appear in more places.*
pub fn prompt_record(max_listed: usize, most_at_once: usize) -> String {
    let w = vec![Widget::new("one", "button", (0, 0, 1, 1), ""),
                 Widget::new("two", "combo box", (0, 0, 1, 1), "")];
    let none: Vec<Widget> = Vec::new();
    let base: Vec<(String, String)> =
        vec![("them".into(), "a".into()), ("you".into(), "b".into())];
    let offer = "to click the button \"one\" - nothing was done; they have not accepted";
    let mut open = base.clone();
    open.push(("offered".into(), offer.into()));
    let mut settled = open.clone();
    settled.push(("did".into(), "clicked the button \"one\"".into()));
    let empty: Vec<(String, String)> = Vec::new();

    let probes: Vec<(&str, bool, &Vec<Widget>, &Vec<(String, String)>)> = vec![
        ("asked, with history, widgets listed", true, &w, &base),
        ("unprompted, no history, widgets listed", false, &w, &empty),
        ("asked, no history, nothing on screen", true, &none, &empty),
        ("asked, an offer nobody has answered, widgets listed", true, &w, &open),
        ("asked, that same offer settled, widgets listed", true, &w, &settled),
        ("unprompted, an offer nobody has answered, widgets listed", false, &w, &open),
    ];

    let mut blocks = vec![format!("MAX_LISTED = {max_listed}\nMOST_AT_ONCE = {most_at_once}\n")];
    let mut texts = Vec::new();
    for (label, asked, widgets, history) in probes {
        let trigger = Trigger { text: "probe".into(), asked };
        let text = render(widgets, &trigger, history, max_listed);
        blocks.push(format!("--- {label} ---\n{text}\n"));
        texts.push(text);
    }
    //: **Two identical renders mean a branch this claims to cover has stopped being reached**, and
    //: the tag would go on looking healthy while covering less than it says. Refused rather than
    //: hashed.
    for i in 0..texts.len() {
        for j in (i + 1)..texts.len() {
            assert_ne!(texts[i], texts[j], "probes {i} and {j} render identically");
        }
    }
    blocks.join("\n")
}

/// `r` plus six characters. The letter says the scheme: `p` hashed `converse.py`, `q` hashed three
/// probes, `r` hashes six.
pub fn prompt_tag() -> String {
    let record = prompt_record(argmind_core::converse::MAX_LISTED,
                               argmind_core::converse::MOST_AT_ONCE);
    format!("r{}", print6(record.as_bytes()))
}

// ---- what a call cost ----------------------------------------------------------------------------

/// What one call cost, off the reply itself.
///
/// **The instrument this directory was missing.** A degeneration was found by hand and by hand
/// only: the model repeated one key for 4,096 tokens and 71 s, the extractor returned nothing, and
/// the turn produced nothing at all — while in the result file it was only a slightly larger
/// `seconds`, which is the one figure that cannot tell that apart from a busy machine.
///
/// `None` in gives all-`None` out rather than zeros: a call that produced no reply is a fact, and a
/// zero would be a measurement.
pub fn spend(reply: Option<&argmind_core::llm::Reply>, seconds: Option<f64>) -> serde_json::Value {
    match reply {
        None => serde_json::json!({"seconds": null, "out_tokens": null, "reason": null,
                                   "thought_chars": null, "truncated": null}),
        Some(r) => serde_json::json!({
            "seconds": seconds.map(|s| (s * 10.0).round() / 10.0),
            "out_tokens": r.out_tokens,
            "reason": r.reason,
            "thought_chars": r.thought,
            "truncated": r.reason == argmind_core::llm::CEILING,
        }),
    }
}

// ---- where a run writes --------------------------------------------------------------------------

/// The path a run will write to, and **whether it may.**
///
/// Every condition in the name, including the axis held fixed: a sweep at one value and a sweep at
/// another are two measurements with the same shape. The recorded failure is a filename that
/// carried the protocol and not the model, then not the prompt, and a third baseline destroyed by
/// the fix for the second.
pub struct Destination {
    pub path: PathBuf,
}

impl Destination {
    pub fn new(dir: &Path, stem: &str, parts: &[String]) -> Self {
        let mut name = String::from(stem);
        for part in parts {
            name.push('-');
            name.push_str(part);
        }
        name.push_str(".json");
        Destination { path: dir.join(name) }
    }

    /// **Resolved before the model is loaded, so a refusal costs a second rather than 45 minutes.**
    /// The name is never trusted to be complete: two runs can still collide on something not in it,
    /// which is why the message says to check the recorded prints rather than assume.
    pub fn refuse_if_taken(&self) -> Result<(), String> {
        if self.path.exists() {
            return Err(format!(
                "REFUSING TO WRITE: {} already exists.\n  \
                 Every condition this harness puts in a name is identical to a run already on \
                 disk.\n  Either this is that run again, or something not in the name moved. \
                 Check the JSON's prints before assuming which.\n  \
                 Move or rename the existing file and re-run. Do not delete it.",
                self.path.file_name().unwrap().to_string_lossy()));
        }
        Ok(())
    }
}
