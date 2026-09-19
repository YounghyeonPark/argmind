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

pub mod capture;
pub mod choosing;
pub mod conversation;
pub mod drive;
pub mod repair;
pub mod replies;
pub mod saying;
pub mod reoffer;
pub mod sending;

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use argmind_core::contract::Widget;
use argmind_core::converse::render;
use argmind_core::intent::Trigger;
use argmind_core::mind::{Mind, DIALS, MOST};
use sha2::{Digest, Sha256};

/// Six characters, which is what every fingerprint in this project is.
pub fn print6(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes)).chars().take(6).collect()
}

/// A [`Mind`] off a command line: `"재치=7,유머=5"`, and `목표=...` for the sentence.
///
/// **Anything unrecognised is refused rather than ignored.** A harness given a misspelt dial that
/// quietly ran at rest would write a file claiming a condition it did not have, which is the one
/// failure mode this whole module exists to prevent.
pub fn mind_from(spec: &str) -> Mind {
    let mut mind = Mind::resting();
    for part in spec.split(',').map(str::trim).filter(|p| !p.is_empty()) {
        let (name, value) = part.split_once('=')
            .unwrap_or_else(|| panic!("{part:?} is not name=value"));
        let (name, value) = (name.trim(), value.trim());
        if name == "목표" || name == "aim" {
            mind.aim = value.to_string();
            continue;
        }
        let n: u8 = value.parse().unwrap_or_else(|_| panic!("{value:?} is not a number"));
        assert!(n <= MOST, "{n} is above {MOST}");
        assert!(DIALS.iter().any(|(k, _)| *k == name),
                "no dial called {name:?} — have {:?}", DIALS.map(|(k, _)| k));
        mind.turn(name, n);
    }
    mind
}

/// `m` plus six characters, or **nothing at all when the mind is at rest** — so a resting run
/// writes exactly the filename it wrote before any of this existed and stays comparable with it.
pub fn mind_print(mind: &Mind) -> Option<String> {
    mind.line().map(|line| format!("m{}", print6(line.as_bytes())))
}

// ---- a picture, for the harnesses that send one ------------------------------------------------
//
// **Here rather than in one bin**, because two of them now draw the same card and a second copy
// would drift: `describing` times the pipeline with it and `overlapping` keeps the vision model
// busy with it, and a card that differed between them would make the two results
// incomparable without anybody noticing.

/// A picture, drawn rather than photographed.
///
/// **Something a description can be checked against without anybody's judgement**: four coloured
/// quadrants and a dark bar, so a caption either mentions colours in the right arrangement or it
/// does not. Not a test of whether the model understands a room — a test of whether the bytes
/// arrived as a picture at all, which is the failure a wrong encoder gives and which would
/// otherwise look like a bad model.
pub fn test_card(w: usize, h: usize) -> Vec<u8> {
    let mut rgb = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 3;
            let (r, g, b) = match (x < w / 2, y < h / 2) {
                (true, true) => (220, 40, 40),
                (false, true) => (40, 160, 60),
                (true, false) => (40, 70, 200),
                (false, false) => (230, 200, 40),
            };
            //: A dark band across the middle third, so a caption has something to get wrong that is
            //: not a colour name.
            let dark = y > h * 7 / 16 && y < h * 9 / 16;
            rgb[i] = if dark { 20 } else { r };
            rgb[i + 1] = if dark { 20 } else { g };
            rgb[i + 2] = if dark { 20 } else { b };
        }
    }
    png(&rgb, w, h)
}

/// A PNG, hand-written, with **deflate's stored blocks** — no compression and no dependency.
///
/// The same argument as the rest of the wire code here: HTTP, D-Bus and the WAV header are all
/// hand-written because the licence rule is cheapest to keep when there is nothing new to check.
/// Stored blocks make the file about as large as the pixels, which for a frame this size is tens of
/// kilobytes over a loopback socket.
pub fn png(rgb: &[u8], w: usize, h: usize) -> Vec<u8> {
    //: Each row is prefixed with a filter byte of 0 — "no filter" — which is what makes the
    //: uncompressed path legal rather than merely small.
    let mut raw = Vec::with_capacity(h * (1 + w * 3));
    for y in 0..h {
        raw.push(0);
        raw.extend_from_slice(&rgb[y * w * 3..(y + 1) * w * 3]);
    }

    let mut z = vec![0x78, 0x01];
    for (i, part) in raw.chunks(65_535).enumerate() {
        let last = (i + 1) * 65_535 >= raw.len();
        z.push(if last { 1 } else { 0 });
        z.extend_from_slice(&(part.len() as u16).to_le_bytes());
        z.extend_from_slice(&(!(part.len() as u16)).to_le_bytes());
        z.extend_from_slice(part);
    }
    z.extend_from_slice(&adler32(&raw).to_be_bytes());

    let mut out = vec![0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&(w as u32).to_be_bytes());
    ihdr.extend_from_slice(&(h as u32).to_be_bytes());
    ihdr.extend_from_slice(&[8, 2, 0, 0, 0]);
    chunk(&mut out, b"IHDR", &ihdr);
    chunk(&mut out, b"IDAT", &z);
    chunk(&mut out, b"IEND", &[]);
    out
}

fn chunk(out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    out.extend_from_slice(kind);
    out.extend_from_slice(data);
    let mut crc = Vec::with_capacity(4 + data.len());
    crc.extend_from_slice(kind);
    crc.extend_from_slice(data);
    out.extend_from_slice(&crc32(&crc).to_be_bytes());
}

fn crc32(data: &[u8]) -> u32 {
    let mut c = 0xffff_ffffu32;
    for b in data {
        c ^= *b as u32;
        for _ in 0..8 {
            c = if c & 1 != 0 { 0xedb8_8320 ^ (c >> 1) } else { c >> 1 };
        }
    }
    !c
}

fn adler32(data: &[u8]) -> u32 {
    let (mut a, mut b) = (1u32, 0u32);
    for x in data {
        a = (a + *x as u32) % 65_521;
        b = (b + a) % 65_521;
    }
    (b << 16) | a
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
        //: **At rest, and that is the point of the tag.** `r322a4c` surviving the arrival
        //: of [`Mind`] is the proof that a dial nobody turned changes nothing.
        let text = render(widgets, &trigger, history, max_listed, &Mind::resting());
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

/// The thinking tick's question, hashed. **`t` in a filename.**
///
/// Added 2026-09-17 because the harness refused a run and was right to. `judging` had its own source
/// in the name and the prompt it sends in `argmind-core`, so three rewrites of that prompt would
/// have written to the same path — or, as it happened, refused the third and said so. **A condition
/// not in the name is a result nobody can compare with anything**, which is the failure this
/// directory keeps a list of, met again from a new direction.
///
/// Two probes rather than one: a tick with nothing yet and a tick carrying a note and an event, so
/// a change to either branch of the question moves the tag.
pub fn thinking_record() -> String {
    use argmind_core::thinking::{ask_for_a_note, Note, MOST_CHARS};
    let fresh = Note::blank();
    let mut held = Note::blank();
    //: A judgement as the note now carries one — a resolved widget, never a sentence.
    held.revise("그가 문서를 보고 있다",
                argmind_core::thinking::Judged::This(
                    argmind_core::thinking::Worth { name: "Save".into(),
                                                    role: "button".into() }));
    held.record("clicked the button \"Save\"");
    let screen = "0: button \"one\"\n1: combo box \"two\"";
    [ask_for_a_note(&fresh, screen, "", &[], MOST_CHARS),
     ask_for_a_note(&held, screen, "나타남: Eraser",
                    &[("them".into(), "a".into())], MOST_CHARS)].join("\n--\n")
}

/// `t` plus six characters.
pub fn thinking_tag() -> String {
    format!("t{}", print6(thinking_record().as_bytes()))
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

/// `json.dumps(obj, ensure_ascii=False, sort_keys=True)` — the serialisation every `f` in this
/// directory is a hash of.
///
/// Written out rather than reached for, because `serde_json::to_string` uses `,`/`:` where Python
/// uses `, `/`: `. The difference is invisible until two fingerprints disagree and somebody assumes
/// the cases moved.
pub fn python_json(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::Null => "null".into(),
        serde_json::Value::Bool(b) => b.to_string(),
        serde_json::Value::Number(n) => n.to_string(),
        serde_json::Value::String(s) => serde_json::to_string(s).unwrap(),
        serde_json::Value::Array(a) =>
            format!("[{}]", a.iter().map(python_json).collect::<Vec<_>>().join(", ")),
        //: Keys come out sorted: `serde_json::Map` is a `BTreeMap` unless `preserve_order` is on,
        //: which is what `sort_keys=True` asks for.
        serde_json::Value::Object(o) => format!("{{{}}}", o.iter()
            .map(|(k, v)| format!("{}: {}", serde_json::to_string(k).unwrap(), python_json(v)))
            .collect::<Vec<_>>().join(", ")),
    }
}

/// Six characters over a case list. `f` in a filename.
///
/// **A case list edited between two runs is two experiments sharing one name**, which is how three
/// baselines were destroyed here, and how the only baseline one harness ever had came to describe a
/// prompt that had been reverted.
pub fn data_print(v: &serde_json::Value) -> String {
    print6(python_json(v).as_bytes())
}

/// The character tables and word lists a scorer classifies with.
///
/// **Loaded, not transcribed.** `PUNCT` and the Korean number words are exactly the kind of table a
/// port gets 95% right, where the missing 5% is a verdict that quietly changes — so both
/// implementations read one copy, dumped from the Python by `native/expectations/saying.py`.
pub struct Tables {
    pub token_chars: String,
    pub punct: String,
    pub acks: Vec<String>,
    pub number_words: Vec<String>,
}

impl Tables {
    pub fn from_json(d: &serde_json::Value) -> Self {
        let list = |k: &str| d[k].as_array().map(|a| a.iter()
            .filter_map(|x| x.as_str().map(str::to_string)).collect()).unwrap_or_default();
        Tables {
            token_chars: d["token_chars"].as_str().unwrap_or("").to_string(),
            punct: d["punct"].as_str().unwrap_or("").to_string(),
            acks: list("acks"),
            number_words: list("number_words"),
        }
    }
}
