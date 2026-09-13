//! Does a Rust `converse()` render **byte-identically** to the Python one?
//!
//! This project already owns the instrument that answers it. `measure/choosing.py` hashes the
//! prompt as `converse()` renders it over six probes, plus `MAX_LISTED` and `MOST_AT_ONCE`, and
//! calls the first six hex characters `r`. The shipped tag is **r322a4c**. A Rust renderer that
//! produces the same tag has provably not changed a single character of what the model is sent —
//! and one that differs says so immediately, without a model, a fixture or an opinion.
//!
//! So this is a port of exactly one thing: the prompt assembly. No HTTP, no session, no effects.
//! If the tag matches, the rest of the port is ordinary work. If it does not, the port is a
//! rewrite of the prompt, which is the one thing this project has measured as never free.

use sha2::{Digest, Sha256};

// ---- the constants, copied verbatim from argmind/core/converse.py ---------------------------------

const MAX_LISTED: usize = 200;
const MOST_AT_ONCE: usize = 3;

const SYSTEM: &str = r#"You are a helper the person talks to while they work at their PC.

You are shown the things currently on their screen, each with a number. Reply as a person would —
briefly, in the language they used.

If they asked you to do something and the listed things can do it, name the numbers. If they were
just talking, or the thing they want is not listed, say so and do nothing: a wrong click is worse
than an honest "I can't see that here"."#;

const UNPROMPTED: &str = "Nobody asked you anything — you noticed this yourself. Mentioning it is usually\nenough. Only propose doing something if it is clearly wanted, and expect to be asked first.";

const STILL_OPEN: &str = "Still open, from earlier:";
const STILL_OPEN_TAIL: &str =
    "What they say next may be a yes to that, a no, or nothing to do with it.";

const OFFER_OPEN: &str = "offered";
const OFFER_SETTLED: [&str; 3] = ["did", "could not", "they said no to"];

// ---- the pieces the prompt is built from ---------------------------------------------------------

struct Widget {
    name: &'static str,
    role: &'static str,
}

struct Trigger {
    text: &'static str,
    asked: bool,
}

/// The list as the model sees it. Names pass through untouched — not lowercased, not translated.
fn describe(widgets: &[Widget], limit: usize) -> String {
    widgets
        .iter()
        .take(limit)
        .enumerate()
        .map(|(i, w)| format!("{}: {} \"{}\"", i, w.role, w.name))
        .collect::<Vec<_>>()
        .join("\n")
}

/// The offers nobody has answered yet, oldest first, in the words the history holds them in.
/// It closes on **any** settling label rather than matching the effect, deliberately: matching
/// would mean parsing `loop.py`'s prose back into an effect.
fn still_open(history: &[(&str, &str)]) -> Vec<String> {
    let mut open: Vec<String> = Vec::new();
    for (who, text) in history {
        if *who == OFFER_OPEN {
            open.push((*text).to_string());
        } else if OFFER_SETTLED.contains(who) {
            open.clear();
        }
    }
    open
}

/// The prompt, and nothing else. `converse()` in Python does this and then calls the model; the
/// call and everything after it is not what this spike is testing.
fn render(widgets: &[Widget], trigger: &Trigger, history: &[(&str, &str)], limit: usize) -> String {
    let listed: Vec<&Widget> = widgets.iter().take(limit).collect();
    let mut lines: Vec<String> = vec![SYSTEM.to_string()];
    if !trigger.asked {
        lines.push(UNPROMPTED.to_string());
    }
    lines.push(String::new());
    if !history.is_empty() {
        lines.push("Recently:".to_string());
        for (who, text) in history {
            lines.push(format!("  {}: {}", who, text));
        }
        lines.push(String::new());
    }
    lines.push("On screen now:".to_string());
    lines.push(if listed.is_empty() {
        "  (nothing named — this surface exposes no widgets)".to_string()
    } else {
        let owned: Vec<Widget> = listed
            .iter()
            .map(|w| Widget { name: w.name, role: w.role })
            .collect();
        describe(&owned, limit)
    });

    let pending = if trigger.asked { still_open(history) } else { Vec::new() };
    if !pending.is_empty() {
        lines.push(String::new());
        lines.push(STILL_OPEN.to_string());
        for text in &pending {
            lines.push(format!("  {}", text));
        }
        lines.push(STILL_OPEN_TAIL.to_string());
    }

    lines.push(String::new());
    lines.push(if trigger.asked {
        format!("They said: {}", trigger.text)
    } else {
        format!("You noticed: {}", trigger.text)
    });
    lines.push(String::new());
    lines.push(
        r#"Reply with only JSON: {"say": "<what you say back>", "do": [ ... ]} where each entry of"#
            .to_string(),
    );
    lines.push(
        r#""do" is {"click": <number>}, {"type": "<text to type>"} or {"key": "<enter|tab|escape>"}."#
            .to_string(),
    );
    lines.push(r#"Leave "do" empty if nothing should happen."#.to_string());
    lines.join("\n")
}

// ---- the six probes, copied verbatim from measure/choosing.py -------------------------------------

fn probe_widgets() -> Vec<Widget> {
    vec![
        Widget { name: "one", role: "button" },
        Widget { name: "two", role: "combo box" },
    ]
}

const PROBE_TEXT: &str = "probe";
const PROBE_OFFER: &str =
    "to click the button \"one\" - nothing was done; they have not accepted";
const PROBE_SETTLED: &str = "clicked the button \"one\"";

fn main() {
    let base: Vec<(&str, &str)> = vec![("them", "a"), ("you", "b")];
    let mut open = base.clone();
    open.push((OFFER_OPEN, PROBE_OFFER));
    let mut settled = open.clone();
    settled.push(("did", PROBE_SETTLED));

    let none: Vec<(&str, &str)> = Vec::new();
    let empty: Vec<Widget> = Vec::new();

    let w = probe_widgets();
    let rows: Vec<(&str, Trigger, &[Widget], &[(&str, &str)])> = vec![
        ("asked, with history, widgets listed",
         Trigger { text: PROBE_TEXT, asked: true }, &w, &base),
        ("unprompted, no history, widgets listed",
         Trigger { text: PROBE_TEXT, asked: false }, &w, &none),
        ("asked, no history, nothing on screen",
         Trigger { text: PROBE_TEXT, asked: true }, &empty, &none),
        ("asked, an offer nobody has answered, widgets listed",
         Trigger { text: PROBE_TEXT, asked: true }, &w, &open),
        ("asked, that same offer settled, widgets listed",
         Trigger { text: PROBE_TEXT, asked: true }, &w, &settled),
        ("unprompted, an offer nobody has answered, widgets listed",
         Trigger { text: PROBE_TEXT, asked: false }, &w, &open),
    ];

    let mut blocks: Vec<String> =
        vec![format!("MAX_LISTED = {}\nMOST_AT_ONCE = {}\n", MAX_LISTED, MOST_AT_ONCE)];
    let mut texts: Vec<String> = Vec::new();
    for (label, trigger, widgets, history) in &rows {
        let text = render(widgets, trigger, history, MAX_LISTED);
        blocks.push(format!("--- {} ---\n{}\n", label, text));
        texts.push(text);
    }

    //: The same refusal `prompt_record()` makes: two identical renders mean a branch this print
    //: claims to cover has stopped being reached, and the fingerprint would look healthy anyway.
    for i in 0..texts.len() {
        for j in (i + 1)..texts.len() {
            assert_ne!(texts[i], texts[j], "probes {} and {} render identically", i, j);
        }
    }

    let record = blocks.join("\n");
    let digest = Sha256::digest(record.as_bytes());
    let tag: String = format!("{:x}", digest).chars().take(6).collect();

    println!("  record {} bytes, {} probes", record.len(), rows.len());
    println!("  rust   r{}", tag);
    println!("  python r322a4c   (the shipped tag, from measure/choosing.py)");
    if tag == "322a4c" {
        println!("\n  IDENTICAL — the Rust renderer sends the same bytes, character for character.");
    } else {
        println!("\n  DIFFERENT — dumping the record so the diff can be found.");
        std::fs::write("rust-record.txt", &record).expect("write");
        println!("  written to rust-record.txt");
    }
}
