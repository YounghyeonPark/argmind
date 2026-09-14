//! Stage 3a of the port: the half of `loop.py` that has no seam.
//!
//! `observe`, `poll`, `perform` and `think` call `Eyes`, `Hands` and a model. Those are the seam,
//! measured at 0.144 ms a crossing against 0.032 ms of work for the whole turn, and they are an
//! argument to have before writing them. What is here is everything else — and it is the part with
//! the worst protection in the file: **three of `remember_doing()`'s four channels have never been
//! fed by any harness**, and `recent()` is where a line-slicing window used to cut a turn in half
//! in 10.7% of prompts.
//!
//! Two things are being checked, and only the first is about Rust.
//!
//! 1. **The wording is identical.** The past tense belongs to `did` and to nothing else; every
//!    other channel says in words that nothing happened, so a reader that ignores the label cannot
//!    conclude a click landed. A port that loses a space or an em dash writes a record the model
//!    reads differently, and this is the only thing that would catch it.
//! 2. **`find()` splits cleanly into I/O and arithmetic.** Its only call out is
//!    `eyes.foreground()`; given the fresh list and what was last seen, resolution is name, then
//!    role, then nearest to where a same-named thing was. That decomposition is most of the porting
//!    decision for this file, so it is written down rather than assumed.
//!
//! Expectations come from `tests/test_loop_golden.py`'s own scene data, so this exercises the
//! branches that transcript was written to pin rather than the ones I would have thought of.

use std::env;
use std::fs;

use serde_json::Value;

// ---- copied verbatim from argmind/core/loop.py ---------------------------------------------------

const NOTHING_DONE: &str = "nothing was done";
const PAST: usize = 0;
const PLAIN: usize = 1;
const ING: usize = 2;

/// `VERBS`: the verb of each effect kind in three forms. A kind that is not listed falls back to
/// `key`'s row — which is how `'to press  — …'` loses its object, a defect the golden transcript
/// records and this port reproduces rather than quietly fixing.
fn verb(kind: &str, form: usize) -> &'static str {
    let row: [&str; 3] = match kind {
        "click" => ["clicked", "click", "clicking"],
        "type" => ["typed", "type", "typing"],
        _ => ["pressed", "press", "pressing"],
    };
    row[form]
}

struct Effect {
    kind: String,
    target: String,
    role: String,
    text: String,
}

#[derive(Clone)]
struct Widget {
    name: String,
    role: String,
    rect: (i64, i64, i64, i64),
}

impl Widget {
    /// The one place a name becomes a coordinate. Integer division, as in the Python.
    fn centre(&self) -> (i64, i64) {
        let (x, y, w, h) = self.rect;
        (x + w.div_euclid(2), y + h.div_euclid(2))
    }
}

/// One short phrase for an effect, in the words the person would recognise. `form` picks the verb
/// and **that is the whole state marker**: `PAST` says it happened and the other two say it did not.
fn describe(effect: &Effect, widget: Option<&Widget>, form: usize) -> String {
    let v = verb(&effect.kind, form);
    if effect.kind == "click" {
        let role = match widget {
            Some(w) => w.role.clone(),
            None => {
                if effect.role.is_empty() { "thing".to_string() } else { effect.role.clone() }
            }
        };
        return format!("{v} the {role} \"{}\"", effect.target);
    }
    if effect.kind == "type" {
        return format!("{v} \"{}\"", effect.text);
    }
    format!("{v} {}", effect.text)
}

/// The four lines `remember_doing()` writes, in the order it writes them. **`offered` first within
/// one call**, which the golden transcript records as a hazard: a caller batching an offer with a
/// settlement has the offer closed by `still_open()` before it is ever rendered.
fn remember_doing(
    history: &mut Vec<(String, String)>,
    did: &[(Effect, Option<Widget>)],
    refused: &[Effect],
    offered: &[Effect],
    turned_down: &[Effect],
) {
    for e in offered {
        history.push(("offered".into(),
            format!("to {} — {NOTHING_DONE}; they have not accepted", describe(e, None, PLAIN))));
    }
    for (e, target) in did {
        history.push(("did".into(), describe(e, target.as_ref(), PAST)));
    }
    for e in refused {
        history.push(("could not".into(),
            format!("{} — it had gone by then; {NOTHING_DONE}", describe(e, None, PLAIN))));
    }
    for e in turned_down {
        history.push(("they said no to".into(),
            format!("{} — {NOTHING_DONE}", describe(e, None, ING))));
    }
}

/// The last `turns` turns, cut where a turn begins and never inside one. `None` is the whole
/// conversation; **0 is none of it, and that is a bug fixed rather than a choice** — this was
/// `history[-remember:]`, and `history[-0:]` is the entire list, so the setting a person reaches
/// for to turn memory off turned it all the way up.
fn recent(history: &[(String, String)], turns: Option<i64>) -> Vec<(String, String)> {
    let turns = match turns {
        None => return history.to_vec(),
        Some(t) => t,
    };
    if turns <= 0 {
        return Vec::new();
    }
    let opens: Vec<usize> = history
        .iter()
        .enumerate()
        .filter(|(_, (who, _))| who == "them" || who == "noticed")
        .map(|(i, _)| i)
        .collect();
    if opens.len() as i64 <= turns {
        return history.to_vec();
    }
    history[opens[opens.len() - turns as usize]..].to_vec()
}

/// `Session.find`'s body with `eyes.foreground()` hoisted out — the whole of its I/O. Matched on
/// the name because **the rectangle is exactly what moves**; the role is half the key because a
/// `text` labelled `Font :` in front of a `combo box` of the same name took a click and did
/// nothing. Among duplicates the one nearest to where a same-named thing was last seen wins.
fn find<'a>(fresh: &'a [Widget], seen: &[Widget], name: &str, role: &str) -> Option<&'a Widget> {
    let same: Vec<&Widget> = fresh
        .iter()
        .filter(|w| w.name == name && (role.is_empty() || w.role == role))
        .collect();
    if same.is_empty() {
        return None;
    }
    if same.len() == 1 {
        return Some(same[0]);
    }
    let was = seen
        .iter()
        .find(|w| w.name == name && (role.is_empty() || w.role == role))
        .map(|w| w.centre())
        .unwrap_or_else(|| same[0].centre());
    //: `min` over a stable order, as Python's `min` keeps the first of equal keys.
    let mut best = same[0];
    let mut best_d = i64::MAX;
    for w in &same {
        let c = w.centre();
        let d = (c.0 - was.0).pow(2) + (c.1 - was.1).pow(2);
        if d < best_d {
            best_d = d;
            best = w;
        }
    }
    Some(best)
}

// ---- checking ------------------------------------------------------------------------------------

fn effect_of(v: &Value) -> Effect {
    Effect {
        kind: v["kind"].as_str().unwrap_or("").to_string(),
        target: v["target"].as_str().unwrap_or("").to_string(),
        role: v["role"].as_str().unwrap_or("").to_string(),
        text: v["text"].as_str().unwrap_or("").to_string(),
    }
}

fn widget_of(v: &Value) -> Widget {
    let r = v["rect"].as_array().expect("rect");
    Widget {
        name: v["name"].as_str().unwrap_or("").to_string(),
        role: v["role"].as_str().unwrap_or("").to_string(),
        rect: (r[0].as_i64().unwrap(), r[1].as_i64().unwrap(),
               r[2].as_i64().unwrap(), r[3].as_i64().unwrap()),
    }
}

fn main() {
    let path = env::args().nth(1).expect("usage: loop-spike <expectations.json>");
    let d: Value = serde_json::from_str(&fs::read_to_string(&path).expect("read")).expect("parse");
    let mut bad = 0;

    for case in d["describe"].as_array().unwrap() {
        let e = effect_of(&case["effect"]);
        let widget = if case["widget"].is_null() { None } else { Some(widget_of(&case["widget"])) };
        let got = describe(&e, widget.as_ref(), case["form"].as_u64().unwrap() as usize);
        let want = case["want"].as_str().unwrap();
        if got != want {
            bad += 1;
            println!("  DESCRIBE {}\n    python {want:?}\n    rust   {got:?}",
                     case["label"].as_str().unwrap());
        }
    }

    for case in d["recent"].as_array().unwrap() {
        let history: Vec<(String, String)> = case["history"].as_array().unwrap().iter()
            .map(|p| (p[0].as_str().unwrap().to_string(), p[1].as_str().unwrap().to_string()))
            .collect();
        let turns = if case["turns"].is_null() { None } else { case["turns"].as_i64() };
        let got = recent(&history, turns);
        let want: Vec<(String, String)> = case["want"].as_array().unwrap().iter()
            .map(|p| (p[0].as_str().unwrap().to_string(), p[1].as_str().unwrap().to_string()))
            .collect();
        if got != want {
            bad += 1;
            println!("  RECENT turns={turns:?}  python {} lines, rust {}", want.len(), got.len());
        }
    }

    for case in d["find"].as_array().unwrap() {
        let fresh: Vec<Widget> = case["fresh"].as_array().unwrap().iter().map(widget_of).collect();
        let seen: Vec<Widget> = case["seen"].as_array().unwrap().iter().map(widget_of).collect();
        let got = find(&fresh, &seen, case["name"].as_str().unwrap(),
                       case["role"].as_str().unwrap());
        let want = &case["want"];
        let same = match (got, want.is_null()) {
            (None, true) => true,
            (Some(w), false) => {
                let x = widget_of(want);
                w.name == x.name && w.role == x.role && w.rect == x.rect
            }
            _ => false,
        };
        if !same {
            bad += 1;
            println!("  FIND {}\n    python {want}\n    rust   {:?}",
                     case["label"].as_str().unwrap(),
                     got.map(|w| (w.name.clone(), w.role.clone(), w.rect)));
        }
    }

    //: **The whole line each channel writes, not just the phrase inside it.** This section was
    //: added after the port passed without it: with only `describe()` pinned, rewording the offer
    //: line's "they have not accepted" was not caught, and that text is the least protected in the
    //: file — three of the four channels have never been fed by any harness.
    let empty: Vec<Effect> = Vec::new();
    for case in d["channels"].as_array().unwrap() {
        let e = effect_of(&case["effect"]);
        let target = if case["target"].is_null() { None } else { Some(widget_of(&case["target"])) };
        let mut history: Vec<(String, String)> = Vec::new();
        match case["channel"].as_str().unwrap() {
            "did" => remember_doing(&mut history, &[(e, target)], &empty, &empty, &empty),
            "refused" => remember_doing(&mut history, &[], &[e], &empty, &empty),
            "offered" => remember_doing(&mut history, &[], &empty, &[e], &empty),
            _ => remember_doing(&mut history, &[], &empty, &empty, &[e]),
        }
        let want: Vec<(String, String)> = case["want"].as_array().unwrap().iter()
            .map(|p| (p[0].as_str().unwrap().to_string(), p[1].as_str().unwrap().to_string()))
            .collect();
        if history != want {
            bad += 1;
            println!("  CHANNEL {}\n    python {want:?}\n    rust   {history:?}",
                     case["label"].as_str().unwrap());
        }
    }

    //: All four in one call. `offered` is written first, which the golden records as a hazard: a
    //: caller batching an offer with a settlement has the offer closed before it is ever rendered.
    let mut history: Vec<(String, String)> = Vec::new();
    remember_doing(&mut history,
        &[(Effect { kind: "click".into(), target: "Cut".into(), role: "button".into(),
                    text: String::new() },
           Some(Widget { name: "Cut".into(), role: "button".into(), rect: (10, 20, 40, 30) }))],
        &[Effect { kind: "type".into(), target: String::new(), role: String::new(),
                   text: "bye".into() }],
        &[Effect { kind: "click".into(), target: "Copy".into(), role: "button".into(),
                   text: String::new() }],
        &[Effect { kind: "key".into(), target: String::new(), role: String::new(),
                   text: "tab".into() }]);
    let want: Vec<(String, String)> = d["order"]["want"].as_array().unwrap().iter()
        .map(|p| (p[0].as_str().unwrap().to_string(), p[1].as_str().unwrap().to_string()))
        .collect();
    if history != want {
        bad += 1;
        println!("  ORDER\n    python {want:?}\n    rust   {history:?}");
    }

    let n = d["describe"].as_array().unwrap().len()
        + d["recent"].as_array().unwrap().len()
        + d["find"].as_array().unwrap().len()
        + d["channels"].as_array().unwrap().len()
        + 1;
    println!("\n  {n} cases from the golden transcript's own scene data, {bad} disagreements");
    if bad == 0 {
        println!("\n  IDENTICAL — the wording, the window and the resolution all match.");
    } else {
        std::process::exit(1);
    }
}
