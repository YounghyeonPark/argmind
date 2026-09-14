//! How long does it hold the thread — the distance sweep, against a live model.
//!
//! ```text
//!   conversation-run --quick                one distance, one repeat, ~2 min
//!   conversation-run --axis d               distance at the shipped remember_turns
//!   conversation-run --axis r               the window at one fixed distance
//! ```
//!
//! **The harness three others defer to.** `choose`, `talk` and `saying` all call the model with a
//! screen and one line, so *"no slide on those three"* is vacuous for anything touching the
//! conversation; this is the only one that sends a history at all.
//!
//! Nothing is clicked: `Fixed` eyes and `Nothing` hands. A harness that clicked would measure the
//! fixture rather than the model, and the fixture cannot change.
//!
//! **Every condition is in the filename, including the axis held fixed.** A window sweep at
//! distance 5 and one at distance 8 are two different measurements with the same shape. The
//! recorded failure is a filename that carried the protocol and not the model, then not the prompt,
//! and a third baseline destroyed by the fix for the second — so the path is resolved and refused
//! **before the model is loaded**, which costs a second instead of forty-five minutes.

use std::path::PathBuf;

use argmind_core::contract::Speaks;
use argmind_core::llm::Reply;
use argmind_env_ollama::Local;
use argmind_measure::conversation::{
    cases, cases_print, cell, cost_table, floor, paired_table, show, summarise, turn_cost_table,
    Case, Records, Row, CONSTRAINT, REFERENCE, VOID,
};
use argmind_measure::{prompt_tag, screen_print, screens, source_print, Destination};
use serde_json::{json, Value};

/// The model, with the last prompt kept.
///
/// **The prompt actually sent is the evidence**, and nothing else in the loop stores it. Rebuilding
/// the window instead would be a second copy of the mechanism under test.
struct Recorder {
    inner: Local,
    last: String,
    reply: Option<Reply>,
    calls: usize,
    /// **Counted, not silent.** The server has answered HTTP 500 once while reloading weights, and
    /// losing forty minutes of sweep to it would be worse than retrying — but a retry nobody can
    /// see is a run whose call count no longer matches its turn count.
    retried: usize,
}

impl Speaks for Recorder {
    fn ask(&mut self, prompt: &str) -> String {
        self.last = prompt.to_string();
        self.calls += 1;
        for attempt in 0..4 {
            match self.inner.ask(prompt) {
                Ok(r) => {
                    let text = r.text.clone();
                    self.reply = Some(r);
                    return text;
                }
                Err(e) => {
                    if attempt == 3 {
                        panic!("the model failed four times running: {e}");
                    }
                    self.retried += 1;
                    println!("    [retry {}/3 after {e}]", attempt + 1);
                    std::thread::sleep(std::time::Duration::from_secs_f64(2.0 * (attempt + 1) as f64));
                }
            }
        }
        unreachable!()
    }
}

impl Records for Recorder {
    fn last_prompt(&self) -> &str {
        &self.last
    }
    fn last_reply(&self) -> Option<&Reply> {
        self.reply.as_ref()
    }
}

fn flag(name: &str) -> bool {
    std::env::args().any(|a| a == name)
}

fn opt(name: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
}

fn list(name: &str, fallback: &str) -> Vec<i64> {
    opt(name).unwrap_or_else(|| fallback.into()).split(',')
        .map(|x| x.trim().parse().expect("a comma-separated list of numbers")).collect()
}

fn main() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap()
        .to_path_buf();
    let dir = root.join("measure");
    let quick = flag("--quick");
    let mut repeat: usize = opt("--repeat").and_then(|a| a.parse().ok()).unwrap_or(3);
    let axis = opt("--axis").unwrap_or_else(|| "d".into());
    let mut distances = list("--distances", "1,3,5,8,12");
    let mut remembers = list("--remembers", "4,8,12,24");
    let at_distance: i64 = opt("--at-distance").and_then(|a| a.parse().ok()).unwrap_or(8);
    let shipped: i64 = opt("--shipped").and_then(|a| a.parse().ok()).unwrap_or(12);
    let turned_down = flag("--turned-down");
    let only = opt("--only");
    let english_rule = flag("--english-rule");
    if quick {
        distances.truncate(1);
        remembers.truncate(1);
        repeat = 1;
    }

    let mut cases: Vec<Case> = cases();
    if let Some(kind) = &only {
        cases.retain(|c| &c.kind == kind);
        assert!(!cases.is_empty(), "--only {kind} matches no case");
    }
    if english_rule {
        //: One axis moved, **in the data, before anything runs** — and the swap is printed so a
        //: reader of the log knows which wording produced the number.
        for c in cases.iter_mut().filter(|c| c.kind == CONSTRAINT) {
            let english = c.english.clone().unwrap();
            println!("    rule reworded: {}  ->  {english}", c.anchor);
            c.anchor = english;
        }
    }

    let loaded = screens(&dir.join("screens"));
    for case in &cases {
        assert!(loaded.contains_key(&case.screen), "no fixture {:?}", case.screen);
    }
    //: **The filler is a condition of the result**, so it is checked rather than trusted: a line
    //: naming a target would put the anchor back into the window and turn a distance-12 probe into
    //: a distance-1 one. A harness that cannot fail.
    let filler: Vec<String> = argmind_measure::conversation::FILLER.iter()
        .map(|s| s.to_string()).collect();
    for line in &filler {
        for case in &cases {
            let mut targets: Vec<&String> = case.forbid.iter().collect();
            if let Some(w) = &case.want {
                targets.push(w);
            }
            for t in targets {
                assert!(!line.to_lowercase().contains(&t.to_lowercase()),
                        "filler {line:?} names {t:?} — that is the confound this measures");
            }
        }
    }

    // ---- where this writes, resolved before the model is loaded ------------------------------
    let s = screen_print(&dir.join("screens"));
    let f = cases_print(&cases, &filler);
    let h = source_print(include_str!("conversation-run.rs"));
    //: **`l` is the file that decides how much of the conversation reaches the prompt.** In this
    //: tree that is `session.rs`, not a `loop.py`, and the stem says `rust` so the two cannot be
    //: read as one quantity — the hazard this project has already hit with a letter meaning two
    //: things in two files.
    let l = source_print(include_str!("../../../argmind-core/src/session.rs"));
    let mut parts = vec!["rust".to_string(), prompt_tag(), format!("s{s}"), format!("f{f}"),
                         format!("h{h}"), format!("l{l}")];
    let join = |v: &[i64]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(".");
    if axis == "d" || axis == "both" {
        parts.push(format!("d{}@r{shipped}", join(&distances)));
    }
    if axis == "r" || axis == "both" {
        parts.push(format!("t{}@d{at_distance}", join(&remembers)));
    }
    parts.push(format!("x{repeat}"));
    if turned_down {
        parts.push("turneddown".into());
    }
    if let Some(kind) = &only {
        parts.push(kind.clone());
    }
    if english_rule {
        parts.push("englishrule".into());
    }
    let out = Destination::new(&dir, "conversation", &parts);
    if let Err(why) = out.refuse_if_taken() {
        println!("\n  ** {why}");
        std::process::exit(3);
    }

    let mut model = Recorder { inner: Local { think: Some(false), ..Local::new("qwen3:8b") },
                               last: String::new(), reply: None, calls: 0, retried: 0 };
    for _ in 0..2 {
        let _ = model.ask("Reply with the single word: ready");
    }
    model.calls = 0;

    println!("  qwen3:8b | thinking off | prompt {} | {repeat} repeats, worst counted",
             prompt_tag());
    println!("  harness h{h} | session.rs l{l} | screens s{s} | cases f{f}");
    println!("  {} cases | filler is ordinary conversation, {} lines available{}\n",
             cases.len(), filler.len(), if turned_down { " | turned_down recorded" } else { "" });

    let started = std::time::Instant::now();
    let mut rows: Vec<Value> = Vec::new();

    // ---- the floor ---------------------------------------------------------------------------
    println!("  THE FLOOR — the same probe with no conversation at all, worst counted\n");
    let mut floors: Vec<Value> = Vec::new();
    for case in &cases {
        let (row, back) = floor(model, &loaded[&case.screen].widgets, case, repeat)
            .unwrap_or_else(|e| panic!("{e}"));
        model = back;
        let note = match case.kind.as_str() {
            REFERENCE => "  <- what forgetting looks like",
            CONSTRAINT => "  <- the control's asymptote",
            _ => "",
        };
        let chose = row["clicked"].as_array().unwrap().iter()
            .map(|c| {
                let names: Vec<&str> = c.as_array().unwrap().iter()
                    .map(|x| x.as_str().unwrap()).collect();
                if names.is_empty() { "-".to_string() } else { names.join("+") }
            }).collect::<Vec<_>>().join(" / ");
        println!("    {:<10} {:<11} chose {:<52}{note}",
                 row["counted"].as_str().unwrap().to_uppercase(), case.id,
                 chose.chars().take(52).collect::<String>());
        floors.push(row);
    }

    // ---- the sweeps --------------------------------------------------------------------------
    let sweep = |model: Recorder, cells: Vec<(i64, i64)>, name: &str|
     -> (Vec<Value>, Recorder) {
        let mut model = model;
        let mut got = Vec::new();
        for case in &cases {
            //: **A reference case has no control arm**: there is nothing to remove that leaves the
            //: question intact. Its own null result is a wrong or silent answer, which it can
            //: produce.
            let arms: &[&str] = if case.kind == REFERENCE { &["with"] } else { &["with", "without"] };
            for arm in arms {
                for (distance, remember) in &cells {
                    let (mut row, back) = cell(model, &loaded[&case.screen].widgets, case,
                                               *distance as usize, *remember, arm, repeat,
                                               turned_down).unwrap_or_else(|e| panic!("{e}"));
                    model = back;
                    row["axis"] = json!(name);
                    print!("{}", show(&Row::from_json(&row)));
                    got.push(row);
                }
            }
        }
        (got, model)
    };

    if axis == "d" || axis == "both" {
        println!("\n  DISTANCE — anchor at turn 1, probe N turns later, at the shipped \
                  remember_turns={shipped}\n");
        let cells: Vec<(i64, i64)> = distances.iter().map(|d| (*d, shipped)).collect();
        let (got, back) = sweep(model, cells, "distance");
        model = back;
        let parsed: Vec<Row> = got.iter().map(Row::from_json).collect();
        print!("{}", summarise(&parsed, "distance"));
        print!("{}", paired_table(&parsed, "distance"));
        print!("{}", turn_cost_table(&parsed));
        rows.extend(got);
    }
    if axis == "r" || axis == "both" {
        println!("\n  REMEMBER — the same tests at a fixed distance of {at_distance} turns\n");
        let cells: Vec<(i64, i64)> = remembers.iter().map(|r| (at_distance, *r)).collect();
        let (got, back) = sweep(model, cells, "remember");
        model = back;
        let parsed: Vec<Row> = got.iter().map(Row::from_json).collect();
        print!("{}", summarise(&parsed, "remember"));
        print!("{}", paired_table(&parsed, "remember"));
        print!("{}", cost_table(&parsed));
        rows.extend(got);
    }
    let took = started.elapsed().as_secs_f64();

    let parsed: Vec<Row> = rows.iter().map(Row::from_json).collect();
    let disagreed = parsed.iter().filter(|r| !r.agreed).count();
    let void = parsed.iter().filter(|r| r.counted == VOID).count();
    println!("\n  {} cells, {} model calls, {:.1} min", rows.len(), model.calls, took / 60.0);
    println!("    model calls retried after a server error  {}", model.retried);
    println!("    disagreed across repeats   {disagreed:>3} of {}{}", rows.len(),
             if disagreed > 0 { " — a single run is not this number" } else { "" });
    println!("    void (the anchor never landed) {void:>3} of {}", rows.len());
    //: The trap the filler is written to avoid, **counted rather than assumed away**: the target
    //: back in the window although the anchor turn itself was truncated out of it.
    let resurfaced: usize = parsed.iter().map(|r| r.target_in_window.iter()
        .zip(&r.anchor_in_window).filter(|(t, a)| **t && !**a).count()).sum();
    let draws: usize = parsed.iter().map(|r| r.n).sum();
    println!("    target in the window without its anchor turn  {resurfaced} of {draws} runs");
    //: Free, and the other half of `talk`: how often ordinary chat in the middle of a conversation
    //: produced an action nobody asked for.
    let fillers: Vec<u64> = rows.iter()
        .flat_map(|r| r["filler_effects"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap()))
        .collect();
    if !fillers.is_empty() {
        println!("    effects proposed during the filler turns  {} over {} conversations ({} of \
                  them had any)", fillers.iter().sum::<u64>(), fillers.len(),
                 fillers.iter().filter(|f| **f > 0).count());
    }

    //: Checked again at the write, because two of these can be in flight at once.
    if let Err(why) = out.refuse_if_taken() {
        println!("\n  ** {why}");
        std::process::exit(3);
    }
    let written = json!({
        "model": "qwen3:8b", "think": "off", "prompt": prompt_tag()[1..].to_string(),
        "prompt_scheme": "r = a hash of the rendered prompt over six probes",
        "screens_print": s, "data_print": f, "harness_print": h,
        //: **The same file under both keys, and the second says which language wrote it.**
        //: `sending::deferral` reads `loop_print` out of these files to decide whether the history
        //: axis has been run at a given tree; a Rust run answers that question about the Rust
        //: tree, and a reader must not take the hash for `loop.py`'s.
        "loop_print": l, "loop_print_lang": "rust: native/argmind-core/src/session.rs",
        "remember_unit": "turns",
        "repeat": repeat, "axis": axis, "distances": distances, "remembers": remembers,
        "shipped_remember": shipped, "at_distance": at_distance, "turned_down": turned_down,
        "filler": filler,
        "cases": cases.iter().map(Case::to_json).collect::<Vec<_>>(),
        "seconds": (took * 10.0).round() / 10.0, "calls": model.calls, "retried": model.retried,
        "floor": floors,
        "screens": loaded.iter().map(|(k, v)| (k.clone(), json!(v.widgets.len())))
            .collect::<serde_json::Map<_, _>>(),
        "rows": rows,
    });
    std::fs::write(&out.path, serde_json::to_string_pretty(&written).unwrap()).unwrap();
    println!("\n  every turn of every conversation -> {}", out.path.display());
}
