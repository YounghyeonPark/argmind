//! Talking: does it keep its hands off when nobody asked it to do anything?
//!
//! **The other half of `choose`.** That one measures whether it acts when it should; this measures
//! whether it stays still when it should not act at all. Citing either alone is incomplete by
//! construction — a model that clicks nothing scores perfectly on this and zero on that, and one
//! that clicks everything does the reverse.
//!
//! Seventeen things a person says that are not requests, on the **smallest and the largest**
//! captured screens. Chosen by widget count rather than by name, so the pair moves with the
//! fixtures instead of going stale beside them: if a long screen provokes action and a short one
//! does not, that shows up here and nowhere else.

use std::path::PathBuf;

use argmind_core::converse::{read_reply, render, MAX_LISTED};
use argmind_core::intent::Trigger;
use argmind_env_ollama::Local;
use argmind_measure::{prompt_tag, screen_print, screens, source_print};
use serde_json::Value;

fn main() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap()
        .to_path_buf();
    let repeat: usize = std::env::args().nth(1).and_then(|a| a.parse().ok()).unwrap_or(2);
    let dir = root.join("measure");
    let loaded = screens(&dir.join("screens"));
    let d: Value = serde_json::from_str(
        &std::fs::read_to_string(root.join("native").join("expectations").join("talking.json"))
            .expect("run python native/expectations/all.py first")).unwrap();

    let mut model = Local::new("qwen3:8b");
    model.think = Some(false);
    for _ in 0..2 {
        let _ = model.ask("Reply with the single word: ready");
    }

    let where_: Vec<&str> = d["where"].as_array().unwrap().iter()
        .map(|w| w.as_str().unwrap()).collect();
    println!("  qwen3:8b | {} things to say x {repeat} | on {} ({} widgets) and {} ({})",
             d["talk"].as_array().unwrap().len(), where_[0], loaded[where_[0]].widgets.len(),
             where_[1], loaded[where_[1]].widgets.len());
    println!("  prompt {} | screens s{} | harness {}\n", prompt_tag(),
             screen_print(&dir.join("screens")), source_print(include_str!("talk.rs")));

    let mut rows = Vec::new();
    let (mut quiet_total, mut total) = (0usize, 0usize);
    for name in &where_ {
        let widgets = &loaded[*name].widgets;
        println!("  {name} — {} widgets", widgets.len());
        for row in d["talk"].as_array().unwrap() {
            let (kind, line) = (row[0].as_str().unwrap(), row[1].as_str().unwrap());
            let mut acted: Vec<Vec<String>> = Vec::new();
            for _ in 0..repeat {
                let trigger = Trigger::said(line);
                let prompt = render(widgets, &trigger, &[], MAX_LISTED);
                let reply = model.ask(&prompt).map(|r| r.text).unwrap_or_default();
                let intent = read_reply(&reply, widgets, &trigger);
                acted.push(intent.effects.iter()
                    .map(|e| format!("{}:{}", e.kind,
                                     if e.target.is_empty() { &e.text } else { &e.target }))
                    .collect());
            }
            //: **The worst draw, not the average.** One turn in three that clicks the eraser when
            //: greeted is a turn that clicked the eraser when greeted.
            let worst = acted.iter().max_by_key(|a| a.len()).cloned().unwrap_or_default();
            let quiet = worst.is_empty();
            total += 1;
            if quiet {
                quiet_total += 1;
            } else {
                println!("    ACTED  {:<20} {worst:?}", line.chars().take(20).collect::<String>());
            }
            rows.push(serde_json::json!({"screen": name, "kind": kind, "said": line,
                                         "acted": acted, "quiet": quiet}));
        }
        println!();
    }

    println!("  by kind, over both screens:");
    let mut kinds: Vec<&str> = Vec::new();
    for row in d["talk"].as_array().unwrap() {
        let k = row[0].as_str().unwrap();
        if !kinds.contains(&k) {
            kinds.push(k);
        }
    }
    for kind in kinds {
        let group: Vec<&Value> = rows.iter().filter(|r| r["kind"] == kind).collect();
        let q = group.iter().filter(|r| r["quiet"] == true).count();
        println!("    {kind:<18} kept its hands off {q} of {}", group.len());
    }
    println!("\n  {quiet_total} of {total} conversational turns produced no action ({}%)",
             (100 * quiet_total) / total.max(1));
    println!("  the other half of this pair is `choose`, where 28 of 37 goals want one");

    let out = dir.join(format!("talking-rust-{}-s{}-x{repeat}.json", prompt_tag(),
                               screen_print(&dir.join("screens"))));
    std::fs::write(&out, serde_json::to_string_pretty(&serde_json::json!({
        "model": "qwen3:8b", "repeat": repeat, "prompt": prompt_tag(),
        "quiet": quiet_total, "turns": total, "rows": rows})).unwrap()).unwrap();
    println!("  every turn -> {}", out.display());
}
