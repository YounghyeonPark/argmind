//! Run the choosing measurement against a live model, and write the result.
//!
//! 37 goals against four captured screens, three repeats, worst counted — the same experiment
//! `measure/choosing.py` runs. The point of this binary is that its verdicts can be compared with
//! that one's **in the same session**, which is the only way to know the port measures the same
//! thing rather than merely something.

use std::collections::BTreeMap;
use std::path::PathBuf;

use argmind_core::contract::Widget;
use argmind_core::converse::{read_reply, render, MAX_LISTED};
use argmind_core::intent::Trigger;
use argmind_env_ollama::Local;
use argmind_measure::choosing::{as_json, goals, score, tally};
use argmind_core::mind::Mind;
use argmind_measure::{mind_from, mind_print, prompt_tag, screen_print, screens,
                      source_print};

fn main() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap()
        .to_path_buf();
    let repeat: usize = std::env::args().nth(1).and_then(|a| a.parse().ok()).unwrap_or(3);
    //: The second argument is the settings dials, e.g. `choose 3 "찬의=7"`.
    //: Absent is at rest, which renders the prompt this harness has always rendered.
    let mind = std::env::args().nth(2).map(|s| mind_from(&s)).unwrap_or_else(Mind::resting);
    let dir = root.join("measure");
    let loaded = screens(&dir.join("screens"));
    let all = goals(&std::fs::read_to_string(dir.join("goals.json")).expect("goals.json"));

    let mut model = Local::new("qwen3:8b");
    model.think = Some(false);
    //: Two throwaway calls, because one does not finish warming: 4.7 s and 5.9 s, then 0.33.
    for _ in 0..2 {
        let _ = model.ask("Reply with the single word: ready");
    }

    println!("  qwen3:8b | {} goals x {repeat} | {} screens | prompt {} | screens {}",
             all.len(), loaded.len(), prompt_tag(), screen_print(&dir.join("screens")));
    println!("  harness {}", source_print(include_str!("choose.rs")));
    match mind.line() {
        Some(line) => println!("  mind {} | {line}\n", mind_print(&mind).unwrap()),
        None => println!("  mind at rest\n"),
    }

    let mut rows = Vec::new();
    let start = std::time::Instant::now();
    for g in &all {
        let Some(screen) = loaded.get(&g.screen) else { continue };
        let widgets: &Vec<Widget> = &screen.widgets;
        let mut chose = Vec::new();
        for _ in 0..repeat {
            let trigger = Trigger::said(&g.goal);
            let prompt = render(widgets, &trigger, &[], MAX_LISTED, &mind);
            let reply = model.ask(&prompt).map(|r| r.text).unwrap_or_default();
            let intent = read_reply(&reply, widgets, &trigger);
            //: The **first** click, as the Python takes it: a reply naming several is one decision
            //: and its first is what would have happened.
            chose.push(intent.effects.iter().find(|e| e.kind == "click")
                       .map(|e| e.target.clone()));
        }
        let row = score(g, widgets, chose);
        println!("  {:<9} {:<14} {:<28} {}", row.counted.to_uppercase(),
                 g.screen.chars().take(14).collect::<String>(),
                 g.goal.chars().take(28).collect::<String>(),
                 row.chose[0].clone().unwrap_or_else(|| "None".into()));
        rows.push(row);
    }

    let t = tally(&rows);
    println!("\n  {} goals in {:.0} s", rows.len(), start.elapsed().as_secs_f64());
    println!("    right      {:>3} / {} answerable", t.right, t.answerable);
    println!("    declined   {:>3} / {} unanswerable", t.declined, t.unanswerable);
    println!("    wrong {} | missed {} | disagreed {} across {repeat} runs at temperature 0",
             t.wrong, t.missed, t.disagreed);
    println!("    -> {} of {} correct decisions", t.right + t.declined, rows.len());

    let counts: BTreeMap<String, usize> =
        loaded.iter().map(|(k, v)| (k.clone(), v.widgets.len())).collect();
    let m = mind_print(&mind).map(|t| format!("-{t}")).unwrap_or_default();
    let out = dir.join(format!("choosing-rust-{}-s{}{m}-x{repeat}.json",
                               prompt_tag(), screen_print(&dir.join("screens"))));
    std::fs::write(&out, serde_json::to_string_pretty(&as_json(&rows, &counts)).unwrap()).unwrap();
    println!("\n  every choice -> {}", out.display());
}
