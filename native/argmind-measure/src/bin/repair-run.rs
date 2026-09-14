//! Run repair, memory and language against a live model, and write the result.
//!
//! The same experiment `measure/repair.py` runs: four two-turn cases scored on the second turn,
//! then 36 Korean and 12 English single turns scored on the script of the reply. Three repeats,
//! **worst counted** — a configuration that is right twice and wrong once is a configuration that
//! is wrong, because the person only ever gets one draw.
//!
//! The English control is not decoration: without it, a model that always answers in Korean scores
//! perfectly.

use std::path::PathBuf;

use argmind_env_ollama::Local;
use argmind_measure::drive::drive;
use argmind_measure::repair::{language_worst, repair_verdict, script_of, sensitivity, worst,
                              REPAIR_WORST};
use argmind_measure::{data_print, prompt_tag, screen_print, screens, source_print};
use serde_json::Value;

struct Speaker(Local);

impl argmind_core::contract::Speaks for Speaker {
    fn ask(&mut self, prompt: &str) -> String {
        self.0.ask(prompt).map(|r| r.text).unwrap_or_default()
    }
}

fn main() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap()
        .to_path_buf();
    let repeat: usize = std::env::args().nth(1).and_then(|a| a.parse().ok()).unwrap_or(3);
    let dir = root.join("measure");
    let loaded = screens(&dir.join("screens"));
    let d: Value = serde_json::from_str(
        &std::fs::read_to_string(root.join("native").join("expectations").join("repair.json"))
            .expect("run python native/expectations/all.py first")).unwrap();

    let blob = serde_json::json!([d["cases"], d["korean"], d["english"],
                                  d["language_screen"], d["original_korean"]]);
    let f = data_print(&blob);
    let remember = d["remember_turns"].as_i64();

    let mut model = Local::new("qwen3:8b");
    model.think = Some(false);
    for _ in 0..2 {
        let _ = model.ask("Reply with the single word: ready");
    }

    println!("  qwen3:8b | {repeat} repeats, worst counted | prompt {} | screens s{} | cases f{f}",
             prompt_tag(), screen_print(&dir.join("screens")));
    println!("  harness {}\n", source_print(include_str!("repair-run.rs")));

    let mut rows: Vec<Value> = Vec::new();

    // ---- repair and memory ------------------------------------------------------------------
    println!("  REPAIR AND MEMORY — n = {} cases x {repeat}, worst counted\n",
             d["cases"].as_array().unwrap().len());
    let mut repair_bad = 0;
    for case in d["cases"].as_array().unwrap() {
        let screen = case["screen"].as_str().unwrap();
        let widgets = &loaded[screen].widgets;
        let lines: Vec<String> = case["lines"].as_array().unwrap().iter()
            .map(|l| l.as_str().unwrap().to_string()).collect();
        let want: Option<(String, String)> = case["want"].as_array()
            .map(|a| (a[0].as_str().unwrap().into(), a[1].as_str().unwrap().into()));
        let mut verdicts = Vec::new();
        let mut draws = Vec::new();
        for _ in 0..repeat {
            let turns = drive(Speaker(Local { think: Some(false), ..Local::new("qwen3:8b") }),
                              widgets, &lines, remember);
            let last = turns.last().expect("a turn");
            let clicks = last.clicks();
            let verdict = repair_verdict(&clicks, want.as_ref());
            verdicts.push(verdict);
            draws.push(serde_json::json!({
                "clicks": clicks.iter().map(|(n, r)| serde_json::json!([n, r]))
                    .collect::<Vec<_>>(),
                "verdict": verdict,
                "turns": turns.iter().map(|t| serde_json::json!({
                    "i": t.i, "said": t.said, "reply": t.reply, "clicked": t.clicked,
                    "clicked_roles": t.clicked_roles, "n_effects": t.n_effects,
                    "history_lines": t.history_lines, "seconds": t.seconds})).collect::<Vec<_>>(),
            }));
        }
        let counted = worst(&verdicts, &REPAIR_WORST);
        if counted != argmind_measure::repair::LANDED {
            repair_bad += 1;
        }
        let agreed = verdicts.iter().collect::<std::collections::BTreeSet<_>>().len() == 1;
        println!("    {:<6} {:<30} wanted {:<24} {}",
                 if counted == argmind_measure::repair::LANDED { "OK" } else { "NO" },
                 case["label"].as_str().unwrap(),
                 want.as_ref().map(|w| format!("{:?}", w)).unwrap_or_else(|| "nothing".into()),
                 if agreed { String::new() } else { format!("DISAGREED {verdicts:?}") });
        rows.push(serde_json::json!({
            "set": "repair", "case": case["label"], "screen": screen, "lines": lines,
            "want": case["want"], "why": case["why"], "verdicts": verdicts,
            "counted": counted, "agreed": agreed, "repeats": draws}));
    }

    // ---- language ---------------------------------------------------------------------------
    let language_screen = d["language_screen"].as_str().unwrap();
    let widgets = &loaded[language_screen].widgets;
    for which in ["korean", "english"] {
        let cases = d[which].as_array().unwrap();
        let mut right = 0;
        for case in cases {
            let (kind, line) = (case[0].as_str().unwrap(), case[1].as_str().unwrap());
            let mut verdicts = Vec::new();
            let mut draws = Vec::new();
            for _ in 0..repeat {
                let turns = drive(Speaker(Local { think: Some(false), ..Local::new("qwen3:8b") }),
                                  widgets, &[line.to_string()], remember);
                let t = &turns[0];
                let (got, mixed) = script_of(&t.reply);
                verdicts.push(got);
                draws.push(serde_json::json!({"say": t.reply, "got": got, "mixed": mixed,
                                              "clicked": t.clicked, "n_effects": t.n_effects}));
            }
            let counted = worst(&verdicts, &language_worst(which));
            if counted == which {
                right += 1;
            }
            rows.push(serde_json::json!({
                "set": which, "kind": kind, "said": line, "verdicts": verdicts,
                "counted": counted,
                "agreed": verdicts.iter().collect::<std::collections::BTreeSet<_>>().len() == 1,
                "repeats": draws}));
        }
        println!("\n  {} in, {} out: {right} of {}", which, which, cases.len());
    }

    println!("\n  repair: {} of {} landed — a perfect repair of the rest would give p = {:.4}",
             d["cases"].as_array().unwrap().len() - repair_bad,
             d["cases"].as_array().unwrap().len(), sensitivity(repair_bad));

    let out = dir.join(format!("repair-rust-{}-s{}-f{f}-x{repeat}.json",
                               prompt_tag(), screen_print(&dir.join("screens"))));
    std::fs::write(&out, serde_json::to_string_pretty(&serde_json::json!({
        "model": "qwen3:8b", "repeat": repeat, "prompt": prompt_tag(), "data_print": f,
        "rows": rows})).unwrap()).unwrap();
    println!("  every turn -> {}", out.display());
}
