//! **Is the model surer of the clicks it gets right?** — and so, can sureness decide when argmind
//! acts on its own idea and when it asks.
//!
//! ```text
//!   cargo run -p argmind-measure --bin sure --release
//! ```
//!
//! From `native/`. **Stop argmind first.** The 37 goals of `choosing` on the captured screens, the
//! shipped prompt, one pass at temperature 0, and each reply's action scored by
//! `converse::action_sureness` — the probability of the tokens of its `do`.
//!
//! ## Why
//!
//! The owner turned on *스스로 판단한 것도 실행* on 2026-10-21, so what stands between argmind's own
//! idea and a click is the screen check and nothing that knows whether the idea was good. Jev's
//! premise is a probability with each decision; Ollama 0.40.6 returns token probabilities, and a
//! probe the same day found them **uncalibrated** — 0.90 for a wrong answer — **but ordered**: the
//! four errands that needed the web scored 0.99 or more and the nine that did not, 0.93 or less.
//! Whether that ordering holds for choosing a widget is this file's question.
//!
//! ## Written before the run
//!
//! Over the goals where it acted — right, or wrong (the wrong widget, or acting on a goal with no
//! answer):
//!
//! * **Separation**: the chance a right action outscores a wrong one (AUC) is **at least 0.75**;
//! * **a usable threshold**: one at which **at least 90%** of actions at or above it are right,
//!   **keeping at least half** of the right ones.
//!
//! Both, or sureness does not gate anything.

use std::path::PathBuf;

use argmind_core::converse::{action_sureness, read_reply, render, MAX_LISTED};
use argmind_core::intent::Trigger;
use argmind_core::mind::Mind;
use argmind_env_ollama::Local;
use argmind_measure::choosing::{goals, outcome};
use argmind_measure::{screens, source_print};

fn main() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap()
        .to_path_buf();
    let dir = root.join("measure");
    let loaded = screens(&dir.join("screens"));
    let all = goals(&std::fs::read_to_string(dir.join("goals.json")).expect("goals.json"));
    let mut model = Local::new("qwen3:8b");
    model.think = Some(false);
    model.logprobs = true;
    for _ in 0..2 {
        let _ = model.ask("Reply with the single word: ready");
    }
    println!("\n  행동의 확신도 — qwen3:8b, choosing 의 {} 목표, 한 번씩", all.len());
    println!("  harness {}\n", source_print(include_str!("sure.rs")));

    let mut acted: Vec<(f64, bool, String)> = Vec::new();
    let (mut declined, mut missed) = (0, 0);
    for g in &all {
        let Some(screen) = loaded.get(&g.screen) else { continue };
        let trigger = Trigger::said(&g.goal);
        let prompt = render(&screen.widgets, &trigger, &[], MAX_LISTED, &Mind::resting());
        let Ok((reply, tokens)) = model.ask_scored(&prompt) else { continue };
        let intent = read_reply(&reply.text, &screen.widgets, &trigger);
        let chose = intent.effects.iter().find(|e| e.kind == "click").map(|e| e.target.clone());
        let result = outcome(chose.as_deref(), g.accept.as_deref());
        let sure = action_sureness(&tokens);
        match (chose.is_some(), sure) {
            (true, Some(s)) => {
                let right = result == "right";
                acted.push((s, right, g.goal.clone()));
            }
            (true, None) => println!("  ** 행동했는데 확신도가 없음: {}", g.goal),
            (false, _) if g.accept.is_none() => declined += 1,
            (false, _) => missed += 1,
        }
    }
    acted.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    println!("  확신도 순 (● 맞음 · ○ 틀림):");
    for (s, right, goal) in &acted {
        println!("    {s:.3} {} {}", if *right { "●" } else { "○" }, goal.chars().take(40).collect::<String>());
    }
    let rights: Vec<f64> = acted.iter().filter(|a| a.1).map(|a| a.0).collect();
    let wrongs: Vec<f64> = acted.iter().filter(|a| !a.1).map(|a| a.0).collect();
    let mut wins = 0.0;
    for r in &rights {
        for w in &wrongs {
            wins += if r > w { 1.0 } else if r == w { 0.5 } else { 0.0 };
        }
    }
    let auc = if rights.is_empty() || wrongs.is_empty() { f64::NAN }
              else { wins / (rights.len() * wrongs.len()) as f64 };
    println!("\n  행동 {} (맞음 {} · 틀림 {}) · 행동 안 함: 맞게 거절 {declined} · 놓침 {missed}",
             acted.len(), rights.len(), wrongs.len());
    println!("  AUC {auc:.2} — {}", if auc >= 0.75 { "0.75 이상" } else { "**0.75 미만**" });

    //: Every threshold the data offers: at each sureness, what the actions at or above it are.
    let mut found = None;
    for (i, (s, _, _)) in acted.iter().enumerate() {
        let above = &acted[..=i];
        let r = above.iter().filter(|a| a.1).count();
        let precision = r as f64 / above.len() as f64;
        let kept = r as f64 / rights.len().max(1) as f64;
        if precision >= 0.9 && kept >= 0.5 {
            found = Some((*s, precision, kept, above.len()));
        }
    }
    match found {
        Some((s, p, k, n)) => println!("  문턱 {s:.3}: 그 위 {n}개 중 {:.0}% 맞음, 맞은 행동의 {:.0}% 유지 — 기준대로",
                                       p * 100.0, k * 100.0),
        None => println!("  **90% 이상 맞고 절반을 남기는 문턱이 없음**"),
    }
    println!("\n  ── 돌리기 전에 적어둔 대로: {} ──",
             if auc >= 0.75 && found.is_some() { "확신도로 관문을 세울 수 있음" }
             else { "**확신도는 관문이 되지 못함**" });
}
