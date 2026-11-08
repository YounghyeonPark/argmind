//! **The seam between a turn and the task loop: does a turn's own reply say *this is work*?**
//!
//! ```text
//!   cargo run -p argmind-measure --bin seam --release
//! ```
//!
//! From `native/`. **Stop argmind first.** Item A of the review was *one path*: every request into
//! the task loop, a conversation being a one-step task. Taken literally it collides with two of
//! the owner's decisions — a turn presses the desktop and a task only the pane, and a turn answers
//! in 0.6 s on qwen3:8b where a task step is seconds on 14b reasoning. The owner said *진행* to the
//! alternative: **merge the seam** — when the turn shows it cannot finish in one look, hand the
//! request to the task loop there. The router in front (`route::routed`) already sends most work
//! away; this is the net behind it, for what it misses.
//!
//! The signal is read off the reply by `task::promises_work`, a word rule written from the three
//! real replies that showed it (*찾아볼까요?*, *검색해볼까요?*, *잠시만 기다려주세요!*) before this
//! ran. **No prompt is changed**, so every conversation harness stays valid.
//!
//! Each sentence of `--bin routing`'s set (`argmind_measure::routed`) is given to a real turn —
//! the shipped prompt, qwen3:8b, thinking off, on a captured screen (a borrowed goal on its own
//! screen, everything else on the remote-desktop one `talk` uses first) — **as if the router had
//! sent it there**, which is the case the seam exists for. Three repeats.
//!
//! ## Written before the run
//!
//! * **harm** — a sentence whose only right path is `turn` handed to work. Ships only if, in the
//!   worst repeat, borrowed (54, the path every conversation harness measured) has **at most 1**
//!   and real + written + held has **at most 1**.
//! * **catch** — of the sentences whose only right path is `work`, how many a turn's reply hands
//!   over. Ships only if **at least half**, in the worst repeat. Below that the net is too loose to
//!   be worth a second way into a task.
//! * Sentences that accept both are printed and not scored.
//!
//! **Read, 2026-11-09: not wired — harm 0 and 0, caught 0 of 23 in all three repeats.** At rest,
//! with no history and no pane, a turn given work does not promise it: it **declines** it — *지금
//! 이 화면에서는 … 할 수 없습니다*, *도구가 보이지 않아요*, *다른 화면으로 전환해 주세요* — on all
//! 23 sentences, the owner's three trip requests among them. The promise the rule was written from
//! came out of the application, with the pane up (*지금은 네이버에서 검색할 수 있어요*) and a
//! history that already held offers; this harness gives neither. So the reading is not *the seam is
//! empty* but *the signal is a refusal here and a promise there*, and a rule on refusals would be
//! written from this run's replies and would also have to leave alone the nine goals a turn is
//! right to decline. Not attempted: it needs a set written before it, and the router in front
//! already sends 54 of 55 of real + written to the right place. `promises_work` stays in the core,
//! pinned by its test and called by nothing in the application.

use std::path::PathBuf;

use argmind_core::converse::{read_reply, render, MAX_LISTED};
use argmind_core::intent::Trigger;
use argmind_core::mind::Mind;
use argmind_core::task::promises_work;
use argmind_env_ollama::Local;
use argmind_measure::routed::{set, WRITTEN};
use argmind_measure::{prompt_tag, screens, source_print};

fn main() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap().to_path_buf();
    let loaded = screens(&root.join("measure").join("screens"));
    let goals: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(
        root.join("measure").join("goals.json")).unwrap()).unwrap();
    let talk: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(
        root.join("native").join("expectations").join("talking.json")).unwrap()).unwrap();
    let first = talk["where"][0].as_str().unwrap().to_string();
    let all = set();
    //: The screen each sentence is said in front of, in `set`'s order: written, goals, talk.
    let mut on: Vec<String> = vec![first.clone(); WRITTEN.len()];
    on.extend(goals["goals"].as_array().unwrap().iter().map(|g| g["screen"].as_str().unwrap().to_string()));
    on.resize(all.len(), first.clone());

    let mut model = Local::new("qwen3:8b");
    model.think = Some(false);
    for _ in 0..2 {
        let _ = model.ask("Reply with the single word: ready");
    }
    println!("\n  이음매 — 턴의 답이 '이건 일이다'라고 말하나\n  prompt {} | harness {}\n", prompt_tag(),
             source_print(include_str!("seam.rs")));

    let only = |ok: &Vec<&str>, p: &str| ok.len() == 1 && ok[0] == p;
    let mut worst = (0usize, 0usize, usize::MAX);
    let mut seen: std::collections::BTreeMap<String, usize> = Default::default();
    for repeat in 1..=3 {
        let (mut harm_b, mut harm_w, mut caught, mut work_n) = (0, 0, 0, 0);
        for ((text, src, ok), screen) in all.iter().zip(&on) {
            let widgets = &loaded[screen].widgets;
            let trigger = Trigger::said(text);
            let prompt = render(widgets, &trigger, &[], MAX_LISTED, &Mind::resting());
            let reply = model.ask(&prompt).map(|r| r.text).unwrap_or_default();
            let said = read_reply(&reply, widgets, &trigger).say;
            let over = promises_work(&said) && !argmind_core::task::asks_about(text);
            if only(ok, "work") {
                work_n += 1;
                caught += over as usize;
            }
            let harmful = over && only(ok, "turn");
            if harmful {
                if *src == "borrowed" { harm_b += 1 } else { harm_w += 1 }
            }
            if over || only(ok, "work") {
                let mark = if harmful { "✗ 해로움" } else if over { "→ 작업" } else { "· 놓침" };
                *seen.entry(format!("{mark} [{src}] {text}  — {}", said.replace('\n', " ").chars().take(90).collect::<String>()))
                    .or_default() += 1;
            }
        }
        println!("    #{repeat}: 해로운 borrowed {harm_b}/54 · 해로운 그 밖 {harm_w} · 작업만 맞는 {work_n}개 중 넘김 {caught}");
        worst = (worst.0.max(harm_b), worst.1.max(harm_w), worst.2.min(caught));
        if repeat == 1 {
            println!("      (작업만 맞는 문장 {work_n}개)");
        }
    }
    println!("\n    넘긴 것과 놓친 것 (세 번 중 몇 번, 답의 앞부분):");
    for (line, k) in &seen {
        println!("      {k}× {line}");
    }
    let work_n = all.iter().filter(|(_, _, ok)| only(ok, "work")).count();
    let ships = worst.0 <= 1 && worst.1 <= 1 && worst.2 * 2 >= work_n;
    println!("\n  ── 돌리기 전에 적어둔 대로 (반복 중 가장 나쁜 것) ──\n    해로운 borrowed {} (≤1) · 해로운 그 밖 {} (≤1) · 넘김 {}/{work_n} (≥절반) — {}",
             worst.0, worst.1, worst.2, if ships { "앱에 붙인다" } else { "앱에 붙이지 않는다" });
}
