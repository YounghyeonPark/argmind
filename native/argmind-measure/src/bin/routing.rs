//! **Where does a request go — and how often is that the wrong place?**
//!
//! ```text
//!   cargo run -p argmind-measure --bin routing --release            # the rules alone, no model
//!   cargo run -p argmind-measure --bin routing --release -- --model # and the two model arms
//! ```
//!
//! From `native/`. The owner's trip request — *11월 일본여행 계획 … 스케쥴을 세워줘* — went to an
//! ordinary turn three times, answered *찾아볼까요?*, and once clicked a Git button. Routing is
//! word rules (`argmind_serve::route::by_rules`, the function `think` itself calls), so before
//! anything replaces them this measures how often they send a request to the wrong place.
//!
//! ## The set, by source, reported apart
//!
//! * **real** — the owner's own sentences: the five in `conversation.jsonl` and the two in the
//!   timelines pasted on 2026-10-21. Seven; too few alone, and the only ones nobody wrote for a test.
//! * **borrowed** — `measure/goals.json` (37 things to do on a screen, including the declines,
//!   which a turn declines) and `expectations/talking.json` (17 conversational turns). Their right
//!   path was settled before this harness existed: a turn.
//! * **written** — sentences written here on 2026-11-03, **with their accepted paths, before any
//!   arm ran**, by somebody who knew the rules — so they include traps on purpose (*문서 작성 잘하는
//!   법 알려줘* is a question; *이 문서 저장해줘* is a button). Read them as a stress test, not as
//!   the distribution of what people say.
//!
//! Paths: `turn` (reply or act on the screen), `work`, `lookup`, `web`, `keep`. A sentence may
//! accept more than one, written beside it. The pane is assumed present, no task running.
//!
//! ## Arms
//!
//! 1. **rules** — `by_rules`, as shipped.
//! 2. **model** — qwen3:8b, thinking off, asked to pick one path with a line on each. `reply` and
//!    `desktop` both count as `turn`.
//! 3. **rules, then model** — the rules first; where they say `turn`, the model's choice instead.
//!    The shape proposed in the review: the model as the router, the rules as a net.
//!
//! Three repeats for the model arms; every repeat printed.
//!
//! ## Added after the first run, and fitted to it
//!
//! The first run read **rules 46 / 55, model 50, rules→model 52 — harmful 8, 3, 3 — and borrowed 54
//! / 54 against 48 for both model arms**, identical on every repeat: neither replaced the rules,
//! on borrowed alone. The model's borrowed misses were things to do in an app it could not see —
//! *글씨 쓰고 싶어* (Paint) went to work, *도움말 보여줘* to the web. So `--window` adds the
//! window in front to the prompt, as the application could: each borrowed screen's title, and
//! *Visual Studio Code* for real and written (the window the owner most often had in front).
//! **The same criteria, but the arm was designed on the failures it is judged on.**
//!
//! ## The shipped router, and a held-out set — 2026-11-03, written before it ran
//!
//! **4. shipped** — `argmind_serve::route::routed`, what `think` now calls: the rules, then the
//! model with the window in front, **read for `work` only**, and never on a question about how
//! (`argmind_core::task::asks_about`, which the rules also obey now). Fitted twice over: the window
//! and the guard were both made from this harness's misses. So a fourth source, **held** — fourteen
//! sentences written after the guard and before any arm saw them: eight questions about how or
//! about argmind's skill that are a turn, six requests that wear a question's or a how's clothes and
//! are work.
//!
//! * **The guard holds** only if, on held, the shipped router sends **none of the eight** questions
//!   to work **and at least five of the six** requests there, in its worst repeat.
//! * **It ships** only if, on real + written, it has no more harmful misses than the rules alone
//!   and borrowed stays ≥ 95%.
//! * **Its cost is printed, not judged**: the router call's median and worst over every sentence
//!   it was asked about, beside the 0.45 s a turn costs warm.
//!
//! ## Written before the run
//!
//! * **The rules are not fit to be the only router if they miss ≥ 20%** of real + written.
//! * **A model arm replaces the rules** only if, on real + written, it is right on **at least 10% of
//!   the sentences more** in its worst repeat, **and** it makes no more *harmful* misses than the
//!   rules: work sent anywhere else (the trip request), or anything that is not work sent to work
//!   (a long task nobody asked for). And it must keep borrowed at ≥ 95% — the turn path is the one
//!   every other harness measured.

use std::collections::BTreeMap;

use argmind_serve::route::{by_rules, Route};
use argmind_measure::routed::{borrowed_or_code, set, WINDOWS};

/// The application's own prompt and reader, so the arms measure the words it sends.
fn model_route(model: &argmind_env_ollama::Local, text: &str, window: Option<&str>) -> String {
    let reply = model.ask(&argmind_serve::route::model_prompt(text, window.unwrap_or("")))
        .map(|r| r.text).unwrap_or_default();
    match argmind_serve::route::model_said(&reply).as_str() {
        "reply" | "desktop" => "turn".into(),
        other => other.to_string(),
    }
}

struct Model<'a>(&'a argmind_env_ollama::Local);
impl argmind_core::contract::Speaks for Model<'_> {
    fn ask(&mut self, prompt: &str) -> String {
        self.0.ask(prompt).map(|r| r.text).unwrap_or_default()
    }
}

/// Right, and harmful misses — work sent elsewhere, or not-work sent to work.
fn score(all: &[(String, &'static str, Vec<&'static str>)], got: &[String], source: &dyn Fn(&str) -> bool)
         -> (usize, usize, usize) {
    let (mut n, mut right, mut harm) = (0, 0, 0);
    for ((_, src, ok), g) in all.iter().zip(got) {
        if !source(src) {
            continue;
        }
        n += 1;
        if ok.contains(&g.as_str()) {
            right += 1;
        } else if ok == &vec!["work"] || (g == "work" && !ok.contains(&"work")) {
            harm += 1;
        }
    }
    (n, right, harm)
}

fn main() {
    let all = set();
    let rules: Vec<String> = all.iter().map(|(t, _, _)| by_rules(t, true, false, false).name().to_string()).collect();
    println!("\n  길 나누기 — 이 말을 어디로 보내나\n  harness {}\n", argmind_measure::source_print(include_str!("routing.rs")));
    let parts: [(&str, &dyn Fn(&str) -> bool); 5] = [
        ("real", &|s: &str| s == "real"), ("written", &|s: &str| s == "written"),
        ("real+written", &|s: &str| s == "real" || s == "written"), ("borrowed", &|s: &str| s == "borrowed"),
        ("held", &|s: &str| s == "held")];
    let report = |name: &str, got: &[String]| {
        let mut line = format!("    {name:<16}");
        for (p, f) in &parts {
            let (n, r, h) = score(&all, got, *f);
            line.push_str(&format!(" {p} {r}/{n} (해로운 {h}) ·"));
        }
        println!("{}", line.trim_end_matches(" ·"));
    };
    report("rules", &rules);
    println!("\n    rules 가 틀린 것:");
    for ((t, src, ok), g) in all.iter().zip(&rules) {
        if !ok.contains(&g.as_str()) {
            println!("      [{src}] {t}  → {g}  (맞는 길 {})", ok.join("/"));
        }
    }
    let (n, r, _) = score(&all, &rules, &|s| s == "real" || s == "written");
    println!("\n    rules 가 놓친 비율 {:.0}% — 미리 적은 대로: {}", 100.0 * (n - r) as f64 / n as f64,
             if (n - r) * 5 >= n { "규칙만으로 길을 나누기에는 맞지 않다" } else { "규칙만으로도 된다" });

    if !std::env::args().any(|a| a == "--model") {
        return;
    }
    let mut model = argmind_env_ollama::Local::new("qwen3:8b");
    model.think = Some(false);
    for _ in 0..2 {
        let _ = model.ask("Reply with the single word: ready");
    }
    let windowed = std::env::args().any(|a| a == "--window");
    let borrowed_windows: Vec<&'static str> = WINDOWS.with(|w| w.borrow().clone());
    let windows: Vec<Option<&str>> = all.iter().scan(0usize, |k, (_, src, _)| {
        Some(if !windowed { None } else if *src == "borrowed" {
            let w = borrowed_windows[*k]; *k += 1; Some(w)
        } else { Some("Visual Studio Code") })
    }).collect();
    if windowed {
        println!("\n    (--window: 앞에 있는 창 제목을 함께 보여 줌 — 첫 실행의 실패를 보고 만든 갈래)");
    }
    let mut worst: [(usize, usize, usize); 2] = [(usize::MAX, 0, 0); 2];
    let mut misses: BTreeMap<String, usize> = BTreeMap::new();
    let mut shipped_misses: BTreeMap<String, usize> = BTreeMap::new();
    //: worst right, worst harmful, worst borrowed %, worst questions sent to work, worst requests kept
    let mut sw = (usize::MAX, 0usize, 100usize, 0usize, usize::MAX);
    let mut costs: Vec<f64> = Vec::new();
    let shown = borrowed_or_code(&all, &borrowed_windows);
    for repeat in 1..=3 {
        //: **The shipped router, with the window always given** — the application always has one.
        let shipped: Vec<String> = all.iter().zip(&shown).map(|((t, _, _), w)| {
            let (route, asked) = argmind_serve::route::routed(t, true, false, false, w, &mut Model(&model));
            if let Some((_, s)) = asked {
                costs.push(s);
            }
            route.name().to_string()
        }).collect();
        println!();
        report(&format!("shipped #{repeat}"), &shipped);
        let (_, r, h) = score(&all, &shipped, &|s| s == "real" || s == "written");
        let (bn, br, _) = score(&all, &shipped, &|s| s == "borrowed");
        let held = |want: &str| all.iter().zip(&shipped)
            .filter(|((_, src, ok), g)| *src == "held" && ok[0] == want && *g == "work").count();
        sw = (sw.0.min(r), sw.1.max(h), sw.2.min(br * 100 / bn.max(1)), sw.3.max(held("turn")), sw.4.min(held("work")));
        for ((t, src, ok), g) in all.iter().zip(&shipped) {
            if !ok.contains(&g.as_str()) {
                *shipped_misses.entry(format!("[{src}] {t}  → {g}  (맞는 길 {})", ok.join("/"))).or_default() += 1;
            }
        }
        let chosen: Vec<String> = all.iter().zip(&windows).map(|((t, _, _), w)| model_route(&model, t, *w)).collect();
        let joined: Vec<String> = rules.iter().zip(&chosen)
            .map(|(r, m)| if r == "turn" { m.clone() } else { r.clone() }).collect();
        println!();
        report(&format!("model #{repeat}"), &chosen);
        report(&format!("rules→model #{repeat}"), &joined);
        for (i, got) in [&chosen, &joined].iter().enumerate() {
            let (_, r, h) = score(&all, got, &|s| s == "real" || s == "written");
            let (bn, br, _) = score(&all, got, &|s| s == "borrowed");
            if r < worst[i].0 {
                worst[i] = (r, h, br * 100 / bn.max(1));
            }
        }
        for ((t, src, ok), g) in all.iter().zip(&joined) {
            if !ok.contains(&g.as_str()) {
                *misses.entry(format!("[{src}] {t}  → {g}  (맞는 길 {})", ok.join("/"))).or_default() += 1;
            }
        }
    }
    println!("\n    shipped 가 틀린 것 (세 번 중 몇 번):");
    for (m, k) in &shipped_misses {
        println!("      {k}× {m}");
    }
    costs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let (rn, rr0, rh0) = score(&all, &rules, &|s| s == "real" || s == "written");
    println!("\n  ── shipped, 돌리기 전에 적어둔 대로 (반복 중 가장 나쁜 것) ──");
    println!("    held: 질문 8개 중 작업으로 간 것 {} · 요청 6개 중 작업으로 간 것 {} — {}", sw.3, sw.4,
             if sw.3 == 0 && sw.4 >= 5 { "막이 버틴다" } else { "막이 버티지 못한다" });
    println!("    real+written {rn}개: 맞음 {} (rules {rr0}) · 해로운 {} (rules {rh0}) · borrowed {}% — {}",
             sw.0, sw.1, sw.2, if sw.1 <= rh0 && sw.2 >= 95 { "앱에 붙인다" } else { "앱에 붙이지 않는다" });
    if !costs.is_empty() {
        println!("    모델에게 물은 것 {}번 (세 번 × {}문장) · 한 번에 중앙값 {:.2}초, 90% {:.2}초, 가장 느린 것 {:.2}초",
                 costs.len(), all.len(), costs[costs.len() / 2], costs[costs.len() * 9 / 10], costs[costs.len() - 1]);
    }
    println!("\n    rules→model 이 틀린 것 (세 번 중 몇 번):");
    for (m, k) in &misses {
        println!("      {k}× {m}");
    }
    let (_, rr, rh) = score(&all, &rules, &|s| s == "real" || s == "written");
    println!("\n  ── 돌리기 전에 적어둔 대로 (real+written {n}개, 반복 중 가장 나쁜 것) ──");
    for (i, name) in ["model", "rules→model"].iter().enumerate() {
        let (r, h, b) = worst[i];
        let replaces = r * 10 >= rr * 10 + n && h <= rh && b >= 95;
        println!("    {name:<12} 맞음 {r} (rules {rr}) · 해로운 {h} (rules {rh}) · borrowed {b}% — {}",
                 if replaces { "규칙을 대신한다" } else { "규칙을 대신하지 않는다" });
    }
}
