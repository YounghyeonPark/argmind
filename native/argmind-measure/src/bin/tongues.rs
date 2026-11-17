//! **Does a request go to the same place in English, Chinese and Japanese as in Korean?**
//!
//! ```text
//!   cargo run -p argmind-measure --bin tongues --release            # the rules alone, no model
//!   cargo run -p argmind-measure --bin tongues --release -- --model # and the shipped router, x3
//! ```
//!
//! From `native/`. The owner, 2026-11-17: *대화언어도 다국어 지원이 되어야함* — the day argmind could
//! be set to answer in four languages and its page could be read in four. What it *understands* was
//! still Korean: every word rule (`asked_to_keep`, `asks_for_work`, `asks_to_look_up`, `web_move`,
//! `asks_about`, *멈춰*) is a list of Korean words, so *remember that I like tea* was a turn, and a
//! turn cannot keep anything.
//!
//! ## The set
//!
//! `argmind_measure::routed::WRITTEN` — the 69 real, written and held sentences `--bin routing`
//! reads — and **the same 69 translated into each language** (`OTHER_TONGUES`), each keeping its
//! Korean's source and accepted paths. Translated before any rule for those languages was written,
//! by the author who then wrote the rules: a stress test of *the same requests*, not the
//! distribution of what people say. Borrowed (`goals.json`, `talking.json`) is Korean only and is
//! `--bin routing`'s. The pane is assumed present and no task running, as there.
//!
//! ## Written before the run — 2026-11-17
//!
//! * **The baseline** is the rules as they were: predicted to send nearly every keep, web, lookup
//!   and work sentence in the other three languages to a turn, and Korean as `--bin routing` reads
//!   it.
//! * **Korean does not move**: the change ships only if all 69 Korean sentences go exactly where
//!   they went before it (printed as a list to compare), and `--bin routing` still reads rules 47
//!   of 55 real + written and 54 of 54 borrowed.
//! * **Rules alone, each language**: right on at least Korean's count **less 5** of the 69, and
//!   harmful misses (work sent elsewhere, or not-work sent to work) at most Korean's **plus 2**.
//! * **The shipped router** (`route::routed`: the rules, then the model shown the window, read for
//!   `work` only), worst of three repeats, each language: right at least Korean's worst **less 3**,
//!   harmful at most Korean's **plus 2**, and **none of the eight held questions about how sent to
//!   work** — the guard `--bin routing` was built to hold.

//!
//! ## And what the work model writes, said in the person's language — `--reports`, written first
//!
//! A task's closing words, an errand's result, a task's question and a request argmind set itself
//! are written by the model from Korean prompts, and reach the person outside a turn — so the
//! rewrite `Session::decide` makes for a reply never touched them. `converse::say_in` is the one
//! the application now calls, taken only when the rewrite keeps every number and every file name.
//! Twelve Korean texts below, shaped on what this project's harnesses and the owner's sessions
//! recorded (none copied from the owner's files), each said into English, Chinese and Japanese:
//!
//! * **in the language** at least 11 of 12 for each — a refused rewrite leaves Korean and counts
//!   against it;
//! * **no fact lost** in what is said, by construction — refusals are counted and printed;
//! * **meaning**, all 36 printed and read: at most **1 of 36** changes what was done or found;
//! * **Korean costs nothing**: said into Korean, 12 of 12 come back unchanged with no model call.

use argmind_measure::routed::written_in;
use argmind_serve::route::by_rules;

const TONGUES: [&str; 4] = ["ko", "en", "zh", "ja"];

struct Model<'a>(&'a argmind_env_ollama::Local);
impl argmind_core::contract::Speaks for Model<'_> {
    fn ask(&mut self, prompt: &str) -> String {
        self.0.ask(prompt).map(|r| r.text).unwrap_or_default()
    }
}

/// Right, harmful, and held questions sent to work.
fn score(all: &[(String, &'static str, Vec<&'static str>)], got: &[String]) -> (usize, usize, usize) {
    let (mut right, mut harm, mut asked) = (0, 0, 0);
    for ((_, src, ok), g) in all.iter().zip(got) {
        if ok.contains(&g.as_str()) {
            right += 1;
        } else if ok == &vec!["work"] || (g == "work" && !ok.contains(&"work")) {
            harm += 1;
        }
        if *src == "held" && ok[0] == "turn" && g == "work" {
            asked += 1;
        }
    }
    (right, harm, asked)
}

fn by_path(all: &[(String, &'static str, Vec<&'static str>)], got: &[String]) -> String {
    let mut out = Vec::new();
    for path in ["work", "lookup", "web", "keep", "turn"] {
        let mine: Vec<usize> = (0..all.len()).filter(|&i| all[i].2[0] == path).collect();
        let right = mine.iter().filter(|&&i| all[i].2.contains(&got[i].as_str())).count();
        out.push(format!("{path} {right}/{}", mine.len()));
    }
    out.join(" · ")
}

const REPORTS: [&str; 12] = [
    "ideas.md 에 부업 아이디어 10개를 정리했습니다. 가장 쉬운 것부터 순서대로 적었어요.",
    "box.stl 을 만들었습니다 — 40 × 20 × 10 mm, 작업 폴더에 있어요.",
    "primes.py 를 실행했더니 1부터 100까지 소수가 25개 나왔습니다: 2, 3, 5, 7, 11, …, 97.",
    "가장 싼 항공권은 이스타항공 직항 369,700원이고, 호텔은 렘 롯폰기 496,290원입니다. 일정은 plan.md 에 적었습니다.",
    "test_calc.py 의 실패하던 테스트 3개를 고쳤고, 다시 돌려서 모두 통과했습니다.",
    "sales.csv 의 합계는 1,234,500원, 평균은 41,150원입니다. 잘못된 줄 1개는 빼고 계산했습니다.",
    "우산은 필요하지 않습니다. 오늘 서울은 맑고 최고 18도입니다.",
    "부모님 생신 선물로는 건강식품, 안마기, 가족 사진 액자를 추천합니다. 예산 5만원 안에서 고를 수 있어요.",
    "회의록에 넣을 참석자 이름을 알려 주시겠어요?",
    "수익 창출 방법 10가지를 조사해서 수익_방법_목록.md 에 정리해줘",
    "같은 호출을 세 번 되풀이해서 멈췄습니다",
    "README.md 의 설치 방법, 사용법, 라이선스 세 부분을 고쳤습니다.",
];

struct Counted<'a>(&'a argmind_env_ollama::Local, usize);
impl argmind_core::contract::Speaks for Counted<'_> {
    fn ask(&mut self, prompt: &str) -> String {
        self.1 += 1;
        self.0.ask(prompt).map(|r| r.text).unwrap_or_default()
    }
}

fn reports() {
    let mut model = argmind_env_ollama::Local::new("qwen3:8b");
    model.think = Some(false);
    for _ in 0..2 {
        let _ = model.ask("Reply with the single word: ready");
    }
    println!("  ── 작업이 쓴 말을 그 언어로 ──");
    let mut ko = Counted(&model, 0);
    let same = REPORTS.iter().filter(|r| argmind_core::converse::say_in(&mut ko, r, "ko") == **r).count();
    println!("    ko: 그대로 {same}/12 · 모델 호출 {}", ko.1);
    let mut times = Vec::new();
    for code in ["en", "zh", "ja"] {
        let (mut inside, mut refused) = (0, 0);
        println!("\n    {code}:");
        for r in REPORTS {
            let mut m = Counted(&model, 0);
            let t = std::time::Instant::now();
            let said = argmind_core::converse::say_in(&mut m, r, code);
            times.push(t.elapsed().as_secs_f64());
            let ok = argmind_core::converse::in_language(&said, &[], code);
            if ok {
                inside += 1;
            }
            //: Refused: asked, and the words came back as they were.
            if m.1 > 0 && said == r {
                refused += 1;
                //: What the model wrote this time, so a refusal can be read for why.
                let raw = model.ask(&argmind_core::converse::rewrite_prompt(r, code).unwrap_or_default())
                    .map(|x| x.text).unwrap_or_default();
                println!("      ✗ {r}\n         (거절된 다시 쓰기, 다시 물은 것: {})", raw.trim());
            } else {
                println!("      {} {said}", if ok { "·" } else { "✗" });
            }
        }
        println!("    {code}: 그 언어로 {inside}/12 — {} · 사실을 잃어 거절 {refused}",
                 if inside >= 11 { "기준 넘음" } else { "기준 못 넘음" });
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!("\n    한 번에 중앙값 {:.2}초, 가장 느린 것 {:.2}초", times[times.len() / 2], times[times.len() - 1]);
}

fn main() {
    if std::env::args().any(|a| a == "--reports") {
        println!("\n  harness {}\n", argmind_measure::source_print(include_str!("tongues.rs")));
        reports();
        return;
    }
    println!("\n  네 언어로 길 나누기 — 같은 요청이 같은 곳으로 가나\n  harness {}\n",
             argmind_measure::source_print(include_str!("tongues.rs")));
    let sets: Vec<_> = TONGUES.iter().map(|c| written_in(c)).collect();
    let n = sets[0].len();
    let rules: Vec<Vec<String>> = sets.iter()
        .map(|all| all.iter().map(|(t, _, _)| by_rules(t, true, false, false).name().to_string()).collect())
        .collect();
    let (ko_right, ko_harm, _) = score(&sets[0], &rules[0]);
    println!("  ── rules ──");
    for (k, code) in TONGUES.iter().enumerate() {
        let (r, h, q) = score(&sets[k], &rules[k]);
        let pass = k == 0 || (r + 5 >= ko_right && h <= ko_harm + 2);
        println!("    {code}  맞음 {r}/{n} · 해로운 {h} · 방법 질문이 작업으로 {q} · {}{}", by_path(&sets[k], &rules[k]),
                 if k == 0 { String::new() } else { format!(" — {}", if pass { "기준 넘음" } else { "기준 못 넘음" }) });
    }
    for (k, code) in TONGUES.iter().enumerate().skip(1) {
        println!("\n    {code} rules 가 틀린 것:");
        for ((t, src, ok), g) in sets[k].iter().zip(&rules[k]) {
            if !ok.contains(&g.as_str()) {
                println!("      [{src}] {t}  → {g}  (맞는 길 {})", ok.join("/"));
            }
        }
    }
    //: **Korean, one line a sentence**, to compare against the run before the change.
    println!("\n    ko 길, 문장마다 (바뀌기 전과 견줄 것):");
    println!("      {}", rules[0].iter().map(|r| &r[..1]).collect::<String>());

    if !std::env::args().any(|a| a == "--model") {
        return;
    }
    let mut model = argmind_env_ollama::Local::new("qwen3:8b");
    model.think = Some(false);
    for _ in 0..2 {
        let _ = model.ask("Reply with the single word: ready");
    }
    //: worst right, worst harmful, worst held questions to work — per language.
    let mut worst = [(usize::MAX, 0usize, 0usize); 4];
    let mut misses: Vec<std::collections::BTreeMap<String, usize>> = vec![Default::default(); 4];
    let mut costs: Vec<f64> = Vec::new();
    for repeat in 1..=3 {
        println!();
        for (k, code) in TONGUES.iter().enumerate() {
            let got: Vec<String> = sets[k].iter().map(|(t, _, _)| {
                let (route, asked) = argmind_serve::route::routed(t, true, false, false, "Visual Studio Code",
                                                                  &mut Model(&model));
                if let Some((_, s)) = asked {
                    costs.push(s);
                }
                route.name().to_string()
            }).collect();
            let (r, h, q) = score(&sets[k], &got);
            println!("    shipped #{repeat} {code}  맞음 {r}/{n} · 해로운 {h} · 방법 질문이 작업으로 {q} · {}",
                     by_path(&sets[k], &got));
            worst[k] = (worst[k].0.min(r), worst[k].1.max(h), worst[k].2.max(q));
            for ((t, src, ok), g) in sets[k].iter().zip(&got) {
                if !ok.contains(&g.as_str()) {
                    *misses[k].entry(format!("[{src}] {t}  → {g}  (맞는 길 {})", ok.join("/"))).or_default() += 1;
                }
            }
        }
    }
    for (k, code) in TONGUES.iter().enumerate() {
        println!("\n    {code} shipped 가 틀린 것 (세 번 중 몇 번):");
        for (m, c) in &misses[k] {
            println!("      {c}× {m}");
        }
    }
    costs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let (kr, kh, _) = worst[0];
    println!("\n  ── shipped, 돌리기 전에 적어둔 대로 (세 번 중 가장 나쁜 것) ──");
    for (k, code) in TONGUES.iter().enumerate() {
        let (r, h, q) = worst[k];
        let pass = r + 3 >= kr && h <= kh + 2 && q == 0;
        println!("    {code}  맞음 {r}/{n} · 해로운 {h} · 방법 질문이 작업으로 {q}{}", if k == 0 { String::new() }
                 else { format!(" — {}", if pass { "기준 넘음" } else { "기준 못 넘음" }) });
    }
    if !costs.is_empty() {
        println!("    모델에게 물은 것 {}번 · 중앙값 {:.2}초, 가장 느린 것 {:.2}초",
                 costs.len(), costs[costs.len() / 2], costs[costs.len() - 1]);
    }
}
