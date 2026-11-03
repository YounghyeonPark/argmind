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

struct Said {
    text: &'static str,
    source: &'static str,
    ok: &'static [&'static str],
}

const fn s(text: &'static str, source: &'static str, ok: &'static [&'static str]) -> Said {
    Said { text, source, ok }
}

const T: &[&str] = &["turn"];
const W: &[&str] = &["work"];
const L: &[&str] = &["lookup"];
const WEB: &[&str] = &["web"];
const K: &[&str] = &["keep"];

const WRITTEN: &[Said] = &[
    s("11월중에 2박3일로 일본여행계획을 세우려고 하는데 2인성인으로 가장 저렴한 항공권이랑 호텔찾아서 스케쥴 작성해줘", "real", W),
    s("11월 2박3일일정으로 일본가는 항공이랑 호텔 찾아서 스케줄 만들어줘", "real", W),
    s("11월 일본여행 계획 2박3일로 성인2명 항공권이랑 호텔 조사해서 적절한 스케쥴을 세워줘", "real", W),
    s("안녕", "real", T),
    s("오늘 날씨 알려줘", "real", L),
    s("오늘 서울 날씨 알려줘", "real", L),
    s("메일 열어줘", "real", &["web", "turn"]),
    // work
    s("다운로드 폴더에 있는 pdf 파일 이름들을 목록으로 정리해줘", "written", W),
    s("이번 달 가계부 엑셀 만들어줘", "written", W),
    s("회의록 초안 작성해줘. 주제는 신제품 출시 일정", "written", W),
    s("간단한 할 일 관리 웹페이지 만들어줘", "written", W),
    s("블로그 글 하나 써줘, 주제는 가을 캠핑", "written", W),
    s("부산 1박2일 맛집 위주로 여행 일정 짜줘", "written", W),
    s("100만원 이하 노트북 세 개 비교해서 추천해줘", "written", W),
    s("주식 공부할 책 5권 조사해서 정리해줘", "written", W),
    s("제주도 렌터카 가격 비교해줘", "written", W),
    s("엑셀 파일에서 중복된 행 지워줘", "written", W),
    s("사진 폴더를 날짜별로 정리해줘", "written", W),
    s("파이썬으로 1부터 100까지 소수 구하는 코드 짜서 돌려줘", "written", W),
    s("자기소개서 첨삭해줘", "written", &["work", "turn"]),
    s("영어로 번역해줘: 내일 회의 3시로 미뤄도 될까요", "written", &["turn", "work"]),
    s("이 논문 요약해줘", "written", &["work", "turn"]),
    s("다음 주 팀 워크숍 일정표 만들어줘", "written", W),
    s("경쟁사 세 곳 서비스를 조사해서 보고서로 써줘", "written", W),
    // lookup
    s("지금 달러 환율 얼마야", "written", L),
    s("오늘 코스피 어때", "written", L),
    s("내일 부산 날씨 알려줘", "written", &["lookup", "work"]),
    s("요즘 개봉한 영화 뭐 있어", "written", &["lookup", "work"]),
    s("지금 서울 미세먼지 어때", "written", L),
    // web
    s("유튜브 열어줘", "written", WEB),
    s("네이버에서 손흥민 검색해줘", "written", &["web", "lookup"]),
    s("쿠팡 열어서 물티슈 찾아줘", "written", &["web", "work"]),
    s("웹 닫아", "written", WEB),
    s("깃허브 들어가줘", "written", WEB),
    // keep
    s("나는 커피보다 차를 좋아한다고 기억해줘", "written", K),
    s("내 생일은 3월 2일이야, 기억해줘", "written", K),
    s("앞으로 존댓말 써", "written", &["keep", "turn"]),
    // the turn: the screen
    s("메모장 열어줘", "written", T),
    s("이 창 닫아줘", "written", T),
    s("저장 버튼 눌러", "written", T),
    s("이 문서 저장해줘", "written", T),
    s("엑셀 열어줘", "written", T),
    s("이 파일 이름 바꿔줘", "written", &["turn", "work"]),
    s("복사해서 붙여넣어줘", "written", T),
    // the turn: talk
    s("고마워", "written", T),
    s("넌 누구야", "written", T),
    s("오늘 좀 피곤하다", "written", T),
    s("파이썬이 뭐야?", "written", T),
    s("점심 뭐 먹을까", "written", T),
    s("여행 가고 싶다", "written", T),
    s("문서 작성 잘하는 법 알려줘", "written", T),
    s("코드 리뷰가 뭐야?", "written", T),
    s("일정 관리 어떻게 해?", "written", T),
    s("계획 세우는 거 잘해?", "written", T),
    s("실행이 안 돼", "written", T),
];

const ASK: &str = "사람이 PC 비서에게 한 말이다. 어떤 처리로 보낼지 하나만 골라라.\n\
- reply: 대화나 지식으로 바로 답할 말 (인사, 질문, 의견, 짧은 번역)\n\
- desktop: 지금 화면의 앱에서 무언가를 누르거나 입력하는 요청 (앱 열기·닫기, 버튼, 저장, 복사)\n\
- web: 웹사이트를 열거나 닫거나, 사이트에서 검색만 해 달라는 요청\n\
- lookup: 지금·오늘의 정보(날씨, 환율, 주가, 뉴스)를 찾아 알려 달라는 요청\n\
- work: 여러 단계가 필요한 일 — 조사해서 정리, 계획 세우기, 비교, 문서·파일·코드를 만들기·고치기·실행\n\
- keep: 무엇을 기억해 달라는 요청\n\n\
한 말: ";

fn model_route(model: &argmind_env_ollama::Local, text: &str, window: Option<&str>) -> String {
    let front = window.map(|w| format!("지금 앞에 있는 창: {w}\n")).unwrap_or_default();
    let prompt = format!("{front}{ASK}{text}\n\nJSON 하나로만 답하라: {{\"route\": \"reply|desktop|web|lookup|work|keep\"}}");
    let reply = model.ask(&prompt).map(|r| r.text).unwrap_or_default();
    let route = argmind_core::llm::first_object(&reply)
        .and_then(|v| v.get("route").and_then(|r| r.as_str()).map(str::to_string))
        .unwrap_or_else(|| "?".into());
    match route.as_str() {
        "reply" | "desktop" => "turn".into(),
        other => other.to_string(),
    }
}

/// The title a borrowed screen had, by its capture's name.
fn title(screen: &str) -> &'static str {
    match screen {
        "untitled---paint" => "Untitled - Paint",
        "character-map" => "Character Map",
        "system-information" => "System Information",
        s if s.starts_with("chrome-remote-desktop") => "Chrome Remote Desktop - Google Chrome",
        _ => "Visual Studio Code",
    }
}

fn set() -> Vec<(String, &'static str, Vec<&'static str>)> {
    let mut all: Vec<(String, &'static str, Vec<&'static str>)> =
        WRITTEN.iter().map(|w| (w.text.to_string(), w.source, w.ok.to_vec())).collect();
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..");
    let goals: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(
        root.join("measure").join("goals.json")).expect("goals.json")).expect("goals");
    for g in goals["goals"].as_array().expect("goals") {
        all.push((g["goal"].as_str().unwrap_or("").to_string(), "borrowed", vec!["turn"]));
        WINDOWS.with(|w| w.borrow_mut().push(title(g["screen"].as_str().unwrap_or(""))));
    }
    let talk: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(
        root.join("native").join("expectations").join("talking.json")).expect("talking.json")).expect("talk");
    for t in talk["talk"].as_array().expect("talk") {
        let text = t[1].as_str().unwrap_or("").to_string();
        //: *오늘 날씨 어때?* was chit-chat when that set was written; with a pane it is a question
        //: about now, and either is right.
        let ok = if text.contains("날씨") { vec!["turn", "lookup"] } else { vec!["turn"] };
        all.push((text, "borrowed", ok));
        let first = talk["where"][0].as_str().unwrap_or("");
        WINDOWS.with(|w| w.borrow_mut().push(title(first)));
    }
    all
}

thread_local! {
    /// The window in front for each borrowed sentence, in order; real and written have none listed.
    static WINDOWS: std::cell::RefCell<Vec<&'static str>> = const { std::cell::RefCell::new(Vec::new()) };
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
    let parts: [(&str, &dyn Fn(&str) -> bool); 4] = [
        ("real", &|s: &str| s == "real"), ("written", &|s: &str| s == "written"),
        ("real+written", &|s: &str| s != "borrowed"), ("borrowed", &|s: &str| s == "borrowed")];
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
    let (n, r, _) = score(&all, &rules, &|s| s != "borrowed");
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
    for repeat in 1..=3 {
        let chosen: Vec<String> = all.iter().zip(&windows).map(|((t, _, _), w)| model_route(&model, t, *w)).collect();
        let joined: Vec<String> = rules.iter().zip(&chosen)
            .map(|(r, m)| if r == "turn" { m.clone() } else { r.clone() }).collect();
        println!();
        report(&format!("model #{repeat}"), &chosen);
        report(&format!("rules→model #{repeat}"), &joined);
        for (i, got) in [&chosen, &joined].iter().enumerate() {
            let (_, r, h) = score(&all, got, &|s| s != "borrowed");
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
    println!("\n    rules→model 이 틀린 것 (세 번 중 몇 번):");
    for (m, k) in &misses {
        println!("      {k}× {m}");
    }
    let (_, rr, rh) = score(&all, &rules, &|s| s != "borrowed");
    println!("\n  ── 돌리기 전에 적어둔 대로 (real+written {n}개, 반복 중 가장 나쁜 것) ──");
    for (i, name) in ["model", "rules→model"].iter().enumerate() {
        let (r, h, b) = worst[i];
        let replaces = r * 10 >= rr * 10 + n && h <= rh && b >= 95;
        println!("    {name:<12} 맞음 {r} (rules {rr}) · 해로운 {h} (rules {rh}) · borrowed {b}% — {}",
                 if replaces { "규칙을 대신한다" } else { "규칙을 대신하지 않는다" });
    }
}
