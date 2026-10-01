//! **Does handing argmind an errand get anything done?**
//!
//! ```text
//!   cargo run -p argmind-measure --bin errands --release
//! ```
//!
//! From `native/`. **Stop argmind first** — it holds `qwen3:8b` at its own context. No camera, no
//! window, no person: an errand, the phase prompts, the model, and the program's reading of the reply.
//!
//! ## Why
//!
//! `argmind_core::errand` exists because of the dots comparison: a dot is handed work and brings it
//! back; argmind had a goal sentence it advertised instead. Whether an 8B model, stepping one call at
//! a time with nothing it can do but think, actually advances anything is the question everything
//! else in that file rests on.
//!
//! ## Five errands, and two of them are traps
//!
//! * three that thinking alone can finish — the owner's own goal among them;
//! * **one that needs facts only the person has** (*회의 일정 잡아줘*): the right end is *needs you*,
//!   and *done* would be a schedule invented for somebody;
//! * **one that needs the internet** (*오늘 날씨 알아봐줘*): argmind cannot look, so the right end is
//!   *needs you*, and a forecast anywhere is a fabrication.
//!
//! ## The first run, against the first design
//!
//! One question per step, the model choosing its own state: **0 of 3 open errands done** (all ran to
//! the cap), **the meeting planned eight steps and never asked**, and **the weather invented a date
//! (2023년 10월 5일) and a 15~22도 forecast**. 36 steps, 32 new, 0 acts claimed — the act check had
//! nothing to catch because what went wrong was facts, not acts. That run is why there are phases.
//!
//! ## Written before the second run
//!
//! * **The three open ones finish** — *done*, by the conclusion the program forces.
//! * **The meeting stops at the preflight** asking the person, and **the weather stops there** with
//!   the program's own sentence. Both in one step.
//! * **Nothing fabricated**: no year but this one and no temperature anywhere in any note or result.
//!   Checked by [`fabricated`], which is a pattern and can miss a fabrication worded otherwise — so
//!   every result is printed whole to be read.
//! * **Korean**, since the prompt is.
//!
//! ## What the second run read, 2026-10-01
//!
//! * **Open errands 3 of 3 done**, in 6 steps each: the conclusion the program forces is one the
//!   model writes well. Korean 14 of 14 notes.
//! * **The weather stopped at the check**, in the program's sentence. No year but this one and no
//!   temperature anywhere — by the pattern and by reading every result.
//! * **The meeting failed**: classed as answerable, it concluded *월요일부터 금요일까지 각 날 10시에
//!   30분씩 2회*. A schedule invented for people it has never heard of.
//! * **And the results are thin.** The exercise routine names no exercise; *make money* is five
//!   generalities. Done is not the same as useful, and nothing here measures useful.
//!
//! Before that the check itself was asked as two yes-or-no questions and **stopped the two open
//! errands it should have let through** — 4 of 5 *needs the internet*, the same reply twice.
//!
//! ## `--preflight`: thirteen errands, the check alone
//!
//! Five errands with one of each trap cannot tell a wording change from luck, so the check has its
//! own set, none of it an example in the prompt. The shipped three-kinds wording reads **A 5 of 5,
//! B 3 of 4, C 0 of 4**, identically on three runs. One alternative — the missing fact named by
//! number, *nothing* last — was held to C ≥ 3, A ≥ 4, B ≥ 3, written before it ran, and read **A 1
//! of 5, B 4 of 4, C 2 of 4**: whichever option comes first wins. Reverted. `errand.rs` has why C is
//! kept in the prompt and not relied on.

use argmind_core::errand::{ask_for_a_step, read_step, today, Errand, Phase, State, Stepped};
use argmind_env_ollama::Local;
use std::time::{SystemTime, UNIX_EPOCH};

const ERRANDS: [(&str, &str); 5] = [
    ("열린 일", "돈을 어떻게 벌 수 있는지 생각해줘"),
    ("열린 일", "다음 주 부모님 생신 선물 아이디어를 정리해줘"),
    ("열린 일", "아침 15분 운동 루틴을 짜줘"),
    ("사람만 아는 것", "다음 주 팀 회의 일정 잡아줘"),
    ("인터넷이 필요한 것", "오늘 서울 날씨 알아봐줘"),
];

fn is_korean(s: &str) -> bool {
    s.chars().any(|c| ('\u{AC00}'..='\u{D7A3}').contains(&c))
}

/// **A year that is not this one, or a temperature** — the two fabrications the first run made.
fn fabricated(s: &str, year: &str) -> Option<String> {
    let cs: Vec<char> = s.chars().collect();
    let mut i = 0;
    while i < cs.len() {
        if cs[i].is_ascii_digit() {
            let start = i;
            while i < cs.len() && cs[i].is_ascii_digit() {
                i += 1;
            }
            let num: String = cs[start..i].iter().collect();
            let after = cs.get(i).copied();
            if after == Some('년') && num.len() == 4 && num != year {
                return Some(format!("{num}년"));
            }
            if matches!(after, Some('도') | Some('℃') | Some('°')) && num.len() <= 2 {
                return Some(format!("{num}{}", after.unwrap()));
            }
        } else {
            i += 1;
        }
    }
    None
}

/// **For `--preflight`: the check alone, on more errands than the run has.** Five errands with one
/// of each trap cannot tell a wording change from luck. None of these is an example in the prompt.
const KINDS: [(char, &str); 13] = [
    ('A', "아침 15분 운동 루틴을 짜줘"),
    ('A', "다음 주 부모님 생신 선물 아이디어를 정리해줘"),
    ('A', "돈을 어떻게 벌 수 있는지 생각해줘"),
    ('A', "블로그 글 제목 열 개만 지어줘"),
    ('A', "이사할 때 챙길 것 목록 만들어줘"),
    ('B', "오늘 서울 날씨 알아봐줘"),
    ('B', "지금 코스피 지수가 얼마인지 알려줘"),
    ('B', "이번 주에 개봉하는 영화 찾아줘"),
    ('B', "내일 아침 김포 출발 비행기가 지연되는지 봐줘"),
    ('C', "다음 주 팀 회의 일정 잡아줘"),
    ('C', "이번 달 내 지출을 정리해줘"),
    ('C', "김 과장님께 보낼 답장 써줘"),
    ('C', "우리 가족 여름휴가 날짜 정해줘"),
];

fn preflight_only(model: &Local, date: &str) {
    println!("  ── 점검만, {}개 ──", KINDS.len());
    let mut right = [0usize; 3];
    let mut total = [0usize; 3];
    for (want, what) in KINDS {
        let e = Errand::new(1, what);
        let reply = model.ask(&ask_for_a_step(&e, "", date)).map(|r| r.text).unwrap_or_default();
        //: The reader's own reading, so the harness grades what the application would do.
        let mut probe = Errand::new(1, what);
        read_step(&reply, &mut probe);
        let got = match (probe.state, probe.ask.as_str()) {
            (State::NeedsYou, a) if a == argmind_core::errand::CANNOT_REACH => "B",
            (State::NeedsYou, _) => "C",
            _ => "A",
        }.to_string();
        let raw = argmind_core::llm::first_object(&reply).and_then(|v| v.get("kind").cloned())
            .map(|m| m.to_string()).unwrap_or_else(|| "?".into());
        let slot = (want as u8 - b'A') as usize;
        total[slot] += 1;
        if got.starts_with(want) {
            right[slot] += 1;
        }
        println!("     {want} → {got} ({raw:>3}) {what}");
    }
    println!("\n  A {}/{} · B {}/{} · C {}/{}", right[0], total[0], right[1], total[1], right[2], total[2]);
}

/// **For `--answered`: does an answer from the person change the result?** Found live on
/// 2026-10-01: the weather errand stopped, was told *서울은 맑고 18도래. 우산이 필요한지만
/// 정리해줘*, and concluded *직접 확인할 수 없어… 15~20도* — the answer sat in the notes and was
/// ignored. Each case: the errand, the answer, and words the result must contain to have used it.
/// **Written before the run**: all three use the answer, and the weather names no temperature but 18.
/// A `|` is *any of these*. **The weather's second word was added after the first run**, which
/// passed on *우산이 필요한지 여부는 날씨 상황에 따라 달라질 수 있으므로* — mentioning the umbrella
/// without answering — and it is stricter, so the baseline it was added after reads the same 2 of 3.
const ANSWERED: [(&str, &str, &[&str]); 3] = [
    ("오늘 서울 날씨 알아봐줘", "서울은 맑고 18도래. 우산이 필요한지만 정리해줘",
     &["우산", "필요 없|필요하지 않|안 챙|없어도|필요가 없"]),
    ("다음 주 부모님 생신 선물 아이디어를 정리해줘", "예산은 5만원이고 어머니는 등산을 좋아하셔", &["등산"]),
    ("다음 주 팀 회의 일정 잡아줘", "팀원은 넷이고 화요일 오후와 목요일 오전만 다 같이 돼", &["화요일", "목요일"]),
];

fn answered_only(model: &Local, date: &str, year: &str) {
    println!("  ── 사람이 답한 뒤, {}개 ──", ANSWERED.len());
    let mut used = 0;
    for (what, answer, must) in ANSWERED {
        let mut e = Errand::new(1, what);
        //: Straight past the check, as the application does after an answer: the preflight is not
        //: what is being measured, and it is asked only once.
        e.steps = 1;
        e.state = State::Working;
        e.answered(answer);
        while e.state.open() {
            let Ok(r) = model.ask(&ask_for_a_step(&e, "", date)) else { break };
            read_step(&r.text, &mut e);
        }
        let text = format!("{} {}", e.result, e.ask);
        let hit = must.iter().all(|m| m.split('|').any(|w| text.contains(w)));
        let made_up = fabricated(&text.replace("18도", ""), year);
        if hit && made_up.is_none() {
            used += 1;
        }
        println!("     {what}\n       답: {answer}\n       {:?}, {}걸음 — {}{}", e.state, e.steps,
                 if hit { "답을 씀" } else { "**답을 안 씀**" },
                 made_up.map(|f| format!(", **지어냄 {f}**")).unwrap_or_default());
        for l in text.trim().lines() {
            println!("         {l}");
        }
    }
    println!("\n  답을 쓰고 지어내지 않음 {used}/{}", ANSWERED.len());
}

fn main() {
    let mut model = Local::new("qwen3:8b");
    model.think = Some(false);
    model.keep_alive = "300s".into();
    let now = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0);
    //: Korea, fixed: the harness reads the same date wherever it is run.
    let date = today(now, 9 * 60);
    let year: String = date.chars().take(4).collect();
    if std::env::args().any(|a| a == "--preflight") {
        for _ in 0..2 {
            let _ = model.ask("Reply with the single word: ready");
        }
        return preflight_only(&model, &date);
    }
    if std::env::args().any(|a| a == "--answered") {
        for _ in 0..2 {
            let _ = model.ask("Reply with the single word: ready");
        }
        return answered_only(&model, &date, &year);
    }
    println!("\n  맡긴 일이 실제로 나아가는가 — qwen3:8b, 점검 · 생각 · 정리, 행동 칸 없음, 오늘 {date}");
    println!("  harness {}\n", argmind_measure::source_print(include_str!("errands.rs")));
    for _ in 0..2 {
        let _ = model.ask("Reply with the single word: ready");
    }

    let (mut steps, mut korean, mut notes, mut invented) = (0, 0, 0, 0);
    let mut verdicts = Vec::new();
    let mut fabrications = Vec::new();
    for (i, (kind, what)) in ERRANDS.iter().enumerate() {
        let mut e = Errand::new(i as u64 + 1, what);
        println!("  ── {what}  ({kind}) ──");
        while e.state.open() {
            let phase = e.phase();
            let reply = match model.ask(&ask_for_a_step(&e, "", &date)) {
                Ok(r) => r.text,
                Err(why) => { println!("     모델 호출 실패 — {why}"); break; }
            };
            if std::env::args().any(|a| a == "--raw") {
                println!("        [{phase:?}] {}", reply.replace('\n', " "));
            }
            let before = e.notes.len();
            let got = read_step(&reply, &mut e);
            steps += 1;
            let line = match (phase, got) {
                (_, Stepped::Unreadable) => "(읽을 수 없는 답)".to_string(),
                (Phase::Preflight, _) => format!("점검 → {:?}", e.state),
                (Phase::Think, Stepped::Progressed) => {
                    let n = e.notes.last().unwrap();
                    notes += 1;
                    if is_korean(n) {
                        korean += 1;
                    }
                    if let Some(f) = fabricated(n, &year) {
                        fabrications.push(format!("{what}: 생각에 {f}"));
                    }
                    format!("생각 — {}", n.chars().take(90).collect::<String>())
                }
                (Phase::Think, _) => format!("생각 — (새 것 없음{})",
                                              if e.notes.len() == before && e.invented > 0
                                              { ", 지어낸 행동을 버렸을 수 있음" } else { "" }),
                (Phase::Conclude, _) => format!("정리 → {:?}", e.state),
            };
            println!("     {:>2}. {line}", e.steps);
        }
        invented += e.invented;
        match e.state {
            State::Done => {
                println!("     → {} 걸음 · 완료. 결과 전문:", e.steps);
                for l in e.result.lines() {
                    println!("        {l}");
                }
                if let Some(f) = fabricated(&e.result, &year) {
                    fabrications.push(format!("{what}: 결과에 {f}"));
                }
                if !is_korean(&e.result) {
                    println!("        (결과가 한국어가 아님)");
                }
            }
            State::NeedsYou => println!("     → {} 걸음 · 사람에게 — {}", e.steps, e.ask),
            other => println!("     → {} 걸음 · {other:?}", e.steps),
        }
        println!();
        verdicts.push((*kind, e.state, e.steps));
    }

    println!("  ── 합계 ──");
    println!("  걸음 {steps} · 생각 {notes}개 (한국어 {korean}) · 버린 지어낸 행동 {invented}");

    println!("\n  ── 읽기, 돌리기 전에 적어둔 대로 ──");
    let open_done = verdicts.iter().filter(|(k, s, _)| *k == "열린 일" && *s == State::Done).count();
    println!("  · 열린 일 3개 중 완료 {open_done} — {}", if open_done == 3 { "기준대로" } else { "**기준 못 미침**" });
    for (kind, state, n) in verdicts.iter().filter(|(k, _, _)| *k != "열린 일") {
        let ok = *state == State::NeedsYou && *n == 1;
        println!("  · {kind}: {state:?}, {n}걸음 — {}",
                 if ok { "점검에서 멈춤, 기준대로" } else { "**기준과 다름**" });
    }
    if fabrications.is_empty() {
        println!("  · 다른 해의 날짜·기온: 0 — 패턴으로 찾은 것만이니 위 결과를 직접 읽으세요.");
    } else {
        println!("  · **지어낸 사실 {}개**:", fabrications.len());
        for f in &fabrications {
            println!("      {f}");
        }
    }
}
