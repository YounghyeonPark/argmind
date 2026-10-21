//! **A history fitted to the model's context: does it remember more, and does it choose worse?**
//!
//! ```text
//!   cargo run -p argmind-measure --bin context --release
//! ```
//!
//! From `native/`. **Stop argmind first.** The owner, 2026-10-21: *대화를 하다보면 이전 대화내용을
//! 기억을 못하는것 같음*, and then asked for the conversation to be kept in a file with a context
//! sized to the model. `Session::context_tokens` fills the prompt with as many whole turns as fit
//! beside everything else; `None` keeps the twelve-turn cut every harness here was measured under.
//!
//! ## Written before the run
//!
//! * **M1, remembering**: a fact said in the first turn, twenty turns of ordinary talk, then a
//!   question about the fact, three facts. **Fitted: at least 2 of 3 recalled. Twelve turns: at
//!   most 1 of 3** — otherwise the twenty turns did not push the fact out and the case tells
//!   nothing apart.
//! * **M2, the cost**: `choosing`'s 37 goals with the same twenty turns of talk before each, fitted,
//!   against no history at all. **Correct decisions drop by at most 3.** A longer prompt is not free
//!   — `crowding` saw choosing collapse under a longer list — and this is where it would show.

use std::path::PathBuf;

use argmind_core::contract::{Speaks, Widget};
use argmind_core::intent::Trigger;
use argmind_core::session::Session;
use argmind_env_ollama::Local;
use argmind_measure::choosing::{goals, outcome};
use argmind_measure::drive::{Fixed, Nothing};
use argmind_measure::{screens, source_print};

struct Model(Local);

impl Speaks for Model {
    fn ask(&mut self, prompt: &str) -> String {
        self.0.ask(prompt).map(|r| r.text).unwrap_or_default()
    }
}

fn model() -> Model {
    let mut m = Local::new("qwen3:8b");
    m.think = Some(false);
    Model(m)
}

/// Twenty turns of the ordinary talk a session has, with fixed replies, no model calls.
fn filler() -> Vec<(String, String)> {
    let said = ["오늘 할 일이 많네", "점심 뭐 먹을까", "이 코드 좀 봐줘", "커피 마시고 올게", "회의록 정리해야 해",
                "메일 확인했어", "날이 좀 춥다", "이거 저장해줘", "잠깐 쉬자", "문서 열어줘",
                "다음 주 일정 어때", "파일이 너무 많네", "이 창 닫아줘", "음악 좀 틀까", "피곤하다",
                "보고서 다 썼어", "이따 다시 보자", "검색 좀 해볼게", "고마워", "다른 거 하자"];
    said.iter().flat_map(|s| [("them".to_string(), s.to_string()),
                              ("you".to_string(), "네, 알겠어요.".to_string())]).collect()
}

fn session(screen: Vec<Widget>, history: Vec<(String, String)>, fitted: bool)
           -> Session<Fixed, Nothing, Model> {
    let mut s = Session::new(Fixed(screen), Nothing, model());
    s.history = history;
    if fitted {
        s.context_tokens = Some(8192);
    }
    s
}

fn main() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap()
        .to_path_buf();
    let dir = root.join("measure");
    let loaded = screens(&dir.join("screens"));
    let all = goals(&std::fs::read_to_string(dir.join("goals.json")).expect("goals.json"));
    let mut warm = model();
    for _ in 0..2 {
        let _ = warm.ask("Reply with the single word: ready");
    }
    println!("\n  컨텍스트에 맞춘 대화 기억 — qwen3:8b, 8192 토큰");
    println!("  harness {}\n", source_print(include_str!("context.rs")));

    //: M1 on a small screen, so the history is most of the prompt.
    let small: Vec<Widget> = ["메모장", "Chrome", "계산기"].iter().enumerate()
        .map(|(i, n)| Widget::new(n, "button", (40 * i as i64, 1040, 40, 40), "작업 표시줄")).collect();
    let facts = [("내 고양이 이름은 나비야", "내 고양이 이름이 뭐라고 했지?", "나비"),
                 ("우리 팀 회의는 목요일 3시야", "팀 회의 몇 시라고 했지?", "3시"),
                 ("내 차는 파란색 아반떼야", "내 차 무슨 색이라고 했지?", "파란")];
    let (mut fit_hits, mut turn_hits) = (0, 0);
    println!("  ── M1: 첫 턴의 사실, 대화 20턴 뒤에 묻기 ──");
    for (fact, question, answer) in facts {
        let mut history = vec![("them".to_string(), fact.to_string()),
                               ("you".to_string(), "기억할게요.".to_string())];
        history.extend(filler());
        for fitted in [true, false] {
            let mut s = session(small.clone(), history.clone(), fitted);
            let shown = s.recent_for("").len();
            let intent = s.decide(&Trigger::said(question));
            let hit = intent.say.contains(answer);
            if fitted { fit_hits += hit as usize } else { turn_hits += hit as usize }
            println!("    {} {:<8} (줄 {shown:>2}) {question} → {}",
                     if hit { "●" } else { "○" }, if fitted { "맞춤" } else { "12턴" },
                     intent.say.chars().take(50).collect::<String>());
        }
    }
    println!("    맞춤 {fit_hits}/3 · 12턴 {turn_hits}/3 — {}",
             if fit_hits >= 2 && turn_hits <= 1 { "기준대로" } else { "**기준과 다름**" });

    println!("\n  ── M2: choosing 37 목표, 앞에 대화 20턴 (맞춤) 대 대화 없음 ──");
    let mut correct = [0usize; 2];
    for g in &all {
        let Some(screen) = loaded.get(&g.screen) else { continue };
        for (arm, history) in [(0, Vec::new()), (1, filler())] {
            let mut s = session(screen.widgets.clone(), history, arm == 1);
            let intent = s.decide(&Trigger::said(&g.goal));
            let chose = intent.effects.iter().find(|e| e.kind == "click").map(|e| e.target.clone());
            let r = outcome(chose.as_deref(), g.accept.as_deref());
            if r == "right" || r == "declined" {
                correct[arm] += 1;
            }
        }
    }
    let drop = correct[0] as i64 - correct[1] as i64;
    println!("    대화 없음 {}/{} · 대화 20턴 {}/{} — 차이 {drop} — {}", correct[0], all.len(),
             correct[1], all.len(), if drop <= 3 { "기준대로" } else { "**3 넘게 떨어짐**" });
}
