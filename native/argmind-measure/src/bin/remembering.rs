//! **Does a sentence the person asked argmind to keep change what it does?**
//!
//! ```text
//!   cargo run -p argmind-measure --bin remembering --release
//! ```
//!
//! From `native/`. **Stop argmind first.** No screen read, no camera, no person: a fixed list of
//! widgets, a `Mind` with one kept sentence or none, and the model.
//!
//! ## Why
//!
//! `Mind::kept` — item 4 of the dots comparison — puts the person's own sentences into every
//! prompt. That is a prompt edit, and prompt edits here were 0 for 5 until one that was *a slot
//! present only when there was something in it*. This is that shape; what is not known is whether
//! the model does anything with it, and a slot it ignores is a feature that claims what it cannot
//! keep.
//!
//! ## Written before the run
//!
//! Four kept sentences, each tried on three different turns, **each with and without the sentence**:
//!
//! * **It follows**: with the sentence, at least 2 of 3 turns show it;
//! * **and it is the sentence doing it**: without, at most 1 of 3 do. A case the model does anyway
//!   proves nothing about the slot.
//!
//! And one control, a sentence that has nothing to do with the turns (*나는 고양이를 좋아해*):
//!
//! * **it does not leak**: *고양이* in at most 1 of 6 replies;
//! * **it does not start actions**: no more spurious actions on the five conversational turns than
//!   with nothing kept.
//!
//! Temperature is the model's default, and this model has answered identically on repeats in
//! every harness this week, so the variation is in the three turns rather than in repeats.

use argmind_core::contract::Widget;
use argmind_core::intent::{Intent, Trigger};
use argmind_core::mind::Mind;
use argmind_core::session::Session;
use argmind_env_ollama::Local;
use argmind_measure::drive::{Fixed, Nothing};
use argmind_measure::source_print;

struct Model(Local);

impl argmind_core::contract::Speaks for Model {
    fn ask(&mut self, prompt: &str) -> String {
        self.0.ask(prompt).map(|r| r.text).unwrap_or_default()
    }
}

/// A taskbar's worth of buttons. **Both editors are there**, so the editor case is a choice.
fn screen() -> Vec<Widget> {
    ["메모장", "Visual Studio Code", "계산기", "파일 탐색기", "Chrome", "설정"].iter().enumerate()
        .map(|(i, n)| Widget::new(n, "button", (40 * i as i64, 1040, 40, 40), "작업 표시줄"))
        .collect()
}

fn hangul(s: &str) -> bool {
    s.chars().any(|c| ('\u{AC00}'..='\u{D7A3}').contains(&c))
}

/// Sentences, by their ends. Rough, and the same rough rule on both arms.
fn sentences(s: &str) -> usize {
    s.split(['.', '!', '?', '。']).filter(|p| p.trim().chars().count() > 1).count()
}

type Follows = fn(&Intent) -> bool;

struct Case {
    kept: &'static str,
    turns: [&'static str; 3],
    what: &'static str,
    follows: Follows,
}

const CASES: [Case; 4] = [
    Case { kept: "나를 대표님이라고 불러", turns: ["안녕", "오늘 기분 어때?", "고마워"],
           what: "대표님이라고 부름", follows: |i| i.say.contains("대표님") },
    Case { kept: "답은 한 문장으로만 해",
           turns: ["집중이 잘 안 될 때 어떻게 해?", "점심 메뉴 추천해줘", "운동을 꾸준히 하는 방법 알려줘"],
           what: "한 문장", follows: |i| !i.say.trim().is_empty() && sentences(&i.say) <= 1 },
    Case { kept: "항상 영어로 대답해", turns: ["안녕", "오늘 날씨 좋다", "뭐 하고 있어?"],
           what: "영어로 답함", follows: |i| !i.say.trim().is_empty() && !hangul(&i.say) },
    Case { kept: "메모장 말고 VS Code를 써",
           turns: ["편집기 열어줘", "글 쓸 프로그램 띄워줘", "메모할 거 하나 열어줘"],
           what: "VS Code 를 누름",
           follows: |i| i.effects.iter().any(|e| e.kind == "click" && e.target == "Visual Studio Code") },
];

const UNRELATED: &str = "나는 고양이를 좋아해";
const SMALL_TALK: [&str; 6] = ["안녕", "점심 뭐 먹을까", "고마워", "오늘 할 일이 많아", "메모장 열어줘", "잘 자"];

fn ask(kept: Option<&str>, said: &str) -> Intent {
    let mut model = Local::new("qwen3:8b");
    model.think = Some(false);
    let mut session = Session::new(Fixed(screen()), Nothing, Model(model));
    let mut mind = Mind::resting();
    if let Some(k) = kept {
        mind.kept = vec![k.to_string()];
    }
    session.mind = mind;
    session.decide(&Trigger::said(said))
}

fn shown(i: &Intent) -> String {
    let did: Vec<String> = i.effects.iter().map(|e| format!("{} {}", e.kind,
        if e.target.is_empty() { &e.text } else { &e.target })).collect();
    format!("{}{}", i.say.chars().take(70).collect::<String>(),
            if did.is_empty() { String::new() } else { format!("  [{}]", did.join(", ")) })
}

fn main() {
    println!("\n  기억한 문장이 행동을 바꾸는가 — qwen3:8b, 문장 있음/없음, 세 가지 말");
    println!("  harness {}\n", source_print(include_str!("remembering.rs")));
    let mut model = Local::new("qwen3:8b");
    model.think = Some(false);
    for _ in 0..2 {
        let _ = model.ask("Reply with the single word: ready");
    }

    let mut passed = 0;
    for case in &CASES {
        println!("  ── 「{}」 — {} ──", case.kept, case.what);
        let (mut with, mut without) = (0, 0);
        for turn in case.turns {
            let a = ask(Some(case.kept), turn);
            let b = ask(None, turn);
            let (fa, fb) = ((case.follows)(&a), (case.follows)(&b));
            with += fa as usize;
            without += fb as usize;
            println!("     {turn}\n       있음 {} {}\n       없음 {} {}",
                     if fa { "●" } else { "○" }, shown(&a), if fb { "●" } else { "○" }, shown(&b));
        }
        let ok = with >= 2 && without <= 1;
        passed += ok as usize;
        println!("     → 있음 {with}/3 · 없음 {without}/3 — {}\n",
                 if ok { "기준대로" } else { "**기준 못 미침**" });
    }

    println!("  ── 상관없는 문장 「{UNRELATED}」 ──");
    let (mut leaks, mut acts_with, mut acts_without) = (0, 0, 0);
    for turn in SMALL_TALK {
        let a = ask(Some(UNRELATED), turn);
        let b = ask(None, turn);
        if a.say.contains("고양이") {
            leaks += 1;
        }
        if turn != "메모장 열어줘" {
            acts_with += !a.effects.is_empty() as usize;
            acts_without += !b.effects.is_empty() as usize;
        }
        println!("     {turn}\n       있음 {}\n       없음 {}", shown(&a), shown(&b));
    }
    let leak_ok = leaks <= 1;
    let acts_ok = acts_with <= acts_without;
    println!("     → 고양이 {leaks}/6 — {} · 대화에서 행동 있음 {acts_with}/5, 없음 {acts_without}/5 — {}\n",
             if leak_ok { "기준대로" } else { "**새어 나옴**" },
             if acts_ok { "기준대로" } else { "**행동이 늘었음**" });

    println!("  ── 합계: 따름 {passed}/{} · 대조 {}", CASES.len(),
             if leak_ok && acts_ok { "통과" } else { "**실패**" });
}
