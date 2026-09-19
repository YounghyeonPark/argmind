//! Does it judge, or does it just narrate and wait?
//!
//! ```text
//!   judging            6 ticks over each of the captured screens
//!   judging 2          two, while working
//! ```
//!
//! **The owner, 2026-09-17**, looking at what the thinking tick produced:
//!
//! > 믿는 것: 사용자가 감사의 말을 반복하며 대화를 조용히 마무리하고 있다.
//! > 기다리는 것: 사용자가 다시 말을 걸어올 수 있는 기회를 기다리고 있다.
//!
//! **생각이 그냥 기다리는중이면 의미가 없음. ai가 계속해서 스스로 판단할줄 알아야 함.**
//!
//! He was right, and the fault was in the question. The second field asked *what are you waiting
//! for*, and a watcher asked that answers *for you to say something* — two sentences of narration
//! with no conclusion anywhere in them. It now asks **what, if anything, is worth doing**, which is
//! a judgement and can be wrong.
//!
//! ## What is written down before the run
//!
//! **`worth` should be empty most of the time.** 31 of 34 conversational turns are measured as ones
//! that should produce no action at all, and there is no reason a screen nobody pointed at should
//! be more actionable than a sentence somebody said. So:
//!
//! * **empty in the clear majority** — a field that is always full is a yes-machine, and a
//!   yes-machine is worse than no field because it makes every prompt carry a suggestion nobody
//!   wanted. If this reads above about 70% full, the field has failed and should come out.
//! * **and not always empty either.** A field that never fills is the old defect wearing different
//!   words: it would mean the model has been told to shut up rather than to judge.
//! * **none of them the idle shape** — *waiting for the user to say something* and its relatives.
//!   That is the exact sentence this replaced, and it reappearing under a new field name is the
//!   failure that would otherwise look like success.
//!
//! The first two are a range rather than a target, because nothing here knows the right rate and
//! pretending to would be inventing a number. **The third is a hard zero**, and it is the one that
//! settles whether the rewrite did anything.
//!
//! ## The harness could not tell silence from silencing, so two made-up screens were added
//!
//! The captured screens are static, so **the only event this could produce was one window becoming
//! another** — the most routine thing argmind ever sees, since something on screen changes 13
//! seconds out of 14. A judgement of *nothing to do* there may simply be right, and a run that is
//! empty everywhere cannot be read either way.
//!
//! So two sequences are built rather than captured, and they are the discriminator:
//!
//! * **a dialog appears** — an ordinary window, then the same window with *저장하지 않은 변경
//!   내용이 있습니다* and three buttons on it. If a judgement is ever worth making, it is here.
//! * **nothing happens** — the same window three times. If one is ever worth withholding, here.
//!
//! **Written before the run:** full on the first, empty on the second. Empty on both means the
//! field was silenced rather than taught — which an all-empty run would otherwise be mistaken for.
//! Full on both means it is noise.
//!
//! ## How it runs
//!
//! Ticks chained the way the agent chains them — each one is shown the note the last one left — and
//! the screen is changed halfway, because a belief that never has to be abandoned is never tested.
//! `held` should climb while the screen is steady and fall back to 1 when it changes; that it is
//! counted rather than asked for is checked in `argmind-core`'s tests, and what is checked here is
//! that it behaves over a real sequence.

use std::path::PathBuf;

use argmind_core::converse::{describe_screen, MAX_LISTED};
use argmind_core::thinking::{ask_for_a_note, read_note, what_changed, Note, MOST_CHARS};
use argmind_env_ollama::Local;
use argmind_measure::{screen_print, screens, source_print, thinking_tag, Destination};
use serde_json::{json, Value};

/// The shapes that mean *I am waiting for you*, which is what this rewrite exists to remove.
///
/// **A word list, and that is a weak instrument said out loud.** It catches the sentence that
/// started this and its near relatives; it cannot catch a paraphrase nobody thought of. Every
/// `worth` is written to the result file verbatim so a later reading can score it differently
/// without running the model again.
const IDLE: [&str; 6] = ["기다리", "말을 걸", "말 걸", "기회를", "지켜보", "대기"];

fn main() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap()
        .to_path_buf();
    let ticks: usize = std::env::args().nth(1).and_then(|a| a.parse().ok()).unwrap_or(6);
    let dir = root.join("measure");
    let loaded = screens(&dir.join("screens"));

    let s = screen_print(&dir.join("screens"));
    let h = source_print(include_str!("judging.rs"));
    //: **The question is a condition of this measurement.** It lives in `argmind-core`, not in this
    //: file, so without `t` a rewritten prompt writes over a run it has nothing in common with.
    //: This harness refused such a run on 2026-09-17, which is how the gap was found.
    let t = thinking_tag();
    let out = Destination::new(&dir, "judging",
                               &["qwen3-8b".into(), format!("s{s}"), format!("h{h}"), t.clone(),
                                 format!("x{ticks}")]);
    if let Err(why) = out.refuse_if_taken() {
        println!("\n  ** {why}");
        std::process::exit(3);
    }

    let mut names: Vec<&String> = loaded.keys().collect();
    names.sort();
    println!("\n  qwen3:8b | {ticks} ticks over each of {} screens, chained | screens s{s} | 질문 {t}",
             names.len());
    println!("  적어둔 예상: 할 만한 것이 대개 비어 있을 것. 70% 넘게 차면 실패.\n");

    let mut model = Local::new("qwen3:8b");
    model.think = Some(false);
    model.keep_alive = "300s".into();

    //: --- the two built sequences, first, because the run turns on them ------------------------
    let plain: Vec<argmind_core::contract::Widget> =
        ["파일", "편집", "보기", "도움말", "제목 없음 - 메모장"].iter()
            .map(|n| argmind_core::contract::Widget::new(n, "menu item", (0, 0, 1, 1), "메모장"))
            .collect();
    let mut asking = plain.clone();
    for (n, role) in [("저장하지 않은 변경 내용이 있습니다", "text"), ("저장", "button"),
                      ("저장 안 함", "button"), ("취소", "button")] {
        asking.push(argmind_core::contract::Widget::new(n, role, (0, 0, 1, 1), "메모장"));
    }
    //: **A third sequence, and it is what the raising rule turns on.** The dialog appears and then
    //: nothing changes for four more looks — which is the only way a judgement can stand long
    //: enough to be worth interrupting somebody over, and the only way to find out whether it does.
    let built: [(&str, Vec<Vec<argmind_core::contract::Widget>>); 3] = [
        ("만든 것 — 대화상자가 뜸", vec![plain.clone(), plain.clone(), asking.clone()]),
        ("만든 것 — 아무 일도 없음", vec![plain.clone(), plain.clone(), plain.clone()]),
        ("만든 것 — 뜨고 나서 가만히", vec![plain.clone(), asking.clone(), asking.clone(),
                                          asking.clone(), asking.clone(), asking.clone()]),
    ];
    let mut rows: Vec<Value> = Vec::new();
    for (label, sequence) in &built {
        let mut note = Note::blank();
        let mut last_seen: Vec<argmind_core::contract::Widget> = Vec::new();
        println!("  --- {label}");
        for (tick, listed) in sequence.iter().enumerate() {
            let screen = describe_screen(listed, MAX_LISTED);
            let changed = what_changed(&last_seen, listed);
            last_seen = listed.clone();
            let prompt = ask_for_a_note(&note, &screen, &changed, &[], MOST_CHARS);
            let reply = model.ask(&prompt).map(|r| r.text).unwrap_or_default();
            if !read_note(&reply, &mut note, listed, &changed) {
                println!("    {tick:>2}  (읽을 수 없는 답)");
                continue;
            }
            //: **A widget, not a sentence.** What is scored is whether it named something and what.
            let worth = note.worth.as_ref()
                .map(|w| format!("{} \"{}\"", w.role, w.name)).unwrap_or_default();
            let idle = IDLE.iter().any(|w| worth.contains(w));
            println!("    {tick:>2}  {:<7} stuck {:>2} {:<8} {}",
                     if changed.is_empty() { "그대로" } else { "달라짐" }, note.stuck,
                     if note.worth_raising().is_some() { "<- 꺼냄" } else { "" },
                     if worth.is_empty() { "비어 있음".to_string() }
                     else { worth.chars().take(52).collect::<String>() });
            let raising = note.worth_raising().is_some();
            rows.push(json!({"screen": label, "tick": tick, "built": true, "held": note.held,
                             "changed": changed, "believes": note.believes, "stuck": note.stuck,
                             "raising": raising, "worth": worth, "idle": idle}));
            //: **Raised once, then put down**, exactly as `app.rs` does it — a harness that left it
            //: standing would report it raised on every remaining tick and call that persistence.
            if raising {
                note.let_go();
            }
        }
        println!();
    }

    for (i, name) in names.iter().enumerate() {
        //: **The screen changes halfway**, so a belief has to be abandoned at least once. One that
        //: never has to be is never tested.
        let other = names[(i + 1) % names.len()];
        let mut note = Note::blank();
        let mut last_seen: Vec<argmind_core::contract::Widget> = Vec::new();
        println!("  --- {name}  (절반부터 {other})");
        for tick in 0..ticks {
            let showing = if tick < ticks / 2 { name } else { other };
            let widgets = &loaded[showing.as_str()].widgets;
            let listed: Vec<_> = widgets.iter().take(MAX_LISTED).cloned().collect();
            let screen = describe_screen(&listed, MAX_LISTED);
            //: **What happened since the last tick**, computed the way the agent computes it. On
            //: the ticks where the screen did not move this is empty, which is exactly the
            //: condition under which `worth` is supposed to be empty too.
            let changed = what_changed(&last_seen, &listed);
            last_seen = listed.clone();
            let prompt = ask_for_a_note(&note, &screen, &changed, &[], MOST_CHARS);
            let reply = model.ask(&prompt).map(|r| r.text).unwrap_or_default();
            if !read_note(&reply, &mut note, &listed, &changed) {
                println!("    {tick:>2}  (읽을 수 없는 답)");
                rows.push(json!({"screen": showing, "tick": tick, "unreadable": true}));
                continue;
            }
            //: **A widget, not a sentence.** What is scored is whether it named something and what.
            let worth = note.worth.as_ref()
                .map(|w| format!("{} \"{}\"", w.role, w.name)).unwrap_or_default();
            let idle = IDLE.iter().any(|w| worth.contains(w));
            let mark = if worth.is_empty() { "비어 있음" }
                       else if idle { "IDLE 모양 " } else { "판단      " };
            println!("    {tick:>2}  held {:>2}  {:<7} {mark} {}", note.held,
                     if changed.is_empty() { "그대로" } else { "달라짐" },
                     worth.chars().take(36).collect::<String>());
            rows.push(json!({"screen": showing, "tick": tick, "held": note.held,
                             "changed": changed, "stuck": note.stuck,
                             "believes": note.believes, "worth": worth, "idle": idle}));
        }
        println!();
    }

    //: **The discriminator, read on its own.** The captured screens can only say how often the
    //: field fills; only the built pair can say whether an empty run is judgement or silence.
    let last_of = |label: &str| -> String {
        rows.iter().filter(|r| r["screen"] == label && r["built"] == true)
            .next_back().map(|r| r["worth"].as_str().unwrap_or("").to_string()).unwrap_or_default()
    };
    let on_dialog = last_of("만든 것 — 대화상자가 뜸");
    let on_nothing = last_of("만든 것 — 아무 일도 없음");
    //: **Did anything ever stand long enough to be raised?** The machinery is checked without a
    //: model in `argmind-core`; what this asks is whether a real sequence ever gets there.
    let raised = rows.iter().filter(|r| r["raising"] == true).count();
    let longest = rows.iter().filter_map(|r| r["stuck"].as_u64()).max().unwrap_or(0);
    println!("  가장 오래 선 판단: {longest}번  |  스스로 꺼낸 횟수: {raised}");
    println!("  대화상자가 떴을 때: {}", if on_dialog.is_empty() {
        "비어 있음  <-- 침묵당한 것".to_string() } else { on_dialog.clone() });
    println!("  아무 일도 없을 때: {}", if on_nothing.is_empty() {
        "비어 있음".to_string() } else { format!("{on_nothing}  <-- 잡음") });
    println!();

    let real: Vec<&Value> = rows.iter()
        .filter(|r| r.get("unreadable").is_none() && r.get("built").is_none()).collect();
    let filled = real.iter().filter(|r| !r["worth"].as_str().unwrap().is_empty()).count();
    let idle = real.iter().filter(|r| r["idle"].as_bool().unwrap_or(false)).count();
    let n = real.len().max(1);
    println!("  {} ticks read | 할 만한 것이 차 있던 것 {filled} ({:.0}%) | \
              그 중 IDLE 모양 {idle}", real.len(), 100.0 * filled as f64 / n as f64);

    //: **Judged against what was written before the run**, both ways, because a field that never
    //: fills is the old defect in new words rather than a success.
    println!();
    if idle > 0 {
        println!("  **실패: 기다린다는 문장이 {idle}번 돌아왔습니다.** 칸 이름만 바뀌고 질문이\n  \
                  그대로인 것이고, 이 측정이 잡으라고 있는 바로 그 경우입니다.");
    } else if filled * 10 > n * 7 {
        println!("  **실패: {:.0}% 가 차 있습니다.** 판단이 아니라 예스맨입니다. 이 칸은 모든\n  \
                  프롬프트에 아무도 원하지 않은 제안을 싣게 되므로 빼는 게 낫습니다.",
                 100.0 * filled as f64 / n as f64);
    } else if on_dialog.is_empty() {
        println!("  **실패: 대화상자가 떠도 아무 말이 없습니다.** 판단하라고 한 게 아니라 입을 다물라고 한 것입니다.");
    } else if !on_nothing.is_empty() {
        println!("  **실패: 아무 일도 없는데 할 말이 있습니다.** 사건에 걸린 판단이 아니라 그냥 채우는 것입니다.");
    } else if filled == 0 {
        println!("  **엇갈림: 담긴 화면에선 한 번도 안 찼지만 대화상자에는 답했습니다.** 담긴 화면이 만들 수 있는 사건은 창이 통째로 바뀌는 것뿐이고, 그건 14초에 13번 있는 일입니다.");
    } else {
        println!("  **예상대로: {:.0}% 만 차 있고 기다린다는 문장은 없습니다.** 대부분의 틱에서\n  \
                  할 게 없다고 판단했고, 그게 31/34 라는 숫자가 가리키는 방향입니다.",
                 100.0 * filled as f64 / n as f64);
    }

    std::fs::write(&out.path, serde_json::to_string_pretty(&json!({
        "model": "qwen3:8b", "ticks": ticks, "screens_print": s, "harness_print": h,
        "thinking_prompt": t,
        "idle_words": IDLE, "rows": rows,
        "filled": filled, "idle_shaped": idle, "read": real.len()})).unwrap()).unwrap();
    println!("\n  every tick -> {}", out.path.display());
}
