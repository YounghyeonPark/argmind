//! **Does the language a person sets decide the language argmind answers in — and only that?**
//!
//! ```text
//!   cargo run -p argmind-measure --bin language --release
//! ```
//!
//! From `native/`. **Stop argmind first.** No screen read and no person: a taskbar's worth of
//! buttons, a `Mind` with a language or none, and the model.
//!
//! ## Why
//!
//! The owner, 2026-11-17: *사용자가 대화할 언어를 설정할수 있으면 함.* The setting is a line in
//! `Mind::line`, written in the language it asks for, because `the dials` found the slot's language
//! decides the reply's and `remembering` found a Korean sentence asking for English moved nothing
//! (0 of 3). A line is a prompt edit, so it has to be shown to move what it claims and to leave
//! the clicks alone.
//!
//! ## Written before the run
//!
//! Twelve Korean turns and the same twelve in English — eight conversational, four asking for a
//! button on screen — under five arms: nothing set (today) and each of [`LANGUAGES`].
//!
//! * **It follows**: with a language set, at least **11 of 12** replies are in it, on the Korean
//!   turns and on the English turns separately.
//! * **The clicks do not move**: per arm, the eight button requests land on the right button at
//!   least as often as with nothing set, less one; and the sixteen conversational turns start at
//!   most one action more than with nothing set. (With the later turns added, sixteen and
//!   thirty-two, so two and two.)
//! * **It holds beside the Korean lines**: English set *and* a kept Korean sentence, and Japanese
//!   set *and* a dial, each at least **11 of 12** on the Korean turns — the dials' and kept lines
//!   are Korean and were measured to pull a reply toward Korean. (Japanese's arm became Chinese's
//!   when Japanese was taken out; see *Read*.)
//!
//! A reply's language is read off its script: Hangul is Korean, kana is Japanese, Han with neither
//! is Chinese, Latin letters with none of those is English. An empty reply is in no language.
//!
//! ## Written after the first run, before the second
//!
//! The first run (`harness 14f195`) read Korean 24 of 24, English 9 + 12, Japanese 8 + 10, Chinese
//! 9 + 11, the clicks unmoved everywhere — and **English beside a kept Korean sentence 1 of 12,
//! Japanese beside a dial 0 of 12**: the frame around those lines is Korean and it won. So the frame
//! and `say`'s placeholder now come in the language set (`mind::Language`). That change was made
//! **looking at these turns**, so it is judged again on twelve more of each, [`KOREAN_LATER`] and
//! [`ENGLISH_LATER`], written before the second run and seen by nothing.
//!
//! **And the reading changed, which flatters, so both are printed.** *Opening 메모장* was counted
//! Korean because the button is called 메모장; a reply that names a button on screen is in the
//! language of its other words. So the second reading takes the screen's labels out before
//! looking. The bars are judged on that one; the raw one is beside it. And a third stress arm, a kept
//! sentence that asks for nothing Korean in the reply (*답은 한 문장으로만 해*), because the first
//! one's *대표님* is a Korean word the person asked to be called, which an English reply rightly says.
//!
//! ## Read
//!
//! **Second run** (`bcd308`): Korean, English and Chinese **12 of 12 on all four sets, held-out
//! included** — 48 of 48 each; English beside the kept *대표님* sentence 12 and 12, beside *답은 한
//! 문장으로만 해* 11 and 11; the clicks 16 of 16 (Chinese 15) against 16 at rest, and fewer actions
//! started in conversation than at rest (2–4 of 32 against 6). **Japanese failed**: Korean turns 8
//! and 7 of 12 — on a screen of Korean buttons it wrote *メモ장이 열렸습니다*, Korean grammar around a
//! half-converted name — and beside a dial 7 and 7. It was taken out of `LANGUAGES` and its stress
//! arm became Chinese beside a dial.
//!
//! **Third run** (`e68cbe`), the same arms again: 48 of 48 for each of the three, reproduced, and
//! English beside the kept *대표님* 12 and 12 — **but two stress arms missed by one**: English beside
//! *답은 한 문장으로만 해* 11 and **10**, Chinese beside a dial 12 and **10**. The misses are real
//! Korean replies (*안녕하세요!*, *메모장이 뜨고 있습니다.*), so a Korean line the person wrote still
//! pulls about one reply in twelve back to Korean. Before the frame was translated the same kind of
//! arm read 1 and 0 of 12. **Shipped against those two bars, by the person who wrote them**, for that
//! reason; it is written here to be overruled.
//!
//! **Choosing, the other half** (`choose 3`, the 37 goals, worst of three, same session; bar written
//! first: 19 of 28 and 6 of 9): at rest **24 / 7**, Korean **23 / 7**, English **21 / 9**, Chinese
//! **20 / 9**. All over the bar, inside the 21–24 this configuration has read across runs — and
//! English and Chinese both moved the same way, three and four fewer found and two more declined.
//!
//! ## Written before the fourth run — Japanese again, 2026-11-17
//!
//! The owner: *일본어도 가능하게 해줘.* Japanese comes back with **its line and frame unchanged**, so
//! the first call reproduces the failure, and a program step after it: when the reply is not in the
//! language set (read off its script with the screen's labels out, `converse::in_language`), the
//! model is asked once more to rewrite only that reply, and the rewrite is taken only when it is in
//! the language. Every arm now prints the first reply's reading beside the final one. A twelve-turn
//! Japanese set, [`JAPANESE`], written now, joins the four; and the stress arm Japanese failed
//! before, Japanese beside a dial.
//!
//! **The final count is partly made by construction**: a rewrite is taken only when it passes the
//! same script reading this harness judges by. So the bars that matter are the ones it cannot pass
//! by itself:
//!
//! * **It follows, after the rewrite**: every language at least **11 of 12** on every set, and every
//!   stress arm at least 11 of 12 on both Korean sets — the bar the third run missed by one twice.
//! * **The first call reproduces the failure**: Japanese on the Korean turns, before any rewrite,
//!   at most 10 of 12 on one of the two Korean sets. If it does not, the rewrite was not what moved it.
//! * **A rewrite says what the first reply said**: every pair is printed, and read; at most **one**
//!   in the whole run adds a claim, answers instead of rewriting, or turns an offer into a statement
//!   or back. And **no offer is lost** — a first reply that `asks_first` reads as asking whose rewrite
//!   it does not: none.
//! * **A person who writes in the language they set does not pay for it**: with Korean set on the
//!   Korean turns, English on the English turns and Japanese on the Japanese turns, the rewrite fires
//!   on at most **one turn in twelve** of each.
//! * **The clicks do not move**, as before — the rewrite cannot touch them, the line can.
//!
//! The cost is reported: how many turns paid a second call and the median seconds of a turn with
//! and without it.
//!
//! ## Read — the fourth and fifth runs (`df2726`)
//!
//! **Japanese now follows: 12, 11, 12, 11 and 12 of 12** on the five sets, the same both runs, and
//! **11 and 12 beside a dial** — the arm that read 0 of 12 on the first run. **The first call
//! reproduced the failure**, 8 and 7 of 12 on the Korean turns both times, so the rewrite is what
//! moved it: 9 of 24 Korean turns paid a second call, and **0 of 12 Japanese turns** did. Korean,
//! English and Chinese held 48 of 48 or one short on the new Japanese set (Chinese quoting *メモ帳*
//! back), and **all four stress arms reached 11** — the two that missed by one on the third run read
//! 12 and 12 now, each with one rewrite. **No offer was lost.** A turn that paid read **0.75–0.77 s**
//! median against 0.38 s; 20 and 21 of 336 turns paid. Clicks 20 of 20 for every language but
//! Chinese, 18 — the two Japanese button requests it could not match to 메모장 and 설정.
//!
//! **The faithfulness bar was missed by one, both runs, on the same two replies.** Of 20 and 21
//! rewrites, read one by one: *メモ장이 떠요* → *メモ紙が飛んでいます* and then *メモ帳が飛んでいます* —
//! 뜨다 taken as *fly* — and *메모장이 뜨고 있습니다* → *The notepad is open.*, a window coming up
//! told as one that is open. *설정을 열어드릴게요* → *設定を用意いたします* drifts and is not counted.
//! Everything else says what it said, offers included (*무엇을 도와드릴까요?* → *何かお手伝いできま
//! すか？*). **Shipped against that bar by its author**: without the rewrite those replies are Korean
//! to somebody who asked for Japanese or English, which is the failure the setting exists to fix.
//! Written here to be overruled.
//!
//! **What it cannot fix**: three replies came back half-converted (*メモ장을 열겠습니다*) or in English
//! (*anytime!*), and so did their rewrites — the model copies *メモ장* as if it were a name — so the
//! rewrite was refused and the model's own words stayed. Probed by hand afterwards with the same
//! request: *メモ장을開きます。*, *anytime! 😊*.
//!
//! **Choosing with Japanese set** (`choose 3 언어=ja`, the 37 goals, bar 19 of 28 and 6 of 9):
//! **25 / 7**, one above the 21–24 this configuration has read at rest. The rewrite cannot reach it,
//! since it never touches `do`.

use argmind_core::contract::Widget;
use argmind_core::converse::asks_first;
use argmind_core::intent::{Intent, Trigger};
use argmind_core::mind::{Mind, LANGUAGES};

const SETS: [(&str, &[(&str, Option<&str>)]); 5] =
    [("한국어로", &KOREAN), ("영어로", &ENGLISH), ("한국어로·새 문장", &KOREAN_LATER), ("영어로·새 문장", &ENGLISH_LATER),
     ("일본어로", &JAPANESE)];
use argmind_core::session::Session;
use argmind_env_ollama::Local;
use argmind_measure::drive::{Fixed, Nothing};
use argmind_measure::{mind_from, source_print};

struct Model(Local);

impl argmind_core::contract::Speaks for Model {
    fn ask(&mut self, prompt: &str) -> String {
        self.0.ask(prompt).map(|r| r.text).unwrap_or_default()
    }
}

fn screen() -> Vec<Widget> {
    ["메모장", "Visual Studio Code", "계산기", "파일 탐색기", "Chrome", "설정"].iter().enumerate()
        .map(|(i, n)| Widget::new(n, "button", (40 * i as i64, 1040, 40, 40), "작업 표시줄"))
        .collect()
}

/// Eight conversational turns, then four that ask for a button and the button they ask for.
const KOREAN: [(&str, Option<&str>); 12] = [
    ("안녕", None), ("오늘 기분 어때?", None), ("점심 뭐 먹을까", None), ("고마워", None),
    ("요즘 너무 피곤해", None), ("주말에 뭐 하면 좋을까", None), ("커피 많이 마시면 안 좋아?", None),
    ("잘 자", None),
    ("메모장 열어줘", Some("메모장")), ("계산기 켜줘", Some("계산기")), ("크롬 열어줘", Some("Chrome")),
    ("설정 열어줘", Some("설정")),
];
const ENGLISH: [(&str, Option<&str>); 12] = [
    ("hi", None), ("how are you today?", None), ("what should I have for lunch?", None),
    ("thanks", None), ("I'm so tired these days", None), ("any ideas for the weekend?", None),
    ("is too much coffee bad for you?", None), ("good night", None),
    ("open Notepad", Some("메모장")), ("open the calculator", Some("계산기")), ("open Chrome", Some("Chrome")),
    ("open settings", Some("설정")),
];

/// Twelve more of each, **written before the second run** and used for nothing else.
const KOREAN_LATER: [(&str, Option<&str>); 12] = [
    ("뭐 하고 있었어?", None), ("심심하다", None), ("좋은 노래 추천해줘", None), ("나 내일 시험 봐", None),
    ("배고파", None), ("잠이 안 와", None), ("재밌는 얘기 해줘", None), ("오늘 좀 우울해", None),
    ("파일 탐색기 열어줘", Some("파일 탐색기")), ("VS Code 열어줘", Some("Visual Studio Code")),
    ("메모장 좀 띄워줘", Some("메모장")), ("계산기 좀 열어줄래?", Some("계산기")),
];
const ENGLISH_LATER: [(&str, Option<&str>); 12] = [
    ("what were you doing?", None), ("I'm bored", None), ("recommend me a song", None),
    ("I have an exam tomorrow", None), ("I'm hungry", None), ("I can't sleep", None),
    ("tell me something fun", None), ("I feel a bit down today", None),
    ("open File Explorer", Some("파일 탐색기")), ("open VS Code", Some("Visual Studio Code")),
    ("bring up Notepad", Some("메모장")), ("can you open the calculator?", Some("계산기")),
];

/// **Written before the fourth run**, with Japanese back: eight conversational turns and four
/// buttons, the way somebody who set Japanese would write.
const JAPANESE: [(&str, Option<&str>); 12] = [
    ("こんにちは", None), ("今日の調子はどう？", None), ("お昼何食べようかな", None), ("ありがとう", None),
    ("最近すごく疲れてる", None), ("週末何しようかな", None), ("コーヒー飲みすぎると体に悪い？", None),
    ("おやすみ", None),
    ("メモ帳を開いて", Some("메모장")), ("電卓を起動して", Some("계산기")), ("Chromeを開いて", Some("Chrome")),
    ("設定を開いて", Some("설정")),
];

/// The reply with the screen's own labels taken out — see the second section above.
fn without_labels(s: &str) -> String {
    let mut out = s.to_string();
    for w in screen() {
        out = out.replace(&w.name, " ");
    }
    out
}

fn tongue(s: &str) -> &'static str {
    let any = |lo: char, hi: char| s.chars().any(|c| (lo..=hi).contains(&c));
    let hangul = any('\u{AC00}', '\u{D7A3}') || any('\u{1100}', '\u{11FF}') || any('\u{3130}', '\u{318F}');
    let kana = any('\u{3040}', '\u{30FF}');
    let han = any('\u{4E00}', '\u{9FFF}');
    let latin = s.chars().any(|c| c.is_ascii_alphabetic());
    match (hangul, kana, han, latin) {
        (true, false, _, _) => "ko",
        (false, true, _, _) => "ja",
        (false, false, true, _) => "zh",
        (false, false, false, true) => "en",
        (false, false, false, false) => "",
        _ => "mixed",
    }
}

fn ask(mind: &Mind, said: &str) -> Intent {
    let mut model = Local::new("qwen3:8b");
    model.think = Some(false);
    let mut session = Session::new(Fixed(screen()), Nothing, Model(model));
    session.mind = mind.clone();
    session.decide(&Trigger::said(said))
}

/// What one arm did on one set of turns.
#[derive(Default, Clone)]
struct Tally {
    in_it: usize,
    raw: usize,
    /// In the language before any rewrite — what the first call alone gave.
    first: usize,
    rewrites: usize,
    lost_offers: usize,
    landed: usize,
    spurious: usize,
    /// Seconds a turn took, with a rewrite and without.
    with_s: Vec<f64>,
    without_s: Vec<f64>,
}

fn run(name: &str, mind: &Mind, turns: &[(&str, Option<&str>)], want: &str) -> Tally {
    let mut t = Tally::default();
    for (said, button) in turns {
        let started = std::time::Instant::now();
        let i = ask(mind, said);
        let took = started.elapsed().as_secs_f64();
        let raw = tongue(&i.say);
        let lang = tongue(&without_labels(&i.say));
        let first = i.rewritten_from.as_deref().unwrap_or(&i.say);
        let clicked = i.effects.iter().find(|e| e.kind == "click").map(|e| e.target.as_str());
        match button {
            Some(b) => t.landed += (clicked == Some(*b)) as usize,
            None => t.spurious += !i.effects.is_empty() as usize,
        }
        let ok = !want.is_empty() && lang == want;
        t.in_it += ok as usize;
        t.raw += (!want.is_empty() && raw == want) as usize;
        t.first += (!want.is_empty() && tongue(&without_labels(first)) == want) as usize;
        if let Some(was) = &i.rewritten_from {
            t.rewrites += 1;
            t.lost_offers += (asks_first(was) && !asks_first(&i.say)) as usize;
            t.with_s.push(took);
        } else {
            t.without_s.push(took);
        }
        println!("     {name:<10} {} {said:<34} [{lang:<5}] {}{}",
                 if want.is_empty() { " " } else if ok { "●" } else { "○" },
                 i.say.chars().take(60).collect::<String>().replace('\n', " "),
                 clicked.map(|c| format!("  [click {c}]")).unwrap_or_default());
        if let Some(was) = &i.rewritten_from {
            println!("     {:<10}   {:<34}  다시 씀 ← {}{}", "", "", was.chars().take(70).collect::<String>().replace('\n', " "),
                     if asks_first(was) != asks_first(&i.say) { "   **묻는 말이 바뀜**" } else { "" });
        }
    }
    t
}

fn median(mut v: Vec<f64>) -> String {
    if v.is_empty() {
        return "-".into();
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    format!("{:.2}s", v[v.len() / 2])
}

fn main() {
    println!("\n  설정한 언어로 대답하는가 — qwen3:8b, think off, 한 번씩");
    println!("  harness {}\n", source_print(include_str!("language.rs")));
    let mut warm = Local::new("qwen3:8b");
    warm.think = Some(false);
    for _ in 0..2 {
        let _ = warm.ask("Reply with the single word: ready");
    }

    let rest = Mind::resting();
    println!("  ── 설정 없음 (오늘) ──");
    let base: Vec<Tally> = SETS.iter().map(|(_, turns)| run("없음", &rest, turns, "")).collect();
    let base_landed: usize = base.iter().map(|t| t.landed).sum();
    let base_spurious: usize = base.iter().map(|t| t.spurious).sum();
    println!("     → 버튼 {base_landed}/20 · 대화에서 행동 {base_spurious}/40\n");

    let mut passed = 0;
    let mut arms = 0;
    let mut every: Vec<Tally> = Vec::new();
    for l in &LANGUAGES {
        println!("  ── {} — {} ──", l.name, l.line);
        let mind = mind_from(&format!("언어={}", l.code));
        let got: Vec<Tally> = SETS.iter().map(|(_, turns)| run(l.code, &mind, turns, l.code)).collect();
        let landed: usize = got.iter().map(|t| t.landed).sum();
        let spurious: usize = got.iter().map(|t| t.spurious).sum();
        let follows = got.iter().all(|t| t.in_it >= 11);
        let clicks = landed + 2 >= base_landed && spurious <= base_spurious + 2;
        //: **Its own language's turns**, where a second call is a cost with nothing to buy.
        let own = SETS.iter().zip(&got)
            .filter(|((n, _), _)| (l.code == "ko" && n.starts_with("한국어"))
                || (l.code == "en" && n.starts_with("영어")) || (l.code == "ja" && n.starts_with("일본어")))
            .all(|(_, t)| t.rewrites <= 1);
        arms += 1;
        passed += (follows && clicks && own) as usize;
        let each: Vec<String> = SETS.iter().zip(&got)
            .map(|((name, _), t)| format!("{name} {}/12 (처음 {}, 다시 씀 {}, 이름 그대로 {})",
                                          t.in_it, t.first, t.rewrites, t.raw)).collect();
        println!("     → {} — {} · 버튼 {landed}/20 · 대화에서 행동 {spurious}/40 — {} · 자기 언어 다시 씀 {}\n",
                 each.join(" · "),
                 if follows { "기준대로" } else { "**기준 못 미침**" },
                 if clicks { "기준대로" } else { "**클릭이 달라짐**" },
                 if own { "기준대로" } else { "**기준 넘음**" });
        every.extend(got);
    }

    for (label, spec, code) in [("영어 + 기억한 한국어 문장", "언어=en,기억=나를 대표님이라고 불러", "en"),
                                ("영어 + 한 문장으로 답하라는 기억", "언어=en,기억=답은 한 문장으로만 해", "en"),
                                ("중국어 + 유머 7", "언어=zh,유머=7", "zh"),
                                ("일본어 + 유머 7", "언어=ja,유머=7", "ja")] {
        let mind = mind_from(spec);
        println!("  ── {label} — {} ──", mind.line().unwrap_or_default());
        let a = run(code, &mind, &KOREAN, code);
        let b = run(code, &mind, &KOREAN_LATER, code);
        let ok = a.in_it >= 11 && b.in_it >= 11;
        arms += 1;
        passed += ok as usize;
        println!("     → 한국어로 {}/12 (처음 {}, 다시 씀 {}) · 새 문장 {}/12 (처음 {}, 다시 씀 {}) — {}\n",
                 a.in_it, a.first, a.rewrites, b.in_it, b.first, b.rewrites,
                 if ok { "기준대로" } else { "**기준 못 미침**" });
        every.push(a);
        every.push(b);
    }

    let rewrites: usize = every.iter().map(|t| t.rewrites).sum();
    let lost: usize = every.iter().map(|t| t.lost_offers).sum();
    let turns = every.iter().map(|t| t.with_s.len() + t.without_s.len()).sum::<usize>();
    println!("  ── 다시 쓴 답 {rewrites}/{turns} · 묻는 말을 잃은 것 {lost} — {} · 한 턴 중앙값: 다시 쓰면 {} · 아니면 {}",
             if lost == 0 { "기준대로" } else { "**기준 못 미침**" },
             median(every.iter().flat_map(|t| t.with_s.clone()).collect()),
             median(every.iter().flat_map(|t| t.without_s.clone()).collect()));
    arms += 1;
    passed += (lost == 0) as usize;

    println!("  ── 합계: {passed}/{arms} 기준대로");
}
