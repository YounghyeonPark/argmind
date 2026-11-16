//! **A reply in a language the person did not set is written again, once — and nothing else is.**
//!
//! Since 2026-11-17. Japanese was taken out of the languages a person could choose because this
//! model, told in Japanese to answer in Japanese, answered Korean turns in Korean or in *メモ장이
//! 열렸습니다* four times in twelve. What brought it back is [`Session::decide`] reading the reply and,
//! when it is in another script, asking the model once more to rewrite only that. These tests hold
//! the shape with a model that is a script: **the second call happens only when it should**, its
//! answer is taken only when it is in the language, and the action is the first call's.

use argmind_core::contract::{Eyes, Hands, Speaks, Widget};
use argmind_core::converse::{asks_first, in_language, read_rewrite, written_in};
use argmind_core::session::Session;
use argmind_core::intent::Trigger;
use std::sync::{Arc, Mutex};

struct Screen;

impl Eyes for Screen {
    fn foreground(&mut self) -> Vec<Widget> {
        screen()
    }
    fn timed_foreground(&mut self) -> (Vec<Widget>, f64) {
        (screen(), 0.0)
    }
}

struct Still;

impl Hands for Still {
    fn live(&self) -> bool {
        false
    }
    fn click(&mut self, _: &Widget) {}
    fn type_text(&mut self, _: &str) {}
    fn press(&mut self, _: &str) {}
}

/// Answers from a list in order and records every prompt it was asked.
struct Script(Vec<&'static str>, Arc<Mutex<Vec<String>>>);

impl Speaks for Script {
    fn ask(&mut self, prompt: &str) -> String {
        self.1.lock().unwrap().push(prompt.to_string());
        if self.0.is_empty() { String::new() } else { self.0.remove(0).to_string() }
    }
}

fn screen() -> Vec<Widget> {
    vec![Widget::new("메모장", "button", (0, 0, 40, 40), "작업 표시줄")]
}

fn turn(code: &str, said: &str, answers: Vec<&'static str>) -> (argmind_core::intent::Intent, Vec<String>) {
    let asked = Arc::new(Mutex::new(Vec::new()));
    let mut s = Session::new(Screen, Still, Script(answers, asked.clone()));
    s.mind.language = code.into();
    let intent = s.decide(&Trigger::said(said));
    let prompts = asked.lock().unwrap().clone();
    (intent, prompts)
}

/// The failure that took Japanese out, and what comes back now.
#[test]
fn a_korean_reply_with_japanese_set_is_rewritten_and_keeps_its_click() {
    let (intent, prompts) = turn("ja", "메모장 열어줘", vec![
        r#"{"say": "メモ장이 열렸습니다.", "do": [{"click": 0}]}"#,
        "メモ帳を開きました。",
    ]);
    assert_eq!(prompts.len(), 2, "{prompts:?}");
    assert!(prompts[1].starts_with("次は、あなたが相手に言おうとした言葉です"), "{}", prompts[1]);
    assert!(prompts[1].ends_with("メモ장이 열렸습니다."), "the reply is what is rewritten: {}", prompts[1]);
    assert_eq!(intent.say, "メモ帳を開きました。");
    assert_eq!(intent.rewritten_from.as_deref(), Some("メモ장이 열렸습니다."));
    //: The action is the first call's, untouched.
    assert_eq!(intent.effects.len(), 1);
    assert_eq!(intent.effects[0].target, "메모장");
}

/// **The control: a reply already in the language costs nothing.** A Japanese reply naming a button
/// that is called 메모장 is Japanese.
#[test]
fn a_reply_already_in_the_language_is_asked_once() {
    let (intent, prompts) = turn("ja", "メモ帳を開いて", vec![
        r#"{"say": "메모장を開きますね。", "do": [{"click": 0}]}"#, "使われてはいけない",
    ]);
    assert_eq!(prompts.len(), 1, "{prompts:?}");
    assert_eq!(intent.say, "메모장を開きますね。");
    assert!(intent.rewritten_from.is_none());
}

/// **Nothing set, nothing checked** — a Korean reply to an English turn is what argmind has always
/// done, and the prompt is asked once.
#[test]
fn with_no_language_set_nothing_is_rewritten() {
    let (intent, prompts) = turn("", "open Notepad", vec![
        r#"{"say": "메모장을 열게요.", "do": []}"#, "should not be asked",
    ]);
    assert_eq!(prompts.len(), 1);
    assert_eq!(intent.say, "메모장을 열게요.");
    assert!(intent.rewritten_from.is_none());
}

/// **A rewrite that is still in the wrong language is not taken**, and nor is one that runs on —
/// the model's own words stay, wrong but not invented.
#[test]
fn a_rewrite_that_fails_leaves_the_models_own_words() {
    let (intent, prompts) = turn("ja", "안녕", vec![r#"{"say": "안녕하세요!", "do": []}"#, "안녕하세요!"]);
    assert_eq!(prompts.len(), 2);
    assert_eq!(intent.say, "안녕하세요!");
    assert!(intent.rewritten_from.is_none());

    let long = "こんにちは！今日はいい天気ですね。ところで、最近どうですか？お昼は何を食べましたか？週末の予定はありますか？";
    assert!(read_rewrite(long, "안녕!", &screen(), "ja").is_none(), "an answer is not a rewrite");
    //: Quotes around the whole of it go; the words inside stay.
    assert_eq!(read_rewrite("「こんにちは！」", "안녕하세요!", &screen(), "ja").as_deref(), Some("こんにちは！"));
    assert!(read_rewrite("", "안녕", &screen(), "ja").is_none());
}

/// A mirrored reply is the interface's to replace and an empty one has nothing to rewrite.
#[test]
fn a_parrot_or_silence_is_not_rewritten() {
    let (_, prompts) = turn("ja", "머해", vec![r#"{"say": "머해?", "do": []}"#, "x"]);
    assert_eq!(prompts.len(), 1, "a parrot was rewritten");
    let (_, prompts) = turn("en", "고마워", vec![r#"{"say": "", "do": []}"#, "x"]);
    assert_eq!(prompts.len(), 1, "silence was rewritten");
}

/// **What counts as which language**, including the cases the reading must not get wrong.
#[test]
fn the_script_decides_and_the_screens_labels_do_not() {
    let s = screen();
    assert_eq!(written_in("メモ장이 열렸습니다"), Some("mixed"));
    assert_eq!(written_in("了解しました"), Some("ja"));
    assert_eq!(written_in("好的，我来打开。"), Some("zh"));
    assert_eq!(written_in("😊 123"), None);
    assert!(in_language("메모장を開きます", &s, "ja"));
    assert!(in_language("Opening 메모장 now", &s, "en"));
    assert!(!in_language("메모장을 열게요", &s, "en"));
    assert!(!in_language("메모장 열게요", &s, "ja"), "Korean around a label is Korean");
    assert!(in_language("😊", &s, "ja"), "no letters is no language, so every one");
    assert!(in_language("아무 말", &s, ""), "nothing set");
    assert!(in_language("아무 말", &s, "fr"), "an unknown code is nothing set");
    assert!(!in_language("메모帳を開きます。안녕", &s, "ja"));
}

/// **An offer is an offer in either reading** — the app holds an action as a row when the reply
/// asks, and a rewrite must not turn *열어드릴까요?* into words the rule cannot read.
#[test]
fn offers_in_japanese_and_chinese_are_offers() {
    assert!(asks_first("メモ帳を開きましょうか？"));
    assert!(asks_first("要我帮你打开吗？"));
    assert!(!asks_first("メモ帳を開きました。"));
    assert!(!asks_first("何をしますか？"));
}
