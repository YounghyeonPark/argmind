//! **Saying their own words back at them is not a reply.**
//!
//! The owner, 2026-09-17, with every dial at 10 and 돈벌기 as the goal: 머해 → `머해?`, then
//! 왜 따라해 → `머해?`, then 왜 반복해 → `머해?`. **바본가.**
//!
//! Three things were established before any code was written, and each is why the guard has the
//! shape it has:
//!
//! * **It is not the dials.** `measure/parroting-rust-r322a4c-s5020f7-hbe2f48-x3.json` reproduces
//!   it at rest — 2 of 6 echoed and 2 stuck with nothing turned on — and the arm with the owner's
//!   exact settings echoed none.
//! * **It is not the reader.** `{"say": "머해?", "do": []}` is what comes off the wire.
//! * **It is not fixable by asking.** Given *상대가 한 말을 그대로 되돌려 말하지 말 것* in the
//!   prompt, the reply was `머해` — the question mark gone and the parrot intact.
//!
//! So it is caught on the way out. **The whole risk is over-catching**, and that is what most of
//! this file is: a greeting returned, a word quoted inside a real sentence, an answer that happens
//! to start with what was asked. Every one of those must survive, and each is a case the obvious
//! implementation of this guard gets wrong.

use argmind_core::contract::Widget;
use argmind_core::converse::{parroted, read_reply};
use argmind_core::intent::Trigger;

fn screen() -> Vec<Widget> {
    vec![Widget::new("Eraser", "button", (0, 0, 1, 1), "")]
}

fn caught(said: &str, back: &str) -> bool {
    parroted(back, &Trigger::said(said))
}

/// The screenshot, turn by turn.
#[test]
fn the_three_turns_that_started_this_are_caught() {
    assert!(caught("머해", "머해?"));
    assert!(caught("왜 따라해", "왜 따라해?"));
    //: And with the space the model sometimes leaves before the mark.
    assert!(caught("왜 반복해", "왜 반복해 ?"));
    //: A Korean keyboard produces this one too.
    assert!(caught("머해", "머해？"));
}

/// **A greeting handed back is a greeting, not a parrot.** The single most likely thing to break
/// if this guard is ever widened to plain equality.
#[test]
fn an_answer_that_is_exactly_what_was_said_is_left_alone() {
    assert!(!caught("안녕", "안녕"));
    assert!(!caught("고마워", "고마워"));
    //: Two 네s running is a conversation.
    assert!(!caught("네", "네"));
}

/// A real answer that **begins** with their words, or **contains** them, is a real answer.
#[test]
fn a_sentence_that_merely_starts_with_their_words_survives() {
    assert!(!caught("머해", "머해라고 물으셨죠? 시스템 정보를 보고 있어요."));
    assert!(!caught("안녕", "안녕하세요! 무엇을 도와드릴까요?"));
    assert!(!caught("지우개", "지우개는 22번 버튼입니다."));
    //: A question of its own, which is a different thing from their question handed back.
    assert!(!caught("머해", "뭘 도와드릴까요?"));
}

/// Nothing to mirror, nothing to catch. **A watch has no trigger text a person wrote**, and an
/// empty one must not make every short reply a parrot.
#[test]
fn an_empty_or_shorter_reply_is_never_a_parrot() {
    assert!(!caught("", "?"));
    assert!(!caught("머해", ""));
    assert!(!caught("무엇을 하고 있나요", "머해?"));
    //: Shorter than what was said cannot be it with a mark added.
    assert!(!caught("왜 따라해", "왜?"));
}

/// **The flag rides on the intent, and the words are not touched.** A transcript that quietly
/// replaced what the model said would make this undiagnosable the next time.
#[test]
fn read_reply_reports_the_mirror_and_keeps_the_words() {
    let trigger = Trigger::said("머해");
    let intent = read_reply(r#"{"say": "머해?", "do": []}"#, &screen(), &trigger);
    assert!(intent.parroted);
    assert_eq!(intent.say, "머해?", "the reply was rewritten where it should have been flagged");

    let fine = read_reply(r#"{"say": "시스템 정보를 보고 있어요.", "do": []}"#, &screen(), &trigger);
    assert!(!fine.parroted);
}

/// English, because nothing about this is Korean — the guard compares two strings and the
/// behaviour it catches is the model's, not the language's.
#[test]
fn it_is_not_a_korean_rule() {
    assert!(caught("what are you doing", "what are you doing?"));
    assert!(!caught("hello", "hello"));
    assert!(!caught("hi", "hi there, what can I do?"));
}
