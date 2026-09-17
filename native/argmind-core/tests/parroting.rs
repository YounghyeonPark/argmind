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

// ---- the fourth identical remark ---------------------------------------------------------------

/// **Saying the same thing about every window is not noticing, it is narrating.**
///
/// The owner's transcript, with watching on by default:
///
/// ```text
///   • the window changed to 'Chrome Remote Desktop - QVB-FLFM-DEV1'
///   Chrome 데스크탑으로 넘어왔어요. 도움이 필요하시면 언제든 말씀해주세요!
///   • the window changed to 'Taskbar'
///   태스크바로 넘어왔어요. 도움이 필요하시면 언제든 말씀해주세요!
/// ```
///
/// Measured over eight window changes: it spoke **8 of 8**, and **6 of those shared one shape**.
/// Telling it to be quiet in the prompt bought no silence and started it wanting to click things,
/// so this is caught on the way out — the answer that worked for the parrot.
#[test]
fn the_same_remark_about_a_different_window_is_caught() {
    use argmind_core::converse::said_before;

    let one = "태스크바로 넘어왔어요. 도움이 필요하시면 언제든 말씀해주세요!";
    let two = "Chrome 데스크탑으로 넘어왔어요. 도움이 필요하시면 언제든 말씀해주세요!";
    let three = "Snipping Tool로 넘어왔어요. 도움이 필요하시면 언제든 말씀해주세요!";

    //: **The first one is said.** The rule is against the fourth identical remark, not the first.
    let mut lately: Vec<String> = Vec::new();
    assert!(!said_before(one, &lately));
    lately.push(one.to_string());

    //: The rest are the same sentence with a different noun in it, and the name being translated —
    //: `Taskbar` arriving as 태스크바 — is exactly what broke the two attempts before this one.
    assert!(said_before(two, &lately));
    assert!(said_before(three, &lately));
}

/// **A different observation still gets through**, which is the whole reason the watch exists.
#[test]
fn something_new_about_a_new_window_is_not_suppressed() {
    use argmind_core::converse::said_before;
    let lately = vec!["태스크바로 넘어왔어요. 도움이 필요하시면 언제든 말씀해주세요!".to_string()];

    assert!(!said_before("지우개가 22번 버튼입니다.", &lately));
    assert!(!said_before("저장하지 않은 그림이 열려 있어요.", &lately));
    assert!(!said_before("빌드가 실패한 것 같습니다 — 오류 창이 떠 있어요.", &lately));
}

/// The threshold, pinned from both sides. **0.6 with the margin shown** — the pair that started
/// this shares four fifths of the shorter sentence and two real remarks share a few characters of a
/// dozen, so nothing between 0.4 and 0.8 changes either verdict.
#[test]
fn the_overlap_separates_the_two_cases_with_room() {
    use argmind_core::converse::{overlap, SAME_REMARK};

    let boilerplate = overlap("태스크바로 넘어왔어요. 도움이 필요하시면 언제든 말씀해주세요!",
                              "Chrome 데스크탑으로 넘어왔어요. 도움이 필요하시면 언제든 말씀해주세요!");
    let different = overlap("태스크바로 넘어왔어요. 도움이 필요하시면 언제든 말씀해주세요!",
                            "지우개가 22번 버튼입니다.");
    assert!(boilerplate > 0.75, "the same sentence scored {boilerplate}");
    assert!(different < 0.4, "two different remarks scored {different}");
    assert!(different < SAME_REMARK && SAME_REMARK < boilerplate);

    //: It is a ratio of the shorter one, so a long remark containing a short one entirely still
    //: counts as the same — which is what a model padding its own boilerplate does.
    assert_eq!(overlap("넘어왔어요", "지금 막 넘어왔어요, 무엇을 도와드릴까요"), 1.0);
    //: And nothing in common is nothing.
    assert_eq!(overlap("abc", "xyz"), 0.0);
    assert_eq!(overlap("", "abc"), 0.0);
}

/// **Almost nothing to compare is not a match.** Two very short remarks overlap by accident, and
/// the rule is about repetition rather than about brevity.
#[test]
fn a_remark_with_almost_nothing_of_its_own_is_left_alone() {
    use argmind_core::converse::said_before;
    let lately = vec!["네".to_string(), "그렇군요".to_string()];
    assert!(!said_before("네", &lately));
    assert!(!said_before("그렇군요", &lately), "too short to be judged a repeat");
}

/// English, because the whole reason for comparing overlap rather than removing names is that it
/// cannot be a rule about one language.
#[test]
fn the_repetition_rule_is_not_a_korean_rule() {
    use argmind_core::converse::said_before;
    let lately = vec!["You switched to the Taskbar. Let me know if you need anything!".to_string()];
    assert!(said_before("You switched to Paint. Let me know if you need anything!", &lately));
    assert!(!said_before("The eraser is button 22.", &lately));
}
