//! **The dials, and the one guarantee that made it safe to add them.**
//!
//! `tests/equivalence.rs` already checks the harder half from outside: with a resting [`Mind`] the
//! six probes still hash to `r322a4c`, a tag recorded from the Python before a dial existed. What
//! is here is the other direction — that the dials are *reachable*, that turning one changes the
//! prompt and turning it back restores it exactly, and that a slider which moves nothing would be
//! caught rather than shipped.

use argmind_core::contract::Widget;
use argmind_core::converse::{render, MAX_LISTED};
use argmind_core::intent::Trigger;
use argmind_core::mind::{Mind, DIALS, MOST};

fn screen() -> Vec<Widget> {
    vec![Widget::new("one", "button", (0, 0, 1, 1), ""),
         Widget::new("two", "combo box", (0, 0, 1, 1), "")]
}

fn prompt(mind: &Mind) -> String {
    render(&screen(), &Trigger::said("probe"), &[], MAX_LISTED, mind)
}

/// **Nothing set, nothing said.**
///
/// Would not notice: a slot that renders empty *text* but still pushes a line — hence the byte
/// comparison against a `Mind::default()` render as well as the `None`.
#[test]
fn a_mind_nobody_touched_says_nothing_at_all() {
    let resting = Mind::resting();
    assert!(resting.at_rest());
    assert_eq!(resting.line(), None);
    assert_eq!(Mind::default(), resting, "the derived default drifted from resting");
    //: A mind with an aim of nothing but spaces is still a mind nobody touched.
    let mut blank = Mind::resting();
    blank.aim = "   \n  ".into();
    assert_eq!(blank.line(), None);
    assert_eq!(prompt(&blank), prompt(&resting));
}

/// **A dial that moves nothing is worse than no dial**, so this fails if one is unwired.
///
/// Every dial separately, because a slot that renders only the first would pass a test that turns
/// them all at once.
#[test]
fn every_dial_on_its_own_changes_what_the_model_is_sent() {
    let resting = prompt(&Mind::resting());
    for (korean, gloss) in DIALS {
        let mut mind = Mind::resting();
        mind.turn(korean, 7);
        let line = mind.line().unwrap_or_else(|| panic!("{korean} renders nothing"));
        assert!(line.contains(korean), "{korean} does not reach the prompt: {line}");
        //: **The gloss goes too, and it is the one the page shows.** A slider labelled one way and
        //: a prompt saying another is the interface lying about what the number does.
        assert!(line.contains(gloss), "{korean} loses its gloss {gloss:?}: {line}");
        //: The number, not just the word — *playful 2* and *playful 9* are different instructions.
        assert!(line.contains(&format!("7/{MOST}")), "{korean} loses its number: {line}");
        let turned = prompt(&mind);
        assert_ne!(turned, resting, "{korean} changes nothing in the prompt");
        assert!(turned.contains(&line), "{korean}'s line is not in the prompt");
    }
}

/// **Turning it back puts the prompt back, to the byte.** Catches a slot that leaves a blank line
/// behind when it empties, which is exactly how a "no change at rest" claim goes quietly false
/// after somebody edits the assembly around it.
#[test]
fn turning_a_dial_up_and_down_again_restores_the_prompt_exactly() {
    let resting = prompt(&Mind::resting());
    let mut mind = Mind::resting();
    mind.turn("재치", MOST);
    mind.aim = "코드 리뷰 중에 놓친 것만 말해줘".into();
    assert_ne!(prompt(&mind), resting);
    mind.turn("재치", 0);
    mind.aim = String::new();
    assert_eq!(prompt(&mind), resting, "back at rest, the prompt did not come back");
}

/// The sentence a person writes is carried **as they wrote it**, and it does not need a dial.
#[test]
fn the_goal_is_a_sentence_and_stands_on_its_own() {
    let mut mind = Mind::resting();
    mind.aim = "내가 회의 중일 때는 말 걸지 마".into();
    let line = mind.line().expect("an aim alone renders nothing");
    assert!(line.contains("내가 회의 중일 때는 말 걸지 마"), "{line}");
    //: No dial was turned, so no dial is claimed.
    for (_, gloss) in DIALS {
        assert!(!line.contains(gloss), "{line}");
    }
}

/// **The counterweight is always there.** Every dial is a reason to talk more, and 31 of 34
/// conversational turns are measured as ones that should produce no action at all.
#[test]
fn whatever_is_set_the_line_ends_by_saying_it_is_not_a_reason_to_speak() {
    let mut dialled = Mind::resting();
    dialled.turn("유머", 9);
    let mut aimed = Mind::resting();
    aimed.aim = "무엇이든".into();
    for mind in [dialled, aimed] {
        let line = mind.line().unwrap();
        assert!(line.contains("할 말이 없는데 말하거나"), "{line}");
        assert!(line.contains("부탁받지 않은 일을 할 이유가 되지는 않습니다"), "{line}");
    }
}

/// A hand-edited file, or a page from a newer version: **a name nobody knows is dropped, not
/// added**, and a number above the top is brought down rather than refused.
#[test]
fn an_unknown_dial_is_ignored_and_an_impossible_number_is_clamped() {
    let mut mind = Mind::resting();
    mind.turn("신중", 5);
    assert!(mind.at_rest(), "an unknown dial got in");
    assert_eq!(mind.of("신중"), 0);

    mind.turn("헌신", 250);
    assert_eq!(mind.of("헌신"), MOST);
    assert!(mind.line().unwrap().contains(&format!("{MOST}/{MOST}")));
}

/// Dials at zero are **absent from the line, not listed as zero** — otherwise every prompt would
/// carry three instructions to aim for nothing.
#[test]
fn a_dial_left_alone_is_not_mentioned() {
    let mut mind = Mind::resting();
    mind.turn("봉사", 4);
    let line = mind.line().unwrap();
    assert!(line.contains("봉사"), "{line}");
    for (korean, gloss) in DIALS {
        if korean != "봉사" {
            assert!(!line.contains(gloss), "{korean} is mentioned at zero: {line}");
        }
    }
    assert!(!line.contains(&format!("0/{MOST}")), "{line}");
}

// ---- kept sentences ----------------------------------------------------------------------------

use argmind_core::mind::{asked_to_keep, worth_offering, Keep};

/// **A goal alone renders exactly as it did**, so every measurement of the dials still describes
/// the shipped prompt; kept sentences follow it under the same label.
#[test]
fn kept_sentences_follow_the_goal_and_the_goal_alone_is_unchanged() {
    let mut mind = Mind::resting();
    mind.aim = "놓친 것만 말해줘".into();
    let alone = mind.line().unwrap();
    assert_eq!(alone, "사용자가 기억해 달라고 한 것: \"놓친 것만 말해줘\" 그렇다고 해서 할 말이 없는데 \
말하거나, 부탁받지 않은 일을 할 이유가 되지는 않습니다.");
    mind.kept = vec!["나를 대표님이라고 불러".into()];
    assert!(mind.line().unwrap().contains("\"놓친 것만 말해줘\" / \"나를 대표님이라고 불러\""));
    let mut only = Mind::resting();
    only.kept = vec!["영어로 대답해".into()];
    assert!(!only.at_rest());
    assert!(only.line().unwrap().starts_with("사용자가 기억해 달라고 한 것: \"영어로 대답해\""));
    only.kept = vec!["  ".into()];
    assert!(only.at_rest() && only.line().is_none(), "a blank kept sentence is rest");
}

#[test]
fn a_request_to_remember_is_read_by_the_program() {
    assert_eq!(asked_to_keep("나는 존댓말이 편해, 기억해줘"),
               Some(Keep::This("나는 존댓말이 편해".into())));
    assert_eq!(asked_to_keep("메모장 말고 VS Code를 쓴다고 기억해 둬."),
               Some(Keep::This("메모장 말고 VS Code를 쓴다".into())));
    assert_eq!(asked_to_keep("기억해: 회의는 화요일"), Some(Keep::This("회의는 화요일".into())));
    assert_eq!(asked_to_keep("회의는 화요일이라고 기억해줘"), Some(Keep::This("회의는 화요일".into())));
    assert_eq!(asked_to_keep("이거 기억해줘"), Some(Keep::Vague));
    //: Questions and sentences that merely mention remembering are not requests.
    assert_eq!(asked_to_keep("내가 뭐라고 했는지 기억해?"), None);
    assert_eq!(asked_to_keep("어제 일 기억나?"), None);
    assert_eq!(asked_to_keep("안녕"), None);
}

#[test]
fn a_correction_is_offered_and_small_talk_is_not() {
    assert!(worth_offering("메모장 말고 VS Code로 열어줘"));
    assert!(worth_offering("앞으로 반말로 해줘"));
    assert!(!worth_offering("고마워"));
    assert!(!worth_offering("이거 말고 뭐 있어?"), "a question is not a preference");
    assert!(!worth_offering("나는 존댓말이 편해, 기억해줘"), "already a request, not an offer");
}
