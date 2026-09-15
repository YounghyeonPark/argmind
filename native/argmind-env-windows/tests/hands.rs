//! The hands, with nothing sent.
//!
//! **Every test here leaves the hands dry**, so the suite still needs no display and touches
//! nothing: an unarmed `WindowsHands` records what it would have done and returns before the cursor
//! moves or an event is queued. What is checked is the part that decides *whether* to act, which is
//! the part the standing rule lives in — and the part a port gets wrong quietly, because a hand
//! that clicks slightly too often looks exactly like a hand that works.

use argmind_core::contract::{Hands, Widget};
use argmind_env_windows::hands::{button_flags, check_delivery, may_touch, typing_units,
                                 virtual_key, Did, KEY_NAMES};
use argmind_env_windows::{Reach, WindowsHands};

fn somewhere() -> Widget {
    Widget::new("Copy", "button", (100, 100, 40, 20), "Character Map")
}

/// **The rule, as a decision.** *Never click anything the harness did not launch itself.*
#[test]
fn a_harness_may_only_touch_the_process_it_started() {
    assert!(may_touch(Reach::OnlyProcess(42), Some(42)));
    assert!(!may_touch(Reach::OnlyProcess(42), Some(43)));
    //: **Nothing resolvable at the point is a refusal too**, not a pass. A click into a point whose
    //: window cannot be identified is exactly the case the rule exists for — and defaulting it the
    //: other way is the single edit that would turn this guard off without looking like it.
    assert!(!may_touch(Reach::OnlyProcess(42), None));
    //: The application's hands touch anything; the person's confirm button is what makes that safe.
    assert!(may_touch(Reach::Anything, Some(43)));
    assert!(may_touch(Reach::Anything, None));
}

/// **A harness that put its own window in the way may act into it, and into nothing else.**
///
/// `obstruction` needs this: it covers the target with a window of its own and then asks whether
/// the click still reaches past. The widening is exactly two processes, both of which this program
/// is responsible for — the one it launched and itself.
#[test]
fn a_harness_may_also_touch_the_cover_it_made_itself() {
    let reach = Reach::OursAnd { ours: 7, theirs: 42 };
    assert!(may_touch(reach, Some(42)), "the application it launched");
    assert!(may_touch(reach, Some(7)), "the cover it made itself");
    //: **And nothing else.** A third process at that point is a stranger's window, and this is
    //: still the rule rather than an exception to it.
    assert!(!may_touch(reach, Some(43)));
    assert!(!may_touch(reach, None));
    //: The two ids are separate, so a mistake that made them the same could not pass unnoticed as
    //: a widening: with `ours == theirs` this is exactly `OnlyProcess`.
    let same = Reach::OursAnd { ours: 42, theirs: 42 };
    assert!(may_touch(same, Some(42)));
    assert!(!may_touch(same, Some(7)));
}

/// An unarmed hand does nothing, and **that is the default**.
#[test]
fn hands_are_dry_until_they_are_armed() {
    let mut hands = WindowsHands::unrestricted();
    assert!(!hands.live(), "hands that are built are not hands that are armed");
    assert_eq!(hands.click_button(&somewhere(), "left"), Did::Sent { live: false });
    assert_eq!(hands.press_key("enter"), Did::Sent { live: false });
    assert_eq!(hands.type_string("hello"), Did::Sent { live: false });
    //: Counted anyway, because a dry run has to be able to say what a live one would have done.
    assert_eq!(hands.clicked, 1);
    assert_eq!(hands.done.len(), 3);
    //: `arm()` is separate from construction: a flag that both builds and arms is a flag somebody
    //: sets while meaning only to build.
    assert!(WindowsHands::unrestricted().arm().live());
}

/// A key is named or it is refused — **guessing a keystroke is how an agent sends Alt+F4.**
#[test]
fn an_unnamed_key_is_refused_with_the_list() {
    let mut hands = WindowsHands::unrestricted();
    match hands.press_key("f4") {
        Did::Refused(why) => {
            assert!(why.contains("Alt+F4"), "{why}");
            assert!(why.contains("enter"), "the refusal has to say what IS allowed: {why}");
        }
        other => panic!("f4 was not refused: {other:?}"),
    }
    assert_eq!(hands.refused, 1);
    //: Every name the refusal advertises actually resolves, or the message is a lie.
    for name in KEY_NAMES {
        assert!(virtual_key(name).is_some(), "{name} is advertised and unknown");
    }
    //: And the two spellings of one key are one key.
    assert_eq!(virtual_key("esc"), virtual_key("escape"));
    assert_eq!(virtual_key("ENTER"), virtual_key("enter"));
    assert!(virtual_key("").is_none());
    assert!(virtual_key("a").is_none(), "a letter is typed, not pressed");
}

#[test]
fn an_unknown_button_is_refused_rather_than_treated_as_a_left_click() {
    let mut hands = WindowsHands::unrestricted();
    assert!(matches!(hands.click_button(&somewhere(), "back"), Did::Refused(_)));
    assert_eq!(hands.clicked, 0, "a refused button must not count as a click");
    for b in ["left", "right", "middle"] {
        assert!(button_flags(b).is_some());
    }
    //: The three are three, and no two share a flag pair.
    let all: Vec<_> = ["left", "right", "middle"].iter().map(|b| button_flags(b).unwrap()).collect();
    for i in 0..all.len() {
        for j in i + 1..all.len() {
            assert_ne!(all[i].0 .0, all[j].0 .0);
        }
    }
}

/// A bound hand refuses a point it does not own, and **the refusal is counted, not silent.**
///
/// On a machine with no desktop the owner lookup answers nothing, which is itself a refusal — so
/// this asserts the arithmetic rather than a particular desktop.
#[test]
fn a_click_that_would_land_elsewhere_is_counted_as_elsewhere() {
    //: A process id that cannot own the window under any point: 0 is the idle process and owns no
    //: window, so every lookup disagrees with it.
    let mut hands = WindowsHands::only_within(0);
    let did = hands.click_button(&somewhere(), "left");
    assert!(matches!(did, Did::Elsewhere { .. }), "{did:?}");
    assert_eq!(hands.elsewhere, 1);
    assert_eq!(hands.clicked, 0, "a click that was never sent is not a click");
    //: **A harness that silently skipped half its clicks would report a clean number**, so the
    //: three outcomes stay apart in the record.
    assert_eq!(hands.done.len(), 1);
}

/// Typing goes out in UTF-16 code units, which is what `KEYEVENTF_UNICODE` takes.
#[test]
fn text_is_measured_in_the_units_the_api_sends() {
    //: Two Korean characters are two units; an emoji outside the basic plane is **two units for one
    //: character**, and sending the code point instead would send something else entirely.
    assert_eq!("안녕".encode_utf16().count(), 2);
    assert_eq!("😀".encode_utf16().count(), 2);
    assert_eq!("😀".chars().count(), 1);
    assert_eq!(typing_units("안녕").len(), 2);
    assert_eq!(typing_units("😀").len(), 2, "one character, two units");
    assert_eq!(typing_units("ab"), vec![0x61, 0x62]);
    assert_eq!(typing_units(""), Vec::<u16>::new());
    let mut hands = WindowsHands::unrestricted();
    assert_eq!(hands.type_string("안녕 😀"), Did::Sent { live: false });
}

/// **Half a click is a button held down**, so a partial delivery is a failure.
///
/// The call that sends is unreachable from a suite that must not send, so the decision it makes is
/// out here where it can be checked and the `unsafe` line has nothing left to get wrong.
#[test]
fn a_partial_delivery_is_a_failure() {
    assert!(check_delivery(2, 2).is_ok());
    assert!(check_delivery(0, 0).is_ok());
    let why = check_delivery(1, 2).unwrap_err();
    assert!(why.contains("1 of 2"), "{why}");
    assert!(why.contains("button"), "the message has to say what is left behind: {why}");
    //: More than asked for is wrong in the other direction and is not silently fine.
    assert!(check_delivery(3, 2).is_err());
}

/// **The pattern path is off unless a harness that is measuring it turns it on.**
///
/// `obstruction` found `3 of 3` with patterns and `3 of 3` without, so nothing has earned them a
/// place in the shipped click. The two harnesses that exist to measure them enable them; the
/// default does not, and that default is the finding acted on.
#[test]
fn control_patterns_are_off_until_a_harness_asks_for_them() {
    assert!(!WindowsHands::unrestricted().patterns);
    assert!(WindowsHands::unrestricted().with_patterns().patterns);
    //: Enabling them does not arm them, and arming does not enable them. Two flags, two meanings.
    assert!(!WindowsHands::unrestricted().with_patterns().live());
    assert!(!WindowsHands::unrestricted().arm().patterns);
}

/// The two paths are counted apart, and **a click that a guard stopped is neither.**
#[test]
fn the_pattern_arm_and_the_click_arm_are_separate_tallies() {
    //: Bound to a process that owns no window, so the guard refuses before either path runs — the
    //: point being that a refusal must not be booked to one of the arms.
    let mut hands = WindowsHands::only_within(0).with_patterns();
    assert!(matches!(hands.click_button(&somewhere(), "left"), Did::Elsewhere { .. }));
    assert_eq!((hands.by_pattern, hands.by_click, hands.elsewhere), (0, 0, 1));

    //: Unbound and dry: no window under the point resolves to the widget's name and role, so the
    //: pattern lookup finds nothing and the click arm takes it — with a reason recorded, because
    //: "fell back" without one is the same as no answer.
    let mut hands = WindowsHands::unrestricted().with_patterns();
    assert_eq!(hands.click_button(&somewhere(), "left"), Did::Sent { live: false });
    assert_eq!((hands.by_pattern, hands.by_click), (0, 1));
    assert!(!hands.why.is_empty(), "a fallback with no reason recorded");

    //: With patterns off the reason stays empty, so a reader cannot mistake "not tried" for
    //: "tried and failed".
    let mut hands = WindowsHands::unrestricted();
    hands.click_button(&somewhere(), "left");
    assert!(hands.why.is_empty());
    assert_eq!((hands.by_pattern, hands.by_click), (0, 1));
}
