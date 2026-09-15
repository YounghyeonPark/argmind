//! The permission model, as arithmetic.
//!
//! **Nothing here performs anything — that is the point.** These check the three ways a
//! confirmation can be wrong: it names a proposal that has gone, it names one whose number was
//! reused, or the button is pressed twice.

use argmind_core::intent::Effect;
use argmind_serve::proposals::{Answered, Proposals};

fn click(target: &str) -> Effect {
    Effect { kind: "click".into(), target: target.into(), role: "button".into(),
             text: String::new(), because_asked: Some(true) }
}

fn gone(a: &Answered) -> bool {
    matches!(a, Answered::Gone)
}

#[test]
fn an_offer_waits_and_only_a_confirmation_releases_it() {
    let mut p = Proposals::default();
    let a = p.offer(click("Copy"), "클릭  button Copy".into(), String::new());
    assert_eq!(p.waiting().len(), 1);
    //: Filing an offer does not act, and reading the queue does not act.
    assert_eq!(p.waiting()[0].effect.target, "Copy");

    match p.confirm(a) {
        Answered::Run(got) => assert_eq!(got.effect.target, "Copy"),
        other => panic!("{other:?}"),
    }
    //: And it is gone: the queue is the record of what is still pressable.
    assert!(p.waiting().is_empty());
}

/// **Pressing twice must not act twice.** A double click on a button, or a reload of a page that
/// still shows it, is ordinary.
#[test]
fn a_proposal_can_only_be_answered_once() {
    let mut p = Proposals::default();
    let a = p.offer(click("Copy"), "a".into(), String::new());
    assert!(matches!(p.confirm(a), Answered::Run(_)));
    assert!(gone(&p.confirm(a)));
    assert!(gone(&p.decline(a)));
    //: Never existing at all is the same answer — not an error, and above all not an action on
    //: whatever happens to be pending now.
    assert!(gone(&p.confirm(9999)));
}

/// **Numbers are never reused**, and this is the case that makes it matter.
#[test]
fn a_stale_button_cannot_confirm_a_later_proposal() {
    let mut p = Proposals::default();
    let first = p.offer(click("Copy"), "copy".into(), String::new());
    p.clear();
    let second = p.offer(click("Delete"), "delete".into(), String::new());
    assert_ne!(first, second, "the number came back and now names a different action");
    //: A page still showing the old button confirms nothing…
    assert!(gone(&p.confirm(first)));
    //: …and the new one is untouched by that.
    match p.confirm(second) {
        Answered::Run(got) => assert_eq!(got.effect.target, "Delete"),
        other => panic!("{other:?}"),
    }
}

/// Saying no is **a thing the person said**, so the effect comes back to be told to the agent.
#[test]
fn declining_returns_the_effect_so_the_conversation_hears_about_it() {
    let mut p = Proposals::default();
    let a = p.offer(click("Close"), "close".into(), String::new());
    match p.decline(a) {
        Answered::TurnedDown(got) => assert_eq!(got.effect.target, "Close"),
        other => panic!("{other:?}"),
    }
    assert!(p.waiting().is_empty());
}

/// A new turn does not leave the last turn's buttons alive.
#[test]
fn clearing_hands_back_what_was_still_waiting() {
    let mut p = Proposals::default();
    p.offer(click("A"), "a".into(), String::new());
    p.offer(click("B"), "b".into(), String::new());
    let dropped = p.clear();
    //: **Handed back, not discarded**: two offers the person never answered are two things the
    //: next turn should be told were withdrawn.
    assert_eq!(dropped.len(), 2);
    assert_eq!(dropped[0].effect.target, "A");
    assert!(p.waiting().is_empty());
}

/// The hint is shown and **never gates**.
#[test]
fn a_disagreeing_hint_does_not_stop_the_button() {
    let mut p = Proposals::default();
    let a = p.offer(click("Close"), "close".into(), "그 지점에는 application".into());
    assert!(!p.waiting()[0].hint.is_empty());
    //: A widget's own centre resolves back to itself about seven times in ten, and a caption
    //: button resolving to `application` is the hit test disagreeing with a pixel that is correct.
    assert!(matches!(p.confirm(a), Answered::Run(_)));
}

/// Two offers from one turn are two buttons, each answerable on its own.
#[test]
fn a_turn_that_offers_three_things_offers_three_buttons() {
    let mut p = Proposals::default();
    let a = p.offer(click("A"), "a".into(), String::new());
    let b = p.offer(click("B"), "b".into(), String::new());
    let c = p.offer(click("C"), "c".into(), String::new());
    assert_eq!(p.waiting().len(), 3);
    //: Declining the middle one leaves the other two exactly where they were — the person is
    //: answering actions, not a batch.
    assert!(matches!(p.decline(b), Answered::TurnedDown(_)));
    assert_eq!(p.waiting().len(), 2);
    assert!(matches!(p.confirm(c), Answered::Run(_)));
    assert!(matches!(p.confirm(a), Answered::Run(_)));
}
