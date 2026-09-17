//! **The blackboard, and the one bit that stops it costing the conversation.**
//!
//! Two things are worth protecting here and only one of them is obvious. A reading has to carry its
//! age, or the model is told *the camera sees an empty chair* when what is true is *the camera saw
//! an empty chair twelve seconds ago*. And a sense has to stand down while somebody is being
//! answered — measured 2026-09-17, without it the answer goes from a median of 1.0 s to 3.2 s and
//! one in ten waits out a whole look at 10.81 s.
//!
//! The second is held by a guard rather than by two stores, and **a guard is exactly the kind of
//! thing that works until an early return**, so most of this file is about the flag clearing itself.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use argmind_serve::senses::{Senses, Turn};

/// Newest wins, nobody queues, and a sense that never reported is simply absent.
#[test]
fn a_sense_keeps_only_its_latest_reading() {
    let senses = Senses::default();
    assert!(senses.of("카메라").is_none());
    assert!(senses.render("").is_none(), "an empty blackboard renders a header over nothing");

    senses.saw("카메라", "빈 의자".into(), 8.1);
    senses.saw("카메라", "사람이 앉아 있음".into(), 7.9);
    assert_eq!(senses.of("카메라").unwrap().what, "사람이 앉아 있음");
    //: **The cost is kept.** A sense nobody can afford should be visible as such rather than
    //: inferred from the agent feeling slow.
    assert_eq!(senses.of("카메라").unwrap().took, 7.9);
    assert_eq!(senses.latest().len(), 1, "the older reading was kept as well as the newer");
}

/// **Every reading says when it was taken**, in the line and not in a field somebody might drop.
#[test]
fn what_goes_into_the_prompt_says_how_old_it_is() {
    let senses = Senses::default();
    senses.saw("카메라", "책상 앞에 사람이 있음".into(), 8.0);
    let block = senses.render("").expect("a sense reported and nothing rendered");
    assert!(block.contains("책상 앞에 사람이 있음"), "{block}");
    //: The age is there as a number of seconds, whatever it rounds to on a fast machine.
    assert!(block.contains("초 전"), "a reading went in without saying when: {block}");
    assert!(block.contains("카메라"), "{block}");
}

/// The screen is rendered elsewhere as a numbered list, so it is skipped here rather than said
/// twice — and **skipping is by name**, so nothing else is caught by it.
#[test]
fn the_sense_the_caller_already_shows_is_left_out() {
    let senses = Senses::default();
    senses.saw("화면", "289 개가 보임".into(), 0.2);
    senses.saw("카메라", "빈 의자".into(), 8.0);

    let block = senses.render("화면").unwrap();
    assert!(!block.contains("289"), "the screen was rendered twice: {block}");
    assert!(block.contains("빈 의자"), "{block}");

    //: And with nothing else to say, skipping the only sense leaves nothing rather than a header.
    let only = Senses::default();
    only.saw("화면", "289 개가 보임".into(), 0.2);
    assert!(only.render("화면").is_none());
}

/// A reading of nothing is not a reading. Otherwise a sense that failed quietly would put an empty
/// line in every prompt for the rest of the session.
#[test]
fn a_blank_reading_is_not_rendered() {
    let senses = Senses::default();
    senses.saw("카메라", "   ".into(), 8.0);
    assert!(senses.render("").is_none());
}

/// **The third rule: while somebody is being answered, a sense stands down.**
#[test]
fn a_turn_in_flight_tells_the_senses_to_stand_down() {
    let turn = Turn::default();
    let carry_on = turn.carry_on();
    assert!(!turn.in_flight(), "nothing has been said yet");
    assert!(carry_on(), "a sense with nobody being answered should carry on");

    let answering = turn.begins();
    assert!(turn.in_flight());
    //: **The same closure**, because a sense holds it across a whole call and asks it repeatedly.
    //: One that read the flag once would abandon nothing.
    assert!(!carry_on(), "a sense was told to carry on during a turn");

    drop(answering);
    assert!(!turn.in_flight());
    assert!(carry_on());
}

/// **The flag clears itself.** The failure this guards is the one that does not look like a
/// failure: an early return leaves it set, every sense stands down for the rest of the session, and
/// the panel simply stops updating with nobody the wiser.
#[test]
fn an_answer_that_returns_early_still_lets_the_senses_back() {
    let turn = Turn::default();

    fn answers_and_gives_up(turn: &Turn) -> &'static str {
        let _answering = turn.begins();
        //: The shape of every early return in `think`: something was missing, so it says so and
        //: leaves.
        return "보고 있는 창이 없습니다";
    }
    assert_eq!(answers_and_gives_up(&turn), "보고 있는 창이 없습니다");
    assert!(!turn.in_flight(), "an early return left the senses standing down");

    //: And a panic, which is the other way a function stops early.
    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _answering = turn.begins();
        panic!("a turn went wrong");
    }));
    assert!(caught.is_err());
    assert!(!turn.in_flight(), "a panic left the senses standing down");
}

/// It is shared, not copied: a sense on another thread sees a turn begin here.
#[test]
fn every_thread_sees_the_same_bit() {
    let turn = Turn::default();
    let carry_on = turn.carry_on();
    let saw_it = Arc::new(AtomicBool::new(false));

    let answering = turn.begins();
    let watcher = {
        let saw_it = Arc::clone(&saw_it);
        std::thread::spawn(move || {
            for _ in 0..200 {
                if !carry_on() {
                    saw_it.store(true, Ordering::SeqCst);
                    return;
                }
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        })
    };
    let _ = watcher.join();
    drop(answering);
    assert!(saw_it.load(Ordering::SeqCst), "a sense on another thread never saw the turn");
}
