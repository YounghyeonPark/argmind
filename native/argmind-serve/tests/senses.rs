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

// ---- what the page is actually sent ------------------------------------------------------------

/// **Every hand-written JSON string in this program is one nobody parses until a page does.**
///
/// `note_json` shipped for two commits with a real newline inside a key name — `\"` where a Rust
/// line-continuation `\` was meant — so `{"tried":"","\n             held":1}` went out and the
/// page's `JSON.parse` threw on every note. The 지금 생각하고 있는 것 panel was simply dead, and
/// nothing in the suite noticed because nothing read what this function produces.
#[test]
fn the_note_the_page_receives_is_json_a_page_can_read() {
    use argmind_core::thinking::{Note, Worth};
    use argmind_serve::app::note_json;

    let mut note = Note::blank();
    let parsed = |n: &Note| -> serde_json::Value {
        let text = note_json(n);
        serde_json::from_str(&text).unwrap_or_else(|e| panic!("the page cannot read {text:?}: {e}"))
    };

    //: A blank note, which is what a page opening mid-session usually gets.
    let d = parsed(&note);
    assert_eq!(d["believes"], "");
    assert_eq!(d["worth"], "");
    assert_eq!(d["held"], 0);

    //: And a full one. **The awkward characters are the point** — a window title with a quote and a
    //: backslash in it is an ordinary Tuesday, and `escape` is the only thing standing between that
    //: and a broken panel.
    note.revise("그가 \"저장\" 대화상자를 보고 있다 \\ C:\\dev",
                argmind_core::thinking::Judged::This(
                    Worth { name: "저장 \"안 함\"".into(), role: "button".into() }));
    note.record("clicked the button \"Save\"\nand waited");
    let d = parsed(&note);
    assert!(d["believes"].as_str().unwrap().contains("C:\\dev"));
    assert!(d["worth"].as_str().unwrap().contains("저장"));
    assert_eq!(d["stuck"], 1);

    //: **Every key the page reads is present and spelled the way it reads it.** The bug was a key
    //: name, not a value, so a test that only checked the values would have passed.
    for key in ["believes", "worth", "tried", "held", "stuck", "revisions"] {
        assert!(d.get(key).is_some(), "the page asks for {key} and it is not in {d}");
    }
}

/// **The same check over every payload the page parses.**
///
/// The note was the one that broke, but nothing about the mistake was particular to it: this crate
/// writes JSON by hand everywhere, and `escape` plus a format string is all that stands between an
/// awkward window title and a panel that silently stops drawing. One bug in the class is reason to
/// cover the class.
///
/// `devices_json` is not here — it asks the platform for real hardware, and this file runs with
/// none.
#[test]
fn every_payload_the_page_parses_is_json_a_page_can_read() {
    use argmind_core::mind::Mind;
    use argmind_serve::app::{asking_json, decisions_json, mind_json, needs_as_json};

    let reads = |what: &str, text: String| {
        serde_json::from_str::<serde_json::Value>(&text)
            .unwrap_or_else(|e| panic!("the page cannot read {what}: {e}\n  {text}"))
    };

    //: **The dials, with the awkward characters a person can actually type into the goal box** —
    //: and into a kept sentence, which is what reaches the line now the goal does not.
    let mut mind = Mind::resting();
    mind.turn("재치", 7);
    mind.aim = "이렇게: \"놓친 것\"만 — C:\\dev\\argmind 볼 때".into();
    mind.kept = vec!["그대로: \"남긴 것\" — C:\\dev".into()];
    let d = reads("mind_json", mind_json(&mind));
    assert!(d["aim"].as_str().unwrap().contains("놓친 것"));
    assert!(d["line"].as_str().unwrap().contains("남긴 것") && !d["line"].as_str().unwrap().contains("놓친 것"));
    assert_eq!(d["dials"].as_array().unwrap().len(), argmind_core::mind::DIALS.len());

    reads("mind_json at rest", mind_json(&Mind::resting()));
    //: Reads the real decisions file, which is a read and nothing else.
    reads("decisions_json", decisions_json());
    reads("asking_json", asking_json("microphone"));
    //: An unknown key still has to produce something a page can read rather than half an object.
    reads("asking_json for a key nobody knows", asking_json("무엇"));
    reads("needs_as_json", needs_as_json(&argmind_serve::needs::every("qwen3:8b"),
                                         &std::collections::BTreeMap::new()));
}

// ---- the camera, and the one thing it is allowed to stop -----------------------------------------

/// **A sense that has stopped is forgotten, not left with a growing age.**
///
/// A blank reading is not rendered but it stays in the map; that is right for a sense that saw
/// nothing and wrong for one that is no longer looking. *마지막 움직임 3초 전* about a camera that
/// has been off for an hour is a lie with a timestamp on it, and the gate in `notice` reads exactly
/// that kind of sentence.
#[test]
fn a_sense_that_stopped_is_forgotten_rather_than_left_stale() {
    let senses = Senses::default();
    senses.saw("카메라", "마지막 움직임 3초 전".into(), 3.0);
    assert!(senses.of("카메라").is_some());
    assert!(senses.render("화면").unwrap().contains("마지막 움직임"));

    senses.forget("카메라");
    assert!(senses.of("카메라").is_none(), "the camera is still on the blackboard");
    assert!(senses.render("화면").is_none(), "a camera that is off is still in the prompt");

    //: Forgetting one leaves the others. The screen is written on every look and must not vanish
    //: because a camera was switched off.
    senses.saw("화면", "12 개가 보임".into(), 0.0);
    senses.saw("카메라", "마지막 움직임 1초 전".into(), 3.0);
    senses.forget("카메라");
    assert!(senses.of("화면").is_some());
    //: And forgetting what was never there is not an error.
    senses.forget("없는감각");
}

/// **The absence of a camera never makes argmind quiet.**
///
/// `somebody_about` is the gate on the one path that speaks without being spoken to, and the rule
/// it has to keep is that *no evidence* is not *nobody*. A camera switched off, refused, or failing
/// to open must leave the watch exactly as it was — otherwise not owning a webcam would be a reason
/// for the agent to stop working.
#[test]
fn no_camera_is_not_an_empty_room() {
    let source = include_str!("../src/app.rs");
    let from = source.find("fn somebody_about(").expect("the gate is gone");
    let body = &source[from..from + 900];
    //: Every path out of it that is not a real reading has to be `None`.
    assert!(body.contains("if !desk.eyeing.load"), "the gate does not check the camera is on");
    assert!(body.matches("return None").count() >= 2, "{body}");

    //: And the caller acts on `Some(false)` only — never on `None`, and never on the absence of a
    //: value by way of `unwrap_or`.
    let at = source.find("somebody_about(desk) == Some(false)")
        .expect("the watch does not gate on an explicit `nobody`");
    assert!(at > from || at < from, "found at {at}");
    assert!(!source.contains("somebody_about(desk).unwrap_or"),
            "the gate's `no evidence` was turned into a decision by unwrap_or");
}
