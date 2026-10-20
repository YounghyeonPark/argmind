//! **A sequence stops at the step that did not land, and typing asks where it is going.**
//!
//! The owner's session, 2026-10-21: the weather asked for on a login page, and a click, *서울* and
//! Enter ran one after another. Each test here is one way a step can fail, and the steps after it
//! must not reach the hands; and **one control where nothing fails and everything runs**, because a
//! rule that stopped every sequence would pass every other test in this file.

use argmind_core::contract::{Eyes, Hands, Speaks, Widget};
use argmind_core::intent::Effect;
use argmind_core::session::Session;

struct Screen(Vec<Widget>);

impl Eyes for Screen {
    fn foreground(&mut self) -> Vec<Widget> {
        self.0.clone()
    }
    fn timed_foreground(&mut self) -> (Vec<Widget>, f64) {
        (self.0.clone(), 0.0)
    }
}

/// Hands that record, with a field that may or may not take text and clicks that may not land.
#[derive(Default)]
struct Recorded {
    field: Option<bool>,
    lands: Option<bool>,
    log: Vec<String>,
}

impl Hands for Recorded {
    fn live(&self) -> bool {
        true
    }
    fn click(&mut self, w: &Widget) {
        self.log.push(format!("click {}", w.name));
    }
    fn type_text(&mut self, text: &str) {
        self.log.push(format!("type {text}"));
    }
    fn press(&mut self, key: &str) {
        self.log.push(format!("key {key}"));
    }
    fn typing_lands(&mut self) -> Option<bool> {
        self.field
    }
    fn landed(&self) -> Option<bool> {
        self.lands
    }
}

struct Silent;

impl Speaks for Silent {
    fn ask(&mut self, _: &str) -> String {
        String::new()
    }
}

fn box_on_screen() -> Vec<Widget> {
    vec![Widget::new("검색", "textbox", (0, 0, 100, 20), "페이지")]
}

fn steps(click: &str) -> Vec<Effect> {
    let e = |kind: &str, target: &str, role: &str, text: &str| Effect {
        kind: kind.into(), target: target.into(), role: role.into(), text: text.into(),
        because_asked: Some(true), ..Default::default()
    };
    vec![e("click", click, "textbox", ""), e("type", "", "", "서울"), e("key", "", "", "enter")]
}

fn run(hands: Recorded, click: &str) -> (Session<Screen, Recorded, Silent>, usize, usize) {
    let mut s = Session::new(Screen(box_on_screen()), hands, Silent);
    let (did, _, refused) = s.perform(&steps(click));
    let (d, r) = (did.len(), refused.len());
    (s, d, r)
}

#[test]
fn everything_runs_when_nothing_fails() {
    let (s, did, refused) = run(Recorded { field: Some(true), lands: Some(true), ..Default::default() },
                                "검색");
    assert_eq!((did, refused), (3, 0));
    assert_eq!(s.hands.log, ["click 검색", "type 서울", "key enter"]);
}

#[test]
fn a_click_on_something_gone_stops_the_typing_after_it() {
    let (s, did, refused) = run(Recorded { field: Some(true), ..Default::default() }, "없는 것");
    assert_eq!((did, refused), (0, 3));
    assert!(s.hands.log.is_empty(), "nothing typed, nothing pressed: {:?}", s.hands.log);
    assert_eq!((s.vanished, s.unattempted), (1, 2));
}

#[test]
fn no_field_means_no_typing_and_no_enter() {
    let (s, did, refused) = run(Recorded { field: Some(false), ..Default::default() }, "검색");
    assert_eq!((did, refused), (1, 2));
    assert_eq!(s.hands.log, ["click 검색"]);
    assert_eq!((s.nowhere_to_type, s.unattempted), (1, 1));
}

#[test]
fn a_click_the_hands_say_did_not_land_is_not_done_and_stops_the_rest() {
    let (s, did, refused) = run(Recorded { field: Some(true), lands: Some(false), ..Default::default() },
                                "검색");
    assert_eq!((did, refused), (0, 3));
    assert_eq!(s.hands.log, ["click 검색"], "tried once, nothing after it");
    assert_eq!((s.not_landed, s.unattempted), (1, 2));
}

#[test]
fn hands_that_cannot_tell_change_nothing() {
    let (s, did, refused) = run(Recorded::default(), "검색");
    assert_eq!((did, refused), (3, 0));
    assert_eq!(s.hands.log.len(), 3);
}
