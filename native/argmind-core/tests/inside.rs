//! **A point inside a place the list cannot see into** — the one shape in which a model's
//! coordinate is allowed, since 2026-11-18, and what `perform` does with it.
//!
//! The owner chose *the picture only where the list is blind, always with a press* after
//! `--bin looking` found a vision model reaching 5–6 of 6 things the list cannot see and being less
//! exact than the list where the list has the thing by name. What is held here is the core's half:
//!
//! * the container is still found **by name and role, now** — a container that moved takes the
//!   point with it, one that is gone or doubled is refused exactly as a named click is;
//! * the point is placed in the container's **fresh** rectangle and cannot leave it;
//! * the hands are asked for a point, never for the container's own press — and hands that have
//!   not said they can press a point press nothing (`click_inside` is `false` by default);
//! * nothing claims the screen moved: the list cannot see in there, which is the whole reason.
//!
//! The application's half — that such an effect is made in one place and always waits for a
//! person — is read off the source by `argmind-rules`.

use argmind_core::contract::{Eyes, Hands, Speaks, Widget};
use argmind_core::intent::{Among, Effect, Inside};
use argmind_core::session::{describe, Session, PAST};

struct Screen(Vec<Widget>);

impl Eyes for Screen {
    fn foreground(&mut self) -> Vec<Widget> {
        self.0.clone()
    }
    fn timed_foreground(&mut self) -> (Vec<Widget>, f64) {
        (self.0.clone(), 0.0)
    }
}

/// Hands that can press a point, and record both kinds of press apart.
#[derive(Default)]
struct Pointing {
    clicked: Vec<Widget>,
    pointed: Vec<(Widget, (i64, i64))>,
}

impl Hands for Pointing {
    fn live(&self) -> bool {
        true
    }
    fn click(&mut self, widget: &Widget) {
        self.clicked.push(widget.clone());
    }
    fn click_inside(&mut self, within: &Widget, at: (i64, i64)) -> bool {
        self.pointed.push((within.clone(), at));
        true
    }
    fn type_text(&mut self, _text: &str) {}
    fn press(&mut self, _key: &str) {}
}

/// Hands that never said they could press a point: the trait's default.
#[derive(Default)]
struct Plain(Vec<Widget>);

impl Hands for Plain {
    fn live(&self) -> bool {
        true
    }
    fn click(&mut self, widget: &Widget) {
        self.0.push(widget.clone());
    }
    fn type_text(&mut self, _text: &str) {}
    fn press(&mut self, _key: &str) {}
}

struct Silent;

impl Speaks for Silent {
    fn ask(&mut self, _prompt: &str) -> String {
        String::new()
    }
}

fn grid(x: i64, y: i64) -> Widget {
    Widget::new("문자 격자", "pane", (x, y, 400, 200), "문자표")
}

fn point_in_grid(x: u16, y: u16) -> Effect {
    Effect { kind: "click".into(), target: "문자 격자".into(), role: "pane".into(), text: "★".into(),
             because_asked: Some(true), which: None, inside: Some(Inside { x, y }) }
}

fn session<H: Hands>(screen: Vec<Widget>, hands: H) -> Session<Screen, H, Silent> {
    let mut s = Session::new(Screen(screen.clone()), hands, Silent);
    s.seen = screen;
    s
}

#[test]
fn the_point_is_placed_in_the_containers_rectangle_as_it_is_now() {
    //: Offered while the grid stood at (100, 100); by the press it has moved to (300, 150).
    let mut s = session(vec![grid(300, 150)], Pointing::default());
    s.seen = vec![grid(100, 100)];
    let (did, _, refused) = s.perform(&[point_in_grid(500, 250)]);
    assert!(refused.is_empty(), "{refused:?}");
    assert_eq!(did.len(), 1);
    assert_eq!(s.hands.pointed, vec![(grid(300, 150), (300 + 200, 150 + 50))],
               "the point follows the container rather than staying where it was seen");
    assert!(s.hands.clicked.is_empty(), "the container's own press is not a point inside it");
}

#[test]
fn nothing_claims_the_screen_moved() {
    let mut s = session(vec![grid(0, 0)], Pointing::default());
    let (did, _, _) = s.perform(&[point_in_grid(10, 10)]);
    assert_eq!(did[0].confirmed, None, "the list cannot see inside, so it cannot confirm");
}

#[test]
fn a_point_cannot_leave_the_container() {
    let mut s = session(vec![grid(0, 0)], Pointing::default());
    s.perform(&[point_in_grid(1000, 1000)]);
    s.perform(&[point_in_grid(65535, 65535)]);
    for (_, (x, y)) in &s.hands.pointed {
        assert!((0..400).contains(x) && (0..200).contains(y), "({x}, {y}) is outside the grid");
    }
}

#[test]
fn a_container_that_is_gone_is_refused_like_any_click() {
    let mut s = session(vec![Widget::new("다른 것", "pane", (0, 0, 400, 200), "문자표")],
                        Pointing::default());
    s.seen = vec![grid(0, 0)];
    let (did, _, refused) = s.perform(&[point_in_grid(500, 500)]);
    assert!(did.is_empty() && refused.len() == 1);
    assert_eq!(s.vanished, 1);
    assert!(s.hands.pointed.is_empty());
}

#[test]
fn two_containers_of_one_name_are_refused_unless_a_person_said_which() {
    let both = vec![grid(0, 0), grid(0, 300)];
    let mut s = session(both.clone(), Pointing::default());
    let (_, _, refused) = s.perform(&[point_in_grid(500, 500)]);
    assert_eq!((refused.len(), s.ambiguous), (1, 1));
    assert!(s.hands.pointed.is_empty());
    //: Answered: the second grid, chosen among two — and the point is in *that* one.
    let mut chosen = point_in_grid(500, 500);
    chosen.which = Some(Among { index: 1, of: 2 });
    let (did, _, _) = s.perform(&[chosen]);
    assert_eq!(did.len(), 1);
    assert_eq!(s.hands.pointed, vec![(grid(0, 300), (200, 400))]);
}

#[test]
fn hands_that_never_said_they_can_point_press_nothing() {
    let mut s = session(vec![grid(0, 0)], Plain::default());
    let (did, _, refused) = s.perform(&[point_in_grid(500, 500)]);
    assert!(did.is_empty(), "a refusal, not a press");
    assert_eq!(refused.len(), 1);
    assert_eq!(s.cannot_point, 1);
    assert!(s.hands.0.is_empty(), "and certainly not the container's own press as a fallback");
}

#[test]
fn a_refused_point_stops_the_steps_after_it() {
    let mut s = session(vec![grid(0, 0)], Plain::default());
    let typing = Effect { kind: "type".into(), text: "안녕".into(), because_asked: Some(true),
                          ..Default::default() };
    let (_, _, refused) = s.perform(&[point_in_grid(500, 500), typing]);
    assert_eq!(refused.len(), 2);
    assert_eq!(s.unattempted, 1);
}

#[test]
fn an_effect_without_a_point_is_pressed_as_it_always_was() {
    //: The control: the same hands, the same container, no point — the old path, untouched.
    let mut s = session(vec![grid(0, 0)], Pointing::default());
    let mut plain = point_in_grid(0, 0);
    plain.inside = None;
    let (did, _, _) = s.perform(&[plain]);
    assert_eq!(did.len(), 1);
    assert_eq!(s.hands.clicked, vec![grid(0, 0)]);
    assert!(s.hands.pointed.is_empty());
}

#[test]
fn the_history_says_where_and_what_and_never_the_point() {
    let e = point_in_grid(512, 256);
    let said = describe(&e, Some(&grid(0, 0)), PAST);
    assert_eq!(said, "clicked inside the pane \"문자 격자\", where the picture showed \"★\"");
    assert!(!said.contains("512") && !said.contains("256"));
}
