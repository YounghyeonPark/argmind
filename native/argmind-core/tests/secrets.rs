//! **A password is the person's to type, and `perform` holds that rather than the prompt.**
//!
//! The hand-off from the dots comparison: an agent on a desktop meets password fields, and what it
//! should do there is stop and say so. Two cases, because a guard that refuses everything passes
//! the first one: typing into a secret is refused and counted, and typing anywhere else is not.

use argmind_core::contract::{Eyes, Hands, Speaks, Widget};
use argmind_core::intent::Effect;
use argmind_core::session::Session;

struct Screen;

impl Eyes for Screen {
    fn foreground(&mut self) -> Vec<Widget> {
        Vec::new()
    }
    fn timed_foreground(&mut self) -> (Vec<Widget>, f64) {
        (Vec::new(), 0.0)
    }
}

/// Hands whose focus is, or is not, a password field — and which record what was typed.
struct Keyboard {
    secret: bool,
    typed: Vec<String>,
}

impl Hands for Keyboard {
    fn live(&self) -> bool {
        true
    }
    fn click(&mut self, _widget: &Widget) {}
    fn type_text(&mut self, text: &str) {
        self.typed.push(text.into());
    }
    fn press(&mut self, _key: &str) {}
    fn into_a_secret(&mut self) -> bool {
        self.secret
    }
}

struct Silent;

impl Speaks for Silent {
    fn ask(&mut self, _prompt: &str) -> String {
        String::new()
    }
}

fn typing(text: &str) -> Effect {
    Effect { kind: "type".into(), text: text.into(), because_asked: Some(true),
             ..Default::default() }
}

#[test]
fn typing_into_a_password_field_is_refused_and_counted() {
    let mut s = Session::new(Screen, Keyboard { secret: true, typed: Vec::new() }, Silent);
    let (did, _, refused) = s.perform(&[typing("hunter2")]);
    assert!(did.is_empty());
    assert_eq!(refused.len(), 1);
    assert_eq!(s.secret, 1);
    assert!(s.hands.typed.is_empty(), "nothing reached the keyboard");
    //: Counted apart from the other refusals, which say different things to the person.
    assert_eq!((s.vanished, s.ambiguous, s.chose_too_late), (0, 0, 0));
}

#[test]
fn typing_anywhere_else_is_untouched() {
    let mut s = Session::new(Screen, Keyboard { secret: false, typed: Vec::new() }, Silent);
    let (did, _, refused) = s.perform(&[typing("안녕하세요")]);
    assert_eq!(did.len(), 1);
    assert!(refused.is_empty());
    assert_eq!(s.secret, 0);
    assert_eq!(s.hands.typed, vec!["안녕하세요".to_string()]);
}
