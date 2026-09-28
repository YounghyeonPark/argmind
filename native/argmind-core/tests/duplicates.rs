//! **A name is not a key, and since 2026-09-28 the lookup says so instead of guessing.**
//!
//! The measured failure, 2026-09-16: Character Map's advanced view has more than one button named
//! `Open`; asked for the font list the agent opened the character set list, and reported success.
//! `clicking` scored 2 of 3 and the miss was this. The old comment on `Session::find` recorded the
//! cause precisely — *the anchor is the first match in the previous reading, so with duplicates the
//! anchor is itself arbitrary and nearest-to-it is arbitrary too* — and then went on guessing.
//!
//! **This is the negative result, protected by a test.** The rule is the one that comment contained:
//! the tie-break means something when the previous reading held **exactly one** of them, and
//! nothing otherwise. Each case below is written so that a change which reinstates the guess fails
//! here rather than in a run somebody has to notice.

use argmind_core::contract::{Eyes, Hands, Speaks, Widget};
use argmind_core::intent::{Among, Effect};
use argmind_core::session::{Found, Session};

/// A screen, handed out once per read and then repeated — `find` reads once, `perform` twice.
struct Screen(Vec<Widget>);

impl Eyes for Screen {
    fn foreground(&mut self) -> Vec<Widget> {
        self.0.clone()
    }
    fn timed_foreground(&mut self) -> (Vec<Widget>, f64) {
        (self.0.clone(), 0.0)
    }
}

/// Hands that record. **`live` is true** so a refusal cannot be mistaken for a dry run.
#[derive(Default)]
struct Pressed(Vec<Widget>);

impl Hands for Pressed {
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

fn at(name: &str, role: &str, x: i64, y: i64) -> Widget {
    Widget { name: name.into(), role: role.into(), rect: (x, y, x + 60, y + 20),
             window: "Character Map".into() }
}

fn looking(screen: Vec<Widget>, before: Vec<Widget>) -> Session<Screen, Pressed, Silent> {
    let mut s = Session::new(Screen(screen), Pressed::default(), Silent);
    s.seen = before;
    s.may_act_unprompted = true;
    s
}

/// **The shape that cost the goal.** Two buttons called `Open`, both already on screen when the
/// question was asked, so nothing distinguishes them and the old code pressed one anyway.
fn character_map() -> Vec<Widget> {
    vec![
        at("Font :", "text", 10, 10),
        at("Font :", "combo box", 80, 10),
        at("Open", "button", 150, 10),
        at("Open", "button", 150, 200),
    ]
}

#[test]
fn two_buttons_of_one_name_are_several_rather_than_one_of_them() {
    let screen = character_map();
    let mut s = looking(screen.clone(), screen);
    match s.find("Open", "button") {
        Found::Several(same) => assert_eq!(same.len(), 2),
        other => panic!("a name that matches twice must not resolve: {other:?}"),
    }
}

#[test]
fn the_click_is_refused_and_counted_apart_from_a_vanishing() {
    let screen = character_map();
    let mut s = looking(screen.clone(), screen);
    let click = Effect { kind: "click".into(), target: "Open".into(), role: "button".into(),
                         text: String::new(), because_asked: Some(true) , which: None};
    let (did, _proposed, refused) = s.perform(std::slice::from_ref(&click));
    assert!(did.is_empty(), "a click with two candidates must not land");
    assert_eq!(refused.len(), 1);
    //: **Two different failures, counted apart.** A world that moved and a name that was never a
    //: key want different answers from whoever reads the counters.
    assert_eq!(s.ambiguous, 1);
    assert_eq!(s.vanished, 0);
    assert!(s.hands.0.is_empty(), "nothing was pressed");
}

/// A click on `Open`, optionally carrying which copy a person picked.
fn click_open(which: Option<Among>) -> Effect {
    Effect { kind: "click".into(), target: "Open".into(), role: "button".into(),
             text: String::new(), because_asked: Some(true), which }
}

#[test]
fn the_copy_a_person_named_is_the_one_pressed() {
    let screen = character_map();
    let mut s = looking(screen.clone(), screen);
    let (did, _, refused) = s.perform(&[click_open(Some(Among { index: 1, of: 2 }))]);
    assert_eq!(did.len(), 1);
    assert!(refused.is_empty());
    //: The second `Open` in tree order is the lower one — see `character_map`.
    assert_eq!(s.hands.0.len(), 1);
    assert_eq!(s.hands.0[0].rect.1, 200, "the first copy was pressed, not the one chosen");
    assert_eq!(s.ambiguous, 0);
    assert_eq!(s.chose_too_late, 0);
}

#[test]
fn an_answer_the_screen_outran_is_refused_rather_than_reinterpreted() {
    //: **The reason `Among` carries the count.** Three buttons became two while somebody read the
    //: question, so *the third one* now names nothing and *the second one* names a different
    //: button. Re-resolving alone does not catch this — the name still matches — which is why the
    //: count is checked before the index is used.
    let screen = character_map();
    let mut s = looking(screen.clone(), screen);
    let (did, _, refused) = s.perform(&[click_open(Some(Among { index: 1, of: 3 }))]);
    assert!(did.is_empty(), "an index chosen against a different screen must not be used");
    assert_eq!(refused.len(), 1);
    assert!(s.hands.0.is_empty());
    //: **Counted apart.** *Nobody was asked* and *asking is too slow here* say different things
    //: about whether this feature is worth having.
    assert_eq!(s.chose_too_late, 1);
    assert_eq!(s.ambiguous, 0);
}

#[test]
fn an_index_past_the_end_is_refused_even_when_the_count_agrees() {
    //: A hand-edited request, or a page that counted wrong. `of` matching does not make `index`
    //: safe, and an index out of range would panic rather than refuse.
    let screen = character_map();
    let mut s = looking(screen.clone(), screen);
    let (did, _, refused) = s.perform(&[click_open(Some(Among { index: 2, of: 2 }))]);
    assert!(did.is_empty());
    assert_eq!(refused.len(), 1);
    assert_eq!(s.chose_too_late, 1);
}

#[test]
fn a_chosen_index_does_nothing_when_the_name_resolves_to_one_thing() {
    //: The index only ever means *among these copies*. With a single match the ordinary path runs
    //: and the answer is simply not needed — it must not skip the resolution or press something else.
    let mut s = looking(vec![at("Open", "button", 40, 40)], vec![]);
    let (did, _, _) = s.perform(&[click_open(Some(Among { index: 5, of: 9 }))]);
    assert_eq!(did.len(), 1);
    assert_eq!(s.hands.0[0].rect.1, 40);
}

#[test]
fn the_role_is_still_what_rescues_the_font_trap() {
    //: The documented remedy for `Font :` is the role, and it still works — `Several` did not
    //: replace it. Two things share the name; one shares the name and the role.
    let screen = character_map();
    let mut s = looking(screen.clone(), screen.clone());
    assert!(matches!(s.find("Font :", ""), Found::Several(_)));
    let mut s = looking(screen.clone(), screen);
    match s.find("Font :", "combo box") {
        Found::One(w) => assert_eq!(w.role, "combo box"),
        other => panic!("the role is half the key and should settle this: {other:?}"),
    }
}

#[test]
fn one_before_and_several_now_is_a_thing_that_moved_and_it_still_resolves() {
    //: **The case the tie-break was built for, and it is untouched.** One `Open` last time, two
    //: now: the anchor is a fact about where that button was, so *nearest to it* is a statement
    //: about where it went rather than a coin toss.
    let mut s = looking(vec![at("Open", "button", 150, 10), at("Open", "button", 150, 400)],
                        vec![at("Open", "button", 152, 12)]);
    match s.find("Open", "button") {
        Found::One(w) => assert_eq!(w.rect.1, 10, "the near one, not the far one"),
        other => panic!("a moved control must still resolve: {other:?}"),
    }
}

#[test]
fn nothing_seen_before_is_not_an_anchor() {
    //: The Python fell back to `same[0]` here — the first one in tree order, for no reason at all.
    let mut s = looking(vec![at("Open", "button", 10, 10), at("Open", "button", 10, 90)], vec![]);
    assert!(matches!(s.find("Open", "button"), Found::Several(_)),
            "with nothing to anchor on, nearest-to-it is nearest to a guess");
}

#[test]
fn a_single_match_is_untouched_by_any_of_this() {
    let mut s = looking(vec![at("지우개", "button", 40, 40)], vec![]);
    match s.find("지우개", "button") {
        Found::One(w) => assert_eq!(w.name, "지우개"),
        other => panic!("the ordinary case must not have moved: {other:?}"),
    }
    let mut s = looking(vec![], vec![at("지우개", "button", 40, 40)]);
    assert_eq!(s.find("지우개", "button"), Found::Gone);
}

#[test]
fn the_order_carried_out_is_the_order_the_eyes_gave() {
    //: **What makes *the third one called `Open`* answerable.** A caller that asks a person which
    //: one gets an index back and has to re-resolve against the same ordering; a `Several` that
    //: sorted or deduplicated would make that index mean something else.
    let screen = vec![at("Open", "button", 10, 300), at("Open", "button", 10, 10),
                      at("Open", "button", 10, 150)];
    let mut s = looking(screen.clone(), screen);
    match s.find("Open", "button") {
        Found::Several(same) => {
            let tops: Vec<i64> = same.iter().map(|w| w.rect.1).collect();
            assert_eq!(tops, vec![300, 10, 150], "the list was reordered");
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn only_drops_the_other_two_cases_and_a_caller_has_to_say_so() {
    let screen = character_map();
    let mut s = looking(screen.clone(), screen);
    assert!(s.find("Open", "button").only().is_none(),
            "`only` exists so that throwing the ambiguity away is written down, not implied");
}
