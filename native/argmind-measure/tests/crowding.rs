//! **The padding has to be padding**, and three ways it could quietly stop being.
//!
//! `--bin crowding` answers one question — does choosing survive a list the size of a web page —
//! and it answers it by making the list longer with widgets borrowed from other screens. Every way
//! that construction can flatter itself is a way the sandbox gets started on a number that was not
//! measured, so the construction is what is tested here rather than the model.
//!
//! Nothing here calls a model. The padding is a pure function and this is why it was written as one.

use argmind_core::contract::Widget;
use argmind_measure::choosing::Goal;

fn w(name: &str, role: &str) -> Widget {
    Widget { name: name.into(), role: role.into(), rect: (0, 0, 10, 10), window: "x".into() }
}

fn goal(accept: Option<&[&str]>) -> Goal {
    Goal {
        screen: "x".into(),
        goal: "그거 눌러".into(),
        accept: accept.map(|a| a.iter().map(|s| s.to_string()).collect()),
    }
}

/// **A right answer must not appear twice.** A distractor carrying an accepted name would be a
/// second correct row, and the model finding it would be scored right for choosing something the
/// experiment invented.
#[test]
fn padding_never_adds_another_right_answer() {
    let real = vec![w("Open", "button"), w("Close", "button")];
    let pool = vec![w("Open", "menu item"), w("Save", "button"), w("Cancel", "button")];
    let (out, _) = argmind_measure::crowding::crowd(&real, &pool, &goal(Some(&["Open"])), 30, 1);
    assert_eq!(out.iter().filter(|x| x.name == "Open").count(), 1,
               "the accepted name appears more than once");
}

/// **And no name may appear twice at all**, accepted or not: the score is read off the *name* the
/// model returned, so two rows sharing one is a choice nobody can attribute.
#[test]
fn padding_never_duplicates_a_name_already_on_the_screen() {
    let real = vec![w("Save", "button"), w("Close", "button")];
    let pool = vec![w("Save", "menu item"), w("Print", "button"), w("Quit", "button")];
    let (out, _) = argmind_measure::crowding::crowd(&real, &pool, &goal(Some(&["Close"])), 24, 7);
    assert_eq!(out.iter().filter(|x| x.name == "Save").count(), 1);
    //: The real ones all survive — padding adds, it does not replace.
    for name in ["Save", "Close"] {
        assert!(out.iter().any(|x| x.name == name), "{name} was padded away");
    }
}

/// **The answer must not sit where a model could find it without reading.** Appending would leave
/// every real widget in the first rows; this inserts, so the target moves down as the list grows.
#[test]
fn the_answer_moves_as_the_list_grows() {
    let real = vec![w("Alpha", "button"), w("Target", "button"), w("Omega", "button")];
    let pool: Vec<Widget> = (0..40).map(|i| w(&format!("Other {i}"), "button")).collect();
    let g = goal(Some(&["Target"]));
    let (short, at_short) = argmind_measure::crowding::crowd(&real, &pool, &g, 3, 9);
    let (long, at_long) = argmind_measure::crowding::crowd(&real, &pool, &g, 200, 9);
    assert_eq!(short.len(), 3);
    assert_eq!(long.len(), 200);
    assert!(at_long.unwrap() > at_short.unwrap(),
            "the target stayed at row {:?} in a list of 200", at_short);
    //: **Reproducible.** One seed, one experiment — a rerun that shuffled differently would make
    //: two runs incomparable and the position confound unreadable.
    let (again, at_again) = argmind_measure::crowding::crowd(&real, &pool, &g, 200, 9);
    assert_eq!(at_long, at_again);
    assert_eq!(long.iter().map(|x| &x.name).collect::<Vec<_>>(),
               again.iter().map(|x| &x.name).collect::<Vec<_>>());
}

/// A goal whose right answer is *decline* has no index, and asking for one would invent it.
#[test]
fn a_goal_with_no_answer_reports_no_position() {
    let real = vec![w("Alpha", "button")];
    let pool = vec![w("Beta", "button"), w("Gamma", "button")];
    let (_, at) = argmind_measure::crowding::crowd(&real, &pool, &goal(None), 20, 3);
    assert_eq!(at, None);
}

/// **Asking for fewer than there are changes nothing.** The shortest arm is the control and has to
/// be the screen exactly as captured, or it is not a control.
#[test]
fn the_shortest_arm_is_the_screen_untouched() {
    let real = vec![w("Alpha", "button"), w("Target", "button")];
    let pool = vec![w("Beta", "button")];
    let (out, at) = argmind_measure::crowding::crowd(&real, &pool, &goal(Some(&["Target"])), 0, 5);
    assert_eq!(out.len(), 2);
    assert_eq!(at, Some(1));
    assert_eq!(out[0].name, "Alpha");
}
