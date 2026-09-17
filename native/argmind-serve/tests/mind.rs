//! **The thinking goal and the dials survive a restart, and a broken file means rest.**
//!
//! Nothing here touches `%LOCALAPPDATA%\argmind\mind.json`. `parse` and `render` are pure and are
//! tested as a pair, for the same reason [`crate::decided`]'s are: a test that wrote the real file
//! would overwrite what the person running it actually set, to check that setting it works.
//!
//! The route half is here too, because a dial whose name the page can invent is the same defect as
//! a permission whose key it can invent, and that one was caught by a test like this.

use argmind_core::mind::{Mind, DIALS, MOST};
use argmind_serve::mind::{is_a_dial, parse, render, AIM};
use argmind_serve::{route, Request, Route};

fn asked(path: &str) -> Route {
    route(&Request { method: "POST".into(), path: path.into(), token: None, body: String::new() })
}

/// **What is written is what comes back.** Both directions, because a renderer and a reader that
/// are each self-consistent can still disagree with one another.
#[test]
fn a_mind_written_down_reads_back_the_same() {
    let mut mind = Mind::resting();
    mind.turn("재치", 7);
    mind.turn("헌신", MOST);
    mind.aim = "코드 리뷰 중일 때는 놓친 것만 말해줘".into();

    let text = render(&mind);
    let back = parse(&text);
    assert_eq!(back, mind, "wrote {text}");
    //: And the line the model would be shown is the same line, which is the thing that actually
    //: has to survive the round trip.
    assert_eq!(back.line(), mind.line());
}

/// **A dial at rest is absent from the file**, so the file says what somebody did rather than
/// listing four zeros nobody chose.
#[test]
fn nothing_set_writes_an_empty_file_and_an_empty_file_is_rest() {
    assert_eq!(render(&Mind::resting()), "{}\n");
    assert!(parse("{}\n").at_rest());
    assert!(parse("").at_rest());

    let mut one = Mind::resting();
    one.turn("유머", 3);
    let text = render(&one);
    assert!(text.contains("\"유머\": 3"), "{text}");
    for (korean, _) in DIALS {
        if korean != "유머" {
            assert!(!text.contains(korean), "{korean} is in a file nobody put it in: {text}");
        }
    }
}

/// A hand-edited file, or one from a version with different dials: **a name nobody knows is
/// dropped and the rest of the file still counts.**
#[test]
fn a_file_with_rubbish_in_it_loses_the_rubbish_and_nothing_else() {
    let text = "{\n  \"재치\": 5,\n  \"신중\": 9,\n  \"유머\": banana,\n  \
                \"헌신\": \"8\",\n  \"목표\": \"조용히\"\n}\n";
    let mind = parse(text);
    assert_eq!(mind.of("재치"), 5);
    assert_eq!(mind.of("신중"), 0, "an invented dial got in");
    assert_eq!(mind.of("유머"), 0, "a value that is not a number got in");
    //: A person who hand-quotes a number still means the number.
    assert_eq!(mind.of("헌신"), 8);
    assert_eq!(mind.aim, "조용히");
}

/// **A goal is prose**, so it gets a colon, a comma and a quote in it and still comes back whole.
#[test]
fn a_goal_with_punctuation_in_it_survives() {
    let awkward = "이렇게: \"놓친 것\"만, 짧게 — C:\\dev\\argmind 볼 때는 특히";
    let mut mind = Mind::resting();
    mind.aim = awkward.into();
    assert_eq!(parse(&render(&mind)).aim, awkward);
}

/// **A number above the top comes down rather than through.** Both on the way in and on the way
/// back out of a file somebody edited.
#[test]
fn an_impossible_number_is_brought_down() {
    assert_eq!(parse("{\"재치\": 200}").of("재치"), MOST);
    let mut mind = Mind::resting();
    mind.turn("재치", 200);
    assert!(render(&mind).contains(&format!("\"재치\": {MOST}")));
}

/// **The page cannot invent a dial**, and the route says so before anything is written.
#[test]
fn the_dial_route_takes_the_four_names_and_no_others() {
    for (korean, _) in DIALS {
        assert!(is_a_dial(korean));
        //: The page percent-encodes the name because it is Korean; the router undoes that.
        let encoded: String = korean.as_bytes().iter().map(|b| format!("%{b:02X}")).collect();
        assert_eq!(asked(&format!("/dial/{encoded}/7")), Route::Dial(korean.to_string(), 7));
    }
    assert!(!is_a_dial("신중"));
    assert!(!is_a_dial(AIM), "the goal is not a dial");
    //: Not a number, and no number at all.
    assert_eq!(asked("/dial/재치/high"), Route::Unknown);
    assert_eq!(asked("/dial/재치"), Route::Unknown);
    assert_eq!(asked("/dial/"), Route::Unknown);
}

/// The sentence travels in the body, and **an empty body is how it is cleared** rather than a
/// separate route nobody would remember to add.
#[test]
fn the_goal_route_carries_the_sentence_in_the_body() {
    let said = |body: &str| route(&Request { method: "POST".into(), path: "/aim".into(),
                                             token: None, body: body.into() });
    assert_eq!(said("조용히 해줘"), Route::Aim("조용히 해줘".into()));
    assert_eq!(said(""), Route::Aim(String::new()));
}
