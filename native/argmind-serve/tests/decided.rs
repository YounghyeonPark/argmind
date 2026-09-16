//! What a standing answer means, and what may have one.
//!
//! **The pure half is all of it that matters.** Whether the file lands in `%LOCALAPPDATA%` is one
//! line; what a stored *yes* is attached to, and what can never have one, is the design.

use argmind_serve::decided::{all, is_decidable, parse, render, what, Answer, Standing, DECIDABLE};

/// **Four answers and two of them are remembered.** That is the whole point of having four.
#[test]
fn a_one_off_leaves_nothing_behind_and_an_always_does() {
    assert_eq!(Answer::Allow.standing(), None);
    assert_eq!(Answer::Deny.standing(), None);
    assert_eq!(Answer::AlwaysAllow.standing(), Some(Standing::Allowed));
    assert_eq!(Answer::AlwaysDeny.standing(), Some(Standing::Denied));

    //: And two of them let the thing happen now. **A standing no still says no now** — pressing
    //: 항상 거부 must not be a way to have it happen one last time.
    assert!(Answer::Allow.permits());
    assert!(Answer::AlwaysAllow.permits());
    assert!(!Answer::Deny.permits());
    assert!(!Answer::AlwaysDeny.permits());
}

/// The words the page sends, and nothing else.
#[test]
fn only_the_four_words_are_answers() {
    for (word, want) in [("allow", Answer::Allow), ("always", Answer::AlwaysAllow),
                         ("deny", Answer::Deny), ("never", Answer::AlwaysDeny)] {
        assert_eq!(Answer::from_word(word), Some(want));
    }
    //: A page that could send anything could send something this does not understand, and a
    //: permission system that guesses is not one.
    for junk in ["", "yes", "ALLOW", "always allow", "true", "1"] {
        assert_eq!(Answer::from_word(junk), None, "{junk:?}");
    }
}

/// **A click is not on the list, and that is the point.**
///
/// Everything here is either *change the machine* or *switch on something that keeps running*. An
/// action the model chose is neither: it gets its button every single time, and the rule that one
/// place in this program performs anything is held by a test of its own.
#[test]
fn nothing_that_performs_an_action_can_be_answered_always() {
    for forbidden in ["click", "type", "key", "perform", "act", "confirm", "anything"] {
        assert!(!is_decidable(forbidden), "{forbidden:?} must never be answerable once and for all");
    }
    for allowed in DECIDABLE {
        assert!(is_decidable(allowed));
        //: And every one of them has a name a person would recognise in a window.
        assert_ne!(what(allowed), "알 수 없는 권한", "{allowed} has no name");
    }
    assert_eq!(DECIDABLE.len(), 6);
}

/// The four needs and the two switches, and no third kind smuggled in by a typo.
#[test]
fn the_list_is_the_needs_and_the_two_switches() {
    assert_eq!(DECIDABLE, ["ollama", "model", "whisper", "korean-voice", "notice", "microphone"]);
    //: Every need `needs::every` can report is decidable, or it is a question with no way to
    //: answer it for good.
    for need in argmind_serve::needs::every("qwen3:8b") {
        assert!(is_decidable(need.key), "the need {:?} cannot be remembered", need.key);
    }
}

/// Written and read back.
#[test]
fn what_is_written_is_what_comes_back() {
    let mut kept = std::collections::BTreeMap::new();
    kept.insert("ollama".to_string(), Standing::Allowed);
    kept.insert("notice".to_string(), Standing::Denied);
    let text = render(&kept);
    assert_eq!(parse(&text), kept);
    //: It is a file a person is meant to be able to open and understand.
    assert!(text.contains("\"ollama\": \"allowed\""), "{text}");
    assert!(text.contains("\"notice\": \"denied\""), "{text}");
    //: And nothing decided is nothing, tidily.
    assert_eq!(render(&std::collections::BTreeMap::new()), "{}\n");
    assert!(parse("{}\n").is_empty());
}

/// **A file somebody edited badly asks again**, which is the safe direction.
#[test]
fn a_key_or_a_value_this_does_not_know_is_dropped() {
    //: A permission the program never asks about cannot be granted by writing it in the file.
    assert!(parse("{\"click\": \"allowed\"}").is_empty());
    assert!(parse("{\"ollamaa\": \"allowed\"}").is_empty(), "a typo is not a permission");
    //: Nor by a word that is not one of the two.
    assert!(parse("{\"ollama\": \"yes\"}").is_empty());
    assert!(parse("{\"ollama\": \"\"}").is_empty());
    //: And nothing at all is no decisions rather than a panic.
    assert!(parse("").is_empty());
    assert!(parse("{}").is_empty());
    assert!(parse("not json at all").is_empty());
    //: The good rows in a half-bad file survive, so one bad line does not forget everything.
    let mixed = parse("{\n  \"ollama\": \"allowed\",\n  \"click\": \"allowed\"\n}");
    assert_eq!(mixed.len(), 1);
    assert_eq!(mixed.get("ollama"), Some(&Standing::Allowed));
}

/// **The settings panel has to be able to show what is still unanswered**, so the list is every
/// decidable thing rather than only the ones with an answer.
#[test]
fn everything_decidable_is_listed_answered_or_not() {
    let rows = all();
    assert_eq!(rows.len(), DECIDABLE.len());
    for (key, name, _) in &rows {
        assert!(is_decidable(key));
        assert!(!name.is_empty());
    }
}
