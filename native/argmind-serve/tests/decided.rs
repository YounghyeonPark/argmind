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
    assert_eq!(DECIDABLE.len(), 8);
}

/// The six needs and the two switches, and no third kind smuggled in by a typo.
#[test]
fn the_list_is_the_needs_and_the_two_switches() {
    assert_eq!(DECIDABLE, ["ollama", "model", "whisper", "korean-voice", "vulkan", "speaker",
                           "notice", "microphone"]);
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


/// **A thing answered once and for all is never asked again.**
///
/// It was, every launch. The page drew the four buttons for anything missing that had a fix, and
/// the standing answer then acted on it a few seconds later and the question vanished — asking and
/// then ignoring the answer, which is worse than either alone. The owner noticed:
/// *권한 물어보는게 앱 실행시마다 물어보는데 이게 맫나*.
///
/// The decisions are handed in rather than read, so this cannot overwrite what somebody actually
/// decided in order to test what happens when they decide it.
#[test]
fn a_standing_answer_is_sent_with_the_question_so_the_page_can_stop_asking() {
    use std::collections::BTreeMap;

    let needs = argmind_serve::needs::every("qwen3:8b");
    assert!(needs.len() >= 4, "only {} needs", needs.len());

    //: Nothing decided: every row says so, and the page draws its buttons.
    let plain = argmind_serve::app::needs_as_json(&needs, &BTreeMap::new());
    assert!(plain.contains("\"standing\":\"\""), "{plain}");
    assert!(!plain.contains("\"standing\":\"allowed\""));

    //: One of each, on whichever keys this machine actually has.
    let mut decided = BTreeMap::new();
    let keys: Vec<&str> = needs.iter().map(|n| n.key).collect();
    decided.insert(keys[0].to_string(), Standing::Allowed);
    if keys.len() > 1 {
        decided.insert(keys[1].to_string(), Standing::Denied);
    }
    let answered = argmind_serve::app::needs_as_json(&needs, &decided);
    assert!(answered.contains("\"standing\":\"allowed\""), "{answered}");
    if keys.len() > 1 {
        assert!(answered.contains("\"standing\":\"denied\""), "{answered}");
    }
    //: **And a key nobody decided still says nothing**, rather than inheriting a neighbour's
    //: answer — which is the way a loop over a map gets this wrong.
    if keys.len() > 2 {
        assert!(answered.contains("\"standing\":\"\""), "{answered}");
    }
}

/// **A standing answer about the microphone is the answer, not a note beside the real one.**
///
/// The owner had `microphone: allowed` in the settings and argmind still would not listen: hearing
/// was a command-line flag and nothing read the file. The page said *전에 항상 허용하셨습니다 —
/// 물어보지 않고 진행합니다* next to a microphone that was switched off, which is the same defect as
/// asking again every launch, in a second place.
#[test]
fn the_settings_decide_whether_it_listens_and_the_flag_only_fills_a_gap() {
    use argmind_serve::app::hearing_from;

    //: Nothing answered: the flag is all there is, and off is the safe direction.
    assert!(!hearing_from(false, None));
    assert!(hearing_from(true, None));

    //: Answered 항상 허용: it listens, flag or no flag. This is the case that was broken.
    assert!(hearing_from(false, Some(Standing::Allowed)));
    assert!(hearing_from(true, Some(Standing::Allowed)));

    //: **Answered 항상 거부 beats the flag**, deliberately. A refusal in the settings is a more
    //: considered thing than whatever is on the shortcut somebody clicked, and a launcher that
    //: could override it would make 거부 mean *until the next time a shortcut is edited*.
    assert!(!hearing_from(true, Some(Standing::Denied)));
    assert!(!hearing_from(false, Some(Standing::Denied)));
}

/// **Watching is allowed unless it was refused, and a default is not an answer.**
///
/// The owner, 2026-09-17: 지켜보기허용을 기본값으로 하고, 설정에서 토글하도록 하자. The cost that
/// kept it off is unchanged — a model call every three seconds at most for as long as it is on —
/// and is now printed under the switch rather than asked about.
#[test]
fn the_watch_is_allowed_by_default_and_the_file_still_says_nothing() {
    use argmind_serve::decided::{written, ALLOWED_UNLESS_REFUSED, standing};

    assert!(ALLOWED_UNLESS_REFUSED.contains(&"notice"));
    //: Every defaulted key must be a key that can be decided at all, or 설정 could never undo it.
    for key in ALLOWED_UNLESS_REFUSED {
        assert!(argmind_serve::decided::is_decidable(key), "{key} cannot be turned off");
    }
    //: **Nothing else got a default by accident.** A permission that installs something or opens a
    //: microphone is not on this list, and this is what would catch it being added quietly.
    for key in argmind_serve::decided::DECIDABLE {
        if key != "notice" {
            assert!(!ALLOWED_UNLESS_REFUSED.contains(&key),
                    "{key} became allowed without anybody being asked");
        }
    }
    //: `standing` fills the default in; `written` never does. The settings panel needs the second
    //: one — a row reading 항상 허용 with a 되돌리기 button that removes nothing is a lie.
    let key = "notice";
    match written(key) {
        //: On a machine where somebody has actually answered, the answer wins either way.
        Some(answer) => assert_eq!(standing(key), Some(answer)),
        None => assert_eq!(standing(key), Some(Standing::Allowed)),
    }
}
