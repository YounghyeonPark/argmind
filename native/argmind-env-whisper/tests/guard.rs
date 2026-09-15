//! The transcript guard, replayed against the shipped Python's own verdicts.
//!
//! **A port of a guard is the kind of thing that looks right and is off by one character.** Python
//! slices by character; a byte window over `한글자막 by ` — 8 characters, 16 bytes — would count a
//! different thing entirely, and the failure would be invisible: the guard would simply stop
//! catching Korean loops while continuing to catch English ones.
//!
//! So `expectations/hearing.json` holds every corpus string with the Python's answer, and this
//! compares row by row. Generate it with `python native/expectations/hearing.py`; it stays as data
//! once the Python goes.

use argmind_env_whisper::guard::{repeating, why_rejected, wordless, REPEAT_FLOOR, REPEAT_PROBE,
                                 REPEAT_TIMES};

fn expectations() -> String {
    let at = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap().join("expectations").join("hearing.json");
    std::fs::read_to_string(&at).unwrap_or_else(|_| {
        panic!("{} is missing. Generate it: python native/expectations/hearing.py", at.display())
    })
}

/// Every row of the record: the text, and the four answers beside it.
fn rows(section: &str) -> Vec<(String, bool, bool, Option<String>, Option<String>)> {
    let raw = expectations();
    let at = raw.find(&format!("\"{section}\"")).unwrap_or_else(|| panic!("no {section} section"));
    //: Read out of the text rather than parsed into a document — this crate takes no JSON crate,
    //: for the same reason it takes no HTTP one. The generator writes indent-2 JSON, so a row is a
    //: run of `"key": value` lines between braces at a known depth.
    let rest = &raw[at..];
    let end = rest.find("\n  ]").unwrap_or(rest.len());
    let mut out = Vec::new();
    let mut text: Option<String> = None;
    let (mut wordless, mut repeating) = (false, false);
    let mut why = None;
    for line in rest[..end].lines() {
        let t = line.trim();
        if let Some(v) = t.strip_prefix("\"text\": ") {
            text = Some(unquote(v));
        } else if let Some(v) = t.strip_prefix("\"wordless\": ") {
            wordless = v.starts_with("true");
        } else if let Some(v) = t.strip_prefix("\"repeating\": ") {
            repeating = v.starts_with("true");
        } else if let Some(v) = t.strip_prefix("\"why\": ") {
            why = maybe(v);
        } else if let Some(v) = t.strip_prefix("\"why_at_5_per_second\": ") {
            out.push((text.take().expect("a row has a text"), wordless, repeating,
                      why.take(), maybe(v)));
        }
    }
    assert!(!out.is_empty(), "no rows read from {section}");
    out
}

fn maybe(value: &str) -> Option<String> {
    let v = value.trim_end_matches(',');
    if v.starts_with("null") || v == "\"\"" { None } else { Some(unquote(v)) }
}

/// The generator writes JSON strings; only the escapes it can emit are handled, and anything else
/// is a corpus this reader has not been shown.
fn unquote(value: &str) -> String {
    let v = value.trim().trim_end_matches(',');
    let inner = v.trim_start_matches('"').trim_end_matches('"');
    let mut out = String::new();
    let mut chars = inner.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('n') => out.push('\n'),
            Some('t') => out.push('\t'),
            Some(other) => out.push(other),
            None => break,
        }
    }
    out
}

/// **The constants are the measured ones.** A floor lowered to make something pass is the failure
/// this whole standard of evidence exists against.
#[test]
fn the_constants_are_the_ones_that_were_measured() {
    let raw = expectations();
    for (name, value) in [("floor", REPEAT_FLOOR), ("probe", REPEAT_PROBE),
                          ("times", REPEAT_TIMES)] {
        assert!(raw.contains(&format!("\"{name}\": {value}")),
                "{name} is {value} here and something else in hearing.json");
    }
    assert_eq!((REPEAT_FLOOR, REPEAT_PROBE, REPEAT_TIMES), (32, 10, 3));
}

/// Every string, every answer, both implementations.
#[test]
fn the_two_guards_agree_on_every_recorded_string() {
    let mut checked = 0;
    for section in ["degenerate", "clean", "repetitive_human", "edges"] {
        for (text, want_wordless, want_repeating, want_why, want_fast) in rows(section) {
            assert_eq!(wordless(&text), want_wordless,
                       "{section}: wordless disagrees on {text:?}");
            assert_eq!(repeating(&text), want_repeating,
                       "{section}: repeating disagrees on {text:?}");
            assert_eq!(why_rejected(&text, 0.0, None).map(str::to_string), want_why,
                       "{section}: the reason disagrees on {text:?}");
            //: One second of audio against a limit of five characters a second, which is what the
            //: generator recorded — this is the half that proves the limit is wired rather than
            //: merely present.
            assert_eq!(why_rejected(&text, 1.0, Some(5.0)).map(str::to_string), want_fast,
                       "{section}: the reason under a pace limit disagrees on {text:?}");
            checked += 1;
        }
    }
    //: Ten degenerations, nine clean, nine a person could say, nine edges. A count rather than a
    //: floor, so a corpus that silently shrank would be noticed.
    assert_eq!(checked, 37, "the record has changed size");
}

/// **7 of the 10 reproduced degenerations are caught, and the other three are not repetitions.**
///
/// Named rather than counted quietly: two copies of `Thank you.` is a person saying 네, 네, and the
/// Portuguese outro occurs once. They belong to the silence finding and this guard does not claim
/// them — a guard credited with catches it did not make is one nobody can size.
#[test]
fn the_guard_catches_the_loops_and_claims_nothing_else() {
    let caught: Vec<String> = rows("degenerate").into_iter()
        .filter(|(_, _, _, why, _)| why.is_some())
        .map(|(t, ..)| t).collect();
    assert_eq!(caught.len(), 7, "{caught:#?}");
    let missed: Vec<String> = rows("degenerate").into_iter()
        .filter(|(_, _, _, why, _)| why.is_none())
        .map(|(t, ..)| t).collect();
    assert_eq!(missed.len(), 3);
    assert!(missed.iter().any(|t| t == "Thank you. Thank you."), "{missed:#?}");
}

/// **Nothing that must survive is thrown away**, which is the side that costs a turn.
///
/// A rejected turn is silent; a fabricated one at least appears in the transcript where the person
/// can see it. So a false positive here is the more expensive mistake.
#[test]
fn no_string_that_had_to_survive_is_rejected() {
    for section in ["clean", "repetitive_human"] {
        for (text, ..) in rows(section) {
            assert!(why_rejected(&text, 0.0, None).is_none(),
                    "the guard would have thrown away {text:?}");
        }
    }
}

/// **The floor is what protects a person repeating themselves**, and it is a cliff, not a margin.
#[test]
fn a_short_repetition_survives_because_it_is_short() {
    let three = "아니 그거 말고 아니 그거 말고 아니 그거 말고";
    assert!(three.chars().count() < REPEAT_FLOOR);
    assert!(!repeating(three));
    //: Five copies is over the floor and is rejected. Written down because it is the guard's
    //: weakest edge and somebody should meet it here rather than in a lost turn.
    let five = "아니 그거 말고 아니 그거 말고 아니 그거 말고 아니 그거 말고 아니 그거 말고";
    assert!(five.chars().count() >= REPEAT_FLOOR);
    assert!(repeating(five));
}

/// **Characters, not bytes**, which is the way this port could have failed silently.
#[test]
fn the_window_is_counted_in_characters() {
    //: `한글자막 by ` is 8 characters and 16 bytes — **the Python's comment says 9 and it is 8**,
    //: counted here while porting. Five copies is 40 characters, over the floor, and every
    //: 10-character window recurs. Counted in bytes the floor would be cleared far too early and
    //: the windows would land mid-character, which does not even slice.
    let korean = "한글자막 by ".repeat(5);
    assert_eq!("한글자막 by ".chars().count(), 8);
    assert!("한글자막 by ".len() > 9, "the byte length is not the character length");
    assert!(repeating(&korean));

    //: And a Korean sentence of the same *byte* length that is not a loop survives.
    let sentence = "화면에 스무 개의 버튼이 있고 지우개는 왼쪽 위에 있습니다";
    assert!(sentence.len() > REPEAT_FLOOR, "the byte length clears the floor");
    assert!(!repeating(sentence), "a byte-counted window would have called this a loop");
}
