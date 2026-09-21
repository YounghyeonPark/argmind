//! **Which voice argmind speaks in, and why it is a number.**
//!
//! The neural talker is a base model with no speakers of its own — `GET /v1/audio/voices` is empty
//! and every name is `unknown voice` — so the timbre is whatever the sampler drew. Measured
//! 2026-09-21: four unseeded calls, four different people; `seed = 7`, the same bytes four times.
//! The owner's report was 음성이 매번 바뀜, and this file is the settings half of the answer.
//!
//! Nothing here writes `devices.json`. The store is reached through [`chosen::voice_in`] and
//! [`chosen::parse`], which take what they are given — a test that exercised the real path would
//! be a test that changed the owner's microphone.

use std::collections::BTreeMap;

use argmind_serve::chosen;

fn map(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
    pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()
}

/// **There is always an answer, and it is never *whatever the sampler feels like*.**
///
/// The three ways a seed can be missing — no file, no key, a hand-edited value that is not a
/// number — all have to land on the same voice, because the defect being fixed is not *the wrong
/// voice*, it is *a different voice each time*.
#[test]
fn a_missing_or_broken_seed_is_still_one_fixed_voice() {
    assert_eq!(chosen::voice_in(&BTreeMap::new()), chosen::A_VOICE);
    assert_eq!(chosen::voice_in(&map(&[("microphone", "{0.0.1…}")])), chosen::A_VOICE);
    for wrong in ["", "   ", "일곱", "7.5", "9e9", "-"] {
        assert_eq!(chosen::voice_in(&map(&[("voice", wrong)])), chosen::A_VOICE,
                   "{wrong:?} should fall back rather than become a voice");
    }
    //: And a real one is read, spaces and all — the file is hand-editable and somebody will.
    assert_eq!(chosen::voice_in(&map(&[("voice", "42")])), 42);
    assert_eq!(chosen::voice_in(&map(&[("voice", " 42 ")])), 42);
}

/// **The seed survives a round trip through the file format**, which is hand-written and hand-read.
#[test]
fn a_voice_written_down_is_a_voice_read_back() {
    let all = map(&[("voice", "123"), ("camera", "Integrated Camera")]);
    let text = chosen::render(&all);
    let back = chosen::parse(&text);
    assert_eq!(chosen::voice_in(&back), 123);
    assert_eq!(back.get("camera").map(String::as_str), Some("Integrated Camera"));
}

/// **A voice is not a device**, and the two lists say so. The page enumerates hardware for
/// `KINDS`; there is no hardware to enumerate for a seed, and `try_device` has nothing to open.
#[test]
fn the_voice_is_a_key_and_not_a_kind() {
    assert!(!chosen::is_a_kind(chosen::VOICE));
    assert!(chosen::is_a_key(chosen::VOICE));
    for kind in chosen::KINDS {
        assert!(chosen::is_a_key(kind), "{kind} is a kind that cannot be written down");
    }
    assert_eq!(chosen::KEYS.len(), chosen::KINDS.len() + 1);
    //: A hand-edited file with a name nobody knows is ignored rather than carried.
    assert!(chosen::parse("{\n  \"voice!\": \"9\"\n}\n").is_empty());
}

// ---- the route ---------------------------------------------------------------------------------

fn asked(path: &str) -> argmind_serve::Route {
    argmind_serve::route(&argmind_serve::Request {
        method: "POST".into(),
        path: path.into(),
        body: String::new(),
        token: None,
    })
}

/// `/voice/<n>` and nothing else. **Anything that is not a number is `Unknown`**, not a zero —
/// a page that asked wrongly is told, rather than quietly given the first voice.
#[test]
fn only_a_number_chooses_a_voice() {
    assert_eq!(asked("/voice/7"), argmind_serve::Route::Voice(7));
    assert_eq!(asked("/voice/0"), argmind_serve::Route::Voice(0));
    for bad in ["/voice/", "/voice", "/voice/abc", "/voice/7/8", "/voice/7.0", "/voice/ 7"] {
        assert_eq!(asked(bad), argmind_serve::Route::Unknown, "{bad}");
    }
}

/// The default is inside the range the page may send, which is the sort of thing that is obviously
/// true until somebody changes one of the two numbers.
#[test]
fn the_default_voice_is_one_the_page_could_have_chosen() {
    assert!((0..=argmind_serve::app::MOST_VOICE).contains(&chosen::A_VOICE),
            "the default voice {} is outside 0..={}", chosen::A_VOICE,
            argmind_serve::app::MOST_VOICE);
}

// ---- what the page is told ----------------------------------------------------------------------

/// **`matters` is the honest field.** A seed chooses a speaker for the neural synthesiser and for
/// nothing else, so with SAPI answering — or with nothing speaking at all — the row has to say the
/// number does nothing. The alternative is a button that appears to work and does not.
#[test]
fn the_row_says_when_the_number_does_nothing() {
    let reads = |text: String| serde_json::from_str::<serde_json::Value>(&text)
        .unwrap_or_else(|e| panic!("the page cannot read {text:?}: {e}"));

    let neural = reads(argmind_serve::app::voice_json("neural"));
    assert_eq!(neural["matters"], true);
    assert_eq!(neural["is"], "neural");
    assert_eq!(neural["seed"], chosen::voice());
    assert_eq!(neural["most"], argmind_serve::app::MOST_VOICE);

    for is in ["windows", "off", "none"] {
        let d = reads(argmind_serve::app::voice_json(is));
        assert_eq!(d["matters"], false, "the seed was claimed to matter with {is} speaking");
        //: **The seed is still reported.** It is what the row will use the moment a neural voice
        //: is adopted, and blanking it would make the setting look unset rather than inert.
        assert_eq!(d["seed"], chosen::voice());
    }
}

/// One short sentence, and it has to be one — the sample is said on every press.
#[test]
fn the_sample_is_something_a_person_will_hear_repeatedly() {
    let sample = argmind_serve::app::A_SAMPLE;
    assert!(!sample.trim().is_empty());
    assert!(sample.chars().count() < 40, "{sample}");
    assert!(sample.chars().any(|c| ('가'..='힣').contains(&c)),
            "the sample is not Korean, and the voice being judged is a Korean one: {sample}");
}
