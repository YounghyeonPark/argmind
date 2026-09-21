//! **Which voice argmind speaks in, and why it is a list rather than a number.**
//!
//! The neural talker is a base model with no speakers of its own, so the timbre is whatever the
//! sampler drew. Two mechanisms were tried, a day apart, and only the second answers the owner's
//! 음성이 매번 바뀜:
//!
//! **A seed.** Reproducible — four unseeded calls gave four different voices, `seed = 7` gave the
//! same bytes four times out of four. But across four *different* sentences a seed held to 119 Hz
//! of itself while different seeds on one sentence spanned 137 Hz, a ratio of 1.2, and long-term
//! average spectra were no more alike within a seed (0.821) than across them (0.813), with the
//! same-seed-same-text control at exactly 1.000. Every reply is a different sentence, so a seed
//! alone is a different person every turn. That reading is kept rather than quietly replaced,
//! because it was the flattering one.
//!
//! **A cloned voice.** argmind draws one reference sentence at a fixed seed and registers it; the
//! clone tracks its reference's pitch at +0.99 and is 0.955-0.982 like itself across sentences.
//! Five of those are the presets, ordered by measured pitch and chosen by measured *separation* —
//! seventy-two seeds cloned, seven that a pitch rule liked, five once the right question was asked.
//! See `no_two_presets_are_the_same_person`.
//!
//! Nothing here writes `devices.json` or speaks to the server. The store is reached through
//! [`chosen::voice_in`] and [`chosen::parse`], which take what they are given — a test that
//! exercised the real path would be a test that changed the owner's microphone.

use std::collections::BTreeMap;

use argmind_serve::{chosen, speaker};

fn map(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
    pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()
}

/// **There is always an answer, and it is never *whatever the sampler feels like*.**
///
/// Every way the setting can be absent or wrong lands on the same voice, because the defect being
/// fixed is not *the wrong voice*, it is *a different voice each time*.
#[test]
fn a_missing_or_unknown_voice_is_still_one_fixed_voice() {
    let fallback = speaker::preset(speaker::A_VOICE).expect("A_VOICE names a preset");
    assert_eq!(chosen::voice_in(&BTreeMap::new()).key, fallback.key);
    assert_eq!(chosen::voice_in(&map(&[("microphone", "{0.0.1…}")])).key, fallback.key);
    //: **`low` and `mid` are in this list on purpose.** They were the keys this shipped with for
    //: an hour, before measuring which presets are actually different *people* replaced five names
    //: with a different five. Anything that wrote them is as stale as a seed.
    for wrong in ["", "   ", "없는목소리", "ONE", "0", "low", "mid"] {
        assert_eq!(chosen::voice_in(&map(&[("voice", wrong)])).key, fallback.key,
                   "{wrong:?} should fall back rather than become a voice");
    }
    //: **Surrounding space is trimmed rather than refused**, because this file is hand-editable and
    //: a key with a space after it is a typo with an obvious intention. Case is not, because a
    //: key is a name and not a sentence.
    assert_eq!(chosen::voice_in(&map(&[("voice", "  one  ")])).key, "one");
    //: **A bare number is the migration case, not a typo.** Until 2026-09-21 this setting held a
    //: seed, so every file written by the version before this one arrives here — and the honest
    //: answer is the default rather than a guess at which preset the number was nearest.
    for old_seed in ["7", "42", "9999"] {
        assert_eq!(chosen::voice_in(&map(&[("voice", old_seed)])).key, fallback.key);
    }
    //: And a real one is read.
    assert_eq!(chosen::voice_in(&map(&[("voice", "one")])).key, "one");
}

/// **The seed survives a round trip through the file format**, which is hand-written and hand-read.
#[test]
fn a_voice_written_down_is_a_voice_read_back() {
    let all = map(&[("voice", "five"), ("camera", "Integrated Camera")]);
    let back = chosen::parse(&chosen::render(&all));
    assert_eq!(chosen::voice_in(&back).key, "five");
    assert_eq!(back.get("camera").map(String::as_str), Some("Integrated Camera"));
}

/// **A voice is not a device**, and the two lists say so. The page enumerates hardware for
/// `KINDS`; there is none to enumerate for a voice, and `try_device` has nothing to open.
#[test]
fn the_voice_is_a_key_and_not_a_kind() {
    assert!(!chosen::is_a_kind(chosen::VOICE));
    assert!(chosen::is_a_key(chosen::VOICE));
    for kind in chosen::KINDS {
        assert!(chosen::is_a_key(kind), "{kind} is a kind that cannot be written down");
    }
    assert_eq!(chosen::KEYS.len(), chosen::KINDS.len() + 1);
    assert!(chosen::parse("{\n  \"voice!\": \"low\"\n}\n").is_empty());
}

// ---- the presets themselves ----------------------------------------------------------------------

/// **The list is ordered by the number it prints**, which is the one thing a reader can check.
#[test]
fn the_presets_are_ordered_by_the_pitch_they_claim() {
    let hz: Vec<u32> = speaker::PRESETS.iter().map(|p| p.hz).collect();
    let mut sorted = hz.clone();
    sorted.sort_unstable();
    assert_eq!(hz, sorted, "the list is not in pitch order: {hz:?}");
}

/// **Every preset is a different person, and pitch is not how that was decided.**
///
/// The first attempt at this rule asked whether neighbours were further apart in pitch than they
/// wobbled, and it failed the list it was written for — correctly, as it turned out, but for the
/// wrong reason. Two voices can share a pitch and be two people; two can differ in pitch and be
/// one. What settles it is whether the spectra separate, and `apart` carries that reading: the
/// gap between how alike a voice is to itself across sentences and how alike it is to its nearest
/// neighbour, in thousandths, with no notion of pitch in it.
///
/// Of seven candidates that passed the pitch rule, two pairs measured **+0.016** and **−0.004** —
/// one person under two names — and were dropped. [`speaker::LEAST_APART`] is the weakest that was
/// kept, written down so the list cannot quietly grow a pair worse than the ones already rejected.
#[test]
fn no_two_presets_are_the_same_person() {
    for p in &speaker::PRESETS {
        assert!(p.apart >= speaker::LEAST_APART,
                "{} is {} thousandths from its nearest neighbour, under the {} that was kept",
                p.key, p.apart, speaker::LEAST_APART);
    }
    //: **And the reading is a separation, not a similarity.** A `apart` near 1000 would mean
    //: somebody had put the wrong quantity in the field, which no threshold below would catch.
    assert!(speaker::PRESETS.iter().all(|p| p.apart < 500), "apart is not a separation");
}

/// **The wobble is n=6 and the reason is a list that was chosen on n=3.**
///
/// Over three sentences these seeds read 4–19 Hz of movement; over six, 16–33. A max-minus-min of
/// three samples underestimates a range, and the list picked on the small numbers had two pairs in
/// it that were one person. The shipped numbers are the larger ones.
#[test]
fn the_wobble_is_the_honest_one() {
    for p in &speaker::PRESETS {
        assert!(p.wobble >= 10,
                "{} claims to move only {} Hz across sentences, which is the n=3 reading",
                p.key, p.wobble);
        //: A voice that moved further than the whole range spans is not a voice.
        assert!(p.wobble < 60, "{} moves {} Hz, which is most of the range", p.key, p.wobble);
    }
}

/// Keys are unique, non-empty and URL-safe, because one goes in a path.
#[test]
fn every_preset_can_be_named_in_a_request() {
    let mut seen = std::collections::BTreeSet::new();
    for p in &speaker::PRESETS {
        assert!(!p.key.is_empty() && !p.what.is_empty(), "{}", p.key);
        assert!(p.key.chars().all(|c| c.is_ascii_lowercase()), "{} is not url-safe", p.key);
        assert!(seen.insert(p.key), "{} appears twice", p.key);
        //: A seed is what the reference is drawn at, and two presets drawn at one seed would be
        //: one voice under two names.
        assert!(p.hz > 0 && p.wobble > 0, "{} has no measurement behind it", p.key);
    }
    let seeds: std::collections::BTreeSet<i64> = speaker::PRESETS.iter().map(|p| p.seed).collect();
    assert_eq!(seeds.len(), speaker::PRESETS.len(), "two presets share a seed");
    assert!(speaker::preset(speaker::A_VOICE).is_some(), "the default is not in the list");
    assert!(speaker::preset("없는것").is_none());
}

/// The reference is one sentence of Korean, said once per launch. **Long enough to carry a person
/// and short enough to cost half a second** — 0.53 s to draw, 0.16 s to extract, measured.
#[test]
fn the_reference_is_a_sentence_the_extractor_can_work_with() {
    let r = speaker::REFERENCE;
    assert!(r.chars().count() > 20 && r.chars().count() < 80, "{r}");
    assert!(r.chars().any(|c| ('가'..='힣').contains(&c)), "{r}");
}

// ---- the route -------------------------------------------------------------------------------

fn asked(path: &str) -> argmind_serve::Route {
    argmind_serve::route(&argmind_serve::Request {
        method: "POST".into(),
        path: path.into(),
        body: String::new(),
        token: None,
    })
}

/// `/voice/<key>`, one segment. **The key is not checked here** — the route parses and the far end
/// decides, the same split as `/choose` and `/dial`.
#[test]
fn a_voice_is_asked_for_by_one_name() {
    assert_eq!(asked("/voice/low"), argmind_serve::Route::Voice("low".into()));
    assert_eq!(asked("/voice/없는것"), argmind_serve::Route::Voice("없는것".into()));
    for bad in ["/voice/", "/voice", "/voice/low/high"] {
        assert_eq!(asked(bad), argmind_serve::Route::Unknown, "{bad}");
    }
}

// ---- what the page is told ----------------------------------------------------------------------

/// **`matters` is the honest field.** The choice reaches the neural synthesiser and nothing else,
/// so with SAPI answering — or with nothing speaking — the row has to say so. The alternative is a
/// dropdown that appears to work and does not.
#[test]
fn the_row_says_when_the_choice_does_nothing() {
    let reads = |text: String| serde_json::from_str::<serde_json::Value>(&text)
        .unwrap_or_else(|e| panic!("the page cannot read {text:?}: {e}"));

    let neural = reads(argmind_serve::app::voice_json("neural"));
    assert_eq!(neural["matters"], true);
    assert_eq!(neural["is"], "neural");
    assert_eq!(neural["chosen"], chosen::voice().key);
    let listed = neural["presets"].as_array().expect("no presets went to the page");
    assert_eq!(listed.len(), speaker::PRESETS.len());
    //: **Every field the page reads is present**, spelled the way it reads it — the note payload
    //: shipped broken for two commits over a key name, so keys get asserted here.
    for row in listed {
        for key in ["key", "what", "hz", "wobble"] {
            assert!(row.get(key).is_some(), "the page asks for {key} and it is not in {row}");
        }
    }
    //: And the chosen one is in the list, or the dropdown opens on nothing selected.
    assert!(listed.iter().any(|r| r["key"] == neural["chosen"]), "{neural}");

    for is in ["windows", "off", "none"] {
        let d = reads(argmind_serve::app::voice_json(is));
        assert_eq!(d["matters"], false, "the choice was claimed to matter with {is} speaking");
        //: **The list is still sent.** It is what the row will use the moment a neural voice is
        //: adopted, and blanking it would make the setting look missing rather than inert.
        assert_eq!(d["presets"].as_array().map(Vec::len), Some(speaker::PRESETS.len()));
    }
}

/// One short sentence, and it has to be one — the sample is said on every change.
#[test]
fn the_sample_is_something_a_person_will_hear_repeatedly() {
    let sample = argmind_serve::app::A_SAMPLE;
    assert!(!sample.trim().is_empty());
    assert!(sample.chars().count() < 40, "{sample}");
    assert!(sample.chars().any(|c| ('가'..='힣').contains(&c)),
            "the sample is not Korean, and the voice being judged is a Korean one: {sample}");
    //: **Not the reference sentence.** They are heard one after another when a voice is chosen, and
    //: two identical sentences would make the demonstration sound like a stutter.
    assert_ne!(sample, speaker::REFERENCE);
}
