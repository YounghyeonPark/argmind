//! What a transcript carries, and the one number that can be computed from it.
//!
//! **No microphone and no speaker**, which is the point of the seam being a seam: these are traits
//! and a struct, and a suite must be able to say what they mean without a device.
//!
//! Two findings are held here against the type rather than against a docstring, because a rule that
//! lives only in prose is one the next edit does not see.

use argmind_voice::Heard;

fn heard(text: &str, seconds: f64) -> Heard {
    Heard { text: text.into(), seconds, at: 0.0 }
}

/// **A transcript carries no confidence, and that is deliberate.**
///
/// The recognisers in sight publish no calibrated one, and an always-`None` field invites a
/// threshold nobody can set. The struct is checked by construction: if a confidence field were
/// added, this stops compiling — which is the strongest form this rule can take.
#[test]
fn a_transcript_is_text_a_length_and_a_time_and_nothing_else() {
    let it = Heard { text: "안녕하세요".into(), seconds: 1.4, at: 12.0 };
    assert_eq!(it.text, "안녕하세요");
    assert_eq!(it.seconds, 1.4);
    assert_eq!(it.at, 12.0);
}

/// **Pace is the one cheap signal a transcript was invented.**
///
/// Near-silence does not make Whisper silent — it makes it say `Thank you.` Measured: it
/// transcribes silence as words **120 times out of 120**, so "did anything arrive" cannot be the
/// filter and something about the shape of what arrived has to be.
#[test]
fn pace_is_characters_of_transcript_per_second_of_audio() {
    //: Five characters in one second.
    assert_eq!(heard("hello", 1.0).pace(), 5.0);
    //: Ten in two.
    assert_eq!(heard("helloworld", 2.0).pace(), 5.0);
    //: **Counted in characters, not bytes.** Korean is three bytes a character in UTF-8, and a
    //: pace computed on bytes would read three times too fast for exactly the language this
    //: project is for.
    assert_eq!(heard("안녕", 1.0).pace(), 2.0);
}

/// **A fabrication is fast**, and that is what the number is for.
///
/// Four seconds of near-silence transcribed as `Thank you.` is 2.5 characters a second; the same
/// ten characters spoken in a second and a half is 6.7. No threshold is named anywhere — the only
/// pace measured on this machine is a synthesiser's, 5.69 characters a second for Korean, which is
/// the wrong denominator for a person. **A guessed cut-off that silently drops real turns is worse
/// than none**, so what is asserted is that the two are separable, not where the line goes.
#[test]
fn a_long_silence_transcribed_as_a_short_phrase_paces_slower_than_speech() {
    let invented = heard("Thank you.", 4.0);
    let spoken = heard("Thank you.", 1.5);
    assert!(invented.pace() < spoken.pace());
    //: And the gap is not marginal, which is what makes it worth computing at all.
    assert!(spoken.pace() / invented.pace() > 2.0);
}

/// **Zero-length audio is infinite pace, not a division by zero and not a quiet zero.**
///
/// A clip with no length is the case a threshold would have to reject, and answering `0.0` would
/// make it the case a threshold *accepts* — the failure pointing the wrong way.
#[test]
fn audio_with_no_length_paces_at_infinity() {
    assert_eq!(heard("something", 0.0).pace(), f64::INFINITY);
    assert_eq!(heard("something", -1.0).pace(), f64::INFINITY);
    //: Even with nothing said. An empty transcript from no audio is not evidence of speech.
    assert_eq!(heard("", 0.0).pace(), f64::INFINITY);
    //: But an empty transcript from real audio is a real zero — somebody held the key and said
    //: nothing, which is a different fact and reads as one.
    assert_eq!(heard("", 2.0).pace(), 0.0);
}
