//! **Who is speaking, and the two ways of getting it wrong.**
//!
//! The talker is a *base* model: it ships with no speakers, `GET /v1/audio/voices` starts empty and
//! every built-in name comes back `unknown voice`. What decides the timbre is the sampler, so every
//! request has to say something about who is talking, and there are two things it can say.
//!
//! **A seed fixes a recording.** Measured 2026-09-21: four unseeded calls with one sentence gave
//! four different voices and a 21% spread in length; `seed = 7` gave the same bytes four times out
//! of four. But across *four different sentences* a seed held to 119 Hz of itself while different
//! seeds on one sentence spanned 137 Hz — 1.2x, which is no ratio — and long-term average spectra
//! were no more alike within a seed (0.821) than across seeds (0.813). Every reply is a different
//! sentence, so a seed alone is a different person every turn.
//!
//! **A cloned voice fixes a person.** The clone's pitch tracks its reference at +0.99, and it holds
//! to a median 26 Hz across sentences. `argmind-measure --bin voicing` is the harness for both.
//!
//! **The failure this file exists for is asymmetry.** There are two request paths — the streamed
//! one, which ships, and the whole-file one, which every measurement goes through. A voice carried
//! by only one of them is worse than none: it would be stable everywhere anybody looked and random
//! everywhere anybody listened.

use argmind_env_qwentts::{asking, Voice};

/// Both kinds put something in the body, and it is the right shape for each.
#[test]
fn every_body_says_who_is_speaking() {
    for tail in ["\"response_format\":\"wav\"", "\"stream\":true"] {
        let drawn = asking("qwen3-tts", "안녕하세요", &Voice::Drawn(7), tail);
        assert!(drawn.contains("\"seed\":7"), "{drawn}");
        //: A quoted seed is a string, and the server refuses it — `opt_int` wants a number.
        assert!(!drawn.contains("\"seed\":\"7\""), "the seed went out as a string: {drawn}");
        assert!(drawn.contains(tail), "{drawn}");

        let cloned = asking("qwen3-tts", "안녕하세요", &Voice::Cloned("argmind-low".into()), tail);
        assert!(cloned.contains("\"voice\":\"argmind-low\""), "{cloned}");
        //: **And no seed beside it.** A registered voice and a seed in one body is two answers to
        //: one question, and which one wins is the server's business rather than something to
        //: guess at.
        assert!(!cloned.contains("\"seed\""), "a cloned voice was sent with a seed: {cloned}");
        assert!(cloned.contains(tail), "{cloned}");
    }
}

/// **Neither path may build its own body.** Held by reading the source, like the rules in
/// `argmind-rules`, and for the same reason: the compiler is perfectly happy with a second format
/// string, and a second format string is exactly how one of the two paths loses its voice.
#[test]
fn nothing_else_writes_a_request_body() {
    let source = include_str!("../src/lib.rs");
    //: An escaped `"input":` appears in a speech request and nowhere else in this crate. The voice
    //: registration body is a different endpoint and has no `input` in it.
    let bodies = source.matches("\\\"input\\\":").count();
    assert_eq!(bodies, 1,
               "{bodies} places build a speech body; there must be exactly one, and both `speak` \
                and `speak_streaming` must go through it");
    let (_, after) = source.split_once("pub fn asking(").expect("asking is gone");
    assert!(after[..300].contains("\\\"input\\\":"), "the one body is not the one in `asking`");
}

/// The text is escaped on the way in. **A reply with a quote in it is the ordinary case**, not an
/// attack, and an unescaped one is a 400 from the server rather than a wrong voice.
#[test]
fn awkward_text_still_makes_one_json_object() {
    let said = "그가 \"저장\" 을 눌렀고\n경로는 C:\\dev 입니다";
    let body = asking("qwen3-tts", said, &Voice::Drawn(42), "\"stream\":true");
    assert!(body.contains("\\\"저장\\\""), "{body}");
    assert!(body.contains("\\n"), "{body}");
    assert!(body.contains("C:\\\\dev"), "{body}");
    assert!(!body.contains('\n'), "a real newline went into the body: {body:?}");

    //: **And a voice name too**, which is argmind's own and still goes through the escaper — a
    //: rule that only covers the text is a rule that a renamed preset walks through.
    let odd = asking("qwen3-tts", "안녕", &Voice::Cloned("a\"b".into()), "\"stream\":true");
    assert!(odd.contains("\\\"voice\\\":\\\"a\\\\\\\"b\\\"") || odd.contains("a\\\"b"), "{odd}");
}

/// A negative seed is what the server calls *choose for me*, and argmind never asks for that.
///
/// **The type allows it and the settings do not** — `chosen::voice()` cannot produce one — so this
/// records the meaning rather than guarding a path. It is here because the next person to read
/// `-1` in a log should know it means nothing was chosen.
#[test]
fn minus_one_is_the_servers_word_for_random() {
    assert!(asking("qwen3-tts", "안녕", &Voice::Drawn(-1), "\"stream\":true")
            .contains("\"seed\":-1"));
}

/// **The reference recording has to survive being turned into base64**, which this crate writes
/// itself rather than take a dependency for.
///
/// Checked against values anybody can verify by hand rather than against another implementation of
/// the same idea, and the awkward cases are the tail: three bytes fill a quad, two leave one `=`,
/// one leaves two.
#[test]
fn a_reference_recording_survives_being_encoded() {
    use argmind_env_qwentts::encoded;
    assert_eq!(encoded(b""), "");
    assert_eq!(encoded(b"f"), "Zg==");
    assert_eq!(encoded(b"fo"), "Zm8=");
    assert_eq!(encoded(b"foo"), "Zm9v");
    assert_eq!(encoded(b"foob"), "Zm9vYg==");
    assert_eq!(encoded(b"fooba"), "Zm9vYmE=");
    assert_eq!(encoded(b"foobar"), "Zm9vYmFy");
    //: **The whole byte range**, because a WAV is not text and the bytes that break a hand-written
    //: encoder are the high ones.
    let all: Vec<u8> = (0..=255u8).collect();
    let out = encoded(&all);
    assert_eq!(out.len(), 344);
    assert!(out.starts_with("AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8gISIjJCUmJygpKissLS4v"), "{out}");
    assert!(out.ends_with("8PHy8/T19vf4+fr7/P3+/w=="), "{out}");
}
