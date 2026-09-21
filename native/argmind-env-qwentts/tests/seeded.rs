//! **Which voice, and the one way to get it wrong.**
//!
//! The talker is a *base* model: it has no speakers, `GET /v1/audio/voices` answers with an empty
//! list, and every name offered to `voice` comes back `unknown voice`. What decides the timbre is
//! the sampler, so the seed is not a tuning knob — it is the whole of *who is speaking*. Measured
//! 2026-09-21 against the running server: four unseeded calls with the same text gave four
//! different voices and a 21% spread in length; `seed = 7` gave the same bytes four times out of
//! four; seeds 8 and 42 gave two more people. `argmind-measure --bin voicing` is the harness.
//!
//! **The failure this file exists for is asymmetry.** There are two request paths — the streamed
//! one, which ships, and the whole-file one, which every measurement goes through. A seed carried
//! by only one of them is worse than no seed at all: the voice would be stable everywhere anybody
//! looked and random everywhere anybody listened.

use argmind_env_qwentts::asking;

/// The seed is in the body, as a number, whatever the tail field is.
#[test]
fn every_body_carries_the_seed() {
    for tail in ["\"response_format\":\"wav\"", "\"stream\":true"] {
        let body = asking("qwen3-tts", "안녕하세요", 7, tail);
        assert!(body.contains("\"seed\":7"), "{body}");
        assert!(body.contains(tail), "{body}");
        //: A quoted seed is a string, and the server refuses it — `opt_int` wants a number.
        assert!(!body.contains("\"seed\":\"7\""), "the seed went out as a string: {body}");
    }
}

/// **Neither path may build its own body.** Held by reading the source, like the four rules in
/// `argmind-rules`, and for the same reason: the compiler is perfectly happy with a second format
/// string, and a second format string is exactly how one of the two paths loses its seed.
#[test]
fn nothing_else_writes_a_request_body() {
    let source = include_str!("../src/lib.rs");
    //: An escaped `"input":` appears in a request body and nowhere else in this crate.
    let bodies = source.matches("\\\"input\\\":").count();
    assert_eq!(bodies, 1,
               "{bodies} places build a request body; there must be exactly one, and both `speak` \
                and `speak_streaming` must go through it");
    let (_, after) = source.split_once("pub fn asking(").expect("asking is gone");
    assert!(after[..300].contains("\\\"input\\\":"), "the one body is not the one in `asking`");
}

/// The text is escaped on the way in. **A reply with a quote in it is the ordinary case**, not an
/// attack, and an unescaped one is a 400 from the server rather than a wrong voice.
#[test]
fn awkward_text_still_makes_one_json_object() {
    let said = "그가 \"저장\" 을 눌렀고\n경로는 C:\\dev 입니다";
    let body = asking("qwen3-tts", said, 42, "\"stream\":true");
    assert!(body.contains("\\\"저장\\\""), "{body}");
    assert!(body.contains("\\n"), "{body}");
    assert!(body.contains("C:\\\\dev"), "{body}");
    assert!(!body.contains('\n'), "a real newline went into the body: {body:?}");
}

/// A negative seed is what the server calls *choose for me*, and argmind never asks for that.
///
/// **The type allows it and the settings do not** — `chosen::voice()` cannot return one — so this
/// records the meaning rather than guarding a path. It is here because the next person to read
/// `-1` in a log should know it means the voice was not chosen.
#[test]
fn minus_one_is_the_servers_word_for_random() {
    assert!(asking("qwen3-tts", "안녕", -1, "\"stream\":true").contains("\"seed\":-1"));
}
