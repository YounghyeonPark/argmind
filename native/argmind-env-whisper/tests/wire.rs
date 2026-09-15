//! The bytes on the wire, and the transcript out of the answer. No server, no microphone.

use argmind_env_whisper::{multipart, transcript_of, wav};

/// Forty-four bytes of header, and **the length fields have to agree with the data**.
#[test]
fn a_wav_header_describes_the_audio_that_follows_it() {
    let out = wav(&[0.0, 0.5, -0.5], 16_000);
    assert_eq!(out.len(), 44 + 6);
    assert_eq!(&out[0..4], b"RIFF");
    assert_eq!(&out[8..12], b"WAVE");
    //: RIFF size is everything after the first eight bytes.
    assert_eq!(u32::from_le_bytes(out[4..8].try_into().unwrap()), (out.len() - 8) as u32);
    //: One channel, 16 kHz, sixteen bits.
    assert_eq!(u16::from_le_bytes(out[22..24].try_into().unwrap()), 1);
    assert_eq!(u32::from_le_bytes(out[24..28].try_into().unwrap()), 16_000);
    assert_eq!(u16::from_le_bytes(out[34..36].try_into().unwrap()), 16);
    //: Byte rate is 16 kHz times two bytes a sample, and a wrong one plays back at the wrong speed.
    assert_eq!(u32::from_le_bytes(out[28..32].try_into().unwrap()), 32_000);
    assert_eq!(&out[36..40], b"data");
    assert_eq!(u32::from_le_bytes(out[40..44].try_into().unwrap()), 6);
    //: An empty recording is still a valid file, not a truncated one.
    assert_eq!(wav(&[], 16_000).len(), 44);
}

/// **Clamped before scaling**, because a cast wraps where saturation was wanted.
#[test]
fn a_sample_outside_the_range_does_not_wrap_into_a_click() {
    let out = wav(&[1.5, -1.5], 16_000);
    let a = i16::from_le_bytes(out[44..46].try_into().unwrap());
    let b = i16::from_le_bytes(out[46..48].try_into().unwrap());
    //: Full scale either way — not a wrap to the opposite sign, which is a loud click mid-word.
    assert_eq!(a, 32767);
    assert_eq!(b, -32767);
    assert!(a > 0 && b < 0);
}

#[test]
fn the_multipart_body_carries_the_audio_and_its_fields() {
    let body = multipart("BOUND", b"AUDIODATA", &[("language", "ko"), ("temperature", "0")]);
    let text = String::from_utf8_lossy(&body);
    assert!(text.contains("--BOUND\r\nContent-Disposition: form-data; name=\"language\""));
    assert!(text.contains("\r\n\r\nko\r\n"));
    assert!(text.contains("name=\"file\"; filename=\"a.wav\""));
    assert!(text.contains("Content-Type: audio/wav"));
    //: The audio arrives whole and the body is properly closed.
    assert!(body.windows(9).any(|w| w == b"AUDIODATA"));
    assert!(text.ends_with("\r\n--BOUND--\r\n"));
}

/// **Korean arrives escaped**, which is the case a substring search would get wrong.
#[test]
fn a_transcript_comes_back_out_of_the_answer() {
    assert_eq!(transcript_of(r#"{"text":"hello"}"#).as_deref(), Some("hello"));
    //: Whitespace around it is the server's, not the person's.
    assert_eq!(transcript_of(r#"{"text":"  hello  "}"#).as_deref(), Some("hello"));
    //: A server with `ensure_ascii` sends `\uXXXX`, and reading those wrongly turns every Korean
    //: turn into mojibake — silently, since the shape of the answer is unchanged.
    assert_eq!(transcript_of(r#"{"text":"지우개"}"#).as_deref(), Some("지우개"));
    //: And a server that does not escape sends the characters directly.
    assert_eq!(transcript_of(r#"{"text":"지우개 골라줘"}"#).as_deref(), Some("지우개 골라줘"));
    //: Escapes inside the sentence, including a quote that must not end it early.
    assert_eq!(transcript_of(r#"{"text":"a \"b\" c"}"#).as_deref(), Some(r#"a "b" c"#));
    assert_eq!(transcript_of(r#"{"text":"one\ntwo"}"#).as_deref(), Some("one\ntwo"));
    //: Fields either side, in either order.
    assert_eq!(transcript_of(r#"{"task":"transcribe","text":"ok","x":1}"#).as_deref(), Some("ok"));
}

/// **An unterminated string is not a transcript.**
///
/// Handing back what was read so far would return half a sentence and call it what the person
/// said — and a truncated answer is exactly what a server killed mid-response produces.
#[test]
fn half_an_answer_is_refused_rather_than_returned() {
    assert_eq!(transcript_of(r#"{"text":"unterminated"#), None);
    assert_eq!(transcript_of(r#"{"text":"bad escape \u12"#), None);
    assert_eq!(transcript_of(r#"{"error":"no model loaded"}"#), None);
    assert_eq!(transcript_of(""), None);
    assert_eq!(transcript_of("<html>500</html>"), None);
    //: An empty transcript is a real answer — the person said nothing the recogniser could hear —
    //: and it is not the same as a broken one.
    assert_eq!(transcript_of(r#"{"text":""}"#).as_deref(), Some(""));
}
