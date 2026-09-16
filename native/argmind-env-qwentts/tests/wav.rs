//! Reading what the synthesiser sends back, **with no server and no speaker.**
//!
//! The wire is exercised by hand — `cargo run -p argmind-env-qwentts --example hello`, which makes
//! a sound — and the numbers from that are in `docs/findings/a-better-voice.md`. What a suite can
//! check is the part that is wrong silently: a header read as forty-four fixed bytes when it is
//! not, a rate assumed rather than taken, and a stereo stream read as mono, which plays everything
//! at half speed.

use argmind_env_qwentts::{escape, from_wav, HZ};

/// A 16-bit PCM WAV around these samples, with an optional extra chunk before `data`.
fn wav(samples: &[i16], hz: u32, channels: u16, extra: Option<(&[u8; 4], &[u8])>) -> Vec<u8> {
    let mut out = Vec::new();
    let data: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
    let mut body = Vec::new();
    body.extend_from_slice(b"WAVE");
    body.extend_from_slice(b"fmt ");
    body.extend_from_slice(&16u32.to_le_bytes());
    body.extend_from_slice(&1u16.to_le_bytes());
    body.extend_from_slice(&channels.to_le_bytes());
    body.extend_from_slice(&hz.to_le_bytes());
    body.extend_from_slice(&(hz * channels as u32 * 2).to_le_bytes());
    body.extend_from_slice(&(channels * 2).to_le_bytes());
    body.extend_from_slice(&16u16.to_le_bytes());
    if let Some((kind, payload)) = extra {
        body.extend_from_slice(kind);
        body.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        body.extend_from_slice(payload);
        if payload.len() % 2 == 1 {
            body.push(0);
        }
    }
    body.extend_from_slice(b"data");
    body.extend_from_slice(&(data.len() as u32).to_le_bytes());
    body.extend_from_slice(&data);

    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(body.len() as u32).to_le_bytes());
    out.extend_from_slice(&body);
    out
}

/// The ordinary answer: mono, 24 kHz, sixteen bits.
#[test]
fn a_plain_mono_answer_comes_back_as_samples_and_its_rate() {
    let (samples, hz) = from_wav(&wav(&[0, 16384, -16384, 32767], HZ, 1, None)).expect("a wav");
    assert_eq!(hz, HZ);
    assert_eq!(samples.len(), 4);
    assert_eq!(samples[0], 0.0);
    assert!((samples[1] - 0.5).abs() < 0.001, "{}", samples[1]);
    assert!((samples[2] + 0.5).abs() < 0.001, "{}", samples[2]);
    //: Scaled by 32768 rather than 32767, so nothing can come back above one and clip on the way
    //: out.
    assert!(samples[3] < 1.0 && samples[3] > 0.999);
}

/// **The rate is taken from the header, not assumed.**
///
/// The one way this could be wrong is a server quietly changing what it emits, and audio at the
/// wrong rate is not a failure — it plays, faster or slower, and sounds like a different person.
#[test]
fn the_rate_is_whatever_the_header_says() {
    for hz in [8_000, 16_000, 22_050, HZ, 44_100, 48_000] {
        let (_, got) = from_wav(&wav(&[0, 1, 2, 3], hz, 1, None)).expect("a wav");
        assert_eq!(got, hz);
    }
}

/// **A chunk before `data` does not become audio.**
///
/// Forty-four bytes is the usual header size and it is not the only one: a `LIST` chunk carrying a
/// name or a date is legal and common. Skipping a fixed forty-four would play that chunk's bytes as
/// a click and shift everything after it.
#[test]
fn a_list_chunk_before_the_data_is_stepped_over() {
    let quiet = [0i16, 0, 0, 0];
    let plain = from_wav(&wav(&quiet, HZ, 1, None)).expect("a wav");
    let padded = from_wav(&wav(&quiet, HZ, 1, Some((b"LIST", b"INFOargmind"))))
        .expect("a wav with a LIST");
    assert_eq!(plain, padded);
    //: An odd-length chunk carries a pad byte that is not counted in its size. Getting that wrong
    //: shifts everything by one byte, which turns silence into noise.
    let odd = from_wav(&wav(&quiet, HZ, 1, Some((b"LIST", b"odd"))))
        .expect("a wav with an odd LIST");
    assert_eq!(plain, odd);
}

/// **Stereo is mixed down, not read as mono.** Reading it as mono halves the speed.
#[test]
fn two_channels_become_one_rather_than_twice_as_many_samples() {
    //: Left and right that average to a known value, so a channel being dropped shows up.
    let stereo = [1000i16, 3000, -1000, -3000];
    let (samples, _) = from_wav(&wav(&stereo, HZ, 2, None)).expect("a wav");
    assert_eq!(samples.len(), 2, "two frames, not four samples");
    assert!((samples[0] - 2000.0 / 32768.0).abs() < 0.001, "{}", samples[0]);
    assert!((samples[1] + 2000.0 / 32768.0).abs() < 0.001, "{}", samples[1]);
}

/// What is not a WAV is refused, rather than played.
#[test]
fn anything_that_is_not_a_wav_is_refused() {
    assert!(from_wav(b"").is_err());
    assert!(from_wav(b"{\"error\":{\"message\":\"unknown voice 'default'\"}}").is_err());
    //: A RIFF that is not a WAVE, and a WAVE with no `data`.
    let mut headerless = wav(&[1, 2], HZ, 1, None);
    headerless[8..12].copy_from_slice(b"AVI ");
    assert!(from_wav(&headerless).is_err());
    assert!(from_wav(&wav(&[], HZ, 1, None)).map(|(s, _)| s.len()) == Ok(0));
    //: Eight-bit audio is not handled and says so rather than playing at the wrong level.
    let mut eight = wav(&[1, 2], HZ, 1, None);
    //: `RIFF`(4) size(4) `WAVE`(4) `fmt `(4) size(4) = 20, then format(2) channels(2) rate(4)
    //: byte rate(4) block align(2) puts the bit depth at 34.
    eight[34..36].copy_from_slice(&8u16.to_le_bytes());
    assert!(from_wav(&eight).is_err());
}

/// The text goes onto the wire as JSON, and a quote in a reply does not end the request.
#[test]
fn what_the_agent_says_survives_being_put_in_json() {
    assert_eq!(escape("지우개"), "지우개");
    assert_eq!(escape("\"따옴표\""), "\\\"따옴표\\\"");
    assert_eq!(escape("한 줄\n다음 줄"), "한 줄\\n다음 줄");
    assert_eq!(escape("C:\\dev"), "C:\\\\dev");
    //: A control character that is not one of the named ones still has to be escaped, or the
    //: server sees a malformed request and the agent goes quiet for a reason nobody can see.
    assert_eq!(escape("\u{7}"), "\\u0007");
}
