//! The audio arithmetic, with nothing opened.
//!
//! **No test here touches a microphone.** What is checked is the part between the device and the
//! recogniser — channel mixing, rate conversion, and the floor under what counts as an utterance —
//! because that is where a wrong answer is silent rather than loud: a recording that is half as
//! long as it should be, or one channel of a headset that puts speech on the other, both sound like
//! a recogniser having a bad day.

use argmind_env_windows::microphone::{from_i16, mono, resample, Clip, FLOOR_SECONDS, WANT_HZ};
use argmind_env_windows::Microphone;

/// **Averaged, not "take the left channel".**
///
/// A headset that puts the microphone on one channel and silence on the other is common, and
/// dropping a channel throws away half the recordings on that hardware — while sounding perfectly
/// correct on the machine it was written on.
#[test]
fn a_headset_that_speaks_on_one_channel_is_still_heard() {
    //: Two frames of stereo, speech on the left and silence on the right.
    let got = mono(&[1.0, 0.0, -1.0, 0.0], 2);
    assert_eq!(got, vec![0.5, -0.5]);
    assert!(got.iter().any(|s| *s != 0.0), "the speech survived the mix");

    //: And on the right instead — the same recording, the same answer.
    assert_eq!(mono(&[0.0, 1.0, 0.0, -1.0], 2), vec![0.5, -0.5]);
    //: Mono in, mono out, untouched.
    assert_eq!(mono(&[0.3, -0.3], 1), vec![0.3, -0.3]);
    assert_eq!(mono(&[], 2), Vec::<f32>::new());
}

/// The length in seconds has to survive the conversion, or every duration downstream is wrong.
#[test]
fn a_second_of_audio_is_still_a_second_afterwards() {
    for from in [8_000u32, 16_000, 22_050, 44_100, 48_000] {
        let one_second: Vec<f32> = (0..from).map(|i| (i as f32 / 100.0).sin()).collect();
        let out = resample(&one_second, from);
        let seconds = out.len() as f64 / WANT_HZ as f64;
        //: Within a sample either way — the floor in `resample` can drop the last partial one.
        assert!((seconds - 1.0).abs() < 0.001, "{from} Hz became {seconds:.4} s");
    }
    //: The rate the recogniser wants is passed through untouched rather than run through the
    //: interpolator for nothing.
    let same: Vec<f32> = (0..1000).map(|i| i as f32).collect();
    assert_eq!(resample(&same, WANT_HZ), same);
    assert_eq!(resample(&[], 48_000), Vec::<f32>::new());
}

/// Downsampling keeps the shape: a slow ramp is still a ramp.
#[test]
fn resampling_keeps_the_signal_rather_than_scrambling_it() {
    let ramp: Vec<f32> = (0..48_000).map(|i| i as f32 / 48_000.0).collect();
    let out = resample(&ramp, 48_000);
    assert_eq!(out.len(), 16_000);
    //: Still monotonic, still spanning the same range.
    assert!(out.windows(2).all(|w| w[1] >= w[0]));
    assert!(out[0] < 0.001);
    assert!(*out.last().unwrap() > 0.99);
}

/// **Divided by 32768, so full-scale negative is exactly −1.0** and nothing leaves the range a
/// recogniser expects.
#[test]
fn integer_samples_land_inside_the_range() {
    let got = from_i16(&[0, i16::MAX, i16::MIN, -16384]);
    assert_eq!(got[0], 0.0);
    assert_eq!(got[2], -1.0);
    assert_eq!(got[3], -0.5);
    assert!(got[1] < 1.0 && got[1] > 0.9999);
    assert!(got.iter().all(|s| (-1.0..=1.0).contains(s)));
}

/// **A tap on the talk key is not a sentence.**
///
/// A recogniser handed a few milliseconds of room noise does not answer "nothing" — Whisper answers
/// `Thank you.` 120 times out of 120, which is the measured reason this floor exists at all.
#[test]
fn a_tap_on_the_key_is_not_an_utterance() {
    let tap = Clip::new(vec![0.0; (WANT_HZ as f64 * 0.05) as usize]);
    assert!(!tap.worth_hearing());
    assert!((tap.seconds - 0.05).abs() < 0.001);

    let said = Clip::new(vec![0.1; (WANT_HZ as f64 * 1.2) as usize]);
    assert!(said.worth_hearing());
    assert!((said.seconds - 1.2).abs() < 0.001);

    //: Exactly at the floor counts — the boundary is inclusive and stated, not discovered.
    let edge = Clip::new(vec![0.0; (WANT_HZ as f64 * FLOOR_SECONDS) as usize]);
    assert!(edge.worth_hearing());
    assert!(!Clip::new(Vec::new()).worth_hearing());
}

/// Peak is **reported and gates nothing**.
#[test]
fn loudness_is_measured_and_not_yet_a_threshold() {
    let quiet = Clip::new(vec![0.001, -0.002, 0.0]);
    assert!((quiet.peak() - 0.002).abs() < 1e-6);
    let loud = Clip::new(vec![0.0, -0.9, 0.4]);
    assert!((loud.peak() - 0.9).abs() < 1e-6);
    //: **A silent clip long enough to pass the floor still passes it.** The threshold that would
    //: reject it needs real recorded speech to set, and that recording does not exist yet — so
    //: peak is a number on the page and not a gate, which is the honest state rather than a
    //: guessed constant.
    let silence = Clip::new(vec![0.0; WANT_HZ as usize]);
    assert_eq!(silence.peak(), 0.0);
    assert!(silence.worth_hearing());
}

/// Holding and letting go, with no device present.
#[test]
fn a_microphone_that_never_opened_produces_nothing_rather_than_failing() {
    let mut m = Microphone::new();
    assert!(!m.held());
    m.hold(true);
    assert!(m.held());
    m.hold(false);
    assert!(!m.held());
    //: Nothing was recorded, so nothing is offered — and the short-clip counter says which of the
    //: two silences this was.
    assert!(m.take().is_none());
    assert_eq!(m.clips, 0);
    assert_eq!(m.too_short, 1, "an empty hold is a tap, and taps are counted");
    //: Letting go twice is not two clips.
    m.hold(false);
    assert_eq!(m.too_short, 1);
    m.release();
}
