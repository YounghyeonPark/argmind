//! The speaker's arithmetic, **with no speaker.**
//!
//! Opening the device makes a noise in somebody's room, so a suite must not. What a suite can check
//! is the resampling — the part that turns a synthesiser's 24 kHz mono into whatever the device
//! wants, and the part that is wrong silently: a rate ratio inverted gives audio at the wrong speed
//! and a channel count ignored gives a click track.
//!
//! The device numbers are measured by hand: `cargo run -p argmind-env-windows --example bargein`.

use argmind_env_windows::loudspeaker::{to_device, Loudspeaker, BUFFER, FROM_HZ};

/// **Up to 48 kHz stereo is the ordinary case** and it is the one to get right.
#[test]
fn a_second_at_24k_is_a_second_at_48k() {
    let one_second = vec![0.0f32; FROM_HZ as usize];
    let out = to_device(&one_second, FROM_HZ, 48_000, 2);
    //: Twice as many frames, two channels each.
    assert_eq!(out.len(), 48_000 * 2);
    //: And the same holds the other way, for a device that happens to run at 24 kHz.
    assert_eq!(to_device(&one_second, FROM_HZ, 24_000, 1).len(), 24_000);
    //: And for the odd rates that turn up on cheap hardware.
    assert_eq!(to_device(&one_second, FROM_HZ, 44_100, 2).len(), 44_100 * 2);
}

/// **The ratio is not inverted**, which is the mistake that plays everything at half speed.
#[test]
fn a_higher_device_rate_means_more_samples_not_fewer() {
    let quarter = vec![0.5f32; FROM_HZ as usize / 4];
    let slow = to_device(&quarter, FROM_HZ, 16_000, 1).len();
    let same = to_device(&quarter, FROM_HZ, 24_000, 1).len();
    let fast = to_device(&quarter, FROM_HZ, 48_000, 1).len();
    assert!(slow < same && same < fast, "{slow} {same} {fast}");
    assert_eq!(same, quarter.len());
}

/// Every channel gets the same mono signal — **not silence in one ear.**
#[test]
fn mono_is_spread_across_the_channels() {
    let ramp: Vec<f32> = (0..FROM_HZ as usize).map(|i| i as f32 / FROM_HZ as f32).collect();
    let out = to_device(&ramp, FROM_HZ, 24_000, 2);
    for frame in out.chunks(2) {
        assert_eq!(frame[0], frame[1], "the two channels disagree");
    }
    //: Six channels is a surround device and it must not be quieter for it.
    let six = to_device(&ramp, FROM_HZ, 24_000, 6);
    assert_eq!(six.len(), ramp.len() * 6);
    assert!(six.chunks(6).all(|f| f.iter().all(|v| *v == f[0])));
}

/// **The signal survives**, rather than being the right length and the wrong shape.
#[test]
fn what_goes_in_comes_out_at_the_same_level() {
    //: A slow ramp, so linear interpolation reproduces it closely and a scrambled one shows up.
    let ramp: Vec<f32> = (0..2400).map(|i| i as f32 / 2400.0).collect();
    let out = to_device(&ramp, FROM_HZ, 48_000, 1);
    assert_eq!(out.first().copied(), Some(0.0));
    assert!(out.last().copied().unwrap_or(0.0) > 0.99);
    //: Monotone in, monotone out.
    assert!(out.windows(2).all(|w| w[1] >= w[0] - 1e-6), "the ramp came back out of order");
}

/// Nothing in, nothing out — and no panic on the degenerate arguments.
#[test]
fn an_empty_or_impossible_request_is_empty_rather_than_a_panic() {
    assert!(to_device(&[], FROM_HZ, 48_000, 2).is_empty());
    assert!(to_device(&[1.0, 2.0], FROM_HZ, 0, 2).is_empty());
    assert!(to_device(&[1.0, 2.0], FROM_HZ, 48_000, 0).is_empty());
    //: One sample upsampled two to one is two frames, and the second interpolates against a
    //: neighbour that is not there — which falls back to the sample itself rather than reading
    //: past the end.
    assert_eq!(to_device(&[1.0], FROM_HZ, 48_000, 2).len(), 4);
    assert!(to_device(&[1.0], FROM_HZ, 48_000, 2).iter().all(|v| *v == 1.0));
}

/// **A new speaker is silent**, and says so.
#[test]
fn nothing_is_speaking_until_something_is_played() {
    let mouth = Loudspeaker::new();
    assert!(!mouth.speaking());
    assert_eq!(mouth.cut_off, 0);
    //: `available` is false until the device has actually been opened — it is not a claim made in
    //: advance.
    assert!(!mouth.available);
}

/// **Stopping something that was never playing is not an error**, and does not invent a cut-off.
#[test]
fn stopping_silence_counts_nothing() {
    let mut mouth = Loudspeaker::new();
    mouth.stop();
    mouth.stop();
    mouth.release();
    assert_eq!(mouth.cut_off, 0);
    assert!(!mouth.speaking());
}

/// **A source rate other than the synthesiser's** — the whole reason the rate travels with the
/// samples rather than being a constant.
#[test]
fn a_source_at_another_rate_is_stretched_by_its_own_ratio() {
    let half_second_at_16k = vec![0.0f32; 8_000];
    //: Half a second in, half a second out, whatever the two rates are.
    assert_eq!(to_device(&half_second_at_16k, 16_000, 48_000, 1).len(), 24_000);
    assert_eq!(to_device(&half_second_at_16k, 16_000, 16_000, 1).len(), 8_000);
    //: And reading it as if it were 24 kHz would give a different answer, which is the bug this
    //: argument exists to prevent.
    assert_ne!(to_device(&half_second_at_16k, 16_000, 48_000, 1).len(),
               to_device(&half_second_at_16k, FROM_HZ, 48_000, 1).len());
}

/// **The device buffer is the floor on how fast a stop can be**, so its size is a decision.
#[test]
fn the_buffer_is_short_enough_that_a_stop_is_not_heard_as_late() {
    let milliseconds = BUFFER as f64 / 10_000.0;
    //: Sixty. Above about a hundred a cut-off starts to read as a delay rather than as a stop;
    //: below about twenty the feeder thread starts to miss deadlines and the audio breaks up.
    assert!((20.0..=100.0).contains(&milliseconds), "{milliseconds} ms");
}
