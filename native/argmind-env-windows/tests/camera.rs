//! The camera's arithmetic, **with no camera**.
//!
//! Opening one turns a light on in front of a person, so a suite must not. What a suite can check
//! is the part that would be wrong silently: the reduction from a megapixel frame to the small grey
//! field, and the two scalars everything downstream reads. The device itself is measured by hand —
//! `cargo run -p argmind-env-windows --example look` — and those numbers are in the module.

use argmind_env_windows::camera::{reduce, Frame, SETTLED, TALL, WIDE};

fn frame(grey: Vec<u8>) -> Frame {
    Frame { grey, took: 0.0 }
}

/// A picture of one colour reduces to that grey, whatever size it came in at.
#[test]
fn a_flat_frame_reduces_to_a_flat_field() {
    for (w, h) in [(1920, 1080), (640, 480), (WIDE, TALL), (17, 13)] {
        //: BGRA. Mid grey in, mid grey out — and the luma weights sum to 256, so a grey stays put
        //: rather than drifting a level per conversion.
        let bgra = vec![128u8; w * h * 4];
        let small = reduce(&bgra, w, h, w * 4);
        assert_eq!(small.len(), WIDE * TALL, "{w}x{h}");
        assert!(small.iter().all(|g| (*g as i32 - 128).abs() <= 1),
                "{w}x{h} gave {:?}…", &small[..4]);
    }
}

/// **The green channel carries the detail and the weights say so.**
///
/// Weighting the three equally is the easy mistake, and it makes a green shirt and a blue wall the
/// same grey.
#[test]
fn the_channels_are_not_weighted_equally() {
    let one = |b: u8, g: u8, r: u8| {
        let mut px = Vec::new();
        for _ in 0..WIDE * TALL {
            px.extend_from_slice(&[b, g, r, 255]);
        }
        reduce(&px, WIDE, TALL, WIDE * 4)[0]
    };
    let (red, green, blue) = (one(0, 0, 255), one(0, 255, 0), one(255, 0, 0));
    assert!(green > red && red > blue, "green {green}, red {red}, blue {blue}");
    //: Pure blue is the darkest by a long way — about a ninth of white.
    assert!(blue < 40, "blue {blue}");
}

/// **A padded row does not shear the picture.**
///
/// A capture buffer's stride is not always the width times four, and reading it as if it were
/// slides every row sideways by a little more than the last. The symptom is a diagonal smear that
/// looks like motion, in a still room.
#[test]
fn the_stride_is_read_rather_than_assumed() {
    let (w, h) = (100, 50);
    let stride = w * 4 + 64;
    let mut padded = vec![0u8; stride * h];
    //: The left half bright, the right half dark, with junk in the padding.
    for y in 0..h {
        for x in 0..w {
            let v = if x < w / 2 { 220 } else { 20 };
            let at = y * stride + x * 4;
            padded[at] = v;
            padded[at + 1] = v;
            padded[at + 2] = v;
        }
        for b in 0..64 {
            padded[y * stride + w * 4 + b] = 255;
        }
    }
    let small = reduce(&padded, w, h, stride);
    //: Still a clean left/right split — the padding stayed out of it, and every row agrees.
    for y in 0..TALL {
        assert!(small[y * WIDE] > 180, "row {y} left");
        assert!(small[y * WIDE + WIDE - 1] < 60, "row {y} right");
    }
}

/// Averaged, not sampled — **the whole reason the small size filters anything.**
#[test]
fn a_block_is_averaged_rather_than_sampled() {
    //: A one-pixel-wide bright stripe every other column. Sampling would give either all bright or
    //: all dark depending on where it landed; averaging gives the middle.
    let (w, h) = (WIDE * 4, TALL * 4);
    let mut striped = vec![0u8; w * h * 4];
    for y in 0..h {
        for x in 0..w {
            if x % 2 == 0 {
                let at = (y * w + x) * 4;
                striped[at] = 255;
                striped[at + 1] = 255;
                striped[at + 2] = 255;
            }
        }
    }
    let small = reduce(&striped, w, h, w * 4);
    assert!(small.iter().all(|g| (90..=165).contains(g)),
            "a sampled reduction would be 0 or 255, got {:?}…", &small[..6]);
}

/// The two scalars, and what they refuse to be.
#[test]
fn a_difference_is_a_number_and_never_a_place() {
    let dark = frame(vec![10; WIDE * TALL]);
    let light = frame(vec![210; WIDE * TALL]);
    assert_eq!(dark.brightness(), 10.0);
    assert_eq!(light.differs_from(&dark), 200.0);
    //: Symmetric, because *how different* has no direction.
    assert_eq!(dark.differs_from(&light), 200.0);
    assert_eq!(dark.differs_from(&dark), 0.0);

    //: **A mean, so one changed corner is small.** Where it changed would be a coordinate, and a
    //: coordinate out of a detector is the thing this whole project refuses to produce — so the
    //: type has no way to say it.
    let mut corner = vec![10u8; WIDE * TALL];
    corner[0] = 255;
    assert!(frame(corner).differs_from(&dark) < 1.0);
}

/// Frames of different sizes are not compared. **Zero, not a panic and not a guess.**
#[test]
fn two_frames_that_are_not_the_same_size_differ_by_nothing() {
    assert_eq!(frame(vec![1; 10]).differs_from(&frame(vec![1; 20])), 0.0);
    assert_eq!(frame(Vec::new()).differs_from(&frame(Vec::new())), 0.0);
    assert_eq!(frame(Vec::new()).brightness(), 0.0);
}

/// **The settling threshold sits in the gap that was measured**, and the gap is what makes it a
/// threshold rather than a preference.
#[test]
fn settling_is_above_the_quiet_room_and_below_the_drift() {
    //: A quiet room read 0.14 to 0.34 in six runs of twenty-five, and up to 2.85 while the exposure
    //: was still walking. Half a unit is above the first and well below the second.
    assert!(SETTLED > 0.34, "{SETTLED} would call a quiet room unsettled");
    assert!(SETTLED < 2.85, "{SETTLED} would call a drifting camera settled");
}

/// A frame that arrived from nothing is a field of zeroes rather than a panic.
#[test]
fn an_empty_buffer_reduces_without_falling_over() {
    assert_eq!(reduce(&[], 0, 0, 0).len(), WIDE * TALL);
    assert_eq!(reduce(&[], 1920, 1080, 1920 * 4), vec![0u8; WIDE * TALL]);
    //: And a buffer that is shorter than its own geometry claims — a truncated capture — is read as
    //: far as it goes rather than off the end of it.
    assert_eq!(reduce(&[255u8; 64], 1920, 1080, 1920 * 4).len(), WIDE * TALL);
}

/// **One quiet pair is not a settled camera**, which is the defect this replaced.
///
/// `settle` used to return on the first consecutive pair under [`SETTLED`]. A warming camera offers
/// one within about five frames — measured 2026-09-22, it reported 329–352 ms on every open while
/// one-second windows went on reading 0.57–2.84 for most of ten seconds. Everything measured in
/// that window was measured on a camera the code had called ready, and `measure/presence` was
/// separating its two phases by *which one ran first*: 2.17 against 0.65 with nobody moving in
/// either, collapsing to 0.22 when the camera was already warm.
#[test]
fn a_single_quiet_frame_does_not_settle_a_drifting_camera() {
    use argmind_env_windows::camera::{quiet_enough, ENOUGH_TO_JUDGE};

    //: The shape that broke it: loud, with one quiet sample in the middle. The old rule returned
    //: `true` on that sample; a median does not notice it.
    let mut drifting = vec![2.8, 3.1, 0.2, 2.4, 1.9, 2.2, 3.0];
    assert_eq!(quiet_enough(&mut drifting), Some(false), "one quiet frame settled a drift");

    //: And the opposite has to still work, or the camera never reports settled at all.
    let mut quiet = vec![0.10, 0.12, 0.14, 0.11, 0.13, 0.12, 4.9];
    assert_eq!(quiet_enough(&mut quiet), Some(true), "one spike unsettled a quiet room");

    //: **Too few is not an answer.** Four frames say nothing, and returning `false` for them would
    //: be indistinguishable from a camera that is genuinely moving.
    let mut few = vec![0.1; ENOUGH_TO_JUDGE - 1];
    assert_eq!(quiet_enough(&mut few), None);
    let mut just_enough = vec![0.1; ENOUGH_TO_JUDGE];
    assert_eq!(quiet_enough(&mut just_enough), Some(true));
}

/// **The limit has to be able to contain the wait it is now limiting.**
///
/// Four seconds was the limit for a check that returned in 330 ms, so it never mattered. Cold, the
/// picture was still moving between three and seven seconds after opening and quiet by ten — n=1,
/// two runs — so a limit under that would report *never settled* on a camera that settles.
#[test]
fn the_limit_outlasts_the_settling_it_was_measured_against() {
    use argmind_env_windows::camera::{SETTLE_LIMIT, SETTLE_WINDOW};
    assert!(SETTLE_LIMIT > 10.0, "{SETTLE_LIMIT} s cannot contain a ten second settle");
    //: And it is enough windows to be a sequence of judgements rather than one.
    assert!(SETTLE_LIMIT / SETTLE_WINDOW >= 5.0);
}

/// **The movement floor sits in the gap that was measured, and the gap is two runs wide.**
///
/// Ten seconds still and ten seconds moving, twice, read as three-second medians with the mean
/// shift removed: the highest still window was 0.05 then 0.06, the lowest moving one 0.46 then
/// 0.41. A floor outside that is not a floor, and one chosen to make something pass is the failure
/// the standard of evidence here exists against.
#[test]
fn the_movement_floor_is_inside_the_measured_gap() {
    use argmind_env_windows::camera::{MOVING, MOVING_WINDOW, SETTLED};

    assert!(MOVING > 0.06, "{MOVING} would call a still room moving");
    assert!(MOVING < 0.41, "{MOVING} would call a moving person still");

    //: **Room on the side that varies.** The still readings sit in 0.01 of each other across two
    //: runs; the moving ones span 0.05. So the threshold is asked to be no closer to the moving
    //: edge than to the still one, in ratio — which is what putting it at the geometric middle
    //: means and what a midpoint would not have given.
    assert!(0.41 / MOVING >= MOVING / 0.06,
            "{MOVING} leaves less room on the moving side than on the still one");

    //: **The window is part of the number.** A median over one second is a different statistic and
    //: separated by half as much — 3.1x against 6.2x — so a threshold read against it would be a
    //: threshold against something else.
    assert_eq!(MOVING_WINDOW, 3.0);

    //: And it is well under the settling floor: a camera that has not settled moves more than a
    //: person does, which is why one is waited out before the other is believed.
    assert!(MOVING < SETTLED, "movement cannot be judged above the settling floor");
}
