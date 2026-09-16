//! How long it takes to start speaking, and **how long it takes to stop.** Makes a sound.
//!
//!     cargo run -p argmind-env-windows --example bargein
//!
//! The second number is the one that matters. A median reply is about three seconds of speech
//! against a 0.61 s model call, so a voice that cannot be interrupted turns a conversation into a
//! lecture — and *stop* is bounded below by whatever the device is already holding.
//!
//! A tone rather than speech, because this is measuring the plumbing and a tone makes a cut-off
//! audible: it either stops when it is told to or it does not.

use std::time::Instant;

use argmind_env_windows::loudspeaker::{Loudspeaker, FROM_HZ};

/// Three seconds of a quiet 440 Hz tone, with a short fade in and out so the start and the end are
/// not clicks.
fn tone(seconds: f64) -> Vec<f32> {
    let n = (FROM_HZ as f64 * seconds) as usize;
    let fade = FROM_HZ as usize / 100;
    (0..n).map(|i| {
        let t = i as f64 / FROM_HZ as f64;
        let level = 0.12
            * (i.min(fade) as f64 / fade as f64).min((n - i).min(fade) as f64 / fade as f64);
        (level * (t * 440.0 * std::f64::consts::TAU).sin()) as f32
    }).collect()
}

fn main() {
    let mut mouth = Loudspeaker::new();

    println!("\n  3초짜리 음을 재생합니다 — 소리가 납니다.");
    let began = Instant::now();
    mouth.play(&tone(3.0));
    //: **Started is not the same as audible.** What is measured is how long until the thread says
    //: it is speaking, which is when the first samples have gone to the device.
    while !mouth.speaking() && began.elapsed().as_secs_f64() < 2.0 {
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    println!("  play() -> 소리 나가기까지: {:.0} ms", began.elapsed().as_secs_f64() * 1000.0);

    std::thread::sleep(std::time::Duration::from_millis(1200));

    println!("  이제 끊습니다 — 여기서 소리가 멎어야 합니다.");
    let cut = Instant::now();
    mouth.stop();
    while mouth.speaking() && cut.elapsed().as_secs_f64() < 2.0 {
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    println!("  stop() -> 멎기까지: {:.0} ms  (장치가 들고 있던 것 포함 최대 {:.0} ms 더)",
             cut.elapsed().as_secs_f64() * 1000.0,
             argmind_env_windows::loudspeaker::BUFFER as f64 / 10_000.0);

    //: **Barge-in is the second utterance arriving, not silence.** The one that matters in use is
    //: `play` over the top of something already speaking, so it is measured too.
    println!("\n  다시 3초를 재생하고, 1초 뒤에 다른 것으로 덮어씁니다.");
    mouth.play(&tone(3.0));
    std::thread::sleep(std::time::Duration::from_millis(1000));
    let over = Instant::now();
    mouth.play(&tone(1.0));
    println!("  play() 덮어쓰기: {:.0} ms 만에 돌아옴, 끊긴 횟수 {}",
             over.elapsed().as_secs_f64() * 1000.0, mouth.cut_off);
    std::thread::sleep(std::time::Duration::from_millis(1400));
    mouth.release();
    println!("  끝\n");
}
