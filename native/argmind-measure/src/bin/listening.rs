//! Can a threshold tell an empty room from somebody talking?
//!
//! ```text
//!   listening            10 s of quiet, then 10 s of talking, then 10 s of quiet again
//!   listening 5          five of each, while working
//! ```
//!
//! **This opens the microphone.** It is the only harness in this directory that does, it says so
//! before each phase, and it records nothing to disk but numbers — the samples are reduced to one
//! peak and one RMS per 30 ms frame inside this process and the audio itself is dropped.
//!
//! ## Why it exists
//!
//! The owner, 2026-09-17: **항상 켜진 상태로 듣고 있으면 안되나.**
//!
//! Push-to-talk is not a placeholder here, it is load-bearing, and the reason is a measured one:
//! **Whisper transcribes silence as words 120 times out of 120** — handed a few milliseconds of
//! room tone it answers `Thank you.` rather than nothing. The talk key is what stops that reaching
//! the model, so an always-open microphone needs something else in that place: a gate that decides
//! *this is speech* before the recogniser is asked anything.
//!
//! `Clip::peak` was written for that gate and deliberately left unused. Its own comment says why:
//!
//! > Not a gate. The threshold that would make it one needs real recorded speech to set, and that
//! > recording does not exist yet; until it does this is reported and nothing is rejected by it.
//!
//! **This makes that recording.** Nothing about always-on listening can be decided honestly before
//! it exists, and a threshold picked by taste would be a number that flatters whoever picked it.
//!
//! ## What it answers, and what it cannot
//!
//! It answers: *on this machine, in this room, with this microphone, is there a level that
//! separates quiet from speech, and what does it cost in each direction?* Both error rates are
//! reported because either alone is meaningless — a gate that never opens rejects every word and
//! scores perfectly on false alarms.
//!
//! It cannot answer: whether a **different** room works, whether a phone call across the desk
//! reads as speech, or whether argmind hearing its own voice is separable this way. Those are
//! different measurements and two of them are the harder half of the question.

use std::io::Write;
use std::path::PathBuf;

use argmind_env_windows::Microphone;
use argmind_measure::Destination;
use serde_json::{json, Value};

/// How long a frame is. **30 ms**, which is what voice-activity work conventionally uses and is
/// short enough that the gap between two words is visible as its own frame.
pub const FRAME_MS: usize = 30;
pub const HZ: u32 = argmind_env_windows::microphone::WANT_HZ;

/// One frame reduced to two numbers.
struct Frame {
    peak: f32,
    rms: f32,
}

fn frames(samples: &[f32]) -> Vec<Frame> {
    let n = HZ as usize * FRAME_MS / 1000;
    samples.chunks(n).filter(|c| c.len() == n).map(|c| {
        let peak = c.iter().fold(0.0f32, |a, s| a.max(s.abs()));
        let rms = (c.iter().map(|s| s * s).sum::<f32>() / c.len() as f32).sqrt();
        Frame { peak, rms }
    }).collect()
}

/// dBFS, because a level is a ratio and reading these as linear floats hides the whole question:
/// speech sits two orders of magnitude above room tone and a linear axis shows one flat line.
fn db(x: f32) -> f32 {
    if x <= 0.0 { -120.0 } else { 20.0 * x.log10() }
}

fn quantiles(mut v: Vec<f32>) -> (f32, f32, f32, f32, f32) {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let at = |q: f64| v[((v.len() - 1) as f64 * q) as usize];
    (v[0], at(0.5), at(0.9), at(0.99), v[v.len() - 1])
}

/// Record one phase with the person told what it is for, first.
fn record(mic: &mut Microphone, seconds: u64, what: &str) -> Vec<f32> {
    println!("\n  >>> {what}");
    for n in (1..=3).rev() {
        print!("      {n}… ");
        let _ = std::io::stdout().flush();
        std::thread::sleep(std::time::Duration::from_secs(1));
    }
    println!("\n      ● 마이크 열림 — {seconds}초");
    mic.hold(true);
    std::thread::sleep(std::time::Duration::from_secs(seconds));
    mic.hold(false);
    println!("      ○ 마이크 닫힘");
    mic.take().map(|c| c.samples).unwrap_or_default()
}

fn main() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap()
        .to_path_buf();
    let seconds: u64 = std::env::args().nth(1).and_then(|a| a.parse().ok()).unwrap_or(10);
    let dir = root.join("measure");

    println!("\n  이 하네스는 마이크를 엽니다. {seconds}초짜리 세 구간을 녹음하고,");
    println!("  30 ms 프레임마다 최대치와 RMS 만 남기고 소리 자체는 버립니다.");
    println!("  파일로 저장되는 것은 숫자뿐입니다.\n");

    let mut mic = Microphone::new();
    //: **Three phases and the third is not a repeat.** Quiet measured only once, before anybody
    //: has spoken, is quiet with the person sitting still and waiting; quiet *after* talking is
    //: the room they actually leave behind, and a gate lives in the second one.
    let quiet_before = record(&mut mic, seconds, "아무 말도 하지 마세요 — 평소 방 소리만");
    if !mic.available {
        println!("\n  ** 마이크를 열지 못했습니다. 입력 장치를 확인하세요.");
        std::process::exit(3);
    }
    let speech = record(&mut mic, seconds, "평소 말하듯이 계속 말해주세요");
    let quiet_after = record(&mut mic, seconds, "다시 아무 말도 하지 마세요");

    let phases: Vec<(&str, Vec<f32>)> = vec![("quiet-before", quiet_before),
                                             ("speech", speech),
                                             ("quiet-after", quiet_after)];
    let mut rows: Vec<Value> = Vec::new();
    let mut by_name: Vec<(&str, Vec<Frame>)> = Vec::new();
    println!("\n  frames of {FRAME_MS} ms, dBFS");
    println!("  {:<14} {:>6}  {:>8} {:>8} {:>8} {:>8} {:>8}",
             "phase", "frames", "min", "median", "p90", "p99", "max");
    for (name, samples) in phases {
        let f = frames(&samples);
        if f.is_empty() {
            println!("  {name:<14} nothing recorded — the device gave no audio");
            continue;
        }
        let (lo, mid, p90, p99, hi) = quantiles(f.iter().map(|x| db(x.rms)).collect());
        println!("  {name:<14} {:>6}  {:>8.1} {:>8.1} {:>8.1} {:>8.1} {:>8.1}",
                 f.len(), lo, mid, p90, p99, hi);
        rows.push(json!({"phase": name, "frames": f.len(),
                         "rms_db": {"min": lo, "median": mid, "p90": p90, "p99": p99, "max": hi},
                         "peak_db_max": db(f.iter().fold(0.0f32, |a, x| a.max(x.peak)))}));
        by_name.push((name, f));
    }

    //: **The whole question in one sweep.** For every candidate level, how much quiet gets through
    //: and how much speech is lost. Both, always: a gate that never opens loses no quiet.
    let quiet: Vec<f32> = by_name.iter().filter(|(n, _)| n.starts_with("quiet"))
        .flat_map(|(_, f)| f.iter().map(|x| db(x.rms))).collect();
    let talk: Vec<f32> = by_name.iter().filter(|(n, _)| *n == "speech")
        .flat_map(|(_, f)| f.iter().map(|x| db(x.rms))).collect();
    let mut sweep: Vec<Value> = Vec::new();
    if !quiet.is_empty() && !talk.is_empty() {
        println!("\n  a level, and what it costs in both directions");
        println!("  {:>8}  {:>18}  {:>18}", "dBFS", "quiet let through", "speech lost");
        let mut best: Option<(f32, f32)> = None;
        for step in -70..=-10 {
            let t = step as f32;
            let through = quiet.iter().filter(|&&x| x >= t).count() as f32 / quiet.len() as f32;
            let lost = talk.iter().filter(|&&x| x < t).count() as f32 / talk.len() as f32;
            sweep.push(json!({"db": t, "quiet_through": through, "speech_lost": lost}));
            if step % 5 == 0 {
                println!("  {t:>8.0}  {:>17.1}%  {:>17.1}%", through * 100.0, lost * 100.0);
            }
            //: **Worst-direction-first**, rather than a sum: a gate is judged by whichever of its
            //: two failures is larger, so a level that is perfect one way and useless the other
            //: cannot win by averaging.
            let worst = through.max(lost);
            if best.is_none_or(|(_, b)| worst < b) {
                best = Some((t, worst));
            }
        }
        if let Some((t, worst)) = best {
            println!("\n  best level {t:.0} dBFS — its worse side is {:.1}%", worst * 100.0);
            println!("  **A gate is only worth having if that number is small.** Speech and an\n  \
                      empty room overlapping means no level separates them here, and always-on\n  \
                      listening would need something other than loudness.");
        }
    }

    let out = Destination::new(&dir, "listening", &[format!("f{FRAME_MS}ms"), format!("x{seconds}s")]);
    if let Err(why) = out.refuse_if_taken() {
        println!("\n  ** {why}");
        std::process::exit(3);
    }
    std::fs::write(&out.path, serde_json::to_string_pretty(
        &json!({"frame_ms": FRAME_MS, "hz": HZ, "seconds_each": seconds,
                "phases": rows, "sweep": sweep})).unwrap()).unwrap();
    println!("\n  every frame's level -> {}", out.path.display());
}
