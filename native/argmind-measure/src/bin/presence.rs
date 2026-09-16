//! Can the camera tell a still room from a person moving in it? **The measurement that decides
//! whether presence is a feature or a guess.**
//!
//! ```text
//!   presence           two phases, ten seconds each — it tells you when to move
//!   presence --quick   four seconds each, for checking the harness rather than the question
//! ```
//!
//! **This turns the camera on and the light comes on with it.** It is the only thing here that
//! does, and it wants a person: a threshold set from a quiet room alone is a threshold that has
//! never seen the thing it is supposed to catch.
//!
//! ## Why it is asked this way
//!
//! `camera.rs` measured what a quiet room does to itself — 0.14 to 0.34 across six runs of
//! twenty-five, **and 1.44 in a seventh nobody can explain.** That last one is the whole problem:
//! it might have been somebody moving, in which case the signal is large and a threshold is easy,
//! or it might have been a reflection, in which case a threshold under 1.5 fires at nothing. One
//! number cannot say which, and averaging the runs would turn *we do not know* into a decimal.
//!
//! So: hold still on purpose, then move on purpose, and compare the two distributions. **The
//! interesting outcome is the one where they overlap** — that is a finding and not a failure, and
//! it would mean this camera at this size cannot do it and something else has to.
//!
//! ## What it will not do
//!
//! **It does not pick a threshold to make the answer come out.** If the two ranges touch, it says
//! they touch and names no number — a floor lowered to make something pass is the failure the whole
//! standard of evidence in this project exists against.
//!
//! ## A frame at a time was the wrong statistic, and running it said so
//!
//! The first version compared single-frame differences and decided on the **worst** still frame
//! against the quietest moving one. Run on a deliberately still room it reported: median 0.05,
//! ninetieth percentile 0.07 — and a **maximum of 23.09**, four hundred times the median. So the
//! quiet distribution is not one shape, it is *almost always nothing* with rare enormous spikes: an
//! exposure step, or the monitor in front of the person changing what it is showing and relighting
//! their face. The earlier unexplained 1.44 was the small end of the same thing.
//!
//! A rule built on the worst still frame can therefore never fire, and one built on the ninetieth
//! percentile fires several times a second. **Neither is a threshold; the statistic was wrong.**
//! So this also measures the **median over a one-second window**: a single flash cannot move it and
//! a person moving for a second cannot avoid moving it. Both are reported, because the frame-level
//! numbers are what say *why* the windowed ones are the right ones to read.
//!
//! ## And the one-second window was not enough either — 2026-09-16
//!
//! Ten seconds still against ten seconds moving, on a person doing it on purpose:
//!
//! | | still | moving |
//! |---|---|---|
//! | one-second windows, median | **0.03** | **0.49** — sixteen times |
//! | one-second windows, range over ten | 0.02–**0.40** | **0.33**–0.82 |
//!
//! **The middle separates by a factor of sixteen and the tails overlap**, on one still second that
//! reached 0.40. The rule was written before the run and it says overlap, so overlap is the answer
//! for that rule. What the run added is the *mechanism*: a person sitting in front of a monitor is
//! lit by it, and when the screen changes their face changes with it — a near-uniform shift of
//! every pixel at once, which is what the 16.50 single-frame spikes are too.
//!
//! So this now also measures [`Frame::moved_from`], which subtracts the mean shift before
//! differencing. **That is aimed at the confound rather than at the answer**: a screen flash
//! relights everything by about the same amount and cancels; a hand through part of the frame does
//! not. Both statistics and two window lengths are reported from one run, so the next attempt does
//! not cost another ten seconds of somebody sitting still twice.
//!
//! Nothing is written to disk, nothing is sent anywhere, and the picture never leaves the 64×36
//! greys `camera.rs` reduces it to — see the note there about what that size can and cannot be.

use std::io::Write;
use std::time::{Duration, Instant};

use argmind_env_windows::camera::Camera;

struct Phase {
    say: &'static str,
    then: &'static str,
}

const PHASES: [Phase; 2] = [
    Phase { say: "가만히 계세요", then: "움직이지 마시고, 화면만 보고 계시면 됩니다" },
    Phase { say: "움직여 주세요", then: "손을 흔들거나 몸을 움직이세요 — 평소처럼" },
];

/// One sample: when it was taken, and the two differences from the look before it.
struct Sample {
    at: f64,
    raw: f64,
    moved: f64,
}

/// Every consecutive pair in one phase, **with the time each was taken.**
///
/// The time is not decoration. A look was measured at 32 ms and then measured at 111 ms in a later
/// run, so a window counted in frames is a window of between one and three and a half seconds
/// depending on what the camera felt like doing. Windows are cut by the clock instead.
fn gather(camera: &mut Camera, seconds: f64) -> Vec<Sample> {
    let began = Instant::now();
    let mut out = Vec::new();
    let mut before = None;
    while began.elapsed().as_secs_f64() < seconds {
        let Ok(now) = camera.look() else { continue };
        if let Some(last) = &before {
            out.push(Sample { at: began.elapsed().as_secs_f64(),
                              raw: now.differs_from(last), moved: now.moved_from(last) });
        }
        before = Some(now);
    }
    out
}

/// **Two window lengths, because the shorter one was not enough.** A monitor changing what it shows
/// lasts a fraction of a second; a person moving lasts seconds. Three of them is the cheapest thing
/// that could separate the two, and it costs nothing to report both.
const WINDOWS: [(f64, &str); 2] = [(1.0, "1초"), (3.0, "3초")];

/// Fewer windows than this and the tail is not being measured, it is being guessed at.
const ENOUGH: usize = 3;

/// The median of each window of `each` seconds, sorted.
///
/// **This is the statistic the frame-level one turned out not to be.** A lighting flash is one
/// frame in a window and moves a median not at all; a person moving through a window moves every
/// frame in it.
///
/// **Cut by the clock, not by a count**, because the frame rate is not a constant — see `gather`.
fn windowed(samples: &[Sample], each: f64, of: impl Fn(&Sample) -> f64) -> Vec<f64> {
    let mut out = Vec::new();
    let mut window: Vec<f64> = Vec::new();
    let mut ends = each;
    for s in samples {
        while s.at > ends {
            //: A window with almost nothing in it is not a median of anything.
            if window.len() >= 5 {
                window.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                out.push(window[window.len() / 2]);
            }
            window.clear();
            ends += each;
        }
        window.push(of(s));
    }
    if window.len() >= 5 {
        window.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        out.push(window[window.len() / 2]);
    }
    out.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    out
}

/// The value at a fraction of the way through a sorted list.
fn at(sorted: &[f64], fraction: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let i = ((sorted.len() - 1) as f64 * fraction).round() as usize;
    sorted[i.min(sorted.len() - 1)]
}

fn countdown(what: &str, why: &str, seconds: u64) {
    println!("\n  {what} — {why}");
    for left in (1..=seconds).rev() {
        print!("\r    {left}…   ");
        let _ = std::io::stdout().flush();
        std::thread::sleep(Duration::from_secs(1));
    }
    println!("\r    지금!      ");
}

fn main() {
    std::process::exit(run());
}

fn run() -> i32 {
    let quick = std::env::args().any(|a| a == "--quick");
    let seconds = if quick { 4.0 } else { 10.0 };

    println!("\n  카메라로 '사람이 있는가'를 잴 수 있는지 확인합니다.");
    println!("  두 번 잽니다: 가만히 있을 때와 움직일 때. 각 {seconds:.0}초.");
    println!("  **카메라 불이 켜집니다.** 아무것도 저장하지 않고, 화면은 64x36 회색으로만 봅니다.\n");

    let opening = Instant::now();
    let mut camera = match Camera::open() {
        Ok(c) => c,
        Err(why) => {
            println!("  카메라를 열지 못했습니다 — {why}");
            return 2;
        }
    };
    println!("  {:?} {}x{} · 여는 데 {:.0} ms · 안정화 {}",
             camera.name, camera.width, camera.height, opening.elapsed().as_secs_f64() * 1000.0,
             match camera.settled_in {
                 Some(s) => format!("{:.0} ms", s * 1000.0),
                 //: **Said rather than swallowed.** Every number below would be about the camera
                 //: rather than about the room.
                 None => "끝내 안정되지 않음 — 아래 숫자는 믿지 마세요".to_string(),
             });

    let mut runs = Vec::new();
    for phase in &PHASES {
        countdown(phase.say, phase.then, 3);
        let samples = gather(&mut camera, seconds);
        if samples.is_empty() {
            println!("  프레임을 하나도 받지 못했습니다");
            return 2;
        }
        println!("    {} 장, 초당 {:.0} 장",
                 samples.len(), samples.len() as f64 / seconds);
        //: **The windows are cut before the sort, and that order matters.** An earlier version
        //: sorted first, which made every "one-second window" a slice of the *sorted* list rather
        //: than of time — order statistics wearing the name of a temporal median. It was robust to
        //: the spike by accident, which is the worst way for a measurement to be right.
        let mut mine = Vec::new();
        let takers: [(&str, fn(&Sample) -> f64); 2] =
            [("그대로", |s| s.raw), ("밝기 뺀 뒤", |s| s.moved)];
        for (kind, of) in takers {
            let mut sorted: Vec<f64> = samples.iter().map(&of).collect();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            println!("    {kind}: 한 장씩 — 중앙값 {:.2}, 90% {:.2}, 최대 {:.2}",
                     at(&sorted, 0.5), at(&sorted, 0.9), at(&sorted, 1.0));
            for (each, name) in WINDOWS {
                let w = windowed(&samples, each, &of);
                if w.len() < ENOUGH {
                    println!("      {name} 창 {} 개 — 자료가 모자랍니다", w.len());
                } else {
                    println!("      {name} 창 {} 개 — 최소 {:.2}, 중앙값 {:.2}, 최대 {:.2}",
                             w.len(), at(&w, 0.0), at(&w, 0.5), at(&w, 1.0));
                }
                mine.push(((kind, name), w));
            }
        }
        runs.push(mine);
    }

    println!("\n  ── 어느 조합이 갈라지는가 ──");
    let mut any = false;
    for i in 0..runs[0].len() {
        let ((kind, name), still) = &runs[0][i];
        let (_, moving) = &runs[1][i];
        //: **The tails, not the middles**: a threshold lives between the worst quiet window and the
        //: quietest moving one, and comparing medians would hide exactly the overlap that matters.
        //: **Too few windows is not a verdict.** With one or two of them the worst is whatever
        //: happened to be in them, and calling that a tail would be the harness guessing.
        if still.len() < ENOUGH || moving.len() < ENOUGH {
            println!("  {kind:<10} {name} 창:  자료가 모자랍니다                       (가만히 {}개, 움직임 {}개, {ENOUGH}개는 있어야 합니다)",
                     still.len(), moving.len());
            continue;
        }
        let worst_still = at(still, 1.0);
        let quiet_moving = at(moving, 0.0);
        let parts = if worst_still > 0.0 && quiet_moving > worst_still * 1.5 {
            any = true;
            let between = (worst_still * quiet_moving).sqrt();
            format!("갈라짐 — 사이 값 {between:.2}, {:.1}배 여유", quiet_moving / worst_still)
        } else {
            "겹침".to_string()
        };
        println!("  {kind:<10} {name} 창:  가만히 최대 {worst_still:>5.2}                     움직임 최소 {quiet_moving:>5.2}   {parts}");
    }

    if any {
        println!("\n  갈라지는 조합이 있습니다. **쓰기 전에 한 번 더 돌려 보세요** — 한 번은 한 번이고,");
        println!("  창 열 개는 꼬리를 재기에 적은 수입니다.");
        return 0;
    }

    //: **A negative result, protected.** Two ranges that touch mean this camera at this size
    //: cannot tell the two apart, and saying so is the answer — picking a number anyway is how a
    //: feature ships that fires at reflections and misses people.
    println!("\n  어느 조합도 갈라지지 않습니다.");
    println!("  이 카메라로 이 크기에서는 '사람이 있는가'를 이 방식으로 가릴 수 없습니다.");
    println!("  임계값을 정하지 않습니다 — 통과시키려고 낮춘 바닥은 아무것도 재지 않습니다.");
    1
}
