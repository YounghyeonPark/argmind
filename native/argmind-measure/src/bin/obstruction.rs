//! When something covers the target, does anything still reach it? **The question that decides
//! whether the control patterns stay.**
//!
//! ```text
//!   obstruction          launch Character Map, cover a check box, try eight ways
//!   obstruction --dry    everything except pressing or operating anything
//! ```
//!
//! Measured 2026-09-09 on three ordinary goals: `3 of 3` with patterns and `3 of 3` without, so
//! they bought nothing — and this project's standard says unearned code is a cost. But there is a
//! case that goal set never touched: **something is sitting on top of the target.** A synthetic
//! click goes to whatever owns that pixel; the hope was that a control pattern does not aim at a
//! pixel at all, and so would reach past.
//!
//! ## What reading the code says before running it
//!
//! **That hope is not true of this implementation, and the Python's docstring said it anyway.**
//! [`argmind_env_windows::patterns::operate`] takes an `x` and a `y`, hands them to
//! `ElementFromPoint`, and walks *upwards* from whatever comes back. Cover the target and it gets
//! the cover's element and the cover's ancestors — never the check box. Then the fallback is a
//! click, and the click is refused by the same guard.
//!
//! So the predictions are written into `ARMS` below, before the run, and each row prints what was
//! predicted beside what happened. A row that disagrees is the interesting one.
//!
//! ## What makes this safe to run on a machine somebody is using
//!
//! The cover is **ours**: this process makes it and destroys it. The widened reach is exactly two
//! process ids — the application this harness launched, and this harness — so the rule is still
//! *nothing we are not responsible for*, and the two guarded rows show the ordinary reach refusing
//! before anything is sent. Character Map is small, launched and closed here, and cannot lose
//! anybody's work.
//!
//! ## It needs the machine to itself, and says so rather than guessing
//!
//! Every arm begins by putting Character Map in front and **verifying that it got there**. That is
//! not fussiness: `ancestry` once reported a clean `0 of 12` while measuring a remote-desktop
//! window that was covering the coordinates, and a harness that clicks wherever the pointer lands
//! is worse than one that refuses.
//!
//! It will refuse on a machine somebody is working on, and the reason is not a bug to route around.
//! Windows only lets the process that **owns the foreground** hand it to another; a harness started
//! from a terminal that is not itself in front has no such privilege to pass on, so
//! `SetForegroundWindow` flashes a taskbar button and nothing more. Run it from a console window
//! that has the focus and then leave the keyboard alone. Anything else — attaching thread input,
//! or any of the other ways of taking a foreground nobody offered — would make this a measurement
//! of a program that seizes the screen, which is not the program.
//!
//! **That advice was half of it.** The console must also not be *inside the session driving the
//! run*: this takes the foreground away from whatever started it, so a harness launched from the
//! window somebody is working in interrupts its own caller. See the note in `clicking.rs`, where
//! three attempts were cut off that way and the wreckage was briefly explained as a locked screen
//! it was not.
//!
//! ## Measured 2026-09-16 — **24 of 24, and the patterns reach past nothing**
//!
//! Three runs, eight arms each, every one agreeing with the prediction written above it:
//!
//! | arm | outcome | how |
//! |---|---|---|
//! | covered, guarded, patterns off | refused | the guard; the point belonged to the cover |
//! | covered, guarded, patterns on | refused | the guard, **before either path is chosen** |
//! | covered, allowed, patterns off | missed | click, and the target gained nothing |
//! | covered, allowed, patterns on | missed | **click** — the pattern would not resolve |
//! | stood aside, patterns off | worked | click, `+ ["Character set :", "Go to Unicode :", "Group by :"]` |
//! | stood aside, patterns on | worked | pattern `invoke` |
//! | nothing covering, patterns off | worked | click, same three |
//! | nothing covering, patterns on | worked | pattern `invoke` |
//!
//! **The decisive pair is the two pattern rows.** Covered, the pattern path falls back to a click —
//! `ElementFromPoint` hands it the cover and the walk upwards never meets the check box. Uncovered
//! or stood aside, the same call resolves and `invoke` runs. So the hope this harness was built to
//! test is answered: **a control pattern does not reach past an obstruction**, because this one is
//! aimed at a pixel after all.
//!
//! The controls hold, which is what makes the rest mean anything: the uncovered rows do the thing,
//! and the rows that missed gained *nothing at all* rather than something different.
//!
//! **What this does not settle.** It says patterns buy nothing *here*. `pointer-theft` measures the
//! other axis and the patterns win it — they operate the control without dragging the cursor off
//! whatever the person was pointing at. This row is one input to that decision and not the decision.
//!
//! ## Why the last two rows exist
//!
//! **A harness that reports "nothing happened" everywhere might simply be broken**, and a uniformly
//! tidy result is evidence the measurement is. The uncovered rows are the control: they must do the
//! thing, or nothing above them means anything.

use std::collections::BTreeSet;
use std::thread::sleep;
use std::time::Duration;

use argmind_core::contract::Widget;
use argmind_env_windows::hands::{Did, WindowsHands};
use argmind_env_windows::{bring_to_front, foreground_is, pointing, Pointing, whats_in_front,
                          windows_of, Cover, Launched, WindowsEyes};

/// The check box, because its effect is unambiguous: the advanced view adds four named things.
const TARGET: (&str, &str) = ("Advanced view", "check box");
/// One of them, and the one the `clicking` harness already uses.
const MARKER: &str = "search";

/// What was predicted, **written here before anything ran.**
#[derive(Debug, Clone, Copy, PartialEq)]
enum Guess {
    /// The reach guard stopped it, so nothing was sent and nothing could have happened.
    Refused,
    /// It acted, and the target did not change — the action went into the cover.
    Missed,
    /// The target changed.
    Worked,
}

impl Guess {
    fn short(self) -> &'static str {
        match self {
            Guess::Refused => "refused",
            Guess::Missed => "missed",
            Guess::Worked => "worked",
        }
    }
}

struct Arm {
    label: &'static str,
    patterns: bool,
    covered: bool,
    /// Lower the cover before acting — what an always-on-top application would have to do.
    aside: bool,
    /// Widen the reach to include the cover this harness made, so the action is actually attempted
    /// rather than refused.
    ours_too: bool,
    guess: Guess,
}

const ARMS: [Arm; 8] = [
    //: The ordinary reach, which is what a harness has. **The guard should stop both**, and it
    //: should stop the pattern arm too — the check happens before either path is chosen.
    Arm { label: "covered, guarded, patterns off", patterns: false, covered: true, aside: false,
          ours_too: false, guess: Guess::Refused },
    Arm { label: "covered, guarded, patterns on", patterns: true, covered: true, aside: false,
          ours_too: false, guess: Guess::Refused },
    //: Guard widened so the attempt actually happens. **This is the Python's question**, and the
    //: prediction from reading `element_for` is that neither path reaches past a cover.
    Arm { label: "covered, allowed, patterns off", patterns: false, covered: true, aside: false,
          ours_too: true, guess: Guess::Missed },
    Arm { label: "covered, allowed, patterns on", patterns: true, covered: true, aside: false,
          ours_too: true, guess: Guess::Missed },
    //: Moving out of the way, which is what the old application did before every click.
    Arm { label: "stood aside, patterns off", patterns: false, covered: true, aside: true,
          ours_too: true, guess: Guess::Worked },
    Arm { label: "stood aside, patterns on", patterns: true, covered: true, aside: true,
          ours_too: true, guess: Guess::Worked },
    //: **The control.** Nothing in the way at all; both paths must do the thing.
    Arm { label: "nothing covering, patterns off", patterns: false, covered: false, aside: false,
          ours_too: false, guess: Guess::Worked },
    Arm { label: "nothing covering, patterns on", patterns: true, covered: false, aside: false,
          ours_too: false, guess: Guess::Worked },
];

/// Every named thing across **every** window the process owns.
///
/// Not just the launched one: a drop-down is its own top-level window, and reading only the main
/// one reports "nothing happened" about a list that did open.
fn named(eyes: &mut WindowsEyes, pid: u32) -> BTreeSet<String> {
    windows_of(pid).into_iter().flat_map(|h| eyes.read(h)).map(|w| w.name).collect()
}

/// The check box, found by name and role — **and if it is not there, what is**, because a target
/// that does not exist on this Windows is a fact about the machine rather than a failure to report
/// as one.
fn target_in(eyes: &mut WindowsEyes, pid: u32) -> Result<Widget, String> {
    let all: Vec<Widget> = windows_of(pid).into_iter().flat_map(|h| eyes.read(h)).collect();
    if let Some(found) = all.iter().find(|w| (w.name.as_str(), w.role.as_str()) == TARGET) {
        return Ok(found.clone());
    }
    let boxes: Vec<&str> = all.iter().filter(|w| w.role == TARGET.1)
        .map(|w| w.name.as_str()).take(6).collect();
    Err(format!("no {:?} {} here; the check boxes are {boxes:?}", TARGET.0, TARGET.1))
}

/// What one arm did, or why it could not be run.
fn attempt(arm: &Arm, app: &str, dry: bool) -> Result<(Guess, String), String> {
    let launched = Launched::start(app).map_err(|e| format!("could not start {app} ({e})"))?;
    let pid = launched.pid;
    let Some(main_window) = launched.window(Duration::from_secs(6)) else {
        return Err(format!("{app} opened no visible window"));
    };
    //: **Asked repeatedly and then verified**, because one `SetForegroundWindow` is not enough:
    //: Windows hands the privilege around and a freshly launched application takes a moment before
    //: it can accept it. Reporting the window as in front on the strength of the request returning
    //: true is how `ancestry` once measured a remote-desktop window instead of the target.
    if !bring_to_front(main_window, pid, Duration::from_secs(4)) {
        let (_, title, owner) = whats_in_front();
        //: **Named, not just refused.** "Could not get the front" sends somebody looking at the
        //: harness; the title of the window holding it is the whole answer.
        return Err(format!("{title:?} (pid {owner:?}) holds the front"));
    }

    let mut eyes = WindowsEyes::new().map_err(|e| format!("UI Automation would not open ({e})"))?;
    let widget = target_in(&mut eyes, pid)?;

    //: Held for the whole attempt and destroyed by `Drop` on every path out, including the `?`s
    //: below — a borderless always-on-top rectangle left behind is a piece of somebody's screen
    //: they cannot click and cannot close.
    let cover = if arm.covered {
        let (x, y, w, h) = widget.rect;
        Some(Cover::over(x, y, w, h)?)
    } else {
        None
    };
    sleep(Duration::from_millis(400));
    if let (Some(cover), true) = (cover.as_ref(), arm.aside) {
        cover.stand_aside()?;
        sleep(Duration::from_millis(300));
    }
    //: **Checked after the cover went up**, because a cover that stole the foreground would make
    //: every row below it a measurement of losing focus instead.
    if !foreground_is(pid) {
        return Err("the cover took the foreground — the arm cannot be run".into());
    }

    let mut hands = if arm.ours_too {
        WindowsHands::only_within_or_ours(pid)
    } else {
        WindowsHands::only_within(pid)
    };
    if arm.patterns {
        hands = hands.with_patterns();
    }
    if !dry {
        hands = hands.arm();
    }

    let before = named(&mut eyes, pid);
    let did = hands.click_button(&widget, "left");
    sleep(Duration::from_millis(900));
    let after = named(&mut eyes, pid);
    drop(cover);

    let arrived = after.difference(&before)
        .any(|n| n.to_lowercase().contains(MARKER));
    let outcome = match (&did, arrived) {
        (Did::Elsewhere { .. }, _) => Guess::Refused,
        (_, true) => Guess::Worked,
        (_, false) => Guess::Missed,
    };
    //: **The delta, always.** A boolean cannot be diagnosed afterwards, and this project has
    //: already lost a run to keeping only one.
    let gained: Vec<&String> = after.difference(&before).take(3).collect();
    let note = match &did {
        Did::Elsewhere { owner } => format!("guard refused; the point belonged to {owner:?}"),
        Did::Operated { how, live } => format!("pattern {how}{}", if *live { "" } else { " (dry)" }),
        Did::Sent { live } => format!("click{}  + {gained:?}",
                                      if *live { "" } else { " (dry)" }),
        Did::Refused(why) => format!("malformed: {why}"),
    };
    let _ = launched;
    Ok((outcome, note))
}

fn main() {
    //: **`std::process::exit` does not run destructors**, so every refusal leaves through here or
    //: the launched application outlives the harness that promised to close it.
    let code = run();
    std::process::exit(code);
}

fn run() -> i32 {
    let dry = std::env::args().any(|a| a == "--dry");
    let app = std::env::args().skip(1).find(|a| a.ends_with(".exe"))
        .unwrap_or_else(|| "charmap.exe".into());

    //: Asked before anything is clicked: on a scaled write path every click is offset and nothing
    //: lands, which would look exactly like an obstruction that was not there.
    match pointing() {
        Pointing::Exact => {}
        //: **Two refusals, told apart.** One is a display setting and one is a locked screen, and
        //: sending somebody to check their DPI while the machine is merely locked wastes an
        //: afternoon.
        Pointing::Scaled => {
            println!("  the cursor does not land where it is sent on this display — refusing to click");
            return 2;
        }
        Pointing::Refused => {
            //: **What is true, and no further.** The first version of this said "the screen is
            //: locked"; measured, the input desktop was `Default` and `SetCursorPos` still failed
            //: with `ERROR_INVALID_HANDLE`, so that explanation was invented rather than found.
            println!("  the desktop will not take input — SetCursorPos is refused. Nothing here can reach the screen until it does; the cause is not established (see Pointing::Refused).");
            return 2;
        }
    }
    println!("\n  {:?} {} covered by a window of ours, then acted on", TARGET.0, TARGET.1);
    println!("  {app} | {}\n",
             if dry { "DRY — nothing will be pressed or operated" } else { "LIVE" });
    println!("  {:<32} {:<9} {:<9} {}", "arm", "predicted", "actual", "what happened");

    let (mut ran, mut as_predicted) = (0, 0);
    for arm in &ARMS {
        match attempt(arm, &app, dry) {
            Ok((outcome, note)) => {
                ran += 1;
                let agrees = outcome == arm.guess;
                if agrees {
                    as_predicted += 1;
                }
                println!("  {:<32} {:<9} {:<9} {}{}", arm.label, arm.guess.short(),
                         outcome.short(), note, if agrees { "" } else { "   <- DISAGREES" });
            }
            //: **Not counted as a result.** An arm that could not be set up says so; folding it in
            //: as "missed" would be the harness reporting its own failure as a finding.
            Err(why) => println!("  {:<32} {:<9} {:<9} {why}", arm.label, arm.guess.short(),
                                 "skipped"),
        }
        sleep(Duration::from_millis(700));
    }

    println!("\n  {as_predicted} of {ran} arms agreed with what was predicted from the code \
              ({} never ran)", ARMS.len() - ran);
    if dry {
        println!("  DRY: nothing was pressed, so every row above is about the guard and the \
                  resolution, not about the effect.");
    }
    i32::from(ran == 0)
}
