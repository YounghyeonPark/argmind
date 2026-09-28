//! **눈치** — reading the room before speaking first, as a number a person turns.
//!
//! The owner, 2026-09-28, after the measurement that the model cannot be trusted with this:
//! *눈치에 대한 수치를 조절하는것도 방법일듯.*
//!
//! ## Why this is not one of the dials beside it
//!
//! 재치 · 유머 · 봉사 · 헌신 are **prompt** dials: each reaches the model as a sentence. This one
//! must not, and a measurement says why. `argmind-measure --bin turned`, 2026-09-28: over 28
//! moments where argmind was about to speak first, four arms and two instructions, **224 of 224
//! spoke** — including under *otherwise leave "say" empty* — and a sentence about the person became
//! a sentence *to* them: *You turned your head, I noticed.* A 눈치 line in the prompt would be the
//! same experiment a third time, with the same answer expected and a slider on top claiming it
//! worked. `mind.rs`'s own rule: **a dial that moves nothing is worse than no dial.**
//!
//! So 눈치 moves the program, not the prompt. **Nothing about it is ever shown to the model.**
//!
//! ## What it does, and every number in it is chosen rather than measured
//!
//! **At 0 it does nothing** — argmind behaves exactly as it did before this file, which is the same
//! ground every dial here was allowed on: nothing set, nothing changed.
//!
//! From 1 up, two things, only for what argmind says **unprompted** — a person who asks is answered:
//!
//! 1. **It does not speak first to somebody whose head is turned away.** It holds the remark and
//!    says it when they look back, *unless the window changed in between* — then it was about a
//!    screen they have left and it is dropped. The newest remark replaces a held one: a tactful
//!    person does not arrive with a backlog.
//! 2. **It leaves N × [`GAP_PER_POINT`] seconds between its own unprompted remarks.** At 10 that is
//!    one every 100 s at most. The watch fires on 13 seconds in 14 and argmind speaks on every one
//!    it is given (measured: 8 of 8 in `noticing`, 224 of 224 here), so the only silence available
//!    on this path is silence the program makes.
//!
//! **No evidence never means silence.** Held only on a positive *turned away*; released on anything
//! else — facing, unreadable, attention off, the camera stalled. A dial that could strand a remark
//! for ever because a camera went away would be a mute button nobody pressed, and `somebody_about`
//! already holds that line for the whole camera.

use std::path::PathBuf;

/// How high it goes. The same as the other dials, so one slider looks like the next.
pub const MOST: u8 = 10;

/// **Seconds between unprompted remarks, per point.** Chosen, not measured: 10 puts the top of the
/// dial at 100 s, which is long enough to feel like restraint and short enough that turning it up
/// does not read as switching the watch off.
pub const GAP_PER_POINT: f64 = 10.0;

/// **How long a held remark keeps.** The same five minutes as `AWAY_AFTER` and for the same reason:
/// past that, the room has changed enough that `somebody_about` stops answering *yes* anyway, and a
/// remark about the screen five minutes ago is not a remark about anything.
pub const HELD_AT_MOST: f64 = 300.0;

/// What the camera last said about where the head points, reduced to what this needs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Evidence {
    TurnedAway,
    /// Facing, unreadable, off, stale — **all the same here**, and deliberately: only a positive
    /// reading of *turned away* is a reason to wait.
    NotTurnedAway,
}

/// What to do with a remark argmind is about to make unprompted.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Now {
    Say,
    Hold,
}

/// The seconds that must pass between two unprompted remarks at this setting.
pub fn gap(nunchi: u8) -> f64 {
    nunchi.min(MOST) as f64 * GAP_PER_POINT
}

/// **Say it now, or hold it.** `since_last` is seconds since argmind last spoke unprompted, `None`
/// if it has not yet this run.
pub fn now(nunchi: u8, evidence: Evidence, since_last: Option<f64>) -> Now {
    if nunchi == 0 {
        return Now::Say;
    }
    if evidence == Evidence::TurnedAway {
        return Now::Hold;
    }
    match since_last {
        Some(s) if s < gap(nunchi) => Now::Hold,
        _ => Now::Say,
    }
}

/// **May a held remark be said now?** And separately: should it be thrown away?
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Held {
    /// Say it.
    Release,
    /// Keep holding.
    Keep,
    /// It is about a screen that is gone, or it has aged out.
    Drop,
}

/// Decide about a held remark on this tick.
///
/// `same_window` is whether the window in front is the one the remark was made about; `age` is how
/// long it has been held.
pub fn held(nunchi: u8, evidence: Evidence, since_last: Option<f64>, same_window: bool,
            age: f64) -> Held {
    //: **Dropping comes first.** A remark about a window somebody has left is wrong to say however
    //: tactfully it is timed.
    if !same_window || age > HELD_AT_MOST {
        return Held::Drop;
    }
    match now(nunchi, evidence, since_last) {
        Now::Say => Held::Release,
        Now::Hold => Held::Keep,
    }
}

// ---- the file --------------------------------------------------------------------------------

/// `%LOCALAPPDATA%\argmind\tact.json`. **Its own file, not `mind.json`**: that one's writer knows
/// only the prompt dials and the goal, so a 눈치 line there would be erased the next time somebody
/// turned 유머 — and the at-rest hash of the prompt is guaranteed by that file, which is the last
/// place a behaviour setting should be leaning on.
pub fn at() -> PathBuf {
    let mut path = std::env::var("LOCALAPPDATA").map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir());
    path.push("argmind");
    path.push("tact.json");
    path
}

/// What is set. **An unreadable file is 0**, which is today's behaviour — the safe direction.
pub fn remembered() -> u8 {
    std::fs::read_to_string(at()).map(|t| parse(&t)).unwrap_or(0)
}

pub fn parse(text: &str) -> u8 {
    for line in text.lines() {
        let line = line.trim().trim_start_matches('{').trim_end_matches('}').trim()
            .trim_end_matches(',');
        let Some((left, right)) = line.split_once(':') else { continue };
        if left.trim().trim_matches('"') == "눈치" {
            if let Ok(n) = right.trim().trim_matches('"').parse::<u8>() {
                return n.min(MOST);
            }
        }
    }
    0
}

pub fn render(nunchi: u8) -> String {
    //: **Zero is no file content at all**, like `mind.json` — the file says only what somebody did.
    if nunchi == 0 {
        "{}\n".to_string()
    } else {
        format!("{{\n  \"눈치\": {}\n}}\n", nunchi.min(MOST))
    }
}

/// Set it and write the file.
pub fn turn(to: u8) -> Result<u8, String> {
    let n = to.min(MOST);
    let path = at();
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    std::fs::write(&path, render(n)).map_err(|e| format!("{path:?}: {e}"))?;
    Ok(n)
}

#[cfg(test)]
mod tests {
    use super::*;

    const AWAY: Evidence = Evidence::TurnedAway;
    const HERE: Evidence = Evidence::NotTurnedAway;

    #[test]
    fn at_zero_it_is_exactly_today() {
        //: **The ground it was allowed on.** Every combination says *say it*, including a head
        //: turned away and a remark a second after the last one.
        for ev in [AWAY, HERE] {
            for since in [None, Some(0.0), Some(1.0), Some(1000.0)] {
                assert_eq!(now(0, ev, since), Now::Say);
            }
        }
        assert_eq!(gap(0), 0.0);
    }

    #[test]
    fn a_turned_head_holds_the_remark() {
        assert_eq!(now(1, AWAY, None), Now::Hold);
        assert_eq!(now(10, AWAY, Some(9999.0)), Now::Hold);
    }

    #[test]
    fn no_evidence_is_never_a_reason_to_wait() {
        //: **Only a positive reading of turned away holds.** Unreadable, off and stale all arrive
        //: here as `NotTurnedAway`, and with the gap passed they say it.
        assert_eq!(now(10, HERE, None), Now::Say);
        assert_eq!(now(10, HERE, Some(1000.0)), Now::Say);
    }

    #[test]
    fn the_gap_grows_with_the_dial() {
        assert_eq!(gap(1), 10.0);
        assert_eq!(gap(10), 100.0);
        assert_eq!(now(3, HERE, Some(29.0)), Now::Hold, "30 s gap, 29 s since");
        assert_eq!(now(3, HERE, Some(31.0)), Now::Say);
        //: Above the top is the top, not a longer wait.
        assert_eq!(gap(200), gap(MOST));
    }

    #[test]
    fn a_held_remark_is_said_when_they_look_back() {
        assert_eq!(held(5, AWAY, None, true, 3.0), Held::Keep);
        assert_eq!(held(5, HERE, None, true, 6.0), Held::Release);
    }

    #[test]
    fn a_remark_about_a_screen_they_have_left_is_dropped_however_it_is_timed() {
        //: Facing, gap passed, fresh — and still dropped, because the window moved on.
        assert_eq!(held(5, HERE, Some(1000.0), false, 1.0), Held::Drop);
        assert_eq!(held(5, AWAY, None, false, 1.0), Held::Drop);
    }

    #[test]
    fn a_remark_held_too_long_goes() {
        assert_eq!(held(5, HERE, None, true, HELD_AT_MOST + 1.0), Held::Drop);
        assert_eq!(held(5, HERE, None, true, HELD_AT_MOST - 1.0), Held::Release);
    }

    #[test]
    fn the_file_round_trips_and_zero_writes_nothing() {
        assert_eq!(parse(&render(7)), 7);
        assert_eq!(parse(&render(0)), 0);
        assert_eq!(render(0), "{}\n");
        assert_eq!(parse(&render(99)), MOST);
        //: Hand-edited on one line, and with quotes a person would add.
        assert_eq!(parse("{\"눈치\": \"4\"}"), 4);
        assert_eq!(parse("not json at all"), 0);
    }
}
