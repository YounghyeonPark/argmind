//! The window that exists to be in the way.
//!
//! **Placed off the visible desktop**, so a suite run while somebody is working puts nothing on
//! their screen. `WindowFromPoint` works in virtual-screen coordinates and does not care whether a
//! monitor is there, so the one thing worth checking — *does this window own those pixels* — can be
//! checked where nobody has to look at it.

use argmind_env_windows::hands::{may_touch, owner_at, Reach};
use argmind_env_windows::Cover;

/// Far outside any monitor. Nothing draws here and nothing else lives here.
const AWAY: (i64, i64, i64, i64) = (-4000, -4000, 200, 120);

fn middle() -> (i64, i64) {
    (AWAY.0 + AWAY.2 / 2, AWAY.1 + AWAY.3 / 2)
}

/// **A cover owns the pixels it covers**, which is the whole premise of the measurement.
///
/// If this were false, `obstruction` would be reporting that an obstruction changes nothing while
/// never having obstructed anything.
#[test]
fn the_window_under_the_point_is_the_cover() {
    let (x, y) = middle();
    let before = owner_at(x, y);
    let cover = Cover::over(AWAY.0, AWAY.1, AWAY.2, AWAY.3).expect("a cover");
    let during = owner_at(x, y);
    assert_eq!(during, Some(std::process::id()),
               "the cover does not own its own pixels (before it was {before:?})");
    drop(cover);
    //: **And gives them back.** A cover that outlived its `Drop` would leave every later arm
    //: measuring an obstruction nobody asked for.
    assert_ne!(owner_at(x, y), Some(std::process::id()));
}

/// **The guard refuses a covered target, and the widened one does not.**
///
/// This is the pair of rows the harness prints first, decided here without a desktop, an
/// application or a click.
#[test]
fn the_ordinary_reach_refuses_a_point_the_cover_took() {
    let (x, y) = middle();
    let cover = Cover::over(AWAY.0, AWAY.1, AWAY.2, AWAY.3).expect("a cover");
    let owner = owner_at(x, y);
    let pretend_app = std::process::id().wrapping_add(1);

    //: What a harness has by default: it may touch the application it launched, and the point now
    //: belongs to the cover instead — so it is refused before anything is sent.
    assert!(!may_touch(Reach::OnlyProcess(pretend_app), owner));
    //: What `obstruction` widens to, and no further: ours as well as theirs.
    assert!(may_touch(Reach::OursAnd { ours: std::process::id(), theirs: pretend_app }, owner));
    //: A third process is still refused — the widening is two ids, not "anything".
    assert!(!may_touch(Reach::OursAnd { ours: pretend_app, theirs: pretend_app + 1 }, owner));
    drop(cover);
}

/// Standing aside stops it being on top, **without the window going away.**
///
/// **The first version of this test asserted the wrong thing** and the failure was worth keeping:
/// it checked that the point stopped belonging to us, and at a coordinate no other window reaches
/// it never does — `WindowFromPoint` returns the only window there whatever its z-order. That is a
/// property of the empty corner this test uses, not of `stand_aside`, and asserting it would have
/// meant either moving the test onto somebody's visible desktop or writing down a guarantee the
/// function does not make. Whether dropping topmost is *enough* to reach the window underneath is
/// exactly what the `obstruction` harness measures on a real desktop.
#[test]
fn standing_aside_drops_topmost_and_coming_back_takes_it_again() {
    let (x, y) = middle();
    let cover = Cover::over(AWAY.0, AWAY.1, AWAY.2, AWAY.3).expect("a cover");
    assert!(cover.is_topmost());
    assert_eq!(owner_at(x, y), Some(std::process::id()));

    cover.stand_aside().expect("lowered");
    assert!(!cover.is_topmost(), "it is still claiming to be above everything");
    //: Still there — this is *aside*, not *gone*, and that difference is the whole reason an
    //: always-on-top application could do it before a click and put itself back afterwards.
    assert_ne!(cover.handle(), 0);

    cover.come_back().expect("raised");
    assert!(cover.is_topmost());
}

/// A cover smaller than what it covers is not covering it.
#[test]
fn a_tiny_rectangle_is_widened_to_something_that_covers() {
    let cover = Cover::over(AWAY.0, AWAY.1, 3, 2).expect("a cover");
    //: The corner it was asked for is still its own; the minimum only grows it.
    assert_eq!(owner_at(AWAY.0 + 1, AWAY.1 + 1), Some(std::process::id()));
    //: And it reaches at least the documented minimum, so a two-pixel widget is genuinely covered.
    assert_eq!(owner_at(AWAY.0 + 50, AWAY.1 + 20), Some(std::process::id()));
    drop(cover);
}
