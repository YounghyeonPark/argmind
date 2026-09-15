//! Which window the agent is looking at — the rule, without a desktop.
//!
//! **The interface is in front whenever the person asks for anything**, so "what is in front" is
//! always the wrong answer. The rule is *the newest window that is not the one in front*, and these
//! check it against the sequences a person actually produces: alt-tabbing between two windows, and
//! a fresh start where nothing else has been seen yet.

use argmind_env_windows::Watcher;

#[test]
fn the_window_being_attended_is_the_one_before_the_interface() {
    let w = Watcher::default();
    //: The person works in Paint, then reaches for the browser.
    w.saw(100);
    w.saw(200);
    //: With the browser (200) in front, the agent reads Paint.
    assert_eq!(w.attended(200), Some(100));
    //: And with Paint in front — a voice turn, one day — it reads Paint, not the browser.
    assert_eq!(w.attended(100), Some(200));
}

/// **Nothing rather than the interface.**
///
/// A fresh start where the only window ever seen is the one the person is reading. Answering with
/// it would make the agent describe its own transcript back to them; answering with nothing makes
/// it say it cannot see anything, which is true.
#[test]
fn a_first_run_attends_nothing_rather_than_itself() {
    let w = Watcher::default();
    assert_eq!(w.attended(200), None, "nothing has been seen at all");
    w.saw(200);
    assert_eq!(w.attended(200), None, "the only window seen is the one in front");
    w.saw(300);
    assert_eq!(w.attended(300), Some(200));
}

/// Alt-tabbing back and forth does not fill the list with one window.
#[test]
fn a_window_returning_to_the_front_moves_rather_than_repeats() {
    let w = Watcher::default();
    for h in [100, 200, 100, 200, 100] {
        w.saw(h);
    }
    assert_eq!(w.attended(100), Some(200));
    assert_eq!(w.attended(200), Some(100));
    //: A third window pushes in front of both, and the one before it is still findable.
    w.saw(300);
    assert_eq!(w.attended(300), Some(100));
}

/// The list is short and **bounded**, because an unbounded one is a leak nobody notices for a week.
#[test]
fn the_history_does_not_grow_without_end() {
    let w = Watcher::default();
    for h in 1..=50 {
        w.saw(h);
    }
    //: Only the recent few survive, so an old window is forgotten rather than resurrected.
    assert_eq!(w.attended(50), Some(49));
    assert_eq!(w.attended(1), Some(50), "window 1 fell out and 50 is simply the newest");
}

/// A handle of zero is not a window — the foreground can genuinely be nothing.
#[test]
fn nothing_in_front_is_not_recorded_as_a_window() {
    let w = Watcher::default();
    w.saw(0);
    assert_eq!(w.attended(123), None);
    w.saw(100);
    w.saw(0);
    assert_eq!(w.attended(0), Some(100), "the zero did not displace the real one");
}
