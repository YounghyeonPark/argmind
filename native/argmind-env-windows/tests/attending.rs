//! Which window the agent is looking at — the rule, without a desktop.
//!
//! **The rule changed on 2026-09-17 and these changed with it.** It used to be *the newest window
//! that is not the one in front*, which was the best available while the page was a tab in
//! somebody else's process: the interface could not be identified, only avoided by position. The
//! page now lives in a window this process owns, so the rule is `app.py`'s original and sharper
//! one — **the newest window that is not ours.**
//!
//! What forced it is in `attending.rs`: the old rule answered *argmind* whenever the person was
//! working in the window they had switched to from argmind, and with watching on by default that
//! was **4 of 8 notices over 150 seconds**, each one a model call spent reading argmind's own
//! transcript. The prediction written before the change was zero; the re-run gave zero.
//!
//! Ownership is handed in here rather than asked of Windows. `ours` is a process-id call, so a test
//! against the real one could only use handles belonging to nobody — and *the person is in our own
//! window* is the whole case the rule turns on.

use argmind_env_windows::Watcher;

/// Our own window, for a test that has no process to ask about.
const MINE: isize = 900;
fn mine(h: isize) -> bool {
    h == MINE
}

/// **When the person is in something else, that is the answer.**
///
/// The case the old rule got wrong: it excluded whatever was in front, so working in the browser
/// after coming from argmind read *argmind*.
#[test]
fn the_window_in_front_is_the_answer_whenever_it_is_not_ours() {
    let w = Watcher::default();
    //: Paint, then argmind, then the browser — the ordinary shape of using this program.
    w.saw(100);
    w.saw(MINE);
    assert_eq!(w.attended_where(200, mine), Some(200), "the browser is what they are in");
    //: And back to Paint.
    assert_eq!(w.attended_where(100, mine), Some(100));
}

/// **When the person is in argmind, it is whatever they came from.** The reason this file exists:
/// answering with the interface would have the agent describe its own transcript back to them.
#[test]
fn our_own_window_in_front_reads_the_one_behind_it() {
    let w = Watcher::default();
    w.saw(100);
    w.saw(200);
    assert_eq!(w.attended_where(MINE, mine), Some(200), "the newest that is not ours");
    //: Ours never becomes the answer however many times it is in front.
    for _ in 0..5 {
        assert_eq!(w.attended_where(MINE, mine), Some(200));
    }
}

/// **Nothing rather than the interface.**
///
/// A fresh start where the only window ever seen is ours. Answering with it would make the agent
/// describe its own transcript; answering with nothing makes it say it cannot see anything, which
/// is true.
#[test]
fn a_first_run_attends_nothing_rather_than_itself() {
    let w = Watcher::default();
    assert_eq!(w.attended_where(MINE, mine), None, "only our own window has ever been seen");
    w.saw(MINE);
    assert_eq!(w.attended_where(MINE, mine), None);
    w.saw(300);
    assert_eq!(w.attended_where(300, mine), Some(300));
}

/// Alt-tabbing back and forth does not fill the list with one window.
#[test]
fn a_window_returning_to_the_front_moves_rather_than_repeats() {
    let w = Watcher::default();
    for h in [100, 200, 100, 200, 100] {
        w.saw(h);
    }
    assert_eq!(w.attended_where(100, mine), Some(100));
    //: Going into argmind from Paint reads Paint; the repeats did not bury the second window.
    assert_eq!(w.attended_where(MINE, mine), Some(100));
    w.saw(200);
    assert_eq!(w.attended_where(MINE, mine), Some(200));
}

/// The list is short and **bounded**, because an unbounded one is a leak nobody notices for a week.
#[test]
fn the_history_does_not_grow_without_end() {
    let w = Watcher::default();
    for h in 1..=50 {
        w.saw(h);
    }
    //: **Ours at the head and three others behind it**, which is what four has to hold for the
    //: rule to have anything to fall back to.
    w.saw(MINE);
    assert_eq!(w.attended_where(MINE, mine), Some(50));
    //: An old window is forgotten rather than resurrected.
    for h in [MINE, 48, 49, 50] {
        w.saw(h);
    }
    assert_eq!(w.attended_where(MINE, mine), Some(50));
}

/// A handle of zero is not a window — the foreground can genuinely be nothing.
#[test]
fn nothing_in_front_is_not_recorded_as_a_window() {
    let w = Watcher::default();
    assert_eq!(w.attended_where(0, mine), None);
    w.saw(100);
    assert_eq!(w.attended_where(0, mine), Some(100), "the zero did not displace the real one");
}

/// **Every window being ours is the same as none**, and the two must not be told apart by accident:
/// a second argmind window — the tray menu, a dialog — cannot become the thing it reads.
#[test]
fn several_windows_of_ours_are_still_nothing() {
    let w = Watcher::default();
    for h in [900, 901, 902] {
        w.saw(h);
    }
    assert_eq!(w.attended_where(901, |h| (900..=902).contains(&h)), None);
}
