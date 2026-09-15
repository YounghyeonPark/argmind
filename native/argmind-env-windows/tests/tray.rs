//! The icon by the clock — the part that can be checked without drawing one.
//!
//! `run` blocks on a message loop and draws on a real desktop, so it is not something a suite can
//! call. What a suite can check is the arithmetic that would be wrong silently: a menu item
//! numbered zero, two items sharing a number, and an icon that is reported as drawn before
//! anything has been drawn.

use argmind_env_windows::tray::{drawn, menu_ids, window_handle};

/// **Zero is not a menu id.**
///
/// `TrackPopupMenu` returns zero to mean *nothing was chosen*, so an item numbered zero is an item
/// whose selection cannot be told from a dismissed menu — the person clicks 끝내기 and nothing
/// happens, which reads as the program having hung.
#[test]
fn no_menu_item_is_numbered_zero() {
    let ids = menu_ids();
    assert!(ids.iter().all(|id| *id != 0), "{ids:?}");
}

/// Two items sharing a number is one item wearing two labels.
#[test]
fn every_menu_item_has_its_own_number() {
    let ids = menu_ids();
    let distinct: std::collections::BTreeSet<&usize> = ids.iter().collect();
    assert_eq!(distinct.len(), ids.len(), "{ids:?}");
}

/// **Nothing is drawn until something draws it**, and the flag says so.
///
/// The flag is what stops the icon being removed twice and, more to the point, what makes the
/// removal on the way out safe to call from three places.
#[test]
fn nothing_is_drawn_before_the_loop_runs() {
    assert!(!drawn(), "the icon reported itself drawn without a message loop");
    assert_eq!(window_handle(), 0, "there is no window before one is created");
}
