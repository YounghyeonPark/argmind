//! **Does UI Automation say *password* about a password box?** The whole hand-off rests on it.
//!
//! Two edit controls in a window this test makes and never shows — one with `ES_PASSWORD`, one
//! without — read by handle. **Nothing is clicked, nothing is focused, nothing comes to the
//! front**, so this runs beside whatever the person running it is doing. Both directions, because a
//! property that read `true` for everything would pass the first case alone.

#![cfg(windows)]

use windows::core::w;
use windows::Win32::Foundation::HWND;
use windows::Win32::UI::WindowsAndMessaging::{
    CreateWindowExW, DestroyWindow, ES_PASSWORD, WINDOW_EX_STYLE, WINDOW_STYLE, WS_CHILD,
    WS_OVERLAPPEDWINDOW,
};

/// **One window test at a time.** Two threads each making a window and asking UI Automation about
/// it, with neither thread processing messages, read `0x80004005` on whichever asked second — three
/// runs in parallel failed, three in series passed. The lock is the test's, not the code's.
static ONE_AT_A_TIME: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn is_password(automation: &windows::Win32::UI::Accessibility::IUIAutomation, h: HWND) -> bool {
    unsafe {
        automation.ElementFromHandle(h).and_then(|e| e.CurrentIsPassword())
            .map(|b| b.as_bool()).expect("UI Automation answered")
    }
}

#[test]
fn a_password_box_says_so_and_an_ordinary_one_does_not() {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let automation = argmind_env_windows::open_automation().expect("UI Automation");
    unsafe {
        //: A plain top-level window, never shown, to hold the two boxes.
        let parent = CreateWindowExW(WINDOW_EX_STYLE(0), w!("STATIC"), w!("argmind secret test"),
                                     WS_OVERLAPPEDWINDOW, 0, 0, 300, 100, None, None, None, None)
            .expect("a window");
        let secret = CreateWindowExW(WINDOW_EX_STYLE(0), w!("EDIT"), w!(""),
                                     WS_CHILD | WINDOW_STYLE(ES_PASSWORD as u32),
                                     0, 0, 200, 20, parent, None, None, None).expect("an edit");
        let plain = CreateWindowExW(WINDOW_EX_STYLE(0), w!("EDIT"), w!(""), WS_CHILD,
                                    0, 30, 200, 20, parent, None, None, None).expect("an edit");
        let (s, p) = (is_password(&automation, secret), is_password(&automation, plain));
        let _ = DestroyWindow(parent);
        assert!(s, "ES_PASSWORD read as an ordinary box");
        assert!(!p, "an ordinary box read as a password");
    }
}

/// **And whether a control takes text** — the question asked before every `type` since 2026-10-21.
/// An edit box yes, a button no, a read-only edit no: both directions, in the same hidden window.
#[test]
fn an_edit_takes_text_and_a_button_and_a_read_only_box_do_not() {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    use windows::Win32::UI::WindowsAndMessaging::ES_READONLY;
    let automation = argmind_env_windows::open_automation().expect("UI Automation");
    unsafe {
        let parent = CreateWindowExW(WINDOW_EX_STYLE(0), w!("STATIC"), w!("argmind takes-text test"),
                                     WS_OVERLAPPEDWINDOW, 0, 0, 300, 100, None, None, None, None)
            .expect("a window");
        let edit = CreateWindowExW(WINDOW_EX_STYLE(0), w!("EDIT"), w!(""), WS_CHILD,
                                   0, 0, 200, 20, parent, None, None, None).expect("an edit");
        let button = CreateWindowExW(WINDOW_EX_STYLE(0), w!("BUTTON"), w!("눌러"), WS_CHILD,
                                     0, 30, 200, 20, parent, None, None, None).expect("a button");
        let fixed = CreateWindowExW(WINDOW_EX_STYLE(0), w!("EDIT"), w!("고정"),
                                    WS_CHILD | WINDOW_STYLE(ES_READONLY as u32),
                                    0, 60, 200, 20, parent, None, None, None).expect("an edit");
        let takes = |h, what: &str| match automation.ElementFromHandle(h) {
            Ok(el) => argmind_env_windows::patterns::takes_text(&el),
            Err(err) => panic!("{what}: {err}"),
        };
        let (e, b, f) = (takes(edit, "edit"), takes(button, "button"), takes(fixed, "read-only"));
        let _ = DestroyWindow(parent);
        assert!(e, "an edit box read as not taking text");
        assert!(!b, "a button read as taking text");
        //: A read-only edit is still an Edit by control type — the case that says the control
        //: type alone is not enough.
        assert!(!f, "a read-only box read as taking text");
    }
}

/// **The time since the last input is a time**: non-negative, and on a machine somebody is using
/// it is not decades. Reading it moves nothing and types nothing.
#[test]
fn the_time_since_input_is_a_sensible_number() {
    let s = argmind_env_windows::seconds_since_input().expect("Windows answers");
    assert!((0.0..60.0 * 60.0 * 24.0 * 50.0).contains(&s), "{s}");
}
