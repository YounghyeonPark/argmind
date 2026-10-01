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

fn is_password(automation: &windows::Win32::UI::Accessibility::IUIAutomation, h: HWND) -> bool {
    unsafe {
        automation.ElementFromHandle(h).and_then(|e| e.CurrentIsPassword())
            .map(|b| b.as_bool()).expect("UI Automation answered")
    }
}

#[test]
fn a_password_box_says_so_and_an_ordinary_one_does_not() {
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
