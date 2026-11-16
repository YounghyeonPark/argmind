//! **The shipped hands, with what no pattern takes posted to the window** — on a desktop of this
//! test's own, so nothing it does can reach the screen of whoever runs it, and with
//! `never_send_input` so that a window refusing a message is a refusal here rather than a key on
//! their keyboard. The person's pointer and foreground window are read before and after.
//!
//! A window this test makes holds a classic edit box that already says `먼저`, and a pad of a class
//! of its own that offers UI Automation nothing and counts the clicks it receives. Patterns are off,
//! so every act takes the posted path — which is the path under test.

#![cfg(windows)]

use std::sync::atomic::{AtomicI32, Ordering};
use std::time::Duration;

use argmind_core::contract::Widget;
use argmind_env_windows::hands::{Did, AIM_LASTS};
use argmind_env_windows::WindowsHands;
use windows::core::{w, PCWSTR};
use windows::Win32::Foundation::{HWND, LPARAM, LRESULT, POINT, RECT, WPARAM};
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::System::StationsAndDesktops::{CloseDesktop, CreateDesktopW, SetThreadDesktop, DESKTOP_CONTROL_FLAGS,
                                                   HDESK};
use windows::Win32::UI::WindowsAndMessaging::*;

static CLICKS: AtomicI32 = AtomicI32::new(0);
static AT_X: AtomicI32 = AtomicI32::new(-1);
static AT_Y: AtomicI32 = AtomicI32::new(-1);

unsafe extern "system" fn pad(h: HWND, m: u32, wp: WPARAM, lp: LPARAM) -> LRESULT {
    if m == WM_LBUTTONUP {
        CLICKS.fetch_add(1, Ordering::SeqCst);
        AT_X.store((lp.0 & 0xffff) as u16 as i16 as i32, Ordering::SeqCst);
        AT_Y.store(((lp.0 >> 16) & 0xffff) as u16 as i16 as i32, Ordering::SeqCst);
        return LRESULT(0);
    }
    DefWindowProcW(h, m, wp, lp)
}

fn person() -> (i32, i32, isize) {
    let mut p = POINT::default();
    unsafe {
        let _ = GetCursorPos(&mut p);
        (p.x, p.y, GetForegroundWindow().0 as isize)
    }
}

fn text(h: HWND) -> String {
    let mut b = [0u16; 256];
    let n = unsafe { GetWindowTextW(h, &mut b) }.max(0) as usize;
    String::from_utf16_lossy(&b[..n])
}

fn rect(h: HWND) -> (i64, i64, i64, i64) {
    let mut r = RECT::default();
    unsafe { let _ = GetWindowRect(h, &mut r); }
    (r.left as i64, r.top as i64, (r.right - r.left) as i64, (r.bottom - r.top) as i64)
}

fn settle() {
    std::thread::sleep(Duration::from_millis(300));
}

#[test]
fn what_no_pattern_takes_reaches_the_window_and_never_the_persons_input() {
    let before = person();
    let name: Vec<u16> = format!("argmind-addressed-{}", std::process::id()).encode_utf16().chain([0]).collect();
    let desk: HDESK = unsafe { CreateDesktopW(PCWSTR(name.as_ptr()), None, None, DESKTOP_CONTROL_FLAGS(0),
                                              0x1000_0000, None) }.expect("a desktop of its own");
    let desk_n = desk.0 as isize;

    //: The window's thread: on that desktop, making the window, pumping its messages.
    let (made, handles) = std::sync::mpsc::channel::<(isize, isize, isize)>();
    let window = std::thread::spawn(move || unsafe {
        SetThreadDesktop(HDESK(desk_n as _)).expect("on it");
        let inst = GetModuleHandleW(None).unwrap();
        let class = WNDCLASSW { lpfnWndProc: Some(pad), hInstance: inst.into(),
                                lpszClassName: w!("argmind-addressed-pad"), ..Default::default() };
        RegisterClassW(&class);
        let top = CreateWindowExW(Default::default(), w!("STATIC"), w!("argmind addressed test"),
                                  WS_OVERLAPPEDWINDOW | WS_VISIBLE, 100, 100, 460, 200, None, None, inst, None)
            .unwrap();
        let edit = CreateWindowExW(Default::default(), w!("EDIT"), w!("먼저"),
                                   WS_CHILD | WS_VISIBLE | WS_BORDER | WINDOW_STYLE(ES_AUTOHSCROLL as u32),
                                   10, 10, 260, 24, top, None, inst, None).unwrap();
        let pad = CreateWindowExW(Default::default(), w!("argmind-addressed-pad"), w!(""),
                                  WS_CHILD | WS_VISIBLE, 10, 60, 120, 60, top, None, inst, None).unwrap();
        made.send((top.0 as isize, edit.0 as isize, pad.0 as isize)).unwrap();
        let mut msg = MSG::default();
        while GetMessageW(&mut msg, None, 0, 0).as_bool() {
            let _ = TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
    });
    let (top, edit, pad) = handles.recv_timeout(Duration::from_secs(5)).expect("the window was made");
    let (top, edit, pad) = (HWND(top as _), HWND(edit as _), HWND(pad as _));

    //: The hands' thread, on the same desktop. **Patterns off and `SendInput` refused**: every act
    //: below is a message or nothing.
    let (edit_n, pad_n) = (edit.0 as isize, pad.0 as isize);
    let report = std::thread::spawn(move || {
        unsafe { SetThreadDesktop(HDESK(desk_n as _)).expect("on it") };
        let (edit, pad) = (HWND(edit_n as _), HWND(pad_n as _));
        let mut hands = WindowsHands::only_within(std::process::id()).with_messages().never_send_input().arm();
        let box_ = Widget::new("", "edit", rect(edit), "argmind addressed test");
        let pad_w = Widget::new("", "pane", rect(pad), "argmind addressed test");
        let mut said = Vec::new();
        said.push(("click the box", hands.click_button(&box_, "left"), String::new()));
        settle();
        let d = hands.type_string("안녕");
        settle();
        said.push(("type into a box that holds something", d, text(edit)));
        let d = hands.press_key("backspace");
        settle();
        said.push(("backspace after it", d, text(edit)));
        let d = hands.click_button(&pad_w, "left");
        settle();
        said.push(("click a pad with no pattern", d, String::new()));
        //: **A key with nothing fresh before it** goes to the window the person was in — here the
        //: runner's own, which these hands may not touch, so it must not be posted.
        std::thread::sleep(AIM_LASTS + Duration::from_millis(300));
        let d = hands.press_key("enter");
        said.push(("enter with nothing fresh", d, String::new()));
        let (_, _, w, h) = rect(pad);
        (said, hands.by_message, hands.by_click, hands.clicked, (w as i32 / 2, h as i32 / 2))
    }).join().expect("the hands' thread");

    unsafe { let _ = PostMessageW(top, WM_CLOSE, WPARAM(0), LPARAM(0)); }
    unsafe { let _ = PostMessageW(top, WM_QUIT, WPARAM(0), LPARAM(0)); }
    let _ = window.join();
    unsafe { let _ = CloseDesktop(desk); }
    let after = person();

    let (said, by_message, by_click, clicked, middle) = report;
    for (what, did, seen) in &said {
        eprintln!("{what}: {did:?} {seen:?}");
    }
    let posted = Did::Operated { how: "posted", live: true };
    assert_eq!(said[0].1, posted, "the click into the box");
    assert_eq!((said[1].1.clone(), said[1].2.as_str()), (posted.clone(), "먼저안녕"),
               "typing is inserted at the caret, not put in place of what was there");
    assert_eq!((said[2].1.clone(), said[2].2.as_str()), (posted.clone(), "먼저안"),
               "the key after the typing went to the box argmind was working in");
    assert_eq!(said[3].1, posted, "the click on the pad");
    assert_eq!(CLICKS.load(Ordering::SeqCst), 1, "the pad counted one click");
    assert_eq!((AT_X.load(Ordering::SeqCst), AT_Y.load(Ordering::SeqCst)), middle,
               "the click arrived at the pad's own middle, in its own coordinates");
    assert!(!matches!(said[4].1, Did::Operated { live: true, .. } | Did::Sent { live: true }),
            "a key with nothing fresh before it reached a window these hands may not touch: {:?}", said[4].1);
    assert_eq!((by_message, by_click, clicked), (4, 0, 0), "something went another way");
    assert_eq!((before.0, before.1, before.2), (after.0, after.1, after.2),
               "the person's pointer or foreground window moved (or they moved it themselves)");
}

/// **Hands that may not send input refuse instead**, whichever act and before the dry check, so a
/// test that turns `never_send_input` on cannot reach its runner's keyboard or pointer even with
/// every other path off.
#[test]
fn hands_that_may_not_send_input_refuse_rather_than_send_it() {
    let mut hands = WindowsHands::unrestricted().never_send_input().arm();
    let w = Widget::new("Copy", "button", (100, 100, 40, 20), "Character Map");
    for did in [hands.click_button(&w, "left"), hands.press_key("enter"), hands.type_string("x")] {
        assert!(matches!(&did, Did::Refused(why) if why.contains("only the person's")), "{did:?}");
    }
    assert_eq!((hands.clicked, hands.by_click, hands.by_message), (0, 0, 0));
}
