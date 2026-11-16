//! **A key, a click and typing addressed to a program's window, not sent down the person's input**
//! — since 2026-11-16.
//!
//! The owner: *컴퓨터작업중에는 사용자가 개입못하게 막는건 어때?*, and when told what a lock costs
//! (an administrator, or a global keyboard hook — the keylogger this project declined) and offered
//! *the person's touch stops argmind* instead: *건드려도 멈추지 않았으면 함.* So neither: argmind
//! stops using the person's input at all where it can. A control pattern already leaves the pointer
//! alone; what still went through `SendInput` was a key, a click no pattern takes, and typing no
//! value pattern takes — and each of those lands wherever the pointer or the focus is *at that
//! instant*, which is exactly what a person's touch changes.
//!
//! `PostMessageW` puts the message in the window's own queue. The person's pointer is not moved,
//! the foreground is not taken, and a person typing in another window does not interleave with it,
//! because nothing here touches the input stream they type into.
//!
//! **Measured before it was wired** (`examples/posted.rs`, a desktop of its own, the person's
//! pointer and foreground read before and after and unchanged every run): a classic Win32 window
//! took posted typing (`A-10422`), a posted backspace (`A-1042`) and a posted click on its button,
//! which read the box and drew its answer; **Edge** — Chromium, a profile of its own, *not* the
//! active window — took a posted click, `안녕 A-1` as characters, Enter (its page saw
//! `event.key === 'Enter'`) and a click on a button (its `onclick` ran); and **these hands**, choosing
//! the window themselves, found Chromium's page window under the box and did the same four through
//! it, none by `SendInput`. With another window lying over the box they refused all three acts as
//! `Elsewhere` — a message goes where a click would have and no further, so a covered target is
//! still not reached, as before. `tests/addressed.rs` drives them on a desktop of its own with
//! `SendInput` refused outright, and fails when a key stops following the window argmind just
//! acted on. One run each. **What was
//! not tried**: WinUI and UWP programs (Notepad, Settings, Calculator — packaged, so not opened from
//! a harness), WPF, Qt, menus, and anything dragged. A control that reads the pointer itself rather
//! than the message (`GetCursorPos`, `GetMessagePos`) will see the person's pointer, and a posted
//! press that lands and does nothing shows up as *화면이 바뀌지 않음*, as any press does.
//!
//! **A posted key is translated by the program's own loop with its own key state**, so a Shift the
//! person is holding in a different program does not reach it — the half of *not interleaving*
//! that `SendInput` could not offer.

use windows::Win32::Foundation::{HWND, LPARAM, POINT, WPARAM};
use windows::Win32::Graphics::Gdi::ScreenToClient;
use windows::Win32::UI::Input::KeyboardAndMouse::{MapVirtualKeyW, MAPVK_VK_TO_VSC};
use windows::Win32::UI::WindowsAndMessaging::{
    GetAncestor, GetGUIThreadInfo, GetWindowThreadProcessId, IsWindow, PostMessageW, WindowFromPoint,
    GA_ROOTOWNER, GUITHREADINFO, WM_CHAR, WM_KEYDOWN, WM_KEYUP, WM_LBUTTONDOWN, WM_LBUTTONUP,
    WM_MBUTTONDOWN, WM_MBUTTONUP, WM_MOUSEMOVE, WM_RBUTTONDOWN, WM_RBUTTONUP,
};

/// **The keys a keyboard sends with the extended bit**: the navigation block and the arrows. A
/// program that reads the scan code tells the arrow from the number pad's 4 by this bit alone.
pub fn extended(vk: u16) -> bool {
    matches!(vk, 0x21..=0x28 | 0x2D | 0x2E)
}

/// A key message's `lParam`: one repeat, the scan code, the extended bit, and for a release the
/// previous-state and transition bits a real key-up carries.
pub fn key_bits(scan: u32, extended: bool, up: bool) -> isize {
    let mut l: u32 = 1 | ((scan & 0xff) << 16);
    if extended {
        l |= 1 << 24;
    }
    if up {
        l |= (1 << 30) | (1 << 31);
    }
    l as isize
}

/// A mouse message's `lParam`: client x in the low word, y in the high, each a signed 16 bits —
/// a point left of or above the client area is negative and must stay so.
pub fn point_bits(x: i32, y: i32) -> isize {
    (((y as i16 as u16 as u32) << 16) | (x as i16 as u16 as u32)) as isize
}

/// The down and up messages for a button, and the `wParam` flag the down carries.
pub fn mouse_messages(button: &str) -> Option<(u32, u32, usize)> {
    Some(match button {
        "left" => (WM_LBUTTONDOWN, WM_LBUTTONUP, 0x0001),
        "right" => (WM_RBUTTONDOWN, WM_RBUTTONUP, 0x0002),
        "middle" => (WM_MBUTTONDOWN, WM_MBUTTONUP, 0x0010),
        _ => return None,
    })
}

fn post(h: HWND, m: u32, w: usize, l: isize) -> Result<(), String> {
    unsafe { PostMessageW(h, m, WPARAM(w), LPARAM(l)) }
        .map_err(|e| format!("the window would not take a message ({e})"))
}

/// **The window a real click at this point would reach** — the same `WindowFromPoint` the reach
/// guard asks, so the guard and the message agree about who is there.
pub(crate) fn window_at(x: i64, y: i64) -> Option<HWND> {
    let h = unsafe { WindowFromPoint(POINT { x: x as i32, y: y as i32 }) };
    (!h.is_invalid()).then_some(h)
}

pub(crate) fn click(h: HWND, x: i64, y: i64, button: &str) -> Result<(), String> {
    let (down, up, held) = mouse_messages(button).ok_or_else(|| format!("unknown button {button:?}"))?;
    let mut p = POINT { x: x as i32, y: y as i32 };
    unsafe {
        let _ = ScreenToClient(h, &mut p);
    }
    let at = point_bits(p.x, p.y);
    //: A move first, so a control that lights up under the pointer before it takes a press does.
    post(h, WM_MOUSEMOVE, 0, at)?;
    post(h, down, held, at)?;
    post(h, up, 0, at)
}

pub(crate) fn key(h: HWND, vk: u16) -> Result<(), String> {
    let scan = unsafe { MapVirtualKeyW(vk as u32, MAPVK_VK_TO_VSC) };
    let ext = extended(vk);
    post(h, WM_KEYDOWN, vk as usize, key_bits(scan, ext, false))?;
    post(h, WM_KEYUP, vk as usize, key_bits(scan, ext, true))
}

/// **Characters, not keys** — UTF-16 units in order, which a Unicode window takes as text with no
/// input method in between, as `KEYEVENTF_UNICODE` did.
pub(crate) fn chars(h: HWND, units: &[u16]) -> Result<(), String> {
    for &u in units {
        post(h, WM_CHAR, u as usize, 1)?;
    }
    Ok(())
}

/// **One program's window and the dialogs it owns**, by the root of the owner chain. A modal
/// dialog is owned by the window that opened it; another Chrome window is owned by nothing, so two
/// windows of one browser are two families even though one thread runs both.
pub(crate) fn same_family(a: HWND, b: HWND) -> bool {
    unsafe { GetAncestor(a, GA_ROOTOWNER) == GetAncestor(b, GA_ROOTOWNER) }
}

/// **Where a key for this window goes** — its thread's focus when that is in the same family (a
/// box inside it, or a dialog it opened), the window itself otherwise.
///
/// **Not the person's focus.** Focus here is per thread, and the person working in another program
/// moves theirs and not this one's. The family check is for the one case where it would: a browser
/// runs every window on one thread, and the person clicking into a second window of it would move
/// that thread's focus there.
pub(crate) fn focus_of(h: HWND) -> HWND {
    unsafe {
        let tid = GetWindowThreadProcessId(h, None);
        let mut info = GUITHREADINFO { cbSize: std::mem::size_of::<GUITHREADINFO>() as u32, ..Default::default() };
        if tid != 0 && GetGUIThreadInfo(tid, &mut info).is_ok() {
            for f in [info.hwndFocus, info.hwndActive] {
                if !f.is_invalid() && same_family(f, h) {
                    return f;
                }
            }
        }
        h
    }
}

pub(crate) fn alive(h: HWND) -> bool {
    unsafe { IsWindow(h) }.as_bool()
}

pub(crate) fn pid(h: HWND) -> Option<u32> {
    crate::launch::pid_of(h.0 as isize)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_key_message_carries_what_a_real_key_does() {
        //: Enter: scan 0x1C, not extended. Down is one repeat and the scan; up adds bits 30 and 31.
        assert_eq!(key_bits(0x1C, false, false), 0x001C_0001);
        assert_eq!(key_bits(0x1C, false, true), 0xC01C_0001u32 as isize);
        assert!(key_bits(0x1C, false, true) > 0, "a key-up's lParam must not go negative on 64 bits");
        //: An arrow is extended; the number pad's 4 shares its scan code and is not.
        assert_eq!(key_bits(0x4B, true, false), 0x014B_0001);
        for vk in [0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x2D, 0x2E] {
            assert!(extended(vk), "{vk:#x}");
        }
        for vk in [0x0D, 0x09, 0x1B, 0x20, 0x08] {
            assert!(!extended(vk), "{vk:#x}");
        }
    }

    #[test]
    fn a_point_left_of_the_window_stays_negative() {
        assert_eq!(point_bits(10, 20), (20 << 16) | 10);
        let l = point_bits(-5, 7);
        assert_eq!((l & 0xffff) as u16 as i16, -5);
        assert_eq!(((l >> 16) & 0xffff) as u16 as i16, 7);
        let l = point_bits(3, -9);
        assert_eq!(((l >> 16) & 0xffff) as u16 as i16, -9);
    }

    #[test]
    fn each_button_has_its_own_messages_and_an_unknown_one_none() {
        let all: Vec<_> = ["left", "right", "middle"].iter().map(|b| mouse_messages(b).unwrap()).collect();
        for i in 0..all.len() {
            for j in i + 1..all.len() {
                assert_ne!(all[i].0, all[j].0);
                assert_ne!(all[i].2, all[j].2);
            }
        }
        assert!(mouse_messages("back").is_none());
    }
}
