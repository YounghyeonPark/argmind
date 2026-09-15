//! An icon by the clock, so there is something to close.
//!
//! **A program with no window is a program you cannot quit.** The interface lives in a browser tab,
//! and a tab is not the program — closing it leaves a server running, and nothing on screen says
//! so. A console window would say so and is not an interface; an invisible background process says
//! nothing at all and is worse. So: an icon, a menu with two items, and a name in the tooltip.
//!
//! No new dependency. `Shell_NotifyIconW` is in the same `windows` crate as everything else here.
//!
//! **The icon is removed on the way out, in every path.** Windows does not tidy up after a process
//! that leaves one behind: it stays by the clock, doing nothing, until somebody hovers over it —
//! which is a worse first impression than never having drawn one.

use std::sync::atomic::{AtomicBool, AtomicIsize, Ordering};
use std::sync::OnceLock;

use windows::core::w;
use windows::Win32::Foundation::{HWND, LPARAM, LRESULT, POINT, WPARAM};
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::UI::Shell::{Shell_NotifyIconW, NIF_ICON, NIF_MESSAGE, NIF_TIP, NIM_ADD,
                                NIM_DELETE, NOTIFYICONDATAW};
use windows::Win32::UI::WindowsAndMessaging::{
    AppendMenuW, CreatePopupMenu, CreateWindowExW, DefWindowProcW, DestroyMenu, DispatchMessageW,
    GetCursorPos, GetMessageW, LoadIconW, PostQuitMessage, RegisterClassW, SetForegroundWindow,
    TrackPopupMenu, TranslateMessage, IDI_APPLICATION, MF_STRING, MSG, TPM_BOTTOMALIGN,
    TPM_RIGHTALIGN, WINDOW_EX_STYLE, WM_APP, WM_COMMAND, WM_DESTROY, WM_LBUTTONDBLCLK,
    WM_RBUTTONUP, WNDCLASSW, WS_OVERLAPPED,
};

/// The message the icon sends this window when somebody clicks it.
const ICON_CLICKED: u32 = WM_APP + 1;
const OPEN: usize = 1;
const QUIT: usize = 2;

static WINDOW: AtomicIsize = AtomicIsize::new(0);
static DRAWN: AtomicBool = AtomicBool::new(false);
static URL: OnceLock<String> = OnceLock::new();

fn open_the_page() {
    if let Some(url) = URL.get() {
        //: `explorer` hands the URL to whatever the person chose as their browser.
        let _ = std::process::Command::new("explorer").arg(url).spawn();
    }
}

/// **Removed here and only here**, and safe to call twice — the two ways out of the menu and the
/// window's own destruction all arrive at it.
fn remove_icon() {
    if !DRAWN.swap(false, Ordering::SeqCst) {
        return;
    }
    let mut data = NOTIFYICONDATAW {
        cbSize: std::mem::size_of::<NOTIFYICONDATAW>() as u32,
        hWnd: HWND(WINDOW.load(Ordering::SeqCst) as _),
        uID: 1,
        ..Default::default()
    };
    unsafe {
        let _ = Shell_NotifyIconW(NIM_DELETE, &mut data);
    }
}

unsafe extern "system" fn proc(window: HWND, message: u32, w: WPARAM, l: LPARAM) -> LRESULT {
    match message {
        ICON_CLICKED => {
            match l.0 as u32 {
                WM_LBUTTONDBLCLK => open_the_page(),
                WM_RBUTTONUP => show_menu(window),
                _ => {}
            }
            LRESULT(0)
        }
        WM_COMMAND => {
            match w.0 & 0xffff {
                OPEN => open_the_page(),
                QUIT => {
                    remove_icon();
                    PostQuitMessage(0);
                }
                _ => {}
            }
            LRESULT(0)
        }
        //: Closed by anything at all — a log-off, a taskkill with a window message — still takes
        //: the icon with it.
        WM_DESTROY => {
            remove_icon();
            PostQuitMessage(0);
            LRESULT(0)
        }
        _ => DefWindowProcW(window, message, w, l),
    }
}

unsafe fn show_menu(window: HWND) {
    let Ok(menu) = CreatePopupMenu() else { return };
    let _ = AppendMenuW(menu, MF_STRING, OPEN, w!("argmind 열기"));
    let _ = AppendMenuW(menu, MF_STRING, QUIT, w!("끝내기"));
    let mut at = POINT::default();
    let _ = GetCursorPos(&mut at);
    //: **Foreground first, or the menu will not close.** A popup owned by a window that is not in
    //: front stays on screen after the click that dismissed it — a documented quirk, and one that
    //: makes the tray look broken.
    let _ = SetForegroundWindow(window);
    let _ = TrackPopupMenu(menu, TPM_RIGHTALIGN | TPM_BOTTOMALIGN, at.x, at.y, 0, window, None);
    let _ = DestroyMenu(menu);
}

/// Draw the icon and run the message loop until somebody chooses 끝내기.
///
/// **This blocks and it is meant to.** It is the main thread of an application whose real work is
/// on other threads; when it returns, the program is finished.
pub fn run(tooltip: &str, url: &str) -> windows::core::Result<()> {
    let _ = URL.set(url.to_string());
    unsafe {
        let instance: windows::Win32::Foundation::HINSTANCE = GetModuleHandleW(None)?.into();
        let class = WNDCLASSW {
            lpfnWndProc: Some(proc),
            hInstance: instance,
            lpszClassName: w!("argmind.tray"),
            ..Default::default()
        };
        RegisterClassW(&class);
        //: A real window rather than a message-only one: `Shell_NotifyIconW` needs somewhere to
        //: send to, and a message-only window cannot be brought to the foreground — which the menu
        //: above needs.
        let window = CreateWindowExW(WINDOW_EX_STYLE(0), w!("argmind.tray"), w!("argmind"),
                                     WS_OVERLAPPED, 0, 0, 0, 0, None, None, instance, None)?;
        WINDOW.store(window.0 as isize, Ordering::SeqCst);

        let mut data = NOTIFYICONDATAW {
            cbSize: std::mem::size_of::<NOTIFYICONDATAW>() as u32,
            hWnd: window,
            uID: 1,
            uFlags: NIF_ICON | NIF_MESSAGE | NIF_TIP,
            uCallbackMessage: ICON_CLICKED,
            //: The stock application icon. A resource of our own would be nicer and would mean a
            //: build step and a file; this needs neither and is recognisable enough beside a name.
            hIcon: LoadIconW(None, IDI_APPLICATION)?,
            ..Default::default()
        };
        for (i, c) in tooltip.encode_utf16().take(data.szTip.len() - 1).enumerate() {
            data.szTip[i] = c;
        }
        Shell_NotifyIconW(NIM_ADD, &mut data).ok()?;
        DRAWN.store(true, Ordering::SeqCst);

        let mut message = MSG::default();
        while GetMessageW(&mut message, None, 0, 0).as_bool() {
            let _ = TranslateMessage(&message);
            DispatchMessageW(&message);
        }
        //: Belt and braces: the loop ends on `WM_QUIT`, which the two paths above post *after*
        //: removing the icon — but a loop that ended some other way must not leave one behind.
        remove_icon();
    }
    Ok(())
}

/// Menu item ids, for a test that they are distinct and non-zero.
///
/// **Zero is not a menu id.** `TrackPopupMenu` uses it to mean *nothing was chosen*, so an item
/// numbered zero is an item whose selection cannot be told from a dismissed menu.
pub fn menu_ids() -> [usize; 2] {
    [OPEN, QUIT]
}

pub fn window_handle() -> isize {
    WINDOW.load(Ordering::SeqCst)
}

/// Is the icon on screen right now?
pub fn drawn() -> bool {
    DRAWN.load(Ordering::SeqCst)
}
