//! A window of our own, put deliberately over something else.
//!
//! **This exists to be in the way.** `measure/obstruction` asks whether an action still reaches a
//! control that something is covering, and the only way to ask is to cover one — so this makes a
//! plain always-on-top rectangle at a given place on the screen, and takes it away again.
//!
//! **It never takes the focus.** `WS_EX_NOACTIVATE` and `SW_SHOWNOACTIVATE` together mean the
//! window under it stays in the foreground, which the harness depends on: a measurement of *is the
//! target reachable while covered* would answer a different question if covering it also
//! deactivated the application.
//!
//! It is a harness instrument and nothing ships with it, but it lives here rather than in the
//! measurement because every other line of Win32 in this project does — and a second place that
//! registers window classes is a second place to get the lifetime wrong.

use std::sync::OnceLock;

use windows::core::w;
use windows::Win32::Foundation::{HWND, LPARAM, LRESULT, WPARAM};
use windows::Win32::Graphics::Gdi::CreateSolidBrush;
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::UI::WindowsAndMessaging::{
    CreateWindowExW, DefWindowProcW, DestroyWindow, DispatchMessageW, GetWindowLongPtrW,
    PeekMessageW, RegisterClassW, SetWindowPos, ShowWindow, TranslateMessage, GWL_EXSTYLE,
    HWND_BOTTOM, HWND_NOTOPMOST, HWND_TOPMOST, MSG, PM_REMOVE, SWP_NOACTIVATE, SWP_NOMOVE,
    SWP_NOSIZE, SW_SHOWNOACTIVATE, WNDCLASSW, WS_EX_NOACTIVATE, WS_EX_TOOLWINDOW, WS_EX_TOPMOST,
    WS_POPUP,
};

/// Registered once. A second registration of the same name fails, and the failure is the only
/// signal that would be lost — so it happens in one place and never again.
fn class() -> bool {
    static DONE: OnceLock<bool> = OnceLock::new();
    *DONE.get_or_init(|| unsafe {
        let Ok(instance) = GetModuleHandleW(None) else { return false };
        let class = WNDCLASSW {
            lpfnWndProc: Some(proc),
            hInstance: instance.into(),
            lpszClassName: w!("argmind.cover"),
            //: A solid brush so `DefWindowProc` fills it on its own. Without one the window shows
            //: whatever was behind it, which for something whose entire purpose is to be opaque
            //: would be a measurement that looked like it had run.
            hbrBackground: CreateSolidBrush(windows::Win32::Foundation::COLORREF(0x00_ee_f1_f2)),
            ..Default::default()
        };
        RegisterClassW(&class) != 0
    })
}

unsafe extern "system" fn proc(window: HWND, message: u32, w: WPARAM, l: LPARAM) -> LRESULT {
    DefWindowProcW(window, message, w, l)
}

/// An opaque rectangle sitting on top of the screen at one place.
pub struct Cover {
    window: HWND,
}

impl Cover {
    /// Put one over this rectangle, in screen coordinates.
    ///
    /// A minimum size because a widget can be a few pixels across and a cover smaller than the
    /// thing it covers is not covering it.
    pub fn over(x: i64, y: i64, w: i64, h: i64) -> Result<Self, String> {
        if !class() {
            return Err("the cover window class would not register".into());
        }
        unsafe {
            let instance = GetModuleHandleW(None).map_err(|e| format!("{e}"))?;
            let window = CreateWindowExW(
                WS_EX_TOPMOST | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE,
                w!("argmind.cover"), w!("argmind"), WS_POPUP,
                x as i32, y as i32, w.max(60) as i32, h.max(24) as i32,
                None, None, instance, None,
            ).map_err(|e| format!("the cover window would not open ({e})"))?;
            //: **No activation.** The application underneath has to stay in the foreground, or the
            //: measurement is about losing focus rather than about being covered.
            let _ = ShowWindow(window, SW_SHOWNOACTIVATE);
            let it = Cover { window };
            it.pump();
            Ok(it)
        }
    }

    /// Let it paint. A window whose messages nobody reads is a grey ghost, and a grey ghost is
    /// still opaque — but a harness that shows one has left something on somebody's screen that
    /// looks broken.
    pub fn pump(&self) {
        unsafe {
            let mut message = MSG::default();
            while PeekMessageW(&mut message, None, 0, 0, PM_REMOVE).as_bool() {
                let _ = TranslateMessage(&message);
                DispatchMessageW(&message);
            }
        }
    }

    /// Get out of the way without going away — **what the application would do before clicking.**
    ///
    /// Dropping topmost is not enough on its own; the window has to go to the bottom as well, or it
    /// is merely an ordinary window that is still in front of the one below it.
    pub fn stand_aside(&self) -> Result<(), String> {
        unsafe {
            SetWindowPos(self.window, HWND_NOTOPMOST, 0, 0, 0, 0,
                         SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE).map_err(|e| format!("{e}"))?;
            SetWindowPos(self.window, HWND_BOTTOM, 0, 0, 0, 0,
                         SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE).map_err(|e| format!("{e}"))?;
        }
        self.pump();
        Ok(())
    }

    /// Back in the way again.
    pub fn come_back(&self) -> Result<(), String> {
        unsafe {
            SetWindowPos(self.window, HWND_TOPMOST, 0, 0, 0, 0,
                         SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE).map_err(|e| format!("{e}"))?;
        }
        self.pump();
        Ok(())
    }

    /// Is it still claiming to be above everything?
    ///
    /// **What [`Cover::stand_aside`] is actually responsible for.** Whether dropping topmost hands
    /// the pixels back to the window underneath depends on there *being* one, so that is a question
    /// for the harness on a real desktop and not for a suite — a cover somewhere no other window
    /// reaches stays the window at that point however far down the order it goes.
    pub fn is_topmost(&self) -> bool {
        unsafe {
            let ex = GetWindowLongPtrW(self.window, GWL_EXSTYLE) as u32;
            ex & WS_EX_TOPMOST.0 != 0
        }
    }

    pub fn handle(&self) -> isize {
        self.window.0 as isize
    }
}

impl Drop for Cover {
    /// **Taken away on every path.** A borderless always-on-top rectangle left behind by a harness
    /// that panicked is a piece of somebody's screen they cannot click and cannot close.
    fn drop(&mut self) {
        unsafe {
            let _ = DestroyWindow(self.window);
        }
        self.pump();
    }
}
