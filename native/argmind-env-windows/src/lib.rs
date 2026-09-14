//! UI Automation for eyes — the deterministic half, natively.
//!
//! **This is the measurement the whole design rests on**: asked *"where is this button in pixels?"*
//! a vision model took 24–82 s and on a dense screen never answered; UI Automation answers exactly
//! in 41–266 ms, in the interface's own language. So the OS says WHERE and the model says WHAT, and
//! nothing in `argmind-core` knows this file exists.
//!
//! A port of `argmind/env/windows.py`'s reading half, line for line where it matters:
//! `ElementFromHandle`, then `FindAll(Descendants, IsOffscreen = false)`, then name, localized
//! control type and bounding rectangle, keeping only what has a name and a positive area.
//!
//! Clicking lives in [`hands`], where the standing rule — *never click anything the harness did
//! not launch itself* — is a type rather than a paragraph: `Reach::OnlyProcess` looks up who owns
//! the window under the point before every click and refuses anything else.

pub mod hands;
pub mod launch;
pub mod speaker;
pub use hands::{Reach, WindowsHands};
pub use launch::Launched;
pub use speaker::SapiSpeaker;

use argmind_core::contract::{Eyes, Widget};
use windows::Win32::Foundation::{HWND, POINT};
use windows::Win32::System::Com::{
    CoCreateInstance, CoInitializeEx, CLSCTX_INPROC_SERVER, COINIT_MULTITHREADED,
};
use windows::Win32::UI::Accessibility::{
    CUIAutomation, IUIAutomation, IUIAutomationElement, TreeScope_Descendants,
    UIA_IsOffscreenPropertyId,
};
use windows::Win32::UI::WindowsAndMessaging::{GetCursorPos, GetForegroundWindow,
    GetSystemMetrics, SetCursorPos, SM_CXSCREEN, SM_CXVIRTUALSCREEN, SM_CYSCREEN,
    SM_CYVIRTUALSCREEN};

pub struct WindowsEyes {
    automation: IUIAutomation,
    /// Elements that disappeared between being listed and being read. **Counted rather than
    /// swallowed**: it is the rate the whole re-resolve-before-acting design is built on.
    pub vanished: u64,
}

impl WindowsEyes {
    /// Opens UI Automation. Fails here rather than later: eyes that cannot see should say so when
    /// they are built.
    pub fn new() -> windows::core::Result<Self> {
        unsafe {
            //: The apartment the calling thread already has is taken as given — `CoInitializeEx`
            //: returning `RPC_E_CHANGED_MODE` means somebody got here first, which is not an error
            //: for a reader. The Python side learned this the hard way with comtypes.
            let _ = CoInitializeEx(None, COINIT_MULTITHREADED);
            let automation: IUIAutomation =
                CoCreateInstance(&CUIAutomation, None, CLSCTX_INPROC_SERVER)?;
            Ok(WindowsEyes { automation, vanished: 0 })
        }
    }

    /// The named things in one window, given its handle.
    ///
    /// Split out from [`Eyes::foreground`] for the same reason it is in the Python: a fixture
    /// capture has to read a window **without bringing it to the front**, because on a desktop
    /// somebody is using, stealing focus both disturbs them and captures whatever they were doing
    /// instead of what was asked for.
    pub fn read(&mut self, handle: isize) -> Vec<Widget> {
        if handle == 0 {
            return Vec::new();
        }
        unsafe {
            let root: IUIAutomationElement = match self.automation.ElementFromHandle(HWND(handle as _)) {
                Ok(e) => e,
                Err(_) => return Vec::new(),
            };
            let window = root.CurrentName().map(|s| s.to_string()).unwrap_or_default();
            let visible = match self
                .automation
                .CreatePropertyCondition(UIA_IsOffscreenPropertyId, &false.into())
            {
                Ok(c) => c,
                Err(_) => return Vec::new(),
            };
            let found = match root.FindAll(TreeScope_Descendants, &visible) {
                Ok(f) => f,
                Err(_) => return Vec::new(),
            };
            let n = found.Length().unwrap_or(0);
            let mut out = Vec::with_capacity(n as usize);
            for i in 0..n {
                //: Every read of every element can fail on its own, because the tree is live. A
                //: `?` here would throw away the whole screen for one control that closed.
                let element = match found.GetElement(i) {
                    Ok(e) => e,
                    Err(_) => {
                        self.vanished += 1;
                        continue;
                    }
                };
                let (name, role, r) = match (
                    element.CurrentName(),
                    element.CurrentLocalizedControlType(),
                    element.CurrentBoundingRectangle(),
                ) {
                    (Ok(a), Ok(b), Ok(c)) => (a.to_string(), b.to_string(), c),
                    _ => {
                        self.vanished += 1;
                        continue;
                    }
                };
                let (w, h) = ((r.right - r.left) as i64, (r.bottom - r.top) as i64);
                if !name.is_empty() && w > 0 && h > 0 {
                    out.push(Widget::new(&name, &role, (r.left as i64, r.top as i64, w, h), &window));
                }
            }
            out
        }
    }

    /// What UI Automation says is at a point on the screen — the read half of the question a
    /// click asks.
    ///
    /// **This is how a click resolves and how it fails.** The hands find an element by asking what
    /// is at the widget's centre; when the answer is something else, the click lands on whatever
    /// occupies that rectangle instead. Returning the name and the role rather than the element
    /// keeps that answerable without holding a live COM pointer across the measurement.
    pub fn at_point(&self, x: i64, y: i64) -> Option<(String, String)> {
        unsafe {
            let at = self.automation.ElementFromPoint(POINT { x: x as i32, y: y as i32 }).ok()?;
            Some((at.CurrentName().ok()?.to_string(),
                  at.CurrentLocalizedControlType().ok()?.to_string()))
        }
    }

    /// The window in front. **Not the same as `attending()` in the Python**, which skips this
    /// process's own windows so the agent does not read itself — that needs the app to exist and
    /// belongs with it, not here.
    pub fn foreground_handle(&self) -> isize {
        unsafe { GetForegroundWindow().0 as isize }
    }
}

impl Eyes for WindowsEyes {
    fn foreground(&mut self) -> Vec<Widget> {
        let h = self.foreground_handle();
        self.read(h)
    }

    fn timed_foreground(&mut self) -> (Vec<Widget>, f64) {
        let t0 = std::time::Instant::now();
        let widgets = self.foreground();
        (widgets, t0.elapsed().as_secs_f64())
    }
}

/// How big the desktop says it is: `(virtual width, virtual height, primary width, primary height)`.
///
/// The two disagree on a multi-monitor desktop, and a coordinate that is fine on one is off-screen
/// on the other.
pub fn screen_metrics() -> (i32, i32, i32, i32) {
    unsafe {
        (GetSystemMetrics(SM_CXVIRTUALSCREEN), GetSystemMetrics(SM_CYVIRTUALSCREEN),
         GetSystemMetrics(SM_CXSCREEN), GetSystemMetrics(SM_CYSCREEN))
    }
}

/// Send the pointer to each point and read back where it actually went — **then put it back.**
///
/// Coordinates read from a DPI-aware process and written from a DPI-unaware one can be virtualised.
/// If the two disagree every click is offset by the scaling factor and nothing lands, which would
/// look exactly like the model choosing badly.
///
/// **It moves the real mouse pointer and presses nothing.** The original position is restored on
/// every exit path, because a measurement that leaves somebody's cursor in the corner of the screen
/// has changed the machine it was measuring. It lives here rather than in a harness for the reason
/// nothing above `argmind/env/` names a platform call.
pub fn cursor_round_trip(points: &[(i64, i64)]) -> Vec<((i64, i64), (i64, i64))> {
    let mut keep = POINT::default();
    unsafe {
        let _ = GetCursorPos(&mut keep);
    }
    let mut out = Vec::new();
    for (x, y) in points {
        unsafe {
            if SetCursorPos(*x as i32, *y as i32).is_err() {
                continue;
            }
            let mut got = POINT::default();
            if GetCursorPos(&mut got).is_ok() {
                out.push(((*x, *y), (got.x as i64, got.y as i64)));
            }
        }
    }
    unsafe {
        let _ = SetCursorPos(keep.x, keep.y);
    }
    out
}
