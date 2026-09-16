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

pub mod ask;
pub mod attending;
pub mod camera;
pub mod cover;
pub mod hands;
pub mod launch;
pub mod microphone;
pub mod patterns;
pub mod speaker;
pub mod supervise;
pub mod tray;
pub use hands::{Reach, WindowsHands};
pub use attending::{Attending, Watcher};
pub use cover::Cover;
pub use microphone::{Clip, Microphone};
pub use patterns::{operate, Operated};
pub use launch::{bring_to_front, foreground_handle_now, foreground_is, whats_in_front, window_title, is_minimised, pid_of, to_front, windows_of, Launched};
pub use speaker::SapiSpeaker;
pub use supervise::Supervised;

use argmind_core::contract::{Eyes, Widget};
use windows::Win32::Foundation::{HWND, POINT};
use windows::Win32::System::Com::{
    CoCreateInstance, CoInitializeEx, CLSCTX_INPROC_SERVER, COINIT_MULTITHREADED,
};
use windows::Win32::UI::Accessibility::{
    CUIAutomation, IUIAutomation, IUIAutomationElement, TreeScope_Descendants,
    UIA_IsOffscreenPropertyId, UIA_PATTERN_ID,
};
use windows::Win32::UI::WindowsAndMessaging::{GetCursorPos, GetForegroundWindow,
    GetSystemMetrics, SetCursorPos, SM_CXSCREEN, SM_CXVIRTUALSCREEN, SM_CYSCREEN,
    SM_CYVIRTUALSCREEN};

/// One step of an ancestor walk.
#[derive(Debug, Clone)]
pub struct Hop {
    pub name: String,
    pub role: String,
    /// It is the thing that was being looked for.
    pub matched: bool,
    /// Only filled in on the hop that matched — see [`WindowsEyes::ancestry_at`].
    pub patterns: Vec<&'static str>,
    /// The element could not be read. The tree is live and a walk can run into a control that
    /// closed; **that is a different answer from "no parent"** and is kept apart.
    pub died: bool,
}

/// Every pattern UI Automation defines, in id order from `UIA_InvokePatternId` (10000).
///
/// **All of them, so the answer is "here is what this control offers"** rather than "none of the
/// four I happened to try". The four the hands were built around were chosen from a guess about
/// which controls expose what, and the combo box that motivated the whole question offered none of
/// them — a guess is not a survey.
pub const PATTERN_NAMES: [&str; 22] = [
    "Invoke", "Selection", "Value", "RangeValue", "Scroll", "ExpandCollapse", "Grid", "GridItem",
    "MultipleView", "Window", "SelectionItem", "Dock", "Table", "TableItem", "Text", "Toggle",
    "Transform", "ScrollItem", "LegacyIAccessible", "ItemContainer", "VirtualizedItem",
    "SynchronizedInput",
];

fn supported(element: &IUIAutomationElement) -> Vec<&'static str> {
    let mut out = Vec::new();
    for (i, name) in PATTERN_NAMES.iter().enumerate() {
        //: An unsupported pattern answers with a null interface, which windows-rs reports as an
        //: error — so "could not ask" and "does not have it" are one answer here, as they were in
        //: the Python's `except: continue`.
        if unsafe { element.GetCurrentPattern(UIA_PATTERN_ID(10000 + i as i32)) }.is_ok() {
            out.push(*name);
        }
    }
    out
}

/// One UI Automation client. **Shared shape, one definition** — the eyes and the pattern path both
/// need one, and two ways of opening it is two ways of getting the apartment wrong.
pub fn open_automation() -> windows::core::Result<IUIAutomation> {
    unsafe {
        //: The apartment the calling thread already has is taken as given — `RPC_E_CHANGED_MODE`
        //: means somebody got here first, which is not an error for a reader.
        let _ = CoInitializeEx(None, COINIT_MULTITHREADED);
        CoCreateInstance(&CUIAutomation, None, CLSCTX_INPROC_SERVER)
    }
}

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

    /// The chain of elements above a point, up to `hops`, stopping when one matches `want`.
    ///
    /// **Why walk up at all.** The centre of a combo box sits on the edit control inside it, so
    /// asking what is at that point answers with the child and the thing that expands is its
    /// parent. Walking up looking for the name *and the role* is cheap and exact — on the name
    /// alone a tab item named `Claude Code` inside a group of the same name resolved to the group,
    /// and operating the group is not what was asked for.
    ///
    /// Patterns are read **only on the hop that matched**, because twenty-two of them at eight hops
    /// on every widget is a COM round trip per question and the answer is only wanted once.
    pub fn ancestry_at(&self, x: i64, y: i64, hops: usize, want: Option<(&str, &str)>) -> Vec<Hop> {
        let mut out = Vec::new();
        unsafe {
            let Ok(walker) = self.automation.RawViewWalker() else { return out };
            let Ok(mut element) = self.automation.ElementFromPoint(POINT { x: x as i32,
                                                                           y: y as i32 })
            else {
                return out;
            };
            for _ in 0..hops {
                let (Ok(name), Ok(role)) = (element.CurrentName(),
                                            element.CurrentLocalizedControlType())
                else {
                    out.push(Hop { name: String::new(), role: String::new(), matched: false,
                                   patterns: Vec::new(), died: true });
                    break;
                };
                let (name, role) = (name.to_string(), role.to_string());
                let matched = want.is_some_and(|(n, r)| name == n && role == r);
                let patterns = if matched { supported(&element) } else { Vec::new() };
                out.push(Hop { name, role, matched, patterns, died: false });
                if matched {
                    break;
                }
                match walker.GetParentElement(&element) {
                    Ok(parent) => element = parent,
                    Err(_) => break,
                }
            }
        }
        out
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

/// Does the cursor land where it is sent? **Asked before clicking anything.**
///
/// Coordinates read from a DPI-aware process and written from a DPI-unaware one can be virtualised,
/// and then every click is offset by the scaling factor and nothing lands. Two points near the
/// middle of the primary display, so the probe itself cannot fall off the desktop.
pub fn pointing_is_exact() -> bool {
    matches!(pointing(), Pointing::Exact)
}

/// Why a click would not land, told apart.
///
/// **These were one answer and they are two problems.** A scaled write path is a display setting.
/// A desktop that will not take input at all is something else entirely, and reporting it as the
/// first sends somebody to check their DPI for a reason that has nothing to do with DPI.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Pointing {
    /// The cursor goes exactly where it is sent.
    Exact,
    /// It moves, and lands somewhere else — a virtualised coordinate.
    Scaled,
    /// **`SetCursorPos` failed.** Reading the cursor still works; writing it does not.
    ///
    /// **What causes it is not settled, and the first guess written here was wrong.** It said
    /// *locked workstation*; when it was actually measured, `OpenInputDesktop` succeeded and named
    /// the input desktop `Default` — not `Winlogon`, not a screen-saver desktop — while
    /// `SetCursorPos` returned `ERROR_INVALID_HANDLE` for thirty seconds together. A locked session
    /// does not look like that. What is established is the symptom and that it comes and goes:
    /// during one afternoon it was `Exact` immediately after the person at the machine used it and
    /// `Refused` a minute later, repeatedly, and the harness runs themselves disturbed the session
    /// enough to be a candidate cause rather than an innocent observer.
    ///
    /// So the message says what is true — the desktop will not take input — and stops there. A
    /// remedy invented for a cause nobody has is how somebody spends an afternoon on the wrong
    /// thing.
    Refused,
}

pub fn pointing() -> Pointing {
    let (_, _, pw, ph) = screen_metrics();
    let probes = [((pw / 2) as i64, (ph / 2) as i64), ((pw / 2) as i64 + 7, (ph / 2) as i64 + 11)];
    let got = cursor_round_trip(&probes);
    if got.len() != probes.len() {
        //: `cursor_round_trip` skips a point it could not set, so a short list *is* the refusal.
        return Pointing::Refused;
    }
    if got.iter().all(|(sent, landed)| sent == landed) {
        Pointing::Exact
    } else {
        Pointing::Scaled
    }
}

/// Where the pointer is, **without moving it.**
///
/// Separate from [`cursor_round_trip`] on purpose: an instrument that answers *"did the pointer
/// move?"* must not be the thing that moved it.
pub fn cursor_now() -> (i64, i64) {
    let mut at = POINT::default();
    unsafe {
        let _ = GetCursorPos(&mut at);
    }
    (at.x as i64, at.y as i64)
}

/// Put the pointer somewhere and **leave it there.** `false` if it did not arrive.
///
/// The third of these three, and each one exists because the other two are wrong for the job.
/// [`cursor_now`] reads and does not move; [`cursor_round_trip`] moves, reads and *puts it back*,
/// which makes it a probe rather than a mover. **`pointer_theft` parked the cursor with the probe**
/// and so never parked it at all: every arm read the position the previous arm's click had left it
/// in, and both arms reported *unmoved* for the same wrong reason — which is the exact mistake the
/// note above `where_is` in that harness describes, fixed there and not here.
pub fn put_cursor(x: i64, y: i64) -> bool {
    unsafe {
        if SetCursorPos(x as i32, y as i32).is_err() {
            return false;
        }
    }
    //: **Verified rather than assumed.** A scaled write path virtualises the coordinate, and a park
    //: that silently landed somewhere else would be the same lie in a new place.
    cursor_now() == (x, y)
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
