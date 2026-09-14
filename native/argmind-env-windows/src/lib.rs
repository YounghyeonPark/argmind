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
//! **Clicking is deliberately not here yet.** Reading is safe to exercise against a real desktop;
//! `SendInput` is not, and this project has a standing rule against clicking anything a harness did
//! not launch itself.

use argmind_core::contract::{Eyes, Widget};
use windows::Win32::Foundation::HWND;
use windows::Win32::System::Com::{
    CoCreateInstance, CoInitializeEx, CLSCTX_INPROC_SERVER, COINIT_MULTITHREADED,
};
use windows::Win32::UI::Accessibility::{
    CUIAutomation, IUIAutomation, IUIAutomationElement, TreeScope_Descendants,
    UIA_IsOffscreenPropertyId,
};
use windows::Win32::UI::WindowsAndMessaging::GetForegroundWindow;

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
