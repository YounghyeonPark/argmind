//! Starting a window so there is something safe to click.
//!
//! **This is what makes [`crate::hands`] testable at all.** The rule is *never click anything the
//! harness did not launch itself*, and a rule like that is only usable if launching something is
//! easy. A [`Launched`] hands out [`WindowsHands`] already bound to its own process, so the safe
//! thing is also the short thing to write.
//!
//! It kills the process on drop. A measurement that leaves an application open has changed the
//! machine it was measuring, and one that leaves it open **after failing** is the case that
//! matters — so the cleanup is in `Drop` rather than at the end of a happy path.

use std::process::{Child, Command};
use std::time::{Duration, Instant};

use windows::Win32::Foundation::{BOOL, HWND, LPARAM, TRUE};
use windows::Win32::UI::WindowsAndMessaging::{EnumWindows, GetWindowTextLengthW,
                                              GetWindowThreadProcessId, IsWindowVisible};

use crate::hands::WindowsHands;

struct Hunt {
    pid: u32,
    found: isize,
}

unsafe extern "system" fn visit(handle: HWND, lparam: LPARAM) -> BOOL {
    let hunt = &mut *(lparam.0 as *mut Hunt);
    let mut pid: u32 = 0;
    GetWindowThreadProcessId(handle, Some(&mut pid));
    //: **Visible and titled.** A process opens several windows before the one a person would call
    //: its window, and the invisible zero-width ones come first.
    if pid == hunt.pid && IsWindowVisible(handle).as_bool() && GetWindowTextLengthW(handle) > 0 {
        hunt.found = handle.0 as isize;
        return BOOL(0);
    }
    TRUE
}

/// Every visible titled window belonging to a process — or rather, the first one found.
pub fn window_of(pid: u32) -> Option<isize> {
    let mut hunt = Hunt { pid, found: 0 };
    unsafe {
        let _ = EnumWindows(Some(visit), LPARAM(&mut hunt as *mut Hunt as isize));
    }
    if hunt.found == 0 { None } else { Some(hunt.found) }
}

/// An application this process started, and the only thing its hands will touch.
pub struct Launched {
    child: Child,
    pub pid: u32,
    pub exe: String,
}

impl Launched {
    pub fn start(exe: &str) -> std::io::Result<Self> {
        let child = Command::new(exe).spawn()?;
        Ok(Launched { pid: child.id(), child, exe: exe.to_string() })
    }

    /// Wait for its window to exist. `None` means it never showed one.
    pub fn window(&self, within: Duration) -> Option<isize> {
        let end = Instant::now() + within;
        while Instant::now() < end {
            if let Some(h) = window_of(self.pid) {
                return Some(h);
            }
            std::thread::sleep(Duration::from_millis(120));
        }
        None
    }

    /// Hands that can touch this application and **nothing else on the screen**. Dry until armed.
    pub fn hands(&self) -> WindowsHands {
        WindowsHands::only_within(self.pid)
    }

    pub fn close(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Drop for Launched {
    fn drop(&mut self) {
        self.close();
    }
}
