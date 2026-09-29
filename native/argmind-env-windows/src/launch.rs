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
use windows::Win32::UI::WindowsAndMessaging::{EnumWindows, GetClassNameW, GetForegroundWindow,
    GetWindowTextLengthW, GetWindowTextW, GetWindowThreadProcessId, IsIconic, IsWindowVisible,
    SetForegroundWindow};

use crate::hands::WindowsHands;

struct Hunt {
    pid: u32,
    found: isize,
    /// When set, any visible window counts — not only a titled one.
    all: Option<Vec<isize>>,
}

unsafe extern "system" fn visit(handle: HWND, lparam: LPARAM) -> BOOL {
    let hunt = &mut *(lparam.0 as *mut Hunt);
    let mut pid: u32 = 0;
    GetWindowThreadProcessId(handle, Some(&mut pid));
    //: **Visible and titled.** A process opens several windows before the one a person would call
    //: its window, and the invisible zero-width ones come first.
    if pid != hunt.pid || !IsWindowVisible(handle).as_bool() {
        return TRUE;
    }
    if let Some(all) = hunt.all.as_mut() {
        all.push(handle.0 as isize);
        return TRUE;
    }
    if GetWindowTextLengthW(handle) > 0 {
        hunt.found = handle.0 as isize;
        return BOOL(0);
    }
    TRUE
}

/// Every visible titled window belonging to a process — or rather, the first one found.
pub fn window_of(pid: u32) -> Option<isize> {
    let mut hunt = Hunt { pid, found: 0, all: None };
    unsafe {
        let _ = EnumWindows(Some(visit), LPARAM(&mut hunt as *mut Hunt as isize));
    }
    if hunt.found == 0 { None } else { Some(hunt.found) }
}

/// **Every** visible window a process owns, titled or not.
///
/// Not the same question as [`window_of`], and the difference is a measured one: a combo box's
/// drop-down is **its own top-level window**, so reading only the main one reports "nothing
/// happened" for a list that did open — and cannot tell that apart from a click that landed on the
/// combo's text area instead of its arrow, which is a real failure.
pub fn windows_of(pid: u32) -> Vec<isize> {
    let mut hunt = Hunt { pid, found: 0, all: Some(Vec::new()) };
    unsafe {
        let _ = EnumWindows(Some(visit), LPARAM(&mut hunt as *mut Hunt as isize));
    }
    hunt.all.unwrap_or_default()
}

/// Which process owns a window.
pub fn pid_of(handle: isize) -> Option<u32> {
    unsafe {
        let mut pid: u32 = 0;
        GetWindowThreadProcessId(HWND(handle as _), Some(&mut pid));
        if pid == 0 { None } else { Some(pid) }
    }
}

/// The window in front, without opening UI Automation.
///
/// [`crate::WindowsEyes::foreground_handle`] answers the same question and needs an automation
/// client to exist first. A thread that only wants to know which window a person is in should not
/// have to hold one — and COM objects do not cross threads.
///
/// # Safety
/// None, in fact: it is `unsafe` only because the Win32 call is, and it reads a handle.
pub unsafe fn foreground_handle_now() -> isize {
    GetForegroundWindow().0 as isize
}

/// Is the window in front owned by this process? **Checked before every click**, because a click
/// goes to a pixel and anything that came to the front takes it.
pub fn foreground_is(pid: u32) -> bool {
    unsafe { pid_of(GetForegroundWindow().0 as isize) == Some(pid) }
}

/// Ask Windows whether a window is minimised.
///
/// **Ask, do not infer from the widget list.** The first version of this expectation read the
/// window afterwards and expected nothing, on the assumption that a minimised window exposes no
/// elements. It reported a failure for a click that had very likely worked: a minimised window
/// still answers UI Automation with some of its tree, so the expectation was wrong about Windows
/// rather than about the agent.
pub fn is_minimised(handle: isize) -> bool {
    unsafe { IsIconic(HWND(handle as _)).as_bool() }
}

/// Bring a window to the front. Best effort — Windows may refuse a process that is not in front.
pub fn to_front(handle: isize) -> bool {
    unsafe { SetForegroundWindow(HWND(handle as _)).as_bool() }
}

/// A window's title, for saying what won instead of what was asked for.
/// A window's **class**, which does not change with the language Windows is installed in —
/// unlike its title, which reads *Taskbar* here and something else on a Korean install.
pub fn window_class(handle: isize) -> String {
    unsafe {
        let mut buf = [0u16; 256];
        let n = GetClassNameW(HWND(handle as _), &mut buf);
        String::from_utf16_lossy(&buf[..n.max(0) as usize])
    }
}

/// **Shell surfaces a person passes through on the way to a window, and never works in.**
///
/// The owner's transcript of 2026-09-17 already had `the window changed to 'Taskbar'` in it, and
/// on 2026-09-29 it was the one thing argmind spoke about: clicking the taskbar to switch apps puts
/// the taskbar in front for a moment, the watch counted that as the person moving to a window
/// called Taskbar, and the remark was about the taskbar rather than about where they went.
///
/// **By class, and only the unambiguous ones.** The primary and secondary taskbars, their overflow,
/// and the desktop. Start and search are left out on purpose: they are `Windows.UI.Core.CoreWindow`,
/// which a packaged app can also be, and a list that silenced a real application would be worse than
/// one remark about the Start menu.
pub const PASSED_THROUGH: [&str; 6] = ["Shell_TrayWnd", "Shell_SecondaryTrayWnd",
                                       "NotifyIconOverflowWindow",
                                       "TopLevelWindowForOverflowXamlIsland",
                                       "Progman", "WorkerW"];

pub fn passed_through(class: &str) -> bool {
    PASSED_THROUGH.contains(&class)
}

#[cfg(test)]
mod passing_through {
    use super::passed_through;

    #[test]
    fn the_taskbar_and_the_desktop_are_passed_through() {
        //: `Shell_TrayWnd` is the class confirmed on this machine on 2026-09-29, with an empty title
        //: — the *Taskbar* in the transcript was UI Automation's name for it, which is why this goes
        //: by class and not by what the window is called.
        for class in ["Shell_TrayWnd", "Shell_SecondaryTrayWnd", "Progman", "WorkerW"] {
            assert!(passed_through(class), "{class}");
        }
    }

    #[test]
    fn a_real_application_is_never_passed_through() {
        //: **The half that matters more.** A list that swallowed a working window would silence the
        //: watch for it entirely, which is worse than one remark about the taskbar.
        for class in ["Chrome_WidgetWin_1", "Notepad", "ApplicationFrameWindow",
                      "Windows.UI.Core.CoreWindow", "CabinetWClass", ""] {
            assert!(!passed_through(class), "{class:?} would be silenced");
        }
    }
}

pub fn window_title(handle: isize) -> String {
    unsafe {
        let mut buf = [0u16; 256];
        let n = GetWindowTextW(HWND(handle as _), &mut buf);
        String::from_utf16_lossy(&buf[..n as usize])
    }
}

/// What is in front right now: its handle, its title and the process that owns it.
///
/// **A refusal that does not say what won instead is half an answer.** Two harnesses refuse when
/// they cannot get the front, and "could not get the front" alone leaves a reader unable to tell a
/// focus-stealing rule from a window that failed to open.
pub fn whats_in_front() -> (isize, String, Option<u32>) {
    let handle = unsafe { GetForegroundWindow().0 as isize };
    (handle, window_title(handle), pid_of(handle))
}

/// Ask for the front and **wait for it to actually arrive**, up to `within`.
///
/// One `SetForegroundWindow` is not enough and the failure is not permanent: Windows hands the
/// foreground privilege around, a freshly launched application takes a moment before it can accept
/// it, and whatever had focus may still be letting go. A single try reported *"could not get the
/// front"* on a machine where the same call succeeded seconds earlier in another harness.
///
/// Returns whether it is genuinely in front — **the answer is the check, not the request**, because
/// `SetForegroundWindow` returning true does not mean the window is in front.
pub fn bring_to_front(handle: isize, pid: u32, within: Duration) -> bool {
    let end = Instant::now() + within;
    loop {
        to_front(handle);
        std::thread::sleep(Duration::from_millis(200));
        if foreground_is(pid) {
            return true;
        }
        if Instant::now() >= end {
            return false;
        }
    }
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
