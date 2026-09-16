//! `SendInput` for hands — and **the standing rule made structural instead of remembered.**
//!
//! A click goes to a pixel. Nothing about it is addressed to a window: whatever is under that point
//! when the button goes down receives it, so a window arriving in front between choosing and acting
//! takes the click meant for something else. That is why this project's rule is *never click
//! anything the harness did not launch itself*, and why the rule lives in a type here rather than
//! in a paragraph somebody has to remember.
//!
//! [`Reach`] is the whole of it. `Reach::Anything` is what the shipped application uses, where the
//! person's confirm button is the thing that makes a click safe. `Reach::OnlyProcess` is what a
//! harness gets: **before every click the process that owns the window under the point is looked
//! up, and a click that would land anywhere else is refused rather than sent.** Refusals are
//! counted, because a harness that silently skipped half its clicks would report a clean number.
//!
//! **Control patterns are deliberately not ported, and the reason is measured on both sides.**
//! `measure/obstruction.py` asked whether they land better and found `3 of 3` with and `3 of 3`
//! without, so they bought nothing there — against about ninety lines, a second COM client, an
//! ancestor walk on every click and four ways to fail. `measure/pointer_theft.py` asked the other
//! question and found the thing they *do* buy: a pattern calls a method on the control and never
//! moves the cursor, where a synthetic click yanks the pointer out from under whoever is using the
//! machine. Neither result is settled at n=3, so the path is left out rather than carried forward
//! unexamined, and this note is where the decision is recorded.

use std::time::{Duration, Instant};

use argmind_core::contract::{Hands, Widget};
use windows::Win32::UI::Accessibility::IUIAutomation;

use crate::patterns::Operated;
use windows::Win32::Foundation::{HWND, POINT};
use windows::Win32::UI::Input::KeyboardAndMouse::{
    SendInput, INPUT, INPUT_0, INPUT_KEYBOARD, INPUT_MOUSE, KEYBDINPUT, KEYBD_EVENT_FLAGS,
    KEYEVENTF_KEYUP, KEYEVENTF_UNICODE, MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP,
    MOUSEEVENTF_MIDDLEDOWN, MOUSEEVENTF_MIDDLEUP, MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP,
    MOUSEINPUT, MOUSE_EVENT_FLAGS, VIRTUAL_KEY,
};
use windows::Win32::UI::WindowsAndMessaging::{GetWindowThreadProcessId, SetCursorPos,
                                              WindowFromPoint};

/// The named keys, **listed rather than guessed**.
///
/// Guessing a keystroke is how an agent sends Alt+F4. A name that is not here is refused with the
/// list, and the list is the whole vocabulary.
pub fn virtual_key(name: &str) -> Option<u16> {
    Some(match name.to_lowercase().as_str() {
        "enter" => 0x0D,
        "tab" => 0x09,
        "escape" | "esc" => 0x1B,
        "space" => 0x20,
        "backspace" => 0x08,
        "delete" => 0x2E,
        "home" => 0x24,
        "end" => 0x23,
        "pageup" => 0x21,
        "pagedown" => 0x22,
        "up" => 0x26,
        "down" => 0x28,
        "left" => 0x25,
        "right" => 0x27,
        _ => return None,
    })
}

/// Every key name this will accept, for the message a refusal prints.
pub const KEY_NAMES: [&str; 15] = ["backspace", "delete", "down", "end", "enter", "esc", "escape",
                                   "home", "left", "pagedown", "pageup", "right", "space", "tab",
                                   "up"];

pub fn button_flags(button: &str) -> Option<(MOUSE_EVENT_FLAGS, MOUSE_EVENT_FLAGS)> {
    Some(match button {
        "left" => (MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP),
        "right" => (MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP),
        "middle" => (MOUSEEVENTF_MIDDLEDOWN, MOUSEEVENTF_MIDDLEUP),
        _ => return None,
    })
}

/// **What these hands are allowed to touch.**
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Reach {
    /// Anything on the screen. The shipped application's, where a person confirms each action and
    /// the confirm button is what makes it safe.
    Anything,
    /// Only windows belonging to this process — what a harness gets, so that *never click anything
    /// the harness did not launch itself* is enforced by the type rather than by care.
    OnlyProcess(u32),
    /// One other process's windows **and our own** — for the harness that puts a window of its own
    /// over the target and then has to be allowed to act into it.
    ///
    /// **Still nothing the harness did not make itself.** The rule was never *one process*; it was
    /// *only what this harness is responsible for*, and a cover this program created is as much
    /// ours as an application it launched. Both ids are carried rather than read from the
    /// environment, so the decision stays pure and testable with no desktop.
    OursAnd { ours: u32, theirs: u32 },
}

/// Would a click at this point be allowed, given what owns the window there?
///
/// Pure, so the decision is testable without a desktop. `None` for the owner means nothing could be
/// resolved at that point, and **that is a refusal too**: a click into a point whose window cannot
/// be identified is exactly the case the rule exists for.
pub fn may_touch(reach: Reach, owner: Option<u32>) -> bool {
    match reach {
        Reach::Anything => true,
        Reach::OnlyProcess(pid) => owner == Some(pid),
        Reach::OursAnd { ours, theirs } => owner == Some(ours) || owner == Some(theirs),
    }
}

/// Which process owns the window under a screen point.
pub fn owner_at(x: i64, y: i64) -> Option<u32> {
    unsafe {
        let hwnd: HWND = WindowFromPoint(POINT { x: x as i32, y: y as i32 });
        if hwnd.0.is_null() {
            return None;
        }
        let mut pid: u32 = 0;
        GetWindowThreadProcessId(hwnd, Some(&mut pid));
        if pid == 0 { None } else { Some(pid) }
    }
}

//: **`Did` moved into the core on 2026-09-16**, when a second environment needed to say the same
//: things. It is re-exported here so every caller that knew it as `hands::Did` still does.
pub use argmind_core::contract::Did;

pub struct WindowsHands {
    pub reach: Reach,
    /// **Off unless a harness that is measuring them turns them on.** The table in
    /// [`crate::patterns`] has what the three harnesses found: nothing on task success, nothing on
    /// reaching past an obstruction, and **the pointer left where it was, 3 of 3** — which is one
    /// real benefit rather than none, and is the owner's call rather than this comment's.
    pub patterns: bool,
    /// Made once and only if a pattern is ever wanted.
    automation: Option<IUIAutomation>,
    pub by_pattern: u64,
    pub by_click: u64,
    /// Why the last attempt fell back to a click. **Empty is not the same as "no reason"** — it
    /// means no fallback happened.
    pub why: String,
    /// **Off by default.** Hands that are built and not armed do nothing, so the failure mode of
    /// forgetting a flag is a run that acts on nothing rather than a run that acts on everything.
    pub live: bool,
    pub max_per_second: f64,
    pub clicked: u64,
    pub refused: u64,
    pub elsewhere: u64,
    pub done: Vec<Did>,
    last: Option<Instant>,
}

impl WindowsHands {
    /// Hands that may touch anything — the shipped application's.
    pub fn unrestricted() -> Self {
        WindowsHands::with(Reach::Anything)
    }

    /// Hands that may touch only this process's windows — a harness's.
    pub fn only_within(pid: u32) -> Self {
        WindowsHands::with(Reach::OnlyProcess(pid))
    }

    /// Hands for a harness that has put a window of its own in the way.
    pub fn only_within_or_ours(pid: u32) -> Self {
        WindowsHands::with(Reach::OursAnd { ours: std::process::id(), theirs: pid })
    }

    fn with(reach: Reach) -> Self {
        WindowsHands { reach, patterns: false, automation: None, by_pattern: 0, by_click: 0,
                       why: String::new(), live: false, max_per_second: 4.0, clicked: 0,
                       refused: 0, elsewhere: 0, done: Vec::new(), last: None }
    }

    /// Arm them. Separate from construction because a flag that both builds and arms is a flag
    /// somebody sets while meaning only to build.
    pub fn arm(mut self) -> Self {
        self.live = true;
        self
    }

    /// Operate controls through UI Automation where one applies, falling back to a click.
    ///
    /// **For the harnesses that measure this path**, and for nothing else — see
    /// [`crate::patterns`] for why it is not on by default.
    pub fn with_patterns(mut self) -> Self {
        self.patterns = true;
        self
    }

    fn pace(&mut self) {
        let gap = Duration::from_secs_f64(1.0 / self.max_per_second);
        if let Some(last) = self.last {
            let since = last.elapsed();
            if since < gap {
                std::thread::sleep(gap - since);
            }
        }
        self.last = Some(Instant::now());
    }

    fn record(&mut self, did: Did) -> Did {
        match &did {
            Did::Sent { .. } | Did::Operated { .. } => {}
            Did::Elsewhere { .. } => self.elsewhere += 1,
            Did::Refused(_) => self.refused += 1,
        }
        self.done.push(did.clone());
        did
    }

    /// A click, with the reach check in front of it.
    pub fn click_button(&mut self, widget: &Widget, button: &str) -> Did {
        let Some((down, up)) = button_flags(button) else {
            return self.record(Did::Refused(format!("unknown button {button:?}")));
        };
        let (x, y) = widget.centre();
        //: **Checked immediately before acting, not when the widget was chosen.** A listed widget
        //: is gone 13 times in 14 over the second a call takes, and a window that arrived in front
        //: in that second is exactly what this refuses.
        let owner = owner_at(x, y);
        if !may_touch(self.reach, owner) {
            return self.record(Did::Elsewhere { owner });
        }
        //: **The pattern path is tried first and is gated by the same guard.** It resolves by
        //: point too — `ElementFromPoint` then upwards — so a foreign window over the target hands
        //: it the foreign element, and operating that is exactly what the guard exists to stop.
        //: One guard, both paths, checked before either.
        self.why = String::new();
        if self.patterns && button == "left" {
            if self.automation.is_none() {
                self.automation = crate::open_automation().ok();
            }
            if let Some(automation) = &self.automation {
                match crate::patterns::operate(automation, x, y, &widget.name, &widget.role,
                                               self.live) {
                    Operated::By(how) | Operated::WouldUse(how) => {
                        self.by_pattern += 1;
                        return self.record(Did::Operated { how, live: self.live });
                    }
                    Operated::Click(why) => self.why = why,
                }
            } else {
                self.why = "UI Automation would not open".into();
            }
        }
        self.clicked += 1;
        self.by_click += 1;
        if !self.live {
            return self.record(Did::Sent { live: false });
        }
        self.pace();
        unsafe {
            let _ = SetCursorPos(x as i32, y as i32);
        }
        send(&[mouse(down), mouse(up)]);
        self.record(Did::Sent { live: true })
    }

    pub fn press_key(&mut self, key: &str) -> Did {
        let Some(vk) = virtual_key(key) else {
            return self.record(Did::Refused(format!(
                "unknown key {key:?}. Named keys are listed explicitly rather than guessed, \
                 because guessing a keystroke is how an agent sends Alt+F4: {}",
                KEY_NAMES.join(", "))));
        };
        if !self.live {
            return self.record(Did::Sent { live: false });
        }
        self.pace();
        send(&[key_event(vk, 0, KEYBD_EVENT_FLAGS(0)),
               key_event(vk, 0, KEYEVENTF_KEYUP)]);
        self.record(Did::Sent { live: true })
    }

    pub fn type_string(&mut self, text: &str) -> Did {
        if !self.live {
            return self.record(Did::Sent { live: false });
        }
        self.pace();
        let mut events = Vec::new();
        //: **UTF-16 code units, not characters.** A character outside the basic plane is two
        //: units and `KEYEVENTF_UNICODE` takes one at a time; sending the code point would send
        //: something else entirely.
        for unit in typing_units(text) {
            events.push(key_event(0, unit, KEYEVENTF_UNICODE));
            events.push(key_event(0, unit, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP));
        }
        send(&events);
        self.record(Did::Sent { live: true })
    }
}

/// The code units one string types, in the order they go out.
///
/// **UTF-16 code units, not characters.** `KEYEVENTF_UNICODE` takes one unit at a time, and a
/// character outside the basic plane is two of them — sending the code point would send something
/// else entirely. Extracted so it can be checked without anything being sent: the code that
/// actually reaches the operating system is untestable by construction, so as little as possible
/// lives there.
pub fn typing_units(text: &str) -> Vec<u16> {
    text.encode_utf16().collect()
}

/// **A partial delivery is a failure and says so.**
///
/// `SendInput` returning fewer events than it was given means something blocked the stream part
/// way through, and half a click is a button held down. Pure, for the same reason as
/// [`typing_units`].
pub fn check_delivery(sent: u32, wanted: usize) -> Result<(), String> {
    if sent as usize == wanted {
        return Ok(());
    }
    Err(format!("SendInput delivered {sent} of {wanted} events; a partial click leaves a button                  down"))
}

fn mouse(flags: MOUSE_EVENT_FLAGS) -> INPUT {
    INPUT {
        r#type: INPUT_MOUSE,
        Anonymous: INPUT_0 {
            mi: MOUSEINPUT { dx: 0, dy: 0, mouseData: 0, dwFlags: flags, time: 0,
                             dwExtraInfo: 0 },
        },
    }
}

fn key_event(vk: u16, scan: u16, flags: KEYBD_EVENT_FLAGS) -> INPUT {
    INPUT {
        r#type: INPUT_KEYBOARD,
        Anonymous: INPUT_0 {
            ki: KEYBDINPUT { wVk: VIRTUAL_KEY(vk), wScan: scan, dwFlags: flags, time: 0,
                             dwExtraInfo: 0 },
        },
    }
}

fn send(events: &[INPUT]) {
    if events.is_empty() {
        return;
    }
    let sent = unsafe { SendInput(events, std::mem::size_of::<INPUT>() as i32) };
    if let Err(why) = check_delivery(sent, events.len()) {
        panic!("{why}");
    }
}

impl Hands for WindowsHands {
    fn live(&self) -> bool {
        self.live
    }
    fn click(&mut self, widget: &Widget) {
        self.click_button(widget, "left");
    }
    fn type_text(&mut self, text: &str) {
        self.type_string(text);
    }
    fn press(&mut self, key: &str) {
        self.press_key(key);
    }
}
