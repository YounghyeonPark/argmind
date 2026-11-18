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
//! **Control patterns are on for clicking, since 2026-09-22.** They were measured in 2026-09-16 and
//! left off, and the reason written down then was *the shipped click is confirmed by a person
//! pressing a button, so the cursor was already about to move to something they chose.* 확인 없이
//! 실행 landed the day before and nobody is pressing. What the measurement says, n=3 throughout:
//! task success **3 of 3 either way**, reaching past a cover **missed either way**, the pointer
//! **unmoved against dragged to the widget, 3 of 3**. One axis, and it is the one that matters on a
//! machine somebody is working at. `platform::hands` is where that is asked for; see
//! [`crate::patterns`].
//!
//! **And what no pattern takes is addressed to the window, since 2026-11-16** —
//! [`WindowsHands::with_messages`], see [`crate::addressed`]. The owner did not want argmind to
//! stop when they touched the machine, nor to lock them out of it; so a key, a click no pattern
//! takes and typing no value pattern takes are posted to the program's own window rather than sent
//! down the person's input, and `SendInput` is left for a window that refuses a message. **Where a
//! key goes** is the window argmind just acted on while that is fresh ([`AIM_LASTS`]), and
//! otherwise the window the person was working in — never argmind's own. **That last is
//! reasoned, not observed**: a pattern click does not bring its window to the front, so an Enter
//! after it went by `SendInput` to whatever was in front, and when the person was typing to argmind
//! that was argmind.
//!
//! ## **Typing still goes through the keyboard, and UI Automation has no answer for it**
//!
//! [`WindowsHands::type_string`] sends `KEYEVENTF_UNICODE` events, so the text lands in whatever has
//! focus — the same collision a click has with the pointer, one device over. The obvious fix is the
//! one Linux already uses: `argmind-env-linux` types through `EditableText.InsertText` and touches
//! no key at all.
//!
//! **They are not the same operation.** AT-SPI inserts *at an offset*; UI Automation's
//! `ValuePattern::SetValue` replaces the whole value, and there is no insert anywhere in UIA —
//! `TextPattern` reads and does not edit. So keystroke-free typing on Windows means *replace this
//! field*, which would delete whatever a person had already put in it. The asymmetry is the same
//! shape as the one already recorded on the other side, where `InsertText`'s length is in bytes and
//! not characters: two accessibility APIs that look alike and are not.
//!
//! **It aims at the widget argmind just clicked, not at the focus.** The first version wrote to
//! `GetFocusedElement` and a run showed that answered nothing — focus is where argmind's own click
//! put it, so writing there is writing wherever anybody's last click went. A `type` effect names no
//! target, but the click before it does, so [`AIM_LASTS`] holds one for one type and no longer.
//!
//! The shape that could work is narrow and is **built, off by default, and unmeasured**:
//! [`crate::patterns::set_value`] reads `CurrentValue`, writes only into an empty control, and
//! falls back to keys for everything else — read-only, no pattern, or a box that already holds
//! something. `WindowsHands::with_typing_patterns` is the switch and it is a **second** switch:
//! turning clicking patterns on must not carry this with it, because one of the two is measured.
//!
//! **Built rather than deferred, which is this crate's own precedent.** `patterns.rs` says it:
//! *this path is off by default and exists so it can be measured... the earlier decision to leave
//! it out would have left two experiments with one arm each, which is the shape this project calls
//! a measurement that cannot answer its own question.* `argmind-measure --bin typing` is the
//! harness, with an arm on each side, and it takes the foreground — so it cannot be run from the
//! session that would lose it, which is why there is no number here yet.

use std::time::{Duration, Instant};

use argmind_core::contract::{Hands, Widget};
use windows::Win32::UI::Accessibility::IUIAutomation;

use crate::patterns::{Operated, Typed};
use windows::Win32::Foundation::{HWND, POINT};
use windows::Win32::UI::Input::KeyboardAndMouse::{
    SendInput, INPUT, INPUT_0, INPUT_KEYBOARD, INPUT_MOUSE, KEYBDINPUT, KEYBD_EVENT_FLAGS,
    KEYEVENTF_KEYUP, KEYEVENTF_UNICODE, MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP,
    MOUSEEVENTF_MIDDLEDOWN, MOUSEEVENTF_MIDDLEUP, MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP,
    MOUSEINPUT, MOUSE_EVENT_FLAGS, VIRTUAL_KEY,
};
use windows::Win32::UI::WindowsAndMessaging::{GetForegroundWindow, GetWindowThreadProcessId,
                                              SetCursorPos, WindowFromPoint};

/// How long a click stays an aim for the typing that follows it.
///
/// **A second, and the number it has to beat is the model's.** Effects inside one `perform` are
/// 250 ms apart at `max_per_second` 4, so a click and its type are always well inside this; a round
/// trip to the model is 0.61 s median, which is what an aim must not survive. Past this the
/// keyboard takes over, which is what happened before any of this existed.
pub const AIM_LASTS: Duration = Duration::from_secs(1);

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
    /// **A second flag, and not the same question.** Clicking through a pattern does what a click
    /// does and leaves the pointer alone; typing through one *replaces the field* where keys insert
    /// at the caret. Turning the first on must not quietly turn the second on, so they are switched
    /// and counted separately — and this one is off, because it has not been measured.
    pub typing_patterns: bool,
    /// How many times text went in through `ValuePattern` rather than through the keyboard.
    pub by_value: u32,
    /// **A third switch**: what no pattern takes goes to the window as a message rather than down
    /// the person's input — [`crate::addressed`]. Off unless asked for, like the other two.
    pub messages: bool,
    /// How many acts went as a message.
    pub by_message: u64,
    /// **May fall back to `SendInput` when a window refuses a message.** On unless a test turns it
    /// off, which is what lets a test drive these hands with a guarantee that nothing it does can
    /// reach the input of whoever is running it.
    pub may_send_input: bool,
    /// Where the last click landed, and when. **Taken once and never reused.**
    ///
    /// A `type` effect names no target — the reply shape is `{"type": "<text>"}` and the model
    /// never says where — so the only thing that knows which box is meant is the click immediately
    /// before it. `MOST_AT_ONCE = 3` is *click, type, enter* and the three arrive in one `perform`
    /// a quarter of a second apart, which is what makes remembering one safe where remembering a
    /// rectangle across a model call would not be.
    aimed: Option<((i64, i64), Instant)>,
    /// **The window argmind's last act reached, and when** — where a key or untargeted typing goes
    /// while it is fresh. Unlike the aim it is not taken: a click, its typing and its Enter are one
    /// program's, a quarter of a second apart.
    reached: Option<(isize, Instant)>,
    /// Which window the person was working in, for a key with nothing just before it.
    attending: Option<crate::Watcher>,
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
        WindowsHands { reach, patterns: false, typing_patterns: false, by_value: 0, aimed: None,
                       messages: false, by_message: 0, may_send_input: true, reached: None,
                       attending: None,
                       automation: None, by_pattern: 0, by_click: 0,
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
    /// Type through `ValuePattern` where the control is empty and takes one. **Off by default and
    /// unmeasured** — see [`crate::patterns::set_value`] for why it is narrow, and
    /// `argmind-measure --bin typing` for the harness with an arm on each side.
    pub fn with_typing_patterns(mut self) -> Self {
        self.typing_patterns = true;
        self
    }

    pub fn with_patterns(mut self) -> Self {
        self.patterns = true;
        self
    }

    /// Post what no pattern takes to the program's window — see [`crate::addressed`].
    pub fn with_messages(mut self) -> Self {
        self.messages = true;
        self
    }

    /// Know which window the person was working in — the eyes' own rule — for a key that follows
    /// nothing argmind did.
    pub fn reading(mut self, watcher: crate::Watcher) -> Self {
        self.attending = Some(watcher);
        self
    }

    /// **Refuse rather than fall back to `SendInput`.** For a test, so that driving these hands
    /// cannot reach the keyboard or pointer of whoever runs it, whatever a window does.
    pub fn never_send_input(mut self) -> Self {
        self.may_send_input = false;
        self
    }

    /// **Where a key, or typing nothing aimed, goes**: the window argmind just acted on while that
    /// is fresh, else the window the person was working in, else the one in front — and never
    /// argmind's own, which is where `SendInput` would send one whenever argmind was in front
    /// (reasoned from a pattern click not bringing its window forward; never observed).
    fn key_target(&self) -> Result<HWND, String> {
        use crate::addressed::{alive, focus_of, pid};
        if let Some((h, at)) = self.reached {
            let h = HWND(h as _);
            if at.elapsed() <= AIM_LASTS && alive(h) {
                return Ok(focus_of(h));
            }
        }
        let me = Some(std::process::id());
        let front = unsafe { GetForegroundWindow() };
        let attended = self.attending.as_ref().and_then(|w| w.attended(front.0 as isize))
            .map(|h| HWND(h as _));
        let target = attended.or_else(|| Some(front).filter(|f| !f.is_invalid()))
            .filter(|h| pid(*h) != me)
            .ok_or_else(|| "argmind 창 말고는 보낼 창이 없다 — 사람이 일하던 창을 모른다".to_string())?;
        Ok(focus_of(target))
    }

    /// Post one act; `send` is the posting itself, run only when armed.
    fn posted(&mut self, h: HWND, send: impl FnOnce(HWND) -> Result<(), String>) -> Result<Did, String> {
        let owner = crate::addressed::pid(h);
        if !may_touch(self.reach, owner) {
            return Ok(self.record(Did::Elsewhere { owner }));
        }
        self.reached = Some((h.0 as isize, Instant::now()));
        if !self.live {
            self.by_message += 1;
            return Ok(self.record(Did::Operated { how: "posted", live: false }));
        }
        self.pace();
        send(h)?;
        self.by_message += 1;
        Ok(self.record(Did::Operated { how: "posted", live: true }))
    }

    fn because(&mut self, why: String) {
        self.why = if self.why.is_empty() { why } else { format!("{}; {why}", self.why) };
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

    /// **Where the type after this click goes** — set by a click and by nothing else: both ways of
    /// clicking come through here, so there is one place that aims.
    fn aim(&mut self, x: i64, y: i64) {
        self.aimed = Some(((x, y), Instant::now()));
    }

    /// A click, with the reach check in front of it.
    pub fn click_button(&mut self, widget: &Widget, button: &str) -> Did {
        let Some((down, up)) = button_flags(button) else {
            return self.record(Did::Refused(format!("unknown button {button:?}")));
        };
        let (x, y) = widget.centre();
        //: **Remembered for the type that may follow**, and only for it — see `aimed`. Set before
        //: the guard rather than after, because a click that was refused still tells the next
        //: effect where argmind was aiming, and `set_value` runs the same guard again anyway.
        self.aim(x, y);
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
        //: **Which program this was**, whichever path presses it, so the Enter after it goes there.
        if self.messages {
            if let Some(h) = crate::addressed::window_at(x, y) {
                self.reached = Some((h.0 as isize, Instant::now()));
            }
        }
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
        //: **Then a message to the window under the point**: the same window a real click would
        //: reach, without the pointer going there.
        if self.messages {
            match crate::addressed::window_at(x, y) {
                Some(h) => match self.posted(h, |h| crate::addressed::click(h, x, y, button)) {
                    Ok(did) => return did,
                    Err(why) => self.because(why),
                },
                None => self.because("no window under the point".into()),
            }
        }
        self.pointer_click(x, y, down, up)
    }

    /// **The person's pointer, as the last resort** — moved to the point and clicked — and only
    /// where these hands were allowed it.
    fn pointer_click(&mut self, x: i64, y: i64, down: MOUSE_EVENT_FLAGS, up: MOUSE_EVENT_FLAGS) -> Did {
        if !self.may_send_input {
            return self.record(Did::Refused(format!("only the person's pointer would reach it: {}", self.why)));
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

    /// **A click at a point inside a place the list cannot see into** — see
    /// `argmind_core::intent::Inside`. **No pattern**: a pattern operates the widget it finds, and
    /// the middle of a remote desktop or a character grid is not the thing inside it that was asked
    /// for. So the same reach guard, then a message to the window under the point, then the
    /// pointer only where these hands may use it.
    pub fn click_point(&mut self, x: i64, y: i64) -> Did {
        let Some((down, up)) = button_flags("left") else {
            return self.record(Did::Refused("no left button".into()));
        };
        self.aim(x, y);
        let owner = owner_at(x, y);
        if !may_touch(self.reach, owner) {
            return self.record(Did::Elsewhere { owner });
        }
        self.why = String::new();
        if self.messages {
            match crate::addressed::window_at(x, y) {
                Some(h) => match self.posted(h, |h| crate::addressed::click(h, x, y, "left")) {
                    Ok(did) => return did,
                    Err(why) => self.because(why),
                },
                None => self.because("no window under the point".into()),
            }
        } else {
            self.because("messages are off".into());
        }
        self.pointer_click(x, y, down, up)
    }

    pub fn press_key(&mut self, key: &str) -> Did {
        let Some(vk) = virtual_key(key) else {
            return self.record(Did::Refused(format!(
                "unknown key {key:?}. Named keys are listed explicitly rather than guessed, \
                 because guessing a keystroke is how an agent sends Alt+F4: {}",
                KEY_NAMES.join(", "))));
        };
        self.why = String::new();
        if self.messages {
            //: **No window to send it to is a refusal, not a fall back**: with only argmind in
            //: front, `SendInput` would press it in argmind's own window.
            let h = match self.key_target() {
                Ok(h) => h,
                Err(why) => return self.record(Did::Refused(why)),
            };
            match self.posted(h, |h| crate::addressed::key(h, vk)) {
                Ok(did) => return did,
                Err(why) => self.because(why),
            }
        }
        if !self.may_send_input {
            return self.record(Did::Refused(format!("only the person's keyboard would reach it: {}", self.why)));
        }
        if !self.live {
            return self.record(Did::Sent { live: false });
        }
        self.pace();
        send(&[key_event(vk, 0, KEYBD_EVENT_FLAGS(0)),
               key_event(vk, 0, KEYEVENTF_KEYUP)]);
        self.record(Did::Sent { live: true })
    }

    /// **Is the focus, or the box just clicked, a password field?** UI Automation's `IsPassword`,
    /// asked of both because text goes to one or the other: keys to the focus, a value pattern to
    /// the aim. **Unreadable reads as no** — see `Hands::into_a_secret` for why that direction.
    pub fn typing_into_a_secret(&mut self) -> bool {
        let (focused, aimed) = self.where_text_goes();
        let secret = |e: &Option<windows::Win32::UI::Accessibility::IUIAutomationElement>| {
            e.as_ref().is_some_and(|e| unsafe { e.CurrentIsPassword() }.is_ok_and(|b| b.as_bool()))
        };
        secret(&focused) || secret(&aimed)
    }

    /// **The two elements text could reach right now**: the focus, where keys go, and the box just
    /// clicked while that click is fresh, where a value pattern writes. The one place in this file
    /// that reads the focus — `tests/hands.rs` holds it to that.
    fn where_text_goes(&mut self) -> (Option<windows::Win32::UI::Accessibility::IUIAutomationElement>,
                                      Option<windows::Win32::UI::Accessibility::IUIAutomationElement>) {
        if self.automation.is_none() {
            self.automation = crate::open_automation().ok();
        }
        let Some(automation) = &self.automation else { return (None, None) };
        let focused = unsafe { automation.GetFocusedElement() }.ok();
        let aimed = match self.aimed {
            Some(((x, y), when)) if when.elapsed() <= AIM_LASTS => unsafe {
                automation.ElementFromPoint(POINT { x: x as i32, y: y as i32 })
            }.ok(),
            _ => None,
        };
        (focused, aimed)
    }

    /// **Is there a field where the text would go?** The box just clicked if that is fresh, the
    /// focus otherwise; `None` when UI Automation will not say.
    pub fn typing_lands_here(&mut self) -> Option<bool> {
        let (focused, aimed) = self.where_text_goes();
        Some(crate::patterns::takes_text(&aimed.or(focused)?))
    }

    pub fn type_string(&mut self, text: &str) -> Did {
        //: **A second line behind `perform`'s.** A harness calls this directly and never goes
        //: through a session, and a password guard that only one caller passes is not a guard.
        if self.typing_into_a_secret() {
            self.aimed = None;
            return self.record(Did::Refused("a password field".into()));
        }
        //: **Tried before the dry check, not after.** A dry run that went straight to
        //: `Sent { live: false }` would report the keyboard arm on a machine where the pattern arm
        //: is what would have happened, and a harness comparing the two would be comparing one.
        self.why = String::new();
        //: **Taken, so one click grants one type.** A second `type` after the same click, or a type
        //: in a later turn whose own click never happened, falls back to the keyboard rather than
        //: writing into a box somebody chose a minute ago.
        let aim = self.aimed.take();
        if self.typing_patterns {
            match aim {
                //: **A stale aim is no aim.** Effects in one `perform` are 250 ms apart at
                //: `max_per_second` 4, so a second is generous; a model round trip is 0.61 s
                //: median, which is the thing this must not span.
                Some((_, when)) if when.elapsed() > AIM_LASTS => {
                    self.why = format!("the click was {:.1} s ago", when.elapsed().as_secs_f64());
                }
                None => self.why = "nothing was clicked first".into(),
                Some(((x, y), _)) => {
                    if self.automation.is_none() {
                        self.automation = crate::open_automation().ok();
                    }
                    if let Some(automation) = &self.automation {
                        let reach = self.reach;
                        match crate::patterns::set_value(automation, x, y, text, self.live,
                                                         &|owner| may_touch(reach, owner)) {
                            Typed::Set | Typed::WouldSet => {
                                self.by_value += 1;
                                return self.record(Did::Sent { live: self.live });
                            }
                            Typed::Keys(why) => self.why = why,
                        }
                    }
                }
            }
        }
        //: **Then characters posted** — to the window under the click just made while it is fresh,
        //: which for a classic box is the box itself whatever has focus, and otherwise where a key
        //: would go. Inserted at the caret, as keys are, so a box already holding something is
        //: added to and not replaced.
        if self.messages {
            let at_aim = match aim {
                Some(((x, y), when)) if when.elapsed() <= AIM_LASTS => crate::addressed::window_at(x, y),
                _ => None,
            };
            let target = match at_aim {
                Some(h) => Ok(h),
                None => self.key_target(),
            };
            let units = typing_units(text);
            match target {
                Err(why) => return self.record(Did::Refused(why)),
                Ok(h) => match self.posted(h, |h| crate::addressed::chars(h, &units)) {
                    Ok(did) => return did,
                    Err(why) => self.because(why),
                },
            }
        }
        if !self.may_send_input {
            return self.record(Did::Refused(format!("only the person's keyboard would reach it: {}", self.why)));
        }
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
    /// **Always sent somewhere or refused on the record** — a covered point is `Elsewhere`, read
    /// back through [`Hands::landed`] as every other click is — so `true`: these hands can point.
    fn click_inside(&mut self, _within: &Widget, at: (i64, i64)) -> bool {
        self.click_point(at.0, at.1);
        true
    }
    fn type_text(&mut self, text: &str) {
        self.type_string(text);
    }
    fn into_a_secret(&mut self) -> bool {
        self.typing_into_a_secret()
    }
    fn typing_lands(&mut self) -> Option<bool> {
        self.typing_lands_here()
    }
    /// **The record of the last action**, which every path through these hands writes.
    fn landed(&self) -> Option<bool> {
        match self.done.last()? {
            Did::Sent { live: true } | Did::Operated { live: true, .. } => Some(true),
            Did::Elsewhere { .. } | Did::Refused(_) => Some(false),
            _ => None,
        }
    }
    fn press(&mut self, key: &str) {
        self.press_key(key);
    }
}
