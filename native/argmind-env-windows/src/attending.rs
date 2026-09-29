//! Which window the agent is looking at — **not the one in front.**
//!
//! `Eyes::foreground` answers "what is in front right now", and for an agent driven from a window
//! of its own that is the wrong question: the moment a person reaches for the interface, the
//! interface is in front, and the agent would read *itself*. `app.py` solves it with an
//! `attending()` that skips this process's own windows.
//!
//! **A browser made it harder, not easier.** The interface was a tab in somebody else's process,
//! so "skip my own windows" did not identify it — and a person types into the interface before
//! every turn, so it is in front at exactly the moment the agent is asked to look.
//!
//! The rule used to be *the newest foreground window that is not the one in front now*, which was
//! the best available while the page was a tab in somebody else's process. **The rule now is
//! `app.py`'s original one: the newest window that does not belong to this process.**
//!
//! ## Why it changed, and the prediction that was written first
//!
//! This file said, of the old rule:
//!
//! > What it assumes, stated so it can be wrong. It assumes the person is in the interface when
//! > they act. That holds for a typed turn — they typed it — and it would not hold for a turn
//! > started by a `Watch` firing while they work elsewhere. **Nothing here fires on a watch yet;
//! > when something does, this is the thing to revisit.**
//!
//! On 2026-09-17 watching became the default and a thinking tick was added, so things fire on a
//! watch constantly — and the assumption broke exactly where it said it would. Excluding *the one
//! in front* means that while the person works in Chrome, and argmind was the window before it,
//! the newest handle that is not Chrome **is argmind**. Over 150 seconds of somebody moving
//! between an editor, two browsers and a file manager, **4 of 8 notices read `the window changed
//! to 'argmind'`** — argmind reading its own transcript, once per model call.
//!
//! The prediction written before the change: under the new rule the same observation produces
//! **zero** notices naming argmind. Recorded in `docs/findings/the-pc-app.md` with what happened.
//!
//! **Both halves had to go.** Keeping the `in_front` exclusion and adding a process filter answers
//! with the window *before* the one the person is using — right for a typed turn and wrong for
//! every watch. The front window is the answer whenever it is not ours.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use argmind_core::contract::{Eyes, Widget};

use crate::launch::{passed_through, pid_of, window_class, window_title};
use crate::WindowsEyes;

#[derive(Default)]
struct Seen {
    /// Newest first, distinct handles, short.
    recent: Vec<isize>,
}

/// A shared record of what the person has been working in.
#[derive(Clone, Default)]
pub struct Watcher(Arc<Mutex<Seen>>);

impl Watcher {
    /// Note what is in front. Cheap enough to call several times a second.
    pub fn saw(&self, handle: isize) {
        if handle == 0 {
            return;
        }
        //: **Never recorded, so never attended.** Passing through the taskbar leaves the last real
        //: window at the head of `recent`, and arriving at the next one makes that the head — so the
        //: watch sees one change, from where they were to where they went, which is what happened.
        //: A handle that has gone answers an empty class and is recorded as before.
        if passed_through(&window_class(handle)) {
            return;
        }
        let Ok(mut seen) = self.0.lock() else { return };
        if seen.recent.first() == Some(&handle) {
            return;
        }
        seen.recent.retain(|h| *h != handle);
        seen.recent.insert(0, handle);
        //: Four is plenty and an unbounded list is a leak nobody notices for a week.
        seen.recent.truncate(4);
    }

    /// The window the agent should read, given what is in front now.
    ///
    /// **The newest one that is not ours.** When the person is working in something else that is
    /// the thing in front; when they are in argmind it is whatever they came from.
    ///
    /// `None` when every window seen is ours — a fresh start where the interface is all there is.
    /// **Answering with the interface would be worse than answering with nothing**: the agent
    /// would describe its own transcript back to the person, and with a watch running it would do
    /// so every time they glanced at it.
    ///
    /// `in_front` is still taken so the caller does not have to know whether it matters; it is
    /// recorded here, so it is at the head of `recent` before the search.
    pub fn attended(&self, in_front: isize) -> Option<isize> {
        self.attended_where(in_front, ours)
    }

    /// The same rule with **ownership handed in rather than asked of the operating system.**
    ///
    /// Pure, and that is the only reason the rule has tests at all: [`ours`] is a `GetWindowThread\
    /// ProcessId` call, so a test written against `attended` can only use handles that belong to
    /// nobody — which is precisely the case the rule does not turn on. The interesting sequence is
    /// *the person is in our own window*, and without this it could not be written down.
    pub fn attended_where(&self, in_front: isize,
                          mine: impl Fn(isize) -> bool) -> Option<isize> {
        self.saw(in_front);
        let seen = self.0.lock().ok()?;
        seen.recent.iter().copied().find(|h| !mine(*h))
    }

    /// Start a thread that samples the foreground until the process ends.
    pub fn watch(&self) {
        let me = self.clone();
        std::thread::spawn(move || {
            let eyes = WindowsEyes::new();
            let Ok(eyes) = eyes else { return };
            loop {
                me.saw(eyes.foreground_handle());
                std::thread::sleep(Duration::from_millis(400));
            }
        });
    }
}

/// Is this window one of ours?
///
/// **By process id, which is the sharp test and is available again** now the page lives in a window
/// this process owns. A handle that has gone answers `None` and is treated as not ours, which is
/// harmless: reading a dead handle produces nothing and the next sample drops it.
pub fn ours(handle: isize) -> bool {
    pid_of(handle) == Some(std::process::id())
}

/// Eyes that read the attended window rather than the one in front.
pub struct Attending {
    pub inner: WindowsEyes,
    pub watcher: Watcher,
}

impl Attending {
    pub fn new(watcher: Watcher) -> windows::core::Result<Self> {
        Ok(Attending { inner: WindowsEyes::new()?, watcher })
    }

    /// The handle being attended, and its title — what the header shows.
    pub fn window(&self) -> (isize, String) {
        let front = self.inner.foreground_handle();
        match self.watcher.attended(front) {
            Some(h) => (h, window_title(h)),
            None => (0, String::new()),
        }
    }

    /// Which process owns it, for binding hands that must not stray.
    pub fn owner(&self) -> Option<u32> {
        pid_of(self.window().0)
    }
}

impl Eyes for Attending {
    fn foreground(&mut self) -> Vec<Widget> {
        let (handle, _) = self.window();
        //: **Nothing rather than the interface.** An empty screen makes the model say it cannot
        //: see anything, which is true and is a better answer than a description of the page the
        //: person is reading.
        if handle == 0 {
            return Vec::new();
        }
        self.inner.read(handle)
    }

    fn timed_foreground(&mut self) -> (Vec<Widget>, f64) {
        let t0 = std::time::Instant::now();
        let widgets = self.foreground();
        (widgets, t0.elapsed().as_secs_f64())
    }
}
