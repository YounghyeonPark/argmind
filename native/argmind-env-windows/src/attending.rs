//! Which window the agent is looking at — **not the one in front.**
//!
//! `Eyes::foreground` answers "what is in front right now", and for an agent driven from a window
//! of its own that is the wrong question: the moment a person reaches for the interface, the
//! interface is in front, and the agent would read *itself*. `app.py` solves it with an
//! `attending()` that skips this process's own windows.
//!
//! **A browser makes it harder, not easier.** The interface is now a tab in somebody else's
//! process, so "skip my own windows" no longer identifies it — and a person types into the browser
//! before every turn, so the browser is in front at exactly the moment the agent is asked to look.
//!
//! So the rule here is different and it is deliberately simple: **the newest foreground window that
//! is not the one in front now.** A watcher samples the foreground and keeps the distinct handles
//! it has seen; while the person is in the interface, that answers with the window they were in
//! before it, which is the one they mean.
//!
//! **What it assumes, stated so it can be wrong.** It assumes the person is in the interface when
//! they act. That holds for a typed turn — they typed it — and it would not hold for a turn started
//! by a `Watch` firing while they work elsewhere. Nothing here fires on a watch yet; when something
//! does, this is the thing to revisit.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use argmind_core::contract::{Eyes, Widget};

use crate::launch::{pid_of, window_title};
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
    /// `None` when nothing else has ever been in front — a fresh start where the interface is the
    /// only window seen. **Answering with the interface would be worse than answering with
    /// nothing**: the agent would describe its own transcript back to the person.
    pub fn attended(&self, in_front: isize) -> Option<isize> {
        let seen = self.0.lock().ok()?;
        seen.recent.iter().copied().find(|h| *h != in_front)
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
