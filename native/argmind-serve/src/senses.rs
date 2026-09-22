//! A blackboard: every sense writes its latest reading here, and thinking reads whatever is there.
//!
//! **The owner, 2026-09-17:** 각각의 인지를 병렬로 구동하고, 계속 리프레시하면 생각하는 부분이
//! 그때그때 처리된걸 읽어오면 더 빠르지 않을까.
//!
//! The shape is right, and for anything slow it is close to mandatory: one thread owns the session
//! because COM says so, and a camera look costs about 8 s — eight seconds in which nothing answers
//! anybody. A sense that refreshes on its own thread and leaves its reading here stops that being
//! true.
//!
//! ## Three parts, and the third is the one the proposal did not have
//!
//! 1. **A slot per sense**, newest wins, nobody waits. Reading is a lock and a clone.
//! 2. **Every reading says when it was taken**, because a sense that refreshes on its own is a
//!    sense whose answer is always a little old, and *a reading* presented where *a reading from
//!    twelve seconds ago* is the truth is a lie the model cannot catch. [`Senses::latest`] hands
//!    back the age and [`Senses::render`] prints it.
//! 3. **A sense stands down while somebody is being answered** — [`Turn`]. Measured 2026-09-17: a
//!    look running beside the conversation takes the answer from a median of 1.0 s to 3.2 s, and
//!    one answer in ten waits out most of a whole look, 10.81 s. Without this, parallelism spends
//!    the interaction budget on filling a panel.
//!
//! Both halves of the third rule are buildable, which was itself a measurement: dropping the socket
//! **does** free the card. An answer after abandoning a look mid-flight costs 1.09 s against 1.12 s
//! with a free card and 3.22 s with a busy one. So a sense both declines to start during a turn and
//! abandons one already running, and `Local::ask_while` is what does the second.
//!
//! ## What must not move onto the blackboard
//!
//! **Resolving a click.** `Session::find` re-reads the screen immediately before acting because a
//! listed widget is gone **13 times in 14** over the second a call takes. Cached perception is for
//! thinking and describing; acting reads fresh, always. Nothing here is on the acting path and
//! nothing here should ever be put on it.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// What one sense last had to say.
#[derive(Debug, Clone)]
pub struct Reading {
    /// In words, because that is what a prompt takes. A sense that produces something else converts
    /// it on its own thread, where the time it costs is nobody's wait.
    pub what: String,
    pub taken: Instant,
    /// What it cost to take, in seconds. **Kept because a sense nobody can afford should be visible
    /// as such**, not inferred from the agent feeling slow.
    pub took: f64,
}

/// Every sense's latest reading.
///
/// **Cloneable and shared**, like [`argmind_env_windows::Watcher`]: the producers hold one and the
/// consumer holds one, and there is no ownership question to get wrong.
#[derive(Clone, Default)]
pub struct Senses(Arc<Mutex<BTreeMap<String, Reading>>>);

impl Senses {
    /// Put down what this sense just found. **Newest wins and nothing queues** — a sense that fell
    /// behind has nothing worth catching up on, only a stale reading to be replaced.
    pub fn saw(&self, sense: &str, what: String, took: f64) {
        if let Ok(mut all) = self.0.lock() {
            all.insert(sense.to_string(), Reading { what, taken: Instant::now(), took });
        }
    }

    /// Everything, with each reading's age in seconds. **Never waits**: a sense that is mid-refresh
    /// hands back what it had, which is the whole point of the arrangement.
    /// Drop a sense's reading entirely.
    ///
    /// **Not the same as reporting nothing.** A blank reading is not rendered, but it stays in the
    /// map with an age that keeps growing; this is for a sense that has *stopped*, where the last
    /// thing it saw is no longer a fact about now and saying *마지막 움직임 3초 전* about a camera
    /// that has been off for an hour would be a lie with a timestamp on it.
    pub fn forget(&self, sense: &str) {
        if let Ok(mut all) = self.0.lock() {
            all.remove(sense);
        }
    }

    pub fn latest(&self) -> Vec<(String, Reading, f64)> {
        let now = Instant::now();
        self.0.lock().map(|all| {
            all.iter().map(|(name, r)| {
                (name.clone(), r.clone(), now.duration_since(r.taken).as_secs_f64())
            }).collect()
        }).unwrap_or_default()
    }

    /// One reading by name, if there is one.
    pub fn of(&self, sense: &str) -> Option<Reading> {
        self.0.lock().ok().and_then(|all| all.get(sense).cloned())
    }

    /// The block a prompt gets, or **nothing at all when no sense has reported.**
    ///
    /// The age is in the line, not in a field somebody might drop: *카메라 (12초 전)* is a
    /// different claim from *카메라*, and the second one is false.
    ///
    /// `skip` is for the sense the caller is already showing in full — the screen is rendered as a
    /// numbered list elsewhere and would otherwise appear twice.
    pub fn render(&self, skip: &str) -> Option<String> {
        let mut lines: Vec<String> = Vec::new();
        for (name, reading, age) in self.latest() {
            if name == skip || reading.what.trim().is_empty() {
                continue;
            }
            lines.push(format!("  {name} ({age:.0}초 전): {}", reading.what.trim()));
        }
        if lines.is_empty() {
            return None;
        }
        lines.insert(0, "다른 감각이 마지막으로 전한 것:".to_string());
        Some(lines.join("\n"))
    }
}

/// Whether somebody is being answered right now.
///
/// **The third rule lives here**, and it is one bit shared between the thread that answers and
/// every thread that senses. A sense asks before starting and keeps asking while it runs.
#[derive(Clone, Default)]
pub struct Turn(Arc<AtomicBool>);

impl Turn {
    /// Somebody just said something. **Hold the returned guard for as long as the answer takes** —
    /// it clears on drop, so an early return or a panic cannot leave the senses standing down for
    /// the rest of the session.
    pub fn begins(&self) -> Answering {
        self.0.store(true, Ordering::SeqCst);
        Answering(Arc::clone(&self.0))
    }

    /// Is an answer in flight? **A sense checks this before it starts and while it runs.**
    pub fn in_flight(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }

    /// The inverse, in the shape `Local::ask_while` wants.
    pub fn carry_on(&self) -> impl Fn() -> bool + use<> {
        let flag = Arc::clone(&self.0);
        move || !flag.load(Ordering::SeqCst)
    }
}

/// An answer in progress. Clears the flag when it goes out of scope.
pub struct Answering(Arc<AtomicBool>);

impl Drop for Answering {
    fn drop(&mut self) {
        self.0.store(false, Ordering::SeqCst);
    }
}
