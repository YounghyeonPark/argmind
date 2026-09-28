//! Noticing: a thought that starts from the screen rather than from being spoken to.
//!
//! **Acting is not a stage that always runs, and neither is thinking.** A `Watch` is a filter on
//! perception change with a floor on how often it may fire, and the floor is not politeness.
//! Measured: something on screen changes in **13 seconds out of 14**, and a model call takes
//! 0.3–2 s — so an unfloored watch is a model that never stops and an agent that never listens.
//!
//! What a watch produces is a `Trigger` that was **not asked for**, and that word travels: an
//! unprompted effect without standing permission is downgraded to speech, and an unprompted
//! sentence is not spoken aloud unless something says it may. Both of those live above the core;
//! what the core owes them is an honest `asked = false`.

use crate::contract::Widget;
use crate::intent::Trigger;

/// A string the way Python's `repr` writes one.
///
/// **A watch's description becomes the text of a `Trigger`, so it is prompt content**, and this
/// project does not change prompts casually — a sentence the model reads is not a log line. Rust's
/// `{:?}` writes `"Character Map"` where Python writes `'Character Map'`, and on a name
/// containing an apostrophe Python switches quotes rather than escaping: `"it's here"`. Three
/// wordings for one event, and nothing downstream would ever notice which one it got.
pub fn py_repr(s: &str) -> String {
    //: Single quotes unless that would need escaping and double quotes would not — which is
    //: exactly Python's rule, not an approximation of it.
    let quote = if s.contains('\'') && !s.contains('"') { '"' } else { '\'' };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            //: Printable non-ASCII stays as itself — Python 3 stopped escaping it, and a
            //: watch announcing escape sequences would tell the model nothing it can read.
            c if (c as u32) < 0x20 || c as u32 == 0x7f => {
                out.push_str(&format!("\\x{:02x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

/// The floor under how often a watch may fire, in seconds.
///
/// Three, because perception changes 13 seconds in 14 and a call costs 0.3–2 s. A watch that fires
/// on every change is a model that never stops.
/// **What the camera says about the person, as the sentence a watch's trigger carries — measured
/// and deliberately NOT wired into `notice`.** Read the second section before changing that.
///
/// The owner, 2026-09-28, asked whether argmind should stay quiet when somebody has turned away:
/// *말을 걸지말지는 상황에 맞게 판단하면 됨* — judge it by the situation. So this was built as a fact
/// for the model rather than a gate, on the assumption that silence is already an answer the model
/// gives on this path.
///
/// ## Why it is not wired: the model narrates it
///
/// `argmind-measure --bin turned`, 28 unprompted moments over four captured screens, four arms,
/// two instructions: **224 of 224 spoke.** Not once did `qwen3:8b` choose silence on this path —
/// including under an instruction that said *otherwise leave "say" empty*. The assumption above
/// was false, and it was the one everything rested on.
///
/// Worse, the fact became something to say: *You turned your head, I noticed.* *사용자가 고개를
/// 돌리고 있어요.* **The neutral control narrated most of all** — [`PRESENT`] drew a remark about
/// the person 18 times in 28 — so it is the presence of a sentence about them, not its content,
/// that does it. Wired as built, argmind would announce that somebody had looked away.
///
/// And telling it to be quiet moved *wanting to click* from 5 to 9 of 28, which is the exact
/// finding `noticing` recorded long before: *telling it to be quiet in the prompt bought no silence
/// and started it wanting to click things.* Reproduced, on a different axis.
///
/// **So what is kept is the instrument.** This function exists so the harness renders the sentence
/// the application would send; the application does not send it. `docs/findings/attention.md`.
///
/// **Parallel on purpose**, *사용자가 ___ 있음* in all three, so a harness can hold everything but the
/// content still. The dials taught that a content-free sentence in a slot moves results on its own
/// — the placebo fixed the index leak as well as the dials did — so the comparison that means
/// anything is [`TURNED`] against [`PRESENT`], never [`TURNED`] against nothing.
pub const FACING: &str = "사용자가 화면 쪽을 보고 있음";
pub const TURNED: &str = "사용자가 고개를 돌리고 있음";
/// **The control, and true every time it could appear**: `notice` only runs while
/// `somebody_about` says somebody is there, so this carries no information about attention.
pub const PRESENT: &str = "사용자가 자리에 있음";

/// A watch's sentence with what is known about where the person is looking, **or unchanged**.
///
/// `None` returns the sentence exactly, which is what makes adding this free at rest: with the
/// switch off, with no models, or with a head that could not be read, the model is told what it
/// was always told. **One function** for the application and the harness, because a harness that
/// wrote the sentence its own way would be measuring a prompt nobody sends — `noticing` already
/// records that lesson about `py_repr`.
pub fn regarding(text: &str, them: Option<&str>) -> String {
    match them {
        Some(fact) => format!("{text} — {fact}"),
        None => text.to_string(),
    }
}

pub const MIN_GAP: f64 = 3.0;

/// What a watch is looking for.
#[derive(Debug, Clone, PartialEq)]
pub enum Looking {
    /// The person moved to a different window.
    ///
    /// The rarest change measured — 6 of 14 over a second, 2 of 14 over 0.3 s — and still frequent
    /// enough that the floor does real work.
    WindowChanged,
    /// Something whose name contains this showed up. The useful shape of *"tell me when X"*.
    Appeared(String),
    /// Something whose name contains this went away.
    Vanished(String),
}

/// A filter on perception change, with a floor on how often it may fire.
#[derive(Debug, Clone)]
pub struct Watch {
    pub looking: Looking,
    pub gap: f64,
    /// Seconds, on the same clock the caller passes to [`Watch::fired`]. **The clock is a
    /// parameter** so a test can move time without waiting for it — and so nothing in the core
    /// reads a wall clock, which is the same rule that keeps a platform out of here.
    last: f64,
    /// Never fired yet. Distinct from `last = 0.0`, which would put the first fire three seconds
    /// after the epoch rather than three seconds after the first one.
    ever: bool,
}

impl Watch {
    pub fn new(looking: Looking) -> Self {
        Watch { looking, gap: MIN_GAP, last: 0.0, ever: false }
    }

    pub fn every(looking: Looking, gap: f64) -> Self {
        Watch { looking, gap, last: 0.0, ever: false }
    }

    /// What happened, or nothing worth a thought. **Ignores the floor** — that is [`fired`]'s job,
    /// and separating them is what lets the description be tested without a clock.
    ///
    /// [`fired`]: Watch::fired
    pub fn look(&self, before: &[Widget], after: &[Widget]) -> Option<String> {
        match &self.looking {
            Looking::WindowChanged => {
                let was = before.first().map(|w| w.window.as_str()).unwrap_or("");
                let now = after.first().map(|w| w.window.as_str()).unwrap_or("");
                //: **And `now` must be something.** A screen that went empty is not the person
                //: moving to a different window, and reporting it as one would have the agent
                //: announce a window called nothing.
                (was != now && !now.is_empty())
                    .then(|| format!("the window changed to {}", py_repr(now)))
            }
            Looking::Appeared(needle) => {
                let needle = needle.to_lowercase();
                let had: std::collections::BTreeSet<&str> =
                    before.iter().map(|w| w.name.as_str()).collect();
                after.iter()
                    .find(|w| w.name.to_lowercase().contains(&needle)
                          && !had.contains(w.name.as_str()))
                    .map(|w| format!("{} appeared", py_repr(&w.name)))
            }
            Looking::Vanished(needle) => {
                let needle = needle.to_lowercase();
                let now: std::collections::BTreeSet<&str> =
                    after.iter().map(|w| w.name.as_str()).collect();
                before.iter()
                    .find(|w| w.name.to_lowercase().contains(&needle)
                          && !now.contains(w.name.as_str()))
                    .map(|w| format!("{} went away", py_repr(&w.name)))
            }
        }
    }

    /// The trigger this watch produces now, if any. `now` is seconds on any monotonic clock.
    pub fn fired(&mut self, before: &[Widget], after: &[Widget], now: f64) -> Option<Trigger> {
        if self.ever && now - self.last < self.gap {
            return None;
        }
        let what = self.look(before, after)?;
        self.last = now;
        self.ever = true;
        //: **`asked = false`, and everything downstream turns on it.** An effect from here is
        //: downgraded to speech without standing permission, and a sentence from here is not said
        //: aloud unless something says it may.
        Some(Trigger::perceived(&what))
    }
}

/// Everything a set of watches has to say about one change.
///
/// **The first call has no `before` and returns nothing**, whatever happened — treating an empty
/// screen as a previous state would report the whole desktop as newly appeared.
pub fn poll(watches: &mut [Watch], before: &[Widget], after: &[Widget], now: f64) -> Vec<Trigger> {
    if watches.is_empty() || before.is_empty() {
        return Vec::new();
    }
    watches.iter_mut().filter_map(|w| w.fired(before, after, now)).collect()
}
