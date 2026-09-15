//! Effects waiting for a person to press a button.
//!
//! **This is where the permission model lives.** An effect the model returned is not something the
//! agent did; it is something it offered. `app.py` makes that argument with one AST test counting
//! the callers of `perform` and finding exactly one, and the same argument has to hold here — so
//! nothing in this module performs anything. It hands back an effect when a confirmation names it,
//! and the single place that acts on that is the server's confirm route.
//!
//! A proposal holds the **effect**, not a resolved widget. Between proposing and confirming the
//! screen moved: a listed widget is gone 13 times in 14 over the second a call takes, so what was
//! at those coordinates when the offer was made says nothing about what is there when the button is
//! pressed. The lookup happens again at that moment, and it can fail — which is a proper outcome
//! and not an error.

use argmind_core::intent::Effect;

/// One effect, waiting.
#[derive(Debug, Clone)]
pub struct Proposal {
    pub token: u64,
    pub effect: Effect,
    /// What the person reads on the button.
    pub label: String,
    /// What UI Automation says is at that point right now.
    ///
    /// **A second opinion, not a gate.** A widget's own centre resolves back to itself about seven
    /// times in ten, and a caption button resolving to `application` is the hit test disagreeing
    /// with a pixel that is correct. It is shown and it never blocks.
    pub hint: String,
}

//: `Effect` in the core derives no `PartialEq`, and adding one there to make a test in a
//: dependent crate shorter would be the tail wagging the dog. Tests compare the fields.
/// What happened to a confirmation or a refusal.
#[derive(Debug, Clone)]
pub enum Answered {
    /// Here is the effect to act on. **The only way one comes out of this module.**
    Run(Proposal),
    /// The person said no. The effect is returned so the conversation can be told — pressing *no*
    /// is a thing the person said, and until `app.py` recorded it the next turn had no idea it had
    /// just been turned down.
    TurnedDown(Proposal),
    /// No proposal by that number is pending: it was already answered, or it never existed.
    ///
    /// **Not an error.** Two clicks on one button, a reload with a stale page, a confirmation
    /// racing a new turn — all ordinary, and all must do nothing rather than act on whatever is
    /// pending now.
    Gone,
}

/// Everything currently waiting.
#[derive(Debug, Default)]
pub struct Proposals {
    pending: Vec<Proposal>,
    next: u64,
}

impl Proposals {
    /// File an offer and return its number.
    ///
    /// **Numbers are never reused.** A stale button in a page nobody reloaded would otherwise
    /// confirm a later proposal that happens to have inherited the number — the person would read
    /// one action and authorise a different one, which is the failure this whole mechanism exists
    /// to prevent.
    pub fn offer(&mut self, effect: Effect, label: String, hint: String) -> u64 {
        let token = self.next;
        self.next += 1;
        self.pending.push(Proposal { token, effect, label, hint });
        token
    }

    pub fn waiting(&self) -> &[Proposal] {
        &self.pending
    }

    fn take(&mut self, token: u64) -> Option<Proposal> {
        let at = self.pending.iter().position(|p| p.token == token)?;
        Some(self.pending.remove(at))
    }

    /// Somebody pressed yes.
    pub fn confirm(&mut self, token: u64) -> Answered {
        match self.take(token) {
            Some(p) => Answered::Run(p),
            None => Answered::Gone,
        }
    }

    /// Somebody pressed no.
    pub fn decline(&mut self, token: u64) -> Answered {
        match self.take(token) {
            Some(p) => Answered::TurnedDown(p),
            None => Answered::Gone,
        }
    }

    /// Forget everything still waiting, returning it.
    ///
    /// **A new turn does not silently keep the last turn's buttons alive.** An offer the person
    /// never answered and then said something else about is an offer they moved on from, and
    /// leaving it pressable means a button whose label describes a screen that has gone.
    pub fn clear(&mut self) -> Vec<Proposal> {
        std::mem::take(&mut self.pending)
    }
}
