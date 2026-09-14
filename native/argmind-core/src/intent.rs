//! What a thought is: a trigger, the effects it wants, and what came of them.

use crate::contract::Widget;

/// Why the agent is thinking. **`asked` is the field the whole permission model turns on**, so a
/// spoken turn is a `said` trigger like any other rather than a kind of its own — a separate kind
/// would make every spoken request unprompted, inverting the gate by a spelling.
#[derive(Clone, Debug)]
pub struct Trigger {
    pub text: String,
    pub asked: bool,
}

impl Trigger {
    pub fn said(text: &str) -> Self {
        Trigger { text: text.into(), asked: true }
    }
    pub fn perceived(text: &str) -> Self {
        Trigger { text: text.into(), asked: false }
    }
}

/// One thing to do. **It carries a name and a role, never a rectangle** — a listed widget is gone
/// 13 times in 14 over the second a call takes, so the target is re-resolved immediately before
/// acting and the rectangle is exactly what moves.
#[derive(Clone, Debug)]
pub struct Effect {
    pub kind: String,
    pub target: String,
    pub role: String,
    pub text: String,
    /// `because.asked`, or `None` when nothing triggered it. **An effect with no trigger is not
    /// unprompted**, which is an open edge in the permission model rather than a decision — what
    /// actually holds it is the single call site in the app.
    pub because_asked: Option<bool>,
}

impl Effect {
    pub fn unprompted(&self) -> bool {
        matches!(self.because_asked, Some(false))
    }
}

/// One effect that actually ran: what was decided, what it resolved to *then*, and whether the
/// world moved. Three separate things on purpose — each can be true while the next is false.
#[derive(Clone, Debug)]
pub struct Done {
    pub effect: Effect,
    pub target: Option<Widget>,
    pub confirmed: Option<bool>,
}

/// What one call produced.
#[derive(Clone, Debug)]
pub struct Intent {
    pub say: String,
    pub effects: Vec<Effect>,
}

/// One whole turn, as it happened.
#[derive(Clone, Debug)]
pub struct Exchange {
    pub n: usize,
    pub trigger: Trigger,
    pub said: String,
    pub did: Vec<Done>,
    /// Effects that were downgraded to speech: nobody asked, and there is no standing permission.
    pub proposed: Vec<Effect>,
    pub refused: Vec<Effect>,
    pub n_widgets: usize,
    pub look_s: f64,
}
