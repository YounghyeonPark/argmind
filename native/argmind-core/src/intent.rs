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

/// **Which of several identically named things**, when a person has said.
///
/// `index` into the list the eyes gave, and `of` how many there were when the question was put.
/// **Both, because the second is what makes the first safe**: three buttons called `Open` become
/// two while somebody reads the question, and *the second one* then means a different button.
/// `Session::perform` compares the count before it uses the index and refuses if it moved.
///
/// **Not a rectangle, and that distinction is the whole reason this is allowed.** A rectangle is
/// stale 13 times in 14 over the second a call takes; an ordinal among same-named widgets is
/// re-resolved at act time against a fresh reading, and the count check is what catches the case
/// where re-resolving is not enough.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Among {
    pub index: usize,
    pub of: usize,
}

/// One thing to do. **It carries a name and a role, never a rectangle** — a listed widget is gone
/// 13 times in 14 over the second a call takes, so the target is re-resolved immediately before
/// acting and the rectangle is exactly what moves.
#[derive(Clone, Debug, Default)]
pub struct Effect {
    pub kind: String,
    pub target: String,
    pub role: String,
    pub text: String,
    /// `because.asked`, or `None` when nothing triggered it. **An effect with no trigger is not
    /// unprompted**, which is an open edge in the permission model rather than a decision — what
    /// actually holds it is the single call site in the app.
    pub because_asked: Option<bool>,
    /// **Set by a person and by nothing else.** The model never emits it — it is shown the same two
    /// fields and cannot name the one it means — and the reader never fills it. It arrives when
    /// somebody answers *which `Open`?*, which is the only thing in this program that knows.
    pub which: Option<Among>,
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
    /// **The reply was what they had just said, with a question mark on it.**
    ///
    /// A flag rather than a replacement sentence, because the replacement has to be in the
    /// person's language and **this crate names no language for the same reason it names no
    /// platform**. The core can tell that a mirror happened; only an interface knows how to say
    /// so. See [`crate::converse::parroted`].
    pub parroted: bool,
    /// **How sure the model was of the action it wrote**: the probability of the tokens of the
    /// reply's `do`, given everything before them. `None` when there was no action or the model
    /// cannot say. A score, not a calibrated probability — 2026-10-21 measured this model saying
    /// 0.90 for a wrong answer — so what it may gate is decided by measurement, not by its value.
    pub sure: Option<f64>,
    /// **The model's own words, when they came back in a language the person did not set and
    /// `say` is their rewrite** — kept, as a mirrored reply's are, so the next report of a wrong
    /// rewrite can be diagnosed, and so an offer in either reading still holds its action.
    pub rewritten_from: Option<String>,
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
