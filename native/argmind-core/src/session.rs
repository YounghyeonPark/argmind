//! observe / poll / decide / perform — separately, because **acting is not a stage that always
//! runs, it is something decided.**

use crate::contract::{Eyes, Hands, Speaks, Widget};
use crate::converse::{read_reply, render, MAX_LISTED};
use crate::intent::{Done, Effect, Exchange, Intent, Trigger};

pub const REMEMBER_TURNS: i64 = 12;
pub const NOTHING_DONE: &str = "nothing was done";
pub const TURN_OPENS: [&str; 2] = ["them", "noticed"];
pub const PAST: usize = 0;
pub const PLAIN: usize = 1;
pub const ING: usize = 2;

/// The verb of each effect kind in three forms. **A kind that is not listed falls back to `key`'s
/// row**, which is how an unlisted kind loses its object — `'to press  — …'`, naming nothing. A
/// defect, reproduced rather than fixed, because a port that fixes one is two changes.
pub fn verb(kind: &str, form: usize) -> &'static str {
    let row: [&str; 3] = match kind {
        "click" => ["clicked", "click", "clicking"],
        "type" => ["typed", "type", "typing"],
        _ => ["pressed", "press", "pressing"],
    };
    row[form]
}

/// One short phrase for an effect. **`form` picks the verb and that is the whole state marker**:
/// past says it happened, the other two say it did not.
pub fn describe(effect: &Effect, widget: Option<&Widget>, form: usize) -> String {
    let v = verb(&effect.kind, form);
    if effect.kind == "click" {
        let role = match widget {
            Some(w) => w.role.clone(),
            None => if effect.role.is_empty() { "thing".into() } else { effect.role.clone() },
        };
        return format!("{v} the {role} \"{}\"", effect.target);
    }
    if effect.kind == "type" {
        return format!("{v} \"{}\"", effect.text);
    }
    format!("{v} {}", effect.text)
}

/// The last `turns` turns, cut where a turn begins and never inside one.
///
/// `None` is the whole conversation; **0 is none of it, and that is a bug fixed rather than a
/// choice** — this was a negative slice, and `history[-0:]` is the entire list, so the one setting
/// a person reaches for to turn memory off turned it all the way up. Cutting on boundaries is the
/// other half: 353 of 3,312 prompts built by the line-slicing version opened mid-turn.
pub fn recent(history: &[(String, String)], turns: Option<i64>) -> Vec<(String, String)> {
    let turns = match turns {
        None => return history.to_vec(),
        Some(t) => t,
    };
    if turns <= 0 {
        return Vec::new();
    }
    let opens: Vec<usize> = history.iter().enumerate()
        .filter(|(_, (who, _))| TURN_OPENS.contains(&who.as_str())).map(|(i, _)| i).collect();
    if opens.len() as i64 <= turns {
        return history.to_vec();
    }
    history[opens[opens.len() - turns as usize]..].to_vec()
}

pub struct Session<E: Eyes, H: Hands, M: Speaks> {
    pub eyes: E,
    pub hands: H,
    pub model: M,
    pub seen: Vec<Widget>,
    pub history: Vec<(String, String)>,
    pub exchanges: usize,
    pub blind: u32,
    pub vanished: u32,
    pub unknown_keys: u32,
    /// **The human is the permission.** An interface with a confirm button turns this on and gates
    /// with the button instead, which is stronger than the flag; one with no button leaves it off.
    pub may_act_unprompted: bool,
    pub remember_turns: Option<i64>,
    pub limit: usize,
}

impl<E: Eyes, H: Hands, M: Speaks> Session<E, H, M> {
    pub fn new(eyes: E, hands: H, model: M) -> Self {
        Session { eyes, hands, model, seen: Vec::new(), history: Vec::new(), exchanges: 0,
                  blind: 0, vanished: 0, unknown_keys: 0, may_act_unprompted: false,
                  remember_turns: Some(REMEMBER_TURNS), limit: MAX_LISTED }
    }

    pub fn observe(&mut self) -> (Vec<Widget>, f64) {
        let (widgets, took) = self.eyes.timed_foreground();
        self.seen = widgets.clone();
        (widgets, took)
    }

    pub fn recent(&self) -> Vec<(String, String)> {
        recent(&self.history, self.remember_turns)
    }

    pub fn knows_key(&self, name: &str) -> bool {
        match self.hands.keys() {
            None => true,
            Some(names) => names.iter().any(|n| n.to_lowercase() == name.to_lowercase()),
        }
    }

    /// The thing with this name and role, **in the world as it is now** — or nothing if it has
    /// gone. The role is half of the key; among duplicates the one nearest to where a same-named
    /// thing was last seen wins, which assumes a thing moves rather than teleporting past its own
    /// copies.
    ///
    /// **Measured 2026-09-16: that assumption does not cover a control with copies, and it cost a
    /// goal.** Character Map in advanced view has more than one button named `Open` — in basic view
    /// the single one belongs to the `Font :` combo — and asked for the font list the agent opened
    /// the character set list instead. The anchor below is found with `.find()`, *the first* match
    /// in the previous reading, so with duplicates the anchor is itself arbitrary and nearest-to-it
    /// is arbitrary too. **This disambiguates a control that moved; it cannot disambiguate a
    /// control that has copies**, and neither can the model, which is shown the same two fields.
    pub fn find(&mut self, name: &str, role: &str) -> Option<Widget> {
        let fresh = self.eyes.foreground();
        let same: Vec<&Widget> = fresh.iter()
            .filter(|w| w.name == name && (role.is_empty() || w.role == role)).collect();
        if same.is_empty() {
            return None;
        }
        if same.len() == 1 {
            return Some(same[0].clone());
        }
        //: The role appears twice in this function and the second copy masks the first: dropping it
        //: from the filter alone changes nothing while this lookup still pulls the tiebreak back.
        let was = self.seen.iter()
            .find(|w| w.name == name && (role.is_empty() || w.role == role))
            .map(|w| w.centre()).unwrap_or_else(|| same[0].centre());
        let (mut best, mut best_d) = (same[0], i64::MAX);
        for w in &same {
            let c = w.centre();
            let d = (c.0 - was.0).pow(2) + (c.1 - was.1).pow(2);
            if d < best_d {
                best_d = d;
                best = w;
            }
        }
        Some(best.clone())
    }

    /// Did the world move? `None` when the hands are not live — a dry run has nothing to confirm,
    /// and `false` would be a claim it did not.
    pub fn confirm(&mut self, widget: &Widget) -> Option<bool> {
        if !self.hands.live() {
            return None;
        }
        let after = self.eyes.foreground();
        let unchanged = after.iter().any(|w| w == widget);
        let window_changed = !after.is_empty() && after[0].window != widget.window;
        Some(window_changed || !unchanged)
    }

    pub fn decide(&mut self, trigger: &Trigger) -> Intent {
        if self.seen.is_empty() {
            self.observe();
        }
        let listed: Vec<Widget> = self.seen.iter().take(self.limit).cloned().collect();
        let prompt = render(&listed, trigger, &self.recent(), self.limit);
        let reply = self.model.ask(&prompt);
        let intent = read_reply(&reply, &listed, trigger);
        let before = intent.effects.len();
        let effects: Vec<Effect> = intent.effects.into_iter()
            .filter(|e| e.kind != "key" || self.knows_key(&e.text)).collect();
        self.unknown_keys += (before - effects.len()) as u32;
        Intent { say: intent.say, effects }
    }

    pub fn perform(&mut self, effects: &[Effect]) -> (Vec<Done>, Vec<Effect>, Vec<Effect>) {
        let (mut did, mut proposed, mut refused) = (Vec::new(), Vec::new(), Vec::new());
        for effect in effects {
            if effect.unprompted() && !self.may_act_unprompted {
                //: Not an error and not silence: it becomes something to say.
                proposed.push(effect.clone());
                continue;
            }
            match effect.kind.as_str() {
                "click" => match self.find(&effect.target, &effect.role) {
                    None => {
                        //: It was there when the question was asked and it is not now. Whatever
                        //: occupies that rectangle was never chosen.
                        self.vanished += 1;
                        refused.push(effect.clone());
                    }
                    Some(target) => {
                        self.hands.click(&target);
                        let confirmed = self.confirm(&target);
                        did.push(Done { effect: effect.clone(), target: Some(target), confirmed });
                    }
                },
                "type" => {
                    self.hands.type_text(&effect.text);
                    did.push(Done { effect: effect.clone(), target: None, confirmed: None });
                }
                "key" => {
                    self.hands.press(&effect.text);
                    did.push(Done { effect: effect.clone(), target: None, confirmed: None });
                }
                _ => refused.push(effect.clone()),
            }
        }
        (did, proposed, refused)
    }

    /// Put what happened into the conversation, beside what was said. **The past tense belongs to
    /// `did` and to nothing else**, and every other line says in words that nothing happened — so a
    /// reader that ignores the label still cannot conclude a click landed.
    ///
    /// `offered` is written first within one call, so a caller batching an offer with a settlement
    /// has the offer closed before it renders. A hazard, recorded rather than tidied.
    pub fn remember_doing(&mut self, did: &[Done], refused: &[Effect], offered: &[Effect],
                          turned_down: &[Effect]) {
        for e in offered {
            self.history.push((crate::converse::OFFER_OPEN.into(),
                format!("to {} — {NOTHING_DONE}; they have not accepted", describe(e, None, PLAIN))));
        }
        for d in did {
            let line = describe(&d.effect, d.target.as_ref(), PAST);
            self.history.push(("did".into(), line));
        }
        for e in refused {
            self.history.push(("could not".into(),
                format!("{} — it had gone by then; {NOTHING_DONE}", describe(e, None, PLAIN))));
        }
        for e in turned_down {
            self.history.push(("they said no to".into(),
                format!("{} — {NOTHING_DONE}", describe(e, None, ING))));
        }
    }

    /// One whole turn.
    ///
    /// **It feeds only `did` and `refused`.** A proposed effect is returned in the exchange and
    /// recorded nowhere, so the standing-offer slot never sees the core's own downgrade — the
    /// bookkeeping is each interface's to remember. Reproduced, not fixed.
    pub fn think(&mut self, trigger: &Trigger) -> Exchange {
        let (_widgets, look_s) = self.observe();
        let intent = self.decide(trigger);
        let (did, proposed, refused) = self.perform(&intent.effects);

        self.history.push(((if trigger.asked { "them" } else { "noticed" }).into(),
                           trigger.text.clone()));
        if !intent.say.is_empty() {
            self.history.push(("you".into(), intent.say.clone()));
        }
        self.remember_doing(&did, &refused, &[], &[]);
        if !self.seen.is_empty() && intent.effects.is_empty() {
            //: Counts a turn the unknown-key filter emptied as well — the number that would justify
            //: anything slower and cleverer, inflated by an invented keystroke. Recorded, not fixed.
            self.blind += 1;
        }
        let n = self.exchanges;
        self.exchanges += 1;
        Exchange { n, trigger: trigger.clone(), said: intent.say, did, proposed, refused,
                   n_widgets: self.seen.len(), look_s }
    }

    pub fn exchange(&mut self, heard: &str) -> Exchange {
        self.think(&Trigger::said(heard))
    }
}
