//! observe / poll / decide / perform — separately, because **acting is not a stage that always
//! runs, it is something decided.**

use crate::contract::{Eyes, Hands, Speaks, Widget};
use crate::converse::{read_reply, MAX_LISTED};
use crate::intent::{Done, Effect, Exchange, Intent, Trigger};
use crate::mind::Mind;
use crate::thinking::Note;

pub const REMEMBER_TURNS: i64 = 12;
/// How many of argmind's own actions the note carries. **Three, matching `MOST_AT_ONCE`** — the
/// longest sequence it can perform in one turn, so the note holds the last thing it did whole.
pub const DID_IN_NOTE: usize = 3;
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

/// **What a lookup found, including the answer that used to be a guess.**
///
/// Three outcomes rather than an `Option`, because *it is gone* and *there are four of them* are
/// different facts and a caller does different things about them — and collapsing two failures into
/// one `None` is pitfall 5 in `docs/findings/harness-pitfalls.md`, which cost this project a round
/// of work once already.
#[derive(Clone, Debug, PartialEq)]
pub enum Found {
    /// Not on screen any more. A listed widget is gone 13 times in 14 over the second a call takes.
    Gone,
    /// Exactly one, or several with an anchor that is a fact rather than a coin toss. See
    /// [`Session::find`].
    One(Widget),
    /// **Several, and nothing deterministic can say which.** Carried in the order the eyes gave
    /// them, which is what makes *the third one called `Open`* a designator somebody can answer
    /// with and this can re-resolve.
    Several(Vec<Widget>),
}

impl Found {
    /// The one, if there is exactly one. **Named so a caller that wants the old behaviour has to
    /// write down that it is dropping the other two cases.**
    pub fn only(self) -> Option<Widget> {
        match self {
            Found::One(w) => Some(w),
            _ => None,
        }
    }
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
    /// **How many clicks were refused for having more than one candidate.** Counted apart from
    /// `vanished` because the two are different failures: one is a world that moved, the other is a
    /// name that was never a key. Before 2026-09-28 this case was not counted at all — it was a
    /// click on whichever copy an arbitrary anchor favoured, reported as a success.
    pub ambiguous: u32,
    /// **How many answers arrived after the screen had moved on.** A person said *the second
    /// `Open`* and by the time the click ran there were no longer three of them, so the index means
    /// a different button and the effect is refused. Counted apart from [`Session::ambiguous`]
    /// because the two are different facts about whether asking is worth doing: one says nobody was
    /// asked, the other says asking is too slow here.
    pub chose_too_late: u32,
    /// Typing refused because it would have landed in a password field. See
    /// [`crate::contract::Hands::into_a_secret`].
    pub secret: u32,
    /// Typing refused because the focus was not a field — see [`Hands::typing_lands`].
    pub nowhere_to_type: u32,
    /// Actions the environment reported as not landing — see [`Hands::landed`].
    pub not_landed: u32,
    /// **Steps not attempted because one before them failed.** Counted apart: these were never
    /// tried, and a person told *the click failed* should also be told what did not follow it.
    pub unattempted: u32,
    pub unknown_keys: u32,
    /// **The human is the permission.** An interface with a confirm button turns this on and gates
    /// with the button instead, which is stronger than the flag; one with no button leaves it off.
    pub may_act_unprompted: bool,
    pub remember_turns: Option<i64>,
    pub limit: usize,
    /// What it is trying to be, from the settings. **Resting until somebody turns a dial**, and
    /// resting renders the prompt this program shipped with, byte for byte.
    pub mind: Mind,
    /// What it has worked out, which is the only thing here that argmind wrote for itself.
    /// **Blank until something revises it**, and blank renders nothing.
    pub note: Note,
}

impl<E: Eyes, H: Hands, M: Speaks> Session<E, H, M> {
    pub fn new(eyes: E, hands: H, model: M) -> Self {
        Session { eyes, hands, model, seen: Vec::new(), history: Vec::new(), exchanges: 0,
                  blind: 0, vanished: 0, ambiguous: 0, chose_too_late: 0, secret: 0, nowhere_to_type: 0,
                  not_landed: 0, unattempted: 0, unknown_keys: 0,
                  may_act_unprompted: false,
                  remember_turns: Some(REMEMBER_TURNS), limit: MAX_LISTED,
                  mind: Mind::resting(), note: Note::blank() }
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

    /// The thing with this name and role, **in the world as it is now** — or [`Found::Gone`], or
    /// [`Found::Several`] when nothing here can say which.
    ///
    /// The role is half of the key; among duplicates the one nearest to where a same-named thing was
    /// last seen wins, which assumes a thing **moves** rather than teleporting past its own copies.
    ///
    /// ## When that assumption holds, and when it was measured not to
    ///
    /// **Measured 2026-09-16: it does not cover a control with copies, and it cost a goal.**
    /// Character Map in advanced view has more than one button named `Open` — in basic view the
    /// single one belongs to the `Font :` combo — and asked for the font list the agent opened the
    /// character set list instead.
    ///
    /// The comment that recorded that already contained the rule this now follows: *the anchor is
    /// found with `.find()`, the first match in the previous reading, so with duplicates the anchor
    /// is itself arbitrary and nearest-to-it is arbitrary too.* So the tie-break is trustworthy in
    /// exactly one case — **the previous reading held exactly one** of them, which makes the anchor
    /// a fact about where that thing was and *nearest to it* a statement about where it went. With
    /// none or with several, the anchor is a guess and this now declines rather than making it.
    ///
    /// **Nothing here asks the model.** It is shown the same two fields and cannot name the one it
    /// means either; the deterministic side knows there are three buttons called `Open`, which is
    /// knowledge a person can act on. That is the shape of every other answer in this program:
    /// WHERE is not something a model supplies.
    pub fn find(&mut self, name: &str, role: &str) -> Found {
        let fresh = self.eyes.foreground();
        let same: Vec<&Widget> = fresh.iter()
            .filter(|w| w.name == name && (role.is_empty() || w.role == role)).collect();
        if same.is_empty() {
            return Found::Gone;
        }
        if same.len() == 1 {
            return Found::One(same[0].clone());
        }
        //: The role appears twice in this function and the second copy masks the first: dropping it
        //: from the filter alone changes nothing while this lookup still pulls the tiebreak back.
        let before: Vec<&Widget> = self.seen.iter()
            .filter(|w| w.name == name && (role.is_empty() || w.role == role)).collect();
        //: **One before and several now: the thing moved and something like it appeared.** The
        //: anchor is a fact. Anything else and it is a coin toss with a distance calculation on top.
        if before.len() != 1 {
            return Found::Several(same.iter().map(|w| (*w).clone()).collect());
        }
        let was = before[0].centre();
        let (mut best, mut best_d) = (same[0], i64::MAX);
        for w in &same {
            let c = w.centre();
            let d = (c.0 - was.0).pow(2) + (c.1 - was.1).pow(2);
            if d < best_d {
                best_d = d;
                best = w;
            }
        }
        Found::One(best.clone())
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
        let prompt = crate::converse::render_with(&listed, trigger, &self.recent(),
                                                  self.limit, &self.mind, &self.note);
        let (reply, tokens) = self.model.ask_scored(&prompt);
        let intent = read_reply(&reply, &listed, trigger);
        let before = intent.effects.len();
        let effects: Vec<Effect> = intent.effects.into_iter()
            .filter(|e| e.kind != "key" || self.knows_key(&e.text)).collect();
        self.unknown_keys += (before - effects.len()) as u32;
        //: How sure of the action, when there is one left after the key filter.
        let sure = if effects.is_empty() { None }
                   else { crate::converse::action_sureness(&tokens) };
        Intent { say: intent.say, effects, parroted: intent.parroted, sure }
    }

    /// **One step at a time, each checked against the screen as it is, and the rest stopped by a
    /// step that did not land.** Since 2026-10-21: the owner asked for the weather with the pane on a
    /// login page, and a click, *서울* and Enter ran one after another — the click re-read the
    /// screen, nothing checked where the text would go, and nothing stopped the steps after a step
    /// that went wrong. A click still re-resolves against a fresh reading; typing now asks whether
    /// there is a field ([`Hands::typing_lands`]); and **any refusal, or an action the hands say did
    /// not land, ends the sequence** — what follows was planned against a screen that step was
    /// supposed to produce.
    pub fn perform(&mut self, effects: &[Effect]) -> (Vec<Done>, Vec<Effect>, Vec<Effect>) {
        let (mut did, mut proposed, mut refused) = (Vec::new(), Vec::new(), Vec::new());
        let mut stopped = false;
        for effect in effects {
            if effect.unprompted() && !self.may_act_unprompted {
                //: Not an error and not silence: it becomes something to say.
                proposed.push(effect.clone());
                continue;
            }
            if stopped {
                self.unattempted += 1;
                refused.push(effect.clone());
                continue;
            }
            let before = refused.len();
            match effect.kind.as_str() {
                "click" => match self.find(&effect.target, &effect.role) {
                    Found::Gone => {
                        //: It was there when the question was asked and it is not now. Whatever
                        //: occupies that rectangle was never chosen.
                        self.vanished += 1;
                        refused.push(effect.clone());
                    }
                    //: **Refused rather than guessed**, which is the whole of the 2026-09-16 fix:
                    //: the old code pressed whichever copy the arbitrary anchor happened to favour
                    //: and reported success.
                    //:
                    //: **Unless a person said which.** `Among` carries the index *and* how many
                    //: there were when they were asked, and the count is checked first: three
                    //: buttons called `Open` can become two while somebody reads the question, and
                    //: *the second one* then means a different button. That case is counted apart
                    //: — it is not *nobody told us*, it is *somebody told us and the screen
                    //: outran them*, and the two say different things about whether asking works.
                    Found::Several(same) => match effect.which {
                        Some(a) if a.of == same.len() && a.index < same.len() => {
                            let target = same[a.index].clone();
                            self.hands.click(&target);
                            let confirmed = self.confirm(&target);
                            did.push(Done { effect: effect.clone(), target: Some(target),
                                            confirmed });
                        }
                        Some(_) => {
                            self.chose_too_late += 1;
                            refused.push(effect.clone());
                        }
                        None => {
                            self.ambiguous += 1;
                            refused.push(effect.clone());
                        }
                    },
                    Found::One(target) => {
                        self.hands.click(&target);
                        let confirmed = self.confirm(&target);
                        did.push(Done { effect: effect.clone(), target: Some(target), confirmed });
                    }
                },
                "type" => {
                    //: **A password is the person's to type.** Refused and counted apart — this
                    //: is neither *gone* nor *ambiguous*, and the person is told something else.
                    if self.hands.into_a_secret() {
                        self.secret += 1;
                        refused.push(effect.clone());
                    } else if self.hands.typing_lands() == Some(false) {
                        //: **Nowhere for it to go**: the focus is not a field.
                        self.nowhere_to_type += 1;
                        refused.push(effect.clone());
                    } else {
                        self.hands.type_text(&effect.text);
                        did.push(Done { effect: effect.clone(), target: None, confirmed: None });
                    }
                }
                "key" => {
                    self.hands.press(&effect.text);
                    did.push(Done { effect: effect.clone(), target: None, confirmed: None });
                }
                _ => refused.push(effect.clone()),
            }
            //: **The hands' own word on whether it landed** — a covered click is sent nowhere and
            //: recorded so; it is not something that happened.
            if refused.len() == before && self.hands.landed() == Some(false) {
                if let Some(d) = did.pop() {
                    refused.push(d.effect);
                }
                self.not_landed += 1;
            }
            if refused.len() > before {
                stopped = true;
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

    /// **One tick of thinking that cannot speak and cannot act.**
    ///
    /// A separate call from [`Self::decide`], asking only for the three fields of a
    /// [`Note`] — there is no `say` and no `do` in the shape it asks for, which is what makes it
    /// safe to run on a timer. Returns whether the note actually changed.
    ///
    /// **It reads the screen and does not write `seen`.** A tick must not make the next real turn
    /// think it has already looked, because a reading from thirty seconds ago is exactly the
    /// staleness the rest of this file is built to avoid.
    pub fn reflect(&mut self) -> bool {
        let widgets = self.eyes.foreground();
        let listed: Vec<Widget> = widgets.iter().take(self.limit).cloned().collect();
        let screen = crate::converse::describe_screen(&listed, self.limit);
        //: **The record, before the question.** What argmind did is a fact it already holds, and
        //: asking a model to recall it is the wrong side of this project's line — the one time it
        //: was asked, it invented having tidied somebody's browser tabs and carried that for three
        //: revisions.
        self.note.record(&crate::thinking::what_it_did(&self.history, DID_IN_NOTE));
        //: **What happened, computed here.** The deterministic side says what changed; the model
        //: is asked only whether it matters. Without it a tick could not tell a screen that had
        //: just turned over from one that had sat still for five minutes, and answered *도움을 줄
        //: 수 있다* to both.
        let changed = crate::thinking::what_changed(&self.seen, &widgets);
        let prompt = crate::thinking::ask_for_a_note(&self.note, &screen, &changed, &self.recent(),
                                                     crate::thinking::MOST_CHARS);
        let reply = self.model.ask(&prompt);
        let before = self.note.clone();
        //: **The same list the model was shown**, because `worth` is a number into it and an index
        //: resolved against a different reading is the staleness `find` exists to avoid.
        if !crate::thinking::read_note(&reply, &mut self.note, &listed, &changed) {
            return false;
        }
        //: **A revision that says the same thing is not a revision.** The count is what a
        //: measurement would use to ask whether a note settles or churns, and inflating it with
        //: no-ops would make a stuck note look busy.
        //: `tried` is not compared: it changes when argmind acts, which is not the note being
        //: reconsidered, and counting it would make a click look like a thought.
        let same = self.note.believes == before.believes && self.note.worth == before.worth;
        if same {
            //: `held` is left as `revise` set it: the belief did come round again, and that is the
            //: fact the next tick needs even when nothing else moved.
            self.note.revisions = before.revisions;
        }
        !same
    }
}
