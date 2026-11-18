//! observe / poll / decide / perform — separately, because **acting is not a stage that always
//! runs, it is something decided.**

use crate::contract::{Eyes, Hands, Speaks, Widget};
use crate::converse::{read_reply, MAX_LISTED};
use crate::intent::{Done, Effect, Exchange, Intent, Trigger};
use crate::mind::Mind;
use crate::thinking::Note;

pub const REMEMBER_TURNS: i64 = 12;
/// **Tokens kept free for the reply** when the history is fitted to the context. Replies are tens
/// of tokens; a thousand leaves room for the long ones without letting the prompt and the reply
/// together overrun the context and have the server cut the instructions off the front.
pub const ANSWER_ROOM: usize = 1024;
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
        //: **A point inside, said as where it was and what the picture showed** — never the point.
        //: The conversation's model reads this in its history, and it is never shown a coordinate.
        if effect.inside.is_some() {
            let showed = if effect.text.is_empty() { String::new() }
                         else { format!(", where the picture showed \"{}\"", effect.text) };
            return format!("{v} inside the {role} \"{}\"{showed}", effect.target);
        }
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
/// **Tokens a text will cost, overestimated on purpose.** Measured on qwen3:8b's tokenizer on
/// 2026-10-21: Korean 0.59–0.66 tokens a character across two samples, English 0.31, a mix 0.48.
/// Fitted, Hangul costs up to 0.9 and everything else 0.31; this charges 1.0 and 0.4, which
/// overestimated every sample by 15–45%. **Over is the safe side**: a prompt longer than the
/// context is cut by the server from the front, and the front is the instructions.
pub fn estimate_tokens(text: &str) -> usize {
    let mut hangul = 0usize;
    let mut other = 0usize;
    for c in text.chars() {
        if ('\u{AC00}'..='\u{D7A3}').contains(&c) || ('\u{3130}'..='\u{318F}').contains(&c) {
            hangul += 1;
        } else {
            other += 1;
        }
    }
    hangul + (other * 2).div_ceil(5)
}

/// **As much of the conversation as fits, in whole turns, newest first.** The budget is tokens, by
/// [`estimate_tokens`], counting each line as the prompt renders it. A turn that does not fit is
/// dropped whole and so is everything older — the same rule as [`recent`], which a line-based cut
/// broke once by opening on a reply whose question had gone.
pub fn recent_within(history: &[(String, String)], budget: usize) -> Vec<(String, String)> {
    let opens: Vec<usize> = history.iter().enumerate()
        .filter(|(_, (who, _))| TURN_OPENS.contains(&who.as_str())).map(|(i, _)| i).collect();
    let cost = |lines: &[(String, String)]| -> usize {
        lines.iter().map(|(who, text)| estimate_tokens(&format!("  {who}: {text}\n"))).sum()
    };
    let mut from = history.len();
    let mut spent = 0usize;
    //: Lines after the last opener belong to the newest turn too.
    let mut ends: Vec<usize> = opens.clone();
    ends.reverse();
    let mut upper = history.len();
    for start in ends {
        let turn = cost(&history[start..upper]);
        if spent + turn > budget {
            break;
        }
        spent += turn;
        from = start;
        upper = start;
    }
    history[from..].to_vec()
}

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
    /// **Points inside a widget the hands could not press** — see [`Hands::click_inside`], which
    /// says no unless an environment has said it can.
    pub cannot_point: u32,
    pub unknown_keys: u32,
    /// **The human is the permission.** An interface with a confirm button turns this on and gates
    /// with the button instead, which is stronger than the flag; one with no button leaves it off.
    pub may_act_unprompted: bool,
    pub remember_turns: Option<i64>,
    pub limit: usize,
    /// **The model's context, in tokens, when the history should fill it** rather than stop at
    /// [`REMEMBER_TURNS`]. `None` is the turn count, as every harness here measured.
    pub context_tokens: Option<usize>,
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
                  not_landed: 0, unattempted: 0, cannot_point: 0, unknown_keys: 0,
                  may_act_unprompted: false,
                  remember_turns: Some(REMEMBER_TURNS), limit: MAX_LISTED, context_tokens: None,
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

    /// **The history this prompt can carry**: by turns when no context size is set — today's
    /// behaviour, which every measurement here was taken under — and otherwise as much as fits
    /// beside `without_history`, the rest of the prompt, leaving [`ANSWER_ROOM`] for the reply.
    pub fn recent_for(&self, without_history: &str) -> Vec<(String, String)> {
        match self.context_tokens {
            None => self.recent(),
            Some(ctx) => {
                //: The heading the history adds, and the blank line after it.
                let fixed = estimate_tokens(without_history) + estimate_tokens("Recently:\n\n");
                recent_within(&self.history, ctx.saturating_sub(ANSWER_ROOM + fixed))
            }
        }
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
        //: **Rendered once without the history to learn what is left for it**, when the history is
        //: fitted to the context; by turns otherwise, exactly as before.
        let history = match self.context_tokens {
            None => self.recent(),
            Some(_) => {
                let bare = crate::converse::render_with(&listed, trigger, &[], self.limit,
                                                        &self.mind, &self.note);
                self.recent_for(&bare)
            }
        };
        let prompt = crate::converse::render_with(&listed, trigger, &history,
                                                  self.limit, &self.mind, &self.note);
        let (reply, tokens) = self.model.ask_scored(&prompt);
        let mut intent = read_reply(&reply, &listed, trigger);
        //: **A reply in a language the person did not set is written again, once** — since
        //: 2026-11-17. Asked for Japanese, this model answered Korean turns in Korean or in Korean
        //: grammar around a Japanese word four times in twelve, and no line in the prompt moved it;
        //: a second call asked only to rewrite does. Only that reply pays for it, its action is the
        //: first call's, and a rewrite that is not in the language either is not taken.
        let code = self.mind.language.clone();
        if !intent.parroted && !intent.say.is_empty()
            && !crate::converse::in_language(&intent.say, &listed, &code) {
            if let Some(ask) = crate::converse::rewrite_prompt(&intent.say, &code) {
                let again = self.model.ask(&ask);
                if let Some(said) = crate::converse::read_rewrite(&again, &intent.say, &listed, &code) {
                    intent.rewritten_from = Some(std::mem::replace(&mut intent.say, said));
                }
            }
        }
        let before = intent.effects.len();
        let effects: Vec<Effect> = intent.effects.into_iter()
            .filter(|e| e.kind != "key" || self.knows_key(&e.text)).collect();
        self.unknown_keys += (before - effects.len()) as u32;
        //: How sure of the action, when there is one left after the key filter.
        let sure = if effects.is_empty() { None }
                   else { crate::converse::action_sureness(&tokens) };
        Intent { say: intent.say, effects, parroted: intent.parroted, sure,
                 rewritten_from: intent.rewritten_from }
    }

    /// **Words argmind's other paths wrote, said in `code`** — since 2026-11-17: a task's closing
    /// words, the question it asks, an errand's result, the request argmind set itself. The work
    /// model writes them from Korean prompts, so somebody spoken to in English heard Korean from
    /// every one; this is the rewrite [`Session::decide`] makes for a turn, for text that is not a
    /// turn. **Not called by the application** — `--bin tongues --reports` found it changing what
    /// was found (a price in the wrong currency, an airline renamed) in six texts of 36. Unchanged
    /// when it is already in `code`, when `code` is no language, or when the rewrite is not taken —
    /// **and not taken when it loses a fact**
    /// ([`crate::converse::keeps_the_facts`]): every number and every file name the words held must
    /// still be in them, since these words report work and a rewrite that drops *369,700* or
    /// `ideas.md` has changed what was done.
    pub fn say_in(&mut self, text: &str, code: &str) -> String {
        crate::converse::say_in(&mut self.model, text, code)
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
                            match self.press(effect, same[a.index].clone()) {
                                Some(done) => did.push(done),
                                None => refused.push(effect.clone()),
                            }
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
                    Found::One(target) => match self.press(effect, target) {
                        Some(done) => did.push(done),
                        None => refused.push(effect.clone()),
                    },
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

    /// **One press on a target already resolved** — by its own means, or, for an effect carrying
    /// [`crate::intent::Inside`], at that point in the target's rectangle *as it is now*. `None`
    /// when the hands cannot press a point inside anything, counted in `cannot_point`.
    fn press(&mut self, effect: &Effect, target: Widget) -> Option<Done> {
        let Some(inside) = effect.inside else {
            self.hands.click(&target);
            let confirmed = self.confirm(&target);
            return Some(Done { effect: effect.clone(), target: Some(target), confirmed });
        };
        let at = inside.within(target.rect);
        if !self.hands.click_inside(&target, at) {
            self.cannot_point += 1;
            return None;
        }
        //: **Not confirmed either way.** The list cannot see into this place — that is why a point
        //: was needed — so whether anything inside it moved is not the list's to say, and `false`
        //: would be a claim it did not.
        Some(Done { effect: effect.clone(), target: Some(target), confirmed: None })
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

/// **One line of the conversation file**: when, who, what. JSON, one object a line, so a line that
/// does not read loses only itself.
pub fn history_line(at: u64, who: &str, text: &str) -> String {
    serde_json::json!({ "at": at, "who": who, "text": text }).to_string()
}

/// The conversation back out of the file, oldest first, the last `most` lines.
pub fn history_from(file: &str, most: usize) -> Vec<(String, String)> {
    let all: Vec<(String, String)> = file.lines().filter_map(|l| {
        let v: serde_json::Value = serde_json::from_str(l).ok()?;
        Some((v["who"].as_str()?.to_string(), v["text"].as_str()?.to_string()))
    }).collect();
    let skip = all.len().saturating_sub(most);
    all[skip..].to_vec()
}
