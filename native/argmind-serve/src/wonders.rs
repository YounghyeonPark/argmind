//! **Things argmind does not know and can ask about.**
//!
//! The owner, 2026-09-23: *모르는건 물어보는 기능도 있어야함.* The immediate case is a face it has
//! just seen for the first time — writing an unnamed row into a settings panel and waiting for
//! somebody to go looking for it is not asking, it is filing.
//!
//! ## What makes this different from everything else that appears in the page
//!
//! **Nobody's model decided to ask.** A `Proposal` comes out of `converse()`; a question here comes
//! out of the deterministic side noticing a gap it can name — a row with no name in it. The text is
//! a constant. That matters because prompt edits in this repository are **1 for 7**, and a feature
//! that adds none carries none of that risk.
//!
//! It is also why this can be trusted to be *right about not knowing*: `people.rs` either has a
//! name for that key or it does not, and no inference is involved.
//!
//! ## The three rules, and each one is a failure this project has already had
//!
//! **One at a time.** [`Wonders`] holds at most one open question. A column of them is the pile of
//! confirm buttons the owner pointed at on 2026-09-22, arriving in a different part of the program.
//!
//! **Asked once.** Answering *or* dismissing marks the subject asked, and it is never raised again.
//! The judgement path learned this the hard way: a thing that stands un-acted-on and keeps being
//! raised is noise, and `RAISE_AFTER` exists because of it. Here the rule is stricter — one ask,
//! ever — because a name is not an opportunity that recurs.
//!
//! **It does not speak.** *The room stays quiet unless somebody asked* allows exactly two `say`
//! calls in `app.rs` and a rule in `argmind-rules` holds it. A question appears in the page and
//! makes no sound, which is also the right behaviour: the camera is on while somebody works, and a
//! voice asking who just walked past would be worse than silence.
//!
//! ## What it is not
//!
//! **It is not the general thing that sentence could mean.** *Ask when you do not know* would also
//! cover the measured case in `clicking`, where two buttons were both called `Open` and argmind
//! picked one — `name` and `role` together are not a key, and asking *which Open?* is the honest
//! answer there. That needs the model to express uncertainty about a choice, which is a prompt
//! change and a measurement. This is the half that needs neither.

/// What is not known.
#[derive(Clone, Debug, PartialEq)]
pub enum About {
    /// A face with a row and no name. Carries the person's key.
    Who(String),
    /// **Several things share the name and role that were chosen**, and no deterministic rule can
    /// say which was meant — see `Session::find`. Measured 2026-09-16: asked for the font list in
    /// Character Map's advanced view, argmind opened the character set list, because `Open` names
    /// more than one button there.
    ///
    /// Carries the proposal's `token`, so answering *is* the confirmation and the action still goes
    /// through `Proposals` and the one call site that performs anything; and `among`, one line per
    /// candidate **in the order the eyes gave them**, which is what makes the answer an index this
    /// can re-resolve against.
    WhichOne { token: u64, what: String, among: Vec<String> },
    /// **A task wants to press or write on argmind's work screen** and the person's 규칙 says
    /// *항상 물어봄* there, or 확인 없이 실행 is off. A press there is not a `Proposal` — it never
    /// reaches `perform`, the work screen is not the person's — so the answer goes back to the
    /// task thread waiting on `ask`, and only the one button means yes; skipping is no.
    Press { ask: u64, what: String, window: String, among: Vec<String> },
}

impl About {
    /// What the person reads. **Built from the situation, never from a model** — see the module
    /// note: the deterministic side counted the buttons, so no prompt was edited to make this
    /// sentence exist and none of the 1-for-7 risk is being taken.
    pub fn asked(&self) -> String {
        match self {
            About::Who(_) => "처음 보는 분이네요. 누구신가요?".into(),
            About::WhichOne { what, among, .. } =>
                format!("{what} 이(가) {}개 있습니다. 어느 것입니까?", among.len()),
            About::Press { what, window, .. } =>
                format!("작업 화면의 「{window}」에서 {what} — 할까요?"),
        }
    }

    /// What goes in the box before anything is typed. Empty when the answer is a choice rather
    /// than a sentence.
    pub fn hint(&self) -> &'static str {
        match self {
            About::Who(_) => "이름 — 비워 두고 넘어가도 됩니다",
            About::WhichOne { .. } | About::Press { .. } => "",
        }
    }

    /// The candidates, when there are any. **A page renders one button each**; a question with a
    /// text box and a question with buttons are the same object here because the *shape* of not
    /// knowing is the same and only the answer differs.
    pub fn among(&self) -> &[String] {
        match self {
            About::Who(_) => &[],
            About::WhichOne { among, .. } | About::Press { among, .. } => among,
        }
    }
}

/// One open question.
#[derive(Clone, Debug, PartialEq)]
pub struct Wonder {
    /// **Counts up and is never reused**, the same discipline `Proposals` uses: an answer to a
    /// question that has already gone must not land on the next one.
    pub id: u64,
    pub about: About,
}

/// At most one question, and a memory of what has been asked.
#[derive(Debug, Default)]
pub struct Wonders {
    open: Option<Wonder>,
    next: u64,
}

/// What happened to an answer.
#[derive(Clone, Debug, PartialEq)]
pub enum Answered {
    /// Answered with this text, about this. Empty text means *skipped*.
    Say(About, String),
    /// There is no question by that id — it was answered already, or it is from before a restart.
    Gone,
}

impl Wonders {
    /// Raise one, unless something is already open. **Returns `None` rather than queueing**: a
    /// question nobody answered is not a reason to stack a second one behind it.
    pub fn ask(&mut self, about: About) -> Option<&Wonder> {
        if self.open.is_some() {
            return None;
        }
        self.next += 1;
        self.open = Some(Wonder { id: self.next, about });
        self.open.as_ref()
    }

    pub fn open(&self) -> Option<&Wonder> {
        self.open.as_ref()
    }

    /// Answer or skip. **Trimmed, and an empty answer is a skip** — the same act as pressing the
    /// button beside the box, because somebody who clears the field and presses enter has said the
    /// same thing as somebody who pressed 넘어가기.
    pub fn answered(&mut self, id: u64, text: &str) -> Answered {
        match self.open.take_if(|w| w.id == id) {
            Some(w) => Answered::Say(w.about, text.trim().chars().take(40).collect()),
            None => Answered::Gone,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_at_a_time_and_the_second_is_dropped_rather_than_queued() {
        let mut w = Wonders::default();
        assert!(w.ask(About::Who("p1".into())).is_some());
        assert!(w.ask(About::Who("p2".into())).is_none(),
                "a second question must not stack behind the first");
        assert_eq!(w.open().unwrap().about, About::Who("p1".into()));
    }

    #[test]
    fn answering_frees_the_slot_and_says_what_it_was_about() {
        let mut w = Wonders::default();
        let id = w.ask(About::Who("p1".into())).unwrap().id;
        assert_eq!(w.answered(id, "  영현 "), Answered::Say(About::Who("p1".into()), "영현".into()));
        assert!(w.open().is_none());
        assert!(w.ask(About::Who("p2".into())).is_some(), "the slot is free again");
    }

    #[test]
    fn an_empty_answer_is_a_skip_and_still_resolves_it() {
        let mut w = Wonders::default();
        let id = w.ask(About::Who("p1".into())).unwrap().id;
        assert_eq!(w.answered(id, "   "), Answered::Say(About::Who("p1".into()), String::new()));
        assert!(w.open().is_none());
    }

    #[test]
    fn an_id_is_never_reused() {
        let mut w = Wonders::default();
        let first = w.ask(About::Who("p1".into())).unwrap().id;
        w.answered(first, "");
        let second = w.ask(About::Who("p2".into())).unwrap().id;
        assert_ne!(first, second);
        //: **And the old id does not land on the new question.** This is the defect `Proposals`
        //: was built to make structurally impossible; the same one arrives here for free only if
        //: somebody writes this test.
        assert_eq!(w.answered(first, "아무개"), Answered::Gone);
        assert_eq!(w.open().unwrap().id, second);
    }

    #[test]
    fn a_stale_answer_changes_nothing() {
        let mut w = Wonders::default();
        assert_eq!(w.answered(99, "누구"), Answered::Gone);
        let id = w.ask(About::Who("p1".into())).unwrap().id;
        assert_eq!(w.answered(id + 7, "누구"), Answered::Gone);
        assert!(w.open().is_some(), "an answer to nothing must not close the real question");
    }

    #[test]
    fn a_very_long_answer_is_cut_rather_than_refused() {
        //: The route length-checks first; this is the second line, and it matches what
        //: `People::call` will keep so the two cannot disagree about what was stored.
        let mut w = Wonders::default();
        let id = w.ask(About::Who("p1".into())).unwrap().id;
        let Answered::Say(_, name) = w.answered(id, &"가".repeat(200)) else { panic!() };
        assert_eq!(name.chars().count(), 40);
    }

    #[test]
    fn what_is_read_is_a_question_and_the_box_says_it_may_be_skipped() {
        let about = About::Who("p1".into());
        assert!(about.asked().ends_with('?'));
        assert!(about.hint().contains("넘어가"),
                "somebody must be able to see that not answering is allowed");
        assert!(about.among().is_empty(), "a name is typed, not chosen");
    }

    fn which(n: usize) -> About {
        About::WhichOne { token: 7, what: "Open 버튼".into(),
                          among: (0..n).map(|i| format!("{}번째", i + 1)).collect() }
    }

    #[test]
    fn a_choice_asks_how_many_there_are_and_offers_each() {
        let about = which(3);
        assert!(about.asked().contains('3'), "{}", about.asked());
        assert!(about.asked().ends_with('?'));
        assert_eq!(about.among().len(), 3);
        //: **No text box.** A free-text answer to *which one* would be a second way to say an
        //: index and a second way to get it wrong.
        assert!(about.hint().is_empty());
    }

    #[test]
    fn a_choice_is_skippable_like_every_other_question() {
        //: Nothing here is asked twice, so a question with no way out is one somebody has to
        //: answer to make it go away.
        let mut w = Wonders::default();
        let id = w.ask(which(2)).unwrap().id;
        assert_eq!(w.answered(id, ""), Answered::Say(which(2), String::new()));
        assert!(w.open().is_none());
    }

    #[test]
    fn a_choice_and_a_name_cannot_stack() {
        //: One at a time is the rule and it does not care which kind. A face question arriving
        //: while somebody is deciding which `Open` would be two things to read and one slot.
        let mut w = Wonders::default();
        assert!(w.ask(which(2)).is_some());
        assert!(w.ask(About::Who("p1".into())).is_none());
    }
}
