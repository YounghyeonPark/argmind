//! A thought that survives the turn that produced it.
//!
//! The owner, 2026-09-17: **입력정보는 전부 읽고 생각을 계속 진행해야 함** — and, a few hours
//! earlier, *그냥 지켜보는것만 업데이트 하는거면 의미없고 계속 자기 생각을 반복해야함.*
//!
//! ## What was missing
//!
//! Before this file, a turn's prompt was the system sentence, twelve turns of history, the screen
//! now, an unanswered offer, and (since this morning) the dials. **Not one of those is something
//! argmind concluded.** `Session` held `seen`, `history` and four counters, and the history holds
//! *what was said*, never *what it worked out*. So the honest answer to *argmind는 어떤 생각을 함*
//! was: it does not think, it answers. Ask it the same question twice and the second answer is not
//! built on the first — it is built on a transcript of the first.
//!
//! A [`Note`] is the first thing in this program that argmind writes for itself and reads back.
//!
//! ## Three fields, and why not one
//!
//! Free text would drift into a diary. Three named slots are three questions it has to answer
//! separately, and each one fails visibly when it is wrong:
//!
//! * **believes** — what it takes to be true about what the person is doing. Wrong beliefs are
//!   readable and can be contradicted.
//! * **waiting** — what it is expecting to happen. **This is the one that makes a thought
//!   continuous**: without it there is nothing for the next tick to be about.
//! * **tried** — what it already did or already offered, so it stops offering the same thing. The
//!   re-offer defect is measured at 9 times in 23 and lives exactly here. **The model does not
//!   write this one.** See below.
//!
//! ## The model is not asked what it did
//!
//! The first run of this asked for all three and got, from a session in which argmind had done
//! precisely nothing: *이미 메모리 사용량이 높은 탭들을 목록으로 정리해두고, 사용자가 필요할 때
//! 쉽게 접근할 수 있도록 했다.* It had not. It carried that invention forward for three revisions.
//!
//! **A fabricated `tried` is worse than an empty one** — the field exists to stop argmind offering
//! the same thing twice, so a false entry suppresses an offer that was never made.
//!
//! And it is the wrong side of this project's line. *The deterministic side says WHERE, the
//! probabilistic side says WHAT.* **What argmind did is not a judgement, it is a record**, and the
//! program holds it: `did`, `offered`, `could not` and `they said no to` are written into the
//! history by [`crate::session::Session::remember_doing`]. So [`what_it_did`] reads them, and the
//! model is asked for two fields rather than three.
//!
//! ## The rule it inherits
//!
//! **An empty note renders nothing, and the prompt is then byte-for-byte what it was.** The same
//! guarantee [`crate::mind`] carries and for the same reason: prompt edits here are 0 for 5, and
//! the two that paid were slots that were absent until there was something to put in them.
//!
//! ## What it is deliberately not
//!
//! Not a memory across runs — it dies with the process, because a belief about what somebody was
//! doing yesterday is worse than no belief. Not unbounded — every field is capped, since a note
//! that grows is a prompt that grows until the model stops reading it. And **not written by the
//! same call that answers**: `say` and `do` are what a person is owed promptly, and a model asked
//! for a third thing in the same breath does all three worse.

/// The longest any one field may be, in characters.
///
/// **Short enough to have to choose.** A note is a working memory, not a log; the failure mode of
/// a generous limit is that the model appends instead of revising, and then every prompt carries
/// everything it has ever believed.
pub const MOST_CHARS: usize = 240;

/// The header the note renders under, when it renders at all.
///
/// **Korean, like the rest of what a person reads here** — and by the same measurement that
/// decided the dials: the language of an added block moves the language of the reply, 28 of 44
/// to 42 with a Korean one and down to 19 with an English one. These are also the exact three
/// labels the panel shows, so the page and the prompt cannot come to word it differently.
pub const THINKING: &str = "지금까지 네가 생각하고 있던 것:";

/// What argmind currently holds to be the case. **Empty is the state this program shipped in.**
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Note {
    /// What it takes to be true about what the person is doing.
    pub believes: String,
    /// What it is expecting — the field that makes the next tick have a subject.
    pub waiting: String,
    /// What it has already done or offered, so it does not offer it again.
    ///
    /// **Written from the record, never by the model.** See the note at the top of this module and
    /// [`what_it_did`].
    pub tried: String,
    /// How many times this note has been revised. **Not rendered** — it is for the interface and
    /// for a measurement asking whether a note settles or churns.
    pub revisions: u64,
}

impl Note {
    pub fn blank() -> Self {
        Note::default()
    }

    /// Is there nothing here? Then nothing goes in the prompt.
    pub fn empty(&self) -> bool {
        self.believes.trim().is_empty()
            && self.waiting.trim().is_empty()
            && self.tried.trim().is_empty()
    }

    /// Replace the two fields the model writes, **trimmed and capped, and counted as one
    /// revision.** `tried` is not among them — see [`Note::record`].
    ///
    /// Revising rather than appending is the whole design: the model is handed the previous note
    /// and asked for the note that should replace it.
    pub fn revise(&mut self, believes: &str, waiting: &str) {
        self.believes = clipped(believes);
        self.waiting = clipped(waiting);
        self.revisions += 1;
    }

    /// Set `tried` from what actually happened. **Not a revision** — nothing was reconsidered.
    pub fn record(&mut self, tried: &str) {
        self.tried = clipped(tried);
    }

    /// The block that goes into the prompt, or **`None` when there is nothing to say.**
    ///
    /// Only the fields that have something in them, each on its own line, because a rendered
    /// `waiting:` with nothing after it teaches the model that empty fields are acceptable output.
    pub fn block(&self) -> Option<String> {
        if self.empty() {
            return None;
        }
        let mut lines = vec![THINKING.to_string()];
        for (label, text) in [("믿는 것", &self.believes),
                              ("기다리는 것", &self.waiting),
                              ("이미 한 것", &self.tried)] {
            let text = text.trim();
            if !text.is_empty() {
                lines.push(format!("  {label}: {text}"));
            }
        }
        Some(lines.join("\n"))
    }
}

fn clipped(s: &str) -> String {
    s.trim().chars().take(MOST_CHARS).collect()
}

/// **The one question that asks for a note and for nothing else.**
///
/// A separate call from the one that answers, on purpose. `converse::render` asks for `say` and
/// `do`, which are what a person is owed promptly; a model asked for a third thing in the same
/// breath does all three worse, and this project has the reply-shape measurement to say that
/// whichever field is written first is written without the others.
///
/// **It cannot speak and it cannot act.** There is no `say` and no `do` in the shape it is asked
/// for, so a tick of continuous thinking has no path to the screen or to the speaker — which is
/// what makes it safe to run on a timer.
pub fn ask_for_a_note(note: &Note, screen: &str, history: &[(String, String)],
                      limit: usize) -> String {
    let mut lines = vec![
        //: **Korean, and that is measured rather than chosen.** The language of an added
        //: block decides the language of the reply — 44 Korean requests, a content-free
        //: English sentence in this position took Korean answers from 28 of 44 down to 19,
        //: and a Korean one took them up to 42. A note is read by the person in a panel with
        //: Korean labels *and* goes back into a prompt that has to come out in their
        //: language, so both halves point the same way. See `crate::mind`.
        "너는 사람이 PC에서 일하는 걸 옆에서 보고 있는 도우미다. 지금 아무도 너에게 말을 걸지 않았다."
            .to_string(),
        String::new(),
        "네 메모를 고쳐 써라. 이건 대답이 아니고 아무도 대답으로 읽지 않는다 \
         — 다음에 그 사람이 말을 걸었을 때 네가 들고 있을 생각이다.".to_string(),
    ];
    if let Some(block) = note.block() {
        lines.push(String::new());
        lines.push(block);
    }
    if !history.is_empty() {
        lines.push(String::new());
        lines.push("최근에:".to_string());
        for (who, text) in history {
            lines.push(format!("  {who}: {text}"));
        }
    }
    lines.push(String::new());
    lines.push("지금 화면에 있는 것:".to_string());
    lines.push(if screen.is_empty() {
        "  (이름이 붙은 것이 없음 — 이 창은 위젯을 내놓지 않는다)".to_string()
    } else {
        screen.to_string()
    });
    lines.push(String::new());
    //: **Revise, not append**, said in the instruction as well as enforced by the shape: three
    //: fields come back and three fields replace what was there.
    lines.push(
        r#"JSON만으로 답하라: {"believes": "...", "waiting": "..."} — 앞의 메모에"#
            .to_string());
    lines.push(
        "덧붙이지 말고 갈아치워라. 각각 짧은 한 문장까지, 쓸 게 없으면 빈 문자열로 두어라. \
         바뀜 게 없으면 있던 걸 그대로 다시 써라.".to_string());
    //: **`waiting` was the field that did not work.** In the first run it paraphrased the
    //: belief with *may* — *User may proceed with the branch renaming* — which is not an
    //: expectation and gives the next tick nothing to be about. So it is asked for by name.
    lines.push(
        "believes 는 지금 무슨 일이 벌어지고 있다고 보는지. waiting 은 네가 실제로 일어나기를 \
         기다리는 구체적인 사건 — 없으면 비워라, 짐작을 적지 마라.".to_string());
    //: **And it is told not to claim it did anything**, because the one time it was asked it
    //: invented having tidied the person's browser tabs. What it did is shown to it above,
    //: from the record, and it is not asked to add to that.
    lines.push(
        "네가 무엇을 했는지는 쓰지 마라. 하지 않은 일을 했다고 적는 것보다 아무것도 안 적는 게 \
         낫다.".to_string());
    //: **Said rather than hoped for.** A note that carried an old belief into new evidence was
    //: the first thing that went wrong: four revisions in, a file manager on `C:` was still
    //: being read as *exploring files related to Younghyeon Park*.
    lines.push(
        "화면이 바뀌었으면 앞의 메모를 이어붙이지 말고 버려라. 지금 보이는 것과 맞지 않는 \
         믿음은 틀린 믿음이다.".to_string());
    lines.push(format!("한 항목에 {limit}자를 넘기지 마라."));
    lines.join("\n")
}

/// What argmind actually did, off the record it keeps anyway.
///
/// The four labels `Session::remember_doing` writes, newest last, most recent few. **Not the
/// model's opinion of what it did** — see the note at the top of this module for the invention that
/// made this a function rather than a field.
pub fn what_it_did(history: &[(String, String)], most: usize) -> String {
    let done: Vec<&str> = history.iter()
        .filter(|(who, _)| matches!(who.as_str(),
                                    "did" | crate::converse::OFFER_OPEN | "could not"
                                    | "they said no to"))
        .map(|(_, text)| text.as_str())
        .collect();
    let from = done.len().saturating_sub(most);
    done[from..].join("; ")
}

/// Read a note back. **Anything unreadable leaves the note exactly as it was** — a tick that
/// produced nothing usable must not erase what the last one worked out.
///
/// Two fields, not three: `tried` is a record and [`what_it_did`] writes it.
pub fn read_note(reply: &str, into: &mut Note) -> bool {
    let Some(got) = crate::llm::first_object(reply) else { return false };
    let field = |k: &str| got.get(k).and_then(|v| v.as_str()).unwrap_or("").to_string();
    let (believes, waiting) = (field("believes"), field("waiting"));
    //: **Both empty is not an update.** The model saying nothing at all is the ordinary outcome of
    //: a quiet minute, and it should leave the previous note standing rather than quietly clearing
    //: it.
    if believes.trim().is_empty() && waiting.trim().is_empty() {
        return false;
    }
    into.revise(&believes, &waiting);
    true
}
