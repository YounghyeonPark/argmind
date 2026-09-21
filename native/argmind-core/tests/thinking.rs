//! **A thought that survives the turn that produced it — and costs nothing until there is one.**
//!
//! The owner, 2026-09-17: 입력정보는 전부 읽고 생각을 계속 진행해야 함. Before this, nothing
//! argmind concluded outlived a turn: the prompt carried the system sentence, the history, the
//! screen and an unanswered offer, and the history holds *what was said*, never *what it worked
//! out*.
//!
//! Two things are worth protecting here and they pull in opposite directions. A note has to
//! actually reach the model, or the whole thing is a field nobody reads; and an empty one has to
//! reach nothing at all, or every prompt in the program changes on the day this shipped.

use argmind_core::contract::Widget;
use argmind_core::converse::{render, render_with, MAX_LISTED};
use argmind_core::intent::Trigger;
use argmind_core::mind::Mind;
use argmind_core::thinking::{ask_for_a_note, read_note, worth_of, Judged,
                             Note, Worth,
                             MOST_CHARS};

fn screen() -> Vec<Widget> {
    vec![Widget::new("one", "button", (0, 0, 1, 1), ""),
         Widget::new("two", "combo box", (0, 0, 1, 1), "")]
}

fn prompt(note: &Note) -> String {
    render_with(&screen(), &Trigger::said("probe"), &[], MAX_LISTED, &Mind::resting(), note)
}

/// **A blank note is the prompt this program shipped with, to the byte.**
///
/// The same guarantee `mind.rs` carries. `tests/equivalence.rs` holds the outside half — the six
/// probes still hashing to `r322a4c` — and this holds the inside one, that the new parameter is
/// what makes no difference rather than the new default.
#[test]
fn a_blank_note_puts_nothing_in_the_prompt() {
    let blank = Note::blank();
    assert!(blank.empty());
    assert_eq!(blank.block(), None);
    assert_eq!(prompt(&blank),
               render(&screen(), &Trigger::said("probe"), &[], MAX_LISTED, &Mind::resting()));

    //: Whitespace in every field is still blank — otherwise a model that answers with three spaces
    //: turns the slot on for the rest of the session.
    let mut spaces = Note::blank();
    spaces.revise("  ", Judged::Nothing);
    spaces.record(" \t ");
    assert!(spaces.empty());
    assert_eq!(prompt(&spaces), prompt(&Note::blank()));
}

/// **And a note that says something reaches the model.** The other half: a slot nobody can see the
/// effect of is a field that quietly stops being rendered.
#[test]
fn every_field_of_a_note_reaches_the_prompt() {
    let mut note = Note::blank();
    note.revise("그가 코드 리뷰 중이다",
                Judged::This(Worth { name: "설정".into(), role: "button".into() }));
    //: `tried` comes from the record, never from the model — so it is set the way the session
    //: sets it.
    note.record("clicked the button \"설정\"");
    let block = note.block().expect("a note with three fields renders nothing");
    for text in ["그가 코드 리뷰 중이다", "button \"설정\"",
                 "clicked the button \"설정\""] {
        assert!(block.contains(text), "{text} is missing from {block}");
    }
    let p = prompt(&note);
    assert!(p.contains(&block), "the block is not in the prompt");
    //: Above the screen, not below it: a slot behind two hundred widget names is a slot nobody
    //: reads. Placement is unargued; that it is *before* is at least checked.
    assert!(p.find(&block) < p.find("On screen now:"), "the note sank below the screen");
}

/// An empty field is **absent**, not rendered with nothing after it — which would teach the model
/// that an empty field is acceptable output.
#[test]
fn a_field_with_nothing_in_it_is_not_printed() {
    let mut note = Note::blank();
    note.revise("그가 문서를 읽고 있다", Judged::Nothing);
    let block = note.block().unwrap();
    assert!(block.contains("믿는 것"), "{block}");
    assert!(!block.contains("할 만한 것"), "{block}");
    assert!(!block.contains("이미 한 것"), "{block}");
}

/// **Revising replaces; it never appends.** The failure this is guarding is a note that grows every
/// tick until the prompt is mostly history of itself.
#[test]
fn revising_replaces_what_was_there_and_counts_once() {
    let mut note = Note::blank();
    note.revise("첫 번째", Judged::Nothing);
    assert_eq!(note.revisions, 1);
    note.revise("두 번째", Judged::Nothing);
    assert_eq!(note.revisions, 2);
    assert_eq!(note.believes, "두 번째");
    assert!(!note.block().unwrap().contains("첫 번째"));

    //: **Capped.** A field longer than the limit is cut rather than carried, because a prompt that
    //: grows without bound is a prompt the model stops reading.
    note.revise(&"가".repeat(MOST_CHARS * 3), Judged::Nothing);
    assert_eq!(note.believes.chars().count(), MOST_CHARS);
}

/// **A tick that produced nothing usable leaves the note exactly as it was.**
///
/// The dangerous direction: a quiet minute is the ordinary case, and a model answering with three
/// empty strings must not erase what the last tick worked out.
#[test]
fn an_unusable_reply_does_not_clear_what_was_already_there() {
    let mut note = Note::blank();
    note.revise("그가 회의 중이다", Judged::Nothing);
    let before = note.clone();

    for reply in [r#"{"believes": "", "worth": null}"#,
                  "not json at all",
                  "",
                  r#"{"something": "else"}"#] {
        assert!(!read_note(reply, &mut note, &screen(), "나타남: Eraser"), "{reply:?} was accepted");
        assert_eq!(note, before, "{reply:?} changed the note");
    }

    //: And a reply with one real field is an update, with the other two cleared — because the model
    //: was asked to replace the note, and a partial merge would make *I no longer believe that*
    //: impossible to say.
    assert!(read_note(r#"{"believes": "그가 돌아왔다", "worth": null}"#, &mut note, &screen(), "나타남: Eraser"));
    assert_eq!(note.believes, "그가 돌아왔다");
    assert_eq!(note.worth, None);
    assert_eq!(note.revisions, before.revisions + 1);
}

/// **The thinking call cannot speak and cannot act**, and that is what makes it safe on a timer.
#[test]
fn the_thinking_question_asks_for_no_speech_and_no_action() {
    let mut note = Note::blank();
    note.revise("그가 코드를 보고 있다", Judged::Nothing);
    let asked = ask_for_a_note(&note, "0: button \"one\"", "", &[], MOST_CHARS);

    //: The two fields it may answer with, and nothing else.
    assert!(asked.contains("\"believes\""), "{asked}");
    assert!(asked.contains("\"worth\""), "{asked}");
    //: **`tried` is not among them.** It is a record the program keeps, and the one time the
    //: model was asked for it, it invented having tidied somebody's browser tabs.
    assert!(!asked.contains("\"tried\""), "the model was asked what it did: {asked}");
    //: **No `say` and no `do`.** There is no shape in which this call reaches the screen or the
    //: speaker, which is the property a periodic call has to have.
    assert!(!asked.contains("\"say\""), "a thinking tick was offered a way to speak: {asked}");
    assert!(!asked.contains("\"do\""), "a thinking tick was offered a way to act: {asked}");
    assert!(!asked.contains("{\"click\""), "{asked}");

    //: It is told nobody is talking to it, so it does not produce a reply to a question that was
    //: never asked.
    assert!(asked.contains("아무도 너에게 말을 걸지 않았다"), "{asked}");
    //: And it is shown what it already thought, or there is nothing continuous about it.
    assert!(asked.contains("그가 코드를 보고 있다"), "{asked}");
}

/// **What argmind did is read off its own record, never asked of the model.**
///
/// The first run of the thinking tick was asked for `tried` and answered, from a session in which
/// argmind had done nothing at all: *이미 메모리 사용량이 높은 탭들을 목록으로 정리해두고…*. It had
/// not, and it carried that for three revisions. A fabricated `tried` is worse than an empty one —
/// the field exists to stop it offering the same thing twice, so a false entry suppresses an offer
/// that was never made.
#[test]
fn what_it_did_comes_from_the_record_and_holds_only_actions() {
    use argmind_core::thinking::what_it_did;

    let history: Vec<(String, String)> = vec![
        ("them".into(), "지우개로 바꿔줘".into()),
        ("you".into(), "지우개로 바꿨어요".into()),
        ("did".into(), "clicked the button \"설정\"".into()),
        ("noticed".into(), "the window changed to 'Paint'".into()),
        ("offered".into(), "to click the button \"Save\" — nothing was done".into()),
        ("could not".into(), "click the button \"Open\" — it had gone by then".into()),
    ];
    let said = what_it_did(&history, 3);

    //: The three action lines, and nothing a person or the screen said.
    assert!(said.contains("clicked the button"), "{said}");
    assert!(said.contains("nothing was done"), "{said}");
    assert!(said.contains("it had gone by then"), "{said}");
    assert!(!said.contains("지우개로 바꿔줘"), "somebody's words became an action: {said}");
    assert!(!said.contains("지우개로 바꿨어요"), "a reply became an action: {said}");
    assert!(!said.contains("the window changed"), "a watch became an action: {said}");

    //: **Bounded**, newest kept. A note is a working memory and an unbounded one is a prompt that
    //: grows for as long as argmind is open.
    let one = what_it_did(&history, 1);
    assert!(one.contains("it had gone by then"), "{one}");
    assert!(!one.contains("clicked the button"), "{one}");

    //: A conversation in which argmind did nothing says nothing, rather than inventing something.
    let quiet: Vec<(String, String)> = vec![("them".into(), "안녕".into()),
                                            ("you".into(), "안녕하세요".into())];
    assert_eq!(what_it_did(&quiet, 3), "");
}

/// **A thought needs a yesterday, and it is counted rather than asked for.**
///
/// The owner, 2026-09-17, looking at what the second field produced: *생각이 그냥 기다리는중이면
/// 의미가 없음. ai가 계속해서 스스로 판단할줄 알아야 함.* Part of the fault was that each tick saw
/// only *now* and the previous note, so it could not say *the fourth look at the same thing* —
/// there was nothing counting. `held` counts, and it is the program's fact, not the model's.
#[test]
fn a_belief_that_comes_back_unchanged_is_counted_as_held() {
    let mut note = Note::blank();
    note.revise("그가 같은 문서를 보고 있다", Judged::Nothing);
    assert_eq!(note.held, 1);
    //: The first time it is not worth saying, so it is not rendered.
    assert!(!note.block().unwrap().contains("번째 그대로"));

    note.revise("그가 같은 문서를 보고 있다", Judged::Nothing);
    note.revise("그가 같은 문서를 보고 있다", Judged::Nothing);
    assert_eq!(note.held, 3);
    assert!(note.block().unwrap().contains("3번째 그대로"), "{:?}", note.block());

    //: **A different belief starts again at one.** Otherwise the count would be a measure of how
    //: long argmind has been running rather than of how long anything has been true.
    note.revise("그가 다른 창으로 옮겼다", Judged::Nothing);
    assert_eq!(note.held, 1);
    assert!(!note.block().unwrap().contains("번째 그대로"));
}

/// **An empty `worth` is the ordinary answer, and it is still an answer.**
///
/// 31 of 34 conversational turns are measured as ones that should produce no action, so a field
/// that is always full is a yes-machine and worse than no field. A reply is therefore judged on
/// `believes` alone — treating *a belief and nothing worth doing* as nothing at all would throw
/// away the useful half of most ticks.
#[test]
fn nothing_worth_doing_is_a_complete_answer() {
    let mut note = Note::blank();
    assert!(read_note(r#"{"believes": "그가 뉴스를 읽고 있다", "worth": null}"#,
                      &mut note, &screen(), "나타남: Eraser"));
    assert_eq!(note.believes, "그가 뉴스를 읽고 있다");
    assert_eq!(note.worth, None);
    //: And it renders: the belief is there, the judgement is absent rather than blank.
    let block = note.block().unwrap();
    assert!(block.contains("믿는 것"), "{block}");
    assert!(!block.contains("할 만한 것"), "{block}");

    //: **The question says so out loud**, because a model that is not told will fill the field.
    //: The exact wording is checked in the next test; what matters here is that saying nothing
    //: is offered as the right answer.
    let asked = ask_for_a_note(&note, "0: button \"one\"", "", &[], MOST_CHARS);
    assert!(asked.contains("대개는 null 이 맞다"), "{asked}");
    assert!(asked.contains("할 게 없다고 판단하는 것도 판단이다"), "{asked}");
    //: **A number, and told so.** Prose in this field failed three times.
    assert!(asked.contains("설명은 쓰지 마라, 번호만"), "{asked}");
}

/// **A judgement needs something to have happened, and what happened is computed here.**
///
/// The first rewrite only renamed the field, and the measurement said so: over 24 ticks `worth`
/// came back *입력을 기다리는 것이 적절하며* and *대기하는 것이 좋다* five times, and most of the
/// rest was *도움을 줄 수 있다* — willingness, not judgement. It could not tell a screen that had
/// just turned over from one that had sat still, so those were the only answers available.
#[test]
fn what_changed_reports_events_and_says_nothing_when_nothing_happened() {
    use argmind_core::thinking::{what_changed, MOST_NAMED};

    let before = vec![Widget::new("Save", "button", (0, 0, 1, 1), "Paint"),
                      Widget::new("Open", "button", (0, 0, 1, 1), "Paint")];

    //: **Nothing happened is the ordinary case**, and it must read as nothing rather than as a
    //: description of the screen.
    assert_eq!(what_changed(&before, &before), "");

    //: **The first look has no before.** Reporting the whole screen as newly appeared would open
    //: every session with a hundred events.
    assert_eq!(what_changed(&[], &before), "");

    let after = vec![Widget::new("Save", "button", (0, 0, 1, 1), "Paint"),
                     Widget::new("Eraser", "button", (0, 0, 1, 1), "Paint")];
    let said = what_changed(&before, &after);
    assert!(said.contains("Eraser"), "{said}");
    assert!(said.contains("Open"), "{said}");
    assert!(said.contains("나타남"), "{said}");
    assert!(said.contains("사라짐"), "{said}");

    //: A window change is an event in its own right and is named as one.
    let elsewhere = vec![Widget::new("Save", "button", (0, 0, 1, 1), "Character Map")];
    assert!(what_changed(&before, &elsewhere).contains("Character Map"));

    //: **A rectangle is never an event.** A widget that moved is not news, and a coordinate is the
    //: one thing the model may not be shown.
    let moved = vec![Widget::new("Save", "button", (500, 500, 600, 600), "Paint"),
                     Widget::new("Open", "button", (500, 500, 600, 600), "Paint")];
    assert_eq!(what_changed(&before, &moved), "", "a move was reported as an event");

    //: **Bounded.** A window change turns the whole list over, and a hundred names is the same
    //: event said a hundred times.
    let many: Vec<Widget> = (0..50)
        .map(|i| Widget::new(&format!("thing {i}"), "button", (0, 0, 1, 1), "Paint")).collect();
    let flood = what_changed(&before, &many);
    assert!(flood.matches("thing ").count() <= MOST_NAMED, "{flood}");
}

/// The rule the model is given is **conditional on the event**, not on how it feels.
#[test]
fn the_question_ties_an_empty_judgement_to_nothing_having_happened() {
    let note = Note::blank();
    let quiet = ask_for_a_note(&note, "0: button \"one\"", "", &[], MOST_CHARS);
    assert!(quiet.contains("달라진 것: 없음"), "{quiet}");
    assert!(quiet.contains("달라진 것이 없으면 worth 는 반드시 null"), "{quiet}");
    //: **An example of the shape rather than a list of what to avoid.** The version that listed
    //: the filler read as *say nothing*: `worth` came back empty on every tick, including one where
    //: an unsaved-changes dialog had just appeared.
    assert!(quiet.contains("예를 들어"), "the question shows no example: {quiet}");
    //: **The example is a number.** The version that showed a sentence got the sentence back,
    //: every tick, on the arm where nothing had happened as well as the one where something did.
    assert!(quiet.contains("worth 는 12 다"), "the example is not a number: {quiet}");

    let busy = ask_for_a_note(&note, "0: button \"one\"", "나타남: Eraser", &[], MOST_CHARS);
    assert!(busy.contains("나타남: Eraser"), "{busy}");
    assert!(!busy.contains("달라진 것: 없음"), "{busy}");
}

/// **A judgement is a widget, and a number that names nothing is nothing.**
///
/// Prose in this field failed three times: filler on 58% of ticks, then silence even when a dialog
/// appeared, then a verbatim copy of the example it was shown. So the model picks a number out of
/// the list it is already looking at — the one thing `choosing` measures it good at — and the
/// number becomes a name and a role here, exactly as `converse::effect_of` does it.
#[test]
fn a_judgement_that_names_nothing_real_is_nothing() {
    let listed = screen();
    let at = |v: serde_json::Value| worth_of(Some(&v), &listed);

    assert_eq!(at(serde_json::json!(0)), Some(Worth { name: "one".into(),
                                                      role: "button".into() }));
    assert_eq!(at(serde_json::json!(1)), Some(Worth { name: "two".into(),
                                                      role: "combo box".into() }));
    //: A model that writes the number as a string still meant the number.
    assert_eq!(at(serde_json::json!("1")).unwrap().name, "two");

    //: **Everything else is nothing**, and the last of these is the whole point of the field: a
    //: sentence where a number was asked for buys no judgement at all.
    for bad in [serde_json::json!(null), serde_json::json!(2), serde_json::json!(-1),
                serde_json::json!("도와줄 수 있다"), serde_json::json!(true),
                serde_json::json!(["one"])] {
        assert_eq!(at(bad.clone()), None, "{bad} was read as a judgement");
    }
    assert_eq!(worth_of(None, &listed), None);
    //: And nothing on screen means nothing can be named, whatever number comes back.
    assert_eq!(worth_of(Some(&serde_json::json!(0)), &[]), None);
}

/// **A judgement that nobody acts on is counted against itself.** The same arithmetic as `held` and
/// the opposite meaning: a belief coming back is confirmation, a judgement coming back is an offer
/// nobody made.
#[test]
fn a_judgement_that_goes_unoffered_is_counted() {
    let save = Worth { name: "Save".into(), role: "button".into() };
    let mut note = Note::blank();

    note.revise("그가 나가려 한다", Judged::This(save.clone()));
    assert_eq!(note.stuck, 1);
    assert!(!note.block().unwrap().contains("아무 일도 없음"));

    note.revise("그가 나가려 한다", Judged::This(save.clone()));
    note.revise("그가 나가려 한다", Judged::This(save.clone()));
    assert_eq!(note.stuck, 3);
    let block = note.block().unwrap();
    assert!(block.contains("3번째인데 아무 일도 없음"), "{block}");
    //: **The sentence around it is the program's**, which is what leaves no room for filler: the
    //: model chose a widget and never wrote a word.
    assert!(block.contains("button"), "{block}");
    assert!(block.contains("Save"), "{block}");

    //: A different widget starts again; dropping the judgement leaves nothing to count.
    note.revise("그가 나가려 한다",
                Judged::This(Worth { name: "Cancel".into(), role: "button".into() }));
    assert_eq!(note.stuck, 1);
    note.revise("그가 나가려 한다", Judged::Nothing);
    assert_eq!(note.stuck, 0);
    assert!(!note.block().unwrap().contains("할 만한 것"));
}

/// **Nothing happened, so there is no judgement — and the program holds that, not the prompt.**
///
/// The question says 달라진 것이 없으면 worth 는 반드시 null. The model ignored it on **6 of 8**
/// such ticks, naming a window title and a browser menu button. A rule in a prompt is not a rule,
/// which this project has written down twice already; `what_changed` is computed here, so whether
/// anything happened is the program's fact and the gate belongs with it.
#[test]
fn a_judgement_needs_an_event_and_the_reader_enforces_it() {
    let listed = screen();
    let picked = r#"{"believes": "무언가 하고 있다", "worth": 0}"#;

    //: Something happened: the choice stands.
    let mut moved = Note::blank();
    assert!(read_note(picked, &mut moved, &listed, "나타남: Eraser"));
    assert_eq!(moved.worth.as_ref().map(|w| w.name.clone()), Some("one".into()));

    //: Nothing happened: the same reply, and the judgement is dropped rather than argued with.
    let mut still = Note::blank();
    assert!(read_note(picked, &mut still, &listed, ""));
    assert_eq!(still.worth, None, "a judgement got through with no event behind it");
    //: **The belief still lands.** Dropping the judgement must not throw away the useful half of a
    //: quiet tick, which is what the note is mostly made of.
    assert_eq!(still.believes, "무언가 하고 있다");

    //: Whitespace is not an event either — a change report of spaces is a change report of nothing.
    let mut spaces = Note::blank();
    assert!(read_note(picked, &mut spaces, &listed, "   \n "));
    assert_eq!(spaces.worth, None);
}

// ---- a reason to start, rather than to answer ---------------------------------------------------

/// **A quiet look does not erase a standing judgement.**
///
/// The first version of the event gate cleared `worth` whenever nothing had changed, and that made
/// `stuck` unable to reach 2: a judgement could not survive one quiet look, so there was never
/// anything for argmind to bring up of its own accord. *Nothing happened* is a reason not to form a
/// new judgement; it is not a reason to forget the one already standing.
#[test]
fn a_standing_judgement_survives_a_quiet_look_and_counts_it() {
    let save = Worth { name: "저장".into(), role: "button".into() };
    let mut note = Note::blank();

    note.revise("그가 나가려 한다", Judged::This(save.clone()));
    assert_eq!(note.stuck, 1);

    //: Two quiet looks. It still stands, and each one is counted against it.
    note.revise("그가 나가려 한다", Judged::NotAsked);
    note.revise("그가 나가려 한다", Judged::NotAsked);
    assert_eq!(note.worth.as_ref(), Some(&save), "a quiet look erased what was standing");
    assert_eq!(note.stuck, 3);

    //: **Something happened and argmind judged there was nothing to do** — that is what clears it,
    //: because that is an answer rather than an absence of one.
    note.revise("그가 돌아왔다", Judged::Nothing);
    assert_eq!(note.worth, None);
    assert_eq!(note.stuck, 0);

    //: And a quiet look with nothing standing stays nothing rather than counting up from zero.
    note.revise("그가 돌아왔다", Judged::NotAsked);
    assert_eq!(note.stuck, 0);
}

/// **The one thing that starts a turn from argmind's own state.**
///
/// A `Watch` fires because the world moved. This fires because a judgement has stood for
/// `RAISE_AFTER` looks and nobody has done anything about it.
#[test]
fn a_judgement_becomes_worth_raising_only_after_it_has_stood_a_while() {
    use argmind_core::thinking::{raising, RAISE_AFTER};
    let save = Worth { name: "저장".into(), role: "button".into() };
    let mut note = Note::blank();

    //: Nothing standing is never worth interrupting somebody over.
    assert!(note.worth_raising().is_none());

    note.revise("그가 나가려 한다", Judged::This(save.clone()));
    for look in 1..RAISE_AFTER {
        assert!(note.worth_raising().is_none(), "raised after only {look} look(s)");
        note.revise("그가 나가려 한다", Judged::NotAsked);
    }
    assert_eq!(note.worth_raising(), Some(&save));

    //: **Raised once.** Putting it down is what stops a thing brought up and ignored coming back
    //: thirty seconds later, which is the defect this project has found in three other shapes.
    note.let_go();
    assert!(note.worth_raising().is_none());
    assert_eq!(note.stuck, 0);

    //: The sentence is the program's and carries the widget's own words — the model wrote none of
    //: it, which is the whole reason the field stopped being prose.
    let said = raising(&save);
    assert!(said.contains("저장"), "{said}");
    assert!(said.contains("button"), "{said}");
}
