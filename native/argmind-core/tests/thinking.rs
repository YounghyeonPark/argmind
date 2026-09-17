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
use argmind_core::thinking::{ask_for_a_note, read_note, Note, MOST_CHARS};

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
    spaces.revise("  ", "\n");
    spaces.record(" \t ");
    assert!(spaces.empty());
    assert_eq!(prompt(&spaces), prompt(&Note::blank()));
}

/// **And a note that says something reaches the model.** The other half: a slot nobody can see the
/// effect of is a field that quietly stops being rendered.
#[test]
fn every_field_of_a_note_reaches_the_prompt() {
    let mut note = Note::blank();
    note.revise("그가 코드 리뷰 중이다", "빌드가 끝나기를");
    //: `tried` comes from the record, never from the model — so it is set the way the session
    //: sets it.
    note.record("clicked the button \"설정\"");
    let block = note.block().expect("a note with three fields renders nothing");
    for text in ["그가 코드 리뷰 중이다", "빌드가 끝나기를",
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
    note.revise("그가 문서를 읽고 있다", "");
    let block = note.block().unwrap();
    assert!(block.contains("믿는 것"), "{block}");
    assert!(!block.contains("기다리는 것"), "{block}");
    assert!(!block.contains("이미 한 것"), "{block}");
}

/// **Revising replaces; it never appends.** The failure this is guarding is a note that grows every
/// tick until the prompt is mostly history of itself.
#[test]
fn revising_replaces_what_was_there_and_counts_once() {
    let mut note = Note::blank();
    note.revise("첫 번째", "a");
    assert_eq!(note.revisions, 1);
    note.revise("두 번째", "c");
    assert_eq!(note.revisions, 2);
    assert_eq!(note.believes, "두 번째");
    assert!(!note.block().unwrap().contains("첫 번째"));

    //: **Capped.** A field longer than the limit is cut rather than carried, because a prompt that
    //: grows without bound is a prompt the model stops reading.
    note.revise(&"가".repeat(MOST_CHARS * 3), "");
    assert_eq!(note.believes.chars().count(), MOST_CHARS);
}

/// **A tick that produced nothing usable leaves the note exactly as it was.**
///
/// The dangerous direction: a quiet minute is the ordinary case, and a model answering with three
/// empty strings must not erase what the last tick worked out.
#[test]
fn an_unusable_reply_does_not_clear_what_was_already_there() {
    let mut note = Note::blank();
    note.revise("그가 회의 중이다", "회의가 끝나기를");
    let before = note.clone();

    for reply in [r#"{"believes": "", "waiting": ""}"#,
                  "not json at all",
                  "",
                  r#"{"something": "else"}"#] {
        assert!(!read_note(reply, &mut note), "{reply:?} was accepted");
        assert_eq!(note, before, "{reply:?} changed the note");
    }

    //: And a reply with one real field is an update, with the other two cleared — because the model
    //: was asked to replace the note, and a partial merge would make *I no longer believe that*
    //: impossible to say.
    assert!(read_note(r#"{"believes": "그가 돌아왔다", "waiting": ""}"#, &mut note));
    assert_eq!(note.believes, "그가 돌아왔다");
    assert_eq!(note.waiting, "");
    assert_eq!(note.revisions, before.revisions + 1);
}

/// **The thinking call cannot speak and cannot act**, and that is what makes it safe on a timer.
#[test]
fn the_thinking_question_asks_for_no_speech_and_no_action() {
    let mut note = Note::blank();
    note.revise("그가 코드를 보고 있다", "");
    let asked = ask_for_a_note(&note, "0: button \"one\"", &[], MOST_CHARS);

    //: The two fields it may answer with, and nothing else.
    assert!(asked.contains("\"believes\""), "{asked}");
    assert!(asked.contains("\"waiting\""), "{asked}");
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
