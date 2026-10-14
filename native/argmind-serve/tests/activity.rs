//! **The timeline reads what the page is told, and a page can read the timeline.**
//!
//! The events here are the exact shapes `app.rs` sends — `as_json` for a proposal, the `ran` and
//! `gone` frames — so a change to either side that stops them agreeing fails here rather than as an
//! empty line in a panel nobody opened.

use argmind_serve::activity::{field, reading, Activity, KEPT};
use argmind_serve::app::as_json;
use argmind_serve::proposals::Proposal;

#[test]
fn what_the_page_is_told_becomes_a_line() {
    assert_eq!(reading("say", "안녕하세요"), Some(("말함", "안녕하세요".into())));
    assert_eq!(reading("them", "메모장 열어줘"), Some(("들음", "메모장 열어줘".into())));
    assert_eq!(reading("ran", r#"{"token":3,"label":"클릭  button “저장”","confirmed":true}"#),
               Some(("실행", "클릭  button \u{201c}저장\u{201d} — 화면이 바뀜".into())));
    assert_eq!(reading("ran", r#"{"token":3,"label":"키  enter","confirmed":null}"#),
               Some(("실행", "키  enter — 확인하지 못함".into())));
    assert_eq!(reading("gone", r#"{"token":-1,"what":"Open"}"#),
               Some(("못 함", "Open — 그새 사라짐".into())));
    assert_eq!(reading("held", "1"), Some(("보류", "할 말을 미뤄 둠".into())));
}

#[test]
fn the_interface_talking_to_itself_is_not_a_line() {
    for (event, data) in [("busy", "1"), ("held", "0"), ("wonder", "null"), ("say", ""),
                          ("mind", "{}"), ("errands", "[]"), ("activity", "[]"),
                          ("activity-line", "{}"), ("attending", "")] {
        assert_eq!(reading(event, data), None, "{event} {data}");
    }
}

/// **The proposal's own JSON, as `app.rs` writes it** — quotes and a newline in the label included.
#[test]
fn a_proposal_reads_back_through_the_shape_it_was_written_in() {
    let p = Proposal { token: 7, label: "입력  \u{201c}그가 \"안녕\"\n했다\u{201d}".into(),
                       hint: String::new(), effect: Default::default() };
    let sent = as_json(&p);
    assert_eq!(field(&sent, "label").as_deref(), Some(p.label.as_str()));
    assert_eq!(reading("propose", &sent), Some(("제안", p.label.clone())));
}

#[test]
fn the_same_window_twice_is_one_line_and_the_list_is_bounded() {
    let mut a = Activity::default();
    a.push(1, "보는 창", "메모장");
    a.push(2, "말함", "안녕");
    a.push(3, "보는 창", "메모장");
    assert_eq!(a.len(), 2, "the window had not changed");
    a.push(4, "보는 창", "Chrome");
    assert_eq!(a.len(), 3);
    for i in 0..KEPT + 10 {
        a.push(10 + i as u64, "말함", "또");
    }
    assert_eq!(a.len(), KEPT);
    assert!(a.push(0, "말함", "   ").is_none(), "nothing is not a line");
}

#[test]
fn the_page_gets_json_that_parses() {
    let mut a = Activity::default();
    a.push(1_790_782_200, "말함", "따옴표 \"와\" 줄\n바꿈 \\ 끝");
    let v: serde_json::Value = serde_json::from_str(&a.json()).unwrap();
    assert_eq!(v[0]["text"], "따옴표 \"와\" 줄\n바꿈 \\ 끝");
    assert_eq!(v[0]["kind"], "말함");
    assert_eq!(v[0]["at"], 1_790_782_200u64);
    a.clear();
    assert_eq!(a.json(), "[]");
}
