//! **맡긴 일 reaches the server, and nothing here writes the real list.**
//!
//! `%LOCALAPPDATA%\argmind\errands.json` holds what the person running this actually asked for, so
//! the file half is tested through `argmind_core::errand::{render_all, parse_all}` and the list
//! functions in `argmind_serve::errands`, and only the routes are tested end to end.

use argmind_serve::{route, Request, Route};

fn post(path: &str, body: &str) -> Route {
    route(&Request { method: "POST".into(), path: path.into(), token: None, body: body.into() })
}

/// The words travel in the body — a person's sentence, never a path.
#[test]
fn an_errand_and_its_answer_carry_their_words_in_the_body() {
    assert_eq!(post("/errand", "선물 아이디어 정리해줘"), Route::Errand("선물 아이디어 정리해줘".into()));
    assert_eq!(post("/errand/3/answer", "예산은 10만원"), Route::ErrandAnswer(3, "예산은 10만원".into()));
    assert_eq!(post("/errand/3/drop", ""), Route::ErrandDrop(3));
}

/// **A path the page could invent is refused**, the same rule the dials and the permissions keep.
#[test]
fn a_malformed_errand_path_is_unknown() {
    for bad in ["/errand/x/answer", "/errand/3/finish", "/errand/3", "/errand/", "/errand/3/drop/4"] {
        assert_eq!(post(bad, ""), Route::Unknown, "{bad}");
    }
}

/// **A list that survives a restart** — written and read back through the same two functions the
/// application uses, without touching the real file.
#[test]
fn the_list_written_down_reads_back_the_same() {
    use argmind_core::errand::{parse_all, render_all, State};
    let mut list = Vec::new();
    argmind_serve::errands::add(&mut list, "운동 루틴 짜줘").unwrap();
    argmind_serve::errands::add(&mut list, "날씨 알아봐줘").unwrap();
    list[1].state = State::NeedsYou;
    list[1].ask = argmind_core::errand::CANNOT_REACH.into();
    assert_eq!(parse_all(&render_all(&list)), list);
}
