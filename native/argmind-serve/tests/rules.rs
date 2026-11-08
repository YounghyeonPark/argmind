//! **규칙 reach the server as a list.** Nothing here writes `%LOCALAPPDATA%\argmind\rules.json` —
//! the reading of a list is `argmind_core::rule::from_lines`, tested in the core.

use argmind_serve::{route, Request, Route};

#[test]
fn the_rules_travel_whole_in_the_body() {
    let body = "카카오톡\ttype\tnever\n\tall\tconfirm";
    let r = route(&Request { method: "POST".into(), path: "/rules".into(), token: None,
                             body: body.into() });
    assert_eq!(r, Route::Rules(body.into()));
    //: An empty list is how every rule is removed, not a malformed request.
    let r = route(&Request { method: "POST".into(), path: "/rules".into(), token: None,
                             body: String::new() });
    assert_eq!(r, Route::Rules(String::new()));
}

/// **A 규칙 reaches the work screen** — the owner's 규칙 had covered only `propose` and `act`, and a
/// task's press on argmind's own desktop went through neither.
#[test]
fn the_persons_rules_reach_the_work_screen() {
    use argmind_core::rule::{Rule, Then};
    use argmind_serve::rules::{on_the_work_screen, OnTheWorkScreen::*};
    let rule = |app: &str, action: &str, then| Rule { app: app.into(), action: action.into(), then };
    //: None at rest: what 확인 없이 실행 says, alone.
    assert_eq!(on_the_work_screen(&[], "주문 조회", "click", true), Go);
    assert_eq!(on_the_work_screen(&[], "주문 조회", "click", false), Ask);
    let rules = [rule("주문", "click", Then::Never), rule("", "type", Then::Confirm)];
    assert!(matches!(on_the_work_screen(&rules, "주문 조회", "click", true), Refuse(s) if s.contains("주문")));
    //: The same rule elsewhere, and another kind in the same window, are not caught by it.
    assert_eq!(on_the_work_screen(&rules, "문자표", "click", true), Go);
    assert_eq!(on_the_work_screen(&rules, "주문 조회", "type", true), Ask, "항상 물어봄 everywhere for typing");
    //: **The strictest wins in either order**, and a looser rule cannot undo 하지 않음.
    let both = [rule("", "all", Then::Confirm), rule("주문", "all", Then::Never)];
    assert!(matches!(on_the_work_screen(&both, "주문 조회", "type", true), Refuse(_)));
    assert!(matches!(on_the_work_screen(&[both[1].clone(), both[0].clone()], "주문 조회", "type", true), Refuse(_)));
}
