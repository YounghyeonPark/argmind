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
