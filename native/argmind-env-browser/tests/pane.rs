//! **The pane's eyes and hands against a page that is a script** — no browser, no window.
//!
//! The wire answers from a table and records what it was sent, so each test can say both what the
//! eyes made of an answer and what the hands put on the wire. **Both directions for every guard**:
//! a reach check that always said yes, or a password check that always said no, would pass the half
//! that only looks at the happy path.

use std::sync::{Arc, Mutex};

use argmind_core::contract::{Did, Eyes, Hands};
use argmind_env_browser::{listing, Pane};
use serde_json::{json, Value};

/// A page whose answers are a function of the method, and a log of every call.
fn page(reaches: bool, secret: bool) -> (Pane, Arc<Mutex<Vec<(String, Value)>>>) {
    let log = Arc::new(Mutex::new(Vec::new()));
    let seen = log.clone();
    let wire = move |method: &str, params: &str| -> Result<String, String> {
        let p: Value = serde_json::from_str(params).unwrap();
        seen.lock().unwrap().push((method.to_string(), p.clone()));
        Ok(match method {
            "Accessibility.getFullAXTree" => tree(),
            "DOMSnapshot.captureSnapshot" => snapshot(),
            "Page.getLayoutMetrics" => metrics(),
            "Runtime.evaluate" => {
                let e = p["expression"].as_str().unwrap();
                let v = if e.contains("activeElement") { json!(secret) }
                        else if e == "document.title" { json!("네이버") }
                        else if e == "location.host" { json!("www.naver.com") }
                        else { json!(null) };
                json!({ "result": { "value": v } })
            }
            "DOM.resolveNode" => json!({ "object": { "objectId": "node-1" } }),
            "Runtime.callFunctionOn" => json!({ "result": { "value": reaches } }),
            _ => json!({}),
        }.to_string())
    };
    (Pane::new(Arc::new(wire)), log)
}

/// Three nodes: a link on screen, a button below the fold, and an ignored one.
fn tree() -> Value {
    json!({ "nodes": [
        { "nodeId": "1", "ignored": false, "role": { "value": "link" }, "name": { "value": "메일" },
          "backendDOMNodeId": 10 },
        { "nodeId": "2", "ignored": false, "role": { "value": "button" }, "name": { "value": "아래" },
          "backendDOMNodeId": 11 },
        { "nodeId": "3", "ignored": true, "role": { "value": "link" }, "name": { "value": "숨김" },
          "backendDOMNodeId": 12 },
        { "nodeId": "4", "ignored": false, "role": { "value": "paragraph" }, "name": { "value": "글" },
          "backendDOMNodeId": 13 },
    ]})
}

/// **Device pixels**, at a display scale of 1.5 — the case the harness's control caught.
fn snapshot() -> Value {
    json!({ "documents": [{
        "nodes": { "backendNodeId": [1, 10, 11, 12, 13] },
        "layout": { "nodeIndex": [1, 2, 3, 4],
                    "bounds": [[150.0, 300.0, 90.0, 30.0], [150.0, 3000.0, 90.0, 30.0],
                               [0.0, 0.0, 30.0, 30.0], [0.0, 60.0, 300.0, 30.0]] }
    }]})
}

fn metrics() -> Value {
    json!({ "layoutViewport": { "pageX": 0, "pageY": 0, "clientWidth": 1920, "clientHeight": 1350 },
            "cssLayoutViewport": { "pageX": 0, "pageY": 0, "clientWidth": 1280, "clientHeight": 900 } })
}

#[test]
fn only_named_pressable_things_on_screen_are_listed_in_css_pixels() {
    let listed = listing(&tree(), &snapshot(), &metrics(), "네이버");
    assert_eq!(listed.len(), 1, "below the fold, ignored and not pressable are all left out");
    let (w, id) = &listed[0];
    assert_eq!((w.name.as_str(), w.role.as_str(), *id), ("메일", "link", 10));
    //: 150,300,90,30 device pixels at 1.5 is 100,200,60,20 CSS.
    assert_eq!(w.rect, (100, 200, 60, 20));
}

#[test]
fn the_window_a_rule_sees_is_the_title_and_the_host() {
    let (pane, _) = page(true, false);
    let seen = pane.eyes().foreground();
    assert_eq!(seen[0].window, "네이버 — www.naver.com");
}

#[test]
fn a_click_that_reaches_its_target_is_sent_at_the_centre_in_css_pixels() {
    let (pane, log) = page(true, false);
    let w = pane.eyes().foreground().remove(0);
    let mut hands = pane.hands(true);
    hands.click(&w);
    assert_eq!(hands.done.last(), Some(&Did::Sent { live: true }));
    let log = log.lock().unwrap();
    let mouse: Vec<&Value> = log.iter().filter(|(m, _)| m == "Input.dispatchMouseEvent")
        .map(|(_, p)| p).collect();
    assert_eq!(mouse.len(), 2, "pressed and released");
    assert_eq!((mouse[0]["x"].as_f64(), mouse[0]["y"].as_f64()), (Some(130.0), Some(210.0)));
}

#[test]
fn a_covered_target_is_refused_and_nothing_is_pressed() {
    let (pane, log) = page(false, false);
    let w = pane.eyes().foreground().remove(0);
    let mut hands = pane.hands(true);
    hands.click(&w);
    assert_eq!(hands.done.last(), Some(&Did::Elsewhere { owner: None }));
    assert!(!log.lock().unwrap().iter().any(|(m, _)| m.starts_with("Input.")));
}

#[test]
fn dry_hands_check_and_send_nothing() {
    let (pane, log) = page(true, false);
    let w = pane.eyes().foreground().remove(0);
    let mut hands = pane.hands(false);
    hands.click(&w);
    hands.type_text("안녕");
    hands.press("enter");
    assert!(hands.done.iter().all(|d| *d == Did::Sent { live: false }));
    assert!(!log.lock().unwrap().iter().any(|(m, _)| m.starts_with("Input.")));
}

#[test]
fn typing_goes_to_the_page_and_a_password_field_is_read_as_one() {
    let (pane, log) = page(true, false);
    let mut hands = pane.hands(true);
    assert!(!hands.into_a_secret());
    hands.type_text("경복궁");
    assert!(log.lock().unwrap().iter()
        .any(|(m, p)| m == "Input.insertText" && p["text"] == "경복궁"));
    let (secret, _) = page(true, true);
    assert!(secret.hands(true).into_a_secret());
}

#[test]
fn a_widget_from_an_older_reading_is_not_guessed_at() {
    let (pane, log) = page(true, false);
    let mut w = pane.eyes().foreground().remove(0);
    w.rect = (0, 0, 1, 1);
    let mut hands = pane.hands(true);
    hands.click(&w);
    assert!(matches!(hands.done.last(), Some(Did::Refused(_))));
    assert!(!log.lock().unwrap().iter().any(|(m, _)| m.starts_with("Input.")));
}

#[test]
fn keys_are_named_and_enter_carries_its_character() {
    let (pane, log) = page(true, false);
    let mut hands = pane.hands(true);
    hands.press("Enter");
    hands.press("f13");
    assert!(matches!(hands.done.last(), Some(Did::Refused(_))));
    let log = log.lock().unwrap();
    let down = log.iter().find(|(m, p)| m == "Input.dispatchKeyEvent" && p["type"] == "keyDown")
        .unwrap();
    assert_eq!(down.1["text"], "\r");
}
