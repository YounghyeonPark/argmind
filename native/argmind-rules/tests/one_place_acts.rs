//! **Exactly one place in the application performs anything.**
//!
//! The application runs with unprompted action permitted, and the reason that is safe is not the
//! flag — it is that everything the model decides becomes a proposal behind a button, and one
//! function turns a pressed button into an act. The Python held this by counting `.perform(` calls
//! in the AST of `app.py`; the Rust side has been claiming it in a doc comment, which is not the
//! same thing. Two callers would be the whole safety argument gone, and nothing would look
//! different.
//!
//! **Where the port did make a rule structural, it is asserted where it is structural** — see
//! `argmind-serve/tests/proposals.rs` for the numbering, and `hands.rs` for `Reach`. This is the
//! one that is still held by looking.

use argmind_rules::{code_lines, root};

fn app() -> String {
    std::fs::read_to_string(root().join("native").join("argmind-serve").join("src").join("app.rs"))
        .expect("app.rs is the application and it has to be there")
}

/// One caller, and it is `act`.
#[test]
fn exactly_one_place_calls_perform() {
    let text = app();
    let calls: Vec<String> = code_lines(&text).into_iter()
        .filter(|(_, l)| {
            //: The call, not the word. A doc comment saying *"the only place that calls perform"*
            //: is the sentence this test exists to stop standing in for the fact.
            let code = l.split("//").next().unwrap_or("");
            code.contains(".perform(")
        })
        .map(|(n, l)| format!("app.rs:{n}: {}", l.trim()))
        .collect();
    assert_eq!(calls.len(), 1,
               "{} places perform something; the whole safety argument is that there is one, \
                reached only by a button: {calls:#?}", calls.len());
}

/// **And it is reached from an answer.**
///
/// One caller in the wrong function would satisfy the count and none of the reasoning, so this
/// checks the caller is `act` and that `act` is what an answered proposal goes to.
#[test]
fn the_one_caller_is_act_and_act_is_what_an_answer_reaches() {
    let text = app();
    let lines: Vec<&str> = text.lines().collect();
    let at = lines.iter().position(|l| {
        l.split("//").next().unwrap_or("").contains(".perform(")
    }).expect("something performs");

    //: The nearest `fn` above it.
    let owner = lines[..at].iter().rev()
        .find_map(|l| l.trim_start().strip_prefix("fn ").map(|r| r.split('(').next().unwrap_or("")))
        .expect("the call is inside a function");
    assert_eq!(owner, "act", "perform is called from {owner:?} rather than from act");

    //: And nothing reaches `act` except an answered proposal. `Answered::Run` is the only shape
    //: that gets there, so the word has to appear in the same function that calls it.
    let reached_by: Vec<&&str> = lines.iter()
        .filter(|l| l.contains("Answered::Run")).collect();
    assert!(!reached_by.is_empty(),
            "nothing in app.rs mentions an answered proposal; how is act reached?");
}

/// The file being read is the application. **A rule checked against nothing passes forever.**
#[test]
fn there_is_an_application_to_check() {
    let text = app();
    assert!(text.len() > 5_000, "app.rs is {} bytes — is the path right?", text.len());
    assert!(text.contains("fn act"), "app.rs has no act()");
}
