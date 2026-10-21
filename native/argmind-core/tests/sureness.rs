//! **How sure the model was of the action, read off the tokens of `do`.**

use argmind_core::converse::action_sureness;

fn toks(parts: &[(&str, f64)]) -> Vec<(String, f64)> {
    parts.iter().map(|(t, p)| (t.to_string(), p.ln())).collect()
}

#[test]
fn the_action_is_the_tokens_after_do() {
    //: {"say": "네", "do": [{"click": 3}]} — the say tokens are certain, the index is not.
    let t = toks(&[("{\"", 1.0), ("say", 1.0), ("\":", 1.0), (" \"", 1.0), ("네", 0.2), ("\",", 1.0),
                   (" \"", 1.0), ("do", 1.0), ("\":", 1.0), (" [{\"", 0.9), ("click", 1.0),
                   ("\":", 1.0), (" ", 1.0), ("3", 0.5), ("}]}", 1.0)]);
    let s = action_sureness(&t).unwrap();
    assert!((s - 0.45).abs() < 1e-9, "{s}: the say's 0.2 must not count, the index's 0.5 must");
}

#[test]
fn no_action_is_no_sureness() {
    for tail in ["null}", "[]}"] {
        let t = toks(&[("{\"say\": \"안녕\", \"do\": ", 1.0), (tail, 0.7)]);
        assert_eq!(action_sureness(&t), None, "{tail}");
    }
    assert_eq!(action_sureness(&toks(&[("{\"say\": \"안녕\"}", 0.9)])), None);
    assert_eq!(action_sureness(&[]), None, "a model that cannot say");
}
