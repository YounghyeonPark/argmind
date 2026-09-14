//! The bytes that go out, against 28 recorded from the shipped Python — and the de-chunker.

use std::fs;
use std::io::Cursor;
use std::path::PathBuf;

use argmind_env_ollama::{py_string, Lines, Local};
use serde_json::Value;

fn cases() -> Vec<Value> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap()
        .join("expectations").join("wire.json");
    let raw = fs::read_to_string(&path).unwrap_or_else(|_| {
        panic!("{} is missing. Generate it: python native/expectations/all.py", path.display())
    });
    serde_json::from_str(&raw).expect("parse")
}

/// **Byte for byte.** Two traps live here and only a byte comparison finds them: `ensure_ascii`
/// turns every Korean character into `\uXXXX`, and Python serialises a dict in insertion order.
#[test]
fn the_request_body_is_what_python_sends() {
    let all = cases();
    assert_eq!(all.len(), 28, "the recorded set changed size");
    let mut escaped = 0;
    for case in &all {
        let mut local = Local::new(case["model"].as_str().unwrap());
        local.ctx = case["ctx"].as_i64().unwrap();
        local.max_tokens = case["max_tokens"].as_i64().unwrap();
        local.keep_alive = case["keep_alive"].as_str().unwrap().to_string();
        local.think = case["think"].as_bool();
        let want = case["body"].as_str().unwrap();
        if want.contains("\\u") {
            escaped += 1;
        }
        assert_eq!(local.body(case["prompt"].as_str().unwrap()), want,
                   "{}", case["label"].as_str().unwrap());
    }
    assert!(escaped >= 8, "the non-ASCII prompts left the set; they are what catch ensure_ascii");
}

#[test]
fn python_escaping_handles_the_awkward_characters() {
    assert_eq!(py_string("a\"b\\c"), r#""a\"b\\c""#);
    assert_eq!(py_string("\n\t\r"), r#""\n\t\r""#);
    //: Everything outside 0x20..=0x7e goes, including DEL.
    assert_eq!(py_string("\u{7f}"), r#""\u007f""#);
    assert_eq!(py_string("복"), r#""\ubcf5""#);
    //: An astral character leaves as a surrogate pair, which is the part a naive escape gets wrong.
    assert_eq!(py_string("👍"), r#""\ud83d\udc4d""#);
}

/// **A chunk boundary is not a line boundary.** A JSON line can be split across two chunks and two
/// lines can share one; treating them as the same thing is the obvious bug here and it would only
/// show on long replies.
#[test]
fn the_dechunker_does_not_confuse_chunks_with_lines() {
    //: **The sizes are computed, not typed.** My first version of this fixture said `10` over a
    //: 13-byte chunk and the test failed against a de-chunker that was right — a hand-written
    //: length is a second implementation of the thing under test, and it was the one with the bug.
    let pieces = ["{\"a\": 1}\n{\"b\"", ": 2}\n{\"c\"", ": 3}\n"];
    let mut body = String::new();
    for piece in pieces {
        body.push_str(&format!("{:x}\r\n{piece}\r\n", piece.len()));
    }
    body.push_str("0\r\n\r\n");
    let body = body.as_str();
    let got: Vec<String> = Lines::new(Cursor::new(body), true).map(|l| l.unwrap()).collect();
    assert_eq!(got, vec![r#"{"a": 1}"#, r#"{"b": 2}"#, r#"{"c": 3}"#]);

    //: The unchunked path, and a body whose last line has no newline.
    let plain = "{\"a\": 1}\n{\"b\": 2}";
    let got: Vec<String> = Lines::new(Cursor::new(plain), false).map(|l| l.unwrap()).collect();
    assert_eq!(got, vec![r#"{"a": 1}"#, r#"{"b": 2}"#]);
}
