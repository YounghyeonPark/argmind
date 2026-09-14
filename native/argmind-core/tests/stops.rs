//! The stream stops, replayed against 128 recorded replies — no server, no model, no network.
//!
//! `tests/golden/llm-replay.txt` records what the shipped Python does to every recorded stream. This
//! asserts the Rust reader does the same thing to the same bytes, field for field: the reason, how
//! many lines it consumed before stopping, the token counts, the thinking length, and the sha of
//! both the text and the parsed object.
//!
//! It is the only way these stops have ever been checked against real output — every harness in
//! this project stores what a reply *meant* and none stored the reply, until the corpus was
//! recorded with the stops **off** so nothing was cut.
//!
//! **What it would fail to notice**: a server that splits a token stream into different lines. This
//! replays bytes rather than asking for new ones, and `GRACE` counts chunks.

use std::fs;
use std::path::{Path, PathBuf};

use argmind_core::llm::{first_object, Reading};
use serde_json::Value;
use sha2::{Digest, Sha256};

const SHIPPED: &str = "under the shipped stops";
const ANSWERED_ONLY: &str = "under the answered stop alone, on the rows where that differs";

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap().to_path_buf()
}

fn sha16(text: &str) -> String {
    format!("{:x}", Sha256::digest(text.as_bytes())).chars().take(16).collect()
}

/// `json.dumps(obj, ensure_ascii=False, sort_keys=True)`, which is what the golden hashed.
///
/// Written out rather than reached for, because `serde_json::to_string` uses `,`/`:` where Python
/// uses `, `/`: ` — a difference invisible until two shas disagree and somebody assumes the port is
/// wrong. Keys are already sorted: `serde_json::Map` is a `BTreeMap` unless `preserve_order` is on.
fn python_json(v: &Value) -> String {
    match v {
        Value::Null => "null".into(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => n.to_string(),
        Value::String(s) => serde_json::to_string(s).unwrap(),
        Value::Array(a) => {
            format!("[{}]", a.iter().map(python_json).collect::<Vec<_>>().join(", "))
        }
        Value::Object(o) => format!("{{{}}}", o.iter()
            .map(|(k, v)| format!("{}: {}", serde_json::to_string(k).unwrap(), python_json(v)))
            .collect::<Vec<_>>().join(", ")),
    }
}

/// Replay one recorded stream and describe what came out, in the golden's own shape.
fn outcome(lines: &[Value], answered: bool, looping: bool) -> Value {
    let mut reading = Reading::new(answered, looping);
    for entry in lines {
        let keep = match entry {
            //: A bare string is the common case: a content chunk and nothing else on the line.
            Value::String(s) => reading.line(0, Some(s), None),
            Value::Object(o) if o.contains_key("blank") => {
                reading.blank();
                true
            }
            Value::Object(o) => {
                let thinking = o.get("t").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                let chunk = o.get("c").and_then(|v| v.as_str());
                let done = o.get("done").map(|d| (
                    d.get("prompt_eval_count").and_then(|v| v.as_i64()),
                    d.get("eval_count").and_then(|v| v.as_i64()),
                    d.get("done_reason").and_then(|v| v.as_str()).unwrap_or("").to_string()));
                reading.line(thinking, chunk, done)
            }
            _ => true,
        };
        if !keep {
            break;
        }
    }
    let read = reading.lines_read;
    let reply = reading.finish();
    let got = first_object(&reply.text);
    let rendered = got.as_ref().map(python_json);
    serde_json::json!({
        "reason": reply.reason,
        "read": [read, lines.len()],
        "out_tokens": reply.out_tokens,
        "prompt_tokens": reply.prompt_tokens,
        "thought": reply.thought,
        "text_chars": reply.text.chars().count(),
        "text_sha": sha16(&reply.text),
        "json_sha": rendered.as_ref().map(|s| sha16(s)),
    })
}

fn golden() -> Value {
    let raw = fs::read_to_string(root().join("tests").join("golden").join("llm-replay.txt"))
        .expect("tests/golden/llm-replay.txt");
    let body = raw.split("#---- expectations ----").nth(1).expect("the sentinel");
    serde_json::from_str(body).expect("the golden is JSON below the sentinel")
}

fn corpora() -> Vec<(String, Value)> {
    let dir = root().join("measure");
    let mut out = Vec::new();
    for entry in fs::read_dir(&dir).expect("measure/").flatten() {
        let path = entry.path();
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        if name.starts_with("replies-") && name.ends_with(".json") {
            //: The **full** stem, fingerprints and all, because that is the key the golden uses —
            //: and it should: two corpora differing only in a fingerprint are two corpora.
            let stem = Path::new(&name).file_stem().unwrap().to_string_lossy().to_string();
            let d: Value = serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
            out.push((stem, d));
        }
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

/// Compare only the fields the golden carries for a row, so a field the Python records and this
/// does not (`text`, `json` in full) is a gap rather than a failure — and is named here.
fn same(got: &Value, want: &Value, where_: &str) {
    for field in ["reason", "read", "out_tokens", "prompt_tokens", "thought", "text_chars",
                  "text_sha", "json_sha"] {
        assert_eq!(&got[field], &want[field], "{where_} / {field}");
    }
}

#[test]
fn the_shipped_stops_do_what_the_golden_says() {
    let g = golden();
    let ship = &g[SHIPPED];
    let mut n = 0;
    for (stem, d) in corpora() {
        let rows = d["rows"].as_array().expect("rows");
        let want_here = &ship[&stem];
        assert!(!want_here.is_null(), "the golden has no section for {stem}");
        for row in rows {
            let key = row["key"].as_str().unwrap();
            let lines = row["lines"].as_array().unwrap();
            let got = outcome(lines, true, true);
            same(&got, &want_here[key], &format!("{stem}/{key}"));
            n += 1;
        }
    }
    assert_eq!(n, 128, "the corpus changed size; 128 replies were recorded");
}

/// The second golden section exists so `GRACE` is pinned by something. With the loop stop on, it
/// reaches the two counting replies first and the answered stop never fires at all — **`closed` is
/// 0 of 128 on the shipped arm** — so without this arm the constant would be held by nothing.
#[test]
fn the_answered_stop_alone_does_what_the_golden_says() {
    let g = golden();
    let want = g[ANSWERED_ONLY].as_object().expect("the second section");
    assert!(!want.is_empty(), "the arm that pins GRACE has emptied");
    let mut seen = 0;
    for (stem, d) in corpora() {
        for row in d["rows"].as_array().unwrap() {
            let key = row["key"].as_str().unwrap();
            let full = format!("{stem}/{key}");
            let Some(expected) = want.get(&full) else { continue };
            let got = outcome(row["lines"].as_array().unwrap(), true, false);
            same(&got, expected, &full);
            seen += 1;
        }
    }
    assert_eq!(seen, want.len(), "a row the golden lists was not found in the corpora");
}

/// **The stops have to be reachable, or this file is asserting that nothing happens.**
#[test]
fn the_corpus_actually_exercises_both_stops() {
    let g = golden();
    let mut reasons = std::collections::BTreeMap::new();
    for (_stem, rows) in g[SHIPPED].as_object().unwrap() {
        for (_key, row) in rows.as_object().unwrap() {
            *reasons.entry(row["reason"].as_str().unwrap().to_string()).or_insert(0) += 1;
        }
    }
    assert_eq!(reasons.get("degenerate"), Some(&3), "the loop stop stopped firing");
    assert_eq!(reasons.get("length"), Some(&3), "nothing reaches the ceiling any more");
    //: `closed` is legitimately 0 here — this model emits nothing after its closing brace, so the
    //: answered stop is unreachable on an ordinary turn. That is why the second arm exists.
    assert_eq!(reasons.get("closed"), None);
    let alone: Vec<_> = g[ANSWERED_ONLY].as_object().unwrap().values()
        .filter(|r| r["reason"] == "closed").collect();
    assert_eq!(alone.len(), 2, "the answered stop is no longer reachable even alone");
}

/// **Read with both stops off, against what the server actually produced.**
///
/// Every recorded row carries a `recorded` block — the reason, length and sha of the reply as the
/// live call returned it, with the stops off so nothing was cut. Replaying with the stops off has
/// to reproduce it, and that is a stronger check than the golden: the golden is what the Python
/// replay produces, this is what Ollama produced.
///
/// It also closes a hole the mutants found in this file. With only the two stopping arms above,
/// **ignoring `stop_when_answered` entirely changed nothing**, because both arms have it on. An arm
/// where the flag is off is what makes the flag mean something.
#[test]
fn with_the_stops_off_it_reproduces_the_live_call() {
    let mut n = 0;
    for (stem, d) in corpora() {
        for row in d["rows"].as_array().unwrap() {
            let key = row["key"].as_str().unwrap();
            let recorded = &row["recorded"];
            let mut reading = Reading::new(false, false);
            for entry in row["lines"].as_array().unwrap() {
                match entry {
                    Value::String(s) => { reading.line(0, Some(s), None); }
                    Value::Object(o) if o.contains_key("blank") => reading.blank(),
                    Value::Object(o) => {
                        let done = o.get("done").map(|x| (
                            x.get("prompt_eval_count").and_then(|v| v.as_i64()),
                            x.get("eval_count").and_then(|v| v.as_i64()),
                            x.get("done_reason").and_then(|v| v.as_str()).unwrap_or("").to_string()));
                        reading.line(o.get("t").and_then(|v| v.as_u64()).unwrap_or(0) as usize,
                                     o.get("c").and_then(|v| v.as_str()), done);
                    }
                    _ => {}
                }
            }
            let reply = reading.finish();
            let where_ = format!("{stem}/{key}");
            assert_eq!(reply.reason, recorded["reason"].as_str().unwrap(), "{where_} reason");
            assert_eq!(reply.text.chars().count(),
                       recorded["text_chars"].as_u64().unwrap() as usize, "{where_} text_chars");
            assert_eq!(sha16(&reply.text), recorded["text_sha"].as_str().unwrap(),
                       "{where_} text_sha");
            n += 1;
        }
    }
    assert_eq!(n, 128);
}
