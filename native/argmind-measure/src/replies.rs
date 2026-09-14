//! Real model replies kept **chunk by chunk**, so the stop logic can be replayed offline.
//!
//! **The record is a list of stream lines and not a string, and that is the whole design.** The
//! stops do not run on the finished text: `Reading` consumes chunk by chunk and `GRACE` counts
//! *chunks* after the envelope closes, not characters — so **the same final text split differently
//! stops in a different place**, and a record of the joined text could not exercise `GRACE` at all.
//! `arrived` counts lines where thinking **or** content was non-empty, so whether a line carried
//! reasoning matters even when it carried no words; and only the final `done` line carries the
//! token counts.
//!
//! **Recorded with the stops OFF.** With them on the stream is cut at the stop and everything after
//! it is never seen — the corpus would then hold exactly the prefixes the shipped logic already
//! produces and could never test the logic it exists for. The flags are in the filename.
//!
//! **What is not stored, and why each is safe.** The reasoning *text* becomes its length: the
//! reader takes `len(reasoning)` and its truthiness and nothing else, so a substitute of the same
//! length is behaviourally identical. The prompt becomes its length and its sha, with enough
//! labelling to render it again. The joined text becomes its length, its sha and its first 120
//! characters — storing it as well as the chunks would let two copies of one fact disagree.
//!
//! **No rate may be computed from this file.** The interesting rows are the ones that broke
//! something, they are rare, and the scenes over-sample them on purpose.

use std::collections::BTreeMap;

use argmind_core::llm::{Reading, Reply};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::python_json;

/// The recording configuration, named because a corpus taken without it is worthless and the one
/// mistake that produces one is forgetting a flag.
pub const STOPS_OFF: (bool, bool) = (false, false);

pub fn sha16(text: &str) -> String {
    format!("{:x}", Sha256::digest(text.as_bytes())).chars().take(16).collect()
}

/// One stream line as the reader sees it: how much thinking, what content, and whether it is the
/// last.
///
/// `content` is `None` for a line the reader skips before parsing — a blank one — which is a
/// different thing from a line that carried an empty string, and **the two count differently
/// toward `arrived`**.
#[derive(Debug, Clone, PartialEq)]
pub struct Seen {
    pub thinking: usize,
    pub content: Option<String>,
    pub done: bool,
}

pub fn project(line: &str) -> Seen {
    if line.trim().is_empty() {
        return Seen { thinking: 0, content: None, done: false };
    }
    let d: Value = serde_json::from_str(line).expect("a stream line that is not JSON");
    let msg = &d["message"];
    Seen {
        //: **Characters, not bytes.** The reader counts `len(reasoning)` over a string, and
        //: reasoning is not ASCII.
        thinking: msg.get("thinking").and_then(|v| v.as_str()).unwrap_or("").chars().count(),
        content: Some(msg.get("content").and_then(|v| v.as_str()).unwrap_or("").to_string()),
        done: d.get("done").and_then(|v| v.as_bool()).unwrap_or(false),
    }
}

/// Twelve characters over the per-line projection of a whole stream.
///
/// **This is what pins the chunk boundaries.** Move the boundary between two adjacent content
/// chunks and the text, both token counts, the thinking length, the line count and `arrived` are
/// all exactly as they were — while `GRACE`, which counts chunks, stops somewhere else. So the one
/// quantity a corpus must not lose is the one nothing else here would notice losing.
///
/// A *join* is the louder cousin: it loses a line too, and the stored line count catches it alone.
pub fn stream_print(lines: &[String]) -> String {
    let rows: Vec<Value> = lines.iter().map(|l| {
        let s = project(l);
        json!([s.thinking, s.content, s.done])
    }).collect();
    format!("{:x}", Sha256::digest(python_json(&json!(rows)).as_bytes())).chars().take(12).collect()
}

/// One stream line, reduced to what the reader reads.
///
/// A bare string is the common case — a content chunk and nothing else on the line — so the common
/// case costs its own characters and two quotes. Everything else is an object.
pub fn encode(line: &str) -> Value {
    if line.trim().is_empty() {
        return json!({"blank": true});
    }
    let d: Value = serde_json::from_str(line).expect("a stream line that is not JSON");
    let msg = &d["message"];
    let thinking = msg.get("thinking").and_then(|v| v.as_str()).unwrap_or("");
    let content = msg.get("content").and_then(|v| v.as_str()).unwrap_or("");
    let done = d.get("done").and_then(|v| v.as_bool()).unwrap_or(false);
    if thinking.is_empty() && !done {
        return json!(content);
    }
    let mut entry = serde_json::Map::new();
    if !thinking.is_empty() {
        //: **The length, not the text.** The reader takes `len(reasoning)` and its truthiness and
        //: nothing else; reasoning is 3,854 characters a call on the thinking arm and would
        //: otherwise be most of the corpus by weight.
        entry.insert("t".into(), json!(thinking.chars().count()));
    }
    if !content.is_empty() {
        entry.insert("c".into(), json!(content));
    }
    if done {
        entry.insert("done".into(), json!({
            "done_reason": d.get("done_reason").cloned().unwrap_or(Value::Null),
            "prompt_eval_count": d.get("prompt_eval_count").cloned().unwrap_or(Value::Null),
            "eval_count": d.get("eval_count").cloned().unwrap_or(Value::Null)}));
    }
    Value::Object(entry)
}

/// One entry back into a line the real reader parses.
///
/// The inverse of [`encode`] **as far as the reader can tell**, which is the only sense in which
/// anything here claims to be an inverse: the reasoning comes back as filler of the right length,
/// and the escaping of the re-serialised line is not the server's. Neither is visible to a reader
/// that parses the line and reads three fields off it — which is what [`check`] verifies rather
/// than assumes.
pub fn decode(entry: &Value) -> String {
    if let Some(s) = entry.as_str() {
        return json!({"message": {"role": "assistant", "content": s}, "done": false}).to_string();
    }
    if entry.get("blank").and_then(|v| v.as_bool()).unwrap_or(false) {
        return "\n".into();
    }
    let mut msg = serde_json::Map::new();
    msg.insert("role".into(), json!("assistant"));
    msg.insert("content".into(),
               json!(entry.get("c").and_then(|v| v.as_str()).unwrap_or("")));
    if let Some(t) = entry.get("t").and_then(|v| v.as_u64()).filter(|t| *t > 0) {
        msg.insert("thinking".into(), json!("x".repeat(t as usize)));
    }
    let mut out = serde_json::Map::new();
    out.insert("message".into(), Value::Object(msg));
    let done = entry.get("done").filter(|d| d.is_object());
    out.insert("done".into(), json!(done.is_some()));
    if let Some(Value::Object(fields)) = done {
        for (k, v) in fields {
            out.insert(k.clone(), v.clone());
        }
    }
    Value::Object(out).to_string()
}

/// How many lines carried thinking or content — what `out_tokens` falls back to when a stop fires
/// and the server's final message never arrives.
pub fn arrived(lines: &[String]) -> usize {
    lines.iter().filter(|l| {
        let s = project(l);
        s.thinking > 0 || s.content.as_deref().is_some_and(|c| !c.is_empty())
    }).count()
}

/// Play a recorded stream back to the reader as if it were the server, counting what was read.
pub fn replay(entries: &[Value], answered: bool, looping: bool) -> (Reply, usize) {
    let mut reading = Reading::new(answered, looping);
    for entry in entries {
        let line = decode(entry);
        let keep = if line.trim().is_empty() {
            reading.blank();
            true
        } else {
            let d: Value = serde_json::from_str(&line).unwrap();
            let msg = &d["message"];
            let done = if d["done"].as_bool().unwrap_or(false) {
                Some((d.get("prompt_eval_count").and_then(|v| v.as_i64()),
                      d.get("eval_count").and_then(|v| v.as_i64()),
                      d.get("done_reason").and_then(|v| v.as_str()).unwrap_or("").to_string()))
            } else {
                None
            };
            reading.line(msg.get("thinking").and_then(|v| v.as_str()).unwrap_or("").chars().count(),
                         msg.get("content").and_then(|v| v.as_str()), done)
        };
        if !keep {
            break;
        }
    }
    let read = reading.lines_read;
    (reading.finish(), read)
}

/// Replay a row just recorded and **refuse it** if it does not reproduce the call.
///
/// **Three things, because a codec can lose exactly one of them at a time.** The text and the token
/// counts catch a dropped or garbled chunk and a thinking line recorded as a content one; the line
/// count catches a dropped line, and a join with it; and `stream_print` catches a **moved
/// boundary**, which is the only loss that changes nothing else at all and still moves where
/// `GRACE` stops.
pub fn check(row: &Value) -> Result<(), String> {
    let key = row["key"].as_str().unwrap_or("?");
    let entries: Vec<Value> = row["lines"].as_array().cloned().unwrap_or_default();
    let was = &row["recorded"];
    let (reply, read) = replay(&entries, STOPS_OFF.0, STOPS_OFF.1);
    let got = json!({"reason": reply.reason, "text_chars": reply.text.chars().count(),
                     "text_sha": sha16(&reply.text), "prompt_tokens": reply.prompt_tokens,
                     "out_tokens": reply.out_tokens, "thought": reply.thought});
    let mine: Value = json!(got.as_object().unwrap().keys()
        .map(|k| (k.clone(), was[k].clone())).collect::<serde_json::Map<_, _>>());
    if got != mine {
        return Err(format!("{key}: the replay does not reproduce the call. \
                            recorded {mine}, replayed {got}"));
    }
    let want_lines = was["stream_lines"].as_u64().unwrap_or(0) as usize;
    if read != want_lines || entries.len() != want_lines {
        return Err(format!("{key}: {want_lines} lines came off the socket, {} were stored, \
                            {read} were replayed", entries.len()));
    }
    let again = stream_print(&entries.iter().map(decode).collect::<Vec<_>>());
    if Some(again.as_str()) != was["stream_print"].as_str() {
        return Err(format!("{key}: the chunk boundaries did not survive the round trip — {} \
                            recorded against {again} decoded", was["stream_print"]));
    }
    Ok(())
}

/// Which shapes the corpus reaches, counted by replaying each row through the **shipped** stops.
///
/// Reported rather than asserted per file: one arm is not expected to reach all of them, and it is
/// the union across the arms that has to.
pub fn coverage(rows: &[Value]) -> BTreeMap<String, usize> {
    let mut got: BTreeMap<String, usize> = BTreeMap::new();
    let mut bump = |k: String| *got.entry(k).or_default() += 1;
    for row in rows {
        let entries: Vec<Value> = row["lines"].as_array().cloned().unwrap_or_default();
        let (shipped, _) = replay(&entries, true, true);
        bump(format!("shipped: {}", shipped.reason));
        let was = &row["recorded"];
        if was["reason"] == "length" {
            bump("ran to the ceiling with the stops off".into());
        }
        if argmind_core::llm::first_object(&shipped.text).is_none() {
            bump("no object to parse".into());
        }
        if was["thought"].as_u64().unwrap_or(0) > 0 {
            bump("carried reasoning".into());
        }
        if was["arrived"].as_i64().unwrap_or(0) < was["stream_lines"].as_i64().unwrap_or(0) - 1 {
            bump("a line carrying neither".into());
        }
    }
    got
}

/// Prompts asked more than once, and whether the answer came back the same.
///
/// **The worst of N is not a summary this file can print** — it stores replies, not verdicts — so
/// what it owes a reader instead is how many repeats of one prompt disagreed. A ceiling that fires
/// on one draw in three is exactly the thing a single draw would report wrongly either way.
pub fn disagreement(rows: &[Value]) -> Vec<Value> {
    let mut by: BTreeMap<String, Vec<&Value>> = BTreeMap::new();
    for row in rows {
        let key = format!("{}/{}/d{}/t{} {}", row["case"], row["arm"], row["distance"],
                          row["turn"], row["said"]);
        by.entry(key).or_default().push(row);
    }
    let mut out = Vec::new();
    for (_, group) in by {
        if group.len() < 2 {
            continue;
        }
        let prints: std::collections::BTreeSet<&str> = group.iter()
            .filter_map(|r| r["recorded"]["stream_print"].as_str()).collect();
        let first = group[0];
        out.push(json!({
            "prompt": format!("{}/{}/d{}/t{} {}",
                              first["case"].as_str().unwrap_or("-"),
                              first["arm"].as_str().unwrap_or("-"),
                              first["distance"], first["turn"],
                              first["said"].as_str().unwrap_or("")),
            "said": first["said"],
            "n": group.len(),
            "identical_streams": prints.len() == 1,
            "reasons": group.iter().map(|r| r["recorded"]["reason"].clone()).collect::<Vec<_>>(),
            "out_tokens": group.iter().map(|r| r["recorded"]["out_tokens"].clone())
                .collect::<Vec<_>>(),
        }));
    }
    out
}
