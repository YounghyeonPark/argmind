//! The recorded corpus, re-derived: the codec, the chunk-boundary fingerprint, and the reading.
//!
//! 128 real replies over four arms, recorded with the stops **off** so nothing was cut. This is the
//! only evidence the stop logic has ever had against real output — every other harness here stores
//! what a reply *meant* and none stored the reply.
//!
//! `native/argmind-core/tests/stops.rs` already replays these streams against the golden. What is
//! checked here is the layer under that: **whether the stored form is still a faithful record.**
//! Moving the boundary between two adjacent content chunks changes the text not at all, the token
//! counts not at all, the thinking length not at all, the line count not at all and `arrived` not
//! at all — and moves where `GRACE` stops, because `GRACE` counts chunks. Only `stream_print`
//! notices, which is why every row carries one and why `check` recomputes it through a full
//! encode/decode round trip.
//!
//! That is a correction to the wording this port was written against, which said *joining* two
//! chunks was the loss nothing else notices. A join also loses a line, and the stored line count
//! catches it on its own; the boundary move is the one that gets past everything else.

use std::path::PathBuf;

use argmind_measure::replies::{arrived, check, coverage, decode, disagreement, encode, project,
                               replay, sha16, stream_print, Seen, STOPS_OFF};
use serde_json::{json, Value};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap().to_path_buf()
}

fn want() -> Value {
    let p = root().join("native").join("expectations").join("replies.json");
    serde_json::from_str(&std::fs::read_to_string(&p).unwrap_or_else(|_| {
        panic!("{} is missing. Generate it: python native/expectations/all.py", p.display())
    })).unwrap()
}

fn corpora() -> Vec<(String, Value)> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(root().join("measure")).unwrap().flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with("replies-") && name.ends_with(".json") {
            out.push((name, serde_json::from_str(
                &std::fs::read_to_string(entry.path()).unwrap()).unwrap()));
        }
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

/// **Every row's own integrity check, re-run.** The recorder refused to write a row that failed
/// this; the port has to reach the same verdict on all 128.
#[test]
fn every_recorded_row_still_reproduces_the_call_it_recorded() {
    let files = corpora();
    assert!(files.len() >= 4, "only {} corpora found", files.len());
    let mut rows = 0;
    let mut reasons: std::collections::BTreeMap<String, usize> = Default::default();
    for (name, d) in &files {
        for row in d["rows"].as_array().unwrap() {
            check(row).unwrap_or_else(|e| panic!("{name}: {e}"));
            *reasons.entry(row["recorded"]["reason"].as_str().unwrap().into()).or_default() += 1;
            rows += 1;
        }
    }
    assert_eq!(rows, 128, "the corpus is {rows} replies, not 128");
    //: **Recorded with the stops off**, so a row that ran to the ceiling is a row that really did.
    //: If none had, the corpus would contain only the easy half and could not test the stops.
    assert!(reasons.contains_key("length"), "nothing in the corpus reached the ceiling: {reasons:?}");
    println!("  {rows} rows re-checked: {reasons:?}");
}

/// The chunk boundaries, and the round trip that has to preserve them.
#[test]
fn the_chunk_boundaries_survive_the_codec() {
    let mut rows = 0;
    let mut bare = 0;
    for (name, d) in &corpora() {
        for row in d["rows"].as_array().unwrap() {
            let entries: Vec<Value> = row["lines"].as_array().unwrap().clone();
            let lines: Vec<String> = entries.iter().map(decode).collect();
            assert_eq!(stream_print(&lines), row["recorded"]["stream_print"].as_str().unwrap(),
                       "{name} / {}", row["key"]);
            //: `arrived` is counted off the wire and is not on `Reply`. It becomes observable only
            //: when a stop fires, so it is re-derived here where every line is still present.
            assert_eq!(arrived(&lines) as i64, row["recorded"]["arrived"].as_i64().unwrap(),
                       "{name} / {} arrived", row["key"]);
            assert_eq!(lines.len() as i64, row["recorded"]["stream_lines"].as_i64().unwrap());
            //: **Re-encoding a decoded entry gives the entry back.** A codec that lost a field on
            //: the way out would still round-trip the projection, so this is the other direction.
            for (entry, line) in entries.iter().zip(&lines) {
                assert_eq!(&encode(line), entry, "{name} / {} re-encode", row["key"]);
                if entry.is_string() {
                    bare += 1;
                }
            }
            rows += 1;
        }
    }
    assert!(rows >= 128);
    //: The common case really is the common case: a content chunk and nothing else on the line.
    assert!(bare > 1000, "only {bare} bare-string entries — the codec's common case is missing");
    println!("  {rows} rows, {bare} bare-string chunk entries round-tripped");
}

/// **The reading a corpus does not carry**: what these streams do under the *shipped* stops, which
/// is a different question from what they were recorded under.
#[test]
fn the_coverage_and_the_disagreements_are_the_ones_python_reports() {
    let d = want();
    let by_file: std::collections::BTreeMap<&str, &Value> = d["files"].as_array().unwrap().iter()
        .map(|f| (f["file"].as_str().unwrap(), f)).collect();
    let mut union: std::collections::BTreeSet<String> = Default::default();
    for (name, run) in &corpora() {
        let f = by_file[name.as_str()];
        let rows: Vec<Value> = run["rows"].as_array().unwrap().clone();
        let got = coverage(&rows);
        let mine: serde_json::Map<String, Value> = got.iter()
            .map(|(k, v)| (k.clone(), json!(v))).collect();
        assert_eq!(&Value::Object(mine), &f["coverage"], "{name} coverage");
        assert_eq!(&json!(disagreement(&rows)), &f["disagreement"], "{name} disagreement");
        union.extend(got.keys().cloned());
    }
    //: One arm is not expected to reach every shape; **the union is.**
    for shape in ["shipped: stop", "carried reasoning", "ran to the ceiling with the stops off"] {
        assert!(union.contains(shape), "no arm reaches {shape:?}: {union:?}");
    }
    println!("  shapes across the four arms: {union:?}");
}

/// A blank line and a line that carried an empty string are **different facts**, and they count
/// differently toward `arrived`.
#[test]
fn a_blank_line_is_not_a_line_that_said_nothing() {
    let blank = project("\n");
    assert_eq!(blank, Seen { thinking: 0, content: None, done: false });
    let empty = project(r#"{"message": {"role": "assistant", "content": ""}, "done": false}"#);
    assert_eq!(empty, Seen { thinking: 0, content: Some(String::new()), done: false });
    assert_ne!(blank, empty);
    //: Neither counts toward `arrived`, and they are still not the same line — the projection keeps
    //: them apart, so `stream_print` does too.
    let as_blank = decode(&json!({"blank": true}));
    let as_empty = decode(&json!({"c": "", "done": {"done_reason": "stop"}}));
    assert_eq!(arrived(&[as_blank.clone(), as_empty.clone()]), 0);
    assert_ne!(stream_print(&[as_blank]), stream_print(&[as_empty]));
}

/// **A moved chunk boundary is the loss nothing else notices**, and it is not the same as a join.
///
/// Joining two chunks also loses a line, so the stored line count catches it. Moving the boundary
/// between them keeps the line count, the text, both token counts, the thinking length **and**
/// `arrived` exactly as they were — and still moves where `GRACE` stops, because `GRACE` counts
/// chunks. That is the case `stream_print` is the only witness to.
#[test]
fn a_moved_chunk_boundary_changes_nothing_except_the_fingerprint() {
    let done = json!({"c": "", "done": {"done_reason": "stop", "prompt_eval_count": 10,
                                        "eval_count": 3}});
    let one: Vec<String> = [json!("he"), json!("llo"), done.clone()].iter().map(decode).collect();
    let other: Vec<String> = [json!("hel"), json!("lo"), done.clone()].iter().map(decode).collect();
    let text = |ls: &[String]| ls.iter()
        .filter_map(|l| project(l).content).collect::<Vec<_>>().join("");
    assert_eq!(text(&one), text(&other));
    assert_eq!(sha16(&text(&one)), sha16(&text(&other)));
    assert_eq!(one.len(), other.len());
    assert_eq!(arrived(&one), arrived(&other));
    //: The only thing that says they are different streams.
    assert_ne!(stream_print(&one), stream_print(&other));

    //: A **join** is a different failure and a louder one: the text still survives it, but a line
    //: is gone and the count says so on its own.
    let joined: Vec<String> = [json!("hello"), done].iter().map(decode).collect();
    assert_eq!(text(&joined), text(&one));
    assert_eq!(joined.len(), one.len() - 1);
    assert_eq!(arrived(&joined), arrived(&one) - 1);
}

/// The reasoning text is not stored, and **nothing the reader does can tell.**
#[test]
fn a_thinking_line_keeps_its_length_and_loses_its_words() {
    let real = r#"{"message": {"role": "assistant", "thinking": "생각 중입니다", "content": ""},
                   "done": false}"#;
    //: Seven characters of Korean, the space included — **characters, not the nineteen bytes.** The reader counts over a
    //: string, and getting this wrong is invisible on ASCII and wrong on everything else.
    assert_eq!(encode(real), json!({"t": 7}));
    assert_eq!(project(&decode(&encode(real))).thinking, 7);
    assert_eq!(project(real).thinking, 7);
    //: A thinking line carries no content, so it counts toward `arrived` anyway — the one way a
    //: line counts while saying nothing.
    assert_eq!(arrived(&[decode(&encode(real))]), 1);
}

/// A row is refused when the replay does not reproduce the call.
#[test]
fn a_row_that_does_not_replay_is_refused_rather_than_stored() {
    let (name, d) = corpora().into_iter().next().unwrap();
    let good = d["rows"][0].clone();
    check(&good).unwrap_or_else(|e| panic!("{name}: {e}"));

    //: A dropped line: the count no longer matches, which is what catches a codec losing one.
    let mut short = good.clone();
    let mut lines = short["lines"].as_array().unwrap().clone();
    lines.pop();
    short["lines"] = json!(lines);
    assert!(check(&short).is_err(), "a dropped line was accepted");

    //: A garbled chunk: the text sha moves.
    let mut wrong = good.clone();
    let mut lines = wrong["lines"].as_array().unwrap().clone();
    lines[0] = json!("not what the model said");
    wrong["lines"] = json!(lines);
    assert!(check(&wrong).is_err(), "a garbled chunk was accepted");

    //: **A moved boundary**: the line count, the text and `arrived` all survive it, and only the
    //: fingerprint says so — this is the case `check` exists for, and the one every other stored
    //: field is blind to.
    let mut moved = good.clone();
    let mut lines = moved["lines"].as_array().unwrap().clone();
    let pair = lines.iter().enumerate().find(|(i, v)| {
        v.as_str().is_some_and(|s| s.chars().count() > 1)
            && lines.get(i + 1).and_then(|n| n.as_str()).is_some_and(|s| !s.is_empty())
    }).map(|(i, _)| i);
    let i = pair.expect("two adjacent non-empty content chunks somewhere in a recorded reply");
    let a: Vec<char> = lines[i].as_str().unwrap().chars().collect();
    let b = lines[i + 1].as_str().unwrap().to_string();
    lines[i] = json!(a[..1].iter().collect::<String>());
    lines[i + 1] = json!(format!("{}{b}", a[1..].iter().collect::<String>()));
    moved["lines"] = json!(lines);
    let (reply, _) = replay(moved["lines"].as_array().unwrap(), STOPS_OFF.0, STOPS_OFF.1);
    assert_eq!(sha16(&reply.text), good["recorded"]["text_sha"].as_str().unwrap(),
               "moving the boundary was supposed to leave the text alone");
    assert_eq!(moved["lines"].as_array().unwrap().len(),
               good["lines"].as_array().unwrap().len());
    assert!(check(&moved).is_err(), "a moved chunk boundary was accepted");
}

/// **What each of `check`'s three comparisons catches that the others do not**, found by mutating
/// them away one at a time. Two survived until this test existed, and they survived for different
/// reasons.
#[test]
fn each_of_the_three_comparisons_catches_something_the_others_cannot() {
    let done = |tokens: i64| json!({"c": "", "done": {"done_reason": "stop",
                                                      "prompt_eval_count": tokens,
                                                      "eval_count": 3}});
    let row = |lines: Vec<Value>, stream_lines: usize, tokens: i64| {
        let text = "hello";
        json!({"key": "made-up", "lines": lines, "recorded": {
            "reason": "stop", "text_chars": 5, "text_sha": sha16(text),
            "prompt_tokens": tokens, "out_tokens": 3, "thought": 0,
            "stream_lines": stream_lines,
            "stream_print": "PLACEHOLDER"}})
    };
    let fix = |mut r: Value| {
        let lines: Vec<String> = r["lines"].as_array().unwrap().iter().map(decode).collect();
        r["recorded"]["stream_print"] = json!(stream_print(&lines));
        r
    };

    //: Sound to begin with, so a failure below is the mutation and not the fixture.
    let good = fix(row(vec![json!("he"), json!("llo"), done(10)], 3, 10));
    check(&good).unwrap();

    //: **The token comparison does work `stream_print` cannot.** The projection carries whether a
    //: line was the last and not what the server counted on it, so a corrupted `prompt_eval_count`
    //: leaves every chunk boundary exactly where it was.
    let mistaken = fix(row(vec![json!("he"), json!("llo"), done(999)], 3, 10));
    assert_eq!(mistaken["recorded"]["stream_print"], good["recorded"]["stream_print"],
               "the corruption was supposed to be invisible to the fingerprint");
    assert!(check(&mistaken).is_err(), "a wrong token count was accepted");

    //: **The line count is a self-consistency check on the row**, and that is all it is: the
    //: fingerprint is computed from the stored lines, so it can only ever agree with them. What
    //: this catches is a row whose *count* disagrees with its own content — the recorder saw four
    //: lines off the socket and stored three.
    let miscounted = fix(row(vec![json!("he"), json!("llo"), done(10)], 4, 10));
    let err = check(&miscounted).unwrap_err();
    assert!(err.contains("4 lines came off the socket"), "{err}");

    //: **And on a real recorder a dropped line moves the fingerprint too**, so for actual recorded
    //: data this comparison cannot fail alone. Worth knowing rather than defending — the mutation
    //: that removes it survives all 128 rows, which is the same shape of finding as a guard that
    //: cannot fire without another one firing first.
    let short = fix(row(vec![json!("he"), done(10)], 2, 10));
    assert_ne!(short["recorded"]["stream_print"], good["recorded"]["stream_print"]);

    //: The reader does **not** stop at the done line — it runs out of stream. So a record with
    //: lines after `done` is a record of a longer reply, and the text comparison says so.
    let trailing = fix(row(vec![json!("he"), json!("llo"), done(10), json!("!")], 4, 10));
    let (r, read) = replay(trailing["lines"].as_array().unwrap(), false, false);
    assert_eq!(r.text, "hello!", "the reader was thought to stop at `done`, and it does not");
    assert_eq!(read, 4);
    assert!(check(&trailing).is_err());
}

/// The scenes, and why they are a table rather than branches.
#[test]
fn the_arms_record_the_scene_lists_their_names_claim() {
    let d = want();
    let comp = d["composition"].as_object().unwrap();
    let arms = d["arms"].as_object().unwrap();
    assert_eq!(comp.len(), arms.len(), "an arm has no composition or the other way round");
    //: **`think = false`, not absent.** Leaving the field out means the model decides, and one
    //: smoke run came back with 4,032 characters of reasoning under a filename that said `off`.
    //: Three values, not two — folding `unset` in with `off` is how a corpus gets named for a
    //: condition it was not recorded under.
    assert_eq!(d["think_label"].as_object().unwrap().len(), 3);
    assert_eq!(d["stops_off"]["stop_when_answered"], json!(false));
    assert_eq!(d["stops_off"]["stop_when_looping"], json!(false));
    for (arm, what) in arms {
        let c = comp[arm].as_array().unwrap();
        assert!(c[0].is_boolean() && c[1].is_i64() && c[2].is_i64(), "{arm}");
        assert!(what[2].as_str().unwrap().len() > 20, "{arm} says nothing about what it is for");
    }
    //: Every conversation scene names a case the distance sweep actually has, so the recorder
    //: renders that harness's prompts and not a second version of them.
    let known: Vec<String> = argmind_measure::conversation::cases().iter()
        .map(|c| c.id.clone()).collect();
    for scene in d["conversations"].as_array().unwrap() {
        let id = scene[0].as_str().unwrap();
        assert!(known.iter().any(|k| k == id), "scene {id:?} is not a case: {known:?}");
        assert!(scene[4].as_str().unwrap().len() > 20, "scene {id} says nothing about why");
    }
    //: **Every stress prompt carries its reason**, because they are the only thing here not drawn
    //: from a measured case list — written to elicit a shape, and whether they work is reported
    //: rather than assumed.
    for s in d["stress"].as_array().unwrap() {
        assert!(s[2].as_str().unwrap().len() > 20);
    }
}
