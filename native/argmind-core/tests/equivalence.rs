//! Every check the four spikes made, in one place, so `cargo test` runs them all.
//!
//! The expectations are produced by the shipped Python — `python native/expectations/all.py` —
//! because a port needs both implementations while it is a port. Each test says what it would fail
//! to notice, because a check whose blind spots are not written down is read as covering more than
//! it does.

use std::fs;
use std::path::PathBuf;

use argmind_core::contract::{Eyes, Hands, Speaks, Widget};
use argmind_core::converse::{render, MOST_AT_ONCE};
use argmind_core::intent::{Done, Effect, Trigger};
use argmind_core::llm::first_object;
use argmind_core::session::{Found, describe, recent, Session};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

fn load(name: &str) -> Value {
    let path: PathBuf = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap().join("expectations").join(name);
    let raw = fs::read_to_string(&path).unwrap_or_else(|_| {
        panic!("{} is missing. Generate it: python native/expectations/all.py", path.display())
    });
    serde_json::from_str(&raw).expect("parse")
}

fn widget_of(v: &Value) -> Widget {
    let r = v["rect"].as_array().unwrap();
    Widget::new(v["name"].as_str().unwrap(), v["role"].as_str().unwrap(),
                (r[0].as_i64().unwrap(), r[1].as_i64().unwrap(),
                 r[2].as_i64().unwrap(), r[3].as_i64().unwrap()),
                v.get("window").and_then(|w| w.as_str()).unwrap_or("Win"))
}

fn pairs(v: &Value) -> Vec<(String, String)> {
    v.as_array().unwrap().iter()
        .map(|p| (p[0].as_str().unwrap().to_string(), p[1].as_str().unwrap().to_string()))
        .collect()
}

fn effect_json(e: &Effect) -> Value {
    json!({"kind": e.kind, "target": e.target, "role": e.role, "text": e.text,
           "unprompted": e.unprompted()})
}

fn widget_json(w: &Widget) -> Value {
    json!({"name": w.name, "role": w.role, "rect": [w.rect.0, w.rect.1, w.rect.2, w.rect.3],
           "window": w.window})
}

// ---- 1. the prompt -------------------------------------------------------------------------------

/// **What goes out is identical, character for character.**
///
/// Would not notice: a change to what `converse()` does with the reply *after* it arrives, since
/// none of that is sent; or a branch of the assembly the six probes do not reach.
#[test]
fn the_prompt_hashes_to_the_shipped_tag() {
    let d = load("prompt.json");
    //: A shrunken expectation file would pass every assertion below while covering nothing.
    assert_eq!(d["probes"].as_array().unwrap().len(), 6, "the probe set changed size");
    let mut blocks = vec![d["header"].as_str().unwrap().to_string()];
    let mut texts: Vec<String> = Vec::new();
    for probe in d["probes"].as_array().unwrap() {
        let widgets: Vec<Widget> =
            probe["widgets"].as_array().unwrap().iter().map(widget_of).collect();
        let trigger = Trigger { text: probe["text"].as_str().unwrap().into(),
                                asked: probe["asked"].as_bool().unwrap() };
        //: **At rest.** This assertion is what makes `mind.rs` safe to have added: the
        //: bytes and the tag below were recorded from the Python, long before a dial
        //: existed, so the prompt still hashing to them is a check against an outside
        //: record rather than against itself.
        let text = render(&widgets, &trigger, &pairs(&probe["history"]),
                          d["max_listed"].as_u64().unwrap() as usize,
                          &argmind_core::mind::Mind::resting());
        blocks.push(format!("--- {} ---\n{}\n", probe["label"].as_str().unwrap(), text));
        texts.push(text);
    }
    //: The same refusal the Python record makes: two identical renders mean a branch this claims to
    //: cover has stopped being reached, and the tag would go on looking healthy.
    for i in 0..texts.len() {
        for j in (i + 1)..texts.len() {
            assert_ne!(texts[i], texts[j], "probes {i} and {j} render identically");
        }
    }
    let record = blocks.join("\n");
    let tag: String = format!("{:x}", Sha256::digest(record.as_bytes())).chars().take(6).collect();
    assert_eq!(record.as_bytes().len(), d["record_bytes"].as_u64().unwrap() as usize);
    assert_eq!(format!("{}{}", d["scheme"].as_str().unwrap(), tag), d["tag"].as_str().unwrap());
    assert_eq!(MOST_AT_ONCE, d["most_at_once"].as_u64().unwrap() as usize);
}

// ---- 2. the reply scanner ------------------------------------------------------------------------

/// **The same object is found in every recorded reply.** 128 real ones and 10 synthetic, the
/// synthetic ones because the corpus is provably blind to braces and escapes inside strings — no
/// recorded reply has one, and a corpus cannot fabricate one.
///
/// Would not notice: a server that splits a token stream into different lines, since this replays
/// bytes rather than asking for new ones.
#[test]
fn the_scanner_finds_the_same_object() {
    let cases = load("reply.json");
    assert!(cases.as_array().unwrap().len() >= 138, "the corpus shrank");
    let mut wide = 0;
    for case in cases.as_array().unwrap() {
        let text = case["text"].as_str().unwrap();
        if text.len() != text.chars().count() {
            wide += 1;
        }
        let mut scan = argmind_core::llm::Balanced::new();
        scan.feed(text);
        assert_eq!((scan.start, scan.end),
                   (case["start"].as_i64().unwrap(), case["end"].as_i64().unwrap()),
                   "indices for {}", case["key"].as_str().unwrap());
        assert_eq!(first_object(text).is_some(), case["reply_json_ok"].as_bool().unwrap(),
                   "parsing for {}", case["key"].as_str().unwrap());

        //: Fed one character at a time — the harshest split — because a streaming caller must land
        //: on the same two numbers as one that hands over the finished text.
        let mut piece = argmind_core::llm::Balanced::new();
        for ch in text.chars() {
            piece.feed(&ch.to_string());
        }
        assert_eq!((piece.start, piece.end), (scan.start, scan.end),
                   "streamed differs for {}", case["key"].as_str().unwrap());
    }
    assert!(wide > 90, "the corpus lost its non-ASCII replies, which are what catch byte indexing");
}

// ---- 3. the pure half of the loop ----------------------------------------------------------------

/// The wording, the window and the resolution. **Three of the four history channels have never
/// been fed by any harness**, which is why the whole line each writes is compared and not just the
/// phrase inside it.
#[test]
fn the_history_the_window_and_the_resolution_match() {
    let d = load("loop.json");
    assert_eq!((d["describe"].as_array().unwrap().len(), d["recent"].as_array().unwrap().len(),
                d["find"].as_array().unwrap().len(), d["channels"].as_array().unwrap().len()),
               (17, 17, 11, 20), "the scene data changed size");

    for case in d["describe"].as_array().unwrap() {
        let e = effect_from(&case["effect"]);
        let widget = if case["widget"].is_null() { None } else { Some(widget_of(&case["widget"])) };
        assert_eq!(describe(&e, widget.as_ref(), case["form"].as_u64().unwrap() as usize),
                   case["want"].as_str().unwrap(), "{}", case["label"].as_str().unwrap());
    }

    for case in d["recent"].as_array().unwrap() {
        let turns = if case["turns"].is_null() { None } else { case["turns"].as_i64() };
        assert_eq!(recent(&pairs(&case["history"]), turns), pairs(&case["want"]),
                   "recent(turns={turns:?})");
    }

    //: **Two of these eleven diverge from the Python on purpose, since 2026-09-28**, and they are
    //: named rather than skipped. The Python answered every duplicate case with *something*; two of
    //: those answers were guesses, and `Found::Several` is the Rust declining to make them.
    //:
    //: The rule is the one the old comment on `Session::find` already contained: the tie-break
    //: anchors on the first same-named widget in the **previous** reading, so it means something
    //: only when the previous reading held exactly one. This data is what confirms the rule rather
    //: than merely permitting it — the two cases it reclassifies are the two where the previous
    //: reading held **none or several**, and the two duplicate cases where it held one still pass
    //: unchanged. **The expectations file is not edited**: it is what the Python produced, it is
    //: evidence, and evidence that moves to agree with the code is not evidence.
    const GUESSES: [&str; 2] = ["the Font : trap, name alone", "duplicates, nothing ever seen"];
    let mut declined = 0;
    for case in d["find"].as_array().unwrap() {
        let fresh: Vec<Widget> = case["fresh"].as_array().unwrap().iter().map(widget_of).collect();
        let seen: Vec<Widget> = case["seen"].as_array().unwrap().iter().map(widget_of).collect();
        let label = case["label"].as_str().unwrap();
        let mut s = session(vec![fresh], &[], false, None, true);
        s.seen = seen;
        let got = s.find(case["name"].as_str().unwrap(), case["role"].as_str().unwrap());
        if GUESSES.contains(&label) {
            //: **Asserted, not excused.** A case in this list that stops being ambiguous means the
            //: rule moved, and the run has to say so rather than passing quietly.
            declined += 1;
            match got {
                Found::Several(same) => assert!(same.len() > 1, "{label}: Several of one"),
                other => panic!("{label}: the Python guessed here and this should decline, \
                                 got {other:?}"),
            }
            continue;
        }
        match (got, case["want"].is_null()) {
            (Found::Gone, true) => {}
            (Found::One(w), false) => assert_eq!(widget_json(&w), case["want"], "{label}"),
            (g, _) => panic!("{label}: got {g:?}"),
        }
    }
    assert_eq!(declined, GUESSES.len(),
               "a case named as a Python guess is no longer in the data; the divergence recorded \
                here has lost its subject");

    for case in d["channels"].as_array().unwrap() {
        let e = effect_from(&case["effect"]);
        let target = if case["target"].is_null() { None } else { Some(widget_of(&case["target"])) };
        let mut s = session(vec![vec![]], &[], false, None, true);
        match case["channel"].as_str().unwrap() {
            "did" => s.remember_doing(&[Done { effect: e, target, confirmed: None }], &[], &[], &[]),
            "refused" => s.remember_doing(&[], &[e], &[], &[]),
            "offered" => s.remember_doing(&[], &[], &[e], &[]),
            _ => s.remember_doing(&[], &[], &[], &[e]),
        }
        assert_eq!(s.history, pairs(&case["want"]), "{}", case["label"].as_str().unwrap());
    }

    let mut s = session(vec![vec![]], &[], false, None, true);
    s.remember_doing(
        &[Done { effect: click("Cut", "button"),
                 target: Some(Widget::new("Cut", "button", (10, 20, 40, 30), "Win")),
                 confirmed: None }],
        &[typing("bye")], &[click("Copy", "button")], &[keying("tab")]);
    assert_eq!(s.history, pairs(&d["order"]["want"]), "the write order within one call");
}

fn effect_from(v: &Value) -> Effect {
    Effect { kind: v["kind"].as_str().unwrap().into(), target: v["target"].as_str().unwrap().into(),
             role: v["role"].as_str().unwrap().into(), text: v["text"].as_str().unwrap().into(),
             because_asked: None, which: None, inside: None }
}
fn click(target: &str, role: &str) -> Effect {
    Effect { kind: "click".into(), target: target.into(), role: role.into(), text: String::new(),
             because_asked: None, which: None, inside: None }
}
fn typing(text: &str) -> Effect {
    Effect { kind: "type".into(), target: String::new(), role: String::new(), text: text.into(),
             because_asked: None, which: None, inside: None }
}
fn keying(text: &str) -> Effect {
    Effect { kind: "key".into(), target: String::new(), role: String::new(), text: text.into(),
             because_asked: None, which: None, inside: None }
}

// ---- 4. a whole turn, across the seam ------------------------------------------------------------

struct ScriptedEyes {
    screens: Vec<Vec<Widget>>,
    reads: usize,
}
impl Eyes for ScriptedEyes {
    fn foreground(&mut self) -> Vec<Widget> {
        let i = self.reads.min(self.screens.len() - 1);
        self.reads += 1;
        self.screens[i].clone()
    }
    fn timed_foreground(&mut self) -> (Vec<Widget>, f64) {
        (self.foreground(), 0.125)
    }
}

struct ScriptedHands {
    live: bool,
    names: Option<Vec<String>>,
    done: Vec<String>,
}
impl Hands for ScriptedHands {
    fn live(&self) -> bool {
        self.live
    }
    fn click(&mut self, w: &Widget) {
        let c = w.centre();
        self.done.push(format!("click {} '{}' at ({}, {})", w.role, w.name, c.0, c.1));
    }
    fn type_text(&mut self, text: &str) {
        self.done.push(format!("type '{text}'"));
    }
    fn press(&mut self, key: &str) {
        //: Refuses a key it never published, so the unknown-key filter fails on the press rather
        //: than on an assertion nobody wrote.
        if let Some(names) = &self.names {
            assert!(names.iter().any(|n| n == key), "the core pressed {key:?}, refused here");
        }
        self.done.push(format!("key '{key}'"));
    }
    fn keys(&self) -> Option<Vec<String>> {
        self.names.clone()
    }
}

struct ScriptedModel {
    texts: Vec<String>,
}
impl Speaks for ScriptedModel {
    fn ask(&mut self, _prompt: &str) -> String {
        if self.texts.is_empty() { r#"{"say": "", "do": []}"#.into() } else { self.texts.remove(0) }
    }
}

fn session(screens: Vec<Vec<Widget>>, replies: &[String], live: bool, keys: Option<Vec<String>>,
           may_act: bool) -> Session<ScriptedEyes, ScriptedHands, ScriptedModel> {
    let mut s = Session::new(ScriptedEyes { screens, reads: 0 },
                             ScriptedHands { live, names: keys, done: Vec::new() },
                             ScriptedModel { texts: replies.to_vec() });
    s.may_act_unprompted = may_act;
    s
}

/// Every field of every turn: the exchange, the history, the three counters, what the hands were
/// told, and **how many times the eyes were read** — up to three a turn, which is the design's
/// point and the port's whole cost.
///
/// Would not notice: timings, which are the environment's number and the wall clock.
#[test]
fn a_whole_turn_produces_the_same_everything() {
    let cases = load("session.json");
    let cases = cases.as_array().unwrap();
    assert_eq!(cases.len(), 21, "the scenario set changed size");
    let mut turns = 0;
    for case in cases {
        let screens: Vec<Vec<Widget>> = case["screens"].as_array().unwrap().iter()
            .map(|s| s.as_array().unwrap().iter().map(widget_of).collect()).collect();
        let replies: Vec<String> = case["replies"].as_array().unwrap().iter()
            .map(|t| t.as_str().unwrap().to_string()).collect();
        let keys = case["hands"]["keys"].as_array()
            .map(|a| a.iter().map(|k| k.as_str().unwrap().to_string()).collect());
        let mut s = session(screens, &replies, case["hands"]["live"].as_bool().unwrap(), keys,
                            case["may_act_unprompted"].as_bool().unwrap());
        for (i, t) in case["turns"].as_array().unwrap().iter().enumerate() {
            turns += 1;
            let ex = s.think(&Trigger { text: t["text"].as_str().unwrap().into(),
                                        asked: t["asked"].as_bool().unwrap() });
            let want = &case["want"][i];
            let label = case["label"].as_str().unwrap();
            assert_eq!(json!(ex.n), want["n"], "{label} n");
            assert_eq!(json!(ex.trigger.asked), want["asked"], "{label} asked");
            assert_eq!(json!(ex.said), want["said"], "{label} said");
            assert_eq!(json!(ex.n_widgets), want["n_widgets"], "{label} n_widgets");
            assert_eq!(json!(ex.did.iter().map(|d| json!({
                "effect": effect_json(&d.effect), "confirmed": d.confirmed,
                "target": d.target.as_ref().map(widget_json)})).collect::<Vec<_>>()),
                want["did"], "{label} did");
            assert_eq!(json!(ex.proposed.iter().map(effect_json).collect::<Vec<_>>()),
                       want["proposed"], "{label} proposed");
            assert_eq!(json!(ex.refused.iter().map(effect_json).collect::<Vec<_>>()),
                       want["refused"], "{label} refused");
            assert_eq!(s.history, pairs(&want["history"]), "{label} history");
            assert_eq!(json!(s.blind), want["blind"], "{label} blind");
            assert_eq!(json!(s.vanished), want["vanished"], "{label} vanished");
            assert_eq!(json!(s.unknown_keys), want["unknown_keys"], "{label} unknown_keys");
            assert_eq!(json!(s.hands.done), want["hands_done"], "{label} hands_done");
            assert_eq!(json!(s.eyes.reads), want["eye_reads"], "{label} eye_reads");
        }
    }
    assert_eq!(turns, 23, "the turn count changed");
}
