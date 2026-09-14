//! Stage 3b: a whole turn, and the seam it crosses.
//!
//! `observe`, `poll`, `decide`, `perform` and `think` are where the core stops being arithmetic and
//! starts being an interface. Three of them read the screen, one presses keys, one calls a model —
//! and `Session` reads the screen **up to three times a turn**: once to observe, once to re-resolve
//! immediately before acting, once to confirm the world moved. Those three reads disagreeing is the
//! design's whole point (a listed widget is gone 13 times in 14 over the second a call takes) and
//! it is also the port's whole cost: 152 widgets cross at 0.144 ms through JSON against 0.032 ms
//! for everything the core does in a turn.
//!
//! So this file is as much about **what the seam has to be** as about whether the logic matches.
//! `Eyes`, `Hands` and `Speaks` are traits here for the same reason they are protocols there: the
//! core calls them, and nothing else about the platform reaches it.
//!
//! Verified against `expectations.json`, produced by driving the real `Session` with the golden
//! transcript's own scripted fakes. Everything a turn produces is compared — the `Exchange`'s
//! fields, the history lines, the three counters, what the hands were told, and how many times the
//! eyes were read. Timings are not: `look_s` is the environment's own number and the other two are
//! wall clock, which is not a property of a port.

use std::env;
use std::fs;

use serde_json::{json, Value};

const MAX_LISTED: usize = 200;
const MOST_AT_ONCE: usize = 3;
const NOTHING_DONE: &str = "nothing was done";
const PAST: usize = 0;
const PLAIN: usize = 1;

fn verb(kind: &str, form: usize) -> &'static str {
    let row: [&str; 3] = match kind {
        "click" => ["clicked", "click", "clicking"],
        "type" => ["typed", "type", "typing"],
        _ => ["pressed", "press", "pressing"],
    };
    row[form]
}

// ---- what an environment must provide ------------------------------------------------------------

#[derive(Clone, PartialEq, Debug)]
struct Widget {
    name: String,
    role: String,
    rect: (i64, i64, i64, i64),
    window: String,
}

impl Widget {
    fn centre(&self) -> (i64, i64) {
        let (x, y, w, h) = self.rect;
        (x + w.div_euclid(2), y + h.div_euclid(2))
    }
    /// `Widget.__eq__` is name, role and rect — **not the window**. `_confirm` leans on exactly
    /// that: the same control in a different window still compares equal, which is why the window
    /// is checked separately there.
    fn same(&self, other: &Widget) -> bool {
        self.name == other.name && self.role == other.role && self.rect == other.rect
    }
}

trait Eyes {
    fn foreground(&mut self) -> Vec<Widget>;
    fn timed_foreground(&mut self) -> (Vec<Widget>, f64) {
        (self.foreground(), 0.125)
    }
    fn reads(&self) -> usize;
}

trait Hands {
    fn live(&self) -> bool;
    fn click(&mut self, widget: &Widget);
    fn type_text(&mut self, text: &str);
    fn press(&mut self, key: &str);
    /// `None` means the environment publishes no vocabulary, and `knows_key` then answers `true` —
    /// read with a fallback rather than made a protocol member, because the whole vocabulary is the
    /// wrong quantity for the one line that consults it.
    fn keys(&self) -> Option<Vec<String>> {
        None
    }
    fn done(&self) -> Vec<String>;
}

trait Speaks {
    fn ask(&mut self, prompt: &str) -> String;
}

// ---- the scripted environment, mirroring tests/test_loop_golden.py -------------------------------

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
    fn reads(&self) -> usize {
        self.reads
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
    fn click(&mut self, widget: &Widget) {
        let c = widget.centre();
        self.done.push(format!("click {} '{}' at ({}, {})", widget.role, widget.name, c.0, c.1));
    }
    fn type_text(&mut self, text: &str) {
        self.done.push(format!("type '{text}'"));
    }
    fn press(&mut self, key: &str) {
        //: `DeclaringHands` raises on a key it never published, so the unknown-key filter fails on
        //: the press rather than on an assertion nobody wrote.
        if let Some(names) = &self.names {
            assert!(names.iter().any(|n| n == key), "the core pressed {key:?}, which this \
                                                     environment refuses");
        }
        self.done.push(format!("key '{key}'"));
    }
    fn keys(&self) -> Option<Vec<String>> {
        self.names.clone()
    }
    fn done(&self) -> Vec<String> {
        self.done.clone()
    }
}

struct ScriptedModel {
    texts: Vec<String>,
}

impl Speaks for ScriptedModel {
    fn ask(&mut self, _prompt: &str) -> String {
        if self.texts.is_empty() {
            r#"{"say": "", "do": []}"#.to_string()
        } else {
            self.texts.remove(0)
        }
    }
}

// ---- what a thought is ---------------------------------------------------------------------------

#[derive(Clone)]
struct Trigger {
    text: String,
    asked: bool,
}

#[derive(Clone)]
struct Effect {
    kind: String,
    target: String,
    role: String,
    text: String,
    /// `because is not None and not because.asked`. Carried as an `Option` for the same reason:
    /// a hand-built effect with no trigger is **not** unprompted, which is an open edge in the
    /// permission model that `tests/test_loop_golden.py` records.
    because_asked: Option<bool>,
}

impl Effect {
    fn unprompted(&self) -> bool {
        matches!(self.because_asked, Some(false))
    }
}

struct Done {
    effect: Effect,
    target: Option<Widget>,
    confirmed: Option<bool>,
}

/// The balanced-brace extractor, ported and verified in `native/reply-spike` against 138 cases.
/// Character indices, never bytes — 94 of the 128 recorded replies are non-ASCII.
fn first_object(text: &str) -> Option<Value> {
    let (mut start, mut end, mut depth, mut in_string, mut escaped, mut at) =
        (-1i64, -1i64, 0i64, false, false, 0i64);
    for ch in text.chars() {
        let here = at;
        at += 1;
        if end >= 0 {
            continue;
        }
        if start < 0 {
            if ch != '{' {
                continue;
            }
            start = here;
        }
        if in_string {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
            }
        } else if ch == '"' {
            in_string = true;
        } else if ch == '{' {
            depth += 1;
        } else if ch == '}' {
            depth -= 1;
            if depth == 0 {
                end = here;
            }
        }
    }
    if end < 0 {
        return None;
    }
    let slice: String = text.chars().skip(start as usize).take((end - start + 1) as usize).collect();
    serde_json::from_str(&slice).ok()
}

/// `converse._effect`: **the number becomes a name and a role here and never travels further as an
/// index.** The role is half of the key — a window with a label and a combo box both called
/// `Font :` resolved to the label and the click did nothing.
fn effect_of(item: &Value, listed: &[Widget], trigger: &Trigger) -> Option<Effect> {
    let obj = item.as_object()?;
    let because = Some(trigger.asked);
    if let Some(v) = obj.get("click") {
        let index = match v {
            Value::Number(n) => n.as_i64()?,
            Value::String(s) => s.parse::<i64>().ok()?,
            _ => return None,
        };
        if index < 0 || index as usize >= listed.len() {
            return None;
        }
        let w = &listed[index as usize];
        return Some(Effect { kind: "click".into(), target: w.name.clone(), role: w.role.clone(),
                             text: String::new(), because_asked: because });
    }
    if let Some(v) = obj.get("type") {
        let s = v.as_str()?;
        if s.is_empty() {
            return None;
        }
        return Some(Effect { kind: "type".into(), target: String::new(), role: String::new(),
                             text: s.to_string(), because_asked: because });
    }
    if let Some(v) = obj.get("key") {
        let s = v.as_str()?;
        return Some(Effect { kind: "key".into(), target: String::new(), role: String::new(),
                             text: s.to_lowercase(), because_asked: because });
    }
    None
}

// ---- the session ---------------------------------------------------------------------------------

struct Session<E: Eyes, H: Hands, M: Speaks> {
    eyes: E,
    hands: H,
    model: M,
    seen: Vec<Widget>,
    history: Vec<(String, String)>,
    exchanges: usize,
    blind: u32,
    vanished: u32,
    unknown_keys: u32,
    may_act_unprompted: bool,
}

impl<E: Eyes, H: Hands, M: Speaks> Session<E, H, M> {
    fn observe(&mut self) -> Vec<Widget> {
        let (widgets, _took) = self.eyes.timed_foreground();
        self.seen = widgets.clone();
        widgets
    }

    //: `recent()` is NOT here. The scripted model ignores the prompt, so calling it would change
    //: nothing observable and would look tested while covering nothing — the window is verified in
    //: `native/loop-spike` against 17 sweeps, which is where it belongs.
    fn knows_key(&self, name: &str) -> bool {
        match self.hands.keys() {
            None => true,
            Some(names) => names.iter().any(|n| n.to_lowercase() == name.to_lowercase()),
        }
    }

    fn find(&mut self, name: &str, role: &str) -> Option<Widget> {
        let fresh = self.eyes.foreground();
        let same: Vec<&Widget> = fresh.iter()
            .filter(|w| w.name == name && (role.is_empty() || w.role == role)).collect();
        if same.is_empty() {
            return None;
        }
        if same.len() == 1 {
            return Some(same[0].clone());
        }
        let was = self.seen.iter()
            .find(|w| w.name == name && (role.is_empty() || w.role == role))
            .map(|w| w.centre()).unwrap_or_else(|| same[0].centre());
        let mut best = same[0];
        let mut best_d = i64::MAX;
        for w in &same {
            let c = w.centre();
            let d = (c.0 - was.0).pow(2) + (c.1 - was.1).pow(2);
            if d < best_d {
                best_d = d;
                best = w;
            }
        }
        Some(best.clone())
    }

    /// Did the world move? `None` when the hands are not live — a dry run has nothing to confirm.
    fn confirm(&mut self, widget: &Widget) -> Option<bool> {
        if !self.hands.live() {
            return None;
        }
        let after = self.eyes.foreground();
        let unchanged = after.iter().any(|w| w.same(widget));
        let window_changed = !after.is_empty() && after[0].window != widget.window;
        Some(window_changed || !unchanged)
    }

    /// One call: what to say, and what if anything to do. The prompt is rendered and verified in
    /// `native/prompt-spike`; here the scripted model ignores it, so only the reply half runs.
    fn decide(&mut self, trigger: &Trigger) -> (String, Vec<Effect>) {
        if self.seen.is_empty() {
            self.observe();
        }
        let listed: Vec<Widget> = self.seen.iter().take(MAX_LISTED).cloned().collect();
        let reply = self.model.ask("(rendered elsewhere)");
        let got = first_object(&reply).unwrap_or(json!({}));
        let said = got.get("say").and_then(|v| v.as_str()).unwrap_or("").trim().to_string();
        let wanted: Vec<Value> = got.get("do").and_then(|v| v.as_array()).cloned().unwrap_or_default();
        if wanted.len() > MOST_AT_ONCE {
            return (said, Vec::new());
        }
        let effects: Vec<Effect> = wanted.iter()
            .filter_map(|item| effect_of(item, &listed, trigger)).collect();
        //: The unknown-key filter, and the counter it feeds. `blind` is documented as thoughts
        //: where widgets were listed and nothing was chosen — the golden transcript records that
        //: this counter double-counts a turn this filter emptied, and the port reproduces it.
        let before = effects.len();
        let usable: Vec<Effect> = effects.into_iter()
            .filter(|e| e.kind != "key" || self.knows_key(&e.text)).collect();
        self.unknown_keys += (before - usable.len()) as u32;
        (said, usable)
    }

    fn perform(&mut self, effects: &[Effect]) -> (Vec<Done>, Vec<Effect>, Vec<Effect>) {
        let (mut did, mut proposed, mut refused) = (Vec::new(), Vec::new(), Vec::new());
        for effect in effects {
            if effect.unprompted() && !self.may_act_unprompted {
                proposed.push(effect.clone());
                continue;
            }
            match effect.kind.as_str() {
                "click" => match self.find(&effect.target, &effect.role) {
                    None => {
                        //: It was there when the question was asked and it is not now. Refusing is
                        //: the only honest option: whatever occupies that rectangle was never chosen.
                        self.vanished += 1;
                        refused.push(effect.clone());
                    }
                    Some(target) => {
                        self.hands.click(&target);
                        let confirmed = self.confirm(&target);
                        did.push(Done { effect: effect.clone(), target: Some(target), confirmed });
                    }
                },
                "type" => {
                    self.hands.type_text(&effect.text);
                    did.push(Done { effect: effect.clone(), target: None, confirmed: None });
                }
                "key" => {
                    self.hands.press(&effect.text);
                    did.push(Done { effect: effect.clone(), target: None, confirmed: None });
                }
                _ => refused.push(effect.clone()),
            }
        }
        (did, proposed, refused)
    }

    fn describe(&self, effect: &Effect, widget: Option<&Widget>, form: usize) -> String {
        let v = verb(&effect.kind, form);
        if effect.kind == "click" {
            let role = match widget {
                Some(w) => w.role.clone(),
                None => if effect.role.is_empty() { "thing".into() } else { effect.role.clone() },
            };
            return format!("{v} the {role} \"{}\"", effect.target);
        }
        if effect.kind == "type" {
            return format!("{v} \"{}\"", effect.text);
        }
        format!("{v} {}", effect.text)
    }

    /// **`think()` feeds only `did` and `refused`.** A proposed effect is returned in the exchange
    /// and recorded nowhere, so the standing-offer block never sees the core's own downgrade — the
    /// bookkeeping is each interface's to remember. Recorded by the golden transcript, reproduced
    /// here rather than fixed, because a port that fixes a defect is two changes.
    fn think(&mut self, trigger: &Trigger) -> Value {
        self.observe();
        let (said, effects) = self.decide(trigger);
        let (did, proposed, refused) = self.perform(&effects);

        self.history.push(((if trigger.asked { "them" } else { "noticed" }).to_string(),
                           trigger.text.clone()));
        if !said.is_empty() {
            self.history.push(("you".into(), said.clone()));
        }
        for d in &did {
            let line = self.describe(&d.effect, d.target.as_ref(), PAST);
            self.history.push(("did".into(), line));
        }
        for e in &refused {
            let line = format!("{} — it had gone by then; {NOTHING_DONE}",
                               self.describe(e, None, PLAIN));
            self.history.push(("could not".into(), line));
        }
        if !self.seen.is_empty() && effects.is_empty() {
            self.blind += 1;
        }
        let n = self.exchanges;
        self.exchanges += 1;
        json!({
            "n": n, "asked": trigger.asked, "trigger": trigger.text, "said": said,
            "n_widgets": self.seen.len(),
            "did": did.iter().map(|d| json!({
                "effect": effect_json(&d.effect),
                "confirmed": d.confirmed,
                "target": d.target.as_ref().map(widget_json),
            })).collect::<Vec<_>>(),
            "proposed": proposed.iter().map(effect_json).collect::<Vec<_>>(),
            "refused": refused.iter().map(effect_json).collect::<Vec<_>>(),
            "history": self.history.iter().map(|(a, b)| json!([a, b])).collect::<Vec<_>>(),
            "blind": self.blind, "vanished": self.vanished, "unknown_keys": self.unknown_keys,
            "hands_done": self.hands.done(), "eye_reads": self.eyes.reads(),
        })
    }
}

fn effect_json(e: &Effect) -> Value {
    json!({"kind": e.kind, "target": e.target, "role": e.role, "text": e.text,
           "unprompted": e.unprompted()})
}

fn widget_json(w: &Widget) -> Value {
    json!({"name": w.name, "role": w.role, "rect": [w.rect.0, w.rect.1, w.rect.2, w.rect.3],
           "window": w.window})
}

fn widget_of(v: &Value) -> Widget {
    let r = v["rect"].as_array().unwrap();
    Widget { name: v["name"].as_str().unwrap().into(), role: v["role"].as_str().unwrap().into(),
             rect: (r[0].as_i64().unwrap(), r[1].as_i64().unwrap(),
                    r[2].as_i64().unwrap(), r[3].as_i64().unwrap()),
             window: v["window"].as_str().unwrap().into() }
}

fn main() {
    let path = env::args().nth(1).expect("usage: session-spike <expectations.json>");
    let cases: Value = serde_json::from_str(&fs::read_to_string(&path).expect("read")).unwrap();
    let (mut turns, mut bad) = (0, 0);

    for case in cases.as_array().unwrap() {
        let screens: Vec<Vec<Widget>> = case["screens"].as_array().unwrap().iter()
            .map(|s| s.as_array().unwrap().iter().map(widget_of).collect()).collect();
        let keys = case["hands"]["keys"].as_array()
            .map(|a| a.iter().map(|k| k.as_str().unwrap().to_string()).collect());
        let mut session = Session {
            eyes: ScriptedEyes { screens, reads: 0 },
            hands: ScriptedHands { live: case["hands"]["live"].as_bool().unwrap(), names: keys,
                                   done: Vec::new() },
            model: ScriptedModel { texts: case["replies"].as_array().unwrap().iter()
                                       .map(|t| t.as_str().unwrap().to_string()).collect() },
            seen: Vec::new(), history: Vec::new(), exchanges: 0,
            blind: 0, vanished: 0, unknown_keys: 0,
            may_act_unprompted: case["may_act_unprompted"].as_bool().unwrap(),
        };
        for (i, t) in case["turns"].as_array().unwrap().iter().enumerate() {
            turns += 1;
            let got = session.think(&Trigger { text: t["text"].as_str().unwrap().into(),
                                               asked: t["asked"].as_bool().unwrap() });
            let want = &case["want"][i];
            for field in ["n", "asked", "trigger", "said", "n_widgets", "did", "proposed",
                          "refused", "history", "blind", "vanished", "unknown_keys",
                          "hands_done", "eye_reads"] {
                if got[field] != want[field] {
                    bad += 1;
                    println!("  {} / turn {i} / {field}\n    python {}\n    rust   {}",
                             case["label"].as_str().unwrap(), want[field], got[field]);
                }
            }
        }
    }

    println!("\n  {} scenarios, {turns} turns, {bad} field disagreements",
             cases.as_array().unwrap().len());
    if bad == 0 {
        println!("\n  IDENTICAL — every field of every turn, through the whole seam.");
    } else {
        std::process::exit(1);
    }
}
