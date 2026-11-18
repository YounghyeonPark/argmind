//! **Can argmind look at the screen, and how fast?** — the list it reads today against a local
//! vision model shown the picture, and Windows' own OCR.
//!
//! ```text
//!   cargo run -p argmind-measure --bin looking --release              every arm, three repeats
//!   cargo run -p argmind-measure --bin looking --release -- --pilot   one goal, the raw reply
//! ```
//!
//! From `native/`. **Stop argmind first** — the vision model and the conversation model both want
//! the card. Pulls nothing: `qwen3-vl:8b-instruct` and `qwen3-vl:4b-instruct` must be in Ollama.
//!
//! ## Why
//!
//! The owner, 2026-11-18, on the public page selling *it reads the screen instead of looking at a
//! picture*: *장점이 아닌거같음 보면서 속도를 빠르게 할수는 없을까.* The number behind that line —
//! **24–82 s** for `qwen3-vl:8b` to give a pixel, against 41–266 ms for UI Automation — was taken
//! with the **thinking** build (Ollama's `qwen3-vl:8b` is it: renderer `qwen3-vl-thinking`, and
//! the same digest as `8b-thinking`) asked for a pixel, and its log was 13,000 characters of
//! `row × row height`. That is a measurement of arithmetic, not of looking. Qwen3-VL is trained to
//! answer a box in 0–1000 coordinates, and an instruct build answers without the essay. Nobody
//! here had asked it the question it was trained on.
//!
//! What the list cannot do is the other half: an icon with no name, a character inside a grid that
//! UI Automation lists as one pane (the owner's ★ task stopped there, 2026-11-09), anything inside
//! a remote desktop, and *which* of three buttons called 열기.
//!
//! ## The screens
//!
//! Five, drawn by `measure/looking/draw.mjs` from `measure/looking/html/` — a file list with 32
//! rows at 19 px, a save dialog, a paint toolbar with two unnamed icons, a Character Map with
//! three 열기 buttons, and a remote desktop. **Drawn, not captured**, so the right answer's box is
//! known to the pixel and nothing anybody did on this machine is in a picture. 1280×800 each, the
//! size of a window rather than a screen. `goals.json`: 26 requests — 13 the list can do, 2 it
//! would ask about, 6 it cannot see, 5 nothing on the screen does.
//!
//! ## The arms
//!
//! * **list** — `qwen3:8b`, the shipped prompt, `render` and `read_reply`, as `choosing` runs it. A
//!   name two widgets share counts as **asked**, because `find` returns `Several` and the
//!   application asks.
//! * **box** — the plain picture; the model answers `{"bbox_2d": [x1, y1, x2, y2]}` in 0–1000, or
//!   null. Right when the box's centre is inside the target.
//! * **marks** — the same picture with every listed widget outlined and numbered; the model answers
//!   a number or null. **The coordinate stays the list's**: the model sees the picture and still
//!   never emits a position, which is the rule this project hangs on.
//! * **ocr** — `Windows.Media.Ocr` in Korean on the plain picture: the time, and whether the
//!   target's own text is found inside the target.
//! * **control** — the installed thinking build asked for a pixel, as the old number was taken, on
//!   the three file-list goals. **It exists to say whether these screens are easier than the old
//!   one**: if the old way is fast here too, every speed below flatters.
//!
//! Three repeats, temperature 0, **worst counted** — right, declined, asked, cannot, missed, wrong.
//! Seconds are wall time around one call, warm, image upload included.
//!
//! ## Written before the run
//!
//! **box**, each model:
//! * *fast enough to look on every turn* — median ≤ 1.5 s and slowest ≤ 3 s;
//! * *fast enough to look when the list cannot answer* — median ≤ 4 s;
//! * *sure enough to press without the list* — listed ≥ 12 of 13, none declined ≥ 4 of 5, at most
//!   1 wrong over all 26;
//! * *useful where the list is blind* — unlisted ≥ 5 of 6 and same-name 2 of 2.
//!
//! **marks**: helps if same-name is 2 of 2, listed is no lower than the list arm's, unlisted wrong
//! is at most 1 (there is no number to give, so declining is right), and median ≤ 3 s.
//!
//! **ocr**: a source of WHERE for text in a picture if median ≤ 0.5 s a screen and the text is
//! found inside its target on ≥ 14 of 16, including 3 of 3 inside the remote desktop.
//!
//! **control**: predicted ≥ 20 s on each of the three. Under 10 s means these screens are easier
//! than the old one.
//!
//! **list**, predicted rather than judged: unlisted 0 of 6 by construction, same-name asked.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use argmind_core::contract::Widget;
use argmind_core::converse::{read_reply, render, MAX_LISTED};
use argmind_core::intent::Trigger;
use argmind_core::mind::Mind;
use argmind_env_ollama::Local;
use argmind_measure::{print6, source_print, Destination};
use serde_json::{json, Value};

const LIST_MODEL: &str = "qwen3:8b";
const VISION: [&str; 2] = ["qwen3-vl:8b-instruct", "qwen3-vl:4b-instruct"];
const OLD_WAY: &str = "qwen3-vl:8b";
const REPEATS: usize = 3;
const ORDER: [&str; 6] = ["right", "declined", "asked", "cannot", "missed", "wrong"];

type Rect = (i64, i64, i64, i64);

struct Listed {
    widget: Widget,
    target: Option<String>,
}

struct Screen {
    widgets: Vec<Listed>,
    targets: BTreeMap<String, Rect>,
    plain: Vec<u8>,
    marked: Vec<u8>,
    w: f64,
    h: f64,
}

struct Goal {
    screen: String,
    goal: String,
    target: Option<String>,
    kind: String,
    shows: Option<String>,
}

fn inside(r: Rect, x: f64, y: f64) -> bool {
    x >= r.0 as f64 && x <= (r.0 + r.2) as f64 && y >= r.1 as f64 && y <= (r.1 + r.3) as f64
}

fn png_size(png: &[u8]) -> (f64, f64) {
    let be = |b: &[u8]| u32::from_be_bytes([b[0], b[1], b[2], b[3]]) as f64;
    (be(&png[16..20]), be(&png[20..24]))
}

fn load(dir: &Path) -> (BTreeMap<String, Screen>, Vec<Goal>, String) {
    let mut printed: Vec<u8> = Vec::new();
    let raw = std::fs::read_to_string(dir.join("goals.json")).expect("goals.json");
    printed.extend(raw.as_bytes());
    let d: Value = serde_json::from_str(&raw).unwrap();
    let goals: Vec<Goal> = d["goals"].as_array().unwrap().iter().map(|g| Goal {
        screen: g["screen"].as_str().unwrap().into(),
        goal: g["goal"].as_str().unwrap().into(),
        target: g["target"].as_str().map(String::from),
        kind: g["kind"].as_str().unwrap().into(),
        shows: g["shows"].as_str().map(String::from),
    }).collect();
    let mut screens = BTreeMap::new();
    for name in ["explorer", "dialog", "paint", "charmap", "remote"] {
        let raw = std::fs::read_to_string(dir.join(format!("{name}.json"))).expect("screen json");
        printed.extend(raw.as_bytes());
        let s: Value = serde_json::from_str(&raw).unwrap();
        let window = s["window"].as_str().unwrap_or("").to_string();
        let rect = |v: &Value| -> Rect {
            let a = v.as_array().unwrap();
            (a[0].as_i64().unwrap(), a[1].as_i64().unwrap(), a[2].as_i64().unwrap(), a[3].as_i64().unwrap())
        };
        let widgets = s["widgets"].as_array().unwrap().iter().map(|w| Listed {
            widget: Widget::new(w["name"].as_str().unwrap(), w["role"].as_str().unwrap(), rect(&w["rect"]), &window),
            target: w["target"].as_str().map(String::from),
        }).collect();
        let targets = s["targets"].as_object().unwrap().iter().map(|(k, v)| (k.clone(), rect(v))).collect();
        let plain = std::fs::read(dir.join(format!("{name}.png"))).unwrap();
        let marked = std::fs::read(dir.join(format!("{name}-marked.png"))).unwrap();
        printed.extend(&plain);
        printed.extend(&marked);
        let (w, h) = png_size(&plain);
        screens.insert(name.to_string(), Screen { widgets, targets, plain, marked, w, h });
    }
    (screens, goals, print6(&printed))
}

// ---- reading what came back ---------------------------------------------------------------------

/// The numbers after `key`, or `Some(vec![])` for a `null` there; `None` when the key is absent.
fn numbers_after(text: &str, key: &str) -> Option<Vec<f64>> {
    let at = text.find(key)?;
    let rest = &text[at + key.len()..];
    let rest = rest.trim_start_matches(|c: char| c == '"' || c == ':' || c.is_whitespace());
    if rest.starts_with("null") {
        return Some(vec![]);
    }
    let mut out = Vec::new();
    let mut cur = String::new();
    for c in rest.chars() {
        if c.is_ascii_digit() || c == '.' || (c == '-' && cur.is_empty()) {
            cur.push(c);
        } else {
            if !cur.is_empty() {
                if let Ok(n) = cur.parse() { out.push(n); }
                cur.clear();
            }
            if c == ']' || c == '}' || out.len() >= 4 {
                break;
            }
        }
    }
    if !cur.is_empty() { if let Ok(n) = cur.parse() { out.push(n); } }
    Some(out)
}

enum Pointed {
    At(f64, f64),
    Nothing,
    Unreadable,
}

/// A box in 0–1000, as Qwen3-VL is trained to give it, turned into a centre in picture pixels.
fn pointed(text: &str, w: f64, h: f64) -> Pointed {
    let got = numbers_after(text, "bbox_2d").or_else(|| numbers_after(text, "point_2d"));
    match got {
        Some(n) if n.is_empty() => Pointed::Nothing,
        Some(n) if n.len() >= 4 => Pointed::At((n[0] + n[2]) / 2.0 * w / 1000.0, (n[1] + n[3]) / 2.0 * h / 1000.0),
        Some(n) if n.len() == 2 => Pointed::At(n[0] * w / 1000.0, n[1] * h / 1000.0),
        _ => Pointed::Unreadable,
    }
}

// ---- scoring ------------------------------------------------------------------------------------

fn worst(results: &[&'static str]) -> &'static str {
    results.iter().copied().max_by_key(|r| ORDER.iter().position(|o| o == r).unwrap_or(ORDER.len()))
        .unwrap_or("missed")
}

fn by_point(g: &Goal, s: &Screen, p: &Pointed) -> &'static str {
    match (p, &g.target) {
        (Pointed::Nothing, None) => "declined",
        (Pointed::Nothing | Pointed::Unreadable, Some(_)) => "missed",
        (Pointed::Unreadable, None) => "missed",
        (Pointed::At(..), None) => "wrong",
        (Pointed::At(x, y), Some(t)) => if inside(s.targets[t], *x, *y) { "right" } else { "wrong" },
    }
}

/// What a centre landed on, for reading a miss: the target's id, a listed name, or nothing.
fn landed(s: &Screen, x: f64, y: f64) -> String {
    let mut hit: Vec<String> = s.targets.iter().filter(|(_, r)| inside(**r, x, y)).map(|(k, _)| k.clone()).collect();
    if hit.is_empty() {
        //: The smallest listed widget under the point, since a pane contains everything in it.
        let mut under: Vec<&Listed> = s.widgets.iter().filter(|l| {
            let r = l.widget.rect;
            inside(r, x, y)
        }).collect();
        under.sort_by_key(|l| l.widget.rect.2 * l.widget.rect.3);
        if let Some(l) = under.first() { hit.push(l.widget.name.clone()); }
    }
    hit.join("+")
}

fn by_mark(g: &Goal, s: &Screen, mark: Option<Option<usize>>) -> &'static str {
    match (mark, &g.target) {
        (None, _) => "missed",
        (Some(None), None) => "declined",
        (Some(None), Some(_)) => if g.kind == "unlisted" { "cannot" } else { "missed" },
        (Some(Some(_)), None) => "wrong",
        (Some(Some(n)), Some(t)) => match s.widgets.get(n.wrapping_sub(1)) {
            Some(l) if l.target.as_deref() == Some(t.as_str()) => "right",
            _ => "wrong",
        },
    }
}

fn by_list(g: &Goal, s: &Screen, chose: Option<(&str, &str)>) -> &'static str {
    let Some((name, role)) = chose else {
        return match (&g.target, g.kind.as_str()) {
            (None, _) => "declined",
            (Some(_), "unlisted") => "cannot",
            _ => "missed",
        };
    };
    let Some(t) = &g.target else { return "wrong" };
    let same: Vec<&Listed> = s.widgets.iter().filter(|l| l.widget.name == name && l.widget.role == role).collect();
    if same.iter().any(|l| l.target.as_deref() == Some(t.as_str())) {
        if same.len() > 1 { "asked" } else { "right" }
    } else {
        "wrong"
    }
}

// ---- asking -------------------------------------------------------------------------------------

fn ask_box(goal: &str) -> String {
    format!("This is a screenshot of a Windows screen. The person said: \"{goal}\"\n\
             Find the one on-screen element they want clicked. Answer with JSON only: \
             {{\"bbox_2d\": [x1, y1, x2, y2]}} with coordinates from 0 to 1000 relative to the image \
             width and height. If nothing on this screen does what they asked, answer {{\"bbox_2d\": null}}.")
}

fn ask_mark(goal: &str) -> String {
    format!("This is a screenshot of a Windows screen. Every element that can be clicked has a pink \
             outline and a yellow number tag at its top-left corner. The person said: \"{goal}\"\n\
             Which number should be clicked? Answer with JSON only: {{\"mark\": N}}. If no numbered \
             element does what they asked, answer {{\"mark\": null}}.")
}

fn ask_pixel(goal: &str, w: f64, h: f64) -> String {
    format!("이 그림은 {w}×{h} 픽셀의 Windows 화면이다. 사람이 말했다: \"{goal}\"\n\
             눌러야 할 곳의 중심을 픽셀 좌표로 답하라. JSON 만: {{\"x\": 픽셀, \"y\": 픽셀}}")
}

fn median(v: &mut [f64]) -> f64 {
    if v.is_empty() { return f64::NAN; }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 { v[n / 2] } else { (v[n / 2 - 1] + v[n / 2]) / 2.0 }
}

fn clip(s: &str) -> String {
    s.chars().take(160).collect::<String>().replace('\n', " ")
}

struct Tally {
    rows: Vec<Value>,
    seconds: Vec<f64>,
    tokens: Vec<f64>,
    counted: Vec<(String, &'static str)>,
}

impl Tally {
    fn new() -> Self { Tally { rows: vec![], seconds: vec![], tokens: vec![], counted: vec![] } }

    fn of(&self, kind: &str, outcome: &str) -> usize {
        self.counted.iter().filter(|(k, o)| k == kind && *o == outcome).count()
    }

    fn wrong(&self) -> usize { self.counted.iter().filter(|(_, o)| *o == "wrong").count() }

    fn summary(&self, arm: &str, model: &str) -> Value {
        let mut s = self.seconds.clone();
        let med = median(&mut s);
        let max = self.seconds.iter().cloned().fold(0.0, f64::max);
        let mut t = self.tokens.clone();
        json!({
            "arm": arm, "model": model,
            "listed_right": self.of("listed", "right"),
            "same_name_right": self.of("same-name", "right"),
            "same_name_asked": self.of("same-name", "asked"),
            "unlisted_right": self.of("unlisted", "right"),
            "unlisted_wrong": self.of("unlisted", "wrong"),
            "none_declined": self.of("none", "declined"),
            "wrong": self.wrong(),
            "median_s": (med * 100.0).round() / 100.0,
            "max_s": (max * 100.0).round() / 100.0,
            "median_prompt_tokens": median(&mut t),
            "goals": self.rows,
        })
    }

    fn line(&self, arm: &str, model: &str) {
        let mut s = self.seconds.clone();
        let med = median(&mut s);
        let max = self.seconds.iter().cloned().fold(0.0, f64::max);
        println!("  {arm:<7} {model:<22} 이름있음 {:>2}/13  같은이름 {}/2 (물음 {})  목록밖 {}/6 (틀림 {})  \
                  없음 거절 {}/5  틀림 {:>2}  중앙값 {med:.2}s  최대 {max:.2}s",
                 self.of("listed", "right"), self.of("same-name", "right"), self.of("same-name", "asked"),
                 self.of("unlisted", "right"), self.of("unlisted", "wrong"), self.of("none", "declined"),
                 self.wrong());
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let pilot = args.iter().any(|a| a == "--pilot");
    let only = args.iter().position(|a| a == "--arms").and_then(|i| args.get(i + 1)).cloned()
        .unwrap_or_else(|| "list,box,marks,ocr,control".into());
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap().to_path_buf();
    let dir = root.join("measure").join("looking");
    let (screens, goals, fixtures) = load(&dir);
    let harness = source_print(include_str!("looking.rs"));

    if pilot {
        //: One goal, the raw reply printed: the check that the box convention is the one the
        //: model answers in, before anything is scored with it.
        let g = goals.iter().find(|g| g.goal == "저장해").unwrap();
        let s = &screens[&g.screen];
        let mut m = Local::new(VISION[0]);
        m.think = Some(false);
        m.max_tokens = 256;
        let began = Instant::now();
        let r = m.look_at(&ask_box(&g.goal), &s.plain).expect("vision call");
        println!("  {} s · {:?} prompt tokens · {:?}", began.elapsed().as_secs_f64(), r.prompt_tokens, r.text);
        println!("  target {:?} in a {}×{} picture", s.targets[g.target.as_ref().unwrap()], s.w, s.h);
        if let Pointed::At(x, y) = pointed(&r.text, s.w, s.h) {
            println!("  read as 0–1000: centre ({x:.0}, {y:.0}) → {}", landed(s, x, y));
        }
        return;
    }

    let out = Destination::new(&dir, "results", &[only.replace(',', "+"), format!("r{REPEATS}"),
                                                  format!("f{fixtures}"), format!("h{harness}")]);
    if let Err(e) = out.refuse_if_taken() { eprintln!("{e}"); std::process::exit(2); }
    println!("\n  보면서 고르기 — 화면 5개, 목표 {}개, {REPEATS}번씩 · 그림 f{fixtures} · harness {harness}\n", goals.len());

    let mut summaries: Vec<Value> = Vec::new();

    // ---- list: today's path ----
    if only.contains("list") {
        let mut model = Local::new(LIST_MODEL);
        model.think = Some(false);
        model.keep_alive = "300s".into();
        let _ = model.ask("Reply with the single word: ready");
        let mut t = Tally::new();
        for g in &goals {
            let s = &screens[&g.screen];
            let widgets: Vec<Widget> = s.widgets.iter().map(|l| l.widget.clone()).collect();
            let trigger = Trigger::said(&g.goal);
            let prompt = render(&widgets, &trigger, &[], MAX_LISTED, &Mind::resting());
            let mut results = Vec::new();
            let mut tries = Vec::new();
            for _ in 0..REPEATS {
                let began = Instant::now();
                let reply = model.ask(&prompt).expect("list call");
                let secs = began.elapsed().as_secs_f64();
                t.seconds.push(secs);
                if let Some(p) = reply.prompt_tokens { t.tokens.push(p as f64); }
                let intent = read_reply(&reply.text, &widgets, &trigger);
                let click = intent.effects.iter().find(|e| e.kind == "click");
                let r = by_list(g, s, click.map(|e| (e.target.as_str(), e.role.as_str())));
                results.push(r);
                tries.push(json!({"seconds": (secs * 100.0).round() / 100.0, "result": r,
                                  "chose": click.map(|e| format!("{} ({})", e.target, e.role))}));
            }
            let w = worst(&results);
            t.counted.push((g.kind.clone(), w));
            t.rows.push(json!({"goal": g.goal, "kind": g.kind, "counted": w, "tries": tries}));
        }
        t.line("list", LIST_MODEL);
        summaries.push(t.summary("list", LIST_MODEL));
    }

    // ---- box and marks: a vision model shown the picture ----
    for name in VISION {
        if !(only.contains("box") || only.contains("marks")) { break; }
        let mut model = Local::new(name);
        model.think = Some(false);
        model.max_tokens = 256;
        model.keep_alive = "300s".into();
        let _ = model.look_at("Answer with one word: ready", &screens["dialog"].plain);
        let mut boxed = Tally::new();
        let mut marked = Tally::new();
        for g in &goals {
            let s = &screens[&g.screen];
            for (arm, tally) in [("box", &mut boxed), ("marks", &mut marked)] {
                if !only.contains(arm) { continue; }
                let mut results = Vec::new();
                let mut tries = Vec::new();
                for _ in 0..REPEATS {
                    let (prompt, picture) = if arm == "box" { (ask_box(&g.goal), &s.plain) } else { (ask_mark(&g.goal), &s.marked) };
                    let began = Instant::now();
                    let reply = model.look_at(&prompt, picture).expect("vision call");
                    let secs = began.elapsed().as_secs_f64();
                    tally.seconds.push(secs);
                    if let Some(p) = reply.prompt_tokens { tally.tokens.push(p as f64); }
                    let (r, extra) = if arm == "box" {
                        let p = pointed(&reply.text, s.w, s.h);
                        let at = match p { Pointed::At(x, y) => json!([x.round(), y.round(), landed(s, x, y)]), _ => Value::Null };
                        (by_point(g, s, &p), at)
                    } else {
                        let m = numbers_after(&reply.text, "mark").map(|n| n.first().map(|v| *v as usize));
                        let on = match m { Some(Some(n)) => s.widgets.get(n.wrapping_sub(1)).map(|l| l.widget.name.clone()), _ => None };
                        (by_mark(g, s, m), json!([m, on]))
                    };
                    results.push(r);
                    tries.push(json!({"seconds": (secs * 100.0).round() / 100.0, "result": r, "at": extra,
                                      "reply": clip(&reply.text), "thought": reply.thought, "reason": reply.reason}));
                }
                let w = worst(&results);
                tally.counted.push((g.kind.clone(), w));
                tally.rows.push(json!({"goal": g.goal, "kind": g.kind, "counted": w, "tries": tries}));
            }
        }
        if only.contains("box") { boxed.line("box", name); summaries.push(boxed.summary("box", name)); }
        if only.contains("marks") { marked.line("marks", name); summaries.push(marked.summary("marks", name)); }
    }

    // ---- control: the old way, on the dense list ----
    if only.contains("control") {
        let mut model = Local::new(OLD_WAY);
        model.keep_alive = "300s".into();
        let _ = model.look_at("Answer with one word: ready", &screens["dialog"].plain);
        let mut t = Tally::new();
        for g in goals.iter().filter(|g| g.screen == "explorer").take(3) {
            let s = &screens[&g.screen];
            let began = Instant::now();
            let reply = model.look_at(&ask_pixel(&g.goal, s.w, s.h), &s.plain).expect("control call");
            let secs = began.elapsed().as_secs_f64();
            t.seconds.push(secs);
            let x = numbers_after(&reply.text, "\"x\"").and_then(|n| n.first().copied());
            let y = numbers_after(&reply.text, "\"y\"").and_then(|n| n.first().copied());
            let r = match (x, y, &g.target) {
                (Some(x), Some(y), Some(tg)) => if inside(s.targets[tg], x, y) { "right" } else { "wrong" },
                _ => "missed",
            };
            println!("  control {secs:>6.1}s · 생각 {}자 · {} · {}", reply.thought, r, clip(&reply.text));
            t.counted.push((g.kind.clone(), r));
            t.rows.push(json!({"goal": g.goal, "seconds": (secs * 10.0).round() / 10.0, "result": r,
                               "thought": reply.thought, "reason": reply.reason, "reply": clip(&reply.text)}));
        }
        t.line("control", OLD_WAY);
        summaries.push(t.summary("control", OLD_WAY));
    }

    // ---- ocr: Windows' own, no model ----
    if only.contains("ocr") {
        summaries.push(ocr_arm(&screens, &goals));
    }

    let record = json!({
        "harness": harness, "fixtures": fixtures, "repeats": REPEATS,
        "when": "2026-11-18", "picture": "1280x800 PNG, drawn", "arms": summaries,
    });
    std::fs::write(&out.path, serde_json::to_string_pretty(&record).unwrap() + "\n").unwrap();
    println!("\n  → {}", out.path.display());
}

fn ocr_arm(screens: &BTreeMap<String, Screen>, goals: &[Goal]) -> Value {
    let engine = match ocr::engine() {
        Ok(e) => e,
        Err(e) => {
            println!("  ocr     Windows OCR 를 만들 수 없음: {e}");
            return json!({"arm": "ocr", "error": e.to_string()});
        }
    };
    let mut seconds = Vec::new();
    let mut read: BTreeMap<&str, Vec<ocr::Line>> = BTreeMap::new();
    let mut per_screen = Vec::new();
    for (name, s) in screens {
        let decode = Instant::now();
        let bmp = ocr::bitmap(&s.plain).expect("decode");
        let decode_s = decode.elapsed().as_secs_f64();
        let mut times = Vec::new();
        let mut lines = Vec::new();
        for i in 0..6 {
            let began = Instant::now();
            lines = ocr::read(&engine, &bmp).expect("recognise");
            let secs = began.elapsed().as_secs_f64();
            //: The first is the engine's own warm-up and is reported, not counted.
            if i > 0 { times.push(secs); seconds.push(secs); }
        }
        let words: usize = lines.iter().map(|l| l.words.len()).sum();
        let med = median(&mut times.clone());
        per_screen.push(json!({"screen": name, "decode_s": (decode_s * 1000.0).round() / 1000.0,
                               "median_s": (med * 1000.0).round() / 1000.0, "words": words}));
        println!("  ocr     {name:<9} {med:.3}s · 단어 {words}");
        //: `LOOKING_WORDS=1` prints every line as read — the only way to tell a miss of the OCR
        //: from a miss of the matcher.
        if std::env::var("LOOKING_WORDS").is_ok() {
            for l in &lines {
                let t: Vec<String> = l.words.iter().map(|w| format!("{}@{:.0},{:.0}", w.text, w.rect.0, w.rect.1)).collect();
                println!("            | {}", t.join(" "));
            }
        }
        read.insert(name.as_str(), lines);
    }
    let mut found = 0;
    let mut remote_found = 0;
    let mut text_goals = 0;
    let mut rows = Vec::new();
    for g in goals {
        let (Some(want), Some(t)) = (&g.shows, &g.target) else { continue };
        text_goals += 1;
        let s = &screens[&g.screen];
        let at = ocr::find(&read[g.screen.as_str()], want);
        let hit = at.iter().any(|(x, y)| inside(s.targets[t], *x, *y));
        if hit { found += 1; if g.screen == "remote" { remote_found += 1; } }
        rows.push(json!({"goal": g.goal, "shows": want, "occurrences": at.len(), "inside_target": hit}));
        println!("          {} {want} ({}곳)", if hit { "●" } else { "○" }, at.len());
    }
    let med = median(&mut seconds.clone());
    println!("  ocr     찾음 {found}/{text_goals} (원격 {remote_found}/3) · 중앙값 {med:.3}s");
    json!({"arm": "ocr", "found": found, "of": text_goals, "remote_found": remote_found,
           "median_s": (med * 1000.0).round() / 1000.0, "screens": per_screen, "goals": rows})
}

/// `Windows.Media.Ocr`, from an unpackaged process — which the documentation says needs package
/// identity, and which this arm is also the test of.
mod ocr {
    use windows::core::{Interface, HSTRING};
    use windows::Foundation::IAsyncOperation;
    use windows::Globalization::Language;
    use windows::Graphics::Imaging::{BitmapDecoder, SoftwareBitmap};
    use windows::Media::Ocr::OcrEngine;
    use windows::Storage::Streams::{DataWriter, InMemoryRandomAccessStream};

    pub struct Word {
        pub text: String,
        pub rect: (f64, f64, f64, f64),
    }

    pub struct Line {
        pub words: Vec<Word>,
    }

    pub fn engine() -> windows::core::Result<OcrEngine> {
        unsafe {
            let _ = windows::Win32::System::WinRT::RoInitialize(windows::Win32::System::WinRT::RO_INIT_MULTITHREADED);
        }
        OcrEngine::TryCreateFromLanguage(&Language::CreateLanguage(&HSTRING::from("ko"))?)
    }

    pub fn bitmap(png: &[u8]) -> windows::core::Result<SoftwareBitmap> {
        let stream = InMemoryRandomAccessStream::new()?;
        let writer = DataWriter::CreateDataWriter(&stream)?;
        writer.WriteBytes(png)?;
        writer.StoreAsync()?.cast::<IAsyncOperation<u32>>()?.get()?;
        writer.FlushAsync()?.get()?;
        writer.DetachStream()?;
        stream.Seek(0)?;
        BitmapDecoder::CreateAsync(&stream)?.get()?.GetSoftwareBitmapAsync()?.get()
    }

    pub fn read(engine: &OcrEngine, bmp: &SoftwareBitmap) -> windows::core::Result<Vec<Line>> {
        let result = engine.RecognizeAsync(bmp)?.get()?;
        let mut out = Vec::new();
        for line in result.Lines()? {
            let mut words = Vec::new();
            for w in line.Words()? {
                let r = w.BoundingRect()?;
                words.push(Word { text: w.Text()?.to_string(),
                                  rect: (r.X as f64, r.Y as f64, r.Width as f64, r.Height as f64) });
            }
            out.push(Line { words });
        }
        Ok(out)
    }

    /// The centre of every run of consecutive words in a line that spells `want`, spaces ignored.
    pub fn find(lines: &[Line], want: &str) -> Vec<(f64, f64)> {
        let want: String = want.chars().filter(|c| !c.is_whitespace()).collect();
        let mut out = Vec::new();
        for line in lines {
            for i in 0..line.words.len() {
                let mut spelled = String::new();
                for j in i..line.words.len() {
                    spelled.extend(line.words[j].text.chars().filter(|c| !c.is_whitespace()));
                    if spelled == want {
                        let ws = &line.words[i..=j];
                        let x0 = ws.iter().map(|w| w.rect.0).fold(f64::MAX, f64::min);
                        let y0 = ws.iter().map(|w| w.rect.1).fold(f64::MAX, f64::min);
                        let x1 = ws.iter().map(|w| w.rect.0 + w.rect.2).fold(0.0, f64::max);
                        let y1 = ws.iter().map(|w| w.rect.1 + w.rect.3).fold(0.0, f64::max);
                        out.push(((x0 + x1) / 2.0, (y0 + y1) / 2.0));
                        break;
                    }
                    if spelled.len() >= want.len() { break; }
                }
            }
        }
        out
    }
}
