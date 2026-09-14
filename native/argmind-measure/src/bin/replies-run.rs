//! Record real model replies **chunk by chunk**, so the stop logic can be replayed offline.
//!
//! ```text
//!   replies-run --plan                    what would be recorded, no model
//!   replies-run --arm off                 the shipped configuration's stream shapes
//!   replies-run --arm on                  --think on, the densest ceiling source
//!   replies-run --arm small               qwen3:0.6b, for replies that do not parse
//!   replies-run --arm stress              asked for a loop, because none turns up now
//! ```
//!
//! **Recorded with the stops OFF**, and that is the one thing that would waste the run. With them
//! on the stream is cut at the stop and everything after it is never seen — the corpus would then
//! hold exactly the prefixes the shipped logic already produces and could never exercise the logic
//! it exists to test.
//!
//! **The selection is deliberate, stratified, and NOT a sample. No rate may be computed from it.**
//! The interesting rows are the ones that broke something and they are rare, so the scenes
//! over-sample them on purpose from the three sources this project has measured. The healthy rows
//! come free: every other turn of the same conversations is recorded too.
//!
//! There is no second read loop here. `Local::ask_into` hands this every line as the shipped reader
//! takes it, so what is stored is what the reader saw and not a recorder's idea of it.

use std::cell::RefCell;
use std::path::PathBuf;
use std::time::Instant;

use argmind_core::contract::Speaks;
use argmind_core::llm::Reply;
use argmind_env_ollama::Local;
use argmind_measure::conversation::{cases, drive, script, Records};
use argmind_measure::replies::{arrived, check, coverage, disagreement, encode, sha16,
                               stream_print, STOPS_OFF};
use argmind_measure::{screen_print, prompt_tag, screens, source_print, Destination};
use serde_json::{json, Value};

/// One call, kept whole: the prompt, every line off the wire, and what the reader made of it.
struct Call {
    prompt: String,
    raw: Vec<String>,
    reply: Reply,
    seconds: f64,
    ttft: Option<f64>,
}

/// The model, recording. Every call it makes is kept.
struct Watched {
    inner: Local,
    calls: RefCell<Vec<Call>>,
    last: String,
    reply: Option<Reply>,
}

impl Speaks for Watched {
    fn ask(&mut self, prompt: &str) -> String {
        self.last = prompt.to_string();
        let raw = RefCell::new(Vec::new());
        let started = Instant::now();
        let ttft = RefCell::new(None);
        let reply = {
            let mut keep = |line: &str| {
                //: **First content, not first line.** A thinking line arrives before any words and
                //: a blank one carries nothing; neither is the moment a person would hear anything.
                if ttft.borrow().is_none()
                    && serde_json::from_str::<Value>(line).ok()
                        .and_then(|d| d["message"]["content"].as_str().map(|c| !c.is_empty()))
                        .unwrap_or(false)
                {
                    *ttft.borrow_mut() = Some(started.elapsed().as_secs_f64());
                }
                raw.borrow_mut().push(line.to_string());
            };
            self.inner.ask_into(prompt, &mut keep).expect("the model did not answer")
        };
        let seconds = started.elapsed().as_secs_f64();
        let text = reply.text.clone();
        self.calls.borrow_mut().push(Call {
            prompt: prompt.to_string(), raw: raw.into_inner(),
            reply: Reply { text: reply.text.clone(), ..reply }, seconds,
            ttft: ttft.into_inner(),
        });
        self.reply = Some(self.calls.borrow().last().map(|c| Reply {
            text: c.reply.text.clone(), prompt_tokens: c.reply.prompt_tokens,
            out_tokens: c.reply.out_tokens, reason: c.reply.reason.clone(),
            thought: c.reply.thought }).unwrap());
        text
    }
}

impl Records for Watched {
    fn last_prompt(&self) -> &str {
        &self.last
    }
    fn last_reply(&self) -> Option<&Reply> {
        self.reply.as_ref()
    }
}

/// One recorded reply: what was asked, what came back, and every line of how it came back.
fn row_of(call: &Call, key: &str, about: Value) -> Value {
    let reply = &call.reply;
    let mut row = serde_json::Map::new();
    row.insert("key".into(), json!(key));
    for (k, v) in about.as_object().unwrap() {
        row.insert(k.clone(), v.clone());
    }
    row.insert("prompt_chars".into(), json!(call.prompt.chars().count()));
    row.insert("prompt_sha".into(), json!(sha16(&call.prompt)));
    row.insert("recorded".into(), json!({
        "reason": reply.reason,
        "text_chars": reply.text.chars().count(),
        "text_sha": sha16(&reply.text),
        "text_head": reply.text.chars().take(120).collect::<String>(),
        "prompt_tokens": reply.prompt_tokens,
        "out_tokens": reply.out_tokens,
        "thought": reply.thought,
        //: Counted here the way the reader counts it, **off the wire**. It is not on `Reply` — with
        //: the stops off `out_tokens` is the server's own count — and it becomes observable from
        //: outside only when a stop fires.
        "arrived": arrived(&call.raw),
        "stream_lines": call.raw.len(),
        "stream_bytes": call.raw.iter().map(|l| l.len()).sum::<usize>(),
        "stream_print": stream_print(&call.raw),
        "seconds": (call.seconds * 1000.0).round() / 1000.0,
        "ttft": call.ttft.map(|t| (t * 1000.0).round() / 1000.0),
    }));
    row.insert("lines".into(),
               json!(call.raw.iter().map(|l| encode(l)).collect::<Vec<_>>()));
    let row = Value::Object(row);
    //: **Refused rather than stored** if the replay does not reproduce the call.
    check(&row).unwrap_or_else(|e| panic!("{e}"));
    row
}

fn opt(name: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
}

fn main() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap()
        .to_path_buf();
    let dir = root.join("measure");
    let plan_only = std::env::args().any(|a| a == "--plan");
    let arm = opt("--arm").unwrap_or_else(|| "off".into());
    let only = opt("--only");
    let d: Value = serde_json::from_str(
        &std::fs::read_to_string(root.join("native").join("expectations").join("replies.json"))
            .expect("run python native/expectations/all.py first")).unwrap();
    let spec = d["arms"].get(&arm).unwrap_or_else(|| panic!("no arm {arm:?}"));
    let (model_name, think) = (spec[0].as_str().unwrap(), spec[1].as_bool().unwrap());
    let comp = &d["composition"][&arm];
    let (do_conversations, single_repeats, stress_repeats) =
        (comp[0].as_bool().unwrap(), comp[1].as_u64().unwrap(), comp[2].as_u64().unwrap());

    let loaded = screens(&dir.join("screens"));
    let all = cases();
    let mut plan: Vec<(String, Value)> = Vec::new();
    if do_conversations {
        for scene in d["conversations"].as_array().unwrap() {
            let (id, distance, side, _turn, why) =
                (scene[0].as_str().unwrap(), scene[1].as_u64().unwrap() as usize,
                 scene[2].as_str().unwrap(), scene[3].as_u64().unwrap(), scene[4].as_str().unwrap());
            if only.as_deref().is_some_and(|o| o != id) {
                continue;
            }
            plan.push((format!("conversation {id}/{side}/d{distance}"),
                       json!({"kind": "conversation", "case": id, "distance": distance,
                              "arm_side": side, "why": why})));
        }
    }
    for _ in 0..single_repeats {
        for s in d["singles"].as_array().unwrap() {
            plan.push((format!("single {}", s[1].as_str().unwrap()),
                       json!({"kind": "single", "screen": s[0], "said": s[1], "why": s[2]})));
        }
    }
    for _ in 0..stress_repeats {
        for s in d["stress"].as_array().unwrap() {
            plan.push((format!("stress {}", s[1].as_str().unwrap()),
                       json!({"kind": "stress", "screen": s[0], "said": s[1], "why": s[2]})));
        }
    }

    let m = source_print(include_str!("../../../argmind-core/src/llm.rs"));
    let c = source_print(include_str!("../conversation.rs"));
    let l = source_print(include_str!("../../../argmind-core/src/session.rs"));
    //: **`h` covers the driver AND its scorer.** In the Python these were one file and one hash;
    //: splitting them in Rust left the scoring rules outside the fingerprint, so an edit to the
    //: judge produced a result under a name that claimed nothing had changed. That is the exact
    //: failure this letter was added for, reintroduced by the port and caught by a tie-break
    //: disagreement in `worst_index`.
    let h = source_print(&format!("{}{}", include_str!("replies-run.rs"),
                                  include_str!("../replies.rs")));
    let s = screen_print(&dir.join("screens"));
    let f = argmind_measure::conversation::cases_print(
        &all, &argmind_measure::conversation::FILLER.iter().map(|x| x.to_string())
            .collect::<Vec<_>>());
    let mut parts = vec![model_name.replace(':', "-"),
                         if think { "on".into() } else { "off".into() }, arm.clone(),
                         "rust".into(), prompt_tag(), format!("s{s}"), format!("f{f}"),
                         format!("h{h}"), format!("c{c}"), format!("l{l}"), format!("m{m}"),
                         //: **In the name, never inferred.** A corpus recorded with the stops on
                         //: holds only the prefixes the shipped logic already produces.
                         "stopsoff".into(), "x1".into()];
    if let Some(o) = &only {
        parts.push(o.clone());
    }
    let out = Destination::new(&dir, "replies", &parts);

    println!("  arm {arm}: {} | {} | thinking {}", spec[2].as_str().unwrap(), model_name,
             if think { "on" } else { "off" });
    println!("  {} scenes planned -> {}", plan.len(), out.path.file_name().unwrap()
        .to_string_lossy());
    for (label, about) in &plan {
        println!("    {:<52} {}", label.chars().take(52).collect::<String>(),
                 about["why"].as_str().unwrap().chars().take(76).collect::<String>());
    }
    if plan_only {
        return;
    }
    if let Err(why) = out.refuse_if_taken() {
        println!("\n  ** {why}");
        std::process::exit(3);
    }

    let mut model = Watched {
        inner: Local { think: Some(think), stop_when_answered: STOPS_OFF.0,
                       stop_when_looping: STOPS_OFF.1, ..Local::new(model_name) },
        calls: RefCell::new(Vec::new()), last: String::new(), reply: None };
    for _ in 0..2 {
        let _ = model.ask("Reply with the single word: ready");
    }
    model.calls.borrow_mut().clear();

    println!();
    let mut rows: Vec<Value> = Vec::new();
    for (label, about) in &plan {
        let before = model.calls.borrow().len();
        let (lines, screen) = match about["kind"].as_str().unwrap() {
            "conversation" => {
                let case = all.iter().find(|c| c.id == about["case"].as_str().unwrap()).unwrap();
                let side = about["arm_side"].as_str().unwrap();
                (script(case, about["distance"].as_u64().unwrap() as usize, side).unwrap(),
                 case.screen.clone())
            }
            _ => (vec![about["said"].as_str().unwrap().to_string()],
                  about["screen"].as_str().unwrap().to_string()),
        };
        let widgets = &loaded[&screen].widgets;
        //: Driven through the distance sweep's own `drive()`, so the prompts are that harness's
        //: prompts and not a second rendering of them.
        let (turns, back) = drive(model, widgets, &lines, Some(12), None).unwrap();
        model = back;
        //: **Matched to the turns by position, and the count is checked** — a retry that slipped a
        //: call in would otherwise shift every row's label by one.
        let calls = model.calls.borrow();
        let made = calls.len() - before;
        assert_eq!(made, turns.len(), "{label}: {made} calls for {} turns", turns.len());
        for (i, (call, turn)) in calls[before..].iter().zip(&turns).enumerate() {
            let key = format!("{label}/t{i}");
            let mut about = about.clone();
            about["turn"] = json!(i);
            about["said"] = json!(turn.said);
            about["screen"] = json!(screen);
            about["arm"] = json!(about.get("arm_side").cloned().unwrap_or(json!("-")));
            about["distance"] = about.get("distance").cloned().unwrap_or(json!(-1));
            about["case"] = about.get("case").cloned().unwrap_or(json!("-"));
            let row = row_of(call, &key, about);
            let r = &row["recorded"];
            println!("    {:<46} {:<10} out {:>5}  lines {:>5}  {:>6.2}s  {:>5} chars",
                     key.chars().take(46).collect::<String>(),
                     r["reason"].as_str().unwrap_or("-"), r["out_tokens"], r["stream_lines"],
                     r["seconds"].as_f64().unwrap_or(0.0), r["text_chars"]);
            rows.push(row);
        }
    }

    println!("\n  {} replies recorded", rows.len());
    for (shape, n) in coverage(&rows) {
        println!("    {shape:<44} {n:>4}");
    }
    let disagreed = disagreement(&rows);
    if !disagreed.is_empty() {
        println!("\n  prompts asked more than once:");
        for d in &disagreed {
            println!("    {:<52} n={} {} {:?}",
                     d["prompt"].as_str().unwrap().chars().take(52).collect::<String>(), d["n"],
                     if d["identical_streams"].as_bool().unwrap() { "identical" } else { "DIFFER" },
                     d["reasons"]);
        }
    }

    if let Err(why) = out.refuse_if_taken() {
        println!("\n  ** {why}");
        std::process::exit(3);
    }
    std::fs::write(&out.path, serde_json::to_string(&json!({
        "model": model_name, "think": if think { "on" } else { "off" }, "arm": arm,
        "prompt": prompt_tag()[1..].to_string(), "screens_print": s, "data_print": f,
        "harness_print": h, "conversation_print": c, "loop_print": l, "llm_print": m,
        "loop_print_lang": "rust: native/argmind-core/src/session.rs",
        "stop_when_answered": STOPS_OFF.0, "stop_when_looping": STOPS_OFF.1,
        "rows": rows,
    })).unwrap()).unwrap();
    println!("\n  every line of every reply -> {}", out.path.display());
}
