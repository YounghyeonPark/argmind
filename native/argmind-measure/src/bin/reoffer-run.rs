//! Saying yes to an offer, saying no to it, and asking about it later — against a live model.
//!
//! ```text
//!   reoffer-run              104 cases x 3 repeats, worst counted
//!   reoffer-run 1            one repeat, while working
//! ```
//!
//! Six groups over four screens, and the one that carries the result is **ACCEPT**: the person says
//! yes one turn after an offer and the agent should do the thing again.
//!
//! **The controls are not decoration.** MEMORY asks about the past and the right answer is to do
//! nothing; REFUSE says no and the right answer is to do nothing; ANEW asks for something else and
//! the right answer is that thing. Without them, an agent that acts on anything following an offer
//! scores perfectly on the primary endpoint and is worse to live with.
//!
//! **The committed baselines here are frozen** — any edit to the Python harness moves its `h` and
//! makes its `--compare` refuse — so this writes under `reoffer-rust-…` and does not pretend to be
//! an arm of them. What it is for is running the experiment again once the Python is gone.

use std::path::PathBuf;

use argmind_core::contract::Speaks;
use argmind_core::llm::Reply;
use argmind_measure::conversation::{drive, Records};
use argmind_measure::reoffer::{good, key_of, narrated, offer_reached, offered_lines, shape_of,
                               slot_lines, slot_shape, verdict_of, worst_order};
use argmind_measure::saying::worst_index;
use argmind_measure::{data_print, prompt_tag, screen_print, screens, source_print, Destination};
use argmind_env_ollama::Local;
use serde_json::{json, Value};

/// The model, keeping **every** prompt rather than only the last one.
///
/// One prompt per turn, and the driver refuses rather than lining them up by guess if it ever is
/// not — a list that quietly ran long would pair the wrong prompt with the wrong turn.
struct Tapped {
    inner: Local,
    prompts: Vec<String>,
    reply: Option<Reply>,
    retried: usize,
    calls: usize,
}

impl Speaks for Tapped {
    fn ask(&mut self, prompt: &str) -> String {
        self.prompts.push(prompt.to_string());
        self.calls += 1;
        for attempt in 0..4 {
            match self.inner.ask(prompt) {
                Ok(r) => {
                    let text = r.text.clone();
                    self.reply = Some(r);
                    return text;
                }
                Err(e) => {
                    if attempt == 3 {
                        panic!("the model failed four times running: {e}");
                    }
                    self.retried += 1;
                    println!("    [retry {}/3 after {e}]", attempt + 1);
                    std::thread::sleep(std::time::Duration::from_secs(2 * (attempt + 1) as u64));
                }
            }
        }
        unreachable!()
    }
}

impl Records for Tapped {
    fn last_prompt(&self) -> &str {
        self.prompts.last().map(String::as_str).unwrap_or("")
    }
    fn last_reply(&self) -> Option<&Reply> {
        self.reply.as_ref()
    }
}

fn main() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap()
        .to_path_buf();
    let repeat: usize = std::env::args().nth(1).and_then(|a| a.parse().ok()).unwrap_or(3);
    assert!(repeat >= 1, "worst-of-N has no meaning at N=0");
    let only = std::env::args().nth(2);
    let dir = root.join("measure");
    let loaded = screens(&dir.join("screens"));
    let d: Value = serde_json::from_str(
        &std::fs::read_to_string(root.join("native").join("expectations").join("reoffer.json"))
            .expect("run python native/expectations/all.py first")).unwrap();

    let all = d["cases"].as_array().unwrap();
    let cases: Vec<&Value> = match &only {
        Some(k) => all.iter().filter(|c| c["kind"].as_str() == Some(k.as_str())).collect(),
        None => all.iter().collect(),
    };
    assert!(!cases.is_empty(), "no case of that kind");
    for c in &cases {
        assert!(loaded.contains_key(c["screen"].as_str().unwrap()), "no fixture {:?}", c["screen"]);
    }
    let remember = d["remember_turns"].as_i64();
    //: `f` over the case list exactly as the Python hashed it — a case list edited between two runs
    //: is two experiments sharing one name.
    let f = data_print(&json!(all));
    assert_eq!(f, d["data_print"].as_str().unwrap(), "the cases no longer hash to the recorded f");

    let s = screen_print(&dir.join("screens"));
    //: **`h` covers the driver AND its scorer.** In the Python these were one file and one hash;
    //: splitting them in Rust left the scoring rules outside the fingerprint, so an edit to the
    //: judge produced a result under a name that claimed nothing had changed. That is the exact
    //: failure this letter was added for, reintroduced by the port and caught by a tie-break
    //: disagreement in `worst_index`.
    let h = source_print(&format!("{}{}", include_str!("reoffer-run.rs"),
                                  include_str!("../reoffer.rs")));
    let l = source_print(include_str!("../../../argmind-core/src/session.rs"));
    let mut parts = vec!["rust".to_string(), prompt_tag(), format!("s{s}"), format!("f{f}"),
                         format!("h{h}"), format!("l{l}"), format!("x{repeat}")];
    if let Some(k) = &only {
        parts.push(k.clone());
    }
    let out = Destination::new(&dir, "reoffer", &parts);
    if let Err(why) = out.refuse_if_taken() {
        println!("\n  ** {why}");
        std::process::exit(3);
    }

    let mut model = Tapped { inner: Local { think: Some(false), ..Local::new("qwen3:8b") },
                             prompts: Vec::new(), reply: None, retried: 0, calls: 0 };
    for _ in 0..2 {
        let _ = model.ask("Reply with the single word: ready");
    }
    model.calls = 0;

    println!("\n  qwen3:8b | thinking off | prompt {} | screens s{s} | cases f{f}", prompt_tag());
    println!("  harness h{h} | session.rs l{l} | remember_turns {:?}", remember);
    println!("  {} cases x {repeat} repeats, worst counted, nothing clicked\n", cases.len());

    let mut rows: Vec<Value> = Vec::new();
    let mut shapes: std::collections::BTreeMap<String, usize> = Default::default();
    let mut slots: std::collections::BTreeMap<String, usize> = Default::default();
    let started = std::time::Instant::now();
    for case in &cases {
        let kind = case["kind"].as_str().unwrap();
        let label = case["label"].as_str().unwrap();
        let want: Option<(&str, &str)> = case["want"].as_array()
            .map(|a| (a[0].as_str().unwrap(), a[1].as_str().unwrap()));
        let filler: Vec<String> = case.get("filler").and_then(|x| x.as_array())
            .map(|a| a.iter().map(|x| x.as_str().unwrap().to_string()).collect())
            .unwrap_or_default();
        let mut lines = vec![case["ask"].as_str().unwrap().to_string()];
        lines.extend(filler.iter().cloned());
        lines.push(case["then"].as_str().unwrap().to_string());

        let mut draws: Vec<Value> = Vec::new();
        for _ in 0..repeat {
            //: **Cleared per case, checked per case.** The warm-up calls and an earlier case's
            //: prompts are not this case's, and a list that ran long would pair the wrong prompt
            //: with the wrong turn.
            model.prompts.clear();
            let (turns, back) = drive(model, &loaded[case["screen"].as_str().unwrap()].widgets,
                                      &lines, remember, None).unwrap();
            model = back;
            assert_eq!(model.prompts.len(), turns.len(),
                       "{} prompts for {} turns on {label}: they cannot be paired, so the slot \
                        check would be reading somebody else's prompt",
                       model.prompts.len(), turns.len());
            let as_json: Vec<Value> = turns.iter().map(|t| t.to_json()).collect();
            let first: Vec<Value> = as_json[0]["effects"].as_array().unwrap().clone();
            let last = as_json.last().unwrap();
            let second: Vec<Value> = last["effects"].as_array().unwrap().clone();
            let window = last["window"].as_str().unwrap_or("");
            let seen = offered_lines(window);
            for line in &seen {
                *shapes.entry(shape_of(line)).or_default() += 1;
            }
            //: The other channel: what the prompt put between the widget list and the person's
            //: words, on **every** turn, counted the same way the offered lines are.
            let slot: Vec<Vec<String>> = model.prompts.iter()
                .map(|p| slot_lines(p).unwrap_or_else(|e| panic!("{label}: {e}"))).collect();
            for one in &slot {
                *slots.entry(slot_shape(one)).or_default() += 1;
            }
            let middle = &turns[1..turns.len() - 1];
            draws.push(json!({
                "verdict": verdict_of(kind, &first, &second, window, want),
                "offered": first.iter().map(|e| { let k = key_of(e); json!([k.0, k.1, k.2]) })
                    .collect::<Vec<_>>(),
                "again": second.iter().map(|e| { let k = key_of(e); json!([k.0, k.1, k.2]) })
                    .collect::<Vec<_>>(),
                //: The two presences, apart. `offer_reached` is the treatment; the count of lines
                //: is how much else was in the block beside it.
                "offer_reached": offer_reached(&first, window),
                "offered_lines": seen,
                "narrated": narrated(&first, last["reply"].as_str().unwrap_or("")),
                //: **The treatment, on the scored turn, verbatim** — and the count over the whole
                //: conversation, so a slot present on the filler turns and absent on the scored one
                //: is not read as a slot that was there.
                "slot": slot.last().cloned().unwrap_or_default(),
                "slot_turns": slot.iter().filter(|x| !x.is_empty()).count(),
                //: Never a bare boolean: what the filler turns did, so a drift at distance 5 can be
                //: read against what else was on offer by then.
                "filler_effects": middle.iter().map(|t| t.n_effects).sum::<usize>(),
                "turns": as_json,
            }));
        }
        let verdicts: Vec<&str> = draws.iter().map(|d| d["verdict"].as_str().unwrap()).collect();
        let order = worst_order(kind);
        let i = worst_index(&verdicts, order);
        let counted = verdicts[i];
        let row = json!({
            "kind": kind, "case": label, "screen": case["screen"],
            "form": case.get("form").cloned().unwrap_or(Value::Null),
            //: **Turns from the offer to the scored turn, 1 = adjacent.** Four filler turns read as
            //: 5, not 4: the yes is the sixth turn and the offer was the first.
            "distance": 1 + filler.len(),
            "lang": case.get("lang").cloned().unwrap_or(json!("ko")),
            "ask": case["ask"], "then": case["then"], "want": case["want"], "why": case["why"],
            "verdicts": verdicts, "counted": counted, "from_repeat": i,
            "agreed": verdicts.iter().collect::<std::collections::BTreeSet<_>>().len() == 1,
            "ok": counted == good(kind), "void": counted == argmind_measure::reoffer::VOID,
            "narrated_counted": draws[i]["narrated"],
            "repeats": draws,
        });
        println!("    {:<9} {:<10} {:<44} {}", counted.to_uppercase(), kind,
                 label.chars().take(44).collect::<String>(),
                 if row["agreed"].as_bool().unwrap() { "" } else { "(disagreed)" });
        rows.push(row);
    }
    let took = started.elapsed().as_secs_f64();

    println!("\n  BY GROUP — the right answer for each, worst of {repeat}");
    for group in d["groups"].as_array().unwrap() {
        let g = group.as_str().unwrap();
        let here: Vec<&Value> = rows.iter().filter(|r| r["kind"] == g).collect();
        if here.is_empty() {
            continue;
        }
        let real: Vec<&&Value> = here.iter().filter(|r| !r["void"].as_bool().unwrap()).collect();
        let ok = real.iter().filter(|r| r["ok"].as_bool().unwrap()).count();
        let mut tally: std::collections::BTreeMap<&str, usize> = Default::default();
        for r in &here {
            *tally.entry(r["counted"].as_str().unwrap()).or_default() += 1;
        }
        //: **A void case is one that could not be asked**, printed apart from the rate rather than
        //: folded into it — counting it as a failure would let a harness that stopped writing
        //: offers score well.
        println!("    {g:<10} {ok} of {} ({} correct: {})   {tally:?}{}",
                 real.len(), g, good(g),
                 if here.len() == real.len() { String::new() }
                 else { format!("  [{} void]", here.len() - real.len()) });
    }

    //: **The delivery check, read generically.** If every offered line and every slot has the same
    //: shape across the run there was no treatment in the transcript at all — and a check that
    //: compared the lines themselves would call a widget change a treatment.
    println!("\n  OFFERED-LINE SHAPES  {shapes:?}");
    println!("  SLOT SHAPES          {slots:?}");
    let narrated_n = rows.iter().filter(|r| r["narrated_counted"].as_bool().unwrap()).count();
    //: A floor on narration and never a rate: widget names are English and the replies are Korean.
    println!("\n  the reply named the offered widget in {narrated_n} of {} cases — a FLOOR, since \
              a correct Korean account of offering `Copy` contains no `copy`", rows.len());
    println!("\n  {} cases, {} model calls, {:.1} min, {} retried",
             rows.len(), model.calls, took / 60.0, model.retried);

    if let Err(why) = out.refuse_if_taken() {
        println!("\n  ** {why}");
        std::process::exit(3);
    }
    std::fs::write(&out.path, serde_json::to_string(&json!({
        "model": "qwen3:8b", "think": "off", "prompt": prompt_tag()[1..].to_string(),
        "prompt_scheme": "r = a hash of the rendered prompt over six probes",
        "screens_print": s, "data_print": f, "harness_print": h,
        "loop_print": l, "loop_print_lang": "rust: native/argmind-core/src/session.rs",
        "remember_turns": remember, "remember_unit": "turns",
        "repeat": repeat, "only": only, "seconds": (took * 10.0).round() / 10.0,
        "calls": model.calls, "retried": model.retried,
        "offered_shapes": shapes, "slot_shapes": slots,
        "cases": all, "filler_far": d["filler_far"],
        "screens": loaded.iter().map(|(k, v)| (k.clone(), json!(v.widgets.len())))
            .collect::<serde_json::Map<_, _>>(),
        "rows": rows,
    })).unwrap()).unwrap();
    println!("\n  every turn of every case -> {}", out.path.display());
}
