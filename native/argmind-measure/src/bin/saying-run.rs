//! Does an internal widget index leak into the speech?
//!
//! ```text
//!   saying-run            32 leak-set + 12 control-set cases x 3 repeats, worst counted
//!   saying-run 1          one repeat, while working
//! ```
//!
//! **The control set is not decoration.** Twelve of the cases are goals whose right answers
//! *contain numbers* — *"버튼이 몇 개야?"* — so the leak measurement cannot be satisfied by a model
//! that has simply stopped saying numbers at all. Citing the leak rate without them is citing half
//! a measurement.
//!
//! The hard part is not finding digits, it is **not counting the legitimate ones**: `Alt+F4`,
//! `U+0021` and `i9-13900HX` are on the screen, and a number that came off the screen is a
//! quotation. Five verdicts, worst counted, and the two that mean something different from each
//! other are `stray` (in range, not provably an index) and `index` (the number of a thing this very
//! turn clicked).
//!
//! Nothing is clicked: the fixtures are captured screens and there are no hands here at all.

use std::path::PathBuf;

use argmind_core::contract::Widget;
use argmind_core::converse::{describe_screen, read_reply, render, MAX_LISTED};
use argmind_core::intent::Trigger;
use argmind_core::llm::Reply;
use argmind_env_ollama::Local;
use argmind_measure::saying::{answer_verdict, classify, has_number, worst_index, ANSWER_ORDER,
                              LEAK_ORDER};
use argmind_measure::{data_print, prompt_tag, screen_print, screens, source_print, spend,
                      Destination, Tables};
use serde_json::{json, Value};

/// The model, with the last prompt kept — the only way `check_numbering` can be honest.
struct Recorder {
    inner: Local,
    last: String,
    reply: Option<Reply>,
}

impl Recorder {
    fn ask(&mut self, prompt: &str) -> String {
        self.last = prompt.to_string();
        match self.inner.ask(prompt) {
            Ok(r) => {
                let text = r.text.clone();
                self.reply = Some(r);
                text
            }
            Err(e) => panic!("the model did not answer: {e}"),
        }
    }
}

/// **Abort unless the model really was shown `0:`-based numbering.**
///
/// Everything on the leak axis rests on *"this number is a list index"*. If the prompt ever
/// renumbers from 1, or stops numbering, every index leak would silently reclassify as `off-list`
/// and the headline would improve for a reason that has nothing to do with the model. A harness
/// that cannot tell those apart reports a clean zero, which is the failure this project keeps
/// finding.
fn check_numbering(prompt: &str, listed: &[Widget]) {
    let block = describe_screen(listed, MAX_LISTED);
    assert!(prompt.contains(&block),
            "the widget block this harness reconstructs is not in the prompt that was sent — \
             converse's screen format has moved, and every `index` verdict here would be wrong. \
             Prompt starts: {:?}", prompt.chars().take(200).collect::<String>());
}

fn main() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap()
        .to_path_buf();
    let repeat: usize = std::env::args().nth(1).and_then(|a| a.parse().ok()).unwrap_or(3);
    assert!(repeat >= 1, "--repeat must be at least 1: worst-of-N has no meaning at N=0");
    let dir = root.join("measure");
    let loaded = screens(&dir.join("screens"));
    let d: Value = serde_json::from_str(
        &std::fs::read_to_string(root.join("native").join("expectations").join("saying.json"))
            .expect("run python native/expectations/all.py first")).unwrap();
    let tables = Tables::from_json(&d);

    //: `f` over both case lists exactly as the Python hashed them, so a Rust run and a Python run
    //: of the same experiment carry the same letter and a different one means the cases moved.
    let f = data_print(&json!([d["leaking"], d["numbers"]]));
    assert_eq!(f, d["data_print"].as_str().unwrap(),
               "the case lists no longer hash to what the Python recorded");

    let mut cases: Vec<(&str, &Value)> = Vec::new();
    for c in d["leaking"].as_array().unwrap() {
        cases.push(("leak", c));
    }
    for c in d["numbers"].as_array().unwrap() {
        cases.push(("number", c));
    }
    //: **A self-check on the labels, not on the model.** The leak set's whole claim is *a correct
    //: reply here contains no number*, and a request that itself contains one would break it — the
    //: model would be right to repeat it back.
    for (kind, c) in &cases {
        if *kind == "leak" {
            let said = c[2].as_str().unwrap();
            assert!(!said.chars().any(|ch| ch.is_ascii_digit()),
                    "leak-set request {said:?} contains a digit — the label does not hold for it");
        }
        assert!(loaded.contains_key(c[1].as_str().unwrap()), "no fixture {:?}", c[1]);
    }

    let s = screen_print(&dir.join("screens"));
    //: **`h` covers the driver AND its scorer.** In the Python these were one file and one hash;
    //: splitting them in Rust left the scoring rules outside the fingerprint, so an edit to the
    //: judge produced a result under a name that claimed nothing had changed. That is the exact
    //: failure this letter was added for, reintroduced by the port and caught by a tie-break
    //: disagreement in `worst_index`.
    let h = source_print(&format!("{}{}", include_str!("saying-run.rs"),
                                  include_str!("../saying.rs")));
    let out = Destination::new(&dir, "saying", &["rust".into(), prompt_tag(), format!("s{s}"),
                                                 format!("f{f}"), format!("h{h}"),
                                                 format!("x{repeat}")]);
    if let Err(why) = out.refuse_if_taken() {
        println!("\n  ** {why}");
        std::process::exit(3);
    }

    let mut model = Recorder { inner: Local { think: Some(false), ..Local::new("qwen3:8b") },
                               last: String::new(), reply: None };
    for _ in 0..2 {
        let _ = model.ask("Reply with the single word: ready");
    }

    let n_leak = cases.iter().filter(|(k, _)| *k == "leak").count();
    println!("\n  qwen3:8b | thinking off | prompt {} | screens s{s} | cases f{f}", prompt_tag());
    println!("  harness h{h}");
    println!("  {n_leak} leak-set + {} control-set cases x {repeat} repeats, worst counted | \
              {} fixture screens, nothing clicked\n", cases.len() - n_leak, loaded.len());

    let mut checked: std::collections::BTreeSet<String> = Default::default();
    let mut rows: Vec<Value> = Vec::new();
    let started = std::time::Instant::now();
    for (kind, c) in &cases {
        let (group, screen, said) = (c[0].as_str().unwrap(), c[1].as_str().unwrap(),
                                     c[2].as_str().unwrap());
        let widgets = &loaded[screen].widgets;
        let listed: Vec<Widget> = widgets.iter().take(MAX_LISTED).cloned().collect();
        let names: Vec<String> = listed.iter().map(|w| w.name.clone()).collect();
        let mut repeats: Vec<Value> = Vec::new();
        for _ in 0..repeat {
            let trigger = Trigger::said(said);
            //: **No history**, which is this harness's whole blind spot and is declared rather than
            //: forgotten — see the caveat printed at the end.
            let prompt = render(&listed, &trigger, &[], MAX_LISTED);
            let t0 = std::time::Instant::now();
            let raw = model.ask(&prompt);
            let took = t0.elapsed().as_secs_f64();
            if checked.insert(screen.to_string()) {
                check_numbering(&model.last, &listed);
            }
            let intent = read_reply(&raw, &listed, &trigger);
            //: **Name and role**, which is how the reply was resolved in the first place. A name
            //: alone matches the `text` label sitting in front of the combo box of the same name,
            //: and that exact confusion was a real defect.
            let mut clicked_at: Vec<usize> = Vec::new();
            for e in intent.effects.iter().filter(|e| e.kind == "click") {
                for (i, w) in listed.iter().enumerate() {
                    if w.name == e.target && w.role == e.role {
                        clicked_at.push(i);
                    }
                }
            }
            clicked_at.sort_unstable();
            clicked_at.dedup();
            let (leak, found) = classify(&intent.say, &names, &clicked_at, &tables);
            let answer = answer_verdict(&intent.say, said, &names, &tables);
            let (present, how) = has_number(&intent.say, &tables.number_words);
            let mut rep = json!({
                //: **`say` verbatim, every repeat.** A later reading gets to ask a question nobody
                //: has thought of yet, which is not possible from a verdict.
                "say": intent.say,
                //: **What the model actually emitted**, beside what was parsed out of it. A leak in
                //: a field nobody reads, or a `say` that lost half its text to the extractor, is
                //: answerable from this and from nothing else.
                "raw": raw,
                "effects": intent.effects.iter().map(|e| json!({
                    "kind": e.kind, "target": e.target, "role": e.role, "text": e.text}))
                    .collect::<Vec<_>>(),
                "clicked_at": clicked_at,
                "leak": leak,
                "numbers": found.iter().map(|n| json!({
                    "run": n.run, "token": n.token, "kind": n.kind, "in_range": n.in_range,
                    "clicked": n.clicked, "context": n.context})).collect::<Vec<_>>(),
                "answer": answer, "has_number": present, "found_by": how,
            });
            //: Latency, generated length and thinking characters — because latency alone cannot
            //: tell a busy machine from a degeneration, and this harness's `empty` verdict is
            //: exactly what a reply that ran to the token ceiling produces.
            for (k, v) in spend(model.reply.as_ref(), Some(took)).as_object().unwrap() {
                rep[k] = v.clone();
            }
            repeats.push(rep);
        }

        let leaks: Vec<&str> = repeats.iter().map(|r| r["leak"].as_str().unwrap()).collect();
        let answers: Vec<&str> = repeats.iter().map(|r| r["answer"].as_str().unwrap()).collect();
        let li = worst_index(&leaks, &LEAK_ORDER);
        let ai = worst_index(&answers, &ANSWER_ORDER);
        let mut row = json!({
            "set": kind, "group": group, "screen": screen, "said": said,
            "n_listed": listed.len(),
            "leak_counted": leaks[li], "leak_from_repeat": li,
            "answer_counted": answers[ai], "answer_from_repeat": ai,
            "leak_agreed": leaks.iter().collect::<std::collections::BTreeSet<_>>().len() == 1,
            "answer_agreed": answers.iter().collect::<std::collections::BTreeSet<_>>().len() == 1,
            "repeats": repeats.clone(),
        });
        if *kind == "number" {
            let present: Vec<bool> = repeats.iter()
                .map(|r| r["has_number"].as_bool().unwrap()).collect();
            //: Worst of the repeats here too: one silent reply out of three is a silent reply.
            row["must"] = c[3].clone();
            row["why"] = c[4].clone();
            row["number_present"] = json!(present.iter().all(|x| *x));
            row["present_each"] = json!(present);
        }
        let mark = if row["leak_agreed"].as_bool().unwrap() { "" } else { "  (disagreed)" };
        println!("    {:<9} {:<11} {:<18} {:<30} {}{mark}",
                 leaks[li].to_uppercase(), kind, group.chars().take(18).collect::<String>(),
                 said.chars().take(30).collect::<String>(), answers[ai]);
        rows.push(row);
    }
    let took = started.elapsed().as_secs_f64();

    // ---- the reading ---------------------------------------------------------------------------
    let leak_rows: Vec<&Value> = rows.iter().filter(|r| r["set"] == "leak").collect();
    let leaked = leak_rows.iter()
        .filter(|r| ["off-list", "stray", "index"]
                .contains(&r["leak_counted"].as_str().unwrap())).count();
    println!("\n  LEAK — a number that did not come off the screen, worst of {repeat}");
    println!("    {leaked} of {} leak-set cases leaked", leak_rows.len());
    let mut kinds: std::collections::BTreeMap<&str, usize> = Default::default();
    for r in &leak_rows {
        *kinds.entry(r["leak_counted"].as_str().unwrap()).or_default() += 1;
    }
    println!("    {kinds:?}");

    //: **The control set, which is what stops a model that stopped saying numbers scoring well.**
    let control: Vec<&Value> = rows.iter().filter(|r| r["set"] == "number").collect();
    let kept = control.iter().filter(|r| r["number_present"].as_bool().unwrap()).count();
    println!("\n  CONTROL — goals whose right answer contains a number");
    println!("    {kept} of {} kept their legitimate number", control.len());
    if kept < control.len() {
        for r in &control {
            if !r["number_present"].as_bool().unwrap() {
                println!("      DROPPED {}", r["said"].as_str().unwrap());
            }
        }
    }

    let mut answers: std::collections::BTreeMap<&str, usize> = Default::default();
    for r in &rows {
        *answers.entry(r["answer_counted"].as_str().unwrap()).or_default() += 1;
    }
    println!("\n  ANSWER SHAPE over all {} cases: {answers:?}", rows.len());
    println!("\n  {} cases, {:.1} min", rows.len(), took / 60.0);
    //: **The caveat is not decoration.** Every number above was produced with no conversation in
    //: the prompt, and a reader who does not know that will read it as covering something it
    //: cannot see.
    println!("{}", argmind_measure::sending::caveat(
        &dir, &prompt_tag(),
        &source_print(include_str!("../../../argmind-core/src/session.rs"))));

    if let Err(why) = out.refuse_if_taken() {
        println!("\n  ** {why}");
        std::process::exit(3);
    }
    std::fs::write(&out.path, serde_json::to_string(&json!({
        "model": "qwen3:8b", "think": "off", "prompt": prompt_tag()[1..].to_string(),
        "prompt_scheme": "r = a hash of the rendered prompt over six probes",
        "screens_print": s, "data_print": f, "harness_print": h,
        "repeat": repeat, "seconds": (took * 10.0).round() / 10.0,
        "screens": loaded.iter().map(|(k, v)| (k.clone(), json!(v.widgets.len())))
            .collect::<serde_json::Map<_, _>>(),
        "rows": rows,
    })).unwrap()).unwrap();
    println!("\n  every reply -> {}", out.path.display());
}
