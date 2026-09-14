//! Re-judge every committed `saying` run: the leak verdict, the answer verdict, and every number.
//!
//! No model, no network. Each row carries what was said, what was clicked and the screen it was
//! said on, so both classifiers can be re-run over months of prompt changes — including the two
//! prompt orderings, where the leak went from 0 of 32 to 5 of 32 and back.

use std::path::PathBuf;

use argmind_measure::saying::{answer_verdict, classify, has_number, is_echo, strip_punct,
                              worst_index, ANSWER_ORDER, CLEAN, EMPTY, INDEX, LEAK_ORDER,
                              OFFLIST, QUOTED, STRAY};
use argmind_measure::{screens, Tables};
use serde_json::Value;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap().to_path_buf()
}

fn tables() -> (Tables, Value) {
    let p = root().join("native").join("expectations").join("saying.json");
    let d: Value = serde_json::from_str(&std::fs::read_to_string(&p).unwrap_or_else(|_| {
        panic!("{} is missing. Generate it: python native/expectations/all.py", p.display())
    })).unwrap();
    (Tables::from_json(&d), d)
}

fn runs() -> Vec<(String, Value)> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(root().join("measure")).unwrap().flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with("saying-") && name.ends_with(".json") {
            let d: Value = serde_json::from_str(&std::fs::read_to_string(entry.path()).unwrap())
                .unwrap();
            if d["rows"].as_array().is_some_and(|r| !r.is_empty()) {
                out.push((name, d));
            }
        }
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

#[test]
fn every_number_in_every_committed_reply_is_classified_the_same() {
    let (t, _) = tables();
    let loaded = screens(&root().join("measure").join("screens"));
    let files = runs();
    assert!(files.len() >= 4, "only {} saying results found", files.len());
    let (mut rows, mut numbers) = (0, 0);
    let mut kinds: std::collections::BTreeMap<&str, usize> = Default::default();
    for (name, d) in &files {
        for row in d["rows"].as_array().unwrap() {
            let screen = row["screen"].as_str().unwrap();
            let names: Vec<String> = loaded[screen].widgets.iter()
                .take(row["n_listed"].as_u64().unwrap() as usize)
                .map(|w| w.name.clone()).collect();
            let mut leaks = Vec::new();
            for rep in row["repeats"].as_array().unwrap() {
                let clicked_at: Vec<usize> = rep["clicked_at"].as_array().unwrap().iter()
                    .map(|v| v.as_u64().unwrap() as usize).collect();
                let (leak, found) = classify(rep["say"].as_str().unwrap(), &names, &clicked_at, &t);
                assert_eq!(leak, rep["leak"].as_str().unwrap(),
                           "{name} / {:?}", row["said"].as_str().unwrap());
                let want = rep["numbers"].as_array().unwrap();
                assert_eq!(found.len(), want.len(), "{name} number count");
                for (got, w) in found.iter().zip(want) {
                    assert_eq!(got.run, w["run"].as_str().unwrap());
                    assert_eq!(got.token, w["token"].as_str().unwrap());
                    assert_eq!(got.kind, w["kind"].as_str().unwrap(), "{name} / {}", got.run);
                    assert_eq!(got.in_range, w["in_range"].as_bool().unwrap());
                    numbers += 1;
                    *kinds.entry(got.kind).or_default() += 1;
                }
                leaks.push(leak);
            }
            assert_eq!(leaks[worst_index(&leaks, &LEAK_ORDER)],
                       row["leak_counted"].as_str().unwrap(), "{name} worst leak");
            rows += 1;
        }
    }
    assert!(rows >= 150, "only {rows} rows re-judged");
    //: **The classifier has to be seen using more than one of its words**, or this is a test that
    //: everything is clean.
    assert!(kinds.get(INDEX).copied().unwrap_or(0) > 0, "no leak in any recorded run: {kinds:?}");
    assert!(kinds.get(QUOTED).copied().unwrap_or(0) > 0, "nothing quoted: {kinds:?}");
    println!("  {numbers} numbers over {rows} rows: {kinds:?}");
}

#[test]
fn the_answer_verdicts_come_out_the_same() {
    let (t, _) = tables();
    let loaded = screens(&root().join("measure").join("screens"));
    let mut seen: std::collections::BTreeMap<&str, usize> = Default::default();
    for (name, d) in &runs() {
        for row in d["rows"].as_array().unwrap() {
            let names: Vec<String> = loaded[row["screen"].as_str().unwrap()].widgets.iter()
                .take(row["n_listed"].as_u64().unwrap() as usize)
                .map(|w| w.name.clone()).collect();
            let asked = row["said"].as_str().unwrap();
            let mut got = Vec::new();
            for rep in row["repeats"].as_array().unwrap() {
                let v = answer_verdict(rep["say"].as_str().unwrap(), asked, &names, &t);
                if let Some(want) = rep.get("answer").and_then(|x| x.as_str()) {
                    assert_eq!(v, want, "{name} / {asked:?}");
                }
                *seen.entry(v).or_default() += 1;
                got.push(v);
            }
            assert_eq!(got[worst_index(&got, &ANSWER_ORDER)],
                       row["answer_counted"].as_str().unwrap(), "{name} / {asked:?} worst answer");
        }
    }
    println!("  answer verdicts seen: {seen:?}");
    assert!(seen.len() > 1, "only one answer verdict ever occurs: {seen:?}");
}

/// **The hard part is not finding digits, it is not counting the legitimate ones.**
#[test]
fn a_number_that_came_from_the_screen_is_not_a_leak() {
    let (t, _) = tables();
    //: `Page 3` and `A` are here because three mutants survived without them: no recorded reply
    //: exercises a **one-character token**, a **partial echo**, or a **one-character widget name**,
    //: and each is a guard doing real work.
    let names: Vec<String> = ["Alt+F4", "U+0021", "i9-13900HX", "Eraser", "Close", "Page 3", "A"]
        .iter().map(|s| s.to_string()).collect();

    //: Quoted: the token is on screen as a whole token.
    assert_eq!(classify("Alt+F4 를 누르세요", &names, &[], &t).0, QUOTED);
    assert_eq!(classify("U+0021 입니다", &names, &[], &t).0, QUOTED);
    assert_eq!(classify("i9-13900HX 입니다", &names, &[], &t).0, QUOTED);
    //: The index of a thing just clicked, in a sentence. The defect itself.
    assert_eq!(classify("지우개는 3번 버튼입니다", &names, &[3], &t).0, INDEX);
    //: In range but not clicked — suspicious, and not provably the index.
    assert_eq!(classify("3개 있어요", &names, &[], &t).0, STRAY);
    //: Out of range: wrong, but a different defect.
    assert_eq!(classify("122개 있어요", &names, &[], &t).0, OFFLIST);
    assert_eq!(classify("복사할게요", &names, &[], &t).0, CLEAN);

    //: A bare number is not an answer; an empty reply is not either, and they are different facts.
    assert_eq!(answer_verdict("3, 122", "몇 개야?", &names, &t), "numeric");
    assert_eq!(answer_verdict("   ", "몇 개야?", &names, &t), EMPTY);
    assert_eq!(answer_verdict("Eraser", "뭐 눌렀어?", &names, &t), "bare-name");
    assert_eq!(answer_verdict("네", "할 수 있어?", &names, &t), "bare-ack");
    assert_eq!(answer_verdict("몇 개야?", "몇 개야?", &names, &t), "echo");
    assert_eq!(answer_verdict("버튼이 다섯 개 있어요", "몇 개야?", &names, &t), "answered");

    //: **A one-character token is never a quotation.** `3` occurs on screen inside `Page 3` as a
    //: whole token, and treating that as the source of *"3개 있어요"* would excuse a leak on any
    //: screen that happens to number something.
    assert_eq!(classify("3개 있어요", &names, &[], &t).0, STRAY);
    //: **A bare name has to be more than one character**, or every reply containing `A` is one.
    assert_eq!(answer_verdict("A", "뭐 눌렀어?", &names, &t), "answered");
    //: **The echo threshold does work.** A two-character reply inside an eleven-character question
    //: is an answer, not the question repeated back.
    assert_eq!(answer_verdict("복사", "복사 버튼이 어디 있어?", &names, &t), "answered");
    assert!(is_echo("abc", "abc", 0.6));
    assert!(!is_echo("", "abc", 0.6));
    assert_eq!(strip_punct("네, 알겠습니다!", &t.punct), "네알겠습니다");
}

/// The control set: goals whose right answers **contain numbers**, so the leak measurement cannot
/// be satisfied by a model that has simply stopped saying numbers at all.
#[test]
fn the_control_set_keeps_its_legitimate_numbers() {
    let (t, d) = tables();
    let words: Vec<String> = t.number_words.clone();
    assert!(d["numbers"].as_array().unwrap().len() >= 12, "the control set shrank");
    assert!(has_number("버튼이 3개 있어요", &words).0);
    assert_eq!(has_number("버튼이 다섯 개 있어요", &words), (true, "word:다섯".to_string()));
    assert_eq!(has_number("복사할게요", &words), (false, String::new()));
}
