//! Re-judge every committed repair run with the Rust scorer. No model, no fixtures, no network.

use std::path::PathBuf;

use argmind_measure::repair::{language_worst, repair_verdict, script_of, sensitivity, worst,
                              ENGLISH_OUT, EXTRA, KOREAN_OUT, LANDED, NEITHER, OTHER,
                              REPAIR_WORST, WRONG};
use serde_json::Value;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap().to_path_buf()
}

fn runs() -> Vec<(String, Value)> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(root().join("measure")).unwrap().flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with("repair-") && name.ends_with(".json") {
            let d: Value = serde_json::from_str(&std::fs::read_to_string(entry.path()).unwrap())
                .unwrap();
            //: **`repair-qwen3-8b-off.json` is skipped and it is the reason to be careful here.**
            //: It predates the fingerprints, carries no `set` on its rows, and
            //: `docs/findings/interaction.md` records that its headline was *never true of any
            //: tree in git* — two prompt experiments overwrote it in place. A scorer checked
            //: against it would be checked against nothing.
            let shaped = d["rows"].as_array()
                .is_some_and(|rows| rows.iter().all(|r| r.get("set").is_some()));
            if shaped {
                out.push((name, d));
            }
        }
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

#[test]
fn the_repair_verdicts_come_out_the_same() {
    let files = runs();
    assert!(!files.is_empty(), "no repair results found");
    let mut n = 0;
    for (name, d) in &files {
        for row in d["rows"].as_array().unwrap() {
            if row["set"] != "repair" {
                continue;
            }
            let want: Option<(String, String)> = row["want"].as_array().map(|a| (
                a[0].as_str().unwrap().to_string(), a[1].as_str().unwrap().to_string()));
            let mut each = Vec::new();
            for rep in row["repeats"].as_array().unwrap() {
                let clicks: Vec<(String, String)> = rep["clicks"].as_array().unwrap().iter()
                    .map(|c| (c[0].as_str().unwrap().to_string(),
                              c[1].as_str().unwrap().to_string())).collect();
                let got = repair_verdict(&clicks, want.as_ref());
                assert_eq!(got, rep["verdict"].as_str().unwrap(),
                           "{name} / {}", row["case"].as_str().unwrap());
                each.push(got);
                n += 1;
            }
            assert_eq!(worst(&each, &REPAIR_WORST), row["counted"].as_str().unwrap(),
                       "{name} / {} worst-of", row["case"].as_str().unwrap());
        }
    }
    assert!(n >= 24, "only {n} repair repeats re-judged");
}

/// The language rows carry what was said, so the script is re-derived from the words rather than
/// from a stored label — which is what makes this a check of the scorer and not of the file.
#[test]
fn the_language_verdicts_come_out_the_same() {
    let mut n = 0;
    let (mut korean, mut english, mut mixed) = (0, 0, 0);
    for (name, d) in &runs() {
        for row in d["rows"].as_array().unwrap() {
            let asked = row["set"].as_str().unwrap();
            if asked == "repair" {
                continue;
            }
            let mut each = Vec::new();
            for rep in row["repeats"].as_array().unwrap() {
                let say = rep["say"].as_str().unwrap_or("");
                let (got, is_mixed) = script_of(say);
                assert_eq!(got, rep["got"].as_str().unwrap(),
                           "{name} / {:?}", row["said"].as_str().unwrap());
                each.push(got);
                if is_mixed {
                    mixed += 1;
                }
                match got {
                    KOREAN_OUT => korean += 1,
                    ENGLISH_OUT => english += 1,
                    _ => {}
                }
                n += 1;
            }
            assert_eq!(worst(&each, &language_worst(asked)), row["counted"].as_str().unwrap(),
                       "{name} / {:?} worst-of", row["said"].as_str().unwrap());
        }
    }
    assert!(n >= 100, "only {n} language repeats re-judged");
    //: **Both answers have to occur**, or the scorer is being checked against one of its words.
    assert!(korean > 20 && english > 20, "korean {korean}, english {english}");
    assert!(mixed > 0, "no reply mixed scripts; the `mixed` flag is untested");
}

/// **Every verdict reachable, and the two orderings actually different.**
#[test]
fn the_scorer_can_say_all_of_its_words() {
    let red = ("Red".to_string(), "list item".to_string());
    let other = ("Orange".to_string(), "list item".to_string());
    assert_eq!(repair_verdict(&[red.clone()], Some(&red)), LANDED);
    assert_eq!(repair_verdict(&[red.clone(), other.clone()], Some(&red)), EXTRA);
    assert_eq!(repair_verdict(&[other.clone()], Some(&red)), WRONG);
    assert_eq!(repair_verdict(&[], Some(&red)), WRONG);
    //: `want: None` means **stopping is the right answer** — the case an always-clicking agent
    //: fails, and half of why this harness exists.
    assert_eq!(repair_verdict(&[], None), LANDED);
    assert_eq!(repair_verdict(&[red.clone()], None), WRONG);
    //: The role is half of the key: the `Font :` label in front of the combo box of that name.
    let label = ("Font :".to_string(), "text".to_string());
    let combo = ("Font :".to_string(), "combo box".to_string());
    assert_eq!(repair_verdict(&[label], Some(&combo)), WRONG);

    assert_eq!(script_of("복사할게요"), (KOREAN_OUT, false));
    assert_eq!(script_of("I'll click Copy"), (ENGLISH_OUT, false));
    //: Hangul decides first: quoting an English widget name inside a Korean sentence is Korean,
    //: and `mixed` is recorded beside the verdict rather than folded into it.
    assert_eq!(script_of("Copy 버튼을 누를게요"), (KOREAN_OUT, true));
    assert_eq!(script_of("こんにちは"), (OTHER, false));
    assert_eq!(script_of("3, 122"), (NEITHER, false));
    assert_eq!(script_of(""), (NEITHER, false));

    //: **The same verdict is the success or the failure depending on what was asked in.**
    assert_eq!(worst(&[KOREAN_OUT, ENGLISH_OUT], &language_worst("korean")), ENGLISH_OUT);
    assert_eq!(worst(&[KOREAN_OUT, ENGLISH_OUT], &language_worst("english")), KOREAN_OUT);
}

/// The floor a repair could reach, before it is attempted — printed for every group because a
/// control that cannot move is as much a design fact as a hypothesis that cannot.
#[test]
fn the_sensitivity_floor_is_what_the_docstrings_quote() {
    assert_eq!(sensitivity(0), 1.0);
    assert!((sensitivity(4) - 0.125).abs() < 1e-12, "4 failures give 0.125 at best");
    assert!((sensitivity(5) - 0.0625).abs() < 1e-12, "5 give 0.0625 — below the floor");
    assert!((sensitivity(6) - 0.03125).abs() < 1e-12, "6 is the first that clears 0.05");
    assert!(sensitivity(1) <= 1.0, "a probability cannot exceed one");
}
