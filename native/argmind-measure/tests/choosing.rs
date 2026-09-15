//! Re-judge every committed choosing run with the Rust scorer, and get the same verdicts.
//!
//! **No model, no fixtures beyond the committed screens, no network.** Each result file carries
//! what was chosen on every repeat, so the scoring can be replayed against runs made months of
//! prompt changes apart — which is a stronger check than one paired run, because it covers models,
//! prompts and configurations that are no longer reachable.

use std::path::PathBuf;

use argmind_measure::choosing::{counted, goals, outcome, score, tally, DECLINED, MISSED, RIGHT, WRONG};
use argmind_measure::screens;
use serde_json::Value;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap().to_path_buf()
}

fn results() -> Vec<(String, Value)> {
    let dir = root().join("measure");
    let mut out = Vec::new();
    for entry in std::fs::read_dir(&dir).unwrap().flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with("choosing-") && name.ends_with(".json") {
            let d: Value = serde_json::from_str(&std::fs::read_to_string(entry.path()).unwrap())
                .unwrap();
            if d.get("rows").is_some() {
                out.push((name, d));
            }
        }
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

#[test]
fn the_scorer_reproduces_every_committed_verdict() {
    let files = results();
    assert!(files.len() >= 10, "only {} choosing results found", files.len());
    let (mut rows_seen, mut declines) = (0, 0);
    for (name, d) in &files {
        for row in d["rows"].as_array().unwrap() {
            let accept: Option<Vec<String>> = row["accept"].as_array()
                .map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect());
            let chose: Vec<Option<String>> = row["chose"].as_array().unwrap().iter()
                .map(|c| c.as_str().map(str::to_string)).collect();
            let got: Vec<&str> = chose.iter()
                .map(|c| outcome(c.as_deref(), accept.as_deref())).collect();
            let want: Vec<&str> = row["results"].as_array().unwrap().iter()
                .map(|r| r.as_str().unwrap()).collect();
            assert_eq!(got, want, "{name} / {}", row["goal"].as_str().unwrap());
            assert_eq!(counted(&got), row["counted"].as_str().unwrap(),
                       "{name} / {} worst-of", row["goal"].as_str().unwrap());
            if accept.is_none() {
                declines += 1;
            }
            rows_seen += 1;
        }
    }
    assert!(rows_seen > 400, "only {rows_seen} rows re-judged");
    //: **The unanswerable half has to be there**, or this is scoring a model that always clicks.
    assert!(declines > 100, "only {declines} unanswerable goals across every recorded run");
}

/// The derived fields, against one committed run — `chose_at` and `right_at` need the screens, so
/// they are checked where the screens still match what the run used.
#[test]
fn the_positions_and_the_tally_match_a_committed_run() {
    let file = root().join("measure")
        .join("choosing-index-qwen3-8b-nothink-qa62805.json");
    let d: Value = serde_json::from_str(&std::fs::read_to_string(&file).unwrap()).unwrap();
    let loaded = screens(&root().join("measure").join("screens"));
    let all = goals(&std::fs::read_to_string(root().join("measure").join("goals.json")).unwrap());

    let mut rows = Vec::new();
    for row in d["rows"].as_array().unwrap() {
        let goal_text = row["goal"].as_str().unwrap();
        let g = all.iter().find(|g| g.goal == goal_text)
            .unwrap_or_else(|| panic!("{goal_text} is in the result and not in goals.json"));
        let widgets = &loaded[&g.screen].widgets;
        let chose: Vec<Option<String>> = row["chose"].as_array().unwrap().iter()
            .map(|c| c.as_str().map(str::to_string)).collect();
        let scored = score(g, widgets, chose);
        assert_eq!(scored.n_listed, row["n_listed"].as_u64().unwrap() as usize, "{goal_text}");
        assert_eq!(scored.agreed, row["agreed"].as_bool().unwrap(), "{goal_text} agreed");
        let want_at: Vec<Option<usize>> = row["chose_at"].as_array().unwrap().iter()
            .map(|v| v.as_u64().map(|n| n as usize)).collect();
        assert_eq!(scored.chose_at, want_at, "{goal_text} chose_at");
        let want_right: Vec<usize> = row["right_at"].as_array().unwrap().iter()
            .map(|v| v.as_u64().unwrap() as usize).collect();
        assert_eq!(scored.right_at, want_right, "{goal_text} right_at");
        rows.push(scored);
    }

    //: The headline this project quotes, re-derived rather than copied.
    let t = tally(&rows);
    assert_eq!((t.right, t.answerable), (21, 28), "21 of 28 answerable right");
    assert_eq!((t.declined, t.unanswerable), (7, 9), "7 of 9 declines");
    assert_eq!(t.right + t.declined, 28, "28 of 37 correct decisions");
}

/// **The scorer has to be able to say every word it knows.** A judge that cannot produce one of its
/// verdicts is a judge nobody has tested for that verdict.
#[test]
fn every_verdict_is_reachable() {
    let ok = vec!["Copy".to_string()];
    assert_eq!(outcome(Some("Copy"), Some(&ok)), RIGHT);
    assert_eq!(outcome(Some("Cut"), Some(&ok)), WRONG);
    assert_eq!(outcome(None, Some(&ok)), MISSED);
    assert_eq!(outcome(None, None), DECLINED);
    //: Clicking anything on a screen where nothing can do it is **wrong, not missed** — the worst
    //: outcome the design exists to avoid.
    assert_eq!(outcome(Some("Cut"), None), WRONG);
    assert_eq!(counted(&[RIGHT, RIGHT, WRONG]), WRONG, "worst of three, not majority");
    assert_eq!(counted(&[RIGHT, DECLINED]), DECLINED);
    assert_eq!(counted(&[MISSED, DECLINED]), MISSED);
    //: **Exact, not case-insensitive.** A widget name is what the interface itself calls the thing,
    //: and the reason a model is here at all is that it matched an English request to the Korean
    //: `닫기` without help — not that names were normalised until they collided. No recorded run
    //: happens to contain a case-only difference, so a mutant relaxing this survived until now.
    assert_eq!(outcome(Some("copy"), Some(&ok)), WRONG);
}
