//! Re-judge every committed `reoffer` run, and re-derive the p-value that carried the result.
//!
//! Each row holds both turns of every repeat — the effects and the rendered history block — so the
//! judge can be re-run with no model. This is the harness whose baselines are **frozen**: any edit
//! to it moves `h` and makes `--compare` refuse, so a second implementation that agrees with it
//! case for case is worth more here than anywhere else.

use std::path::PathBuf;

use argmind_measure::reoffer::{is_right, key_of, offer_reached, offered_lines, sign_test,
                               slot_lines, verdict_of, ACCEPT, ACTED, AGAIN, ANEW, DRIFTED,
                               IGNORED, LANDED, MEMORY, QUIET, SILENT, STUCK, VOID, WRONG};
use serde_json::{json, Value};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap().to_path_buf()
}

fn runs() -> Vec<(String, Value)> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(root().join("measure")).unwrap().flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with("reoffer-") && name.ends_with(".json") {
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
fn every_committed_verdict_comes_out_the_same() {
    let files = runs();
    assert!(files.len() >= 4, "only {} reoffer results found", files.len());
    let (mut rows, mut reps, mut checked_reached) = (0, 0, 0);
    let mut pre_split_silent = 0;
    let mut seen: std::collections::BTreeMap<&str, usize> = Default::default();
    for (name, d) in &files {
        for row in d["rows"].as_array().unwrap() {
            let kind = row["kind"].as_str().unwrap();
            let want: Option<(&str, &str)> = row["want"].as_array()
                .map(|a| (a[0].as_str().unwrap(), a[1].as_str().unwrap()));
            for (i, rep) in row["repeats"].as_array().unwrap().iter().enumerate() {
                let turns = rep["turns"].as_array().unwrap();
                let first = turns[0]["effects"].as_array().unwrap();
                let last = turns.last().unwrap();
                let second = last["effects"].as_array().unwrap();
                let window = last["window"].as_str().unwrap_or("");
                let got = verdict_of(kind, first, second, window, want);
                //: **`silent` was split out of `wrong` on 2026-09-13**, and only the enlarged
                //: 104-case arms (`hde897e`) were scored under the split. Re-judging an older run
                //: with today's judge therefore differs by exactly that one word — which is the
                //: scoring change itself, recorded rather than smoothed over. Everything else must
                //: still match, and it does.
                //: **Named, not inferred.** This was `name.contains("hde897e")` — the harness
                //: hash of the first post-split Python run — which silently classified every
                //: *later* run as pre-split, including one written by the Rust harness that does
                //: know about the split. The set of pre-split results is closed and frozen: two
                //: Python harnesses, four files, and it can never grow. A list is right where a
                //: heuristic was wrong.
                let pre_split = name.contains("h95da85") || name.contains("h57e605");
                let post_split = !pre_split;
                let want_verdict = row["verdicts"][i].as_str().unwrap();
                let comparable = if post_split || got != SILENT { got } else { WRONG };
                assert_eq!(comparable, want_verdict,
                           "{name} / {} / r{i}", row["case"].as_str().unwrap());
                if got == SILENT && !post_split {
                    pre_split_silent += 1;
                }
                //: **Only where the field exists.** The oldest results predate it, and my first
                //: version defaulted a missing field to `true` — which turned "not recorded" into
                //: a claim and failed against a judge that was right.
                if let Some(want_reached) = rep.get("offer_reached").and_then(|v| v.as_bool()) {
                    assert_eq!(offer_reached(first, window), want_reached,
                               "{name} / {} / r{i} offer_reached", row["case"].as_str().unwrap());
                    checked_reached += 1;
                }
                *seen.entry(got).or_default() += 1;
                reps += 1;
            }
            //: `ok` is the row's own answer to "was this the right thing for this group", and it is
            //: derived here rather than copied.
            if !row["void"].as_bool().unwrap_or(false) {
                assert_eq!(is_right(kind, row["counted"].as_str().unwrap()),
                           row["ok"].as_bool().unwrap(),
                           "{name} / {}", row["case"].as_str().unwrap());
            }
            //: Neither word is right for its group, so the split moved a name and no verdict.
            assert!(!is_right(kind, SILENT) && !is_right(kind, WRONG));
            rows += 1;
        }
    }
    assert!(rows >= 200, "only {rows} rows re-judged");
    assert!(checked_reached >= 100, "offer_reached was checked on only {checked_reached} repeats");
    //: **The judge has to be seen using most of its vocabulary.** A verdict nothing produces is a
    //: verdict nobody has tested.
    for word in [ACTED, IGNORED, QUIET, AGAIN, LANDED, VOID] {
        assert!(seen.contains_key(word), "no recorded run ever produced {word}: {seen:?}");
    }
    println!("  {reps} repeats over {rows} rows: {seen:?}");
    println!("  {pre_split_silent} repeats where today's judge says `silent` and the run said `wrong`");
    assert!(pre_split_silent > 0, "the silent/wrong split is not exercised by any older run");
    //: And the closed set really is closed: five files, two harnesses, none added since.
    assert_eq!(files.iter().filter(|(n, _)| n.contains("h95da85") || n.contains("h57e605")).count(),
               5, "the pre-split set has changed size, which it cannot do");
}

/// The number the whole result rests on, re-derived: **15 fixed, 0 broken, p = 0.00006.**
#[test]
fn the_sign_test_gives_the_p_values_the_commits_quote() {
    //: The enlarged ACCEPT arm.
    assert!((sign_test(15, 0) - 6.103_515_625e-5).abs() < 1e-12);
    //: FARACCEPT, secondary and never pooled with it.
    assert!((sign_test(7, 0) - 0.015625).abs() < 1e-12);
    //: The 44-case set, which could not answer its own question: 4 discordant pairs, all one way.
    assert!((sign_test(4, 0) - 0.125).abs() < 1e-12);
    //: The floor the plan was written against: 6 clears 0.05 and 5 does not.
    assert!(sign_test(6, 0) < 0.05);
    assert!(sign_test(5, 0) > 0.05);
    //: Two-sided, so the direction does not matter and neither does which arm is called first.
    assert_eq!(sign_test(3, 1), sign_test(1, 3));
    //: MEMORY at 8 fixed and 2 broken — a control that moved, and did not clear the floor.
    assert!((sign_test(8, 2) - 0.109375).abs() < 1e-9);
    assert_eq!(sign_test(0, 0), 1.0);
}

/// **A case is `void` when the offer never reached the second prompt** — not a failure, a case that
/// could not be asked. Without this a harness that stopped writing offers would score every
/// acceptance as a failure to accept: a plausible zero produced by the measuring code.
#[test]
fn an_offer_that_never_arrived_is_void_rather_than_failed() {
    let copy = json!([{"kind": "click", "target": "Copy", "role": "button", "text": ""}]);
    let first = copy.as_array().unwrap();
    let with = "  them: 복사 눌러줘\n  offered: to click the button \"Copy\" — nothing was done\n";
    let without = "  them: 복사 눌러줘\n  you: 네\n";

    assert_eq!(verdict_of(ACCEPT, first, first, with, None), ACTED);
    assert_eq!(verdict_of(ACCEPT, first, first, without, None), VOID);
    //: An offer of nothing is void too: there was no offer to accept.
    assert_eq!(verdict_of(ACCEPT, &[], first, with, None), VOID);

    //: **An offered line that names something else is not this offer arriving.** Nothing in 380
    //: recorded rows has one, so without this case `offer_reached` could ignore the name entirely
    //: and every test still passed — it would then call any history block with any offer in it
    //: "the offer reached", which is the whole guard.
    let someone_else = "  offered: to click the button \"Cut\" — nothing was done
";
    assert_eq!(verdict_of(ACCEPT, first, first, someone_else, None), VOID);
    assert!(!offer_reached(first, someone_else));
    assert!(offer_reached(first, with));

    assert_eq!(offered_lines(with).len(), 1);
    assert!(offered_lines(without).is_empty());
    //: `offered.is_empty()` in `verdict_of` is **belt and braces**: with no effects there are no
    //: names, so `offer_reached` is already false. Removing it changes nothing, which is worth
    //: knowing rather than defending — a guard that cannot fail alone is documentation.
    assert!(!offer_reached(&[], with));
}

/// Every verdict reachable, and the groups reading the same effects differently.
#[test]
fn the_same_two_turns_mean_different_things_in_different_groups() {
    let copy = json!([{"kind": "click", "target": "Copy", "role": "button", "text": ""}]);
    let cut = json!([{"kind": "click", "target": "Cut", "role": "button", "text": ""}]);
    let (first, other) = (copy.as_array().unwrap(), cut.as_array().unwrap());
    let w = "  offered: to click the button \"Copy\" — nothing was done\n";

    //: Doing it again: right for an acceptance, **wrong for a question about the past**.
    assert_eq!(verdict_of(ACCEPT, first, first, w, None), ACTED);
    assert_eq!(verdict_of(MEMORY, first, first, w, None), AGAIN);
    //: Doing nothing: the failure for an acceptance, the right answer for the controls.
    assert_eq!(verdict_of(ACCEPT, first, &[], w, None), IGNORED);
    assert_eq!(verdict_of(MEMORY, first, &[], w, None), QUIET);
    //: Doing something else.
    assert_eq!(verdict_of(ACCEPT, first, other, w, None), DRIFTED);

    //: ANEW names a target, so the vocabulary changes again — and `silent` is split from `wrong`
    //: because "said the right thing and did nothing" is a different defect.
    assert_eq!(verdict_of(ANEW, first, other, w, Some(("Cut", "button"))), LANDED);
    assert_eq!(verdict_of(ANEW, first, &[], w, Some(("Cut", "button"))), SILENT);
    assert_eq!(verdict_of(ANEW, first, first, w, Some(("Cut", "button"))), STUCK);
    let elsewhere = json!([{"kind": "key", "target": "", "role": "", "text": "tab"}]);
    assert_eq!(verdict_of(ANEW, first, elsewhere.as_array().unwrap(), w,
                          Some(("Cut", "button"))), WRONG);

    //: The role is half of the key here too: a label of the same name is not the combo box.
    let label = json!([{"kind": "click", "target": "Cut", "role": "text", "text": ""}]);
    assert_eq!(verdict_of(ANEW, first, label.as_array().unwrap(), w,
                          Some(("Cut", "button"))), WRONG);
    assert_eq!(key_of(&copy[0]), ("click".into(), "Copy".into(), "button".into()));
}

/// **The delivery check reads the treatment rather than guessing at it.**
///
/// The one it replaced compared offered wordings, which a `converse.py` change does not touch — so
/// it printed *"no treatment was applied at all"* beside a valid comparison and a real p-value.
#[test]
fn the_slot_between_the_widgets_and_the_turn_is_read_generically() {
    let bare = "On screen now:\n0: button \"Copy\"\n1: button \"Cut\"\n\nThey said: 응\n";
    assert_eq!(slot_lines(bare).unwrap(), Vec::<String>::new());

    let with_slot = "On screen now:\n0: button \"Copy\"\n\nStill open, from earlier:\n  \
                     to click the button \"Copy\" — nothing was done\n\nThey said: 응\n";
    let got = slot_lines(with_slot).unwrap();
    assert_eq!(got.len(), 2, "{got:?}");
    assert!(got[0].contains("Still open"));

    //: An unprompted turn is bounded by a different marker, and both have to work.
    let noticed = "On screen now:\n0: button \"Copy\"\n\nYou noticed: something\n";
    assert_eq!(slot_lines(noticed).unwrap(), Vec::<String>::new());

    //: **It refuses rather than returning nothing** when the prompt's shape has moved: an empty
    //: list would read as "no treatment" and be believed.
    assert!(slot_lines("nothing like a prompt").is_err());
    assert!(slot_lines("On screen now:\n0: button \"Copy\"\n").is_err());
}
