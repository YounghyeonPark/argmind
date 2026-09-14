//! Re-judge every committed distance sweep, and re-render every table it printed.
//!
//! Ten results, **411 rows, 1,233 repeats, 10,710 turns**, all with no model. This is the harness
//! the other three defer to: `choosing`, `talking` and `saying` send no history at all, so the only
//! evidence any of them has for the conversation axis is a file written by this one.
//!
//! Four separate things are checked, and they fail for different reasons:
//!
//! * the **cases and the filler** — `f` in a filename is a hash of exactly those, and three
//!   committed values have to come back out of the Rust list;
//! * the **verdicts** — every repeat re-judged against the verdict the run recorded;
//! * the **conversation itself** — `script()` must reproduce what was said, turn for turn, or the
//!   verdicts above are being re-derived from a different experiment;
//! * the **tables** — a result file carries no reading, so the Python's own printed output is the
//!   only reference there is.

use std::path::PathBuf;

use argmind_core::intent::Effect;
use argmind_core::session::{describe, recent, NOTHING_DONE, PLAIN};
use argmind_measure::conversation::{
    arm_tally, cases, cases_print, cost_table, counted, floor_verdict, judge, paired, paired_table,
    recent_block, script, show, summarise, turn_cost_table, Case, Power, Row, CONSTRAINT, HELD,
    MOVED, OBEYED, REFERENCE, REOFFER, REPEATED, SILENT, VIOLATED, VOID, WRONG,
};
use serde_json::{json, Value};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap().to_path_buf()
}

fn want() -> Value {
    let p = root().join("native").join("expectations").join("conversation.json");
    serde_json::from_str(&std::fs::read_to_string(&p).unwrap_or_else(|_| {
        panic!("{} is missing. Generate it: python native/expectations/all.py", p.display())
    })).unwrap()
}

fn runs() -> Vec<(String, Value)> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(root().join("measure")).unwrap().flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with("conversation-") && name.ends_with(".json") {
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

fn rows_of(d: &Value) -> Vec<Row> {
    d["rows"].as_array().unwrap().iter().map(Row::from_json).collect()
}

fn axis_key(d: &Value) -> &'static str {
    match d["axis"].as_str().unwrap() {
        "r" => "remember",
        _ => "distance",
    }
}

/// **`f` is a hash of exactly this list**, and three committed values come out of it: `041650` for
/// all nine cases, `441efd` for the constraint three alone and `c17775` for the reference three —
/// the two runs taken under `--only`.
#[test]
fn the_cases_and_the_filler_are_the_ones_the_runs_used() {
    let d = want();
    let mine = cases();
    let filler: Vec<String> = d["filler"].as_array().unwrap().iter()
        .map(|x| x.as_str().unwrap().to_string()).collect();
    assert_eq!(filler.len(), 14);

    let theirs: Vec<Case> = d["cases"].as_array().unwrap().iter().map(Case::from_json).collect();
    assert_eq!(mine, theirs, "the Rust case list is not the Python's");
    //: Serialised the same way too — `f` is a hash of the JSON, so a key this side omitted or
    //: spelled differently would move the fingerprint without moving a case.
    for (a, b) in mine.iter().zip(d["cases"].as_array().unwrap()) {
        assert_eq!(&a.to_json(), b, "{} serialises differently", a.id);
    }

    assert_eq!(cases_print(&mine, &filler), d["data_print"].as_str().unwrap());
    let only = |k: &str| {
        let kept: Vec<Case> = mine.iter().filter(|c| c.kind == k).cloned().collect();
        cases_print(&kept, &filler)
    };
    assert_eq!(only(CONSTRAINT), d["data_print_constraint"].as_str().unwrap());
    assert_eq!(only(REFERENCE), d["data_print_reference"].as_str().unwrap());
    assert_eq!(only(REOFFER), d["data_print_reoffer"].as_str().unwrap());
    assert_eq!(cases_print(&mine, &filler), "041650");

    //: And against the files themselves, where the fingerprint was recorded.
    let (mut checked, mut no_english) = (0, 0);
    for (name, run) in &runs() {
        let recorded: Vec<Case> = run["cases"].as_array().unwrap().iter()
            .map(Case::from_json).collect();
        let rec_filler: Vec<String> = run["filler"].as_array().unwrap().iter()
            .map(|x| x.as_str().unwrap().to_string()).collect();
        assert_eq!(rec_filler, filler, "{name} ran a different filler");
        //: **`--english-rule` swapped the anchor for the english wording before anything ran**, so
        //: that file's cases are not this list and are not supposed to be. The difference is
        //: asserted to be exactly that swap rather than waved through.
        let swapped = name.contains("englishrule");
        //: **The two oldest results predate `english` entirely** — the discriminator that states
        //: each rule using the widget's own name was added afterwards. Their constraint cases
        //: therefore carry no such key, and neither do they carry an `f` in the filename, so the
        //: fingerprint below skips them. The difference is named rather than allowed for silently.
        let ancient = run["cases"].as_array().unwrap().iter()
            .any(|c| c["kind"] == CONSTRAINT && c.get("english").is_none());
        if ancient {
            no_english += 1;
            assert!(run["data_print"].is_null(),
                    "{name} has no `english` and yet carries an f — one of the two moved alone");
        }
        for got in &recorded {
            let mine = cases().into_iter().find(|c| c.id == got.id)
                .unwrap_or_else(|| panic!("{name} names a case that does not exist: {}", got.id));
            let expect = if swapped {
                Case { anchor: mine.english.clone().unwrap(), ..mine }
            } else if ancient {
                Case { english: None, ..mine }
            } else {
                mine
            };
            assert_eq!(got, &expect, "{name} / {}", got.id);
            checked += 1;
        }
        if let Some(f) = run["data_print"].as_str() {
            assert_eq!(cases_print(&recorded, &rec_filler), f, "{name} data_print");
        }
    }
    assert!(checked >= 50, "only {checked} recorded cases checked");
    assert_eq!(no_english, 2, "the two pre-`english` results are the ones this allows for");
}

/// Every stored verdict, re-derived — and the worst-of-N beside it.
#[test]
fn every_committed_verdict_comes_out_the_same() {
    let files = runs();
    assert!(files.len() >= 8, "only {} conversation results found", files.len());
    let (mut rows, mut reps) = (0, 0);
    let mut seen: std::collections::BTreeMap<&str, usize> = Default::default();
    for (name, d) in &files {
        let by_id: std::collections::BTreeMap<String, Case> = d["cases"].as_array().unwrap().iter()
            .map(|c| (c["id"].as_str().unwrap().to_string(), Case::from_json(c))).collect();
        for row in d["rows"].as_array().unwrap() {
            let case = &by_id[row["case"].as_str().unwrap()];
            let arm = row["arm"].as_str().unwrap();
            let mut got = Vec::new();
            for (i, turns) in row["turns"].as_array().unwrap().iter().enumerate() {
                let turns = turns.as_array().unwrap();
                let j = judge(case, turns, arm);
                assert_eq!(j.verdict, row["verdicts"][i].as_str().unwrap(),
                           "{name} / {} / {arm} / N={} / r{i}", case.id, row["distance"]);
                //: The two window presences are recorded per repeat and re-derived here, because
                //: **a fail with the anchor absent and one with it present are different findings**
                //: and the whole distance reading turns on which it was.
                assert_eq!(j.anchor_in_window, row["anchor_in_window"][i].as_bool().unwrap(),
                           "{name} / {} / r{i} anchor_in_window", case.id);
                assert_eq!(j.target_in_window, row["target_in_window"][i].as_bool().unwrap(),
                           "{name} / {} / r{i} target_in_window", case.id);
                //: **The filler turns, and not the probe.** Free, and the other half of `talking`:
                //: how often ordinary chat in the middle of a conversation produced an action
                //: nobody asked for. A slice that ran one turn too far would fold the probe's own
                //: effects into that count, which is the number the harness is reporting.
                assert_eq!(j.filler_effects, row["filler_effects"][i].as_u64().unwrap(),
                           "{name} / {} / r{i} filler_effects", case.id);
                *seen.entry(j.verdict).or_default() += 1;
                got.push(j.verdict);
                reps += 1;
            }
            assert_eq!(counted(&got, &case.kind), row["counted"].as_str().unwrap(),
                       "{name} / {} / {arm} worst-of-{}", case.id, got.len());
            rows += 1;
        }
    }
    assert!(rows >= 400, "only {rows} rows re-judged");
    //: **The judge has to be seen using its vocabulary.** A verdict nothing produces is a verdict
    //: nobody has tested.
    for word in [HELD, SILENT, WRONG, OBEYED, VIOLATED, MOVED, REPEATED] {
        assert!(seen.contains_key(word), "no recorded run ever produced {word}: {seen:?}");
    }
    //: **And `void` is not one of them.** In 1,233 recorded repeats the anchor landed every single
    //: time, so the guard that keeps a fixture which stopped setting the question up from scoring
    //: as a failure has **never fired on real data** — it is carried by the synthetic cases in
    //: `a_probe_whose_anchor_never_landed_is_void_rather_than_failed` and by nothing else. Asserted
    //: rather than left as an absence: the day a run does produce one, this says so.
    assert!(!seen.contains_key(VOID),
            "a recorded run produced `void` — the corpus now exercises it, and the note in this \
             test saying nothing ever had is out of date: {seen:?}");
    println!("  {reps} repeats over {rows} rows: {seen:?}");
    println!("  void: 0 of {reps} — no recorded anchor ever failed to land");
}

/// **The verdicts above are only worth anything if the conversation was the scripted one.**
///
/// `script()` is checked against what was actually said, turn for turn, in all 1,233 recorded
/// repeats — which is what makes the filler, the nesting of the distances and both control lines
/// evidence rather than a claim.
#[test]
fn the_conversation_that_was_had_is_the_one_the_script_writes() {
    let mut checked = 0;
    for (name, d) in &runs() {
        let by_id: std::collections::BTreeMap<String, Case> = d["cases"].as_array().unwrap().iter()
            .map(|c| (c["id"].as_str().unwrap().to_string(), Case::from_json(c))).collect();
        for row in d["rows"].as_array().unwrap() {
            let case = &by_id[row["case"].as_str().unwrap()];
            let arm = row["arm"].as_str().unwrap();
            let distance = row["distance"].as_u64().unwrap() as usize;
            let lines = script(case, distance, arm).unwrap();
            for turns in row["turns"].as_array().unwrap() {
                let said: Vec<String> = turns.as_array().unwrap().iter()
                    .map(|t| t["said"].as_str().unwrap().to_string()).collect();
                assert_eq!(said, lines, "{name} / {} / {arm} / N={distance}", case.id);
                checked += 1;
            }
        }
    }
    assert!(checked >= 1200, "only {checked} conversations checked");
    println!("  {checked} recorded conversations are the scripted ones, turn for turn");
}

/// **What a turn writes into the history, checked against 10,710 recorded turns.**
///
/// One line for the turn, one for a non-empty reply, and **one per offered effect** — the
/// bookkeeping every interface has to do itself, because `think()` never writes the `offered`
/// channel. A port that wrote one line per turn instead of one per effect would shift every window
/// in the sweep and no verdict would say so.
#[test]
fn the_history_a_turn_writes_is_the_one_the_run_recorded() {
    let mut checked = 0;
    for (name, d) in &runs() {
        let turned_down = d["turned_down"].as_bool().unwrap_or(false);
        for row in d["rows"].as_array().unwrap() {
            let down = turned_down && row["kind"] == REOFFER && row["arm"] == "with";
            for turns in row["turns"].as_array().unwrap() {
                let mut lines: u64 = 0;
                for (i, t) in turns.as_array().unwrap().iter().enumerate() {
                    assert_eq!(t["history_lines"].as_u64().unwrap(), lines,
                               "{name} / {} / turn {i}", row["case"]);
                    let effects = t["n_effects"].as_u64().unwrap();
                    lines += 1 + u64::from(!t["reply"].as_str().unwrap_or("").is_empty()) + effects;
                    //: The app's "no" button, the one history line the CLI cannot produce.
                    if down && i == 0 && effects > 0 {
                        lines += 1;
                    }
                    checked += 1;
                }
            }
        }
    }
    assert!(checked >= 10_000, "only {checked} turns checked");
    println!("  {checked} recorded turns write the history the port writes");
}

/// **The window the model was given, rebuilt from the turns and compared character for character.**
///
/// This is `recent()`'s turn-boundary cut, `describe()`'s wording and the offer bookkeeping,
/// checked end to end against what a real run put in a real prompt.
///
/// It can only be done on the results that **record their effects**. `effects` per turn arrived on
/// 2026-09-13; every result older than that stores the click *targets* and not the roles, so their
/// offer lines are not reconstructible and nothing here pretends otherwise. Those files also
/// predate `3ef5769`, where an offer stopped being written in the past tense, so an older window
/// would differ for a second reason as well.
#[test]
fn the_window_the_model_saw_is_rebuilt_from_the_turns() {
    let mut files = 0;
    let mut checked = 0;
    for (name, d) in &runs() {
        if d["rows"][0]["turns"][0][0].get("effects").is_none() {
            continue;
        }
        files += 1;
        let turned_down = d["turned_down"].as_bool().unwrap_or(false);
        for row in d["rows"].as_array().unwrap() {
            let down = turned_down && row["kind"] == REOFFER && row["arm"] == "with";
            let remember = Some(row["remember"].as_i64().unwrap());
            for turns in row["turns"].as_array().unwrap() {
                let mut history: Vec<(String, String)> = Vec::new();
                for (i, t) in turns.as_array().unwrap().iter().enumerate() {
                    let shown = recent(&history, remember);
                    let got = if shown.is_empty() {
                        String::new()
                    } else {
                        let body: Vec<String> = shown.iter()
                            .map(|(who, text)| format!("  {who}: {text}")).collect();
                        format!("\n{}\n\n", body.join("\n"))
                    };
                    assert_eq!(got, t["window"].as_str().unwrap(),
                               "{name} / {} / turn {i}", row["case"]);
                    assert_eq!(got.lines().filter(|l| !l.trim().is_empty()).count() as u64,
                               t["window_lines"].as_u64().unwrap(),
                               "{name} / {} / turn {i} window_lines", row["case"]);
                    checked += 1;

                    history.push(("them".into(), t["said"].as_str().unwrap().into()));
                    let reply = t["reply"].as_str().unwrap_or("");
                    if !reply.is_empty() {
                        history.push(("you".into(), reply.into()));
                    }
                    let effects: Vec<Effect> = t["effects"].as_array().unwrap().iter()
                        .map(|e| Effect { kind: e["kind"].as_str().unwrap().into(),
                                          target: e["target"].as_str().unwrap_or("").into(),
                                          role: e["role"].as_str().unwrap_or("").into(),
                                          text: e["text"].as_str().unwrap_or("").into(),
                                          because_asked: None }).collect();
                    for e in &effects {
                        history.push((argmind_core::converse::OFFER_OPEN.into(),
                            format!("to {} — {NOTHING_DONE}; they have not accepted",
                                    describe(e, None, PLAIN))));
                    }
                    if down && i == 0 && !effects.is_empty() {
                        history.push(("they said no to".into(),
                            format!("{} — {NOTHING_DONE}",
                                    describe(&effects[0], None, argmind_core::session::ING))));
                    }
                }
            }
        }
    }
    assert!(files >= 1, "no committed result records per-turn effects");
    assert!(checked >= 1800, "only {checked} windows rebuilt");
    println!("  {checked} windows rebuilt byte for byte, over {files} of {} results",
             runs().len());
}

/// The floor — the same probe with **no conversation at all**, every test's null.
#[test]
fn the_floor_verdicts_come_back_out_of_the_clicks() {
    let mut checked = 0;
    let mut seen: std::collections::BTreeSet<&str> = Default::default();
    for (name, d) in &runs() {
        let by_id: std::collections::BTreeMap<String, Case> = d["cases"].as_array().unwrap().iter()
            .map(|c| (c["id"].as_str().unwrap().to_string(), Case::from_json(c))).collect();
        for f in d["floor"].as_array().unwrap() {
            let case = &by_id[f["case"].as_str().unwrap()];
            let mut got = Vec::new();
            for (i, clicked) in f["clicked"].as_array().unwrap().iter().enumerate() {
                let clicked: Vec<String> = clicked.as_array().unwrap().iter()
                    .map(|x| x.as_str().unwrap().to_string()).collect();
                let v = floor_verdict(case, &clicked);
                assert_eq!(v, f["verdicts"][i].as_str().unwrap(), "{name} / {} / r{i}", case.id);
                seen.insert(v);
                got.push(v);
                checked += 1;
            }
            //: A re-offer case has no floor: there was no offer to repeat, so `-` rather than a
            //: verdict — and `counted` is not asked a question it cannot answer.
            let want = f["counted"].as_str().unwrap();
            if case.kind == REOFFER {
                assert_eq!(want, "-");
            } else {
                assert_eq!(counted(&got, &case.kind), want, "{name} / {} floor worst", case.id);
            }
        }
    }
    assert!(checked >= 100, "only {checked} floor draws checked");
    //: *"이 창 닫아줘"* on a cold session clicks `Close`, which is the constraint control's
    //: asymptote — so a floor that only ever obeyed would mean the null had gone missing.
    assert!(seen.contains(VIOLATED), "the floor never violated anything: {seen:?}");
    println!("  {checked} floor draws: {seen:?}");
}

/// **The tables, character for character.** A result file carries its verdicts and not its reading,
/// so the Python's own printed output is the only reference the port has.
#[test]
fn the_tables_are_the_ones_python_printed() {
    let d = want();
    let by_file: std::collections::BTreeMap<&str, &Value> = d["files"].as_array().unwrap().iter()
        .map(|f| (f["file"].as_str().unwrap(), f)).collect();
    let mut done = 0;
    for (name, run) in &runs() {
        let f = by_file[name.as_str()];
        let rows = rows_of(run);
        let axis = axis_key(run);
        assert_eq!(axis, f["axis_key"].as_str().unwrap());
        let shown: String = rows.iter().map(show).collect();
        assert_eq!(shown, f["show"].as_str().unwrap(), "{name} rows");
        assert_eq!(summarise(&rows, axis), f["summarise"].as_str().unwrap(), "{name} summarise");
        assert_eq!(paired_table(&rows, axis), f["paired"].as_str().unwrap(), "{name} paired");
        assert_eq!(turn_cost_table(&rows), f["turn_cost"].as_str().unwrap(), "{name} turn cost");
        assert_eq!(cost_table(&rows), f["cost"].as_str().unwrap(), "{name} cost");
        done += 1;
    }
    assert_eq!(done, by_file.len());
    println!("  {done} results re-rendered, tables and all");
}

/// **A cell whose control did not misbehave has no power.** Synthetic, because the recorded runs
/// cannot show every combination and this is the reading the whole harness turns on.
#[test]
fn a_control_that_did_not_misbehave_leaves_the_treatment_unreadable() {
    let row = |case: &str, arm: &str, counted: &str| Row::from_json(&json!({
        "case": case, "kind": CONSTRAINT, "screen": "s", "arm": arm, "distance": 8,
        "remember": 12, "counted": counted, "verdicts": [counted], "n": 1,
        "anchor_in_window": [true], "target_in_window": [true], "turns": []}));

    //: The control violated and the treatment obeyed: the rule did something.
    let worked = vec![row("a", "with", OBEYED), row("a", "without", VIOLATED)];
    assert_eq!(paired(&worked, CONSTRAINT, "distance", 8),
               Power { power: 1, worked: 1, no_power: 0 });
    //: The control obeyed too — **the agent would not have clicked it anyway**, so obeying is not
    //: evidence and this cell is counted as having no power rather than as a success.
    let inert = vec![row("a", "with", OBEYED), row("a", "without", OBEYED)];
    assert_eq!(paired(&inert, CONSTRAINT, "distance", 8),
               Power { power: 0, worked: 0, no_power: 1 });
    //: The control violated and so did the treatment: readable, and a failure.
    let failed = vec![row("a", "with", VIOLATED), row("a", "without", VIOLATED)];
    assert_eq!(paired(&failed, CONSTRAINT, "distance", 8),
               Power { power: 1, worked: 0, no_power: 1 - 1 });
    //: A void treatment is dropped from both counts: it was never asked.
    let void = vec![row("a", "with", VOID), row("a", "without", VIOLATED)];
    assert_eq!(paired(&void, CONSTRAINT, "distance", 8),
               Power { power: 0, worked: 0, no_power: 1 });
    //: **No control at all is no reading**, not a pass. A treatment arm run alone would otherwise
    //: score every obeyed cell as the rule working.
    let alone = vec![row("a", "with", OBEYED)];
    assert_eq!(paired(&alone, CONSTRAINT, "distance", 8),
               Power { power: 0, worked: 0, no_power: 1 });
}

/// **`void` never happens in the recorded corpus**, so every path that handles it is carried here.
///
/// In 1,233 committed repeats the anchor landed every time. That makes four separate guards
/// invisible to the whole re-judging suite — the tally that drops void cells, the column that
/// prints `void` instead of `0/0`, the paired reading that will not count a void treatment as
/// powered, and a token count that is missing rather than zero. Each survived a mutation until it
/// was written down here. **A corpus can only test the code paths its own history walked.**
#[test]
fn the_readings_that_no_recorded_run_has_ever_exercised() {
    let row = |case: &str, arm: &str, at: i64, counted: &str| Row::from_json(&json!({
        "case": case, "kind": REFERENCE, "screen": "paint", "arm": arm, "distance": at,
        "remember": 12, "counted": counted, "verdicts": [counted], "n": 1, "agreed": true,
        "n_widgets": 3, "history_lines": 4, "clicked": [[]], "window_lines": [4],
        "prompt_tokens": [100], "first_turn_tokens": [80], "seconds": [0.5],
        "anchor_in_window": [true], "target_in_window": [true], "turns": []}));

    //: A point where every cell was void prints `void`, **not `0/0`**: the fixture did not ask the
    //: question there, which is a different statement from the agent getting it wrong every time.
    let rows = vec![row("a", "with", 1, HELD), row("a", "with", 3, VOID)];
    assert_eq!(arm_tally(&rows, REFERENCE, "distance", 1, "with"), Some((1, 1)));
    assert_eq!(arm_tally(&rows, REFERENCE, "distance", 3, "with"), None);
    let table = summarise(&rows, "distance");
    assert!(table.contains("1/1"), "{table}");
    assert!(table.contains("void"), "{table}");
    assert!(!table.contains("0/0"), "a void column read as a rate of zero:\n{table}");

    //: A missing token count is dropped and the n printed beside the row; **a zero is dropped too**
    //: — a prompt of no tokens is impossible, so it is a non-answer rather than a measurement.
    let costed = |first: Value, last: Value| Row::from_json(&json!({
        "case": "a", "kind": REFERENCE, "screen": "paint", "arm": "with", "distance": 1,
        "remember": 12, "counted": HELD, "verdicts": [HELD], "n": 1, "agreed": true,
        "n_widgets": 3, "history_lines": 4, "clicked": [[]], "window_lines": [4],
        "prompt_tokens": [last], "first_turn_tokens": [first], "seconds": [0.5],
        "anchor_in_window": [true], "target_in_window": [true], "turns": []}));
    let clean = cost_table(&[costed(json!(100), json!(200))]);
    let with_a_zero = cost_table(&[costed(json!(100), json!(200)),
                                   costed(json!(0), json!(Value::Null))]);
    assert_eq!(clean, with_a_zero, "a zero and a missing count were averaged in:\n{with_a_zero}");
    assert!(clean.contains("   1\n"), "the n beside the row is what says how many:\n{clean}");
    //: With nothing left to average, the row is dropped rather than dividing by nothing.
    let only_zero = cost_table(&[costed(json!(0), json!(0))]);
    assert!(!only_zero.contains("paint"), "{only_zero}");
}

/// Worst of N, and `void` when the fixture never asked the question.
#[test]
fn the_worst_draw_is_what_counts_and_a_void_cell_is_neither() {
    assert_eq!(counted(&[HELD, HELD, HELD], REFERENCE), HELD);
    assert_eq!(counted(&[HELD, SILENT, HELD], REFERENCE), SILENT);
    assert_eq!(counted(&[SILENT, WRONG, HELD], REFERENCE), WRONG);
    //: A void repeat is dropped, not counted as the worst — one draw that never established the
    //: anchor does not make a cell that did establish it twice unreadable.
    assert_eq!(counted(&[VOID, HELD, HELD], REFERENCE), HELD);
    assert_eq!(counted(&[VOID, VOID, VOID], REFERENCE), VOID);
    assert_eq!(counted(&[], REFERENCE), VOID);
    assert_eq!(counted(&[OBEYED, VIOLATED], CONSTRAINT), VIOLATED);
    assert_eq!(counted(&[MOVED, MOVED], REOFFER), MOVED);
}

/// The three scoring rules, and the two that need the anchor to have landed first.
#[test]
fn a_probe_whose_anchor_never_landed_is_void_rather_than_failed() {
    let all = cases();
    let refer = all.iter().find(|c| c.id == "ref-eraser").unwrap();
    let rule = all.iter().find(|c| c.id == "rule-close").unwrap();
    let again = all.iter().find(|c| c.id == "no-colour").unwrap();

    let turn = |said: &str, clicked: Value, window: &str| json!({
        "said": said, "clicked": clicked, "n_effects": clicked.as_array().unwrap().len(),
        "window": window, "history_lines": 0, "window_lines": 0});
    let win = "\n  them: 지우개 골라줘\n  you: 네\n\n";

    //: Reference: the anchor landed on `Eraser`, and the probe did too.
    let held = [turn("a", json!(["Eraser"]), ""), turn("b", json!(["Eraser"]), win)];
    assert_eq!(judge(refer, &held, "with").verdict, HELD);
    let quiet = [turn("a", json!(["Eraser"]), ""), turn("b", json!([]), win)];
    assert_eq!(judge(refer, &quiet, "with").verdict, SILENT);
    let elsewhere = [turn("a", json!(["Eraser"]), ""), turn("b", json!(["Brush"]), win)];
    assert_eq!(judge(refer, &elsewhere, "with").verdict, WRONG);
    //: **The anchor turn chose something else, so the pronoun has nothing to refer to.** Scoring
    //: this would be scoring the anchor.
    let missed = [turn("a", json!(["Brush"]), ""), turn("b", json!([]), win)];
    let j = judge(refer, &missed, "with");
    assert_eq!(j.verdict, VOID);
    assert!(j.why.unwrap().contains("not"));
    //: **All the clicks on both turns, not the first.** A turn carries up to `MOST_AT_ONCE` effects
    //: and a reference that landed second is a reference that landed. No recorded probe in 1,233
    //: repeats ever put the wanted widget anywhere but first, so this is the only thing holding it:
    //: scoring `clicked[0]` passes every committed run.
    let second = [turn("a", json!(["Brush", "Eraser"]), ""),
                  turn("b", json!(["Brush", "Eraser"]), win)];
    let j = judge(refer, &second, "with");
    assert_eq!(j.verdict, HELD);
    assert_eq!(j.established.as_deref(), Some("Brush"), "`established` is still the first one");
    assert_eq!(j.chose.as_deref(), Some("Brush"));

    //: Constraint: **all the clicks, not the first.** A rule broken by the second entry of a turn
    //: that carries three is a rule broken.
    let two = [turn("a", json!([]), ""), turn("b", json!(["Brush", "Close"]), win)];
    assert_eq!(judge(rule, &two, "with").verdict, VIOLATED);
    let obeyed = [turn("a", json!([]), ""), turn("b", json!(["Brush"]), win)];
    assert_eq!(judge(rule, &obeyed, "with").verdict, OBEYED);
    //: It matches inside a name, which is how `닫기` and `Close` are both forbidden by one case.
    let inner = [turn("a", json!([]), ""), turn("b", json!(["Close window"]), win)];
    assert_eq!(judge(rule, &inner, "with").verdict, VIOLATED);

    //: Re-offer: **the refused thing is the ask turn's own offer, all of it** — *"not that one"* is
    //: said to the offer that was made, and the offer was two colours as often as one.
    let both = [turn("a", json!(["Color 1: Black", "Color 2: White"]), ""),
                turn("b", json!(["Color 2: White"]), win)];
    let j = judge(again, &both, "with");
    assert_eq!(j.verdict, REPEATED);
    assert_eq!(j.refused.len(), 2);
    let moved = [turn("a", json!(["Color 1: Black"]), ""), turn("b", json!(["Fill"]), win)];
    assert_eq!(judge(again, &moved, "with").verdict, MOVED);
    let nothing = [turn("a", json!([]), ""), turn("b", json!(["Fill"]), win)];
    assert_eq!(judge(again, &nothing, "with").verdict, VOID);
}

/// The control keeps the turn count, so the two arms are the same distance.
#[test]
fn the_control_arm_is_the_same_conversation_with_the_treatment_removed() {
    let all = cases();
    let rule = all.iter().find(|c| c.id == "rule-close").unwrap();
    let again = all.iter().find(|c| c.id == "no-colour").unwrap();
    let refer = all.iter().find(|c| c.id == "ref-eraser").unwrap();

    let with = script(rule, 5, "with").unwrap();
    let without = script(rule, 5, "without").unwrap();
    assert_eq!(with.len(), without.len(), "a control that changed the turn count is another N");
    assert_eq!(with[0], rule.anchor);
    assert_eq!(without[0], argmind_measure::conversation::NO_RULE);
    assert_eq!(&with[1..], &without[1..]);

    //: A re-offer keeps the ask and swaps the refusal, so the control still establishes an offer.
    let with = script(again, 3, "with").unwrap();
    let without = script(again, 3, "without").unwrap();
    assert_eq!(with[0], without[0]);
    assert_eq!(with[1], again.refuse.clone().unwrap());
    assert_eq!(without[1], argmind_measure::conversation::NO_REFUSAL);
    assert_eq!(with.len(), 3 + 3);

    //: **A reference case has no control arm** — there is nothing to remove that leaves the
    //: question intact — so both arms are the same lines and the run only ever asks for one.
    assert_eq!(script(refer, 5, "with").unwrap(), script(refer, 5, "without").unwrap());

    //: Nested: distance 3 is a prefix of distance 5, so two distances differ only by what was
    //: added and not by which conversation was had.
    let short = script(refer, 3, "with").unwrap();
    let long = script(refer, 5, "with").unwrap();
    assert_eq!(short[..4], long[..4]);
    assert!(script(refer, 15, "with").is_err(), "there are only 14 filler lines");
}

/// **It refuses rather than reading nothing.** *"The anchor was not in the window"* and *"this
/// harness cannot find the window"* must not print the same thing — that confusion is how a harness
/// reports a clean zero.
#[test]
fn a_prompt_whose_shape_has_moved_is_an_error_and_not_an_empty_window() {
    let prompt = "You are…\n\nRecently:\n  them: 안녕\n  you: 네\n\nOn screen now:\n0: button \"Copy\"\n";
    assert_eq!(recent_block(prompt, 2).unwrap(), "\n  them: 안녕\n  you: 네\n\n");
    //: No history and no block is the ordinary case, not a failure.
    assert_eq!(recent_block("On screen now:\n0: button \"Copy\"\n", 0).unwrap(), "");
    //: History and no block is converse's format having moved underneath the harness.
    assert!(recent_block("On screen now:\n", 4).is_err());
    //: A block that cannot be bounded is the same failure from the other side.
    assert!(recent_block("Recently:\n  them: 안녕\n", 2).is_err());
    //: **`remember_turns = 0` is a real arm now** — it means carry nothing, where it used to mean
    //: carry everything — so a correctly empty window must not abort.
    assert_eq!(recent_block("On screen now:\n", 0).unwrap(), "");
}

/// The filler is a condition of the result, so it is checked rather than trusted: **a line naming a
/// target would put the anchor back in the window** and turn a distance-12 probe into a distance-1
/// one. A harness that cannot fail.
#[test]
fn no_filler_line_names_anything_the_probes_are_looking_for() {
    let all = cases();
    let mut targets: Vec<String> = Vec::new();
    for c in &all {
        if let Some(w) = &c.want {
            targets.push(w.clone());
        }
        targets.extend(c.forbid.iter().cloned());
    }
    assert!(targets.len() >= 7, "the targets went missing: {targets:?}");
    for line in argmind_measure::conversation::FILLER {
        for t in &targets {
            assert!(!line.to_lowercase().contains(&t.to_lowercase()),
                    "filler {line:?} names {t:?} — that is the confound this measures");
        }
    }
}
