//! Choosing: given a screen and a goal, does it name the right thing — or honestly nothing?
//!
//! The oldest measurement here and the most quoted: **21 of 28 answerable right, 7 of 9 declines,
//! 28 of 37 correct decisions**, `qwen3:8b` thinking off, three repeats, worst counted. It is also
//! the one whose noise is known: 21, 23 and 24 across three runs of the same configuration.
//!
//! **Worst of three, not best and not majority.** A configuration that is right twice and wrong
//! once is a configuration that is wrong, because the person only ever gets one draw.
//!
//! **A screen with no right answer is half the experiment.** Nine of the 37 goals cannot be done
//! with what is listed, and saying so is the correct answer — without them a model that clicks
//! something every time scores well, which is the failure mode that matters most: *a wrong click is
//! worse than an honest "I can't see that here".*

use std::collections::BTreeMap;

use argmind_core::contract::Widget;

pub const RIGHT: &str = "right";
pub const WRONG: &str = "wrong";
pub const MISSED: &str = "missed";
pub const DECLINED: &str = "declined";

/// **Worst first when sorted**: right, declined, missed, wrong. `counted` is the last of these that
/// a goal produced across its repeats.
pub const SEVERITY: [&str; 4] = [RIGHT, DECLINED, MISSED, WRONG];

/// One goal: a screen, what the person wants, and what counts as having done it.
#[derive(Clone, Debug)]
pub struct Goal {
    pub screen: String,
    pub goal: String,
    /// `None` means **nothing on this screen can do it** and the right answer is to say so.
    pub accept: Option<Vec<String>>,
}

/// One goal's result. `chosen` is the name it clicked, or nothing.
pub fn outcome(chosen: Option<&str>, accept: Option<&[String]>) -> &'static str {
    match (accept, chosen) {
        (None, None) => DECLINED,
        (None, Some(_)) => WRONG,
        (Some(_), None) => MISSED,
        (Some(ok), Some(c)) => if ok.iter().any(|a| a == c) { RIGHT } else { WRONG },
    }
}

/// The verdict for a goal, over its repeats: **the worst one**.
pub fn counted(results: &[&str]) -> &'static str {
    let mut worst = RIGHT;
    let mut rank = 0;
    for r in results {
        let here = SEVERITY.iter().position(|s| s == r).unwrap_or(SEVERITY.len() - 1);
        if here >= rank {
            rank = here;
            worst = SEVERITY[here];
        }
    }
    worst
}

/// One scored row, in the shape the result files have carried since the first run.
#[derive(Debug)]
pub struct Row {
    pub screen: String,
    pub goal: String,
    pub accept: Option<Vec<String>>,
    pub chose: Vec<Option<String>>,
    pub results: Vec<&'static str>,
    pub counted: &'static str,
    /// Whether the repeats agreed. **A single run is not this number** — at temperature 0 they
    /// still differ, and how often is a property of the configuration worth recording.
    pub agreed: bool,
    /// Where in the listed screen each choice was, and where the right answers are. Recorded
    /// because *"it picks the first thing"* and *"it picks the right thing"* look identical on a
    /// screen whose right answer is first.
    pub chose_at: Vec<Option<usize>>,
    pub right_at: Vec<usize>,
    pub n_listed: usize,
}

/// Score what was chosen, without choosing it — so a recorded run can be re-judged with no model.
pub fn score(goal: &Goal, widgets: &[Widget], chose: Vec<Option<String>>) -> Row {
    let accept = goal.accept.as_deref();
    let results: Vec<&'static str> =
        chose.iter().map(|c| outcome(c.as_deref(), accept)).collect();
    let names: Vec<&str> = widgets.iter().map(|w| w.name.as_str()).collect();
    let chose_at = chose.iter()
        .map(|c| c.as_deref().and_then(|c| names.iter().position(|n| *n == c))).collect();
    let right_at = names.iter().enumerate()
        .filter(|(_, n)| accept.is_some_and(|a| a.iter().any(|x| x == *n)))
        .map(|(i, _)| i).collect();
    let agreed = chose.iter().collect::<std::collections::BTreeSet<_>>().len() == 1;
    Row { screen: goal.screen.clone(), goal: goal.goal.clone(), accept: goal.accept.clone(),
          counted: counted(&results), results, chose, agreed, chose_at, right_at,
          n_listed: names.len() }
}

/// The headline, and **it is two numbers rather than one**: how many answerable goals were right,
/// and how many unanswerable ones were declined. Quoting either alone is incomplete by
/// construction — a model that always clicks scores the first and fails the second.
#[derive(Debug, Default, PartialEq)]
pub struct Tally {
    pub right: usize,
    pub wrong: usize,
    pub missed: usize,
    pub declined: usize,
    pub answerable: usize,
    pub unanswerable: usize,
    pub disagreed: usize,
}

pub fn tally(rows: &[Row]) -> Tally {
    let mut t = Tally::default();
    for row in rows {
        match row.counted {
            RIGHT => t.right += 1,
            WRONG => t.wrong += 1,
            MISSED => t.missed += 1,
            _ => t.declined += 1,
        }
        if row.accept.is_some() { t.answerable += 1 } else { t.unanswerable += 1 }
        if !row.agreed { t.disagreed += 1 }
    }
    t
}

/// Read `measure/goals.json`.
pub fn goals(raw: &str) -> Vec<Goal> {
    let d: serde_json::Value = serde_json::from_str(raw).expect("goals.json");
    d["goals"].as_array().expect("goals").iter().map(|g| Goal {
        screen: g["screen"].as_str().unwrap_or("").to_string(),
        goal: g["goal"].as_str().unwrap_or("").to_string(),
        //: `null` is not "no accepted answers", it is **"there is no right answer"** — a different
        //: fact, and the one that makes the declines scoreable.
        accept: g["accept"].as_array().map(|a| {
            a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect()
        }),
    }).collect()
}

/// The whole run, as the result file carries it.
pub fn as_json(rows: &[Row], screens: &BTreeMap<String, usize>) -> serde_json::Value {
    serde_json::json!({
        "rows": rows.iter().map(|r| serde_json::json!({
            "screen": r.screen, "goal": r.goal, "accept": r.accept, "chose": r.chose,
            "results": r.results, "counted": r.counted, "agreed": r.agreed,
            "chose_at": r.chose_at, "right_at": r.right_at, "n_listed": r.n_listed,
        })).collect::<Vec<_>>(),
        "screens": screens,
    })
}
