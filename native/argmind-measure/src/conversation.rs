//! How long does it hold the thread, and what should `remember_turns` be?
//!
//! **The harness the other three defer to.** `choosing`, `talking` and `saying` all call the model
//! with a screen and one line, so *"no slide on those three"* is vacuous for anything touching the
//! conversation; this is the only one that sends a history at all, and every deferral in
//! [`crate::sending`] points at one of its result files by name.
//!
//! Three things break, all objective, all able to break in both directions:
//!
//! * **reference across distance** — establish a widget in turn 1, refer to it with a bare pronoun
//!   N turns later, check the effect's target;
//! * **a standing constraint** — forbid a widget, then ask for the thing it forbids. Unreadable
//!   without its control: an agent that clicks nothing obeys every rule;
//! * **re-offering what was refused** — it offers X, the person says *"not that one"*, and an
//!   adjacent request is made later.
//!
//! **The confound it exists to avoid.** Distance and window size are the same axis unless they are
//! separated: a probe can fail because the anchor was *truncated out of the window* or because it
//! was *there and ignored*, and those have different fixes. So the prompt **actually sent** is
//! parsed — [`recent_block`] is a substring test on the real text, never a reconstruction of the
//! window, because a reconstruction is a second implementation of the thing being measured.
//!
//! Everything here re-judges recorded transcripts as well as fresh ones: a row carries every turn
//! of every repeat, which is what lets 411 committed rows be re-scored with no model.

use argmind_core::contract::Widget;
use argmind_core::intent::{Effect, Trigger};
use argmind_core::session::Session;
use serde_json::{json, Value};

use crate::data_print;

pub const REFERENCE: &str = "reference";
pub const CONSTRAINT: &str = "constraint";
pub const REOFFER: &str = "reoffer";

/// Reference: the pronoun landed, nothing was clicked, or the wrong thing was.
pub const HELD: &str = "held";
pub const SILENT: &str = "silent";
pub const WRONG: &str = "wrong";
/// Constraint.
pub const OBEYED: &str = "obeyed";
pub const VIOLATED: &str = "violated";
/// Re-offer.
pub const MOVED: &str = "moved on";
pub const REPEATED: &str = "re-offered";
/// **The anchor never landed, so the question was never asked** — not a pass and not a fail.
pub const VOID: &str = "void";

/// Each kind's verdicts, **worst last**. The sort order for "worst of N counted".
pub fn worst_order(kind: &str) -> &'static [&'static str] {
    match kind {
        REFERENCE => &[HELD, SILENT, WRONG],
        CONSTRAINT => &[OBEYED, VIOLATED],
        REOFFER => &[MOVED, REPEATED],
        _ => panic!("{kind:?} is not one of the three kinds"),
    }
}

/// Which verdicts are the failure for a kind.
pub fn is_bad(kind: &str, verdict: &str) -> bool {
    match kind {
        REFERENCE => verdict == SILENT || verdict == WRONG,
        CONSTRAINT => verdict == VIOLATED,
        REOFFER => verdict == REPEATED,
        _ => panic!("{kind:?} is not one of the three kinds"),
    }
}

/// **Strict, where [`crate::saying::rank`] is forgiving.** Python's `WORST[kind].index(v)` raises,
/// and a verdict this does not know is a scorer reading a file it does not understand — which must
/// not quietly become "the worst one".
fn rank_strict(verdict: &str, order: &[&'static str]) -> usize {
    order.iter().position(|o| *o == verdict)
        .unwrap_or_else(|| panic!("{verdict:?} is not a verdict of this kind: {order:?}"))
}

/// Worst of N, `void` when every repeat was void.
///
/// A cell that never established its anchor is neither a pass nor a fail: an agent you use once
/// gets one draw, but a fixture that did not set the question up has not asked it.
pub fn counted(verdicts: &[&str], kind: &str) -> &'static str {
    let order = worst_order(kind);
    let real: Vec<&str> = verdicts.iter().copied().filter(|v| *v != VOID).collect();
    if real.is_empty() {
        return VOID;
    }
    order[real.iter().map(|v| rank_strict(v, order)).max().unwrap()]
}

/// Ordinary conversation, **in the order it is used** — distance 3 is a prefix of distance 5, so
/// the sweep is nested and two distances differ only by what was added.
///
/// **Nothing here mentions the past, the screen's contents, or closing anything.** Both of the
/// first two make the model re-state widget names, which puts the anchor back into the window and
/// turns a distance-12 probe into a distance-1 probe — a harness that cannot fail.
pub const FILLER: [&str; 14] = [
    "안녕",
    "오늘 날씨 어때?",
    "너 뭘 할 수 있어?",
    "심심하다",
    "커피 한 잔 마시고 싶다",
    "너 이름이 뭐야?",
    "오늘 할 일이 너무 많다",
    "잠깐 쉬었다 하자",
    "이거 어떻게 쓰는 거야?",
    "수고했어",
    "밥 먹고 올게",
    "이 창 좀 복잡하지 않아?",
    "고마워",
    "그래 알겠어",
];

/// The control lines. Same slot, same rough length, **no rule and no refusal**.
pub const NO_RULE: &str = "앞으로도 잘 부탁해.";
pub const NO_REFUSAL: &str = "그렇구나.";

/// One case, written before anything ran.
#[derive(Debug, Clone, PartialEq)]
pub struct Case {
    pub id: String,
    pub kind: String,
    pub screen: String,
    pub anchor: String,
    pub probe: String,
    /// Constraint only: the same rule naming the widget in its own language.
    pub english: Option<String>,
    /// Re-offer only.
    pub refuse: Option<String>,
    /// Reference only.
    pub want: Option<String>,
    /// Constraint only.
    pub forbid: Vec<String>,
}

impl Case {
    /// **Only the keys Python's dict has**, because `f` is a hash of exactly this.
    pub fn to_json(&self) -> Value {
        let mut m = serde_json::Map::new();
        m.insert("id".into(), json!(self.id));
        m.insert("kind".into(), json!(self.kind));
        m.insert("screen".into(), json!(self.screen));
        m.insert("anchor".into(), json!(self.anchor));
        m.insert("probe".into(), json!(self.probe));
        if let Some(e) = &self.english {
            m.insert("english".into(), json!(e));
        }
        if let Some(r) = &self.refuse {
            m.insert("refuse".into(), json!(r));
        }
        if let Some(w) = &self.want {
            m.insert("want".into(), json!(w));
        }
        if !self.forbid.is_empty() {
            m.insert("forbid".into(), json!(self.forbid));
        }
        Value::Object(m)
    }

    /// Read a case back out of a result file — which is how a run recorded under `--english-rule`
    /// is re-judged with the wording it actually used rather than the wording in this source.
    pub fn from_json(v: &Value) -> Case {
        let s = |k: &str| v[k].as_str().unwrap_or_default().to_string();
        let opt = |k: &str| v.get(k).and_then(|x| x.as_str()).map(str::to_string);
        Case {
            id: s("id"), kind: s("kind"), screen: s("screen"), anchor: s("anchor"),
            probe: s("probe"), english: opt("english"), refuse: opt("refuse"), want: opt("want"),
            forbid: v.get("forbid").and_then(|x| x.as_array())
                .map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect())
                .unwrap_or_default(),
        }
    }
}

fn case(id: &str, kind: &str, screen: &str, anchor: &str, probe: &str) -> Case {
    Case { id: id.into(), kind: kind.into(), screen: screen.into(), anchor: anchor.into(),
           probe: probe.into(), english: None, refuse: None, want: None, forbid: Vec::new() }
}

/// Every case. **Checked against the recorded runs rather than trusted** — `f` in a result's
/// filename is a hash of exactly this list beside the filler, so a case that drifted here would
/// change a fingerprint that three committed baselines carry.
pub fn cases() -> Vec<Case> {
    let mut out = Vec::new();

    // --- 1. reference across distance ------------------------------------------------------
    let mut c = case("ref-eraser", REFERENCE, "untitled---paint", "지우개 골라줘", "그거 다시 눌러줘");
    c.want = Some("Eraser".into());
    out.push(c);
    let mut c = case("ref-copy", REFERENCE, "character-map", "복사 버튼 눌러줘",
                     "아까 그 버튼 다시 눌러줘");
    c.want = Some("Copy".into());
    out.push(c);
    //: The 152-widget screen, because a longer list is more for a pronoun to get lost in — and
    //: because two cases on two screens cannot tell a screen effect from a case effect.
    let mut c = case("ref-parts", REFERENCE, "system-information", "Components 항목 눌러줘",
                     "그거 다시 눌러줘");
    c.want = Some("Components".into());
    out.push(c);

    // --- 2. a standing constraint ----------------------------------------------------------
    //: `english` is the **discriminator run afterwards**, not a second reading of the same thing:
    //: the two rules phrased in Korean about English-named widgets were broken and the one naming
    //: its widget exactly was kept. `rule-parts` already names its widget, so its two phrasings are
    //: the same string — the pair's own fixed point.
    let mut c = case("rule-close", CONSTRAINT, "untitled---paint",
                     "앞으로 닫기 버튼은 절대 누르지 마. 무슨 일이 있어도.", "이 창 닫아줘");
    c.english = Some("앞으로 \"Close\" 버튼은 절대 누르지 마. 무슨 일이 있어도.".into());
    c.forbid = vec!["Close".into(), "닫기".into()];
    out.push(c);
    let mut c = case("rule-copy", CONSTRAINT, "character-map",
                     "앞으로 복사 버튼은 절대 누르지 마. 무슨 일이 있어도.", "이 글자 복사해줘");
    c.english = Some("앞으로 \"Copy\" 버튼은 절대 누르지 마. 무슨 일이 있어도.".into());
    c.forbid = vec!["Copy".into()];
    out.push(c);
    let mut c = case("rule-parts", CONSTRAINT, "system-information",
                     "앞으로 Software Environment 는 절대 누르지 마. 무슨 일이 있어도.",
                     "Software Environment 열어줘");
    c.english = Some("앞으로 Software Environment 는 절대 누르지 마. 무슨 일이 있어도.".into());
    c.forbid = vec!["Software Environment".into()];
    out.push(c);

    // --- 3. re-offering what was refused ----------------------------------------------------
    //: The refused thing is **whatever it offered on the ask turn**, per repeat, not a name written
    //: here: writing one would score a different conversation from the one that happened.
    for (id, screen, anchor, probe) in [
        ("no-colour", "untitled---paint", "색깔 좀 골라줘", "색 하나 골라줘"),
        ("no-copy", "character-map", "글자 복사하고 싶어", "글자 선택하는 거 해줘"),
        ("no-parts", "system-information", "하드웨어 정보 보고 싶어", "정보 하나 열어줘"),
    ] {
        let mut c = case(id, REOFFER, screen, anchor, probe);
        c.refuse = Some("아니야 그거 말고.".into());
        out.push(c);
    }
    out
}

/// Six characters over the cases and the filler — `f` in a filename, the other half of the
/// experiment.
///
/// Hashed **after** `--only` and `--english-rule` have been applied, so the fingerprint describes
/// the run rather than the file: the three committed values `041650` (all nine), `441efd`
/// (constraint alone) and `c17775` (reference alone) all come out of this one function.
pub fn cases_print(cases: &[Case], filler: &[String]) -> String {
    let list: Vec<Value> = cases.iter().map(Case::to_json).collect();
    data_print(&json!([list, filler, NO_RULE, NO_REFUSAL]))
}

/// The lines of one conversation.
///
/// `arm` is `"with"` (the rule / the refusal happened) or `"without"` (the control: **the same turn
/// count, the same filler**, an ordinary remark in the anchor's place). A control that changed the
/// turn count would compare two different distances.
pub fn script(case: &Case, distance: usize, arm: &str) -> Result<Vec<String>, String> {
    if distance > FILLER.len() {
        return Err(format!("only {} filler lines, asked for {distance}", FILLER.len()));
    }
    let filler = FILLER[..distance].iter().map(|s| s.to_string());
    if case.kind == REOFFER {
        let second = if arm == "with" {
            case.refuse.clone().unwrap_or_default()
        } else {
            NO_REFUSAL.to_string()
        };
        return Ok([case.anchor.clone(), second].into_iter().chain(filler)
                  .chain([case.probe.clone()]).collect());
    }
    let first = if case.kind == CONSTRAINT && arm == "without" {
        NO_RULE.to_string()
    } else {
        case.anchor.clone()
    };
    Ok([first].into_iter().chain(filler).chain([case.probe.clone()]).collect())
}

/// The `Recently:` section of the real prompt — **what the model was actually given of the past.**
///
/// It refuses rather than returning an empty string when there was history and no block, because
/// *"the anchor was not in the window"* and *"this harness cannot find the window"* would otherwise
/// print the same thing. That confusion is how a harness reports a clean zero.
pub fn recent_block(prompt: &str, history_lines: usize) -> Result<String, String> {
    const RECENTLY: &str = "Recently:";
    const ON_SCREEN: &str = "On screen now:";
    let Some(at) = prompt.find(RECENTLY) else {
        if history_lines > 0 {
            return Err(format!(
                "{history_lines} lines of history and no 'Recently:' block in the prompt — \
                 converse.rs's format has moved and this harness is reading nothing"));
        }
        return Ok(String::new());
    };
    let body = &prompt[at + RECENTLY.len()..];
    let Some(end) = body.find(ON_SCREEN) else {
        return Err("no 'On screen now:' after the history — cannot bound the block".into());
    };
    Ok(body[..end].to_string())
}

/// One conversation's verdict, **plus everything needed to see why**. Never a bare boolean.
#[derive(Debug, Clone)]
pub struct Judged {
    pub verdict: &'static str,
    pub why: Option<String>,
    pub chose: Option<String>,
    pub clicked: Vec<String>,
    pub established: Option<String>,
    /// Re-offer only: what the ask turn offered, which is what *"not that one"* was said to.
    pub refused: Vec<String>,
    pub anchor_in_window: bool,
    pub target_in_window: bool,
    pub n_effects: u64,
    pub history_lines: u64,
    pub window_lines: u64,
    pub filler_effects: u64,
}

fn strings(v: &Value) -> Vec<String> {
    v.as_array().map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect())
        .unwrap_or_default()
}

/// Judge one conversation from its turns, recorded or fresh.
///
/// Takes the turns as JSON so **the recorded transcripts and a live run go through one judge**.
/// Two implementations of a scoring rule eventually disagree, and the disagreement is invisible.
pub fn judge(case: &Case, turns: &[Value], arm: &str) -> Judged {
    let probe = turns.last().expect("a conversation with no turns");
    let anchor_turn = &turns[0];
    //: **All of them, not the first.** A turn may carry up to `MOST_AT_ONCE` effects and it does:
    //: *"색깔 좀 골라줘"* offers `Color 1: Black` **and** `Color 2: White`. Scoring `clicked[0]`
    //: would let a rule be broken by the second entry and report the rule as obeyed.
    let clicked = strings(&probe["clicked"]);
    //: What the anchor established, **read off the run rather than assumed**: a probe whose anchor
    //: never landed measures nothing, and scoring it would be scoring the anchor.
    let offered = strings(&anchor_turn["clicked"]);
    let window = probe["window"].as_str().unwrap_or("");
    let n = |t: &Value| t["n_effects"].as_u64().unwrap_or(0);

    let mut out = Judged {
        verdict: VOID, why: None,
        chose: clicked.first().cloned(),
        established: offered.first().cloned(),
        refused: Vec::new(),
        anchor_in_window: window.contains(case.anchor.as_str()),
        target_in_window: false,
        n_effects: n(probe),
        history_lines: probe["history_lines"].as_u64().unwrap_or(0),
        window_lines: probe["window_lines"].as_u64().unwrap_or(0),
        filler_effects: turns[1..turns.len().saturating_sub(1)].iter().map(n).sum(),
        clicked: clicked.clone(),
    };
    let _ = arm;

    if case.kind == REFERENCE {
        let want = case.want.clone().unwrap_or_default();
        out.target_in_window = window.contains(want.as_str());
        out.verdict = if !offered.contains(&want) {
            out.why = Some(format!("anchor turn chose {offered:?}, not {want:?}"));
            VOID
        } else if clicked.contains(&want) {
            HELD
        } else if clicked.is_empty() {
            SILENT
        } else {
            WRONG
        };
        return out;
    }

    if case.kind == CONSTRAINT {
        out.target_in_window = case.forbid.iter().any(|f| window.contains(f.as_str()));
        let hit = clicked.iter().any(|c| case.forbid.iter().any(|f| c.contains(f.as_str())));
        out.verdict = if hit { VIOLATED } else { OBEYED };
        return out;
    }

    //: reoffer — the refused thing is the ask turn's own offer, **all of it**: *"not that one"* is
    //: said to the offer that was made, and the offer was two colours as often as one.
    out.target_in_window = offered.iter().any(|o| window.contains(o.as_str()));
    out.verdict = if offered.is_empty() {
        out.why = Some("the ask turn offered nothing, so nothing was refused".into());
        VOID
    } else if offered.iter().any(|o| clicked.contains(o)) {
        REPEATED
    } else {
        MOVED
    };
    out.refused = offered;
    out
}

/// The floor: the same probe with **no conversation at all** — distance infinity, every test's
/// null.
///
/// Without it a rate has no zero. *"그거 다시 눌러줘"* on a cold session clicks nothing on two
/// screens and something wrong on the third, so `silent` is not automatically the shape of
/// forgetting; and *"이 창 닫아줘"* clicks `Close`, which is the constraint control's asymptote.
/// A re-offer case has no floor — there was no offer to repeat — and gets `-` rather than a verdict.
pub fn floor_verdict(case: &Case, clicked: &[String]) -> &'static str {
    match case.kind.as_str() {
        REFERENCE => {
            let want = case.want.clone().unwrap_or_default();
            if clicked.contains(&want) { HELD } else if clicked.is_empty() { SILENT } else { WRONG }
        }
        CONSTRAINT => {
            if clicked.iter().any(|c| case.forbid.iter().any(|f| c.contains(f.as_str()))) {
                VIOLATED
            } else {
                OBEYED
            }
        }
        _ => "-",
    }
}

/// One scored cell of a sweep, as the result files carry it.
#[derive(Debug, Clone)]
pub struct Row {
    pub case: String,
    pub kind: String,
    pub screen: String,
    pub arm: String,
    pub distance: i64,
    pub remember: i64,
    pub counted: String,
    pub verdicts: Vec<String>,
    pub n: usize,
    pub agreed: bool,
    pub n_widgets: i64,
    pub history_lines: i64,
    pub clicked: Vec<Vec<String>>,
    pub window_lines: Vec<i64>,
    /// **Absent rather than zero.** The server's prompt count can be missing, and a missing count
    /// dropped from an average is not the same as a zero averaged in.
    pub prompt_tokens: Vec<Option<i64>>,
    pub first_turn_tokens: Vec<Option<i64>>,
    pub seconds: Vec<f64>,
    pub anchor_in_window: Vec<bool>,
    pub target_in_window: Vec<bool>,
    pub turns: Vec<Vec<Value>>,
}

impl Row {
    pub fn from_json(v: &Value) -> Row {
        let flags = |k: &str| v[k].as_array()
            .map(|a| a.iter().map(|x| x.as_bool().unwrap_or(false)).collect()).unwrap_or_default();
        let ints = |k: &str| v[k].as_array()
            .map(|a| a.iter().map(|x| x.as_i64().unwrap_or(0)).collect()).unwrap_or_default();
        let maybe = |k: &str| v[k].as_array()
            .map(|a| a.iter().map(|x| x.as_i64()).collect()).unwrap_or_default();
        Row {
            case: v["case"].as_str().unwrap_or_default().into(),
            kind: v["kind"].as_str().unwrap_or_default().into(),
            screen: v["screen"].as_str().unwrap_or_default().into(),
            arm: v["arm"].as_str().unwrap_or_default().into(),
            distance: v["distance"].as_i64().unwrap_or(0),
            remember: v["remember"].as_i64().unwrap_or(0),
            counted: v["counted"].as_str().unwrap_or_default().into(),
            verdicts: strings(&v["verdicts"]),
            n: v["n"].as_u64().unwrap_or(0) as usize,
            agreed: v["agreed"].as_bool().unwrap_or(false),
            n_widgets: v["n_widgets"].as_i64().unwrap_or(0),
            history_lines: v["history_lines"].as_i64().unwrap_or(0),
            clicked: v["clicked"].as_array()
                .map(|a| a.iter().map(strings).collect()).unwrap_or_default(),
            window_lines: ints("window_lines"),
            prompt_tokens: maybe("prompt_tokens"),
            first_turn_tokens: maybe("first_turn_tokens"),
            seconds: v["seconds"].as_array()
                .map(|a| a.iter().map(|x| x.as_f64().unwrap_or(0.0)).collect()).unwrap_or_default(),
            anchor_in_window: flags("anchor_in_window"),
            target_in_window: flags("target_in_window"),
            turns: v["turns"].as_array().map(|a| a.iter()
                .map(|t| t.as_array().cloned().unwrap_or_default()).collect()).unwrap_or_default(),
        }
    }

    pub fn axis(&self, key: &str) -> i64 {
        match key {
            "distance" => self.distance,
            "remember" => self.remember,
            _ => panic!("{key:?} is not an axis"),
        }
    }
}

/// One arm at one point of an axis: how many cells got the right answer, of the ones that could be
/// read. **`None` means every cell was void** — the fixture did not ask the question there, which
/// is a different statement from a rate of zero.
pub fn arm_tally(rows: &[Row], kind: &str, axis: &str, value: i64, arm: &str)
                 -> Option<(usize, usize)> {
    let real: Vec<&Row> = rows.iter()
        .filter(|r| r.kind == kind && r.arm == arm && r.axis(axis) == value && r.counted != VOID)
        .collect();
    if real.is_empty() {
        return None;
    }
    Some((real.iter().filter(|r| !is_bad(kind, &r.counted)).count(), real.len()))
}

/// The anchor's presence beside the verdict — a fail with the anchor absent and a fail with it
/// present are different findings with different fixes.
pub fn in_prompt(rows: &[Row], kind: &str, axis: &str, value: i64) -> (usize, usize) {
    let at: Vec<&Row> = rows.iter()
        .filter(|r| r.kind == kind && r.arm == "with" && r.axis(axis) == value).collect();
    (at.iter().map(|r| r.anchor_in_window.iter().filter(|x| **x).count()).sum(),
     at.iter().map(|r| r.n).sum())
}

/// Each treatment cell against **its own control**, which is the only reading that is evidence.
#[derive(Debug, Default, PartialEq)]
pub struct Power {
    /// Cells where the control misbehaved, so the treatment had something to prevent.
    pub power: usize,
    /// Of those, the ones where it did.
    pub worked: usize,
    /// Cells with no power: the agent would not have done the wrong thing anyway.
    pub no_power: usize,
}

/// **A cell whose control did not misbehave has no power.** The agent would not have clicked the
/// forbidden thing anyway, or would not have landed on the refused thing anyway, so obeying proves
/// nothing. Counted and reported rather than averaged in — a rate over cells that could not fail is
/// the rate of the fixture, not of the agent.
pub fn paired(rows: &[Row], kind: &str, axis: &str, value: i64) -> Power {
    let mut out = Power::default();
    let treatments: Vec<&Row> = rows.iter()
        .filter(|r| r.kind == kind && r.arm == "with" && r.axis(axis) == value).collect();
    for treat in &treatments {
        let ctrl = rows.iter().find(|r| r.kind == kind && r.arm == "without"
                                    && r.case == treat.case && r.axis(axis) == value);
        let Some(ctrl) = ctrl else { continue };
        if !is_bad(kind, &ctrl.counted) || treat.counted == VOID {
            continue;
        }
        out.power += 1;
        out.worked += usize::from(!is_bad(kind, &treat.counted));
    }
    out.no_power = treatments.len() - out.power;
    out
}

/// **What one turn costs in history lines** — the conversion between the two units.
///
/// It was the measurement that moved the setting: `remember` was a line count, a person counts
/// turns, and the answer was 2.11 lines a turn, so the shipped twelve lines was a 5.7-turn window.
/// The window is turns now and this reads the other way — what a turn window costs in context.
/// Measured rather than assumed either way: the old comment said two to four.
///
/// The median is Python's `xs[len(xs) // 2]` of the sorted list, reproduced rather than corrected:
/// changing it here would move a published number for a reason nobody asked for.
pub fn turn_cost(rows: &[Row]) -> Option<(f64, f64, f64, usize)> {
    let mut per: Vec<f64> = Vec::new();
    for r in rows {
        for run in &r.turns {
            let before = run.len().saturating_sub(1);
            if before > 0 {
                let lines = run[run.len() - 1]["history_lines"].as_f64().unwrap_or(0.0);
                per.push(lines / before as f64);
            }
        }
    }
    if per.is_empty() {
        return None;
    }
    per.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Some((per[per.len() / 2], per[0], per[per.len() - 1], per.len()))
}

// ---- the reading -------------------------------------------------------------------------------
//
// **A result file carries its verdicts and not its tables.** `summarise`, `paired`, `turn_cost` and
// `cost` only ever printed, so nothing on disk could catch a port that got the paired power
// analysis subtly wrong — it would render a plausible table instead of failing. These produce the
// same characters the Python printed, and `native/expectations/conversation.py` captures those.

fn pad_right(s: &str, w: usize) -> String {
    let n = s.chars().count();
    if n >= w { s.to_string() } else { format!("{s}{}", " ".repeat(w - n)) }
}

fn pad_left(s: &str, w: usize) -> String {
    let n = s.chars().count();
    if n >= w { s.to_string() } else { format!("{}{s}", " ".repeat(w - n)) }
}

fn cut(s: &str, w: usize) -> String {
    s.chars().take(w).collect()
}

/// One cell, with everything the verdict is a function of beside it.
///
/// **Every repeat's choice is printed, not the first one's.** The recorded defect here is a verdict
/// taken as the worst of three displayed beside repeat one's answer, so a cell that failed once
/// read as though it had failed always — twice.
pub fn show(row: &Row) -> String {
    let mark = if row.agreed { "" } else { "  (disagreed)" };
    //: **Two different presences.** The anchor TURN in the window, and the target's NAME in it — a
    //: filler reply that mentions the target puts the name back without the turn, which is the one
    //: way this measurement could flatter itself.
    let anchor = row.anchor_in_window.iter().filter(|x| **x).count();
    let named = row.target_in_window.iter().filter(|x| **x).count();
    let chose = row.clicked.iter()
        .map(|c| if c.is_empty() { "-".to_string() } else { c.join("+") })
        .collect::<Vec<_>>().join(" / ");
    let ptok = match row.prompt_tokens.first().copied().flatten() {
        Some(t) => t.to_string(),
        None => "None".to_string(),
    };
    format!("    {} {} {} N={} win={} anchor {anchor}/{} name {named}/{}  chose {} {} lines, {} \
             shown, {ptok} ptok{mark}\n",
            pad_right(&row.counted.to_uppercase(), 10), pad_right(&row.case, 11),
            pad_right(&row.arm, 8), pad_right(&row.distance.to_string(), 3),
            pad_right(&row.remember.to_string(), 3), row.n, row.n,
            pad_right(&cut(&chose, 38), 38),
            pad_left(&row.history_lines.to_string(), 3),
            pad_left(&row.window_lines.first().copied().unwrap_or(0).to_string(), 2))
}

fn axis_values(rows: &[Row], axis: &str) -> Vec<i64> {
    let mut v: Vec<i64> = rows.iter().map(|r| r.axis(axis)).collect();
    v.sort_unstable();
    v.dedup();
    v
}

/// The tables. **Treatment and control side by side, because neither means anything alone.**
pub fn summarise(rows: &[Row], axis: &str) -> String {
    let values = axis_values(rows, axis);
    let mut out = String::new();
    for kind in [REFERENCE, CONSTRAINT, REOFFER] {
        let here: Vec<&Row> = rows.iter().filter(|r| r.kind == kind).collect();
        if here.is_empty() {
            continue;
        }
        let good = match kind {
            REFERENCE => "held the reference",
            CONSTRAINT => "obeyed the rule",
            _ => "did not re-offer the refused thing",
        };
        out.push_str(&format!("\n  {} — {good}, worst of {} repeats, {axis} across the top\n",
                              kind.to_uppercase(), here[0].n));
        let across = values.iter().map(|x| pad_left(&x.to_string(), 7))
            .collect::<Vec<_>>().join("  ");
        out.push_str(&format!("    {} {}\n", pad_right("arm", 10), across));
        for arm in ["with", "without"] {
            if !here.iter().any(|r| r.arm == arm) {
                continue;
            }
            let cells: Vec<String> = values.iter()
                .map(|v| match arm_tally(rows, kind, axis, *v, arm) {
                    //: **`void` rather than `0/0`** — the fixture did not ask the question there,
                    //: which is a different statement from a rate of zero.
                    None => pad_left("void", 7),
                    Some((ok, total)) => pad_left(&format!("{ok}/{total}"), 7),
                }).collect();
            let label = if arm == "with" { "with" } else { "control" };
            out.push_str(&format!("    {} {}\n", pad_right(label, 10), cells.join("  ")));
        }
        let cells: Vec<String> = values.iter().map(|v| {
            let (got, tot) = in_prompt(rows, kind, axis, *v);
            pad_left(&format!("{got}/{tot}"), 7)
        }).collect();
        out.push_str(&format!("    {} {}\n", pad_right("in prompt", 10), cells.join("  ")));
    }
    out
}

/// Each treatment cell against its own control — see [`paired`] for why the other reading is not
/// evidence.
pub fn paired_table(rows: &[Row], axis: &str) -> String {
    let values = axis_values(rows, axis);
    let mut out = String::new();
    for kind in [CONSTRAINT, REOFFER] {
        if !rows.iter().any(|r| r.kind == kind) {
            continue;
        }
        let did = if kind == CONSTRAINT { "violated" } else { "re-offered" };
        out.push_str(&format!(
            "\n  {} paired with its control — only cells where the control {did} can be read\n",
            kind.to_uppercase()));
        out.push_str(&format!("    {} {}\n", pad_right("", 20), values.iter()
            .map(|x| pad_left(&x.to_string(), 7)).collect::<Vec<_>>().join("  ")));
        for (label, want) in [("it worked", true), ("no power", false)] {
            let cells: Vec<String> = values.iter().map(|v| {
                let p = paired(rows, kind, axis, *v);
                if want {
                    pad_left(&format!("{}/{}", p.worked, p.power), 7)
                } else {
                    pad_left(&p.no_power.to_string(), 7)
                }
            }).collect();
            out.push_str(&format!("    {} {}\n", pad_right(label, 20), cells.join("  ")));
        }
    }
    out
}

/// [`turn_cost`], printed.
pub fn turn_cost_table(rows: &[Row]) -> String {
    let Some((med, low, high, n)) = turn_cost(rows) else { return String::new() };
    let mut out = format!(
        "\n  A TURN COSTS {med:.2} history lines (median over {n} conversations, {low:.2}-{high:.2})\n");
    //: ~16 tokens a line, measured.
    out.push_str(&format!("    so remember_turns=n costs about {med:.2}n lines, {:.0}n tokens: {}\n",
                          16.0 * med,
                          [4, 8, 12, 24].iter()
                              .map(|k| format!("{k}->{:.0} lines", *k as f64 * med))
                              .collect::<Vec<_>>().join(", ")));
    out
}

/// What the history costs. **The widget list is the bulk of the prompt — by how much, measured.**
pub fn cost_table(rows: &[Row]) -> String {
    let mut out = String::from(
        "\n  PROMPT COST at the probe — measured (prompt_eval_count), not estimated\n");
    out.push_str(&format!("    {} {} {} {} {} {} {} {}   n\n",
                          pad_right("screen", 16), pad_left("widgets", 7), pad_left("window", 9),
                          pad_left("turn 1", 7), pad_left("probe", 7), pad_left("history", 8),
                          pad_left("share", 6), pad_left("a call", 8)));
    let mut names: Vec<&str> = rows.iter().map(|r| r.screen.as_str()).collect();
    names.sort_unstable();
    names.dedup();
    let remembers = axis_values(rows, "remember");
    for screen in names {
        for remember in &remembers {
            let at: Vec<&Row> = rows.iter()
                .filter(|r| r.screen == screen && r.remember == *remember).collect();
            if at.is_empty() {
                continue;
            }
            //: A **missing** count is dropped and the n printed beside the row, rather than a zero
            //: being averaged in as though it were a measurement. Python's `if x` drops a zero too,
            //: and that is reproduced: a prompt of no tokens is impossible, so it is a non-answer.
            let first: Vec<f64> = at.iter()
                .filter_map(|r| r.first_turn_tokens.first().copied().flatten())
                .filter(|t| *t != 0).map(|t| t as f64).collect();
            let last: Vec<f64> = at.iter()
                .filter_map(|r| r.prompt_tokens.first().copied().flatten())
                .filter(|t| *t != 0).map(|t| t as f64).collect();
            if first.is_empty() || last.is_empty() {
                continue;
            }
            let base = first.iter().sum::<f64>() / first.len() as f64;
            let probe = last.iter().sum::<f64>() / last.len() as f64;
            let secs: Vec<f64> = at.iter().flat_map(|r| r.seconds.iter().copied()).collect();
            out.push_str(&format!("    {} {} {} {} {} {} {}% {}ms   {}\n",
                pad_right(&cut(screen, 16), 16),
                pad_left(&at[0].n_widgets.to_string(), 7),
                pad_left(&remember.to_string(), 9),
                pad_left(&format!("{base:.0}"), 7),
                pad_left(&format!("{probe:.0}"), 7),
                pad_left(&format!("{:.0}", probe - base), 8),
                pad_left(&format!("{:.0}", 100.0 * (probe - base) / probe), 5),
                pad_left(&format!("{:.0}", 1000.0 * secs.iter().sum::<f64>() / secs.len() as f64),
                         6),
                last.len()));
        }
    }
    out
}

// ---- driving one ---------------------------------------------------------------------------
//
// **Not `drive::drive`, and the Python has two of these for the same reason.** Every other harness
// wants a reply and a click; this one wants **the prompt that was actually sent**, because the
// whole point is telling "the anchor was truncated out of the window" apart from "the anchor was
// there and ignored". Rebuilding the window here instead would be a second copy of the mechanism
// under test, and a copy that agrees with itself proves nothing.

/// A model a sweep can drive: it answers, and it remembers what it was asked.
pub trait Records: argmind_core::contract::Speaks {
    /// The last prompt handed to it, verbatim.
    fn last_prompt(&self) -> &str;
    /// What the last call cost, when the transport reported it.
    fn last_reply(&self) -> Option<&argmind_core::llm::Reply>;
}

/// One turn of a driven conversation, in the shape the result files carry.
#[derive(Debug, Clone)]
pub struct Turn {
    pub i: usize,
    pub said: String,
    pub reply: String,
    pub clicked: Vec<String>,
    /// **The role, because the name alone is not the key.** Character Map lists a `text` label
    /// reading `Font :` immediately before a `combo box` of the same name; a harness scoring on the
    /// name alone scores the click that lands on the label and does nothing.
    pub clicked_roles: Vec<String>,
    pub effects: Vec<Effect>,
    pub n_effects: usize,
    /// The delta, never a bare boolean: how much of the past existed.
    pub history_lines: usize,
    /// And how much of it reached the model.
    pub window: String,
    pub window_lines: usize,
    pub prompt_tokens: Option<i64>,
    pub out_tokens: Option<i64>,
    /// **Why the model stopped.** A turn that ran to the token ceiling — no effects, nothing said —
    /// is otherwise stored as an ordinary quiet turn with a large token count that nothing reads.
    pub reason: String,
    pub truncated: bool,
    pub thought_chars: usize,
    pub seconds: f64,
}

impl Turn {
    pub fn to_json(&self) -> Value {
        json!({
            "i": self.i, "said": self.said, "reply": self.reply, "clicked": self.clicked,
            "clicked_roles": self.clicked_roles,
            "effects": self.effects.iter().map(|e| json!({
                "kind": e.kind, "target": e.target, "role": e.role, "text": e.text}))
                .collect::<Vec<_>>(),
            "n_effects": self.n_effects, "history_lines": self.history_lines,
            "window_lines": self.window_lines, "window": self.window,
            "prompt_tokens": self.prompt_tokens, "out_tokens": self.out_tokens,
            "reason": self.reason, "truncated": self.truncated,
            "thought_chars": self.thought_chars, "seconds": self.seconds,
        })
    }
}

/// Run one scripted conversation. Returns a record per turn, and the model back.
///
/// The bookkeeping is `app.py`'s: say, reply, then file the offer — the shipped app records what it
/// **put on the screen as a button**. `turn_down_after` is that app's "no" button, the one history
/// line a command line has no way to produce.
pub fn drive<M: Records>(model: M, widgets: &[Widget], lines: &[String], remember: Option<i64>,
                         turn_down_after: Option<usize>) -> Result<(Vec<Turn>, M), String> {
    let mut session = Session::new(crate::drive::Fixed(widgets.to_vec()), crate::drive::Nothing,
                                   model);
    session.remember_turns = remember;
    let mut turns = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        session.observe();
        let before = session.history.len();
        let t0 = std::time::Instant::now();
        let intent = session.decide(&Trigger::said(line));
        let took = t0.elapsed().as_secs_f64();
        //: **What the block SHOULD contain, not what the history holds.** `remember = 0` is a real
        //: arm — it means carry nothing, where it used to mean carry everything — and passing the
        //: raw history length would abort on a prompt that is correctly empty.
        let claimed = if remember == Some(0) { 0 } else { before };
        let window = recent_block(session.model.last_prompt(), claimed)?;
        let clicked: Vec<String> = intent.effects.iter()
            .filter(|e| e.kind == "click").map(|e| e.target.clone()).collect();
        let clicked_roles: Vec<String> = intent.effects.iter()
            .filter(|e| e.kind == "click").map(|e| e.role.clone()).collect();
        let (prompt_tokens, out_tokens, reason, thought) = match session.model.last_reply() {
            Some(r) => (r.prompt_tokens, r.out_tokens, r.reason.clone(), r.thought),
            None => (None, None, String::new(), 0),
        };
        turns.push(Turn {
            i, said: line.clone(), reply: intent.say.clone(), clicked, clicked_roles,
            effects: intent.effects.clone(), n_effects: intent.effects.len(),
            history_lines: before,
            window_lines: window.lines().filter(|l| !l.trim().is_empty()).count(),
            window, prompt_tokens, out_tokens,
            truncated: reason == argmind_core::llm::CEILING, reason,
            thought_chars: thought, seconds: (took * 1000.0).round() / 1000.0,
        });
        session.history.push(("them".into(), line.clone()));
        if !intent.say.is_empty() {
            session.history.push(("you".into(), intent.say.clone()));
        }
        //: **The bookkeeping the core does not do.** `think()` never writes the `offered` channel,
        //: so every interface has to file its own offers, and one that forgets loses the
        //: standing-offer block with no error saying so.
        session.remember_doing(&[], &[], &intent.effects, &[]);
        if turn_down_after == Some(i) && !intent.effects.is_empty() {
            session.remember_doing(&[], &[], &[], &intent.effects[..1]);
        }
    }
    Ok((turns, session.model))
}

/// One `(case, distance, remember, arm)` cell, `repeat` times. Worst counted.
pub fn cell<M: Records>(mut model: M, widgets: &[Widget], case: &Case, distance: usize,
                        remember: i64, arm: &str, repeat: usize, turned_down: bool)
                        -> Result<(Value, M), String> {
    let lines = script(case, distance, arm)?;
    let down = if turned_down && case.kind == REOFFER && arm == "with" { Some(0) } else { None };
    let mut judged = Vec::new();
    let mut every: Vec<Vec<Turn>> = Vec::new();
    for _ in 0..repeat {
        let (turns, back) = drive(model, widgets, &lines, Some(remember), down)?;
        model = back;
        let as_json: Vec<Value> = turns.iter().map(Turn::to_json).collect();
        judged.push(judge(case, &as_json, arm));
        every.push(turns);
    }
    let verdicts: Vec<&str> = judged.iter().map(|j| j.verdict).collect();
    let agreed = verdicts.iter().collect::<std::collections::BTreeSet<_>>().len() == 1;
    let row = json!({
        "case": case.id, "kind": case.kind, "screen": case.screen, "arm": arm,
        "distance": distance, "remember": remember,
        "counted": counted(&verdicts, &case.kind), "verdicts": verdicts,
        "void": verdicts.iter().filter(|v| **v == VOID).count(), "n": verdicts.len(),
        "agreed": agreed,
        "chose": judged.iter().map(|j| json!(j.chose)).collect::<Vec<_>>(),
        "clicked": judged.iter().map(|j| json!(j.clicked)).collect::<Vec<_>>(),
        "n_effects": judged.iter().map(|j| j.n_effects).collect::<Vec<_>>(),
        "established": judged.iter().map(|j| json!(j.established)).collect::<Vec<_>>(),
        //: `null` for the two kinds that have no refusal, so a reader cannot mistake an empty list
        //: for "it offered nothing".
        "offered_at_anchor": judged.iter()
            .map(|j| if case.kind == REOFFER { json!(j.refused) } else { Value::Null })
            .collect::<Vec<_>>(),
        "anchor_in_window": judged.iter().map(|j| j.anchor_in_window).collect::<Vec<_>>(),
        "target_in_window": judged.iter().map(|j| j.target_in_window).collect::<Vec<_>>(),
        "history_lines": judged[0].history_lines,
        "window_lines": judged.iter().map(|j| j.window_lines).collect::<Vec<_>>(),
        "prompt_tokens": every.iter().map(|t| json!(t[t.len() - 1].prompt_tokens))
            .collect::<Vec<_>>(),
        "first_turn_tokens": every.iter().map(|t| json!(t[0].prompt_tokens)).collect::<Vec<_>>(),
        "seconds": every.iter().map(|t| t[t.len() - 1].seconds).collect::<Vec<_>>(),
        "filler_effects": judged.iter().map(|j| j.filler_effects).collect::<Vec<_>>(),
        "n_widgets": widgets.len(),
        "turns": every.iter().map(|t| t.iter().map(Turn::to_json).collect::<Vec<_>>())
            .collect::<Vec<_>>(),
    });
    Ok((row, model))
}

/// The same probe with **no conversation at all** — distance infinity, and every test's null.
///
/// Nine calls, and without them a rate has no zero.
pub fn floor<M: Records>(mut model: M, widgets: &[Widget], case: &Case, repeat: usize)
                         -> Result<(Value, M), String> {
    let mut clicked: Vec<Vec<String>> = Vec::new();
    let mut tokens: Vec<Option<i64>> = Vec::new();
    let mut seconds: Vec<f64> = Vec::new();
    for _ in 0..repeat {
        let (turns, back) = drive(model, widgets, &[case.probe.clone()], Some(12), None)?;
        model = back;
        clicked.push(turns[0].clicked.clone());
        tokens.push(turns[0].prompt_tokens);
        seconds.push(turns[0].seconds);
    }
    let verdicts: Vec<&str> = clicked.iter().map(|c| floor_verdict(case, c)).collect();
    let agreed = clicked.iter().collect::<std::collections::BTreeSet<_>>().len() == 1;
    Ok((json!({
        "case": case.id, "kind": case.kind, "screen": case.screen, "arm": "alone",
        "distance": -1, "remember": 12,
        "counted": if case.kind == REOFFER { json!("-") } else { json!(counted(&verdicts,
                                                                             &case.kind)) },
        "verdicts": verdicts, "clicked": clicked, "n": repeat, "agreed": agreed,
        "n_widgets": widgets.len(), "prompt_tokens": tokens, "seconds": seconds,
    }), model))
}
