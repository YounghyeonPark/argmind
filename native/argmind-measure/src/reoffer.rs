//! Saying yes to an offer, saying no to it, and asking about it later.
//!
//! Six groups over four screens, 104 cases, and the one that carries the result is **ACCEPT**: the
//! person says yes one turn after an offer and the agent should do the thing again. It went from
//! 15 of 39 to 30 of 39 — 15 fixed, 0 broken, p = 0.00006 — when an unanswered offer began being
//! rendered beside the turn.
//!
//! **The controls are not decoration.** MEMORY asks about the past and the right answer is to do
//! nothing; REFUSE says no and the right answer is to do nothing; ANEW asks for something else and
//! the right answer is that thing. Without them, an agent that acts on anything following an offer
//! scores perfectly on the primary endpoint and is worse to live with.
//!
//! **A case is `void` when the offer never reached the second prompt.** That is not a failure, it
//! is a case that could not be asked — and counting it as one would let a harness that stopped
//! writing offers score well.

use crate::saying::rank;

pub const MEMORY: &str = "memory";
pub const REFUSE: &str = "refuse";
pub const ACCEPT: &str = "accept";
pub const FARACCEPT: &str = "faraccept";
pub const RESTATE: &str = "restate";
pub const ANEW: &str = "anew";
pub const GROUPS: [&str; 6] = [MEMORY, REFUSE, ACCEPT, FARACCEPT, RESTATE, ANEW];

/// The controls: doing nothing is right.
pub const QUIET: &str = "quiet";
pub const OTHER: &str = "other";
pub const AGAIN: &str = "again";
/// Acceptance: doing the offered thing again is right.
pub const ACTED: &str = "acted";
pub const DRIFTED: &str = "drifted";
pub const IGNORED: &str = "ignored";
/// Naming a target: doing that one thing is right.
pub const LANDED: &str = "landed";
pub const EXTRA: &str = "extra";
pub const WRONG: &str = "wrong";
pub const STUCK: &str = "stuck";
/// **Split from `wrong` on 2026-09-13** — "said the right thing and did nothing" and "did the wrong
/// thing" are different defects, and the fix's one control break turned out to be the first.
pub const SILENT: &str = "silent";
pub const VOID: &str = "void";

/// What an effect is, for the purpose of *"is this the same thing again?"*
///
/// **A click is its name and its role**; anything else is its kind and its text. The role is in
/// here for the reason it is in `find()`: a label and a combo box of one name are two things.
pub fn key_of(effect: &serde_json::Value) -> (String, String, String) {
    let kind = effect["kind"].as_str().unwrap_or("").to_string();
    if kind == "click" {
        return (kind, effect["target"].as_str().unwrap_or("").into(),
                effect["role"].as_str().unwrap_or("").into());
    }
    (kind, effect["text"].as_str().unwrap_or("").into(), String::new())
}

/// The `offered:` lines in a rendered history block.
pub fn offered_lines(window: &str) -> Vec<String> {
    window.lines()
        .filter_map(|x| x.strip_prefix("  offered: ").map(str::to_string))
        .collect()
}

/// Did the offer actually reach the prompt the second turn was answered from?
///
/// **The guard that makes a `void` verdict possible.** Without it a harness that stopped writing
/// offers would score every acceptance as a failure to accept, which is a plausible zero produced
/// by the measuring code rather than by the model.
pub fn offer_reached(first_effects: &[serde_json::Value], last_window: &str) -> bool {
    let names: Vec<String> = first_effects.iter().map(|e| {
        let t = e["target"].as_str().unwrap_or("");
        if t.is_empty() { e["text"].as_str().unwrap_or("").to_string() } else { t.to_string() }
    }).collect();
    let lines = offered_lines(last_window);
    !lines.is_empty()
        && names.iter().any(|n| !n.is_empty() && lines.iter().any(|l| l.contains(n.as_str())))
}

/// One case's verdict, from the two turns it is made of.
pub fn verdict_of(kind: &str, first_effects: &[serde_json::Value],
                  second_effects: &[serde_json::Value], last_window: &str,
                  want: Option<(&str, &str)>) -> &'static str {
    let offered: Vec<_> = first_effects.iter().map(key_of).collect();
    let again: Vec<_> = second_effects.iter().map(key_of).collect();
    if offered.is_empty() || !offer_reached(first_effects, last_window) {
        return VOID;
    }
    let repeated = again.iter().any(|k| offered.contains(k));
    if kind == MEMORY || kind == REFUSE {
        return if again.is_empty() { QUIET } else if repeated { AGAIN } else { OTHER };
    }
    if kind == ACCEPT || kind == FARACCEPT {
        return if repeated { ACTED } else if again.is_empty() { IGNORED } else { DRIFTED };
    }
    let Some((name, role)) = want else { return VOID };
    let clicks: Vec<(String, String)> = second_effects.iter()
        .filter(|e| e["kind"] == "click")
        .map(|e| (e["target"].as_str().unwrap_or("").to_string(),
                  e["role"].as_str().unwrap_or("").to_string())).collect();
    if clicks.iter().any(|(n, r)| n == name && r == role) {
        return if second_effects.len() == 1 { LANDED } else { EXTRA };
    }
    if repeated {
        return STUCK;
    }
    if again.is_empty() { SILENT } else { WRONG }
}

/// Whether a verdict is the right answer for its group.
pub fn is_right(kind: &str, verdict: &str) -> bool {
    match kind {
        MEMORY | REFUSE => verdict == QUIET,
        ACCEPT | FARACCEPT => verdict == ACTED,
        _ => verdict == LANDED,
    }
}

/// **The exact two-sided paired sign test.** No approximation: the numbers here are small enough
/// that one would be doing the arithmetic wrong on purpose.
///
/// `k >= 6` discordant pairs one way gives p = 0.031; `k = 5` gives 0.0625 and does not clear the
/// floor. That is why a set with fewer than six failures cannot show a repair as significant
/// however good the repair is — a fact worth printing before a run rather than discovering after.
pub fn sign_test(fixed: usize, broken: usize) -> f64 {
    let n = fixed + broken;
    if n == 0 {
        return 1.0;
    }
    let k = fixed.min(broken);
    let mut tail = 0.0f64;
    for i in 0..=k {
        tail += binomial(n, i);
    }
    (2.0 * tail * 0.5f64.powi(n as i32)).min(1.0)
}

fn binomial(n: usize, k: usize) -> f64 {
    let mut out = 1.0f64;
    for i in 0..k {
        out = out * (n - i) as f64 / (i + 1) as f64;
    }
    out
}

/// Worst-of over repeats, given a group's ordering with the worst last.
pub fn worst<'a>(verdicts: &[&'a str], order: &[&'a str]) -> &'a str {
    let mut best = order[0];
    let mut at = 0;
    for v in verdicts {
        let here = rank(v, order);
        if here >= at {
            at = here;
            best = order[here.min(order.len() - 1)];
        }
    }
    best
}

/// What the prompt puts **between the widget list and the turn** — the region the standing-offer
/// block occupies.
///
/// **Read generically rather than by knowing what the block says.** The check this replaced
/// compared offered wordings, which a `converse.py` change does not touch, so it printed *"no
/// treatment was applied at all"* beside a real p-value. A delivery check has to read the
/// treatment, not guess at it.
pub fn slot_lines(prompt: &str) -> Result<Vec<String>, String> {
    const ON_SCREEN: &str = "On screen now:";
    const NO_WIDGETS: &str = "(nothing named";
    if !prompt.contains(ON_SCREEN) {
        return Err("no 'On screen now:' in the prompt — converse's format has moved and this is \
                    reading nothing".into());
    }
    let tail = prompt.split(ON_SCREEN).nth(1).unwrap();
    let mut cuts: Vec<&str> = Vec::new();
    for mark in ["\nThey said: ", "\nYou noticed: "] {
        if let Some(before) = tail.split(mark).next() {
            if tail.contains(mark) {
                cuts.push(before);
            }
        }
    }
    let Some(region) = cuts.into_iter().min_by_key(|c| c.len()) else {
        return Err("no 'They said:' or 'You noticed:' after the widget list — the region between \
                    them cannot be bounded".into());
    };
    Ok(region.lines()
        .map(|x| x.trim_end().to_string())
        .filter(|x| !x.trim().is_empty()
                && !x.split_once(": ").is_some_and(|(n, _)| n.chars().all(|c| c.is_ascii_digit())
                                                   && !n.is_empty())
                && !x.contains(NO_WIDGETS))
        .collect())
}

/// The verdicts of each group, **worst last** — and which group borrows which vocabulary **is the
/// pre-registration**.
///
/// `REFUSE` is scored by `MEMORY`'s rule (nothing should happen), `FARACCEPT` by `ACCEPT`'s (the
/// offer again), `RESTATE` by `ANEW`'s (a named target). A group with its own private vocabulary
/// would be a group whose failures cannot be compared with anything. Unlike the distance sweep,
/// **`void` is inside the ordering here**, as the worst outcome rather than a dropped one.
pub fn worst_order(kind: &str) -> &'static [&'static str] {
    match kind {
        MEMORY | REFUSE => &[QUIET, OTHER, AGAIN, VOID],
        ACCEPT | FARACCEPT => &[ACTED, DRIFTED, IGNORED, VOID],
        RESTATE | ANEW => &[LANDED, EXTRA, SILENT, WRONG, STUCK, VOID],
        _ => panic!("{kind:?} is not one of the six groups"),
    }
}

/// The one verdict per group that counts as correct.
pub fn good(kind: &str) -> &'static str {
    match kind {
        MEMORY | REFUSE => QUIET,
        ACCEPT | FARACCEPT => ACTED,
        RESTATE | ANEW => LANDED,
        _ => panic!("{kind:?} is not one of the six groups"),
    }
}

/// One offered line with every quoted name replaced by `<x>` — **the wording, without the widget.**
///
/// This is the evidence that two arms differ *in the transcript*. If both print the same shape, the
/// `offered:` lines got no treatment at all — and a check that compared the lines themselves would
/// call a widget change a treatment.
pub fn shape_of(line: &str) -> String {
    let mut out = String::new();
    let mut inside = false;
    for c in line.chars() {
        if c == '"' {
            if !inside {
                out.push_str("<x>");
            }
            inside = !inside;
            continue;
        }
        if !inside {
            out.push(c);
        }
    }
    //: An unterminated quote leaves the tail swallowed, which is what the regex does too: it only
    //: replaces balanced pairs, so a lone quote is not a pair and its text is not a name.
    if inside {
        return line.trim().to_string();
    }
    out.trim().to_string()
}

pub const NO_SLOT: &str = "(nothing between the widget list and the turn)";

/// One turn's slot region as a single key: the wording, with the widget names taken out.
pub fn slot_shape(lines: &[String]) -> String {
    if lines.is_empty() {
        return NO_SLOT.to_string();
    }
    lines.iter().map(|x| shape_of(x)).collect::<Vec<_>>().join(" | ")
}

/// Does the reply name the thing that was offered? **A display, not a verdict.**
///
/// The recorded observation is that the agent *answers correctly* and acts anyway, so a run that
/// only counted actions would lose half of what was seen. **It undercounts by construction**:
/// widget names are English and the replies are mostly Korean, so *"복사 버튼을 클릭하려고
/// 했어요"* is a correct account of having offered `Copy` and contains no `copy`. A floor on
/// narration, never a rate — which is why nothing scores on it.
pub fn narrated(first_effects: &[serde_json::Value], reply: &str) -> bool {
    let say = reply.to_lowercase();
    first_effects.iter().any(|e| {
        let t = e["target"].as_str().unwrap_or("");
        let n = if t.is_empty() { e["text"].as_str().unwrap_or("") } else { t };
        !n.is_empty() && say.contains(&n.to_lowercase())
    })
}
