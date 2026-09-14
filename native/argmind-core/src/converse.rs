//! One call: what to say, and what if anything to do.
//!
//! The prompt this renders is fingerprinted by `measure/choosing.py` as `r322a4c` — a hash of the
//! rendered text over six probes. **A renderer that produces that tag has provably not changed a
//! character of what the model is sent**, which matters because prompt edits in this project have
//! been measured as never free: five in a row each raised one number and lowered another.

use crate::contract::Widget;
use crate::intent::{Effect, Intent, Trigger};
use crate::llm::first_object;

pub const MAX_LISTED: usize = 200;
/// **Derived, not chosen.** 27 of 28 real requests want one effect or two; spurious ones come in
/// threes, fours and an eight. Three is the documented sequence — click, type, enter.
pub const MOST_AT_ONCE: usize = 3;

pub const SYSTEM: &str = r#"You are a helper the person talks to while they work at their PC.

You are shown the things currently on their screen, each with a number. Reply as a person would —
briefly, in the language they used.

If they asked you to do something and the listed things can do it, name the numbers. If they were
just talking, or the thing they want is not listed, say so and do nothing: a wrong click is worse
than an honest "I can't see that here"."#;

pub const UNPROMPTED: &str = "Nobody asked you anything — you noticed this yourself. Mentioning it is usually\nenough. Only propose doing something if it is clearly wanted, and expect to be asked first.";

pub const STILL_OPEN: &str = "Still open, from earlier:";
pub const STILL_OPEN_TAIL: &str =
    "What they say next may be a yes to that, a no, or nothing to do with it.";

pub const OFFER_OPEN: &str = "offered";
pub const OFFER_SETTLED: [&str; 3] = ["did", "could not", "they said no to"];

/// The list as the model sees it. Names pass through untouched — not lowercased, not translated:
/// the model matched an English request to the Korean widget `닫기` without help, which is the case
/// a keyword matcher fails and the reason a model is here at all.
pub fn describe_screen(widgets: &[Widget], limit: usize) -> String {
    widgets
        .iter()
        .take(limit)
        .enumerate()
        .map(|(i, w)| format!("{}: {} \"{}\"", i, w.role, w.name))
        .collect::<Vec<_>>()
        .join("\n")
}

/// The offers nobody has answered yet, oldest first, in the words the history holds them in.
/// Closes on **any** settling label rather than matching the effect — matching would mean parsing
/// the history's prose back into an effect.
pub fn still_open(history: &[(String, String)]) -> Vec<String> {
    let mut open: Vec<String> = Vec::new();
    for (who, text) in history {
        if who == OFFER_OPEN {
            open.push(text.clone());
        } else if OFFER_SETTLED.contains(&who.as_str()) {
            open.clear();
        }
    }
    open
}

/// The prompt, exactly. Verified against `r322a4c`.
pub fn render(widgets: &[Widget], trigger: &Trigger, history: &[(String, String)],
              limit: usize) -> String {
    let listed: Vec<&Widget> = widgets.iter().take(limit).collect();
    let mut lines: Vec<String> = vec![SYSTEM.to_string()];
    if !trigger.asked {
        lines.push(UNPROMPTED.to_string());
    }
    lines.push(String::new());
    if !history.is_empty() {
        lines.push("Recently:".to_string());
        for (who, text) in history {
            lines.push(format!("  {who}: {text}"));
        }
        lines.push(String::new());
    }
    lines.push("On screen now:".to_string());
    lines.push(if listed.is_empty() {
        "  (nothing named — this surface exposes no widgets)".to_string()
    } else {
        let owned: Vec<Widget> = listed.iter().map(|w| (*w).clone()).collect();
        describe_screen(&owned, limit)
    });

    //: **A slot fed by conversation state, not a sentence.** Present only when an offer is open,
    //: recomputed every turn, and its tail names all three readings on purpose — a line leaning
    //: one way is what slid the five prompt edits before it.
    let pending = if trigger.asked { still_open(history) } else { Vec::new() };
    if !pending.is_empty() {
        lines.push(String::new());
        lines.push(STILL_OPEN.to_string());
        for text in &pending {
            lines.push(format!("  {text}"));
        }
        lines.push(STILL_OPEN_TAIL.to_string());
    }

    lines.push(String::new());
    lines.push(if trigger.asked {
        format!("They said: {}", trigger.text)
    } else {
        format!("You noticed: {}", trigger.text)
    });
    lines.push(String::new());
    lines.push(
        r#"Reply with only JSON: {"say": "<what you say back>", "do": [ ... ]} where each entry of"#
            .to_string());
    lines.push(
        r#""do" is {"click": <number>}, {"type": "<text to type>"} or {"key": "<enter|tab|escape>"}."#
            .to_string());
    lines.push(r#"Leave "do" empty if nothing should happen."#.to_string());
    lines.join("\n")
}

/// One entry of `do`, validated against what is actually on screen.
///
/// **The number becomes a name and a role here and never travels further as an index.** That one
/// line is the staleness fix; the role is half of the key, and leaving it out was a real defect —
/// a window with a label and a combo box both called `Font :` resolved to the label and the click
/// did nothing.
pub fn effect_of(item: &serde_json::Value, listed: &[Widget], trigger: &Trigger) -> Option<Effect> {
    let obj = item.as_object()?;
    let because = Some(trigger.asked);
    if let Some(v) = obj.get("click") {
        let index = match v {
            serde_json::Value::Number(n) => n.as_i64()?,
            serde_json::Value::String(s) => s.parse::<i64>().ok()?,
            _ => return None,
        };
        if index < 0 || index as usize >= listed.len() {
            return None;
        }
        let w = &listed[index as usize];
        return Some(Effect { kind: "click".into(), target: w.name.clone(), role: w.role.clone(),
                             text: String::new(), because_asked: because });
    }
    if let Some(v) = obj.get("type") {
        let s = v.as_str()?;
        if s.is_empty() {
            return None;
        }
        return Some(Effect { kind: "type".into(), target: String::new(), role: String::new(),
                             text: s.to_string(), because_asked: because });
    }
    if let Some(v) = obj.get("key") {
        let s = v.as_str()?;
        return Some(Effect { kind: "key".into(), target: String::new(), role: String::new(),
                             text: s.to_lowercase(), because_asked: because });
    }
    None
}

/// What a reply means, given what was on screen. An empty `Intent` is a real answer.
pub fn read_reply(reply: &str, listed: &[Widget], trigger: &Trigger) -> Intent {
    let got = first_object(reply).unwrap_or(serde_json::json!({}));
    let say = got.get("say").and_then(|v| v.as_str()).unwrap_or("").trim().to_string();
    let wanted = got.get("do").and_then(|v| v.as_array()).cloned().unwrap_or_default();
    if wanted.len() > MOST_AT_ONCE {
        //: The whole batch goes, not the first three. A reply asking for eight things has
        //: misunderstood, and doing three of them is worse than saying so.
        return Intent { say, effects: Vec::new() };
    }
    let effects = wanted.iter().filter_map(|item| effect_of(item, listed, trigger)).collect();
    Intent { say, effects }
}
