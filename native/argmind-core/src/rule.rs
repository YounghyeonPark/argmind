//! **Where argmind may not act, and where it must always ask** — rules a person writes, per app and
//! per kind of action.
//!
//! From the dots comparison: an agent that acts on a desktop needs a way to say *not in my bank*
//! and *never type in the messenger* that does not depend on the model remembering. Every gate
//! here before this was about **when** (asked or not, confirmed or not); this is about **where**,
//! and like every gate here it is **the program's, never shown to the model** — a rule the model
//! is told is a rule it can break while looking at it (`rule-following` holds 36 of 36 at distance
//! ≤ 8, and *a standing rule is broken while it is visible in the prompt* is in the same file).
//!
//! **An app is matched by its window title**, case-insensitive substring: *카카오톡*, *Chrome*,
//! *국민은행*. Weak by construction — a browser's title is the page's — and stated as such: the
//! title is what the contract carries (`Widget::window`), and a process name would be a contract
//! change argued from one implementation.
//!
//! **At rest there are none**, and with none every answer is `None`: today's behaviour exactly.

use serde_json::{json, Value};

/// What a rule does when it applies.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Then {
    /// Offer it, but **never carry it out without the button**, whatever 확인 없이 실행 says.
    Confirm,
    /// Do not offer it at all. Ordered after `Confirm` so the stricter one wins.
    Never,
}

impl Then {
    pub fn word(self) -> &'static str {
        match self {
            Then::Confirm => "confirm",
            Then::Never => "never",
        }
    }
    pub fn from_word(w: &str) -> Option<Then> {
        match w {
            "confirm" => Some(Then::Confirm),
            "never" => Some(Then::Never),
            _ => None,
        }
    }
}

/// The kinds a rule can name. `all` is every one of them.
pub const ACTIONS: [&str; 4] = ["all", "click", "type", "key"];

/// The most rules kept. Chosen: a list longer than a screen is a list nobody checks.
pub const MOST_RULES: usize = 30;

#[derive(Clone, Debug, PartialEq)]
pub struct Rule {
    /// Part of a window title. **Empty means everywhere.**
    pub app: String,
    /// One of [`ACTIONS`].
    pub action: String,
    pub then: Then,
}

impl Rule {
    pub fn covers(&self, window: &str, kind: &str) -> bool {
        let app = self.app.trim().to_lowercase();
        (app.is_empty() || window.to_lowercase().contains(&app))
            && (self.action == "all" || self.action == kind)
    }
}

/// **The strictest rule that applies, or `None`.** `Never` beats `Confirm` whichever was written
/// first, so adding a looser rule can never undo a stricter one.
pub fn applies(rules: &[Rule], window: &str, kind: &str) -> Option<Then> {
    rules.iter().filter(|r| r.covers(window, kind)).map(|r| r.then).max()
}

/// The sentence a person reads when a rule stopped something. **The program's words.**
pub fn told(rules: &[Rule], window: &str, kind: &str) -> String {
    let what = match kind {
        "click" => "누르지",
        "type" => "입력하지",
        "key" => "키를 누르지",
        _ => "하지",
    };
    let r = rules.iter().filter(|r| r.covers(window, kind)).max_by_key(|r| r.then);
    match r {
        Some(r) if r.app.trim().is_empty() => format!("어디서든 {what} 않도록 정해 두셨습니다"),
        Some(r) => format!("「{}」에서는 {what} 않도록 정해 두셨습니다", r.app.trim()),
        None => String::new(),
    }
}

/// **The page sends the whole list, one rule per line, `app<TAB>action<TAB>then`.** A line that
/// does not read is an error rather than skipped: a rule somebody wrote and the program dropped
/// without saying is the one failure a list of safeguards must not have.
pub fn from_lines(text: &str) -> Result<Vec<Rule>, String> {
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split('\t').collect();
        let [app, action, then] = parts[..] else {
            return Err(format!("{}번째 규칙을 읽지 못했습니다", i + 1));
        };
        let action = action.trim();
        if !ACTIONS.contains(&action) {
            return Err(format!("{}번째 규칙의 동작 {action:?} 은 없는 것입니다", i + 1));
        }
        let Some(then) = Then::from_word(then.trim()) else {
            return Err(format!("{}번째 규칙의 처리 {then:?} 은 없는 것입니다", i + 1));
        };
        let app: String = app.trim().chars().take(80).collect();
        out.push(Rule { app, action: action.into(), then });
    }
    if out.len() > MOST_RULES {
        return Err(format!("규칙은 {MOST_RULES}개까지입니다"));
    }
    Ok(out)
}

pub fn render_all(rules: &[Rule]) -> String {
    serde_json::to_string_pretty(&json!({ "rules": rows(rules) })).unwrap_or_else(|_| "{}".into())
        + "\n"
}

/// For the page, on one line.
pub fn for_the_page(rules: &[Rule]) -> String {
    rows(rules).to_string()
}

fn rows(rules: &[Rule]) -> Value {
    Value::Array(rules.iter().map(|r| json!({
        "app": r.app, "action": r.action, "then": r.then.word(),
    })).collect())
}

/// **A row that does not read is dropped, not the file** — but unlike [`from_lines`] this is the
/// file on disk, which a person may have edited by hand, and losing every rule to one typo would be
/// the worse failure.
pub fn parse_all(text: &str) -> Vec<Rule> {
    let Ok(v) = serde_json::from_str::<Value>(text) else { return Vec::new() };
    let Some(rows) = v.get("rules").and_then(Value::as_array) else { return Vec::new() };
    rows.iter().filter_map(|r| {
        let s = |k: &str| r.get(k).and_then(Value::as_str).unwrap_or("").to_string();
        let action = s("action");
        if !ACTIONS.contains(&action.as_str()) {
            return None;
        }
        Some(Rule { app: s("app"), action, then: Then::from_word(&s("then"))? })
    }).take(MOST_RULES).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rule(app: &str, action: &str, then: Then) -> Rule {
        Rule { app: app.into(), action: action.into(), then }
    }

    #[test]
    fn no_rules_is_today() {
        assert_eq!(applies(&[], "아무 창", "click"), None);
    }

    #[test]
    fn an_app_matches_by_part_of_the_title_ignoring_case() {
        let rules = [rule("kakaotalk", "type", Then::Never)];
        assert_eq!(applies(&rules, "KakaoTalk - 친구", "type"), Some(Then::Never));
        assert_eq!(applies(&rules, "KakaoTalk - 친구", "click"), None, "only the action named");
        assert_eq!(applies(&rules, "메모장", "type"), None, "only the app named");
    }

    #[test]
    fn the_stricter_rule_wins_in_either_order() {
        let a = [rule("", "all", Then::Confirm), rule("은행", "all", Then::Never)];
        let b = [rule("은행", "all", Then::Never), rule("", "all", Then::Confirm)];
        for rules in [&a[..], &b[..]] {
            assert_eq!(applies(rules, "국민은행 - Chrome", "click"), Some(Then::Never));
            assert_eq!(applies(rules, "메모장", "click"), Some(Then::Confirm));
        }
    }

    #[test]
    fn the_sentence_names_the_app_or_says_everywhere() {
        let rules = [rule("카카오톡", "type", Then::Never)];
        assert_eq!(told(&rules, "카카오톡", "type"), "「카카오톡」에서는 입력하지 않도록 정해 두셨습니다");
        let rules = [rule("", "key", Then::Never)];
        assert_eq!(told(&rules, "아무", "key"), "어디서든 키를 누르지 않도록 정해 두셨습니다");
    }

    #[test]
    fn lines_from_the_page_read_or_say_why_not() {
        let r = from_lines("카카오톡\ttype\tnever\n\n\tall\tconfirm\n").unwrap();
        assert_eq!(r, vec![rule("카카오톡", "type", Then::Never), rule("", "all", Then::Confirm)]);
        assert!(from_lines("카카오톡\tsing\tnever").is_err(), "an unknown action is not skipped");
        assert!(from_lines("카카오톡\ttype\tmaybe").is_err());
        assert!(from_lines("카카오톡 type never").is_err());
        assert!(from_lines(&"x\tall\tnever\n".repeat(MOST_RULES + 1)).is_err());
    }

    #[test]
    fn the_file_round_trips_and_a_bad_row_loses_only_itself() {
        let rules = vec![rule("은행", "all", Then::Never), rule("", "key", Then::Confirm)];
        assert_eq!(parse_all(&render_all(&rules)), rules);
        let mixed = r#"{"rules": [{"app": "a", "action": "click", "then": "never"},
                                  {"app": "b", "action": "dance", "then": "never"}]}"#;
        assert_eq!(parse_all(mixed).len(), 1);
        assert!(parse_all("not json").is_empty());
        assert!(!for_the_page(&rules).contains('\n'));
    }
}
