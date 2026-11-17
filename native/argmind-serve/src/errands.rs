//! **맡긴 일** — work handed to argmind, thought about in the gaps, and reported when somebody is there.
//!
//! What it is and what it can do is `argmind_core::errand`, including the two runs that shaped it.
//! This file is the application's half: where the list lives, which one is stepped next, when a
//! finished one is said, and how the page is told.
//!
//! ## What the dots comparison asked for, and what of it is here
//!
//! A dot is handed work and brings it back, and **works while you are away**. Both halves are here
//! and both are narrow:
//!
//! * **It works whenever nobody is being answered** — away or not, because a step abandons the
//!   moment a turn begins (measured: dropping the socket frees the card, 1.09 s against 1.12 s).
//!   Every [`STEP_EVERY`] seconds, so an errand takes about half a minute, not an afternoon.
//! * **It reports only to somebody who is there.** A result that arrives while the camera says
//!   nobody has moved for five minutes, or while 눈치 says the head is turned, waits in the list
//!   with `told = false` and is said on the first tick that allows it. **No camera never means
//!   silence** — the same rule `somebody_about` keeps.
//!
//! **What it is not**: it reads no screen, opens nothing and presses nothing. Every answer shape
//! has no `say` and no `do`, and the step is given an empty screen on purpose — a person who hands
//! over *생신 선물 아이디어* has not agreed to have their open windows read into it.

use argmind_core::errand::{Errand, State};
use std::path::PathBuf;

/// **Seconds between steps.** Chosen: a step is one call, about 0.6 s warm, so this keeps the card
/// free four-fifths of the time while still finishing an errand in about thirty seconds.
pub const STEP_EVERY: f64 = 5.0;

/// **How many errands the list holds**, finished ones included. Chosen: a list longer than a
/// screen is a list nobody reads, and finished results are the person's to clear.
pub const MOST_KEPT: usize = 20;

/// How long the words of an errand may be. The same bound the answer route uses.
pub const MOST_CHARS: usize = 200;

/// `%LOCALAPPDATA%\argmind\errands.json`, beside the rest. **Readable and deletable**, like the
/// face roster: it holds what somebody asked for in their own words.
pub fn at() -> PathBuf {
    let mut path = std::env::var("LOCALAPPDATA").map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir());
    path.push("argmind");
    path.push("errands.json");
    path
}

/// What is on the list. **An unreadable file is an empty list**, and a mangled row loses only
/// itself — `parse_all`'s rule.
pub fn remembered() -> Vec<Errand> {
    std::fs::read_to_string(at()).map(|t| argmind_core::errand::parse_all(&t)).unwrap_or_default()
}

pub fn keep(errands: &[Errand]) -> Result<(), String> {
    let path = at();
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    std::fs::write(&path, argmind_core::errand::render_all(errands))
        .map_err(|e| format!("{path:?}: {e}"))
}

/// Put one on the list. **Refused rather than trimmed** when the list is full, so nothing the
/// person wrote disappears without them deciding it should.
pub fn add(errands: &mut Vec<Errand>, what: &str) -> Result<u64, &'static str> {
    let what = what.trim();
    if what.is_empty() {
        return Err("맡길 일을 적어 주세요");
    }
    if what.chars().count() > MOST_CHARS {
        return Err("너무 깁니다 — 200자 안으로 적어 주세요");
    }
    if errands.len() >= MOST_KEPT {
        return Err("맡긴 일이 스무 개입니다 — 끝난 것을 지우고 다시 맡겨 주세요");
    }
    let id = errands.iter().map(|e| e.id).max().unwrap_or(0) + 1;
    errands.push(Errand::new(id, what));
    Ok(id)
}

/// **Which one to step: the oldest still open.** One at a time and in order, because two half-done
/// errands are worse than one done and one waiting.
pub fn next_open(errands: &[Errand]) -> Option<&Errand> {
    errands.iter().find(|e| e.state.open())
}

/// **The oldest one with something to tell** — finished, or stopped on a question — that has not
/// been told yet.
pub fn due(errands: &[Errand]) -> Option<usize> {
    errands.iter().position(|e| !e.told && matches!(e.state, State::Done | State::NeedsYou))
}

/// **What is said, and the wording is the program's** around the model's result: the errand is
/// named in the person's own words, so a report arriving minutes later says what it is about.
pub fn report(e: &Errand) -> String {
    report_in(e, "ko", &mut |t: &str| t.to_string())
}

/// **[`report`], in the language the person is spoken to in** — since 2026-11-17. The program's
/// words come from `crate::spoken`; the errand is named in the person's own words, as they wrote
/// it; and the result or the question is the model's, Korean from a Korean prompt, so it goes
/// through `into` — which leaves it as written while `app::REWRITES_WORK` is off.
pub fn report_in(e: &Errand, code: &str, into: &mut dyn FnMut(&str) -> String) -> String {
    use crate::spoken::{say, ERRAND_ASKS, ERRAND_DONE};
    match e.state {
        State::Done => format!("{}\n{}", say(&ERRAND_DONE, code, &[&e.what]), into(&e.result)),
        State::NeedsYou => say(&ERRAND_ASKS, code, &[&e.what, &into(&e.ask)]),
        _ => String::new(),
    }
}

/// **Apply a step only if the errand is still what was stepped.** The thinker works on a copy; an
/// answer typed, or a delete pressed, while a step was in flight must not be overwritten by a
/// result computed from before it.
pub fn apply(errands: &mut [Errand], before: &Errand, after: Errand) -> bool {
    match errands.iter_mut().find(|e| e.id == before.id) {
        Some(e) if e == before => {
            *e = after;
            true
        }
        _ => false,
    }
}

/// **An answer to a question, or a correction to a result.** Answering a finished one starts it
/// again with the correction, because *이게 아니라 저거* is how a person replies to a draft.
pub fn answer(errands: &mut [Errand], id: u64, text: &str) -> bool {
    let Some(e) = errands.iter_mut().find(|e| e.id == id) else { return false };
    match e.state {
        State::Done => e.again(text),
        _ => e.answered(text),
    }
    true
}

pub fn drop(errands: &mut Vec<Errand>, id: u64) -> bool {
    let before = errands.len();
    errands.retain(|e| e.id != id);
    errands.len() != before
}

/// For the page. **Written by the core**, which already has a JSON writer, because this crate
/// writes its JSON by hand on purpose and a list of free text is where hand-written JSON breaks.
pub fn list_json(errands: &[Errand]) -> String {
    argmind_core::errand::for_the_page(errands)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adding_numbers_them_and_refuses_rather_than_trims() {
        let mut list = Vec::new();
        assert_eq!(add(&mut list, "  선물 아이디어  "), Ok(1));
        assert_eq!(add(&mut list, "운동 루틴"), Ok(2));
        assert_eq!(list[0].what, "선물 아이디어");
        assert!(add(&mut list, "   ").is_err());
        assert!(add(&mut list, &"가".repeat(MOST_CHARS + 1)).is_err());
        while list.len() < MOST_KEPT {
            add(&mut list, "또").unwrap();
        }
        assert!(add(&mut list, "하나 더").is_err(), "a full list refuses");
        assert_eq!(list.len(), MOST_KEPT);
    }

    #[test]
    fn a_stale_step_is_not_applied() {
        let mut list = Vec::new();
        add(&mut list, "선물").unwrap();
        let before = list[0].clone();
        let mut after = before.clone();
        after.steps = 1;
        after.state = State::Working;
        //: The person answered while the step was in flight.
        answer(&mut list, 1, "예산은 10만원");
        assert!(!apply(&mut list, &before, after.clone()), "the answer must survive");
        assert!(list[0].notes.iter().any(|n| n.contains("10만원")));
        //: And an untouched one takes the step.
        let before = list[0].clone();
        let mut after = before.clone();
        after.steps += 1;
        assert!(apply(&mut list, &before, after));
        //: Deleted mid-step: nothing to apply to.
        drop(&mut list, 1);
        assert!(!apply(&mut list, &before, before.clone()));
    }

    #[test]
    fn the_oldest_open_is_stepped_and_the_oldest_untold_is_said() {
        let mut list = Vec::new();
        for w in ["하나", "둘", "셋"] {
            add(&mut list, w).unwrap();
        }
        list[0].state = State::Done;
        list[0].result = "결과".into();
        list[1].state = State::NeedsYou;
        list[1].ask = "물음".into();
        assert_eq!(next_open(&list).map(|e| e.id), Some(3));
        assert_eq!(due(&list), Some(0));
        assert!(report(&list[0]).contains("「하나」") && report(&list[0]).contains("결과"));
        list[0].told = true;
        assert_eq!(due(&list), Some(1));
        assert!(report(&list[1]).ends_with("물음"));
    }

    #[test]
    fn answering_a_finished_one_starts_it_again() {
        let mut list = Vec::new();
        add(&mut list, "선물").unwrap();
        list[0].state = State::Done;
        list[0].result = "앨범".into();
        list[0].told = true;
        assert!(answer(&mut list, 1, "앨범 말고 먹을 것으로"));
        assert!(list[0].state.open() && list[0].result.is_empty() && !list[0].told);
        assert!(!answer(&mut list, 9, "없는 것"));
    }

    #[test]
    fn the_page_gets_one_line_that_parses() {
        let mut list = Vec::new();
        add(&mut list, "따옴표 \"와\" 줄\n바꿈").unwrap();
        list[0].notes.push("생각\n하나".into());
        let s = list_json(&list);
        assert!(!s.contains('\n'), "an event's data ends at a newline");
        let v: serde_json::Value = serde_json::from_str(&s).unwrap();
        assert_eq!(v[0]["lately"], "생각\n하나");
    }
}
