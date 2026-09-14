//! Repair, memory and language — **can you talk to it?**
//!
//! The owner's priority since 2026-09-10 is interaction and communication, not accuracy: *a small
//! drop in choosing is acceptable; a turn that cannot be repaired is not.* This is the first thing
//! in the project that measured whether you can.
//!
//! Three questions, and the third is here because of a trap. **Korean in, Korean out** looks like
//! a fair measure until you notice that a model which always answers in Korean scores perfectly —
//! so the English control is not decoration, it is what makes the row mean anything.

/// The scripts a reply can come back in. `other` is a script that is neither; `neither` is a reply
/// with no letters at all, which is a different fact from answering in the wrong language.
pub const KOREAN_OUT: &str = "korean";
pub const ENGLISH_OUT: &str = "english";
pub const OTHER: &str = "other";
pub const NEITHER: &str = "neither";

/// **Worst last, and the order depends on what was asked in.** Answering Korean in English is the
/// failure when the person spoke Korean; the same verdict is the success when they spoke English.
pub fn language_worst(asked: &str) -> [&'static str; 4] {
    if asked == "korean" {
        [KOREAN_OUT, ENGLISH_OUT, OTHER, NEITHER]
    } else {
        [ENGLISH_OUT, KOREAN_OUT, OTHER, NEITHER]
    }
}

pub const LANDED: &str = "landed";
pub const EXTRA: &str = "extra";
pub const WRONG: &str = "wrong";

/// Worst last. **`EXTRA` exists because "it clicked the right thing" and "it clicked the right
/// thing and three others" are not the same outcome**, and a name-only verdict cannot tell them
/// apart.
pub const REPAIR_WORST: [&str; 3] = [LANDED, EXTRA, WRONG];

/// What script a reply is in, and whether it also carries Latin letters.
///
/// Hangul decides first: a Korean sentence with an English widget name in it is Korean, and
/// scoring it as mixed would fail the model for quoting the interface correctly. `mixed` is
/// recorded beside the verdict rather than folded into it.
pub fn script_of(text: &str) -> (&'static str, bool) {
    let hangul = text.chars().any(|c| {
        ('가'..='힣').contains(&c) || ('\u{1100}'..='\u{11FF}').contains(&c)
            || ('\u{3130}'..='\u{318F}').contains(&c)
    });
    let latin = text.chars().any(|c| c.is_ascii() && c.is_alphabetic());
    if hangul {
        return (KOREAN_OUT, latin);
    }
    if latin {
        return (ENGLISH_OUT, false);
    }
    (if text.chars().any(|c| c.is_alphabetic()) { OTHER } else { NEITHER }, false)
}

/// Did the correction land? `want` is the (name, role) that should have been clicked, or **nothing
/// at all** — which is a real answer and the one an always-clicking agent fails.
///
/// **The role is half of the key here too.** One case is the `Font :` label sitting in front of the
/// `combo box` of the same name: a name-only verdict cannot tell a click that would work from one
/// that would do nothing.
pub fn repair_verdict(clicks: &[(String, String)], want: Option<&(String, String)>) -> &'static str {
    match want {
        None => if clicks.is_empty() { LANDED } else { WRONG },
        Some(w) => {
            if !clicks.iter().any(|c| c == w) {
                WRONG
            } else if clicks.len() == 1 {
                LANDED
            } else {
                EXTRA
            }
        }
    }
}

/// The worst of some verdicts, given an ordering with the worst last.
pub fn worst<'a>(verdicts: &[&'a str], order: &[&'a str]) -> &'a str {
    let mut best = order[0];
    let mut rank = 0;
    for v in verdicts {
        let here = order.iter().position(|o| o == v).unwrap_or(order.len() - 1);
        if here >= rank {
            rank = here;
            best = order[here];
        }
    }
    best
}

/// **The floor a repair could reach, before it is attempted.**
///
/// The exact paired sign test over `bad` discordant pairs is `2 x 0.5^k`, so a set with fewer than
/// six failures cannot show a repair as significant however good the repair is. Printed for every
/// group because *a control that cannot move is as much a design fact as a hypothesis that cannot.*
pub fn sensitivity(bad: usize) -> f64 {
    if bad == 0 {
        return 1.0;
    }
    (2.0 * 0.5f64.powi(bad as i32)).min(1.0)
}
