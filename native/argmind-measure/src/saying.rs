//! Is a reply something you would say to a person?
//!
//! **The defect this exists for**: internal widget indices leak into the speech — *"3, 122"*,
//! *"지우개는 22번 버튼입니다"*. The number is real inside the prompt and meaningless to a
//! listener, and it becomes audible the moment there is a voice: 3.05 s of somebody reading out an
//! index.
//!
//! Two verdicts on every reply, because they fail independently. **The leak** is about numbers in
//! the sentence; **the answer** is about whether there is a sentence at all. A reply can be clean
//! and empty, or a full answer with an index in it.
//!
//! The hard part is not finding digits, it is **not counting the legitimate ones**. `Alt+F4`,
//! `U+0021`, `i9-13900HX` and *"버튼이 3개 있어요"* all contain digits and only the last is
//! suspicious — so a number that appears in a listed widget's name is `quoted` and not a leak, and
//! there is a control set of goals whose right answers contain numbers.

use crate::Tables;

pub const CLEAN: &str = "clean";
pub const QUOTED: &str = "quoted";
pub const OFFLIST: &str = "off-list";
pub const STRAY: &str = "stray";
pub const INDEX: &str = "index";
/// Worst last. `quoted` is *better* than a leak because the number came from the screen.
pub const LEAK_ORDER: [&str; 5] = [CLEAN, QUOTED, OFFLIST, STRAY, INDEX];

pub const ANSWERED: &str = "answered";
pub const BARE_NAME: &str = "bare-name";
pub const BARE_ACK: &str = "bare-ack";
pub const ECHO: &str = "echo";
pub const NUMERIC: &str = "numeric";
pub const EMPTY: &str = "empty";
pub const ANSWER_ORDER: [&str; 6] = [ANSWERED, BARE_NAME, BARE_ACK, ECHO, NUMERIC, EMPTY];

/// One number found in a reply, and everything known about why it might be there.
#[derive(Debug, Clone)]
pub struct Found {
    pub run: String,
    pub token: String,
    pub kind: &'static str,
    pub in_range: bool,
    pub clicked: bool,
    pub context: String,
}

/// The whole token a run of digits sits inside — `13900` inside `i9-13900HX` is not a number in a
/// sentence, it is part of a name.
pub fn token_around(text: &[char], start: usize, end: usize, token_chars: &str) -> String {
    let (mut i, mut j) = (start, end);
    while i > 0 && token_chars.contains(text[i - 1]) {
        i -= 1;
    }
    while j < text.len() && token_chars.contains(text[j]) {
        j += 1;
    }
    text[i..j].iter().collect()
}

/// Does this token appear in a listed name **as a whole token**, rather than as a substring?
///
/// The boundary check is what keeps `2` from matching inside `U+0021` and calling a leak a
/// quotation.
pub fn on_screen_token(token: &str, names: &[String], token_chars: &str) -> bool {
    for name in names {
        let chars: Vec<char> = name.chars().collect();
        let needle: Vec<char> = token.chars().collect();
        if needle.is_empty() || needle.len() > chars.len() {
            continue;
        }
        for i in 0..=(chars.len() - needle.len()) {
            if chars[i..i + needle.len()] != needle[..] {
                continue;
            }
            let before = if i > 0 { Some(chars[i - 1]) } else { None };
            let after = chars.get(i + needle.len()).copied();
            let bounded = before.is_none_or(|c| !token_chars.contains(c))
                && after.is_none_or(|c| !token_chars.contains(c));
            if bounded {
                return true;
            }
        }
    }
    false
}

/// Every number in a reply, classified, and the worst of them.
///
/// * `quoted` — the token appears on screen. Not a leak: it came from the interface.
/// * `index` — a number in range **and it is one of the things just clicked**. The defect itself.
/// * `stray` — a number in range that was not clicked. Suspicious, and not provably the index.
/// * `off-list` — a number out of range. Wrong, but not this defect.
pub fn classify(say: &str, names: &[String], clicked_at: &[usize], tables: &Tables)
    -> (&'static str, Vec<Found>) {
    let chars: Vec<char> = say.chars().collect();
    let mut found = Vec::new();
    let mut worst = CLEAN;
    let mut i = 0;
    while i < chars.len() {
        if !chars[i].is_ascii_digit() {
            i += 1;
            continue;
        }
        let start = i;
        while i < chars.len() && chars[i].is_ascii_digit() {
            i += 1;
        }
        let run: String = chars[start..i].iter().collect();
        let token = token_around(&chars, start, i, &tables.token_chars);
        let value: usize = run.parse().unwrap_or(usize::MAX);
        let in_range = value < names.len();
        let on_screen = token.chars().count() > 1
            && on_screen_token(&token, names, &tables.token_chars);
        let kind = if on_screen {
            QUOTED
        } else if in_range && clicked_at.contains(&value) {
            INDEX
        } else if in_range {
            STRAY
        } else {
            OFFLIST
        };
        let from = start.saturating_sub(10);
        let to = (i + 10).min(chars.len());
        found.push(Found { run, token, kind, in_range, clicked: clicked_at.contains(&value),
                           context: chars[from..to].iter().collect() });
        if rank(kind, &LEAK_ORDER) > rank(worst, &LEAK_ORDER) {
            worst = kind;
        }
    }
    (worst, found)
}

pub fn strip_punct(text: &str, punct: &str) -> String {
    text.chars().filter(|c| !punct.contains(*c) && !c.is_whitespace()).collect()
}

/// Is one of these two strings the other, or most of it?
pub fn is_echo(a: &str, b: &str, share: f64) -> bool {
    if a.is_empty() || b.is_empty() {
        return false;
    }
    if a == b {
        return true;
    }
    let (la, lb) = (a.chars().count() as f64, b.chars().count() as f64);
    (a.contains(b) || b.contains(a)) && la.min(lb) / la.max(lb) >= share
}

/// Is there a sentence here at all?
///
/// **A separate axis from the leak**, because they fail independently: repeating the question back
/// is not a leak, and neither is answering with a bare *"네"*. Both are replies a person would
/// notice and neither shows up in a count of numbers.
pub fn answer_verdict(say: &str, asked: &str, names: &[String], tables: &Tables) -> &'static str {
    let bare = strip_punct(say, &tables.punct);
    if bare.is_empty() {
        return EMPTY;
    }
    if !bare.is_empty() && bare.chars().all(|c| c.is_ascii_digit()) {
        return NUMERIC;
    }
    if is_echo(&strip_punct(asked, &tables.punct), &bare, 0.6) {
        return ECHO;
    }
    if tables.acks.iter().any(|a| a.to_lowercase() == bare.to_lowercase()) {
        return BARE_ACK;
    }
    if names.iter().any(|n| n.chars().count() > 1
                        && strip_punct(n, &tables.punct).to_lowercase() == bare.to_lowercase()) {
        return BARE_NAME;
    }
    ANSWERED
}

/// Does the reply carry a number at all — as a digit, or as a Korean word?
///
/// The word list exists because the control set asks *"버튼이 몇 개야?"*, and a model answering
/// *"다섯 개"* has answered with a number without using one.
pub fn has_number(say: &str, words: &[String]) -> (bool, String) {
    if say.chars().any(|c| c.is_ascii_digit()) {
        return (true, "digit".into());
    }
    let low = say.to_lowercase();
    for w in words {
        if low.contains(w.as_str()) {
            return (true, format!("word:{w}"));
        }
    }
    (false, String::new())
}

pub fn rank(verdict: &str, order: &[&str]) -> usize {
    order.iter().position(|o| *o == verdict).unwrap_or(order.len())
}

/// The index of the worst draw, so the transcript printed beside a verdict is the one it describes.
///
/// Printing repeat one's reply next to a worst-of-three verdict has happened twice in this project,
/// the second time hours after the first was criticised, and it makes a correct answer display as a
/// defect.
///
/// **The tie-break is load-bearing and it is the FIRST maximum.** Rust's `max_by_key` returns the
/// last, Python's `max(range(n), key=...)` returns the first, and every repeat agreeing is the
/// common case — so the two disagreed on the stored index of every row where nothing went wrong.
/// No verdict moves either way, which is why it survived: the tests compared
/// `verdicts[worst_index(..)]` and never the index itself. Found by scoring a fresh Rust run with
/// the Python judge and diffing the fields one at a time.
pub fn worst_index(verdicts: &[&str], order: &[&str]) -> usize {
    let mut best = 0;
    for i in 1..verdicts.len() {
        if rank(verdicts[i], order) > rank(verdicts[best], order) {
            best = i;
        }
    }
    best
}
