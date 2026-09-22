//! What the person has already answered, and what that means next time.
//!
//! **Four answers, not two.** *Yes* and *no* leave a program asking the same question every time it
//! starts, which is how people learn to press the first button without reading it — the exact
//! failure the approval text was written to prevent. So: **허용 / 항상 허용 / 거부 / 항상 거부**, and
//! the two standing ones are written down.
//!
//! ## What may be answered *always*, and what may not
//!
//! **Not a click and not a keystroke.** Those keep their button every single time, and the rule
//! that there is exactly one place in this program that performs anything is held by a test. A
//! standing *yes* to clicking would be a standing yes to whatever the model decides next, which is
//! a different program from this one.
//!
//! What can be answered once and for all is the other kind: **changing the machine** — installing,
//! building, fetching, elevating — and **switching something on that then keeps running**: watching
//! the screen, and opening the microphone. Those are decisions about a capability rather than about
//! an act, and re-asking them every morning is noise.
//!
//! ## Where it is kept, and why not in the browser
//!
//! `%LOCALAPPDATA%\argmind\decisions.json`, beside the models and the build. The page is not the
//! program: a decision kept in `localStorage` would not exist for the command line, would not
//! survive clearing site data, and could not be read by the person who wants to know what they have
//! agreed to. This one is a small file they can open, read and delete.

use std::collections::BTreeMap;
use std::path::PathBuf;

/// What a person just pressed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Answer {
    /// This time.
    Allow,
    /// This time and from now on.
    AlwaysAllow,
    /// Not this time.
    Deny,
    /// Not this time and not again — until it is taken back in the settings.
    AlwaysDeny,
}

impl Answer {
    pub fn from_word(word: &str) -> Option<Answer> {
        match word {
            "allow" => Some(Answer::Allow),
            "always" => Some(Answer::AlwaysAllow),
            "deny" => Some(Answer::Deny),
            "never" => Some(Answer::AlwaysDeny),
            _ => None,
        }
    }

    /// Does this let the thing happen now?
    pub fn permits(self) -> bool {
        matches!(self, Answer::Allow | Answer::AlwaysAllow)
    }

    /// What it leaves behind, if anything. **A one-off leaves nothing** — that is the whole
    /// difference between the two pairs.
    pub fn standing(self) -> Option<Standing> {
        match self {
            Answer::AlwaysAllow => Some(Standing::Allowed),
            Answer::AlwaysDeny => Some(Standing::Denied),
            _ => None,
        }
    }
}

/// A decision that outlives the moment it was made.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Standing {
    /// Do it without asking.
    Allowed,
    /// Do not do it, and do not ask.
    Denied,
}

impl Standing {
    fn word(self) -> &'static str {
        match self {
            Standing::Allowed => "allowed",
            Standing::Denied => "denied",
        }
    }

    fn from_word(word: &str) -> Option<Standing> {
        match word {
            "allowed" => Some(Standing::Allowed),
            "denied" => Some(Standing::Denied),
            _ => None,
        }
    }
}

/// The things that can be decided once and for all.
///
/// **A fixed list rather than any string the page sends.** A key is what a stored *yes* is attached
/// to, so a page that could invent one could invent a permission the program never asks about — and
/// a typo would silently create a second, always-unanswered copy of a real one.
pub const DECIDABLE: [&str; 11] = [
    //: The needs from `needs.rs`. `korean-voice` and `vulkan` are elevation, the rest change the
    //: machine.
    "ollama", "model", "whisper", "korean-voice", "vulkan", "speaker",
    //: Switched on, and then it keeps going: a model call every few seconds, a device that opens
    //: while a key is held, and a program that makes a noise in a room.
    "notice", "microphone", "aloud",
    //: **And the one that presses buttons.** Everything above this line changes what argmind may
    //: *know* or *say*; this changes what it may *do*, and it was deliberately not on the list
    //: until the owner asked for it on 2026-09-22. See [`what`] for what it does not cover.
    "acting",
    //: **And the one that leaves a light on.** The microphone opens while a key is held; this stays
    //: open for as long as it is allowed, which is the owner's choice of 2026-09-22 over blinking
    //: every half minute — a light that is on means a camera that is on, and a blinking one says
    //: *mostly off*, which would not be true.
    "camera",
];

/// What to call it in the window that asks.
pub fn what(key: &str) -> &'static str {
    match key {
        //: **Not "Ollama 설치".** Since 2026-09-21 saying yes to this key can mean *install it*
        //: or *start the one that is already here*, and a settings row that named only the first
        //: would be describing a standing answer by one of the two things it permits — the same
        //: shape of mislabelling as `speaker` against `aloud`, one row down. `whisper` and
        //: `speaker` were already neutral for the same reason.
        "ollama" => "Ollama",
        "model" => "모델 받기",
        "whisper" => "받아쓰기 서버",
        "korean-voice" => "한국어 음성 설치",
        "vulkan" => "Vulkan SDK 설치",
        "speaker" => "목소리 서버",
        //: **Not the same thing as `speaker`, which is permission to build the server.** This is
        //: whether argmind makes a sound in the room, and the two were conflated exactly once: the
        //: server was allowed and argmind still needed `--speak` on the command line, which is the
        //: shape of defect the owner has now pointed at twice.
        "aloud" => "소리 내어 말하기",
        "notice" => "화면 지켜보기",
        "microphone" => "마이크",
        //: **확인 without 없이 would read as the opposite.** The thing being allowed is the
        //: *absence* of the button, so the name has to carry that and not sound like a promise to
        //: keep asking.
        "acting" => "확인 없이 실행",
        //: **Not 카메라 사용**, which would read as *may it take a picture*. What is allowed is a
        //: camera that stays on; what leaves the module is never a picture.
        "camera" => "카메라로 사람 알아보기",
        _ => "알 수 없는 권한",
    }
}

pub fn is_decidable(key: &str) -> bool {
    DECIDABLE.contains(&key)
}

/// Where the file lives.
pub fn at() -> PathBuf {
    let mut path = std::env::var("LOCALAPPDATA").map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir());
    path.push("argmind");
    path.push("decisions.json");
    path
}

/// Read what is written down.
///
/// **A file that cannot be read is no decisions rather than an error.** The absence of the file is
/// the ordinary case on a first run, and a corrupt one should make the program ask again — which is
/// the safe direction — rather than refuse to start.
pub fn remembered() -> BTreeMap<String, Standing> {
    let Ok(text) = std::fs::read_to_string(at()) else {
        return BTreeMap::new();
    };
    parse(&text)
}

/// The two-word JSON this writes, and nothing else.
///
/// Hand-read for the same reason the licence record is: this crate takes no third-party
/// dependency, and the shape is `{"key": "allowed"}` with keys from a fixed list. **Anything that
/// is not one of those is dropped**, so a hand-edited file with a typo asks again rather than
/// carrying a permission nobody can name.
pub fn parse(text: &str) -> BTreeMap<String, Standing> {
    let mut out = BTreeMap::new();
    for line in text.lines() {
        let line = line.trim().trim_end_matches(',');
        let Some((left, right)) = line.split_once(':') else { continue };
        let key = left.trim().trim_matches('"');
        let value = right.trim().trim_matches('"');
        if let (true, Some(s)) = (is_decidable(key), Standing::from_word(value)) {
            out.insert(key.to_string(), s);
        }
    }
    out
}

pub fn render(all: &BTreeMap<String, Standing>) -> String {
    //: **Nothing decided is `{}`**, not a pair of braces with a blank line between them. This is a
    //: file somebody is meant to open and understand, and the tidy empty case is free.
    if all.is_empty() {
        return "{}\n".to_string();
    }
    let rows: Vec<String> = all.iter()
        .map(|(k, v)| format!("  \"{k}\": \"{}\"", v.word()))
        .collect();
    format!("{{\n{}\n}}\n", rows.join(",\n"))
}

/// **What is allowed without being asked.**
///
/// The owner, 2026-09-17: 지켜보기허용을 기본값으로 하고, 설정에서 토글하도록 하자.
///
/// Watching was off until asked for and the reason was a cost, not a risk: something on screen
/// changes 13 seconds out of 14, so it is a standing invitation to spend a model call every three
/// seconds for as long as it is on. **That is the owner's bill and the owner has decided to pay
/// it.** The cost is unchanged and is still what the settings row says out loud.
///
/// **A default is not an answer.** Nothing is written to `decisions.json` by being on this list, so
/// 거부 in the settings is still a decision that is recorded and still wins — and taking a key off
/// this list later turns it back into a question rather than into a silent no.
pub const ALLOWED_UNLESS_REFUSED: [&str; 1] = ["notice"];

/// What has been decided about this, if anything — **or the default, for the keys that have one.**
pub fn standing(key: &str) -> Option<Standing> {
    remembered().get(key).copied().or_else(|| {
        ALLOWED_UNLESS_REFUSED.contains(&key).then_some(Standing::Allowed)
    })
}

/// What is actually written down, with no default filled in.
///
/// **The settings panel needs this and `standing` would lie to it**: a row reading *항상 허용* for
/// something nobody ever answered, with a 잊어버려 button that removes nothing, is an interface
/// describing a decision that was never made.
pub fn written(key: &str) -> Option<Standing> {
    remembered().get(key).copied()
}

/// Write one decision down, or take it back with `None`.
///
/// **Read, change, write** rather than kept in memory: the file is small, it is the person's to
/// edit, and a program holding a stale copy of somebody's permissions is the wrong kind of
/// disagreement to have.
pub fn keep(key: &str, how: Option<Standing>) -> Result<(), String> {
    if !is_decidable(key) {
        return Err(format!("{key:?} 은(는) 기억해 둘 수 있는 권한이 아닙니다"));
    }
    let mut all = remembered();
    match how {
        Some(s) => {
            all.insert(key.to_string(), s);
        }
        None => {
            all.remove(key);
        }
    }
    let path = at();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("{e}"))?;
    }
    std::fs::write(&path, render(&all)).map_err(|e| format!("{e}"))
}

/// Every decidable thing and what is decided about it — **including the ones nobody has answered**,
/// because a settings panel that only lists what was said is a panel that cannot show what is
/// still being asked.
pub fn all() -> Vec<(&'static str, &'static str, Option<Standing>)> {
    let kept = remembered();
    DECIDABLE.iter().map(|k| (*k, what(k), kept.get(*k).copied())).collect()
}

/// The same, plus **whether an unanswered row is nevertheless allowed.** Four states rather than
/// three, because *허용해 둠* and *기본값으로 허용* are different things to be told.
pub fn all_with_defaults() -> Vec<(&'static str, &'static str, Option<Standing>, bool)> {
    all().into_iter()
        .map(|(k, what, how)| (k, what, how, ALLOWED_UNLESS_REFUSED.contains(&k)))
        .collect()
}
