//! **The language the page is written in — since 2026-11-17**, the owner's *인터페이스언어도
//! 국가별로 설정할수 있게 해줘*, the day after argmind could be set to *answer* in a language.
//!
//! Two settings because they are two questions: what argmind says is the model's, and is
//! `mind.json`'s *대화 언어*; what the page says around it — the switches, the rows, the setup
//! panel, the program's own notes — is this. A person can read Japanese chrome around a Korean
//! conversation, and nothing here forces the two together.
//!
//! **The server stays Korean.** Every sentence it writes is still the Korean one, byte for byte;
//! the page translates what it shows from one table, `ui/src/words.ts`, at the moment it shows it,
//! and leaves alone what a person, a model or a screen wrote. So this file only remembers the
//! choice, and an unset one means *the system's language*, which the page reads for itself.
//!
//! Its own file rather than a key in `mind.json`: the mind is what the model is shown, and its
//! writer rewrites the file whole — `tact.json` is apart for the same reason.

use std::path::PathBuf;

/// The languages the page has words for, in the order it lists them.
pub const LANGUAGES: [&str; 4] = ["ko", "en", "zh", "ja"];

const KEY: &str = "화면 언어";

pub fn at() -> PathBuf {
    let mut path = std::env::var("LOCALAPPDATA").map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir());
    path.push("argmind");
    path.push("interface.json");
    path
}

pub fn is_one(code: &str) -> bool {
    LANGUAGES.contains(&code)
}

/// The code written down, or `""` — nothing chosen, or a code nobody offers.
pub fn parse(text: &str) -> String {
    text.lines()
        .filter_map(|l| l.trim().strip_prefix(&format!("\"{KEY}\"")))
        .filter_map(|rest| rest.trim_start().strip_prefix(':'))
        .map(|v| v.trim().trim_end_matches(',').trim().trim_matches('"').to_string())
        .find(|c| is_one(c))
        .unwrap_or_default()
}

/// `{}` when nothing is chosen, so a file nobody touched says nothing.
pub fn render(code: &str) -> String {
    if is_one(code) { format!("{{\n  \"{KEY}\": \"{code}\"\n}}\n") } else { "{}\n".to_string() }
}

pub fn remembered() -> String {
    std::fs::read_to_string(at()).map(|t| parse(&t)).unwrap_or_default()
}

/// Write the choice; `""` goes back to the system's language. An unknown code is refused and
/// nothing is written.
pub fn choose(code: &str) -> Result<(), String> {
    let code = code.trim();
    if !code.is_empty() && !is_one(code) {
        return Err(format!("{code} 라는 언어는 없습니다"));
    }
    let path = at();
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    std::fs::write(&path, render(code)).map_err(|e| format!("{} 에 쓰지 못했습니다 ({e})", path.display()))
}

/// **The language the tray menu is written in**: the one chosen, or the system's when it is one of
/// ours, or Korean — the same order the page follows, so the two do not disagree.
pub fn for_the_tray(chosen: &str, system: Option<&str>) -> &'static str {
    let pick = if is_one(chosen) { chosen } else { system.filter(|s| is_one(s)).unwrap_or("ko") };
    LANGUAGES.into_iter().find(|c| *c == pick).unwrap_or("ko")
}

/// The tray menu's two words, *show* and *quit*, in `code`.
pub fn tray_words(code: &str) -> [&'static str; 2] {
    match code {
        "en" => ["Show", "Quit"],
        "zh" => ["显示", "退出"],
        "ja" => ["表示", "終了"],
        _ => ["보이기", "끝내기"],
    }
}

/// The page with the choice stamped into it, as the token is: the page knows its language before
/// it draws anything, so there is no moment of Korean before the switch.
pub fn stamped(page: &str, code: &str) -> String {
    page.replace("__LANG__", if is_one(code) { code } else { "" })
}
