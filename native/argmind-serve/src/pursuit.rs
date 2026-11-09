//! **What argmind did toward the goal by itself, kept** — `argmind_core::pursuit`'s record, on disk.
//!
//! `%LOCALAPPDATA%\argmind\pursuit.jsonl`, one line a piece of work, appended when it ends and read
//! back for the next pick, so closing argmind does not start the goal over. Written by the program
//! from what the task loop recorded — the request, whether its finish was taken, its closing words,
//! the files it made — never by the model. Deleting the file starts the goal from nothing.

use std::io::Write;
use std::path::PathBuf;

use argmind_core::pursuit::Pursued;

pub fn at() -> PathBuf {
    let mut path = std::env::var("LOCALAPPDATA").map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir());
    path.push("argmind");
    path.push("pursuit.jsonl");
    path
}

/// The record, oldest first.
pub fn remembered() -> Vec<Pursued> {
    std::fs::read_to_string(at()).map(|t| argmind_core::pursuit::record_from(&t)).unwrap_or_default()
}

/// Add one to the end.
pub fn append(p: &Pursued) -> Result<(), String> {
    let path = at();
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(&path)
        .map_err(|e| format!("{path:?}: {e}"))?;
    writeln!(f, "{}", argmind_core::pursuit::record_line(p, crate::activity::now() as f64))
        .map_err(|e| format!("{path:?}: {e}"))
}

/// **What argmind says when somebody comes back** — short, the owner's choice: how many, what each
/// was, and which files. Empty when there is nothing untold.
pub fn report(untold: &[Pursued]) -> String {
    if untold.is_empty() {
        return String::new();
    }
    let each: Vec<String> = untold.iter().map(|p| {
        let what: String = p.asked.chars().take(50).collect();
        let files = if p.made.is_empty() { String::new() } else { format!(" ({})", p.made.join(", ")) };
        format!("「{what}」 {}{files}", if p.finished { "끝냈어요" } else { "다 못 했어요" })
    }).collect();
    format!("목표로 스스로 한 일 {}가지 — {}. 작업 폴더에 있어요.", untold.len(), each.join(" / "))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_report_is_one_short_sentence() {
        let p = Pursued { asked: "부업 아이디어 10개를 ideas.md 에 정리해줘".into(), finished: true, said: "정리했습니다".into(),
                          made: vec!["ideas.md".into()] };
        let said = report(&[p.clone(), Pursued { finished: false, made: vec![], ..p }]);
        assert!(said.starts_with("목표로 스스로 한 일 2가지") && said.contains("끝냈어요 (ideas.md)") && said.contains("다 못 했어요"),
                "{said}");
        assert_eq!(report(&[]), "");
    }
}
