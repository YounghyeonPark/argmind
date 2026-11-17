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
    report_in(untold, "ko", &mut |t: &str| t.to_string())
}

/// **[`report`], in the language the person is spoken to in** — since 2026-11-17: the program's
/// words from `crate::spoken`, and each request argmind set itself — the work model's, written in
/// Korean — through `into`, which leaves it as written while `app::REWRITES_WORK` is off.
pub fn report_in(untold: &[Pursued], code: &str, into: &mut dyn FnMut(&str) -> String) -> String {
    use crate::spoken::{say, OWN_FILES, OWN_FINISHED, OWN_GAVE_WAY, OWN_REPORT, OWN_UNFINISHED};
    if untold.is_empty() {
        return String::new();
    }
    //: **What the first report in the application got wrong** (2026-11-16): a piece stopped
    //: because the person handed something over was *다 못 했어요 … 작업 폴더에 있어요* — with nothing
    //: made and nothing in the folder — and its request was cut mid-word.
    let each: Vec<String> = untold.iter().map(|p| {
        let what: String = p.asked.split(" — 결과는").next().unwrap_or(&p.asked).trim().to_string();
        let what = into(&what);
        let what = if what.chars().count() > 40 { format!("{}…", what.chars().take(40).collect::<String>()) } else { what };
        let files = if p.made.is_empty() { String::new() } else { format!(" ({})", p.made.join(", ")) };
        let how = if p.finished { &OWN_FINISHED }
                  else if p.said.contains("사람이 멈췄") { &OWN_GAVE_WAY }
                  else { &OWN_UNFINISHED };
        format!("「{what}」 {}{files}", say(how, code, &[]))
    }).collect();
    let folder = if untold.iter().any(|p| !p.made.is_empty()) { say(&OWN_FILES, code, &[]) } else { String::new() };
    format!("{}{folder}", say(&OWN_REPORT, code, &[&untold.len().to_string(), &each.join(" / ")]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_report_is_one_short_sentence() {
        let p = Pursued { asked: "부업 아이디어 10개를 ideas.md 에 정리해줘".into(), finished: true, said: "정리했습니다".into(),
                          made: vec!["ideas.md".into()] };
        let said = report(&[p.clone(), Pursued { finished: false, made: vec![], ..p.clone() }]);
        assert!(said.starts_with("목표로 스스로 한 일 2가지") && said.contains("끝냈어요 (ideas.md)") && said.contains("다 못 했어요")
                && said.ends_with("작업 폴더에 있어요."), "{said}");
        assert_eq!(report(&[]), "");
        //: The application's first report: given way, nothing made, the request cut mid-word.
        let gave_way = Pursued { asked: "수익_방법_목록.md 파일에 기재된 수익 창출 방법의 개수를 정확히 세어 부업_비교.md에 기록해줘".into(),
                                 finished: false, said: "작업을 다 끝내지 못했습니다 — 사람이 멈췄습니다".into(), made: vec![] };
        let said = report(&[gave_way]);
        assert!(said.contains("맡기신 일 때문에 멈췄어요") && !said.contains("작업 폴더에 있어요") && said.contains("…」"), "{said}");
        //: The program's own tail is not the request.
        let tailed = Pursued { asked: "부업을 비교해줘 — 결과는 작업 폴더의 목표_3.md 에 써줘".into(), ..p };
        assert!(report(&[tailed]).contains("「부업을 비교해줘」"));
    }
}
