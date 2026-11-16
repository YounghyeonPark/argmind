//! **스스로 정한 일** — the person's goal, pursued when nobody needs argmind. Since 2026-11-09.
//!
//! The owner, on what the 목표 box was for: *목표는 사용자 작업이 없을떄 스스로 목표를 설정하고
//! 이루도록 수행하는것을 이야기함.* It had been rendered into every turn's prompt as *사용자가 기억해
//! 달라고 한 것*, so *돈을어떻게벌수있는지생각해줘* came back in answers to unrelated things — *안녕하세요!
//! 돈을 어떻게 벌 수 있는지 고민 중이시군요* to a window change. It is not shown to a turn any more; it
//! is what argmind works on by itself, and asked how, the owner chose: **continuously, the
//! conversation and handed work first**, with **everything a task can use**, **no limit on how
//! much**, and **a short word when they come back**.
//!
//! **The model picks the next piece; the program runs it as a task and keeps the record.** One call
//! here — the goal, today, what was done so far as the record has it, and the work folder's files —
//! answers with one request, written as a person would hand it over, which the task loop then
//! carries out and checks like any other. Nothing here runs anything. What the record says was done
//! is the program's (`Pursued`), never the model's account, for the reason `thinking.rs` gives:
//! asked what it had done, the model once invented tidying somebody's browser tabs.
//!
//! **What it may not pick is the program's too**: a request that would sign up, log in, pay, buy,
//! post, send or order **in the person's name** is refused by its words, whatever the goal — the
//! goal *make money* is exactly the one that leads there, and none of those can be taken back.
//! Chosen, not measured.

use serde_json::Value;

/// One piece of work argmind set itself, as the program recorded it.
#[derive(Debug, Clone, PartialEq)]
pub struct Pursued {
    /// The request it handed the task loop.
    pub asked: String,
    /// Whether the task's finish was accepted.
    pub finished: bool,
    /// The task's closing words, or why it stopped.
    pub said: String,
    /// Files it made or changed in the work folder, relative.
    pub made: Vec<String>,
}

/// How much of the record the prompt carries, newest last. Chosen: enough to build on the last
/// several pieces without the prompt becoming a log.
pub const SHOWN: usize = 12;

/// The longest request taken. A request is a sentence or two; past this it is a plan.
pub const MOST_CHARS: usize = 300;

/// Doing words that act in the person's name and cannot be taken back — as verbs, so *결제 수수료를
/// 비교해* is research and *결제해* is not. **It over-refuses** — *가입하는 방법을 조사해* is refused too,
/// and the next pick is asked for — which is the side to err on. *올리* alone was left out: *수익을
/// 올리는 방법* is the goal itself.
const IN_THEIR_NAME: [&str; 27] = ["가입하", "가입해", "로그인하", "로그인해", "결제하", "결제해", "구매하", "구매해",
                                    "송금하", "송금해", "게시하", "게시해", "올려줘", "올려 줘", "업로드하", "업로드해",
                                    "보내줘", "보내 줘", "발송", "주문하", "주문해", "신청하", "신청해", "등록하", "등록해",
                                    "입금하", "입금해"];

/// The prompt that asks for the next piece of work.
/// `refused` is why the last pick was not taken, when it was not: the model is deterministic at a
/// fixed prompt, and asked the same thing again it picks the same thing again — in the application,
/// twice running, until the backoff.
pub fn ask_for_next(aim: &str, done: &[Pursued], files: &[String], today: &str, refused: Option<&str>) -> String {
    let mut p = String::new();
    p.push_str("너는 사람의 PC에서 일하는 에이전트다. 사람이 맡긴 일이 없을 때는 사람이 정해 둔 목표를 향해 \
스스로 일한다. 한 번에 작업 하나씩, 작업 폴더에서 한다.\n\n");
    p.push_str(&format!("사람이 정한 목표: {}\n오늘 날짜: {today}\n\n", aim.trim()));
    p.push_str("지금까지 스스로 한 일 (오래된 것부터):\n");
    let shown = &done[done.len().saturating_sub(SHOWN)..];
    if shown.is_empty() {
        p.push_str("(아직 없음)\n");
    }
    for d in shown {
        let said: String = d.said.chars().take(160).collect();
        let made = if d.made.is_empty() { String::new() } else { format!(" (만든 파일: {})", d.made.join(", ")) };
        p.push_str(&format!("- 「{}」 → {}: {said}{made}\n", d.asked, if d.finished { "끝남" } else { "못 끝냄" }));
    }
    p.push_str("\n스스로 한 일이 만든 파일: ");
    if files.is_empty() {
        p.push_str("(없음)");
    } else {
        p.push_str(&files.iter().take(40).cloned().collect::<Vec<_>>().join(", "));
    }
    if let Some(why) = refused {
        p.push_str(&format!("\n\n직전에 고른 일은 받아들이지 않았다: {why}. 다른 일을 골라라."));
    }
    p.push_str("\n\n이 목표를 향해 다음에 할 일 하나를 정하라.\n\
- 작업 한 번으로 끝낼 수 있는 구체적인 일 하나. 결과는 작업 폴더의 파일로 남기고, 그 파일 이름을 요청에 적는다.\n\
- 이미 한 일을 되풀이하지 않는다. 앞의 결과를 읽고 이어서 더 나아간다. 못 끝낸 일은 더 작게 나눠 다시 할 수 있다.\n\
- 사람에게 묻지 않고 혼자 할 수 있는 일이어야 한다.\n\
- 사람의 이름으로 가입·로그인·결제·구매·송금·게시·전송·주문·신청하는 일은 고르지 않는다. 알아보고 정리하고 만들어 보는 일을 한다.\n\n\
JSON 하나로만 답하라: {\"next\": \"사람이 작업을 맡길 때처럼 쓴 요청 한 문장\", \"file\": \"결과를 쓸 파일 이름 (예: 부업_비교.md)\", \
\"why\": \"목표와 어떻게 이어지는지 한 문장\"}");
    p
}

/// Does a request name a file — a name with an extension? **Not the word 파일**: the first piece in
/// the application, *3D 인쇄 가능한 파일로 저장해줘*, named none and passed as if it had. **And a
/// particle may follow with no space** — *부업_비교.md에 써줘* was read as naming nothing, and the
/// program wrote the same file onto its end a second time (2026-11-09): the extension is the ASCII
/// run after the dot, whatever comes after it.
pub fn names_a_file(s: &str) -> bool {
    s.split(|c: char| c.is_whitespace() || "'\"「」()".contains(c))
        .any(|w| w.rsplit_once('.').is_some_and(|(a, rest)| {
            let ext: String = rest.chars().take_while(|c| c.is_ascii_alphanumeric()).collect();
            !a.is_empty() && (2..=5).contains(&ext.len()) && !rest[ext.len()..].starts_with(|c: char| c.is_ascii())
        }))
}

/// A file name the model gave, if it is one the work folder can take: no folder in it, an
/// extension, short.
fn plain_file(name: &str) -> Option<String> {
    let n = name.trim();
    let ok = !n.is_empty() && n.chars().count() <= 60 && !n.contains(['/', '\\', ':']) && !n.starts_with('.')
        && n.rsplit_once('.').is_some_and(|(a, ext)| !a.is_empty() && (2..=5).contains(&ext.len()));
    ok.then(|| n.to_string())
}

/// Why a request the model gave cannot be run, or the request.
///
/// **And where it ends up is the program's** — since the first `--bin pursuit` run: five picks,
/// all readable and none repeated, and **not one named a file** (*수익 창출 방법을 분류하고 목록을
/// 작성해 보세요*), so nothing would have been left to check or to tell. A request that names none
/// gets the `file` the reply gave, or one the program names, written onto its end.
pub fn read_next(reply: &str, done: &[Pursued]) -> Result<String, String> {
    let v = crate::llm::first_object(reply).ok_or("다음 할 일을 읽지 못했다")?;
    let next = v.get("next").and_then(Value::as_str).map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .ok_or("다음 할 일을 읽지 못했다")?;
    if next.chars().count() > MOST_CHARS {
        return Err(format!("다음 할 일이 {MOST_CHARS}자를 넘는다 — 작업 하나가 아니다"));
    }
    //: **A finished piece is not done again; an unfinished one may be, once more.** The rule had
    //: been *not one of the last three*, and in the application (2026-11-16) a piece stopped for the
    //: person's request was picked again — rightly — and refused twice running, which the backoff
    //: then counts as a goal that keeps failing.
    let squeezed = |s: &str| s.split_whitespace().collect::<Vec<_>>().join(" ");
    let same: Vec<&Pursued> = done.iter().filter(|d| squeezed(&d.asked) == squeezed(&next)).collect();
    if same.iter().any(|d| d.finished) {
        return Err(format!("「{next}」은(는) 이미 끝낸 일과 같다"));
    }
    if same.iter().filter(|d| !d.said.contains("사람이 멈췄")).count() >= 2 {
        return Err(format!("「{next}」은(는) 두 번 해서 못 끝낸 일과 같다 — 더 작게 나눠서 다르게 하라"));
    }
    if let Some(w) = IN_THEIR_NAME.iter().find(|w| next.contains(*w)) {
        return Err(format!("「{next}」은(는) 사람의 이름으로 하는 일이다 (「{w}」) — 스스로는 하지 않는다"));
    }
    if names_a_file(&next) {
        return Ok(next);
    }
    let file = v.get("file").and_then(Value::as_str).and_then(plain_file)
        .unwrap_or_else(|| format!("목표_{}.md", done.len() + 1));
    Ok(format!("{} — 결과는 작업 폴더의 {file} 에 써줘", next.trim_end_matches(['.', ' '])))
}

/// One record line, with the time it was kept.
pub fn record_line(p: &Pursued, at: f64) -> String {
    serde_json::json!({"at": at, "asked": p.asked, "finished": p.finished, "said": p.said, "made": p.made}).to_string()
}

/// The record, oldest first. A line that does not read is passed over.
pub fn record_from(text: &str) -> Vec<Pursued> {
    text.lines().filter_map(|l| serde_json::from_str::<Value>(l).ok()).filter_map(|v| {
        Some(Pursued {
            asked: v.get("asked")?.as_str()?.to_string(),
            finished: v.get("finished").and_then(Value::as_bool).unwrap_or(false),
            said: v.get("said").and_then(Value::as_str).unwrap_or("").to_string(),
            made: v.get("made").and_then(Value::as_array)
                .map(|a| a.iter().filter_map(|m| m.as_str().map(String::from)).collect()).unwrap_or_default(),
        })
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_record_line_reads_back_as_it_was_written() {
        let p = Pursued { asked: "부업 아이디어 \"10개\"를 ideas.md 에".into(), finished: true, said: "정리했습니다\n끝".into(),
                          made: vec!["ideas.md".into()] };
        let back = record_from(&format!("{}\n이건 읽히지 않는 줄\n{}\n", record_line(&p, 1.0),
                                        record_line(&Pursued { finished: false, ..p.clone() }, 2.0)));
        assert_eq!(back.len(), 2);
        assert_eq!(back[0], p);
        assert!(!back[1].finished);
    }

    fn did(asked: &str, finished: bool) -> Pursued {
        Pursued { asked: asked.into(), finished, said: "ideas.md 에 정리했습니다".into(), made: vec!["ideas.md".into()] }
    }

    #[test]
    fn the_prompt_carries_the_goal_the_record_and_the_folder() {
        let p = ask_for_next("돈을어떻게벌수있는지생각해줘", &[did("부업 아이디어 10개를 조사해 ideas.md 에 정리해줘", true)],
                             &["ideas.md".into()], "2026년 11월 9일", None);
        assert!(p.contains("목표: 돈을어떻게벌수있는지생각해줘"));
        assert!(p.contains("「부업 아이디어 10개를 조사해 ideas.md 에 정리해줘」 → 끝남"));
        assert!(p.contains("(만든 파일: ideas.md)"));
        assert!(p.contains("스스로 한 일이 만든 파일: ideas.md"));
        assert!(ask_for_next("x", &[], &[], "오늘", None).contains("(아직 없음)"));
        assert!(!ask_for_next("x", &[], &[], "오늘", None).contains("받아들이지 않았다"));
        assert!(ask_for_next("x", &[], &[], "오늘", Some("「a」은(는) 이미 끝낸 일과 같다")).contains("직전에 고른 일은 받아들이지 않았다: 「a」"));
        //: Only the newest SHOWN.
        let many: Vec<Pursued> = (0..20).map(|i| did(&format!("일 {i}"), true)).collect();
        let p = ask_for_next("x", &many, &[], "오늘", None);
        assert!(!p.contains("「일 7」") && p.contains("「일 8」") && p.contains("「일 19」"));
    }

    #[test]
    fn a_request_is_read_and_a_repeat_or_an_act_in_their_name_is_refused() {
        let done = vec![did("부업 아이디어 10개를 조사해 ideas.md 에 정리해줘", true)];
        assert_eq!(read_next(r#"{"next": "ideas.md 의 아이디어마다 초기 비용을 조사해 costs.md 에 표로 정리해줘", "why": "x"}"#, &done),
                   Ok("ideas.md 의 아이디어마다 초기 비용을 조사해 costs.md 에 표로 정리해줘".into()));
        assert!(read_next(r#"{"next": "부업 아이디어 10개를 조사해  ideas.md 에 정리해줘"}"#, &done).is_err());
        //: Unfinished once — or stopped for the person — may be tried again; twice unfinished may not.
        let again = "costs.md 에 비용을 정리해줘";
        let mut tried = vec![Pursued { asked: again.into(), finished: false, said: "작업을 다 끝내지 못했습니다 — 사람이 멈췄습니다".into(), made: vec![] }];
        let pick = format!(r#"{{"next": "{again}"}}"#);
        assert!(read_next(&pick, &tried).is_ok());
        tried.push(Pursued { said: "끝났다고 했지만 확인되지 않아 멈췄습니다".into(), ..tried[0].clone() });
        assert!(read_next(&pick, &tried).is_ok());
        tried.push(tried[1].clone());
        assert!(read_next(&pick, &tried).is_err_and(|e| e.contains("두 번")));
        for acting in ["크몽에 가입해서 서비스를 등록해줘", "블로그에 글을 올려줘", "고객에게 메일을 보내줘", "상품을 구매해줘"] {
            let r = read_next(&format!(r#"{{"next": "{acting}"}}"#), &done);
            assert!(r.as_ref().is_err_and(|e| e.contains("사람의 이름으로")), "{acting}: {r:?}");
        }
        //: Reading about those things is not doing them.
        assert!(read_next(r#"{"next": "온라인 결제 수수료를 비교해 fees.md 에 정리해줘"}"#, &done).is_ok());
        assert!(read_next("그냥 쉬겠습니다", &done).is_err());
        //: A request that names no file is given one — the reply's, or the program's.
        assert_eq!(read_next(r#"{"next": "수익 창출 방법을 분류해 목록을 작성해 보세요.", "file": "수익_방법.md"}"#, &done),
                   Ok("수익 창출 방법을 분류해 목록을 작성해 보세요 — 결과는 작업 폴더의 수익_방법.md 에 써줘".into()));
        assert_eq!(read_next(r#"{"next": "수익 창출 방법을 분류해 보세요", "file": "C:\\x\\a.md"}"#, &done),
                   Ok("수익 창출 방법을 분류해 보세요 — 결과는 작업 폴더의 목표_2.md 에 써줘".into()));
        assert!(names_a_file("costs.md 에 표로") && !names_a_file("파일로 남겨줘") && !names_a_file("목록을 작성해 보세요"));
        //: The owner's run, 2026-11-09: a particle with no space, and the file written on twice.
        assert!(names_a_file("결과는 작업 폴더의 부업_비교.md에 써줘"));
        assert_eq!(read_next(r#"{"next": "수익성 기준으로 비교해줘 — 결과는 작업 폴더의 부업_비교.md에 써줘", "file": "부업_비교.md"}"#, &done),
                   Ok("수익성 기준으로 비교해줘 — 결과는 작업 폴더의 부업_비교.md에 써줘".into()));
        assert!(!names_a_file("1.5배 늘었다") && !names_a_file("v2.0") && !names_a_file("e.g."));
        assert!(read_next(&format!(r#"{{"next": "{}"}}"#, "가".repeat(MOST_CHARS + 1)), &done).is_err());
    }
}
