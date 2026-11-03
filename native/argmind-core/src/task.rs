//! **작업** — a piece of work argmind carries out to the end with tools, one step at a time.
//!
//! The owner, 2026-11-03: *복잡한 업무도 처리가능해야함 문서수정이나 작성 그리고 코드도 필요하면
//! 생성해야하고 실행도 스스로 할수 있어야함.* A turn picks at most three clicks from one look at the
//! screen; real work is a loop — do one thing, see what came of it, decide the next.
//!
//! ## Who decides what
//!
//! The same split as everything else here, because it is the only one this project has measured to
//! work with a small model:
//!
//! * **The model picks one tool and its arguments a step**, or says the work is done. Nothing else.
//! * **The program runs the tool and writes down what came back** — file contents, a command's
//!   output, an error — verbatim and cut to [`OBSERVED_CHARS`]. The next prompt shows that record,
//!   so *I ran it and it printed 55* is something the model reads, never something it gets to say.
//!   The note invented having tidied somebody's tabs the one time it was asked what it had done.
//! * **The program decides when to stop**: [`MOST_STEPS`], [`ERRORS_IN_A_ROW`] failures, the same
//!   call made twice with the same result, or replies that do not read. A loop the model controls
//!   is a loop that does not end — `errand` measured 0 of 3 open errands finishing when the model
//!   chose its own state.
//!
//! ## What the core knows
//!
//! Tools by name, arguments as JSON, results as text. [`Tools`] is what an environment provides —
//! files in a workspace, a command in a sandbox, a web search — and **nothing here names a
//! platform**: where a command runs is the environment's business.

use serde_json::{json, Value};

use crate::contract::Speaks;

/// The most steps a task takes before it is handed back unfinished. Chosen: enough to write a file,
/// run it, read the error, fix it and run it again several times over; few enough that a task going
/// round in circles costs minutes, not an afternoon.
pub const MOST_STEPS: usize = 25;

/// Failures in a row — a tool refusing, a command failing to start — before the task stops.
pub const ERRORS_IN_A_ROW: usize = 3;

/// How much of a tool's result is kept and shown. Chosen: a page of output, which is what a person
/// reads before deciding the next thing; more is the prompt filling with a log.
pub const OBSERVED_CHARS: usize = 2000;

/// Steps shown in full in the prompt; older ones are one line each.
pub const STEPS_SHOWN_WHOLE: usize = 6;

/// Times a `done` may be refused before the task is stopped instead.
pub const DONE_REFUSED_MOST: usize = 2;

/// Said beside a call that repeats the one before it with the same result. The third in a row stops.
const SAME_AGAIN: &str = "\n(바로 앞과 같은 호출에 같은 결과다. 같은 것을 또 하지 말고 다른 방법을 써라)";

/// One tool an environment offers: its name, what it does, and its arguments as an example.
#[derive(Clone, Debug)]
pub struct ToolSpec {
    pub name: String,
    pub does: String,
    pub args: Value,
}

/// **What an environment must provide for a task.** Results are text the model will read; an error
/// is text too, and counted.
pub trait Tools {
    fn list(&self) -> Vec<ToolSpec>;
    fn call(&mut self, tool: &str, args: &Value) -> Result<String, String>;
    /// Of these paths, the ones that are not there. An environment that cannot tell says none are
    /// missing, which only makes the check on `done` weaker.
    fn missing(&self, _paths: &[String]) -> Vec<String> {
        Vec::new()
    }
}

/// **Several tool sets as one** — the workbench's files and commands beside the web, say. A name
/// goes to the first set that offers it; a path is missing if any set says so.
pub struct Joined(pub Vec<Box<dyn Tools + Send>>);

impl Tools for Joined {
    fn list(&self) -> Vec<ToolSpec> {
        self.0.iter().flat_map(|t| t.list()).collect()
    }
    fn call(&mut self, tool: &str, args: &Value) -> Result<String, String> {
        for t in self.0.iter_mut() {
            if t.list().iter().any(|s| s.name == tool) {
                return t.call(tool, args);
            }
        }
        Err(format!("그런 도구는 없다: {tool}"))
    }
    fn missing(&self, paths: &[String]) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        for t in &self.0 {
            for p in t.missing(paths) {
                if !out.contains(&p) {
                    out.push(p);
                }
            }
        }
        out
    }
}

/// One step as it happened.
#[derive(Clone, Debug, PartialEq)]
pub struct Step {
    pub tool: String,
    pub args: Value,
    /// What came back, cut — or the error, prefixed.
    pub observed: String,
    pub failed: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Ended {
    /// The model said so, with what to tell the person.
    Done(String),
    /// The program stopped it, and why — in the program's words.
    Stopped(String),
}

#[derive(Clone, Debug)]
pub struct Task {
    pub what: String,
    pub steps: Vec<Step>,
    pub ended: Option<Ended>,
    /// Replies that did not read as a tool call or `done`.
    pub unreadable: usize,
}

impl Task {
    pub fn new(what: &str) -> Task {
        Task { what: what.trim().to_string(), steps: Vec::new(), ended: None, unreadable: 0 }
    }
}

/// What a reply asks for.
#[derive(Debug, PartialEq)]
pub enum Next {
    Call { tool: String, args: Value },
    Done(String),
    Unreadable,
}

fn cut(text: &str, most: usize) -> String {
    let n = text.chars().count();
    if n <= most {
        return text.to_string();
    }
    //: **The end is kept, not the start**: a command's verdict — the error, the last line of a
    //: test run — is at the bottom of its output.
    let tail: String = text.chars().skip(n - most).collect();
    format!("…(앞 {}자 생략)\n{tail}", n - most)
}

/// The prompt for the next step.
pub fn ask_for_a_step(task: &Task, tools: &[ToolSpec], today: &str) -> String {
    let mut p = String::new();
    p.push_str("너는 사람의 PC에서 일을 대신 끝까지 해내는 에이전트다. 도구를 한 번에 하나씩 쓰고, \
결과를 보고 다음을 정한다.\n\n");
    p.push_str(&format!("맡은 일: {}\n오늘 날짜: {today}\n\n", task.what));
    p.push_str("쓸 수 있는 도구:\n");
    for t in tools {
        p.push_str(&format!("- {} {} — {}\n", t.name, t.args, t.does));
    }
    p.push('\n');
    if task.steps.is_empty() {
        p.push_str("아직 한 것이 없다.\n\n");
    } else {
        p.push_str("지금까지 한 것 (결과는 도구가 돌려준 그대로다):\n");
        let older = task.steps.len().saturating_sub(STEPS_SHOWN_WHOLE);
        for (i, s) in task.steps.iter().enumerate() {
            if i < older {
                let first = s.observed.lines().next().unwrap_or("").chars().take(80).collect::<String>();
                p.push_str(&format!("{}. {} {} → {}{first}\n", i + 1, s.tool, s.args,
                                    if s.failed { "실패: " } else { "" }));
            } else {
                p.push_str(&format!("{}. {} {}\n결과{}:\n{}\n", i + 1, s.tool, s.args,
                                    if s.failed { " (실패)" } else { "" }, s.observed));
            }
        }
        p.push('\n');
    }
    p.push_str("규칙:\n- 파일 내용이나 실행 결과는 도구로 확인한 것만 믿어라. 확인하지 않은 것을 했다고 말하지 마라.\n\
- 오류가 나면 오류를 읽고 고쳐서 다시 해 보라.\n\
- 맡은 일을 다 했으면 done 으로 끝내고, 사람에게 무엇을 했는지와 결과를 말하라.\n\n\
JSON 하나로만 답하라. 다음 중 하나:\n\
{\"tool\": \"도구 이름\", \"args\": {...}}\n\
{\"done\": \"사람에게 줄 결과\"}");
    p
}

/// What a reply asks for. **Prose around the object is tolerated**; an object with neither a tool
/// nor `done` is unreadable.
pub fn read_step(reply: &str) -> Next {
    let Some(v) = crate::llm::first_object(reply) else { return Next::Unreadable };
    if let Some(done) = v.get("done").and_then(Value::as_str) {
        return Next::Done(done.trim().to_string());
    }
    match v.get("tool").and_then(Value::as_str) {
        Some(t) if !t.trim().is_empty() => Next::Call {
            tool: t.trim().to_string(),
            args: v.get("args").cloned().unwrap_or_else(|| json!({})),
        },
        _ => Next::Unreadable,
    }
}

/// File names a sentence mentions — `summary.md에` reads `summary.md`. ASCII only, an extension
/// with a letter in it, so `3.14` and `316000.0` are not files.
pub fn file_names(text: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for word in text.split(|c: char| !(c.is_ascii_alphanumeric() || "._-/".contains(c))) {
        let word = word.trim_matches(|c| c == '.' || c == '/');
        let Some((stem, ext)) = word.rsplit_once('.') else { continue };
        if !stem.is_empty() && (1..=5).contains(&ext.len())
            && ext.chars().all(|c| c.is_ascii_alphanumeric())
            //: **An extension starts with a letter**: *15.5cm* read as a file called `15` with the
            //: extension `5cm`, and a phone comparison was refused for a file that was a length.
            && ext.starts_with(|c: char| c.is_ascii_alphabetic()) && !out.iter().any(|o| o == word) {
            out.push(word.to_string());
        }
    }
    out
}

/// **Is this a piece of work for the task loop, rather than a turn?** A word rule, not a model call
/// — the same reason `surface::web_move` and `mind::asked_to_keep` are: a thing to work on (a file
/// name, 파일, 문서, 코드…) and something to do to it (만들, 고쳐, 실행, 요약…). An address is
/// not a file, so `naver.com` does not count; *메모장 열어줘* has nothing to do to a thing and stays
/// a turn on the desktop.
///
/// **And research that ends in a plan** — since 2026-11-03, when *11월 일본여행 계획 2박3일로 성인2명
/// 항공권이랑 호텔 조사해서 적절한 스케쥴을 세워줘* went to an ordinary turn three times, was answered
/// *찾아볼까요?* each time, and once clicked a Git button in the editor.
pub fn asks_for_work(text: &str) -> bool {
    const THINGS: [&str; 22] = ["파일", "문서", "코드", "스크립트", "프로그램", "보고서", "엑셀", "csv",
                                "CSV", "파이썬", "python", "Python", "테스트", "작업 폴더",
                                "계획", "일정", "스케줄", "스케쥴", "여행", "비교", "목록", "리스트"];
    const DOING: [&str; 22] = ["만들", "작성", "고쳐", "고치", "수정", "바꿔", "정리", "실행", "계산",
                               "요약", "써 줘", "써줘", "짜 줘", "짜줘", "돌려", "분석", "번역",
                               "조사", "세워", "알아봐", "찾아서", "찾아 "];
    const ADDRESSES: [&str; 6] = ["com", "net", "org", "kr", "io", "html"];
    let names_a_file = file_names(text).iter().any(|f| {
        let ext = f.rsplit('.').next().unwrap_or("");
        !ADDRESSES.contains(&ext.to_ascii_lowercase().as_str())
    });
    (names_a_file || THINGS.iter().any(|w| text.contains(w))) && DOING.iter().any(|w| text.contains(w))
}

/// Numbers of three digits or more, grouping commas removed.
pub fn numbers(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let joined: String = chars.iter().enumerate().filter(|&(i, &c)| {
        //: A comma between digits is grouping; anywhere else it ends the number.
        !(c == ',' && i > 0 && chars[i - 1].is_ascii_digit()
          && chars.get(i + 1).is_some_and(|n| n.is_ascii_digit()))
    }).map(|(_, &c)| c).collect();
    //: **And any number with 만 or 억 after it, written out** — *48만원* is a price of two digits,
    //: and a trip plan's *48만~58만원* went past a rule that only read three. Written out on both
    //: sides, so *48만원* on a page answers *480,000원* in a plan and the other way round.
    let jc: Vec<char> = joined.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < jc.len() {
        if !jc[i].is_ascii_digit() {
            i += 1;
            continue;
        }
        let start = i;
        while i < jc.len() && jc[i].is_ascii_digit() {
            i += 1;
        }
        let digits: String = jc[start..i].iter().collect();
        //: **억, 만 and 천 add up**, in that order: *114만 5천 원* is 1,145,000 and was read as
        //: 1,140,000 — which no page said, so a right price was refused.
        let mut value: Option<u128> = None;
        let mut group: u128 = digits.parse().unwrap_or(0);
        let mut j = i;
        loop {
            let k = (j..jc.len()).find(|&k| !jc[k].is_whitespace()).unwrap_or(jc.len());
            let scale: u128 = match jc.get(k) {
                Some('억') => 100_000_000,
                Some('만') => 10_000,
                Some('천') => 1_000,
                _ => break,
            };
            value = Some(value.unwrap_or(0) + group * scale);
            j = k + 1;
            //: Another group follows only as digits and a smaller unit.
            let m = (j..jc.len()).find(|&m| !jc[m].is_whitespace()).unwrap_or(jc.len());
            let mut e = m;
            while e < jc.len() && jc[e].is_ascii_digit() {
                e += 1;
            }
            if e == m {
                break;
            }
            let next: String = jc[m..e].iter().collect();
            let unit = (e..jc.len()).find(|&u| !jc[u].is_whitespace()).and_then(|u| jc.get(u));
            if !matches!(unit, Some('억' | '만' | '천')) {
                break;
            }
            group = next.parse().unwrap_or(0);
            j = e;
        }
        match value {
            Some(v) => {
                out.push(v.to_string());
                i = j.max(i);
            }
            None if digits.len() >= 3 => out.push(digits),
            None => {}
        }
    }
    out
}

/// What the task wrote, by path: whole files and the replacing half of edits.
fn written_files(task: &Task) -> Vec<(String, String)> {
    task.steps.iter().filter(|s| !s.failed && (s.tool == "write_file" || s.tool == "edit_file"))
        .filter_map(|s| {
            let path = s.args.get("path")?.as_str()?.to_string();
            let body = s.args.get("content").or_else(|| s.args.get("replace"))?.as_str()?.to_string();
            Some((path, body))
        }).collect()
}

/// A file that is a program, whose numbers are its own business rather than claims about the world.
fn is_code(path: &str) -> bool {
    const CODE: [&str; 14] = ["py", "js", "ts", "sh", "ps1", "rs", "c", "cpp", "h", "java", "go", "rb",
                              "bat", "sql"];
    path.rsplit_once('.').is_some_and(|(_, ext)| CODE.contains(&ext.to_ascii_lowercase().as_str()))
}

/// **Why a `done` cannot be taken at its word**, or `None`. From the first measurement: qwen3:8b
/// ended five of six tasks with `done` after nought to two steps, naming a summary it had not
/// written and a total — *합계는 150000* — that no command had printed. So the program checks what
/// it can: that something was done at all, that the files the closing words name are there, and
/// that every number in them came from the work, the request or the date. **What it cannot check
/// is a summary that is there and wrong.**
pub fn why_not_done(task: &Task, said: &str, tools: &dyn Tools, today: &str) -> Option<String> {
    if !task.steps.iter().any(|s| !s.failed) {
        return Some("아직 도구로 한 일이 하나도 없다. 직접 해 보고 확인한 다음에 끝내라".into());
    }
    //: **The last thing run failed and nothing ran since**: from `--bin complex`, a test file that
    //: could not import what it tested, run, failed, and was called done.
    let ran = |s: &&Step| s.tool.starts_with("run");
    if let Some(last) = task.steps.iter().filter(ran).last() {
        if last.failed {
            return Some("마지막으로 실행한 명령이 실패했다. 오류를 고쳐서 성공하는 것을 확인한 다음에 끝내라".into());
        }
    }
    let gone = tools.missing(&file_names(said));
    if !gone.is_empty() {
        return Some(format!("{} 은(는) 작업 폴더에 없다. 아직 만들지 않았다", gone.join(", ")));
    }
    //: **Not the refusals**: a refusal quotes the number it refused, and the first version let the
    //: same invented *150000* through on the second try because it had now been "seen".
    //: **And the code the work wrote — code, not prose.** A test file holding `is_leap(1900)` is a
    //: source for *1900*; leaving it out refused a right finish in `--bin complex`. The first fix
    //: counted *every* write, and a trip plan then wrote hotel prices no page had shown into
    //: `notes.md` and was taken on the strength of its own file. So a document's numbers are
    //: checked like the closing words', and only code is a source.
    let code: Vec<String> = written_files(task).into_iter().filter(|(p, _)| is_code(p))
        .map(|(_, body)| body).collect();
    let seen: String = task.steps.iter().filter(|s| s.tool != "done").map(|s| s.observed.as_str())
        .chain(code.iter().map(String::as_str))
        .chain([task.what.as_str(), today]).collect::<Vec<_>>().join("\n");
    let seen_numbers = numbers(&seen);
    let prose: String = written_files(task).into_iter().filter(|(p, _)| !is_code(p))
        .map(|(_, body)| body).collect::<Vec<_>>().join("\n");
    let mut unseen: Vec<String> = Vec::new();
    for n in numbers(said).into_iter().chain(numbers(&prose)) {
        if !seen_numbers.iter().any(|s| s.contains(n.as_str())) && !unseen.contains(&n) {
            unseen.push(n);
        }
    }
    if !unseen.is_empty() {
        return Some(format!("{} 은(는) 어떤 도구 결과에도 나온 적이 없는 숫자다. 계산은 명령을 실행해서 \
결과로 확인하라", unseen.join(", ")));
    }
    None
}

/// **One step**: ask, read, run, record — and stop the task when the program says so. Returns
/// whether the task is still going.
pub fn step(task: &mut Task, model: &mut dyn Speaks, tools: &mut dyn Tools, today: &str) -> bool {
    if task.ended.is_some() {
        return false;
    }
    if task.steps.len() >= MOST_STEPS {
        task.ended = Some(Ended::Stopped(format!("{MOST_STEPS}단계를 넘겨 멈췄습니다")));
        return false;
    }
    let reply = model.ask(&ask_for_a_step(task, &tools.list(), today));
    match read_step(&reply) {
        Next::Done(said) => {
            task.unreadable = 0;
            let Some(why) = why_not_done(task, &said, tools, today) else {
                task.ended = Some(Ended::Done(said));
                return false;
            };
            let refused = task.steps.iter().filter(|s| s.tool == "done").count() + 1;
            task.steps.push(Step { tool: "done".into(), args: json!({"said": said}),
                                   observed: format!("끝났다는 말을 받아들이지 않았다: {why}"), failed: true });
            if refused >= DONE_REFUSED_MOST {
                task.ended = Some(Ended::Stopped(format!("끝났다고 했지만 확인되지 않아 멈췄습니다 — {why}")));
                return false;
            }
            true
        }
        Next::Unreadable => {
            task.unreadable += 1;
            if task.unreadable >= ERRORS_IN_A_ROW {
                task.ended = Some(Ended::Stopped("답을 읽지 못해 멈췄습니다".into()));
                return false;
            }
            true
        }
        Next::Call { tool, args } => {
            task.unreadable = 0;
            let known = tools.list().iter().any(|t| t.name == tool);
            let (observed, failed) = if !known {
                (format!("그런 도구는 없다: {tool}"), true)
            } else {
                match tools.call(&tool, &args) {
                    Ok(text) => (cut(&text, OBSERVED_CHARS), false),
                    Err(why) => (cut(&format!("오류: {why}"), OBSERVED_CHARS), true),
                }
            };
            //: **The same call with the same result is a circle**, not progress. The first repeat is
            //: told so; the third in a row stops. Stopping at the first repeat ended tasks a word
            //: might have turned — a `list_files` sent twice, a failing edit sent twice.
            let same = task.steps.iter().rev().take_while(|s| s.tool == tool && s.args == args
                && s.observed.trim_end_matches(SAME_AGAIN) == observed).count();
            let observed = if same >= 1 { format!("{observed}{SAME_AGAIN}") } else { observed };
            let repeated = same >= 2;
            task.steps.push(Step { tool, args, observed, failed });
            let errors = task.steps.iter().rev().take_while(|s| s.failed).count();
            if errors >= ERRORS_IN_A_ROW {
                task.ended = Some(Ended::Stopped(format!("{ERRORS_IN_A_ROW}번 연속 실패해서 멈췄습니다")));
                return false;
            }
            if repeated {
                task.ended = Some(Ended::Stopped("같은 일을 되풀이해서 멈췄습니다".into()));
                return false;
            }
            true
        }
    }
}

/// Run a task to its end — the program's end or the model's.
pub fn run(task: &mut Task, model: &mut dyn Speaks, tools: &mut dyn Tools, today: &str,
           carry_on: &dyn Fn() -> bool) {
    while carry_on() && step(task, model, tools, today) {}
    if task.ended.is_none() {
        task.ended = Some(Ended::Stopped("사람이 멈췄습니다".into()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Scripted(Vec<&'static str>);
    impl Speaks for Scripted {
        fn ask(&mut self, _: &str) -> String {
            if self.0.is_empty() { String::new() } else { self.0.remove(0).to_string() }
        }
    }

    #[derive(Default)]
    struct Box_ {
        files: std::collections::BTreeMap<String, String>,
        calls: usize,
    }
    impl Tools for Box_ {
        fn list(&self) -> Vec<ToolSpec> {
            ["write_file", "read_file", "run"].iter().map(|n| ToolSpec {
                name: n.to_string(), does: String::new(), args: json!({}) }).collect()
        }
        fn call(&mut self, tool: &str, args: &Value) -> Result<String, String> {
            self.calls += 1;
            let path = args["path"].as_str().unwrap_or("").to_string();
            match tool {
                "write_file" => {
                    self.files.insert(path, args["content"].as_str().unwrap_or("").into());
                    Ok("썼다".into())
                }
                "read_file" => self.files.get(&path).cloned().ok_or_else(|| "없는 파일".into()),
                "run" => Ok("55".into()),
                _ => Err("?".into()),
            }
        }
    }

    #[test]
    fn a_task_runs_tools_and_ends_on_done_with_the_record_in_the_prompt() {
        let mut model = Scripted(vec![
            r#"{"tool": "write_file", "args": {"path": "a.py", "content": "print(55)"}}"#,
            r#"{"tool": "run", "args": {"command": "python3 a.py"}}"#,
            r#"{"done": "a.py 를 만들어 실행했고 55가 나왔습니다"}"#,
        ]);
        let mut tools = Box_::default();
        let mut t = Task::new("1부터 10까지 더해 줘");
        run(&mut t, &mut model, &mut tools, "오늘", &|| true);
        assert_eq!(t.steps.len(), 2);
        assert_eq!(t.steps[1].observed, "55");
        assert!(matches!(t.ended, Some(Ended::Done(_))));
        let p = ask_for_a_step(&t, &tools.list(), "오늘");
        assert!(p.contains("결과:\n55"), "the record is what the model reads");
    }

    #[test]
    fn the_program_stops_failures_circles_unknown_tools_and_long_tasks() {
        //: Three failures in a row — different files, so it is not also a circle.
        let mut model = Scripted(vec![r#"{"tool": "read_file", "args": {"path": "x"}}"#,
                                      r#"{"tool": "read_file", "args": {"path": "y"}}"#,
                                      r#"{"tool": "read_file", "args": {"path": "z"}}"#]);
        let mut t = Task::new("x");
        run(&mut t, &mut model, &mut Box_::default(), "오늘", &|| true);
        assert!(matches!(&t.ended, Some(Ended::Stopped(w)) if w.contains("연속 실패")));

        //: The same call, the same result: a circle.
        let mut model = Scripted(vec![r#"{"tool": "run", "args": {"command": "ls"}}"#; 5]);
        let mut t = Task::new("x");
        run(&mut t, &mut model, &mut Box_::default(), "오늘", &|| true);
        assert!(matches!(&t.ended, Some(Ended::Stopped(w)) if w.contains("되풀이")));
        assert_eq!(t.steps.len(), 3, "the first repeat is told, the third stops");
        assert!(t.steps[1].observed.contains("같은 것을 또 하지 말고"));

        //: A tool nobody offers is a failure, not a call.
        let mut model = Scripted(vec![r#"{"tool": "format_disk", "args": {}}"#; 3]);
        let mut tools = Box_::default();
        let mut t = Task::new("x");
        run(&mut t, &mut model, &mut tools, "오늘", &|| true);
        assert_eq!(tools.calls, 0, "never passed to the environment");

        //: Replies that do not read.
        let mut model = Scripted(vec!["음…"; 5]);
        let mut t = Task::new("x");
        run(&mut t, &mut model, &mut Box_::default(), "오늘", &|| true);
        assert!(matches!(&t.ended, Some(Ended::Stopped(w)) if w.contains("읽지 못해")));

        //: And a person can stop it.
        let mut model = Scripted(vec![r#"{"tool": "run", "args": {"command": "a"}}"#]);
        let mut t = Task::new("x");
        run(&mut t, &mut model, &mut Box_::default(), "오늘", &|| false);
        assert!(matches!(&t.ended, Some(Ended::Stopped(w)) if w.contains("사람이")));
    }

    #[test]
    fn a_done_is_checked_against_the_record() {
        //: Nothing done: refused, and refused again stops.
        let mut model = Scripted(vec![r#"{"done": "다 했습니다"}"#; 3]);
        let mut t = Task::new("x");
        run(&mut t, &mut model, &mut Box_::default(), "오늘", &|| true);
        assert!(matches!(&t.ended, Some(Ended::Stopped(w)) if w.contains("확인되지 않아")));

        //: A number no tool printed: refused, then corrected.
        let mut model = Scripted(vec![
            r#"{"tool": "run", "args": {"command": "python3 a.py"}}"#,
            r#"{"done": "합계는 150000 입니다"}"#,
            r#"{"done": "결과는 55 입니다"}"#,
        ]);
        let mut t = Task::new("x");
        run(&mut t, &mut model, &mut Box_::default(), "오늘", &|| true);
        assert!(matches!(&t.ended, Some(Ended::Done(s)) if s.contains("55")), "{:?}", t.ended);
        assert!(t.steps[1].observed.contains("150000"));

        //: The same invented number twice is refused twice — the refusal is not a source.
        let mut model = Scripted(vec![
            r#"{"tool": "run", "args": {"command": "python3 a.py"}}"#,
            r#"{"done": "합계는 150000 입니다"}"#,
            r#"{"done": "합계는 150000 입니다"}"#,
        ]);
        let mut t = Task::new("x");
        run(&mut t, &mut model, &mut Box_::default(), "오늘", &|| true);
        assert!(matches!(&t.ended, Some(Ended::Stopped(_))), "{:?}", t.ended);

        //: A file the closing words name and the tools say is missing.
        struct Missing(Box_);
        impl Tools for Missing {
            fn list(&self) -> Vec<ToolSpec> { self.0.list() }
            fn call(&mut self, a: &str, b: &Value) -> Result<String, String> { self.0.call(a, b) }
            fn missing(&self, p: &[String]) -> Vec<String> {
                p.iter().filter(|p| !self.0.files.contains_key(*p)).cloned().collect()
            }
        }
        let mut model = Scripted(vec![
            r#"{"tool": "read_file", "args": {"path": "m.txt"}}"#,
            r#"{"done": "summary.md에 요약했습니다"}"#,
        ]);
        let mut tools = Missing(Box_::default());
        tools.0.files.insert("m.txt".into(), "회의".into());
        let mut t = Task::new("x");
        step(&mut t, &mut model, &mut tools, "오늘");
        step(&mut t, &mut model, &mut tools, "오늘");
        assert!(t.ended.is_none() && t.steps[1].observed.contains("summary.md"), "{:?}", t.steps);

        //: A number in a document it wrote, from no page: refused, though the closing words name
        //: no number at all.
        let mut model = Scripted(vec![
            r#"{"tool": "run", "args": {"command": "python3 a.py"}}"#,
            r#"{"tool": "write_file", "args": {"path": "plan.md", "content": "호텔 150,000원"}}"#,
            r#"{"done": "계획을 썼습니다"}"#,
        ]);
        let mut t = Task::new("x");
        for _ in 0..3 {
            step(&mut t, &mut model, &mut Box_::default(), "오늘");
        }
        assert!(t.ended.is_none() && t.steps[2].observed.contains("150000"), "{:?}", t.steps);

        //: A number the code itself wrote is a source: the test file holds 1900.
        let mut model = Scripted(vec![
            r#"{"tool": "write_file", "args": {"path": "t.py", "content": "assert not is_leap(1900)"}}"#,
            r#"{"done": "1900년이 윤년이 아님을 검사합니다"}"#,
        ]);
        let mut t = Task::new("x");
        run(&mut t, &mut model, &mut Box_::default(), "오늘", &|| true);
        assert!(matches!(t.ended, Some(Ended::Done(_))), "{:?}", t.steps);
    }

    #[test]
    fn a_failed_last_run_refuses_the_finish() {
        struct Fails(Box_);
        impl Tools for Fails {
            fn list(&self) -> Vec<ToolSpec> { self.0.list() }
            fn call(&mut self, a: &str, b: &Value) -> Result<String, String> {
                if a == "run" { Err("Traceback".into()) } else { self.0.call(a, b) }
            }
        }
        let mut model = Scripted(vec![r#"{"tool": "write_file", "args": {"path": "a.py", "content": "x"}}"#,
                                      r#"{"tool": "run", "args": {"command": "python3 a.py"}}"#,
                                      r#"{"done": "됐습니다"}"#]);
        let mut tools = Fails(Box_::default());
        let mut t = Task::new("x");
        for _ in 0..3 {
            step(&mut t, &mut model, &mut tools, "오늘");
        }
        assert!(t.steps[2].observed.contains("실패했다"), "{:?}", t.steps);
    }

    #[test]
    fn work_is_told_from_a_turn_by_its_words() {
        for yes in ["1부터 100까지의 합을 계산하는 sum.py 를 만들고 실행해서 결과를 알려줘",
                    "meeting.txt 를 읽고 회의 내용을 summary.md 에 세 줄로 요약해줘",
                    "calc.py 를 고쳐서 테스트가 통과하게 해줘",
                    "보고서 문서 하나 작성해줘", "파이썬으로 엑셀 정리하는 코드 짜줘"] {
            assert!(asks_for_work(yes), "{yes}");
        }
        for yes in ["11월 일본여행 계획 2박3일로 성인2명 항공권이랑 호텔 조사해서 적절한 스케쥴을 세워줘",
                    "노트북 세 개 가격 비교해서 표로 정리해줘"] {
            assert!(asks_for_work(yes), "{yes}");
        }
        for no in ["메모장 열어줘", "오늘 서울 날씨 알려줘", "네이버에서 메일 열어줘", "검색창 눌러줘",
                   "여행 가고 싶다", "일정 보여줘",
                   "naver.com 열어서 정리해줘", "고마워", "이 파일 열어줘", "나를 대표님이라고 불러"] {
            assert!(!asks_for_work(no), "{no}");
        }
    }

    #[test]
    fn names_and_numbers_are_read_the_narrow_way() {
        assert_eq!(file_names("summary.md에 쓰고 calc.py 를 고쳤다. 3.14 와 316000.0"),
                   vec!["summary.md", "calc.py"]);
        assert_eq!(numbers("합계 1,580,000원, 평균 316000.0, 12월 5일, 2개"), vec!["1580000", "316000"]);
        assert_eq!(numbers("48만~58만원, 1억 원, 3 만"), vec!["480000", "580000", "100000000", "30000"]);
        assert_eq!(numbers("114만 5천 원, 1억 2000만, 5천원, 365,600원"),
                   vec!["1145000", "120000000", "5000", "365600"]);
        assert_eq!(file_names("15.5cm (6.1\") 와 156.4mm"), Vec::<String>::new());
    }

    #[test]
    fn long_output_keeps_its_end() {
        let long = format!("{}마지막 줄: FAILED", "가".repeat(5000));
        let c = cut(&long, 100);
        assert!(c.ends_with("마지막 줄: FAILED"));
        assert!(c.chars().count() < 200);
    }
}
