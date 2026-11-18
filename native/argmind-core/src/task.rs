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

/// Items in a plan, most. Chosen: a plan longer than this is a list of steps, not of results.
pub const MOST_ITEMS: usize = 6;

/// Questions a task may put to the person, most. Chosen: a task that needs more than two answers
/// was not handed over, it was dictated.
pub const MOST_QUESTIONS: usize = 2;

/// Said when a call carries a result the record cut.
pub const CUT_COPIED: &str = "오류: 이 인자에는 잘려서 보인 도구 결과(…자 생략)가 들어 있다. 잘린 것은 보여 주기 위해 줄인 것이라 내용이 빠져 있다 — 명령의 출력을 직접 파일로 저장하라 (예: 명령 > 파일 또는 Out-File)";

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
/// **A file read the way a task reads it** — through the workbench's own `read_file`, so a Word
/// file is the text a task was shown. For the program's own reading, after a task.
pub fn read_through(tools: &mut dyn Tools, path: &str) -> Result<String, String> {
    tools.call("read_file", &json!({ "path": path }))
}

pub trait Tools {
    fn list(&self) -> Vec<ToolSpec>;
    fn call(&mut self, tool: &str, args: &Value) -> Result<String, String>;
    /// Of these paths, the ones that are not there. An environment that cannot tell says none are
    /// missing, which only makes the check on `done` weaker.
    fn missing(&self, _paths: &[String]) -> Vec<String> {
        Vec::new()
    }
    /// **Calls the program makes before the model's first step**, as `(tool, args)` — what the
    /// request names that the model would otherwise start without. Each is run and recorded as a
    /// step like any other. None by default.
    fn opening(&self, _what: &str) -> Vec<(String, Value)> {
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
    fn opening(&self, what: &str) -> Vec<(String, Value)> {
        self.0.iter().flat_map(|t| t.opening(what)).collect()
    }
}

/// **Which way in, first** — the owner, 2026-11-08: *작업 수행시 접근 우선순위는 mcp또는 api, 코드또는
/// 스크립트 작성및 실행, 그래도 안되면 샌드박스작업 수행.* A program's own interface says what it
/// can do in names; a script is checked by what it prints; pressing a program's buttons on a screen
/// is the slowest, the least checkable and the most fragile, so it is last.
///
/// Tool sets with a rank: [`Rank::Api`], [`Rank::Code`], [`Rank::Screen`], or [`Rank::Other`] for
/// what the order does not speak to (the web pane). **Listed in that order, each description saying
/// its rank**, and the program holds the last one: the first `screen_open` of a task in which no
/// API or code tool has been used is refused once and told why. Asked again, it goes — a second
/// asking is a decision, the same ground a *which one?* answer stands on — so a task that truly
/// needs a screen is slowed by one step, never stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Rank {
    Api = 1,
    Code = 2,
    Screen = 3,
    Other = 4,
}

impl Rank {
    fn label(self) -> &'static str {
        match self {
            Rank::Api => "(1순위: MCP·API) ",
            Rank::Code => "(2순위: 코드·스크립트) ",
            Rank::Screen => "(3순위: 화면 조작 — MCP·API 나 코드로 안 될 때만) ",
            Rank::Other => "",
        }
    }
}

/// Said when a screen is opened before anything above it was tried.
pub const SCREEN_LAST: &str = "화면 조작은 마지막 수단이다. 먼저 MCP·API 도구나 코드(스크립트 작성과 실행)로 할 수 \
있는지 해 보라. 그 방법으로는 정말 안 되는 일이면 screen_open 을 한 번 더 부르면 열린다";

pub struct Tiered {
    sets: Vec<(Rank, Box<dyn Tools + Send>)>,
    tried_above: bool,
    warned: bool,
}

impl Tiered {
    pub fn new(mut sets: Vec<(Rank, Box<dyn Tools + Send>)>) -> Self {
        sets.sort_by_key(|(r, _)| *r);
        Tiered { sets, tried_above: false, warned: false }
    }

    /// **The person named the work screen**, so the order does not send the task elsewhere first.
    /// The order is the owner's default for work that does not say how; *작업 화면에서 문자표를
    /// 열어서…* says how, and holding it back would have the task try PowerShell — which opens a
    /// program on the person's own desktop — before the place it was told to use.
    pub fn screen_asked_for(mut self, what: &str) -> Self {
        if what.contains("작업 화면") || what.contains("작업화면") {
            self.warned = true;
        }
        self
    }
}

impl Tools for Tiered {
    fn list(&self) -> Vec<ToolSpec> {
        self.sets.iter().flat_map(|(r, t)| t.list().into_iter().map(move |mut s| {
            s.does = format!("{}{}", r.label(), s.does);
            s
        })).collect()
    }
    fn call(&mut self, tool: &str, args: &Value) -> Result<String, String> {
        let Some(i) = self.sets.iter().position(|(_, t)| t.list().iter().any(|s| s.name == tool)) else {
            return Err(format!("그런 도구는 없다: {tool}"));
        };
        let rank = self.sets[i].0;
        if rank == Rank::Screen && tool == "screen_open" && !self.tried_above && !self.warned {
            self.warned = true;
            return Err(SCREEN_LAST.into());
        }
        //: **Reading is not trying**: the opening read of a document the request names is a
        //: `read_file` the program made, and listing a folder says nothing about whether code can.
        if matches!(rank, Rank::Api | Rank::Code) && !matches!(tool, "read_file" | "list_files") {
            self.tried_above = true;
        }
        self.sets[i].1.call(tool, args)
    }
    fn missing(&self, paths: &[String]) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        for (_, t) in &self.sets {
            for p in t.missing(paths) {
                if !out.contains(&p) {
                    out.push(p);
                }
            }
        }
        out
    }
    fn opening(&self, what: &str) -> Vec<(String, Value)> {
        self.sets.iter().flat_map(|(_, t)| t.opening(what)).collect()
    }
}

/// **Pressing on a page, as tools** — the step from *read the web* to *use it*: a form to fill, a
/// tab to open, a button whose answer appears only after it is pressed. The owner, 2026-11-06,
/// chose **argmind's own pane only**, since a task runs for minutes beside a person using their
/// own desktop.
///
/// **The model picks a number; the program turns it into a name and a role.** `look_pane` lists
/// what can be pressed, numbered, with no rectangle — the turn's own rule — and `click` takes the
/// number from the last look and nothing else. After a press the numbers are forgotten, since what
/// they named may have moved: a click on a stale number is refused and the model told to look.
///
/// What looking and pressing *are* is the environment's: `look` returns `(role, name)` pairs, and
/// `press` takes the effect and returns, in words, what came of it — in the application, the same
/// `propose` and `act` a turn goes through, rules, password fields and screen checks included.
pub struct Pressing<L, P> {
    look: L,
    press: P,
    seen: Vec<(String, String)>,
    /// Whether the work is argmind's own — `crate::pursuit` — rather than a person's request.
    own: bool,
}

impl<L, P> Pressing<L, P>
where L: FnMut() -> Result<Vec<(String, String)>, String>, P: FnMut(crate::intent::Effect) -> String {
    pub fn new(look: L, press: P) -> Self {
        Pressing { look, press, seen: Vec::new(), own: false }
    }

    /// **For work argmind set itself**: every press goes out as argmind's own idea, so the
    /// application gates it on 스스로 실행 rather than 확인 없이 실행 — the two questions the owner
    /// answered apart.
    pub fn own(look: L, press: P) -> Self {
        Pressing { look, press, seen: Vec::new(), own: true }
    }
}

/// How many pressable things one look shows. `MAX_LISTED` is a turn's; a page's list past this is
/// `crowding`'s collapse from 123.
pub const MOST_PRESSABLE: usize = 100;

fn effect(kind: &str, target: &str, role: &str, text: &str, own: bool) -> crate::intent::Effect {
    //: **Asked for**: a task is a request the person made — `because_asked` true, so a press is
    //: gated as *what you asked for* is, never as argmind's own idea. **Unless it is not**: work
    //: argmind set itself presses as its own idea.
    crate::intent::Effect { kind: kind.into(), target: target.into(), role: role.into(), text: text.into(),
                            because_asked: Some(!own), which: None, inside: None }
}

impl<L, P> Tools for Pressing<L, P>
where L: FnMut() -> Result<Vec<(String, String)>, String>, P: FnMut(crate::intent::Effect) -> String {
    fn list(&self) -> Vec<ToolSpec> {
        vec![
            ToolSpec { name: "look_pane".into(), args: json!({}),
                       does: "웹 칸에 지금 열린 페이지에서 누를 수 있는 것을 번호와 함께 본다. click·type_text·press_key 의 \
결과에는 새 목록이 이미 붙어 있다".into() },
            ToolSpec { name: "click".into(), args: json!({"item": 3}),
                       does: "가장 최근 목록의 번호의 것을 누른다. 입력칸이면 누른 뒤 type_text 로 쓴다. 결과에 새 목록이 붙는다".into() },
            ToolSpec { name: "type_text".into(), args: json!({"text": "A-1042"}),
                       does: "방금 누른 입력칸에 글을 쓴다".into() },
            ToolSpec { name: "press_key".into(), args: json!({"key": "Enter"}),
                       does: "키 하나를 누른다 (Enter, Tab, Escape …)".into() },
        ]
    }
    fn call(&mut self, tool: &str, args: &Value) -> Result<String, String> {
        let text = |k: &str| args.get(k).and_then(Value::as_str).map(str::trim).filter(|s| !s.is_empty())
            .map(String::from).ok_or_else(|| format!("{k} 가 없다"));
        let pressed = match tool {
            "look_pane" => {
                let seen = (self.look)()?;
                return Ok(self.listed(seen, OBSERVED_CHARS));
            }
            "click" => {
                let n = args.get("item").and_then(|v| v.as_u64().or_else(|| v.as_str()?.trim().parse().ok()))
                    .ok_or("item 은 look_pane 의 번호다")? as usize;
                let Some((role, name)) = self.seen.get(n).cloned() else {
                    return Err(if self.seen.is_empty() { "먼저 look_pane 으로 보라 — 번호는 마지막으로 본 목록의 것이다".into() }
                               else { format!("{n} 번은 마지막 목록에 없다 (0–{})", self.seen.len() - 1) });
                };
                (self.press)(effect("click", &name, &role, "", self.own))
            }
            "type_text" => (self.press)(effect("type", "", "", &text("text")?, self.own)),
            "press_key" => (self.press)(effect("key", "", "", &text("key")?, self.own)),
            other => return Err(format!("그런 도구는 없다: {other}")),
        };
        //: **What the numbers named may have moved — so it looks again by itself**, since
        //: 2026-11-09: the work screen's press forgot its numbers the same way and the owner's task
        //: stopped on three refusals in a row for want of a look (`screen.rs`). The result carries
        //: the new list, and its numbers are the ones held.
        self.seen.clear();
        let room = OBSERVED_CHARS.saturating_sub(pressed.chars().count() + 40);
        Ok(match (self.look)() {
            Ok(seen) => format!("{pressed}\n{} (페이지 글은 read_pane)", self.listed(seen, room)),
            Err(why) => format!("{pressed}\n(다시 보지 못했다 — {why}. 누르려면 look_pane 으로 다시 보라)"),
        })
    }
}

impl<L, P> Pressing<L, P> {
    /// The numbered list, kept within `room` characters — the record cuts a longer result from the
    /// front, which for a list is its first numbers — and the numbers past the end forgotten with it.
    fn listed(&mut self, mut seen: Vec<(String, String)>, room: usize) -> String {
        seen.truncate(MOST_PRESSABLE);
        if seen.is_empty() {
            self.seen.clear();
            return "웹 칸에 누를 수 있는 것이 보이지 않는다".into();
        }
        let head = "웹 칸에서 누를 수 있는 것 (번호: 종류 \"이름\"):";
        let mut used = head.chars().count() + 80;
        let mut lines = Vec::new();
        for (i, (role, name)) in seen.iter().enumerate() {
            let line = format!("{i}: {role} \"{name}\"");
            used += line.chars().count() + 1;
            if used > room {
                break;
            }
            lines.push(line);
        }
        let more = seen.len() - lines.len();
        seen.truncate(lines.len());
        self.seen = seen;
        let end = if more > 0 { format!("\n(목록이 길어 {}번까지만 실었다 — 그 뒤 {more}개는 번호가 없다)", lines.len().saturating_sub(1)) }
                  else { String::new() };
        format!("{head}\n{}{end}", lines.join("\n"))
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

/// **One thing that must be true before the work is finished**, and the words that showed it.
#[derive(Clone, Debug, PartialEq)]
pub struct Item {
    pub what: String,
    /// A quotation from a tool's result — checked to be there, by the program — or nothing yet.
    pub evidence: Option<String>,
    /// **Or taken out, and why** — in the model's words, which go to the person with the result.
    /// The first run with plans: *회의 내용을 담은 meeting_notes.md 가 있는지 확인한다*, written
    /// before anything was looked at, for a file that did not exist — and a right `deadline.md` was
    /// stopped for want of it. A plan written before the work guesses; a guess must be retractable,
    /// and in the open.
    pub skipped: Option<String>,
}

#[derive(Clone, Debug)]
pub struct Task {
    pub what: String,
    pub steps: Vec<Step>,
    pub ended: Option<Ended>,
    /// Replies that did not read as a tool call or `done`.
    pub unreadable: usize,
    /// **What must be true before `done`**, written by the model once, before any tool — see
    /// [`Task::planned`]. Empty when there is no plan, and then nothing about it is asked or shown.
    pub plan: Vec<Item>,
    /// Whether to ask for a plan first, and whether that has happened.
    pub planning: bool,
    pub planned: bool,
    /// **Reason only where it counts** — see [`reasons_here`]. Off: every call is asked as the
    /// model is set, which is how everything before this was measured.
    pub sparing: bool,
    /// How many of `steps` the program made before the model's first — see [`open`].
    pub opened: usize,
}

impl Task {
    pub fn new(what: &str) -> Task {
        Task { what: what.trim().to_string(), steps: Vec::new(), ended: None, unreadable: 0,
               plan: Vec::new(), planning: false, planned: false, sparing: false, opened: 0 }
    }

    /// **A task that writes down, before it starts, what must be true when it ends** — and cannot
    /// end until each of those has a quotation from a tool's result beside it. From the trip plan:
    /// *상세히 기록 완료* with no file written, and a plan with no flight times — the request said
    /// what was wanted and nothing compared the finish against it. The plan is written before any
    /// tool runs, while there is nothing done to make the work look smaller than it was.
    pub fn planned(what: &str) -> Task {
        Task { planning: true, ..Task::new(what) }
    }
}

/// What a reply asks for.
#[derive(Debug, PartialEq)]
pub enum Next {
    Call { tool: String, args: Value },
    Done(String),
    Unreadable,
}

/// **Does a call carry a cut result from where its visible part begins?** The first two lines after
/// the marker — the first of them usually half a line, where the cut fell — standing together in
/// any text the call passes. Only the beginning: a later line of a long listing quoted on its own is
/// a name somebody may well want, and half a line followed by the next one is nobody's choice.
/// Fitted to the one case that showed it, `--bin screen`'s *fonts*, 2026-11-09.
fn copies_a_cut(steps: &[Step], args: &Value) -> bool {
    fn texts<'a>(v: &'a Value, out: &mut Vec<&'a str>) {
        match v {
            Value::String(s) => out.push(s),
            Value::Array(a) => a.iter().for_each(|x| texts(x, out)),
            Value::Object(o) => o.values().for_each(|x| texts(x, out)),
            _ => {}
        }
    }
    let mut given = Vec::new();
    texts(args, &mut given);
    let given: Vec<String> = given.iter().map(|t| t.replace('\r', "")).collect();
    steps.iter().filter(|s| s.observed.starts_with("…(앞 ")).any(|s| {
        let tail = s.observed.replace('\r', "");
        let mut lines = tail.lines().skip(1);
        let (Some(a), Some(b)) = (lines.next(), lines.next()) else { return false };
        let start = format!("{a}\n{b}");
        start.trim().chars().count() >= 12 && given.iter().any(|g| g.contains(&start))
    })
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

/// The prompt for the plan — asked once, before any tool.
pub fn ask_for_a_plan(task: &Task, tools: &[ToolSpec], today: &str) -> String {
    let mut p = String::new();
    p.push_str("너는 사람의 PC에서 일을 대신 끝까지 해내는 에이전트다. 일을 시작하기 전에, 끝났다고 말하려면 \
무엇이 확인돼야 하는지 적는다.\n\n");
    p.push_str(&format!("맡은 일: {}\n오늘 날짜: {today}\n\n쓸 수 있는 도구:\n", task.what));
    for t in tools {
        p.push_str(&format!("- {} — {}\n", t.name, t.does));
    }
    p.push_str(&format!("\n맡은 일이 요구하는 결과를 빠짐없이, 2~{MOST_ITEMS}개 항목으로 적어라. 각 항목은 \
도구 결과로 확인할 수 있는 것이어야 한다 (예: \"report.md 에 범주별 합계를 썼다\", \"테스트가 통과했다\", \
\"항공편의 가격과 시간을 조회했다\"). 아직 아무것도 하지 마라.\n\n\
JSON 하나로만 답하라: {{\"plan\": [\"항목\", ...]}}"));
    p
}

/// The plan's items, or none when the reply does not read — a task without a plan is the task as it
/// was before plans, not a failure.
pub fn read_plan(reply: &str) -> Vec<Item> {
    crate::llm::first_object(reply)
        .and_then(|v| v.get("plan").and_then(Value::as_array).cloned())
        .unwrap_or_default().iter()
        .filter_map(|i| i.as_str().map(str::trim).filter(|t| !t.is_empty()))
        .take(MOST_ITEMS)
        .map(|t| Item { what: t.to_string(), evidence: None, skipped: None }).collect()
}

fn squash(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// **`check`: an item is done when the model quotes the result that shows it**, and the quotation
/// is found in a tool's result by the program. What it cannot check is a quotation that is there
/// and about something else — the item and the quote sit side by side for a person to read.
fn check_item(task: &mut Task, args: &Value) -> (String, bool) {
    let n = args.get("item").and_then(|v| v.as_u64().or_else(|| v.as_str()?.trim().parse().ok()));
    let Some(n) = n.map(|n| n as usize).filter(|n| (1..=task.plan.len()).contains(n)) else {
        return (format!("item 은 1~{} 중 하나여야 한다", task.plan.len()), true);
    };
    if let Some(why) = args.get("skip").and_then(Value::as_str).map(str::trim).filter(|w| !w.is_empty()) {
        task.plan[n - 1].skipped = Some(why.to_string());
        task.plan[n - 1].evidence = None;
        return (format!("{n}번을 계획에서 뺐다: {} — 이유는 결과와 함께 사람에게 전한다", task.plan[n - 1].what), false);
    }
    let quote = args.get("evidence").and_then(Value::as_str).unwrap_or("");
    let quote = squash(quote.trim().trim_matches(['"', '\'', '「', '」', '“', '”']));
    if quote.chars().count() < 2 {
        return ("evidence 에 도구 결과에서 그대로 옮긴 글을 넣어라".into(), true);
    }
    let found = task.steps.iter().filter(|s| !s.failed && s.tool != "check" && s.tool != "done")
        .any(|s| squash(&s.observed).contains(&quote));
    if !found {
        let short: String = quote.chars().take(60).collect();
        return (format!("「{short}」 은(는) 어떤 도구 결과에도 없다. 결과에 실제로 있는 글을 그대로 옮겨라"), true);
    }
    task.plan[n - 1].evidence = Some(quote);
    task.plan[n - 1].skipped = None;
    (format!("{n}번 확인: {}", task.plan[n - 1].what), false)
}

/// **The person, as a tool**: `ask_person`, at most [`MOST_QUESTIONS`] a task. What the person
/// answers is a tool's result like any other — a source for numbers and for a `check`. No answer is
/// an error that says to carry on with what is known and say what was assumed.
pub struct Asking<F> {
    ask: F,
    pub asked: usize,
}

impl<F: FnMut(&str) -> Option<String>> Asking<F> {
    pub fn new(ask: F) -> Self {
        Asking { ask, asked: 0 }
    }
}

impl<F: FnMut(&str) -> Option<String>> Tools for Asking<F> {
    fn list(&self) -> Vec<ToolSpec> {
        vec![ToolSpec {
            name: "ask_person".into(),
            does: format!("사람만 아는 것(개인 일정, 정한 내용, 취향, 예산 등)이 없어서 일을 할 수 없을 때만 \
하나를 묻는다. 찾아보거나 파일에서 읽거나 알아서 정해도 되는 것은 묻지 마라. 최대 {MOST_QUESTIONS}번"),
            args: json!({"question": "물을 것"}),
        }]
    }
    fn call(&mut self, _tool: &str, args: &Value) -> Result<String, String> {
        let q = args.get("question").or_else(|| args.get("value")).and_then(Value::as_str)
            .unwrap_or("").trim().to_string();
        if q.is_empty() {
            return Err("question 이 비었다".into());
        }
        if self.asked >= MOST_QUESTIONS {
            return Err(format!("이미 {MOST_QUESTIONS}번 물었다. 아는 것으로 하고, 가정한 것은 결과에 밝혀라"));
        }
        self.asked += 1;
        match (self.ask)(&q) {
            Some(a) => Ok(format!("사람의 답: {a}")),
            None => Err("답이 없었다. 아는 것으로 하고, 가정한 것은 결과에 밝혀라".into()),
        }
    }
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
    if !task.plan.is_empty() {
        p.push_str("- check {\"item\": 번호, \"evidence\": \"도구 결과에서 그대로 옮긴 부분\"} — 계획의 한 항목이 \
끝났음을 표시한다. evidence 는 아래 결과에 실제로 있는 글이어야 한다. 해 보니 할 수 없거나 필요 없는 항목이면 \
{\"item\": 번호, \"skip\": \"이유\"} 로 뺀다 — 이유는 사람에게 그대로 전해진다\n");
    }
    p.push('\n');
    if !task.plan.is_empty() {
        p.push_str("끝내기 전에 확인할 것 (일을 시작하기 전에 세운 계획):\n");
        for (i, item) in task.plan.iter().enumerate() {
            match (&item.evidence, &item.skipped) {
                (Some(e), _) => p.push_str(&format!("[x] {}. {} — 근거: 「{}」\n", i + 1, item.what,
                                                    e.chars().take(80).collect::<String>())),
                (None, Some(w)) => p.push_str(&format!("[-] {}. {} — 뺌: {w}\n", i + 1, item.what)),
                (None, None) => p.push_str(&format!("[ ] {}. {}\n", i + 1, item.what)),
            }
        }
        p.push('\n');
    }
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
        //: **The tool's name as the key**: given `ask_person`, qwen3:14b wrote
        //: `{"ask_person": "마감일이 언제예요?"}` three times in three, and an unreadable reply tells
        //: the model nothing, so it said it again. Read as a call; a name that is no tool then comes
        //: back as *그런 도구는 없다*, which is something the model can read.
        _ => match v.as_object().filter(|o| o.len() == 1).and_then(|o| o.iter().next()) {
            Some((name, args)) if !name.trim().is_empty() && name != "plan" => Next::Call {
                tool: name.trim().to_string(),
                args: match args {
                    Value::Object(_) => args.clone(),
                    Value::String(s) => json!({"value": s}),
                    _ => json!({}),
                },
            },
            _ => Next::Unreadable,
        },
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
    //: **The work screen is the task's** — nothing else has one, so a sentence that names it is
    //: work. The owner's *작업 화면에서 문자표(charmap)를 열어서…* went to a turn (2026-11-09), which
    //: asked *작업 화면에서 실행할까요?* about a screen it cannot reach.
    let names_the_work_screen = text.contains("작업 화면") || text.contains("작업화면");
    (names_the_work_screen || ((names_a_file || THINGS.iter().any(|w| text.contains(w)))
        && DOING.iter().any(|w| text.contains(w)))
        || work_in_other_tongues(text, names_a_file))
        && !asks_about(text)
}

/// **The same two lists in English, Chinese and Japanese** — since 2026-11-17, translated from the
/// Korean ones above rather than written from the sentences `--bin tongues` reads, so a request
/// the Korean rule misses (*블로그 글 하나 써줘*) is missed in the other three too. English is read
/// by the start of a word, so *files* and *organizing* count and *latest* is not a *test*. Each
/// language only in a sentence written in it (`tongue::reads_as`).
fn work_in_other_tongues(text: &str, names_a_file: bool) -> bool {
    use crate::tongue::{has_stem, lower, reads_as};
    if reads_as(text, "en") {
        let low = lower(text);
        const THINGS: [&str; 18] = ["file", "document", "code", "script", "program", "report", "excel",
                                    "spreadsheet", "csv", "python", "test", "work folder", "plan", "schedule",
                                    "itinerar", "trip", "compar", "list"];
        const DOING: [&str; 25] = ["make", "creat", "write", "draft", "fix", "edit", "modif", "chang", "organiz",
                                   "organis", "sort out", "run", "calculat", "summar", "put together", "analy",
                                   "translat", "research", "set up", "draw up", "look into", "find out", "find",
                                   "plan", "build"];
        return low.contains("work screen")
            || ((names_a_file || THINGS.iter().any(|w| has_stem(&low, w)))
                && DOING.iter().any(|w| has_stem(&low, w)));
    }
    if reads_as(text, "zh") {
        const THINGS: [&str; 21] = ["文件", "文档", "代码", "脚本", "程序", "报告", "Excel", "excel", "表格",
                                    "csv", "CSV", "Python", "python", "测试", "计划", "日程", "行程", "旅行",
                                    "比较", "列表", "清单"];
        const DOING: [&str; 25] = ["做", "制作", "创建", "生成", "写", "编写", "撰写", "修改", "改", "修复",
                                   "修正", "整理", "运行", "执行", "计算", "总结", "汇总", "分析", "翻译", "调查",
                                   "研究", "制定", "安排", "找出", "查找"];
        return text.contains("工作屏幕")
            || ((names_a_file || THINGS.iter().any(|w| text.contains(w)))
                && DOING.iter().any(|w| text.contains(w)));
    }
    if reads_as(text, "ja") {
        const THINGS: [&str; 24] = ["ファイル", "文書", "ドキュメント", "コード", "スクリプト", "プログラム",
                                    "レポート", "報告書", "エクセル", "Excel", "csv", "CSV", "Python", "python",
                                    "パイソン", "テスト", "計画", "予定", "日程", "スケジュール", "旅行", "比較",
                                    "一覧", "リスト"];
        const DOING: [&str; 22] = ["作って", "作成", "作る", "書いて", "直して", "修正", "変えて", "変更", "整理",
                                   "まとめ", "実行", "動かして", "計算", "要約", "翻訳", "分析", "調べて", "調査",
                                   "立てて", "組んで", "探して", "作り"];
        return text.contains("作業画面")
            || ((names_a_file || THINGS.iter().any(|w| text.contains(w)))
                && DOING.iter().any(|w| text.contains(w)));
    }
    false
}

/// **A turn's reply that says it is going to go and do something it cannot do in one look** — the
/// seam between a turn and the task loop, read off the reply rather than asked of a model.
///
/// The owner's trip request reached a turn three times and was answered *먼저 항공권과 호텔을
/// 찾아볼까요?* beside a click on a search box, and once on a Git button; *오늘 서울 날씨 알려줘* got
/// *잠시만 기다려주세요!* and nothing. Both are a reply promising work the turn has no way to do.
/// So the words are a doing that takes more than a look — looking for, looking into, putting
/// together, writing up — with an ending that offers or promises it, and *잠시만 기다려*. Written
/// from those three replies **before `--bin seam` ran**. A click promised (*눌러 드릴게요*,
/// *열어 볼게요*) is not here: that is what a turn does.
pub fn promises_work(reply: &str) -> bool {
    let r: String = reply.split_whitespace().collect();
    const DOING: [&str; 10] = ["찾아", "알아", "조사해", "검색해", "정리해", "작성해", "만들어", "짜", "비교해", "계획을세워"];
    const ENDING: [&str; 8] = ["볼게", "볼까요", "드릴게", "드릴까요", "드리겠", "보겠", "할게", "할까요"];
    r.contains("잠시만기다려")
        || DOING.iter().any(|d| ENDING.iter().any(|e| r.contains(&format!("{d}{e}"))))
}

/// **A question about how, or about whether argmind is good at something — not a request.** *문서
/// 작성 잘하는 법 알려줘* has a thing and a doing in it and went to work; *계획 세우는 거 잘해?* and
/// *일정 관리 어떻게 해?* went to work from the model three times in three. A question is answered
/// in a sentence, and starting a two-minute task on one is the harm.
///
/// Read off the ending: *…법 알려줘*, or a question with no request in it. *짜줄래?* and *정리해줄
/// 수 있어?* carry a 줘 or a 줄 and are requests. *만들 수 있어?* is left a request — chosen, not
/// measured: somebody asking whether argmind can make a thing usually wants it made.
pub fn asks_about(text: &str) -> bool {
    //: **One reading a sentence, by its script**: the Korean one's *a question with no 줘 in it* is
    //: true of every English question, *can you make a list?* among them, so a sentence with no
    //: Hangul is read in its own language — and every sentence with Hangul exactly as before.
    match crate::converse::written_in(text) {
        Some("en") | Some("zh") | Some("ja") => asks_about_in_other_tongues(text),
        _ => korean_asks_about(text),
    }
}

/// **The same question in English, Chinese and Japanese** — since 2026-11-17, the shape of the
/// Korean above: *how to* asked to be told (*tell me how to…*, *告诉我…的方法*, *…方法を教えて*), or a
/// question with no request in it. A request in a question's clothes carries its own words —
/// *can you*, *could you*, *will you*, *please*; *帮我*, *请*, *能*, *可以*, *给我*; *てくれる*,
/// *てもらえる*, *ください*, *お願い* — and stays a request.
fn asks_about_in_other_tongues(text: &str) -> bool {
    use crate::tongue::{has_word, lower, reads_as};
    let t = text.trim().trim_end_matches(['.', '!', '~', ' ', '。', '！']);
    let asked = t.ends_with('?') || t.ends_with('？');
    if reads_as(t, "en") {
        let low = lower(t);
        const HOW: [&str; 6] = ["tell me how", "show me how", "teach me how", "explain how", "what's the best way",
                                "what is the best way"];
        const REQUEST: [&str; 9] = ["can you", "could you", "would you", "will you", "please", "can i get",
                                    "i want you to", "i need you to", "would you mind"];
        //: A question without its mark, as people type one: it starts the way a question does.
        const STARTS: [&str; 12] = ["how ", "what ", "what's ", "why ", "when ", "which ", "who ", "are you ",
                                    "do you ", "is it ", "does ", "should i "];
        let opens = STARTS.iter().any(|s| low.starts_with(s));
        return HOW.iter().any(|h| low.contains(h))
            || ((asked || opens) && !REQUEST.iter().any(|r| has_word(&low, r)));
    }
    if reads_as(t, "zh") {
        const REQUEST: [&str; 7] = ["帮我", "请", "能", "可以", "给我", "替我", "麻烦"];
        let how = ["怎么", "如何", "方法"].iter().any(|w| t.contains(w));
        let q = t.trim_end_matches(['?', '？']);
        return ((t.contains("告诉我") || t.contains("教我")) && how)
            || ((asked || q.ends_with('吗') || q.ends_with('呢') || how)
                && !REQUEST.iter().any(|r| t.contains(r)));
    }
    if reads_as(t, "ja") {
        const HOW: [&str; 6] = ["方法を教えて", "やり方を教えて", "方法教えて", "方法を教えてください", "か教えて",
                                "仕方を教えて"];
        const REQUEST: [&str; 6] = ["てくれ", "でくれ", "てもらえ", "でもらえ", "ください", "お願い"];
        let how = ["どうやって", "どうすれば", "ってどう", "どうやる"].iter().any(|w| t.contains(w));
        return HOW.iter().any(|h| t.ends_with(h))
            || ((asked || how) && !REQUEST.iter().any(|r| t.contains(r)));
    }
    false
}

/// The Korean reading, as it was before the other three languages were added.
fn korean_asks_about(text: &str) -> bool {
    let t = text.trim().trim_end_matches(['.', '!', '~', ' ']);
    const HOW: [&str; 4] = ["법 알려줘", "법 알려 줘", "법 알려주세요", "법 좀 알려줘"];
    if HOW.iter().any(|h| t.ends_with(h)) {
        return true;
    }
    const REQUEST: [&str; 7] = ["줘", "줄래", "줄 수", "주세요", "주실", "줄까", "수 있"];
    let q = t.trim_end_matches('?');
    const ASKING: [&str; 6] = ["어떻게 해", "어떻게 하지", "잘해", "잘 해", "뭐야", "뭐부터 해야 돼"];
    (t.ends_with('?') || ASKING.iter().any(|a| q.ends_with(a)))
        && !REQUEST.iter().any(|r| q.contains(r))
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
    //: **And a drawing's markup** — since 2026-11-08, from `--bin mcp`: a plate drawn as an SVG by
    //: `write_file` was refused for *2000*, which is `xmlns="http://www.w3.org/2000/svg"`. A
    //: drawing's numbers are geometry the work chose, like a script's. Fitted to that one case.
    const CODE: [&str; 17] = ["py", "js", "ts", "sh", "ps1", "rs", "c", "cpp", "h", "java", "go", "rb",
                              "bat", "sql", "svg", "scad", "css"];
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
    let open: Vec<String> = task.plan.iter().enumerate()
        .filter(|(_, i)| i.evidence.is_none() && i.skipped.is_none())
        .map(|(n, i)| format!("{}번({})", n + 1, i.what)).collect();
    if !open.is_empty() {
        return Some(format!("계획의 {} 이(가) 아직 확인되지 않았다. 해 보고, 결과를 check 로 표시한 다음에 \
끝내라 — 할 수 없는 항목이면 skip 과 이유로 빼라", open.join(", ")));
    }
    //: **The last thing run failed and nothing ran since**: from `--bin complex`, a test file that
    //: could not import what it tested, run, failed, and was called done.
    //: **Anything that runs something** — a command, a program's own interface, an MCP server's
    //: tool. With only `run…` counted, a failed `python3 box.py` followed by a `blender_python`
    //: that exported the box still read as *the last command failed* (2026-11-09).
    let ran = |s: &&Step| s.tool.starts_with("run") || s.tool == "blender_python" || s.tool == "kicad_cli"
        || s.tool.contains("__");
    if let Some(last) = task.steps.iter().filter(ran).last() {
        if last.failed {
            return Some("마지막으로 실행한 명령이 실패했다. 오류를 고쳐서 성공하는 것을 확인한 다음에 끝내라".into());
        }
    }
    let gone = tools.missing(&file_names(said));
    if !gone.is_empty() {
        return Some(format!("{} 은(는) 작업 폴더에 없다. 아직 만들지 않았다", gone.join(", ")));
    }
    if let Some(why) = screen_contradicts(task, said) {
        return Some(why);
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
    //: Nor a `check`: its result repeats the plan's words, which the model wrote before anything ran.
    let seen: String = task.steps.iter().filter(|s| s.tool != "done" && s.tool != "check")
        .map(|s| s.observed.as_str())
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
    //: **A count, of any length, printed whole somewhere** — `--bin programs` had a BOM exported
    //: and the task close on *총 47개의 부품* without reading it; there were 11, and two digits pass
    //: the check above. A count a model made by looking is the one number it cannot be trusted to
    //: have, so a count in the closing words must stand as a whole number in what was run or read.
    let whole: Vec<String> = digit_runs(&seen);
    let uncounted: Vec<String> = counts(said).into_iter().filter(|n| !whole.contains(n)).collect();
    if !uncounted.is_empty() {
        return Some(format!("{} 은(는) 센 개수인데 어떤 도구 결과에도 그대로 나온 적이 없다. 명령으로 세고 \
(예: python 으로 len) 그 출력으로 확인하라", uncounted.iter().map(|n| format!("{n}개")).collect::<Vec<_>>().join(", ")));
    }
    None
}

/// The input fields of the newest numbered list in the record — the shape a look prints,
/// `N: role "name" = "value"` — as (name, value), the value empty when the line shows none. A
/// result with no list in it is passed over, so a look that read nothing leaves the last one known.
fn last_fields(steps: &[Step]) -> Vec<(String, String)> {
    const INPUTS: [&str; 6] = ["edit", "combo box", "document", "편집", "콤보 상자", "문서"];
    for s in steps.iter().rev() {
        let mut fields = Vec::new();
        let mut listed = false;
        for line in s.observed.lines() {
            let Some((num, rest)) = line.split_once(": ") else { continue };
            let Some((role, quoted)) = rest.split_once(" \"") else { continue };
            if num.trim().parse::<usize>().is_err() {
                continue;
            }
            listed = true;
            if !INPUTS.contains(&role.trim()) {
                continue;
            }
            let (name, value) = match quoted.split_once("\" = \"") {
                Some((n, v)) => (n, v.rsplit_once('"').map(|(v, _)| v).unwrap_or(v)),
                None => (quoted.split('"').next().unwrap_or(""), ""),
            };
            fields.push((name.trim().trim_end_matches(':').trim().to_string(), value.to_string()));
        }
        if listed {
            return fields;
        }
    }
    Vec::new()
}

/// What a request names to be put somewhere, word for word: a quoted piece (*'A-1042'*, 「…」) or a
/// symbol (★, →, an emoji) — the things a closing sentence can say went in, and a field can be read
/// for.
fn things_asked_for(what: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for (open, close) in [('\'', '\''), ('"', '"'), ('「', '」'), ('『', '』'), ('‘', '’'), ('“', '”')] {
        let mut rest = what;
        while let Some(a) = rest.find(open) {
            let after = &rest[a + open.len_utf8()..];
            let Some(b) = after.find(close) else { break };
            let piece = after[..b].trim();
            if (1..=40).contains(&piece.chars().count()) && !out.iter().any(|o| o == piece) {
                out.push(piece.to_string());
            }
            rest = &after[b + close.len_utf8()..];
        }
    }
    for c in what.chars() {
        let symbol = (0x2190..=0x2BFF).contains(&(c as u32)) || (0x1F000..=0x1FAFF).contains(&(c as u32));
        if symbol && !out.iter().any(|o| o == &c.to_string()) {
            out.push(c.to_string());
        }
    }
    out
}

/// **A finish the screen contradicts** — since 2026-11-09, from `--bin screen`'s *charmap*: the
/// task closed on *★ 를 'Characters to copy' 칸에 넣었습니다* with the newest look reading that box as
/// `!!`, twice. When the closing words name fields of the last list and something the request asked
/// to be put somewhere, one of those fields must hold it. A closing that says it could **not** is
/// left alone — refusing an honest failure is the wrong direction. Fitted to that case.
fn screen_contradicts(task: &Task, said: &str) -> Option<String> {
    const NOT: [&str; 6] = ["못", "없", "않", "실패", "안 됐", "안됐"];
    if NOT.iter().any(|n| said.contains(n)) {
        return None;
    }
    let fields = last_fields(&task.steps);
    let named: Vec<&(String, String)> = fields.iter()
        .filter(|(n, _)| n.chars().count() >= 2 && said.contains(n.as_str())).collect();
    if named.is_empty() {
        return None;
    }
    let claimed = things_asked_for(&task.what).into_iter().filter(|t| said.contains(t.as_str()));
    for t in claimed {
        if !named.iter().any(|(_, v)| v.contains(t.as_str())) {
            let shown: Vec<String> = named.iter().map(|(n, v)| format!("'{n}' = \"{v}\"")).collect();
            return Some(format!("끝맺음은 {t} 이(가) 들어갔다고 하지만 마지막으로 본 화면에서는 {} 이다. 화면에서 \
그 칸의 값으로 확인한 다음에 끝내라", shown.join(", ")));
        }
    }
    None
}

/// Every run of digits, grouping commas removed, as written.
fn digit_runs(text: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut cur = String::new();
    let chars: Vec<char> = text.chars().collect();
    for (i, &c) in chars.iter().enumerate() {
        if c.is_ascii_digit() {
            cur.push(c);
        } else if c == ',' && !cur.is_empty() && chars.get(i + 1).is_some_and(|n| n.is_ascii_digit()) {
            continue;
        } else if !cur.is_empty() {
            out.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// **Counts** — a number with a counting word after it: *11개*, *37 건*, *5명*. Not a measure (*10mm*),
/// not a date, not an ordinal (*3번째*, *2단계*).
pub fn counts(text: &str) -> Vec<String> {
    const COUNTERS: [&str; 9] = ["개", "건", "명", "줄", "곳", "권", "행", "종류", "가지"];
    let chars: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        if !chars[i].is_ascii_digit() || (i > 0 && (chars[i - 1].is_ascii_digit() || chars[i - 1] == '.')) {
            i += 1;
            continue;
        }
        let mut j = i;
        let mut n = String::new();
        while j < chars.len() && (chars[j].is_ascii_digit() || (chars[j] == ',' && chars.get(j + 1).is_some_and(|c| c.is_ascii_digit()))) {
            if chars[j] != ',' {
                n.push(chars[j]);
            }
            j += 1;
        }
        let after: String = chars[j..].iter().skip_while(|c| **c == ' ').take(2).collect();
        if chars.get(j) != Some(&'.') && COUNTERS.iter().any(|w| after.starts_with(w)) && !after.starts_with("개월")
            && !out.contains(&n) {
            out.push(n);
        }
        i = j;
    }
    out
}

/// **One step**: ask, read, run, record — and stop the task when the program says so. Returns
/// whether the task is still going.
pub fn step(task: &mut Task, model: &mut dyn Speaks, tools: &mut dyn Tools, today: &str) -> bool {
    if task.ended.is_some() {
        return false;
    }
    //: A `check` is bookkeeping, not work, so it does not spend a step.
    if task.steps.iter().filter(|s| s.tool != "check").count() >= MOST_STEPS {
        task.ended = Some(Ended::Stopped(format!("{MOST_STEPS}단계를 넘겨 멈췄습니다")));
        return false;
    }
    if task.planning && !task.planned {
        task.planned = true;
        task.plan = read_plan(&model.ask(&ask_for_a_plan(task, &tools.list(), today)));
        return true;
    }
    let prompt = ask_for_a_step(task, &tools.list(), today);
    let reply = if !task.sparing {
        model.ask(&prompt)
    } else if reasons_here(task) {
        model.ask_reasoning(&prompt, true)
    } else {
        //: **A finish is decided twice**: asked quickly, and if the quick answer is *done*, asked
        //: again with reasoning on, and the second answer is the one that counts. Ending is the one
        //: step that cannot be taken back.
        let quick = model.ask_reasoning(&prompt, false);
        if matches!(read_step(&quick), Next::Done(_)) { model.ask_reasoning(&prompt, true) } else { quick }
    };
    match read_step(&reply) {
        Next::Done(said) => {
            task.unreadable = 0;
            let Some(why) = why_not_done(task, &said, tools, today) else {
                //: What was taken out of the plan goes with the result, in the model's words.
                let out: Vec<String> = task.plan.iter().filter_map(|i| Some(format!("{} ({})", i.what, i.skipped.as_ref()?)))
                    .collect();
                task.ended = Some(Ended::Done(if out.is_empty() { said }
                                              else { format!("{said}\n(계획에서 뺀 것: {})", out.join("; ")) }));
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
            let (observed, failed) = if tool == "check" && !task.plan.is_empty() {
                check_item(task, &args)
            } else if !known {
                (format!("그런 도구는 없다: {tool}"), true)
            } else if args.to_string().contains("자 생략)") || copies_a_cut(&task.steps, &args) {
                //: **A cut result copied into a call** — since 2026-11-08, from `--bin screen`: a font
                //: list read through PowerShell was written to `fonts.txt` as the record showed it,
                //: *…(앞 5521자 생략)* and the tail, and Arial was in the part cut. The record is cut
                //: for the prompt, never for the work, so the command's own output goes to the file.
                //: **And without the marker** (2026-11-09, the same case on the next run): the tail
                //: was copied from where it starts, *egoesc.ttf* — half of `segoesc.ttf` — and on.
                (CUT_COPIED.to_string(), true)
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

/// **Where reasoning is worth its wait**, for a task that spares it: the model's first step — the
/// one that decides how the work goes — and the step after anything failed, where the error has to
/// be read rather than glanced at. Every other step is asked quickly, and a quick *done* is asked
/// again with reasoning (see [`step`]).
pub fn reasons_here(task: &Task) -> bool {
    task.steps.len() <= task.opened || task.steps.last().is_some_and(|s| s.failed)
}

/// **The calls the program makes before the model's first step** — [`Tools::opening`], run and
/// recorded as steps. Once, on a task with no steps yet; the application calls it as `run` does.
pub fn open(task: &mut Task, tools: &mut dyn Tools) {
    if !task.steps.is_empty() {
        return;
    }
    for (tool, args) in tools.opening(&task.what) {
        let (observed, failed) = match tools.call(&tool, &args) {
            Ok(o) => (cut(&o, OBSERVED_CHARS), false),
            Err(e) => (cut(&format!("오류: {e}"), OBSERVED_CHARS), true),
        };
        task.steps.push(Step { tool, args, observed, failed });
    }
    task.opened = task.steps.len();
}

/// Run a task to its end — the program's end or the model's.
pub fn run(task: &mut Task, model: &mut dyn Speaks, tools: &mut dyn Tools, today: &str,
           carry_on: &dyn Fn() -> bool) {
    open(task, tools);
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

    /// Tools that ask for one file to be read before anything else.
    struct Opens(Box_);
    impl Tools for Opens {
        fn list(&self) -> Vec<ToolSpec> { self.0.list() }
        fn call(&mut self, tool: &str, args: &Value) -> Result<String, String> { self.0.call(tool, args) }
        fn opening(&self, what: &str) -> Vec<(String, Value)> {
            if what.contains("plan.docx") { vec![("read_file".into(), json!({"path": "plan.docx"}))] } else { vec![] }
        }
    }

    #[test]
    fn what_the_request_names_is_read_before_the_first_step() {
        let mut files = Box_::default();
        files.files.insert("plan.docx".into(), "제2조 금액 5,000,000원".into());
        let mut tools = Opens(files);
        let mut task = Task::new("plan.docx 의 금액을 고쳐줘");
        let mut model = Scripted(vec![r#"{"done": "고칠 금액이 5,000,000원으로 적혀 있습니다"}"#]);
        run(&mut task, &mut model, &mut tools, "2026-11-06", &|| true);
        assert_eq!(task.steps[0].tool, "read_file");
        assert_eq!(task.steps[0].observed, "제2조 금액 5,000,000원");
        //: Recorded like any other step, so what it read is a source for the finish's numbers.
        assert!(matches!(task.ended, Some(Ended::Done(_))), "{:#?}", task.steps);
        //: And nothing is read for a request that names nothing.
        let mut other = Task::new("메모 정리해줘");
        run(&mut other, &mut Scripted(vec![]), &mut tools, "2026-11-06", &|| false);
        assert!(other.steps.is_empty());
    }

    /// Answers from a script and writes down, for each call, whether reasoning was asked for.
    struct Switched(Vec<&'static str>, Vec<Option<bool>>);
    impl Speaks for Switched {
        fn ask(&mut self, _: &str) -> String {
            self.1.push(None);
            if self.0.is_empty() { String::new() } else { self.0.remove(0).to_string() }
        }
        fn ask_reasoning(&mut self, p: &str, reason: bool) -> String {
            let reply = self.ask(p);
            *self.1.last_mut().unwrap() = Some(reason);
            reply
        }
    }

    #[test]
    fn a_sparing_task_reasons_first_after_a_failure_and_before_it_ends() {
        let mut model = Switched(vec![
            r#"{"tool": "read_file", "args": {"path": "nope"}}"#,
            r#"{"tool": "write_file", "args": {"path": "a.py", "content": "print(55)"}}"#,
            r#"{"tool": "run", "args": {"command": "python3 a.py"}}"#,
            //: Quick, and *done* — so asked again with reasoning, and that answer counts.
            r#"{"done": "끝"}"#,
            r#"{"done": "a.py 를 실행해 55가 나왔습니다"}"#,
        ], Vec::new());
        let mut tools = Box_::default();
        let mut t = Task::new("1부터 10까지 더해 줘");
        t.sparing = true;
        run(&mut t, &mut model, &mut tools, "오늘", &|| true);
        assert_eq!(model.1, vec![Some(true), Some(true), Some(false), Some(false), Some(true)]);
        assert!(matches!(&t.ended, Some(Ended::Done(s)) if s.contains("55")), "{:?}", t.ended);
        //: And a task that does not spare asks as the model is set, every time — as measured before.
        let mut model = Switched(vec![r#"{"tool": "run", "args": {"command": "python3 a.py"}}"#,
                                      r#"{"done": "55"}"#], Vec::new());
        let mut t = Task::new("x");
        run(&mut t, &mut model, &mut tools, "오늘", &|| true);
        assert_eq!(model.1, vec![None, None]);
    }

    #[test]
    fn the_first_model_step_reasons_even_after_the_program_read_first() {
        let mut files = Box_::default();
        files.files.insert("plan.docx".into(), "금액 5,000,000원".into());
        let mut t = Task::new("plan.docx 고쳐줘");
        t.sparing = true;
        open(&mut t, &mut Opens(files));
        assert_eq!(t.opened, 1);
        assert!(reasons_here(&t));
        t.steps.push(Step { tool: "run".into(), args: json!({}), observed: "ok".into(), failed: false });
        assert!(!reasons_here(&t));
    }

    /// A page's list longer than the record keeps is ended where the record would have cut it —
    /// from the front, its first numbers — and a number past the end is refused, not pressed.
    #[test]
    fn a_long_page_list_is_ended_inside_the_record_and_its_tail_has_no_numbers() {
        let mut hands = Pressing::new(
            || Ok((0..150).map(|i| ("link".to_string(), format!("기사 제목이 꽤 긴 링크 {i}번"))).collect()),
            |_| "실행".to_string());
        let seen = hands.call("look_pane", &json!({})).unwrap();
        assert!(seen.chars().count() <= OBSERVED_CHARS, "{}", seen.chars().count());
        assert!(seen.starts_with("웹 칸에서 누를 수 있는 것") && seen.contains("0: link") && seen.contains("번호가 없다"));
        let last: usize = seen.lines().filter_map(|l| l.split(':').next()?.parse().ok()).max().unwrap();
        assert!(hands.call("click", &json!({"item": last})).is_ok());
        assert!(hands.call("click", &json!({"item": 149})).is_err());
    }

    #[test]
    fn a_press_names_what_the_last_look_showed_and_nothing_else() {
        let pressed = std::rc::Rc::new(std::cell::RefCell::new(Vec::<crate::intent::Effect>::new()));
        let log = pressed.clone();
        let mut hands = Pressing::new(
            || Ok(vec![("textbox".to_string(), "주문번호".to_string()), ("button".to_string(), "조회".to_string())]),
            move |e| { log.borrow_mut().push(e); "실행 — 화면이 바뀜".to_string() });
        //: No look yet: refused, and nothing pressed.
        assert!(hands.call("click", &json!({"item": 1})).is_err());
        let seen = hands.call("look_pane", &json!({})).unwrap();
        assert!(seen.contains("1: button \"조회\""), "{seen}");
        assert_eq!(seen.lines().skip(1).collect::<Vec<_>>(), ["0: textbox \"주문번호\"", "1: button \"조회\""], "a role and a name, never a coordinate");
        let after = hands.call("click", &json!({"item": "1"})).unwrap();
        //: The press carries the new list, so its numbers are the ones held — no look between.
        assert!(after.contains("1: button \"조회\""), "{after}");
        assert!(hands.call("click", &json!({"item": 0})).is_ok());
        assert!(hands.call("click", &json!({"item": 7})).is_err_and(|e| e.contains("마지막 목록에 없다")));
        hands.call("type_text", &json!({"text": "A-1042"})).unwrap();
        let p = pressed.borrow();
        assert_eq!((p[0].kind.as_str(), p[0].target.as_str(), p[0].role.as_str()), ("click", "조회", "button"));
        assert_eq!((p[1].kind.as_str(), p[1].target.as_str()), ("click", "주문번호"));
        assert_eq!((p[2].kind.as_str(), p[2].text.as_str()), ("type", "A-1042"));
        assert!(p.iter().all(|e| e.because_asked == Some(true)), "a task is something the person asked for");
        //: **And work argmind set itself presses as its own idea**, gated on 스스로 실행.
        let own = std::rc::Rc::new(std::cell::RefCell::new(Vec::<crate::intent::Effect>::new()));
        let log = own.clone();
        let mut hands = Pressing::own(|| Ok(vec![("button".to_string(), "조회".to_string())]),
                                      move |e| { log.borrow_mut().push(e); "실행".to_string() });
        hands.call("look_pane", &json!({})).unwrap();
        hands.call("click", &json!({"item": 0})).unwrap();
        assert_eq!(own.borrow()[0].because_asked, Some(false));
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
    fn the_screen_is_listed_last_and_opened_only_when_asked_twice_or_after_code() {
        struct Screen;
        impl Tools for Screen {
            fn list(&self) -> Vec<ToolSpec> {
                vec![ToolSpec { name: "screen_open".into(), does: "연다".into(), args: json!({}) }]
            }
            fn call(&mut self, _: &str, _: &Value) -> Result<String, String> { Ok("열렸다".into()) }
        }
        let made = || Tiered::new(vec![(Rank::Screen, Box::new(Screen) as Box<dyn Tools + Send>),
                                       (Rank::Code, Box::new(Box_::default()))]);
        let mut t = made();
        let names: Vec<String> = t.list().iter().map(|s| s.name.clone()).collect();
        assert_eq!(names.last().map(String::as_str), Some("screen_open"), "{names:?}");
        assert!(t.list()[0].does.starts_with("(2순위"), "{}", t.list()[0].does);
        assert!(t.list().last().unwrap().does.starts_with("(3순위"));
        //: Nothing tried: refused once, then it goes.
        assert_eq!(t.call("screen_open", &json!({})), Err(SCREEN_LAST.into()));
        //: Not when the person named the work screen.
        let mut asked = made().screen_asked_for("작업 화면에서 문자표 열어줘");
        assert_eq!(asked.call("screen_open", &json!({})), Ok("열렸다".into()));
        let mut other = made().screen_asked_for("글꼴 목록을 fonts.txt 로");
        assert_eq!(other.call("screen_open", &json!({})), Err(SCREEN_LAST.into()));
        assert_eq!(t.call("screen_open", &json!({})), Ok("열렸다".into()));
        //: Reading is not trying; writing is.
        let mut t = made();
        let _ = t.call("read_file", &json!({"path": "a"}));
        assert!(t.call("screen_open", &json!({})).is_err());
        let mut t = made();
        let _ = t.call("write_file", &json!({"path": "a.py", "content": "x"}));
        assert_eq!(t.call("screen_open", &json!({})), Ok("열렸다".into()));
    }

    #[test]
    fn a_cut_result_copied_into_a_call_is_refused_and_not_run() {
        let long = "가".repeat(OBSERVED_CHARS + 50);
        let shown = cut(&long, OBSERVED_CHARS);
        let mut model = Scripted(vec![
            r#"{"tool": "run", "args": {"command": "x"}}"#.to_string().leak(),
            Box::leak(format!(r#"{{"tool": "write_file", "args": {{"path": "a.txt", "content": {}}}}}"#,
                              serde_json::to_string(&shown).unwrap()).into_boxed_str()),
        ]);
        let mut tools = Box_::default();
        let mut t = Task::new("x");
        step(&mut t, &mut model, &mut tools, "오늘");
        let calls = tools.calls;
        step(&mut t, &mut model, &mut tools, "오늘");
        assert_eq!(t.steps[1].observed, CUT_COPIED);
        assert_eq!(tools.calls, calls, "not run");
        assert!(tools.files.is_empty());
    }

    /// `--bin screen`'s *charmap*, 2026-11-09: closed on ★ being in the box the newest look read as
    /// `!!`. Refused; taken with ★ there; an honest *could not* left alone; and the order case — a
    /// field named, nothing quoted or symbolic asked for — untouched.
    #[test]
    fn a_finish_the_last_look_contradicts_is_refused() {
        let look = |copy: &str| format!("\"Select\": 눌렀다. 번호는 아래 새 목록의 것이다.\n작업 화면의 창: 'Character Map'\n\
(번호: 종류 \"이름\" = 값)\n0: text \"Characters to copy :\"\n1: edit \"Characters to copy :\"{copy}\n2: button \"Select\"\n\
3: button \"Copy\" (꺼짐)\n10: edit \"Go to Unicode :\" = \"U+2605\"");
        let asked = "작업 화면에서 문자표(charmap)를 열어서 ★ 문자를 골라 '복사할 문자' 칸에 넣어줘";
        let said = "작업 화면의 'Character Map'에서 ★(U+2605) 문자를 선택하고 'Characters to copy :' 칸에 복사했습니다.";
        let task = |copy: &str, what: &str| {
            let mut t = Task::new(what);
            t.steps.push(Step { tool: "screen_press".into(), args: json!({"item": 2}), observed: look(copy), failed: false });
            //: A look that read nothing passes the last list on.
            t.steps.push(Step { tool: "screen_look".into(), args: json!({}),
                                observed: "작업 화면의 창: \n(번호: 종류 \"이름\" = 값)\n(읽을 수 있는 것이 없다)".into(), failed: false });
            t
        };
        let why = why_not_done(&task(" = \"!!\"", asked), said, &Box_::default(), "오늘");
        assert!(why.as_deref().is_some_and(|w| w.contains("'Characters to copy' = \"!!\"")), "{why:?}");
        assert_eq!(screen_contradicts(&task("", asked), said).is_some(), true, "an empty box holds no ★ either");
        assert_eq!(screen_contradicts(&task(" = \"★\"", asked), said), None);
        assert_eq!(screen_contradicts(&task(" = \"!!\"", asked), "★를 'Characters to copy' 칸에 넣지 못했습니다"), None);
        let order = "주문번호 A-1042 를 조회하고, 나온 결과를 answer.md 에 적어줘";
        let mut t = Task::new(order);
        t.steps.push(Step { tool: "screen_press".into(), args: json!({"item": 2}), failed: false,
                            observed: "\"조회\": 눌렀다.\n1: edit \"주문번호\" = \"A-1042\"\n3: text \"A-1042: 출고 완료\"".into() });
        assert_eq!(screen_contradicts(&t, "주문번호 A-1042의 조회 결과 '출고 완료 — 운송장 6038-2271-9954'를 answer.md에 기록했습니다."), None);
        assert_eq!(things_asked_for(asked), vec!["복사할 문자".to_string(), "★".to_string()]);
    }

    /// The marker left off, as `--bin screen`'s *fonts* wrote it on 2026-11-09: from the half line
    /// the cut fell in, on. A later line of the same listing on its own is still anybody's to use.
    #[test]
    fn a_cut_result_copied_without_its_marker_is_refused_too() {
        let listing: String = (0..400).map(|i| format!("font{i:03}.ttf\r\n")).collect();
        let shown = cut(&listing, OBSERVED_CHARS);
        let tail: String = shown.lines().skip(1).collect::<Vec<_>>().join("\n");
        let one_line = "font399.ttf";
        assert!(shown.contains(one_line));
        for (content, refused) in [(tail.as_str(), true), (one_line, false)] {
            let mut model = Scripted(vec![Box::leak(format!(r#"{{"tool": "write_file", "args": {{"path": "fonts.txt", "content": {}}}}}"#,
                                                            serde_json::to_string(content).unwrap()).into_boxed_str())]);
            let mut t = Task::new("x");
            t.steps.push(Step { tool: "run_windows".into(), args: json!({"command": "ls"}), observed: shown.clone(), failed: false });
            let mut tools = Box_::default();
            step(&mut t, &mut model, &mut tools, "오늘");
            assert_eq!(t.steps[1].observed == CUT_COPIED, refused, "{content:.40}");
            assert_eq!(tools.files.is_empty(), refused);
        }
    }

    #[test]
    fn a_drawings_namespace_is_not_an_invented_number() {
        let mut model = Scripted(vec![
            r#"{"tool": "write_file", "args": {"path": "plate.svg", "content": "<svg xmlns=\"http://www.w3.org/2000/svg\"><circle cx=\"50\" r=\"10\"/></svg>"}}"#,
            r#"{"done": "plate.svg 를 만들었습니다"}"#,
        ]);
        let mut t = Task::new("x");
        run(&mut t, &mut model, &mut Box_::default(), "오늘", &|| true);
        assert!(matches!(t.ended, Some(Ended::Done(_))), "{:?}", t.steps);
        //: And a document is still checked: the same number in a note is refused.
        let mut model = Scripted(vec![
            r#"{"tool": "write_file", "args": {"path": "plan.md", "content": "호텔 2000000원"}}"#,
            r#"{"done": "plan.md 를 만들었습니다"}"#,
            r#"{"done": "plan.md 를 만들었습니다"}"#,
            r#"{"done": "plan.md 를 만들었습니다"}"#,
        ]);
        let mut t = Task::new("x");
        run(&mut t, &mut model, &mut Box_::default(), "오늘", &|| true);
        assert!(!matches!(t.ended, Some(Ended::Done(_))), "{:?}", t.ended);
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
                   "naver.com 열어서 정리해줘", "고마워", "이 파일 열어줘", "나를 대표님이라고 불러",
                   "문서 작성 잘하는 법 알려줘", "파이썬 스크립트는 어떻게 실행해?"] {
            assert!(!asks_for_work(no), "{no}");
        }
    }

    #[test]
    fn a_plan_holds_the_finish_until_each_item_is_quoted() {
        let mut task = Task::planned("sum.py 를 만들어 실행하고 결과를 알려줘");
        let mut model = Scripted(vec![
            r#"{"plan": ["sum.py 를 썼다", "실행해서 합을 얻었다"]}"#,
            r#"{"tool": "write_file", "args": {"path": "sum.py", "content": "print(55)"}}"#,
            r#"{"tool": "run", "args": {"command": "python3 sum.py"}}"#,
            r#"{"done": "합은 55"}"#,
            //: A quotation that is not in any result is refused.
            r#"{"tool": "check", "args": {"item": 2, "evidence": "합은 55 입니다"}}"#,
            r#"{"tool": "check", "args": {"item": 1, "evidence": "썼다"}}"#,
            r#"{"tool": "check", "args": {"item": 2, "evidence": "55"}}"#,
            r#"{"done": "합은 55"}"#,
        ]);
        let mut tools = Box_::default();
        run(&mut task, &mut model, &mut tools, "2026-11-03", &|| true);
        assert_eq!(task.plan.len(), 2);
        assert_eq!(task.ended, Some(Ended::Done("합은 55".into())), "{:#?}", task.steps);
        let refused: Vec<&Step> = task.steps.iter().filter(|s| s.failed).collect();
        assert_eq!(refused.len(), 2, "{refused:#?}");
        assert!(refused[0].observed.contains("1번(sum.py 를 썼다)"), "{}", refused[0].observed);
        assert!(refused[1].observed.contains("어떤 도구 결과에도 없다"), "{}", refused[1].observed);
        assert_eq!(task.plan[1].evidence.as_deref(), Some("55"));
    }

    #[test]
    fn an_item_taken_out_goes_to_the_person_with_its_reason() {
        let mut task = Task::planned("deadline.md 에 적어줘");
        let mut model = Scripted(vec![
            r#"{"plan": ["meeting_notes.md 를 읽었다", "deadline.md 를 썼다"]}"#,
            r#"{"tool": "write_file", "args": {"path": "deadline.md", "content": "21일"}}"#,
            r#"{"tool": "check", "args": {"item": 2, "evidence": "썼다"}}"#,
            //: A skip with no reason is no skip.
            r#"{"tool": "check", "args": {"item": 1, "skip": " "}}"#,
            r#"{"tool": "check", "args": {"item": 1, "skip": "그런 파일이 없다"}}"#,
            r#"{"done": "deadline.md 를 썼다"}"#,
        ]);
        let mut tools = Box_::default();
        run(&mut task, &mut model, &mut tools, "2026-11-03", &|| true);
        assert_eq!(task.ended, Some(Ended::Done(
            "deadline.md 를 썼다\n(계획에서 뺀 것: meeting_notes.md 를 읽었다 (그런 파일이 없다))".into())),
            "{:#?}", task.steps);
    }

    #[test]
    fn a_plan_that_does_not_read_is_no_plan() {
        let mut task = Task::planned("sum.py 를 만들어 줘");
        let mut model = Scripted(vec![
            "계획은 이렇습니다",
            r#"{"tool": "write_file", "args": {"path": "sum.py", "content": "print(55)"}}"#,
            r#"{"done": "sum.py 를 만들었다"}"#,
        ]);
        let mut tools = Box_::default();
        run(&mut task, &mut model, &mut tools, "2026-11-03", &|| true);
        assert!(task.plan.is_empty());
        assert_eq!(task.ended, Some(Ended::Done("sum.py 를 만들었다".into())));
    }

    #[test]
    fn a_tool_named_as_the_key_is_a_call() {
        assert_eq!(read_step(r#"{"ask_person": "마감일이 언제예요?"}"#),
                   Next::Call { tool: "ask_person".into(), args: json!({"value": "마감일이 언제예요?"}) });
        assert_eq!(read_step(r#"{"read_file": {"path": "a.txt"}}"#),
                   Next::Call { tool: "read_file".into(), args: json!({"path": "a.txt"}) });
        //: Two keys, or a plan where a step was asked for, still do not read.
        assert_eq!(read_step(r#"{"a": 1, "b": 2}"#), Next::Unreadable);
        assert_eq!(read_step(r#"{"plan": ["x"]}"#), Next::Unreadable);
        let mut asking = Asking::new(|_: &str| Some("21일".to_string()));
        assert_eq!(asking.call("ask_person", &json!({"value": "언제?"})), Ok("사람의 답: 21일".into()));
    }

    #[test]
    fn the_person_is_asked_at_most_twice() {
        let mut answers = vec![Some("11월 14일".to_string()), None];
        let mut asking = Asking::new(move |_q: &str| if answers.is_empty() { Some("또".into()) } else { answers.remove(0) });
        let q = json!({"question": "마감일이 언제예요?"});
        assert_eq!(asking.call("ask_person", &q), Ok("사람의 답: 11월 14일".into()));
        assert!(asking.call("ask_person", &q).unwrap_err().contains("답이 없었다"));
        assert!(asking.call("ask_person", &q).unwrap_err().contains("이미 2번"));
        assert!(asking.call("ask_person", &json!({})).is_err());
    }

    #[test]
    fn a_question_about_how_is_not_a_request() {
        for q in ["문서 작성 잘하는 법 알려줘", "계획 세우는 거 잘해?", "일정 관리 어떻게 해?",
                  "코드 리뷰가 뭐야?"] {
            assert!(asks_about(q), "{q}");
        }
        for r in ["다음 달 여행 일정 좀 짜줄래?", "이 폴더 파일 목록 정리해줄 수 있어?",
                  "보고서 쓰는 법을 정리해서 문서로 만들어줘", "엑셀 만들 수 있어?", "블로그 글 하나 써줘"] {
            assert!(!asks_about(r), "{r}");
        }
    }

    #[test]
    fn a_count_is_a_number_with_a_counting_word() {
        assert_eq!(counts("총 47개의 부품이 있습니다"), ["47"]);
        assert_eq!(counts("오류: 0개, 경고: 37 개, 1,204건"), ["0", "37", "1204"]);
        //: `개월` is a span of time, not a count.
        assert!(counts("40mm 직육면체, 2026년 11월 9일, 3번째 단계, 2.5개월, 3개월").is_empty());
    }

    struct Nothing;
    impl Tools for Nothing {
        fn list(&self) -> Vec<ToolSpec> { Vec::new() }
        fn call(&mut self, _: &str, _: &Value) -> Result<String, String> { Err(String::new()) }
    }

    #[test]
    fn a_failed_command_is_undone_by_a_program_that_ran_after_it() {
        let step = |tool: &str, failed: bool| Step { tool: tool.into(), args: json!({}), failed,
                                                     observed: "(종료 코드 0)".into() };
        let mut task = Task::new("상자를 part.stl 로");
        task.steps.push(step("run", true));
        task.steps.push(step("blender_python", false));
        assert_eq!(why_not_done(&task, "part.stl 로 저장했습니다", &Nothing, "2026년 11월 9일"), None);
        //: Reading after a failed run is not running it again.
        let mut task = Task::new("상자를 part.stl 로");
        task.steps.push(step("run", true));
        task.steps.push(step("read_file", false));
        assert!(why_not_done(&task, "part.stl 로 저장했습니다", &Nothing, "2026년 11월 9일").is_some());
        //: And a program's own failure counts as the last command failing.
        let mut task = Task::new("상자를 part.stl 로");
        task.steps.push(step("write_file", false));
        task.steps.push(step("blender_python", true));
        assert!(why_not_done(&task, "part.stl 로 저장했습니다", &Nothing, "2026년 11월 9일").is_some());
    }

    #[test]
    fn the_work_screen_named_is_work() {
        assert!(asks_for_work("작업 화면에서 문자표(charmap)를 열어서 ★ 문자를 골라 '복사할 문자' 칸에 넣어줘"));
        assert!(asks_for_work("작업화면에서 계산기 열어줘"));
        assert!(!asks_for_work("작업 화면이 뭐야?"), "a question about it is not work");
    }

    #[test]
    fn a_count_nobody_printed_is_not_taken() {
        let mut task = Task::new("amp.kicad_sch 부품 목록을 parts.csv 로 만들고 몇 개인지 알려줘");
        task.steps.push(Step { tool: "kicad_cli".into(), args: json!(["sch"]), failed: false,
                               observed: "Wrote bill of materials to 'parts.csv'.\n\"C1\",\"10uF\"\n\"R4\",\"1k\"\n(종료 코드 0)".into() });
        let why = why_not_done(&task, "parts.csv 를 만들었습니다. 총 47개의 부품이 있습니다", &Nothing, "2026년 11월 9일");
        assert!(why.as_deref().is_some_and(|w| w.contains("47개")), "{why:?}");
        //: Counted by a command and printed, it is taken.
        task.steps.push(Step { tool: "run".into(), args: json!({"command": "wc"}), failed: false, observed: "11\n(종료 코드 0)".into() });
        assert_eq!(why_not_done(&task, "parts.csv 를 만들었습니다. 총 11개의 부품이 있습니다", &Nothing, "2026년 11월 9일"), None);
        //: And a count of something it did not count but the record printed whole — *Found 37 violations*.
        assert_eq!(why_not_done(&task, "parts.csv 완료, 1개 파일", &Nothing, "2026년 11월 9일"), None);
    }

    #[test]
    fn a_reply_promising_work_a_turn_cannot_do_is_the_seam() {
        //: The three replies it was written from, word for word.
        for said in ["일본 여행 계획을 도와드릴게요! 먼저 항공권과 호텔을 찾아볼까요? 지금은 네이버에서 검색할 수 있어요.",
                     "네이버에서 일본 여행 계획을 도와드릴 수 있어요. 먼저 '일본'을 검색해볼까요?",
                     "잠시만 기다려주세요!"] {
            assert!(promises_work(said), "{said}");
        }
        //: What a turn does, and what a turn says.
        for said in ["검색창을 눌러 드릴게요", "메모장을 열어 볼게요", "저장 버튼을 눌렀습니다", "안녕하세요! 무엇을 도와드릴까요?",
                     "천만에요", "파이썬은 프로그래밍 언어예요", "이 창을 닫을게요"] {
            assert!(!promises_work(said), "{said}");
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
