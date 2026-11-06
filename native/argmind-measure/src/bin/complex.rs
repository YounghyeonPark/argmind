//! **Harder work: several bugs, a spec, a dirty file, tests that must be able to fail.**
//!
//! ```text
//!   cargo run -p argmind-measure --bin complex --release -- qwen3:14b
//! ```
//!
//! From `native/`. **Stop argmind first.** `--bin tasks` asked whether the loop finishes a job of
//! two to five steps; this asks it of jobs a person would actually hand over — more than one fault,
//! a format to meet, a row to leave out, a document edited in three places at once. Each task gets
//! a fresh workspace, the real workbench, and a check the program makes afterwards against the files
//! and against the sandbox, **never against the model's own account**.
//!
//! ## Written before the run
//!
//! Nine tasks, each run **twice** (`REPEATS`), each passing only on what the program sees:
//!
//! 1. **three bugs**: `test_inventory.py` passes in the sandbox, and the test file is byte-identical
//!    — a test edited until it passes is not a fix.
//! 2. **a spec**: `wordcount.py` prints `the: 5`, `cat: 3`, `dog: 2` as its first three lines.
//! 3. **a dirty CSV**: `report.md` holds each category beside its total (식품 50000, 전자 250000,
//!    의류 60000) and names order 5 as left out.
//! 4. **three edits at once**: `README.md`'s non-empty lines are exactly the expected ones.
//! 5. **tests that can fail**: `test_util.py` passes on `util.py`, **and fails on each of two wrong
//!    versions** the harness swaps in — a test that passes everything is not a test.
//! 6. **a refactor**: the functions live in `helpers.py`, `app.py` defines neither, and `app.py`
//!    prints exactly what it printed before.
//! 7. **a merge**: `merged.json` parses to exactly the expected list.
//! 8. **minutes**: three sections in order; each person with their own deadline under 할 일; the
//!    release date under 결정; the budget under 미룬 것.
//! 9. **Windows**: `archive.zip` holds `a.txt` and `b.txt` and not `c.log`.
//!
//! **What it decided**, on the first run's 18: ≥ 12 pass, complex work is in reach; ≤ 6, the loop
//! needs a structural change; between, read the failures. **It read 8 of 18, between** — and the
//! two repeats were identical, so 4 of 9. Read, the failures were six false finishes; the program
//! now refuses a finish after a failed run, looks once at the changed files before taking one, and
//! the workbench refuses a long `python3 -c`. **Those were designed on cases 1–9, so 1–9 are
//! fitted from here on.**
//!
//! ## Written before the second run
//!
//! Three held-out tasks, 10–12, not looked at when the change was made:
//!
//! 10. **lines out of a log**: `errors.txt` is exactly the three ERROR lines, in order.
//! 11. **a config**: `config.ini` is exactly the expected lines, spaces around `=` ignored.
//! 12. **run and keep**: `out.txt` is exactly fifteen lines of fizzbuzz.
//!
//! **The change generalises if at least 2 of 3 held-out tasks pass and none of them ends in a
//! false finish.** On all twelve the old thresholds stand as fractions: ≥ 2/3 in reach, ≤ 1/3
//! structural.
//!
//! And one number beside it that matters more than the count: **said done and failed the check** —
//! a finish that is false is the failure a person cannot see from the reply.
//!
//! ## A plan, and the person as a tool — written 2026-11-03, before either ran
//!
//! `COMPLEX_PLAN=1` makes each task a `Task::planned`: the model lists what must be true before it
//! starts, and `done` is refused until each item has a quotation from a tool's result that the
//! program finds there. `COMPLEX_ASK=1` adds `ask_person`, answered here by a person who says only
//! *알아서 판단해서 해 주세요* — every one of the twelve is fully specified, so any question is one
//! nobody needed.
//!
//! * **The plan stays** only if, thinking on, it passes **at least 10 of 12** (the run without it
//!   read 11; one run each, so a one-task drop is noise and two is not) **with no false finish**.
//! * **Asking is not a nuisance** only if the twelve put **at most 2** questions in all.
//!
//! `--asking` is the other half: three tasks whose facts only the person has — a deadline and who
//! owns it, this week's three to-dos, a dinner's date, time and place. Each passes only if the file
//! holds the person's facts. Run **with** `ask_person` (answering the facts to any question) and
//! **without** it, as the control: without, nothing can know them, so **a pass there means the
//! check is broken**. Asking works if **at least 2 of 3** pass with it and **0 of 3** without; a
//! finish without it is counted apart — that is a task inventing what only the person knew.
//!
//! **`COMPLEX_SPARING=1` reasons only where it counts** (2026-11-06): the model's first step and
//! the step after a failure are asked with reasoning, every other step without, and a quick *done*
//! is asked again with reasoning and the second answer counts — `task::reasons_here`. The same
//! switch is `DOCUMENTS_SPARING=1` in `--bin documents`. Against reasoning on every step, run the
//! same day and the same code: **complex 12 of 12, no false finish, 110 s a task**; **documents 12
//! of 13, one false finish, 71 s**. Written before the sparing run, it **stays** only if
//!
//! * complex reads **at least 11 of 12** and documents **at least 11 of 13** — one run each, so a
//!   one-task drop is noise —
//! * with **at most one false finish across the two**, as reasoning everywhere had, and
//! * a task takes **at most 60% as long**: 66 s on complex and 43 s on documents.
//!
//! The harness prints how many calls reasoned and how long each kind took, and the largest prompt
//! against the context, so *faster* can be read as *fewer reasoned calls* rather than assumed.
//!
//! **It failed every part of the bar but documents' count.** Complex **9 of 12** (a spec printed
//! nothing, the README's three edits stopped at one, a refactor circled on a missing file),
//! documents **12 of 13**, **two false finishes** across the two, and **68 s and 69 s** a task
//! against bars of 66 and 43. *Fewer reasoned calls* is true and did not buy the time: 33 of 81
//! calls reasoned on complex and took **725 s against 76 s for the other 48**, and on documents 47
//! of 82 took 812 s against 58 — a failure and a finish both reason, and documents' work is made of
//! both. `task::Task::sparing` is kept and off (`TASK_SPARES`). **And the context is not the
//! limit**: the largest prompt in either run was **2,067 tokens** of 8,192.
//!
//! **`qwen3:30b-a3b` against `qwen3:14b`** (2026-11-06, written while it downloaded): a mixture of
//! experts, 3 B active of 30 B, Apache-2.0, 19 GB — more than the card's 16 GB, so part of it runs
//! from the CPU, and it cannot share the card with the conversation's 8b the way 14b does: **every
//! turn during a task would move one of them off the card**. So equal is not enough. It replaces 14b
//! only if, in the same reasoning mode as 14b's winning arm, it reads **more** on complex or
//! documents and **no fewer** on the other, with **no more false finishes**, at **no more than 1.5x**
//! 14b's time a task.
//!
//! **It did not, by the half that asks for no fewer**: complex **10 of 12** against 14b's 12 (a
//! spec whose script was never written, the CSV report not naming the row it left out), documents
//! **13 of 13 with no false finish** against 12 and one; false finishes one against one; **157 s and
//! 79 s a task** against 110 and 71, inside 1.5x. One run each, so 13 against 12 and 10 against 12
//! are both within what a rerun moves — which is why *equal* was not enough and is what this reads
//! as. Ollama put **26% of it on the CPU** with nothing else loaded. 14b stays.

use std::path::{Path, PathBuf};
use std::time::Instant;

use argmind_core::contract::Speaks;
use argmind_core::task::{run, Asking, Ended, Joined, Task, Tools};
use argmind_env_ollama::Local;
use argmind_env_windows::workspace::Workspace;

/// One: the first run's two repeats were identical step for step, so a second bought nothing.
const REPEATS: usize = 1;

/// The model, and **what its calls cost**: how many reasoned and how many did not, the seconds in
/// each, and the largest prompt — which says whether a task ever came near the context.
struct Model(Local, Spent);

#[derive(Default)]
struct Spent { reasoned: usize, quick: usize, reasoned_s: f64, quick_s: f64, most_prompt: i64 }

impl Speaks for Model {
    fn ask_reasoning(&mut self, prompt: &str, reason: bool) -> String {
        let was = self.0.think;
        self.0.think = Some(reason);
        let reply = self.ask(prompt);
        self.0.think = was;
        reply
    }
    fn ask(&mut self, prompt: &str) -> String {
        let began = Instant::now();
        let asked = self.0.ask(prompt);
        let secs = began.elapsed().as_secs_f64();
        if self.0.think == Some(true) { self.1.reasoned += 1; self.1.reasoned_s += secs }
        else { self.1.quick += 1; self.1.quick_s += secs }
        if let Ok(r) = &asked {
            self.1.most_prompt = self.1.most_prompt.max(r.prompt_tokens.unwrap_or(0));
        }
        let (reply, why) = asked.map(|r| (r.text, r.reason))
            .unwrap_or_else(|e| (String::new(), format!("오류: {e}")));
        //: `COMPLEX_RAW=1` prints every reply whole — what a step that did not read actually said,
        //: and why the model stopped saying it.
        if std::env::var("COMPLEX_RAW").is_ok() {
            println!("         « {} » ({why})", reply.replace('\n', " ⏎ ").chars().take(400).collect::<String>());
        }
        reply
    }
}

struct Case {
    name: &'static str,
    what: &'static str,
    files: &'static [(&'static str, &'static str)],
    /// **A right answer, written by hand** — the control: `--control` checks that each check passes
    /// on this and fails on the untouched files, so a check that cannot pass, or cannot fail, is
    /// found before a model is blamed for it.
    solved: &'static [(&'static str, &'static str)],
    check: fn(&Path, &Task, &Workspace) -> Result<(), String>,
}

fn read(dir: &Path, f: &str) -> Result<String, String> {
    std::fs::read_to_string(dir.join(f)).map_err(|_| format!("{f} 없음"))
}

fn sandbox(w: &Workspace, cmd: &str) -> (bool, String) {
    match w.run_sandboxed(cmd) {
        Ok(o) => (true, o),
        Err(e) => (false, e),
    }
}

fn need(ok: bool, why: &str) -> Result<(), String> {
    if ok { Ok(()) } else { Err(why.to_string()) }
}

fn lines(s: &str) -> Vec<String> {
    s.lines().map(|l| l.trim_end().to_string()).filter(|l| !l.trim().is_empty()).collect()
}

fn no_commas(s: &str) -> String {
    s.replace(',', "")
}

/// The lines under a `## heading`, up to the next heading.
fn section(doc: &str, heading: &str) -> Option<String> {
    let mut out = None::<Vec<&str>>;
    for l in doc.lines() {
        let t = l.trim();
        if t.starts_with('#') {
            if out.is_some() {
                break;
            }
            if t.trim_start_matches('#').trim() == heading {
                out = Some(Vec::new());
            }
            continue;
        }
        if let Some(v) = out.as_mut() {
            v.push(l);
        }
    }
    out.map(|v| v.join("\n"))
}

const INVENTORY: &str = "def total_value(items):\n    total = 0\n    for name, qty, price in items:\n        total += qty + price\n    return total\n\n\
def low_stock(items, limit):\n    return [name for name, qty, price in items if qty > limit]\n\n\
def restock(items, name, amount):\n    out = []\n    for n, q, p in items:\n        if n == name:\n            q = amount\n        out.append((n, q, p))\n    return out\n";
const TEST_INVENTORY: &str = "from inventory import total_value, low_stock, restock\n\
items = [(\"pen\", 10, 500), (\"note\", 2, 1500), (\"bag\", 1, 30000)]\n\
assert total_value(items) == 10*500 + 2*1500 + 30000, \"total\"\n\
assert low_stock(items, 3) == [\"note\", \"bag\"], \"low\"\n\
assert restock(items, \"note\", 5)[1] == (\"note\", 7, 1500), \"restock\"\n\
print(\"all ok\")\n";

const README: &str = "# 도구 v1.2\n\n## 설치\npip install tool==1.2\n\n## 기능\n- 파일 읽기\n- 파일 쓰기\n\n\
## 알려진 문제\n- 큰 파일에서 느림\n\n## 라이선스\nMIT\n";
const README_AFTER: &str = "# 도구 v1.3\n## 설치\npip install tool==1.3\n## 기능\n- 파일 읽기\n- 파일 쓰기\n- 파일 압축\n\
## 라이선스\nMIT";

const UTIL: &str = "def is_leap(year):\n    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)\n\n\
def clamp(x, lo, hi):\n    return max(lo, min(x, hi))\n";
const UTIL_WRONG: [&str; 2] = [
    "def is_leap(year):\n    return year % 4 == 0\n\ndef clamp(x, lo, hi):\n    return max(lo, min(x, hi))\n",
    "def is_leap(year):\n    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)\n\n\
def clamp(x, lo, hi):\n    return min(x, hi)\n",
];

const APP: &str = "def parse(line):\n    name, score = line.split(\",\")\n    return name.strip(), int(score)\n\n\
def grade(score):\n    if score >= 90:\n        return \"A\"\n    if score >= 80:\n        return \"B\"\n    return \"C\"\n\n\
data = [\"kim, 95\", \"lee, 82\", \"park, 70\"]\nfor line in data:\n    n, s = parse(line)\n    print(n, grade(s))\n";

const NOTES: &str = "11/3 회의 메모\n- 출시일 12월 5일 확정\n- 김: 로그인 화면 디자인 11/15까지\n\
- 이: 결제 모듈 테스트 11/20까지\n- 박: 마케팅 문구 초안 11/10까지\n- 예산 논의는 다음 주로 미룸\n";

fn has_date(line: &str, m: u32, d: u32) -> bool {
    line.contains(&format!("{m}/{d}")) || line.contains(&format!("{m}월 {d}일")) || line.contains(&format!("{m}월{d}일"))
}

const CASES: [Case; 12] = [
    Case { name: "버그 셋",
           what: "inventory.py 에 버그가 여러 개 있어서 test_inventory.py 가 실패해. inventory.py 를 고쳐서 테스트가 통과하게 해줘. 테스트 파일은 고치지 마",
           files: &[("inventory.py", INVENTORY), ("test_inventory.py", TEST_INVENTORY)],
           solved: &[("inventory.py", "def total_value(items):\n    return sum(q * p for _, q, p in items)\n\ndef low_stock(items, limit):\n    return [n for n, q, _ in items if q < limit]\n\ndef restock(items, name, amount):\n    return [(n, q + amount if n == name else q, p) for n, q, p in items]\n")],
           check: |dir, _, w| {
               need(read(dir, "test_inventory.py")? == TEST_INVENTORY, "테스트 파일을 고쳤다")?;
               let (ok, o) = sandbox(w, "python3 test_inventory.py");
               need(ok && o.contains("all ok"), &format!("테스트 실패: {}", o.lines().rev().find(|l| !l.trim().is_empty()).unwrap_or("")))
           } },
    Case { name: "명세대로",
           what: "text.txt 에서 가장 많이 나온 단어 3개를 '단어: 횟수' 형식으로 한 줄에 하나씩, 많은 순서로 출력하는 wordcount.py 를 만들고 실행해줘. 대소문자는 구분하지 말고 구두점은 무시해",
           files: &[("text.txt", "The cat and the dog. The cat ran! The dog, a cat, the end.\n")],
           solved: &[("wordcount.py", "import re, collections\nwords = re.findall(r\"[a-z]+\", open(\"text.txt\").read().lower())\nfor w, n in collections.Counter(words).most_common(3):\n    print(f\"{w}: {n}\")\n")],
           check: |_, _, w| {
               let (ok, o) = sandbox(w, "python3 wordcount.py");
               let got: Vec<String> = o.lines().map(|l| l.replace(' ', "").to_lowercase())
                   .filter(|l| l.contains(':')).take(3).collect();
               need(ok && got == ["the:5", "cat:3", "dog:2"], &format!("출력: {got:?}"))
           } },
    Case { name: "지저분한 CSV",
           what: "orders.csv 에서 분류별 금액 합계를 계산해서 report.md 에 표로 정리해줘. 금액이 숫자가 아닌 줄은 계산에서 빼고, 뺀 줄의 주문번호도 report.md 에 적어줘",
           files: &[("orders.csv", "주문번호,분류,금액\n1,식품,12000\n2,전자,250000\n3,식품,8000\n4,의류,45000\n5,전자,오류\n6,의류,15000\n7,식품,30000\n")],
           solved: &[("report.md", "| 분류 | 합계 |\n|---|---|\n| 식품 | 50,000 |\n| 전자 | 250,000 |\n| 의류 | 60,000 |\n\n제외한 주문번호: 5 (금액이 숫자가 아님)\n")],
           check: |dir, _, _| {
               let r = no_commas(&read(dir, "report.md")?);
               for (k, v) in [("식품", "50000"), ("전자", "250000"), ("의류", "60000")] {
                   need(r.lines().any(|l| l.contains(k) && l.split(|c: char| !c.is_ascii_digit()).any(|d| d == v)),
                        &format!("{k} {v} 없음"))?;
               }
               //: **The keyword's line or the three after it** — the first version wanted both on one
               //: line and failed a right report that put 5 under a heading of its own.
               let rows: Vec<&str> = r.lines().collect();
               let keyword = |l: &str| ["제외", "뺀", "빼", "빠", "오류", "숫자가 아닌", "잘못", "유효하지"].iter().any(|w| l.contains(w));
               let five = |l: &str| l.split(|c: char| !c.is_ascii_digit()).any(|d| d == "5");
               need(rows.iter().enumerate().any(|(i, l)| keyword(l) && rows[i..(i + 4).min(rows.len())].iter().any(|m| five(m))),
                    "뺀 주문번호 5 없음")
           } },
    Case { name: "세 군데 고치기",
           what: "README.md 를 고쳐줘: 버전 1.2 를 모두 1.3 으로 바꾸고, 기능 목록 끝에 '- 파일 압축' 을 추가하고, '## 알려진 문제' 섹션은 통째로 지워줘. 나머지는 그대로 둬",
           files: &[("README.md", README)],
           solved: &[("README.md", "# 도구 v1.3\n\n## 설치\npip install tool==1.3\n\n## 기능\n- 파일 읽기\n- 파일 쓰기\n- 파일 압축\n\n## 라이선스\nMIT\n")],
           check: |dir, _, _| {
               let got = lines(&read(dir, "README.md")?);
               need(got == lines(README_AFTER), &format!("{got:?}"))
           } },
    Case { name: "실패할 수 있는 테스트",
           what: "util.py 의 두 함수를 검사하는 test_util.py 를 만들어줘. python3 test_util.py 로 실행했을 때 통과해야 하고, 함수가 틀리면 실패해야 해. 윤년의 100년, 400년 예외와 clamp 의 아래쪽 경계도 검사해",
           files: &[("util.py", UTIL)],
           solved: &[("test_util.py", "from util import is_leap, clamp\nassert is_leap(2024) and not is_leap(1900) and is_leap(2000) and not is_leap(2023)\nassert clamp(5, 0, 10) == 5 and clamp(-3, 0, 10) == 0 and clamp(30, 0, 10) == 10\nprint(\"ok\")\n")],
           check: |dir, _, w| {
               need(dir.join("test_util.py").exists(), "test_util.py 없음")?;
               let (ok, o) = sandbox(w, "python3 test_util.py");
               need(ok, &format!("맞는 구현에서 실패: {}", o.lines().last().unwrap_or("")))?;
               for (i, wrong) in UTIL_WRONG.iter().enumerate() {
                   std::fs::write(dir.join("util.py"), wrong).unwrap();
                   let (ok, _) = sandbox(w, "python3 test_util.py");
                   std::fs::write(dir.join("util.py"), UTIL).unwrap();
                   need(!ok, &format!("틀린 구현 {}번을 통과시켰다", i + 1))?;
               }
               Ok(())
           } },
    Case { name: "나눠 옮기기",
           what: "app.py 의 함수 parse 와 grade 를 helpers.py 로 옮기고, app.py 는 helpers 에서 가져다 쓰게 바꿔줘. 실행 결과는 그대로여야 해",
           files: &[("app.py", APP)],
           solved: &[("helpers.py", "def parse(line):\n    name, score = line.split(\",\")\n    return name.strip(), int(score)\n\ndef grade(score):\n    if score >= 90:\n        return \"A\"\n    if score >= 80:\n        return \"B\"\n    return \"C\"\n"),
                    ("app.py", "from helpers import parse, grade\n\ndata = [\"kim, 95\", \"lee, 82\", \"park, 70\"]\nfor line in data:\n    n, s = parse(line)\n    print(n, grade(s))\n")],
           check: |dir, _, w| {
               let h = read(dir, "helpers.py")?;
               let a = read(dir, "app.py")?;
               need(h.contains("def parse") && h.contains("def grade"), "helpers.py 에 함수 없음")?;
               need(!a.contains("def parse") && !a.contains("def grade") && a.contains("helpers"), "app.py 가 그대로")?;
               let (ok, o) = sandbox(w, "python3 app.py");
               let got: Vec<&str> = o.lines().take_while(|l| !l.starts_with("(종료")).filter(|l| !l.trim().is_empty()).collect();
               need(ok && got == ["kim A", "lee B", "park C"], &format!("출력: {got:?}"))
           } },
    Case { name: "합치기",
           what: "a.json 과 b.json 을 합쳐서 merged.json 을 만들어줘. id 가 겹치면 b.json 쪽을 쓰고, id 순서로 정렬해줘",
           files: &[("a.json", "[{\"id\": 1, \"name\": \"kim\"}, {\"id\": 3, \"name\": \"park\"}]"),
                    ("b.json", "[{\"id\": 2, \"name\": \"lee\"}, {\"id\": 3, \"name\": \"park2\"}]")],
           solved: &[("merged.json", "[{\"id\": 1, \"name\": \"kim\"}, {\"id\": 2, \"name\": \"lee\"}, {\"id\": 3, \"name\": \"park2\"}]")],
           check: |dir, _, _| {
               let got: serde_json::Value = serde_json::from_str(&read(dir, "merged.json")?).map_err(|e| e.to_string())?;
               let want = serde_json::json!([{"id": 1, "name": "kim"}, {"id": 2, "name": "lee"}, {"id": 3, "name": "park2"}]);
               need(got == want, &format!("{got}"))
           } },
    Case { name: "회의록",
           what: "notes.txt 로 회의록 minutes.md 를 써줘. '## 결정', '## 할 일', '## 미룬 것' 세 섹션으로 나누고, 할 일에는 담당자와 마감일을 함께 적어줘",
           files: &[("notes.txt", NOTES)],
           solved: &[("minutes.md", "# 11/3 회의록\n\n## 결정\n- 출시일 12월 5일 확정\n\n## 할 일\n- 김: 로그인 화면 디자인 (11/15까지)\n- 이: 결제 모듈 테스트 (11/20까지)\n- 박: 마케팅 문구 초안 (11/10까지)\n\n## 미룬 것\n- 예산 논의 (다음 주)\n")],
           check: |dir, _, _| {
               let m = read(dir, "minutes.md")?;
               let (Some(d), Some(t), Some(p)) = (section(&m, "결정"), section(&m, "할 일"), section(&m, "미룬 것"))
               else { return Err("섹션이 없다".into()) };
               need(m.find("## 결정") < m.find("## 할 일") && m.find("## 할 일") < m.find("## 미룬 것"), "섹션 순서")?;
               need(has_date(&d, 12, 5), "결정에 출시일 없음")?;
               for (who, mm, dd) in [("김", 11, 15), ("이", 11, 20), ("박", 11, 10)] {
                   need(t.lines().any(|l| l.contains(who) && has_date(l, mm, dd)), &format!("할 일에 {who} {mm}/{dd} 없음"))?;
               }
               need(p.contains("예산"), "미룬 것에 예산 없음")
           } },
    Case { name: "압축 (Windows)",
           what: "작업 폴더의 .txt 파일들만 archive.zip 으로 압축해줘. Windows 명령으로 해",
           files: &[("a.txt", "a"), ("b.txt", "b"), ("c.log", "c")],
           solved: &[],
           check: |dir, _, _| {
               let b = std::fs::read(dir.join("archive.zip")).map_err(|_| "archive.zip 없음".to_string())?;
               let s = String::from_utf8_lossy(&b);
               need(b.starts_with(b"PK") && s.contains("a.txt") && s.contains("b.txt") && !s.contains("c.log"),
                    "zip 내용이 틀림")
           } },
    Case { name: "로그 골라내기 *",
           what: "log.txt 에서 ERROR 가 들어간 줄만 골라 errors.txt 에 저장해줘. 순서는 그대로 둬",
           files: &[("log.txt", "10:00 INFO start\n10:01 ERROR db timeout\n10:02 INFO retry\n10:03 ERROR db timeout\n10:04 WARN slow\n10:05 ERROR disk full\n")],
           solved: &[("errors.txt", "10:01 ERROR db timeout\n10:03 ERROR db timeout\n10:05 ERROR disk full\n")],
           check: |dir, _, _| {
               let got = lines(&read(dir, "errors.txt")?);
               need(got == ["10:01 ERROR db timeout", "10:03 ERROR db timeout", "10:05 ERROR disk full"], &format!("{got:?}"))
           } },
    Case { name: "설정 고치기 *",
           what: "config.ini 에서 [db] 의 port 를 5433 으로 바꾸고, [cache] 섹션에 ttl = 60 을 추가해줘. 나머지는 그대로 둬",
           files: &[("config.ini", "[db]\nhost = localhost\nport = 5432\n\n[cache]\nsize = 100\n")],
           solved: &[("config.ini", "[db]\nhost = localhost\nport = 5433\n\n[cache]\nsize = 100\nttl = 60\n")],
           check: |dir, _, _| {
               let got: Vec<String> = lines(&read(dir, "config.ini")?).iter().map(|l| l.replace(' ', "")).collect();
               need(got == ["[db]", "host=localhost", "port=5433", "[cache]", "size=100", "ttl=60"], &format!("{got:?}"))
           } },
    Case { name: "실행하고 남기기 *",
           what: "1부터 15까지 fizzbuzz 를 출력하는 fizzbuzz.py 를 만들고 (3의 배수는 Fizz, 5의 배수는 Buzz, 둘 다면 FizzBuzz), 실행한 결과를 out.txt 에도 저장해줘",
           files: &[],
           solved: &[("out.txt", "1\n2\nFizz\n4\nBuzz\nFizz\n7\n8\nFizz\nBuzz\n11\nFizz\n13\n14\nFizzBuzz\n")],
           check: |dir, _, _| {
               let got = lines(&read(dir, "out.txt")?);
               let want: Vec<String> = (1..=15).map(|i| match (i % 3, i % 5) {
                   (0, 0) => "FizzBuzz".into(), (0, _) => "Fizz".into(), (_, 0) => "Buzz".into(), _ => i.to_string() }).collect();
               need(got == want, &format!("{got:?}"))
           } },
];

fn fresh(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("argmind-complex-{name}-{}",
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Each check on a right answer and on the untouched files. The zip has no hand-written answer —
/// it is made here with PowerShell, the way the task asks.
fn control() {
    println!("\n  대조: 맞는 답에서 통과하고, 손대지 않은 파일에서 실패해야 한다\n");
    let mut broken = 0;
    for (i, case) in CASES.iter().enumerate() {
        let task = Task::new(case.what);
        let untouched = fresh(&format!("control-raw-{i}"));
        let right = fresh(&format!("control-ok-{i}"));
        for (f, body) in case.files {
            std::fs::write(untouched.join(f), body).unwrap();
            std::fs::write(right.join(f), body).unwrap();
        }
        for (f, body) in case.solved {
            std::fs::write(right.join(f), body).unwrap();
        }
        let (wr, wu) = (Workspace::new(right.clone()), Workspace::new(untouched.clone()));
        if case.solved.is_empty() {
            let _ = wr.run_windows("Compress-Archive -Path *.txt -DestinationPath archive.zip");
        }
        let ok = (case.check)(&right, &task, &wr);
        let raw = (case.check)(&untouched, &task, &wu);
        let fine = ok.is_ok() && raw.is_err();
        broken += !fine as usize;
        println!("    {} {}. {:<10} 맞는 답 {:?} / 손대지 않음 {:?}", if fine { "●" } else { "✗" }, i + 1, case.name,
                 ok.err().unwrap_or_else(|| "통과".into()), raw.err().unwrap_or_else(|| "통과".into()));
        let _ = std::fs::remove_dir_all(&untouched);
        let _ = std::fs::remove_dir_all(&right);
    }
    println!("\n    → 대조가 깨진 검사 {broken}개\n");
}

/// Three tasks whose facts only the person has, the answer the person gives, and what the file
/// must hold.
const ASKING: [(&str, &str, &str, &str, &[&str]); 3] = [
    ("deadline", "어제 회의에서 정한 프로젝트 마감일과 담당자를 deadline.md 에 적어줘",
     "마감은 11월 21일, 담당은 박서연이에요", "deadline.md", &["21", "박서연"]),
    ("todo", "내 이번 주 할 일 세 가지를 todo.md 에 체크리스트로 정리해줘",
     "보고서 제출, 치과 예약, 엄마 생신 선물 사기", "todo.md", &["보고서", "치과", "선물"]),
    ("dinner", "우리 팀 회식 공지문을 notice.md 로 써줘",
     "11월 20일 저녁 7시, 강남역 소담 식당이에요", "notice.md", &["20", "소담"]),
];

fn asking(model: &mut Model) {
    println!("  ── 사람만 아는 것: ask_person 이 있을 때와 없을 때 ──");
    let (mut with, mut without, mut invented) = (0, 0, 0);
    for (i, (name, what, answer, file, must)) in ASKING.iter().enumerate() {
        for ask in [true, false] {
            let dir = fresh(&format!("asking-{i}-{ask}"));
            let asked = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let mut sets: Vec<Box<dyn Tools + Send>> = vec![Box::new(Workspace::new(dir.clone()))];
            if ask {
                let (answer, asked) = (answer.to_string(), asked.clone());
                sets.push(Box::new(Asking::new(move |q: &str| {
                    asked.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    println!("         ? {q}");
                    Some(answer.clone())
                })));
            }
            let mut bench = Joined(sets);
            let mut task = if std::env::var("COMPLEX_PLAN").is_ok() { Task::planned(what) } else { Task::new(what) };
            run(&mut task, model, &mut bench, "2026년 11월 3일", &|| true);
            let body = std::fs::read_to_string(dir.join(file)).unwrap_or_default();
            let pass = !body.is_empty() && must.iter().all(|m| body.contains(m));
            let done = matches!(task.ended, Some(Ended::Done(_)));
            if ask { with += pass as usize } else {
                without += pass as usize;
                invented += (done && !pass) as usize;
            }
            println!("    {} {name:<9} {} 물음 {} · {} 단계 · {} — {}", if pass { "●" } else { "○" },
                     if ask { "ask 있음" } else { "ask 없음" }, asked.load(std::sync::atomic::Ordering::SeqCst),
                     task.steps.len(), match &task.ended { Some(Ended::Done(_)) => "끝".to_string(),
                         Some(Ended::Stopped(w)) => format!("멈춤({})", w.chars().take(40).collect::<String>()), None => "?".into() },
                     body.replace('\n', " ⏎ ").chars().take(120).collect::<String>());
            let _ = std::fs::remove_dir_all(&dir);
        }
    }
    println!("\n    → ask 있음 {with}/3 · ask 없음 {without}/3, 거기서 지어내고 끝낸 것 {invented}");
    println!("    ── 돌리기 전에 적어둔 대로: {} ──\n",
             if without > 0 { "대조가 깨졌다 — 검사를 고쳐야 한다" }
             else if with >= 2 { "묻기가 된다" } else { "묻기가 되지 않는다" });
}

fn main() {
    if std::env::args().any(|a| a == "--control") {
        return control();
    }
    if std::env::args().any(|a| a == "--asking") {
        let mut local = Local::new("qwen3:14b");
        local.think = Some(std::env::var("COMPLEX_THINK").is_ok());
        local.keep_alive = "300s".into();
        println!("\n  harness {}\n", argmind_measure::source_print(include_str!("complex.rs")));
        return asking(&mut Model(local, Spent::default()));
    }
    let models: Vec<String> = std::env::args().skip(1).filter(|a| !a.starts_with("--")).collect();
    let models = if models.is_empty() { vec!["qwen3:14b".to_string()] } else { models };
    println!("\n  복잡한 작업 — 버그 여럿, 명세, 지저분한 자료, 실패할 수 있는 테스트");
    println!("  harness {}\n", argmind_measure::source_print(include_str!("complex.rs")));
    for name in &models {
        let mut local = Local::new(name);
        //: `COMPLEX_THINK=1` lets the model reason before each step — slower, and the one lever
        //: that needs no download.
        local.think = Some(std::env::var("COMPLEX_THINK").is_ok());
        local.keep_alive = "300s".into();
        for _ in 0..2 {
            let _ = local.ask("Reply with the single word: ready");
        }
        let mut model = Model(local, Spent::default());
        println!("  ── {name} ──");
        let (mut passed, mut false_done, mut runs, mut secs_all) = (0, 0, 0, 0.0);
        let (mut fitted, mut held, mut held_false) = (0, 0, 0);
        let questions = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        for (i, case) in CASES.iter().enumerate() {
            for r in 0..REPEATS {
                let dir = fresh(&format!("{}-{i}-{r}", name.replace(':', "_")));
                for (f, body) in case.files {
                    std::fs::write(dir.join(f), body).unwrap();
                }
                let bench = Workspace::new(dir.clone());
                let mut sets: Vec<Box<dyn Tools + Send>> = vec![Box::new(Workspace::new(dir.clone()))];
                if std::env::var("COMPLEX_ASK").is_ok() {
                    let asked = questions.clone();
                    sets.push(Box::new(Asking::new(move |q: &str| {
                        asked.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                        println!("         ? {q}");
                        Some("알아서 판단해서 해 주세요".to_string())
                    })));
                }
                let mut tools = Joined(sets);
                let mut task = if std::env::var("COMPLEX_PLAN").is_ok() { Task::planned(case.what) } else { Task::new(case.what) };
                task.sparing = std::env::var("COMPLEX_SPARING").is_ok();
                let began = Instant::now();
                run(&mut task, &mut model, &mut tools, "2026년 11월 3일", &|| true);
                let secs = began.elapsed().as_secs_f64();
                secs_all += secs;
                let verdict = (case.check)(&dir, &task, &bench);
                let said_done = matches!(task.ended, Some(Ended::Done(_)));
                runs += 1;
                passed += verdict.is_ok() as usize;
                false_done += (said_done && verdict.is_err()) as usize;
                if i < 9 {
                    fitted += verdict.is_ok() as usize;
                } else {
                    held += verdict.is_ok() as usize;
                    held_false += (said_done && verdict.is_err()) as usize;
                }
                let ended = match &task.ended {
                    Some(Ended::Done(_)) => "끝".to_string(),
                    Some(Ended::Stopped(w)) => format!("멈춤({})", w.chars().take(40).collect::<String>()),
                    None => "?".into(),
                };
                let tools: Vec<String> = task.steps.iter()
                    .map(|s| format!("{}{}", s.tool, if s.failed { "✗" } else { "" })).collect();
                println!("    {} {}.{} {:<10} {:>2}단계 {secs:>4.0}초 {ended}{}", if verdict.is_ok() { "●" } else { "○" },
                         i + 1, r + 1, case.name, task.steps.len(),
                         match &verdict { Err(why) => format!(" — {}", why.chars().take(90).collect::<String>()), Ok(()) => String::new() });
                println!("         [{}]", tools.join(" "));
                if !task.plan.is_empty() {
                    for (n, item) in task.plan.iter().enumerate() {
                        println!("         {} {}. {} {}", if item.evidence.is_some() { "☑" } else if item.skipped.is_some() { "⊘" } else { "☐" },
                                 n + 1, item.what,
                                 item.evidence.as_deref().map(|e| format!("— 「{}」", e.chars().take(50).collect::<String>()))
                                     .or(item.skipped.as_deref().map(|w| format!("— 뺌: {w}"))).unwrap_or_default());
                    }
                }
                if std::env::var("COMPLEX_VERBOSE").is_ok() {
                    for s in &task.steps {
                        println!("         · {} {} → {}", s.tool,
                                 s.args.to_string().chars().take(220).collect::<String>(),
                                 s.observed.replace('\n', " ⏎ ").chars().take(220).collect::<String>());
                    }
                }
                let _ = std::fs::remove_dir_all(&dir);
            }
        }
        println!("\n    → {passed}/{runs} 통과, 끝났다고 하고 틀린 것 {false_done}, 평균 {:.0}초",
                 secs_all / runs as f64);
        println!("      맞춘 것(1–9) {}/9 · 따로 둔 것(10–12) {}/3, 거기서 끝났다고 하고 틀린 것 {}",
                 fitted, held, held_false);
        let m = &model.1;
        println!("      모델 호출: 추론 {}번 {:.0}초 · 추론 없이 {}번 {:.0}초 · 가장 긴 프롬프트 {} 토큰 (컨텍스트 {})",
                 m.reasoned, m.reasoned_s, m.quick, m.quick_s, m.most_prompt, model.0.ctx);
        if std::env::var("COMPLEX_PLAN").is_ok() {
            println!("    ── 계획, 돌리기 전에 적어둔 대로: {} ──",
                     if passed >= 10 && false_done == 0 { "계획을 둔다" } else { "계획을 두지 않는다" });
        }
        if std::env::var("COMPLEX_ASK").is_ok() {
            let q = questions.load(std::sync::atomic::Ordering::SeqCst);
            println!("    ── 묻기, 돌리기 전에 적어둔 대로: 필요 없는 물음 {q}개 — {} ──",
                     if q <= 2 { "성가시지 않다" } else { "성가시다" });
        }
        println!("    ── 돌리기 전에 적어둔 대로: {} · 따로 둔 셋: {} ──\n",
                 if 3 * passed >= 2 * runs { "지금의 작업 반복으로 복잡한 일이 된다" }
                 else if 3 * passed <= runs { "작업 반복 구조를 바꿔야 한다" }
                 else { "실패를 읽어야 한다 — 어느 결론도 아니다" },
                 if held >= 2 && held_false == 0 { "바꾼 것이 일반화된다" } else { "일반화된다고 할 수 없다" });
    }
}
