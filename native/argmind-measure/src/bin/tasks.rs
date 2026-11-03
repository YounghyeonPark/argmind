//! **Can argmind carry real work to the end — and which model can?**
//!
//! ```text
//!   cargo run -p argmind-measure --bin tasks --release -- qwen3:8b qwen3:14b
//! ```
//!
//! From `native/`. **Stop argmind first.** Each task gets a fresh workspace under the temp folder,
//! the files it needs, `argmind_core::task::run` with the real workbench — files, the
//! `argmind-sandbox` WSL distro, PowerShell in the folder — and a check the program makes afterwards
//! against the files and the record, never against the model's own account.
//!
//! ## Written before the run
//!
//! Six tasks, each passing only on what the program can see:
//!
//! 1. **sum.py**: the file exists, a `run` in the record printed `5050`, and the closing words say 5050.
//! 2. **summary**: `summary.md` exists, has 2–5 non-empty lines, and names the release date
//!    (*12월 5일*) from `meeting.txt`.
//! 3. **one edit**: `report.md`'s conclusion line is the new sentence and every other line is
//!    byte-identical to the original.
//! 4. **a bug**: after the task the harness itself runs `python3 test_calc.py` in the sandbox and it
//!    passes.
//! 5. **a sum and a mean**: the closing words hold the right total (*1,580,000* in any grouping) and
//!    mean (*316,000*).
//! 6. **Windows**: `files.txt` exists and names the workspace's other files.
//!
//! **The model is chosen by this**: qwen3:14b replaces qwen3:8b only if it passes **at least two
//! more** of the six; otherwise 8b stays for its speed. Steps and seconds are printed beside each.

use std::path::{Path, PathBuf};
use std::time::Instant;

use argmind_core::contract::Speaks;
use argmind_core::task::{run, Ended, Task};
use argmind_env_ollama::Local;
use argmind_env_windows::workspace::Workspace;

struct Model(Local);

impl Speaks for Model {
    fn ask(&mut self, prompt: &str) -> String {
        self.0.ask(prompt).map(|r| r.text).unwrap_or_default()
    }
}

struct Case {
    what: &'static str,
    files: &'static [(&'static str, &'static str)],
    check: fn(&Path, &Task, &Workspace) -> bool,
}

const REPORT: &str = "# 분기 보고\n\n## 진행\n신규 기능 세 가지를 마쳤다.\n\n## 결론\n프로젝트를 잠시 멈춘다.\n\n## 부록\n표 1 참조.\n";

fn closing(t: &Task) -> String {
    match &t.ended {
        Some(Ended::Done(s)) => s.clone(),
        _ => String::new(),
    }
}

fn digits(s: &str) -> String {
    s.chars().filter(|c| c.is_ascii_digit() || c.is_whitespace()).collect::<String>()
        .split_whitespace().collect::<Vec<_>>().join(" ")
}

const CASES: [Case; 6] = [
    Case { what: "1부터 100까지의 합을 계산하는 sum.py 를 만들고 실행해서 결과를 알려줘",
           files: &[],
           check: |dir, t, _| dir.join("sum.py").exists()
               && t.steps.iter().any(|s| s.tool == "run" && s.observed.contains("5050"))
               && closing(t).contains("5050") },
    Case { what: "meeting.txt 를 읽고 회의 내용을 summary.md 에 세 줄로 요약해줘",
           files: &[("meeting.txt", "회의 일시: 11월 2일\n참석: 김, 이, 박\n결정 1: 새 앱의 출시일을 12월 5일로 정했다.\n\
결정 2: 베타 테스트는 11월 20일부터 2주간 한다.\n결정 3: 마케팅 예산은 다음 회의에서 정한다.\n잡담: 점심은 국수였다.\n")],
           check: |dir, _, _| std::fs::read_to_string(dir.join("summary.md")).is_ok_and(|s| {
               let n = s.lines().filter(|l| !l.trim().is_empty()).count();
               (2..=5).contains(&n) && s.contains("12월 5일") }) },
    Case { what: "report.md 의 '## 결론' 아래 문장만 '프로젝트를 1월에 재개한다.' 로 바꿔줘. 다른 부분은 그대로 둬",
           files: &[("report.md", REPORT)],
           check: |dir, _, _| std::fs::read_to_string(dir.join("report.md")).is_ok_and(|s|
               s == REPORT.replace("프로젝트를 잠시 멈춘다.", "프로젝트를 1월에 재개한다.")) },
    Case { what: "calc.py 에 버그가 있어서 test_calc.py 가 실패해. calc.py 를 고쳐서 테스트가 통과하게 해줘",
           files: &[("calc.py", "def add(a, b):\n    return a - b\n\ndef mul(a, b):\n    return a * b\n"),
                    ("test_calc.py", "from calc import add, mul\nassert add(2, 3) == 5, 'add'\nassert mul(2, 3) == 6, 'mul'\nprint('ok')\n")],
           check: |_, _, w| w.run_sandboxed("python3 test_calc.py")
               .is_ok_and(|o| o.contains("ok") && o.contains("종료 코드 0")) },
    Case { what: "sales.csv 의 금액 열의 합계와 평균을 계산하는 스크립트를 만들고 실행해서 결과를 알려줘",
           files: &[("sales.csv", "날짜,품목,금액\n11-01,커피,450000\n11-02,빵,230000\n11-03,커피,510000\n11-04,케이크,120000\n11-05,빵,270000\n")],
           check: |_, t, _| { let c = closing(t); let d = digits(&c.replace(',', ""));
               (d.contains("1580000") || c.contains("158만")) && (d.contains("316000") || c.contains("31만 6")) } },
    Case { what: "작업 폴더에 있는 파일 목록을 Windows 명령으로 만들어서 files.txt 에 저장해줘",
           files: &[("a.txt", "a"), ("b.txt", "b")],
           check: |dir, _, _| std::fs::read(dir.join("files.txt")).is_ok_and(|b| {
               let s = String::from_utf8_lossy(&b).replace('\0', "");
               s.contains("a.txt") && s.contains("b.txt") }) },
];

fn fresh(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("argmind-tasks-{name}-{}",
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn main() {
    let models: Vec<String> = std::env::args().skip(1).collect();
    let models = if models.is_empty() { vec!["qwen3:8b".to_string()] } else { models };
    println!("\n  업무를 끝까지 — 작업 반복 구조, 실제 도구");
    println!("  harness {}\n", argmind_measure::source_print(include_str!("tasks.rs")));
    let mut passed_by = Vec::new();
    for name in &models {
        let mut local = Local::new(name);
        local.think = Some(false);
        local.keep_alive = "300s".into();
        for _ in 0..2 {
            let _ = local.ask("Reply with the single word: ready");
        }
        let mut model = Model(local);
        println!("  ── {name} ──");
        let mut passed = 0;
        for (i, case) in CASES.iter().enumerate() {
            let dir = fresh(&format!("{}-{i}", name.replace(':', "_")));
            for (f, body) in case.files {
                std::fs::write(dir.join(f), body).unwrap();
            }
            let mut bench = Workspace::new(dir.clone());
            let mut task = Task::new(case.what);
            let began = Instant::now();
            run(&mut task, &mut model, &mut bench, "2026년 11월 3일", &|| true);
            let secs = began.elapsed().as_secs_f64();
            let ok = (case.check)(&dir, &task, &bench);
            passed += ok as usize;
            let ended = match &task.ended {
                Some(Ended::Done(s)) => format!("끝: {}", s.chars().take(70).collect::<String>()),
                Some(Ended::Stopped(w)) => format!("**멈춤**: {w}"),
                None => "?".into(),
            };
            let tools: Vec<String> = task.steps.iter()
                .map(|s| format!("{}{}", s.tool, if s.failed { "✗" } else { "" })).collect();
            println!("    {} {}. {}단계 {secs:.0}초 [{}]\n       {ended}", if ok { "●" } else { "○" },
                     i + 1, task.steps.len(), tools.join(" "));
            if std::env::var("TASKS_VERBOSE").is_ok() {
                for s in &task.steps {
                    println!("         · {} {} → {}", s.tool,
                             s.args.to_string().chars().take(200).collect::<String>(),
                             s.observed.replace('\n', " ⏎ ").chars().take(200).collect::<String>());
                }
            }
            let _ = std::fs::remove_dir_all(&dir);
        }
        println!("    → {passed}/{}\n", CASES.len());
        passed_by.push((name.clone(), passed));
    }
    if let (Some(a), Some(b)) = (passed_by.iter().find(|p| p.0 == "qwen3:8b"),
                                 passed_by.iter().find(|p| p.0 == "qwen3:14b")) {
        println!("  ── 돌리기 전에 적어둔 대로: 14b {} 대 8b {} — {} ──", b.1, a.1,
                 if b.1 >= a.1 + 2 { "14b 로 바꾼다" } else { "8b 를 유지한다" });
    }
}
