//! **The owner's order of ways in, held by a task — and the screen when nothing else will do.**
//!
//! ```text
//!   cargo build --release -p argmind-env-windows --example order-window
//!   cargo build --release -p argmind-measure --bin mcp
//!   cargo run -p argmind-measure --bin screen --release
//! ```
//!
//! From `native/`. **Stop argmind first** — qwen3:14b with reasoning on, the application's task
//! model. The owner, 2026-11-08: *작업 수행시 접근 우선순위는 mcp또는 api, 코드또는 스크립트 작성및
//! 실행, 그래도 안되면 샌드박스작업 수행.* Every task here gets all three, ranked as the application
//! ranks them (`task::Tiered`): the stand-in CAD server from `--bin mcp` where there is one, the real
//! workbench, and `argmind_env_windows::screen` — a desktop of argmind's own.
//!
//! ## Written before the run
//!
//! 1. **order** — `order-window.exe` draws an order's status only after 조회 is pressed, and nothing
//!    else holds it, so only the screen gets there. Passes when `answer.md` holds the waybill
//!    `6038-2271-9954` **and** the person's foreground window and physical pointer are what they
//!    were before the task.
//! 2. **plate**, with the CAD server: passes when `plate.svg` is right, a `cad__` call made it, and
//!    **no `screen_open` succeeded**.
//! 3. **fonts**, no server: *설치된 글꼴 목록을 fonts.txt 에 저장해줘* — Character Map is right there
//!    and so is PowerShell. Passes when `fonts.txt` names Arial or 맑은 고딕 and **no `screen_open`
//!    succeeded**.
//!
//! **The order holds** only at 3 of 3. One run each: a demonstration, not a rate. Refusals of the
//! first `screen_open` are counted and printed, since that is the program holding the order rather
//! than the model.
//!
//! **Read, 2026-11-08. First run 2 of 3, and the miss was not the order.** *order* tried
//! `run_windows` first, then opened the window on the work screen, wrote *A-1042*, pressed 조회
//! (once on a number the write had made stale, refused, looked again) and wrote the waybill — 142 s,
//! the person's foreground and physical pointer unchanged. *plate* went to `cad__` four times and
//! never opened a screen. *fonts* used PowerShell and never opened one either — the order held in
//! all three — **and failed on the file**: it wrote the record as shown, *…(앞 5521자 생략)* and the
//! tail, and Arial was in the part cut. So a call carrying a cut result is now refused before it
//! runs (`task::CUT_COPIED`), **fitted to that one case**. **Re-run: 3 of 3** — *fonts* had its
//! copy refused and wrote the command's output to the file itself. **What neither run exercised is
//! the gate on `screen_open`**: every task that opened a screen had tried code first, so the
//! refusal never fired with a model; it is held by `tiered` tests only.

use std::path::{Path, PathBuf};
use std::time::Instant;

use argmind_core::contract::Speaks;
use argmind_core::task::{run, Ended, Rank, Task, Tiered, Tools, SCREEN_LAST};
use argmind_env_mcp::{Configured, Servers};
use argmind_env_ollama::Local;
use argmind_env_windows::screen::Screen;
use argmind_env_windows::workspace::Workspace;
use windows::Win32::Foundation::POINT;
use windows::Win32::UI::WindowsAndMessaging::{GetForegroundWindow, GetPhysicalCursorPos};

struct Model(Local);

impl Speaks for Model {
    fn ask(&mut self, prompt: &str) -> String {
        self.0.ask(prompt).map(|r| r.text).unwrap_or_default()
    }
}

fn plate_ok(dir: &Path) -> bool {
    let Ok(svg) = std::fs::read_to_string(dir.join("plate.svg")) else { return false };
    svg.contains("width=\"100\"") && svg.contains("height=\"60\"") && svg.contains("cx=\"50\"")
        && svg.contains("cy=\"30\"") && svg.contains("r=\"10\"")
}

fn fresh(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("argmind-screen-{name}-{}",
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn pointer() -> (i32, i32) {
    let mut p = POINT::default();
    unsafe { let _ = GetPhysicalCursorPos(&mut p); }
    (p.x, p.y)
}

fn main() {
    let here = std::env::current_exe().unwrap().parent().unwrap().to_path_buf();
    let window = here.join("examples").join("order-window.exe");
    let cad = here.join("mcp.exe");
    assert!(window.exists() && cad.exists(), "build order-window and mcp first — see the top of screen.rs");
    println!("\n  작업 우선순위 — MCP·API, 코드, 그래도 안 되면 작업 화면");
    println!("  harness {}\n", argmind_measure::source_print(include_str!("screen.rs")));
    let mut local = Local::new("qwen3:14b");
    local.think = Some(true);
    local.keep_alive = "300s".into();
    let _ = local.ask("Reply with the single word: ready");
    let mut model = Model(local);

    let cases: [(&str, String, bool); 3] = [
        ("order", format!("주문 조회 프로그램({})을 열어서 주문번호 A-1042 를 조회하고, 나온 결과를 answer.md 에 적어줘",
                          window.display()), false),
        ("plate", "가로 100mm 세로 60mm 판 도면을 그려줘. 바깥은 사각형, 정가운데에 반지름 10mm 구멍(원) 하나. plate.svg 로 저장해줘".into(), true),
        ("fonts", "이 PC에 설치된 글꼴 목록을 fonts.txt 에 저장해줘".into(), false),
    ];
    let mut passed = 0;
    for (name, what, with_cad) in &cases {
        let dir = fresh(name);
        let mut sets: Vec<(Rank, Box<dyn Tools + Send>)> = vec![(Rank::Code, Box::new(Workspace::new(dir.clone()))),
                                                                (Rank::Screen, Box::new(Screen::new()))];
        if *with_cad {
            let c = Configured { name: "cad".into(), command: cad.display().to_string(),
                                 args: vec!["--serve".into(), dir.display().to_string()], env: vec![] };
            let (servers, failed) = Servers::start(&[c]);
            assert!(failed.is_empty(), "{failed:?}");
            sets.push((Rank::Api, Box::new(servers)));
        }
        let mut bench = Tiered::new(sets);
        let mut task = Task::new(what);
        let (p0, f0) = (pointer(), unsafe { GetForegroundWindow() });
        let began = Instant::now();
        run(&mut task, &mut model, &mut bench, "2026년 11월 8일", &|| true);
        let secs = began.elapsed().as_secs_f64();
        drop(bench);
        let untouched = pointer() == p0 && unsafe { GetForegroundWindow() } == f0;
        let opened = task.steps.iter().any(|s| s.tool == "screen_open" && !s.failed);
        let held = task.steps.iter().filter(|s| s.tool == "screen_open" && s.observed.contains(SCREEN_LAST)).count();
        let via_cad = task.steps.iter().any(|s| s.tool.starts_with("cad__") && !s.failed);
        let ok = match *name {
            "order" => std::fs::read_to_string(dir.join("answer.md")).is_ok_and(|s| s.contains("6038-2271-9954")) && untouched,
            "plate" => plate_ok(&dir) && via_cad && !opened,
            _ => std::fs::read(dir.join("fonts.txt")).is_ok_and(|b| {
                let s = String::from_utf8_lossy(&b).replace('\0', "").to_lowercase();
                s.contains("arial") || s.contains("맑은 고딕") || s.contains("malgun")
            }) && !opened,
        };
        passed += ok as usize;
        let ended = match &task.ended {
            Some(Ended::Done(s)) => format!("끝: {}", s.chars().take(90).collect::<String>()),
            Some(Ended::Stopped(w)) => format!("**멈춤**: {w}"),
            None => "?".into(),
        };
        let tools: Vec<String> = task.steps.iter().map(|s| format!("{}{}", s.tool, if s.failed { "✗" } else { "" })).collect();
        println!("    {} {name} — {}단계 {secs:.0}초, 화면 {}, 처음 화면 열기 되돌림 {held}, 사람 화면 {} [{}]\n       {ended}",
                 if ok { "●" } else { "○" }, task.steps.len(), if opened { "열었음" } else { "안 엶" },
                 if untouched { "그대로" } else { "**바뀜**" }, tools.join(" "));
        if std::env::var("SCREEN_VERBOSE").is_ok() {
            for s in &task.steps {
                println!("         · {} {} → {}", s.tool, s.args.to_string().chars().take(160).collect::<String>(),
                         s.observed.replace('\n', " ⏎ ").chars().take(220).collect::<String>());
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
    println!("\n  ── 돌리기 전에 적어둔 대로: {passed}/3 — {} ──",
             if passed == 3 { "순서가 지켜진다" } else { "보여주지 못했다" });
}
