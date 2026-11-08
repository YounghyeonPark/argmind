//! **Installed programs by their own scripting interfaces, used by a task** — Blender's Python and
//! `kicad-cli`, ranked with the APIs (`argmind_env_windows::programs`).
//!
//! ```text
//!   cargo run -p argmind-measure --bin programs --release
//!   cargo run -p argmind-measure --bin programs --release -- --control
//! ```
//!
//! From `native/`. **Stop argmind first** — qwen3:14b with reasoning on, the application's task
//! model. The owner, 2026-11-08, chose *connect a real program* after the work screen, and this
//! machine has Blender 5.1 and KiCad 10 installed. Every task gets what the application gives one —
//! the programs (`Rank::Api`), the workbench (`Rank::Code`) and the work screen (`Rank::Screen`),
//! through `task::Tiered`; `--control` takes the programs away and leaves the rest, so `run_windows`
//! could still find either program on its own.
//!
//! ## Written before the run
//!
//! The KiCad cases work on a copy of KiCad's own demo `ecc83-pp.kicad_sch`, and every expected
//! answer is computed here by running `kicad-cli` on it, never typed in.
//!
//! 1. **bom** — *amp.kicad_sch 의 부품 목록(참조번호와 값)을 parts.csv 로, 부품이 모두 몇 개인지.*
//!    Passes when `parts.csv` names every reference in kicad-cli's own BOM **and** the closing words
//!    hold the count.
//! 2. **erc** — *ERC 를 돌려 결과를 erc.txt 로, 오류와 경고가 각각 몇 개인지.* Passes when
//!    `erc.txt` exists **and** the closing words hold both counts as kicad-cli's report has them.
//! 3. **box** — *가로 40mm 세로 20mm 높이 10mm 직육면체를 part.stl 과 part.blend 로.* Passes when
//!    `part.blend` exists and `part.stl`'s bounding box, sorted, is 4 : 2 : 1 within 1% with its
//!    largest side 40 (millimetres) or 0.04 (metres) — Blender's unit is the metre and STL has
//!    none, so both readings of the request are right.
//!
//! **With the programs: 3 of 3, each through `kicad_cli` or `blender_python`.** One run each, a
//! demonstration and not a rate. The control has no bar; it says whether the tools were needed.
//!
//! **Read, 2026-11-08 — three runs, two fixes, both fitted.** First **1 of 3**, and both KiCad
//! misses were the tool: qwen3:14b called `{"kicad_cli": ["sch", …]}` six times with the right
//! arguments and the tool wanted `{"args": […]}`, so the bare list is now read too. Second **2 of
//! 3**: *erc* right (0 errors, 37 warnings), *box* right in metres, and *bom* exported correctly and
//! finished on *총 47개의 부품* without reading the file — there were 11, and two digits pass the
//! finish check, which reads three. So a short text file kicad-cli wrote is shown in its result.
//! Third **3 of 3**: *11개*. **The control read 0 of 3 and never went looking**: *bom* spent 1,006 s
//! parsing the schematic as XML and then as lines and found 0 parts, *erc* said no tool was given,
//! *box* ran `blender` in the sandbox and was told it was not there — each an honest *cannot*,
//! none a false finish. So on this machine the tools are what make the work possible.

use std::path::{Path, PathBuf};
use std::time::Instant;

use argmind_core::contract::Speaks;
use argmind_core::task::{run, Ended, Rank, Task, Tiered, Tools};
use argmind_env_ollama::Local;
use argmind_env_windows::programs::{kicad_cli, Programs};
use argmind_env_windows::screen::Screen;
use argmind_env_windows::workspace::Workspace;

struct Model(Local);

impl Speaks for Model {
    fn ask(&mut self, prompt: &str) -> String {
        self.0.ask(prompt).map(|r| r.text).unwrap_or_default()
    }
}

fn fresh(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("argmind-programs-{name}-{}",
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// kicad-cli's own answers about the demo: the references in its BOM, and the ERC's errors and
/// warnings.
fn truth(kicad: &Path, demo: &Path) -> (Vec<String>, usize, usize) {
    let dir = fresh("truth");
    std::fs::copy(demo, dir.join("amp.kicad_sch")).unwrap();
    let run = |args: &[&str]| std::process::Command::new(kicad).args(args).current_dir(&dir).output().unwrap();
    run(&["sch", "export", "bom", "-o", "b.csv", "amp.kicad_sch"]);
    run(&["sch", "erc", "-o", "e.rpt", "amp.kicad_sch"]);
    let bom = std::fs::read_to_string(dir.join("b.csv")).unwrap();
    let refs: Vec<String> = bom.lines().skip(1).filter_map(|l| l.split("\",\"").next())
        .flat_map(|r| r.trim_matches('"').split([',', ' ']).map(str::trim).filter(|s| !s.is_empty())
            .map(String::from).collect::<Vec<_>>()).collect();
    let erc = std::fs::read_to_string(dir.join("e.rpt")).unwrap();
    let count = |what: &str| erc.lines().filter(|l| l.trim() == format!("; {what}")).count();
    let (errors, warnings) = (count("error"), count("warning"));
    let _ = std::fs::remove_dir_all(&dir);
    (refs, errors, warnings)
}

/// The bounding box of an STL, binary or text, as its three sides.
fn stl_sides(path: &Path) -> Option<[f64; 3]> {
    let b = std::fs::read(path).ok()?;
    let mut pts: Vec<[f64; 3]> = Vec::new();
    if b.starts_with(b"solid") && String::from_utf8_lossy(&b).contains("vertex") {
        for l in String::from_utf8_lossy(&b).lines() {
            let v: Vec<f64> = l.trim().strip_prefix("vertex")?.split_whitespace().filter_map(|n| n.parse().ok()).collect();
            if v.len() == 3 {
                pts.push([v[0], v[1], v[2]]);
            }
        }
    } else if b.len() >= 84 {
        let n = u32::from_le_bytes(b[80..84].try_into().ok()?) as usize;
        for t in 0..n {
            for k in 0..3 {
                let at = 84 + t * 50 + 12 + k * 12;
                let f = |i: usize| b.get(at + i * 4..at + i * 4 + 4).map(|s| f32::from_le_bytes(s.try_into().unwrap()) as f64);
                pts.push([f(0)?, f(1)?, f(2)?]);
            }
        }
    }
    if pts.is_empty() {
        return None;
    }
    let side = |i: usize| pts.iter().map(|p| p[i]).fold(f64::MIN, f64::max) - pts.iter().map(|p| p[i]).fold(f64::MAX, f64::min);
    Some([side(0), side(1), side(2)])
}

fn box_ok(dir: &Path) -> bool {
    let Some(mut s) = stl_sides(&dir.join("part.stl")) else { return false };
    s.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let near = |a: f64, b: f64| (a - b).abs() <= b.abs() * 0.01;
    dir.join("part.blend").is_file() && (near(s[0], 40.0) || near(s[0], 0.04))
        && near(s[1], s[0] / 2.0) && near(s[2], s[0] / 4.0)
}

/// Every number in the closing words, as written (digits only).
fn numbers_in(text: &str) -> Vec<usize> {
    text.split(|c: char| !c.is_ascii_digit()).filter_map(|s| s.parse().ok()).collect()
}

fn main() {
    let control = std::env::args().any(|a| a == "--control");
    let kicad = kicad_cli().expect("KiCad is not installed");
    let demo = kicad.parent().unwrap().parent().unwrap().join(r"share\kicad\demos\ecc83\ecc83-pp.kicad_sch");
    assert!(demo.is_file(), "{} is gone", demo.display());
    let (refs, errors, warnings) = truth(&kicad, &demo);
    println!("\n  설치된 프로그램을 작업이 쓴다{}", if control { " — 대조군: 프로그램 도구 없이" } else { "" });
    println!("  harness {}", argmind_measure::source_print(include_str!("programs.rs")));
    println!("  kicad-cli 가 낸 답: 부품 {}개 {:?}, ERC 오류 {errors} 경고 {warnings}\n", refs.len(), refs);

    let mut local = Local::new("qwen3:14b");
    local.think = Some(true);
    local.keep_alive = "300s".into();
    let _ = local.ask("Reply with the single word: ready");
    let mut model = Model(local);

    let cases: [(&str, &str, bool); 3] = [
        ("bom", "작업 폴더의 amp.kicad_sch 회로도에 들어간 부품 목록(참조번호와 값)을 parts.csv 로 만들고, 부품이 모두 몇 개인지 알려줘", true),
        ("erc", "작업 폴더의 amp.kicad_sch 회로도의 전기 규칙 검사(ERC)를 해서 결과를 erc.txt 로 저장하고, 오류와 경고가 각각 몇 개인지 알려줘", true),
        ("box", "가로 40mm 세로 20mm 높이 10mm 직육면체 부품을 3D 로 만들어서 part.stl 로 내보내고 Blender 파일 part.blend 로도 저장해줘", false),
    ];
    let mut passed = 0;
    for (name, what, schematic) in cases {
        let dir = fresh(name);
        if schematic {
            std::fs::copy(&demo, dir.join("amp.kicad_sch")).unwrap();
        }
        let mut sets: Vec<(Rank, Box<dyn Tools + Send>)> = vec![(Rank::Code, Box::new(Workspace::new(dir.clone()))),
                                                                (Rank::Screen, Box::new(Screen::new()))];
        if !control {
            sets.push((Rank::Api, Box::new(Programs::found(dir.clone()).expect("no programs"))));
        }
        let mut bench = Tiered::new(sets);
        let mut task = Task::new(what);
        let began = Instant::now();
        run(&mut task, &mut model, &mut bench, "2026년 11월 8일", &|| true);
        let secs = began.elapsed().as_secs_f64();
        drop(bench);
        let closing = match &task.ended { Some(Ended::Done(s)) => s.clone(), _ => String::new() };
        let said = numbers_in(&closing);
        let via = task.steps.iter().any(|s| (s.tool == "kicad_cli" || s.tool == "blender_python") && !s.failed);
        let ok = match name {
            "bom" => std::fs::read_to_string(dir.join("parts.csv"))
                .is_ok_and(|t| refs.iter().all(|r| t.contains(r.as_str()))) && said.contains(&refs.len()),
            "erc" => dir.join("erc.txt").is_file() && said.contains(&errors) && said.contains(&warnings),
            _ => box_ok(&dir),
        } && (control || via);
        passed += ok as usize;
        let ended = match &task.ended {
            Some(Ended::Done(s)) => format!("끝: {}", s.replace('\n', " ").chars().take(140).collect::<String>()),
            Some(Ended::Stopped(w)) => format!("**멈춤**: {w}"),
            None => "?".into(),
        };
        let tools: Vec<String> = task.steps.iter().map(|s| format!("{}{}", s.tool, if s.failed { "✗" } else { "" })).collect();
        println!("    {} {name} — {}단계 {secs:.0}초, 프로그램 도구 {} [{}]\n       {ended}",
                 if ok { "●" } else { "○" }, task.steps.len(), if via { "씀" } else { "안 씀" }, tools.join(" "));
        if name == "box" {
            println!("       part.stl 변: {:?}", stl_sides(&dir.join("part.stl")));
        }
        if std::env::var("PROGRAMS_VERBOSE").is_ok() {
            for s in &task.steps {
                println!("         · {} {} → {}", s.tool, s.args.to_string().chars().take(200).collect::<String>(),
                         s.observed.replace('\n', " ⏎ ").chars().take(240).collect::<String>());
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
    println!("\n  ── 돌리기 전에 적어둔 대로: {passed}/3{} ──",
             if control { " (대조군 — 기준 없음)" } else if passed == 3 { " — 설치된 프로그램으로 된다" } else { " — 보여주지 못했다" });
}
