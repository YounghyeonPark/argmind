//! **Does a task use what an MCP server offers — and only when there is one?**
//!
//! ```text
//!   cargo run -p argmind-measure --bin mcp --release
//! ```
//!
//! From `native/`. **Stop argmind first** — the model is the application's task model, qwen3:14b
//! with reasoning on. The owner, 2026-11-08: *문서작성이나 도면 설계는 코딩이나 mcp로 접근하는 방법도
//! 있음.* `argmind-env-mcp` makes a configured server's tools a task's; this asks whether the task
//! loop picks them up from the list alone, with nothing in the prompt about MCP.
//!
//! The server is this program run with `--serve <folder>`, over stdio, through the same client the
//! application uses — a stand-in for a CAD program's server and a parts catalogue: `lookup_part`
//! answers from a table nobody could guess, and `new_drawing` / `add_rectangle` / `add_circle` /
//! `save_drawing` write an SVG into the work folder. Every task has the real workbench beside it.
//!
//! ## Written before the run
//!
//! Two tasks, each with the server (**MCP**) and without it (**control**):
//!
//! 1. **parts**: *R17 과 C3 의 사양을 찾아 parts.md 에 적어줘* — passes when `parts.md` holds both
//!    part numbers (`RN73R1ETTP3831B`, `GRM188R71H224KA93D`), which are in no prompt.
//! 2. **plate**: a 100 × 60 rectangle with a radius-10 circle in its centre, saved as `plate.svg` —
//!    passes when the file holds a `<rect>` at 0,0 100×60 and a `<circle>` at 50,30 r 10. **Either
//!    route passes**: an SVG written by `write_file` is the coding route the owner named, and it is
//!    printed which one was taken.
//!
//! **The wiring works** only if MCP passes both and the record shows a `cad__` call in each, *and*
//! the control does not pass **parts**. A control that finishes **parts** with `done` anyway is
//! counted as a false finish. One run each, so it is a demonstration, not a rate.
//!
//! **Read, 2026-11-08, one run: as written — MCP 2 of 2, each through `cad__` calls, and the
//! control did not pass *parts*.** *parts* looked both codes up (70 s, 4 steps), said *done* before
//! writing the file, was sent back by the finish check and wrote it. *plate* was
//! `new_drawing` → `add_rectangle` → `add_circle` → `save_drawing` with the right numbers, 55 s.
//! The control's *parts* listed an empty folder and finished *부품 목록을 확인할 수 없습니다* —
//! **counted a false finish by the definition above, and reading it, it claims nothing**: an honest
//! *cannot*, which the definition was too blunt to tell apart. The control's *plate* passed by the
//! coding route, an SVG from `write_file`, after **the finish check refused it for *2000*** — the
//! `xmlns` URL. That was the check's fault, and a drawing's markup is now a source like code.

use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use argmind_core::contract::Speaks;
use argmind_core::task::{run, Ended, Joined, Task, Tools};
use argmind_env_mcp::{Configured, Servers};
use argmind_env_ollama::Local;
use argmind_env_windows::workspace::Workspace;
use serde_json::{json, Value};

struct Model(Local);

impl Speaks for Model {
    fn ask(&mut self, prompt: &str) -> String {
        self.0.ask(prompt).map(|r| r.text).unwrap_or_default()
    }
}

const PARTS: [(&str, &str); 3] = [
    ("R17", "3.83kΩ 0402 ±0.1% 1/16W — KOA RN73R1ETTP3831B"),
    ("C3", "220nF 0603 X7R 50V — Murata GRM188R71H224KA93D"),
    ("U2", "LDO 3.3V 500mA SOT-23-5 — TI TLV75533PDBVR"),
];

/// **The stand-in server**: MCP over stdio, the same shape `examples/mcp-echo.rs` tests.
fn serve(folder: &Path) {
    let mut shapes: Vec<String> = Vec::new();
    let mut size: Option<(f64, f64)> = None;
    let mut out = std::io::stdout();
    let tool = |name: &str, does: &str, props: Value, required: &[&str]| json!({"name": name, "description": does,
        "inputSchema": {"type": "object", "properties": props, "required": required}});
    for line in std::io::stdin().lock().lines().map_while(Result::ok) {
        let Ok(m) = serde_json::from_str::<Value>(&line) else { continue };
        let (Some(id), Some(method)) = (m.get("id").cloned(), m.get("method").and_then(Value::as_str)) else { continue };
        let num = |a: &Value, k: &str| a.get(k).and_then(|v| v.as_f64().or_else(|| v.as_str()?.parse().ok()));
        let text = |t: String, err: bool| json!({"content": [{"type": "text", "text": t}], "isError": err});
        let result = match method {
            "initialize" => json!({"protocolVersion": "2025-06-18", "capabilities": {"tools": {}},
                                   "serverInfo": {"name": "cad", "version": "0"}}),
            "tools/list" => json!({"tools": [
                tool("lookup_part", "부품 목록에서 부품 번호(예: R17)의 사양과 제조사 품번을 찾는다",
                     json!({"code": {"type": "string"}}), &["code"]),
                tool("new_drawing", "새 도면을 시작한다. 단위는 mm", json!({"width": {"type": "number"}, "height": {"type": "number"}}), &["width", "height"]),
                tool("add_rectangle", "사각형을 넣는다. x, y 는 왼쪽 위 모서리",
                     json!({"x": {"type": "number"}, "y": {"type": "number"}, "w": {"type": "number"}, "h": {"type": "number"}}), &["x", "y", "w", "h"]),
                tool("add_circle", "원을 넣는다. cx, cy 는 중심", json!({"cx": {"type": "number"}, "cy": {"type": "number"}, "r": {"type": "number"}}), &["cx", "cy", "r"]),
                tool("save_drawing", "지금 도면을 작업 폴더에 SVG 파일로 저장한다", json!({"file": {"type": "string"}}), &["file"]),
            ]}),
            "tools/call" => {
                let a = &m["params"]["arguments"];
                match m["params"]["name"].as_str().unwrap_or("") {
                    "lookup_part" => {
                        let code = a.get("code").and_then(Value::as_str).unwrap_or("").trim().to_uppercase();
                        match PARTS.iter().find(|(c, _)| *c == code) {
                            Some((c, spec)) => text(format!("{c}: {spec}"), false),
                            None => text(format!("{code} 은 부품 목록에 없다"), true),
                        }
                    }
                    "new_drawing" => match (num(a, "width"), num(a, "height")) {
                        (Some(w), Some(h)) => { size = Some((w, h)); shapes.clear(); text(format!("새 도면 {w}×{h} mm"), false) }
                        _ => text("width 와 height 가 필요하다".into(), true),
                    },
                    "add_rectangle" => match (size, num(a, "x"), num(a, "y"), num(a, "w"), num(a, "h")) {
                        (None, ..) => text("먼저 new_drawing".into(), true),
                        (_, Some(x), Some(y), Some(w), Some(h)) => {
                            shapes.push(format!("<rect x=\"{x}\" y=\"{y}\" width=\"{w}\" height=\"{h}\" fill=\"none\" stroke=\"black\"/>"));
                            text(format!("사각형 넣음 (도형 {}개)", shapes.len()), false)
                        }
                        _ => text("x, y, w, h 가 필요하다".into(), true),
                    },
                    "add_circle" => match (size, num(a, "cx"), num(a, "cy"), num(a, "r")) {
                        (None, ..) => text("먼저 new_drawing".into(), true),
                        (_, Some(cx), Some(cy), Some(r)) => {
                            shapes.push(format!("<circle cx=\"{cx}\" cy=\"{cy}\" r=\"{r}\" fill=\"none\" stroke=\"black\"/>"));
                            text(format!("원 넣음 (도형 {}개)", shapes.len()), false)
                        }
                        _ => text("cx, cy, r 이 필요하다".into(), true),
                    },
                    "save_drawing" => {
                        let file = a.get("file").and_then(Value::as_str).unwrap_or("").trim().to_string();
                        match size {
                            None => text("먼저 new_drawing".into(), true),
                            _ if file.is_empty() || file.contains("..") || file.contains('/') || file.contains('\\') =>
                                text("file 은 작업 폴더 안의 이름 하나다".into(), true),
                            Some((w, h)) => {
                                let svg = format!("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{w}mm\" height=\"{h}mm\" viewBox=\"0 0 {w} {h}\">\n{}\n</svg>\n", shapes.join("\n"));
                                match std::fs::write(folder.join(&file), svg) {
                                    Ok(()) => text(format!("{file} 에 저장했다 (도형 {}개)", shapes.len()), false),
                                    Err(e) => text(e.to_string(), true),
                                }
                            }
                        }
                    }
                    other => text(format!("그런 도구는 없다: {other}"), true),
                }
            }
            _ => {
                let _ = writeln!(out, "{}", json!({"jsonrpc": "2.0", "id": id, "error": {"code": -32601, "message": "no"}}));
                let _ = out.flush();
                continue;
            }
        };
        let _ = writeln!(out, "{}", json!({"jsonrpc": "2.0", "id": id, "result": result}));
        let _ = out.flush();
    }
}

/// Numeric attributes of every `<tag …>` in an SVG.
fn shapes(svg: &str, tag: &str) -> Vec<std::collections::HashMap<String, f64>> {
    svg.split(&format!("<{tag}")).skip(1).map(|rest| {
        let body = rest.split('>').next().unwrap_or("");
        let mut attrs = std::collections::HashMap::new();
        for part in body.split_whitespace() {
            if let Some((k, v)) = part.split_once('=') {
                if let Ok(n) = v.trim_matches(|c| c == '"' || c == '\'' || c == '/').trim_end_matches("mm").parse::<f64>() {
                    attrs.insert(k.to_string(), n);
                }
            }
        }
        attrs
    }).collect()
}

fn plate_ok(dir: &Path) -> bool {
    let Ok(svg) = std::fs::read_to_string(dir.join("plate.svg")) else { return false };
    let near = |m: &std::collections::HashMap<String, f64>, k: &str, v: f64| m.get(k).is_some_and(|x| (x - v).abs() < 0.01);
    let at = |m: &std::collections::HashMap<String, f64>, k: &str, v: f64| m.get(k).map_or(v == 0.0, |x| (x - v).abs() < 0.01);
    shapes(&svg, "rect").iter().any(|r| at(r, "x", 0.0) && at(r, "y", 0.0) && near(r, "width", 100.0) && near(r, "height", 60.0))
        && shapes(&svg, "circle").iter().any(|c| near(c, "cx", 50.0) && near(c, "cy", 30.0) && near(c, "r", 10.0))
}

fn parts_ok(dir: &Path) -> bool {
    std::fs::read_to_string(dir.join("parts.md"))
        .is_ok_and(|s| s.contains("RN73R1ETTP3831B") && s.contains("GRM188R71H224KA93D"))
}

struct Case {
    name: &'static str,
    what: &'static str,
    check: fn(&Path) -> bool,
}

const CASES: [Case; 2] = [
    Case { name: "parts", what: "부품 목록에서 R17 과 C3 의 사양과 제조사 품번을 찾아 parts.md 에 정리해줘", check: parts_ok },
    Case { name: "plate", what: "가로 100mm 세로 60mm 판 도면을 그려줘. 바깥은 사각형, 정가운데에 반지름 10mm 구멍(원) 하나. plate.svg 로 저장해줘", check: plate_ok },
];

fn fresh(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("argmind-mcp-{name}-{}",
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("--serve") {
        return serve(Path::new(&args[2]));
    }
    println!("\n  MCP 서버의 도구를 작업이 쓰는가 — 있을 때와 없을 때");
    println!("  harness {}\n", argmind_measure::source_print(include_str!("mcp.rs")));
    let mut local = Local::new("qwen3:14b");
    local.think = Some(std::env::var("MCP_NO_THINK").is_err());
    local.keep_alive = "300s".into();
    let _ = local.ask("Reply with the single word: ready");
    let mut model = Model(local);
    let me = std::env::current_exe().unwrap().display().to_string();
    let mut results = Vec::new();
    for with in [true, false] {
        println!("  ── {} ──", if with { "MCP" } else { "control" });
        for case in &CASES {
            let dir = fresh(case.name);
            let bench = Workspace::new(dir.clone());
            let mut sets: Vec<Box<dyn Tools + Send>> = vec![Box::new(bench)];
            if with {
                let c = Configured { name: "cad".into(), command: me.clone(),
                                     args: vec!["--serve".into(), dir.display().to_string()], env: vec![] };
                let (servers, failed) = Servers::start(&[c]);
                assert!(failed.is_empty(), "{failed:?}");
                sets.push(Box::new(servers));
            }
            let mut bench = Joined(sets);
            let mut task = Task::new(case.what);
            let began = Instant::now();
            run(&mut task, &mut model, &mut bench, "2026년 11월 8일", &|| true);
            let secs = began.elapsed().as_secs_f64();
            drop(bench);
            let ok = (case.check)(&dir);
            let via_mcp = task.steps.iter().any(|s| s.tool.starts_with("cad__") && !s.failed);
            let done = matches!(task.ended, Some(Ended::Done(_)));
            let ended = match &task.ended {
                Some(Ended::Done(s)) => format!("끝: {}", s.chars().take(90).collect::<String>()),
                Some(Ended::Stopped(w)) => format!("**멈춤**: {w}"),
                None => "?".into(),
            };
            let tools: Vec<String> = task.steps.iter().map(|s| format!("{}{}", s.tool, if s.failed { "✗" } else { "" })).collect();
            println!("    {} {} — {}단계 {secs:.0}초, {}{} [{}]\n       {ended}", if ok { "●" } else { "○" }, case.name,
                     task.steps.len(), if via_mcp { "MCP 로" } else { "MCP 없이" },
                     if done && !ok { ", **거짓 끝**" } else { "" }, tools.join(" "));
            if std::env::var("MCP_VERBOSE").is_ok() {
                for s in &task.steps {
                    println!("         · {} {} → {}", s.tool, s.args.to_string().chars().take(160).collect::<String>(),
                             s.observed.replace('\n', " ⏎ ").chars().take(160).collect::<String>());
                }
            }
            results.push((with, case.name, ok, via_mcp, done && !ok));
            let _ = std::fs::remove_dir_all(&dir);
        }
    }
    let get = |w: bool, n: &str| results.iter().find(|r| r.0 == w && r.1 == n).copied().unwrap();
    let works = ["parts", "plate"].iter().all(|n| { let r = get(true, n); r.2 && r.3 }) && !get(false, "parts").2;
    let false_finish = results.iter().filter(|r| r.4).count();
    println!("\n  ── 돌리기 전에 적어둔 대로: {} — 거짓 끝 {false_finish} ──",
             if works { "MCP 도구를 쓴다" } else { "보여주지 못했다" });
}
