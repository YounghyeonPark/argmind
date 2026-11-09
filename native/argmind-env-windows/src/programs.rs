//! **Programs already installed, by their own scripting interfaces** — the first way in, beside
//! MCP, for programs that have one. The owner, 2026-11-08: *문서작성이나 도면 설계는 코딩이나
//! mcp로 접근하는 방법도 있음*, then *mcp또는 api, 코드또는 스크립트 작성및 실행, 그래도 안되면
//! 샌드박스작업*.
//!
//! Two, because they are what this machine has and each has an interface that needs no window:
//!
//! * **Blender** — `blender --background --factory-startup --python <file>`: its Python API (`bpy`)
//!   makes, changes and exports a 3D model with no window at all.
//! * **KiCad** — `kicad-cli`: checks a schematic or a board (ERC, DRC) and exports a netlist, a
//!   BOM, a PDF, an SVG, Gerbers.
//!
//! Microsoft Office is installed too and is **not** here: COM may attach to the Word the person
//! has open, which is the single-instance hazard the work screen refuses. Documents go through
//! `python-docx` and `openpyxl` in the sandbox.
//!
//! **Found, not assumed**: a tool is listed only when its program is where an installer puts it,
//! the newest version first. Both run in the work folder, as the person, outside the sandbox —
//! `run_windows`'s standing, and weaker in the same way: a Blender script is Python and can reach
//! whatever the person can. `kicad-cli`'s arguments are refused when one names a place outside the
//! folder. Neither program's code is linked or shipped: they are processes started by path, like
//! PowerShell.

use std::path::{Path, PathBuf};
use std::process::Command;

use argmind_core::task::{ToolSpec, Tools};
use serde_json::{json, Value};

use crate::workspace::{RUN_SECONDS, Workspace};

/// The newest `blender.exe` under an installer's folder.
pub fn blender() -> Option<PathBuf> {
    newest(&program_files().join("Blender Foundation"), &["blender.exe"])
}

/// The newest `kicad-cli.exe`.
pub fn kicad_cli() -> Option<PathBuf> {
    newest(&program_files().join("KiCad"), &["bin", "kicad-cli.exe"])
}

fn program_files() -> PathBuf {
    std::env::var_os("ProgramFiles").map(PathBuf::from).unwrap_or_else(|| PathBuf::from(r"C:\Program Files"))
}

/// `<under>/<version>/<rest>` with the highest version that has it. Versions compare by their
/// numbers, so `10.0` is after `9.0` where text would put it first.
fn newest(under: &Path, rest: &[&str]) -> Option<PathBuf> {
    let mut found: Vec<(Vec<u64>, PathBuf)> = std::fs::read_dir(under).ok()?.filter_map(Result::ok).filter_map(|e| {
        let mut p = e.path();
        for r in rest {
            p.push(r);
        }
        p.is_file().then(|| (numbers(&e.file_name().to_string_lossy()), p))
    }).collect();
    found.sort();
    found.pop().map(|f| f.1)
}

fn numbers(name: &str) -> Vec<u64> {
    name.split(|c: char| !c.is_ascii_digit()).filter(|s| !s.is_empty()).filter_map(|s| s.parse().ok()).collect()
}

/// An argument that names a place outside the work folder — a drive, a root, a `..`.
fn leaves_the_folder(arg: &str) -> bool {
    let a = arg.trim().trim_matches('"');
    let b = a.as_bytes();
    (b.len() >= 2 && b[0].is_ascii_alphabetic() && b[1] == b':') || a.starts_with('\\') || a.starts_with('/')
        || a.split(['/', '\\', '=']).any(|part| part == "..")
}

pub struct Programs {
    work: Workspace,
    root: PathBuf,
    blender: Option<PathBuf>,
    kicad: Option<PathBuf>,
}

impl Programs {
    /// What this machine has, working in `root`. `None` when it has neither.
    pub fn found(root: PathBuf) -> Option<Programs> {
        Self::with(root, blender(), kicad_cli())
    }

    pub fn with(root: PathBuf, blender: Option<PathBuf>, kicad: Option<PathBuf>) -> Option<Programs> {
        (blender.is_some() || kicad.is_some()).then(|| Programs { work: Workspace::new(root.clone()), root, blender, kicad })
    }

    fn version(p: &Path) -> String {
        p.ancestors().find_map(|a| {
            let n = a.file_name()?.to_string_lossy().to_string();
            if let Some(v) = n.strip_prefix("Blender ") {
                return Some(v.to_string());
            }
            n.chars().next()?.is_ascii_digit().then_some(n)
        }).unwrap_or_default()
    }
}

impl Tools for Programs {
    fn list(&self) -> Vec<ToolSpec> {
        let mut out = Vec::new();
        if let Some(b) = &self.blender {
            out.push(ToolSpec { name: "blender_python".into(), does: format!(
                "설치된 Blender {}의 파이썬(bpy)으로 3D 모델을 만들고·바꾸고·내보낸다. 창 없이 돈다. \
코드는 먼저 write_file 로 작업 폴더의 .py 파일에 쓰고 그 이름을 script 로 준다. 스크립트의 현재 폴더가 작업 폴더다. \
빈 장면에서 시작한다 (기본 상자·카메라·조명 없음). 저장은 bpy.ops.wm.save_as_mainfile(filepath=...), \
STL 은 bpy.ops.wm.stl_export(filepath=...), OBJ 는 bpy.ops.wm.obj_export(filepath=...). print 한 것이 결과로 온다",
                Self::version(b)), args: json!({"script": "box.py"}) });
        }
        if let Some(k) = &self.kicad {
            out.push(ToolSpec { name: "kicad_cli".into(), does: format!(
                "설치된 KiCad {}의 kicad-cli 를 작업 폴더에서 실행한다: 회로도(.kicad_sch)·PCB(.kicad_pcb)를 검사(sch erc, \
pcb drc)하고 넷리스트·BOM·PDF·SVG·거버로 내보낸다. args 는 kicad-cli 뒤에 붙는 인자 목록이고, 파일은 작업 폴더 기준 \
경로만 된다. 쓰는 법을 모르면 [\"sch\",\"export\",\"bom\",\"--help\"] 처럼 --help 를 붙여 본다",
                Self::version(k)), args: json!({"args": ["sch", "export", "bom", "-o", "bom.csv", "board.kicad_sch"]}) });
        }
        out
    }

    fn call(&mut self, tool: &str, args: &Value) -> Result<String, String> {
        match tool {
            "blender_python" => {
                let exe = self.blender.as_ref().ok_or("Blender 가 없다")?;
                let script = args.get("script").and_then(Value::as_str).map(str::trim).filter(|s| !s.is_empty())
                    .ok_or("script 는 작업 폴더의 .py 파일 이름이다 — 코드는 write_file 로 먼저 쓴다")?;
                let path = self.work.inside(script)?;
                if !path.is_file() {
                    return Err(format!("{script} 이(가) 작업 폴더에 없다 — write_file 로 먼저 쓴다"));
                }
                let mut c = Command::new(exe);
                c.args(["--background", "--factory-startup", "--python-exit-code", "1", "--python"]).arg(&path)
                    .current_dir(&self.root).env("PYTHONIOENCODING", "utf-8");
                crate::workspace::finish(c, None, RUN_SECONDS).map(|t| quieter(&t)).map_err(|t| renamed(&quieter(&t)))
            }
            "kicad_cli" => {
                let exe = self.kicad.as_ref().ok_or("KiCad 가 없다")?;
                //: **The list itself is also an answer** — `{"kicad_cli": ["sch", …]}` is how qwen3:14b
                //: called it six times in the first `--bin programs` run, and all six were refused
                //: here while the arguments were right. Fitted to that run.
                let bare = if args.is_array() { Some(args) } else { None };
                let list: Vec<String> = match bare.or_else(|| args.get("args")) {
                    Some(Value::Array(a)) => a.iter().map(|v| v.as_str().map(String::from).unwrap_or_else(|| v.to_string())).collect(),
                    Some(Value::String(s)) => s.split_whitespace().map(String::from).collect(),
                    _ => return Err("args 는 kicad-cli 뒤에 붙는 인자 목록이다".into()),
                };
                if let Some(bad) = list.iter().find(|a| leaves_the_folder(a)) {
                    return Err(format!("작업 폴더 밖을 가리키는 인자라 실행하지 않는다: {bad}"));
                }
                let mut c = Command::new(exe);
                c.args(&list).current_dir(&self.root);
                let done = crate::workspace::finish(c, None, RUN_SECONDS)?;
                //: **What it wrote, when that is short text** — the first `--bin programs` run had
                //: the BOM exported correctly and the task finish on *총 47개의 부품* without
                //: reading it; there were 11. Fitted to that one case.
                let out = list.iter().position(|a| a == "-o" || a == "--output").and_then(|i| list.get(i + 1).cloned())
                    .or_else(|| list.iter().find_map(|a| a.strip_prefix("--output=").map(String::from)));
                let shown = out.and_then(|o| {
                    let text = std::fs::read_to_string(self.work.inside(&o).ok()?).ok()?;
                    let text_like = ["csv", "txt", "rpt", "net", "xml", "tsv", "json"]
                        .iter().any(|e| o.to_lowercase().ends_with(&format!(".{e}")));
                    (text_like && text.chars().count() <= 6000)
                        .then(|| format!("\n\n{o} 의 내용 ({}줄):\n{}", text.lines().count(), text.trim_end()))
                });
                Ok(done + &shown.unwrap_or_default())
            }
            _ => Err(format!("그런 도구는 없다: {tool}")),
        }
    }

    fn missing(&self, paths: &[String]) -> Vec<String> {
        self.work.missing(paths)
    }
}

/// **Operators that are gone, and what took their place** — Blender 4 replaced the old exporters,
/// and a model trained on older scripts writes them. The owner's first box in the application
/// (2026-11-09) called `bpy.ops.export_mesh.stl`, got *could not be found*, and ran the same
/// script twice more unchanged; `--bin programs` once invented `primitive_box_add`. The error
/// already says what is wrong; this adds what is right, from Blender's own operator list.
const RENAMED: [(&str, &str); 8] = [
    ("export_mesh.stl", "bpy.ops.wm.stl_export(filepath='part.stl')"),
    ("import_mesh.stl", "bpy.ops.wm.stl_import(filepath='part.stl')"),
    ("export_scene.obj", "bpy.ops.wm.obj_export(filepath='part.obj')"),
    ("import_scene.obj", "bpy.ops.wm.obj_import(filepath='part.obj')"),
    ("export_mesh.ply", "bpy.ops.wm.ply_export(filepath='part.ply')"),
    ("import_mesh.ply", "bpy.ops.wm.ply_import(filepath='part.ply')"),
    ("primitive_box_add", "bpy.ops.mesh.primitive_cube_add(size=1) 다음에 obj.dimensions = (x, y, z)"),
    ("primitive_cuboid_add", "bpy.ops.mesh.primitive_cube_add(size=1) 다음에 obj.dimensions = (x, y, z)"),
];

fn renamed(error: &str) -> String {
    match RENAMED.iter().find(|(old, _)| error.contains(&format!("bpy.ops.{old}\" error, could not be found"))) {
        Some((old, new)) => format!("{error}\n\n이 Blender 에는 bpy.ops.{old} 가 없다. 대신 {new} 를 쓴다. \
스크립트를 edit_file 로 고친 다음에 다시 실행한다 — 고치지 않고 다시 실행하면 같은 오류가 난다"),
        None => error.to_string(),
    }
}

/// Blender's own banner and quit lines, which say nothing about the script.
fn quieter(text: &str) -> String {
    text.lines().filter(|l| {
        let t = l.trim();
        !(t.starts_with("Blender ") && (t.contains("(hash") || t.contains("quit")))
            && !t.starts_with("Read prefs:") && !t.is_empty()
    }).collect::<Vec<_>>().join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_newest_version_is_found_by_its_numbers() {
        assert!(numbers("10.0") > numbers("9.0"));
        assert!(numbers("Blender 5.1") > numbers("Blender 4.2"));
        let dir = std::env::temp_dir().join(format!("argmind-programs-{}", std::process::id()));
        for v in ["9.0", "10.0", "8.0-empty"] {
            std::fs::create_dir_all(dir.join(v).join("bin")).unwrap();
        }
        for v in ["9.0", "10.0"] {
            std::fs::write(dir.join(v).join("bin").join("kicad-cli.exe"), b"").unwrap();
        }
        assert_eq!(newest(&dir, &["bin", "kicad-cli.exe"]), Some(dir.join("10.0").join("bin").join("kicad-cli.exe")));
        assert_eq!(newest(&dir.join("none"), &["x"]), None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_argument_naming_a_place_outside_the_folder_is_refused() {
        for bad in ["C:\\Users\\x\\a.kicad_sch", "c:/a", "\\\\server\\share", "/etc/passwd", "../a.net",
                    "out\\..\\..\\a", "--output=../x"] {
            assert!(leaves_the_folder(bad), "{bad}");
        }
        for ok in ["sch", "-o", "bom.csv", "boards/a.kicad_sch", "--fields=Reference,Value", "--help"] {
            assert!(!leaves_the_folder(ok), "{ok}");
        }
        let mut p = Programs::with(std::env::temp_dir(), None, Some(PathBuf::from("kicad-cli.exe"))).unwrap();
        let r = p.call("kicad_cli", &json!({"args": ["sch", "export", "bom", "-o", "C:\\Windows\\x.csv", "a.kicad_sch"]}));
        assert!(r.is_err_and(|e| e.contains("작업 폴더 밖")));
        //: The bare list reads the same as the wrapped one.
        let r = p.call("kicad_cli", &json!(["sch", "export", "bom", "-o", "C:\\Windows\\x.csv", "a.kicad_sch"]));
        assert!(r.is_err_and(|e| e.contains("작업 폴더 밖")));
    }

    #[test]
    fn a_gone_operator_is_answered_with_its_replacement() {
        let e = "AttributeError: Calling operator \"bpy.ops.export_mesh.stl\" error, could not be found";
        assert!(renamed(e).contains("bpy.ops.wm.stl_export"), "{}", renamed(e));
        //: Any other error is left as it was.
        let other = "NameError: name 'cube' is not defined";
        assert_eq!(renamed(other), other);
    }

    #[test]
    fn only_what_is_installed_is_listed() {
        assert!(Programs::with(std::env::temp_dir(), None, None).is_none());
        let p = Programs::with(std::env::temp_dir(), None, Some(PathBuf::from("C:\\x\\KiCad\\10.0\\bin\\kicad-cli.exe"))).unwrap();
        let names: Vec<String> = p.list().into_iter().map(|t| t.name).collect();
        assert_eq!(names, ["kicad_cli"]);
        assert!(p.list()[0].does.contains("10.0"), "{}", p.list()[0].does);
    }
}
