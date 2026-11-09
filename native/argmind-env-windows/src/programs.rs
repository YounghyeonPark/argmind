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

/// Where code given inline to `blender_python` is saved before it runs.
pub const INLINE: &str = "blender_script.py";

/// Text in a file-name argument that is code — more than one line, or Python's own words.
fn looks_like_code(s: &str) -> bool {
    s.contains('\n') || s.contains("import ") || s.contains("bpy.")
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

    /// **The script to run, and a line saying where code given inline was put.** The 2026-11-09
    /// run of `--bin programs` spent a step on each Blender case: once the code itself went in
    /// `script` and was refused as a path, once a file was named before it was written. So code is
    /// taken as code — in `code`, as a bare string, or in `script` when it is plainly not a file
    /// name — and saved through the workbench's own `write_file`, which backs up what it replaces,
    /// to [`INLINE`], the name the result gives back for the next `edit_file`.
    fn script_of(&mut self, args: &Value) -> Result<(String, String), String> {
        let given = args.get("code").and_then(Value::as_str)
            .or_else(|| args.as_str())
            .or_else(|| args.get("script").and_then(Value::as_str).filter(|s| looks_like_code(s)));
        if let Some(code) = given {
            self.work.call("write_file", &json!({"path": INLINE, "content": code}))?;
            return Ok((INLINE.into(), format!("(코드를 {INLINE} 에 저장해 실행했다 — 고칠 때는 이 파일을 edit_file 로)\n")));
        }
        let script = args.get("script").and_then(Value::as_str).map(str::trim).filter(|s| !s.is_empty())
            .ok_or("code 에 코드를 주거나, script 에 작업 폴더의 .py 파일 이름을 준다")?;
        Ok((script.to_string(), String::new()))
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
코드를 그대로 code 로 주면 작업 폴더의 blender_script.py 에 저장해 실행하고, 이미 있는 .py 는 그 이름을 script 로 준다. \
스크립트의 현재 폴더가 작업 폴더다. \
빈 장면에서 시작한다 (기본 상자·카메라·조명 없음). 저장은 bpy.ops.wm.save_as_mainfile(filepath=...), \
STL 은 bpy.ops.wm.stl_export(filepath=...), OBJ 는 bpy.ops.wm.obj_export(filepath=...). \
Blender 의 길이 단위는 m 이고 STL·OBJ 에는 단위가 없어 다른 프로그램은 mm 로 읽는다: 크기를 mm 로 받았으면 \
모델은 m 로 만들고 global_scale=1000 으로 내보낸다. 내보낸 STL 의 크기는 결과에 붙어 온다. print 한 것이 결과로 온다",
                Self::version(b)), args: json!({"code": "import bpy\nbpy.ops.mesh.primitive_cube_add(size=1)"}) });
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
                let exe = self.blender.clone().ok_or("Blender 가 없다")?;
                let (script, saved) = self.script_of(args)?;
                let path = self.work.inside(&script)?;
                if !path.is_file() {
                    return Err(format!("{script} 이(가) 작업 폴더에 없다 — write_file 로 먼저 쓴다"));
                }
                let mut c = Command::new(exe);
                //: **An empty scene, made so rather than said so.** The tool text has promised one since
                //: the first day and `--factory-startup` gives a camera, a light and a 2 m cube: every
                //: run that passed had deleted them itself, and the first that did not exported the
                //: cube with its own and measured 2000 × 2000 × 2000 (2026-11-09). Checked: objects
                //: `['Camera', 'Cube', 'Light']` without this line, `[]` with it.
                c.args(["--background", "--factory-startup", "--python-exit-code", "1",
                        "--python-expr", "import bpy; bpy.ops.wm.read_factory_settings(use_empty=True)",
                        "--python"]).arg(&path)
                    .current_dir(&self.root).env("PYTHONIOENCODING", "utf-8");
                let began = std::time::SystemTime::now();
                let ran = crate::workspace::finish(c, None, RUN_SECONDS).map(|t| quieter(&t))
                    .map_err(|t| saved.clone() + &renamed(&quieter(&t)))?;
                Ok(saved + &ran + &sizes_of_new_stl(&self.root, began))
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

/// **What a run exported, measured by the program** — every `.stl` in the work folder written
/// since `began`, its three sides read off the file. The owner's first box in the application
/// (2026-11-09) came out at 0.04 × 0.02 × 0.01: Blender's metres, which a slicer reads as
/// millimetres, so a 40 mm box opened as 0.04 mm and the task's closing words said it was right.
/// The sizes are the file's, not the model's account, and the note says how they will be read.
fn sizes_of_new_stl(root: &Path, began: std::time::SystemTime) -> String {
    let Ok(dir) = std::fs::read_dir(root) else { return String::new() };
    let mut out = Vec::new();
    for e in dir.filter_map(Result::ok) {
        let p = e.path();
        let fresh = e.metadata().and_then(|m| m.modified()).is_ok_and(|t| t >= began);
        if fresh && p.extension().is_some_and(|x| x.eq_ignore_ascii_case("stl")) {
            if let Some(s) = stl_sides(&p) {
                let n = |v: f64| format!("{}", (v * 10_000.0).round() / 10_000.0);
                out.push(format!("{}: {} × {} × {} (단위 없음 — 다른 프로그램은 mm 로 읽는다)",
                                 p.file_name().unwrap_or_default().to_string_lossy(), n(s[0]), n(s[1]), n(s[2])));
            }
        }
    }
    if out.is_empty() { String::new() } else { format!("\n\n내보낸 STL 의 크기:\n{}", out.join("\n")) }
}

/// The bounding box of an STL, binary or text, as its three sides.
pub fn stl_sides(path: &Path) -> Option<[f64; 3]> {
    let b = std::fs::read(path).ok()?;
    let mut lo = [f64::MAX; 3];
    let mut hi = [f64::MIN; 3];
    let mut any = false;
    let mut take = |p: [f64; 3]| {
        any = true;
        for i in 0..3 {
            lo[i] = lo[i].min(p[i]);
            hi[i] = hi[i].max(p[i]);
        }
    };
    let text = String::from_utf8_lossy(&b);
    if b.starts_with(b"solid") && text.contains("vertex") {
        for l in text.lines() {
            if let Some(v) = l.trim().strip_prefix("vertex") {
                let v: Vec<f64> = v.split_whitespace().filter_map(|n| n.parse().ok()).collect();
                if v.len() == 3 {
                    take([v[0], v[1], v[2]]);
                }
            }
        }
    } else if b.len() >= 84 {
        let n = u32::from_le_bytes(b[80..84].try_into().ok()?) as usize;
        for t in 0..n {
            for k in 0..3 {
                let at = 84 + t * 50 + 12 + k * 12;
                let f = |i: usize| -> Option<f64> {
                    Some(f32::from_le_bytes(b.get(at + i * 4..at + i * 4 + 4)?.try_into().ok()?) as f64)
                };
                take([f(0)?, f(1)?, f(2)?]);
            }
        }
    }
    any.then(|| [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]])
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
    fn an_exported_stl_is_measured_from_the_file() {
        let dir = std::env::temp_dir().join(format!("argmind-stl-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let began = std::time::SystemTime::now() - std::time::Duration::from_secs(1);
        //: One binary triangle spanning 40 × 20 × 10, and a text one spanning 1 × 2 × 3.
        let mut b = vec![0u8; 80];
        b.extend_from_slice(&1u32.to_le_bytes());
        b.extend_from_slice(&[0u8; 12]);
        for v in [[0f32, 0., 0.], [40., 20., 0.], [0., 0., 10.]] {
            for c in v {
                b.extend_from_slice(&c.to_le_bytes());
            }
        }
        b.extend_from_slice(&[0u8; 2]);
        std::fs::write(dir.join("part.stl"), &b).unwrap();
        std::fs::write(dir.join("t.stl"), "solid t\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 1 2 0\nvertex 0 0 3\nendloop\nendfacet\nendsolid t\n").unwrap();
        assert_eq!(stl_sides(&dir.join("part.stl")), Some([40.0, 20.0, 10.0]));
        assert_eq!(stl_sides(&dir.join("t.stl")), Some([1.0, 2.0, 3.0]));
        let said = sizes_of_new_stl(&dir, began);
        assert!(said.contains("part.stl: 40 × 20 × 10") && said.contains("t.stl: 1 × 2 × 3"), "{said}");
        //: A file older than the run is not this run's.
        assert_eq!(sizes_of_new_stl(&dir, std::time::SystemTime::now() + std::time::Duration::from_secs(60)), "");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn code_given_inline_is_saved_and_named_back() {
        let dir = std::env::temp_dir().join(format!("argmind-inline-{}", std::process::id()));
        let mut p = Programs::with(dir.clone(), Some(PathBuf::from("blender.exe")), None).unwrap();
        let code = "import bpy\nprint('hi')";
        //: In `code`, bare, and in `script` where it is plainly not a file name.
        for args in [json!({"code": code}), json!(code), json!({"script": code})] {
            let (script, said) = p.script_of(&args).unwrap();
            assert_eq!(script, INLINE);
            assert!(said.contains(INLINE), "{said}");
            assert_eq!(std::fs::read_to_string(dir.join(INLINE)).unwrap(), code);
        }
        //: A file name is still a file name, and nothing is written for it.
        let (script, said) = p.script_of(&json!({"script": "box.py"})).unwrap();
        assert_eq!((script.as_str(), said.as_str()), ("box.py", ""));
        assert!(p.script_of(&json!({})).is_err());
        let _ = std::fs::remove_dir_all(&dir);
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
