//! **The workbench a task works at**: files in one folder, commands in a sandbox, and Windows
//! commands in that same folder when the work needs Windows.
//!
//! The owner, 2026-11-03: code argmind writes should run by itself — *실행도 스스로* — and asked for
//! both places: **a dedicated WSL distro by default, and the folder on Windows for what only runs
//! there**. So:
//!
//! * **`run`** goes to `argmind-sandbox`, a WSL distro of its own (not the person's Ubuntu) with
//!   **drive mounting and Windows interop switched off** in its `/etc/wsl.conf` — checked: `/mnt/c`
//!   empty, `cmd.exe` and `powershell.exe` not found inside. Each command mounts **only the
//!   workspace folder** at `/work`, so what runs there can reach that folder and nothing else on the
//!   machine.
//! * **`run_windows`** is PowerShell in the workspace folder. **Weaker and said so**: a Windows
//!   process started in a folder is not confined to it. It exists for what has no Linux equivalent.
//! * **Files** are paths *inside* the folder — absolute paths and `..` are refused — and anything a
//!   write or an edit would replace is copied to `.argmind-backup` first, so a bad edit is undone by
//!   copying a file back.
//!
//! * **Documents** — `.docx` and `.xlsx` — are made and changed in the sandbox, with `python-docx`
//!   and `openpyxl` (both MIT, from Ubuntu's own packages, installed without their recommends so
//!   that Pillow's GPL-3 `libimagequant` stays out — `docs/licences.toml`). `read_file` on one
//!   returns its text, read there by the same libraries, and `write_file` and `edit_file` refuse
//!   one: either would write characters into a zip and leave a file nothing opens.
//!
//! Every command has a time limit and its output is cut by the task loop; a command that runs past
//! [`RUN_SECONDS`] is killed and says so.

use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use argmind_core::task::{ToolSpec, Tools};
use serde_json::{json, Value};

/// The sandbox's name, as `wsl -l` lists it.
pub const SANDBOX: &str = "argmind-sandbox";
/// How long one command may run.
pub const RUN_SECONDS: u64 = 120;
/// The largest file `read_file` returns whole.
pub const READ_MOST: u64 = 512 * 1024;
/// The longest `python3 -c` taken; past it a program belongs in a file. Chosen: a one-liner that
/// prints something fits, a loop with a `with` around it does not.
pub const PYTHON_ONE_LINE: usize = 120;
/// What the sandbox has to set up once, as root: `apt-get install --no-install-recommends
/// python3-docx python3-openpyxl`.
pub const DOCUMENT_PACKAGES: &str = "python3-docx python3-openpyxl";
/// Files that are zips of XML: text goes in through a library, never through `write_file`.
pub const DOCUMENTS: [&str; 3] = ["docx", "xlsx", "xlsm"];
/// **Reads a document's text** in the sandbox: paragraphs and tables in body order with any style
/// but the plain one named **after** the text, in brackets no document uses — written first as a
/// `[Heading 1] ` prefix, a model took the prefix for the text and its script looked for paragraphs
/// starting with it, changed none, and finished — and a workbook cell by cell with formulas as written — `openpyxl`
/// cannot compute them, and a formula is what somebody asked for when they asked for one.
const READ_DOCUMENT: &str = r###"
import sys
p = sys.argv[1]
if p.lower().endswith(".docx"):
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph
    d = docx.Document(p)
    for el in d.element.body.iterchildren():
        tag = el.tag.split("}")[-1]
        if tag == "p":
            para = Paragraph(el, d)
            style = para.style.name if para.style is not None else ""
            if para.text.strip():
                print(para.text if style in ("", "Normal") else f"{para.text}  〔서식: {style}〕")
        elif tag == "tbl":
            for row in Table(el, d).rows:
                print("| " + " | ".join(c.text for c in row.cells) + " |")
else:
    import openpyxl
    for ws in openpyxl.load_workbook(p).worksheets:
        print(f"## 시트: {ws.title}")
        for row in ws.iter_rows():
            cells = [f"{c.coordinate}={c.value}" for c in row if c.value is not None]
            if cells:
                print("  ".join(cells))
"###;

/// Is this a document — a zip of XML that `write_file` would break?
pub fn is_document(rel: &str) -> bool {
    Path::new(rel.trim()).extension().and_then(|e| e.to_str())
        .is_some_and(|e| DOCUMENTS.contains(&e.to_ascii_lowercase().as_str()))
}

/// Where replaced files go, inside the workspace.
pub const BACKUPS: &str = ".argmind-backup";

pub struct Workspace {
    pub root: PathBuf,
    pub sandbox: String,
}

impl Workspace {
    pub fn new(root: PathBuf) -> Workspace {
        let _ = std::fs::create_dir_all(&root);
        Workspace { root, sandbox: SANDBOX.into() }
    }

    /// **A path inside the workspace, or a refusal.** Relative, no `..`, no drive, no root.
    pub fn inside(&self, rel: &str) -> Result<PathBuf, String> {
        let p = Path::new(rel.trim());
        if rel.trim().is_empty() {
            return Ok(self.root.clone());
        }
        for c in p.components() {
            match c {
                Component::Normal(_) | Component::CurDir => {}
                _ => return Err(format!("작업 폴더 밖이라 쓸 수 없다: {rel}")),
            }
        }
        Ok(self.root.join(p))
    }

    fn backup(&self, path: &Path, rel: &str) -> Result<(), String> {
        if !path.exists() {
            return Ok(());
        }
        let dir = self.root.join(BACKUPS);
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis()).unwrap_or(0);
        let flat: String = rel.chars().map(|c| if c == '/' || c == '\\' { '_' } else { c }).collect();
        std::fs::copy(path, dir.join(format!("{stamp}-{flat}"))).map(|_| ()).map_err(|e| e.to_string())
    }

    fn arg<'a>(args: &'a Value, key: &str) -> Result<&'a str, String> {
        args.get(key).and_then(Value::as_str).ok_or_else(|| format!("{key} 가 없다"))
    }

    fn list_files(&self, rel: &str) -> Result<String, String> {
        let start = self.inside(rel)?;
        let mut out = Vec::new();
        fn walk(dir: &Path, base: &Path, depth: usize, out: &mut Vec<String>) {
            let Ok(entries) = std::fs::read_dir(dir) else { return };
            let mut entries: Vec<_> = entries.flatten().collect();
            entries.sort_by_key(|e| e.file_name());
            for e in entries {
                if out.len() >= 200 {
                    return;
                }
                let name = e.file_name().to_string_lossy().to_string();
                if name == BACKUPS {
                    continue;
                }
                let path = e.path();
                let shown = path.strip_prefix(base).unwrap_or(&path).to_string_lossy().replace('\\', "/");
                if path.is_dir() {
                    out.push(format!("{shown}/"));
                    if depth < 3 {
                        walk(&path, base, depth + 1, out);
                    }
                } else {
                    let size = e.metadata().map(|m| m.len()).unwrap_or(0);
                    out.push(format!("{shown} ({size} bytes)"));
                }
            }
        }
        walk(&start, &self.root, 1, &mut out);
        Ok(if out.is_empty() { "(비어 있음)".into() } else { out.join("\n") })
    }

    /// **A command in the sandbox**, with only the workspace mounted — **this** workspace: the distro
    /// outlives a command, so a mount left by the last one is undone first. The first version kept
    /// it, and the fourth task of a run tested the first task's folder (`No such file` on a file the
    /// model had just read).
    pub fn run_sandboxed(&self, command: &str) -> Result<String, String> {
        let root = self.root.to_string_lossy().to_string();
        let prelude = format!(
            "mkdir -p /work && (umount -l /work 2>/dev/null; true) && mount -t drvfs '{}' /work && \
             cd /work && timeout {RUN_SECONDS} bash -l", root.replace('\'', ""));
        let mut c = Command::new("wsl.exe");
        c.args(["-d", &self.sandbox, "-u", "root", "--", "bash", "-c", &prelude]);
        finish(c, Some(command), RUN_SECONDS + 15)
    }

    /// **A document's text**, read in the sandbox — see [`READ_DOCUMENT`].
    pub fn read_document(&self, rel: &str) -> Result<String, String> {
        let rel = rel.trim().replace('\\', "/");
        if rel.contains('\'') {
            return Err(format!("{rel}: 따옴표가 든 이름은 읽지 않는다"));
        }
        let text = self.run_sandboxed(&format!("python3 - '{rel}' <<'ARGMIND_DOC'\n{READ_DOCUMENT}\nARGMIND_DOC"))
            .map_err(|e| format!("{rel} 을 문서로 읽지 못했다: {e}"))?;
        let text = text.strip_suffix("(종료 코드 0)").unwrap_or(&text).trim_end();
        Ok(format!("({rel} 에서 글만 뽑았다. 〔 〕 안은 글이 아니라 그 문단의 서식이다 — 고치려면 run 에서 python-docx 나 openpyxl 로)\n{text}"))
    }

    /// Every document in the workspace, with when it last changed and its size. Backups are not
    /// the person's documents.
    fn documents(&self) -> Vec<(String, std::time::SystemTime, u64)> {
        fn walk(dir: &Path, base: &Path, depth: usize, out: &mut Vec<(String, std::time::SystemTime, u64)>) {
            let Ok(entries) = std::fs::read_dir(dir) else { return };
            for e in entries.flatten() {
                let p = e.path();
                let rel = p.strip_prefix(base).unwrap_or(&p).to_string_lossy().replace('\\', "/");
                if p.is_dir() {
                    if depth < 3 && !rel.starts_with(BACKUPS) {
                        walk(&p, base, depth + 1, out);
                    }
                } else if is_document(&rel) {
                    if let Ok(m) = e.metadata() {
                        out.push((rel, m.modified().unwrap_or(std::time::UNIX_EPOCH), m.len()));
                    }
                }
            }
        }
        let mut out = Vec::new();
        walk(&self.root, &self.root, 0, &mut out);
        out
    }

    /// **A command in PowerShell**, in the workspace folder. UTF-8 out, so Korean reads.
    pub fn run_windows(&self, command: &str) -> Result<String, String> {
        let mut c = Command::new("powershell.exe");
        c.args(["-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command",
                &format!("[Console]::OutputEncoding=[Text.Encoding]::UTF8; {command}")]);
        c.current_dir(&self.root);
        finish(c, None, RUN_SECONDS)
    }
}

fn cut_for_refusal(text: &str) -> String {
    text.chars().take(1500).collect()
}

/// Run, feed `stdin`, wait at most `seconds`, and return stdout and stderr with the exit code.
fn finish(mut c: Command, stdin: Option<&str>, seconds: u64) -> Result<String, String> {
    use std::os::windows::process::CommandExt;
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    c.creation_flags(CREATE_NO_WINDOW)
        .stdin(if stdin.is_some() { Stdio::piped() } else { Stdio::null() })
        .stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = c.spawn().map_err(|e| format!("시작하지 못했다: {e}"))?;
    if let (Some(text), Some(mut pipe)) = (stdin, child.stdin.take()) {
        let _ = pipe.write_all(text.as_bytes());
        let _ = pipe.write_all(b"\n");
    }
    let mut out = child.stdout.take().unwrap();
    let mut err = child.stderr.take().unwrap();
    let a = std::thread::spawn(move || { let mut v = Vec::new(); let _ = out.read_to_end(&mut v); v });
    let b = std::thread::spawn(move || { let mut v = Vec::new(); let _ = err.read_to_end(&mut v); v });
    let began = Instant::now();
    let status = loop {
        if let Ok(Some(s)) = child.try_wait() {
            break Some(s);
        }
        if began.elapsed() > Duration::from_secs(seconds) {
            let _ = child.kill();
            let _ = child.wait();
            break None;
        }
        std::thread::sleep(Duration::from_millis(50));
    };
    let mut text = String::from_utf8_lossy(&a.join().unwrap_or_default()).to_string();
    let e = String::from_utf8_lossy(&b.join().unwrap_or_default()).to_string();
    if !e.trim().is_empty() {
        text.push_str("\n[stderr]\n");
        text.push_str(&e);
    }
    match status {
        //: **A nonzero exit is a failure**, so the record says *(실패)* beside it — `dir /B` in
        //: PowerShell came back as an ordinary result and was simply sent again.
        Some(s) => {
            text.push_str(&format!("\n(종료 코드 {})", s.code().unwrap_or(-1)));
            if s.success() { Ok(text.trim().to_string()) } else { Err(text.trim().to_string()) }
        }
        None => Err(format!("{seconds}초 안에 끝나지 않아 멈췄다\n{}", text.trim())),
    }
}

impl Tools for Workspace {
    fn list(&self) -> Vec<ToolSpec> {
        let t = |name: &str, does: &str, args: Value| ToolSpec { name: name.into(), does: does.into(), args };
        vec![
            t("list_files", "작업 폴더의 파일 목록을 본다", json!({"path": "."})),
            t("read_file", "작업 폴더의 파일을 읽는다", json!({"path": "notes.md"})),
            t("write_file", "파일을 새로 쓰거나 통째로 바꾼다 (원래 것은 백업된다)",
              json!({"path": "notes.md", "content": "..."})),
            t("edit_file", "파일에서 정확히 한 군데를 찾아 바꾼다",
              json!({"path": "notes.md", "find": "바꿀 부분", "replace": "새 내용"})),
            t("run", "리눅스(bash)에서 명령을 실행한다. 작업 폴더가 현재 폴더다. python3 표준 라이브러리와 Word 문서(.docx)용 python-docx(import docx), 엑셀(.xlsx)용 openpyxl 이 있다 (pandas, pip 없음 — csv 모듈을 써라). .docx 와 .xlsx 는 이것들로만 만들고 고친다. 여러 줄 코드는 python3 -c 에 넣지 말고 write_file 로 .py 파일을 만든 뒤 실행하라",
              json!({"command": "python3 main.py"})),
            t("run_windows", "Windows PowerShell 에서 명령을 실행한다. Windows 에서만 되는 일에만 쓴다",
              json!({"command": "Get-ChildItem"})),
        ]
    }

    fn call(&mut self, tool: &str, args: &Value) -> Result<String, String> {
        match tool {
            "list_files" => self.list_files(args.get("path").and_then(Value::as_str).unwrap_or(".")),
            "read_file" => {
                let rel = Self::arg(args, "path")?;
                let p = self.inside(rel)?;
                let meta = std::fs::metadata(&p).map_err(|_| format!("없는 파일: {rel}"))?;
                if meta.len() > READ_MOST {
                    return Err(format!("{rel} 은 {} 바이트라 통째로 읽지 않는다", meta.len()));
                }
                if is_document(rel) {
                    return self.read_document(rel);
                }
                let bytes = std::fs::read(&p).map_err(|e| e.to_string())?;
                Ok(String::from_utf8_lossy(&bytes).to_string())
            }
            "write_file" => {
                let rel = Self::arg(args, "path")?;
                let content = Self::arg(args, "content")?;
                if is_document(rel) {
                    return Err(format!("{rel} 은 글자 파일이 아니라 압축된 문서다 — run 에서 python3 와 \
python-docx(.docx) 나 openpyxl(.xlsx) 로 만들거나 고쳐라"));
                }
                let p = self.inside(rel)?;
                self.backup(&p, rel)?;
                if let Some(dir) = p.parent() {
                    std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
                }
                std::fs::write(&p, content).map_err(|e| e.to_string())?;
                Ok(format!("{rel} 에 {}자를 썼다", content.chars().count()))
            }
            "edit_file" => {
                let rel = Self::arg(args, "path")?;
                let find = Self::arg(args, "find")?;
                let replace = Self::arg(args, "replace")?;
                if is_document(rel) {
                    return Err(format!("{rel} 은 글자 파일이 아니라 압축된 문서다 — run 에서 python3 와 \
python-docx(.docx) 나 openpyxl(.xlsx) 로 만들거나 고쳐라"));
                }
                let p = self.inside(rel)?;
                let text = std::fs::read_to_string(&p).map_err(|_| format!("없는 파일: {rel}"))?;
                //: **Exactly once**, or the edit is a guess about which one was meant — the same
                //: rule as a click with two candidates.
                match text.matches(find).count() {
                    //: The file comes back with the refusal: `find` is literal text, and a model
                    //: that wrote a regex there sent the same regex again with nothing new to go on.
                    0 => Err(format!("{rel} 에서 찾을 부분이 없다. find 는 정규식이 아니라 파일에 있는 \
글자 그대로여야 한다. 지금 파일 내용:\n{}", cut_for_refusal(&text))),
                    1 => {
                        self.backup(&p, rel)?;
                        std::fs::write(&p, text.replacen(find, replace, 1)).map_err(|e| e.to_string())?;
                        Ok(format!("{rel} 에서 한 군데를 바꿨다"))
                    }
                    n => Err(format!("{rel} 에서 {n}군데가 맞아 어느 것인지 모른다 — 더 길게 찾아라")),
                }
            }
            "run" => {
                let command = Self::arg(args, "command")?;
                //: **A program on one line is refused here, not explained in the prompt**: told in
                //: the tool's own description, qwen3:14b still packed `with … for … if` into
                //: `python3 -c` three times running, and bash or Python rejected it each time.
                if command.contains("python3 -c") && command.chars().count() > PYTHON_ONE_LINE {
                    return Err("긴 python3 -c 는 받지 않는다. write_file 로 .py 파일을 만든 다음 python3 그파일.py 로 실행하라".into());
                }
                self.run_sandboxed(command)
            }
            "run_windows" => self.run_windows(Self::arg(args, "command")?),
            other => Err(format!("그런 도구는 없다: {other}")),
        }
    }

    fn missing(&self, paths: &[String]) -> Vec<String> {
        paths.iter().filter(|p| self.inside(p).is_ok_and(|full| !full.exists())).cloned().collect()
    }

    /// **Every document the request names is read before the model's first step** — documents
    /// only, since a text file is one the model reads for itself. `--bin documents`: in six of seven
    /// failures no named document was read before the script that changed it was written.
    fn opening(&self, what: &str) -> Vec<(String, Value)> {
        self.documents().into_iter()
            .filter(|(rel, _, _)| what.contains(rel.as_str())
                || Path::new(rel).file_name().is_some_and(|n| what.contains(n.to_string_lossy().as_ref())))
            .map(|(rel, _, _)| ("read_file".to_string(), json!({"path": rel})))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_document_is_known_by_its_extension() {
        for d in ["report.docx", "a/b.XLSX", " budget.xlsm "] {
            assert!(is_document(d), "{d}");
        }
        for t in ["notes.md", "docx", "report.docx.txt", "data.csv", "x.doc"] {
            assert!(!is_document(t), "{t}");
        }
    }

    #[test]
    fn text_is_not_written_into_a_document() {
        let dir = std::env::temp_dir().join(format!("argmind-ws-doc-{}", std::process::id()));
        let mut w = Workspace::new(dir.clone());
        let wrote = w.call("write_file", &json!({"path": "r.docx", "content": "hello"}));
        assert!(wrote.is_err_and(|e| e.contains("python-docx")));
        assert!(!dir.join("r.docx").exists(), "nothing was written");
        std::fs::write(dir.join("b.xlsx"), b"PK").unwrap();
        let edited = w.call("edit_file", &json!({"path": "b.xlsx", "find": "PK", "replace": "x"}));
        assert!(edited.is_err_and(|e| e.contains("openpyxl")));
        assert_eq!(std::fs::read(dir.join("b.xlsx")).unwrap(), b"PK", "untouched");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
