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
//!   **Since 2026-11-09 it runs on a desktop of argmind's own** (`crate::apart`), as every command
//!   here does: a model refused a screen once ran `Start-Process` and the window opened on the
//!   person's desktop. What it starts is not shown to anybody and ends when the command returns.
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

use std::path::{Component, Path, PathBuf};
use std::process::Command;

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

/// What a command's result says when it left programs running, which ended with it.
pub const LEFT_ENDED: &str = "이 명령이 띄우고 남겨 둔 프로그램";

/// Run, feed `stdin`, wait at most `seconds`, and return stdout and stderr with the exit code —
/// **on argmind's own desktop and in a job of its own** (`crate::apart`), so nothing it starts
/// reaches the person's screen or outlives it.
pub(crate) fn finish(c: Command, stdin: Option<&str>, seconds: u64) -> Result<String, String> {
    let ran = crate::apart::run(&c, stdin, seconds).map_err(|e| format!("시작하지 못했다: {e}"))?;
    let mut text = ran.out;
    if !ran.err.trim().is_empty() {
        text.push_str("\n[stderr]\n");
        text.push_str(&ran.err);
    }
    if !ran.left.is_empty() {
        text.push_str(&format!("\n({LEFT_ENDED}({})을 명령과 함께 끝냈다 — 명령으로 띄운 창은 사람에게도 작업 화면에도 \
보이지 않는다. 창이 있는 프로그램은 screen_open 으로 작업 화면에서 연다)", ran.left.join(", ")));
    }
    match ran.code {
        //: **A nonzero exit is a failure**, so the record says *(실패)* beside it — `dir /B` in
        //: PowerShell came back as an ordinary result and was simply sent again.
        Some(code) => {
            text.push_str(&format!("\n(종료 코드 {code})"));
            if code == 0 { Ok(text.trim().to_string()) } else { Err(text.trim().to_string()) }
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
            t("run_windows", "Windows PowerShell 에서 명령을 실행한다. Windows 에서만 되는 일에만 쓴다. 여기서 띄운 프로그램의 창은 \
아무에게도 보이지 않고 명령이 끝날 때 함께 끝난다 — 창이 있는 프로그램은 screen_open 으로 연다",
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

/// **The Ubuntu a sandbox is made from**, by name in WSL's own distribution list — 24.04, which is
/// what the sandbox in use was made from (its `python3-docx`, 1.1.0, is 24.04's).
///
/// **Where a sandbox comes from — since 2026-11-17**, when the repository was being readied to go
/// public and a person cloning it would have met a `run` failing on a distro nobody had told them to
/// make. The one in use was made by hand; [`make_sandbox`] is the same steps, run in order on one
/// approval and checked the way it was checked. **The image is WSL's**: `wsl --install` fetches the
/// file Microsoft's list names, with the hash the list gives, so argmind downloads nothing itself.
///
/// Measured on a probe under another name the same day: installing **30 s**, the settings and the two
/// libraries **51 s**; then `/mnt/c` empty, `cmd.exe` and `powershell.exe` not found, `import docx,
/// openpyxl` working, one folder mounted by `drvfs` read and written into — and the probe removed.
pub const SANDBOX_BASE: &str = "Ubuntu-24.04";

/// What WSL downloads for [`SANDBOX_BASE`], in MB: the image its list named on 2026-11-17 is
/// 388,975,696 bytes. The packages come on top, from Ubuntu's archive.
pub const SANDBOX_MB: u64 = 390;

/// **What makes a distro a sandbox**: no drive mounted, no Windows program reachable from inside,
/// and root by default since nobody logs in. Each command then mounts one folder itself.
pub const SANDBOX_CONF: &str = "[automount]\nenabled = false\nmountFsTab = false\n\n\
[interop]\nenabled = false\nappendWindowsPath = false\n\n[user]\ndefault = root\n";

/// **Run inside the new distro, once, as root**: the settings, then the libraries a document task
/// needs — **without recommends**, which is what keeps Pillow's GPL-3 `libimagequant` out
/// (`docs/licences.toml`). `set -e`, so a failed download is a failure and not a half-made sandbox.
pub fn sandbox_setup() -> String {
    format!("set -e\ncat > /etc/wsl.conf <<'ARGMIND_CONF'\n{SANDBOX_CONF}ARGMIND_CONF\n\
             export DEBIAN_FRONTEND=noninteractive\napt-get update -qq\n\
             apt-get install -y -qq --no-install-recommends python3 {DOCUMENT_PACKAGES}\n")
}

/// **What the check reads**, one fact a line, after the distro has been restarted so its settings
/// hold. [`sandbox_checks_out`] says what each must be.
pub const SANDBOX_CHECK: &str = "echo mnt_c=$(ls -A /mnt/c 2>/dev/null | wc -l)\n\
command -v cmd.exe >/dev/null && echo cmd=yes || echo cmd=no\n\
command -v powershell.exe >/dev/null && echo ps=yes || echo ps=no\n\
python3 -c 'import docx, openpyxl' 2>/dev/null && echo docs=yes || echo docs=no\n";

/// **Is the sandbox what it says it is?** — nothing of `C:` inside, no Windows program reachable, and
/// the libraries there. Every one of the four, read off [`SANDBOX_CHECK`]'s output.
pub fn sandbox_checks_out(said: &str) -> Result<(), String> {
    let wrong: Vec<&str> = ["mnt_c=0", "cmd=no", "ps=no", "docs=yes"].into_iter()
        .filter(|want| !said.lines().any(|l| l.trim() == *want))
        .collect();
    if wrong.is_empty() {
        Ok(())
    } else {
        Err(format!("샌드박스 확인에서 어긋났다 — 기대한 {} 이(가) 아니었다: {}", wrong.join(", "),
                    said.trim().replace('\n', " · ")))
    }
}

/// Is a distro by this name in `wsl -l -q`'s output? Names are compared whole and without case,
/// and a byte-order mark or a carriage return is not part of one.
pub fn listed(out: &str, name: &str) -> bool {
    out.lines().any(|l| l.trim_matches(|c: char| c.is_whitespace() || c == '\u{feff}' || c == '\0')
        .eq_ignore_ascii_case(name))
}

/// A quick `wsl.exe` question — no window, UTF-8 out — and whether it exited 0.
fn wsl_asked(args: &[&str]) -> Option<(bool, String)> {
    use std::os::windows::process::CommandExt;
    let out = Command::new("wsl.exe").args(args).env("WSL_UTF8", "1")
        .creation_flags(0x0800_0000).output().ok()?;
    Some((out.status.success(), String::from_utf8_lossy(&out.stdout).into_owned()))
}

/// **Is WSL here at all?** `wsl --status` exits 0. Without it there is nothing to make a sandbox
/// in, and installing WSL wants an administrator and a restart — the person's to do.
pub fn wsl_here() -> bool {
    wsl_asked(&["--status"]).is_some_and(|(ok, _)| ok)
}

/// Is there a distro called `name`? About 0.15 s, measured, and it starts nothing.
pub fn sandbox_here(name: &str) -> bool {
    wsl_asked(&["-l", "-q"]).is_some_and(|(ok, out)| ok && listed(&out, name))
}

/// A long `wsl.exe` step, on argmind's own desktop (`crate::apart`) so nothing it opens reaches
/// the person, with what it printed if it failed.
fn wsl_step(args: &[&str], stdin: Option<&str>, seconds: u64) -> Result<String, String> {
    let mut c = Command::new("wsl.exe");
    c.args(args).env("WSL_UTF8", "1");
    let ran = crate::apart::run(&c, stdin, seconds).map_err(|e| format!("wsl.exe 를 시작하지 못했다: {e}"))?;
    let said = format!("{}\n{}", ran.out.trim(), ran.err.trim()).trim().to_string();
    match ran.code {
        Some(0) => Ok(said),
        Some(code) => Err(format!("wsl {} 이(가) 실패했다 (종료 코드 {code}): {}", args.first().unwrap_or(&""),
                                  said.lines().rev().take(4).collect::<Vec<_>>().into_iter().rev()
                                      .collect::<Vec<_>>().join(" · "))),
        None => Err(format!("wsl {} 이(가) {seconds}초 안에 끝나지 않았다", args.first().unwrap_or(&""))),
    }
}

/// **Make the sandbox**: install [`SANDBOX_BASE`] as `name` at `at`, write [`SANDBOX_CONF`] and the
/// libraries, restart it so the settings hold, and check all four facts. **A sandbox that does not
/// check out is removed** — one left with `C:` mounted would read as present, never be offered again,
/// and be no sandbox — and only one this call made: a name already there is left alone and reported.
pub fn make_sandbox(name: &str, at: &Path, say: &mut dyn FnMut(&str)) -> Result<String, String> {
    make_checked(name, at, SANDBOX_CHECK, say)
}

/// [`make_sandbox`] with the check as an argument, so a test can make the check fail and see the
/// half-made distro go.
fn make_checked(name: &str, at: &Path, check: &str, say: &mut dyn FnMut(&str)) -> Result<String, String> {
    if !wsl_here() {
        return Err("WSL이 없어 샌드박스를 만들 수 없다 — 관리자 PowerShell 에서 wsl --install --no-distribution 뒤 다시 시작".into());
    }
    if sandbox_here(name) {
        return Ok(format!("{name} 은(는) 이미 있다 — 새로 만들지 않았다"));
    }
    if let Some(parent) = at.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
    }
    let place = at.to_string_lossy().to_string();
    say(&format!("{SANDBOX_BASE} 를 내려받아 {name} 으로 설치합니다 (약 {SANDBOX_MB} MB)"));
    let made = (|| {
        wsl_step(&["--install", SANDBOX_BASE, "--name", name, "--location", &place, "--no-launch",
                   "--web-download"], None, 900)?;
        say("C: 와 Windows 프로그램을 막는 설정을 쓰고, Word·Excel 라이브러리를 설치합니다");
        wsl_step(&["-d", name, "-u", "root", "--", "sh", "-s"], Some(&sandbox_setup()), 900)?;
        say("설정이 들도록 다시 시작하고 확인합니다");
        wsl_step(&["--terminate", name], None, 60)?;
        let said = wsl_step(&["-d", name, "-u", "root", "--", "sh", "-s"], Some(check), 120)?;
        sandbox_checks_out(&said)
    })();
    if let Err(e) = made {
        //: **Ours to remove**: the name was not there when this call began.
        if sandbox_here(name) {
            say("끝까지 되지 않아 만들던 것을 지웁니다");
            let _ = wsl_step(&["--unregister", name], None, 120);
        }
        return Err(e);
    }
    Ok(format!("{name} 을(를) 만들었습니다 — 안에서 C: 와 Windows 프로그램이 보이지 않고, Word·Excel 라이브러리가 있습니다"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_sandbox_checks_out_only_on_all_four_facts() {
        let good = "mnt_c=0\ncmd=no\nps=no\ndocs=yes\n";
        assert!(sandbox_checks_out(good).is_ok());
        assert!(sandbox_checks_out("mnt_c=0\r\ncmd=no\r\nps=no\r\ndocs=yes\r\n").is_ok(), "CRLF");
        for (from, to) in [("mnt_c=0", "mnt_c=12"), ("cmd=no", "cmd=yes"), ("ps=no", "ps=yes"), ("docs=yes", "docs=no")] {
            let bad = good.replace(from, to);
            assert!(sandbox_checks_out(&bad).is_err_and(|e| e.contains(from)), "{bad:?} passed");
        }
        assert!(sandbox_checks_out("").is_err(), "nothing read is not a sandbox");
        //: Inside a longer line is not the fact.
        assert!(sandbox_checks_out("xmnt_c=0\ncmd=no\nps=no\ndocs=yes").is_err());
    }

    #[test]
    fn a_distro_is_listed_by_its_whole_name() {
        let out = "\u{feff}Ubuntu\r\nargmind-sandbox\r\n";
        assert!(listed(out, "argmind-sandbox"));
        assert!(listed(out, "ARGMIND-SANDBOX"));
        assert!(!listed(out, "argmind"));
        assert!(!listed("argmind-sandbox-probe\n", "argmind-sandbox"));
        assert!(!listed("", "argmind-sandbox"));
    }

    /// **What the setup writes is the sandbox's settings**, and installs without recommends.
    #[test]
    fn the_setup_writes_the_settings_and_leaves_recommends_out() {
        let s = sandbox_setup();
        for want in ["set -e", "[automount]\nenabled = false", "[interop]\nenabled = false",
                     "appendWindowsPath = false", "default = root", "--no-install-recommends",
                     DOCUMENT_PACKAGES] {
            assert!(s.contains(want), "{want:?} missing from:\n{s}");
        }
        //: The heredoc's end marker on its own line, or the settings swallow the commands after it.
        assert!(s.contains("\nARGMIND_CONF\n"), "{s}");
    }

    /// **Made, checked and removed, under another name** — never `argmind-sandbox`. Downloads about
    /// 390 MB and takes a minute or two, so it runs only by hand:
    /// `cargo test -p argmind-env-windows --lib making_a_sandbox -- --ignored --nocapture`.
    /// The second half makes the check fail on purpose and wants the half-made distro gone.
    #[test]
    #[ignore]
    fn making_a_sandbox_under_another_name_and_removing_it() {
        let name = format!("argmind-sandbox-probe-{}", std::process::id());
        assert_ne!(name, SANDBOX);
        let at = std::env::temp_dir().join(&name);
        let mut say = |s: &str| eprintln!("  {s}");
        let made = make_sandbox(&name, &at, &mut say);
        eprintln!("{made:?}");
        let there = sandbox_here(&name);
        let _ = wsl_step(&["--unregister", &name], None, 120);
        assert!(made.is_ok() && there, "{made:?}");
        assert!(!sandbox_here(&name), "the probe was not removed");

        let failed = make_checked(&name, &at, "echo mnt_c=5\n", &mut say);
        eprintln!("{failed:?}");
        assert!(failed.is_err_and(|e| e.contains("mnt_c=0")));
        assert!(!sandbox_here(&name), "a sandbox that did not check out was left behind");
        let _ = std::fs::remove_dir_all(&at);
    }

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
