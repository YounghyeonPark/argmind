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
            t("run", "리눅스(bash)에서 명령을 실행한다. 작업 폴더가 현재 폴더다. python3 표준 라이브러리만 있다 (pandas, pip 없음 — csv 모듈을 써라). 여러 줄 코드는 python3 -c 에 넣지 말고 write_file 로 .py 파일을 만든 뒤 실행하라",
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
                let bytes = std::fs::read(&p).map_err(|e| e.to_string())?;
                Ok(String::from_utf8_lossy(&bytes).to_string())
            }
            "write_file" => {
                let rel = Self::arg(args, "path")?;
                let content = Self::arg(args, "content")?;
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
}
