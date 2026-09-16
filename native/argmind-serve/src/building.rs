//! Building somebody else's program on this machine: one commit, one compiler, one binary.
//!
//! **Two things are built this way now** — the recogniser in [`crate::recogniser`] and the voice in
//! [`crate::speaker`] — and this file is what they share. It was extracted when the second arrived
//! rather than copied, for the reason [`crate::fix`] gives about `apply`: the two things that must
//! never drift apart are *what the window said* and *what happened*, and a second copy of the doing
//! is exactly how they come to differ.
//!
//! **No constant lives here.** The repository, the commit, the flags, the arguments and the sizes
//! belong beside the thing they describe. A build is a [`Recipe`] handed to this file, not a branch
//! inside it — so adding a third would add no line here at all.
//!
//! ## What the second build taught, which the first had no way to find out
//!
//! Three failures, all of which look like something else, all of them now handled rather than
//! remembered. They are written down because each cost an afternoon and none of them names itself.
//!
//! **The compiler's own directory has to come first on the child's PATH.** A build that runs a
//! program it just compiled — a shader generator, in the case that found this — loads that
//! program's DLLs by PATH order, and Git for Windows ships a `mingw64\bin` full of libraries with
//! exactly the names the toolchain uses. The result was `0xC0000139`, *entry point not found*,
//! from a binary that had linked cleanly ten seconds earlier. **This is the same defect
//! [`bring_runtime`] fixes for the shipped binary**, arriving during the build instead of after it,
//! and it is fixed the same way: say which directory, rather than hope.
//!
//! **A build directory can be too long.** Windows still has a 260-character limit for a great many
//! callers, cmake's nested try-compile scratch paths are about 130 characters on their own, and the
//! error it produces is *"The C compiler is not able to compile a simple test program"* — which
//! reads as a broken toolchain and is not. [`Recipe::into`] is the build folder's name and it is
//! short on purpose.
//!
//! **A process does not see an environment variable that was set after it started.** This is not
//! obscure and it is easy to miss here, because the sequence that hits it is the ordinary one:
//! argmind asks the person to install a build tool, they install it, and argmind — still running,
//! with the environment it was born with — cannot find it. [`machine_env`] reads the machine's
//! setting from where the installer actually wrote it.

use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// One thing to build, described where it belongs rather than here.
pub struct Recipe {
    /// The source tree.
    pub at: PathBuf,
    /// The build folder, **relative to `at` and short on purpose** — see the note above about the
    /// 260-character limit. `build` is fine for a tree that does not generate anything; a tree that
    /// does may need `b`.
    pub into: String,
    /// The one cmake target. Not the whole tree of examples.
    pub target: String,
    /// `-D` arguments, whole and already spelled.
    pub defines: Vec<String>,
    /// Environment the child needs and this process may not have. See [`machine_env`].
    pub env: Vec<(String, String)>,
    /// Directories to put at the **front** of the child's PATH, ahead of whatever is already there.
    pub path_first: Vec<PathBuf>,
}

impl Recipe {
    /// Configure and compile, reporting the percentage as it goes.
    ///
    /// **A button that is silent for ten minutes is a button people press twice.** cmake prints
    /// `[ 42%]` on stdout, which is exactly the thing to pass on and nothing else is.
    pub fn run(&self, say: &mut dyn FnMut(&str)) -> Result<(), String> {
        say("빌드 준비 중…");
        let mut configure = Command::new("cmake");
        self.dress(&mut configure);
        configure.args(["-B", &self.into, "-G", "MinGW Makefiles", "-DCMAKE_BUILD_TYPE=Release"]);
        configure.args(&self.defines);
        walk(configure, say,
             &|l| l.starts_with("-- Configuring") || l.starts_with("-- Generating"))?;

        let jobs = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
        say(&format!("컴파일 중 — {jobs}개 동시, 몇 분 걸립니다"));
        let mut make = Command::new("cmake");
        self.dress(&mut make);
        make.args(["--build", &self.into, "--config", "Release",
                   "--target", &self.target, "-j", &jobs.to_string()]);
        let mut told = 0usize;
        walk(make, &mut |line| {
            //: Every ten per cent. A build emits hundreds of these and a page that scrolls past
            //: says no more than one that says nothing.
            if let Some(pct) = percent_of(line) {
                //: **A target with dependencies counts to a hundred once per sub-build.** The
                //: voice's does it six times, and six lines reading `100%` is a progress bar that
                //: has stopped meaning anything. A number lower than the last one is a new
                //: sub-build starting, so the floor goes back down with it.
                if pct < told {
                    told = 0;
                }
                if pct >= told + 10 || (pct == 100 && told < 100) {
                    told = pct - pct % 10;
                    say(&format!("{pct}%"));
                }
            }
        }, &|l| l.trim_start().starts_with('['))?;
        Ok(())
    }

    /// The working directory, the environment and the PATH the child is given.
    ///
    /// **PATH is rebuilt rather than appended to**, because the whole point is order: a directory
    /// added at the end is a directory that loses to Git's.
    fn dress(&self, command: &mut Command) {
        command.current_dir(&self.at);
        for (name, value) in &self.env {
            command.env(name, value);
        }
        let mut ahead: Vec<PathBuf> = self.path_first.clone();
        //: The compiler's own directory, always — see the module note. Asked for by
        //: `where g++.exe`, so it is the compiler that will actually run.
        if let Some(mine) = toolchain_dir() {
            if !ahead.contains(&mine) {
                ahead.insert(0, mine);
            }
        }
        if ahead.is_empty() {
            return;
        }
        let existing = std::env::var_os("PATH").unwrap_or_default();
        let mut all: Vec<PathBuf> = ahead;
        all.extend(std::env::split_paths(&existing));
        if let Ok(joined) = std::env::join_paths(all) {
            command.env("PATH", joined);
        }
    }
}

/// Run a command, reporting the lines worth reporting and keeping the rest for the error.
///
/// Both streams are drained — **stderr on its own thread**, because a pipe nobody reads fills up
/// and the child stops on the write, which looks exactly like a hung build.
pub fn walk(mut command: Command, say: &mut dyn FnMut(&str),
            worth_saying: &dyn Fn(&str) -> bool) -> Result<(), String> {
    command.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = command.spawn().map_err(|e| format!("실행하지 못했습니다 ({e})"))?;

    let complaints = child.stderr.take().map(|err| {
        std::thread::spawn(move || {
            BufReader::new(err).lines().map_while(Result::ok).collect::<Vec<_>>()
        })
    });
    let mut last = Vec::new();
    if let Some(out) = child.stdout.take() {
        for line in BufReader::new(out).lines().map_while(Result::ok) {
            if worth_saying(&line) {
                say(line.trim());
            }
            //: A tail, not the whole log: a failing build prints thousands of lines and the ones
            //: that say why are at the end.
            last.push(line);
            if last.len() > 12 {
                last.remove(0);
            }
        }
    }
    let how = child.wait().map_err(|e| format!("{e}"))?;
    let complaints = complaints.and_then(|t| t.join().ok()).unwrap_or_default();
    if how.success() {
        return Ok(());
    }
    let tail = last_few(&complaints, 12).iter().chain(last.iter());
    Err(format!("실패했습니다 —\n{}",
                tail.map(|l| format!("    {}", l.trim_end())).collect::<Vec<_>>().join("\n")))
}

/// The last `how_many`, in order. **The end of a log is where it says why.**
pub fn last_few<T>(all: &[T], how_many: usize) -> &[T] {
    &all[all.len().saturating_sub(how_many)..]
}

fn git(at: &Path, args: &[&str]) -> Result<String, String> {
    let out = Command::new("git").current_dir(at).args(args).stdin(Stdio::null())
        .output().map_err(|e| format!("git을 실행하지 못했습니다 ({e})"))?;
    if !out.status.success() {
        return Err(format!("git {} 실패 — {}", args.join(" "),
                           String::from_utf8_lossy(&out.stderr).trim()));
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

/// Put exactly one commit of one repository into a folder, and prove that is what is there.
///
/// `git fetch <sha>` asks for a commit by its content name; git verifies every object it writes
/// against the name it was asked for, so a server that returned something else fails the fetch
/// rather than the check afterwards. The `rev-parse` at the end is belt and braces, and it is the
/// line that would catch a folder somebody had edited between runs.
///
/// **`submodules` is not free and is not always wanted.** A tree that vendors ggml as a submodule
/// does not compile without it; a tree that vendors it as files does not need the extra fetch. The
/// caller knows which it is, so the caller says.
pub fn clone_at(repo: &str, commit: &str, into: &Path, submodules: bool,
                say: &mut dyn FnMut(&str)) -> Result<(), String> {
    if into.join(".git").is_dir() {
        if let Ok(head) = git(into, &["rev-parse", "HEAD"]) {
            if head == commit {
                say("소스는 이미 받아져 있습니다");
                return Ok(());
            }
        }
        //: A tree at some other commit is not ours to reconcile. Starting again costs a minute and
        //: leaves nothing ambiguous.
        say("다른 커밋이 있어 다시 받습니다");
        std::fs::remove_dir_all(into).map_err(|e| format!("{e}"))?;
    }
    std::fs::create_dir_all(into).map_err(|e| format!("{e}"))?;
    say(&format!("소스를 받는 중: {repo} @ {}", &commit[..12]));
    git(into, &["init", "-q", "."])?;
    //: `remote add` fails if it is already there, which is fine — this is a fresh folder.
    let _ = git(into, &["remote", "add", "origin", repo]);
    git(into, &["fetch", "--depth", "1", "--quiet", "origin", commit])?;
    git(into, &["checkout", "--quiet", "FETCH_HEAD"])?;

    let head = git(into, &["rev-parse", "HEAD"])?;
    if head != commit {
        return Err(format!("받은 커밋이 다릅니다 — {head}, 기대한 것 {commit}"));
    }
    say(&format!("커밋 확인: {}", &head[..12]));
    if submodules {
        //: **Pinned by the superproject, not by a branch.** A submodule is recorded in the parent
        //: commit as an object name, so this is the same guarantee the fetch above gives, one level
        //: down — and the parent commit was just verified.
        say("딸린 소스를 받는 중…");
        git(into, &["submodule", "update", "--init", "--recursive", "--depth", "1", "--quiet"])?;
    }
    Ok(())
}

/// The `42` in `[ 42%] Building CXX object …`, and `None` for every other line.
pub fn percent_of(line: &str) -> Option<usize> {
    let line = line.trim_start();
    let inside = line.strip_prefix('[')?.split(']').next()?;
    inside.trim().strip_suffix('%')?.trim().parse().ok()
}

/// Which of these programs this machine has not got.
///
/// **Checked before the button is offered, not after it is pressed.** A fix that cannot work is a
/// button that fails, and a person who pressed it has learned nothing except that argmind is
/// unreliable. Each is asked for its version, which is the cheapest way to find out whether the
/// name on the path is really the program.
pub fn missing(tools: &[&'static str]) -> Vec<&'static str> {
    tools.iter().copied()
        .filter(|tool| {
            Command::new(tool).arg("--version")
                .stdout(Stdio::null()).stderr(Stdio::null()).stdin(Stdio::null())
                .status().map(|s| !s.success()).unwrap_or(true)
        })
        .collect()
}

/// A machine-wide environment variable, **read from where the installer wrote it** rather than from
/// this process.
///
/// A running program keeps the environment it was given at birth. The sequence that makes this
/// matter is the ordinary one and not an edge case at all: argmind says a build tool is missing,
/// the person installs it, the installer sets a machine variable, and argmind — the same process,
/// minutes later — still cannot see it. Every fix in this program is designed so that saying yes
/// once is enough, and a build that failed here would be the person saying yes and then being told
/// to restart the program.
///
/// `reg.exe` rather than a registry crate: no dependency, and it is the same key the installer
/// wrote to. The value is whatever follows the type, **taken to the end of the line** because a
/// path may contain spaces.
pub fn machine_env(name: &str) -> Option<String> {
    if let Ok(mine) = std::env::var(name) {
        if !mine.trim().is_empty() {
            return Some(mine);
        }
    }
    let out = Command::new("reg")
        .args(["query", r"HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Environment",
               "/v", name])
        .stdin(Stdio::null()).output().ok()?;
    if !out.status.success() {
        return None;
    }
    value_in(&String::from_utf8_lossy(&out.stdout), name)
}

/// The value out of what `reg query /v NAME` printed.
///
/// **Its own function so it can be given text.** The failure this shape has is silent: a path with
/// a space in it, split on whitespace, becomes `C:\Program` and the build then fails somewhere
/// else entirely. The type name is the separator and everything after it is the value.
pub fn value_in(printed: &str, name: &str) -> Option<String> {
    let line = printed.lines().find(|l| l.trim_start().starts_with(name))?;
    let after = ["REG_SZ", "REG_EXPAND_SZ"].iter()
        .find_map(|kind| line.split_once(kind))?.1;
    let value = after.trim();
    (!value.is_empty()).then(|| value.to_string())
}

/// Copy the compiler's own runtime libraries next to the binary.
///
/// **This is the step whose absence would only show up on somebody else's machine.** A MinGW build
/// links against `libstdc++-6.dll` and friends, which live beside the compiler and are not on the
/// ordinary Windows path — argmind found the server here because a shell had put them there, and a
/// program started from the tray would have got a missing-DLL box instead.
///
/// **The list is read from the binary, not remembered.** `objdump -p` names what it actually
/// imports, transitively, and the filter for "is this ours to copy" is *does the toolchain have a
/// file by that name* — so a system library is never touched and a rebuild with a different
/// compiler brings the right set without this having to be updated.
pub fn bring_runtime(exe: &Path, say: &mut dyn FnMut(&str)) -> Result<usize, String> {
    let Some(beside) = toolchain_dir() else {
        return Err("컴파일러를 찾지 못했습니다".into());
    };
    let Some(here) = exe.parent() else { return Err("경로가 이상합니다".into()) };

    let mut todo = vec![exe.to_path_buf()];
    let mut seen: Vec<String> = Vec::new();
    let mut copied = 0;
    while let Some(look_at) = todo.pop() {
        for name in imports_of(&look_at)? {
            let key = name.to_lowercase();
            if seen.contains(&key) {
                continue;
            }
            seen.push(key);
            let from = beside.join(&name);
            if !from.is_file() {
                //: Not the toolchain's — a Windows library, and nothing to do.
                continue;
            }
            let to = here.join(&name);
            if !to.exists() {
                std::fs::copy(&from, &to).map_err(|e| format!("{name}을(를) 못 옮겼습니다 ({e})"))?;
                copied += 1;
            }
            //: What it brings with it, in turn.
            todo.push(to);
        }
    }
    if copied > 0 {
        say(&format!("실행에 필요한 라이브러리 {copied}개를 함께 두었습니다"));
    }
    Ok(copied)
}

/// The DLL names a binary imports, as `objdump -p` reports them.
pub fn imports_of(binary: &Path) -> Result<Vec<String>, String> {
    let out = Command::new("objdump").arg("-p").arg(binary).stdin(Stdio::null())
        .output().map_err(|e| format!("objdump를 실행하지 못했습니다 ({e})"))?;
    if !out.status.success() {
        return Err(format!("{}의 의존성을 읽지 못했습니다", binary.display()));
    }
    Ok(String::from_utf8_lossy(&out.stdout).lines()
        .filter_map(|l| l.trim().strip_prefix("DLL Name:"))
        .map(|n| n.trim().to_string())
        .collect())
}

/// The folder the C++ compiler lives in, which is where its runtime lives too.
pub fn toolchain_dir() -> Option<PathBuf> {
    let out = Command::new("where").arg("g++.exe").stdin(Stdio::null()).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let first = String::from_utf8_lossy(&out.stdout).lines().next()?.trim().to_string();
    PathBuf::from(first).parent().map(Path::to_path_buf)
}

/// Where a thing argmind builds keeps its source and its build.
///
/// Beside the person's own files rather than beside the executable, for the same reason as the
/// models: a folder under the program needs write access to wherever argmind was unpacked.
///
/// **Short, and that is not cosmetic** — see the module note on the 260-character limit.
pub fn shed(named: &str) -> PathBuf {
    let mut at = std::env::var("LOCALAPPDATA").map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir());
    at.push("argmind");
    at.push(named);
    at
}

/// Which commit the binary sitting there was built from, if anything was.
///
/// **A file rather than `git rev-parse`.** This is read every time the page asks what is missing,
/// and a subprocess per check is a cost paid forever for an answer that only changes when a build
/// finishes. The build writes it; nothing else does.
pub fn built_at(exe: &Path) -> Option<String> {
    std::fs::read_to_string(exe.with_file_name("built-at.txt"))
        .ok().map(|s| s.trim().to_string())
}

/// Record what was built, so the next question can be the shorter one.
pub fn remember_build(exe: &Path, commit: &str) {
    let _ = std::fs::write(exe.with_file_name("built-at.txt"), commit);
}

/// Is there a binary here already, built from the commit argmind would ask for?
///
/// **The commit matters, not just the file.** A binary left by an older argmind is not the one the
/// window would be describing, and saying "it is already built" about it would be a lie that only
/// shows up as a program behaving unlike the one documented.
pub fn ready(exe: &Path, commit: &str) -> bool {
    exe.is_file() && built_at(exe).as_deref() == Some(commit)
}
