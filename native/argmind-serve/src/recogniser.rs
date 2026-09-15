//! Getting the recogniser onto this machine, from source, without asking twice.
//!
//! Voice input needs two things argmind does not ship: a **model**, which is data, and a
//! **server**, which is a program. Ollama publishes an installer; whisper.cpp does not — the
//! supported way to have it is to build it. So this clones, configures, compiles, and starts the
//! result.
//!
//! **The commit is pinned by its SHA, and that is the strongest verification available here.** A
//! published hash checks a file against a number somebody wrote down; a git object name *is* the
//! content, so asking for `8a9ad78…` and getting anything else is not something git can be talked
//! into. A tag can be moved and a branch certainly moves. Neither is asked for.
//!
//! **It is still somebody else's code running on this machine**, and the window says so in those
//! words. Compiling it yourself changes who produced the bytes, not who wrote the instructions.
//!
//! **What is built here dies with argmind** — see [`argmind_env_windows::supervise`]. A recogniser
//! left listening on a port after the program that started it has gone is a process the next
//! argmind will find and cannot account for.

use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use argmind_env_windows::supervise::Supervised;

use crate::fix::Fix;

/// Where the source comes from, and exactly which commit.
pub const REPO: &str = "https://github.com/ggml-org/whisper.cpp.git";
/// whisper.cpp v1.7.4. **A SHA rather than the tag**: the tag points here today.
pub const COMMIT: &str = "8a9ad7844d6e2a10cddf4b92de4089d7ac2b14a9";
/// The cmake target. One binary, not the whole tree of examples.
pub const TARGET: &str = "whisper-server";

pub const MODEL_URL: &str =
    "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin";
/// Published by the project, and checked against what arrives.
pub const MODEL_SHA256: &str =
    "60ed5bc3dd14eea856493d334349b405782ddcaf0028d4b5df4088345fba2efe";
/// **Measured**, not estimated: 147,951,465 bytes.
pub const MODEL_MB: u64 = 141;

/// How long to say it might take, as an **upper bound rather than an estimate**.
///
/// **Measured**, twice, on the machine this was written on — MinGW g++ 16.1, `-march=native`:
/// 4 minutes at `-j 8`, and **121 seconds for the whole sequence at `-j 32`**, that figure
/// including the 141 MB model download and the wait for the server to answer.
///
/// Ten is neither of those on purpose. A number told to somebody before they commit an evening
/// should be the one they will not be disappointed by, and the machines this will run on are not
/// this one — the rate carries its conditions and those conditions are a 32-thread desktop.
pub const MINUTES: u64 = 10;

/// Where the source tree and the build live.
///
/// Beside the person's own files rather than beside the executable, for the same reason as the
/// models: a folder under the program needs write access to wherever argmind was unpacked.
pub fn shed() -> PathBuf {
    let mut at = std::env::var("LOCALAPPDATA").map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir());
    at.push("argmind");
    at.push("whisper.cpp");
    at
}

pub fn server_exe() -> PathBuf {
    shed().join("build").join("bin").join("whisper-server.exe")
}

/// Which commit the binary sitting there was built from, if anything was.
///
/// **A file rather than `git rev-parse`.** This is read every time the page asks what is missing,
/// and a subprocess per check is a cost paid forever for an answer that only changes when a build
/// finishes. The build writes it; nothing else does.
pub fn built_at() -> Option<String> {
    std::fs::read_to_string(server_exe().with_file_name("built-at.txt"))
        .ok().map(|s| s.trim().to_string())
}

/// Is there a server here already, built from the commit argmind would ask for?
///
/// **The commit matters, not just the file.** A binary left by an older argmind is not the one the
/// window would be describing, and saying "it is already built" about it would be a lie that only
/// shows up as a recogniser behaving unlike the one documented.
pub fn ready() -> bool {
    server_exe().is_file() && built_at().as_deref() == Some(COMMIT)
}

/// Record what was built, so the next question can be the shorter one.
pub fn remember_build(commit: &str) {
    let _ = std::fs::write(server_exe().with_file_name("built-at.txt"), commit);
}

pub fn model_at() -> PathBuf {
    crate::needs::model_shed().join("ggml-base.bin")
}

/// The fix that would fetch the model on its own.
///
/// [`crate::fix::apply`] runs *this* for the model half rather than a second copy of the same
/// download, so the two cannot come to mean different things.
pub fn model_fix() -> Fix {
    Fix::Fetch {
        url: MODEL_URL.into(),
        into: model_at().to_string_lossy().to_string(),
        why: "받아쓰기 모델입니다.".into(),
        about_mb: MODEL_MB,
        sha256: Some(MODEL_SHA256.into()),
    }
}

/// What a build needs and this machine has not got.
///
/// **Checked before the button is offered, not after it is pressed.** A fix that cannot work is a
/// button that fails, and a person who pressed it has learned nothing except that argmind is
/// unreliable. Each is asked for its version, which is the cheapest way to find out whether the
/// name on the path is really the program.
pub fn missing_tools() -> Vec<&'static str> {
    ["git", "cmake", "g++", "mingw32-make", "objdump"].into_iter()
        .filter(|tool| {
            Command::new(tool).arg("--version")
                .stdout(Stdio::null()).stderr(Stdio::null()).stdin(Stdio::null())
                .status().map(|s| !s.success()).unwrap_or(true)
        })
        .collect()
}

/// Run a command, reporting the lines worth reporting and keeping the rest for the error.
///
/// Both streams are drained — **stderr on its own thread**, because a pipe nobody reads fills up
/// and the child stops on the write, which looks exactly like a hung build.
fn walk(mut command: Command, say: &mut dyn FnMut(&str),
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
pub fn clone_at(repo: &str, commit: &str, into: &Path, say: &mut dyn FnMut(&str))
                -> Result<(), String> {
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
    Ok(())
}

/// Configure and compile, reporting the percentage as it goes.
///
/// **A button that is silent for ten minutes is a button people press twice.** cmake prints
/// `[ 42%]` on stdout, which is exactly the thing to pass on and nothing else is.
pub fn build(at: &Path, target: &str, say: &mut dyn FnMut(&str)) -> Result<(), String> {
    say("빌드 준비 중…");
    let mut configure = Command::new("cmake");
    configure.current_dir(at).args([
        "-B", "build", "-G", "MinGW Makefiles",
        "-DCMAKE_BUILD_TYPE=Release",
        //: Only the one binary is wanted, and the tests download audio.
        "-DWHISPER_BUILD_TESTS=OFF",
        "-DWHISPER_BUILD_EXAMPLES=ON",
    ]);
    walk(configure, say, &|l| l.starts_with("-- Configuring") || l.starts_with("-- Generating"))?;

    let jobs = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    say(&format!("컴파일 중 — {jobs}개 동시, 몇 분 걸립니다"));
    let mut make = Command::new("cmake");
    make.current_dir(at).args(["--build", "build", "--config", "Release",
                               "--target", target, "-j", &jobs.to_string()]);
    let mut told = 0usize;
    walk(make, &mut |line| {
        //: Every ten per cent. A build emits hundreds of these and a page that scrolls past says
        //: no more than one that says nothing.
        if let Some(pct) = percent_of(line) {
            if pct >= told + 10 || pct == 100 {
                told = pct - pct % 10;
                say(&format!("{pct}%"));
            }
        }
    }, &|l| l.trim_start().starts_with('['))?;
    Ok(())
}

/// The `42` in `[ 42%] Building CXX object …`, and `None` for every other line.
pub fn percent_of(line: &str) -> Option<usize> {
    let line = line.trim_start();
    let inside = line.strip_prefix('[')?.split(']').next()?;
    inside.trim().strip_suffix('%')?.trim().parse().ok()
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

/// The one recogniser argmind started, held for as long as argmind runs.
///
/// A global because there is exactly one: a second server on the same port would not start, and a
/// second on another port is not a thing anything here asks for. **Its cleanup is the job object's
/// rather than this value's** — a static is never dropped, which is precisely why the guarantee
/// was put in the operating system instead.
static RUNNING: Mutex<Option<Supervised>> = Mutex::new(None);

/// Is a recogniser answering on the port?
pub fn answering() -> bool {
    std::net::TcpStream::connect_timeout(
        &argmind_env_whisper::WHISPER.parse().expect("a literal address"),
        Duration::from_millis(400)).is_ok()
}

/// Did *argmind* start the one that is running?
pub fn ours() -> bool {
    RUNNING.lock().map(|mut it| it.as_mut().is_some_and(Supervised::alive)).unwrap_or(false)
}

/// Start the built server and wait until it answers.
///
/// **Waiting is the point.** Loading the model takes a few seconds, and reporting success on a
/// process that has started but is not yet listening produces a first attempt at speaking that
/// fails for no visible reason.
pub fn serve(say: &mut dyn FnMut(&str)) -> Result<(), String> {
    let exe = server_exe();
    if !exe.is_file() {
        return Err(format!("{} 이(가) 없습니다", exe.display()));
    }
    let model = model_at();
    if !model.is_file() {
        return Err(format!("{} 이(가) 없습니다", model.display()));
    }
    let (host, port) = argmind_env_whisper::WHISPER.rsplit_once(':')
        .ok_or("주소가 이상합니다")?;

    let log = shed().join("server.log");
    let model = model.to_string_lossy().to_string();
    say(&format!("받아쓰기 서버를 {} 에 띄웁니다", argmind_env_whisper::WHISPER));
    let started = Supervised::start(
        &exe,
        &["-m", &model, "--host", host, "--port", port, "-l", "ko"],
        //: From the source tree: the server serves its own page folder relative to here.
        Some(&shed()),
        Some(&log),
    ).map_err(|e| format!("서버를 시작하지 못했습니다 ({e})"))?;
    let mut started = started;

    let by = Instant::now() + Duration::from_secs(90);
    while Instant::now() < by {
        if answering() {
            let promise = if started.supervised() {
                "argmind를 닫으면 함께 꺼집니다"
            } else {
                //: The weaker promise, said as the weaker promise.
                "argmind가 정상적으로 끝나면 함께 꺼집니다"
            };
            say(&format!("서버가 응답합니다 — {promise}"));
            *RUNNING.lock().map_err(|_| "잠금 실패")? = Some(started);
            return Ok(());
        }
        if !started.alive() {
            let why = std::fs::read_to_string(&log).unwrap_or_default();
            let all: Vec<&str> = why.lines().collect();
            return Err(format!("서버가 바로 꺼졌습니다 —\n{}",
                               last_few(&all, 8).iter().map(|l| format!("    {l}"))
                                   .collect::<Vec<_>>().join("\n")));
        }
        std::thread::sleep(Duration::from_millis(300));
    }
    Err(format!("90초 안에 응답하지 않았습니다 — {} 를 보세요", log.display()))
}
