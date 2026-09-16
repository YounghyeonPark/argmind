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
//!
//! **The machinery is in [`crate::building`]**, which this and [`crate::speaker`] share. What is
//! left here is what is true of the recogniser and of nothing else: which repository, which commit,
//! which model, and what the server is told when it starts.

use std::path::PathBuf;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use argmind_env_windows::supervise::Supervised;

use crate::building::{self, Recipe};
use crate::fix::Fix;

//: Re-exported rather than moved out of reach: these are the same functions under the same names,
//: and the tests that check them are checking the thing that runs.
pub use crate::building::{bring_runtime, imports_of, last_few, percent_of, toolchain_dir};

/// Where the source comes from, and exactly which commit.
pub const REPO: &str = "https://github.com/ggml-org/whisper.cpp.git";
/// whisper.cpp v1.7.4. **A SHA rather than the tag**: the tag points here today.
pub const COMMIT: &str = "8a9ad7844d6e2a10cddf4b92de4089d7ac2b14a9";
/// The cmake target. One binary, not the whole tree of examples.
pub const TARGET: &str = "whisper-server";
/// What a build needs on the machine before it can start.
pub const TOOLS: [&str; 5] = ["git", "cmake", "g++", "mingw32-make", "objdump"];

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
pub fn shed() -> PathBuf {
    building::shed("whisper.cpp")
}

pub fn server_exe() -> PathBuf {
    shed().join("build").join("bin").join("whisper-server.exe")
}

/// Which commit the binary sitting there was built from, if anything was.
pub fn built_at() -> Option<String> {
    building::built_at(&server_exe())
}

/// Is there a server here already, built from the commit argmind would ask for?
pub fn ready() -> bool {
    building::ready(&server_exe(), COMMIT)
}

/// Record what was built, so the next question can be the shorter one.
pub fn remember_build(commit: &str) {
    building::remember_build(&server_exe(), commit);
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
pub fn missing_tools() -> Vec<&'static str> {
    building::missing(&TOOLS)
}

/// Put one commit of whisper.cpp in the shed.
///
/// **No submodules**: whisper.cpp vendors ggml as files in its own tree rather than as a link, so
/// the shallow fetch above is the whole source.
pub fn clone_at(repo: &str, commit: &str, into: &std::path::Path, say: &mut dyn FnMut(&str))
                -> Result<(), String> {
    building::clone_at(repo, commit, into, false, say)
}

/// Configure and compile.
pub fn build(at: &std::path::Path, target: &str, say: &mut dyn FnMut(&str)) -> Result<(), String> {
    Recipe {
        at: at.to_path_buf(),
        //: A tree that generates nothing during its own build, so the ordinary name fits inside the
        //: path limit that bit the voice — see [`crate::building`].
        into: "build".into(),
        target: target.into(),
        defines: vec![
            //: Only the one binary is wanted, and the tests download audio.
            "-DWHISPER_BUILD_TESTS=OFF".into(),
            "-DWHISPER_BUILD_EXAMPLES=ON".into(),
        ],
        env: Vec::new(),
        path_first: Vec::new(),
    }.run(say)
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
