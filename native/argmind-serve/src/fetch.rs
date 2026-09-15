//! Fetching a file, and knowing it is the right one.
//!
//! **This is the part that makes an approval mean something.** Somebody pressing 예 to *"download
//! and run the Ollama installer"* is agreeing to a specific file from a specific place; if what
//! arrives is not checked, they have agreed to whatever the network handed back. So a fetch names
//! its source, arrives over TLS with the certificate verified, and — where a published digest
//! exists — is compared against it before anything runs.
//!
//! **`curl.exe` rather than a TLS crate.** It ships with Windows 10 and later, it validates
//! certificates against the system store, and it is one process rather than a dependency tree with
//! a cryptographic library in it. The alternative was `rustls` and its chain; the licence rule and
//! the size of the thing being avoided both point the same way.
//!
//! **Nothing here runs what it downloaded.** Fetching and executing are separate functions in
//! separate steps, so a reader can see which one an approval reached.

use std::path::{Path, PathBuf};

/// Where downloads land: a folder of ours under the user's temp, made fresh.
///
/// **Not beside the executable and not in the user's Downloads.** The first needs write access to
/// wherever argmind was unpacked; the second puts an installer somebody did not choose to keep into
/// the folder they look at every day.
pub fn shed() -> PathBuf {
    let mut at = std::env::temp_dir();
    at.push("argmind-setup");
    at
}

/// Download one file, checking the certificate and, if given, the digest.
///
/// `sha256` is the published hash in lower-case hex. **`None` means nobody published one**, which
/// is a fact about the source rather than permission to skip the check — the caller says so in the
/// approval text, and the person decides with that in front of them.
pub fn get(url: &str, into: &Path, sha256: Option<&str>, say: &mut dyn FnMut(&str))
           -> Result<(), String> {
    if !url.starts_with("https://") {
        //: **Plain HTTP is refused outright.** A download nobody can authenticate is one an
        //: approval cannot cover, whatever the dialog said.
        return Err(format!("{url} 은(는) https가 아닙니다 — 받지 않습니다"));
    }
    std::fs::create_dir_all(shed()).map_err(|e| format!("{e}"))?;
    say(&format!("받는 중: {url}"));

    let done = std::process::Command::new("curl.exe")
        .args(["--fail", "--location", "--silent", "--show-error",
               //: `--proto =https` and `--tlsv1.2` are belt and braces on top of the https check
               //: above: a redirect that tried to step down to plain HTTP is refused by curl
               //: rather than followed.
               "--proto", "=https", "--proto-redir", "=https", "--tlsv1.2",
               "--max-time", "1800", "--output"])
        .arg(into)
        .arg(url)
        .output()
        .map_err(|e| format!("curl을 실행하지 못했습니다 ({e})"))?;
    if !done.status.success() {
        let why = String::from_utf8_lossy(&done.stderr);
        return Err(format!("받지 못했습니다 — {}", why.trim()));
    }

    let size = std::fs::metadata(into).map(|m| m.len()).unwrap_or(0);
    if size == 0 {
        return Err("받은 파일이 비어 있습니다".into());
    }
    say(&format!("받음: {:.1} MB", size as f64 / 1_048_576.0));

    let Some(want) = sha256 else {
        return Ok(());
    };
    say("확인 중…");
    let got = digest(into)?;
    if !got.eq_ignore_ascii_case(want) {
        //: **The file goes.** Leaving a wrong download on disk invites somebody to run it later,
        //: and the name it is under says nothing about having failed a check.
        let _ = std::fs::remove_file(into);
        return Err(format!("해시가 다릅니다 — 받은 것: {got}, 기대한 것: {want}. 파일을 지웠습니다"));
    }
    say("해시 일치");
    Ok(())
}

/// The SHA-256 of a file, in lower-case hex.
///
/// Windows ships `certutil`, so this is one process rather than a hashing crate — the same trade as
/// `curl`, for the same reason.
pub fn digest(of: &Path) -> Result<String, String> {
    let done = std::process::Command::new("certutil")
        .args(["-hashfile"]).arg(of).arg("SHA256")
        .output().map_err(|e| format!("certutil을 실행하지 못했습니다 ({e})"))?;
    if !done.status.success() {
        return Err("해시를 계산하지 못했습니다".into());
    }
    //: `certutil` prints a banner, the hash on its own line, and a footer. The hash is the one line
    //: that is 64 hex characters and nothing else.
    String::from_utf8_lossy(&done.stdout)
        .lines()
        .map(|l| l.replace(' ', ""))
        .find(|l| l.len() == 64 && l.chars().all(|c| c.is_ascii_hexdigit()))
        .map(|l| l.to_lowercase())
        .ok_or_else(|| "certutil의 출력에서 해시를 찾지 못했습니다".into())
}

/// Run something that was just fetched, and wait for it.
///
/// **Separate from [`get`] on purpose.** Fetching and executing are two different things to have
/// agreed to, and a reader following an approval should be able to see which one it reached.
pub fn run(what: &Path, args: &[&str], say: &mut dyn FnMut(&str)) -> Result<(), String> {
    say(&format!("실행: {}", what.display()));
    let done = std::process::Command::new(what).args(args).status()
        .map_err(|e| format!("실행하지 못했습니다 ({e})"))?;
    if done.success() {
        return Ok(());
    }
    //: An installer the person cancelled exits non-zero, and that is a refusal rather than a fault.
    Err(match done.code() {
        Some(code) => format!("설치 프로그램이 {code} 로 끝났습니다 — 취소하셨을 수 있습니다"),
        None => "설치 프로그램이 중단되었습니다".into(),
    })
}
