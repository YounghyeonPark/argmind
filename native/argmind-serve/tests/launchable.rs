//! **The application has to start when somebody double-clicks it**, not only under `cargo run`.
//!
//! This is `bring_runtime`'s rule turned on argmind itself. That function exists because a MinGW
//! build of whisper.cpp ran from a shell and failed from the tray: the compiler's libraries were
//! on the shell's path and on nothing else. **The same defect arrived again on 2026-09-16, in the
//! other direction**, when the page moved into a window of its own:
//!
//! * `cargo run -p argmind-window --example peek` opened a window and rendered a page.
//! * `argmind-app.exe`, started as a file, exited immediately with `0xC0000135`, which is
//!   *STATUS_DLL_NOT_FOUND* — no window, no message, nothing in the log.
//!
//! The difference was `WebView2Loader.dll`. `cargo run` puts every `rustc-link-search` path from
//! every build script onto the child's `PATH`; double-clicking puts none of them there. The binary
//! was fine and the way it had been tested was not.
//!
//! So: read what the built executable imports and check that each one is either a system library
//! or a file sitting beside it. **Nothing here trusts a list of names** — the imports come from the
//! binary and the answer to *is this ours to ship* is whether Windows can find it without help.
//!
//! Needs no network, no model and no display. It needs `objdump`, which is already required to
//! build the recogniser, and says so and stops rather than passing if it is absent.

#[cfg(windows)]
mod windows_only {
    use std::path::{Path, PathBuf};
    use std::process::Command;

    /// Where a launched program looks, **read from the registry rather than from this process.**
    ///
    /// **The first version of this asked `where.exe`, and it passed a binary that cannot start.**
    /// That is the same trap the test exists to catch, one level up: `cargo test` puts every build
    /// script's `rustc-link-search` path onto the test process's `PATH`, so inside the test the
    /// missing DLL is findable and outside it is not. A check that inherits the environment it is
    /// checking for pollution is not a check.
    ///
    /// So the search list is built the way the loader builds it — the system directories, then the
    /// machine and user `Path` values as the registry holds them, which is what a program started
    /// from Explorer is given.
    fn where_the_loader_looks() -> Vec<PathBuf> {
        let mut places = Vec::new();
        if let Ok(windows) = std::env::var("SystemRoot") {
            let root = PathBuf::from(windows);
            places.push(root.join("System32"));
            places.push(root.join("System"));
            places.push(root);
        }
        for (root, key) in [(r"HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Environment",
                             "Path"),
                            (r"HKCU\Environment", "Path")] {
            let Ok(out) = Command::new("reg").args(["query", root, "/v", key])
                .stdin(std::process::Stdio::null()).output() else { continue };
            let text = String::from_utf8_lossy(&out.stdout).to_string();
            let Some(line) = text.lines().find(|l| l.trim_start().starts_with(key)) else { continue };
            let Some(value) = ["REG_SZ", "REG_EXPAND_SZ"].iter()
                .find_map(|kind| line.split_once(kind)) else { continue };
            places.extend(value.1.trim().split(';')
                .filter(|p| !p.trim().is_empty())
                .map(|p| PathBuf::from(expand(p.trim()))));
        }
        places
    }

    /// `%SystemRoot%\system32` and friends, as the registry stores them.
    ///
    /// **`REG_EXPAND_SZ` means the value is a template**, and a path left unexpanded matches
    /// nothing — which would make this test complain about a DLL that is right there. The values
    /// come from this process's environment, which is safe: it is the *variable* being read, not
    /// the search list.
    fn expand(raw: &str) -> String {
        let mut out = String::with_capacity(raw.len());
        let mut rest = raw;
        while let Some(at) = rest.find('%') {
            out.push_str(&rest[..at]);
            let after = &rest[at + 1..];
            match after.find('%') {
                Some(end) => {
                    let name = &after[..end];
                    match std::env::var(name) {
                        Ok(value) => out.push_str(&value),
                        //: Unknown: put it back as it was rather than dropping it, so a path that
                        //: happens to contain a per cent sign is not silently mangled.
                        Err(_) => { out.push('%'); out.push_str(name); out.push('%'); }
                    }
                    rest = &after[end + 1..];
                }
                None => { out.push_str(&rest[at..]); rest = ""; }
            }
        }
        out.push_str(rest);
        out
    }

    /// Is this a library the loader will find on its own?
    ///
    /// **`api-ms-win-*` and `ext-ms-*` are API sets**: contract names resolved through a schema
    /// inside the kernel, never files on disk. Every Rust binary on Windows imports several, and
    /// an earlier version of this test failed a perfectly good binary over them.
    fn findable(name: &str, places: &[PathBuf]) -> bool {
        if name.starts_with("api-ms-") || name.starts_with("ext-ms-") {
            return true;
        }
        places.iter().any(|at| at.join(name).is_file())
    }

    fn imports(binary: &Path) -> Option<Vec<String>> {
        let out = Command::new("objdump").arg("-p").arg(binary)
            .stdin(std::process::Stdio::null()).output().ok()?;
        if !out.status.success() {
            return None;
        }
        let mut names: Vec<String> = String::from_utf8_lossy(&out.stdout).lines()
            .filter_map(|l| l.trim().strip_prefix("DLL Name:"))
            .map(|n| n.trim().to_lowercase())
            .collect();
        names.sort();
        names.dedup();
        Some(names)
    }

    pub fn check(exe: &str) {
        let exe = PathBuf::from(exe);
        assert!(exe.is_file(), "{} was not built", exe.display());
        let Some(beside) = exe.parent() else { panic!("no folder") };

        let Some(wanted) = imports(&exe) else {
            //: **Stops rather than passes.** A check that quietly does nothing when its tool is
            //: missing is worse than no check, because it reads green.
            panic!("objdump could not read {} — it is needed to build the recogniser too, so \
                    install it rather than skipping this", exe.display());
        };
        assert!(wanted.len() > 5, "only {} imports read: {wanted:?}", wanted.len());

        //: The application's own folder, which the loader checks first, then everywhere else.
        let places = where_the_loader_looks();
        let orphans: Vec<&String> = wanted.iter()
            .filter(|name| !beside.join(name).is_file() && !findable(name, &places))
            .collect();
        assert!(orphans.is_empty(),
                "{} imports {orphans:?}, which Windows will not find when it is started as a \
                 file — it would run under `cargo run` and fail on a double-click with \
                 0xC0000135 and no message.\n\nOn Windows build with the MSVC toolchain, where \
                 `webview2-com-sys` links the loader statically and there is nothing to ship: \
                 `rustup default stable-x86_64-pc-windows-msvc`, or `cargo +stable-msvc build`. \
                 A GNU build needs WebView2Loader.dll copied beside the executable from \
                 target/*/build/webview2-com-sys-*/out/x64/.",
                exe.display());
    }
}

/// The windowed application starts as a file.
#[test]
#[cfg(windows)]
fn the_application_can_be_started_by_double_clicking_it() {
    windows_only::check(env!("CARGO_BIN_EXE_argmind-app"));
}

/// And so does the console one, which is the fallback the other's error message names.
#[test]
#[cfg(windows)]
fn so_can_the_console_one() {
    windows_only::check(env!("CARGO_BIN_EXE_serve"));
}
