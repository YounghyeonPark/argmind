//! **The notices that ship beside the two executables**, written from what the crates say about
//! themselves, into `THIRD-PARTY-NOTICES.txt` at the root.
//!
//! ```text
//!   cargo run --manifest-path native/Cargo.toml -p argmind-rules --example notices
//!   cargo run --manifest-path native/Cargo.toml -p argmind-rules --example notices -- --release <dir>
//! ```
//!
//! The second also puts beside the file what a release has to carry with `argmind-app.exe` and
//! `argmind.exe`: `LICENSE`, `NOTICE`, the file itself, and the Rust standard library's own notice
//! (`COPYRIGHT-library.html` from the toolchain that is running, 1.5 MB) as `RUST-STD-COPYRIGHT.html`.
//!
//! **Why it exists.** Until 2026-11-17 nothing here was distributed as a program, only as source,
//! and source carries each crate's licence in `Cargo.lock`'s registry. An executable carries none of
//! that, and MIT, the licence most of this graph is under, asks for its copyright notice in every
//! copy. A file written by hand would be out of date the first time a crate moved a patch version,
//! so the file is generated, and `tests/third_party_notices.rs` fails when it and the graph
//! disagree.
//!
//! **What it does not do: decide that a licence is fine.** It copies what each package ships. That
//! the declared licence is true is the same unknown `tests/licences.rs` writes down.

use argmind_rules::{linked, root, Linked, RELEASE_TARGET, WEBVIEW2_SDK, WEBVIEW2_SYS};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const RULE: &str = "================================================================================";

/// Where the terms of the Apache License start and end. Everything between is compared with this
/// repository's own `LICENSE`, so a hundred identical copies are printed once.
const TERMS_START: &str = "TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION";
const TERMS_END: &str = "END OF TERMS AND CONDITIONS";

fn main() {
    let root = root();
    let crates = linked().unwrap_or_else(|e| die(&e));
    let ours = read(&root.join("LICENSE"));
    let apache = Apache::of(&ours).unwrap_or_else(|| die("LICENSE is not the Apache License 2.0"));
    let registries = registries();
    let rustc = run("rustc", &["--version"]);

    let mut body = String::new();
    let mut seen: BTreeMap<String, String> = BTreeMap::new();
    let mut needs_standard: BTreeSet<String> = BTreeSet::new();
    let mut without_files = Vec::new();
    for c in &crates {
        body.push_str(RULE);
        body.push('\n');
        body.push_str(&format!("crate: {} {}\n", c.name, c.version));
        body.push_str(&format!("licence (declared): {}\n", c.licence));
        if !c.repository.is_empty() {
            body.push_str(&format!("repository: {}\n", c.repository));
        }
        if c.build_time {
            body.push_str("build time only: a procedural macro, or something only one uses. None of \
                           its code is linked; it is listed because the code it writes is.\n");
        }
        let dir = find(&registries, c);
        let files = licence_files(&dir);
        let who = format!("{} {}", c.name, c.version);
        if files.is_empty() {
            without_files.push(who.clone());
            if c.name.starts_with("webview2-com") && c.repository == "https://github.com/wravery/webview2-rs" {
                body.push_str(&format!("\nThe package ships no licence file. The repository's own LICENSE, \
                                        which is the same text at the commit this package names ({}), \
                                        verbatim:\n", vcs_commit(&dir)));
                emit(&mut body, &mut seen, &apache, &who, "LICENSE (webview2-rs)",
                     &read(&root.join("native/argmind-rules/notices/webview2-rs-LICENSE.txt")));
            } else {
                body.push_str(&format!("\nThe package ships no licence file. The standard text of {} \
                                        is at the end of this file.\n", c.licence));
                needs_standard.insert(c.licence.clone());
            }
        }
        for (name, text) in &files {
            emit(&mut body, &mut seen, &apache, &who, name, text);
        }
        if c.name == "webview2-com-sys" {
            if c.version != WEBVIEW2_SYS {
                die(&format!("webview2-com-sys is {} and the loader licence was read for {WEBVIEW2_SYS}: \
                              read the WebView2 SDK licence of the loader it now carries",
                             c.version));
            }
            body.push_str(&format!(
                "\nThis crate carries Microsoft's prebuilt WebView2 loader (x64/WebView2LoaderStatic.lib, \
                 WebView2 SDK {WEBVIEW2_SDK}), which is linked into argmind-app.exe. The SDK's licence and \
                 notice, verbatim, from the Microsoft.Web.WebView2 {WEBVIEW2_SDK} package:\n"));
            emit(&mut body, &mut seen, &apache, &who, "LICENSE.txt (WebView2 SDK)",
                 &read(&root.join("native/argmind-rules/notices/webview2-sdk-LICENSE.txt")));
            emit(&mut body, &mut seen, &apache, &who, "NOTICE.txt (WebView2 SDK)",
                 &read(&root.join("native/argmind-rules/notices/webview2-sdk-NOTICE.txt")));
        }
        body.push('\n');
    }
    if !needs_standard.is_empty() {
        body.push_str(RULE);
        body.push_str("\nSTANDARD TEXTS, for the crates above that ship no licence file\n\n");
        let mut wanted = BTreeSet::new();
        for licence in &needs_standard {
            let known: Vec<String> = alternatives(licence).into_iter()
                .filter(|a| a == "MIT" || a == "Apache-2.0").collect();
            if known.is_empty() {
                die(&format!("no standard text kept for {licence}; add one rather than ship nothing"));
            }
            wanted.extend(known);
        }
        if wanted.contains("MIT") {
            body.push_str(MIT);
        }
        if wanted.contains("Apache-2.0") {
            body.push_str("Apache-2.0: the text is LICENSE, beside this file.\n\n");
        }
    }

    let build_time = crates.iter().filter(|c| c.build_time).count();
    let mut out = String::new();
    out.push_str(&header(crates.len(), build_time, &rustc));
    out.push('\n');
    out.push_str(&body);
    let path = root.join("THIRD-PARTY-NOTICES.txt");
    fs::write(&path, &out).unwrap_or_else(|e| die(&format!("{}: {e}", path.display())));

    let licences: BTreeSet<&str> = crates.iter().map(|c| c.licence.as_str()).collect();
    println!("  {} crates ({} build time only), {} bytes, {} declared licences",
             crates.len(), build_time, out.len(), licences.len());
    println!("  no licence file in the package: {}", if without_files.is_empty() {
        "none".to_string() } else { without_files.join(", ") });
    println!("  written: {}", path.display());

    let args: Vec<String> = std::env::args().collect();
    if let Some(i) = args.iter().position(|a| a == "--release") {
        let dir = PathBuf::from(args.get(i + 1).unwrap_or_else(|| die("--release takes a folder")));
        fs::create_dir_all(&dir).unwrap_or_else(|e| die(&format!("{}: {e}", dir.display())));
        let sysroot = run("rustc", &["--print", "sysroot"]);
        let std_notice = Path::new(sysroot.trim()).join("share/doc/rust/COPYRIGHT-library.html");
        for (from, to) in [(root.join("LICENSE"), "LICENSE.txt"), (root.join("NOTICE"), "NOTICE.txt"),
                           (path.clone(), "THIRD-PARTY-NOTICES.txt"),
                           (std_notice, "RUST-STD-COPYRIGHT.html")] {
            fs::copy(&from, dir.join(to)).unwrap_or_else(|e| die(&format!("{}: {e}", from.display())));
            println!("  copied: {}", dir.join(to).display());
        }
    }
}

fn header(count: usize, build_time: usize, rustc: &str) -> String {
    format!("\
THIRD-PARTY NOTICES

argmind-app.exe and argmind.exe are Apache-2.0 (LICENSE and NOTICE, beside this file). They are
built from the third-party code below, and this file carries the licence and copyright notices
that code asks to be passed on with it.

Generated, never written by hand:
    cargo run --manifest-path native/Cargo.toml -p argmind-rules --example notices
from `cargo tree -p argmind-serve -e normal --target {RELEASE_TARGET}`, offline and locked:
{count} crates, {build_time} of them build time only. Each section holds every LICENSE*, LICENCE*,
COPYING*, COPYRIGHT*, NOTICE* and UNLICENSE* file the crate's package ships, verbatim, with two
exceptions: a text already given once is named rather than repeated, and an Apache License 2.0
whose terms are word for word those of LICENSE is not printed again (anything else in that file,
a copyright line included, is). native/argmind-rules/tests/third_party_notices.rs fails when this
file and the graph disagree.

Not from crates.io:
  * The Rust standard library, in every Rust executable: Apache-2.0 OR MIT, copyrights retained by
    the Rust Project Developers (https://thanks.rust-lang.org), with the exceptions it lists
    itself. Its full notice is share/doc/rust/COPYRIGHT-library.html in the toolchain that built
    the executables, and a release carries it as RUST-STD-COPYRIGHT.html. This file was generated
    with {rustc}.
  * Microsoft's WebView2 loader (WebView2LoaderStatic.lib, WebView2 SDK {WEBVIEW2_SDK}), linked into
    argmind-app.exe through webview2-com-sys. Its licence and notice are in that crate's section.

Not redistributed, and so not here: the WebView2 runtime, which is found on the machine; Ollama
and its models, the speech recogniser and the voice, which argmind offers to download or build and
which carry their own licences.
", rustc = rustc.trim())
}

/// Write one licence file under its crate, or say which earlier one it is.
fn emit(out: &mut String, seen: &mut BTreeMap<String, String>, apache: &Apache, who: &str, file: &str,
        text: &str) {
    let text = text.replace("\r\n", "\n");
    out.push_str(&format!("\n--- {file} ---\n"));
    if let Some(a) = Apache::of(&text) {
        if a.terms == apache.terms {
            if a.head == apache.head && a.tail == apache.tail {
                out.push_str("The Apache License, Version 2.0: the same text as LICENSE, word for word.\n");
            } else {
                out.push_str("The Apache License, Version 2.0, its terms word for word those of LICENSE. \
                              The rest of this file, verbatim:\n");
                if a.head != apache.head {
                    out.push_str(a.raw_head.trim_end());
                    out.push('\n');
                }
                out.push_str("[ the terms and conditions, as in LICENSE ]\n");
                if a.tail != apache.tail {
                    out.push_str(a.raw_tail.trim());
                    out.push('\n');
                }
            }
            return;
        }
    }
    let key = normalise(&text);
    if let Some(first) = seen.get(&key) {
        out.push_str(&format!("The same text as {first}, word for word.\n"));
        return;
    }
    seen.insert(key, format!("{who}, {file}"));
    out.push_str(text.trim_end());
    out.push('\n');
}

/// An Apache License 2.0 split at its terms, each part compared with whitespace collapsed.
struct Apache {
    head: String,
    terms: String,
    tail: String,
    raw_head: String,
    raw_tail: String,
}

impl Apache {
    fn of(text: &str) -> Option<Apache> {
        let start = text.find(TERMS_START)?;
        let end = text[start..].find(TERMS_END)? + start + TERMS_END.len();
        Some(Apache {
            head: normalise(&text[..start]),
            terms: normalise(&text[start..end]),
            tail: normalise(&text[end..]),
            raw_head: text[..start].to_string(),
            raw_tail: text[end..].to_string(),
        })
    }
}

fn normalise(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn alternatives(licence: &str) -> Vec<String> {
    licence.split(['/', '(', ')']).flat_map(|p| p.split(" OR ")).flat_map(|p| p.split(" AND "))
        .map(|p| p.trim().to_string()).filter(|p| !p.is_empty()).collect()
}

/// Every `registry/src/<index>/` under the cargo home.
fn registries() -> Vec<PathBuf> {
    let home = std::env::var_os("CARGO_HOME").map(PathBuf::from).unwrap_or_else(|| {
        let user = std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME"))
            .unwrap_or_else(|| die("no CARGO_HOME, USERPROFILE or HOME"));
        PathBuf::from(user).join(".cargo")
    });
    let src = home.join("registry").join("src");
    let mut out: Vec<PathBuf> = fs::read_dir(&src)
        .unwrap_or_else(|e| die(&format!("{}: {e}", src.display())))
        .flatten().map(|e| e.path()).filter(|p| p.is_dir()).collect();
    out.sort();
    out
}

fn find(registries: &[PathBuf], c: &Linked) -> PathBuf {
    registries.iter().map(|r| r.join(format!("{}-{}", c.name, c.version))).find(|p| p.is_dir())
        .unwrap_or_else(|| die(&format!("{} {} is not in the cargo registry here", c.name, c.version)))
}

/// The licence files a package ships at its top, and inside a top-level `license*` folder.
fn licence_files(dir: &Path) -> Vec<(String, String)> {
    const STARTS: [&str; 5] = ["licen", "copying", "notice", "copyright", "unlicen"];
    let named = |p: &Path| {
        let n = p.file_name().map(|n| n.to_string_lossy().to_lowercase()).unwrap_or_default();
        STARTS.iter().any(|s| n.starts_with(s))
    };
    let mut out = Vec::new();
    let mut entries: Vec<PathBuf> = fs::read_dir(dir).map(|r| r.flatten().map(|e| e.path()).collect())
        .unwrap_or_default();
    entries.sort();
    for p in entries {
        if !named(&p) {
            continue;
        }
        if p.is_file() {
            out.push((file_name(&p), read(&p)));
        } else if p.is_dir() {
            let mut inner: Vec<PathBuf> = fs::read_dir(&p).map(|r| r.flatten().map(|e| e.path()).collect())
                .unwrap_or_default();
            inner.sort();
            for q in inner.into_iter().filter(|q| q.is_file()) {
                out.push((format!("{}/{}", file_name(&p), file_name(&q)), read(&q)));
            }
        }
    }
    out
}

/// The commit a package was published from, off its `.cargo_vcs_info.json`, shortened.
fn vcs_commit(dir: &Path) -> String {
    let info = fs::read_to_string(dir.join(".cargo_vcs_info.json")).unwrap_or_default();
    info.split("\"sha1\"").nth(1).and_then(|r| r.split('"').nth(1)).map(|s| s.chars().take(7).collect())
        .unwrap_or_else(|| die(&format!("{} names no commit", dir.display())))
}

fn file_name(p: &Path) -> String {
    p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()
}

fn read(p: &Path) -> String {
    let bytes = fs::read(p).unwrap_or_else(|e| die(&format!("{}: {e}", p.display())));
    String::from_utf8_lossy(&bytes).replace("\r\n", "\n")
}

fn run(program: &str, args: &[&str]) -> String {
    let got = Command::new(program).args(args).output()
        .unwrap_or_else(|e| die(&format!("{program} did not start: {e}")));
    if !got.status.success() {
        die(&format!("{program} {args:?} failed"));
    }
    String::from_utf8_lossy(&got.stdout).into_owned()
}

fn die(why: &str) -> ! {
    eprintln!("notices: {why}");
    std::process::exit(1);
}

/// The MIT licence as SPDX gives it. **It has no copyright line of its own**: a crate that ships no
/// file gives none, and inventing one would be worse than the gap.
const MIT: &str = "\
MIT License (the standard text; the crate names no copyright holder in a file)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the \"Software\"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

";
