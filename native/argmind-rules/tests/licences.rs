//! Only things with no open-source licence problem, held by a test rather than remembered.
//!
//! The owner's rule, 2026-09-14: **오픈소스라이센스에 문제 없는것만 사용할것.** A rule nothing
//! enforces is a rule nobody keeps, and this project has the receipt — the `NOTICE` it shipped
//! until that day attributed OSQP, CVXPY and SciPy for a library removed on 2026-09-09, and named
//! none of the things actually imported. Nobody noticed for five days, because nothing looked.
//!
//! So: every dependency declared in a `Cargo.toml` under `native/` must appear in
//! `docs/licences.toml` with a licence from its `allowed` list. **Adding a dependency fails the
//! suite until a line is added there**, which is the whole mechanism — the cost of a new dependency
//! is paid by the person taking it on, at the moment they take it on.
//!
//! **This is the Rust successor to `tests/test_licences.py`**, and it carries the same three
//! disclaimers, said plainly rather than implied:
//!
//! * **Transitive dependencies are not checked.** `windows-rs` pulls a dozen crates. This checks
//!   what this project chose, not what those chose; `cargo tree` is the tool for that and it does
//!   not run offline.
//! * **That a declared licence is true is not checked.** A crate that says MIT and contains
//!   somebody else's GPL code is not something a test can see.
//! * **Model weights and services are not dependencies.** They are in `[external]` and nothing
//!   here redistributes them.
//!
//! Needs no network and no model: it reads text files.

use argmind_rules::{allowed, outside_dependencies, recorded, root};

fn record() -> String {
    std::fs::read_to_string(root().join("docs").join("licences.toml"))
        .expect("docs/licences.toml is the record and it has to be there")
}

/// Every crate this project declares is written down, with an approved licence.
#[test]
fn every_dependency_is_recorded_and_its_licence_is_allowed() {
    let record = record();
    let ok = allowed(&record);
    assert!(ok.len() >= 5, "the allowed list looks unread: {ok:?}");
    let rows = recorded(&record).expect("the record parses");

    let mut missing: Vec<String> = Vec::new();
    let mut wrong: Vec<String> = Vec::new();
    for manifest in std::fs::read_dir(root().join("native")).expect("native/ is there")
        .filter_map(Result::ok)
        .map(|e| e.path().join("Cargo.toml"))
        .filter(|p| p.is_file())
    {
        let text = std::fs::read_to_string(&manifest).expect("a manifest reads");
        let crate_name = manifest.parent().and_then(|p| p.file_name())
            .map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
        for want in outside_dependencies(&text) {
            match rows.iter().find(|r| r.name == want) {
                None => missing.push(format!("{crate_name} depends on {want:?}")),
                Some(row) if !ok.contains(&row.licence) =>
                    wrong.push(format!("{want:?} declares {:?}", row.licence)),
                Some(_) => {}
            }
        }
    }
    assert!(missing.is_empty(),
            "not in docs/licences.toml — add a line before taking the dependency on: {missing:?}");
    assert!(wrong.is_empty(), "recorded with a licence that is not in `allowed`: {wrong:?}");
}

/// **The record does not name things this project no longer takes.**
///
/// The other direction, and the one the old `NOTICE` failed: a record that keeps listing what was
/// removed reads as thorough while being wrong, and nothing about it looks stale.
#[test]
fn nothing_in_the_rust_table_has_stopped_being_a_dependency() {
    let record = record();
    let rows = recorded(&record).expect("the record parses");
    let mut declared: Vec<String> = Vec::new();
    for manifest in std::fs::read_dir(root().join("native")).expect("native/ is there")
        .filter_map(Result::ok)
        .map(|e| e.path().join("Cargo.toml"))
        .filter(|p| p.is_file())
    {
        let text = std::fs::read_to_string(&manifest).expect("a manifest reads");
        declared.extend(outside_dependencies(&text));
    }
    let stale: Vec<&str> = rows.iter()
        .filter(|r| r.table == "rust" && !declared.contains(&r.name))
        .map(|r| r.name.as_str()).collect();
    assert!(stale.is_empty(),
            "docs/licences.toml still records {stale:?} under [rust]; nothing declares them");
}

/// **Everything argmind fetches or runs is written down somewhere**, even when it is not linked.
///
/// The recogniser is compiled from source and executed on the person's machine. That is not a
/// `Cargo.toml` line and no dependency check would ever see it, which is exactly why it has to be
/// named — and why `[external]` exists.
#[test]
fn what_argmind_downloads_and_runs_is_in_the_record() {
    let record = record();
    let rows = recorded(&record).expect("the record parses");
    let external: Vec<&str> = rows.iter().filter(|r| r.table == "external")
        .map(|r| r.name.as_str()).collect();
    for wanted in ["ollama", "whisper.cpp", "qwentts.cpp", "Qwen3-TTS GGUF"] {
        assert!(external.contains(&wanted),
                "argmind fetches or runs {wanted:?} and the record does not name it: {external:?}");
    }
    //: **And the note beside it has to still be true.** It read *"not yet a dependency"* while
    //: `Fix::BuildRecogniser` was compiling and running it, which is the record drifting in the
    //: direction that flatters.
    assert!(!record.contains("Not yet a dependency"),
            "docs/licences.toml still says whisper.cpp is not a dependency; argmind now builds \
             and runs it");
}

/// The reader is small and its limits are known, so they are asserted rather than assumed.
#[test]
fn the_reader_understands_the_shapes_it_is_given_and_says_so_when_it_cannot() {
    let sample = "allowed = [\n  \"MIT\",\n  \"Apache-2.0\",\n]\n\n\
                  [rust]\nwindows = { licence = \"MIT OR Apache-2.0\", why = \"x\" }\n\
                  [external]\n\"whisper.cpp\" = { licence = \"MIT\", why = \"y\" }\n";
    assert_eq!(allowed(sample), ["MIT", "Apache-2.0"]);
    let rows = recorded(sample).expect("parses");
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].table, "rust");
    assert_eq!(rows[1].name, "whisper.cpp", "a quoted key keeps its dots");
    assert_eq!(rows[1].licence, "MIT");

    //: **An entry with no licence is refused rather than skipped.** Skipping it would let a line
    //: somebody half-wrote count as a dependency that had been looked at.
    assert!(recorded("[rust]\nthing = { why = \"no licence here\" }\n").is_err());
}

/// A manifest's own crates are not third-party anything.
#[test]
fn path_dependencies_are_not_demanded_of_the_record() {
    let manifest = "[dependencies]\n\
                    argmind-core = { path = \"../argmind-core\" }\n\
                    serde_json = \"1\"\n\
                    windows = { version = \"0.58\", features = [\n    \"Win32_Foundation\",\n] }\n";
    let found = outside_dependencies(manifest);
    assert_eq!(found, ["serde_json", "windows"],
               "a path dependency or a feature line was counted as a third-party crate");
}
