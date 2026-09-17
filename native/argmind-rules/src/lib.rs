//! Reading the repository, so a rule can be checked against it instead of remembered.
//!
//! **Three rules moved here from the Python suite**, because that suite is going and the rules are
//! not: only dependencies with a licence somebody approved, a core that names no platform and asks
//! no model for a coordinate, and exactly one place in the application that acts. Each was held by
//! a test walking Python source; each is now held by a test walking Rust source, and the tests are
//! in `tests/` beside this — one file per rule, named after the rule.
//!
//! **This reads text rather than parsing a language, and that is weaker.** The Python used `ast`
//! for two of the three and `tomllib` for the last. Rust has neither in its standard library, and
//! taking a dependency in order to police dependencies is a circle worth avoiding — so what is here
//! understands the two file shapes it actually reads and gives up loudly on anything else. Where
//! that costs precision it is said at the point where it costs it.

use std::path::{Path, PathBuf};

/// The repository root, from this crate's own location.
pub fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).parent().and_then(Path::parent)
        .expect("this crate lives two levels below the root").to_path_buf()
}

/// Lines that are not blank and not a whole-line comment, numbered from one.
///
/// **Only code lines**, because a rule about what the source *does* should not be broken by a
/// comment that discusses it — the platform rule exists partly so a contract can compare Windows
/// with Android in prose without that counting as depending on either.
pub fn code_lines(source: &str) -> Vec<(usize, &str)> {
    source.lines().enumerate()
        .map(|(i, l)| (i + 1, l))
        .filter(|(_, l)| {
            let t = l.trim_start();
            !t.is_empty() && !t.starts_with("//") && !t.starts_with("*")
        })
        .collect()
}

/// Every `.rs` file directly inside a directory, sorted.
pub fn rust_files(dir: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(dir) else { return Vec::new() };
    let mut out: Vec<PathBuf> = entries.filter_map(Result::ok).map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "rs")).collect();
    out.sort();
    out
}

/// One entry of the licence record: which table it was under, its name, and the licence it declares.
#[derive(Debug, Clone, PartialEq)]
pub struct Recorded {
    pub table: String,
    pub name: String,
    pub licence: String,
}

/// The `allowed = [...]` list at the top of the record.
pub fn allowed(record: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut inside = false;
    for line in record.lines() {
        let t = line.trim();
        if t.starts_with("allowed") && t.contains('[') {
            inside = true;
            continue;
        }
        if !inside {
            continue;
        }
        if t.starts_with(']') {
            break;
        }
        if let Some(value) = quoted(t) {
            out.push(value);
        }
    }
    out
}

/// Every `name = { licence = "…" }` in the record, with the table it sits under.
///
/// **A line that looks like an entry and has no licence is an error, not a skip.** A record whose
/// malformed lines vanish silently is a record that can quietly stop covering something.
pub fn recorded(record: &str) -> Result<Vec<Recorded>, String> {
    let mut out = Vec::new();
    let mut table = String::new();
    for (n, line) in record.lines().enumerate() {
        let t = line.trim();
        if let Some(rest) = t.strip_prefix('[') {
            table = rest.trim_end_matches(']').to_string();
            continue;
        }
        let Some((left, right)) = t.split_once('=') else { continue };
        if !right.trim_start().starts_with('{') {
            continue;
        }
        let name = quoted(left.trim()).unwrap_or_else(|| left.trim().to_string());
        let licence = right.split_once("licence")
            .and_then(|(_, after)| after.split_once('='))
            .and_then(|(_, value)| quoted(value.trim()))
            .ok_or_else(|| format!("line {}: {t:?} declares no licence", n + 1))?;
        out.push(Recorded { table: table.clone(), name, licence });
    }
    Ok(out)
}

/// The first double-quoted run in a fragment, if there is one.
fn quoted(fragment: &str) -> Option<String> {
    let after = fragment.split_once('"')?.1;
    let (inside, _) = after.split_once('"')?;
    Some(inside.to_string())
}

/// The dependencies a `Cargo.toml` declares that come from outside this repository.
///
/// **Path dependencies are this project's own crates** and are not third-party anything, so they
/// are not recorded and must not be demanded. Everything else is a name somebody has to have
/// looked at.
pub fn outside_dependencies(manifest: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut inside = false;
    for line in manifest.lines() {
        let t = line.trim();
        if t.starts_with('[') {
            //: **Including the target-conditional tables**, which is not fussiness: adding
            //: `dbus` under `[target.'cfg(target_os = "linux")'.dependencies]` on 2026-09-17 was
            //: invisible to this rule, and the rule is the only thing that makes a new dependency
            //: cost anything. It was noticed because the *other* half of the check complained that
            //: the record named a crate nothing declared — the record was right and the reader was
            //: wrong, which is the pleasanter of the two ways round.
            let table = t.trim_start_matches('[').trim_end_matches(']');
            inside = matches!(table, "dependencies" | "dev-dependencies" | "build-dependencies")
                || (table.starts_with("target.")
                    && (table.ends_with(".dependencies")
                        || table.ends_with(".dev-dependencies")
                        || table.ends_with(".build-dependencies")));
            continue;
        }
        if !inside || t.starts_with('#') {
            continue;
        }
        let Some((left, right)) = t.split_once('=') else { continue };
        let name = left.trim();
        //: A continuation line of a multi-line table — `features = [` and the entries under it —
        //: is not a dependency. Only a line whose left side is a bare name at the start of the
        //: line counts, and `line` rather than `t` is what says so.
        if line.starts_with(char::is_whitespace) || name.is_empty() || name.contains(' ') {
            continue;
        }
        if right.contains("path") {
            continue;
        }
        out.push(name.to_string());
    }
    out.sort();
    out.dedup();
    out
}
