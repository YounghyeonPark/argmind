//! **What a task made, found by the program** — since 2026-11-07, the half of D that was left:
//! *결과물은 작업 폴더에 파일로 남기고 브라우저 칸에 열어 주며, 말로는 요약만.*
//!
//! The work folder is read before a task and again after it, and a file is *made by the task* when
//! it is new or its size or time changed. Not from the steps: a `.docx` is written by a script the
//! task `run`s, and no step names it as a write. The backups the workbench keeps are not results.
//! What is found is shown in the pane by `app.rs`, read through the workbench's own `read_file`, so
//! a Word file is shown as the text the task itself was shown.

use std::collections::BTreeMap;
use std::path::Path;
use std::time::SystemTime;

/// The folder at a moment: path relative to it, `/`-separated, with its time and size.
pub type Snapshot = BTreeMap<String, (Option<SystemTime>, u64)>;

/// Files shown at most — a task that made forty files is summarised in words, not in a page.
pub const MOST_SHOWN: usize = 3;
/// Characters of each file shown — enough to read a plan or a script, not a log.
pub const MOST_CHARS: usize = 8_000;

/// What can be read as a page: text the workbench reads as it is, and the two documents it reads
/// in the sandbox.
const SHOWN: [&str; 16] = ["md", "txt", "csv", "json", "py", "js", "ts", "html", "rs", "sh",
                           "toml", "yaml", "yml", "log", "docx", "xlsx"];

pub fn snapshot(root: &Path) -> Snapshot {
    let mut seen = Snapshot::new();
    walk(root, "", &mut seen, 0);
    seen
}

fn walk(dir: &Path, prefix: &str, seen: &mut Snapshot, depth: usize) {
    //: Deep enough for a project a task made, shallow enough that a `node_modules` is not a tour.
    if depth > 4 || seen.len() > 5_000 {
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        //: The workbench's backups, and anything hidden, are not something a person asked for.
        if name.starts_with('.') {
            continue;
        }
        let rel = if prefix.is_empty() { name.clone() } else { format!("{prefix}/{name}") };
        let Ok(meta) = entry.metadata() else { continue };
        if meta.is_dir() {
            walk(&entry.path(), &rel, seen, depth + 1);
        } else {
            seen.insert(rel, (meta.modified().ok(), meta.len()));
        }
    }
}

/// **Made or changed since `before`**, readable as a page, newest first, at most [`MOST_SHOWN`].
pub fn made(before: &Snapshot, after: &Snapshot) -> Vec<String> {
    changed(before, after).into_iter()
        .filter(|rel| rel.rsplit_once('.').is_some_and(|(_, ext)| SHOWN.contains(&ext.to_ascii_lowercase().as_str())))
        .take(MOST_SHOWN).collect()
}

/// **Every file made or changed since `before`**, whatever it is, newest first — what a record of
/// the work names, where [`made`] is what a page can show. An STL argmind made by itself was missing
/// from its record because a page cannot show one (2026-11-09).
pub fn changed(before: &Snapshot, after: &Snapshot) -> Vec<String> {
    let mut changed: Vec<(&String, Option<SystemTime>)> = after.iter()
        .filter(|(rel, now)| before.get(*rel) != Some(now))
        .map(|(rel, (at, _))| (rel, *at))
        .collect();
    changed.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
    changed.into_iter().map(|(rel, _)| rel.clone()).collect()
}

/// At most [`MOST_CHARS`] of a file's text, and said so when cut.
pub fn trimmed(text: &str) -> String {
    if text.chars().count() <= MOST_CHARS {
        return text.to_string();
    }
    let kept: String = text.chars().take(MOST_CHARS).collect();
    format!("{kept}\n\n… (여기까지 — 나머지는 작업 폴더의 파일에)")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn what_was_made_or_changed_is_found_and_backups_and_untouched_files_are_not() {
        let root = std::env::temp_dir().join(format!("argmind-deliver-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join(".argmind-backup")).unwrap();
        std::fs::create_dir_all(root.join("sub")).unwrap();
        std::fs::write(root.join("old.md"), "그대로").unwrap();
        std::fs::write(root.join("edited.md"), "전").unwrap();
        let before = snapshot(&root);
        std::fs::write(root.join("edited.md"), "후, 더 길게").unwrap();
        std::fs::write(root.join("sub/new.docx"), "x").unwrap();
        std::fs::write(root.join("picture.png"), "x").unwrap();
        std::fs::write(root.join(".argmind-backup/edited.md"), "전").unwrap();
        let mut got = made(&before, &snapshot(&root));
        got.sort();
        assert_eq!(got, ["edited.md", "sub/new.docx"]);
        //: Every changed file is named, a picture included; backups and untouched files still not.
        let mut all = changed(&before, &snapshot(&root));
        all.sort();
        assert_eq!(all, ["edited.md", "picture.png", "sub/new.docx"]);
        //: And nothing changed is nothing to show.
        assert!(made(&snapshot(&root), &snapshot(&root)).is_empty());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_long_file_is_cut_and_says_so() {
        assert_eq!(trimmed("짧다"), "짧다");
        let long = "가".repeat(MOST_CHARS + 5);
        let t = trimmed(&long);
        assert!(t.starts_with(&"가".repeat(MOST_CHARS)) && t.ends_with("작업 폴더의 파일에)"));
    }
}
