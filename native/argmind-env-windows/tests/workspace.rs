//! **The workbench keeps to its folder**, and a write or an edit can be undone.
//!
//! Files only — nothing here starts a command, so the suite needs no sandbox and no PowerShell.

use argmind_core::task::Tools;
use argmind_env_windows::workspace::{Workspace, BACKUPS};
use serde_json::json;

fn bench() -> (Workspace, std::path::PathBuf) {
    let dir = std::env::temp_dir().join(format!("argmind-ws-{}-{}", std::process::id(),
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
    (Workspace::new(dir.clone()), dir)
}

#[test]
fn paths_outside_the_folder_are_refused() {
    let (mut w, dir) = bench();
    for bad in ["../x.txt", "C:\\Windows\\win.ini", "/etc/passwd", "a/../../b", "\\\\server\\share\\x"] {
        assert!(w.call("write_file", &json!({"path": bad, "content": "x"})).is_err(), "{bad}");
        assert!(w.call("read_file", &json!({"path": bad})).is_err(), "{bad}");
    }
    assert!(w.call("write_file", &json!({"path": "sub/ok.txt", "content": "안녕"})).is_ok());
    assert_eq!(std::fs::read_to_string(dir.join("sub").join("ok.txt")).unwrap(), "안녕");
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn a_replaced_file_is_backed_up_and_an_edit_must_be_unambiguous() {
    let (mut w, dir) = bench();
    w.call("write_file", &json!({"path": "r.md", "content": "서론\n결론: 미정\n결론 끝"})).unwrap();
    //: Twice: refused, nothing changed.
    assert!(w.call("edit_file", &json!({"path": "r.md", "find": "결론", "replace": "X"})).is_err());
    assert!(w.call("edit_file", &json!({"path": "r.md", "find": "없는 말", "replace": "X"})).is_err());
    w.call("edit_file", &json!({"path": "r.md", "find": "결론: 미정", "replace": "결론: 진행"})).unwrap();
    assert_eq!(std::fs::read_to_string(dir.join("r.md")).unwrap(), "서론\n결론: 진행\n결론 끝");
    let backups: Vec<_> = std::fs::read_dir(dir.join(BACKUPS)).unwrap().flatten().collect();
    assert_eq!(backups.len(), 1, "the edit kept the original");
    assert_eq!(std::fs::read_to_string(backups[0].path()).unwrap(), "서론\n결론: 미정\n결론 끝");
    let listed = w.call("list_files", &json!({"path": "."})).unwrap();
    assert!(listed.contains("r.md") && !listed.contains(BACKUPS), "{listed}");
    let _ = std::fs::remove_dir_all(dir);
}
