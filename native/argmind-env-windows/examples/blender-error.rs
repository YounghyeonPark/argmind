//! What a task sees when its Blender script fails — `cargo run --release -p argmind-env-windows --example blender-error -- <dir>`.
#[cfg(windows)]
fn main() {
    use argmind_core::task::Tools;
    let dir = std::path::PathBuf::from(std::env::args().nth(1).expect("dir"));
    let mut p = argmind_env_windows::programs::Programs::found(dir).expect("none");
    match p.call("blender_python", &serde_json::json!({"script": "box.py"})) {
        Ok(t) => println!("OK ({} chars)\n{t}", t.chars().count()),
        Err(e) => println!("ERR ({} chars)\n{e}", e.chars().count()),
    }
}
#[cfg(not(windows))]
fn main() {}
