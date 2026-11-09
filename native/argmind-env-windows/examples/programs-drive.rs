//! **The installed programs' tools, driven by hand** — no model: Blender makes a 2 × 1 × 0.5 box
//! and exports it, and kicad-cli exports a demo schematic's BOM, each in a fresh work folder.
//!
//!     cargo run --release -p argmind-env-windows --example programs-drive

#[cfg(windows)]
fn main() {
    use argmind_core::task::Tools;
    use serde_json::json;
    let dir = std::env::temp_dir().join(format!("argmind-programs-drive-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let mut p = argmind_env_windows::programs::Programs::found(dir.clone()).expect("neither Blender nor KiCad");
    for t in p.list() {
        println!("  {}: {}", t.name, t.does.chars().take(60).collect::<String>());
    }
    std::fs::write(dir.join("box.py"), "import bpy\nbpy.ops.mesh.primitive_cube_add(size=1)\no = bpy.context.object\n\
o.scale = (2, 1, 0.5)\nbpy.ops.object.transform_apply(scale=True)\nbpy.ops.wm.stl_export(filepath='box.stl')\n\
bpy.ops.wm.save_as_mainfile(filepath='box.blend')\nprint('dims', tuple(o.dimensions))\n").unwrap();
    let began = std::time::Instant::now();
    let r = p.call("blender_python", &json!({"script": "box.py"}));
    println!("\n  blender_python ({} ms): {r:?}", began.elapsed().as_millis());
    println!("  box.stl {:?} bytes, box.blend {:?} bytes", std::fs::metadata(dir.join("box.stl")).map(|m| m.len()).ok(),
             std::fs::metadata(dir.join("box.blend")).map(|m| m.len()).ok());
    let demo = std::path::Path::new(r"C:\Program Files\KiCad\10.0\share\kicad\demos\ecc83\ecc83-pp.kicad_sch");
    if demo.exists() {
        std::fs::copy(demo, dir.join("ecc83.kicad_sch")).unwrap();
        let began = std::time::Instant::now();
        let r = p.call("kicad_cli", &json!({"args": ["sch", "export", "bom", "-o", "bom.csv", "ecc83.kicad_sch"]}));
        println!("\n  kicad_cli ({} ms): {r:?}", began.elapsed().as_millis());
        println!("  bom.csv: {:?}", std::fs::read_to_string(dir.join("bom.csv")).ok().map(|s| s.lines().take(4).map(String::from).collect::<Vec<_>>()));
        println!("  outside refused: {:?}", p.call("kicad_cli", &json!({"args": ["sch", "export", "bom", "-o", "C:/x.csv", "ecc83.kicad_sch"]})));
    }
    //: The workbench's two commands, which since 2026-11-09 start on argmind's own desktop.
    let mut w = argmind_env_windows::workspace::Workspace::new(dir.clone());
    println!("\n  run: {:?}", w.call("run", &json!({"command": "python3 -c 'print(6*7)'"})));
    println!("  run_windows: {:?}", w.call("run_windows", &json!({"command": "Start-Process winver; Write-Output 끝"})));
    let _ = std::fs::remove_dir_all(&dir);
}

#[cfg(not(windows))]
fn main() {}
