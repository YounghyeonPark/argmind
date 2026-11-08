//! **The work screen's tools, driven by hand** — no model: open the order window on argmind's own
//! desktop, look, write the order number, press 조회, look again, close; and whether the person's
//! foreground window and physical pointer stayed where they were.
//!
//!     cargo build --release -p argmind-env-windows --example order-window
//!     cargo run --release -p argmind-env-windows --example screen-drive

#[cfg(windows)]
fn main() {
    use argmind_core::task::Tools;
    use serde_json::json;
    use windows::Win32::Foundation::POINT;
    use windows::Win32::UI::WindowsAndMessaging::{GetForegroundWindow, GetPhysicalCursorPos};

    let pointer = || { let mut p = POINT::default(); unsafe { let _ = GetPhysicalCursorPos(&mut p); } (p.x, p.y) };
    let (p0, f0) = (pointer(), unsafe { GetForegroundWindow() });
    let exe = std::env::current_exe().unwrap().parent().unwrap().join("order-window.exe");
    let view = argmind_env_windows::screen::View::default();
    let mut s = argmind_env_windows::screen::Screen::shown(view.clone());
    let shown = |when: &str| match view.lock().unwrap().as_ref() {
        Some(p) => {
            let at = std::env::temp_dir().join(format!("argmind-work-screen-{when}.bmp"));
            let _ = std::fs::write(&at, &p.bmp);
            println!("  picture {when}: #{} {:?} {} bytes -> {}", p.seq, p.titles, p.bmp.len(), at.display());
        }
        None => println!("  picture {when}: none"),
    };
    let mut go = |tool: &str, args: serde_json::Value| {
        let began = std::time::Instant::now();
        let r = s.call(tool, &args);
        println!("  {tool} {args} ({} ms)\n    {}", began.elapsed().as_millis(),
                 match &r { Ok(t) => t.replace('\n', "\n    "), Err(e) => format!("✗ {e}") });
        r.unwrap_or_default()
    };
    go("screen_open", json!({"program": exe.display().to_string()}));
    let look = go("screen_look", json!({}));
    let num = |look: &str, role_has: &str| look.lines().find(|l| l.contains(role_has))
        .and_then(|l| l.split(':').next()?.trim().parse::<u64>().ok());
    let edit = num(&look, ": edit ").or_else(|| num(&look, "편집"));
    go("screen_type", json!({"item": edit, "text": "A-1042"}));
    let look = go("screen_look", json!({}));
    go("screen_press", json!({"item": num(&look, "\"조회\"")}));
    let look = go("screen_look", json!({}));
    println!("\n  answer read: {}", look.contains("6038-2271-9954"));
    shown("after");
    let seq = view.lock().unwrap().as_ref().map(|p| p.seq);
    std::thread::sleep(std::time::Duration::from_millis(2500));
    println!("  idle 2.5 s, picture unchanged: {}", view.lock().unwrap().as_ref().map(|p| p.seq) == seq);
    go("screen_close", json!({}));
    shown("closed");
    println!("  foreground unchanged: {}", unsafe { GetForegroundWindow() } == f0);
    println!("  pointer (physical) {p0:?} -> {:?}", pointer());
}

#[cfg(not(windows))]
fn main() {}
