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
    //: `--gated`: a gate that refuses every click and lets writes through, as a 하지 않음 rule on
    //: clicking would — the order number goes in, 조회 is refused, and the answer never appears.
    if std::env::args().any(|a| a == "--gated") {
        let asked = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
        let log = asked.clone();
        let mut s = argmind_env_windows::screen::Screen::new().gated(Box::new(move |window, kind, what| {
            log.lock().unwrap().push(format!("{window} | {kind} | {what}"));
            if kind == "click" { Err("누르지 않도록 정해 두셨습니다".into()) } else { Ok(()) }
        }));
        s.call("screen_open", &json!({"program": exe.display().to_string()})).unwrap();
        let look = s.call("screen_look", &json!({})).unwrap();
        let num = |look: &str, has: &str| look.lines().find(|l| l.contains(has))
            .and_then(|l| l.split(':').next()?.trim().parse::<u64>().ok());
        println!("  type: {:?}", s.call("screen_type", &json!({"item": num(&look, ": edit ").or_else(|| num(&look, "편집")), "text": "A-1042"})));
        let look = s.call("screen_look", &json!({})).unwrap();
        println!("  press: {:?}", s.call("screen_press", &json!({"item": num(&look, "\"조회\"")})));
        std::thread::sleep(std::time::Duration::from_millis(800));
        let look = s.call("screen_look", &json!({})).unwrap();
        println!("  answer appeared: {}", look.contains("6038-2271-9954"));
        println!("  the gate was asked: {:#?}", asked.lock().unwrap());
        let _ = s.call("screen_close", &json!({}));
        return;
    }
    //: `--program <name> [<label>…]`: open any program, then for each label press the first thing
    //: the last result lists under that name — and print each result whole with its length, which
    //: is what a task's record keeps (`task::OBSERVED_CHARS`) and cuts from the front.
    let args: Vec<String> = std::env::args().collect();
    if let Some(at) = args.iter().position(|a| a == "--program") {
        //: And whether the person's clipboard moved — its sequence number, never its contents.
        //: Character Map's *Select* empties it unless the work screen holds it shut (2026-11-09).
        let clip = || unsafe { windows::Win32::System::DataExchange::GetClipboardSequenceNumber() };
        let clip_before = clip();
        let mut s = argmind_env_windows::screen::Screen::new();
        let show = |tool: &str, a: serde_json::Value, s: &mut argmind_env_windows::screen::Screen| {
            let r = s.call(tool, &a);
            let t = match &r { Ok(t) => t.clone(), Err(e) => format!("✗ {e}") };
            println!("── {tool} {a} → {} chars\n{t}", t.chars().count());
            t
        };
        let mut last = show("screen_open", json!({"program": args[at + 1]}), &mut s);
        if !last.contains('\n') {
            last = show("screen_look", json!({}), &mut s);
        }
        for label in &args[at + 2..] {
            let n = last.lines().find(|l| l.contains(&format!("\"{label}\"")))
                .and_then(|l| l.split(':').next()?.trim().parse::<u64>().ok());
            last = show("screen_press", json!({"item": n}), &mut s);
            if !last.contains('\n') {
                last = show("screen_look", json!({}), &mut s);
            }
        }
        let _ = s.call("screen_close", &json!({}));
        println!("── the person's clipboard unchanged: {}", clip() == clip_before);
        return;
    }
    //: `--leave`: open something and end abruptly, as a killed argmind does — then check from
    //: outside that nothing it opened is still running.
    if std::env::args().any(|a| a == "--leave") {
        println!("  {:?}", s.call("screen_open", &json!({"program": exe.display().to_string()})));
        std::process::exit(0);
    }
    let mut go = |tool: &str, args: serde_json::Value| {
        let began = std::time::Instant::now();
        let r = s.call(tool, &args);
        println!("  {tool} {args} ({} ms)\n    {}", began.elapsed().as_millis(),
                 match &r { Ok(t) => t.replace('\n', "\n    "), Err(e) => format!("✗ {e}") });
        r.unwrap_or_default()
    };
    //: **No `screen_look` between steps** — since 2026-11-09 opening, writing and pressing each end
    //: with the new list, and its numbers are the ones the next step uses.
    let opened = go("screen_open", json!({"program": exe.display().to_string()}));
    let num = |look: &str, role_has: &str| look.lines().find(|l| l.contains(role_has))
        .and_then(|l| l.split(':').next()?.trim().parse::<u64>().ok());
    let edit = num(&opened, ": edit ").or_else(|| num(&opened, "편집"));
    let typed = go("screen_type", json!({"item": edit, "text": "A-1042"}));
    let pressed = go("screen_press", json!({"item": num(&typed, "\"조회\"")}));
    println!("\n  answer read from the press itself: {}", pressed.contains("6038-2271-9954"));
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
