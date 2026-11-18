//! **The whole of *the picture only where the list is blind*, on a real window** — Character Map,
//! on a desktop of its own, so nothing here can reach the person's screen, pointer or keyboard.
//!
//! `--bin looking --arms gated` measured the path on drawn screens. What a drawing cannot say is
//! whether the pieces work on Windows: the list UI Automation gives for the grid, a picture
//! `PrintWindow` draws of a real window, the model's box on that picture, the gate over the real
//! list, and a click posted to a point inside a control no pattern can press into. This runs each
//! in the application's own functions (`argmind_serve::looking`, `argmind_env_windows::picture`,
//! `Session::find`, `Inside::within`, `WindowsHands::click_point` — what `perform` does with a point,
//! written out, since a harness may not call `perform`), and reads the answer off the window
//! itself: Character Map's status bar names the character that is selected.
//!
//! **Ten characters on the first page of its first font, and ★, which is not on it.** A fresh
//! Character Map for every request: a posted press can leave Character Map's enlarged character
//! drawn over the grid, and the first runs here measured that — the next request pointed at the
//! last one's enlargement (*ß* → *"AE symbol"*). Three pictures of the same window:
//!
//! * **as is** — the window, never enlarged;
//! * **window 1280** — the window made 1280 wide;
//! * **place 1280** — the blind place alone, cut out and made 1280 wide.
//!
//! The person's pointer and foreground window are read before and after, as the control that
//! nothing reached them; the hands may touch only that Character Map and never use `SendInput`.
//!
//!     cargo run --release -p argmind-measure --example pointing
//!     POINTING_MODEL=qwen3-vl:8b-instruct POINTING_ONLY="place 1280" POINTING_SAVE=<dir> …
//!
//! `POINTING_SAVE` writes the picture each request was shown, the row's marked picture made as
//! `app::look_inside` makes it, and the place after the press — so the ring can be held against the
//! cell Character Map actually selected.
//!
//! **Read, 2026-11-18, at 150%.** The first run read 0, 0 and 1 of 10 and was this harness: not
//! per-monitor aware, so `GetWindowRect` and UI Automation answered in scaled pixels while
//! `PrintWindow` drew at real size, and the picture held two thirds of the grid. Then, with one
//! Character Map reused between requests, the 4b read as is 5, window 1280 3, place 1280 7 and 7,
//! and the 8b place 1280 7 — the 8b's misses all on the previous request's enlargement, which is
//! why each request now has a window of its own. **A fresh window each request**: the 4b as is 5,
//! window 1280 6 (★ declined), place 1280 **8 and 8, ★ pressed both times**; **the 8b on place 1280
//! 8 and 8, ★ declined both times** — the arm the application ships. Every run: nothing by the
//! person's pointer, their pointer and foreground window unchanged.

#[cfg(windows)]
fn main() {
    use std::time::{Duration, Instant};
    use argmind_core::contract::{Eyes, Hands, Widget};
    use argmind_core::session::{Found, Session};
    use argmind_env_ollama::Local;
    use argmind_env_windows::hands::WindowsHands;
    use argmind_serve::looking::{asking, blind, effect, landed, on_screen, read, Landed, Saw, MOST_WIDE, VISION_MODEL};
    use windows::core::{PCWSTR, PWSTR};
    use windows::Win32::Foundation::{CloseHandle, BOOL, HANDLE, HWND, LPARAM, POINT};
    use windows::Win32::System::JobObjects::AssignProcessToJobObject;
    use windows::Win32::System::StationsAndDesktops::{CreateDesktopW, EnumDesktopWindows, SetThreadDesktop,
                                                       DESKTOP_CONTROL_FLAGS, HDESK};
    use windows::Win32::System::Threading::{CreateProcessW, ResumeThread, CREATE_SUSPENDED,
                                             PROCESS_INFORMATION, STARTUPINFOW};
    use windows::Win32::UI::WindowsAndMessaging::*;

    fn wide(s: &str) -> Vec<u16> { s.encode_utf16().chain([0]).collect() }
    fn title(h: HWND) -> String {
        let mut b = [0u16; 512];
        let n = unsafe { GetWindowTextW(h, &mut b) }.max(0) as usize;
        String::from_utf16_lossy(&b[..n])
    }
    fn person() -> (POINT, isize) {
        let mut p = POINT::default();
        unsafe { let _ = GetCursorPos(&mut p); }
        (p, unsafe { GetForegroundWindow() }.0 as isize)
    }
    fn windows_on(desk: HDESK) -> Vec<HWND> {
        unsafe extern "system" fn each(h: HWND, l: LPARAM) -> BOOL {
            let v = &mut *(l.0 as *mut Vec<HWND>);
            if IsWindowVisible(h).as_bool() { v.push(h); }
            BOOL(1)
        }
        let mut v: Vec<HWND> = Vec::new();
        unsafe { let _ = EnumDesktopWindows(desk, Some(each), LPARAM(&mut v as *mut _ as isize)); }
        v
    }

    /// Eyes that read the one window this run opened, which is what `perform` re-resolves against.
    struct One(argmind_env_windows::WindowsEyes, isize);
    impl Eyes for One {
        fn foreground(&mut self) -> Vec<Widget> { self.0.read(self.1) }
        fn timed_foreground(&mut self) -> (Vec<Widget>, f64) { (self.0.read(self.1), 0.0) }
    }
    struct Mute;
    impl argmind_core::contract::Speaks for Mute {
        fn ask(&mut self, _: &str) -> String { String::new() }
    }

    //: **Per-monitor aware, as the application is** (its window crate sets this at start).
    unsafe {
        let _ = windows::Win32::UI::HiDpi::SetProcessDpiAwarenessContext(
            windows::Win32::UI::HiDpi::DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
    }
    let before = person();
    let report = std::thread::spawn(move || -> Vec<String> {
        let mut said = Vec::new();
        let name = format!("argmind-pointing-{}", std::process::id());
        let n = wide(&name);
        let desk = unsafe { CreateDesktopW(PCWSTR(n.as_ptr()), None, None, DESKTOP_CONTROL_FLAGS(0), 0x1000_0000, None) }
            .expect("a desktop of its own");
        unsafe { SetThreadDesktop(desk) }.expect("on it");
        //: A Character Map of its own, in a job that ends it when the handle is closed.
        let open = || -> Option<(HANDLE, HWND, u32)> {
            let mut where_ = wide(&format!("WinSta0\\{name}"));
            let mut cmd = wide(r"C:\Windows\System32\charmap.exe");
            let si = STARTUPINFOW { cb: std::mem::size_of::<STARTUPINFOW>() as u32,
                                    lpDesktop: PWSTR(where_.as_mut_ptr()), ..Default::default() };
            let mut pi = PROCESS_INFORMATION::default();
            unsafe { CreateProcessW(None, PWSTR(cmd.as_mut_ptr()), None, None, false, CREATE_SUSPENDED,
                                    None, None, &si, &mut pi) }.ok()?;
            let job = argmind_env_windows::supervise::kept_apart().ok()?;
            unsafe {
                AssignProcessToJobObject(job, pi.hProcess).ok()?;
                ResumeThread(pi.hThread);
                let _ = CloseHandle(pi.hThread);
                let _ = CloseHandle(pi.hProcess);
            }
            let until = Instant::now() + Duration::from_secs(15);
            while Instant::now() < until {
                let mine = windows_on(desk).into_iter().find(|&h| {
                    let mut pid = 0u32;
                    unsafe { GetWindowThreadProcessId(h, Some(&mut pid)); }
                    let t = title(h);
                    pid == pi.dwProcessId && (t == "문자표" || t == "Character Map")
                });
                if let Some(h) = mine {
                    std::thread::sleep(Duration::from_millis(700));
                    return Some((job, h, pi.dwProcessId));
                }
                std::thread::sleep(Duration::from_millis(150));
            }
            unsafe { let _ = CloseHandle(job); }
            None
        };

        let model = std::env::var("POINTING_MODEL").unwrap_or_else(|_| VISION_MODEL.to_string());
        let save = std::env::var("POINTING_SAVE").ok();
        let only = std::env::var("POINTING_ONLY").ok();
        said.push(format!("model {model}"));
        let mut looker = Local::new(&model);
        looker.think = Some(false);
        looker.max_tokens = 256;
        looker.keep_alive = "300s".into();
        let _ = looker.ask("Reply with the single word: ready");

        let wanted: [(&str, Option<&str>); 11] = [
            ("©", Some("U+00A9")), ("@", Some("U+0040")), ("%", Some("U+0025")), ("Æ", Some("U+00C6")),
            ("ß", Some("U+00DF")), ("¼", Some("U+00BC")), ("¶", Some("U+00B6")), ("÷", Some("U+00F7")),
            ("ñ", Some("U+00F1")), ("Ø", Some("U+00D8")), ("★", None),
        ];
        let (mut posted, mut by_pointer) = (0, 0);
        for variant in ["as is", "window 1280", "place 1280"] {
            if only.as_deref().is_some_and(|o| o != variant) { continue; }
            let (mut right, mut declined_star, mut secs_all) = (0, false, Vec::new());
            for (c, want) in wanted {
                let Some((job, top, pid)) = open() else { said.push("Character Map never appeared".into()); continue };
                let handle = top.0 as isize;
                let eyes = argmind_env_windows::WindowsEyes::new().expect("UI Automation");
                let hands = WindowsHands::only_within(pid).with_messages().never_send_input().arm();
                let mut session = Session::new(One(eyes, handle), hands, Mute);
                session.may_act_unprompted = true;
                session.seen = session.eyes.foreground();
                let places: Vec<Widget> = session.seen.iter().filter(|w| blind(w, &session.seen)).cloned().collect();
                if said.len() == 1 {
                    said.push(format!("listed {} · blind places {:?}", session.seen.len(),
                                      places.iter().map(|w| format!("{} ({}) {:?}", w.name, w.role, w.rect)).collect::<Vec<_>>()));
                }
                let grid = places.iter().find(|w| w.role == "pane").map(|w| w.rect);
                let asked = format!("{c} 문자 눌러줘");
                let d = argmind_env_windows::picture::drawn(handle).expect("a picture");
                let shot = match variant {
                    "as is" => d.shot(None, MOST_WIDE.min(d.wide)),
                    "window 1280" => d.shot(None, MOST_WIDE),
                    _ => d.shot(grid, MOST_WIDE),
                }.expect("a shot");
                if let Some(dir) = &save {
                    let name: String = variant.chars().filter(|c| c.is_ascii_alphanumeric()).collect();
                    let _ = std::fs::write(format!("{dir}/{name}-{:04x}.png", c.chars().next().unwrap() as u32), &shot.png);
                }
                let began = Instant::now();
                let reply = looker.look_at(&asking(&asked), &shot.png).expect("the vision model");
                let secs = began.elapsed().as_secs_f64();
                secs_all.push(secs);
                let line = match read(&reply.text) {
                    Saw::At { x, y, label } => {
                        let (px, py) = on_screen(x, y, shot.rect);
                        let seen = session.seen.clone();
                        match landed(&seen, px, py) {
                            Landed::Inside(container) => {
                                let e = effect(container, px, py, &label);
                                //: **The row's picture, made as `app::look_inside` makes it** — the
                                //: point where the press will land, ringed on the picture the model
                                //: was shown — and, after the press, the place as it then looks, so
                                //: the ring can be held against the cell that was actually chosen.
                                let tag = format!("{:04x}", c.chars().next().unwrap() as u32);
                                if let Some(dir) = &save {
                                    let picture = argmind_serve::looking::Picture {
                                        png: Vec::new(), rect: shot.rect, bgra: shot.bgra.clone(),
                                        wide: shot.wide, tall: shot.tall };
                                    let at = e.inside.map(|i| i.within(container.rect)).and_then(|(x, y)| picture.at(x, y));
                                    if let Some(at) = at {
                                        let png = argmind_env_windows::picture::marked(&picture.bgra, picture.wide,
                                            picture.tall, at, argmind_serve::looking::SHOWN_WIDE);
                                        let _ = std::fs::write(format!("{dir}/marked-{tag}.png"), png);
                                    }
                                }
                                //: **What `perform` does with it, written out**, because a harness may
                                //: not call `perform` (`argmind-rules`, `the_voice_cannot_act`): the
                                //: container found again by name and role, the point placed in its
                                //: fresh rectangle, the hands asked for that point.
                                let pressed = match session.find(&e.target, &e.role) {
                                    Found::One(w) => {
                                        let at = e.inside.expect("a point").within(w.rect);
                                        session.hands.click_inside(&w, at)
                                    }
                                    _ => false,
                                };
                                std::thread::sleep(Duration::from_millis(400));
                                if let Some(dir) = &save {
                                    if let Some(after) = argmind_env_windows::picture::drawn(handle)
                                        .and_then(|d| d.shot(Some(shot.rect), argmind_serve::looking::SHOWN_WIDE)) {
                                        let _ = std::fs::write(format!("{dir}/after-{tag}.png"), after.png);
                                    }
                                }
                                let status: Vec<String> = session.eyes.foreground().into_iter()
                                    .filter(|w| w.name.starts_with("U+")).map(|w| w.name).collect();
                                let hit = match want {
                                    Some(code) if status.iter().any(|s| s.starts_with(code)) => { right += 1; "RIGHT" }
                                    Some(_) => "wrong",
                                    None => "pressed something — not on this page",
                                };
                                format!("{c}: {secs:.2}s · {label:?} inside {} · pressed {pressed} · {status:?} → {hit}",
                                        container.name)
                            }
                            other => {
                                if want.is_none() { declined_star = true; }
                                format!("{c}: {secs:.2}s · {label:?} dropped: {other:?}")
                            }
                        }
                    }
                    Saw::Nothing => {
                        if want.is_none() { declined_star = true; }
                        format!("{c}: {secs:.2}s · nothing")
                    }
                    other => format!("{c}: {secs:.2}s · {other:?} · {:?}", reply.text),
                };
                posted += session.hands.by_message;
                by_pointer += session.hands.by_click;
                said.push(format!("[{variant}] {line}"));
                unsafe { let _ = CloseHandle(job); }
                std::thread::sleep(Duration::from_millis(200));
            }
            secs_all.sort_by(|a, b| a.partial_cmp(b).unwrap());
            said.push(format!("[{variant}] RIGHT {right} of 10 · ★ declined {declined_star} · median {:.2}s · slowest {:.2}s",
                              secs_all[secs_all.len() / 2], secs_all.last().unwrap()));
        }
        said.push(format!("posted {posted} · by the person's pointer {by_pointer}"));
        said
    }).join().expect("the run");
    let after = person();
    for line in &report {
        println!("  {line}");
    }
    println!("  the person's pointer {:?} → {:?}, foreground {} → {}: {}",
             (before.0.x, before.0.y), (after.0.x, after.0.y), before.1, after.1,
             if before == after { "unchanged" } else { "CHANGED" });
}

#[cfg(not(windows))]
fn main() {}
