//! **Does a message addressed to a window do what a key or a click does — without the person's
//! pointer or keyboard?** The owner, 2026-11-16: *건드려도 멈추지 않았으면 함*. Locking the person
//! out was weighed and set aside (it needs an administrator or a global keyboard hook); what is left
//! is argmind not using the person's input at all. Three of its acts still do — a key, a click no
//! pattern takes, and typing no value pattern takes — and each goes to whatever the pointer or the
//! focus is on *at that moment*, which is what a person's touch changes.
//!
//! This asks whether `PostMessageW` to the control's own window can stand in for each, in two kinds
//! of program: a classic Win32 window (`order-window`, ours) and Chromium (Edge, with a profile of
//! its own in the scratchpad, so it never joins the person's Edge). **Everything runs on a desktop
//! of its own** — nothing here can reach the person's screen, pointer or keyboard, and the person's
//! pointer and foreground window are read before and after as the control for that.
//!
//! Edge is opened first and `order-window` after it, so **Edge is not the active window when it is
//! written to** — the condition a person clicking somewhere else mid-sequence makes.
//!
//! **Read, 2026-11-16, one run each after the harness's own faults were out**: the order window took
//! `A-10422`, a backspace (`A-1042`) and a click on 조회, which drew the waybill; Edge took a click
//! into the page's box, `안녕 A-1`, Enter (`ENTER:안녕 A-1` in its title) and a click on its button
//! (`CLICK:안녕 A-1`); the person's pointer and foreground unchanged every run. **Three faults were
//! the harness's**: `GetWindowText` cannot read another process's edit (it read `""` from a box
//! holding `A-1042`, which the button's answer proved); the `file:` address came up *Untitled*;
//! and UI Automation's first edit was **the address bar**, so the first good run typed there and
//! Enter searched Bing for `안녕 A-1` — which is why the page's elements are now found under its
//! document, and why Edge is given no network.
//!
//! **And the shipped hands** (`with_messages`, `never_send_input`, reach only Edge), which choose
//! the window themselves: under the box was `Chrome_RenderWidgetHostHWND`, and they took a click
//! into it, ` 둘` added to what was there, Enter and the button — **4 posted, 0 `SendInput`**, Edge
//! still not active. Their first run read `Elsewhere` three times: the order window lay over the
//! box, and the guard refused to press whatever covered the point, which is the rule unchanged —
//! a message goes where a click would have, and no further.
//!
//!     cargo build --release -p argmind-env-windows --example order-window --example posted
//!     target\release\examples\posted.exe <scratch dir>

#[cfg(windows)]
fn main() {
    use std::time::{Duration, Instant};
    use windows::core::{PCWSTR, PWSTR};
    use windows::Win32::Foundation::{CloseHandle, BOOL, HWND, LPARAM, POINT, WPARAM};
    use windows::Win32::System::JobObjects::AssignProcessToJobObject;
    use windows::Win32::System::StationsAndDesktops::{CloseDesktop, CreateDesktopW, EnumDesktopWindows,
                                                       SetThreadDesktop, DESKTOP_CONTROL_FLAGS};
    use windows::Win32::System::Threading::{CreateProcessW, ResumeThread, CREATE_SUSPENDED,
                                             PROCESS_INFORMATION, STARTUPINFOW};
    use windows::Win32::UI::Input::KeyboardAndMouse::{VK_BACK, VK_RETURN};
    use windows::Win32::UI::WindowsAndMessaging::*;

    let scratch = std::env::args().nth(1).expect("a scratch directory");
    let here = std::env::current_exe().unwrap();
    let order = here.parent().unwrap().join("order-window.exe");
    assert!(order.exists(), "build the order-window example first");

    fn wide(s: &str) -> Vec<u16> { s.encode_utf16().chain([0]).collect() }
    fn title(h: HWND) -> String {
        let mut b = [0u16; 512];
        let n = unsafe { GetWindowTextW(h, &mut b) }.max(0) as usize;
        String::from_utf16_lossy(&b[..n])
    }
    //: `GetWindowText` reads another process's caption and never its edit's text, so the edit is
    //: asked for its text by message — the first run read "" from a box holding `A-1042`.
    fn text(h: HWND) -> String {
        let mut b = [0u16; 512];
        let mut got = 0usize;
        unsafe {
            let _ = SendMessageTimeoutW(h, WM_GETTEXT, WPARAM(b.len()), LPARAM(b.as_mut_ptr() as isize),
                                        SMTO_ABORTIFHUNG, 2000, Some(&mut got));
        }
        String::from_utf16_lossy(&b[..got.min(b.len())])
    }
    fn person() -> (POINT, isize) {
        let mut p = POINT::default();
        unsafe { let _ = GetCursorPos(&mut p); }
        (p, unsafe { GetForegroundWindow() }.0 as isize)
    }
    fn post(h: HWND, m: u32, w: usize, l: isize) -> bool {
        unsafe { PostMessageW(h, m, WPARAM(w), LPARAM(l)) }.is_ok()
    }
    fn chars(h: HWND, text: &str) -> bool {
        text.encode_utf16().all(|u| post(h, WM_CHAR, u as usize, 1))
    }
    fn key(h: HWND, vk: u16) -> bool {
        post(h, WM_KEYDOWN, vk as usize, 1) && post(h, WM_KEYUP, vk as usize, (1 | (0xC000_0000u32 as isize)) as isize)
    }
    fn click(h: HWND, x: i32, y: i32) -> bool {
        let at = ((y as isize & 0xffff) << 16) | (x as isize & 0xffff);
        post(h, WM_MOUSEMOVE, 0, at) && post(h, WM_LBUTTONDOWN, 1, at) && post(h, WM_LBUTTONUP, 0, at)
    }
    fn settle() { std::thread::sleep(Duration::from_millis(600)); }

    let before = person();
    let html = std::path::Path::new(&scratch).join("posted-probe.html");
    std::fs::write(&html, r#"<!doctype html><meta charset="utf-8"><title>probe-ready</title>
<body style="margin:0">
<button id="b" style="position:fixed;left:0;top:0;width:220px;height:110px"
 onclick="document.title='CLICK:'+document.getElementById('a').value">조회</button>
<input id="a" autofocus style="position:fixed;left:0;top:140px;width:300px"
 onkeydown="if(event.key==='Enter'){document.title='ENTER:'+this.value}">
<script>document.getElementById('a').focus()</script>"#).unwrap();

    let report = std::thread::spawn(move || -> Vec<String> {
        let mut said = Vec::new();
        let name = format!("argmind-posted-{}", std::process::id());
        let n = wide(&name);
        let desk = unsafe { CreateDesktopW(PCWSTR(n.as_ptr()), None, None, DESKTOP_CONTROL_FLAGS(0), 0x1000_0000, None) }
            .expect("a desktop of its own");
        unsafe { SetThreadDesktop(desk) }.expect("on it");
        let mut jobs = Vec::new();
        let mut start = |cmd: String, plain: bool| {
            let mut where_ = wide(&format!("WinSta0\\{name}"));
            let mut c = wide(&cmd);
            let si = STARTUPINFOW { cb: std::mem::size_of::<STARTUPINFOW>() as u32,
                                    lpDesktop: PWSTR(where_.as_mut_ptr()), ..Default::default() };
            let mut pi = PROCESS_INFORMATION::default();
            unsafe { CreateProcessW(None, PWSTR(c.as_mut_ptr()), None, None, false, CREATE_SUSPENDED,
                                    None, None, &si, &mut pi) }.expect("started");
            //: Edge in a plain kill-on-close job: its own sandbox nests jobs under this one, and
            //: whether the UI limits break its renderer is not what this asks.
            let job = if plain { argmind_env_windows::supervise::kill_on_close() }
                      else { argmind_env_windows::supervise::kept_apart() }.expect("a job");
            unsafe {
                AssignProcessToJobObject(job, pi.hProcess).expect("in the job");
                ResumeThread(pi.hThread);
                let _ = CloseHandle(pi.hThread);
                let _ = CloseHandle(pi.hProcess);
            }
            jobs.push(job);
            pi.dwProcessId
        };
        let windows = |desk| -> Vec<HWND> {
            unsafe extern "system" fn each(h: HWND, l: LPARAM) -> BOOL {
                let v = &mut *(l.0 as *mut Vec<HWND>);
                if IsWindowVisible(h).as_bool() { v.push(h); }
                BOOL(1)
            }
            let mut v: Vec<HWND> = Vec::new();
            unsafe { let _ = EnumDesktopWindows(desk, Some(each), LPARAM(&mut v as *mut _ as isize)); }
            v
        };
        let wait = |want: &dyn Fn(&str) -> bool, secs: u64| -> Option<HWND> {
            let until = Instant::now() + Duration::from_secs(secs);
            while Instant::now() < until {
                if let Some(h) = windows(desk).into_iter().find(|&h| want(&title(h))) {
                    return Some(h);
                }
                std::thread::sleep(Duration::from_millis(200));
            }
            None
        };

        //: Edge first, so the order window opened after it is the active one.
        let edge = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe";
        //: A fresh profile a run — one killed last time asks to restore its pages — and the page as
        //: a `data:` address, since the `file:` one came up *Untitled* twice.
        let profile = std::path::Path::new(&scratch).join(format!("posted-edge-{}", std::process::id()));
        let page = std::fs::read_to_string(&html).unwrap();
        let encoded: String = page.bytes().map(|b| if b.is_ascii_alphanumeric() { (b as char).to_string() }
                                               else { format!("%{b:02X}") }).collect();
        let edge_pid = start(format!("\"{edge}\" --user-data-dir=\"{}\" --no-first-run --no-default-browser-check \
                       --disable-sync --disable-features=CalculateNativeWinOcclusion \
                       --host-resolver-rules=\"MAP * ~NOTFOUND\" --new-window \
                       data:text/html;charset=utf-8,{encoded}", profile.display()), true);
        let browser = wait(&|t| t.contains("probe-ready"), 40);
        let loaded = browser.is_some();
        start(format!("\"{}\"", order.display()), false);
        let win32 = wait(&|t| t == "주문 조회", 10);

        // ---- Win32: the order window ----
        match win32 {
            None => said.push("win32: order window never appeared".into()),
            Some(top) => {
                settle();
                let edit = unsafe { FindWindowExW(top, None, PCWSTR(wide("Edit").as_ptr()), PCWSTR::null()) }
                    .unwrap_or_default();
                let button = unsafe { FindWindowExW(top, None, PCWSTR(wide("Button").as_ptr()), PCWSTR::null()) }
                    .unwrap_or_default();
                let result = || {
                    let mut s = Vec::new();
                    let mut c = unsafe { FindWindowExW(top, None, PCWSTR(wide("Static").as_ptr()), PCWSTR::null()) }
                        .unwrap_or_default();
                    while !c.is_invalid() {
                        s.push(title(c));
                        c = unsafe { FindWindowExW(top, c, PCWSTR(wide("Static").as_ptr()), PCWSTR::null()) }
                            .unwrap_or_default();
                    }
                    s.join(" | ")
                };
                let active = unsafe { GetForegroundWindow() } == top;
                said.push(format!("win32: edit {:?} button {:?}, order window is this desktop's foreground: {active}",
                                  edit.0, button.0));
                let typed = chars(edit, "A-10422");
                settle();
                said.push(format!("win32 chars posted {typed}: edit holds {:?}", text(edit)));
                let keyed = key(edit, VK_BACK.0);
                settle();
                said.push(format!("win32 backspace key posted {keyed}: edit holds {:?}", text(edit)));
                let clicked = click(button, 10, 10);
                settle();
                said.push(format!("win32 click posted {clicked}: statics read {:?}", result()));
            }
        }

        // ---- Chromium: Edge, not the active window ----
        //: Read through UI Automation, which is what argmind reads with anyway: the page's
        //: document is named by its title and the box's value pattern holds what it holds.
        use windows::Win32::UI::Accessibility::*;
        use windows::core::Interface;
        let ua = argmind_env_windows::open_automation().expect("ui automation");
        let under = |root: &IUIAutomationElement| -> Vec<IUIAutomationElement> {
            let mut v = Vec::new();
            unsafe {
                if let Ok(t) = ua.CreateTrueCondition() {
                    if let Ok(found) = root.FindAll(TreeScope_Descendants, &t) {
                        for i in 0..found.Length().unwrap_or(0) {
                            if let Ok(e) = found.GetElement(i) { v.push(e); }
                        }
                    }
                }
            }
            v
        };
        //: **The page's own elements only.** The first run's first edit was the address bar, and
        //: the posted typing and Enter went there and searched Bing.
        let all = |top: HWND| -> Vec<IUIAutomationElement> {
            let Ok(root) = (unsafe { ua.ElementFromHandle(top) }) else { return Vec::new() };
            let doc = under(&root).into_iter()
                .find(|e| unsafe { e.CurrentControlType() }.unwrap_or_default() == UIA_DocumentControlTypeId);
            let mut v: Vec<IUIAutomationElement> = doc.iter().cloned().collect();
            if let Some(d) = &doc {
                v.extend(under(d));
            }
            v
        };
        let kind = |e: &IUIAutomationElement| unsafe { e.CurrentControlType() }.unwrap_or_default();
        let name = |e: &IUIAutomationElement| unsafe { e.CurrentName() }.map(|n| n.to_string()).unwrap_or_default();
        let state = |top: HWND| -> String {
            let es = all(top);
            let doc = es.iter().find(|e| kind(e) == UIA_DocumentControlTypeId).map(name).unwrap_or_default();
            let value = es.iter().find(|e| kind(e) == UIA_EditControlTypeId && name(e).is_empty()
                                         || kind(e) == UIA_EditControlTypeId)
                .and_then(|e| unsafe { e.GetCurrentPattern(UIA_ValuePatternId) }.ok())
                .and_then(|r| r.cast::<IUIAutomationValuePattern>().ok())
                .and_then(|v| unsafe { v.CurrentValue() }.ok()).map(|v| v.to_string());
            format!("document {doc:?}, box {value:?}, window {:?}", title(top))
        };
        match browser {
            None => said.push(format!("edge: no window; windows here: {:?}",
                                      windows(desk).iter().map(|&h| title(h)).collect::<Vec<_>>())),
            Some(top) => {
                settle();
                //: Chromium builds its tree on the first question; give it a few seconds to.
                let until = Instant::now() + Duration::from_secs(8);
                while Instant::now() < until && !all(top).iter().any(|e| kind(e) == UIA_EditControlTypeId) {
                    std::thread::sleep(Duration::from_millis(300));
                }
                let active = unsafe { GetForegroundWindow() } == top;
                said.push(format!("edge: loaded {loaded}, foreground here {active}; {}", state(top)));
                let button = all(top).into_iter().find(|e| kind(e) == UIA_ButtonControlTypeId && name(e) == "조회");
                let input = all(top).into_iter().find(|e| kind(e) == UIA_EditControlTypeId);
                let client_centre = |e: &IUIAutomationElement| -> Option<(i32, i32)> {
                    let r = unsafe { e.CurrentBoundingRectangle() }.ok()?;
                    let mut p = POINT { x: (r.left + r.right) / 2, y: (r.top + r.bottom) / 2 };
                    unsafe { let _ = windows::Win32::Graphics::Gdi::ScreenToClient(top, &mut p); }
                    Some((p.x, p.y))
                };
                said.push(format!("edge: box at client {:?}, button at client {:?}",
                                  input.as_ref().and_then(client_centre), button.as_ref().and_then(client_centre)));
                if let Some((x, y)) = input.as_ref().and_then(client_centre) {
                    let c = click(top, x, y);
                    settle();
                    said.push(format!("edge click into box posted {c}: {}", state(top)));
                }
                let typed = chars(top, "안녕 A-1");
                settle();
                said.push(format!("edge chars posted {typed}: {}", state(top)));
                let keyed = key(top, VK_RETURN.0);
                settle();
                said.push(format!("edge enter posted {keyed}: {}", state(top)));
                if let Some((x, y)) = button.as_ref().and_then(client_centre) {
                    let c = click(top, x, y);
                    settle();
                    said.push(format!("edge click on button posted {c}: {}", state(top)));
                }
                //: **And the shipped hands**, which choose the window themselves — whatever
                //: `WindowFromPoint` gives under the point, which in Chromium may be a child it
                //: draws through rather than the window the raw posts above went to.
                let rect_of = |e: &IUIAutomationElement| -> (i64, i64, i64, i64) {
                    let r = unsafe { e.CurrentBoundingRectangle() }.unwrap_or_default();
                    (r.left as i64, r.top as i64, (r.right - r.left) as i64, (r.bottom - r.top) as i64)
                };
                //: **The order window lies over Edge's box** — the first run with the hands read
                //: `Elsewhere` three times, the guard refusing to press whatever covers the point.
                //: Moved off the screen without being activated, so Edge is still not active.
                if let Some(o) = win32 {
                    unsafe { let _ = SetWindowPos(o, None, -3000, -3000, 0, 0,
                                                  SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE); }
                    settle();
                }
                said.push(format!("hands: Edge active here now: {}", unsafe { GetForegroundWindow() } == top));
                if let (Some(input), Some(button)) = (&input, &button) {
                    let iw = argmind_core::contract::Widget::new("", "edit", rect_of(input), "");
                    let bw = argmind_core::contract::Widget::new("조회", "button", rect_of(button), "");
                    let (cx, cy) = iw.centre();
                    let under = unsafe { WindowFromPoint(POINT { x: cx as i32, y: cy as i32 }) };
                    let class = {
                        let mut b = [0u16; 128];
                        let n = unsafe { GetClassNameW(under, &mut b) }.max(0) as usize;
                        String::from_utf16_lossy(&b[..n])
                    };
                    said.push(format!("hands: under the box is {class:?}{}", if under == top { " (the top window)" } else { "" }));
                    let mut hands = argmind_env_windows::WindowsHands::only_within(edge_pid)
                        .with_messages().never_send_input().arm();
                    let d = hands.click_button(&iw, "left");
                    settle();
                    said.push(format!("hands click into box: {d:?}"));
                    let d = hands.type_string(" 둘");
                    settle();
                    said.push(format!("hands type: {d:?}: {}", state(top)));
                    let d = hands.press_key("enter");
                    settle();
                    said.push(format!("hands enter: {d:?}: {}", state(top)));
                    let d = hands.click_button(&bw, "left");
                    settle();
                    said.push(format!("hands click on button: {d:?}: {}", state(top)));
                    said.push(format!("hands tallies: posted {}, SendInput {}", hands.by_message, hands.by_click));
                }
            }
        }
        for j in jobs {
            unsafe { let _ = CloseHandle(j); }
        }
        std::thread::sleep(Duration::from_millis(500));
        unsafe { let _ = CloseDesktop(desk); }
        said
    }).join().unwrap();

    for line in report {
        println!("{line}");
    }
    let after = person();
    println!("person: pointer {:?} -> {:?}, foreground {} -> {} — {}",
             (before.0.x, before.0.y), (after.0.x, after.0.y), before.1, after.1,
             if before.0.x == after.0.x && before.0.y == after.0.y && before.1 == after.1 { "unchanged" }
             else { "CHANGED (the person may have moved it themselves)" });
}

#[cfg(not(windows))]
fn main() {}
