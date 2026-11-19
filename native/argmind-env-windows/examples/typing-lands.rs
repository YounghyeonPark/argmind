//! **Does the check before a `type` refuse the programs a person actually types into?** Since
//! 2026-11-19 the application's hands forward `typing_lands`, which reads the element under the
//! point just clicked (or the focus) and refuses unless it takes text. Before that day the check
//! never ran in the application, so what it refuses there had never been seen.
//!
//! The worry, reasoned from the code and not observed: the element under a click is not always
//! the element that takes the keys. In VS Code it is a rendered line of code, and the box that
//! takes keys is a hidden one beside the caret; a web editor's body may be a plain `div`.
//!
//! The hands are built the way `argmind-serve`'s `platform::hands` builds them: patterns, typing by
//! value, messages. Two differences: reach is only the program this started, and **`SendInput` is
//! never used**. Every program runs **on a desktop of its own**, started by this, in a job that
//! ends it, so nothing here reaches the person's screen, pointer, keyboard or open programs.
//! The person's pointer and foreground window are read before and after as the control for that.
//!
//! For each target: the hands click it, `typing_lands` is asked as `perform` asks it, and **the
//! text is typed anyway**, so a refusal can be scored as right (the text would not have arrived)
//! or wrong (it would have). Where the text went is read back through UI Automation.
//!
//!     cargo build --release -p argmind-env-windows --example typing-lands
//!     target\release\examples\typing-lands.exe <scratch dir> [--list]
//!
//! `--list` only opens each program and prints what the eyes list in it, pressing nothing.
//! `TYPING_ONLY=edge,chrome` picks programs; `TYPING_VERBOSE=1` keeps VS Code's own output.
//!
//! **Read, 2026-11-19, one run each, criteria written before each.** On a web page in Edge and in
//! Chrome — a search box, a textarea, a `contenteditable` with role `textbox`, a plain
//! `contenteditable`, a group that cannot be edited, a button, the address bar:
//! - **Before the fix, 3 wrong refusals.** The plain `contenteditable` in both browsers (a Group
//!   with no value pattern; typed anyway, the text arrived), and Edge's address bar, whose click
//!   took **2,020–2,075 ms** while the aim lasts 1 s from when it was set, so the check read the
//!   focus instead. The non-editable group and the button were refused, both browsers.
//! - **After, 0.** `takes_text` accepts the text-edit pattern, which the plain `contenteditable`
//!   offers and the group and the button do not; the hands keep what was under the point before
//!   the click; and an aim's second runs from when its click is done. Everything allowed before
//!   was still allowed, and the group and the button still refused.
//! - **Apart, older than the check**: Chrome's address bar, allowed, took nothing — it already
//!   holds the address, so the value is not written (`N characters already there`), and the
//!   characters posted to an inactive Chrome are dropped where Edge's are not. Word read
//!   *Unlicensed Product* and took nothing either way: void. And a button's Invoke took about 2 s
//!   in both browsers.
//! - **Then Chrome's address bar, the same day.** After argmind's click the bar holds its address
//!   **all selected**, as after a person's, so typing would replace it — and `set_value` now writes
//!   into a box that holds something when, and only when, all of it is selected. It arrived, and
//!   the Enter after it, at `perform`'s pace, moved the page; the rest read as before. **Read with
//!   the old rule on a real-looking `https://www.` address the failure did not reproduce**: the
//!   point read an unnamed element, the characters were posted and arrived, and Enter moved the
//!   page. So what failed was the `data:` page, and how often a person meets it is not known; the
//!   rule is kept because writing a fully selected box is what typing does. **The Enter had its
//!   own expiry**: the window it goes to was stamped when the click began, and a written value did
//!   not stamp it at all — both now stamp when the act is done.
//! - **Not run**: VS Code, which will not start while the person's own VS Code holds its update
//!   (`checkInnoSetupMutex: vscode-updating is held`); Notepad, a packaged launcher; KakaoTalk,
//!   one instance, the person's. **The harness's own faults first**: reading the whole tree between
//!   the click and the check, which expired the aim; and `ELECTRON_RUN_AS_NODE` inherited from the
//!   VS Code this was run from, which made `Code.exe` a bare node that exited.
//! - On a desktop of its own, where the focus is the desktop's pane: what the check reads when the
//!   aim has expired is not what it would read on the person's desktop.

#[cfg(windows)]
fn main() {
    use argmind_core::contract::{Hands, Widget};
    use std::time::{Duration, Instant};
    use windows::core::{Interface, PCWSTR, PWSTR};
    use windows::Win32::Foundation::{CloseHandle, BOOL, HANDLE, HWND, LPARAM, POINT};
    use windows::Win32::System::JobObjects::AssignProcessToJobObject;
    use windows::Win32::System::StationsAndDesktops::{CloseDesktop, CreateDesktopW, EnumDesktopWindows,
                                                       SetThreadDesktop, DESKTOP_CONTROL_FLAGS};
    use windows::Win32::System::Threading::{CreateProcessW, ResumeThread, CREATE_SUSPENDED,
                                             PROCESS_INFORMATION, STARTUPINFOW};
    use windows::Win32::UI::Accessibility::*;
    use windows::Win32::UI::WindowsAndMessaging::*;

    argmind_env_windows::per_monitor_pixels();
    //: **Run from inside VS Code, this inherits `ELECTRON_RUN_AS_NODE`**, which makes `Code.exe` a
    //: bare node that exits — the first run found no window — and `VSCODE_IPC_HOOK`, which could
    //: hand the new window to the person's own VS Code. Neither reaches anything this starts.
    for (key, _) in std::env::vars_os() {
        let k = key.to_string_lossy().to_uppercase();
        if k.starts_with("ELECTRON_") || k.starts_with("VSCODE_") {
            unsafe { std::env::remove_var(&key) };
        }
    }
    let scratch = std::path::PathBuf::from(std::env::args().nth(1).expect("a scratch directory"));
    let list_only = std::env::args().any(|a| a == "--list");
    std::fs::create_dir_all(&scratch).unwrap();

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
    fn settle(ms: u64) { std::thread::sleep(Duration::from_millis(ms)); }

    //: **The web page**: the boxes a person types into on the web, each labelled as a real page
    //: labels them, and a button as the control that must be refused. Its title reports every
    //: box's contents on every input, so arrival is read off the window as well as the tree.
    let page = r#"<!doctype html><meta charset="utf-8"><title>probe-ready</title>
<body style="margin:20px;font:16px sans-serif">
<input id="q" aria-label="검색어" placeholder="검색어" style="display:block;width:400px;height:32px;margin:10px 0">
<textarea id="m" aria-label="메모" style="display:block;width:400px;height:80px;margin:10px 0"></textarea>
<div id="c" contenteditable="true" role="textbox" aria-label="본문" style="border:1px solid #888;width:400px;height:80px;margin:10px 0"></div>
<div id="d" contenteditable="true" aria-label="댓글" style="border:1px solid #888;width:400px;height:80px;margin:10px 0"></div>
<div id="n" role="group" aria-label="안내" style="border:1px dashed #888;width:400px;height:60px;margin:10px 0">이 칸은 읽기만 합니다</div>
<button id="b" style="width:120px;height:40px">보내기</button>
<script>
const show=()=>{document.title='q=['+q.value+'] m=['+m.value+'] c=['+c.innerText.trim()+'] d=['+d.innerText.trim()+']'};
for (const e of [q,m,c,d]) e.addEventListener('input',show);
</script>"#;
    let encoded: String = page.bytes().map(|b| if b.is_ascii_alphanumeric() { (b as char).to_string() }
                                           else { format!("%{b:02X}") }).collect();
    let work = scratch.join("vscode-work");
    std::fs::create_dir_all(&work).unwrap();
    let note = work.join("note.txt");
    std::fs::write(&note, "첫 줄\n둘째 줄\n").unwrap();

    struct App { name: &'static str, cmd: String, ready: &'static str }
    let edge = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe";
    let chrome = r"C:\Program Files\Google\Chrome\Application\chrome.exe";
    let code = format!(r"{}\Programs\Microsoft VS Code\Code.exe", std::env::var("LOCALAPPDATA").unwrap());
    let word = r"C:\Program Files\Microsoft Office\root\Office16\WINWORD.EXE";
    let browser = |exe: &str, tag: &str| format!(
        "\"{exe}\" --user-data-dir=\"{}\" --no-first-run --no-default-browser-check --disable-sync \
         --disable-features=CalculateNativeWinOcclusion --host-resolver-rules=\"MAP * ~NOTFOUND\" \
         --new-window data:text/html;charset=utf-8,{encoded}",
        scratch.join(format!("{tag}-profile-{}", std::process::id())).display());
    let through_cmd = |cmd: String| if std::env::var("TYPING_VERBOSE").is_ok() {
        format!("cmd /c \"{cmd} > \"{}\" 2>&1\"", scratch.join("vscode-out.txt").display())
    } else { cmd };
    let apps = vec![
        App { name: "edge", cmd: browser(edge, "edge"), ready: "probe-ready" },
        App { name: "chrome", cmd: browser(chrome, "chrome"), ready: "probe-ready" },
        //: **An address shown shorter than it is**: Chrome hides `https://www.` until the bar is
        //: clicked, the case a real site makes. The network is off, so the page is Chrome's own.
        App { name: "chrome-site", ready: "example.com",
              cmd: format!("\"{chrome}\" --user-data-dir=\"{}\" --no-first-run --no-default-browser-check \
                            --disable-sync --disable-features=CalculateNativeWinOcclusion \
                            --host-resolver-rules=\"MAP * ~NOTFOUND\" --new-window https://www.example.com/page",
                           scratch.join(format!("chrome-site-profile-{}", std::process::id())).display()) },
        App { name: "vscode", ready: "note.txt",
              cmd: through_cmd(format!("\"{code}\" --user-data-dir \"{}\" --extensions-dir \"{}\" --disable-extensions \
                            --skip-release-notes --skip-welcome --disable-workspace-trust --new-window \"{}\" \"{}\"",
                           scratch.join(format!("vscode-user-{}", std::process::id())).display(),
                           scratch.join(format!("vscode-ext-{}", std::process::id())).display(),
                           work.display(), note.display())
                       + &if std::env::var("TYPING_VERBOSE").is_ok() { " --verbose --log trace".to_string() } else { String::new() }) },
        //: `TYPING_VERBOSE`: VS Code's own account of its start, into the scratch folder.
        //: `/w`: a new instance with a blank document, so it does not join a Word somebody has open.
        App { name: "word", cmd: format!("\"{word}\" /q /w"), ready: "" },
    ];

    let before = person();
    let report = std::thread::spawn(move || -> Vec<String> {
        let mut said = Vec::new();
        let name = format!("argmind-typing-{}", std::process::id());
        let n = wide(&name);
        let desk = unsafe { CreateDesktopW(PCWSTR(n.as_ptr()), None, None, DESKTOP_CONTROL_FLAGS(0), 0x1000_0000, None) }
            .expect("a desktop of its own");
        unsafe { SetThreadDesktop(desk) }.expect("on it");
        let start = |cmd: &str| -> (u32, HANDLE) {
            let mut where_ = wide(&format!("WinSta0\\{name}"));
            let mut c = wide(cmd);
            let si = STARTUPINFOW { cb: std::mem::size_of::<STARTUPINFOW>() as u32,
                                    lpDesktop: PWSTR(where_.as_mut_ptr()), ..Default::default() };
            let mut pi = PROCESS_INFORMATION::default();
            unsafe { CreateProcessW(None, PWSTR(c.as_mut_ptr()), None, None, false, CREATE_SUSPENDED,
                                    None, None, &si, &mut pi) }.expect("started");
            let job = argmind_env_windows::supervise::kill_on_close().expect("a job");
            unsafe {
                AssignProcessToJobObject(job, pi.hProcess).expect("in the job");
                ResumeThread(pi.hThread);
                let _ = CloseHandle(pi.hThread);
                let _ = CloseHandle(pi.hProcess);
            }
            (pi.dwProcessId, job)
        };
        //: **Every window on this desktop is one this started**, so a window is found by its title
        //: and its owner read off it: VS Code's window is not the process that was started.
        let windows_of = |_pid: u32| -> Vec<(HWND, u32)> {
            unsafe extern "system" fn each(h: HWND, l: LPARAM) -> BOOL {
                let v = &mut *(l.0 as *mut Vec<HWND>);
                if IsWindowVisible(h).as_bool() { v.push(h); }
                BOOL(1)
            }
            let mut v: Vec<HWND> = Vec::new();
            unsafe { let _ = EnumDesktopWindows(desk, Some(each), LPARAM(&mut v as *mut _ as isize)); }
            v.into_iter().filter(|&h| !title(h).is_empty()).map(|h| {
                let mut owner = 0u32;
                unsafe { GetWindowThreadProcessId(h, Some(&mut owner)); }
                (h, owner)
            }).collect()
        };
        let ua = argmind_env_windows::open_automation().expect("ui automation");
        let kind_name = |t: UIA_CONTROLTYPE_ID| -> &'static str {
            match t {
                x if x == UIA_EditControlTypeId => "Edit",
                x if x == UIA_DocumentControlTypeId => "Document",
                x if x == UIA_TextControlTypeId => "Text",
                x if x == UIA_GroupControlTypeId => "Group",
                x if x == UIA_PaneControlTypeId => "Pane",
                x if x == UIA_ButtonControlTypeId => "Button",
                x if x == UIA_ComboBoxControlTypeId => "ComboBox",
                x if x == UIA_CustomControlTypeId => "Custom",
                _ => "other",
            }
        };
        //: What an element is, as the check sees it: its type, class, name, and its value pattern's
        //: read-only flag — `None` when it has no value pattern, which is when the type decides.
        let describe = |e: &IUIAutomationElement| -> String {
            unsafe {
                let ro = e.GetCurrentPattern(UIA_ValuePatternId).ok()
                    .and_then(|raw| raw.cast::<IUIAutomationValuePattern>().ok())
                    .and_then(|v| v.CurrentIsReadOnly().ok()).map(|r| r.as_bool());
                let name: String = e.CurrentName().map(|n| n.to_string()).unwrap_or_default().chars().take(40).collect();
                format!("{} class {:?} name {name:?} value-read-only {ro:?} takes-text {}",
                        kind_name(e.CurrentControlType().unwrap_or_default()),
                        e.CurrentClassName().map(|c| c.to_string()).unwrap_or_default(),
                        argmind_env_windows::patterns::takes_text(e))
            }
        };
        //: **What else it offers** — what a wider check could read: the text and text-edit patterns,
        //: focusability, and MSAA's role and read-only state.
        let offers = |e: &IUIAutomationElement| -> String {
            unsafe {
                let has = |id: UIA_PATTERN_ID| e.GetCurrentPattern(id).is_ok_and(|p| !p.as_raw().is_null());
                let legacy = e.GetCurrentPattern(UIA_LegacyIAccessiblePatternId).ok()
                    .and_then(|raw| raw.cast::<IUIAutomationLegacyIAccessiblePattern>().ok());
                let (role, state) = legacy.map(|l| (l.CurrentRole().unwrap_or(0), l.CurrentState().unwrap_or(0)))
                    .unwrap_or((0, 0));
                format!("text {} text-edit {} focusable {} aria-role {:?} msaa-role {role} msaa-state {state:#x} (read-only {})",
                        has(UIA_TextPatternId), has(UIA_TextEditPatternId),
                        e.CurrentIsKeyboardFocusable().map(|b| b.as_bool()).unwrap_or(false),
                        e.CurrentAriaRole().map(|r| r.to_string()).unwrap_or_default(),
                        state & 0x40 != 0)
            }
        };
        //: **What is selected in it**, through the text pattern — whether typing would replace it.
        let selection = |e: &IUIAutomationElement| -> String {
            unsafe {
                let Some(p) = e.GetCurrentPattern(UIA_TextPatternId).ok()
                    .and_then(|raw| raw.cast::<IUIAutomationTextPattern>().ok()) else { return "no text pattern".into() };
                let Ok(ranges) = p.GetSelection() else { return "no selection".into() };
                let n = ranges.Length().unwrap_or(0);
                let texts: Vec<String> = (0..n).filter_map(|i| ranges.GetElement(i).ok())
                    .map(|r| r.GetText(200).map(|t| t.to_string()).unwrap_or_default().chars().take(40).collect())
                    .collect();
                format!("{n} ranges {texts:?}")
            }
        };
        //: **Where a piece of text is**, anywhere in the window: a value, a name, a document's text.
        let find_text = |top: HWND, needle: &str| -> Vec<String> {
            let mut at = Vec::new();
            if title(top).contains(needle) {
                at.push("the window's title".to_string());
            }
            unsafe {
                let Ok(root) = ua.ElementFromHandle(top) else { return at };
                let Ok(t) = ua.CreateTrueCondition() else { return at };
                let Ok(all) = root.FindAll(TreeScope_Subtree, &t) else { return at };
                for i in 0..all.Length().unwrap_or(0) {
                    let Ok(e) = all.GetElement(i) else { continue };
                    let kind = kind_name(e.CurrentControlType().unwrap_or_default());
                    let value = e.GetCurrentPattern(UIA_ValuePatternId).ok()
                        .and_then(|raw| raw.cast::<IUIAutomationValuePattern>().ok())
                        .and_then(|v| v.CurrentValue().ok()).map(|v| v.to_string()).unwrap_or_default();
                    let text = if kind == "Document" || kind == "Edit" {
                        e.GetCurrentPattern(UIA_TextPatternId).ok()
                            .and_then(|raw| raw.cast::<IUIAutomationTextPattern>().ok())
                            .and_then(|p| p.DocumentRange().ok())
                            .and_then(|r| r.GetText(4000).ok()).map(|s| s.to_string()).unwrap_or_default()
                    } else { String::new() };
                    let name = e.CurrentName().map(|n| n.to_string()).unwrap_or_default();
                    for (how, s) in [("value", &value), ("text", &text), ("name", &name)] {
                        if s.contains(needle) {
                            let label: String = name.chars().take(30).collect();
                            at.push(format!("{how} of {kind} {label:?}"));
                            break;
                        }
                    }
                }
            }
            at.sort();
            at.dedup();
            at
        };
        let mut eyes = argmind_env_windows::WindowsEyes::new().expect("eyes");
        let mut arm = 0;

        let only = std::env::var("TYPING_ONLY").ok();
        for app in apps.iter().filter(|a| only.as_deref().map_or(true, |o| o.split(',').any(|x| x == a.name))) {
            let (started, job) = start(&app.cmd);
            let until = Instant::now() + Duration::from_secs(40);
            let mut top = None;
            while Instant::now() < until {
                if let Some(found) = windows_of(started).into_iter().find(|&(h, _)| title(h).contains(app.ready)) {
                    top = Some(found);
                    break;
                }
                settle(300);
            }
            let Some((top, pid)) = top else {
                //: Every window on the desktop, shown or not, to say what did come up.
                unsafe extern "system" fn every(h: HWND, l: LPARAM) -> BOOL {
                    let v = &mut *(l.0 as *mut Vec<HWND>);
                    v.push(h);
                    BOOL(1)
                }
                let mut all: Vec<HWND> = Vec::new();
                unsafe { let _ = EnumDesktopWindows(desk, Some(every), LPARAM(&mut all as *mut _ as isize)); }
                let seen: Vec<String> = all.iter().map(|&h| {
                    let mut b = [0u16; 128];
                    let n = unsafe { GetClassNameW(h, &mut b) }.max(0) as usize;
                    let mut owner = 0u32;
                    unsafe { GetWindowThreadProcessId(h, Some(&mut owner)); }
                    format!("{:?}/{}/{}/{}", title(h), String::from_utf16_lossy(&b[..n]), owner,
                            unsafe { IsWindowVisible(h) }.as_bool())
                }).collect();
                said.push(format!("{}: no window; every window here: {seen:?}", app.name));
                unsafe { let _ = CloseHandle(job); }
                continue;
            };
            //: A tree is built on the first question and grows for a while; read until it holds still.
            let mut list = Vec::new();
            let until = Instant::now() + Duration::from_secs(15);
            while Instant::now() < until {
                settle(1000);
                let now = eyes.read(top.0 as isize);
                let still = now.len() == list.len() && now.len() > 0;
                list = now;
                if still { break; }
            }
            said.push(format!("{}: window {:?}, {} listed, owner {pid} (started {started})", app.name, title(top), list.len()));
            let typey = |w: &Widget| matches!(w.role.as_str(), "edit" | "document" | "combo box" | "group" | "text");
            let listing: Vec<String> = list.iter().map(|w| format!("  {:<12} {:?} at {:?}", w.role,
                                                                    w.name.chars().take(60).collect::<String>(), w.rect)).collect();
            std::fs::write(scratch.join(format!("list-{}.txt", app.name)), listing.join("\n")).unwrap();
            for w in list.iter().filter(|w| typey(w) && w.role != "text") {
                said.push(format!("  {:<10} {:?} at {:?}", w.role, w.name.chars().take(60).collect::<String>(), w.rect));
            }
            if list_only {
                unsafe { let _ = CloseHandle(job); }
                settle(1000);
                continue;
            }
            //: The targets, by name and role as the model would name them.
            let wants: Vec<(&str, &str, bool)> = match app.name {
                "edge" | "chrome" => vec![("edit", "검색어", true), ("edit", "메모", true), ("edit", "본문", true),
                                          ("group", "댓글", true), ("group", "안내", false), ("button", "보내기", false),
                                          ("edit", "Address and search bar", true)],
                "chrome-site" => vec![("edit", "Address and search bar", true)],
                "vscode" => vec![("edit", "", true)],
                "word" => vec![("document", "", true)],
                _ => vec![],
            };
            for (role, name_part, should) in wants {
                let Some(w) = list.iter().find(|w| w.role == role && w.name.contains(name_part)).cloned() else {
                    said.push(format!("  [{}] no {role} named like {name_part:?} in the list", app.name));
                    continue;
                };
                arm += 1;
                let needle = format!("안녕{arm}h");
                let mut hands = argmind_env_windows::WindowsHands::only_within(pid)
                    .with_patterns().with_typing_patterns().with_messages().never_send_input().arm();
                //: **In `perform`'s order and at its pace**: the click, the check at once, the type.
                //: The first run read the tree between them, which took longer than `AIM_LASTS`, so
                //: the aim had expired and the check read the focus — this desktop's own pane.
                let before_text = find_text(top, &needle);
                let (cx, cy) = w.centre();
                let at_point = || unsafe { ua.ElementFromPoint(POINT { x: cx as i32, y: cy as i32 }) }.ok()
                    .map(|e| describe(&e)).unwrap_or_else(|| "nothing".into());
                let ahead = at_point();
                let clock = Instant::now();
                hands.click(&w);
                let click_took = clock.elapsed().as_millis();
                let click_landed = hands.landed();
                let lands = hands.typing_lands();
                let check_took = clock.elapsed().as_millis() - click_took;
                let checked = at_point();
                let selected = unsafe { ua.ElementFromPoint(POINT { x: cx as i32, y: cy as i32 }) }.ok()
                    .map(|e| selection(&e)).unwrap_or_default();
                hands.type_text(&needle);
                let type_landed = hands.landed();
                let typed = hands.done.last().cloned();
                let typed_why = hands.why.clone();
                //: **The address bar's Enter, in `perform`'s order and at its pace**: click, type,
                //: enter is the documented sequence, a quarter of a second apart.
                let mut entered = String::new();
                if name_part == "Address and search bar" {
                    let was = title(top);
                    let pressed = hands.press_key("enter");
                    settle(2500);
                    entered = format!("\n      enter: {pressed:?}; the window's title {was:?} -> {:?}", title(top));
                }
                let under = unsafe { ua.ElementFromPoint(POINT { x: cx as i32, y: cy as i32 }) }.ok()
                    .map(|e| format!("{}\n        offers: {}", describe(&e), offers(&e))).unwrap_or_else(|| "nothing".into());
                let focus = unsafe { ua.GetFocusedElement() }.ok()
                    .map(|e| describe(&e)).unwrap_or_else(|| "nothing".into());
                settle(700);
                let arrived = find_text(top, &needle);
                let verdict = match (lands, !arrived.is_empty()) {
                    (Some(false), true) => "WRONG REFUSAL — it would have arrived",
                    (Some(false), false) => "refused, and it would not have arrived",
                    (_, true) => "allowed, arrived",
                    (_, false) => "allowed, DID NOT ARRIVE",
                };
                said.push(format!("  [{}] {role} {:?} (a person types here: {should})\n      click {:?} landed {click_landed:?} in {click_took} ms, the check {check_took} ms\n      under the point before the click: {ahead}\n      just after the check: {checked}, selected {selected}\n      after the type: {under}\n      focus, after: {focus}\n      typing_lands {lands:?}; typed anyway {typed:?} landed {type_landed:?}; found before {before_text:?}, after {arrived:?}{entered}\n      → {verdict}",
                                  app.name, w.name.chars().take(40).collect::<String>(),
                                  hands.done.first()));
                said.push(format!("      why, after the type: {typed_why:?}"));
                said.push(format!("      tallies: pattern {}, value {}, posted {}, SendInput {}",
                                  hands.by_pattern, hands.by_value, hands.by_message, hands.by_click));
            }
            unsafe { let _ = CloseHandle(job); }
            settle(1500);
        }
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
