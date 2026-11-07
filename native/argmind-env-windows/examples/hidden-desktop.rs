//! **Can argmind work on a desktop nobody is looking at?** — a probe, 2026-11-08.
//!
//!     cargo run -p argmind-env-windows --example hidden-desktop
//!
//! The owner: *사용자의 컴퓨터 사용과 충돌하지 않게 별도의 샌드박스 창에서 작업을 수행하는것 고려.*
//! Windows Home has no Windows Sandbox, and the Linux screen has no office program with a licence
//! the project takes (LibreOffice brings GPL `libpoppler`). So: a second Windows desktop, made with
//! `CreateDesktopW`, the person's own programs started on it, read and operated through UI
//! Automation's control patterns — **never input**, since `SendInput` reaches only the desktop the
//! person is on, and that is the one this must not touch.
//!
//! Starts Character Map on a desktop of its own and nothing else; operates only that. Prints what
//! the native client reads, whether `Toggle`, `SetValue` and `Invoke` work there, whether the
//! window can be captured, and whether the foreground and the pointer were left alone. The
//! pointer is read with `GetPhysicalCursorPos`, so the process's DPI awareness changing on the way
//! — which the PowerShell probe read as a moved pointer, 1.5x, this screen's scale — cannot.

#[cfg(windows)]
fn main() {
    use windows::core::{Interface, PWSTR};
    use windows::Win32::Foundation::{BOOL, HWND, LPARAM, POINT, RECT};
    use windows::Win32::Graphics::Gdi::{CreateCompatibleBitmap, CreateCompatibleDC, DeleteDC,
                                        DeleteObject, GetDC, GetPixel, ReleaseDC, SelectObject};
    use windows::Win32::Storage::Xps::{PrintWindow, PRINT_WINDOW_FLAGS};
    use windows::Win32::System::Com::{CoCreateInstance, CoInitializeEx, CLSCTX_INPROC_SERVER,
                                      COINIT_MULTITHREADED};
    use windows::Win32::System::StationsAndDesktops::{CloseDesktop, CreateDesktopW, EnumDesktopWindows,
                                                      SetThreadDesktop, DESKTOP_CONTROL_FLAGS, HDESK};
    use windows::Win32::System::Threading::{CreateProcessW, TerminateProcess, PROCESS_CREATION_FLAGS,
                                            PROCESS_INFORMATION, STARTUPINFOW};
    use windows::Win32::UI::Accessibility::*;
    use windows::Win32::UI::WindowsAndMessaging::{GetForegroundWindow, GetPhysicalCursorPos,
                                                  GetWindowRect, GetWindowThreadProcessId, IsWindowVisible};

    fn wide(s: &str) -> Vec<u16> { s.encode_utf16().chain([0]).collect() }
    fn pointer() -> (i32, i32) {
        let mut p = POINT::default();
        unsafe { let _ = GetPhysicalCursorPos(&mut p); }
        (p.x, p.y)
    }

    let (pointer_before, front_before) = (pointer(), unsafe { GetForegroundWindow() });
    let name = wide("argmind-work-probe");
    let desk = unsafe { CreateDesktopW(windows::core::PCWSTR(name.as_ptr()), None, None,
                                       DESKTOP_CONTROL_FLAGS(0), 0x1000_0000, None) }
        .expect("CreateDesktopW");
    println!("  desktop made");
    let mut where_ = wide("WinSta0\\argmind-work-probe");
    //: `PROBE_WORD=1` starts Word instead, with a blank document and no splash — only if Word is
    //: not already running, since a second Word joins the first, on whichever desktop it is.
    let word = std::env::var("PROBE_WORD").is_ok();
    let program = if word { "\"C:\\Program Files\\Microsoft Office\\root\\Office16\\WINWORD.EXE\" /n /q" }
                  else { "C:\\Windows\\System32\\charmap.exe" };
    let mut cmd = wide(program);
    let si = STARTUPINFOW { cb: std::mem::size_of::<STARTUPINFOW>() as u32,
                            lpDesktop: PWSTR(where_.as_mut_ptr()), ..Default::default() };
    let mut pi = PROCESS_INFORMATION::default();
    unsafe { CreateProcessW(None, PWSTR(cmd.as_mut_ptr()), None, None, false, PROCESS_CREATION_FLAGS(0),
                            None, None, &si, &mut pi) }.expect("CreateProcessW");
    println!("  charmap pid {}", pi.dwProcessId);

    let (pid, desk_raw) = (pi.dwProcessId, desk.0 as isize);
    let worked = std::thread::spawn(move || -> Result<(), String> {
        let desk = HDESK(desk_raw as *mut _);
        unsafe { SetThreadDesktop(desk) }.map_err(|e| format!("SetThreadDesktop {e}"))?;
        unsafe { let _ = CoInitializeEx(None, COINIT_MULTITHREADED); }
        struct Find { pid: u32, hwnd: HWND }
        unsafe extern "system" fn each(h: HWND, l: LPARAM) -> BOOL {
            let f = &mut *(l.0 as *mut Find);
            let mut p = 0u32;
            GetWindowThreadProcessId(h, Some(&mut p));
            if p == f.pid && IsWindowVisible(h).as_bool() { f.hwnd = h; return BOOL(0); }
            BOOL(1)
        }
        let mut find = Find { pid, hwnd: HWND::default() };
        for _ in 0..if word { 300 } else { 50 } {
            std::thread::sleep(std::time::Duration::from_millis(100));
            unsafe { let _ = EnumDesktopWindows(desk, Some(each), LPARAM(&mut find as *mut Find as isize)); }
            if !find.hwnd.is_invalid() { break; }
        }
        if find.hwnd.is_invalid() { return Err("no window".into()); }
        let ua: IUIAutomation = unsafe { CoCreateInstance(&CUIAutomation, None, CLSCTX_INPROC_SERVER) }
            .map_err(|e| e.to_string())?;
        let began = std::time::Instant::now();
        let root = unsafe { ua.ElementFromHandle(find.hwnd) }.map_err(|e| e.to_string())?;
        let all = unsafe { root.FindAll(TreeScope_Descendants, &ua.CreateTrueCondition().map_err(|e| e.to_string())?) }
            .map_err(|e| e.to_string())?;
        let n = unsafe { all.Length() }.unwrap_or(0);
        println!("  read: {:?}, {n} elements, {} ms",
                 unsafe { root.CurrentName() }.map(|s| s.to_string()).unwrap_or_default(), began.elapsed().as_millis());
        let mut elements = Vec::new();
        for i in 0..n {
            if let Ok(e) = unsafe { all.GetElement(i) } { elements.push(e); }
        }
        let kind = |e: &IUIAutomationElement| unsafe { e.CurrentControlType() }.unwrap_or_default();
        let named = |e: &IUIAutomationElement| unsafe { e.CurrentName() }.map(|s| s.to_string()).unwrap_or_default();

        if word {
            let mut kinds = std::collections::BTreeMap::new();
            for e in &elements { *kinds.entry(kind(e).0).or_insert(0) += 1; }
            println!("  control types: {kinds:?}");
            let names: Vec<String> = elements.iter().map(named).filter(|n| !n.is_empty()).take(25).collect();
            println!("  names: {names:?}");
            //: Close it the way a person would, so Word keeps no recovery file for somebody's next start.
            let closed = unsafe { root.GetCurrentPattern(UIA_WindowPatternId) }
                .and_then(|r| r.cast::<IUIAutomationWindowPattern>()).and_then(|w| unsafe { w.Close() });
            println!("  close: {closed:?}");
            std::thread::sleep(std::time::Duration::from_secs(2));
            return Ok(());
        }
        //: Toggle, read back.
        if let Some(b) = elements.iter().find(|e| kind(e) == UIA_CheckBoxControlTypeId) {
            let t = unsafe { b.GetCurrentPattern(UIA_TogglePatternId) }.and_then(|r| r.cast::<IUIAutomationTogglePattern>());
            match t {
                Ok(t) => {
                    let before = unsafe { t.CurrentToggleState() }.map(|s| s.0).unwrap_or(-1);
                    let did = unsafe { t.Toggle() };
                    std::thread::sleep(std::time::Duration::from_millis(500));
                    let after = unsafe { t.CurrentToggleState() }.map(|s| s.0).unwrap_or(-1);
                    println!("  toggle {:?}: {before} -> {after} ({did:?})", named(b));
                }
                Err(e) => println!("  toggle: no pattern {e}"),
            }
        }
        //: SetValue on the copy box, then Invoke on 'Select' — which appends the chosen character
        //: to that box, so the box says whether the button did anything.
        let edit = elements.iter().find(|e| kind(e) == UIA_EditControlTypeId).cloned();
        let value = |e: &IUIAutomationElement| unsafe { e.GetCurrentPattern(UIA_ValuePatternId) }
            .and_then(|r| r.cast::<IUIAutomationValuePattern>());
        if let Some(edit) = &edit {
            if let Ok(v) = value(edit) {
                let did = unsafe { v.SetValue(&windows::core::BSTR::from("숨은화면")) };
                println!("  set value: {did:?} -> {:?}", unsafe { v.CurrentValue() }.map(|s| s.to_string()));
            }
        }
        let buttons: Vec<String> = elements.iter().filter(|e| kind(e) == UIA_ButtonControlTypeId).map(named).collect();
        println!("  buttons: {buttons:?}");
        if let Some(b) = elements.iter().find(|e| kind(e) == UIA_ButtonControlTypeId
                                              && ["Select", "선택"].iter().any(|w| named(e).starts_with(w))) {
            let did = unsafe { b.GetCurrentPattern(UIA_InvokePatternId) }
                .and_then(|r| r.cast::<IUIAutomationInvokePattern>()).and_then(|p| unsafe { p.Invoke() });
            std::thread::sleep(std::time::Duration::from_millis(500));
            let after = edit.as_ref().and_then(|e| value(e).ok()).and_then(|v| unsafe { v.CurrentValue() }.ok())
                .map(|s| s.to_string());
            println!("  invoke {:?}: {did:?} -> box now {after:?}", named(b));
        }
        //: A picture of it, for somebody to watch.
        unsafe {
            let mut r = RECT::default();
            let _ = GetWindowRect(find.hwnd, &mut r);
            let (w, h) = (r.right - r.left, r.bottom - r.top);
            let screen = GetDC(None);
            let dc = CreateCompatibleDC(screen);
            let bmp = CreateCompatibleBitmap(screen, w, h);
            let old = SelectObject(dc, bmp);
            let ok = PrintWindow(find.hwnd, dc, PRINT_WINDOW_FLAGS(2)).as_bool();
            let (mut lit, mut total) = (0, 0);
            for y in (0..h).step_by(4) { for x in (0..w).step_by(4) {
                total += 1;
                let c = GetPixel(dc, x, y).0;
                if (c & 0xff) + ((c >> 8) & 0xff) + ((c >> 16) & 0xff) > 30 { lit += 1; }
            } }
            println!("  capture {ok}: {w}x{h}, {}% not black", 100 * lit / total.max(1));
            SelectObject(dc, old);
            let _ = DeleteObject(bmp);
            let _ = DeleteDC(dc);
            ReleaseDC(None, screen);
        }
        Ok(())
    }).join();
    println!("  thread: {worked:?}");
    unsafe {
        let _ = TerminateProcess(pi.hProcess, 0);
        std::thread::sleep(std::time::Duration::from_millis(300));
        let _ = CloseDesktop(desk);
    }
    println!("  foreground unchanged: {}", unsafe { GetForegroundWindow() } == front_before);
    println!("  pointer (physical) {pointer_before:?} -> {:?}", pointer());
}

#[cfg(not(windows))]
fn main() {}
